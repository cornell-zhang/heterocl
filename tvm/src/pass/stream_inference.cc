/*!
 *  Copyright (c) 2020 by Contributors
 * \file stream_inference.cc
 * \brief mutate ir for scheduling streaming ops
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <unordered_map>
#include "./ir_util.h"

namespace TVM {
namespace ir {

inline Expr Type2Expr(const Type& t) {
  if (t.code()  ==Type::Handle) 
    return StringImm::make("handle");
  std::ostringstream os;
  os << t;
  return StringImm::make(os.str());
}

// group and merge the device scope stmts
Stmt MergeStmts(std::vector<Stmt>& stack) {
  // break the sequence into subgroups 
  std::vector<std::vector<Stmt>> stmt_blocks;
  std::vector<Stmt> stmt_block;
  Stmt body = stack.back(); stack.pop_back();
  // pop out attr and irrelavent stmts 
  while (!body.as<For>() && !body.as<KernelDef>()) {
    body = stack.back(); stack.pop_back();
  }
  for (auto ri = stack.rbegin(); ri != stack.rend(); ++ri) {
    Stmt s = *ri;
    if (s.as<AttrStmt>()) {
      auto n = std::make_shared<AttrStmt>(*s.as<AttrStmt>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<Allocate>()) {
      auto n = std::make_shared<Allocate>(*s.as<Allocate>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else { // stack body stmt  
      s = Block::make(s, body);
      body = s;
    }
  }
  // merge into stmt blocks 
  return body;
};

class StreamCollector final : public IRVisitor {
 public:
  void Visit_(const Allocate* op) {
    auto v = op->buffer_var.get();
    shape_[v] = op->extents;
    IRVisitor::Visit_(op);
  }
  void Visit_(const StreamExpr* op) {
    channel_buf_ = VarExpr(op->buffer_var.node_);
    var_shape_ = shape_[op->buffer_var.get()]; 
    IRVisitor::Visit_(op);
  }
  std::unordered_map<const Variable*, Array<Expr>> shape_;
  VarExpr channel_buf_;
  Array<Expr> var_shape_;
};

// collect type and shape of assigned vars
class TypeCollector : public IRVisitor {
 public:
  explicit TypeCollector(const Array<Var>& vars) {
      for (auto& v : vars) vars_.insert(v.get());
    }
  // collect shape and type
  void Visit_(const Allocate* op) {
    auto v = op->buffer_var.get();
    if (vars_.find(v) != vars_.end()) {
      shape_[v] = op->extents;
      dtype_[v] = op->type;
    }
    IRVisitor::Visit_(op);
  }
  std::unordered_set<const Variable*> vars_;
  std::unordered_map<const Variable*, Array<Expr>> shape_;
  std::unordered_map<const Variable*, Type> dtype_;
};

/*!
 * \brief An IRVisitor to collect information 
 *
 * Collect streaming information:
 *   1. Streaming data access pattern consistency:
 *      the access index of rd & wr must match to 
 *      avoid streaming channel overflow 
 *   2. Streaming related variable analysis:
 *      check the defined and undefined variable
 *      in the xcel scope 
 *   3. Record Stmts
 *
 * and add information into IR nodes
 *
 * */
class StreamAnalysis final : public IRMutator {
 public:
  StreamAnalysis(Array<NodeRef>& api_args) {
    for (size_t i = 0; i < api_args.size(); i++) { 
      if (const Variable* v = api_args[i].as<Variable>()) {
        Expr expr = Expr(api_args[i].node_);
        bind_buffer_map_[v->name_hint] = expr;
        top_arg_names.insert(v->name_hint);

      // replace buffers with tensor expr
      } else if (auto buf = api_args[i].as<BufferNode>()) {
        CHECK(buf->data.as<Variable>());
        bind_buffer_map_[buf->name] = Expr(buf->data.node_);
        top_arg_names.insert(buf->name); 
      }
    }
  };

  // record undefined var in each scope 
  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::device_scope) {
      auto scope = op->value.as<StringImm>()->value;
      if (scope != scope_) { 
        scope_ = scope;
        if (scope != "cpu") { // from host to xcel
          record_ = true;
        } else if (scope == "cpu") {
          record_ = false;
          streaming_vars.push_back(undefined_);
          undefined_ = {};
        }
      }
    }
    // replace the buffer node
    if (auto buf = op->node.as<BufferNode>()) {
      std::string name = buf->name;
      VarExpr buf_var(bind_buffer_map_[name].node_);
      Stmt body = this->Mutate(op->body);
      return AttrStmt::make(buf_var, op->attr_key, op->value, body);
    } else { 
      return IRMutator::Mutate_(op, s); 
    }
  }

  Expr Mutate_(const Variable *op, const Expr& e) final {
    this->HandleUse(e);
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Load *op, const Expr& e) final {
    if (auto buf = op->buffer_var.as<BufferNode>()) {
      std::string name = buf->name;
      VarExpr buf_var(bind_buffer_map_[name].node_);
      this->HandleUse(buf_var);
      return Load::make(op->type, buf_var, op->index, op->predicate);

    } else { // regualr load
      this->HandleUse(op->buffer_var);
      return IRMutator::Mutate_(op, e);
    }
  }

  Expr Mutate_(const StreamExpr *op, const Expr& e) final {
    if (auto buf = op->buffer_var.as<BufferNode>()) {
      std::string name = buf->name;
      VarExpr buf_var(bind_buffer_map_[name].node_);
      this->HandleUse(buf_var);
      return StreamExpr::make(op->type, buf_var, op->stream_type, op->depth,
                              op->annotate_keys, op->annotate_values);
    } else { // regualr load
      this->HandleUse(op->buffer_var);
      return IRMutator::Mutate_(op, e);
    }
  }

  Stmt Mutate_(const LetStmt *op, const Stmt& s) final {
    this->HandleDef(op->var.get());
    Stmt body = this->Mutate(op->body);
    Expr value = this->Mutate(op->value);
    if (body.same_as(op->body) &&
        value.same_as(op->value)) {
      return s;
    } else {
      return LetStmt::make(op->var, value, body);
    }
  }
  
  Stmt Mutate_(const KernelDef *op, const Stmt& s) {
    for (auto arg : op->args) {
      this->HandleDef(arg.get());
    }
    Stmt body = this->Mutate(op->body);
    for (auto arg : op->args) {
      def_count_[arg.get()] = 0;
    }
    return s;
  }

  Stmt Mutate_(const For *op, const Stmt& s) final {
    this->HandleDef(op->loop_var.get());
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt& s) final {
    auto v = op->buffer_var.get();
    auto name = v->name_hint; 
    bind_buffer_map_[name] = Expr(op->buffer_var.node_);
    // tolerate multiple channel buffer decl
    if (name.find("c_buf_") == std::string::npos || !def_count_.count(v)) 
      this->HandleDef(v);
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt& s) final {
    if (auto buf = op->buffer_var.as<BufferNode>()) {
      std::string name = buf->name;
      VarExpr buf_var(bind_buffer_map_[name].node_);
      this->HandleUse(buf_var);
      Expr value = this->Mutate(op->value);
      return Store::make(buf_var, value, op->index, op->predicate);

    } else { // regular store
      this->HandleUse(op->buffer_var);
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const KernelStmt* op, const Stmt& s) final {
    int pos = 0;
    for (auto arg : op->args) {
      this->HandleUse(arg);
      auto name = arg.as<Variable>()->name_hint;
      if (top_arg_names.find(name) != top_arg_names.end())
        kernel_arg_scope_[op->name].insert(pos);
      pos += 1;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const StreamStmt* op, const Stmt& s) final {
    if (auto buf = op->buffer_var.as<BufferNode>()) {
      std::string name = buf->name;
      VarExpr buf_var(bind_buffer_map_[name].node_);
      this->HandleUse(buf_var);
      Expr value = this->Mutate(op->value);
      return StreamStmt::make(buf_var, value, op->stream_type, op->depth,
                              op->annotate_keys, op->annotate_values);

    } else { // regular store
      this->HandleUse(op->buffer_var);
      return IRMutator::Mutate_(op, s);
    }
  }

  void HandleDef(const Variable* v) {
    if (record_) { // record
      CHECK(!def_count_.count(v))
          << "variable " << v->name_hint
          << " has already been defined, the Stmt is not SSA";
      CHECK(!use_count_.count(v))
          << "variable " << v->name_hint
          << " has been used before definition!";
      use_count_[v] = 0;
      def_count_[v] = 1;
    }
  }

  void HandleUse(const Expr& v) {
    if (record_) {
      CHECK(v.as<Variable>());
      Var var(v.node_);
      auto it = use_count_.find(var.get());
      if (it != use_count_.end()) {
        if (it->second >= 0) {
          ++it->second;
        }
      } else {
        undefined_.push_back(var);
        use_count_[var.get()] = -1;
      }
    }
  }

  // map of channel num to access index
  std::unordered_map<int, Expr> index_map;
  // recording flag
  bool record_{false};
  // init device scope 
  std::string scope_{"cpu"};
  // undefined variable 
  Array<Var> undefined_;
  // use count and def count 
  std::unordered_map<const Variable*, int> use_count_;
  std::unordered_map<const Variable*, int> def_count_;
  // vector of undefined vars in xcel scope 
  std::vector<Array<Var>> streaming_vars;
  // replace unbinded buffer with tensor 
  std::unordered_map<std::string, Expr> bind_buffer_map_;
  // top argument name set
  std::unordered_set<std::string> top_arg_names;
  // kernel name to global scope arg position 
  std::unordered_map<std::string, std::unordered_set<int>> kernel_arg_scope_;

};


class StreamMutator : public IRMutator {
 public:
  explicit StreamMutator(int bus_bandwidth) {
    bus_bandwidth_ = bus_bandwidth;
  }

  // remove unnecessary stmt_stacked device scope attr
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (op->attr_key == attr::device_scope) {
      auto scope = op->value.as<StringImm>()->value;
      auto inner = op->body.as<AttrStmt>();
      if (inner != nullptr && 
          inner->attr_key == attr::device_scope &&
          inner->value.as<StringImm>()->value == scope)  
        return stmt.as<AttrStmt>()->body;
    }
    return stmt;
  }

  Stmt Mutate_(const KernelDef *op, const Stmt& s) final {
    // check the kernel channels 
    CHECK(op->channels.size() % 2 == 0) 
      << "wrong index number in channels";
    // TODO: match buffer to extract graph
    for (auto& arg : op->args) {
      std::string name = arg.get()->name_hint;
      name = name.substr(name.find_last_of(".") + 1);
      kernel_arg[op->name].insert(name);
      edges[op->name] = {};
      // update edge {kernel : set}
      for (auto it = edges.begin(); it != edges.end(); it++) {
        if (it->first != op->name) {
          for (auto& s : kernel_arg[it->first]) {
            auto& curr_arg_set = kernel_arg[op->name];
            if (curr_arg_set.find(s) != curr_arg_set.end()) {
               edges[op->name].insert(it->first);
               edges[it->first].insert(op->name);
              // LOG(INFO) << op->name << ":" << it->first;
            }
          }
        }
      }
    }

    // insert (position, channel idx) into map
    for (size_t i = 0; i < op->channels.size(); i+=2) {
      auto pos = op->channels[i].as<IntImm>()->value;
      auto idx = op->channels[i+1].as<IntImm>()->value;
      kernel_arg_map[op->name].push_back(pos);
      kernel_arg_map[op->name].push_back(idx);
      kernel_channel_map[op->name].insert(idx);
    }
    // document groups connected with streaming channels
    bool found = false;
    for (auto it = kernel_channel_map.begin(); 
         it != kernel_channel_map.end(); ++it) {
      if (found) break;
      if (op->name != it->first) {
        for (auto i = kernel_channel_map[op->name].begin();
             i != kernel_channel_map[op->name].end(); i++) {
          if (it->second.find(*i) != it->second.end()) {
            // add kernel op->name to *it
            auto index = kernel_idx_map[it->first]; 
            kernel_grp_id[index].insert(op->name);
            kernel_idx_map[op->name] = index;
            found = true; break;
          } 
        }
      }
    } 
    if (!found) { // create new group if not found
      auto group_index = kernel_grp_id.size(); 
      kernel_idx_map[op->name] = group_index;
      kernel_grp_id.push_back({op->name});
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    return stmt;
  }

  // insert channel index & infer scheduling group 
  Stmt Mutate_(const KernelStmt *op, const Stmt& s) {

    // step 1: save buffer marker (discarded)
    if (op->annotate_keys.size() > 0) {
      for (size_t i = 0; i < op->annotate_keys.size(); i++) {
        auto pos = op->annotate_values[i].as<IntImm>()->value;
        marked_buffer.push_back(VarExpr(op->args[pos].node_));
      }
    }

    // step 2: do init shecduling 
    auto vector = kernel_arg_map[op->name];
    CHECK(vector.size() % 2 == 0) << "wrong size";
    Array<Expr> keys, values;

    // push into thread group id & timestep
    auto group_id = getThreadGroup(op->name);
    auto time_step = getTimeStep(group_id, op->name);
    keys.push_back(StringImm::make("group"));
    values.push_back(IntImm::make(Int(32), group_id));
    keys.push_back(StringImm::make("timestep"));
    values.push_back(IntImm::make(Int(32), time_step));
    // LOG(INFO) << op->name << " group:" << group_id 
    //           << " time:" << time_step;

    for (size_t i = 0; i < vector.size(); i++) {
      if (i % 2 == 0) { // create position index
        keys.push_back(StringImm::make("pos"));
        values.push_back(IntImm::make(Int(32), vector[i]));
      } else { // create entry for channel index
        keys.push_back(StringImm::make("index"));
        values.push_back(IntImm::make(Int(32), vector[i]));
      }
    } // return new kernel stmt
    return KernelStmt::make(op->args, op->name, 
                            keys, values);
  }

  // insert index into kernel stmt 
  Expr Mutate_(const KernelExpr *op, const Expr& e) {
    auto vector = kernel_arg_map[op->name];
    CHECK(vector.size() % 2 == 0) << "wrong size";
    Array<Expr> keys, values;
    // push into thread group id & timestep
    auto group_id = getThreadGroup(op->name);
    auto time_step = getTimeStep(group_id, op->name);
    keys.push_back(StringImm::make("group"));
    values.push_back(IntImm::make(Int(32), group_id));
    keys.push_back(StringImm::make("timestep"));
    values.push_back(IntImm::make(Int(32), time_step));

    for (size_t i = 0; i < vector.size(); i++) {
      if (i % 2 == 0) { // create position index
        keys.push_back(StringImm::make("pos"));
        values.push_back(IntImm::make(Int(32), vector[i]));
      } else { // create entry for channel index
        keys.push_back(StringImm::make("index"));
        values.push_back(IntImm::make(Int(32), vector[i]));
      }
    } // return new kernel stmt
    return KernelExpr::make(op->type, op->args, 
               op->name, keys, values);
  }

  Stmt Mutate_(const StreamStmt* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<StreamStmt>();
    return stmt;
  }

  Expr Mutate_(const StreamExpr* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<StreamExpr>();
    return expr;
  }

  // return thread group index for kernel stmt / expr
  int getThreadGroup(std::string name) {
    int num = -1;
    // all streaming connection
    if (edges[name].size() == kernel_channel_map[name].size()) {
      num = thread_group.size();
    } else { // has memory access
      for (auto it = edges[name].begin(); it != edges[name].end(); it++) {
        // has no channel overlap with neighbor *it
        bool overlap = false;
        auto neighbor = kernel_channel_map[*it];
        auto curr = kernel_channel_map[name];
        for (auto i = curr.begin(); i != curr.end(); i++) {
          if (neighbor.find(*i) != neighbor.end()) {
            // LOG(WARNING) << "overlap " << *it << ":" << name;
            overlap = true; break;  
          }
        }
        if (!overlap) { // check neighbor thread id
          for (size_t k = 0; k < thread_group.size(); k ++) {
            if (thread_group[k].find(*it) != thread_group[k].end()) {
              num = k; break;
            }
          } 
        } 
      }
      if (num == -1) 
        num = thread_group.size(); 
    }
    CHECK(num > -1) 
      << "not found group index";
    thread_group[num].insert(name); 
    return num;
  }

  // greedily schedule timestep for kernel stmt / expr
  int getTimeStep(int group_id, std::string name) {
    auto curr = timestep.size();
    if (curr == 0) {
      timestep.push_back({{group_id, name}});
      return 0;
    } else { // perform scheduling 
      for (size_t i = 0; i < curr; i++) {
        auto& map = timestep[i];
        if (map.find(group_id) == map.end() &&
            map.size() < thread_limit) {
          map[group_id] = name; 
          return i;
        }
      } // insert into next stage
      timestep.push_back({{group_id, name}});
      return curr;
    }
  }

  // time step vector for thread group allocation 
  std::vector<std::unordered_map<int, std::string>> timestep;
  // map from thread group id to kernel name set
  std::unordered_map<int, std::unordered_set<std::string>> thread_group;
  // connected components group of kernels
  std::vector<std::set<std::string>> kernel_grp_id;
  // egde set of kernel stmts
  std::unordered_map<std::string, std::unordered_set<std::string>> edges; 
  // kernel_grp_id index map for each kernel
  std::unordered_map<std::string, int> kernel_idx_map;
  // buffer varexprs to be marked (to generate pragma)
  std::vector<VarExpr> marked_buffer;

 private:
  int bus_bandwidth_;
  // thread limit 
  size_t thread_limit{16};
  // map from kernel name to vector of (pos, channel) index pair 
  std::unordered_map<std::string, std::vector<int>> kernel_arg_map;
  // map from kernel name to connected channel index set
  std::unordered_map<std::string, std::unordered_set<int>> kernel_channel_map;
  // connected group index to alloc num
  std::unordered_map<int, int> idx_count; 
  // kernel argument maps
  std::unordered_map<std::string, std::unordered_set<std::string>> kernel_arg;
};

// ir mutator to add info and update scheduling 
class InfoUpdater final : public IRMutator {
 public:
  InfoUpdater(
      std::vector<std::unordered_map<int, std::string>>& timestep,
      const std::vector<std::set<std::string>> connected_grp,
      const std::unordered_map<std::string, int> kernel_index_map,
      const std::vector<VarExpr>& marked_buffer)
      : timestep_(timestep), connected_grp_(connected_grp),
        kernel_index_map_(kernel_index_map),
        marked_buffer_(marked_buffer) {
      // perform reschduling (to avoid thread sync violation)
      for (size_t i = 0; i < timestep_.size(); i++) {
        if (i == 0) continue;
        auto& curr = timestep_[i];
        for (size_t j = 0; j < i; j++) { // previous steps
          auto& prev = timestep_[j];
          for (auto desc = curr.begin(); desc != curr.end(); desc++) { // check each desc
            for (auto pred = prev.begin(); pred != prev.end();) { // compare with pred
              // LOG(INFO) << "check " << desc->second << " : " << pred->second;
              bool remove = false;
              for (auto& set : connected_grp_) {
                if (set.find(desc->second) != set.end() &&
                    set.find(pred->second) != set.end() &&
                    set.find(pred->second) != set.find(desc->second)) {
                  // LOG(INFO) << "found violation " 
                  //           << desc->second << " : " << pred->second;
                  update_ = true; // delay pred op (into curr) 
                  curr[pred->first] = pred->second; 
                  changes_record[pred->second] = i;
                  pred = prev.erase(pred); 
                  remove = true; break;
                } 
              }
              if (!remove) pred++; 
            } 
          }
        }
      }
      if (false) { // print final scheduling 
        for (size_t i = 0; i < timestep_.size(); i++)
          for(auto& kv : timestep_[i]) 
            LOG(INFO) << i << ":" << kv.second;
      }
    }

  Stmt Mutate_(const KernelStmt* op, const Stmt& s) {
    Array<Expr> keys, values;
    for (size_t i = 0; i < op->annotate_keys.size(); i++) {
      // check annotate keys and udpate
      if (update_ && changes_record.find(op->name) != changes_record.end()  &&
          op->annotate_keys[i].as<StringImm>()->value == "timestep") {
        keys.push_back(StringImm::make("timestep"));
        values.push_back(IntImm::make(Int(32), changes_record[op->name]));
        auto num = timestep_[changes_record[op->name]].size();
        keys.push_back(StringImm::make("thread_num"));
        values.push_back(IntImm::make(Int(32), num));
      // insert thread num without updating
      } else if (op->annotate_keys[i].as<StringImm>()->value == "timestep") {
        auto step = op->annotate_values[i].as<IntImm>()->value;
        keys.push_back(StringImm::make("timestep"));
        values.push_back(IntImm::make(Int(32), step));
        auto num = timestep_[step].size();
        keys.push_back(StringImm::make("thread_num"));
        values.push_back(IntImm::make(Int(32), num));
      } else { // original information
        keys.push_back(op->annotate_keys[i]);
        values.push_back(op->annotate_values[i]);
      }
    }
    return KernelStmt::make(op->args, op->name, keys, values);
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    std::string name = op->buffer_var.get()->name_hint;
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();

    // 1. remove unnecessary alloc for kernel 
    while (! name.empty() && 
           name.find_last_of("0123456789") != std::string::npos) {
      if (name.find_last_of("0123456789") != name.size()-1 || 
          name.find("reuse") != std::string::npos) break;
      name.erase(name.size()-1, 1);
      if (kernel_index_map_.count(name)) {
        // LOG(INFO) << name << ":" << op->buffer_var.get()->name_hint;
        return op->body;
      }
    }

    // 2. mark the buffer allocator with pragma attr
    for (size_t i = 0; i < marked_buffer_.size(); i++) {
      if (op->buffer_var.get() == marked_buffer_[i].get()) {
        auto attrs = op->attrs;
        attrs.push_back(StreamStmt::make(VarExpr(op->buffer_var.node_), 
                            IntImm::make(Int(32), 0), StreamType::FIFO, 1,
                            Array<Expr>(), Array<Expr>()));
        return Allocate::make(
            op->buffer_var, op->type,
            op->extents, op->condition, op->body, attrs,
            op->new_expr, op->free_function);
      }
    }
    return stmt;
  }

 private:
  std::vector<std::unordered_map<int, std::string>>& timestep_;
  const std::vector<std::set<std::string>> connected_grp_;
  const std::unordered_map<std::string, int> kernel_index_map_;
  bool update_{false}; // schedule updating indicator
  std::unordered_map<std::string, int> changes_record;
  const std::vector<VarExpr>& marked_buffer_;
};

// group stmt into vectros
// since the device scope can be in the e.g. For stmt body
// we first decompose the nested stmts into sequence and re-combine 
// them into stmt blocks; here we use the  
class StmtGrouper final : public IRMutator {
 public:

  // record each stmt for each scope
  Stmt Mutate(Stmt s) final {
    // update stmt record flag
    if (const AttrStmt* op = s.as<AttrStmt>()) {
      if (op->attr_key == attr::device_scope) {
        auto scope = op->value.as<StringImm>()->value;
        if (scope != scope_) { 
          scope_ = scope;
          if (scope != "cpu") record_ = true; 
          else record_ = false; // stop recording 
        }
      }
    }

    if (record_) { // push into device stack 
      Stmt no_op = Evaluate::make(0);
      if (const For* op = s.as<For>()) {
        stmt_stack.emplace_back(s);
        return s; // TODO: allow device scope change in for stmt 
        stmt_stack.emplace_back(For::make(
            op->loop_var, op->min, op->extent,
            op->for_type, op->device_api, no_op,
            op->annotate_keys, op->annotate_values));
        return IRMutator::Mutate(s);

      } else if (const LetStmt* op = s.as<LetStmt>()) {
        stmt_stack.emplace_back(LetStmt::make(op->var, op->value, no_op));
        return s;

      } else if (s.as<IfThenElse>()) {
        stmt_stack.emplace_back(s);
        return s; // TODO: allow device scope change in if-else 

      } else if (const AttrStmt* op = s.as<AttrStmt>()) {
        if (op->attr_key != attr::device_scope)
          stmt_stack.emplace_back(AttrStmt::make(
              op->node, op->attr_key, op->value, no_op));
        Stmt ret = IRMutator::Mutate(s);
        return ret;

      } else if (s.as<ProducerConsumer>()) {
        Stmt ret = IRMutator::Mutate(s);
        return ret;

      } else if (s.as<Block>()) {
        return IRMutator::Mutate(s);

      } else if (const Allocate* op = s.as<Allocate>()) {
        stmt_stack.emplace_back(Allocate::make(
            op->buffer_var, op->type, op->extents, op->condition, 
            no_op, op->attrs, op->new_expr, op->free_function));
        return IRMutator::Mutate(s);

      } else if (s.as<StreamStmt>()) {
        stmt_stack.emplace_back(s);
        return s; // smallest granularity 

      } else if (s.as<Store>()) {
        stmt_stack.emplace_back(s);
        return s;

      } else if (const KernelDef* op = s.as<KernelDef>()) {
        std::vector<Stmt> stack;
        stack.push_back(stmt_stack.back()); stmt_stack.pop_back();
        stack.push_back(stmt_stack.back()); stmt_stack.pop_back();
        stack.push_back(s);
        kernel_stack.emplace_back(MergeStmts(stack));
        return s; // TODO: allow device scope change in kernel 
        stmt_stack.emplace_back(KernelDef::make(
            op->args, op->api_args, op->api_types, no_op, 
            op->ret_void, op->ret_type, op->name, op->channels));

      } else if (s.as<KernelStmt>()) {
        stmt_stack.emplace_back(s);
        return IRMutator::Mutate(s);

      } else {
        return IRMutator::Mutate(s);
      }

    } else { // push in host stack
    }
    return IRMutator::Mutate(s);
  }

  int level{0}; // scope indicator
  std::string scope_{"cpu"};
  bool record_{false}; // record stmt in device scope
  std::vector<Stmt> stmt_stack; // kernel def exclusive
  std::vector<Stmt> kernel_stack;

};

// split program based on device type 
// 1. simple case (blocked stmt nest) we first group the stmts 
//    into vector and then return a combo of KernelDef & KernelStmt
//    grouping should be conducted in a higher level than stmt. 
//    first stage to record stmt; second to perform kernel def creation
//    & insertion (if a stmt node is in) 
//
class SplitDevice final : public IRMutator {
 public:
  SplitDevice(Array<NodeRef>& api_args,
              std::vector<Array<Var>>& undefined_vars,
              std::vector<Stmt>& stacked_stmt,
              std::vector<Stmt>& stacked_kernel) 
  : undefined_vars_(undefined_vars), stmt_stack_(stacked_stmt),
    kernel_stack(stacked_kernel) {
    for (size_t i = 0; i < api_args.size(); i++) {
      if (api_args[i].as<BufferNode>()) {
        Buffer buf(api_args[i].node_);
        CHECK(buf->data.as<Variable>());
        api_shape_[buf->data.get()] = buf->shape;
        api_dtype_[buf->data.get()] = buf->dtype;
      } else { // as variable
        auto v = api_args[i].as<Variable>();
        CHECK(v) << "invalid input buf type";
        api_shape_[v] = Array<Expr>({1});
      }
    }
  }

  // record each stmt for each scope
  Stmt Mutate(Stmt s) final {
    if (return_) { // only return host stmt
      if (s.as<For>()) { // create host channel allocation 
        StreamCollector sc; sc.Visit(s);
        if (sc.channel_buf_.defined()) {
          VarExpr buf(sc.channel_buf_.node_);
          // TODO find a better to alloc host channel buf
          if (buf.get()->name_hint.substr(0,6) == "c_buf_")
            return IRMutator::Mutate(s);
          Type type = sc.channel_buf_->type;
          s = Allocate::make(VarExpr(sc.channel_buf_.node_), 
                     type, sc.var_shape_, make_const(Bool(type.lanes()), true), s);
          return AttrStmt::make(buf, attr::storage_scope, StringImm::make("global"), s);
        }
      }
      return IRMutator::Mutate(s);

    } else {
      // TODO: allow device scope change in for stmt 
      if (s.as<For>() || s.as<LetStmt>() || s.as<IfThenElse>() ||
          s.as<StreamStmt>() || s.as<Store>() || s.as<KernelStmt>() ||
          s.as<KernelDef>()) {
        Stmt no_op = Evaluate::make(0);
        return no_op;
      } else { // traverse inner stmts
        if (const Allocate* op = s.as<Allocate>()) {
          return this->Mutate(op->body);
        } else if (const AttrStmt* op = s.as<AttrStmt>()) {
          if (op->attr_key != attr::device_scope)
            return this->Mutate(op->body);
        }
        Stmt ret = IRMutator::Mutate(s);
        return ret;
      }
    }
  }

  // create kernel def node for host-xcel transition  
  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::device_scope) {
      auto scope = op->value.as<StringImm>()->value;
      if (scope != scope_) { 
        scope_ = scope;
        if (scope != "cpu") { 
          return_ = false; // skip device scope stmt
          Stmt body = this->Mutate(op->body);
          return BuildKernel(op->node, op->value, body); 
        } else { // stop recording 
          return_ = true;
          return IRMutator::Mutate_(op, s);
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt SplitScope(Stmt stmt) {
    // collect shape & type of device func args
    for (size_t i = 0; i < undefined_vars_.size(); i++) {
      TypeCollector collector(undefined_vars_[i]);
      collector.Visit(stmt);
      shape_.push_back(collector.shape_);
      dtype_.push_back(collector.dtype_);
    }
    Stmt s = Mutate(stmt);
    for (auto& k : kernel_defs_) s = Block::make(k, s);
    for (auto& k : kernel_stack) s = Block::make(k, s);
    return RemoveNoOp(s);
  }

 private:
  // insert kernel def & stmt to wrap in scope stmts
  Stmt BuildKernel(NodeRef target, Expr scope, Stmt& stmt) {
    Map<Var, Expr> subst; // create substitute variables 
    auto arg_vars = undefined_vars_[dev_scope_count]; 
    Array<VarExpr> new_vars;
    Array<Expr> func_call_args;
    auto shape_map = shape_[dev_scope_count];
    auto dtype_map = dtype_[dev_scope_count];

    std::vector<Stmt> channels;
    // create shape & dtype & substitute map
    Array<Array<Expr>> shapes; Array<Expr> types;
    for (size_t k = 0; k < arg_vars.size(); k++) {

      auto var = Var(arg_vars[k].node_);
      Type type = dtype_map[var.get()];
      Array<Expr> shape = shape_map[var.get()];
      // handle host allocated channel buffer
      std::string var_name = var.get()->name_hint;
      if (var_name.find(".channel") != std::string::npos) {
        Stmt no_op = Evaluate::make(0);
        auto stmt = Allocate::make(VarExpr(var.node_), type, shape,
                        make_const(Bool(type.lanes()), true), no_op);
        channels.push_back(stmt); continue;
      }

      if (shape.size() == 0) {
        type  = api_dtype_[var.get()];
        shape = api_shape_[var.get()];
      }
      shapes.push_back(shape);
      types.push_back(Type2Expr(type));
      
      // VarExpr new_var = Variable::make(var.type(), var->name_hint + ".new");
      VarExpr new_var(var.node_);
      new_vars.push_back(new_var);
      func_call_args.push_back(Expr(arg_vars[k].node_));
      subst.Set(var, Expr(new_var.node_));
    }

    auto body = Substitute(MergeStmts(stmt_stack_), subst);
    for (size_t k = 0; k < channels.size(); k++) {
      auto op = channels[k].as<Allocate>();
      body = Allocate::make(op->buffer_var, op->type, op->extents,
                            op->condition, body); 
      body = AttrStmt::make(VarExpr(op->buffer_var.node_), attr::storage_scope, 
                            StringImm::make("local"), body);
    }
    std::string name = "top_function_" + std::to_string(dev_scope_count);
    auto kernel = KernelDef::make(new_vars, shapes, types, body, 
                      UIntImm::make(UInt(1), 1), /*ret_void*/
                      UInt(32), name, Array<Expr>() /*channels indices*/); 
    kernel_defs_.push_back(kernel);
    auto kernel_stmt = KernelStmt::make(func_call_args, name);
    stmt = Block::make(stmt, kernel_stmt);

    dev_scope_count += 1;
    return AttrStmt::make(
        target, ir::attr::device_scope, scope, stmt);
  }

  std::string scope_{"cpu"};
  int dev_scope_count{0}; 
  bool return_{true};
  const std::vector<Array<Var>>& undefined_vars_;
  std::vector<Stmt>& stmt_stack_;
  std::vector<Stmt> kernel_defs_;
  std::vector<Stmt> kernel_stack;
  std::vector<std::unordered_map<const Variable*, Array<Expr>>> shape_;
  std::vector<std::unordered_map<const Variable*, Type>> dtype_;
  // shape & dtype from api_args buffers
  std::unordered_map<const Variable*, Array<Expr>> api_shape_;
  std::unordered_map<const Variable*, Type> api_dtype_;
};

// 1. add annotation to kernel def node 
// 2. mutate the producer marked with .new 
class KernelAnnotator final : public IRMutator {
 public:
  KernelAnnotator(
    std::unordered_map<std::string, std::unordered_set<int>> map,
    Array<NodeRef>& api_args) :
    arg_scope_map_(map) {

      for (size_t i = 0; i < api_args.size(); i++) {
        if (api_args[i].as<BufferNode>()) {
          Buffer buf(api_args[i].node_);
          CHECK(buf->data.as<Variable>());
          args.insert(buf->data.get());
        } else { // as variable
          auto v = api_args[i].as<Variable>();
          CHECK(v) << "invalid input buf type";
          args.insert(v);
        }
      }
    } 

  // mark the kernel def (arg[pos] is in global scope)
  Stmt Mutate_(const KernelDef *op, const Stmt& s) final {
    Stmt body = this->Mutate(op->body);
    if (arg_scope_map_.count(op->name)) {
      auto set = arg_scope_map_[op->name];
      Array<Expr> channels = op->channels;

      // insert annotation (pos : index = -1) indicate global
      for (size_t i = 0; i < op->args.size(); i++) {
        if (set.find(i) != set.end()) {
          channels.push_back(IntImm::make(Int(32), i));
          channels.push_back(IntImm::make(Int(32), -1));
        }
      }
      return KernelDef::make(op->args, op->api_args, op->api_types, 
                 body, op->ret_void, op->ret_type, op->name, channels);
    }
    return s;
  }

  // record the shape & type of allocated buffer
  Stmt Mutate_(const Allocate *op, const Stmt& s) final {
    auto v = op->buffer_var.get();
    type_[v] = op->type;
    shape_[v] = op->extents;
    return IRMutator::Mutate_(op, s);
  }

  // create buffer on host if tensor allocated in xcel scope 
  Stmt Mutate_(const ProducerConsumer *op, const Stmt& s) final {
    if (op->is_producer &&
        op->func->func_name().find(".new") != std::string::npos) {
      if (auto attr = op->body.as<AttrStmt>()) {
        if (attr->attr_key == attr::device_scope) {
          if (attr->value.as<StringImm>()->value == "cpu" && 
              !args.count(attr->node.as<Variable>())) {
            auto v = attr->node.as<Variable>();
            Stmt body = Allocate::make(VarExpr(attr->node.node_), type_[v], 
                            shape_[v], make_const(Bool(type_[v].lanes()), true), op->body);
            return AttrStmt::make(VarExpr(attr->node.node_), attr::storage_scope,
                                  StringImm::make("global"), body);
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s); 
  }

 private:
  std::unordered_map<const Variable*, Type> type_;
  std::unordered_map<const Variable*, Array<Expr>> shape_;
  std::unordered_map<std::string, std::unordered_set<int>> arg_scope_map_;
  std::unordered_set<const Variable*> args;
};

Stmt InferStream(Stmt stmt,  
                 Array<NodeRef> api_args,
                 int bus_bandwidth) {
  // TODO: bind buffer & tensor in StorageFlatten
  StreamAnalysis analyzer(api_args);
  stmt = analyzer.Mutate(stmt);
  StreamMutator mutator(bus_bandwidth);
  stmt = mutator.Mutate(stmt); 
  // update timestep scheduling 
  InfoUpdater updater(/*vector of {groupId:name}*/mutator.timestep,
                      /*streaming connectivity*/mutator.kernel_grp_id,
                      /*connection group id*/mutator.kernel_idx_map,
                      /*streamed buffer mark*/mutator.marked_buffer);
  stmt = updater.Mutate(stmt);
  // organize the isolated sub-graph into calls
  StmtGrouper grouper; grouper.Mutate(stmt);
  stmt = SplitDevice(/*buffer or input args*/api_args,
                     /*undefined arg array*/analyzer.streaming_vars, 
                     /*device scope stmts*/grouper.stmt_stack,
                     /*kernel stmts*/grouper.kernel_stack).SplitScope(stmt);

  // mark kernel def with storage scope
  stmt = KernelAnnotator(analyzer.kernel_arg_scope_,
                         api_args).Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace TVM
