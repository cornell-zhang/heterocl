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
  if (t.code()  == Type::Handle) 
    return StringImm::make("handle");
  std::ostringstream os;
  os << t;
  return StringImm::make(os.str());
}

class StoreToStreamStmtConverter final : public IRMutator {
  public: 
    StoreToStreamStmtConverter(
        const std::string& target,
        const ir::StreamType& type,
        const VarExpr& channel_buf,
        const int channel_depth,
        int channel_index,
        const Array<Expr> shape,
        std::unordered_map<const Variable*, Expr>& range) 
      : target_(target), type_(type), channel_buf_(channel_buf),
        channel_depth_(channel_depth), channel_index_(channel_index), 
        shape_(shape), range_(range) {} 

    Stmt Mutate_(const Store* op, const Stmt& s) {
      Expr index = op->index;
      Expr value = this->Mutate(op->value);
      std::string target_name = op->buffer_var.get()->name_hint;
      if (target_name == target_) {
        Array<Expr> keys, values;
        // push channel and access information 
        keys.push_back(StringImm::make("index"));
        values.push_back(index);
        keys.push_back(StringImm::make("channel"));
        values.push_back(IntImm::make(Int(32), channel_index_));
        return StreamStmt::make(VarExpr(channel_buf_.node_), value, 
                                type_, channel_depth_, keys, values); 
      } else {
        return Store::make(op->buffer_var, value, 
                           index, op->predicate);
      }
    }

  private:
    const std::string target_;
    const ir::StreamType type_;
    const VarExpr& channel_buf_;
    const int channel_depth_;
    const int channel_index_;
    const Array<Expr> shape_;
    std::unordered_map<const Variable*, Expr>& range_;
};

class LoadToStreamExprConverter final : public IRMutator {
  public: 
    LoadToStreamExprConverter(
        const std::string& target,
        const ir::StreamType& type,
        const VarExpr& channel_buf,
        const int channel_depth,
        int channel_index, 
        const Array<Expr> shape,
        std::unordered_map<const Variable*, Expr>& range) 
      : target_(target), type_(type), channel_buf_(channel_buf),
        channel_depth_(channel_depth), channel_index_(channel_index), 
        shape_(shape), range_(range) {} 

    // record axis to mutate streaming sender 
    Stmt Mutate_(const For* op, const Stmt& s) {
      Stmt stmt = IRMutator::Mutate_(op, s);
      if (found) // in the right track
        loop_vars.push_back(op->loop_var.get());
      return stmt;
    }

    // single load repalcement 
    Expr Mutate_(const Load* op, const Expr& e) {
      Expr index = op->index;
      std::string target_name = op->buffer_var.get()->name_hint;
      if (target_ == target_name) {
        Array<Expr> keys, values;
        // push channel and access information 
        keys.push_back(StringImm::make("index"));
        values.push_back(std::move(op->index));

        keys.push_back(StringImm::make("channel"));
        values.push_back(IntImm::make(Int(32), channel_index_));
        return StreamExpr::make(op->type, VarExpr(channel_buf_.node_), 
                                type_, channel_depth_, keys, values);
      } else {
        return Load::make(op->type, op->buffer_var, 
                          index, op->predicate);
      }
   }
    std::vector<const Variable*> loop_vars;

  private:
    bool found{false};           // found tagret load op 
    const std::string target_;   // stream variable name 
    const ir::StreamType type_;  // stream types (fifo, channel, pipe)
    const VarExpr& channel_buf_; // streaming channel buffer
    const int channel_depth_;    // stream channel depth (no less than 0)
    const int channel_index_;    // stream channel index (share no more than 2 agents)
    const Array<Expr> shape_;    // shape array of target load op
    std::unordered_map<const Variable*, Expr>& range_; // range map of IterVar
};

// block kernel body with reuse buffer
Stmt BufferInserter(Stmt stmt, /*original extern op body*/
                    Array<Expr> shape, /*target buffer shape*/ 
                    const VarExpr& target, /*target load & store buf*/
                    const VarExpr& c_buf, /*channel buffer*/
                    bool load_mode, /*load or store mode*/
                    StreamType type, int channel_depth) {
  // compute indices for load / store
  std::vector<Expr> indices;
  std::vector<VarExpr> loop_vars;
  for (size_t i = 0; i < shape.size(); i++) {
    VarExpr iter("buf_" + std::to_string(i));
    indices.push_back(iter);
    loop_vars.push_back(iter);
  }
  Expr index = FlattenIndices(indices, shape); 
  
  if (load_mode) { // local buffer reading from stream channel  
    Expr stream = StreamExpr::make(target->type,
                                   VarExpr(c_buf.node_),
                                   type, channel_depth);
    // store op initialized with variable node
    Stmt for_stmt = Store::make(VarExpr(target.node_),
                                stream, index,
                                UIntImm::make(UInt(1), 1));
    
    auto type = ForType::Serial;
    for (size_t j = 0; j < shape.size(); j++) {
      auto iter = loop_vars[j];
      // DMA burst loading from sys memory  
      if (j == shape.size() - 1) type = ForType::Pipelined; 
      for_stmt = For::make(VarExpr(iter.node_), 0, shape[j],
                           type, DeviceAPI::None, for_stmt);
    }
    stmt = Block::make(for_stmt, stmt); 

  } else { // multiple stores : sending at end 
    Expr load = Load::make(target->type,
                           VarExpr(target.node_), index, 
                           UIntImm::make(UInt(1), 1));
    Stmt for_stmt = StreamStmt::make(VarExpr(c_buf.node_),
                                     load, type, channel_depth);

    auto type = ForType::Serial;
    for (size_t j = 0; j < shape.size(); j++) {
      auto iter = loop_vars[j];
      // DMA burst store to sys memory  
      if (j == shape.size() - 1) type = ForType::Pipelined; 
      for_stmt = For::make(VarExpr(iter.node_), 0, shape[j],
                           type, DeviceAPI::None, for_stmt);
    }
    stmt = Block::make(stmt, for_stmt); 
  }

  return stmt;
};

// collect access pattern for target vars
class AccessCollector : public ir::IRMutator {
 public:
  explicit AccessCollector(
      const VarExpr& target_buf, const Array<Expr>& shape,
      const std::unordered_map<const Variable*, Expr>& range,
      const std::string channel_name)
      : target_buf_(target_buf), shape_(shape), range_(range), 
        channel_name_(channel_name) {}

  // trace buffer allocation 
  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    std::string target_name = op->buffer_var.get()->name_hint;
    // whether the target buffer has been allocated 
    if (target_name == channel_name_) buf_alloc = true;
    return s;
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    Expr value = this->Mutate(op->value);
    std::string target_name = op->buffer_var.get()->name_hint;
    // check if target buffer matches
    if (op->buffer_var.get() == target_buf_.get()) {
      store_num += 1; 
      store_var = VarExpr(op->buffer_var.node_);
      // check index access regularity 
      auto max_bound = Substitute(op->index, range_); 
      reg_store = is_zero(Simplify(max_bound - get_max(shape_)));
    }
    return s;
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    std::string target_name = op->buffer_var.get()->name_hint;
    // check if buffer matches
    if (op->buffer_var.get() == target_buf_.get()) {
      load_num += 1; 
      load_var = VarExpr(op->buffer_var.node_);
      // check index access regularity 
      auto max_bound = Substitute(op->index, range_); 
      reg_load = is_zero(Simplify(max_bound - get_max(shape_)));
    }
    return e;
  }

  int load_num{0};
  int store_num{0};
  VarExpr load_var;
  VarExpr store_var;
  bool reg_store{true};
  bool reg_load{true};
  bool buf_alloc{false};

 private:
  const VarExpr& target_buf_; /*stream variable buffer*/ 
  const Array<Expr>& shape_;  /*stream variable shape*/ 
  const std::unordered_map<const Variable*, Expr>& range_;
  const std::string channel_name_;

  Expr get_max(Array<Expr> shape) {
    Expr ret(shape[0]); 
    for (size_t i = 1; i < shape.size(); i++) ret *= shape[i]; 
    return Simplify(ret - 1); 
  }
};

// create streaming channels across loop iterations
class LoopbackMutator : public ir::IRMutator {
 public:
  explicit LoopbackMutator(
    const VarExpr& target_buf, const Array<Expr>& shape,
    const std::unordered_map<const Variable*, Expr>& range, 
    Type type)
  : target_buf_(target_buf), shape_(shape), 
    range_(range), type_(type) {} 

  // FIXME: buffer mismatch 
  Stmt Mutate_(const Store* op, const Stmt& s) {
    if (op->buffer_var->name_hint == target_buf_->name_hint) {
      if (store_count == 0) { 
        store_count += 1;
        CHECK(!temp_.defined());
        temp_ = VarExpr("temp_" + target_buf_->name_hint); 
        auto index = IntImm::make(Int(32), 0);
        Expr load_expr = Load::make(type_, 
                             temp_, index, op->predicate);
        save_stmt = Store::make(op->buffer_var, 
                        load_expr, op->index, op->predicate);

        Stmt stmt = Store::make(temp_, op->value, index, op->predicate);
        stmt = Allocate::make(temp_, type_, Array<Expr>(),
            make_const(Bool(type_.lanes()), true), stmt);
        stmt = AttrStmt::make(temp_, attr::storage_scope,
            StringImm::make("local"), stmt);
        return stmt;

      } else {
        store_count += 1;
        auto index = IntImm::make(Int(32), 0);
        return Store::make(temp_, op->value, index, op->predicate);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    if (op->buffer_var->name_hint == target_buf_->name_hint) {
      if (store_count > 0) { 
        auto index = IntImm::make(Int(32), 0);
        return Load::make(op->type, temp_, index, op->predicate);
      }
    }
    return e;
  }

  // create stream array
  Stmt Mutate_(const For* op, const Stmt& s) {

    if (op->body.as<For>() == nullptr) {
      Stmt stmt = this->Mutate(op->body);
      stmt = Block::make(stmt, save_stmt);
      return For::make(
          op->loop_var, op->min, op->extent, op->for_type,
          op->device_api, stmt, op->annotate_keys,
          op->annotate_values);

    } else {
      Stmt stmt = this->Mutate(op->body);
      return For::make(
          op->loop_var, op->min, op->extent, op->for_type,
          op->device_api, stmt, op->annotate_keys,
          op->annotate_values);
    }
  }

  private:
   const VarExpr& target_buf_;
   const Array<Expr>& shape_;
   const std::unordered_map<const Variable*, Expr>& range_;
   Type type_; 
   VarExpr temp_;
   int store_count{0};
   Stmt save_stmt;
};

/*!
 * \brief An IRVisitor to collect information 
 *        of undefined variables 
 *
 * Collect streaming information:
 *   1. collect undefined variable residing in 
 *      device scope  
 *   2. bind umatched buffers with top function
 *      tensors in the api_args
 *
 * */
class StreamAnalyzer final : public IRMutator {
 public:
  StreamAnalyzer(Array<NodeRef>& api_args) {
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

        // data type and shape info
        shape_[buf->data.get()->name_hint] = buf->shape;
        dtype_[buf->data.get()->name_hint] = buf->dtype;
      }
    }
  };

  // record undefined var in each scope 
  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::device_scope) {

      Stmt body;
      // xcel scope wrapper 
      if (!op->node.defined()) { 
        record_ = true;
        body = this->Mutate(op->body);
        record_ = false;

        // TODO: use unifed undef var 
        if (new_vars.size()) {
          streaming_vars.push_back(new_vars);
          new_vars = {};
        } else {
          CHECK(undefined_.size());
          streaming_vars.push_back(undefined_);
          undefined_ = {};
        }

      } else {
        body = this->Mutate(op->body);
      }

      // replace the buffer node
      if (auto buf = op->node.as<BufferNode>()) {
        std::string name = buf->name;
        VarExpr buf_var(bind_buffer_map_[name].node_);
        CHECK(body.defined()) << "body not defined";
        return AttrStmt::make(buf_var, 
                   op->attr_key, op->value, body);

      } else {
        CHECK(body.defined()) << "body not defined";
        return AttrStmt::make(op->node, 
                   op->attr_key, op->value, body);
      }
    }
    return IRMutator::Mutate_(op, s); 
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

    } else { 
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
    } else { 
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
    itervars_.insert(op->loop_var.get());
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt& s) final {
    auto v = op->buffer_var.get();
    auto name = v->name_hint; 
    bind_buffer_map_[name] = Expr(op->buffer_var.node_);

    // save shape and dtype information
    shape_[name] = op->extents;
    dtype_[name] = op->type;

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

    } else { 
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

    // TODO add config info to ir node
    if (auto val = op->value.as<StringImm>()) {
      if (val->value == "config") {
        CHECK(op->annotate_values.size() == 4);
        Array<Expr> dev_port(op->annotate_values);
        auto buffer = op->buffer_var.as<BufferNode>();
        CHECK(buffer != nullptr);
        mem_ports[buffer->name] = dev_port;
        return Evaluate::make(0);
      }
    } 

    if (auto buf = op->buffer_var.as<BufferNode>()) {
      std::string name = buf->name;
      VarExpr buf_var(bind_buffer_map_[name].node_);
      this->HandleUse(buf_var);
      Expr value = this->Mutate(op->value);
      return StreamStmt::make(buf_var, value, op->stream_type, op->depth,
                              op->annotate_keys, op->annotate_values);

    } else { 
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
      CHECK(v.as<Variable>()) << v << " not a variable";
      Var var(v.node_);
      if (var->name_hint.find("channel") != std::string::npos) { 
        if (!new_var_nodes.count(var.get())) {
          new_vars.push_back(var); 
          new_var_nodes.insert(var.get());
          return;
        }
      }
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

  std::unordered_map<int, Expr> index_map;
  bool record_{false};
  Array<Var> undefined_;
  Array<Var> new_vars;
  std::unordered_map<const Variable*, int> use_count_;
  std::unordered_map<const Variable*, int> def_count_;
  std::vector<Array<Var>> streaming_vars;
  std::unordered_map<std::string, Expr> bind_buffer_map_;
  std::unordered_set<std::string> top_arg_names;
  std::unordered_map<std::string, std::unordered_set<int>> kernel_arg_scope_;
  std::unordered_set<const Variable*> itervars_;
  std::unordered_set<const Variable*> new_var_nodes;
  std::unordered_map<std::string, Array<Expr>> shape_;
  std::unordered_map<std::string, Type> dtype_;
  // extract memory interface information
  std::unordered_map<std::string, Array<Expr>> mem_ports;

};

class StreamMutator : public IRMutator {
 public:
  explicit StreamMutator() {}

  Stmt Mutate_(const KernelDef *op, const Stmt& s) final {
    // check the kernel channels 
    CHECK(op->channels.size() <= op->args.size()) 
      << "conflicting entries in op->channels";
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
    for (size_t i = 0; i < op->channels.size(); i++) {
      Array<Expr> info = op->channels[i];
      CHECK(info.size() == 6);
      auto pos = info[0].as<IntImm>()->value;
      auto idx = info[1].as<IntImm>()->value;
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

// create local copy and sync with data copy 
class MultiLoadMutator : public IRMutator {
 public:
  explicit MultiLoadMutator(
    std::string& target,
    std::vector<VarExpr>& channels, Type type)
    : target_(target), channels_(channels), type_(type) {}

  Stmt Mutate(Stmt stmt) final {
    Stmt ret = IRMutator::Mutate(stmt);
    if (found && !alloc) { 
      for (auto& channel : channels_) {
        auto stream_expr = StreamExpr::make(type_, 
            VarExpr(channel.node_), StreamType::FIFO, 
            1, Array<Expr>(), Array<Expr>()); 

        auto store = Store::make(temp_, 
                stream_expr, Expr(0), const_true());
        ret = Block::make(store, ret);
      }
      ret = Allocate::make(temp_, type_, Array<Expr>(),
          make_const(Bool(type_.lanes()), true), ret);
      ret = AttrStmt::make(temp_, attr::storage_scope,
          StringImm::make("local"), ret);
      alloc = true;
    }
    return ret;
  }

  Expr Mutate_(const Load *op, const Expr& e) final {
    Expr index = op->index;
    std::string target_name = op->buffer_var.get()->name_hint;

    Stmt stmt;
    if (target_name == target_) {
      found = true;
      temp_ = VarExpr("temp_" + target_);
      return Load::make(op->type, temp_, index, op->predicate);
    } else {
      return Load::make(op->type, op->buffer_var, index, op->predicate);
    }
  }

 private:
  std::string& target_;
  std::vector<VarExpr>& channels_;
  Type type_;
  VarExpr temp_;
  bool found{false};
  bool alloc{false};
};

// create local copy and multiple streaming channels
class MultiCastMutator : public IRMutator {
 public:
  explicit MultiCastMutator(
    std::string& target,
    std::vector<VarExpr>& channels, Type type)
    : target_(target), channels_(channels), type_(type) {}

  Stmt Mutate_(const Store *op, const Stmt& s) final {
    Expr index = op->index;
    Expr value = this->Mutate(op->value);
    std::string target_name = op->buffer_var.get()->name_hint;
    if (target_name == target_) {
      VarExpr temp("temp");
      Stmt stmt = Store::make(temp, value, Expr(0), op->predicate);
      for (auto& channel : channels_) {
        auto stream_stmt = StreamStmt::make(
            VarExpr(channel.node_), temp, 
            StreamType::FIFO, 1, Array<Expr>(), Array<Expr>()); 
        stmt = Block::make(stmt, stream_stmt);
      }
      stmt = Allocate::make(temp, type_, Array<Expr>(),
          make_const(Bool(type_.lanes()), true), stmt);
      stmt = AttrStmt::make(temp, attr::storage_scope,
          StringImm::make("local"), stmt);
      return stmt;
    } else {
      return Store::make(op->buffer_var, value, 
                         index, op->predicate);
    }
  }

 private:
  std::string& target_;
  std::vector<VarExpr>& channels_;
  Type type_;

};

// analyze varibles in decorated scope  
class StmtGrpReplacer final : public IRMutator {
 public:
  explicit StmtGrpReplacer(
      std::vector<Array<Var>>& undefined_vars,
      std::unordered_map<std::string, Array<Expr>>& shape,
      std::unordered_map<std::string, Type>& dtype)
  : undef_vars(undefined_vars), 
    shape_(shape), dtype_(dtype) {}

  // move channel allocation from xcel to host 
  Stmt Mutate_(const Allocate *op, const Stmt& s) final {
    auto v = op->buffer_var;
    auto name = v->name_hint;

    if (xcel_scope) {
      size_t start_pos = name.find(".channel");
      if(start_pos != std::string::npos) {
        channel_map_[name] = v;
        return this->Mutate(op->body);
      } 
    }
    return IRMutator::Mutate_(op, s);
  }


  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::device_scope) {

      // attr stmt with empty node  
      if (!op->node.defined()) { 
        xcel_scope = true; 
        Stmt body = this->Mutate(op->body);
        xcel_scope = false; 

        // create kernel def body 
        Map<Var, Expr> subst; 
        CHECK((unsigned)scope_counter < undef_vars.size());

        auto arg_vars = undef_vars[scope_counter];
        Array<VarExpr> new_vars;
        Array<Expr> func_call_args;
        std::set<std::string> arg_names;

        // shape dtype and substitute map for channels
        Array<Array<Expr>> shapes; Array<Expr> types;
        for (size_t k = 0; k < arg_vars.size(); k++) {

          auto var = Var(arg_vars[k].node_);
          std::string name = var.get()->name_hint;
          Type type = dtype_[name];
          Array<Expr> shape = shape_[name];
          arg_names.insert(name);
          if (shape.size() == 0) shape = {1};

          shapes.push_back(shape);
          types.push_back(Type2Expr(type));
          
          VarExpr new_var(var.node_);
          new_vars.push_back(new_var);
          func_call_args.push_back(Expr(var.node_));
          subst.Set(var, Expr(new_var.node_));
        }

        body = Substitute(body, subst);
        // create buffers if api_args used in kernel body
        auto undefs = UndefinedVars(body, Array<Var>());
        for (auto& var : undefs) {
          if (var->name_hint.find(".channel") == std::string::npos) {
            auto name = var->name_hint;
            if (arg_names.count(name)) continue;
            Type type = dtype_[name];
            Array<Expr> shape = shape_[name];
            LOG(INFO) << "Creating allocate statement for UndefinedVar " << var;
            body = Allocate::make(var, type, shape,
                make_const(Bool(type.lanes()), true), body);
            body = AttrStmt::make(var, attr::storage_scope,
                StringImm::make("global"), body);
          }
        }
        auto kernel = KernelDef::make(new_vars, shapes, types, 
                          Array<FunctionRef>(), body, 
                          UIntImm::make(UInt(1), 1),
                          UInt(32), "test", Array<Array<Expr>>()); 
        kernel_defs_.push_back(kernel);
        Stmt stmt = KernelStmt::make(func_call_args, "test");

        // allocate channel buffers to host
        for (auto& var : arg_vars) {
          auto name = var->name_hint; 
          if (channel_map_.count(name)) {
            CHECK(dtype_.count(name)) << "dtype not found";
            CHECK(shape_.count(name)) << "shape not found";

            VarExpr var(channel_map_[name].node_);
            Type type = dtype_[name];
            Array<Expr> shape = shape_[name];

            stmt = Allocate::make(var, type, shape,
                make_const(Bool(type.lanes()), true), stmt);
            stmt = AttrStmt::make(var, attr::storage_scope,
                StringImm::make("global"), stmt);
          }
        }

        scope_counter += 1;
        return AttrStmt::make(
              op->node, op->attr_key, op->value, stmt);

      } else { // mutate inner tensor 

        auto target = VarExpr(op->node.node_)->name_hint;
        int index = op->value.as<IntImm>()->value;

        // self-loopback
        if (index == 0) {
          Stmt stmt = this->Mutate(op->body);
          Type dtype = dtype_[target];
          Array<Expr> shape = shape_[target];
          auto range_ = CollectIterRange(op->body);

          // replace with local temp
          auto target_buf = VarExpr(op->node.node_);
          LoopbackMutator mutator(
              target_buf, shape, range_, dtype);
          stmt = mutator.Mutate(stmt);
          return stmt; 
        }

        bool data_load = index < 0 ? false : true;
        if (index < 0) index = -1 * index;
   
        // map from target to streaming channels 
        std::unordered_map<std::string, std::vector<StreamInfo>> map;
        StreamInfo si{index, data_load};
        map[target].push_back(si);

        Stmt body = op->body;
        while (const AttrStmt* attr = body.as<AttrStmt>()) {
          if (attr->attr_key == attr::device_scope) {
            auto new_target = VarExpr(attr->node.node_)->name_hint;
            int new_index = attr->value.as<IntImm>()->value;
            bool new_load = new_index < 0 ? false : true;

            // check wrapped attr stmt in multi-cast 
            if (map[new_target].size() > 0) {
              // CHECK(!new_load) << "only support multiple writing in nest attrs";
              for (auto& v : map[new_target])
                if (v.data_load) {
                  LOG(WARNING) << "joining target tensor " << new_target;
                }
            }

            if (new_index < 0) new_index = -1 * new_index;
            map[new_target].push_back({new_index, new_load});
          }
          body = attr->body;
        }

        // mutate inner stmt & expr
        auto range_ = CollectIterRange(body);
        for (auto& kv : map) {
          Array<Expr> shape = shape_[target];
          Type dtype = dtype_[target];

          // p2p data movement 
          if (kv.second.size() == 1) {
            target = kv.first;
            data_load = kv.second[0].data_load;
            index = kv.second[0].index;
            std::string name = target + ".pipe" + std::to_string(index);

            if (data_load) {
              CHECK(channel_map_.count(name))
                << "cannot find channel buffer " << name;
              auto channel_buf = VarExpr(channel_map_[name].node_); 
              LoadToStreamExprConverter mutator(
                  target, StreamType::FIFO, 
                  channel_buf, 1, index, shape, range_);
              return mutator.Mutate(body);

            } else {
              auto channel_buf = VarExpr(name); 
              channel_map_[name] = channel_buf;

              StoreToStreamStmtConverter mutator(
                  target, StreamType::FIFO,
                  channel_buf, 1, index, shape, range_); 
              Stmt stmt = mutator.Mutate(body);

              stmt = Allocate::make(
                         VarExpr(channel_buf.node_), dtype, shape,
                         make_const(Bool(dtype.lanes()), true), stmt);
              stmt = AttrStmt::make(
                         VarExpr(channel_buf.node_), 
                         attr::storage_scope, 
                         StringImm::make("global"), stmt);
              return stmt;
            }

          } else { 
            target = kv.first;
            std::vector<VarExpr> channels;

            // create channel buffers
            size_t load_count = 0; 
            for (auto& v : kv.second) {
              if (v.data_load) load_count += 1;
              std::string name = target + ".pipe" + std::to_string(v.index);
              auto channel_buf = VarExpr(name); 
              channel_map_[name] = channel_buf;
              channels.push_back(channel_buf);
            }

            Stmt stmt;
            // multi-casting data
            if (load_count == 0) {
              MultiCastMutator mutator(target, channels, dtype);
              stmt = mutator.Mutate(body);
            // multi-loading data
            } else if (load_count == kv.second.size()){
              MultiLoadMutator mutator(target, channels, dtype);
              stmt = mutator.Mutate(body);
            }

            // allocate channel buffers
            CHECK(stmt.defined());
            for (auto& channel : channels) {
              stmt = Allocate::make(
                         VarExpr(channel.node_), dtype, shape,
                         make_const(Bool(dtype.lanes()), true), stmt);
              stmt = AttrStmt::make(
                         VarExpr(channel.node_), 
                         attr::storage_scope, 
                         StringImm::make("global"), stmt);
            }
            return stmt;
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt SplitScope(Stmt stmt) {
    Stmt s = Mutate(stmt);
    for (auto& k : kernel_defs_) 
      s = Block::make(k, s);
    return RemoveNoOp(s);
  }

 private:
  struct StreamInfo {
    int  index;
    bool data_load;
  };
  std::vector<Array<Var>>& undef_vars;
  std::unordered_map<std::string, Array<Expr>>& shape_;
  std::unordered_map<std::string, Type>& dtype_;
  std::unordered_map<std::string, VarExpr> channel_map_;
  std::vector<Stmt> kernel_defs_;
  bool xcel_scope{false};
  int scope_counter{0};
};

// 1. add annotation to kernel def node 
// 2. mutate the producer marked with .new 
// 3. remove defined but unused vars
class KernelAnnotator final : public IRMutator {
 public:
  KernelAnnotator(
    std::unordered_map<std::string, std::unordered_set<int>> map,
    std::unordered_map<std::string, Array<Expr>> mem_ports, 
    std::unordered_set<const Variable*>& unused_vars) :
    arg_scope_map_(map), mem_ports_(mem_ports), unused_vars_(unused_vars) {} 

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    std::string target_name = op->buffer_var.get()->name_hint;
    if (target_name != "_top") {
      if (target_name == "test" || unused_vars_.count(op->buffer_var.get())) {
        LOG(INFO) << "Removed unused var " << target_name;
        return this->Mutate(op->body);
      }
    }
    return stmt;
  }

  Stmt Mutate_(const KernelDef *op, const Stmt& s) final {
    Stmt body = this->Mutate(op->body);
    Array<Array<Expr>> channels = op->channels;

    // insert annotation for top function 
    if (op->name == "test") {
      int count = 0;
      for (auto& arg : op->args) {
        auto name = arg->name_hint;
        // skip inner loop movement case 
        if (!mem_ports_.count(name)) {
          LOG(INFO) << "device function within loop or zerocopy mode";
          break;
        }
        auto dev_port = mem_ports_[name];
        CHECK(dev_port.size() == 4);
        auto direction = dev_port[3];
        // pos, channel index, depth, is_sedner, dev_type, mem_port
        Array<Expr> info = {
            count, /*arg position index*/ 
            -1,    /*arg streaming channel index*/ 
            -1,    /*streaming channel depth*/ 
            dev_port[3], /*if it is the producer*/ 
            dev_port[0], /*memory type*/ 
            dev_port[1], /*memory channel port*/
            dev_port[2], /*stream type*/
        };
        count = count + 1;
        channels.push_back(info);
      }
      return KernelDef::make(
                 op->args, op->arg_shapes, op->arg_types, 
                 op->arg_tensors, body, op->ret_void, 
                 op->ret_type, op->name, channels);
    }

    // mutate kernel def body 
    if (channels.size() > 0) {
      for (size_t i = 0; i < channels.size(); i++) {
        auto info = channels[i];
        CHECK(info.size() == 6);
        auto pos = info[0].as<IntImm>()->value;
        auto channel = info[1].as<IntImm>()->value;
        auto depth = info[2].as<IntImm>()->value;
        auto is_sender = info[3].as<IntImm>()->value;

        // create shared channel buffer 
        VarExpr channel_buf;
        if (channel_map_.count(channel)) {
          channel_buf = VarExpr(channel_map_[channel].node_);
        } else {
          channel_buf = VarExpr("c_buf_" + 
                            std::to_string(channel));
          channel_map_[channel] = channel_buf;
        }
        VarExpr target = VarExpr(op->args[pos].node_);
        auto shape = op->arg_shapes[pos];
          
        body = KernelRebuild(channel_buf, depth, channel, 
                   is_sender, target, shape, body);
      }
    }

    if (arg_scope_map_.count(op->name)) {
      auto set = arg_scope_map_[op->name];

      // insert annotation (pos : index = -1) indicate global
      for (size_t i = 0; i < op->args.size(); i++) {
        if (set.find(i) != set.end()) {
          // position, channel index and depth
          Array<Expr> info_new;
          info_new.push_back(IntImm::make(Int(32), i));
          info_new.push_back(IntImm::make(Int(32), -1));
          info_new.push_back(IntImm::make(Int(32), -1));
          info_new.push_back(IntImm::make(Int(32), -1));
          info_new.push_back(IntImm::make(Int(32), -1));
          info_new.push_back(IntImm::make(Int(32), -1));
          channels.push_back(info_new);
        }
      }
    }
    return KernelDef::make(
               op->args, op->arg_shapes, op->arg_types, 
               op->arg_tensors, body, op->ret_void, 
               op->ret_type, op->name, channels);
  }

  // attach atributes to kernel function calls 
  Stmt Mutate_(const KernelStmt* op, const Stmt& s) final {
    if (op->name == "test") {
      int count = 0;
      Array<Expr> keys, values;
      for (auto& arg : op->args) {
        auto name = arg.as<Variable>()->name_hint;
        // skip inner loop movement case 
        if (!mem_ports_.count(name)) {
          LOG(INFO) << "device function within loop or zerocopy mode";
          break;
        }
        auto dev_port = mem_ports_[name];
        CHECK(dev_port.size() == 4);
        // pos, channel index, depth, is_sedner, dev_type, mem_port
        keys.push_back(StringImm::make("pos"));
        values.push_back(IntImm::make(Int(32), count));

        keys.push_back(StringImm::make("mem"));
        values.push_back(dev_port[0]);
        keys.push_back(StringImm::make("port"));
        values.push_back(dev_port[1]);
        keys.push_back(StringImm::make("stream_type"));
        values.push_back(dev_port[2]);
        keys.push_back(StringImm::make("direction"));
        values.push_back(dev_port[3]);

        count = count + 1;
      }
      return KernelStmt::make(op->args, op->name, keys, values);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<std::string, 
      std::unordered_set<int>> arg_scope_map_;
  std::unordered_map<int, VarExpr> channel_map_; 
  std::unordered_map<std::string, Array<Expr>> mem_ports_;
  std::unordered_set<const Variable*>& unused_vars_;

  // mutate kernel def body
  Stmt KernelRebuild(const VarExpr& channel_buf,
                     const int depth,
                     const int index,
                     const int is_sender,
                     const VarExpr& target_buf,
                     const Array<Expr> shape,
                     const Stmt& body) { 
    
    auto c_name = channel_buf.get()->name_hint;
    auto range_ = CollectIterRange(body);
    AccessCollector ac(target_buf, shape, range_, c_name); 
    ac.Mutate(body); 

    Stmt stmt;
    std::string target = target_buf.get()->name_hint;

    // self feedback loop
    if (is_sender == -1) {

    // sender mutate target store
    } else if (is_sender == 1) {
      if (ac.reg_store && ac.store_num == 1) {
        StoreToStreamStmtConverter mutator(
            target, StreamType::FIFO,
            channel_buf, depth, index, shape, range_); 
        stmt = mutator.Mutate(body);

      } else if (ac.store_num > 0) {
        if (!ac.reg_store)
          LOG(CLEAN) << "irregular \"" << target
                     << "\" access found; "
                     << "create reuse local buffer";
        if (ac.store_num > 1)
          LOG(CLEAN) << "multiple \"" << target
                     << "\" store found; "
                     << "create reuse local buffer";

        CHECK(ac.store_var.as<Variable>()) << "not a variable";
        VarExpr buf_var(ac.store_var.node_);
        stmt = BufferInserter(
                   body, shape, buf_var, channel_buf, false,
                   StreamType::FIFO, depth);
      } else {
        LOG(FATAL) << "target variable " 
                   << target << " not found; "
                   << "schedule does not apply";
      }

    // receiver mutate target load 
    } else if (is_sender == 0) {

      if (ac.reg_load && ac.load_num == 1) {
        LoadToStreamExprConverter mutator(
            target, StreamType::FIFO, 
            channel_buf, depth, index, shape, range_);
        stmt = mutator.Mutate(body);

      } else if (ac.load_num > 0) {
        if (!ac.reg_load)
          LOG(CLEAN) << "irregular \"" << target
                     << "\" access found; "
                     << "create reuse local buffer";
        if (ac.load_num > 1)
          LOG(CLEAN) << "multiple \"" << target
                     << "\" store found; "
                     << "create reuse local buffer";
        CHECK(ac.load_var.as<Variable>()) << "not a variable";
        VarExpr buf_var(ac.load_var.node_);
        stmt = BufferInserter(
                   body, shape, buf_var, channel_buf, true,
                   StreamType::FIFO, depth);
      } else {
        LOG(FATAL) << "target variable " 
                   << target << " not found; "
                   << "schedule does not apply";
      }
    }

    // create channel buffer
    if (not ac.buf_alloc) {
      auto dtype = channel_buf->type;
      stmt = Allocate::make(
                 VarExpr(channel_buf.node_), dtype, shape,
                 make_const(Bool(dtype.lanes()), true), stmt);
      stmt = AttrStmt::make(
                 VarExpr(channel_buf.node_), 
                 attr::storage_scope, 
                 StringImm::make("local"), stmt);
    }

    CHECK(stmt.defined());
    return stmt;
  }
};

// replace the mismatched buffers
class BufferReplacer final : public IRMutator {
 public:
  BufferReplacer(
      std::unordered_map<std::string, Expr>& bind_buffer_map,
      Array<Var>& undefined_vars) 
  : bind_buffer_map_(bind_buffer_map), 
    undefined_vars_(undefined_vars) {}

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    auto name = op->buffer_var->name_hint;
    CHECK(bind_buffer_map_.count(name)) << name;
    CHECK(bind_buffer_map_[name].get() == op->buffer_var.get());
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    return stmt;
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    auto name = op->buffer_var->name_hint;
    CHECK(bind_buffer_map_.count(name)) << name;
    if (bind_buffer_map_[name].get() != op->buffer_var.get()) {
      Expr index = op->index;
      Expr value = this->Mutate(op->value);
      auto new_buf = VarExpr(bind_buffer_map_[name].node_);
      CHECK(bind_buffer_map_[name].get() == new_buf.get());
      return Store::make(new_buf, value, index, op->predicate);
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    return stmt;
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    auto name = op->buffer_var->name_hint;
    CHECK(bind_buffer_map_.count(name)) << name;
    if (bind_buffer_map_[name].get() != op->buffer_var.get()) {
      Expr index = op->index;
      auto new_buf = VarExpr(bind_buffer_map_[name].node_);
      CHECK(bind_buffer_map_[name].get() == new_buf.get());
      return Load::make(op->type, new_buf, index, op->predicate);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Partition* op, const Stmt& s) {
    auto name = op->buffer_var->name_hint;
    CHECK(bind_buffer_map_.count(name)) << name;
    if (bind_buffer_map_[name].get() != op->buffer_var.get()) {
      auto new_buf = VarExpr(bind_buffer_map_[name].node_);
      return Partition::make(new_buf, op->dim, op->factor, op->partition_type);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<std::string, Expr>& bind_buffer_map_;
  Array<Var>& undefined_vars_;
};

Stmt InferStream(Stmt stmt, Array<NodeRef> api_args) {

  StreamAnalyzer analyzer(api_args);
  stmt = analyzer.Mutate(stmt);
  // FIXME: var buffer binding error 
  if (analyzer.undefined_.size() > 0 ) {
    stmt = BufferReplacer(analyzer.bind_buffer_map_,
            analyzer.undefined_).Mutate(stmt); 
  }

  StreamMutator mutator;
  stmt = mutator.Mutate(stmt); 

  // update timestep scheduling 
  InfoUpdater updater(
      /*vector of {groupId : name}*/mutator.timestep,
      /*streaming connectivity*/mutator.kernel_grp_id,
      /*connection group id*/mutator.kernel_idx_map,
      /*streamed buffer mark*/mutator.marked_buffer);
  stmt = updater.Mutate(stmt);

  // analyze and mutate attr scope decorated stmts
  stmt = StmtGrpReplacer(analyzer.streaming_vars,
                         analyzer.shape_,
                         analyzer.dtype_).SplitScope(stmt);

  // mark kernel def with storage scope
  std::unordered_set<const Variable*> unused_vars; // = UnusedVars(stmt);
  stmt = KernelAnnotator(analyzer.kernel_arg_scope_,
                         analyzer.mem_ports, unused_vars).Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace TVM
