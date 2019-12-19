/*!
 *  Copyright (c) 2019 by Contributors
 * \file stream_inference.cc
 * \brief mutate ir for scheduling streaming ops
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <unordered_map>

namespace TVM {
namespace ir {

/*!
 * \brief An IRMutator to collect information 
 *
 * Collect streaming information:
 *   1. Streaming data access pattern consistency
 *      the access index of rd & wr must match to 
 *      avoid streaming channel overflow 
 *
 * and add information into IR nodes
 *
 * */
class StreamUseDefAnalysis final : public IRVisitor {
 public:
  StreamUseDefAnalysis(std::string initial_scope)
    : scope_(initial_scope) {};

  void Visit_(const AttrStmt* op) {
    IRVisitor::Visit_(op);
  }

  void Visit_(const Allocate* op) final {
    IRVisitor::Visit_(op);
  }

  // check load index in stream stmt
  void Visit_(const Load *op) {
    IRVisitor::Visit_(op);
  }

  void Visit_(const KernelDef *op) {
    CHECK(op->channels.size() % 2 == 0) 
      << "wrong index number in channels";
    IRVisitor::Visit_(op);
  }

  void Visit_(const KernelStmt *op) {
    IRVisitor::Visit_(op);
  }

  void Visit_(const KernelExpr *op) {
    IRVisitor::Visit_(op);
  }

  // check store index containing stream expr
  void Visit_(const Store* op) {
    IRVisitor::Visit_(op);
  }

  void Visit_(const For* op) {
    IRVisitor::Visit_(op);
  }

  void Visit_(const StreamStmt* op) {
    auto index = op->annotate_values[0];
    auto channel = op->annotate_values[1].as<IntImm>()->value;
    checkAccessPattern(channel, index);
    IRVisitor::Visit_(op);
  }

  void Visit_(const StreamExpr* op) {
    auto index = op->annotate_values[0];
    auto channel = op->annotate_values[1].as<IntImm>()->value;
    checkAccessPattern(channel, index);
    IRVisitor::Visit_(op);
  }

  // check access pattern consistency 
  void checkAccessPattern(int channel, Expr index) {
    auto it = index_map.find(channel);
    if (it != index_map.end()) {
    } else {
      index_map[channel] = index;
    }
  } 

  // map of channel num to access index
  std::unordered_map<int, Expr> index_map;
  // init device scope 
  std::string scope_;
};


class StreamMutator : public IRMutator {
 public:
  explicit StreamMutator(int bus_bandwidth) {
    bus_bandwidth_ = bus_bandwidth;
  }

  // remove unnecessary nested device scope attr
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

  // split loop if bitwidth larger than bus bandwidth 
  Stmt Mutate_(const For* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<For>();
    // get streaming stmt & expr info
    // StreamUseDefAnalysis visitor("cpu");
    // visitor.Visit(stmt);
    if (auto inner = op->body.as<StreamStmt>()) {
      auto extent = op->extent.as<IntImm>()->value;
      auto min = op->min.as<IntImm>()->value;
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
    return KernelStmt::make(op->args, op->name, keys, values);
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
    // check access pattern consistency  
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
      const std::unordered_map<std::string, int> kernel_index_map)
      : timestep_(timestep), connected_grp_(connected_grp),
        kernel_index_map_(kernel_index_map) {
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

  // remove unnecessary alloc for kernel 
  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    std::string name = op->buffer_var.get()->name_hint;
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    if (kernel_index_map_.count(name))
      return op->body;
    return stmt;
  }

 private:
  std::vector<std::unordered_map<int, std::string>>& timestep_;
  const std::vector<std::set<std::string>> connected_grp_;
  const std::unordered_map<std::string, int> kernel_index_map_;
  bool update_{false}; // schedule updating indicator
  std::unordered_map<std::string, int> changes_record;
};

Stmt InferStream(Stmt stmt, 
                 int bus_bandwidth) {
  StreamMutator mutator(bus_bandwidth);
  stmt = mutator.Mutate(stmt); 
  // update timestep scheduling 
  InfoUpdater updater(/*vector of {groupId:name}*/mutator.timestep,
                      /*streaming connectivity*/mutator.kernel_grp_id,
                      /*connection group id*/mutator.kernel_idx_map);
  return updater.Mutate(stmt);
}

}  // namespace ir
}  // namespace TVM
