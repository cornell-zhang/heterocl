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
    if (op->channels.size() > 0) {
      CHECK(op->channels.size() % 2 == 0) 
        << "wrong index number in channels";
      // insert (position, channel idx) into map
      for (size_t i = 0; i < op->channels.size(); i+=2) {
        auto pos = op->channels[i].as<IntImm>()->value;
        auto idx = op->channels[i+1].as<IntImm>()->value;
        kernel_arg_map[op->name].push_back(pos);
        kernel_arg_map[op->name].push_back(idx);
        kernel_channel_map[op->name].insert(idx);
      }
      // create entry for scc groups
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
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    return stmt;
  }

  // insert channel index & infer scheduling group 
  Stmt Mutate_(const KernelStmt *op, const Stmt& s) {
    auto vector = kernel_arg_map[op->name];
    if (vector.size() > 0) {
      CHECK(vector.size() % 2 == 0) << "wrong size";
      Array<Expr> keys, values;
      // push into thread group id & timestep
      auto group_id = getThreadGroup(op->name);
      auto time_step = getTimeStep(group_id);
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
      return KernelStmt::make(op->args, op->name, keys, values);
    } else { // return original stmt
      Stmt stmt = IRMutator::Mutate_(op, s);
      return stmt;
    }
  }

  // insert index into kernel stmt 
  Expr Mutate_(const KernelExpr *op, const Expr& e) {
    auto vector = kernel_arg_map[op->name];
    if (vector.size() > 0) {
      CHECK(vector.size() % 2 == 0) << "wrong size";
      Array<Expr> keys, values;
      // push into thread group id & timestep
      auto group_id = getThreadGroup(op->name);
      auto time_step = getTimeStep(group_id);
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
    } else { // return original expr
      Expr expr = IRMutator::Mutate_(op, e);
      return expr;
    }
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
    for (size_t i = 0; i < kernel_grp_id.size(); i++) {
      auto set = kernel_grp_id[i];
      if (set.find(name) != set.end()) { 
        if (idx_count.find(i) == idx_count.end())
          idx_count[i] = 1;
        else idx_count[i] += 1;
        num = idx_count[i]; 
        break;
      }
    }
    CHECK(num > -1) 
      << "not found group index";
    return num;
  }

  // greedily schedule timestep for kernel stmt / expr
  int getTimeStep(int group_id) {
    auto curr = timestep.size();
    if (curr == 0) {
      timestep.push_back({group_id});
      return 0;
    } else { // perform scheduling 
      for (size_t i = 0; i < curr; i++) {
        auto& set = timestep[i];
        if (set.find(group_id) == set.end() &&
            set.size() < thread_limit) {
          set.insert(group_id); 
          return i;
        }
      } // insert into next stage
      timestep.push_back({group_id});
      return curr;
    }
  }

  // time step vector for thread allocation 
  std::vector<std::unordered_set<int>> timestep;

 private:
  int bus_bandwidth_;
  // thread limit 
  size_t thread_limit{16};
  // map from kernel name to vector of (pos, channel) index pair 
  std::unordered_map<std::string, std::vector<int>> kernel_arg_map;
  // map from kernel name to connected channel index set
  std::unordered_map<std::string, std::unordered_set<int>> kernel_channel_map;
  // connected components group of kernels
  std::vector<std::set<std::string>> kernel_grp_id;
  // kernel_grp_id index map for each kernel
  std::unordered_map<std::string, int> kernel_idx_map;
  // connected group index to alloc num
  std::unordered_map<int, int> idx_count; 
};

class InfoUpdater final : public IRMutator {
 public:
  explicit InfoUpdater(
      const std::vector<std::unordered_set<int>> timestep)
      : timestep_(timestep) {}
  Stmt Mutate_(const KernelStmt* op, const Stmt& s) {
   auto keys = op->annotate_keys;
   auto values = op->annotate_values;
   for (size_t i = 0; i < op->annotate_keys.size(); i++) {
     if (op->annotate_keys[i].as<StringImm>()->value == "timestep") {
       auto num = timestep_[op->annotate_values[i].as<IntImm>()->value].size();
       keys.push_back(StringImm::make("thread_num"));
       values.push_back(IntImm::make(Int(32), num));
     }
   }
   return KernelStmt::make(op->args, op->name, keys, values);
  }
 private:
  const std::vector<std::unordered_set<int>> timestep_;
};

Stmt InferStream(Stmt stmt, 
                 int bus_bandwidth) {
  StreamMutator mutator(bus_bandwidth);
  stmt = mutator.Mutate(stmt); 
  InfoUpdater updater(mutator.timestep);
  return updater.Mutate(stmt);
}

}  // namespace ir
}  // namespace TVM
