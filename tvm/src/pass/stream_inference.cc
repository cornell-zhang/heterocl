/*!
 *  Copyright (c) 2019 by Contributors
 * \file remove_no_op.cc
 * \brief Remove no op from the stmt
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
 *   1. Variable use and definition 
 *   2. Streaming arguments in kernel function
 *   3. Streaming data bitwidth
 *
 * and add information into IR nodes
 *
 * */
class StreamUseDefAnalysis final : public IRVisitor {
 public:
  Array<Var> host_undefined_;
  std::unordered_map<const Variable*, int> host_use_count_;
  std::unordered_map<const Variable*, int> host_def_count_;
  StreamUseDefAnalysis(std::string initial_scope)
    : scope_(initial_scope) {};

  void Visit_(const AttrStmt* op) {
    if (op->attr_key == attr::device_scope) { 
      if (op->value.as<StringImm>()->value != scope_)
        switch_on = true;
      else switch_on = false;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Allocate* op) final {
    this->HandleDef(op->buffer_var.get());
    IRVisitor::Visit_(op);
  }

  void Visit_(const Load *op) {
    this->HandleUse(op->buffer_var);
    IRVisitor::Visit_(op);
  }

  void Visit_(const KernelDef *op) {
    // check the kernel channels 
    CHECK(op->channels.size() % 2 == 0) 
      << "wrong index number in channels";
    for (size_t i = 0; i < op->channels.size(); i+=2) {
      auto pos = op->channels[i].as<IntImm>()->value;
      auto idx = op->channels[i+1].as<IntImm>()->value;
      kernel_arg_map[op->name].push_back(pos);
      kernel_arg_map[op->name].push_back(idx);
    }
    IRVisitor::Visit_(op);
  }

  // insert index into kernel stmt 
  void Visit_(const KernelStmt *op) {
    IRVisitor::Visit_(op);
  }

  // check the kernel channels 
  void Visit_(const KernelExpr *op) {
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store* op) {
    if (auto val = op->value.as<StreamExpr>()) {
      this->HandleDef(op->buffer_var.get());
    }
    this->HandleUse(op->buffer_var);
    IRVisitor::Visit_(op);
  }

  /* count variable usage on its scope */
  void HandleUse(const Expr& v) {
    CHECK(v.as<Variable>());
    Var var(v.node_);
    auto it = host_use_count_.find(var.get());
    if (!switch_on) { // def on host scope 
      if (it != host_use_count_.end()) {
        if (it->second >= 0) {
          ++it->second;
        }
      } else {
        if (!stream_table_.count(var.get())) {
          host_undefined_.push_back(var);
          host_use_count_[var.get()] = -1;
        }
      }
    }
  }
  /* register variable on its scope */
  void HandleDef(const Variable* v) {
    if (!switch_on) { // def on host scope 
      CHECK(!host_def_count_.count(v))
          << "variable " << v->name_hint
          << " has already been defined, the Stmt is not SSA";
      CHECK(!host_use_count_.count(v))
          << "variable " << v->name_hint
          << " has been used before definition!";
      host_use_count_[v] = 0;
      host_def_count_[v] = 1;
    }
  }
 private: 
  std::unordered_map<const Variable*, bool> stream_table_;
  std::unordered_map<std::string, std::vector<int>> kernel_arg_map;
  std::string scope_;
  bool switch_on{true};
};


class StreamMutator : public IRMutator {
 public:
  explicit StreamMutator(int bus_bandwidth) {
    bus_bandwidth_ = bus_bandwidth;
  }
  // move device attr to allocate level
  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    // if (op->attr_key == attr::device_scope)
    //   return stmt.as<AttrStmt>()->body;
    return stmt;
  }

  // Stmt Mutate_(const For* op, const Stmt& s) final {
  //   Stmt stmt = IRMutator::Mutate_(op, s);
  //   op = stmt.as<For>();
  //   auto extent = op->extent.as<IntImm>()->value;
  //   auto min = op->min.as<IntImm>()->value;
  //   // mutate sender: split and block inner loop
  //   if (auto stream_op = op->body.as<StreamStmt>()) {
  //     if (extent - min > bus_bandwidth_) {
  //       // LOG(WARNING) << "large";
  //     } else {
  //     }
  //   // mutate receiver : (StreamExpr + For(Store = GetSlice))
  //   } else if (auto store_op = op->body.as<Store>()) {
  //     if (store_op->value.as<StreamExpr>() == nullptr) return stmt;
  //     if (extent - min > bus_bandwidth_) {
  //       // LOG(WARNING) << "large";
  //     } else {
  //       return stmt;
  //       // allocate intermediate buffer
  //       VarExpr new_var(store_op->buffer_var.get()->name_hint + "_save");
  //       Expr new_load = Load::make(store_op->buffer_var.type(), new_var, 0, const_true());
  //       Stmt new_store = Store::make(store_op->buffer_var, new_load,
  //                                    store_op->index, store_op->predicate);
  //       Stmt new_for = For::make(op->loop_var, op->min, op->extent, op->for_type,
  //                                op->device_api, new_store);
  //       // save stream data into intermediate buffer
  //       Stmt read_in = Store::make(new_var, store_op->value, 
  //                                  Expr(0), const_true());
  //       // allocate intermediate buffer
  //       return Allocate::make(new_var, 
  //                             store_op->value.type(),
  //                             {make_const(Int(bus_bandwidth_), 1)}, 
  //                             const_true(), Block::make(read_in, new_for));
  //     }
  //   }
  //   return stmt;
  // }

  Stmt Mutate_(const KernelDef *op, const Stmt& s) final {
    // check the kernel channels 
    CHECK(op->channels.size() % 2 == 0) 
      << "wrong index number in channels";
    for (size_t i = 0; i < op->channels.size(); i+=2) {
      auto pos = op->channels[i].as<IntImm>()->value;
      auto idx = op->channels[i+1].as<IntImm>()->value;
      kernel_arg_map[op->name].push_back(pos);
      kernel_arg_map[op->name].push_back(idx);
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    return stmt;
  }

  // insert index into kernel stmt 
  Stmt Mutate_(const KernelStmt *op, const Stmt& s) {
    auto vector = kernel_arg_map[op->name];
    if (vector.size() > 0) {
      Array<Expr> keys, values;
      for (size_t i = 0; i < vector.size(); i++) {
        if (i % 2 == 0) { // create position index
          keys.push_back(StringImm::make("pos"));
          values.push_back(IntImm::make(Int(32), vector[i]));
        } else { // create entry for channel index
          keys.push_back(StringImm::make("index"));
          values.push_back(IntImm::make(Int(32), vector[i]));
        }
      } // return new kernel stmt
      // return KernelStmt::make()
      Stmt stmt = IRMutator::Mutate_(op, s);
      return stmt;
    } else { // return original stmt
      Stmt stmt = IRMutator::Mutate_(op, s);
      return stmt;
    }
  }

  // insert index into kernel stmt 
  Expr Mutate_(const KernelExpr *op, const Expr& e) {
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const StreamStmt* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<StreamStmt>();
    const Variable* v = op->buffer_var.get();
    stream_type_map_[v] = op->buffer_var.type();
    return stmt;
  }

  Expr Mutate_(const StreamExpr* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<StreamExpr>();
    const Variable* v = op->buffer_var.get();
    stream_type_map_[v] = op->buffer_var.type();
    return expr;
  }
 private:
  int bus_bandwidth_;
  bool is_host_{true}; 
  std::unordered_map<const Variable*, Type> stream_type_map_;
  std::unordered_map<std::string, std::vector<int>> kernel_arg_map;
};

Stmt InferStream(Stmt stmt, 
                 int bus_bandwidth) {
  StreamUseDefAnalysis visitor("cpu");
  visitor.Visit(stmt);
  return StreamMutator(bus_bandwidth).Mutate(stmt); 
}

}  // namespace ir
}  // namespace TVM
