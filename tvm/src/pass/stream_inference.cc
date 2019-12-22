/*!
 *  Copyright (c) 2019 by Contributors
 * \file remove_no_op.cc
 * \brief Remove no op from the stmt
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <unordered_map>

namespace TVM {
namespace ir {

// use/def analysis to capture host xcel deps 
class StreamUseDefAnalysis : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt& s) final {
    if (op->attr_key == attr::device_scope) {
      if (op->value.as<StringImm>()->value == "fpga")
        host_scope_ = false;
      return IRMutator::Mutate_(op, s);
    } else {
      return IRMutator::Mutate_(op, s);
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

  Stmt Mutate_(const For *op, const Stmt& s) final {
    this->HandleDef(op->loop_var.get());
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt& s) final {
    this->HandleDef(op->buffer_var.get());
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt& s) final {
    this->HandleUse(op->buffer_var);
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const StreamStmt *op, const Stmt& s) final {
    this->HandleUse(op->buffer_var);
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Let *op, const Expr& e) final {
    this->HandleDef(op->var.get());
    Expr body = this->Mutate(op->body);
    Expr value = this->Mutate(op->value);
    if (body.same_as(op->body) &&
        value.same_as(op->value)) {
      return e;
    } else {
      return Let::make(op->var, value, body);
    }
  }

  Expr Mutate_(const Variable *op, const Expr& e) final {
    this->HandleUse(e);
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Load *op, const Expr& e) final {
    this->HandleUse(op->buffer_var);
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const StreamExpr *op, const Expr& e) final {
    this->HandleUse(op->buffer_var);
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const KernelDef *op, const Stmt& s) {
    for (auto arg : op->args) {
      this->HandleDef(arg.get());
    }
    Stmt body = this->Mutate(op->body);
    for (auto arg : op->args) {
      xcel_def_count_[arg.get()] = 0;
    }
    return s;
  }

  void HandleDef(const Variable* v) {
    if (host_scope_) {
      CHECK(!host_def_count_.count(v))
          << "variable " << v->name_hint
          << " has already been defined, the Stmt is not SSA";
      CHECK(!host_use_count_.count(v))
          << "variable " << v->name_hint
          << " has been used before definition!";
      host_use_count_[v] = 0;
      host_def_count_[v] = 1;
    } else {
      CHECK(!xcel_def_count_.count(v))
          << "variable " << v->name_hint
          << " has already been defined, the Stmt is not SSA";
      CHECK(!xcel_use_count_.count(v))
          << "variable " << v->name_hint
          << " has been used before definition!";
      xcel_use_count_[v] = 0;
      xcel_def_count_[v] = 1;
    }
  }

  void HandleUse(const Expr& v) {
    CHECK(v.as<Variable>());
    Var var(v.node_);
    if (host_scope_) {
      auto it = host_use_count_.find(var.get());
      if (it != host_use_count_.end()) {
        if (it->second >= 0) {
          ++it->second;
        }
      } else {
        host_undefined_.push_back(var);
        host_use_count_[var.get()] = -1;
      }
    } else {
      auto it = xcel_use_count_.find(var.get());
      if (it != xcel_use_count_.end()) {
        if (it->second >= 0) {
          ++it->second;
        }
      } else {
        xcel_undefined_.push_back(var);
        xcel_use_count_[var.get()] = -1;
      }
    }
  }

  bool host_scope_{true};
  Array<Var> host_undefined_;
  Array<Var> xcel_undefined_;
  std::unordered_map<const Variable*, int> host_use_count_;
  std::unordered_map<const Variable*, int> host_def_count_;
  std::unordered_map<const Variable*, int> xcel_use_count_;
  std::unordered_map<const Variable*, int> xcel_def_count_;
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

  Stmt Mutate_(const For* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<For>();
    auto extent = op->extent.as<IntImm>()->value;
    auto min = op->min.as<IntImm>()->value;
    // mutate sender: split and block inner loop
    if (auto stream_op = op->body.as<StreamStmt>()) {
      if (extent - min > bus_bandwidth_) {
        LOG(WARNING) << "large";
      } else {
      }
    // mutate receiver : (StreamExpr + For(Store = GetSlice))
    } else if (auto store_op = op->body.as<Store>()) {
      if (store_op->value.as<StreamExpr>() == nullptr) return stmt;
      if (extent - min > bus_bandwidth_) {
        LOG(WARNING) << "large";
      } else {
        return stmt;
        // allocate intermediate buffer
        VarExpr new_var(store_op->buffer_var.get()->name_hint + "_save");
        Expr new_load = Load::make(store_op->buffer_var.type(), new_var, 0, const_true());
        Stmt new_store = Store::make(store_op->buffer_var, new_load,
                                     store_op->index, store_op->predicate);
        Stmt new_for = For::make(op->loop_var, op->min, op->extent, op->for_type,
                                 op->device_api, new_store);
        // save stream data into intermediate buffer
        Stmt read_in = Store::make(new_var, store_op->value, 
                                   Expr(0), const_true());
        // allocate intermediate buffer
        return Allocate::make(new_var, 
                              store_op->value.type(),
                              {make_const(Int(bus_bandwidth_), 1)}, 
                              const_true(), Block::make(read_in, new_for));
      }
    }
    return stmt;
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
};

// Mark the statment scope of each stage.
class StreamInferer : public IRMutator {
 public:
  explicit StreamInferer(int bus_bandwidth) {
    bus_bandwidth_ = bus_bandwidth;
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    if (auto block = op->body.as<Block>()) {
      if (auto producer = block->first.as<ProducerConsumer>()){
        if (const AttrStmt* attr_stmt = producer->body.as<AttrStmt>()) {
          if (const AttrStmt* device_attr = attr_stmt->body.as<AttrStmt>()) {
            if (device_attr->attr_key == attr::device_scope) {
              // mutate allocate body
              StreamMutator mutator(bus_bandwidth_);
              // allocate stream for host 
              Stmt new_body = mutator.Mutate(op->body);
              Stmt new_stmt = Allocate::make(op->buffer_var,
                                             op->type,
                                             op->extents,
                                             op->condition,
                                             new_body);
              return AttrStmt::make(device_attr->node,
                                    attr::device_scope,
                                    device_attr->value,
                                    new_stmt);
            }
          }
        }
      }
    }
    return stmt;
  }

  // Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) final {
  //   Stmt stmt = IRMutator::Mutate_(op, s);
  //   op = stmt.as<ProducerConsumer>();
  //   return is_no_op(op->body) ? op->body : stmt;
  // }

  // Stmt Mutate_(const Store* op, const Stmt& s) final {
  //   Stmt stmt = IRMutator::Mutate_(op, s);
  //   op = stmt.as<Store>();
  //   auto it = var_remap_.find(op->buffer_var.get());
  //   if (it != var_remap_.end() &&
  //       !it->second.same_as(op->buffer_var)) {
  //     CHECK(it->second.as<Variable>());
  //     VarExpr buf_var(it->second.node_);
  //     if (has_stencil_) outputs_.insert(buf_var);
  //     return Store::make(buf_var, op->value, op->index, op->predicate);
  //   } else {
  //     return stmt;
  //   }
  // }

  // Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
  //   if (op->attr_key == attr::realize_scope) {
  //     storage_scope_[op->node.get()] = op->value.as<StringImm>()->value;
  //     return this->Mutate(op->body);
  //   } else if (op->attr_key == attr::double_buffer_scope) {
  //     Operation func(op->node.node_);
  //     Stmt body = Mutate(op->body);
  //     for (int i = 0; i < func->num_outputs(); ++i) {
  //       TensorKey key{func, i};
  //       auto it = buf_map_.find(key);
  //       CHECK(it != buf_map_.end())
  //           << "Cannot find allocated buffer for " << key.f;
  //       body = AttrStmt::make(
  //           it->second.buffer->data, op->attr_key, op->value, body);
  //     }
  //     return body;
  //   } else if (op->attr_key == attr::thread_extent) {
  //     IterVar iv(op->node.node_);
  //     ThreadScope ts = ThreadScope::make(iv->thread_tag);
  //     curr_thread_scope_.push_back(ts);
  //     Stmt stmt = IRMutator::Mutate_(op, s);
  //     curr_thread_scope_.pop_back();
  //     return stmt;
  //   } else if (op->attr_key == attr::buffer_bind_scope) {

  // Stmt Mutate_(const For* op, const Stmt& s) final {
  //   Stmt stmt = IRMutator::Mutate_(op, s);
  //   op = stmt.as<For>();
  //   return is_no_op(op->body) ? MakeEvaluate({op->min, op->extent}) : stmt;
  // }

 private:
  int bus_bandwidth_;
  Stmt MakeEvaluate(Expr value) {
    if (HasSideEffect(value)) {
      return Evaluate::make(value);
    } else {
      return Evaluate::make(0);
    }
  }
  Stmt MakeEvaluate(const Array<Expr>& values) {
    Stmt stmt;
    for (Expr e : values) {
      if (HasSideEffect(e)) {
        if (stmt.defined()) {
          stmt = Block::make(stmt, Evaluate::make(e));
        } else {
          stmt = Evaluate::make(e);
        }
      }
    }
    return stmt.defined() ? stmt : Evaluate::make(0);
  }
};

Stmt InferStream(Stmt stmt, 
                 int bus_bandwidth) {
  return StreamInferer(bus_bandwidth).Mutate(stmt); 
}

}  // namespace ir
}  // namespace TVM
