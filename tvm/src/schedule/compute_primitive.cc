/*!
 *  Copyright (c) 2019 by Contributors
 * \file compute_primitive.cc
 */
#include <tvm/schedule.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include "./graph.h"
#include "../op/op_util.h"

namespace tvm {

using namespace ir;

class LoopSplitter final : public IRMutator {
  public:
    LoopSplitter(const IterVar& parent,
                 const Expr factor,
                 const Expr nparts,
                 const IterVar& outer,
                 const IterVar& inner,
                 std::unordered_map<const Variable*, Expr>& sub)
      : parent_(parent), factor_(factor), nparts_(nparts), outer_(outer), inner_(inner), sub_(sub) {}

    Stmt Mutate(Stmt stmt) final {
      if (const For* op = stmt.as<For>()) {
        if (op->loop_var.get() == parent_->var.get()) {
          Expr recovered_iv = outer_->var * factor_ + inner_->var;
          sub_[op->loop_var.get()] = recovered_iv;
          condition_ = LT::make(recovered_iv, parent_->dom->extent);
          // check whether we should insert condition statement
          if (!Equal(Simplify(parent_->dom->extent % factor_), 0)) insert_ = true;
          const AttrStmt* attr_stmt = op->body.as<AttrStmt>();
          Stmt body = attr_stmt -> body;
          // if we need to insert a condition stmt
          if (insert_) {
            // check if we can move the stmt to lower loops
            if (body.as<For>()) body = this->Mutate(body);
            // check if we need to insert the stmt at the current level
            if (insert_) {
              insert_ = false;
              body = IfThenElse::make(condition_, body);
            }
          }
          Stmt inner_attr = AttrStmt::make(inner_, attr::loop_scope, inner_->var, body);
          Stmt inner_for = For::make(inner_->var, inner_->dom->min, inner_->dom->extent,
                                     op->for_type, op->device_api, inner_attr,
                                     op->annotate_keys, op->annotate_values);
          Stmt outer_attr = AttrStmt::make(outer_, attr::loop_scope, outer_->var, inner_for);
          Stmt outer_for = For::make(outer_->var, outer_->dom->min, outer_->dom->extent,
                                     op->for_type, op->device_api, outer_attr,
                                     op->annotate_keys, op->annotate_values);
          return outer_for;
        } else if (insert_) {
          // check if the condition can move here safely
          bool min_has_var = ExprUseVar(op->min, parent_->var);
          bool extent_has_var = ExprUseVar(op->extent, parent_->var);
          // do not insert here
          if (min_has_var || extent_has_var) return IRMutator::Mutate(stmt);
          // otherwise check if we can further push the condition downward
          const AttrStmt* attr_stmt = op->body.as<AttrStmt>();
          Stmt body = attr_stmt->body;
          // if there is a loop right below, check if we can push it
          if (body.as<For>()) body = this->Mutate(body);
          // finally we insert the condition if needed
          if (insert_) {
            insert_ = false;
            body = IfThenElse::make(condition_, body);
            body = AttrStmt::make(attr_stmt->node, attr_stmt->attr_key, 
                                  attr_stmt->value, body);
            return For::make(op->loop_var, op->min, op->extent, op->for_type,
                             op->device_api, body, op->annotate_keys, op->annotate_values);
          }
          // otherwise return the updated body
          return For::make(op->loop_var, op->min, op->extent, op->for_type,
                           op->device_api, body, op->annotate_keys, op->annotate_values);
        } else {
          return IRMutator::Mutate(stmt);
        }
      } else {
        return IRMutator::Mutate(stmt);
      }
    }

  private:
    const IterVar& parent_;
    const Expr factor_;
    const Expr nparts_;
    const IterVar& outer_;
    const IterVar& inner_;
    Expr condition_;
    bool insert_{false};
    std::unordered_map<const Variable*, Expr>& sub_;
};

class LoopFuser final : public IRMutator {
  public:
    LoopFuser(const IterVar& outer, 
              const IterVar& inner, 
              const IterVar& fused, 
              std::unordered_map<const Variable*, Expr>& sub)
      : inner_(inner), outer_(outer), fused_(fused), sub_(sub) {}

    Stmt Mutate(Stmt stmt) final {
      if (const For* op = stmt.as<For>()) {
        if (op->loop_var.get() == outer_->var.get()) {
          valid_ = true;
          sub_[op->loop_var.get()] = fused_->var / inner_->dom->extent;
          return this->Mutate(op->body);
        } else if (op->loop_var.get() == inner_->var.get()) {
          if (!valid_) LOG(FATAL) << "Cannot fuse " << outer_ << "with" << inner_;
          Expr min = inner_->dom->min + outer_->dom->min * inner_->dom->extent;
          Expr extent = inner_->dom->extent * outer_->dom->extent;
          sub_[op->loop_var.get()] = fused_->var % inner_->dom->extent;
          const AttrStmt* s = op->body.as<AttrStmt>();
          Stmt body = AttrStmt::make(fused_, attr::loop_scope, fused_->var, s->body);
          return For::make(fused_->var, min, extent, op->for_type, op->device_api, body,
                           op->annotate_keys, op->annotate_values);
        } else {
          valid_ = false;
          return IRMutator::Mutate(stmt);
        }
      } else if (const AttrStmt* op = stmt.as<AttrStmt>()) {
        if (valid_ && op->attr_key == attr::loop_scope) {
          return this->Mutate(op->body);
        }
        return IRMutator::Mutate(stmt);
      } else {
        valid_ = false;
        return IRMutator::Mutate(stmt);
      }
    }

  private:
    const IterVar& inner_;
    const IterVar& outer_;
    const IterVar& fused_;
    bool valid_{false};
    std::unordered_map<const Variable*, Expr>& sub_;
};

class LoopReorderer final : public IRMutator {
  public:
    LoopReorderer(const Array<IterVar>& order) : order_(order) {
      body_list = std::vector<Stmt>(order.size());
    }

    Stmt Mutate(Stmt stmt) final {
      if (const For* op = stmt.as<For>()) {
        int index = var_index_in_list(op->loop_var);
        if (index != -1) {
          body_list[index] = stmt;
          const AttrStmt* attr_stmt = op->body.as<AttrStmt>();
          counter += 1;
          // check valid reorder?
          Stmt body = this->Mutate(attr_stmt->body);
          counter -= 1;
          IterVar new_var = order_[counter];
          body = AttrStmt::make(new_var, attr::loop_scope, new_var->var, body);
          const For* new_for = body_list[counter].as<For>();
          return For::make(new_for->loop_var, new_for->min, new_for->extent,
                           new_for->for_type, new_for->device_api, body,
                           new_for->annotate_keys, new_for->annotate_values);
        }
        return IRMutator::Mutate(stmt);
      }
      return IRMutator::Mutate(stmt);
    }

  private:
    const Array<IterVar>& order_;
    int counter{0};
    std::vector<Stmt> body_list;

    int var_index_in_list(const VarExpr& var) {
      for (size_t i = 0; i < order_.size(); i++) {
        if (order_[i]->var.get() == var.get())
          return i;
      }
      return -1;
    }
};

class IterVarAttrUpdater final : public IRMutator {
  public:
    IterVarAttrUpdater(const IterVar& var, const IterVarAttrNode* node)
      : var_(var), node_(node) {}

    Stmt Mutate(Stmt stmt) final {
      if (const For* op = stmt.as<For>()) {
        ForType for_type = ForType::Serial;
        Array<Expr> keys = Array<Expr>(op->annotate_keys);
        Array<Expr> values = Array<Expr>(op->annotate_values);
        if (op->loop_var.get() == var_->var.get()) {
          switch (node_->iter_type) {
            case kUnrolled: for_type = ForType::Unrolled; break;
            case kVectorized: for_type = ForType::Vectorized; break;
            case kParallelized: for_type = ForType::Parallel; break;
            case kPipelined: for_type = ForType::Pipelined; break;
            case kDataPar: break;
            case kTensorized: break;
            default: LOG(FATAL) << "Unknown iter type" << node_->iter_type;
          }
          auto new_keys = node_->for_loop_annotate_keys;
          auto new_values = node_->for_loop_annotate_values;
          int size = new_keys.size();
          for (int i = 0; i < size; i++) {
            keys.push_back(new_keys[i]);
            values.push_back(new_values[i]);
          }
          return For::make(var_->var, op->min, op->extent,
                           for_type, op->device_api, op->body,
                           keys, values);
        }
        return IRMutator::Mutate(stmt);
      }
      return IRMutator::Mutate(stmt);
    }

  private:
    const IterVar& var_;
    const IterVarAttrNode* node_;
};

Stmt SplitLoop(Stmt& stmt,
               const IterVar& parent,
               const Expr factor,
               const Expr nparts,
               const IterVar& outer,
               const IterVar& inner) {
  std::unordered_map<const Variable*, Expr> sub;
  LoopSplitter mutator(parent, factor, nparts, outer, inner, sub);
  stmt = mutator.Mutate(stmt);
  stmt = op::Substitute(stmt, sub);
  return stmt;
}

Stmt FuseLoop(Stmt& stmt,
              const IterVar& inner,
              const IterVar& outer,
              const IterVar& fused) {
  std::unordered_map<const Variable*, Expr> sub;
  LoopFuser mutator(inner, outer, fused, sub);
  stmt = mutator.Mutate(stmt);
  stmt = op::Substitute(stmt, sub);
  return stmt;
}

Stmt ReorderLoop(Stmt& stmt, const Array<IterVar>& order) {
  LoopReorderer mutator(order);
  stmt = mutator.Mutate(stmt);
  return stmt;
}

Stmt UpdateIterVarAttr(Stmt& stmt,
                   const IterVar& var,
                   const IterVarAttrNode* node) {
  IterVarAttrUpdater mutator(var, node);
  return mutator.Mutate(stmt);
}

} // namespace tvm
