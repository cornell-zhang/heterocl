/*!
 *  Copyright (c) 2019 by Contributors
 * \file schedule_primitive.cc
 */
#include <tvm/schedule.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
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
          valid_ = true;
          Expr recovered_iv = outer_->var * factor_ + inner_->var;
          sub_[op->loop_var.get()] = recovered_iv;
          const AttrStmt* untouched = op->body.as<AttrStmt>();
          Expr condition = LT::make(recovered_iv, parent_->dom->extent);
          Stmt inner_if = IfThenElse::make(condition, untouched->body);
          Stmt inner_attr = AttrStmt::make(inner_, attr::loop_scope, inner_->var, inner_if);
          Stmt inner_for = For::make(inner_->var, inner_->dom->min, inner_->dom->extent,
                                     op->for_type, op->device_api, inner_attr,
                                     op->annotate_keys, op->annotate_values);
          Stmt outer_attr = AttrStmt::make(outer_, attr::loop_scope, inner_->var, inner_for);
          Stmt outer_for = For::make(outer_->var, outer_->dom->min, outer_->dom->extent,
                                     op->for_type, op->device_api, outer_attr,
                                     op->annotate_keys, op->annotate_values);
          return outer_for;
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
    const IterVar& parent_;
    const Expr factor_;
    const Expr nparts_;
    const IterVar& outer_;
    const IterVar& inner_;
    bool valid_{false};
    std::unordered_map<const Variable*, Expr>& sub_;
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

} // namespace tvm
