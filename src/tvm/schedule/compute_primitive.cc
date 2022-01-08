/*!
 *  Copyright (c) 2019 by Contributors
 * \file compute_primitive.cc
 */
#include "compute_primitive.h"
#include <arithmetic/Substitute.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule.h>
#include <unordered_set>
#include "../op/op_util.h"
#include "./graph.h"

namespace TVM {

using namespace ir;

namespace {

class LoopSplitter final : public IRMutator {
 public:
  LoopSplitter(const IterVar& parent, const Expr factor, const Expr nparts,
               const IterVar& outer, const IterVar& inner,
               std::unordered_map<const Variable*, Expr>& sub)
      : parent_(parent),
        factor_(factor),
        nparts_(nparts),
        outer_(outer),
        inner_(inner),
        sub_(sub) {}

  Stmt Mutate(Stmt stmt) final {
    if (const For* op = stmt.as<For>()) {
      if (op->loop_var.get() == parent_->var.get()) {
        Expr recovered_iv = inner_->var + outer_->var * factor_;
        sub_[op->loop_var.get()] = recovered_iv;
        condition_ = LT::make(recovered_iv, parent_->dom->extent);
        // check whether we should insert condition statement
        if (!Equal(Simplify(parent_->dom->extent % factor_), 0)) insert_ = true;
        const AttrStmt* attr_stmt = op->body.as<AttrStmt>();
        Stmt body = attr_stmt->body;
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
        Stmt inner_attr =
            AttrStmt::make(inner_, attr::loop_scope, inner_->var, body);
        Stmt inner_for = For::make(
            inner_->var, inner_->dom->min, inner_->dom->extent, op->for_type,
            op->device_api, inner_attr, op->annotate_keys, op->annotate_values);
        Stmt outer_attr =
            AttrStmt::make(outer_, attr::loop_scope, outer_->var, inner_for);
        Stmt outer_for = For::make(
            outer_->var, outer_->dom->min, outer_->dom->extent, op->for_type,
            op->device_api, outer_attr, op->annotate_keys, op->annotate_values);
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
                           op->device_api, body, op->annotate_keys,
                           op->annotate_values);
        }
        // otherwise return the updated body
        body = AttrStmt::make(attr_stmt->node, attr_stmt->attr_key,
                              attr_stmt->value, body);
        return For::make(op->loop_var, op->min, op->extent, op->for_type,
                         op->device_api, body, op->annotate_keys,
                         op->annotate_values);
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
  LoopFuser(const IterVar& outer, const IterVar& inner, const IterVar& fused,
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
        Stmt body =
            AttrStmt::make(fused_, attr::loop_scope, fused_->var, s->body);
        return For::make(fused_->var, min, extent, op->for_type, op->device_api,
                         body, op->annotate_keys, op->annotate_values);
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
      if (order_[i]->var.get() == var.get()) return i;
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
          case kUnrolled:
            for_type = ForType::Unrolled;
            break;
          case kVectorized:
            for_type = ForType::Vectorized;
            break;
          case kParallelized:
            for_type = ForType::Parallel;
            break;
          case kPipelined:
            for_type = ForType::Pipelined;
            break;
          case kDataPar:
            break;
          case kTensorized:
            break;
          default:
            LOG(FATAL) << "Unknown iter type" << node_->iter_type;
        }
        auto new_keys = node_->for_loop_annotate_keys;
        auto new_values = node_->for_loop_annotate_values;
        int size = new_keys.size();
        for (int i = 0; i < size; i++) {
          keys.push_back(new_keys[i]);
          values.push_back(new_values[i]);
        }
        return For::make(var_->var, op->min, op->extent, for_type,
                         op->device_api, op->body, keys, values);
      }
      return IRMutator::Mutate(stmt);
    }
    return IRMutator::Mutate(stmt);
  }

 private:
  const IterVar& var_;
  const IterVarAttrNode* node_;
};

class ComputeAtProducerExtracter : public IRMutator {
 public:
  ComputeAtProducerExtracter(const size_t& level, const IterVar& var,
                             const std::vector<VarExpr>& consumer_axes,
                             std::vector<VarExpr>& producer_axes,
                             const std::vector<Expr>& consumer_bound,
                             std::unordered_map<const Variable*, Expr>& sub)
      : level_(level),
        var_(var),
        consumer_axes_(consumer_axes),
        producer_axes_(producer_axes),
        consumer_bound_(consumer_bound),
        sub_(sub) {}

  Stmt Mutate(Stmt stmt) final {
    if (const For* op = stmt.as<For>()) {
      producer_axes_.push_back(op->loop_var);
      sub_[op->loop_var.get()] = consumer_axes_[counter_];
      counter_ += 1;
      const AttrStmt* attr_stmt = op->body.as<AttrStmt>();
      Stmt body;
      if (counter_ == level_) {
        body = attr_stmt->body;
      } else {
        body = this->Mutate(attr_stmt->body);
      }
      // if the consumer bound is greater than the producer bound
      if (is_one(Simplify(op->extent < consumer_bound_[counter_ - 1])))
        body =
            IfThenElse::make(consumer_axes_[counter_ - 1] < op->extent, body);
      counter_ -= 1;
      return body;
    } else {
      return IRMutator::Mutate(stmt);
    }
  }

 private:
  const size_t& level_;
  const IterVar& var_;
  const std::vector<VarExpr>& consumer_axes_;
  std::vector<VarExpr>& producer_axes_;
  const std::vector<Expr>& consumer_bound_;
  std::unordered_map<const Variable*, Expr>& sub_;
  size_t counter_{0};
};

class ProducerReplacer final : public IRMutator {
 public:
  ProducerReplacer(const Variable* target, const Array<Expr>& target_shape,
                   const Array<Expr>& reuse_shape,
                   const std::vector<VarExpr>& old_axes,
                   const std::vector<VarExpr>& new_axes,
                   std::unordered_map<const Variable*, Expr>& range)
      : target_(target),
        target_shape_(target_shape),
        reuse_shape_(reuse_shape),
        old_axes_(old_axes),
        new_axes_(new_axes),
        range_(range) {
    for (size_t i = 0; i < new_axes.size(); i++) {
      if (new_axes[i].defined()) {
        new_axis_subst_[old_axes[i].get()] = old_axes[i] + new_axes[i];
        old_axis_subst_[old_axes[i].get()] = new_axes[i];
      } else {
        old_axis_subst_[old_axes[i].get()] = 0;
      }
    }
  }

  Expr Mutate_(const Variable* op, const Expr& e) {
    auto it = new_axis_subst_.find(op);
    if (it != new_axis_subst_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    Expr index = op->index;
    Expr value = this->Mutate(op->value);
    if (op->buffer_var.get() == target_) {
      std::vector<Expr> new_indices =
          ExtractIndices(index, target_shape_, range_);
      index = FlattenIndices(new_indices, reuse_shape_);
      index = Simplify(substitute(old_axis_subst_, index));
      return Store::make(op->buffer_var, value, index, op->predicate);
    } else {
      return Store::make(op->buffer_var, value, index, op->predicate);
    }
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    Expr index = op->index;
    if (op->buffer_var.get() == target_) {
      std::vector<Expr> new_indices =
          ExtractIndices(index, target_shape_, range_);
      index = FlattenIndices(new_indices, reuse_shape_);
      index = Simplify(substitute(old_axis_subst_, index));
      return Load::make(op->type, op->buffer_var, index, op->predicate);
    } else {
      return Load::make(op->type, op->buffer_var, index, op->predicate);
    }
  }

 private:
  const Variable* target_;
  const Array<Expr>& target_shape_;
  const Array<Expr>& reuse_shape_;
  const std::vector<VarExpr>& old_axes_;
  const std::vector<VarExpr>& new_axes_;
  std::map<const Variable*, Expr> new_axis_subst_;
  std::map<const Variable*, Expr> old_axis_subst_;
  std::unordered_map<const Variable*, Expr>& range_;
};

class ConsumerReplacer final : public IRMutator {
 public:
  ConsumerReplacer(const Variable* target, const Array<Expr>& target_shape,
                   const Array<Expr>& reuse_shape,
                   const std::vector<VarExpr>& old_axes,
                   const std::vector<VarExpr>& new_axes,
                   std::unordered_map<const Variable*, Expr>& range)
      : target_(target),
        target_shape_(target_shape),
        reuse_shape_(reuse_shape),
        old_axes_(old_axes),
        new_axes_(new_axes),
        range_(range) {
    for (size_t i = 0; i < new_axes.size(); i++) {
      null_axis_subst_[old_axes[i].get()] = 0;
    }
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    Expr index = op->index;
    Expr value = this->Mutate(op->value);
    if (op->buffer_var.get() == target_) {
      std::vector<Expr> new_indices =
          ExtractIndices(index, target_shape_, range_);
      index = FlattenIndices(new_indices, reuse_shape_);
      index = Simplify(substitute(null_axis_subst_, index));
      return Store::make(op->buffer_var, value, index, op->predicate);
    } else {
      return Store::make(op->buffer_var, value, index, op->predicate);
    }
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    Expr index = op->index;
    if (op->buffer_var.get() == target_) {
      std::vector<Expr> new_indices =
          ExtractIndices(index, target_shape_, range_);
      index = FlattenIndices(new_indices, reuse_shape_);
      index = Simplify(substitute(null_axis_subst_, index));
      return Load::make(op->type, op->buffer_var, index, op->predicate);
    } else {
      return Load::make(op->type, op->buffer_var, index, op->predicate);
    }
  }

 private:
  const Variable* target_;
  const Array<Expr>& target_shape_;
  const Array<Expr>& reuse_shape_;
  const std::vector<VarExpr>& old_axes_;
  const std::vector<VarExpr>& new_axes_;
  std::map<const Variable*, Expr> null_axis_subst_;
  std::unordered_map<const Variable*, Expr>& range_;
};

class ComputeAtConsumerMerger : public IRMutator {
 public:
  ComputeAtConsumerMerger(Stmt& producer, Buffer& producer_buf,
                          const IterVar& var, size_t& attach_level,
                          std::unordered_map<const Variable*, Expr>& sub,
                          std::unordered_map<const Variable*, Expr>& range)
      : producer_(producer),
        producer_buf_(producer_buf),
        var_(var),
        attach_level_(attach_level),
        sub_(sub),
        range_(range) {}

  Stmt Mutate(Stmt stmt) final {
    if (const For* op = stmt.as<For>()) {
      attach_level_ += 1;
      const AttrStmt* attr_stmt = op->body.as<AttrStmt>();
      consumer_axes_.push_back(op->loop_var);
      consumer_bound_.push_back(op->extent);
      if (op->loop_var.get() == var_->var.get()) {
        // infer the reuse bound for the compute at producer
        // also update the shape of the buffer directly
        Array<Expr> reuse_shape = InferReuseBound(
            op->body, producer_buf_->data.get(), producer_buf_->shape, range_);
        // extract the producer body, count the attach level,
        // and subst producer axes with consumer axes
        std::vector<VarExpr> producer_axes;
        ComputeAtProducerExtracter mutator(attach_level_, var_, consumer_axes_,
                                           producer_axes, consumer_bound_,
                                           sub_);
        producer_ = mutator.Mutate(producer_);
        producer_ = op::Substitute(producer_, sub_);
        Stmt body = attr_stmt->body;
        if (reuse_shape.size() != 0) {
          producer_buf_->shape = reuse_shape;
          // create new axes if we have reuse in those dimensions
          std::vector<VarExpr> new_axes;
          for (size_t i = 0; i < attach_level_; i++) {
            // only create a new axis if the bound is not one
            if (!is_one(reuse_shape[i])) {
              new_axes.push_back(
                  VarExpr(producer_axes[i]->name_hint + ".compat"));
            } else {
              new_axes.push_back(VarExpr());
            }
          }
          // replace producer properly
          ProducerReplacer prod_mutator(producer_buf_->data.get(),
                                        producer_buf_->shape, reuse_shape,
                                        consumer_axes_, new_axes, range_);
          producer_ = prod_mutator.Mutate(producer_);
          // add the loops in a reversed order
          for (int i = new_axes.size() - 1; i >= 0; i--) {
            if (new_axes[i].defined()) {
              producer_ =
                  For::make(new_axes[i], 0, reuse_shape[i], ForType::Serial,
                            DeviceAPI::None, producer_);
            }
          }
          // replace consumer properly
          ConsumerReplacer cons_mutator(producer_buf_->data.get(),
                                        producer_buf_->shape, reuse_shape,
                                        consumer_axes_, new_axes, range_);
          body = cons_mutator.Mutate(body);
        }
        // add proper attr stmts
        body = AttrStmt::make(var_, attr::attach_scope, var_->var, body);
        body = AttrStmt::make(attr_stmt->node, attr_stmt->attr_key,
                              attr_stmt->value, body);
        return For::make(op->loop_var, op->min, op->extent, op->for_type,
                         op->device_api, body, op->annotate_keys,
                         op->annotate_values);
      } else {
        return IRMutator::Mutate(stmt);
      }
    } else {
      return IRMutator::Mutate(stmt);
    }
  }

 private:
  Stmt& producer_;
  Buffer& producer_buf_;
  const IterVar& var_;
  size_t& attach_level_;
  std::vector<VarExpr> consumer_axes_;
  std::vector<Expr> consumer_bound_;
  std::unordered_map<const Variable*, Expr>& sub_;
  std::unordered_map<const Variable*, Expr>& range_;
};
}  // end namespace

Stmt SplitLoop(Stmt& stmt, const IterVar& parent, const Expr factor,
               const Expr nparts, const IterVar& outer, const IterVar& inner,
               std::unordered_map<const Variable*, Expr>& sub) {
  LoopSplitter mutator(parent, factor, nparts, outer, inner, sub);
  stmt = mutator.Mutate(stmt);
  stmt = op::Substitute(stmt, sub);
  return stmt;
}

Stmt FuseLoop(Stmt& stmt, const IterVar& inner, const IterVar& outer,
              const IterVar& fused,
              std::unordered_map<const Variable*, Expr>& sub) {
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

Stmt UpdateIterVarAttr(Stmt& stmt, const IterVar& var,
                       const IterVarAttrNode* node) {
  IterVarAttrUpdater mutator(var, node);
  return mutator.Mutate(stmt);
}

Stmt PerformComputeAt(Stmt& producer, Stmt& consumer, Buffer& producer_buf,
                      const IterVar& var, size_t& attach_level,
                      std::unordered_map<const Variable*, Expr>& sub) {
  std::unordered_map<const Variable*, Expr> range = CollectIterRange(consumer);
  ComputeAtConsumerMerger mutator(producer, producer_buf, var, attach_level,
                                  sub, range);
  return mutator.Mutate(consumer);
}

}  // namespace TVM
