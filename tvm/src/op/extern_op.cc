/*!
 *  Copyright (c) 2017 by Contributors
 * \brief External computation rule.
 * \file extern_op.cc
 */
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>
#include "./op_util.h"
#include "./extern_op.h"

#include <iostream>

namespace tvm {
using namespace ir;

namespace {
const Variable* GetBufferVar(const Stmt& stmt) {
  std::vector<const Variable*> buffer_vars;
  auto get_buffer_var = [&](const NodeRef& n) {
    if (const Store* e = n.as<Store>()) {
      buffer_vars.push_back(e->buffer_var.get());
    }
  };
  PostOrderVisit(stmt, get_buffer_var);
  return buffer_vars[0];
}

void UpdateVarsAfterAttach(const Stage& stage,
                           int axis_size, int attach_level,
                           std::unordered_map<const Variable*, Expr>& vars_delete_inner,
                           VarsDeleteOuterMap& vars_delete_outer,
                           std::unordered_map<const Variable*, Expr>& vars_sub) {
  Stmt stmt = stage->op.as<ExternOpNode>()->body;
  const Variable* buffer_var = GetBufferVar(stmt);
  // PostOrderVisit starts from the inner most loop
  int current_level = axis_size;
  auto fvisit = [&](const NodeRef& n) {
    if (const For* e = n.as<For>()) {
      if (current_level <= attach_level) {
        vars_delete_inner[e->loop_var.get()] = make_const(Int(32), 0);
        vars_delete_outer[buffer_var][stage->attach_stage->iter_var_exprs[current_level - 1].as<Variable>()] = make_const(Int(32), 0);
        vars_sub[e->loop_var.get()] = stage->attach_stage->iter_var_exprs[current_level - 1];
      }
      current_level--;
    }
  };
  PostOrderVisit(stmt, fvisit);
}
}  // namespace

std::unordered_map<const Variable*, Expr>
GetVarsDeleteInner(const Stage& stage, int axis_size, int attach_level) {
  std::unordered_map<const Variable*, Expr> vars_delete_inner;
  VarsDeleteOuterMap vars_delete_outer;
  std::unordered_map<const Variable*, Expr> vars_sub;
  UpdateVarsAfterAttach(stage, axis_size, attach_level,
                        vars_delete_inner, vars_delete_outer, vars_sub);
  return vars_delete_inner;
}

VarsDeleteOuterMap
GetVarsDeleteOuter(const Stage& stage, int axis_size, int attach_level) {
  std::unordered_map<const Variable*, Expr> vars_delete_inner;
  VarsDeleteOuterMap vars_delete_outer;
  std::unordered_map<const Variable*, Expr> vars_sub;
  UpdateVarsAfterAttach(stage, axis_size, attach_level,
                        vars_delete_inner, vars_delete_outer, vars_sub);
  return vars_delete_outer;
}

std::unordered_map<const Variable*, Expr>
GetVarsSub(const Stage& stage, int axis_size, int attach_level) {
  std::unordered_map<const Variable*, Expr> vars_delete_inner;
  VarsDeleteOuterMap vars_delete_outer;
  std::unordered_map<const Variable*, Expr> vars_sub;
  UpdateVarsAfterAttach(stage, axis_size, attach_level,
                        vars_delete_inner, vars_delete_outer, vars_sub);
  return vars_sub;
}

int CountAttachLevel(const Stage& stage) {
  int attach_level = 0;
  for (auto iv : stage->attach_stage->iter_var_exprs) {
    attach_level += 1;
    if (stage->attach_ivar->var.same_as(iv) ||
        stage->origin_attach_ivar->var.same_as(iv)) {
      break;
    }
  }
  int axis_size = stage->op.as<ExternOpNode>()->axis.size();
  attach_level = std::min(attach_level, axis_size);
  return attach_level;
}

// ExternOpNode
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ExternOpNode>([](const ExternOpNode *op, IRPrinter *p) {
    p->stream << "extern(" << op->name << ", " << op << ")";
  });

TVM_REGISTER_NODE_TYPE(ExternOpNode);

int ExternOpNode::num_outputs() const {
  return static_cast<int>(output_placeholders.size());
}

Array<IterVar> ExternOpNode::root_iter_vars() const {
  return axis;
}

Type ExternOpNode::output_dtype(size_t i) const {
  return output_placeholders[i]->dtype;
}

Array<Expr> ExternOpNode::output_shape(size_t i) const {
  return output_placeholders[i]->shape;
}

Operation ExternOpNode::make(std::string name,
                             std::string tag,
                             Array<IterVar> axis,
                             Array<Tensor> inputs,
                             Array<Buffer> input_placeholders,
                             Array<Buffer> output_placeholders,
                             Stmt body) {
  auto n = std::make_shared<ExternOpNode>();
  n->name = name;
  n->tag = tag;
  n->axis = axis;
  CHECK_EQ(inputs.size(), input_placeholders.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    CHECK_EQ(inputs[i]->dtype, input_placeholders[i]->dtype);
    CHECK(inputs[i]->shape.same_as(input_placeholders[i]->shape));
    CHECK_EQ(input_placeholders[i]->strides.size(), 0U);
  }
  n->inputs = inputs;
  n->input_placeholders = input_placeholders;
  n->output_placeholders = output_placeholders;
  n->body = body;
  return Operation(n);
}

Array<Tensor> ExternOpNode::InputTensors() const {
  return inputs;
}

Operation ExternOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  auto n = std::make_shared<ExternOpNode>(*this);
  n->body = op::ReplaceTensor(this->body, rmap);
  for (size_t i = 0; i < n->inputs.size(); ++i) {
    Tensor t = n->inputs[i];
    if (rmap.count(t)) {
      n->inputs.Set(i, rmap.at(t));
    }
  }
  if (body.same_as(n->body) &&
      inputs.same_as(n->inputs)) {
    return self;
  } else {
    return Operation(n);
  }
}

void ExternOpNode::PropBoundToInputs(
    const Operation& self,
    const std::unordered_map<const Variable*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  for (Tensor t : this->inputs) {
    auto it = out_dom_map->find(t);
    if (it == out_dom_map->end()) continue;
    TensorDom& dom = it->second;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      dom.data[i].emplace_back(IntSet::range(
          Range::make_by_min_extent(
              make_const(t->shape[i].type(), 0), t->shape[i])));
    }
  }
}

void ExternOpNode::GatherBound(
    const Operation& self,
    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  const TensorDom& tdom = tensor_dom.at(self.output(0));
  for (size_t i = 0; i < this->axis.size(); ++i) {
    Range r;
    if (i < tdom.data.size()) r = arith::Union(tdom.data.at(i)).cover_range(this->axis[i]->dom);
    else r = this->axis[i]->dom;
    CHECK(!out_dom_map->count(this->axis[i]));
    (*out_dom_map)[this->axis[i]] = r;
  }
}

Stmt ExternOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& realize_map,
    const Stmt& body) const {
  CHECK_EQ(stage->op.get(), this);
  // handle attachment
  std::unordered_map<const Variable*, Expr> vars_delete_inner;
  if (stage->attach_ivar.defined()) {
    auto extern_node = stage->op.as<ExternOpNode>();
    int axis_size = extern_node->axis.size();
    int attach_level = CountAttachLevel(stage);
    vars_delete_inner = GetVarsDeleteInner(stage, axis_size, attach_level);
  }
  Stmt realize_body = body;
  auto f_push_bind = [&realize_body, &vars_delete_inner](Buffer buffer, Tensor tensor) {
    Array<NodeRef> bind_spec;
    Array<Expr> tuple;
    bind_spec.push_back(buffer);
    bind_spec.push_back(tensor);
    for (size_t k = 0; k < buffer->shape.size(); ++k) {
      tuple.push_back(make_const(buffer->shape[k].type(), 0));
      if (k < vars_delete_inner.size()) {
        tuple.push_back(make_const(buffer->shape[k].type(), 1));
      } else {
        tuple.push_back(buffer->shape[k]);
      }
    }
    realize_body = AttrStmt::make(
        bind_spec, attr::buffer_bind_scope,
        Call::make(Handle(), intrinsic::tvm_tuple, tuple, Call::Intrinsic), realize_body);
  };
  for (size_t i = output_placeholders.size(); i != 0; --i) {
    f_push_bind(output_placeholders[i - 1], stage->op.output(i - 1));
  }
  for (size_t i = inputs.size(); i != 0; --i) {
    f_push_bind(input_placeholders[i - 1], inputs[i - 1]);
  }
  for (int k = 0; k < num_outputs(); ++k) {
    Tensor t = stage->op.output(k);
    HalideIR::Internal::Region bounds;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      if (i < vars_delete_inner.size()) {
        bounds.push_back(
          Range::make_by_min_extent(
            make_const(t->shape[i].type(), 0), make_const(t->shape[i].type(), 1)));
      } else {
        bounds.push_back(
          Range::make_by_min_extent(
            make_const(t->shape[i].type(), 0), t->shape[i]));
      }
    }
    realize_body = ir::Realize::make(
        t->op, t->value_index, t->dtype,
        bounds, const_true(), realize_body);
  }
  return realize_body;
}


class ForTypeRewriter : public IRMutator {
  public:
    explicit ForTypeRewriter(const Stage& stage) : stage_(stage) {}

    Stmt Mutate_(const For* op, const Stmt& s) final {
      Stmt body = Mutate(op->body);
      const AttrStmt* attr = op->body.as<AttrStmt>();
      if (attr != nullptr) {
        IterVar iv(attr->node.node_);
        ForType for_type = ForType::Serial;
        IterVarAttr it_attr;
        if (stage_->iter_var_attrs.count(iv)) {
          it_attr = stage_->iter_var_attrs[iv];
        }
        if (it_attr.defined()) {
          switch (it_attr->iter_type) {
            case kUnrolled: for_type = ForType::Unrolled; break;
            case kVectorized: for_type = ForType::Vectorized; break;
            case kParallelized: for_type = ForType::Parallel; break;
            case kDataPar: break;
            case kTensorized: break;
            case kPipelined: for_type = ForType::Pipelined; break;
            default: LOG(FATAL) << "Unknown iter type" << it_attr->iter_type;
          }
          return For::make(iv->var, op->min, op->extent,
                           for_type, op->device_api, body,
                           it_attr->for_loop_annotate_keys,
                           it_attr->for_loop_annotate_values);
        }
        return For::make(iv->var, op->min, op->extent,
                         for_type, op->device_api, body);
      } else {
        return IRMutator::Mutate_(op, s);
      }
    }
  private:
    const Stage& stage_;
};


class MakeFuseLoop final : public IRMutator {
  public:
    MakeFuseLoop(const IterVar& inner, const IterVar& outer, const IterVar& fused, std::unordered_map<const Variable*, Expr>& sub)
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


class MakeSplitLoop final : public IRMutator {
  public:
    MakeSplitLoop(const IterVar& parent,
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


class MakeReorderLoop final : public IRMutator {
  public:
    MakeReorderLoop(const Array<IterVar>& order)
      : order_(order) {}

    Stmt Reorder(const Stmt& stmt) {
      if_conditions_ = CollectIfConditions(stmt);
      return this->Mutate(stmt);
    }

    Stmt Mutate(Stmt stmt) final {
      if (const For* op = stmt.as<For>()) {
        if (FindForInOrder(op)) {
          loop_depth_ ++;
          Stmt rest;
          if (const IfThenElse* if_stmt = op->body.as<AttrStmt>()->body.as<IfThenElse>()) {
            rest = this->Mutate(if_stmt->then_case);
          } else {
            const AttrStmt* attr_stmt = op->body.as<AttrStmt>();
            rest = this->Mutate(attr_stmt->body);
          }
          loop_depth_ --;
          const IterVar& iv = order_[loop_depth_ - 1];
          Expr min = iv->dom->min;
          Expr extent = iv->dom->extent;
          if (if_conditions_.count(iv->var.get())) {
            Expr condition = if_conditions_[iv->var.get()];
            rest = IfThenElse::make(condition, rest);
            // Remove keys with duplicated values from the map.
            // This ensures that IF is associated with only the
            // inner for loop generated by split.
            std::vector<const Variable*> removed_keys;
            for (auto it = if_conditions_.begin(); it != if_conditions_.end(); it++) {
              if (it->second.same_as(condition) && !(it->first == iv->var.get())) {
                removed_keys.push_back(it->first);
              }
            }
            for (auto key: removed_keys) {
              if_conditions_.erase(key);
            }
          }
          rest = AttrStmt::make(iv, attr::loop_scope, iv->var, rest);
          return For::make(iv->var, min, extent, op->for_type, op->device_api, rest,
                            op->annotate_keys, op->annotate_values);
        } else {
          return IRMutator::Mutate(stmt);
        }
      } else {
        return IRMutator::Mutate(stmt);
      }
    }

  private:
    const Array<IterVar>& order_;
    int loop_depth_{1};
    std::unordered_map<const Variable*, Expr> if_conditions_;

    inline bool FindForInOrder(const For* op) {
      for (decltype(order_.size()) i = 0; i < order_.size(); i++) {
        if (op->loop_var.get() == order_[i]->var.get()) return true;
      }
      return false;
    }

    // Collect IF conditions generated by split schedule.
    // Associate IF with both inner loop and outer loop
    std::unordered_map<const Variable*, Expr> CollectIfConditions(Stmt stmt) {
      std::unordered_map<const Variable*, Expr> if_conditions;
      auto fvisit = [&if_conditions](const NodeRef& n) {
        if (const For* s1 = n.as<For>()) {
          if (s1->loop_var->name_hint.find(".outer") != std::string::npos ||
              s1->loop_var->name_hint.find(".inner") != std::string::npos) {
            if (const AttrStmt* s2 = s1->body.as<AttrStmt>()) {
              if (const IfThenElse* s3 = s2->body.as<IfThenElse>()) {
                if_conditions[s1->loop_var.get()] = s3->condition;
              } else if (const For* s4 = s2->body.as<For>()) {
                if (const AttrStmt* s5 = s4->body.as<AttrStmt>()) {
                  if (const IfThenElse* s6 = s5->body.as<IfThenElse>()) {
                    if_conditions[s1->loop_var.get()] = s6->condition;
                  }
                }
              }
            }
          }
        }
      };
      PostOrderVisit(stmt, fvisit);
      return if_conditions;
    }

};


class ComputeAtScheduler final : public IRMutator {
  public:
    ComputeAtScheduler(const Stage& stage) : stage_(stage) {
      auto extern_node = stage_->op.as<ExternOpNode>();
      axis_size_ = extern_node->axis.size();
      attach_level_ = CountAttachLevel(stage_);
      vars_delete_inner_ = GetVarsDeleteInner(stage_, axis_size_, attach_level_);
      vars_sub_ = GetVarsSub(stage_, axis_size_, attach_level_);
    }

    Stmt Schedule(const Stmt& stmt) {
      return op::Substitute(this->Mutate(stmt), vars_sub_);;
    }

    Stmt Mutate_(const Store* op, const Stmt& s) final {
      // mutate the index after attachment
      Expr index = ir::Substitute(op->index, vars_delete_inner_);
      Expr value = this->Mutate(op->value);
      Expr predicate = this->Mutate(op->predicate);
      if (predicate.same_as(op->predicate) &&
          value.same_as(op->value) &&
          index.same_as(op->index)) {
        return s;
      } else {
        return Store::make(op->buffer_var, value, index, predicate);
      }
    }

    Stmt Mutate_(const Realize* op, const Stmt& s) final {
      IRMutator* m = this;
      HalideIR::Internal::Region new_bounds;
      bool bounds_changed = false;
      for (size_t i = 0; i < op->bounds.size(); i++) {
        Expr old_min = op->bounds[i]->min;
        Expr old_extent = op->bounds[i]->extent;
        Expr new_min = old_min;
        Expr new_extent = old_extent;
        // mutate the bounds after attachment
        if (i < attach_level_) {
          new_min = make_const(Int(32), 0);
          new_extent = make_const(Int(32), 1);
        }
        if (!new_min.same_as(old_min)) bounds_changed = true;
        if (!new_extent.same_as(old_extent)) bounds_changed = true;
        new_bounds.push_back(
            Range::make_by_min_extent(new_min, new_extent));
      }
      Stmt body = m->Mutate(op->body);
      Expr condition = m->Mutate(op->condition);
      if (!bounds_changed &&
          body.same_as(op->body) &&
          condition.same_as(op->condition)) {
        return s;
      } else {
        return Realize::make(op->func, op->value_index,
                             op->type, new_bounds,
                             condition, body);
      }
    }

    Stmt Mutate_(const For *op, const Stmt& s) final {
      current_level_++;
      // remove several outer loops after attachment
      if (current_level_ <= attach_level_) {
        return this->Mutate(op->body.as<AttrStmt>()->body);
      } else {
        Expr min = this->Mutate(op->min);
        Expr extent = this->Mutate(op->extent);
        Stmt body = this->Mutate(op->body);
        if (min.same_as(op->min) &&
            extent.same_as(op->extent) &&
            body.same_as(op->body)) {
          return s;
        } else {
          return For::make(
              op->loop_var, min, extent, op->for_type, op->device_api, body,
              op->annotate_keys, op->annotate_values);
        }
      }
    }

  private:
    const Stage& stage_;
    size_t axis_size_;
    size_t attach_level_;
    size_t current_level_{0};
    std::unordered_map<const Variable*, Expr> vars_delete_inner_;
    std::unordered_map<const Variable*, Expr> vars_sub_;
};


Stmt ExternOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool del_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt stmt = this->body;
  // construct the body
  if (stage->attach_ivar.defined()) {
    ComputeAtScheduler scheduler(stage);
    stmt = scheduler.Schedule(stmt);
  }
  for (auto rel : stage->relations) {
    if (const FuseNode* r = rel.as<FuseNode>()) {
      std::unordered_map<const Variable*, Expr> sub;
      MakeFuseLoop mutator(r->inner, r->outer, r->fused, sub);
      stmt = mutator.Mutate(stmt);
      stmt = op::Substitute(stmt, sub);
    } else if (const ReorderNode* r = rel.as<ReorderNode>()) {
      MakeReorderLoop mutator(r->order);
      stmt = mutator.Reorder(stmt);
    } else if (const SplitNode* r = rel.as<SplitNode>()) {
      std::unordered_map<const Variable*, Expr> sub;
      MakeSplitLoop mutator(r->parent, r->factor, r->nparts, r->outer, r->inner, sub);
      stmt = mutator.Mutate(stmt);
      stmt = op::Substitute(stmt, sub);
    }
  }
  ForTypeRewriter rewriter(stage);
  stmt = rewriter.Mutate(stmt);
  return AttrStmt::make(make_zero(Int(32)), attr::extern_scope, 0, stmt);
}
}  // namespace tvm
