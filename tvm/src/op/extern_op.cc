/*!
 *  Copyright (c) 2017 by Contributors
 * \brief External computation rule.
 * \file extern_op.cc
 */
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>
#include "./op_util.h"

namespace tvm {

using namespace ir;
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
  Stmt realize_body = body;
  auto f_push_bind = [&realize_body](Buffer buffer, Tensor tensor) {
    Array<NodeRef> bind_spec;
    Array<Expr> tuple;
    bind_spec.push_back(buffer);
    bind_spec.push_back(tensor);
    for (size_t k = 0; k < buffer->shape.size(); ++k) {
      tuple.push_back(make_const(buffer->shape[k].type(), 0));
      tuple.push_back(buffer->shape[k]);
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
      bounds.push_back(
          Range::make_by_min_extent(
              make_const(t->shape[i].type(), 0), t->shape[i]));
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

    std::unordered_map<const Variable*, Expr> CollectIfConditions(Stmt stmt) {
      std::unordered_map<const Variable*, Expr> if_conditions;
      auto fvisit = [&if_conditions](const NodeRef& n) {
        if (const For* s1 = n.as<For>()) {
          if (const AttrStmt* s2 = s1->body.as<AttrStmt>()) {
            if (const IfThenElse* s3 = s2->body.as<IfThenElse>()) {
                if_conditions[s1->loop_var.get()] = s3->condition;
            }
          }
        }
      };
      PostOrderVisit(stmt, fvisit);
      return if_conditions;
    }

};

int CountLevel(const Stage& stage, const IterVar& ivar) {
  int level = 0;
  for (auto iv : stage->attach_stage->leaf_iter_vars) {
    level += 1;
    if (stage->attach_ivar == iv || 
        stage->origin_attach_ivar == iv) {
      break;
    }
  }
  return level;
}

int UnfuseLevel(const Stage& stage, const IterVar& ivar) {
  for (auto rel : stage->attach_stage->relations) {
    if (const FuseNode* node = rel.as<FuseNode>()) {
      if (node->fused == ivar)
        return UnfuseLevel(stage, node->outer) + UnfuseLevel(stage, node->inner);
    }
  }
  return 1;
}

Stmt ExternOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool del_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt stmt = this->body;
  // construct the body
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
  if (stage->attach_ivar.defined()) {
    int attach_level = CountLevel(stage, stage->origin_attach_ivar) 
                       + UnfuseLevel(stage, stage->origin_attach_ivar) - 1;
    int self_level = this->axis.size();
    int level = std::min(attach_level, self_level);
    std::unordered_map<const Variable*, Expr> sub;
    for (int i = 0; i < level; i++) {
      const For* f = stmt.as<For>();
      if (f == nullptr) {
        LOG(FATAL) << "Incorrect usage of compute_at: " 
          << stage->op->name << " @ " 
          << stage->attach_stage->op->name << " : "
          << stage->origin_attach_ivar->var;
      }
      sub[f->loop_var.get()] = stage->attach_stage->iter_var_exprs[i];
      const AttrStmt* a = f->body.as<AttrStmt>();
      stmt = a->body;
    }
    stmt = op::Substitute(stmt, sub);
  }
  ForTypeRewriter rewriter(stage);
  stmt = rewriter.Mutate(stmt);
  return AttrStmt::make(make_zero(Int(32)), attr::extern_scope, 0, stmt);
}
}  // namespace tvm
