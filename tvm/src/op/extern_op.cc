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
#include <ir/IREquality.h>
#include <unordered_set>
#include "./op_util.h"
#include "./extern_op.h"

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
  // the top level (outer most) buffer var
  return buffer_vars.back();
}

std::vector<int> MapIterVarExprsIndex(std::vector<Expr> iter_var_exprs_before_reorder,
                                      std::vector<Expr> iter_var_exprs_after_reorder) {
  std::vector<int> index_table_;
  if (iter_var_exprs_before_reorder.size() != iter_var_exprs_after_reorder.size()) {
    LOG(FATAL) << "size of iter_var_exprs_before_reorder "
               << "and iter_var_exprs_after_reorder don't match";
  }
  for (size_t i = 0; i < iter_var_exprs_after_reorder.size(); i++) {
    Expr elem = iter_var_exprs_after_reorder[i];
    for (size_t index = 0; index < iter_var_exprs_before_reorder.size(); index++) {
      if (HalideIR::Internal::equal(elem, iter_var_exprs_before_reorder[index])) {
        index_table_.push_back(index);
      }
   }
  }
  return index_table_;
}

std::vector<Expr>
GetBoundInnerStore(const Stage& stage, int axis_size, int attach_level) {
  auto extern_node = stage->op.as<ExternOpNode>();
  std::vector<Expr> bounds;
  for (int i = 0; i < axis_size; i++) {
    if (i <= attach_level) {
      bounds.push_back(make_const(Int(32), 1));
    } else {
      bounds.push_back(extern_node->axis[i]->dom->extent);
    }
  }
  return bounds;
}
}  // namespace

int CountAttachLevel(const Stage& stage) {
  /*
  int attach_level = 0;
  const ExternOpNode* node = stage->attach_stage->op.as<ExternOpNode>();
  for (auto iv : node->axis) {
    if (stage->attach_ivar->var.get() == iv->var.get()) {
      break;
    }
    attach_level += 1;
  }
  LOG(INFO) << attach_level;
  return attach_level;*/
  return stage->attach_level;
}

std::unordered_map<const Variable*, std::vector<IterVar> >
GetAxisInnerStoreRemain(const Stage& stage, int axis_size, int attach_level) {
  auto extern_node = stage->op.as<ExternOpNode>();
  Stmt stmt = extern_node->body;
  const Variable* buffer_var = GetBufferVar(stmt);
  std::unordered_map<const Variable*, std::vector<IterVar> > axis_remain;
  axis_remain[buffer_var] = {};
  for (int i = attach_level + 1; i < axis_size; i++) {
    axis_remain[buffer_var].push_back(extern_node->axis[i]);
  }
  return axis_remain;
}

std::unordered_map<const Variable*, std::vector<IterVar> >
GetAxisOuterLoadRemain(const Stage& stage, int axis_size, int attach_level) {
  auto extern_node = stage->op.as<ExternOpNode>();
  auto attach_node = stage->attach_stage->op.as<ExternOpNode>();
  Stmt stmt = extern_node->body;
  const Variable* buffer_var = GetBufferVar(stmt);
  std::unordered_map<const Variable*, std::vector<IterVar> > axis_remain;
  axis_remain[buffer_var] = {};
  for (int i = attach_level + 1; i < axis_size; i++) {
    axis_remain[buffer_var].push_back(attach_node->axis[i]);
  }
  return axis_remain;
}

std::vector<IterVar>
GetIterVarsInIndexRemain(Expr index, std::vector<IterVar> iv_remain) {
  std::vector<const Variable*> vars_in_index;
  auto f = [&vars_in_index](const NodeRef& n) {
    if (const Variable* var = n.as<Variable>()) {
      vars_in_index.push_back(var);
    }
  };
  PostOrderVisit(index, f);
  std::vector<IterVar> ret;
  for (auto iv : iv_remain) {
    for (auto var : vars_in_index) {
      if (var == iv->var.get()) {
        ret.push_back(iv);
      }
    }
  }
  return ret;
}

std::unordered_map<const Variable*, Expr>
GetVarsInnerLoadSub(const Stage& stage, int axis_size, int attach_level) {
  auto extern_node = stage->op.as<ExternOpNode>();
  Stmt stmt = extern_node->body;
  std::vector<int> index_table = MapIterVarExprsIndex(stage->attach_stage->iter_var_exprs_before_reorder,
                                                      stage->attach_stage->iter_var_exprs_after_reorder);
  std::unordered_map<const Variable*, Expr> vars_inner_load_sub;
  for (int i = 0; i <= attach_level; i++) {
    vars_inner_load_sub[extern_node->axis[index_table[i]]->var.get()] =
      stage->attach_stage->iter_var_exprs_after_reorder[i];
  }
  return vars_inner_load_sub;
}

Expr MakeIndexFromIterVars(std::vector<IterVar> vars) {
  Expr index = make_const(Int(32), 0);
  for (auto iv : vars) {
    index = Add::make(Mul::make(index, iv->dom->extent), iv->var);
  }
  return index;
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
  auto extern_node = stage->op.as<ExternOpNode>();
  std::vector<Expr> bounds_inner;
  if (stage->attach_ivar.defined()) {
    int axis_size = extern_node->axis.size();
    int attach_level = CountAttachLevel(stage);
    LOG(INFO) << attach_level;
    bounds_inner = GetBoundInnerStore(stage, axis_size, attach_level);
  }
  Stmt realize_body = body;
  auto f_push_bind = [&](Buffer buffer, Tensor tensor) {
    Array<NodeRef> bind_spec;
    Array<Expr> tuple;
    bind_spec.push_back(buffer);
    bind_spec.push_back(tensor);
    for (size_t k = 0; k < buffer->shape.size(); ++k) {
      tuple.push_back(make_const(buffer->shape[k].type(), 0));
      if (stage->attach_ivar.defined()) {
        tuple.push_back(bounds_inner[k]);
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
      if (stage->attach_ivar.defined()) {
        bounds.push_back(
          Range::make_by_min_extent(
            make_const(t->shape[i].type(), 0), bounds_inner[i]));
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

/*
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
*/
/*
class ComputeAtScheduler final : public IRMutator {
  public:
    ComputeAtScheduler(const Stage& stage) : stage_(stage) {
      auto extern_node = stage_->op.as<ExternOpNode>();
      axis_size_ = extern_node->axis.size();
      attach_level_ = CountAttachLevel(stage_);
      axis_remain_ = GetAxisInnerStoreRemain(stage_, axis_size_, attach_level_);
      index_table_ = MapIterVarExprsIndex(stage_->attach_stage->iter_var_exprs_before_reorder,
                                          stage_->attach_stage->iter_var_exprs_after_reorder);
      need_reorder_compute_axis_ = NeedReorderComputeAxis(index_table_[attach_level_],
                                                          stage_->attach_stage->iter_var_exprs_before_reorder,
                                                          stage_->attach_stage->iter_var_exprs_after_reorder);
      vars_inner_load_sub_ = GetVarsInnerLoadSub(stage_, axis_size_, attach_level_);
    }

    Stmt Schedule(const Stmt& stmt) {
      Stmt ret = stmt;
      if (need_reorder_compute_axis_) {
        std::vector<IterVar> order;
        for (int i = 0; i < axis_size_; i++) {
          order.push_back(stage_->op.as<ExternOpNode>()->axis[index_table_[i]]);
        }
        MakeReorderLoop mutator(order);
        ret = mutator.Reorder(ret);
      }
      return op::Substitute(this->Mutate(ret), vars_inner_load_sub_);;
    }

    Stmt Mutate_(const Store* op, const Stmt& s) final {
      // mutate the index after attachment
      Expr index = op->index;
      auto it = axis_remain_.find(op->buffer_var.get());
      if (it != axis_remain_.end()) {
        std::vector<IterVar> vars_in_index_remain =  GetIterVarsInIndexRemain(index, it->second);
        index = MakeIndexFromIterVars(vars_in_index_remain);
      }
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
        if (int(i) <= attach_level_) {
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
      if (current_level_ - 1 <= attach_level_) {
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
    int axis_size_;
    int attach_level_;
    int current_level_{0};
    bool need_reorder_compute_axis_;
    std::vector<int> index_table_;
    std::unordered_map<const Variable*, std::vector<IterVar> > axis_remain_;
    std::unordered_map<const Variable*, Expr> vars_inner_load_sub_;

    bool NeedReorderComputeAxis(int attach_level,
                                std::vector<Expr> iter_var_exprs_before_reorder,
                                std::vector<Expr> iter_var_exprs_after_reorder) {
      if (iter_var_exprs_before_reorder.size() != iter_var_exprs_after_reorder.size()) {
        LOG(FATAL) << "size of iter_var_exprs_before_reorder "
                   << "and iter_var_exprs_after_reorder don't match";
      }
      for (int i = 0; i <= attach_level; i++) {
        if (!HalideIR::Internal::equal(iter_var_exprs_before_reorder.at(i),
                                       iter_var_exprs_after_reorder.at(i)))
          return true;
      }
      return false;
    }
};
*/

Stmt ExternOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool del_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt stmt = this->body;
  // // construct the body
  // if (stage->attach_ivar.defined()) {
  //   ComputeAtScheduler scheduler(stage);
  //   stmt = scheduler.Schedule(stmt);
  // }
  // for (auto rel : stage->relations) {
  //   if (const FuseNode* r = rel.as<FuseNode>()) {
  //     std::unordered_map<const Variable*, Expr> sub;
  //     MakeFuseLoop mutator(r->inner, r->outer, r->fused, sub);
  //     stmt = mutator.Mutate(stmt);
  //     stmt = op::Substitute(stmt, sub);
  //   } else if (const ReorderNode* r = rel.as<ReorderNode>()) {
  //     MakeReorderLoop mutator(r->order);
  //     stmt = mutator.Reorder(stmt);
  //   } else if (const SplitNode* r = rel.as<SplitNode>()) {
  //     std::unordered_map<const Variable*, Expr> sub;
  //     MakeSplitLoop mutator(r->parent, r->factor, r->nparts, r->outer, r->inner, sub);
  //     stmt = mutator.Mutate(stmt);
  //     stmt = op::Substitute(stmt, sub);
  //   }
  // }
  // ForTypeRewriter rewriter(stage);
  // stmt = rewriter.Mutate(stmt);
  return AttrStmt::make(make_zero(Int(32)), attr::extern_scope, 0, stmt);
  //return stmt;
}
}  // namespace tvm
