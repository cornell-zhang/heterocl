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

Stmt ExternOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool del_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt stmt = this->body;
  return AttrStmt::make(make_zero(Int(32)), attr::extern_scope, 0, stmt);
}
}  // namespace tvm
