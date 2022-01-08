/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_visitor.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>

namespace TVM {
namespace ir {
// visitor to implement apply
class IRApplyVisit : public IRVisitor {
 public:
  explicit IRApplyVisit(std::function<void(const NodeRef &)> f) : f_(f) {}

  void Visit(const NodeRef &node) final {
    if (visited_.count(node.get()) != 0) return;
    visited_.insert(node.get());
    IRVisitor::Visit(node);
    f_(node);
  }

 private:
  std::function<void(const NodeRef &)> f_;
  std::unordered_set<const Node *> visited_;
};

void PostOrderVisit(const NodeRef &node,
                    std::function<void(const NodeRef &)> fvisit) {
  IRApplyVisit(fvisit).Visit(node);
}

IRVisitor::FVisit &IRVisitor::vtable() {  // NOLINT(*)
  static FVisit inst;
  return inst;
}

inline void VisitArray(const Array<Expr> &arr, IRVisitor *v) {
  for (size_t i = 0; i < arr.size(); i++) {
    v->Visit(arr[i]);
  }
}

inline void VisitRDom(const Array<IterVar> &rdom, IRVisitor *v) {
  for (size_t i = 0; i < rdom.size(); i++) {
    Range r = rdom[i]->dom;
    v->Visit(r->min);
    v->Visit(r->extent);
  }
}

void IRVisitor::Visit_(const Variable *op) {}

void IRVisitor::Visit_(const LetStmt *op) {
  this->Visit(op->value);
  this->Visit(op->body);
}

void IRVisitor::Visit_(const AttrStmt *op) {
  this->Visit(op->value);
  this->Visit(op->body);
}

void IRVisitor::Visit_(const For *op) {
  IRVisitor *v = this;
  v->Visit(op->min);
  v->Visit(op->extent);
  v->Visit(op->body);
}

void IRVisitor::Visit_(const Allocate *op) {
  IRVisitor *v = this;
  for (size_t i = 0; i < op->extents.size(); i++) {
    v->Visit(op->extents[i]);
  }
  for (size_t i = 0; i < op->attrs.size(); i++) {
    v->Visit(op->attrs[i]);
  }
  v->Visit(op->body);
  v->Visit(op->condition);
  if (op->new_expr.defined()) {
    v->Visit(op->new_expr);
  }
}

void IRVisitor::Visit_(const Load *op) {
  this->Visit(op->index);
  this->Visit(op->predicate);
}

void IRVisitor::Visit_(const Store *op) {
  this->Visit(op->value);
  this->Visit(op->index);
  this->Visit(op->predicate);
}

void IRVisitor::Visit_(const IfThenElse *op) {
  this->Visit(op->condition);
  this->Visit(op->then_case);
  if (op->else_case.defined()) {
    this->Visit(op->else_case);
  }
}

void IRVisitor::Visit_(const Let *op) {
  this->Visit(op->value);
  this->Visit(op->body);
}

void IRVisitor::Visit_(const Free *op) {}

void IRVisitor::Visit_(const Call *op) { VisitArray(op->args, this); }

#define DEFINE_BINOP_VISIT_(OP)          \
  void IRVisitor::Visit_(const OP *op) { \
    this->Visit(op->a);                  \
    this->Visit(op->b);                  \
  }

DEFINE_BINOP_VISIT_(Add)
DEFINE_BINOP_VISIT_(Sub)
DEFINE_BINOP_VISIT_(Mul)
DEFINE_BINOP_VISIT_(Div)
DEFINE_BINOP_VISIT_(Mod)
DEFINE_BINOP_VISIT_(Min)
DEFINE_BINOP_VISIT_(Max)
DEFINE_BINOP_VISIT_(EQ)
DEFINE_BINOP_VISIT_(NE)
DEFINE_BINOP_VISIT_(LT)
DEFINE_BINOP_VISIT_(LE)
DEFINE_BINOP_VISIT_(GT)
DEFINE_BINOP_VISIT_(GE)
DEFINE_BINOP_VISIT_(And)
DEFINE_BINOP_VISIT_(Or)

void IRVisitor::Visit_(const Reduce *op) {
  VisitRDom(op->axis, this);
  VisitArray(op->source, this);
  this->Visit(op->condition);
}

void IRVisitor::Visit_(const Cast *op) { this->Visit(op->value); }

void IRVisitor::Visit_(const Not *op) { this->Visit(op->a); }

void IRVisitor::Visit_(const Select *op) {
  this->Visit(op->condition);
  this->Visit(op->true_value);
  this->Visit(op->false_value);
}

void IRVisitor::Visit_(const Ramp *op) {
  this->Visit(op->base);
  this->Visit(op->stride);
}

void IRVisitor::Visit_(const Broadcast *op) { this->Visit(op->value); }

void IRVisitor::Visit_(const AssertStmt *op) {
  this->Visit(op->condition);
  this->Visit(op->message);
  this->Visit(op->body);
}

void IRVisitor::Visit_(const ProducerConsumer *op) { this->Visit(op->body); }

void IRVisitor::Visit_(const Provide *op) {
  VisitArray(op->args, this);
  this->Visit(op->value);
}

void IRVisitor::Visit_(const Realize *op) {
  for (size_t i = 0; i < op->bounds.size(); i++) {
    this->Visit(op->bounds[i]->min);
    this->Visit(op->bounds[i]->extent);
  }

  this->Visit(op->body);
  this->Visit(op->condition);
}

void IRVisitor::Visit_(const Prefetch *op) {
  for (size_t i = 0; i < op->bounds.size(); i++) {
    this->Visit(op->bounds[i]->min);
    this->Visit(op->bounds[i]->extent);
  }
}

void IRVisitor::Visit_(const Block *op) {
  this->Visit(op->first);
  this->Visit(op->rest);
}

void IRVisitor::Visit_(const Evaluate *op) { this->Visit(op->value); }

void IRVisitor::Visit_(const GetBit *op) {
  this->Visit(op->a);
  this->Visit(op->index);
}

void IRVisitor::Visit_(const GetSlice *op) {
  this->Visit(op->a);
  this->Visit(op->index_left);
  this->Visit(op->index_right);
}

void IRVisitor::Visit_(const SetBit *op) {
  this->Visit(op->a);
  this->Visit(op->value);
  this->Visit(op->index);
}

void IRVisitor::Visit_(const SetSlice *op) {
  this->Visit(op->a);
  this->Visit(op->value);
  this->Visit(op->index_left);
  this->Visit(op->index_right);
}

void IRVisitor::Visit_(const Quantize *op) {
  this->Visit(op->body);
  this->Visit(op->bitwidth);
}

void IRVisitor::Visit_(const KernelDef *op) {
  for (size_t i = 0; i < op->args.size(); i++) {
    this->Visit(op->args[i]);
  }
  this->Visit(op->ret_void);
}

void IRVisitor::Visit_(const KernelExpr *op) {
  for (size_t i = 0; i < op->args.size(); i++) {
    this->Visit(op->args[i]);
  }
}

void IRVisitor::Visit_(const KernelStmt *op) {
  for (size_t i = 0; i < op->args.size(); i++) {
    this->Visit(op->args[i]);
  }
}

void IRVisitor::Visit_(const StreamStmt *op) { this->Visit(op->value); }

void IRVisitor::Visit_(const StreamExpr *op) {}

void IRVisitor::Visit_(const Return *op) { this->Visit(op->value); }

void IRVisitor::Visit_(const Break *op) {}

void IRVisitor::Visit_(const While *op) {
  this->Visit(op->condition);
  this->Visit(op->body);
}

void IRVisitor::Visit_(const Reuse *op) { this->Visit(op->body); }

void IRVisitor::Visit_(const Partition *op) {}

void IRVisitor::Visit_(const Stencil *op) { this->Visit(op->body); }

void IRVisitor::Visit_(const ExternModule *op) {
  this->Visit(op->value);
  this->Visit(op->body);
}

void IRVisitor::Visit_(const Print *op) {
  for (size_t i = 0; i < op->values.size(); i++) {
    this->Visit(op->values[i]);
  }
}

void IRVisitor::Visit_(const MultiBlock *op) {
  for (size_t i = 0; i < op->stmts.size(); i++) {
    this->Visit(op->stmts[i]);
  }
}

void IRVisitor::Visit_(const Assert *op) {
  this->Visit(op->condition);
  for (size_t i = 0; i < op->values.size(); i++) {
    this->Visit(op->values[i]);
  }
}

#define DEFINE_OP_NO_VISIT_(OP) \
  void IRVisitor::Visit_(const OP *op) {}

DEFINE_OP_NO_VISIT_(IntImm)
DEFINE_OP_NO_VISIT_(UIntImm)
DEFINE_OP_NO_VISIT_(FloatImm)
DEFINE_OP_NO_VISIT_(StringImm)

#define DISPATCH_TO_VISIT(OP) \
  set_dispatch<OP>([](const OP *op, IRVisitor *v) { v->Visit_(op); })

TVM_STATIC_IR_FUNCTOR(IRVisitor, vtable)
    .DISPATCH_TO_VISIT(Variable)
    .DISPATCH_TO_VISIT(LetStmt)
    .DISPATCH_TO_VISIT(AttrStmt)
    .DISPATCH_TO_VISIT(IfThenElse)
    .DISPATCH_TO_VISIT(For)
    .DISPATCH_TO_VISIT(Allocate)
    .DISPATCH_TO_VISIT(Load)
    .DISPATCH_TO_VISIT(Store)
    .DISPATCH_TO_VISIT(Let)
    .DISPATCH_TO_VISIT(Free)
    .DISPATCH_TO_VISIT(Call)
    .DISPATCH_TO_VISIT(Add)
    .DISPATCH_TO_VISIT(Sub)
    .DISPATCH_TO_VISIT(Mul)
    .DISPATCH_TO_VISIT(Div)
    .DISPATCH_TO_VISIT(Mod)
    .DISPATCH_TO_VISIT(Min)
    .DISPATCH_TO_VISIT(Max)
    .DISPATCH_TO_VISIT(EQ)
    .DISPATCH_TO_VISIT(NE)
    .DISPATCH_TO_VISIT(LT)
    .DISPATCH_TO_VISIT(LE)
    .DISPATCH_TO_VISIT(GT)
    .DISPATCH_TO_VISIT(GE)
    .DISPATCH_TO_VISIT(And)
    .DISPATCH_TO_VISIT(Or)
    .DISPATCH_TO_VISIT(Reduce)
    .DISPATCH_TO_VISIT(Cast)
    .DISPATCH_TO_VISIT(Not)
    .DISPATCH_TO_VISIT(Select)
    .DISPATCH_TO_VISIT(Ramp)
    .DISPATCH_TO_VISIT(Broadcast)
    .DISPATCH_TO_VISIT(AssertStmt)
    .DISPATCH_TO_VISIT(ProducerConsumer)
    .DISPATCH_TO_VISIT(Provide)
    .DISPATCH_TO_VISIT(Realize)
    .DISPATCH_TO_VISIT(Block)
    .DISPATCH_TO_VISIT(Evaluate)
    .DISPATCH_TO_VISIT(IntImm)
    .DISPATCH_TO_VISIT(UIntImm)
    .DISPATCH_TO_VISIT(FloatImm)
    .DISPATCH_TO_VISIT(StringImm)
    .DISPATCH_TO_VISIT(Prefetch)
    .DISPATCH_TO_VISIT(GetBit)
    .DISPATCH_TO_VISIT(GetSlice)
    .DISPATCH_TO_VISIT(SetBit)
    .DISPATCH_TO_VISIT(SetSlice)
    .DISPATCH_TO_VISIT(Quantize)
    .DISPATCH_TO_VISIT(KernelDef)
    .DISPATCH_TO_VISIT(KernelExpr)
    .DISPATCH_TO_VISIT(KernelStmt)
    .DISPATCH_TO_VISIT(StreamStmt)
    .DISPATCH_TO_VISIT(StreamExpr)
    .DISPATCH_TO_VISIT(Return)
    .DISPATCH_TO_VISIT(Break)
    .DISPATCH_TO_VISIT(While)
    .DISPATCH_TO_VISIT(Reuse)
    .DISPATCH_TO_VISIT(Partition)
    .DISPATCH_TO_VISIT(Stencil)
    .DISPATCH_TO_VISIT(ExternModule)
    .DISPATCH_TO_VISIT(Print)
    .DISPATCH_TO_VISIT(MultiBlock)
    .DISPATCH_TO_VISIT(Assert);

}  // namespace ir
}  // namespace TVM
