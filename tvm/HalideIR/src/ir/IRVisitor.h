/*!
 *  Copyright (c) 2016 by Contributors
 */
#ifndef HALIDEIR_IR_VISITOR_H
#define HALIDEIR_IR_VISITOR_H

#include <map>
#include <set>
#include <string>

#include "IR.h"
#include "base/Util.h"

/** \file
 * Defines the base class for things that recursively walk over the IR
 */

namespace Halide {
namespace Internal {

/** A base class for algorithms that need to recursively walk over the
 * IR. The default implementations just recursively walk over the
 * children. Override the ones you care about.
 */
class IRVisitor {
 public:
  EXPORT virtual ~IRVisitor();
  EXPORT virtual void visit(const IntImm *, const Expr &);
  EXPORT virtual void visit(const UIntImm *, const Expr &);
  EXPORT virtual void visit(const FloatImm *, const Expr &);
  EXPORT virtual void visit(const StringImm *, const Expr &);
  EXPORT virtual void visit(const Cast *, const Expr &);
  EXPORT virtual void visit(const Variable *, const Expr &);
  EXPORT virtual void visit(const Add *, const Expr &);
  EXPORT virtual void visit(const Sub *, const Expr &);
  EXPORT virtual void visit(const Mul *, const Expr &);
  EXPORT virtual void visit(const Div *, const Expr &);
  EXPORT virtual void visit(const Mod *, const Expr &);
  EXPORT virtual void visit(const Min *, const Expr &);
  EXPORT virtual void visit(const Max *, const Expr &);
  EXPORT virtual void visit(const EQ *, const Expr &);
  EXPORT virtual void visit(const NE *, const Expr &);
  EXPORT virtual void visit(const LT *, const Expr &);
  EXPORT virtual void visit(const LE *, const Expr &);
  EXPORT virtual void visit(const GT *, const Expr &);
  EXPORT virtual void visit(const GE *, const Expr &);
  EXPORT virtual void visit(const And *, const Expr &);
  EXPORT virtual void visit(const Or *, const Expr &);
  EXPORT virtual void visit(const Not *, const Expr &);
  EXPORT virtual void visit(const Select *, const Expr &);
  EXPORT virtual void visit(const Load *, const Expr &);
  EXPORT virtual void visit(const Ramp *, const Expr &);
  EXPORT virtual void visit(const Broadcast *, const Expr &);
  EXPORT virtual void visit(const Call *, const Expr &);
  EXPORT virtual void visit(const Let *, const Expr &);
  EXPORT virtual void visit(const Shuffle *, const Expr &);
  EXPORT virtual void visit(const LetStmt *, const Stmt &);
  EXPORT virtual void visit(const AttrStmt *, const Stmt &);
  EXPORT virtual void visit(const AssertStmt *, const Stmt &);
  EXPORT virtual void visit(const ProducerConsumer *, const Stmt &);
  EXPORT virtual void visit(const For *, const Stmt &);
  EXPORT virtual void visit(const Store *, const Stmt &);
  EXPORT virtual void visit(const Provide *, const Stmt &);
  EXPORT virtual void visit(const Allocate *, const Stmt &);
  EXPORT virtual void visit(const Free *, const Stmt &);
  EXPORT virtual void visit(const Realize *, const Stmt &);
  EXPORT virtual void visit(const Prefetch *, const Stmt &);
  EXPORT virtual void visit(const Block *, const Stmt &);
  EXPORT virtual void visit(const IfThenElse *, const Stmt &);
  EXPORT virtual void visit(const Evaluate *, const Stmt &);
  EXPORT virtual void visit(const GetBit *, const Expr &);
  EXPORT virtual void visit(const GetSlice *, const Expr &);
  EXPORT virtual void visit(const SetBit *, const Expr &);
  EXPORT virtual void visit(const SetSlice *, const Expr &);
  EXPORT virtual void visit(const Quantize *, const Expr &);
  EXPORT virtual void visit(const KernelDef *, const Stmt &);
  EXPORT virtual void visit(const KernelExpr *, const Expr &);
  EXPORT virtual void visit(const KernelStmt *, const Stmt &);
  EXPORT virtual void visit(const Return *, const Stmt &);
  EXPORT virtual void visit(const Break *, const Stmt &);
  EXPORT virtual void visit(const While *, const Stmt &);
  EXPORT virtual void visit(const Reuse *, const Stmt &);
  EXPORT virtual void visit(const Partition *, const Stmt &);
  EXPORT virtual void visit(const Stencil *, const Stmt &);
  EXPORT virtual void visit(const ExternModule *, const Stmt &);
  EXPORT virtual void visit(const StreamStmt *, const Stmt &);
  EXPORT virtual void visit(const StreamExpr *, const Expr &);
  EXPORT virtual void visit(const Print *, const Stmt &);
  EXPORT virtual void visit(const MultiBlock *, const Stmt &);
  EXPORT virtual void visit(const Assert *, const Stmt &);
};

/** A base class for algorithms that walk recursively over the IR
 * without visiting the same node twice. This is for passes that are
 * capable of interpreting the IR as a DAG instead of a tree. */
class IRGraphVisitor : public IRVisitor {
 protected:
  /** By default these methods add the node to the visited set, and
   * return whether or not it was already there. If it wasn't there,
   * it delegates to the appropriate visit method. You can override
   * them if you like. */
  // @{
  EXPORT virtual void include(const Expr &);
  EXPORT virtual void include(const Stmt &);
  // @}

  /** The nodes visited so far */
  std::set<const IRNode *> visited;

 public:
  /** These methods should call 'include' on the children to only
   * visit them if they haven't been visited already. */
  // @{
  EXPORT virtual void visit(const IntImm *, const Expr &);
  EXPORT virtual void visit(const UIntImm *, const Expr &);
  EXPORT virtual void visit(const FloatImm *, const Expr &);
  EXPORT virtual void visit(const StringImm *, const Expr &);
  EXPORT virtual void visit(const Cast *, const Expr &);
  EXPORT virtual void visit(const Variable *, const Expr &);
  EXPORT virtual void visit(const Add *, const Expr &);
  EXPORT virtual void visit(const Sub *, const Expr &);
  EXPORT virtual void visit(const Mul *, const Expr &);
  EXPORT virtual void visit(const Div *, const Expr &);
  EXPORT virtual void visit(const Mod *, const Expr &);
  EXPORT virtual void visit(const Min *, const Expr &);
  EXPORT virtual void visit(const Max *, const Expr &);
  EXPORT virtual void visit(const EQ *, const Expr &);
  EXPORT virtual void visit(const NE *, const Expr &);
  EXPORT virtual void visit(const LT *, const Expr &);
  EXPORT virtual void visit(const LE *, const Expr &);
  EXPORT virtual void visit(const GT *, const Expr &);
  EXPORT virtual void visit(const GE *, const Expr &);
  EXPORT virtual void visit(const And *, const Expr &);
  EXPORT virtual void visit(const Or *, const Expr &);
  EXPORT virtual void visit(const Not *, const Expr &);
  EXPORT virtual void visit(const Select *, const Expr &);
  EXPORT virtual void visit(const Load *, const Expr &);
  EXPORT virtual void visit(const Ramp *, const Expr &);
  EXPORT virtual void visit(const Broadcast *, const Expr &);
  EXPORT virtual void visit(const Call *, const Expr &);
  EXPORT virtual void visit(const Let *, const Expr &);
  EXPORT virtual void visit(const Shuffle *, const Expr &);
  EXPORT virtual void visit(const LetStmt *, const Stmt &);
  EXPORT virtual void visit(const AssertStmt *, const Stmt &);
  EXPORT virtual void visit(const ProducerConsumer *, const Stmt &);
  EXPORT virtual void visit(const For *, const Stmt &);
  EXPORT virtual void visit(const Store *, const Stmt &);
  EXPORT virtual void visit(const Provide *, const Stmt &);
  EXPORT virtual void visit(const Allocate *, const Stmt &);
  EXPORT virtual void visit(const Free *, const Stmt &);
  EXPORT virtual void visit(const Realize *, const Stmt &);
  EXPORT virtual void visit(const Prefetch *, const Stmt &);
  EXPORT virtual void visit(const Block *, const Stmt &);
  EXPORT virtual void visit(const IfThenElse *, const Stmt &);
  EXPORT virtual void visit(const Evaluate *, const Stmt &);
  EXPORT virtual void visit(const GetBit *, const Expr &);
  EXPORT virtual void visit(const GetSlice *, const Expr &);
  EXPORT virtual void visit(const SetBit *, const Expr &);
  EXPORT virtual void visit(const SetSlice *, const Expr &);
  EXPORT virtual void visit(const Quantize *, const Expr &);
  EXPORT virtual void visit(const KernelDef *, const Stmt &);
  EXPORT virtual void visit(const KernelExpr *, const Expr &);
  EXPORT virtual void visit(const KernelStmt *, const Stmt &);
  EXPORT virtual void visit(const Return *, const Stmt &);
  EXPORT virtual void visit(const Break *, const Stmt &);
  EXPORT virtual void visit(const While *, const Stmt &);
  EXPORT virtual void visit(const Reuse *, const Stmt &);
  EXPORT virtual void visit(const Partition *, const Stmt &);
  EXPORT virtual void visit(const Stencil *, const Stmt &);
  EXPORT virtual void visit(const ExternModule *, const Stmt &);
  EXPORT virtual void visit(const StreamExpr *, const Expr &);
  EXPORT virtual void visit(const StreamStmt *, const Stmt &);
  EXPORT virtual void visit(const Print *, const Stmt &);
  EXPORT virtual void visit(const MultiBlock *, const Stmt &);
  EXPORT virtual void visit(const Assert *, const Stmt &);
  // @}
};

}  // namespace Internal
}  // namespace Halide

#endif
