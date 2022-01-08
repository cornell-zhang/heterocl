/*!
 *  Copyright (c) 2016 by Contributors
 */
#include <type_traits>

#include "../ir/IROperator.h"
#include "Polynomial.h"
#include "Simplify.h"

namespace Halide {
namespace Internal {

using std::is_same;
using std::unordered_map;

// (x+y)*z -> x*z+y*z
template <typename MUL_OP, typename ADD_OP>
bool ExpandMutator::LeftExpand(Expr* expr, const Expr& a, const Expr& b) {
  if (a.as<ADD_OP>() != nullptr) {
    Expr x = a.as<ADD_OP>()->a;
    Expr y = a.as<ADD_OP>()->b;
    Expr z = b;
    *expr = mutate(ADD_OP::make(MUL_OP::make(x, z), MUL_OP::make(y, z)));
    return true;
  }
  return false;
}

// x*(y+z) -> x*y+x*z
template <typename MUL_OP, typename ADD_OP>
bool ExpandMutator::RightExpand(Expr* expr, const Expr& a, const Expr& b) {
  if (b.as<ADD_OP>() != nullptr) {
    Expr x = a;
    Expr y = b.as<ADD_OP>()->a;
    Expr z = b.as<ADD_OP>()->b;
    *expr = mutate(ADD_OP::make(MUL_OP::make(x, y), MUL_OP::make(x, z)));
    return true;
  }
  return false;
}

// x+(y+z) -> x+y+z
// x+(y-z) -> x+y-z
// x-(y+z) -> x-y-z
// x-(y-z) -> x-y+z
template <typename OP1, typename OP2>
bool ExpandMutator::Associate(Expr* expr, const Expr& a, const Expr& b) {
  if ((is_same<OP1, Add>::value || is_same<OP1, Sub>::value) &&
      (is_same<OP2, Add>::value || is_same<OP2, Sub>::value)) {
    if (b.as<OP2>() != nullptr) {
      Expr x = a;
      Expr y = b.as<OP2>()->a;
      Expr z = b.as<OP2>()->b;
      if (is_same<OP1, OP2>::value) {
        *expr = mutate(Add::make(OP1::make(x, y), z));
      } else {
        *expr = mutate(Sub::make(OP1::make(x, y), z));
      }
      return true;
    }
  }
  return false;
}

void ExpandMutator::visit(const Mul* op, const Expr& e) {
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  if (LeftExpand<Mul, Add>(&expr, a, b)) return;
  if (LeftExpand<Mul, Sub>(&expr, a, b)) return;
  if (RightExpand<Mul, Add>(&expr, a, b)) return;
  if (RightExpand<Mul, Sub>(&expr, a, b)) return;
  if (a.same_as(op->a) && b.same_as(op->b)) {
    expr = e;
  } else {
    expr = Mul::make(a, b);
  }
}

void ExpandMutator::visit(const Div* op, const Expr& e) {
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  if (LeftExpand<Div, Add>(&expr, a, b)) return;
  if (LeftExpand<Div, Sub>(&expr, a, b)) return;
  if (a.same_as(op->a) && b.same_as(op->b)) {
    expr = e;
  } else {
    expr = Div::make(a, b);
  }
}

// x+(y+z) -> x+y+z
// x+(y-z) -> x+y-z
void ExpandMutator::visit(const Add* op, const Expr& e) {
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  if (Associate<Add, Add>(&expr, a, b)) return;
  if (Associate<Add, Sub>(&expr, a, b)) return;
  if (a.same_as(op->a) && b.same_as(op->b)) {
    expr = e;
  } else {
    expr = Add::make(a, b);
  }
}

// x-(y+z) -> x-y-z
// x-(y-z) -> x-y+z
void ExpandMutator::visit(const Sub* op, const Expr& e) {
  Expr a = mutate(op->a);
  Expr b = mutate(op->b);

  if (Associate<Sub, Add>(&expr, a, b)) return;
  if (Associate<Sub, Sub>(&expr, a, b)) return;
  if (a.same_as(op->a) && b.same_as(op->b)) {
    expr = e;
  } else {
    expr = Sub::make(a, b);
  }
}

void IsAffineMutator::AffineVisitTerm(const Expr& e, bool positive) {
  const Expr operand = RemoveCast(e);
  VarExpr key;
  int64_t value = positive ? 1 : -1;
  bool is_affine = false;

  if (const IntImm* coeff = operand.as<IntImm>()) {
    // Constants.
    value *= coeff->value;
    is_affine = true;
  } else if (const Mul* op = operand.as<Mul>()) {
    // Terms with coefficient != 1.
    const Expr lhs = RemoveCast(op->a);
    const Expr rhs = RemoveCast(op->b);
    if (lhs.as<Variable>()) {
      if (const IntImm* coeff = rhs.as<IntImm>()) {
        key = VarExpr(lhs.node_);
        value *= coeff->value;
        is_affine = true;
      }
    }
    if (!is_affine) {
      is_affine_ = false;
      return;
    }
  } else if (operand.as<Variable>()) {
    // Terms with coefficient 1.
    key = VarExpr(operand.node_);
    is_affine = true;
  }

  if (is_affine) {
    if (coefficients_.count(key) == 0) {
      coefficients_[key] = value;
    } else {
      coefficients_[key] += value;
    }
    return;
  }
  is_affine_ = false;
}

void IsAffineMutator::AffineVisitAddOrSub(const Expr& e) {
  Expr next_e = e;
  for (;;) {
    if (const Add* next_op = next_e.as<Add>()) {
      next_e = next_op->a;
      AffineVisitTerm(next_op->b, true);
    } else if (const Sub* next_op = next_e.as<Sub>()) {
      next_e = next_op->a;
      AffineVisitTerm(next_op->b, false);
    } else {
      AffineVisitTerm(next_e, true);
      break;
    }
  }
}

VarExprInt64UnorderedMap GetAffineCoeff(const Expr& e) {
  IsAffineMutator mutator;
  mutator.mutate(Expand(simplify(e)));
  if (mutator.IsAffine()) {
    return mutator.GetCoefficients();
  }
  return VarExprInt64UnorderedMap();
}

}  // namespace Internal
}  // namespace Halide
