/*!
 *  Copyright (c) 2016 by Contributors
 */
#ifndef HALIDEIR_POLYNOMIAL_H
#define HALIDEIR_POLYNOMIAL_H

#include <unordered_map>

#include "ir/IR.h"
#include "ir/IRMutator.h"

/** \file
 * Methods for handling polynomials in HalideIR
 */

namespace Halide {
namespace Internal {

template <typename T>
inline bool MapEqual(const T& lhs, const T& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto elem : lhs) {
    if (rhs.count(elem.first) == 0) return false;
    if (rhs.at(elem.first) != elem.second) return false;
  }
  return true;
}

typedef std::unordered_map<VarExpr, int64_t, ExprHash, ExprEqual>
    VarExprInt64UnorderedMap;
inline bool operator==(const VarExprInt64UnorderedMap& lhs,
                       const VarExprInt64UnorderedMap& rhs) {
  return MapEqual(lhs, rhs);
}
inline bool operator!=(const VarExprInt64UnorderedMap& lhs,
                       const VarExprInt64UnorderedMap& rhs) {
  return !MapEqual(lhs, rhs);
}

class ExpandMutator : public IRMutator {
  template <typename MUL_OP, typename ADD_OP>
  bool LeftExpand(Expr* expr, const Expr& a, const Expr& b);
  template <typename MUL_OP, typename ADD_OP>
  bool RightExpand(Expr* expr, const Expr& a, const Expr& b);
  template <typename OP1, typename OP2>
  bool Associate(Expr* expr, const Expr& a, const Expr& b);

  void visit(const Mul* op, const Expr& e);
  void visit(const Div* op, const Expr& e);
  void visit(const Add* op, const Expr& e);
  void visit(const Sub* op, const Expr& e);

  using IRMutator::visit;
};

class IsAffineMutator : public IRMutator {
  VarExprInt64UnorderedMap coefficients_;
  bool is_affine_ = true;

  void AffineVisitTerm(const Expr& e, bool positive);
  void AffineVisitAddOrSub(const Expr& e);

  void visit(const Add* op, const Expr& e) { AffineVisitAddOrSub(e); }
  void visit(const Sub* op, const Expr& e) { AffineVisitAddOrSub(e); }

  using IRMutator::visit;

 public:
  VarExprInt64UnorderedMap GetCoefficients() const { return coefficients_; }
  bool IsAffine() const { return is_affine_; }
};

inline Expr Expand(const Expr& e) { return ExpandMutator().mutate(e); }
inline Expr RemoveCast(const Expr& e) {
  if (const Cast* cast = e.as<Cast>()) return cast->value;
  return e;
}

// Maps a variable to its coefficent, e.g. a*2+b*3+5.
VarExprInt64UnorderedMap GetAffineCoeff(const Expr& e);

}  // namespace Internal
}  // namespace Halide

#endif
