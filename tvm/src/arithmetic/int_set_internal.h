/*!
 *  Copyright (c) 2017 by Contributors
 * \file int_set_internal.h
 * \brief Implementations of integer set
 */
#ifndef ARITHMETIC_INT_SET_INTERNAL_H_
#define ARITHMETIC_INT_SET_INTERNAL_H_

#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

namespace TVM {
namespace arith {

using Halide::Internal::Interval;

/*! \brief Set of continuous interval */
struct IntervalSet : public IntSetNode {
  /*! \brief the internal interval*/
  Interval i;

  static IntSet make(Interval i) {
    std::shared_ptr<IntervalSet> n = std::make_shared<IntervalSet>();
    n->i = i;
    return IntSet(n);
  }
  static IntSet make(Expr min, Expr max) {
    std::shared_ptr<IntervalSet> n = std::make_shared<IntervalSet>();
    n->i.min = min;
    n->i.max = max;
    return IntSet(n);
  }

  static constexpr const char* _type_key = "IntervalSet";
  TVM_DECLARE_NODE_TYPE_INFO(IntervalSet, IntSetNode);
};

/*!
 * \brief set represented by strided integers
 *  Reserved for cases where strided access is supported.
 */
struct StrideSet : public IntSetNode {
  /*! \brief the base inetrval */
  Interval base;
  /*! \brief additional extents in positive number */
  Array<Expr> extents;
  /*! \brief additional strides in positive number */
  Array<Expr> strides;

  static constexpr const char* _type_key = "StrideSet";
  TVM_DECLARE_NODE_TYPE_INFO(StrideSet, IntSetNode);
};

/*!
 * \brief Set represented by range of ModularEntry.
 *  Used for front-end modular analysis.
 */
struct ModularSet : public IntSetNode {
  /*! \brief Internal modular entry */
  ModularEntry e;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("base", &(e.base));
    v->Visit("coeff", &(e.coeff));
  }
  static constexpr const char* _type_key = "ModularSet";
  TVM_DECLARE_NODE_TYPE_INFO(ModularSet, IntSetNode);
};

}  // namespace arith
}  // namespace TVM

#endif  // ARITHMETIC_INT_SET_INTERNAL_H_
