/*!
 *  Copyright (c) 2017 by Contributors
 * \file op_util.h
 * \brief Common utility used in operator construction.
 */
#ifndef OP_OP_UTIL_H_
#define OP_OP_UTIL_H_

#include <tvm/expr.h>
#include <tvm/schedule.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../pass/arg_binder.h"
#include "../pass/ir_util.h"

namespace TVM {
namespace op {

using ir::MergeNest;

/*!
 * \brief Create a nest of if checking the predicates.
 *
 * \param predicates The predicates to be checked.
 * \return List of If nest that checks the predicates.
 */
std::vector<Stmt> MakeIfNest(const std::vector<Expr>& predicates);

/*!
 * \brief Replace the tensor reference in stmt by the replace map.
 * \param stmt The statement to be processed.
 * \param replace The replacement rule.
 */
Stmt ReplaceTensor(Stmt stmt,
                   const std::unordered_map<Tensor, Tensor>& replace);
/*!
 * \brief Replace the tensor reference in expr by the replace map.
 * \param expr The expression to be processed.
 * \param replace The replacement rule.
 */
Expr ReplaceTensor(Expr expr,
                   const std::unordered_map<Tensor, Tensor>& replace);

/*!
 * \brief Substitute the variables of stmt by value map.
 * \param stmt the statment
 * \param value_map The value map.
 * \return Substituted result.
 */
Stmt Substitute(Stmt stmt, const std::unordered_map<IterVar, Expr>& value_map);

Stmt Substitute(Stmt stmt,
                const std::unordered_map<const Variable*, Expr>& value_map);

}  // namespace op
}  // namespace TVM
#endif  // OP_OP_UTIL_H_
