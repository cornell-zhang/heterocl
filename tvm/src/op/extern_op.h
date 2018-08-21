/*!
 *  Copyright (c) 2018 by Contributors
 * \brief Helper utilities to implement extern_op.
 * \file extern_op.h
 */
#ifndef TVM_OP_EXTERN_OP_H_
#define TVM_OP_EXTERN_OP_H_

#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <unordered_map>

namespace tvm {
// BufferVar -> (deleted IterVar -> 0)
using VarsDeleteOuterMap = std::unordered_map<const Variable*, std::unordered_map<const Variable*, Expr> >;

/*!
 * \brief Count the attach level of extern op.
 * \param stage The schedule stage.
 * \return The attach level.
 */
int CountAttachLevel(const Stage& stage);

/*!
 * \brief Get the IterVars to delete in the inner stage,
    which is attached to another stage.
 * \param stage The schedule stage.
 * \param axis_size The axis size.
 * \param attach_level The attach level.
 * \return The IterVars to delete in the inner stage.
 */
std::unordered_map<const Variable*, Expr>
GetVarsDeleteInner(const Stage& stage, int axis_size, int attach_level);

/*!
 * \brief Get the IterVars to delete in the outer stage,
    which is attached by another stage.
 * \param stage The schedule stage.
 * \param axis_size The axis size.
 * \param attach_level The attach level.
 * \return The IterVars to delete in the outer stage.
 */
VarsDeleteOuterMap
GetVarsDeleteOuter(const Stage& stage, int axis_size, int attach_level);

/*!
 * \brief Get the IterVars to substitute.
 * \param stage The schedule stage.
 * \param axis_size The axis size.
 * \return The IterVars to substitute.
 */
std::unordered_map<const Variable*, Expr>
GetVarsSub(const Stage& stage, int axis_size, int attach_level);

}  // namespace tvm

#endif  // TVM_OP_EXTERN_OP_H_
