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
/*!
 * \brief Count the attach level of extern op.
 * \param stage The schedule stage.
 * \return The attach level.
 */
//int CountAttachLevel(const Stage& stage);

/*!
 * \brief Get the iter vars that remain in the inner stage,
 *  which is attached to another stage. Other iter vars in
 *  the index of Store will be deleted.
 * \param stage The schedule stage.
 * \param axis_size The axis size.
 * \param attach_level The attach level.
 * \return The iter vars that remain in the inner stage,
 *  associated with buffer var.
 */
std::unordered_map<const Variable*, std::vector<IterVar> >
GetAxisInnerStoreRemain(const Stage& stage, int axis_size, int attach_level);

/*!
 * \brief Get the iter vars that remain in the outer stage,
 *  which is attached by another stage. Other iter vars in
 *  the index of Load will be deleted.
 * \param stage The schedule stage.
 * \param axis_size The axis size.
 * \param attach_level The attach level.
 * \return The iter vars that remain in the outer stage,
 *  associated with buffer var. 
 */
std::unordered_map<const Variable*, std::vector<IterVar> >
GetAxisOuterLoadRemain(const Stage& stage, int axis_size, int attach_level);

/*!
 * \brief Get the iter vars to substitute after attachment.
 * \param stage The schedule stage.
 * \param axis_size The axis size.
 * \param attach_level The attach level.
 * \return The iter vars to substitute.
 */
std::unordered_map<const Variable*, Expr>
GetVarsInnerLoadSub(const Stage& stage, int axis_size, int attach_level);

/*!
 * \brief Get the iter vars that remain in index after attachment.
 * \param index The index expr.
 * \param iv_remain The iter vars that remain.
 * \return The intersection of iv_remain and vars in index.
 */
std::vector<IterVar>
GetIterVarsInIndexRemain(Expr index, std::vector<IterVar> iv_remain);

/*!
 * \brief Make index expr from iter vars.
 * \param vars The iter vars.
 * \return The index expr.
 */
Expr MakeIndexFromIterVars(std::vector<IterVar> vars);

}  // namespace tvm

#endif  // TVM_OP_EXTERN_OP_H_
