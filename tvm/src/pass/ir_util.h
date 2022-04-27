/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_util.h
 * \brief Helper functions to construct and compose IR nodes.
 */
#ifndef PASS_IR_UTIL_H_
#define PASS_IR_UTIL_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/runtime/device_api.h>
#include <vector>
#include "../codegen/build_common.h"

namespace TVM {
namespace ir {
/*!
 * \brief combine the nest stmt, whose body is not defined.
 * \param nest A list of For and LetStmt, whose body is not defined.
 * \param body body
 * \return The combined Stmt
 */
Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body);

/*!
 * \brief combine the nest stmt, whose body is not defined.
 * \param nest A list of For and LetStmt, whose body is not defined.
 * \param body body
 * \return The combined Stmt
 */
Stmt MergeNest(const std::vector<std::vector<Stmt> >& nest, Stmt body);

/*!
 * \brief combine sequence of operations.
 * \param seq The sequence.
 * \return The combined Stmt
 */
Stmt MergeSeq(const std::vector<Stmt>& seq);

/*!
 * \brief update array with an unary function
 * \param arr array
 * \param fupdate an unary function
 * \tparam T type of array element
 * \tparam F type of the unary function
 * \return if update happens, return the new array, else return the
 *  original array
 */
template <typename T, typename F>
inline Array<T> UpdateArray(Array<T> arr, F fupdate) {
  std::vector<T> new_arr(arr.size());
  bool changed = false;
  for (size_t i = 0; i < arr.size(); ++i) {
    T old_elem = arr[i];
    T new_elem = fupdate(old_elem);
    if (!new_elem.same_as(old_elem)) changed = true;
    new_arr[i] = new_elem;
  }
  if (!changed) {
    return arr;
  } else {
    return Array<T>(new_arr);
  }
}

/*!
 * \brief Get construct from struct
 * \param dtype The data type.
 * \param handle the struct handle.
 * \param index the offset index.
 * \param kind The data kind.
 * \return the get expression.
 */
inline Expr TVMStructGet(Type dtype, Var handle, int index,
                         intrinsic::TVMStructFieldKind kind) {
  Array<Expr> args = {handle, make_const(Int(32), index),
                      make_const(Int(32), kind)};
  return Call::make(dtype, intrinsic::tvm_struct_get, args,
                    Call::PureIntrinsic);
}

/*!
 * \brief Address of handle + offset
 * \param handle the array handle.
 * \param dtype The data type.
 * \param offset the offset index.
 */
inline Expr AddressOffset(Var handle, Type dtype, int offset) {
  return Call::make(
      Handle(), intrinsic::tvm_address_of,
      {Load::make(dtype, handle, make_const(Int(32), offset * dtype.lanes()),
                  const_true(dtype.lanes()))},
      Call::PureIntrinsic);
}

/*!
 * \brief Address of handle + offset
 * \param handle the array handle.
 * \param dtype The data type.
 * \param offset the offset index.
 */
inline Expr AddressOffset(Var handle, Type dtype, Expr offset) {
  if (dtype.lanes() != 1) {
    offset = offset * make_const(offset.type(), dtype.lanes());
    offset = Ramp::make(offset, make_const(offset.type(), 1), dtype.lanes());
  }
  return Call::make(
      Handle(), intrinsic::tvm_address_of,
      {Load::make(dtype, handle, offset, const_true(dtype.lanes()))},
      Call::PureIntrinsic);
}

/*!
 * \brief Set value into struct.
 * \param handle the struct handle.
 * \param index the offset index.
 * \param kind The data kind.
 * \param value The value to be set.
 * \return the set stmt.
 */
inline Stmt TVMStructSet(Var handle, int index,
                         intrinsic::TVMStructFieldKind kind, Expr value) {
  Array<Expr> args = {handle, make_const(Int(32), index),
                      make_const(Int(32), kind), value};
  return Evaluate::make(
      Call::make(Int(32), intrinsic::tvm_struct_set, args, Call::Intrinsic));
}

/*!
 * \brief Get the type that is passed around TVM PackedFunc API.
 * \param t The original type.
 * \return The corresponding API type.
 */
inline Type APIType(Type t) {
  if (t.is_handle()) return t;
  CHECK_EQ(t.lanes(), 1) << "Cannot pass vector type through packed API.";
  if (t.is_ufixed() || t.is_fixed()) return Int(64);
  CHECK(t.is_float());
  return Float(64);
}

/*!
 * \brief Rule to get allocation alignment requirement for a given const array.
 * \param type The type of allocation.
 * \param const_size The constant size of the array.
 * \return the alignment
 */
inline int GetTempAllocaAlignment(Type type, int32_t const_size) {
  int align = runtime::kTempAllocaAlignment;
  if (const_size > 0) {
    int64_t const_s =
        static_cast<int64_t>(const_size) * type.bits() * type.lanes() / 8;
    while (align > const_s) {
      align = align / 2;
    }
  }
  return align;
}

// Collect function call arguments names and types.
// Visit KernelExpr and KernelDef nodes in the IR.
// Example:
//  HierarchyVisitor hierarchy_visitor;
//  hierarchy_visitor.Visit(stmt);
//  std::list<std::string> defs = hierarchy_visitor.get_submodule_def();
//  std::list<std::string> calls = hierarchy_visitor.get_submodules();
//  std::map<std::string, std::vector<Expr>> submodule_args =
//      hierarchy_visitor.get_submodule_args();
//  std::map<std::string, std::vector<Expr>> submodule_arg_types =
//      hierarchy_visitor.get_arg_types();
//  std::map<std::string, std::vector<std::string>> submodule_arg_names =
//      hierarchy_visitor.get_arg_names();
class HierarchyVisitor : public IRVisitor {
 public:
  HierarchyVisitor() : _args{}, _arg_types{}, _arg_names{} {}

  void Visit_(const KernelExpr* op) final {
    _call_stack.push_back(op->name);
    // collect args info
    for (Expr arg : op->args) {
      _args[op->name].push_back(arg);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const KernelDef* op) final {
    _def_list.push_back(op->name);
    for (unsigned int i = 0; i < op->args.size(); i++) {
      auto e = op->arg_types[i];
      _arg_types[op->name].push_back(e);
      VarExpr arg = op->args[i];
      std::string n = arg.as<Variable>()->name_hint;
      _arg_names[op->name].push_back(n);
    }
  }

  std::list<std::string> get_submodule_def() { return _def_list; }

  std::list<std::string> get_submodules() { return _call_stack; }

  std::map<std::string, std::vector<Expr> > get_submodule_args() {
    return _args;
  }

  std::map<std::string, std::vector<Expr> > get_arg_types() {
    return _arg_types;
  }

  std::map<std::string, std::vector<std::string> > get_arg_names() {
    return _arg_names;
  }

 private:
  std::list<std::string> _call_stack;
  std::list<std::string> _def_list;
  // from KernelExpr
  std::map<std::string, std::vector<Expr> > _args;
  // from KernelDef
  std::map<std::string, std::vector<Expr> > _arg_types;
  std::map<std::string, std::vector<std::string> > _arg_names;
};


enum class PortType { ChannelIn, ChannelOut, Memory, OffChipMemory, Default };

// Get port direction for function arguments
// StratusHLS backend requires port direction for function arguments
// to be specified.
// Example:
//  PortDirectionFinder port_visitor(ports, scalars);
//  port_visitor.Visit(Stmt);
//  PortType port_type = port_visitor.get_direction(arg_name);
class PortDirectionFinder : public IRVisitor {
 public:
  explicit PortDirectionFinder(const std::vector<std::string>& ports,
                         const std::vector<std::string>& scalars)
      : _ports(ports), _scalars(scalars) {}

  void Visit_(const Load* op) final {
    std::string var_name = op->buffer_var.get()->name_hint;
    TVM::codegen::canonicalize_string(var_name);
    auto it = std::find(_ports.begin(), _ports.end(), var_name);
    if (it != _ports.end()) {
      _in_ports.push_back(var_name);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store* op) final {
    std::string var_name = op->buffer_var.get()->name_hint;
    TVM::codegen::canonicalize_string(var_name);
    auto it = std::find(_ports.begin(), _ports.end(), var_name);
    if (it != _ports.end()) {
      _out_ports.push_back(var_name);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Cast* op) final {
    if (const Variable* v = op->value.as<Variable>()) {
      std::string var_name = v->name_hint;
      TVM::codegen::canonicalize_string(var_name);
      auto it = std::find(_ports.begin(), _ports.end(), var_name);
      if (it != _ports.end()) {
        _in_ports.push_back(var_name);
      }
    }
    IRVisitor::Visit_(op);
  }

  bool isOffChip(std::string name);
  PortType get_direction(std::string var_name);

 private:
  std::list<std::string> _in_ports;
  std::list<std::string> _out_ports;
  std::vector<std::string> _ports;
  std::vector<std::string> _scalars;
};
}  // namespace ir
}  // namespace TVM
#endif  // PASS_IR_UTIL_H_
