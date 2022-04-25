/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_util.h
 * \brief Helper functions to construct and compose IR nodes.
 */
#ifndef PASS_IR_UTIL_H_
#define PASS_IR_UTIL_H_

#include <tvm/ir.h>
#include <tvm/runtime/device_api.h>
#include <tvm/ir_mutator.h>
#include <vector>

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

// Remove cast in binary expressions
// Example: (cast(x) + cast(y)) -> (x + y)
// Usage: CastRemover castRemover;
//        Expr expr = castRemover.Mutate(expr);
class CastRemover final : public IRMutator {
 public:
  CastRemover() {}

  Expr Mutate_(const Cast* op, const Expr& e) {
    return op->value;
  }

  Expr Mutate_(const Add* op, const Expr& e) {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (const Cast* ca = a.as<Cast>()) {
      a = ca->value;
    }
    if (const Cast* cb = b.as<Cast>()) {
      b = cb->value;
    }
    if (a.type() != b.type())
      LOG(FATAL) << "CastRemover: type mismatch "
        << a.type() << " vs " << b.type();
    return Add::make(a, b);
  }

  Expr Mutate_(const Sub* op, const Expr& e) {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (const Cast* ca = a.as<Cast>()) {
      a = ca->value;
    }
    if (const Cast* cb = b.as<Cast>()) {
      b = cb->value;
    }
    if (a.type() != b.type())
      LOG(FATAL) << "CastRemover: type mismatch "
        << a.type() << " vs " << b.type();
    return Sub::make(a, b);
  }

  Expr Mutate_(const Mul* op, const Expr& e) {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (const Cast* ca = a.as<Cast>()) {
      a = ca->value;
    }
    if (const Cast* cb = b.as<Cast>()) {
      b = cb->value;
    }
    if (a.type() != b.type())
      LOG(FATAL) << "CastRemover: type mismatch "
        << a.type() << " vs " << b.type();
    return Mul::make(a, b);
  }

  Expr Mutate_(const Div* op, const Expr& e) {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (const Cast* ca = a.as<Cast>()) {
      a = ca->value;
    }
    if (const Cast* cb = b.as<Cast>()) {
      b = cb->value;
    }
    if (a.type() != b.type())
      LOG(FATAL) << "CastRemover: type mismatch "
        << a.type() << " vs " << b.type();
    return Div::make(a, b);
  }

  Expr Mutate_(const Mod* op, const Expr& e) {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (const Cast* ca = a.as<Cast>()) {
      a = ca->value;
    }
    if (const Cast* cb = b.as<Cast>()) {
      b = cb->value;
    }
    if (a.type() != b.type())
      LOG(FATAL) << "CastRemover: type mismatch "
        << a.type() << " vs " << b.type();
    return Mod::make(a, b);
  }

  Expr Mutate_(const Min* op, const Expr& e) {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (const Cast* ca = a.as<Cast>()) {
      a = ca->value;
    }
    if (const Cast* cb = b.as<Cast>()) {
      b = cb->value;
    }
    if (a.type() != b.type())
      LOG(FATAL) << "CastRemover: type mismatch "
        << a.type() << " vs " << b.type();
    return Min::make(a, b);
  }

  Expr Mutate_(const Max* op, const Expr& e) {
    Expr a = this->Mutate(op->a);
    Expr b = this->Mutate(op->b);
    if (const Cast* ca = a.as<Cast>()) {
      a = ca->value;
    }
    if (const Cast* cb = b.as<Cast>()) {
      b = cb->value;
    }
    if (a.type() != b.type())
      LOG(FATAL) << "CastRemover: type mismatch "
        << a.type() << " vs " << b.type();
    return Max::make(a, b);
  }
};
}  // namespace ir
}  // namespace TVM
#endif  // PASS_IR_UTIL_H_
