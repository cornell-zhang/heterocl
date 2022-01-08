/*!
 *  Copyright (c) 2021 by Contributors
 * \file tvm/src/codegen/code_analysis.cc
 */
#include "code_analysis.h"
#include <cctype>
#include <iomanip>
#include <regex>
#include "./../arithmetic/compute_expr.h"
#include "./codegen_common.h"

namespace TVM {
namespace codegen {

using namespace ir;

void CodeAnalysis::Init() {}

void CodeAnalysis::InitFuncState(LoweredFunc f) {
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  map_arg_type_.clear();
  CodeGenSourceBase::ClearFuncState();
}
void CodeAnalysis::AddFunction(LoweredFunc f) {
  // Clear previous generated state.
  this->InitFuncState(f);

  // Add to alloc buffer type.
  for (const auto& kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  // Record the arguments for analyzing the type
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    std::string vid = AllocVarID(v.get());
    this->map_arg_type_[vid];
  }
  int func_scope = this->BeginScope();
  VisitStmt(f->body);
  this->EndScope(func_scope);
}

str2tupleMap<std::string, Type> CodeAnalysis::Finish() {
  return this->map_arg_type_;
}

void CodeAnalysis::PrintExpr(const Expr& n,
                             std::ostream& os) {  // NOLINT(*)
  VisitExpr(n, os);
}

void CodeAnalysis::PrintSSAAssign(const std::string& target,
                                  const std::string& src, Type t) {
  PrintType(t, stream);
  stream << ' ' << target << " = ";
  if (src.length() > 3 && src[0] == '(' && src[src.length() - 1] == ')') {
    stream << src.substr(1, src.length() - 2);
  } else {
    stream << src;
  }
  stream << ";\n";
}

// Print a reference expression to a buffer.
std::string CodeAnalysis::GetBufferRef(Type t, const Variable* buffer,
                                       Expr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  std::string scope;
  if (alloc_storage_scope_.count(buffer)) {
    scope = alloc_storage_scope_.at(buffer);
  }
  bool is_vol = volatile_buf_.count(buffer) != 0;
  if (t.lanes() == 1) {
    if (!HandleTypeMatch(buffer, t) || is_vol) {
      os << "((";
      if (is_vol) {
        os << "volatile ";
      }
      if (scope.length() != 0) {
        PrintStorageScope(scope, os);
      }
      os << ' ';
      PrintType(t, os);
      os << "*)" << vid << ')';
    } else {
      os << vid;
    }
    os << '[';
    PrintExpr(index, os);
    os << ']';
  } else {
    // Buffer declared as vector type.
    // optimize for case where it is in register,
    if (HandleTypeMatch(buffer, t) && !is_vol) {
      // optimize for constant access
      int offset;
      if (arith::GetConstInt(index, &offset)) {
        CHECK_EQ(offset % t.lanes(), 0)
            << "Find unaligned vector load to a vector type";
        os << vid << '[' << (offset / t.lanes()) << ']';
        return os.str();
      }
    }
    os << "((";
    if (is_vol) {
      os << "volatile ";
    }
    if (scope.length() != 0) {
      PrintStorageScope(scope, os);
    }
    os << ' ';
    PrintType(t, os);
    os << "*)(";
    if (!HandleTypeMatch(buffer, t.element_of())) {
      os << '(';
      if (scope.length() != 0) {
        PrintStorageScope(scope, os);
      }
      os << ' ';
      PrintType(t.element_of(), os);
      os << "*)";
    }
    os << vid << " + ";
    PrintExpr(index, os);
    os << "))[0]";
  }
  return os.str();
}

// Print a reference expression to a buffer.
std::string CodeAnalysis::GetStructRef(Type t, const Expr& buffer,
                                       const Expr& index, int kind) {
  if (kind < intrinsic::kArrKindBound_) {
    std::ostringstream os;
    os << "(((TVMArray*)";
    this->PrintExpr(buffer, os);
    os << ")";
    if (kind == intrinsic::kArrAddr) {
      os << " + ";
      this->PrintExpr(index, os);
      os << ")";
      return os.str();
    }
    os << '[';
    this->PrintExpr(index, os);
    os << "].";
    // other case: get fields.
    switch (kind) {
      case intrinsic::kArrData:
        os << "data";
        break;
      case intrinsic::kArrShape:
        os << "shape";
        break;
      case intrinsic::kArrStrides:
        os << "strides";
        break;
      case intrinsic::kArrNDim:
        os << "ndim";
        break;
      case intrinsic::kArrTypeCode:
        os << "dtype.code";
        break;
      case intrinsic::kArrTypeBits:
        os << "dtype.bits";
        break;
      case intrinsic::kArrTypeLanes:
        os << "dtype.lanes";
        break;
      case intrinsic::kArrTypeFracs:
        os << "dtype.fracs";
        break;
      case intrinsic::kArrDeviceId:
        os << "ctx.device_id";
        break;
      case intrinsic::kArrDeviceType:
        os << "ctx.device_type";
        break;
      default:
        os << "unknown_field_code_" << kind;
    }
    os << ')';
    return os.str();
  } else {
    CHECK_LT(kind, intrinsic::kTVMValueKindBound_);
    std::ostringstream os;
    os << "(((TVMValue*)";
    this->PrintExpr(buffer, os);
    os << ")[" << index << "].";
    if (t.is_handle()) {
      os << "v_handle";
    } else if (t.is_float()) {
      os << "v_float64";
    } else if (t.is_int()) {
      os << "v_int64";
    } else {
      os << t;
    }
    os << ")";
    return os.str();
  }
}

bool CodeAnalysis::HandleTypeMatch(const Variable* buf_var, Type t) const {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) return false;
  return it->second == t;
}

void CodeAnalysis::RegisterHandleType(const Variable* buf_var, Type t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_[buf_var] = t;
  } else {
    CHECK(it->second == t) << "conflicting buf var type";
  }
}

void CodeAnalysis::PrintVecElemLoad(const std::string& vec, Type t, int i,
                                    std::ostream& os) {  // NOLINT(*)
  os << vec << ".s" << std::hex << i << std::dec;
}

void CodeAnalysis::PrintVecElemStore(const std::string& vec, Type t, int i,
                                     const std::string& value) {
  this->PrintIndent();
  stream << vec << ".s" << std::hex << i << " = " << value << ";\n" << std::dec;
}

std::string CodeAnalysis::GetVecLoad(Type t, const Variable* buffer,
                                     Expr base) {
  return GetBufferRef(t, buffer, base);
}

void CodeAnalysis::PrintVecStore(const Variable* buffer, Type t, Expr base,
                                 const std::string& value) {
  std::string ref = GetBufferRef(t, buffer, base);
  this->PrintIndent();
  stream << ref << " = " << value << ";\n";
}

std::string CodeAnalysis::CastFromTo(std::string value, Type from,
                                     Type target) {
  if (from == target) return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")" << value << ")";
  return os.str();
}

void CodeAnalysis::BindThreadIndex(const IterVar& iv) {
  LOG(FATAL) << "not implemented";
}

void CodeAnalysis::PrintStorageSync(const Call* op) {  // NOLINT(*)
}

void CodeAnalysis::PrintStorageScope(const std::string& scope,
                                     std::ostream& os) {  // NOLINT(*)
}

std::string CodeAnalysis::GetType(Type t) {  // NOLINT(*)
  std::ostringstream os;
  PrintType(t, os);
  return os.str();
}

void CodeAnalysis::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  CHECK_EQ(t.lanes(), 1) << "do not yet support vector types";
  if (t.is_handle()) {
    os << "void*";
    return;
  }
  if (t.is_float()) {
    if (t.bits() == 32) {
      os << "float";
      return;
    }
    if (t.bits() == 64) {
      os << "double";
      return;
    }
  } else if (t.is_uint()) {
    switch (t.bits()) {
      case 8:
      case 16:
      case 32:
      case 64: {
        os << "uint" << t.bits() << "_t";
        return;
      }
      case 1:
        os << "int";
        return;
    }
  } else if (t.is_int()) {
    switch (t.bits()) {
      case 8:
      case 16:
      case 32:
      case 64: {
        os << "int" << t.bits() << "_t";
        return;
      }
    }
  }
  os << t;
}

inline void PrintConst(const IntImm* op, std::ostream& os,
                       CodeAnalysis* p) {  // NOLINT(*)
  if (op->type == Int(32)) {
    std::ostringstream temp;
    temp << op->value;
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(op->type, os);
    os << ")" << op->value;
  }
}

inline void PrintConst(const UIntImm* op, std::ostream& os,
                       CodeAnalysis* p) {  // NOLINT(*)
  if (op->type == UInt(32)) {
    std::ostringstream temp;
    temp << op->value << "U";
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(op->type, os);
    os << ")" << op->value;
  }
}

inline void PrintConst(const FloatImm* op, std::ostream& os,
                       CodeAnalysis* p) {  // NOLINT(*)
  switch (op->type.bits()) {
    case 64:
    case 32: {
      std::ostringstream temp;
      temp << std::scientific << op->value;
      if (op->type.bits() == 32) temp << 'f';
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << '(';
      p->PrintType(op->type, os);
      os << ')' << std::scientific << op->value << 'f';
      break;
    }
    default:
      os << op << "\n";
  }
}

void CodeAnalysis::VisitExpr_(const IntImm* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeAnalysis::VisitExpr_(const UIntImm* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeAnalysis::VisitExpr_(const FloatImm* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeAnalysis::VisitExpr_(const StringImm* op,
                              std::ostream& os) {  // NOLINT(*)
  os << "\"" << op->value << "\"";
}

template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeAnalysis* p) {
  if (op->type.lanes() == 1) {
    if (isalpha(opstr[0])) {
      os << opstr << '(';
      p->PrintExpr(op->a, os);
      os << ", ";
      p->PrintExpr(op->b, os);
      os << ')';
    } else {
      os << '(';
      p->PrintExpr(op->a, os);
      os << ' ' << opstr << ' ';
      p->PrintExpr(op->b, os);
      os << ')';
    }
  } else {
    p->PrintVecBinaryOp(opstr, op->type, op->a, op->b, os);
  }
}

inline void PrintBinaryIntrinsitc(const Call* op, const char* opstr,
                                  std::ostream& os,  // NOLINT(*)
                                  CodeAnalysis* p) {
  if (op->type.lanes() == 1) {
    CHECK_EQ(op->args.size(), 2U);
    os << '(';
    p->PrintExpr(op->args[0], os);
    os << opstr;
    p->PrintExpr(op->args[1], os);
    os << ')';
  } else {
    p->PrintVecBinaryOp(opstr, op->type, op->args[0], op->args[1], os);
  }
}
void CodeAnalysis::VisitExpr_(const Cast* op,
                              std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  this->PrintExpr(op->value, value);
  os << CastFromTo(value.str(), op->value.type(), op->type);
}
void CodeAnalysis::VisitExpr_(const Variable* op,
                              std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}
void CodeAnalysis::VisitExpr_(const Add* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
void CodeAnalysis::VisitExpr_(const Sub* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
void CodeAnalysis::VisitExpr_(const Mul* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}
void CodeAnalysis::VisitExpr_(const Div* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}
void CodeAnalysis::VisitExpr_(const Mod* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}
void CodeAnalysis::VisitExpr_(const Min* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "min", os, this);
}
void CodeAnalysis::VisitExpr_(const Max* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "max", os, this);
}
void CodeAnalysis::VisitExpr_(const EQ* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "==", os, this);
}
void CodeAnalysis::VisitExpr_(const NE* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "!=", os, this);
}
void CodeAnalysis::VisitExpr_(const LT* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<", os, this);
}
void CodeAnalysis::VisitExpr_(const LE* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<=", os, this);
}
void CodeAnalysis::VisitExpr_(const GT* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">", os, this);
}
void CodeAnalysis::VisitExpr_(const GE* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">=", os, this);
}
void CodeAnalysis::VisitExpr_(const And* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "&&", os, this);
}
void CodeAnalysis::VisitExpr_(const Or* op,
                              std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "||", os, this);
}
void CodeAnalysis::VisitExpr_(const Not* op,
                              std::ostream& os) {  // NOLINT(*)
  os << '!';
  PrintExpr(op->a, os);
}

void CodeAnalysis::VisitExpr_(const Call* op,
                              std::ostream& os) {  // NOLINT(*)
  if (op->call_type == Call::Extern || op->call_type == Call::PureExtern) {
    os << op->name << "(";
    for (size_t i = 0; i < op->args.size(); i++) {
      this->PrintExpr(op->args[i], os);
      if (i < op->args.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
  } else if (op->is_intrinsic(Call::bitwise_and)) {
    PrintBinaryIntrinsitc(op, " & ", os, this);
  } else if (op->is_intrinsic(Call::bitwise_xor)) {
    PrintBinaryIntrinsitc(op, " ^ ", os, this);
  } else if (op->is_intrinsic(Call::bitwise_or)) {
    PrintBinaryIntrinsitc(op, " | ", os, this);
  } else if (op->is_intrinsic(Call::bitwise_not)) {
    CHECK_EQ(op->args.size(), 1U);
    os << "(~";
    this->PrintExpr(op->args[0], os);
    os << ')';
  } else if (op->is_intrinsic(Call::shift_left)) {
    PrintBinaryIntrinsitc(op, " << ", os, this);
  } else if (op->is_intrinsic(Call::shift_right)) {
    PrintBinaryIntrinsitc(op, " >> ", os, this);
  } else if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
    os << "(";
    PrintExpr(op->args[0], os);
    os << " ? ";
    PrintExpr(op->args[1], os);
    os << " : ";
    PrintExpr(op->args[2], os);
    os << ")";
  } else if (op->is_intrinsic(intrinsic::tvm_address_of)) {
    const Load* l = op->args[0].as<Load>();
    CHECK(op->args.size() == 1 && l);
    os << "((";
    this->PrintType(l->type.element_of(), os);
    os << " *)" << this->GetVarID(l->buffer_var.get()) << " + ";
    this->PrintExpr(l->index, os);
    os << ')';
  } else if (op->is_intrinsic(intrinsic::tvm_struct_get)) {
    CHECK_EQ(op->args.size(), 3U);
    os << GetStructRef(op->type, op->args[0], op->args[1],
                       op->args[2].as<IntImm>()->value);
  } else if (op->is_intrinsic(intrinsic::tvm_handle_is_null)) {
    CHECK_EQ(op->args.size(), 1U);
    os << "(";
    this->PrintExpr(op->args[0], os);
    os << " == NULL)";
  } else {
    os << op->name << "()";
  }
}

void CodeAnalysis::PrintVecBinaryOp(const std::string& op, Type t, Expr lhs,
                                    Expr rhs,
                                    std::ostream& os) {  // NOLINT(*)
  if (isalpha(op[0])) {
    os << op << "(";
    this->PrintExpr(lhs, os);
    os << ", ";
    this->PrintExpr(rhs, os);
    os << ")";
  } else {
    os << "(";
    this->PrintExpr(lhs, os);
    os << ' ' << op << ' ';
    this->PrintExpr(rhs, os);
    os << ")";
  }
}

inline bool TryGetRamp1Base(Expr index, int lanes, Expr* base) {
  const Ramp* r = index.as<Ramp>();
  if (!r) return false;
  if (!is_one(r->stride)) return false;
  CHECK_EQ(r->lanes, lanes);
  *base = r->base;
  return true;
}

void CodeAnalysis::VisitExpr_(const Load* op,
                              std::ostream& os) {  // NOLINT(*)
  int lanes = op->type.lanes();
  // delcare type.
  if (op->type.lanes() == 1) {
    std::string ref = GetBufferRef(op->type, op->buffer_var.get(), op->index);
    os << ref;
  } else {
    CHECK(is_one(op->predicate)) << "predicated load is not supported";
    Expr base;
    if (TryGetRamp1Base(op->index, op->type.lanes(), &base)) {
      std::string ref = GetVecLoad(op->type, op->buffer_var.get(), base);
      os << ref;
    } else {
      // The assignment below introduces side-effect, and the resulting value
      // cannot be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // load seperately.
      std::string svalue = GetUniqueName("_");
      this->PrintIndent();
      this->PrintType(op->type, stream);
      stream << ' ' << svalue << ";\n";
      std::string sindex = SSAGetID(PrintExpr(op->index), op->index.type());
      std::string vid = GetVarID(op->buffer_var.get());
      Type elem_type = op->type.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (op->buffer_var.get()->type.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
              value_temp << ' ';
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, op->index.type(), i, value_temp);
        value_temp << ']';
        PrintVecElemStore(svalue, op->type, i, value_temp.str());
      }
      os << svalue;
      EndScope(vec_scope);
    }
  }
}

void CodeAnalysis::VisitStmt_(const Store* op) {
  Type t = op->value.type();
  if (t.lanes() == 1) {
    std::string value = this->PrintExpr(op->value);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();
    stream << ref << " = " << value << ";\n";
  } else {
    CHECK(is_one(op->predicate)) << "Predicated store is not supported";
    Expr base;
    if (TryGetRamp1Base(op->index, t.lanes(), &base)) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer_var.get(), t, base, value);
    } else {
      // The assignment below introduces side-effect, and the resulting value
      // cannot be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements seperately
      std::string index = SSAGetID(PrintExpr(op->index), op->index.type());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.type());
      std::string vid = GetVarID(op->buffer_var.get());
      for (int i = 0; i < t.lanes(); ++i) {
        this->PrintIndent();
        Type elem_type = t.element_of();
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          stream << "((";
          if (op->buffer_var.get()->type.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, stream);
              stream << ' ';
            }
          }
          PrintType(elem_type, stream);
          stream << "*)" << vid << ')';
        } else {
          stream << vid;
        }
        stream << '[';
        PrintVecElemLoad(index, op->index.type(), i, stream);
        stream << "] = ";
        PrintVecElemLoad(value, op->value.type(), i, stream);
        stream << ";\n";
      }
      EndScope(vec_scope);
    }
  }
}

void CodeAnalysis::VisitExpr_(const Let* op,
                              std::ostream& os) {  // NOLINT(*)
  std::string value = PrintExpr(op->value);
  CHECK(!var_idmap_.count(op->var.get()));
  var_idmap_[op->var.get()] = value;
}

void CodeAnalysis::VisitExpr_(const Ramp* op,
                              std::ostream& os) {  // NOLINT(*)
  // constraint of current logic
  CHECK_EQ(op->base.type(), Int(32));
  os << "((int" << op->lanes << ")(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != op->lanes - 1) os << ", ";
  }
  os << "))";
}

void CodeAnalysis::VisitExpr_(const Broadcast* op,
                              std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Broadcast: not supported ";
}

void CodeAnalysis::VisitExpr_(const StreamExpr* op,
                              std::ostream& os) {  // NOLINT(*)
}

void CodeAnalysis::VisitExpr_(const Select* op,
                              std::ostream& os) {  // NOLINT(*)
  os << "(";
  PrintExpr(op->condition, os);
  os << " ? ";
  PrintExpr(op->true_value, os);
  os << " : ";
  PrintExpr(op->false_value, os);
  os << ")";
}

void CodeAnalysis::VisitExpr_(const GetBit* op,
                              std::ostream& os) {  // NOLINT(*)
  os << "(";
  PrintExpr(op->a, os);
  os << " & (1 << (";
  PrintExpr(op->index, os);
  os << " - 1)))";
}

void CodeAnalysis::VisitExpr_(const GetSlice* op,
                              std::ostream& os) {  // NOLINT(*)
  // 1. a' = SHR a for Idx_R bits
  // 2. mask: 1.(length).1
  //          (1 << (L - R + 1)) - 1
  // 3. a' & mask

  os << "((";
  PrintExpr(op->a, os);
  os << " >> ";
  PrintExpr(op->index_right, os);
  os << ") & ((1 << (";
  PrintExpr(op->index_left, os);
  os << " - ";
  PrintExpr(op->index_right, os);
  os << " + 1)) - 1))";
}

void CodeAnalysis::VisitExpr_(const SetBit* op,
                              std::ostream& os) {  // NOLINT(*)
  os << "(";
  PrintExpr(op->a, os);
  os << " | (1 << (";
  PrintExpr(op->index, os);
  os << " - 1)))";
}

void CodeAnalysis::VisitExpr_(const SetSlice* op,
                              std::ostream& os) {  // NOLINT(*)
  // 1. mask: 0.(Idx L).01..10.(Idx R).0
  //          ((1 << (L - R + 1)) - 1) << R
  // 2. a & mask

  os << "(";
  PrintExpr(op->a, os);
  os << " & (((1 << (";
  PrintExpr(op->index_left, os);
  os << " - ";
  PrintExpr(op->index_right, os);
  os << " + 1)) - 1) << ";
  PrintExpr(op->index_right, os);
  os << "))";
}

void CodeAnalysis::VisitExpr_(const Quantize* op,
                              std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Quantize is not yet support";
}

void CodeAnalysis::VisitExpr_(const KernelExpr* op,
                              std::ostream& os) {  // NOLINT(*)
}

void CodeAnalysis::VisitStmt_(const LetStmt* op) {
  std::string arg_vid = "unknown";
  std::string str = PrintExpr(op->value);
  if (std::regex_match(str,
                       std::regex("(.+)TVMArray(.+)(arg)(.+)(data)(.+)"))) {
    size_t pos_arg = str.find("arg");
    size_t pos_data = str.find("data");
    arg_vid = str.substr(pos_arg, pos_data - pos_arg - 5);
  } else if (std::regex_match(str, std::regex("arg(.+)"))) {
    arg_vid = str;
  }

  std::string vid = AllocVarID(op->var.get());
  if (this->map_arg_type_.find(arg_vid) == this->map_arg_type_.end()) {
    if ("unknown" != arg_vid)
      LOG(WARNING) << arg_vid << " not found in the argument mapping\n";
  } else {
    Type type = op->var.type();
    if (op->var.type() == Handle() && handle_data_type_.count(op->var.get()))
      type = handle_data_type_.at(op->var.get());
    this->map_arg_type_[arg_vid] = std::make_tuple(vid, type);
  }
  VisitStmt(op->body);
}

void CodeAnalysis::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  if (op->new_expr.defined()) {
    // Prefer global static allocation for the program
    CHECK_EQ(op->free_function, "nop");
    std::string new_data = PrintExpr(op->new_expr);
    this->PrintIndent();
    PrintType(op->type, stream);
    stream << "* " << vid << '=' << new_data << ";\n";
  } else {
    this->PrintIndent();
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";
    const Variable* buffer = op->buffer_var.as<Variable>();
    std::string scope = alloc_storage_scope_.at(buffer);
    PrintStorageScope(scope, stream);
    stream << ' ';
    PrintType(op->type, stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeAnalysis::VisitStmt_(const AttrStmt* op) {
  if (op->attr_key == ir::attr::thread_extent) {
    IterVar iv(op->node.node_);
    if (iv->thread_tag.length() != 0) {
      if (!var_idmap_.count(iv->var.get())) {
        BindThreadIndex(iv);
      }
    }
  } else if (op->attr_key == ir::attr::storage_scope) {
    const Variable* v = op->node.as<Variable>();
    CHECK(v);
    alloc_storage_scope_[v] = op->value.as<StringImm>()->value;
  } else if (op->attr_key == ir::attr::volatile_scope) {
    const Variable* v = op->node.as<Variable>();
    CHECK(v);
    volatile_buf_.insert(v);
  }
  this->PrintStmt(op->body);
}

void CodeAnalysis::VisitStmt_(const AssertStmt* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (op->message.as<StringImm>()) {
    // GLOG style check
    stream << "CHECK(" << cond << ") << \""
           << op->message.as<StringImm>()->value << "\";\n";
  } else {
    stream << "assert(" << cond << ");\n";
  }
  this->PrintStmt(op->body);
}

void CodeAnalysis::VisitStmt_(const For* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  CHECK(is_zero(op->min));
  stream << "for (";
  PrintType(op->loop_var.type(), stream);
  stream << ' ' << vid << " = 0; " << vid << " < " << extent << "; ++" << vid
         << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeAnalysis::VisitStmt_(const IfThenElse* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (cond[0] == '(' && cond[cond.length() - 1] == ')') {
    stream << "if " << cond << " {\n";
  } else {
    stream << "if (" << cond << ") {\n";
  }
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);

  if (op->else_case.defined()) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case);
    this->EndScope(else_scope);
  }
  PrintIndent();
  stream << "}\n";
}

void CodeAnalysis::VisitStmt_(const Block* op) {
  PrintStmt(op->first);
  if (op->rest.defined()) PrintStmt(op->rest);
}

void CodeAnalysis::VisitStmt_(const Evaluate* op) {
  if (is_const(op->value)) return;
  const Call* call = op->value.as<Call>();
  if (call) {
    if (call->is_intrinsic(intrinsic::tvm_storage_sync)) {
      this->PrintStorageSync(call);
      return;
    } else if (call->is_intrinsic(intrinsic::tvm_struct_set)) {
      CHECK_EQ(call->args.size(), 4);
      std::string value = PrintExpr(call->args[3]);
      std::string ref =
          GetStructRef(call->args[3].type(), call->args[0], call->args[1],
                       call->args[2].as<IntImm>()->value);
      this->PrintIndent();
      this->stream << ref << " = " << value << ";\n";
      return;
    }
  }
  std::string vid = this->PrintExpr(op->value);
  this->PrintIndent();
  this->stream << "(void)" << vid << ";\n";
}

void CodeAnalysis::VisitStmt_(const ProducerConsumer* op) {
  PrintStmt(op->body);
}

void CodeAnalysis::VisitStmt_(const KernelDef* op) {}

void CodeAnalysis::VisitStmt_(const KernelStmt* op) {}

void CodeAnalysis::VisitStmt_(const Return* op) {
  this->stream << "return ";
  PrintExpr(op->value);
  this->stream << ";\n";
}

void CodeAnalysis::VisitStmt_(const Break* op) {
  // TODO(seanlatias): Check if the break statement is used correctly
  this->stream << "break;\n";
}

void CodeAnalysis::VisitStmt_(const While* op) {
  std::string condition = PrintExpr(op->condition);
  PrintIndent();
  stream << "while (" << condition << ") {\n";
  int while_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(while_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeAnalysis::VisitStmt_(const Reuse* op) {
  LOG(FATAL) << "KernelDef is not yet support";
}

void CodeAnalysis::VisitStmt_(const Partition* op) {}

void CodeAnalysis::VisitStmt_(const StreamStmt* op) {}

void CodeAnalysis::VisitStmt_(const Stencil* op) { PrintStmt(op->body); }

void CodeAnalysis::VisitStmt_(const ExternModule* op) {
  this->PrintStmt(op->body);
}

void CodeAnalysis::VisitStmt_(const Print* op) {}

}  // namespace codegen
}  // namespace TVM
