/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_c.cc
 */
#include <tvm/build_module.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <iomanip>
#include <cctype>
#include "./codegen_c.h"
#include "./merlinc/codeanalys_merlinc.h"
#include "../arithmetic/compute_expr.h"

namespace TVM {
namespace codegen {

using namespace ir;

Type String2Type(std::string& s) {
  if (s.front() == '\"' && s.back() == '\"') {
    s.erase(0, 1);
    s.pop_back();
  }
  std::istringstream is(s);
  halideir_type_code_t code = Type::Int;
  int bits = 32, lanes = 1;
  if (s.substr(0, 3) == "int") {
    code = Type::Int; s = s.substr(3);

  } else if (s.substr(0, 4) == "uint") {
    code = Type::UInt; s = s.substr(4);

  } else if (s.substr(0, 5) == "float") {
    code = Type::Float; s = s.substr(5);

  } else if (s.substr(0, 5) == "fixed") {
    code = Type::Int; s = s.substr(5);
    int integer = 0;
    if (sscanf(s.c_str(), "%d_%d", &bits, &integer) == 0) 
      LOG(FATAL) << "unknown type " << s;
    CHECK(integer <= bits) << "invalid type " << s;
    return Type(code, bits, lanes, integer);

  } else if (s.substr(0, 6) == "ufixed") {
    code = Type::UInt; s = s.substr(6);
    int integer = 0;
    if (sscanf(s.c_str(), "%d_%d", &bits, &integer) == 0) 
      LOG(FATAL) << "unknown type " << s;
    CHECK(integer <= bits) << "invalid type " << s;
    return Type(code, bits, lanes, bits - integer);

  } else if (s == "handle") {
    return Handle();

  } else {
    LOG(FATAL) << "unknown type " << s;
  }
  if (sscanf(s.c_str(), "%dx%d", &bits, &lanes) == 0) {
    LOG(FATAL) << "unknown type " << s;
  }
  return Type(code, bits, lanes);
}

// generate row major index
std::string getIndex(std::vector<int> shape) {
  std::string str;
  int mul = 1;
  for (size_t i = shape.size(); i > 0; i--) {
    mul = mul * shape[i-1];
    str += "i" + std::to_string(i-1) +
           "*" + std::to_string(mul);
    if (i != 1) str += "+ ";
  }
  return str;
}

void CodeGenC::Init(bool output_ssa) {
  print_ssa_form_ = output_ssa;
}

void CodeGenC::InitFuncState(LoweredFunc f) {
  alloc_set_.clear();
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  var_shape_map_.clear();
  range_.clear();
  CodeGenSourceBase::ClearFuncState();
}

void CodeGenC::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  // clear previous generated state.
  this->InitFuncState(f);
  map_arg_type_ = map_arg_type;
  // add to alloc buffer type.
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  // generate function signature 
  this->stream << "void " << f->name << "(";
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    // check type in the arg map
    if (map_arg_type.find(vid) == map_arg_type.end()) {
      LOG(WARNING) << vid << " type not found\n";
      PrintType(v.type(), this->stream);
      this->stream << ' ' << vid;
    } else {
      auto arg = map_arg_type[vid];
      PrintType(std::get<1>(arg), this->stream);
      this->stream << "* " << std::get<0>(arg);
      const BufferNode* buf = f->api_args[i].as<BufferNode>();
      if (v.type().is_handle() && buf) {
        var_shape_map_[buf->data.get()] = buf->shape;
        auto it = alloc_storage_scope_.find(v.get());
        if (it != alloc_storage_scope_.end())
          PrintStorageScope(it->second, stream);
      }
    }
  }

  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

std::string CodeGenC::GetConfig() {
  return this->cfg_stream.str(); 
}

std::string CodeGenC::GetHost() {
  return decl_stream.str() + 
      this->stream.str(); 
}

std::string CodeGenC::GetDevice() {
  return decl_stream.str() + 
      module_stream.str(); 
}

std::string CodeGenC::Finish() {
  return decl_stream.str() + 
         module_stream.str() + 
         stream.str();
}

void CodeGenC::PrintExpr(const Expr& n, std::ostream& os) {  // NOLINT(*)
  if (print_ssa_form_) {
    std::ostringstream temp;
    VisitExpr(n, temp);
    os << SSAGetID(temp.str(), n.type());
  } else {
    VisitExpr(n, os);
  }
}

void CodeGenC::PrintSSAAssign(
    const std::string& target, const std::string& src, Type t) {
  PrintType(t, stream);
  stream << ' ' << target << " = ";
  if (src.length() > 3 &&
      src[0] == '(' && src[src.length() - 1] == ')') {
    stream << src.substr(1, src.length() - 2);
  } else {
    stream << src;
  }
  stream << ";\n";
}

// Print a reference expression to a buffer.
std::string CodeGenC::GetBufferRef(
    Type t, const Variable* buffer, Expr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  std::string scope;
  if (alloc_storage_scope_.count(buffer)) {
    scope = alloc_storage_scope_.at(buffer);
  }
  bool is_vol = volatile_buf_.count(buffer) != 0;
  if (t.lanes() == 1) {
    bool is_scalar = (buf_length_map_.count(buffer) == 1 &&
        buf_length_map_[buffer] == 1);
    if (is_scalar) {
      os << vid;
    } else {
      /* Don't need this!!
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
      } */
      os << vid;
      os << '[';
      PrintExpr(index, os);
      os << ']';
    }
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
std::string CodeGenC::GetStructRef(
    Type t, const Expr& buffer, const Expr& index, int kind) {
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
      case intrinsic::kArrData: os << "data"; break;
      case intrinsic::kArrShape: os << "shape"; break;
      case intrinsic::kArrStrides: os << "strides"; break;
      case intrinsic::kArrNDim: os << "ndim"; break;
      case intrinsic::kArrTypeCode: os << "dtype.code"; break;
      case intrinsic::kArrTypeBits: os << "dtype.bits"; break;
      case intrinsic::kArrTypeLanes: os << "dtype.lanes"; break;
      case intrinsic::kArrTypeFracs: os << "dtype.fracs"; break;
      case intrinsic::kArrDeviceId: os << "ctx.device_id"; break;
      case intrinsic::kArrDeviceType: os << "ctx.device_type"; break;
      default: LOG(FATAL) << "unknown field code";
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
      LOG(FATAL) << "donot know how to handle type" << t;
    }
    os << ")";
    return os.str();
  }
}


bool CodeGenC::HandleTypeMatch(const Variable* buf_var, Type t) const {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) return false;
  return it->second == t;
}

void CodeGenC::RegisterHandleType(const Variable* buf_var, Type t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_[buf_var] = t;
  } else {
    CHECK(it->second == t)
        << "conflicting buf var type";
  }
}

void CodeGenC::PrintVecElemLoad(const std::string& vec,
                                Type t, int i,
                                std::ostream& os) {  // NOLINT(*)
  os << vec << ".s" << std::hex << i << std::dec;
}

void CodeGenC::PrintVecElemStore(const std::string& vec,
                                 Type t, int i,
                                 const std::string& value) {
  this->PrintIndent();
  stream << vec << ".s" << std::hex << i
         << " = " << value << ";\n" << std::dec;
}

std::string CodeGenC::GetVecLoad(
    Type t, const Variable* buffer, Expr base) {
  return GetBufferRef(t, buffer, base);
}

void CodeGenC::PrintVecStore(const Variable* buffer,
                             Type t, Expr base,
                             const std::string& value) {
  std::string ref = GetBufferRef(t, buffer, base);
  this->PrintIndent();
  stream << ref << " = " << value << ";\n";
}

std::string CodeGenC::CastFromTo(std::string value, Type from, Type target) {
  if (from == target) return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")" << value << ")";
  return os.str();
}

void CodeGenC::BindThreadIndex(const IterVar& iv) {
  LOG(FATAL) << "not implemented";
}

void CodeGenC::PrintStorageSync(const Call* op) { // NOLINT(*)
}

void CodeGenC::PrintStorageScope(const std::string& scope, std::ostream& os) { // NOLINT(*)
  // CHECK_EQ(scope, "global");
}

void CodeGenC::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  CHECK_EQ(t.lanes(), 1)
     << "do not yet support vector types";
  if (t.is_handle()) {
    os << "void*"; return;
  }
  if (t.is_float()) {
    if (t.bits() == 32) {
      os << "float"; return;
    }
    if (t.bits() == 64) {
      os << "double"; return;
    }
  } else if (t.is_uint()) {
    switch (t.bits()) {
      case 8: case 16: case 32: case 64: {
        os << "uint" << t.bits() << "_t"; return;
      }
      case 1: os << "int"; return;
    }
    if (t.bits() < 8) { os << "int8_t";  return;
    } else if (t.bits() < 16)  { os << "uint16_t"; return;
    } else if (t.bits() < 32)  { os << "uint32_t"; return;
    } else if (t.bits() < 64)  { os << "uint64_t"; return;
    } else if (t.bits() < 128) { os << "uint64_t"; return;
    } else {
      LOG(FATAL) << "Cannot convert type " << t << " to C type";
    }
  } else if (t.is_int()) {
    switch (t.bits()) {
      case 8: case 16: case 32: case 64: {
        os << "int" << t.bits() << "_t";  return;
      }
    }
    if (t.bits() < 8) { os << "int8_t";  return;
    } else if (t.bits() < 16)  { os << "int16_t"; return;
    } else if (t.bits() < 32)  { os << "int32_t"; return;
    } else if (t.bits() < 64)  { os << "int64_t"; return;
    } else if (t.bits() < 128) { os << "int64_t"; return;
    } else {
      LOG(FATAL) << "Cannot convert type " << t << " to C type";
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

inline void PrintConst(const IntImm* op, std::ostream& os, CodeGenC* p) { // NOLINT(*)
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

inline void PrintConst(const UIntImm* op, std::ostream& os, CodeGenC* p) { // NOLINT(*)
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

inline void PrintConst(const FloatImm* op, std::ostream& os, CodeGenC* p) { // NOLINT(*)
  switch (op->type.bits()) {
    case 64: case 32: {
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
      os << ')' << std::scientific <<op->value << 'f';
      break;
    }
    default: LOG(FATAL) << "Bad bit-width for float: " << op->type << "\n";
  }
}

void CodeGenC::VisitExpr_(const IntImm *op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeGenC::VisitExpr_(const UIntImm *op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeGenC::VisitExpr_(const FloatImm *op, std::ostream& os) { // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeGenC::VisitExpr_(const StringImm *op, std::ostream& os) { // NOLINT(*)
  os << "\"" << op->value << "\"";
}

template<typename T>
inline void PrintBinaryExpr(const T* op,
                            const char *opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenC* p) {
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

inline void PrintBinaryIntrinsitc(const Call* op,
                                  const char *opstr,
                                  std::ostream& os,  // NOLINT(*)
                                  CodeGenC* p) {
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
void CodeGenC::VisitExpr_(const Cast *op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  this->PrintExpr(op->value, value);
  os << CastFromTo(value.str(), op->value.type(), op->type);
}
void CodeGenC::VisitExpr_(const Variable *op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}
void CodeGenC::VisitExpr_(const Add *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
void CodeGenC::VisitExpr_(const Sub *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
void CodeGenC::VisitExpr_(const Mul *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}
void CodeGenC::VisitExpr_(const Div *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}
void CodeGenC::VisitExpr_(const Mod *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}
void CodeGenC::VisitExpr_(const Min *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "min", os, this);
}
void CodeGenC::VisitExpr_(const Max *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "max", os, this);
}
void CodeGenC::VisitExpr_(const EQ *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "==", os, this);
}
void CodeGenC::VisitExpr_(const NE *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "!=", os, this);
}
void CodeGenC::VisitExpr_(const LT *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<", os, this);
}
void CodeGenC::VisitExpr_(const LE *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<=", os, this);
}
void CodeGenC::VisitExpr_(const GT *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">", os, this);
}
void CodeGenC::VisitExpr_(const GE *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">=", os, this);
}
void CodeGenC::VisitExpr_(const And *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "&&", os, this);
}
void CodeGenC::VisitExpr_(const Or *op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "||", os, this);
}
void CodeGenC::VisitExpr_(const Not *op, std::ostream& os) {  // NOLINT(*)
  os << '!';
  PrintExpr(op->a, os);
}

void CodeGenC::VisitExpr_(const Call *op, std::ostream& os) {  // NOLINT(*)
  if (op->call_type == Call::Extern ||
      op->call_type == Call::PureExtern) {
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
  } else if (op->is_intrinsic(Call::bitcast)) {
    this->PrintIndent();
    std::string conv_name = GetUniqueName("_converter");
    int bits = op->args[0].type().bits();
    if (op->args[0].type().code() == Type::Float ||
        op->type.code() == Type::Float) {
      CHECK(bits == 32 || bits == 64);
      std::string ty_from = bits == 32 ? "float" : "double";
      std::string ty_to = bits == 32 ? "uint32_t" : "uint64_t";
      bool from_float = op->args[0].type().code() == Type::Float;
      stream << "union { ";
      if (from_float) stream << ty_from;
      else            stream << ty_to;
      stream << " from; ";
      if (from_float) stream << ty_to;
      else            stream << ty_from;
      stream << " to;} " << conv_name << ";\n";
      this->PrintIndent();
      stream << conv_name << ".from = ";
      this->PrintExpr(op->args[0], stream);
      stream << ";\n";
      os << conv_name << ".to";
    } else {
      this->PrintType(op->type, stream);
      stream << " " << conv_name << ";\n";
      this->PrintIndent();
      stream << conv_name << "(" << bits-1 << ", 0) = ";
      this->PrintExpr(op->args[0], stream);
      stream << "(" << bits-1 << ", 0)";
      stream << ";\n";
      os << conv_name;
    }
  } else if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
    os << "(";
    PrintExpr(op->args[0], os);
    os << " ? ";
    // type casting when mismatching
    auto& v1 = op->args[1];
    auto& v2 = op->args[2];
    bool cast_value = false;
    if (v1.as<IntImm>() || v1.as<UIntImm>() || v1.as<FloatImm>()) {
      if (auto var = v2.as<Load>()) {
        cast_value = true;
        Type type = handle_data_type_[var->buffer_var.get()];
        std::stringstream value;
        this->PrintExpr(v1, value);
        os << "((";
        this->PrintType(type, os);
        os << ")" << value.str() << ")";

        os << " : ";
        PrintExpr(op->args[2], os);
        os << ")";
      }
    } else if (v2.as<IntImm>() || v2.as<UIntImm>() || v2.as<FloatImm>()) {
      if (auto var = v1.as<Load>()) {
        cast_value = true;
        PrintExpr(op->args[1], os);
        os << " : ";

        Type type = handle_data_type_[var->buffer_var.get()];
        std::stringstream value;
        this->PrintExpr(v2, value);
        os << "((";
        this->PrintType(type, os);
        os << ")" << value.str() << ")";
        os << ")";
      }
    } 
    if (!cast_value) {
      PrintExpr(op->args[1], os);
      os << " : ";
      PrintExpr(op->args[2], os);
      os << ")";
    }
  } else if (op->is_intrinsic(intrinsic::tvm_address_of)) {
    const Load *l = op->args[0].as<Load>();
    CHECK(op->args.size() == 1 && l);
    os << "((";
    this->PrintType(l->type.element_of(), os);
    os << " *)" << this->GetVarID(l->buffer_var.get())
       << " + ";
    this->PrintExpr(l->index, os);
    os << ')';
  } else if (op->is_intrinsic(intrinsic::tvm_struct_get)) {
    CHECK_EQ(op->args.size(), 3U);
    os << GetStructRef(
        op->type, op->args[0], op->args[1],
        op->args[2].as<IntImm>()->value);
  } else if (op->is_intrinsic(intrinsic::tvm_handle_is_null)) {
    CHECK_EQ(op->args.size(), 1U);
    os << "(";
    this->PrintExpr(op->args[0], os);
    os << " == NULL)";
  } else {
    if (op->call_type == Call::Intrinsic ||
        op->call_type == Call::PureIntrinsic) {
      LOG(FATAL) << "Unresolved intrinsic " << op->name
                 << " with return type " << op->type;
    } else {
      LOG(FATAL) << "Unresolved call type " << op->call_type;
    }
  }
}

void CodeGenC::PrintVecBinaryOp(
    const std::string& op, Type t,
    Expr lhs, Expr rhs, std::ostream& os) {  // NOLINT(*)
  if (isalpha(op[0])) {
    os << op << "(";
    this->PrintExpr(lhs, os);
    os << ", ";
    this->PrintExpr(rhs, os);
    os << ")";
  } else {
    os <<"(";
    this->PrintExpr(lhs, os);
    os << ' ' << op << ' ';
    this->PrintExpr(rhs, os);
    os << ")";
  }
}

inline bool TryGetRamp1Base(Expr index, int lanes, Expr *base) {
  const Ramp* r = index.as<Ramp>();
  if (!r) return false;
  if (!is_one(r->stride)) return false;
  CHECK_EQ(r->lanes, lanes);
  *base = r->base;
  return true;
}

void CodeGenC::VisitExpr_(const Load* op, std::ostream& os) {  // NOLINT(*)
  int lanes = op->type.lanes();
  // delcare type.
  if (op->type.lanes() == 1) {
    std::string ref = GetBufferRef(op->type, op->buffer_var.get(), op->index);
    os << ref;
  } else {
    CHECK(is_one(op->predicate))
        << "predicated load is not supported";
    Expr base;
    if (TryGetRamp1Base(op->index, op->type.lanes(), &base)) {
      std::string ref = GetVecLoad(op->type, op->buffer_var.get(), base);
      os << ref;
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
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

void CodeGenC::VisitStmt_(const Store* op) {
  Type t = op->value.type();
  if (t.lanes() == 1) {
    std::string value = this->PrintExpr(op->value);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();
    stream << ref << " = " << value << ";\n";
  } else {
    CHECK(is_one(op->predicate))
        << "Predicated store is not supported";
    Expr base;
    if (TryGetRamp1Base(op->index, t.lanes(), &base)) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer_var.get(), t, base, value);
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements seperately
      std::string index = SSAGetID(PrintExpr(op->index), op->index.type());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.type());
      std::string vid = GetVarID(op->buffer_var.get());
      for (int i = 0; i < t.lanes(); ++i) {
        this->PrintIndent();
        stream << vid;
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

void CodeGenC::VisitExpr_(const Let* op, std::ostream& os) {  // NOLINT(*)
  CHECK(print_ssa_form_)
      << "LetExpr is only supported by print SSA form";
  std::string value = PrintExpr(op->value);
  CHECK(!var_idmap_.count(op->var.get()));
  var_idmap_[op->var.get()] = value;
}

void CodeGenC::VisitExpr_(const Ramp* op, std::ostream& os) {  // NOLINT(*)
  // constraint of current logic
  CHECK_EQ(op->base.type(), Int(32));
  os << "((int" << op->lanes << ")(";
  for (int i = 0; i < op->lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")" << "+(" << PrintExpr(op->stride) << "*" << i <<")";
    if (i != op->lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenC::VisitExpr_(const Broadcast* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Broadcast: not supported ";
}

void CodeGenC::VisitExpr_(const Select* op, std::ostream& os) {  // NOLINT(*)
  os << "(";
  PrintExpr(op->condition, os);
  os << " ? ";
  PrintExpr(op->true_value, os);
  os << " : ";
  PrintExpr(op->false_value, os);
  os << ")";
}

void CodeGenC::VisitExpr_(const GetBit *op, std::ostream& os) { // NOLINT(*)
  os << "((";
  PrintExpr(op->a, os);
  os << " & (1L << ";
  PrintExpr(op->index, os);
  os << ")) >> ";
  PrintExpr(op->index, os);
  os << ")";
}

void CodeGenC::VisitExpr_(const GetSlice *op, std::ostream& os) { // NOLINT(*)
  // 1. a' = SHR a for Idx_R bits
  // 2. mask: 1.(length).1
  //          (1 << (L - R + 1)) - 1
  // 3. a' & mask

  os << "((";
  PrintExpr(op->a, os);
  os << " >> ";
  PrintExpr(op->index_left, os);
  os << ") & ((1L << (";
  PrintExpr(op->index_right, os);
  os << " - ";
  PrintExpr(op->index_left, os);
  os << ")) - 1))";
}

void CodeGenC::VisitExpr_(const SetBit *op, std::ostream& os) { // NOLINT(*)
  LOG(FATAL) << "SetBit is not implemented yet in C";
}

void CodeGenC::VisitExpr_(const SetSlice *op, std::ostream& os) { // NOLINT(*)
  LOG(FATAL) << "SetSlice is not implemented yet in C";
}

void CodeGenC::VisitExpr_(const Quantize *op, std::ostream& os) { // NOLINT(*)
  LOG(FATAL) << "Quantize is not yet support in C";
}

void CodeGenC::VisitExpr_(const StreamExpr *op, std::ostream& os) { // NOLINT(*)
  auto v = op->buffer_var.get();
  auto it = var_idmap_.find(v);
  CHECK(it != var_idmap_.end())
    << "variable " << v->name_hint << " not decalred";
  std::string vid = GetVarID(op->buffer_var.get()); 
  os << vid << ".read()";
}

void CodeGenC::VisitExpr_(const KernelExpr *op, std::ostream& os) { // NOLINT(*)
  os << op->name << "(";
  for (size_t i = 0; i < op->args.size(); ++i) {
    PrintExpr(op->args[i], os);
    if (i != op->args.size() - 1) os << ", ";
  }
  os << ")";
}

void CodeGenC::VisitStmt_(const StreamStmt *op) { // NOLINT(*)
  PrintIndent();
  std::string vid = GetVarID(op->buffer_var.get()); 
  stream << vid << ".write(";
  PrintExpr(op->value, stream);
  stream << ");\n";
}

void CodeGenC::VisitStmt_(const LetStmt* op) {
  std::string value = PrintExpr(op->value);
  // Skip the argument retrieving assign statement
  std::string vid = AllocVarID(op->var.get());
  if (print_ssa_form_) {
    CHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    if (op->var.type() != Handle() &&
        value.find("TVMArray") == std::string::npos &&
        value.find("arg") != 0) {
      PrintIndent();
      PrintType(op->var.type(), this->stream);
      this->stream << ' '
                   << vid
                   << " = " << value << ";\n";

    // collect top args variable id
    } else if (value.find("data") != std::string::npos ||
               value.substr(0, 3) == "arg") {
      arg_names.push_back(vid);
      alloc_set_.insert(vid);
    }
    PrintStmt(op->body);
  }
}

void CodeGenC::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  if (op->new_expr.defined()) {
    // Prefer global static allocation for the program
    CHECK_EQ(op->free_function, "nop");
    std::string new_data = PrintExpr(op->new_expr);
    this->PrintIndent();
    PrintType(op->type, stream);
    stream << "* "<< vid << '=' << new_data << ";\n";

  } else {
    this->PrintIndent();
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";
    const Variable* buffer = op->buffer_var.as<Variable>();

    std::string scope; // allocate on local scope by default 
    auto it = alloc_storage_scope_.find(buffer);
    if (it != alloc_storage_scope_.end())
      scope = alloc_storage_scope_.at(buffer);
    else scope = "local";

    PrintStorageScope(scope, stream);
    PrintType(op->type, stream);
    stream << ' '<< vid;
    if (constant_size > 1) // Transfer length one array to scalar
      stream << '[' << constant_size << "]";
    stream << ";\n";
    buf_length_map_[buffer] = constant_size;
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const AttrStmt* op) {

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

void CodeGenC::VisitStmt_(const ExternModule* op) {
  LOG(FATAL) << "does not support ExternModule in C";
}

void CodeGenC::VisitStmt_(const AssertStmt* op) {
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

void CodeGenC::VisitStmt_(const For* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  CHECK(is_zero(op->min));
  stream << "for (";
  PrintType(op->loop_var.type(), stream);
  stream << ' ' << vid << " = 0; "
            << vid << " < " << extent
            << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenC::VisitStmt_(const IfThenElse* op) {
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

void CodeGenC::VisitStmt_(const Block *op) {
  PrintStmt(op->first);
  if (op->rest.defined()) PrintStmt(op->rest);
}

void CodeGenC::VisitStmt_(const Evaluate *op) {
  if (is_const(op->value)) return;
  const Call* call = op->value.as<Call>();
  if (call) {
    if (call->is_intrinsic(intrinsic::tvm_storage_sync)) {
      this->PrintStorageSync(call); return;
    } else if (call->is_intrinsic(intrinsic::tvm_struct_set)) {
      CHECK_EQ(call->args.size(), 4);
      std::string value = PrintExpr(call->args[3]);
      std::string ref = GetStructRef(
          call->args[3].type(),
          call->args[0],
          call->args[1],
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

void CodeGenC::VisitStmt_(const ProducerConsumer *op) {
  PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const KernelDef* op) {
  LoweredFunc f;
  // save func states
  SaveFuncState(f);
  InitFuncState(f);
  std::ostringstream save;
  save << this->stream.str();
  this->stream.str("");
  this->stream.clear();

  // skip the first underscore
  GetUniqueName("_");
  // add to alloc buffer : type.
  for (const auto & k : op->args) {
    RegisterHandleType(k.get(), k.get()->type);
  }
  // print function signature
  PrintType(op->ret_type, stream);
  stream << " " << op->name << "(";
  for (size_t i = 0; i < op->args.size(); ++i) {
    VarExpr v = op->args[i];
    var_shape_map_[v.get()] = op->arg_shapes[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    std::string str = PrintExpr(op->arg_types[i]);
    Type type = String2Type(str);
    PrintType(type, stream);

    this->stream << " " << vid;
    if (v.type().is_handle()) {
      this->stream << "[";
      for (size_t j = 0; j < op->arg_shapes[i].size(); j++) {
        if (j != 0) stream << "* ";
        auto dim = op->arg_shapes[i][j].as<IntImm>()->value;
        this->stream << dim;
      }
      this->stream << ']';
    }
  }  
  stream << ") {\n";
  int func_scope = BeginScope();
  range_ = CollectIterRange(op->body);
  PrintStmt(op->body);
  EndScope(func_scope);
  stream << "}\n\n";

  // restore default stream
  module_stream << this->stream.str();
  this->stream.str(""); 
  this->stream.clear();
  this->stream << save.str();
  RestoreFuncState(f);
}

void CodeGenC::VisitStmt_(const KernelStmt *op) {
  PrintIndent();
  stream << op->name << "(";
  for (size_t i = 0; i < op->args.size(); i++) {
    PrintExpr(op->args[i], stream);
    if (i < op->args.size() -1) stream << ", ";
  }
  stream << ");\n";
}

void CodeGenC::VisitStmt_(const Return *op) {
  this->stream << "return ";
  PrintExpr(op->value, stream);
  this->stream << ";\n";
}

void CodeGenC::VisitStmt_(const Break *op) {
  // TODO: Check if the break statement is used correctly
  PrintIndent();
  this->stream << "break;\n";
}

void CodeGenC::VisitStmt_(const While *op) {
  std::string condition = PrintExpr(op->condition);
  PrintIndent();
  stream << "while (" << condition << ") {\n";
  int while_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(while_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenC::VisitStmt_(const Partition* op) {
}

void CodeGenC::SaveFuncState(LoweredFunc f) {
  // clear save info copy
  alloc_set_save.clear();
  alloc_storage_scope_save.clear();
  handle_data_type_save.clear();
  var_shape_map_save.clear();
  range_save.clear();
  // backup func info and clear
  alloc_set_save = alloc_set_;
  alloc_storage_scope_save = alloc_storage_scope_;
  handle_data_type_save = handle_data_type_;
  var_shape_map_save = var_shape_map_;
  range_save = range_;
  CodeGenSourceBase::SaveFuncState();
}

void CodeGenC::RestoreFuncState(LoweredFunc f) {
  this->InitFuncState(f);
  alloc_set_ = alloc_set_save;
  alloc_storage_scope_ = alloc_storage_scope_save;
  handle_data_type_ = handle_data_type_save;
  var_shape_map_ = var_shape_map_save;
  range_ = range_save;
  CodeGenSourceBase::RestoreFuncState();
}

}  // namespace codegen
}  // namespace TVM
