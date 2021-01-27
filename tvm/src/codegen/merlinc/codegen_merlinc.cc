/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_merlinc.cc
 */
#include "codegen_merlinc.h"
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/config.h>
#include <regex>
#include <string>
#include <tuple>
#include <vector>
#include "../../runtime/thread_storage_scope.h"

namespace TVM {
namespace codegen {

CodeGenMerlinC::CodeGenMerlinC() {
  restrict_keyword_ = "restrict";  // FIXME: Check if this is useful
  return;
}

void CodeGenMerlinC::InitFuncState(LoweredFunc f) {
  CodeGenC::InitFuncState(f);
  for (Var arg : f->args) {
    if (arg.type().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
  return;
}

void CodeGenMerlinC::AddFunction(LoweredFunc f,
                                 str2tupleMap<std::string, Type> map_arg_type) {
  // Clear previous generated state
  this->InitFuncState(f);

  // Skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");

  // Register alloc buffer type
  for (const auto& kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  // Write header files
  this->stream << "#include <string.h>\n";
  this->stream << "#include <math.h>\n";
  this->stream << "#include <assert.h>\n";

  // Write entry function name
  this->stream << "#pragma ACCEL kernel\n";
  this->stream << "void " << f->name << "(";

  // Write arguments
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) this->stream << ", ";
    if (map_arg_type.find(vid) == map_arg_type.end()) {
      LOG(WARNING) << vid << " type not found\n";
      PrintType(v.type(), this->stream);
      this->stream << ' ' << vid;
    } else {
      auto arg = map_arg_type[vid];
      PrintType(std::get<1>(arg), this->stream);
      if (v.type().is_handle()) this->stream << "*";
      this->stream << ' ' << std::get<0>(arg);
    }
  }
  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

std::string CodeGenMerlinC::Finish() { return CodeGenC::Finish(); }

void CodeGenMerlinC::BindThreadIndex(const IterVar& iv) {
  LOG(FATAL) << "Merlin doesn't support thread binding";
  return;
}

void CodeGenMerlinC::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    // LOG(FATAL) << "The buffer shouldn't call PrintType for printing type";
    os << "void*";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      case 128:
        os << "double double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "unsigned ";
    }
    if (t.bits() == 8 && t.lanes() == 4) {
      // directly 4 8 bit int in integer.
      os << "int";
      return;
    }

    int target_bit = 1;
    while (target_bit < t.bits()) target_bit <<= 1;

    switch (target_bit) {
      case 1:
        os << "int";
        break;
      case 2:
        os << "char";
        break;
      case 4:
        os << "char";
        break;
      case 8:
        os << "char";
        break;
      case 16:
        os << "short";
        break;
      case 32:
        os << "int";
        break;
      case 64:
        os << "long";
        break;
      case 128:
        os << "long";
        break;  // FIXME: Should use long long
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    // FIXME: Not yet support multiple lanes
    // if (!fail && (lanes >= 2 && lanes <= 16)) {
    //  os << lanes; return;
    //}
  }
  os << t;
  LOG(WARNING) << "Cannot convert type " << t;
  return;
}

void CodeGenMerlinC::PrintVecAddr(const Variable* buffer, Type t, Expr base,
                                  std::ostream& os) {  // NOLINT(*)
  // FIXME: What's this node for?
  if (!HandleTypeMatch(buffer, t.element_of())) {
    os << '(';
    auto it = alloc_storage_scope_.find(buffer);
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    os << ' ';
    PrintType(t.element_of(), os);
    os << "*)";
  }
  os << GetVarID(buffer) << " + ";
  PrintExpr(base, os);
  return;
}

void CodeGenMerlinC::PrintVecStore(const Variable* buffer, Type t, Expr base,
                                   const std::string& value) {
  // FIXME: What's this node for?
  this->PrintIndent();
  stream << "vstore" << t.lanes() << "(" << value << ", 0, ";
  PrintVecAddr(buffer, t, base, stream);
  stream << ");\n";
  return;
}

void CodeGenMerlinC::PrintStorageSync(const Call* op) {
  const std::string& sync = op->args[0].as<StringImm>()->value;
  if (sync == "warp") {
    LOG(FATAL) << "warp sync not supported in Merlin";
  } else if (sync == "shared") {
    LOG(FATAL) << "shared sync not supported in Merlin";
  } else if (sync == "global") {
    LOG(FATAL) << "global sync not supported in Merlin";
  }
  return;
}

void CodeGenMerlinC::PrintStorageScope(const std::string& scope,
                                       std::ostream& os) {  // NOLINT(*)
  return;
}

void CodeGenMerlinC::VisitExpr_(const Broadcast* op,
                                std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "((";
  PrintType(op->type, os);
  os << ")(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "))";
  return;
}

void CodeGenMerlinC::VisitStmt_(const LetStmt* op) {
  std::string value = PrintExpr(op->value);
  // Skip the argument retrieving assign statement
  std::string vid = AllocVarID(op->var.get());
  if (op->var.type() != Handle() &&
      value.find("TVMArray") == std::string::npos && value.find("arg") != 0) {
    PrintIndent();
    PrintType(op->var.type(), this->stream);
    this->stream << ' ' << vid << " = " << value << ";\n";
  }
  PrintStmt(op->body);
}

void CodeGenMerlinC::VisitStmt_(const For* op) {
  if (op->for_type == ForType::Parallel) {
    stream << "#pragma ACCEL parallel\n";
  } else if (op->for_type == ForType::Unrolled) {
    int unroll_factor = 0;
    int i = 0;
    for (auto key : op->annotate_keys) {
      if (auto str = key.as<StringImm>()) {
        auto factor = op->annotate_values[i].as<IntImm>();
        if (str->value == "factor" && factor != nullptr && factor->value > 1) {
          unroll_factor = factor->value;
          break;
        }
      }
      i++;
    }
    stream << "#pragma ACCEL parallel ";
    if (unroll_factor > 0) stream << "factor=" << unroll_factor << " ";
    stream << "flatten\n";
  } else if (op->for_type == ForType::Pipelined) {
    stream << "#pragma ACCEL pipeline\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenMerlinC::VisitStmt_(const IfThenElse* op) {
  std::string cond = PrintExpr(op->condition);

  // Skip the buffer data checking
  if (std::regex_match(cond, std::regex("!\\((arg)(.+)(== NULL)\\)"))) return;

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
}  // namespace codegen
}  // namespace TVM
