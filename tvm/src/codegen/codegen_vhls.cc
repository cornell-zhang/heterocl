/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.cc
 */
#include <tvm/build_module.h>
#include <vector>
#include <string>
#include <regex>
#include "./codegen_vhls.h"
#include "./build_common.h"

namespace tvm {
namespace codegen {

void CodeGenVivadoHLS::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  // Clear previous generated state
  this->InitFuncState(f);
  // Register alloc buffer type
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }
  // Write header files
  this->stream << "#include <ap_int.h>\n";
  this->stream << "#include <ap_fixed.h>\n";
  this->stream << "#include <math.h>\n";
  // Write entry function name
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
    }
    else {
      auto arg = map_arg_type[vid];
      PrintType(std::get<1>(arg), this->stream);
      if (v.type().is_handle())
        this->stream << "*";
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

void CodeGenVivadoHLS::PrintType(Type t, std::ostream& os) {
  if (t.is_uint()) {
    os << "ap_uint<" << t.bits() << ">";
  } else if (t.is_int()) {
    os << "ap_int<" << t.bits() << ">";
  } else if (t.is_ufixed()) {
    os << "ap_ufixed<" << t.bits() << ", " << t.fracs() << ">";
  } else if (t.is_fixed()) {
    os << "ap_fixed<" << t.bits() << ", " << t.fracs() << ">";
  } else {
    CodeGenC::PrintType(t, os);
  }
}

void CodeGenVivadoHLS::VisitStmt_(const Store* op) {
  // handle SetSlice
  if (const SetSlice* ss = op->value.as<SetSlice>()) {
    PrintIndent();
    this->stream << ss->a
                 << "(" << (ss->index_left - 1) << ", " << ss->index_right
                 << ") = " << ss->value << "\n";
    return;
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenVivadoHLS::VisitStmt_(const LetStmt* op) {
  std::string value = PrintExpr(op->value);
  // Skip the argument retrieving assign statement
  std::string vid = AllocVarID(op->var.get());
  if (op->var.type() != Handle() &&
      value.find("TVMArray") == std::string::npos &&
      value.find("arg") != 0) {
    PrintIndent();
    PrintType(op->var.type(), this->stream);
    this->stream << ' '
                 << vid
                 << " = " << value << ";\n";
  }
  PrintStmt(op->body);
}

void CodeGenVivadoHLS::VisitStmt_(const For* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  CHECK(is_zero(op->min));
  stream << "for (";
  PrintType(op->loop_var.type(), stream);
  stream << ' ' << vid << " = 0; "
            << vid << " < " << extent
            << "; ++" << vid << ") {\n";
  // pragmas begin
  if (op->for_type == ForType::Unrolled) {
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
    stream << "#pragma HLS unroll";
    if (unroll_factor > 0) {
      stream << " factor=" << unroll_factor << "\n";
    } else {
      stream << "\n";
    }
  }
  else if (op->for_type == ForType::Pipelined) {
    int II = 0;
    int i = 0;
    for (auto key : op->annotate_keys) {
      if (auto str = key.as<StringImm>()) {
        auto initiation_interval = op->annotate_values[i].as<IntImm>();
        if (str->value == "initiation_interval" &&
            initiation_interval != nullptr &&
            initiation_interval->value > 1) {
          II = initiation_interval->value;
          break;
        }
      }
      i++;
    }
    stream << "#pragma HLS pipeline";
    if (II > 0) {
      stream << " II=" << II << "\n";
    } else {
      stream << "\n";
    }
  }
  // pragmas end
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenVivadoHLS::VisitStmt_(const IfThenElse* op) {
  std::string cond = PrintExpr(op->condition);
  // Skip the buffer data checking
  if (std::regex_match(cond, std::regex("!\\((arg)(.+)(== NULL)\\)")))
      return ;
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
}  // namespace tvm
