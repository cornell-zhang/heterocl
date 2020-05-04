/*
 * \file codegen_rv64_ppac.cc
 */
 
#include <tvm/build_module.h>
#include <tvm/ir_pass.h>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <sys/types.h>
#include "./codegen_rv64_ppac.h"
#include "../build_common.h"

namespace TVM {
namespace codegen {

void CodeGenRV64PPAC::AddFunction(LoweredFunc f, 
        str2tupleMap<std::string, Type> map_arg_type) {
  // Clear previous generated state
  this->InitFuncState(f);
  // Register alloc buffer type
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }
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

void CodeGenRV64PPAC::VisitStmt_(const For* op) {
  std::string func_name;
  bool is_ppac_func = false;
  uint8_t i = 0;
  for (auto key: op->annotate_keys) {
    if (auto str = key.as<StringImm>()) {
      if (str->value == "_ppac_func_name") {
        auto name = op->annotate_values[i].as<StringImm>();
        func_name = name->value;
        is_ppac_func = true;
        break;
      }
    }
    ++i;
  }
  if (is_ppac_func) {
    // scan along the annotate list to find parameters
    std::string ret, arg0, arg1;
    int batch_num, in_block_num, out_channel_num;
    i = 0;
    uint8_t param_num = 0;
    for (auto key: op->annotate_keys) {
      if (auto str = key.as<StringImm>()) {
        if (str->value == "_ret") {
          auto v = op->annotate_values[i].as<StringImm>();
          ret = v->value;
          ++param_num;         
        } else if (str->value == "_arg0") {
          auto v = op->annotate_values[i].as<StringImm>();
          arg0 = v->value;
          ++param_num;         
        } else if (str->value == "_arg1") {
          auto v = op->annotate_values[i].as<StringImm>();
          arg1 = v->value;
          ++param_num;         
        } else if (str->value == "_batch_num") {
          auto v = op->annotate_values[i].as<IntImm>();
          batch_num = v->value;
          ++param_num;
        } else if (str->value == "_in_block_num") {
          auto v = op->annotate_values[i].as<IntImm>();
          in_block_num = v->value;
          ++param_num;
        } else if (str->value == "_out_channel_num") {
          auto v = op->annotate_values[i].as<IntImm>();
          out_channel_num = v->value;
          ++param_num;
        }
      }
      ++i;
    }
    if (param_num != 6) {
      LOG(FATAL) << "PPAC function call need exactly 6 parameters but found " << param_num;
    }
    // print ppac function call
    PrintIndent();
    stream << func_name << "(" 
           << ret << ", "
           << arg0 << ", " 
           << arg1 << ", "
           << batch_num << ", " 
           << in_block_num << ", "
           << out_channel_num 
           << ");\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenRV64PPAC::VisitStmt_(const LetStmt* op) {
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

void CodeGenRV64PPAC::VisitStmt_(const IfThenElse* op) {
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

void CodeGenRV64PPAC::PrintType(Type t, std::ostream& os) {
  CHECK_EQ(t.lanes(), 1)
      << "do not support vector types";
  if (t.is_uint() || t.is_int()) {
    if (t.is_uint())  {
      if (t.bits() <= 8) {
        os << "uint8_t"; return;
      } else if (t.bits() <= 16) {
        os << "uint16_t"; return;
      } else if (t.bits() <= 32) {
        os << "uint32_t"; return;
      } else if (t.bits() <= 64) {
        os << "uint64_t"; return;
      } else {
        LOG(WARNING) << "Casting type " << t << " to uint64_t";
        os << "uint64_t"; 
        return;
      }
    }
    else if (t.is_int()) {
      if (t.bits() <= 8) {
        os << "int8_t"; return;
      } else if (t.bits() <= 16) {
        os << "int16_t"; return;
      } else if (t.bits() <= 32) {
        os << "int32_t"; return;
      } else if (t.bits() <= 64) {
        os << "int64_t"; return;
      } else {
        LOG(WARNING) << "Casting type " << t << " to int64_t";
        os << "int64_t"; 
        return;
      }
    }
  }
  os << t;
}

} //namespace codegen
} //namespace TVM