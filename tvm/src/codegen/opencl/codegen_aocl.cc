/*
    Yang.Bai
    yb269@cornell.edu
*/
# include <regex>
# include <tvm/runtime/config.h>
# include <tvm/packed_func_ext.h>
# include <vector>
# include <string>
# include <regex>
# include "./codegen_aocl.h"
# include "../../runtime/thread_storage_scope.h"

namespace TVM {
namespace codegen {


void CodeGenAOCL::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  // Clear previous generated state
  this->InitFuncState(f);

  // Skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");

  // Register alloc buffer type
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }


  this->stream << "#pragma OPENCL EXTENSION cl_intel_arbitrary_precision_integers : enable" << "\n";
  this->stream << "__kernel " << "void " << f->name << "(";

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
      this->stream << "__global ";
      // this->stream << "global ";
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
  // this->stream << ' '<< ' ' << "return;\n";
  this->stream << "}\n\n";
}



void CodeGenAOCL::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  CHECK_EQ(t.lanes(), 1)
      << "do not yet support vector types";
  if (t.is_handle()) {
    os << "void*"; return;
  }

  if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "ap_uint<" << t.bits() << ">" << "uintd_t";
    }
    else if ( t.is_int()) {
      os << "ap_int<" << t.bits() << ">" << "intd_t";
    }
    else {
      if (t.is_float()) {
        if (t.bits() == 16) {
          enable_fp16_ = true;
          os << "half"; return;
        }
        if (t.bits() == 32) {
          os << "float"; return;
        }
        if (t.bits() == 64) {
          enable_fp64_ = true;
          os << "double"; return;
        }
      } else if (t.is_uint()) {
        switch (t.bits()) {
          case 8: case 16: case 32: case 64: {
            os << "ap_uint<" << t.bits() << ">" << "uintd_t"; return;
            // os << "uint" << t.bits() << "_t"; return;
          }
          case 1: os << "int"; return;
        }
      } else if (t.is_int()) {
        switch (t.bits()) {
          case 8: case 16: case 32: case 64: {
            os << "ap_int<" << t.bits() << ">" << "intd_t"; return; 
            // os << "int" << t.bits() << "_t";  return;
          }
        }
      }
    }
  }
}


void CodeGenAOCL::VisitStmt_(const For* op) {
  std::ostringstream os;
  if (op->for_type == ForType::Unrolled) {
    int unroll_factor = 0, i = 0;
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
    os << "#pragma unroll";
    if (unroll_factor > 0) os << " " << unroll_factor << "\n";
    else                   os << "\n";
  }
  else if (op->for_type == ForType::Pipelined) {
    int II = 1, i = 0;
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
    os << "#pragma";
    os << " ii " << II << "\n";
  }
  CodeGenAOCL::GenForStmt(op, os.str(), true);
}




} // namespace codegen
} // namespace TVM
