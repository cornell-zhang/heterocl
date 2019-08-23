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
# include "./codegen_sdaccel.h"
# include "../../runtime/thread_storage_scope.h"

namespace TVM {
namespace codegen {

void CodeGenSDACCEL::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  // Clear previous generated state
  this->InitFuncState(f);

  // Skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");

  // Register alloc buffer type
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }


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


void CodeGenSDACCEL::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    CHECK_EQ(lanes, 1)
        << "do not yet support vector types";
    os << "void*"; return;
  }
  if ( t== Bool() ) {
      os << "bool"; return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        enable_fp16_ = true;
        break;
      case 32: 
        os << "float"; 
        break;
      case 64:
        os << "double";
        enable_fp64_ = true;
        break;
      default: 
        fail = true; 
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes; return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    if (t.bits() == 8 && t.lanes() == 4) {
      // directly 4 8 bit int in integer.
      os << "int"; return;
    }
    switch (t.bits()) {
      case 8: os << "char"; break;
      case 16: os << "short"; break;
      case 32: os << "int"; break;
      case 64: os << "long"; break;
      case 1: os << "int"; break;
      default: fail = true; break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes; return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to SDAccel type";
}




void CodeGenSDACCEL::VisitStmt_(const For* op) {
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
    os << "__attribute__((opencl_unroll_hint(";
    if (unroll_factor > 0) os << unroll_factor << ")))\n";
    else
      os << "\n";

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
    os << "__attribute__((xcl_pipeline_loop(";
    os << II << ")))\n";
  }
  CodeGenSDACCEL::GenForStmt(op, os.str(), true);
}



void CodeGenSDACCEL::VisitStmt_(const Partition* op) {
  std::string vid = GetVarID(op->buffer_var.get());
  stream << vid << " ";
  if (op->partition_type != PartitionType::Complete) {
    stream << "__attribute__((xcl_array_partition(";
    switch (op->partition_type) {
      // case PartitionType::Complete:
      //   stream << "complete,";
      //   break;
      case PartitionType::Block:
        stream << "block,";
        break;
      case PartitionType::Cyclic:
        stream << "cyclic,";
        break;
      }
    stream << op->factor << ",";
    stream << op->dim << ")))\n";
  }else {
    if (op->dim == 0) {
      stream << "__attribute__((xcl_array_partition))\n";
    } else {
      stream << "__attribute__((xcl_array_partition(";
      stream << "complete,";
      stream << op->factor << ",";
      stream << op->dim << ")))\n";
      }
    }
}


} // namespace codegen
} // namespace TVM
