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

CodeGenSDACCEL::CodeGenSDACCEL() {
    restrict_keyword_ = "restrict";
}

void CodeGenSDACCEL::InitFuncState(LoweredFunc f) {
    CodeGenC::InitFuncState(f);
    for (Var arg: f->args) {
        if (arg.type().is_handle()) {
            alloc_storage_scope_[arg.get()] = "global";
        }
    }
}


// void CodeGenSDACCEL::AddFunction(LoweredFunc f) {
//   this->stream << "__kernel ";
//   CodeGenC::AddFunction(f);
// }

// void CodeGenSDACCEL::AddFunction(LoweredFunc f) {
  // this->stream << "# pragma once\n";
  // this->stream << "# define CL_HPP_CL_1_2_DEFAULT_BUILD\n";
  // this->stream << "# define CL_HPP_TARGET_OPENCL_VERSION 120\n";
  // this->stream << "# define CL_HPP_MINIMUM_OPENCL_VERSION 120\n";
  // this->stream << "# define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1\n";
  // this->stream << "# include <CL/cl2.hpp>\n";
  // this->stream << "# include <fstream>\n";
  // this->stream << "# include <cstdlib>\n";
  // this->stream << "# include <cstdio>\n";
  // this->stream << "# include <iostream>\n";
  // this->stream << "# include <vector>\n\n";
  // this->stream << "__kernel ";
  
//   CodeGenC::AddFunction(f);
// }

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

  // Write head files
  // stream.open("host.cpp");
  // this->stream << "# pragma once\n";
  // this->stream << "# define CL_HPP_CL_1_2_DEFAULT_BUILD\n";
  // this->stream << "# define CL_HPP_TARGET_OPENCL_VERSION 120\n";
  // this->stream << "# define CL_HPP_MINIMUM_OPENCL_VERSION 120\n";
  // this->stream << "# define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1\n";
  // this->stream << "# include <CL/cl2.hpp>\n";
  // this->stream << "# include <fstream>\n";
  // this->stream << "# include <cstdlib>\n";
  // this->stream << "# include <cstdio>\n";
  // this->stream << "# include <iostream>\n";
  // this->stream << "# include <vector>\n\n";

  // Write entry function name
  // this->stream << "__kernel " << f->name << "(";
  // this->stream << "__kernel " << "void " << "__attribute__ " << "((reqd_work_group_size(1, 1, 1)))\n";
  // this->stream << f->name << "(";
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




// void CodeGenSDACCEL::AddFunction(LoweredFunc f, 
//   str2tupleMap<std::string, Type> map_arg_type) {
//     // Don't Write header flies
//     // Clear previous generated state
//     this->InitFuncState(f);    
//     // Register alloc buffer type
//     for ( const auto & kv : f->handle_data_type ) {
//       this->stream << kv.first.get();
//       this->stream << kv.second.type();
//       RegisterHandleType(kv.first.get(), kv.second.type());
//     }
//     // Write entry function name
//     this->stream << "__kernel ";
//     // Write arguments
//     for ( size_t i = 0; i < f->args.size(); i++ ) {
//       Var v = f->args[i];
//       std::string vid = AllocVarID(v.get());
//       if ( i!= 0 ) {
//         this->stream << ", ";
//       }
//       if ( map_arg_type.find(vid) == map_arg_type.end()) {
//         LOG(WARNING) << vid << " type not found\n";
//         PrintType(v.type(), this->stream);
//         this->stream << ' ' << vid;
//       }
//       else {
//         auto arg = map_arg_type[vid];
//         PrintType(std::get<1>(arg), this->stream);
//         if (v.type().is_handle()) {
//           this->stream << "*";
//         }
//         this->stream << ' ' << std::get<0>(arg);

//       }
//       stream << ") {\n";
//       int func_scope = this->BeginScope();
//       this->PrintStmt(f->body);
//       this->EndScope(func_scope);
//       this->PrintIndent();
//       this->stream << "}\n\n";
//     }
//     CodeGenSDACCEL::AddFunction(f, map_arg_type);
// }

std::string CodeGenSDACCEL::Finish() {
  // inject extension enable pragma for fp16 and fp64
  if (enable_fp16_) {
    decl_stream
        << "#ifdef cl_khr_fp16\n"
           "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
           "#elif defined(cl_amd_fp16)\n"
           "#pragma OPENCL EXTENSION cl_amd_fp16 : enable\n"
           "#else\n"
           "#error \"Half precision floating point not supported"
                    "by OpenCL implementation on your device.\" \n"
           "#endif\n\n";
  }

  if (enable_fp64_) {
    decl_stream
        << "#ifdef cl_khr_fp64\n"
           "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
           "#elif defined(cl_amd_fp64)\n"
           "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
           "#else\n"
           "#error \"Double precision floating point not supported"
                    "by OpenCL implementation on your device.\" \n"
           "#endif\n\n";
  }

  return CodeGenC::Finish();
}

void CodeGenSDACCEL::BindThreadIndex(const IterVar& iv) {
  CHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::make(iv->thread_tag);
  std::ostringstream os;
  if (ts.rank == 1) {
    os << "get_local_id(" << ts.dim_index << ")";
  } else {
    os << "get_group_id(" << ts.dim_index << ")";
  }
  var_idmap_[iv->var.get()] =
      CastFromTo(os.str(), UInt(64), iv->var.type());
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
    // if (!fail && (lanes >= 2 && lanes <= 16)) {
    //   os << lanes; return;
    // }
  }
  // LOG(FATAL) << "Cannot convert type " << t << " to OpenCL C type";
}


// void CodeGenSDACCEL::PrintType(Type t, std::ostream& os) {  // NOLINT(*)
//   CHECK_EQ(t.lanes(), 1)
//       << "do not yet support vector types";
//   if (t.is_handle()) {
//     os << "void*"; return;
//   }
//   if (t.is_float()) {
//     if (t.bits() == 16) {
//       enable_fp16_ = true;
//       os << "half"; return;
//     }
//     if (t.bits() == 32) {
//       os << "float"; return;
//     }
//     if (t.bits() == 64) {
//       enable_fp64_ = true;
//       os << "double"; return;
//     }
//   } else if (t.is_uint() || t.is_int()) {
//     if (t.is_uint()) {
//       os << 'u'; 
//     }
//     if (t.bits() == 8 && t.lanes() == 4) {
//       os << "int"; return;
//     }
//     switch (t.bits()) {
//       case 8: os << "char"; break;
//       case 16: os << "short"; break;
//       case 32: os << "int"; break;
//       case 64: os << "long"; break;
//       case 1: os << "int"; break;
//     }
//   }

//   os << t;
// }




void CodeGenSDACCEL::PrintVecAddr(const Variable* buffer, Type t,
                                 Expr base, std::ostream& os) {  // NOLINT(*)
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
}
std::string CodeGenSDACCEL::GetVecLoad(
    Type t, const Variable* buffer, Expr base) {
  std::ostringstream os;
  os << "vload" << t.lanes() << "(0, ";
  PrintVecAddr(buffer, t, base, os);
  os << ")";
  return os.str();
}

void CodeGenSDACCEL::PrintVecStore(const Variable* buffer,
                                  Type t, Expr base,
                                  const std::string& value) {
  this->PrintIndent();
  stream << "vstore" << t.lanes() << "(" << value << ", 0, ";
  PrintVecAddr(buffer, t, base, stream);
  stream << ");\n";
}

void CodeGenSDACCEL::PrintStorageSync(const Call* op) {
  const std::string& sync = op->args[0].as<StringImm>()->value;
  if (sync == "warp") {
    LOG(FATAL) << "warp sync not supported in opencl";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "barrier(CLK_LOCAL_MEM_FENCE);\n";
  } else if (sync == "global") {
    LOG(FATAL) << "not supported";
  }
}

// void CodeGenSDACCEL::PrintStorageScope(
//     const std::string& scope, std::ostream& os) { // NOLINT(*)
//   if (scope == "global") {
//     os << "__global ";
//   } else if (scope == "shared") {
//     os << "__local ";
//   }
// }

void CodeGenSDACCEL::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
  if (scope == "global") {
    os << "global ";
  } else if (scope == "shared") {
    os << "local ";
  }
}


std::string CodeGenSDACCEL::CastFromTo(std::string value, Type from, Type target) {
  if (from == target) return value;
  std::ostringstream os;
  if (target.lanes() == 1) {
    os << "((";
    this->PrintType(target, os);
    os << ")" << value << ")";
  } else {  // convert vector type
    os << "(";
    os << "convert_";
    this->PrintType(target, os);
    os << "(" << value << "))";
  }
  return os.str();
}

void CodeGenSDACCEL::VisitExpr_(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "((";
  PrintType(op->type, os);
  os << ")(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "))";
}

void CodeGenSDACCEL::VisitExpr_(const Call * op, std::ostream& os) { // NOLINT(*)
    if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
        os << "(";
        PrintType(op->args[2].type(), os);
        os << ")";
    }
    CodeGenC::VisitExpr_(op, os);
}

void CodeGenSDACCEL::VisitStmt_(const LetStmt* op) {
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


void CodeGenSDACCEL::VisitExpr_(const FloatImm * op, std::ostream& os) { // NOLINT(*)
    if (std::isinf(op->value)) {
        if ( op->value < 0) {
            os << "-";
        }
        os << "INFINITY";
    } else if (std::isnan(op->value)) {
        os << "NAN";
    } else {
        CodeGenC::VisitExpr_(op, os);
    }
}

void CodeGenSDACCEL::VisitExpr_(const Select * op, std::ostream& os ) { // NOINT(*)
    os << "(";
    PrintType(op->true_value.type(), os);
    os << ")";
    CodeGenC::VisitExpr_(op, os);
} 

void CodeGenSDACCEL::VisitStmt_(const IfThenElse* op) {
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


void CodeGenSDACCEL::GenForStmt(const For* op, std::string pragma, bool before) {
  std::string extent = PrintExpr(op->extent);
  std::string vid = AllocVarID(op->loop_var.get());
  CHECK(is_zero(op->min));
  if (before && pragma.length() > 0) {
    PrintIndent();
    stream << pragma;
  }
  PrintIndent();
  stream << "for (";
  PrintType(op->loop_var.type(), stream);
  stream << ' ' << vid << " = 0; "
            << vid << " < " << extent
            << "; ++" << vid << ") {\n";
  if (!before && pragma.length() > 0) {
    PrintIndent();
    stream << pragma;
  }
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
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

// void CodeGenSDACCEL::VisitStmt_(const Partition* op) {
//   std::string vid = GetVarID(op->buffer_var.get());
//   stream << vid << " ";
//   stream << "__attribute__((xcl_array_partition(";
//   switch (op->partition_type) {
//     case PartitionType::Complete:
//       stream << "complete,";
//       break;
//     case PartitionType::Block:
//       stream << "block,";
//       break;
//     case PartitionType::Cyclic:
//       stream << "cyclic,";
//       break;
//   }
//   if (op->partition_type != PartitionType::Complete) {
//     stream << op->factor << ",";
//     // stream << " factor=" << op->factor;
//   }
//   stream << op->dim << ")))";

//   stream << "\n";
// }


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
