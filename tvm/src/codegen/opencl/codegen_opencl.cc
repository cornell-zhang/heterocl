# include <tvm/runtime/config.h>
# include <tvm/packed_func_ext.h>
# include <tvm/ir_pass.h>
# include <vector>
# include <string>
# include <cmath>
# include <regex>
# include "./codegen_opencl.h"
# include "../../runtime/thread_storage_scope.h"

namespace TVM{
namespace codegen{
  
CodeGenOpenCL::CodeGenOpenCL(){
  restrict_keyword_ = "restrict";
}

std::string CodeGenOpenCL::Finish() {
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

void CodeGenOpenCL::BindThreadIndex(const IterVar& iv) {
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

void CodeGenOpenCL::PrintArray(const Array<Expr>& array, const std::vector<size_t>& extents, std::ostringstream& stream, size_t offset, size_t level) {
  // check if is the last level
  if (level == extents.size()-1) {
    // stream << "{";
    for (size_t i = 0; i < extents[level]; i++) {
      PrintExpr(array[offset+i], stream);
      if (i != extents[level]-1) stream << ", ";
    }
    // stream << "}";
  } else {
    // stream << "{";
    for (size_t i = 0; i < extents[level]; i++) {
      size_t size = 1;
      for (size_t j = level+1; j < extents.size(); j++) {
        size *= extents[j];
      }
      PrintArray(array, extents, stream, offset + size*i, level+1);
      if (i != extents[level]-1) stream << ", ";
    }
    // stream << "}";
  }
}

void CodeGenOpenCL::PrintVecAddr(const Variable* buffer, Type t,
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
std::string CodeGenOpenCL::GetVecLoad(
    Type t, const Variable* buffer, Expr base) {
  std::ostringstream os;
  os << "vload" << t.lanes() << "(0, ";
  PrintVecAddr(buffer, t, base, os);
  os << ")";
  return os.str();
}

std::string CodeGenOpenCL::GetBufferRef(Type t, const Variable* buffer, Expr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  if (t.lanes() == 1) {
    bool is_scalar = (buf_length_map_.count(buffer) == 1 &&
        buf_length_map_[buffer] == 1);
    if (is_scalar) {
      os << vid;
    } else { 
      os << vid;
      CHECK(var_shape_map_.count(buffer)) 
        << "buffer " << buffer->name_hint << " not found in var_shape_map";
      // Checking scope of the buffer
      if (top_args.count(vid) && !enable_native_dtype) {
        os << "[";
        PrintExpr(index, os); 
        os << "]";
      } else {
        std::vector<Expr> indices = ExtractIndices(index, var_shape_map_[buffer], range_);
        for (size_t i = 0; i < indices.size(); i++) {
          os << '[';
          PrintExpr(indices[i], os);
          os << ']';
        }
      }
    }
  }  
  return os.str();
}

void CodeGenOpenCL::PrintVecStore(const Variable* buffer,
                                  Type t, Expr base,
                                  const std::string& value) {
  this->PrintIndent();
  stream << "vstore" << t.lanes() << "(" << value << ", 0, ";
  PrintVecAddr(buffer, t, base, stream);
  stream << ");\n";
}

void CodeGenOpenCL::PrintStorageSync(const Call* op) {
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

void CodeGenOpenCL::PrintStorageScope(
    const std::string& scope, std::ostream& os) { // NOLINT(*)
  if (scope == "global") {
    // os << "global ";
  } else if (scope == "shared") {
    // os << "local ";
  }
}

std::string CodeGenOpenCL::CastFromTo(std::string value, Type from, Type target) {
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

void CodeGenOpenCL::VisitExpr_(const Broadcast* op, std::ostream& os) {   // NOLINT(*)
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

void CodeGenOpenCL::VisitExpr_(const Call * op, std::ostream& os) { // NOLINT(*)
    if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
        os << "(";
        PrintType(op->args[2].type(), os);
        os << ")";
    }
    CodeGenC::VisitExpr_(op, os);
}

void CodeGenOpenCL::VisitStmt_(const LetStmt* op) {
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


void CodeGenOpenCL::VisitExpr_(const FloatImm * op, std::ostream& os) { // NOLINT(*)
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

void CodeGenOpenCL::VisitExpr_(const Select * op, std::ostream& os ) { // NOINT(*)
    os << "(";
    PrintType(op->true_value.type(), os);
    os << ")";
    CodeGenC::VisitExpr_(op, os);
} 

void CodeGenOpenCL::VisitStmt_(const IfThenElse* op) {
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

void CodeGenOpenCL::GenForStmt(const For* op, std::string pragma, bool before) {
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

} // namespace codegen
} // namespace TVM
