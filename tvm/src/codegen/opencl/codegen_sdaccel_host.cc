/*!
 *  Copyright (c) 2020 by Contributors
 * \file codegen_sdaccel_host.cc
 */
#include <tvm/build_module.h>
#include <tvm/ir_pass.h>
#include <vector>
#include <string>
#include <regex>
#include "./codegen_sdaccel_host.h"
#include "../build_common.h"

namespace TVM {
namespace codegen {

void CodeGenSDAccelHost::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  CodeGenC::AddFunction(f, map_arg_type);
}

void CodeGenSDAccelHost::PrintType(Type t, std::ostream& os) {
  if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed()) {
    if (t.is_uint())        os << "ap_uint<" << t.bits() << ">";
    else if (t.is_int())    os << "ap_int<" << t.bits() << ">";
    else if (t.is_ufixed()) os << "ap_ufixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
    else                    os << "ap_fixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
  } else {
    CodeGenC::PrintType(t, os);
  }
}

std::string CodeGenSDAccelHost::GetBufferRef(Type t, const Variable* buffer, Expr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  if (t.lanes() == 1) {
    bool is_scalar = (buf_length_map_.count(buffer) == 1 &&
        buf_length_map_[buffer] == 1);
    if (is_scalar) {
      os << vid;
    } else { 
      os << vid << "[";
      PrintExpr(index, os);
      os << "]";
    }
  }  
  return os.str();
}

void CodeGenSDAccelHost::VisitExpr_(const Min *op, std::ostream& os) {  // NOLINT(*)
  os << "std::min(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenSDAccelHost::VisitExpr_(const Max *op, std::ostream& os) {  // NOLINT(*)
  os << "std::max(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenSDAccelHost::VisitStmt_(const For* op) {
  // ignore the data tranmission for stmts
  if (const For* for_op = op->body.as<For>()) {
    while (for_op->body.as<For>())
      for_op = for_op->body.as<For>();
    if (for_op->body.as<StreamStmt>()) { 
      return;
    } else if (auto st = for_op->body.as<Store>()) {
      if (st->value.as<StreamExpr>()) return;
    }
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenSDAccelHost::VisitStmt_(const Store* op) {
  // handle SetSlice
  if (const SetSlice* ss = op->value.as<SetSlice>()) {
    Type t = op->value.type();
    Expr new_index_left = ir::Simplify(ss->index_left - 1);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    PrintIndent(); 
    this->stream << ref
                 << "(" << PrintExpr(new_index_left) << ", " << PrintExpr(ss->index_right)
                 << ") = " << PrintExpr(ss->value) << ";\n";
  } else if (const SetBit* sb = op->value.as<SetBit>()) {
    Type t = op->value.type();
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    PrintIndent();
    this->stream << ref
                 << "[" << PrintExpr(sb->index)
                 << "] = " << PrintExpr(sb->value) << ";\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenSDAccelHost::GenForStmt(const For* op, std::string pragma, bool before) {
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

void CodeGenSDAccelHost::VisitStmt_(const IfThenElse* op) {
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

void CodeGenSDAccelHost::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  this->PrintIndent();
  int32_t constant_size = op->constant_allocation_size();
  CHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";
  const Variable* buffer = op->buffer_var.as<Variable>();
  var_shape_map_[buffer] = op->extents;

  std::string scope; // allocate on local scope by default 
  auto it = alloc_storage_scope_.find(buffer);
  if (it != alloc_storage_scope_.end())
    scope = alloc_storage_scope_.at(buffer);
  else scope = "local";
  PrintStorageScope(scope, stream);

  // do not allocate host-to-global channel
  if (vid.find("_channel") == std::string::npos &&
      vid.find("_new") == std::string::npos) { 
    PrintType(op->type, stream);
    stream << ' '<< vid;
    if (constant_size > 1) {// Transfer length one array to scalar
      stream << "[";
      for (size_t i = 0; i < op->extents.size(); i++) {
        PrintExpr(op->extents[i], stream);
        if (i != op->extents.size()-1) stream << " * ";
      }
      stream << "]";
    }
    stream << ";\n";
  }
  buf_length_map_[buffer] = constant_size;
  RegisterHandleType(op->buffer_var.get(), op->type);
  for (size_t i = 0; i < op->attrs.size(); i++) {
    this->PrintStmt(op->attrs[i]);
  }
  this->PrintStmt(op->body);
}

void CodeGenSDAccelHost::VisitStmt_(const KernelStmt* op) {
  std::string name = op->name;
  // extract annotation information 
  // initialize buffers and opencl kernel 
  int mem_count = 0;
  if (name.find("top_function_") != std::string::npos) {
    // create kernels
    stream << "\n";
    PrintIndent();
    stream << "CLKernel " << name 
           << "(world.getContext(), world.getProgram(), \""
           <<  name << "\", world.getDevice());\n"; 

    for (size_t k = 0; k < op->args.size(); k++) {

      auto v = op->args[k].as<Variable>();
      CHECK(v) << "invalid input var";
      auto shape = var_shape_map_[v];
      if (shape.size() == 0) continue;

      PrintIndent();
      stream << "CLMemObj source_" << mem_count << "((void*)"; 
      PrintExpr(op->args[k], stream); 
      stream << ", sizeof(";
      PrintType(handle_data_type_[v], stream);
      stream << "), ";
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) stream << " *";
        stream << shape[i];
      }
      stream  << ", CL_MEM_READ_WRITE);\n";
      mem_count += 1;

    }

    // set up memory and work size
    for (int i = 0; i < mem_count; i++) {
      PrintIndent();
      stream << "world.addMemObj(source_" << i << ");\n";
    }
    stream << R"(
  int global_size[3] = {1, 1, 1};
  int local_size[3]  = {1, 1, 1};
  )";
    stream << name << ".set_global(global_size);\n"; 
    PrintIndent();
    stream << name << ".set_local(local_size);\n"; 
    PrintIndent();
    stream << "world.addKernel(" << name << ");\n\n";
  
    // set arguments for kernel
    mem_count = 0;
    for (size_t k = 0; k < op->args.size(); k++) {
      auto v = op->args[k].as<Variable>();
      CHECK(v) << "invalid input var";
      auto shape = var_shape_map_[v];
      if (shape.size() == 0) {
        PrintIndent();
        stream << "world.setIntKernelArg(0, " << k << ", ";
        PrintExpr(op->args[k], stream); 
        stream << ");\n";
        continue;
      };

      stream << "";
      PrintIndent();
      stream << "world.setMemKernelArg(0, " << k << ", "
             << mem_count << ");\n";
      mem_count += 1;
    }

    // launch kernel function 
    stream << "\n";
    PrintIndent();
    stream << "world.runKernels();\n";
  
    // collecting data back from global ddr 
    for (int k = 0; k < mem_count; k++) {
      PrintIndent();
      stream << "world.readMemObj(" << k << ");\n";
    }

  } else { // regular sub-function 
    PrintIndent();
    stream << op->name << "(";
    for (size_t i = 0; i < op->args.size(); i++) {
      PrintExpr(op->args[i], stream);
      if (i < op->args.size() -1) stream << ", ";
    }
    stream << ");\n";
  }


}

}  // namespace codegen
}  // namespace TVM
