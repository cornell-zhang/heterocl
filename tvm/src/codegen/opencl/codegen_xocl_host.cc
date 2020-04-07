/*!
 *  Copyright (c) 2020 by Contributors
 * \file codegen_xocl_host.cc
 */
#include <tvm/build_module.h>
#include <tvm/ir_pass.h>
#include <vector>
#include <string>
#include <regex>
#include "./codegen_xocl_host.h"
#include "../build_common.h"

namespace TVM {
namespace codegen {

void CodeGenXOCLHost::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  CodeGenC::AddFunction(f, map_arg_type);
}

void CodeGenXOCLHost::PrintType(Type t, std::ostream& os) {
  CodeGenC::PrintType(t, os);
}

std::string CodeGenXOCLHost::GetBufferRef(Type t, const Variable* buffer, Expr index) {
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

void CodeGenXOCLHost::VisitExpr_(const Min *op, std::ostream& os) {  // NOLINT(*)
  os << "std::min(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenXOCLHost::VisitExpr_(const Max *op, std::ostream& os) {  // NOLINT(*)
  os << "std::max(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenXOCLHost::VisitStmt_(const For* op) {
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

void CodeGenXOCLHost::VisitStmt_(const Store* op) {
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

void CodeGenXOCLHost::GenForStmt(const For* op, std::string pragma, bool before) {
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

void CodeGenXOCLHost::VisitStmt_(const IfThenElse* op) {
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

void CodeGenXOCLHost::VisitStmt_(const Allocate* op) {
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

  bool not_alloc = false;
  if (vid.find("_new") != std::string::npos) {
    not_alloc = true;
    vid.replace(vid.find("_new"), 4, "");
    var_idmap_[op->buffer_var.get()] = vid; 

  // skip if buffer allocated in host scope 
  } else if (vid.find("_channel") != std::string::npos) {
    vid.replace(vid.find("_channel"), 8, "");

    // handle output-update-in-kernel case
    if (vid.find("_update") != std::string::npos) {
      auto name = var_idmap_[op->buffer_var.get()]; 
      name.replace(name.find("_update"), 7, "");
      vid.replace(vid.find("_update"), 7, "");
      var_idmap_[op->buffer_var.get()] = name;
    }

    if (alloc_set_.find(vid) != alloc_set_.end()) {
      not_alloc = true;
    }
  }

  // not allocate for moved data  
  if (!not_alloc) { 
    PrintType(op->type, stream);
    alloc_set_.insert(vid);
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

void CodeGenXOCLHost::VisitStmt_(const KernelStmt* op) {
  std::string name = op->name;
  // extract annotation information 
  // initialize buffers and opencl kernel 
  int mem_count = 0;
  if (name.find("test") != std::string::npos) {

    // create kernels
    stream << "\n";
    PrintIndent();

    stream << "cl::Kernel kernel(program, \""
           << name << "\", &err);\n";

    // create device buffers
    std::vector<std::string> kernel_args;
    for (size_t k = 0; k < op->args.size(); k++) {
      auto v = op->args[k].as<Variable>();
      CHECK(v) << "invalid input var";
      auto shape = var_shape_map_[v];
      if (shape.size() == 0) {
        kernel_args.push_back(PrintExpr(op->args[k]));
        continue;
      }

      std::string arg_name = PrintExpr(op->args[k]);
      CHECK(arg_name.find("_channel")) 
        << op->args[k] << " not a channel";
      arg_name.replace(arg_name.find("_channel"), 8, "");
      kernel_args.push_back(arg_name);
 
      PrintIndent();
      stream << "cl::Buffer buffer_" 
             << arg_name
             << "(context, " 
             << "CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, "
             << "sizeof(";
      PrintType(handle_data_type_[v], stream);
      stream << ")*";
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) stream << "*";
        stream << shape[i];
      }

      stream << ", " << arg_name
             << ", &err);\n";
      mem_count += 1;
    }

    // set kernel arguments
    stream << "\n  // set device kernel buffer\n";
    CHECK(op->args.size() == kernel_args.size());
    for (size_t k = 0; k < kernel_args.size(); k++) {
      PrintIndent();
      stream << "err = kernel.setArg(" << k << ", "
             << "buffer_" << kernel_args[k] << ");\n";
    }

    // migrate memory objects
    PrintIndent();
    stream << "err = q.enqueueMigrateMemObjects({";
    for (size_t k = 0; k < kernel_args.size(); k++) {
      if (k != 0) stream << ", ";
      stream << "buffer_" << kernel_args[k];
    }
    stream << "}, 0/*from host*/);\n";

    // launch kernel execution  
    stream << "\n  // enqueue kernel function\n";
    PrintIndent();
    stream << "cl::Event event;\n";
    PrintIndent();
    stream << "err = q.enqueueTask(kernel, NULL, &event);\n";

    // retrieve data from global buffer 
    PrintIndent();
    stream << "err = q.enqueueMigrateMemObjects({";
    for (size_t k = 0; k < kernel_args.size(); k++) {
      if (k != 0) stream << ", ";
      stream << "buffer_" << kernel_args[k];
    }
    stream << "}, CL_MIGRATE_MEM_OBJECT_HOST);\n";

    PrintIndent();
    stream << "err = q.finish();\n";
    stream << "\n  // execution on host \n";
  
  } else {  
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
