/*!
 *  Copyright (c) 2020 by Contributors
 * \file codegen_aocl_host.cc
 */
#include <tvm/build_module.h>
#include <tvm/ir_pass.h>
#include <vector>
#include <string>
#include <regex>
#include "./codegen_aocl_host.h"
#include "../build_common.h"

namespace TVM {
namespace codegen {

void CodeGenAOCLHost::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  CodeGenC::AddFunction(f, map_arg_type);
}

void CodeGenAOCLHost::PrintType(Type t, std::ostream& os) {
  if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed()) {

    if (t.is_uint()) {
      switch (t.bits()) {
        case 32: 
          os << "uint";
          break;
        case 64:
          os << "uint64_t";
          break;
        default:
          os << "uint";
          break;
      }

    } else if (t.is_int()) {
      switch (t.bits()) {
        case 32: 
          os << "int";
          break;
        case 64:
          os << "int64_t";
          break;
        default:
          os << "uint";
          break;
      }
    } else if(t.is_float()) {
      os << "cl_float";
    } else if (t.is_fixed() || t.is_ufixed()) {
      int bits = t.bits() <= 64 ? t.bits() : 64;
      if (t.is_fixed()) {
        os << "int" << bits << "_t";
      } else {
        os << "uint" << bits << "_t";
      }
    }
  } else {
    CodeGenC::PrintType(t, os);
  }
}

std::string CodeGenAOCLHost::GetBufferRef(Type t, const Variable* buffer, Expr index) {
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
      std::vector<Expr> indices = ExtractIndices(index, var_shape_map_[buffer], range_);
      for (size_t i = 0; i < indices.size(); i++) {
        os << '[';
        PrintExpr(indices[i], os);
        os << ']';
      }
    }
  }  
  return os.str();
}

void CodeGenAOCLHost::VisitExpr_(const Min *op, std::ostream& os) {  // NOLINT(*)
  os << "std::min(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenAOCLHost::VisitExpr_(const Max *op, std::ostream& os) {  // NOLINT(*)
  os << "std::max(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenAOCLHost::VisitStmt_(const For* op) {
  Stmt stmt = op->body;
  while (const For* for_op = stmt.as<For>())
    stmt = for_op->body;

  if (auto s = stmt.as<StreamStmt>()) { 
    if (s->buffer_var.get()->name_hint.find("channel") 
        != std::string::npos) return;
  } else if (auto st = stmt.as<Store>()) {
    if (auto e = st->value.as<StreamExpr>()) {
      if (e->buffer_var.get()->name_hint.find("channel")
          != std::string::npos) return;

    } else { 
      auto value = st->value;
      if (auto c = value.as<Cast>()) value = c->value;
      if (auto v = value.as<IntImm>()) {
        if (v->value == 0) return;
      } else if (auto v = value.as<FloatImm>()) {
        if (v->value == 0) return;
      } else if (auto v = value.as<UIntImm>()) {
        if (v->value == 0) return;
      }
    }
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenAOCLHost::VisitStmt_(const Store* op) {
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

void CodeGenAOCLHost::GenForStmt(const For* op, std::string pragma, bool before) {
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

void CodeGenAOCLHost::VisitStmt_(const IfThenElse* op) {
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

void CodeGenAOCLHost::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
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

  this->PrintIndent();
  PrintType(op->type, stream);
  alloc_set_.insert(vid);
  stream << ' '<< vid;
  if (constant_size > 1) {// Transfer length one array to scalar
    stream << "[";
    for (size_t i = 0; i < op->extents.size(); i++) {
      PrintExpr(op->extents[i], stream);
      if (i != op->extents.size()-1) stream << "][";
    }
    stream << "]";
  }
  stream << ";\n";
  
  buf_length_map_[buffer] = constant_size;
  RegisterHandleType(op->buffer_var.get(), op->type);
  for (size_t i = 0; i < op->attrs.size(); i++) {
    this->PrintStmt(op->attrs[i]);
  }
  this->PrintStmt(op->body);
}

void CodeGenAOCLHost::VisitStmt_(const KernelStmt* op) {
  std::string name = op->name;

  // Extract annotation information 
  struct argInfo {
    std::string     name;
    DeviceType      dev_type;
    StorageType     mem_type;
    int             mem_port;
    StreamType      stream_type;
    int             channel_depth;
  };

  std::vector<argInfo> args_info;
  for (size_t i = 0; i < op->annotate_keys.size(); i++) {
    auto info = op->annotate_values[i].as<StringImm>(); CHECK(info);
    auto v = op->args[i].as<Variable>(); CHECK(v);
    auto arg_name = v->name_hint;

    std::string s = info->value;
    size_t pos = 0;
    std::string delimiter = ":";
    std::string token;
    std::vector<int> numbers;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        numbers.push_back(std::stoi(token));
        s.erase(0, pos + delimiter.length());
    }

    // Memory type, MemPort, StreamType, ChannelDepth
    numbers.push_back(std::stoi(s));
    CHECK(numbers.size() == 5);

    auto dev_type = static_cast<DeviceType>(numbers[0]);
    auto mem_dev = static_cast<StorageType>(numbers[1]);
    int mem_port = numbers[2];
    auto stream_type = static_cast<StreamType>(numbers[3]);
    int channel_depth = numbers[4];

    argInfo arg_info = {arg_name, dev_type, mem_dev, 
                        mem_port, stream_type, channel_depth};
    args_info.push_back(arg_info);
  }

  if (args_info.size() > 0) {
    // Create kernels
    stream << "\n";
    PrintIndent();

    stream << "cl_kernel kernel = clCreateKernel(program, \""
           << name << "\", &status);\n";

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
      kernel_args.push_back(arg_name);
 
      PrintIndent();
      stream << "cl_mem buffer_" 
             << arg_name
             << " = clCreateBuffer(context, " 
             << "CL_MEM_READ_WRITE, "
             << "sizeof(";
      PrintType(handle_data_type_[v], stream);
      stream << ")*";
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) stream << "*";
        stream << shape[i];
      }

      stream << ", NULL, &status); CHECK(status);\n";
    }

    stream << "\n  // Write buffers to device\n";
    for (size_t k = 0; k < op->args.size(); k++) {
      auto v = op->args[k].as<Variable>();
      CHECK(v) << "invalid input var";
      auto shape = var_shape_map_[v];
      PrintIndent();
      stream << "status = clEnqueueWriteBuffer(" 
             << "cmdQueue, buffer_" << kernel_args[k]
             << ", CL_TRUE, 0, sizeof(";
      PrintType(handle_data_type_[v], stream);
      stream << ")*";
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) stream << "*";
        stream << shape[i];
      }
      stream << ", " << kernel_args[k]
             << ", 0, NULL, NULL); CHECK(status);\n";
    }

    // set kernel arguments
    stream << "\n  // set device kernel buffer\n";
    CHECK(op->args.size() == kernel_args.size());
    for (size_t k = 0; k < kernel_args.size(); k++) {
      PrintIndent();
      stream << "status = clSetKernelArg(kernel, " << k << ", "
             << "sizeof(cl_mem), (void*)&buffer_" 
             << kernel_args[k] << "); CHECK(status);\n";
    }

    
    PrintIndent();
    stream << "status = clEnqueueNDRangeKernel(" 
           << "cmdQueue, kernel, 1, NULL, globalWorkSize, "
           << "localWorkSize, 0, NULL, &kernel_exec_event); CHECK(status);\n";

    // launch kernel execution  
    stream << "\n  // enqueue kernel function\n";
    PrintIndent();
    stream << "status = clFlush(cmdQueue); CHECK(status);\n";
    PrintIndent();
    stream << "status = clFinish(cmdQueue); CHECK(status);;\n";

    // retrieve data from global buffer 
    for (size_t k = 0; k < kernel_args.size(); k++) {
      auto v = op->args[k].as<Variable>();
      auto shape = var_shape_map_[v];
      PrintIndent();
      stream << "clEnqueueReadBuffer("
             << "cmdQueue, buffer_" << kernel_args[k]
             << ", CL_TRUE, 0, sizeof(";
      PrintType(handle_data_type_[v], stream);
      stream << ")*";
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) stream << "*";
        stream << shape[i];
      }
      stream << ", " << kernel_args[k] 
             << ", 0, NULL, NULL);\n";
    }

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

void CodeGenAOCLHost::VisitStmt_(const ExternModule* op) {
  this->PrintStmt(op->body);
}

}  // namespace codegen
}  // namespace TVM
