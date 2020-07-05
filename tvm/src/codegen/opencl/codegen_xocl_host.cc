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

struct ArgInfo {
    StorageType storage_type;
    uint32_t mem_channel;
    StreamType stream_type;
    DeviceType target_device;
    Array<Expr> tensor_shape;
    Type data_type;
};

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

    // ptr mode: check name availability
    if (alloc_set_.find(vid) != alloc_set_.end()) {
      not_alloc = true;
    } else {
      for (auto& name : arg_names) {
        if (name == vid) not_alloc = true;
      }
    }
  }

  // not allocate for moved data  
  if (!not_alloc) { 
    this->PrintIndent();
    PrintType(op->type, stream);
    alloc_set_.insert(vid);
    stream << ' '<< vid;
    if (constant_size > 1) {// Transfer length one array to scalar
      stream << "[";
      for (size_t i = 0; i < op->extents.size(); i++) {
        PrintExpr(op->extents[i], stream);
        if (i != op->extents.size()-1) {
            stream << "][";
        }
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
  std::unordered_map<int, std::vector<int>> mem_mapping;
  CHECK(op->annotate_values.size() == 5 * op->args.size());
  for (size_t i = 0; i < op->args.size(); i++) {
    int pos  = op->annotate_values[5*i+0].as<IntImm>()->value;
    int mem  = op->annotate_values[5*i+1].as<IntImm>()->value;
    int port = op->annotate_values[5*i+2].as<IntImm>()->value;
    int type = op->annotate_values[5*i+3].as<IntImm>()->value;
    int direction = op->annotate_values[5*i+4].as<IntImm>()->value;
    mem_mapping[pos] = {mem, port, type, direction};
  }

  // initialize buffers and opencl kernel 
  if (name.find("test") != std::string::npos) {

    // create kernels
    stream << "\n";
    PrintIndent();

    stream << "cl::Kernel kernel(program, \""
           << name << "\", &err);\n";

    // create device buffers
    std::vector<std::string> kernel_args;
    std::unordered_map<std::string, ArgInfo> arg_map;
    int stream_arg_num = 0;

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
 
      // check buffer types 
      CHECK(mem_mapping.count(k)) << k << "-th arg not found in " << op->args;
      CHECK(mem_mapping.at(k).size() == 4);
      auto type = static_cast<StorageType>(mem_mapping[k][0]);
      unsigned int port = mem_mapping[k][1];
      auto stream_type = static_cast<StreamType>(mem_mapping[k][2]);
      auto direction = static_cast<DeviceType>(mem_mapping[k][3]);
      auto dtype = handle_data_type_[v];
      arg_map[arg_name] = {type, port, stream_type, direction, shape, dtype};

      // TODO: check xrt stream with other storage media 
      if (type == StorageType::devDRAM) {
        switch (stream_type) {
          case StreamType::DMA: {
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
            break;
          }
          case StreamType::FIFO: {
            stream_arg_num += 1;
            if (decl_stream.str().find("cl_ext_xilinx.h") == std::string::npos) {
              decl_stream << "#include <thread>\n";
              decl_stream << "#include <CL/cl_ext_xilinx.h>\n";
              decl_stream << R"(
// Declaration of custom stream APIs that binds to Xilinx Streaming APIs.
decltype(&clCreateStream) xcl::Stream::createStream = nullptr;
decltype(&clReleaseStream) xcl::Stream::releaseStream = nullptr;
decltype(&clReadStream) xcl::Stream::readStream = nullptr;
decltype(&clWriteStream) xcl::Stream::writeStream = nullptr;
decltype(&clPollStreams) xcl::Stream::pollStreams = nullptr;
)";

              stream << "  " << "cl_platform_id platform_id = ";
              stream << "device.getInfo<CL_DEVICE_PLATFORM>(&err);\n";
              stream << "  " << "xcl::Stream::init(platform_id);\n\n";

              // create external mem pointer
              std::string name = "ext";
              stream << "  " << "cl_mem_ext_ptr_t " << name << ";\n";
              stream << "  " << name << ".param = kernel.get();\n";
              stream << "  " << name << ".obj = NULL;\n\n";
            }
            stream << "  " << "ext.flags = " << k << ";\n";
            // create xcl stream
            std::string mode = "CL_STREAM_READ_ONLY";
            if (direction == DeviceType::devHost)
                mode = "CL_STREAM_WRITE_ONLY";
            stream << "  " << "cl_stream StreamExt_" + arg_name << " = "
                   << "xcl::Stream::createStream(device.get(), " << mode << ", "
                   << "CL_STREAM, &ext, &err);\n";
            break;

          }
        }

      // high bandwidth memory 
      } else if (type == StorageType::devHBM) {
        if (decl_stream.str().find("HBM") == std::string::npos) {
          decl_stream << R"(
#define MAX_HBM_BANKCOUNT 32
#define BANK(n) n | XCL_MEM_TOPOLOGY
const int bank[MAX_HBM_BANKCOUNT] = {
    BANK(0),  BANK(1),  BANK(2),  BANK(3),  BANK(4),
    BANK(5),  BANK(6),  BANK(7),  BANK(8),  BANK(9),
    BANK(10), BANK(11), BANK(12), BANK(13), BANK(14),
    BANK(15), BANK(16), BANK(17), BANK(18), BANK(19),
    BANK(20), BANK(21), BANK(22), BANK(23), BANK(24),
    BANK(25), BANK(26), BANK(27), BANK(28), BANK(29),
    BANK(30), BANK(31)
};
)";
          // create tcl script 
          cfg_stream << "[connectivity]\n";
        }
        auto name = "BufExt_" + arg_name; 
        // create external mem pointer
        stream << "  " << "cl_mem_ext_ptr_t " << name << ";\n";
        stream << "  " << name << ".flags = bank[" << port << "];\n"; 
        stream << "  " << name << ".param = 0;\n"; 
        stream << "  " << name << ".obj = &" << arg_name << "[0];\n"; 
        PrintIndent();
        stream << "cl::Buffer buffer_" 
               << arg_name
               << "(context, " 
               << "CL_MEM_EXT_PTR_XILINX | "
               << "CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, "
               << "sizeof(";
        PrintType(handle_data_type_[v], stream);
        stream << ")*";
        for (size_t i = 0; i < shape.size(); i++) {
          if (i != 0) stream << "*";
          stream << shape[i];
        }
        stream << ", &" << name << ", &err);\n\n";
        // assign memory channel ports
        cfg_stream << "sp=" << op->name << "_1."
                   << arg_name << ":HBM[" << port << "]\n";
      }
    }

    // set kernel arguments
    stream << "\n  // set device kernel buffer\n";
    CHECK(op->args.size() == kernel_args.size());
    for (size_t k = 0; k < kernel_args.size(); k++) {
      auto arg_name = kernel_args[k];
      CHECK(arg_map.count(arg_name));
      if (arg_map[arg_name].stream_type == StreamType::DMA) {
        PrintIndent();
        stream << "err = kernel.setArg(" << k << ", "
               << "buffer_" << kernel_args[k] << ");\n";
      }
    }

    // migrate memory objects
    bool first = true;
    PrintIndent();
    stream << "err = q.enqueueMigrateMemObjects({";
    for (size_t k = 0; k < kernel_args.size(); k++) {
      auto arg_name = kernel_args[k];
      CHECK(arg_map.count(arg_name));
      if (arg_map[arg_name].stream_type == StreamType::DMA) {
        if (!first) stream << ", ";
        stream << "buffer_" << kernel_args[k];
        first = false;
      }
    }
    stream << "}, 0/*from host*/);\n";
    stream << "  q.finish();\n";

    // set up timer and start execution 
    stream << "\n  // enqueue kernel function\n";
    stream << "  std::chrono::duration<double> kernel_time(0);\n"; 
    stream << "  auto kernel_start = std::chrono::high_resolution_clock::now();\n";
    stream << "  cl::Event event;\n";
    stream << "  err = q.enqueueTask(kernel, NULL, &event);\n\n";

    // initialize write and read stream
    if (stream_arg_num > 0) {
      for (size_t k = 0; k < kernel_args.size(); k++) {
        auto arg_name = kernel_args[k];
        CHECK(arg_map.count(arg_name));
        auto arg_info = arg_map.at(arg_name);
        auto direction = arg_info.target_device; 
        if (arg_info.stream_type == StreamType::DMA) continue;

        // xcl read stream 
        // TODO: add non-blocking stream
        if (direction == DeviceType::devHost) {
          stream << "  " << "cl_stream_xfer_req rd_req_" << arg_name << "{0};\n";
          stream << "  " << "rd_req_" << arg_name << ".flags = CL_STREAM_EOT;\n";
          stream << "  " << "rd_req_" << arg_name << ".priv_data = "
                 << "(void*)\"read_" << arg_name << "\";\n";
          stream << "  " << "std::thread thrd_" << arg_name << "("
                 << "xcl::Stream::readStream, StreamExt_" << arg_name << ", &"
                 << arg_name << "[0], sizeof(";
          PrintType(arg_info.data_type, stream);
          stream << ")";
          for (auto v : arg_info.tensor_shape)
             stream << "*" << v;  
          stream << ", &rd_req_" << arg_name << ", &err);\n\n";

        } else {
          stream << "  " << "cl_stream_xfer_req wr_req_" << arg_name << "{0};\n";
          stream << "  " << "wr_req_" << arg_name << ".flags = CL_STREAM_EOT;\n";
          stream << "  " << "wr_req_" << arg_name << ".priv_data = "
                 << "(void*)\"write_" << arg_name << "\";\n";
          stream << "  " << "std::thread thrd_" << arg_name << "("
                 << "xcl::Stream::writeStream, StreamExt_" << arg_name << ", &"
                 << arg_name << "[0], sizeof(";
          PrintType(arg_info.data_type, stream);
          stream << ")";
          for (auto v : arg_info.tensor_shape)
             stream << "*" << v;  
          stream << ", &wr_req_" << arg_name << ", &err);\n\n";
        }
      }
      // waiting for threads to join
      for (size_t k = 0; k < kernel_args.size(); k++) {
        auto arg_info = arg_map.at(kernel_args[k]);
        if (arg_info.stream_type == StreamType::DMA) continue;
        stream << "  " << "thrd_" << kernel_args[k] << ".join();\n";
      }
      stream << "\n";
    }

    stream << "  err = q.finish();\n";
    stream << "  auto kernel_end = std::chrono::high_resolution_clock::now();\n";
    stream << "  kernel_time = std::chrono::duration<double>"
           << "(kernel_end - kernel_start);\n";
    stream << "  auto kernel_time_in_sec = kernel_time.count();\n";
    stream << "  std::cout << \"Execution Time:\" <<  kernel_time_in_sec;\n";

    // copy data back to host  
    if (stream_arg_num < (signed)kernel_args.size()) {
      bool first = true;
      PrintIndent();
      stream << "err = q.enqueueMigrateMemObjects({";
      for (size_t k = 0; k < kernel_args.size(); k++) {
        auto arg_name = kernel_args[k];
        CHECK(arg_map.count(arg_name));
        auto arg_info = arg_map.at(arg_name);
        if (arg_info.stream_type != StreamType::DMA)
          continue;
        if (!first) stream << ", ";
        stream << "buffer_" << kernel_args[k];
        first = false;
      }
      stream << "}, CL_MIGRATE_MEM_OBJECT_HOST);\n";
    }

    // realease xcl stream
    if (stream_arg_num > 0) {
      for (size_t k = 0; k < kernel_args.size(); k++) {
        auto arg_info = arg_map.at(kernel_args[k]);
        if (arg_info.stream_type == StreamType::DMA) continue;
        stream << "  " << "xcl::Stream::releaseStream("
               << "StreamExt_" << kernel_args[k] << ");\n";
      }
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

void CodeGenXOCLHost::VisitStmt_(const ExternModule* op) {
  std::string name;
  for (size_t i = 0; i < op->annotate_keys.size(); i++) {
    auto key = op->annotate_keys[i].as<StringImm>()->value;
    auto value = op->annotate_values[i].as<StringImm>()->value;
    if (key == "name") { 
      name = value;
    }
  }
  this->PrintStmt(op->body);
}

}  // namespace codegen
}  // namespace TVM
