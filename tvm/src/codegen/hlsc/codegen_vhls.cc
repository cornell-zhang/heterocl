/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.cc
 */
#include "codegen_vhls.h"
#include <sys/types.h>
#include <sys/wait.h>
#include <tvm/build_module.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>
#include <fstream>
#include <regex>
#include <string>
#include <vector>
#include "../../pass/stencil.h"
#include "../build_common.h"
#include "../build_soda.h"
#include "../codegen_soda.h"

namespace TVM {
namespace codegen {

struct argInfo {
  std::string name;
  StorageType mem_type;
  int mem_port;
  StreamType stream_type;
  int channel_depth;
  bool is_written;
};

void CodeGenVivadoHLS::AddFunction(
    LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type) {
  // write header files
  this->decl_stream << "#include <ap_int.h>\n";
  this->decl_stream << "#include <ap_fixed.h>\n";
  this->decl_stream << "#include <ap_axi_sdata.h>\n";
  this->decl_stream << "#include <hls_stream.h>\n";
  this->decl_stream << "#include <hls_math.h>\n";
  this->decl_stream << "#include <math.h>\n";
  this->decl_stream << "#include <stdint.h>\n";

  // setup codegen mode
  if (map_arg_type.count("sdsoc")) {
    sdsoc_mode = true;
    this->decl_stream << "#include \"sds_lib.h\"\n\n";
  } else if (map_arg_type.count("sdaccel")) {
    extern_c_wrapper = true;
    this->decl_stream << "\n";
  }

  // clear previous generated state.
  this->InitFuncState(f);
  map_arg_type_ = map_arg_type;
  // add to alloc buffer type.
  for (const auto& kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  HCL_DEBUG_LEVEL(2) << "Adding VHLS function...";
  bool has_const = PrintConstants(f->body, true);
  if (has_const) stream << "#include \"global_consts.h\"\n";

  // generate top function signature
  this->stream << "void " << f->name << "(";
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    // check type in the arg map
    if (map_arg_type.find(vid) == map_arg_type.end()) {
      LOG(WARNING) << vid << " type not found\n";
      PrintType(v.type(), this->stream);
      this->stream << ' ' << vid;
    } else {
      // Note: this `map_arg_type` map is used to map name-erased
      // variables to the named variables and their types. For example
      // the original input Halide IR qill have `Let A = arg1` to assign
      // the name-erased variable (e.g. arg1) to the varaible you defined (e.g.
      // A) we just use this map to query the name and data type from the key
      // (i.e. arg1)
      auto arg = map_arg_type[vid];
      PrintType(std::get<1>(arg), this->stream);
      // this->stream << "* " << std::get<0>(arg);
      const BufferNode* buf = f->api_args[i].as<BufferNode>();
      if (v.type().is_handle() && buf) {
        var_shape_map_[buf->data.get()] = buf->shape;
        auto it = alloc_storage_scope_.find(v.get());
        if (it != alloc_storage_scope_.end()) {
          PrintStorageScope(it->second, stream);
        }
        this->stream << " " << std::get<0>(arg);

        // print multi-dim array
        this->stream << "[";
        int count = 0;
        for (auto& s : buf->shape) {
          if (count != 0) this->stream << "][";
          this->stream << s;
          count = count + 1;
        }
        this->stream << "]";
      } else {
        this->stream << " " << std::get<0>(arg);
      }
    }
  }

  stream << ") {\n";
  int func_scope = this->BeginScope();
  range_ = CollectIterRange(f->body);
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";

  // close soda header handle
  if (soda_header_.is_open()) soda_header_.close();
}

// print data type
void CodeGenVivadoHLS::PrintType(Type t, std::ostream& os) {
  if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed()) {
    if (t.is_uint()) {
      if (!enable_native_dtype) {
        if (t.bits() == 32) {
          os << "unsigned int";
        } else {
          os << "ap_uint<" << t.bits() << ">";
        }
      } else {
        if (t.bits() == 8 || t.bits() == 16 || t.bits() == 32 ||
            t.bits() == 64) {
          os << "unsigned int";
        }
      }
    } else if (t.is_int()) {
      if (!enable_native_dtype) {
        if (t.bits() == 32) {
          os << "int";
        } else {
          os << "ap_int<" << t.bits() << ">";
        }
      } else {
        if (t.bits() == 8 || t.bits() == 16 || t.bits() == 32 ||
            t.bits() == 64) {
          os << "int";
        }
      }
    } else if (t.is_ufixed()) {
      os << "ap_ufixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
    } else {
      os << "ap_fixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
    }
  } else {
    CodeGenC::PrintType(t, os);
  }
}

void CodeGenVivadoHLS::VisitExpr_(const Min* op,
                                  std::ostream& os) {  // NOLINT(*)
  os << "hls::min(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenVivadoHLS::VisitExpr_(const Max* op,
                                  std::ostream& os) {  // NOLINT(*)
  os << "hls::max(";
  PrintExpr(op->a, os);
  os << ", ";
  PrintExpr(op->b, os);
  os << ")";
}

void CodeGenVivadoHLS::VisitExpr_(const GetBit* op, std::ostream& os) {
  PrintExpr(op->a, os);
  os << "[";
  PrintExpr(op->index, os);
  os << "]";
}

void CodeGenVivadoHLS::VisitExpr_(const GetSlice* op, std::ostream& os) {
  PrintExpr(op->a, os);
  os << "(";
  Expr new_index_left = ir::Simplify(op->index_left - 1);
  PrintExpr(new_index_left, os);
  os << ", ";
  PrintExpr(op->index_right, os);
  os << ")";
}

void CodeGenVivadoHLS::VisitExpr_(const Load* op, std::ostream& os) {
  std::string vid = GetVarID(op->buffer_var.get());
  // TODO(Hecmay): find a betetr way to track streaming channels
  if (stream_vars.find(vid) != stream_vars.end()) {
    PrintIndent();
    stream << vid << "_temp = " << vid << ".read();\n";
    os << vid << "_temp.get_data()";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenVivadoHLS::VisitStmt_(const Store* op) {
  std::string vid = GetVarID(op->buffer_var.get());
  if (stream_vars.find(vid) != stream_vars.end()) {
    PrintIndent();
    auto bits = handle_data_type_[op->buffer_var.get()].bits();
    stream << "pkt_b" << bits << " " << vid << "_temp;\n";
    PrintIndent();
    stream << vid << "_temp.set_data(" << PrintExpr(op->value) << ");\n";
    PrintIndent();
    stream << vid << "_temp.set_keep(-1);\n";
    PrintIndent();
    stream << vid << ".write(" << vid << "_temp);\n";
    return;
  }

  // handle SetSlice. For example, if A is a fixed-point variable
  // we used this IR to set certain bits of A: A[3:0] = 0b101
  if (const SetSlice* ss = op->value.as<SetSlice>()) {
    Type t = op->value.type();
    Expr new_index_left = ir::Simplify(ss->index_left - 1);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    std::string rhs = PrintExpr(ss->value);
    PrintIndent();
    this->stream << ref << "(" << PrintExpr(new_index_left) << ", "
                 << PrintExpr(ss->index_right) << ") = " << rhs << ";\n";
  } else if (const SetBit* sb = op->value.as<SetBit>()) {
    Type t = op->value.type();
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    PrintIndent();
    this->stream << ref << "[" << PrintExpr(sb->index)
                 << "] = " << PrintExpr(sb->value) << ";\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

// Create expression of function call. Example ret = func_call(arg1, arg2)
void CodeGenVivadoHLS::VisitExpr_(const Call* op,
                                  std::ostream& os) {  // NOLINT(*)
  if ((op->call_type == Call::Intrinsic ||
       op->call_type == Call::PureIntrinsic) &&
      op->name == "sqrt") {
    os << "sqrt(";
    for (size_t i = 0; i < op->args.size(); i++) {
      this->PrintExpr(op->args[i], os);
      if (i < op->args.size() - 1) {
        os << ", ";
      }
    }
    os << ")";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

// Allocate a buffer. Same as declaration in C/C++
void CodeGenVivadoHLS::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  if (op->new_expr.defined()) {
    CHECK_EQ(op->free_function, "nop");
    std::string new_data = PrintExpr(op->new_expr);
    this->PrintIndent();
    PrintType(op->type, stream);
    stream << "* " << vid << '=' << new_data << ";\n";
  } else {
    int32_t constant_size = op->constant_allocation_size();
    CHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now";
    const Variable* buffer = op->buffer_var.as<Variable>();
    var_shape_map_[buffer] = op->extents;

    std::string scope;  // Allocate on local scope by default
    auto it = alloc_storage_scope_.find(buffer);
    if (it != alloc_storage_scope_.end())
      scope = alloc_storage_scope_.at(buffer);
    else
      scope = "local";

    // FIFO Checking
    bool is_fifo = false;
    for (auto attr : op->attrs) {
      if (attr.as<StreamStmt>()) {
        is_fifo = true;
        break;
      }
    }
    // Auto-apply dataflow
    if (is_fifo) {
      if (stream.str().find("#pragma HLS dataflow") == std::string::npos) {
        LOG(INFO) << "Auto-applying dataflow optimization...";
        PrintIndent();
        stream << "#pragma HLS dataflow\n";
      }
    }

    this->PrintIndent();
    if (constant_size > 1) {  // Transform length one array to scalar
      if (sdsoc_mode) {
        // Allocate continuous physical mem
        PrintType(op->type, stream);
        stream << "* " << vid << " = (";
        PrintType(op->type, stream);
        stream << " *)sds_alloc(sizeof(";
        PrintType(op->type, stream);
        stream << ")";

        for (auto& v : op->extents) {
          stream << "*" << v;
        }
        stream << ")";

      } else {
        if (is_fifo) {
          stream << "hls::stream<";
          PrintType(op->type, stream);
          stream << " > " << vid;

        } else {
          PrintType(op->type, stream);
          stream << ' ' << vid;
          for (size_t i = 0; i < op->extents.size(); i++) {
            stream << '[';
            PrintExpr(op->extents[i], stream);
            stream << "]";
          }
        }
      }

    } else {
      PrintType(op->type, stream);
      stream << ' ' << vid;
    }
    buf_length_map_[buffer] = constant_size;

    stream << ";\n";
    for (size_t i = 0; i < op->attrs.size(); i++) this->PrintStmt(op->attrs[i]);
    buf_length_map_[buffer] = constant_size;
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

// Create a for loop
void CodeGenVivadoHLS::VisitStmt_(const For* op) {
  std::ostringstream os;

  Stmt stmt = op->body;
  while (const For* for_op = stmt.as<For>()) stmt = for_op->body;

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
    os << "#pragma HLS unroll";
    if (unroll_factor > 0)
      os << " factor=" << unroll_factor << "\n";
    else
      os << "\n";
  } else if (op->for_type == ForType::Pipelined) {
    int II = 0, i = 0;
    for (auto key : op->annotate_keys) {
      if (auto str = key.as<StringImm>()) {
        auto initiation_interval = op->annotate_values[i].as<IntImm>();
        if (str->value == "initiation_interval" &&
            initiation_interval != nullptr && initiation_interval->value > 1) {
          II = initiation_interval->value;
          break;
        }
      }
      i++;
    }
    os << "#pragma HLS pipeline";
    if (II > 0)
      os << " II=" << II << "\n";
    else
      os << "\n";
  }
  GenForStmt(op, os.str(), false);
}

// print partition pragma
void CodeGenVivadoHLS::VisitStmt_(const Partition* op) {
  PrintIndent();
  stream << "#pragma HLS array_partition variable=";
  std::string vid = GetVarID(op->buffer_var.get());
  stream << vid << " ";
  switch (op->partition_type) {
    case PartitionType::Complete:
      stream << "complete";
      break;
    case PartitionType::Block:
      stream << "block";
      break;
    case PartitionType::Cyclic:
      stream << "cyclic";
      break;
  }
  stream << " dim=" << op->dim;
  if (op->partition_type != PartitionType::Complete) {
    stream << " factor=" << op->factor;
  }
  stream << "\n";
}

// Stream reading channel
void CodeGenVivadoHLS::VisitExpr_(const StreamExpr* op, std::ostream& os) {
  std::string vid = GetVarID(op->buffer_var.get());
  os << vid << ".read()";
}

void CodeGenVivadoHLS::VisitStmt_(const StreamStmt* op) {
  std::string vid = GetVarID(op->buffer_var.get());
  PrintIndent();
  if (op->stream_type == StreamType::ATTR) {
    stream << "#pragma HLS stream variable=" << vid << " depth=" << op->depth
           << "\n";
  } else {
    stream << vid << ".write(" << PrintExpr(op->value) << ");\n";
  }
}

class AllocateCollector final : public IRVisitor {
 public:
  AllocateCollector(std::vector<const Allocate*>& alloc_list,
                    VarExprUnorderedSet& outputs)
      : alloc_list_(alloc_list), outputs_(outputs) {}

  void Visit_(const Allocate* op) {
    if (outputs_.count(op->buffer_var)) alloc_list_.push_back(op);
    this->Visit(op->body);
  }

 private:
  std::vector<const Allocate*>& alloc_list_;
  VarExprUnorderedSet& outputs_;
};

void CodeGenVivadoHLS::VisitStmt_(const KernelStmt* op) {
  PrintIndent();
  stream << op->name << "(";

  // Extract annotation values
  std::vector<argInfo> args_info;
  for (size_t k = 0; k < op->annotate_keys.size(); k++) {
    auto key = op->annotate_values[k].as<StringImm>();
    CHECK(key);
  }
  // Print kernel function arguments
  for (size_t i = 0; i < op->args.size(); i++) {
    std::string arg_name = PrintExpr(op->args[i]);
    stream << arg_name;
    if (i < op->args.size() - 1) stream << ", ";
  }
  stream << ");\n";
}

void CodeGenVivadoHLS::VisitStmt_(const AttrStmt* op) {
  if (op->attr_key == "dataflow") {
    PrintIndent();
    stream << "#pragma HLS dataflow\n";
    PrintStmt(op->body);
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenVivadoHLS::VisitStmt_(const KernelDef* op) {
  LoweredFunc f;
  // save func states
  CodeGenC::SaveFuncState(f);
  CodeGenC::InitFuncState(f);
  std::ostringstream save;
  std::ostringstream pragma;
  save << this->stream.str();
  this->stream.str("");
  this->stream.clear();

  // skip the first underscore
  GetUniqueName("_");
  // add to alloc buffer : type.
  for (const auto& k : op->args) {
    RegisterHandleType(k.get(), k.get()->type);
  }

  // collect argument information
  std::vector<argInfo> args_info;
  bool is_kernel_func = false;
  for (size_t i = 0; i < op->attributes.size(); i++) {
    auto info = op->attributes[i];
    CHECK_GE(info.size(), 2);
    auto arg_name = info[0].as<StringImm>()->value;
    for (size_t i = 0; i < arg_name.size(); ++i) {
      if (arg_name[i] == '.') arg_name[i] = '_';
    }

    if (info.size() > 2) {
      is_kernel_func = true;
      CHECK_EQ(info.size(), 6);
      auto mem_dev = static_cast<StorageType>(info[1].as<IntImm>()->value);
      int mem_port = info[2].as<IntImm>()->value;
      auto stream_type = static_cast<StreamType>(info[3].as<IntImm>()->value);
      int channel_depth = info[4].as<IntImm>()->value;
      bool is_written = info[5].as<IntImm>()->value == 1 ? true : false;
      argInfo arg_info = {arg_name,    mem_dev,       mem_port,
                          stream_type, channel_depth, is_written};
      args_info.push_back(arg_info);

      // For regular HCL module function
      // only IO direction information is injected
    } else {
      bool is_written = info[1].as<IntImm>()->value == 1 ? true : false;
      argInfo arg_info;
      arg_info.is_written = is_written;
      args_info.push_back(arg_info);
    }
  }

  // Lambda function to calculate buffer size
  auto const_size = [&](Array<Expr> shape) -> int32_t {
    int32_t res = 1;
    for (auto s : shape) {
      CHECK(s.as<IntImm>());
      auto v = s.as<IntImm>()->value;
      res = res * v;
    }
    return res;
  };

  // print top-level kernel function
  if (is_kernel_func) {
    int extern_scope = -1;
    if (extern_c_wrapper) {
      extern_scope = BeginScope();
      stream << "extern \"C\" {\n";
    }

    stream << "void " << op->name << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      VarExpr v = op->args[i];
      var_shape_map_[v.get()] = op->arg_shapes[i];
      int32_t constant_size = const_size(op->arg_shapes[i]);
      CHECK_GT(constant_size, 0) << "Input arg size must be greater than 0...";
      buf_length_map_[v.get()] = constant_size;
      std::string vid = AllocVarID(v.get());

      if (i != 0) stream << ", ";
      std::string str = PrintExpr(op->arg_types[i]);
      Type type = String2Type(str);

      // pass-by-value arguments
      if (var_shape_map_[v.get()].size() == 1 &&
          var_shape_map_[v.get()][0].as<IntImm>()->value == 1) {
        PrintType(type, stream);
        this->stream << " " << vid;

        // pass-by-pointer arguments
      } else {
        CHECK(args_info.size() > i) << i << ":" << args_info.size();
        auto info = args_info[i];

        if (info.stream_type == StreamType::FIFO) {
          auto bits = type.bits();
          if (decl_stream.str().find("typedef qdma_axis<" +
                                     std::to_string(bits)) ==
              std::string::npos) {
            decl_stream << "typedef qdma_axis<" << bits << ", 0, 0, 0> pkt_b"
                        << bits << ";\n";
          }
          stream << "hls::stream<pkt_b" << bits << "> &" << vid;

          // Memory-mapped pointers
        } else {
          PrintType(type, stream);
          auto size = var_shape_map_[v.get()];
          stream << " " << vid;
          for (auto& s : size) {
            stream << "[" << s << "]";
          }
        }
      }
    }
    stream << ") {\n";

    if (extern_c_wrapper) {
      // Port-level protocol interface
      CHECK(op->args.size() == op->args.size());
      for (size_t i = 0; i < op->args.size(); i++) {
        if (op->arg_shapes[i].size() == 1 &&
            op->arg_shapes[i][0].as<IntImm>()->value == 1) {
          continue;
        } else {
          PrintIndent();
          auto info = args_info[i];

          if (info.stream_type == StreamType::FIFO) {
            stream << "#pragma HLS INTERFACE axis port=" << info.name << "\n";
          } else {
            stream << "#pragma HLS INTERFACE m_axi port=" << info.name << " "
                   << "offset=slave bundle=gmem" << i << "\n";
          }
        }
      }

      // Block-level control interface
      for (size_t i = 0; i < op->args.size(); i++) {
        auto info = args_info[i];
        if (info.stream_type == StreamType::FIFO) continue;
        PrintIndent();
        stream << "#pragma HLS INTERFACE s_axilite port=" << info.name << " "
               << "bundle=control\n";
      }
      PrintIndent();
      stream << "#pragma HLS INTERFACE s_axilite"
             << " port=return bundle=control\n";
    }

    // function body
    int func_scope = BeginScope();
    range_ = CollectIterRange(op->body);
    PrintStmt(op->body);

    EndScope(func_scope);
    PrintIndent();
    stream << "}\n";

    if (extern_c_wrapper) {
      stream << "}\n\n";
      EndScope(extern_scope);
    }

    // Non-top kernel function
  } else {
    std::ostringstream func_os;
    func_os << "static void " << op->name << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      VarExpr v = op->args[i];
      var_shape_map_[v.get()] = op->arg_shapes[i];

      int32_t constant_size = const_size(op->arg_shapes[i]);
      CHECK_GT(constant_size, 0) << "Input arg size must be greater than 0...";
      buf_length_map_[v.get()] = constant_size;
      std::string vid = AllocVarID(v.get());
      if (i != 0) func_os << ", ";
      std::string str = PrintExpr(op->arg_types[i]);
      Type type = String2Type(str);

      // Scalar input
      CHECK_GT(op->arg_shapes[i].size(), 0);
      if (op->arg_shapes[i].size() == 1) {
        auto dim = op->arg_shapes[i][0].as<IntImm>();
        CHECK(dim);
        if (dim->value == 1 || dim->value == 0) {
          PrintType(type, func_os);
          auto info = args_info[i];
          if (info.is_written) func_os << "&";
          func_os << " " << vid;
          continue;
        }
      }

      if (op->arg_shapes[i].size() > 0) {
        auto shape = op->arg_shapes[i];
        PrintType(type, func_os);
        func_os << " " << vid;
        func_os << "[";
        for (size_t k = 0; k < shape.size(); k++) {
          if (k != shape.size() - 1) func_os << "][";
          func_os << shape[k];
        }
        func_os << "]";
      }
    }
    decl_stream << func_os.str() << ");\n";
    stream << func_os.str() << ") {\n";

    PrintIndent();
    stream << "#pragma HLS inline off\n";

    // function body
    int func_scope = BeginScope();
    range_ = CollectIterRange(op->body);
    PrintStmt(op->body);
    EndScope(func_scope);
    PrintIndent();
    stream << "}\n\n";
  }

  // restore default stream
  module_stream << this->stream.str();
  this->stream.str("");
  this->stream.clear();
  this->stream << save.str();
  RestoreFuncState(f);
}

void CodeGenVivadoHLS::VisitStmt_(const Stencil* op) {
  // Use SODA codegen for stencil analysis
  CodeGenSODA cg_soda;
  cg_soda.Init(false);
  VarExprUnorderedSet inputs;
  VarExprUnorderedSet outputs;
  for (size_t i = 0; i < op->inputs.size(); i++) inputs.insert(op->inputs[i]);
  for (size_t i = 0; i < op->outputs.size(); i++) {
    outputs.insert(op->outputs[i]);
  }
  std::vector<const Allocate*> alloc_list;
  AllocateCollector collector(alloc_list, outputs);
  collector.Visit(op->body);
  std::string kernel_name;
  cg_soda.PrintSODA(op, &kernel_name);
  std::string code = cg_soda.Finish();

  // Generate SODA HLSC code
  SODA2HLSC(code);

  PrintIndent();
  // Create a new file for the stencil function if not exists
  if (!soda_header_.is_open()) {
    soda_header_.open("soda_stencil.h");
    stream << "#include \"soda_stencil.h\"\n";
  }
  // Allocate output tensors if needed
  for (size_t i = 0; i < alloc_list.size(); i++) {
    auto alloc = alloc_list[i];
    PrintIndent();
    PrintType(alloc->type, stream);
    std::string vid = AllocVarID(alloc->buffer_var.get());
    stream << ' ' << vid;
    const Variable* buffer = alloc->buffer_var.as<Variable>();
    var_shape_map_[buffer] = alloc->extents;
    for (size_t j = 0; j < alloc->extents.size(); j++) {
      stream << '[';
      PrintExpr(alloc->extents[j], stream);
      stream << ']';
    }
    stream << ";\n";
  }
  // Print the function call to SODA function
  PrintIndent();
  soda_header_ << "void " + kernel_name + "(";
  stream << kernel_name + "(";
  for (size_t i = 0; i < op->inputs.size(); i++) {
    PrintType(cg_soda.var_type_map_[op->inputs[i].get()], soda_header_);
    soda_header_ << "* ";
    PrintExpr(op->inputs[i], soda_header_);
    PrintExpr(op->inputs[i], stream);
    soda_header_ << ", ";
    stream << ", ";
  }
  for (size_t i = 0; i < op->outputs.size(); i++) {
    PrintType(cg_soda.var_type_map_[op->outputs[i].get()], soda_header_);
    soda_header_ << "* ";
    PrintExpr(op->outputs[i], soda_header_);
    PrintExpr(op->outputs[i], stream);
    if (i < op->outputs.size() - 1) {
      soda_header_ << ", ";
      stream << ", ";
    }
  }
  soda_header_ << ");\n";
  stream << ");\n";

  // Generate SODA HLSC code
  std::ofstream soda_file;
  soda_file.open(kernel_name + ".cpp");
  soda_file << "#include \"soda_stencil.h\"\n";
  soda_file << code;
  soda_file.close();
}

void CodeGenVivadoHLS::VisitStmt_(const ExternModule* op) {
  PrintIndent();
  if (const auto* f = runtime::Registry::Get("process_extern_module")) {
    // Get the original body printed in HLS
    std::ostringstream current;
    current << stream.str();

    stream.str("");
    stream.clear();
    stream << "\n";

    enable_native_dtype = true;
    auto undef = UndefinedVars(op->body, {});
    for (auto& var : undef) {
      auto var_ptr = var.get();
      CHECK(var_shape_map_.count(var_ptr));
      CHECK(handle_data_type_.count(var_ptr));
      auto shape = var_shape_map_.at(var_ptr);
      auto type = handle_data_type_.at(var_ptr);

      PrintIndent();
      PrintType(type, stream);
      stream << " " << var.get()->name_hint;
      for (auto& dim : shape) {
        stream << "[" << PrintExpr(dim) << "]";
      }
      stream << ";\n";
    }

    stream << "#pragma scop\n";
    PrintStmt(op->body);
    enable_native_dtype = false;
    stream << "#pragma endscop\n";

    // Add the printer to keep tensor alive
    for (auto& var : undef) {
      auto var_ptr = var.get();
      CHECK(var_shape_map_.count(var_ptr));
      CHECK(handle_data_type_.count(var_ptr));
      auto shape = var_shape_map_.at(var_ptr);
      auto type = handle_data_type_.at(var_ptr);

      std::string token = "[0]";
      PrintIndent();

      if (type.code() == Type::Float) {
        stream << "printf(\"%f\", " << var_ptr->name_hint;
      } else {
        stream << "printf(\"%d\", " << var_ptr->name_hint;
      }
      for (size_t k = 0; k < shape.size(); k++) {
        stream << token;
      }
      stream << ");\n";
    }

    std::string body = stream.str();
    // Restore the original string copy
    stream.str("");
    stream.clear();
    stream << current.str();

    Array<Expr> ret =
        (*f)(op->attr_key, op->annotate_keys, op->annotate_values, body);
    CHECK_EQ(ret.size(), 2);
    CHECK(ret[0].as<StringImm>());
    CHECK(ret[1].as<StringImm>());

    std::string code = ret[1].as<StringImm>()->value;
    std::string header = ret[0].as<StringImm>()->value;
    HCL_DEBUG_LEVEL(2) << code;
    stream << code;
    decl_stream << header;
  }
}

}  // namespace codegen
}  // namespace TVM
