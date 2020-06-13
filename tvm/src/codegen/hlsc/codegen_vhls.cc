/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.cc
 */
#include <tvm/build_module.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "./codegen_vhls.h"
#include "../build_common.h"
#include "../build_soda.h"
#include "../codegen_soda.h"
#include "../../pass/stencil.h"

namespace TVM {
namespace codegen {

class StreamChecker final : public IRVisitor {
  public:
    bool stream_fifo{false};
    void Visit_(const Allocate* op) {
      if (op->attrs.size() > 0) stream_fifo = true;
      this->Visit(op->body);
    }
};

void CodeGenVivadoHLS::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  // write header files
  this->decl_stream << "#include <ap_int.h>\n";
  this->decl_stream << "#include <ap_fixed.h>\n";
  this->decl_stream << "#include <ap_axi_sdata.h>\n";
  this->decl_stream << "#include <hls_stream.h>\n";
  this->decl_stream << "#include <math.h>\n";
  this->decl_stream << "#include <stdint.h>\n";

  // setup codegen mode
  if (map_arg_type.count("sdsoc")) {
    sdsoc_mode = true;
    ptr_mode = true;
    this->decl_stream << "#include \"sds_lib.h\"\n\n";
  } else if (map_arg_type.count("sdaccel")) {
    ptr_mode = true;
    this->decl_stream << "\n";
  }

  // clear previous generated state.
  this->InitFuncState(f);
  map_arg_type_ = map_arg_type;
  // add to alloc buffer type.
  for (const auto & kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

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
  if (soda_header_.is_open())
    soda_header_.close();
}

void CodeGenVivadoHLS::PrintType(Type t, std::ostream& os) {
  if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed()) {
    if (t.is_uint()) {
      os << "ap_uint<" << t.bits() << ">";
    } else if (t.is_int()) {
      os << "ap_int<" << t.bits() << ">";
    } else if (t.is_ufixed()) {
      os << "ap_ufixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
    } else {
      os << "ap_fixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
    }
  } else {
    CodeGenC::PrintType(t, os);
  }
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
  // TODO: find a betetr way to track streaming channels 
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
    stream << "pkt_b" << bits << " " << vid <<  "_temp;\n";
    PrintIndent(); 
    stream << vid <<  "_temp.set_data(" << PrintExpr(op->value) << ");\n";
    PrintIndent(); 
    stream << vid <<  "_temp.set_keep(-1);\n";
    PrintIndent(); 
    stream << vid << ".write(" << vid << "_temp);\n";
    return;
  }

  // handle SetSlice
  if (const SetSlice* ss = op->value.as<SetSlice>()) {
    Type t = op->value.type();
    Expr new_index_left = ir::Simplify(ss->index_left - 1);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    std::string rhs = PrintExpr(ss->value);
    PrintIndent();
    this->stream << ref
                 << "(" << PrintExpr(new_index_left) << ", " << PrintExpr(ss->index_right)
                 << ") = " << rhs << ";\n";
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

void CodeGenVivadoHLS::VisitExpr_(const Call *op, std::ostream& os) {  // NOLINT(*)
  if ((op->call_type == Call::Extern ||
      op->call_type == Call::PureExtern) && op->name == "sqrtf") {
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

void CodeGenVivadoHLS::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  if (op->new_expr.defined()) {
    CHECK_EQ(op->free_function, "nop");
    std::string new_data = PrintExpr(op->new_expr);
    this->PrintIndent();
    PrintType(op->type, stream);
    stream << "* "<< vid << '=' << new_data << ";\n";
  } else {
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

    bool not_alloc = false;
    // ptr mode for host in c++ (sdsoc)
    if (ptr_mode) {
      if (vid.find("_new") != std::string::npos) {
        not_alloc = true;
        vid.replace(vid.find("_new"), 4, "");
        var_idmap_[op->buffer_var.get()] = vid;

      // skip if buffer allocated in host scope
      } else if (vid.find("_channel") != std::string::npos) {
        vid.replace(vid.find("_channel"), 8, "");
        var_idmap_[op->buffer_var.get()] = vid;

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
      } else if (alloc_set_.find(vid) != alloc_set_.end()) {
        not_alloc = true;
      }

    // complete mode for host in c++ (vivado hls)
    } else {
      if (vid.find("_new") != std::string::npos) {
        vid.replace(vid.find("_new"), 4, "");
        var_idmap_[op->buffer_var.get()] = vid;
      }
      if (alloc_set_.find(vid) != alloc_set_.end())
        not_alloc = true;
    }

    // not allocate buffer for channel or moved data
    if (!not_alloc) {
      alloc_set_.insert(vid);
      this->PrintIndent();

      // allocate stream channels
      if (vid.find("_channel") != std::string::npos ||
          vid.find("_pipe") != std::string::npos) {

          stream << "hls::stream<";
          PrintType(op->type, stream);
          stream << " > " << vid << ";\n";

      } else {
        if (constant_size > 1) { // Transfer length one array to scalar
          if (vid.find("_reuse") != std::string::npos) {
            PrintType(op->type, stream);
            stream << ' '<< vid;
            for (size_t i = 0; i < op->extents.size(); i++) {
              stream << '[';
              PrintExpr(op->extents[i], stream);
              stream << "]";
            }
          } else {
            if (sdsoc_mode) {
              // allocate continuous phy mem
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
              PrintType(op->type, stream);
              stream << ' '<< vid;
              // stream << '[' << constant_size << "]";
              for (size_t i = 0; i < op->extents.size(); i++) {
                stream << '[';
                PrintExpr(op->extents[i], stream);
                stream << "]";
              }
            }
          }
        } else {
          PrintType(op->type, stream);
          stream << ' '<< vid;
        }
        stream << ";\n";
        for (size_t i = 0; i < op->attrs.size(); i++) 
          this->PrintStmt(op->attrs[i]);

      }
    }
    buf_length_map_[buffer] = constant_size;
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeGenVivadoHLS::VisitStmt_(const For* op) {
  std::ostringstream os;

  if (ptr_mode) {
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
  }

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
    if (unroll_factor > 0) os << " factor=" << unroll_factor << "\n";
    else                   os << "\n";
  }
  else if (op->for_type == ForType::Pipelined) {
    int II = 0, i = 0;
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
    os << "#pragma HLS pipeline";
    if (II > 0) os << " II=" << II << "\n";
    else        os << "\n";
  }
  GenForStmt(op, os.str(), false);
}

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

void CodeGenVivadoHLS::VisitExpr_(const StreamExpr* op, std::ostream& os) {
  std::string vid = GetVarID(op->buffer_var.get());
  os << vid << ".read()";
}

// generate the module as blackbox
void CodeGenVivadoHLS::VisitStmt_(const ExternModule* op) {
  std::string ip_name, func, header;
  std::vector<std::string> args_in, args_out, indices; 

  PrintIndent();
  for (size_t i = 0; i < op->annotate_keys.size(); i++) {
    auto key = op->annotate_keys[i].as<StringImm>()->value;
    if (key == "name") {
      ip_name = op->annotate_values[i].as<StringImm>()->value;
    } else if (key == "header") {
      header = op->annotate_values[i].as<StringImm>()->value;
    } else if (key == "func") {
      func = op->annotate_values[i].as<StringImm>()->value;
    } else if (key.find("input") != std::string::npos) { 
      auto arg = op->annotate_values[i].as<StringImm>()->value;
      args_in.push_back(arg);
    } else if (key.find("output") != std::string::npos) {
      auto arg = op->annotate_values[i].as<StringImm>()->value;
      args_out.push_back(arg);
    } else if (key.find("index") != std::string::npos) {
      auto idx = op->annotate_values[i].as<StringImm>()->value;
      indices.push_back(idx);
    }
  }

  // generate external ip core
  if (indices.size() > 0) {
    CHECK(indices.size() == args_in.size() + args_out.size());
    // initialize temp values
    for (auto arg : args_out) {
      stream << "ap_int<32> " << arg << "_temp;\n";
      PrintIndent();
    }

    stream << ip_name << "(";
    auto index = 0;
    for (auto arg : args_in) {
      if (index > 0) stream << ", ";
      stream << arg << "[" << indices[index] << "]";
      index++;
    }
    for (auto arg : args_out) {
      if (index > 0) stream << ", ";
      stream << arg << "_temp"; index++;
    }
    stream << ");\n";

    // assign temp value back
    index = args_in.size();
    for (auto arg : args_out) {
      PrintIndent();
      stream << arg << "[" << indices[index++]
             << "] = " << arg << "_temp;\n";
    }

  } else {
    stream << func << "\n";
  }

  // generate TCL and Makefile
  decl_stream << header << "\n";
}

void CodeGenVivadoHLS::VisitStmt_(const StreamStmt* op) {
  std::string vid = GetVarID(op->buffer_var.get());
  // ptr operation for host-device communication in sdsoc
  switch (op->stream_type) {
    case StreamType::FIFO:
    case StreamType::DoubleBuffer:
    case StreamType::Copy:
      PrintIndent();
      stream << vid << ".write(";
      PrintExpr(op->value, stream);
      stream << ");\n";
      break;
  }
}

class AllocateCollector final : public IRVisitor {
  public:
    AllocateCollector(std::vector<const Allocate*>& alloc_list,
                      VarExprUnorderedSet& outputs)
      : alloc_list_(alloc_list), outputs_(outputs) {}

    void Visit_(const Allocate* op) {
      if (outputs_.count(op->buffer_var))
        alloc_list_.push_back(op);
      this->Visit(op->body);
    }

  private:
    std::vector<const Allocate*>& alloc_list_;
    VarExprUnorderedSet& outputs_;
};

void CodeGenVivadoHLS::VisitStmt_(const KernelStmt *op) {
  PrintIndent();
  stream << op->name << "(";
  std::unordered_map<int, int> arg_info;
  for (size_t k = 0; k < op->annotate_keys.size(); k++) {
    auto key = op->annotate_keys[k].as<StringImm>()->value;
    if (key == "pos") {
      auto pos = op->annotate_values[k].as<IntImm>()->value;
      auto idx = op->annotate_values[k+1].as<IntImm>()->value;
      arg_info[pos] = idx;
    }
  }
  for (size_t i = 0; i < op->args.size(); i++) {
    PrintExpr(op->args[i], stream);
    if (i < op->args.size() - 1) stream << ", ";
  }
  stream << ");\n";
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
  for (const auto & k : op->args) {
    RegisterHandleType(k.get(), k.get()->type);
  }

  // collect argument information
  std::unordered_map<int, int> arg_info;
  for (size_t i = 0; i < op->channels.size(); i++) {
    auto info = op->channels[i];
    auto pos = info[0].as<IntImm>()->value;
    auto idx = info[1].as<IntImm>()->value;
    if (idx > 0) arg_info[pos] = idx;
  }

  // print kernel function
  if (op->name.find("test") != std::string::npos) {

    int stream_arg_num = 0;
    // extract the memory port information
    std::unordered_map<int, std::vector<int>> mem_mapping;
    CHECK(op->channels.size() == op->args.size());
    for (size_t i = 0; i < op->channels.size();i++) {
      auto info = op->channels[i];
      CHECK(info.size() == 7);
      auto pos         = info[0].as<IntImm>()->value;
      // auto channel   = info[1].as<IntImm>()->value;
      // auto depth     = info[2].as<IntImm>()->value;
      // auto is_sender = info[3].as<IntImm>()->value;
      int mem          = info[4].as<IntImm>()->value;
      int port         = info[5].as<IntImm>()->value;
      int stream_type  = info[6].as<IntImm>()->value;
      mem_mapping[pos] = {mem, port, stream_type}; 
      if (static_cast<StreamType>(stream_type) == StreamType::FIFO) 
        stream_arg_num += 1;
    }

    // used as OpenCL kernel
    if (ptr_mode) {
      int extern_scope = -1;
      if (!sdsoc_mode) {
        extern_scope  = BeginScope();
        stream << "extern \"C\" {\n";
        PrintIndent();
      }

      stream << "void " << op->name << "(";
      std::vector<std::string> kernel_args;
      for (size_t i = 0; i < op->args.size(); ++i) {
        VarExpr v = op->args[i];
        var_shape_map_[v.get()] = op->arg_shapes[i];
        std::string vid = AllocVarID(v.get());

        CHECK(vid.find("_channel"))
          << vid << " not a channel";
        vid.replace(vid.find("_channel"), 8, "");

        // handle output-update-in-kernel case
        if (vid.find("_update") != std::string::npos) {
          vid.replace(vid.find("_update"), 7, "");
        }

        alloc_set_.insert(vid);
        alloc_set_.insert(vid + "_new");
        kernel_args.push_back(vid);

        if (i != 0) stream << ", ";
        std::string str = PrintExpr(op->arg_types[i]);
        Type type = String2Type(str);

        // pass-by-value argument
        if (var_shape_map_[v.get()].size() == 1 &&
            var_shape_map_[v.get()][0].as<IntImm>()->value == 1) {
          this->stream << "int " << vid;
        } else {
          CHECK(mem_mapping.count(i));
          CHECK(mem_mapping.at(i).size() == 3);
          auto stream_type = static_cast<StreamType>(mem_mapping[i][2]);

          if (stream_type == StreamType::FIFO) {
            auto bits = type.bits();
            if (decl_stream.str().find("typedef qdma_axis<" + 
                    std::to_string(bits)) == std::string::npos) {
              decl_stream << "typedef qdma_axis<" << bits 
                          << ", 0, 0, 0> pkt_b" << bits << ";\n";
            }
            stream << "hls::stream<pkt_b" << bits << "> &" << vid;
            stream_vars.insert(vid);
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

      // port-level protocol interface
      CHECK(op->args.size() == kernel_args.size());
      for (size_t i = 0; i < kernel_args.size(); i++) {
        if (op->arg_shapes[i].size() == 1 &&
            op->arg_shapes[i][0].as<IntImm>()->value == 1) {
          continue;
        } else {
          PrintIndent();
          auto port = mem_mapping[i][1];
          auto type = static_cast<StreamType>(mem_mapping[i][2]);

          if (type == StreamType::FIFO) {
            stream << "#pragma HLS INTERFACE axis port="
                   << kernel_args[i] << "\n";
          } else {
            stream << "#pragma HLS INTERFACE m_axi port="
                   << kernel_args[i] << " "
                   << "offset=slave bundle=gmem" << port << "\n";
          }
        }
      }

      // block-level control interface 
      for (size_t i = 0; i < kernel_args.size(); i++) {
        auto type = static_cast<StreamType>(mem_mapping[i][2]);
        if (type == StreamType::FIFO) continue;
        PrintIndent();
        stream << "#pragma HLS INTERFACE s_axilite port="
               << kernel_args[i] << " "
               << "bundle=control\n";
      }
      PrintIndent();
      stream << "#pragma HLS INTERFACE s_axilite"
             << " port=return bundle=control\n";

      // TODO: add dataflow premitive
      StreamChecker sc; sc.Visit(op->body);
      if (sc.stream_fifo) {
        stream << "\n";
        PrintIndent();
        stream << "#pragma HLS dataflow\n";
      }

      // function body
      int func_scope = BeginScope();
      range_ = CollectIterRange(op->body);
      PrintStmt(op->body);

      EndScope(func_scope);
      PrintIndent();
      stream << "}\n";

      // end extern c scope
      if (!sdsoc_mode) {
        stream << "}\n\n";
        EndScope(extern_scope);
      }
      stream_vars.clear();

    // used as VHLS kernel
    } else {

      PrintIndent();
      stream << "void " << op->name << "(";
      std::vector<std::string> kernel_args;
      for (size_t i = 0; i < op->args.size(); ++i) {
        VarExpr v = op->args[i];
        var_shape_map_[v.get()] = op->arg_shapes[i];
        std::string vid = AllocVarID(v.get());
        kernel_args.push_back(vid);

        if (i != 0) stream << ", ";
        std::string str = PrintExpr(op->arg_types[i]);
        Type type = String2Type(str);

        // pass-by-value argument
        if (var_shape_map_[v.get()].size() == 1 &&
            var_shape_map_[v.get()][0].as<IntImm>()->value == 1) {
          this->stream << "int " << vid;
        } else {
          stream << "hls::stream<";
          PrintType(type, stream);
          stream << " >& " << vid;
        }
      }
      stream << ") {\n";

      // port-level protocol interface
      CHECK(op->args.size() == kernel_args.size());
      for (size_t i = 0; i < kernel_args.size(); i++) {
        if (op->arg_shapes[i].size() == 1 &&
            op->arg_shapes[i][0].as<IntImm>()->value == 1) {
          continue;
        } else {
          PrintIndent();
          stream << "#pragma HLS INTERFACE axis port="
                 << kernel_args[i]
                 << " offset=slave bundle=gmem" << i << "\n";
        }
      }
      // TODO: allow AXI memory copy  
      // block-level control interface 
      // for (size_t i = 0; i < kernel_args.size(); i++) {
      //   PrintIndent();
      //   stream << "#pragma HLS INTERFACE s_axilite port="
      //          << kernel_args[i] << " "
      //          << "bundle=control\n";
      // }
      PrintIndent();
      stream << "#pragma HLS INTERFACE s_axilite"
             << " port=return bundle=control\n";

      StreamChecker sc; sc.Visit(op->body);
      if (sc.stream_fifo) {
        stream << "\n";
        PrintIndent();
        stream << "#pragma HLS dataflow\n";
      }

      // function body
      int func_scope = BeginScope();
      range_ = CollectIterRange(op->body);
      PrintStmt(op->body);

      EndScope(func_scope);
      PrintIndent();
      stream << "}\n";

    }

  } else { // regular vhls function

    stream << "static void " << op->name << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      VarExpr v = op->args[i];
      var_shape_map_[v.get()] = op->arg_shapes[i];
      std::string vid = AllocVarID(v.get());
      if (i != 0) stream << ", ";
      std::string str = PrintExpr(op->arg_types[i]);
      Type type = String2Type(str);

      // arg as streaming channel
      if (arg_info.find(i) != arg_info.end()) {
        stream << "hls::stream<";
        PrintType(type, stream);
        stream << " >& " << vid;

      } else {
        PrintType(type, stream);
        if (op->arg_shapes[i].size() == 0)
          this->stream << " " << vid;
        else stream << "][" << vid;
      }
    }
    stream << ") {\n";

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
  for (size_t i = 0; i < op->inputs.size(); i++)
    inputs.insert(op->inputs[i]);
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
    if (i < op->outputs.size()-1) {
      soda_header_ << ", ";
      stream << ", ";
    }
  }
  soda_header_ << ");\n";
  stream << ");\n";

  // Generate SODA HLSC code
  std::ofstream soda_file;
  soda_file.open(kernel_name+".cpp");
  soda_file << "#include \"soda_stencil.h\"\n";
  soda_file << code;
  soda_file.close();
}

}  // namespace codegen
}  // namespace TVM
