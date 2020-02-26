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
  // Write header files
  this->decl_stream << "#include <ap_int.h>\n";
  this->decl_stream << "#include <ap_fixed.h>\n";
  this->decl_stream << "#include <hls_stream.h>\n";
  this->decl_stream << "#include <math.h>\n";
  this->decl_stream << "#include <stdint.h>\n\n";
  if (map_arg_type.find("sdsoc") != map_arg_type.end())
    sdsoc_mode = true;
  CodeGenHLSC::AddFunction(f, map_arg_type);
  if (soda_header_.is_open())
    soda_header_.close();
}

void CodeGenVivadoHLS::PrintType(Type t, std::ostream& os) {
  if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed()) {
    if (t.is_uint())        os << "ap_uint<" << t.bits() << ">";
    else if (t.is_int())    os << "ap_int<" << t.bits() << ">";
    else if (t.is_ufixed()) os << "ap_ufixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
    else                    os << "ap_fixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
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

void CodeGenVivadoHLS::VisitStmt_(const Store* op) {
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

    // ignore channel and pipe buffers
    if (vid.find("c_buf_") == std::string::npos &&
        vid.find("channel") == std::string::npos &&
        vid.find("_new") == std::string::npos) {

      this->PrintIndent();
      PrintType(op->type, stream);
      stream << ' '<< vid;
      if (constant_size > 1) { // Transfer length one array to scalar
        if (vid.find("_reuse") != std::string::npos) {
          for (size_t i = 0; i < op->extents.size(); i++) {
            stream << '[';
            PrintExpr(op->extents[i], stream);
            stream << "]";
          }
        } else {
          stream << '[' << constant_size << "]";
        }
      }
      stream << ";\n";
      // pragmas associated with allocate 
      // for (auto& k : op->attrs) {
      //   if (!k.as<StreamStmt>()) this->PrintStmt(k);
      // }
    }
    buf_length_map_[buffer] = constant_size;
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeGenVivadoHLS::VisitStmt_(const For* op) {
  std::ostringstream os;
  // ignore the data tranmission for stmts
  if (const For* for_op = op->body.as<For>()) {
    while (for_op->body.as<For>())
      for_op = for_op->body.as<For>();
    if (auto s = for_op->body.as<StreamStmt>()) { 
      if (s->buffer_var.get()->name_hint.find("channel") 
          != std::string::npos) return;
    } else if (auto st = for_op->body.as<Store>()) {
      if (auto e = st->value.as<StreamExpr>()) {
        if (e->buffer_var.get()->name_hint.find("channel")
            != std::string::npos) return;

      } else {// ignore the initilization 
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
  CodeGenC::VisitExpr_(op, os);
  std::string vid = GetVarID(op->buffer_var.get());
  int channel_index = 0; 
  Expr index_expr;
  for (size_t i = 0; i < op->annotate_keys.size(); i++) {
    auto key = op->annotate_keys[i].as<StringImm>()->value;
    if (key == "channel") {
      channel_index = op->annotate_values[i].as<IntImm>()->value;
    } else if (key == "index") {
      index_expr = op->annotate_values[i];
    }
  }
  if (channel_index == 0 && !sdsoc_mode) {
    os << vid << ".read()";
  } else { // axi stream: set the removed itervar as zero 
    // os << vid << "[" 
    //    << PrintExpr(index_expr) << "]";
  }
}

void CodeGenVivadoHLS::VisitStmt_(const StreamStmt* op) {
  std::string vid = GetVarID(op->buffer_var.get());
  switch (op->stream_type) {
    case StreamType::FIFO:
      PrintIndent();
      stream << "#pragma HLS stream variable="
             << vid << " depth=" << op->depth << "\n"; 
      break;
    case StreamType::Channel:
      PrintIndent();
      stream << vid << " << ";
      PrintExpr(op->value, stream);
      stream << ";\n"; 
      break;
    case StreamType::Pipe:
      PrintIndent();
      stream << vid << " << ";
      PrintExpr(op->value, stream);
      stream << ";\n"; 
      break;
  }
  // int channel_index = 0;
  // Expr index_expr;
  // for (size_t i = 0; i < op->annotate_keys.size(); i++) {
  //   auto key = op->annotate_keys[i].as<StringImm>()->value;
  //   if (key == "channel") 
  //     channel_index = op->annotate_values[i].as<IntImm>()->value;
  //   else if (key == "index")
  //     index_expr = op->annotate_values[i];
  // }
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
    if (arg_info.find(i) != arg_info.end()) {
      if (arg_info[i] == 0 && !sdsoc_mode) 
        stream << "fd_";
    }
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
  for (size_t i = 0; i < op->channels.size(); i=i+2) {
    auto pos = op->channels[i].as<IntImm>()->value;
    auto idx = op->channels[i+1].as<IntImm>()->value;
    if (idx > 0) arg_info[pos] = idx;
  }

  // print kernel function
  if (op->name.find("top_function_") != std::string::npos) {
    int extern_scope = BeginScope();
    stream << "extern \"C\" {\n";

    PrintIndent();
    stream << "void " << op->name << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      VarExpr v = op->args[i];
      var_shape_map_[v.get()] = op->api_args[i];
      std::string vid = AllocVarID(v.get());
      if (i != 0) stream << ", ";
      std::string str = PrintExpr(op->api_types[i]);
      Type type = String2Type(str);

      if (var_shape_map_[v.get()].size() == 1 &&
          var_shape_map_[v.get()][0].as<IntImm>()->value == 1) { 
        this->stream << "int " << vid;
      } else {
        PrintType(type, stream);
        stream << "* " << vid;
      }
    }
    stream << ") {\n";

    // memory interface 
    for (size_t i = 0; i < op->args.size(); i++) {
      if (op->api_args[i].size() == 1 &&
          op->api_args[i][0].as<IntImm>()->value == 1) {
        continue;
      } else {
        PrintIndent();
        stream << "#pragma HLS INTERFACE m_axi port="
               << GetVarID(op->args[i].get()) << " "
               << "offset=slave bundle=gmem" << i << "\n";
      }
    }
    // control interface 
    for (size_t i = 0; i < op->args.size(); i++) {
      PrintIndent();
      stream << "#pragma HLS INTERFACE s_axilite port="
             << GetVarID(op->args[i].get()) << " "
             << "bundle=control\n";
    }
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

    // end extern c scope
    stream << "}\n\n";
    EndScope(extern_scope);

  } else { // regular vhls function  

    stream << "static void " << op->name << "(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      VarExpr v = op->args[i];
      var_shape_map_[v.get()] = op->api_args[i];
      std::string vid = AllocVarID(v.get());
      if (i != 0) stream << ", ";
      std::string str = PrintExpr(op->api_types[i]);
      Type type = String2Type(str);

      // arg as streaming channel 
      if (arg_info.find(i) != arg_info.end()) {
        stream << "hls::stream<";
        PrintType(type, stream);
        stream << ">& " << vid;

      } else {
        PrintType(type, stream);
        if (op->api_args[i].size() == 0) 
          this->stream << " " << vid;
        else stream << "* " << vid;
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
  std::string func_name = "soda_" + 
                          op->inputs[0]->name_hint + "_" +
                          op->outputs[0]->name_hint;
  cg_soda.PrintSODA(func_name, op->burst_width, op->unroll_factor,
      op->num_iteration, op->body, inputs, outputs);
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
  soda_header_ << "void " + func_name + "_kernel(";
  stream << func_name + "_kernel(";
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
  soda_file.open(func_name+".cpp");
  soda_file << "#include \"soda_stencil.h\"\n";
  soda_file << code;
  soda_file.close();
}

}  // namespace codegen
}  // namespace TVM
