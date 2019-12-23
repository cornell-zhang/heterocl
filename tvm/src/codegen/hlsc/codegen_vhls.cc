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

class VarCollector : public IRVisitor {
 public:
  explicit VarCollector(
      std::unordered_map<const Variable*, Expr>& range)
      : range_(range) {}
  void Visit_(const Variable* op) {
    if (range_.find(op) == range_.end()) 
      var_list_[op] = Expr(0);
  }
  std::unordered_map<const Variable*, Expr> var_list_;
 private:
  std::unordered_map<const Variable*, Expr>& range_;
};

// add local buffers to generated kernel code 
void CodeGenVivadoHLS::PreProcess(std::ostringstream& os) {
  if (sdsoc_mode) return;
  os << "\n";
  int indent = 2;
  for (size_t i = 0; i < arg_vars.size(); i++) {
    auto v = arg_vars[i];
    std::string arg_name;
    if (stream_table[v]) 
      arg_name = arg_top_vars[v].name;
    else arg_name = GetVarID(v); 

    // create local buffer saving result
    auto shape = arg_top_vars[v].shape;
    auto dtype = arg_top_vars[v].type;
    if (!stream_table[v]) { // unstreamed args 
      // allocate local buffer 
      for (int k = 0; k < indent; k++) os << ' ';
      PrintType(dtype, os); 
      os << " " << arg_name << "[";
      for (size_t n = 0; n < shape.size(); n++) {
        os << shape[n];
        if (n != shape.size() - 1) os << "* ";
      } 
      os << "];\n";
 
      for (size_t j = 0; j < shape.size(); j++) {
        for (int k = 0; k < indent; k++) os << ' ';
        os << "for (int i" << j << " = 0; i"
           << j << "< " << shape[j] << "; i" 
           << j << "++) {\n";
        // pass stream reference
        if (j == shape.size() - 1) {
          for (int k = 0; k < indent; k++) os << ' ';
          os << "  " << arg_name << "["
             << getIndex(shape) << "] = "  
             << "fd_" << arg_name << ".read();\n";
        }
        indent += 2;
      }
      for (size_t m = 0; m < shape.size(); m++) {
        indent -= 2;
        for (int k = 0; k < indent; k++) os << ' ';
        os << "}\n";
      }
    } else if (i == arg_vars.size() - 1 || true) { 
      // allocate for return variable 
      for (int k = 0; k < indent; k++) os << ' ';
      PrintType(dtype, os); 
      os << " " << arg_name << "[";
      for (size_t n = 0; n < shape.size(); n++) {
        os << shape[n];
        if (n != shape.size() - 1) os << "* ";
      } 
      os << "];\n";
    }
  }
}

void CodeGenVivadoHLS::PostProcess(std::ostringstream& os) {
}

void CodeGenVivadoHLS::AddFunction(LoweredFunc f,
        str2tupleMap<std::string, Type> map_arg_type) {
  // Write header files
  this->decl_stream << "#include <ap_int.h>\n";
  this->decl_stream << "#include <ap_fixed.h>\n";
  this->decl_stream << "#include <hls_stream.h>\n";
  this->decl_stream << "#include <math.h>\n\n";
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
  } else if (const StreamExpr* se = op->value.as<StreamExpr>()) {
    std::string vid = GetVarID(se->buffer_var.get()); 
    vid = vid.substr(0, vid.find("_stream_send")); 
    PrintIndent();
    if (!sdsoc_mode) { // loading 
      this->stream << vid << "["
                   << op->index << "] = "
                   << "fd_" << vid << ".read();\n";
    } else { // skip load in xcel 
      if (fpga_scope_)
        this->stream << "void(0);\n";
      else // read data back
        this->stream << vid << "["
                     << op->index << "] = "
                     << "fd_" << vid << "[" 
                     << op->index << "];\n";
    }
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenVivadoHLS::VisitStmt_(const For* op) {
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
  vid = vid.substr(0, vid.find("_stream_send")); 
  int channel_index = 0; Expr index_expr;
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
    VarCollector visitor(range_);
    visitor.Visit(index_expr);
    index_expr = Simplify(Substitute(index_expr, visitor.var_list_));
    os << vid << "[" 
       << PrintExpr(index_expr) << "]";
  }
}

void CodeGenVivadoHLS::VisitStmt_(const StreamStmt* op) {
  CodeGenC::VisitStmt_(op);
  std::string vid = GetVarID(op->buffer_var.get());
  switch (op->stream_type) {
    case StreamType::Channel:
      break;
    case StreamType::FIFO:
      break;
    case StreamType::Pipe:
      break;
  }
  int channel_index = 0;
  Expr index_expr;
  for (size_t i = 0; i < op->annotate_keys.size(); i++) {
    auto key = op->annotate_keys[i].as<StringImm>()->value;
    if (key == "channel") 
      channel_index = op->annotate_values[i].as<IntImm>()->value;
    else if (key == "index")
      index_expr = op->annotate_values[i];
  }
  vid = vid.substr(0, vid.find("_stream_send")); 
  auto load = op->value.as<Load>();
  if (channel_index == 0) {
    if (sdsoc_mode) { 
      if (fpga_scope_) { // skip writing from xcel
        stream << "void(0);\n"; return;
      } else { // init value on host
        stream << "fd_" << vid << "[";
        PrintExpr(load->index, stream);
        stream << "] = " << vid << "[";
        PrintExpr(load->index, stream);
        stream << "];\n";
      }
    } else { // write into hls stream 
      stream << "fd_" << vid << ".write(" 
             << vid << "[";
      PrintExpr(load->index, stream);
      stream << "]);\n";
    }
  } else { // hls axis interface
    VarCollector visitor(range_);
    visitor.Visit(index_expr);
    index_expr = Simplify(Substitute(index_expr, visitor.var_list_));
    stream << vid << "[" << PrintExpr(index_expr) << "] = ";
    stream << GetVarID(load->buffer_var.get()) << "[";
    PrintExpr(load->index, stream);
    stream << "];\n";
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

void CodeGenVivadoHLS::VisitStmt_(const AttrStmt* op) {
  if (op->attr_key == ir::attr::device_scope) {
    // print top( ... in host and enter fpga scope 
    if (op->value.as<StringImm>()->value == "fpga" && !fpga_scope_) {
      fpga_scope_ = true;
      PrintIndent();
       
      // track the stream usage
      StreamCollector collector(stream_table, "cpu");
      collector.Visit(op->body);

      // update data type and name 
      for (auto k : collector.host_undefined_) {
        auto v = k.get();
        arg_vars.push_back(v);
        stream_table[v] = true;
        auto prev = arg_top_vars[v];
        arg_top_vars[v] = {v->name_hint, prev.type, prev.shape};
      }
  
      // generte function calls 
      stream << "top(";
      for (size_t i = 0; i < arg_vars.size(); i++) {
        auto v = arg_vars[i];
        auto shape = arg_top_vars[v].shape;
        std::string arg_name;
        if (stream_table[v]) 
          arg_name = arg_top_vars[v].name;
        else arg_name = GetVarID(v); 
        if (i != 0) stream << ", ";
        if (shape.size() > 1 || shape[0] > 1) stream << "fd_"; 
        stream << arg_name;

        // generate kernel func definition
        if (i != 0) arg_stream << ", ";
        if ((shape.size() > 1 || shape[0] > 1) && !sdsoc_mode) { 
          arg_stream << "hls::stream<";
          PrintType(arg_top_vars[v].type, arg_stream);
          arg_stream << ">& fd_" << arg_name;
        } else { // for scalar & sdsoc
          PrintType(arg_top_vars[v].type, arg_stream);
          arg_stream << "* " << arg_name;
        }
      }
      stream << ");\n";
  
      // switch context to device scope
      host_stream << this->stream.str();
      this->stream.str("");
      this->stream.clear();
  
    // swtich from device to host
    } else if (op->value.as<StringImm>()->value == "cpu" && 
               fpga_scope_) {
      fpga_scope_ = false;
      device_stream << this->stream.str();
      this->stream.str("");
      this->stream.clear();
    }
    this->PrintStmt(op->body);

  } else if (op->attr_key == "pragma") {
    auto alloc = op->body.as<Allocate>();
    CHECK(alloc != nullptr) << "wrong scope";

    int32_t constant_size = alloc->constant_allocation_size();
    std::string vid = AllocVarID(alloc->buffer_var.get());

    PrintType(alloc->type, stream);
    stream << ' '<< vid;
    if (constant_size > 1) {
      stream << "[";
      for (size_t i = 0; i < alloc->extents.size(); i++) {
        PrintExpr(alloc->extents[i], stream);
        if (i != alloc->extents.size()-1) stream << "* ";
      }
      stream << "]";
    }
    stream << ";\n";
    PrintIndent(); // consider add alloc->attrs
    stream << "#pragma HLS STREAM variable=" 
           << vid << " depth=10\n";
    this->PrintStmt(alloc->body);

  } else {
    CodeGenC::VisitStmt_(op);
  }
}

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
    arg_info[pos] = idx;
  }
    
  // print function signature
  PrintType(op->ret_type, stream);
  size_t active_spot = 0;
  stream << " " << op->name << "(";
  for (size_t i = 0; i < op->args.size(); ++i) {
    VarExpr v = op->args[i];
    var_shape_map_[v.get()] = op->api_args[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    std::string str = PrintExpr(op->api_types[i]);
    Type type = String2Type(str);

    // TODO: broadcast in hlsc (one wr multi read) 
    bool axis_enable = false;

    // in sdsoc mode    
    if (sdsoc_mode) {
      PrintType(type, stream);
      stream << "* " << vid;
      // check arg streming statue
      if (arg_info.find(i) != arg_info.end()) {
        active_spot = i;
        if (pragma.str().length() == 0)
          pragma << "#pragma SDS data access_pattern(";
        if (active_spot > 0 && active_spot == i)
          pragma << ", ";
        pragma << vid << ":SEQUENTIAL";
      }
      // output comma
      if (i == op->args.size() - 1 &&
          pragma.str().length() > 0 ) {
        pragma << ")\n"; 
      }
      continue;
    }

    // pass the stream channel reference 
    if (arg_info.find(i) != arg_info.end()) {
      if (arg_info[i] == 0) {
        stream << "hls::stream<";
        PrintType(type, stream);
        stream << ">& " << vid;
        pragma << "#pragma HLS stream depth=1 variable=" 
               <<  vid << "\n";
      } else { axis_enable = true; }
    } // ordinary arg or hls axis interface 
    if (arg_info.find(i) == arg_info.end() ||
        axis_enable == true) {
      PrintType(type, stream);
      this->stream << " " << vid << "[";
      int mul = 1;
      for (size_t j = 0; j < op->api_args[i].size(); j++) {
        auto dim = op->api_args[i][j].as<IntImm>()->value;
        mul = mul * dim;
      }
      this->stream << mul << "]";
      if (axis_enable)
        pragma << "#pragma HLS interface axis port=" 
               <<  vid << "\n";
    }
  }  
  stream << ") {\n";
  stream << pragma.str();
  int func_scope = BeginScope();
  range_ = CollectIterRange(op->body);
  PrintStmt(op->body);
  EndScope(func_scope);
  stream << "}\n\n";

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
