/*!
 * Copyright (c) 2021 by Contributors
 * \file codegen_shls.cc
 */
#include <sys/types.h>
#include <sys/wait.h>
#include <tvm/build_module.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <vector>
#include "../../pass/stencil.h"
#include "../build_common.h"
#include "../build_soda.h"
#include "../codegen_soda.h"
#include "./hierarchy.h"
#include "./port_direction.h"
#include "codegen_shls.h"

namespace TVM {
namespace codegen {

inline void printHeader(std::ostream& os) { os << "#include <cynw_p2p.h>\n"; }

void CodeGenStratusHLS::printTclFile() {
  // Add following two lines to project.tcl
  // if there is external memory access
  if (this->ext_mem.size() == 0) return;
  this->support_fnames.push_back("project.tcl");
  this->support_files.push_back(
      "use_hls_lib \"./memlib\"\n"
      "define_external_array_access -to System -from dut");
}

void CodeGenStratusHLS::GenerateModule(
    std::string name, bool top_level,
    str2tupleMap<std::string, Type> map_arg_type, const Array<Expr> arg_types,
    const Array<Array<Expr>> arg_shapes, const Stmt body,
    const Array<VarExpr> args, std::ostringstream& decl_os,
    std::ostringstream& ctor_os, std::ostringstream& body_os) {
  if (top_level) _top_name = name;
  printHeader(decl_os);

  decl_os << "SC_MODULE("
          << "name"
          << ") \n{\n";
  int module_scope = this->BeginScopeHeader();
  PrintIndentAnyStream(decl_os, h_indent_);
  decl_os << "sc_in<bool> clk;\n";
  PrintIndentAnyStream(decl_os, h_indent_);
  decl_os << "sc_in<bool> rst;\n";
  if (top_level) {
    PrintIndentAnyStream(decl_os, h_indent_);
    decl_os << "sc_out<bool> finish;\n\n";
  }

  std::vector<std::string> port_names;
  _port_names.push_back(port_names);
  level_ += 1;

  if (top_level) {
    for (auto it = map_arg_type.begin(); it != map_arg_type.end(); it++) {
      _port_names[level_].push_back(std::get<0>(it->second));
    }
  } else {
    for (size_t i = 0; i < args.size(); ++i) {
      VarExpr v = args[i];
      std::string vid = AllocVarID(v.get());
      _port_names[level_].push_back(vid);
    }
  }

  // a list of scalar input argument
  std::vector<std::string> scalars;
  for (size_t i = 0; i < args.size(); ++i) {
    std::string vid = _port_names[level_][i];
    if (arg_shapes[i].size() == 1 &&
        arg_shapes[i][0].as<IntImm>()->value == 1) {
      scalars.push_back(vid);
    }
  }

  // Infer port direction
  PortDirection port_visitor(_port_names[level_], scalars);
  port_visitor.Visit(body);

  // Generate port definitions
  PrintIndentAnyStream(decl_os, h_indent_);
  decl_os << "// port definitions\n";
  // External memory passed from constructor
  std::stringstream ext_mem_str;
  // Iterate all op arguments to generate ports
  for (size_t i = 0; i < args.size(); ++i) {
    std::string vid;
    if (top_level) {
      VarExpr v = args[i];
      vid = AllocVarID(v.get());
    } else {
      vid = _port_names[level_][i];
    }

    if (map_arg_type.find(vid) == map_arg_type.end() && top_level) {
      LOG(WARNING) << vid << " type not found\n";
    } else {
      std::tuple<std::string, Type> arg;
      if (top_level) arg = map_arg_type[vid];
      std::string arg_name;
      if (top_level)
        arg_name = std::get<0>(arg);
      else
        arg_name = vid;
      PortType port_type = port_visitor.get_direction(arg_name);
      PrintIndentAnyStream(decl_os, h_indent_);
      // memory port
      if (port_type == PortType::Memory ||
          port_type == PortType::OffChipMemory) {
        this->_port_type.insert(
            std::pair<std::string, std::string>(arg_name, "mem"));
        if (top_level)
          PrintType(std::get<1>(arg), decl_os);
        else
          PrintTypeStringImm(arg_types[i].as<StringImm>(), decl_os);
        if (port_type == PortType::OffChipMemory) {
          decl_os << "*\t" << arg_name << ";\n";
          this->ext_mem.push_back(arg_name);
          ext_mem_str << ", ";
          if (top_level)
            PrintType(std::get<1>(arg), ext_mem_str);
          else
            PrintTypeStringImm(arg_types[i].as<StringImm>(), ext_mem_str);
          ext_mem_str << " _" << arg_name << arg_shapes[i];
        } else {
          decl_os << "\t" << arg_name;
          decl_os << "[";
          int count = 0;
          for (auto& s : arg_shapes[i]) {
            if (count != 0) decl_os << "][";
            decl_os << s;
            count = count + 1;
          }
          decl_os << "];\n";
        }
      } else {  // channel port
        this->_port_type.insert(
            std::pair<std::string, std::string>(arg_name, "p2p"));
        std::string direction =
            (port_type == PortType::ChannelIn) ? "in" : "out";
        decl_os << "cynw_p2p < ";
        if (top_level)
          PrintType(std::get<1>(arg), decl_os);
        else
          PrintTypeStringImm(arg_types[i].as<StringImm>(), decl_os);
        decl_os << " >";
        decl_os << "::" << direction << "\t";
        decl_os << arg_name;  // print port name
        decl_os << ";\n";
      }
    }
  }
  decl_os << "\n";

  // find KernelDef nodes in LoweredFunc's body,
  // to avoid printing the redundant allocations.
  // this->sub_names will be checked in Allocate node.
  Hierarchy hierarchy;
  hierarchy.Visit(body);
  std::list<std::string> submodule_def = hierarchy.get_submodule_def();
  for (std::string sub_name : submodule_def)
    this->sub_names.push_back(sub_name);
  this->sub_names.push_back("_top");
  // print submodule instantiations
  std::list<std::string> submodules = hierarchy.get_submodules();
  std::map<std::string, std::vector<Expr>> submodule_args =
      hierarchy.get_submodule_args();
  std::map<std::string, std::vector<Expr>> submodule_arg_types =
      hierarchy.get_arg_types();
  std::map<std::string, std::vector<std::string>> submodule_arg_names =
      hierarchy.get_arg_names();
  UpdateSubPortTypes(submodule_arg_types);
  UpdateSubPortNames(submodule_arg_names);

  if (submodules.size() > 0) {
    PrintIndentAnyStream(decl_os, h_indent_);
    decl_os << "// submodule instantiations\n";
  }

  for (std::string submodule : submodules) {
    PrintIndentAnyStream(decl_os, h_indent_);
    decl_os << submodule << "\t" << submodule << "_inst;\n";
    if (_sub_port_types.count(submodule) == 0) {
      LOG(WARNING) << "Sub-module " << submodule << " argument types unknown.";
    }
    // print input argument definitions
    for (size_t i = 0; i < submodule_args[submodule].size(); ++i) {
      Expr arg = submodule_args[submodule][i];
      std::string arg_name = arg.as<Variable>()->name_hint;
      std::string type = _sub_port_types[submodule][i];
      canonicalize_string(arg_name);
      PrintIndentAnyStream(decl_os, h_indent_);
      decl_os << "cynw_p2p < " << type;
      decl_os << " >\t" << submodule << "_" << arg_name << ";\n";
    }
  }

  // Generate Constructor
  int ctor_scope = this->BeginScopeCtor();
  ctor_os << "\n";
  PrintIndentAnyStream(ctor_os, c_indent_);
  ctor_os << "SC_HAS_PROCESS(" << name << ");\n";
  PrintIndentAnyStream(ctor_os, c_indent_);
  ctor_os << name << "(sc_module_name name" << ext_mem_str.str() << ")\n";
  // initialize clock and reset
  PrintIndentAnyStream(ctor_os, c_indent_);
  ctor_os << ": "
          << "clk( "
          << "\"clk\""
          << " )\n";
  PrintIndentAnyStream(ctor_os, c_indent_);
  ctor_os << ", "
          << "rst( "
          << "\"rst\""
          << " )\n";
  // initialize i/o ports
  for (auto it = _port_names[level_].begin(); it != _port_names[level_].end();
       ++it) {
    std::string name = *it;
    if (!IsP2P(name)) continue;
    PrintIndentAnyStream(ctor_os, c_indent_);
    ctor_os << ", " << name << "( \"" << name << "\" )\n";
  }
  // pass external memory
  for (std::string mem_name : this->ext_mem) {
    PrintIndentAnyStream(ctor_os, c_indent_);
    ctor_os << ", " << mem_name << "(_" << mem_name << ")\n";
  }
  PrintIndentAnyStream(ctor_os, c_indent_);
  ctor_os << "{\n";
  // initlialize clocked thread
  int ctor_scope_inner = this->BeginScopeCtor();
  PrintIndentAnyStream(ctor_os, c_indent_);
  ctor_os << "SC_CTHREAD(thread1, clk.pos());\n";
  // setup reset signal
  PrintIndentAnyStream(ctor_os, c_indent_);
  ctor_os << "reset_signal_is(rst, 0);\n";
  // connect clk and rst power to modular interface ports
  for (auto it = _port_names[level_].begin(); it != _port_names[level_].end();
       ++it) {
    std::string name = *it;
    if (!IsP2P(name)) continue;
    PrintIndentAnyStream(ctor_os, c_indent_);
    ctor_os << name << '.' << "clk_rst(clk, rst);\n";
  }
  // Add directive for external memory
  // note that tolower(mem_name) is
  // supposed to present in the memlib
  // of the Stratus project
  for (std::string mem_name : this->ext_mem) {
    PrintIndentAnyStream(ctor_os, c_indent_);
    std::string mem_name_lib = mem_name;
    std::transform(mem_name_lib.begin(), mem_name_lib.end(),
                   mem_name_lib.begin(), ::tolower);
    ctor_os << "HLS_MAP_TO_MEMORY(" << mem_name << ", \"" << mem_name_lib
            << "\");\n";
  }
  // connect submodule ports
  for (std::string submodule : submodules) {
    ctor_os << "\n";
    PrintIndentAnyStream(ctor_os, c_indent_);
    ctor_os << "// " << submodule << "\n";
    // check arg name availability
    if (_sub_port_names.count(submodule) == 0) {
      LOG(WARNING) << "Submodule " << submodule << " arg names unknown.";
    }
    for (unsigned int i = 0; i < submodule_args[submodule].size(); i++) {
      Expr arg = submodule_args[submodule][i];
      std::string arg_name = arg.as<Variable>()->name_hint;
      std::string def_arg_name = _sub_port_names[submodule][i];
      canonicalize_string(arg_name);
      canonicalize_string(def_arg_name);
      PrintIndentAnyStream(ctor_os, c_indent_);
      ctor_os << submodule << "_inst." << def_arg_name << "(";
      ctor_os << submodule << "_" << arg_name << ");\n";
    }
    PrintIndentAnyStream(ctor_os, c_indent_);
    ctor_os << submodule << "_inst.clk(clk);\n";
    PrintIndentAnyStream(ctor_os, c_indent_);
    ctor_os << submodule << "_inst.rst(rst);\n";
  }

  // Generate SC_THREAD function implementation
  // generate process function
  body_os << "#include \"" << name << ".h\"\n";
  this->PrintIndent();
  body_os << "void " << name << "::thread1()\n";
  this->PrintIndent();
  body_os << "{\n";
  // generate reset code
  int reset_scope_outer = this->BeginScope();
  this->PrintIndent();
  body_os << "{\n";
  int reset_scope_inner = this->BeginScope();
  this->PrintIndent();
  body_os << "HLS_DEFINE_PROTOCOL(\"reset\");\n";
  for (auto it = _port_names[level_].begin(); it != _port_names[level_].end();
       ++it) {
    if (!IsP2P(*it)) continue;
    this->PrintIndent();
    body_os << *it << '.' << "reset();\n";
  }
  if (top_level) {
    this->PrintIndent();
    body_os << "finish.write(0);\n";
  }
  this->PrintIndent();
  body_os << "wait();\n";
  this->EndScope(reset_scope_inner);
  this->PrintIndent();
  body_os << "}\n";
  this->EndScope(reset_scope_outer);
  // generate function body
  int thread_scope = this->BeginScope();
  this->PrintIndent();
  body_os << "while( true ) \n";
  this->PrintIndent();
  body_os << "{\n";

  int while_scope = this->BeginScope();
  range_ = CollectIterRange(body);
  this->PrintStmt(body);  // print function body

  if (top_level) {
    this->PrintIndent();
    body_os << "finish.write(true);\n";
  }

  this->EndScope(while_scope);
  this->PrintIndent();
  body_os << "}\n";  // while true end scope
  this->EndScope(thread_scope);
  this->PrintIndent();
  body_os << "}\n";  // thread func end scope

  // Finished SC_THREAD function implementation
  this->EndScopeCtor(ctor_scope_inner);  // constructor end scope
  PrintIndentAnyStream(ctor_os, c_indent_);
  ctor_os << "}\n\n";
  this->EndScopeCtor(ctor_scope);

  // declare thread function
  decl_os << "\n";
  PrintIndentAnyStream(decl_os, h_indent_);
  decl_os << "void thread1();\n";

  ctor_os << "};\n\n";
  this->EndScopeHeader(module_scope);  // module declaration end scope
}

void CodeGenStratusHLS::AddFunction(
    LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type) {
  // clear previous generated state.
  this->InitFuncState(f);
  // add to alloc buffer type.
  for (const auto& kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  // Note: Var is a child class of VarExpr
  Array<VarExpr> args;
  for (size_t i = 0; i < f->args.size(); ++i) {
    VarExpr v = f->args[i];  // implicit upcast
    args.push_back(v);
    // allocate var id
    // AllocVarID(v.get());
  }

  // build arg shapes
  Array<Array<Expr>> arg_shapes;
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    const BufferNode* buf = f->api_args[i].as<BufferNode>();
    if (v.type().is_handle() && buf) {
      arg_shapes.push_back(buf->shape);
      // update var_shape_map
      var_shape_map_[buf->data.get()] = buf->shape;
    } else {
      Array<Expr> shape;
      Expr expr(1);
      shape.push_back(expr);
      arg_shapes.push_back(shape);
    }
  }

  Array<Expr> dummy_arg_types;

  GenerateModule(f->name, true, map_arg_type, dummy_arg_types, arg_shapes,
                 f->body, args, this->decl_stream, this->ctor_stream,
                 this->stream);
  printTclFile();
}

void CodeGenStratusHLS::PrintType(Type t, std::ostream& os) {
  PrintType(t, os, true);
}

void CodeGenStratusHLS::PrintType(Type t, std::ostream& os, bool is_index) {
  bool big = t.bits() > 64;
  if (big && is_index) {
    LOG(WARNING) << "index expression doesn't support bitwidth wider than "
                 << " 64 bit.";
    return;
  }
  if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed()) {
    if (t.is_uint()) {
      if (big)
        os << "sc_biguint<" << t.bits() << ">";
      else
        os << "sc_uint<" << t.bits() << ">";
    } else if (t.is_int()) {
      if (big)
        os << "sc_bigint<" << t.bits() << ">";
      else
        os << "sc_int<" << t.bits() << ">";
    } else if (t.is_ufixed()) {
      os << "sc_ufixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
    } else {
      os << "sc_fixed<" << t.bits() << ", " << t.bits() - t.fracs() << ">";
    }
  } else {
    CodeGenC::PrintType(t, os);
  }
}

void CodeGenStratusHLS::PrintTypeStringImm(const StringImm* t,
                                           std::ostream& os) {
  if (t->value.find("int") != std::string::npos) {
    os << "sc_int<" << t->value.substr(3, std::string::npos) << ">";
  } else if (t->value.find("uint") != std::string::npos) {
    os << "sc_uint<" << t->value.substr(3, std::string::npos) << ">";
  }
}

void CodeGenStratusHLS::VisitStmt_(const For* op) {
  std::ostringstream os;
  GenForStmt(op, os.str(), false);
}

void CodeGenStratusHLS::GenForStmt(const For* op, std::string pragma,
                                   bool before) {
  std::string extent = PrintExpr(op->extent);
  std::string vid = AllocVarID(op->loop_var.get());
  CHECK(is_zero(op->min));
  if (before && pragma.length() > 0) {
    PrintIndent();
    stream << pragma;
  }
  PrintIndent();
  // print loop labels
  std::string loop_name;
  bool loop_stage_name = false;
  for (unsigned int i = 0; i < op->annotate_keys.size(); i++) {
    if (auto str = op->annotate_keys[i].as<StringImm>()) {
      if (str->value == "stage_name") {
        loop_stage_name = true;
        auto label = op->annotate_values[i].as<StringImm>();
        if (label->value == "") {
          stream << vid;
          loop_name = vid;
        } else {
          stream << label->value << "_" << vid;
          loop_name = label->value + "_" + vid;
        }
        stream << ": ";
        break;
      }
    }
  }
  if (!loop_stage_name) stream << vid << ": ";
  stream << "for (";
  PrintType(op->loop_var.type(), stream);
  stream << ' ' << vid << " = 0; " << vid << " < " << extent << "; ++" << vid
         << ") {\n";
  if (!before && pragma.length() > 0) {
    PrintIndent();
    stream << pragma;
  }
  int for_scope = BeginScope();

  // Stratus HLS's loop transformations are annotated in loop body
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
    stream << "HLS_UNROLL_LOOP(COMPLETE, " << unroll_factor << "\"" << loop_name
           << "\");\n";
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
    stream << "HLS_PIPELINE_LOOP(HARD_STALL, ";
    if (II > 0) stream << II;
    stream << ", \"" << loop_name << "\");\n";
  }
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

bool CodeGenStratusHLS::IsP2P(const std::string& vid) {
  bool is_p2p = false;
  auto it =
      std::find(_port_names[level_].begin(), _port_names[level_].end(), vid);
  if (it != _port_names[level_].end()) {
    if (this->_port_type[vid].compare("p2p") == 0) {
      is_p2p = true;
    }
  }
  return is_p2p;
}

void CodeGenStratusHLS::VisitExpr_(const Load* op, std::ostream& os) {
  std::string vid = GetVarID(op->buffer_var.get());
  if (IsP2P(vid)) {
    PrintIndent();
    os << vid << ".get()";
  } else {
    // call codegen C's load node visitor
    CodeGenC::VisitExpr_(op, os);
  }
}

std::string CodeGenStratusHLS::CastFromTo(std::string value, Type from,
                                          Type target) {
  if (from == target) return value;
  std::ostringstream os;
  if (target.bits() > 64) {
    os << value;
  } else {
    os << "((";
    this->PrintType(target, os);
    os << ")" << value << ")";
  }
  return os.str();
}

void CodeGenStratusHLS::VisitExpr_(const Cast* op, std::ostream& os) {
  std::stringstream value;
  this->PrintExpr(op->value, value);

  // check if Cast node's value is a P2P port variable
  if (const Variable* v = op->value.as<Variable>()) {
    if (IsP2P(v->name_hint)) {
      value << ".get()";
    }
  }
  os << CastFromTo(value.str(), op->value.type(), op->type);
}

void CodeGenStratusHLS::VisitStmt_(const Store* op) {
  std::string vid = GetVarID(op->buffer_var.get());
  std::string index = PrintExpr(op->index);
  std::string value = PrintExpr(op->value);
  // decide if variable is a p2p port
  bool left_isp2p = IsP2P(vid);
  bool right_isp2p = IsP2P(value);
  if (right_isp2p) {
    value = value + ".get()";
  }
  PrintIndent();
  // ref is a multi-dimensional reference to the target element.
  // e.g. A[32][2][0]
  std::string ref =
      this->GetBufferRef(op->value.type(), op->buffer_var.get(), op->index);
  if (left_isp2p) {
    stream << vid << ".put(" << value << ");\n";
  } else {
    if (const SetSlice* ss = op->value.as<SetSlice>()) {
      Expr new_index_left = ir::Simplify(ss->index_left - 1);
      stream << ref << "(" << PrintExpr(new_index_left) << ", "
             << PrintExpr(ss->index_right) << ") = " << PrintExpr(ss->value)
             << ";\n";
    } else if (const SetBit* sb = op->value.as<SetBit>()) {
      stream << ref << "[" << PrintExpr(sb->index)
             << "] = " << PrintExpr(sb->value) << ";\n";
    } else {
      stream << ref << " = " << value << ";\n";
    }
  }
}

std::string CodeGenStratusHLS::GetBufferRef(Type t, const Variable* buffer,
                                            Expr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  // check if variable is p2p port
  bool is_p2p = IsP2P(vid);
  bool is_scalar =
      (buf_length_map_.count(buffer) == 1 && buf_length_map_[buffer] == 1);
  if (is_scalar && !is_p2p) {
    os << vid;
  } else {
    os << vid;
    CHECK(var_shape_map_.count(buffer))
        << "[SystemC Backend][GetBufferRef] buffer " << buffer->name_hint
        << " not found in var_shape_map";
    if (is_p2p) {
      os << ".get()";
    } else {
      // support multi-dimensional array access
      std::vector<Expr> indices =
          ExtractIndices(index, var_shape_map_[buffer], range_);
      for (size_t i = 0; i < indices.size(); i++) {
        os << '[';
        PrintExpr(indices[i], os);
        os << ']';
      }
    }
  }
  return os.str();
}

void CodeGenStratusHLS::VisitStmt_(const Allocate* op) {
  CHECK(!is_zero(op->condition));
  if (!op->is_const) {
    std::string vid = AllocVarID(op->buffer_var.get());
    if (op->new_expr.defined()) {
      LOG(ERROR) << "SystemC does not support malloc/free allocation";
    } else {
      int32_t constant_size = op->constant_allocation_size();
      CHECK_GT(constant_size, 0)
          << "Can only handle constant size stack allocation for now";
      const Variable* buffer = op->buffer_var.as<Variable>();
      var_shape_map_[buffer] = op->extents;

      std::string scope;  // allocate on local scope by default
      auto it = alloc_storage_scope_.find(buffer);
      if (it != alloc_storage_scope_.end())
        scope = alloc_storage_scope_.at(buffer);
      else
        scope = "local";

      // determine if the variable has been allocated
      bool not_alloc = false;
      if (vid.find("_new") != std::string::npos) {
        vid.replace(vid.find("_new"), 4, "");
        var_idmap_[op->buffer_var.get()] = vid;
      }
      if (alloc_set_.find(vid) != alloc_set_.end()) not_alloc = true;
      // avoid printing unused module allocation nodes
      auto it_name = std::find(sub_names.begin(), sub_names.end(), vid);
      if (it_name != sub_names.end()) not_alloc = true;

      // not allocated buffer for channel or moved data
      if (!not_alloc) {
        alloc_set_.insert(vid);
        this->PrintIndentHeader();

        if (constant_size > 1) {  // Transfer length one array to scalar
          if (vid.find("_reuse") != std::string::npos) {
            PrintType(op->type, this->decl_stream, false);
            this->decl_stream << ' ' << vid;
            for (size_t i = 0; i < op->extents.size(); i++) {
              this->decl_stream << '[';
              PrintExpr(op->extents[i], this->decl_stream);
              this->decl_stream << "]";
            }
          } else {
            PrintType(op->type, this->decl_stream, false);
            this->decl_stream << ' ' << vid;
            for (size_t i = 0; i < op->extents.size(); i++) {
              this->decl_stream << '[';
              PrintExpr(op->extents[i], this->decl_stream);
              this->decl_stream << "]";
            }
          }
        } else {  // allocate scalar
          PrintType(op->type, this->decl_stream, false);
          this->decl_stream << ' ' << vid;
        }
        this->decl_stream << ";\n";
        for (size_t i = 0; i < op->attrs.size(); i++)
          this->PrintStmt(op->attrs[i]);
      }
      buf_length_map_[buffer] = constant_size;
    }
  }
  RegisterHandleType(op->buffer_var.get(), op->type);
  this->PrintStmt(op->body);
}

void CodeGenStratusHLS::VisitStmt_(const Partition* op) {
  // Array Partition node
  // For Stratus HLS, array mapping directives need to be
  // declared in the constructor.
  PrintIndentCtor();
  std::string vid = GetVarID(op->buffer_var.get());

  switch (op->partition_type) {
    case PartitionType::Complete:
      ctor_stream << "HLS_FLATTEN_ARRAY(";
      break;
    case PartitionType::Block:
      ctor_stream << "HLS_MAP_TO_REG_BANK(";
      break;
    case PartitionType::Cyclic:
      ctor_stream << "HLS_MAP_TO_REG_BANK(";
      break;
  }

  ctor_stream << vid;
  ctor_stream << ");\n";
}

// print indent for SC_MODULE
void CodeGenStratusHLS::PrintIndentHeader() {
  for (int i = 0; i < h_indent_; ++i) {
    this->decl_stream << ' ';
  }
}
// print indent for SC_CTOR
void CodeGenStratusHLS::PrintIndentCtor() {
  for (int i = 0; i < c_indent_; ++i) {
    this->ctor_stream << ' ';
  }
}

void CodeGenStratusHLS::PrintIndentAnyStream(std::ostream& os, int indent) {
  for (int i = 0; i < indent; ++i) {
    os << ' ';
  }
}

// increase indent for SC_MODULE
int CodeGenStratusHLS::BeginScopeHeader() {
  int sid = static_cast<int>(h_scope_mark_.size());
  h_scope_mark_.push_back(true);
  h_indent_ += 2;
  return sid;
}

// decrease indent for SC_MODULE
void CodeGenStratusHLS::EndScopeHeader(int scope_id) {
  h_scope_mark_[scope_id] = false;
  h_indent_ -= 2;
}

// increase indent for SC_CTOR
int CodeGenStratusHLS::BeginScopeCtor() {
  int sid = static_cast<int>(c_scope_mark_.size());
  c_scope_mark_.push_back(true);
  c_indent_ += 2;
  return sid;
}

// decrease indent for SC_CTOR
void CodeGenStratusHLS::EndScopeCtor(int scope_id) {
  c_scope_mark_[scope_id] = false;
  c_indent_ -= 2;
}

void CodeGenStratusHLS::VisitStmt_(const KernelDef* op) {
  // Submodule Definition Node
  std::ostringstream sub_stream, sub_decl_stream, sub_ctor_stream, tmp_stream;
  // stash this->stream
  tmp_stream << this->stream.str();
  this->stream.str("");
  this->stream.clear();
  // stash indents
  int ctor_indent_stash = this->c_indent_;
  this->c_indent_ = 0;
  int decl_indent_stash = this->h_indent_;
  this->h_indent_ = 0;
  int indent_stash = GetIndent();
  SetIndent(0);

  // Update var_shape_map
  for (size_t i = 0; i < op->args.size(); ++i) {
    VarExpr v = op->args[i];
    var_shape_map_[v.get()] = op->arg_shapes[i];
  }

  str2tupleMap<std::string, Type> map_arg_type;

  GenerateModule(op->name, false, map_arg_type, op->arg_types, op->arg_shapes,
                 op->body, op->args, sub_decl_stream, sub_ctor_stream,
                 this->stream);

  // save submodule's name
  this->sub_names.push_back(op->name);

  // Put this->stream content to sub_stream
  sub_stream << this->stream.str();
  // Restore this->stream from stash
  this->stream.str("");
  this->stream.clear();
  this->stream << tmp_stream.str();

  // reset indentation
  this->c_indent_ = ctor_indent_stash;
  this->h_indent_ = decl_indent_stash;
  SetIndent(indent_stash);

  this->sub_ctors.push_back(sub_ctor_stream.str());
  this->sub_decls.push_back(sub_decl_stream.str());
  this->sub_threads.push_back(sub_stream.str());
}

void CodeGenStratusHLS::UpdateSubPortTypes(
    const std::map<std::string, std::vector<Expr>> arg_types) {
  for (auto pair : arg_types) {
    std::string module_name = std::get<0>(pair);
    if (this->_sub_port_types.count(module_name) > 0) continue;
    std::vector<std::string> type_strs;
    for (Expr type : std::get<1>(pair)) {
      std::ostringstream ss;
      PrintTypeStringImm(type.as<StringImm>(), ss);
      std::string type_str = ss.str();
      type_strs.push_back(type_str);
    }
    this->_sub_port_types.insert(
        std::pair<std::string, std::vector<std::string>>(module_name,
                                                         type_strs));
  }
}

void CodeGenStratusHLS::UpdateSubPortNames(
    const std::map<std::string, std::vector<std::string>> arg_names) {
  for (auto pair : arg_names) {
    std::string module_name = std::get<0>(pair);
    bool exists = _sub_port_names.count(module_name) > 0;
    if (exists) continue;
    for (std::string arg_name : std::get<1>(pair)) {
      _sub_port_names[module_name].push_back(arg_name);
    }
  }
}

void CodeGenStratusHLS::VisitExpr_(const KernelExpr* op, std::ostream& os) {
  // Function Call Node
  os << op->name << "(";
  for (size_t i = 0; i < op->args.size(); ++i) {
    PrintExpr(op->args[i], os);
    if (i != op->args.size() - 1) os << ", ";
  }
  os << ")";
}

void CodeGenStratusHLS::VisitStmt_(const KernelStmt* op) {
  PrintIndent();
  stream << op->name << "(";
  for (size_t i = 0; i < op->args.size(); i++) {
    PrintExpr(op->args[i], stream);
    if (i < op->args.size() - 1) stream << ", ";
  }
  stream << ");\n";
}

void CodeGenStratusHLS::VisitStmt_(const Return* op) {
  PrintIndent();
  this->stream << "return ";
  PrintExpr(op->value, stream);
  this->stream << ";\n";
}

void CodeGenStratusHLS::VisitStmt_(const Assert* op) {
  PrintIndent();
  this->stream << "assert " << op->condition << ";\n";
}

void CodeGenStratusHLS::VisitExpr_(const SetSlice* op, std::ostream& os) {
  // Note: SetSlice is implemented in Store node.
}

void CodeGenStratusHLS::VisitExpr_(const SetBit* op, std::ostream& os) {
  // Note: SetSlice is implemented in Store node.
}

std::string CodeGenStratusHLS::Finish() {
  // top-level module
  std::string finalstr = "[filename] " + _top_name + ".h\n";
  finalstr.append(decl_stream.str() + ctor_stream.str());
  finalstr.append("[filename] " + _top_name + ".cc\n");
  finalstr.append(stream.str());
  for (unsigned i = 0; i < this->sub_ctors.size(); i++) {
    finalstr.append("[filename] " + sub_names[i] + ".h\n");
    finalstr.append(sub_decls[i]);
    finalstr.append(sub_ctors[i]);
    finalstr.append("[filename] " + sub_names[i] + ".cc\n");
    finalstr.append(sub_threads[i]);
  }
  for (unsigned i = 0; i < this->support_fnames.size(); i++) {
    finalstr.append("[filename] " + support_fnames[i] + "\n");
    finalstr.append(support_files[i]);
  }
  return finalstr;
}

std::string CodeGenStratusHLS::GetHost() {
  std::string hoststr = "// test bench";
  return hoststr;
}

std::string CodeGenStratusHLS::GetDevice() { return Finish(); }

inline void PrintConst(const IntImm* op, std::ostream& os,
                       CodeGenStratusHLS* p) {  // NOLINT(*)
  bool big = op->type.bits() > 64;
  if (op->type == Int(32)) {
    std::ostringstream temp;
    temp << op->value;
    os << temp.str();
  } else {
    if (!big) {
      os << "(";
      p->PrintType(op->type, os);
      os << ")";
    }
    os << op->value;
  }
}

inline void PrintConst(const UIntImm* op, std::ostream& os,
                       CodeGenStratusHLS* p) {  // NOLINT(*)
  bool big = op->type.bits() > 64;
  if (op->type == UInt(32)) {
    std::ostringstream temp;
    temp << op->value << "U";
    os << temp.str();
  } else {
    if (!big) {
      os << "(";
      p->PrintType(op->type, os);
      os << ")";
    }
    os << op->value;
  }
}

void CodeGenStratusHLS::VisitExpr_(const IntImm* op, std::ostream& os) {
  PrintConst(op, os, this);
}

void CodeGenStratusHLS::VisitExpr_(const UIntImm* op, std::ostream& os) {
  PrintConst(op, os, this);
}

}  // namespace codegen
}  // namespace TVM
