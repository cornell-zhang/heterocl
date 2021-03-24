/*!
 * Copyright (c) 2021 by Contributors
 * \file codegen_shls.cc
 */
#include "codegen_shls.h"
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
#include "./port_direction.h"
#include "./hierarchy.h"

namespace TVM {
namespace codegen {

void CodeGenStratusHLS::AddFunction(
    LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type) {

  // write header files
  this->decl_stream << "#include <cynw_p2p.h>\n";
  
  // clear previous generated state.
  this->InitFuncState(f);
  // add to alloc buffer type.
  for (const auto& kv : f->handle_data_type) {
    RegisterHandleType(kv.first.get(), kv.second.type());
  }

  // generate SC MODULE
  this->decl_stream << "SC_MODULE(" << f->name << ") \n{\n";
  // we fix the clock and reset for now
  int module_scope = this->BeginScopeHeader();
  this->PrintIndentHeader();
  this->decl_stream << "sc_in<bool> clk;\n";
  this->PrintIndentHeader();
  this->decl_stream << "sc_in<bool> rst;\n\n";

  // map_arg_type
  // keys = "arg0", "arg1", "arg2"
  // values = ("A", "int32"), ("B", "int32"), ("C", "int32")

  for (auto it = map_arg_type.begin(); it != map_arg_type.end(); it++) {
    _port_names.push_back(std::get<0>(it->second));
  }

  // Infer port direction
  PortDirection visitor(_port_names);
  visitor.Visit(f->body);
  
  for (size_t i = 0; i < f->args.size(); ++i) {
    Var v = f->args[i];
    std::string vid = AllocVarID(v.get());
    // check type in the arg map
    if (map_arg_type.find(vid) == map_arg_type.end()) {
      LOG(WARNING) << vid << " type not found\n";
    } else {
      auto arg = map_arg_type[vid];
      std::string arg_name = std::get<0>(arg);
      
      this->PrintIndentHeader();
      this->decl_stream << "cynw_p2p < ";
      PrintType(std::get<1>(arg), this->decl_stream);
      this->decl_stream << " >";
      std::string port_name = std::get<0>(arg);
      std::string port_direction = visitor.get_direction(arg_name);
      this->_is_inport.insert(std::pair<std::string, bool>(port_name, visitor.is_inport(port_name)));

      this->decl_stream << "::" << port_direction << "\t";
      this->decl_stream << std::get<0>(arg); // print arg name
      this->decl_stream << ";\n";
      // allocate storage
      const BufferNode* buf = f->api_args[i].as<BufferNode>();
      if (v.type().is_handle() && buf) {
        var_shape_map_[buf->data.get()] = buf->shape;
      }
    }
  }
  this->decl_stream << "\n";

  // generate constructor
  int ctor_scope = this->BeginScopeCtor();
  this->ctor_stream << "\n";
  this->PrintIndentCtor();
  this->ctor_stream << "SC_CTOR( " << f->name << " ) \n";
  // initialize clock and reset
  this->PrintIndentCtor();
  this->ctor_stream << ": " << "clk( " << "\"clk\"" << " )\n";
  this->PrintIndentCtor();
  this->ctor_stream << ", " << "rst( " << "\"rst\"" << " )\n";
  // initialize i/o ports
  for (auto it = _port_names.begin(); it != _port_names.end(); ++it) {
    std::string name = *it;
    this->PrintIndentCtor();
    this->ctor_stream << ", " << name << "( \"" << name << "\" )\n";
  }
  this->PrintIndentCtor();
  this->ctor_stream << "{\n";
  // initlialize clocked thread
  int ctor_scope_inner = this->BeginScopeCtor();
  this->PrintIndentCtor();
  this->ctor_stream << "SC_CTHREAD( thread1, clk.pos() );\n";
  // setup reset signal
  this->PrintIndentCtor();
  this->ctor_stream << "reset_signal_is( rst, 0 );\n";
  //connect clk and rst power to modular interface ports
  for (auto it = _port_names.begin(); it != _port_names.end(); ++it) {
    std::string name = *it;
    this->PrintIndentCtor();
    this->ctor_stream << name << '.' << "clk_rst( clk, rst );\n";
  }




  /* ---------------- dut.cc -------------------------*/
  // generate process function
  this->PrintIndent();
  this->stream << "void " << f->name << "::thread1()\n";
  this->PrintIndent();
  this->stream << "{\n";
  // generate reset code
  int reset_scope_outer = this->BeginScope();
  this->PrintIndent();
  this->stream << "{\n";
  int reset_scope_inner = this->BeginScope();
  this->PrintIndent();
  this->stream << "HLS_DEFINE_PROTOCOL(\"reset\");\n";
  for (auto it = _port_names.begin(); it != _port_names.end(); ++it) {
    this->PrintIndent();
    this->stream << *it << '.' << "reset();\n";
  }
  this->PrintIndent();
  this->stream << "wait();\n"; 
  this->EndScope(reset_scope_inner);
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(reset_scope_outer);
  // generate function body
  int func_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "while( true ) \n";
  this->PrintIndent();
  this->stream << "{\n";
  
  int func_body_scope = this->BeginScope();
  range_ = CollectIterRange(f->body);
  LOG(INFO) << "start visiting LoweredFunc's body";
  this->PrintStmt(f->body); // print function body
  LOG(INFO) << "Finish visiting LoweredFunc's body";
  this->EndScope(func_body_scope);
  this->PrintIndent();
  this->stream << "}\n"; // while true end scope
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n"; // thread func end scope

  /* ---------------------- dut.h continued -----------------------*/
  this->EndScopeCtor(ctor_scope_inner); // constructor end scope
  this->PrintIndentCtor();
  this->ctor_stream << "}\n\n";
  this->EndScopeCtor(ctor_scope);

  // declare thread function
  this->decl_stream << "\n";
  this->PrintIndentHeader();
  this->decl_stream << "void thread1();\n";

  this->ctor_stream << "};\n\n";
  this->EndScopeHeader(module_scope); // module declaration end scope




}



void CodeGenStratusHLS::PrintType(Type t, std::ostream& os) {
  if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed()) {
    if (t.is_uint()) {
      os << "sc_uint<" << t.bits() << ">";
    } else if (t.is_int()) {
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



void CodeGenStratusHLS::VisitStmt_(const For* op) {
  std::ostringstream os;
  
  GenForStmt(op, os.str(), false);
}

inline bool TryGetRamp1Base(Expr index, int lanes, Expr* base) {
  const Ramp* r = index.as<Ramp>();
  if (!r) return false;
  if (!is_one(r->stride)) return false;
  CHECK_EQ(r->lanes, lanes);
  *base = r->base;
  return true;
}

void CodeGenStratusHLS::VisitStmt_(const Store* op) {
  LOG(INFO) << "in Store, op->name is: " << op->buffer_var->name_hint;
  std::string vid = GetVarID(op->buffer_var.get());
  std::string index = SSAGetID(PrintExpr(op->index), op->index.type());
  std::string value = SSAGetID(PrintExpr(op->value), op->value.type());
  auto it = std::find(_port_names.begin(), _port_names.end(), vid);
  bool is_port = (it!=_port_names.end());

  PrintIndent();
  if (is_port) {
    stream << vid << ".put(" << value << ");\n";
    return;
  } else {
    stream << vid << "[" << index << "] = " << value << ";\n";
    return;
  }

  //handle SetSlice
  // if (const SetSlice* ss = op->value.as<SetSlice>()) {
  //   stream << "in set slice branch";
  //   Type t = op->value.type();
  //   Expr new_index_left = ir::Simplify(ss->index_left - 1);
  //   std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
  //   std::string rhs = PrintExpr(ss->value);
  //   PrintIndent();
  //   this->stream << ref << "(" << PrintExpr(new_index_left) << ", "
  //                << PrintExpr(ss->index_right) << ") = " << rhs << ";\n";
  // } else if (const SetBit* sb = op->value.as<SetBit>()) {
  //   stream << "in set bit branch";
  //   Type t = op->value.type();
  //   std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
  //   PrintIndent();
  //   this->stream << ref << "[" << PrintExpr(sb->index)
  //                << "] = " << PrintExpr(sb->value) << ";\n";
  // } else {
  //   CodeGenC::VisitStmt_(op);
  // }
}


std::string CodeGenStratusHLS::GetBufferRef(Type t, const Variable* buffer, Expr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  // decide if variable is port
  auto it = std::find(_port_names.begin(), _port_names.end(), vid);
  bool is_port = (it!=_port_names.end());
  bool is_inport = this->_is_inport[vid];
  if (t.lanes() == 1) {
    bool is_scalar =
        (buf_length_map_.count(buffer) == 1 && buf_length_map_[buffer] == 1);
    if (is_scalar) {
      os << vid;
    } else {
      os << vid;
      CHECK(var_shape_map_.count(buffer))
          << "buffer " << buffer->name_hint << " not found in var_shape_map";
      if (is_port) {
        if (is_inport){
          os << ".get()";
        }
      } else {
        os << "[";
        PrintExpr(index, os);
        os << "]";
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
      

      // not allocated buffer for channel or moved data
      if (!not_alloc) {
        alloc_set_.insert(vid);
        this->PrintIndentHeader();

        if (constant_size > 1) {  // Transfer length one array to scalar
          if (vid.find("_reuse") != std::string::npos) {
            PrintType(op->type, this->decl_stream); // print array to header
            this->decl_stream << ' ' << vid;
            for (size_t i = 0; i < op->extents.size(); i++) {
              this->decl_stream << '[';
              PrintExpr(op->extents[i], this->decl_stream);
              this->decl_stream << "]";
            }
          } else {
            PrintType(op->type, this->decl_stream);
            this->decl_stream << ' ' << vid;
            for (size_t i = 0; i < op->extents.size(); i++) {
              this->decl_stream << '[';
              PrintExpr(op->extents[i], this->decl_stream);
              this->decl_stream << "]";
            }
          }
        } else { // allocate scalar
          PrintType(op->type, this->decl_stream);
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


/*
  Array Partition node

  For Stratus HLS, array mapping directives need to be
  declared in the constructor.
*/
void CodeGenStratusHLS::VisitStmt_(const Partition* op) {
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
  
  // stream << " dim=" << op->dim;
  //if (op->partition_type != PartitionType::Complete) {
  //  stream << " factor=" << op->factor;
  // }
  ctor_stream << vid;
  ctor_stream << ");\n";
}



std::string CodeGenStratusHLS::Finish(){
  std::string finalstr = decl_stream.str() + ctor_stream.str() + stream.str();
  for (int i = 0; i < this->sub_ctors.size(); i++) {
    finalstr.append("\n\n\n\n");
    finalstr.append(sub_decls[i]);
    finalstr.append(sub_ctors[i]);
    finalstr.append(sub_threads[i]);
  }
  return finalstr;
}

// print indent for SC_MODULE
void CodeGenStratusHLS::PrintIndentHeader(){
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

// print indent for custom ostringstream
void CodeGenStratusHLS::PrintIndentCustom(std::ostringstream* s, int indent) {
  for(int i = 0; i < indent; ++i) {
    *s << ' ';
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
  //c_indent_ = h_indent_ > c_indent_ ? h_indent_ : c_indent_;
  c_indent_ += 2;
  return sid;
}

// decrease indent for SC_CTOR
void CodeGenStratusHLS::EndScopeCtor(int scope_id) {
  c_scope_mark_[scope_id] = false;
  c_indent_ -= 2;
}


/* Module definition node
 */
void CodeGenStratusHLS::VisitStmt_(const KernelDef* op) {
  LOG(INFO) << "Visiting KernelDef";
  std::ostringstream sub_stream, sub_decl_stream, sub_ctor_stream, tmp_stream;
  // stash this->stream
  tmp_stream << this->stream.str();
  this->stream.str("");
  this->stream.clear();
  // stash constructor indent
  int ctor_indent_stash = this->c_indent_;
  this->c_indent_ = 0;
  int indent_stash = GetIndent();
  SetIndent(0);

  // generate SC_MODULE
  sub_decl_stream << "SC_MODULE(" << op->name << ") \n{\n";  
  //int submodule_scope = this->BeginScope();
  this->PrintIndentCustom(&sub_decl_stream, h_indent_);
  sub_decl_stream << "sc_in<bool> clk;\n";
  this->PrintIndentCustom(&sub_decl_stream, h_indent_);
  sub_decl_stream << "sc_in<bool> rst;\n\n";
  // print port
  for (size_t i = 0; i < op->args.size(); ++i) {
    VarExpr v = op->args[i];
    var_shape_map_[v.get()] = op->arg_shapes[i];
    std::string vid = AllocVarID(v.get());
    this->PrintIndentCustom(&sub_decl_stream, h_indent_); 
    sub_decl_stream << "cynw_p2p < ";
    LOG(INFO) << "KernelDef op arg: " << op->args[i] << " type: " << op->args[i];
    // op->arg_types[i].type() returns handle64. its type is handle64
    // op->arg_types[i].get() returns 0x7fa6b2c722c8
    //PrintType(op->arg_types[i], sub_decl_stream);
    sub_decl_stream << op->arg_types[i];
    sub_decl_stream << " >" << "::in\t" << vid << ";\n";
    // note: these variables are all input ports,
    // there are no output ports for KernelDef node
    // only return value. So we need to turn the return
    // expression into a port variable
  }
  // generate output port
  this->PrintIndentCustom(&sub_decl_stream, h_indent_);
  sub_decl_stream << "cynw_p2p < ";
  PrintType(op->ret_type, sub_decl_stream);
  sub_decl_stream << " >" << "::out\t" << "return_var" << " ;\n";

  // collect submodules
  Hierarchy hierarchy;
  hierarchy.Visit(op->body);
  std::list<std::string> submodules = hierarchy.get_submodules();
  std::map<std::string, std::list<Expr>> submodule_args = hierarchy.get_submodule_args();
  
  for (std::string submodule : submodules) {
    PrintIndentCustom(&sub_decl_stream, h_indent_);
    // submodule instantiation
    sub_decl_stream << submodule << "\t" << submodule << "_0;\n"; 
    // the input argument
    for (Expr arg : submodule_args[submodule]) {
      PrintIndentCustom(&sub_decl_stream, h_indent_);
      sub_decl_stream << "cynw_p2p < " << arg.type() << " >\t" << arg << ";\n";
    }
  }

  
  // generate constructor
  sub_ctor_stream << "\n";
  int ctor_out_scope = this->BeginScopeCtor();
  this->PrintIndentCustom(&sub_ctor_stream, c_indent_);
  sub_ctor_stream << "SC_CTOR( " << op->name << " ) \n";
  //intialize clock, reset
  this->PrintIndentCustom(&sub_ctor_stream, c_indent_);
  sub_ctor_stream << ": " << "clk( " << "\"clk\"" << " )\n";
  this->PrintIndentCustom(&sub_ctor_stream, c_indent_);
  sub_ctor_stream << ", " << "rst( " << "\"rst\"" << " )\n";
  this->PrintIndentCustom(&sub_ctor_stream, c_indent_);
  sub_ctor_stream << "{\n";
  // intialize clocked thread
  int ctor_scope = this->BeginScopeCtor();
  this->PrintIndentCustom(&sub_ctor_stream, c_indent_);
  sub_ctor_stream << "SC_CTHREAD( thread1, clk.pos() );\n";
  this->PrintIndentCustom(&sub_ctor_stream, c_indent_);
  sub_ctor_stream << "reset_signal_is(rst, 0);\n";
  // connect clk and rst power to modular interface ports
  this->EndScopeCtor(ctor_scope);


  /*-----------sub module .cc -----------*/
  this->PrintIndent();
  this->stream << "void " << op->name << "::thread1()\n";
  this->PrintIndent();
  this->stream << "{\n";
  // generate reset code
  int reset_scope_outer = this->BeginScope();
  this->PrintIndent();
  this->stream << "{\n";
  int reset_scope_inner = this->BeginScope();
  this->PrintIndent();
  this->stream << "HLS_DEFINE_PROTOCOL(\"reset\");\n";
  for (auto it = _port_names.begin(); it != _port_names.end(); ++it) {
    this->PrintIndent();
    this->stream << *it << '.' << "reset();\n";
  }
  this->PrintIndent();
  this->stream << "wait();\n"; 
  this->EndScope(reset_scope_inner);
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(reset_scope_outer);
  // generate function body
  int func_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "while( true ) \n";
  this->PrintIndent();
  this->stream << "{\n";
  // generate function body
  int func_body_scope = this->BeginScope();
  PrintStmt(op->body);
  this->EndScope(func_body_scope);
  this->PrintIndent();
  this->stream << "}\n"; // while true end scope
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n"; // thread func end scope

  /*------------------- header file continued ------------------*/
  sub_decl_stream << "\n";
  this->PrintIndentCustom(&sub_ctor_stream, c_indent_);
  sub_ctor_stream << "}\n\n";
  this->EndScopeCtor(ctor_out_scope);
  // declare thread function
  sub_decl_stream << "\n";
  this->PrintIndentCustom(&sub_decl_stream, h_indent_);
  sub_decl_stream << "void thread1();\n";

  sub_ctor_stream << "};\n\n";
  //this->EndScopeHeader(submodule_scope); // module declaration end scope



//-----------end -----------------------

  sub_stream << this->stream.str();
  this->stream.str("");
  this->stream.clear();
  this->stream << tmp_stream.str();
  // reset indentation
  this->c_indent_ = ctor_indent_stash;
  SetIndent(indent_stash);

  this->sub_ctors.push_back(sub_ctor_stream.str());
  this->sub_decls.push_back(sub_decl_stream.str());
  this->sub_threads.push_back(sub_stream.str());

}

/* Module call
*/
void CodeGenStratusHLS::VisitExpr_(const KernelExpr* op, std::ostream& os) {
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
  // can't really determine if the return is in 
  // top module or sub module
  PrintIndent();
  this->stream << "return ";
  // Note: you can check if it is returning a KernelExpr
  // this->stream << "THIS IS RETURN" << op->value.node_->type_key();
  PrintExpr(op->value, stream);
  this->stream << ";\n";
}

} // namespace codegen
} // namespace TVM