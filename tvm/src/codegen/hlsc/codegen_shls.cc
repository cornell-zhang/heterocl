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
      //TODO: what do we do when type is not found
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

  // generate constructor
  this->decl_stream << "\n";
  this->PrintIndentHeader();
  this->decl_stream << "SC_CTOR( " << f->name << " ) \n";
  // initialize clock and reset
  this->PrintIndentHeader();
  this->decl_stream << ": " << "clk( " << "\"clk\"" << " )\n";
  this->PrintIndentHeader();
  this->decl_stream << ", " << "rst( " << "\"rst\"" << " )\n";
  // initialize i/o ports
  for (auto it = _port_names.begin(); it != _port_names.end(); ++it) {
    std::string name = *it;
    this->PrintIndentHeader();
    this->decl_stream << ", " << name << "( \"" << name << "\" )\n";
  }
  this->PrintIndentHeader();
  this->decl_stream << "{\n";
  // initlialize clocked thread
  int ctor_scope = this->BeginScopeHeader();
  this->PrintIndentHeader();
  this->decl_stream << "SC_CTHREAD( thread1, clk.pos() );\n";
  // setup reset signal
  this->PrintIndentHeader();
  this->decl_stream << "reset_signal_is( rst, 0 );\n";
  //connect clk and rst power to modular interface ports
  for (auto it = _port_names.begin(); it != _port_names.end(); ++it) {
    std::string name = *it;
    this->PrintIndentHeader();
    this->decl_stream << name << '.' << "clk_rst( clk, rst );\n";
  }
  this->EndScopeHeader(ctor_scope);
  this->PrintIndentHeader();
  this->decl_stream << "}\n\n";

  // declare thread function
  this->PrintIndentHeader();
  this->decl_stream << "void thread1();\n";



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
  this->PrintStmt(f->body); // print function body
  this->EndScope(func_body_scope);
  this->PrintIndent();
  this->stream << "}\n"; // whiile true end scope
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n"; // thread func end scope


  this->decl_stream << "};\n\n";
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


void CodeGenStratusHLS::VisitStmt_(const Partition* op) {
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



// std::string CodeGenStratusHLS::Finish(){
//   return decl_stream.str() + stream.str() + thread_stream.str();
// }

void CodeGenStratusHLS::PrintIndentHeader(){
  for (int i = 0; i < h_indent_; ++i) {
    this->decl_stream << ' ';
  }
}

int CodeGenStratusHLS::BeginScopeHeader() {
  int sid = static_cast<int>(h_scope_mark_.size());
  h_scope_mark_.push_back(true);
  h_indent_ += 2;
  return sid;
}

void CodeGenStratusHLS::EndScopeHeader(int scope_id) {
  h_scope_mark_[scope_id] = false;
  h_indent_ -= 2;
}





} // namespace codegen
} // namespace TVM