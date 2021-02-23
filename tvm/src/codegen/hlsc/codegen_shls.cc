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
  this->stream << "SC_MODULE(" << f->name << ") \n{\n";
  // we fix the clock and reset for now
  int module_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "sc_in<bool> clk;\n";
  this->PrintIndent();
  this->stream << "sc_in<bool> rst;\n\n";
  // generate ports
  std::list<std::string> port_names;

  // map_arg_type
  // keys = "arg0", "arg1", "arg2"
  // values = ("A", "int32"), ("B", "int32"), ("C", "int32")

  for (auto it = map_arg_type.begin(); it != map_arg_type.end(); it++) {
    port_names.push_back(std::get<0>(it->second));
  }

  // Infer port direction
  PortDirection visitor(port_names);
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
      
      this->PrintIndent();
      this->stream << "cynw_p2p < ";
      PrintType(std::get<1>(arg), this->stream);
      this->stream << " >";
      std::string port_name = std::get<0>(arg);
      std::string port_direction = visitor.get_direction(arg_name);
      this->_is_inport.insert(std::pair<std::string, bool>(port_name, visitor.is_inport(port_name)));

      this->stream << "::" << port_direction << "\t";
      this->stream << std::get<0>(arg); // print arg name
      this->stream << ";\n";
      // allocate storage
      const BufferNode* buf = f->api_args[i].as<BufferNode>();
      if (v.type().is_handle() && buf) {
        var_shape_map_[buf->data.get()] = buf->shape;
      }
    }
  }

  // generate constructor
  this->stream << "\n";
  this->PrintIndent();
  this->stream << "SC_CTOR( " << f->name << " ) \n";
  // initialize clock and reset
  this->PrintIndent();
  this->stream << ": " << "clk( " << "\"clk\"" << " )\n";
  this->PrintIndent();
  this->stream << ", " << "rst( " << "\"rst\"" << " )\n";
  // initialize i/o ports
  for (auto it = port_names.begin(); it != port_names.end(); ++it) {
    std::string name = *it;
    this->PrintIndent();
    this->stream << ", " << name << "( \"" << name << "\" )\n";
  }
  this->PrintIndent();
  this->stream << "{\n";
  // initlialize clocked thread
  int ctor_scope = this->BeginScope();
  this->PrintIndent();
  this->stream << "SC_CTHREAD( thread1, clk.pos() );\n";
  // setup reset signal
  this->PrintIndent();
  this->stream << "reset_signal_is( rst, 0 );\n";
  //connect clk and rst power to modular interface ports
  for (auto it = port_names.begin(); it != port_names.end(); ++it) {
    std::string name = *it;
    this->PrintIndent();
    this->stream << name << '.' << "clk_rst( clk, rst );\n";
  }
  this->EndScope(ctor_scope);
  this->PrintIndent();
  this->stream << "}\n\n";

  // generate process function
  this->PrintIndent();
  this->stream << "void thread1()\n";
  this->PrintIndent();
  this->stream << "{\n";
  // generate reset code
  int reset_scope_outer = this->BeginScope();
  this->PrintIndent();
  this->stream << "{\n";
  int reset_scope_inner = this->BeginScope();
  this->PrintIndent();
  this->stream << "HLS_DEFINE_PROTOCOL(\"reset\");\n";
  for (auto it = port_names.begin(); it != port_names.end(); ++it) {
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
  this->PrintStmt(f->body);
  this->EndScope(func_body_scope);
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n";
  this->stream << "};\n\n";
  this->EndScope(module_scope);

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
  Type t = op->value.type();
  if (t.lanes() == 1) {
    std::string value = this->PrintExpr(op->value);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();
    stream << ref << ".put(" << value << ");\n";
  } else {
    CHECK(is_one(op->predicate)) << "Predicated store is not supported";
    Expr base;
    if (TryGetRamp1Base(op->index, t.lanes(), &base)) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer_var.get(), t, base, value);
    } else {
      // The assignment below introduces side-effect, and the resulting value
      // cannot be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements seperately
      std::string index = SSAGetID(PrintExpr(op->index), op->index.type());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.type());
      std::string vid = GetVarID(op->buffer_var.get());
      for (int i = 0; i < t.lanes(); ++i) {
        // TODO: modify vector store to .put()
        this->PrintIndent();
        stream << vid;
        stream << '[';
        PrintVecElemLoad(index, op->index.type(), i, stream);
        stream << "] = ";
        PrintVecElemLoad(value, op->value.type(), i, stream);
        stream << ";\n";
      }
      EndScope(vec_scope);
    }
  }
}

void CodeGenStratusHLS::VisitExpr_(const Load* op, std::ostream& os){
  int lanes = op->type.lanes();
  // delcare type.
  if (op->type.lanes() == 1) {
    std::string ref = GetBufferRef(op->type, op->buffer_var.get(), op->index);
    os << ref;
  } else {
    CHECK(is_one(op->predicate)) << "predicated load is not supported";
    Expr base;
    if (TryGetRamp1Base(op->index, op->type.lanes(), &base)) {
      std::string ref = GetVecLoad(op->type, op->buffer_var.get(), base);
      os << ref;
    } else {
      // The assignment below introduces side-effect, and the resulting value
      // cannot be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // load seperately.
      std::string svalue = GetUniqueName("_");
      this->PrintIndent();
      this->PrintType(op->type, stream);
      stream << ' ' << svalue << ";\n";
      std::string sindex = SSAGetID(PrintExpr(op->index), op->index.type());
      std::string vid = GetVarID(op->buffer_var.get());
      Type elem_type = op->type.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (op->buffer_var.get()->type.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
              value_temp << ' ';
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, op->index.type(), i, value_temp);
        value_temp << ']';
        PrintVecElemStore(svalue, op->type, i, value_temp.str());
      }
      os << svalue;
      EndScope(vec_scope);
    }
  }
}

void CodeGenStratusHLS::PrintVecStore(const Variable* buffer,
                             Type t, Expr base,
                             const std::string& value) {
  std::string ref = GetBufferRef(t, buffer, base);
  this->PrintIndent();
  stream << ref << ".put(" << value << ");\n";
}

std::string CodeGenStratusHLS::GetBufferRef(Type t, const Variable* buffer, Expr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
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
      if (is_inport){
        os << ".get()";
      }
    }
  }
  return os.str();
}


} // namespace codegen
} // namespace TVM