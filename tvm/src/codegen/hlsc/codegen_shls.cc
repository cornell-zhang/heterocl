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
  this->stream << "};\n\n";
  this->EndScope(module_scope);

  }


} // namespace codegen
} // namespace TVM