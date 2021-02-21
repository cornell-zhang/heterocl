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

  // Infer port direction
  PortDirection visitor;
  visitor.Visit(f->body);

  // generate SC MODULE
  this->stream << "SC_MODULE(" << f->name << ") \n{\n";
  // we fix the clock and reset for now
  Print(Indent);
  this->stream << "\t" << "sc_in<bool> clk;\n";
  this->stream << "\t" << "sc_in<bool> rst;\n\n";
  // generate ports
  std::list<std::string> port_names;
  
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
      
      this->stream << "\t" << "cynw_p2p < ";
      PrintType(std::get<1>(arg), this->stream);
      this->stream << " >";
      std::string port_name = std::get<0>(arg);
      std::string port_direction;
      if (visitor.is_inport(port_name)) {
        port_direction = "in";
      } else if (visitor.is_outport(port_name)) {
        port_direction = "out";
      } else {
        LOG(ERROR) << "Can't decide port direction.";
      }

      this->stream << "::" << port_direction << "\t";
      this->stream << std::get<0>(arg); // print arg name
      this->stream << ";\n";
      // add port name to list
      port_names.push_back(std::get<0>(arg));
      // allocate storage
      // TODO: find out what this does
      const BufferNode* buf = f->api_args[i].as<BufferNode>();
      if (v.type().is_handle() && buf) {
        var_shape_map_[buf->data.get()] = buf->shape;
      }
    }
  }

  // generate constructor
  this->stream << "\n\t" << "SC_CTOR( " << f->name << " ) \n\t:";
  // initialize clock and reset
  this->stream << " " << "clk( " << "\"clk\"" << " )\n";
  this->stream << "\t, " << "rst( " << "\"rst\"" << " )\n";
  // initialize i/o ports
  for (auto it = port_names.begin(); it != port_names.end(); ++it) {
    std::string name = *it;
    this->stream << "\t, " << name << "( \"" << name << "\" )\n";
  }
  this->stream << "\t{\n";
  // initlialize clocked thread
  this->stream << "\t\t" << "SC_CTHREAD( thread1, clk.pos() );\n";
  // setup reset signal
  this->stream << "\t\t" << "reset_signal_is( rst, 0 );\n";
  //connect clk and rst power to modular interface ports
  for (auto it = port_names.begin(); it != port_names.end(); ++it) {
    std::string name = *it;
    this->stream << "\t\t" << name << '.' << "clk_rst( clk, rst );\n";
  }
  this->stream << "\t" << "}\n\n";

  // generate process function
  this->stream << "\t" << "void thread1()\n\t{\n";
  // generate reset code
  this->stream << "\t\t" << "{\n";
  this->stream << "\t\t" << "HLS_DEFINE_PROTOCOL(\"reset\");\n";
  for (auto it = port_names.begin(); it != port_names.end(); ++it) 
    this->stream << "\t\t" << *it << '.' << "reset();\n";
  this->stream << "\t\t" << "wait();" << "\n" << "\t\t}\n";
  // generate function body
  this->stream << "\twhile( true ) \n\t{\n";
  
  int func_scope = this->BeginScope();
  range_ = CollectIterRange(f->body);
  this->PrintStmt(f->body);
  this->EndScope(func_scope);

  this->stream << "\t" << "}\n";
  
  this->stream << "};\n\n";

  }


} // namespace codegen
} // namespace TVM