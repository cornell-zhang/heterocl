/*!
 *  Copyright (c) 2021 by Contributors
 * \file port_direction.h
 * \brief Implements an IR pass to infer SystemC module port directions
 */

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include "./codegen_shls.h"

namespace TVM {
namespace ir {

class PortDirection : public IRVisitor {
 public:
  // we can't use CodeGenC because we can't add PortDirection
  // as friend class in CodeGenBaseClass
  // explicit PortDirection(codegen::CodeGenStratusHLS* cgen) : _cgen(cgen) {}

  void Visit_(const Load* op) final {
    std::string var_name = op->buffer_var.get()->name_hint;
    LOG(INFO) << "Load var name " << var_name;
    in_ports.push_back(var_name);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store* op) final {
    std::string var_name = op->buffer_var.get()->name_hint;
    LOG(INFO) << "Store var name " << var_name;
    out_ports.push_back(var_name);
    IRVisitor::Visit_(op);
  }

  bool is_inport(std::string name) {
    auto it_inports = std::find(in_ports.begin(), in_ports.end(), name);
    return it_inports != in_ports.end();
  }

  bool is_outport(std::string name) {
    auto it_outports = std::find(out_ports.begin(), out_ports.end(), name);
    return it_outports != out_ports.end();
  }


 private:
  //codegen::CodeGenStratusHLS* _cgen;
  std::list<std::string> in_ports;
  std::list<std::string> out_ports;
};

} // namespace ir
} // namespace TVM
