/*!
 *  Copyright (c) 2021 by Contributors
 * \file port_direction.h
 * \brief Implements an IR pass to infer SystemC module port directions
 */
#ifndef PORT_DIRECTION_H_
#define PORT_DIRECTION_H_

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include "./codegen_shls.h"

namespace TVM {
namespace ir {

class PortDirection : public IRVisitor {
 public:
  explicit PortDirection(std::list<std::string> ports) : _ports(ports) {}

  void Visit_(const Load* op) final {
    std::string var_name = op->buffer_var.get()->name_hint;
    auto it = std::find(_ports.begin(), _ports.end(), var_name);
    if (it!=_ports.end())
      _port_direction.insert(std::pair<std::string, std::string>(var_name, "in"));
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store* op) final {
    std::string var_name = op->buffer_var.get()->name_hint;
    auto it = std::find(_ports.begin(), _ports.end(), var_name);
    if (it!=_ports.end())
      _port_direction.insert(std::pair<std::string, std::string>(var_name, "out"));
    IRVisitor::Visit_(op);
  }

  std::string get_direction(std::string var_name) {
    CHECK(_port_direction.count(var_name));
    return _port_direction[var_name];
  }

 private:
  std::list<std::string> _ports;
  std::map<std::string, std::string> _port_direction;
};

} // namespace ir
} // namespace TVM

#endif // PORT_DIRECTION_H_