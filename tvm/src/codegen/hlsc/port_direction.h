/*!
 *  Copyright (c) 2021 by Contributors
 * \file port_direction.h
 * \brief Implements an IR pass to infer SystemC module port directions
 */
#ifndef CODEGEN_HLSC_PORT_DIRECTION_H_
#define CODEGEN_HLSC_PORT_DIRECTION_H_

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>

namespace TVM {
namespace ir {

enum class PortType { ChannelIn, ChannelOut, Memory, OffChipMemory };

class PortDirection : public IRVisitor {
 public:
  explicit PortDirection(const std::list<std::string> &ports) : _ports(ports) {}

  void Visit_(const Load* op) final {
    std::string var_name = op->buffer_var.get()->name_hint;
    auto it = std::find(_ports.begin(), _ports.end(), var_name);
    if (it !=_ports.end()) {
      _in_ports.push_back(var_name);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store* op) final {
    std::string var_name = op->buffer_var.get()->name_hint;
    auto it = std::find(_ports.begin(), _ports.end(), var_name);
    if (it !=_ports.end()) {
      _out_ports.push_back(var_name);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Cast*op) final {
    if (const Variable* v = op->value.as<Variable>()) {
      std::string var_name = v->name_hint;
      auto it = std::find(_ports.begin(), _ports.end(), var_name);
      if (it != _ports.end()) {
        _in_ports.push_back(var_name);
      }
    }
    IRVisitor::Visit_(op);
  }

  bool isOffChip(std::string name) {
    const std::string targets[1] = {"DRAM"};
    bool offchip = false;
    for (const std::string target : targets) {
      std::string name_upper = name;
      std::transform(name_upper.begin(), name_upper.end(), 
        name_upper.begin(), ::toupper);
      offchip = name_upper.find(target) != std::string::npos;
    }
    return offchip;
  }

  PortType get_direction(std::string var_name) {
    auto it_in  = std::find(_in_ports.begin(), _in_ports.end(), var_name);
    auto it_out = std::find(_out_ports.begin(), _out_ports.end(), var_name);
    bool is_in = it_in != _in_ports.end();
    bool is_out = it_out != _out_ports.end();
    if (is_in && is_out) {
      bool offchip = isOffChip(var_name);
      return offchip ? PortType::OffChipMemory : PortType::Memory;
    } else if (is_in) {
      return PortType::ChannelIn;
    } else if (is_out) {
      return PortType::ChannelOut;
    } else {
      LOG(FATAL) << "[SystemC Backend][PortDirectionInfer]"
                 <<" cannot infer the port type and direction for port: " << var_name;
    }
  }


 private:
  std::list<std::string> _in_ports;
  std::list<std::string> _out_ports;
  std::list<std::string> _ports;
};

}  // namespace ir
}  // namespace TVM

#endif  // CODEGEN_HLSC_PORT_DIRECTION_H_
