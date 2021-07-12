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

class PortDirection : public IRVisitor {
 public:
  explicit PortDirection(const std::list<std::string> &ports) : _ports(ports) {}

  void Visit_(const Load* op) final {
    std::string var_name = op->buffer_var.get()->name_hint;
    auto it = std::find(_ports.begin(), _ports.end(), var_name);
    if (it !=_ports.end()) {
      _in_ports.push_back(var_name);
      // LOG(INFO) << "[AccessPattern] " << "Load op name: " << var_name;
      // LOG(INFO) << "[AccessPattern] " << "Load op index: " << op->index;
      // LOG(INFO) << "[AccessPattern] "
      //           << "Load op index's type key: " << op->index->type_key();
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store* op) final {
    std::string var_name = op->buffer_var.get()->name_hint;
    auto it = std::find(_ports.begin(), _ports.end(), var_name);
    if (it !=_ports.end()) {
      _out_ports.push_back(var_name);
      // LOG(INFO) << "[AccessPattern] " << "Store op name: " << var_name;
      // LOG(INFO) << "[AccessPattern] " << "Store op index: " << op->index;
      // LOG(INFO) << "[AccessPattern] " << "Store op index's type key: "
      //                                << op->index->type_key();
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

  std::string get_direction(std::string var_name) {
    auto it_in  = std::find(_in_ports.begin(), _in_ports.end(), var_name);
    auto it_out = std::find(_out_ports.begin(), _out_ports.end(), var_name);
    bool is_in = it_in != _in_ports.end();
    bool is_out = it_out != _out_ports.end();
    if (is_in && is_out) {
      return "inout";
    } else if (is_in) {
      return "in";
    } else if (is_out) {
      return "out";
    } else {
      LOG(FATAL) << "[SystemC Backend][PortDirectionInfer]"
                 <<" can't decide the port direction for port: " << var_name;
      return "not_port";
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
