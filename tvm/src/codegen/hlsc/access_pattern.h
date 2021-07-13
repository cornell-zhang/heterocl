/*!
 *  Copyright (c) 2021 by Contributors
 * \file access_pattern.h
 * \brief Implements an IR pass to analyze the access pattern for variable of interest
 */
#ifndef CODEGEN_HLSC_ACCESS_PATTERN_H_
#define CODEGEN_HLSC_ACCESS_PATTERN_H_

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>

namespace TVM {
namespace ir {

class AccessPattern : public IRVisitor {
/*
    Questions:

    1. What is predicate in Load/Save?

    2. Variable node's name that we are interested in -> index Expr from Load/Store (e.g. Variable or Mul (contains Variable), etc.)
                                                      -> loop var from For (e.g. a Variable)
        What to do when index Expr is a Mul? For example, (i*2)
        Maybe we can continue to Visit_ this Mul node, and get step from here 
*/

 public:
  explicit AccessPattern(std::list<std::string> variable_names)
    : _target_variables(variable_names) {}

  void Visit_(const Load* op) final {
    // do stuff here
    std::string name = op->buffer_var.get()->name_hint;
    LOG(INFO) << "[AccessPattern] " << "Load op name: " << name;
    LOG(INFO) << "[AccessPattern] " << "Load op index: " << op->index;
    LOG(INFO) << "[AccessPattern] " << "Load op index's type key: "
                                    << op->index->type_key();
    auto it = std::find(_target_variables.begin(),
                        _target_variables.end(), name);
    if (it != _target_variables.end()) {
      if (op->index->is_type<Variable>())
        this->_is_affine.push_back(name);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store* op) final {
    // do stuff here
    std::string name = op->buffer_var.get()->name_hint;
    LOG(INFO) << "[AccessPattern] " << "Store op name: " << name;
    LOG(INFO) << "[AccessPattern] " << "Store op index: "
                                    << op->index;  // an Expr
    LOG(INFO) << "[AccessPattern] " << "Store op index's type key: "
              << op->index->type_key();  // could be Variable or Mul?
    auto it = std::find(_target_variables.begin(),
                        _target_variables.end(), name);
    if (it != _target_variables.end()) {
      if (op->index->is_type<Variable>())
        this->_is_affine.push_back(name);
    }
    IRVisitor::Visit_(op);
  }

  // get iteration variable's min and extent from For
  void Visit_(const For* op) final {
    // do stuff here
    LOG(INFO) << "[AccessPattern] " << "For op loop_var: " << op->loop_var;
    LOG(INFO) << "[AccessPattern] " << "For op min: " << op->min;
    LOG(INFO) << "[AccessPattern] " << "For op extent: " << op->extent;
    LOG(INFO) << "[AccessPattern] " << "For op for_type: " << op->for_type;
    IRVisitor::Visit_(op);
  }

  bool is_affine(std::string var_name) {
    auto it = std::find(_is_affine.begin(), _is_affine.end(), var_name);
    return it != _is_affine.end();
  }

 private:
  std::list<std::string> _target_variables;
  std::list<std::string> _is_affine;
};

}  // namespace ir
}  // namespace TVM

#endif  // CODEGEN_HLSC_ACCESS_PATTERN_H_
