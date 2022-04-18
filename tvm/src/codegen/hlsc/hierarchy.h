/*!
 *  Copyright (c) 2021 by Contributors
 * \file hierarchy.h
 * \brief Analyze design hierarchy
 */
#ifndef CODEGEN_HLSC_HIERARCHY_H_
#define CODEGEN_HLSC_HIERARCHY_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>

namespace TVM {
namespace ir {

class Hierarchy : public IRVisitor {
 public:
  Hierarchy() : _args{}, _arg_types{}, _arg_names{} {}

  void Visit_(const KernelExpr* op) final {
    _call_stack.push_back(op->name);
    // collect args info
    for (Expr arg : op->args) {
      _args[op->name].push_back(arg);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const KernelDef* op) final {
    _def_list.push_back(op->name);
    for (unsigned int i = 0; i < op->args.size(); i++) {
      auto e = op->arg_types[i];
      _arg_types[op->name].push_back(e);
      VarExpr arg = op->args[i];
      std::string n = arg.as<Variable>()->name_hint;
      _arg_names[op->name].push_back(n);
    }
  }

  std::list<std::string> get_submodule_def() { return _def_list; }

  std::list<std::string> get_submodules() { return _call_stack; }

  std::map<std::string, std::vector<Expr> > get_submodule_args() {
    return _args;
  }

  std::map<std::string, std::vector<Expr> > get_arg_types() {
    return _arg_types;
  }

  std::map<std::string, std::vector<std::string> > get_arg_names() {
    return _arg_names;
  }

 private:
  std::list<std::string> _call_stack;
  std::list<std::string> _def_list;
  // from KernelExpr
  std::map<std::string, std::vector<Expr> > _args;
  // from KernelDef
  std::map<std::string, std::vector<Expr> > _arg_types;
  std::map<std::string, std::vector<std::string> > _arg_names;
};

}  // namespace ir
}  // namespace TVM

#endif  // CODEGEN_HLSC_HIERARCHY_H_
