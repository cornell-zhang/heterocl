/*!
 *  Copyright (c) 2021 by Contributors
 * \file hierarchy.h
 * \brief Analyze design hierarchy 
 */
#ifndef HIERARCHY_H_
#define HIERARCHY_H_

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>

namespace TVM {
namespace ir {

class Hierarchy : public IRVisitor {
 public:
  explicit Hierarchy(){}

  void Visit_(const KernelExpr* op) final {
    LOG(INFO) << "KernelExpr op name: " << op->name;
    _call_stack.push_back(op->name);
    // collect args info
    for (Expr arg : op->args) {
      _args[op->name].push_back(arg);
      LOG(INFO) << "KernelExpr arg: " << arg;
    }
    IRVisitor::Visit_(op);
  }

  std::list<std::string> get_submodules() {
    return _call_stack;
  }

  std::map<std::string, std::list<Expr> > get_submodule_args() {
    return _args;
  }
  
 private:
  std::list<std::string> _call_stack;
  std::map<std::string, std::list<Expr> > _args;
};

} // namespace ir
} // namespace TVM

#endif // HIERARCHY_H_