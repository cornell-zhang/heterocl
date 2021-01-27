/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Utility to make loop nest.
 * \file op_util.cc
 */
#include "op_util.h"
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

namespace TVM {
namespace op {

using namespace arith;
using namespace ir;

// replacer to replace tensors
class TensorReplacer : public ir::IRMutator {
 public:
  explicit TensorReplacer(const std::unordered_map<Tensor, Tensor>& vmap)
      : vmap_(vmap) {}

  Expr Mutate_(const ir::Call* op, const Expr& e) {
    if (op->call_type == ir::Call::Halide) {
      Tensor t = Operation(op->func.node_).output(op->value_index);
      auto it = vmap_.find(t);
      if (it != vmap_.end()) {
        Expr ret = ir::Call::make(op->type, it->second->op->name, op->args,
                                  op->call_type, it->second->op,
                                  it->second->value_index);
        found = true;
        return IRMutator::Mutate_(ret.as<ir::Call>(), ret);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  // whether it is found.
  bool found{false};

 private:
  const std::unordered_map<Tensor, Tensor>& vmap_;
};

Stmt ReplaceTensor(Stmt stmt,
                   const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  Stmt ret = repl.Mutate(stmt);
  return repl.found ? ret : stmt;
}
Expr ReplaceTensor(Expr expr,
                   const std::unordered_map<Tensor, Tensor>& replace) {
  TensorReplacer repl(replace);
  Expr ret = repl.Mutate(expr);
  return repl.found ? ret : expr;
}

Stmt Substitute(Stmt s, const std::unordered_map<IterVar, Expr>& value_map) {
  std::unordered_map<const Variable*, Expr> init;
  for (const auto& kv : value_map) {
    init[kv.first->var.get()] = kv.second;
  }
  return ir::Substitute(s, init);
}

Stmt Substitute(Stmt s,
                const std::unordered_map<const Variable*, Expr>& value_map) {
  std::unordered_map<const Variable*, Expr> init;
  for (const auto& kv : value_map) {
    init[kv.first] = kv.second;
  }
  return ir::Substitute(s, init);
}

}  // namespace op
}  // namespace TVM
