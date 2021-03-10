/*!
 *  Copyright (c) 2020 by Contributors
 * \file dce.cc
 * \brief dead code elimination
 */
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include <unordered_map>
#include "./ir_util.h"
#include <ir/IREquality.h>
#include <arithmetic/Substitute.h>

namespace TVM {
namespace ir {

using std::string;
using std::vector;
using std::unordered_map;
using std::unordered_set;

class UnusedStageBufferRemover final : public IRMutator {
 public: 
  UnusedStageBufferRemover(
    std::unordered_set<const Variable*>& unused_vars_)
    : unused_vars(unused_vars_) {}
  
  // Remove loops where extent=1
  Stmt Mutate_(const For* op, const Stmt& s) {
    if (auto v = op->extent.as<IntImm>()) {
      if (v->value == 1) {
        HCL_DEBUG_LEVEL(2) << "  remove loop extent=1. " << op->loop_var;
        std::unordered_map<const Variable*, Expr> vmap;
        vmap[op->loop_var.get()] = IntImm::make(Int(32), 0);
        return Substitute(this->Mutate(op->body), vmap);
      }
    }
    return IRMutator::Mutate_(op, s);;
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    for (auto& v : unused_vars) {
        if (op->buffer_var.get()->name_hint == v->name_hint) {
          HCL_DEBUG_LEVEL(2) << "Removed unused var " << v;
          return this->Mutate(op->body);
        }
    }
    
    return stmt;
  }

  std::unordered_set<const Variable*>& unused_vars;
  bool remove_producer{false};
};

Stmt DeadCodeElimination(Stmt stmt, Array<NodeRef> arg_list) {
  std::map<const Variable*, Array<Expr> > shape_map;
  std::map<const Variable*, VarExpr> buffer_map;
  Array<Var> input_args;
  for (size_t i = 0; i < arg_list.size(); i++) {
    if (const BufferNode* node = arg_list[i].as<BufferNode>()) {
      shape_map[node->data.get()] = node->shape;
      input_args.push_back(node->data);
      buffer_map[node->data.get()] = node->data;
    }
  }

  HCL_DEBUG_LEVEL(2) << "------------- DCE ----------------";
  auto unused_vars = UnusedVars(stmt, input_args);
  // Remove the unused buffers
  UnusedStageBufferRemover ubr(unused_vars);
  stmt = ubr.Mutate(stmt);
  
  HCL_DEBUG_LEVEL(2) << "----------------------------------";
  HCL_DEBUG_LEVEL(2) << stmt;
  return Simplify(stmt);
}

}  // namespace ir
}  // namespace TVM
