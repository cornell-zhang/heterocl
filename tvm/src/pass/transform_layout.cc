/*!
 *  Copyright (c) 2021 by Contributors
 *  Restore task graph and tranform layout.
 */
// Transform the tensor layout based on annotation
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "../arithmetic/compute_expr.h"

namespace TVM {
namespace ir {

using std::unordered_map;
using std::vector;
using std::string;

// Each task in the graph represents the logic performed by 
// a HCL stage. The task graph is a coarse grained DFG. 
// There is no no control flow branching across different tasks
class TaskGraphBuilder : public IRMutator {
 public:
  explicit TaskGraphBuilder(Array<NodeRef> api_args) { }

  Stmt Mutate_(const KernelDef* op, const Stmt& s) {
    if (op->name == "test") {
      device_scope_ = true;

      // Save the input tensors
      for (auto& v : op->args) {
        kernel_input_args.push_back(v);
      }
      Stmt body = this->Mutate(op->body);
      device_scope_ = false;
      return KernelDef::make(op->args, op->arg_shapes, 
                op->arg_types, op->arg_tensors,
                body, op->ret_void, op->ret_type, 
                op->name, op->attributes);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) {
    if (top_level_producer_ && device_scope_) {
      top_level_producer_ = false;
      Stmt body = this->Mutate(op->body);
      if (op->is_producer) {
        std::string name = op->func->func_name();
        task_name_to_stmt[name] = s;
        HCL_DEBUG_LEVEL(2) << "[ debug ] producing tensor " << name;
      
        // Checking depending input tensors
        Array<Var> kernel_input_vars;
        for (auto& input: kernel_input_args) {
          Var v(input.node_);
          kernel_input_vars.push_back(v);
        }
        Array<Var> undefs = UndefinedVars(body, kernel_input_vars);
        for (auto& var: undefs) {
          string tensor_name = var.get()->name_hint;
          CHECK(task_name_to_stmt.count(tensor_name)) 
            << "Cannot locate producer of tensor " << tensor_name;
        }
      }
      top_level_producer_ = true;
      return ProducerConsumer::make(op->func, op->is_producer, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  // Check the task graph inside the device scope
  bool device_scope_{false};
  bool top_level_producer_{true};
  // Input tensor to the top level function
  Array<VarExpr> kernel_input_args;
  unordered_map<string, Stmt> task_name_to_stmt;
  unordered_map<string, vector<string> > dep_task_map;
};

class LayoutTransformer : public IRMutator {
 public:
  explicit LayoutTransformer(
    unordered_map<string, vector<string> >& dep_task_map)
    : dep_task_map_(dep_task_map) { }
  unordered_map<string, vector<string> >& dep_task_map_;
};

Stmt TransformLayout(Stmt stmt, Array<NodeRef> api_args) {
  // Restore the task graph from the IR
  TaskGraphBuilder tgb(api_args);
  stmt = tgb.Mutate(stmt);

  // Check the tensor in worklist (to be transposed or packed)
  // Here we create a new attribute statment key "tensor_view"
  LayoutTransformer ltm(tgb.dep_task_map);
  stmt = ltm.Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace TVM
