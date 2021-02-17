/*!
 *  Copyright (c) 2021 by Contributors
 *  Restore task graph and tranform layout.
 */
// Transform the tensor layout based on annotation
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "../arithmetic/compute_expr.h"

namespace TVM {
namespace ir {

using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::string;

struct TaskNode {
  string name;
  // Tensor being updated in the task
  unordered_set<const Variable*> updated_tensors;
  // Input tensors needed by the task
  unordered_set<const Variable*> input_tensors;
  // Children tasks name
  unordered_set<string> children;
  // Parent tasks name
  unordered_set<string> parents;
};

// TODO: save common passes into ir_pass.h
class IterRangeCollector final : public IRVisitor {
  public:
    IterRangeCollector(
      std::unordered_map<const Variable*, Expr>& range, 
      std::vector<const Variable*>& loop_iter_vars)
      : range_(range), loop_iter_vars_(loop_iter_vars) {}
    void Visit_(const For* op) override {
      range_[op->loop_var.get()] = Simplify(op->extent - 1);
      loop_iter_vars_.push_back(op->loop_var.get());
      this->Visit(op->body);
    }
    std::unordered_map<const Variable*, Expr>& range_;
    std::vector<const Variable*>& loop_iter_vars_;
};

// Reorder the loop iterator in the buffer index
// Example: if the tensor B is accessed (with in the loop nest)
//     B[a1*factor + a2]
// Then after mutation, we got B[a2*factor+a1]
Stmt ReorderBufferAccess(Stmt s, string tensor_name) {
  std::unordered_map<const Variable*, Expr> range_;
  std::vector<const Variable*> loop_iter_vars_;
  IterRangeCollector irc(range_, loop_iter_vars_);
  irc.Visit(s);

  std::unordered_map<const Variable*, Expr> vmap;
  for (size_t k = 0; k < irc.loop_iter_vars_.size(); k++) {
  }
  Stmt stmt = Substitute(s, vmap);
  return stmt;
};

// Collect tensor type and shape information
class TypeShapeCollector final : public IRMutator {
  public: 
    TypeShapeCollector(Array<NodeRef>& api_args) {
    for (size_t i = 0; i < api_args.size(); i++) { 
      if (const Variable* v = api_args[i].as<Variable>()) {
        top_arg_names.insert(v->name_hint);

      } else if (auto buf = api_args[i].as<BufferNode>()) {
        CHECK(buf->data.as<Variable>());
        top_arg_names.insert(buf->name); 
        shape_[buf->data.get()->name_hint] = buf->shape;
        dtype_[buf->data.get()->name_hint] = buf->dtype;
        HCL_DEBUG_LEVEL(2) << "  [ collect shape ] " << buf->name;
      }
    }
  };

  Stmt Mutate_(const Allocate *op, const Stmt& s) final {
    auto v = op->buffer_var.get();
    auto name = v->name_hint; 
    // Save shape and dtype information
    shape_[name] = op->extents;
    dtype_[name] = op->type;
    HCL_DEBUG_LEVEL(2) << "  [ collect shape ] " << name;
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const KernelDef* op, const Stmt& s) {
    for (size_t i = 0; i < op->args.size(); i++) {
      string name = op->args[i].get()->name_hint;
      auto shape = op->arg_shapes[i];
      shape_[name] = shape;
      CHECK(op->arg_types[i].as<StringImm>());
      dtype_[name] = Str2Type(op->arg_types[i].as<StringImm>()->value);
      HCL_DEBUG_LEVEL(2) << "  [ collect shape ] " << name;
    }
    return IRMutator::Mutate_(op, s);
  }

  Type Str2Type(string type_str) {
    if (type_str.find("int") == 0) {
      type_str.erase(0, 3);
      int bits = std::atoi(type_str.c_str());
      return Int(bits);
    } else if (type_str.find("uint") == 0) {
      type_str.erase(0, 4);
      int bits = std::atoi(type_str.c_str());
      return UInt(bits);
    } else if (type_str.find("float") == 0) {
      type_str.erase(0, 5);
      int bits = std::atoi(type_str.c_str());
      return Float(bits);
    }
    return Int(32);
  }

  unordered_set<string> top_arg_names;
  unordered_map<string, Array<Expr> > shape_;
  unordered_map<string, Type> dtype_;
};


void CollectTypeShape(Stmt body, unordered_map<string, Array<Expr> >& shape, 
  unordered_map<string, Type>& dtype, Array<NodeRef>& api_args) {
    HCL_DEBUG_LEVEL(2) << "---------- collect shape/dtype ---------";
    TypeShapeCollector tsc(api_args);
    tsc.Mutate(body);
    dtype = tsc.dtype_;
    shape = tsc.shape_;
};

// Check all the tensors in the Stmt. Get information
// of their access pattern (write_only, read_only or read_write)
class BufferStatusCollector : public ir::IRMutator {
 public:
  Stmt Mutate_(const Store* op, const Stmt& s) {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    if (!local_buffers.count(op->buffer_var.get())) {
      updated_tensors.insert(op->buffer_var.get());
    }
    return stmt;
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    auto name = op->buffer_var.get()->name_hint;
    if (!local_buffers.count(op->buffer_var.get())) {
      input_tensors.insert(op->buffer_var.get());
    }
    return IRMutator::Mutate_(op, e);
  }  

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    local_buffers.insert(op->buffer_var.get());
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    return stmt;
  }

  unordered_set<const Variable*> local_buffers;
  unordered_set<const Variable*> updated_tensors;
  unordered_set<const Variable*> input_tensors;
};

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

        // Create a task node in the graph
        BufferStatusCollector bsc;
        bsc.Mutate(op->body);
        TaskNode task = {
          name, bsc.updated_tensors, bsc.input_tensors, {}, {}
        };
      
        // Checking depending input tensors
        Array<Var> kernel_input_vars;
        for (auto& input: kernel_input_args) {
          Var v(input.node_);
          kernel_input_vars.push_back(v);
        }
        Array<Var> undefs = UndefinedVars(body, kernel_input_vars);
        // The task can be a producer of a tensor, or it will just update
        // a set of tensors. If the input tensor is not defined in this task 
        // nor in the input arguments, then it must have been defined in the 
        // previous tasks visited in the traversal
        for (auto& var: undefs) {
          auto parents = checkTensorLiveness(var.get());
          for (auto& parent_task_name: parents) {
            task.parents.insert(parent_task_name);
            CHECK(task_map.count(parent_task_name));
            task_map[parent_task_name].children.insert(name);
          }

        }
        task_map[name] = task;
        HCL_DEBUG_LEVEL(2) << "[ debug ] producing tensor " << name;
      }
      top_level_producer_ = true;
      return ProducerConsumer::make(op->func, op->is_producer, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  // Return the nearest parent task that a tensor has been updated
  vector<string> checkTensorLiveness(const Variable* var) {
      // Tasks where the tensor has been updated
      vector<string> parents;
      for (auto& kv: task_map) {
        for (auto& t: kv.second.updated_tensors) {
          if (t == var) {
            HCL_DEBUG_LEVEL(2) << "[ debug ] Tensor " << var->name_hint 
              << " has been updated in task " << kv.second.name;
            parents.push_back(kv.second.name);
          }
        }
      }
      return parents;
  };

  // Check the task graph inside the device scope
  bool device_scope_{false};
  bool top_level_producer_{true};
  // Input tensor to the top level function
  Array<VarExpr> kernel_input_args;
  // Map from task name to TaskNode
  unordered_map<string, TaskNode> task_map;

};

// 1. Locate the which tensor (in which stage) will be layout transformed
// 2. Locate its parent task and insert the layout mutation statements 
class LayoutTransformer : public IRMutator {
 public:
  explicit LayoutTransformer(
    unordered_map<string, TaskNode>& task_map,
    Array<NodeRef>& api_args)
    : task_map_(task_map), api_args_(api_args) { }

  unordered_map<string, TaskNode>& task_map_;
  Array<NodeRef>& api_args_;
  unordered_map<string, Array<Expr> > shape_;
  unordered_map<string, Type> dtype_;

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) {
    // The tensor to be transformed
    if (op->attr_key == attr::tensor_layout_attrs) {
        VarExpr var(op->node.node_);
        auto name = var.get()->name_hint;
        CHECK(shape_.count(name)) << name;
        CHECK(dtype_.count(name)) << name;
            
        size_t pos = 0;
        string delimiter = ":";
        string token;
        vector<int> target_shape;

        CHECK(op->value.as<StringImm>());
        string s(op->value.as<StringImm>()->value);

        int target_total_width = 1;
        while ((pos = s.find(delimiter)) != string::npos) {
          token = s.substr(0, pos);
          target_shape.push_back(std::stoi(token));
          s.erase(0, pos + delimiter.length());
          target_total_width *= std::stoi(token);
        }
        target_total_width *= std::stoi(s);

        // Check tranform type (tranpose or packing)
        int origin_total_width = 1;
        for (auto& dim: shape_.at(name)) {
          CHECK(dim.as<IntImm>());
          origin_total_width *= dim.as<IntImm>()->value;
        }

        HCL_DEBUG_LEVEL(2) << origin_total_width << ", " << target_total_width;
        // TODO(hecmay): handle reshape
        if (origin_total_width == target_total_width) {
          HCL_DEBUG_LEVEL(2) << "[ debug ] Transpose layout of tensor " 
            << name << "(" << shape_[name] << ") to (" << 
            op->value << ")";
          // Just change the access order
          // The (in-place) tranposition is done in another pass
          return ReorderBufferAccess(op->body, name);

        } else {
          // Pack the last dimension by default
          HCL_DEBUG_LEVEL(2) << "[ debug ] Pack layout of tensor " 
            << name << "(" << shape_[name] << ") to (" << 
            op->value << ")";
        }
        return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Transform(Stmt s) {
    CollectTypeShape(s, shape_, dtype_, api_args_);
    return this->Mutate(s);
  }

};

Stmt TransformLayout(Stmt stmt, Array<NodeRef> api_args) {
  // Restore the task graph from the IR
  HCL_DEBUG_LEVEL(2) << "------------ Transform Layout --------------";
  TaskGraphBuilder tgb(api_args);
  stmt = tgb.Mutate(stmt);

  // Check the tensor in worklist (to be transposed or packed)
  // Here we create a new attribute statment key "tensor_view
  LayoutTransformer ltm(tgb.task_map, api_args);
  stmt = ltm.Transform(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace TVM
