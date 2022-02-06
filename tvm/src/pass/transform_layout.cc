/*!
 *  Copyright (c) 2021 by Contributors
 *  Restore task graph and tranform layout.
 */
// Transform the tensor layout based on annotation
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../arithmetic/compute_expr.h"

namespace TVM {
namespace ir {

using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;

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

struct TransformInfo {
  string name;
  VarExpr var;
  string anchor_producer;
  Array<Expr> origin_shape;
  Array<Expr> target_shape;
  Type origin_type;
  Type type;
  bool is_transpose;
  bool is_pack;
  int pack_factor;
  bool is_written;
};

class TensorSubstitution final : public IRMutator {
 public:
  TensorSubstitution(unordered_map<const Variable*, Expr>& vmap)
      : vmap_(vmap) {}

  Stmt Mutate_(const KernelStmt* op, const Stmt& s) final {
    Array<Expr> new_args;
    for (auto& e : op->args) {
      auto ptr = e.as<Variable>();
      CHECK(ptr) << e;
      bool is_found = false;
      Expr new_buf;
      for (auto& kv : vmap_) {
        if (kv.first->name_hint == ptr->name_hint) {
          HCL_DEBUG_LEVEL(2) << "  -- [substitute] " << ptr->name_hint
                             << " in kernel " << op->name;
          is_found = true;
          new_buf = Expr(kv.second.node_);
        }
      }
      if (is_found) {
        CHECK(new_buf.defined());
        new_args.push_back(new_buf);
      } else {
        new_args.push_back(e);
      }
    }
    return KernelStmt::make(new_args, op->name, op->annotate_keys,
                            op->annotate_values);
  }
  unordered_map<const Variable*, Expr>& vmap_;
};

Stmt SubstituteTensor(Stmt s, unordered_map<const Variable*, Expr> vmap) {
  return TensorSubstitution(vmap).Mutate(s);
}

// Return string repr of type
string Type2Str(Type type) {
  string str = "int";
  if (type.code() == Type::Float) {
    str = "float";
  } else if (type.code() == Type::Int) {
    str = "int";
  } else if (type.code() == Type::UInt) {
    str = "uint";
  }
  return str + std::to_string(type.bits());
}

class TransformedBufferInserter final : public IRMutator {
 public:
  TransformedBufferInserter(std::string target_producer, TransformInfo& info)
      : target_producer_(target_producer), info_(info) {}

  // Insert buffer before the producer stage
  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) {
    if (op->is_producer) {
      std::string name = op->func->func_name();
      if (name == target_producer_) {
        Stmt body = this->Mutate(op->body);
        HCL_DEBUG_LEVEL(2) << "[ debug ] insert layout transformation before "
                           << name;
        VarExpr var(info_.name + ".new");
        VarExpr old_var(info_.var.node_);
        Type type = info_.type;
        Array<Expr> origin_shape = info_.origin_shape;

        std::string dtype;
        if (info_.origin_type.code() == Type::Int) {
          dtype = "int";
        } else if (info_.origin_type.code() == Type::UInt) {
          dtype = "uint";
        } else if (info_.origin_type.code() == Type::Float) {
          dtype = "float";
        }

        // For packed-only var on interface, since the passed into memory
        // is stored in major fashion continuously, so the data is automatically
        // packed already
        if (info_.is_pack && !info_.is_transpose) {
          // Insert serilization intrisic for AutoSA
          if (!info_.is_written) {
            // Insert allocation dev_ser_tensor
            VarExpr new_var(info_.name + ".dev.ser");
            unordered_map<const Variable*, Expr> vmap;
            vmap[info_.var.get()] = new_var;

            // resize the serialized buffer since if may have replicates
            // due to certain access pattern
            body = SubstituteTensor(body, vmap);
            body = ProducerConsumer::make(op->func, op->is_producer, body);

            Stmt serialize = Evaluate::make(Call::make(
                Int(32), "serialize", {info_.name, dtype}, Call::Intrinsic));
            body = Block::make(serialize, body);

            body =
                Allocate::make(new_var, info_.origin_type, info_.origin_shape,
                               make_const(Bool(type.lanes()), true), body);
            body = AttrStmt::make(new_var, attr::storage_scope,
                                  StringImm::make("global"), body);
            return body;

          } else {
            VarExpr new_var(info_.name + ".dev.deser");
            unordered_map<const Variable*, Expr> vmap;
            vmap[info_.var.get()] = new_var;
            body = SubstituteTensor(body, vmap);
            body = ProducerConsumer::make(op->func, op->is_producer, body);

            Stmt deserialize = Evaluate::make(Call::make(
                Int(32), "deserialize", {info_.name, dtype}, Call::Intrinsic));
            body = ProducerConsumer::make(op->func, op->is_producer, body);
            body = Block::make(body, deserialize);

            body =
                Allocate::make(new_var, info_.origin_type, info_.origin_shape,
                               make_const(Bool(type.lanes()), true), body);
            body = AttrStmt::make(new_var, attr::storage_scope,
                                  StringImm::make("global"), body);
            return body;
          }

          // Insert an instrinsic to do in-place matrix tranposition
        } else if (info_.is_transpose) {
          int size = 1;
          for (auto& dim : origin_shape) {
            auto ptr = dim.as<IntImm>();
            CHECK(ptr);
            size *= ptr->value;
          }

          VarExpr new_var(info_.name + ".dev.ser");
          unordered_map<const Variable*, Expr> vmap;
          vmap[info_.var.get()] = new_var;
          body = SubstituteTensor(body, vmap);
          body = ProducerConsumer::make(op->func, op->is_producer, body);

          Stmt serialize = Evaluate::make(Call::make(
              Int(32), "serialize", {info_.name, dtype}, Call::Intrinsic));
          body = Block::make(serialize, body);

          body = Allocate::make(new_var, info_.origin_type, info_.origin_shape,
                                make_const(Bool(type.lanes()), true), body);
          body = AttrStmt::make(new_var, attr::storage_scope,
                                StringImm::make("global"), body);

          // In-place matrix transposition
          Stmt trans = Evaluate::make(
              Call::make(Int(32), "transpose", {old_var, size, origin_shape[0]},
                         Call::Intrinsic));
          body = Block::make(trans, body);
          return body;

          // Insert reshaping logic explicitly
        } else {
          // Substitute buffer
          unordered_map<const Variable*, Expr> vmap;
          vmap[info_.var.get()] = var;
          body = SubstituteTensor(body, vmap);
          HCL_DEBUG_LEVEL(2) << "------------- Substitue ---------";
          HCL_DEBUG_LEVEL(2) << "  from " << info_.var << " to " << var;
          HCL_DEBUG_LEVEL(2) << "Inside body: " << body;
        }

        // Insert pack-only loop
        if (info_.is_pack) {
          std::vector<Expr> indices, new_indices;
          std::vector<VarExpr> loop_vars;
          std::unordered_map<const Variable*, Expr> range_;
          for (size_t i = 0; i < origin_shape.size(); i++) {
            VarExpr iter(name + ".pack.r" + std::to_string(i));
            indices.push_back(iter);
            new_indices.push_back(iter);
            loop_vars.push_back(iter);
            range_[iter.get()] = Simplify(origin_shape[i] - 1);
          }
          // Dim for data packing
          VarExpr iter(name + ".pack.r");
          indices.push_back(iter);
          loop_vars.push_back(iter);

          // Expected output IR (example 512-packing)
          // for i (0, 64)
          //   for j (0, 4)
          //     A.new[i,j] = 0
          //     for p (0, 16)
          //        A.new[i,j](32*p+32, 32*p) = A[i,j*16+p]
          Array<Expr> pack_shape = info_.target_shape;
          pack_shape.push_back(info_.pack_factor);
          Expr pack_index = FlattenIndices(indices, pack_shape);
          Expr new_index = FlattenIndices(new_indices, info_.target_shape);

          // Pack + tranpose
          // Expected output IR (example 512-packing)
          // for i (0, 64)
          //   for j (0, 4)
          //     A.new[i,j] = 0
          //     for p (0, 16)
          //        A.new[i,j](32*p+32, 32*p) = A[j*16+p,i]
          if (info_.is_transpose) {
            // Move last two iters to the front (i,(j,p)) to ((j,p),i). Left
            // shifting
            std::vector<Expr> transpose_indices = {indices[1], indices[2],
                                                   indices[0]};
            pack_index = FlattenIndices(transpose_indices, pack_shape);
          }
          Expr load =
              Load::make(type, old_var, pack_index, UIntImm::make(UInt(1), 1));
          Expr slice =
              SetSlice::make(var, load, (1 + iter) * info_.pack_factor - 1,
                             iter * info_.pack_factor);
          Stmt for_stmt =
              Store::make(var, slice, new_index, UIntImm::make(UInt(1), 1));

          auto for_type = ForType::Serial;
          int bound = pack_shape.size();
          for (int j = bound - 1; j >= 0; j--) {
            auto iter = loop_vars[j];
            for_stmt = For::make(VarExpr(iter.node_), 0, pack_shape[j],
                                 for_type, DeviceAPI::None, for_stmt);
            // Insert initialization store
            if (j == bound - 1) {
              Stmt init =
                  Store::make(var, 0, new_index, UIntImm::make(UInt(1), 1));
              for_stmt = Block::make(init, for_stmt);
            }
          }
          body = Block::make(for_stmt, body);

          // Tensor transpose only
        } else {
          std::vector<Expr> indices;
          std::vector<Expr> reverse_indices;
          std::vector<VarExpr> loop_vars;
          for (size_t i = 0; i < origin_shape.size(); i++) {
            VarExpr iter(name + ".transpose.r" + std::to_string(i));
            indices.push_back(iter);
            reverse_indices.insert(reverse_indices.begin(), iter);
            loop_vars.push_back(iter);
          }
          Expr reverse_index = FlattenIndices(reverse_indices, origin_shape);
          Expr index = FlattenIndices(indices, origin_shape);
          Expr load =
              Load::make(type, old_var, index, UIntImm::make(UInt(1), 1));
          Stmt for_stmt =
              Store::make(var, load, reverse_index, UIntImm::make(UInt(1), 1));

          auto for_type = ForType::Serial;
          for (size_t j = 0; j < origin_shape.size(); j++) {
            auto iter = loop_vars[j];
            for_stmt = For::make(VarExpr(iter.node_), 0, origin_shape[j],
                                 for_type, DeviceAPI::None, for_stmt);
          }
          body = Block::make(for_stmt, body);
          HCL_DEBUG_LEVEL(2) << "[  debug  ] tranpose loop for " << var;
          HCL_DEBUG_LEVEL(2) << for_stmt;
        }

        body = Allocate::make(var, type, info_.target_shape,
                              make_const(Bool(type.lanes()), true), body);
        body = AttrStmt::make(var, attr::storage_scope,
                              StringImm::make("global"), body);
        return ProducerConsumer::make(op->func, op->is_producer, body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }
  std::string target_producer_;
  TransformInfo& info_;
};

class IndicesTransformer final : public IRMutator {
 public:
  IndicesTransformer(std::unordered_map<const Variable*, Expr>& range,
                     std::vector<VarExpr>& loop_iter_vars, TransformInfo& info)
      : range_(range), loop_iter_vars_(loop_iter_vars), info_(info) {}

  // For AutoSA backend. Just inject the information without
  // changing the IR
  Stmt Mutate_(const ExternModule* op, const Stmt& s) {
    has_autosa_module = true;
    Expr value = this->Mutate(op->value);
    Stmt body = this->Mutate(op->body);
    auto annotate_keys = op->annotate_keys;
    auto annotate_values = op->annotate_values;

    annotate_keys.push_back(StringImm::make(info_.name));
    string attr = info_.is_transpose ? "1" : "0";
    attr += "," + std::to_string(info_.pack_factor);
    annotate_values.push_back(StringImm::make(attr));

    return ExternModule::make(op->attr_key, value, body, annotate_keys,
                              annotate_values);
  }

  // Mutate the function argument
  Stmt Mutate_(const KernelDef* op, const Stmt& s) override {
    has_autosa_module = false;
    Stmt body = this->Mutate(op->body);
    Array<VarExpr> args;
    Array<Array<Expr>> arg_shapes;
    Array<Expr> arg_types;

    for (size_t k = 0; k < op->args.size(); k++) {
      auto name = op->args[k].get()->name_hint;
      if (name == info_.name && !has_autosa_module) {
        // Create arg with same node
        VarExpr new_var(info_.name, info_.type);
        args.push_back(new_var);
        arg_shapes.push_back(info_.target_shape);
        string type = Type2Str(info_.type);
        arg_types.push_back(StringImm::make(type));
      } else {
        args.push_back(op->args[k]);
        arg_shapes.push_back(op->arg_shapes[k]);
        arg_types.push_back(op->arg_types[k]);
      }
    }
    has_autosa_module = false;
    return KernelDef::make(args, arg_shapes, arg_types, op->arg_tensors, body,
                           op->ret_void, op->ret_type, op->name,
                           op->attributes);
  }

  // Collect for loop information
  Stmt Mutate_(const For* op, const Stmt& s) override {
    range_[op->loop_var.get()] = Simplify(op->extent - 1);
    loop_iter_vars_.push_back(op->loop_var);
    Stmt stmt = IRMutator::Mutate_(op, s);
    return stmt;
  }

  Stmt Mutate_(const Store* op, const Stmt& s) {
    string target_tensor_name_ = info_.name;
    Array<Expr> shape_ = info_.origin_shape;
    if (target_tensor_name_ == op->buffer_var.get()->name_hint) {
      info_.is_written = true;
      if (info_.is_transpose) {
        auto indices = ExtractIndices(op->index, shape_, range_);
        std::reverse(indices.begin(), indices.end());
        auto new_index = FlattenIndices(indices, shape_);
        return Store::make(op->buffer_var, op->value, new_index, op->predicate);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    string target_tensor_name_ = info_.name;
    Array<Expr> shape_ = info_.origin_shape;

    if (info_.is_transpose) {
      if (target_tensor_name_ == op->buffer_var.get()->name_hint) {
        auto indices = ExtractIndices(op->index, shape_, range_);
        std::reverse(indices.begin(), indices.end());
        auto new_index = FlattenIndices(indices, shape_);
        return Load::make(op->type, op->buffer_var, new_index, op->predicate);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  std::unordered_map<const Variable*, Expr>& range_;
  std::vector<VarExpr>& loop_iter_vars_;
  TransformInfo& info_;
  bool has_autosa_module{false};
};

// Insert new buffer before anchor (producer) stage
Stmt InsertReshapeBuffer(Stmt s, TransformInfo& info,
                         unordered_map<string, TaskNode>& task_map_,
                         vector<string> kernel_input_names) {
  string producer = info.anchor_producer;
  string tensor_name = info.name;

  CHECK(task_map_.count(producer));
  bool is_top_arg = false;
  int arg_index = 0;
  for (auto v : kernel_input_names) {
    if (v == tensor_name) {
      is_top_arg = true;
      break;
    }
    arg_index++;
  }

  // TODO(hecmay): handles on-chip data packing as well
  if (is_top_arg) {
    HCL_DEBUG_LEVEL(2) << "    [ debug ] tensor " << tensor_name
                       << " is on top function interface";
    string target_producer = "test";
    TransformedBufferInserter tbi(target_producer, info);
    return tbi.Mutate(s);
  }
  return s;
}

// Update the buffer indices. If we want to
// tranpose, then reverse. Otherwise insert
// unpacking logic by default
Stmt UpdateBufferLayout(Stmt s, TransformInfo& info,
                        unordered_map<string, TaskNode>& task_map_,
                        vector<string> kernel_input_names) {
  string producer = info.anchor_producer;
  string tensor_name = info.name;
  Stmt stmt = s;

  CHECK(task_map_.count(producer));
  bool is_top_arg = false;
  int arg_index = 0;
  for (auto v : kernel_input_names) {
    if (v == tensor_name) {
      is_top_arg = true;
      break;
    }
    arg_index++;
  }

  // Update buffer access indices and kernel
  // function signature as well
  if (is_top_arg) {
    std::unordered_map<const Variable*, Expr> range_;
    std::vector<VarExpr> loop_iter_vars_;
    IndicesTransformer ivc(range_, loop_iter_vars_, info);
    stmt = ivc.Mutate(stmt);
  } else {
  }
  return stmt;
}

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
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
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
  unordered_map<string, Array<Expr>> shape_;
  unordered_map<string, Type> dtype_;
};

void CollectTypeShape(Stmt body, unordered_map<string, Array<Expr>>& shape,
                      unordered_map<string, Type>& dtype,
                      Array<NodeRef>& api_args) {
  HCL_DEBUG_LEVEL(2) << "---------- collect shape/dtype ---------";
  TypeShapeCollector tsc(api_args);
  tsc.Mutate(body);
  dtype = tsc.dtype_;
  shape = tsc.shape_;
}

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
  explicit TaskGraphBuilder(Array<NodeRef> api_args) {}

  Stmt Mutate_(const KernelDef* op, const Stmt& s) {
    if (op->name == "test") {
      device_scope_ = true;

      // Save the input tensors
      for (auto& v : op->args) {
        kernel_input_args.push_back(v);
      }
      Stmt body = this->Mutate(op->body);
      device_scope_ = false;
      return KernelDef::make(op->args, op->arg_shapes, op->arg_types,
                             op->arg_tensors, body, op->ret_void, op->ret_type,
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
        TaskNode task = {name, bsc.updated_tensors, bsc.input_tensors, {}, {}};

        // Checking depending input tensors
        Array<Var> kernel_input_vars;
        for (auto& input : kernel_input_args) {
          Var v(input.node_);
          kernel_input_vars.push_back(v);
        }
        Array<Var> undefs = UndefinedVars(body, kernel_input_vars);
        // The task can be a producer of a tensor, or it will just update
        // a set of tensors. If the input tensor is not defined in this task
        // nor in the input arguments, then it must have been defined in the
        // previous tasks visited in the traversal
        for (auto& var : undefs) {
          auto parents = checkTensorLiveness(var.get());
          for (auto& parent_task_name : parents) {
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
    for (auto& kv : task_map) {
      for (auto& t : kv.second.updated_tensors) {
        if (t == var) {
          HCL_DEBUG_LEVEL(2) << "[ debug ] Tensor " << var->name_hint
                             << " has been updated in task " << kv.second.name;
          parents.push_back(kv.second.name);
        }
      }
    }
    return parents;
  }

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
  explicit LayoutTransformer(unordered_map<string, TaskNode>& task_map,
                             Array<NodeRef>& api_args,
                             vector<string> kernel_inputs)
      : task_map_(task_map),
        api_args_(api_args),
        kernel_inputs_(kernel_inputs) {}

  unordered_map<string, TaskNode>& task_map_;
  Array<NodeRef>& api_args_;
  unordered_map<string, Array<Expr>> shape_;
  unordered_map<string, Type> dtype_;
  vector<string> kernel_inputs_;

  std::string current_producer;
  // Map from producer key to target tensor name
  unordered_map<string, TransformInfo> worklist;

  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) {
    current_producer = op->func->func_name();
    return IRMutator::Mutate_(op, s);
  }

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
      Array<Expr> target_shape;

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
      target_shape.push_back(std::stoi(s));

      // Check tranform type (tranpose or packing)
      int origin_total_width = 1;
      for (auto& dim : shape_.at(name)) {
        CHECK(dim.as<IntImm>());
        origin_total_width *= dim.as<IntImm>()->value;
      }

      // TODO(Hecmay): handle reshape
      if (origin_total_width == target_total_width) {
        HCL_DEBUG_LEVEL(2) << "[ debug ] Transpose layout of tensor " << name
                           << "(" << shape_[name] << ") to (" << target_shape
                           << ")";

        CHECK(dtype_.count(name));
        TransformInfo info = {name,
                              var,
                              current_producer,
                              shape_[name],
                              target_shape,
                              dtype_[name],
                              dtype_[name],
                              true,
                              false,
                              1,
                              false};

        if (!worklist.count(name)) {
          worklist[name] = info;
          // The tensor has been packed
          // Recalculate the packing shape
        } else {
          Array<Expr> new_shape;
          int shape_size = info.target_shape.size();
          for (int k = 0; k < shape_size; k++) {
            if (k == shape_size - 1) {
              int factor = worklist[name].pack_factor;
              int new_dim = info.target_shape[k].as<IntImm>()->value;
              new_dim /= factor;
              new_shape.push_back(new_dim);
            } else {
              new_shape.push_back(info.target_shape[k]);
            }
          }
          worklist[name].target_shape = new_shape;
          worklist[name].is_transpose = true;
        }

      } else {
        // Pack the last dimension by default
        int pack_factor = origin_total_width / target_total_width;
        HCL_DEBUG_LEVEL(2) << "[ debug ] Pack layout of tensor " << name << "("
                           << shape_[name] << ") to (" << op->value << ")";

        Type new_type = Int(dtype_[name].bits() * pack_factor);
        TransformInfo info = {name,         var,          current_producer,
                              shape_[name], target_shape, dtype_[name],
                              new_type,     false,        true,
                              pack_factor,  false};

        if (!worklist.count(name)) {
          worklist[name] = info;
          // if the target has been transposed
          // first tranpose and then data-packing
        } else {
          Array<Expr> new_shape;
          int shape_size = worklist[name].target_shape.size();
          for (int k = 0; k < shape_size; k++) {
            if (k == shape_size - 1) {
              int factor = pack_factor;
              int new_dim = worklist[name].target_shape[k].as<IntImm>()->value;
              new_dim /= factor;
              new_shape.push_back(new_dim);
            } else {
              new_shape.push_back(worklist[name].target_shape[k]);
            }
          }
          worklist[name].target_shape = new_shape;
          worklist[name].type = Int(pack_factor * worklist[name].type.bits());
          worklist[name].is_pack = true;
          worklist[name].pack_factor = pack_factor;
        }
      }

      return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Transform(Stmt s) {
    CollectTypeShape(s, shape_, dtype_, api_args_);
    Stmt stmt = this->Mutate(s);
    // Process the worklist one by one
    for (auto& kv : worklist) {
      auto tensor_name = kv.first;
      auto& info = kv.second;
      auto producer_name = info.anchor_producer;
      CHECK(shape_.count(tensor_name)) << tensor_name;

      string status = "Processing tensor " + tensor_name + "(pack:";
      status += info.is_pack ? std::to_string(info.pack_factor) : "no";
      status += ", transpose:";
      status += info.is_transpose ? "yes)" : "no)";

      HCL_DEBUG_LEVEL(2) << "--------------";
      HCL_DEBUG_LEVEL(2) << "[ INFO ] " << status << ". shape "
                         << info.target_shape << ", type " << info.type;

      VarExpr new_buf(tensor_name + ".new");
      HCL_DEBUG_LEVEL(2) << "    [ debug ] transform layout of tensor "
                         << tensor_name << " from stage " << producer_name;

      // Mutate tensor access indices from all children stages
      stmt = UpdateBufferLayout(stmt, info, task_map_, kernel_inputs_);
      // Insert new buffer before anchor stage
      CHECK(task_map_.count(producer_name));
      stmt = InsertReshapeBuffer(stmt, info, task_map_, kernel_inputs_);
    }
    return stmt;
  }
};

Stmt TransformLayout(Stmt stmt, Array<NodeRef> api_args) {
  // Restore the task graph from the IR
  HCL_DEBUG_LEVEL(2) << "------------ Transform Layout --------------";
  TaskGraphBuilder tgb(api_args);
  stmt = tgb.Mutate(stmt);

  // Iterate thru tensors in worklist (to be transposed or packed)
  vector<string> kernel_inputs;
  for (auto& arg : tgb.kernel_input_args) {
    kernel_inputs.push_back(arg.get()->name_hint);
  }
  LayoutTransformer ltm(tgb.task_map, api_args, kernel_inputs);
  stmt = ltm.Transform(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace TVM
