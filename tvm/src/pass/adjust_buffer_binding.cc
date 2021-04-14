/*!
 *  Copyright (c) 2019 by Contributors
 * \file adjust_buffer_binding.cc
 */
#include <arithmetic/Substitute.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

namespace TVM {
namespace ir {

class BufferBindingAdjuster final : public IRMutator {
 public:
  BufferBindingAdjuster(std::map<const Variable*, Array<Expr> >& shape_map,
                        std::map<const Variable*, VarExpr>& buffer_map,
                        Array<Var> undefined_vars)
      : shape_map_(shape_map),
        buffer_map_(buffer_map),
        undefined_vars_(undefined_vars) {
    for (auto& kv : shape_map_) {
      std::string name = kv.first->name_hint;
      CHECK(buffer_map_.count(kv.first));
      VarExpr buf = buffer_map_.at(kv.first);
      name_var_map_[name] = VarExpr(buf.node_);
    }
  }

  Stmt Mutate_(const LetStmt* op, const Stmt& s) {
    HandleDef(op->var);
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Let* op, const Expr& e) {
    HandleDef(op->var);
    return this->Mutate(op->body);
  }

  Stmt Mutate_(const For* op, const Stmt& s) {
    HandleDef(op->loop_var);
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    HandleDef(op->buffer_var);
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const KernelDef* op, const Stmt& s) {
    std::map<std::string, VarExpr> name_var_map_save;
    std::map<const Variable*, Array<Expr> > shape_map_save;
    name_var_map_save.clear();
    shape_map_save.clear();
    name_var_map_save = name_var_map_;
    shape_map_save = shape_map_;
    name_var_map_.clear();
    shape_map_.clear();

    for (auto& arg : op->args) {
      HCL_DEBUG_LEVEL(2) << "[ adjust buffer ] register kernel arg " << arg;
      HandleDef(arg);
    }
    inside_function = true;
    func_name = op->name;
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<KernelDef>();

    name_var_map_.clear();
    shape_map_.clear();
    name_var_map_ = name_var_map_save;
    shape_map_ = shape_map_save;

    inside_function = false;
    return stmt;
  }

  // Usage after definition
  Stmt Mutate_(const Store* op, const Stmt& s) {
    if (HandleUse(op->buffer_var)) {
      HCL_DEBUG_LEVEL(2) << "Undefined Store buffer: " << s;
      auto buffer_name = op->buffer_var.get()->name_hint;
      if (!name_var_map_.count(buffer_name)) {
        for (auto& kv : name_var_map_) {
          HCL_DEBUG_LEVEL(2) << kv.first;
        }
        if (inside_function) {
          auto new_name = "_top." + func_name + "." + buffer_name;
          if (name_var_map_.count(new_name)) {
            VarExpr new_buf(name_var_map_[new_name].node_);
            return Store::make(new_buf, op->value, op->index, op->predicate);
          }
        }
      }
      CHECK(name_var_map_.count(buffer_name));
      VarExpr new_buf(name_var_map_[buffer_name].node_);
      return Store::make(new_buf, op->value, op->index, op->predicate);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const StreamStmt* op, const Stmt& s) {
    if (HandleUse(op->buffer_var)) {
      HCL_DEBUG_LEVEL(2) << "Undefined StreamStmt buffer: " << s;
      auto buffer_name = op->buffer_var.get()->name_hint;
      CHECK(name_var_map_.count(buffer_name)) << buffer_name;

      VarExpr new_buf(name_var_map_[buffer_name].node_);
      HCL_DEBUG_LEVEL(2) << "    Replace " << op->buffer_var << "("
                         << op->buffer_var.get() << ") with " << new_buf << "("
                         << new_buf.get() << ")";
      return StreamStmt::make(new_buf, op->index, op->value, op->axis,
                              op->stream_type, op->depth, op->annotate_keys,
                              op->annotate_values);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable* op, const Expr& e) {
    if (HandleUse(e)) {
      HCL_DEBUG_LEVEL(2) << "Undefined Variable buffer: " << e;
      auto buffer_name = op->name_hint;
      CHECK(name_var_map_.count(buffer_name)) << buffer_name;
      VarExpr new_buf(name_var_map_[buffer_name].node_);
      return new_buf;
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Partition* op, const Stmt& s) {
    if (HandleUse(op->buffer_var)) {
      HCL_DEBUG_LEVEL(2) << "Undefined Partition buffer: " << s;
      auto buffer_name = op->buffer_var.get()->name_hint;
      CHECK(name_var_map_.count(buffer_name)) << buffer_name;

      VarExpr new_buf(name_var_map_[buffer_name].node_);
      HCL_DEBUG_LEVEL(2) << "    Replace " << op->buffer_var << "("
                         << op->buffer_var.get() << ") with " << new_buf << "("
                         << new_buf.get() << ")";
      return Partition::make(new_buf, op->dim, op->factor, op->partition_type);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    if (HandleUse(op->buffer_var)) {
      HCL_DEBUG_LEVEL(2) << "Undefined Load buffer: " << e;
      auto buffer_name = op->buffer_var.get()->name_hint;
      if (!name_var_map_.count(buffer_name)) {
        if (inside_function) {
          auto new_name = "_top." + func_name + "." + buffer_name;
          if (name_var_map_.count(new_name)) {
            VarExpr new_buf(name_var_map_[new_name].node_);
            return Load::make(op->type, new_buf, op->index, op->predicate);
          }
        }
      }
      CHECK(name_var_map_.count(buffer_name)) << buffer_name;
      VarExpr new_buf(name_var_map_[buffer_name].node_);
      HCL_DEBUG_LEVEL(2) << "    Replace " << op->buffer_var << "("
                         << op->buffer_var.get() << ") with " << new_buf << "("
                         << new_buf.get() << ")";
      return Load::make(op->type, new_buf, op->index, op->predicate);
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const StreamExpr* op, const Expr& e) {
    if (HandleUse(op->buffer_var)) {
      HCL_DEBUG_LEVEL(2) << "Undefined StreamExpr buffer: " << e;
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Reuse* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Reuse>();
    if (HandleUse(op->buffer_var)) {
      HCL_DEBUG_LEVEL(2) << "Undefined Reuse buffer: " << op->buffer_var;

      auto buffer_name = op->buffer_var.get()->name_hint;
      CHECK(name_var_map_.count(buffer_name));
      VarExpr new_buf(name_var_map_[buffer_name].node_);
      return Reuse::make(new_buf, op->body);

    } else {
      return stmt;
    }
  }

  Stmt Mutate_(const KernelStmt* op, const Stmt& s) final {
    Array<Expr> new_args;
    for (auto& e : op->args) {
      if (HandleUse(e)) {
        HCL_DEBUG_LEVEL(2) << "Undefined KernelStmt Arg: " << e;
        CHECK(e.as<Variable>());
        auto name = e.as<Variable>()->name_hint;
        CHECK(name_var_map_.count(name)) << name;
        Expr new_buf(name_var_map_[name].node_);
        new_args.push_back(new_buf);
      } else {
        new_args.push_back(e);
      }
    }
    return KernelStmt::make(new_args, op->name, op->annotate_keys,
                            op->annotate_values);
  }

  void HandleDef(const VarExpr& var) {
    const Variable* v = var.get();
    CHECK(!shape_map_.count(v))
        << "variable " << v->name_hint << " has been used before definition!";
    std::string name = v->name_hint;
    shape_map_[v] = {1};
    name_var_map_[name] = VarExpr(var.node_);
  }

  bool HandleUse(const Expr& v) {
    CHECK(v.as<Variable>());
    Var var(v.node_);
    auto it = shape_map_.find(var.get());
    if (it == shape_map_.end()) {
      return true;
    }
    return false;
  }

 private:
  std::map<std::string, VarExpr> name_var_map_;
  std::map<const Variable*, Array<Expr> >& shape_map_;
  bool inside_function{false};
  std::string func_name;

  std::map<const Variable*, VarExpr>& buffer_map_;
  Array<Var>& undefined_vars_;
};

Stmt AdjustBufferBinding(Stmt stmt, Array<NodeRef> arg_list) {
  std::map<const Variable*, Array<Expr> > shape_map;
  std::map<const Variable*, VarExpr> buffer_map;
  Array<Var> input_args;
  for (size_t i = 0; i < arg_list.size(); i++) {
    if (const BufferNode* node = arg_list[i].as<BufferNode>()) {
      shape_map[node->data.get()] = node->shape;
      input_args.push_back(node->data);
      buffer_map[node->data.get()] = node->data;
    } else {
      const Variable* v = arg_list[i].as<Variable>();
      CHECK(v) << "Illegal argument " << arg_list[i];
      Var input_var(arg_list[i].node_);
      shape_map[v] = {1};
      input_args.push_back(input_var);
      buffer_map[v] = input_var;
    }
  }
  Array<Var> undefined = UndefinedVars(stmt, input_args);
  if (undefined.size() > 0) {
    HCL_DEBUG_LEVEL(2) << "Fonud mismatching buffers in the stmt...";
    for (auto& v : undefined) {
      HCL_DEBUG_LEVEL(2) << "    " << v << "(" << v.get() << ")";
    }
  }
  BufferBindingAdjuster mutator(shape_map, buffer_map, undefined);
  stmt = mutator.Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace TVM
