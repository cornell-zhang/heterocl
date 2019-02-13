/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_ops.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule_pass.h>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "./graph.h"
#include "../op/op_util.h"
#include "../op/extern_op.h"
#include "../pass/ir_util.h"

namespace tvm {
namespace schedule {

using namespace ir;

Stmt MakePipeline(const Stage& s,
                  const std::unordered_map<IterVar, Range>& dom_map,
                  Stmt consumer,
                  bool del_trivial_loop) {
  Stmt producer = s->op->BuildProvide(s, dom_map, del_trivial_loop);
  if (producer.defined()) {
    producer = ProducerConsumer::make(s->op, true, producer);
  }
  if (s->double_buffer) {
    producer = AttrStmt::make(
        s->op, ir::attr::double_buffer_scope, 1, producer);
  }
  Stmt pipeline = producer;

  if (consumer.defined() && !is_no_op(consumer)) {
    consumer = ProducerConsumer::make(s->op, false, consumer);
    pipeline = Block::make(producer, consumer);
  }
  pipeline = s->op->BuildRealize(s, dom_map, pipeline);
  // use attribute to mark scope of the operation.
  pipeline = AttrStmt::make(
      s->op, ir::attr::realize_scope,
      StringImm::make(s->scope),
      pipeline);

  if (s->is_opengl) {
    pipeline = AttrStmt::make(
        s->op, ir::attr::opengl_stage_scope, StringImm::make(""), pipeline);
  }
  return pipeline;
}

class InjectStmt : public IRMutator {
  public:
    InjectStmt(const Stage& stage,
               const std::unordered_map<IterVar, Range>& dom_map) 
      : stage_(stage), dom_map_(dom_map) {}

    Stmt inject(Stmt stmt) {
      stmt = this->Mutate(stmt);
      if (!inserted) {
        stmt = MakePipeline(stage_, dom_map_, stmt, true);
      }
      return stmt;
    }

    Stmt Mutate(Stmt stmt) final {
      CHECK(stmt.defined());
      stmt = IRMutator::Mutate(stmt);
      const AttrStmt* op = stmt.as<AttrStmt>();
      if (op != nullptr) {
        if (op->attr_key == attr::attach_scope) {
          const ExternOpNode* node = stage_->op.as<ExternOpNode>();
          if (op->node == node->output_placeholders[0]) {
            stmt = MakePipeline(stage_, dom_map_, op->body, true);
            inserted = true;
          }
        } else if(op->attr_key == attr::buffer_bind_scope) {
          Array<NodeRef> arr(op->node.node_);
          CHECK_EQ(arr.size(), 2U);
          const BufferNode* buffer = arr[0].as<BufferNode>();
          const ExternOpNode* ext_op = stage_->op.as<ExternOpNode>();
          if (ext_op != nullptr) {
            bool remove = false;
            for (auto b : ext_op -> output_placeholders) {
              const BufferNode* buf = b.as<BufferNode>();
              if (buf == buffer) remove = true;
            }
            if (remove) stmt = op->body;
          }
        }
      }
      return stmt;
    }
  private:
    const Stage& stage_;
    const std::unordered_map<IterVar, Range>& dom_map_;
    bool inserted{false};
};

// inject the operator's realization on the stmt.
class InjectAttach : public IRMutator {
 public:
  InjectAttach(const Stage& stage,
               const Stage& attach_spec,
               const std::unordered_map<IterVar, Range>& dom_map,
               const Schedule& sch)
      : stage_(stage), attach_spec_(attach_spec), dom_map_(dom_map), sch_(sch) {}

  Stmt Mutate(Stmt stmt) final {
    CHECK(stmt.defined());
    stmt = IRMutator::Mutate(stmt);
    const AttrStmt* op = stmt.as<AttrStmt>();
    if (op != nullptr) {
      if (op->attr_key == attr::attach_scope) {
        if (stage_->attach_ivar == op->node) {
          CHECK(!found_attach)
            << "Find IterVar" << attach_spec_->attach_ivar
            << " in multiple places in the IR";
          found_attach = true;
          stmt = MakePipeline(stage_, dom_map_, op->body, true);
          /*
          stmt = AttrStmt::make(
              op->node, op->attr_key, op->value,
              MakePipeline(stage_, dom_map_, op->body, true));
          */
        }
      }
      else if(op->attr_key == attr::buffer_bind_scope) {
        Array<NodeRef> arr(op->node.node_);
        CHECK_EQ(arr.size(), 2U);
        const BufferNode* buffer = arr[0].as<BufferNode>();
        const ExternOpNode* ext_op = stage_->op.as<ExternOpNode>();
        if (ext_op != nullptr) {
          bool remove = false;
          for (auto b : ext_op -> output_placeholders) {
            const BufferNode* buf = b.as<BufferNode>();
            if (buf == buffer) remove = true;
          }
          if (remove) stmt = op->body;
        }
      }
    }
    return stmt;
  }
  // whether attach point is found
  bool found_attach{false};

 private:
  // The stage.
  const Stage& stage_;
  // The attach spec, may not contain op.
  const Stage& attach_spec_;
  // domain map
  const std::unordered_map<IterVar, Range>& dom_map_;
  const Schedule sch_;
};

// Postprocessing of schedule op
class SchedulePostProc : public IRMutator {
 public:
  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) final {
    auto it = replace_op_.find(op->func.get());
    if (it != replace_op_.end()) {
      Stmt body = this->Mutate(op->body);
      if (it->second.defined()) {
        return ProducerConsumer::make(
            it->second, op->is_producer, body);
      } else {
        return body;
      }
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }
  Stmt Mutate_(const LetStmt* op, const Stmt& s) final {
    if (!HasSideEffect(op->value)) {
      var_value_[op->var.get()] = Mutate(op->value);
      return this->Mutate(op->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::loop_scope) {
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::thread_extent) {
      // delete duplicated thread extent attr
      auto it = thread_extent_scope_.find(op->node.get());
      if (it != thread_extent_scope_.end()) {
        CHECK(is_zero(ir::Simplify(it->second - op->value)));
        return this->Mutate(op->body);
      } else {
        thread_extent_scope_[op->node.get()] = op->value;
        Stmt ret = IRMutator::Mutate_(op, s);
        thread_extent_scope_.erase(op->node.get());
        return ret;
      }
    } else if (op->attr_key == ir::attr::realize_scope ||
               op->attr_key == ir::attr::double_buffer_scope) {
      auto it = replace_op_.find(op->node.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          Stmt ret = AttrStmt::make(
              it->second, op->attr_key, op->value, op->body);
          return this->Mutate(ret);
        } else {
          return this->Mutate(op->body);
        }
      }
    } else if (op->attr_key == ir::attr::buffer_bind_scope) {
      Array<NodeRef> tuple(op->node.node_);
      Tensor tensor(tuple[1].node_);
      auto it = replace_op_.find(tensor->op.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          return AttrStmt::make(
              Array<NodeRef>{tuple[0], it->second.output(tensor->value_index)},
              op->attr_key, op->value, Mutate(op->body));
        } else {
          return this->Mutate(op->body);
        }
      }
    } else if (op->attr_key == ir::attr::buffer_dim_align) {
      Tensor tensor(op->node.node_);
      auto it = replace_op_.find(tensor->op.get());
      if (it != replace_op_.end()) {
        if (it->second.defined()) {
          return AttrStmt::make(
              it->second.output(tensor->value_index),
              op->attr_key, op->value, Mutate(op->body));
        } else {
          return this->Mutate(op->body);
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize* op, const Stmt& s) final {
    TensorKey key{op->func, op->value_index};
    auto it = replace_realize_.find(key);
    if (it != replace_realize_.end()) {
      if (it->second.defined()) {
        Stmt ret = Realize::make(
            it->second->op, it->second->value_index,
            op->type, op->bounds, op->condition, op->body);
        return this->Mutate(ret);
      } else {
        return this->Mutate(op->body);
      }
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const Provide* op, const Stmt& s) final {
    TensorKey key{op->func, op->value_index};
    auto it = replace_buffer_.find(key);
    if (it != replace_buffer_.end()) {
      const Tensor& dst = it->second;
      Stmt ret = Provide::make(
          dst->op, dst->value_index, op->value, op->args);
      return this->Mutate(ret);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Expr Mutate_(const Call* op, const Expr& e) final {
    if (op->call_type == Call::Halide) {
      TensorKey key{op->func, op->value_index};
      auto it = replace_buffer_.find(key);
      if (it != replace_buffer_.end()) {
        const Tensor& dst = it->second;
        Expr ret = Call::make(
            op->type, dst->op->name, op->args,
            op->call_type, dst->op, dst->value_index);
        return this->Mutate(ret);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = var_value_.find(op);
    if (it != var_value_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr index = op->index;
    auto it = axis_remain_load_.find(op->buffer_var.get());
    if (it != axis_remain_load_.end()) {
      std::vector<IterVar> vars_in_index_remain = GetIterVarsInIndexRemain(index, it->second);
      index = MakeIndexFromIterVars(vars_in_index_remain);
    }
    Expr pred = this->Mutate(op->predicate);
    if (index.same_as(op->index) && pred.same_as(op->predicate)) {
      return e;
    } else {
      return Load::make(op->type, op->buffer_var, index, pred);
    }
  }

  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Expr index = op->index;
    auto it = axis_remain_store_.find(op->buffer_var.get());
    if (it != axis_remain_store_.end()) {
      std::vector<IterVar> vars_in_index_remain = GetIterVarsInIndexRemain(index, it->second);
      index = MakeIndexFromIterVars(vars_in_index_remain);
    }
    Expr pred = this->Mutate(op->predicate);
    Expr value = this->Mutate(op->value);
    if (index.same_as(op->index) && pred.same_as(op->predicate && value.same_as(op->value))) {
      return s;
    } else {
      return Store::make(op->buffer_var, value, index, pred);
    }
  }

  void Init(const Schedule& sch) {
    for (Stage s : sch->stages) {
      for (auto kv : s->iter_var_attrs) {
        // Update bind thread information.
        if (kv.second->bind_thread.defined()) {
          const Var& from = kv.first->var;
          const Var& to = kv.second->bind_thread->var;
          CHECK(!var_value_.count(from.get()));
          var_value_[from.get()] = to;
        }
      }
      // This must be checked for all ops, including scan.
      if (!s->op.same_as(s->origin_op)) {
        for (int i = 0; i < s->op->num_outputs(); ++i) {
          Tensor target = s->origin_op.output(i);
          AddReplace(s->op.output(i), target,
                     target, s->origin_op);
        }
      }
      // Special handle for extern op
      if (s->op.as<ExternOpNode>()) {
        const ExternOpNode* extern_node = s->op.as<ExternOpNode>();
        if (s->attach_ivar.defined()) {
          int axis_size = extern_node->axis.size();
          int attach_level = CountAttachLevel(s);
          auto tmp_load = GetAxisOuterLoadRemain(s, axis_size, attach_level);
          auto tmp_store = GetAxisInnerStoreRemain(s, axis_size, attach_level);
          for (auto x : tmp_load) {
            axis_remain_load_[x.first] = x.second;
          }
          for (auto x : tmp_store) {
            axis_remain_store_[x.first] = x.second;
          }
        }
      }
    }
  }

 private:
  void AddReplace(Tensor src,
                  Tensor dst,
                  Tensor repl_realize = Tensor(),
                  Operation repl_op = Operation()) {
    TensorKey key{src->op, src->value_index};
    replace_buffer_[key] = dst;
    replace_realize_[key] = repl_realize;
    replace_op_[src->op.get()] = repl_op;
  }
  // The thread extent scope.
  std::unordered_map<const Node*, Expr> thread_extent_scope_;
  // The scan value
  std::unordered_map<const Variable*, Expr> var_value_;
  // buffer replacement
  std::unordered_map<TensorKey, Tensor> replace_buffer_;
  // buffere realization to be replaced
  std::unordered_map<TensorKey, Tensor> replace_realize_;
  // replace producer consumer.
  std::unordered_map<const Node*, Operation> replace_op_;
  // The iter vars that remain, associated with buffer var.
  std::unordered_map<const Variable*, std::vector<IterVar> > axis_remain_load_;
  // The IterVars that are outside ...
  std::unordered_map<const Variable*, std::vector<IterVar> > axis_remain_store_;
};

Stmt ScheduleOps(
    Schedule sch, Map<IterVar, Range> dom_map_, bool del_trivial_loop) {
  Stmt body = Stmt();
  std::unordered_map<IterVar, Range> dom_map = as_unordered_map(dom_map_);
  // verify correctness of group.
  for (Stage g : sch->groups) {
    CHECK(!g->op.defined());
    CHECK_EQ(g->leaf_iter_vars.size(), 0U);
  }
  // reverse the post DFS order.
  for (size_t i = sch->stages.size(); i != 0; --i) {
    Stage s = sch->stages[i - 1];
    CHECK_NE(s->attach_type, kInline)
        << "call schedule.normalize before scheduleops";
    CHECK(s->op.defined());
    // no need to specify place holder op.
    if (s->op.as<PlaceholderOpNode>()) continue;
    // Remove grouping sugar, get the real attach spec.
    Stage attach_spec = s.GetAttachSpec();

    if (attach_spec->attach_type == kInlinedAlready) {
      // do nothing
    } else if (attach_spec->attach_type == kGroupRoot) {
      CHECK(!s->group.defined());
      if (body.defined()) {
        InjectStmt mutator(s, dom_map);
        body = mutator.inject(body);
      } else {
        body = MakePipeline(s, dom_map, body, del_trivial_loop);
      }
    } else {
      CHECK_EQ(attach_spec->attach_type, kScope);
      CHECK(body.defined());
      InjectAttach mutator(s, attach_spec, dom_map, sch);
      body = mutator.Mutate(body);
      CHECK(mutator.found_attach)
          << "did not find attachment point for " << s << " in "
          << attach_spec->attach_stage->op  << " x " << attach_spec->attach_ivar
          << ", body:\n"
          << body;
    }
  }
  SchedulePostProc post_proc;
  post_proc.Init(sch);
  return post_proc.Mutate(body);
}

}  // namespace schedule
}  // namespace tvm
