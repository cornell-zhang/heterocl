/*!
 *  Copyright (c) 2019 by Contributors
 * \file loop_partition.cc
 */
#include <arithmetic/Substitute.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

namespace TVM {
namespace ir {

// Lifting up the Partition IR node
class PartitionLifter final : public IRMutator {
 public:
  PartitionLifter() {}

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) {
    Stmt body = this->Mutate(op->body);
    if (op->attr_key == attr::storage_scope) {
      if (remove_attr_) {
        remove_attr_ = false;
        return body;
      }
    }
    return AttrStmt::make(op->node, op->attr_key, op->value, body);
  }

  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    // find if we have a corresponding partition node
    const Variable* var = op->buffer_var.get();
    allocate_vars_.push_back(var);
    Stmt body = this->Mutate(op->body);

    if (!body.defined() && attr_found_) {
      attr_found_ = false;
      return Evaluate::make(0);

      // remove attr stmt if it has partition node inside
    } else if (const Block* block = body.as<Block>()) {
      if (block->first.as<Partition>()) {
        remove_attr_ = true;
        return body;
      }
    }
    // push partition ir into allocate::attrs
    if (allocate_attrs_.count(var)) {
      Array<Stmt> attrs = op->attrs;
      attrs.push_back(allocate_attrs_[var]);
      return Allocate::make(op->buffer_var, op->type, op->extents,
                            op->condition, body, attrs, op->new_expr,
                            op->free_function, op->init_values, op->is_const);
    }
    return Allocate::make(op->buffer_var, op->type, op->extents, op->condition,
                          body, op->attrs, op->new_expr, op->free_function,
                          op->init_values, op->is_const);
  }

  Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) {
    if (op->is_producer) {
      if (const AttrStmt* attr_extern = op->body.as<AttrStmt>()) {
        if (const Partition* partition = attr_extern->body.as<Partition>()) {
          bool found = false;
          const Variable* var = partition->buffer_var.get();
          for (auto v : allocate_vars_) {
            if (v == var) {
              found = true;
              break;
            }
          }

          // remove attr atop partition ir
          if (found) {
            allocate_attrs_[var] = attr_extern->body;
            attr_found_ = true;
            return Evaluate::make(0);
          } else {
            return attr_extern->body;
          }
        }  // if partition
      }    // if attr_stmt
    }      // if is_producer
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::vector<const Variable*> allocate_vars_;
  std::map<const Variable*, Stmt> allocate_attrs_;
  bool attr_found_{false};
  bool remove_attr_{false};
};

Stmt LiftAllocateAttrs(Stmt stmt) {
  PartitionLifter mutator;
  stmt = mutator.Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace TVM
