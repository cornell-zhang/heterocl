/*!
 *  Copyright (c) 2019 by Contributors
 * \file loop_partition.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>
#include <arithmetic/Substitute.h>

namespace tvm {
namespace ir {

Expr calculate_index(std::vector<Expr> indices, const Array<Expr> shape) {
  Expr ret = indices[0];
  Expr mul = 1;
  for (size_t i = 1; i < indices.size(); i++) {
    mul = Simplify(mul * shape[i-1]);
    ret = Simplify(ret + indices[i] * mul);
  }
  return ret;
}

// recover a 1D index back to multi-dimensional index
std::vector<Expr> recover_index(Expr index, const Array<Expr>& shape) {
  std::vector<Expr> new_index;
  for (size_t i = 0; i < shape.size() - 1; i++) {
    Expr simple_index = Simplify(index % shape[i]);
    // remove modulo
    const Mod* op = simple_index.as<Mod>();
    simple_index = op->a;
    new_index.push_back(simple_index);
    // simplify the rest
    index = Simplify((index - simple_index) / shape[i]);
  }
  new_index.push_back(index);
  return new_index;
}

// collect all load indices that contains the reuse target
class LoadExpressionCollector final : public IRVisitor {
  public:
    LoadExpressionCollector(
        const VarExpr& target, 
        std::vector<std::vector<Expr> >& expr_list,
        std::map<const Variable*, Expr>& min_map,
        std::map<const Variable*, Expr>& max_map,
        std::map<const Variable*, Array<Expr> >& shape_map) 
      : target_(target), expr_list_(expr_list),
        min_map_(min_map), max_map_(max_map), 
        shape_map_(shape_map) {};

    void Visit_(const Load* op) {
      this->Visit(op->index);
      if (op->buffer_var.get() == target_.get()) {
        LOG(INFO) << op->index;
        Array<Expr> shape = shape_map_[op->buffer_var.get()];
        std::vector<Expr> new_index = recover_index(op->index, shape);
        for (size_t i = 0; i < new_index.size(); i++)
          LOG(INFO) << new_index[i];
        expr_list_.push_back(new_index);
      }
    }

    void Visit_(const For* op) {
      min_map_[op->loop_var.get()] = op->min;
      max_map_[op->loop_var.get()] = op->extent - 1;
      this->Visit(op->body);
    }

  private:
    const VarExpr& target_;
    // a list of indices; each index is represented with a tuple
    // e.g., [[x, y], [x+1, y], [x+2, y]]
    // e.g., [[x+r, y+c]]
    std::vector<std::vector<Expr> >& expr_list_;
    // key, value = loop_var, min
    std::map<const Variable*, Expr>& min_map_;
    // key, value = loop_var, extent-1
    std::map<const Variable*, Expr>& max_map_;
    std::map<const Variable*, Array<Expr> >& shape_map_;
};

class ProduceBodyReplacer final : public IRMutator {
  public:
    ProduceBodyReplacer(const Stmt& replace_stmt, 
                        const VarExpr& target,
                        const VarExpr& reuse,
                        const Array<Expr>& target_shape,
                        const Array<Expr>& reuse_shape,
                        const std::map<const Variable*, Expr>& null_axis_subst)
      : replace_stmt_(replace_stmt), target_(target), reuse_(reuse),
        target_shape_(target_shape), reuse_shape_(reuse_shape),
        null_axis_subst_(null_axis_subst) {};

    Stmt Mutate_(const ProducerConsumer* op, const Stmt& s) {
      // replace the nearest producer
      if (op->is_producer && !replaced_) {
        replaced_ = true;
        return ProducerConsumer::make(op->func, op->is_producer, replace_stmt_);
      } else {
        return IRMutator::Mutate_(op, s);
      }
    }

    Expr Mutate_(const Load* op, const Expr& e) {
      Expr index = this->Mutate(op->index);
      if (op->buffer_var.get() == target_.get()) {
        // need to recalculate the index according to the new shape
        index = Simplify(substitute(null_axis_subst_, index));
        std::vector<Expr> new_indices = recover_index(index, target_shape_);
        index = calculate_index(new_indices, reuse_shape_);
        return Load::make(op->type, reuse_, index, op->predicate);
      } else {
        // recursively 
        return e;
      }
    }

  private:
    const Stmt& replace_stmt_;
    const VarExpr& target_;
    const VarExpr& reuse_;
    const Array<Expr>& target_shape_;
    const Array<Expr>& reuse_shape_;
    const std::map<const Variable*, Expr>& null_axis_subst_;
    bool replaced_{false};
};


class ReuseBufferInserter final : public IRMutator {
  public:
    ReuseBufferInserter(std::map<const Variable*, Array<Expr> >& shape_map) 
      : shape_map_(shape_map) {};

    Stmt Mutate_(const For* op, const Stmt& s) {
      null_axis_subst_[op->loop_var.get()] = 0;
      if (const Reuse* node = op->body.as<Reuse>()) {
        LOG(INFO) << node->buffer_var.get();
        VarExpr target = node->buffer_var;
        Array<Expr> target_shape = shape_map_[target.get()];
        Stmt body = op->body;
        // collect load expression related to the target
        std::vector<std::vector<Expr> > expr_list;
        std::vector<std::vector<Expr> > diff_list;
        std::vector<Expr> min_list;
        std::map<const Variable*, Expr> min_map;
        std::map<const Variable*, Expr> max_map;
        LoadExpressionCollector visitor(target, 
                                        expr_list, 
                                        min_map, 
                                        max_map,
                                        shape_map_);
        visitor.Visit(body);
        int reuse = -1;
        // find the min_expr and max_expr for each dimension
        Array<Expr> reuse_shape;
        size_t ndim = expr_list[0].size();
        for (size_t dim = 0; dim < ndim; dim++) {
          // find the bound
          // e.g. x+r with {r=[0, 2], c=[0, 2], x=[0, 7]}
          // min_expr = 0, max_expr = 9
          // e.g. y+c with {r=[0, 2], c=[0, 2], x=[0, 7]}
          // min_expr = y, max_expr = y+2
          // e.g. [x, x+1, x+2] with {}
          // min_expr = x, max_expr = x+2
          Expr min_expr = substitute(min_map, expr_list[0][dim]);
          Expr max_expr = substitute(max_map, expr_list[0][dim]);
          size_t min_index = 0;
          for (size_t i = 1; i < expr_list.size(); i++) {
            Expr new_min_expr = substitute(min_map, expr_list[i][dim]);
            Expr new_max_expr = substitute(max_map, expr_list[i][dim]);
            Expr min_diff = Simplify(min_expr - new_min_expr);
            Expr max_diff = Simplify(new_max_expr - max_expr);
            if (!is_const(min_diff) || !is_const(max_diff))
              LOG(FATAL) << "The bound of the reuse region cannot be determined";
            if (is_one(Simplify(min_diff > 0))) {
              min_index = i;
              min_expr = new_min_expr;
            }
            if (is_one(Simplify(max_diff > 0))) max_expr = new_max_expr;
          }
          // check if the bounde is constant
          // e.g. x+r => diff_expr = 10
          // e.g. y+c => diff_expr = 3
          Expr diff_expr = Simplify(max_expr - min_expr + 1);
          if (!is_const(diff_expr)) // e.g. y*(y+c) would be illegal
            LOG(FATAL) << "Irregular access pattern is not yet supported";
          // check if the specified axis is reused by running the next iteration
          std::map<const Variable*, Expr> next_subst;
          next_subst[op->loop_var.get()] = op->loop_var + 1;
          // first check if the axis is the specified reuse axis
          // e.g. y => y+1
          Expr next_min = substitute(next_subst, min_expr);
          Expr next_diff = Simplify(next_min - min_expr);
          if (!is_const(next_diff)) // e.g. y*y+c would be illegal
            LOG(FATAL) << "Irregular access pattern is not yet supported";
          // then check if we there is reuse in this axis
          // e.g. y+c => incr_index_diff = 1
          if (!is_zero(next_diff)) {
            if (!is_one(next_diff)) // e.g. 2*y+c would be illegal
              LOG(FATAL) << "Irregular access pattern is not yet supported";
            // check if there is overlap between reuse axis
            // e.g. next_min = y+1, max_incr = y+2
            Expr compare = Simplify(max_expr > next_min);
            if (is_zero(compare))
              LOG(FATAL) << "No reuse is found in axis " << op->loop_var; 
            reuse = dim;
          }
          if (auto imm = diff_expr.as<IntImm>())
            diff_expr = IntImm::make(Int(32), imm->value);
          LOG(INFO) << diff_expr;
          reuse_shape.push_back(diff_expr);
          min_list.push_back(expr_list[min_index][dim]);
        } // end for each dim
        if (reuse == -1)
          LOG(FATAL) << "No reuse dimension found in the body";

        // build the updating function for the reuse buffer
        // the main update function is LB[reuse_indices] = IN[orgin_indices]
        // collect reuse_indices
        std::vector<Expr> reuse_indices;
        std::vector<Expr> update_indices;
        std::vector<Expr> shift_indices;
        std::vector<VarExpr> reuse_loop_vars;
        for (size_t dim = 0; dim < ndim; dim++) {
          Expr index = min_list[dim];
          Expr reuse_index = Simplify(substitute(null_axis_subst_, index));
          // create a new variable if the shape is not one
          if (!is_one(reuse_shape[dim])) {
            VarExpr new_loop_var(target->name_hint + "." + std::to_string(dim)); // TODO: fix the name
            // replace the RHS with the new loop var
            Expr rhs = substitute(reuse_index, new_loop_var, index);
            // special case when the reuse index is 0
            if (is_zero(reuse_index)) rhs = rhs + new_loop_var;
            reuse_indices.push_back(new_loop_var);
            reuse_loop_vars.push_back(new_loop_var);
            update_indices.push_back(rhs);
            LOG(INFO) << new_loop_var;
            LOG(INFO) << rhs;
          } else {
            reuse_indices.push_back(0);
            reuse_loop_vars.push_back(VarExpr());
            update_indices.push_back(index);
            LOG(INFO) << index;
          }
          if (dim == reuse)
            shift_indices.push_back(reuse_indices[dim] + 1);
          else
            shift_indices.push_back(reuse_indices[dim]);
        }
        // build the for loop
        const AttrStmt* attr_alloc = node->body.as<AttrStmt>();
        const Allocate* alloc = attr_alloc->body.as<Allocate>();
        // 1. build the if branch
        Expr reuse_index = calculate_index(reuse_indices, reuse_shape);
        Expr update_index = calculate_index(update_indices, target_shape);
        Expr predicate = UIntImm::make(UInt(1), 1);
        LOG(INFO) << reuse_index;
        LOG(INFO) << update_index;
        Stmt update_store = Store::make(
            alloc->buffer_var,
            Load::make(alloc->type, target, update_index, predicate),
            reuse_index,
            predicate);
        // 2. build the else branch -- shift operation
        Expr shift_index = calculate_index(shift_indices, reuse_shape);
        Stmt shift_store = Store::make(
            alloc->buffer_var,
            Load::make(alloc->type, alloc->buffer_var, shift_index, predicate),
            reuse_index,
            predicate);
        // 3. build the if statement
        Expr reuse_bound = Simplify(reuse_shape[reuse] - 1);
        Stmt if_stmt = IfThenElse::make(
            Or::make(op->loop_var == 0, reuse_indices[reuse] == reuse_bound),
            update_store,
            shift_store);
        // 4. build the for loops
        Stmt for_stmt = if_stmt;
        for (size_t dim = 0; dim < ndim; dim++) {
          for_stmt = For::make(
              VarExpr(reuse_loop_vars[dim]),
              0, reuse_shape[dim],
              ForType::Serial,
              DeviceAPI::None,
              for_stmt);
        }
        // 5.  nullify the indices
        // Stmt alloc_body = substitute(null_axis_subst_, alloc->body); // TODO: incorrect!!
        // 6. replace the produce body
        ProduceBodyReplacer mutator(
            for_stmt, 
            target, alloc->buffer_var, 
            target_shape, reuse_shape,
            null_axis_subst_);
        Stmt alloc_body = mutator.Mutate(alloc->body);
        // 7. build the for loop first
        for_stmt = For::make(op->loop_var, op->min, op->extent, op->for_type,
                             op->device_api, alloc_body, op->annotate_keys,
                             op->annotate_values);
        // 8. build the alloc node
        Stmt new_alloc = Allocate::make(
            alloc->buffer_var,
            alloc->type,
            reuse_shape,
            alloc->condition,
            for_stmt);
        // 8. add back the attribute
        Stmt new_attr = AttrStmt::make(
            attr_alloc->node,
            attr_alloc->attr_key,
            attr_alloc->value,
            new_alloc);

        return new_attr;
      } else {
        return IRMutator::Mutate_(op, s);
      }
    }

    Stmt Mutate_(const Allocate* op, const Stmt& s) {
      shape_map_[op->buffer_var.get()] = op->extents;
      return IRMutator::Mutate_(op, s);
    }

  private:
    std::map<const Variable*, Array<Expr> >& shape_map_;
    std::map<const Variable*, Expr> null_axis_subst_;

};

Stmt GenerateReuseBuffer(Stmt stmt, Array<NodeRef> arg_list) {
  LOG(INFO) << stmt;
  std::map<const Variable*, Array<Expr> > shape_map;
  for (size_t i = 0; i < arg_list.size(); i++) {
    if (const BufferNode* node = arg_list[i].as<BufferNode>()) {
      shape_map[node->data.get()] = node->shape;
    }
  }
  ReuseBufferInserter mutator(shape_map);
  stmt = mutator.Mutate(stmt);
  LOG(INFO) << stmt;
  return stmt;
}

} // namespace ir
} // namespace tvm
