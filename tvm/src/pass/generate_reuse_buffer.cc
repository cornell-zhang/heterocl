/*!
 *  Copyright (c) 2019 by Contributors
 * \file loop_partition.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <arithmetic/Substitute.h>

namespace tvm {
namespace ir {

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
    // recover a 1D index back to multi-dimensional index
    std::vector<Expr> recover_index(Expr index, Array<Expr>& shape) {
      std::vector<Expr> new_index;
      for (size_t i = 0; i < shape.size() - 1; i++) {
        Expr simple_index = Simplify(index % shape[i]);
        LOG(INFO) << index % shape[i];
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

class ReuseBufferInserter final : public IRMutator {
  public:
    ReuseBufferInserter(std::map<const Variable*, Array<Expr> >& shape_map) 
      : shape_map_(shape_map) {};

    Stmt Mutate_(const For* op, const Stmt& s) {
      if (const Reuse* node = op->body.as<Reuse>()) {
        LOG(INFO) << node->buffer_var.get();
        VarExpr target = node->buffer_var;
        // collect load expression related to the target
        std::vector<std::vector<Expr> > expr_list;
        std::vector<std::vector<Expr> > diff_list;
        std::map<const Variable*, Expr> min_map;
        std::map<const Variable*, Expr> max_map;
        LoadExpressionCollector visitor(target, 
                                        expr_list, 
                                        min_map, 
                                        max_map,
                                        shape_map_);
        visitor.Visit(op->body);
        int reuse = -1;
        // find the min_expr and max_expr for each dimension
        std::vector<Expr> min_list;
        std::vector<Expr> max_list;
        Array<Expr> reuse_shape;
        for (size_t dim = 0; dim < expr_list[0].size(); dim++) {
          // find the bound
          // e.g. x+r with {r=[0, 2], c=[0, 2], x=[0, 7]}
          // min_expr = 0, max_expr = 9
          // e.g. y+c with {r=[0, 2], c=[0, 2], x=[0, 7]}
          // min_expr = y, max_expr = y+2
          // e.g. [x, x+1, x+2] with {}
          // min_expr = x, max_expr = x+2
          Expr min_expr = substitute(min_map, expr_list[0][dim]);
          Expr max_expr = substitute(max_map, expr_list[0][dim]);
          for (size_t i = 1; i < expr_list.size(); i++) {
            Expr new_min_expr = substitute(min_map, expr_list[i][dim]);
            Expr new_max_expr = substitute(max_map, expr_list[i][dim]);
            Expr min_diff = Simplify(min_expr - new_min_expr);
            Expr max_diff = Simplify(new_max_expr - max_expr);
            if (!is_const(min_diff) || !is_const(max_diff))
              LOG(FATAL) << "The bound of the reuse region cannot be determined";
            if (is_one(Simplify(min_diff > 0))) min_expr = new_min_expr;
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
        }

        // build the updating function for the reuse buffer
        // the main update function is LB[]
        // modify the corresponding allocate node
        return s;
      } else {
        Stmt body = IRMutator::Mutate(op->body);
        return For::make(op->loop_var, op->min, op->extent, op->for_type,
                         op->device_api, body, op->annotate_keys, 
                         op->annotate_values);
      }
    }

    Stmt Mutate_(const Allocate* op, const Stmt& s) {
      shape_map_[op->buffer_var.get()] = op->extents;
      return IRMutator::Mutate_(op, s);
    }

  private:
    std::map<const Variable*, Array<Expr> >& shape_map_;
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
  return stmt;
}

} // namespace ir
} // namespace tvm
