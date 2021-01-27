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

// collect all load indices that contains the reuse target
class LoadCollector final : public IRVisitor {
 public:
  LoadCollector(const Variable* target,
                std::vector<std::vector<Expr> >& expr_list,
                std::map<const Variable*, Expr>& min_map,
                std::map<const Variable*, Expr>& max_map,
                const Array<Expr>& target_shape,
                std::unordered_map<const Variable*, Expr>& range)
      : target_(target),
        expr_list_(expr_list),
        min_map_(min_map),
        max_map_(max_map),
        target_shape_(target_shape),
        range_(range) {}

  void Visit_(const Load* op) {
    this->Visit(op->index);
    if (op->buffer_var.get() == target_) {
      std::vector<Expr> new_index =
          ExtractIndices(op->index, target_shape_, range_);
      for (size_t i = 0; i < new_index.size(); i++)
        expr_list_.push_back(new_index);
    }
  }

  void Visit_(const For* op) {
    min_map_[op->loop_var.get()] = op->min;
    max_map_[op->loop_var.get()] = op->extent - 1;
    this->Visit(op->body);
  }

 private:
  const Variable* target_;
  // a list of indices; each index is represented with a tuple
  // e.g., [[x, y], [x+1, y], [x+2, y]]
  // e.g., [[x+r, y+c]]
  std::vector<std::vector<Expr> >& expr_list_;
  // key, value = loop_var, min
  std::map<const Variable*, Expr>& min_map_;
  // key, value = loop_var, extent-1
  std::map<const Variable*, Expr>& max_map_;
  const Array<Expr>& target_shape_;
  std::unordered_map<const Variable*, Expr>& range_;
};

Array<Expr> InferReuseBound(const Stmt& body, const Variable* target,
                            const Array<Expr>& target_shape,
                            std::unordered_map<const Variable*, Expr>& range) {
  // collect load expression related to the target
  std::vector<std::vector<Expr> > expr_list;
  std::vector<std::vector<Expr> > diff_list;
  std::vector<Expr> min_list;
  std::map<const Variable*, Expr> min_map;
  std::map<const Variable*, Expr> max_map;
  LoadCollector visitor(target, expr_list, min_map, max_map, target_shape,
                        range);
  visitor.Visit(body);
  // int reuse = -1;
  // find the min_expr and max_expr for each dimension
  Array<Expr> reuse_shape;
  // if nothing can be reused
  if (expr_list.size() == 0) return reuse_shape;
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
    if (!is_const(diff_expr))  // e.g. y*(y+c) would be illegal
      LOG(FATAL) << "Irregular access pattern is not yet supported";
    /* TODO(Sean): add this back when merge with reuse buffer
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
      if (!is_zero(compare))
        reuse = dim;
    }
    */
    if (auto imm = diff_expr.as<IntImm>())
      diff_expr = IntImm::make(Int(32), imm->value);
    reuse_shape.push_back(diff_expr);
    min_list.push_back(expr_list[min_index][dim]);
  }  // end for each dim
  return reuse_shape;
}

}  // end namespace ir
}  // end namespace TVM
