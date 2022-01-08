/*!
 *  Copyright (c) 2016 by Contributors
 * \file simple_passes.cc
 * \brief Implementation of simple passes
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>

namespace TVM {
namespace ir {

class IRSideEffect : public IRVisitor {
 public:
  void Visit(const NodeRef& e) final {
    if (has_side_effect_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Call* op) final {
    if (!op->is_pure()) {
      has_side_effect_ = true;
      return;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  bool has_side_effect_{false};
};

bool HasSideEffect(const Expr& e) {
  IRSideEffect v;
  v.Visit(e);
  return v.has_side_effect_;
}

class IRSubstitue : public IRMutator {
 public:
  explicit IRSubstitue(const std::unordered_map<const Variable*, Expr>& smap)
      : smap_(smap) {}

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = smap_.find(op);
    if (it != smap_.end()) {
      return it->second;
    } else {
      return e;
    }
  }

 private:
  const std::unordered_map<const Variable*, Expr>& smap_;
};

Stmt Substitute(Stmt stmt,
                const std::unordered_map<const Variable*, Expr>& value_map) {
  if (value_map.size() == 0) return stmt;
  return IRSubstitue(value_map).Mutate(stmt);
}

Expr Substitute(Expr expr,
                const std::unordered_map<const Variable*, Expr>& value_map) {
  if (value_map.size() == 0) return expr;
  return IRSubstitue(value_map).Mutate(expr);
}

Stmt Substitute(Stmt stmt, const Map<Var, Expr>& value_map) {
  std::unordered_map<const Variable*, Expr> vmap;
  for (const auto& kv : value_map) {
    vmap[kv.first.get()] = kv.second;
  }
  return Substitute(stmt, vmap);
}

Expr Substitute(Expr expr, const Map<Var, Expr>& value_map) {
  std::unordered_map<const Variable*, Expr> vmap;
  for (const auto& kv : value_map) {
    vmap[kv.first.get()] = kv.second;
  }
  return Substitute(expr, vmap);
}

class VarTouchVisitor : public IRVisitor {
 public:
  void Visit(const NodeRef& e) final {
    if (use_var_) return;
    IRVisitor::Visit(e);
  }

  void Visit_(const Variable* op) final { Handle(op); }

  void Visit_(const Load* op) final {
    Handle(op->buffer_var.get());
    IRVisitor::Visit_(op);
  }

  virtual void Handle(const Variable* var) = 0;

  bool use_var_{false};
};

class ExprUseVarVisitor : public VarTouchVisitor {
 public:
  explicit ExprUseVarVisitor(const Variable* var) : var_(var) {}

  void Handle(const Variable* var) final {
    if (var == var_) use_var_ = true;
  }

 private:
  const Variable* var_;
};

class ExprUseVSetVisitor : public VarTouchVisitor {
 public:
  explicit ExprUseVSetVisitor(const std::unordered_set<const Variable*>& vset)
      : vset_(vset) {}

  void Handle(const Variable* var) final {
    if (vset_.count(var)) use_var_ = true;
  }

 private:
  const std::unordered_set<const Variable*>& vset_;
};

bool ExprUseVar(const Expr& e, const Var& v) {
  ExprUseVarVisitor visitor(v.get());
  visitor.Visit(e);
  return visitor.use_var_;
}

bool ExprUseVar(const Expr& e,
                const std::unordered_set<const Variable*>& vset) {
  ExprUseVSetVisitor visitor(vset);
  visitor.Visit(e);
  return visitor.use_var_;
}

class IterRangeCollector final : public IRVisitor {
 public:
  IterRangeCollector(std::unordered_map<const Variable*, Expr>& range)
      : range_(range) {}

  void Visit_(const For* op) override {
    range_[op->loop_var.get()] = Simplify(op->extent - 1);
    this->Visit(op->body);
  }

 private:
  std::unordered_map<const Variable*, Expr>& range_;
};

class ExprNotUseVarVisitor : public VarTouchVisitor {
 public:
  bool used_unknown_var{false};
  explicit ExprNotUseVarVisitor(
      std::unordered_map<const Variable*, Expr>& range)
      : range_(range) {}

  void Handle(const Variable* var) final {
    if (!range_.count(var)) used_unknown_var = true;
  }

 private:
  std::unordered_map<const Variable*, Expr>& range_;
};

bool has_unknown_vals(Expr expr,
                      std::unordered_map<const Variable*, Expr>& range) {
  ExprNotUseVarVisitor visitor(range);
  visitor.Visit(expr);
  return visitor.used_unknown_var;
}

std::unordered_map<const Variable*, Expr> CollectIterRange(Stmt stmt) {
  std::unordered_map<const Variable*, Expr> range;
  IterRangeCollector visitor(range);
  visitor.Visit(stmt);
  return range;
}

std::vector<Expr> ExtractIndices(
    Expr index, const Array<Expr>& shape,
    std::unordered_map<const Variable*, Expr>& range) {
  std::vector<Expr> new_index;
  for (size_t i = shape.size() - 1; i >= 1; i--) {
    Expr simple_index = Simplify(index % shape[i]);
    // remove modulo
    if (const Mod* op = simple_index.as<Mod>()) {
      Expr max = Simplify(Substitute(op->a, range) + 1);
      Expr comp = Simplify(max <= op->b);
      if (is_one(comp) || has_unknown_vals(max, range)) {
        simple_index = op->a;
      }
    }
    new_index.push_back(simple_index);
    // simplify the rest
    index = Simplify((index - simple_index) / shape[i]);
  }
  new_index.push_back(index);
  std::reverse(new_index.begin(), new_index.end());
  return new_index;
}

Expr FlattenIndices(std::vector<Expr> indices, const Array<Expr> shape) {
  size_t ndim = indices.size();
  Expr ret = indices[ndim - 1];
  Expr mul = 1;
  for (size_t i = ndim - 1; i >= 1; --i) {
    mul = Simplify(mul * shape[i]);
    ret = Simplify(ret + indices[i - 1] * mul);
  }
  return ret;
}

}  // namespace ir
}  // namespace TVM
