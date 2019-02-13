#ifndef HALIDEIR_STENCIL_H
#define HALIDEIR_STENCIL_H

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "ir/IR.h"
#include "ir/IRMutator.h"
#include "ir/IRVisitor.h"

/** \file
 * Defines Stencil - Represent information of a stencil filter
 */

namespace HalideIR {
namespace Internal {

template<typename T>
inline bool SetEqual(const T& lhs, const T& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto elem : lhs) if (rhs.count(elem) == 0) return false;
  return true;
}

typedef std::unordered_set<Expr, ExprHash, ExprEqual> ExprUnorderedSet;
inline bool operator==(const ExprUnorderedSet& lhs,
                       const ExprUnorderedSet& rhs) {
  return SetEqual(lhs, rhs);
}
inline bool operator!=(const ExprUnorderedSet& lhs,
                       const ExprUnorderedSet& rhs) {
  return not SetEqual(lhs, rhs);
}

typedef std::unordered_set<VarExpr, ExprHash, ExprEqual> VarExprUnorderedSet;
inline bool operator==(const VarExprUnorderedSet& lhs,
                       const VarExprUnorderedSet& rhs) {
  return SetEqual(lhs, rhs);
}
inline bool operator!=(const VarExprUnorderedSet& lhs,
                       const VarExprUnorderedSet& rhs) {
  return not SetEqual(lhs, rhs);
}

typedef std::unordered_map<VarExpr, VarExpr, ExprHash, ExprEqual>
  VarExprVarExprUnorderedMap;

/** A Halide stencil.
 */
class Stencil : public IRMutator {
  // Maps a outer For to a vector of nested Fors.
  std::unordered_map<Stmt, std::vector<Stmt> > stencil_fors_;
  VarExprUnorderedSet buffers_;
  VarExprVarExprUnorderedMap args_;
  int pass_ = 0;
  uint32_t unroll_factor_ = 0;

  void visit(const For*, const Stmt&);
  void visit(const LetStmt*, const Stmt&);
  using IRMutator::visit;

public:
  // Make a Stencil object if p is stencil, nullptr otherwise.
  static std::shared_ptr<Stencil> GetStencil(const Stmt&);

  bool HasStencil() const {return not stencil_fors_.empty();}
  uint32_t UnrollFactor() {return std::max(unroll_factor_, 1U);}

  std::unordered_map<Stmt, std::vector<Stmt> > GetStencilFors() const {
    return stencil_fors_;
  }
  VarExprUnorderedSet GetBuffers() const {return buffers_;}
  VarExprVarExprUnorderedMap GetArgs() const {return args_;}
}; // class Stencil

class AccessedVars : public IRVisitor {
  VarExprUnorderedSet accessed_vars_;

  void visit(const Variable* op, const Expr& e) {
    accessed_vars_.insert(VarExpr(e.node_));
    IRVisitor::visit(op, e);
  }

  using IRVisitor::visit;

public:
  static VarExprUnorderedSet GetAccessedVars(const Stmt& s) {
    AccessedVars l;
    s.accept(&l);
    return l.accessed_vars_;
  }
  static VarExprUnorderedSet GetAccessedVars(const Expr& e) {
    AccessedVars l;
    e.accept(&l);
    return l.accessed_vars_;
  }
};

class LocalVars : public IRVisitor {
  VarExprUnorderedSet local_vars_;

  void visit(const Let* op, const Expr& e) {
    local_vars_.insert(op->var);
    IRVisitor::visit(op, e);
  }

  void visit(const LetStmt* op, const Stmt& s) {
    local_vars_.insert(op->var);
    IRVisitor::visit(op, s);
  }

  using IRVisitor::visit;

public:
  static VarExprUnorderedSet GetLocalVars(const Stmt& s) {
    LocalVars l;
    s.accept(&l);
    return l.local_vars_;
  }
};

class Loads : public IRVisitor {
  ExprUnorderedSet loads_;

  void visit(const Load* op, const Expr& e) {
    loads_.insert(e);
    IRVisitor::visit(op, e);
  }

  using IRVisitor::visit;

public:
  static ExprUnorderedSet GetLoads(const Stmt& s) {
    Loads l;
    s.accept(&l);
    return l.loads_;
  }
  static ExprUnorderedSet GetLoads(const Expr& e) {
    Loads l;
    e.accept(&l);
    return l.loads_;
  }
  // T can be container of Stmt or Expr
  template<typename T> static ExprUnorderedSet GetLoads(const T& nodes) {
    Loads l;
    for (const auto& node : nodes) {
      node.accept(&l);
    }
    return l.loads_;
  }
};

class Stores : public IRVisitor {
  std::unordered_set<Stmt> stores_;
  std::unordered_map<Stmt, std::vector<Stmt> > store_let_stmts_;
  std::vector<Stmt> let_stmts_;

  void visit(const Store* op, const Stmt& s) {
    stores_.insert(s);
    store_let_stmts_[s] = let_stmts_;
    IRVisitor::visit(op, s);
  }

  void visit(const LetStmt* op, const Stmt& s) override {
    let_stmts_.push_back(s);
    IRVisitor::visit(op, s);
    let_stmts_.pop_back();
  }

  using IRVisitor::visit;

public:
  static std::unordered_set<Stmt> GetStores(
      const Stmt& s,
      std::unordered_map<Stmt, std::vector<Stmt> >* let_stmts = nullptr) {
    Stores l;
    s.accept(&l);
    if (let_stmts != nullptr) {
      *let_stmts = l.store_let_stmts_;
    }
    return l.stores_;
  }
};

class Allocates : public IRMutator {
  std::unordered_set<Stmt> allocates_;
  VarExprVarExprUnorderedMap vars_;

  void visit(const Allocate* op, const Stmt& s) override;
  void visit(const Block* op, const Stmt& e) override;
  void visit(const Load* op, const Expr& e) override;
  using IRMutator::visit;

public:
  static Stmt Replace(const Stmt& s) {
    return Allocates().mutate(s);
  }
  static std::unordered_set<Stmt> GetAllocates(const Stmt& s) {
    Allocates l;
    s.accept(&l);
    return l.allocates_;
  }
};

} // namespace Internal
} // namespace HalideIR

#endif//HALIDEIR_STENCIL_H
