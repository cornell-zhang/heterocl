#ifndef HALIDEIR_STENCIL_H
#define HALIDEIR_STENCIL_H

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "ir/IR.h"
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
class Stencil : public IRVisitor {
  // Maps a outer For to a vector of nested Fors.
  std::unordered_map<Stmt, std::vector<Stmt> > stencil_fors_;
  VarExprUnorderedSet buffers_;
  VarExprVarExprUnorderedMap args_;

  void visit(const For*, const Stmt&);
  void visit(const LetStmt*, const Stmt&);
  using IRVisitor::visit;

public:
  // Make a Stencil object if p is stencil, nullptr otherwise.
  static std::shared_ptr<Stencil> GetStencil(const Stmt&);

  bool HasStencil() const {return not stencil_fors_.empty();}

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
};

class Stores : public IRVisitor {
  std::unordered_set<Stmt> stores_;

  void visit(const Store* op, const Stmt& s) {
    stores_.insert(s);
    IRVisitor::visit(op, s);
  }

  using IRVisitor::visit;

public:
  static std::unordered_set<Stmt> GetStores(const Stmt& s) {
    Stores l;
    s.accept(&l);
    return l.stores_;
  }
};

} // namespace Internal
} // namespace HalideIR

#endif//HALIDEIR_STENCIL_H
