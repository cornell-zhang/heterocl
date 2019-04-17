#ifndef HALIDEIR_STENCIL_H
#define HALIDEIR_STENCIL_H

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

/** \file
 * Defines Stencil - Represent information of a stencil filter
 */

namespace tvm {
namespace ir {

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
class StencilFinder final : public IRMutator {
  // Maps a outer For to a vector of nested Fors.
  std::unordered_map<Stmt, std::vector<Stmt> > stencil_fors_;
  VarExprUnorderedSet buffers_;
  VarExprVarExprUnorderedMap args_;
  int pass_ = 0;
  uint32_t unroll_factor_ = 0;

  Stmt Mutate_(const For*, const Stmt&);
  Stmt Mutate_(const LetStmt*, const Stmt&);

public:
  // Make a Stencil object if p is stencil, nullptr otherwise.
  static std::shared_ptr<StencilFinder> GetStencil(const Stmt&);

  bool HasStencil() const {return not stencil_fors_.empty();}
  uint32_t UnrollFactor() {return std::max(unroll_factor_, 1U);}

  std::unordered_map<Stmt, std::vector<Stmt> > GetStencilFors() const {
    return stencil_fors_;
  }
  VarExprUnorderedSet GetBuffers() const {return buffers_;}
  VarExprVarExprUnorderedMap GetArgs() const {return args_;}
}; // class StencilFinder

class AccessedVarsCollector final : public IRVisitor {
  public:
    AccessedVarsCollector(std::unordered_set<const Variable*>& accessed_vars)
      : accessed_vars_(accessed_vars) {};

    void Visit_(const Variable* op) {
      accessed_vars_.insert(op);
    } 

  private:
    std::unordered_set<const Variable*>& accessed_vars_;
};

class LocalVarsCollector final : public IRVisitor {
  public:
    LocalVarsCollector(std::unordered_set<const Variable*>& local_vars)
      : local_vars_(local_vars) {};

    void Visit_(const Let* op) {
      local_vars_.insert(op->var.get());
      IRVisitor::Visit_(op);
    }

    void Visit_(const LetStmt* op) {
      local_vars_.insert(op->var.get());
      IRVisitor::Visit_(op);
    }

  private:
    std::unordered_set<const Variable*>& local_vars_;
};

class LoadsCollector final : public IRVisitor {
  public:
    LoadsCollector(std::vector<const Load*>& loads): loads_(loads) {}

    void Visit_(const Load* op) {
      loads_.push_back(op);
      IRVisitor::Visit_(op);
    }

  private:
    std::vector<const Load*>& loads_;
};

class StoresCollector final : public IRVisitor {
  public:
    StoresCollector(
        std::vector<const Store*>& stores,
        std::unordered_map<const Store*, std::vector<const LetStmt*> >& store_let_stmts)
      : stores_(stores), store_let_stmts_(store_let_stmts) {}

    void Visit_(const Store* op) {
      stores_.push_back(op);
      store_let_stmts_[op] = let_stmts_;
      IRVisitor::Visit_(op);
    }

    void Visit_(const LetStmt* op) {
      let_stmts_.push_back(op);
      IRVisitor::Visit_(op);
      let_stmts_.pop_back();
    }

  private:
    std::vector<const Store*>& stores_;
    std::unordered_map<const Store*, std::vector<const LetStmt*> >& store_let_stmts_;
    std::vector<const LetStmt*> let_stmts_;
};

class AllocateLetReplacer final : public IRMutator {
  public:
    Stmt Mutate_(const Allocate* op, const Stmt& s);
    Stmt Mutate_(const Block* op, const Stmt& e);
    Expr Mutate_(const Load* op, const Expr& e);

  private:
    std::unordered_map<const Variable*, VarExpr> vars_;
};

} // namespace tvm
} // namespace ir

#endif//HALIDEIR_STENCIL_H
