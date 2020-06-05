#ifndef SODA_STENCIL_H
#define SODA_STENCIL_H

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

namespace TVM {
namespace ir {

typedef std::unordered_set<Expr, ExprHash, ExprEqual> ExprUnorderedSet;
typedef std::unordered_set<VarExpr, ExprHash, ExprEqual> VarExprUnorderedSet;
typedef std::unordered_map<VarExpr, VarExpr, ExprHash, ExprEqual>
  VarExprVarExprUnorderedMap;

namespace soda {

std::vector<const Stencil*> FindStencil(Stmt body);
void FindStencil(Stmt body, VarExprUnorderedSet& buffers,
                 VarExprVarExprUnorderedMap& args,
                 std::unordered_map<Stmt, std::vector<Stmt> >& stencil_fors,
                 uint32_t& unroll_factor);
inline void FindStencil(
    Stmt body, VarExprUnorderedSet* buffers, VarExprVarExprUnorderedMap* args,
    std::unordered_map<Stmt, std::vector<Stmt> >* stencil_fors,
    uint32_t* unroll_factor) {
  VarExprUnorderedSet buffers_placeholder;
  VarExprVarExprUnorderedMap args_placeholder;
  std::unordered_map<Stmt, std::vector<Stmt> > stencil_fors_placeholder;
  uint32_t unroll_factor_placeholder;
  FindStencil(body, buffers ? *buffers : buffers_placeholder,
              args ? *args : args_placeholder,
              stencil_fors ? *stencil_fors : stencil_fors_placeholder,
              unroll_factor ? *unroll_factor : unroll_factor_placeholder);
}

std::vector<const Load*> FindLoads(Stmt body);
std::vector<const Load*> FindLoads(Expr body);
void FindLoads(Stmt body, std::vector<const Load*>& loads);

std::vector<const Store*> FindStores(Stmt body);
std::vector<const Store*> FindStores(
    Stmt body,
    std::unordered_map<const Store*, std::vector<const LetStmt*> >& store_let_stmts);

} // namespace soda
} // namespace TVM
} // namespace ir

#endif//SODA_STENCIL_H
