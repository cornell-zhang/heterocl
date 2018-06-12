#include <memory>
#include <unordered_set>

#include "arithmetic/Polynomial.h"
#include "arithmetic/Simplify.h"
#include "arithmetic/Substitute.h"
#include "base/Stencil.h"
#include "ir/IRMutator.h"
#include "ir/IROperator.h"

using std::shared_ptr;
using std::unordered_set;
using std::vector;

namespace HalideIR {
namespace Internal {

shared_ptr<Stencil> Stencil::GetStencil(const Stmt& s) {
  shared_ptr<Stencil> stencil(new Stencil);
  s.accept(stencil.get());
  if (stencil->HasStencil()) {
    return stencil;
  }
  return nullptr;
}

void Stencil::visit(const For* op, const Stmt& s) {
  vector<Stmt> nested_loop;
  Stmt next_s = s;
  VarExprUnorderedSet loop_vars;
  while (const For* next_op = next_s.as<For>()) {
    nested_loop.push_back(next_s);
    next_s = next_op->body;
    loop_vars.insert(next_op->loop_var);
  }

  // Check for static iteration domain.
  Expr const_expr(0);
  for (const Stmt& loop_s : nested_loop) {
    const For* loop_op = loop_s.as<For>();
    Expr min_expr = loop_op->min;
    Expr extent_expr = loop_op->extent;
    // Replace all outer loop vars with a constant to check for static bounds.
    for (auto outer_loop_s = nested_loop.begin();
         not outer_loop_s->same_as(loop_s); ++ outer_loop_s) {
      const For* outer_loop_op = outer_loop_s->as<For>();
      const VarExpr& outer_loop_var = outer_loop_op->loop_var;
      min_expr = substitute(outer_loop_var, const_expr, min_expr);
      extent_expr = substitute(outer_loop_var, const_expr, extent_expr);
    }
    if (not is_const(simplify(min_expr))) return;
    if (not is_const(simplify(extent_expr))) return;
  }

  // Accessed vars are either local vars, loop vars, or extra params
  // Extra params must not be written. (TODO)
  VarExprUnorderedSet extra_params;
  VarExprUnorderedSet local_vars = LocalVars::GetLocalVars(next_s);
  for (auto accessed_var : AccessedVars::GetAccessedVars(next_s)) {
    if (local_vars.count(accessed_var) == 0 and
        loop_vars.count(accessed_var) == 0) {
      extra_params.insert(accessed_var);
    }
  }

  // Find all Loads and Stores and examine the indices
  ExprUnorderedSet loads = Loads::GetLoads(next_s);
  unordered_set<Stmt> stores = Stores::GetStores(next_s);

  VarExprUnorderedSet load_vars;
  for (auto load : loads) load_vars.insert(load.as<Load>()->buffer_var);
  for (auto store : stores) {
    // Doesn't allow the same variable to be read and written in the same loop.
    if (load_vars.count(store.as<Store>()->buffer_var)) return;
  }

  for (auto load : loads) {
    for (auto var : AccessedVars::GetAccessedVars(load.as<Load>()->index)) {
      // Load indices must be loop vars
      if (loop_vars.count(var) == 0) return;
    }

    // Index must be affine
    VarExprInt64UnorderedMap affine_coeffs =
      GetAffineCoeff(load.as<Load>()->index);
    if (affine_coeffs.empty()) return;
  }
  for (auto store : stores) {
    for (auto var : AccessedVars::GetAccessedVars(store.as<Store>()->index)) {
      // Store indices must be loop vars
      if (loop_vars.count(var) == 0) return;
    }

    // Index must be affine
    VarExprInt64UnorderedMap affine_coeffs =
      GetAffineCoeff(store.as<Store>()->index);
    if (affine_coeffs.empty()) return;
  }

  stencil_fors_[s] = nested_loop;
}

void Stencil::visit(const LetStmt* op, const Stmt& s) {
  if (const Call* call = op->value.as<Call>()) {
    if (call->name == "tvm_struct_get") {
      if (call->args[2].as<IntImm>()->value == 1) {
        LOG(INFO)<<"Buffer "<<op->var<<" allocated for arg "<<call->args[0];
        buffers_.insert(op->var);
        args_[VarExpr(call->args[0].node_)] = op->var;
      }
    }
  }
  IRVisitor::visit(op, s);
}

} // namespace Internal
} // namespace HalideIR
