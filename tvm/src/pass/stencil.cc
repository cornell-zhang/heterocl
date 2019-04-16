#include "stencil.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <unordered_set>

#include <arithmetic/Polynomial.h>
#include <arithmetic/Simplify.h>
#include <arithmetic/Substitute.h>
#include <tvm/ir_pass.h>

using std::numeric_limits;
using std::shared_ptr;
using std::string;
using std::unordered_set;
using std::vector;

namespace tvm {
namespace ir {

using HalideIR::Internal::VarExprInt64UnorderedMap;

shared_ptr<StencilFinder> StencilFinder::GetStencil(const Stmt& s) {
  shared_ptr<StencilFinder> stencil(new StencilFinder);
  // 1st-pass mutates the Stmt to unroll innner-loop.
  Stmt new_stmt = stencil->Mutate(s);
  stencil->pass_ = 1;
  // 2nd-pass retrieves the stencil loops. Has to separate the two passes or the
  // retrieved loops won't be able to see the mutations.
  LOG(INFO) << "Mutated stencil stmt: \n" << new_stmt;
  stencil->Mutate(new_stmt);
  if (stencil->HasStencil()) {
    return stencil;
  }
  return nullptr;
}

Stmt StencilFinder::Mutate_(const For* op, const Stmt& s) {
  vector<Stmt> nested_loop;
  Stmt next_s = s;
  //VarExprUnorderedSet loop_vars;
  unordered_set<const Variable*> loop_vars;
  uint32_t unroll_factor = 1;
  while (const For* for_ = next_s.as<For>()) {
    nested_loop.push_back(next_s);
    next_s = for_->body;
    loop_vars.insert(for_->loop_var.get());
    LOG(INFO) << "Find nested loop of " << for_->loop_var;
    int i = 0;
    for (const auto& key : for_->annotate_keys) {
      if (const StringImm* str = key.as<StringImm>()) {
        const IntImm* factor = for_->annotate_values[i].as<IntImm>();
        // Any unroll factor <= 1 is ignored.
        if (str->value == "factor" && factor != nullptr && factor->value > 1) {
          unroll_factor *= factor->value;
        }
      }
      ++i;
    }
  }
  if (unroll_factor_ == 0) {
    unroll_factor_ = unroll_factor;
    LOG(INFO) << "Set stencil unroll factor: " << unroll_factor_;
  } else if (unroll_factor != unroll_factor_) {
    LOG(ERROR) << "Find inconsistent stencil unroll factors. Previous loop: "
               << unroll_factor_ << "; current loop: " << unroll_factor;
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
    if (not is_const(simplify(min_expr))) return s;
    if (not is_const(simplify(extent_expr))) return s;
    LOG(INFO) << "Find static iteration domain of " << loop_op->loop_var;
  }

  if (pass_ == 0) {
    LOG(INFO) << "First pass.";
    // Unroll inner-loops and replace scalar allocates with lets.
    AllocateLetReplacer replacer;
    next_s = UnrollLoop(
        next_s, numeric_limits<int>::max(), numeric_limits<int>::max(),
        numeric_limits<int>::max(), true);
    next_s = replacer.Mutate(next_s);
    next_s = simplify(next_s);
    LOG(INFO) << "Processsed stmt:\n" << next_s;
    for (auto iter = nested_loop.rbegin(); iter != nested_loop.rend(); ++iter) {
      const For* op = iter->as<For>();
      next_s = For::make(op->loop_var, op->min, op->extent, op->for_type,
                         op->device_api, next_s, op->annotate_keys,
                         op->annotate_values);
    }
    return next_s;
  } else {
    LOG(INFO) << "Second pass.";
  }

  // Accessed vars are either local vars, loop vars, or extra params
  // Extra params must not be written. (TODO)
  std::unordered_set<const Variable*> extra_params, local_vars;
  std::unordered_set<const Variable*> accessed_vars;
  //VarExprUnorderedSet extra_params;
  //VarExprUnorderedSet local_vars = LocalVars::GetLocalVars(next_s);
  LocalVarsCollector local_vars_collector(local_vars);
  local_vars_collector.Visit(next_s);
  AccessedVarsCollector accessed_vars_collector(accessed_vars);
  accessed_vars_collector.Visit(next_s);
  for (auto accessed_var : accessed_vars) {
    if (local_vars.count(accessed_var) == 0 and
        loop_vars.count(accessed_var) == 0) {
      extra_params.insert(accessed_var);
    }
  }

  // Find all Loads and Stores and examine the indices
  std::vector<const Load*> loads;
  LoadsCollector loads_collector(loads);
  loads_collector.Visit(next_s);
  //ExprUnorderedSet loads = Loads::GetLoads(next_s);
  std::vector<const Store*> stores;
  std::unordered_map<const Store*, std::vector<const LetStmt*> > store_let_stmts;
  StoresCollector stores_collector(stores, store_let_stmts);
  stores_collector.Visit(next_s);

  VarExprUnorderedSet load_vars;
  for (auto load : loads) load_vars.insert(load->buffer_var);
  for (auto store : stores) {
    // Doesn't allow the same variable to be read and written in the same loop.
    if (load_vars.count(store->buffer_var)) return s;
  }
  LOG(INFO) << "No tensor is read and written in the same loop, good.";

  for (auto load : loads) {
    accessed_vars.clear();
    accessed_vars_collector.Visit(load->index);
    for (auto var : accessed_vars) {
      // Load indices must be loop vars
      if (loop_vars.count(var) == 0) {
        LOG(INFO) << "Load index acesses variable " << var << ", which is not "
                  << "a loop variable. Give up.";
        return s;
      }
    }

    // Index must be affine
    VarExprInt64UnorderedMap affine_coeffs = GetAffineCoeff(load->index);
    if (affine_coeffs.empty()) {
      LOG(INFO) << "Load " << load << " is not affine.";
      return s;
    }
  }
  LOG(INFO) << "Load indices are affine, good.";

  for (auto store : stores) {
    accessed_vars.clear();
    accessed_vars_collector.Visit(store->index);
    for (auto var : accessed_vars) {
      // Store indices must be loop vars
      if (loop_vars.count(var) == 0) return s;
    }

    // Index must be affine
    VarExprInt64UnorderedMap affine_coeffs =
      GetAffineCoeff(store->index);
    if (affine_coeffs.empty()) return s;
  }
  LOG(INFO) << "Store indices are affine, good.";

  stencil_fors_[s] = nested_loop;

  return s;
}

Stmt StencilFinder::Mutate_(const LetStmt* op, const Stmt& s) {
  if (pass_ != 0) {
    if (const Call* call = op->value.as<Call>()) {
      if (call->name == "tvm_struct_get") {
        if (call->args[2].as<IntImm>()->value == 1) {
          LOG(INFO)<<"Buffer "<<op->var<<" allocated for arg "<<call->args[0];
          buffers_.insert(op->var);
          args_[VarExpr(call->args[0].node_)] = op->var;
        }
      }
    }
  }
  return IRMutator::Mutate_(op, s);
}

Stmt AllocateLetReplacer::Mutate_(const Allocate* op, const Stmt& s) {
  // Only mutates singleton tensor allocations (which is in fact a scalar and is
  // used in reductions).
  if (op->extents.size() == 1) {
    if (is_one(op->extents[0])) {
      if (auto block = op->body.as<Block>()) {
        if (auto producer = block->first.as<ProducerConsumer>()) {
          if (auto attr = producer->body.as<AttrStmt>()) {
            if (auto store = attr->body.as<Store>()) {
              Expr store_val = this->Mutate(store->value);
              VarExpr var(op->buffer_var->name_hint + "_ssa0", op->type);
              vars_[op->buffer_var.get()] = var;
              Stmt stmt = LetStmt::make(var, store_val, this->Mutate(block->rest));
              vars_.erase(op->buffer_var.get());
              return stmt;
            }
          }
        }
      }
    }
  }
  return IRMutator::Mutate_(op, s);
}

/*
Stmt Allocates::Mutate_(const Block* op, const Stmt& s) {
  const Store* store = op->first.as<Store>();
  if (store != nullptr) {
    if (vars_.count(store->buffer_var)) {
      if (op->rest.defined()) {
        const Expr&& store_val = mutate(store->value);
        const Variable* old_var = vars_[store->buffer_var].get();
        string name_hint = old_var->name_hint;
        const size_t pos = name_hint.rfind("_ssa");
        const uint64_t counter = std::stoull(name_hint.substr(pos + 4));
        name_hint = name_hint.substr(0, pos + 4) + std::to_string(counter + 1);
        const VarExpr&& var_expr = Variable::make(
            old_var->type, name_hint);
        vars_[store->buffer_var] = var_expr;
        stmt = LetStmt::make(var_expr, store_val, this->Mutate(op->rest));
        //IRMutator::visit(stmt.as<LetStmt>(), stmt);
      } else {
        LOG(INFO) << "Undefined rest:\n" << stmt;
      }
      return;
    }
  }
  return stmt;
}*/

Expr AllocateLetReplacer::Mutate_(const Load* op, const Expr& e) {
  Expr index = this->Mutate(op->index);
  if (vars_.count(op->buffer_var.get())) {
    return Load::make(op->type, vars_[op->buffer_var.get()], index, op->predicate);
  } else {
    return Load::make(op->type, op->buffer_var, index, op->predicate);
  }
}

} // namespace tvm
} // namespace ir
