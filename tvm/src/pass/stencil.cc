/*!
 *  Copyright (c) 2017 by Contributors
 * \file storage_access.cc
 */
#include "stencil.h"
#include <arithmetic/Polynomial.h>
#include <arithmetic/Substitute.h>
#include <tvm/ir_pass.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <unordered_set>

using std::numeric_limits;
using std::shared_ptr;
using std::string;
using std::unordered_set;
using std::vector;

namespace TVM {
namespace ir {
namespace soda {

using Halide::Internal::VarExprInt64UnorderedMap;

namespace {

class AccessedVarsCollector final : public IRVisitor {
 public:
  AccessedVarsCollector(std::unordered_set<const Variable*>& accessed_vars)
      : accessed_vars_(accessed_vars) {}

  void Visit_(const Variable* op) { accessed_vars_.insert(op); }

 private:
  std::unordered_set<const Variable*>& accessed_vars_;
};

class LocalVarsCollector final : public IRVisitor {
 public:
  LocalVarsCollector(std::unordered_set<const Variable*>& local_vars)
      : local_vars_(local_vars) {}

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
  LoadsCollector(std::vector<const Load*>& loads) : loads_(loads) {}

  void Visit_(const Load* op) {
    loads_.push_back(op);
    IRVisitor::Visit_(op);
  }

 private:
  std::vector<const Load*>& loads_;
};

class StoresCollector final : public IRVisitor {
 public:
  StoresCollector(std::vector<const Store*>& stores,
                  std::unordered_map<const Store*, std::vector<const LetStmt*>>&
                      store_let_stmts)
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
  std::unordered_map<const Store*, std::vector<const LetStmt*>>&
      store_let_stmts_;
  std::vector<const LetStmt*> let_stmts_;
};

class AllocateLetReplacer final : public IRMutator {
 public:
  Stmt Mutate_(const Allocate* op, const Stmt& s) {
    // Only mutates singleton tensor allocations (which is in fact a
    // scalar and is used in reductions).
    if (op->extents.size() == 1) {
      if (is_one(op->extents[0])) {
        if (auto block = op->body.as<Block>()) {
          if (auto producer = block->first.as<ProducerConsumer>()) {
            if (auto attr = producer->body.as<AttrStmt>()) {
              if (auto store = attr->body.as<Store>()) {
                Expr store_val = this->Mutate(store->value);
                VarExpr var(op->buffer_var->name_hint + "_ssa0", op->type);
                vars_[op->buffer_var.get()] = var;
                Stmt stmt =
                    LetStmt::make(var, store_val, this->Mutate(block->rest));
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

  Stmt Mutate_(const Block* op, const Stmt& s) {
    if (auto store = op->first.as<Store>()) {
      if (vars_.count(store->buffer_var.get())) {
        if (op->rest.defined()) {
          Expr store_val = this->Mutate(store->value);
          const Variable* old_var = vars_[store->buffer_var.get()].get();
          string name_hint = old_var->name_hint;
          const size_t pos = name_hint.rfind("_ssa");
          const uint64_t counter = std::stoull(name_hint.substr(pos + 4));
          name_hint =
              name_hint.substr(0, pos + 4) + std::to_string(counter + 1);
          VarExpr var_expr(name_hint, old_var->type);
          vars_[store->buffer_var.get()] = var_expr;
          return LetStmt::make(var_expr, store_val, this->Mutate(op->rest));
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load* op, const Expr& e) {
    if (vars_.count(op->buffer_var.get())) {
      return vars_[op->buffer_var.get()];
    } else {
      return e;
    }
  }

 private:
  std::unordered_map<const Variable*, VarExpr> vars_;
};

class StencilFinder final : public IRMutator {
 public:
  StencilFinder(VarExprUnorderedSet& buffers, VarExprVarExprUnorderedMap& args,
                std::unordered_map<Stmt, std::vector<Stmt>>& stencil_fors,
                uint32_t& unroll_factor)
      : buffers_(buffers),
        args_(args),
        stencil_fors_(stencil_fors),
        unroll_factor_(unroll_factor) {
    unroll_factor_ = 0;
  }

  void set_pass(int pass) { pass_ = pass; }

  Stmt Mutate_(const For* op, const Stmt& s) {
    vector<Stmt> nested_loop;
    Stmt next_s = s;
    unordered_set<const Variable*> loop_vars;
    uint32_t unroll_factor = 1;
    while (const For* for_ = next_s.as<For>()) {
      nested_loop.push_back(next_s);
      next_s = for_->body;
      loop_vars.insert(for_->loop_var.get());
      int i = 0;
      for (const auto& key : for_->annotate_keys) {
        if (const StringImm* str = key.as<StringImm>()) {
          const IntImm* factor = for_->annotate_values[i].as<IntImm>();
          // Any unroll factor <= 1 is ignored.
          if (str->value == "factor" && factor != nullptr &&
              factor->value > 1) {
            unroll_factor *= factor->value;
          }
        }
        ++i;
      }
    }
    if (unroll_factor_ == 0) {
      unroll_factor_ = unroll_factor;
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
           !outer_loop_s->same_as(loop_s); ++outer_loop_s) {
        const For* outer_loop_op = outer_loop_s->as<For>();
        const VarExpr& outer_loop_var = outer_loop_op->loop_var;
        min_expr = substitute(outer_loop_var, const_expr, min_expr);
        extent_expr = substitute(outer_loop_var, const_expr, extent_expr);
      }
      if (!is_const(Simplify(min_expr))) return s;
      if (!is_const(Simplify(extent_expr))) return s;
    }

    if (pass_ == 0) {
      // Unroll inner-loops and replace scalar allocates with lets.
      AllocateLetReplacer replacer;
      next_s = UnrollLoop(next_s, numeric_limits<int>::max(),
                          numeric_limits<int>::max(),
                          numeric_limits<int>::max(), true);
      next_s = replacer.Mutate(next_s);
      next_s = Simplify(next_s);
      for (auto iter = nested_loop.rbegin(); iter != nested_loop.rend();
           ++iter) {
        const For* op = iter->as<For>();
        next_s = For::make(op->loop_var, op->min, op->extent, op->for_type,
                           op->device_api, next_s, op->annotate_keys,
                           op->annotate_values);
      }
      return next_s;
    }

    // Accessed vars are either local vars, loop vars, or extra params
    // Extra params must not be written. TODO(blaok)
    std::unordered_set<const Variable*> extra_params, local_vars;
    std::unordered_set<const Variable*> accessed_vars;
    LocalVarsCollector local_vars_collector(local_vars);
    local_vars_collector.Visit(next_s);
    AccessedVarsCollector accessed_vars_collector(accessed_vars);
    accessed_vars_collector.Visit(next_s);
    for (auto accessed_var : accessed_vars) {
      if (local_vars.count(accessed_var) == 0 &&
          loop_vars.count(accessed_var) == 0) {
        extra_params.insert(accessed_var);
      }
    }

    // Find all Loads and Stores and examine the indices
    std::vector<const Load*> loads = FindLoads(next_s);
    std::vector<const Store*> stores = FindStores(next_s);

    VarExprUnorderedSet load_vars;
    for (auto load : loads) load_vars.insert(load->buffer_var);
    for (auto store : stores) {
      // Doesn't allow the same variable to be read & written in the same loop.
      if (load_vars.count(store->buffer_var)) return s;
    }

    for (auto load : loads) {
      accessed_vars.clear();
      accessed_vars_collector.Visit(load->index);
      for (auto var : accessed_vars) {
        // Load indices must be loop vars
        if (loop_vars.count(var) == 0) {
          LOG(INFO) << "Load index acesses variable " << var
                    << ", which is not "
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

    for (auto store : stores) {
      accessed_vars.clear();
      accessed_vars_collector.Visit(store->index);
      for (auto var : accessed_vars) {
        // Store indices must be loop vars
        if (loop_vars.count(var) == 0) return s;
      }

      // Index must be affine
      VarExprInt64UnorderedMap affine_coeffs = GetAffineCoeff(store->index);
      if (affine_coeffs.empty()) return s;
    }

    stencil_fors_[s] = nested_loop;
    return s;
  }

  Stmt Mutate_(const LetStmt* op, const Stmt& s) {
    if (pass_ != 0) {
      if (const Call* call = op->value.as<Call>()) {
        if (call->name == "tvm_struct_get") {
          if (call->args[2].as<IntImm>()->value == 1) {
            buffers_.insert(op->var);
            args_[VarExpr(call->args[0].node_)] = op->var;
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  // Maps a outer For to a vector of nested Fors.
  VarExprUnorderedSet& buffers_;
  VarExprVarExprUnorderedMap& args_;
  std::unordered_map<Stmt, std::vector<Stmt>>& stencil_fors_;
  uint32_t& unroll_factor_;
  int pass_{0};
};  // class StencilFinder
}  // namespace

void FindStencil(Stmt stmt, VarExprUnorderedSet& buffers,
                 VarExprVarExprUnorderedMap& args,
                 std::unordered_map<Stmt, std::vector<Stmt>>& stencil_fors,
                 uint32_t& unroll_factor) {
  StencilFinder finder(buffers, args, stencil_fors, unroll_factor);
  // 1st-pass mutates the Stmt to unroll innner-loop.
  // Stmt new_stmt = stencil->Mutate(s);
  stmt = finder.Mutate(stmt);
  finder.set_pass(1);
  // 2nd-pass retrieves the stencil loops. Has to separate the two passes or the
  // retrieved loops won't be able to see the mutations.
  // LOG(INFO) << "Mutated stencil stmt: \n" << stmt;
  finder.Mutate(stmt);
}

// Return all Stencil IR nodes in stmt as a vector.
vector<const Stencil*> FindStencil(Stmt stmt) {
  vector<const Stencil*> stencils;

  class StencilCollector final : public IRVisitor {
   public:
    StencilCollector(std::vector<const Stencil*>& stencils)
        : stencils_(stencils) {}

    void Visit_(const Stencil* op) {
      stencils_.push_back(op);
      IRVisitor::Visit_(op);
    }

   private:
    std::vector<const Stencil*>& stencils_;
  };

  StencilCollector(stencils).Visit(stmt);
  return stencils;
}

std::vector<const Load*> FindLoads(Stmt body) {
  std::vector<const Load*> loads;
  LoadsCollector visitor(loads);
  visitor.Visit(body);
  return loads;
}

std::vector<const Load*> FindLoads(Expr body) {
  std::vector<const Load*> loads;
  LoadsCollector visitor(loads);
  visitor.Visit(body);
  return loads;
}

void FindLoads(Stmt body, std::vector<const Load*>& loads) {
  LoadsCollector visitor(loads);
  visitor.Visit(body);
}

std::vector<const Store*> FindStores(Stmt body) {
  std::vector<const Store*> stores;
  std::unordered_map<const Store*, std::vector<const LetStmt*>> store_let_stmts;
  StoresCollector visitor(stores, store_let_stmts);
  visitor.Visit(body);
  return stores;
}

std::vector<const Store*> FindStores(
    Stmt body, std::unordered_map<const Store*, std::vector<const LetStmt*>>&
                   store_let_stmts) {
  std::vector<const Store*> stores;
  StoresCollector visitor(stores, store_let_stmts);
  visitor.Visit(body);
  return stores;
}

}  // namespace soda
}  // namespace ir
}  // namespace TVM
