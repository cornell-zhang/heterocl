#include <string>
#include <map>
#include <vector>
#include <unordered_set>
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>
#include "./codegen_soda.h"
#include "../runtime/thread_storage_scope.h"
#include "base/Stencil.h"
#include "arithmetic/Polynomial.h"

using std::map;
using std::unordered_set;
using std::vector;
using HalideIR::Internal::ExprUnorderedSet;
using HalideIR::Internal::GetAffineCoeff;
using HalideIR::Internal::Loads;
using HalideIR::Internal::Stencil;
using HalideIR::Internal::Stores;
using HalideIR::Internal::VarExprInt64UnorderedMap;
using HalideIR::Internal::VarExprUnorderedSet;

namespace tvm {
namespace codegen {

void CodeGenSODA::AddFunction(LoweredFunc f) {
  stencil_ = Stencil::GetStencil(f->body);
  if (stencil_ != nullptr) {
    stream<<"kernel: "<<f->name<<"\n";
    // TODO: pass these parameters from outside.
    stream<<"burst width: 512\n";
    stream<<"dram separate: no\n";
    stream<<"dram bank: 1\n";
    stream<<"unroll factor: 8\n";
    stream<<"border: ignore\n";
    stream<<"cluster: none\n";  
    stream<<"iterate: 1\n";  

    auto args = stencil_->GetArgs();
    auto buffers = stencil_->GetBuffers();
    VarExprUnorderedSet inouts;
    for (Var arg : f->args) {
      inouts.insert(args[arg]);
    }
    VarExprUnorderedSet inputs;
    VarExprUnorderedSet outputs;
    VarExprUnorderedSet locals;
    for (auto for_pair: stencil_->GetStencilFors()) {
      unordered_set<Stmt> stores = Stores::GetStores(
        for_pair.second.rbegin()->as<For>()->body);
      for (auto store_stmt : stores) {
        const Store* store = store_stmt.as<Store>();
        if (inouts.count(store->buffer_var) != 0) {
          outputs.insert(store->buffer_var);
          PrintOutputTensor(store_stmt, for_pair.second);
        } else {
          locals.insert(store->buffer_var);
          PrintLocalTensor(store_stmt, for_pair.second);
        }
        ExprUnorderedSet loads = Loads::GetLoads(store->value);
        for (auto load_expr : loads) {
          const Load* load = load_expr.as<Load>();
          if (inouts.count(load->buffer_var) != 0) {
            if (inputs.count(load->buffer_var) == 0) {
              PrintInputTensor(load_expr, for_pair.second);
              inputs.insert(load->buffer_var);
            }
          }
        }
      }
    }
  }
}

void CodeGenSODA::PrintInputTensor(const Expr& load_expr,
                                   const vector<Stmt>& nested_loops) {
  if (const Load* load = load_expr.as<Load>()) {
    stream<<"input "<<load->type<<": ";
    stream<<load->buffer_var.get()->name_hint<<"(";
    bool innermost = true;
    for (auto loop = nested_loops.rbegin();
         loop != nested_loops.rend()-1; ++loop) {
      if (innermost) {
        stream<<loop->as<For>()->extent<<",";
        innermost = false;
      } else {
        stream<<" "<<loop->as<For>()->extent<<",";
      }
    }
    stream<<")\n";
  } else {
    LOG(ERROR)<<"Cannot print anything other and a Load as input tensor.";
  }
}

void PrintIndex(const Expr& index_expr, std::ostream& os) {
  VarExprInt64UnorderedMap affine_coeffs = GetAffineCoeff(index_expr);
  LOG(INFO)<<"print index for "<<index_expr;
  int64_t const_offset = 0;
  if (affine_coeffs.count(VarExpr()) != 0) {
    const_offset = affine_coeffs[VarExpr()];
  }

  map<int64_t, VarExpr> loop_vars;  // Stride is unique.
  for (auto term : affine_coeffs) {
    if (not term.first.same_as(VarExpr())) {
      loop_vars[term.second] = term.first;
    }
  }

  VarExprInt64UnorderedMap indices;
  for (auto loop_var = loop_vars.rbegin();
       loop_var != loop_vars.rend(); ++loop_var) {
    const VarExpr& loop_var_expr = loop_var->second;
    int64_t stride = loop_var->first;
    // Any chance the indices can be preserved?
    if (const_offset > 0) {
      indices[loop_var_expr] = (const_offset+stride/2) / stride;
    } else {
      indices[loop_var_expr] = (const_offset-stride/2) / stride;
    }
    const_offset -= indices[loop_var_expr] * stride;
  }

  bool innermost = true;
  for (auto term : loop_vars) {
    const VarExpr& loop_var_expr = term.second;
    int64_t index = indices[loop_var_expr];
    LOG(INFO)<<"index of "<<loop_var_expr<<" : "<< index;
    if (innermost) {
      os<<index;
      innermost = false;
    } else {
      os<<", "<<index;
    }
  }
}

void CodeGenSODA::PrintLocalOrOutputTensor(
    const Stmt& store_stmt, const vector<Stmt>& nested_loops, bool is_local) {
  const char* type_str = (is_local ? "local" : "output");
  if (const Store* store = store_stmt.as<Store>()) {
    stream<<type_str<<" "<<store->value.type()<<": ";
    stream<<store->buffer_var.get()->name_hint<<"(";
    PrintIndex(store->index, stream);
    stream<<") = ";
    stream<<PrintExpr(store->value);
    stream<<"\n";
  } else {
    LOG(ERROR)<<"Cannot print anything other and a Store as "<<
      type_str<<" tensor.";
  }
}

void CodeGenSODA::VisitExpr_(const Load* op, std::ostream& os) {
  os<<op->buffer_var.get()->name_hint<<"(";
  PrintIndex(op->index, os);
  os<<")";
}

void CodeGenSODA::VisitExpr_(const IntImm* op, std::ostream& os) {
  os<<op->value;
  if (op->type.bits() > 32) {
    os<<"L";
  }
}

void CodeGenSODA::VisitExpr_(const UIntImm* op, std::ostream& os) {
  os<<op->value;
  if (op->type.bits() > 32) {
    os<<"UL";
  }
}

void CodeGenSODA::VisitExpr_(const FloatImm* op, std::ostream& os) {
  os<<op->value;
  if (op->type.bits() < 64) {
    os<<"F";
  }
}

}  // namespace codegen
}  // namespace tvm
