#include "codegen_soda.h"

#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>

#include "../pass/stencil.h"
#include "../runtime/thread_storage_scope.h"
#include "arithmetic/Polynomial.h"

using std::map;
using std::ostringstream;
using std::unordered_map;
using std::unordered_set;
using std::vector;

namespace TVM {
namespace codegen {

using namespace ir;
using Halide::Internal::VarExprInt64UnorderedMap;

void CodeGenSODA::AddFunction(LoweredFunc f) {
  VarExprUnorderedSet buffers;
  VarExprVarExprUnorderedMap args;
  std::unordered_map<Stmt, std::vector<Stmt> > stencil_fors;
  uint32_t unroll_factor;

  soda::FindStencil(f->body, buffers, args, stencil_fors, unroll_factor);

  if (stencil_fors.size()) {

    stream<<"kernel: "<<f->name<<"\n";
    // TODO: pass these parameters from outside.
    stream<<"burst width: 512\n";
    stream<<"unroll factor: "<<unroll_factor<<"\n";
    stream<<"iterate: 1\n";  

    VarExprUnorderedSet inouts;
    for (Var arg : f->args) {
      inouts.insert(args[arg]);
    }
    VarExprUnorderedSet inputs;
    VarExprUnorderedSet outputs;
    VarExprUnorderedSet locals;
    for (const auto& for_pair: stencil_fors) {
      unordered_map<const Store*, vector<const LetStmt*> > lets;
      vector<const Store*> stores = soda::FindStores(
          for_pair.second.rbegin()->as<For>()->body, lets);
      for (auto store : stores) {
        if (inouts.count(store->buffer_var) != 0) {
          outputs.insert(store->buffer_var);
          PrintOutputTensor(store, lets[store], for_pair.second);
        } else {
          locals.insert(store->buffer_var);
          PrintLocalTensor(store, lets[store], for_pair.second);
        }
        vector<const Load*> loads_in_lets;
        for (auto let : lets[store]) {
          soda::FindLoads(let->body, loads_in_lets);
        }
        vector<const Load*> loads = soda::FindLoads(store->value);
        loads.insert(loads.end(), loads_in_lets.begin(), loads_in_lets.end());
        for (auto load : loads) {
          if (inouts.count(load->buffer_var) != 0) {
            if (inputs.count(load->buffer_var) == 0) {
              PrintInputTensor(load, for_pair.second);
              inputs.insert(load->buffer_var);
            }
          }
        }
      }
    }
  }
}

void CodeGenSODA::PrintSODA(
      std::string name, int burst_width, int unroll_factor, int num_iteration,
      Stmt stmt, VarExprUnorderedSet& inputs, VarExprUnorderedSet& outputs) {
  VarExprUnorderedSet buffers;
  VarExprVarExprUnorderedMap args;
  std::unordered_map<Stmt, std::vector<Stmt> > stencil_fors;
  uint32_t unroll_factor_;

  soda::FindStencil(stmt, buffers, args, stencil_fors, unroll_factor_);

  if (stencil_fors.size()) {
    stream<<"kernel: " << name << "\n";
    stream<<"burst width: " << burst_width << "\n";
    stream<<"unroll factor: "<< unroll_factor << "\n";
    stream<<"iterate: " << num_iteration << "\n";
    VarExprUnorderedSet printed_inputs;

    for (const auto& for_pair: stencil_fors) {
      std::unordered_map<const Store*, std::vector<const LetStmt*> > lets;
      std::vector<const Store*> stores = soda::FindStores(
          for_pair.second.rbegin()->as<For>()->body, lets);
      for (auto store : stores) {
        if (outputs.count(store->buffer_var)) {
          PrintOutputTensor(store, lets[store], for_pair.second);
        } else {
          PrintLocalTensor(store, lets[store], for_pair.second);
        }
        std::vector<const Load*> loads_in_lets;
        for (auto let : lets[store]) {
          soda::FindLoads(let->body, loads_in_lets);
        }
        std::vector<const Load*> loads = soda::FindLoads(store->value);
        loads.insert(loads.end(), loads_in_lets.begin(), loads_in_lets.end());
        for (auto load : loads) {
          if (inputs.count(load->buffer_var) && 
              !printed_inputs.count(load->buffer_var)) {
            PrintInputTensor(load, for_pair.second);
            printed_inputs.insert(load->buffer_var);
          }
        }
      }
    }
  }
}

void CodeGenSODA::PrintLet(const LetStmt* let_stmt) {
  stream << AllocVarID(let_stmt->var.get());
  stream << " = ";
  PrintExpr(let_stmt->value, stream);
  stream << "\n";
}

void CodeGenSODA::PrintInputTensor(const Load* load,
                                   const vector<Stmt>& nested_loops) {
  var_type_map_[load->buffer_var.get()] = load->type;
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
  stream<<" *)\n";
}

void PrintIndex(const Expr& index_expr, std::ostream& os) {
  VarExprInt64UnorderedMap affine_coeffs = GetAffineCoeff(index_expr);
  //LOG(INFO)<<"print index for "<<index_expr;
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
    //LOG(INFO)<<"index of "<<loop_var_expr<<" : "<< index;
    if (innermost) {
      os<<index;
      innermost = false;
    } else {
      os<<", "<<index;
    }
  }
}

void CodeGenSODA::PrintLocalOrOutputTensor(
    const Store* store, const vector<const LetStmt*>& lets,
    const vector<Stmt>& nested_loops, bool is_local) {
  const char* type_str = (is_local ? "local" : "output");
  var_type_map_[store->buffer_var.get()] = store->value.type();
  stream<<type_str<<" "<<store->value.type()<<":\n";
  for (auto let : lets) {
    stream<<"  ";
    PrintLet(let);
  }
  stream<<"  "<<store->buffer_var.get()->name_hint<<"(";
  PrintIndex(store->index, stream);
  stream<<") = ";
  stream<<PrintExpr(store->value);
  stream<<"\n";
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
  ostringstream tmp;
  tmp<<std::showpoint<<op->value;
  os<<tmp.str();
  if (op->type.bits() < 64) {
    os<<"F";
  }
}

void CodeGenSODA::VisitExpr_(const Cast* op, std::ostream& os) {
  os<<op->type<<"(";
  PrintExpr(op->value, os);
  os<<")";
}

}  // namespace codegen
}  // namespace TVM
