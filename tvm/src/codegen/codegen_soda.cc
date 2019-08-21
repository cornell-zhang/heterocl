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
  VarExprUnorderedSet inouts;
  for (Var arg : f->args) {
    inouts.insert(arg);
  }
  PrintSODA(/* kernel: */f->name, /* burst width: */512, /* unroll factor: */0,
            /* iterate: */1, /* stmt: */f->body, /* inputs: */ inouts,
            /* outputs: */inouts, /* map_args: */ true);
  return;
}

void CodeGenSODA::PrintSODA(
      std::string name, int burst_width, int unroll_factor, int num_iteration,
      Stmt stmt, const VarExprUnorderedSet& inputs,
      const VarExprUnorderedSet& outputs, bool map_args) {
  VarExprUnorderedSet buffers;
  VarExprVarExprUnorderedMap args;
  std::unordered_map<Stmt, std::vector<Stmt> > stencil_fors;
  uint32_t unroll_factor_from_loop;

  soda::FindStencil(stmt, buffers, args, stencil_fors, unroll_factor_from_loop);

  // if unroll_factor is 0, use the detected value from the loops
  if (unroll_factor == 0) {
    unroll_factor = unroll_factor_from_loop;
  }

  // if map_args is true, use the detected argument mapping
  VarExprUnorderedSet new_inputs;
  VarExprUnorderedSet new_outputs;
  const VarExprUnorderedSet* inputs_ptr = &inputs;
  const VarExprUnorderedSet* outputs_ptr = &outputs;
  if (map_args) {
    for (auto arg : inputs) {
      if (args.count(arg)) {
        new_inputs.insert(args[arg]);
      } else {
        new_inputs.insert(arg);
      }
    }
    for (auto arg : outputs) {
      if (args.count(arg)) {
        new_outputs.insert(args[arg]);
      } else {
        new_outputs.insert(arg);
      }
    }
    inputs_ptr = &new_inputs;
    outputs_ptr = &new_outputs;
  }

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
        if (outputs_ptr->count(store->buffer_var)) {
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
          if (inputs_ptr->count(load->buffer_var) &&
              !printed_inputs.count(load->buffer_var)) {
            PrintInputTensor(load, for_pair.second);
            printed_inputs.insert(load->buffer_var);
          }
        }
      }
    }
    stream << input_tensors;
    stream << local_tensors;
    stream << output_tensors;
  }
}

void CodeGenSODA::PrintLet(const LetStmt* let_stmt, std::ostream& os) {
  os << AllocVarID(let_stmt->var.get());
  os << " = ";
  PrintExpr(let_stmt->value, os);
  os << "\n";
}

void CodeGenSODA::PrintInputTensor(const Load* load,
                                   const vector<Stmt>& nested_loops) {
  ostringstream os;
  var_type_map_[load->buffer_var.get()] = load->type;
  os << "input " << load->type << ": ";
  os << load->buffer_var.get()->name_hint << "(";
  bool innermost = true;
  for (auto loop = nested_loops.rbegin();
       loop != nested_loops.rend()-1; ++loop) {
    if (innermost) {
      os << loop->as<For>()->extent << ",";
      innermost = false;
    } else {
      os << " " << loop->as<For>()->extent << ",";
    }
  }
  os << " *)\n";
  input_tensors += os.str();
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
  ostringstream os;
  const char* type_str = (is_local ? "local" : "output");
  var_type_map_[store->buffer_var.get()] = store->value.type();
  os << type_str << " " << store->value.type() << ":\n";
  for (auto let : lets) {
    os << "  ";
    PrintLet(let, os);
  }
  os << "  " << store->buffer_var.get()->name_hint << "(";
  PrintIndex(store->index, os);
  os << ") = ";
  PrintExpr(store->value, os);
  os << "\n";
  if (is_local) {
    local_tensors += os.str();
  } else {
    output_tensors += os.str();
  }
}

void CodeGenSODA::VisitExpr_(const Load* op, std::ostream& os) {
  os<<op->buffer_var.get()->name_hint<<"(";
  PrintIndex(op->index, os);
  os<<")";
}

void CodeGenSODA::PrintSelect(const Expr& condition, const Expr& true_value,
                              const Expr& false_value, std::ostream& os) {
  os << "select(";
  PrintExpr(condition, os);
  os << ", ";
  PrintExpr(true_value, os);
  os << ", ";
  PrintExpr(false_value, os);
  os << ")";
}

void CodeGenSODA::VisitExpr_(const Call* op, std::ostream& os) {
  if (op->is_intrinsic(intrinsic::tvm_if_then_else)) {
    PrintSelect(op->args[0], op->args[1], op->args[2], os);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenSODA::VisitExpr_(const Select* op, std::ostream& os) {
  PrintSelect(op->condition, op->true_value, op->false_value, os);
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
