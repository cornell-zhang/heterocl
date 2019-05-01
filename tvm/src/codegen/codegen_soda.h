#ifndef TVM_CODEGEN_CODEGEN_SODA_H_
#define TVM_CODEGEN_CODEGEN_SODA_H_

#include <memory>
#include <string>
#include <vector>

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>

#include "../pass/stencil.h"
#include "codegen_c.h"

namespace TVM {
namespace codegen {

class CodeGenSODA final : public CodeGenC {
 public:
  void AddFunction(LoweredFunc f);
  std::string Finish() {return CodeGenC::Finish();}

  void PrintSODA(
      std::string name, int burst_width, int unroll_factor, int num_iteration,
      Stmt stmt, VarExprUnorderedSet& inputs, VarExprUnorderedSet& outputs);
  void PrintLet(const LetStmt* let_stmt);
  void PrintInputTensor(const Load* load,
      const std::vector<Stmt>& nested_loops);
  void PrintLocalOrOutputTensor(
      const Store* store, const std::vector<const LetStmt*>& lets,
      const std::vector<Stmt>& nested_loops, bool is_local);
  void PrintLocalTensor(const Store* store, const std::vector<const LetStmt*>& lets,
                        const std::vector<Stmt>& nested_loops) {
    PrintLocalOrOutputTensor(store, lets, nested_loops, true);
  }
  void PrintOutputTensor(const Store* store, const std::vector<const LetStmt*>& lets,
                         const std::vector<Stmt>& nested_loops) {
    PrintLocalOrOutputTensor(store, lets, nested_loops, false);
  }

  void VisitExpr_(const Load* op, std::ostream& os);

  // SODA doesn't handle types right now.
  void VisitExpr_(const IntImm* op, std::ostream& os);
  void VisitExpr_(const UIntImm* op, std::ostream& os);
  void VisitExpr_(const FloatImm* op, std::ostream& os);
  void VisitExpr_(const Cast* op, std::ostream& os);

  std::map<const Variable*, Type> var_type_map_;
};

}  // namespace codegen
}  // namespace TVM

#endif  // TVM_CODEGEN_CODEGEN_SODA_H_
