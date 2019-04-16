#ifndef TVM_CODEGEN_CODEGEN_SODA_H_
#define TVM_CODEGEN_CODEGEN_SODA_H_

#include <memory>
#include <string>
#include <vector>

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>

#include "../pass/stencil.h"
#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenSODA final : public CodeGenC {
 public:
  void AddFunction(LoweredFunc f);
  std::string Finish() {return CodeGenC::Finish();}

  void PrintInputTensor(const Expr& load_stmt,
                        const std::vector<Stmt>& nested_loops);
  void PrintLet(const Stmt& let_stmt);
  void PrintLocalOrOutputTensor(
      const Stmt& store_stmt, const std::vector<Stmt>& lets,
      const std::vector<Stmt>& nested_loops, bool is_local);
  void PrintLocalTensor(const Stmt& store_stmt, const std::vector<Stmt>& lets,
                        const std::vector<Stmt>& nested_loops) {
    PrintLocalOrOutputTensor(store_stmt, lets, nested_loops, true);
  }
  void PrintOutputTensor(const Stmt& store_stmt, const std::vector<Stmt>& lets,
                         const std::vector<Stmt>& nested_loops) {
    PrintLocalOrOutputTensor(store_stmt, lets, nested_loops, false);
  }

  void VisitExpr_(const Load* op, std::ostream& os);

  // SODA doesn't handle types right now.
  void VisitExpr_(const IntImm* op, std::ostream& os);
  void VisitExpr_(const UIntImm* op, std::ostream& os);
  void VisitExpr_(const FloatImm* op, std::ostream& os);
  void VisitExpr_(const Cast* op, std::ostream& os);

 private:
  std::shared_ptr<HalideIR::Internal::StencilFinder> stencil_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_SODA_H_
