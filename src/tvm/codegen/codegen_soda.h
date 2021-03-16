/*!
 *  Copyright (c) 2019 by Contributors
 */
#ifndef CODEGEN_CODEGEN_SODA_H_
#define CODEGEN_CODEGEN_SODA_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>

#include <memory>
#include <string>
#include <vector>

#include "../pass/stencil.h"
#include "codegen_c.h"

namespace TVM {
namespace codegen {

class CodeGenSODA final : public CodeGenC {
 public:
  void AddFunction(LoweredFunc f);
  std::string Finish() { return CodeGenC::Finish(); }

  void PrintSODA(const Stencil* stencil, std::string* kernel_name = nullptr);
  void PrintLet(const LetStmt* let_stmt, std::ostream& os);
  void PrintInputTensor(const Load* load,
                        const std::vector<Stmt>& nested_loops);
  void PrintLocalOrOutputTensor(const Store* store,
                                const std::vector<const LetStmt*>& lets,
                                const std::vector<Stmt>& nested_loops,
                                bool is_local);
  void PrintLocalTensor(const Store* store,
                        const std::vector<const LetStmt*>& lets,
                        const std::vector<Stmt>& nested_loops) {
    PrintLocalOrOutputTensor(store, lets, nested_loops, true);
  }
  void PrintOutputTensor(const Store* store,
                         const std::vector<const LetStmt*>& lets,
                         const std::vector<Stmt>& nested_loops) {
    PrintLocalOrOutputTensor(store, lets, nested_loops, false);
  }

  void PrintSelect(const Expr& condition, const Expr& true_value,
                   const Expr& false_value, std::ostream& os);

  void VisitExpr_(const Load* op, std::ostream& os) final;

  void VisitExpr_(const Call* op, std::ostream& os) final;
  void VisitExpr_(const Select* op, std::ostream& os) final;

  // SODA doesn't handle types right now.
  void VisitExpr_(const IntImm* op, std::ostream& os) final;
  void VisitExpr_(const UIntImm* op, std::ostream& os) final;
  void VisitExpr_(const FloatImm* op, std::ostream& os) final;
  void VisitExpr_(const Cast* op, std::ostream& os) final;

  std::map<const Variable*, Type> var_type_map_;

  // SODA does not allow interleaving local and output tensors
  // therefore we need to rememeber all tensors before actually printing them
  std::string input_tensors;
  std::string local_tensors;
  std::string output_tensors;
};

}  // namespace codegen
}  // namespace TVM

#endif  // CODEGEN_CODEGEN_SODA_H_
