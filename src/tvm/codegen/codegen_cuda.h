/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_cuda.h
 * \brief Utility to generate cuda code
 */
#ifndef CODEGEN_CODEGEN_CUDA_H_
#define CODEGEN_CODEGEN_CUDA_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "./codegen_c.h"

namespace TVM {
namespace codegen {

class CodeGenCUDA final : public CodeGenC {
 public:
  CodeGenCUDA();
  void Init(bool output_ssa);
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  // override behavior
  void VisitStmt_(const ir::For* op) final;
  void PrintStorageSync(const Call* op) final;
  void PrintStorageScope(const std::string& scope,
                         std::ostream& os) final;  // NOLINT(*)
  void PrintVecBinaryOp(const std::string& op, Type t, Expr lhs, Expr rhs,
                        std::ostream& os) final;   // NOLINT(*)
  void PrintType(Type t, std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemLoad(const std::string& vec, Type t, int i,
                        std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(const std::string& vec, Type t, int i,
                         const std::string& value) final;
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  // overload visitor
  void VisitExpr_(const Ramp* op, std::ostream& os) final;       // NOLINT(*)
  void VisitExpr_(const Broadcast* op, std::ostream& os) final;  // NOLINT(*)
  void VisitStmt_(const Evaluate* op) final;

 private:
  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;
};

}  // namespace codegen
}  // namespace TVM

#endif  // CODEGEN_CODEGEN_CUDA_H_
