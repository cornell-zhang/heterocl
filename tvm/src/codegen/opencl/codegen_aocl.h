/*!
 *  Copyright (c) 2019 by Contributors
 */
#ifndef CODEGEN_OPENCL_CODEGEN_AOCL_H_
#define CODEGEN_OPENCL_CODEGEN_AOCL_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include "./codegen_opencl.h"

namespace TVM {
namespace codegen {

class CodeGenAOCL : public CodeGenOpenCL {
 public:
  CodeGenAOCL() {}
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  void PrintType(Type t, std::ostream& os) override;  // NOLINT(*)

  void VisitStmt_(const Allocate* op) override;      // NOLINT(*)
  void VisitStmt_(const For* op) override;           // NOLINT(*)
  void VisitStmt_(const StreamStmt* op) override;    // NOLINT(*)
  void VisitStmt_(const KernelDef* op) override;     // NOLINT(*)
  void VisitStmt_(const KernelStmt* op) override;    // NOLINT(*)
  void VisitStmt_(const ExternModule* op) override;  // NOLINT(*)

  void VisitExpr_(const StreamExpr* op,
                  std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const KernelExpr* op,
                  std::ostream& os) override;  // NOLINT(*)

 private:
  // whether to enable streaming
  bool stream_pragma{false};
  // map from kernel name to set of streamed arg position index
  std::unordered_map<std::string, std::unordered_set<int>> stream_arg_pos;
};
}  // namespace codegen
}  // namespace TVM

#endif  // CODEGEN_OPENCL_CODEGEN_AOCL_H_
