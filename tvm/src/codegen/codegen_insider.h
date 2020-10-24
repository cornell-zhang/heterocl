/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_opencl.h
 * \brief Generate OpenCL device code.
 */
#ifndef TVM_CODEGEN_CODEGEN_MERLINC_H_
#define TVM_CODEGEN_CODEGEN_MERLINC_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "./merlinc/codeanalys_merlinc.h"
#include "./codegen_c.h"

namespace TVM {
namespace codegen {

class CodeGenInsider final : public CodeGenC {
 public:
  CodeGenInsider();
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  std::string Finish();

  // override print thread tag.
  void InitFuncState(LoweredFunc f) final;
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  void PrintStorageScope(const std::string& scope, std::ostream& os) final; // NOLINT(*)
  void PrintStorageSync(const Call* op) final;  // NOLINT(*)
  void PrintType(Type t, std::ostream& os) final; // NOLINT(*)
  void PrintVecStore(const Variable* buffer,
                     Type t, Expr base,
                     const std::string& value) final;  // NOLINT(*)
  // the address of load/store
  void PrintVecAddr(const Variable* buffer, Type t,
                    Expr base, std::ostream& os);  // NOLINT(*)
  void GenForStmt(const For* op, std::string pragma, bool before);
  // overload visitor
  void VisitStmt_(const For* op) override;
  void VisitExpr_(const Broadcast* op, std::ostream& os) final; // NOLINT(*)
  void VisitStmt_(const LetStmt* op) final; // NOLINT(*)
  void VisitStmt_(const IfThenElse* op) final; // NOLINT(*)
  void VisitStmt_(const Store* op) final; // NOLINT(*)
  void VisitExpr_(const Load* op, std::ostream& os) final; // NOLINT(*)
 private:
  std::unordered_set<std::string> top_args;
};

}  // namespace codegen
}  // namespace TVM

#endif  // TVM_CODEGEN_CODEGEN_MERLINC_H_
