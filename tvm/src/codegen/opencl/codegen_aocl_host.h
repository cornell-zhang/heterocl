/*!
 *  Copyright (c) 2020 by Contributors
 * \file codegen_aocl_host.h
 * \brief Generate cpp kernel code for AOCL.
 */
#ifndef CODEGEN_OPENCL_CODEGEN_AOCL_HOST_H_
#define CODEGEN_OPENCL_CODEGEN_AOCL_HOST_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "../codegen_c.h"

namespace TVM {
namespace codegen {

class CodeGenAOCLHost : public CodeGenC {
 public:
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  void PrintType(Type t, std::ostream& os) override;

  void VisitExpr_(const Min* op, std::ostream& os) override;
  void VisitExpr_(const Max* op, std::ostream& os) override;

  void VisitStmt_(const For* op) override;
  void VisitStmt_(const IfThenElse* op) override;
  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const KernelStmt* op) override;
  void VisitStmt_(const Store* op) override;

  void GenForStmt(const For* op, std::string pragma, bool before);

 protected:
  std::string GetBufferRef(Type t, const Variable* buffer, Expr index);
};

}  // namespace codegen
}  // namespace TVM

#endif  // CODEGEN_OPENCL_CODEGEN_AOCL_HOST_H_
