/*!
 *  Copyright (c) 2018 by Contributors
 */
#ifndef CODEGEN_HLSC_CODEGEN_HLSC_H_
#define CODEGEN_HLSC_CODEGEN_HLSC_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "../codegen_c.h"

namespace TVM {
namespace codegen {

class CodeGenHLSC : public CodeGenC {
 public:
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);

  void VisitExpr_(const Min* op, std::ostream& os) override;
  void VisitExpr_(const Max* op, std::ostream& os) override;

  void VisitStmt_(const LetStmt* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const IfThenElse* op) override;
  void VisitStmt_(const Allocate* op) override;

  void GenForStmt(const For* op, std::string pragma, bool before);
  bool enable_native_dtype{false};

 protected:
  std::string GetBufferRef(Type t, const Variable* buffer, Expr index);
};

}  // namespace codegen
}  // namespace TVM

#endif  // CODEGEN_HLSC_CODEGEN_HLSC_H_
