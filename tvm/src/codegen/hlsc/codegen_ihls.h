/*!
 *  Copyright (c) 2018 by Contributors
 */
#ifndef CODEGEN_HLSC_CODEGEN_IHLS_H_
#define CODEGEN_HLSC_CODEGEN_IHLS_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "./codegen_hlsc.h"

namespace TVM {
namespace codegen {

class CodeGenIntelHLS final : public CodeGenHLSC {
 public:
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  void PrintType(Type t, std::ostream& os) override;

  void VisitExpr_(const GetBit* op, std::ostream& os) override;
  void VisitExpr_(const GetSlice* op, std::ostream& os) override;

  void VisitStmt_(const Store* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const Partition* op) override;
};

}  // namespace codegen
}  // namespace TVM

#endif  // CODEGEN_HLSC_CODEGEN_IHLS_H_
