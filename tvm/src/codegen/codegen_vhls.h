/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.h
 * \brief Generate Vivado HLS kernel code.
 */
#ifndef TVM_CODEGEN_CODEGEN_VHLS_H_
#define TVM_CODEGEN_CODEGEN_VHLS_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "./codegen_c.h"
#include "./merlinc/codeanalys_merlinc.h"

namespace tvm {
namespace codegen {

class CodeGenVivadoHLS final : public CodeGenC {
 public:
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  void PrintType(Type t, std::ostream& os) override;
  
  void VisitExpr_(const GetBit* op, std::ostream& os) override;
  void VisitExpr_(const GetSlice* op, std::ostream& os) override;

  void VisitStmt_(const Store* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const LetStmt* op) override;
  void VisitStmt_(const IfThenElse* op) override;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_VHLS_H_
