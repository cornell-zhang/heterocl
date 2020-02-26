/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.h
 * \brief Generate Vivado HLS kernel code.
 */
#ifndef TVM_CODEGEN_CODEGEN_VHLS_H_
#define TVM_CODEGEN_CODEGEN_VHLS_H_

#include <fstream>
#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "./codegen_hlsc.h"
#include "../merlinc/codeanalys_merlinc.h"

namespace TVM {
namespace codegen {

class CodeGenVivadoHLS final : public CodeGenHLSC {
 public:
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  void PrintType(Type t, std::ostream& os) override;
  
  void VisitExpr_(const GetBit* op, std::ostream& os) override;
  void VisitExpr_(const GetSlice* op, std::ostream& os) override;
  void VisitExpr_(const StreamExpr* op, std::ostream& os) override;

  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const Store* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const Partition* op) override;
  void VisitStmt_(const Stencil* op) override;
  void VisitStmt_(const StreamStmt* op) override;
  void VisitStmt_(const KernelDef* op) override;
  void VisitStmt_(const KernelStmt* op) override;

 private:
  std::ofstream soda_header_;
};

}  // namespace codegen
}  // namespace TVM

#endif  // TVM_CODEGEN_CODEGEN_VHLS_H_
