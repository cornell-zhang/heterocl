/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.h
 * \brief Generate Vivado HLS kernel code.
 */
#ifndef CODEGEN_HLSC_CODEGEN_VHLS_H_
#define CODEGEN_HLSC_CODEGEN_VHLS_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <fstream>
#include <string>
#include "./codegen_hlsc.h"

namespace TVM {
namespace codegen {

class CodeGenVivadoHLS final : public CodeGenHLSC {
 public:
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  void PrintType(Type t, std::ostream& os) override;

  void VisitExpr_(const Min* op, std::ostream& os) override;
  void VisitExpr_(const Max* op, std::ostream& os) override;
  void VisitExpr_(const GetBit* op, std::ostream& os) override;
  void VisitExpr_(const GetSlice* op, std::ostream& os) override;
  void VisitExpr_(const StreamExpr* op, std::ostream& os) override;
  void VisitExpr_(const Call* op, std::ostream& os) override;
  void VisitExpr_(const Load* op, std::ostream& os) override;

  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const AttrStmt* op) override;
  void VisitStmt_(const Store* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const Partition* op) override;
  void VisitStmt_(const Stencil* op) override;
  void VisitStmt_(const ExternModule* op) override;
  void VisitStmt_(const StreamStmt* op) override;
  void VisitStmt_(const KernelDef* op) override;
  void VisitStmt_(const KernelStmt* op) override;

 private:
  std::ofstream soda_header_;
  bool sdsoc_mode{false};
  bool extern_c_wrapper{false};
  std::unordered_set<std::string> stream_vars;
};

}  // namespace codegen
}  // namespace TVM

#endif  // CODEGEN_HLSC_CODEGEN_VHLS_H_
