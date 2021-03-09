/*!
 * Copyright (c) 2021 by Contributors
 * \file codegen_vhls.h
 * \brief Generate Stratus HLS kernel code.
 */
#ifndef CODEGEN_HLSC_CODEGEN_SHLS_H_
#define CODEGEN_HLSC_CODEGEN_SHLS_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <fstream>
#include <string>
#include "../merlinc/codeanalys_merlinc.h"
#include "./codegen_hlsc.h"
#include "./codegen_vhls.h"

namespace TVM {
namespace codegen {

class CodeGenStratusHLS final : public CodeGenVivadoHLS {
 public:
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  void PrintType(Type t, std::ostream& os);

  // Stmt Printing
  void VisitStmt_(const For* op);
  void VisitStmt_(const Store* op);  
  void VisitStmt_(const Allocate* op);
  void VisitStmt_(const Partition* op);

  // Expr Printing
  //void VisitExpr_(const Load* op, std::ostream& os);

  // Finish
  std::string Finish();
  
  // Misc
  std::string GetBufferRef(Type t, const Variable* buffer, Expr index);
  
  // Formatting
  void PrintIndentHeader();
  void PrintIndentCtor();
  int  BeginScopeHeader();
  void EndScopeHeader(int scope_id);


 private:
  std::map<std::string, bool> _is_inport;
  std::list<std::string> _port_names;
  int h_indent_{0};
  std::vector<bool> h_scope_mark_;
  std::ostringstream ctor_stream;
};


} // namespace codegen
} // namespace TVM

#endif // CODEGEN_HLSC_CODEGEN_SHLS_H_