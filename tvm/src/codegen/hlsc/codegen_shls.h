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

  // Expr Printing
  void VisitExpr_(const Load* op, std::ostream& os);
  
  // Misc
  //void GenForStmt(const For* op, std::string pragma, bool before);
  std::string GetBufferRef(Type t, const Variable* buffer, Expr index);
  void PrintVecStore(const Variable* buffer, Type t, Expr base, const std::string& value);

 private:
  std::map<std::string, bool> _is_inport;

};


} // namespace codegen
} // namespace TVM

#endif // CODEGEN_HLSC_CODEGEN_SHLS_H_