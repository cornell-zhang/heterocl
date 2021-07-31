/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_rv64_ppac.h
 */

#ifndef CODEGEN_PPAC_CODEGEN_RV64_PPAC_H_
#define CODEGEN_PPAC_CODEGEN_RV64_PPAC_H_

#include <tvm/codegen.h>
#include <string>
#include "../codegen_c.h"

namespace TVM {
namespace codegen {

class CodeGenRV64PPAC : public CodeGenC {
 public:
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
  void PrintType(Type t, std::ostream& os) override;
  void VisitStmt_(const LetStmt* op) override;
  void VisitStmt_(const IfThenElse* op) override;
  void VisitStmt_(const For* op) override;
};

}  // namespace codegen
}  // namespace TVM

#endif  // CODEGEN_PPAC_CODEGEN_RV64_PPAC_H_
