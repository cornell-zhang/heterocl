/*
 * \file codegen_rv64_ppac.h
 */
 
#ifndef TVM_CODEGEN_CODEGEN_RV64_PPAC_H_
#define TVM_CODEGEN_CODEGEN_RV64_PPAC_H_

#include <tvm/codegen.h>
#include <string>
#include "../codegen_c.h"
#include "../merlinc/codeanalys_merlinc.h"

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

#endif   //TVM_CODEGEN_CODEGEN_RV64_PPAC_H_