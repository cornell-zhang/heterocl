/*
    author Guyue Huang (gh424@cornell.edu)
 */
 
#ifndef TVM_CODEGEN_CODEGEN_RV64_PPAC_H_
#define TVM_CODEGEN_CODEGEN_RV64_PPAC_H_

#include <tvm/codegen.h>
#include <string>
#include "./merlinc/codeanalys_merlinc.h"
#include "./codegen_c.h"

namespace TVM {
namespace codegen {

class CodeGenRV64PPAC : public CodeGenC {
  public:
    void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
    void PrintType(Type t, std::ostream& os) override;
    void VisitStmt_(const LetStmt* op) override;
    void VisitStmt_(const IfThenElse* op) override;
    //void VisitStmt_(const Allocate* op) override;
    //std::map<const Variable*, Array<Expr> > var_shape_map_;
    
  protected:
    //std::string GetBufferRef(Type t, const Variable* buffer, Expr index);
};

}
}

#endif   //TVM_CODEGEN_CODEGEN_RV64_PPAC_H_