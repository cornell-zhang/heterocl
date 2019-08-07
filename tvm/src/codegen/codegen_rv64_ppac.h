/*
    author Guyue Huang (gh424@cornell.edu)
 */
 
#ifndef TVM_CODEGEN_CODEGEN_RV64_PPAC_H_
#define TVM_CODEGEN_CODEGEN_RV64_PPAC_H_

#include <tvm/codegen.h>
#include "./codegen_c.h"

namespace TVM {
namespace codegen {

class CodeGenRV64PPAC : public CodeGenC {
  public:
    void PrintType(Type t, std::ostream& os) override;
};

}
}

#endif   //TVM_CODEGEN_CODEGEN_RV64_PPAC_H_