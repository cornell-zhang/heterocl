#ifndef TVM_CODEGEN_CODEGEN_AOCL_H_
#define TVM_CODEGEN_CODEGEN_AOCL_H_

# include <tvm/codegen.h>
# include <tvm/packed_func_ext.h>
# include "./codegen_opencl.h"

namespace TVM {
namespace codegen {

class CodeGenAOCL : public CodeGenOpenCL {
  public:
    CodeGenAOCL(){}
    void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);
    void PrintType(Type t, std::ostream& os) override; //NOLINT(*)

    void VisitStmt_(const For* op) override; //NOLINT(*)
    void VisitStmt_(const StreamStmt* op) override; //NOLINT(*)
    void VisitStmt_(const KernelDef* op) override; //NOLINT(*)

    void VisitExpr_(const StreamExpr* op, std::ostream& os) override;

  private:
    bool stream_pragma{false}; 
};
} // namespace codegen
} // namespace TVM

#endif // TVM_CODEGEN_CODEGEN_AOCL_H_
