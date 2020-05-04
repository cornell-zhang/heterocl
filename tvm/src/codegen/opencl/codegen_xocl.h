#ifndef TVM_CODEGEN_CODEGEN_XOCL_H_
#define TVM_CODEGEN_CODEGEN_XOCL_H_

# include <tvm/codegen.h>
# include <tvm/packed_func_ext.h>
# include "./codegen_opencl.h"

namespace TVM {
namespace codegen {

class CodeGenXOCL : public CodeGenOpenCL {
  public:
    CodeGenXOCL(){}
    void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);

    void PrintType(Type t, std::ostream& os) override; //NOLINT(*)
    void PrintStorageScope(const std::string& scope, std::ostream& os) override; //NOLINT(*)

    void VisitStmt_(const For* op) override; //NOLINT(*)
    void VisitStmt_(const Partition* op) override; //NOLINT(*)
    void VisitStmt_(const StreamStmt* op) override; //NOLINT(*)
    void VisitStmt_(const ExternModule* op) override; //NOLINT(*)

    void VisitExpr_(const StreamExpr* op, std::ostream& os) override; //NOLINT(*)
   
};
} // namespace codegen
} // namespace TVM

#endif // TVM_CODEGEN_CODEGEN_XOCL_H_
