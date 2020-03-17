#include "./codegen_aocl.h"
#include "./codegen_xocl.h"
#include "./codegen_aocl_host.h"
#include "./codegen_xocl_host.h"
#include "../build_common.h"
#include "../merlinc/codeanalys_merlinc.h"

namespace TVM {
namespace codegen {

#if HCL_SDACCEL_RUNTIME
#endif

template<class CodeGen>
std::string BuildOpenCL(
    Array<LoweredFunc> funcs,int output_mode){
  using TVM::runtime::Registry;
  CodeAnalysMerlinC ca;
  CodeGen cg;
  for(LoweredFunc f: funcs){
      ca.AddFunction(f);
      str2tupleMap<std::string, Type>map_arg_type;
      map_arg_type = ca.Finish();
      cg.AddFunction(f, map_arg_type);
  }

  std::string code;
  switch (output_mode) {
    case 0: {code = cg.Finish(); break;}
    case 1: {code = cg.GetHost(); break;}
    case 2: {code = cg.GetDevice(); break;}
    default:
      LOG(FATAL) << "Unsupported output mode";
  }
  return code;
}

TVM_REGISTER_API("codegen.build_xocl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    if (args.size() == 1) {
      *rv = BuildOpenCL<CodeGenXOCL>(args[0], 0);
    } else {
      CHECK(args.size() == 2);
      *rv = BuildOpenCL<CodeGenXOCLHost>(args[0], 
          static_cast<int>(args[1]));
    } 
  });

TVM_REGISTER_API("codegen.build_aocl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    if (args.size() == 1) {
      *rv = BuildOpenCL<CodeGenAOCL>(args[0], 0);
    } else {
      CHECK(args.size() == 2);
      *rv = BuildOpenCL<CodeGenAOCL>(args[0], 
          static_cast<int>(args[1]));
    } 
  });
} // namespace codegen
} // namespace TVM
