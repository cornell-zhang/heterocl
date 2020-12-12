#include <fstream>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include "./build_common.h"
#include "./codegen_insider.h"

namespace TVM {
namespace codegen {

std::string BuildInsider(Array<LoweredFunc> funcs) {
  using TVM::runtime::Registry;
  bool output_ssa = false;

  CodeAnalysMerlinC ca;
  CodeGenInsider cg;
  cg.Init(output_ssa);
  for (LoweredFunc f : funcs) {
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    cg.AddFunction(f, map_arg_type);
  }
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_soda_postproc")) {
    code = (*f)(code).operator std::string();
  }

  LOG(WARNING) << "Insider doesn't have runtime, return kernel code";
  return code;
}

TVM_REGISTER_API("codegen.build_insider")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = BuildInsider(args[0]);
    });

}  // namespace codegen
}  // namespace TVM
