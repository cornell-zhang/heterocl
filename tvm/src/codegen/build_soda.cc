#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include "./codegen_soda.h"
#include "./build_common.h"

namespace tvm {
namespace codegen {

std::string BuildSODA(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenSODA cg;
  cg.Init(output_ssa);
  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_soda_postproc")) {
    code = (*f)(code).operator std::string();
  }
  LOG(WARNING) << "SODA doesn't have runtime, return kernel code";
  return code;
}

TVM_REGISTER_API("codegen.build_soda")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildSODA(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
