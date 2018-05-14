/*!
 *  Copyright (c) 2017 by Contributors
 *  Build merlinc modules from source.
 * \file build_merlinc.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include "./codegen_merlinc.h"
#include "./build_common.h"

namespace tvm {
namespace codegen {

std::string BuildMerlinC(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenMerlinC cg;
  cg.Init(output_ssa);
  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_merlinc_postproc")) {
    code = (*f)(code).operator std::string();
  }
  LOG(WARNING) << "MerlinC doesn't have runtime, return kernel code";
  return code;
}

TVM_REGISTER_API("codegen.build_merlinc")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildMerlinC(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
