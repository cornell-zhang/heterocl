/*!
 *  Copyright (c) 2018 by Contributors
 *  Build HLS C modules from source.
 * \file tvm/src/codegen/hlsc/build_hlsc.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include "./codegen_hlsc.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {

std::string BuildHLSC(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;

  CodeGenHLSC cg;
  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_hlsc_postproc")) {
    code = (*f)(code).operator std::string();
  }
  LOG(WARNING) << "HLS C doesn't have runtime, return kernel code";
  return code;
}

TVM_REGISTER_API("codegen.build_hlsc")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildHLSC(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
