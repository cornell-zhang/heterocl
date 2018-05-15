/*!
 *  Copyright (c) 2017 by Contributors
 *  Build merlinc modules from source.
 * \file build_merlinc.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
//#include "../codeanalys_common.h"
#include "./codegen_merlinc.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {

std::string BuildMerlinC(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;

  // 1st pass: Analyze AST and collect necessary information
//  CodeAnalysCommon ca;
//  for (LoweredFunc f : funcs) {
//    ca.AddFunction(f);
//  }
//  std::string code = ca.Finish();

  // 2nd pass: Generate kernel code
  CodeGenMerlinC cg;
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
