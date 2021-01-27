/*!
 *  Copyright (c) 2017 by Contributors
 *  Build merlinc modules from source.
 * \file build_merlinc.cc
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include <unordered_map>
#include "../build_common.h"
#include "./codeanalys_merlinc.h"
#include "./codegen_merlinc.h"

namespace TVM {
namespace codegen {

std::string BuildMerlinC(Array<LoweredFunc> funcs) {
  using TVM::runtime::Registry;

  CodeAnalysMerlinC ca;
  CodeGenMerlinC cg;
  for (LoweredFunc f : funcs) {
    // 1st pass: Analyze AST and collect necessary information
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();

    // 2nd pass: Generate kernel code
    cg.AddFunction(f, map_arg_type);
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
}  // namespace TVM
