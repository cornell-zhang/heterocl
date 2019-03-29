/*!
 *  Copyright (c) 2018 by Contributors
 * \file build_vhls.cc
 * \brief Build Intel HLS modules from source.
 */
#include "./codegen_ihls.h"
#include "../build_common.h"

namespace tvm {
namespace codegen {

std::string BuildIntelHLS(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;

  CodeAnalysMerlinC ca;
  CodeGenIntelHLS cg;
  for (LoweredFunc f : funcs) {
    // 1st pass: Analyze AST and collect necessary information
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    // 2nd pass: Generate kernel code
    cg.AddFunction(f, map_arg_type);
  }
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_vhls_postproc")) {
    code = (*f)(code).operator std::string();
  }
  LOG(WARNING) << "Intel HLS doesn't have runtime, return kernel code";
  return code;
}

TVM_REGISTER_API("codegen.build_ihls")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildIntelHLS(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
