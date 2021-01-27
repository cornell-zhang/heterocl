/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_llvm.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/api_registry.h>
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <sstream>
#include "./intrin_rule_llvm.h"

namespace TVM {
namespace codegen {

inline void DispatchExternOCML(const TVMArgs& args, TVMRetValue* rv) {
  Expr e = args[0];
  using namespace ir;
  const Call* call = e.as<Call>();
  CHECK(call != nullptr);
  std::ostringstream intrinsic_name;
  intrinsic_name << "__ocml_" << call->name << "_f" << call->type.bits();
  *rv = Call::make(call->type, intrinsic_name.str(), call->args,
                   Call::PureExtern);
}

namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.exp").set_body(DispatchExternOCML);

// On AMD GPU, fma is slower than mac
// removing fma dispatch allows backend to generate faster mac instruction
TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.fma")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.log").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.sqrt").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.pow").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.tanh").set_body(DispatchExternOCML);
}  // namespace llvm
}  // namespace codegen
}  // namespace TVM

#endif  // LLVM_VERSION
