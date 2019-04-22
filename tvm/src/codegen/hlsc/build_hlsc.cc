/*!
 *  Copyright (c) 2018 by Contributors
 * \file build_vhls.cc
 * \brief Build HLS C modules from source.
 */
#include "./codegen_ihls.h"
#include "./codegen_vhls.h"
#include "../build_common.h"

//TODO: add ifdef to guard incorrect usgae
#include <tvm/runtime/packed_func.h>

namespace tvm {
namespace codegen {

using runtime::ModuleNode;
using runtime::PackedFunc;

class VivadoHLSModuleNode final : public ModuleNode {
 public:
  VivadoHLSModuleNode(std::string& test_file) 
    : test_file_(test_file) {}

  const char* type_key() const {
    return "vivado_hls_csim";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    return PackedFunc([](TVMArgs args, TVMRetValue* rv){
        *rv = 1;
      });
  }

 private:
  std::string test_file_;
};

runtime::Module BuildVivadoHLSCSim(Array<LoweredFunc> funcs) {
  CodeAnalysMerlinC ca;
  CodeGenVivadoHLS cg;
  for (LoweredFunc f : funcs) {
    // 1st pass: Analyze AST and collect necessary information
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    // 2nd pass: Generate kernel code
    cg.AddFunction(f, map_arg_type);
  }
  std::string code = cg.Finish();

  std::shared_ptr<VivadoHLSModuleNode> n =
    std::make_shared<VivadoHLSModuleNode>(code);
  return runtime::Module(n);
}

template<class CodeGen>
std::string BuildHLSC(Array<LoweredFunc> funcs) {
  CodeAnalysMerlinC ca;
  CodeGen cg;
  for (LoweredFunc f : funcs) {
    // 1st pass: Analyze AST and collect necessary information
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    // 2nd pass: Generate kernel code
    cg.AddFunction(f, map_arg_type);
  }
  std::string code = cg.Finish();

  LOG(WARNING) << "HLS C doesn't have runtime, return kernel code";
  return code;
}

TVM_REGISTER_API("codegen.build_ihls")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildHLSC<CodeGenIntelHLS>(args[0]);
  });
TVM_REGISTER_API("codegen.build_vhls")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildHLSC<CodeGenVivadoHLS>(args[0]);
  });
TVM_REGISTER_API("codegen.build_vhls_csim")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildVivadoHLSCSim(args[0]);
  });
}  // namespace codegen
}  // namespace tvm
