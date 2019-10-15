/*!
 *  Copyright (c) 2019 by Contributors
 * \file build_common.cc
 * \brief Build unified simulation module
 */
#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include "./build_common.h"

#include "merlinc/codeanalys_merlinc.h"
#include "hlsc/codegen_vhls.h"
#include "opencl/codegen_aocl.h"

namespace TVM {
namespace runtime {

class SimModuleNode final : public ModuleNode {
 public:
  SimModuleNode(LoweredFunc func, std::string test_file) 
    : func_(func), test_file_(test_file) {}

  const char* type_key() const {
    return "unified_sim";
  }

  // unified simulation function
  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv){
        if (args.size() != (int)func_->args.size())
          LOG(FATAL) << "The function should take in " << func_->args.size() 
                     << " inputs but get " << args.size();
        std::vector<size_t> arg_sizes;
        std::vector<TVMType> arg_types;
        std::vector<int> shmids;
        // CollectArgInfo(args, func_, arg_sizes, arg_types);
        // GenSharedMem(args, shmids, arg_sizes);
        // GenHostCode(args, shmids, arg_types, func_, test_file_);
        // TODO: find a better way to do the following
        LOG(CLEAN) << "Compiling the generated HLS C code ...";
        system("g++ main.cpp -o out");
        LOG(CLEAN) << "Running C simulation ...";
        system("./out");
        LOG(CLEAN) << "Finished C simulation";
        system("rm out main.cpp");
        FreeSharedMem(args, shmids, arg_sizes);
        // extract resource information
        if (const auto* f = Registry::Get("tvm_callback_syn_postproc")) {
          std::string code;
          code = (*f)("test").operator std::string();
          LOG(CLEAN) << "extract res info";
        }

      });
  }

 private:
  LoweredFunc func_;
  std::string test_file_;
};

Module CreateSimModule(
    LoweredFunc func,
    std::string code) {
  std::shared_ptr<SimModuleNode> n =
    std::make_shared<SimModuleNode>(func, code);
  return Module(n);
}
} // namespace runtime

namespace codegen {
// unified simulation function for diff platforms 
runtime::Module BuildSimModule(Array<LoweredFunc> funcs,
                               Array<Expr> attrs,
                               Array<Expr> values) {
  CodeAnalysMerlinC ca;
  CodeGenAOCL cg_host;
  CodeGenVivadoHLS cg_dev;
  for (LoweredFunc f : funcs) {
    // analyze AST and collect arg info
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    // generate kernel code
    cg_host.AddFunction(f, map_arg_type);
    cg_dev.AddFunction(f, map_arg_type);
  }

  std::string code = cg_host.Finish();
  return runtime::CreateSimModule(funcs[0], code);
}

TVM_REGISTER_API("codegen.build_sim")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildSimModule(args[0], args[1], args[2]);
  });

}  // namespace codegen
}  // namespace TVM
