/*!
 *  Copyright (c) 2019 by Contributors
 * \file build_common.cc
 * \brief Build unified simulation module
 */
#include <tvm/base.h>
#include <tvm/ir_visitor.h>
#include <tvm/runtime/config.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/build_module.h>
#include "./build_common.h"
#include "./build_util.h"

#include <fstream>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <iostream>

#include "merlinc/codeanalys_merlinc.h"
#include "hlsc/codegen_vhls.h"
#include "opencl/codegen_aocl.h"
#include "ppac/codegen_rv64_ppac.h"

namespace TVM {
namespace runtime {

class SimModuleNode final : public ModuleNode {
 public:
  SimModuleNode(LoweredFunc func, 
                std::string host_code,
                argInfo arg_info,
                std::string dev_code, std::string platform, 
                std::unordered_map<std::string, std::string> options)
    : func_(func), 
      host_(host_code), 
      arg_info_(arg_info),
      dev_(dev_code), 
      platform_(platform), 
      options_(options) { 
  }

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
        std::vector<int> shmids;
        std::vector<size_t> arg_sizes;
        std::vector<TVMType> arg_types;

        CollectArgInfo(args, func_, arg_sizes, arg_types);
        GenSharedMem(args, shmids, arg_sizes);

        LOG(CLEAN) << "Generating harness files ...";
        system("rm -rf __tmp__; mkdir __tmp__");
        std::string path; 
        if (const auto* f = Registry::Get("get_util_path")) 
          path = (*f)(platform_).operator std::string();
        system(("cp -r " + path + "/* __tmp__/").c_str());
        LOG(CLEAN) << "Running SW simulation on " + platform_;

        if (platform_ == "sdaccel") {
          GenWrapperCode(args, shmids, arg_types, arg_info_, func_);
          GenHostCode(args, shmids, arg_types, func_, 
                      platform_, host_, arg_info_);
          GenKernelCode(dev_);

          LOG(CLEAN) << "Running SW simulation ...";
          system("cd __tmp__; source ./run_sw.sh");

        } else if (platform_ == "rocket") {
          // generate host and run proxy kernel test 
          GenHostCode(args, shmids, arg_types, func_, 
                      platform_, host_, arg_info_);
          std::string compile = "cd __tmp__;";
          compile += std::string("autoconf; mkdir build; cd build;") +
                     std::string("../configure --with-riscvtools=") + 
                     options_["RISCV"] + std::string(";make -j8");
          system(compile.c_str());

        } else if (platform_ == "vivado_hls") {
          GenHostCode(args, shmids, arg_types, func_, 
                      platform_, host_, arg_info_);
          GenKernelCode(dev_);
          system("cd __tmp__; make csim");
        } else {
          LOG(FATAL) << "unrecognized platform " << platform_;  
        } 

        // clean & extract resource information
        FreeSharedMem(args, shmids, arg_sizes);
        if (const auto* f = Registry::Get("tvm_callback_syn_postproc")) {
          std::string code;
          code = (*f)("test").operator std::string();
          LOG(CLEAN) << "extract res info";
        }
      });
  }

 private:
  LoweredFunc func_;
  std::string host_;
  argInfo arg_info_;
  std::string dev_;
  std::string platform_;
  std::unordered_map<std::string, std::string> options_;
};

using var2nameType = std::unordered_map<const Variable*, 
    std::tuple<std::string, Type, std::vector<int>>>; 

Module CreateSimModule(
    LoweredFunc func,
    std::string host_code,
    std::string dev_code,
    argInfo arg_types,
    std::string platform, 
    std::unordered_map<std::string, std::string> options) {
  std::shared_ptr<SimModuleNode> n =
    std::make_shared<SimModuleNode>(func, host_code, 
                                    arg_types, dev_code,
                                    platform, options);
  return Module(n);
}
} // namespace runtime

namespace codegen {
using var2nameType = std::unordered_map<const Variable*, 
    std::tuple<std::string, Type, std::vector<int>>>; 

using argInfo = 
    std::vector<std::tuple<std::string, bool, Type, std::vector<int>>>;

// unified simulation function for diff platforms 
template<class CGHost, class CGXcel>
runtime::Module BuildSimModule(Array<LoweredFunc> funcs,
                               Array<Expr> attrs,
                               Array<Expr> values) {
  CodeAnalysMerlinC ca;
  CGHost cg_host;
  CGXcel cg_dev;
  
  for (LoweredFunc f : funcs) {
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    cg_host.AddFunction(f, map_arg_type);
    cg_dev.AddFunction(f, map_arg_type);
  }
  // vector {vars} 
  auto& arg_vars = cg_dev.arg_vars;
  // map {var : is_streamed(bool) }
  auto& stream_table = cg_dev.stream_table;
  // map {var : (vid, Type, shape)}
  auto& arg_top_vars = cg_dev.arg_top_vars;

  argInfo arg_info;
  for (size_t i = 0 ; i < arg_vars.size(); i++) {
    auto v = arg_vars[i];
    auto nameType = arg_top_vars[v];
    bool is_stream;
    if (stream_table[v])
      is_stream = true;
    else is_stream = false;
    auto item = std::make_tuple(
        /*var name*/std::get<0>(nameType),
        /*whether is streamed*/is_stream, 
        /*data type*/std::get<1>(nameType), 
        /*shape*/std::get<2>(nameType));
    arg_info.push_back(item);
  }
  // tool option mapping and platform 
  std::string platform = values[0].as<StringImm>()->value;
  std::unordered_map<std::string, std::string> options;
  for (size_t k = 1; k < attrs.size(); k++) {
    auto key = attrs[k].as<StringImm>()->value;
    auto val = values[k].as<StringImm>()->value;
    options[key] = val;
  }
  return runtime::CreateSimModule(funcs[0], 
                                  cg_host.GetHost(),
                                  cg_dev.GetDevice(),
                                  arg_info, platform, options);
}

TVM_REGISTER_API("codegen.build_sim")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    // dispatch to corr codegen
    auto& sptr = args[2].node_sptr();
    CHECK(sptr->is_type<ArrayNode>());
    auto* n = static_cast<const ArrayNode*>(sptr.get());
    auto data = n->data[static_cast<size_t>(0)];

    // create module node for simulation 
    std::string type = Expr(data).as<StringImm>()->value;
    if (type == "rocket") {
      *rv = BuildSimModule<CodeGenRV64PPAC, CodeGenRV64PPAC>
                (args[0], args[1], args[2]);
    } else if (type == "sdaccel") {
      *rv = BuildSimModule<CodeGenAOCL, CodeGenVivadoHLS>
                (args[0], args[1], args[2]);
    } else if (type == "vivado_hls") {
      *rv = BuildSimModule<CodeGenVivadoHLS, CodeGenVivadoHLS>
                (args[0], args[1], args[2]);
    } else {
    }
  });

}  // namespace codegen
}  // namespace TVM
