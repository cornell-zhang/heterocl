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
#include "opencl/codegen_sdaccel.h"
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
      options_(options) {}

  ~SimModuleNode() {
    for (size_t i = 0; i < shmids.size(); i++) {
      int shmid = shmids[i];
      void* mem = shmat(shmid, nullptr, 0);
      shmdt(mem);
      shmctl(shmid, IPC_RMID, nullptr);
    }
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
        // check whether init needed
        if (shmids.size() > 0) {
          CHECK(shmids.size() == (unsigned)args.size()) 
            << "invalid inputs";

        } else { // perform init
          std::vector<TVMType> arg_types;
          int added_args_num = 0;
          if (options_["added_args_num"].length() > 0)
            added_args_num = std::stoi(options_["added_args_num"]);
          CollectArgInfo(args, func_, arg_sizes, arg_types);
          GenSharedMem(args, shmids, arg_sizes);

          LOG(CLEAN) << "Generating harness files ...";
          system("rm -rf __tmp__; mkdir __tmp__");
          if (const auto* f = Registry::Get("get_util_path")) 
            (*f)(platform_).operator std::string();
          LOG(CLEAN) << "Running SW simulation on " + platform_;

          if (platform_ == "sdaccel") {
            GenWrapperCode(args, shmids, arg_types, arg_info_, func_);
            GenHostCode(args, shmids, arg_types, func_, 
                        platform_, host_, arg_info_, added_args_num);
            GenKernelCode(dev_, platform_, arg_info_);

            LOG(CLEAN) << "Running SW simulation ...";
            system("cd __tmp__; source ./run_sw.sh");

          } else if (platform_ == "rocket") {
            // generate host and run proxy kernel test 
            GenHostCode(args, shmids, arg_types, func_, 
                        platform_, host_, arg_info_, added_args_num);
            std::string compile = "cd __tmp__;";
            compile += std::string("autoconf; mkdir build; cd build;") +
                       std::string("../configure --with-riscvtools=") + 
                       options_["RISCV"] + std::string(";make -j8");
            system(compile.c_str());

          } else if (platform_ == "vivado_hls" || 
                     platform_ == "vivado" || platform_ == "sdsoc") {
            GenHostCode(args, shmids, arg_types, func_, 
                        platform_, host_, arg_info_, added_args_num);
            GenKernelCode(dev_, platform_, arg_info_); // kernel + header

          } else { // unsupported platform
            LOG(FATAL) << "unrecognized platform " << platform_;  
          } 
        }

        // execute program & extract resource information
        if (const auto* f = Registry::Get("tvm_callback_syn_postproc")) {
          std::string code;
          code = (*f)(platform_).operator std::string();
          LOG(CLEAN) << "Execution complete \n";
        }

        // copy data back to TVM Args
        for (int i = 0; i < args.size(); i++) {
          TVMArray* arr = args[i];
          int shmid = shmids[i];
          void* mem = shmat(shmid, nullptr, 0);
          memcpy(arr->data, mem, arg_sizes[i]);
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
  std::vector<int> shmids;
  std::vector<size_t> arg_sizes;
};

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
using runtime::argItem;
using argInfo = std::vector<argItem>;

// unified simulation function for diff platforms 
template<class CodeGenHost, class CodeGenXcel>
runtime::Module BuildSimModule(Array<LoweredFunc> funcs,
                               Array<Expr> attrs,
                               Array<Expr> values) {
  CodeAnalysMerlinC ca;
  CodeGenHost cg_host;
  CodeGenXcel cg_dev;
  
  // generate code based on platform info
  std::string platform = values[0].as<StringImm>()->value;

  for (LoweredFunc f : funcs) {
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    if (platform == "sdsoc") 
      map_arg_type["sdsoc"] = std::make_tuple("sdsoc", Handle());
    cg_host.AddFunction(f, map_arg_type);
    cg_dev.AddFunction(f, map_arg_type);
  }
  // vector {vars} with extern op arg 
  auto& arg_vars = cg_dev.arg_vars;
  // map {var : is_streamed(bool) }
  auto& stream_table = cg_dev.stream_table;
  // map {var : (vid, Type, shape)}
  auto& arg_top_vars = cg_dev.arg_top_vars;
  // tool option mapping and platform 
  std::unordered_map<std::string, std::string> options;
  // num of added arg (host consumed & xcel defined)
  std::vector<int> add_args_num;
  for (auto& kv : cg_dev.host_undefined) {
    add_args_num.push_back(kv.second.size());
    options["added_args_num"] = 
        std::to_string(kv.second.size());
  }

  argInfo arg_info;
  for (size_t i = 0 ; i < arg_vars.size(); i++) {
    auto v = arg_vars[i];
    auto nameType = arg_top_vars[v];
    bool is_stream;
    if (stream_table[v])
      is_stream = true;
    else is_stream = false;
    arg_info.push_back({/*var name*/nameType.name,
                        /*whether is streamed*/is_stream, 
                        /*data type*/nameType.type, 
                        /*shape*/nameType.shape});
  }
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
    // dispatch to corresponding codegen
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
      // *rv = BuildSimModule<CodeGenAOCL, CodeGenVivadoHLS>
      *rv = BuildSimModule<CodeGenSDACCEL, CodeGenVivadoHLS>
                (args[0], args[1], args[2]);
    } else if (type == "vivado_hls" || 
               type == "vivado" || type == "sdsoc") {
      *rv = BuildSimModule<CodeGenVivadoHLS, CodeGenVivadoHLS>
                (args[0], args[1], args[2]);
    } else {
      LOG(FATAL) << "unrecognized platform " << type;
    }
  });

}  // namespace codegen
}  // namespace TVM
