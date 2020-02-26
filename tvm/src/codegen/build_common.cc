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
#include "hlsc/codegen_ihls.h"
#include "opencl/codegen_aocl.h"
#include "opencl/codegen_aocl_host.h"
#include "opencl/codegen_sdaccel_host.h"
#include "opencl/codegen_sdaccel.h"
#include "ppac/codegen_rv64_ppac.h"

namespace TVM {
namespace runtime {

class SimModuleNode final : public ModuleNode {
 public:
  SimModuleNode(LoweredFunc func, 
                std::string host_code,
                std::vector<std::string> arg_names,
                std::string dev_code, std::string platform, 
                std::unordered_map<std::string, std::string> options)
    : func_(func), 
      host_(host_code), 
      arg_names_(arg_names),
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
        bool init = true;
        if (shmids.size() > 0) {
          init = false; // requires mem update
          CHECK(shmids.size() == (unsigned)args.size()) 
            << "invalid inputs";

        } else { // perform init
          std::vector<TVMType> arg_types;
          CollectArgInfo(args, func_, arg_sizes, arg_types);
          GenSharedMem(args, shmids, arg_sizes);

          LOG(CLEAN) << "Generating harness files ...";
          system("rm -rf __tmp__; mkdir __tmp__");

          if (platform_ == "sdaccel") {
            GenHostCode(args, shmids, arg_types, func_, 
                        platform_, host_, arg_names_);
            GenKernelCode(dev_, platform_, options_["backend"]);

          } else if (platform_ == "aocl") {
            GenHostCode(args, shmids, arg_types, func_, 
                        platform_, host_, arg_names_);
            GenKernelCode(dev_, platform_, options_["backend"]);

          } else if (platform_ == "rocket") {
            // generate host and run proxy kernel test 
            GenHostCode(args, shmids, arg_types, func_, 
                        platform_, host_, arg_names_);
            std::string compile = "cd __tmp__;";
            compile += std::string("autoconf; mkdir build; cd build;") +
                       std::string("../configure --with-riscvtools=") + 
                       options_["RISCV"] + std::string(";make -j8");
            system(compile.c_str());

          } else if (platform_ == "vivado_hls" || 
                     platform_ == "vivado" || platform_ == "sdsoc") {
            GenHostCode(args, shmids, arg_types, func_, 
                        platform_, host_, arg_names_);
            GenKernelCode(dev_, platform_, options_["backend"]); 

          } else { // unsupported platform
            LOG(FATAL) << "unrecognized platform " << platform_;  
          } 

          // copy files and compile tp binary  
          LOG(CLEAN) << "Compiling the program ...";
          if (const auto* f = Registry::Get("copy_and_compile")) { 
            CHECK(options_.count("mode")) << "mode mot set";
            auto mode = options_["mode"];
            auto backend = options_["backend"];
            (*f)(platform_, mode, backend).operator std::string();
          }
        }

        // update shared memory (TVMArg is temporary value. and we
        // cannot get address from it, which is a illegal object)  
        if (!init) { // call compiled function 
          for (int i = 0; i < args.size(); i++) {

            if (args[i].type_code() == kArrayHandle) {
              TVMArray* arr = args[i];
              int shmid = shmids[i];
              void* mem = shmat(shmid, nullptr, 0);
              memcpy(mem, arr->data, arg_sizes[i]);

            } else { // update var arg
              if (args[i].type_code() == kDLInt ||
                  args[i].type_code() == kDLUInt) {
                int data = int64_t(args[i]);
                int shmid = shmids[i];
                void* mem = shmat(shmid, nullptr, 0);
                memcpy(mem, &data, arg_sizes[i]);
              }
            }
          }
        }

        // perform execution and information extraction 
        if (const auto* f = Registry::Get("tvm_callback_exec_evaluate")) {
          std::string code;
          std::string mode = options_["mode"];
          code = (*f)(platform_, mode).operator std::string();
          LOG(CLEAN) << "Execution complete \n";
        }

        // copy data back to TVM Args
        for (int i = 0; i < args.size(); i++) {
          if (args[i].type_code() == kArrayHandle) {
            TVMArray* arr = args[i];
            int shmid = shmids[i];
            void* mem = shmat(shmid, nullptr, 0);
            memcpy(arr->data, mem, arg_sizes[i]);
          }
        }
      });
  }

 private:
  LoweredFunc func_;
  std::string host_;
  std::vector<std::string> arg_names_;
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
    std::vector<std::string> arg_names,
    std::string platform, 
    std::unordered_map<std::string, std::string> options) {
  std::shared_ptr<SimModuleNode> n =
    std::make_shared<SimModuleNode>(func, host_code, 
                                    arg_names, dev_code,
                                    platform, options);
  return Module(n);
}
} // namespace runtime

namespace codegen {

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
  std::string backend  = values[2].as<StringImm>()->value;

  for (LoweredFunc f : funcs) {
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    if (platform == "sdsoc") 
      map_arg_type["sdsoc"] = std::make_tuple("sdsoc", Handle());
    cg_host.AddFunction(f, map_arg_type);
    cg_dev.AddFunction(f, map_arg_type);
  }
  // tool option mapping and platform 
  std::unordered_map<std::string, std::string> options;
  options["backend"] = backend;

  for (size_t k = 1; k < attrs.size(); k++) {
    auto key = attrs[k].as<StringImm>()->value;
    auto val = values[k].as<StringImm>()->value;
    options[key] = val;
  }
  return runtime::CreateSimModule(funcs[0], 
                                  cg_host.GetHost(),
                                  cg_dev.GetDevice(),
                                  cg_host.arg_names, 
                                  platform, options);
}

TVM_REGISTER_API("codegen.build_sim")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    // dispatch to corresponding codegen
    auto& sptr = args[2].node_sptr();
    CHECK(sptr->is_type<ArrayNode>());
    auto* n = static_cast<const ArrayNode*>(sptr.get());
    auto data = n->data[static_cast<size_t>(0)];
    auto lang = Expr(n->data[static_cast<size_t>(2)]).as<StringImm>()->value;

    // create module node for simulation 
    std::string type = Expr(data).as<StringImm>()->value;
    if (type == "rocket") {
      *rv = BuildSimModule<CodeGenRV64PPAC, CodeGenRV64PPAC>
                (args[0], args[1], args[2]);

    } else if (type == "sdaccel") {
      if (lang == "sdaccel") {
        *rv = BuildSimModule<CodeGenSDAccelHost, CodeGenSDACCEL>
                  (args[0], args[1], args[2]);
      } else if (lang == "vhls") {
        *rv = BuildSimModule<CodeGenSDAccelHost, CodeGenVivadoHLS>
                  (args[0], args[1], args[2]);
      } else {
        LOG(FATAL) << "sdaccel does not support "
                   << lang << " backend";
      }

    } else if (type == "vivado_hls" || 
               type == "vivado" || type == "sdsoc") {
      *rv = BuildSimModule<CodeGenVivadoHLS, CodeGenVivadoHLS>
                (args[0], args[1], args[2]);

    } else if (type == "aocl") {
      if (lang == "aocl") {
        *rv = BuildSimModule<CodeGenAOCLHost, CodeGenAOCL>
                  (args[0], args[1], args[2]);
      } else if (lang == "ihls") {
        *rv = BuildSimModule<CodeGenAOCLHost, CodeGenIntelHLS>
                  (args[0], args[1], args[2]);
      } else {
        LOG(FATAL) << "aocl does not support "
                   << lang << " backend";
      }

    } else {
      LOG(FATAL) << "unrecognized platform " << type;
    }
  });

}  // namespace codegen
}  // namespace TVM
