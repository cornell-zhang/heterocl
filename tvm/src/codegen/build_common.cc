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
#include "opencl/codegen_xocl_host.h"
#include "opencl/codegen_xocl.h"
#include "ppac/codegen_rv64_ppac.h"

// rapidjson headers
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

namespace TVM {
namespace runtime {

class SimModuleNode final : public ModuleNode {
 public:
  SimModuleNode(LoweredFunc func, 
                std::string host_code,
                std::vector<std::string> arg_names,
                std::unordered_map<std::string, bool> arg_access_status,
                std::string dev_code, std::string cfg_code, std::string platform, 
                std::unordered_map<std::string, std::string> options)
    : func_(func), 
      host_(host_code), 
      dev_(dev_code), 
      cfg_(cfg_code), 
      arg_names_(arg_names),
      arg_access_status_(arg_access_status),
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

        bool empty = false; // whether kernel is empty
        if (dev_.find_first_not_of(" \t\n") == std::string::npos) empty = true;

        std::vector<TVMType> arg_types;
        CollectArgInfo(args, func_, arg_sizes, arg_types);

        // Use a unified status control function 
        const auto* f = Registry::Get("hcl_status_control");
        CHECK(f) << "Cannot find hcl_status_control function.";
        std::hash<std::string> hasher;
        size_t hash = hasher(dev_) & 0xFFFFFFFF;
        auto code = (*f)(empty, hash).operator std::string();

        if (code == "codegen") {
          // GenSharedMem(args, shmids, arg_sizes);
          GenHostCode(args, shmids, arg_types, func_, 
                  platform_, host_, arg_names_, arg_access_status_,
                  empty, options_["project"]);
          // Generate JSON inputs
          GenJSONInputs(args, arg_names_, arg_sizes, arg_types, options_["project"]);
          LOG(CLEAN) << "Generating kernel code (harness files copied) ...";
          GenKernelCode(dev_, arg_names_, platform_, options_["backend"], options_["project"]); 
          // Generate configuration
          GenConfigCode(cfg_, platform_, options_["project"]);
          return;       

        } else if (code == "execute") {
          LOG(CLEAN) << "Execution finished. Loading results back...";
          std::string file_name = options_["project"] + "/inputs.json";

          FILE *f = fopen(file_name.c_str(), "r");
          CHECK(f) << "Output JSON file does not exist: " << file_name;
          char readBuffer[65536];
          rapidjson::FileReadStream is(f, readBuffer, sizeof(readBuffer));

          rapidjson::Document document;
          document.ParseStream(is);
          fclose(f);
        
          for (int i = 0; i < args.size(); i++) { 
            if (args[i].type_code() == kArrayHandle) {
              TVMArray* arr = args[i];
              std::string arg_name = arg_names_[i];
              assert(document.HasMember(rapidjson::GenericStringRef<char>(arg_name.c_str())));
              const rapidjson::Value& data = document[arg_name.c_str()];
              assert(data.IsArray());

              int mul = 1;
              for (int j = arr->ndim-1; j >= 0; j--) {
                mul *= arr->shape[j];
              }
              void* mem = (void *)malloc(arg_sizes[i]);
              CHECK(mem) << "Malloc returns null memory pointer...";
              if (arg_types[i].code == kDLFloat || arr->dtype.fracs > 0) {

                for (int k = 0; k < mul; k++) {
                   *((float*)(mem) + k) = (float)data[k].GetFloat();
                }
                memcpy(arr->data, mem, arg_sizes[i]);
              } else {
                for (int k = 0; k < mul; k++) {
                  *((int*)(mem) + k) = (int)data[k].GetInt();
                }
                memcpy(arr->data, mem, arg_sizes[i]);
              }
              free(mem);
            }
          }
        }
      });
  }

 private:
  LoweredFunc func_;
  std::string host_, dev_, cfg_;
  std::vector<std::string> arg_names_;
  std::unordered_map<std::string, bool> arg_access_status_;
  std::string platform_;
  std::unordered_map<std::string, std::string> options_;
  std::vector<int> shmids;
  std::vector<size_t> arg_sizes;
};

Module CreateSimModule(
    LoweredFunc func, std::string host_code,
    std::string dev_code, std::string cfg_code, 
    std::vector<std::string> arg_names,
    std::unordered_map<std::string, bool> arg_access_status,
    std::string platform, std::unordered_map<std::string, std::string> options) {

  std::shared_ptr<SimModuleNode> n =
    std::make_shared<SimModuleNode>(
            func, host_code, arg_names, arg_access_status, dev_code,
            cfg_code, platform, options);
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

    // set up modes for codegen
    if (platform == "sdsoc") { 
      map_arg_type["sdsoc"] = std::make_tuple("sdsoc", Handle());
    } else if (platform == "sdaccel" || platform == "vitis") {
      map_arg_type["sdaccel"] = std::make_tuple("sdaccel", Handle());
    }

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
  return runtime::CreateSimModule(
          funcs[0], cg_host.GetHost(), cg_dev.GetDevice(),
          cg_dev.GetConfig(), cg_host.arg_names,
          cg_host.arg_access_status, platform, options);
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

    } else if (type == "sdaccel" || type == "vitis") {
      if (lang == "xocl") {
        *rv = BuildSimModule<CodeGenXOCLHost, CodeGenXOCL>
                  (args[0], args[1], args[2]);
      } else if (lang == "vhls") {
        *rv = BuildSimModule<CodeGenXOCLHost, CodeGenVivadoHLS>
                  (args[0], args[1], args[2]);
      } else {
        LOG(FATAL) << "sdaccel does not support "
                   << lang << " backend";
      }

    } else if (type == "vivado_hls" || type == "sdsoc") {
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
