#include <fstream>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <tvm/base.h>
#include <tvm/runtime/config.h>
#include "../build_common.h"
#include "./codegen_catapultc.h"
#include "./codegen_catapultc_tb.h"

namespace TVM {
namespace codegen {

template<class CodeGen>
std::string BuildCatapultC(Array<LoweredFunc> funcs, OutputMode mode) {
  using TVM::runtime::Registry;
  // bool output_ssa = false;

  CodeAnalysMerlinC ca;
  CodeGen cg;
  // cg.Init(output_ssa);
  for (LoweredFunc f : funcs) {
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    cg.AddFunction(f, map_arg_type);
  }
  // std::string code = cg.Finish();
  std::string code;
  switch (mode) {
    case OutputMode::HostDevice : {code = cg.Finish(); break;}
    case OutputMode::HostOnly   : {code = cg.GetHost(); break;}
    case OutputMode::DeviceOnly : {code = cg.GetDevice(); break;}
    default:
      LOG(FATAL) << "Unsupported output mode";
  }

  // if (const auto* f = Registry::Get("tvm_callback_soda_postproc")) {
  //   code = (*f)(code).operator std::string();
  // }

  // LOG(WARNING) << "CatapultC doesn't have runtime, return kernel code";
  return code;
}

TVM_REGISTER_API("codegen.build_catapultc")
    // .set_body([](TVMArgs args, TVMRetValue* rv) {
    //   *rv = BuildCatapultC(args[0]);
    // });
.set_body([](TVMArgs args, TVMRetValue* rv) {
    if (args.size() == 1) {
      *rv = BuildCatapultC<CodeGenCatapultC>(args[0], OutputMode::HostDevice);
    } else {
      auto mode = static_cast<OutputMode>(args[1].operator int());
      if (mode == OutputMode::HostOnly) {
        LOG(INFO) << "mode == host_only\n";
        *rv = BuildCatapultC<CodeGenCatapultCTB>(args[0], mode); 
      } else if (mode == OutputMode::DeviceOnly) {
        LOG(INFO) << "mode == device_only\n";
        *rv = BuildCatapultC<CodeGenCatapultC>(args[0], mode);
      }
      else {
        LOG(INFO) << "mode == host_device\n";
      }
    } 
  });

}  // namespace codegen
}  // namespace TVM