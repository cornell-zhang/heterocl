/*!
 *  Copyright (c) 2019 by Contributors
 */
#include "../build_common.h"
#include "../code_analysis.h"
#include "./codegen_aocl.h"
#include "./codegen_aocl_host.h"
#include "./codegen_xocl.h"
#include "./codegen_xocl_host.h"

namespace TVM {
namespace codegen {

template <class CodeGen>
std::string BuildOpenCL(Array<LoweredFunc> funcs, OutputMode mode) {
  using TVM::runtime::Registry;
  CodeAnalysis ca;
  CodeGen cg;
  for (LoweredFunc f : funcs) {
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    cg.AddFunction(f, map_arg_type);
  }

  std::string code;
  switch (mode) {
    case OutputMode::HostDevice: {
      code = cg.Finish();
      break;
    }
    case OutputMode::HostOnly: {
      code = cg.GetHost();
      break;
    }
    case OutputMode::DeviceOnly: {
      code = cg.GetDevice();
      break;
    }
    default:
      LOG(FATAL) << "Unsupported output mode";
  }
  return code;
}

TVM_REGISTER_API("codegen.build_xocl")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      if (args.size() == 1) {
        *rv = BuildOpenCL<CodeGenXOCL>(args[0], OutputMode::HostDevice);
      } else {
        CHECK_EQ(args.size(), 3);
        *rv = BuildOpenCL<CodeGenXOCLHost>(
            args[0], static_cast<OutputMode>(args[1].operator int()));
      }
    });

TVM_REGISTER_API("codegen.build_aocl")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      if (args.size() == 1) {
        *rv = BuildOpenCL<CodeGenAOCL>(args[0], OutputMode::HostDevice);
      } else {
        auto mode = static_cast<OutputMode>(args[1].operator int());
        if (mode == OutputMode::HostOnly) {
          *rv = BuildOpenCL<CodeGenAOCLHost>(args[0], mode);
        } else if (mode == OutputMode::DeviceOnly) {
          *rv = BuildOpenCL<CodeGenAOCL>(args[0], mode);
        }
      }
    });
}  // namespace codegen
}  // namespace TVM
