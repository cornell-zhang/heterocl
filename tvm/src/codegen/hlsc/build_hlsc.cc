/*!
 *  Copyright (c) 2018 by Contributors
 * \file build_hlsc.cc
 * \brief Build HLS C modules from source.
 */
#include "../build_common.h"
#include "../code_analysis.h"
#include "./codegen_ihls.h"
#include "./codegen_vhls.h"
#include "./vhls_module.h"

namespace TVM {
namespace codegen {

#if HCL_VHLS_RUNTIME
runtime::Module BuildVivadoHLSCSim(Array<LoweredFunc> funcs) {
  CodeAnalysis ca;
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
  return runtime::CreateVivadoHLSModule(funcs[0], code);
}

TVM_REGISTER_API("codegen.build_vhls_csim")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      *rv = BuildVivadoHLSCSim(args[0]);
    });
#endif

// Only used for legacy string build interface
// or for returning code in debug mode
template <class CodeGen>
std::string BuildHLSC(Array<LoweredFunc> funcs, OutputMode mode,
                      TargetTool tool) {
  CodeAnalysis ca;
  CodeGen cg;
  for (LoweredFunc f : funcs) {
    // 1st pass: Analyze AST and collect necessary information
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();

    // Setup CodeGen modes
    if (tool == TargetTool::SDAccel || tool == TargetTool::Vitis) {
      map_arg_type["sdaccel"] = std::make_tuple("sdaccel", Handle());
    } else if (tool == TargetTool::SDSoC) {
      map_arg_type["sdsoc"] = std::make_tuple("sdsoc", Handle());
    }

    // 2nd pass: Generate kernel code
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

TVM_REGISTER_API("codegen.build_ihls")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      if (args.size() == 1) {
        *rv = BuildHLSC<CodeGenIntelHLS>(args[0], OutputMode::HostDevice,
                                         TargetTool::IntelHLS);
      } else {
        CHECK_EQ(args.size(), 3);
        *rv = BuildHLSC<CodeGenIntelHLS>(
            args[0], static_cast<OutputMode>(args[1].operator int()),
            TargetTool::IntelHLS);
      }
    });

TVM_REGISTER_API("codegen.build_vhls")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      // Legacy interface
      if (args.size() == 1) {
        *rv = BuildHLSC<CodeGenVivadoHLS>(args[0], OutputMode::HostDevice,
                                          TargetTool::VivadoHLS);
        // Returning host or dev code
      } else {
        CHECK_EQ(args.size(), 3);
        *rv = BuildHLSC<CodeGenVivadoHLS>(
            args[0], static_cast<OutputMode>(args[1].operator int()),
            static_cast<TargetTool>(args[2].operator int()));
      }
    });
}  // namespace codegen
}  // namespace TVM
