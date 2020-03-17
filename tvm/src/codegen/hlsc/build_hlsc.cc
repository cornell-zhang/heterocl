/*!
 *  Copyright (c) 2018 by Contributors
 * \file build_vhls.cc
 * \brief Build HLS C modules from source.
 */
#include "./codegen_ihls.h"
#include "./codegen_vhls.h"
#include "./vhls_module.h"
#include "../build_common.h"

namespace TVM {
namespace codegen {

#if HCL_VHLS_RUNTIME
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
  return runtime::CreateVivadoHLSModule(funcs[0], code);
}

TVM_REGISTER_API("codegen.build_vhls_csim")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildVivadoHLSCSim(args[0]);
  });
#endif

template<class CodeGen>
std::string BuildHLSC(
    Array<LoweredFunc> funcs, int output_mode) {
  CodeAnalysMerlinC ca;
  CodeGen cg;
  for (LoweredFunc f : funcs) {
    // 1st pass: Analyze AST and collect necessary information
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    if (output_mode == 2) {
      map_arg_type["sdaccel"] = 
          std::make_tuple("sdaccel", Handle());
    }

    // 2nd pass: Generate kernel code
    cg.AddFunction(f, map_arg_type);
  }

  std::string code;
  switch (output_mode) {
    case 0: {code = cg.Finish(); break;}
    case 1: {code = cg.GetHost(); break;}
    case 2: {code = cg.GetDevice(); break;}
    default:
      LOG(FATAL) << "Unsupported output mode";
  }
  return code;
}

TVM_REGISTER_API("codegen.build_ihls")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    if (args.size() == 1) {
      *rv = BuildHLSC<CodeGenIntelHLS>(args[0], 0);
    } else {
      CHECK(args.size() == 2);
      *rv = BuildHLSC<CodeGenIntelHLS>(args[0], 
          static_cast<int>(args[1]));
    } 
  });

TVM_REGISTER_API("codegen.build_vhls")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    if (args.size() == 1) {
      *rv = BuildHLSC<CodeGenVivadoHLS>(args[0], 0);
    } else {
      CHECK(args.size() == 2);
      *rv = BuildHLSC<CodeGenVivadoHLS>(args[0], 
          static_cast<int>(args[1]));
    } 
  });
}  // namespace codegen
}  // namespace TVM
