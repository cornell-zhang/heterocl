/*
    Yang.Bai
    yb269@cornell.edu
*/

# include <tvm/runtime/packed_func.h>
# include <memory>
# include <string>
# include <vector>
# include <unordered_map>
# include "../../runtime/meta_data.h"
# include <tvm/base.h>
# include "./codegen_sdaccel.h"
# include "./codegen_aocl.h"
# include "./codeanalys_openclc.h"
# include "../build_common.h"
// # include "./sdaccel/sdaccel_module.h"
// # include "./aocl/aocl_module.h"


namespace TVM {
namespace codegen {

// #if OPENCL_SDACCEL_RUNTIME
// runtime::Module BuildSDAccelSwEmu(Array<LoweredFunc> funcs) {
//     CodeAnalysOpenCLC ca;
//     CodeGenSDACCEL cg;
//     for (LoweredFunc f : funcs) {
//         ca.AddFunction(f);
//         str2tupleMap<std::string, Type> map_arg_type;
//         map_arg_type = ca.Finish();
//         cg.AddFunction(f, map_arg_type);
//     }
//     std::string code = cg.Finish();
    
//     return runtime::CreateSDAccelModule(funcs[0], code);
// }

// TVM_REGISTER_API("codegen.build_sdaccel_sw_emu")
// .set_body([](TVMArgs args, TVMRetValue* rv) {
//     *rv = BuildSDAccelSwEmu(args[0]);
//   });
// #endif



// #if OPENCL_AOCL_RUNTIME

// #endif

// std::string BuildOpenCL(Array<LoweredFunc> funcs) {
//     using TVM::runtime::Registry;
    
//     CodeAnalysOpenCLC ca;
//     CodeGenSDACCEL cg;
//     for (LoweredFunc f : funcs) {
//         ca.AddFunction(f);
//         str2tupleMap<std::string, Type> map_arg_type;
//         map_arg_type = ca.Finish();

//         cg.AddFunction(f, map_arg_type);
//     }
//     std::string code = cg.Finish();

//     if (const auto*f = Registry::Get("tvm_callback_opencl_postproc")) {
//         code = (*f)(code).operator std::string();
//     }
//     LOG(WARNING) << "OpenCL doesn't have runtime, return kernel code";
//     return code;
// }

// std::string BuildOpenCL(Array<LoweredFunc> funcs) {
//     using TVM::runtime::Registry;
//     bool output_ssa = false;
//     CodeGenSDACCEL cg;
//     cg.Init(output_ssa);

//     for (LoweredFunc f : funcs) {
//         cg.AddFunction(f);
//     }
//     std::string code = cg.Finish();

//     if (const auto*f = Registry::Get("tvm_callback_opencl_postproc")) {
//         code = (*f)(code).operator std::string();
//     }
//     LOG(WARNING) << "OpenCL doesn't have runtime, return kernel code";
//     return code;
// }





// codegen for AOCL 
// std::string BuildAOCL(Array<LoweredFunc> funcs) {
//     using TVM::runtime::Registry;
//     bool output_ssa = false;
//     CodeGenAOCL cg;
//     cg.Init(output_ssa);
//     for ( LoweredFunc f : funcs ) {
//         cg.AddFunction(f);
//     }
//     std::string code = cg.Finish();

//     if ( const auto * f = Registry::Get("tvm_callback_opencl_postproc")) {
//         code = (*f)(code).operator std::string();
//     }
//     LOG(WARNING) << "AOCL doesn't have runtime, return kernel code";
//     return code;
// }


// codegen for AOCL 
// std::string BuildAOCL(Array<LoweredFunc> funcs) {
//     using TVM::runtime::Registry;
//     bool output_ssa = false;
//     CodeGenAOCL cg;
//     cg.Init(output_ssa);
//     for ( LoweredFunc f : funcs ) {
//         cg.AddFunction(f);
//     }
//     std::string code = cg.Finish();

//     if ( const auto * f = Registry::Get("tvm_callback_opencl_postproc")) {
//         code = (*f)(code).operator std::string();
//     }
//     LOG(WARNING) << "AOCL doesn't have runtime, return kernel code";
//     return code;
// }


// codegen for SDACCEL
// std::string BuildSDACCEL(Array<LoweredFunc> funcs) {
//     using TVM::runtime::Registry;
//     bool output_ssa = false;
//     CodeGenSDACCEL cg;
//     cg.Init(output_ssa);
//     for (LoweredFunc f : funcs) {
//         cg.AddFunction(f);
//     }
//     std::string code = cg.Finish();

//     // if ( const auto * f = Registry::Get("tvm_callback_opencl_postproc")) {
//     //     code = (*f)(code).operator std::string();
//     // }
//     LOG(WARNING) << "SDAccel doesn't have runtime, return kernel code";
//     return code;
// }


// codegen for SDACCEL_WITH_ANALYSIS xxx
std::string BuildSDACCEL(Array<LoweredFunc> funcs) {
    using TVM::runtime::Registry;
    CodeAnalysOpenCLC ca;
    CodeGenSDACCEL cg;
    for (LoweredFunc f : funcs) {
        ca.AddFunction(f);
        str2tupleMap<std::string, Type> map_arg_type;
        map_arg_type = ca.Finish();

        cg.AddFunction(f, map_arg_type);

    }
    std::string code = cg.Finish();

    if (const auto* f = Registry::Get("tvm_callback_sdaccel_postproc")) {
        code = (*f)(code).operator std::string();
    }
    
    LOG(WARNING) << "SDaccel doesn't have runtime, return kernel code";
    return code;
}


//runtime::Module BuildSDACCELXCLBIN(Array<LoweredFunc> funcs)



// codegen for AOCL_WITH_ANALYSIS xxx
std::string BuildAOCL(Array<LoweredFunc> funcs) {
    using TVM::runtime::Registry;
    CodeAnalysOpenCLC ca;
    CodeGenAOCL cg;
    for (LoweredFunc f : funcs) {
        ca.AddFunction(f);
        str2tupleMap<std::string, Type> map_arg_type;
        map_arg_type = ca.Finish();

        cg.AddFunction(f, map_arg_type);

    }
    std::string code = cg.Finish();

    if (const auto* f = Registry::Get("tvm_callback_aocl_postproc")) {
        code = (*f)(code).operator std::string();
    }
    
    LOG(WARNING) << "AOCL doesn't have runtime, return kernel code";
    return code;
}



// codegen for OPENCL_WITH_ANALYSIS xxx
// std::string BuildOpenCL(Array<LoweredFunc> funcs) {
//     using TVM::runtime::Registry;
//     CodeAnalysOpenCLC ca;
//     CodeGenOpenCL cg;
//     for (LoweredFunc f : funcs) {
//         ca.AddFunction(f);
//         str2tupleMap<std::string, Type> map_arg_type;
//         map_arg_type = ca.Finish();

//         cg.AddFunction(f, map_arg_type);

//     }
//     std::string code = cg.Finish();

//     if (const auto* f = Registry::Get("tvm_callback_opencl_postproc")) {
//         code = (*f)(code).operator std::string();
//     }
    
//     LOG(WARNING) << "OpenCL doesn't have runtime, return kernel code";
//     return code;
// }





// codegen for OpenCL
// std::string BuildOpenCL(Array<LoweredFunc> funcs) {
//     using TVM::runtime::Registry;
//     bool output_ssa = false;
//     CodeGenOpenCL cg;
//     cg.Init(output_ssa);
//     for (LoweredFunc f : funcs) {
//         cg.AddFunction(f);
//     }
//     std::string code = cg.Finish();

//     LOG(WARNING) << "OpenCL doesn't have runtime, return kernel code";
//     return code;
// }



// codegen for SDACCEL
// template <class CodeGen>
// std::string BuildOpenCL(Array<LoweredFunc> funcs) {
//     CodeAnalysOpenCL ca;
//     CodeGen cg;
//     for (LoweredFunc f : funcs) {
//         ca.AddFunction(f);
//         str2tupleMap<std::string, Type> map_arg_type;
//         map_arg_type = ca.Finish();
//         cg.AddFunction(f, map_arg_type);
//     }
//     std::string code = cg.Finish();

//     // if ( const auto * f = Registry::Get("tvm_callback_opencl_postproc")) {
//     //     code = (*f)(code).operator std::string();
//     // }
//     LOG(WARNING) << "SDAccel doesn't have runtime, return kernel code";
//     // std::unordered_map<std::string, runtime::FunctionInfo> 
//     // std::unordered_map<std::string, runtime::FunctionInfo> temp = ExtractFuncInfo(funcs);
//     return code;
// }




TVM_REGISTER_API("codegen.build_sdaccel")
.set_body([]( TVMArgs args, TVMRetValue * rv ) {
    * rv = BuildSDACCEL(args[0]);
    });

TVM_REGISTER_API("codegen.build_aocl")
.set_body([]( TVMArgs args, TVMRetValue * rv ) {
    * rv = BuildAOCL(args[0]);
    });
// TVM_REGISTER_API("codegen.build_opencl")
// .set_body([]( TVMArgs args, TVMRetValue * rv ) {
//     * rv = BuildOpenCL(args[0]);
//     });


// template mode for opencl
// template<class CodeGen>
// std::string BuildOpenCL(Array<LoweredFunc> funcs) {
//   CodeAnalysOpenCLC ca;
//   CodeGen cg;
//   for (LoweredFunc f : funcs) {
//     // 1st pass: Analyze AST and collect necessary information
//     ca.AddFunction(f);
//     str2tupleMap<std::string, Type> map_arg_type;
//     map_arg_type = ca.Finish();
//     // 2nd pass: Generate kernel code
//     cg.AddFunction(f, map_arg_type);
//   }
//   std::string code = cg.Finish();

//   LOG(WARNING) << "OpenCL C doesn't have runtime, return kernel code";
//   return code;
// }

// TVM_REGISTER_API("codegen.build_sdaccel")
// .set_body([](TVMArgs args, TVMRetValue* rv) {
//     *rv = BuildOpenCL<CodeGenSDACCEL>(args[0]);
//   });
// TVM_REGISTER_API("codegen.build_aocl")
// .set_body([](TVMArgs args, TVMRetValue* rv) {
//     *rv = BuildOpenCL<CodeGenAOCL>(args[0]);
//   });

// For runtime 
// TVM_REGISTER_API("codegen.build_sdaccel_xclbin")
// .set_body([]( TVMArgs args, TVMRetValue * rv ) {
//     * rv = BuildSDACCEL(args[0]);
//     });


// TVM_REGISTER_API("codegen.build_opencl")
// .set_body([]( TVMArgs args, TVMRetValue * rv ) {
//     * rv = BuildOpenCL(args[0]);
//     });

// TVM_REGISTER_API("codegen.build_aocl")
// .set_body([]( TVMArgs args, TVMRetValue * rv ) {
//     * rv = BuildOpenCL(args[0]);
//     });
} // namespace codegen
} // namespace TVM
