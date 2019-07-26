/*
    Yang.Bai
    yb269@cornell.edu
*/

#include <tvm/runtime/packed_func.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "../../runtime/meta_data.h"
# include <tvm/base.h>
# include "./codegen_sdaccel.h"
# include "./codegen_aocl.h"
# include "../build_common.h"


namespace TVM {
namespace codegen {

// #if OPENCL_SDACCEL_RUNTIME

// #endif

// #if OPENCL_AOCL_RUNTIME

// #endif


// codegen for AOCL 
std::string BuildAOCL(Array<LoweredFunc> funcs) {
    using TVM::runtime::Registry;
    bool output_ssa = false;
    CodeGenAOCL cg;
    cg.Init(output_ssa);
    for ( LoweredFunc f : funcs ) {
        cg.AddFunction(f);
    }
    std::string code = cg.Finish();

    if ( const auto * f = Registry::Get("tvm_callback_opencl_postproc")) {
        code = (*f)(code).operator std::string();
    }
    LOG(WARNING) << "AOCL doesn't have runtime, return kernel code";
    return code;
}


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

// codegen for SDACCEL
std::string BuildSDACCEL(Array<LoweredFunc> funcs) {
    using TVM::runtime::Registry;
    bool output_ssa = false;
    CodeGenSDACCEL cg;
    cg.Init(output_ssa);
    for (LoweredFunc f : funcs) {
        cg.AddFunction(f);
    }
    std::string code = cg.Finish();

    // if ( const auto * f = Registry::Get("tvm_callback_opencl_postproc")) {
    //     code = (*f)(code).operator std::string();
    // }
    LOG(WARNING) << "SDAccel doesn't have runtime, return kernel code";
    // std::unordered_map<std::string, runtime::FunctionInfo> 
    std::unordered_map<std::string, runtime::FunctionInfo> temp = ExtractFuncInfo(funcs);


    return code;
}




TVM_REGISTER_API("codegen.build_sdaccel")
.set_body([]( TVMArgs args, TVMRetValue * rv ) {
    * rv = BuildSDACCEL(args[0]);
    });

TVM_REGISTER_API("codegen.build_aocl")
.set_body([]( TVMArgs args, TVMRetValue * rv ) {
    * rv = BuildAOCL(args[0]);
    });
} // namespace codegen
} // namespace TVM