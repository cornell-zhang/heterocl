/*!
 *  Copyright (c) 2018 by Contributors
 */
#ifndef CODEGEN_HLSC_VHLS_MODULE_H_
#define CODEGEN_HLSC_VHLS_MODULE_H_

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include "../build_common.h"

namespace TVM {
namespace runtime {

Module CreateVivadoHLSModule(LoweredFunc func, std::string code);

}  // namespace runtime
}  // namespace TVM

#endif  // CODEGEN_HLSC_VHLS_MODULE_H_
