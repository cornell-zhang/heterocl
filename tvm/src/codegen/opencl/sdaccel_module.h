/*
    Yang.Bai
    yb269@cornell.edu
*/
#ifndef SDACCEL_MODULE_H
#define SDACCEL_MODULE_H

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include "../build_common.h"

namespace TVM {
namespace runtime {

Module CreateSDAccelModule(
    LoweredFunc func,
    std::string code);

} // namespace runtime
} // namespace TVM

#endif
