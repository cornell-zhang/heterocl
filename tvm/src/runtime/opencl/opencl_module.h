/*!
 *  Copyright (c) 2017 by Contributors
 * \file opencl_module.h
 * \brief Execution handling of OPENCL kernels
 */
#ifndef RUNTIME_OPENCL_OPENCL_MODULE_H_
#define RUNTIME_OPENCL_OPENCL_MODULE_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/packed_func.h>
#include <memory>
#include <string>
#include <vector>
#include "../meta_data.h"

namespace TVM {
namespace runtime {
/*!
 * \brief create a opencl module from data.
 *
 * \param data The module data.
 * \param fmt The format of the data, can be "clbin", "cl"
 * \param fmap The map function information map of each function.
 */
Module OpenCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap);
}  // namespace runtime
}  // namespace TVM
#endif  // RUNTIME_OPENCL_OPENCL_MODULE_H_
