/*!
 *  Copyright (c) 2017 by Contributors
 * \file module_util.h
 * \brief Helper utilities for module building
 */
#ifndef RUNTIME_MODULE_UTIL_H_
#define RUNTIME_MODULE_UTIL_H_

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <vector>

extern "C" {
// Function signature for generated packed function in shared library
typedef int (*BackendPackedCFunc)(void* args, int* type_codes, int num_args);
}  // extern "C"

namespace TVM {
namespace runtime {
/*!
 * \brief Wrap a BackendPackedCFunc to packed function.
 * \param faddr The function address
 * \param mptr The module pointer node.
 */
PackedFunc WrapPackedFunc(BackendPackedCFunc faddr,
                          const std::shared_ptr<ModuleNode>& mptr);
/*!
 * \brief Load and append module blob to module list
 * \param mblob The module blob.
 * \param module_list The module list to append to
 */
void ImportModuleBlob(const char* mblob, std::vector<Module>* module_list);

/*!
 * \brief Utility to initialize conext function symbols during startup
 * \param flookup A symbol lookup function.
 * \tparam FLookup a function of signature string->void*
 */
template <typename FLookup>
void InitContextFunctions(FLookup flookup) {
#define TVM_INIT_CONTEXT_FUNC(FuncName)                                      \
  if (auto* fp =                                                             \
          reinterpret_cast<decltype(&FuncName)*>(flookup("__" #FuncName))) { \
    *fp = FuncName;                                                          \
  }
  // Initialize the functions
  TVM_INIT_CONTEXT_FUNC(HCLTVMFuncCall);
  TVM_INIT_CONTEXT_FUNC(HCLTVMAPISetLastError);
  TVM_INIT_CONTEXT_FUNC(HCLTVMBackendGetFuncFromEnv);
  TVM_INIT_CONTEXT_FUNC(HCLTVMBackendAllocWorkspace);
  TVM_INIT_CONTEXT_FUNC(HCLTVMBackendFreeWorkspace);
  TVM_INIT_CONTEXT_FUNC(HCLTVMBackendParallelLaunch);
  TVM_INIT_CONTEXT_FUNC(HCLTVMBackendParallelBarrier);

#undef TVM_INIT_CONTEXT_FUNC
}
}  // namespace runtime
}  // namespace TVM
#endif  // RUNTIME_MODULE_UTIL_H_
