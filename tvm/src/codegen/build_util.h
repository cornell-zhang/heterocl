/*!
 *  Copyright (c) 2019 by Contributors
 *  Common build utilities
 * \file build_util.h
 */
#ifndef TVM_CODEGEN_BUILD_HELPER_H_
#define TVM_CODEGEN_BUILD_HELPER_H_

#include <tvm/codegen.h>
#include <unordered_map>
#include <string>
#include "../runtime/meta_data.h"

namespace TVM {
namespace runtime {

// get current work directory
std::string getpath(void);
void PrintIndent(std::ofstream& stream, int indent);
inline size_t GetTypeSize(TVMType t);
inline size_t GetDataSize(TVMArray* arr);
inline TVMType Type2TVMType(Type t);
inline std::string PrintHalideType(Type t);
inline std::string Type2Str(TVMType t);
inline std::string Type2ExtStr(TVMType t);
inline std::string Type2WrapStr(TVMType t);
inline std::string Type2Byte(TVMType t);

void CollectArgInfo(TVMArgs& args, 
                    LoweredFunc func,
                    std::vector<size_t>& arg_sizes,
                    std::vector<TVMType>& arg_types);

void GenSharedMem(TVMArgs& args,
                  std::vector<int>& shmids,
                  std::vector<size_t>& arg_sizes);

void FreeSharedMem(TVMArgs& args, 
                   const std::vector<int>& shmids,
                   std::vector<size_t>& arg_sizes);

void PrintCopy(TVMArray* arr, 
               std::ofstream& stream, 
               int indent, size_t nth_arr);

void PrintCopyBack(TVMArray* arr, 
                   std::ofstream& stream, 
                   int indent, size_t nth_arr);

void GenKernelCode(std::string& test_file, 
                   std::string platform,
                   std::string backend);

void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc func,
                 std::string platform,
                 std::string host_code,
                 std::vector<std::string> arg_names);
} // namespace runtime
} // namespace TVM
#endif  // TVM_CODEGEN_BUILD_HELPER_H_
