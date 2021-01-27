/*!
 *  Copyright (c) 2019 by Contributors
 */
#ifndef CODEGEN_BUILD_SODA_H_
#define CODEGEN_BUILD_SODA_H_

#include <sys/types.h>
#include <sys/wait.h>
#include <tvm/base.h>
#include <unistd.h>
#include <fstream>

namespace TVM {
namespace codegen {

void SODA2HLSC(std::string& code);

}
}  // namespace TVM

#endif  // CODEGEN_BUILD_SODA_H_
