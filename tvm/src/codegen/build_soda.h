#ifndef BUILD_SODA_H
#define BUILD_SODA_H

#include <fstream>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <tvm/base.h>

namespace TVM {
namespace codegen {

void SODA2HLSC(std::string& code);

}
}

#endif
