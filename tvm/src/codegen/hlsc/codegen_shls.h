/*!
 * Copyright (c) 2021 by Contributors
 * \file codegen_vhls.h
 * \brief Generate Stratus HLS kernel code.
 */
#ifndef CODEGEN_HLSC_CODEGEN_SHLS_H_
#define CODEGEN_HLSC_CODEGEN_SHLS_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <fstream>
#include <string>
#include "../merlinc/codeanalys_merlinc.h"
#include "./codegen_hlsc.h"
#include "./codegen_vhls.h"

namespace TVM {
namespace codegen {

class CodeGenStratusHLS final : public CodeGenVivadoHLS {
 public:
  void AddFunction(LoweredFunc f, str2tupleMap<std::string, Type> map_arg_type);

};


} // namespace codegen
} // namespace TVM

#endif // CODEGEN_HLSC_CODEGEN_SHLS_H_