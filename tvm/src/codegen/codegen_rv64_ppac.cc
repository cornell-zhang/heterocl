/*
    author Guyue Huang (gh424@cornell.edu)
 */
 
#include <vector>
#include <string>
#include <fstream>
#include <sys/types.h>
#include <iomanip>
#include "./codegen_rv64_ppac.h"
 
namespace TVM {
namespace codegen {

void CodeGenRV64PPAC::PrintType(Type t, std::ostream& os) {
  CHECK_EQ(t.lanes(), 1)
      << "do not yet support vector types";
  if (t.is_handle()) {
    os << "void*"; return;
  }
  if (t.is_float()) {
    if (t.bits() == 32) {
      os << "float"; return;
    }
    if (t.bits() == 64) {
      os << "double"; return;
    }
  } else if (t.is_uint()) {
    if (t.bits() <= 8) {
      os << "uint8_t"; return;
    }
    else if (t.bits() <= 16) {
      os << "uint16_t"; return;
    }
    else if (t.bits() <= 32) {
      os << "uint32_t"; return;
    }
    else if (t.bits() <= 64) {
      os << "uint64_t"; return;
    }
    else {
      os << "uint64_t";
      LOG(WARNING) << "Casting type " << t << " to uint64_t";
    }
  } else if (t.is_int()) {
    if (t.bits() <= 8) {
      os << "int8_t"; return;
    }
    else if (t.bits() <= 16) {
      os << "int16_t"; return;
    }
    else if (t.bits() <= 32) {
      os << "int32_t"; return;
    }
    else if (t.bits() <= 64) {
      os << "int64_t"; return;
    }
    else {
      os << "int64_t";
      LOG(WARNING) << "Casting type " << t << " to int64_t";
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

}
}