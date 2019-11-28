#include "./sdaccel_module.h"
#include <fstream>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <iostream>
#include <cstring>
#include <typeinfo>

namespace TVM {
namespace runtime {

namespace {

void PrintIndent(std::ofstream& stream, int indent) {
  for (int i = 0; i < indent; i++)
    stream << ' ';
}

inline size_t GetTypeSize(TVMType t) {
  size_t byte = (t.bits + 7) / 8;
  if (byte > 2){
    if (byte <= 4) byte = 4;
    else if (byte <= 8) byte = 8;
    else byte = 16;
  }
  return byte;
}

inline size_t GetDataSize(TVMArray* arr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < arr->ndim; ++i) {
    size *= arr->shape[i];
  }
  size_t byte = (arr->dtype.bits + 7) / 8;
  if (byte > 2){
    if (byte <= 4) byte = 4;
    else if (byte <= 8) byte = 8;
    else byte = 16;
  }
  size *= (byte * 8 * arr->dtype.lanes + 7) / 8;
  return size;
}

inline TVMType Type2TVMType(Type t) {
  TVMType tt;
  if (t.is_int())        tt.code = kDLInt;
  else if (t.is_uint())  tt.code = kDLUInt;
  else if (t.is_float()) tt.code = kDLFloat;
  else                   LOG(FATAL) << "Unacceptable type: " << t;
  tt.bits = static_cast<uint8_t>(t.bits());
  tt.fracs = static_cast<uint8_t>(t.fracs());
  return tt;
}


// inline std::string Type2Str(TVMType t) {
//   std::string str = "";
//   if (t.code == kDLInt) {
//     if (t.fracs > 0) str += "ap_fixed<";
//     else             str += "ap_int<";
//     str += std::to_string(static_cast<int>(t.bits));
//     if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits - t.fracs)) + ">";
//     else             str += ">";
//   } else if (t.code == kDLUInt) {
//     if (t.fracs > 0) str += "ap_ufixed<";
//     else             str += "ap_uint<";
//     str += std::to_string(static_cast<int>(t.bits));
//     if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits - t.fracs)) + ">";
//     else             str += ">";
//   } else if (t.code == kDLFloat) {
//     str += "float";
//   } else {
//     LOG(FATAL) << "Unknown type";
//   }
//   return str;
// }

inline std::string Type2Str(TVMType t) {
  std::string str = "";
  if (t.code == kDLInt) {
    str += "int";
    // if (t.fracs > 0) str += "ap_fixed<";
    // else             str += "ap_int<";
    // str += std::to_string(static_cast<int>(t.bits));
    // if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits - t.fracs)) + ">";
    // else             str += ">";
  } else if (t.code == kDLUInt) {
    str += "unsigned int";
    // if (t.fracs > 0) str += "ap_ufixed<";
    // else             str += "ap_uint<";
    // str += std::to_string(static_cast<int>(t.bits));
    // if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits - t.fracs)) + ">";
    // else             str += ">";
  } else if (t.code == kDLFloat) {
    str += "float";
  } else {
    LOG(FATAL) << "Unknown type";
  }
  return str;
}

inline std::string Type2ExtStr(TVMType t) {
  std::string str = "";
  if (t.code == kDLInt) {
    if (t.fracs > 0) str += "ap_fixed<";
    else             str += "ap_int<";
    str += std::to_string(static_cast<int>(t.bits + t.fracs));
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits)) + ">";
    else             str += ">";
  } else if (t.code == kDLUInt) {
    if (t.fracs > 0) str += "ap_ufixed<";
    else             str += "ap_uint<";
    str += std::to_string(static_cast<int>(t.bits + t.fracs));
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits)) + ">";
    else             str += ">";
  } else if (t.code == kDLFloat) {
    str += "float";
  } else {
    LOG(FATAL) << "Unknown type";
  }
  return str;
}

inline std::string Type2Byte(TVMType t) {
  std::string str = "";
  if (t.code == kDLFloat) {
    str += "float";
  } else if (t.code == kDLInt || t.code == kDLUInt) {
    if (t.code == kDLUInt) str += "unsigned";
    str += "int";
    if      (t.bits <= 8)  str += "8";
    else if (t.bits <= 16) str += "16";
    else if (t.bits <= 32) str += "32";
    else                   str += "64";
    // str += "_t";
  }
  return str;
}

void CollectArgInfo(TVMArgs& args, 
                    LoweredFunc func,
                    std::vector<size_t>& arg_sizes,
                    std::vector<TVMType>& arg_types) {
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      arg_sizes.push_back(GetDataSize(arr));
      arg_types.push_back(arr->dtype);
    } else {
      const Variable* var = func->api_args[i].as<Variable>();
      TVMType t = Type2TVMType(var->type);
      arg_sizes.push_back(GetTypeSize(t));
      arg_types.push_back(t);
    }
  }
}

void GenSharedMem(TVMArgs& args,
                  std::vector<int>& shmids,
                  std::vector<size_t>& arg_sizes) {
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      // generate shared memory key and id
      // TODO: maybe get the current path??
      key_t key = ftok("/", i+1);
      int shmid = shmget(key, arg_sizes[i], 0666|IPC_CREAT);
      shmids.push_back(shmid);
      // copy mem from TVM args to the shared memory
      void* mem = shmat(shmid, nullptr, 0);
      memcpy(mem, arr->data, arg_sizes[i]);
    } else {
      shmids.push_back(0);
    }
  }
}

void FreeSharedMem(TVMArgs& args, 
                   const std::vector<int>& shmids,
                   std::vector<size_t>& arg_sizes) {
  for (size_t i = 0; i < shmids.size(); i++) {
    // if (args[i].type_code() == kArrayHandle) {
    //   TVMArray* arr = args[i];
    //   int shmid = shmids[i];
    //   void* mem = shmat(shmid, nullptr, 0);
    //   memcpy(arr->data, mem, arg_sizes[i]);
    //   shmdt(mem);
    //   shmctl(shmid, IPC_RMID, nullptr);
    // }
      TVMArray* arr = args[i];
      int shmid = shmids[i];
      void* mem = shmat(shmid, nullptr, 0);
      memcpy(arr->data, mem, arg_sizes[i]);
      shmdt(mem);
      shmctl(shmid, IPC_RMID, nullptr);
  }
}

// copy values from the shared mem to local mem
void PrintCopy(TVMArray* arr, 
               std::ofstream& stream, 
               int indent, size_t nth_arr) {
  for (int i = 0; i < arr->ndim; i++) {
    PrintIndent(stream, indent);
    stream << "for (size_t i" << i << " = 0; ";
    stream << "i" << i << " < " << arr->shape[i] << "; ";
    stream << "i" << i << "++) {\n";
    indent += 2;
    if (i == arr->ndim-1) {
      PrintIndent(stream, indent);
      stream << "source_" << nth_arr;
      stream << "[i" << arr->ndim-1;
      int mul = 1;
      for (int j = arr->ndim-2;j >= 0;j--) {
        mul *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul;
      }
      stream << "] = ";
      stream << "arg_" << nth_arr;
      stream << "[i" << arr->ndim - 1;

      int mul2 = 1;
      for (int j = arr->ndim-2;j >= 0;j--) {
        mul2 *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul2;
      }
      stream << "]";
      if (arr->dtype.fracs > 0)
        stream << " >> " << static_cast<int>(arr->dtype.fracs);
      stream << ";\n";
    }
  }
  for (int i = 0; i < arr->ndim; i++) {
    indent -= 2;
    PrintIndent(stream, indent);
    stream << "}\n";
  }
}

// copy values from local mem back to shared mem
void PrintCopyBack(TVMArray* arr, 
                   std::ofstream& stream, 
                   int indent, size_t nth_arr) {
  for (int i = 0; i < arr->ndim; i++) {
    PrintIndent(stream, indent);
    stream << "for (size_t i" << i << " = 0; ";
    stream << "i" << i << " < " << arr->shape[i] << "; ";
    stream << "i" << i << "++) {\n";
    indent += 2;
    if (i == arr->ndim-1) {
      PrintIndent(stream, indent);
      stream << "arg_" << nth_arr;
      stream << "[i" << arr->ndim-1;
      int mul = 1;
      for (int j = arr->ndim-2; j >= 0; j--) {
        mul *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul;
      }
      stream << "] = ";
      // stream << Type2ExtStr(arr->dtype);
      stream << "source_" << nth_arr;
      stream << "[i" << arr->ndim - 1;
      int mul2 = 1;
      for (int j = arr->ndim-2;j >=0;j--) {
        mul2 *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul2;
      }
      stream << "]";
      if (arr->dtype.fracs > 0)
        stream << " << " << static_cast<int>(arr->dtype.fracs);
      stream << ";\n";
    }
  }
  for (int i = 0; i < arr->ndim; i++) {
    indent -= 2;
    PrintIndent(stream, indent);
    stream << "}\n";
  }
}

void GenMakFile() {
  int indent = 0;
  std::ofstream stream;
  stream.open("sdaccel.mk");
  indent += 4;

  stream << "ifndef XILINX_SDX\n";
  stream << "$(error Environment variable XILINX_SDX is required and should point to SDAccel install area)\n";
  stream << "endif\n";

  stream << "SDA_FLOW = cpu_emu\n";
  stream << "HOST_SRCS = host.cpp\n";
  stream << "HOST_EXE_DIR=.\n";
  stream << "HOST_EXE = host\n";
  stream << "HOST_CFLAGS = -g -Wall -DFPGA_DEVICE -DC_KERNEL\n";
  stream << "HOST_LFLAGS = \n";
  stream << "KERNEL_SRCS = default_function.cl\n";
  stream << "KERNEL_NAME = default_function\n";
  stream << "KERNEL_DEFS =\n";
  stream << "KERNEL_INCS =\n";
  stream << "XDEVICE=xilinx:adm-pcie-7v3:1ddr:3.0\n";
  stream << "XDEVICE_REPO_PATH=\n";
  stream << "KEEP_TEMP=1\n";
  stream << "KERNEL_DEBUG=\n";
  stream << "XCLBIN_NAME=bin_krnl\n";
  stream << "HOST_CFLAGS+=-DTARGET_DEVICE=\\\"${XDEVICE}\\\"\n";
  stream << "BOARD_SETUP_FILE=setup.sh\n";
  stream << "ifeq (${SDA_FLOW},cpu_emu)\n";
  PrintIndent(stream, indent);
  stream << "CLCC_OPT += -t sw_emu\n";
  PrintIndent(stream, indent);
  stream << "XCLBIN = ${XCLBIN_NAME}_cpu_emu.xclbin\n"; 
  stream << "else ifeq (${SDA_FLOW},hw_emu)\n";
  PrintIndent(stream, indent);
  stream << "CLCC_OPT += -t hw_emu\n";
  PrintIndent(stream, indent);
  stream << "XCLBIN = ${XCLBIN_NAME}_hw_emu.xclbin\n";
  stream << "else ifeq (${SDA_FLOW},hw)\n";
  PrintIndent(stream, indent);
  stream << "XCLBIN = ${XCLBIN_NAME}_hw.xclbin\n";
  stream << "CLCC_OPT += -t hw\n";
  stream << "endif\n";

  stream << "HOST_ARGS = ${XCLBIN}\n";
  stream << "COMMON_DIR = ./common\n";
  stream << "include ${COMMON_DIR}/common.mk\n";

  stream.close();
}

void GenCommonFile() {
  int indent = 0;
  std::ofstream stream;
  stream.open("./common/common.mk");
  indent += 4;
  stream << "SHELL = /bin/bash\n";
  stream << "VPATH = ./\n";
  stream << "CC = xcpp\n";
  stream << "CLCC = xocc\n";
  stream << "ifeq ($(XDEVICE_REPO_PATH),)\n";
  PrintIndent(stream, indent);
  stream << "DEVICE_REPO_OPT = \n";
  stream << "else\n";
  stream << "DEVICE_REPO_OPT = --xp prop:solution.device_repo_paths=${XDEVICE_REPO_PATH}\n";
  stream << "endif\n";
  stream << "HOST_CFLAGS += -I${XILINX_SDX}/runtime/include/1_2\n";
  stream << "HOST_LFLAGS += -L${XILINX_SDX}/runtime/lib/x86_64 -lxilinxopencl -lrt -pthread\n";
  stream << "CLCC_OPT += $(CLCC_OPT_LEVEL) ${DEVICE_REPO_OPT} --xdevice ${XDEVICE} -o ${XCLBIN} ${KERNEL_DEFS} ${KERNEL_INCS}\n";
  stream << "ifeq (${KEEP_TEMP},1)\n";
  PrintIndent(stream, indent);
  stream << "CLCC_OPT += -s\n";
  stream << "endif\n";
  stream << "ifeq (${KERNEL_DEBUG},1)\n";
  PrintIndent(stream, indent);
  stream << "CLCC_OPT += -g\n";
  stream << "endif\n";
  stream << "CLCC_OPT += --kernel ${KERNEL_NAME}\n";
  stream << "OBJECTS := $(HOST_SRCS:.cpp=.o)\n";
  stream << ".PHONY: all\n";
  stream << "all: run\n";

  stream << "host: ${HOST_EXE_DIR}/${HOST_EXE}\n";
  stream << "xbin_cpu_em:\n";
  PrintIndent(stream, indent);
  stream << "make SDA_FLOW=cpu_emu xbin -f sdaccel.mk\n";
  stream << "xbin_hw_em:\n";
  PrintIndent(stream, indent);
  stream << "make SDA_FLOW=hw_emu xbin -f sdaccel.mk\n";
  stream << "xbin_hw :\n";
  PrintIndent(stream, indent);
  stream << "make SDA_FLOW=hw xbin -f sdaccel.mk\n";
  stream << "xbin: ${XCLBIN}\n";
  stream << "run_cpu_em: \n";
  PrintIndent(stream, indent);
  stream << "make SDA_FLOW=cpu_emu run_em -f sdaccel.mk\n";
  stream << "run_hw_em: \n";
  PrintIndent(stream, indent);
  stream << "make SDA_FLOW=hw_emu run_em -f sdaccel.mk\n";
  stream << "run_hw : \n";
  PrintIndent(stream, indent);
  stream << "make SDA_FLOW=hw run_hw_int -f sdaccel.mk\n";
  stream << "run_em: xconfig host xbin\n";
  PrintIndent(stream, indent);
  stream << "XCL_EMULATION_MODE=true ${HOST_EXE_DIR}/${HOST_EXE} ${HOST_ARGS}\n";
  stream << "run_hw_int : host xbin_hw\n";
  PrintIndent(stream, indent);
  stream << "source ${BOARD_SETUP_FILE};${HOST_EXE_DIR}/${HOST_EXE} ${HOST_ARGS}\n";
  stream << "estimate : \n";
  PrintIndent(stream, indent);
  stream << "${CLCC} -c -t hw_emu --xdevice ${XDEVICE} --report estimate ${KERNEL_SRCS}\n";
  stream << "xconfig : emconfig.json\n";
  stream << "emconfig.json :\n";
  PrintIndent(stream, indent);
  stream << "emconfigutil --xdevice ${XDEVICE} ${DEVICE_REPO_OPT} --od .\n";
  stream << "${HOST_EXE_DIR}/${HOST_EXE} : ${OBJECTS}\n";
  PrintIndent(stream, indent);
  stream << "${CC} ${HOST_LFLAGS} ${OBJECTS} -o $@\n";
  stream << "${XCLBIN}:\n";
  PrintIndent(stream, indent);
  stream << "${CLCC} ${CLCC_OPT} ${KERNEL_SRCS}\n";
  stream << "%.o: %.cpp\n";
  PrintIndent(stream, indent);
  stream << "${CC} ${HOST_CFLAGS} -c $< -o $@\n";
  stream << "clean:\n";
  PrintIndent(stream, indent);
  stream << "${RM} -rf ${HOST_EXE} ${OBJECTS} ${XCLBIN} emconfig.json _xocc_${XCLBIN_NAME}_*.dir .Xil\n";
  stream << "cleanall: clean\n";
  PrintIndent(stream, indent);
  stream << "${RM} -rf *.xclbin sdaccel_profile_summary.* _xocc_* TempConfig *.log *.jou\n";

  stream.close();
}

void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc func,
                 std::string test_file) {
  int indent = 0;
  std::ofstream stream;
  stream.open("host.cpp");
  indent += 2;

  stream << "#define CL_HPP_CL_1_2_DEFAULT_BUILD\n";
  stream << "#define CL_HPP_TARGET_OPENCL_VERSION 120\n";
  stream << "#define CL_HPP_MINIMUM_OPENCL_VERSION 120\n";
  stream << "#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1\n";
  stream << "#include <CL/cl2.hpp>\n";
  stream << "#include <fstream>\n";
  stream << "#include <sys/types.h>\n";
  stream << "#include <sys/stat.h>\n";
  stream << "#include <fcntl.h>\n";
  stream << "#include <unistd.h>\n";
  stream << "#include <stdlib.h>\n";
  stream << "#include <stdio.h>\n";
  stream << "#include <cstring>\n";
  stream << "#include <iostream>\n";
  stream << "#include <iomanip>\n";
  // stream << "#include <math.h>\n";
  stream << "#include <cmath>\n";
  stream << "#include <sys/ipc.h>\n";
  stream << "#include <sys/shm.h>\n";
  stream << "#pragma once\n";
  stream << "\n\n";
  
  // stream << test_file;
  stream << "\n\n";

  stream << "int main(void) { \n";

  stream << "#if defined(SDX_PLATFORM) && !defined(TARGET_DEVICE)\n";
  indent += 2;
  stream << "  #define STR_VALUE(arg) #arg\n";
  stream << "  #define GET_STRING(name) STR_VALUE(name)\n";
  stream << "  #define TARGET_DEVICE GET_STRING(SDX_PLATFORM)\n";
  stream << "#endif\n";

  // get the krnl code
  PrintIndent(stream, indent);
  stream << "char* xclbinFilename = argv[1];\n";
  stream << "\n";


  // Source Memories
  // std::vector<unsigned int> source_a(LENGTH);
  // for (int i = 0;i < args.size();i++) {
  //   PrintIndent(stream, indent);
  //   stream << Type2Str(arg_types[i]) << " ";
  //   stream << arg_types[i] << " ";
  //   stream << "arg_" << i;
  //   TVMArray* arr = args[i];
  //   for (int j = 0;j < arr->ndim;j++) {
  //     stream << "[" << arr->shape[j] << "]";
  //   }
  //   stream << ";\n";
  // }
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "std::vector<" << Type2Str(arg_types[i]);
    stream << "> ";
    stream << "source_" << i << "(";
    TVMArray* arr = args[i];
    for (int j = 0;j < arr->ndim;j++) {
      if (j == arr->ndim-1) {
        stream << arr->shape[j] << ")";
      } else {
        // stream << " * " << arr->shape[j] << ")";
        stream << arr->shape[j] << " * ";
      }
    }
    stream << ";\n";
  }
  stream << "\n";

  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "size_t vector_size_bytes_" << i;
    stream << " = sizeof(" << Type2Str(arg_types[i]);
    stream << ")";
    TVMArray* arr = args[i];
    for (int j = 0;j < arr->ndim;j++) {
      stream << " * " << arr->shape[j];
    }
    stream << ";\n";
  }
  stream << "\n";

  for (int i = 0;i < args.size();i++ ) {
    // if (args[i].type_code() == kArrayHandle) {
    //   // read from the shared memory
    //   PrintIndent(stream, indent);
    //   stream << Type2Str(arg_types[i]) << "* ";
    //   stream << "arg_" << i << " = ";
    //   stream << "(" << Type2Str(arg_types[i]) << "*)";
    //   stream << "shmat(" << shmids[i] << ", nullptr, 0);\n";
    //   TVMArray* arr = args[i];
    //   // copy from shared mem  
    //   PrintCopy(arr, stream, indent, i);
    // }
      // read from the shared memory
      PrintIndent(stream, indent);
      stream << Type2Str(arg_types[i]) << "* ";
      stream << "arg_" << i << " = ";
      stream << "(" << Type2Str(arg_types[i]) << "*)";
      stream << "shmat(" << shmids[i] << ", nullptr, 0);\n";
      TVMArray* arr = args[i];
      // copy from shared mem  
      PrintCopy(arr, stream, indent, i);
  }



  // Getting First Platform
  PrintIndent(stream, indent);
  stream << "std::vector<cl::Platform> platforms;\n";
  PrintIndent(stream, indent);
  stream << "cl::Platform::get(&platforms);\n";
  PrintIndent(stream, indent);
  stream << "cl::Platform platform = platforms[0];\n";
  stream << "\n";


  // Getting ACCELERATOR Devices and selecting 1st such device
  PrintIndent(stream, indent);
  stream << "std::vector<cl::Device> devices;\n";
  PrintIndent(stream, indent);
  stream << "platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);\n";
  PrintIndent(stream, indent);
  stream << "cl::Device device = devices[0];\n";
  stream << "\n";

  // Creating Context and Command Queue for selected Device
  PrintIndent(stream, indent);
  stream << "cl::Context context(device);\n";
  PrintIndent(stream, indent);
  stream << "cl::CommandQueue q(context, device);\n";
  stream << "\n";


  // Loading XCL Bin into char buffer
  PrintIndent(stream, indent);
  stream << "std::ifstream bin_file(xclbinFilename, std::ifstream::binary);\n";
  PrintIndent(stream, indent);
  stream << "bin_file.seekg (0, bin_file.end);\n";
  PrintIndent(stream, indent);
  stream << "unsigned nb = bin_file.tellg();\n";
  PrintIndent(stream, indent);
  stream << "bin_file.seekg (0, bin_file.beg);\n";
  PrintIndent(stream, indent);
  stream << "char *buf = new char [nb];\n";
  PrintIndent(stream, indent);
  stream << "bin_file.read(buf, nb);\n";
  stream << "\n";


  // Creating Program from Binary File
  PrintIndent(stream, indent);
  stream << "cl::Program::Binaries bins;\n";
  PrintIndent(stream, indent);
  stream << "bins.push_back({buf,nb});\n";
  PrintIndent(stream, indent);
  stream << "devices.resize(1);\n";
  PrintIndent(stream, indent);
  stream << "cl::Program program(context, devices, bins);\n";
  stream << "\n";


  // Creating Kernel and Functor of Kernel
  PrintIndent(stream, indent);
  stream << "int err1;\n";
  PrintIndent(stream, indent);
  stream << "cl::Kernel kernel(program, \"default_function\", &err1);\n";
  PrintIndent(stream, indent);
  stream << "auto default_function = cl::KernelFunctor<";
  for (int i = 0;i < args.size();i++) {
    if (i == args.size() - 1) {
      stream << "cl::Buffer&>(kernel);\n";
    } else {
      stream << "cl::Buffer&, ";
    }
  }
  // stream << "auto default_function = cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&>(kernel);\n";
  stream << "\n";


  // Creating Buffers inside Device
  // cl::Buffer buffer_a(context, CL_MEM_READ_ONLY,  vector_size_bytes);
  // cl::Buffer buffer_b(context, CL_MEM_WRITE_ONLY, vector_size_bytes);
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "cl::Buffer buffer_" << i;
    stream << "(context, CL_MEM_READ_WRITE, vector_size_bytes_" << i << ");\n";
  }
  stream << "\n";

  // Copying input data to Device buffer from host memory
  // q.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, vector_size_bytes, source_a.data());
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    stream << "q.enqueueWriteBuffer(buffer_" << i;
    stream << ", CL_TRUE, 0, vector_size_bytes_" << i;
    stream << ", source_" << i << ".data());\n"; 
  }
  stream << "\n";

  // Running Kernel
  PrintIndent(stream, indent);
  stream << func->name << "(";
  stream << "cl::EnqueueArgs(q, cl::NDRange(1,1,1), cl::NDRange(1,1,1)),";
  for (int i = 0; i < args.size(); i++) {
    stream << "buffer_" << i;
    if (i != args.size()-1) 
      stream << ", ";
  }
  stream << ");\n";

  PrintIndent(stream, indent);
  stream << "q.finish();\n";
  stream << "\n";


  // Copying Device result data to Host memory
  // q.enqueueReadBuffer(buffer_c, CL_TRUE, 0, vector_size_bytes, result_krnl.data());
  for (int i = 0;i < args.size(); i++) {
    PrintIndent(stream, indent);
    stream << "q.enqueueReadBuffer(buffer_" << i;
    stream << ", CL_TRUE, 0, vector_size_bytes_" << i;
    stream << ", source_" << i << ".data());\n";
  }
  stream << "\n";

  // copy to shared mem
  for (int i = 0;i < args.size();i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      PrintCopyBack(arr, stream, indent, i);
      PrintIndent(stream, indent);
      stream << "shmdt(";
      stream << "arg_" << i << ");\n";
    }
  }

  stream << "}\n";
  stream.close();
}
} // namespace


class SDAccelModuleNode final : public ModuleNode {
 public:
  SDAccelModuleNode(LoweredFunc func, std::string test_file) 
    : func_(func), test_file_(test_file) {}

  const char* type_key() const {
    return "sdaccel_sw_emu";

  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv){

        if (args.size() != (int)func_->args.size())
          LOG(FATAL) << "The function should take in " << func_->args.size() 
                     << " inputs but get " << args.size();
        std::vector<size_t> arg_sizes;
        std::vector<TVMType> arg_types;
        std::vector<int> shmids;
        CollectArgInfo(args, func_, arg_sizes, arg_types);
        GenSharedMem(args, shmids, arg_sizes);
        LOG(CLEAN) << "Creating a Host file for SDAccel Runtime ...";
        GenHostCode(args, shmids, arg_types, func_, test_file_);

        LOG(CLEAN) << "Creating a Common folder for common.mk ...";
        system("mkdir common");
        GenCommonFile();

        LOG(CLEAN) << "Creating a Makfile for compling the SDAccel OpenCL Code ...";
        GenMakFile();
        // TODO: find a better way to do the following
        LOG(CLEAN) << "Compiling the generated SDAccel OpenCL Code ...";
        // system("make -f ./sdaccel.mk run_cpu_em");
        LOG(CLEAN) << "Running SDAccel OpenCL Software Simulation ...";
        LOG(CLEAN) << "Finished SDAccel OpenCL Software Simulation ...";
        // system("make -f sdaccel.mk cleanall");
        FreeSharedMem(args, shmids, arg_sizes);
      });
  }

 private:
  LoweredFunc func_;
  std::string test_file_;
};

Module CreateSDAccelModule(LoweredFunc func,
                           std::string code) {
  std::shared_ptr<SDAccelModuleNode> n =
    std::make_shared<SDAccelModuleNode>(func, code);

  return Module(n);
}

} // namespace runtime
} // namespace TVM
