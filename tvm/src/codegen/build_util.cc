/*!
 *  Copyright (c) 2019 by Contributors
 * \file build_util.cc
 * \brief Build unified simulation module
 */
#include <tvm/base.h>
#include <tvm/ir_visitor.h>
#include <tvm/runtime/config.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/build_module.h>
#include "./build_common.h"
#include "./build_util.h"

#include <fstream>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <iostream>
#include <regex>
#include <string>

#include "merlinc/codeanalys_merlinc.h"
#include "hlsc/codegen_vhls.h"
#include "opencl/codegen_aocl.h"
#include "ppac/codegen_rv64_ppac.h"

namespace TVM {
namespace runtime {

std::string getpath(void) {
   char buff[256];
   getcwd(buff, 256);
   std::string cwd(buff);
   return cwd;
}

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

inline std::vector<int> GetShape(TVMArray* arr) {
  std::vector<int> shape;
  for (tvm_index_t i = 0; i < arr->ndim; ++i) 
    shape.push_back(arr->shape[i]);
  return shape;
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

inline std::string Type2Str(TVMType t) {
  std::string str = "";
  if (t.code == kDLInt) {
    if (t.fracs > 0) str += "ap_fixed<";
    else             str += "ap_int<";
    str += std::to_string(static_cast<int>(t.bits));
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits - t.fracs)) + ">";
    else             str += ">";
  } else if (t.code == kDLUInt) {
    if (t.fracs > 0) str += "ap_ufixed<";
    else             str += "ap_uint<";
    str += std::to_string(static_cast<int>(t.bits));
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits - t.fracs)) + ">";
    else             str += ">";
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

inline std::string Type2WrapStr(TVMType t) {
  std::string str = "";
  if (t.code == kDLInt) {
    if (t.fracs > 0) {
      str += "ap_fixed<";
      str += std::to_string(static_cast<int>(t.bits + t.fracs));
    } else {
      str += "ap_int<";
      if      (t.bits <= 8)  str += std::to_string(static_cast<int>(t.bits));
      else if (t.bits <= 16) str += "16";
      else if (t.bits <= 32) str += "32";
      else                   str += "64";
    }     
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.bits)) + ">";
    else             str += ">";
  } else if (t.code == kDLUInt) {
    if (t.fracs > 0) {
      str += "ap_ufixed<";
      str += std::to_string(static_cast<int>(t.bits + t.fracs));
    } else {
      str += "ap_uint<";
      if      (t.bits <= 8)  str += std::to_string(static_cast<int>(t.bits));
      else if (t.bits <= 16) str += "16";
      else if (t.bits <= 32) str += "32";
      else                   str += "64"; 
    }
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
    if (t.code == kDLUInt) str += "u";
    str += "int";
    if      (t.bits <= 8)  str += "8";
    else if (t.bits <= 16) str += "16";
    else if (t.bits <= 32) str += "32";
    else                   str += "64";
    str += "_t";
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

    } else { // shared memory for var
      key_t key = ftok("/", i+1);
      int shmid = shmget(key, arg_sizes[i], 0666|IPC_CREAT);
      shmids.push_back(shmid);
      // copy mem from TVM Var to the shared memory
      int data = int64_t(args[i]);
      void* mem = shmat(shmid, nullptr, 0);
      memcpy(mem, &data, arg_sizes[i]);
    }
  }
}

void FreeSharedMem(TVMArgs& args, 
                   const std::vector<int>& shmids,
                   std::vector<size_t>& arg_sizes) {
  for (size_t i = 0; i < shmids.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      int shmid = shmids[i];
      void* mem = shmat(shmid, nullptr, 0);
      memcpy(arr->data, mem, arg_sizes[i]);
      shmdt(mem);
      shmctl(shmid, IPC_RMID, nullptr);
    }
  }
}

// copy values from the shared mem to local mem
void PrintCopy(TVMArray* arr, 
               std::vector<std::string> arg_names,
               std::ofstream& stream, 
               int indent, size_t nth_arr) {
  for (int i = 0; i < arr->ndim; i++) {
    PrintIndent(stream, indent);
    stream << "for (size_t i" << i << " = 0; ";
    stream << "i" << i << " < " << arr->shape[i] << "; ";
    stream << "i" << i << "++) {\n";
    indent += 2;
    if (i == arr->ndim - 1) {
      PrintIndent(stream, indent);
      stream << arg_names[nth_arr];
      stream << "[i" << arr->ndim-1;
      int mul2 = 1;
      for (int j = arr->ndim-2; j >= 0; j--) {
        mul2 *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul2;
      }
      stream << "]";

      stream << " = (";
      // stream << Type2ExtStr(arr->dtype);
      stream << Type2Byte(arr->dtype);

      stream << ")(arg_" << nth_arr;
      stream << "[i" << arr->ndim-1;
      int mul = 1;
      for (int j = arr->ndim-2; j >= 0; j--) {
        mul *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul;
      }
      stream << "])";
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
                   std::vector<std::string> arg_names,
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
      stream << "] = (";
      stream << Type2Byte(arr->dtype);
      stream << ")(" << arg_names[nth_arr];
      stream << "[i" << arr->ndim - 1;
      int mul2 = 1;
      for (int j = arr->ndim-2; j >= 0; j--) {
        mul2 *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul2;
      }

      stream << "])";
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

// generate kernel code into files 
void GenKernelCode(std::string& test_file, std::vector<std::string> arg_names, 
                   std::string platform, std::string backend) {
  if (test_file.find_first_not_of(" \t\n") == std::string::npos) return;
  std::ofstream stream;

  std::string kernel_ext = "cpp";
  if (platform == "sdaccel" && backend == "sdaccel") kernel_ext = "cl";
  if (platform == "aocl") kernel_ext = "cl";
  stream.open("project/kernel." + kernel_ext);

  // generate hash
  std::hash<std::string> hasher;
  stream << "// HASH:" << (size_t)hasher(test_file) % 100000 << "\n";

  // create typedef and header 
  if (platform == "vivado" || platform == "vivado_hls" ||
      platform == "sdsoc") { 

    // add header file to host code 
    auto pos = test_file.rfind("#include ");
    auto next = test_file.find('\n', pos);
    test_file.insert(next + 1, "#include \"kernel.h\"\n");

    // create typedef list 
    std::unordered_map<std::string, std::string> typedef_map({ 
        { "ap_uint<32>" , "ubit32" }, 
        { "ap_int<32>"  , "bit32"  } 
    });

    for (auto& kv : typedef_map) {
      while (test_file.find(kv.first) != std::string::npos)
        test_file.replace(test_file.find(kv.first), 
            kv.first.length(), kv.second);
    }

    // generate header file
    std::ofstream header;
    header.open("project/kernel.h");
    header << "#ifndef __KERNEL_H__\n" 
           << "#define __KERNEL_H__\n\n";
    header << "#include <ap_int.h>\n";
    header << "#include <ap_fixed.h>\n";
    header << "#include <hls_stream.h>\n";
    for (auto& kv : typedef_map) {
      header << "typedef " << kv.first << " "
             << kv.second << ";\n";
    }

    // locate top function
    CHECK(test_file.find("test(") != std::string::npos) 
      << "cannot find top function";
    size_t dut = test_file.find("test(");
    size_t begin = test_file.rfind('\n', dut);
    size_t end = test_file.find(')', dut) + 1;

    // TODO: better way to specify prgamas
    if (platform == "sdsoc") {
      // TODO: direct memory interface with PL and DDR
      header << "#pragma SDS data copy(";
      for (size_t k = 0; k < arg_names.size(); k++) {
        if (k != 0) header << ", ";
        header << arg_names[k] << "[0:256]";
      }
      header << ")\n";
      header << "#pragma SDS data access_pattern(";
      for (size_t k = 0; k < arg_names.size(); k++) {
        if (k != 0) header << ", ";
        header << arg_names[k] << ":SEQUENTIAL";
      }
      header << ")\n";
      // generate AFU with AXI DMA
      header << "#pragma SDS data data_mover(";
      for (size_t k = 0; k < arg_names.size(); k++) {
        if (k != 0) header << ", ";
        header << arg_names[k] << ":AXIDMA_SG";
      }
      header << ")";
    }

    header << test_file.substr(begin, end - begin) 
           << ";\n" << "\n#endif";
    header.close();
    stream << "#include <ap_int.h>\n";
    stream << "#include <ap_fixed.h>\n";

  } else if (platform == "aocl")  {
    stream << "#include \"ihc_apint.h\"\n";

  } else if (platform == "vitis") {
    stream << "#include <ap_int.h>\n";
    stream << "#include <ap_fixed.h>\n";
    stream << "#include <hls_math.h>\n";
  }

  stream << test_file;
  stream.close();
}

// generate opencl wrapper for sdaccel sim
void GenHostHeaders(std::ofstream& stream,
                    std::string platform, std::string include) {
  stream << R"(
#include <sys/ipc.h>
#include <sys/shm.h>

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <time.h>
#include <sys/time.h>
#include <cassert>

)";
  
  if (platform == "sdaccel" || platform == "vitis") {
    stream << "// opencl harness headers\n";
    stream << "#include \"xcl2.hpp\"\n";
    stream << "#include \"ap_fixed.h\"\n";
    stream << "#include \"ap_int.h\"\n";
    stream << "#include <cmath>\n";
    stream << "#include <vector>\n\n";

  } else if (platform == "vivado_hls" || 
             platform == "vivado" || platform == "sdsoc") {

    if (platform == "sdsoc") 
      stream << "#include \"sds_lib.h\"\n";

    stream << "// vivado hls headers\n";
    stream << "#include <ap_int.h>\n";
    stream << "#include <ap_fixed.h>\n";
    stream << "#include <hls_stream.h>\n";
    stream << "#include \"kernel.h\"\n\n";

  } else if (platform == "aocl") {
    stream << "#include \"CL/opencl.h\"\n";
    stream << "#pragma message (\"* Compiling for ALTERA CL\")\n";
    stream << "#define AOCX_FILE \"kernel.aocx\"\n\n";

    stream << R"(

#define CHECK(status) 							\
    if (status != CL_SUCCESS)						\
{									\
    fprintf(stderr, "error %d in line %d.\n", status, __LINE__);	\
    exit(1);								\
}									\

void* acl_aligned_malloc (size_t size) {
  void *result = NULL;
  posix_memalign (&result, 64, size);
  return result;
}

)";

  }
  stream << include << "\n";

}

// separate host code into partitions 
std::string SplitHostCode(std::string host_code, std::string& include) {
  // TODO: create a osstringstream for include string
  size_t pos = host_code.find("default_function");
  include = host_code.substr(0, host_code.rfind("void", pos));

  std::string main_body = host_code.substr(host_code.find("{", pos) + 1);
  auto begin = main_body.find_first_not_of(" \t\n");
  auto length = main_body.rfind("}") - begin;
  main_body = main_body.substr(begin, length);

  return "\n  " + main_body;
}

// generate host code according to platform type
void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc lowered_func,std::string platform,
                 std::string host_code, 
                 std::vector<std::string> arg_names,
                 bool kernel_is_empty) {
  int indent = 0;
  std::ofstream stream;
  stream.open("project/host.cpp");

  std::string include;
  auto code = SplitHostCode(host_code, include); 

  GenHostHeaders(stream, platform, include);
  CHECK((signed)arg_names.size() == args.size());

  stream << "int main(int argc, char ** argv) {\n";
  indent += 2;

  int cnt = 0; // label the constant value
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      // read from the shared memory
      PrintIndent(stream, indent);
      stream << Type2Byte(arg_types[i]) << "* "; 
      stream << "arg_" << i << " = ";
      stream << "(" << Type2Byte(arg_types[i]) << "*)";
      stream << "shmat(/*" << arg_names[i] << "*/" 
             << shmids[i] << ", nullptr, 0);\n";
      PrintIndent(stream, indent);

      stream << Type2Byte(arg_types[i]) << "* ";
      stream << arg_names[i];
      stream << " = new " << Type2Byte(arg_types[i]);
      TVMArray* arr = args[i];

      stream << "[";
      for (int j = 0; j < arr->ndim; j++) {
        if (j == arr->ndim - 1) {
          stream << arr->shape[j];
        } else {
          stream << arr->shape[j];
          stream << " * ";
        }
      }
      stream << "];\n";
      PrintCopy(arr, arg_names, stream, indent, i);

    } else {
      // read from shared mem for var 
      PrintIndent(stream, indent);
      stream << Type2Byte(arg_types[i]) << "* ";

      stream << "arg_" << i << " = ";
      stream << "(" << Type2Byte(arg_types[i]) << "*)";
      stream << "shmat(" << shmids[i] << ", nullptr, 0);\n";

      PrintIndent(stream, indent);
      stream << Type2Byte(arg_types[i]) << " ";
      stream << arg_names[i];
      stream << " = (";
      stream << "arg_" << i << "[0])";

      if (arg_types[i].fracs > 0)
        stream << " >> " << static_cast<int>(arg_types[i].fracs);
      stream << ";\n";
      cnt += 1;
    }
    stream << "\n";
  }

  if (!kernel_is_empty) {
    if (platform == "sdaccel" || platform == "vitis") {
      stream << R"(
  if (argc != 2) {
      std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
      return EXIT_FAILURE;
  }

  auto binaryFile = argv[1];
  cl_int err = CL_SUCCESS;

  // create binary file and program
  auto devices = xcl::get_xil_devices();
  auto device_count = devices.size();
  auto device = devices[0];

  cl::Context context(device, NULL, NULL, NULL, &err);
  cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  auto device_name = device.getInfo<CL_DEVICE_NAME>();
  std::cout << "Found Device=" << device_name.c_str() << std::endl;

  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  devices.resize(1);
  cl::Program program(context, devices, bins, NULL, &err);

)";

    } else if (platform == "aocl") {
      stream << R"(

  cl_int status;
  cl_uint numDevices = 0;
  cl_uint numPlatforms = 0;
  cl_platform_id* platforms = NULL;
  const cl_uint maxDevices = 4;
  cl_device_id devices[maxDevices];
  cl_event kernel_exec_event;

  // global and local worksize
  size_t globalWorkSize[1] = {1};
  size_t localWorkSize[1] = {1};

  // get platform and device information 
  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  platforms = (cl_platform_id*) acl_aligned_malloc (numPlatforms * sizeof(cl_platform_id));
  status = clGetPlatformIDs(numPlatforms, platforms, NULL); CHECK(status);
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,
      maxDevices, devices, &numDevices); CHECK(status);

  // create contex and command queue 
  cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, &status);
  CHECK(status);
  cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[0], 
      CL_QUEUE_PROFILING_ENABLE, &status);
  CHECK(status);

  // read aocx and create binary
  FILE *fp = fopen(AOCX_FILE, "rb");
  fseek(fp, 0, SEEK_END);
  size_t  binary_length = ftell(fp);

  // create program from binary 
  const unsigned char *binary;
  binary = (unsigned char*) malloc(sizeof(unsigned char) * binary_length);
  assert(binary && "Malloc failed"); rewind(fp);
  if (fread((void*)binary, binary_length, 1, fp) == 0) {
    printf("Failed to read from the AOCX file (fread).\n");
    return -1;
  }
  fclose(fp);
  cl_program program = clCreateProgramWithBinary(context, 1, devices,
      &binary_length, (const unsigned char **)&binary, &status, NULL);

  status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  CHECK(status);

)";
    }

    stream << "\n";
  }
  PrintIndent(stream, indent);
  stream << "// compute and kernel call from host";
  stream << code << "\n";

  // copy to shared mem
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      PrintCopyBack(arr, arg_names, stream, indent, i);
      PrintIndent(stream, indent);
      stream << "shmdt(";
      stream << "arg_" << i << ");\n";
    }
  }

  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "}\n";
  stream.close();

}
}  // namespace runtime
}  // namespace TVM
