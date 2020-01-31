/*!
 *  Copyright (c) 2019 by Contributors
 * \file build_common.cc
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
void GenKernelCode(std::string& test_file, 
                   std::string platform, argInfo& arg_info) {
  std::ofstream stream;
  std::string kernel_ext = "cpp";
  if (platform == "sdaccel") kernel_ext = "cl";
  stream.open("__tmp__/kernel." + kernel_ext);

  if (platform == "vivado" || platform == "vivado_hls" ||
      platform == "sdsoc") { // insert header
    auto pos = test_file.rfind("#include ");
    auto next = test_file.find('\n', pos);
    std::string type = "ap_uint<32>";
    while (test_file.find(type) != std::string::npos)
      test_file.replace(test_file.find(type), type.length(), "bit32");
    test_file.insert(next + 1, "#include \"kernel.h\"\n");

    // generate header file
    std::ofstream header;
    std::string include = test_file.substr(0, next);
    header.open("__tmp__/kernel.h");
    header << "#ifndef __KERNEL_H__\n" 
           << "#define __KERNEL_H__\n\n"
           << include << "\n\n" << "typedef ap_uint<32> bit32;\n";

    // locate top function
    size_t dut = test_file.find("top(");
    size_t begin = test_file.rfind('\n', dut);
    size_t end = test_file.find(')', dut) + 1;

    if (platform == "sdsoc") { 
      // insert kernel with sds pragmas
      bool stream_pragma = false;
      size_t last_active_spot = 0;
      for (size_t i = 0; i < arg_info.size(); i++) {
        auto& info = arg_info[i];
        if (info.streamed) { // TODO: copy, mover
          if (!stream_pragma) { 
            stream_pragma = true;
            header << "#pragma SDS data access_pattern(";
          }
          if (i != 0 && last_active_spot == i - 1) header << ", ";
          last_active_spot = i;
          header << info.name << ":SEQUENTIAL";
        }
      }
      if (stream_pragma) header << ")";
    }
    header << test_file.substr(begin, end - begin) 
           << ";\n" << "\n#endif";
    header.close();
  } 
  stream << test_file;
  stream.close();
}

// memory and control interface 
void GenWrapperCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 argInfo& arg_info,
                 LoweredFunc func) {
  std::ofstream stream;
  int indent = 0;
  std::string path(getenv("PWD"));
  stream.open("__tmp__/interface.cpp");
  stream << "#include <stdio.h>\n";
  stream << "#include \"" + path + "/__tmp__/kernel.cpp\"\n";
  stream << "\n\n";
  stream << "extern \"C\" \n";
  stream << "{\n";
  indent += 2;
  PrintIndent(stream, indent);

  // wrapper func for FPGA kernel
  stream << "void App( ";
  for (size_t i = 0; i < arg_types.size(); i++) {
    if (i != 0) stream << ", ";
    stream << Type2WrapStr(arg_types[i]);
    stream << "*";
    stream << " source_wrapper_" << i;
  }
  stream << " ) {\n";

  // memeory and control pragma 
  for (int i = 0; i < args.size(); i++) {
    std::string interface = " m_axi ";
    PrintIndent(stream, indent);
    stream << "#pragma HLS INTERFACE" + interface + "port=";
    stream << "source_wrapper_" << i;
    stream << " offset=slave bundle=gmem\n";
  }
  for (int i = 0; i < args.size(); i++) {
    std::string interface = " s_axilite ";;
    PrintIndent(stream, indent);
    stream << "#pragma HLS INTERFACE" + interface + "port=";
    stream << "source_wrapper_" << i;
    stream << " bundle=control\n";
  }
  PrintIndent(stream, indent);
  stream << "#pragma HLS INTERFACE s_axilite port=return bundle=control\n";
  stream << "\n";

  // variable init memory alloc 
  for (int i = 0; i < args.size(); i++) {
    PrintIndent(stream, indent);
    stream << Type2WrapStr(arg_types[i]);
    stream << " source_wrapper_temp_" << i;
    // var shape & alloc size
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      auto shape = GetShape(arr);
      for (size_t j = 0; j < shape.size(); j++) 
        stream << "[" << shape[j] << "]";
    } else {
      stream << "[1]";
    }
    // if (shape.size() == 0) stream << "[1]";
    stream << ";\n";
  }

  // move data from shared memory to temp
  for (int i = 0; i < args.size(); i++) {
    std::vector<int> shape;
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      shape = GetShape(arr);
    } 

    for (size_t j = 0; j < shape.size(); j++) {
      PrintIndent(stream, indent);
      stream << "for (int i" << j << " = 0; ";
      stream << "i" << j << " < " << shape[j] << "; ";
      stream << "i" << j << "++) {\n";
      indent += 2;
      if (j == shape.size() - 1) {
        PrintIndent(stream, indent);
        stream << "source_wrapper_temp_" << i;
        for (size_t k = 0; k < shape.size(); k++) {
          stream << "[i" << k << "]";
        }
        stream << " = ";
        stream << "source_wrapper_" << i;
        stream << "[i" << shape.size() - 1;
        int mul = 1;
        for (size_t k = shape.size() - 1; k > 0; k--) {
          mul *= shape[k];
          stream << "+ i" << k - 1 << "*" << mul;
        }
        stream << "];\n";
      }
    }
    for (size_t j = 0; j < shape.size(); j++) {
      indent -= 2;
      PrintIndent(stream, indent);
      stream << "}\n";
    }
    if (shape.size() == 0) {
      PrintIndent(stream, indent);
      stream << "source_wrapper_temp_" << i;
      stream << "[0] = source_wrapper_" << i << "[0];\n";
    }
  }

  // print top func
  stream << "\n";
  PrintIndent(stream, indent);
  stream << "top_function_0(";
  for (int i = 0; i < args.size(); i++) {
    if (i != args.size() - 1){
      stream << "source_wrapper_temp_" << i;
      stream << ", ";
    } else {
      stream << "source_wrapper_temp_" << i;
      stream << ");\n";
    }

  }
  stream << "\n";

  // read back return val
  for (int k = args.size() - 1; 
       k > args.size() - 2; k--) {
    std::vector<int> shape;
    if (args[k].type_code() == kArrayHandle) {
      TVMArray* arr = args[k];
      shape = GetShape(arr);
    } 
    for (size_t i = 0; i < shape.size(); i++) {
      PrintIndent(stream, indent);
      stream << "for (int i" << i << " = 0; ";
      stream << "i" << i << " < " << shape[i] <<  "; ";
      stream << "i" << i << "++) {\n";
      indent += 2;
    
      if (i == shape.size() - 1) {
        PrintIndent(stream, indent);
        stream << "source_wrapper_" << k;
        stream << "[i" << shape.size() - 1;
        int mul = 1;
        for (size_t j = shape.size() - 1; j > 0; j--) {
          mul *= shape[j];
          stream << " + i" << j - 1 << "*" << mul;
        }
        stream << " ] = ";
    
        stream << "source_wrapper_temp_" << k;
        for (size_t j = 0; j < shape.size(); j++) {
          stream << "[i" << j << "]";
        }
        stream <<";\n";
      }
    }
    for (size_t i = 0; i < shape.size(); i++) {
        indent -= 2;
        PrintIndent(stream, indent);
        stream << "}\n";
    }
  }
  stream << "}\n";
  indent -= 2;
  stream << "}\n";
  stream.close();
}

// generate opencl wrapper for sdaccel sim
void GenHostHeaders(std::ofstream& stream,
                    std::string platform) {
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

)";
  
  if (platform == "sdaccel") {
    stream << "// opencl harness headers\n";
    stream << "#include \"CLWorld.h\"\n";
    stream << "#include \"CLKernel.h\"\n";
    stream << "#include \"CLMemObj.h\"\n";
    stream << "#include \"utils.h\"\n";
    stream << "#include <cmath>\n\n";
    stream << "// harness namespace\n";
    stream << "using namespace rosetta;\n\n";
  } else if (platform == "vivado_hls" || 
             platform == "vivado" || platform == "sdsoc") {
    if (platform == "sdsoc") 
      stream << "#include \"sds_lib.h\"\n";
    stream << "// vivado hls headers\n";
    stream << "#include <ap_int.h>\n";
    stream << "#include <ap_fixed.h>\n";
    stream << "#include <hls_stream.h>\n";
    stream << "#include \"kernel.h\"\n\n";
  }
}

// initialization before executing kernel 
void KernelInit(std::ofstream& stream,
                std::string platform,
                TVMArgs& args, 
                const std::vector<TVMType>& arg_types,
                std::vector<std::string> arg_names,
                int added_args_num) {
  int indent = 2;
  stream << R"(
  // parse command line arguments for opencl version 
  std::string kernelFile("");
  parse_sdaccel_command_line_args(argc, argv, kernelFile);

  // create OpenCL world
  CLWorld world = CLWorld(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);

  // add the bitstream file
  world.addProgram(kernelFile);
 
  // create kernels
  CLKernel App(world.getContext(), world.getProgram(), "top_function_0", world.getDevice());

)";

  PrintIndent(stream, indent);
  stream << "// create mem objects\n";
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      PrintIndent(stream, indent);
      stream << "CLMemObj source_" << i;
      stream << "((void*)" << arg_names[i];
      stream << ", sizeof(" << Type2Byte(arg_types[i]) << "), ";

      TVMArray* arr = args[i];
      for (int j = 0;j < arr->ndim;j++) {
        if (j==0) {
          stream << arr->shape[j] << " ";
        } else {
          stream << "* " << arr->shape[j];
        }
      }
      stream << ", ";
      stream << "CL_MEM_READ_WRITE);\n";
    }
  }

  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// add them to the world\n";
  for (int i = 0; i < args.size();i++) {
    if (args[i].type_code() == kArrayHandle) {
      PrintIndent(stream, indent);
      stream << "world.addMemObj(source_" << i;
      stream << ");\n";
    }
  }

  stream << R"(
  // set work size
  int global_size[3] = {1, 1, 1};
  int local_size[3] = {1, 1, 1};
  App.set_global(global_size);
  App.set_local(local_size);

  // add them to the world
  world.addKernel(App);

  // set kernel arguments
)";

  // TODO: push arg-mem setups to codegen  
  stream << R"(
  world.setIntKernelArg(0, 1, test_image);
  world.setMemKernelArg(0, 0, 0);
  world.setMemKernelArg(0, 2, 1);
)";

  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// run\n";
  PrintIndent(stream, indent);
  stream << "world.runKernels();\n\n";
  PrintIndent(stream, indent);
  stream << "// read the data back\n";
  PrintIndent(stream, indent);
  stream << "world.readMemObj(1);\n";
}

// separate host code into partitions 
std::vector<std::string> SplitHostCode(std::string host_code,
    std::vector<std::string>& names) {
  // extract the top arg name 
  size_t pos = host_code.find("default_function");
  auto func_ = host_code.substr(0, host_code.find('\n', pos)); 
  std::regex e(R"(\s(\w+?)(,|\)))");
  std::sregex_iterator iter(func_.begin(), func_.end(), e);
  std::sregex_iterator end;
  while(iter != end) {
    CHECK(iter->size() > 0) << "cannot find arg top";
    names.push_back((*iter)[1]);
    ++iter;
  }
  // separate the host code with delimiter  
  std::string delimiter = "top_function_";
  size_t func_pos = 0;
  std::vector<std::string> segments;
  host_code = host_code.substr(host_code.find('\n', pos) + 1);
  while ((func_pos = host_code.find(delimiter)) != std::string::npos) {
    auto seg = host_code.substr(0, func_pos);
    seg = seg.substr(seg.find_first_not_of(' '));
    segments.push_back(seg);
    host_code.erase(0, host_code.find(';', func_pos));
  }
  host_code = host_code.substr(host_code.find("\n"), host_code.rfind("}") - 1);
  segments.push_back(host_code);
  return segments;
}

// generate host code according to platform type
void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc lowered_func,std::string platform,
                 std::string host_code, argInfo& arg_info,
                 int added_args_num) {
  int indent = 0;
  std::ofstream stream;
  stream.open("__tmp__/host.cpp");
  GenHostHeaders(stream, platform);
  std::vector<std::string> arg_names;
  auto code = SplitHostCode(host_code, arg_names); 
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
      stream << "shmat(" << shmids[i] << ", nullptr, 0);\n";
      PrintIndent(stream, indent);

      stream << Type2Byte(arg_types[i]) << " ";
      stream << arg_names[i];
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

  // generate host side (before kernel)
  PrintIndent(stream, indent);
  stream << "// compute before kernel function\n";

  if (code.size() > 1) {
    PrintIndent(stream, indent);
    if (platform == "sdaccel") {
      // create variable wrapper
      stream << code[0] << "\n";
      KernelInit(stream, platform, args,
                 arg_types, arg_names, added_args_num);

    } else if (platform == "vivado_hls" || platform == "vivado" ||
               platform == "sdsoc") {
      // init hls stream channels 
      for (size_t k = 0; k < arg_info.size(); k++) {
        auto& info = arg_info[k]; 
        PrintIndent(stream, indent);
        // use hls::stream for pure vhls simulation
        if (platform != "sdsoc" && info.streamed) {
          stream << "hls::stream<" 
                 << Type2Str(Type2TVMType(info.type))
                 << "> " << "fd_" << info.name << ";\n";
        } else { // use sdsoc_alloc
          std::string size = "sizeof(" + 
              Type2Str(Type2TVMType(info.type)) + ")*";
          for (auto v : info.shape)
            size += std::to_string(v) + "*";
          size = size.substr(0, size.size()-1);
          stream << Type2Str(Type2TVMType(info.type)) << "* " << "fd_" 
                 << info.name << " = (" << Type2Str(Type2TVMType(info.type)) << " *)" 
                 << "sds_alloc(" << size << ")" << ";\n";
        }
      }
      PrintIndent(stream, indent);
      stream << code[0] << "\n";

      // create kernel call from host 
      PrintIndent(stream, indent);
      stream << "top(";
      for (size_t i = 0; i < arg_info.size(); i++) {
        auto& info = arg_info[i];
        auto shape = info.shape;
        if (i != 0) stream << ", ";
        if (platform != "sdsoc" && 
            shape.size() == 1 && shape[0] == 1) void(0);
        else stream << "fd_"; // pass in continuous mem ptr 
        stream << info.name;
      }
      stream << ");\n";
    }

    // generate host (post-kernel)
    PrintIndent(stream, indent);
    stream << "// compute after kernel function\n";

    // alloc buffers for host undefined
    for (int k = 0; k < added_args_num; k++) {
      auto size = arg_info.size() - 1;
      auto& info = arg_info[size-k];
      PrintIndent(stream, indent);
      stream << Type2Str(Type2TVMType(info.type)) << " "
             << info.name << "[";
      int mul = 1;
      for (size_t j = 0; j < info.shape.size(); j++) 
        mul *= info.shape[j];
      stream << mul << "];\n";
    }
    stream << code[1];

  } else { // without 
    stream << host_code;
  }

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
