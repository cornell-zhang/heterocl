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

inline std::string PrintHalideType(Type t) {
  std::string str = "";
  if (t.is_uint() || t.is_int() || t.is_fixed() || t.is_ufixed()) {
    if (t.is_uint())        str += "ap_uint<" + std::to_string(t.bits()) + ">";
    else if (t.is_int())    str += "ap_int<" + std::to_string(t.bits()) + ">";
    else if (t.is_ufixed()) str += "ap_ufixed<" + std::to_string(t.bits()) + ", " + std::to_string(t.bits() - t.fracs()) + ">";
    else                    str += "ap_fixed<" + std::to_string(t.bits()) + ", " + std::to_string(t.bits() - t.fracs()) + ">";
  } else {
    LOG(FATAL) << "Cannot convert type " << t << " to C type";
  }
  return str;
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
    } else {
      shmids.push_back(0);
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
               argInfo& arg_info,
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
      stream << std::get<0>(arg_info[nth_arr]);
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
                   argInfo& arg_info,
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
      stream << ")(" << std::get<0>(arg_info[nth_arr]);
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

void GenKernelCode(std::string test_file) {
  std::ofstream stream;
  stream.open("__tmp__/kernel.cpp");
  stream << test_file;
  stream.close();
}

// interface pragma to specify mem and ctrl interface in sdx
void GenWrapperCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 argInfo& arg_stream_types,
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

  // wrapper func interface
  stream << "void App( ";
  size_t ex_arg_count = 0;
  ex_arg_count = arg_stream_types.size() - arg_types.size();
  for (size_t i = 0; i < arg_types.size(); i++) {
    if (i != 0) stream << ", ";
    stream << Type2WrapStr(arg_types[i]);
    stream << "*";
    stream << " source_wrapper_" << i;
  }
  for (size_t k = 0; k < ex_arg_count; k++) {
    if (k != ex_arg_count) stream << ", ";
    stream << PrintHalideType(std::get<2>(arg_stream_types[k + arg_types.size()])); 
    stream << "*";
    stream << " source_wrapper_" << k + arg_types.size();
  }  
  stream << " ) {\n";

  // memeory and control pragma 
  for (size_t i = 0; i < arg_stream_types.size(); i++) {
    std::string interface;
    if (std::get<1>(arg_stream_types[i])) interface = " m_axi ";
    else interface = " m_axi ";
    PrintIndent(stream, indent);
    stream << "#pragma HLS INTERFACE" + interface + "port=";
    stream << "source_wrapper_" << i;
    stream << " offset=slave bundle=gmem\n";
  }
  for (size_t i = 0; i < arg_stream_types.size(); i++) {
    std::string interface;
    if (std::get<1>(arg_stream_types[i])) interface = " s_axilite ";
    else interface = " s_axilite ";
    PrintIndent(stream, indent);
    stream << "#pragma HLS INTERFACE" + interface + "port=";
    stream << "source_wrapper_" << i;
    stream << " bundle=control\n";
  }
  PrintIndent(stream, indent);
  stream << "#pragma HLS INTERFACE s_axilite port=return bundle=control\n";
  stream << "\n";

  // intermediate vars init alloc 
  for (size_t i = 0; i < arg_stream_types.size(); i++) {
    PrintIndent(stream, indent);
    stream << PrintHalideType(std::get<2>(arg_stream_types[i]));
    stream << " source_wrapper_temp_" << i;
    auto shape = std::get<3>(arg_stream_types[i]);
    for (size_t j = 0; j < shape.size(); j++) 
      stream << "[" << shape[j] << "]";
    if (shape.size() == 0) stream << "[1]";
    stream << ";\n";
  }

  // vars init for values
  for (size_t i = 0; i < arg_stream_types.size(); i++) {
    auto shape = std::get<3>(arg_stream_types[i]);
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
  stream << "top( ";
  for (size_t i = 0;i < arg_stream_types.size(); i++) {
    if (i != arg_stream_types.size() - 1){
      stream << "source_wrapper_temp_" << i;
      stream << ", ";
    } else {
      stream << "source_wrapper_temp_" << i;
      stream << ");\n";
    }

  }
  stream << "\n";

  // read back return val
  for (int k = arg_stream_types.size() - 1; 
       k > args.size() - 2; k--) {
    auto shape = std::get<3>(arg_stream_types[k]);
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
    for (size_t i = 0;i < shape.size(); i++) {
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
  stream << "#include <sys/ipc.h>\n";
  stream << "#include <sys/shm.h>\n\n";
  stream << "// standard C/C++ headers\n";
  stream << "#include <cstdio>\n";
  stream << "#include <cstdlib>\n";
  stream << "#include <getopt.h>\n";
  stream << "#include <string>\n";
  stream << "#include <time.h>\n";
  stream << "#include <sys/time.h>\n\n";
  
  if (platform == "sdaccel") {
    stream << "// opencl harness headers\n";
    stream << "#include \"CLWorld.h\"\n";
    stream << "#include \"CLKernel.h\"\n";
    stream << "#include \"CLMemObj.h\"\n";
    stream << "#include \"utils.h\"\n";
    stream << "// harness namespace\n";
    stream << "using namespace rosetta;\n";
  } else if (platform == "vivado_hls") {
    stream << "// vivado hls headers\n";
    stream << "#include <ap_int.h>\n";
    stream << "#include <ap_fixed.h>\n";
    stream << "#include <hls_stream.h>\n";
    stream << "#include \"kernel.cpp\"\n\n";
  }
}

// initialization before executing kernel 
void KernelInit(std::ofstream& stream,
                std::string platform,
                TVMArgs& args, 
                const std::vector<TVMType>& arg_types,
                argInfo& arg_stream_types) {
  int indent = 2;
  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// parse command line arguments for opencl version\n";
  PrintIndent(stream, indent);
  stream << "std::string kernelFile(\"\");\n";
  PrintIndent(stream, indent);
  stream << "parse_sdaccel_command_line_args(argc, argv, kernelFile);\n";
  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// create OpenCL world\n";
  PrintIndent(stream, indent);
  stream << "CLWorld world = CLWorld(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);\n";
  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// add the bitstream file\n";
  PrintIndent(stream, indent);
  stream << "dworld.addProgram(kernelFile);\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// create kernels\n";
  PrintIndent(stream, indent);
  stream << "CLKernel App(world.getContext(), world.getProgram(), \"App\", world.getDevice());\n";
  stream << "\n\n";

  PrintIndent(stream, indent);
  stream << "// create mem objects\n";
  for (int i = 0;i < args.size(); i++) {
    PrintIndent(stream, indent);
    stream << "CLMemObj source_" << i;
    stream << "((void*)arg_top_" << i;
    stream << ", sizeof(" << Type2Byte(arg_types[i]) << "), ";

    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      for (int j = 0;j < arr->ndim;j++) {
        if (j==0) {
          stream << arr->shape[j] << " ";
        } else {
          stream << "* " << arr->shape[j];
        }
      }
    } else {
      stream << "1";
    }
    stream << ", ";
    stream << "CL_MEM_READ_WRITE);\n";
  }
  // additional streamed data
  for (size_t k = args.size(); k < arg_stream_types.size(); k++) {
    auto type = std::get<2>(arg_stream_types[k]);
    auto shape = std::get<3>(arg_stream_types[k]);
    PrintIndent(stream, indent);
    stream << "CLMemObj source_" << k;
    stream << "((void*)knn_mat";
    stream << ", sizeof(" << Type2Byte(Type2TVMType(type)) << "), ";
    if (shape.size() > 0) {
      for (size_t j = 0; j < shape.size(); j++) {
        if (j == 0) {
          stream << shape[j] << " ";
        } else {
          stream << "* " << shape[j];
        }
      }
    } else {
      stream << "1";
    }
    stream << ", ";
    stream << "CL_MEM_READ_WRITE);\n";
  }

  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// add them to the world\n";
  for (size_t i = 0;i < arg_stream_types.size();i++) {
    PrintIndent(stream, indent);
    stream << "world.addMemObj(source_" << i;
    stream << ");\n";
  }

  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << " // set work size\n";
  PrintIndent(stream, indent);
  int size = arg_stream_types.size();
  std::string arr = "[" + std::to_string(size) + "] = {";
  for (int i = 0; i < size; i++) {
    if (i != size -1) arr += "1, ";
    else arr += "1};\n";
  }
  stream << "int global_size" + arr;
  PrintIndent(stream, indent);
  stream << "int local_size" + arr;
  PrintIndent(stream, indent);
  stream << "App.set_global(global_size);\n";
  PrintIndent(stream, indent);
  stream << "App.set_local(local_size);\n";
  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// add them to the world\n";
  PrintIndent(stream, indent);
  stream << "world.addKernel(App);\n";
  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// set kernel arguments\n";
  for (size_t i = 0; i < arg_stream_types.size(); i++) {
    PrintIndent(stream, indent);
    stream << "world.setMemKernelArg(0, "<< i << ", " << i;
    stream << ");\n";
  }

  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// run\n";
  PrintIndent(stream, indent);
  stream << "world.runKernels();\n\n";
  PrintIndent(stream, indent);
  stream << "// read the data back\n";
  for (size_t i = args.size() - 1; i < arg_stream_types.size(); i++) {
    PrintIndent(stream, indent);
    stream << "world.readMemObj(" << i << ");\n";
  }
}

// generate host code according to platform type
void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc lowered_func,
                 std::string platform,
                 std::string host_code,
                 argInfo& arg_info) {
  int indent = 0;
  std::ofstream stream;
  stream.open("__tmp__/host.cpp");
  GenHostHeaders(stream, platform);

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
      stream << std::get<0>(arg_info[i]);
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
      PrintCopy(arr, arg_info, stream, indent, i);

    } else {
      // directly assign the value to the variable
      PrintIndent(stream, indent);
      stream << Type2Byte(arg_types[i]) << " ";
      stream << "arg_" << i << " = ";
      stream << "(" << Type2Byte(arg_types[i]) << ")";
      if (args[i].type_code() == kDLInt || 
          args[i].type_code() == kDLUInt) {
        stream << int64_t(args[i]);
      }
      stream << ";\n";
      PrintIndent(stream, indent);
      stream << Type2Byte(arg_types[i]) << " ";
      stream << std::get<0>(arg_info[i]);
      stream << "[1] = { ";

      stream << "arg_" << i << " }";
      if (arg_types[i].fracs > 0)
        stream << " >> " << static_cast<int>(arg_types[i].fracs);
      stream << ";\n";
      cnt += 1;
    }
    stream << "\n";
  }

  // allocate mem for stream vars
  for (size_t k = args.size(); k < arg_info.size(); k++) {
    auto type = std::get<2>(arg_info[k]);
    auto shape = std::get<3>(arg_info[k]);
    PrintIndent(stream, indent);
    stream << Type2Byte(Type2TVMType(type)) << " " << "name[";
    if (shape.size() > 0) {
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != shape.size() - 1)
          stream << shape[i] << " * ";
        else stream << shape[i];
      }
    } else {
      stream << "1";
    }
    stream << "];\n";
  }

  // generate host side (before kernel)
  PrintIndent(stream, indent);
  stream << "printf(\"Finished setting up shared memory\\n\");\n";
  PrintIndent(stream, indent);
  stream << "// compute bofore kernel function\n";
  size_t pos = host_code.find("top(");
  std::string pre_kernel  = host_code.substr(0, pos -1);
  std::string post_kernel = host_code.substr(host_code.find('\n', pos) + 1);
  pre_kernel = pre_kernel.substr(pre_kernel.find_first_not_of("\n"));
  pre_kernel = pre_kernel.substr(pre_kernel.find_first_not_of(" "));
  PrintIndent(stream, indent);
  
  if (platform == "sdaccel") {
    // create variable wrapper
    stream << pre_kernel << "\n";
    KernelInit(stream, platform, args,
               arg_types, arg_info);
  } else if (platform == "vivado_hls") {
    // init hls stream channels 
    for (size_t k = 0; k < arg_info.size(); k++) {
      auto info = arg_info[k]; 
      if (std::get<1>(info)) {
        PrintIndent(stream, indent);
        stream << "hls::stream<" 
               << PrintHalideType(std::get<2>(info)) 
               << "> " << "fd_" << std::get<0>(info) << ";\n";
      }  
    }
    PrintIndent(stream, indent);
    stream << pre_kernel << "\n";
    PrintIndent(stream, indent);
    // create kernel call from host 
    stream << "top(";
    for (size_t i = 0; i < arg_info.size(); i++) {
      auto info = arg_info[i];
      auto name = std::get<0>(info);
      auto shape = std::get<3>(info);
      if (i != 0) stream << ", ";
      if (shape.size() == 1 && shape[0] == 1) void(0);
      else stream << "fd_"; 
      stream << name;
    }
    stream << ");\n";
  }

  // generate host (post-kernel)
  PrintIndent(stream, indent);
  stream << "// compute after kernel function\n";
  stream << post_kernel;

  // copy to shared mem
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      PrintCopyBack(arr, arg_info, stream, indent, i);
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
