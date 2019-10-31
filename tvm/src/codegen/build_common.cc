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

#include <fstream>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <iostream>

#include "merlinc/codeanalys_merlinc.h"
#include "hlsc/codegen_vhls.h"
#include "opencl/codegen_aocl.h"

namespace TVM {
namespace runtime {

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
      // stream << "arg_top_" << nth_arr;
      // for (int j = 0; j < arr->ndim; j++) {
      //   stream << "[i" << j << "]"; 
      // }

      stream << "arg_top_" << nth_arr;
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
      // stream << Type2ExtStr(arr->dtype);
      stream << Type2Byte(arr->dtype);
      stream << ")(arg_top_" << nth_arr;
      stream << "[i" << arr->ndim-1;
      int mul2 = 1;
      for (int j = arr->ndim-2; j >= 0; j--) {
        mul2 *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul2;
      }

      stream << "])";

      // for (int j = 0; j < arr->ndim; j++) {
      //   stream << "[i" << j << "]"; 
      // }
      // stream << ")";
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
  // stream.open("/home/centos/src/project_data/lab_digitrec_aws/solution/src/kernel/knn_vhls.cpp");
  stream.open("__tmp__/kernel.cpp");
  stream << test_file;
  stream.close();
}

// interface pragma to specify mem and ctrl interface in sdx
void GenWrapperCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 const std::vector<std::tuple<bool, Type, std::vector<int>>>& arg_stream_types,
                 LoweredFunc func) {
  std::ofstream stream;
  // stream.open("/home/centos/src/project_data/lab_digitrec_aws/solution/src/kernel/digitrec.cpp");
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
    stream << PrintHalideType(std::get<1>(arg_stream_types[k + arg_types.size()])); 
    stream << "*";
    stream << " source_wrapper_" << k + arg_types.size();
  }  
  stream << " ) {\n";

  // memeory and control pragma 
  for (size_t i = 0; i < arg_stream_types.size(); i++) {
    std::string interface;
    if (std::get<0>(arg_stream_types[i])) interface = " m_axi ";
    else interface = " m_axi ";
    PrintIndent(stream, indent);
    stream << "#pragma HLS INTERFACE" + interface + "port=";
    stream << "source_wrapper_" << i;
    stream << " offset=slave bundle=gmem\n";
  }
  for (size_t i = 0; i < arg_stream_types.size(); i++) {
    std::string interface;
    if (std::get<0>(arg_stream_types[i])) interface = " s_axilite ";
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
    stream << PrintHalideType(std::get<1>(arg_stream_types[i]));
    stream << " source_wrapper_temp_" << i;
    auto shape = std::get<2>(arg_stream_types[i]);
    for (size_t j = 0; j < shape.size(); j++) 
      stream << "[" << shape[j] << "]";
    if (shape.size() == 0) stream << "[1]";
    stream << ";\n";
  }

  for (size_t i = 0; i < arg_stream_types.size(); i++) {
    auto shape = std::get<2>(arg_stream_types[i]);
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
        stream << "source_wrapper_" << j;
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
    auto shape = std::get<2>(arg_stream_types[k]);
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

// generate opencl kernel and mem obj
void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc func,
                 std::string pre_kernel,
                 std::string post_kernel,
                 std::vector<std::tuple<bool, Type, std::vector<int>>>& arg_stream_types) {
  int indent = 0;
  std::ofstream stream;
  stream.open("__tmp__/host.cpp");
  // stream.open("/home/centos/src/project_data/lab_digitrec_aws/solution/src/host/digit_recognition.cpp");
  stream << "#include <sys/ipc.h>\n";
  stream << "#include <sys/shm.h>\n";
  stream << "\n";
  stream << "// standard C/C++ headers\n";
  stream << "#include <cstdio>\n";
  stream << "#include <cstdlib>\n";
  stream << "#include <getopt.h>\n";
  stream << "#include <string>\n";
  stream << "#include <time.h>\n";
  stream << "#include <sys/time.h>\n";
  stream << "\n";
  stream << "// opencl harness headers\n";
  stream << "#include \"CLWorld.h\"\n";
  stream << "#include \"CLKernel.h\"\n";
  stream << "#include \"CLMemObj.h\"\n";
  stream << "// harness namespace\n";
  stream << "using namespace rosetta;\n";
  stream << "\n";
  stream << "//other headers\n";
  stream << "#include \"utils.h\"\n";
  // stream << "#include \"typedefs.h\"\n";
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
      // stream << Type2Str(arg_types[i]) << " ";
      stream << "arg_top_" << i;
      TVMArray* arr = args[i];

      stream << "[";
      for (int j = 0; j < arr->ndim; j++) {
        //stream << "[" << arr->shape[j] << "]";
        if (j == arr->ndim-1) {
          stream << arr->shape[j];
        } else {
          stream << arr->shape[j];
          stream << " * ";
        }
      }
      stream << "];\n";
      // copy from shared mem
      PrintCopy(arr, stream, indent, i);

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
      stream << "arg_top_" << i;
      stream << "[1] = { ";

      stream << "arg_" << i << " }";
      if (arg_types[i].fracs > 0)
        stream << " >> " << static_cast<int>(arg_types[i].fracs);
      stream << ";\n";

      // PrintIndent(stream, indent);
      // stream << Type2Byte(arg_types[i]) << " ";
      // stream << "fool_" << cnt << "[1] = { arg_top_" << i << " };\n";
      cnt += 1;
    }
    stream << "\n";
  }
  // allocate mem for stream vars
  for (size_t k = args.size(); k < arg_stream_types.size(); k++) {
    auto type = std::get<1>(arg_stream_types[k]);
    auto shape = std::get<2>(arg_stream_types[k]);
    PrintIndent(stream, indent);
    stream << Type2Byte(Type2TVMType(type)) << " " << "knn_mat[";
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

  // generate host side (before) on arg_top_k
  PrintIndent(stream,indent);
  stream << "printf(\"Host Side Application\\n\");\n";
  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// compute bofore kernel function";
  // stream being axis interface host, channel for kernel 
  stream << pre_kernel;

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
  stream << "CLWorld digit_rec_world = CLWorld(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);\n";
  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// add the bitstream file\n";
  PrintIndent(stream, indent);
  stream << "digit_rec_world.addProgram(kernelFile);\n";
  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "// create kernels\n";
  PrintIndent(stream, indent);
  stream << "CLKernel App(digit_rec_world.getContext(), digit_rec_world.getProgram(), \"App\", digit_rec_world.getDevice());\n";
  stream << "\n\n";

  PrintIndent(stream, indent);
  stream << "// create mem objects\n";
  for (int i = 0;i < args.size();i++) {
    PrintIndent(stream, indent);
    // if (cnt!=0) {
    //   stream << "CLMemObj source_" << i;
    //   stream << "((void*)fool_" << cnt - 1;
    //   stream << ", sizeof(" << Type2Byte(arg_types[i]) << "), ";
    //   stream << "1, ";
    //   stream << "CL_MEM_READ_WRITE);\n";
    //   cnt--;
    //   continue;
    // }
    stream << "CLMemObj source_" << i;
    stream << "((void*)arg_top_" << i;
    stream << ", sizeof(" << Type2Byte(arg_types[i]) << "), ";
    // stream << ", sizeof(" << Type2ExtStr(arg_types[i]) << "), ";

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
  // addiion streamed data
  for (size_t k = args.size(); k < arg_stream_types.size(); k++) {
    auto type = std::get<1>(arg_stream_types[k]);
    auto shape = std::get<2>(arg_stream_types[k]);
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
    stream << "digit_rec_world.addMemObj(source_" << i;
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
  stream << "digit_rec_world.addKernel(App);\n";
  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// set kernel arguments\n";
  // PrintIndent(stream, indent);
  // stream << "digit_rec_world.setConstKernelArg(0, 0, arg_top_0);\n";
  for (size_t i = 0;i < arg_stream_types.size();i++) {
    PrintIndent(stream, indent);
    stream << "digit_rec_world.setMemKernelArg(0, "<< i << ", " << i;
    stream << ");\n";
  }

  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// run\n";
  PrintIndent(stream, indent);
  stream << "digit_rec_world.runKernels();\n\n";
  PrintIndent(stream, indent);
  stream << "// read the data back\n";
  PrintIndent(stream, indent);
  stream << "digit_rec_world.readMemObj(2);\n";

  // generate host (post-kernel)
  stream << "\n";
  PrintIndent(stream, indent);
  stream << "// compute after kernel function\n";
  // stream being axis interface host, channel for kernel 
  stream << post_kernel;

  // copy to shared mem
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      PrintCopyBack(arr, stream, indent, i);
      // PrintCopyBack2(arr, stream, indent, i);
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

class SimModuleNode final : public ModuleNode {
 public:
  SimModuleNode(LoweredFunc func, 
                std::string pre_host_code,
                std::string post_host_code,
                std::vector<std::tuple<bool, Type, std::vector<int>>> arg_stream_types,
                std::string dev_code) 
    : func_(func), 
      pre_host_(pre_host_code), 
      post_host_(post_host_code), 
      arg_stream_types_(arg_stream_types),
      dev_(dev_code) { 
  }

  const char* type_key() const {
    return "unified_sim";
  }

  // unified simulation function
  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv){
        if (args.size() != (int)func_->args.size())
          LOG(FATAL) << "The function should take in " << func_->args.size() 
                     << " inputs but get " << args.size();
        std::vector<int> shmids;
        std::vector<size_t> arg_sizes;
        std::vector<TVMType> arg_types;

        CollectArgInfo(args, func_, arg_sizes, arg_types);
        GenSharedMem(args, shmids, arg_sizes);

        LOG(CLEAN) << "Generating harness files ...";
        system("rm -rf __tmp__; mkdir __tmp__");
        // generate interface wrapper for kernel args 
        GenWrapperCode(args, shmids, arg_types, arg_stream_types_, func_);
        // host code invoking extern c wrapped hlsc kernel 
        GenHostCode(args, shmids, arg_types, func_, 
                    pre_host_, post_host_, arg_stream_types_);
        GenKernelCode(dev_);
        std::string path; 
        if (const auto* f = Registry::Get("get_util_path")) 
          path = (*f)("aws_f1").operator std::string();
        system(("cp " + path + "/* __tmp__/").c_str());

        LOG(CLEAN) << "Running SW simulation ...";
        system("cd __tmp__; source ./run_sw.sh");
        LOG(CLEAN) << "Finished C simulation";
        FreeSharedMem(args, shmids, arg_sizes);
        // extract resource information
        if (const auto* f = Registry::Get("tvm_callback_syn_postproc")) {
          std::string code;
          code = (*f)("test").operator std::string();
          LOG(CLEAN) << "extract res info";
        }
      });
  }

 private:
  LoweredFunc func_;
  std::string pre_host_;
  std::string post_host_;
  std::vector<std::tuple<bool, Type, std::vector<int>>> arg_stream_types_;
  std::string dev_;
};

using var2nameType = std::unordered_map<const Variable*, 
    std::tuple<std::string, Type, std::vector<int>>>; 

Module CreateSimModule(
    LoweredFunc func,
    std::string pre_host_code,
    std::string post_host_code,
    std::vector<const Variable*>& arg_vars,
    std::unordered_map<const Variable*, bool>& stream_table,
    var2nameType& arg_top_vars,
    std::string dev_code) {
    // process info: shape type and stream 
    std::vector<std::tuple<bool, Type, std::vector<int>>> arg_type;
    for (size_t i = 0 ; i < arg_vars.size(); i++) {
      auto v = arg_vars[i];
      auto nameType = arg_top_vars[v];
      bool is_stream;
      if (stream_table[v])
        is_stream = true;
      else is_stream = false;
      auto item = std::make_tuple(is_stream, std::get<1>(nameType), 
                                  std::get<2>(nameType));
      arg_type.push_back(item);
    }
  std::shared_ptr<SimModuleNode> n =
    std::make_shared<SimModuleNode>(func, pre_host_code, post_host_code, 
                                    arg_type, dev_code);
  return Module(n);
}
} // namespace runtime

namespace codegen {
using var2nameType = std::unordered_map<const Variable*, 
    std::tuple<std::string, Type, std::vector<int>>>; 

// collect type info for vars
class TypeCollector final : public IRVisitor {
  public:
    var2nameType& top_args_;
    TypeCollector(var2nameType& top_args)
      : top_args_(top_args) {}
    void Visit_(const Allocate *op) {
      auto v = op->buffer_var.get();
      
      // record type and shape
      if (top_args_.count(v)) {
        std::vector<int> shape;
        for (size_t i = 0; i < op->extents.size(); i++) 
          shape.push_back(op->extents[i].as<IntImm>()->value);
        top_args_[v] = std::make_tuple(
                           std::get<0>(top_args_[v]),
                           op->type, shape);
      }
      IRVisitor::Visit_(op);
    }
};

// record <name, type> of vars for top func signature
// vars include passed-in and not registered vars on host
class StreamCollector final : public IRVisitor {
  public:
    StreamCollector(std::vector<const Variable*>& arg_vars,
                    std::unordered_map<const Variable*, bool>& stream_table,
                    std::string initial_scope)
      : arg_vars_(arg_vars),
        stream_table_(stream_table),
        scope_(initial_scope) {}

    // record alloc on host 
    void Visit_(const Allocate *op) {
      if (!switch_on) 
        this->HandleDef(op->buffer_var.get());
      IRVisitor::Visit_(op);
    }
    
    void Visit_(const Load *op) {
      if (!switch_on) {
        this->HandleUse(op->buffer_var);
      }
      IRVisitor::Visit_(op);
    }

    // update placeholder status
    void Visit_(const Store* op) {
      if (switch_on) {
        if (auto val = op->value.as<StreamExpr>()) {
          const Variable* v = val->buffer_var.get();
          for (size_t i = 0; i < arg_vars_.size(); i++) {
            std::string name = arg_vars_[i]->name_hint;
            if (v->name_hint.find(name) != std::string::npos) {
              // record in VisitStmt StreamStmt
              // LOG(WARNING) << op->buffer_var << ":" << v->name_hint;
            }
          }
        }
      } else { // count use on host
        this->HandleUse(op->buffer_var);
      }
      IRVisitor::Visit_(op);
    }

    void Visit_(const StreamStmt* op) {
      if (switch_on) { // in xcel scope
        const Variable* v = op->buffer_var.get();
        // LOG(WARNING) << v->name_hint;  
      }
      IRVisitor::Visit_(op);
    }

    void Visit_(const AttrStmt* op) {
      if (op->attr_key == attr::device_scope) { 
        if (op->value.as<StringImm>()->value != scope_)
          switch_on = true;
        else switch_on = false;
      }
      IRVisitor::Visit_(op);
    }

    // additional data saved into stream table (for streamed 
    // data we keep the new id for arg_stream in var_idmap, 
    // and non-streamed using the repalced arg_top_k name)
    void HandleDef(const Variable* v) {
      CHECK(!host_def_count_.count(v))
          << "variable " << v->name_hint
          << " has already been defined, the Stmt is not SSA";
      CHECK(!host_use_count_.count(v))
          << "variable " << v->name_hint
          << " has been used before definition!";
      host_use_count_[v] = 0;
      host_def_count_[v] = 1;
    }

    void HandleUse(const Expr& v) {
      CHECK(v.as<Variable>());
      Var var(v.node_);
      auto it = host_use_count_.find(var.get());
      if (it != host_use_count_.end()) {
        if (it->second >= 0) {
          ++it->second;
        }
      } else {
        if (!stream_table_.count(var.get())) {
          host_undefined_.push_back(var);
          host_use_count_[var.get()] = -1;
        }
      }
    }

    bool host_scope_{false};
    Array<Var> host_undefined_;
    std::unordered_map<const Variable*, int> host_use_count_;
    std::unordered_map<const Variable*, int> host_def_count_;

  private:
    std::vector<const Variable*>& arg_vars_;
    std::unordered_map<const Variable*, bool>& stream_table_;
    std::string scope_;
    bool switch_on{true};
};

// codegen for accelerators 
class CodeGenXcel : public CodeGenVivadoHLS {
  public:
    int arg_top_count{0};
    std::string pre_kernel;
    std::string post_kernel;
    // map for generating wrapper
    var2nameType arg_top_vars;
    std::vector<const Variable*> arg_vars;
    std::unordered_map<const Variable*, bool> stream_table;
    str2tupleMap<std::string, Type> map_arg_type_;
    LoweredFunc f_;

  void AddFunction(LoweredFunc f,
           str2tupleMap<std::string, Type> map_arg_type) {
    map_arg_type_ = map_arg_type; f_ = f;
    CodeGenVivadoHLS::AddFunction(f, map_arg_type);
  };

  void VisitStmt_(const AttrStmt* op) {
     if (op->attr_key == ir::attr::device_scope) {
      // print top( ... in host and enter fpga scope 
      if (op->value.as<StringImm>()->value == "fpga" && !fpga_scope_) {
        fpga_scope_ = true;
        PrintIndent();
         
        // track the stream usage
        StreamCollector collector(arg_vars, stream_table, "cpu");
        collector.Visit(op->body);

        // update data type and name 
        for (auto k : collector.host_undefined_) {
          auto v = k.get();
          arg_vars.push_back(v);
          stream_table[v] = true;
          auto tuple = arg_top_vars[v];
          arg_top_vars[v] = std::make_tuple(v->name_hint,
                                            std::get<1>(tuple),
                                            std::get<2>(tuple)); 
        }
        TypeCollector visitor(arg_top_vars);
        visitor.Visit(op->body);
  
        // generte function calls 
        stream << "top(";
        int index = 0;
        for (size_t i = 0; i < arg_vars.size(); i++) {
          auto v = arg_vars[i];
          std::string arg_name;
          if (stream_table[v]) 
            arg_name = std::get<0>(arg_top_vars[v]);
          else arg_name = GetVarID(v); 
          if (index !=0) stream << ", ";
          stream << arg_name;
          // print kernel func signature
          if (index !=0) arg_stream << ", ";
          PrintType(std::get<1>(arg_top_vars[v]), arg_stream);
          auto shape = std::get<2>(arg_top_vars[v]);
          arg_stream << " " << arg_name;
          for (size_t k = 0; k < shape.size(); k++)
            arg_stream << "[" << shape[k] << "]";
          index++;
        }
        stream << ");\n";
  
        // switch context to device scope
        host_stream << this->stream.str();
        this->stream.str("");
        this->stream.clear();
  
      // swtich from device to host
      } else if (op->value.as<StringImm>()->value == "cpu" && 
                 fpga_scope_) {
        fpga_scope_ = false;
        device_stream << this->stream.str();
        this->stream.str("");
        this->stream.clear();
      }
    }
    CodeGenC::VisitStmt_(op);
  }
    void VisitStmt_(const Store* op) {
      std::string vid = GetVarID(op->buffer_var.get());
      if (vid.find("stream_") == std::string::npos)
        CodeGenVivadoHLS::VisitStmt_(op);
    };

    void VisitStmt_(const LetStmt* op) {
      std::string value = PrintExpr(op->value);
      // Skip the argument retrieving assign statement
      std::string vid = AllocVarID(op->var.get());
      if (op->var.type() != Handle() &&
          value.find("TVMArray") == std::string::npos &&
          value.find("arg") != 0) {
        PrintIndent();
        PrintType(op->var.type(), this->stream);
        this->stream << ' '
                     << vid
                     << " = " << value << ";\n";
      // modify var idmap for passed in args
      } else if (value.find("data") != std::string::npos ||
                 value.substr(0, 3) == "arg") {
        auto v = op->var.get();
        auto tuple = arg_top_vars[v]; 
        arg_vars.push_back(v);
        stream_table[v] = false;
        var_idmap_[v] = "arg_top_" + std::to_string(arg_top_count);
        std::string api_name = "arg" + std::to_string(arg_top_count);
        auto arg = map_arg_type_[api_name];
        // PrintType(std::get<1>(arg), arg_stream);
        std::vector<int> shape;
        if (auto buf = f_->api_args[arg_top_count].as<BufferNode>())
          for (size_t i = 0; i < buf->shape.size(); i++) 
            shape.push_back(buf->shape[i].as<IntImm>()->value);
        arg_top_vars[v] = std::make_tuple(vid, std::get<1>(arg), shape);
        arg_top_count += 1;
      }
      PrintStmt(op->body);
    };

    void VisitStmt_(const StreamStmt* op) {
      //TODO: fix this
      // std::string vid = GetVarID(op->buffer_var.get());
      std::string vid;
      if (!var_idmap_.count(op->buffer_var.get())) 
        vid = AllocVarID(op->buffer_var.get());
      else vid = GetVarID(op->buffer_var.get());
      PrintIndent();
      auto load_op = op->value.as<Load>(); 
      auto v = load_op->buffer_var.as<Variable>();
      // placeholder args using recv name 
      if (stream_table.count(v)) {
        auto tuple = arg_top_vars[v];
        vid.replace(vid.find("stream_send"), 12, "stream_recv");
        arg_top_vars[v] = std::make_tuple(vid, std::get<1>(tuple),
                                          std::get<2>(tuple));
        stream_table[v] = true;
      } // else: streamed externop defined in analysis
      // PrintExpr(op->value, stream);
      // stream << vid << ".write()\n";
    };

    void VisitStmt_(const Allocate* op) {
      std::string vid = AllocVarID(op->buffer_var.get());
      CHECK(!is_zero(op->condition));
      int32_t constant_size = op->constant_allocation_size();
      CHECK_GT(constant_size, 0)
          << "Can only handle constant size stack allocation for now";
      const Variable* buffer = op->buffer_var.as<Variable>();
      var_shape_map_[buffer] = op->extents;
      std::string scope = alloc_storage_scope_.at(buffer);
      PrintStorageScope(scope, stream);

      // initlize hls stream channel
      if (arg_top_vars.count(buffer) ||
          vid.find("stream_") != std::string::npos) { 
      } else {
        this->PrintIndent();
        PrintType(op->type, stream);
        stream << ' '<< vid;
        if (constant_size > 1) {// Transfer length one array to scalar
          for (size_t i = 0; i < op->extents.size(); i++) {
            stream << '[';
            PrintExpr(op->extents[i], stream);
            stream << "]";
          }
        }
        stream << ";\n";
      }
      buf_length_map_[buffer] = constant_size;
      RegisterHandleType(op->buffer_var.get(), op->type);
      for (size_t i = 0; i < op->attrs.size(); i++) {
        this->PrintStmt(op->attrs[i]);
      }
      this->PrintStmt(op->body);
    };
};

// replace host-device interface args with pragma 
class CodeGenHost : public CodeGenAOCL {
  public:
    int arg_top_count{0};
    std::string pre_kernel;
    std::string post_kernel;
    // map for generating wrapper
    std::vector<const Variable*> arg_vars;
    std::unordered_map<const Variable*, bool> stream_table;
    var2nameType arg_top_vars;

  void PrintType(Type t, std::ostream &os) {
    int lanes = t.lanes();
    
    if(t.is_handle())
    {
      os << "void*";return;
    }
    if(t==Bool())
    {
      os <<"bool"; return;
    }
    CHECK_EQ(lanes,1)
        << "do not yet support vector types";
    
    bool fail = false;
    if(t.is_float())
    {
      switch(t.bits())
      {
        case 16:
          os<<"half";
          // enable_fp16_ = true;
          break;
        case 32:
          os<<"float";
          break;
        case 64:
          os<< "double";
          // enable_fp64_ = true;
          break;
        default:
          fail = true;
          break;
      }
      if(!fail && lanes ==1)return;
      if(!fail&&(lanes >= 2 && lanes <=16))
      {
        os<<lanes; return;
      }
    }
    else if(t.is_uint()||t.is_int())
    {
      switch(t.bits())
      {
        case 8: os<< "char"; break;
        case 16: os<<"short"; break;
        case 32: 
          if(t.is_uint())
            os<<"u";
          os<<"int";
          break;
        case 64: os<<"long";break;
        default : fail = true;break;
      }
      if(!fail && lanes == 1)return;
      if(!fail && (lanes >=2 && lanes <= 16))
      {
        os<<lanes; return;
      }
      if(fail && lanes==1)
      {
        if(t.is_uint())
        {
          if (t.bits() > 64) {
            os << "uint" << "64" << "_t"; return;
          } else {
            std::string str;
            if      (t.bits() <= 8)  str = "8";
            else if (t.bits() <= 16) str = "16";
            else if (t.bits() <= 32) str = "32";
            else                   str = "64";
            os<< "uint"<<  str  <<"_t"; return;
          }
        }
        if(t.is_int())
        {
          if (t.bits() > 64) {
            os << "int" << "64" << "_t"; return;
          } else {
            std::string str;
            if      (t.bits() <= 8)  str = "8";
            else if (t.bits() <= 16) str = "16";
            else if (t.bits() <= 32) str = "32";
            else                   str = "64";
            os << "int" << str << "_t"; return;
          }
        }
      }
    }

    LOG(FATAL) << "Cannot convert type"<<t<<"to AOCL type";
  };

  void VisitStmt_(const AttrStmt* op) {
     if (op->attr_key == ir::attr::device_scope) {
      // print top( ... in host and enter fpga scope 
      if (op->value.as<StringImm>()->value == "fpga" && !fpga_scope_) {
        fpga_scope_ = true;
        PrintIndent();
        
        // track the stream usage
        var2nameType unreg_vars;
        StreamCollector collector(arg_vars, stream_table, "cpu");
        collector.Visit(op->body);
        // update data type and name 
        for (size_t k = 0; k < arg_vars.size(); k ++)
          arg_top_vars[arg_vars[k]]; 
        for (auto k : collector.host_undefined_) 
          arg_top_vars[k.get()];
        TypeCollector visitor(arg_top_vars);
        visitor.Visit(op->body);
  
        // generte function calls 
        stream << "top(";
        // int index = 0;
        // for (auto op : stream_stmts) {
        //   if (index !=0) stream << ", ";
        //   std::string vid;
        //   if (!var_idmap_.count(op->buffer_var.get())) 
        //     vid = AllocVarID(op->buffer_var.get());
        //   else vid = GetVarID(op->buffer_var.get());
        //   stream << vid;
        //   if (vid.find("stream_send") != std::string::npos || 
        //       vid.find("stream_recv") != std::string::npos) {
        //     if (index !=0) arg_stream << ", ";
        //     PrintType(op->buffer_var.type(), arg_stream);
        //     arg_stream << " " << vid;
        //   } 
        //   index++;
        // }
        // for (auto op : stream_exprs) {
        //   if (index !=0) stream << ", ";
        //   std::string vid;
        //   if (!var_idmap_.count(op->buffer_var.get())) 
        //     vid = AllocVarID(op->buffer_var.get());
        //   else vid = GetVarID(op->buffer_var.get());
        //   stream << vid;
        //   // stream << op->buffer_var.get()->name_hint;
        //   if (vid.find("stream_send") != std::string::npos || 
        //       vid.find("stream_recv") != std::string::npos) {
        //     if (index !=0) arg_stream << ", ";
        //     PrintType(op->buffer_var.type(), arg_stream);
        //     arg_stream << " " << vid;
        //   } 
        //   index++;
        // }
        stream << ");\n";
  
        // switch context to device scope
        host_stream << this->stream.str();
        this->stream.str("");
        this->stream.clear();
  
      // swtich from device to host
      } else if (op->value.as<StringImm>()->value == "cpu" && 
                 fpga_scope_) {
        fpga_scope_ = false;
        device_stream << this->stream.str();
        this->stream.str("");
        this->stream.clear();
      }
    }
    CodeGenC::VisitStmt_(op);
  }

    void VisitStmt_(const Allocate* op) {
      std::string vid = AllocVarID(op->buffer_var.get());
      if (vid.find("stream_") != std::string::npos) { 
        // do not print alloc stream 
        this->PrintStmt(op->body);
      } else {
        CHECK(!is_zero(op->condition));
        this->PrintIndent();
        int32_t constant_size = op->constant_allocation_size();
        CHECK_GT(constant_size, 0)
            << "Can only handle constant size stack allocation for now";
        const Variable* buffer = op->buffer_var.as<Variable>();
        var_shape_map_[buffer] = op->extents;
        std::string scope = alloc_storage_scope_.at(buffer);
        PrintStorageScope(scope, stream);

        // initlize hls stream channel
        if (vid.find("stream_in") != std::string::npos || 
            vid.find("stream_out") != std::string::npos) {
          stream << "hls::stream<";
          PrintType(op->type, stream);
          stream << "> " << vid << ";\n";
        } else {
          PrintType(op->type, stream);
          stream << ' '<< vid;
          if (constant_size > 1) {// Transfer length one array to scalar
            for (size_t i = 0; i < op->extents.size(); i++) {
              stream << '[';
              PrintExpr(op->extents[i], stream);
              stream << "]";
            }
          }
          stream << ";\n";
        }
        buf_length_map_[buffer] = constant_size;
        RegisterHandleType(op->buffer_var.get(), op->type);
        for (size_t i = 0; i < op->attrs.size(); i++) {
          this->PrintStmt(op->attrs[i]);
        }
        this->PrintStmt(op->body);
      }
    };

    void VisitExpr_(const StreamExpr* op, std::ostream& os) {
      std::string vid;
      if (!var_idmap_.count(op->buffer_var.get())) 
        vid = AllocVarID(op->buffer_var.get());
      else vid = GetVarID(op->buffer_var.get());
      // os << vid << ".read()";
    };

    void VisitStmt_(const Store* op) {
      std::string vid = GetVarID(op->buffer_var.get());
      if (vid.find("stream_") == std::string::npos)
        CodeGenC::VisitStmt_(op);
    };

    void VisitStmt_(const StreamStmt* op) {
      std::string vid;
      if (!var_idmap_.count(op->buffer_var.get())) 
        vid = AllocVarID(op->buffer_var.get());
      else vid = GetVarID(op->buffer_var.get());
      PrintIndent();
      auto load_op = op->value.as<Load>(); 
      auto v = load_op->buffer_var.as<Variable>();
      // placeholder args using recv name 
      if (stream_table.count(v)) {
        auto tuple = arg_top_vars[v];
        arg_top_vars[v] = std::make_tuple(vid, std::get<1>(tuple),
                                          std::get<2>(tuple));
        stream_table[v] = true;
      } // else: streamed externop defined in analysis
      // PrintExpr(op->value, stream);
      // stream << vid << ".write()\n";
    };

    void VisitStmt_(const LetStmt* op) {
      std::string value = PrintExpr(op->value);
      // Skip the argument retrieving assign statement
      std::string vid = AllocVarID(op->var.get());
      if (op->var.type() != Handle() &&
          value.find("TVMArray") == std::string::npos &&
          value.find("arg") != 0) {
        PrintIndent();
        PrintType(op->var.type(), this->stream);
        this->stream << ' '
                     << vid
                     << " = " << value << ";\n";
      // locate arg data and update arg_top_vars
      } else if (value.find("data") != std::string::npos ||
                 value.substr(0, 3) == "arg") {
        auto v = op->var.get();
        auto tuple = arg_top_vars[v]; 
        arg_vars.push_back(v);
        stream_table[v] = false;
        var_idmap_[v] = "arg_top_" + std::to_string(arg_top_count);
        arg_top_vars[v] = std::make_tuple(vid, std::get<1>(tuple),
                                          std::get<2>(tuple));
        arg_top_count += 1;
      }
      PrintStmt(op->body);
    };

    // Split host into pre/post kernel
    void SplitHost() {
      std::string code = this->GetHost();
      size_t pos = code.find("top(");
      pre_kernel = code.substr(0, pos -1);
      post_kernel = code.substr(code.find('\n', pos) + 1);
    } 
};

// unified simulation function for diff platforms 
runtime::Module BuildSimModule(Array<LoweredFunc> funcs,
                               Array<Expr> attrs,
                               Array<Expr> values) {
  CodeAnalysMerlinC ca;
  CodeGenHost cg_host;
  CodeGenXcel cg_dev;
  for (LoweredFunc f : funcs) {
    ca.AddFunction(f);
    str2tupleMap<std::string, Type> map_arg_type;
    map_arg_type = ca.Finish();
    cg_host.AddFunction(f, map_arg_type);
    cg_dev.AddFunction(f, map_arg_type);
  }
  cg_host.SplitHost();
  return runtime::CreateSimModule(funcs[0], 
                                  cg_host.pre_kernel,
                                  cg_host.post_kernel,
                                  cg_dev.arg_vars,
                                  cg_dev.stream_table,
                                  cg_dev.arg_top_vars,
                                  cg_dev.GetDevice());
}

TVM_REGISTER_API("codegen.build_sim")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildSimModule(args[0], args[1], args[2]);
  });

}  // namespace codegen
}  // namespace TVM
