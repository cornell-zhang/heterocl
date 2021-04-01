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

// rapidjson headers
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

namespace TVM {
namespace runtime {

std::string getpath(void) {
   char buff[256];
   char* ptr = getcwd(buff, 256);
   if (ptr == NULL) 
    LOG(FATAL) << "getcwd failed";
   std::string cwd(buff);
   return cwd;
}

void PrintIndent(std::ofstream& stream, int indent) {
  for (int i = 0; i < indent; i++)
    stream << ' ';
}

inline size_t GetTypeSize(TVMType t) {
  size_t byte = (t.bits + 7) / 8;
  size_t new_byte = 1;
  while (new_byte < byte) {
    new_byte <<= 1;
  }
  return new_byte;
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

inline std::string Type2ByteVHLS(TVMType t) {
  std::string str = "";
  if (t.code == kDLFloat) {
    str += "float";
  } else if (t.code == kDLInt || t.code == kDLUInt) {
    str += "ap_";
    if (t.code == kDLUInt) str += "u";
    if (t.fracs == 0) {
        str += "int<" + std::to_string(t.bits) + ">";
    } else {
        str += "fixed<" + std::to_string(t.bits)  + "," +
               std::to_string(t.bits - t.fracs) + ">";
    }
  }
  return str;
}

inline std::string Type2ByteCatapultC(TVMType t) {
  std::string str = "";
  if (t.code == kDLFloat) {
    str += "float";
  } else if (t.code == kDLInt) {
    str += "ac_";
    if (t.fracs == 0) {
        str += "int<" + std::to_string(t.bits) + ", true>";
    } else {
        str += "fixed<" + std::to_string(t.bits)  + "," +
               std::to_string(t.bits - t.fracs) + ">";
    }
  } else if (t.code == kDLUInt) {
    str += "ac_";
    if (t.fracs == 0) {
        str += "int<" + std::to_string(t.bits) + ", false>";
    } else {
        str += "fixed<" + std::to_string(t.bits)  + "," +
               std::to_string(t.bits - t.fracs) + ">";
    }
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


void GenJSONInputs(TVMArgs& args,
                  std::vector<std::string> arg_names,
                  std::vector<size_t>& arg_sizes,
                  const std::vector<TVMType>& arg_types,
                  std::string project) {
    
  // Write data into the JSON file
  rapidjson::Document jsonDoc;
  jsonDoc.SetObject();
  rapidjson::Value myArray(rapidjson::kArrayType);
  rapidjson::Document::AllocatorType& allocator = jsonDoc.GetAllocator();

  std::string input_json = project + "/inputs.json";
  FILE* outfile = fopen(input_json.c_str(), "w");
  char writeBuffer[65536];

  for (int i = 0; i < args.size(); i++) {
    TVMArray* arr = args[i];
    rapidjson::Value v(rapidjson::kArrayType);
    void* mem = (void *)malloc(arg_sizes[i]);
    memcpy(mem, arr->data, arg_sizes[i]);

    int shape = 1;
    for (int j = arr->ndim-1; j >= 0; j--) {
      shape *= arr->shape[j];
    }

    HCL_DEBUG_LEVEL(2) << "[ debug ] Dumping " << arg_names[i] << " (size="
      << arg_sizes[i] << ", shape=" << shape << ") into JSON...";
    if (arg_types[i].code == kDLFloat || arg_types[i].fracs > 0) {
      float* data = (float*)mem;
      for (int k = 0; k < shape; k++) {
        v.PushBack(rapidjson::Value().SetFloat(data[k]), allocator);
      }
    } else if (arg_types[i].code == kDLInt || arg_types[i].code == kDLUInt) {
      int* data = (int*)mem;
      for (int k = 0; k < shape; k++) {
        v.PushBack(rapidjson::Value().SetInt(data[k]), allocator);
      }
    } else {
      CHECK(false) << arg_types[i].code;
    }
    const std::string name = arg_names[i];
    rapidjson::Value n(name.c_str(), allocator);
    jsonDoc.AddMember(n, v, allocator);
    free(mem);
  } 

  LOG(CLEAN) << "Generating JSON inputs into " << input_json << "...";
  rapidjson::FileWriteStream os(outfile, writeBuffer, sizeof(writeBuffer));
  rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
  jsonDoc.Accept(writer);
  fclose(outfile);
}

void GenSharedMem(TVMArgs& args,
                  std::vector<int>& shmids,
                  std::vector<size_t>& arg_sizes) {
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      // generate shared memory key and id
      key_t key = ftok(getpath().c_str(), i+1);
      int shmid = shmget(key, arg_sizes[i], 0666|IPC_CREAT);
      if (shmid < 0)
        LOG(FATAL) << "shmid failed";
      shmids.push_back(shmid);
      // copy mem from TVM args to the shared memory
      void* mem = shmat(shmid, nullptr, 0);
      memcpy(mem, arr->data, arg_sizes[i]);

    } else { // shared memory for var
      key_t key = ftok(getpath().c_str(), i+1);
      int shmid = shmget(key, arg_sizes[i], 0666|IPC_CREAT);
      if (shmid < 0)
        LOG(FATAL) << "shmid failed";
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
               int indent, size_t nth_arr, 
               std::string get_type, bool multi_dim_arr) {
  for (int i = 0; i < arr->ndim; i++) {
    PrintIndent(stream, indent);
    stream << "for (size_t i" << i << " = 0; ";
    stream << "i" << i << " < " << arr->shape[i] << "; ";
    stream << "i" << i << "++) {\n";
    indent += 2;

    if (i == arr->ndim - 1) {
      PrintIndent(stream, indent);
      auto arg_name = arg_names[nth_arr];

      if (multi_dim_arr) {
        stream << arg_name << "[i0";
        for (int j = 1; j < arr->ndim; j++) {
          stream << "][i" << j;
        }
        stream << "]";
      } else {
        stream << arg_name << "[i" << arr->ndim-1;
        int base = 1;
        for (int j = arr->ndim-2; j >= 0; j--) {
          base *= arr->shape[j+1];
          stream << " + i" << j << "*" << base;
        }        
        stream << "]";
      }

      stream << " = (" << arg_name << "_d";
      stream << "[i" << arr->ndim-1;
      int mul = 1;
      for (int j = arr->ndim-2; j >= 0; j--) {
        mul *= arr->shape[j+1];
        stream << " + i" << j << "*" << mul;
      }
      stream << "]." << get_type << ")";
      stream << ";\n";
    }
  }
  for (int i = 0; i < arr->ndim; i++) {
    indent -= 2;
    PrintIndent(stream, indent);
    stream << "}\n";
  }
}

// Copy values from local mem back to shared mem
void PrintCopyBack(TVMArray* arr, 
                   std::vector<std::string> arg_names,
                   std::unordered_map<std::string, bool> arg_access_status,
                   std::ofstream& stream, 
                   int indent, size_t nth_arr,
                   bool multi_dim_arr) {
                     
  std::string arg_name = arg_names[nth_arr];
  CHECK(arg_access_status.count(arg_name));
  if (!arg_access_status.at(arg_name)) {
    HCL_DEBUG_LEVEL(2) << "[  INFO  ] Tensor " << arg_name << " not written. Skip copyback.";
    return;
  }

  stream << "  document[\"" << arg_name << "\"].Clear();\n";
  stream << "  rapidjson::Value v_" << arg_name << "(rapidjson::kArrayType);\n";

  for (int i = 0; i < arr->ndim; i++) {
    PrintIndent(stream, indent);
    stream << "for (size_t i" << i << " = 0; ";
    stream << "i" << i << " < " << arr->shape[i] << "; ";
    stream << "i" << i << "++) {\n";
    indent += 2;
    if (i == arr->ndim-1) {

      std::string get_type;
      if (arr->dtype.code == kDLFloat || arr->dtype.fracs > 0) {
        get_type = "SetFloat";
      } else {
        get_type = "SetInt";
      }
      PrintIndent(stream, indent);
      stream << "v_" << arg_names[nth_arr] << ".PushBack(rapidjson::Value()."
             << get_type << "(";
      stream << arg_names[nth_arr];

      if (multi_dim_arr) {     
        stream << "[i0";
        for (int j = 1; j < arr->ndim; j++) {
          stream << "][i" << j;
        }
      } else {
        stream << "[i" << arr->ndim-1;
        int base = 1;
        for (int j = arr->ndim-2; j >= 0; j--) {
          base *= arr->shape[j+1];
          stream << " + i" << j << "*" << base;
        }        
      }
      stream << "]), allocator);\n";
    }
  }
  for (int i = 0; i < arr->ndim; i++) {
    indent -= 2;
    PrintIndent(stream, indent);
    stream << "}\n";
  }
  stream << "  document[\"" << arg_names[nth_arr] << "\"] = v_" << arg_names[nth_arr] << ";\n";
}

// Generate config code (TCL/INI e.t.c)
void GenConfigCode(std::string& test_file, 
    std::string platform, std::string project) {
  if (test_file.find_first_not_of(" \t\n") == std::string::npos) return;
  if (platform == "vitis") {
    std::ofstream stream;
    std::string config_ext = "ini";
    stream.open(project + "/config." + config_ext);
    stream << test_file;
    stream.close();
  }
}

// generate kernel code into files 
void GenKernelCode(std::string& test_file, std::vector<std::string> arg_names, 
                   std::string platform, std::string backend, std::string project) {
  if (test_file.find_first_not_of(" \t\n") == std::string::npos) return;
  std::ofstream stream;

  std::string kernel_ext = "cpp";
  if (platform == "sdaccel" && backend == "sdaccel") kernel_ext = "cl";
  if (platform == "aocl") kernel_ext = "cl";
  stream.open(project + "/kernel." + kernel_ext);

  // generate hash
  std::hash<std::string> hasher;
  stream << "// HASH:" << ((size_t)hasher(test_file) & 0xFFFFFFFF) << "\n";

  // create typedef and header 
  if (platform == "vivado_hls" || platform == "sdsoc") { 

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
    header.open(project + "/kernel.h");
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
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <cassert>

// rapidjson headers
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"
using namespace rapidjson;

)";
  
  if (platform == "sdaccel" || platform == "vitis") {
    stream << "// opencl harness headers\n";
    stream << "#include \"xcl2.hpp\"\n";
    stream << "#include \"ap_fixed.h\"\n";
    stream << "#include \"ap_int.h\"\n";
    stream << "#include <cmath>\n";
    stream << "#include <vector>\n\n";

  } else if (platform == "vivado_hls" || platform == "sdsoc") {

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

#define CHECK(status) 							                              \
    if (status != CL_SUCCESS)						                          \
{									                                                \
    fprintf(stderr, "error %d in line %d.\n", status, __LINE__);	\
    exit(1);								                                      \
}									                                                \

void* acl_aligned_malloc (size_t size) {
  void *result = NULL;
  posix_memalign (&result, 64, size);
  return result;
}

double compute_kernel_execution_time(cl_event &event, double &start_d, double &end_d)
{
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    start_d = (double)1.0e-9 * start;
    end_d   = (double)1.0e-9 * end;
    return 	(double)1.0e-9 * (end - start); // nanoseconds to seconds
}

)";

  }
  stream << include << "\n";

}

// separate host code into partitions 
std::string SplitHostCode(std::string platform, std::string host_code, 
  std::string& include) {
  std::string return_main_body;
  if (platform == "catapultc") {
    std::string key = "#include"; // find the last occurance of '#include' as the ending of header
    std::size_t found = host_code.rfind(key);
    std::size_t split_point = host_code.find("\n", found);
    split_point = host_code.find("\n", split_point); // after two newlines
    include = host_code.substr(0, split_point);
    return_main_body = host_code.substr(split_point);
  } else {
    size_t pos = host_code.find("default_function");
    include = host_code.substr(0, host_code.rfind("void", pos));
    return_main_body = host_code.substr(host_code.find("{", pos) + 1);
    auto begin = return_main_body.find_first_not_of(" \t\n");
    auto length = return_main_body.rfind("}") - begin;
    return_main_body = return_main_body.substr(begin, length);
  }

  return "\n  " + return_main_body;
}

// generate host code according to platform type
void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc lowered_func, std::string platform,
                 std::string host_code, std::string top_code, 
                 std::vector<std::string> arg_names,
                 std::unordered_map<std::string, bool> arg_access_status,
                 bool kernel_is_empty,
                 std::string project) {
  int indent = 0;
  std::ofstream stream;
  if (platform == "catapultc") {
    stream.open(project + "/testbench.cpp");
    std::ofstream head_stream(project + "/test.h");
    head_stream << top_code;
  }
  else {
    HCL_DEBUG_LEVEL(2) << project << " host.cpp";
    stream.open(project + "/host.cpp");
  }

  std::string include;
  auto code = SplitHostCode(platform, host_code, include); 

  GenHostHeaders(stream, platform, include);
  CHECK((signed)arg_names.size() == args.size());

  if (platform == "catapultc") 
    stream << "CCS_MAIN(int argc, char **argv) {\n";
  else
    stream << "int main(int argc, char ** argv) {\n";
  indent += 2;
  stream << "  std::cout << \"[INFO] Initialize input buffers...\\n\";\n";
  
  // Create read buffers
  if (platform == "catapultc") {
    stream << R"(
    FILE *f = fopen(")"; 
    stream << getpath();
    stream << R"(
      /project/inputs.json", "r");
    char readBuffer[65536];
    FileReadStream is(f, readBuffer, sizeof(readBuffer));

    Document document;
    document.ParseStream(is);
    fclose(f);
  )"; 
  } else {
    stream << R"(
    FILE *f = fopen("inputs.json", "r");
    char readBuffer[65536];
    FileReadStream is(f, readBuffer, sizeof(readBuffer));

    Document document;
    document.ParseStream(is);
    fclose(f);
  )";
  }

  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      stream << "  assert(document.HasMember(\"" << arg_names[i] << "\"));\n";
      stream << "  const Value& " << arg_names[i] << "_d = document[\"" << arg_names[i] << "\"];\n";
      stream << "  assert(" << arg_names[i] << "_d.IsArray());\n";
      TVMArray* arr = args[i];
      std::string dtype; 
      auto t = arg_types[i].code;
      if (t == kDLFloat || arr->dtype.fracs > 0) {
        dtype = "GetFloat()";
      } else if (t == kDLInt || t == kDLUInt) {
        dtype = "GetInt()";
      }

      // Create host side OpenCL buffers
      PrintIndent(stream, indent);
      auto arg_name = arg_names[i];

      // Use XRT API to allocate page-pinned buffer (1-dim)
      bool multi_dim_arr = true;
      if (platform == "vitis") {
        multi_dim_arr = false;
        int bits = arg_types[i].bits;
        CHECK(bits % 8 == 0) 
            << "[ Error ] Vitis requires the input arg of bitwidth "
            << "to be 8's multiple. The current input width is " 
            << bits << "...\n";
        size_t constant_size = 1;
        for (int j = 0; j < arr->ndim; j++) {
          constant_size *= arr->shape[j];
        }
        stream << "std::vector<int, aligned_allocator<int>> " << arg_name
               << "(" << constant_size << ");\n ";
    
      } else if (platform == "catapultc") {
        stream << Type2ByteCatapultC(arg_types[i]) << " " << arg_name; 
        stream << "[";
        for (int j = 0; j < arr->ndim; j++) {
          if (j == arr->ndim - 1) {
            stream << arr->shape[j];
          } else {
            stream << arr->shape[j];
            stream << "][";
          }
        }
        stream << "];\n";
      } else {
        stream << "auto " << arg_name << " = new ";
        if (platform == "vivado_hls") {
          stream << Type2ByteVHLS(arg_types[i]);
        } else {
          stream << Type2Byte(arg_types[i]);
        }
        // Print shapes
        stream << "[";
        for (int j = 0; j < arr->ndim; j++) {
          if (j == arr->ndim - 1) {
            stream << arr->shape[j];
          } else {
            stream << arr->shape[j];
            stream << "][";
          }
        }
        stream << "];\n";
      }
      PrintCopy(arr, arg_names, stream, indent, i, 
        dtype, multi_dim_arr);

    } else { 
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
    }
    stream << "\n";
  }

  stream << "  std::cout << \"[ INFO ] Initialize RTE...\\n\";\n";
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
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

  cl::Context context;
  cl::CommandQueue q;
  cl::Program program;
  auto devices = xcl::get_xil_devices();
  int valid_device = 0;

  for (unsigned int i = 0; i < devices.size(); i++) {
      auto device = devices[i];
      // Creating Context and Command Queue for selected Device
      context = cl::Context(device, NULL, NULL, NULL, &err);
      q = cl::CommandQueue(
          context, device, CL_QUEUE_PROFILING_ENABLE, &err);

      std::cout << "Trying to program device[" << i
                << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
      program = cl::Program(context, {device}, bins, NULL, &err);
      if (err != CL_SUCCESS) {
          std::cout << "Failed to program device[" << i
                    << "] with xclbin file!\n";
      } else {
          std::cout << "Device[" << i << "]: program successful!\n";
          valid_device++;
          break; // we break because we found a valid device
      }
  }
  if (valid_device == 0) {
      std::cout << "Failed to program any device found, exit!\n";
      exit(EXIT_FAILURE);
  }

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
  FILE *handle = fopen(AOCX_FILE, "rb");
  fseek(handle, 0, SEEK_END);
  size_t  binary_length = ftell(handle);

  // create program from binary 
  const unsigned char *binary;
  binary = (unsigned char*) malloc(sizeof(unsigned char) * binary_length);
  assert(binary && "Malloc failed"); rewind(handle);
  if (fread((void*)binary, binary_length, 1, handle) == 0) {
    printf("Failed to read from the AOCX file (fread).\n");
    return -1;
  }
  fclose(handle);
  cl_program program = clCreateProgramWithBinary(context, 1, devices,
      &binary_length, (const unsigned char **)&binary, &status, NULL);

  status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  CHECK(status);

)";
    }

    stream << "\n";
  }
  PrintIndent(stream, indent);
  stream << "// Compute and kernel call from host";
  stream << code << "\n";

  stream << "  rapidjson::Document::AllocatorType& allocator"
         << " = document.GetAllocator();\n";

  // Modify the JSON object
  bool multi_dim_arr = true;
  if (platform == "vitis") {
    multi_dim_arr = false;
  }
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      PrintCopyBack(arr, arg_names, arg_access_status, 
        stream, indent, i, multi_dim_arr);
    }
  }

  // Print runtime measurement
  stream << "  std::cout << \"[ INFO ] Finish running...\\n\";\n";
  if (platform == "aocl") {
    stream << R"(
  double k_start_time;	
  double k_end_time;
  double k_exec_time;

  k_exec_time = compute_kernel_execution_time(kernel_exec_event, k_start_time, k_end_time);     
  printf("FPGA Execution time %.8f s \\n");
  )";
  }

  // Write back to JSON
  if (platform == "catapultc") {
    stream << R"(
    FILE* fp = fopen(")";
    stream << getpath();
    stream << R"(
    /project/inputs.json", "w"); 
  
    char writeBuffer[65536];
    FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
  
    Writer<FileWriteStream> writer(os);
    document.Accept(writer);
    fclose(fp);

    )";
  } else {
    stream << R"(
    FILE* fp = fopen("inputs.json", "w"); 
  
    char writeBuffer[65536];
    FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
  
    Writer<FileWriteStream> writer(os);
    document.Accept(writer);
    fclose(fp);

    )";
  }

  if (platform == "catapultc")
    stream << "CCS_RETURN(0);";

  stream << "\n\n";
  PrintIndent(stream, indent);
  stream << "}\n";
  stream.close();

}
}  // namespace runtime
}  // namespace TVM
