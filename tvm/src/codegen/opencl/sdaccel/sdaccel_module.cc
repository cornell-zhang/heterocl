/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-07-30 15:15:28
 * @LastEditTime: 2019-08-14 16:16:03
 * @LastEditors: Please set LastEditors
 */
/*
    Yang.Bai
    yb269@cornell.edu
*/
# include "./sdaccel_module.h"
# include <fstream>
# include <unistd.h>
# include <sys/ipc.h>
# include <sys/shm.h>
# include <iostream>

namespace TVM {
namespace runtime {

namespace {

void PrintIndent(std::ofstream& stream, int indent) {
    for (int i = 0;i < indent; i++ ) {
        stream << ' ';
    }
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

inline std::string Type2Str(TVMType t) {

}

inline std::string Tpye2ExtStr(TVMType t) {

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
void PrintCopy()




// copy values from local mem back to shared mem
void PrintCopyBack()



void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc func,
                 std::string test_file) {
  int indent = 0;
  std::ofstream stream;
  stream.open("host.cpp");

  // write the header files and macro commmands.
  stream << "# define CL_HPP_CL_1_2_DEFAULT_BUILD\n";
  stream << "# define CL_HPP_TARGET_OPENCL_VERSION 120\n";
  stream << "# define CL_HPP_MINIMUM_OPENCL_VERSION 120\n";
  stream << "# define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1\n";
  stream << "# include <CL/cl2.hpp>\n";
  stream << "# include <fstream>\n";
  stream << "# include <sys/types.h>\n";
  stream << "# include <sys/stat.h>\n";
  stream << "# include <fcntl.h>\n";
  stream << "# include <unistd.h>\n";
  stream << "# include <stdlib.h>\n";
  stream << "# include <stdio.h>\n";
  stream << "# include <cstring>\n";
  stream << "# include <iostream>\n";
  stream << "# include <iomanip>\n";
  stream << "# include <math.h>\n";
  stream << "# pragram once\n";
  stream << "# define LENGTH (1024)\n";
  stream << "# define NUM_WORKGROUPS (1)\n";
  stream << "# define WORKGROUP_SIZE (16)\n";
  stream << test_file;
  stream << "int main(void) { \n";
  indent += 2;


  // get the platform and devices
  stream << "#if define(SDX_PLATFORM) && !defined(TARGET_DEVICE)\n";
  PrintIndent(stream, indent);
  stream << "# define STR_VALUE(arg)    #arg\n";
  PrintIndent(stream, indent);
  stream << "# define GET_STRING(name) STR_VALUE(name)\n";
  PrintIndent(stream, indent);
  stream << "# define TARGET_DEVICE GET_STRING(SDX_PLATFORM)\n"
  stream << "#endif";


  // get the xclbin filename .
  stream << "char * xclbinFilename = argv[1]\n";
  stream << "size_t \n";

  // source memories

  
  // create the test data and goldn data locally 




  // OpenCL HOST CODE AREA START
  // get First Platform
  stream << "std::vector<cl::Platform> platforms;\n";
  stream << "cl::Platform::get(&platforms)\n;";
  stream << "cl::Platform platform = platform[0];\n";
  stream << "std::cout << "" "

  // get accelerator devices and select 1st such device

  // create context and command queue for selected device


  // load xcl binary into the buffer


  // creat program from binary file

  // create kernel 

  // create buffers inside device

  // copy input data to device buffer from host memory 

  // run the kernel 

  // copy device result data to host memory 
  // OpenCL HOST CODE AREA END



  // compare the results of the kernel to the simulation 




  for ( int i = 0;i < args.size(); i++ ) {
      if (args[i].type_code() == kArrayHandle) {
          // read from the shared memory
          PrintIndent(stream, indent);
          stream << Type2Byte(arg_types[i]) << "* ";
          stream << Type2Byte(arg_types)[i] << "*";
          PrintIndent(stream, indent);


      }
  }

  // call the function
  PrintIndent(stream, indent);
  stream << func->name << "(";
  for (int i = 0;i < args.size();i++) {
      if (i != args.size()-1) {
          stream << ", ";
      }
  }
  stream << ");\n";

  // copy to shared mem
  for (int i = 0;i < args.size();i++ ) {
      if (args[i].type_code() == kArrayHandle) {
          TVMArray* arr = args[i];
          PrintCopyBack(arr, stream, indent, i);
          PrintIndent(stream, indent);
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
        GenHostCode(args, shmids, arg_types, func_, test_file_);
        // TODO: find a better way to do the following
        LOG(CLEAN) << "Compiling the generated SDAccel OpenCL code ...";
        LOG(CLEAN) << "Running SDAccel OpenCL simulation ...";
        system("make -f sdaccel.mk run_cpu_em");
        // system("./out");
        LOG(CLEAN) << "Finished SDAccel OpenCL simulation";
        system("make -f sdaccel.mk clean");
        FreeSharedMem(args, shmids, arg_sizes);
      });
  }

 private:
  LoweredFunc func_;
  std::string test_file_;
};

Module CreateSDAccelModule(
    LoweredFunc func,
    std::string code) {

  std::shared_ptr<SDAccelModuleNode> n =
    std::make_shared<SDAccelModuleNode>(func, code);

  return Module(n);
}


} // namespace runtime
} // namespace TVM