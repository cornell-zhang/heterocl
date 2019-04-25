/*!
 *  Copyright (c) 2018 by Contributors
 * \file build_vhls.cc
 * \brief Build HLS C modules from source.
 */
#include "./vhls_module.h"
#include <fstream>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <iostream>

namespace tvm {
namespace runtime {

namespace {

void PrintIndent(std::ofstream& stream, int indent) {
  for (int i = 0; i < indent; i++)
    stream << ' ';
}

inline size_t GetDataSize(TVMArray* arr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < arr->ndim; ++i) {
    size *= arr->shape[i];
  }
  size_t byte = (arr->dtype.bits + 7) / 8;
  if (byte > 2){
    if (byte < 4) byte = 4;
    else if (byte < 8) byte = 8;
    else byte = 16;
  }
  size *= (byte * 8 * arr->dtype.lanes + 7) / 8;
  return size;
}

inline std::string Type2Str(TVMType t) {
  std::string str = "";
  if (t.code == kDLInt) {
    if (t.fracs > 0) str += "ap_fixed<";
    else             str += "ap_int<";
    str += std::to_string(static_cast<int>(t.bits));
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.fracs)) + ">";
    else             str += ">";
  } else if (t.code == kDLUInt) {
    if (t.fracs > 0) str += "ap_ufixed<";
    else             str += "ap_uint<";
    str += std::to_string(static_cast<int>(t.bits));
    if (t.fracs > 0) str += ", " + std::to_string(static_cast<int>(t.fracs)) + ">";
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
                    std::vector<size_t>& arg_sizes,
                    std::vector<TVMType>& arg_types) {
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      arg_sizes.push_back(GetDataSize(arr));
      arg_types.push_back(arr->dtype);
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
      key_t key = ftok("/tmp", i+1);
      int shmid = shmget(key, 1024, 0666|IPC_CREAT);
      shmids.push_back(shmid);
      // copy mem from TVM args to the shared memory
      void* mem = shmat(shmid, nullptr, 0);
      memcpy(mem, arr->data, arg_sizes[i]);
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

void PrintLoop(TVMArray* arr, 
               std::ofstream& stream, 
               int indent, size_t nth_arr) {
  for (int i = arr->ndim - 1; i >= 0; i--) {
    PrintIndent(stream, indent);
    stream << "for (size_t i" << i << " = 0; ";
    stream << "i" << i << " < " << arr->shape[i] << "; ";
    stream << "i" << i << "++) {\n";
    indent += 2;
    // copy data TODO: need to shift for fixed-point
    if (i == 0) {
      PrintIndent(stream, indent);
      stream << "arg_top_" << nth_arr;
      for (int j = arr->ndim-1; j >= 0; j--) {
        stream << "[i" << j << "]"; 
      }
      stream << " = (";
      stream << Type2Str(arr->dtype);
      stream << ")(arg_" << nth_arr;
      stream << "[i0";
      int mul = 1;
      for (int j = 1; j < arr->ndim; j++) {
        mul *= arr->shape[j-1];
        stream << " + i" << j << "*" << mul;
      }
      stream << "]);\n";
    }
  }
  for (int i = 0; i < arr->ndim; i++) {
    indent -= 2;
    PrintIndent(stream, indent);
    stream << "}\n";
  }
}

void PrintLoopBack(TVMArray* arr, 
                   std::ofstream& stream, 
                   int indent, size_t nth_arr) {
  for (int i = arr->ndim - 1; i >= 0; i--) {
    PrintIndent(stream, indent);
    stream << "for (size_t i" << i << " = 0; ";
    stream << "i" << i << " < " << arr->shape[i] << "; ";
    stream << "i" << i << "++) {\n";
    indent += 2;
    // copy data TODO: need to shift for fixed-point
    if (i == 0) {
      PrintIndent(stream, indent);
      stream << "arg_" << nth_arr;
      stream << "[i0";
      int mul = 1;
      for (int j = 1; j < arr->ndim; j++) {
        mul *= arr->shape[j-1];
        stream << " + i" << j << "*" << mul;
      }
      stream << "] = (";
      stream << Type2Byte(arr->dtype);
      stream << ")(arg_top_" << nth_arr;
      for (int j = arr->ndim-1; j >= 0; j--) {
        stream << "[i" << j << "]"; 
      }
      stream << ");\n";
    }
  }
  for (int i = 0; i < arr->ndim; i++) {
    indent -= 2;
    PrintIndent(stream, indent);
    stream << "}\n";
  }
}

void GenHostCode(TVMArgs& args,
                 const std::vector<int>& shmids,
                 const std::vector<TVMType>& arg_types,
                 LoweredFunc func,
                 std::string test_file) {
  int indent = 0;
  std::ofstream stream;
  stream.open("main.cpp");
  stream << "#include <sys/ipc.h>\n";
  stream << "#include <sys/shm.h>\n";
  /*
  stream << "#include <ap_int.h>\n";
  stream << "#include <ap_fixed.h>\n";
  */
  stream << test_file;
  //stream << "#include \"" << func_->name << ".h\"\n";
  stream << "int main(void) { \n";
  indent += 2;
  for (size_t i = 0; i < shmids.size(); i++) {
    PrintIndent(stream, indent);
    stream << Type2Byte(arg_types[i]) << "* "; 
    stream << "arg_" << i << " = ";
    stream << "(" << Type2Byte(arg_types[i]) << "*)";
    stream << "shmat(" << shmids[i] << ", nullptr, 0);\n";
    PrintIndent(stream, indent);
    stream << Type2Str(arg_types[i]) << " ";
    stream << "arg_top_" << i;
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      for (int j = arr->ndim-1; j >=0; j--)
        stream << "[" << arr->shape[j] << "]";
    }
    stream << ";\n";
  }
  // copy from shared mem
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      PrintLoop(arr, stream, indent, i);
    }
  }
  // call the function
  PrintIndent(stream, indent);
  stream << func->name << "(";
  for (int i = 0; i < args.size(); i++) {
    stream << "arg_top_" << i;
    if (i != args.size()-1) 
      stream << ", ";
  }
  stream << ");\n";
  // copy to shared mem
  for (int i = 0; i < args.size(); i++) {
    if (args[i].type_code() == kArrayHandle) {
      TVMArray* arr = args[i];
      PrintLoopBack(arr, stream, indent, i);
    }
  }
  for (size_t i = 0; i < shmids.size(); i++) {
    PrintIndent(stream, indent);
    stream << "shmdt(";
    stream << "arg_" << i << ");\n";
  }
  stream << "}\n";
  stream.close();
}
} // namespace

class VivadoHLSModuleNode final : public ModuleNode {
 public:
  VivadoHLSModuleNode(LoweredFunc func, std::string test_file) 
    : func_(func), test_file_(test_file) {}

  const char* type_key() const {
    return "vivado_hls_csim";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv){
        // need to check if # args == # inputs to top func
        std::vector<size_t> arg_sizes;
        std::vector<TVMType> arg_types;
        std::vector<int> shmids;
        CollectArgInfo(args, arg_sizes, arg_types);
        GenSharedMem(args, shmids, arg_sizes);
        GenHostCode(args, shmids, arg_types, this->func_, this->test_file_);
        system("g++ main.cpp -o out");
        system("./out");
        FreeSharedMem(args, shmids, arg_sizes);
        *rv = 1;
      });
  }

 private:
  LoweredFunc func_;
  std::string test_file_;
};

Module CreateVivadoHLSModule(
    LoweredFunc func,
    std::string code) {

  std::shared_ptr<VivadoHLSModuleNode> n =
    std::make_shared<VivadoHLSModuleNode>(func, code);

  return Module(n);
}

} // namespace runtime
} // namespace tvm
