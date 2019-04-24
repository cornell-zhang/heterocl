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

class VivadoHLSModuleNode final : public ModuleNode {
 public:
  VivadoHLSModuleNode(LoweredFunc func, std::string& test_file) 
    : func_(func), test_file_(test_file) {}

  const char* type_key() const {
    return "vivado_hls_csim";
  }

  void PrintIndent(std::ofstream& stream, int indent) {
    for (int i = 0; i < indent; i++)
      stream << ' ';
  }

  void GenHostCode(TVMArgs& args, int shmid) {
    int indent = 0;
    std::ofstream stream;
    stream.open("main.cpp");
    stream << "#include <sys/ipc.h>\n";
    stream << "#include <sys/shm.h>\n";
    //stream << "#include \"" << func_->name << ".h\"\n";
    stream << "int main(void) { \n";
    indent += 2;
    PrintIndent(stream, indent);
    TVMArray* arr = args[0];
    stream << "int* A= (int*)shmat(" << shmid << ", (void*)0, 0);\n";
    PrintIndent(stream, indent);
    stream << "A[0] = 40;\n";
    stream << "}\n";
    stream.close();
    // print args as arrays
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv){
        for (size_t i = 0; i < args.size(); i++) {
          TVMArray* arr = args[i];
          if (arr->dtype.code == kDLInt) LOG(INFO) << "int";
          else if (arr->dtype.code == kDLFloat) LOG(INFO) << "float";
          int32_t* data_int = (int32_t*)(arr->data);
          float* data_float = (float*)(arr->data);
          for (size_t j = 0; j < arr->shape[0]; j++) {
            for (size_t k = 0; k < arr->shape[1]; k++) {
              std::cout << (int)*(data_int + (k + j*arr->shape[1])) << " ";
            }
            std::cout << "\n";
          }
        }
        // create a shared memory - should create 1 for each arg
        TVMArray* arr = args[0];
        key_t key = ftok("shmfile", 65);
        int shmid = shmget(key, 1024, 0666|IPC_CREAT);
        int* A = (int*)shmat(shmid, (void*)0, 0);
        memcpy(A, arr->data, 100*4);
        this->GenHostCode(args, shmid);
        system("g++ main.cpp -o out");
        LOG(INFO) << "hereQQ";
        system("./out");
        memcpy(arr->data, A, 100*4);
        *rv = 1;
      });
  }

 private:
  std::string& test_file_;
  LoweredFunc func_;
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
