#include <tvm/runtime/c_backend_api.h>
#include <dmlc/logging.h>
#include <thread>
#include <mutex>
#include <vector>

using std::thread;
using std::vector;

namespace TVM {
namespace runtime {

class StreamThreadPool {
 public:
  StreamThreadPool() {
  }

  ~StreamThreadPool() {
  }

  static StreamThreadPool* Global() {
    static StreamThreadPool inst;
    return &inst;
  }

  int Launch(FKernelLambda flambda, void* cdata) {
    threads.push_back(thread(flambda, cdata));
    return 0;
  }

  int Sync() {
    for (size_t i = 0; i < threads.size(); i++) {
      threads[i].join();
    }
    threads.clear();
    return 0;
  }

 private:
  vector<thread> threads{};
};

} // namespace runtime
} // namespace TVM

int TVMBackendKernelThreadLaunch(FKernelLambda flambda, void* cdata) {
  return TVM::runtime::StreamThreadPool::Global()->Launch(flambda, cdata);
}

int TVMBackendKernelThreadSync() {
  return TVM::runtime::StreamThreadPool::Global()->Sync();
}

