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
    //this->Sync();
  }

  static StreamThreadPool* Global() {
    static StreamThreadPool inst;
    return &inst;
  }

  int Launch(FKernelLambda flambda, void* cdata) {
    //flambda(cdata);
    threads.push_back(thread(flambda, cdata));
    /*
    threads.push_back(thread([flambda, cdata]{
        LOG(INFO) << flambda(cdata);
        }));
    */
    //LOG(INFO) << "here";
    return 0;
  }

  int Sync() {
    //LOG(INFO) << threads.size();
    for (size_t i = 0; i < threads.size(); i++) {
      //LOG(INFO) << threads[i].get_id();
      threads[i].join();
    }
    //LOG(INFO) << "complete";
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

