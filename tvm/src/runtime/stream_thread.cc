#include <tvm/runtime/c_backend_api.h>
#include <dmlc/logging.h>
#include <thread>
#include <mutex>
#include <vector>

using std::thread;
using std::vector;
using std::mutex;

mutex stream_buffer_mtx;

namespace TVM {
namespace runtime {

class StreamBuffer {
 public:
  StreamBuffer(uint64_t depth=0) : depth(depth) {
    buffer.resize(depth);
  }

  void set_depth(int depth) { depth = dpeth; }

  int get_depth() { return depth; }

  bool empty() { return head == tail; }

  bool full() { return head + depth == tail; }

  bool try_read() {
    if (empty()) return false;
    else return true;
  }

  int read() {
    while (!try_read()) {
      std::this_thread::yield();
    }
    int val = buffer[tail%depth];
    tail++;
    return val;
  }

  bool try_write() {
    if (full()) return false;
    else return true;
  }

  void write(int val) {
    while (!try_write()) {
      std::this_thread::yield();
    }
    buffer[head%depth] = val;
    head++;
  }

 private:
  uint64_t head{0};
  uint64_t tail{0};
  uint64_t depth;
  vector<int> buffer;
};

class StreamBufferPool {
 public:
  StreamBufferPool() {}

  ~StreamBufferPool() {}

  static StreamBufferPool* Global() {
    static StreamBufferPool inst;
    return &inst;
  }

  int Create(int depth) {
    streams.push_back(StreamBuffer(depth));
    return streams.size()-1;
  }

  int BlockingRead(int id, int depth, int* val) {
    stream_buffer_mtx.lock();
    if (id > int(streams.size())) {
      streams.resize(id+1);
      streams[id] = this->Create(depth);
    } else if (depth != stream[id].get_depth()) {
      stream[id].set_depth(depth);
    }
    stream_buffer_mtx.unlock();
    int ret = streams[id].read();
    val = &ret;
    return ret;
  }

  int BlockingWrite(int id, int depth, int val) {
    stream_buffer_mtx.lock();
    if (id > int(streams.size())) {
      streams.resize(id+1);
      streams[id] = this->Create(depth);
    } else if (depth != stream[id].get_depth()) {
      stream[id].set_depth(depth);
    }
    stream_buffer_mtx.unlock();
    streams[id].write(val);
    return 0;
  }

 private:
  vector<StreamBuffer> streams{};
};

class StreamThreadPool {
 public:
  StreamThreadPool() {}

  ~StreamThreadPool() {}

  static StreamThreadPool* Global() {
    static StreamThreadPool inst;
    return &inst;
  }

  int Launch(FKernelLambda flambda, void* cdata) {
    // TODO: need to check if we reach the maximum number of threads
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

int TVMBackendStreamBlockingRead(int id, int depth, int* val) {
  return TVM::runtime::StreamBufferPool::Global()->BlockingRead(id, depth, val);
}

int TVMBackendStreamBlockingWrite(int id, int depth, int val) {
  return TVM::runtime::StreamBufferPool::Global()->BlockingWrite(id, depth, val);
}
