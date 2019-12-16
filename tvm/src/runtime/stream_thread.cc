#include <tvm/runtime/c_backend_api.h>
#include <dmlc/logging.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <map>
#include <vector>
#include <tuple>

using std::thread;
using std::vector;
using std::map;
using std::mutex;
using std::atomic;
using std::tuple;
using std::get;

mutex stream_buffer_mtx;
mutex thread_pool_mtx;

namespace TVM {
namespace runtime {

class StreamBuffer {
 public:
  StreamBuffer(uint64_t depth=0) : depth(depth) {
    buffer.resize(depth);
  }

  bool empty() { return head == tail; }

  bool full() { return head == depth + tail; }

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
  atomic<uint64_t> head{0};
  atomic<uint64_t> tail{0};
  uint64_t depth;
  vector<int> buffer;
};

class StreamBufferPool {
 public:
  StreamBufferPool() {}

  ~StreamBufferPool() {
    for (auto kv: streams) {
      delete kv.second;
    }
  }

  static StreamBufferPool* Global() {
    static StreamBufferPool inst;
    return &inst;
  }

  int BlockingRead(int id, int depth, int* val) {
    stream_buffer_mtx.lock();
    if (streams.find(id) == streams.end()) {
      streams[id] = new StreamBuffer(depth);
    }
    stream_buffer_mtx.unlock();
    int ret = streams[id]->read();
    return ret;
  }

  int BlockingWrite(int id, int depth, int val) {
    stream_buffer_mtx.lock();
    if (streams.find(id) == streams.end()) {
      streams[id] = new StreamBuffer(depth);
    }
    stream_buffer_mtx.unlock();
    streams[id]->write(val);
    return 0;
  }

 private:
  map<int, StreamBuffer*> streams{};
};

class StreamThreadPool {
 public:
  static StreamThreadPool* Global() {
    static StreamThreadPool inst;
    return &inst;
  }

  void Wait(int timestep) {
    for (auto& thread : threads[timestep]) {
      if (thread.joinable()) thread.join();
    }
  }

  int Launch(FKernelLambda flambda, void* cdata, int timestep, int num_group) {
    // TODO: need to check if we reach the maximum number of threads
    bool add_thread = true;
    for (; current_step < timestep; ++current_step) {
      if (int(threads[current_step].size()) < max_groups[current_step]) {
        if (timestep >= int(thread_queue.size())) thread_queue.resize(timestep+1);
        thread_queue[timestep].push_back(std::make_tuple(flambda, cdata, num_group));
        add_thread = false;
        break;
      }
      Wait(current_step);
    }
    if (add_thread) {
      threads[timestep].push_back(thread(flambda, cdata));
      max_groups[timestep] = num_group;
    }
    return 0;
  }

  int Sync() {
    Wait(current_step);
    for (size_t i = 0; i < thread_queue.size(); i++) {
      for (auto& t : thread_queue[i]) {
        Launch(get<0>(t), get<1>(t), i, get<2>(t));
      }
      thread_queue[i].clear();
    }
    Wait(current_step);
    return 0;
  }

 private:
  int current_step{0};
  map<int, vector<thread> > threads{};
  vector<vector<tuple<FKernelLambda, void*, int> > > thread_queue{};
  map<int, int> max_groups{};
};

} // namespace runtime
} // namespace TVM

int TVMBackendKernelThreadLaunch(FKernelLambda flambda, void* cdata, int timestep, int num_group) {
  return TVM::runtime::StreamThreadPool::Global()->Launch(flambda, cdata, timestep, num_group);
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
