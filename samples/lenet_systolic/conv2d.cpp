------ Host Code ------

void default_function(float* Input, float* Weight, float* ret) {
  ap_int<32> __device_scope;

  cl::Kernel kernel(program, "test", &err);
  cl::Buffer buffer_ret(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*1000*20*24*24, ret, &err);
  cl::Buffer buffer_Input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*1000*20*24*24, Input, &err);
  cl::Buffer buffer_Weight(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(float)*20*20*5*5, Weight, &err);

  // set device kernel buffer
  err = kernel.setArg(0, buffer_ret);
  err = kernel.setArg(1, buffer_Input);
  err = kernel.setArg(2, buffer_Weight);
  err = q.enqueueMigrateMemObjects({buffer_ret, buffer_Input, buffer_Weight}, 0/*from host*/);
  q.finish();

  // enqueue kernel function
  std::chrono::duration<double> kernel_time(0);
  auto kernel_start = std::chrono::high_resolution_clock::now();
  cl::Event event;
  err = q.enqueueTask(kernel, NULL, &event);

  err = q.finish();
  auto kernel_end = std::chrono::high_resolution_clock::now();
  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
  auto kernel_time_in_sec = kernel_time.count();
  std::cout << "Execution Time:" <<  kernel_time_in_sec;
  err = q.enqueueMigrateMemObjects({buffer_ret, buffer_Input, buffer_Weight}, CL_MIGRATE_MEM_OBJECT_HOST);

  // execution on host 
}

------ Xcel Code ------

#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>

extern "C" {
void test(float ret[1000][20][24][24], float Input[1000][20][24][24], float Weight[20][20][5][5]) {
    #pragma HLS INTERFACE m_axi port=ret offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=Input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=Weight offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=ret bundle=control
    #pragma HLS INTERFACE s_axilite port=Input bundle=control
    #pragma HLS INTERFACE s_axilite port=Weight bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
      float _top;
      #pragma HLS dataflow
      hls::stream<float > cfg_in;
      #pragma HLS stream variable=cfg_in depth=1
      hls::stream<float > cfg_out;
      #pragma HLS stream variable=cfg_out depth=1
      float conv2d_nchw_systolic;
      kernel(Input, Weight, ret, cfg_in, cfg_out);
    }
}

