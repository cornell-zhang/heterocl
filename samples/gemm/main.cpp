#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#include <CL/cl2.hpp>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sys/ipc.h>
#include <sys/shm.h>
#pragma once




int main(void) { 
#if defined(SDX_PLATFORM) && !defined(TARGET_DEVICE)
  #define STR_VALUE(arg) #arg
  #define GET_STRING(name) STR_VALUE(name)
  #define TARGET_DEVICE GET_STRING(SDX_PLATFORM)
#endif
    char* xclbinFilename = argv[1];

    std::vector<int> source_0(10 * 10);
    std::vector<int> source_1(10 * 10);
    std::vector<int> source_2(10 * 10);

    size_t vector_size_bytes_0 = sizeof(int) * 10 * 10;
    size_t vector_size_bytes_1 = sizeof(int) * 10 * 10;
    size_t vector_size_bytes_2 = sizeof(int) * 10 * 10;

    int* arg_0 = (int*)shmat(3866625, nullptr, 0);
    for (size_t i0 = 0; i0 < 10; i0++) {
      for (size_t i1 = 0; i1 < 10; i1++) {
        source_0[i1 + i0*10] = arg_0[i1 + i0*10];
      }
    }
    int* arg_1 = (int*)shmat(3866624, nullptr, 0);
    for (size_t i0 = 0; i0 < 10; i0++) {
      for (size_t i1 = 0; i1 < 10; i1++) {
        source_1[i1 + i0*10] = arg_1[i1 + i0*10];
      }
    }
    int* arg_2 = (int*)shmat(3276805, nullptr, 0);
    for (size_t i0 = 0; i0 < 10; i0++) {
      for (size_t i1 = 0; i1 < 10; i1++) {
        source_2[i1 + i0*10] = arg_2[i1 + i0*10];
      }
    }
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device);

    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);

    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);

    int err1;
    cl::Kernel kernel(program, "default_function", &err1);
    auto default_function = cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&>(kernel);

    cl::Buffer buffer_0(context, CL_MEM_READ_WRITE, vector_size_bytes_0);
    cl::Buffer buffer_1(context, CL_MEM_READ_WRITE, vector_size_bytes_1);
    cl::Buffer buffer_2(context, CL_MEM_READ_WRITE, vector_size_bytes_2);

    q.enqueueWriteBuffer(buffer_0, CL_TRUE, 0, vector_size_bytes_0, source_0.data());
    q.enqueueWriteBuffer(buffer_1, CL_TRUE, 0, vector_size_bytes_1, source_1.data());
    q.enqueueWriteBuffer(buffer_2, CL_TRUE, 0, vector_size_bytes_2, source_2.data());

    default_function(cl::EnqueueArgs(q, cl::NDRange(1,1,1), cl::NDRange(1,1,1)),buffer_0, buffer_1, buffer_2);
    q.finish();

    q.enqueueReadBuffer(buffer_0, CL_TRUE, 0, vector_size_bytes_0, source_0.data());
    q.enqueueReadBuffer(buffer_1, CL_TRUE, 0, vector_size_bytes_1, source_1.data());
    q.enqueueReadBuffer(buffer_2, CL_TRUE, 0, vector_size_bytes_2, source_2.data());

    for (size_t i0 = 0; i0 < 10; i0++) {
      for (size_t i1 = 0; i1 < 10; i1++) {
        arg_0[i1 + i0*10] = source_0[i1 + i0*10];
      }
    }
    shmdt(arg_0);
    for (size_t i0 = 0; i0 < 10; i0++) {
      for (size_t i1 = 0; i1 < 10; i1++) {
        arg_1[i1 + i0*10] = source_1[i1 + i0*10];
      }
    }
    shmdt(arg_1);
    for (size_t i0 = 0; i0 < 10; i0++) {
      for (size_t i1 = 0; i1 < 10; i1++) {
        arg_2[i1 + i0*10] = source_2[i1 + i0*10];
      }
    }
    shmdt(arg_2);
}
