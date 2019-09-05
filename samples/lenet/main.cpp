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

    std::vector<float> source_0(50 * 1 * 28 * 28);
    std::vector<int> source_1(20 * 1 * 5 * 5);
    std::vector<int> source_2(50 * 20 * 5 * 5);
    std::vector<int> source_3(500 * 800);
    std::vector<int> source_4(10 * 500);
    std::vector<float> source_5(50 * 10);

    size_t vector_size_bytes_0 = sizeof(float) * 50 * 1 * 28 * 28;
    size_t vector_size_bytes_1 = sizeof(int) * 20 * 1 * 5 * 5;
    size_t vector_size_bytes_2 = sizeof(int) * 50 * 20 * 5 * 5;
    size_t vector_size_bytes_3 = sizeof(int) * 500 * 800;
    size_t vector_size_bytes_4 = sizeof(int) * 10 * 500;
    size_t vector_size_bytes_5 = sizeof(float) * 50 * 10;

    float* arg_0 = (float*)shmat(2949125, nullptr, 0);
    for (size_t i0 = 0; i0 < 50; i0++) {
      for (size_t i1 = 0; i1 < 1; i1++) {
        for (size_t i2 = 0; i2 < 28; i2++) {
          for (size_t i3 = 0; i3 < 28; i3++) {
            source_0[i3 + i2*28 + i1*784 + i0*784] = arg_0[i3 + i2*28 + i1*784 + i0*784];
          }
        }
      }
    }
    int* arg_1 = (int*)shmat(3473408, nullptr, 0);
    for (size_t i0 = 0; i0 < 20; i0++) {
      for (size_t i1 = 0; i1 < 1; i1++) {
        for (size_t i2 = 0; i2 < 5; i2++) {
          for (size_t i3 = 0; i3 < 5; i3++) {
            source_1[i3 + i2*5 + i1*25 + i0*25] = arg_1[i3 + i2*5 + i1*25 + i0*25] >> 14;
          }
        }
      }
    }
    int* arg_2 = (int*)shmat(3473409, nullptr, 0);
    for (size_t i0 = 0; i0 < 50; i0++) {
      for (size_t i1 = 0; i1 < 20; i1++) {
        for (size_t i2 = 0; i2 < 5; i2++) {
          for (size_t i3 = 0; i3 < 5; i3++) {
            source_2[i3 + i2*5 + i1*25 + i0*500] = arg_2[i3 + i2*5 + i1*25 + i0*500] >> 14;
          }
        }
      }
    }
    int* arg_3 = (int*)shmat(2097154, nullptr, 0);
    for (size_t i0 = 0; i0 < 500; i0++) {
      for (size_t i1 = 0; i1 < 800; i1++) {
        source_3[i1 + i0*800] = arg_3[i1 + i0*800] >> 14;
      }
    }
    int* arg_4 = (int*)shmat(1835011, nullptr, 0);
    for (size_t i0 = 0; i0 < 10; i0++) {
      for (size_t i1 = 0; i1 < 500; i1++) {
        source_4[i1 + i0*500] = arg_4[i1 + i0*500] >> 14;
      }
    }
    float* arg_5 = (float*)shmat(1703940, nullptr, 0);
    for (size_t i0 = 0; i0 < 50; i0++) {
      for (size_t i1 = 0; i1 < 10; i1++) {
        source_5[i1 + i0*10] = arg_5[i1 + i0*10];
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
    auto default_function = cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(kernel);

    cl::Buffer buffer_0(context, CL_MEM_READ_WRITE, vector_size_bytes_0);
    cl::Buffer buffer_1(context, CL_MEM_READ_WRITE, vector_size_bytes_1);
    cl::Buffer buffer_2(context, CL_MEM_READ_WRITE, vector_size_bytes_2);
    cl::Buffer buffer_3(context, CL_MEM_READ_WRITE, vector_size_bytes_3);
    cl::Buffer buffer_4(context, CL_MEM_READ_WRITE, vector_size_bytes_4);
    cl::Buffer buffer_5(context, CL_MEM_READ_WRITE, vector_size_bytes_5);

    q.enqueueWriteBuffer(buffer_0, CL_TRUE, 0, vector_size_bytes_0, source_0.data());
    q.enqueueWriteBuffer(buffer_1, CL_TRUE, 0, vector_size_bytes_1, source_1.data());
    q.enqueueWriteBuffer(buffer_2, CL_TRUE, 0, vector_size_bytes_2, source_2.data());
    q.enqueueWriteBuffer(buffer_3, CL_TRUE, 0, vector_size_bytes_3, source_3.data());
    q.enqueueWriteBuffer(buffer_4, CL_TRUE, 0, vector_size_bytes_4, source_4.data());
    q.enqueueWriteBuffer(buffer_5, CL_TRUE, 0, vector_size_bytes_5, source_5.data());

    default_function(cl::EnqueueArgs(q, cl::NDRange(1,1,1), cl::NDRange(1,1,1)),buffer_0, buffer_1, buffer_2, buffer_3, buffer_4, buffer_5);
    q.finish();

    q.enqueueReadBuffer(buffer_0, CL_TRUE, 0, vector_size_bytes_0, source_0.data());
    q.enqueueReadBuffer(buffer_1, CL_TRUE, 0, vector_size_bytes_1, source_1.data());
    q.enqueueReadBuffer(buffer_2, CL_TRUE, 0, vector_size_bytes_2, source_2.data());
    q.enqueueReadBuffer(buffer_3, CL_TRUE, 0, vector_size_bytes_3, source_3.data());
    q.enqueueReadBuffer(buffer_4, CL_TRUE, 0, vector_size_bytes_4, source_4.data());
    q.enqueueReadBuffer(buffer_5, CL_TRUE, 0, vector_size_bytes_5, source_5.data());

    for (size_t i0 = 0; i0 < 50; i0++) {
      for (size_t i1 = 0; i1 < 1; i1++) {
        for (size_t i2 = 0; i2 < 28; i2++) {
          for (size_t i3 = 0; i3 < 28; i3++) {
            arg_0[i3 + i2*28 + i1*784 + i0*784] = source_0[i3 + i2*28 + i1*784 + i0*784];
          }
        }
      }
    }
    shmdt(arg_0);
    for (size_t i0 = 0; i0 < 20; i0++) {
      for (size_t i1 = 0; i1 < 1; i1++) {
        for (size_t i2 = 0; i2 < 5; i2++) {
          for (size_t i3 = 0; i3 < 5; i3++) {
            arg_1[i3 + i2*5 + i1*25 + i0*25] = source_1[i3 + i2*5 + i1*25 + i0*25] << 14;
          }
        }
      }
    }
    shmdt(arg_1);
    for (size_t i0 = 0; i0 < 50; i0++) {
      for (size_t i1 = 0; i1 < 20; i1++) {
        for (size_t i2 = 0; i2 < 5; i2++) {
          for (size_t i3 = 0; i3 < 5; i3++) {
            arg_2[i3 + i2*5 + i1*25 + i0*500] = source_2[i3 + i2*5 + i1*25 + i0*500] << 14;
          }
        }
      }
    }
    shmdt(arg_2);
    for (size_t i0 = 0; i0 < 500; i0++) {
      for (size_t i1 = 0; i1 < 800; i1++) {
        arg_3[i1 + i0*800] = source_3[i1 + i0*800] << 14;
      }
    }
    shmdt(arg_3);
    for (size_t i0 = 0; i0 < 10; i0++) {
      for (size_t i1 = 0; i1 < 500; i1++) {
        arg_4[i1 + i0*500] = source_4[i1 + i0*500] << 14;
      }
    }
    shmdt(arg_4);
    for (size_t i0 = 0; i0 < 50; i0++) {
      for (size_t i1 = 0; i1 < 10; i1++) {
        arg_5[i1 + i0*10] = source_5[i1 + i0*10];
      }
    }
    shmdt(arg_5);
}
