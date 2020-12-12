
#include <sys/ipc.h>
#include <sys/shm.h>

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <cassert>

// rapidjson headers
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"
using namespace rapidjson;

// opencl harness headers
#include "xcl2.hpp"
#include "ap_fixed.h"
#include "ap_int.h"
#include <cmath>
#include <vector>


int main(int argc, char ** argv) {
  std::cout << "[INFO] Initialize input buffers...\n";

  FILE *f = fopen("inputs.json", "r");
  char readBuffer[65536];
  FileReadStream is(f, readBuffer, sizeof(readBuffer));

  Document document;
  document.ParseStream(is);
  fclose(f);
  assert(document.HasMember("hcl_in_train"));
  const Value& hcl_in_train_d = document["hcl_in_train"];
  assert(hcl_in_train_d.IsArray());
  auto hcl_in_train = new ap_uint<64>[6238][156];
  for (size_t i0 = 0; i0 < 6238; i0++) {
    for (size_t i1 = 0; i1 < 156; i1++) {
      hcl_in_train[i0][i1] = (hcl_in_train_d[i1 + i0*156].GetInt());
    }
  }

  assert(document.HasMember("hcl_trainLabels"));
  const Value& hcl_trainLabels_d = document["hcl_trainLabels"];
  assert(hcl_trainLabels_d.IsArray());
  auto hcl_trainLabels = new ap_int<32>[6238];
  for (size_t i0 = 0; i0 < 6238; i0++) {
    hcl_trainLabels[i0] = (hcl_trainLabels_d[i0].GetInt());
  }

  assert(document.HasMember("hcl_in_test"));
  const Value& hcl_in_test_d = document["hcl_in_test"];
  assert(hcl_in_test_d.IsArray());
  auto hcl_in_test = new ap_uint<64>[1559][156];
  for (size_t i0 = 0; i0 < 1559; i0++) {
    for (size_t i1 = 0; i1 < 156; i1++) {
      hcl_in_test[i0][i1] = (hcl_in_test_d[i1 + i0*156].GetInt());
    }
  }

  assert(document.HasMember("hcl_testLabels"));
  const Value& hcl_testLabels_d = document["hcl_testLabels"];
  assert(hcl_testLabels_d.IsArray());
  auto hcl_testLabels = new ap_int<32>[1559];
  for (size_t i0 = 0; i0 < 1559; i0++) {
    hcl_testLabels[i0] = (hcl_testLabels_d[i0].GetInt());
  }

  assert(document.HasMember("hcl_rdv3"));
  const Value& hcl_rdv3_d = document["hcl_rdv3"];
  assert(hcl_rdv3_d.IsArray());
  auto hcl_rdv3 = new ap_int<32>[26][9984];
  for (size_t i0 = 0; i0 < 26; i0++) {
    for (size_t i1 = 0; i1 < 9984; i1++) {
      hcl_rdv3[i0][i1] = (hcl_rdv3_d[i1 + i0*9984].GetInt());
    }
  }

  assert(document.HasMember("hcl_epoch"));
  const Value& hcl_epoch_d = document["hcl_epoch"];
  assert(hcl_epoch_d.IsArray());
  auto hcl_epoch = new ap_int<32>[1];
  for (size_t i0 = 0; i0 < 1; i0++) {
    hcl_epoch[i0] = (hcl_epoch_d[i0].GetInt());
  }

  std::cout << "[INFO] Initialize RTE...\n";

  if (argc != 2) {
      std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
      return EXIT_FAILURE;
  }

  auto binaryFile = argv[1];
  cl_int err = CL_SUCCESS;

  // create binary file and program
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

  cl::Context context;
  cl::CommandQueue q;
  cl::Program program;
  auto devices = xcl::get_xil_devices();
  int valid_device = 0;

  for (unsigned int i = 0; i < devices.size(); i++) {
      auto device = devices[i];
      // Creating Context and Command Queue for selected Device
      context = cl::Context(device, NULL, NULL, NULL, &err);
      q = cl::CommandQueue(
          context, device, CL_QUEUE_PROFILING_ENABLE, &err);

      std::cout << "Trying to program device[" << i
                << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
      program = cl::Program(context, {device}, bins, NULL, &err);
      if (err != CL_SUCCESS) {
          std::cout << "Failed to program device[" << i
                    << "] with xclbin file!\n";
      } else {
          std::cout << "Device[" << i << "]: program successful!\n";
          valid_device++;
          break; // we break because we found a valid device
      }
  }
  if (valid_device == 0) {
      std::cout << "Failed to program any device found, exit!\n";
      exit(EXIT_FAILURE);
  }


  // Compute and kernel call from host
  ap_int<32> __device_scope;

  cl::Kernel kernel(program, "test", &err);
  cl::Buffer buffer_hcl_rdv3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_int<32>)*26*9984, hcl_rdv3, &err);
  cl::Buffer buffer_hcl_trainLabels(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_int<32>)*6238, hcl_trainLabels, &err);
  cl::Buffer buffer_hcl_in_train(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_uint<64>)*6238*156, hcl_in_train, &err);
  cl::Buffer buffer_hcl_in_test(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_uint<64>)*1559*156, hcl_in_test, &err);
  cl::Buffer buffer_hcl_testLabels(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_int<32>)*1559, hcl_testLabels, &err);
  cl::Buffer buffer_hcl_epoch(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_int<32>)*1, hcl_epoch, &err);

  // set device kernel buffer
  err = kernel.setArg(0, buffer_hcl_rdv3);
  err = kernel.setArg(1, buffer_hcl_trainLabels);
  err = kernel.setArg(2, buffer_hcl_in_train);
  err = kernel.setArg(3, buffer_hcl_in_test);
  err = kernel.setArg(4, buffer_hcl_testLabels);
  err = kernel.setArg(5, buffer_hcl_epoch);
  err = q.enqueueMigrateMemObjects({buffer_hcl_rdv3, buffer_hcl_trainLabels, buffer_hcl_in_train, buffer_hcl_in_test, buffer_hcl_testLabels, buffer_hcl_epoch}, 0/*from host*/);
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
  err = q.enqueueMigrateMemObjects({buffer_hcl_rdv3, buffer_hcl_trainLabels, buffer_hcl_in_train, buffer_hcl_in_test, buffer_hcl_testLabels, buffer_hcl_epoch}, CL_MIGRATE_MEM_OBJECT_HOST);

  // execution on host 

  rapidjson::Document::AllocatorType& allocator = document.GetAllocator();
  document["hcl_in_train"].Clear();
  rapidjson::Value v_hcl_in_train(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 6238; i0++) {
    for (size_t i1 = 0; i1 < 156; i1++) {
      v_hcl_in_train.PushBack(rapidjson::Value().SetInt(hcl_in_train[i0][i1]), allocator);
    }
  }
  document["hcl_in_train"] = v_hcl_in_train;
  document["hcl_trainLabels"].Clear();
  rapidjson::Value v_hcl_trainLabels(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 6238; i0++) {
    v_hcl_trainLabels.PushBack(rapidjson::Value().SetInt(hcl_trainLabels[i0]), allocator);
  }
  document["hcl_trainLabels"] = v_hcl_trainLabels;
  document["hcl_in_test"].Clear();
  rapidjson::Value v_hcl_in_test(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 1559; i0++) {
    for (size_t i1 = 0; i1 < 156; i1++) {
      v_hcl_in_test.PushBack(rapidjson::Value().SetInt(hcl_in_test[i0][i1]), allocator);
    }
  }
  document["hcl_in_test"] = v_hcl_in_test;
  document["hcl_testLabels"].Clear();
  rapidjson::Value v_hcl_testLabels(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 1559; i0++) {
    v_hcl_testLabels.PushBack(rapidjson::Value().SetInt(hcl_testLabels[i0]), allocator);
  }
  document["hcl_testLabels"] = v_hcl_testLabels;
  document["hcl_rdv3"].Clear();
  rapidjson::Value v_hcl_rdv3(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 26; i0++) {
    for (size_t i1 = 0; i1 < 9984; i1++) {
      v_hcl_rdv3.PushBack(rapidjson::Value().SetInt(hcl_rdv3[i0][i1]), allocator);
    }
  }
  document["hcl_rdv3"] = v_hcl_rdv3;
  document["hcl_epoch"].Clear();
  rapidjson::Value v_hcl_epoch(rapidjson::kArrayType);
  for (size_t i0 = 0; i0 < 1; i0++) {
    v_hcl_epoch.PushBack(rapidjson::Value().SetInt(hcl_epoch[i0]), allocator);
  }
  document["hcl_epoch"] = v_hcl_epoch;

  FILE* fp = fopen("inputs.json", "w"); 
 
  char writeBuffer[65536];
  FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
 
  Writer<FileWriteStream> writer(os);
  document.Accept(writer);
  fclose(fp);

  

  }
