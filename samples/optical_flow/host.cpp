
#include <sys/ipc.h>
#include <sys/shm.h>

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <time.h>
#include <sys/time.h>

// opencl harness headers
#include "CLWorld.h"
#include "CLKernel.h"
#include "CLMemObj.h"
#include "utils.h"
#include "ap_fixed.h"
#include "Image.h"
#include "Error.h"
#include "ImageIO.h"
#include "Convert.h"
#include <cmath>

// harness namespace
using namespace rosetta;

int main(int argc, char ** argv) {
                
  CByteImage arg_0;
  ReadImage(arg_0, "datasets/current/frame1.ppm");
  arg_0 = ConvertToGray(arg_0);
                
  int32_t* input_image0 = new int32_t[436 * 1024];
  for (size_t i0 = 0; i0 < 436; i0++) {
    for (size_t i1 = 0; i1 < 1024; i1++) {
      input_image0[i1 + i0*1024] = arg_0.Pixel(i1,i0,0);
    }
  }

  CByteImage arg_1;
  ReadImage(arg_1, "datasets/current/frame2.ppm");
  arg_1 = ConvertToGray(arg_1);
  int32_t* input_image1 = new int32_t[436 * 1024];
  for (size_t i0 = 0; i0 < 436; i0++) {
    for (size_t i1 = 0; i1 < 1024; i1++) {
      input_image1[i1 + i0*1024] = arg_1.Pixel(i1,i0,0);
    }
  }

  CByteImage arg_2;
  ReadImage(arg_2, "datasets/current/frame3.ppm");
  arg_2 = ConvertToGray(arg_2);
  int32_t* input_image2 = new int32_t[436 * 1024];
  for (size_t i0 = 0; i0 < 436; i0++) {
    for (size_t i1 = 0; i1 < 1024; i1++) {
      input_image2[i1 + i0*1024] = arg_2.Pixel(i1,i0,0);
    }
  }

  CByteImage arg_3;
  ReadImage(arg_3, "datasets/current/frame3.ppm");
  arg_3 = ConvertToGray(arg_3);
  int32_t* input_image2_0 = new int32_t[436 * 1024];
  for (size_t i0 = 0; i0 < 436; i0++) {
    for (size_t i1 = 0; i1 < 1024; i1++) {
      input_image2_0[i1 + i0*1024] = arg_3.Pixel(i1,i0,0);
    }
  }

  CByteImage arg_4;
  ReadImage(arg_4, "datasets/current/frame3.ppm");
  arg_4 = ConvertToGray(arg_4);
  int32_t* input_image2_1 = new int32_t[436 * 1024];
  for (size_t i0 = 0; i0 < 436; i0++) {
    for (size_t i1 = 0; i1 < 1024; i1++) {
      input_image2_1[i1 + i0*1024] = arg_4.Pixel(i1,i0,0);
    }
  }

  CByteImage arg_5;
  ReadImage(arg_5, "datasets/current/frame4.ppm");
  arg_5 = ConvertToGray(arg_5);
  int32_t* input_image3 = new int32_t[436 * 1024];
  for (size_t i0 = 0; i0 < 436; i0++) {
    for (size_t i1 = 0; i1 < 1024; i1++) {
      input_image3[i1 + i0*1024] = arg_5.Pixel(i1,i0,0);
    }
  }

  CByteImage arg_6;
  ReadImage(arg_6, "datasets/current/frame5.ppm");
  arg_6 = ConvertToGray(arg_6);
  int32_t* input_image4 = new int32_t[436 * 1024];
  for (size_t i0 = 0; i0 < 436; i0++) {
    for (size_t i1 = 0; i1 < 1024; i1++) {
      input_image4[i1 + i0*1024] = arg_6.Pixel(i1,i0,0);
    }
  }

  int32_t* arg_7 = new int32_t[463 * 1024];
  int32_t* output_image_0 = new int32_t[463 * 1024];
  for (size_t i0 = 0; i0 < 463; i0++) {
    for (size_t i1 = 0; i1 < 1024; i1++) {
        output_image_0[i1 + i0*1024] = 0;
    }
  }

  int32_t* arg_8 = new int32_t[463 * 1024];
  int32_t* output_image_1 = new int32_t[463 * 1024];
  for (size_t i0 = 0; i0 < 463; i0++) {
    for (size_t i1 = 0; i1 < 1024; i1++) {
        output_image_1[i1 + i0*1024] = 0;
    }
  }

  // parse command line arguments for opencl version 
  std::string kernelFile("");
  parse_sdaccel_command_line_args(argc, argv, kernelFile);
 
  // create OpenCL world
  CLWorld world = CLWorld(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);
  world.addProgram(kernelFile);

  // compute and kernel call from host
                
  CLKernel top_function_0(world.getContext(), world.getProgram(), "top_function_0", world.getDevice());
  CLMemObj source_0((void*)input_image0, sizeof(ap_fixed<32, 20>), 436 *1024, CL_MEM_READ_WRITE);
  CLMemObj source_1((void*)input_image1, sizeof(ap_fixed<32, 20>), 436 *1024, CL_MEM_READ_WRITE);
  CLMemObj source_2((void*)input_image2_0, sizeof(ap_fixed<32, 20>), 436 *1024, CL_MEM_READ_WRITE);
  CLMemObj source_3((void*)input_image2_1, sizeof(ap_fixed<32, 20>), 436 *1024, CL_MEM_READ_WRITE);
  CLMemObj source_4((void*)input_image2, sizeof(ap_fixed<32, 20>), 436 *1024, CL_MEM_READ_WRITE);
  CLMemObj source_5((void*)input_image3, sizeof(ap_fixed<32, 20>), 436 *1024, CL_MEM_READ_WRITE);
  CLMemObj source_6((void*)input_image4, sizeof(ap_fixed<32, 20>), 436 *1024, CL_MEM_READ_WRITE);
  CLMemObj source_7((void*)output_image_0, sizeof(ap_fixed<32, 20>), 436 *1024, CL_MEM_READ_WRITE);
  CLMemObj source_8((void*)output_image_1, sizeof(ap_fixed<32, 20>), 436 *1024, CL_MEM_READ_WRITE);
  world.addMemObj(source_0);
  world.addMemObj(source_1);
  world.addMemObj(source_2);
  world.addMemObj(source_3);
  world.addMemObj(source_4);
  world.addMemObj(source_5);
  world.addMemObj(source_6);
  world.addMemObj(source_7);
  world.addMemObj(source_8);

  int global_size[3] = {1, 1, 1};
  int local_size[3]  = {1, 1, 1};
  top_function_0.set_global(global_size);
  top_function_0.set_local(local_size);
  world.addKernel(top_function_0);

  world.setMemKernelArg(0, 0, 0);
  world.setMemKernelArg(0, 1, 1);
  world.setMemKernelArg(0, 2, 2);
  world.setMemKernelArg(0, 3, 3);
  world.setMemKernelArg(0, 4, 4);
  world.setMemKernelArg(0, 5, 5);
  world.setMemKernelArg(0, 6, 6);
  world.setMemKernelArg(0, 7, 7);
  world.setMemKernelArg(0, 8, 8);

  world.runKernels();
                
  world.readMemObj(0);
  world.readMemObj(1);
  world.readMemObj(2);
  world.readMemObj(3);
  world.readMemObj(4);
  world.readMemObj(5);
  world.readMemObj(6);
  world.readMemObj(7);
  world.readMemObj(8);
  world.releaseWorld();
  
  }
