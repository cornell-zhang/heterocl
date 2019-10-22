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
// harness namespace
using namespace rosetta;


//other headers
#include "utils.h"
#include "typedefs.h"
int main(int argc, char ** argv) {
  uint64_t arg_0 = (uint64_t)207249344512;
  uint64_t arg_top_0[1] = { arg_0 };


  uint64_t* arg_1 = (uint64_t*)shmat(90701824, nullptr, 0);
  uint64_t arg_top_1[10 * 1800];
  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 1800; i1++) {
      arg_top_1[i1 + i0*1800] = (uint64_t)(arg_1[i1 + i0*1800]);
    }
  }


  uint8_t* arg_2 = (uint8_t*)shmat(90734593, nullptr, 0);
  uint8_t arg_top_2[10];
  for (size_t i0 = 0; i0 < 10; i0++) {
    arg_top_2[i0] = (uint8_t)(arg_2[i0]);
  }


  printf("Digit Recognition Application\n");

  // compute bofore kernel function
   
  // parse command line arguments for opencl version
  std::string kernelFile("");
  parse_sdaccel_command_line_args(argc, argv, kernelFile);


  // create OpenCL world
  CLWorld digit_rec_world = CLWorld(TARGET_DEVICE, CL_DEVICE_TYPE_ACCELERATOR);


  // add the bitstream file
  digit_rec_world.addProgram(kernelFile);


  // create kernels
  CLKernel App(digit_rec_world.getContext(), digit_rec_world.getProgram(), "App", digit_rec_world.getDevice());


  // create mem objects
  CLMemObj source_0((void*)arg_top_0, sizeof(uint64_t), 1, CL_MEM_READ_WRITE);
  CLMemObj source_1((void*)arg_top_1, sizeof(uint64_t), 10 * 1800, CL_MEM_READ_WRITE);
  CLMemObj source_2((void*)arg_top_2, sizeof(uint8_t), 10 , CL_MEM_READ_WRITE);


  // add them to the world
  digit_rec_world.addMemObj(source_0);
  digit_rec_world.addMemObj(source_1);
  digit_rec_world.addMemObj(source_2);


   // set work size
  int global_size[3] = {1, 1, 1};
  int local_size[3] = {1, 1, 1};
  App.set_global(global_size);
  App.set_local(local_size);


  // add them to the world
  digit_rec_world.addKernel(App);


  // set kernel arguments
  digit_rec_world.setMemKernelArg(0, 0, 0);
  digit_rec_world.setMemKernelArg(0, 1, 1);
  digit_rec_world.setMemKernelArg(0, 2, 2);

  // run
  digit_rec_world.runKernels();

  // read the data back
  digit_rec_world.readMemObj(2);

  // compute after kernel function
  for (int x = 0; x < 10; ++x) {
    int id0;
    id0 = 0;
    int id1;
    id1 = 0;
    int id2;
    id2 = 0;
    int count;
    count = 0;
    for (int i = 0; i < 10; ++i) {
      if (knn_mat[(i * 3)] < knn_mat[(id0 * 3)]) {
        id0 = i;
      }
    }
    for (int i1 = 0; i1 < 10; ++i1) {
      if (knn_mat[(i1 * 3)] < knn_mat[(id1 * 3)]) {
        id1 = i1;
      }
    }
    for (int i2 = 0; i2 < 10; ++i2) {
      if (knn_mat[(i2 * 3)] < knn_mat[(id2 * 3)]) {
        id2 = i2;
      }
    }
    if (x == id0) {
      count = (count + 1);
    } else {
      if (x == id1) {
        count = (count + 1);
      } else {
        if (x == id2) {
          count = (count + 1);
        }
      }
    }
    arg_top_2[x] = count;
  }

  for (size_t i0 = 0; i0 < 10; i0++) {
    for (size_t i1 = 0; i1 < 1800; i1++) {
      arg_1[i1 + i0*1800] = (uint64_t)(arg_top_1[i1 + i0*1800]);
    }
  }
  shmdt(arg_1);
  for (size_t i0 = 0; i0 < 10; i0++) {
    arg_2[i0] = (uint8_t)(arg_top_2[i0]);
  }
  shmdt(arg_2);


  }
