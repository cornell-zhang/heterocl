
#include <gmp.h>
#define __gmp_const const
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

// vivado hls headers
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include "kernel.h"

#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>

int main(int argc, char ** argv) {
  std::cout << " Initialize shared memory...\n";
auto input_image = new ap_fixed<32,20>[1][3][32][32];

auto fc = new ap_fixed<32,20>[1][10];

  std::cout << " Initialize RTE...\n";

  // compute and kernel call from host
  ap_int<32> __device_scope;
  test(input_image, fc);



  }
