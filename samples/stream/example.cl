#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable
channel int ret_add_c;
channel int ret_mul_c;
__kernel void ret_add(__global int* restrict ret_add_a, __global int* restrict ret_add_b) {
    for (int i = 0; i < 10; ++i) {
      for (int i1 = 0; i1 < 20; ++i1) {
        write_channel_intel(ret_add_c, ((int)(((int33_t)ret_add_a[(i1 + (i * 20))]) + ((int33_t)ret_add_b[(i1 + (i * 20))]))));
      }
    }
}

__kernel void ret_mul(__global int* restrict ret_mul_d, __global int* restrict ret_mul_e) {
    for (int i = 0; i < 10; ++i) {
      for (int i1 = 0; i1 < 20; ++i1) {
        ret_mul_e[(i1 + (i * 20))] = ((int)(((long)read_channel_intel(ret_mul_c)) * ((long)ret_mul_d[(i1 + (i * 20))])));
      }
    }
}

__kernel void default_function(__global int* restrict a, __global int* restrict b, __global int* restrict c, __global int* restrict d, __global int* restrict e) {
  int ret_add;
  int ret_mul;
  for (int x = 0; x < 10; ++x) {
    for (int y = 0; y < 20; ++y) {
      c[(y + (x * 20))] = 0;
    }
  }
  int ret_add0;
  ret_add(a, b);
  int ret_mul0;
  ret_mul(d, e);
}

