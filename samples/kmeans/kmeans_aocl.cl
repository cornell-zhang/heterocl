#include "ihc_apint.h"
__kernel void default_function(__global int* restrict placeholder2, __global int* restrict placeholder3, __global int* restrict compute3) {
  for (int x = 0; x < 32; ++x) {
    compute3[x] = 0;
  }
  int main_loop;
  for (int _1 = 0; _1 < 10; ++_1) {
    #pragma ii 1
    for (int N = 0; N < 32; ++N) {
      int local2;
      local2 = 100000;
      for (int i = 0; i < 6; ++i) {
        int local3;
        local3 = 0;
        for (int i1 = 0; i1 < 3; ++i1) {
          local3 = ((int)(((int64_t)local3) + ((int64_t)(((int64_t)((int33_t)(placeholder2[(i1 + (N * 3))] - placeholder3[(i1 + (i * 3))]))) * ((int64_t)((int33_t)(placeholder2[(i1 + (N * 3))] - placeholder3[(i1 + (i * 3))])))))));
        }
        if (local3 < local2) {
          local2 = local3;
          compute3[N] = i;
        }
      }
    }
    int compute4[6];
    for (int x1 = 0; x1 < 6; ++x1) {
      compute4[x1] = 0;
    }
    int compute5[18];
    for (int x2 = 0; x2 < 6; ++x2) {
      for (int y = 0; y < 3; ++y) {
        compute5[(y + (x2 * 3))] = 0;
      }
    }
    int calc_sum;
    #pragma unroll
    for (int n = 0; n < 32; ++n) {
      compute4[compute3[n]] = (compute4[compute3[n]] + 1);
      for (int i2 = 0; i2 < 3; ++i2) {
        compute5[(i2 + (compute3[n] * 3))] = ((int)(((int33_t)compute5[(i2 + (compute3[n] * 3))]) + ((int33_t)placeholder2[(i2 + (n * 3))])));
      }
    }
    int update_mean;
    #pragma unroll
    for (int k_d_fused = 0; k_d_fused < 18; ++k_d_fused) {
      placeholder3[k_d_fused] = (compute5[k_d_fused] / compute4[(k_d_fused / 3)]);
    }
  }
}

