#include <string.h>
#include <math.h>
#include <assert.h>
#pragma ACCEL kernel
void default_function(int* placeholder2, int* placeholder3, int* compute3) {
  for (int x = 0; x < 320; ++x) {
    compute3[x] = 0;
  }
  int main_loop;
  for (int _1 = 0; _1 < 200; ++_1) {
#pragma ACCEL pipeline
    for (int N = 0; N < 320; ++N) {
      int local2;
      local2 = 100000;
      for (int i = 0; i < 16; ++i) {
        int local3;
        local3 = 0;
        for (int i1 = 0; i1 < 32; ++i1) {
          local3 = ((int)(((long)local3) + ((long)(((long)((long)(placeholder2[(i1 + (N * 32))] - placeholder3[(i1 + (i * 32))]))) * ((long)((long)(placeholder2[(i1 + (N * 32))] - placeholder3[(i1 + (i * 32))])))))));
        }
        if (local3 < local2) {
          local2 = local3;
          compute3[N] = i;
        }
      }
    }
    int compute4[16];
    for (int x1 = 0; x1 < 16; ++x1) {
      compute4[x1] = 0;
    }
    int compute5[512];
    for (int x2 = 0; x2 < 16; ++x2) {
      for (int y = 0; y < 32; ++y) {
        compute5[(y + (x2 * 32))] = 0;
      }
    }
    int calc_sum;
#pragma ACCEL parallel flatten
    for (int n = 0; n < 320; ++n) {
      compute4[compute3[n]] = (compute4[compute3[n]] + 1);
      for (int i2 = 0; i2 < 32; ++i2) {
        compute5[(i2 + (compute3[n] * 32))] = ((int)(((long)compute5[(i2 + (compute3[n] * 32))]) + ((long)placeholder2[(i2 + (n * 32))])));
      }
    }
    int update_mean;
#pragma ACCEL parallel flatten
    for (int k_d_fused = 0; k_d_fused < 512; ++k_d_fused) {
      placeholder3[k_d_fused] = (compute5[k_d_fused] / compute4[(k_d_fused / 32)]);
    }
  }
}

