__kernel void default_function(__global int* placeholder4, __global int* placeholder5, __global int* compute6) {
  for (int x = 0; x < 320; ++x) {
    compute6[x] = 0;
  }
  __local int main_loop;
  for (int _1 = 0; _1 < 200; ++_1) {
    __attribute__((xcl_pipeline_loop(1)))
    for (int N = 0; N < 320; ++N) {
      __local int local4;
      local4 = 100000;
      for (int i = 0; i < 16; ++i) {
        __local int local5;
        local5 = 0;
        for (int i1 = 0; i1 < 32; ++i1) {
          local5 = ((int)(((long)local5) + ((long)(((long)((long)(placeholder4[(i1 + (N * 32))] - placeholder5[(i1 + (i * 32))]))) * ((long)((long)(placeholder4[(i1 + (N * 32))] - placeholder5[(i1 + (i * 32))])))))));
        }
        if (local5 < local4) {
          local4 = local5;
          compute6[N] = i;
        }
      }
    }
    __local int compute7[16];
    for (int x1 = 0; x1 < 16; ++x1) {
      compute7[x1] = 0;
    }
    __local int compute8[512];
    for (int x2 = 0; x2 < 16; ++x2) {
      for (int y = 0; y < 32; ++y) {
        compute8[(y + (x2 * 32))] = 0;
      }
    }
    __local int calc_sum;
    
    for (int n = 0; n < 320; ++n) {
      compute7[compute6[n]] = (compute7[compute6[n]] + 1);
      for (int i2 = 0; i2 < 32; ++i2) {
        compute8[(i2 + (compute6[n] * 32))] = ((int)(((long)compute8[(i2 + (compute6[n] * 32))]) + ((long)placeholder4[(i2 + (n * 32))])));
      }
    }
    __local int update_mean;
    
    for (int k_d_fused = 0; k_d_fused < 512; ++k_d_fused) {
      placeholder5[k_d_fused] = (compute8[k_d_fused] / compute7[(k_d_fused / 32)]);
    }
  }
}

