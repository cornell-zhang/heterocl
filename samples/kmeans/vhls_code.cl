#include <ap_int.h>
#include <ap_fixed.h>
#include <math.h>

void default_function(ap_int<32> placeholder6[320][32], ap_int<32> placeholder7[16][32], ap_int<32> compute9[320]) {
  for (ap_int<32> x = 0; x < 320; ++x) {
    compute9[x] = 0;
  }
  ap_int<32> main_loop;
  for (ap_int<32> _ = 0; _ < 200; ++_) {
    for (ap_int<32> N = 0; N < 320; ++N) {
    #pragma HLS pipeline
      ap_int<32> local6;
      local6 = 100000;
      for (ap_int<32> i = 0; i < 16; ++i) {
        ap_int<32> local7;
        local7 = 0;
        for (ap_int<32> i1 = 0; i1 < 32; ++i1) {
          local7 = ((ap_int<32>)(((ap_int<67>)local7) + ((ap_int<67>)(((ap_int<66>)((ap_int<33>)(placeholder6[N][i1] - placeholder7[i][i1]))) * ((ap_int<66>)((ap_int<33>)(placeholder6[N][i1] - placeholder7[i][i1])))))));
        }
        if (local7 < local6) {
          local6 = local7;
          compute9[N] = i;
        }
      }
    }
    ap_int<32> compute10[16];
    for (ap_int<32> x1 = 0; x1 < 16; ++x1) {
      compute10[x1] = 0;
    }
    ap_int<32> compute11[16][32];
    for (ap_int<32> x2 = 0; x2 < 16; ++x2) {
      for (ap_int<32> y = 0; y < 32; ++y) {
        compute11[x2][y] = 0;
      }
    }
    ap_int<32> calc_sum;
    for (ap_int<32> n = 0; n < 320; ++n) {
    #pragma HLS unroll
      compute10[compute9[n]] = (compute10[compute9[n]] + 1);
      for (ap_int<32> i2 = 0; i2 < 32; ++i2) {
        compute11[compute9[n]][i2] = ((ap_int<32>)(((ap_int<33>)compute11[compute9[n]][i2]) + ((ap_int<33>)placeholder6[n][i2])));
      }
    }
    ap_int<32> update_mean;
    for (ap_int<32> k_d_fused = 0; k_d_fused < 512; ++k_d_fused) {
    #pragma HLS unroll
      placeholder7[(k_d_fused / 32)][(k_d_fused % 32)] = (compute11[(k_d_fused / 32)][(k_d_fused % 32)] / compute10[(k_d_fused / 32)]);
    }
  }
}

