#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
void default_function(ap_int<32> placeholder6[320*32], ap_int<32> placeholder7[16*32], ap_int<32> labels[320]) {
  #pragma HLS array_partition variable=placeholder7 complete dim=0
  #pragma HLS array_partition variable=placeholder6 cyclic factor=32
  #pragma HLS array_partition variable=labels complete
  ap_int<32> _top;
  for (ap_int<32> x = 0; x < 320; ++x) {
    labels[x] = 0;
  }
  ap_int<32> main_loop;
  for (ap_int<32> _ = 0; _ < 200; ++_) {
    for (ap_int<32> N = 0; N < 320; ++N) {
    #pragma HLS pipeline
      ap_int<32> scalar9;
      scalar9 = 100000;
      for (ap_int<32> i = 0; i < 16; ++i) {
        ap_int<32> scalar10;
        scalar10 = 0;
        for (ap_int<32> i1 = 0; i1 < 32; ++i1) {
          ap_int<32> scalar11;
          scalar11 = (placeholder6[(i1 + (N * 32))] - placeholder7[(i1 + (i * 32))]);
          scalar10 = ((ap_int<32>)(((ap_int<65>)scalar10) + ((ap_int<65>)(((ap_int<64>)scalar11) * ((ap_int<64>)scalar11)))));
        }
        if (scalar10 < scalar9) {
          scalar9 = scalar10;
          labels[N] = i;
        }
      }
    }
    ap_int<32> compute6[16];
    for (ap_int<32> x1 = 0; x1 < 16; ++x1) {
      compute6[x1] = 0;
    }
    ap_int<32> compute7[512];
    for (ap_int<32> x2 = 0; x2 < 16; ++x2) {
      for (ap_int<32> y = 0; y < 32; ++y) {
        compute7[(y + (x2 * 32))] = 0;
      }
    }
    ap_int<32> calc_sum;
    for (ap_int<32> n = 0; n < 320; ++n) {
    #pragma HLS unroll
      int index = labels[n];
      compute6[index] = (compute6[index] + 1);
      for (ap_int<32> i2 = 0; i2 < 32; ++i2) {
        compute7[(i2 + (index * 32))] = ((ap_int<32>)(((ap_int<33>)compute7[(i2 + (index * 32))]) + ((ap_int<33>)placeholder6[(i2 + (n * 32))])));
      }
    }
    ap_int<32> update_mean;
    for (ap_int<32> k_d_fused = 0; k_d_fused < 512; ++k_d_fused) {
    #pragma HLS unroll
      placeholder7[k_d_fused] = (compute7[k_d_fused] / compute6[(k_d_fused / 32)]);
    }
  }
}

