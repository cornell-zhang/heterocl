#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
void test(ap_int<32> ret[10], ap_int<32> A[10], ap_int<32> B[10]) {
  #pragma HLS INTERFACE m_axi port=ret offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem0
  #pragma HLS INTERFACE s_axilite port=ret bundle=control
  #pragma HLS INTERFACE s_axilite port=A bundle=control
  #pragma HLS INTERFACE s_axilite port=B bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control
    ap_int<32> vadd;
    #include <ap_int.h>
#include <ap_fixed.h>

extern "C" {
    void vadd(ap_int<32>* A, ap_int<32>* B, ap_int<32>* ret) {
        for (size_t k = 0; k < length; k++) {
            ret[k] = A[k] + B[k];
        }
    }
}
  }
void default_function(ap_int<32> A[10], ap_int<32> B[10], ap_int<32> out[10]) {
  ap_int<32> _top;
  ap_int<32> ret[10];
  test(ret, A, B);
  for (ap_int<32> args = 0; args < 10; ++args) {
    out[args] = (ret[args] * 2);
  }
}

