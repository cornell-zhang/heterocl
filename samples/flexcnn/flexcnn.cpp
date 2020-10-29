#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
void default_function(ap_uint<32> global_cin[12625160], ap_uint<32> global_prev_cin[12625160], ap_uint<32> global_weight[560032], ap_uint<32> global_bias[16544], ap_uint<32> global_cout[826274], ap_uint<32> config[2789]) {
  float _top;
  float top_kernel;
  i: for (ap_int<32> i = 0; i < 86; ++i) {
    float layer_config[32];
    layer_config_x: for (ap_int<32> x = 0; x < 32; ++x) {
      layer_config[x] = ((float)config[(x + (i * 32))]);
    }
  }
}

