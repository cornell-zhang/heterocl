// HASH:3850330148
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
typedef ap_int<32> bit32;
typedef ap_uint<32> ubit32;
#include "const.h"

extern "C" {

void test(ap_fixed<32, 20> input_image[1][3][32][32], ap_fixed<32, 20> fc[1][10]) {
    #pragma HLS INTERFACE m_axi port=input_image offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=fc offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=input_image bundle=control
    #pragma HLS INTERFACE s_axilite port=fc bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    bit32 _top;
    ap_fixed<32, 20> conv0_pad[1][3][34][34];
    hls::stream<ap_fixed<32, 20> > conv0_pad_pipe_1;
    #pragma HLS dataflow
    #pragma HLS stream variable=conv0_pad_pipe_1 depth=3468
    conv0_pad_not_zero: for (bit32 not_zero = 0; not_zero < 3; ++not_zero) {
      conv0_pad_index_tuple: for (bit32 index_tuple = 0; index_tuple < 34; ++index_tuple) {
        conv0_pad_i: for (bit32 i = 0; i < 34; ++i) {
      #pragma HLS pipeline
          ap_fixed<32, 20> conv0_pad_temp;
          if (((((1 <= index_tuple) && (index_tuple < 33)) && (1 <= i)) && (i < 33))) { 
            conv0_pad_temp = input_image[(((((i - ((i + -1) % 32)) + (index_tuple * 32)) + (not_zero * 1024)) + -33) / 3072)][((((((i - ((i + -1) % 32)) + (index_tuple * 32)) + (not_zero * 1024)) + -33) / 1024) % 3)][((((((i - ((i + -1) % 32)) + (index_tuple * 32)) + (not_zero * 1024)) + -33) / 32) % 32)][((i + -1) % 32)];
          } else { 
            conv0_pad_temp = (ap_fixed<32, 20>)0;
          }
          conv0_pad_pipe_1.write(conv0_pad_temp);
          conv0_pad[0][not_zero][index_tuple][i] = conv0_pad_temp;
        }
      }
    }
    ap_fixed<32, 20> conv0[1][16][32][32];
    ap_fixed<32, 20> conv0_LB[1][3][3][34];
    ap_fixed<32, 20> conv0_WB[1][3][3][3];
    // #pragma HLS array_partition variable=conv0_WB complete dim=4
    hls::stream<ap_fixed<32, 20> > conv0_pipe_2;
    #pragma HLS stream variable=conv0_pipe_2 depth=16384
    conv0_yy_reuse: for (bit32 yy_reuse = 0; yy_reuse < 34; ++yy_reuse) {
      conv0_xx_reuse: for (bit32 xx_reuse = 0; xx_reuse < 34; ++xx_reuse) {
        loop_conv0_pad_2: for (bit32 conv0_pad_2 = 0; conv0_pad_2 < 3; ++conv0_pad_2) {
          loop_conv0_pad_1: for (bit32 conv0_pad_1 = 0; conv0_pad_1 < 2; ++conv0_pad_1) {
            conv0_LB[0][conv0_pad_2][conv0_pad_1][xx_reuse] = conv0_LB[0][conv0_pad_2][(conv0_pad_1 + 1)][xx_reuse];
          }
          ap_fixed<32, 20> conv0_pad_temp1;
          conv0_pad_temp1 = conv0_pad_pipe_1.read();
          conv0_LB[0][conv0_pad_2][2][xx_reuse] = conv0_pad_temp1;
        }
        if (2 <= yy_reuse) {
          loop_conv0_LB_1: for (bit32 conv0_LB_1 = 0; conv0_LB_1 < 3; ++conv0_LB_1) {
            loop_conv0_LB_2: for (bit32 conv0_LB_2 = 0; conv0_LB_2 < 3; ++conv0_LB_2) {
              loop_conv0_LB_0: for (bit32 conv0_LB_0 = 0; conv0_LB_0 < 2; ++conv0_LB_0) {
                conv0_WB[0][conv0_LB_2][conv0_LB_1][conv0_LB_0] = conv0_WB[0][conv0_LB_2][conv0_LB_1][(conv0_LB_0 + 1)];
              }
              conv0_WB[0][conv0_LB_2][conv0_LB_1][2] = conv0_LB[0][conv0_LB_2][conv0_LB_1][xx_reuse];
            }
          }
            if (2 <= xx_reuse) {
          conv0_ff: for (bit32 ff = 0; ff < 16; ++ff) {
      #pragma HLS pipeline
              ap_fixed<32, 20> sum;
              sum = ((ap_fixed<32, 20>)0);
              conv0_rc: for (bit32 rc = 0; rc < 3; ++rc) {
                conv0_ry: for (bit32 ry = 0; ry < 3; ++ry) {
                  conv0_rx: for (bit32 rx = 0; rx < 3; ++rx) {
                    sum = ((ap_fixed<32, 20>)(((ap_fixed<65, 41>)(((ap_fixed<64, 52>)conv0_WB[0][rc][ry][rx]) * ((ap_fixed<64, 52>)w_conv1[ff][rc][ry][rx]))) + ((ap_fixed<65, 41>)sum)));
                  }
                }
              }
              ap_fixed<32, 20> conv0_temp;
              conv0_temp = sum;
              conv0_pipe_2.write(conv0_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> bn1[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > bn1_pipe_3;
    #pragma HLS stream variable=bn1_pipe_3 depth=16384
    hls::stream<ap_fixed<32, 20> > bn1_pipe_115;
    #pragma HLS stream variable=bn1_pipe_115 depth=16384
    bn1_args0: for (bit32 args0 = 0; args0 < 16; ++args0) {
      bn1_args1: for (bit32 args1 = 0; args1 < 32; ++args1) {
        bn1_args2: for (bit32 args2 = 0; args2 < 32; ++args2) {
        #pragma HLS pipeline
          ap_fixed<32, 20> conv0_temp1;
          conv0_temp1 = conv0_pipe_2.read();
          ap_fixed<32, 20> bn1_temp;
          bn1_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)conv0_temp1) - ((ap_fixed<33, 21>)w_bn1_3[args0]))) / sqrt((((float)w_bn1_4[args0]) + 1.000000e-07f))) * ((float)w_bn1_1[args0])) + ((float)w_bn1_2[args0])));
          bn1_pipe_115.write(bn1_temp);
          bn1_pipe_3.write(bn1_temp);
        }
      }
    }
    ap_uint<16> layer1_0_rsign1[1][1][32][32];
    hls::stream<ap_uint<16> > layer1_0_rsign1_pipe_4;
    #pragma HLS stream variable=layer1_0_rsign1_pipe_4 depth=1024
    layer1_0_rsign1_hh: for (bit32 hh = 0; hh < 32; ++hh) {
      layer1_0_rsign1_ww: for (bit32 ww = 0; ww < 32; ++ww) {
      #pragma HLS pipeline
        ap_uint<16> layer1_0_rsign1_pack;
        layer1_0_rsign1_pack = (ap_uint<16>)0;
        loop_i1: for (bit32 i1 = 0; i1 < 16; ++i1) {
          ap_fixed<32, 20> bn1_temp1;
          bn1_temp1 = bn1_pipe_3.read();
          layer1_0_rsign1_pack(i1, i1) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)bn1_temp1) + ((ap_fixed<33, 21>)w_layer1_0_rsign1[i1])))) ? ((bit32)1) : ((bit32)0));
        }
        ap_uint<16> layer1_0_rsign1_temp;
        layer1_0_rsign1_temp = layer1_0_rsign1_pack;
        layer1_0_rsign1_pipe_4.write(layer1_0_rsign1_temp);
      }
    }
    ap_uint<16> layer1_0_conv1_pad[1][1][34][34];
    hls::stream<ap_uint<16> > layer1_0_conv1_pad_pipe_5;
    #pragma HLS stream variable=layer1_0_conv1_pad_pipe_5 depth=1156
    layer1_0_conv1_pad_hh1: for (bit32 hh1 = 0; hh1 < 34; ++hh1) {
      layer1_0_conv1_pad_ww1: for (bit32 ww1 = 0; ww1 < 34; ++ww1) {
    #pragma HLS pipeline
        ap_uint<16> layer1_0_conv1_pad_temp;
        layer1_0_conv1_pad_temp = ((ap_uint<16>)(((((1 <= ww1) && (ww1 < 33)) && (1 <= hh1)) && (hh1 < 33)) ? (((ubit32)layer1_0_rsign1_pipe_4.read())) : ((ubit32)0U)));
        layer1_0_conv1_pad_pipe_5.write(layer1_0_conv1_pad_temp);
        layer1_0_conv1_pad[0][0][hh1][ww1] = layer1_0_conv1_pad_temp;
      }
    }
    ap_int<8> layer1_0_conv1[1][16][32][32];
    ap_uint<16> layer1_0_conv1_LB[1][1][3][34];
    ap_uint<16> layer1_0_conv1_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer1_0_conv1_WB complete dim=4
    hls::stream<ap_int<8> > layer1_0_conv1_pipe_6;
    #pragma HLS stream variable=layer1_0_conv1_pipe_6 depth=16384
    layer1_0_conv1_yy_reuse1: for (bit32 yy_reuse1 = 0; yy_reuse1 < 34; ++yy_reuse1) {
      layer1_0_conv1_xx_reuse1: for (bit32 xx_reuse1 = 0; xx_reuse1 < 34; ++xx_reuse1) {
        loop_layer1_0_conv1_pad_1: for (bit32 layer1_0_conv1_pad_1 = 0; layer1_0_conv1_pad_1 < 2; ++layer1_0_conv1_pad_1) {
          layer1_0_conv1_LB[0][0][layer1_0_conv1_pad_1][xx_reuse1] = layer1_0_conv1_LB[0][0][(layer1_0_conv1_pad_1 + 1)][xx_reuse1];
        }
        ap_uint<16> layer1_0_conv1_pad_temp1;
        layer1_0_conv1_pad_temp1 = layer1_0_conv1_pad_pipe_5.read();
        layer1_0_conv1_LB[0][0][2][xx_reuse1] = layer1_0_conv1_pad_temp1;
        if (2 <= yy_reuse1) {
          loop_layer1_0_conv1_LB_1: for (bit32 layer1_0_conv1_LB_1 = 0; layer1_0_conv1_LB_1 < 3; ++layer1_0_conv1_LB_1) {
            loop_layer1_0_conv1_LB_0: for (bit32 layer1_0_conv1_LB_0 = 0; layer1_0_conv1_LB_0 < 2; ++layer1_0_conv1_LB_0) {
              layer1_0_conv1_WB[0][0][layer1_0_conv1_LB_1][layer1_0_conv1_LB_0] = layer1_0_conv1_WB[0][0][layer1_0_conv1_LB_1][(layer1_0_conv1_LB_0 + 1)];
            }
            layer1_0_conv1_WB[0][0][layer1_0_conv1_LB_1][2] = layer1_0_conv1_LB[0][0][layer1_0_conv1_LB_1][xx_reuse1];
          }
            if (2 <= xx_reuse1) {
          layer1_0_conv1_ff1: for (bit32 ff1 = 0; ff1 < 16; ++ff1) {
      #pragma HLS pipeline
              ap_int<8> layer1_0_conv1_sum;
              layer1_0_conv1_sum = (ap_int<8>)0;
              layer1_0_conv1_layer1_0_conv1_ry: for (bit32 layer1_0_conv1_ry = 0; layer1_0_conv1_ry < 3; ++layer1_0_conv1_ry) {
                layer1_0_conv1_layer1_0_conv1_rx: for (bit32 layer1_0_conv1_rx = 0; layer1_0_conv1_rx < 3; ++layer1_0_conv1_rx) {
                  layer1_0_conv1_layer1_0_conv1_rb: for (bit32 layer1_0_conv1_rb = 0; layer1_0_conv1_rb < 16; ++layer1_0_conv1_rb) {
                    layer1_0_conv1_sum = ((ap_int<8>)(((ap_int<18>)(layer1_0_conv1_WB[0][0][layer1_0_conv1_ry][layer1_0_conv1_rx] ^ w_layer1_0_conv1[ff1][0][layer1_0_conv1_ry][layer1_0_conv1_rx])[layer1_0_conv1_rb]) + ((ap_int<18>)layer1_0_conv1_sum)));
                  }
                }
              }
              ap_int<8> layer1_0_conv1_temp;
              layer1_0_conv1_temp = ((ap_int<8>)(144 - ((bit32)(layer1_0_conv1_sum << 1))));
              layer1_0_conv1_pipe_6.write(layer1_0_conv1_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer1_0_bn1[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_0_bn1_pipe_7;
    #pragma HLS stream variable=layer1_0_bn1_pipe_7 depth=16384
    layer1_0_bn1_args01: for (bit32 args01 = 0; args01 < 16; ++args01) {
      layer1_0_bn1_args11: for (bit32 args11 = 0; args11 < 32; ++args11) {
        layer1_0_bn1_args21: for (bit32 args21 = 0; args21 < 32; ++args21) {
        #pragma HLS pipeline
          ap_int<8> layer1_0_conv1_temp1;
          layer1_0_conv1_temp1 = layer1_0_conv1_pipe_6.read();
          ap_fixed<32, 20> layer1_0_bn1_temp;
          layer1_0_bn1_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer1_0_conv1_temp1) - ((ap_fixed<33, 21>)w_layer1_0_bn1_11[args01]))) / sqrt((((float)w_layer1_0_bn1_12[args01]) + 1.000000e-07f))) * ((float)w_layer1_0_bn1_9[args01])) + ((float)w_layer1_0_bn1_10[args01])));
          layer1_0_bn1_pipe_7.write(layer1_0_bn1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_0_residual1[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_0_residual1_pipe_8;
    #pragma HLS stream variable=layer1_0_residual1_pipe_8 depth=16384
    layer1_0_residual1_cc: for (bit32 cc = 0; cc < 16; ++cc) {
      layer1_0_residual1_ww2: for (bit32 ww2 = 0; ww2 < 32; ++ww2) {
        layer1_0_residual1_hh2: for (bit32 hh2 = 0; hh2 < 32; ++hh2) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_0_bn1_temp1;
          layer1_0_bn1_temp1 = layer1_0_bn1_pipe_7.read();
          ap_fixed<32, 20> layer1_0_residual1_temp;
          ap_fixed<32, 20> bn1_temp2;
          bn1_temp2 = bn1_pipe_115.read();
          layer1_0_residual1_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer1_0_bn1_temp1) + ((ap_fixed<33, 21>)bn1_temp2)));
          layer1_0_residual1_pipe_8.write(layer1_0_residual1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_0_rprelu1[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_0_rprelu1_pipe_9;
    #pragma HLS stream variable=layer1_0_rprelu1_pipe_9 depth=16384
    hls::stream<ap_fixed<32, 20> > layer1_0_rprelu1_pipe_116;
    #pragma HLS stream variable=layer1_0_rprelu1_pipe_116 depth=16384
    layer1_0_rprelu1_cc1: for (bit32 cc1 = 0; cc1 < 16; ++cc1) {
      layer1_0_rprelu1_ww3: for (bit32 ww3 = 0; ww3 < 32; ++ww3) {
        layer1_0_rprelu1_hh3: for (bit32 hh3 = 0; hh3 < 32; ++hh3) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_0_residual1_temp1;
          layer1_0_residual1_temp1 = layer1_0_residual1_pipe_8.read();
          ap_fixed<32, 20> layer1_0_rprelu1_temp;
          layer1_0_rprelu1_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_0_residual1_temp1) + ((ap_fixed<33, 21>)w_layer1_0_rprelu1_0[cc1])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer1_0_residual1_temp1) + ((ap_fixed<33, 21>)w_layer1_0_rprelu1_0[cc1])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer1_0_rprelu1_2[cc1]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer1_0_residual1_temp1) + ((ap_fixed<33, 21>)w_layer1_0_rprelu1_0[cc1]))))))) + ((ap_fixed<66, 42>)w_layer1_0_rprelu1_1[cc1])));
          layer1_0_rprelu1_pipe_116.write(layer1_0_rprelu1_temp);
          layer1_0_rprelu1_pipe_9.write(layer1_0_rprelu1_temp);
        }
      }
    }
    ap_uint<16> layer1_0_rsign2[1][1][32][32];
    hls::stream<ap_uint<16> > layer1_0_rsign2_pipe_10;
    #pragma HLS stream variable=layer1_0_rsign2_pipe_10 depth=1024
    layer1_0_rsign2_hh4: for (bit32 hh4 = 0; hh4 < 32; ++hh4) {
      layer1_0_rsign2_ww4: for (bit32 ww4 = 0; ww4 < 32; ++ww4) {
      #pragma HLS pipeline
        ap_uint<16> layer1_0_rsign2_pack;
        layer1_0_rsign2_pack = (ap_uint<16>)0;
        loop_i2: for (bit32 i2 = 0; i2 < 16; ++i2) {
          ap_fixed<32, 20> layer1_0_rprelu1_temp1;
          layer1_0_rprelu1_temp1 = layer1_0_rprelu1_pipe_9.read();
          layer1_0_rsign2_pack(i2, i2) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_0_rprelu1_temp1) + ((ap_fixed<33, 21>)w_layer1_0_rsign2[i2])))) ? ((bit32)1) : ((bit32)0));
        }
        ap_uint<16> layer1_0_rsign2_temp;
        layer1_0_rsign2_temp = layer1_0_rsign2_pack;
        layer1_0_rsign2_pipe_10.write(layer1_0_rsign2_temp);
      }
    }
    ap_uint<16> layer1_0_conv2_pad[1][1][34][34];
    hls::stream<ap_uint<16> > layer1_0_conv2_pad_pipe_11;
    #pragma HLS stream variable=layer1_0_conv2_pad_pipe_11 depth=1156
    layer1_0_conv2_pad_hh5: for (bit32 hh5 = 0; hh5 < 34; ++hh5) {
      layer1_0_conv2_pad_ww5: for (bit32 ww5 = 0; ww5 < 34; ++ww5) {
    #pragma HLS pipeline
        ap_uint<16> layer1_0_rsign2_temp1;
        ap_uint<16> layer1_0_conv2_pad_temp;
        layer1_0_conv2_pad_temp = ((ap_uint<16>)(((((1 <= ww5) && (ww5 < 33)) && (1 <= hh5)) && (hh5 < 33)) ? (((ubit32)layer1_0_rsign2_pipe_10.read())) : ((ubit32)0U)));
        layer1_0_conv2_pad_pipe_11.write(layer1_0_conv2_pad_temp);
        layer1_0_conv2_pad[0][0][hh5][ww5] = layer1_0_conv2_pad_temp;
      }
    }
    ap_int<8> layer1_0_conv2[1][16][32][32];
    ap_uint<16> layer1_0_conv2_LB[1][1][3][34];
    ap_uint<16> layer1_0_conv2_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer1_0_conv2_WB complete dim=4
    hls::stream<ap_int<8> > layer1_0_conv2_pipe_12;
    #pragma HLS stream variable=layer1_0_conv2_pipe_12 depth=16384
    layer1_0_conv2_yy_reuse2: for (bit32 yy_reuse2 = 0; yy_reuse2 < 34; ++yy_reuse2) {
      layer1_0_conv2_xx_reuse2: for (bit32 xx_reuse2 = 0; xx_reuse2 < 34; ++xx_reuse2) {
        loop_layer1_0_conv2_pad_1: for (bit32 layer1_0_conv2_pad_1 = 0; layer1_0_conv2_pad_1 < 2; ++layer1_0_conv2_pad_1) {
          layer1_0_conv2_LB[0][0][layer1_0_conv2_pad_1][xx_reuse2] = layer1_0_conv2_LB[0][0][(layer1_0_conv2_pad_1 + 1)][xx_reuse2];
        }
        ap_uint<16> layer1_0_conv2_pad_temp1;
        layer1_0_conv2_pad_temp1 = layer1_0_conv2_pad_pipe_11.read();
        layer1_0_conv2_LB[0][0][2][xx_reuse2] = layer1_0_conv2_pad_temp1;
        if (2 <= yy_reuse2) {
          loop_layer1_0_conv2_LB_1: for (bit32 layer1_0_conv2_LB_1 = 0; layer1_0_conv2_LB_1 < 3; ++layer1_0_conv2_LB_1) {
            loop_layer1_0_conv2_LB_0: for (bit32 layer1_0_conv2_LB_0 = 0; layer1_0_conv2_LB_0 < 2; ++layer1_0_conv2_LB_0) {
              layer1_0_conv2_WB[0][0][layer1_0_conv2_LB_1][layer1_0_conv2_LB_0] = layer1_0_conv2_WB[0][0][layer1_0_conv2_LB_1][(layer1_0_conv2_LB_0 + 1)];
            }
            layer1_0_conv2_WB[0][0][layer1_0_conv2_LB_1][2] = layer1_0_conv2_LB[0][0][layer1_0_conv2_LB_1][xx_reuse2];
          }
            if (2 <= xx_reuse2) {
          layer1_0_conv2_ff2: for (bit32 ff2 = 0; ff2 < 16; ++ff2) {
      #pragma HLS pipeline
              ap_int<8> layer1_0_conv2_sum;
              layer1_0_conv2_sum = (ap_int<8>)0;
              layer1_0_conv2_layer1_0_conv2_ry: for (bit32 layer1_0_conv2_ry = 0; layer1_0_conv2_ry < 3; ++layer1_0_conv2_ry) {
                layer1_0_conv2_layer1_0_conv2_rx: for (bit32 layer1_0_conv2_rx = 0; layer1_0_conv2_rx < 3; ++layer1_0_conv2_rx) {
                  layer1_0_conv2_layer1_0_conv2_rb: for (bit32 layer1_0_conv2_rb = 0; layer1_0_conv2_rb < 16; ++layer1_0_conv2_rb) {
                    layer1_0_conv2_sum = ((ap_int<8>)(((ap_int<18>)(layer1_0_conv2_WB[0][0][layer1_0_conv2_ry][layer1_0_conv2_rx] ^ w_layer1_0_conv2[ff2][0][layer1_0_conv2_ry][layer1_0_conv2_rx])[layer1_0_conv2_rb]) + ((ap_int<18>)layer1_0_conv2_sum)));
                  }
                }
              }
              ap_int<8> layer1_0_conv2_temp;
              layer1_0_conv2_temp = ((ap_int<8>)(144 - ((bit32)(layer1_0_conv2_sum << 1))));
              layer1_0_conv2_pipe_12.write(layer1_0_conv2_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer1_0_bn2[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_0_bn2_pipe_13;
    #pragma HLS stream variable=layer1_0_bn2_pipe_13 depth=16384
    layer1_0_bn2_args02: for (bit32 args02 = 0; args02 < 16; ++args02) {
      layer1_0_bn2_args12: for (bit32 args12 = 0; args12 < 32; ++args12) {
        layer1_0_bn2_args22: for (bit32 args22 = 0; args22 < 32; ++args22) {
        #pragma HLS pipeline
          ap_int<8> layer1_0_conv2_temp1;
          layer1_0_conv2_temp1 = layer1_0_conv2_pipe_12.read();
          ap_fixed<32, 20> layer1_0_bn2_temp;
          layer1_0_bn2_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer1_0_conv2_temp1) - ((ap_fixed<33, 21>)w_layer1_0_bn2_16[args02]))) / sqrt((((float)w_layer1_0_bn2_17[args02]) + 1.000000e-07f))) * ((float)w_layer1_0_bn2_14[args02])) + ((float)w_layer1_0_bn2_15[args02])));
          layer1_0_bn2_pipe_13.write(layer1_0_bn2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_0_residual2[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_0_residual2_pipe_14;
    #pragma HLS stream variable=layer1_0_residual2_pipe_14 depth=16384
    layer1_0_residual2_cc2: for (bit32 cc2 = 0; cc2 < 16; ++cc2) {
      layer1_0_residual2_ww6: for (bit32 ww6 = 0; ww6 < 32; ++ww6) {
        layer1_0_residual2_hh6: for (bit32 hh6 = 0; hh6 < 32; ++hh6) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_0_bn2_temp1;
          layer1_0_bn2_temp1 = layer1_0_bn2_pipe_13.read();
          ap_fixed<32, 20> layer1_0_residual2_temp;
          ap_fixed<32, 20> layer1_0_rprelu1_temp2;
          layer1_0_rprelu1_temp2 = layer1_0_rprelu1_pipe_116.read();
          layer1_0_residual2_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer1_0_bn2_temp1) + ((ap_fixed<33, 21>)layer1_0_rprelu1_temp2)));
          layer1_0_residual2_pipe_14.write(layer1_0_residual2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_0_rprelu2[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_0_rprelu2_pipe_15;
    #pragma HLS stream variable=layer1_0_rprelu2_pipe_15 depth=16384
    hls::stream<ap_fixed<32, 20> > layer1_0_rprelu2_pipe_117;
    #pragma HLS stream variable=layer1_0_rprelu2_pipe_117 depth=16384
    layer1_0_rprelu2_cc3: for (bit32 cc3 = 0; cc3 < 16; ++cc3) {
      layer1_0_rprelu2_ww7: for (bit32 ww7 = 0; ww7 < 32; ++ww7) {
        layer1_0_rprelu2_hh7: for (bit32 hh7 = 0; hh7 < 32; ++hh7) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_0_residual2_temp1;
          layer1_0_residual2_temp1 = layer1_0_residual2_pipe_14.read();
          ap_fixed<32, 20> layer1_0_rprelu2_temp;
          layer1_0_rprelu2_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_0_residual2_temp1) + ((ap_fixed<33, 21>)w_layer1_0_rprelu2_3[cc3])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer1_0_residual2_temp1) + ((ap_fixed<33, 21>)w_layer1_0_rprelu2_3[cc3])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer1_0_rprelu2_5[cc3]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer1_0_residual2_temp1) + ((ap_fixed<33, 21>)w_layer1_0_rprelu2_3[cc3]))))))) + ((ap_fixed<66, 42>)w_layer1_0_rprelu2_4[cc3])));
          layer1_0_rprelu2_pipe_117.write(layer1_0_rprelu2_temp);
          layer1_0_rprelu2_pipe_15.write(layer1_0_rprelu2_temp);
        }
      }
    }
    ap_uint<16> layer1_1_rsign1[1][1][32][32];
    hls::stream<ap_uint<16> > layer1_1_rsign1_pipe_16;
    #pragma HLS stream variable=layer1_1_rsign1_pipe_16 depth=1024
    layer1_1_rsign1_hh8: for (bit32 hh8 = 0; hh8 < 32; ++hh8) {
      layer1_1_rsign1_ww8: for (bit32 ww8 = 0; ww8 < 32; ++ww8) {
      #pragma HLS pipeline
        ap_uint<16> layer1_1_rsign1_pack;
        layer1_1_rsign1_pack = (ap_uint<16>)0;
        loop_i3: for (bit32 i3 = 0; i3 < 16; ++i3) {
          ap_fixed<32, 20> layer1_0_rprelu2_temp1;
          layer1_0_rprelu2_temp1 = layer1_0_rprelu2_pipe_15.read();
          layer1_1_rsign1_pack(i3, i3) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_0_rprelu2_temp1) + ((ap_fixed<33, 21>)w_layer1_1_rsign1[i3])))) ? ((bit32)1) : ((bit32)0));
        }
        ap_uint<16> layer1_1_rsign1_temp;
        layer1_1_rsign1_temp = layer1_1_rsign1_pack;
        layer1_1_rsign1_pipe_16.write(layer1_1_rsign1_temp);
      }
    }
    ap_uint<16> layer1_1_conv1_pad[1][1][34][34];
    hls::stream<ap_uint<16> > layer1_1_conv1_pad_pipe_17;
    #pragma HLS stream variable=layer1_1_conv1_pad_pipe_17 depth=1156
    layer1_1_conv1_pad_hh9: for (bit32 hh9 = 0; hh9 < 34; ++hh9) {
      layer1_1_conv1_pad_ww9: for (bit32 ww9 = 0; ww9 < 34; ++ww9) {
    #pragma HLS pipeline
        ap_uint<16> layer1_1_rsign1_temp1;
        ap_uint<16> layer1_1_conv1_pad_temp;
        layer1_1_conv1_pad_temp = ((ap_uint<16>)(((((1 <= ww9) && (ww9 < 33)) && (1 <= hh9)) && (hh9 < 33)) ? (((ubit32)layer1_1_rsign1_pipe_16.read())) : ((ubit32)0U)));
        layer1_1_conv1_pad_pipe_17.write(layer1_1_conv1_pad_temp);
        layer1_1_conv1_pad[0][0][hh9][ww9] = layer1_1_conv1_pad_temp;
      }
    }
    ap_int<8> layer1_1_conv1[1][16][32][32];
    ap_uint<16> layer1_1_conv1_LB[1][1][3][34];
    ap_uint<16> layer1_1_conv1_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer1_1_conv1_WB complete dim=4
    hls::stream<ap_int<8> > layer1_1_conv1_pipe_18;
    #pragma HLS stream variable=layer1_1_conv1_pipe_18 depth=16384
    layer1_1_conv1_yy_reuse3: for (bit32 yy_reuse3 = 0; yy_reuse3 < 34; ++yy_reuse3) {
      layer1_1_conv1_xx_reuse3: for (bit32 xx_reuse3 = 0; xx_reuse3 < 34; ++xx_reuse3) {
        loop_layer1_1_conv1_pad_1: for (bit32 layer1_1_conv1_pad_1 = 0; layer1_1_conv1_pad_1 < 2; ++layer1_1_conv1_pad_1) {
          layer1_1_conv1_LB[0][0][layer1_1_conv1_pad_1][xx_reuse3] = layer1_1_conv1_LB[0][0][(layer1_1_conv1_pad_1 + 1)][xx_reuse3];
        }
        ap_uint<16> layer1_1_conv1_pad_temp1;
        layer1_1_conv1_pad_temp1 = layer1_1_conv1_pad_pipe_17.read();
        layer1_1_conv1_LB[0][0][2][xx_reuse3] = layer1_1_conv1_pad_temp1;
        if (2 <= yy_reuse3) {
          loop_layer1_1_conv1_LB_1: for (bit32 layer1_1_conv1_LB_1 = 0; layer1_1_conv1_LB_1 < 3; ++layer1_1_conv1_LB_1) {
            loop_layer1_1_conv1_LB_0: for (bit32 layer1_1_conv1_LB_0 = 0; layer1_1_conv1_LB_0 < 2; ++layer1_1_conv1_LB_0) {
              layer1_1_conv1_WB[0][0][layer1_1_conv1_LB_1][layer1_1_conv1_LB_0] = layer1_1_conv1_WB[0][0][layer1_1_conv1_LB_1][(layer1_1_conv1_LB_0 + 1)];
            }
            layer1_1_conv1_WB[0][0][layer1_1_conv1_LB_1][2] = layer1_1_conv1_LB[0][0][layer1_1_conv1_LB_1][xx_reuse3];
          }
            if (2 <= xx_reuse3) {
          layer1_1_conv1_ff3: for (bit32 ff3 = 0; ff3 < 16; ++ff3) {
      #pragma HLS pipeline
              ap_int<8> layer1_1_conv1_sum;
              layer1_1_conv1_sum = (ap_int<8>)0;
              layer1_1_conv1_layer1_1_conv1_ry: for (bit32 layer1_1_conv1_ry = 0; layer1_1_conv1_ry < 3; ++layer1_1_conv1_ry) {
                layer1_1_conv1_layer1_1_conv1_rx: for (bit32 layer1_1_conv1_rx = 0; layer1_1_conv1_rx < 3; ++layer1_1_conv1_rx) {
                  layer1_1_conv1_layer1_1_conv1_rb: for (bit32 layer1_1_conv1_rb = 0; layer1_1_conv1_rb < 16; ++layer1_1_conv1_rb) {
                    layer1_1_conv1_sum = ((ap_int<8>)(((ap_int<18>)(layer1_1_conv1_WB[0][0][layer1_1_conv1_ry][layer1_1_conv1_rx] ^ w_layer1_1_conv1[ff3][0][layer1_1_conv1_ry][layer1_1_conv1_rx])[layer1_1_conv1_rb]) + ((ap_int<18>)layer1_1_conv1_sum)));
                  }
                }
              }
              ap_int<8> layer1_1_conv1_temp;
              layer1_1_conv1_temp = ((ap_int<8>)(144 - ((bit32)(layer1_1_conv1_sum << 1))));
              layer1_1_conv1_pipe_18.write(layer1_1_conv1_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer1_1_bn1[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_1_bn1_pipe_19;
    #pragma HLS stream variable=layer1_1_bn1_pipe_19 depth=16384
    layer1_1_bn1_args03: for (bit32 args03 = 0; args03 < 16; ++args03) {
      layer1_1_bn1_args13: for (bit32 args13 = 0; args13 < 32; ++args13) {
        layer1_1_bn1_args23: for (bit32 args23 = 0; args23 < 32; ++args23) {
        #pragma HLS pipeline
          ap_int<8> layer1_1_conv1_temp1;
          layer1_1_conv1_temp1 = layer1_1_conv1_pipe_18.read();
          ap_fixed<32, 20> layer1_1_bn1_temp;
          layer1_1_bn1_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer1_1_conv1_temp1) - ((ap_fixed<33, 21>)w_layer1_1_bn1_11[args03]))) / sqrt((((float)w_layer1_1_bn1_12[args03]) + 1.000000e-07f))) * ((float)w_layer1_1_bn1_9[args03])) + ((float)w_layer1_1_bn1_10[args03])));
          layer1_1_bn1_pipe_19.write(layer1_1_bn1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_1_residual1[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_1_residual1_pipe_20;
    #pragma HLS stream variable=layer1_1_residual1_pipe_20 depth=16384
    layer1_1_residual1_cc4: for (bit32 cc4 = 0; cc4 < 16; ++cc4) {
      layer1_1_residual1_ww10: for (bit32 ww10 = 0; ww10 < 32; ++ww10) {
        layer1_1_residual1_hh10: for (bit32 hh10 = 0; hh10 < 32; ++hh10) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_1_bn1_temp1;
          layer1_1_bn1_temp1 = layer1_1_bn1_pipe_19.read();
          ap_fixed<32, 20> layer1_1_residual1_temp;
          ap_fixed<32, 20> layer1_0_rprelu2_temp2;
          layer1_0_rprelu2_temp2 = layer1_0_rprelu2_pipe_117.read();
          layer1_1_residual1_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer1_1_bn1_temp1) + ((ap_fixed<33, 21>)layer1_0_rprelu2_temp2)));
          layer1_1_residual1_pipe_20.write(layer1_1_residual1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_1_rprelu1[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_1_rprelu1_pipe_21;
    #pragma HLS stream variable=layer1_1_rprelu1_pipe_21 depth=16384
    hls::stream<ap_fixed<32, 20> > layer1_1_rprelu1_pipe_118;
    #pragma HLS stream variable=layer1_1_rprelu1_pipe_118 depth=16384
    layer1_1_rprelu1_cc5: for (bit32 cc5 = 0; cc5 < 16; ++cc5) {
      layer1_1_rprelu1_ww11: for (bit32 ww11 = 0; ww11 < 32; ++ww11) {
        layer1_1_rprelu1_hh11: for (bit32 hh11 = 0; hh11 < 32; ++hh11) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_1_residual1_temp1;
          layer1_1_residual1_temp1 = layer1_1_residual1_pipe_20.read();
          ap_fixed<32, 20> layer1_1_rprelu1_temp;
          layer1_1_rprelu1_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_1_residual1_temp1) + ((ap_fixed<33, 21>)w_layer1_1_rprelu1_0[cc5])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer1_1_residual1_temp1) + ((ap_fixed<33, 21>)w_layer1_1_rprelu1_0[cc5])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer1_1_rprelu1_2[cc5]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer1_1_residual1_temp1) + ((ap_fixed<33, 21>)w_layer1_1_rprelu1_0[cc5]))))))) + ((ap_fixed<66, 42>)w_layer1_1_rprelu1_1[cc5])));
          layer1_1_rprelu1_pipe_118.write(layer1_1_rprelu1_temp);
          layer1_1_rprelu1_pipe_21.write(layer1_1_rprelu1_temp);
        }
      }
    }
    ap_uint<16> layer1_1_rsign2[1][1][32][32];
    hls::stream<ap_uint<16> > layer1_1_rsign2_pipe_22;
    #pragma HLS stream variable=layer1_1_rsign2_pipe_22 depth=1024
    layer1_1_rsign2_hh12: for (bit32 hh12 = 0; hh12 < 32; ++hh12) {
      layer1_1_rsign2_ww12: for (bit32 ww12 = 0; ww12 < 32; ++ww12) {
      #pragma HLS pipeline
        ap_uint<16> layer1_1_rsign2_pack;
        layer1_1_rsign2_pack = (ap_uint<16>)0;
        loop_i4: for (bit32 i4 = 0; i4 < 16; ++i4) {
          ap_fixed<32, 20> layer1_1_rprelu1_temp1;
          layer1_1_rprelu1_temp1 = layer1_1_rprelu1_pipe_21.read();
          layer1_1_rsign2_pack(i4, i4) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_1_rprelu1_temp1) + ((ap_fixed<33, 21>)w_layer1_1_rsign2[i4])))) ? ((bit32)1) : ((bit32)0));
        }
        ap_uint<16> layer1_1_rsign2_temp;
        layer1_1_rsign2_temp = layer1_1_rsign2_pack;
        layer1_1_rsign2_pipe_22.write(layer1_1_rsign2_temp);
      }
    }
    ap_uint<16> layer1_1_conv2_pad[1][1][34][34];
    hls::stream<ap_uint<16> > layer1_1_conv2_pad_pipe_23;
    #pragma HLS stream variable=layer1_1_conv2_pad_pipe_23 depth=1156
    layer1_1_conv2_pad_hh13: for (bit32 hh13 = 0; hh13 < 34; ++hh13) {
      layer1_1_conv2_pad_ww13: for (bit32 ww13 = 0; ww13 < 34; ++ww13) {
    #pragma HLS pipeline
        ap_uint<16> layer1_1_rsign2_temp1;
        ap_uint<16> layer1_1_conv2_pad_temp;
        layer1_1_conv2_pad_temp = ((ap_uint<16>)(((((1 <= ww13) && (ww13 < 33)) && (1 <= hh13)) && (hh13 < 33)) ? (((ubit32)layer1_1_rsign2_pipe_22.read())) : ((ubit32)0U)));
        layer1_1_conv2_pad_pipe_23.write(layer1_1_conv2_pad_temp);
        layer1_1_conv2_pad[0][0][hh13][ww13] = layer1_1_conv2_pad_temp;
      }
    }
    ap_int<8> layer1_1_conv2[1][16][32][32];
    ap_uint<16> layer1_1_conv2_LB[1][1][3][34];
    ap_uint<16> layer1_1_conv2_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer1_1_conv2_WB complete dim=4
    hls::stream<ap_int<8> > layer1_1_conv2_pipe_24;
    #pragma HLS stream variable=layer1_1_conv2_pipe_24 depth=16384
    layer1_1_conv2_yy_reuse4: for (bit32 yy_reuse4 = 0; yy_reuse4 < 34; ++yy_reuse4) {
      layer1_1_conv2_xx_reuse4: for (bit32 xx_reuse4 = 0; xx_reuse4 < 34; ++xx_reuse4) {
        loop_layer1_1_conv2_pad_1: for (bit32 layer1_1_conv2_pad_1 = 0; layer1_1_conv2_pad_1 < 2; ++layer1_1_conv2_pad_1) {
          layer1_1_conv2_LB[0][0][layer1_1_conv2_pad_1][xx_reuse4] = layer1_1_conv2_LB[0][0][(layer1_1_conv2_pad_1 + 1)][xx_reuse4];
        }
        ap_uint<16> layer1_1_conv2_pad_temp1;
        layer1_1_conv2_pad_temp1 = layer1_1_conv2_pad_pipe_23.read();
        layer1_1_conv2_LB[0][0][2][xx_reuse4] = layer1_1_conv2_pad_temp1;
        if (2 <= yy_reuse4) {
          loop_layer1_1_conv2_LB_1: for (bit32 layer1_1_conv2_LB_1 = 0; layer1_1_conv2_LB_1 < 3; ++layer1_1_conv2_LB_1) {
            loop_layer1_1_conv2_LB_0: for (bit32 layer1_1_conv2_LB_0 = 0; layer1_1_conv2_LB_0 < 2; ++layer1_1_conv2_LB_0) {
              layer1_1_conv2_WB[0][0][layer1_1_conv2_LB_1][layer1_1_conv2_LB_0] = layer1_1_conv2_WB[0][0][layer1_1_conv2_LB_1][(layer1_1_conv2_LB_0 + 1)];
            }
            layer1_1_conv2_WB[0][0][layer1_1_conv2_LB_1][2] = layer1_1_conv2_LB[0][0][layer1_1_conv2_LB_1][xx_reuse4];
          }
            if (2 <= xx_reuse4) {
          layer1_1_conv2_ff4: for (bit32 ff4 = 0; ff4 < 16; ++ff4) {
      #pragma HLS pipeline
              ap_int<8> layer1_1_conv2_sum;
              layer1_1_conv2_sum = (ap_int<8>)0;
              layer1_1_conv2_layer1_1_conv2_ry: for (bit32 layer1_1_conv2_ry = 0; layer1_1_conv2_ry < 3; ++layer1_1_conv2_ry) {
                layer1_1_conv2_layer1_1_conv2_rx: for (bit32 layer1_1_conv2_rx = 0; layer1_1_conv2_rx < 3; ++layer1_1_conv2_rx) {
                  layer1_1_conv2_layer1_1_conv2_rb: for (bit32 layer1_1_conv2_rb = 0; layer1_1_conv2_rb < 16; ++layer1_1_conv2_rb) {
                    layer1_1_conv2_sum = ((ap_int<8>)(((ap_int<18>)(layer1_1_conv2_WB[0][0][layer1_1_conv2_ry][layer1_1_conv2_rx] ^ w_layer1_1_conv2[ff4][0][layer1_1_conv2_ry][layer1_1_conv2_rx])[layer1_1_conv2_rb]) + ((ap_int<18>)layer1_1_conv2_sum)));
                  }
                }
              }
              ap_int<8> layer1_1_conv2_temp;
              layer1_1_conv2_temp = ((ap_int<8>)(144 - ((bit32)(layer1_1_conv2_sum << 1))));
              layer1_1_conv2_pipe_24.write(layer1_1_conv2_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer1_1_bn2[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_1_bn2_pipe_25;
    #pragma HLS stream variable=layer1_1_bn2_pipe_25 depth=16384
    layer1_1_bn2_args04: for (bit32 args04 = 0; args04 < 16; ++args04) {
      layer1_1_bn2_args14: for (bit32 args14 = 0; args14 < 32; ++args14) {
        layer1_1_bn2_args24: for (bit32 args24 = 0; args24 < 32; ++args24) {
        #pragma HLS pipeline
          ap_int<8> layer1_1_conv2_temp1;
          layer1_1_conv2_temp1 = layer1_1_conv2_pipe_24.read();
          ap_fixed<32, 20> layer1_1_bn2_temp;
          layer1_1_bn2_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer1_1_conv2_temp1) - ((ap_fixed<33, 21>)w_layer1_1_bn2_16[args04]))) / sqrt((((float)w_layer1_1_bn2_17[args04]) + 1.000000e-07f))) * ((float)w_layer1_1_bn2_14[args04])) + ((float)w_layer1_1_bn2_15[args04])));
          layer1_1_bn2_pipe_25.write(layer1_1_bn2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_1_residual2[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_1_residual2_pipe_26;
    #pragma HLS stream variable=layer1_1_residual2_pipe_26 depth=16384
    layer1_1_residual2_cc6: for (bit32 cc6 = 0; cc6 < 16; ++cc6) {
      layer1_1_residual2_ww14: for (bit32 ww14 = 0; ww14 < 32; ++ww14) {
        layer1_1_residual2_hh14: for (bit32 hh14 = 0; hh14 < 32; ++hh14) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_1_bn2_temp1;
          layer1_1_bn2_temp1 = layer1_1_bn2_pipe_25.read();
          ap_fixed<32, 20> layer1_1_residual2_temp;
          ap_fixed<32, 20> layer1_1_rprelu1_temp2;
          layer1_1_rprelu1_temp2 = layer1_1_rprelu1_pipe_118.read();
          layer1_1_residual2_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer1_1_bn2_temp1) + ((ap_fixed<33, 21>)layer1_1_rprelu1_temp2)));
          layer1_1_residual2_pipe_26.write(layer1_1_residual2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_1_rprelu2[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_1_rprelu2_pipe_27;
    #pragma HLS stream variable=layer1_1_rprelu2_pipe_27 depth=16384
    hls::stream<ap_fixed<32, 20> > layer1_1_rprelu2_pipe_119;
    #pragma HLS stream variable=layer1_1_rprelu2_pipe_119 depth=16384
    layer1_1_rprelu2_cc7: for (bit32 cc7 = 0; cc7 < 16; ++cc7) {
      layer1_1_rprelu2_ww15: for (bit32 ww15 = 0; ww15 < 32; ++ww15) {
        layer1_1_rprelu2_hh15: for (bit32 hh15 = 0; hh15 < 32; ++hh15) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_1_residual2_temp1;
          layer1_1_residual2_temp1 = layer1_1_residual2_pipe_26.read();
          ap_fixed<32, 20> layer1_1_rprelu2_temp;
          layer1_1_rprelu2_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_1_residual2_temp1) + ((ap_fixed<33, 21>)w_layer1_1_rprelu2_3[cc7])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer1_1_residual2_temp1) + ((ap_fixed<33, 21>)w_layer1_1_rprelu2_3[cc7])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer1_1_rprelu2_5[cc7]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer1_1_residual2_temp1) + ((ap_fixed<33, 21>)w_layer1_1_rprelu2_3[cc7]))))))) + ((ap_fixed<66, 42>)w_layer1_1_rprelu2_4[cc7])));
          layer1_1_rprelu2_pipe_119.write(layer1_1_rprelu2_temp);
          layer1_1_rprelu2_pipe_27.write(layer1_1_rprelu2_temp);
        }
      }
    }
    ap_uint<16> layer1_2_rsign1[1][1][32][32];
    hls::stream<ap_uint<16> > layer1_2_rsign1_pipe_28;
    #pragma HLS stream variable=layer1_2_rsign1_pipe_28 depth=1024
    layer1_2_rsign1_hh16: for (bit32 hh16 = 0; hh16 < 32; ++hh16) {
      layer1_2_rsign1_ww16: for (bit32 ww16 = 0; ww16 < 32; ++ww16) {
      #pragma HLS pipeline
        ap_uint<16> layer1_2_rsign1_pack;
        layer1_2_rsign1_pack = (ap_uint<16>)0;
        loop_i5: for (bit32 i5 = 0; i5 < 16; ++i5) {
          ap_fixed<32, 20> layer1_1_rprelu2_temp1;
          layer1_1_rprelu2_temp1 = layer1_1_rprelu2_pipe_27.read();
          layer1_2_rsign1_pack(i5, i5) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_1_rprelu2_temp1) + ((ap_fixed<33, 21>)w_layer1_2_rsign1[i5])))) ? ((bit32)1) : ((bit32)0));
        }
        ap_uint<16> layer1_2_rsign1_temp;
        layer1_2_rsign1_temp = layer1_2_rsign1_pack;
        layer1_2_rsign1_pipe_28.write(layer1_2_rsign1_temp);
      }
    }
    ap_uint<16> layer1_2_conv1_pad[1][1][34][34];
    hls::stream<ap_uint<16> > layer1_2_conv1_pad_pipe_29;
    #pragma HLS stream variable=layer1_2_conv1_pad_pipe_29 depth=1156
    layer1_2_conv1_pad_hh17: for (bit32 hh17 = 0; hh17 < 34; ++hh17) {
      layer1_2_conv1_pad_ww17: for (bit32 ww17 = 0; ww17 < 34; ++ww17) {
    #pragma HLS pipeline
        ap_uint<16> layer1_2_rsign1_temp1;
        ap_uint<16> layer1_2_conv1_pad_temp;
        layer1_2_conv1_pad_temp = ((ap_uint<16>)(((((1 <= ww17) && (ww17 < 33)) && (1 <= hh17)) && (hh17 < 33)) ? (((ubit32)layer1_2_rsign1_pipe_28.read())) : ((ubit32)0U)));
        layer1_2_conv1_pad_pipe_29.write(layer1_2_conv1_pad_temp);
        layer1_2_conv1_pad[0][0][hh17][ww17] = layer1_2_conv1_pad_temp;
      }
    }
    ap_int<8> layer1_2_conv1[1][16][32][32];
    ap_uint<16> layer1_2_conv1_LB[1][1][3][34];
    ap_uint<16> layer1_2_conv1_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer1_2_conv1_WB complete dim=4
    hls::stream<ap_int<8> > layer1_2_conv1_pipe_30;
    #pragma HLS stream variable=layer1_2_conv1_pipe_30 depth=16384
    layer1_2_conv1_yy_reuse5: for (bit32 yy_reuse5 = 0; yy_reuse5 < 34; ++yy_reuse5) {
      layer1_2_conv1_xx_reuse5: for (bit32 xx_reuse5 = 0; xx_reuse5 < 34; ++xx_reuse5) {
        loop_layer1_2_conv1_pad_1: for (bit32 layer1_2_conv1_pad_1 = 0; layer1_2_conv1_pad_1 < 2; ++layer1_2_conv1_pad_1) {
          layer1_2_conv1_LB[0][0][layer1_2_conv1_pad_1][xx_reuse5] = layer1_2_conv1_LB[0][0][(layer1_2_conv1_pad_1 + 1)][xx_reuse5];
        }
        ap_uint<16> layer1_2_conv1_pad_temp1;
        layer1_2_conv1_pad_temp1 = layer1_2_conv1_pad_pipe_29.read();
        layer1_2_conv1_LB[0][0][2][xx_reuse5] = layer1_2_conv1_pad_temp1;
        if (2 <= yy_reuse5) {
          loop_layer1_2_conv1_LB_1: for (bit32 layer1_2_conv1_LB_1 = 0; layer1_2_conv1_LB_1 < 3; ++layer1_2_conv1_LB_1) {
            loop_layer1_2_conv1_LB_0: for (bit32 layer1_2_conv1_LB_0 = 0; layer1_2_conv1_LB_0 < 2; ++layer1_2_conv1_LB_0) {
              layer1_2_conv1_WB[0][0][layer1_2_conv1_LB_1][layer1_2_conv1_LB_0] = layer1_2_conv1_WB[0][0][layer1_2_conv1_LB_1][(layer1_2_conv1_LB_0 + 1)];
            }
            layer1_2_conv1_WB[0][0][layer1_2_conv1_LB_1][2] = layer1_2_conv1_LB[0][0][layer1_2_conv1_LB_1][xx_reuse5];
          }
            if (2 <= xx_reuse5) {
          layer1_2_conv1_ff5: for (bit32 ff5 = 0; ff5 < 16; ++ff5) {
      #pragma HLS pipeline
              ap_int<8> layer1_2_conv1_sum;
              layer1_2_conv1_sum = (ap_int<8>)0;
              layer1_2_conv1_layer1_2_conv1_ry: for (bit32 layer1_2_conv1_ry = 0; layer1_2_conv1_ry < 3; ++layer1_2_conv1_ry) {
                layer1_2_conv1_layer1_2_conv1_rx: for (bit32 layer1_2_conv1_rx = 0; layer1_2_conv1_rx < 3; ++layer1_2_conv1_rx) {
                  layer1_2_conv1_layer1_2_conv1_rb: for (bit32 layer1_2_conv1_rb = 0; layer1_2_conv1_rb < 16; ++layer1_2_conv1_rb) {
                    layer1_2_conv1_sum = ((ap_int<8>)(((ap_int<18>)(layer1_2_conv1_WB[0][0][layer1_2_conv1_ry][layer1_2_conv1_rx] ^ w_layer1_2_conv1[ff5][0][layer1_2_conv1_ry][layer1_2_conv1_rx])[layer1_2_conv1_rb]) + ((ap_int<18>)layer1_2_conv1_sum)));
                  }
                }
              }
              ap_int<8> layer1_2_conv1_temp;
              layer1_2_conv1_temp = ((ap_int<8>)(144 - ((bit32)(layer1_2_conv1_sum << 1))));
              layer1_2_conv1_pipe_30.write(layer1_2_conv1_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer1_2_bn1[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_2_bn1_pipe_31;
    #pragma HLS stream variable=layer1_2_bn1_pipe_31 depth=16384
    layer1_2_bn1_args05: for (bit32 args05 = 0; args05 < 16; ++args05) {
      layer1_2_bn1_args15: for (bit32 args15 = 0; args15 < 32; ++args15) {
        layer1_2_bn1_args25: for (bit32 args25 = 0; args25 < 32; ++args25) {
        #pragma HLS pipeline
          ap_int<8> layer1_2_conv1_temp1;
          layer1_2_conv1_temp1 = layer1_2_conv1_pipe_30.read();
          ap_fixed<32, 20> layer1_2_bn1_temp;
          layer1_2_bn1_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer1_2_conv1_temp1) - ((ap_fixed<33, 21>)w_layer1_2_bn1_11[args05]))) / sqrt((((float)w_layer1_2_bn1_12[args05]) + 1.000000e-07f))) * ((float)w_layer1_2_bn1_9[args05])) + ((float)w_layer1_2_bn1_10[args05])));
          layer1_2_bn1_pipe_31.write(layer1_2_bn1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_2_residual1[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_2_residual1_pipe_32;
    #pragma HLS stream variable=layer1_2_residual1_pipe_32 depth=16384
    layer1_2_residual1_cc8: for (bit32 cc8 = 0; cc8 < 16; ++cc8) {
      layer1_2_residual1_ww18: for (bit32 ww18 = 0; ww18 < 32; ++ww18) {
        layer1_2_residual1_hh18: for (bit32 hh18 = 0; hh18 < 32; ++hh18) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_2_bn1_temp1;
          layer1_2_bn1_temp1 = layer1_2_bn1_pipe_31.read();
          ap_fixed<32, 20> layer1_2_residual1_temp;
          ap_fixed<32, 20> layer1_1_rprelu2_temp2;
          layer1_1_rprelu2_temp2 = layer1_1_rprelu2_pipe_119.read();
          layer1_2_residual1_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer1_2_bn1_temp1) + ((ap_fixed<33, 21>)layer1_1_rprelu2_temp2)));
          layer1_2_residual1_pipe_32.write(layer1_2_residual1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_2_rprelu1[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_2_rprelu1_pipe_33;
    #pragma HLS stream variable=layer1_2_rprelu1_pipe_33 depth=16384
    hls::stream<ap_fixed<32, 20> > layer1_2_rprelu1_pipe_120;
    #pragma HLS stream variable=layer1_2_rprelu1_pipe_120 depth=16384
    layer1_2_rprelu1_cc9: for (bit32 cc9 = 0; cc9 < 16; ++cc9) {
      layer1_2_rprelu1_ww19: for (bit32 ww19 = 0; ww19 < 32; ++ww19) {
        layer1_2_rprelu1_hh19: for (bit32 hh19 = 0; hh19 < 32; ++hh19) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_2_residual1_temp1;
          layer1_2_residual1_temp1 = layer1_2_residual1_pipe_32.read();
          ap_fixed<32, 20> layer1_2_rprelu1_temp;
          layer1_2_rprelu1_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_2_residual1_temp1) + ((ap_fixed<33, 21>)w_layer1_2_rprelu1_0[cc9])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer1_2_residual1_temp1) + ((ap_fixed<33, 21>)w_layer1_2_rprelu1_0[cc9])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer1_2_rprelu1_2[cc9]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer1_2_residual1_temp1) + ((ap_fixed<33, 21>)w_layer1_2_rprelu1_0[cc9]))))))) + ((ap_fixed<66, 42>)w_layer1_2_rprelu1_1[cc9])));
          layer1_2_rprelu1_pipe_120.write(layer1_2_rprelu1_temp);
          layer1_2_rprelu1_pipe_33.write(layer1_2_rprelu1_temp);
        }
      }
    }
    ap_uint<16> layer1_2_rsign2[1][1][32][32];
    hls::stream<ap_uint<16> > layer1_2_rsign2_pipe_34;
    #pragma HLS stream variable=layer1_2_rsign2_pipe_34 depth=1024
    layer1_2_rsign2_hh20: for (bit32 hh20 = 0; hh20 < 32; ++hh20) {
      layer1_2_rsign2_ww20: for (bit32 ww20 = 0; ww20 < 32; ++ww20) {
      #pragma HLS pipeline
        ap_uint<16> layer1_2_rsign2_pack;
        layer1_2_rsign2_pack = (ap_uint<16>)0;
        loop_i6: for (bit32 i6 = 0; i6 < 16; ++i6) {
          ap_fixed<32, 20> layer1_2_rprelu1_temp1;
          layer1_2_rprelu1_temp1 = layer1_2_rprelu1_pipe_33.read();
          layer1_2_rsign2_pack(i6, i6) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_2_rprelu1_temp1) + ((ap_fixed<33, 21>)w_layer1_2_rsign2[i6])))) ? ((bit32)1) : ((bit32)0));
        }
        ap_uint<16> layer1_2_rsign2_temp;
        layer1_2_rsign2_temp = layer1_2_rsign2_pack;
        layer1_2_rsign2_pipe_34.write(layer1_2_rsign2_temp);
      }
    }
    ap_uint<16> layer1_2_conv2_pad[1][1][34][34];
    hls::stream<ap_uint<16> > layer1_2_conv2_pad_pipe_35;
    #pragma HLS stream variable=layer1_2_conv2_pad_pipe_35 depth=1156
    layer1_2_conv2_pad_hh21: for (bit32 hh21 = 0; hh21 < 34; ++hh21) {
      layer1_2_conv2_pad_ww21: for (bit32 ww21 = 0; ww21 < 34; ++ww21) {
    #pragma HLS pipeline
        ap_uint<16> layer1_2_rsign2_temp1;
        ap_uint<16> layer1_2_conv2_pad_temp;
        layer1_2_conv2_pad_temp = ((ap_uint<16>)(((((1 <= ww21) && (ww21 < 33)) && (1 <= hh21)) && (hh21 < 33)) ? (((ubit32)layer1_2_rsign2_pipe_34.read())) : ((ubit32)0U)));
        layer1_2_conv2_pad_pipe_35.write(layer1_2_conv2_pad_temp);
        layer1_2_conv2_pad[0][0][hh21][ww21] = layer1_2_conv2_pad_temp;
      }
    }
    ap_int<8> layer1_2_conv2[1][16][32][32];
    ap_uint<16> layer1_2_conv2_LB[1][1][3][34];
    ap_uint<16> layer1_2_conv2_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer1_2_conv2_WB complete dim=4
    hls::stream<ap_int<8> > layer1_2_conv2_pipe_36;
    #pragma HLS stream variable=layer1_2_conv2_pipe_36 depth=16384
    layer1_2_conv2_yy_reuse6: for (bit32 yy_reuse6 = 0; yy_reuse6 < 34; ++yy_reuse6) {
      layer1_2_conv2_xx_reuse6: for (bit32 xx_reuse6 = 0; xx_reuse6 < 34; ++xx_reuse6) {
        loop_layer1_2_conv2_pad_1: for (bit32 layer1_2_conv2_pad_1 = 0; layer1_2_conv2_pad_1 < 2; ++layer1_2_conv2_pad_1) {
          layer1_2_conv2_LB[0][0][layer1_2_conv2_pad_1][xx_reuse6] = layer1_2_conv2_LB[0][0][(layer1_2_conv2_pad_1 + 1)][xx_reuse6];
        }
        ap_uint<16> layer1_2_conv2_pad_temp1;
        layer1_2_conv2_pad_temp1 = layer1_2_conv2_pad_pipe_35.read();
        layer1_2_conv2_LB[0][0][2][xx_reuse6] = layer1_2_conv2_pad_temp1;
        if (2 <= yy_reuse6) {
          loop_layer1_2_conv2_LB_1: for (bit32 layer1_2_conv2_LB_1 = 0; layer1_2_conv2_LB_1 < 3; ++layer1_2_conv2_LB_1) {
            loop_layer1_2_conv2_LB_0: for (bit32 layer1_2_conv2_LB_0 = 0; layer1_2_conv2_LB_0 < 2; ++layer1_2_conv2_LB_0) {
              layer1_2_conv2_WB[0][0][layer1_2_conv2_LB_1][layer1_2_conv2_LB_0] = layer1_2_conv2_WB[0][0][layer1_2_conv2_LB_1][(layer1_2_conv2_LB_0 + 1)];
            }
            layer1_2_conv2_WB[0][0][layer1_2_conv2_LB_1][2] = layer1_2_conv2_LB[0][0][layer1_2_conv2_LB_1][xx_reuse6];
          }
            if (2 <= xx_reuse6) {
          layer1_2_conv2_ff6: for (bit32 ff6 = 0; ff6 < 16; ++ff6) {
      #pragma HLS pipeline
              ap_int<8> layer1_2_conv2_sum;
              layer1_2_conv2_sum = (ap_int<8>)0;
              layer1_2_conv2_layer1_2_conv2_ry: for (bit32 layer1_2_conv2_ry = 0; layer1_2_conv2_ry < 3; ++layer1_2_conv2_ry) {
                layer1_2_conv2_layer1_2_conv2_rx: for (bit32 layer1_2_conv2_rx = 0; layer1_2_conv2_rx < 3; ++layer1_2_conv2_rx) {
                  layer1_2_conv2_layer1_2_conv2_rb: for (bit32 layer1_2_conv2_rb = 0; layer1_2_conv2_rb < 16; ++layer1_2_conv2_rb) {
                    layer1_2_conv2_sum = ((ap_int<8>)(((ap_int<18>)(layer1_2_conv2_WB[0][0][layer1_2_conv2_ry][layer1_2_conv2_rx] ^ w_layer1_2_conv2[ff6][0][layer1_2_conv2_ry][layer1_2_conv2_rx])[layer1_2_conv2_rb]) + ((ap_int<18>)layer1_2_conv2_sum)));
                  }
                }
              }
              ap_int<8> layer1_2_conv2_temp;
              layer1_2_conv2_temp = ((ap_int<8>)(144 - ((bit32)(layer1_2_conv2_sum << 1))));
              layer1_2_conv2_pipe_36.write(layer1_2_conv2_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer1_2_bn2[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_2_bn2_pipe_37;
    #pragma HLS stream variable=layer1_2_bn2_pipe_37 depth=16384
    layer1_2_bn2_args06: for (bit32 args06 = 0; args06 < 16; ++args06) {
      layer1_2_bn2_args16: for (bit32 args16 = 0; args16 < 32; ++args16) {
        layer1_2_bn2_args26: for (bit32 args26 = 0; args26 < 32; ++args26) {
        #pragma HLS pipeline
          ap_int<8> layer1_2_conv2_temp1;
          layer1_2_conv2_temp1 = layer1_2_conv2_pipe_36.read();
          ap_fixed<32, 20> layer1_2_bn2_temp;
          layer1_2_bn2_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer1_2_conv2_temp1) - ((ap_fixed<33, 21>)w_layer1_2_bn2_16[args06]))) / sqrt((((float)w_layer1_2_bn2_17[args06]) + 1.000000e-07f))) * ((float)w_layer1_2_bn2_14[args06])) + ((float)w_layer1_2_bn2_15[args06])));
          layer1_2_bn2_pipe_37.write(layer1_2_bn2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_2_residual2[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_2_residual2_pipe_38;
    #pragma HLS stream variable=layer1_2_residual2_pipe_38 depth=16384
    layer1_2_residual2_cc10: for (bit32 cc10 = 0; cc10 < 16; ++cc10) {
      layer1_2_residual2_ww22: for (bit32 ww22 = 0; ww22 < 32; ++ww22) {
        layer1_2_residual2_hh22: for (bit32 hh22 = 0; hh22 < 32; ++hh22) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_2_bn2_temp1;
          layer1_2_bn2_temp1 = layer1_2_bn2_pipe_37.read();
          ap_fixed<32, 20> layer1_2_residual2_temp;
          ap_fixed<32, 20> layer1_2_rprelu1_temp2;
          layer1_2_rprelu1_temp2 = layer1_2_rprelu1_pipe_120.read();
          layer1_2_residual2_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer1_2_bn2_temp1) + ((ap_fixed<33, 21>)layer1_2_rprelu1_temp2)));
          layer1_2_residual2_pipe_38.write(layer1_2_residual2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer1_2_rprelu2[1][16][32][32];
    hls::stream<ap_fixed<32, 20> > layer1_2_rprelu2_pipe_39;
    #pragma HLS stream variable=layer1_2_rprelu2_pipe_39 depth=16384
    hls::stream<ap_fixed<32, 20> > layer1_2_rprelu2_pipe_121;
    #pragma HLS stream variable=layer1_2_rprelu2_pipe_121 depth=16384
    layer1_2_rprelu2_cc11: for (bit32 cc11 = 0; cc11 < 16; ++cc11) {
      layer1_2_rprelu2_ww23: for (bit32 ww23 = 0; ww23 < 32; ++ww23) {
        layer1_2_rprelu2_hh23: for (bit32 hh23 = 0; hh23 < 32; ++hh23) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer1_2_residual2_temp1;
          layer1_2_residual2_temp1 = layer1_2_residual2_pipe_38.read();
          ap_fixed<32, 20> layer1_2_rprelu2_temp;
          layer1_2_rprelu2_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_2_residual2_temp1) + ((ap_fixed<33, 21>)w_layer1_2_rprelu2_3[cc11])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer1_2_residual2_temp1) + ((ap_fixed<33, 21>)w_layer1_2_rprelu2_3[cc11])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer1_2_rprelu2_5[cc11]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer1_2_residual2_temp1) + ((ap_fixed<33, 21>)w_layer1_2_rprelu2_3[cc11]))))))) + ((ap_fixed<66, 42>)w_layer1_2_rprelu2_4[cc11])));
          layer1_2_rprelu2_pipe_121.write(layer1_2_rprelu2_temp);
          layer1_2_rprelu2_pipe_39.write(layer1_2_rprelu2_temp);
        }
      }
    }
    ap_uint<16> layer2_0_rsign1[1][1][32][32];
    hls::stream<ap_uint<16> > layer2_0_rsign1_pipe_40;
    #pragma HLS stream variable=layer2_0_rsign1_pipe_40 depth=1024
    layer2_0_rsign1_hh24: for (bit32 hh24 = 0; hh24 < 32; ++hh24) {
      layer2_0_rsign1_ww24: for (bit32 ww24 = 0; ww24 < 32; ++ww24) {
      #pragma HLS pipeline
        ap_uint<16> layer2_0_rsign1_pack;
        layer2_0_rsign1_pack = (ap_uint<16>)0;
        loop_i7: for (bit32 i7 = 0; i7 < 16; ++i7) {
          ap_fixed<32, 20> layer1_2_rprelu2_temp1;
          layer1_2_rprelu2_temp1 = layer1_2_rprelu2_pipe_39.read();
          layer2_0_rsign1_pack(i7, i7) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer1_2_rprelu2_temp1) + ((ap_fixed<33, 21>)w_layer2_0_rsign1[i7])))) ? ((bit32)1) : ((bit32)0));
        }
        ap_uint<16> layer2_0_rsign1_temp;
        layer2_0_rsign1_temp = layer2_0_rsign1_pack;
        layer2_0_rsign1_pipe_40.write(layer2_0_rsign1_temp);
      }
    }
    ap_uint<16> layer2_0_conv1_pad[1][1][34][34];
    hls::stream<ap_uint<16> > layer2_0_conv1_pad_pipe_41;
    #pragma HLS stream variable=layer2_0_conv1_pad_pipe_41 depth=1156
    layer2_0_conv1_pad_hh: for (bit32 hh = 0; hh < 34; ++hh) {
      layer2_0_conv1_pad_ww: for (bit32 ww = 0; ww < 34; ++ww) {
    #pragma HLS pipeline
        layer2_0_conv1_pad_pipe_41.write(((ap_uint<16>)(((((1 <= ww) && (ww < 33)) && (1 <= hh)) && (hh < 33)) ? (((ubit32)layer2_0_rsign1_pipe_40.read())) : ((ubit32)0U))));
      }
    }
    ap_int<8> layer2_0_conv1[1][32][16][16];
    ap_uint<16> layer2_0_conv1_LB[1][1][3][18];
    ap_uint<16> layer2_0_conv1_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer2_0_conv1_WB complete dim=4
    hls::stream<ap_int<8> > layer2_0_conv1_pipe_42;
    #pragma HLS stream variable=layer2_0_conv1_pipe_42 depth=8192
      layer2_0_conv1_yy_reuse: for (bit32 yy_reuse = 0; yy_reuse < 34; ++yy_reuse) {
        layer2_0_conv1_xx_reuse: for (bit32 xx_reuse = 0; xx_reuse < 34; ++xx_reuse) {
          loop_layer2_0_conv1_pad_1: for (bit32 layer2_0_conv1_pad_1 = 0; layer2_0_conv1_pad_1 < 2; ++layer2_0_conv1_pad_1) {
            layer2_0_conv1_LB[0][0][layer2_0_conv1_pad_1][xx_reuse] = layer2_0_conv1_LB[0][0][(layer2_0_conv1_pad_1 + 1)][xx_reuse];
          }
          layer2_0_conv1_LB[0][0][2][xx_reuse] = layer2_0_conv1_pad_pipe_41.read();
          if (2 <= yy_reuse && ((yy_reuse - 2) % 2 == 0)) { // not so correct
            loop_layer2_0_conv1_LB_1: for (bit32 layer2_0_conv1_LB_1 = 0; layer2_0_conv1_LB_1 < 3; ++layer2_0_conv1_LB_1) {
              loop_layer2_0_conv1_LB_0: for (bit32 layer2_0_conv1_LB_0 = 0; layer2_0_conv1_LB_0 < 2; ++layer2_0_conv1_LB_0) {
                layer2_0_conv1_WB[0][0][layer2_0_conv1_LB_1][layer2_0_conv1_LB_0] = layer2_0_conv1_WB[0][0][layer2_0_conv1_LB_1][(layer2_0_conv1_LB_0 + 1)];
              }
              layer2_0_conv1_WB[0][0][layer2_0_conv1_LB_1][2] = layer2_0_conv1_LB[0][0][layer2_0_conv1_LB_1][xx_reuse];
            }
            if (2 <= xx_reuse && ((xx_reuse - 2) % 2 == 0)) {
              ap_uint<16> layer2_0_conv1_sum;
              layer2_0_conv1_sum = (ap_uint<16>)0;
    layer2_0_conv1_ff: for (bit32 ff = 0; ff < 32; ++ff) {
      #pragma HLS pipeline
              layer2_0_conv1_layer2_0_conv1_ry: for (bit32 layer2_0_conv1_ry = 0; layer2_0_conv1_ry < 3; ++layer2_0_conv1_ry) {
                layer2_0_conv1_layer2_0_conv1_rx: for (bit32 layer2_0_conv1_rx = 0; layer2_0_conv1_rx < 3; ++layer2_0_conv1_rx) {
                  layer2_0_conv1_layer2_0_conv1_rb: for (bit32 layer2_0_conv1_rb = 0; layer2_0_conv1_rb < 16; ++layer2_0_conv1_rb) {
                    layer2_0_conv1_sum = ((ap_uint<16>)(((ap_uint<17>)(layer2_0_conv1_WB[0][0][layer2_0_conv1_ry][layer2_0_conv1_rx] ^ w_layer2_0_conv1[ff][0][layer2_0_conv1_ry][layer2_0_conv1_rx])[layer2_0_conv1_rb]) + ((ap_uint<17>)layer2_0_conv1_sum)));
                  }
                }
              }
              layer2_0_conv1_pipe_42.write((ap_uint<16>)(144U - ((ubit32)(layer2_0_conv1_sum << 1))));
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer2_0_bn1[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_0_bn1_pipe_43;
    #pragma HLS stream variable=layer2_0_bn1_pipe_43 depth=8192
    layer2_0_bn1_args07: for (bit32 args07 = 0; args07 < 32; ++args07) {
      layer2_0_bn1_args17: for (bit32 args17 = 0; args17 < 16; ++args17) {
        layer2_0_bn1_args27: for (bit32 args27 = 0; args27 < 16; ++args27) {
        #pragma HLS pipeline
          ap_int<8> layer2_0_conv1_temp1;
          layer2_0_conv1_temp1 = layer2_0_conv1_pipe_42.read();
          ap_fixed<32, 20> layer2_0_bn1_temp;
          layer2_0_bn1_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer2_0_conv1_temp1) - ((ap_fixed<33, 21>)w_layer2_0_bn1_11[args07]))) / sqrt((((float)w_layer2_0_bn1_12[args07]) + 1.000000e-07f))) * ((float)w_layer2_0_bn1_9[args07])) + ((float)w_layer2_0_bn1_10[args07])));
          layer2_0_bn1_pipe_43.write(layer2_0_bn1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_0_avgpool_res[1][16][16][16];
    ap_fixed<32, 20> layer2_0_avgpool_LB[2][32];
    bit32 layer2_0_avgpool;
    hls::stream<ap_fixed<32, 20> > layer2_0_avgpool_res_pipe_122;
    #pragma HLS stream variable=layer2_0_avgpool_res_pipe_122 depth=4096
    layer2_0_avgpool_cc12: for (bit32 cc12 = 0; cc12 < 16; ++cc12) {
      layer2_0_avgpool_hh26: for (bit32 hh26 = 0; hh26 < 16; ++hh26) {
      #pragma HLS pipeline
        loop_layer2_0_avgpool_LB_i: for (bit32 layer2_0_avgpool_LB_i = 0; layer2_0_avgpool_LB_i < 2; ++layer2_0_avgpool_LB_i) {
          loop_layer2_0_avgpool_LB_j: for (bit32 layer2_0_avgpool_LB_j = 0; layer2_0_avgpool_LB_j < 32; ++layer2_0_avgpool_LB_j) {
            ap_fixed<32, 20> layer1_2_rprelu2_temp2;
            layer1_2_rprelu2_temp2 = layer1_2_rprelu2_pipe_121.read();
            layer2_0_avgpool_LB[layer2_0_avgpool_LB_i][layer2_0_avgpool_LB_j] = layer1_2_rprelu2_temp2;
          }
        }
        loop_layer2_0_avgpool_ww: for (bit32 layer2_0_avgpool_ww = 0; layer2_0_avgpool_ww < 16; ++layer2_0_avgpool_ww) {
          ap_fixed<32, 20> layer2_0_avgpool_val;
          layer2_0_avgpool_val = ((ap_fixed<32, 20>)0);
          loop_layer2_0_avgpool_ry: for (bit32 layer2_0_avgpool_ry = 0; layer2_0_avgpool_ry < 2; ++layer2_0_avgpool_ry) {
            loop_layer2_0_avgpool_rx: for (bit32 layer2_0_avgpool_rx = 0; layer2_0_avgpool_rx < 2; ++layer2_0_avgpool_rx) {
              layer2_0_avgpool_val = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer2_0_avgpool_val) + ((ap_fixed<33, 21>)layer2_0_avgpool_LB[layer2_0_avgpool_ry][((layer2_0_avgpool_ww * 2) + layer2_0_avgpool_rx)])));
            }
          }
          ap_fixed<32, 20> layer2_0_avgpool_res_temp;
          layer2_0_avgpool_res_temp = ((ap_fixed<32, 20>)(((ap_fixed<64, 20>)layer2_0_avgpool_val) / (ap_fixed<64, 20>)4));
          layer2_0_avgpool_res_pipe_122.write(layer2_0_avgpool_res_temp);
          layer2_0_avgpool_res[0][cc12][hh26][layer2_0_avgpool_ww] = layer2_0_avgpool_res_temp;
        }
      }
    }
    ap_fixed<32, 20> layer2_0_concat[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_0_concat_pipe_123;
    #pragma HLS stream variable=layer2_0_concat_pipe_123 depth=8192
    layer2_0_concat_cc13: for (bit32 cc13 = 0; cc13 < 16; ++cc13) { // need to split 2
      layer2_0_concat_ww26: for (bit32 ww26 = 0; ww26 < 16; ++ww26) {
      #pragma HLS pipeline
        layer2_0_concat_hh27: for (bit32 hh27 = 0; hh27 < 16; ++hh27) {
          ap_fixed<32, 20> layer2_0_avgpool_res_temp1;
          layer2_0_avgpool_res_temp1 = layer2_0_avgpool_res_pipe_122.read();
          ap_fixed<32, 20> layer2_0_concat_temp;
          layer2_0_concat_temp = layer2_0_avgpool_res_temp1;
          layer2_0_concat_pipe_123.write(layer2_0_concat_temp);
          layer2_0_concat_pipe_123.write(layer2_0_concat_temp); // send the same data two times, but actually this is incorrect (need to do memory layout transformation)
        }
      }
    }
    ap_fixed<32, 20> layer2_0_residual1[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_0_residual1_pipe_44;
    #pragma HLS stream variable=layer2_0_residual1_pipe_44 depth=8192
    layer2_0_residual1_cc14: for (bit32 cc14 = 0; cc14 < 32; ++cc14) {
      layer2_0_residual1_ww27: for (bit32 ww27 = 0; ww27 < 16; ++ww27) {
        layer2_0_residual1_hh28: for (bit32 hh28 = 0; hh28 < 16; ++hh28) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_0_bn1_temp1;
          layer2_0_bn1_temp1 = layer2_0_bn1_pipe_43.read();
          ap_fixed<32, 20> layer2_0_residual1_temp;
          ap_fixed<32, 20> layer2_0_concat_temp1;
          layer2_0_concat_temp1 = layer2_0_concat_pipe_123.read();
          layer2_0_residual1_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer2_0_bn1_temp1) + ((ap_fixed<33, 21>)layer2_0_concat_temp1)));
          layer2_0_residual1_pipe_44.write(layer2_0_residual1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_0_rprelu1[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_0_rprelu1_pipe_45;
    #pragma HLS stream variable=layer2_0_rprelu1_pipe_45 depth=8192
    hls::stream<ap_fixed<32, 20> > layer2_0_rprelu1_pipe_124;
    #pragma HLS stream variable=layer2_0_rprelu1_pipe_124 depth=8192
    layer2_0_rprelu1_cc15: for (bit32 cc15 = 0; cc15 < 32; ++cc15) {
      layer2_0_rprelu1_ww28: for (bit32 ww28 = 0; ww28 < 16; ++ww28) {
        layer2_0_rprelu1_hh29: for (bit32 hh29 = 0; hh29 < 16; ++hh29) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_0_residual1_temp1;
          layer2_0_residual1_temp1 = layer2_0_residual1_pipe_44.read();
          ap_fixed<32, 20> layer2_0_rprelu1_temp;
          layer2_0_rprelu1_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_0_residual1_temp1) + ((ap_fixed<33, 21>)w_layer2_0_rprelu1_0[cc15])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer2_0_residual1_temp1) + ((ap_fixed<33, 21>)w_layer2_0_rprelu1_0[cc15])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer2_0_rprelu1_2[cc15]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer2_0_residual1_temp1) + ((ap_fixed<33, 21>)w_layer2_0_rprelu1_0[cc15]))))))) + ((ap_fixed<66, 42>)w_layer2_0_rprelu1_1[cc15])));
          layer2_0_rprelu1_pipe_124.write(layer2_0_rprelu1_temp);
          layer2_0_rprelu1_pipe_45.write(layer2_0_rprelu1_temp);
        }
      }
    }
    ubit32 layer2_0_rsign2[1][1][16][16];
    hls::stream<ubit32 > layer2_0_rsign2_pipe_46;
    #pragma HLS stream variable=layer2_0_rsign2_pipe_46 depth=256
    layer2_0_rsign2_hh30: for (bit32 hh30 = 0; hh30 < 16; ++hh30) {
      layer2_0_rsign2_ww29: for (bit32 ww29 = 0; ww29 < 16; ++ww29) {
      #pragma HLS pipeline
        ubit32 layer2_0_rsign2_pack;
        layer2_0_rsign2_pack = 0U;
        loop_i8: for (bit32 i8 = 0; i8 < 32; ++i8) {
          ap_fixed<32, 20> layer2_0_rprelu1_temp1;
          layer2_0_rprelu1_temp1 = layer2_0_rprelu1_pipe_45.read();
          layer2_0_rsign2_pack(i8, i8) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_0_rprelu1_temp1) + ((ap_fixed<33, 21>)w_layer2_0_rsign2[i8])))) ? ((bit32)1) : ((bit32)0));
        }
        ubit32 layer2_0_rsign2_temp;
        layer2_0_rsign2_temp = layer2_0_rsign2_pack;
        layer2_0_rsign2_pipe_46.write(layer2_0_rsign2_temp);
      }
    }
    ubit32 layer2_0_conv2_pad[1][1][18][18];
    hls::stream<ubit32 > layer2_0_conv2_pad_pipe_47;
    #pragma HLS stream variable=layer2_0_conv2_pad_pipe_47 depth=324
    layer2_0_conv2_pad_hh31: for (bit32 hh31 = 0; hh31 < 18; ++hh31) {
      layer2_0_conv2_pad_ww30: for (bit32 ww30 = 0; ww30 < 18; ++ww30) {
    #pragma HLS pipeline
        ubit32 layer2_0_rsign2_temp1;
        ubit32 layer2_0_conv2_pad_temp;
        if (((((1 <= ww30) && (ww30 < 17)) && (1 <= hh31)) && (hh31 < 17))) { 
          layer2_0_conv2_pad_temp = layer2_0_rsign2_pipe_46.read();
        } else { 
          layer2_0_conv2_pad_temp = 0U;
        }
        layer2_0_conv2_pad_pipe_47.write(layer2_0_conv2_pad_temp);
      }
    }
    ap_int<8> layer2_0_conv2[1][32][16][16];
    ubit32 layer2_0_conv2_LB[1][1][3][18];
    ubit32 layer2_0_conv2_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer2_0_conv2_WB complete dim=4
    hls::stream<ap_int<8> > layer2_0_conv2_pipe_48;
    #pragma HLS stream variable=layer2_0_conv2_pipe_48 depth=8192
    layer2_0_conv2_yy_reuse7: for (bit32 yy_reuse7 = 0; yy_reuse7 < 18; ++yy_reuse7) {
      layer2_0_conv2_xx_reuse7: for (bit32 xx_reuse7 = 0; xx_reuse7 < 18; ++xx_reuse7) {
        loop_layer2_0_conv2_pad_1: for (bit32 layer2_0_conv2_pad_1 = 0; layer2_0_conv2_pad_1 < 2; ++layer2_0_conv2_pad_1) {
          layer2_0_conv2_LB[0][0][layer2_0_conv2_pad_1][xx_reuse7] = layer2_0_conv2_LB[0][0][(layer2_0_conv2_pad_1 + 1)][xx_reuse7];
        }
        ubit32 layer2_0_conv2_pad_temp1;
        layer2_0_conv2_pad_temp1 = layer2_0_conv2_pad_pipe_47.read();
        layer2_0_conv2_LB[0][0][2][xx_reuse7] = layer2_0_conv2_pad_temp1;
        if (2 <= yy_reuse7) {
          loop_layer2_0_conv2_LB_1: for (bit32 layer2_0_conv2_LB_1 = 0; layer2_0_conv2_LB_1 < 3; ++layer2_0_conv2_LB_1) {
            loop_layer2_0_conv2_LB_0: for (bit32 layer2_0_conv2_LB_0 = 0; layer2_0_conv2_LB_0 < 2; ++layer2_0_conv2_LB_0) {
              layer2_0_conv2_WB[0][0][layer2_0_conv2_LB_1][layer2_0_conv2_LB_0] = layer2_0_conv2_WB[0][0][layer2_0_conv2_LB_1][(layer2_0_conv2_LB_0 + 1)];
            }
            layer2_0_conv2_WB[0][0][layer2_0_conv2_LB_1][2] = layer2_0_conv2_LB[0][0][layer2_0_conv2_LB_1][xx_reuse7];
          }
            if (2 <= xx_reuse7) {
          layer2_0_conv2_ff8: for (bit32 ff8 = 0; ff8 < 32; ++ff8) {
      #pragma HLS pipeline
              ap_int<8> layer2_0_conv2_sum;
              layer2_0_conv2_sum = (ap_int<8>)0;
              layer2_0_conv2_layer2_0_conv2_ry: for (bit32 layer2_0_conv2_ry = 0; layer2_0_conv2_ry < 3; ++layer2_0_conv2_ry) {
                layer2_0_conv2_layer2_0_conv2_rx: for (bit32 layer2_0_conv2_rx = 0; layer2_0_conv2_rx < 3; ++layer2_0_conv2_rx) {
                  layer2_0_conv2_layer2_0_conv2_rb: for (bit32 layer2_0_conv2_rb = 0; layer2_0_conv2_rb < 32; ++layer2_0_conv2_rb) {
                    layer2_0_conv2_sum = ((ap_int<8>)(((ap_int<34>)(layer2_0_conv2_WB[0][0][layer2_0_conv2_ry][layer2_0_conv2_rx] ^ w_layer2_0_conv2[ff8][0][layer2_0_conv2_ry][layer2_0_conv2_rx])[layer2_0_conv2_rb]) + ((ap_int<34>)layer2_0_conv2_sum)));
                  }
                }
              }
              ap_int<8> layer2_0_conv2_temp;
              layer2_0_conv2_temp = ((ap_int<8>)(288 - ((bit32)(layer2_0_conv2_sum << 1))));
              layer2_0_conv2_pipe_48.write(layer2_0_conv2_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer2_0_bn2[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_0_bn2_pipe_49;
    #pragma HLS stream variable=layer2_0_bn2_pipe_49 depth=8192
    layer2_0_bn2_args08: for (bit32 args08 = 0; args08 < 32; ++args08) {
      layer2_0_bn2_args18: for (bit32 args18 = 0; args18 < 16; ++args18) {
        layer2_0_bn2_args28: for (bit32 args28 = 0; args28 < 16; ++args28) {
        #pragma HLS pipeline
          ap_int<8> layer2_0_conv2_temp1;
          layer2_0_conv2_temp1 = layer2_0_conv2_pipe_48.read();
          ap_fixed<32, 20> layer2_0_bn2_temp;
          layer2_0_bn2_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer2_0_conv2_temp1) - ((ap_fixed<33, 21>)w_layer2_0_bn2_16[args08]))) / sqrt((((float)w_layer2_0_bn2_17[args08]) + 1.000000e-07f))) * ((float)w_layer2_0_bn2_14[args08])) + ((float)w_layer2_0_bn2_15[args08])));
          layer2_0_bn2_pipe_49.write(layer2_0_bn2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_0_residual2[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_0_residual2_pipe_50;
    #pragma HLS stream variable=layer2_0_residual2_pipe_50 depth=8192
    layer2_0_residual2_cc16: for (bit32 cc16 = 0; cc16 < 32; ++cc16) {
      layer2_0_residual2_ww31: for (bit32 ww31 = 0; ww31 < 16; ++ww31) {
        layer2_0_residual2_hh32: for (bit32 hh32 = 0; hh32 < 16; ++hh32) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_0_bn2_temp1;
          layer2_0_bn2_temp1 = layer2_0_bn2_pipe_49.read();
          ap_fixed<32, 20> layer2_0_residual2_temp;
          ap_fixed<32, 20> layer2_0_rprelu1_temp2;
          layer2_0_rprelu1_temp2 = layer2_0_rprelu1_pipe_124.read();
          layer2_0_residual2_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer2_0_bn2_temp1) + ((ap_fixed<33, 21>)layer2_0_rprelu1_temp2)));
          layer2_0_residual2_pipe_50.write(layer2_0_residual2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_0_rprelu2[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_0_rprelu2_pipe_51;
    #pragma HLS stream variable=layer2_0_rprelu2_pipe_51 depth=8192
    hls::stream<ap_fixed<32, 20> > layer2_0_rprelu2_pipe_125;
    #pragma HLS stream variable=layer2_0_rprelu2_pipe_125 depth=8192
    layer2_0_rprelu2_cc17: for (bit32 cc17 = 0; cc17 < 32; ++cc17) {
      layer2_0_rprelu2_ww32: for (bit32 ww32 = 0; ww32 < 16; ++ww32) {
        layer2_0_rprelu2_hh33: for (bit32 hh33 = 0; hh33 < 16; ++hh33) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_0_residual2_temp1;
          layer2_0_residual2_temp1 = layer2_0_residual2_pipe_50.read();
          ap_fixed<32, 20> layer2_0_rprelu2_temp;
          layer2_0_rprelu2_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_0_residual2_temp1) + ((ap_fixed<33, 21>)w_layer2_0_rprelu2_3[cc17])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer2_0_residual2_temp1) + ((ap_fixed<33, 21>)w_layer2_0_rprelu2_3[cc17])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer2_0_rprelu2_5[cc17]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer2_0_residual2_temp1) + ((ap_fixed<33, 21>)w_layer2_0_rprelu2_3[cc17]))))))) + ((ap_fixed<66, 42>)w_layer2_0_rprelu2_4[cc17])));
          layer2_0_rprelu2_pipe_125.write(layer2_0_rprelu2_temp);
          layer2_0_rprelu2_pipe_51.write(layer2_0_rprelu2_temp);
        }
      }
    }
    ubit32 layer2_1_rsign1[1][1][16][16];
    hls::stream<ubit32 > layer2_1_rsign1_pipe_52;
    #pragma HLS stream variable=layer2_1_rsign1_pipe_52 depth=256
    layer2_1_rsign1_hh34: for (bit32 hh34 = 0; hh34 < 16; ++hh34) {
      layer2_1_rsign1_ww33: for (bit32 ww33 = 0; ww33 < 16; ++ww33) {
      #pragma HLS pipeline
        ubit32 layer2_1_rsign1_pack;
        layer2_1_rsign1_pack = 0U;
        loop_i9: for (bit32 i9 = 0; i9 < 32; ++i9) {
          ap_fixed<32, 20> layer2_0_rprelu2_temp1;
          layer2_0_rprelu2_temp1 = layer2_0_rprelu2_pipe_51.read();
          layer2_1_rsign1_pack(i9, i9) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_0_rprelu2_temp1) + ((ap_fixed<33, 21>)w_layer2_1_rsign1[i9])))) ? ((bit32)1) : ((bit32)0));
        }
        ubit32 layer2_1_rsign1_temp;
        layer2_1_rsign1_temp = layer2_1_rsign1_pack;
        layer2_1_rsign1_pipe_52.write(layer2_1_rsign1_temp);
      }
    }
    ubit32 layer2_1_conv1_pad[1][1][18][18];
    hls::stream<ubit32 > layer2_1_conv1_pad_pipe_53;
    #pragma HLS stream variable=layer2_1_conv1_pad_pipe_53 depth=324
    layer2_1_conv1_pad_hh35: for (bit32 hh35 = 0; hh35 < 18; ++hh35) {
      layer2_1_conv1_pad_ww34: for (bit32 ww34 = 0; ww34 < 18; ++ww34) {
    #pragma HLS pipeline
        ubit32 layer2_1_rsign1_temp1;
        ubit32 layer2_1_conv1_pad_temp;
        if (((((1 <= ww34) && (ww34 < 17)) && (1 <= hh35)) && (hh35 < 17))) { 
          layer2_1_conv1_pad_temp = layer2_1_rsign1_pipe_52.read();
        } else { 
          layer2_1_conv1_pad_temp = 0U;
        }
        layer2_1_conv1_pad_pipe_53.write(layer2_1_conv1_pad_temp);
        layer2_1_conv1_pad[0][0][hh35][ww34] = layer2_1_conv1_pad_temp;
      }
    }
    ap_int<8> layer2_1_conv1[1][32][16][16];
    ubit32 layer2_1_conv1_LB[1][1][3][18];
    ubit32 layer2_1_conv1_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer2_1_conv1_WB complete dim=4
    hls::stream<ap_int<8> > layer2_1_conv1_pipe_54;
    #pragma HLS stream variable=layer2_1_conv1_pipe_54 depth=8192
    layer2_1_conv1_yy_reuse7: for (bit32 yy_reuse7 = 0; yy_reuse7 < 18; ++yy_reuse7) {
      layer2_1_conv1_xx_reuse7: for (bit32 xx_reuse7 = 0; xx_reuse7 < 18; ++xx_reuse7) {
        loop_layer2_1_conv1_pad_1: for (bit32 layer2_1_conv1_pad_1 = 0; layer2_1_conv1_pad_1 < 2; ++layer2_1_conv1_pad_1) {
          layer2_1_conv1_LB[0][0][layer2_1_conv1_pad_1][xx_reuse7] = layer2_1_conv1_LB[0][0][(layer2_1_conv1_pad_1 + 1)][xx_reuse7];
        }
        ubit32 layer2_1_conv1_pad_temp1;
        layer2_1_conv1_pad_temp1 = layer2_1_conv1_pad_pipe_53.read();
        layer2_1_conv1_LB[0][0][2][xx_reuse7] = layer2_1_conv1_pad_temp1;
        if (2 <= yy_reuse7) {
          loop_layer2_1_conv1_LB_1: for (bit32 layer2_1_conv1_LB_1 = 0; layer2_1_conv1_LB_1 < 3; ++layer2_1_conv1_LB_1) {
            loop_layer2_1_conv1_LB_0: for (bit32 layer2_1_conv1_LB_0 = 0; layer2_1_conv1_LB_0 < 2; ++layer2_1_conv1_LB_0) {
              layer2_1_conv1_WB[0][0][layer2_1_conv1_LB_1][layer2_1_conv1_LB_0] = layer2_1_conv1_WB[0][0][layer2_1_conv1_LB_1][(layer2_1_conv1_LB_0 + 1)];
            }
            layer2_1_conv1_WB[0][0][layer2_1_conv1_LB_1][2] = layer2_1_conv1_LB[0][0][layer2_1_conv1_LB_1][xx_reuse7];
          }
            if (2 <= xx_reuse7) {
          layer2_1_conv1_ff9: for (bit32 ff9 = 0; ff9 < 32; ++ff9) {
      #pragma HLS pipeline
              ap_int<8> layer2_1_conv1_sum;
              layer2_1_conv1_sum = (ap_int<8>)0;
              layer2_1_conv1_layer2_1_conv1_ry: for (bit32 layer2_1_conv1_ry = 0; layer2_1_conv1_ry < 3; ++layer2_1_conv1_ry) {
                layer2_1_conv1_layer2_1_conv1_rx: for (bit32 layer2_1_conv1_rx = 0; layer2_1_conv1_rx < 3; ++layer2_1_conv1_rx) {
                  layer2_1_conv1_layer2_1_conv1_rb: for (bit32 layer2_1_conv1_rb = 0; layer2_1_conv1_rb < 32; ++layer2_1_conv1_rb) {
                    layer2_1_conv1_sum = ((ap_int<8>)(((ap_int<34>)(layer2_1_conv1_WB[0][0][layer2_1_conv1_ry][layer2_1_conv1_rx] ^ w_layer2_1_conv1[ff9][0][layer2_1_conv1_ry][layer2_1_conv1_rx])[layer2_1_conv1_rb]) + ((ap_int<34>)layer2_1_conv1_sum)));
                  }
                }
              }
              ap_int<8> layer2_1_conv1_temp;
              layer2_1_conv1_temp = ((ap_int<8>)(288 - ((bit32)(layer2_1_conv1_sum << 1))));
              layer2_1_conv1_pipe_54.write(layer2_1_conv1_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer2_1_bn1[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_1_bn1_pipe_55;
    #pragma HLS stream variable=layer2_1_bn1_pipe_55 depth=8192
    layer2_1_bn1_args09: for (bit32 args09 = 0; args09 < 32; ++args09) {
      layer2_1_bn1_args19: for (bit32 args19 = 0; args19 < 16; ++args19) {
        layer2_1_bn1_args29: for (bit32 args29 = 0; args29 < 16; ++args29) {
        #pragma HLS pipeline
          ap_int<8> layer2_1_conv1_temp1;
          layer2_1_conv1_temp1 = layer2_1_conv1_pipe_54.read();
          ap_fixed<32, 20> layer2_1_bn1_temp;
          layer2_1_bn1_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer2_1_conv1_temp1) - ((ap_fixed<33, 21>)w_layer2_1_bn1_11[args09]))) / sqrt((((float)w_layer2_1_bn1_12[args09]) + 1.000000e-07f))) * ((float)w_layer2_1_bn1_9[args09])) + ((float)w_layer2_1_bn1_10[args09])));
          layer2_1_bn1_pipe_55.write(layer2_1_bn1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_1_residual1[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_1_residual1_pipe_56;
    #pragma HLS stream variable=layer2_1_residual1_pipe_56 depth=8192
    layer2_1_residual1_cc18: for (bit32 cc18 = 0; cc18 < 32; ++cc18) {
      layer2_1_residual1_ww35: for (bit32 ww35 = 0; ww35 < 16; ++ww35) {
        layer2_1_residual1_hh36: for (bit32 hh36 = 0; hh36 < 16; ++hh36) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_1_bn1_temp1;
          layer2_1_bn1_temp1 = layer2_1_bn1_pipe_55.read();
          ap_fixed<32, 20> layer2_1_residual1_temp;
          ap_fixed<32, 20> layer2_0_rprelu2_temp2;
          layer2_0_rprelu2_temp2 = layer2_0_rprelu2_pipe_125.read();
          layer2_1_residual1_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer2_1_bn1_temp1) + ((ap_fixed<33, 21>)layer2_0_rprelu2_temp2)));
          layer2_1_residual1_pipe_56.write(layer2_1_residual1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_1_rprelu1[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_1_rprelu1_pipe_57;
    #pragma HLS stream variable=layer2_1_rprelu1_pipe_57 depth=8192
    hls::stream<ap_fixed<32, 20> > layer2_1_rprelu1_pipe_126;
    #pragma HLS stream variable=layer2_1_rprelu1_pipe_126 depth=8192
    layer2_1_rprelu1_cc19: for (bit32 cc19 = 0; cc19 < 32; ++cc19) {
      layer2_1_rprelu1_ww36: for (bit32 ww36 = 0; ww36 < 16; ++ww36) {
        layer2_1_rprelu1_hh37: for (bit32 hh37 = 0; hh37 < 16; ++hh37) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_1_residual1_temp1;
          layer2_1_residual1_temp1 = layer2_1_residual1_pipe_56.read();
          ap_fixed<32, 20> layer2_1_rprelu1_temp;
          layer2_1_rprelu1_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_1_residual1_temp1) + ((ap_fixed<33, 21>)w_layer2_1_rprelu1_0[cc19])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer2_1_residual1_temp1) + ((ap_fixed<33, 21>)w_layer2_1_rprelu1_0[cc19])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer2_1_rprelu1_2[cc19]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer2_1_residual1_temp1) + ((ap_fixed<33, 21>)w_layer2_1_rprelu1_0[cc19]))))))) + ((ap_fixed<66, 42>)w_layer2_1_rprelu1_1[cc19])));
          layer2_1_rprelu1_pipe_126.write(layer2_1_rprelu1_temp);
          layer2_1_rprelu1_pipe_57.write(layer2_1_rprelu1_temp);
        }
      }
    }
    ubit32 layer2_1_rsign2[1][1][16][16];
    hls::stream<ubit32 > layer2_1_rsign2_pipe_58;
    #pragma HLS stream variable=layer2_1_rsign2_pipe_58 depth=256
    layer2_1_rsign2_hh38: for (bit32 hh38 = 0; hh38 < 16; ++hh38) {
      layer2_1_rsign2_ww37: for (bit32 ww37 = 0; ww37 < 16; ++ww37) {
      #pragma HLS pipeline
        ubit32 layer2_1_rsign2_pack;
        layer2_1_rsign2_pack = 0U;
        loop_i10: for (bit32 i10 = 0; i10 < 32; ++i10) {
          ap_fixed<32, 20> layer2_1_rprelu1_temp1;
          layer2_1_rprelu1_temp1 = layer2_1_rprelu1_pipe_57.read();
          layer2_1_rsign2_pack(i10, i10) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_1_rprelu1_temp1) + ((ap_fixed<33, 21>)w_layer2_1_rsign2[i10])))) ? ((bit32)1) : ((bit32)0));
        }
        ubit32 layer2_1_rsign2_temp;
        layer2_1_rsign2_temp = layer2_1_rsign2_pack;
        layer2_1_rsign2_pipe_58.write(layer2_1_rsign2_temp);
      }
    }
    ubit32 layer2_1_conv2_pad[1][1][18][18];
    hls::stream<ubit32 > layer2_1_conv2_pad_pipe_59;
    #pragma HLS stream variable=layer2_1_conv2_pad_pipe_59 depth=324
    layer2_1_conv2_pad_hh39: for (bit32 hh39 = 0; hh39 < 18; ++hh39) {
      layer2_1_conv2_pad_ww38: for (bit32 ww38 = 0; ww38 < 18; ++ww38) {
    #pragma HLS pipeline
        ubit32 layer2_1_rsign2_temp1;
        ubit32 layer2_1_conv2_pad_temp;
        if (((((1 <= ww38) && (ww38 < 17)) && (1 <= hh39)) && (hh39 < 17))) { 
          layer2_1_conv2_pad_temp = layer2_1_rsign2_pipe_58.read();
        } else { 
          layer2_1_conv2_pad_temp = 0U;
        }
        layer2_1_conv2_pad_pipe_59.write(layer2_1_conv2_pad_temp);
        layer2_1_conv2_pad[0][0][hh39][ww38] = layer2_1_conv2_pad_temp;
      }
    }
    ap_int<8> layer2_1_conv2[1][32][16][16];
    ubit32 layer2_1_conv2_LB[1][1][3][18];
    ubit32 layer2_1_conv2_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer2_1_conv2_WB complete dim=4
    hls::stream<ap_int<8> > layer2_1_conv2_pipe_60;
    #pragma HLS stream variable=layer2_1_conv2_pipe_60 depth=8192
    layer2_1_conv2_yy_reuse8: for (bit32 yy_reuse8 = 0; yy_reuse8 < 18; ++yy_reuse8) {
      layer2_1_conv2_xx_reuse8: for (bit32 xx_reuse8 = 0; xx_reuse8 < 18; ++xx_reuse8) {
        loop_layer2_1_conv2_pad_1: for (bit32 layer2_1_conv2_pad_1 = 0; layer2_1_conv2_pad_1 < 2; ++layer2_1_conv2_pad_1) {
          layer2_1_conv2_LB[0][0][layer2_1_conv2_pad_1][xx_reuse8] = layer2_1_conv2_LB[0][0][(layer2_1_conv2_pad_1 + 1)][xx_reuse8];
        }
        ubit32 layer2_1_conv2_pad_temp1;
        layer2_1_conv2_pad_temp1 = layer2_1_conv2_pad_pipe_59.read();
        layer2_1_conv2_LB[0][0][2][xx_reuse8] = layer2_1_conv2_pad_temp1;
        if (2 <= yy_reuse8) {
          loop_layer2_1_conv2_LB_1: for (bit32 layer2_1_conv2_LB_1 = 0; layer2_1_conv2_LB_1 < 3; ++layer2_1_conv2_LB_1) {
            loop_layer2_1_conv2_LB_0: for (bit32 layer2_1_conv2_LB_0 = 0; layer2_1_conv2_LB_0 < 2; ++layer2_1_conv2_LB_0) {
              layer2_1_conv2_WB[0][0][layer2_1_conv2_LB_1][layer2_1_conv2_LB_0] = layer2_1_conv2_WB[0][0][layer2_1_conv2_LB_1][(layer2_1_conv2_LB_0 + 1)];
            }
            layer2_1_conv2_WB[0][0][layer2_1_conv2_LB_1][2] = layer2_1_conv2_LB[0][0][layer2_1_conv2_LB_1][xx_reuse8];
          }
            if (2 <= xx_reuse8) {
          layer2_1_conv2_ff10: for (bit32 ff10 = 0; ff10 < 32; ++ff10) {
      #pragma HLS pipeline
              ap_int<8> layer2_1_conv2_sum;
              layer2_1_conv2_sum = (ap_int<8>)0;
              layer2_1_conv2_layer2_1_conv2_ry: for (bit32 layer2_1_conv2_ry = 0; layer2_1_conv2_ry < 3; ++layer2_1_conv2_ry) {
                layer2_1_conv2_layer2_1_conv2_rx: for (bit32 layer2_1_conv2_rx = 0; layer2_1_conv2_rx < 3; ++layer2_1_conv2_rx) {
                  layer2_1_conv2_layer2_1_conv2_rb: for (bit32 layer2_1_conv2_rb = 0; layer2_1_conv2_rb < 32; ++layer2_1_conv2_rb) {
                    layer2_1_conv2_sum = ((ap_int<8>)(((ap_int<34>)(layer2_1_conv2_WB[0][0][layer2_1_conv2_ry][layer2_1_conv2_rx] ^ w_layer2_1_conv2[ff10][0][layer2_1_conv2_ry][layer2_1_conv2_rx])[layer2_1_conv2_rb]) + ((ap_int<34>)layer2_1_conv2_sum)));
                  }
                }
              }
              ap_int<8> layer2_1_conv2_temp;
              layer2_1_conv2_temp = ((ap_int<8>)(288 - ((bit32)(layer2_1_conv2_sum << 1))));
              layer2_1_conv2_pipe_60.write(layer2_1_conv2_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer2_1_bn2[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_1_bn2_pipe_61;
    #pragma HLS stream variable=layer2_1_bn2_pipe_61 depth=8192
    layer2_1_bn2_args010: for (bit32 args010 = 0; args010 < 32; ++args010) {
      layer2_1_bn2_args110: for (bit32 args110 = 0; args110 < 16; ++args110) {
        layer2_1_bn2_args210: for (bit32 args210 = 0; args210 < 16; ++args210) {
        #pragma HLS pipeline
          ap_int<8> layer2_1_conv2_temp1;
          layer2_1_conv2_temp1 = layer2_1_conv2_pipe_60.read();
          ap_fixed<32, 20> layer2_1_bn2_temp;
          layer2_1_bn2_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer2_1_conv2_temp1) - ((ap_fixed<33, 21>)w_layer2_1_bn2_16[args010]))) / sqrt((((float)w_layer2_1_bn2_17[args010]) + 1.000000e-07f))) * ((float)w_layer2_1_bn2_14[args010])) + ((float)w_layer2_1_bn2_15[args010])));
          layer2_1_bn2_pipe_61.write(layer2_1_bn2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_1_residual2[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_1_residual2_pipe_62;
    #pragma HLS stream variable=layer2_1_residual2_pipe_62 depth=8192
    layer2_1_residual2_cc20: for (bit32 cc20 = 0; cc20 < 32; ++cc20) {
      layer2_1_residual2_ww39: for (bit32 ww39 = 0; ww39 < 16; ++ww39) {
        layer2_1_residual2_hh40: for (bit32 hh40 = 0; hh40 < 16; ++hh40) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_1_bn2_temp1;
          layer2_1_bn2_temp1 = layer2_1_bn2_pipe_61.read();
          ap_fixed<32, 20> layer2_1_residual2_temp;
          ap_fixed<32, 20> layer2_1_rprelu1_temp2;
          layer2_1_rprelu1_temp2 = layer2_1_rprelu1_pipe_126.read();
          layer2_1_residual2_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer2_1_bn2_temp1) + ((ap_fixed<33, 21>)layer2_1_rprelu1_temp2)));
          layer2_1_residual2_pipe_62.write(layer2_1_residual2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_1_rprelu2[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_1_rprelu2_pipe_63;
    #pragma HLS stream variable=layer2_1_rprelu2_pipe_63 depth=8192
    hls::stream<ap_fixed<32, 20> > layer2_1_rprelu2_pipe_127;
    #pragma HLS stream variable=layer2_1_rprelu2_pipe_127 depth=8192
    layer2_1_rprelu2_cc21: for (bit32 cc21 = 0; cc21 < 32; ++cc21) {
      layer2_1_rprelu2_ww40: for (bit32 ww40 = 0; ww40 < 16; ++ww40) {
        layer2_1_rprelu2_hh41: for (bit32 hh41 = 0; hh41 < 16; ++hh41) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_1_residual2_temp1;
          layer2_1_residual2_temp1 = layer2_1_residual2_pipe_62.read();
          ap_fixed<32, 20> layer2_1_rprelu2_temp;
          layer2_1_rprelu2_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_1_residual2_temp1) + ((ap_fixed<33, 21>)w_layer2_1_rprelu2_3[cc21])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer2_1_residual2_temp1) + ((ap_fixed<33, 21>)w_layer2_1_rprelu2_3[cc21])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer2_1_rprelu2_5[cc21]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer2_1_residual2_temp1) + ((ap_fixed<33, 21>)w_layer2_1_rprelu2_3[cc21]))))))) + ((ap_fixed<66, 42>)w_layer2_1_rprelu2_4[cc21])));
          layer2_1_rprelu2_pipe_127.write(layer2_1_rprelu2_temp);
          layer2_1_rprelu2_pipe_63.write(layer2_1_rprelu2_temp);
        }
      }
    }
    ubit32 layer2_2_rsign1[1][1][16][16];
    hls::stream<ubit32 > layer2_2_rsign1_pipe_64;
    #pragma HLS stream variable=layer2_2_rsign1_pipe_64 depth=256
    layer2_2_rsign1_hh42: for (bit32 hh42 = 0; hh42 < 16; ++hh42) {
      layer2_2_rsign1_ww41: for (bit32 ww41 = 0; ww41 < 16; ++ww41) {
      #pragma HLS pipeline
        ubit32 layer2_2_rsign1_pack;
        layer2_2_rsign1_pack = 0U;
        loop_i11: for (bit32 i11 = 0; i11 < 32; ++i11) {
          ap_fixed<32, 20> layer2_1_rprelu2_temp1;
          layer2_1_rprelu2_temp1 = layer2_1_rprelu2_pipe_63.read();
          layer2_2_rsign1_pack(i11, i11) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_1_rprelu2_temp1) + ((ap_fixed<33, 21>)w_layer2_2_rsign1[i11])))) ? ((bit32)1) : ((bit32)0));
        }
        ubit32 layer2_2_rsign1_temp;
        layer2_2_rsign1_temp = layer2_2_rsign1_pack;
        layer2_2_rsign1_pipe_64.write(layer2_2_rsign1_temp);
      }
    }
    ubit32 layer2_2_conv1_pad[1][1][18][18];
    hls::stream<ubit32 > layer2_2_conv1_pad_pipe_65;
    #pragma HLS stream variable=layer2_2_conv1_pad_pipe_65 depth=324
    layer2_2_conv1_pad_hh43: for (bit32 hh43 = 0; hh43 < 18; ++hh43) {
      layer2_2_conv1_pad_ww42: for (bit32 ww42 = 0; ww42 < 18; ++ww42) {
    #pragma HLS pipeline
        ubit32 layer2_2_rsign1_temp1;
        ubit32 layer2_2_conv1_pad_temp;
        if (((((1 <= ww42) && (ww42 < 17)) && (1 <= hh43)) && (hh43 < 17))) { 
          layer2_2_conv1_pad_temp = layer2_2_rsign1_pipe_64.read();
        } else { 
          layer2_2_conv1_pad_temp = 0U;
        }
        layer2_2_conv1_pad_pipe_65.write(layer2_2_conv1_pad_temp);
        layer2_2_conv1_pad[0][0][hh43][ww42] = layer2_2_conv1_pad_temp;
      }
    }
    ap_int<8> layer2_2_conv1[1][32][16][16];
    ubit32 layer2_2_conv1_LB[1][1][3][18];
    ubit32 layer2_2_conv1_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer2_2_conv1_WB complete dim=4
    hls::stream<ap_int<8> > layer2_2_conv1_pipe_66;
    #pragma HLS stream variable=layer2_2_conv1_pipe_66 depth=8192
    layer2_2_conv1_yy_reuse9: for (bit32 yy_reuse9 = 0; yy_reuse9 < 18; ++yy_reuse9) {
      layer2_2_conv1_xx_reuse9: for (bit32 xx_reuse9 = 0; xx_reuse9 < 18; ++xx_reuse9) {
        loop_layer2_2_conv1_pad_1: for (bit32 layer2_2_conv1_pad_1 = 0; layer2_2_conv1_pad_1 < 2; ++layer2_2_conv1_pad_1) {
          layer2_2_conv1_LB[0][0][layer2_2_conv1_pad_1][xx_reuse9] = layer2_2_conv1_LB[0][0][(layer2_2_conv1_pad_1 + 1)][xx_reuse9];
        }
        ubit32 layer2_2_conv1_pad_temp1;
        layer2_2_conv1_pad_temp1 = layer2_2_conv1_pad_pipe_65.read();
        layer2_2_conv1_LB[0][0][2][xx_reuse9] = layer2_2_conv1_pad_temp1;
        if (2 <= yy_reuse9) {
          loop_layer2_2_conv1_LB_1: for (bit32 layer2_2_conv1_LB_1 = 0; layer2_2_conv1_LB_1 < 3; ++layer2_2_conv1_LB_1) {
            loop_layer2_2_conv1_LB_0: for (bit32 layer2_2_conv1_LB_0 = 0; layer2_2_conv1_LB_0 < 2; ++layer2_2_conv1_LB_0) {
              layer2_2_conv1_WB[0][0][layer2_2_conv1_LB_1][layer2_2_conv1_LB_0] = layer2_2_conv1_WB[0][0][layer2_2_conv1_LB_1][(layer2_2_conv1_LB_0 + 1)];
            }
            layer2_2_conv1_WB[0][0][layer2_2_conv1_LB_1][2] = layer2_2_conv1_LB[0][0][layer2_2_conv1_LB_1][xx_reuse9];
          }
            if (2 <= xx_reuse9) {
          layer2_2_conv1_ff11: for (bit32 ff11 = 0; ff11 < 32; ++ff11) {
      #pragma HLS pipeline
              ap_int<8> layer2_2_conv1_sum;
              layer2_2_conv1_sum = (ap_int<8>)0;
              layer2_2_conv1_layer2_2_conv1_ry: for (bit32 layer2_2_conv1_ry = 0; layer2_2_conv1_ry < 3; ++layer2_2_conv1_ry) {
                layer2_2_conv1_layer2_2_conv1_rx: for (bit32 layer2_2_conv1_rx = 0; layer2_2_conv1_rx < 3; ++layer2_2_conv1_rx) {
                  layer2_2_conv1_layer2_2_conv1_rb: for (bit32 layer2_2_conv1_rb = 0; layer2_2_conv1_rb < 32; ++layer2_2_conv1_rb) {
                    layer2_2_conv1_sum = ((ap_int<8>)(((ap_int<34>)(layer2_2_conv1_WB[0][0][layer2_2_conv1_ry][layer2_2_conv1_rx] ^ w_layer2_2_conv1[ff11][0][layer2_2_conv1_ry][layer2_2_conv1_rx])[layer2_2_conv1_rb]) + ((ap_int<34>)layer2_2_conv1_sum)));
                  }
                }
              }
              ap_int<8> layer2_2_conv1_temp;
              layer2_2_conv1_temp = ((ap_int<8>)(288 - ((bit32)(layer2_2_conv1_sum << 1))));
              layer2_2_conv1_pipe_66.write(layer2_2_conv1_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer2_2_bn1[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_2_bn1_pipe_67;
    #pragma HLS stream variable=layer2_2_bn1_pipe_67 depth=8192
    layer2_2_bn1_args011: for (bit32 args011 = 0; args011 < 32; ++args011) {
      layer2_2_bn1_args111: for (bit32 args111 = 0; args111 < 16; ++args111) {
        layer2_2_bn1_args211: for (bit32 args211 = 0; args211 < 16; ++args211) {
        #pragma HLS pipeline
          ap_int<8> layer2_2_conv1_temp1;
          layer2_2_conv1_temp1 = layer2_2_conv1_pipe_66.read();
          ap_fixed<32, 20> layer2_2_bn1_temp;
          layer2_2_bn1_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer2_2_conv1_temp1) - ((ap_fixed<33, 21>)w_layer2_2_bn1_11[args011]))) / sqrt((((float)w_layer2_2_bn1_12[args011]) + 1.000000e-07f))) * ((float)w_layer2_2_bn1_9[args011])) + ((float)w_layer2_2_bn1_10[args011])));
          layer2_2_bn1_pipe_67.write(layer2_2_bn1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_2_residual1[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_2_residual1_pipe_68;
    #pragma HLS stream variable=layer2_2_residual1_pipe_68 depth=8192
    layer2_2_residual1_cc22: for (bit32 cc22 = 0; cc22 < 32; ++cc22) {
      layer2_2_residual1_ww43: for (bit32 ww43 = 0; ww43 < 16; ++ww43) {
        layer2_2_residual1_hh44: for (bit32 hh44 = 0; hh44 < 16; ++hh44) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_2_bn1_temp1;
          layer2_2_bn1_temp1 = layer2_2_bn1_pipe_67.read();
          ap_fixed<32, 20> layer2_2_residual1_temp;
          ap_fixed<32, 20> layer2_1_rprelu2_temp2;
          layer2_1_rprelu2_temp2 = layer2_1_rprelu2_pipe_127.read();
          layer2_2_residual1_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer2_2_bn1_temp1) + ((ap_fixed<33, 21>)layer2_1_rprelu2_temp2)));
          layer2_2_residual1_pipe_68.write(layer2_2_residual1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_2_rprelu1[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_2_rprelu1_pipe_69;
    #pragma HLS stream variable=layer2_2_rprelu1_pipe_69 depth=8192
    hls::stream<ap_fixed<32, 20> > layer2_2_rprelu1_pipe_128;
    #pragma HLS stream variable=layer2_2_rprelu1_pipe_128 depth=8192
    layer2_2_rprelu1_cc23: for (bit32 cc23 = 0; cc23 < 32; ++cc23) {
      layer2_2_rprelu1_ww44: for (bit32 ww44 = 0; ww44 < 16; ++ww44) {
        layer2_2_rprelu1_hh45: for (bit32 hh45 = 0; hh45 < 16; ++hh45) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_2_residual1_temp1;
          layer2_2_residual1_temp1 = layer2_2_residual1_pipe_68.read();
          ap_fixed<32, 20> layer2_2_rprelu1_temp;
          layer2_2_rprelu1_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_2_residual1_temp1) + ((ap_fixed<33, 21>)w_layer2_2_rprelu1_0[cc23])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer2_2_residual1_temp1) + ((ap_fixed<33, 21>)w_layer2_2_rprelu1_0[cc23])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer2_2_rprelu1_2[cc23]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer2_2_residual1_temp1) + ((ap_fixed<33, 21>)w_layer2_2_rprelu1_0[cc23]))))))) + ((ap_fixed<66, 42>)w_layer2_2_rprelu1_1[cc23])));
          layer2_2_rprelu1_pipe_128.write(layer2_2_rprelu1_temp);
          layer2_2_rprelu1_pipe_69.write(layer2_2_rprelu1_temp);
        }
      }
    }
    ubit32 layer2_2_rsign2[1][1][16][16];
    hls::stream<ubit32 > layer2_2_rsign2_pipe_70;
    #pragma HLS stream variable=layer2_2_rsign2_pipe_70 depth=256
    layer2_2_rsign2_hh46: for (bit32 hh46 = 0; hh46 < 16; ++hh46) {
      layer2_2_rsign2_ww45: for (bit32 ww45 = 0; ww45 < 16; ++ww45) {
      #pragma HLS pipeline
        ubit32 layer2_2_rsign2_pack;
        layer2_2_rsign2_pack = 0U;
        loop_i12: for (bit32 i12 = 0; i12 < 32; ++i12) {
          ap_fixed<32, 20> layer2_2_rprelu1_temp1;
          layer2_2_rprelu1_temp1 = layer2_2_rprelu1_pipe_69.read();
          layer2_2_rsign2_pack(i12, i12) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_2_rprelu1_temp1) + ((ap_fixed<33, 21>)w_layer2_2_rsign2[i12])))) ? ((bit32)1) : ((bit32)0));
        }
        ubit32 layer2_2_rsign2_temp;
        layer2_2_rsign2_temp = layer2_2_rsign2_pack;
        layer2_2_rsign2_pipe_70.write(layer2_2_rsign2_temp);
      }
    }
    ubit32 layer2_2_conv2_pad[1][1][18][18];
    hls::stream<ubit32 > layer2_2_conv2_pad_pipe_71;
    #pragma HLS stream variable=layer2_2_conv2_pad_pipe_71 depth=324
    layer2_2_conv2_pad_hh47: for (bit32 hh47 = 0; hh47 < 18; ++hh47) {
      layer2_2_conv2_pad_ww46: for (bit32 ww46 = 0; ww46 < 18; ++ww46) {
    #pragma HLS pipeline
        ubit32 layer2_2_rsign2_temp1;
        ubit32 layer2_2_conv2_pad_temp;
        if (((((1 <= ww46) && (ww46 < 17)) && (1 <= hh47)) && (hh47 < 17))) { 
          layer2_2_conv2_pad_temp = layer2_2_rsign2_pipe_70.read();
        } else { 
          layer2_2_conv2_pad_temp = 0U;
        }
        layer2_2_conv2_pad_pipe_71.write(layer2_2_conv2_pad_temp);
        layer2_2_conv2_pad[0][0][hh47][ww46] = layer2_2_conv2_pad_temp;
      }
    }
    ap_int<8> layer2_2_conv2[1][32][16][16];
    ubit32 layer2_2_conv2_LB[1][1][3][18];
    ubit32 layer2_2_conv2_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer2_2_conv2_WB complete dim=4
    hls::stream<ap_int<8> > layer2_2_conv2_pipe_72;
    #pragma HLS stream variable=layer2_2_conv2_pipe_72 depth=8192
    layer2_2_conv2_yy_reuse10: for (bit32 yy_reuse10 = 0; yy_reuse10 < 18; ++yy_reuse10) {
      layer2_2_conv2_xx_reuse10: for (bit32 xx_reuse10 = 0; xx_reuse10 < 18; ++xx_reuse10) {
        loop_layer2_2_conv2_pad_1: for (bit32 layer2_2_conv2_pad_1 = 0; layer2_2_conv2_pad_1 < 2; ++layer2_2_conv2_pad_1) {
          layer2_2_conv2_LB[0][0][layer2_2_conv2_pad_1][xx_reuse10] = layer2_2_conv2_LB[0][0][(layer2_2_conv2_pad_1 + 1)][xx_reuse10];
        }
        ubit32 layer2_2_conv2_pad_temp1;
        layer2_2_conv2_pad_temp1 = layer2_2_conv2_pad_pipe_71.read();
        layer2_2_conv2_LB[0][0][2][xx_reuse10] = layer2_2_conv2_pad_temp1;
        if (2 <= yy_reuse10) {
          loop_layer2_2_conv2_LB_1: for (bit32 layer2_2_conv2_LB_1 = 0; layer2_2_conv2_LB_1 < 3; ++layer2_2_conv2_LB_1) {
            loop_layer2_2_conv2_LB_0: for (bit32 layer2_2_conv2_LB_0 = 0; layer2_2_conv2_LB_0 < 2; ++layer2_2_conv2_LB_0) {
              layer2_2_conv2_WB[0][0][layer2_2_conv2_LB_1][layer2_2_conv2_LB_0] = layer2_2_conv2_WB[0][0][layer2_2_conv2_LB_1][(layer2_2_conv2_LB_0 + 1)];
            }
            layer2_2_conv2_WB[0][0][layer2_2_conv2_LB_1][2] = layer2_2_conv2_LB[0][0][layer2_2_conv2_LB_1][xx_reuse10];
          }
            if (2 <= xx_reuse10) {
          layer2_2_conv2_ff12: for (bit32 ff12 = 0; ff12 < 32; ++ff12) {
      #pragma HLS pipeline
              ap_int<8> layer2_2_conv2_sum;
              layer2_2_conv2_sum = (ap_int<8>)0;
              layer2_2_conv2_layer2_2_conv2_ry: for (bit32 layer2_2_conv2_ry = 0; layer2_2_conv2_ry < 3; ++layer2_2_conv2_ry) {
                layer2_2_conv2_layer2_2_conv2_rx: for (bit32 layer2_2_conv2_rx = 0; layer2_2_conv2_rx < 3; ++layer2_2_conv2_rx) {
                  layer2_2_conv2_layer2_2_conv2_rb: for (bit32 layer2_2_conv2_rb = 0; layer2_2_conv2_rb < 32; ++layer2_2_conv2_rb) {
                    layer2_2_conv2_sum = ((ap_int<8>)(((ap_int<34>)(layer2_2_conv2_WB[0][0][layer2_2_conv2_ry][layer2_2_conv2_rx] ^ w_layer2_2_conv2[ff12][0][layer2_2_conv2_ry][layer2_2_conv2_rx])[layer2_2_conv2_rb]) + ((ap_int<34>)layer2_2_conv2_sum)));
                  }
                }
              }
              ap_int<8> layer2_2_conv2_temp;
              layer2_2_conv2_temp = ((ap_int<8>)(288 - ((bit32)(layer2_2_conv2_sum << 1))));
              layer2_2_conv2_pipe_72.write(layer2_2_conv2_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer2_2_bn2[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_2_bn2_pipe_73;
    #pragma HLS stream variable=layer2_2_bn2_pipe_73 depth=8192
    layer2_2_bn2_args012: for (bit32 args012 = 0; args012 < 32; ++args012) {
      layer2_2_bn2_args112: for (bit32 args112 = 0; args112 < 16; ++args112) {
        layer2_2_bn2_args212: for (bit32 args212 = 0; args212 < 16; ++args212) {
        #pragma HLS pipeline
          ap_int<8> layer2_2_conv2_temp1;
          layer2_2_conv2_temp1 = layer2_2_conv2_pipe_72.read();
          ap_fixed<32, 20> layer2_2_bn2_temp;
          layer2_2_bn2_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer2_2_conv2_temp1) - ((ap_fixed<33, 21>)w_layer2_2_bn2_16[args012]))) / sqrt((((float)w_layer2_2_bn2_17[args012]) + 1.000000e-07f))) * ((float)w_layer2_2_bn2_14[args012])) + ((float)w_layer2_2_bn2_15[args012])));
          layer2_2_bn2_pipe_73.write(layer2_2_bn2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_2_residual2[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_2_residual2_pipe_74;
    #pragma HLS stream variable=layer2_2_residual2_pipe_74 depth=8192
    layer2_2_residual2_cc24: for (bit32 cc24 = 0; cc24 < 32; ++cc24) {
      layer2_2_residual2_ww47: for (bit32 ww47 = 0; ww47 < 16; ++ww47) {
        layer2_2_residual2_hh48: for (bit32 hh48 = 0; hh48 < 16; ++hh48) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_2_bn2_temp1;
          layer2_2_bn2_temp1 = layer2_2_bn2_pipe_73.read();
          ap_fixed<32, 20> layer2_2_residual2_temp;
          ap_fixed<32, 20> layer2_2_rprelu1_temp2;
          layer2_2_rprelu1_temp2 = layer2_2_rprelu1_pipe_128.read();
          layer2_2_residual2_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer2_2_bn2_temp1) + ((ap_fixed<33, 21>)layer2_2_rprelu1_temp2)));
          layer2_2_residual2_pipe_74.write(layer2_2_residual2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer2_2_rprelu2[1][32][16][16];
    hls::stream<ap_fixed<32, 20> > layer2_2_rprelu2_pipe_75;
    #pragma HLS stream variable=layer2_2_rprelu2_pipe_75 depth=8192
    hls::stream<ap_fixed<32, 20> > layer2_2_rprelu2_pipe_129;
    #pragma HLS stream variable=layer2_2_rprelu2_pipe_129 depth=8192
    layer2_2_rprelu2_cc25: for (bit32 cc25 = 0; cc25 < 32; ++cc25) {
      layer2_2_rprelu2_ww48: for (bit32 ww48 = 0; ww48 < 16; ++ww48) {
        layer2_2_rprelu2_hh49: for (bit32 hh49 = 0; hh49 < 16; ++hh49) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer2_2_residual2_temp1;
          layer2_2_residual2_temp1 = layer2_2_residual2_pipe_74.read();
          ap_fixed<32, 20> layer2_2_rprelu2_temp;
          layer2_2_rprelu2_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_2_residual2_temp1) + ((ap_fixed<33, 21>)w_layer2_2_rprelu2_3[cc25])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer2_2_residual2_temp1) + ((ap_fixed<33, 21>)w_layer2_2_rprelu2_3[cc25])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer2_2_rprelu2_5[cc25]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer2_2_residual2_temp1) + ((ap_fixed<33, 21>)w_layer2_2_rprelu2_3[cc25]))))))) + ((ap_fixed<66, 42>)w_layer2_2_rprelu2_4[cc25])));
          layer2_2_rprelu2_pipe_129.write(layer2_2_rprelu2_temp);
          layer2_2_rprelu2_pipe_75.write(layer2_2_rprelu2_temp);
        }
      }
    }
    ubit32 layer3_0_rsign1[1][1][16][16];
    hls::stream<ubit32 > layer3_0_rsign1_pipe_76;
    #pragma HLS stream variable=layer3_0_rsign1_pipe_76 depth=256
    layer3_0_rsign1_hh50: for (bit32 hh50 = 0; hh50 < 16; ++hh50) {
      layer3_0_rsign1_ww49: for (bit32 ww49 = 0; ww49 < 16; ++ww49) {
      #pragma HLS pipeline
        ubit32 layer3_0_rsign1_pack;
        layer3_0_rsign1_pack = 0U;
        loop_i13: for (bit32 i13 = 0; i13 < 32; ++i13) {
          ap_fixed<32, 20> layer2_2_rprelu2_temp1;
          layer2_2_rprelu2_temp1 = layer2_2_rprelu2_pipe_75.read();
          layer3_0_rsign1_pack(i13, i13) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer2_2_rprelu2_temp1) + ((ap_fixed<33, 21>)w_layer3_0_rsign1[i13])))) ? ((bit32)1) : ((bit32)0));
        }
        ubit32 layer3_0_rsign1_temp;
        layer3_0_rsign1_temp = layer3_0_rsign1_pack;
        layer3_0_rsign1_pipe_76.write(layer3_0_rsign1_temp);
      }
    }
    ubit32 layer3_0_conv1_pad[1][1][18][18];
    hls::stream<ubit32 > layer3_0_conv1_pad_pipe_77;
    #pragma HLS stream variable=layer3_0_conv1_pad_pipe_77 depth=324
    layer3_0_conv1_pad_hh: for (bit32 hh = 0; hh < 18; ++hh) {
      layer3_0_conv1_pad_ww: for (bit32 ww = 0; ww < 18; ++ww) {
      #pragma HLS pipeline
        ubit32 layer3_0_conv1_pad_temp;
        if (((((1 <= ww) && (ww < 17)) && (1 <= hh)) && (hh < 17))) { 
          layer3_0_conv1_pad_temp = layer3_0_rsign1_pipe_76.read();
        } else { 
          layer3_0_conv1_pad_temp = 0U;
        }
        layer3_0_conv1_pad_pipe_77.write(layer3_0_conv1_pad_temp);
      }
    }
    ubit32 layer3_0_conv1_LB[1][1][3][18];
    ubit32 layer3_0_conv1_WB[1][1][3][3];
    // #pragma HLS array_partition variable=layer3_0_conv1_WB complete dim=4
    hls::stream<ap_int<8> > layer3_0_conv1_pipe_78;
    #pragma HLS stream variable=layer3_0_conv1_pipe_78 depth=4096
      layer3_0_conv1_yy_reuse: for (bit32 yy_reuse = 0; yy_reuse < 18; ++yy_reuse) {
        layer3_0_conv1_xx_reuse: for (bit32 xx_reuse = 0; xx_reuse < 18; ++xx_reuse) {
          loop_layer3_0_conv1_pad_1: for (bit32 layer3_0_conv1_pad_1 = 0; layer3_0_conv1_pad_1 < 2; ++layer3_0_conv1_pad_1) {
            layer3_0_conv1_LB[0][0][layer3_0_conv1_pad_1][xx_reuse] = layer3_0_conv1_LB[0][0][(layer3_0_conv1_pad_1 + 1)][xx_reuse];
          }
          layer3_0_conv1_LB[0][0][2][xx_reuse] = layer3_0_conv1_pad_pipe_77.read();
          if (2 <= yy_reuse && (yy_reuse - 2) % 2 == 0) {
            loop_layer3_0_conv1_LB_1: for (bit32 layer3_0_conv1_LB_1 = 0; layer3_0_conv1_LB_1 < 3; ++layer3_0_conv1_LB_1) {
              loop_layer3_0_conv1_LB_0: for (bit32 layer3_0_conv1_LB_0 = 0; layer3_0_conv1_LB_0 < 2; ++layer3_0_conv1_LB_0) {
                layer3_0_conv1_WB[0][0][layer3_0_conv1_LB_1][layer3_0_conv1_LB_0] = layer3_0_conv1_WB[0][0][layer3_0_conv1_LB_1][(layer3_0_conv1_LB_0 + 1)];
              }
              layer3_0_conv1_WB[0][0][layer3_0_conv1_LB_1][2] = layer3_0_conv1_LB[0][0][layer3_0_conv1_LB_1][xx_reuse];
            }
            if (2 <= xx_reuse && (xx_reuse - 2) % 2 == 0) {
    layer3_0_conv1_ff: for (bit32 ff = 0; ff < 64; ++ff) {
      #pragma HLS pipeline
              ap_uint<16> layer3_0_conv1_sum;
              layer3_0_conv1_sum = (ap_uint<16>)0;
              layer3_0_conv1_layer3_0_conv1_ry: for (bit32 layer3_0_conv1_ry = 0; layer3_0_conv1_ry < 3; ++layer3_0_conv1_ry) {
                layer3_0_conv1_layer3_0_conv1_rx: for (bit32 layer3_0_conv1_rx = 0; layer3_0_conv1_rx < 3; ++layer3_0_conv1_rx) {
                  layer3_0_conv1_layer3_0_conv1_rb: for (bit32 layer3_0_conv1_rb = 0; layer3_0_conv1_rb < 32; ++layer3_0_conv1_rb) {
                    layer3_0_conv1_sum = ((ap_uint<16>)(((ap_uint<33>)(layer3_0_conv1_WB[0][0][layer3_0_conv1_ry][layer3_0_conv1_rx] ^ w_layer3_0_conv1[ff][0][layer3_0_conv1_ry][layer3_0_conv1_rx])[layer3_0_conv1_rb]) + ((ap_uint<33>)layer3_0_conv1_sum)));
                  }
                }
              }
              layer3_0_conv1_pipe_78.write((ap_uint<16>)(288U - ((ubit32)(layer3_0_conv1_sum << 1))));
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer3_0_bn1[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_0_bn1_pipe_79;
    #pragma HLS stream variable=layer3_0_bn1_pipe_79 depth=4096
    layer3_0_bn1_args013: for (bit32 args013 = 0; args013 < 64; ++args013) {
      layer3_0_bn1_args113: for (bit32 args113 = 0; args113 < 8; ++args113) {
        layer3_0_bn1_args213: for (bit32 args213 = 0; args213 < 8; ++args213) {
        #pragma HLS pipeline
          ap_int<8> layer3_0_conv1_temp1;
          layer3_0_conv1_temp1 = layer3_0_conv1_pipe_78.read();
          ap_fixed<32, 20> layer3_0_bn1_temp;
          layer3_0_bn1_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer3_0_conv1_temp1) - ((ap_fixed<33, 21>)w_layer3_0_bn1_11[args013]))) / sqrt((((float)w_layer3_0_bn1_12[args013]) + 1.000000e-07f))) * ((float)w_layer3_0_bn1_9[args013])) + ((float)w_layer3_0_bn1_10[args013])));
          layer3_0_bn1_pipe_79.write(layer3_0_bn1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_0_avgpool_res[1][32][8][8];
    ap_fixed<32, 20> layer3_0_avgpool_LB[2][16];
    bit32 layer3_0_avgpool;
    hls::stream<ap_fixed<32, 20> > layer3_0_avgpool_res_pipe_130;
    #pragma HLS stream variable=layer3_0_avgpool_res_pipe_130 depth=2048
    layer3_0_avgpool_cc26: for (bit32 cc26 = 0; cc26 < 32; ++cc26) {
      layer3_0_avgpool_hh52: for (bit32 hh52 = 0; hh52 < 8; ++hh52) {
      #pragma HLS pipeline
        loop_layer3_0_avgpool_LB_i: for (bit32 layer3_0_avgpool_LB_i = 0; layer3_0_avgpool_LB_i < 2; ++layer3_0_avgpool_LB_i) {
          loop_layer3_0_avgpool_LB_j: for (bit32 layer3_0_avgpool_LB_j = 0; layer3_0_avgpool_LB_j < 16; ++layer3_0_avgpool_LB_j) {
            ap_fixed<32, 20> layer2_2_rprelu2_temp2;
            layer2_2_rprelu2_temp2 = layer2_2_rprelu2_pipe_129.read();
            layer3_0_avgpool_LB[layer3_0_avgpool_LB_i][layer3_0_avgpool_LB_j] = layer2_2_rprelu2_temp2;
          }
        }
        loop_layer3_0_avgpool_ww: for (bit32 layer3_0_avgpool_ww = 0; layer3_0_avgpool_ww < 8; ++layer3_0_avgpool_ww) {
          ap_fixed<32, 20> layer3_0_avgpool_val;
          layer3_0_avgpool_val = ((ap_fixed<32, 20>)0);
          loop_layer3_0_avgpool_ry: for (bit32 layer3_0_avgpool_ry = 0; layer3_0_avgpool_ry < 2; ++layer3_0_avgpool_ry) {
            loop_layer3_0_avgpool_rx: for (bit32 layer3_0_avgpool_rx = 0; layer3_0_avgpool_rx < 2; ++layer3_0_avgpool_rx) {
              layer3_0_avgpool_val = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer3_0_avgpool_val) + ((ap_fixed<33, 21>)layer3_0_avgpool_LB[layer3_0_avgpool_ry][((layer3_0_avgpool_ww * 2) + layer3_0_avgpool_rx)])));
            }
          }
          ap_fixed<32, 20> layer3_0_avgpool_res_temp;
          layer3_0_avgpool_res_temp = ((ap_fixed<32, 20>)(((ap_fixed<64, 20>)layer3_0_avgpool_val) / (ap_fixed<64, 20>)4));
          layer3_0_avgpool_res_pipe_130.write(layer3_0_avgpool_res_temp);
          layer3_0_avgpool_res[0][cc26][hh52][layer3_0_avgpool_ww] = layer3_0_avgpool_res_temp;
        }
      }
    }
    ap_fixed<32, 20> layer3_0_concat[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_0_concat_pipe_131;
    #pragma HLS stream variable=layer3_0_concat_pipe_131 depth=4096
    layer3_0_concat_cc27: for (bit32 cc27 = 0; cc27 < 32; ++cc27) { // 2 times
      layer3_0_concat_ww51: for (bit32 ww51 = 0; ww51 < 8; ++ww51) {
      #pragma HLS pipeline
        layer3_0_concat_hh53: for (bit32 hh53 = 0; hh53 < 8; ++hh53) {
          ap_fixed<32, 20> layer3_0_avgpool_res_temp1;
          layer3_0_avgpool_res_temp1 = layer3_0_avgpool_res_pipe_130.read();
          ap_fixed<32, 20> layer3_0_concat_temp;
          layer3_0_concat_temp = layer3_0_avgpool_res_temp1;
          layer3_0_concat_pipe_131.write(layer3_0_concat_temp);
          layer3_0_concat_pipe_131.write(layer3_0_concat_temp); // 2 times, incorrect
        }
      }
    }
    ap_fixed<32, 20> layer3_0_residual1[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_0_residual1_pipe_80;
    #pragma HLS stream variable=layer3_0_residual1_pipe_80 depth=4096
    layer3_0_residual1_cc28: for (bit32 cc28 = 0; cc28 < 64; ++cc28) {
      layer3_0_residual1_ww52: for (bit32 ww52 = 0; ww52 < 8; ++ww52) {
        layer3_0_residual1_hh54: for (bit32 hh54 = 0; hh54 < 8; ++hh54) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_0_bn1_temp1;
          layer3_0_bn1_temp1 = layer3_0_bn1_pipe_79.read();
          ap_fixed<32, 20> layer3_0_residual1_temp;
          ap_fixed<32, 20> layer3_0_concat_temp1;
          layer3_0_concat_temp1 = layer3_0_concat_pipe_131.read();
          layer3_0_residual1_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer3_0_bn1_temp1) + ((ap_fixed<33, 21>)layer3_0_concat_temp1)));
          layer3_0_residual1_pipe_80.write(layer3_0_residual1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_0_rprelu1[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_0_rprelu1_pipe_81;
    #pragma HLS stream variable=layer3_0_rprelu1_pipe_81 depth=4096
    hls::stream<ap_fixed<32, 20> > layer3_0_rprelu1_pipe_132;
    #pragma HLS stream variable=layer3_0_rprelu1_pipe_132 depth=4096
    layer3_0_rprelu1_cc29: for (bit32 cc29 = 0; cc29 < 64; ++cc29) {
      layer3_0_rprelu1_ww53: for (bit32 ww53 = 0; ww53 < 8; ++ww53) {
        layer3_0_rprelu1_hh55: for (bit32 hh55 = 0; hh55 < 8; ++hh55) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_0_residual1_temp1;
          layer3_0_residual1_temp1 = layer3_0_residual1_pipe_80.read();
          ap_fixed<32, 20> layer3_0_rprelu1_temp;
          layer3_0_rprelu1_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_0_residual1_temp1) + ((ap_fixed<33, 21>)w_layer3_0_rprelu1_0[cc29])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer3_0_residual1_temp1) + ((ap_fixed<33, 21>)w_layer3_0_rprelu1_0[cc29])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer3_0_rprelu1_2[cc29]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer3_0_residual1_temp1) + ((ap_fixed<33, 21>)w_layer3_0_rprelu1_0[cc29]))))))) + ((ap_fixed<66, 42>)w_layer3_0_rprelu1_1[cc29])));
          layer3_0_rprelu1_pipe_132.write(layer3_0_rprelu1_temp);
          layer3_0_rprelu1_pipe_81.write(layer3_0_rprelu1_temp);
        }
      }
    }
    ubit32 layer3_0_rsign2[1][2][8][8];
    hls::stream<ubit32 > layer3_0_rsign2_pipe_82;
    #pragma HLS stream variable=layer3_0_rsign2_pipe_82 depth=128
    layer3_0_rsign2_cc30: for (bit32 cc30 = 0; cc30 < 2; ++cc30) {
      layer3_0_rsign2_hh56: for (bit32 hh56 = 0; hh56 < 8; ++hh56) {
        layer3_0_rsign2_ww54: for (bit32 ww54 = 0; ww54 < 8; ++ww54) {
        #pragma HLS pipeline
          ubit32 layer3_0_rsign2_pack;
          layer3_0_rsign2_pack = 0U;
          loop_i14: for (bit32 i14 = 0; i14 < 32; ++i14) {
            ap_fixed<32, 20> layer3_0_rprelu1_temp1;
            layer3_0_rprelu1_temp1 = layer3_0_rprelu1_pipe_81.read();
            layer3_0_rsign2_pack(i14, i14) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_0_rprelu1_temp1) + ((ap_fixed<33, 21>)w_layer3_0_rsign2[((cc30 * 32) + i14)])))) ? ((bit32)1) : ((bit32)0));
          }
          ubit32 layer3_0_rsign2_temp;
          layer3_0_rsign2_temp = layer3_0_rsign2_pack;
          layer3_0_rsign2_pipe_82.write(layer3_0_rsign2_temp);
        }
      }
    }
    ubit32 layer3_0_conv2_pad[1][2][10][10];
    hls::stream<ubit32 > layer3_0_conv2_pad_pipe_83;
    #pragma HLS stream variable=layer3_0_conv2_pad_pipe_83 depth=200
    layer3_0_conv2_pad_cc31: for (bit32 cc31 = 0; cc31 < 2; ++cc31) {
      layer3_0_conv2_pad_hh57: for (bit32 hh57 = 0; hh57 < 10; ++hh57) {
      #pragma HLS pipeline
        layer3_0_conv2_pad_ww55: for (bit32 ww55 = 0; ww55 < 10; ++ww55) {
          ubit32 layer3_0_rsign2_temp1;
          ubit32 layer3_0_conv2_pad_temp;
          if (((((1 <= ww55) && (ww55 < 9)) && (1 <= hh57)) && (hh57 < 9))) { 
            layer3_0_conv2_pad_temp = layer3_0_rsign2_pipe_82.read();
          } else { 
            layer3_0_conv2_pad_temp = 0U;
          }
          layer3_0_conv2_pad_pipe_83.write(layer3_0_conv2_pad_temp);
        }
      }
    }
    ap_int<8> layer3_0_conv2[1][64][8][8];
    ubit32 layer3_0_conv2_LB[1][2][3][10];
    ubit32 layer3_0_conv2_WB[1][2][3][3];
    // #pragma HLS array_partition variable=layer3_0_conv2_WB complete dim=4
    hls::stream<ap_int<8> > layer3_0_conv2_pipe_84;
    #pragma HLS stream variable=layer3_0_conv2_pipe_84 depth=4096
    layer3_0_conv2_yy_reuse12: for (bit32 yy_reuse12 = 0; yy_reuse12 < 10; ++yy_reuse12) {
      layer3_0_conv2_xx_reuse12: for (bit32 xx_reuse12 = 0; xx_reuse12 < 10; ++xx_reuse12) {
        loop_layer3_0_conv2_pad_2: for (bit32 layer3_0_conv2_pad_2 = 0; layer3_0_conv2_pad_2 < 2; ++layer3_0_conv2_pad_2) {
          loop_layer3_0_conv2_pad_1: for (bit32 layer3_0_conv2_pad_1 = 0; layer3_0_conv2_pad_1 < 2; ++layer3_0_conv2_pad_1) {
            layer3_0_conv2_LB[0][layer3_0_conv2_pad_2][layer3_0_conv2_pad_1][xx_reuse12] = layer3_0_conv2_LB[0][layer3_0_conv2_pad_2][(layer3_0_conv2_pad_1 + 1)][xx_reuse12];
          }
          ubit32 layer3_0_conv2_pad_temp1;
          layer3_0_conv2_pad_temp1 = layer3_0_conv2_pad_pipe_83.read();
          layer3_0_conv2_LB[0][layer3_0_conv2_pad_2][2][xx_reuse12] = layer3_0_conv2_pad_temp1;
        }
        if (2 <= yy_reuse12) {
          loop_layer3_0_conv2_LB_1: for (bit32 layer3_0_conv2_LB_1 = 0; layer3_0_conv2_LB_1 < 3; ++layer3_0_conv2_LB_1) {
            loop_layer3_0_conv2_LB_2: for (bit32 layer3_0_conv2_LB_2 = 0; layer3_0_conv2_LB_2 < 2; ++layer3_0_conv2_LB_2) {
              loop_layer3_0_conv2_LB_0: for (bit32 layer3_0_conv2_LB_0 = 0; layer3_0_conv2_LB_0 < 2; ++layer3_0_conv2_LB_0) {
                layer3_0_conv2_WB[0][layer3_0_conv2_LB_2][layer3_0_conv2_LB_1][layer3_0_conv2_LB_0] = layer3_0_conv2_WB[0][layer3_0_conv2_LB_2][layer3_0_conv2_LB_1][(layer3_0_conv2_LB_0 + 1)];
              }
              layer3_0_conv2_WB[0][layer3_0_conv2_LB_2][layer3_0_conv2_LB_1][2] = layer3_0_conv2_LB[0][layer3_0_conv2_LB_2][layer3_0_conv2_LB_1][xx_reuse12];
            }
          }
            if (2 <= xx_reuse12) {
          layer3_0_conv2_ff14: for (bit32 ff14 = 0; ff14 < 64; ++ff14) {
      #pragma HLS pipeline
              ap_int<8> layer3_0_conv2_sum;
              layer3_0_conv2_sum = (ap_int<8>)0;
              layer3_0_conv2_layer3_0_conv2_rc: for (bit32 layer3_0_conv2_rc = 0; layer3_0_conv2_rc < 2; ++layer3_0_conv2_rc) {
                layer3_0_conv2_layer3_0_conv2_ry: for (bit32 layer3_0_conv2_ry = 0; layer3_0_conv2_ry < 3; ++layer3_0_conv2_ry) {
                  layer3_0_conv2_layer3_0_conv2_rx: for (bit32 layer3_0_conv2_rx = 0; layer3_0_conv2_rx < 3; ++layer3_0_conv2_rx) {
                    layer3_0_conv2_layer3_0_conv2_rb: for (bit32 layer3_0_conv2_rb = 0; layer3_0_conv2_rb < 32; ++layer3_0_conv2_rb) {
                      layer3_0_conv2_sum = ((ap_int<8>)(((ap_int<34>)(layer3_0_conv2_WB[0][layer3_0_conv2_rc][layer3_0_conv2_ry][layer3_0_conv2_rx] ^ w_layer3_0_conv2[ff14][layer3_0_conv2_rc][layer3_0_conv2_ry][layer3_0_conv2_rx])[layer3_0_conv2_rb]) + ((ap_int<34>)layer3_0_conv2_sum)));
                    }
                  }
                }
              }
              ap_int<8> layer3_0_conv2_temp;
              layer3_0_conv2_temp = ((ap_int<8>)(576 - ((bit32)(layer3_0_conv2_sum << 1))));
              layer3_0_conv2_pipe_84.write(layer3_0_conv2_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer3_0_bn2[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_0_bn2_pipe_85;
    #pragma HLS stream variable=layer3_0_bn2_pipe_85 depth=4096
    layer3_0_bn2_args014: for (bit32 args014 = 0; args014 < 64; ++args014) {
      layer3_0_bn2_args114: for (bit32 args114 = 0; args114 < 8; ++args114) {
        layer3_0_bn2_args214: for (bit32 args214 = 0; args214 < 8; ++args214) {
        #pragma HLS pipeline
          ap_int<8> layer3_0_conv2_temp1;
          layer3_0_conv2_temp1 = layer3_0_conv2_pipe_84.read();
          ap_fixed<32, 20> layer3_0_bn2_temp;
          layer3_0_bn2_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer3_0_conv2_temp1) - ((ap_fixed<33, 21>)w_layer3_0_bn2_16[args014]))) / sqrt((((float)w_layer3_0_bn2_17[args014]) + 1.000000e-07f))) * ((float)w_layer3_0_bn2_14[args014])) + ((float)w_layer3_0_bn2_15[args014])));
          layer3_0_bn2_pipe_85.write(layer3_0_bn2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_0_residual2[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_0_residual2_pipe_86;
    #pragma HLS stream variable=layer3_0_residual2_pipe_86 depth=4096
    layer3_0_residual2_cc32: for (bit32 cc32 = 0; cc32 < 64; ++cc32) {
      layer3_0_residual2_ww56: for (bit32 ww56 = 0; ww56 < 8; ++ww56) {
        layer3_0_residual2_hh58: for (bit32 hh58 = 0; hh58 < 8; ++hh58) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_0_bn2_temp1;
          layer3_0_bn2_temp1 = layer3_0_bn2_pipe_85.read();
          ap_fixed<32, 20> layer3_0_residual2_temp;
          ap_fixed<32, 20> layer3_0_rprelu1_temp2;
          layer3_0_rprelu1_temp2 = layer3_0_rprelu1_pipe_132.read();
          layer3_0_residual2_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer3_0_bn2_temp1) + ((ap_fixed<33, 21>)layer3_0_rprelu1_temp2)));
          layer3_0_residual2_pipe_86.write(layer3_0_residual2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_0_rprelu2[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_0_rprelu2_pipe_87;
    #pragma HLS stream variable=layer3_0_rprelu2_pipe_87 depth=4096
    hls::stream<ap_fixed<32, 20> > layer3_0_rprelu2_pipe_133;
    #pragma HLS stream variable=layer3_0_rprelu2_pipe_133 depth=4096
    layer3_0_rprelu2_cc33: for (bit32 cc33 = 0; cc33 < 64; ++cc33) {
      layer3_0_rprelu2_ww57: for (bit32 ww57 = 0; ww57 < 8; ++ww57) {
        layer3_0_rprelu2_hh59: for (bit32 hh59 = 0; hh59 < 8; ++hh59) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_0_residual2_temp1;
          layer3_0_residual2_temp1 = layer3_0_residual2_pipe_86.read();
          ap_fixed<32, 20> layer3_0_rprelu2_temp;
          layer3_0_rprelu2_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_0_residual2_temp1) + ((ap_fixed<33, 21>)w_layer3_0_rprelu2_3[cc33])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer3_0_residual2_temp1) + ((ap_fixed<33, 21>)w_layer3_0_rprelu2_3[cc33])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer3_0_rprelu2_5[cc33]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer3_0_residual2_temp1) + ((ap_fixed<33, 21>)w_layer3_0_rprelu2_3[cc33]))))))) + ((ap_fixed<66, 42>)w_layer3_0_rprelu2_4[cc33])));
          layer3_0_rprelu2_pipe_133.write(layer3_0_rprelu2_temp);
          layer3_0_rprelu2_pipe_87.write(layer3_0_rprelu2_temp);
        }
      }
    }
    ubit32 layer3_1_rsign1[1][2][8][8];
    hls::stream<ubit32 > layer3_1_rsign1_pipe_88;
    #pragma HLS stream variable=layer3_1_rsign1_pipe_88 depth=128
    layer3_1_rsign1_cc34: for (bit32 cc34 = 0; cc34 < 2; ++cc34) {
      layer3_1_rsign1_hh60: for (bit32 hh60 = 0; hh60 < 8; ++hh60) {
        layer3_1_rsign1_ww58: for (bit32 ww58 = 0; ww58 < 8; ++ww58) {
        #pragma HLS pipeline
          ubit32 layer3_1_rsign1_pack;
          layer3_1_rsign1_pack = 0U;
          loop_i15: for (bit32 i15 = 0; i15 < 32; ++i15) {
            ap_fixed<32, 20> layer3_0_rprelu2_temp1;
            layer3_0_rprelu2_temp1 = layer3_0_rprelu2_pipe_87.read();
            layer3_1_rsign1_pack(i15, i15) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_0_rprelu2_temp1) + ((ap_fixed<33, 21>)w_layer3_1_rsign1[((cc34 * 32) + i15)])))) ? ((bit32)1) : ((bit32)0));
          }
          ubit32 layer3_1_rsign1_temp;
          layer3_1_rsign1_temp = layer3_1_rsign1_pack;
          layer3_1_rsign1_pipe_88.write(layer3_1_rsign1_temp);
        }
      }
    }
    ubit32 layer3_1_conv1_pad[1][2][10][10];
    hls::stream<ubit32 > layer3_1_conv1_pad_pipe_89;
    #pragma HLS stream variable=layer3_1_conv1_pad_pipe_89 depth=200
    layer3_1_conv1_pad_cc35: for (bit32 cc35 = 0; cc35 < 2; ++cc35) {
      layer3_1_conv1_pad_hh61: for (bit32 hh61 = 0; hh61 < 10; ++hh61) {
      #pragma HLS pipeline
        layer3_1_conv1_pad_ww59: for (bit32 ww59 = 0; ww59 < 10; ++ww59) {
          ubit32 layer3_1_rsign1_temp1;
          ubit32 layer3_1_conv1_pad_temp;
          if (((((1 <= ww59) && (ww59 < 9)) && (1 <= hh61)) && (hh61 < 9))) { 
            layer3_1_conv1_pad_temp = layer3_1_rsign1_pipe_88.read();
          } else { 
            layer3_1_conv1_pad_temp = 0U;
          }
          layer3_1_conv1_pad_pipe_89.write(layer3_1_conv1_pad_temp);
          layer3_1_conv1_pad[0][cc35][hh61][ww59] = layer3_1_conv1_pad_temp;
        }
      }
    }
    ap_int<8> layer3_1_conv1[1][64][8][8];
    ubit32 layer3_1_conv1_LB[1][2][3][10];
    ubit32 layer3_1_conv1_WB[1][2][3][3];
    // #pragma HLS array_partition variable=layer3_1_conv1_WB complete dim=4
    hls::stream<ap_int<8> > layer3_1_conv1_pipe_90;
    #pragma HLS stream variable=layer3_1_conv1_pipe_90 depth=4096
    layer3_1_conv1_yy_reuse11: for (bit32 yy_reuse11 = 0; yy_reuse11 < 10; ++yy_reuse11) {
      layer3_1_conv1_xx_reuse11: for (bit32 xx_reuse11 = 0; xx_reuse11 < 10; ++xx_reuse11) {
        loop_layer3_1_conv1_pad_2: for (bit32 layer3_1_conv1_pad_2 = 0; layer3_1_conv1_pad_2 < 2; ++layer3_1_conv1_pad_2) {
          loop_layer3_1_conv1_pad_1: for (bit32 layer3_1_conv1_pad_1 = 0; layer3_1_conv1_pad_1 < 2; ++layer3_1_conv1_pad_1) {
            layer3_1_conv1_LB[0][layer3_1_conv1_pad_2][layer3_1_conv1_pad_1][xx_reuse11] = layer3_1_conv1_LB[0][layer3_1_conv1_pad_2][(layer3_1_conv1_pad_1 + 1)][xx_reuse11];
          }
          ubit32 layer3_1_conv1_pad_temp1;
          layer3_1_conv1_pad_temp1 = layer3_1_conv1_pad_pipe_89.read();
          layer3_1_conv1_LB[0][layer3_1_conv1_pad_2][2][xx_reuse11] = layer3_1_conv1_pad_temp1;
        }
        if (2 <= yy_reuse11) {
          loop_layer3_1_conv1_LB_1: for (bit32 layer3_1_conv1_LB_1 = 0; layer3_1_conv1_LB_1 < 3; ++layer3_1_conv1_LB_1) {
            loop_layer3_1_conv1_LB_2: for (bit32 layer3_1_conv1_LB_2 = 0; layer3_1_conv1_LB_2 < 2; ++layer3_1_conv1_LB_2) {
              loop_layer3_1_conv1_LB_0: for (bit32 layer3_1_conv1_LB_0 = 0; layer3_1_conv1_LB_0 < 2; ++layer3_1_conv1_LB_0) {
                layer3_1_conv1_WB[0][layer3_1_conv1_LB_2][layer3_1_conv1_LB_1][layer3_1_conv1_LB_0] = layer3_1_conv1_WB[0][layer3_1_conv1_LB_2][layer3_1_conv1_LB_1][(layer3_1_conv1_LB_0 + 1)];
              }
              layer3_1_conv1_WB[0][layer3_1_conv1_LB_2][layer3_1_conv1_LB_1][2] = layer3_1_conv1_LB[0][layer3_1_conv1_LB_2][layer3_1_conv1_LB_1][xx_reuse11];
            }
          }
            if (2 <= xx_reuse11) {
          layer3_1_conv1_ff15: for (bit32 ff15 = 0; ff15 < 64; ++ff15) {
      #pragma HLS pipeline
              ap_int<8> layer3_1_conv1_sum;
              layer3_1_conv1_sum = (ap_int<8>)0;
              layer3_1_conv1_layer3_1_conv1_rc: for (bit32 layer3_1_conv1_rc = 0; layer3_1_conv1_rc < 2; ++layer3_1_conv1_rc) {
                layer3_1_conv1_layer3_1_conv1_ry: for (bit32 layer3_1_conv1_ry = 0; layer3_1_conv1_ry < 3; ++layer3_1_conv1_ry) {
                  layer3_1_conv1_layer3_1_conv1_rx: for (bit32 layer3_1_conv1_rx = 0; layer3_1_conv1_rx < 3; ++layer3_1_conv1_rx) {
                    layer3_1_conv1_layer3_1_conv1_rb: for (bit32 layer3_1_conv1_rb = 0; layer3_1_conv1_rb < 32; ++layer3_1_conv1_rb) {
                      layer3_1_conv1_sum = ((ap_int<8>)(((ap_int<34>)(layer3_1_conv1_WB[0][layer3_1_conv1_rc][layer3_1_conv1_ry][layer3_1_conv1_rx] ^ w_layer3_1_conv1[ff15][layer3_1_conv1_rc][layer3_1_conv1_ry][layer3_1_conv1_rx])[layer3_1_conv1_rb]) + ((ap_int<34>)layer3_1_conv1_sum)));
                    }
                  }
                }
              }
              ap_int<8> layer3_1_conv1_temp;
              layer3_1_conv1_temp = ((ap_int<8>)(576 - ((bit32)(layer3_1_conv1_sum << 1))));
              layer3_1_conv1_pipe_90.write(layer3_1_conv1_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer3_1_bn1[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_1_bn1_pipe_91;
    #pragma HLS stream variable=layer3_1_bn1_pipe_91 depth=4096
    layer3_1_bn1_args015: for (bit32 args015 = 0; args015 < 64; ++args015) {
      layer3_1_bn1_args115: for (bit32 args115 = 0; args115 < 8; ++args115) {
        layer3_1_bn1_args215: for (bit32 args215 = 0; args215 < 8; ++args215) {
        #pragma HLS pipeline
          ap_int<8> layer3_1_conv1_temp1;
          layer3_1_conv1_temp1 = layer3_1_conv1_pipe_90.read();
          ap_fixed<32, 20> layer3_1_bn1_temp;
          layer3_1_bn1_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer3_1_conv1_temp1) - ((ap_fixed<33, 21>)w_layer3_1_bn1_11[args015]))) / sqrt((((float)w_layer3_1_bn1_12[args015]) + 1.000000e-07f))) * ((float)w_layer3_1_bn1_9[args015])) + ((float)w_layer3_1_bn1_10[args015])));
          layer3_1_bn1_pipe_91.write(layer3_1_bn1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_1_residual1[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_1_residual1_pipe_92;
    #pragma HLS stream variable=layer3_1_residual1_pipe_92 depth=4096
    layer3_1_residual1_cc36: for (bit32 cc36 = 0; cc36 < 64; ++cc36) {
      layer3_1_residual1_ww60: for (bit32 ww60 = 0; ww60 < 8; ++ww60) {
        layer3_1_residual1_hh62: for (bit32 hh62 = 0; hh62 < 8; ++hh62) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_1_bn1_temp1;
          layer3_1_bn1_temp1 = layer3_1_bn1_pipe_91.read();
          ap_fixed<32, 20> layer3_1_residual1_temp;
          ap_fixed<32, 20> layer3_0_rprelu2_temp2;
          layer3_0_rprelu2_temp2 = layer3_0_rprelu2_pipe_133.read();
          layer3_1_residual1_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer3_1_bn1_temp1) + ((ap_fixed<33, 21>)layer3_0_rprelu2_temp2)));
          layer3_1_residual1_pipe_92.write(layer3_1_residual1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_1_rprelu1[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_1_rprelu1_pipe_93;
    #pragma HLS stream variable=layer3_1_rprelu1_pipe_93 depth=4096
    hls::stream<ap_fixed<32, 20> > layer3_1_rprelu1_pipe_134;
    #pragma HLS stream variable=layer3_1_rprelu1_pipe_134 depth=4096
    layer3_1_rprelu1_cc37: for (bit32 cc37 = 0; cc37 < 64; ++cc37) {
      layer3_1_rprelu1_ww61: for (bit32 ww61 = 0; ww61 < 8; ++ww61) {
        layer3_1_rprelu1_hh63: for (bit32 hh63 = 0; hh63 < 8; ++hh63) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_1_residual1_temp1;
          layer3_1_residual1_temp1 = layer3_1_residual1_pipe_92.read();
          ap_fixed<32, 20> layer3_1_rprelu1_temp;
          layer3_1_rprelu1_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_1_residual1_temp1) + ((ap_fixed<33, 21>)w_layer3_1_rprelu1_0[cc37])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer3_1_residual1_temp1) + ((ap_fixed<33, 21>)w_layer3_1_rprelu1_0[cc37])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer3_1_rprelu1_2[cc37]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer3_1_residual1_temp1) + ((ap_fixed<33, 21>)w_layer3_1_rprelu1_0[cc37]))))))) + ((ap_fixed<66, 42>)w_layer3_1_rprelu1_1[cc37])));
          layer3_1_rprelu1_pipe_134.write(layer3_1_rprelu1_temp);
          layer3_1_rprelu1_pipe_93.write(layer3_1_rprelu1_temp);
        }
      }
    }
    ubit32 layer3_1_rsign2[1][2][8][8];
    hls::stream<ubit32 > layer3_1_rsign2_pipe_94;
    #pragma HLS stream variable=layer3_1_rsign2_pipe_94 depth=128
    layer3_1_rsign2_cc38: for (bit32 cc38 = 0; cc38 < 2; ++cc38) {
      layer3_1_rsign2_hh64: for (bit32 hh64 = 0; hh64 < 8; ++hh64) {
        layer3_1_rsign2_ww62: for (bit32 ww62 = 0; ww62 < 8; ++ww62) {
        #pragma HLS pipeline
          ubit32 layer3_1_rsign2_pack;
          layer3_1_rsign2_pack = 0U;
          loop_i16: for (bit32 i16 = 0; i16 < 32; ++i16) {
            ap_fixed<32, 20> layer3_1_rprelu1_temp1;
            layer3_1_rprelu1_temp1 = layer3_1_rprelu1_pipe_93.read();
            layer3_1_rsign2_pack(i16, i16) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_1_rprelu1_temp1) + ((ap_fixed<33, 21>)w_layer3_1_rsign2[((cc38 * 32) + i16)])))) ? ((bit32)1) : ((bit32)0));
          }
          ubit32 layer3_1_rsign2_temp;
          layer3_1_rsign2_temp = layer3_1_rsign2_pack;
          layer3_1_rsign2_pipe_94.write(layer3_1_rsign2_temp);
        }
      }
    }
    ubit32 layer3_1_conv2_pad[1][2][10][10];
    hls::stream<ubit32 > layer3_1_conv2_pad_pipe_95;
    #pragma HLS stream variable=layer3_1_conv2_pad_pipe_95 depth=200
    layer3_1_conv2_pad_cc39: for (bit32 cc39 = 0; cc39 < 2; ++cc39) {
      layer3_1_conv2_pad_hh65: for (bit32 hh65 = 0; hh65 < 10; ++hh65) {
      #pragma HLS pipeline
        layer3_1_conv2_pad_ww63: for (bit32 ww63 = 0; ww63 < 10; ++ww63) {
          ubit32 layer3_1_rsign2_temp1;
          ubit32 layer3_1_conv2_pad_temp;
          if (((((1 <= ww63) && (ww63 < 9)) && (1 <= hh65)) && (hh65 < 9))) { 
            layer3_1_conv2_pad_temp = layer3_1_rsign2_pipe_94.read();
          } else { 
            layer3_1_conv2_pad_temp = 0U;
          }
          layer3_1_conv2_pad_pipe_95.write(layer3_1_conv2_pad_temp);
          layer3_1_conv2_pad[0][cc39][hh65][ww63] = layer3_1_conv2_pad_temp;
        }
      }
    }
    ap_int<8> layer3_1_conv2[1][64][8][8];
    ubit32 layer3_1_conv2_LB[1][2][3][10];
    ubit32 layer3_1_conv2_WB[1][2][3][3];
    // #pragma HLS array_partition variable=layer3_1_conv2_WB complete dim=4
    hls::stream<ap_int<8> > layer3_1_conv2_pipe_96;
    #pragma HLS stream variable=layer3_1_conv2_pipe_96 depth=4096
    layer3_1_conv2_yy_reuse12: for (bit32 yy_reuse12 = 0; yy_reuse12 < 10; ++yy_reuse12) {
      layer3_1_conv2_xx_reuse12: for (bit32 xx_reuse12 = 0; xx_reuse12 < 10; ++xx_reuse12) {
        loop_layer3_1_conv2_pad_2: for (bit32 layer3_1_conv2_pad_2 = 0; layer3_1_conv2_pad_2 < 2; ++layer3_1_conv2_pad_2) {
          loop_layer3_1_conv2_pad_1: for (bit32 layer3_1_conv2_pad_1 = 0; layer3_1_conv2_pad_1 < 2; ++layer3_1_conv2_pad_1) {
            layer3_1_conv2_LB[0][layer3_1_conv2_pad_2][layer3_1_conv2_pad_1][xx_reuse12] = layer3_1_conv2_LB[0][layer3_1_conv2_pad_2][(layer3_1_conv2_pad_1 + 1)][xx_reuse12];
          }
          ubit32 layer3_1_conv2_pad_temp1;
          layer3_1_conv2_pad_temp1 = layer3_1_conv2_pad_pipe_95.read();
          layer3_1_conv2_LB[0][layer3_1_conv2_pad_2][2][xx_reuse12] = layer3_1_conv2_pad_temp1;
        }
        if (2 <= yy_reuse12) {
          loop_layer3_1_conv2_LB_1: for (bit32 layer3_1_conv2_LB_1 = 0; layer3_1_conv2_LB_1 < 3; ++layer3_1_conv2_LB_1) {
            loop_layer3_1_conv2_LB_2: for (bit32 layer3_1_conv2_LB_2 = 0; layer3_1_conv2_LB_2 < 2; ++layer3_1_conv2_LB_2) {
              loop_layer3_1_conv2_LB_0: for (bit32 layer3_1_conv2_LB_0 = 0; layer3_1_conv2_LB_0 < 2; ++layer3_1_conv2_LB_0) {
                layer3_1_conv2_WB[0][layer3_1_conv2_LB_2][layer3_1_conv2_LB_1][layer3_1_conv2_LB_0] = layer3_1_conv2_WB[0][layer3_1_conv2_LB_2][layer3_1_conv2_LB_1][(layer3_1_conv2_LB_0 + 1)];
              }
              layer3_1_conv2_WB[0][layer3_1_conv2_LB_2][layer3_1_conv2_LB_1][2] = layer3_1_conv2_LB[0][layer3_1_conv2_LB_2][layer3_1_conv2_LB_1][xx_reuse12];
            }
          }
            if (2 <= xx_reuse12) {
          layer3_1_conv2_ff16: for (bit32 ff16 = 0; ff16 < 64; ++ff16) {
      #pragma HLS pipeline
              ap_int<8> layer3_1_conv2_sum;
              layer3_1_conv2_sum = (ap_int<8>)0;
              layer3_1_conv2_layer3_1_conv2_rc: for (bit32 layer3_1_conv2_rc = 0; layer3_1_conv2_rc < 2; ++layer3_1_conv2_rc) {
                layer3_1_conv2_layer3_1_conv2_ry: for (bit32 layer3_1_conv2_ry = 0; layer3_1_conv2_ry < 3; ++layer3_1_conv2_ry) {
                  layer3_1_conv2_layer3_1_conv2_rx: for (bit32 layer3_1_conv2_rx = 0; layer3_1_conv2_rx < 3; ++layer3_1_conv2_rx) {
                    layer3_1_conv2_layer3_1_conv2_rb: for (bit32 layer3_1_conv2_rb = 0; layer3_1_conv2_rb < 32; ++layer3_1_conv2_rb) {
                      layer3_1_conv2_sum = ((ap_int<8>)(((ap_int<34>)(layer3_1_conv2_WB[0][layer3_1_conv2_rc][layer3_1_conv2_ry][layer3_1_conv2_rx] ^ w_layer3_1_conv2[ff16][layer3_1_conv2_rc][layer3_1_conv2_ry][layer3_1_conv2_rx])[layer3_1_conv2_rb]) + ((ap_int<34>)layer3_1_conv2_sum)));
                    }
                  }
                }
              }
              ap_int<8> layer3_1_conv2_temp;
              layer3_1_conv2_temp = ((ap_int<8>)(576 - ((bit32)(layer3_1_conv2_sum << 1))));
              layer3_1_conv2_pipe_96.write(layer3_1_conv2_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer3_1_bn2[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_1_bn2_pipe_97;
    #pragma HLS stream variable=layer3_1_bn2_pipe_97 depth=4096
    layer3_1_bn2_args016: for (bit32 args016 = 0; args016 < 64; ++args016) {
      layer3_1_bn2_args116: for (bit32 args116 = 0; args116 < 8; ++args116) {
        layer3_1_bn2_args216: for (bit32 args216 = 0; args216 < 8; ++args216) {
        #pragma HLS pipeline
          ap_int<8> layer3_1_conv2_temp1;
          layer3_1_conv2_temp1 = layer3_1_conv2_pipe_96.read();
          ap_fixed<32, 20> layer3_1_bn2_temp;
          layer3_1_bn2_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer3_1_conv2_temp1) - ((ap_fixed<33, 21>)w_layer3_1_bn2_16[args016]))) / sqrt((((float)w_layer3_1_bn2_17[args016]) + 1.000000e-07f))) * ((float)w_layer3_1_bn2_14[args016])) + ((float)w_layer3_1_bn2_15[args016])));
          layer3_1_bn2_pipe_97.write(layer3_1_bn2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_1_residual2[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_1_residual2_pipe_98;
    #pragma HLS stream variable=layer3_1_residual2_pipe_98 depth=4096
    layer3_1_residual2_cc40: for (bit32 cc40 = 0; cc40 < 64; ++cc40) {
      layer3_1_residual2_ww64: for (bit32 ww64 = 0; ww64 < 8; ++ww64) {
        layer3_1_residual2_hh66: for (bit32 hh66 = 0; hh66 < 8; ++hh66) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_1_bn2_temp1;
          layer3_1_bn2_temp1 = layer3_1_bn2_pipe_97.read();
          ap_fixed<32, 20> layer3_1_residual2_temp;
          ap_fixed<32, 20> layer3_1_rprelu1_temp2;
          layer3_1_rprelu1_temp2 = layer3_1_rprelu1_pipe_134.read();
          layer3_1_residual2_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer3_1_bn2_temp1) + ((ap_fixed<33, 21>)layer3_1_rprelu1_temp2)));
          layer3_1_residual2_pipe_98.write(layer3_1_residual2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_1_rprelu2[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_1_rprelu2_pipe_99;
    #pragma HLS stream variable=layer3_1_rprelu2_pipe_99 depth=4096
    hls::stream<ap_fixed<32, 20> > layer3_1_rprelu2_pipe_135;
    #pragma HLS stream variable=layer3_1_rprelu2_pipe_135 depth=4096
    layer3_1_rprelu2_cc41: for (bit32 cc41 = 0; cc41 < 64; ++cc41) {
      layer3_1_rprelu2_ww65: for (bit32 ww65 = 0; ww65 < 8; ++ww65) {
        layer3_1_rprelu2_hh67: for (bit32 hh67 = 0; hh67 < 8; ++hh67) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_1_residual2_temp1;
          layer3_1_residual2_temp1 = layer3_1_residual2_pipe_98.read();
          ap_fixed<32, 20> layer3_1_rprelu2_temp;
          layer3_1_rprelu2_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_1_residual2_temp1) + ((ap_fixed<33, 21>)w_layer3_1_rprelu2_3[cc41])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer3_1_residual2_temp1) + ((ap_fixed<33, 21>)w_layer3_1_rprelu2_3[cc41])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer3_1_rprelu2_5[cc41]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer3_1_residual2_temp1) + ((ap_fixed<33, 21>)w_layer3_1_rprelu2_3[cc41]))))))) + ((ap_fixed<66, 42>)w_layer3_1_rprelu2_4[cc41])));
          layer3_1_rprelu2_pipe_135.write(layer3_1_rprelu2_temp);
          layer3_1_rprelu2_pipe_99.write(layer3_1_rprelu2_temp);
        }
      }
    }
    ubit32 layer3_2_rsign1[1][2][8][8];
    hls::stream<ubit32 > layer3_2_rsign1_pipe_100;
    #pragma HLS stream variable=layer3_2_rsign1_pipe_100 depth=128
    layer3_2_rsign1_cc42: for (bit32 cc42 = 0; cc42 < 2; ++cc42) {
      layer3_2_rsign1_hh68: for (bit32 hh68 = 0; hh68 < 8; ++hh68) {
        layer3_2_rsign1_ww66: for (bit32 ww66 = 0; ww66 < 8; ++ww66) {
        #pragma HLS pipeline
          ubit32 layer3_2_rsign1_pack;
          layer3_2_rsign1_pack = 0U;
          loop_i17: for (bit32 i17 = 0; i17 < 32; ++i17) {
            ap_fixed<32, 20> layer3_1_rprelu2_temp1;
            layer3_1_rprelu2_temp1 = layer3_1_rprelu2_pipe_99.read();
            layer3_2_rsign1_pack(i17, i17) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_1_rprelu2_temp1) + ((ap_fixed<33, 21>)w_layer3_2_rsign1[((cc42 * 32) + i17)])))) ? ((bit32)1) : ((bit32)0));
          }
          ubit32 layer3_2_rsign1_temp;
          layer3_2_rsign1_temp = layer3_2_rsign1_pack;
          layer3_2_rsign1_pipe_100.write(layer3_2_rsign1_temp);
        }
      }
    }
    ubit32 layer3_2_conv1_pad[1][2][10][10];
    hls::stream<ubit32 > layer3_2_conv1_pad_pipe_101;
    #pragma HLS stream variable=layer3_2_conv1_pad_pipe_101 depth=200
    layer3_2_conv1_pad_cc43: for (bit32 cc43 = 0; cc43 < 2; ++cc43) {
      layer3_2_conv1_pad_hh69: for (bit32 hh69 = 0; hh69 < 10; ++hh69) {
      #pragma HLS pipeline
        layer3_2_conv1_pad_ww67: for (bit32 ww67 = 0; ww67 < 10; ++ww67) {
          ubit32 layer3_2_rsign1_temp1;
          ubit32 layer3_2_conv1_pad_temp;
          if (((((1 <= ww67) && (ww67 < 9)) && (1 <= hh69)) && (hh69 < 9))) { 
            layer3_2_conv1_pad_temp = layer3_2_rsign1_pipe_100.read();
          } else { 
            layer3_2_conv1_pad_temp = 0U;
          }
          layer3_2_conv1_pad_pipe_101.write(layer3_2_conv1_pad_temp);
          layer3_2_conv1_pad[0][cc43][hh69][ww67] = layer3_2_conv1_pad_temp;
        }
      }
    }
    ap_int<8> layer3_2_conv1[1][64][8][8];
    ubit32 layer3_2_conv1_LB[1][2][3][10];
    ubit32 layer3_2_conv1_WB[1][2][3][3];
    // #pragma HLS array_partition variable=layer3_2_conv1_WB complete dim=4
    hls::stream<ap_int<8> > layer3_2_conv1_pipe_102;
    #pragma HLS stream variable=layer3_2_conv1_pipe_102 depth=4096
    layer3_2_conv1_yy_reuse13: for (bit32 yy_reuse13 = 0; yy_reuse13 < 10; ++yy_reuse13) {
      layer3_2_conv1_xx_reuse13: for (bit32 xx_reuse13 = 0; xx_reuse13 < 10; ++xx_reuse13) {
        loop_layer3_2_conv1_pad_2: for (bit32 layer3_2_conv1_pad_2 = 0; layer3_2_conv1_pad_2 < 2; ++layer3_2_conv1_pad_2) {
          loop_layer3_2_conv1_pad_1: for (bit32 layer3_2_conv1_pad_1 = 0; layer3_2_conv1_pad_1 < 2; ++layer3_2_conv1_pad_1) {
            layer3_2_conv1_LB[0][layer3_2_conv1_pad_2][layer3_2_conv1_pad_1][xx_reuse13] = layer3_2_conv1_LB[0][layer3_2_conv1_pad_2][(layer3_2_conv1_pad_1 + 1)][xx_reuse13];
          }
          ubit32 layer3_2_conv1_pad_temp1;
          layer3_2_conv1_pad_temp1 = layer3_2_conv1_pad_pipe_101.read();
          layer3_2_conv1_LB[0][layer3_2_conv1_pad_2][2][xx_reuse13] = layer3_2_conv1_pad_temp1;
        }
        if (2 <= yy_reuse13) {
          loop_layer3_2_conv1_LB_1: for (bit32 layer3_2_conv1_LB_1 = 0; layer3_2_conv1_LB_1 < 3; ++layer3_2_conv1_LB_1) {
            loop_layer3_2_conv1_LB_2: for (bit32 layer3_2_conv1_LB_2 = 0; layer3_2_conv1_LB_2 < 2; ++layer3_2_conv1_LB_2) {
              loop_layer3_2_conv1_LB_0: for (bit32 layer3_2_conv1_LB_0 = 0; layer3_2_conv1_LB_0 < 2; ++layer3_2_conv1_LB_0) {
                layer3_2_conv1_WB[0][layer3_2_conv1_LB_2][layer3_2_conv1_LB_1][layer3_2_conv1_LB_0] = layer3_2_conv1_WB[0][layer3_2_conv1_LB_2][layer3_2_conv1_LB_1][(layer3_2_conv1_LB_0 + 1)];
              }
              layer3_2_conv1_WB[0][layer3_2_conv1_LB_2][layer3_2_conv1_LB_1][2] = layer3_2_conv1_LB[0][layer3_2_conv1_LB_2][layer3_2_conv1_LB_1][xx_reuse13];
            }
          }
            if (2 <= xx_reuse13) {
          layer3_2_conv1_ff17: for (bit32 ff17 = 0; ff17 < 64; ++ff17) {
      #pragma HLS pipeline
              ap_int<8> layer3_2_conv1_sum;
              layer3_2_conv1_sum = (ap_int<8>)0;
              layer3_2_conv1_layer3_2_conv1_rc: for (bit32 layer3_2_conv1_rc = 0; layer3_2_conv1_rc < 2; ++layer3_2_conv1_rc) {
                layer3_2_conv1_layer3_2_conv1_ry: for (bit32 layer3_2_conv1_ry = 0; layer3_2_conv1_ry < 3; ++layer3_2_conv1_ry) {
                  layer3_2_conv1_layer3_2_conv1_rx: for (bit32 layer3_2_conv1_rx = 0; layer3_2_conv1_rx < 3; ++layer3_2_conv1_rx) {
                    layer3_2_conv1_layer3_2_conv1_rb: for (bit32 layer3_2_conv1_rb = 0; layer3_2_conv1_rb < 32; ++layer3_2_conv1_rb) {
                      layer3_2_conv1_sum = ((ap_int<8>)(((ap_int<34>)(layer3_2_conv1_WB[0][layer3_2_conv1_rc][layer3_2_conv1_ry][layer3_2_conv1_rx] ^ w_layer3_2_conv1[ff17][layer3_2_conv1_rc][layer3_2_conv1_ry][layer3_2_conv1_rx])[layer3_2_conv1_rb]) + ((ap_int<34>)layer3_2_conv1_sum)));
                    }
                  }
                }
              }
              ap_int<8> layer3_2_conv1_temp;
              layer3_2_conv1_temp = ((ap_int<8>)(576 - ((bit32)(layer3_2_conv1_sum << 1))));
              layer3_2_conv1_pipe_102.write(layer3_2_conv1_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer3_2_bn1[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_2_bn1_pipe_103;
    #pragma HLS stream variable=layer3_2_bn1_pipe_103 depth=4096
    layer3_2_bn1_args017: for (bit32 args017 = 0; args017 < 64; ++args017) {
      layer3_2_bn1_args117: for (bit32 args117 = 0; args117 < 8; ++args117) {
        layer3_2_bn1_args217: for (bit32 args217 = 0; args217 < 8; ++args217) {
        #pragma HLS pipeline
          ap_int<8> layer3_2_conv1_temp1;
          layer3_2_conv1_temp1 = layer3_2_conv1_pipe_102.read();
          ap_fixed<32, 20> layer3_2_bn1_temp;
          layer3_2_bn1_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer3_2_conv1_temp1) - ((ap_fixed<33, 21>)w_layer3_2_bn1_11[args017]))) / sqrt((((float)w_layer3_2_bn1_12[args017]) + 1.000000e-07f))) * ((float)w_layer3_2_bn1_9[args017])) + ((float)w_layer3_2_bn1_10[args017])));
          layer3_2_bn1_pipe_103.write(layer3_2_bn1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_2_residual1[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_2_residual1_pipe_104;
    #pragma HLS stream variable=layer3_2_residual1_pipe_104 depth=4096
    layer3_2_residual1_cc44: for (bit32 cc44 = 0; cc44 < 64; ++cc44) {
      layer3_2_residual1_ww68: for (bit32 ww68 = 0; ww68 < 8; ++ww68) {
        layer3_2_residual1_hh70: for (bit32 hh70 = 0; hh70 < 8; ++hh70) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_2_bn1_temp1;
          layer3_2_bn1_temp1 = layer3_2_bn1_pipe_103.read();
          ap_fixed<32, 20> layer3_2_residual1_temp;
          ap_fixed<32, 20> layer3_1_rprelu2_temp2;
          layer3_1_rprelu2_temp2 = layer3_1_rprelu2_pipe_135.read();
          layer3_2_residual1_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer3_2_bn1_temp1) + ((ap_fixed<33, 21>)layer3_1_rprelu2_temp2)));
          layer3_2_residual1_pipe_104.write(layer3_2_residual1_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_2_rprelu1[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_2_rprelu1_pipe_105;
    #pragma HLS stream variable=layer3_2_rprelu1_pipe_105 depth=4096
    hls::stream<ap_fixed<32, 20> > layer3_2_rprelu1_pipe_136;
    #pragma HLS stream variable=layer3_2_rprelu1_pipe_136 depth=4096
    layer3_2_rprelu1_cc45: for (bit32 cc45 = 0; cc45 < 64; ++cc45) {
      layer3_2_rprelu1_ww69: for (bit32 ww69 = 0; ww69 < 8; ++ww69) {
        layer3_2_rprelu1_hh71: for (bit32 hh71 = 0; hh71 < 8; ++hh71) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_2_residual1_temp1;
          layer3_2_residual1_temp1 = layer3_2_residual1_pipe_104.read();
          ap_fixed<32, 20> layer3_2_rprelu1_temp;
          layer3_2_rprelu1_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_2_residual1_temp1) + ((ap_fixed<33, 21>)w_layer3_2_rprelu1_0[cc45])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer3_2_residual1_temp1) + ((ap_fixed<33, 21>)w_layer3_2_rprelu1_0[cc45])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer3_2_rprelu1_2[cc45]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer3_2_residual1_temp1) + ((ap_fixed<33, 21>)w_layer3_2_rprelu1_0[cc45]))))))) + ((ap_fixed<66, 42>)w_layer3_2_rprelu1_1[cc45])));
          layer3_2_rprelu1_pipe_136.write(layer3_2_rprelu1_temp);
          layer3_2_rprelu1_pipe_105.write(layer3_2_rprelu1_temp);
        }
      }
    }
    ubit32 layer3_2_rsign2[1][2][8][8];
    hls::stream<ubit32 > layer3_2_rsign2_pipe_106;
    #pragma HLS stream variable=layer3_2_rsign2_pipe_106 depth=128
    layer3_2_rsign2_cc46: for (bit32 cc46 = 0; cc46 < 2; ++cc46) {
      layer3_2_rsign2_hh72: for (bit32 hh72 = 0; hh72 < 8; ++hh72) {
        layer3_2_rsign2_ww70: for (bit32 ww70 = 0; ww70 < 8; ++ww70) {
        #pragma HLS pipeline
          ubit32 layer3_2_rsign2_pack;
          layer3_2_rsign2_pack = 0U;
          loop_i18: for (bit32 i18 = 0; i18 < 32; ++i18) {
            ap_fixed<32, 20> layer3_2_rprelu1_temp1;
            layer3_2_rprelu1_temp1 = layer3_2_rprelu1_pipe_105.read();
            layer3_2_rsign2_pack(i18, i18) = (((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_2_rprelu1_temp1) + ((ap_fixed<33, 21>)w_layer3_2_rsign2[((cc46 * 32) + i18)])))) ? ((bit32)1) : ((bit32)0));
          }
          ubit32 layer3_2_rsign2_temp;
          layer3_2_rsign2_temp = layer3_2_rsign2_pack;
          layer3_2_rsign2_pipe_106.write(layer3_2_rsign2_temp);
        }
      }
    }
    ubit32 layer3_2_conv2_pad[1][2][10][10];
    hls::stream<ubit32 > layer3_2_conv2_pad_pipe_107;
    #pragma HLS stream variable=layer3_2_conv2_pad_pipe_107 depth=200
    layer3_2_conv2_pad_cc47: for (bit32 cc47 = 0; cc47 < 2; ++cc47) {
      layer3_2_conv2_pad_hh73: for (bit32 hh73 = 0; hh73 < 10; ++hh73) {
      #pragma HLS pipeline
        layer3_2_conv2_pad_ww71: for (bit32 ww71 = 0; ww71 < 10; ++ww71) {
          ubit32 layer3_2_rsign2_temp1;
          ubit32 layer3_2_conv2_pad_temp;
          if (((((1 <= ww71) && (ww71 < 9)) && (1 <= hh73)) && (hh73 < 9))) { 
            layer3_2_conv2_pad_temp = layer3_2_rsign2_pipe_106.read();
          } else { 
            layer3_2_conv2_pad_temp = 0U;
          }
          layer3_2_conv2_pad_pipe_107.write(layer3_2_conv2_pad_temp);
          layer3_2_conv2_pad[0][cc47][hh73][ww71] = layer3_2_conv2_pad_temp;
        }
      }
    }
    ap_int<8> layer3_2_conv2[1][64][8][8];
    ubit32 layer3_2_conv2_LB[1][2][3][10];
    ubit32 layer3_2_conv2_WB[1][2][3][3];
    // #pragma HLS array_partition variable=layer3_2_conv2_WB complete dim=4
    hls::stream<ap_int<8> > layer3_2_conv2_pipe_108;
    #pragma HLS stream variable=layer3_2_conv2_pipe_108 depth=4096
    layer3_2_conv2_yy_reuse14: for (bit32 yy_reuse14 = 0; yy_reuse14 < 10; ++yy_reuse14) {
      layer3_2_conv2_xx_reuse14: for (bit32 xx_reuse14 = 0; xx_reuse14 < 10; ++xx_reuse14) {
        loop_layer3_2_conv2_pad_2: for (bit32 layer3_2_conv2_pad_2 = 0; layer3_2_conv2_pad_2 < 2; ++layer3_2_conv2_pad_2) {
          loop_layer3_2_conv2_pad_1: for (bit32 layer3_2_conv2_pad_1 = 0; layer3_2_conv2_pad_1 < 2; ++layer3_2_conv2_pad_1) {
            layer3_2_conv2_LB[0][layer3_2_conv2_pad_2][layer3_2_conv2_pad_1][xx_reuse14] = layer3_2_conv2_LB[0][layer3_2_conv2_pad_2][(layer3_2_conv2_pad_1 + 1)][xx_reuse14];
          }
          ubit32 layer3_2_conv2_pad_temp1;
          layer3_2_conv2_pad_temp1 = layer3_2_conv2_pad_pipe_107.read();
          layer3_2_conv2_LB[0][layer3_2_conv2_pad_2][2][xx_reuse14] = layer3_2_conv2_pad_temp1;
        }
        if (2 <= yy_reuse14) {
          loop_layer3_2_conv2_LB_1: for (bit32 layer3_2_conv2_LB_1 = 0; layer3_2_conv2_LB_1 < 3; ++layer3_2_conv2_LB_1) {
            loop_layer3_2_conv2_LB_2: for (bit32 layer3_2_conv2_LB_2 = 0; layer3_2_conv2_LB_2 < 2; ++layer3_2_conv2_LB_2) {
              loop_layer3_2_conv2_LB_0: for (bit32 layer3_2_conv2_LB_0 = 0; layer3_2_conv2_LB_0 < 2; ++layer3_2_conv2_LB_0) {
                layer3_2_conv2_WB[0][layer3_2_conv2_LB_2][layer3_2_conv2_LB_1][layer3_2_conv2_LB_0] = layer3_2_conv2_WB[0][layer3_2_conv2_LB_2][layer3_2_conv2_LB_1][(layer3_2_conv2_LB_0 + 1)];
              }
              layer3_2_conv2_WB[0][layer3_2_conv2_LB_2][layer3_2_conv2_LB_1][2] = layer3_2_conv2_LB[0][layer3_2_conv2_LB_2][layer3_2_conv2_LB_1][xx_reuse14];
            }
          }
            if (2 <= xx_reuse14) {
          layer3_2_conv2_ff18: for (bit32 ff18 = 0; ff18 < 64; ++ff18) {
      #pragma HLS pipeline
              ap_int<8> layer3_2_conv2_sum;
              layer3_2_conv2_sum = (ap_int<8>)0;
              layer3_2_conv2_layer3_2_conv2_rc: for (bit32 layer3_2_conv2_rc = 0; layer3_2_conv2_rc < 2; ++layer3_2_conv2_rc) {
                layer3_2_conv2_layer3_2_conv2_ry: for (bit32 layer3_2_conv2_ry = 0; layer3_2_conv2_ry < 3; ++layer3_2_conv2_ry) {
                  layer3_2_conv2_layer3_2_conv2_rx: for (bit32 layer3_2_conv2_rx = 0; layer3_2_conv2_rx < 3; ++layer3_2_conv2_rx) {
                    layer3_2_conv2_layer3_2_conv2_rb: for (bit32 layer3_2_conv2_rb = 0; layer3_2_conv2_rb < 32; ++layer3_2_conv2_rb) {
                      layer3_2_conv2_sum = ((ap_int<8>)(((ap_int<34>)(layer3_2_conv2_WB[0][layer3_2_conv2_rc][layer3_2_conv2_ry][layer3_2_conv2_rx] ^ w_layer3_2_conv2[ff18][layer3_2_conv2_rc][layer3_2_conv2_ry][layer3_2_conv2_rx])[layer3_2_conv2_rb]) + ((ap_int<34>)layer3_2_conv2_sum)));
                    }
                  }
                }
              }
              ap_int<8> layer3_2_conv2_temp;
              layer3_2_conv2_temp = ((ap_int<8>)(576 - ((bit32)(layer3_2_conv2_sum << 1))));
              layer3_2_conv2_pipe_108.write(layer3_2_conv2_temp);
            }
          }
        }
      }
    }
    ap_fixed<32, 20> layer3_2_bn2[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_2_bn2_pipe_109;
    #pragma HLS stream variable=layer3_2_bn2_pipe_109 depth=4096
    layer3_2_bn2_args018: for (bit32 args018 = 0; args018 < 64; ++args018) {
      layer3_2_bn2_args118: for (bit32 args118 = 0; args118 < 8; ++args118) {
        layer3_2_bn2_args218: for (bit32 args218 = 0; args218 < 8; ++args218) {
        #pragma HLS pipeline
          ap_int<8> layer3_2_conv2_temp1;
          layer3_2_conv2_temp1 = layer3_2_conv2_pipe_108.read();
          ap_fixed<32, 20> layer3_2_bn2_temp;
          layer3_2_bn2_temp = ((ap_fixed<32, 20>)(((((float)(((ap_fixed<33, 21>)layer3_2_conv2_temp1) - ((ap_fixed<33, 21>)w_layer3_2_bn2_16[args018]))) / sqrt((((float)w_layer3_2_bn2_17[args018]) + 1.000000e-07f))) * ((float)w_layer3_2_bn2_14[args018])) + ((float)w_layer3_2_bn2_15[args018])));
          layer3_2_bn2_pipe_109.write(layer3_2_bn2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_2_residual2[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_2_residual2_pipe_110;
    #pragma HLS stream variable=layer3_2_residual2_pipe_110 depth=4096
    layer3_2_residual2_cc48: for (bit32 cc48 = 0; cc48 < 64; ++cc48) {
      layer3_2_residual2_ww72: for (bit32 ww72 = 0; ww72 < 8; ++ww72) {
        layer3_2_residual2_hh74: for (bit32 hh74 = 0; hh74 < 8; ++hh74) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_2_bn2_temp1;
          layer3_2_bn2_temp1 = layer3_2_bn2_pipe_109.read();
          ap_fixed<32, 20> layer3_2_residual2_temp;
          ap_fixed<32, 20> layer3_2_rprelu1_temp2;
          layer3_2_rprelu1_temp2 = layer3_2_rprelu1_pipe_136.read();
          layer3_2_residual2_temp = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)layer3_2_bn2_temp1) + ((ap_fixed<33, 21>)layer3_2_rprelu1_temp2)));
          layer3_2_residual2_pipe_110.write(layer3_2_residual2_temp);
        }
      }
    }
    ap_fixed<32, 20> layer3_2_rprelu2[1][64][8][8];
    hls::stream<ap_fixed<32, 20> > layer3_2_rprelu2_pipe_111;
    #pragma HLS stream variable=layer3_2_rprelu2_pipe_111 depth=4096
    layer3_2_rprelu2_cc49: for (bit32 cc49 = 0; cc49 < 64; ++cc49) {
      layer3_2_rprelu2_ww73: for (bit32 ww73 = 0; ww73 < 8; ++ww73) {
        layer3_2_rprelu2_hh75: for (bit32 hh75 = 0; hh75 < 8; ++hh75) {
        #pragma HLS pipeline
          ap_fixed<32, 20> layer3_2_residual2_temp1;
          layer3_2_residual2_temp1 = layer3_2_residual2_pipe_110.read();
          ap_fixed<32, 20> layer3_2_rprelu2_temp;
          layer3_2_rprelu2_temp = ((ap_fixed<32, 20>)(((ap_fixed<66, 42>)(((ap_fixed<44, 32>)0 < ((ap_fixed<44, 32>)(((ap_fixed<33, 21>)layer3_2_residual2_temp1) + ((ap_fixed<33, 21>)w_layer3_2_rprelu2_3[cc49])))) ? (((ap_fixed<65, 41>)(((ap_fixed<33, 21>)layer3_2_residual2_temp1) + ((ap_fixed<33, 21>)w_layer3_2_rprelu2_3[cc49])))) : ((ap_fixed<65, 41>)(((ap_fixed<65, 53>)w_layer3_2_rprelu2_5[cc49]) * ((ap_fixed<65, 53>)(((ap_fixed<33, 21>)layer3_2_residual2_temp1) + ((ap_fixed<33, 21>)w_layer3_2_rprelu2_3[cc49]))))))) + ((ap_fixed<66, 42>)w_layer3_2_rprelu2_4[cc49])));
          layer3_2_rprelu2_pipe_111.write(layer3_2_rprelu2_temp);
        }
      }
    }
    ap_fixed<32, 20> avgpool_res[1][64][1][1];
    ap_fixed<32, 20> avgpool_LB[8][8];
    bit32 avgpool;
    hls::stream<ap_fixed<32, 20> > avgpool_res_pipe_112;
    #pragma HLS stream variable=avgpool_res_pipe_112 depth=64
    avgpool_cc50: for (bit32 cc50 = 0; cc50 < 64; ++cc50) {
    #pragma HLS pipeline
      loop_avgpool_LB_i: for (bit32 avgpool_LB_i = 0; avgpool_LB_i < 8; ++avgpool_LB_i) {
        loop_avgpool_LB_j: for (bit32 avgpool_LB_j = 0; avgpool_LB_j < 8; ++avgpool_LB_j) {
          ap_fixed<32, 20> layer3_2_rprelu2_temp1;
          layer3_2_rprelu2_temp1 = layer3_2_rprelu2_pipe_111.read();
          avgpool_LB[avgpool_LB_i][avgpool_LB_j] = layer3_2_rprelu2_temp1;
        }
      }
      ap_fixed<32, 20> avgpool_val;
      avgpool_val = ((ap_fixed<32, 20>)0);
      loop_avgpool_ry: for (bit32 avgpool_ry = 0; avgpool_ry < 8; ++avgpool_ry) {
        loop_avgpool_rx: for (bit32 avgpool_rx = 0; avgpool_rx < 8; ++avgpool_rx) {
          avgpool_val = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)avgpool_val) + ((ap_fixed<33, 21>)avgpool_LB[avgpool_ry][avgpool_rx])));
        }
      }
      ap_fixed<32, 20> avgpool_res_temp;
      avgpool_res_temp = ((ap_fixed<32, 20>)(((ap_fixed<64, 20>)avgpool_val) / (ap_fixed<64, 20>)64));
      avgpool_res_pipe_112.write(avgpool_res_temp);
      avgpool_res[0][cc50][0][0] = avgpool_res_temp;
    }
    ap_fixed<32, 20> flatten[1][64];
    hls::stream<ap_fixed<32, 20> > flatten_pipe_113;
    #pragma HLS stream variable=flatten_pipe_113 depth=64
    flatten_j: for (bit32 j = 0; j < 64; ++j) {
    #pragma HLS pipeline
      ap_fixed<32, 20> avgpool_res_temp1;
      avgpool_res_temp1 = avgpool_res_pipe_112.read();
      ap_fixed<32, 20> flatten_temp;
      flatten_temp = avgpool_res_temp1;
      flatten_pipe_113.write(flatten_temp);
    }
    ap_fixed<32, 20> fc_matmul[1][10];
    ap_fixed<32, 20> flatten_temp[1][64];
    hls::stream<ap_fixed<32, 20> > fc_matmul_pipe_114;
    #pragma HLS stream variable=fc_matmul_pipe_114 depth=10
    fc_matmul_j1: for (bit32 j1 = 0; j1 < 10; ++j1) {
      float reducer0;
      reducer0 = 0.000000e+00f;
      if (j1 == 0) { // avoid reading multiple times
      for (int i = 0; i < 64; ++i)
        fc_matmul[0][i] = flatten_pipe_113.read();
      }
      fc_matmul_ra6: for (bit32 ra6 = 0; ra6 < 64; ++ra6) {
      #pragma HLS pipeline
        ap_fixed<32, 20> flatten_temp1;
        flatten_temp1 = fc_matmul[0][ra6];
        reducer0 = (((float)(((ap_fixed<64, 52>)flatten_temp1) * ((ap_fixed<64, 52>)w_fc_167[j1][ra6]))) + reducer0);
      }
      ap_fixed<32, 20> fc_matmul_temp;
      fc_matmul_temp = ((ap_fixed<32, 20>)reducer0);
      fc_matmul_pipe_114.write(fc_matmul_temp);
    }
    fc_j2: for (bit32 j2 = 0; j2 < 10; ++j2) {
    #pragma HLS pipeline
      ap_fixed<32, 20> fc_matmul_temp1;
      fc_matmul_temp1 = fc_matmul_pipe_114.read();
      fc[0][j2] = ((ap_fixed<32, 20>)(((ap_fixed<33, 21>)fc_matmul_temp1) + ((ap_fixed<33, 21>)w_fc_168[j2])));
    }
}
}