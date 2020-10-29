------ Host Code ------

void main(ap_uint<128>* global_cin, ap_uint<128>* global_prev_cin, ap_uint<128>* global_weight, ap_uint<128>* global_bias, ap_uint<128>* global_cout, ap_uint<32>* config) {
  ap_int<32> __device_scope;

  cl::Kernel kernel(program, "test", &err);
  cl::Buffer buffer_config(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_int<32>)*2789, config, &err);
  cl::Buffer buffer_global_cin(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_int<32>)*12625160, global_cin, &err);
  cl::Buffer buffer_global_prev_cin(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_int<32>)*12625160, global_prev_cin, &err);
  cl::Buffer buffer_global_weight(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_int<32>)*560032, global_weight, &err);
  cl::Buffer buffer_global_bias(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_int<32>)*16544, global_bias, &err);
  cl::Buffer buffer_global_cout(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(ap_int<32>)*826274, global_cout, &err);

  // set device kernel buffer
  err = kernel.setArg(0, buffer_config);
  err = kernel.setArg(1, buffer_global_cin);
  err = kernel.setArg(2, buffer_global_prev_cin);
  err = kernel.setArg(3, buffer_global_weight);
  err = kernel.setArg(4, buffer_global_bias);
  err = kernel.setArg(5, buffer_global_cout);
  err = q.enqueueMigrateMemObjects({buffer_config, buffer_global_cin, buffer_global_prev_cin, buffer_global_weight, buffer_global_bias, buffer_global_cout}, 0/*from host*/);
  q.finish();

  // enqueue kernel function
  std::chrono::duration<double> kernel_time(0);
  auto kernel_start = std::chrono::high_resolution_clock::now();
  cl::Event event;
  err = q.enqueueTask(kernel, NULL, &event);

  err = q.finish();
  auto kernel_end = std::chrono::high_resolution_clock::now();
  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
  auto kernel_time_in_sec = kernel_time.count();
  std::cout << "Execution Time:" <<  kernel_time_in_sec;
  err = q.enqueueMigrateMemObjects({buffer_config, buffer_global_cin, buffer_global_prev_cin, buffer_global_weight, buffer_global_bias, buffer_global_cout}, CL_MIGRATE_MEM_OBJECT_HOST);

  // execution on host 
}

------ Xcel Code ------

#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>

static void depth_conv(ap_uint<192> _top_depth_conv_cin[][1][63][633], ap_uint<0> _top_depth_conv_weight[][64][3][332], ap_uint<192> _top_depth_conv_config_in[32], ap_uint<0> _top_depth_conv_cout[][1][26][2664], ap_uint<0> _top_depth_conv_config_out[32]);
static void pool(float _top_pool_cin[][1][26][263], float _top_pool_config_in[32], float _top_pool_cout[][1][13][133], float _top_pool_config_out[32]);
static void relu_bn(ap_uint<0> _top_relu_bn_cin[][1][26][263], ap_uint<0> _top_relu_bn_config_in[32], ap_uint<0> _top_relu_bn_cout[][1][26][263], ap_uint<192> _top_relu_bn_config_out[32], ap_uint<0> _top_relu_bn_gamma_conv, ap_uint<192> _top_relu_bn_beta_conv);
static void nearest_neighbor_upsample(ap_uint<192> _top_nearest_neighbor_upsample_cin[][1][26][263], ap_uint<0> _top_nearest_neighbor_upsample_config_in[32], ap_uint<192> _top_nearest_neighbor_upsample_cout[][1][52][523], ap_uint<0> _top_nearest_neighbor_upsample_config_out[32]);
static void add(ap_uint<0> _top_add_cin1[][1][26][263], ap_uint<192> _top_add_cin2[][1][26][263], ap_uint<0> _top_add_config_in[32], ap_uint<192> _top_add_cout[][1][26][263], ap_uint<0> _top_add_config_out[32]);
static void depth_conv(ap_uint<192> _top_depth_conv_cin[][1][63][633], ap_uint<0> _top_depth_conv_weight[][64][3][332], ap_uint<192> _top_depth_conv_config_in[32], ap_uint<0> _top_depth_conv_cout[][1][26][2664], ap_uint<0> _top_depth_conv_config_out[32]) {
      #pragma HLS inline off
        ap_uint<32> compute0;
        compute0__1: for (ap_int<32> _1 = 0; _1 < 1; ++_1) {
          compute0 = ((ap_uint<32>)_top_depth_conv_config_in[19]);
        }
        ap_uint<1> unpack0[32];
        unpack0_indices: for (ap_int<32> indices = 0; indices < 32; ++indices) {
          ap_uint<1> unpack0_temp;
          unpack0_temp(0, 0) = ((ap_uint<1>)compute0(indices, indices));
          unpack0[indices] = unpack0_temp;
        }
        float compute1;
        compute1__2: for (ap_int<32> _2 = 0; _2 < 1; ++_2) {
          compute1 = ((float)unpack0[1]);
        }
        ap_uint<32> compute2;
        compute2__3: for (ap_int<32> _3 = 0; _3 < 1; ++_3) {
          compute2 = ((ap_uint<32>)_top_depth_conv_config_in[16]);
        }
        ap_uint<16> unpack1[2];
        unpack1_indices1: for (ap_int<32> indices1 = 0; indices1 < 2; ++indices1) {
          ap_uint<16> unpack1_temp;
          unpack1_temp(15, 0) = ((ap_uint<16>)compute2(((indices1 * 16) + 15), (indices1 * 16)));
          unpack1[indices1] = unpack1_temp;
        }
        float FILTER_S1;
        FILTER_S1__4: for (ap_int<32> _4 = 0; _4 < 1; ++_4) {
          FILTER_S1 = ((float)unpack1[0]);
        }
        ap_uint<32> compute3;
        compute3__5: for (ap_int<32> _5 = 0; _5 < 1; ++_5) {
          compute3 = ((ap_uint<32>)_top_depth_conv_config_in[6]);
        }
        float compute4[32];
        compute4_args: for (ap_int<32> args = 0; args < 32; ++args) {
          compute4[args] = ((float)_top_depth_conv_config_in[args]);
        }
        if (compute1) {
          if (FILTER_S1) {
            float temp[1][65][65][3];
            temp_indices2: for (ap_int<32> indices2 = 0; indices2 < 1; ++indices2) {
              temp_not_zero: for (ap_int<32> not_zero = 0; not_zero < 65; ++not_zero) {
                temp_index_tuple: for (ap_int<32> index_tuple = 0; index_tuple < 65; ++index_tuple) {
                  temp_i: for (ap_int<32> i = 0; i < 3; ++i) {
                    temp[indices2][not_zero][index_tuple][i] = ((float)(((((1 <= not_zero) && (not_zero < 64)) && (1 <= index_tuple)) && (index_tuple < 64)) ? ((ap_uint<0>)_top_depth_conv_cin[(((((index_tuple - ((index_tuple + -1) % 63)) + (not_zero * 63)) + (indices2 * 3969)) + -64) / 3969)][((((((index_tuple - ((index_tuple + -1) % 63)) + (not_zero * 63)) + (indices2 * 3969)) + -64) / 63) % 63)][((index_tuple + -1) % 63)][i]) : ((ap_uint<0>)0)));
                  }
                }
              }
            }
            float compute5[1][2][63][32];
            compute5_nn: for (ap_int<32> nn = 0; nn < 1; ++nn) {
              compute5_yy: for (ap_int<32> yy = 0; yy < 2; ++yy) {
                compute5_xx: for (ap_int<32> xx = 0; xx < 63; ++xx) {
                  compute5_ff: for (ap_int<32> ff = 0; ff < 32; ++ff) {
                    ap_uint<0> conv2d;
                    compute5_ry: for (ap_int<32> ry = 0; ry < 64; ++ry) {
                      compute5_rx: for (ap_int<32> rx = 0; rx < 3; ++rx) {
                        compute5_rc: for (ap_int<32> rc = 0; rc < 3; ++rc) {
                          conv2d = ((ap_uint<0>)((temp[nn][(yy + ry)][(xx + rx)][rc] * ((float)_top_depth_conv_weight[ry][rx][rc][ff])) + ((float)conv2d)));
                        }
                      }
                    }
                    compute5[nn][yy][xx][ff] = ((float)conv2d);
                  }
                }
              }
            }
          } else {
            float temp_[1][65][65][3];
            temp_indices3: for (ap_int<32> indices3 = 0; indices3 < 1; ++indices3) {
              temp_not_zero1: for (ap_int<32> not_zero1 = 0; not_zero1 < 65; ++not_zero1) {
                temp_index_tuple1: for (ap_int<32> index_tuple1 = 0; index_tuple1 < 65; ++index_tuple1) {
                  temp_i1: for (ap_int<32> i1 = 0; i1 < 3; ++i1) {
                    temp_[indices3][not_zero1][index_tuple1][i1] = ((float)(((((1 <= not_zero1) && (not_zero1 < 64)) && (1 <= index_tuple1)) && (index_tuple1 < 64)) ? ((ap_uint<0>)_top_depth_conv_cin[(((((index_tuple1 - ((index_tuple1 + -1) % 63)) + (not_zero1 * 63)) + (indices3 * 3969)) + -64) / 3969)][((((((index_tuple1 - ((index_tuple1 + -1) % 63)) + (not_zero1 * 63)) + (indices3 * 3969)) + -64) / 63) % 63)][((index_tuple1 + -1) % 63)][i1]) : ((ap_uint<0>)0)));
                  }
                }
              }
            }
            float compute6[1][2][63][32];
            compute6_nn1: for (ap_int<32> nn1 = 0; nn1 < 1; ++nn1) {
              compute6_yy1: for (ap_int<32> yy1 = 0; yy1 < 2; ++yy1) {
                compute6_xx1: for (ap_int<32> xx1 = 0; xx1 < 63; ++xx1) {
                  compute6_ff1: for (ap_int<32> ff1 = 0; ff1 < 32; ++ff1) {
                    ap_uint<0> conv2d_;
                    compute6_ry1: for (ap_int<32> ry1 = 0; ry1 < 64; ++ry1) {
                      compute6_rx1: for (ap_int<32> rx1 = 0; rx1 < 3; ++rx1) {
                        compute6_rc1: for (ap_int<32> rc1 = 0; rc1 < 3; ++rc1) {
                          conv2d_ = ((ap_uint<0>)((temp_[nn1][(yy1 + ry1)][(xx1 + rx1)][rc1] * ((float)_top_depth_conv_weight[ry1][rx1][rc1][ff1])) + ((float)conv2d_)));
                        }
                      }
                    }
                    compute6[nn1][yy1][xx1][ff1] = ((float)conv2d_);
                  }
                }
              }
            }
          }
        } else {
          float compute7[1][63][63][3];
          compute7_args1: for (ap_int<32> args1 = 0; args1 < 1; ++args1) {
            compute7_args0: for (ap_int<32> args0 = 0; args0 < 63; ++args0) {
              compute7_args11: for (ap_int<32> args11 = 0; args11 < 63; ++args11) {
                compute7_args2: for (ap_int<32> args2 = 0; args2 < 3; ++args2) {
                  compute7[args1][args0][args11][args2] = ((float)_top_depth_conv_cin[args1][args0][args11][args2]);
                }
              }
            }
          }
        }
      }

static void pool(float _top_pool_cin[][1][26][263], float _top_pool_config_in[32], float _top_pool_cout[][1][13][133], float _top_pool_config_out[32]) {
      #pragma HLS inline off
        ap_uint<32> compute8;
        compute8__1: for (ap_int<32> _1 = 0; _1 < 1; ++_1) {
          compute8 = ((ap_uint<32>)_top_pool_config_in[19]);
        }
        ap_uint<1> unpack2[32];
        unpack2_indices: for (ap_int<32> indices = 0; indices < 32; ++indices) {
          ap_uint<1> unpack2_temp;
          unpack2_temp(0, 0) = ((ap_uint<1>)compute8(indices, indices));
          unpack2[indices] = unpack2_temp;
        }
        ap_uint<1> compute9;
        compute9__2: for (ap_int<32> _2 = 0; _2 < 1; ++_2) {
          compute9 = unpack2[5];
        }
        float compute10[32];
        compute10_args: for (ap_int<32> args = 0; args < 32; ++args) {
          compute10[args] = _top_pool_config_in[args];
        }
        if (compute9) {
          float pad[1][26][26][3];
          pad_indices1: for (ap_int<32> indices1 = 0; indices1 < 1; ++indices1) {
            pad_not_zero: for (ap_int<32> not_zero = 0; not_zero < 26; ++not_zero) {
              pad_index_tuple: for (ap_int<32> index_tuple = 0; index_tuple < 26; ++index_tuple) {
                pad_i: for (ap_int<32> i = 0; i < 3; ++i) {
                  pad[indices1][not_zero][index_tuple][i] = _top_pool_cin[indices1][not_zero][index_tuple][i];
                }
              }
            }
          }
          float pool_[1][13][13][3];
          pool_i1: for (ap_int<32> i1 = 0; i1 < 1; ++i1) {
            pool_h: for (ap_int<32> h = 0; h < 13; ++h) {
              pool_w: for (ap_int<32> w = 0; w < 13; ++w) {
                pool_c: for (ap_int<32> c = 0; c < 3; ++c) {
                  float reducer0;
                  reducer0_x: for (ap_int<32> x = 0; x < 1; ++x) {
                    reducer0 = -3.402823e+38f;
                  }
                  pool__ra0: for (ap_int<32> ra0 = 0; ra0 < 2; ++ra0) {
                    pool__ra1: for (ap_int<32> ra1 = 0; ra1 < 2; ++ra1) {
                      reducer0 = hls::max(pad[i1][((h * 2) + ra0)][((w * 2) + ra1)][c], reducer0);
                    }
                  }
                  pool_[i1][h][w][c] = reducer0;
                }
              }
            }
          }
        } else {
          float compute11[1][26][26][3];
          compute11_args1: for (ap_int<32> args1 = 0; args1 < 1; ++args1) {
            compute11_args0: for (ap_int<32> args0 = 0; args0 < 26; ++args0) {
              compute11_args11: for (ap_int<32> args11 = 0; args11 < 26; ++args11) {
                compute11_args2: for (ap_int<32> args2 = 0; args2 < 3; ++args2) {
                  compute11[args1][args0][args11][args2] = _top_pool_cin[args1][args0][args11][args2];
                }
              }
            }
          }
        }
      }

static void relu_bn(ap_uint<0> _top_relu_bn_cin[][1][26][263], ap_uint<0> _top_relu_bn_config_in[32], ap_uint<0> _top_relu_bn_cout[][1][26][263], ap_uint<192> _top_relu_bn_config_out[32], ap_uint<0> _top_relu_bn_gamma_conv, ap_uint<192> _top_relu_bn_beta_conv) {
      #pragma HLS inline off
        ap_uint<32> compute12;
        compute12__1: for (ap_int<32> _1 = 0; _1 < 1; ++_1) {
          compute12 = ((ap_uint<32>)_top_relu_bn_config_in[19]);
        }
        ap_uint<1> unpack3[32];
        unpack3_indices: for (ap_int<32> indices = 0; indices < 32; ++indices) {
          ap_uint<1> unpack3_temp;
          unpack3_temp(0, 0) = ((ap_uint<1>)compute12(indices, indices));
          unpack3[indices] = unpack3_temp;
        }
        ap_uint<1> compute13;
        compute13__2: for (ap_int<32> _2 = 0; _2 < 1; ++_2) {
          compute13 = unpack3[2];
        }
        ap_uint<1> compute14;
        compute14__3: for (ap_int<32> _3 = 0; _3 < 1; ++_3) {
          compute14 = unpack3[3];
        }
        ap_uint<1> compute15;
        compute15__4: for (ap_int<32> _4 = 0; _4 < 1; ++_4) {
          compute15 = unpack3[4];
        }
        ap_uint<1> compute16;
        compute16__5: for (ap_int<32> _5 = 0; _5 < 1; ++_5) {
          compute16 = unpack3[7];
        }
        ap_uint<1> compute17;
        compute17__6: for (ap_int<32> _6 = 0; _6 < 1; ++_6) {
          compute17 = unpack3[10];
        }
        ap_uint<1> compute18;
        compute18__7: for (ap_int<32> _7 = 0; _7 < 1; ++_7) {
          compute18 = unpack3[12];
        }
        float compute19[32];
        compute19_args: for (ap_int<32> args = 0; args < 32; ++args) {
          compute19[args] = ((float)_top_relu_bn_config_in[args]);
        }
        if (((compute14 || compute15) || compute16) || compute17) {
          if ((compute16 && compute13) || compute17) {
            float compute20[1][26][26][3];
            compute20_args1: for (ap_int<32> args1 = 0; args1 < 1; ++args1) {
              compute20_args0: for (ap_int<32> args0 = 0; args0 < 26; ++args0) {
                compute20_args11: for (ap_int<32> args11 = 0; args11 < 26; ++args11) {
                  compute20_args2: for (ap_int<32> args2 = 0; args2 < 3; ++args2) {
                    compute20[args1][args0][args11][args2] = ((float)(((ap_uint<1>)(_top_relu_bn_cin[args1][args0][args11][args2] * _top_relu_bn_gamma_conv)) + ((ap_uint<1>)_top_relu_bn_beta_conv)));
                  }
                }
              }
            }
          }
          if (compute15 && (((ap_uint<32>)compute18) == 0U)) {
            float compute21[1][26][26][3];
            compute21_args3: for (ap_int<32> args3 = 0; args3 < 1; ++args3) {
              compute21_args01: for (ap_int<32> args01 = 0; args01 < 26; ++args01) {
                compute21_args12: for (ap_int<32> args12 = 0; args12 < 26; ++args12) {
                  compute21_args21: for (ap_int<32> args21 = 0; args21 < 3; ++args21) {
                    compute21[args3][args01][args12][args21] = ((float)((6U < ((ap_uint<32>)_top_relu_bn_cin[args3][args01][args12][args21])) ? ((ap_uint<0>)6) : ((ap_uint<0>)_top_relu_bn_cin[args3][args01][args12][args21])));
                  }
                }
              }
            }
          } else {
            if (compute14) {
              float compute22[1][26][26][3];
              compute22_args4: for (ap_int<32> args4 = 0; args4 < 1; ++args4) {
                compute22_args02: for (ap_int<32> args02 = 0; args02 < 26; ++args02) {
                  compute22_args13: for (ap_int<32> args13 = 0; args13 < 26; ++args13) {
                    compute22_args22: for (ap_int<32> args22 = 0; args22 < 3; ++args22) {
                      compute22[args4][args02][args13][args22] = ((float)((((ap_uint<32>)_top_relu_bn_cin[args4][args02][args13][args22]) < 0U) ? ((ap_uint<0>)0) : ((ap_uint<0>)_top_relu_bn_cin[args4][args02][args13][args22])));
                    }
                  }
                }
              }
            }
          }
        } else {
          float compute23[1][26][26][3];
          compute23_args5: for (ap_int<32> args5 = 0; args5 < 1; ++args5) {
            compute23_args03: for (ap_int<32> args03 = 0; args03 < 26; ++args03) {
              compute23_args14: for (ap_int<32> args14 = 0; args14 < 26; ++args14) {
                compute23_args23: for (ap_int<32> args23 = 0; args23 < 3; ++args23) {
                  compute23[args5][args03][args14][args23] = ((float)_top_relu_bn_cin[args5][args03][args14][args23]);
                }
              }
            }
          }
        }
      }

static void nearest_neighbor_upsample(ap_uint<192> _top_nearest_neighbor_upsample_cin[][1][26][263], ap_uint<0> _top_nearest_neighbor_upsample_config_in[32], ap_uint<192> _top_nearest_neighbor_upsample_cout[][1][52][523], ap_uint<0> _top_nearest_neighbor_upsample_config_out[32]) {
      #pragma HLS inline off
        ap_uint<32> compute24;
        compute24__1: for (ap_int<32> _1 = 0; _1 < 1; ++_1) {
          compute24 = ((ap_uint<32>)_top_nearest_neighbor_upsample_config_in[19]);
        }
        ap_uint<1> unpack4[32];
        unpack4_indices: for (ap_int<32> indices = 0; indices < 32; ++indices) {
          ap_uint<1> unpack4_temp;
          unpack4_temp(0, 0) = ((ap_uint<1>)compute24(indices, indices));
          unpack4[indices] = unpack4_temp;
        }
        float compute25;
        compute25__2: for (ap_int<32> _2 = 0; _2 < 1; ++_2) {
          compute25 = ((float)unpack4[6]);
        }
        float compute26[32];
        compute26_args: for (ap_int<32> args = 0; args < 32; ++args) {
          compute26[args] = ((float)_top_nearest_neighbor_upsample_config_in[args]);
        }
        if (compute25) {
          float compute27[1][52][52][3];
          compute27_n_i: for (ap_int<32> n_i = 0; n_i < 1; ++n_i) {
            compute27_h_i: for (ap_int<32> h_i = 0; h_i < 52; ++h_i) {
              compute27_w_i: for (ap_int<32> w_i = 0; w_i < 52; ++w_i) {
                compute27_c_i: for (ap_int<32> c_i = 0; c_i < 3; ++c_i) {
                  compute27[n_i][h_i][w_i][c_i] = ((float)_top_nearest_neighbor_upsample_cin[n_i][(h_i / 2)][(w_i / 2)][c_i]);
                }
              }
            }
          }
        } else {
          float compute28[1][26][26][3];
          compute28_args1: for (ap_int<32> args1 = 0; args1 < 1; ++args1) {
            compute28_args0: for (ap_int<32> args0 = 0; args0 < 26; ++args0) {
              compute28_args11: for (ap_int<32> args11 = 0; args11 < 26; ++args11) {
                compute28_args2: for (ap_int<32> args2 = 0; args2 < 3; ++args2) {
                  compute28[args1][args0][args11][args2] = ((float)_top_nearest_neighbor_upsample_cin[args1][args0][args11][args2]);
                }
              }
            }
          }
        }
      }

static void add(ap_uint<0> _top_add_cin1[][1][26][263], ap_uint<192> _top_add_cin2[][1][26][263], ap_uint<0> _top_add_config_in[32], ap_uint<192> _top_add_cout[][1][26][263], ap_uint<0> _top_add_config_out[32]) {
      #pragma HLS inline off
        ap_uint<32> compute29;
        compute29__1: for (ap_int<32> _1 = 0; _1 < 1; ++_1) {
          compute29 = ((ap_uint<32>)_top_add_config_in[19]);
        }
        ap_uint<1> unpack5[32];
        unpack5_indices: for (ap_int<32> indices = 0; indices < 32; ++indices) {
          ap_uint<1> unpack5_temp;
          unpack5_temp(0, 0) = ((ap_uint<1>)compute29(indices, indices));
          unpack5[indices] = unpack5_temp;
        }
        float compute30;
        compute30__2: for (ap_int<32> _2 = 0; _2 < 1; ++_2) {
          compute30 = ((float)unpack5[11]);
        }
        float compute31[32];
        compute31_args: for (ap_int<32> args = 0; args < 32; ++args) {
          compute31[args] = ((float)_top_add_config_in[args]);
        }
        if (compute30) {
          float compute32[1][26][26][3];
          compute32_args1: for (ap_int<32> args1 = 0; args1 < 1; ++args1) {
            compute32_args0: for (ap_int<32> args0 = 0; args0 < 26; ++args0) {
              compute32_args11: for (ap_int<32> args11 = 0; args11 < 26; ++args11) {
                compute32_args2: for (ap_int<32> args2 = 0; args2 < 3; ++args2) {
                  compute32[args1][args0][args11][args2] = ((float)(((ap_uint<1>)_top_add_cin1[args1][args0][args11][args2]) + ((ap_uint<1>)_top_add_cin2[args1][args0][args11][args2])));
                }
              }
            }
          }
        } else {
          float compute33[1][26][26][3];
          compute33_args3: for (ap_int<32> args3 = 0; args3 < 1; ++args3) {
            compute33_args01: for (ap_int<32> args01 = 0; args01 < 26; ++args01) {
              compute33_args12: for (ap_int<32> args12 = 0; args12 < 26; ++args12) {
                compute33_args21: for (ap_int<32> args21 = 0; args21 < 3; ++args21) {
                  compute33[args3][args01][args12][args21] = ((float)_top_add_cin2[args3][args01][args12][args21]);
                }
              }
            }
          }
        }
      }

extern "C" {
void test(ap_uint<32> config[2789], ap_uint<128> global_cin[12625160], ap_uint<128> global_prev_cin[12625160], ap_uint<128> global_weight[560032], ap_uint<128> global_bias[16544], ap_uint<128> global_cout[826274]) {
    #pragma HLS INTERFACE m_axi port=config offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=global_cin offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=global_prev_cin offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=global_weight offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=global_bias offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=global_cout offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=config bundle=control
    #pragma HLS INTERFACE s_axilite port=global_cin bundle=control
    #pragma HLS INTERFACE s_axilite port=global_prev_cin bundle=control
    #pragma HLS INTERFACE s_axilite port=global_weight bundle=control
    #pragma HLS INTERFACE s_axilite port=global_bias bundle=control
    #pragma HLS INTERFACE s_axilite port=global_cout bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
      float _top;
      float depth_conv;
      float pool;
      float relu_bn;
      float nearest_neighbor_upsample;
      float add;
      float top_kernel;
      i: for (ap_int<32> i = 0; i < 86; ++i) {
        float layer_config[32];
        layer_config_x: for (ap_int<32> x = 0; x < 32; ++x) {
          layer_config[x] = ((float)config[(x + (i * 32))]);
        }
        ap_uint<0> fifo_cin_load_0[1000];
        ap_uint<0> fifo_weight_load_0[1000];
        ap_uint<0> fifo_weight_load_1[1000];
        ap_uint<0> fifo_depth_conv_0[1000];
        ap_uint<0> fifo_relu6_0[1000];
        ap_uint<0> fifo_conv_0[1000];
        ap_uint<0> fifo_add_0[1000];
        ap_uint<0> fifo_relu_0[1000];
        ap_uint<0> fifo_upsample_0[1000];
        ap_uint<0> fifo_upsample_1[1000];
        ap_uint<0> fifo_merge_0[1000];
        ap_uint<0> fifo_cin_prev_0[1000];
        ap_uint<0> fifo_beta_depth[1000];
        ap_uint<0> fifo_gamma_depth[1000];
        ap_uint<0> fifo_beta_conv[1000];
        ap_uint<0> fifo_gamma_conv[1000];
        ap_uint<192> config_prev_load[1000];
        ap_uint<192> config_weight_load[1000];
        ap_uint<192> config_depth_conv[1000];
        ap_uint<192> config_relu6[1000];
        ap_uint<192> config_conv[1000];
        ap_uint<192> config_add[1000];
        ap_uint<192> config_relu[1000];
        ap_uint<192> config_upsample[1000];
        ap_uint<192> config_merge[1000];
        ap_uint<192> config_data_write[1000];
        float cin_load;
        cin_load(global_cin, layer_config, fifo_cin_load_0, config_prev_load);
        float cin_load_prev;
        cin_load_prev(global_prev_cin, config_prev_load, fifo_cin_prev_0, config_weight_load);
        float weight_load;
        weight_load(global_weight, global_bias, config_weight_load, fifo_weight_load_0, fifo_weight_load_1, fifo_beta_depth, fifo_gamma_depth, fifo_beta_conv, fifo_gamma_conv, config_depth_conv);
        float depth_conv0;
        depth_conv(fifo_cin_load_0, fifo_weight_load_0, config_depth_conv, fifo_depth_conv_0, config_relu6);
        float relu_bn0;
        relu_bn(fifo_depth_conv_0, config_relu6, fifo_relu6_0, config_conv, fifo_beta_depth, fifo_gamma_depth);
        float conv;
        kernel(fifo_relu6_0, fifo_weight_load_1, fifo_conv_0, config_conv, config_relu);
        float relu_bn1;
        relu_bn(fifo_conv_0, config_relu, fifo_relu6_0, config_add, fifo_beta_conv, fifo_gamma_conv);
        float add0;
        add(fifo_cin_prev_0, fifo_relu_0, config_add, fifo_add_0, config_upsample);
        float nearest_neighbor_upsample0;
        nearest_neighbor_upsample(fifo_add_0, config_upsample, fifo_upsample_0, fifo_merge_0);
        float cout_write;
        cout_write(fifo_merge_0, config_data_write, global_cout);
      }
    }
}

