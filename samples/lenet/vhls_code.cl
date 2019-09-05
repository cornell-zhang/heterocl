#include <ap_int.h>
#include <ap_fixed.h>
#include <math.h>

void default_function(ap_int<32> input_image[1000][1][28][28], ap_int<32> weight_conv1[20][1][5][5], ap_int<32> weight_conv2[50][20][5][5], ap_int<32> weight_fc1[500][800], ap_int<32> weight_fc2[10][500], ap_int<32> lenet[1000][10]) {
  ap_int<32> conv2d[1000][20][24][24];
  for (ap_int<32> nn = 0; nn < 1000; ++nn) {
    for (ap_int<32> ff = 0; ff < 20; ++ff) {
      for (ap_int<32> yy = 0; yy < 24; ++yy) {
        for (ap_int<32> xx = 0; xx < 24; ++xx) {
          float reducer12;
          reducer12 = 0.000000e+00f;
          for (ap_int<32> ra29 = 0; ra29 < 5; ++ra29) {
            for (ap_int<32> ra30 = 0; ra30 < 5; ++ra30) {
              reducer12 = (((float)(((ap_int<64>)input_image[nn][0][(yy + ra29)][(xx + ra30)]) * ((ap_int<64>)weight_conv1[ff][0][ra29][ra30]))) + reducer12);
            }
          }
          conv2d[nn][ff][yy][xx] = ((ap_int<32>)reducer12);
        }
      }
    }
  }
  ap_int<32> tanh1[1000][20][24][24];
  for (ap_int<32> args = 0; args < 1000; ++args) {
    for (ap_int<32> args0 = 0; args0 < 20; ++args0) {
      for (ap_int<32> args1 = 0; args1 < 24; ++args1) {
        for (ap_int<32> args2 = 0; args2 < 24; ++args2) {
          tanh1[args][args0][args1][args2] = ((ap_int<32>)tanh(((double)conv2d[args][args0][args1][args2])));
        }
      }
    }
  }
  ap_int<32> max_pool[1000][20][12][12];
  for (ap_int<32> i = 0; i < 1000; ++i) {
    for (ap_int<32> c = 0; c < 20; ++c) {
      for (ap_int<32> h = 0; h < 12; ++h) {
        for (ap_int<32> w = 0; w < 12; ++w) {
          float reducer13;
          reducer13 = -1.000000e+00f;
          for (ap_int<32> ra31 = 0; ra31 < 2; ++ra31) {
            for (ap_int<32> ra32 = 0; ra32 < 2; ++ra32) {
              reducer13 = std::max(((float)tanh1[i][c][((h * 2) + ra31)][((w * 2) + ra32)]), reducer13);
            }
          }
          max_pool[i][c][h][w] = ((ap_int<32>)reducer13);
        }
      }
    }
  }
  ap_int<32> conv2d1[1000][50][8][8];
  for (ap_int<32> nn1 = 0; nn1 < 1000; ++nn1) {
    for (ap_int<32> ff1 = 0; ff1 < 50; ++ff1) {
      for (ap_int<32> yy1 = 0; yy1 < 8; ++yy1) {
        for (ap_int<32> xx1 = 0; xx1 < 8; ++xx1) {
          float reducer14;
          reducer14 = 0.000000e+00f;
          for (ap_int<32> ra33 = 0; ra33 < 20; ++ra33) {
            for (ap_int<32> ra34 = 0; ra34 < 5; ++ra34) {
              for (ap_int<32> ra35 = 0; ra35 < 5; ++ra35) {
                reducer14 = (((float)(((ap_int<64>)max_pool[nn1][ra33][(yy1 + ra34)][(xx1 + ra35)]) * ((ap_int<64>)weight_conv2[ff1][ra33][ra34][ra35]))) + reducer14);
              }
            }
          }
          conv2d1[nn1][ff1][yy1][xx1] = ((ap_int<32>)reducer14);
        }
      }
    }
  }
  ap_int<32> tanh2[1000][50][8][8];
  for (ap_int<32> args3 = 0; args3 < 1000; ++args3) {
    for (ap_int<32> args01 = 0; args01 < 50; ++args01) {
      for (ap_int<32> args11 = 0; args11 < 8; ++args11) {
        for (ap_int<32> args21 = 0; args21 < 8; ++args21) {
          tanh2[args3][args01][args11][args21] = ((ap_int<32>)tanh(((double)conv2d1[args3][args01][args11][args21])));
        }
      }
    }
  }
  ap_int<32> max_pool1[1000][50][4][4];
  for (ap_int<32> i1 = 0; i1 < 1000; ++i1) {
    for (ap_int<32> c1 = 0; c1 < 50; ++c1) {
      for (ap_int<32> h1 = 0; h1 < 4; ++h1) {
        for (ap_int<32> w1 = 0; w1 < 4; ++w1) {
          float reducer15;
          reducer15 = -1.000000e+00f;
          for (ap_int<32> ra36 = 0; ra36 < 2; ++ra36) {
            for (ap_int<32> ra37 = 0; ra37 < 2; ++ra37) {
              reducer15 = std::max(((float)tanh2[i1][c1][((h1 * 2) + ra36)][((w1 * 2) + ra37)]), reducer15);
            }
          }
          max_pool1[i1][c1][h1][w1] = ((ap_int<32>)reducer15);
        }
      }
    }
  }
  ap_int<32> compute6[1000][800];
  for (ap_int<32> i2 = 0; i2 < 1000; ++i2) {
    for (ap_int<32> j = 0; j < 800; ++j) {
      compute6[i2][j] = max_pool1[i2][(j / 16)][((j / 4) % 4)][(j % 4)];
    }
  }
  ap_int<32> dense[1000][500];
  for (ap_int<32> i3 = 0; i3 < 1000; ++i3) {
    for (ap_int<32> j1 = 0; j1 < 500; ++j1) {
      float reducer16;
      reducer16 = 0.000000e+00f;
      for (ap_int<32> ra38 = 0; ra38 < 800; ++ra38) {
        reducer16 = (((float)(((ap_int<64>)compute6[i3][ra38]) * ((ap_int<64>)weight_fc1[j1][ra38]))) + reducer16);
      }
      dense[i3][j1] = ((ap_int<32>)reducer16);
    }
  }
  ap_int<32> tanh3[1000][500];
  for (ap_int<32> args4 = 0; args4 < 1000; ++args4) {
    for (ap_int<32> args02 = 0; args02 < 500; ++args02) {
      tanh3[args4][args02] = ((ap_int<32>)tanh(((double)dense[args4][args02])));
    }
  }
  ap_int<32> dense1[1000][10];
  for (ap_int<32> i4 = 0; i4 < 1000; ++i4) {
    for (ap_int<32> j2 = 0; j2 < 10; ++j2) {
      float reducer17;
      reducer17 = 0.000000e+00f;
      for (ap_int<32> ra39 = 0; ra39 < 500; ++ra39) {
        reducer17 = (((float)(((ap_int<64>)tanh3[i4][ra39]) * ((ap_int<64>)weight_fc2[j2][ra39]))) + reducer17);
      }
      dense1[i4][j2] = ((ap_int<32>)reducer17);
    }
  }
  ap_int<32> compute7[1000];
  for (ap_int<32> i5 = 0; i5 < 1000; ++i5) {
    ap_int<32> max;
    max = 0;
    for (ap_int<32> ra40 = 0; ra40 < 10; ++ra40) {
      max = std::max(dense1[i5][ra40], max);
    }
    compute7[i5] = max;
  }
  ap_int<32> compute8[1000];
  for (ap_int<32> i6 = 0; i6 < 1000; ++i6) {
    ap_int<32> sum;
    sum = 0;
    for (ap_int<32> ra41 = 0; ra41 < 10; ++ra41) {
      sum = ((ap_int<32>)(exp(((double)((ap_int<33>)(dense1[i6][ra41] - compute7[i6])))) + ((double)sum)));
    }
    compute8[i6] = sum;
  }
  ap_int<32> update2;
  for (ap_int<32> i7 = 0; i7 < 1000; ++i7) {
    for (ap_int<32> j3 = 0; j3 < 10; ++j3) {
      lenet[i7][j3] = ((ap_int<32>)(exp(((double)((ap_int<33>)(dense1[i7][j3] - compute7[i7])))) / ((double)compute8[i7])));
    }
  }
}

