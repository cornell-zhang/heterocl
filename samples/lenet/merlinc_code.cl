#include <string.h>
#include <math.h>
#include <assert.h>
#pragma ACCEL kernel
void default_function(int* input_image, int* weight_conv1, int* weight_conv2, int* weight_fc1, int* weight_fc2, int* lenet) {
  int conv2d[11520000];
  for (int nn = 0; nn < 1000; ++nn) {
    for (int ff = 0; ff < 20; ++ff) {
      for (int yy = 0; yy < 24; ++yy) {
        for (int xx = 0; xx < 24; ++xx) {
          float reducer0;
          reducer0 = 0.000000e+00f;
          for (int ra1 = 0; ra1 < 5; ++ra1) {
            for (int ra2 = 0; ra2 < 5; ++ra2) {
              reducer0 = (((float)(((long)input_image[(((xx + ra2) + ((yy + ra1) * 28)) + (nn * 784))]) * ((long)weight_conv1[((ra2 + (ra1 * 5)) + (ff * 25))]))) + reducer0);
            }
          }
          conv2d[(((xx + (yy * 24)) + (ff * 576)) + (nn * 11520))] = ((int)reducer0);
        }
      }
    }
  }
  int tanh1[11520000];
  for (int args = 0; args < 1000; ++args) {
    for (int args0 = 0; args0 < 20; ++args0) {
      for (int args1 = 0; args1 < 24; ++args1) {
        for (int args2 = 0; args2 < 24; ++args2) {
          tanh1[(((args2 + (args1 * 24)) + (args0 * 576)) + (args * 11520))] = ((int)tanh(((double)conv2d[(((args2 + (args1 * 24)) + (args0 * 576)) + (args * 11520))])));
        }
      }
    }
  }
  int max_pool[2880000];
  for (int i = 0; i < 1000; ++i) {
    for (int c = 0; c < 20; ++c) {
      for (int h = 0; h < 12; ++h) {
        for (int w = 0; w < 12; ++w) {
          float reducer1;
          reducer1 = -1.000000e+00f;
          for (int ra3 = 0; ra3 < 2; ++ra3) {
            for (int ra4 = 0; ra4 < 2; ++ra4) {
              reducer1 = max(((float)tanh1[(((((w * 2) + ra4) + (((h * 2) + ra3) * 24)) + (c * 576)) + (i * 11520))]), reducer1);
            }
          }
          max_pool[(((w + (h * 12)) + (c * 144)) + (i * 2880))] = ((int)reducer1);
        }
      }
    }
  }
  int conv2d1[3200000];
  for (int nn1 = 0; nn1 < 1000; ++nn1) {
    for (int ff1 = 0; ff1 < 50; ++ff1) {
      for (int yy1 = 0; yy1 < 8; ++yy1) {
        for (int xx1 = 0; xx1 < 8; ++xx1) {
          float reducer2;
          reducer2 = 0.000000e+00f;
          for (int ra5 = 0; ra5 < 20; ++ra5) {
            for (int ra6 = 0; ra6 < 5; ++ra6) {
              for (int ra7 = 0; ra7 < 5; ++ra7) {
                reducer2 = (((float)(((long)max_pool[((((xx1 + ra7) + ((yy1 + ra6) * 12)) + (ra5 * 144)) + (nn1 * 2880))]) * ((long)weight_conv2[(((ra7 + (ra6 * 5)) + (ra5 * 25)) + (ff1 * 500))]))) + reducer2);
              }
            }
          }
          conv2d1[(((xx1 + (yy1 * 8)) + (ff1 * 64)) + (nn1 * 3200))] = ((int)reducer2);
        }
      }
    }
  }
  int tanh2[3200000];
  for (int args3 = 0; args3 < 1000; ++args3) {
    for (int args01 = 0; args01 < 50; ++args01) {
      for (int args11 = 0; args11 < 8; ++args11) {
        for (int args21 = 0; args21 < 8; ++args21) {
          tanh2[(((args21 + (args11 * 8)) + (args01 * 64)) + (args3 * 3200))] = ((int)tanh(((double)conv2d1[(((args21 + (args11 * 8)) + (args01 * 64)) + (args3 * 3200))])));
        }
      }
    }
  }
  int max_pool1[800000];
  for (int i1 = 0; i1 < 1000; ++i1) {
    for (int c1 = 0; c1 < 50; ++c1) {
      for (int h1 = 0; h1 < 4; ++h1) {
        for (int w1 = 0; w1 < 4; ++w1) {
          float reducer3;
          reducer3 = -1.000000e+00f;
          for (int ra8 = 0; ra8 < 2; ++ra8) {
            for (int ra9 = 0; ra9 < 2; ++ra9) {
              reducer3 = max(((float)tanh2[(((((w1 * 2) + ra9) + (((h1 * 2) + ra8) * 8)) + (c1 * 64)) + (i1 * 3200))]), reducer3);
            }
          }
          max_pool1[(((w1 + (h1 * 4)) + (c1 * 16)) + (i1 * 800))] = ((int)reducer3);
        }
      }
    }
  }
  int compute0[800000];
  for (int i2 = 0; i2 < 1000; ++i2) {
    for (int j = 0; j < 800; ++j) {
      compute0[(j + (i2 * 800))] = max_pool1[((((((j / 4) % 4) * 4) + (j % 4)) + ((j / 16) * 16)) + (i2 * 800))];
    }
  }
  int dense[500000];
  for (int i3 = 0; i3 < 1000; ++i3) {
    for (int j1 = 0; j1 < 500; ++j1) {
      float reducer4;
      reducer4 = 0.000000e+00f;
      for (int ra10 = 0; ra10 < 800; ++ra10) {
        reducer4 = (((float)(((long)compute0[(ra10 + (i3 * 800))]) * ((long)weight_fc1[(ra10 + (j1 * 800))]))) + reducer4);
      }
      dense[(j1 + (i3 * 500))] = ((int)reducer4);
    }
  }
  int tanh3[500000];
  for (int args4 = 0; args4 < 1000; ++args4) {
    for (int args02 = 0; args02 < 500; ++args02) {
      tanh3[(args02 + (args4 * 500))] = ((int)tanh(((double)dense[(args02 + (args4 * 500))])));
    }
  }
  int dense1[10000];
  for (int i4 = 0; i4 < 1000; ++i4) {
    for (int j2 = 0; j2 < 10; ++j2) {
      float reducer5;
      reducer5 = 0.000000e+00f;
      for (int ra11 = 0; ra11 < 500; ++ra11) {
        reducer5 = (((float)(((long)tanh3[(ra11 + (i4 * 500))]) * ((long)weight_fc2[(ra11 + (j2 * 500))]))) + reducer5);
      }
      dense1[(j2 + (i4 * 10))] = ((int)reducer5);
    }
  }
  int compute1[1000];
  for (int i5 = 0; i5 < 1000; ++i5) {
    int max;
    max = 0;
    for (int ra12 = 0; ra12 < 10; ++ra12) {
      max = max(dense1[(ra12 + (i5 * 10))], max);
    }
    compute1[i5] = max;
  }
  int compute2[1000];
  for (int i6 = 0; i6 < 1000; ++i6) {
    int sum;
    sum = 0;
    for (int ra13 = 0; ra13 < 10; ++ra13) {
      sum = ((int)(exp(((double)((long)(dense1[(ra13 + (i6 * 10))] - compute1[i6])))) + ((double)sum)));
    }
    compute2[i6] = sum;
  }
  int update0;
  for (int i7 = 0; i7 < 1000; ++i7) {
    for (int j3 = 0; j3 < 10; ++j3) {
      lenet[(j3 + (i7 * 10))] = ((int)(exp(((double)((long)(dense1[(j3 + (i7 * 10))] - compute1[i7])))) / ((double)compute2[i7])));
    }
  }
}

