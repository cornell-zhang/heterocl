__kernel void default_function(__global int* input_image, __global int* weight_conv1, __global int* weight_conv2, __global int* weight_fc1, __global int* weight_fc2, __global int* lenet) {
  __local int conv2d[11520000];
  for (int nn = 0; nn < 1000; ++nn) {
    for (int ff = 0; ff < 20; ++ff) {
      for (int yy = 0; yy < 24; ++yy) {
        for (int xx = 0; xx < 24; ++xx) {
          __local float reducer6;
          reducer6 = 0.000000e+00f;
          for (int ra15 = 0; ra15 < 5; ++ra15) {
            for (int ra16 = 0; ra16 < 5; ++ra16) {
              reducer6 = (((float)(((long)input_image[(((xx + ra16) + ((yy + ra15) * 28)) + (nn * 784))]) * ((long)weight_conv1[((ra16 + (ra15 * 5)) + (ff * 25))]))) + reducer6);
            }
          }
          conv2d[(((xx + (yy * 24)) + (ff * 576)) + (nn * 11520))] = ((int)reducer6);
        }
      }
    }
  }
  __local int tanh1[11520000];
  for (int args = 0; args < 1000; ++args) {
    for (int args0 = 0; args0 < 20; ++args0) {
      for (int args1 = 0; args1 < 24; ++args1) {
        for (int args2 = 0; args2 < 24; ++args2) {
          tanh1[(((args2 + (args1 * 24)) + (args0 * 576)) + (args * 11520))] = ((int)tanh(((double)conv2d[(((args2 + (args1 * 24)) + (args0 * 576)) + (args * 11520))])));
        }
      }
    }
  }
  __local int max_pool[2880000];
  for (int i = 0; i < 1000; ++i) {
    for (int c = 0; c < 20; ++c) {
      for (int h = 0; h < 12; ++h) {
        for (int w = 0; w < 12; ++w) {
          __local float reducer7;
          reducer7 = -1.000000e+00f;
          for (int ra17 = 0; ra17 < 2; ++ra17) {
            for (int ra18 = 0; ra18 < 2; ++ra18) {
              reducer7 = max(((float)tanh1[(((((w * 2) + ra18) + (((h * 2) + ra17) * 24)) + (c * 576)) + (i * 11520))]), reducer7);
            }
          }
          max_pool[(((w + (h * 12)) + (c * 144)) + (i * 2880))] = ((int)reducer7);
        }
      }
    }
  }
  __local int conv2d1[3200000];
  for (int nn1 = 0; nn1 < 1000; ++nn1) {
    for (int ff1 = 0; ff1 < 50; ++ff1) {
      for (int yy1 = 0; yy1 < 8; ++yy1) {
        for (int xx1 = 0; xx1 < 8; ++xx1) {
          __local float reducer8;
          reducer8 = 0.000000e+00f;
          for (int ra19 = 0; ra19 < 20; ++ra19) {
            for (int ra20 = 0; ra20 < 5; ++ra20) {
              for (int ra21 = 0; ra21 < 5; ++ra21) {
                reducer8 = (((float)(((long)max_pool[((((xx1 + ra21) + ((yy1 + ra20) * 12)) + (ra19 * 144)) + (nn1 * 2880))]) * ((long)weight_conv2[(((ra21 + (ra20 * 5)) + (ra19 * 25)) + (ff1 * 500))]))) + reducer8);
              }
            }
          }
          conv2d1[(((xx1 + (yy1 * 8)) + (ff1 * 64)) + (nn1 * 3200))] = ((int)reducer8);
        }
      }
    }
  }
  __local int tanh2[3200000];
  for (int args3 = 0; args3 < 1000; ++args3) {
    for (int args01 = 0; args01 < 50; ++args01) {
      for (int args11 = 0; args11 < 8; ++args11) {
        for (int args21 = 0; args21 < 8; ++args21) {
          tanh2[(((args21 + (args11 * 8)) + (args01 * 64)) + (args3 * 3200))] = ((int)tanh(((double)conv2d1[(((args21 + (args11 * 8)) + (args01 * 64)) + (args3 * 3200))])));
        }
      }
    }
  }
  __local int max_pool1[800000];
  for (int i1 = 0; i1 < 1000; ++i1) {
    for (int c1 = 0; c1 < 50; ++c1) {
      for (int h1 = 0; h1 < 4; ++h1) {
        for (int w1 = 0; w1 < 4; ++w1) {
          __local float reducer9;
          reducer9 = -1.000000e+00f;
          for (int ra22 = 0; ra22 < 2; ++ra22) {
            for (int ra23 = 0; ra23 < 2; ++ra23) {
              reducer9 = max(((float)tanh2[(((((w1 * 2) + ra23) + (((h1 * 2) + ra22) * 8)) + (c1 * 64)) + (i1 * 3200))]), reducer9);
            }
          }
          max_pool1[(((w1 + (h1 * 4)) + (c1 * 16)) + (i1 * 800))] = ((int)reducer9);
        }
      }
    }
  }
  __local int compute3[800000];
  for (int i2 = 0; i2 < 1000; ++i2) {
    for (int j = 0; j < 800; ++j) {
      compute3[(j + (i2 * 800))] = max_pool1[((((((j / 4) % 4) * 4) + (j % 4)) + ((j / 16) * 16)) + (i2 * 800))];
    }
  }
  __local int dense[500000];
  for (int i3 = 0; i3 < 1000; ++i3) {
    for (int j1 = 0; j1 < 500; ++j1) {
      __local float reducer10;
      reducer10 = 0.000000e+00f;
      for (int ra24 = 0; ra24 < 800; ++ra24) {
        reducer10 = (((float)(((long)compute3[(ra24 + (i3 * 800))]) * ((long)weight_fc1[(ra24 + (j1 * 800))]))) + reducer10);
      }
      dense[(j1 + (i3 * 500))] = ((int)reducer10);
    }
  }
  __local int tanh3[500000];
  for (int args4 = 0; args4 < 1000; ++args4) {
    for (int args02 = 0; args02 < 500; ++args02) {
      tanh3[(args02 + (args4 * 500))] = ((int)tanh(((double)dense[(args02 + (args4 * 500))])));
    }
  }
  __local int dense1[10000];
  for (int i4 = 0; i4 < 1000; ++i4) {
    for (int j2 = 0; j2 < 10; ++j2) {
      __local float reducer11;
      reducer11 = 0.000000e+00f;
      for (int ra25 = 0; ra25 < 500; ++ra25) {
        reducer11 = (((float)(((long)tanh3[(ra25 + (i4 * 500))]) * ((long)weight_fc2[(ra25 + (j2 * 500))]))) + reducer11);
      }
      dense1[(j2 + (i4 * 10))] = ((int)reducer11);
    }
  }
  __local int compute4[1000];
  for (int i5 = 0; i5 < 1000; ++i5) {
    __local int max;
    max = 0;
    for (int ra26 = 0; ra26 < 10; ++ra26) {
      max = max(dense1[(ra26 + (i5 * 10))], max);
    }
    compute4[i5] = max;
  }
  __local int compute5[1000];
  for (int i6 = 0; i6 < 1000; ++i6) {
    __local int sum;
    sum = 0;
    for (int ra27 = 0; ra27 < 10; ++ra27) {
      sum = ((int)(exp(((double)((long)(dense1[(ra27 + (i6 * 10))] - compute4[i6])))) + ((double)sum)));
    }
    compute5[i6] = sum;
  }
  __local int update1;
  for (int i7 = 0; i7 < 1000; ++i7) {
    for (int j3 = 0; j3 < 10; ++j3) {
      lenet[(j3 + (i7 * 10))] = ((int)(exp(((double)((long)(dense1[(j3 + (i7 * 10))] - compute4[i7])))) / ((double)compute5[i7])));
    }
  }
}

