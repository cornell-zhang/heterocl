// HASH:1033564507
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

extern "C" {
void test(ap_int<32> hcl_rdv3[26][9984], ap_int<32> hcl_trainLabels[6238], ap_uint<64> hcl_in_train[6238][156], ap_uint<64> hcl_in_test[1559][156], ap_int<32> hcl_testLabels[1559], ap_int<32> hcl_epoch) {
    #pragma HLS INTERFACE m_axi port=hcl_rdv3 offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=hcl_trainLabels offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=hcl_in_train offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=hcl_in_test offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=hcl_testLabels offset=slave bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=hcl_rdv3 bundle=control
    #pragma HLS INTERFACE s_axilite port=hcl_trainLabels bundle=control
    #pragma HLS INTERFACE s_axilite port=hcl_in_train bundle=control
    #pragma HLS INTERFACE s_axilite port=hcl_in_test bundle=control
    #pragma HLS INTERFACE s_axilite port=hcl_testLabels bundle=control
    #pragma HLS INTERFACE s_axilite port=hcl_epoch bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
      ap_int<32> _top;
      ap_uint<64> pack_rdv3[26][4992];
      pack_rdv3_indices: for (ap_int<32> indices = 0; indices < 26; ++indices) {
        pack_rdv3_temp: for (ap_int<32> temp = 0; temp < 4992; ++temp) {
          ap_uint<64> pack_rdv3_temp;
          pack_rdv3_temp_x: for (ap_int<32> x = 0; x < 1; ++x) {
            pack_rdv3_temp = (ap_uint<64>)0;
          }
          i: for (ap_int<32> i = 0; i < 2; ++i) {
            pack_rdv3_temp(((i * 32) + 31), (i * 32)) = hcl_rdv3[indices][((temp * 2) + i)];
          }
          pack_rdv3[indices][temp] = pack_rdv3_temp;
        }
      }
      ap_uint<64> prototype[26][156];
      prototype_x1: for (ap_int<32> x1 = 0; x1 < 26; ++x1) {
        prototype_y: for (ap_int<32> y = 0; y < 156; ++y) {
          prototype[x1][y] = (ap_uint<64>)0;
        }
      }
      ap_int<32> prototypeCounter[26][9984];
      prototypeCounter_x2: for (ap_int<32> x2 = 0; x2 < 26; ++x2) {
        prototypeCounter_y1: for (ap_int<32> y1 = 0; y1 < 9984; ++y1) {
          prototypeCounter[x2][y1] = 0;
        }
      }
      ap_int<32> compute0[26];
      compute0_x3: for (ap_int<32> x3 = 0; x3 < 26; ++x3) {
        compute0[x3] = 0;
      }
      ap_int<32> learn;
      learn_k: for (ap_int<32> k = 0; k < 26; ++k) {
        ap_uint<64> match[6238][156];
        match_x4: for (ap_int<32> x4 = 0; x4 < 6238; ++x4) {
          match_y2: for (ap_int<32> y2 = 0; y2 < 156; ++y2) {
            match[x4][y2] = (ap_uint<64>)0;
          }
        }
        i1: for (ap_int<32> i1 = 0; i1 < 6238; ++i1) {
          if (hcl_trainLabels[i1] == k) {
            compute0[k] = (compute0[k] + 1);
            i2: for (ap_int<32> i2 = 0; i2 < 156; ++i2) {
              match[i1][i2] = hcl_in_train[i1][i2];
            }
          } else {
            i3: for (ap_int<32> i3 = 0; i3 < 156; ++i3) {
              match[i1][i3] = (ap_uint<64>)0;
            }
          }
        }
        i4: for (ap_int<32> i4 = 0; i4 < 64; ++i4) {
          ap_int<32> result[156];
          result_y3: for (ap_int<32> y3 = 0; y3 < 156; ++y3) {
            ap_int<32> sum;
            sum_x5: for (ap_int<32> x5 = 0; x5 < 1; ++x5) {
              sum = 0;
            }
            result_r: for (ap_int<32> r = 0; r < 6238; ++r) {
              sum = ((ap_int<32>)(((ap_int<66>)match[r][y3][i4]) + ((ap_int<66>)sum)));
            }
            result[y3] = sum;
          }
          ap_uint<1> sum1[156];
          sum1_x6: for (ap_int<32> x6 = 0; x6 < 156; ++x6) {
            sum1[x6] = (ap_uint<1>)0;
          }
          if ((compute0[k] % 2) == 0) {
            ap_int<32> update0;
            update0_x7: for (ap_int<32> x7 = 0; x7 < 156; ++x7) {
              sum1[x7] = ((ap_uint<1>)(((ap_int<67>)0 < (((ap_int<67>)(((ap_int<66>)result[x7]) + ((ap_int<66>)pack_rdv3[k][x7][i4]))) - ((ap_int<67>)(compute0[k] / 2)))) ? ((ap_int<32>)1) : ((ap_int<32>)0)));
            }
          } else {
            ap_int<32> update1;
            update1_x8: for (ap_int<32> x8 = 0; x8 < 156; ++x8) {
              sum1[x8] = ((ap_uint<1>)(((ap_int<33>)0 < (((ap_int<33>)result[x8]) - ((ap_int<33>)(compute0[k] / 2)))) ? ((ap_int<32>)1) : ((ap_int<32>)0)));
            }
          }
          i5: for (ap_int<32> i5 = 0; i5 < 156; ++i5) {
            prototype[k][i5][i4] = sum1[i5];
            prototypeCounter[(((((ap_int<33>)(i5 * 64)) + ((ap_int<33>)i4)) + ((ap_int<33>)(k * 9984))) / (ap_int<33>)9984)][(((((ap_int<33>)(i5 * 64)) + ((ap_int<33>)i4)) + ((ap_int<33>)(k * 9984))) % (ap_int<33>)9984)] = result[i5];
          }
        }
      }
      ap_int<32> test_train_accu;
      test_train_accu_x9: for (ap_int<32> x9 = 0; x9 < 1; ++x9) {
        ap_uint<64> distance1[156];
        distance1_x10: for (ap_int<32> x10 = 0; x10 < 156; ++x10) {
          distance1[x10] = (ap_uint<64>)0;
        }
        ap_int<32> pre_hamming[156];
        pre_hamming_x11: for (ap_int<32> x11 = 0; x11 < 156; ++x11) {
          pre_hamming[x11] = 0;
        }
        ap_int<32> hamming_dist1[26];
        hamming_dist1_x12: for (ap_int<32> x12 = 0; x12 < 26; ++x12) {
          hamming_dist1[x12] = 0;
        }
        ap_int<32> correct1;
        correct1_x13: for (ap_int<32> x13 = 0; x13 < 1; ++x13) {
          correct1 = 0;
        }
        i6: for (ap_int<32> i6 = 0; i6 < 6238; ++i6) {
          i7: for (ap_int<32> i7 = 0; i7 < 26; ++i7) {
            ap_int<32> update2;
            update2_x14: for (ap_int<32> x14 = 0; x14 < 156; ++x14) {
              distance1[x14] = (hcl_in_train[i6][x14] ^ prototype[i7][x14]);
            }
            ap_int<32> update3;
            update3_x15: for (ap_int<32> x15 = 0; x15 < 156; ++x15) {
              ap_int<32> count;
              count_x16: for (ap_int<32> x16 = 0; x16 < 1; ++x16) {
                count = 0;
              }
              ap_uint<64> numb;
              numb_x17: for (ap_int<32> x17 = 0; x17 < 1; ++x17) {
                numb = distance1[x15];
              }
              while (((ap_uint<64>)0 < numb)) {
                count = (count + 1);
                numb = (numb & (numb - (ap_uint<64>)1));
              }
              pre_hamming[x15] = count;
            }
            ap_int<32> sum_;
            sum_x18: for (ap_int<32> x18 = 0; x18 < 1; ++x18) {
              sum_ = 0;
            }
            test_train_accu_m1: for (ap_int<32> m1 = 0; m1 < 156; ++m1) {
              sum_ = ((ap_int<32>)(((ap_int<33>)pre_hamming[m1]) + ((ap_int<33>)sum_)));
            }
            hamming_dist1[i7] = sum_;
          }
          ap_int<32> pred1;
          pred1_x19: for (ap_int<32> x19 = 0; x19 < 1; ++x19) {
            pred1 = 0;
          }
          i8: for (ap_int<32> i8 = 0; i8 < 26; ++i8) {
            if (hamming_dist1[i8] < hamming_dist1[pred1]) {
              pred1 = i8;
            }
          }
          if (pred1 == hcl_trainLabels[i6]) {
            correct1 = (correct1 + 1);
          }
        }
        float all1;
        all1_x20: for (ap_int<32> x20 = 0; x20 < 1; ++x20) {
          all1 = 6.238000e+03f;
        }
        float accuracy1;
        accuracy1_x21: for (ap_int<32> x21 = 0; x21 < 1; ++x21) {
          accuracy1 = ((((float)correct1) / all1) * 1.000000e+02f);
        }
      }
      ap_int<32> test_test_accu;
      test_test_accu_x22: for (ap_int<32> x22 = 0; x22 < 1; ++x22) {
        ap_uint<64> distance1_[156];
        distance1_x23: for (ap_int<32> x23 = 0; x23 < 156; ++x23) {
          distance1_[x23] = (ap_uint<64>)0;
        }
        ap_int<32> pre_hamming_[156];
        pre_hamming_x24: for (ap_int<32> x24 = 0; x24 < 156; ++x24) {
          pre_hamming_[x24] = 0;
        }
        ap_int<32> hamming_dist1_[26];
        hamming_dist1_x25: for (ap_int<32> x25 = 0; x25 < 26; ++x25) {
          hamming_dist1_[x25] = 0;
        }
        ap_int<32> correct1_;
        correct1_x26: for (ap_int<32> x26 = 0; x26 < 1; ++x26) {
          correct1_ = 0;
        }
        i9: for (ap_int<32> i9 = 0; i9 < 1559; ++i9) {
          i10: for (ap_int<32> i10 = 0; i10 < 26; ++i10) {
            ap_int<32> update4;
            update4_x27: for (ap_int<32> x27 = 0; x27 < 156; ++x27) {
              distance1_[x27] = (hcl_in_test[i9][x27] ^ prototype[i10][x27]);
            }
            ap_int<32> update5;
            update5_x28: for (ap_int<32> x28 = 0; x28 < 156; ++x28) {
              ap_int<32> count_;
              count_x29: for (ap_int<32> x29 = 0; x29 < 1; ++x29) {
                count_ = 0;
              }
              ap_uint<64> numb_;
              numb_x30: for (ap_int<32> x30 = 0; x30 < 1; ++x30) {
                numb_ = distance1_[x28];
              }
              while (((ap_uint<64>)0 < numb_)) {
                count_ = (count_ + 1);
                numb_ = (numb_ & (numb_ - (ap_uint<64>)1));
              }
              pre_hamming_[x28] = count_;
            }
            ap_int<32> sum__;
            sum_x31: for (ap_int<32> x31 = 0; x31 < 1; ++x31) {
              sum__ = 0;
            }
            test_test_accu_m11: for (ap_int<32> m11 = 0; m11 < 156; ++m11) {
              sum__ = ((ap_int<32>)(((ap_int<33>)pre_hamming_[m11]) + ((ap_int<33>)sum__)));
            }
            hamming_dist1_[i10] = sum__;
          }
          ap_int<32> pred1_;
          pred1_x32: for (ap_int<32> x32 = 0; x32 < 1; ++x32) {
            pred1_ = 0;
          }
          i11: for (ap_int<32> i11 = 0; i11 < 26; ++i11) {
            if (hamming_dist1_[i11] < hamming_dist1_[pred1_]) {
              pred1_ = i11;
            }
          }
          if (pred1_ == hcl_testLabels[i9]) {
            correct1_ = (correct1_ + 1);
          }
        }
        float all1_;
        all1_x33: for (ap_int<32> x33 = 0; x33 < 1; ++x33) {
          all1_ = 1.559000e+03f;
        }
        float accuracy1_;
        accuracy1_x34: for (ap_int<32> x34 = 0; x34 < 1; ++x34) {
          accuracy1_ = ((((float)correct1_) / all1_) * 1.000000e+02f);
        }
      }
      ap_int<32> update;
      update_x35: for (ap_int<32> x35 = 0; x35 < hcl_epoch[0]; ++x35) {
        ap_uint<64> distance[156];
        distance_x36: for (ap_int<32> x36 = 0; x36 < 156; ++x36) {
          distance[x36] = (ap_uint<64>)0;
        }
        ap_int<32> pre_dist[156];
        pre_dist_x37: for (ap_int<32> x37 = 0; x37 < 156; ++x37) {
          pre_dist[x37] = 0;
        }
        ap_int<32> hamming_dist[26];
        hamming_dist_x38: for (ap_int<32> x38 = 0; x38 < 26; ++x38) {
          hamming_dist[x38] = 0;
        }
        i12: for (ap_int<32> i12 = 0; i12 < 6238; ++i12) {
          i13: for (ap_int<32> i13 = 0; i13 < 26; ++i13) {
            ap_int<32> update6;
            update6_x39: for (ap_int<32> x39 = 0; x39 < 156; ++x39) {
              distance[x39] = (hcl_in_train[i12][x39] ^ prototype[i13][x39]);
            }
            ap_int<32> update7;
            update7_x40: for (ap_int<32> x40 = 0; x40 < 156; ++x40) {
              ap_int<32> count__;
              count_x41: for (ap_int<32> x41 = 0; x41 < 1; ++x41) {
                count__ = 0;
              }
              ap_uint<64> numb__;
              numb_x42: for (ap_int<32> x42 = 0; x42 < 1; ++x42) {
                numb__ = distance[x40];
              }
              while (((ap_uint<64>)0 < numb__)) {
                count__ = (count__ + 1);
                numb__ = (numb__ & (numb__ - (ap_uint<64>)1));
              }
              pre_dist[x40] = count__;
            }
            ap_int<32> sum___;
            sum_x43: for (ap_int<32> x43 = 0; x43 < 1; ++x43) {
              sum___ = 0;
            }
            update_m: for (ap_int<32> m = 0; m < 156; ++m) {
              sum___ = ((ap_int<32>)(((ap_int<33>)pre_dist[m]) + ((ap_int<33>)sum___)));
            }
            hamming_dist[i13] = sum___;
          }
          ap_int<32> pred;
          pred_x44: for (ap_int<32> x44 = 0; x44 < 1; ++x44) {
            pred = 0;
          }
          i14: for (ap_int<32> i14 = 0; i14 < 26; ++i14) {
            if (hamming_dist[i14] < hamming_dist[pred]) {
              pred = i14;
            }
          }
          if (pred != hcl_trainLabels[i12]) {
            compute0[hcl_trainLabels[i12]] = (compute0[hcl_trainLabels[i12]] + 1);
            compute0[pred] = (compute0[pred] + -1);
            i15: for (ap_int<32> i15 = 0; i15 < 156; ++i15) {
              i16: for (ap_int<32> i16 = 0; i16 < 64; ++i16) {
                prototypeCounter[(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(hcl_trainLabels[i12] * 9984))) / (ap_int<33>)9984)][(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(hcl_trainLabels[i12] * 9984))) % (ap_int<33>)9984)] = ((ap_int<32>)(((ap_int<66>)prototypeCounter[(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(hcl_trainLabels[i12] * 9984))) / (ap_int<33>)9984)][(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(hcl_trainLabels[i12] * 9984))) % (ap_int<33>)9984)]) + ((ap_int<66>)hcl_in_train[i12][i15][i16])));
                prototypeCounter[(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(pred * 9984))) / (ap_int<33>)9984)][(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(pred * 9984))) % (ap_int<33>)9984)] = ((ap_int<32>)(((ap_int<66>)prototypeCounter[(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(pred * 9984))) / (ap_int<33>)9984)][(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(pred * 9984))) % (ap_int<33>)9984)]) - ((ap_int<66>)hcl_in_train[i12][i15][i16])));
                if ((compute0[hcl_trainLabels[i12]] % 2) == 0) {
                  if (((ap_int<33>)prototypeCounter[(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(hcl_trainLabels[i12] * 9984))) / (ap_int<33>)9984)][(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(hcl_trainLabels[i12] * 9984))) % (ap_int<33>)9984)]) == ((ap_int<33>)(compute0[hcl_trainLabels[i12]] / 2))) {
                    prototype[hcl_trainLabels[i12]][i15][i16] = (prototype[hcl_trainLabels[i12]][i15][i16] % (ap_uint<64>)2);
                  }
                } else {
                  prototype[hcl_trainLabels[i12]][i15][i16] = (((ap_int<33>)0 < (((ap_int<33>)prototypeCounter[(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(hcl_trainLabels[i12] * 9984))) / (ap_int<33>)9984)][(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(hcl_trainLabels[i12] * 9984))) % (ap_int<33>)9984)]) - ((ap_int<33>)(compute0[hcl_trainLabels[i12]] / 2)))) ? ((ap_int<32>)1) : ((ap_int<32>)0));
                }
                if ((compute0[pred] % 2) == 0) {
                  if (((ap_int<33>)prototypeCounter[(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(pred * 9984))) / (ap_int<33>)9984)][(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(pred * 9984))) % (ap_int<33>)9984)]) == ((ap_int<33>)(compute0[pred] / 2))) {
                    prototype[pred][i15][i16] = (prototype[pred][i15][i16] % (ap_uint<64>)2);
                  }
                } else {
                  prototype[pred][i15][i16] = (((ap_int<33>)0 < (((ap_int<33>)prototypeCounter[(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(pred * 9984))) / (ap_int<33>)9984)][(((((ap_int<33>)(i15 * 64)) + ((ap_int<33>)i16)) + ((ap_int<33>)(pred * 9984))) % (ap_int<33>)9984)]) - ((ap_int<33>)(compute0[pred] / 2)))) ? ((ap_int<32>)1) : ((ap_int<32>)0));
                }
              }
            }
          }
        }
        ap_int<32> training_update;
        training_update_x45: for (ap_int<32> x45 = 0; x45 < 1; ++x45) {
          ap_uint<64> distance1__[156];
          distance1_x46: for (ap_int<32> x46 = 0; x46 < 156; ++x46) {
            distance1__[x46] = (ap_uint<64>)0;
          }
          ap_int<32> pre_hamming__[156];
          pre_hamming_x47: for (ap_int<32> x47 = 0; x47 < 156; ++x47) {
            pre_hamming__[x47] = 0;
          }
          ap_int<32> hamming_dist1__[26];
          hamming_dist1_x48: for (ap_int<32> x48 = 0; x48 < 26; ++x48) {
            hamming_dist1__[x48] = 0;
          }
          ap_int<32> correct1__;
          correct1_x49: for (ap_int<32> x49 = 0; x49 < 1; ++x49) {
            correct1__ = 0;
          }
          i17: for (ap_int<32> i17 = 0; i17 < 6238; ++i17) {
            i18: for (ap_int<32> i18 = 0; i18 < 26; ++i18) {
              ap_int<32> update8;
              update8_x50: for (ap_int<32> x50 = 0; x50 < 156; ++x50) {
                distance1__[x50] = (hcl_in_train[i17][x50] ^ prototype[i18][x50]);
              }
              ap_int<32> update9;
              update9_x51: for (ap_int<32> x51 = 0; x51 < 156; ++x51) {
                ap_int<32> count___;
                count_x52: for (ap_int<32> x52 = 0; x52 < 1; ++x52) {
                  count___ = 0;
                }
                ap_uint<64> numb___;
                numb_x53: for (ap_int<32> x53 = 0; x53 < 1; ++x53) {
                  numb___ = distance1__[x51];
                }
                while (((ap_uint<64>)0 < numb___)) {
                  count___ = (count___ + 1);
                  numb___ = (numb___ & (numb___ - (ap_uint<64>)1));
                }
                pre_hamming__[x51] = count___;
              }
              ap_int<32> sum____;
              sum_x54: for (ap_int<32> x54 = 0; x54 < 1; ++x54) {
                sum____ = 0;
              }
              training_update_m12: for (ap_int<32> m12 = 0; m12 < 156; ++m12) {
                sum____ = ((ap_int<32>)(((ap_int<33>)pre_hamming__[m12]) + ((ap_int<33>)sum____)));
              }
              hamming_dist1__[i18] = sum____;
            }
            ap_int<32> pred1__;
            pred1_x55: for (ap_int<32> x55 = 0; x55 < 1; ++x55) {
              pred1__ = 0;
            }
            i19: for (ap_int<32> i19 = 0; i19 < 26; ++i19) {
              if (hamming_dist1__[i19] < hamming_dist1__[pred1__]) {
                pred1__ = i19;
              }
            }
            if (pred1__ == hcl_trainLabels[i17]) {
              correct1__ = (correct1__ + 1);
            }
          }
          float all1__;
          all1_x56: for (ap_int<32> x56 = 0; x56 < 1; ++x56) {
            all1__ = 6.238000e+03f;
          }
          float accuracy1__;
          accuracy1_x57: for (ap_int<32> x57 = 0; x57 < 1; ++x57) {
            accuracy1__ = ((((float)correct1__) / all1__) * 1.000000e+02f);
          }
        }
        ap_int<32> testing_update;
        testing_update_x58: for (ap_int<32> x58 = 0; x58 < 1; ++x58) {
          ap_uint<64> distance1___[156];
          distance1_x59: for (ap_int<32> x59 = 0; x59 < 156; ++x59) {
            distance1___[x59] = (ap_uint<64>)0;
          }
          ap_int<32> pre_hamming___[156];
          pre_hamming_x60: for (ap_int<32> x60 = 0; x60 < 156; ++x60) {
            pre_hamming___[x60] = 0;
          }
          ap_int<32> hamming_dist1___[26];
          hamming_dist1_x61: for (ap_int<32> x61 = 0; x61 < 26; ++x61) {
            hamming_dist1___[x61] = 0;
          }
          ap_int<32> correct1___;
          correct1_x62: for (ap_int<32> x62 = 0; x62 < 1; ++x62) {
            correct1___ = 0;
          }
          i20: for (ap_int<32> i20 = 0; i20 < 1559; ++i20) {
            i21: for (ap_int<32> i21 = 0; i21 < 26; ++i21) {
              ap_int<32> update10;
              update10_x63: for (ap_int<32> x63 = 0; x63 < 156; ++x63) {
                distance1___[x63] = (hcl_in_test[i20][x63] ^ prototype[i21][x63]);
              }
              ap_int<32> update11;
              update11_x64: for (ap_int<32> x64 = 0; x64 < 156; ++x64) {
                ap_int<32> count____;
                count_x65: for (ap_int<32> x65 = 0; x65 < 1; ++x65) {
                  count____ = 0;
                }
                ap_uint<64> numb____;
                numb_x66: for (ap_int<32> x66 = 0; x66 < 1; ++x66) {
                  numb____ = distance1___[x64];
                }
                while (((ap_uint<64>)0 < numb____)) {
                  count____ = (count____ + 1);
                  numb____ = (numb____ & (numb____ - (ap_uint<64>)1));
                }
                pre_hamming___[x64] = count____;
              }
              ap_int<32> sum_____;
              sum_x67: for (ap_int<32> x67 = 0; x67 < 1; ++x67) {
                sum_____ = 0;
              }
              testing_update_m13: for (ap_int<32> m13 = 0; m13 < 156; ++m13) {
                sum_____ = ((ap_int<32>)(((ap_int<33>)pre_hamming___[m13]) + ((ap_int<33>)sum_____)));
              }
              hamming_dist1___[i21] = sum_____;
            }
            ap_int<32> pred1___;
            pred1_x68: for (ap_int<32> x68 = 0; x68 < 1; ++x68) {
              pred1___ = 0;
            }
            i22: for (ap_int<32> i22 = 0; i22 < 26; ++i22) {
              if (hamming_dist1___[i22] < hamming_dist1___[pred1___]) {
                pred1___ = i22;
              }
            }
            if (pred1___ == hcl_testLabels[i20]) {
              correct1___ = (correct1___ + 1);
            }
          }
          float all1___;
          all1_x69: for (ap_int<32> x69 = 0; x69 < 1; ++x69) {
            all1___ = 1.559000e+03f;
          }
          float accuracy1___;
          accuracy1_x70: for (ap_int<32> x70 = 0; x70 < 1; ++x70) {
            accuracy1___ = ((((float)correct1___) / all1___) * 1.000000e+02f);
          }
        }
      }
    }
}

