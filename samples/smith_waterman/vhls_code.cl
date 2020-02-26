#include <ap_int.h>
#include <ap_fixed.h>
#include <math.h>

void default_function(ap_uint<3> seqAs[1024][128], ap_uint<3> seqBs[1024][128], ap_uint<3> outAs[1024][256], ap_uint<3> outBs[1024][256]) {
  ap_int<32> B;
  for (ap_int<32> t_outer = 0; t_outer < 32; ++t_outer) {
  #pragma HLS pipeline
    for (ap_int<32> t_inner = 0; t_inner < 32; ++t_inner) {
    #pragma HLS unroll
      ap_int<32> maxtrix_max;
      maxtrix_max = 0;
      ap_int<32> i_max;
      i_max = 0;
      ap_int<32> j_max;
      j_max = 0;
      ap_int<16> matrix[129][129];
      for (ap_int<32> x = 0; x < 129; ++x) {
        for (ap_int<32> y = 0; y < 129; ++y) {
          matrix[x][y] = (ap_int<16>)0;
        }
      }
      ap_int<16> action[129][129];
      for (ap_int<32> x1 = 0; x1 < 129; ++x1) {
        for (ap_int<32> y1 = 0; y1 < 129; ++y1) {
          action[x1][y1] = (ap_int<16>)3;
        }
      }
      ap_int<32> mutate3;
      for (ap_int<32> i = 0; i < 129; ++i) {
        for (ap_int<32> j = 0; j < 129; ++j) {
          ap_int<32> trace_back[4];
          for (ap_int<32> x2 = 0; x2 < 4; ++x2) {
            trace_back[x2] = 0;
          }
          if ((i != 0) && (j != 0)) {
            trace_back[0] = ((ap_int<32>)(((ap_int<33>)matrix[(i + -1)][(j + -1)]) + ((ap_int<33>)((seqAs[(t_inner + (t_outer * 32))][(i + -1)] == seqBs[(t_inner + (t_outer * 32))][(j + -1)]) ? 1 : -4))));
            trace_back[1] = (((ap_int<32>)matrix[(i + -1)][j]) + -4);
            trace_back[2] = (((ap_int<32>)matrix[i][(j + -1)]) + -4);
            trace_back[3] = 0;
            ap_int<32> max;
            max = trace_back[0];
            ap_int<32> act;
            act = 0;
            for (ap_int<32> i1 = 0; i1 < 4; ++i1) {
              if (max < trace_back[i1]) {
                max = trace_back[i1];
                act = i1;
              }
            }
            matrix[i][j] = ((ap_int<16>)max);
            action[i][j] = ((ap_int<16>)act);
            if (maxtrix_max < ((ap_int<32>)matrix[i][j])) {
              maxtrix_max = ((ap_int<32>)matrix[i][j]);
              i_max = i;
              j_max = j;
            }
          }
        }
      }
      ap_int<32> T;
      ap_int<32> curr_i;
      curr_i = i_max;
      ap_int<32> curr_j;
      curr_j = j_max;
      ap_int<32> next_i;
      next_i = 0;
      ap_int<32> next_j;
      next_j = 0;
      ap_int<32> act1;
      act1 = ((ap_int<32>)action[((curr_j / 129) + curr_i)][(curr_j % 129)]);
      ap_int<32> next_i1;
      next_i1 = 0;
      ap_int<32> next_j1;
      next_j1 = 0;
      if (act1 == 0) {
        next_i1 = (curr_i + -1);
        next_j1 = (curr_j + -1);
      } else {
        if (act1 == 1) {
          next_i1 = (curr_i + -1);
          next_j1 = curr_j;
        } else {
          if (act1 == 2) {
            next_i1 = curr_i;
            next_j1 = (curr_j + -1);
          } else {
            next_i1 = curr_i;
            next_j1 = curr_j;
          }
        }
      }
      next_i = next_i1;
      next_j = next_j1;
      ap_int<32> tick;
      tick = 0;
      while (((curr_i != next_i) || (curr_j != next_j))) {
        ap_int<32> a;
        a = 0;
        ap_int<32> b;
        b = 0;
        if (next_i == curr_i) {
          a = 0;
        } else {
          a = ((ap_int<32>)seqAs[((((curr_i - ((curr_i + -1) % 128)) + ((t_inner + (t_outer * 32)) * 128)) + -1) / 128)][((curr_i + -1) % 128)]);
        }
        if (next_j == curr_j) {
          b = 0;
        } else {
          b = ((ap_int<32>)seqBs[((((curr_j - ((curr_j + -1) % 128)) + ((t_inner + (t_outer * 32)) * 128)) + -1) / 128)][((curr_j + -1) % 128)]);
        }
        outAs[((tick / 256) + (t_inner + (t_outer * 32)))][(tick % 256)] = ((ap_uint<3>)a);
        outBs[((tick / 256) + (t_inner + (t_outer * 32)))][(tick % 256)] = ((ap_uint<3>)b);
        curr_i = next_i;
        curr_j = next_j;
        ap_int<32> act2;
        act2 = ((ap_int<32>)action[((curr_j / 129) + curr_i)][(curr_j % 129)]);
        ap_int<32> next_i2;
        next_i2 = 0;
        ap_int<32> next_j2;
        next_j2 = 0;
        if (act2 == 0) {
          next_i2 = (curr_i + -1);
          next_j2 = (curr_j + -1);
        } else {
          if (act2 == 1) {
            next_i2 = (curr_i + -1);
            next_j2 = curr_j;
          } else {
            if (act2 == 2) {
              next_i2 = curr_i;
              next_j2 = (curr_j + -1);
            } else {
              next_i2 = curr_i;
              next_j2 = curr_j;
            }
          }
        }
        next_i = next_i2;
        next_j = next_j2;
        tick = (tick + 1);
      }
    }
  }
}

