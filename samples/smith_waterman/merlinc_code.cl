#include <string.h>
#include <math.h>
#include <assert.h>
#pragma ACCEL kernel
void default_function(unsigned char* seqAs, unsigned char* seqBs, unsigned char* outAs, unsigned char* outBs) {
  int B;
#pragma ACCEL pipeline
  for (int t_outer = 0; t_outer < 32; ++t_outer) {
#pragma ACCEL parallel
    for (int t_inner = 0; t_inner < 32; ++t_inner) {
      int maxtrix_max;
      maxtrix_max = 0;
      int i_max;
      i_max = 0;
      int j_max;
      j_max = 0;
      short matrix[16641];
      for (int x = 0; x < 129; ++x) {
        for (int y = 0; y < 129; ++y) {
          matrix[(y + (x * 129))] = (short)0;
        }
      }
      short action[16641];
      for (int x1 = 0; x1 < 129; ++x1) {
        for (int y1 = 0; y1 < 129; ++y1) {
          action[(y1 + (x1 * 129))] = (short)3;
        }
      }
      int mutate3;
      for (int i = 0; i < 129; ++i) {
        for (int j = 0; j < 129; ++j) {
          int trace_back[4];
          for (int x2 = 0; x2 < 4; ++x2) {
            trace_back[x2] = 0;
          }
          if ((i != 0) && (j != 0)) {
            trace_back[0] = ((int)(((long)matrix[((j + (i * 129)) + -130)]) + ((long)((seqAs[((i + ((t_inner + (t_outer * 32)) * 128)) + -1)] == seqBs[((j + ((t_inner + (t_outer * 32)) * 128)) + -1)]) ? 1 : -4))));
            trace_back[1] = (((int)matrix[((j + (i * 129)) + -129)]) + -4);
            trace_back[2] = (((int)matrix[((j + (i * 129)) + -1)]) + -4);
            trace_back[3] = 0;
            int max;
            max = trace_back[0];
            int act;
            act = 0;
            for (int i1 = 0; i1 < 4; ++i1) {
              if (max < trace_back[i1]) {
                max = trace_back[i1];
                act = i1;
              }
            }
            matrix[(j + (i * 129))] = ((short)max);
            action[(j + (i * 129))] = ((short)act);
            if (maxtrix_max < ((int)matrix[(j + (i * 129))])) {
              maxtrix_max = ((int)matrix[(j + (i * 129))]);
              i_max = i;
              j_max = j;
            }
          }
        }
      }
      int T;
      int curr_i;
      curr_i = i_max;
      int curr_j;
      curr_j = j_max;
      int next_i;
      next_i = 0;
      int next_j;
      next_j = 0;
      int act1;
      act1 = ((int)action[(curr_j + (curr_i * 129))]);
      int next_i1;
      next_i1 = 0;
      int next_j1;
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
      int tick;
      tick = 0;
      while (((curr_i != next_i) || (curr_j != next_j))) {
        int a;
        a = 0;
        int b;
        b = 0;
        if (next_i == curr_i) {
          a = 0;
        } else {
          a = ((int)seqAs[((curr_i + ((t_inner + (t_outer * 32)) * 128)) + -1)]);
        }
        if (next_j == curr_j) {
          b = 0;
        } else {
          b = ((int)seqBs[((curr_j + ((t_inner + (t_outer * 32)) * 128)) + -1)]);
        }
        outAs[(tick + ((t_inner + (t_outer * 32)) * 256))] = ((unsigned char)a);
        outBs[(tick + ((t_inner + (t_outer * 32)) * 256))] = ((unsigned char)b);
        curr_i = next_i;
        curr_j = next_j;
        int act2;
        act2 = ((int)action[(curr_j + (curr_i * 129))]);
        int next_i2;
        next_i2 = 0;
        int next_j2;
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

