__kernel void default_function(__global unsigned char* seqAs, __global unsigned char* seqBs, __global unsigned char* outAs, __global unsigned char* outBs) {
  __local int B;
  __attribute__((xcl_pipeline_loop(1)))
  for (int t_outer = 0; t_outer < 2; ++t_outer) {
    
    for (int t_inner = 0; t_inner < 32; ++t_inner) {
      __local int maxtrix_max;
      maxtrix_max = 0;
      __local int i_max;
      i_max = 0;
      __local int j_max;
      j_max = 0;
      __local short matrix[841];
      for (int x = 0; x < 29; ++x) {
        for (int y = 0; y < 29; ++y) {
          matrix[(y + (x * 29))] = (short)0;
        }
      }
      __local short action[841];
      for (int x1 = 0; x1 < 29; ++x1) {
        for (int y1 = 0; y1 < 29; ++y1) {
          action[(y1 + (x1 * 29))] = (short)3;
        }
      }
      __local int mutate1;
      for (int i = 0; i < 29; ++i) {
        for (int j = 0; j < 29; ++j) {
          __local int trace_back[4];
          for (int x2 = 0; x2 < 4; ++x2) {
            trace_back[x2] = 0;
          }
          if ((i != 0) && (j != 0)) {
            trace_back[0] = ((int)(((long)matrix[((j + (i * 29)) + -30)]) + ((long)(int)((seqAs[((i + ((t_inner + (t_outer * 32)) * 28)) + -1)] == seqBs[((j + ((t_inner + (t_outer * 32)) * 28)) + -1)]) ? 1 : -4))));
            trace_back[1] = (((int)matrix[((j + (i * 29)) + -29)]) + -4);
            trace_back[2] = (((int)matrix[((j + (i * 29)) + -1)]) + -4);
            trace_back[3] = 0;
            __local int max;
            max = trace_back[0];
            __local int act;
            act = 0;
            for (int i1 = 0; i1 < 4; ++i1) {
              if (max < trace_back[i1]) {
                max = trace_back[i1];
                act = i1;
              }
            }
            matrix[(j + (i * 29))] = ((short)max);
            action[(j + (i * 29))] = ((short)act);
            if (maxtrix_max < ((int)matrix[(j + (i * 29))])) {
              maxtrix_max = ((int)matrix[(j + (i * 29))]);
              i_max = i;
              j_max = j;
            }
          }
        }
      }
      __local int T;
      __local int curr_i;
      curr_i = i_max;
      __local int curr_j;
      curr_j = j_max;
      __local int next_i;
      next_i = 0;
      __local int next_j;
      next_j = 0;
      __local int act1;
      act1 = ((int)action[(curr_j + (curr_i * 29))]);
      __local int next_i1;
      next_i1 = 0;
      __local int next_j1;
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
      __local int tick;
      tick = 0;
      while (((curr_i != next_i) || (curr_j != next_j))) {
        __local int a;
        a = 0;
        __local int b;
        b = 0;
        if (next_i == curr_i) {
          a = 0;
        } else {
          a = ((int)seqAs[((curr_i + ((t_inner + (t_outer * 32)) * 28)) + -1)]);
        }
        if (next_j == curr_j) {
          b = 0;
        } else {
          b = ((int)seqBs[((curr_j + ((t_inner + (t_outer * 32)) * 28)) + -1)]);
        }
        outAs[(tick + ((t_inner + (t_outer * 32)) * 56))] = ((unsigned char)a);
        outBs[(tick + ((t_inner + (t_outer * 32)) * 56))] = ((unsigned char)b);
        curr_i = next_i;
        curr_j = next_j;
        __local int act2;
        act2 = ((int)action[(curr_j + (curr_i * 29))]);
        __local int next_i2;
        next_i2 = 0;
        __local int next_j2;
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

