int main(int argc, char **argv) {

      int Y0[64][64];
      int W[64][64];
      int X[64][64];
#pragma scop
      for (int i = 0; i < 64; ++i) {
        for (int i1 = 0; i1 < 64; ++i1) {
          Y0[i][i1] = 0;
          for (int i2 = 0; i2 < 64; ++i2) {
            Y0[i][i1] = (Y0[i][i1] + (W[i][i2] * X[i2][i1]));
          }
        }
      }
#pragma endscop
}