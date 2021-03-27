#include <stdio.h>
int main(int argc, char **argv) {

      float Y0[1024][1024];
      float A[1024][1024];
      float B[1024][1024];
#pragma scop
      for (int i = 0; i < 1024; ++i) {
        for (int j = 0; j < 1024; ++j) {
          Y0[i][j] = 0.000000e+00f;
          for (int k = 0; k < 1024; ++k) {
            Y0[i][j] = (Y0[i][j] + (A[i][k] * B[j][k]));
          }
        }
      }
#pragma endscop
      printf("%f", Y0[0][0]);
      printf("%f", A[0][0]);
      printf("%f", B[0][0]);
}