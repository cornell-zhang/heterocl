#include <stdio.h>
int main(int argc, char **argv) {

      int X[32];
      int W[3];
      int Y[30];
#pragma scop
      for (int x = 0; x < 30; ++x) {
        int sum;
        sum = 0;
        for (int k = 0; k < 3; ++k) {
          sum = ((X[(x + k)] * W[k]) + sum);
        }
        Y[x] = sum;
      }
#pragma endscop
      printf("%d", X[0]);
      printf("%d", W[0]);
      printf("%d", Y[0]);
}