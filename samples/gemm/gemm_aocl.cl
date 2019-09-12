#include "ihc_apint.h"
__kernel void default_function(__global int* restrict placeholder0, __global int* restrict placeholder1, __global int* restrict matrix_3) {
  for (int x = 0; x < 10; ++x) {
    for (int y = 0; y < 10; ++y) {
      global int sum;
      sum = 0;
      for (int k = 0; k < 10; ++k) {
        sum = ((int)(((int65_t)(((long)placeholder0[(k + (x * 10))]) * ((long)placeholder1[(y + (k * 10))]))) + ((int65_t)sum)));
      }
      matrix_3[(y + (x * 10))] = sum;
    }
  }
}

