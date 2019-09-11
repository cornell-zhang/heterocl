#include "ihc_apint.h"
__kernel void default_function(__global int* A, __global int* B, __global int* C) {
  for (int x = 0; x < 10; ++x) {
    C[x] = ((int)(((int33_t)A[x]) + ((int33_t)B[x])));
  }
}