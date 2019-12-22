__kernel void default_function(__global int* placeholder0, __global int* placeholder1, __global int* matrix_3) {
  for (int x = 0; x < 10; ++x) {
    for (int y = 0; y < 10; ++y) {
      __local int sum;
      sum = 0;
      for (int k = 0; k < 10; ++k) {
        sum = ((int)(((long)(((long)placeholder0[(k + (x * 10))]) * ((long)placeholder1[(y + (k * 10))]))) + ((long)sum)));
      }
      matrix_3[(y + (x * 10))] = sum;
    }
  }
}

