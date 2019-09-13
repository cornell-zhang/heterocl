
#include "ihc_apint.h"
__kernel void default_function(__global uint* restrict test_image, __global uint* restrict train_images, __global uint* restrict knn_mat) {
  for (int x = 0; x < 10; ++x) {
    for (int y = 0; y < 3; ++y) {
      knn_mat[(y + (x * 3))] = (uint2_t)2;
    }
  }
  uint4_t knn_update;
  #pragma unroll
  for (int y1 = 0; y1 < 20; ++y1) {
    for (int x1 = 0; x1 < 10; ++x1) {
      uint2_t dist;
      uint4_t diff;
      diff = ((uint4_t)(train_images[(y1 + (x1 * 20))]) ^ (uint4_t)(test_image));
      uint2_t out;
      out = (uint2_t)0;
      for (int i = 0; i < 4; ++i) {
        out = ((uint2_t)(((uint5_t)out) + ((uint5_t)((diff & (1L << i)) >> i))));
      }
      dist = out;
      uint4_t max_id;
      max_id = (uint4_t)0;
      for (int i1 = 0; i1 < 3; ++i1) {
        if (knn_mat[(((int)max_id) + (x1 * 3))] < knn_mat[(i1 + (x1 * 3))]) {
          max_id = ((uint4_t)i1);
        }
      }
      if (dist < knn_mat[(((int)max_id) + (x1 * 3))]) {
        knn_mat[(((int)max_id) + (x1 * 3))] = dist;
      }
    }
  }
}

