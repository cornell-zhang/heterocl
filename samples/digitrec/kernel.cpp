#include <string.h>
#include <math.h>
#include <assert.h>
#pragma ACCEL kernel
void default_function(unsigned long test_image, unsigned long* train_images, unsigned char* knn_mat) {
  for (int x = 0; x < 10; ++x) {
    for (int y = 0; y < 3; ++y) {
      knn_mat[(y + (x * 3))] = (unsigned char)50;
    }
  }
  unsigned long knn_update;
#pragma ACCEL parallel
  for (int y1 = 0; y1 < 1800; ++y1) {
#pragma ACCEL pipeline
    for (int x1 = 0; x1 < 10; ++x1) {
      unsigned char dist;
      unsigned long diff;
      diff = (train_images[(y1 + (x1 * 1800))] ^ test_image);
      unsigned char out;
      out = (unsigned char)0;
      for (int i = 0; i < 49; ++i) {
        out = ((unsigned char)(((unsigned long)out) + ((unsigned long)((diff & (1L << i)) >> i))));
      }
      dist = out;
      unsigned long max_id;
      max_id = (unsigned long)0;
      for (int i1 = 0; i1 < 3; ++i1) {
        if (knn_mat[(((long)max_id) + ((long)(x1 * 3)))] < knn_mat[(i1 + (x1 * 3))]) {
          max_id = ((unsigned long)i1);
        }
      }
      if (dist < knn_mat[(((long)max_id) + ((long)(x1 * 3)))]) {
        knn_mat[(((long)max_id) + ((long)(x1 * 3)))] = dist;
      }
    }
  }
}

