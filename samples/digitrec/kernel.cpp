#include <ap_int.h>
#include <ap_fixed.h>
#include <math.h>

void top(ap_uint<64>* arg_top_0, ap_uint<64>* train_images_stream_recv, ap_int<32>* arg_top_2, ap_uint<6>* knn_mat){
    ap_uint<6> knn_mat[10][3];
  for (ap_int<32> x = 0; x < 10; ++x) {
    for (ap_int<32> y = 0; y < 3; ++y) {
      knn_mat[x][y] = (ap_uint<6>)50;
    }
  }
  ap_int<32> knn_update;
  for (ap_int<32> y1 = 0; y1 < 1800; ++y1) {
    for (ap_int<32> x1 = 0; x1 < 10; ++x1) {
    #pragma HLS pipeline
      ap_uint<6> dist;
      ap_int<32> diff;
      diff = ((ap_int<32>)(train_images_stream_recv[x1][y1] ^ arg_top_0));
      ap_uint<6> out;
      out = (ap_uint<6>)0;
      for (ap_int<32> i = 0; i < 64; ++i) {
        out = ((ap_uint<6>)(((ap_int<34>)out) + ((ap_int<34>)diff[i])));
      }
      dist = out;
      ap_int<32> max_id;
      max_id = 0;
      for (ap_int<32> i1 = 0; i1 < 3; ++i1) {
        if (knn_mat[((max_id / 3) + x1)][(max_id % 3)] < knn_mat[x1][i1]) {
          max_id = i1;
        }
      }
      if (dist < knn_mat[((max_id / 3) + x1)][(max_id % 3)]) {
        knn_mat[((max_id / 3) + x1)][(max_id % 3)] = dist;
      }
    }
  }
  ap_int<32> sort;
  for (ap_int<32> x2 = 0; x2 < 10; ++x2) {
    for (ap_int<32> y2 = 0; y2 < 3; ++y2) {
      ap_int<32> val;
      val = 0;
      if (y2 == 1) {
        if (knn_mat[x2][2] < knn_mat[x2][1]) {
          val = ((ap_int<32>)knn_mat[x2][1]);
          knn_mat[x2][1] = knn_mat[x2][2];
          knn_mat[x2][2] = ((ap_uint<6>)val);
        }
      } else {
        if (knn_mat[x2][1] < knn_mat[x2][0]) {
          val = ((ap_int<32>)knn_mat[x2][0]);
          knn_mat[x2][0] = knn_mat[x2][1];
          knn_mat[x2][1] = ((ap_uint<6>)val);
        }
      }
    }
  }
  ap_int<32> new[10][3];
  for (ap_int<32> x3 = 0; x3 < 10; ++x3) {
    for (ap_int<32> y3 = 0; y3 < 3; ++y3) {
      new[x3][y3] = ((ap_int<32>)knn_mat[x3][y3]);
    }
  }
    }

