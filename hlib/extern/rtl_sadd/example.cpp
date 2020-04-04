// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
// #include "example.h"
#include "ap_int.h"

typedef ap_int<32> data_t;
//--------------------------------------------------------
void rtl_model(data_t a1, data_t b1, data_t &z1);
//--------------------------------------------------------
void example(data_t a1, data_t b1, data_t &sigma) {

  data_t tmp;
  rtl_model(a1, b1, tmp);
  sigma = tmp + 1;

}
//--------------------------------------------------------
