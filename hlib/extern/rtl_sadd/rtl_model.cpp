// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
//--------------------------------------------------------
#include "ap_int.h"
void rtl_model(ap_int<32> a1, ap_int<32> b1, ap_int<32> &z1)
{
#pragma HLS inline off
  z1=a1+b1;
}
