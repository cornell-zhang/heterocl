#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
typedef ap_int<32> bit32;
typedef ap_uint<32> ubit32;

void test(ap_fixed<32, 20> input_image[1][3][32][32], ap_fixed<32, 20> fc[1][10]);

#endif