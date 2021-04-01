#include <iostream>

#include "ap_int.h"
#include "ap_fixed.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_LAYER_2 64
#define N_LAYER_5 32
#define N_LAYER_8 32
#define N_LAYER_11 5

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<16,6> result_t;

// #include "myproject.h"
#include "parameters.h"

void myproject(
    input_t input1[N_INPUT_1_1],
    result_t layer13_out[N_LAYER_11],
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input1,layer13_out 
    #pragma HLS PIPELINE 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 1024>(w2, "w2.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b2, "b2.txt");
        nnet::load_weights_from_txt<model_default_t, 2048>(w5, "w5.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b5, "b5.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(w8, "w8.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b8, "b8.txt");
        nnet::load_weights_from_txt<model_default_t, 160>(w11, "w11.txt");
        nnet::load_weights_from_txt<model_default_t, 5>(b11, "b11.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense_latency<input_t, layer2_t, config2>(input1, layer2_out, w2, b2);

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out);

    layer5_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::dense_latency<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5);

    layer7_t layer7_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer5_t, layer7_t, relu_config7>(layer5_out, layer7_out);

    layer8_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense_latency<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8);

    layer10_t layer10_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::relu<layer8_t, layer10_t, relu_config10>(layer8_out, layer10_out);

    layer11_t layer11_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense_latency<layer10_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11);

    nnet::softmax<layer11_t, result_t, softmax_config13>(layer11_out, layer13_out);

}
