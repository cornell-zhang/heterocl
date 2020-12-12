/*
Function list:

weight_load_depth_conv_weight_write
weight_load_conv_weight_write
weight_load_depth_norm_write
weight_load

*/


/*
 * Function name: weight_load_depth_conv_weight_write
 * Function description: This function writes depth_conv weights to depth_conv module.
 */
void weight_load_depth_conv_weight_write(
		bus_t1 weight_burst_buf1[],
		hls::stream<WeightLoadData0Type> &fifo_depth_conv_weight,
		ConfigInst inst0,
		ConfigInst inst1,
		ConfigInst inst2,
		ConfigInst inst3,
		uint in_num_iter,
		uint out_num_iter
){
	// Refer to cin_load module to understand the meaning of the instructions
	// inst0
	ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
	ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
	ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
	ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
	ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
	ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
	// inst1
	ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
	ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
	ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
	ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
	ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
	ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
	// inst2
	ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
	ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
	ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
	ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
	ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
	ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
	ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
	// inst3
	ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
	ap_uint<32> PREV_CIN_OFFSET  = inst3(32*1+31, 32*1);
	ap_uint<16> LAYER_IN_NUM_T   = inst3(32*2+15, 32*2);
	ap_uint<16> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2+16);
	ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
	ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);

	ap_uint<1>  CONV_1ST_EN    = LAYER_EN[0];
	ap_uint<1>  DEPTH_CONV_EN  = LAYER_EN[1];
	ap_uint<1>  CONV_EN        = LAYER_EN[2];
	ap_uint<1>  RELU_EN        = LAYER_EN[3];
	ap_uint<1>  RELU6_EN       = LAYER_EN[4];
	ap_uint<1>  POOL_EN        = LAYER_EN[5];
	ap_uint<1>  UP_SAMPLE_EN   = LAYER_EN[6];

	if (DEPTH_CONV_EN == 1){
		// write to downstream modules: depth_conv
		int ii = 0;
		int p = 0;
		int q = 0;
		bool done = 0;
		while(!done){
#pragma HLS PIPELINE II=1
// Data layout of the corresponding buffer: F * F * Tn
			uint local_w_idx = p * FILTER_S1 * LAYER_IN_NUM_T + q * LAYER_IN_NUM_T + ii * SIMD_LANE;
			uint bus_w_idx = local_w_idx / BUS_PACK_FACTOR1;
			uint bus_w_offset = local_w_idx % BUS_PACK_FACTOR1;
			bus_t1 bus_w_data = weight_burst_buf1[bus_w_idx];
			WeightLoadData0Type fifo_w_data;
			
// DATA_SEL_FACTOR = BUS_PACK_FACTOR / SIMD_LANE
// BUS_PACK_FACTOR is the number of elements packed in one to enable memory coalescing
// Since each entry in FIFOs will be SIMD_LANE elements of the data, we should unpack based on SIMD_LANE
#if DATA_SEL_FACTOR1 == 1
			fifo_w_data = bus_w_data;
#elif DATA_SEL_FACTOR1 == 2
			switch(bus_w_offset / SIMD_LANE){
			case 0:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
				break;
			}
#elif DATA_SEL_FACTOR1 == 4         
			switch(bus_w_offset / SIMD_LANE){
			case 0:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
				break;
			case 2:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
				break;
			case 3:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
				break;
			}
#elif DATA_SEL_FACTOR1 == 8
			switch(bus_w_offset / SIMD_LANE){
			case 0:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
				break;
			case 2:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
				break;
			case 3:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
				break;
			case 4:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 5 - 1, DATA_W1 * SIMD_LANE * 4);
				break;
			case 5:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 6 - 1, DATA_W1 * SIMD_LANE * 5);
				break;
			case 6:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 7 - 1, DATA_W1 * SIMD_LANE * 6);
				break;
			case 7:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 8 - 1, DATA_W1 * SIMD_LANE * 7);
				break;
			}
#elif DATA_SEL_FACTOR1 == 16
			switch(bus_w_offset / SIMD_LANE){
			case 0:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
				break;
			case 2:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
				break;
			case 3:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
				break;
			case 4:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 5 - 1, DATA_W1 * SIMD_LANE * 4);
				break;
			case 5:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 6 - 1, DATA_W1 * SIMD_LANE * 5);
				break;
			case 6:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 7 - 1, DATA_W1 * SIMD_LANE * 6);
				break;
			case 7:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 8 - 1, DATA_W1 * SIMD_LANE * 7);
				break;
			case 8:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 9 - 1, DATA_W1 * SIMD_LANE * 8);
				break;
			case 9:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 10 - 1, DATA_W1 * SIMD_LANE * 9);
				break;
			case 10:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 11 - 1, DATA_W1 * SIMD_LANE * 10);
				break;
			case 11:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 12 - 1, DATA_W1 * SIMD_LANE * 11);
				break;
			case 12:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 13 - 1, DATA_W1 * SIMD_LANE * 12);
				break;
			case 13:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 14 - 1, DATA_W1 * SIMD_LANE * 13);
				break;
			case 14:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 15 - 1, DATA_W1 * SIMD_LANE * 14);
				break;
			case 15:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 16 - 1, DATA_W1 * SIMD_LANE * 15);
				break;
			}
#endif         
			fifo_depth_conv_weight.write(fifo_w_data);
			
			// Repeat until all of the weights for this tile is inserted into its FIFO
			q++;
			if (q == FILTER_S1){
				q = 0;
				p++;
				if (p == FILTER_S1){
					p = 0;
					ii++;
					if (ii == LAYER_IN_NUM_T / SIMD_LANE){
						ii = 0;
						done = 1;
					}
				}
			}

		}
	}
}

/*
 * Function name: weight_load_conv_weight_write
 * Function description: this function writes conv weights to conv module.
 * It has the same functionality as weight_load_depth_conv_weight_write
 */
void weight_load_conv_weight_write(
		bus_t1 weight_burst_buf2[],
		hls::stream<WeightLoadData1Type> &fifo_conv_weight,
		ConfigInst inst0,
		ConfigInst inst1,
		ConfigInst inst2,
		ConfigInst inst3,
		uint in_num_iter,
		uint out_num_iter
){
	// inst0
	ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
	ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
	ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
	ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
	ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
	ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
	// inst1
	ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
	ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
	ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
	ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
	ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
	ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
	// inst2
	ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
	ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
	ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
	ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
	ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
	ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
	ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
	// inst3
	ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
	ap_uint<32> PREV_CIN_OFFSET  = inst3(32*1+31, 32*1);
	ap_uint<16> LAYER_IN_NUM_T   = inst3(32*2+15, 32*2);
	ap_uint<16> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2+16);
	ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
	ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);

	ap_uint<1>  CONV_1ST_EN    = LAYER_EN[0];
	ap_uint<1>  DEPTH_CONV_EN  = LAYER_EN[1];
	ap_uint<1>  CONV_EN        = LAYER_EN[2];
	ap_uint<1>  RELU_EN        = LAYER_EN[3];
	ap_uint<1>  RELU6_EN       = LAYER_EN[4];
	ap_uint<1>  POOL_EN        = LAYER_EN[5];
	ap_uint<1>  UP_SAMPLE_EN   = LAYER_EN[6];  // reserved

	if (CONV_EN == 1){
		int oo = 0;
		int p = 0;
		int q = 0;
		int ii = 0;
		bool done = 0;
		while(!done){
#pragma HLS PIPELINE II=1

#ifdef DEBUG_weight2
			cout << "in loading weights " << DATA_SEL_FACTOR1 << " " << ii << " " << q << " " << p << " " << oo <<" " << LAYER_OUT_NUM_T << endl;
#endif          
			uint local_w_idx = oo * FILTER_S2 * FILTER_S2 * LAYER_IN_NUM_T + p * FILTER_S2 * LAYER_IN_NUM_T + q * LAYER_IN_NUM_T + ii * SIMD_LANE;
			uint bus_w_idx = local_w_idx / BUS_PACK_FACTOR1;
			uint bus_w_offset = local_w_idx % BUS_PACK_FACTOR1;
			bus_t1 bus_w_data = weight_burst_buf2[bus_w_idx];
			WeightLoadData1Type fifo_w_data;
#if DATA_SEL_FACTOR1 == 1
			fifo_w_data = bus_w_data;
#elif DATA_SEL_FACTOR1 == 2
			switch(bus_w_offset / SIMD_LANE){
			case 0:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
				break;
			}
#elif DATA_SEL_FACTOR1 == 4
			switch(bus_w_offset / SIMD_LANE){
			case 0:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
				break;
			case 2:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
				break;
			case 3:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
				break;
			}
#elif DATA_SEL_FACTOR1 == 8
			switch(bus_w_offset / SIMD_LANE){
			case 0:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
				break;
			case 2:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
				break;
			case 3:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
				break;
			case 4:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 5 - 1, DATA_W1 * SIMD_LANE * 4);
				break;
			case 5:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 6 - 1, DATA_W1 * SIMD_LANE * 5);
				break;
			case 6:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 7 - 1, DATA_W1 * SIMD_LANE * 6);
				break;
			case 7:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 8 - 1, DATA_W1 * SIMD_LANE * 7);
				break;
			}
#elif DATA_SEL_FACTOR1 == 16
			switch(bus_w_offset / SIMD_LANE){
			case 0:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 1 - 1, DATA_W1 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 2 - 1, DATA_W1 * SIMD_LANE * 1);
				break;
			case 2:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 3 - 1, DATA_W1 * SIMD_LANE * 2);
				break;
			case 3:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 4 - 1, DATA_W1 * SIMD_LANE * 3);
				break;
			case 4:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 5 - 1, DATA_W1 * SIMD_LANE * 4);
				break;
			case 5:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 6 - 1, DATA_W1 * SIMD_LANE * 5);
				break;
			case 6:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 7 - 1, DATA_W1 * SIMD_LANE * 6);
				break;
			case 7:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 8 - 1, DATA_W1 * SIMD_LANE * 7);
				break;
			case 8:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 9 - 1, DATA_W1 * SIMD_LANE * 8);
				break;
			case 9:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 10 - 1, DATA_W1 * SIMD_LANE * 9);
				break;
			case 10:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 11 - 1, DATA_W1 * SIMD_LANE * 10);
				break;
			case 11:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 12 - 1, DATA_W1 * SIMD_LANE * 11);
				break;
			case 12:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 13 - 1, DATA_W1 * SIMD_LANE * 12);
				break;
			case 13:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 14 - 1, DATA_W1 * SIMD_LANE * 13);
				break;
			case 14:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 15 - 1, DATA_W1 * SIMD_LANE * 14);
				break;
			case 15:
				fifo_w_data = bus_w_data(DATA_W1 * SIMD_LANE * 16 - 1, DATA_W1 * SIMD_LANE * 15);
				break;
			}
#endif          
			fifo_conv_weight.write(fifo_w_data);

			ii++;
			if (ii == LAYER_IN_NUM_T / SIMD_LANE){
				ii = 0;
				q++;
				if (q == FILTER_S2){
					q = 0;
					p++;
					if (p == FILTER_S2){
						p = 0;
						oo++;
						if (oo == LAYER_OUT_NUM_T){
							oo = 0;
							done = 1;
						}
					}
				}
			}
		}
	}
}

/*
 * Function name: weight_load_bias_write
 * Function description: This function writes bias to relu module.
 */
void weight_load_bias_write(
		bus_t2 bias_burst_buf[],
		hls::stream<WeightLoadData2Type> &fifo_bias,
		ConfigInst inst0,
		ConfigInst inst1,
		ConfigInst inst2,
		ConfigInst inst3,
		uint in_num_iter,
		uint out_num_iter
){
	// inst0
	ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
	ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
	ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
	ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
	ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
	ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
	// inst1
	ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
	ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
	ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
	ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
	ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
	ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
	// inst2
	ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
	ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
	ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
	ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
	ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
	ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
	ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
	// inst3
	ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
	ap_uint<32> PREV_CIN_OFFSET  = inst3(32*1+31, 32*1);
	ap_uint<16> LAYER_IN_NUM_T   = inst3(32*2+15, 32*2);
	ap_uint<16> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2+16);
	ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
	ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);

	ap_uint<1>  CONV_1ST_EN    = LAYER_EN[0];
	ap_uint<1>  DEPTH_CONV_EN  = LAYER_EN[1];
	ap_uint<1>  CONV_EN        = LAYER_EN[2];
	ap_uint<1>  RELU_EN        = LAYER_EN[3];
	ap_uint<1>  RELU6_EN       = LAYER_EN[4];
	ap_uint<1>  POOL_EN        = LAYER_EN[5];
	ap_uint<1>  UP_SAMPLE_EN   = LAYER_EN[6];  // reserved

	if (CONV_EN == 1){
		if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
			bias_write_loop: for (int oo = 0; oo < LAYER_OUT_NUM_T / SIMD_LANE; oo++){
#pragma HLS PIPELINE II=1
				uint local_b_idx = oo * SIMD_LANE;
				uint bus_b_idx = local_b_idx / BUS_PACK_FACTOR2;
				uint bus_b_offset = local_b_idx % BUS_PACK_FACTOR2;
				bus_t2 bus_b_data = bias_burst_buf[bus_b_idx];
				WeightLoadData2Type fifo_b_data;
				
// DATA_SEL_FACTOR = BUS_PACK_FACTOR / SIMD_LANE
// BUS_PACK_FACTOR is the number of elements packed in one to enable memory coalescing
// Since each entry in FIFOs will be SIMD_LANE elements of the data, we should unpack based on SIMD_LANE
#if DATA_SEL_FACTOR2 == 1
				fifo_b_data = bus_b_data;
#elif DATA_SEL_FACTOR2 == 2
				switch(bus_b_offset / SIMD_LANE){
				case 0:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
					break;
				case 1:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
					break;
				}
#elif DATA_SEL_FACTOR2 == 4
				switch(bus_b_offset / SIMD_LANE){
				case 0:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
					break;
				case 1:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
					break;
				case 2:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 3 - 1, DATA_W2 * SIMD_LANE * 2);
					break;
				case 3:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 4 - 1, DATA_W2 * SIMD_LANE * 3);
					break;
				}
#elif DATA_SEL_FACTOR2 == 8
				switch(bus_b_offset / SIMD_LANE){
				case 0:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
					break;
				case 1:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
					break;
				case 2:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 3 - 1, DATA_W2 * SIMD_LANE * 2);
					break;
				case 3:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 4 - 1, DATA_W2 * SIMD_LANE * 3);
					break;
				case 4:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 5 - 1, DATA_W2 * SIMD_LANE * 4);
					break;
				case 5:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 6 - 1, DATA_W2 * SIMD_LANE * 5);
					break;
				case 6:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 7 - 1, DATA_W2 * SIMD_LANE * 6);
					break;
				case 7:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 8 - 1, DATA_W2 * SIMD_LANE * 7);
					break;
				}
#elif DATA_SEL_FACTOR2 == 16
				switch(bus_b_offset / SIMD_LANE){
				case 0:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
					break;
				case 1:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
					break;
				case 2:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 3 - 1, DATA_W2 * SIMD_LANE * 2);
					break;
				case 3:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 4 - 1, DATA_W2 * SIMD_LANE * 3);
					break;
				case 4:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 5 - 1, DATA_W2 * SIMD_LANE * 4);
					break;
				case 5:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 6 - 1, DATA_W2 * SIMD_LANE * 5);
					break;
				case 6:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 7 - 1, DATA_W2 * SIMD_LANE * 6);
					break;
				case 7:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 8 - 1, DATA_W2 * SIMD_LANE * 7);
					break;
				case 8:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 9 - 1, DATA_W2 * SIMD_LANE * 8);
					break;
				case 9:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 10 - 1, DATA_W2 * SIMD_LANE * 9);
					break;
				case 10:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 11 - 1, DATA_W2 * SIMD_LANE * 10);
					break;
				case 11:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 12 - 1, DATA_W2 * SIMD_LANE * 11);
					break;
				case 12:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 13 - 1, DATA_W2 * SIMD_LANE * 12);
					break;
				case 13:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 14 - 1, DATA_W2 * SIMD_LANE * 13);
					break;
				case 14:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 15 - 1, DATA_W2 * SIMD_LANE * 14);
					break;
				case 15:
					fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 16 - 1, DATA_W2 * SIMD_LANE * 15);
					break;
				}
#endif       
				fifo_bias.write(fifo_b_data);
			}
		}
	}
}

/*
 * Function name: weight_load_depth_norm_write
 * Function description: This function writes parameters for normalization to relu6 module.
 */
void weight_load_depth_norm_write(
		bus_t2 bias_burst_buf[],
		hls::stream<WeightLoadData2Type> &fifo_bias,
		ConfigInst inst0,
		ConfigInst inst1,
		ConfigInst inst2,
		ConfigInst inst3,
		uint in_num_iter,
		uint out_num_iter
){
	// Refer to cin_load module to understand the meaning of the instructions
	// inst0
	ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
	ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
	ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
	ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
	ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
	ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
	// inst1
	ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
	ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
	ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
	ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
	ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
	ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
	// inst2
	ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
	ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
	ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
	ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
	ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
	ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
	ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
	// inst3
	ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
	ap_uint<32> PREV_CIN_OFFSET  = inst3(32*1+31, 32*1);
	ap_uint<16> LAYER_IN_NUM_T   = inst3(32*2+15, 32*2);
	ap_uint<16> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2+16);
	ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
	ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);

	ap_uint<1>  CONV_1ST_EN    = LAYER_EN[0];
	ap_uint<1>  DEPTH_CONV_EN  = LAYER_EN[1];
	ap_uint<1>  CONV_EN        = LAYER_EN[2];
	ap_uint<1>  RELU_EN        = LAYER_EN[3];
	ap_uint<1>  RELU6_EN       = LAYER_EN[4];
	ap_uint<1>  POOL_EN        = LAYER_EN[5];
	ap_uint<1>  UP_SAMPLE_EN   = LAYER_EN[6];  // reserved

	if (DEPTH_CONV_EN == 1 && RELU6_EN == 1){
		//if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
		bias_write_loop: for (int oo = 0; oo < LAYER_IN_NUM_T / SIMD_LANE; oo++){
#pragma HLS PIPELINE II=1
			uint local_b_idx = oo * SIMD_LANE;
			uint bus_b_idx = local_b_idx / BUS_PACK_FACTOR2;
			uint bus_b_offset = local_b_idx % BUS_PACK_FACTOR2;
			bus_t2 bus_b_data = bias_burst_buf[bus_b_idx];
			WeightLoadData2Type fifo_b_data;
			
// DATA_SEL_FACTOR = BUS_PACK_FACTOR / SIMD_LANE
// BUS_PACK_FACTOR is the number of elements packed in one to enable memory coalescing
// Since each entry in FIFOs will be SIMD_LANE elements of the data, we should unpack based on SIMD_LANE
#if DATA_SEL_FACTOR2 == 1
			fifo_b_data = bus_b_data;
#elif DATA_SEL_FACTOR2 == 2
			switch(bus_b_offset / SIMD_LANE){
			case 0:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
				break;
			}
#elif DATA_SEL_FACTOR2 == 4
			switch(bus_b_offset / SIMD_LANE){
			case 0:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
				break;
			case 2:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 3 - 1, DATA_W2 * SIMD_LANE * 2);
				break;
			case 3:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 4 - 1, DATA_W2 * SIMD_LANE * 3);
				break;
			}
#elif DATA_SEL_FACTOR2 == 8
			switch(bus_b_offset / SIMD_LANE){
			case 0:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
				break;
			case 2:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 3 - 1, DATA_W2 * SIMD_LANE * 2);
				break;
			case 3:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 4 - 1, DATA_W2 * SIMD_LANE * 3);
				break;
			case 4:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 5 - 1, DATA_W2 * SIMD_LANE * 4);
				break;
			case 5:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 6 - 1, DATA_W2 * SIMD_LANE * 5);
				break;
			case 6:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 7 - 1, DATA_W2 * SIMD_LANE * 6);
				break;
			case 7:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 8 - 1, DATA_W2 * SIMD_LANE * 7);
				break;
			}
#elif DATA_SEL_FACTOR2 == 16
			switch(bus_b_offset / SIMD_LANE){
			case 0:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 1 - 1, DATA_W2 * SIMD_LANE * 0);
				break;
			case 1:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 2 - 1, DATA_W2 * SIMD_LANE * 1);
				break;
			case 2:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 3 - 1, DATA_W2 * SIMD_LANE * 2);
				break;
			case 3:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 4 - 1, DATA_W2 * SIMD_LANE * 3);
				break;
			case 4:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 5 - 1, DATA_W2 * SIMD_LANE * 4);
				break;
			case 5:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 6 - 1, DATA_W2 * SIMD_LANE * 5);
				break;
			case 6:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 7 - 1, DATA_W2 * SIMD_LANE * 6);
				break;
			case 7:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 8 - 1, DATA_W2 * SIMD_LANE * 7);
				break;
			case 8:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 9 - 1, DATA_W2 * SIMD_LANE * 8);
				break;
			case 9:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 10 - 1, DATA_W2 * SIMD_LANE * 9);
				break;
			case 10:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 11 - 1, DATA_W2 * SIMD_LANE * 10);
				break;
			case 11:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 12 - 1, DATA_W2 * SIMD_LANE * 11);
				break;
			case 12:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 13 - 1, DATA_W2 * SIMD_LANE * 12);
				break;
			case 13:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 14 - 1, DATA_W2 * SIMD_LANE * 13);
				break;
			case 14:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 15 - 1, DATA_W2 * SIMD_LANE * 14);
				break;
			case 15:
				fifo_b_data = bus_b_data(DATA_W2 * SIMD_LANE * 16 - 1, DATA_W2 * SIMD_LANE * 15);
				break;
			}
#endif       
			fifo_bias.write(fifo_b_data);
		}
		//}
	}
}


/**
 * Function name: weight_load
 * Function description: This function loads weights and distributes them to downstream modules.
 */
void weight_load(
		bus_t1                           *global_weight,
		bus_t2                           *global_bias,
		hls::stream<ConfigInst>          &fifo_config_in,
		hls::stream<WeightLoadData0Type> &fifo_depth_conv_weight,
		hls::stream<WeightLoadData1Type> &fifo_conv_weight,
		hls::stream<ConvData0Type>       &fifo_gamma_depth,
		hls::stream<ConvData0Type>       &fifo_beta_depth,
		hls::stream<ConvData0Type>       &fifo_gamma_conv,
		hls::stream<ConvData0Type>       &fifo_beta_conv,
		hls::stream<ConfigInst>          &fifo_config_out
){
#pragma HLS INLINE off 
	// on-chip buffers
	bus_t1 weight_burst_buf1[IN_NUM_T * K_T * K_T / BUS_PACK_FACTOR1];
	bus_t1 weight_burst_buf2[OUT_NUM_T * IN_NUM_T * K_T * K_T / BUS_PACK_FACTOR1];
//	bus_t2 bias_burst_buf[OUT_NUM_T / BUS_PACK_FACTOR2];
	bus_t2 beta_depth_burst_buf[IN_NUM_T / BUS_PACK_FACTOR2];
	bus_t2 gamma_depth_burst_buf[IN_NUM_T / BUS_PACK_FACTOR2];
	bus_t2 beta_conv_burst_buf[OUT_NUM_T / BUS_PACK_FACTOR2];
	bus_t2 gamma_conv_burst_buf[OUT_NUM_T / BUS_PACK_FACTOR2];
#pragma HLS RESOURCE variable=weight_burst_buf1 core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=weight_burst_buf2 core=XPM_MEMORY uram  
//#pragma HLS RESOURCE variable=bias_burst_buf core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=beta_depth_burst_buf core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=gamma_depth_burst_buf core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=beta_conv_burst_buf core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=gamma_conv_burst_buf core=XPM_MEMORY uram

	// tiling iterators
	uint in_num_iter = 0;
	uint out_num_iter = 0;
	uint in_h_iter = 0;
	uint in_w_iter = 0;
	uint layer_iter = 0;

	// Read instructions
	ConfigInst inst0 = fifo_config_in.read();
	fifo_config_out.write(inst0);
	ConfigInst inst1 = fifo_config_in.read();
	fifo_config_out.write(inst1);
	ConfigInst inst2 = fifo_config_in.read();
	fifo_config_out.write(inst2);
	ConfigInst inst3 = fifo_config_in.read();
	fifo_config_out.write(inst3);
	ConfigInst inst4 = fifo_config_in.read();
	fifo_config_out.write(inst4);

	ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

	bool layer_start = 0;
	bool done = 0;
	// We assum that cin has been pre-padded with zeros
	while(!done){
		if (layer_start){
			inst0 = fifo_config_in.read();
			fifo_config_out.write(inst0);
			inst1 = fifo_config_in.read();
			fifo_config_out.write(inst1);
			inst2 = fifo_config_in.read();
			fifo_config_out.write(inst2);
			inst3 = fifo_config_in.read();
			fifo_config_out.write(inst3);
			inst4 = fifo_config_in.read();
			fifo_config_out.write(inst4);
			layer_start = 0;
		}

		// Refer to cin_load module to understand the meaning of the instructions
		// inst0
		ap_uint<32> LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
		ap_uint<32> LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
		ap_uint<32> LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
		ap_uint<32> LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
		ap_uint<32> LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
		ap_uint<32> LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
		// inst1
		ap_uint<32> LAYER_IN_NUM     = inst1(32*0+31, 32*0);
		ap_uint<32> LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
		ap_uint<32> LAYER_IN_H       = inst1(32*2+31, 32*2);
		ap_uint<32> LAYER_IN_W       = inst1(32*3+31, 32*3);
		ap_uint<32> LAYER_OUT_H      = inst1(32*4+31, 32*4);
		ap_uint<32> LAYER_OUT_W      = inst1(32*5+31, 32*5);
		// inst2
		ap_uint<32> CIN_OFFSET       = inst2(32*0+31, 32*0);
		ap_uint<32> WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
		ap_uint<32> BIAS_OFFSET      = inst2(32*2+31, 32*2);
		ap_uint<32> COUT_OFFSET      = inst2(32*3+31, 32*3);
		ap_uint<16> FILTER_S1        = inst2(32*4+15, 32*4);
		ap_uint<16> FILTER_S2        = inst2(32*4+31, 32*4+16);
		ap_uint<32> STRIDE           = inst2(32*5+31, 32*5);
		// inst3
		ap_uint<32> LAYER_EN         = inst3(32*0+31, 32*0);
		ap_uint<32> PREV_CIN_OFFSET  = inst3(32*1+31, 32*1);
		ap_uint<16> LAYER_IN_NUM_T   = inst3(32*2+15, 32*2);
		ap_uint<16> LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2+16);
		ap_uint<32> LAYER_IN_H_T     = inst3(32*3+31, 32*3);
		ap_uint<32> LAYER_IN_W_T     = inst3(32*4+31, 32*4);

		ap_uint<1>  CONV_1ST_EN    = LAYER_EN[0];
		ap_uint<1>  DEPTH_CONV_EN  = LAYER_EN[1];
		ap_uint<1>  CONV_EN        = LAYER_EN[2];
		ap_uint<1>  RELU_EN        = LAYER_EN[3];
		ap_uint<1>  RELU6_EN       = LAYER_EN[4];
		ap_uint<1>  POOL_EN        = LAYER_EN[5];
		ap_uint<1>  UP_SAMPLE_EN   = LAYER_EN[6];  // reserved
    ap_uint<1>  BIAS_EN        = LAYER_EN[7];
		ap_uint<1>  BATCH_NORM_EN  = LAYER_EN[10];
    ap_uint<1>  BATCH_NORM_EN_DEPTH  = LAYER_EN[12];

#ifdef DEBUG_config
		cout << LAYER_IN_NUM_HW << " " << LAYER_OUT_NUM_HW << " " << LAYER_IN_H_HW << " " << LAYER_IN_W_HW << " " << LAYER_OUT_H_HW << " " << LAYER_OUT_W_HW << endl;
		cout << LAYER_IN_NUM << " " << LAYER_OUT_NUM << " " << LAYER_IN_H << " " << LAYER_IN_W << " " << LAYER_OUT_H << " " << LAYER_OUT_W << endl;
		cout << CIN_OFFSET << " " << WEIGHT_OFFSET << " " << BIAS_OFFSET << " " << COUT_OFFSET << " " << FILTER_S1 << " " << FILTER_S2 << " " << STRIDE << endl;
		cout << LAYER_EN << " " << PREV_CIN_OFFSET << " " << LAYER_IN_NUM_T << " " << LAYER_OUT_NUM_T << " " << LAYER_IN_H_T << " " << LAYER_IN_W_T << endl;
#endif

		// Set up some configuration signals
		bool bias_en = (CONV_EN == 1 && BIAS_EN == 1);
		bool norm_depth_en = (DEPTH_CONV_EN == 1 && BATCH_NORM_EN_DEPTH == 1);;
		bool norm_conv_en = (CONV_EN == 1 && BATCH_NORM_EN == 1);

		uint beta_depth_offset = 0;
		uint gamma_depth_offset = 0;
		uint beta_conv_offset = 0;
		uint gamma_conv_offset = 0;
		uint bias_offset = BIAS_OFFSET;

		// Set the offsets if batch normalization is used (final_result = gamma * computed_result + beta)
		// Depthwise separable convolution has two sublayers of computation,
		// one is the DW sublayer and the other is the normal 1x1 conv sublayer
		// Both of these layers may need normalization
		// In DRAM, for each layer, first the BETAs are stored and then the GAMMAs are stored
		if (norm_depth_en) {
			beta_depth_offset = bias_offset;
			gamma_depth_offset = bias_offset + LAYER_IN_NUM_HW;
			beta_conv_offset = gamma_depth_offset + LAYER_IN_NUM_HW;
			gamma_conv_offset = beta_conv_offset + LAYER_OUT_NUM_HW;
		} else if (norm_conv_en) {
			beta_conv_offset = bias_offset;
			gamma_conv_offset = beta_conv_offset + LAYER_OUT_NUM_HW;
		}

		// offsets
		uint weight_offset1 = 0;
		uint weight_offset2 = 0;
		weight_offset1 = WEIGHT_OFFSET;

		if (DEPTH_CONV_EN == 1)
			weight_offset2 = WEIGHT_OFFSET + LAYER_IN_NUM_HW * FILTER_S1 * FILTER_S1;
		else
			weight_offset2 = WEIGHT_OFFSET;

		// Load bias (when batch normalization is not used: final_result = computed_result + bias)
		// Set GAMMAs to zero
		if (bias_en){
			// Only write out in the last iteration
			if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
				uint global_bias_offset = bias_offset + out_num_iter;
        for (int i = 0; i < OUT_NUM_T / BUS_PACK_FACTOR2; i++){
        #pragma HLS pipeline
				  gamma_conv_burst_buf[i] = 0;
        }
				memcpy((void*)beta_conv_burst_buf, (void*)&global_bias[global_bias_offset / BUS_PACK_FACTOR2], sizeof(data_t2) * LAYER_OUT_NUM_T);
			}
		} else{

      	// Load batch normalization info for depth conv
      	if (norm_depth_en){
      		uint global_beta_offset = beta_depth_offset + in_num_iter;
      		memcpy((void*)beta_depth_burst_buf, (void*)&global_bias[global_beta_offset / BUS_PACK_FACTOR2], sizeof(data_t2) * LAYER_IN_NUM_T);
      		
      		uint global_gamma_offset = gamma_depth_offset + in_num_iter;
      		memcpy((void*)gamma_depth_burst_buf, (void*)&global_bias[global_gamma_offset / BUS_PACK_FACTOR2], sizeof(data_t2) * LAYER_IN_NUM_T);
      		
      	}
      
      	// Load batch normalization info for conv
      	if (norm_conv_en){
      		if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
      			uint global_bias_offset = beta_conv_offset + out_num_iter;
      #ifdef DEBUG_weight
      			cout << global_bias_offset << " beta " << beta_conv_offset << endl;
      #endif
      			memcpy((void*)beta_conv_burst_buf, (void*)&global_bias[global_bias_offset / BUS_PACK_FACTOR2], sizeof(data_t2) * LAYER_OUT_NUM_T);
      		}
      
      		if (in_num_iter + LAYER_IN_NUM_T >= LAYER_IN_NUM){
      			uint global_bias_offset = gamma_conv_offset + out_num_iter;
      #ifdef DEBUG_weight
      			cout << global_bias_offset << " gamma " << gamma_conv_offset << endl;
      #endif
      			memcpy((void*)gamma_conv_burst_buf, (void*)&global_bias[global_bias_offset / BUS_PACK_FACTOR2], sizeof(data_t2) * LAYER_OUT_NUM_T);
      		}
      	}
    }

#ifdef DEBUG_weight
		cout << "loaded beta and gamma" << endl;
#endif

		// Load weights of the depth conv module
		if (DEPTH_CONV_EN == 1){
			// load from DRAM
			uint global_weight_offset = weight_offset1 + in_num_iter * FILTER_S1 * FILTER_S1;
			if (FILTER_S1 == 1){
				memcpy((void*)&weight_burst_buf1, (void*)&global_weight[global_weight_offset / BUS_PACK_FACTOR1], sizeof(data_t1) * LAYER_IN_NUM_T * 1 * 1);
			} else if (FILTER_S1 == 3){
				memcpy((void*)&weight_burst_buf1, (void*)&global_weight[global_weight_offset / BUS_PACK_FACTOR1], sizeof(data_t1) * LAYER_IN_NUM_T * 3 * 3);
			}
		}
		// Load weights of the conv module
		if (CONV_EN == 1){
			uint global_weight_offset = weight_offset2 + out_num_iter * LAYER_IN_NUM_HW * FILTER_S2 * FILTER_S2 + in_num_iter * LAYER_OUT_NUM_T * FILTER_S2 * FILTER_S2;
			if (FILTER_S2 == 1){
				memcpy((void*)&weight_burst_buf2[0], (void*)&global_weight[global_weight_offset / BUS_PACK_FACTOR1], sizeof(data_t1) * LAYER_OUT_NUM_T * LAYER_IN_NUM_T * 1 * 1);
			} else if (FILTER_S2 == 3){
				memcpy((void*)&weight_burst_buf2[0], (void*)&global_weight[global_weight_offset / BUS_PACK_FACTOR1], sizeof(data_t1) * LAYER_OUT_NUM_T * LAYER_IN_NUM_T * 3 * 3);
			}
		}

#ifdef DEBUG_weight
		cout << "loaded weights" << endl;
#endif
		// Fill the FIFOs with the loaded data
		weight_load_depth_conv_weight_write(weight_burst_buf1, fifo_depth_conv_weight, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);

#ifdef DEBUG_weight
		cout << "loaded weights" << endl;
#endif

		weight_load_conv_weight_write(weight_burst_buf2, fifo_conv_weight, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);

#ifdef DEBUG_weight
		cout << "loaded weights" << endl;
#endif

	// Load BETAs and GAMMAs to their FIFOs
	// If there doesn't exist a batch normalization and it's a normal bias,
	// beta = bias, gamma = 0
    if (bias_en) {
		  weight_load_bias_write(beta_conv_burst_buf, fifo_beta_conv, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);
      weight_load_bias_write(gamma_conv_burst_buf, fifo_gamma_conv, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);
    }
		else if(norm_conv_en){
			weight_load_bias_write(beta_conv_burst_buf, fifo_beta_conv, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);
			weight_load_bias_write(gamma_conv_burst_buf, fifo_gamma_conv, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);
		}

		weight_load_depth_norm_write(beta_depth_burst_buf, fifo_beta_depth, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);
		weight_load_depth_norm_write(gamma_depth_burst_buf, fifo_gamma_depth, inst0, inst1, inst2, inst3, in_num_iter, out_num_iter);
#ifdef DEBUG_weight
		cout << in_num_iter << " in num iter " << endl;
#endif
		// Repeat until all the tiles are read
		// Then, have to repeat reading to calculate all LAYER_OUT_NUM output feature maps
		in_num_iter += LAYER_IN_NUM_T;
		if (in_num_iter >= LAYER_IN_NUM){
			in_num_iter = 0;
			in_h_iter += LAYER_IN_H_T;
			if (in_h_iter >= LAYER_IN_H){
				in_h_iter = 0;
				in_w_iter += LAYER_IN_W_T;
				if (in_w_iter >= LAYER_IN_W){
					in_w_iter = 0;
					out_num_iter += LAYER_OUT_NUM_T;
					if (out_num_iter >= LAYER_OUT_NUM){
						out_num_iter = 0;
						layer_iter += 1;
						layer_start = 1;
						if (layer_iter == LAYER_BATCH){
							layer_iter = 0;
							done = 1;
						}
					}
				}
			}
		}
	}
}