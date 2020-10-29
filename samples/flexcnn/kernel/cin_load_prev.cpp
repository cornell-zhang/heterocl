/*
Functions:

cin_load_fifo_write_prev
cin_load_prev

Note: 

function cin_load_ddr_read is in cin_load.cpp

*/


/*
 * Function name: cin_load_fifo_write_prev
 * Function description: This function writes cin data to the downstream modules.
 * It has the same functionality as cin_load_fifo_write
 */
void cin_load_fifo_write_prev(
		bus_t0                         cin_burst_buf[],
		hls::stream<CinLoadData0Type>  &fifo_cin,
		uint                           LAYER_IN_NUM_T,
		uint                           LAYER_IN_H_T,
		uint                           LAYER_IN_W_T,
		uint                           FILTER_S
){
	int ii = 0;
	int hh = 0;
	int ww = 0;
	bool done = 0;
	uint count = 0;
	while(!done){
#pragma HLS PIPELINE II=1
// Data layout of the buffer: Th * Tw * Tn
		uint local_cin_idx = hh * (LAYER_IN_W_T + FILTER_S - 1) * LAYER_IN_NUM_T + ww * LAYER_IN_NUM_T + ii * DEPTH_CONV_LANE;
		uint bus_cin_idx = local_cin_idx / BUS_PACK_FACTOR0;
		uint bus_cin_offset = local_cin_idx % BUS_PACK_FACTOR0;
		bus_t0 bus_cin_data = cin_burst_buf[bus_cin_idx];
		CinLoadData0Type fifo_cin_data;

// DATA_SEL_FACTOR = BUS_PACK_FACTOR / SIMD_LANE
// BUS_PACK_FACTOR is the number of elements packed in one to enable memory coalescing
// Since each entry is FIFOs will be SIMD_LANE elements of the data, we should unpack based on SIMD_LANE
#if DATA_SEL_FACTOR0 == 1
		fifo_cin_data = bus_cin_data;
#elif DATA_SEL_FACTOR0 == 2 
		switch(bus_cin_offset / DEPTH_CONV_LANE){
		case 0:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 1 - 1, DATA_W0 * DEPTH_CONV_LANE * 0);
			break;
		case 1:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 2 - 1, DATA_W0 * DEPTH_CONV_LANE * 1);
			break;
		}
#elif DATA_SEL_FACTOR0 == 4
		switch(bus_cin_offset / DEPTH_CONV_LANE){
		case 0:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 1 - 1, DATA_W0 * DEPTH_CONV_LANE * 0);
			break;
		case 1:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 2 - 1, DATA_W0 * DEPTH_CONV_LANE * 1);
			break;
		case 2:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 3 - 1, DATA_W0 * DEPTH_CONV_LANE * 2);
			break;
		case 3:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 4 - 1, DATA_W0 * DEPTH_CONV_LANE * 3);
			break;
		}
#elif DATA_SEL_FACTOR0 == 8
		switch(bus_cin_offset / DEPTH_CONV_LANE){
		case 0:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 1 - 1, DATA_W0 * DEPTH_CONV_LANE * 0);
			break;
		case 1:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 2 - 1, DATA_W0 * DEPTH_CONV_LANE * 1);
			break;
		case 2:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 3 - 1, DATA_W0 * DEPTH_CONV_LANE * 2);
			break;
		case 3:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 4 - 1, DATA_W0 * DEPTH_CONV_LANE * 3);
			break;
		case 4:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 5 - 1, DATA_W0 * DEPTH_CONV_LANE * 4);
			break;
		case 5:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 6 - 1, DATA_W0 * DEPTH_CONV_LANE * 5);
			break;
		case 6:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 7 - 1, DATA_W0 * DEPTH_CONV_LANE * 6);
			break;
		case 7:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 8 - 1, DATA_W0 * DEPTH_CONV_LANE * 7);
			break;
		}
#elif DATA_SEL_FACTOR0 == 16
		switch(bus_cin_offset / DEPTH_CONV_LANE){
		case 0:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 1 - 1, DATA_W0 * DEPTH_CONV_LANE * 0);
			break;
		case 1:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 2 - 1, DATA_W0 * DEPTH_CONV_LANE * 1);
			break;
		case 2:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 3 - 1, DATA_W0 * DEPTH_CONV_LANE * 2);
			break;
		case 3:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 4 - 1, DATA_W0 * DEPTH_CONV_LANE * 3);
			break;
		case 4:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 5 - 1, DATA_W0 * DEPTH_CONV_LANE * 4);
			break;
		case 5:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 6 - 1, DATA_W0 * DEPTH_CONV_LANE * 5);
			break;
		case 6:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 7 - 1, DATA_W0 * DEPTH_CONV_LANE * 6);
			break;
		case 7:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 8 - 1, DATA_W0 * DEPTH_CONV_LANE * 7);
			break;
		case 8:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 9 - 1, DATA_W0 * DEPTH_CONV_LANE * 8);
			break;
		case 9:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 10 - 1, DATA_W0 * DEPTH_CONV_LANE * 9);
			break;
		case 10:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 11 - 1, DATA_W0 * DEPTH_CONV_LANE * 10);
			break;
		case 11:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 12 - 1, DATA_W0 * DEPTH_CONV_LANE * 11);
			break;
		case 12:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 13 - 1, DATA_W0 * DEPTH_CONV_LANE * 12);
			break;
		case 13:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 14 - 1, DATA_W0 * DEPTH_CONV_LANE * 13);
			break;
		case 14:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 15 - 1, DATA_W0 * DEPTH_CONV_LANE * 14);
			break;
		case 15:
			fifo_cin_data = bus_cin_data(DATA_W0 * DEPTH_CONV_LANE * 16 - 1, DATA_W0 * DEPTH_CONV_LANE * 15);
			break;
		}
#endif            
		fifo_cin.write(fifo_cin_data);
#ifdef DEBUG_load_prev
		for (int lane = 0; lane < RELU_LANE; lane++){
#pragma HLS UNROLL
			ap_uint<DATA_W0> u32_beta = fifo_cin_data(DATA_W0 - 1, 0);
			data_t2 a = Reinterpret<data_t2>(u32_beta);
			fifo_cin_data = fifo_cin_data >> DATA_W0;
			cout << a << " ";
		}
		cout << endl;
#endif
		
		// Repeat until the whole tile is read
		ww++;
		if (ww == LAYER_IN_W_T + FILTER_S - 1){
			ww = 0;
			hh++;
			if (hh == LAYER_IN_H_T + FILTER_S - 1){
				hh = 0;
				ii++;
				if (ii == LAYER_IN_NUM_T / DEPTH_CONV_LANE){
					ii = 0;
					done = 1;
				}
			}
		}
	}

}



/*
 * Function name: cin_load
 * Function description: This function loads and distributes cin to the previous layer and instructions.
 */
void cin_load_prev(
		bus_t0                         *global_cin,
		hls::stream<ConfigInst>        &fifo_config_in,
		hls::stream<CinLoadData0Type>  &fifo_cin_prev,
		hls::stream<ConfigInst>        &fifo_config_out
){
#pragma HLS INLINE off 
	// on-chip buffer for cin data
	bus_t0 prev_cin_burst_buf_ping[OUT_NUM_T * (IN_H_T + K_T - 1) * (IN_W_T + K_T - 1) / BUS_PACK_FACTOR0];
	bus_t0 prev_cin_burst_buf_pong[OUT_NUM_T * (IN_H_T + K_T - 1) * (IN_W_T + K_T - 1) / BUS_PACK_FACTOR0];
#pragma HLS RESOURCE variable=prev_cin_burst_buf_ping core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=prev_cin_burst_buf_pong core=XPM_MEMORY uram  



	// tiling iterators
	uint in_num_iter = 0;
	uint out_num_iter = 0;
	uint in_h_iter = 0;
	uint in_w_iter = 0;
	uint layer_iter = 0;

	uint in_num_iter_prev = 0;
	uint out_num_iter_prev = 0;
	uint in_h_iter_prev = 0;
	uint in_w_iter_prev = 0;
	uint layer_iter_prev = 0;


	uint LAYER_IN_NUM_T_prev;
	uint LAYER_OUT_NUM_T_prev;
	uint LAYER_IN_H_T_prev;
	uint LAYER_IN_W_T_prev;
	uint FILTER_S_prev;

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
	ap_uint<1>  LOAD_PREV_CIN = 0;


	uint task_cnt = 0;
	bool layer_start = 0;
	bool layer_start_prev = 0;
	bool done = 0;
	// We assum that cin has been pre-padded with zeros
	uint prev = 0;
	uint init = 1;
	uint num_tile = 0;
	bool write_last_cin = 0;
	bool write_last_prev_cin = 0;
	bool start_prev = 0;
	bool done_prev = 0;
	bool change_layout = 0;
	//uint prev_iter = 0;
	while(!done_prev){

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
		ap_uint<1>  INTER_LOAD_EN  = LAYER_EN[8];
		ap_uint<1>  INTER_WRITE_EN = LAYER_EN[9];
		ap_uint<1>  BATCH_NORM_EN  = LAYER_EN[10];
		LOAD_PREV_CIN  = LAYER_EN[11];


#ifdef DEBUG_config_cin
		cout << LAYER_IN_NUM_HW << " " << LAYER_OUT_NUM_HW << " " << LAYER_IN_H_HW << " " << LAYER_IN_W_HW << " " << LAYER_OUT_H_HW << " " << LAYER_OUT_W_HW << endl;
		cout << LAYER_IN_NUM << " " << LAYER_OUT_NUM << " " << LAYER_IN_H << " " << LAYER_IN_W << " " << LAYER_OUT_H << " " << LAYER_OUT_W << endl;
		cout << CIN_OFFSET << " " << WEIGHT_OFFSET << " " << BIAS_OFFSET << " " << COUT_OFFSET << " " << FILTER_S1 << " " << FILTER_S2 << " " << STRIDE << endl;
		cout << LAYER_EN << " " << PREV_CIN_OFFSET << " " << LAYER_IN_NUM_T << " " << LAYER_OUT_NUM_T << " " << LAYER_IN_H_T << " " << LAYER_IN_W_T << endl;
#endif


		// offsets
		uint cin_offset = CIN_OFFSET;
		uint prev_cin_offset = PREV_CIN_OFFSET;

		if (prev == 1) start_prev = 1;

		// set up some configuration signals
		uint FILTER_S = (DEPTH_CONV_EN == 1)? (uint)FILTER_S1: ((CONV_EN == 1)? (uint)FILTER_S2: 1);
		bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
		bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
		bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);
		change_layout = (((LAYER_OUT_W_HW == LAYER_OUT_W) || (LAYER_OUT_W_HW == LAYER_IN_W_T)) && ((LAYER_OUT_H_HW == LAYER_OUT_H) || (LAYER_OUT_H_HW == LAYER_IN_H_T))); // if next filter = 1 : change the layout to num_tile, h_t, w_t, in_t
    uint layer_out_h = (LAYER_IN_H_T > LAYER_OUT_H) ? LAYER_IN_H_T : LAYER_OUT_H;
    uint layer_out_w = (LAYER_IN_W_T > LAYER_OUT_W) ? LAYER_IN_W_T : LAYER_OUT_W;

		if (INTER_LOAD_EN == 0){
			if (((max_pool || UP_SAMPLE_EN) && out_num_iter_prev == 0) || separable_conv || conv2d){
				if (task_cnt == 0){
					// First load previous cin
					// Load only if the enable signal for this module is on
					if(LOAD_PREV_CIN == 1){
						cin_load_ddr_read(global_cin, prev_cin_burst_buf_ping, layer_out_h, layer_out_w, LAYER_OUT_NUM_T, LAYER_IN_H_T, LAYER_IN_W_T, 1, prev_cin_offset, out_num_iter_prev, in_h_iter_prev, in_w_iter_prev, num_tile, change_layout, max_pool, 1);
					} 
				} else {
					// Apply double buffering for reading the data and filling the FIFO
					// Load only if the enable signal for this module is on
					if(LOAD_PREV_CIN == 1){
						if (task_cnt % 2 == 1){
							cin_load_ddr_read(global_cin, prev_cin_burst_buf_pong, layer_out_h, layer_out_w, LAYER_OUT_NUM_T, LAYER_IN_H_T, LAYER_IN_W_T, 1, prev_cin_offset, out_num_iter_prev, in_h_iter_prev, in_w_iter_prev, num_tile, change_layout, max_pool, 1);
							cin_load_fifo_write_prev(prev_cin_burst_buf_ping, fifo_cin_prev, LAYER_OUT_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, 1);
						} else {
							cin_load_ddr_read(global_cin, prev_cin_burst_buf_ping, layer_out_h, layer_out_w, LAYER_OUT_NUM_T, LAYER_IN_H_T, LAYER_IN_W_T, 1, prev_cin_offset, out_num_iter_prev, in_h_iter_prev, in_w_iter_prev, num_tile, change_layout, max_pool, 1);
							cin_load_fifo_write_prev(prev_cin_burst_buf_pong, fifo_cin_prev, LAYER_OUT_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, 1);
						}
					} 
				}

				task_cnt++;
				LAYER_IN_NUM_T_prev = LAYER_IN_NUM_T;
				LAYER_OUT_NUM_T_prev = LAYER_OUT_NUM_T;
				LAYER_IN_H_T_prev = LAYER_IN_H_T;
				LAYER_IN_W_T_prev = LAYER_IN_W_T;
				FILTER_S_prev = FILTER_S;
			}
		}

		// Continue until all tiles are read
		// No need to read tiles multiple times 
		// since it's only read to add them to the result of this layer
		num_tile++;
		out_num_iter_prev += LAYER_OUT_NUM_T;
		if (out_num_iter_prev >= LAYER_OUT_NUM){
			out_num_iter_prev = 0;
			in_h_iter_prev += LAYER_IN_H_T;
			if (in_h_iter_prev >= LAYER_IN_H){
				in_h_iter_prev = 0;
				in_w_iter_prev += LAYER_IN_W_T;
				if (in_w_iter_prev >= LAYER_IN_W){
					in_w_iter_prev = 0;
					layer_iter_prev += 1;
					layer_start_prev = 1;
					if (layer_iter_prev == LAYER_BATCH){
						layer_iter_prev = 0;
						out_num_iter_prev = 0;
						in_h_iter_prev = 0;
						in_w_iter_prev = 0;
						done_prev = 1;
					}
				}
			}
		}


	}


	
	if (LOAD_PREV_CIN){
		// Load the last tile to its FIFO
		if (task_cnt % 2 == 1){
			cin_load_fifo_write_prev(prev_cin_burst_buf_ping, fifo_cin_prev, LAYER_OUT_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, 1);
		} else {
			cin_load_fifo_write_prev(prev_cin_burst_buf_pong, fifo_cin_prev, LAYER_OUT_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, 1);
		}
	} 

}
