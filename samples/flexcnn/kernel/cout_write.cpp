/*
Function list:

cout_write_fifo_read
cout_write_ddr_write
cout_write

*/


/**
 * Function name: cout_write_fifo_read
 * Function description: This function reads cout data.
 */
void cout_write_fifo_read(
		bus_t0 cout_burst_buf[],
		hls::stream<PoolData0Type>  &fifo_cout,
		bool en,
		bool up_sample,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_H,
		uint LAYER_OUT_W,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_H_T,
		uint LAYER_IN_W_T,
		uint in_h_iter,
		uint in_w_iter
){
	PoolData0Type cout_buf[DATA_SEL_FACTOR0];
#pragma HLS ARRAY_PARTITION variable=cout_buf complete

#ifdef DEBUG
	ofstream cout_debug;
	cout_debug.open("hw_cout_write_patch.dat", ios::app);
#endif        

	uint write = 0;
	// Set up the writing mode
	if (en == 0 && up_sample == 0) write = 0; // normal writing
	else if (en == 1 && up_sample == 0) write = 1; // writing after pooling
	else if (up_sample == 1) write = 2; // writing after upsampling
  
	// Should store the data as Th * Tw * Tn
    int num = 0;
	switch(write){
	case 0:
	{
		int o = 0;
		int h = 0;
		int w = 0;
		bool done = 0;
		while(!done){
#pragma HLS PIPELINE II=1       
#pragma HLS DEPENDENCE INTER false variable=cout_burst_buf
			uint local_cout_idx = h * LAYER_IN_W_T * LAYER_OUT_NUM_T + w * LAYER_OUT_NUM_T + o * POOL_LANE;
			bus_t0 wide_tmp = cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0];
			for (int lane = 0; lane < DATA_SEL_FACTOR0; lane++){
#pragma HLS UNROLL
				cout_buf[lane] = wide_tmp(DATA_W0 * POOL_LANE - 1, 0);
				wide_tmp = wide_tmp >> DATA_W0 * POOL_LANE;
			}
			PoolData0Type tmp = fifo_cout.read();
			if (in_h_iter + h < LAYER_OUT_H && in_w_iter + w < LAYER_OUT_W)
				cout_buf[(local_cout_idx % BUS_PACK_FACTOR0) / POOL_LANE] = tmp;
			else
				cout_buf[(local_cout_idx % BUS_PACK_FACTOR0) / POOL_LANE] = tmp;

			bus_t0 wide_pack = (
#if DATA_SEL_FACTOR0 == 1
					cout_buf[0]
#elif DATA_SEL_FACTOR0 == 2
					cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 4
					cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 8
					cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
					cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 16
				    cout_buf[15], cout_buf[14], cout_buf[13], cout_buf[12],
				    cout_buf[11], cout_buf[10], cout_buf[9], cout_buf[8],
				    cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
				    cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#endif                  
			);
			cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0] = wide_pack;

			// Repeat until the whole tile is read
			w++;
			if (w == LAYER_IN_W_T){
				w = 0;
				h++;
				if (h == LAYER_IN_H_T){
					h = 0;
					o++;
					if (o == LAYER_OUT_NUM_T / POOL_LANE){
						o = 0;
						done = 1;
					}
				}
			}
		}
		//        }
		break;
	}
	case 1:
	{
		int o = 0;
		int h = 0;
		int w = 0;
		bool done = 0;
		while(!done){
#pragma HLS PIPELINE II=1
			//#pragma HLS DEPENDENCE INTER false variable=cout_burst_buf
			uint local_cout_idx = h * (LAYER_IN_W_T / 2) * LAYER_OUT_NUM_T + w * LAYER_OUT_NUM_T + o * POOL_LANE;
			bus_t0 wide_tmp = cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0];
			for (int lane = 0; lane < DATA_SEL_FACTOR0; lane++){
#pragma HLS UNROLL
				cout_buf[lane] = wide_tmp(DATA_W0 * POOL_LANE - 1, 0);
				wide_tmp = wide_tmp >> DATA_W0 * POOL_LANE;
			}
			PoolData0Type tmp = fifo_cout.read();
			if (in_h_iter / 2 + h < LAYER_OUT_H && in_w_iter / 2 + w < LAYER_OUT_W)
				cout_buf[(local_cout_idx % BUS_PACK_FACTOR0) / POOL_LANE] = tmp;
			else
				cout_buf[(local_cout_idx % BUS_PACK_FACTOR0) / POOL_LANE] = 0;
			bus_t0 wide_pack = (
#if DATA_SEL_FACTOR0 == 1
					cout_buf[0]
#elif DATA_SEL_FACTOR0 == 2
					cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 4
					cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 8
					cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
					cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 16
					cout_buf[15], cout_buf[14], cout_buf[13], cout_buf[12],
					cout_buf[11], cout_buf[10], cout_buf[9], cout_buf[8],
					cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
					cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#endif                  
			);
			cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0] = wide_pack;

			// Repeat until the whole tile is read
			w++;
			if (w == LAYER_IN_W_T / 2){
				w = 0;
				h++;
				if (h == LAYER_IN_H_T / 2){
					h = 0;
					o++;
					if (o == LAYER_OUT_NUM_T / POOL_LANE){
						o = 0;
						done = 1;
					}
				}
			}
		}
	}
	break;
	case 2:
	{
		int o = 0;
		int h = 0;
		int w = 0;
		bool done = 0;
		while(!done){
#pragma HLS PIPELINE II=1
			//#pragma HLS DEPENDENCE INTER false variable=cout_burst_buf
			uint local_cout_idx = h * (LAYER_IN_W_T * 2) * LAYER_OUT_NUM_T + w * LAYER_OUT_NUM_T + o * POOL_LANE;
			bus_t0 wide_tmp = cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0];
			for (int lane = 0; lane < DATA_SEL_FACTOR0; lane++){
#pragma HLS UNROLL
				cout_buf[lane] = wide_tmp(DATA_W0 * POOL_LANE - 1, 0);
				wide_tmp = wide_tmp >> DATA_W0 * POOL_LANE;
			}
			PoolData0Type tmp = fifo_cout.read();

			if (in_h_iter * 2 + h < LAYER_OUT_H && in_w_iter * 2 + w < LAYER_OUT_W)
				cout_buf[(local_cout_idx % BUS_PACK_FACTOR0) / POOL_LANE] = tmp;
			else
				cout_buf[(local_cout_idx % BUS_PACK_FACTOR0) / POOL_LANE] = 0;
			bus_t0 wide_pack = (
#if DATA_SEL_FACTOR0 == 1
					cout_buf[0]
#elif DATA_SEL_FACTOR0 == 2
					cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 4
					cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 8
					cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
					cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#elif DATA_SEL_FACTOR0 == 16
					cout_buf[15], cout_buf[14], cout_buf[13], cout_buf[12],
					cout_buf[11], cout_buf[10], cout_buf[9], cout_buf[8],
					cout_buf[7], cout_buf[6], cout_buf[5], cout_buf[4],
					cout_buf[3], cout_buf[2], cout_buf[1], cout_buf[0]
#endif                  
			);
			cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0] = wide_pack;
			num++;
#ifdef DEBUG_up_fifo
			cout << "after merge: " << local_cout_idx / BUS_PACK_FACTOR0 << " " ;
			for (int lane = 0; lane < CONV_LANE*2; lane++){
				ap_uint<DATA_W0> u32_tmp = wide_pack(DATA_W0 - 1, 0);
				cout << "lane: " << lane<< " " << Reinterpret<data_t0>(u32_tmp) << " ";
				wide_pack = wide_pack >> DATA_W0;
			}
			cout << endl;
#endif
			// Repeat until the whole tile is read
			w++;
			if (w == LAYER_IN_W_T * 2){
				w = 0;
				h++;
				if (h == LAYER_IN_H_T * 2){
					h = 0;
					o++;
					if (o == LAYER_OUT_NUM_T / POOL_LANE){
						o = 0;
						done = 1;
					}
				}
			}
		}
	}
	break;
	}
}

/**
 * Function name: cout_write_ddr_write
 * Function description: This function writes out cout results to off-chip DRAM.
 */
void cout_write_ddr_write(
		bus_t0 cout_burst_buf[],
		bus_t0 *global_cout,
		bool en,
		bool up_sample,
		uint num_iter,
		uint in_h_iter,
		uint in_w_iter,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_NUM,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_H_T,
		uint LAYER_IN_W_T,
		uint LAYER_OUT_H_HW,
		uint LAYER_OUT_W_HW,
		uint num_tile,
		uint ind_w_t,
		uint ind_w,
		uint cout_offset,
		bool change_layout,
		bool run
){
	// Set up the writing mode
	uint write = 0;
	if (up_sample == 1) write = 2; // writing after upsampling
	else if (en == 0) write = 0; // normal writing
	else if (en == 1) write = 1; // writing after pooling
	// The default data layout is ceil(N / Tn) * H * ceil(W / Tw) * Tw * Tn
	// If filter size is 1, the data layout should change to ceil(N / Tn) * ceil(H / Th) * ceil(W / Tw) * Th * Tw * Tn
	if (change_layout) write += 3; 
  
  
	if (run){
		switch(write){
		case 0:
		{
			// write out
			for (int hh  = 0; hh < LAYER_IN_H_T; hh++){
				uint h = in_h_iter + hh;
				uint global_cout_idx = num_iter / LAYER_OUT_NUM_T * LAYER_OUT_H_HW * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + h * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + in_w_iter * LAYER_OUT_NUM_T + cout_offset;
				uint local_cout_idx = hh * LAYER_IN_W_T * LAYER_OUT_NUM_T;
				memcpy((void*)&global_cout[global_cout_idx / BUS_PACK_FACTOR0], (void*)&cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_W_T * LAYER_OUT_NUM_T);
			}
		}
		break;
		case 1:
		{
			for (int hh = 0; hh < LAYER_IN_H_T / 2; hh++){
				uint h = in_h_iter / 2 + hh;
				uint global_cout_idx;
				global_cout_idx = num_iter / LAYER_OUT_NUM_T * LAYER_OUT_H_HW * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + h * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + in_w_iter / 2 * LAYER_OUT_NUM_T + cout_offset;
				uint local_cout_idx = hh * LAYER_IN_W_T / 2 * LAYER_OUT_NUM_T;
				memcpy((void*)&global_cout[global_cout_idx / BUS_PACK_FACTOR0], (void*)&cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_W_T / 2 * LAYER_OUT_NUM_T);
			}
		}
		break;
		case 2:
		{
			for (int hh = 0; hh < LAYER_IN_H_T * 2; hh++){
				uint h = in_h_iter * 2 + hh;
				uint global_cout_idx;
				global_cout_idx = num_iter / LAYER_OUT_NUM_T * LAYER_OUT_H_HW * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + h * LAYER_OUT_W_HW * LAYER_OUT_NUM_T + in_w_iter * 2 * LAYER_OUT_NUM_T + cout_offset;
				uint local_cout_idx = hh * LAYER_IN_W_T * 2 * LAYER_OUT_NUM_T;
#ifdef DEBUG_cout_ddr_up
				cout << "global:" << global_cout_idx << " h: " << h << " in_h_iter: " << in_h_iter << " local: " << local_cout_idx << endl;
#endif
				memcpy((void*)&global_cout[global_cout_idx / BUS_PACK_FACTOR0], (void*)&cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_W_T * 2 * LAYER_OUT_NUM_T);
			}
		}
		break;
		case 3:
		{
			// write out
			for (int hh  = 0; hh < 1; hh++){
				uint global_cout_idx = cout_offset + num_tile * LAYER_OUT_NUM_T * LAYER_IN_W_T * LAYER_IN_H_T;
				uint local_cout_idx = hh * LAYER_IN_W_T * LAYER_OUT_NUM_T;
				memcpy((void*)&global_cout[global_cout_idx / BUS_PACK_FACTOR0], (void*)&cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_W_T * LAYER_OUT_NUM_T * LAYER_IN_H_T);
			}
		}
		break;
		case 4:
		{
			for (int hh = 0; hh < LAYER_IN_H_T / 2; hh++){
				uint global_cout_idx = 0;
				if (LAYER_IN_W_T / 2 == LAYER_OUT_W_HW) {
					global_cout_idx = cout_offset + ind_w * LAYER_OUT_NUM_T * LAYER_IN_W_T * LAYER_IN_H_T / 4 + num_tile * LAYER_OUT_NUM_T * LAYER_IN_W_T * LAYER_IN_H_T / 4 + hh * LAYER_IN_W_T / 2 * LAYER_OUT_NUM_T;
				} else {
					global_cout_idx = cout_offset + ind_w * LAYER_OUT_NUM_T * LAYER_IN_W_T * LAYER_IN_H_T / 2 + num_tile * LAYER_OUT_NUM_T * LAYER_IN_W_T * LAYER_IN_H_T / 2 + ind_w_t * LAYER_OUT_NUM_T * LAYER_IN_W_T / 2 + hh * LAYER_IN_W_T * LAYER_OUT_NUM_T;
				}
				uint local_cout_idx = hh * LAYER_IN_W_T / 2 * LAYER_OUT_NUM_T;
				memcpy((void*)&global_cout[global_cout_idx / BUS_PACK_FACTOR0], (void*)&cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_W_T / 2 * LAYER_OUT_NUM_T);
			}
		}
		break;
		case 5:
		{
			for (int hh = 0; hh < 1; hh++){
				uint global_cout_idx = cout_offset + num_tile * LAYER_OUT_NUM_T * LAYER_IN_W_T * LAYER_IN_H_T * 4;
				uint local_cout_idx = hh * LAYER_IN_W_T * 2 * LAYER_OUT_NUM_T;
#ifdef DEBUG_cout_ddr_up
				cout << "global:" << global_cout_idx << " h: " << h << " in_h_iter: " << in_h_iter << " local: " << local_cout_idx << endl;
#endif
				memcpy((void*)&global_cout[global_cout_idx / BUS_PACK_FACTOR0], (void*)&cout_burst_buf[local_cout_idx / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_W_T * 2 * LAYER_OUT_NUM_T * LAYER_IN_H_T * 2);
			}
		}
		break;
		}
	}
}





/**
 * Function name: cout_write
 * Function description: This function collects and writes out cout results.
 */
void cout_write(
		hls::stream<PoolData0Type>  &fifo_cout,
		hls::stream<ConfigInst>     &fifo_config_in,
		bus_t0                      *global_cout
){
	bus_t0 cout_burst_buf_ping[OUT_H_T * OUT_W_T * OUT_NUM_T * 3 / 2 / BUS_PACK_FACTOR0];
	bus_t0 cout_burst_buf_pong[OUT_H_T * OUT_W_T * OUT_NUM_T * 3 / 2 / BUS_PACK_FACTOR0];
#pragma HLS RESOURCE variable=cout_burst_buf_ping core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=cout_burst_buf_pong core=XPM_MEMORY uram  

	// iterators
	uint num_iter = 0;
	uint in_h_iter = 0;
	uint in_w_iter = 0;
	uint layer_iter = 0;

	uint cout_offset = 0;

	uint num_iter_prev = 0;
	uint in_h_iter_prev = 0;
	uint in_w_iter_prev = 0;

	// parameters
	// inst0
	ap_uint<32> LAYER_IN_NUM_HW;
	ap_uint<32> LAYER_OUT_NUM_HW;
	ap_uint<32> LAYER_IN_H_HW;
	ap_uint<32> LAYER_IN_W_HW;
	ap_uint<32> LAYER_OUT_H_HW;
	ap_uint<32> LAYER_OUT_W_HW;
	// inst1
	ap_uint<32> LAYER_IN_NUM;
	ap_uint<32> LAYER_OUT_NUM;
	ap_uint<32> LAYER_IN_H;
	ap_uint<32> LAYER_IN_W;
	ap_uint<32> LAYER_OUT_H;
	ap_uint<32> LAYER_OUT_W;
	// inst2
	ap_uint<32> CIN_OFFSET;
	ap_uint<32> WEIGHT_OFFSET;
	ap_uint<32> BIAS_OFFSET;
	ap_uint<32> COUT_OFFSET;
	ap_uint<16> FILTER_S1;
	ap_uint<16> FILTER_S2;
	ap_uint<32> STRIDE;
	// inst3
	ap_uint<32> LAYER_EN;
	ap_uint<32> PREV_CIN_OFFSET;
	ap_uint<16> LAYER_IN_NUM_T;
	ap_uint<16> LAYER_OUT_NUM_T;
	ap_uint<32> LAYER_IN_H_T;
	ap_uint<32> LAYER_IN_W_T;
	ap_uint<1>  CONV_1ST_EN;
	ap_uint<1>  DEPTH_CONV_EN;
	ap_uint<1>  CONV_EN;
	ap_uint<1>  RELU_EN;
	ap_uint<1>  RELU6_EN;
	ap_uint<1>  POOL_EN;
	ap_uint<1>  UP_SAMPLE_EN;
	ap_uint<1>  BIAS_EN;
	ap_uint<1>  INTER_LOAD_EN;
	ap_uint<1>  INTER_WRITE_EN;

	// Read instructions
	ConfigInst inst0 = fifo_config_in.read();
	ConfigInst inst1 = fifo_config_in.read();
	ConfigInst inst2 = fifo_config_in.read();
	ConfigInst inst3 = fifo_config_in.read();
	ConfigInst inst4 = fifo_config_in.read();

	ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

	bool en_prev;
	bool max_pool_prev;
	bool up_sample_prev;
	bool change_layout;
	uint LAYER_IN_NUM_prev;
	uint LAYER_OUT_NUM_prev;
	uint LAYER_IN_NUM_T_prev;
	uint LAYER_OUT_NUM_T_prev;
	uint LAYER_IN_H_T_prev;
	uint LAYER_IN_W_T_prev;
	uint LAYER_OUT_H_HW_prev;
	uint LAYER_OUT_W_HW_prev;
	uint cout_offset_prev;

	bool write_done = 0;
	uint task_cnt = 0;
	uint iter_h = 1;
	uint num_tile = 0;
	uint ind_w = 0;
	uint ind_w_t = 0;
	uint num_tile_prev = 0;
	uint ind_w_t_prev = 0;
	uint ind_w_prev = 0;
	bool layer_start = 0;
	bool done = 0;
	bool change_layout_prev = 0;
	// We assum that cin has been pre-padded with zeros
	while(!done){

		if (layer_start){
			inst0 = fifo_config_in.read();
			inst1 = fifo_config_in.read();
			inst2 = fifo_config_in.read();
			inst3 = fifo_config_in.read();
			inst4 = fifo_config_in.read();
			layer_start = 0;
		}

		// Refer to cin_load module to understand the meaning of the instructions
		// inst0
		LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
		LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
		LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
		LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
		LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
		LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
		// inst1
		LAYER_IN_NUM     = inst1(32*0+31, 32*0);
		LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
		LAYER_IN_H       = inst1(32*2+31, 32*2);
		LAYER_IN_W       = inst1(32*3+31, 32*3);
		LAYER_OUT_H      = inst1(32*4+31, 32*4);
		LAYER_OUT_W      = inst1(32*5+31, 32*5);
		// inst2
		CIN_OFFSET       = inst2(32*0+31, 32*0);
		WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
		BIAS_OFFSET      = inst2(32*2+31, 32*2);
		COUT_OFFSET      = inst2(32*3+31, 32*3);
		FILTER_S1        = inst2(32*4+15, 32*4);
		FILTER_S2        = inst2(32*4+31, 32*4+16);
		STRIDE           = inst2(32*5+31, 32*5);
		// inst3
		LAYER_EN         = inst3(32*0+31, 32*0);
		PREV_CIN_OFFSET  = inst3(32*1+31, 32*1);
		LAYER_IN_NUM_T   = inst3(32*2+15, 32*2);
		LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2+16);
		LAYER_IN_H_T     = inst3(32*3+31, 32*3);
		LAYER_IN_W_T     = inst3(32*4+31, 32*4);

		CONV_1ST_EN    = LAYER_EN[0];
		DEPTH_CONV_EN  = LAYER_EN[1];
		CONV_EN        = LAYER_EN[2];
		RELU_EN        = LAYER_EN[3];
		RELU6_EN       = LAYER_EN[4];
		POOL_EN        = LAYER_EN[5];
		UP_SAMPLE_EN   = LAYER_EN[6];  // reserved
		BIAS_EN        = LAYER_EN[7];
		INTER_LOAD_EN  = LAYER_EN[8];
		INTER_WRITE_EN = LAYER_EN[9];

		// Set up some configuration signals
		cout_offset = COUT_OFFSET;
		bool en = POOL_EN || (POOL_EN == 0 && STRIDE == 2);
		bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
		bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
		bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);
		bool up_sample = UP_SAMPLE_EN;
		change_layout = (((LAYER_OUT_W_HW == LAYER_OUT_W) || (LAYER_OUT_W_HW == LAYER_IN_W_T)) && ((LAYER_OUT_H_HW == LAYER_OUT_H) || (LAYER_OUT_H_HW == LAYER_IN_H_T))); // if next filter = 1 : change the layout to num_tile, h_t, w_t, in_t
		
		// If it is supposed to store the result in DRAM
		if (INTER_WRITE_EN == 0){

			if (task_cnt == 0){
				// First, read the data of the first tile from FIFO
				cout_write_fifo_read(
						cout_burst_buf_ping, fifo_cout, en, up_sample,
						LAYER_IN_NUM, LAYER_OUT_H, LAYER_OUT_W,
						LAYER_IN_NUM_T, LAYER_OUT_NUM_T,
						LAYER_IN_H_T, LAYER_IN_W_T,
						in_h_iter, in_w_iter
				);
			} else {
				// Apply double buffering for reading the data from FIFO and writing to DRAM
				if (task_cnt % 2 == 1){
					cout_write_fifo_read(
							cout_burst_buf_pong, fifo_cout, en, up_sample,
							LAYER_IN_NUM, LAYER_OUT_H, LAYER_OUT_W,
							LAYER_IN_NUM_T, LAYER_OUT_NUM_T,
							LAYER_IN_H_T, LAYER_IN_W_T,
							in_h_iter, in_w_iter
					);

					cout_write_ddr_write(
							cout_burst_buf_ping, global_cout,
							en_prev, up_sample_prev,
							num_iter_prev, in_h_iter_prev, in_w_iter_prev,
							LAYER_IN_NUM_prev, LAYER_OUT_NUM_prev,
							LAYER_IN_NUM_T_prev, LAYER_OUT_NUM_T_prev,
							LAYER_IN_H_T_prev, LAYER_IN_W_T_prev,
							LAYER_OUT_H_HW_prev, LAYER_OUT_W_HW_prev,
							num_tile_prev,
							ind_w_t_prev,
							ind_w_prev,
							cout_offset_prev,
							change_layout_prev,
							!write_done
					);
				} else {
					cout_write_fifo_read(
							cout_burst_buf_ping, fifo_cout, en, up_sample,
							LAYER_IN_NUM, LAYER_OUT_H, LAYER_OUT_W,
							LAYER_IN_NUM_T, LAYER_OUT_NUM_T,
							LAYER_IN_H_T, LAYER_IN_W_T,
							in_h_iter, in_w_iter
					);

					cout_write_ddr_write(
							cout_burst_buf_pong, global_cout,
							en_prev, up_sample_prev,
							num_iter_prev, in_h_iter_prev, in_w_iter_prev,
							LAYER_IN_NUM_prev, LAYER_OUT_NUM_prev,
							LAYER_IN_NUM_T_prev, LAYER_OUT_NUM_T_prev,
							LAYER_IN_H_T_prev, LAYER_IN_W_T_prev,
							LAYER_OUT_H_HW_prev, LAYER_OUT_W_HW_prev,
							num_tile_prev,
							ind_w_t_prev,
							ind_w_prev,
							cout_offset_prev,
							change_layout_prev,
							!write_done
					);
				}
			}

			if (task_cnt > 0){
				write_done = 1;
			}
			
			// need to know the config of the current tile in the next iteration since we are using double buffering
			task_cnt++;
			num_iter_prev = num_iter;
			in_h_iter_prev = in_h_iter;
			in_w_iter_prev = in_w_iter;
			en_prev = en;
			up_sample_prev = up_sample;
			max_pool_prev = max_pool;
			LAYER_IN_NUM_prev = LAYER_IN_NUM;
			LAYER_OUT_NUM_prev = LAYER_OUT_NUM;
			LAYER_IN_NUM_T_prev = LAYER_IN_NUM_T;
			LAYER_OUT_NUM_T_prev = LAYER_OUT_NUM_T;
			LAYER_IN_H_T_prev = LAYER_IN_H_T;
			LAYER_IN_W_T_prev = LAYER_IN_W_T;
			LAYER_OUT_H_HW_prev = LAYER_OUT_H_HW;
			LAYER_OUT_W_HW_prev = LAYER_OUT_W_HW;
			cout_offset_prev = cout_offset;
			num_tile_prev = num_tile;
			ind_w_t_prev = ind_w_t;
			ind_w_prev = ind_w;
			change_layout_prev = change_layout;
			write_done = 0;
			
		}

		// Repeat until all the tiles are stored
		if (max_pool || up_sample){
			//num_tile = task_cnt - 1;
			num_tile = task_cnt;
			num_iter += LAYER_IN_NUM_T;
			if (num_iter >= LAYER_IN_NUM){
				num_iter = 0;
				in_h_iter += LAYER_IN_H_T;
				if (in_h_iter >= LAYER_IN_H){
					in_h_iter = 0;
					in_w_iter += LAYER_IN_W_T;
					if (in_w_iter >= LAYER_IN_W){
						in_w_iter = 0;
						layer_iter += 1;
						layer_start = 1;
						if (layer_iter == LAYER_BATCH){
							layer_iter = 0;
							done = 1;
						}
					}
				}
			}
		} else if (STRIDE == 2){
			num_tile += 1;
			if (num_tile == (LAYER_IN_H / LAYER_IN_H_T)){
				num_tile = 0;
				ind_w_t += 1;
				if (LAYER_IN_W_T / 2 == LAYER_OUT_W_HW) ind_w_t += 1;
				if (ind_w_t == 2){
					ind_w_t = 0;
					ind_w += (LAYER_IN_H / LAYER_IN_H_T);
				}
			}
			in_h_iter += LAYER_IN_H_T;
			if (in_h_iter >= LAYER_IN_H){
				in_h_iter = 0;
				in_w_iter += LAYER_IN_W_T;
				if (in_w_iter >= LAYER_IN_W){
					in_w_iter = 0;
					num_iter += LAYER_OUT_NUM_T;
					if (num_iter >= LAYER_OUT_NUM){
						num_iter = 0;
						layer_iter += 1;
						layer_start = 1;
						if (layer_iter == LAYER_BATCH){
							layer_iter = 0;
							done = 1;
						}
					}
				}
			}
		} else {
			//num_tile = task_cnt - 1;
			num_tile = task_cnt;
			in_h_iter += LAYER_IN_H_T;
			if (in_h_iter >= LAYER_IN_H){
				in_h_iter = 0;
				in_w_iter += LAYER_IN_W_T;
				if (in_w_iter >= LAYER_IN_W){
					in_w_iter = 0;
					num_iter += LAYER_OUT_NUM_T;
					if (num_iter >= LAYER_OUT_NUM){
						num_iter = 0;
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

	// Store the last tile
	if (INTER_WRITE_EN == 0){
		if (task_cnt % 2 == 1){
			cout_write_ddr_write(
					cout_burst_buf_ping, global_cout,
					en_prev, up_sample_prev,
					num_iter_prev, in_h_iter_prev, in_w_iter_prev,
					LAYER_IN_NUM_prev, LAYER_OUT_NUM_prev,
					LAYER_IN_NUM_T_prev, LAYER_OUT_NUM_T_prev,
					LAYER_IN_H_T_prev, LAYER_IN_W_T_prev,
					LAYER_OUT_H_HW_prev, LAYER_OUT_W_HW_prev,
					num_tile_prev,
					ind_w_t_prev,
					ind_w_prev,
					cout_offset_prev,
					change_layout_prev,
					!write_done
			);
		} else {
			cout_write_ddr_write(
					cout_burst_buf_pong, global_cout,
					en_prev, up_sample_prev,
					num_iter_prev, in_h_iter_prev, in_w_iter_prev,
					LAYER_IN_NUM_prev, LAYER_OUT_NUM_prev,
					LAYER_IN_NUM_T_prev, LAYER_OUT_NUM_T_prev,
					LAYER_IN_H_T_prev, LAYER_IN_W_T_prev,
					LAYER_OUT_H_HW_prev, LAYER_OUT_W_HW_prev,
					num_tile_prev,
					ind_w_t_prev,
					ind_w_prev,
					cout_offset_prev,
					change_layout_prev,
					!write_done
			);
		}
	}
}

