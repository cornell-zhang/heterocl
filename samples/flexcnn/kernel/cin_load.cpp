/*
Function list:

cin_load_ddr_read
cin_load_fifo_write
cin_load

*/

/*
 * Function name: cin_load_ddr_read
 * Function description: This function loads cin results from off-chip DRAM.
 *                       Two modes are enabled. If the whole feature maps of the layer could fit
 *                       in the on-chip buffer, they will be loaded as a whole. Otherwise, each time,
 *                       LAYER_IN_NUM_T * (LAYER_IN_W_T + FILTER_S - 1) of data are loaded.
 */
void cin_load_ddr_read(
		bus_t0  *global_cin,
		bus_t0  cin_burst_buf[],
		uint    LAYER_IN_H_HW,
		uint    LAYER_IN_W_HW,
		uint    LAYER_IN_NUM_T,
		uint    LAYER_IN_H_T,
		uint    LAYER_IN_W_T,
		uint    FILTER_S,
		uint    cin_offset,
		uint    in_num_iter,
		uint    in_h_iter,
		uint    in_w_iter,
		uint    num_tile,
		bool    change,
		bool    max_pool,
		bool    write
){
	if ((LAYER_IN_H_HW <= IN_H_T + K_T - 1) && (LAYER_IN_W_HW <= IN_W_T + K_T - 1) && !max_pool){
		// Load the patch as a whole if it fits in on-chip buffer.
		uint global_cin_offset = in_num_iter * LAYER_IN_H_HW * LAYER_IN_W_HW + cin_offset;
		memcpy((void*)cin_burst_buf, (void*)&global_cin[global_cin_offset / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_NUM_T * LAYER_IN_H_HW * LAYER_IN_W_HW);
	} else {
		// Read the data based on the data layout used.
		// If filter size is 1, the data layout is ceil(N / Tn) * ceil(H / Th) * ceil(W / Tw) * Th * Tw * Tn
		// o.w. ceil(N / Tn) * H * ceil(W / Tw) * Tw * Tn
		// The data in on-chip buffer will have the data layout of Th * Tw * Tn
		if (change && FILTER_S == 1){
			for (int hh = 0; hh < /*LAYER_IN_H_T + FILTER_S - */1; hh++){
				uint local_cin_offset = 0;
				uint global_cin_offset = cin_offset + num_tile * (LAYER_IN_W_T) * (LAYER_IN_H_T) * LAYER_IN_NUM_T;
#ifdef DEBUG_cin
				if(write)
					cout << global_cin_offset << endl;
#endif
				memcpy((void*)&cin_burst_buf[local_cin_offset / BUS_PACK_FACTOR0], (void*)&global_cin[global_cin_offset / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_NUM_T * (LAYER_IN_W_T) * (LAYER_IN_H_T));

			}
		} else {
			for (int hh = 0; hh < LAYER_IN_H_T + FILTER_S - 1; hh++){
				uint h = in_h_iter + hh;
				uint local_cin_offset = hh * (LAYER_IN_W_T + FILTER_S - 1) * LAYER_IN_NUM_T;
				uint global_cin_offset = in_num_iter * LAYER_IN_H_HW * LAYER_IN_W_HW + h * LAYER_IN_W_HW * LAYER_IN_NUM_T + in_w_iter * LAYER_IN_NUM_T + cin_offset;
#ifdef DEBUG_cin
				if(write)
					cout << global_cin_offset << endl;
#endif
				memcpy((void*)&cin_burst_buf[local_cin_offset / BUS_PACK_FACTOR0], (void*)&global_cin[global_cin_offset / BUS_PACK_FACTOR0], sizeof(data_t0) * LAYER_IN_NUM_T * (LAYER_IN_W_T + FILTER_S - 1));

			}
		}

	}
}

/*
 * Function name: cin_load_fifo_write
 * Function description: This function writes cin data to the downstream modules.
 */
void cin_load_fifo_write(
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
// Since each entry in FIFOs will be SIMD_LANE elements of the data, we should unpack based on SIMD_LANE
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
		//count++;
#ifdef DEBUG_load_change
		if(FILTER_S == 1){
			cout << " cin ";
			for (int lane = 0; lane < RELU_LANE; lane++){
#pragma HLS UNROLL
				ap_uint<DATA_W0> u32_beta = fifo_cin_data(DATA_W0 - 1, 0);
				data_t2 a = Reinterpret<data_t2>(u32_beta);
				fifo_cin_data = fifo_cin_data >> DATA_W0;
				cout << a << " ";
			}
			cout << endl;
		}
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
 * Function description: This function loads and distributes cin and instructions.
 */
void cin_load(
		bus_t0                         *global_cin,
		uint                           config[CONFIG_PARAMS],
		hls::stream<CinLoadData0Type>  &fifo_cin,
		hls::stream<ConfigInst>        &fifo_config_out
){
#pragma HLS INLINE off 
	// on-chip buffer for cin data
	bus_t0 cin_burst_buf_ping[IN_NUM_T * (IN_H_T + K_T - 1) * (IN_W_T + K_T - 1) / BUS_PACK_FACTOR0];
	bus_t0 cin_burst_buf_pong[IN_NUM_T * (IN_H_T + K_T - 1) * (IN_W_T + K_T - 1) / BUS_PACK_FACTOR0];
#pragma HLS RESOURCE variable=cin_burst_buf_ping core=XPM_MEMORY uram
#pragma HLS RESOURCE variable=cin_burst_buf_pong core=XPM_MEMORY uram  



	// layer batch
	ap_uint<32> LAYER_BATCH = config[25];

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
	ap_uint<16> LAYER_IN_NUM_T;
	ap_uint<16> LAYER_OUT_NUM_T;
	ap_uint<32> LAYER_IN_H_T;
	ap_uint<32> LAYER_IN_W_T;
	ap_uint<32> PREV_CIN_OFFSET;
	// inst4
	ap_uint<32> LAYER_TASK_NUM1;
	ap_uint<32> LAYER_TASK_NUM2;
	ap_uint<32> LAYER_LOCAL_ACCUM_NUM;
	ap_uint<32> LAYER_LOCAL_REG_NUM;
	ap_uint<32> LAYER_ROW_IL_FACTOR;
	ap_uint<32> LAYER_COL_IL_FACTOR;

	ap_uint<1>  CONV_1ST_EN;
	ap_uint<1>  LOAD_PREV_CIN;

	uint LAYER_IN_NUM_T_prev;
	uint LAYER_OUT_NUM_T_prev;
	uint LAYER_IN_H_T_prev;
	uint LAYER_IN_W_T_prev;
	uint FILTER_S_prev;

	uint task_cnt = 0;
	bool layer_start = 1;
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
	uint inter_tile = 0;
	uint channel_iter = 0;
	while(!done){
		// Read and extract the parameters/config from the instructions
		// Refer to pose.h or the README of the repo to find how the instructions are made
		// inst0 : The hardware sizes of each dimension (the sizes after tiling is applied)
		LAYER_IN_NUM_HW  = config[0 + layer_iter * CONFIG_PARAMS];
		LAYER_OUT_NUM_HW = config[1 + layer_iter * CONFIG_PARAMS];
		LAYER_IN_H_HW    = config[2 + layer_iter * CONFIG_PARAMS];
		LAYER_IN_W_HW    = config[3 + layer_iter * CONFIG_PARAMS];
		LAYER_OUT_H_HW   = config[4 + layer_iter * CONFIG_PARAMS];
		LAYER_OUT_W_HW   = config[5 + layer_iter * CONFIG_PARAMS];

		// inst1 : The actual sizes of each dimension
		LAYER_IN_NUM  = config[6 + layer_iter * CONFIG_PARAMS];
		LAYER_OUT_NUM = config[7 + layer_iter * CONFIG_PARAMS];
		LAYER_IN_H    = config[8 + layer_iter * CONFIG_PARAMS];
		LAYER_IN_W    = config[9 + layer_iter * CONFIG_PARAMS];
		LAYER_OUT_H   = config[10 + layer_iter * CONFIG_PARAMS];
		LAYER_OUT_W   = config[11 + layer_iter * CONFIG_PARAMS];

		// inst2 : The DRAM locations for reading/writing the data of this layer + Filter and Stride sizes
		CIN_OFFSET    = config[12 + layer_iter * CONFIG_PARAMS];
		WEIGHT_OFFSET = config[13 + layer_iter * CONFIG_PARAMS];
		BIAS_OFFSET   = config[14 + layer_iter * CONFIG_PARAMS];
		COUT_OFFSET   = config[15 + layer_iter * CONFIG_PARAMS];
		FILTER_S1     = config[16 + layer_iter * CONFIG_PARAMS];
		FILTER_S2     = config[17 + layer_iter * CONFIG_PARAMS];
		STRIDE        = config[18 + layer_iter * CONFIG_PARAMS];

		// inst3 : The enable signlas of the modules + DRAM location of the input to the previous layer + Tile sizes
		LAYER_EN        = config[19 + layer_iter * CONFIG_PARAMS]; // contains the enable signals for the modules
		PREV_CIN_OFFSET = config[20 + layer_iter * CONFIG_PARAMS];
		LAYER_IN_NUM_T  = config[21 + layer_iter * CONFIG_PARAMS];
		LAYER_OUT_NUM_T = config[22 + layer_iter * CONFIG_PARAMS];
		LAYER_IN_H_T    = config[23 + layer_iter * CONFIG_PARAMS];
		LAYER_IN_W_T    = config[24 + layer_iter * CONFIG_PARAMS];

		CONV_1ST_EN    = LAYER_EN[0];
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
		ap_uint<1>  BATCH_NORM_EN_DEPTH  = LAYER_EN[12];
		LOAD_PREV_CIN  = LAYER_EN[11];

		// inst4 : The info needed to run the systolic array
		LAYER_TASK_NUM1       = config[26 + layer_iter * CONFIG_PARAMS];
		LAYER_TASK_NUM2       = config[27 + layer_iter * CONFIG_PARAMS];
		LAYER_LOCAL_ACCUM_NUM = config[28 + layer_iter * CONFIG_PARAMS];
		LAYER_LOCAL_REG_NUM   = config[29 + layer_iter * CONFIG_PARAMS];
		LAYER_ROW_IL_FACTOR   = config[30 + layer_iter * CONFIG_PARAMS];
		LAYER_COL_IL_FACTOR   = config[31 + layer_iter * CONFIG_PARAMS];

#ifdef DEBUG_config_cin
		cout << LAYER_IN_NUM_HW << " " << LAYER_OUT_NUM_HW << " " << LAYER_IN_H_HW << " " << LAYER_IN_W_HW << " " << LAYER_OUT_H_HW << " " << LAYER_OUT_W_HW << endl;
		cout << LAYER_IN_NUM << " " << LAYER_OUT_NUM << " " << LAYER_IN_H << " " << LAYER_IN_W << " " << LAYER_OUT_H << " " << LAYER_OUT_W << endl;
		cout << CIN_OFFSET << " " << WEIGHT_OFFSET << " " << BIAS_OFFSET << " " << COUT_OFFSET << " " << FILTER_S1 << " " << FILTER_S2 << " " << STRIDE << endl;
		cout << LAYER_EN << " " << PREV_CIN_OFFSET << " " << LAYER_IN_NUM_T << " " << LAYER_OUT_NUM_T << " " << LAYER_IN_H_T << " " << LAYER_IN_W_T << endl;
#endif
		
		// Pack the parameters to pass them throught the config FIFOs
		ConfigInst inst0 = (LAYER_OUT_W_HW, LAYER_OUT_H_HW, LAYER_IN_W_HW, LAYER_IN_H_HW, LAYER_OUT_NUM_HW, LAYER_IN_NUM_HW);
		ConfigInst inst1 = (LAYER_OUT_W, LAYER_OUT_H, LAYER_IN_W, LAYER_IN_H, LAYER_OUT_NUM, LAYER_IN_NUM);
		ConfigInst inst2 = (STRIDE, FILTER_S2, FILTER_S1, COUT_OFFSET, BIAS_OFFSET, WEIGHT_OFFSET, CIN_OFFSET);
		ConfigInst inst3 = (LAYER_BATCH, LAYER_IN_W_T, LAYER_IN_H_T, LAYER_OUT_NUM_T, LAYER_IN_NUM_T, PREV_CIN_OFFSET, LAYER_EN);
		ConfigInst inst4 = (LAYER_COL_IL_FACTOR, LAYER_ROW_IL_FACTOR, LAYER_LOCAL_REG_NUM, LAYER_LOCAL_ACCUM_NUM, LAYER_TASK_NUM2, LAYER_TASK_NUM1);

		// Pass the config/instructions
		if (layer_start){
			fifo_config_out.write(inst0);
			fifo_config_out.write(inst1);
			fifo_config_out.write(inst2);
			fifo_config_out.write(inst3);
			fifo_config_out.write(inst4);
			layer_start = 0;
		}

		// offsets
		uint cin_offset = CIN_OFFSET;
		uint prev_cin_offset = PREV_CIN_OFFSET;


		if (prev == 1) start_prev = 1;

		// set up some configuration signals
		uint FILTER_S = (DEPTH_CONV_EN == 1)? (uint)FILTER_S1: ((CONV_EN == 1)? (uint)FILTER_S2: 1);
		bool separable_conv = (DEPTH_CONV_EN == 1) && (CONV_EN == 1);
		bool conv2d = (DEPTH_CONV_EN == 0) && (CONV_EN == 1);
		bool max_pool = (DEPTH_CONV_EN == 0) && (CONV_EN == 0);
		change_layout = (((LAYER_IN_W_HW == LAYER_IN_W) || (LAYER_IN_W_HW == LAYER_IN_W_T)) && ((LAYER_IN_H_HW == LAYER_IN_H) || (LAYER_IN_H_HW == LAYER_IN_H_T))); // if next filter = 1 : change the layout to num_tile, Th, Tw, Tn

		// If it has to read from DRAM and not the stored data in on-chip storage
		if (INTER_LOAD_EN == 0){
			if ((max_pool && out_num_iter == 0) || separable_conv || conv2d || (UP_SAMPLE_EN && out_num_iter == 0)){
				if (task_cnt == 0){
					// first load cin
					cin_load_ddr_read(global_cin, cin_burst_buf_ping, LAYER_IN_H_HW, LAYER_IN_W_HW, LAYER_IN_NUM_T, LAYER_IN_H_T, LAYER_IN_W_T, FILTER_S, cin_offset, in_num_iter, in_h_iter, in_w_iter, num_tile, change_layout, max_pool, 0);
					
				} else {
					// Apply double buffering for reading the data and filling the FIFO
					if (task_cnt % 2 == 1){
						cin_load_ddr_read(global_cin, cin_burst_buf_pong, LAYER_IN_H_HW, LAYER_IN_W_HW, LAYER_IN_NUM_T, LAYER_IN_H_T, LAYER_IN_W_T, FILTER_S, cin_offset, in_num_iter, in_h_iter, in_w_iter, num_tile, change_layout, max_pool, 0);
						cin_load_fifo_write(cin_burst_buf_ping, fifo_cin, LAYER_IN_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, FILTER_S_prev);
					} else {
						cin_load_ddr_read(global_cin, cin_burst_buf_ping, LAYER_IN_H_HW, LAYER_IN_W_HW, LAYER_IN_NUM_T, LAYER_IN_H_T, LAYER_IN_W_T, FILTER_S, cin_offset, in_num_iter, in_h_iter, in_w_iter, num_tile, change_layout, max_pool, 0);
						cin_load_fifo_write(cin_burst_buf_pong, fifo_cin, LAYER_IN_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, FILTER_S_prev);
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

		// Continue until all the tiles are read
		// Since each layer produces LAYER_OUT_NUM feature maps, 
		// repeat reading the tiles LAYER_OUT_NUM times
		in_num_iter += LAYER_IN_NUM_T;
		if (in_num_iter < LAYER_IN_NUM){
			channel_iter += ((LAYER_IN_W / LAYER_IN_W_T) * (LAYER_IN_H / LAYER_IN_H_T));
		} else {
			channel_iter = 0;
			inter_tile++;
		}
		num_tile = channel_iter + inter_tile;
		if (in_num_iter >= LAYER_IN_NUM){
			in_num_iter = 0;
			channel_iter = 0;
			in_h_iter += LAYER_IN_H_T;
			if (in_h_iter >= LAYER_IN_H){
				in_h_iter = 0;
				in_w_iter += LAYER_IN_W_T;
				if (in_w_iter >= LAYER_IN_W){
					in_w_iter = 0;
					out_num_iter += LAYER_OUT_NUM_T;
					num_tile = 0;
					inter_tile = 0;
					channel_iter = 0;
					if (out_num_iter >= LAYER_OUT_NUM){
						out_num_iter = 0;
						layer_iter += 1;
						prev = 0;
						layer_start = 1;
						if (layer_iter == LAYER_BATCH){
							layer_iter = 0;
							out_num_iter = 0;
							in_h_iter = 0;
							in_w_iter = 0;
							done = 1;
						}
					}
				}
			}
		}
	}


	// Fill the FIFOs with the data for the last tile
	if (task_cnt % 2 == 1){
		cin_load_fifo_write(cin_burst_buf_ping, fifo_cin, LAYER_IN_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, FILTER_S_prev);
	} else {
		cin_load_fifo_write(cin_burst_buf_pong, fifo_cin, LAYER_IN_NUM_T_prev, LAYER_IN_H_T_prev, LAYER_IN_W_T_prev, FILTER_S_prev);
	}

}
