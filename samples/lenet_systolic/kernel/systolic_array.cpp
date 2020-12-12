#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"

// common headers
#include <stdio.h>
#include <string.h>

using namespace hls;

#define cal_aligned_size(x,y) ((x+y-1)/y*y)

typedef ap_uint<192> U1_ConfigInst;

// Data types
typedef float U1_data_t0;
typedef ap_uint<512> U1_bus_t0;
#define U1_DATA0_WIDTH 32
#define U1_DATA0_PACK_FACTOR (512/U1_DATA0_WIDTH)
typedef float U1_data_t1;
typedef ap_uint<512> U1_bus_t1;
#define U1_DATA1_WIDTH 32
#define U1_DATA1_PACK_FACTOR (512/U1_DATA1_WIDTH)
typedef float U1_data_t2;
typedef ap_uint<512> U1_bus_t2;
#define U1_DATA2_WIDTH 32
#define U1_DATA2_PACK_FACTOR (512/U1_DATA2_WIDTH)
typedef unsigned int uint;
union ufloat{
	float f;
	unsigned int u;
};

// Macros
#define U1_IN_IMG_W_T 50
#define U1_OUT_NUM 960
#define U1_IN_IMG_H_T 14
#define U1_OUT_IMG_H 384
#define U1_LAYER_BATCH 2
#define U1_STRIDE 1
#define U1_K 3
#define U1_OUT_NUM_T 96
#define U1_IN_IMG_W 386
#define U1_IN_IMG_H 386
#define U1_IN_NUM_T 96
#define U1_IN_NUM 960
#define U1_OUT_IMG_H_T 12
#define U1_OUT_IMG_W_T 48
#define U1_OUT_IMG_W 384
#define U1_DATA0_SIZE 143036160
#define U1_DATA0_SIZE_ALIGNED (cal_aligned_size(143036160, U1_DATA0_PACK_FACTOR))
#define U1_DATA1_SIZE 8294400
#define U1_DATA1_SIZE_ALIGNED (cal_aligned_size(8294400, U1_DATA1_PACK_FACTOR))
#define U1_DATA2_SIZE 141557760
#define U1_DATA2_SIZE_ALIGNED (cal_aligned_size(141557760, U1_DATA2_PACK_FACTOR))

#define U1_ROW_IL_FACTOR 12
#define U1_COL_IL_FACTOR 6
#define U1_SA_ROWS 8
#define U1_SA_COLS 8
#define U1_LOCAL_REG_NUM 864
#define U1_LOCAL_ACCUM_NUM 108
#define U1_SIMD_FACTOR 8
#define U1_DATA0_FC_SIMD_FACTOR 8
#define U1_DATA0_FC_GROUP_FACTOR 1
#define U1_DATA0_FC_SPLIT_FACTOR 1
#define U1_DATA1_FC_SIMD_FACTOR 8
#define U1_DATA1_FC_GROUP_FACTOR 1
#define U1_DATA1_FC_SPLIT_FACTOR 1
#define U1_DATA2_FC_SIMD_FACTOR 8
#define U1_DATA2_FC_GROUP_FACTOR 1
#define U1_DATA2_FC_SPLIT_FACTOR 1

#define U1_DATA0_BUF_SIZE 10752
#define U1_DATA0_HEAD_BUF_SIZE 672000
#define U1_DATA1_BUF_SIZE 10368
#define U1_DATA1_HEAD_BUF_SIZE 829440
#define U1_DATA2_BUF_SIZE 6912
#define U1_DATA2_HEAD_BUF_SIZE 552960
    

// Functions and structs
struct U1_Data0TransferChannelType{
	U1_Data0TransferChannelType(){}
	U1_Data0TransferChannelType(
			ap_uint<U1_DATA0_WIDTH*U1_DATA0_FC_SIMD_FACTOR> data_t,
			unsigned int feeder_id_t,
			bool new_pair_t,
			bool last_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		feeder_id = feeder_id_t;
		new_pair = new_pair_t;
		last_pair = last_pair_t;
		FILTER_S = filter_s_t;
	}
	ap_uint<U1_DATA0_WIDTH*U1_DATA0_FC_SIMD_FACTOR> data;
	unsigned int feeder_id;
	bool new_pair;
	bool last_pair;
	unsigned int FILTER_S;
};

struct U1_Data1TransferChannelType{
	U1_Data1TransferChannelType(){}
	U1_Data1TransferChannelType(
			ap_uint<U1_DATA1_WIDTH*U1_DATA1_FC_SIMD_FACTOR> data_t,
			unsigned int feeder_id_t,
			bool new_pair_t,
			bool last_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		feeder_id = feeder_id_t;
		new_pair = new_pair_t;
		last_pair = last_pair_t;
		FILTER_S = filter_s_t;
	}
	ap_uint<U1_DATA1_WIDTH*U1_DATA1_FC_SIMD_FACTOR> data;
	unsigned int feeder_id;
	bool new_pair;
	bool last_pair;
	unsigned int FILTER_S;
};

struct U1_Data2TransferChannelType{
	U1_Data2TransferChannelType(){}
	U1_Data2TransferChannelType(
			ap_uint<U1_DATA2_WIDTH*U1_DATA2_FC_SIMD_FACTOR> data_t){
		data = data_t;
	}
	ap_uint<U1_DATA2_WIDTH*U1_DATA2_FC_SIMD_FACTOR> data;
};

struct U1_Data0PEChannelType{
	U1_Data0PEChannelType(){}
	U1_Data0PEChannelType(
			ap_uint<256> data_t
	){
		data = data_t;
	}
	U1_Data0PEChannelType(
			ap_uint<256> data_t,
			bool new_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		new_pair = new_pair_t;
		FILTER_S = filter_s_t;
	}
	U1_Data0PEChannelType(
			ap_uint<256> data_t,
			bool new_pair_t,
			bool last_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		new_pair = new_pair_t;
		last_pair = last_pair_t;
		FILTER_S = filter_s_t;
	}
	ap_uint<256> data;
	bool new_pair;
	bool last_pair;
	unsigned int FILTER_S;
};

typedef ap_uint<256> U1_Data0SIMDType;

struct U1_Data1PEChannelType{
	U1_Data1PEChannelType(){}
	U1_Data1PEChannelType(
			ap_uint<256> data_t
	){
		data = data_t;
	}
	U1_Data1PEChannelType(
			ap_uint<256> data_t,
			bool new_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		new_pair = new_pair_t;
		FILTER_S = filter_s_t;
	}
	U1_Data1PEChannelType(
			ap_uint<256> data_t,
			bool new_pair_t,
			bool last_pair_t,
			unsigned int filter_s_t
	){
		data = data_t;
		new_pair = new_pair_t;
		last_pair = last_pair_t;
		FILTER_S = filter_s_t;
	}
	ap_uint<256> data;
	bool new_pair;
	bool last_pair;
	unsigned int FILTER_S;
};

typedef ap_uint<256> U1_Data1SIMDType;

struct U1_Data2PEChannelType{
	U1_Data2PEChannelType(){}
	U1_Data2PEChannelType(
			U1_data_t2 data_t){
		data = data_t;
	}
	U1_data_t2 data;
};

void U1_DataFeed0Head_Shim(
		U1_bus_t0* cin,
		stream<ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> > &fifo_transfer_cin,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_NUM,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H,
		uint LAYER_IN_IMG_W,
		uint LAYER_OUT_IMG_H,
		uint LAYER_OUT_IMG_W,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint LAYER_FILTER_S,
		uint LAYER_BATCH,
		uint LAYER_STRIDE,
		stream<U1_ConfigInst> &fifo_kernel_config_out
);

void U1_DataFeed0Head(
		stream<ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> > &fifo_transfer_in,
		stream<U1_Data0TransferChannelType> &fifo_transfer_out0,
		stream<U1_ConfigInst> &fifo_kernel_config_in,
		stream<U1_ConfigInst> &fifo_kernel_config_out,
		stream<uint> &fifo_config_out0,
		stream<uint> &fifo_config_out1
);

void U1_DataFeed1Head_Shim(
		U1_bus_t1* weight,
		stream<ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> > &fifo_transfer_weight,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_NUM,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H,
		uint LAYER_IN_IMG_W,
		uint LAYER_OUT_IMG_H,
		uint LAYER_OUT_IMG_W,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint LAYER_FILTER_S,
		uint LAYER_BATCH,
		uint LAYER_STRIDE
);

void U1_DataFeed1Head(
		stream<ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> > &fifo_transfer_in,
		stream<U1_Data1TransferChannelType> &fifo_transfer_out0,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
);

void U1_DataCollect2Head_Shim(
		U1_bus_t2* cout,
		stream<ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> > &fifo_transfer_cout,
		stream<U1_ConfigInst> &fifo_kernel_config_in,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_NUM,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H,
		uint LAYER_IN_IMG_W,
		uint LAYER_OUT_IMG_H,
		uint LAYER_OUT_IMG_W,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint LAYER_FILTER_S,
		uint LAYER_STRIDE
);

void U1_DataCollect2Head(
		stream<ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> > &fifo_transfer_out,
		stream<U1_Data2TransferChannelType> &fifo_transfer_in0,
		stream<uint> &fifo_config_in
);

void U1_DataFeed0Engine0_wrapper(
		stream<U1_Data0TransferChannelType> &fifo_transfer_in,
		stream<U1_Data0TransferChannelType> &fifo_transfer_out,
		stream<U1_Data0PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out0,
		stream<uint> &fifo_config_out1
);

void U1_DataFeed0EngineLast(
		stream<U1_Data0TransferChannelType> &fifo_transfer_in,
		stream<U1_Data0PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out1
);

void U1_DataFeed1Engine0_wrapper(
		stream<U1_Data1TransferChannelType> &fifo_transfer_in,
		stream<U1_Data1TransferChannelType> &fifo_transfer_out,
		stream<U1_Data1PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out0
);

void U1_DataFeed1EngineLast(
		stream<U1_Data1TransferChannelType> &fifo_transfer_in,
		stream<U1_Data1PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in
);

void U1_DataCollect2Engine0_wrapper(
		stream<U1_Data2TransferChannelType> &fifo_transfer_in,
		stream<U1_Data2TransferChannelType> &fifo_transfer_out,
		stream<U1_Data2PEChannelType> &fifo_collect_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in0,
		stream<uint> &fifo_config_in1,
		stream<uint> &fifo_config_out
);

void U1_DataCollect2EngineLast(
		stream<U1_Data2TransferChannelType> &fifo_transfer_out,
		stream<U1_Data2PEChannelType> &fifo_collect_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in0,
		stream<uint> &fifo_config_out
);

void U1_kernel(
		U1_bus_t0* cin,
		U1_bus_t1* weight,
		U1_bus_t2* cout,
		bool init,
		unsigned int FILTER_S
);

//template<typename To, typename From>
//inline To Reinterpret(const From& val){
//  return reinterpret_cast<const To&>(val);
//}

template<class data_t, class bus_t, int WIDTH>
data_t data_select(
		bus_t bus_data,
		uint offset
){
	data_t ret;
	ret = Reinterpret<data_t>((ap_uint<WIDTH>)bus_data(WIDTH-1 + offset*WIDTH, offset*WIDTH));
	return ret;
}
/**
 *  This file is automatically generated by PolySA CodeGen.
 *  Version: 1.0
 *  Authos: Jie Wang
 */

//#include "common_header_U1.h"
//#include <iostream>
//using namespace std;

void U1_DataFeed0Head_Shim(
		U1_bus_t0* cin,
		stream<ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> > &fifo_transfer_cin,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_NUM,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H,
		uint LAYER_IN_IMG_W,
		uint LAYER_OUT_IMG_H,
		uint LAYER_OUT_IMG_W,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint LAYER_FILTER_S,
		uint LAYER_STRIDE,
		uint LAYER_BATCH,
		stream<U1_ConfigInst> &fifo_kernel_config_out
){
#pragma HLS INLINE off

	uint LAYER_TASK_NUM1 = (LAYER_IN_NUM / LAYER_IN_NUM_T) * (LAYER_OUT_NUM / LAYER_OUT_NUM_T) * (LAYER_OUT_IMG_H / LAYER_IN_IMG_H_T * LAYER_STRIDE) * (LAYER_OUT_IMG_W / LAYER_IN_IMG_W_T * LAYER_STRIDE);
	uint LAYER_TASK_NUM2 = (LAYER_OUT_NUM / LAYER_OUT_NUM_T) * (LAYER_OUT_IMG_H / LAYER_IN_IMG_H_T * LAYER_STRIDE) * (LAYER_OUT_IMG_W / LAYER_IN_IMG_W_T * LAYER_STRIDE);
	uint LAYER_LOCAL_ACCUM_NUM = LAYER_IN_NUM_T / U1_SIMD_FACTOR * LAYER_FILTER_S * LAYER_FILTER_S;
	uint LAYER_LOCAL_REG_NUM = (LAYER_IN_IMG_H_T / LAYER_STRIDE) * (LAYER_IN_IMG_W_T / U1_SA_COLS / LAYER_STRIDE) * LAYER_OUT_NUM_T / U1_SA_ROWS;
	uint LAYER_ROW_IL_FACTOR = LAYER_OUT_NUM_T / U1_SA_ROWS;
	uint LAYER_COL_IL_FACTOR = LAYER_IN_IMG_W_T / U1_SA_COLS / LAYER_STRIDE;

	ap_uint<32> CIN_OFFSET = 0;
	ap_uint<32> WEIGHT_OFFSET = 0;
	ap_uint<32> BIAS_OFFSET = 0;
	ap_uint<32> COUT_OFFSET = 0;
	ap_uint<16> FILTER_S1 = LAYER_FILTER_S;
	ap_uint<16> FILTER_S2 = LAYER_FILTER_S;
	ap_uint<32> STRIDE = LAYER_STRIDE;
	ap_uint<32> LAYER_EN = 0;
	ap_uint<32> LAYER_IN_NUM_cast = LAYER_IN_NUM;
	ap_uint<32> LAYER_OUT_NUM_cast = LAYER_OUT_NUM;
	ap_uint<32> LAYER_IN_NUM_T_cast = LAYER_IN_NUM_T;
	ap_uint<32> LAYER_OUT_NUM_T_cast = LAYER_OUT_NUM_T;
	ap_uint<32> LAYER_IN_IMG_H_T_cast = LAYER_IN_IMG_H_T;
	ap_uint<32> LAYER_IN_IMG_W_T_cast = LAYER_IN_IMG_W_T;
	ap_uint<32> LAYER_IN_IMG_H_cast = LAYER_IN_IMG_H;
	ap_uint<32> LAYER_IN_IMG_W_cast = LAYER_IN_IMG_W;
	ap_uint<32> LAYER_OUT_IMG_H_cast = LAYER_OUT_IMG_H;
	ap_uint<32> LAYER_OUT_IMG_W_cast = LAYER_OUT_IMG_W;
	ap_uint<32> LAYER_BATCH_cast = LAYER_BATCH;

	ap_uint<32> LAYER_TASK_NUM1_cast = LAYER_TASK_NUM1;
	ap_uint<32> LAYER_TASK_NUM2_cast = LAYER_TASK_NUM2;
	ap_uint<32> LAYER_LOCAL_ACCUM_NUM_cast = LAYER_LOCAL_ACCUM_NUM;
	ap_uint<32> LAYER_LOCAL_REG_NUM_cast = LAYER_LOCAL_REG_NUM;
	ap_uint<32> LAYER_ROW_IL_FACTOR_cast = LAYER_ROW_IL_FACTOR;
	ap_uint<32> LAYER_COL_IL_FACTOR_cast = LAYER_COL_IL_FACTOR;

	U1_bus_t0 cin_buf[U1_DATA0_HEAD_BUF_SIZE / U1_DATA0_PACK_FACTOR];
	ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> sel_tmp[U1_DATA0_PACK_FACTOR / U1_DATA0_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sel_tmp complete dim=1

	for (ap_uint<2> layer_iter = 0; layer_iter < LAYER_BATCH; layer_iter++){
		U1_ConfigInst inst0 = (LAYER_OUT_IMG_W_cast, LAYER_OUT_IMG_H_cast, LAYER_IN_IMG_W_cast, LAYER_IN_IMG_H_cast, LAYER_OUT_NUM_cast, LAYER_IN_NUM_cast);
		U1_ConfigInst inst1 = (LAYER_OUT_IMG_W_cast, LAYER_OUT_IMG_H_cast, LAYER_IN_IMG_W_cast, LAYER_IN_IMG_H_cast, LAYER_OUT_NUM_cast, LAYER_IN_NUM_cast);
		U1_ConfigInst inst2 = (STRIDE, FILTER_S2, FILTER_S1, COUT_OFFSET, BIAS_OFFSET, WEIGHT_OFFSET, CIN_OFFSET);
		U1_ConfigInst inst3 = (LAYER_BATCH_cast, LAYER_IN_IMG_W_T_cast, LAYER_IN_IMG_H_T_cast, LAYER_OUT_NUM_T_cast, LAYER_IN_NUM_T_cast, LAYER_EN);
		U1_ConfigInst inst4 = (LAYER_COL_IL_FACTOR_cast, LAYER_ROW_IL_FACTOR_cast, LAYER_LOCAL_REG_NUM_cast, LAYER_LOCAL_ACCUM_NUM_cast, LAYER_TASK_NUM2_cast, LAYER_TASK_NUM1_cast);

		fifo_kernel_config_out.write(inst0);
		fifo_kernel_config_out.write(inst1);
		fifo_kernel_config_out.write(inst2);
		fifo_kernel_config_out.write(inst3);
		fifo_kernel_config_out.write(inst4);

		for (int out_img_h_t = 0; out_img_h_t < LAYER_OUT_IMG_H; out_img_h_t += LAYER_IN_IMG_H_T / LAYER_STRIDE){
			for (int out_img_w_t = 0; out_img_w_t < LAYER_OUT_IMG_W; out_img_w_t += LAYER_IN_IMG_W_T / LAYER_STRIDE){
				for (int out_num_t = 0; out_num_t < LAYER_OUT_NUM; out_num_t += LAYER_OUT_NUM_T){
					uint chunk_offset = out_img_h_t * LAYER_IN_IMG_W * LAYER_IN_NUM;
					for (int in_img_h_t = 0; in_img_h_t < LAYER_IN_IMG_H_T + LAYER_FILTER_S - 1; in_img_h_t++){
						uint local_chunk_offset = chunk_offset + in_img_h_t * LAYER_IN_IMG_W * LAYER_IN_NUM + out_img_w_t * LAYER_IN_NUM;
						memcpy((void*)(cin_buf + in_img_h_t * (LAYER_IN_IMG_W_T + LAYER_FILTER_S - 1) * LAYER_IN_NUM / U1_DATA0_PACK_FACTOR), (void*)(cin + local_chunk_offset / U1_DATA0_PACK_FACTOR), sizeof(U1_data_t0) * (LAYER_IN_IMG_W_T + LAYER_FILTER_S - 1) * LAYER_IN_NUM);
					}
					for (int in_num_t = 0; in_num_t < LAYER_IN_NUM; in_num_t += LAYER_IN_NUM_T){
						for (int ii = 0; ii < LAYER_IN_NUM_T / U1_DATA0_FC_SIMD_FACTOR; ii++){
							for (int hh = 0; hh < LAYER_IN_IMG_H_T + LAYER_FILTER_S - 1; hh++){
								for (int ww = 0; ww < LAYER_IN_IMG_W_T + LAYER_FILTER_S - 1; ww++){
#pragma HLS PIPELINE II=1
									uint cin_local_idx = hh * (LAYER_IN_IMG_W_T + LAYER_FILTER_S - 1) * LAYER_IN_NUM + ww * LAYER_IN_NUM + (in_num_t + ii * U1_DATA0_FC_SIMD_FACTOR);
									uint cin_bus_idx = cin_local_idx / U1_DATA0_PACK_FACTOR;
									uint cin_bus_offset = cin_local_idx % U1_DATA0_PACK_FACTOR;
									U1_bus_t0 bus_data = cin_buf[cin_bus_idx];
									ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> fifo_cin_data;
									for (ap_uint<2> s = 0; s < U1_DATA0_PACK_FACTOR / U1_DATA0_FC_SIMD_FACTOR; s++){
#pragma HLS UNROLL
										sel_tmp[s] = bus_data(U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR - 1, 0);
										bus_data = bus_data >> (U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR);
									}
									fifo_cin_data = sel_tmp[cin_bus_offset / U1_DATA0_FC_SIMD_FACTOR];
									fifo_transfer_cin.write(fifo_cin_data);
								}
							}
						}
					}
				}
			}
		}
	}
}

void U1_DataFeed0Head(
		stream<ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> > &fifo_transfer_in,
		stream<U1_Data0TransferChannelType> &fifo_transfer_out0,
		stream<U1_ConfigInst> &fifo_kernel_config_in,
		stream<U1_ConfigInst> &fifo_kernel_config_out,
		stream<uint> &fifo_config_out0,
		stream<uint> &fifo_config_out1
){
#pragma HLS INLINE off
#pragma HLS DATA_PACK variable=fifo_transfer_out0

	// loader buffer
	ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> cin_buf[U1_IN_NUM_T * U1_IN_IMG_H_T * U1_IN_IMG_W_T / U1_DATA0_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=cin_buf dim=1 block factor=1

	// Read instructions
	U1_ConfigInst inst0 = fifo_kernel_config_in.read();
	fifo_kernel_config_out.write(inst0);
	U1_ConfigInst inst1 = fifo_kernel_config_in.read();
	fifo_kernel_config_out.write(inst1);
	U1_ConfigInst inst2 = fifo_kernel_config_in.read();
	fifo_kernel_config_out.write(inst2);
	U1_ConfigInst inst3 = fifo_kernel_config_in.read();
	fifo_kernel_config_out.write(inst3);
	U1_ConfigInst inst4 = fifo_kernel_config_in.read();
	fifo_kernel_config_out.write(inst4);
	ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

	ap_uint<2> layer_iter = 0;
	bool done1 = 0;
	while(!done1){
		if (layer_iter > 0){
			// Read instructions
			inst0 = fifo_kernel_config_in.read();
			fifo_kernel_config_out.write(inst0);
			inst1 = fifo_kernel_config_in.read();
			fifo_kernel_config_out.write(inst1);
			inst2 = fifo_kernel_config_in.read();
			fifo_kernel_config_out.write(inst2);
			inst3 = fifo_kernel_config_in.read();
			fifo_kernel_config_out.write(inst3);
			inst4 = fifo_kernel_config_in.read();
			fifo_kernel_config_out.write(inst4);
		}
		ap_uint<32> EXT_LAYER_IN_NUM_HW  = inst0(32*0+31, 32*0);
		ap_uint<32> EXT_LAYER_OUT_NUM_HW = inst0(32*1+31, 32*1);
		ap_uint<32> EXT_LAYER_IN_H_HW    = inst0(32*2+31, 32*2);
		ap_uint<32> EXT_LAYER_IN_W_HW    = inst0(32*3+31, 32*3);
		ap_uint<32> EXT_LAYER_OUT_H_HW   = inst0(32*4+31, 32*4);
		ap_uint<32> EXT_LAYER_OUT_W_HW   = inst0(32*5+31, 32*5);
		// inst1
		ap_uint<32> EXT_LAYER_IN_NUM     = inst1(32*0+31, 32*0);
		ap_uint<32> EXT_LAYER_OUT_NUM    = inst1(32*1+31, 32*1);
		ap_uint<32> EXT_LAYER_IN_H       = inst1(32*2+31, 32*2);
		ap_uint<32> EXT_LAYER_IN_W       = inst1(32*3+31, 32*3);
		ap_uint<32> EXT_LAYER_OUT_H      = inst1(32*4+31, 32*4);
		ap_uint<32> EXT_LAYER_OUT_W      = inst1(32*5+31, 32*5);
		// inst2
		ap_uint<32> EXT_CIN_OFFSET       = inst2(32*0+31, 32*0);
		ap_uint<32> EXT_WEIGHT_OFFSET    = inst2(32*1+31, 32*1);
		ap_uint<32> EXT_BIAS_OFFSET      = inst2(32*2+31, 32*2);
		ap_uint<32> EXT_COUT_OFFSET      = inst2(32*3+31, 32*3);
		ap_uint<16> EXT_FILTER_S1        = inst2(32*4+15, 32*4);
		ap_uint<16> EXT_FILTER_S2        = inst2(32*4+31, 32*4+16);
		ap_uint<32> EXT_STRIDE           = inst2(32*5+31, 32*5);
		// inst3
		ap_uint<32> EXT_LAYER_EN         = inst3(32*0+31, 32*0);
		ap_uint<32> EXT_PREV_CIN_OFFSET  = inst3(32*1+32, 32*1);
		ap_uint<16> EXT_LAYER_IN_NUM_T   = inst3(32*2+15, 32*2);
		ap_uint<16> EXT_LAYER_OUT_NUM_T  = inst3(32*2+31, 32*2+16);
		ap_uint<32> EXT_LAYER_IN_IMG_H_T = inst3(32*3+31, 32*3);
		ap_uint<32> EXT_LAYER_IN_IMG_W_T = inst3(32*4+31, 32*4);
		ap_uint<1>  EXT_CONV_1ST_EN      = EXT_LAYER_EN[0];
		ap_uint<1>  EXT_DEPTH_CONV_EN    = EXT_LAYER_EN[1];
		ap_uint<1>  EXT_CONV_EN          = EXT_LAYER_EN[2];
		ap_uint<1>  EXT_RELU_EN          = EXT_LAYER_EN[3];
		ap_uint<1>  EXT_RELU6_EN         = EXT_LAYER_EN[4];
		ap_uint<1>  EXT_POOL_EN          = EXT_LAYER_EN[5];

		ap_uint<32> EXT_LAYER_TASK_NUM1        = inst4(32*0+31, 32*0);
		ap_uint<32> EXT_LAYER_TASK_NUM2        = inst4(32*1+31, 32*1);
		ap_uint<32> EXT_LAYER_LOCAL_ACCUM_NUM  = inst4(32*2+31, 32*2);
		ap_uint<32> EXT_LAYER_LOCAL_REG_NUM    = inst4(32*3+31, 32*3);
		ap_uint<32> EXT_LAYER_ROW_IL_FACTOR    = inst4(32*4+31, 32*4);
		ap_uint<32> EXT_LAYER_COL_IL_FACTOR    = inst4(32*5+31, 32*5);

		uint EXT_FILTER_S = (EXT_CONV_EN == 1)? (uint)EXT_FILTER_S2: 1;
		bool separable_conv = (EXT_DEPTH_CONV_EN == 1) && (EXT_CONV_EN == 1);
		bool conv2d = (EXT_DEPTH_CONV_EN == 0) && (EXT_CONV_EN == 1);
		bool max_pool = (EXT_DEPTH_CONV_EN == 0) && (EXT_CONV_EN == 0);
		uint stride1 = (EXT_DEPTH_CONV_EN == 0)? 1 : (uint)EXT_STRIDE;
		uint stride2 = (EXT_DEPTH_CONV_EN == 0)? (uint)EXT_STRIDE : 1;

		uint LAYER_IN_IMG_H = (EXT_DEPTH_CONV_EN == 1)? (uint)EXT_LAYER_IN_H_HW - (uint)EXT_FILTER_S1 + 1: (uint)EXT_LAYER_IN_H_HW;
		uint LAYER_IN_IMG_W = (EXT_DEPTH_CONV_EN == 1)? (uint)EXT_LAYER_IN_W_HW - (uint)EXT_FILTER_S1 + 1: (uint)EXT_LAYER_IN_W_HW;
		uint LAYER_OUT_IMG_H = EXT_LAYER_OUT_H;
		uint LAYER_OUT_IMG_W = EXT_LAYER_OUT_W;
		uint LAYER_IN_NUM = EXT_LAYER_IN_NUM_HW;
		uint LAYER_OUT_NUM = EXT_LAYER_OUT_NUM_HW;
		uint LAYER_IN_NUM_T = EXT_LAYER_IN_NUM_T;
		//cout << LAYER_IN_NUM_T << endl;
		uint LAYER_OUT_NUM_T = EXT_LAYER_OUT_NUM_T;
		uint LAYER_IN_IMG_H_T;
		uint LAYER_IN_IMG_W_T;
		if (stride1 == 1){
			LAYER_IN_IMG_H_T = EXT_LAYER_IN_IMG_H_T;
			LAYER_IN_IMG_W_T = EXT_LAYER_IN_IMG_W_T;
		} else if (stride1 == 2){
			LAYER_IN_IMG_H_T = EXT_LAYER_IN_IMG_H_T / 2;
			LAYER_IN_IMG_W_T = EXT_LAYER_IN_IMG_W_T / 2;
		}
		uint LAYER_FILTER_S = EXT_FILTER_S2;
		uint LAYER_STRIDE = stride2;

		uint LAYER_TASK_NUM1 = EXT_LAYER_TASK_NUM1;
		uint LAYER_TASK_NUM2 = EXT_LAYER_TASK_NUM2;
		uint LAYER_LOCAL_ACCUM_NUM = EXT_LAYER_LOCAL_ACCUM_NUM;
		uint LAYER_LOCAL_REG_NUM = EXT_LAYER_LOCAL_REG_NUM;
		uint LAYER_ROW_IL_FACTOR = EXT_LAYER_ROW_IL_FACTOR;
		uint LAYER_COL_IL_FACTOR = EXT_LAYER_COL_IL_FACTOR;

		// write out configurations
		fifo_config_out0.write(LAYER_IN_NUM_T);
		fifo_config_out0.write(LAYER_OUT_NUM_T);
		fifo_config_out0.write(LAYER_IN_IMG_H_T);
		fifo_config_out0.write(LAYER_IN_IMG_W_T);
		fifo_config_out0.write(LAYER_FILTER_S);
		fifo_config_out0.write(LAYER_TASK_NUM1);
		fifo_config_out0.write(LAYER_TASK_NUM2);
		fifo_config_out0.write(LAYER_LOCAL_ACCUM_NUM);
		fifo_config_out0.write(LAYER_LOCAL_REG_NUM);
		fifo_config_out0.write(LAYER_ROW_IL_FACTOR);
		fifo_config_out0.write(LAYER_COL_IL_FACTOR);
		fifo_config_out0.write(LAYER_STRIDE);
		fifo_config_out0.write(LAYER_BATCH);

		fifo_config_out1.write(LAYER_IN_NUM);
		fifo_config_out1.write(LAYER_IN_NUM_T);
		fifo_config_out1.write(LAYER_OUT_NUM_T);
		fifo_config_out1.write(LAYER_IN_IMG_H_T);
		fifo_config_out1.write(LAYER_IN_IMG_W_T);
		fifo_config_out1.write(LAYER_FILTER_S);
		fifo_config_out1.write(LAYER_TASK_NUM1);
		fifo_config_out1.write(LAYER_TASK_NUM2);
		fifo_config_out1.write(LAYER_LOCAL_ACCUM_NUM);
		fifo_config_out1.write(LAYER_LOCAL_REG_NUM);
		fifo_config_out1.write(LAYER_ROW_IL_FACTOR);
		fifo_config_out1.write(LAYER_COL_IL_FACTOR);
		fifo_config_out1.write(LAYER_STRIDE);
		fifo_config_out1.write(LAYER_BATCH);

		ap_uint<29> task_iter = 0;
		ap_uint<11> in_num_t = 0;
		bool done2 = 0;
		while(!done2){
			if (LAYER_FILTER_S > 1){
				bool done3 = 0;
				ap_uint<11> ii = 0;
				ap_uint<10> hh = 0;
				ap_uint<10> ww = 0;
				while(!done3){
#pragma HLS PIPELINE II=1
					uint cin_local_idx = hh *  (LAYER_IN_IMG_W_T + LAYER_FILTER_S - 1) * LAYER_IN_NUM_T + ww * LAYER_IN_NUM_T + ii * U1_DATA0_FC_SIMD_FACTOR;
					cin_buf[cin_local_idx / U1_DATA0_FC_SIMD_FACTOR] = fifo_transfer_in.read();
					ww++;
					if (ww == LAYER_IN_IMG_W_T + LAYER_FILTER_S - 1){
						ww = 0;
						hh++;
						if (hh == LAYER_IN_IMG_H_T + LAYER_FILTER_S - 1){
							hh = 0;
							ii++;
							if (ii == LAYER_IN_NUM_T / U1_DATA0_FC_SIMD_FACTOR){
								ii = 0;
								done3 = 1;
							}
						}
					}
				}
			}
			bool init_final = (in_num_t == 0);
			bool last = (in_num_t == (LAYER_IN_NUM - LAYER_IN_NUM_T));
			// write to SA
			ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> sel_tmp0[U1_DATA0_PACK_FACTOR / U1_DATA0_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sel_tmp0 complete dim=1
			ap_uint<11> t3 = 0;
			ap_uint<10> t1 = 0;
			ap_uint<4> t0 = 0;
			ap_uint<7> t2 = 0;
			bool done4 = 0;
			while(!done4){
#pragma HLS PIPELINE II=1
				uint local_in_img_w = t0 * (LAYER_IN_IMG_W_T / U1_SA_COLS) + t2;
				uint local_in_num = in_num_t + t3 * U1_DATA0_FC_SIMD_FACTOR;
				uint local_in_img_h = t1;
				uint feeder_id = t0 / U1_DATA0_FC_GROUP_FACTOR;
				ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> wide_data0;
				if (LAYER_FILTER_S > 1){
					uint cin_local_index = local_in_img_h * (LAYER_IN_IMG_W_T + LAYER_FILTER_S - 1) * LAYER_IN_NUM_T + local_in_img_w * LAYER_IN_NUM_T + t3 * U1_DATA0_FC_SIMD_FACTOR;
					uint cin_bus_index = cin_local_index / U1_DATA0_FC_SIMD_FACTOR;
					wide_data0 = cin_buf[cin_bus_index];
				} else {
					wide_data0 = fifo_transfer_in.read();
				}
				fifo_transfer_out0.write(U1_Data0TransferChannelType(
						wide_data0,
						(uint)feeder_id, init_final, last, LAYER_FILTER_S));

				t2++;
				if (t2 == LAYER_IN_IMG_W_T / U1_SA_COLS + LAYER_FILTER_S - 1){
					t2 = 0;
					t0++;
					if (t0 == U1_SA_COLS / U1_DATA0_FC_SPLIT_FACTOR){
						t0 = 0;
						t1++;
						if (t1 == LAYER_IN_IMG_H_T + LAYER_FILTER_S - 1){
							t1 = 0;
							t3++;
							if (t3 == LAYER_IN_NUM_T / U1_DATA0_FC_SIMD_FACTOR){
								t3 = 0;
								done4 = 1;
							}
						}
					}
				}
			}

			in_num_t += LAYER_IN_NUM_T;
			if (in_num_t == LAYER_IN_NUM){
				in_num_t = 0;
				task_iter++;
				if (task_iter == LAYER_TASK_NUM2){
					task_iter = 0;
					done2 = 1;
				}
			}
		}
		layer_iter++;
		if (layer_iter == LAYER_BATCH){
			layer_iter = 0;
			done1 = 1;
		}
	}
}

void U1_DataFeed1Head_Shim(
		U1_bus_t1* weight,
		stream<ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> > &fifo_transfer_weight,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_NUM,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H,
		uint LAYER_IN_IMG_W,
		uint LAYER_OUT_IMG_H,
		uint LAYER_OUT_IMG_W,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint FILTER_S,
		uint LAYER_STRIDE,
		uint LAYER_BATCH
){
#pragma HLS INLINE off
	U1_bus_t1 weight_buf[U1_DATA1_HEAD_BUF_SIZE / U1_DATA1_PACK_FACTOR];
	ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> sel_tmp[U1_DATA1_PACK_FACTOR / U1_DATA1_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sel_tmp complete dim=1

	for (ap_uint<2> layer_iter = 0; layer_iter < LAYER_BATCH; layer_iter++){
		for (int out_img_h_t = 0; out_img_h_t < LAYER_OUT_IMG_H; out_img_h_t += LAYER_IN_IMG_H_T / LAYER_STRIDE){
			for (int out_img_w_t = 0; out_img_w_t < LAYER_OUT_IMG_W; out_img_w_t += LAYER_IN_IMG_W_T / LAYER_STRIDE){
				for (int out_num_t = 0; out_num_t < LAYER_OUT_NUM; out_num_t += LAYER_OUT_NUM_T){
					uint chunk_offset = out_num_t * FILTER_S * FILTER_S * LAYER_IN_NUM;
					memcpy((void*)weight_buf, (void*)(weight + chunk_offset / U1_DATA1_PACK_FACTOR), sizeof(U1_data_t1) * LAYER_OUT_NUM_T * FILTER_S * FILTER_S * LAYER_IN_NUM);
					for (int in_num_t = 0; in_num_t < LAYER_IN_NUM; in_num_t += LAYER_IN_NUM_T){
						for (int oo =0; oo < LAYER_OUT_NUM_T; oo++){
							for (int p = 0; p < FILTER_S; p++){
								for (int q = 0; q < FILTER_S; q++){
									for (int ii = 0; ii < LAYER_IN_NUM_T / U1_DATA1_FC_SIMD_FACTOR; ii++){
#pragma HLS PIPELINE II=1
										uint weight_local_idx = oo * FILTER_S * FILTER_S * LAYER_IN_NUM + p * FILTER_S * LAYER_IN_NUM + q * LAYER_IN_NUM + (in_num_t + ii * U1_DATA1_FC_SIMD_FACTOR);
										uint weight_bus_idx = weight_local_idx / U1_DATA1_PACK_FACTOR;
										uint weight_bus_offset = weight_local_idx % U1_DATA1_PACK_FACTOR;
										U1_bus_t1 bus_data = weight_buf[weight_bus_idx];
										ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> fifo_weight_data;
										for (ap_uint<2> s = 0; s < U1_DATA1_PACK_FACTOR / U1_DATA1_FC_SIMD_FACTOR; s++){
#pragma HLS UNROLL
											sel_tmp[s] = bus_data(U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR - 1, 0);
											bus_data = bus_data >> (U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR);
										}
										fifo_weight_data = sel_tmp[weight_bus_offset / U1_DATA1_FC_SIMD_FACTOR];
										fifo_transfer_weight.write(fifo_weight_data);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

void U1_DataFeed1Head(
		stream<ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> > &fifo_transfer_in,
		stream<U1_Data1TransferChannelType> &fifo_transfer_out0,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
#pragma HLS INLINE off
#pragma HLS DATA_PACK variable=fifo_transfer_out0

	// read in configurations
	uint LAYER_IN_NUM = fifo_config_in.read();
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// loader buffer
	ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> weight_buf[U1_IN_NUM_T * U1_OUT_NUM_T * U1_K * U1_K / U1_DATA1_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=weight_buf dim=1 block factor=1

	bool done1 = 0;
	ap_uint<2> layer_iter = 0;
	while(!done1){
		if (layer_iter > 0){
			LAYER_IN_NUM = fifo_config_in.read();
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			LAYER_BATCH = fifo_config_in.read();

		}
		// write out configurations
		fifo_config_out.write(LAYER_IN_NUM_T);
		fifo_config_out.write(LAYER_OUT_NUM_T);
		fifo_config_out.write(LAYER_IN_IMG_H_T);
		fifo_config_out.write(LAYER_IN_IMG_W_T);
		fifo_config_out.write(LAYER_FILTER_S);
		fifo_config_out.write(LAYER_TASK_NUM1);
		fifo_config_out.write(LAYER_TASK_NUM2);
		fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
		fifo_config_out.write(LAYER_LOCAL_REG_NUM);
		fifo_config_out.write(LAYER_ROW_IL_FACTOR);
		fifo_config_out.write(LAYER_COL_IL_FACTOR);
		fifo_config_out.write(LAYER_STRIDE);
		fifo_config_out.write(LAYER_BATCH);

		bool done2 = 0;
		uint task_iter = 0;
		ap_uint<11> in_num_t = 0;
		while(!done2){
			bool init_final = (in_num_t == 0);
			bool last = (in_num_t == (LAYER_IN_NUM - LAYER_IN_NUM_T));
			// write to SA
			ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> sel_tmp0[U1_DATA1_PACK_FACTOR / U1_DATA1_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sel_tmp0 complete dim=1
			ap_uint<4> t0 = 0;
			ap_uint<5> t1 = 0;
			ap_uint<3> t2 = 0;
			ap_uint<3> t3 = 0;
			ap_uint<11> t4 = 0;
			bool done3 = 0;
			while(!done3){
#pragma HLS PIPELINE II=1
				ap_uint<4> feeder_id = t0 / U1_DATA1_FC_GROUP_FACTOR;
				ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> wide_data0;
				wide_data0 = fifo_transfer_in.read();
				fifo_transfer_out0.write(U1_Data1TransferChannelType(
						wide_data0,
						(uint)feeder_id, init_final, last, LAYER_FILTER_S));

				t4++;
				if (t4 == LAYER_IN_NUM_T / U1_DATA1_FC_SIMD_FACTOR){
					t4 = 0;
					t3++;
					if (t3 == LAYER_FILTER_S){
						t3 = 0;
						t2++;
						if (t2 == LAYER_FILTER_S){
							t2 = 0;
							t1++;
							if (t1 == LAYER_ROW_IL_FACTOR){
								t1 = 0;
								t0++;
								if (t0 == U1_SA_ROWS / U1_DATA1_FC_SPLIT_FACTOR){
									t0 = 0;
									done3 = 1;
								}
							}
						}
					}
				}
			}

			in_num_t += LAYER_IN_NUM_T;
			if (in_num_t == LAYER_IN_NUM){
				in_num_t = 0;
				task_iter++;
				if (task_iter == LAYER_TASK_NUM2){
					task_iter = 0;
					done2 = 1;
				}
			}
		}
		layer_iter++;
		if (layer_iter == LAYER_BATCH){
			layer_iter = 0;
			done1 = 1;
		}
	}
}

void U1_DataCollect2Head_Shim(
		U1_bus_t2* cout,
		stream<ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> > &fifo_transfer_cout,
		stream<U1_ConfigInst> &fifo_kernel_config_in,
		uint LAYER_IN_NUM,
		uint LAYER_OUT_NUM,
		uint LAYER_IN_NUM_T,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H,
		uint LAYER_IN_IMG_W,
		uint LAYER_OUT_IMG_H,
		uint LAYER_OUT_IMG_W,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint LAYER_FILTER_S,
		uint LAYER_STRIDE
){
#pragma HLS INLINE off

	U1_ConfigInst inst0 = fifo_kernel_config_in.read();
	U1_ConfigInst inst1 = fifo_kernel_config_in.read();
	U1_ConfigInst inst2 = fifo_kernel_config_in.read();
	U1_ConfigInst inst3 = fifo_kernel_config_in.read();
	U1_ConfigInst inst4 = fifo_kernel_config_in.read();
	ap_uint<32> LAYER_BATCH = inst3(32*5+31, 32*5);

	U1_bus_t2 cout_buf[U1_DATA2_HEAD_BUF_SIZE / U1_DATA2_PACK_FACTOR];
	ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> sel_tmp[U1_DATA2_PACK_FACTOR / U1_DATA2_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sel_tmp complete dim=1

	for (ap_uint<2> layer_iter = 0; layer_iter < LAYER_BATCH; layer_iter++){
		if (layer_iter > 0){
			U1_ConfigInst inst0 = fifo_kernel_config_in.read();
			U1_ConfigInst inst1 = fifo_kernel_config_in.read();
			U1_ConfigInst inst2 = fifo_kernel_config_in.read();
			U1_ConfigInst inst3 = fifo_kernel_config_in.read();
			U1_ConfigInst inst4 = fifo_kernel_config_in.read();
		}
		for (int out_img_h_t = 0; out_img_h_t < LAYER_OUT_IMG_H; out_img_h_t += LAYER_IN_IMG_H_T / LAYER_STRIDE){
			for (int out_img_w_t = 0; out_img_w_t < LAYER_OUT_IMG_W; out_img_w_t += LAYER_IN_IMG_W_T / LAYER_STRIDE){
				for (int out_num_t = 0; out_num_t < LAYER_OUT_NUM; out_num_t += LAYER_OUT_NUM_T){
					for (int o = 0; o < LAYER_OUT_NUM_T / U1_DATA2_PACK_FACTOR; o++){
						for (int oo = 0; oo < U1_DATA2_PACK_FACTOR / U1_DATA2_FC_SIMD_FACTOR; oo++){
							for (int h = 0; h < LAYER_IN_IMG_H_T / LAYER_STRIDE; h++){
								for (int w = 0; w < LAYER_IN_IMG_W_T / LAYER_STRIDE; w++){
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE INTER false variable=cout_buf
									uint cout_local_index = h * LAYER_IN_IMG_W_T / LAYER_STRIDE * LAYER_OUT_NUM + w * LAYER_OUT_NUM + o * U1_DATA2_PACK_FACTOR + oo * U1_DATA2_FC_SIMD_FACTOR + out_num_t;
									U1_bus_t2 bus_data = cout_buf[cout_local_index / U1_DATA2_PACK_FACTOR];
									for (ap_uint<2> s = 0; s < U1_DATA2_PACK_FACTOR / U1_DATA2_FC_SIMD_FACTOR; s++){
#pragma HLS UNROLL
										sel_tmp[s] = bus_data(U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR - 1, 0);
										bus_data = bus_data >> (U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR);
									}
									ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> fifo_cout_data = fifo_transfer_cout.read();
									sel_tmp[oo] = fifo_cout_data;
									U1_bus_t2 wide_pack = (
#if U1_DATA2_PACK_FACTOR / U1_DATA2_FC_SIMD_FACTOR == 1
											sel_tmp[0]
#elif U1_DATA2_PACK_FACTOR / U1_DATA2_FC_SIMD_FACTOR == 2
													sel_tmp[1], sel_tmp[0]
#elif U1_DATA2_PACK_FACTOR / U1_DATA2_FC_SIMD_FACTOR == 4
																		sel_tmp[3], sel_tmp[2], sel_tmp[1], sel_tmp[0]
#elif U1_DATA2_PACK_FACTOR / U1_DATA2_FC_SIMD_FACTOR == 8
																													sel_tmp[7], sel_tmp[6], sel_tmp[5], sel_tmp[4],
																													sel_tmp[3], sel_tmp[2], sel_tmp[1], sel_tmp[0]
#elif U1_DATA2_PACK_FACTOR / U1_DATA2_FC_SIMD_FACTOR == 16
																																								sel_tmp[15], sel_tmp[14], sel_tmp[13], sel_tmp[12],
																																								sel_tmp[11], sel_tmp[10], sel_tmp[9], sel_tmp[8],
																																								sel_tmp[7], sel_tmp[6], sel_tmp[5], sel_tmp[4],
																																								sel_tmp[3], sel_tmp[2], sel_tmp[1], sel_tmp[0]
#endif
									);
									cout_buf[cout_local_index / U1_DATA2_PACK_FACTOR] = wide_pack;
								}
							}
						}
					}
				}
				unsigned int chunk_offset = out_img_h_t * U1_OUT_IMG_W * LAYER_OUT_NUM;
				for (int h = 0; h < LAYER_IN_IMG_H_T / LAYER_STRIDE; h++){
					uint local_chunk_offset = chunk_offset + h * U1_OUT_IMG_W * LAYER_OUT_NUM + out_img_w_t * LAYER_OUT_NUM;
					memcpy((void*)(cout + local_chunk_offset / U1_DATA2_PACK_FACTOR), (void*)(cout_buf + h * U1_OUT_IMG_W_T * LAYER_OUT_NUM / U1_DATA2_PACK_FACTOR), sizeof(U1_data_t2) * LAYER_IN_IMG_W_T / LAYER_STRIDE * LAYER_OUT_NUM);
				}
			}
		}
	}
}

void U1_DataCollect2Head(
		stream<ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> > &fifo_transfer_out,
		stream<U1_Data2TransferChannelType> &fifo_transfer_in0,
		stream<uint> &fifo_config_in
){
#pragma HLS INLINE off
#pragma HLS DATA_PACK variable=fifo_transfer_in0

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// loader buffer
	ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> cout_buf[U1_OUT_IMG_H_T * U1_OUT_IMG_W_T * U1_OUT_NUM_T / U1_DATA2_FC_SIMD_FACTOR];
	ap_uint<2> layer_iter = 0;
	bool done1 = 0;
	while(!done1){
		if (layer_iter > 0){
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			LAYER_BATCH = fifo_config_in.read();

		}
		int task_num = 0;
		ap_uint<11> t3 = 0;
		ap_uint<5> t1 = 0;
		ap_uint<5> t1_bound = LAYER_IN_IMG_H_T / LAYER_STRIDE;
		ap_uint<4> t0 = 0;
		ap_uint<4> t2 = 0;
		ap_uint<4> t2_bound = LAYER_IN_IMG_W_T / U1_SA_COLS / LAYER_STRIDE;
		bool done2 = 0;
		while(!done2){
#pragma HLS PIPELINE II=1
			U1_Data2TransferChannelType fifo_data0 = fifo_transfer_in0.read();
			fifo_transfer_out.write(fifo_data0.data);
			t2++;
			if (t2 == t2_bound){
				t2 = 0;
				t0++;
				if (t0 == U1_SA_COLS / U1_DATA2_FC_SPLIT_FACTOR){
					t0 = 0;
					t1++;
					if (t1 == t1_bound){
						t1 = 0;
						t3++;
						if (t3 == LAYER_OUT_NUM_T / U1_DATA2_FC_SIMD_FACTOR){
							t3 = 0;
							task_num++;
							if (task_num == LAYER_TASK_NUM2){
								task_num = 0;
								done2 = 1;
							}
						}
					}
				}
			}
		}
		layer_iter++;
		if (layer_iter == LAYER_BATCH){
			layer_iter = 0;
			done1 = 1;
		}
	}
}

/**
 *  This file is automatically generated by PolySA CodeGen.
 *  Version: 1.0
 *  Authos: Jie Wang
 */

//#include "common_header_U1.h"

void U1_Data0FeedData0(
		U1_Data0TransferChannelType buffer[U1_DATA0_FC_GROUP_FACTOR][U1_DATA0_BUF_SIZE/U1_DATA0_FC_SIMD_FACTOR],
		stream<U1_Data0PEChannelType> &fifo_feed_0,
		uint LAYER_IN_NUM_T,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_FILTER_S,
		uint LAYER_STRIDE,
		uint LAYER_ROW_IL_FACTOR,
		uint LAYER_COL_IL_FACTOR
){
#pragma HLS INLINE off
	bool more_to_feed_to_sys_arr = true;

	ap_uint<5> c0_counter = 0;
	ap_uint<5> c1_counter = 0;
	ap_uint<4> c2_counter = 0;
	ap_uint<3> c3_counter = 0;
	ap_uint<3> c4_counter = 0;
	ap_uint<5> c5_counter = 0;

	ap_uint<5> c0_counter_bound;
	if (LAYER_STRIDE == 1){
		c0_counter_bound = LAYER_IN_IMG_H_T;
	} else if (LAYER_STRIDE == 2){
		c0_counter_bound = LAYER_IN_IMG_H_T / 2;
	}

	ap_uint<U1_DATA0_WIDTH*U1_SIMD_FACTOR> sel_tmp_0[U1_DATA0_FC_SIMD_FACTOR/U1_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sel_tmp_0 complete dim=1

	while(more_to_feed_to_sys_arr){
#pragma HLS PIPELINE II=1
		ap_uint<15> buffer_ind_to_feed_to_sys_arr;
		ap_uint<15> w_idx, h_idx;
		if (LAYER_STRIDE == 1){
			w_idx = c2_counter + c4_counter;
			h_idx = c0_counter + c3_counter;
		} else if (LAYER_STRIDE == 2){
			w_idx = c2_counter * 2 + 1 + c4_counter;
			h_idx = c0_counter * 2 + 1 + c3_counter;
		}
		ap_uint<15> w_bound = LAYER_COL_IL_FACTOR * LAYER_STRIDE + LAYER_FILTER_S - 1;
		ap_uint<15> h_bound = LAYER_IN_IMG_H_T + LAYER_FILTER_S - 1;
		buffer_ind_to_feed_to_sys_arr = (w_idx + h_idx * w_bound + c5_counter * U1_SIMD_FACTOR / U1_DATA0_FC_SIMD_FACTOR * h_bound * w_bound) * U1_DATA0_FC_SIMD_FACTOR + c5_counter * U1_SIMD_FACTOR % U1_DATA0_FC_SIMD_FACTOR;

		ap_uint<15> wide_index = buffer_ind_to_feed_to_sys_arr / U1_DATA0_FC_SIMD_FACTOR;
		ap_uint<15> wide_offset = buffer_ind_to_feed_to_sys_arr % U1_DATA0_FC_SIMD_FACTOR;

		U1_Data0TransferChannelType buf_data_0 = buffer[0][wide_index];
		ap_uint<U1_DATA0_WIDTH*U1_DATA0_FC_SIMD_FACTOR> wide_data_0 = buf_data_0.data;
		ap_uint<U1_DATA0_WIDTH*U1_SIMD_FACTOR> data_to_feed_0;
		for (int s = 0; s < U1_DATA0_FC_SIMD_FACTOR / U1_SIMD_FACTOR; s++){
#pragma HLS UNROLL
			sel_tmp_0[s] = wide_data_0(U1_DATA0_WIDTH * U1_SIMD_FACTOR-1, 0);
			wide_data_0 = wide_data_0 >> (U1_DATA0_WIDTH * U1_SIMD_FACTOR);
		}
		data_to_feed_0 = sel_tmp_0[wide_offset / U1_SIMD_FACTOR];

		U1_Data0PEChannelType fifo_data_to_feed_0;
		fifo_data_to_feed_0 = U1_Data0PEChannelType(data_to_feed_0, buf_data_0.new_pair, buf_data_0.last_pair, buf_data_0.FILTER_S);
		fifo_feed_0.write(fifo_data_to_feed_0);

		// counter logic
		c0_counter++;
		if (c0_counter == c0_counter_bound){
			c0_counter = 0;
			c1_counter++;
			if (c1_counter == LAYER_ROW_IL_FACTOR){
				c1_counter = 0;
				c2_counter++;
				if (c2_counter == LAYER_COL_IL_FACTOR){
					c2_counter = 0;
					c3_counter++;
					if (c3_counter == LAYER_FILTER_S){
						c3_counter = 0;
						c4_counter++;
						if (c4_counter == LAYER_FILTER_S){
							c4_counter = 0;
							c5_counter++;
							if (c5_counter == LAYER_IN_NUM_T / U1_SIMD_FACTOR){
								c5_counter = 0;
								more_to_feed_to_sys_arr = false;
							}
						}
					}
				}
			}
		}
	}
}

void U1_Data1FeedData0(
		U1_Data1TransferChannelType buffer[U1_DATA1_FC_GROUP_FACTOR][U1_DATA1_BUF_SIZE/U1_DATA1_FC_SIMD_FACTOR],
		stream<U1_Data1PEChannelType> &fifo_feed_0,
		uint LAYER_IN_NUM_T,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_FILTER_S,
		uint LAYER_STRIDE,
		uint LAYER_ROW_IL_FACTOR,
		uint LAYER_COL_IL_FACTOR
){
#pragma HLS INLINE off
	bool more_to_feed_to_sys_arr = true;

	ap_uint<5> c0_counter = 0;
	ap_uint<5> c1_counter = 0;
	ap_uint<4> c2_counter = 0;
	ap_uint<3> c3_counter = 0;
	ap_uint<3> c4_counter = 0;
	ap_uint<5> c5_counter = 0;

	ap_uint<5> c0_counter_bound;
	if (LAYER_STRIDE == 1){
		c0_counter_bound = LAYER_IN_IMG_H_T;
	} else if (LAYER_STRIDE == 2){
		c0_counter_bound = LAYER_IN_IMG_H_T / 2;
	}

	ap_uint<U1_DATA1_WIDTH*U1_SIMD_FACTOR> sel_tmp_0[U1_DATA1_FC_SIMD_FACTOR/U1_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sel_tmp_0 complete dim=1

	while(more_to_feed_to_sys_arr){
#pragma HLS PIPELINE II=1
		ap_uint<15> buffer_ind_to_feed_to_sys_arr;
		buffer_ind_to_feed_to_sys_arr = c1_counter * LAYER_FILTER_S * LAYER_FILTER_S * LAYER_IN_NUM_T + c3_counter * LAYER_FILTER_S * LAYER_IN_NUM_T + c4_counter * LAYER_IN_NUM_T + c5_counter * U1_SIMD_FACTOR;
		ap_uint<15> wide_index = buffer_ind_to_feed_to_sys_arr / U1_DATA1_FC_SIMD_FACTOR;
		ap_uint<15> wide_offset = buffer_ind_to_feed_to_sys_arr % U1_DATA1_FC_SIMD_FACTOR;

		U1_Data1TransferChannelType buf_data_0 = buffer[0][wide_index];
		ap_uint<U1_DATA1_WIDTH*U1_DATA1_FC_SIMD_FACTOR> wide_data_0 = buf_data_0.data;
		ap_uint<U1_DATA1_WIDTH*U1_SIMD_FACTOR> data_to_feed_0;
		for (int s = 0; s < U1_DATA1_FC_SIMD_FACTOR/U1_SIMD_FACTOR; s++){
#pragma HLS UNROLL
			sel_tmp_0[s] = wide_data_0(U1_DATA1_WIDTH * U1_SIMD_FACTOR-1, 0);
			wide_data_0 = wide_data_0 >> (U1_DATA1_WIDTH * U1_SIMD_FACTOR);
		}
		data_to_feed_0 = sel_tmp_0[wide_offset / U1_SIMD_FACTOR];

		U1_Data1PEChannelType fifo_data_to_feed_0;
		fifo_data_to_feed_0 = U1_Data1PEChannelType(data_to_feed_0, buf_data_0.new_pair, buf_data_0.last_pair, buf_data_0.FILTER_S);
		fifo_feed_0.write(fifo_data_to_feed_0);

		// counter logic
		c0_counter++;
		if (c0_counter == c0_counter_bound){
			c0_counter = 0;
			c1_counter++;
			if (c1_counter == LAYER_ROW_IL_FACTOR){
				c1_counter = 0;
				c2_counter++;
				if (c2_counter == LAYER_COL_IL_FACTOR){
					c2_counter = 0;
					c3_counter++;
					if (c3_counter == LAYER_FILTER_S){
						c3_counter = 0;
						c4_counter++;
						if (c4_counter == LAYER_FILTER_S){
							c4_counter = 0;
							c5_counter++;
							if (c5_counter == LAYER_IN_NUM_T / U1_SIMD_FACTOR){
								c5_counter = 0;
								more_to_feed_to_sys_arr = false;
							}
						}
					}
				}
			}
		}
	}
}

void U1_Data0ReadData0(
		U1_Data0TransferChannelType buffer[U1_DATA0_FC_GROUP_FACTOR][U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR],
		stream<U1_Data0TransferChannelType> &fifo_transfer_in,
		stream<U1_Data0TransferChannelType> &fifo_transfer_out,
		unsigned int engine_id,
		uint LAYER_IN_NUM_T,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_FILTER_S,
		uint LAYER_STRIDE,
		uint LAYER_ROW_IL_FACTOR,
		uint LAYER_COL_IL_FACTOR
){
#pragma HLS INLINE off
	bool LAST_ENGINE = (engine_id == 8 / U1_DATA0_FC_SPLIT_FACTOR - 1);
	ap_uint<15> transfer_counter = 0;
	ap_uint<15> data0_buf_size;
	ap_uint<15> local_transfer_size;
	bool more_to_write_to_buffer = true;
	bool more_to_feed_to_sys_arr = false;
	bool more_to_forward = true;
	ap_uint<12> buffer_write_counter = 0;
	ap_uint<1> buffer_gs_id = 0;

	// the first read
	data0_buf_size = LAYER_IN_NUM_T * (LAYER_IN_IMG_H_T + LAYER_FILTER_S - 1) * (LAYER_COL_IL_FACTOR * LAYER_STRIDE + LAYER_FILTER_S - 1) / U1_DATA0_FC_SIMD_FACTOR;
	local_transfer_size = data0_buf_size * (8 / U1_DATA0_FC_SPLIT_FACTOR - engine_id) * U1_DATA0_FC_GROUP_FACTOR;

	while(more_to_forward){
#pragma HLS PIPELINE II=1
		U1_Data0TransferChannelType data_read_from_fifo = fifo_transfer_in.read();
		bool data_is_to_buffer;
		bool data_is_to_forward;
		unsigned int feeder_id = data_read_from_fifo.feeder_id;
		data_is_to_buffer = LAST_ENGINE || (!LAST_ENGINE && feeder_id == engine_id);
		data_is_to_forward = !LAST_ENGINE && (feeder_id != engine_id);
		if (!LAST_ENGINE){
			if (data_is_to_forward){
				fifo_transfer_out.write(data_read_from_fifo);
			}
		}
		ap_uint<12> buffer_ind_to_write_to_buffer = buffer_write_counter;

		if (data_is_to_buffer){
			buffer[buffer_gs_id][buffer_ind_to_write_to_buffer] = data_read_from_fifo;
			buffer_write_counter++;
			if (buffer_write_counter == data0_buf_size){
				buffer_write_counter = 0;
				buffer_gs_id++;
				if (buffer_gs_id == U1_DATA0_FC_GROUP_FACTOR){
					buffer_gs_id = 0;
					more_to_write_to_buffer = false;
				}
			}
		}
		transfer_counter++;
		if (transfer_counter == local_transfer_size){
			transfer_counter = 0;
			more_to_forward = false;
		}
	}

}

void U1_Data0ReadDataLast(
		U1_Data0TransferChannelType buffer[U1_DATA0_FC_GROUP_FACTOR][U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR],
		stream<U1_Data0TransferChannelType> &fifo_transfer_in,
		unsigned int engine_id,
		uint LAYER_IN_NUM_T,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_FILTER_S,
		uint LAYER_STRIDE,
		uint LAYER_ROW_IL_FACTOR,
		uint LAYER_COL_IL_FACTOR
){
#pragma HLS INLINE off
	bool LAST_ENGINE = (engine_id == 8 / U1_DATA0_FC_SPLIT_FACTOR - 1);
	bool buffer_id_to_write_to_buffer = 0;
	bool buffer_id_to_feed_to_sys_arr = 1;
	ap_uint<15> transfer_counter = 0;
	ap_uint<15> data0_buf_size;
	ap_uint<15> local_transfer_size;
	bool more_to_write_to_buffer = true;
	bool more_to_feed_to_sys_arr = false;
	bool more_to_forward = true;
	ap_uint<12> buffer_write_counter = 0;
	ap_uint<1> buffer_gs_id = 0;

	// the first read
	data0_buf_size = LAYER_IN_NUM_T * (LAYER_IN_IMG_H_T + LAYER_FILTER_S - 1) * (LAYER_COL_IL_FACTOR * LAYER_STRIDE + LAYER_FILTER_S - 1) / U1_DATA0_FC_SIMD_FACTOR;
	local_transfer_size = data0_buf_size * (8 / U1_DATA0_FC_SPLIT_FACTOR - engine_id) * U1_DATA0_FC_GROUP_FACTOR;

	while(more_to_forward){
#pragma HLS PIPELINE II=1
		U1_Data0TransferChannelType data_read_from_fifo = fifo_transfer_in.read();
		bool data_is_to_buffer;
		bool data_is_to_forward;
		unsigned int feeder_id = data_read_from_fifo.feeder_id;
		data_is_to_buffer = LAST_ENGINE || (!LAST_ENGINE && feeder_id == engine_id);
		data_is_to_forward = !LAST_ENGINE && (feeder_id != engine_id);
		ap_uint<12> buffer_ind_to_write_to_buffer = buffer_write_counter;

		if (data_is_to_buffer){
			buffer[buffer_gs_id][buffer_ind_to_write_to_buffer] = data_read_from_fifo;
			buffer_write_counter++;
			if (buffer_write_counter == data0_buf_size){
				buffer_write_counter = 0;
				buffer_gs_id++;
				if (buffer_gs_id == U1_DATA0_FC_GROUP_FACTOR){
					buffer_gs_id = 0;
					more_to_write_to_buffer = false;
				}
			}
		}
		transfer_counter++;
		if (transfer_counter == local_transfer_size){
			transfer_counter = 0;
			more_to_forward = false;
		}
	}

}

void U1_Data1ReadData0(
		U1_Data1TransferChannelType buffer[U1_DATA1_FC_GROUP_FACTOR][U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR],
		stream<U1_Data1TransferChannelType> &fifo_transfer_in,
		stream<U1_Data1TransferChannelType> &fifo_transfer_out,
		unsigned int engine_id,
		uint LAYER_IN_NUM_T,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_FILTER_S,
		uint LAYER_STRIDE,
		uint LAYER_ROW_IL_FACTOR,
		uint LAYER_COL_IL_FACTOR
){
#pragma HLS INLINE off
	bool LAST_ENGINE = (engine_id == 8 / U1_DATA1_FC_SPLIT_FACTOR - 1);
	ap_uint<15> transfer_counter = 0;
	ap_uint<15> data1_buf_size;
	ap_uint<15> local_transfer_size;
	bool more_to_write_to_buffer = true;
	bool more_to_feed_to_sys_arr = false;
	bool more_to_forward = true;
	ap_uint<12> buffer_write_counter = 0;
	ap_uint<1> buffer_gs_id = 0;

	// the first read
	data1_buf_size = LAYER_IN_NUM_T * LAYER_ROW_IL_FACTOR * LAYER_FILTER_S * LAYER_FILTER_S / U1_DATA1_FC_SIMD_FACTOR;
	local_transfer_size = data1_buf_size * (8 / U1_DATA1_FC_SPLIT_FACTOR - engine_id) * U1_DATA1_FC_GROUP_FACTOR;

	while(more_to_forward){
#pragma HLS PIPELINE II=1
		U1_Data1TransferChannelType data_read_from_fifo = fifo_transfer_in.read();
		bool data_is_to_buffer;
		bool data_is_to_forward;
		unsigned int feeder_id = data_read_from_fifo.feeder_id;
		data_is_to_buffer = LAST_ENGINE || (!LAST_ENGINE && feeder_id == engine_id);
		data_is_to_forward = !LAST_ENGINE && (feeder_id != engine_id);
		if (!LAST_ENGINE){
			if (data_is_to_forward){
				fifo_transfer_out.write(data_read_from_fifo);
			}
		}
		ap_uint<12> buffer_ind_to_write_to_buffer = buffer_write_counter;

		if (data_is_to_buffer){
			buffer[buffer_gs_id][buffer_ind_to_write_to_buffer] = data_read_from_fifo;
			buffer_write_counter++;
			if (buffer_write_counter == data1_buf_size){
				buffer_write_counter = 0;
				buffer_gs_id++;
				if (buffer_gs_id == U1_DATA1_FC_GROUP_FACTOR){
					buffer_gs_id = 0;
					more_to_write_to_buffer = false;
				}
			}
		}
		transfer_counter++;
		if (transfer_counter == local_transfer_size){
			transfer_counter = 0;
			more_to_forward = false;
		}
	}

}

void U1_Data1ReadDataLast(
		U1_Data1TransferChannelType buffer[U1_DATA1_FC_GROUP_FACTOR][U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR],
		stream<U1_Data1TransferChannelType> &fifo_transfer_in,
		unsigned int engine_id,
		uint LAYER_IN_NUM_T,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_FILTER_S,
		uint LAYER_STRIDE,
		uint LAYER_ROW_IL_FACTOR,
		uint LAYER_COL_IL_FACTOR
){
#pragma HLS INLINE off
	bool LAST_ENGINE = (engine_id == 8 / U1_DATA1_FC_SPLIT_FACTOR - 1);
	bool buffer_id_to_write_to_buffer = 0;
	bool buffer_id_to_feed_to_sys_arr = 1;
	ap_uint<15> transfer_counter = 0;
	ap_uint<15> data1_buf_size;
	ap_uint<15> local_transfer_size;
	bool more_to_write_to_buffer = true;
	bool more_to_feed_to_sys_arr = false;
	bool more_to_forward = true;
	ap_uint<12> buffer_write_counter = 0;
	ap_uint<1> buffer_gs_id = 0;

	// the first read
	data1_buf_size = LAYER_IN_NUM_T * LAYER_ROW_IL_FACTOR * LAYER_FILTER_S * LAYER_FILTER_S / U1_DATA1_FC_SIMD_FACTOR;
	local_transfer_size = data1_buf_size * (8 / U1_DATA1_FC_SPLIT_FACTOR - engine_id) * U1_DATA1_FC_GROUP_FACTOR;

	while(more_to_forward){
#pragma HLS PIPELINE II=1
		U1_Data1TransferChannelType data_read_from_fifo = fifo_transfer_in.read();
		bool data_is_to_buffer;
		bool data_is_to_forward;
		unsigned int feeder_id = data_read_from_fifo.feeder_id;
		data_is_to_buffer = LAST_ENGINE || (!LAST_ENGINE && feeder_id == engine_id);
		data_is_to_forward = !LAST_ENGINE && (feeder_id != engine_id);
		ap_uint<12> buffer_ind_to_write_to_buffer = buffer_write_counter;

		if (data_is_to_buffer){
			buffer[buffer_gs_id][buffer_ind_to_write_to_buffer] = data_read_from_fifo;
			buffer_write_counter++;
			if (buffer_write_counter == data1_buf_size){
				buffer_write_counter = 0;
				buffer_gs_id++;
				if (buffer_gs_id == U1_DATA1_FC_GROUP_FACTOR){
					buffer_gs_id = 0;
					more_to_write_to_buffer = false;
				}
			}
		}
		transfer_counter++;
		if (transfer_counter == local_transfer_size){
			transfer_counter = 0;
			more_to_forward = false;
		}
	}

}

void U1_DataFeed0Engine0(
		stream<U1_Data0TransferChannelType> &fifo_transfer_in,
		stream<U1_Data0TransferChannelType> &fifo_transfer_out,
		stream<U1_Data0PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out0,
		stream<uint> &fifo_config_out1
){
#pragma HLS DATA_PACK variable=fifo_transfer_in
#pragma HLS DATA_PACK variable=fifo_transfer_out
#pragma HLS DATA_PACK variable=fifo_feed_0
#pragma HLS INLINE off

	uint task_iter = 0;
	uint LAYER_IN_NUM_T_prev;
	uint LAYER_IN_IMG_H_T_prev;
	uint LAYER_FILTER_S_prev;
	uint LAYER_STRIDE_prev;
	uint LAYER_ROW_IL_FACTOR_prev;
	uint LAYER_COL_IL_FACTOR_prev;
	uint dummy;

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// write out configurations
	fifo_config_out0.write(LAYER_IN_NUM_T);
	fifo_config_out0.write(LAYER_OUT_NUM_T);
	fifo_config_out0.write(LAYER_IN_IMG_H_T);
	fifo_config_out0.write(LAYER_IN_IMG_W_T);
	fifo_config_out0.write(LAYER_FILTER_S);
	fifo_config_out0.write(LAYER_TASK_NUM1);
	fifo_config_out0.write(LAYER_TASK_NUM2);
	fifo_config_out0.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out0.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out0.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out0.write(LAYER_COL_IL_FACTOR);
	fifo_config_out0.write(LAYER_STRIDE);
	fifo_config_out0.write(LAYER_BATCH);

	fifo_config_out1.write(LAYER_IN_NUM_T);
	fifo_config_out1.write(LAYER_OUT_NUM_T);
	fifo_config_out1.write(LAYER_IN_IMG_H_T);
	fifo_config_out1.write(LAYER_IN_IMG_W_T);
	fifo_config_out1.write(LAYER_FILTER_S);
	fifo_config_out1.write(LAYER_TASK_NUM1);
	fifo_config_out1.write(LAYER_TASK_NUM2);
	fifo_config_out1.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out1.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out1.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out1.write(LAYER_COL_IL_FACTOR);
	fifo_config_out1.write(LAYER_STRIDE);
	fifo_config_out1.write(LAYER_BATCH);

	U1_Data0TransferChannelType ping_buffer[U1_DATA0_FC_GROUP_FACTOR][U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR];
	U1_Data0TransferChannelType pong_buffer[U1_DATA0_FC_GROUP_FACTOR][U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR];
#pragma HLS RESOURCE variable=ping_buffer core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=pong_buffer core=RAM_2P_BRAM
#pragma HLS DATA_PACK variable=ping_buffer
#pragma HLS DATA_PACK variable=pong_buffer
#pragma HLS ARRAY_PARTITION variable=ping_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=pong_buffer dim=1 complete

	unsigned int initial_round = 0;

	bool done = 0;
	ap_uint<2> layer_iter = 0;
	bool layer_start = 0;
	while(!done){
		if (layer_start){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			dummy = fifo_config_in.read();

			// write out configurations
			fifo_config_out0.write(LAYER_IN_NUM_T);
			fifo_config_out0.write(LAYER_OUT_NUM_T);
			fifo_config_out0.write(LAYER_IN_IMG_H_T);
			fifo_config_out0.write(LAYER_IN_IMG_W_T);
			fifo_config_out0.write(LAYER_FILTER_S);
			fifo_config_out0.write(LAYER_TASK_NUM1);
			fifo_config_out0.write(LAYER_TASK_NUM2);
			fifo_config_out0.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out0.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out0.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out0.write(LAYER_COL_IL_FACTOR);
			fifo_config_out0.write(LAYER_STRIDE);
			fifo_config_out0.write(LAYER_BATCH);

			fifo_config_out1.write(LAYER_IN_NUM_T);
			fifo_config_out1.write(LAYER_OUT_NUM_T);
			fifo_config_out1.write(LAYER_IN_IMG_H_T);
			fifo_config_out1.write(LAYER_IN_IMG_W_T);
			fifo_config_out1.write(LAYER_FILTER_S);
			fifo_config_out1.write(LAYER_TASK_NUM1);
			fifo_config_out1.write(LAYER_TASK_NUM2);
			fifo_config_out1.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out1.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out1.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out1.write(LAYER_COL_IL_FACTOR);
			fifo_config_out1.write(LAYER_STRIDE);
			fifo_config_out1.write(LAYER_BATCH);

			layer_start = 0;
		}

		if (initial_round == 0){
			U1_Data0ReadData0(ping_buffer, fifo_transfer_in, fifo_transfer_out, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
		} else {
			if (initial_round % 2 == 1){
				U1_Data0ReadData0(pong_buffer, fifo_transfer_in, fifo_transfer_out, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
				U1_Data0FeedData0(
						ping_buffer,
						fifo_feed_0,
						LAYER_IN_NUM_T_prev,
						LAYER_IN_IMG_H_T_prev,
						LAYER_FILTER_S_prev,
						LAYER_STRIDE_prev,
						LAYER_ROW_IL_FACTOR_prev,
						LAYER_COL_IL_FACTOR_prev);
			} else {
				U1_Data0ReadData0(ping_buffer, fifo_transfer_in, fifo_transfer_out, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
				U1_Data0FeedData0(
						pong_buffer,
						fifo_feed_0,
						LAYER_IN_NUM_T_prev,
						LAYER_IN_IMG_H_T_prev,
						LAYER_FILTER_S_prev,
						LAYER_STRIDE_prev,
						LAYER_ROW_IL_FACTOR_prev,
						LAYER_COL_IL_FACTOR_prev);
			}
		}

		initial_round++;
		LAYER_IN_NUM_T_prev = LAYER_IN_NUM_T;
		LAYER_IN_IMG_H_T_prev = LAYER_IN_IMG_H_T;
		LAYER_FILTER_S_prev = LAYER_FILTER_S;
		LAYER_STRIDE_prev = LAYER_STRIDE;
		LAYER_ROW_IL_FACTOR_prev = LAYER_ROW_IL_FACTOR;
		LAYER_COL_IL_FACTOR_prev = LAYER_COL_IL_FACTOR;

		task_iter++;
		if (task_iter == LAYER_TASK_NUM1){
			task_iter = 0;
			layer_iter += 1;
			layer_start = 1;
			if (layer_iter == LAYER_BATCH){
				layer_iter = 0;
				done = 1;
			}
		}
	}

	if (initial_round % 2 == 1){
		U1_Data0FeedData0(
				ping_buffer,
				fifo_feed_0,
				LAYER_IN_NUM_T_prev,
				LAYER_IN_IMG_H_T_prev,
				LAYER_FILTER_S_prev,
				LAYER_STRIDE_prev,
				LAYER_ROW_IL_FACTOR_prev,
				LAYER_COL_IL_FACTOR_prev);
	} else {
		U1_Data0FeedData0(
				pong_buffer,
				fifo_feed_0,
				LAYER_IN_NUM_T_prev,
				LAYER_IN_IMG_H_T_prev,
				LAYER_FILTER_S_prev,
				LAYER_STRIDE_prev,
				LAYER_ROW_IL_FACTOR_prev,
				LAYER_COL_IL_FACTOR_prev);
	}
}

void U1_DataFeed0Engine0_wrapper(
		stream<U1_Data0TransferChannelType> &fifo_transfer_in,
		stream<U1_Data0TransferChannelType> &fifo_transfer_out,
		stream<U1_Data0PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out0,
		stream<uint> &fifo_config_out1
){
	U1_DataFeed0Engine0(
			fifo_transfer_in,
			fifo_transfer_out,
			fifo_feed_0,
			engine_id,
			fifo_config_in,
			fifo_config_out0,
			fifo_config_out1
	);
}

void U1_DataFeed0EngineLast(
		stream<U1_Data0TransferChannelType> &fifo_transfer_in,
		stream<U1_Data0PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out1
){
#pragma HLS DATA_PACK variable=fifo_transfer_in
#pragma HLS DATA_PACK variable=fifo_feed_0
#pragma HLS INLINE off

	uint task_iter = 0;
	uint LAYER_IN_NUM_T_prev;
	uint LAYER_IN_IMG_H_T_prev;
	uint LAYER_FILTER_S_prev;
	uint LAYER_STRIDE_prev;
	uint LAYER_ROW_IL_FACTOR_prev;
	uint LAYER_COL_IL_FACTOR_prev;
	uint dummy;

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// write out configurations
	fifo_config_out1.write(LAYER_IN_NUM_T);
	fifo_config_out1.write(LAYER_OUT_NUM_T);
	fifo_config_out1.write(LAYER_IN_IMG_H_T);
	fifo_config_out1.write(LAYER_IN_IMG_W_T);
	fifo_config_out1.write(LAYER_FILTER_S);
	fifo_config_out1.write(LAYER_TASK_NUM1);
	fifo_config_out1.write(LAYER_TASK_NUM2);
	fifo_config_out1.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out1.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out1.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out1.write(LAYER_COL_IL_FACTOR);
	fifo_config_out1.write(LAYER_STRIDE);
	fifo_config_out1.write(LAYER_BATCH);

	U1_Data0TransferChannelType ping_buffer[U1_DATA0_FC_GROUP_FACTOR][U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR];
	U1_Data0TransferChannelType pong_buffer[U1_DATA0_FC_GROUP_FACTOR][U1_DATA0_BUF_SIZE / U1_DATA0_FC_SIMD_FACTOR];
#pragma HLS RESOURCE variable=ping_buffer core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=pong_buffer core=RAM_2P_BRAM
#pragma HLS DATA_PACK variable=ping_buffer
#pragma HLS DATA_PACK variable=pong_buffer
#pragma HLS ARRAY_PARTITION variable=ping_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=pong_buffer dim=1 complete

	unsigned int initial_round = 0;

	bool done = 0;
	ap_uint<2> layer_iter = 0;
	bool layer_start = 0;
	while(!done){
		if (layer_start){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			dummy = fifo_config_in.read();

			// write out configurations
			fifo_config_out1.write(LAYER_IN_NUM_T);
			fifo_config_out1.write(LAYER_OUT_NUM_T);
			fifo_config_out1.write(LAYER_IN_IMG_H_T);
			fifo_config_out1.write(LAYER_IN_IMG_W_T);
			fifo_config_out1.write(LAYER_FILTER_S);
			fifo_config_out1.write(LAYER_TASK_NUM1);
			fifo_config_out1.write(LAYER_TASK_NUM2);
			fifo_config_out1.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out1.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out1.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out1.write(LAYER_COL_IL_FACTOR);
			fifo_config_out1.write(LAYER_STRIDE);
			fifo_config_out1.write(LAYER_BATCH);

			layer_start = 0;
		}

		if (initial_round == 0){
			U1_Data0ReadDataLast(ping_buffer, fifo_transfer_in, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
		} else {
			if (initial_round % 2 == 1){
				U1_Data0ReadDataLast(pong_buffer, fifo_transfer_in, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
				U1_Data0FeedData0(
						ping_buffer,
						fifo_feed_0,
						LAYER_IN_NUM_T_prev,
						LAYER_IN_IMG_H_T_prev,
						LAYER_FILTER_S_prev,
						LAYER_STRIDE_prev,
						LAYER_ROW_IL_FACTOR_prev,
						LAYER_COL_IL_FACTOR_prev);
			} else {
				U1_Data0ReadDataLast(ping_buffer, fifo_transfer_in, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
				U1_Data0FeedData0(
						pong_buffer,
						fifo_feed_0,
						LAYER_IN_NUM_T_prev,
						LAYER_IN_IMG_H_T_prev,
						LAYER_FILTER_S_prev,
						LAYER_STRIDE_prev,
						LAYER_ROW_IL_FACTOR_prev,
						LAYER_COL_IL_FACTOR_prev);
			}
		}

		initial_round++;
		LAYER_IN_NUM_T_prev = LAYER_IN_NUM_T;
		LAYER_IN_IMG_H_T_prev = LAYER_IN_IMG_H_T;
		LAYER_FILTER_S_prev = LAYER_FILTER_S;
		LAYER_STRIDE_prev = LAYER_STRIDE;
		LAYER_ROW_IL_FACTOR_prev = LAYER_ROW_IL_FACTOR;
		LAYER_COL_IL_FACTOR_prev = LAYER_COL_IL_FACTOR;

		task_iter++;
		if (task_iter == LAYER_TASK_NUM1){
			task_iter = 0;
			layer_iter += 1;
			layer_start = 1;
			if (layer_iter == LAYER_BATCH){
				layer_iter = 0;
				done = 1;
			}
		}
	}

	if (initial_round % 2 == 1){
		U1_Data0FeedData0(
				ping_buffer,
				fifo_feed_0,
				LAYER_IN_NUM_T_prev,
				LAYER_IN_IMG_H_T_prev,
				LAYER_FILTER_S_prev,
				LAYER_STRIDE_prev,
				LAYER_ROW_IL_FACTOR_prev,
				LAYER_COL_IL_FACTOR_prev);
	} else {
		U1_Data0FeedData0(
				pong_buffer,
				fifo_feed_0,
				LAYER_IN_NUM_T_prev,
				LAYER_IN_IMG_H_T_prev,
				LAYER_FILTER_S_prev,
				LAYER_STRIDE_prev,
				LAYER_ROW_IL_FACTOR_prev,
				LAYER_COL_IL_FACTOR_prev);
	}
}

void U1_DataFeed1Engine0(
		stream<U1_Data1TransferChannelType> &fifo_transfer_in,
		stream<U1_Data1TransferChannelType> &fifo_transfer_out,
		stream<U1_Data1PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out0
){
#pragma HLS DATA_PACK variable=fifo_transfer_in
#pragma HLS DATA_PACK variable=fifo_transfer_out
#pragma HLS DATA_PACK variable=fifo_feed_0
#pragma HLS INLINE off

	uint task_iter = 0;
	uint LAYER_IN_NUM_T_prev;
	uint LAYER_IN_IMG_H_T_prev;
	uint LAYER_FILTER_S_prev;
	uint LAYER_STRIDE_prev;
	uint LAYER_ROW_IL_FACTOR_prev;
	uint LAYER_COL_IL_FACTOR_prev;
	uint dummy;

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// write out configurations
	fifo_config_out0.write(LAYER_IN_NUM_T);
	fifo_config_out0.write(LAYER_OUT_NUM_T);
	fifo_config_out0.write(LAYER_IN_IMG_H_T);
	fifo_config_out0.write(LAYER_IN_IMG_W_T);
	fifo_config_out0.write(LAYER_FILTER_S);
	fifo_config_out0.write(LAYER_TASK_NUM1);
	fifo_config_out0.write(LAYER_TASK_NUM2);
	fifo_config_out0.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out0.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out0.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out0.write(LAYER_COL_IL_FACTOR);
	fifo_config_out0.write(LAYER_STRIDE);
	fifo_config_out0.write(LAYER_BATCH);
	U1_Data1TransferChannelType ping_buffer[U1_DATA1_FC_GROUP_FACTOR][U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR];
	U1_Data1TransferChannelType pong_buffer[U1_DATA1_FC_GROUP_FACTOR][U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR];
#pragma HLS RESOURCE variable=ping_buffer core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=pong_buffer core=RAM_2P_BRAM
#pragma HLS DATA_PACK variable=ping_buffer
#pragma HLS DATA_PACK variable=pong_buffer
#pragma HLS ARRAY_PARTITION variable=ping_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=pong_buffer dim=1 complete

	unsigned int initial_round = 0;

	bool done = 0;
	ap_uint<2> layer_iter = 0;
	bool layer_start = 0;
	while(!done){
		if (layer_start){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			dummy = fifo_config_in.read();

			// write out configurations
			fifo_config_out0.write(LAYER_IN_NUM_T);
			fifo_config_out0.write(LAYER_OUT_NUM_T);
			fifo_config_out0.write(LAYER_IN_IMG_H_T);
			fifo_config_out0.write(LAYER_IN_IMG_W_T);
			fifo_config_out0.write(LAYER_FILTER_S);
			fifo_config_out0.write(LAYER_TASK_NUM1);
			fifo_config_out0.write(LAYER_TASK_NUM2);
			fifo_config_out0.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out0.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out0.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out0.write(LAYER_COL_IL_FACTOR);
			fifo_config_out0.write(LAYER_STRIDE);
			fifo_config_out0.write(LAYER_BATCH);

			layer_start = 0;
		}

		if (initial_round == 0){
			U1_Data1ReadData0(ping_buffer, fifo_transfer_in, fifo_transfer_out, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
		} else {
			if (initial_round % 2 == 1){
				U1_Data1ReadData0(pong_buffer, fifo_transfer_in, fifo_transfer_out, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
				U1_Data1FeedData0(
						ping_buffer,
						fifo_feed_0,
						LAYER_IN_NUM_T_prev,
						LAYER_IN_IMG_H_T_prev,
						LAYER_FILTER_S_prev,
						LAYER_STRIDE_prev,
						LAYER_ROW_IL_FACTOR_prev,
						LAYER_COL_IL_FACTOR_prev);
			} else {
				U1_Data1ReadData0(ping_buffer, fifo_transfer_in, fifo_transfer_out, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
				U1_Data1FeedData0(
						pong_buffer,
						fifo_feed_0,
						LAYER_IN_NUM_T_prev,
						LAYER_IN_IMG_H_T_prev,
						LAYER_FILTER_S_prev,
						LAYER_STRIDE_prev,
						LAYER_ROW_IL_FACTOR_prev,
						LAYER_COL_IL_FACTOR_prev);
			}
		}

		initial_round++;
		LAYER_IN_NUM_T_prev = LAYER_IN_NUM_T;
		LAYER_IN_IMG_H_T_prev = LAYER_IN_IMG_H_T;
		LAYER_FILTER_S_prev = LAYER_FILTER_S;
		LAYER_STRIDE_prev = LAYER_STRIDE;
		LAYER_ROW_IL_FACTOR_prev = LAYER_ROW_IL_FACTOR;
		LAYER_COL_IL_FACTOR_prev = LAYER_COL_IL_FACTOR;

		task_iter++;
		if (task_iter == LAYER_TASK_NUM1){
			task_iter = 0;
			layer_iter += 1;
			layer_start = 1;
			if (layer_iter == LAYER_BATCH){
				layer_iter = 0;
				done = 1;
			}
		}
	}

	if (initial_round % 2 == 1){
		U1_Data1FeedData0(
				ping_buffer,
				fifo_feed_0,
				LAYER_IN_NUM_T_prev,
				LAYER_IN_IMG_H_T_prev,
				LAYER_FILTER_S_prev,
				LAYER_STRIDE_prev,
				LAYER_ROW_IL_FACTOR_prev,
				LAYER_COL_IL_FACTOR_prev);
	} else {
		U1_Data1FeedData0(
				pong_buffer,
				fifo_feed_0,
				LAYER_IN_NUM_T_prev,
				LAYER_IN_IMG_H_T_prev,
				LAYER_FILTER_S_prev,
				LAYER_STRIDE_prev,
				LAYER_ROW_IL_FACTOR_prev,
				LAYER_COL_IL_FACTOR_prev);
	}
}

void U1_DataFeed1Engine0_wrapper(
		stream<U1_Data1TransferChannelType> &fifo_transfer_in,
		stream<U1_Data1TransferChannelType> &fifo_transfer_out,
		stream<U1_Data1PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out0
){
	U1_DataFeed1Engine0(
			fifo_transfer_in,
			fifo_transfer_out,
			fifo_feed_0,
			engine_id,
			fifo_config_in,
			fifo_config_out0
	);
}

void U1_DataFeed1EngineLast(
		stream<U1_Data1TransferChannelType> &fifo_transfer_in,
		stream<U1_Data1PEChannelType> &fifo_feed_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in
){
#pragma HLS DATA_PACK variable=fifo_transfer_in
#pragma HLS DATA_PACK variable=fifo_feed_0
#pragma HLS INLINE off

	uint task_iter = 0;
	uint LAYER_IN_NUM_T_prev;
	uint LAYER_IN_IMG_H_T_prev;
	uint LAYER_FILTER_S_prev;
	uint LAYER_STRIDE_prev;
	uint LAYER_ROW_IL_FACTOR_prev;
	uint LAYER_COL_IL_FACTOR_prev;
	uint dummy;

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	U1_Data1TransferChannelType ping_buffer[U1_DATA1_FC_GROUP_FACTOR][U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR];
	U1_Data1TransferChannelType pong_buffer[U1_DATA1_FC_GROUP_FACTOR][U1_DATA1_BUF_SIZE / U1_DATA1_FC_SIMD_FACTOR];
#pragma HLS RESOURCE variable=ping_buffer core=RAM_2P_BRAM
#pragma HLS RESOURCE variable=pong_buffer core=RAM_2P_BRAM
#pragma HLS DATA_PACK variable=ping_buffer
#pragma HLS DATA_PACK variable=pong_buffer
#pragma HLS ARRAY_PARTITION variable=ping_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=pong_buffer dim=1 complete

	unsigned int initial_round = 0;

	bool done = 0;
	ap_uint<2> layer_iter = 0;
	bool layer_start = 0;
	while(!done){
		if (layer_start){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			dummy = fifo_config_in.read();

			layer_start = 0;
		}

		if (initial_round == 0){
			U1_Data1ReadDataLast(ping_buffer, fifo_transfer_in, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
		} else {
			if (initial_round % 2 == 1){
				U1_Data1ReadDataLast(pong_buffer, fifo_transfer_in, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
				U1_Data1FeedData0(
						ping_buffer,
						fifo_feed_0,
						LAYER_IN_NUM_T_prev,
						LAYER_IN_IMG_H_T_prev,
						LAYER_FILTER_S_prev,
						LAYER_STRIDE_prev,
						LAYER_ROW_IL_FACTOR_prev,
						LAYER_COL_IL_FACTOR_prev);
			} else {
				U1_Data1ReadDataLast(ping_buffer, fifo_transfer_in, engine_id, LAYER_IN_NUM_T, LAYER_IN_IMG_H_T, LAYER_FILTER_S, LAYER_STRIDE, LAYER_ROW_IL_FACTOR, LAYER_COL_IL_FACTOR);
				U1_Data1FeedData0(
						pong_buffer,
						fifo_feed_0,
						LAYER_IN_NUM_T_prev,
						LAYER_IN_IMG_H_T_prev,
						LAYER_FILTER_S_prev,
						LAYER_STRIDE_prev,
						LAYER_ROW_IL_FACTOR_prev,
						LAYER_COL_IL_FACTOR_prev);
			}
		}

		initial_round++;
		LAYER_IN_NUM_T_prev = LAYER_IN_NUM_T;
		LAYER_IN_IMG_H_T_prev = LAYER_IN_IMG_H_T;
		LAYER_FILTER_S_prev = LAYER_FILTER_S;
		LAYER_STRIDE_prev = LAYER_STRIDE;
		LAYER_ROW_IL_FACTOR_prev = LAYER_ROW_IL_FACTOR;
		LAYER_COL_IL_FACTOR_prev = LAYER_COL_IL_FACTOR;

		task_iter++;
		if (task_iter == LAYER_TASK_NUM1){
			task_iter = 0;
			layer_iter += 1;
			layer_start = 1;
			if (layer_iter == LAYER_BATCH){
				layer_iter = 0;
				done = 1;
			}
		}
	}

	if (initial_round % 2 == 1){
		U1_Data1FeedData0(
				ping_buffer,
				fifo_feed_0,
				LAYER_IN_NUM_T_prev,
				LAYER_IN_IMG_H_T_prev,
				LAYER_FILTER_S_prev,
				LAYER_STRIDE_prev,
				LAYER_ROW_IL_FACTOR_prev,
				LAYER_COL_IL_FACTOR_prev);
	} else {
		U1_Data1FeedData0(
				pong_buffer,
				fifo_feed_0,
				LAYER_IN_NUM_T_prev,
				LAYER_IN_IMG_H_T_prev,
				LAYER_FILTER_S_prev,
				LAYER_STRIDE_prev,
				LAYER_ROW_IL_FACTOR_prev,
				LAYER_COL_IL_FACTOR_prev);
	}
}

/**
 *  This file is automatically generated by PolySA CodeGen.
 *  Version: 1.0
 *  Authos: Jie Wang
 */

//#include "common_header_U1.h"

void U1_Data2WriteData0(
		U1_data_t2 buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR][U1_DATA2_FC_SIMD_FACTOR],
		stream<U1_Data2TransferChannelType> &fifo_transfer_in,
		stream<U1_Data2TransferChannelType> &fifo_transfer_out,
		unsigned int engine_id,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint LAYER_COL_IL_FACTOR,
		uint LAYER_STRIDE
){
#pragma HLS INLINE off

	bool LAST_ENGINE = (engine_id == 8 / U1_DATA2_FC_SPLIT_FACTOR - 1);

	bool more_to_read_from_buffer = true;
	bool more_to_collect_from_sys_arr = true;
	bool data_is_from_local_buffer;
	bool data_is_from_external_buffer;
	ap_uint<8> oo = 0;
	ap_uint<5> h = 0;
	ap_uint<5> h_bound = LAYER_IN_IMG_H_T / LAYER_STRIDE;
	ap_uint<7> w = 0;
	ap_uint<7> w_bound = LAYER_IN_IMG_W_T / LAYER_STRIDE;
	bool done = 0;

	while(!done){
#pragma HLS PIPELINE II=1
		ap_uint<18> local_buf_idx = h * LAYER_COL_IL_FACTOR * LAYER_OUT_NUM_T + (w % LAYER_COL_IL_FACTOR) * LAYER_OUT_NUM_T + oo * U1_DATA2_FC_SIMD_FACTOR;
		if (w >= engine_id * LAYER_COL_IL_FACTOR){
			ap_uint<7> collector_id = w / LAYER_COL_IL_FACTOR;
			data_is_from_local_buffer = (collector_id == engine_id);
			data_is_from_external_buffer = !data_is_from_local_buffer;

			U1_Data2TransferChannelType data_write_to_fifo;

			if (data_is_from_external_buffer){
				data_write_to_fifo = fifo_transfer_in.read();
			} else {
				U1_data_t2 data0 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][0];
				ap_uint<U1_DATA2_WIDTH> data0_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data0);
				U1_data_t2 data1 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][1];
				ap_uint<U1_DATA2_WIDTH> data1_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data1);
				U1_data_t2 data2 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][2];
				ap_uint<U1_DATA2_WIDTH> data2_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data2);
				U1_data_t2 data3 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][3];
				ap_uint<U1_DATA2_WIDTH> data3_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data3);
				U1_data_t2 data4 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][4];
				ap_uint<U1_DATA2_WIDTH> data4_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data4);
				U1_data_t2 data5 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][5];
				ap_uint<U1_DATA2_WIDTH> data5_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data5);
				U1_data_t2 data6 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][6];
				ap_uint<U1_DATA2_WIDTH> data6_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data6);
				U1_data_t2 data7 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][7];
				ap_uint<U1_DATA2_WIDTH> data7_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data7);
				ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> pack_data = (
						data7_cast,
						data6_cast,
						data5_cast,
						data4_cast,
						data3_cast,
						data2_cast,
						data1_cast,
						data0_cast
				);
				data_write_to_fifo.data = pack_data;
			}

			fifo_transfer_out.write(data_write_to_fifo);
		}
		w++;
		if (w == w_bound){
			w = 0;
			h++;
			if (h == h_bound){
				h = 0;
				oo++;
				if (oo == LAYER_OUT_NUM_T / U1_DATA2_FC_SIMD_FACTOR){
					oo = 0;
					done = 1;
				}
			}
		}
	}

}

void U1_Data2WriteDataLast(
		U1_data_t2 buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR][U1_DATA2_FC_SIMD_FACTOR],
		stream<U1_Data2TransferChannelType> &fifo_transfer_out,
		unsigned int engine_id,
		uint LAYER_OUT_NUM_T,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_IN_IMG_W_T,
		uint LAYER_COL_IL_FACTOR,
		uint LAYER_STRIDE
){
#pragma HLS INLINE off

	bool LAST_ENGINE = (engine_id == 8 / U1_DATA2_FC_SPLIT_FACTOR - 1);

	bool more_to_read_from_buffer = true;
	bool more_to_collect_from_sys_arr = true;
	bool data_is_from_local_buffer;
	bool data_is_from_external_buffer;
	ap_uint<8> oo = 0;
	ap_uint<5> h = 0;
	ap_uint<5> h_bound = LAYER_IN_IMG_H_T / LAYER_STRIDE;
	ap_uint<7> w = 0;
	ap_uint<7> w_bound = LAYER_IN_IMG_W_T / LAYER_STRIDE;
	bool done = 0;

	while(!done){
#pragma HLS PIPELINE II=1
		ap_uint<18> local_buf_idx = h * LAYER_COL_IL_FACTOR * LAYER_OUT_NUM_T + (w % LAYER_COL_IL_FACTOR) * LAYER_OUT_NUM_T + oo * U1_DATA2_FC_SIMD_FACTOR;
		if (w >= engine_id * LAYER_COL_IL_FACTOR){
			ap_uint<7> collector_id = w / LAYER_COL_IL_FACTOR;
			data_is_from_local_buffer = (collector_id == engine_id);
			data_is_from_external_buffer = !data_is_from_local_buffer;

			U1_Data2TransferChannelType data_write_to_fifo;

			if (data_is_from_external_buffer){
			} else {
				U1_data_t2 data0 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][0];
				ap_uint<U1_DATA2_WIDTH> data0_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data0);
				U1_data_t2 data1 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][1];
				ap_uint<U1_DATA2_WIDTH> data1_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data1);
				U1_data_t2 data2 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][2];
				ap_uint<U1_DATA2_WIDTH> data2_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data2);
				U1_data_t2 data3 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][3];
				ap_uint<U1_DATA2_WIDTH> data3_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data3);
				U1_data_t2 data4 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][4];
				ap_uint<U1_DATA2_WIDTH> data4_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data4);
				U1_data_t2 data5 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][5];
				ap_uint<U1_DATA2_WIDTH> data5_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data5);
				U1_data_t2 data6 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][6];
				ap_uint<U1_DATA2_WIDTH> data6_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data6);
				U1_data_t2 data7 = buffer[0][local_buf_idx / U1_DATA2_FC_SIMD_FACTOR][7];
				ap_uint<U1_DATA2_WIDTH> data7_cast = Reinterpret<ap_uint<U1_DATA2_WIDTH> >(data7);
				ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> pack_data = (
						data7_cast,
						data6_cast,
						data5_cast,
						data4_cast,
						data3_cast,
						data2_cast,
						data1_cast,
						data0_cast
				);
				data_write_to_fifo.data = pack_data;
			}

			fifo_transfer_out.write(data_write_to_fifo);
		}
		w++;
		if (w == w_bound){
			w = 0;
			h++;
			if (h == h_bound){
				h = 0;
				oo++;
				if (oo == LAYER_OUT_NUM_T / U1_DATA2_FC_SIMD_FACTOR){
					oo = 0;
					done = 1;
				}
			}
		}
	}

}

void U1_Data2ReadData0(
		U1_data_t2 buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR][U1_DATA2_FC_SIMD_FACTOR],
		stream<U1_Data2PEChannelType> &fifo_collect_0,
		uint LAYER_IN_IMG_H_T,
		uint LAYER_ROW_IL_FACTOR,
		uint LAYER_COL_IL_FACTOR,
		uint LAYER_STRIDE
){
#pragma HLS INLINE off

	bool more_to_collect_from_sys_arr = true;
	ap_uint<1> buffer_gs_id = 0;
	ap_uint<14> buffer_read_counter = 0;
	ap_uint<5> c0_counter = 0;
	ap_uint<5> c1_counter = 0;
	ap_uint<4> c2_counter = 0;
	ap_uint<4> c3_counter = 0;
	ap_uint<5> c0_counter_bound = LAYER_IN_IMG_H_T / LAYER_STRIDE;

	while(more_to_collect_from_sys_arr){
#pragma HLS PIPELINE II=1
		ap_uint<14> buffer_ind_to_collect_from_sys_arr = c0_counter * LAYER_COL_IL_FACTOR * U1_SA_ROWS * LAYER_ROW_IL_FACTOR + c2_counter * U1_SA_ROWS * LAYER_ROW_IL_FACTOR + ((U1_SA_ROWS - 1 - c3_counter) * LAYER_ROW_IL_FACTOR + c1_counter);

		U1_Data2PEChannelType data_to_collect_0;
		data_to_collect_0 = fifo_collect_0.read();
		buffer[0][buffer_ind_to_collect_from_sys_arr / U1_DATA2_FC_SIMD_FACTOR][buffer_ind_to_collect_from_sys_arr % U1_DATA2_FC_SIMD_FACTOR] = data_to_collect_0.data;

		// counter logic
		c0_counter++;
		if (c0_counter == c0_counter_bound){
			c0_counter = 0;
			c1_counter++;
			if (c1_counter == LAYER_ROW_IL_FACTOR){
				c1_counter = 0;
				c2_counter++;
				if (c2_counter == LAYER_COL_IL_FACTOR){
					c2_counter = 0;
					c3_counter++;
					if (c3_counter == U1_SA_ROWS){
						c3_counter = 0;
						more_to_collect_from_sys_arr = false;
					}
				}
			}
		}
	}
}

void U1_DataCollect2Engine0(
		stream<U1_Data2TransferChannelType> &fifo_transfer_in,
		stream<U1_Data2TransferChannelType> &fifo_transfer_out,
		stream<U1_Data2PEChannelType> &fifo_collect_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in0, // from PE
		stream<uint> &fifo_config_in1, // from other engines
		stream<uint> &fifo_config_out
){
#pragma HLS DATA_PACK variable=fifo_transfer_in
#pragma HLS DATA_PACK variable=fifo_transfer_out
#pragma HLS DATA_PACK variable=fifo_collect_0
#pragma HLS INLINE off

	uint LAYER_OUT_NUM_T_prev;
	uint LAYER_IN_IMG_H_T_prev;
	uint LAYER_IN_IMG_W_T_prev;
	uint LAYER_COL_IL_FACTOR_prev;
	uint LAYER_STRIDE_prev;
	uint task_iter = 0;
	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in0.read();
	uint LAYER_OUT_NUM_T = fifo_config_in0.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in0.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in0.read();
	uint LAYER_FILTER_S = fifo_config_in0.read();
	uint LAYER_TASK_NUM1 = fifo_config_in0.read();
	uint LAYER_TASK_NUM2 = fifo_config_in0.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in0.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in0.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in0.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in0.read();
	uint LAYER_STRIDE = fifo_config_in0.read();
	uint LAYER_BATCH = fifo_config_in0.read();

	// dummpy read
	LAYER_IN_NUM_T = fifo_config_in1.read();
	LAYER_OUT_NUM_T = fifo_config_in1.read();
	LAYER_IN_IMG_H_T = fifo_config_in1.read();
	LAYER_IN_IMG_W_T = fifo_config_in1.read();
	LAYER_FILTER_S = fifo_config_in1.read();
	LAYER_TASK_NUM1 = fifo_config_in1.read();
	LAYER_TASK_NUM2 = fifo_config_in1.read();
	LAYER_LOCAL_ACCUM_NUM = fifo_config_in1.read();
	LAYER_LOCAL_REG_NUM = fifo_config_in1.read();
	LAYER_ROW_IL_FACTOR = fifo_config_in1.read();
	LAYER_COL_IL_FACTOR = fifo_config_in1.read();
	LAYER_STRIDE = fifo_config_in1.read();
	LAYER_BATCH = fifo_config_in1.read();

	// write out configurations
	fifo_config_out.write(LAYER_IN_NUM_T);
	fifo_config_out.write(LAYER_OUT_NUM_T);
	fifo_config_out.write(LAYER_IN_IMG_H_T);
	fifo_config_out.write(LAYER_IN_IMG_W_T);
	fifo_config_out.write(LAYER_FILTER_S);
	fifo_config_out.write(LAYER_TASK_NUM1);
	fifo_config_out.write(LAYER_TASK_NUM2);
	fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out.write(LAYER_COL_IL_FACTOR);
	fifo_config_out.write(LAYER_STRIDE);
	fifo_config_out.write(LAYER_BATCH);

	U1_data_t2 ping_buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR][U1_DATA2_FC_SIMD_FACTOR];
	U1_data_t2 pong_buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR][U1_DATA2_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=ping_buffer dim=3 complete
#pragma HLS ARRAY_PARTITION variable=pong_buffer dim=3 complete
#pragma HLS DATA_PACK variable=ping_buffer
#pragma HLS DATA_PACK variable=pong_buffer

	unsigned int initial_round = 0;
	bool done = 0;
	ap_uint<2> layer_iter = 0;
	bool layer_start = 0;
	while(!done){
		if (layer_start){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in0.read();
			LAYER_OUT_NUM_T = fifo_config_in0.read();
			LAYER_IN_IMG_H_T = fifo_config_in0.read();
			LAYER_IN_IMG_W_T = fifo_config_in0.read();
			LAYER_FILTER_S = fifo_config_in0.read();
			LAYER_TASK_NUM1 = fifo_config_in0.read();
			LAYER_TASK_NUM2 = fifo_config_in0.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in0.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in0.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in0.read();
			LAYER_COL_IL_FACTOR = fifo_config_in0.read();
			LAYER_STRIDE = fifo_config_in0.read();
			LAYER_BATCH = fifo_config_in0.read();

			// dummpy read
			LAYER_IN_NUM_T = fifo_config_in1.read();
			LAYER_OUT_NUM_T = fifo_config_in1.read();
			LAYER_IN_IMG_H_T = fifo_config_in1.read();
			LAYER_IN_IMG_W_T = fifo_config_in1.read();
			LAYER_FILTER_S = fifo_config_in1.read();
			LAYER_TASK_NUM1 = fifo_config_in1.read();
			LAYER_TASK_NUM2 = fifo_config_in1.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in1.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in1.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in1.read();
			LAYER_COL_IL_FACTOR = fifo_config_in1.read();
			LAYER_STRIDE = fifo_config_in1.read();
			LAYER_BATCH = fifo_config_in1.read();

			// write out configurations
			fifo_config_out.write(LAYER_IN_NUM_T);
			fifo_config_out.write(LAYER_OUT_NUM_T);
			fifo_config_out.write(LAYER_IN_IMG_H_T);
			fifo_config_out.write(LAYER_IN_IMG_W_T);
			fifo_config_out.write(LAYER_FILTER_S);
			fifo_config_out.write(LAYER_TASK_NUM1);
			fifo_config_out.write(LAYER_TASK_NUM2);
			fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out.write(LAYER_COL_IL_FACTOR);
			fifo_config_out.write(LAYER_STRIDE);
			fifo_config_out.write(LAYER_BATCH);

			layer_start = 0;
		}

		if (initial_round == 0){
			U1_Data2ReadData0(
					ping_buffer,
					fifo_collect_0,
					LAYER_IN_IMG_H_T,
					LAYER_ROW_IL_FACTOR,
					LAYER_COL_IL_FACTOR,
					LAYER_STRIDE
			);
		} else {
			if (initial_round % 2 == 1){
				U1_Data2ReadData0(
						pong_buffer,
						fifo_collect_0,
						LAYER_IN_IMG_H_T,
						LAYER_ROW_IL_FACTOR,
						LAYER_COL_IL_FACTOR,
						LAYER_STRIDE
				);
				U1_Data2WriteData0(ping_buffer, fifo_transfer_in, fifo_transfer_out, engine_id, LAYER_OUT_NUM_T_prev, LAYER_IN_IMG_H_T_prev, LAYER_IN_IMG_W_T_prev, LAYER_COL_IL_FACTOR_prev, LAYER_STRIDE_prev);
			} else {
				U1_Data2ReadData0(
						ping_buffer,
						fifo_collect_0,
						LAYER_IN_IMG_H_T,
						LAYER_ROW_IL_FACTOR,
						LAYER_COL_IL_FACTOR,
						LAYER_STRIDE
				);
				U1_Data2WriteData0(pong_buffer, fifo_transfer_in, fifo_transfer_out, engine_id, LAYER_OUT_NUM_T_prev, LAYER_IN_IMG_H_T_prev, LAYER_IN_IMG_W_T_prev, LAYER_COL_IL_FACTOR_prev, LAYER_STRIDE_prev);
			}
		}
		initial_round++;
		LAYER_OUT_NUM_T_prev = LAYER_OUT_NUM_T;
		LAYER_IN_IMG_H_T_prev = LAYER_IN_IMG_H_T;
		LAYER_IN_IMG_W_T_prev = LAYER_IN_IMG_W_T;
		LAYER_COL_IL_FACTOR_prev = LAYER_COL_IL_FACTOR;
		LAYER_STRIDE_prev = LAYER_STRIDE;

		task_iter += 1;
		if (task_iter == LAYER_TASK_NUM2){
			task_iter = 0;
			layer_iter += 1;
			layer_start = 1;
			if (layer_iter == LAYER_BATCH){
				layer_iter = 0;
				done = 1;
			}
		}
	}

	if (initial_round % 2 == 1){
		U1_Data2WriteData0(ping_buffer, fifo_transfer_in, fifo_transfer_out, engine_id, LAYER_OUT_NUM_T_prev, LAYER_IN_IMG_H_T_prev, LAYER_IN_IMG_W_T_prev, LAYER_COL_IL_FACTOR_prev, LAYER_STRIDE_prev);
	} else {
		U1_Data2WriteData0(pong_buffer, fifo_transfer_in, fifo_transfer_out, engine_id, LAYER_OUT_NUM_T_prev, LAYER_IN_IMG_H_T_prev, LAYER_IN_IMG_W_T_prev, LAYER_COL_IL_FACTOR_prev, LAYER_STRIDE_prev);
	}
}

void U1_DataCollect2Engine0_wrapper(
		stream<U1_Data2TransferChannelType> &fifo_transfer_in,
		stream<U1_Data2TransferChannelType> &fifo_transfer_out,
		stream<U1_Data2PEChannelType> &fifo_collect_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in0,
		stream<uint> &fifo_config_in1,
		stream<uint> &fifo_config_out
){
	U1_DataCollect2Engine0(
			fifo_transfer_in,
			fifo_transfer_out,
			fifo_collect_0,
			engine_id,
			fifo_config_in0,
			fifo_config_in1,
			fifo_config_out
	);
}

void U1_DataCollect2EngineLast(
		stream<U1_Data2TransferChannelType> &fifo_transfer_out,
		stream<U1_Data2PEChannelType> &fifo_collect_0,
		unsigned int engine_id,
		stream<uint> &fifo_config_in0,
		stream<uint> &fifo_config_out
){
#pragma HLS DATA_PACK variable=fifo_transfer_out
#pragma HLS DATA_PACK variable=fifo_collect_0
#pragma HLS INLINE off

	uint LAYER_OUT_NUM_T_prev;
	uint LAYER_IN_IMG_H_T_prev;
	uint LAYER_IN_IMG_W_T_prev;
	uint LAYER_COL_IL_FACTOR_prev;
	uint LAYER_STRIDE_prev;

	uint task_iter = 0;
	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in0.read();
	uint LAYER_OUT_NUM_T = fifo_config_in0.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in0.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in0.read();
	uint LAYER_FILTER_S = fifo_config_in0.read();
	uint LAYER_TASK_NUM1 = fifo_config_in0.read();
	uint LAYER_TASK_NUM2 = fifo_config_in0.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in0.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in0.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in0.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in0.read();
	uint LAYER_STRIDE = fifo_config_in0.read();
	uint LAYER_BATCH = fifo_config_in0.read();

	// write out configurations
	fifo_config_out.write(LAYER_IN_NUM_T);
	fifo_config_out.write(LAYER_OUT_NUM_T);
	fifo_config_out.write(LAYER_IN_IMG_H_T);
	fifo_config_out.write(LAYER_IN_IMG_W_T);
	fifo_config_out.write(LAYER_FILTER_S);
	fifo_config_out.write(LAYER_TASK_NUM1);
	fifo_config_out.write(LAYER_TASK_NUM2);
	fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out.write(LAYER_COL_IL_FACTOR);
	fifo_config_out.write(LAYER_STRIDE);
	fifo_config_out.write(LAYER_BATCH);

	U1_data_t2 ping_buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR][U1_DATA2_FC_SIMD_FACTOR];
	U1_data_t2 pong_buffer[U1_DATA2_FC_GROUP_FACTOR][U1_DATA2_BUF_SIZE / U1_DATA2_FC_SIMD_FACTOR][U1_DATA2_FC_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=ping_buffer dim=3 complete
#pragma HLS ARRAY_PARTITION variable=pong_buffer dim=3 complete
#pragma HLS DATA_PACK variable=ping_buffer
#pragma HLS DATA_PACK variable=pong_buffer

	unsigned int initial_round = 0;
	bool done = 0;
	ap_uint<2> layer_iter = 0;
	bool layer_start = 0;
	while(!done){
		if (layer_start){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in0.read();
			LAYER_OUT_NUM_T = fifo_config_in0.read();
			LAYER_IN_IMG_H_T = fifo_config_in0.read();
			LAYER_IN_IMG_W_T = fifo_config_in0.read();
			LAYER_FILTER_S = fifo_config_in0.read();
			LAYER_TASK_NUM1 = fifo_config_in0.read();
			LAYER_TASK_NUM2 = fifo_config_in0.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in0.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in0.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in0.read();
			LAYER_COL_IL_FACTOR = fifo_config_in0.read();
			LAYER_STRIDE = fifo_config_in0.read();
			LAYER_BATCH = fifo_config_in0.read();

			// write out configurations
			fifo_config_out.write(LAYER_IN_NUM_T);
			fifo_config_out.write(LAYER_OUT_NUM_T);
			fifo_config_out.write(LAYER_IN_IMG_H_T);
			fifo_config_out.write(LAYER_IN_IMG_W_T);
			fifo_config_out.write(LAYER_FILTER_S);
			fifo_config_out.write(LAYER_TASK_NUM1);
			fifo_config_out.write(LAYER_TASK_NUM2);
			fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out.write(LAYER_COL_IL_FACTOR);
			fifo_config_out.write(LAYER_STRIDE);
			fifo_config_out.write(LAYER_BATCH);

			layer_start = 0;
		}

		if (initial_round == 0){
			U1_Data2ReadData0(
					ping_buffer,
					fifo_collect_0,
					LAYER_IN_IMG_H_T,
					LAYER_ROW_IL_FACTOR,
					LAYER_COL_IL_FACTOR,
					LAYER_STRIDE
			);
		} else {
			if (initial_round % 2 == 1){
				U1_Data2ReadData0(
						pong_buffer,
						fifo_collect_0,
						LAYER_IN_IMG_H_T,
						LAYER_ROW_IL_FACTOR,
						LAYER_COL_IL_FACTOR,
						LAYER_STRIDE
				);
				U1_Data2WriteDataLast(ping_buffer, fifo_transfer_out, engine_id, LAYER_OUT_NUM_T_prev, LAYER_IN_IMG_H_T_prev, LAYER_IN_IMG_W_T_prev, LAYER_COL_IL_FACTOR_prev, LAYER_STRIDE_prev);
			} else {
				U1_Data2ReadData0(
						ping_buffer,
						fifo_collect_0,
						LAYER_IN_IMG_H_T,
						LAYER_ROW_IL_FACTOR,
						LAYER_COL_IL_FACTOR,
						LAYER_STRIDE
				);
				U1_Data2WriteDataLast(pong_buffer, fifo_transfer_out, engine_id, LAYER_OUT_NUM_T_prev, LAYER_IN_IMG_H_T_prev, LAYER_IN_IMG_W_T_prev, LAYER_COL_IL_FACTOR_prev, LAYER_STRIDE_prev);
			}
		}
		initial_round++;
		LAYER_OUT_NUM_T_prev = LAYER_OUT_NUM_T;
		LAYER_IN_IMG_H_T_prev = LAYER_IN_IMG_H_T;
		LAYER_IN_IMG_W_T_prev = LAYER_IN_IMG_W_T;
		LAYER_COL_IL_FACTOR_prev = LAYER_COL_IL_FACTOR;
		LAYER_STRIDE_prev = LAYER_STRIDE;

		task_iter += 1;
		if (task_iter == LAYER_TASK_NUM2){
			task_iter = 0;
			layer_iter += 1;
			layer_start = 1;
			if (layer_iter == LAYER_BATCH){
				layer_iter = 0;
				done = 1;
			}
		}
	}

	if (initial_round % 2 == 1){
		U1_Data2WriteDataLast(ping_buffer, fifo_transfer_out, engine_id, LAYER_OUT_NUM_T_prev, LAYER_IN_IMG_H_T_prev, LAYER_IN_IMG_W_T_prev, LAYER_COL_IL_FACTOR_prev, LAYER_STRIDE_prev);
	} else {
		U1_Data2WriteDataLast(pong_buffer, fifo_transfer_out, engine_id, LAYER_OUT_NUM_T_prev, LAYER_IN_IMG_H_T_prev, LAYER_IN_IMG_W_T_prev, LAYER_COL_IL_FACTOR_prev, LAYER_STRIDE_prev);
	}
}

/**
 *  This file is automatically generated by PolySA CodeGen.
 *  Version: 1.0
 *  Authos: Jie Wang
 */

//#include "common_header_U1.h"

void U1_PE_MAC(
		U1_Data0SIMDType op0,
		U1_Data1SIMDType op1,
		U1_data_t2* op2,
		bool init
){
#pragma HLS INLINE
#pragma HLS DATA_PACK variable=op0
#pragma HLS DATA_PACK variable=op1
	ap_uint<256> op0_data = op0;
	ap_uint<256> op1_data = op1;

	float op0_u[U1_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=op0_u complete
	float op1_u[U1_SIMD_FACTOR];
#pragma HLS ARRAY_PARTITION variable=op1_u complete

	for (int i = 0; i < U1_SIMD_FACTOR; i++){
#pragma HLS UNROLL
		ap_uint<U1_DATA0_WIDTH> sel0 = op0_data(U1_DATA0_WIDTH-1, 0);
		op0_u[i] = Reinterpret<U1_data_t0>(sel0);
		op0_data = op0_data >> U1_DATA0_WIDTH;
		ap_uint<U1_DATA1_WIDTH> sel1 = op1_data(U1_DATA1_WIDTH-1, 0);
		op1_u[i] = Reinterpret<U1_data_t1>(sel1);
		op1_data = op1_data >> U1_DATA1_WIDTH;
	}

	U1_data_t2 sum = (init == 1)? (U1_data_t2) 0: *op2;

	U1_data_t2 mult0 = op0_u[0] * op1_u[0];
	U1_data_t2 mult1 = op0_u[1] * op1_u[1];
	U1_data_t2 mult2 = op0_u[2] * op1_u[2];
	U1_data_t2 mult3 = op0_u[3] * op1_u[3];
	U1_data_t2 mult4 = op0_u[4] * op1_u[4];
	U1_data_t2 mult5 = op0_u[5] * op1_u[5];
	U1_data_t2 mult6 = op0_u[6] * op1_u[6];
	U1_data_t2 mult7 = op0_u[7] * op1_u[7];

	U1_data_t2 sum2_0 = mult0 + mult1;
	U1_data_t2 sum2_1 = mult2 + mult3;
	U1_data_t2 sum2_2 = mult4 + mult5;
	U1_data_t2 sum2_3 = mult6 + mult7;

	U1_data_t2 sum1_0 = sum2_0 + sum2_1;
	U1_data_t2 sum1_1 = sum2_2 + sum2_3;

	U1_data_t2 sum0_0 = sum1_0 + sum1_1;

	sum += sum0_0;

	*op2 = sum;
}

void U1_op0_transfer(
		stream<U1_Data0PEChannelType> &fifo0_in,
		stream<U1_Data0PEChannelType> &fifo0_out,
		stream<U1_Data0PEChannelType> &fifo0_local,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
#pragma HLS DATA_PACK variable=fifo0_in
#pragma HLS DATA_PACK variable=fifo0_out
#pragma HLS DATA_PACK variable=fifo0_local
#pragma HLS INLINE off

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// write out configurations
	fifo_config_out.write(LAYER_IN_NUM_T);
	fifo_config_out.write(LAYER_OUT_NUM_T);
	fifo_config_out.write(LAYER_IN_IMG_H_T);
	fifo_config_out.write(LAYER_IN_IMG_W_T);
	fifo_config_out.write(LAYER_FILTER_S);
	fifo_config_out.write(LAYER_TASK_NUM1);
	fifo_config_out.write(LAYER_TASK_NUM2);
	fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out.write(LAYER_COL_IL_FACTOR);
	fifo_config_out.write(LAYER_STRIDE);
	fifo_config_out.write(LAYER_BATCH);

	ap_uint<2> layer_iter = 0;
	bool done1 = 0;
	while(!done1){
		if (layer_iter > 0){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			LAYER_BATCH = fifo_config_in.read();

			// write out configurations
			fifo_config_out.write(LAYER_IN_NUM_T);
			fifo_config_out.write(LAYER_OUT_NUM_T);
			fifo_config_out.write(LAYER_IN_IMG_H_T);
			fifo_config_out.write(LAYER_IN_IMG_W_T);
			fifo_config_out.write(LAYER_FILTER_S);
			fifo_config_out.write(LAYER_TASK_NUM1);
			fifo_config_out.write(LAYER_TASK_NUM2);
			fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out.write(LAYER_COL_IL_FACTOR);
			fifo_config_out.write(LAYER_STRIDE);
			fifo_config_out.write(LAYER_BATCH);
		}

		ap_uint<38> task_num = 0;
		ap_uint<12> la_counter = 0;
		ap_uint<11> local_reg_id = 0;
		bool done2 = 0;
		while(!done2){
#pragma HLS PIPELINE II=1
			U1_Data0PEChannelType fifo0_in_data;
			fifo0_in_data = fifo0_in.read();
			fifo0_out.write(fifo0_in_data);
			fifo0_local.write(fifo0_in_data);
			local_reg_id++;
			if (local_reg_id == LAYER_LOCAL_REG_NUM){
				local_reg_id = 0;
				la_counter++;
				if (la_counter == LAYER_LOCAL_ACCUM_NUM){
					la_counter = 0;
					task_num++;
					if (task_num == LAYER_TASK_NUM1){
						task_num = 0;
						done2 = 1;
					}
				}
			}
		}
		layer_iter++;
		if (layer_iter == LAYER_BATCH){
			layer_iter = 0;
			done1 = 1;
		}
	}
}

void U1_op0_transfer_wrapper(
		stream<U1_Data0PEChannelType> &fifo0_in,
		stream<U1_Data0PEChannelType> &fifo0_out,
		stream<U1_Data0PEChannelType> &fifo0_local,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
	U1_op0_transfer(
			fifo0_in,
			fifo0_out,
			fifo0_local,
			fifo_config_in,
			fifo_config_out
	);
}

void U1_op0_transfer_last(
		stream<U1_Data0PEChannelType> &fifo0_in,
		stream<U1_Data0PEChannelType> &fifo0_local,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
#pragma HLS DATA_PACK variable=fifo0_in
#pragma HLS DATA_PACK variable=fifo0_local
#pragma HLS INLINE off

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// write out configurations
	fifo_config_out.write(LAYER_IN_NUM_T);
	fifo_config_out.write(LAYER_OUT_NUM_T);
	fifo_config_out.write(LAYER_IN_IMG_H_T);
	fifo_config_out.write(LAYER_IN_IMG_W_T);
	fifo_config_out.write(LAYER_FILTER_S);
	fifo_config_out.write(LAYER_TASK_NUM1);
	fifo_config_out.write(LAYER_TASK_NUM2);
	fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out.write(LAYER_COL_IL_FACTOR);
	fifo_config_out.write(LAYER_STRIDE);
	fifo_config_out.write(LAYER_BATCH);

	ap_uint<2> layer_iter = 0;
	bool done1 = 0;
	while(!done1){
		if (layer_iter > 0){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			LAYER_BATCH = fifo_config_in.read();

			// write out configurations
			fifo_config_out.write(LAYER_IN_NUM_T);
			fifo_config_out.write(LAYER_OUT_NUM_T);
			fifo_config_out.write(LAYER_IN_IMG_H_T);
			fifo_config_out.write(LAYER_IN_IMG_W_T);
			fifo_config_out.write(LAYER_FILTER_S);
			fifo_config_out.write(LAYER_TASK_NUM1);
			fifo_config_out.write(LAYER_TASK_NUM2);
			fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out.write(LAYER_COL_IL_FACTOR);
			fifo_config_out.write(LAYER_STRIDE);
			fifo_config_out.write(LAYER_BATCH);
		}

		ap_uint<38> task_num = 0;
		ap_uint<12> la_counter = 0;
		ap_uint<11> local_reg_id = 0;
		bool done2 = 0;
		while(!done2){
#pragma HLS PIPELINE II=1
			U1_Data0PEChannelType fifo0_in_data;
			fifo0_in_data = fifo0_in.read();
			fifo0_local.write(fifo0_in_data);
			local_reg_id++;
			if (local_reg_id == LAYER_LOCAL_REG_NUM){
				local_reg_id = 0;
				la_counter++;
				if (la_counter == LAYER_LOCAL_ACCUM_NUM){
					la_counter = 0;
					task_num++;
					if (task_num == LAYER_TASK_NUM1){
						task_num = 0;
						done2 = 1;
					}
				}
			}
		}
		layer_iter++;
		if (layer_iter == LAYER_BATCH){
			layer_iter = 0;
			done1 = 1;
		}
	}
}

void U1_op0_transfer_last_wrapper(
		stream<U1_Data0PEChannelType> &fifo0_in,
		stream<U1_Data0PEChannelType> &fifo0_local,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
	U1_op0_transfer_last(
			fifo0_in,
			fifo0_local,
			fifo_config_in,
			fifo_config_out
	);
}

void U1_op1_transfer(
		stream<U1_Data1PEChannelType> &fifo1_in,
		stream<U1_Data1PEChannelType> &fifo1_out,
		stream<U1_Data1PEChannelType> &fifo1_local,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
#pragma HLS DATA_PACK variable=fifo1_in
#pragma HLS DATA_PACK variable=fifo1_out
#pragma HLS DATA_PACK variable=fifo1_local
#pragma HLS INLINE off

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// write out configurations
	fifo_config_out.write(LAYER_IN_NUM_T);
	fifo_config_out.write(LAYER_OUT_NUM_T);
	fifo_config_out.write(LAYER_IN_IMG_H_T);
	fifo_config_out.write(LAYER_IN_IMG_W_T);
	fifo_config_out.write(LAYER_FILTER_S);
	fifo_config_out.write(LAYER_TASK_NUM1);
	fifo_config_out.write(LAYER_TASK_NUM2);
	fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out.write(LAYER_COL_IL_FACTOR);
	fifo_config_out.write(LAYER_STRIDE);
	fifo_config_out.write(LAYER_BATCH);

	ap_uint<2> layer_iter = 0;
	bool done1 = 0;
	while(!done1){
		if (layer_iter > 0){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			LAYER_BATCH = fifo_config_in.read();

			// write out configurations
			fifo_config_out.write(LAYER_IN_NUM_T);
			fifo_config_out.write(LAYER_OUT_NUM_T);
			fifo_config_out.write(LAYER_IN_IMG_H_T);
			fifo_config_out.write(LAYER_IN_IMG_W_T);
			fifo_config_out.write(LAYER_FILTER_S);
			fifo_config_out.write(LAYER_TASK_NUM1);
			fifo_config_out.write(LAYER_TASK_NUM2);
			fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out.write(LAYER_COL_IL_FACTOR);
			fifo_config_out.write(LAYER_STRIDE);
			fifo_config_out.write(LAYER_BATCH);
		}

		ap_uint<38> task_num = 0;
		ap_uint<12> la_counter = 0;
		ap_uint<11> local_reg_id = 0;
		bool done2 = 0;
		while(!done2){
#pragma HLS PIPELINE II=1
			U1_Data1PEChannelType fifo1_in_data;
			fifo1_in_data = fifo1_in.read();
			fifo1_out.write(fifo1_in_data);
			fifo1_local.write(fifo1_in_data);
			local_reg_id++;
			if (local_reg_id == LAYER_LOCAL_REG_NUM){
				local_reg_id = 0;
				la_counter++;
				if (la_counter == LAYER_LOCAL_ACCUM_NUM){
					la_counter = 0;
					task_num++;
					if (task_num == LAYER_TASK_NUM1){
						task_num = 0;
						done2 = 1;
					}
				}
			}
		}
		layer_iter++;
		if (layer_iter == LAYER_BATCH){
			layer_iter = 0;
			done1 = 1;
		}
	}
}

void U1_op1_transfer_wrapper(
		stream<U1_Data1PEChannelType> &fifo1_in,
		stream<U1_Data1PEChannelType> &fifo1_out,
		stream<U1_Data1PEChannelType> &fifo1_local,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
	U1_op1_transfer(
			fifo1_in,
			fifo1_out,
			fifo1_local,
			fifo_config_in,
			fifo_config_out
	);
}

void U1_op1_transfer_last(
		stream<U1_Data1PEChannelType> &fifo1_in,
		stream<U1_Data1PEChannelType> &fifo1_local,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
#pragma HLS DATA_PACK variable=fifo1_in
#pragma HLS DATA_PACK variable=fifo1_local
#pragma HLS INLINE off

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// write out configurations
	fifo_config_out.write(LAYER_IN_NUM_T);
	fifo_config_out.write(LAYER_OUT_NUM_T);
	fifo_config_out.write(LAYER_IN_IMG_H_T);
	fifo_config_out.write(LAYER_IN_IMG_W_T);
	fifo_config_out.write(LAYER_FILTER_S);
	fifo_config_out.write(LAYER_TASK_NUM1);
	fifo_config_out.write(LAYER_TASK_NUM2);
	fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out.write(LAYER_COL_IL_FACTOR);
	fifo_config_out.write(LAYER_STRIDE);
	fifo_config_out.write(LAYER_BATCH);

	ap_uint<2> layer_iter = 0;
	bool done1 = 0;
	while(!done1){
		if (layer_iter > 0){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			LAYER_BATCH = fifo_config_in.read();

			// write out configurations
			fifo_config_out.write(LAYER_IN_NUM_T);
			fifo_config_out.write(LAYER_OUT_NUM_T);
			fifo_config_out.write(LAYER_IN_IMG_H_T);
			fifo_config_out.write(LAYER_IN_IMG_W_T);
			fifo_config_out.write(LAYER_FILTER_S);
			fifo_config_out.write(LAYER_TASK_NUM1);
			fifo_config_out.write(LAYER_TASK_NUM2);
			fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out.write(LAYER_COL_IL_FACTOR);
			fifo_config_out.write(LAYER_STRIDE);
			fifo_config_out.write(LAYER_BATCH);
		}

		ap_uint<38> task_num = 0;
		ap_uint<12> la_counter = 0;
		ap_uint<11> local_reg_id = 0;
		bool done2 = 0;
		while(!done2){
#pragma HLS PIPELINE II=1
			U1_Data1PEChannelType fifo1_in_data;
			fifo1_in_data = fifo1_in.read();
			fifo1_local.write(fifo1_in_data);
			local_reg_id++;
			if (local_reg_id == LAYER_LOCAL_REG_NUM){
				local_reg_id = 0;
				la_counter++;
				if (la_counter == LAYER_LOCAL_ACCUM_NUM){
					la_counter = 0;
					task_num++;
					if (task_num == LAYER_TASK_NUM1){
						task_num = 0;
						done2 = 1;
					}
				}
			}
		}
		layer_iter++;
		if (layer_iter == LAYER_BATCH){
			layer_iter = 0;
			done1 = 1;
		}
	}
}

void U1_op1_transfer_last_wrapper(
		stream<U1_Data1PEChannelType> &fifo1_in,
		stream<U1_Data1PEChannelType> &fifo1_local,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
	U1_op1_transfer_last(
			fifo1_in,
			fifo1_local,
			fifo_config_in,
			fifo_config_out
	);
}

void U1_compute(
		stream<U1_Data0PEChannelType> &fifo0_local,
		stream<U1_Data1PEChannelType> &fifo1_local,
		stream<U1_Data2PEChannelType> &fifo2_local,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
#pragma HLS DATA_PACK variable=fifo0_local
#pragma HLS DATA_PACK variable=fifo1_local
#pragma HLS INLINE off

	U1_data_t2 local_buffer[U1_LOCAL_REG_NUM];

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// write out configurations
	fifo_config_out.write(LAYER_IN_NUM_T);
	fifo_config_out.write(LAYER_OUT_NUM_T);
	fifo_config_out.write(LAYER_IN_IMG_H_T);
	fifo_config_out.write(LAYER_IN_IMG_W_T);
	fifo_config_out.write(LAYER_FILTER_S);
	fifo_config_out.write(LAYER_TASK_NUM1);
	fifo_config_out.write(LAYER_TASK_NUM2);
	fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out.write(LAYER_COL_IL_FACTOR);
	fifo_config_out.write(LAYER_STRIDE);
	fifo_config_out.write(LAYER_BATCH);

	ap_uint<2> layer_iter = 0;
	bool done1 = 0;
	while(!done1){
		if (layer_iter > 0){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			LAYER_BATCH = fifo_config_in.read();

			// write out configurations
			fifo_config_out.write(LAYER_IN_NUM_T);
			fifo_config_out.write(LAYER_OUT_NUM_T);
			fifo_config_out.write(LAYER_IN_IMG_H_T);
			fifo_config_out.write(LAYER_IN_IMG_W_T);
			fifo_config_out.write(LAYER_FILTER_S);
			fifo_config_out.write(LAYER_TASK_NUM1);
			fifo_config_out.write(LAYER_TASK_NUM2);
			fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out.write(LAYER_COL_IL_FACTOR);
			fifo_config_out.write(LAYER_STRIDE);
			fifo_config_out.write(LAYER_BATCH);
		}

		ap_uint<38> task_num = 0;
		ap_uint<12> la_counter = 0;
		ap_uint<11> local_reg_id = 0;
		bool done2 = 0;
		while(!done2){
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE inter false variable=local_buffer
			U1_Data0PEChannelType fifo0_in_data;
			fifo0_in_data = fifo0_local.read();
			U1_Data1PEChannelType fifo1_in_data;
			fifo1_in_data = fifo1_local.read();
			bool init = fifo0_in_data.new_pair;
			bool last = fifo0_in_data.last_pair;
			U1_PE_MAC(fifo0_in_data.data, fifo1_in_data.data, &local_buffer[local_reg_id], (init == 1 && la_counter == 0)? 1:0);
			if (la_counter == LAYER_LOCAL_ACCUM_NUM - 1 && last){
				fifo2_local.write(U1_Data2PEChannelType(local_buffer[local_reg_id]));
			}
			local_reg_id++;
			if (local_reg_id == LAYER_LOCAL_REG_NUM){
				local_reg_id = 0;
				la_counter++;
				if (la_counter == LAYER_LOCAL_ACCUM_NUM){
					la_counter = 0;
					task_num++;
					if (task_num == LAYER_TASK_NUM1){
						task_num = 0;
						done2 = 1;
					}
				}
			}
		}
		layer_iter++;
		if (layer_iter == LAYER_BATCH){
			layer_iter = 0;
			done1 = 1;
		}
	}
}

void U1_compute_wrapper(
		stream<U1_Data0PEChannelType> &fifo0_local,
		stream<U1_Data1PEChannelType> &fifo1_local,
		stream<U1_Data2PEChannelType> &fifo2_local,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
	U1_compute(
			fifo0_local,
			fifo1_local,
			fifo2_local,
			fifo_config_in,
			fifo_config_out
	);
}

void U1_res_transfer(
		stream<U1_Data2PEChannelType> &fifo2_local,
		stream<U1_Data2PEChannelType> &fifo2_in,
		stream<U1_Data2PEChannelType> &fifo2_out,
		ap_uint<4> pe_row_id,
		ap_uint<4> pe_col_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
#pragma HLS DATA_PACK variable=fifo2_in
#pragma HLS DATA_PACK variable=fifo2_out
#pragma HLS INLINE off

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// write out configurations
	fifo_config_out.write(LAYER_IN_NUM_T);
	fifo_config_out.write(LAYER_OUT_NUM_T);
	fifo_config_out.write(LAYER_IN_IMG_H_T);
	fifo_config_out.write(LAYER_IN_IMG_W_T);
	fifo_config_out.write(LAYER_FILTER_S);
	fifo_config_out.write(LAYER_TASK_NUM1);
	fifo_config_out.write(LAYER_TASK_NUM2);
	fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out.write(LAYER_COL_IL_FACTOR);
	fifo_config_out.write(LAYER_STRIDE);
	fifo_config_out.write(LAYER_BATCH);

	U1_data_t2 local_buffer[U1_LOCAL_REG_NUM];

	for (ap_uint<2> layer_iter = 0; layer_iter < LAYER_BATCH; layer_iter++){
		if (layer_iter > 0){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			LAYER_BATCH = fifo_config_in.read();

			// write out configurations
			fifo_config_out.write(LAYER_IN_NUM_T);
			fifo_config_out.write(LAYER_OUT_NUM_T);
			fifo_config_out.write(LAYER_IN_IMG_H_T);
			fifo_config_out.write(LAYER_IN_IMG_W_T);
			fifo_config_out.write(LAYER_FILTER_S);
			fifo_config_out.write(LAYER_TASK_NUM1);
			fifo_config_out.write(LAYER_TASK_NUM2);
			fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out.write(LAYER_COL_IL_FACTOR);
			fifo_config_out.write(LAYER_STRIDE);
			fifo_config_out.write(LAYER_BATCH);
		}

		for (ap_uint<29> task_num = 0; task_num < LAYER_TASK_NUM2; task_num++)
		{
			for (ap_uint<11> local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
				if (local_reg_id < LAYER_LOCAL_REG_NUM){
					U1_Data2PEChannelType fifo2_local_data = fifo2_local.read();
					local_buffer[local_reg_id] = fifo2_local_data.data;
				} else {
					break;
				}
			}

			for (int transfer_iter = pe_row_id + 1 - 1; transfer_iter >= 0; transfer_iter--){
				for (ap_uint<11> local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
					if (local_reg_id < LAYER_LOCAL_REG_NUM){
						fifo2_out.write(U1_Data2PEChannelType(local_buffer[local_reg_id]));
						if (transfer_iter > 0){
							U1_Data2PEChannelType fifo2_in_data = fifo2_in.read();
							local_buffer[local_reg_id] = fifo2_in_data.data;
						}
					} else {
						break;
					}
				}
			}
		}
	}
}

void U1_res_transfer_wrapper(
		stream<U1_Data2PEChannelType> &fifo2_local,
		stream<U1_Data2PEChannelType> &fifo2_in,
		stream<U1_Data2PEChannelType> &fifo2_out,
		ap_uint<4> pe_row_id,
		ap_uint<4> pe_col_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
	U1_res_transfer(
			fifo2_local,
			fifo2_in,
			fifo2_out,
			pe_row_id,
			pe_col_id,
			fifo_config_in,
			fifo_config_out
	);
}

void U1_res_transfer_first(
		stream<U1_Data2PEChannelType> &fifo2_local,
		stream<U1_Data2PEChannelType> &fifo2_out,
		ap_uint<4> pe_row_id,
		ap_uint<4> pe_col_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
#pragma HLS DATA_PACK variable=fifo2_out
#pragma HLS INLINE off

	// read in configurations
	uint LAYER_IN_NUM_T = fifo_config_in.read();
	uint LAYER_OUT_NUM_T = fifo_config_in.read();
	uint LAYER_IN_IMG_H_T = fifo_config_in.read();
	uint LAYER_IN_IMG_W_T = fifo_config_in.read();
	uint LAYER_FILTER_S = fifo_config_in.read();
	uint LAYER_TASK_NUM1 = fifo_config_in.read();
	uint LAYER_TASK_NUM2 = fifo_config_in.read();
	uint LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
	uint LAYER_LOCAL_REG_NUM = fifo_config_in.read();
	uint LAYER_ROW_IL_FACTOR = fifo_config_in.read();
	uint LAYER_COL_IL_FACTOR = fifo_config_in.read();
	uint LAYER_STRIDE = fifo_config_in.read();
	uint LAYER_BATCH = fifo_config_in.read();

	// write out configurations
	fifo_config_out.write(LAYER_IN_NUM_T);
	fifo_config_out.write(LAYER_OUT_NUM_T);
	fifo_config_out.write(LAYER_IN_IMG_H_T);
	fifo_config_out.write(LAYER_IN_IMG_W_T);
	fifo_config_out.write(LAYER_FILTER_S);
	fifo_config_out.write(LAYER_TASK_NUM1);
	fifo_config_out.write(LAYER_TASK_NUM2);
	fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
	fifo_config_out.write(LAYER_LOCAL_REG_NUM);
	fifo_config_out.write(LAYER_ROW_IL_FACTOR);
	fifo_config_out.write(LAYER_COL_IL_FACTOR);
	fifo_config_out.write(LAYER_STRIDE);
	fifo_config_out.write(LAYER_BATCH);

	U1_data_t2 local_buffer[U1_LOCAL_REG_NUM];

	for (ap_uint<2> layer_iter = 0; layer_iter < LAYER_BATCH; layer_iter++){
		if (layer_iter > 0){
			// read in configurations
			LAYER_IN_NUM_T = fifo_config_in.read();
			LAYER_OUT_NUM_T = fifo_config_in.read();
			LAYER_IN_IMG_H_T = fifo_config_in.read();
			LAYER_IN_IMG_W_T = fifo_config_in.read();
			LAYER_FILTER_S = fifo_config_in.read();
			LAYER_TASK_NUM1 = fifo_config_in.read();
			LAYER_TASK_NUM2 = fifo_config_in.read();
			LAYER_LOCAL_ACCUM_NUM = fifo_config_in.read();
			LAYER_LOCAL_REG_NUM = fifo_config_in.read();
			LAYER_ROW_IL_FACTOR = fifo_config_in.read();
			LAYER_COL_IL_FACTOR = fifo_config_in.read();
			LAYER_STRIDE = fifo_config_in.read();
			LAYER_BATCH = fifo_config_in.read();

			// write out configurations
			fifo_config_out.write(LAYER_IN_NUM_T);
			fifo_config_out.write(LAYER_OUT_NUM_T);
			fifo_config_out.write(LAYER_IN_IMG_H_T);
			fifo_config_out.write(LAYER_IN_IMG_W_T);
			fifo_config_out.write(LAYER_FILTER_S);
			fifo_config_out.write(LAYER_TASK_NUM1);
			fifo_config_out.write(LAYER_TASK_NUM2);
			fifo_config_out.write(LAYER_LOCAL_ACCUM_NUM);
			fifo_config_out.write(LAYER_LOCAL_REG_NUM);
			fifo_config_out.write(LAYER_ROW_IL_FACTOR);
			fifo_config_out.write(LAYER_COL_IL_FACTOR);
			fifo_config_out.write(LAYER_STRIDE);
			fifo_config_out.write(LAYER_BATCH);
		}

		for (ap_uint<29> task_num = 0; task_num < LAYER_TASK_NUM2; task_num++)
		{
			for (ap_uint<11> local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
				if (local_reg_id < LAYER_LOCAL_REG_NUM){
					U1_Data2PEChannelType fifo2_local_data = fifo2_local.read();
					local_buffer[local_reg_id] = fifo2_local_data.data;
				} else {
					break;
				}
			}

			for (int transfer_iter = pe_row_id + 1 - 1; transfer_iter >= 0; transfer_iter--){
				for (ap_uint<11> local_reg_id = 0; local_reg_id < U1_LOCAL_REG_NUM; local_reg_id++){
#pragma HLS PIPELINE II=1
					if (local_reg_id < LAYER_LOCAL_REG_NUM){
						fifo2_out.write(U1_Data2PEChannelType(local_buffer[local_reg_id]));
					} else {
						break;
					}
				}
			}
		}
	}
}

void U1_res_transfer_first_wrapper(
		stream<U1_Data2PEChannelType> &fifo2_local,
		stream<U1_Data2PEChannelType> &fifo2_out,
		ap_uint<4> pe_row_id,
		ap_uint<4> pe_col_id,
		stream<uint> &fifo_config_in,
		stream<uint> &fifo_config_out
){
	U1_res_transfer_first(
			fifo2_local,
			fifo2_out,
			pe_row_id,
			pe_col_id,
			fifo_config_in,
			fifo_config_out
	);
}

void kernel(
		stream<ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> > &fifo_cin,
		stream<ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> > &fifo_weight,
		stream<ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> > &fifo_cout,
		stream<U1_ConfigInst> &fifo_kernel_config_in,
		stream<U1_ConfigInst> &fifo_kernel_config_out
){
#pragma HLS DATAFLOW

	// FIFOs
	stream<U1_Data0PEChannelType> fifo0_feed0_0;
#pragma HLS STREAM variable=fifo0_feed0_0 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed0_1;
#pragma HLS STREAM variable=fifo0_feed0_1 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed0_2;
#pragma HLS STREAM variable=fifo0_feed0_2 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed0_3;
#pragma HLS STREAM variable=fifo0_feed0_3 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed0_4;
#pragma HLS STREAM variable=fifo0_feed0_4 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed0_5;
#pragma HLS STREAM variable=fifo0_feed0_5 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed0_6;
#pragma HLS STREAM variable=fifo0_feed0_6 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed0_7;
#pragma HLS STREAM variable=fifo0_feed0_7 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed0_8;
#pragma HLS STREAM variable=fifo0_feed0_8 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed1_0;
#pragma HLS STREAM variable=fifo0_feed1_0 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed1_1;
#pragma HLS STREAM variable=fifo0_feed1_1 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed1_2;
#pragma HLS STREAM variable=fifo0_feed1_2 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed1_3;
#pragma HLS STREAM variable=fifo0_feed1_3 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed1_4;
#pragma HLS STREAM variable=fifo0_feed1_4 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed1_5;
#pragma HLS STREAM variable=fifo0_feed1_5 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed1_6;
#pragma HLS STREAM variable=fifo0_feed1_6 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed1_7;
#pragma HLS STREAM variable=fifo0_feed1_7 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed1_8;
#pragma HLS STREAM variable=fifo0_feed1_8 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed2_0;
#pragma HLS STREAM variable=fifo0_feed2_0 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed2_1;
#pragma HLS STREAM variable=fifo0_feed2_1 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed2_2;
#pragma HLS STREAM variable=fifo0_feed2_2 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed2_3;
#pragma HLS STREAM variable=fifo0_feed2_3 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed2_4;
#pragma HLS STREAM variable=fifo0_feed2_4 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed2_5;
#pragma HLS STREAM variable=fifo0_feed2_5 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed2_6;
#pragma HLS STREAM variable=fifo0_feed2_6 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed2_7;
#pragma HLS STREAM variable=fifo0_feed2_7 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed2_8;
#pragma HLS STREAM variable=fifo0_feed2_8 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed3_0;
#pragma HLS STREAM variable=fifo0_feed3_0 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed3_1;
#pragma HLS STREAM variable=fifo0_feed3_1 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed3_2;
#pragma HLS STREAM variable=fifo0_feed3_2 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed3_3;
#pragma HLS STREAM variable=fifo0_feed3_3 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed3_4;
#pragma HLS STREAM variable=fifo0_feed3_4 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed3_5;
#pragma HLS STREAM variable=fifo0_feed3_5 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed3_6;
#pragma HLS STREAM variable=fifo0_feed3_6 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed3_7;
#pragma HLS STREAM variable=fifo0_feed3_7 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed3_8;
#pragma HLS STREAM variable=fifo0_feed3_8 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed4_0;
#pragma HLS STREAM variable=fifo0_feed4_0 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed4_1;
#pragma HLS STREAM variable=fifo0_feed4_1 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed4_2;
#pragma HLS STREAM variable=fifo0_feed4_2 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed4_3;
#pragma HLS STREAM variable=fifo0_feed4_3 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed4_4;
#pragma HLS STREAM variable=fifo0_feed4_4 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed4_5;
#pragma HLS STREAM variable=fifo0_feed4_5 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed4_6;
#pragma HLS STREAM variable=fifo0_feed4_6 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed4_7;
#pragma HLS STREAM variable=fifo0_feed4_7 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed4_8;
#pragma HLS STREAM variable=fifo0_feed4_8 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed5_0;
#pragma HLS STREAM variable=fifo0_feed5_0 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed5_1;
#pragma HLS STREAM variable=fifo0_feed5_1 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed5_2;
#pragma HLS STREAM variable=fifo0_feed5_2 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed5_3;
#pragma HLS STREAM variable=fifo0_feed5_3 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed5_4;
#pragma HLS STREAM variable=fifo0_feed5_4 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed5_5;
#pragma HLS STREAM variable=fifo0_feed5_5 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed5_6;
#pragma HLS STREAM variable=fifo0_feed5_6 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed5_7;
#pragma HLS STREAM variable=fifo0_feed5_7 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed5_8;
#pragma HLS STREAM variable=fifo0_feed5_8 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed6_0;
#pragma HLS STREAM variable=fifo0_feed6_0 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed6_1;
#pragma HLS STREAM variable=fifo0_feed6_1 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed6_2;
#pragma HLS STREAM variable=fifo0_feed6_2 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed6_3;
#pragma HLS STREAM variable=fifo0_feed6_3 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed6_4;
#pragma HLS STREAM variable=fifo0_feed6_4 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed6_5;
#pragma HLS STREAM variable=fifo0_feed6_5 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed6_6;
#pragma HLS STREAM variable=fifo0_feed6_6 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed6_7;
#pragma HLS STREAM variable=fifo0_feed6_7 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed6_8;
#pragma HLS STREAM variable=fifo0_feed6_8 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed7_0;
#pragma HLS STREAM variable=fifo0_feed7_0 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed7_1;
#pragma HLS STREAM variable=fifo0_feed7_1 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed7_2;
#pragma HLS STREAM variable=fifo0_feed7_2 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed7_3;
#pragma HLS STREAM variable=fifo0_feed7_3 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed7_4;
#pragma HLS STREAM variable=fifo0_feed7_4 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed7_5;
#pragma HLS STREAM variable=fifo0_feed7_5 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed7_6;
#pragma HLS STREAM variable=fifo0_feed7_6 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed7_7;
#pragma HLS STREAM variable=fifo0_feed7_7 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed7_8;
#pragma HLS STREAM variable=fifo0_feed7_8 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed8_0;
#pragma HLS STREAM variable=fifo0_feed8_0 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed8_1;
#pragma HLS STREAM variable=fifo0_feed8_1 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed8_2;
#pragma HLS STREAM variable=fifo0_feed8_2 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed8_3;
#pragma HLS STREAM variable=fifo0_feed8_3 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed8_4;
#pragma HLS STREAM variable=fifo0_feed8_4 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed8_5;
#pragma HLS STREAM variable=fifo0_feed8_5 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed8_6;
#pragma HLS STREAM variable=fifo0_feed8_6 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed8_7;
#pragma HLS STREAM variable=fifo0_feed8_7 depth=2
	stream<U1_Data0PEChannelType> fifo0_feed8_8;
#pragma HLS STREAM variable=fifo0_feed8_8 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed0_0;
#pragma HLS STREAM variable=fifo1_feed0_0 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed0_1;
#pragma HLS STREAM variable=fifo1_feed0_1 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed0_2;
#pragma HLS STREAM variable=fifo1_feed0_2 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed0_3;
#pragma HLS STREAM variable=fifo1_feed0_3 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed0_4;
#pragma HLS STREAM variable=fifo1_feed0_4 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed0_5;
#pragma HLS STREAM variable=fifo1_feed0_5 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed0_6;
#pragma HLS STREAM variable=fifo1_feed0_6 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed0_7;
#pragma HLS STREAM variable=fifo1_feed0_7 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed0_8;
#pragma HLS STREAM variable=fifo1_feed0_8 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed1_0;
#pragma HLS STREAM variable=fifo1_feed1_0 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed1_1;
#pragma HLS STREAM variable=fifo1_feed1_1 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed1_2;
#pragma HLS STREAM variable=fifo1_feed1_2 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed1_3;
#pragma HLS STREAM variable=fifo1_feed1_3 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed1_4;
#pragma HLS STREAM variable=fifo1_feed1_4 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed1_5;
#pragma HLS STREAM variable=fifo1_feed1_5 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed1_6;
#pragma HLS STREAM variable=fifo1_feed1_6 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed1_7;
#pragma HLS STREAM variable=fifo1_feed1_7 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed1_8;
#pragma HLS STREAM variable=fifo1_feed1_8 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed2_0;
#pragma HLS STREAM variable=fifo1_feed2_0 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed2_1;
#pragma HLS STREAM variable=fifo1_feed2_1 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed2_2;
#pragma HLS STREAM variable=fifo1_feed2_2 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed2_3;
#pragma HLS STREAM variable=fifo1_feed2_3 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed2_4;
#pragma HLS STREAM variable=fifo1_feed2_4 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed2_5;
#pragma HLS STREAM variable=fifo1_feed2_5 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed2_6;
#pragma HLS STREAM variable=fifo1_feed2_6 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed2_7;
#pragma HLS STREAM variable=fifo1_feed2_7 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed2_8;
#pragma HLS STREAM variable=fifo1_feed2_8 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed3_0;
#pragma HLS STREAM variable=fifo1_feed3_0 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed3_1;
#pragma HLS STREAM variable=fifo1_feed3_1 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed3_2;
#pragma HLS STREAM variable=fifo1_feed3_2 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed3_3;
#pragma HLS STREAM variable=fifo1_feed3_3 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed3_4;
#pragma HLS STREAM variable=fifo1_feed3_4 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed3_5;
#pragma HLS STREAM variable=fifo1_feed3_5 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed3_6;
#pragma HLS STREAM variable=fifo1_feed3_6 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed3_7;
#pragma HLS STREAM variable=fifo1_feed3_7 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed3_8;
#pragma HLS STREAM variable=fifo1_feed3_8 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed4_0;
#pragma HLS STREAM variable=fifo1_feed4_0 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed4_1;
#pragma HLS STREAM variable=fifo1_feed4_1 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed4_2;
#pragma HLS STREAM variable=fifo1_feed4_2 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed4_3;
#pragma HLS STREAM variable=fifo1_feed4_3 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed4_4;
#pragma HLS STREAM variable=fifo1_feed4_4 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed4_5;
#pragma HLS STREAM variable=fifo1_feed4_5 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed4_6;
#pragma HLS STREAM variable=fifo1_feed4_6 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed4_7;
#pragma HLS STREAM variable=fifo1_feed4_7 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed4_8;
#pragma HLS STREAM variable=fifo1_feed4_8 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed5_0;
#pragma HLS STREAM variable=fifo1_feed5_0 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed5_1;
#pragma HLS STREAM variable=fifo1_feed5_1 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed5_2;
#pragma HLS STREAM variable=fifo1_feed5_2 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed5_3;
#pragma HLS STREAM variable=fifo1_feed5_3 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed5_4;
#pragma HLS STREAM variable=fifo1_feed5_4 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed5_5;
#pragma HLS STREAM variable=fifo1_feed5_5 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed5_6;
#pragma HLS STREAM variable=fifo1_feed5_6 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed5_7;
#pragma HLS STREAM variable=fifo1_feed5_7 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed5_8;
#pragma HLS STREAM variable=fifo1_feed5_8 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed6_0;
#pragma HLS STREAM variable=fifo1_feed6_0 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed6_1;
#pragma HLS STREAM variable=fifo1_feed6_1 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed6_2;
#pragma HLS STREAM variable=fifo1_feed6_2 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed6_3;
#pragma HLS STREAM variable=fifo1_feed6_3 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed6_4;
#pragma HLS STREAM variable=fifo1_feed6_4 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed6_5;
#pragma HLS STREAM variable=fifo1_feed6_5 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed6_6;
#pragma HLS STREAM variable=fifo1_feed6_6 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed6_7;
#pragma HLS STREAM variable=fifo1_feed6_7 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed6_8;
#pragma HLS STREAM variable=fifo1_feed6_8 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed7_0;
#pragma HLS STREAM variable=fifo1_feed7_0 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed7_1;
#pragma HLS STREAM variable=fifo1_feed7_1 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed7_2;
#pragma HLS STREAM variable=fifo1_feed7_2 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed7_3;
#pragma HLS STREAM variable=fifo1_feed7_3 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed7_4;
#pragma HLS STREAM variable=fifo1_feed7_4 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed7_5;
#pragma HLS STREAM variable=fifo1_feed7_5 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed7_6;
#pragma HLS STREAM variable=fifo1_feed7_6 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed7_7;
#pragma HLS STREAM variable=fifo1_feed7_7 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed7_8;
#pragma HLS STREAM variable=fifo1_feed7_8 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed8_0;
#pragma HLS STREAM variable=fifo1_feed8_0 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed8_1;
#pragma HLS STREAM variable=fifo1_feed8_1 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed8_2;
#pragma HLS STREAM variable=fifo1_feed8_2 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed8_3;
#pragma HLS STREAM variable=fifo1_feed8_3 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed8_4;
#pragma HLS STREAM variable=fifo1_feed8_4 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed8_5;
#pragma HLS STREAM variable=fifo1_feed8_5 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed8_6;
#pragma HLS STREAM variable=fifo1_feed8_6 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed8_7;
#pragma HLS STREAM variable=fifo1_feed8_7 depth=2
	stream<U1_Data1PEChannelType> fifo1_feed8_8;
#pragma HLS STREAM variable=fifo1_feed8_8 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect0_0;
#pragma HLS STREAM variable=fifo2_collect0_0 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect0_1;
#pragma HLS STREAM variable=fifo2_collect0_1 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect0_2;
#pragma HLS STREAM variable=fifo2_collect0_2 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect0_3;
#pragma HLS STREAM variable=fifo2_collect0_3 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect0_4;
#pragma HLS STREAM variable=fifo2_collect0_4 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect0_5;
#pragma HLS STREAM variable=fifo2_collect0_5 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect0_6;
#pragma HLS STREAM variable=fifo2_collect0_6 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect0_7;
#pragma HLS STREAM variable=fifo2_collect0_7 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect0_8;
#pragma HLS STREAM variable=fifo2_collect0_8 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect1_0;
#pragma HLS STREAM variable=fifo2_collect1_0 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect1_1;
#pragma HLS STREAM variable=fifo2_collect1_1 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect1_2;
#pragma HLS STREAM variable=fifo2_collect1_2 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect1_3;
#pragma HLS STREAM variable=fifo2_collect1_3 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect1_4;
#pragma HLS STREAM variable=fifo2_collect1_4 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect1_5;
#pragma HLS STREAM variable=fifo2_collect1_5 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect1_6;
#pragma HLS STREAM variable=fifo2_collect1_6 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect1_7;
#pragma HLS STREAM variable=fifo2_collect1_7 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect1_8;
#pragma HLS STREAM variable=fifo2_collect1_8 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect2_0;
#pragma HLS STREAM variable=fifo2_collect2_0 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect2_1;
#pragma HLS STREAM variable=fifo2_collect2_1 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect2_2;
#pragma HLS STREAM variable=fifo2_collect2_2 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect2_3;
#pragma HLS STREAM variable=fifo2_collect2_3 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect2_4;
#pragma HLS STREAM variable=fifo2_collect2_4 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect2_5;
#pragma HLS STREAM variable=fifo2_collect2_5 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect2_6;
#pragma HLS STREAM variable=fifo2_collect2_6 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect2_7;
#pragma HLS STREAM variable=fifo2_collect2_7 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect2_8;
#pragma HLS STREAM variable=fifo2_collect2_8 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect3_0;
#pragma HLS STREAM variable=fifo2_collect3_0 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect3_1;
#pragma HLS STREAM variable=fifo2_collect3_1 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect3_2;
#pragma HLS STREAM variable=fifo2_collect3_2 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect3_3;
#pragma HLS STREAM variable=fifo2_collect3_3 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect3_4;
#pragma HLS STREAM variable=fifo2_collect3_4 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect3_5;
#pragma HLS STREAM variable=fifo2_collect3_5 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect3_6;
#pragma HLS STREAM variable=fifo2_collect3_6 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect3_7;
#pragma HLS STREAM variable=fifo2_collect3_7 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect3_8;
#pragma HLS STREAM variable=fifo2_collect3_8 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect4_0;
#pragma HLS STREAM variable=fifo2_collect4_0 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect4_1;
#pragma HLS STREAM variable=fifo2_collect4_1 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect4_2;
#pragma HLS STREAM variable=fifo2_collect4_2 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect4_3;
#pragma HLS STREAM variable=fifo2_collect4_3 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect4_4;
#pragma HLS STREAM variable=fifo2_collect4_4 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect4_5;
#pragma HLS STREAM variable=fifo2_collect4_5 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect4_6;
#pragma HLS STREAM variable=fifo2_collect4_6 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect4_7;
#pragma HLS STREAM variable=fifo2_collect4_7 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect4_8;
#pragma HLS STREAM variable=fifo2_collect4_8 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect5_0;
#pragma HLS STREAM variable=fifo2_collect5_0 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect5_1;
#pragma HLS STREAM variable=fifo2_collect5_1 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect5_2;
#pragma HLS STREAM variable=fifo2_collect5_2 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect5_3;
#pragma HLS STREAM variable=fifo2_collect5_3 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect5_4;
#pragma HLS STREAM variable=fifo2_collect5_4 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect5_5;
#pragma HLS STREAM variable=fifo2_collect5_5 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect5_6;
#pragma HLS STREAM variable=fifo2_collect5_6 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect5_7;
#pragma HLS STREAM variable=fifo2_collect5_7 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect5_8;
#pragma HLS STREAM variable=fifo2_collect5_8 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect6_0;
#pragma HLS STREAM variable=fifo2_collect6_0 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect6_1;
#pragma HLS STREAM variable=fifo2_collect6_1 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect6_2;
#pragma HLS STREAM variable=fifo2_collect6_2 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect6_3;
#pragma HLS STREAM variable=fifo2_collect6_3 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect6_4;
#pragma HLS STREAM variable=fifo2_collect6_4 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect6_5;
#pragma HLS STREAM variable=fifo2_collect6_5 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect6_6;
#pragma HLS STREAM variable=fifo2_collect6_6 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect6_7;
#pragma HLS STREAM variable=fifo2_collect6_7 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect6_8;
#pragma HLS STREAM variable=fifo2_collect6_8 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect7_0;
#pragma HLS STREAM variable=fifo2_collect7_0 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect7_1;
#pragma HLS STREAM variable=fifo2_collect7_1 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect7_2;
#pragma HLS STREAM variable=fifo2_collect7_2 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect7_3;
#pragma HLS STREAM variable=fifo2_collect7_3 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect7_4;
#pragma HLS STREAM variable=fifo2_collect7_4 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect7_5;
#pragma HLS STREAM variable=fifo2_collect7_5 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect7_6;
#pragma HLS STREAM variable=fifo2_collect7_6 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect7_7;
#pragma HLS STREAM variable=fifo2_collect7_7 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect7_8;
#pragma HLS STREAM variable=fifo2_collect7_8 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect8_0;
#pragma HLS STREAM variable=fifo2_collect8_0 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect8_1;
#pragma HLS STREAM variable=fifo2_collect8_1 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect8_2;
#pragma HLS STREAM variable=fifo2_collect8_2 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect8_3;
#pragma HLS STREAM variable=fifo2_collect8_3 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect8_4;
#pragma HLS STREAM variable=fifo2_collect8_4 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect8_5;
#pragma HLS STREAM variable=fifo2_collect8_5 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect8_6;
#pragma HLS STREAM variable=fifo2_collect8_6 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect8_7;
#pragma HLS STREAM variable=fifo2_collect8_7 depth=2
	stream<U1_Data2PEChannelType> fifo2_collect8_8;
#pragma HLS STREAM variable=fifo2_collect8_8 depth=2
	stream<U1_Data0TransferChannelType> fifo0_transfer0;
#pragma HLS STREAM variable=fifo0_transfer0 depth=2
	stream<U1_Data0TransferChannelType> fifo0_transfer1;
#pragma HLS STREAM variable=fifo0_transfer1 depth=2
	stream<U1_Data0TransferChannelType> fifo0_transfer2;
#pragma HLS STREAM variable=fifo0_transfer2 depth=2
	stream<U1_Data0TransferChannelType> fifo0_transfer3;
#pragma HLS STREAM variable=fifo0_transfer3 depth=2
	stream<U1_Data0TransferChannelType> fifo0_transfer4;
#pragma HLS STREAM variable=fifo0_transfer4 depth=2
	stream<U1_Data0TransferChannelType> fifo0_transfer5;
#pragma HLS STREAM variable=fifo0_transfer5 depth=2
	stream<U1_Data0TransferChannelType> fifo0_transfer6;
#pragma HLS STREAM variable=fifo0_transfer6 depth=2
	stream<U1_Data0TransferChannelType> fifo0_transfer7;
#pragma HLS STREAM variable=fifo0_transfer7 depth=2
	stream<U1_Data0TransferChannelType> fifo0_transfer8;
#pragma HLS STREAM variable=fifo0_transfer8 depth=2
	stream<U1_Data1TransferChannelType> fifo1_transfer0;
#pragma HLS STREAM variable=fifo1_transfer0 depth=2
	stream<U1_Data1TransferChannelType> fifo1_transfer1;
#pragma HLS STREAM variable=fifo1_transfer1 depth=2
	stream<U1_Data1TransferChannelType> fifo1_transfer2;
#pragma HLS STREAM variable=fifo1_transfer2 depth=2
	stream<U1_Data1TransferChannelType> fifo1_transfer3;
#pragma HLS STREAM variable=fifo1_transfer3 depth=2
	stream<U1_Data1TransferChannelType> fifo1_transfer4;
#pragma HLS STREAM variable=fifo1_transfer4 depth=2
	stream<U1_Data1TransferChannelType> fifo1_transfer5;
#pragma HLS STREAM variable=fifo1_transfer5 depth=2
	stream<U1_Data1TransferChannelType> fifo1_transfer6;
#pragma HLS STREAM variable=fifo1_transfer6 depth=2
	stream<U1_Data1TransferChannelType> fifo1_transfer7;
#pragma HLS STREAM variable=fifo1_transfer7 depth=2
	stream<U1_Data1TransferChannelType> fifo1_transfer8;
#pragma HLS STREAM variable=fifo1_transfer8 depth=2
	stream<U1_Data2TransferChannelType> fifo2_transfer0;
#pragma HLS STREAM variable=fifo2_transfer0 depth=2
	stream<U1_Data2TransferChannelType> fifo2_transfer1;
#pragma HLS STREAM variable=fifo2_transfer1 depth=2
	stream<U1_Data2TransferChannelType> fifo2_transfer2;
#pragma HLS STREAM variable=fifo2_transfer2 depth=2
	stream<U1_Data2TransferChannelType> fifo2_transfer3;
#pragma HLS STREAM variable=fifo2_transfer3 depth=2
	stream<U1_Data2TransferChannelType> fifo2_transfer4;
#pragma HLS STREAM variable=fifo2_transfer4 depth=2
	stream<U1_Data2TransferChannelType> fifo2_transfer5;
#pragma HLS STREAM variable=fifo2_transfer5 depth=2
	stream<U1_Data2TransferChannelType> fifo2_transfer6;
#pragma HLS STREAM variable=fifo2_transfer6 depth=2
	stream<U1_Data2TransferChannelType> fifo2_transfer7;
#pragma HLS STREAM variable=fifo2_transfer7 depth=2
	stream<U1_Data2TransferChannelType> fifo2_transfer8;
#pragma HLS STREAM variable=fifo2_transfer8 depth=2
	stream<ap_uint<U1_DATA0_WIDTH * U1_DATA0_FC_SIMD_FACTOR> > fifo0_shim;
#pragma HLS STREAM variable=fifo0_shim depth=2
	stream<ap_uint<U1_DATA1_WIDTH * U1_DATA1_FC_SIMD_FACTOR> > fifo1_shim;
#pragma HLS STREAM variable=fifo1_shim depth=2
	stream<ap_uint<U1_DATA2_WIDTH * U1_DATA2_FC_SIMD_FACTOR> > fifo2_shim;
#pragma HLS STREAM variable=fifo2_shim depth=2

	stream<uint> fifo_DataFeed0Head_config_out0;
#pragma HLS STREAM variable=fifo_DataFeed0Head_config_out0 depth=16
	stream<uint> fifo_DataFeed0Head_config_out1;
#pragma HLS STREAM variable=fifo_DataFeed0Head_config_out1 depth=16
	stream<uint> fifo_DataFeed1Head_config_out0;
#pragma HLS STREAM variable=fifo_DataFeed1Head_config_out0 depth=16

	stream<uint> fifo_DataFeed0Engine0_config_out0;
	stream<uint> fifo_DataFeed0Engine0_config_out1;
#pragma HLS STREAM variable=fifo_DataFeed0Engine0_config_out0 depth=16
#pragma HLS STREAM variable=fifo_DataFeed0Engine0_config_out1 depth=16
	stream<uint> fifo_DataFeed0Engine1_config_out0;
	stream<uint> fifo_DataFeed0Engine1_config_out1;
#pragma HLS STREAM variable=fifo_DataFeed0Engine1_config_out0 depth=16
#pragma HLS STREAM variable=fifo_DataFeed0Engine1_config_out1 depth=16
	stream<uint> fifo_DataFeed0Engine2_config_out0;
	stream<uint> fifo_DataFeed0Engine2_config_out1;
#pragma HLS STREAM variable=fifo_DataFeed0Engine2_config_out0 depth=16
#pragma HLS STREAM variable=fifo_DataFeed0Engine2_config_out1 depth=16
	stream<uint> fifo_DataFeed0Engine3_config_out0;
	stream<uint> fifo_DataFeed0Engine3_config_out1;
#pragma HLS STREAM variable=fifo_DataFeed0Engine3_config_out0 depth=16
#pragma HLS STREAM variable=fifo_DataFeed0Engine3_config_out1 depth=16
	stream<uint> fifo_DataFeed0Engine4_config_out0;
	stream<uint> fifo_DataFeed0Engine4_config_out1;
#pragma HLS STREAM variable=fifo_DataFeed0Engine4_config_out0 depth=16
#pragma HLS STREAM variable=fifo_DataFeed0Engine4_config_out1 depth=16
	stream<uint> fifo_DataFeed0Engine5_config_out0;
	stream<uint> fifo_DataFeed0Engine5_config_out1;
#pragma HLS STREAM variable=fifo_DataFeed0Engine5_config_out0 depth=16
#pragma HLS STREAM variable=fifo_DataFeed0Engine5_config_out1 depth=16
	stream<uint> fifo_DataFeed0Engine6_config_out0;
	stream<uint> fifo_DataFeed0Engine6_config_out1;
#pragma HLS STREAM variable=fifo_DataFeed0Engine6_config_out0 depth=16
#pragma HLS STREAM variable=fifo_DataFeed0Engine6_config_out1 depth=16
	stream<uint> fifo_DataFeed0Engine7_config_out1;
#pragma HLS STREAM variable=fifo_DataFeed0Engine7_config_out1 depth=16

	stream<uint> fifo_DataFeed1Engine0_config_out0;
#pragma HLS STREAM variable=fifo_DataFeed1Engine0_config_out0 depth=16
	stream<uint> fifo_DataFeed1Engine1_config_out0;
#pragma HLS STREAM variable=fifo_DataFeed1Engine1_config_out0 depth=16
	stream<uint> fifo_DataFeed1Engine2_config_out0;
#pragma HLS STREAM variable=fifo_DataFeed1Engine2_config_out0 depth=16
	stream<uint> fifo_DataFeed1Engine3_config_out0;
#pragma HLS STREAM variable=fifo_DataFeed1Engine3_config_out0 depth=16
	stream<uint> fifo_DataFeed1Engine4_config_out0;
#pragma HLS STREAM variable=fifo_DataFeed1Engine4_config_out0 depth=16
	stream<uint> fifo_DataFeed1Engine5_config_out0;
#pragma HLS STREAM variable=fifo_DataFeed1Engine5_config_out0 depth=16
	stream<uint> fifo_DataFeed1Engine6_config_out0;
#pragma HLS STREAM variable=fifo_DataFeed1Engine6_config_out0 depth=16

	stream<uint> fifo_DataCollect2Engine0_config_out;
#pragma HLS STREAM variable=fifo_DataCollect2Engine0_config_out depth=16
	stream<uint> fifo_DataCollect2Engine1_config_out;
#pragma HLS STREAM variable=fifo_DataCollect2Engine1_config_out depth=16
	stream<uint> fifo_DataCollect2Engine2_config_out;
#pragma HLS STREAM variable=fifo_DataCollect2Engine2_config_out depth=16
	stream<uint> fifo_DataCollect2Engine3_config_out;
#pragma HLS STREAM variable=fifo_DataCollect2Engine3_config_out depth=16
	stream<uint> fifo_DataCollect2Engine4_config_out;
#pragma HLS STREAM variable=fifo_DataCollect2Engine4_config_out depth=16
	stream<uint> fifo_DataCollect2Engine5_config_out;
#pragma HLS STREAM variable=fifo_DataCollect2Engine5_config_out depth=16
	stream<uint> fifo_DataCollect2Engine6_config_out;
#pragma HLS STREAM variable=fifo_DataCollect2Engine6_config_out depth=16
	stream<uint> fifo_DataCollect2Engine7_config_out;
#pragma HLS STREAM variable=fifo_DataCollect2Engine7_config_out depth=16

	stream<uint> fifo_PE0_0_op0_config_out;
	stream<uint> fifo_PE0_0_op1_config_out;
	stream<uint> fifo_PE0_0_compute_config_out;
	stream<uint> fifo_PE0_0_res_config_out;
#pragma HLS STREAM variable=fifo_PE0_0_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_0_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_0_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_0_res_config_out depth=2
	stream<uint> fifo_PE0_1_op0_config_out;
	stream<uint> fifo_PE0_1_op1_config_out;
	stream<uint> fifo_PE0_1_compute_config_out;
	stream<uint> fifo_PE0_1_res_config_out;
#pragma HLS STREAM variable=fifo_PE0_1_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_1_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_1_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_1_res_config_out depth=2
	stream<uint> fifo_PE0_2_op0_config_out;
	stream<uint> fifo_PE0_2_op1_config_out;
	stream<uint> fifo_PE0_2_compute_config_out;
	stream<uint> fifo_PE0_2_res_config_out;
#pragma HLS STREAM variable=fifo_PE0_2_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_2_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_2_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_2_res_config_out depth=2
	stream<uint> fifo_PE0_3_op0_config_out;
	stream<uint> fifo_PE0_3_op1_config_out;
	stream<uint> fifo_PE0_3_compute_config_out;
	stream<uint> fifo_PE0_3_res_config_out;
#pragma HLS STREAM variable=fifo_PE0_3_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_3_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_3_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_3_res_config_out depth=2
	stream<uint> fifo_PE0_4_op0_config_out;
	stream<uint> fifo_PE0_4_op1_config_out;
	stream<uint> fifo_PE0_4_compute_config_out;
	stream<uint> fifo_PE0_4_res_config_out;
#pragma HLS STREAM variable=fifo_PE0_4_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_4_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_4_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_4_res_config_out depth=2
	stream<uint> fifo_PE0_5_op0_config_out;
	stream<uint> fifo_PE0_5_op1_config_out;
	stream<uint> fifo_PE0_5_compute_config_out;
	stream<uint> fifo_PE0_5_res_config_out;
#pragma HLS STREAM variable=fifo_PE0_5_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_5_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_5_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_5_res_config_out depth=2
	stream<uint> fifo_PE0_6_op0_config_out;
	stream<uint> fifo_PE0_6_op1_config_out;
	stream<uint> fifo_PE0_6_compute_config_out;
	stream<uint> fifo_PE0_6_res_config_out;
#pragma HLS STREAM variable=fifo_PE0_6_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_6_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_6_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_6_res_config_out depth=2
	stream<uint> fifo_PE0_7_op0_config_out;
	stream<uint> fifo_PE0_7_op1_config_out;
	stream<uint> fifo_PE0_7_compute_config_out;
	stream<uint> fifo_PE0_7_res_config_out;
#pragma HLS STREAM variable=fifo_PE0_7_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_7_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_7_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE0_7_res_config_out depth=2
	stream<uint> fifo_PE1_0_op0_config_out;
	stream<uint> fifo_PE1_0_op1_config_out;
	stream<uint> fifo_PE1_0_compute_config_out;
	stream<uint> fifo_PE1_0_res_config_out;
#pragma HLS STREAM variable=fifo_PE1_0_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_0_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_0_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_0_res_config_out depth=2
	stream<uint> fifo_PE1_1_op0_config_out;
	stream<uint> fifo_PE1_1_op1_config_out;
	stream<uint> fifo_PE1_1_compute_config_out;
	stream<uint> fifo_PE1_1_res_config_out;
#pragma HLS STREAM variable=fifo_PE1_1_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_1_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_1_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_1_res_config_out depth=2
	stream<uint> fifo_PE1_2_op0_config_out;
	stream<uint> fifo_PE1_2_op1_config_out;
	stream<uint> fifo_PE1_2_compute_config_out;
	stream<uint> fifo_PE1_2_res_config_out;
#pragma HLS STREAM variable=fifo_PE1_2_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_2_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_2_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_2_res_config_out depth=2
	stream<uint> fifo_PE1_3_op0_config_out;
	stream<uint> fifo_PE1_3_op1_config_out;
	stream<uint> fifo_PE1_3_compute_config_out;
	stream<uint> fifo_PE1_3_res_config_out;
#pragma HLS STREAM variable=fifo_PE1_3_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_3_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_3_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_3_res_config_out depth=2
	stream<uint> fifo_PE1_4_op0_config_out;
	stream<uint> fifo_PE1_4_op1_config_out;
	stream<uint> fifo_PE1_4_compute_config_out;
	stream<uint> fifo_PE1_4_res_config_out;
#pragma HLS STREAM variable=fifo_PE1_4_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_4_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_4_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_4_res_config_out depth=2
	stream<uint> fifo_PE1_5_op0_config_out;
	stream<uint> fifo_PE1_5_op1_config_out;
	stream<uint> fifo_PE1_5_compute_config_out;
	stream<uint> fifo_PE1_5_res_config_out;
#pragma HLS STREAM variable=fifo_PE1_5_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_5_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_5_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_5_res_config_out depth=2
	stream<uint> fifo_PE1_6_op0_config_out;
	stream<uint> fifo_PE1_6_op1_config_out;
	stream<uint> fifo_PE1_6_compute_config_out;
	stream<uint> fifo_PE1_6_res_config_out;
#pragma HLS STREAM variable=fifo_PE1_6_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_6_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_6_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_6_res_config_out depth=2
	stream<uint> fifo_PE1_7_op0_config_out;
	stream<uint> fifo_PE1_7_op1_config_out;
	stream<uint> fifo_PE1_7_compute_config_out;
	stream<uint> fifo_PE1_7_res_config_out;
#pragma HLS STREAM variable=fifo_PE1_7_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_7_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_7_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE1_7_res_config_out depth=2
	stream<uint> fifo_PE2_0_op0_config_out;
	stream<uint> fifo_PE2_0_op1_config_out;
	stream<uint> fifo_PE2_0_compute_config_out;
	stream<uint> fifo_PE2_0_res_config_out;
#pragma HLS STREAM variable=fifo_PE2_0_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_0_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_0_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_0_res_config_out depth=2
	stream<uint> fifo_PE2_1_op0_config_out;
	stream<uint> fifo_PE2_1_op1_config_out;
	stream<uint> fifo_PE2_1_compute_config_out;
	stream<uint> fifo_PE2_1_res_config_out;
#pragma HLS STREAM variable=fifo_PE2_1_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_1_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_1_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_1_res_config_out depth=2
	stream<uint> fifo_PE2_2_op0_config_out;
	stream<uint> fifo_PE2_2_op1_config_out;
	stream<uint> fifo_PE2_2_compute_config_out;
	stream<uint> fifo_PE2_2_res_config_out;
#pragma HLS STREAM variable=fifo_PE2_2_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_2_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_2_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_2_res_config_out depth=2
	stream<uint> fifo_PE2_3_op0_config_out;
	stream<uint> fifo_PE2_3_op1_config_out;
	stream<uint> fifo_PE2_3_compute_config_out;
	stream<uint> fifo_PE2_3_res_config_out;
#pragma HLS STREAM variable=fifo_PE2_3_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_3_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_3_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_3_res_config_out depth=2
	stream<uint> fifo_PE2_4_op0_config_out;
	stream<uint> fifo_PE2_4_op1_config_out;
	stream<uint> fifo_PE2_4_compute_config_out;
	stream<uint> fifo_PE2_4_res_config_out;
#pragma HLS STREAM variable=fifo_PE2_4_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_4_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_4_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_4_res_config_out depth=2
	stream<uint> fifo_PE2_5_op0_config_out;
	stream<uint> fifo_PE2_5_op1_config_out;
	stream<uint> fifo_PE2_5_compute_config_out;
	stream<uint> fifo_PE2_5_res_config_out;
#pragma HLS STREAM variable=fifo_PE2_5_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_5_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_5_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_5_res_config_out depth=2
	stream<uint> fifo_PE2_6_op0_config_out;
	stream<uint> fifo_PE2_6_op1_config_out;
	stream<uint> fifo_PE2_6_compute_config_out;
	stream<uint> fifo_PE2_6_res_config_out;
#pragma HLS STREAM variable=fifo_PE2_6_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_6_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_6_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_6_res_config_out depth=2
	stream<uint> fifo_PE2_7_op0_config_out;
	stream<uint> fifo_PE2_7_op1_config_out;
	stream<uint> fifo_PE2_7_compute_config_out;
	stream<uint> fifo_PE2_7_res_config_out;
#pragma HLS STREAM variable=fifo_PE2_7_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_7_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_7_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE2_7_res_config_out depth=2
	stream<uint> fifo_PE3_0_op0_config_out;
	stream<uint> fifo_PE3_0_op1_config_out;
	stream<uint> fifo_PE3_0_compute_config_out;
	stream<uint> fifo_PE3_0_res_config_out;
#pragma HLS STREAM variable=fifo_PE3_0_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_0_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_0_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_0_res_config_out depth=2
	stream<uint> fifo_PE3_1_op0_config_out;
	stream<uint> fifo_PE3_1_op1_config_out;
	stream<uint> fifo_PE3_1_compute_config_out;
	stream<uint> fifo_PE3_1_res_config_out;
#pragma HLS STREAM variable=fifo_PE3_1_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_1_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_1_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_1_res_config_out depth=2
	stream<uint> fifo_PE3_2_op0_config_out;
	stream<uint> fifo_PE3_2_op1_config_out;
	stream<uint> fifo_PE3_2_compute_config_out;
	stream<uint> fifo_PE3_2_res_config_out;
#pragma HLS STREAM variable=fifo_PE3_2_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_2_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_2_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_2_res_config_out depth=2
	stream<uint> fifo_PE3_3_op0_config_out;
	stream<uint> fifo_PE3_3_op1_config_out;
	stream<uint> fifo_PE3_3_compute_config_out;
	stream<uint> fifo_PE3_3_res_config_out;
#pragma HLS STREAM variable=fifo_PE3_3_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_3_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_3_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_3_res_config_out depth=2
	stream<uint> fifo_PE3_4_op0_config_out;
	stream<uint> fifo_PE3_4_op1_config_out;
	stream<uint> fifo_PE3_4_compute_config_out;
	stream<uint> fifo_PE3_4_res_config_out;
#pragma HLS STREAM variable=fifo_PE3_4_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_4_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_4_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_4_res_config_out depth=2
	stream<uint> fifo_PE3_5_op0_config_out;
	stream<uint> fifo_PE3_5_op1_config_out;
	stream<uint> fifo_PE3_5_compute_config_out;
	stream<uint> fifo_PE3_5_res_config_out;
#pragma HLS STREAM variable=fifo_PE3_5_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_5_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_5_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_5_res_config_out depth=2
	stream<uint> fifo_PE3_6_op0_config_out;
	stream<uint> fifo_PE3_6_op1_config_out;
	stream<uint> fifo_PE3_6_compute_config_out;
	stream<uint> fifo_PE3_6_res_config_out;
#pragma HLS STREAM variable=fifo_PE3_6_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_6_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_6_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_6_res_config_out depth=2
	stream<uint> fifo_PE3_7_op0_config_out;
	stream<uint> fifo_PE3_7_op1_config_out;
	stream<uint> fifo_PE3_7_compute_config_out;
	stream<uint> fifo_PE3_7_res_config_out;
#pragma HLS STREAM variable=fifo_PE3_7_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_7_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_7_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE3_7_res_config_out depth=2
	stream<uint> fifo_PE4_0_op0_config_out;
	stream<uint> fifo_PE4_0_op1_config_out;
	stream<uint> fifo_PE4_0_compute_config_out;
	stream<uint> fifo_PE4_0_res_config_out;
#pragma HLS STREAM variable=fifo_PE4_0_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_0_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_0_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_0_res_config_out depth=2
	stream<uint> fifo_PE4_1_op0_config_out;
	stream<uint> fifo_PE4_1_op1_config_out;
	stream<uint> fifo_PE4_1_compute_config_out;
	stream<uint> fifo_PE4_1_res_config_out;
#pragma HLS STREAM variable=fifo_PE4_1_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_1_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_1_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_1_res_config_out depth=2
	stream<uint> fifo_PE4_2_op0_config_out;
	stream<uint> fifo_PE4_2_op1_config_out;
	stream<uint> fifo_PE4_2_compute_config_out;
	stream<uint> fifo_PE4_2_res_config_out;
#pragma HLS STREAM variable=fifo_PE4_2_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_2_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_2_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_2_res_config_out depth=2
	stream<uint> fifo_PE4_3_op0_config_out;
	stream<uint> fifo_PE4_3_op1_config_out;
	stream<uint> fifo_PE4_3_compute_config_out;
	stream<uint> fifo_PE4_3_res_config_out;
#pragma HLS STREAM variable=fifo_PE4_3_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_3_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_3_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_3_res_config_out depth=2
	stream<uint> fifo_PE4_4_op0_config_out;
	stream<uint> fifo_PE4_4_op1_config_out;
	stream<uint> fifo_PE4_4_compute_config_out;
	stream<uint> fifo_PE4_4_res_config_out;
#pragma HLS STREAM variable=fifo_PE4_4_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_4_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_4_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_4_res_config_out depth=2
	stream<uint> fifo_PE4_5_op0_config_out;
	stream<uint> fifo_PE4_5_op1_config_out;
	stream<uint> fifo_PE4_5_compute_config_out;
	stream<uint> fifo_PE4_5_res_config_out;
#pragma HLS STREAM variable=fifo_PE4_5_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_5_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_5_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_5_res_config_out depth=2
	stream<uint> fifo_PE4_6_op0_config_out;
	stream<uint> fifo_PE4_6_op1_config_out;
	stream<uint> fifo_PE4_6_compute_config_out;
	stream<uint> fifo_PE4_6_res_config_out;
#pragma HLS STREAM variable=fifo_PE4_6_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_6_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_6_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_6_res_config_out depth=2
	stream<uint> fifo_PE4_7_op0_config_out;
	stream<uint> fifo_PE4_7_op1_config_out;
	stream<uint> fifo_PE4_7_compute_config_out;
	stream<uint> fifo_PE4_7_res_config_out;
#pragma HLS STREAM variable=fifo_PE4_7_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_7_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_7_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE4_7_res_config_out depth=2
	stream<uint> fifo_PE5_0_op0_config_out;
	stream<uint> fifo_PE5_0_op1_config_out;
	stream<uint> fifo_PE5_0_compute_config_out;
	stream<uint> fifo_PE5_0_res_config_out;
#pragma HLS STREAM variable=fifo_PE5_0_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_0_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_0_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_0_res_config_out depth=2
	stream<uint> fifo_PE5_1_op0_config_out;
	stream<uint> fifo_PE5_1_op1_config_out;
	stream<uint> fifo_PE5_1_compute_config_out;
	stream<uint> fifo_PE5_1_res_config_out;
#pragma HLS STREAM variable=fifo_PE5_1_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_1_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_1_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_1_res_config_out depth=2
	stream<uint> fifo_PE5_2_op0_config_out;
	stream<uint> fifo_PE5_2_op1_config_out;
	stream<uint> fifo_PE5_2_compute_config_out;
	stream<uint> fifo_PE5_2_res_config_out;
#pragma HLS STREAM variable=fifo_PE5_2_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_2_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_2_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_2_res_config_out depth=2
	stream<uint> fifo_PE5_3_op0_config_out;
	stream<uint> fifo_PE5_3_op1_config_out;
	stream<uint> fifo_PE5_3_compute_config_out;
	stream<uint> fifo_PE5_3_res_config_out;
#pragma HLS STREAM variable=fifo_PE5_3_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_3_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_3_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_3_res_config_out depth=2
	stream<uint> fifo_PE5_4_op0_config_out;
	stream<uint> fifo_PE5_4_op1_config_out;
	stream<uint> fifo_PE5_4_compute_config_out;
	stream<uint> fifo_PE5_4_res_config_out;
#pragma HLS STREAM variable=fifo_PE5_4_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_4_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_4_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_4_res_config_out depth=2
	stream<uint> fifo_PE5_5_op0_config_out;
	stream<uint> fifo_PE5_5_op1_config_out;
	stream<uint> fifo_PE5_5_compute_config_out;
	stream<uint> fifo_PE5_5_res_config_out;
#pragma HLS STREAM variable=fifo_PE5_5_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_5_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_5_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_5_res_config_out depth=2
	stream<uint> fifo_PE5_6_op0_config_out;
	stream<uint> fifo_PE5_6_op1_config_out;
	stream<uint> fifo_PE5_6_compute_config_out;
	stream<uint> fifo_PE5_6_res_config_out;
#pragma HLS STREAM variable=fifo_PE5_6_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_6_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_6_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_6_res_config_out depth=2
	stream<uint> fifo_PE5_7_op0_config_out;
	stream<uint> fifo_PE5_7_op1_config_out;
	stream<uint> fifo_PE5_7_compute_config_out;
	stream<uint> fifo_PE5_7_res_config_out;
#pragma HLS STREAM variable=fifo_PE5_7_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_7_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_7_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE5_7_res_config_out depth=2
	stream<uint> fifo_PE6_0_op0_config_out;
	stream<uint> fifo_PE6_0_op1_config_out;
	stream<uint> fifo_PE6_0_compute_config_out;
	stream<uint> fifo_PE6_0_res_config_out;
#pragma HLS STREAM variable=fifo_PE6_0_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_0_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_0_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_0_res_config_out depth=2
	stream<uint> fifo_PE6_1_op0_config_out;
	stream<uint> fifo_PE6_1_op1_config_out;
	stream<uint> fifo_PE6_1_compute_config_out;
	stream<uint> fifo_PE6_1_res_config_out;
#pragma HLS STREAM variable=fifo_PE6_1_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_1_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_1_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_1_res_config_out depth=2
	stream<uint> fifo_PE6_2_op0_config_out;
	stream<uint> fifo_PE6_2_op1_config_out;
	stream<uint> fifo_PE6_2_compute_config_out;
	stream<uint> fifo_PE6_2_res_config_out;
#pragma HLS STREAM variable=fifo_PE6_2_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_2_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_2_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_2_res_config_out depth=2
	stream<uint> fifo_PE6_3_op0_config_out;
	stream<uint> fifo_PE6_3_op1_config_out;
	stream<uint> fifo_PE6_3_compute_config_out;
	stream<uint> fifo_PE6_3_res_config_out;
#pragma HLS STREAM variable=fifo_PE6_3_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_3_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_3_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_3_res_config_out depth=2
	stream<uint> fifo_PE6_4_op0_config_out;
	stream<uint> fifo_PE6_4_op1_config_out;
	stream<uint> fifo_PE6_4_compute_config_out;
	stream<uint> fifo_PE6_4_res_config_out;
#pragma HLS STREAM variable=fifo_PE6_4_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_4_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_4_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_4_res_config_out depth=2
	stream<uint> fifo_PE6_5_op0_config_out;
	stream<uint> fifo_PE6_5_op1_config_out;
	stream<uint> fifo_PE6_5_compute_config_out;
	stream<uint> fifo_PE6_5_res_config_out;
#pragma HLS STREAM variable=fifo_PE6_5_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_5_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_5_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_5_res_config_out depth=2
	stream<uint> fifo_PE6_6_op0_config_out;
	stream<uint> fifo_PE6_6_op1_config_out;
	stream<uint> fifo_PE6_6_compute_config_out;
	stream<uint> fifo_PE6_6_res_config_out;
#pragma HLS STREAM variable=fifo_PE6_6_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_6_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_6_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_6_res_config_out depth=2
	stream<uint> fifo_PE6_7_op0_config_out;
	stream<uint> fifo_PE6_7_op1_config_out;
	stream<uint> fifo_PE6_7_compute_config_out;
	stream<uint> fifo_PE6_7_res_config_out;
#pragma HLS STREAM variable=fifo_PE6_7_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_7_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_7_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE6_7_res_config_out depth=2
	stream<uint> fifo_PE7_0_op0_config_out;
	stream<uint> fifo_PE7_0_op1_config_out;
	stream<uint> fifo_PE7_0_compute_config_out;
	stream<uint> fifo_PE7_0_res_config_out;
#pragma HLS STREAM variable=fifo_PE7_0_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_0_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_0_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_0_res_config_out depth=2
	stream<uint> fifo_PE7_1_op0_config_out;
	stream<uint> fifo_PE7_1_op1_config_out;
	stream<uint> fifo_PE7_1_compute_config_out;
	stream<uint> fifo_PE7_1_res_config_out;
#pragma HLS STREAM variable=fifo_PE7_1_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_1_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_1_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_1_res_config_out depth=2
	stream<uint> fifo_PE7_2_op0_config_out;
	stream<uint> fifo_PE7_2_op1_config_out;
	stream<uint> fifo_PE7_2_compute_config_out;
	stream<uint> fifo_PE7_2_res_config_out;
#pragma HLS STREAM variable=fifo_PE7_2_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_2_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_2_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_2_res_config_out depth=2
	stream<uint> fifo_PE7_3_op0_config_out;
	stream<uint> fifo_PE7_3_op1_config_out;
	stream<uint> fifo_PE7_3_compute_config_out;
	stream<uint> fifo_PE7_3_res_config_out;
#pragma HLS STREAM variable=fifo_PE7_3_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_3_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_3_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_3_res_config_out depth=2
	stream<uint> fifo_PE7_4_op0_config_out;
	stream<uint> fifo_PE7_4_op1_config_out;
	stream<uint> fifo_PE7_4_compute_config_out;
	stream<uint> fifo_PE7_4_res_config_out;
#pragma HLS STREAM variable=fifo_PE7_4_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_4_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_4_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_4_res_config_out depth=2
	stream<uint> fifo_PE7_5_op0_config_out;
	stream<uint> fifo_PE7_5_op1_config_out;
	stream<uint> fifo_PE7_5_compute_config_out;
	stream<uint> fifo_PE7_5_res_config_out;
#pragma HLS STREAM variable=fifo_PE7_5_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_5_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_5_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_5_res_config_out depth=2
	stream<uint> fifo_PE7_6_op0_config_out;
	stream<uint> fifo_PE7_6_op1_config_out;
	stream<uint> fifo_PE7_6_compute_config_out;
	stream<uint> fifo_PE7_6_res_config_out;
#pragma HLS STREAM variable=fifo_PE7_6_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_6_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_6_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_6_res_config_out depth=2
	stream<uint> fifo_PE7_7_op0_config_out;
	stream<uint> fifo_PE7_7_op1_config_out;
	stream<uint> fifo_PE7_7_compute_config_out;
	stream<uint> fifo_PE7_7_res_config_out;
#pragma HLS STREAM variable=fifo_PE7_7_op0_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_7_op1_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_7_compute_config_out depth=2
#pragma HLS STREAM variable=fifo_PE7_7_res_config_out depth=2

	stream<U1_Data0PEChannelType> PE0_0_fifo0_local;
#pragma HLS STREAM variable=PE0_0_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE0_0_fifo1_local;
#pragma HLS STREAM variable=PE0_0_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE0_0_fifo2_local;
#pragma HLS STREAM variable=PE0_0_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE0_1_fifo0_local;
#pragma HLS STREAM variable=PE0_1_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE0_1_fifo1_local;
#pragma HLS STREAM variable=PE0_1_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE0_1_fifo2_local;
#pragma HLS STREAM variable=PE0_1_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE0_2_fifo0_local;
#pragma HLS STREAM variable=PE0_2_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE0_2_fifo1_local;
#pragma HLS STREAM variable=PE0_2_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE0_2_fifo2_local;
#pragma HLS STREAM variable=PE0_2_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE0_3_fifo0_local;
#pragma HLS STREAM variable=PE0_3_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE0_3_fifo1_local;
#pragma HLS STREAM variable=PE0_3_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE0_3_fifo2_local;
#pragma HLS STREAM variable=PE0_3_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE0_4_fifo0_local;
#pragma HLS STREAM variable=PE0_4_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE0_4_fifo1_local;
#pragma HLS STREAM variable=PE0_4_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE0_4_fifo2_local;
#pragma HLS STREAM variable=PE0_4_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE0_5_fifo0_local;
#pragma HLS STREAM variable=PE0_5_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE0_5_fifo1_local;
#pragma HLS STREAM variable=PE0_5_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE0_5_fifo2_local;
#pragma HLS STREAM variable=PE0_5_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE0_6_fifo0_local;
#pragma HLS STREAM variable=PE0_6_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE0_6_fifo1_local;
#pragma HLS STREAM variable=PE0_6_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE0_6_fifo2_local;
#pragma HLS STREAM variable=PE0_6_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE0_7_fifo0_local;
#pragma HLS STREAM variable=PE0_7_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE0_7_fifo1_local;
#pragma HLS STREAM variable=PE0_7_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE0_7_fifo2_local;
#pragma HLS STREAM variable=PE0_7_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE1_0_fifo0_local;
#pragma HLS STREAM variable=PE1_0_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE1_0_fifo1_local;
#pragma HLS STREAM variable=PE1_0_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE1_0_fifo2_local;
#pragma HLS STREAM variable=PE1_0_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE1_1_fifo0_local;
#pragma HLS STREAM variable=PE1_1_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE1_1_fifo1_local;
#pragma HLS STREAM variable=PE1_1_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE1_1_fifo2_local;
#pragma HLS STREAM variable=PE1_1_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE1_2_fifo0_local;
#pragma HLS STREAM variable=PE1_2_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE1_2_fifo1_local;
#pragma HLS STREAM variable=PE1_2_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE1_2_fifo2_local;
#pragma HLS STREAM variable=PE1_2_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE1_3_fifo0_local;
#pragma HLS STREAM variable=PE1_3_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE1_3_fifo1_local;
#pragma HLS STREAM variable=PE1_3_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE1_3_fifo2_local;
#pragma HLS STREAM variable=PE1_3_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE1_4_fifo0_local;
#pragma HLS STREAM variable=PE1_4_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE1_4_fifo1_local;
#pragma HLS STREAM variable=PE1_4_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE1_4_fifo2_local;
#pragma HLS STREAM variable=PE1_4_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE1_5_fifo0_local;
#pragma HLS STREAM variable=PE1_5_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE1_5_fifo1_local;
#pragma HLS STREAM variable=PE1_5_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE1_5_fifo2_local;
#pragma HLS STREAM variable=PE1_5_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE1_6_fifo0_local;
#pragma HLS STREAM variable=PE1_6_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE1_6_fifo1_local;
#pragma HLS STREAM variable=PE1_6_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE1_6_fifo2_local;
#pragma HLS STREAM variable=PE1_6_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE1_7_fifo0_local;
#pragma HLS STREAM variable=PE1_7_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE1_7_fifo1_local;
#pragma HLS STREAM variable=PE1_7_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE1_7_fifo2_local;
#pragma HLS STREAM variable=PE1_7_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE2_0_fifo0_local;
#pragma HLS STREAM variable=PE2_0_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE2_0_fifo1_local;
#pragma HLS STREAM variable=PE2_0_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE2_0_fifo2_local;
#pragma HLS STREAM variable=PE2_0_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE2_1_fifo0_local;
#pragma HLS STREAM variable=PE2_1_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE2_1_fifo1_local;
#pragma HLS STREAM variable=PE2_1_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE2_1_fifo2_local;
#pragma HLS STREAM variable=PE2_1_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE2_2_fifo0_local;
#pragma HLS STREAM variable=PE2_2_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE2_2_fifo1_local;
#pragma HLS STREAM variable=PE2_2_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE2_2_fifo2_local;
#pragma HLS STREAM variable=PE2_2_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE2_3_fifo0_local;
#pragma HLS STREAM variable=PE2_3_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE2_3_fifo1_local;
#pragma HLS STREAM variable=PE2_3_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE2_3_fifo2_local;
#pragma HLS STREAM variable=PE2_3_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE2_4_fifo0_local;
#pragma HLS STREAM variable=PE2_4_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE2_4_fifo1_local;
#pragma HLS STREAM variable=PE2_4_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE2_4_fifo2_local;
#pragma HLS STREAM variable=PE2_4_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE2_5_fifo0_local;
#pragma HLS STREAM variable=PE2_5_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE2_5_fifo1_local;
#pragma HLS STREAM variable=PE2_5_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE2_5_fifo2_local;
#pragma HLS STREAM variable=PE2_5_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE2_6_fifo0_local;
#pragma HLS STREAM variable=PE2_6_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE2_6_fifo1_local;
#pragma HLS STREAM variable=PE2_6_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE2_6_fifo2_local;
#pragma HLS STREAM variable=PE2_6_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE2_7_fifo0_local;
#pragma HLS STREAM variable=PE2_7_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE2_7_fifo1_local;
#pragma HLS STREAM variable=PE2_7_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE2_7_fifo2_local;
#pragma HLS STREAM variable=PE2_7_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE3_0_fifo0_local;
#pragma HLS STREAM variable=PE3_0_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE3_0_fifo1_local;
#pragma HLS STREAM variable=PE3_0_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE3_0_fifo2_local;
#pragma HLS STREAM variable=PE3_0_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE3_1_fifo0_local;
#pragma HLS STREAM variable=PE3_1_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE3_1_fifo1_local;
#pragma HLS STREAM variable=PE3_1_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE3_1_fifo2_local;
#pragma HLS STREAM variable=PE3_1_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE3_2_fifo0_local;
#pragma HLS STREAM variable=PE3_2_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE3_2_fifo1_local;
#pragma HLS STREAM variable=PE3_2_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE3_2_fifo2_local;
#pragma HLS STREAM variable=PE3_2_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE3_3_fifo0_local;
#pragma HLS STREAM variable=PE3_3_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE3_3_fifo1_local;
#pragma HLS STREAM variable=PE3_3_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE3_3_fifo2_local;
#pragma HLS STREAM variable=PE3_3_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE3_4_fifo0_local;
#pragma HLS STREAM variable=PE3_4_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE3_4_fifo1_local;
#pragma HLS STREAM variable=PE3_4_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE3_4_fifo2_local;
#pragma HLS STREAM variable=PE3_4_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE3_5_fifo0_local;
#pragma HLS STREAM variable=PE3_5_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE3_5_fifo1_local;
#pragma HLS STREAM variable=PE3_5_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE3_5_fifo2_local;
#pragma HLS STREAM variable=PE3_5_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE3_6_fifo0_local;
#pragma HLS STREAM variable=PE3_6_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE3_6_fifo1_local;
#pragma HLS STREAM variable=PE3_6_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE3_6_fifo2_local;
#pragma HLS STREAM variable=PE3_6_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE3_7_fifo0_local;
#pragma HLS STREAM variable=PE3_7_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE3_7_fifo1_local;
#pragma HLS STREAM variable=PE3_7_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE3_7_fifo2_local;
#pragma HLS STREAM variable=PE3_7_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE4_0_fifo0_local;
#pragma HLS STREAM variable=PE4_0_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE4_0_fifo1_local;
#pragma HLS STREAM variable=PE4_0_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE4_0_fifo2_local;
#pragma HLS STREAM variable=PE4_0_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE4_1_fifo0_local;
#pragma HLS STREAM variable=PE4_1_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE4_1_fifo1_local;
#pragma HLS STREAM variable=PE4_1_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE4_1_fifo2_local;
#pragma HLS STREAM variable=PE4_1_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE4_2_fifo0_local;
#pragma HLS STREAM variable=PE4_2_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE4_2_fifo1_local;
#pragma HLS STREAM variable=PE4_2_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE4_2_fifo2_local;
#pragma HLS STREAM variable=PE4_2_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE4_3_fifo0_local;
#pragma HLS STREAM variable=PE4_3_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE4_3_fifo1_local;
#pragma HLS STREAM variable=PE4_3_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE4_3_fifo2_local;
#pragma HLS STREAM variable=PE4_3_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE4_4_fifo0_local;
#pragma HLS STREAM variable=PE4_4_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE4_4_fifo1_local;
#pragma HLS STREAM variable=PE4_4_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE4_4_fifo2_local;
#pragma HLS STREAM variable=PE4_4_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE4_5_fifo0_local;
#pragma HLS STREAM variable=PE4_5_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE4_5_fifo1_local;
#pragma HLS STREAM variable=PE4_5_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE4_5_fifo2_local;
#pragma HLS STREAM variable=PE4_5_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE4_6_fifo0_local;
#pragma HLS STREAM variable=PE4_6_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE4_6_fifo1_local;
#pragma HLS STREAM variable=PE4_6_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE4_6_fifo2_local;
#pragma HLS STREAM variable=PE4_6_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE4_7_fifo0_local;
#pragma HLS STREAM variable=PE4_7_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE4_7_fifo1_local;
#pragma HLS STREAM variable=PE4_7_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE4_7_fifo2_local;
#pragma HLS STREAM variable=PE4_7_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE5_0_fifo0_local;
#pragma HLS STREAM variable=PE5_0_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE5_0_fifo1_local;
#pragma HLS STREAM variable=PE5_0_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE5_0_fifo2_local;
#pragma HLS STREAM variable=PE5_0_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE5_1_fifo0_local;
#pragma HLS STREAM variable=PE5_1_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE5_1_fifo1_local;
#pragma HLS STREAM variable=PE5_1_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE5_1_fifo2_local;
#pragma HLS STREAM variable=PE5_1_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE5_2_fifo0_local;
#pragma HLS STREAM variable=PE5_2_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE5_2_fifo1_local;
#pragma HLS STREAM variable=PE5_2_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE5_2_fifo2_local;
#pragma HLS STREAM variable=PE5_2_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE5_3_fifo0_local;
#pragma HLS STREAM variable=PE5_3_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE5_3_fifo1_local;
#pragma HLS STREAM variable=PE5_3_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE5_3_fifo2_local;
#pragma HLS STREAM variable=PE5_3_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE5_4_fifo0_local;
#pragma HLS STREAM variable=PE5_4_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE5_4_fifo1_local;
#pragma HLS STREAM variable=PE5_4_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE5_4_fifo2_local;
#pragma HLS STREAM variable=PE5_4_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE5_5_fifo0_local;
#pragma HLS STREAM variable=PE5_5_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE5_5_fifo1_local;
#pragma HLS STREAM variable=PE5_5_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE5_5_fifo2_local;
#pragma HLS STREAM variable=PE5_5_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE5_6_fifo0_local;
#pragma HLS STREAM variable=PE5_6_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE5_6_fifo1_local;
#pragma HLS STREAM variable=PE5_6_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE5_6_fifo2_local;
#pragma HLS STREAM variable=PE5_6_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE5_7_fifo0_local;
#pragma HLS STREAM variable=PE5_7_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE5_7_fifo1_local;
#pragma HLS STREAM variable=PE5_7_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE5_7_fifo2_local;
#pragma HLS STREAM variable=PE5_7_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE6_0_fifo0_local;
#pragma HLS STREAM variable=PE6_0_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE6_0_fifo1_local;
#pragma HLS STREAM variable=PE6_0_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE6_0_fifo2_local;
#pragma HLS STREAM variable=PE6_0_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE6_1_fifo0_local;
#pragma HLS STREAM variable=PE6_1_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE6_1_fifo1_local;
#pragma HLS STREAM variable=PE6_1_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE6_1_fifo2_local;
#pragma HLS STREAM variable=PE6_1_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE6_2_fifo0_local;
#pragma HLS STREAM variable=PE6_2_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE6_2_fifo1_local;
#pragma HLS STREAM variable=PE6_2_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE6_2_fifo2_local;
#pragma HLS STREAM variable=PE6_2_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE6_3_fifo0_local;
#pragma HLS STREAM variable=PE6_3_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE6_3_fifo1_local;
#pragma HLS STREAM variable=PE6_3_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE6_3_fifo2_local;
#pragma HLS STREAM variable=PE6_3_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE6_4_fifo0_local;
#pragma HLS STREAM variable=PE6_4_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE6_4_fifo1_local;
#pragma HLS STREAM variable=PE6_4_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE6_4_fifo2_local;
#pragma HLS STREAM variable=PE6_4_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE6_5_fifo0_local;
#pragma HLS STREAM variable=PE6_5_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE6_5_fifo1_local;
#pragma HLS STREAM variable=PE6_5_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE6_5_fifo2_local;
#pragma HLS STREAM variable=PE6_5_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE6_6_fifo0_local;
#pragma HLS STREAM variable=PE6_6_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE6_6_fifo1_local;
#pragma HLS STREAM variable=PE6_6_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE6_6_fifo2_local;
#pragma HLS STREAM variable=PE6_6_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE6_7_fifo0_local;
#pragma HLS STREAM variable=PE6_7_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE6_7_fifo1_local;
#pragma HLS STREAM variable=PE6_7_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE6_7_fifo2_local;
#pragma HLS STREAM variable=PE6_7_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE7_0_fifo0_local;
#pragma HLS STREAM variable=PE7_0_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE7_0_fifo1_local;
#pragma HLS STREAM variable=PE7_0_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE7_0_fifo2_local;
#pragma HLS STREAM variable=PE7_0_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE7_1_fifo0_local;
#pragma HLS STREAM variable=PE7_1_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE7_1_fifo1_local;
#pragma HLS STREAM variable=PE7_1_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE7_1_fifo2_local;
#pragma HLS STREAM variable=PE7_1_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE7_2_fifo0_local;
#pragma HLS STREAM variable=PE7_2_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE7_2_fifo1_local;
#pragma HLS STREAM variable=PE7_2_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE7_2_fifo2_local;
#pragma HLS STREAM variable=PE7_2_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE7_3_fifo0_local;
#pragma HLS STREAM variable=PE7_3_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE7_3_fifo1_local;
#pragma HLS STREAM variable=PE7_3_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE7_3_fifo2_local;
#pragma HLS STREAM variable=PE7_3_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE7_4_fifo0_local;
#pragma HLS STREAM variable=PE7_4_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE7_4_fifo1_local;
#pragma HLS STREAM variable=PE7_4_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE7_4_fifo2_local;
#pragma HLS STREAM variable=PE7_4_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE7_5_fifo0_local;
#pragma HLS STREAM variable=PE7_5_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE7_5_fifo1_local;
#pragma HLS STREAM variable=PE7_5_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE7_5_fifo2_local;
#pragma HLS STREAM variable=PE7_5_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE7_6_fifo0_local;
#pragma HLS STREAM variable=PE7_6_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE7_6_fifo1_local;
#pragma HLS STREAM variable=PE7_6_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE7_6_fifo2_local;
#pragma HLS STREAM variable=PE7_6_fifo2_local depth=2
	stream<U1_Data0PEChannelType> PE7_7_fifo0_local;
#pragma HLS STREAM variable=PE7_7_fifo0_local depth=2
	stream<U1_Data1PEChannelType> PE7_7_fifo1_local;
#pragma HLS STREAM variable=PE7_7_fifo1_local depth=2
	stream<U1_Data2PEChannelType> PE7_7_fifo2_local;
#pragma HLS STREAM variable=PE7_7_fifo2_local depth=2

	// modules
	U1_DataFeed0Head(
			fifo_cin,
			fifo0_transfer0,
			fifo_kernel_config_in,
			fifo_kernel_config_out,
			fifo_DataFeed0Head_config_out0, fifo_DataFeed0Head_config_out1
	);

	U1_DataFeed0Engine0_wrapper(
			fifo0_transfer0,
			fifo0_transfer1,
			fifo0_feed0_0,
			0,
			fifo_DataFeed0Head_config_out0,
			fifo_DataFeed0Engine0_config_out0,
			fifo_DataFeed0Engine0_config_out1
	);

	U1_DataFeed0Engine0_wrapper(
			fifo0_transfer1,
			fifo0_transfer2,
			fifo0_feed0_1,
			1,
			fifo_DataFeed0Engine0_config_out0,
			fifo_DataFeed0Engine1_config_out0,
			fifo_DataFeed0Engine1_config_out1
	);

	U1_DataFeed0Engine0_wrapper(
			fifo0_transfer2,
			fifo0_transfer3,
			fifo0_feed0_2,
			2,
			fifo_DataFeed0Engine1_config_out0,
			fifo_DataFeed0Engine2_config_out0,
			fifo_DataFeed0Engine2_config_out1
	);

	U1_DataFeed0Engine0_wrapper(
			fifo0_transfer3,
			fifo0_transfer4,
			fifo0_feed0_3,
			3,
			fifo_DataFeed0Engine2_config_out0,
			fifo_DataFeed0Engine3_config_out0,
			fifo_DataFeed0Engine3_config_out1
	);

	U1_DataFeed0Engine0_wrapper(
			fifo0_transfer4,
			fifo0_transfer5,
			fifo0_feed0_4,
			4,
			fifo_DataFeed0Engine3_config_out0,
			fifo_DataFeed0Engine4_config_out0,
			fifo_DataFeed0Engine4_config_out1
	);

	U1_DataFeed0Engine0_wrapper(
			fifo0_transfer5,
			fifo0_transfer6,
			fifo0_feed0_5,
			5,
			fifo_DataFeed0Engine4_config_out0,
			fifo_DataFeed0Engine5_config_out0,
			fifo_DataFeed0Engine5_config_out1
	);

	U1_DataFeed0Engine0_wrapper(
			fifo0_transfer6,
			fifo0_transfer7,
			fifo0_feed0_6,
			6,
			fifo_DataFeed0Engine5_config_out0,
			fifo_DataFeed0Engine6_config_out0,
			fifo_DataFeed0Engine6_config_out1
	);

	U1_DataFeed0EngineLast(
			fifo0_transfer7,
			fifo0_feed0_7,
			7,
			fifo_DataFeed0Engine6_config_out0,
			fifo_DataFeed0Engine7_config_out1
	);

	U1_DataFeed1Head(
			fifo_weight,
			fifo1_transfer0,
			fifo_DataFeed0Head_config_out1, fifo_DataFeed1Head_config_out0
	);

	U1_DataFeed1Engine0_wrapper(
			fifo1_transfer0,
			fifo1_transfer1,
			fifo1_feed0_0,
			0,
			fifo_DataFeed1Head_config_out0,
			fifo_DataFeed1Engine0_config_out0
	);

	U1_DataFeed1Engine0_wrapper(
			fifo1_transfer1,
			fifo1_transfer2,
			fifo1_feed1_0,
			1,
			fifo_DataFeed1Engine0_config_out0,
			fifo_DataFeed1Engine1_config_out0
	);

	U1_DataFeed1Engine0_wrapper(
			fifo1_transfer2,
			fifo1_transfer3,
			fifo1_feed2_0,
			2,
			fifo_DataFeed1Engine1_config_out0,
			fifo_DataFeed1Engine2_config_out0
	);

	U1_DataFeed1Engine0_wrapper(
			fifo1_transfer3,
			fifo1_transfer4,
			fifo1_feed3_0,
			3,
			fifo_DataFeed1Engine2_config_out0,
			fifo_DataFeed1Engine3_config_out0
	);

	U1_DataFeed1Engine0_wrapper(
			fifo1_transfer4,
			fifo1_transfer5,
			fifo1_feed4_0,
			4,
			fifo_DataFeed1Engine3_config_out0,
			fifo_DataFeed1Engine4_config_out0
	);

	U1_DataFeed1Engine0_wrapper(
			fifo1_transfer5,
			fifo1_transfer6,
			fifo1_feed5_0,
			5,
			fifo_DataFeed1Engine4_config_out0,
			fifo_DataFeed1Engine5_config_out0
	);

	U1_DataFeed1Engine0_wrapper(
			fifo1_transfer6,
			fifo1_transfer7,
			fifo1_feed6_0,
			6,
			fifo_DataFeed1Engine5_config_out0,
			fifo_DataFeed1Engine6_config_out0
	);

	U1_DataFeed1EngineLast(
			fifo1_transfer7,
			fifo1_feed7_0,
			7,
			fifo_DataFeed1Engine6_config_out0
	);

	// PE modules
	U1_op0_transfer_wrapper(
			fifo0_feed0_0,
			fifo0_feed1_0,
			PE0_0_fifo0_local,
			fifo_DataFeed0Engine0_config_out1,
			fifo_PE0_0_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed0_0,
			fifo1_feed0_1,
			PE0_0_fifo1_local,
			fifo_PE0_0_op0_config_out,
			fifo_PE0_0_op1_config_out
	);

	U1_compute_wrapper(
			PE0_0_fifo0_local,
			PE0_0_fifo1_local,
			PE0_0_fifo2_local,
			fifo_PE0_0_op1_config_out,
			fifo_PE0_0_compute_config_out
	);

	U1_res_transfer_first_wrapper(
			PE0_0_fifo2_local,
			fifo2_collect0_0,
			0,
			0,
			fifo_PE0_0_compute_config_out,
			fifo_PE0_0_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed0_1,
			fifo0_feed1_1,
			PE0_1_fifo0_local,
			fifo_DataFeed0Engine1_config_out1,
			fifo_PE0_1_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed0_1,
			fifo1_feed0_2,
			PE0_1_fifo1_local,
			fifo_PE0_1_op0_config_out,
			fifo_PE0_1_op1_config_out
	);

	U1_compute_wrapper(
			PE0_1_fifo0_local,
			PE0_1_fifo1_local,
			PE0_1_fifo2_local,
			fifo_PE0_1_op1_config_out,
			fifo_PE0_1_compute_config_out
	);

	U1_res_transfer_first_wrapper(
			PE0_1_fifo2_local,
			fifo2_collect0_1,
			0,
			1,
			fifo_PE0_1_compute_config_out,
			fifo_PE0_1_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed0_2,
			fifo0_feed1_2,
			PE0_2_fifo0_local,
			fifo_DataFeed0Engine2_config_out1,
			fifo_PE0_2_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed0_2,
			fifo1_feed0_3,
			PE0_2_fifo1_local,
			fifo_PE0_2_op0_config_out,
			fifo_PE0_2_op1_config_out
	);

	U1_compute_wrapper(
			PE0_2_fifo0_local,
			PE0_2_fifo1_local,
			PE0_2_fifo2_local,
			fifo_PE0_2_op1_config_out,
			fifo_PE0_2_compute_config_out
	);

	U1_res_transfer_first_wrapper(
			PE0_2_fifo2_local,
			fifo2_collect0_2,
			0,
			2,
			fifo_PE0_2_compute_config_out,
			fifo_PE0_2_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed0_3,
			fifo0_feed1_3,
			PE0_3_fifo0_local,
			fifo_DataFeed0Engine3_config_out1,
			fifo_PE0_3_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed0_3,
			fifo1_feed0_4,
			PE0_3_fifo1_local,
			fifo_PE0_3_op0_config_out,
			fifo_PE0_3_op1_config_out
	);

	U1_compute_wrapper(
			PE0_3_fifo0_local,
			PE0_3_fifo1_local,
			PE0_3_fifo2_local,
			fifo_PE0_3_op1_config_out,
			fifo_PE0_3_compute_config_out
	);

	U1_res_transfer_first_wrapper(
			PE0_3_fifo2_local,
			fifo2_collect0_3,
			0,
			3,
			fifo_PE0_3_compute_config_out,
			fifo_PE0_3_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed0_4,
			fifo0_feed1_4,
			PE0_4_fifo0_local,
			fifo_DataFeed0Engine4_config_out1,
			fifo_PE0_4_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed0_4,
			fifo1_feed0_5,
			PE0_4_fifo1_local,
			fifo_PE0_4_op0_config_out,
			fifo_PE0_4_op1_config_out
	);

	U1_compute_wrapper(
			PE0_4_fifo0_local,
			PE0_4_fifo1_local,
			PE0_4_fifo2_local,
			fifo_PE0_4_op1_config_out,
			fifo_PE0_4_compute_config_out
	);

	U1_res_transfer_first_wrapper(
			PE0_4_fifo2_local,
			fifo2_collect0_4,
			0,
			4,
			fifo_PE0_4_compute_config_out,
			fifo_PE0_4_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed0_5,
			fifo0_feed1_5,
			PE0_5_fifo0_local,
			fifo_DataFeed0Engine5_config_out1,
			fifo_PE0_5_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed0_5,
			fifo1_feed0_6,
			PE0_5_fifo1_local,
			fifo_PE0_5_op0_config_out,
			fifo_PE0_5_op1_config_out
	);

	U1_compute_wrapper(
			PE0_5_fifo0_local,
			PE0_5_fifo1_local,
			PE0_5_fifo2_local,
			fifo_PE0_5_op1_config_out,
			fifo_PE0_5_compute_config_out
	);

	U1_res_transfer_first_wrapper(
			PE0_5_fifo2_local,
			fifo2_collect0_5,
			0,
			5,
			fifo_PE0_5_compute_config_out,
			fifo_PE0_5_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed0_6,
			fifo0_feed1_6,
			PE0_6_fifo0_local,
			fifo_DataFeed0Engine6_config_out1,
			fifo_PE0_6_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed0_6,
			fifo1_feed0_7,
			PE0_6_fifo1_local,
			fifo_PE0_6_op0_config_out,
			fifo_PE0_6_op1_config_out
	);

	U1_compute_wrapper(
			PE0_6_fifo0_local,
			PE0_6_fifo1_local,
			PE0_6_fifo2_local,
			fifo_PE0_6_op1_config_out,
			fifo_PE0_6_compute_config_out
	);

	U1_res_transfer_first_wrapper(
			PE0_6_fifo2_local,
			fifo2_collect0_6,
			0,
			6,
			fifo_PE0_6_compute_config_out,
			fifo_PE0_6_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed0_7,
			fifo0_feed1_7,
			PE0_7_fifo0_local,
			fifo_DataFeed0Engine7_config_out1,
			fifo_PE0_7_op0_config_out
	);

	U1_op1_transfer_last_wrapper(
			fifo1_feed0_7,
			PE0_7_fifo1_local,
			fifo_PE0_7_op0_config_out,
			fifo_PE0_7_op1_config_out
	);

	U1_compute_wrapper(
			PE0_7_fifo0_local,
			PE0_7_fifo1_local,
			PE0_7_fifo2_local,
			fifo_PE0_7_op1_config_out,
			fifo_PE0_7_compute_config_out
	);

	U1_res_transfer_first_wrapper(
			PE0_7_fifo2_local,
			fifo2_collect0_7,
			0,
			7,
			fifo_PE0_7_compute_config_out,
			fifo_PE0_7_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed1_0,
			fifo0_feed2_0,
			PE1_0_fifo0_local,
			fifo_PE0_0_res_config_out,
			fifo_PE1_0_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed1_0,
			fifo1_feed1_1,
			PE1_0_fifo1_local,
			fifo_PE1_0_op0_config_out,
			fifo_PE1_0_op1_config_out
	);

	U1_compute_wrapper(
			PE1_0_fifo0_local,
			PE1_0_fifo1_local,
			PE1_0_fifo2_local,
			fifo_PE1_0_op1_config_out,
			fifo_PE1_0_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE1_0_fifo2_local,
			fifo2_collect0_0,
			fifo2_collect1_0,
			1,
			0,
			fifo_PE1_0_compute_config_out,
			fifo_PE1_0_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed1_1,
			fifo0_feed2_1,
			PE1_1_fifo0_local,
			fifo_PE0_1_res_config_out,
			fifo_PE1_1_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed1_1,
			fifo1_feed1_2,
			PE1_1_fifo1_local,
			fifo_PE1_1_op0_config_out,
			fifo_PE1_1_op1_config_out
	);

	U1_compute_wrapper(
			PE1_1_fifo0_local,
			PE1_1_fifo1_local,
			PE1_1_fifo2_local,
			fifo_PE1_1_op1_config_out,
			fifo_PE1_1_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE1_1_fifo2_local,
			fifo2_collect0_1,
			fifo2_collect1_1,
			1,
			1,
			fifo_PE1_1_compute_config_out,
			fifo_PE1_1_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed1_2,
			fifo0_feed2_2,
			PE1_2_fifo0_local,
			fifo_PE0_2_res_config_out,
			fifo_PE1_2_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed1_2,
			fifo1_feed1_3,
			PE1_2_fifo1_local,
			fifo_PE1_2_op0_config_out,
			fifo_PE1_2_op1_config_out
	);

	U1_compute_wrapper(
			PE1_2_fifo0_local,
			PE1_2_fifo1_local,
			PE1_2_fifo2_local,
			fifo_PE1_2_op1_config_out,
			fifo_PE1_2_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE1_2_fifo2_local,
			fifo2_collect0_2,
			fifo2_collect1_2,
			1,
			2,
			fifo_PE1_2_compute_config_out,
			fifo_PE1_2_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed1_3,
			fifo0_feed2_3,
			PE1_3_fifo0_local,
			fifo_PE0_3_res_config_out,
			fifo_PE1_3_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed1_3,
			fifo1_feed1_4,
			PE1_3_fifo1_local,
			fifo_PE1_3_op0_config_out,
			fifo_PE1_3_op1_config_out
	);

	U1_compute_wrapper(
			PE1_3_fifo0_local,
			PE1_3_fifo1_local,
			PE1_3_fifo2_local,
			fifo_PE1_3_op1_config_out,
			fifo_PE1_3_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE1_3_fifo2_local,
			fifo2_collect0_3,
			fifo2_collect1_3,
			1,
			3,
			fifo_PE1_3_compute_config_out,
			fifo_PE1_3_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed1_4,
			fifo0_feed2_4,
			PE1_4_fifo0_local,
			fifo_PE0_4_res_config_out,
			fifo_PE1_4_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed1_4,
			fifo1_feed1_5,
			PE1_4_fifo1_local,
			fifo_PE1_4_op0_config_out,
			fifo_PE1_4_op1_config_out
	);

	U1_compute_wrapper(
			PE1_4_fifo0_local,
			PE1_4_fifo1_local,
			PE1_4_fifo2_local,
			fifo_PE1_4_op1_config_out,
			fifo_PE1_4_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE1_4_fifo2_local,
			fifo2_collect0_4,
			fifo2_collect1_4,
			1,
			4,
			fifo_PE1_4_compute_config_out,
			fifo_PE1_4_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed1_5,
			fifo0_feed2_5,
			PE1_5_fifo0_local,
			fifo_PE0_5_res_config_out,
			fifo_PE1_5_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed1_5,
			fifo1_feed1_6,
			PE1_5_fifo1_local,
			fifo_PE1_5_op0_config_out,
			fifo_PE1_5_op1_config_out
	);

	U1_compute_wrapper(
			PE1_5_fifo0_local,
			PE1_5_fifo1_local,
			PE1_5_fifo2_local,
			fifo_PE1_5_op1_config_out,
			fifo_PE1_5_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE1_5_fifo2_local,
			fifo2_collect0_5,
			fifo2_collect1_5,
			1,
			5,
			fifo_PE1_5_compute_config_out,
			fifo_PE1_5_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed1_6,
			fifo0_feed2_6,
			PE1_6_fifo0_local,
			fifo_PE0_6_res_config_out,
			fifo_PE1_6_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed1_6,
			fifo1_feed1_7,
			PE1_6_fifo1_local,
			fifo_PE1_6_op0_config_out,
			fifo_PE1_6_op1_config_out
	);

	U1_compute_wrapper(
			PE1_6_fifo0_local,
			PE1_6_fifo1_local,
			PE1_6_fifo2_local,
			fifo_PE1_6_op1_config_out,
			fifo_PE1_6_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE1_6_fifo2_local,
			fifo2_collect0_6,
			fifo2_collect1_6,
			1,
			6,
			fifo_PE1_6_compute_config_out,
			fifo_PE1_6_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed1_7,
			fifo0_feed2_7,
			PE1_7_fifo0_local,
			fifo_PE0_7_res_config_out,
			fifo_PE1_7_op0_config_out
	);

	U1_op1_transfer_last_wrapper(
			fifo1_feed1_7,
			PE1_7_fifo1_local,
			fifo_PE1_7_op0_config_out,
			fifo_PE1_7_op1_config_out
	);

	U1_compute_wrapper(
			PE1_7_fifo0_local,
			PE1_7_fifo1_local,
			PE1_7_fifo2_local,
			fifo_PE1_7_op1_config_out,
			fifo_PE1_7_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE1_7_fifo2_local,
			fifo2_collect0_7,
			fifo2_collect1_7,
			1,
			7,
			fifo_PE1_7_compute_config_out,
			fifo_PE1_7_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed2_0,
			fifo0_feed3_0,
			PE2_0_fifo0_local,
			fifo_PE1_0_res_config_out,
			fifo_PE2_0_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed2_0,
			fifo1_feed2_1,
			PE2_0_fifo1_local,
			fifo_PE2_0_op0_config_out,
			fifo_PE2_0_op1_config_out
	);

	U1_compute_wrapper(
			PE2_0_fifo0_local,
			PE2_0_fifo1_local,
			PE2_0_fifo2_local,
			fifo_PE2_0_op1_config_out,
			fifo_PE2_0_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE2_0_fifo2_local,
			fifo2_collect1_0,
			fifo2_collect2_0,
			2,
			0,
			fifo_PE2_0_compute_config_out,
			fifo_PE2_0_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed2_1,
			fifo0_feed3_1,
			PE2_1_fifo0_local,
			fifo_PE1_1_res_config_out,
			fifo_PE2_1_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed2_1,
			fifo1_feed2_2,
			PE2_1_fifo1_local,
			fifo_PE2_1_op0_config_out,
			fifo_PE2_1_op1_config_out
	);

	U1_compute_wrapper(
			PE2_1_fifo0_local,
			PE2_1_fifo1_local,
			PE2_1_fifo2_local,
			fifo_PE2_1_op1_config_out,
			fifo_PE2_1_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE2_1_fifo2_local,
			fifo2_collect1_1,
			fifo2_collect2_1,
			2,
			1,
			fifo_PE2_1_compute_config_out,
			fifo_PE2_1_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed2_2,
			fifo0_feed3_2,
			PE2_2_fifo0_local,
			fifo_PE1_2_res_config_out,
			fifo_PE2_2_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed2_2,
			fifo1_feed2_3,
			PE2_2_fifo1_local,
			fifo_PE2_2_op0_config_out,
			fifo_PE2_2_op1_config_out
	);

	U1_compute_wrapper(
			PE2_2_fifo0_local,
			PE2_2_fifo1_local,
			PE2_2_fifo2_local,
			fifo_PE2_2_op1_config_out,
			fifo_PE2_2_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE2_2_fifo2_local,
			fifo2_collect1_2,
			fifo2_collect2_2,
			2,
			2,
			fifo_PE2_2_compute_config_out,
			fifo_PE2_2_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed2_3,
			fifo0_feed3_3,
			PE2_3_fifo0_local,
			fifo_PE1_3_res_config_out,
			fifo_PE2_3_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed2_3,
			fifo1_feed2_4,
			PE2_3_fifo1_local,
			fifo_PE2_3_op0_config_out,
			fifo_PE2_3_op1_config_out
	);

	U1_compute_wrapper(
			PE2_3_fifo0_local,
			PE2_3_fifo1_local,
			PE2_3_fifo2_local,
			fifo_PE2_3_op1_config_out,
			fifo_PE2_3_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE2_3_fifo2_local,
			fifo2_collect1_3,
			fifo2_collect2_3,
			2,
			3,
			fifo_PE2_3_compute_config_out,
			fifo_PE2_3_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed2_4,
			fifo0_feed3_4,
			PE2_4_fifo0_local,
			fifo_PE1_4_res_config_out,
			fifo_PE2_4_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed2_4,
			fifo1_feed2_5,
			PE2_4_fifo1_local,
			fifo_PE2_4_op0_config_out,
			fifo_PE2_4_op1_config_out
	);

	U1_compute_wrapper(
			PE2_4_fifo0_local,
			PE2_4_fifo1_local,
			PE2_4_fifo2_local,
			fifo_PE2_4_op1_config_out,
			fifo_PE2_4_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE2_4_fifo2_local,
			fifo2_collect1_4,
			fifo2_collect2_4,
			2,
			4,
			fifo_PE2_4_compute_config_out,
			fifo_PE2_4_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed2_5,
			fifo0_feed3_5,
			PE2_5_fifo0_local,
			fifo_PE1_5_res_config_out,
			fifo_PE2_5_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed2_5,
			fifo1_feed2_6,
			PE2_5_fifo1_local,
			fifo_PE2_5_op0_config_out,
			fifo_PE2_5_op1_config_out
	);

	U1_compute_wrapper(
			PE2_5_fifo0_local,
			PE2_5_fifo1_local,
			PE2_5_fifo2_local,
			fifo_PE2_5_op1_config_out,
			fifo_PE2_5_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE2_5_fifo2_local,
			fifo2_collect1_5,
			fifo2_collect2_5,
			2,
			5,
			fifo_PE2_5_compute_config_out,
			fifo_PE2_5_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed2_6,
			fifo0_feed3_6,
			PE2_6_fifo0_local,
			fifo_PE1_6_res_config_out,
			fifo_PE2_6_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed2_6,
			fifo1_feed2_7,
			PE2_6_fifo1_local,
			fifo_PE2_6_op0_config_out,
			fifo_PE2_6_op1_config_out
	);

	U1_compute_wrapper(
			PE2_6_fifo0_local,
			PE2_6_fifo1_local,
			PE2_6_fifo2_local,
			fifo_PE2_6_op1_config_out,
			fifo_PE2_6_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE2_6_fifo2_local,
			fifo2_collect1_6,
			fifo2_collect2_6,
			2,
			6,
			fifo_PE2_6_compute_config_out,
			fifo_PE2_6_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed2_7,
			fifo0_feed3_7,
			PE2_7_fifo0_local,
			fifo_PE1_7_res_config_out,
			fifo_PE2_7_op0_config_out
	);

	U1_op1_transfer_last_wrapper(
			fifo1_feed2_7,
			PE2_7_fifo1_local,
			fifo_PE2_7_op0_config_out,
			fifo_PE2_7_op1_config_out
	);

	U1_compute_wrapper(
			PE2_7_fifo0_local,
			PE2_7_fifo1_local,
			PE2_7_fifo2_local,
			fifo_PE2_7_op1_config_out,
			fifo_PE2_7_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE2_7_fifo2_local,
			fifo2_collect1_7,
			fifo2_collect2_7,
			2,
			7,
			fifo_PE2_7_compute_config_out,
			fifo_PE2_7_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed3_0,
			fifo0_feed4_0,
			PE3_0_fifo0_local,
			fifo_PE2_0_res_config_out,
			fifo_PE3_0_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed3_0,
			fifo1_feed3_1,
			PE3_0_fifo1_local,
			fifo_PE3_0_op0_config_out,
			fifo_PE3_0_op1_config_out
	);

	U1_compute_wrapper(
			PE3_0_fifo0_local,
			PE3_0_fifo1_local,
			PE3_0_fifo2_local,
			fifo_PE3_0_op1_config_out,
			fifo_PE3_0_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE3_0_fifo2_local,
			fifo2_collect2_0,
			fifo2_collect3_0,
			3,
			0,
			fifo_PE3_0_compute_config_out,
			fifo_PE3_0_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed3_1,
			fifo0_feed4_1,
			PE3_1_fifo0_local,
			fifo_PE2_1_res_config_out,
			fifo_PE3_1_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed3_1,
			fifo1_feed3_2,
			PE3_1_fifo1_local,
			fifo_PE3_1_op0_config_out,
			fifo_PE3_1_op1_config_out
	);

	U1_compute_wrapper(
			PE3_1_fifo0_local,
			PE3_1_fifo1_local,
			PE3_1_fifo2_local,
			fifo_PE3_1_op1_config_out,
			fifo_PE3_1_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE3_1_fifo2_local,
			fifo2_collect2_1,
			fifo2_collect3_1,
			3,
			1,
			fifo_PE3_1_compute_config_out,
			fifo_PE3_1_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed3_2,
			fifo0_feed4_2,
			PE3_2_fifo0_local,
			fifo_PE2_2_res_config_out,
			fifo_PE3_2_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed3_2,
			fifo1_feed3_3,
			PE3_2_fifo1_local,
			fifo_PE3_2_op0_config_out,
			fifo_PE3_2_op1_config_out
	);

	U1_compute_wrapper(
			PE3_2_fifo0_local,
			PE3_2_fifo1_local,
			PE3_2_fifo2_local,
			fifo_PE3_2_op1_config_out,
			fifo_PE3_2_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE3_2_fifo2_local,
			fifo2_collect2_2,
			fifo2_collect3_2,
			3,
			2,
			fifo_PE3_2_compute_config_out,
			fifo_PE3_2_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed3_3,
			fifo0_feed4_3,
			PE3_3_fifo0_local,
			fifo_PE2_3_res_config_out,
			fifo_PE3_3_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed3_3,
			fifo1_feed3_4,
			PE3_3_fifo1_local,
			fifo_PE3_3_op0_config_out,
			fifo_PE3_3_op1_config_out
	);

	U1_compute_wrapper(
			PE3_3_fifo0_local,
			PE3_3_fifo1_local,
			PE3_3_fifo2_local,
			fifo_PE3_3_op1_config_out,
			fifo_PE3_3_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE3_3_fifo2_local,
			fifo2_collect2_3,
			fifo2_collect3_3,
			3,
			3,
			fifo_PE3_3_compute_config_out,
			fifo_PE3_3_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed3_4,
			fifo0_feed4_4,
			PE3_4_fifo0_local,
			fifo_PE2_4_res_config_out,
			fifo_PE3_4_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed3_4,
			fifo1_feed3_5,
			PE3_4_fifo1_local,
			fifo_PE3_4_op0_config_out,
			fifo_PE3_4_op1_config_out
	);

	U1_compute_wrapper(
			PE3_4_fifo0_local,
			PE3_4_fifo1_local,
			PE3_4_fifo2_local,
			fifo_PE3_4_op1_config_out,
			fifo_PE3_4_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE3_4_fifo2_local,
			fifo2_collect2_4,
			fifo2_collect3_4,
			3,
			4,
			fifo_PE3_4_compute_config_out,
			fifo_PE3_4_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed3_5,
			fifo0_feed4_5,
			PE3_5_fifo0_local,
			fifo_PE2_5_res_config_out,
			fifo_PE3_5_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed3_5,
			fifo1_feed3_6,
			PE3_5_fifo1_local,
			fifo_PE3_5_op0_config_out,
			fifo_PE3_5_op1_config_out
	);

	U1_compute_wrapper(
			PE3_5_fifo0_local,
			PE3_5_fifo1_local,
			PE3_5_fifo2_local,
			fifo_PE3_5_op1_config_out,
			fifo_PE3_5_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE3_5_fifo2_local,
			fifo2_collect2_5,
			fifo2_collect3_5,
			3,
			5,
			fifo_PE3_5_compute_config_out,
			fifo_PE3_5_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed3_6,
			fifo0_feed4_6,
			PE3_6_fifo0_local,
			fifo_PE2_6_res_config_out,
			fifo_PE3_6_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed3_6,
			fifo1_feed3_7,
			PE3_6_fifo1_local,
			fifo_PE3_6_op0_config_out,
			fifo_PE3_6_op1_config_out
	);

	U1_compute_wrapper(
			PE3_6_fifo0_local,
			PE3_6_fifo1_local,
			PE3_6_fifo2_local,
			fifo_PE3_6_op1_config_out,
			fifo_PE3_6_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE3_6_fifo2_local,
			fifo2_collect2_6,
			fifo2_collect3_6,
			3,
			6,
			fifo_PE3_6_compute_config_out,
			fifo_PE3_6_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed3_7,
			fifo0_feed4_7,
			PE3_7_fifo0_local,
			fifo_PE2_7_res_config_out,
			fifo_PE3_7_op0_config_out
	);

	U1_op1_transfer_last_wrapper(
			fifo1_feed3_7,
			PE3_7_fifo1_local,
			fifo_PE3_7_op0_config_out,
			fifo_PE3_7_op1_config_out
	);

	U1_compute_wrapper(
			PE3_7_fifo0_local,
			PE3_7_fifo1_local,
			PE3_7_fifo2_local,
			fifo_PE3_7_op1_config_out,
			fifo_PE3_7_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE3_7_fifo2_local,
			fifo2_collect2_7,
			fifo2_collect3_7,
			3,
			7,
			fifo_PE3_7_compute_config_out,
			fifo_PE3_7_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed4_0,
			fifo0_feed5_0,
			PE4_0_fifo0_local,
			fifo_PE3_0_res_config_out,
			fifo_PE4_0_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed4_0,
			fifo1_feed4_1,
			PE4_0_fifo1_local,
			fifo_PE4_0_op0_config_out,
			fifo_PE4_0_op1_config_out
	);

	U1_compute_wrapper(
			PE4_0_fifo0_local,
			PE4_0_fifo1_local,
			PE4_0_fifo2_local,
			fifo_PE4_0_op1_config_out,
			fifo_PE4_0_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE4_0_fifo2_local,
			fifo2_collect3_0,
			fifo2_collect4_0,
			4,
			0,
			fifo_PE4_0_compute_config_out,
			fifo_PE4_0_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed4_1,
			fifo0_feed5_1,
			PE4_1_fifo0_local,
			fifo_PE3_1_res_config_out,
			fifo_PE4_1_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed4_1,
			fifo1_feed4_2,
			PE4_1_fifo1_local,
			fifo_PE4_1_op0_config_out,
			fifo_PE4_1_op1_config_out
	);

	U1_compute_wrapper(
			PE4_1_fifo0_local,
			PE4_1_fifo1_local,
			PE4_1_fifo2_local,
			fifo_PE4_1_op1_config_out,
			fifo_PE4_1_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE4_1_fifo2_local,
			fifo2_collect3_1,
			fifo2_collect4_1,
			4,
			1,
			fifo_PE4_1_compute_config_out,
			fifo_PE4_1_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed4_2,
			fifo0_feed5_2,
			PE4_2_fifo0_local,
			fifo_PE3_2_res_config_out,
			fifo_PE4_2_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed4_2,
			fifo1_feed4_3,
			PE4_2_fifo1_local,
			fifo_PE4_2_op0_config_out,
			fifo_PE4_2_op1_config_out
	);

	U1_compute_wrapper(
			PE4_2_fifo0_local,
			PE4_2_fifo1_local,
			PE4_2_fifo2_local,
			fifo_PE4_2_op1_config_out,
			fifo_PE4_2_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE4_2_fifo2_local,
			fifo2_collect3_2,
			fifo2_collect4_2,
			4,
			2,
			fifo_PE4_2_compute_config_out,
			fifo_PE4_2_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed4_3,
			fifo0_feed5_3,
			PE4_3_fifo0_local,
			fifo_PE3_3_res_config_out,
			fifo_PE4_3_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed4_3,
			fifo1_feed4_4,
			PE4_3_fifo1_local,
			fifo_PE4_3_op0_config_out,
			fifo_PE4_3_op1_config_out
	);

	U1_compute_wrapper(
			PE4_3_fifo0_local,
			PE4_3_fifo1_local,
			PE4_3_fifo2_local,
			fifo_PE4_3_op1_config_out,
			fifo_PE4_3_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE4_3_fifo2_local,
			fifo2_collect3_3,
			fifo2_collect4_3,
			4,
			3,
			fifo_PE4_3_compute_config_out,
			fifo_PE4_3_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed4_4,
			fifo0_feed5_4,
			PE4_4_fifo0_local,
			fifo_PE3_4_res_config_out,
			fifo_PE4_4_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed4_4,
			fifo1_feed4_5,
			PE4_4_fifo1_local,
			fifo_PE4_4_op0_config_out,
			fifo_PE4_4_op1_config_out
	);

	U1_compute_wrapper(
			PE4_4_fifo0_local,
			PE4_4_fifo1_local,
			PE4_4_fifo2_local,
			fifo_PE4_4_op1_config_out,
			fifo_PE4_4_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE4_4_fifo2_local,
			fifo2_collect3_4,
			fifo2_collect4_4,
			4,
			4,
			fifo_PE4_4_compute_config_out,
			fifo_PE4_4_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed4_5,
			fifo0_feed5_5,
			PE4_5_fifo0_local,
			fifo_PE3_5_res_config_out,
			fifo_PE4_5_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed4_5,
			fifo1_feed4_6,
			PE4_5_fifo1_local,
			fifo_PE4_5_op0_config_out,
			fifo_PE4_5_op1_config_out
	);

	U1_compute_wrapper(
			PE4_5_fifo0_local,
			PE4_5_fifo1_local,
			PE4_5_fifo2_local,
			fifo_PE4_5_op1_config_out,
			fifo_PE4_5_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE4_5_fifo2_local,
			fifo2_collect3_5,
			fifo2_collect4_5,
			4,
			5,
			fifo_PE4_5_compute_config_out,
			fifo_PE4_5_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed4_6,
			fifo0_feed5_6,
			PE4_6_fifo0_local,
			fifo_PE3_6_res_config_out,
			fifo_PE4_6_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed4_6,
			fifo1_feed4_7,
			PE4_6_fifo1_local,
			fifo_PE4_6_op0_config_out,
			fifo_PE4_6_op1_config_out
	);

	U1_compute_wrapper(
			PE4_6_fifo0_local,
			PE4_6_fifo1_local,
			PE4_6_fifo2_local,
			fifo_PE4_6_op1_config_out,
			fifo_PE4_6_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE4_6_fifo2_local,
			fifo2_collect3_6,
			fifo2_collect4_6,
			4,
			6,
			fifo_PE4_6_compute_config_out,
			fifo_PE4_6_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed4_7,
			fifo0_feed5_7,
			PE4_7_fifo0_local,
			fifo_PE3_7_res_config_out,
			fifo_PE4_7_op0_config_out
	);

	U1_op1_transfer_last_wrapper(
			fifo1_feed4_7,
			PE4_7_fifo1_local,
			fifo_PE4_7_op0_config_out,
			fifo_PE4_7_op1_config_out
	);

	U1_compute_wrapper(
			PE4_7_fifo0_local,
			PE4_7_fifo1_local,
			PE4_7_fifo2_local,
			fifo_PE4_7_op1_config_out,
			fifo_PE4_7_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE4_7_fifo2_local,
			fifo2_collect3_7,
			fifo2_collect4_7,
			4,
			7,
			fifo_PE4_7_compute_config_out,
			fifo_PE4_7_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed5_0,
			fifo0_feed6_0,
			PE5_0_fifo0_local,
			fifo_PE4_0_res_config_out,
			fifo_PE5_0_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed5_0,
			fifo1_feed5_1,
			PE5_0_fifo1_local,
			fifo_PE5_0_op0_config_out,
			fifo_PE5_0_op1_config_out
	);

	U1_compute_wrapper(
			PE5_0_fifo0_local,
			PE5_0_fifo1_local,
			PE5_0_fifo2_local,
			fifo_PE5_0_op1_config_out,
			fifo_PE5_0_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE5_0_fifo2_local,
			fifo2_collect4_0,
			fifo2_collect5_0,
			5,
			0,
			fifo_PE5_0_compute_config_out,
			fifo_PE5_0_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed5_1,
			fifo0_feed6_1,
			PE5_1_fifo0_local,
			fifo_PE4_1_res_config_out,
			fifo_PE5_1_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed5_1,
			fifo1_feed5_2,
			PE5_1_fifo1_local,
			fifo_PE5_1_op0_config_out,
			fifo_PE5_1_op1_config_out
	);

	U1_compute_wrapper(
			PE5_1_fifo0_local,
			PE5_1_fifo1_local,
			PE5_1_fifo2_local,
			fifo_PE5_1_op1_config_out,
			fifo_PE5_1_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE5_1_fifo2_local,
			fifo2_collect4_1,
			fifo2_collect5_1,
			5,
			1,
			fifo_PE5_1_compute_config_out,
			fifo_PE5_1_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed5_2,
			fifo0_feed6_2,
			PE5_2_fifo0_local,
			fifo_PE4_2_res_config_out,
			fifo_PE5_2_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed5_2,
			fifo1_feed5_3,
			PE5_2_fifo1_local,
			fifo_PE5_2_op0_config_out,
			fifo_PE5_2_op1_config_out
	);

	U1_compute_wrapper(
			PE5_2_fifo0_local,
			PE5_2_fifo1_local,
			PE5_2_fifo2_local,
			fifo_PE5_2_op1_config_out,
			fifo_PE5_2_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE5_2_fifo2_local,
			fifo2_collect4_2,
			fifo2_collect5_2,
			5,
			2,
			fifo_PE5_2_compute_config_out,
			fifo_PE5_2_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed5_3,
			fifo0_feed6_3,
			PE5_3_fifo0_local,
			fifo_PE4_3_res_config_out,
			fifo_PE5_3_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed5_3,
			fifo1_feed5_4,
			PE5_3_fifo1_local,
			fifo_PE5_3_op0_config_out,
			fifo_PE5_3_op1_config_out
	);

	U1_compute_wrapper(
			PE5_3_fifo0_local,
			PE5_3_fifo1_local,
			PE5_3_fifo2_local,
			fifo_PE5_3_op1_config_out,
			fifo_PE5_3_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE5_3_fifo2_local,
			fifo2_collect4_3,
			fifo2_collect5_3,
			5,
			3,
			fifo_PE5_3_compute_config_out,
			fifo_PE5_3_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed5_4,
			fifo0_feed6_4,
			PE5_4_fifo0_local,
			fifo_PE4_4_res_config_out,
			fifo_PE5_4_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed5_4,
			fifo1_feed5_5,
			PE5_4_fifo1_local,
			fifo_PE5_4_op0_config_out,
			fifo_PE5_4_op1_config_out
	);

	U1_compute_wrapper(
			PE5_4_fifo0_local,
			PE5_4_fifo1_local,
			PE5_4_fifo2_local,
			fifo_PE5_4_op1_config_out,
			fifo_PE5_4_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE5_4_fifo2_local,
			fifo2_collect4_4,
			fifo2_collect5_4,
			5,
			4,
			fifo_PE5_4_compute_config_out,
			fifo_PE5_4_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed5_5,
			fifo0_feed6_5,
			PE5_5_fifo0_local,
			fifo_PE4_5_res_config_out,
			fifo_PE5_5_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed5_5,
			fifo1_feed5_6,
			PE5_5_fifo1_local,
			fifo_PE5_5_op0_config_out,
			fifo_PE5_5_op1_config_out
	);

	U1_compute_wrapper(
			PE5_5_fifo0_local,
			PE5_5_fifo1_local,
			PE5_5_fifo2_local,
			fifo_PE5_5_op1_config_out,
			fifo_PE5_5_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE5_5_fifo2_local,
			fifo2_collect4_5,
			fifo2_collect5_5,
			5,
			5,
			fifo_PE5_5_compute_config_out,
			fifo_PE5_5_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed5_6,
			fifo0_feed6_6,
			PE5_6_fifo0_local,
			fifo_PE4_6_res_config_out,
			fifo_PE5_6_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed5_6,
			fifo1_feed5_7,
			PE5_6_fifo1_local,
			fifo_PE5_6_op0_config_out,
			fifo_PE5_6_op1_config_out
	);

	U1_compute_wrapper(
			PE5_6_fifo0_local,
			PE5_6_fifo1_local,
			PE5_6_fifo2_local,
			fifo_PE5_6_op1_config_out,
			fifo_PE5_6_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE5_6_fifo2_local,
			fifo2_collect4_6,
			fifo2_collect5_6,
			5,
			6,
			fifo_PE5_6_compute_config_out,
			fifo_PE5_6_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed5_7,
			fifo0_feed6_7,
			PE5_7_fifo0_local,
			fifo_PE4_7_res_config_out,
			fifo_PE5_7_op0_config_out
	);

	U1_op1_transfer_last_wrapper(
			fifo1_feed5_7,
			PE5_7_fifo1_local,
			fifo_PE5_7_op0_config_out,
			fifo_PE5_7_op1_config_out
	);

	U1_compute_wrapper(
			PE5_7_fifo0_local,
			PE5_7_fifo1_local,
			PE5_7_fifo2_local,
			fifo_PE5_7_op1_config_out,
			fifo_PE5_7_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE5_7_fifo2_local,
			fifo2_collect4_7,
			fifo2_collect5_7,
			5,
			7,
			fifo_PE5_7_compute_config_out,
			fifo_PE5_7_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed6_0,
			fifo0_feed7_0,
			PE6_0_fifo0_local,
			fifo_PE5_0_res_config_out,
			fifo_PE6_0_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed6_0,
			fifo1_feed6_1,
			PE6_0_fifo1_local,
			fifo_PE6_0_op0_config_out,
			fifo_PE6_0_op1_config_out
	);

	U1_compute_wrapper(
			PE6_0_fifo0_local,
			PE6_0_fifo1_local,
			PE6_0_fifo2_local,
			fifo_PE6_0_op1_config_out,
			fifo_PE6_0_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE6_0_fifo2_local,
			fifo2_collect5_0,
			fifo2_collect6_0,
			6,
			0,
			fifo_PE6_0_compute_config_out,
			fifo_PE6_0_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed6_1,
			fifo0_feed7_1,
			PE6_1_fifo0_local,
			fifo_PE5_1_res_config_out,
			fifo_PE6_1_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed6_1,
			fifo1_feed6_2,
			PE6_1_fifo1_local,
			fifo_PE6_1_op0_config_out,
			fifo_PE6_1_op1_config_out
	);

	U1_compute_wrapper(
			PE6_1_fifo0_local,
			PE6_1_fifo1_local,
			PE6_1_fifo2_local,
			fifo_PE6_1_op1_config_out,
			fifo_PE6_1_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE6_1_fifo2_local,
			fifo2_collect5_1,
			fifo2_collect6_1,
			6,
			1,
			fifo_PE6_1_compute_config_out,
			fifo_PE6_1_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed6_2,
			fifo0_feed7_2,
			PE6_2_fifo0_local,
			fifo_PE5_2_res_config_out,
			fifo_PE6_2_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed6_2,
			fifo1_feed6_3,
			PE6_2_fifo1_local,
			fifo_PE6_2_op0_config_out,
			fifo_PE6_2_op1_config_out
	);

	U1_compute_wrapper(
			PE6_2_fifo0_local,
			PE6_2_fifo1_local,
			PE6_2_fifo2_local,
			fifo_PE6_2_op1_config_out,
			fifo_PE6_2_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE6_2_fifo2_local,
			fifo2_collect5_2,
			fifo2_collect6_2,
			6,
			2,
			fifo_PE6_2_compute_config_out,
			fifo_PE6_2_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed6_3,
			fifo0_feed7_3,
			PE6_3_fifo0_local,
			fifo_PE5_3_res_config_out,
			fifo_PE6_3_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed6_3,
			fifo1_feed6_4,
			PE6_3_fifo1_local,
			fifo_PE6_3_op0_config_out,
			fifo_PE6_3_op1_config_out
	);

	U1_compute_wrapper(
			PE6_3_fifo0_local,
			PE6_3_fifo1_local,
			PE6_3_fifo2_local,
			fifo_PE6_3_op1_config_out,
			fifo_PE6_3_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE6_3_fifo2_local,
			fifo2_collect5_3,
			fifo2_collect6_3,
			6,
			3,
			fifo_PE6_3_compute_config_out,
			fifo_PE6_3_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed6_4,
			fifo0_feed7_4,
			PE6_4_fifo0_local,
			fifo_PE5_4_res_config_out,
			fifo_PE6_4_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed6_4,
			fifo1_feed6_5,
			PE6_4_fifo1_local,
			fifo_PE6_4_op0_config_out,
			fifo_PE6_4_op1_config_out
	);

	U1_compute_wrapper(
			PE6_4_fifo0_local,
			PE6_4_fifo1_local,
			PE6_4_fifo2_local,
			fifo_PE6_4_op1_config_out,
			fifo_PE6_4_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE6_4_fifo2_local,
			fifo2_collect5_4,
			fifo2_collect6_4,
			6,
			4,
			fifo_PE6_4_compute_config_out,
			fifo_PE6_4_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed6_5,
			fifo0_feed7_5,
			PE6_5_fifo0_local,
			fifo_PE5_5_res_config_out,
			fifo_PE6_5_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed6_5,
			fifo1_feed6_6,
			PE6_5_fifo1_local,
			fifo_PE6_5_op0_config_out,
			fifo_PE6_5_op1_config_out
	);

	U1_compute_wrapper(
			PE6_5_fifo0_local,
			PE6_5_fifo1_local,
			PE6_5_fifo2_local,
			fifo_PE6_5_op1_config_out,
			fifo_PE6_5_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE6_5_fifo2_local,
			fifo2_collect5_5,
			fifo2_collect6_5,
			6,
			5,
			fifo_PE6_5_compute_config_out,
			fifo_PE6_5_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed6_6,
			fifo0_feed7_6,
			PE6_6_fifo0_local,
			fifo_PE5_6_res_config_out,
			fifo_PE6_6_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed6_6,
			fifo1_feed6_7,
			PE6_6_fifo1_local,
			fifo_PE6_6_op0_config_out,
			fifo_PE6_6_op1_config_out
	);

	U1_compute_wrapper(
			PE6_6_fifo0_local,
			PE6_6_fifo1_local,
			PE6_6_fifo2_local,
			fifo_PE6_6_op1_config_out,
			fifo_PE6_6_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE6_6_fifo2_local,
			fifo2_collect5_6,
			fifo2_collect6_6,
			6,
			6,
			fifo_PE6_6_compute_config_out,
			fifo_PE6_6_res_config_out
	);

	U1_op0_transfer_wrapper(
			fifo0_feed6_7,
			fifo0_feed7_7,
			PE6_7_fifo0_local,
			fifo_PE5_7_res_config_out,
			fifo_PE6_7_op0_config_out
	);

	U1_op1_transfer_last_wrapper(
			fifo1_feed6_7,
			PE6_7_fifo1_local,
			fifo_PE6_7_op0_config_out,
			fifo_PE6_7_op1_config_out
	);

	U1_compute_wrapper(
			PE6_7_fifo0_local,
			PE6_7_fifo1_local,
			PE6_7_fifo2_local,
			fifo_PE6_7_op1_config_out,
			fifo_PE6_7_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE6_7_fifo2_local,
			fifo2_collect5_7,
			fifo2_collect6_7,
			6,
			7,
			fifo_PE6_7_compute_config_out,
			fifo_PE6_7_res_config_out
	);

	U1_op0_transfer_last_wrapper(
			fifo0_feed7_0,
			PE7_0_fifo0_local,
			fifo_PE6_0_res_config_out,
			fifo_PE7_0_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed7_0,
			fifo1_feed7_1,
			PE7_0_fifo1_local,
			fifo_PE7_0_op0_config_out,
			fifo_PE7_0_op1_config_out
	);

	U1_compute_wrapper(
			PE7_0_fifo0_local,
			PE7_0_fifo1_local,
			PE7_0_fifo2_local,
			fifo_PE7_0_op1_config_out,
			fifo_PE7_0_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE7_0_fifo2_local,
			fifo2_collect6_0,
			fifo2_collect7_0,
			7,
			0,
			fifo_PE7_0_compute_config_out,
			fifo_PE7_0_res_config_out
	);

	U1_op0_transfer_last_wrapper(
			fifo0_feed7_1,
			PE7_1_fifo0_local,
			fifo_PE6_1_res_config_out,
			fifo_PE7_1_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed7_1,
			fifo1_feed7_2,
			PE7_1_fifo1_local,
			fifo_PE7_1_op0_config_out,
			fifo_PE7_1_op1_config_out
	);

	U1_compute_wrapper(
			PE7_1_fifo0_local,
			PE7_1_fifo1_local,
			PE7_1_fifo2_local,
			fifo_PE7_1_op1_config_out,
			fifo_PE7_1_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE7_1_fifo2_local,
			fifo2_collect6_1,
			fifo2_collect7_1,
			7,
			1,
			fifo_PE7_1_compute_config_out,
			fifo_PE7_1_res_config_out
	);

	U1_op0_transfer_last_wrapper(
			fifo0_feed7_2,
			PE7_2_fifo0_local,
			fifo_PE6_2_res_config_out,
			fifo_PE7_2_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed7_2,
			fifo1_feed7_3,
			PE7_2_fifo1_local,
			fifo_PE7_2_op0_config_out,
			fifo_PE7_2_op1_config_out
	);

	U1_compute_wrapper(
			PE7_2_fifo0_local,
			PE7_2_fifo1_local,
			PE7_2_fifo2_local,
			fifo_PE7_2_op1_config_out,
			fifo_PE7_2_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE7_2_fifo2_local,
			fifo2_collect6_2,
			fifo2_collect7_2,
			7,
			2,
			fifo_PE7_2_compute_config_out,
			fifo_PE7_2_res_config_out
	);

	U1_op0_transfer_last_wrapper(
			fifo0_feed7_3,
			PE7_3_fifo0_local,
			fifo_PE6_3_res_config_out,
			fifo_PE7_3_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed7_3,
			fifo1_feed7_4,
			PE7_3_fifo1_local,
			fifo_PE7_3_op0_config_out,
			fifo_PE7_3_op1_config_out
	);

	U1_compute_wrapper(
			PE7_3_fifo0_local,
			PE7_3_fifo1_local,
			PE7_3_fifo2_local,
			fifo_PE7_3_op1_config_out,
			fifo_PE7_3_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE7_3_fifo2_local,
			fifo2_collect6_3,
			fifo2_collect7_3,
			7,
			3,
			fifo_PE7_3_compute_config_out,
			fifo_PE7_3_res_config_out
	);

	U1_op0_transfer_last_wrapper(
			fifo0_feed7_4,
			PE7_4_fifo0_local,
			fifo_PE6_4_res_config_out,
			fifo_PE7_4_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed7_4,
			fifo1_feed7_5,
			PE7_4_fifo1_local,
			fifo_PE7_4_op0_config_out,
			fifo_PE7_4_op1_config_out
	);

	U1_compute_wrapper(
			PE7_4_fifo0_local,
			PE7_4_fifo1_local,
			PE7_4_fifo2_local,
			fifo_PE7_4_op1_config_out,
			fifo_PE7_4_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE7_4_fifo2_local,
			fifo2_collect6_4,
			fifo2_collect7_4,
			7,
			4,
			fifo_PE7_4_compute_config_out,
			fifo_PE7_4_res_config_out
	);

	U1_op0_transfer_last_wrapper(
			fifo0_feed7_5,
			PE7_5_fifo0_local,
			fifo_PE6_5_res_config_out,
			fifo_PE7_5_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed7_5,
			fifo1_feed7_6,
			PE7_5_fifo1_local,
			fifo_PE7_5_op0_config_out,
			fifo_PE7_5_op1_config_out
	);

	U1_compute_wrapper(
			PE7_5_fifo0_local,
			PE7_5_fifo1_local,
			PE7_5_fifo2_local,
			fifo_PE7_5_op1_config_out,
			fifo_PE7_5_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE7_5_fifo2_local,
			fifo2_collect6_5,
			fifo2_collect7_5,
			7,
			5,
			fifo_PE7_5_compute_config_out,
			fifo_PE7_5_res_config_out
	);

	U1_op0_transfer_last_wrapper(
			fifo0_feed7_6,
			PE7_6_fifo0_local,
			fifo_PE6_6_res_config_out,
			fifo_PE7_6_op0_config_out
	);

	U1_op1_transfer_wrapper(
			fifo1_feed7_6,
			fifo1_feed7_7,
			PE7_6_fifo1_local,
			fifo_PE7_6_op0_config_out,
			fifo_PE7_6_op1_config_out
	);

	U1_compute_wrapper(
			PE7_6_fifo0_local,
			PE7_6_fifo1_local,
			PE7_6_fifo2_local,
			fifo_PE7_6_op1_config_out,
			fifo_PE7_6_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE7_6_fifo2_local,
			fifo2_collect6_6,
			fifo2_collect7_6,
			7,
			6,
			fifo_PE7_6_compute_config_out,
			fifo_PE7_6_res_config_out
	);

	U1_op0_transfer_last_wrapper(
			fifo0_feed7_7,
			PE7_7_fifo0_local,
			fifo_PE6_7_res_config_out,
			fifo_PE7_7_op0_config_out
	);

	U1_op1_transfer_last_wrapper(
			fifo1_feed7_7,
			PE7_7_fifo1_local,
			fifo_PE7_7_op0_config_out,
			fifo_PE7_7_op1_config_out
	);

	U1_compute_wrapper(
			PE7_7_fifo0_local,
			PE7_7_fifo1_local,
			PE7_7_fifo2_local,
			fifo_PE7_7_op1_config_out,
			fifo_PE7_7_compute_config_out
	);

	U1_res_transfer_wrapper(
			PE7_7_fifo2_local,
			fifo2_collect6_7,
			fifo2_collect7_7,
			7,
			7,
			fifo_PE7_7_compute_config_out,
			fifo_PE7_7_res_config_out
	);

	U1_DataCollect2EngineLast(
			fifo2_transfer0,
			fifo2_collect7_7,
			7,
			fifo_PE7_7_res_config_out,
			fifo_DataCollect2Engine7_config_out
	);

	U1_DataCollect2Engine0_wrapper(
			fifo2_transfer0,
			fifo2_transfer1,
			fifo2_collect7_6,
			6,
			fifo_PE7_6_res_config_out,
			fifo_DataCollect2Engine7_config_out,
			fifo_DataCollect2Engine6_config_out
	);

	U1_DataCollect2Engine0_wrapper(
			fifo2_transfer1,
			fifo2_transfer2,
			fifo2_collect7_5,
			5,
			fifo_PE7_5_res_config_out,
			fifo_DataCollect2Engine6_config_out,
			fifo_DataCollect2Engine5_config_out
	);

	U1_DataCollect2Engine0_wrapper(
			fifo2_transfer2,
			fifo2_transfer3,
			fifo2_collect7_4,
			4,
			fifo_PE7_4_res_config_out,
			fifo_DataCollect2Engine5_config_out,
			fifo_DataCollect2Engine4_config_out
	);

	U1_DataCollect2Engine0_wrapper(
			fifo2_transfer3,
			fifo2_transfer4,
			fifo2_collect7_3,
			3,
			fifo_PE7_3_res_config_out,
			fifo_DataCollect2Engine4_config_out,
			fifo_DataCollect2Engine3_config_out
	);

	U1_DataCollect2Engine0_wrapper(
			fifo2_transfer4,
			fifo2_transfer5,
			fifo2_collect7_2,
			2,
			fifo_PE7_2_res_config_out,
			fifo_DataCollect2Engine3_config_out,
			fifo_DataCollect2Engine2_config_out
	);

	U1_DataCollect2Engine0_wrapper(
			fifo2_transfer5,
			fifo2_transfer6,
			fifo2_collect7_1,
			1,
			fifo_PE7_1_res_config_out,
			fifo_DataCollect2Engine2_config_out,
			fifo_DataCollect2Engine1_config_out
	);

	U1_DataCollect2Engine0_wrapper(
			fifo2_transfer6,
			fifo2_transfer7,
			fifo2_collect7_0,
			0,
			fifo_PE7_0_res_config_out,
			fifo_DataCollect2Engine1_config_out,
			fifo_DataCollect2Engine0_config_out
	);

	U1_DataCollect2Head(
			fifo_cout,
			fifo2_transfer7,
			fifo_DataCollect2Engine0_config_out
	);

}
    
