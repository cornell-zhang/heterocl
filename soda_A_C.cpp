#include "soda_stencil.h"
#include<float.h>
#include<math.h>
#include<stdbool.h>
#include<stddef.h>
#include<stdint.h>
#include<stdio.h>
#include<string.h>
#include<ap_int.h>
#include<hls_stream.h>


#ifndef BURST_WIDTH
#define BURST_WIDTH 512
#endif//BURST_WIDTH

#ifdef UNROLL_FACTOR
#if UNROLL_FACTOR != 1
#error UNROLL_FACTOR != 1
#endif//UNROLL_FACTOR != 1
#endif//UNROLL_FACTOR
#ifdef TILE_SIZE_DIM_0
#if TILE_SIZE_DIM_0 != 8
#error TILE_SIZE_DIM_0 != 8
#endif//TILE_SIZE_DIM_0 != 8
#endif//TILE_SIZE_DIM_0
#ifdef BURST_WIDTH
#if BURST_WIDTH != 512
#error BURST_WIDTH != 512
#endif//BURST_WIDTH != 512
#endif//BURST_WIDTH

template<typename T> struct Data
{
  T data;
  bool ctrl;
};
template<typename To, typename From>
inline To Reinterpret(const From& val)
{
  return reinterpret_cast<const To&>(val);
}
template<typename T> inline bool ReadData(T* data, hls::stream<Data<T>>* from)
{
#pragma HLS inline
  const Data<T>& tmp = from->read();
  *data = tmp.data;
  return tmp.ctrl;
}
template<typename T> inline void WriteData(hls::stream<Data<T>>* to, const T& data, bool ctrl)
{
#pragma HLS inline
  Data<T> tmp;
  tmp.data = data;
  tmp.ctrl = ctrl;
  to->write(tmp);
}
void BurstRead(hls::stream<Data<ap_uint<BURST_WIDTH>>>* to, ap_uint<BURST_WIDTH>* from, uint64_t data_num)
{
load_epoch:
  for (uint64_t epoch = 0; epoch < data_num;)
  {
#pragma HLS pipeline II=1
    const uint64_t next_epoch = epoch + 1;
    WriteData(to, from[epoch], next_epoch < data_num);
    epoch = next_epoch;
  }
}
void BurstWrite(ap_uint<BURST_WIDTH>* to, hls::stream<Data<ap_uint<BURST_WIDTH>>>* from, uint64_t data_num)
{
store_epoch:
  for (uint64_t epoch = 0; epoch < data_num; ++epoch)
  {
#pragma HLS pipeline II=1
    ap_uint<BURST_WIDTH> buf;
    ReadData(&buf, from);
    to[epoch] = buf;
  }
}
void Module0Func(
  /*output*/ hls::stream<Data<int32_t>>* fifo_st_0, 
  /* input*/ hls::stream<Data<ap_uint<512>>>* dram_A_bank_0_fifo)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = dram_A_bank_0_fifo
module_0_epoch:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II=16
    if (!dram_A_bank_0_fifo->empty())
    {
      ap_uint<512> dram_A_bank_0_buf;
      const bool dram_A_bank_0_buf_enable = ReadData(&dram_A_bank_0_buf, dram_A_bank_0_fifo);
      const bool enabled = dram_A_bank_0_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(31, 0))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(63, 32))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(95, 64))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(127, 96))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(159, 128))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(191, 160))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(223, 192))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(255, 224))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(287, 256))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(319, 288))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(351, 320))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(383, 352))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(415, 384))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(447, 416))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(479, 448))), true);
      WriteData(fifo_st_0, Reinterpret<int32_t>(static_cast<ap_uint<32>>(dram_A_bank_0_buf(511, 480))), enabled);
    } // if not empty
  } // for module_0_epoch
} // Module0Func
void Module1Func(
  /*output*/ hls::stream<Data<int32_t>>* fifo_st_0, 
  /*output*/ hls::stream<Data<int32_t>>* fifo_st_1, 
  /* input*/ hls::stream<Data<int32_t>>* fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_ld_0
module_1_epoch:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II=1
    if (!fifo_ld_0->empty())
    {
      int32_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(&fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, int32_t(fifo_ref_0), enabled);
      WriteData(fifo_st_1, int32_t(fifo_ref_0), enabled);
    } // if not empty
  } // for module_1_epoch
} // Module1Func
void Module2Func(
  /*output*/ hls::stream<Data<int32_t>>* fifo_st_0, 
  /*output*/ hls::stream<Data<int32_t>>* fifo_st_1, 
  /* input*/ hls::stream<Data<int32_t>>* fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_ld_0
  int32_t fifo_ref_0_delayed_9_buf[9];
  ap_uint<4> fifo_ref_0_delayed_9_ptr = 0;
module_2_epoch:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=fifo_ref_0_delayed_9_buf inter false
    if (!fifo_ld_0->empty())
    {
      int32_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(&fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int32_t fifo_ref_0_delayed_9 = fifo_ref_0_delayed_9_buf[fifo_ref_0_delayed_9_ptr];;
      const int32_t let_0 = fifo_ref_0_delayed_9;
      WriteData(fifo_st_0, int32_t(let_0), enabled);
      WriteData(fifo_st_1, int32_t(let_0), enabled);
      fifo_ref_0_delayed_9_buf[fifo_ref_0_delayed_9_ptr] = fifo_ref_0;
      fifo_ref_0_delayed_9_ptr = fifo_ref_0_delayed_9_ptr < 8 ? ap_uint<4>(fifo_ref_0_delayed_9_ptr+1) : ap_uint<4>(0);
    } // if not empty
  } // for module_2_epoch
} // Module2Func
void Module3Func(
  /*output*/ hls::stream<Data<int32_t>>* fifo_st_0, 
  /* input*/ hls::stream<Data<int32_t>>* fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_ld_0
  int32_t fifo_ref_0_delayed_9_buf[9];
  ap_uint<4> fifo_ref_0_delayed_9_ptr = 0;
module_3_epoch:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=fifo_ref_0_delayed_9_buf inter false
    if (!fifo_ld_0->empty())
    {
      int32_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(&fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int32_t fifo_ref_0_delayed_9 = fifo_ref_0_delayed_9_buf[fifo_ref_0_delayed_9_ptr];;
      const int32_t let_0 = fifo_ref_0_delayed_9;
      WriteData(fifo_st_0, int32_t(let_0), enabled);
      fifo_ref_0_delayed_9_buf[fifo_ref_0_delayed_9_ptr] = fifo_ref_0;
      fifo_ref_0_delayed_9_ptr = fifo_ref_0_delayed_9_ptr < 8 ? ap_uint<4>(fifo_ref_0_delayed_9_ptr+1) : ap_uint<4>(0);
    } // if not empty
  } // for module_3_epoch
} // Module3Func
void Module4Func(
  /*output*/ hls::stream<Data<int32_t>>* fifo_st_0, 
  /* input*/ hls::stream<Data<int32_t>>* fifo_ld_0, 
  /* input*/ hls::stream<Data<int32_t>>* fifo_ld_1, 
  /* input*/ hls::stream<Data<int32_t>>* fifo_ld_2)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_ld_0
#pragma HLS data_pack variable = fifo_ld_1
#pragma HLS data_pack variable = fifo_ld_2
module_4_epoch:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II=1
    if (!fifo_ld_0->empty() && !fifo_ld_1->empty() && !fifo_ld_2->empty())
    {
      int32_t fifo_ref_0;
      int32_t fifo_ref_1;
      int32_t fifo_ref_2;
      const bool fifo_ref_0_enable = ReadData(&fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(&fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(&fifo_ref_2, fifo_ld_2);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable;
      enable = enabled;
      WriteData(fifo_st_0, int32_t(static_cast<int32_t >(static_cast<ap_int<34> >(static_cast<ap_int<33> >(fifo_ref_0) + static_cast<ap_int<33> >(fifo_ref_1)) + static_cast<ap_int<34> >(fifo_ref_2))), enabled);
    } // if not empty
  } // for module_4_epoch
} // Module4Func
void Module5Func(
  /*output*/ hls::stream<Data<int32_t>>* fifo_st_0, 
  /*output*/ hls::stream<Data<int32_t>>* fifo_st_1, 
  /* input*/ hls::stream<Data<int32_t>>* fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_ld_0
  int32_t fifo_ref_0_delayed_1_buf[1];
  ap_uint<1> fifo_ref_0_delayed_1_ptr = 0;
module_5_epoch:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=fifo_ref_0_delayed_1_buf inter false
    if (!fifo_ld_0->empty())
    {
      int32_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(&fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int32_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[fifo_ref_0_delayed_1_ptr];;
      const int32_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, int32_t(let_0), enabled);
      WriteData(fifo_st_1, int32_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[fifo_ref_0_delayed_1_ptr] = fifo_ref_0;
      fifo_ref_0_delayed_1_ptr = fifo_ref_0_delayed_1_ptr < 0 ? ap_uint<1>(fifo_ref_0_delayed_1_ptr+1) : ap_uint<1>(0);
    } // if not empty
  } // for module_5_epoch
} // Module5Func
void Module6Func(
  /*output*/ hls::stream<Data<int32_t>>* fifo_st_0, 
  /* input*/ hls::stream<Data<int32_t>>* fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_ld_0
  int32_t fifo_ref_0_delayed_1_buf[1];
  ap_uint<1> fifo_ref_0_delayed_1_ptr = 0;
module_6_epoch:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II=1
#pragma HLS dependence variable=fifo_ref_0_delayed_1_buf inter false
    if (!fifo_ld_0->empty())
    {
      int32_t fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(&fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const int32_t fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[fifo_ref_0_delayed_1_ptr];;
      const int32_t let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, int32_t(let_0), enabled);
      fifo_ref_0_delayed_1_buf[fifo_ref_0_delayed_1_ptr] = fifo_ref_0;
      fifo_ref_0_delayed_1_ptr = fifo_ref_0_delayed_1_ptr < 0 ? ap_uint<1>(fifo_ref_0_delayed_1_ptr+1) : ap_uint<1>(0);
    } // if not empty
  } // for module_6_epoch
} // Module6Func
void Module7Func(
  /*output*/ hls::stream<Data<ap_uint<512>>>* dram_C_bank_0_fifo, 
  /* input*/ hls::stream<Data<int32_t>>* fifo_ld_0)
{
#pragma HLS data_pack variable = dram_C_bank_0_fifo
#pragma HLS data_pack variable = fifo_ld_0
module_7_epoch:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II=16
    if (!fifo_ld_0->empty())
    {
      int32_t fifo_ref_0;
      ap_uint<512> dram_C_bank_0_buf;
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(31, 0) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(63, 32) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(95, 64) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(127, 96) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(159, 128) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(191, 160) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(223, 192) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(255, 224) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(287, 256) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(319, 288) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(351, 320) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(383, 352) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(415, 384) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(447, 416) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(&fifo_ref_0, fifo_ld_0);
      dram_C_bank_0_buf(479, 448) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      const bool fifo_ref_0_enable = ReadData(&fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      dram_C_bank_0_buf(511, 480) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      WriteData(dram_C_bank_0_fifo, dram_C_bank_0_buf, enabled);
    } // if not empty
  } // for module_7_epoch
} // Module7Func
extern "C"
{

void soda_A_C_kernel(
  ap_uint<512>* bank_0_C,
  ap_uint<512>* bank_0_A,
  uint64_t coalesced_data_num)
{
#pragma HLS interface m_axi port=bank_0_C offset=slave depth=65536 bundle=C_bank_0
#pragma HLS interface m_axi port=bank_0_A offset=slave depth=65536 bundle=A_bank_0

#pragma HLS interface s_axilite port=bank_0_C bundle=control
#pragma HLS interface s_axilite port=bank_0_A bundle=control
#pragma HLS interface s_axilite port=coalesced_data_num bundle=control
#pragma HLS interface s_axilite port=return bundle=control

  hls::stream<Data<ap_uint<512>>> bank_0_A_buf("bank_0_A_buf");
#pragma HLS stream variable=bank_0_A_buf depth=32
#pragma HLS data_pack variable=bank_0_A_buf
  hls::stream<Data<ap_uint<512>>> bank_0_C_buf("bank_0_C_buf");
#pragma HLS stream variable=bank_0_C_buf depth=32
#pragma HLS data_pack variable=bank_0_C_buf

  hls::stream<Data<int32_t>> from_super_source_to_A_offset_0("from_super_source_to_A_offset_0");
#pragma HLS stream variable=from_super_source_to_A_offset_0 depth=16
#pragma HLS data_pack variable=from_super_source_to_A_offset_0
  hls::stream<Data<int32_t>> from_A_offset_0_to_A_offset_9("from_A_offset_0_to_A_offset_9");
#pragma HLS stream variable=from_A_offset_0_to_A_offset_9 depth=16
#pragma HLS data_pack variable=from_A_offset_0_to_A_offset_9
  hls::stream<Data<int32_t>> from_A_offset_0_to_B_pe_0("from_A_offset_0_to_B_pe_0");
#pragma HLS stream variable=from_A_offset_0_to_B_pe_0 depth=16
#pragma HLS data_pack variable=from_A_offset_0_to_B_pe_0
  hls::stream<Data<int32_t>> from_A_offset_9_to_A_offset_18("from_A_offset_9_to_A_offset_18");
#pragma HLS stream variable=from_A_offset_9_to_A_offset_18 depth=16
#pragma HLS data_pack variable=from_A_offset_9_to_A_offset_18
  hls::stream<Data<int32_t>> from_A_offset_9_to_B_pe_0("from_A_offset_9_to_B_pe_0");
#pragma HLS stream variable=from_A_offset_9_to_B_pe_0 depth=16
#pragma HLS data_pack variable=from_A_offset_9_to_B_pe_0
  hls::stream<Data<int32_t>> from_A_offset_18_to_B_pe_0("from_A_offset_18_to_B_pe_0");
#pragma HLS stream variable=from_A_offset_18_to_B_pe_0 depth=16
#pragma HLS data_pack variable=from_A_offset_18_to_B_pe_0
  hls::stream<Data<int32_t>> from_B_pe_0_to_B_offset_0("from_B_pe_0_to_B_offset_0");
#pragma HLS stream variable=from_B_pe_0_to_B_offset_0 depth=16
#pragma HLS data_pack variable=from_B_pe_0_to_B_offset_0
  hls::stream<Data<int32_t>> from_B_offset_0_to_B_offset_1("from_B_offset_0_to_B_offset_1");
#pragma HLS stream variable=from_B_offset_0_to_B_offset_1 depth=16
#pragma HLS data_pack variable=from_B_offset_0_to_B_offset_1
  hls::stream<Data<int32_t>> from_B_offset_0_to_C_pe_0("from_B_offset_0_to_C_pe_0");
#pragma HLS stream variable=from_B_offset_0_to_C_pe_0 depth=16
#pragma HLS data_pack variable=from_B_offset_0_to_C_pe_0
  hls::stream<Data<int32_t>> from_B_offset_1_to_B_offset_2("from_B_offset_1_to_B_offset_2");
#pragma HLS stream variable=from_B_offset_1_to_B_offset_2 depth=16
#pragma HLS data_pack variable=from_B_offset_1_to_B_offset_2
  hls::stream<Data<int32_t>> from_B_offset_1_to_C_pe_0("from_B_offset_1_to_C_pe_0");
#pragma HLS stream variable=from_B_offset_1_to_C_pe_0 depth=16
#pragma HLS data_pack variable=from_B_offset_1_to_C_pe_0
  hls::stream<Data<int32_t>> from_B_offset_2_to_C_pe_0("from_B_offset_2_to_C_pe_0");
#pragma HLS stream variable=from_B_offset_2_to_C_pe_0 depth=16
#pragma HLS data_pack variable=from_B_offset_2_to_C_pe_0
  hls::stream<Data<int32_t>> from_C_pe_0_to_super_sink("from_C_pe_0_to_super_sink");
#pragma HLS stream variable=from_C_pe_0_to_super_sink depth=16
#pragma HLS data_pack variable=from_C_pe_0_to_super_sink

#pragma HLS dataflow
  BurstRead(&bank_0_A_buf, bank_0_A, coalesced_data_num);
  Module0Func(
    /*output*/ &from_super_source_to_A_offset_0, 
    /* input*/ &bank_0_A_buf);
  Module1Func(
    /*output*/ &from_A_offset_0_to_A_offset_9, 
    /*output*/ &from_A_offset_0_to_B_pe_0, 
    /* input*/ &from_super_source_to_A_offset_0);
  Module2Func(
    /*output*/ &from_A_offset_9_to_A_offset_18, 
    /*output*/ &from_A_offset_9_to_B_pe_0, 
    /* input*/ &from_A_offset_0_to_A_offset_9);
  Module3Func(
    /*output*/ &from_A_offset_18_to_B_pe_0, 
    /* input*/ &from_A_offset_9_to_A_offset_18);
  Module4Func(
    /*output*/ &from_B_pe_0_to_B_offset_0, 
    /* input*/ &from_A_offset_18_to_B_pe_0, 
    /* input*/ &from_A_offset_9_to_B_pe_0, 
    /* input*/ &from_A_offset_0_to_B_pe_0);
  Module1Func(
    /*output*/ &from_B_offset_0_to_B_offset_1, 
    /*output*/ &from_B_offset_0_to_C_pe_0, 
    /* input*/ &from_B_pe_0_to_B_offset_0);
  Module5Func(
    /*output*/ &from_B_offset_1_to_B_offset_2, 
    /*output*/ &from_B_offset_1_to_C_pe_0, 
    /* input*/ &from_B_offset_0_to_B_offset_1);
  Module6Func(
    /*output*/ &from_B_offset_2_to_C_pe_0, 
    /* input*/ &from_B_offset_1_to_B_offset_2);
  Module4Func(
    /*output*/ &from_C_pe_0_to_super_sink, 
    /* input*/ &from_B_offset_2_to_C_pe_0, 
    /* input*/ &from_B_offset_1_to_C_pe_0, 
    /* input*/ &from_B_offset_0_to_C_pe_0);
  Module7Func(
    /*output*/ &bank_0_C_buf, 
    /* input*/ &from_C_pe_0_to_super_sink);
  BurstWrite(bank_0_C, &bank_0_C_buf, coalesced_data_num);
}

}//extern "C"
