#include <cfloat>
#include <cmath>
#include <cstdbool>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <algorithm>

#include <ap_int.h>
#include <hls_stream.h>

#ifdef __SYNTHESIS__
#warning this file should be used for simulation only
#warning synthesis result may be sub-optimal
#endif  // __SYNTHESIS__

// this file can be generated from the following SODA DSL
/*
kernel: soda_input_output
burst width: 512
iterate: 1
unroll factor: 8
input float32: input(640, *)
output float32:
  float32 reduce_ssa1_s = input(0, 0)
  float32 reduce_ssa2 = (input(1, 0) * 3699.65F) + (reduce_ssa1_s * 4620.30F)
  float32 reduce_ssa3_s = input(2, 0)
  float32 reduce_ssa4 = (input(0, 1) * 3699.65F) + ((reduce_ssa3_s * 1899.46F) + reduce_ssa2)
  float32 reduce_ssa5_s = input(1, 1)
  float32 reduce_ssa6 = (input(2, 1) * 1520.97F) + ((reduce_ssa5_s * 2962.45F) + reduce_ssa4)
  float32 reduce_ssa7_s = input(0, 2)
  float32 reduce_ssa8 = (input(1, 2) * 1520.97F) + ((reduce_ssa7_s * 1899.46F) + reduce_ssa6)
  float32 reduce_ssa9_s = input(2, 2)
  output(0, 0) = (reduce_ssa9_s * 780.892F) + reduce_ssa8
border: None
cluster: None
*/

// stencil window size: (3, 3)
// stencil distace: 1282
// data layout is documented at
// https://github.com/Blaok/soda/blob/master/docs/data-layout.md

template<typename To, typename From>
To Reinterpret(From val) {
#pragma HLS inline
  return reinterpret_cast<To&>(val);
}

template<typename T>
struct Data {
  T data;
  bool ctrl;
};

template<typename T>
bool ReadData(T& data, hls::stream<Data<T>>& from) {
#pragma HLS inline
  const auto tmp = from.read();
  data = tmp.data;
  return tmp.ctrl;
}

template<typename T>
void WriteData(hls::stream<Data<T>>& to, const T& data, bool ctrl) {
#pragma HLS inline
  Data<T> tmp;
  tmp.data = data;
  tmp.ctrl = ctrl;
  to.write(tmp);
}

template <typename T>
void BurstRead(hls::stream<Data<T>>& to, T* from, uint64_t data_num) {
load:
  for (uint64_t i = 0; i < data_num;) {
#pragma HLS pipeline II = 1
    const uint64_t next_i = i + 1;
    WriteData(to, from[i], next_i < data_num);
    i = next_i;
  }
}

template <typename T>
void BurstWrite(T* to, hls::stream<Data<T>>& from, uint64_t data_num) {
store:
  for (uint64_t i = 0; i < data_num; ++i) {
#pragma HLS pipeline II = 1
    T buf;
    ReadData(buf, from);
    to[i] = buf;
  }
}

void Module0Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /*output*/ hls::stream<Data<float>>& fifo_st_4, 
  /*output*/ hls::stream<Data<float>>& fifo_st_5, 
  /*output*/ hls::stream<Data<float>>& fifo_st_6, 
  /*output*/ hls::stream<Data<float>>& fifo_st_7, 
  /* input*/ hls::stream<Data<ap_uint<512>>>& dram_input_bank_0_fifo)
{
#pragma HLS data_pack variable = dram_input_bank_0_fifo
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_st_2
#pragma HLS data_pack variable = fifo_st_3
#pragma HLS data_pack variable = fifo_st_4
#pragma HLS data_pack variable = fifo_st_5
#pragma HLS data_pack variable = fifo_st_6
#pragma HLS data_pack variable = fifo_st_7
module_0:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 2
    if (!dram_input_bank_0_fifo.empty())
    {
      ap_uint<512> dram_input_bank_0_buf;
      const bool dram_input_bank_0_buf_enable = ReadData(dram_input_bank_0_buf, dram_input_bank_0_fifo);
      const bool enabled = dram_input_bank_0_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(191, 160))), true);
      WriteData(fifo_st_1, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(223, 192))), true);
      WriteData(fifo_st_2, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(255, 224))), true);
      WriteData(fifo_st_3, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(159, 128))), true);
      WriteData(fifo_st_4, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(127, 96))), true);
      WriteData(fifo_st_5, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(95, 64))), true);
      WriteData(fifo_st_6, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(63, 32))), true);
      WriteData(fifo_st_7, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(31, 0))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(447, 416))), enabled);
      WriteData(fifo_st_1, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(479, 448))), enabled);
      WriteData(fifo_st_2, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(511, 480))), enabled);
      WriteData(fifo_st_3, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(415, 384))), enabled);
      WriteData(fifo_st_4, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(383, 352))), enabled);
      WriteData(fifo_st_5, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(351, 320))), enabled);
      WriteData(fifo_st_6, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(319, 288))), enabled);
      WriteData(fifo_st_7, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(287, 256))), enabled);
    } // if not empty
  } // for module_0
} // Module0Func

void Module1Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_st_2
#pragma HLS data_pack variable = fifo_st_3
#pragma HLS data_pack variable = fifo_ld_0
module_1:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, float(fifo_ref_0), enabled);
      WriteData(fifo_st_1, float(fifo_ref_0), enabled);
      WriteData(fifo_st_2, float(fifo_ref_0), enabled);
      WriteData(fifo_st_3, float(fifo_ref_0), enabled);
    } // if not empty
  } // for module_1
} // Module1Func

void Module2Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_st_2
#pragma HLS data_pack variable = fifo_ld_0
module_2:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, float(fifo_ref_0), enabled);
      WriteData(fifo_st_1, float(fifo_ref_0), enabled);
      WriteData(fifo_st_2, float(fifo_ref_0), enabled);
    } // if not empty
  } // for module_2
} // Module2Func

void Module3Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_ld_0
module_3:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      WriteData(fifo_st_0, float(fifo_ref_0), enabled);
      WriteData(fifo_st_1, float(fifo_ref_0), enabled);
    } // if not empty
  } // for module_3
} // Module3Func

void Module4Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /*output*/ hls::stream<Data<float>>& fifo_st_3, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_st_2
#pragma HLS data_pack variable = fifo_st_3
#pragma HLS data_pack variable = fifo_ld_0
  float fifo_ref_0_delayed_80_buf[80];
  ap_uint<7> ptr_delay_80 = 0;
module_4:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = fifo_ref_0_delayed_80_buf inter false
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_80 = fifo_ref_0_delayed_80_buf[ptr_delay_80];;
      const float let_0 = fifo_ref_0_delayed_80;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      WriteData(fifo_st_3, float(let_0), enabled);
      fifo_ref_0_delayed_80_buf[ptr_delay_80] = fifo_ref_0;
      ptr_delay_80 = ptr_delay_80 < 79 ? (++ptr_delay_80) : (ptr_delay_80 = 0);
    } // if not empty
  } // for module_4
} // Module4Func

void Module5Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_ld_0
  float fifo_ref_0_delayed_1_buf[1];
  ap_uint<1> ptr_delay_1 = 0;
module_5:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = fifo_ref_0_delayed_1_buf inter false
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const float let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 = ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_5
} // Module5Func

void Module6Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_st_2
#pragma HLS data_pack variable = fifo_ld_0
  float fifo_ref_0_delayed_1_buf[1];
  ap_uint<1> ptr_delay_1 = 0;
module_6:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = fifo_ref_0_delayed_1_buf inter false
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const float let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 = ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_6
} // Module6Func

void Module7Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_st_2
#pragma HLS data_pack variable = fifo_ld_0
  float fifo_ref_0_delayed_80_buf[80];
  ap_uint<7> ptr_delay_80 = 0;
module_7:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = fifo_ref_0_delayed_80_buf inter false
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_80 = fifo_ref_0_delayed_80_buf[ptr_delay_80];;
      const float let_0 = fifo_ref_0_delayed_80;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      fifo_ref_0_delayed_80_buf[ptr_delay_80] = fifo_ref_0;
      ptr_delay_80 = ptr_delay_80 < 79 ? (++ptr_delay_80) : (ptr_delay_80 = 0);
    } // if not empty
  } // for module_7
} // Module7Func

void Module8Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /*output*/ hls::stream<Data<float>>& fifo_st_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_st_2
#pragma HLS data_pack variable = fifo_ld_0
  float fifo_ref_0_delayed_79_buf[79];
  ap_uint<7> ptr_delay_79 = 0;
module_8:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = fifo_ref_0_delayed_79_buf inter false
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_79 = fifo_ref_0_delayed_79_buf[ptr_delay_79];;
      const float let_0 = fifo_ref_0_delayed_79;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      WriteData(fifo_st_2, float(let_0), enabled);
      fifo_ref_0_delayed_79_buf[ptr_delay_79] = fifo_ref_0;
      ptr_delay_79 = ptr_delay_79 < 78 ? (++ptr_delay_79) : (ptr_delay_79 = 0);
    } // if not empty
  } // for module_8
} // Module8Func

void Module9Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_st_1
#pragma HLS data_pack variable = fifo_ld_0
  float fifo_ref_0_delayed_79_buf[79];
  ap_uint<7> ptr_delay_79 = 0;
module_9:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = fifo_ref_0_delayed_79_buf inter false
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_79 = fifo_ref_0_delayed_79_buf[ptr_delay_79];;
      const float let_0 = fifo_ref_0_delayed_79;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_79_buf[ptr_delay_79] = fifo_ref_0;
      ptr_delay_79 = ptr_delay_79 < 78 ? (++ptr_delay_79) : (ptr_delay_79 = 0);
    } // if not empty
  } // for module_9
} // Module9Func

void Module10Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_3, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_4, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_5, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_6, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_7, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_8)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_ld_0
#pragma HLS data_pack variable = fifo_ld_1
#pragma HLS data_pack variable = fifo_ld_2
#pragma HLS data_pack variable = fifo_ld_3
#pragma HLS data_pack variable = fifo_ld_4
#pragma HLS data_pack variable = fifo_ld_5
#pragma HLS data_pack variable = fifo_ld_6
#pragma HLS data_pack variable = fifo_ld_7
#pragma HLS data_pack variable = fifo_ld_8
module_10:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty() && !fifo_ld_4.empty() && !fifo_ld_5.empty() && !fifo_ld_6.empty() && !fifo_ld_7.empty() && !fifo_ld_8.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      float fifo_ref_2;
      float fifo_ref_3;
      float fifo_ref_4;
      float fifo_ref_5;
      float fifo_ref_6;
      float fifo_ref_7;
      float fifo_ref_8;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool fifo_ref_4_enable = ReadData(fifo_ref_4, fifo_ld_4);
      const bool fifo_ref_5_enable = ReadData(fifo_ref_5, fifo_ld_5);
      const bool fifo_ref_6_enable = ReadData(fifo_ref_6, fifo_ld_6);
      const bool fifo_ref_7_enable = ReadData(fifo_ref_7, fifo_ld_7);
      const bool fifo_ref_8_enable = ReadData(fifo_ref_8, fifo_ld_8);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable && fifo_ref_4_enable && fifo_ref_5_enable && fifo_ref_6_enable && fifo_ref_7_enable && fifo_ref_8_enable;
      enable = enabled;
      const float reduce_ssa1_s = fifo_ref_0;
      const float reduce_ssa2 = (fifo_ref_1 * 3699.65F) + (reduce_ssa1_s * 4620.30F);
      const float reduce_ssa3_s = fifo_ref_2;
      const float reduce_ssa4 = (fifo_ref_3 * 3699.65F) + ((reduce_ssa3_s * 1899.46F) + reduce_ssa2);
      const float reduce_ssa5_s = fifo_ref_4;
      const float reduce_ssa6 = (fifo_ref_5 * 1520.97F) + ((reduce_ssa5_s * 2962.45F) + reduce_ssa4);
      const float reduce_ssa7_s = fifo_ref_6;
      const float reduce_ssa8 = (fifo_ref_7 * 1520.97F) + ((reduce_ssa7_s * 1899.46F) + reduce_ssa6);
      const float reduce_ssa9_s = fifo_ref_8;
      WriteData(fifo_st_0, float(((reduce_ssa9_s * 780.892F) + reduce_ssa8)), enabled);
    } // if not empty
  } // for module_10
} // Module10Func

void Module11Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
#pragma HLS data_pack variable = fifo_st_0
#pragma HLS data_pack variable = fifo_ld_0
  float fifo_ref_0_delayed_1_buf[1];
  ap_uint<1> ptr_delay_1 = 0;
module_11:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = fifo_ref_0_delayed_1_buf inter false
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[ptr_delay_1];;
      const float let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, float(let_0), enabled);
      fifo_ref_0_delayed_1_buf[ptr_delay_1] = fifo_ref_0;
      ptr_delay_1 = ptr_delay_1 < 0 ? (++ptr_delay_1) : (ptr_delay_1 = 0);
    } // if not empty
  } // for module_11
} // Module11Func

void Module12Func(
  /*output*/ hls::stream<Data<ap_uint<512>>>& dram_output_bank_0_fifo, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_3, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_4, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_5, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_6, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_7)
{
#pragma HLS data_pack variable = dram_output_bank_0_fifo
#pragma HLS data_pack variable = fifo_ld_0
#pragma HLS data_pack variable = fifo_ld_1
#pragma HLS data_pack variable = fifo_ld_2
#pragma HLS data_pack variable = fifo_ld_3
#pragma HLS data_pack variable = fifo_ld_4
#pragma HLS data_pack variable = fifo_ld_5
#pragma HLS data_pack variable = fifo_ld_6
#pragma HLS data_pack variable = fifo_ld_7
module_12:
  for (bool enable = true; enable;)
  {
#pragma HLS pipeline II = 2
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty() && !fifo_ld_4.empty() && !fifo_ld_5.empty() && !fifo_ld_6.empty() && !fifo_ld_7.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      float fifo_ref_2;
      float fifo_ref_3;
      float fifo_ref_4;
      float fifo_ref_5;
      float fifo_ref_6;
      float fifo_ref_7;
      ap_uint<512> dram_output_bank_0_buf;
      ReadData(fifo_ref_0, fifo_ld_0);
      ReadData(fifo_ref_1, fifo_ld_1);
      ReadData(fifo_ref_2, fifo_ld_2);
      ReadData(fifo_ref_3, fifo_ld_3);
      ReadData(fifo_ref_4, fifo_ld_4);
      ReadData(fifo_ref_5, fifo_ld_5);
      ReadData(fifo_ref_6, fifo_ld_6);
      ReadData(fifo_ref_7, fifo_ld_7);
      dram_output_bank_0_buf(191, 160) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      dram_output_bank_0_buf(159, 128) = Reinterpret<ap_uint<32>>(fifo_ref_1);
      dram_output_bank_0_buf(127, 96) = Reinterpret<ap_uint<32>>(fifo_ref_2);
      dram_output_bank_0_buf(95, 64) = Reinterpret<ap_uint<32>>(fifo_ref_3);
      dram_output_bank_0_buf(223, 192) = Reinterpret<ap_uint<32>>(fifo_ref_4);
      dram_output_bank_0_buf(255, 224) = Reinterpret<ap_uint<32>>(fifo_ref_5);
      dram_output_bank_0_buf(63, 32) = Reinterpret<ap_uint<32>>(fifo_ref_6);
      dram_output_bank_0_buf(31, 0) = Reinterpret<ap_uint<32>>(fifo_ref_7);
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool fifo_ref_4_enable = ReadData(fifo_ref_4, fifo_ld_4);
      const bool fifo_ref_5_enable = ReadData(fifo_ref_5, fifo_ld_5);
      const bool fifo_ref_6_enable = ReadData(fifo_ref_6, fifo_ld_6);
      const bool fifo_ref_7_enable = ReadData(fifo_ref_7, fifo_ld_7);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable && fifo_ref_4_enable && fifo_ref_5_enable && fifo_ref_6_enable && fifo_ref_7_enable;
      enable = enabled;
      dram_output_bank_0_buf(447, 416) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      dram_output_bank_0_buf(415, 384) = Reinterpret<ap_uint<32>>(fifo_ref_1);
      dram_output_bank_0_buf(383, 352) = Reinterpret<ap_uint<32>>(fifo_ref_2);
      dram_output_bank_0_buf(351, 320) = Reinterpret<ap_uint<32>>(fifo_ref_3);
      dram_output_bank_0_buf(479, 448) = Reinterpret<ap_uint<32>>(fifo_ref_4);
      dram_output_bank_0_buf(511, 480) = Reinterpret<ap_uint<32>>(fifo_ref_5);
      dram_output_bank_0_buf(319, 288) = Reinterpret<ap_uint<32>>(fifo_ref_6);
      dram_output_bank_0_buf(287, 256) = Reinterpret<ap_uint<32>>(fifo_ref_7);
      WriteData(dram_output_bank_0_fifo, dram_output_bank_0_buf, enabled);
    } // if not empty
  } // for module_12
} // Module12Func

extern "C" {

void soda_input_output_kernel(
  ap_uint<512>* bank_0_output, 
  ap_uint<512>* bank_0_input, 
  uint64_t coalesced_data_num)
{
#pragma HLS interface m_axi port = bank_0_output offset = slave bundle = output_bank_0
#pragma HLS interface m_axi port = bank_0_input offset = slave bundle = input_bank_0
#pragma HLS interface s_axilite port = bank_0_output bundle = control
#pragma HLS interface s_axilite port = bank_0_input bundle = control
#pragma HLS interface s_axilite port = coalesced_data_num bundle = control
#pragma HLS interface s_axilite port = return bundle = control

  hls::stream<Data<ap_uint<512>>> bank_0_input_buf("bank_0_input_buf");
#pragma HLS stream variable = bank_0_input_buf depth = 32
#pragma HLS data_pack variable = bank_0_input_buf
  hls::stream<Data<ap_uint<512>>> bank_0_output_buf("bank_0_output_buf");
#pragma HLS stream variable = bank_0_output_buf depth = 32
#pragma HLS data_pack variable = bank_0_output_buf

  hls::stream<Data<float>> from_input_bank_0_to_input_offset_2("from_input_bank_0_to_input_offset_2");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_2 depth = 7
#pragma HLS data_pack variable = from_input_bank_0_to_input_offset_2
  hls::stream<Data<float>> from_input_bank_0_to_input_offset_1("from_input_bank_0_to_input_offset_1");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_1 depth = 3
#pragma HLS data_pack variable = from_input_bank_0_to_input_offset_1
  hls::stream<Data<float>> from_input_bank_0_to_input_offset_0("from_input_bank_0_to_input_offset_0");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_0 depth = 3
#pragma HLS data_pack variable = from_input_bank_0_to_input_offset_0
  hls::stream<Data<float>> from_input_bank_0_to_input_offset_3("from_input_bank_0_to_input_offset_3");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_3 depth = 7
#pragma HLS data_pack variable = from_input_bank_0_to_input_offset_3
  hls::stream<Data<float>> from_input_bank_0_to_input_offset_4("from_input_bank_0_to_input_offset_4");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_4 depth = 7
#pragma HLS data_pack variable = from_input_bank_0_to_input_offset_4
  hls::stream<Data<float>> from_input_bank_0_to_input_offset_5("from_input_bank_0_to_input_offset_5");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_5 depth = 7
#pragma HLS data_pack variable = from_input_bank_0_to_input_offset_5
  hls::stream<Data<float>> from_input_bank_0_to_input_offset_6("from_input_bank_0_to_input_offset_6");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_6 depth = 7
#pragma HLS data_pack variable = from_input_bank_0_to_input_offset_6
  hls::stream<Data<float>> from_input_bank_0_to_input_offset_7("from_input_bank_0_to_input_offset_7");
#pragma HLS stream variable = from_input_bank_0_to_input_offset_7 depth = 7
#pragma HLS data_pack variable = from_input_bank_0_to_input_offset_7
  hls::stream<Data<float>> from_input_offset_2_to_input_offset_642("from_input_offset_2_to_input_offset_642");
#pragma HLS stream variable = from_input_offset_2_to_input_offset_642 depth = 3
#pragma HLS data_pack variable = from_input_offset_2_to_input_offset_642
  hls::stream<Data<float>> from_input_offset_2_to_output_pe_7("from_input_offset_2_to_output_pe_7");
#pragma HLS stream variable = from_input_offset_2_to_output_pe_7 depth = 7
#pragma HLS data_pack variable = from_input_offset_2_to_output_pe_7
  hls::stream<Data<float>> from_input_offset_2_to_output_pe_6("from_input_offset_2_to_output_pe_6");
#pragma HLS stream variable = from_input_offset_2_to_output_pe_6 depth = 7
#pragma HLS data_pack variable = from_input_offset_2_to_output_pe_6
  hls::stream<Data<float>> from_input_offset_2_to_output_pe_5("from_input_offset_2_to_output_pe_5");
#pragma HLS stream variable = from_input_offset_2_to_output_pe_5 depth = 7
#pragma HLS data_pack variable = from_input_offset_2_to_output_pe_5
  hls::stream<Data<float>> from_input_offset_1_to_input_offset_9("from_input_offset_1_to_input_offset_9");
#pragma HLS stream variable = from_input_offset_1_to_input_offset_9 depth = 3
#pragma HLS data_pack variable = from_input_offset_1_to_input_offset_9
  hls::stream<Data<float>> from_input_offset_1_to_output_pe_7("from_input_offset_1_to_output_pe_7");
#pragma HLS stream variable = from_input_offset_1_to_output_pe_7 depth = 11
#pragma HLS data_pack variable = from_input_offset_1_to_output_pe_7
  hls::stream<Data<float>> from_input_offset_1_to_output_pe_6("from_input_offset_1_to_output_pe_6");
#pragma HLS stream variable = from_input_offset_1_to_output_pe_6 depth = 11
#pragma HLS data_pack variable = from_input_offset_1_to_output_pe_6
  hls::stream<Data<float>> from_input_offset_0_to_input_offset_8("from_input_offset_0_to_input_offset_8");
#pragma HLS stream variable = from_input_offset_0_to_input_offset_8 depth = 3
#pragma HLS data_pack variable = from_input_offset_0_to_input_offset_8
  hls::stream<Data<float>> from_input_offset_0_to_output_pe_7("from_input_offset_0_to_output_pe_7");
#pragma HLS stream variable = from_input_offset_0_to_output_pe_7 depth = 11
#pragma HLS data_pack variable = from_input_offset_0_to_output_pe_7
  hls::stream<Data<float>> from_input_offset_3_to_input_offset_643("from_input_offset_3_to_input_offset_643");
#pragma HLS stream variable = from_input_offset_3_to_input_offset_643 depth = 3
#pragma HLS data_pack variable = from_input_offset_3_to_input_offset_643
  hls::stream<Data<float>> from_input_offset_3_to_output_pe_6("from_input_offset_3_to_output_pe_6");
#pragma HLS stream variable = from_input_offset_3_to_output_pe_6 depth = 7
#pragma HLS data_pack variable = from_input_offset_3_to_output_pe_6
  hls::stream<Data<float>> from_input_offset_3_to_output_pe_5("from_input_offset_3_to_output_pe_5");
#pragma HLS stream variable = from_input_offset_3_to_output_pe_5 depth = 7
#pragma HLS data_pack variable = from_input_offset_3_to_output_pe_5
  hls::stream<Data<float>> from_input_offset_3_to_output_pe_4("from_input_offset_3_to_output_pe_4");
#pragma HLS stream variable = from_input_offset_3_to_output_pe_4 depth = 7
#pragma HLS data_pack variable = from_input_offset_3_to_output_pe_4
  hls::stream<Data<float>> from_input_offset_4_to_input_offset_644("from_input_offset_4_to_input_offset_644");
#pragma HLS stream variable = from_input_offset_4_to_input_offset_644 depth = 3
#pragma HLS data_pack variable = from_input_offset_4_to_input_offset_644
  hls::stream<Data<float>> from_input_offset_4_to_output_pe_5("from_input_offset_4_to_output_pe_5");
#pragma HLS stream variable = from_input_offset_4_to_output_pe_5 depth = 7
#pragma HLS data_pack variable = from_input_offset_4_to_output_pe_5
  hls::stream<Data<float>> from_input_offset_4_to_output_pe_4("from_input_offset_4_to_output_pe_4");
#pragma HLS stream variable = from_input_offset_4_to_output_pe_4 depth = 7
#pragma HLS data_pack variable = from_input_offset_4_to_output_pe_4
  hls::stream<Data<float>> from_input_offset_4_to_output_pe_3("from_input_offset_4_to_output_pe_3");
#pragma HLS stream variable = from_input_offset_4_to_output_pe_3 depth = 7
#pragma HLS data_pack variable = from_input_offset_4_to_output_pe_3
  hls::stream<Data<float>> from_input_offset_5_to_input_offset_645("from_input_offset_5_to_input_offset_645");
#pragma HLS stream variable = from_input_offset_5_to_input_offset_645 depth = 3
#pragma HLS data_pack variable = from_input_offset_5_to_input_offset_645
  hls::stream<Data<float>> from_input_offset_5_to_output_pe_4("from_input_offset_5_to_output_pe_4");
#pragma HLS stream variable = from_input_offset_5_to_output_pe_4 depth = 7
#pragma HLS data_pack variable = from_input_offset_5_to_output_pe_4
  hls::stream<Data<float>> from_input_offset_5_to_output_pe_3("from_input_offset_5_to_output_pe_3");
#pragma HLS stream variable = from_input_offset_5_to_output_pe_3 depth = 7
#pragma HLS data_pack variable = from_input_offset_5_to_output_pe_3
  hls::stream<Data<float>> from_input_offset_5_to_output_pe_2("from_input_offset_5_to_output_pe_2");
#pragma HLS stream variable = from_input_offset_5_to_output_pe_2 depth = 7
#pragma HLS data_pack variable = from_input_offset_5_to_output_pe_2
  hls::stream<Data<float>> from_input_offset_6_to_input_offset_646("from_input_offset_6_to_input_offset_646");
#pragma HLS stream variable = from_input_offset_6_to_input_offset_646 depth = 3
#pragma HLS data_pack variable = from_input_offset_6_to_input_offset_646
  hls::stream<Data<float>> from_input_offset_6_to_output_pe_3("from_input_offset_6_to_output_pe_3");
#pragma HLS stream variable = from_input_offset_6_to_output_pe_3 depth = 7
#pragma HLS data_pack variable = from_input_offset_6_to_output_pe_3
  hls::stream<Data<float>> from_input_offset_6_to_output_pe_2("from_input_offset_6_to_output_pe_2");
#pragma HLS stream variable = from_input_offset_6_to_output_pe_2 depth = 7
#pragma HLS data_pack variable = from_input_offset_6_to_output_pe_2
  hls::stream<Data<float>> from_input_offset_6_to_output_pe_1("from_input_offset_6_to_output_pe_1");
#pragma HLS stream variable = from_input_offset_6_to_output_pe_1 depth = 9
#pragma HLS data_pack variable = from_input_offset_6_to_output_pe_1
  hls::stream<Data<float>> from_input_offset_7_to_input_offset_647("from_input_offset_7_to_input_offset_647");
#pragma HLS stream variable = from_input_offset_7_to_input_offset_647 depth = 3
#pragma HLS data_pack variable = from_input_offset_7_to_input_offset_647
  hls::stream<Data<float>> from_input_offset_7_to_output_pe_2("from_input_offset_7_to_output_pe_2");
#pragma HLS stream variable = from_input_offset_7_to_output_pe_2 depth = 7
#pragma HLS data_pack variable = from_input_offset_7_to_output_pe_2
  hls::stream<Data<float>> from_input_offset_7_to_output_pe_1("from_input_offset_7_to_output_pe_1");
#pragma HLS stream variable = from_input_offset_7_to_output_pe_1 depth = 9
#pragma HLS data_pack variable = from_input_offset_7_to_output_pe_1
  hls::stream<Data<float>> from_input_offset_7_to_output_pe_0("from_input_offset_7_to_output_pe_0");
#pragma HLS stream variable = from_input_offset_7_to_output_pe_0 depth = 9
#pragma HLS data_pack variable = from_input_offset_7_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_642_to_input_offset_1282("from_input_offset_642_to_input_offset_1282");
#pragma HLS stream variable = from_input_offset_642_to_input_offset_1282 depth = 3
#pragma HLS data_pack variable = from_input_offset_642_to_input_offset_1282
  hls::stream<Data<float>> from_input_offset_642_to_output_pe_7("from_input_offset_642_to_output_pe_7");
#pragma HLS stream variable = from_input_offset_642_to_output_pe_7 depth = 5
#pragma HLS data_pack variable = from_input_offset_642_to_output_pe_7
  hls::stream<Data<float>> from_input_offset_642_to_output_pe_6("from_input_offset_642_to_output_pe_6");
#pragma HLS stream variable = from_input_offset_642_to_output_pe_6 depth = 5
#pragma HLS data_pack variable = from_input_offset_642_to_output_pe_6
  hls::stream<Data<float>> from_input_offset_642_to_output_pe_5("from_input_offset_642_to_output_pe_5");
#pragma HLS stream variable = from_input_offset_642_to_output_pe_5 depth = 5
#pragma HLS data_pack variable = from_input_offset_642_to_output_pe_5
  hls::stream<Data<float>> from_input_offset_9_to_input_offset_641("from_input_offset_9_to_input_offset_641");
#pragma HLS stream variable = from_input_offset_9_to_input_offset_641 depth = 3
#pragma HLS data_pack variable = from_input_offset_9_to_input_offset_641
  hls::stream<Data<float>> from_input_offset_9_to_output_pe_0("from_input_offset_9_to_output_pe_0");
#pragma HLS stream variable = from_input_offset_9_to_output_pe_0 depth = 11
#pragma HLS data_pack variable = from_input_offset_9_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_8_to_input_offset_640("from_input_offset_8_to_input_offset_640");
#pragma HLS stream variable = from_input_offset_8_to_input_offset_640 depth = 3
#pragma HLS data_pack variable = from_input_offset_8_to_input_offset_640
  hls::stream<Data<float>> from_input_offset_8_to_output_pe_1("from_input_offset_8_to_output_pe_1");
#pragma HLS stream variable = from_input_offset_8_to_output_pe_1 depth = 11
#pragma HLS data_pack variable = from_input_offset_8_to_output_pe_1
  hls::stream<Data<float>> from_input_offset_8_to_output_pe_0("from_input_offset_8_to_output_pe_0");
#pragma HLS stream variable = from_input_offset_8_to_output_pe_0 depth = 11
#pragma HLS data_pack variable = from_input_offset_8_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_643_to_input_offset_1283("from_input_offset_643_to_input_offset_1283");
#pragma HLS stream variable = from_input_offset_643_to_input_offset_1283 depth = 3
#pragma HLS data_pack variable = from_input_offset_643_to_input_offset_1283
  hls::stream<Data<float>> from_input_offset_643_to_output_pe_6("from_input_offset_643_to_output_pe_6");
#pragma HLS stream variable = from_input_offset_643_to_output_pe_6 depth = 5
#pragma HLS data_pack variable = from_input_offset_643_to_output_pe_6
  hls::stream<Data<float>> from_input_offset_643_to_output_pe_5("from_input_offset_643_to_output_pe_5");
#pragma HLS stream variable = from_input_offset_643_to_output_pe_5 depth = 5
#pragma HLS data_pack variable = from_input_offset_643_to_output_pe_5
  hls::stream<Data<float>> from_input_offset_643_to_output_pe_4("from_input_offset_643_to_output_pe_4");
#pragma HLS stream variable = from_input_offset_643_to_output_pe_4 depth = 5
#pragma HLS data_pack variable = from_input_offset_643_to_output_pe_4
  hls::stream<Data<float>> from_input_offset_644_to_input_offset_1284("from_input_offset_644_to_input_offset_1284");
#pragma HLS stream variable = from_input_offset_644_to_input_offset_1284 depth = 3
#pragma HLS data_pack variable = from_input_offset_644_to_input_offset_1284
  hls::stream<Data<float>> from_input_offset_644_to_output_pe_5("from_input_offset_644_to_output_pe_5");
#pragma HLS stream variable = from_input_offset_644_to_output_pe_5 depth = 5
#pragma HLS data_pack variable = from_input_offset_644_to_output_pe_5
  hls::stream<Data<float>> from_input_offset_644_to_output_pe_4("from_input_offset_644_to_output_pe_4");
#pragma HLS stream variable = from_input_offset_644_to_output_pe_4 depth = 5
#pragma HLS data_pack variable = from_input_offset_644_to_output_pe_4
  hls::stream<Data<float>> from_input_offset_644_to_output_pe_3("from_input_offset_644_to_output_pe_3");
#pragma HLS stream variable = from_input_offset_644_to_output_pe_3 depth = 5
#pragma HLS data_pack variable = from_input_offset_644_to_output_pe_3
  hls::stream<Data<float>> from_input_offset_645_to_input_offset_1285("from_input_offset_645_to_input_offset_1285");
#pragma HLS stream variable = from_input_offset_645_to_input_offset_1285 depth = 3
#pragma HLS data_pack variable = from_input_offset_645_to_input_offset_1285
  hls::stream<Data<float>> from_input_offset_645_to_output_pe_4("from_input_offset_645_to_output_pe_4");
#pragma HLS stream variable = from_input_offset_645_to_output_pe_4 depth = 5
#pragma HLS data_pack variable = from_input_offset_645_to_output_pe_4
  hls::stream<Data<float>> from_input_offset_645_to_output_pe_3("from_input_offset_645_to_output_pe_3");
#pragma HLS stream variable = from_input_offset_645_to_output_pe_3 depth = 5
#pragma HLS data_pack variable = from_input_offset_645_to_output_pe_3
  hls::stream<Data<float>> from_input_offset_645_to_output_pe_2("from_input_offset_645_to_output_pe_2");
#pragma HLS stream variable = from_input_offset_645_to_output_pe_2 depth = 5
#pragma HLS data_pack variable = from_input_offset_645_to_output_pe_2
  hls::stream<Data<float>> from_input_offset_646_to_input_offset_1286("from_input_offset_646_to_input_offset_1286");
#pragma HLS stream variable = from_input_offset_646_to_input_offset_1286 depth = 3
#pragma HLS data_pack variable = from_input_offset_646_to_input_offset_1286
  hls::stream<Data<float>> from_input_offset_646_to_output_pe_3("from_input_offset_646_to_output_pe_3");
#pragma HLS stream variable = from_input_offset_646_to_output_pe_3 depth = 5
#pragma HLS data_pack variable = from_input_offset_646_to_output_pe_3
  hls::stream<Data<float>> from_input_offset_646_to_output_pe_2("from_input_offset_646_to_output_pe_2");
#pragma HLS stream variable = from_input_offset_646_to_output_pe_2 depth = 5
#pragma HLS data_pack variable = from_input_offset_646_to_output_pe_2
  hls::stream<Data<float>> from_input_offset_646_to_output_pe_1("from_input_offset_646_to_output_pe_1");
#pragma HLS stream variable = from_input_offset_646_to_output_pe_1 depth = 7
#pragma HLS data_pack variable = from_input_offset_646_to_output_pe_1
  hls::stream<Data<float>> from_input_offset_647_to_input_offset_1287("from_input_offset_647_to_input_offset_1287");
#pragma HLS stream variable = from_input_offset_647_to_input_offset_1287 depth = 3
#pragma HLS data_pack variable = from_input_offset_647_to_input_offset_1287
  hls::stream<Data<float>> from_input_offset_647_to_output_pe_2("from_input_offset_647_to_output_pe_2");
#pragma HLS stream variable = from_input_offset_647_to_output_pe_2 depth = 5
#pragma HLS data_pack variable = from_input_offset_647_to_output_pe_2
  hls::stream<Data<float>> from_input_offset_647_to_output_pe_1("from_input_offset_647_to_output_pe_1");
#pragma HLS stream variable = from_input_offset_647_to_output_pe_1 depth = 7
#pragma HLS data_pack variable = from_input_offset_647_to_output_pe_1
  hls::stream<Data<float>> from_input_offset_647_to_output_pe_0("from_input_offset_647_to_output_pe_0");
#pragma HLS stream variable = from_input_offset_647_to_output_pe_0 depth = 7
#pragma HLS data_pack variable = from_input_offset_647_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_1282_to_output_pe_7("from_input_offset_1282_to_output_pe_7");
#pragma HLS stream variable = from_input_offset_1282_to_output_pe_7 depth = 3
#pragma HLS data_pack variable = from_input_offset_1282_to_output_pe_7
  hls::stream<Data<float>> from_input_offset_1282_to_output_pe_6("from_input_offset_1282_to_output_pe_6");
#pragma HLS stream variable = from_input_offset_1282_to_output_pe_6 depth = 3
#pragma HLS data_pack variable = from_input_offset_1282_to_output_pe_6
  hls::stream<Data<float>> from_input_offset_1282_to_output_pe_5("from_input_offset_1282_to_output_pe_5");
#pragma HLS stream variable = from_input_offset_1282_to_output_pe_5 depth = 3
#pragma HLS data_pack variable = from_input_offset_1282_to_output_pe_5
  hls::stream<Data<float>> from_input_offset_641_to_input_offset_649("from_input_offset_641_to_input_offset_649");
#pragma HLS stream variable = from_input_offset_641_to_input_offset_649 depth = 3
#pragma HLS data_pack variable = from_input_offset_641_to_input_offset_649
  hls::stream<Data<float>> from_input_offset_641_to_output_pe_7("from_input_offset_641_to_output_pe_7");
#pragma HLS stream variable = from_input_offset_641_to_output_pe_7 depth = 7
#pragma HLS data_pack variable = from_input_offset_641_to_output_pe_7
  hls::stream<Data<float>> from_input_offset_641_to_output_pe_6("from_input_offset_641_to_output_pe_6");
#pragma HLS stream variable = from_input_offset_641_to_output_pe_6 depth = 7
#pragma HLS data_pack variable = from_input_offset_641_to_output_pe_6
  hls::stream<Data<float>> from_input_offset_640_to_input_offset_648("from_input_offset_640_to_input_offset_648");
#pragma HLS stream variable = from_input_offset_640_to_input_offset_648 depth = 3
#pragma HLS data_pack variable = from_input_offset_640_to_input_offset_648
  hls::stream<Data<float>> from_input_offset_640_to_output_pe_7("from_input_offset_640_to_output_pe_7");
#pragma HLS stream variable = from_input_offset_640_to_output_pe_7 depth = 7
#pragma HLS data_pack variable = from_input_offset_640_to_output_pe_7
  hls::stream<Data<float>> from_input_offset_1283_to_output_pe_6("from_input_offset_1283_to_output_pe_6");
#pragma HLS stream variable = from_input_offset_1283_to_output_pe_6 depth = 3
#pragma HLS data_pack variable = from_input_offset_1283_to_output_pe_6
  hls::stream<Data<float>> from_input_offset_1283_to_output_pe_5("from_input_offset_1283_to_output_pe_5");
#pragma HLS stream variable = from_input_offset_1283_to_output_pe_5 depth = 3
#pragma HLS data_pack variable = from_input_offset_1283_to_output_pe_5
  hls::stream<Data<float>> from_input_offset_1283_to_output_pe_4("from_input_offset_1283_to_output_pe_4");
#pragma HLS stream variable = from_input_offset_1283_to_output_pe_4 depth = 3
#pragma HLS data_pack variable = from_input_offset_1283_to_output_pe_4
  hls::stream<Data<float>> from_input_offset_1284_to_output_pe_5("from_input_offset_1284_to_output_pe_5");
#pragma HLS stream variable = from_input_offset_1284_to_output_pe_5 depth = 3
#pragma HLS data_pack variable = from_input_offset_1284_to_output_pe_5
  hls::stream<Data<float>> from_input_offset_1284_to_output_pe_4("from_input_offset_1284_to_output_pe_4");
#pragma HLS stream variable = from_input_offset_1284_to_output_pe_4 depth = 3
#pragma HLS data_pack variable = from_input_offset_1284_to_output_pe_4
  hls::stream<Data<float>> from_input_offset_1284_to_output_pe_3("from_input_offset_1284_to_output_pe_3");
#pragma HLS stream variable = from_input_offset_1284_to_output_pe_3 depth = 3
#pragma HLS data_pack variable = from_input_offset_1284_to_output_pe_3
  hls::stream<Data<float>> from_output_pe_5_to_output_bank_0("from_output_pe_5_to_output_bank_0");
#pragma HLS stream variable = from_output_pe_5_to_output_bank_0 depth = 5
#pragma HLS data_pack variable = from_output_pe_5_to_output_bank_0
  hls::stream<Data<float>> from_input_offset_1285_to_output_pe_4("from_input_offset_1285_to_output_pe_4");
#pragma HLS stream variable = from_input_offset_1285_to_output_pe_4 depth = 3
#pragma HLS data_pack variable = from_input_offset_1285_to_output_pe_4
  hls::stream<Data<float>> from_input_offset_1285_to_output_pe_3("from_input_offset_1285_to_output_pe_3");
#pragma HLS stream variable = from_input_offset_1285_to_output_pe_3 depth = 3
#pragma HLS data_pack variable = from_input_offset_1285_to_output_pe_3
  hls::stream<Data<float>> from_input_offset_1285_to_output_pe_2("from_input_offset_1285_to_output_pe_2");
#pragma HLS stream variable = from_input_offset_1285_to_output_pe_2 depth = 3
#pragma HLS data_pack variable = from_input_offset_1285_to_output_pe_2
  hls::stream<Data<float>> from_output_pe_4_to_output_bank_0("from_output_pe_4_to_output_bank_0");
#pragma HLS stream variable = from_output_pe_4_to_output_bank_0 depth = 5
#pragma HLS data_pack variable = from_output_pe_4_to_output_bank_0
  hls::stream<Data<float>> from_input_offset_1286_to_output_pe_3("from_input_offset_1286_to_output_pe_3");
#pragma HLS stream variable = from_input_offset_1286_to_output_pe_3 depth = 3
#pragma HLS data_pack variable = from_input_offset_1286_to_output_pe_3
  hls::stream<Data<float>> from_input_offset_1286_to_output_pe_2("from_input_offset_1286_to_output_pe_2");
#pragma HLS stream variable = from_input_offset_1286_to_output_pe_2 depth = 3
#pragma HLS data_pack variable = from_input_offset_1286_to_output_pe_2
  hls::stream<Data<float>> from_input_offset_1286_to_output_pe_1("from_input_offset_1286_to_output_pe_1");
#pragma HLS stream variable = from_input_offset_1286_to_output_pe_1 depth = 5
#pragma HLS data_pack variable = from_input_offset_1286_to_output_pe_1
  hls::stream<Data<float>> from_output_pe_3_to_output_bank_0("from_output_pe_3_to_output_bank_0");
#pragma HLS stream variable = from_output_pe_3_to_output_bank_0 depth = 5
#pragma HLS data_pack variable = from_output_pe_3_to_output_bank_0
  hls::stream<Data<float>> from_input_offset_1287_to_output_pe_2("from_input_offset_1287_to_output_pe_2");
#pragma HLS stream variable = from_input_offset_1287_to_output_pe_2 depth = 3
#pragma HLS data_pack variable = from_input_offset_1287_to_output_pe_2
  hls::stream<Data<float>> from_input_offset_1287_to_output_pe_1("from_input_offset_1287_to_output_pe_1");
#pragma HLS stream variable = from_input_offset_1287_to_output_pe_1 depth = 5
#pragma HLS data_pack variable = from_input_offset_1287_to_output_pe_1
  hls::stream<Data<float>> from_input_offset_1287_to_output_pe_0("from_input_offset_1287_to_output_pe_0");
#pragma HLS stream variable = from_input_offset_1287_to_output_pe_0 depth = 5
#pragma HLS data_pack variable = from_input_offset_1287_to_output_pe_0
  hls::stream<Data<float>> from_output_pe_2_to_output_bank_0("from_output_pe_2_to_output_bank_0");
#pragma HLS stream variable = from_output_pe_2_to_output_bank_0 depth = 5
#pragma HLS data_pack variable = from_output_pe_2_to_output_bank_0
  hls::stream<Data<float>> from_input_offset_649_to_input_offset_1281("from_input_offset_649_to_input_offset_1281");
#pragma HLS stream variable = from_input_offset_649_to_input_offset_1281 depth = 3
#pragma HLS data_pack variable = from_input_offset_649_to_input_offset_1281
  hls::stream<Data<float>> from_input_offset_649_to_output_pe_0("from_input_offset_649_to_output_pe_0");
#pragma HLS stream variable = from_input_offset_649_to_output_pe_0 depth = 7
#pragma HLS data_pack variable = from_input_offset_649_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_648_to_input_offset_1280("from_input_offset_648_to_input_offset_1280");
#pragma HLS stream variable = from_input_offset_648_to_input_offset_1280 depth = 3
#pragma HLS data_pack variable = from_input_offset_648_to_input_offset_1280
  hls::stream<Data<float>> from_input_offset_648_to_output_pe_1("from_input_offset_648_to_output_pe_1");
#pragma HLS stream variable = from_input_offset_648_to_output_pe_1 depth = 7
#pragma HLS data_pack variable = from_input_offset_648_to_output_pe_1
  hls::stream<Data<float>> from_input_offset_648_to_output_pe_0("from_input_offset_648_to_output_pe_0");
#pragma HLS stream variable = from_input_offset_648_to_output_pe_0 depth = 7
#pragma HLS data_pack variable = from_input_offset_648_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_1281_to_input_offset_1289("from_input_offset_1281_to_input_offset_1289");
#pragma HLS stream variable = from_input_offset_1281_to_input_offset_1289 depth = 3
#pragma HLS data_pack variable = from_input_offset_1281_to_input_offset_1289
  hls::stream<Data<float>> from_input_offset_1281_to_output_pe_7("from_input_offset_1281_to_output_pe_7");
#pragma HLS stream variable = from_input_offset_1281_to_output_pe_7 depth = 3
#pragma HLS data_pack variable = from_input_offset_1281_to_output_pe_7
  hls::stream<Data<float>> from_input_offset_1281_to_output_pe_6("from_input_offset_1281_to_output_pe_6");
#pragma HLS stream variable = from_input_offset_1281_to_output_pe_6 depth = 3
#pragma HLS data_pack variable = from_input_offset_1281_to_output_pe_6
  hls::stream<Data<float>> from_output_pe_6_to_output_bank_0("from_output_pe_6_to_output_bank_0");
#pragma HLS stream variable = from_output_pe_6_to_output_bank_0 depth = 5
#pragma HLS data_pack variable = from_output_pe_6_to_output_bank_0
  hls::stream<Data<float>> from_input_offset_1280_to_input_offset_1288("from_input_offset_1280_to_input_offset_1288");
#pragma HLS stream variable = from_input_offset_1280_to_input_offset_1288 depth = 3
#pragma HLS data_pack variable = from_input_offset_1280_to_input_offset_1288
  hls::stream<Data<float>> from_input_offset_1280_to_output_pe_7("from_input_offset_1280_to_output_pe_7");
#pragma HLS stream variable = from_input_offset_1280_to_output_pe_7 depth = 3
#pragma HLS data_pack variable = from_input_offset_1280_to_output_pe_7
  hls::stream<Data<float>> from_output_pe_7_to_output_bank_0("from_output_pe_7_to_output_bank_0");
#pragma HLS stream variable = from_output_pe_7_to_output_bank_0 depth = 5
#pragma HLS data_pack variable = from_output_pe_7_to_output_bank_0
  hls::stream<Data<float>> from_input_offset_1289_to_output_pe_0("from_input_offset_1289_to_output_pe_0");
#pragma HLS stream variable = from_input_offset_1289_to_output_pe_0 depth = 3
#pragma HLS data_pack variable = from_input_offset_1289_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_1288_to_output_pe_1("from_input_offset_1288_to_output_pe_1");
#pragma HLS stream variable = from_input_offset_1288_to_output_pe_1 depth = 3
#pragma HLS data_pack variable = from_input_offset_1288_to_output_pe_1
  hls::stream<Data<float>> from_input_offset_1288_to_output_pe_0("from_input_offset_1288_to_output_pe_0");
#pragma HLS stream variable = from_input_offset_1288_to_output_pe_0 depth = 3
#pragma HLS data_pack variable = from_input_offset_1288_to_output_pe_0
  hls::stream<Data<float>> from_output_pe_1_to_output_bank_0("from_output_pe_1_to_output_bank_0");
#pragma HLS stream variable = from_output_pe_1_to_output_bank_0 depth = 3
#pragma HLS data_pack variable = from_output_pe_1_to_output_bank_0
  hls::stream<Data<float>> from_output_pe_0_to_output_bank_0("from_output_pe_0_to_output_bank_0");
#pragma HLS stream variable = from_output_pe_0_to_output_bank_0 depth = 3
#pragma HLS data_pack variable = from_output_pe_0_to_output_bank_0

#pragma HLS dataflow
  BurstRead(bank_0_input_buf, bank_0_input, coalesced_data_num);
  Module0Func(
      /*output*/ from_input_bank_0_to_input_offset_2, 
      /*output*/ from_input_bank_0_to_input_offset_1, 
      /*output*/ from_input_bank_0_to_input_offset_0, 
      /*output*/ from_input_bank_0_to_input_offset_3, 
      /*output*/ from_input_bank_0_to_input_offset_4, 
      /*output*/ from_input_bank_0_to_input_offset_5, 
      /*output*/ from_input_bank_0_to_input_offset_6, 
      /*output*/ from_input_bank_0_to_input_offset_7, 
      /* input*/ bank_0_input_buf);
  Module1Func(
      /*output*/ from_input_offset_2_to_input_offset_642, 
      /*output*/ from_input_offset_2_to_output_pe_7, 
      /*output*/ from_input_offset_2_to_output_pe_6, 
      /*output*/ from_input_offset_2_to_output_pe_5, 
      /* input*/ from_input_bank_0_to_input_offset_2);
  Module2Func(
      /*output*/ from_input_offset_1_to_input_offset_9, 
      /*output*/ from_input_offset_1_to_output_pe_7, 
      /*output*/ from_input_offset_1_to_output_pe_6, 
      /* input*/ from_input_bank_0_to_input_offset_1);
  Module3Func(
      /*output*/ from_input_offset_0_to_input_offset_8, 
      /*output*/ from_input_offset_0_to_output_pe_7, 
      /* input*/ from_input_bank_0_to_input_offset_0);
  Module1Func(
      /*output*/ from_input_offset_3_to_input_offset_643, 
      /*output*/ from_input_offset_3_to_output_pe_6, 
      /*output*/ from_input_offset_3_to_output_pe_5, 
      /*output*/ from_input_offset_3_to_output_pe_4, 
      /* input*/ from_input_bank_0_to_input_offset_3);
  Module1Func(
      /*output*/ from_input_offset_4_to_input_offset_644, 
      /*output*/ from_input_offset_4_to_output_pe_5, 
      /*output*/ from_input_offset_4_to_output_pe_4, 
      /*output*/ from_input_offset_4_to_output_pe_3, 
      /* input*/ from_input_bank_0_to_input_offset_4);
  Module1Func(
      /*output*/ from_input_offset_5_to_input_offset_645, 
      /*output*/ from_input_offset_5_to_output_pe_4, 
      /*output*/ from_input_offset_5_to_output_pe_3, 
      /*output*/ from_input_offset_5_to_output_pe_2, 
      /* input*/ from_input_bank_0_to_input_offset_5);
  Module1Func(
      /*output*/ from_input_offset_6_to_input_offset_646, 
      /*output*/ from_input_offset_6_to_output_pe_3, 
      /*output*/ from_input_offset_6_to_output_pe_2, 
      /*output*/ from_input_offset_6_to_output_pe_1, 
      /* input*/ from_input_bank_0_to_input_offset_6);
  Module1Func(
      /*output*/ from_input_offset_7_to_input_offset_647, 
      /*output*/ from_input_offset_7_to_output_pe_2, 
      /*output*/ from_input_offset_7_to_output_pe_1, 
      /*output*/ from_input_offset_7_to_output_pe_0, 
      /* input*/ from_input_bank_0_to_input_offset_7);
  Module4Func(
      /*output*/ from_input_offset_642_to_input_offset_1282, 
      /*output*/ from_input_offset_642_to_output_pe_7, 
      /*output*/ from_input_offset_642_to_output_pe_6, 
      /*output*/ from_input_offset_642_to_output_pe_5, 
      /* input*/ from_input_offset_2_to_input_offset_642);
  Module5Func(
      /*output*/ from_input_offset_9_to_input_offset_641, 
      /*output*/ from_input_offset_9_to_output_pe_0, 
      /* input*/ from_input_offset_1_to_input_offset_9);
  Module6Func(
      /*output*/ from_input_offset_8_to_input_offset_640, 
      /*output*/ from_input_offset_8_to_output_pe_1, 
      /*output*/ from_input_offset_8_to_output_pe_0, 
      /* input*/ from_input_offset_0_to_input_offset_8);
  Module4Func(
      /*output*/ from_input_offset_643_to_input_offset_1283, 
      /*output*/ from_input_offset_643_to_output_pe_6, 
      /*output*/ from_input_offset_643_to_output_pe_5, 
      /*output*/ from_input_offset_643_to_output_pe_4, 
      /* input*/ from_input_offset_3_to_input_offset_643);
  Module4Func(
      /*output*/ from_input_offset_644_to_input_offset_1284, 
      /*output*/ from_input_offset_644_to_output_pe_5, 
      /*output*/ from_input_offset_644_to_output_pe_4, 
      /*output*/ from_input_offset_644_to_output_pe_3, 
      /* input*/ from_input_offset_4_to_input_offset_644);
  Module4Func(
      /*output*/ from_input_offset_645_to_input_offset_1285, 
      /*output*/ from_input_offset_645_to_output_pe_4, 
      /*output*/ from_input_offset_645_to_output_pe_3, 
      /*output*/ from_input_offset_645_to_output_pe_2, 
      /* input*/ from_input_offset_5_to_input_offset_645);
  Module4Func(
      /*output*/ from_input_offset_646_to_input_offset_1286, 
      /*output*/ from_input_offset_646_to_output_pe_3, 
      /*output*/ from_input_offset_646_to_output_pe_2, 
      /*output*/ from_input_offset_646_to_output_pe_1, 
      /* input*/ from_input_offset_6_to_input_offset_646);
  Module4Func(
      /*output*/ from_input_offset_647_to_input_offset_1287, 
      /*output*/ from_input_offset_647_to_output_pe_2, 
      /*output*/ from_input_offset_647_to_output_pe_1, 
      /*output*/ from_input_offset_647_to_output_pe_0, 
      /* input*/ from_input_offset_7_to_input_offset_647);
  Module7Func(
      /*output*/ from_input_offset_1282_to_output_pe_7, 
      /*output*/ from_input_offset_1282_to_output_pe_6, 
      /*output*/ from_input_offset_1282_to_output_pe_5, 
      /* input*/ from_input_offset_642_to_input_offset_1282);
  Module8Func(
      /*output*/ from_input_offset_641_to_input_offset_649, 
      /*output*/ from_input_offset_641_to_output_pe_7, 
      /*output*/ from_input_offset_641_to_output_pe_6, 
      /* input*/ from_input_offset_9_to_input_offset_641);
  Module9Func(
      /*output*/ from_input_offset_640_to_input_offset_648, 
      /*output*/ from_input_offset_640_to_output_pe_7, 
      /* input*/ from_input_offset_8_to_input_offset_640);
  Module7Func(
      /*output*/ from_input_offset_1283_to_output_pe_6, 
      /*output*/ from_input_offset_1283_to_output_pe_5, 
      /*output*/ from_input_offset_1283_to_output_pe_4, 
      /* input*/ from_input_offset_643_to_input_offset_1283);
  Module7Func(
      /*output*/ from_input_offset_1284_to_output_pe_5, 
      /*output*/ from_input_offset_1284_to_output_pe_4, 
      /*output*/ from_input_offset_1284_to_output_pe_3, 
      /* input*/ from_input_offset_644_to_input_offset_1284);
  Module10Func(
      /*output*/ from_output_pe_5_to_output_bank_0, 
      /* input*/ from_input_offset_1284_to_output_pe_5, 
      /* input*/ from_input_offset_1283_to_output_pe_5, 
      /* input*/ from_input_offset_1282_to_output_pe_5, 
      /* input*/ from_input_offset_644_to_output_pe_5, 
      /* input*/ from_input_offset_643_to_output_pe_5, 
      /* input*/ from_input_offset_642_to_output_pe_5, 
      /* input*/ from_input_offset_4_to_output_pe_5, 
      /* input*/ from_input_offset_3_to_output_pe_5, 
      /* input*/ from_input_offset_2_to_output_pe_5);
  Module7Func(
      /*output*/ from_input_offset_1285_to_output_pe_4, 
      /*output*/ from_input_offset_1285_to_output_pe_3, 
      /*output*/ from_input_offset_1285_to_output_pe_2, 
      /* input*/ from_input_offset_645_to_input_offset_1285);
  Module10Func(
      /*output*/ from_output_pe_4_to_output_bank_0, 
      /* input*/ from_input_offset_1285_to_output_pe_4, 
      /* input*/ from_input_offset_1284_to_output_pe_4, 
      /* input*/ from_input_offset_1283_to_output_pe_4, 
      /* input*/ from_input_offset_645_to_output_pe_4, 
      /* input*/ from_input_offset_644_to_output_pe_4, 
      /* input*/ from_input_offset_643_to_output_pe_4, 
      /* input*/ from_input_offset_5_to_output_pe_4, 
      /* input*/ from_input_offset_4_to_output_pe_4, 
      /* input*/ from_input_offset_3_to_output_pe_4);
  Module7Func(
      /*output*/ from_input_offset_1286_to_output_pe_3, 
      /*output*/ from_input_offset_1286_to_output_pe_2, 
      /*output*/ from_input_offset_1286_to_output_pe_1, 
      /* input*/ from_input_offset_646_to_input_offset_1286);
  Module10Func(
      /*output*/ from_output_pe_3_to_output_bank_0, 
      /* input*/ from_input_offset_1286_to_output_pe_3, 
      /* input*/ from_input_offset_1285_to_output_pe_3, 
      /* input*/ from_input_offset_1284_to_output_pe_3, 
      /* input*/ from_input_offset_646_to_output_pe_3, 
      /* input*/ from_input_offset_645_to_output_pe_3, 
      /* input*/ from_input_offset_644_to_output_pe_3, 
      /* input*/ from_input_offset_6_to_output_pe_3, 
      /* input*/ from_input_offset_5_to_output_pe_3, 
      /* input*/ from_input_offset_4_to_output_pe_3);
  Module7Func(
      /*output*/ from_input_offset_1287_to_output_pe_2, 
      /*output*/ from_input_offset_1287_to_output_pe_1, 
      /*output*/ from_input_offset_1287_to_output_pe_0, 
      /* input*/ from_input_offset_647_to_input_offset_1287);
  Module10Func(
      /*output*/ from_output_pe_2_to_output_bank_0, 
      /* input*/ from_input_offset_1287_to_output_pe_2, 
      /* input*/ from_input_offset_1286_to_output_pe_2, 
      /* input*/ from_input_offset_1285_to_output_pe_2, 
      /* input*/ from_input_offset_647_to_output_pe_2, 
      /* input*/ from_input_offset_646_to_output_pe_2, 
      /* input*/ from_input_offset_645_to_output_pe_2, 
      /* input*/ from_input_offset_7_to_output_pe_2, 
      /* input*/ from_input_offset_6_to_output_pe_2, 
      /* input*/ from_input_offset_5_to_output_pe_2);
  Module5Func(
      /*output*/ from_input_offset_649_to_input_offset_1281, 
      /*output*/ from_input_offset_649_to_output_pe_0, 
      /* input*/ from_input_offset_641_to_input_offset_649);
  Module6Func(
      /*output*/ from_input_offset_648_to_input_offset_1280, 
      /*output*/ from_input_offset_648_to_output_pe_1, 
      /*output*/ from_input_offset_648_to_output_pe_0, 
      /* input*/ from_input_offset_640_to_input_offset_648);
  Module8Func(
      /*output*/ from_input_offset_1281_to_input_offset_1289, 
      /*output*/ from_input_offset_1281_to_output_pe_7, 
      /*output*/ from_input_offset_1281_to_output_pe_6, 
      /* input*/ from_input_offset_649_to_input_offset_1281);
  Module10Func(
      /*output*/ from_output_pe_6_to_output_bank_0, 
      /* input*/ from_input_offset_1283_to_output_pe_6, 
      /* input*/ from_input_offset_1282_to_output_pe_6, 
      /* input*/ from_input_offset_1281_to_output_pe_6, 
      /* input*/ from_input_offset_643_to_output_pe_6, 
      /* input*/ from_input_offset_642_to_output_pe_6, 
      /* input*/ from_input_offset_641_to_output_pe_6, 
      /* input*/ from_input_offset_3_to_output_pe_6, 
      /* input*/ from_input_offset_2_to_output_pe_6, 
      /* input*/ from_input_offset_1_to_output_pe_6);
  Module9Func(
      /*output*/ from_input_offset_1280_to_input_offset_1288, 
      /*output*/ from_input_offset_1280_to_output_pe_7, 
      /* input*/ from_input_offset_648_to_input_offset_1280);
  Module10Func(
      /*output*/ from_output_pe_7_to_output_bank_0, 
      /* input*/ from_input_offset_1282_to_output_pe_7, 
      /* input*/ from_input_offset_1281_to_output_pe_7, 
      /* input*/ from_input_offset_1280_to_output_pe_7, 
      /* input*/ from_input_offset_642_to_output_pe_7, 
      /* input*/ from_input_offset_641_to_output_pe_7, 
      /* input*/ from_input_offset_640_to_output_pe_7, 
      /* input*/ from_input_offset_2_to_output_pe_7, 
      /* input*/ from_input_offset_1_to_output_pe_7, 
      /* input*/ from_input_offset_0_to_output_pe_7);
  Module11Func(
      /*output*/ from_input_offset_1289_to_output_pe_0, 
      /* input*/ from_input_offset_1281_to_input_offset_1289);
  Module5Func(
      /*output*/ from_input_offset_1288_to_output_pe_1, 
      /*output*/ from_input_offset_1288_to_output_pe_0, 
      /* input*/ from_input_offset_1280_to_input_offset_1288);
  Module10Func(
      /*output*/ from_output_pe_1_to_output_bank_0, 
      /* input*/ from_input_offset_1288_to_output_pe_1, 
      /* input*/ from_input_offset_1287_to_output_pe_1, 
      /* input*/ from_input_offset_1286_to_output_pe_1, 
      /* input*/ from_input_offset_648_to_output_pe_1, 
      /* input*/ from_input_offset_647_to_output_pe_1, 
      /* input*/ from_input_offset_646_to_output_pe_1, 
      /* input*/ from_input_offset_8_to_output_pe_1, 
      /* input*/ from_input_offset_7_to_output_pe_1, 
      /* input*/ from_input_offset_6_to_output_pe_1);
  Module10Func(
      /*output*/ from_output_pe_0_to_output_bank_0, 
      /* input*/ from_input_offset_1289_to_output_pe_0, 
      /* input*/ from_input_offset_1288_to_output_pe_0, 
      /* input*/ from_input_offset_1287_to_output_pe_0, 
      /* input*/ from_input_offset_649_to_output_pe_0, 
      /* input*/ from_input_offset_648_to_output_pe_0, 
      /* input*/ from_input_offset_647_to_output_pe_0, 
      /* input*/ from_input_offset_9_to_output_pe_0, 
      /* input*/ from_input_offset_8_to_output_pe_0, 
      /* input*/ from_input_offset_7_to_output_pe_0);
  Module12Func(
      /*output*/ bank_0_output_buf, 
      /* input*/ from_output_pe_5_to_output_bank_0, 
      /* input*/ from_output_pe_4_to_output_bank_0, 
      /* input*/ from_output_pe_3_to_output_bank_0, 
      /* input*/ from_output_pe_2_to_output_bank_0, 
      /* input*/ from_output_pe_6_to_output_bank_0, 
      /* input*/ from_output_pe_7_to_output_bank_0, 
      /* input*/ from_output_pe_1_to_output_bank_0, 
      /* input*/ from_output_pe_0_to_output_bank_0);
  BurstWrite(bank_0_output, bank_0_output_buf, coalesced_data_num);
}

}  // extern "C"
