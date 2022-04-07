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
unroll factor: 1
input float32: input(640, *)
output float32: output(0, 0) = ((((input(-1, 1) + input(0, 0)) + input(0, 1)) + input(1, 1)) + input(0, 2)) * 0.200000F
border: None
cluster: None
*/

// stencil window size: (3, 3)
// stencil distace: 1281
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
  /* input*/ hls::stream<Data<ap_uint<512>>>& dram_input_bank_0_fifo)
{
  #pragma HLS data_pack variable = dram_input_bank_0_fifo
  #pragma HLS data_pack variable = fifo_st_0
module_0:
  for (bool enable = true; enable;)
  {
    #pragma HLS pipeline II = 16
    if (!dram_input_bank_0_fifo.empty())
    {
      ap_uint<512> dram_input_bank_0_buf;
      const bool dram_input_bank_0_buf_enable = ReadData(dram_input_bank_0_buf, dram_input_bank_0_fifo);
      const bool enabled = dram_input_bank_0_buf_enable;
      enable = enabled;
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(31, 0))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(63, 32))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(95, 64))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(127, 96))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(159, 128))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(191, 160))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(223, 192))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(255, 224))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(287, 256))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(319, 288))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(351, 320))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(383, 352))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(415, 384))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(447, 416))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(479, 448))), true);
      WriteData(fifo_st_0, Reinterpret<float>(static_cast<ap_uint<32>>(dram_input_bank_0_buf(511, 480))), enabled);
    } // if not empty
  } // for module_0
} // Module0Func

void Module1Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
  #pragma HLS data_pack variable = fifo_st_0
  #pragma HLS data_pack variable = fifo_st_1
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
    } // if not empty
  } // for module_1
} // Module1Func

void Module2Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /*output*/ hls::stream<Data<float>>& fifo_st_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
  #pragma HLS data_pack variable = fifo_st_0
  #pragma HLS data_pack variable = fifo_st_1
  #pragma HLS data_pack variable = fifo_ld_0
  float fifo_ref_0_delayed_639_buf[639];
  ap_uint<10> fifo_ref_0_delayed_639_ptr = 0;
module_2:
  for (bool enable = true; enable;)
  {
    #pragma HLS pipeline II = 1
    #pragma HLS dependence variable = fifo_ref_0_delayed_639_buf inter false
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_639 = fifo_ref_0_delayed_639_buf[fifo_ref_0_delayed_639_ptr];;
      const float let_0 = fifo_ref_0_delayed_639;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_639_buf[fifo_ref_0_delayed_639_ptr] = fifo_ref_0;
      fifo_ref_0_delayed_639_ptr = fifo_ref_0_delayed_639_ptr < 638 ? ap_uint<10>(fifo_ref_0_delayed_639_ptr+1) : ap_uint<10>(0);
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
  float fifo_ref_0_delayed_1_buf[1];
  ap_uint<1> fifo_ref_0_delayed_1_ptr = 0;
module_3:
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
      const float fifo_ref_0_delayed_1 = fifo_ref_0_delayed_1_buf[fifo_ref_0_delayed_1_ptr];;
      const float let_0 = fifo_ref_0_delayed_1;
      WriteData(fifo_st_0, float(let_0), enabled);
      WriteData(fifo_st_1, float(let_0), enabled);
      fifo_ref_0_delayed_1_buf[fifo_ref_0_delayed_1_ptr] = fifo_ref_0;
      fifo_ref_0_delayed_1_ptr = fifo_ref_0_delayed_1_ptr < 0 ? ap_uint<1>(fifo_ref_0_delayed_1_ptr+1) : ap_uint<1>(0);
    } // if not empty
  } // for module_3
} // Module3Func

void Module4Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
  #pragma HLS data_pack variable = fifo_st_0
  #pragma HLS data_pack variable = fifo_ld_0
  float fifo_ref_0_delayed_639_buf[639];
  ap_uint<10> fifo_ref_0_delayed_639_ptr = 0;
module_4:
  for (bool enable = true; enable;)
  {
    #pragma HLS pipeline II = 1
    #pragma HLS dependence variable = fifo_ref_0_delayed_639_buf inter false
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      const float fifo_ref_0_delayed_639 = fifo_ref_0_delayed_639_buf[fifo_ref_0_delayed_639_ptr];;
      const float let_0 = fifo_ref_0_delayed_639;
      WriteData(fifo_st_0, float(let_0), enabled);
      fifo_ref_0_delayed_639_buf[fifo_ref_0_delayed_639_ptr] = fifo_ref_0;
      fifo_ref_0_delayed_639_ptr = fifo_ref_0_delayed_639_ptr < 638 ? ap_uint<10>(fifo_ref_0_delayed_639_ptr+1) : ap_uint<10>(0);
    } // if not empty
  } // for module_4
} // Module4Func

void Module5Func(
  /*output*/ hls::stream<Data<float>>& fifo_st_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_1, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_2, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_3, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_4)
{
  #pragma HLS data_pack variable = fifo_st_0
  #pragma HLS data_pack variable = fifo_ld_0
  #pragma HLS data_pack variable = fifo_ld_1
  #pragma HLS data_pack variable = fifo_ld_2
  #pragma HLS data_pack variable = fifo_ld_3
  #pragma HLS data_pack variable = fifo_ld_4
module_5:
  for (bool enable = true; enable;)
  {
    #pragma HLS pipeline II = 1
    if (!fifo_ld_0.empty() && !fifo_ld_1.empty() && !fifo_ld_2.empty() && !fifo_ld_3.empty() && !fifo_ld_4.empty())
    {
      float fifo_ref_0;
      float fifo_ref_1;
      float fifo_ref_2;
      float fifo_ref_3;
      float fifo_ref_4;
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool fifo_ref_1_enable = ReadData(fifo_ref_1, fifo_ld_1);
      const bool fifo_ref_2_enable = ReadData(fifo_ref_2, fifo_ld_2);
      const bool fifo_ref_3_enable = ReadData(fifo_ref_3, fifo_ld_3);
      const bool fifo_ref_4_enable = ReadData(fifo_ref_4, fifo_ld_4);
      const bool enabled = fifo_ref_0_enable && fifo_ref_1_enable && fifo_ref_2_enable && fifo_ref_3_enable && fifo_ref_4_enable;
      enable = enabled;
      WriteData(fifo_st_0, float((((((fifo_ref_0 + fifo_ref_1) + fifo_ref_2) + fifo_ref_3) + fifo_ref_4) * 0.200000F)), enabled);
    } // if not empty
  } // for module_5
} // Module5Func

void Module6Func(
  /*output*/ hls::stream<Data<ap_uint<512>>>& dram_output_bank_0_fifo, 
  /* input*/ hls::stream<Data<float>>& fifo_ld_0)
{
  #pragma HLS data_pack variable = dram_output_bank_0_fifo
  #pragma HLS data_pack variable = fifo_ld_0
module_6:
  for (bool enable = true; enable;)
  {
    #pragma HLS pipeline II = 16
    if (!fifo_ld_0.empty())
    {
      float fifo_ref_0;
      ap_uint<512> dram_output_bank_0_buf;
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(31, 0) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(63, 32) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(95, 64) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(127, 96) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(159, 128) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(191, 160) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(223, 192) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(255, 224) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(287, 256) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(319, 288) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(351, 320) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(383, 352) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(415, 384) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(447, 416) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      ReadData(fifo_ref_0, fifo_ld_0);
      dram_output_bank_0_buf(479, 448) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      const bool fifo_ref_0_enable = ReadData(fifo_ref_0, fifo_ld_0);
      const bool enabled = fifo_ref_0_enable;
      enable = enabled;
      dram_output_bank_0_buf(511, 480) = Reinterpret<ap_uint<32>>(fifo_ref_0);
      WriteData(dram_output_bank_0_fifo, dram_output_bank_0_buf, enabled);
    } // if not empty
  } // for module_6
} // Module6Func

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

  hls::stream<Data<float>> from_input_bank_0_to_input_offset_0("from_input_bank_0_to_input_offset_0");
  #pragma HLS stream variable = from_input_bank_0_to_input_offset_0 depth = 3
  #pragma HLS data_pack variable = from_input_bank_0_to_input_offset_0
  hls::stream<Data<float>> from_input_offset_0_to_input_offset_639("from_input_offset_0_to_input_offset_639");
  #pragma HLS stream variable = from_input_offset_0_to_input_offset_639 depth = 3
  #pragma HLS data_pack variable = from_input_offset_0_to_input_offset_639
  hls::stream<Data<float>> from_input_offset_0_to_output_pe_0("from_input_offset_0_to_output_pe_0");
  #pragma HLS stream variable = from_input_offset_0_to_output_pe_0 depth = 11
  #pragma HLS data_pack variable = from_input_offset_0_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_639_to_input_offset_640("from_input_offset_639_to_input_offset_640");
  #pragma HLS stream variable = from_input_offset_639_to_input_offset_640 depth = 3
  #pragma HLS data_pack variable = from_input_offset_639_to_input_offset_640
  hls::stream<Data<float>> from_input_offset_639_to_output_pe_0("from_input_offset_639_to_output_pe_0");
  #pragma HLS stream variable = from_input_offset_639_to_output_pe_0 depth = 9
  #pragma HLS data_pack variable = from_input_offset_639_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_640_to_input_offset_641("from_input_offset_640_to_input_offset_641");
  #pragma HLS stream variable = from_input_offset_640_to_input_offset_641 depth = 3
  #pragma HLS data_pack variable = from_input_offset_640_to_input_offset_641
  hls::stream<Data<float>> from_input_offset_640_to_output_pe_0("from_input_offset_640_to_output_pe_0");
  #pragma HLS stream variable = from_input_offset_640_to_output_pe_0 depth = 7
  #pragma HLS data_pack variable = from_input_offset_640_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_641_to_input_offset_1280("from_input_offset_641_to_input_offset_1280");
  #pragma HLS stream variable = from_input_offset_641_to_input_offset_1280 depth = 3
  #pragma HLS data_pack variable = from_input_offset_641_to_input_offset_1280
  hls::stream<Data<float>> from_input_offset_641_to_output_pe_0("from_input_offset_641_to_output_pe_0");
  #pragma HLS stream variable = from_input_offset_641_to_output_pe_0 depth = 5
  #pragma HLS data_pack variable = from_input_offset_641_to_output_pe_0
  hls::stream<Data<float>> from_input_offset_1280_to_output_pe_0("from_input_offset_1280_to_output_pe_0");
  #pragma HLS stream variable = from_input_offset_1280_to_output_pe_0 depth = 3
  #pragma HLS data_pack variable = from_input_offset_1280_to_output_pe_0
  hls::stream<Data<float>> from_output_pe_0_to_output_bank_0("from_output_pe_0_to_output_bank_0");
  #pragma HLS stream variable = from_output_pe_0_to_output_bank_0 depth = 3
  #pragma HLS data_pack variable = from_output_pe_0_to_output_bank_0

#pragma HLS dataflow
  BurstRead(bank_0_input_buf, bank_0_input, coalesced_data_num);
  Module0Func(
      /*output*/ from_input_bank_0_to_input_offset_0, 
      /* input*/ bank_0_input_buf);
  Module1Func(
      /*output*/ from_input_offset_0_to_input_offset_639, 
      /*output*/ from_input_offset_0_to_output_pe_0, 
      /* input*/ from_input_bank_0_to_input_offset_0);
  Module2Func(
      /*output*/ from_input_offset_639_to_input_offset_640, 
      /*output*/ from_input_offset_639_to_output_pe_0, 
      /* input*/ from_input_offset_0_to_input_offset_639);
  Module3Func(
      /*output*/ from_input_offset_640_to_input_offset_641, 
      /*output*/ from_input_offset_640_to_output_pe_0, 
      /* input*/ from_input_offset_639_to_input_offset_640);
  Module3Func(
      /*output*/ from_input_offset_641_to_input_offset_1280, 
      /*output*/ from_input_offset_641_to_output_pe_0, 
      /* input*/ from_input_offset_640_to_input_offset_641);
  Module4Func(
      /*output*/ from_input_offset_1280_to_output_pe_0, 
      /* input*/ from_input_offset_641_to_input_offset_1280);
  Module5Func(
      /*output*/ from_output_pe_0_to_output_bank_0, 
      /* input*/ from_input_offset_641_to_output_pe_0, 
      /* input*/ from_input_offset_1280_to_output_pe_0, 
      /* input*/ from_input_offset_640_to_output_pe_0, 
      /* input*/ from_input_offset_639_to_output_pe_0, 
      /* input*/ from_input_offset_0_to_output_pe_0);
  Module6Func(
      /*output*/ bank_0_output_buf, 
      /* input*/ from_output_pe_0_to_output_bank_0);
  BurstWrite(bank_0_output, bank_0_output_buf, coalesced_data_num);
}

}  // extern "C"
