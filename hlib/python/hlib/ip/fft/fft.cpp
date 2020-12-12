#include "hls_fft.h"
#include <complex>

struct config : hls::ip_fft::params_t {
  static const unsigned ordering_opt = hls::ip_fft::natural_order;
  static const unsigned config_width = 16; // FFT_CONFIG_WIDTH
};
typedef ap_fixed<16,1> data_t;
typedef std::complex<data_t> fxpComplex;

void fft_wrapper(int* X_real, int* X_img, float* F_real, float* F_img, int size) {
  hls::ip_fft::config_t<config> fft_config;
  hls::ip_fft::status_t<config> fft_status;
  #pragma HLS INTERFACE ap_fifo port=fft_config
  fft_config.setDir(0);
  fft_config.setSch(0x2AB);
  std::complex<data_t> xn[size];
  std::complex<data_t> xk[size];
  #pragma HLS INTERFACE ap_fifo port=xn depth=16
  #pragma HLS INTERFACE ap_fifo port=xk depth=16
  for (int i = 0; i < size; i++) {{ 
    #pragma HLS pipeline rewind
    xn[i] = fxpComplex(X_real[i], F_real[i]);
  }}
  hls::fft<config>(xn, xk, &fft_status, &fft_config); 
  for (int i = 0; i < size; i++) {{
    #pragma HLS pipeline rewind
    F_real[i] = xk[i].real();
    F_img[i] = xk[i].imag();
  }}
}
