///////////////////////////////////////////////////////////////////////////
//
// NAME
//  Convolve.h -- separable and non-separable linear convolution
//
// SPECIFICATION
//  void Convolve(CImageOf<T> src, CImageOf<T>& dst,
//                CFloatImage kernel,
//                int decimate, int interpolate);
//
//  void ConvolveSeparable(CImageOf<T> src, CImageOf<T>& dst,
//                         CFloatImage xKernel, CFloatImage yKernel,
//                         int decimate, int interpolate);
//
// PARAMETERS
//  src                 source image
//  dst                 destination image
//  kernel              2-D convolution kernel
//  xKernel, yKernel    1-D convolution kernels (1-row images)
//  decimate            decimation factor (1 = none, 2 = half, ...)
//  interpolate			interpolation factor (1 = none, 2 = double, ...)
//
// DESCRIPTION
//  Perform a 2D or separable 1D convolution.  The convolution kernels
//  are supplied as 2D floating point images  (1 row images for the separable
//  kernels).  The position of the "0" pixel in the kernel is determined
//  by the kernel.origin[] parameters, which specify the offset (coordinate,
//  usually negative) of the first (top-left) pixel in the kernel.
//
//  Because a temporary row buffer is used for row padding and separable
//  convolution intermediaries, the src and dst images can be the same
//  (in place convolution), so long as decimation = interpolation = 1.
//
//  If dst is not of the right shape, it is reallocated to the right shape.
//
//  The padding type of src (src.borderMode) determines how pixels are
//  filled for convolutions.
//
// SEE ALSO
//  Convolve.cpp        implementation
//  Image.h             image class definition
//
// Copyright © Richard Szeliski, 2001.
// See Copyright.h for more details
//
///////////////////////////////////////////////////////////////////////////

template <class T>
void Convolve(CImageOf<T> src, CImageOf<T>& dst,
              CFloatImage kernel,
              float scale, float offset);


template <class T>
void ConvolveSeparable(CImageOf<T> src, CImageOf<T>& dst,
                       CFloatImage x_kernel, CFloatImage y_kernel,
                       float scale, float offset,
                       int decimate, int interpolate);

extern CFloatImage ConvolveKernel_121;
extern CFloatImage ConvolveKernel_14641;
extern CFloatImage ConvolveKernel_8TapLowPass;
