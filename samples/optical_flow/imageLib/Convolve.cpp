///////////////////////////////////////////////////////////////////////////
//
// NAME
//  Convolve.cpp -- separable and non-separable linear convolution
//
// DESIGN NOTES
//  An intermediate row buffer is allocated to fill in the border pixels
//  required so that all convolutions are well defined.  The row buffer
//  is floating point, so that all intermediate results can be computed
//  with minimal loss in precision.  This also avoids excessive type
//  conversion during convolution, since the kernels are floats anyway.
//
//  If fixpoint variants are desired for efficiency (e.g., using
//  multimedia extensions), then this would have to be modified.
//
//  TODO:  this current version is not very efficient.  For example,
//  the separable code uses an intermediate image, instead of just a
//  row buffer.  Also, downsampling convolution could be more efficient.
//
// SEE ALSO
//  Convolve.h          longer description of these routines
//
// Copyright © Richard Szeliski, 2001.
// See Copyright.h for more details
//
///////////////////////////////////////////////////////////////////////////

#include "Image.h"
#include "Error.h"
#include "Convert.h"
#include "Convolve.h"

static int TrimIndex(int k, EBorderMode e, int n)
{
    // Compute the index value 0 <= k < n (return -1 for Zero mode)
    switch (e)
    {
    case eBorderReplicate:  // replicate border values
        return __max(0, __min(n-1, k));
    case eBorderZero:       // zero padding
        return -1;
    case eBorderReflect:    // reflect border pixels
        while (k < 0 || k >= n)
        {
            k = (k < 0) ? -k : 2*n-1-k;
        }
        return k;
    case eBorderCyclic:     // wrap pixel values
        return (k % n + n) % n;
    }
    throw CError("Convolve[Separable]: %d is not a valid borderMode", int(e));
}

template <class T>
static void FillRowBuffer(float buf[], CImageOf<T>& src, CFloatImage& kernel,
                          int k, int n)
{
    // Compute the real row address
    CShape sShape = src.Shape();
    int nB = sShape.nBands;
    int k0 = TrimIndex(k + kernel.origin[1], src.borderMode, sShape.height);
    if (k0 < 0)
    {
        memset(buf, 0, n * sizeof(float));
        return;
    }

    // Fill the row
    T* srcP = &src.Pixel(0, k0, 0);
    int m = n / nB;
    for (int l = 0; l < m; l++, buf += nB)
    {
        int l0 = TrimIndex(l + kernel.origin[0], src.borderMode, sShape.width);
        if (l0 < 0)
            memset(buf, 0, nB * sizeof(T));
        else
            for (int b = 0; b < nB; b++)
                buf[b] = (float)srcP[l0*nB + b];
    }
}

static
void ConvolveRow2D(CFloatImage& buffer, CFloatImage& kernel, float dst[],
                   int n)
{
    CShape kShape = kernel.Shape();
    int kX  = kShape.width;
    int kY  = kShape.height;
    CShape bShape = buffer.Shape();
    int nB  = bShape.nBands;

    for (int i = 0; i < n; i++)
    {
        for (int b = 0; b < nB; b++)
        {
            float sum = 0.0f;
            for (int k = 0; k < kY; k++)
            {
                float* kPtr = &kernel.Pixel(0, k, 0);
                float* bPtr = &buffer.Pixel(i, k, b);
                for (int l = 0; l < kX; l++, bPtr += nB)
                    sum += kPtr[l] * bPtr[0];
            }
            *dst++ = sum;
        }
    }
}

template <class T>
void Convolve(CImageOf<T> src, CImageOf<T>& dst,
              CFloatImage kernel,
              float scale, float offset)
{
    // Determine the shape of the kernel and row buffer
    CShape kShape = kernel.Shape();
    CShape sShape = src.Shape();
    CShape bShape(sShape.width + kShape.width, kShape.height, sShape.nBands);
    int bWidth = bShape.width * bShape.nBands;

    // Allocate the result, if necessary, and the row buffer
    dst.ReAllocate(sShape, false);
    CFloatImage buffer(bShape);
    if (sShape.width * sShape.height * sShape.nBands == 0)
        return;
    CFloatImage output(CShape(sShape.width, 1, sShape.nBands));

    // Fill up the row buffer initially
    for (int k = 0; k < kShape.height; k++)
        FillRowBuffer(&buffer.Pixel(0, k, 0), src, kernel, k, bWidth);

    // Determine if clipping is required
    //  (we assume up-conversion to float never requires clipping, i.e.,
    //   floats have the highest dynamic range)
    T minVal = dst.MinVal();
    T maxVal = dst.MaxVal();
    if (minVal <= buffer.MinVal() && maxVal >= buffer.MaxVal())
        minVal = maxVal = 0;

    // Process each row
    for (int y = 0; y < sShape.height; y++)
    {
        // Do the convolution
        ConvolveRow2D(buffer, kernel, &output.Pixel(0, 0, 0),
                      sShape.width);

        // Scale, offset, and type convert
        ScaleAndOffsetLine(&output.Pixel(0, 0, 0), &dst.Pixel(0, y, 0),
                           sShape.width * sShape.nBands,
                           scale, offset, minVal, maxVal);

        // Shift up the row buffer and fill the last line
        if (y < sShape.height-1)
        {
            int k;
            for (k = 0; k < kShape.height-1; k++)
                memcpy(&buffer.Pixel(0, k, 0), &buffer.Pixel(0, k+1, 0),
                       bWidth * sizeof(float));
            FillRowBuffer(&buffer.Pixel(0, k, 0), src, kernel, y+k+1, bWidth);
        }
    }
}

template <class T>
void ConvolveSeparable(CImageOf<T> src, CImageOf<T>& dst,
                       CFloatImage x_kernel, CFloatImage y_kernel,
                       float scale, float offset,
                       int decimate, int interpolate)
{
    // Allocate the result, if necessary
    CShape dShape = src.Shape();
    if (decimate > 1)
    {
        dShape.width  = (dShape.width  + decimate-1) / decimate;
        dShape.height = (dShape.height + decimate-1) / decimate;
    }
    dst.ReAllocate(dShape, false);

    // Allocate the intermediate images
    CImageOf<T> tmpImg1(src.Shape());
    CImageOf<T> tmpImg2(src.Shape());

    // Create a proper vertical convolution kernel
    CFloatImage v_kernel(1, y_kernel.Shape().width, 1);
    for (int k = 0; k < y_kernel.Shape().width; k++)
        v_kernel.Pixel(0, k, 0) = y_kernel.Pixel(k, 0, 0);
    v_kernel.origin[1] = y_kernel.origin[0];

    // Perform the two convolutions
    Convolve(src, tmpImg1, x_kernel, 1.0f, 0.0f);
    Convolve(tmpImg1, tmpImg2, v_kernel, scale, offset);

    // Downsample or copy
    for (int y = 0; y < dShape.height; y++)
    {
        T* sPtr = &tmpImg2.Pixel(0, y * decimate, 0);
        T* dPtr = &dst.Pixel(0, y, 0);
        int nB  = dShape.nBands;
        for (int x = 0; x < dShape.width; x++)
        {
            for (int b = 0; b < nB; b++)
                dPtr[b] = sPtr[b];
            sPtr += decimate * nB;
            dPtr += nB;
        }
    }

    interpolate++; // to get rid of "unused parameter" warning
}

template <class T>
void InstantiateConvolutionOf(CImageOf<T> img)
{
    CFloatImage kernel;
    Convolve(img, img, kernel, 1.0, 0.0);
    ConvolveSeparable(img, img, kernel, kernel, 1.0, 0.0, 1, 1);
}

void InstantiateConvolutions()
{
    InstantiateConvolutionOf(CByteImage());
    InstantiateConvolutionOf(CIntImage());
    InstantiateConvolutionOf(CFloatImage());
}

//
//  Default kernels
//

CFloatImage ConvolveKernel_121;
CFloatImage ConvolveKernel_14641;
CFloatImage ConvolveKernel_8TapLowPass;

struct KernelInit
{
    KernelInit();
};

KernelInit::KernelInit()
{
#if 0   // not currently used:
    static float k_11[2] = {0.5f, 0.5f};
#endif
    static float k_121[3] = {0.25f, 0.5f, 0.25f};
    static float k_14641[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
#if 0   // not currently used:
    static float k_8ptFP[8] = {-0.044734f, -0.059009f,  0.156544f,  0.449199f,
                                0.449199f,  0.156544f, -0.059009f, -0.044734f};
#endif
    // The following are derived as fix-point /256 fractions of the above:
    //  -12, -15, 40, 115
    static float k_8ptI [8] = {-0.04687500f, -0.05859375f,  0.15625000f,  0.44921875f,
                                0.44921875f,  0.15625000f, -0.05859375f, -0.04687500f};

    ConvolveKernel_121.ReAllocate(CShape(3, 1, 1), k_121, false, 3);
    ConvolveKernel_121.origin[0] = -1;
    ConvolveKernel_14641.ReAllocate(CShape(5, 1, 1), k_14641, false, 5);
    ConvolveKernel_14641.origin[0] = -2;
    ConvolveKernel_8TapLowPass.ReAllocate(CShape(8, 1, 1), k_8ptI, false, 8);
    ConvolveKernel_8TapLowPass.origin[0] = -4;
}

KernelInit ConvKernelInitializer;
