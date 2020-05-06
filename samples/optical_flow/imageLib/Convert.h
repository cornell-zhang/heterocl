///////////////////////////////////////////////////////////////////////////
//
// NAME
//  Convert.h -- convert between image types, copy images, select bands
//
// DESCRIPTION
//  This file defines a number of conversion/copying utilities:
//
//  void ScaleAndOffset(CImageOf<T1>& src, CImageOf<T2>& dst,
//                       float scale, float offset);
//      -- scale and offset one image into another (optionally convert type)
//
//  void CopyPixels(CImageOf<T1>& src, CImageOf<T2>& dst);
//      -- convert pixel types or just copy pixels from src to dst
//
//  CImageOf<T> ConvertToRGBA(CImageOf<T> src);
//      -- convert from gray (1-band) image to RGBA (alpha == 255)
//
//  CImageOf<T> ConvertToGray(CImageOf<T> src);
//      -- convert from RGBA (4-band) image to gray, using Y formula,
//          Y = 0.212671 * R + 0.715160 * G + 0.072169 * B
//
//  void BandSelect(CImageOf<T>& src, CImageOf<T>& dst, int sBand, int dBand);
//      -- copy the sBand from src into the dBand in dst
//
//  The ScaleAndOffset and CopyPixels routines will reallocate dst if it
//  doesn't conform in shape to src.  So will BandSelect, except that the
//  number of bands in src and dst is allowed to differ (if dst is
//  unitialized, it will be set to a 1-band image).
//
// PARAMETERS
//  src                 source image
//  dst                 destination image
//  scale               floating point scale value  (1.0 = no change)
//  offset              floating point offset value (0.0 = no change)
//  sBand               source band (0...)
//  dBand               destination band (0...)
//
// SEE ALSO
//  Convert.cpp         implementation
//  Image.h             image class definition
//
// Copyright © Richard Szeliski, 2001.
// See Copyright.h for more details
//
///////////////////////////////////////////////////////////////////////////

template <class T1, class T2>
void ScaleAndOffsetLine(T1* src, T2* dst, int n,
                        float scale, float offset,
                        T2 minVal, T2 maxVal)
{
    // This routine does NOT round values when converting from float to int
    const bool scaleOffset = (scale != 1.0f) || (offset != 0.0f);
    const bool clip = (minVal < maxVal);

    if (scaleOffset)
        for (int i = 0; i < n; i++)
        {
            float val = src[i] * scale + offset;
            if (clip)
                val = __min(__max(val, minVal), maxVal);
            dst[i] = (T2) val;
        }
    else if (clip)
        for (int i = 0; i < n; i++)
        {
            dst[i] = (T2) __min(__max(src[i], minVal), maxVal);
        }
    else if (typeid(T1) == typeid(T2))
        memcpy(dst, src, n*sizeof(T2));
    else
        for (int i = 0; i < n; i++)
        {
            dst[i] = (T2) src[i];
        }
}

template <class T1, class T2>
void ScaleAndOffset(CImageOf<T1>& src, CImageOf<T2>& dst,
                    float scale, float offset);

template <class T1, class T2>
void CopyPixels(CImageOf<T1>& src, CImageOf<T2>& dst)
{
    ScaleAndOffset(src, dst, 1.0f, 0.0f);
}

template <class T>
CImageOf<T> ConvertToRGBA(CImageOf<T> src);

template <class T>
static CImageOf<T> ConvertToGray(CImageOf<T> src)
{
    // Check if already gray
    CShape sShape = src.Shape();
    if (sShape.nBands == 1)
        return src;

    // Make sure the source is a color image
    if (sShape.nBands != 4 || src.alphaChannel != 3)
        throw CError("ConvertToGray: can only convert from 4-band (RGBA) image");

    // Allocate the new image
    CShape dShape(sShape.width, sShape.height, 1);
    CImageOf<T> dst(dShape);

    // Process each row
    T minVal = dst.MinVal();
    T maxVal = dst.MaxVal();
    for (int y = 0; y < sShape.height; y++)
    {
        T* srcP = &src.Pixel(0, y, 0);
        T* dstP = &dst.Pixel(0, y, 0);
        for (int x = 0; x < sShape.width; x++, srcP += 4, dstP++)
        {
            RGBA<T>& p = *(RGBA<T> *) srcP;
            // OLD FORMULA: float Y = (float)(0.212671 * p.R + 0.715160 * p.G + 0.072169 * p.B);
	    // Changed to commonly used formula 6/4/07 DS
            float Y = (float)(0.299 * p.R + 0.587 * p.G + 0.114 * p.B);
            *dstP = (T) __min(maxVal, __max(minVal, Y));
        }
    }
    return dst;
}

template <class T>
void BandSelect(CImageOf<T>& src, CImageOf<T>& dst, int sBand, int dBand);
