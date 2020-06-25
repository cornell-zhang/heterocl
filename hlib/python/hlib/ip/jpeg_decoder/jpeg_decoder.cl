// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#pragma OPENCL EXTENSION cl_intel_channels : enable
#include "../host/inc/jpeg.h"

struct entropyData {
   uchar bits;
   uchar bitsToReset;
   uint data;
   bool keep_going;
};

struct huffmanContext {
   short data[16];
   int output;
   short cols;
   uint tile;
   uchar downsample;
   bool eob;
};

struct DCTContext {
   short data[16];
   short cols;
   uint tile;
   uchar downsample;
   int output;
};

channel struct entropyData entropy_without_markers[COPIES];
channel struct huffmanContext results[COPIES] __attribute__((depth(512)));
channel struct DCTContext toDCT __attribute__((depth(16)));
channel uint config[COPIES];

// Order of Huffman tables
#define HDC0 0
#define HDC1 1
#define HAC0 2
#define HAC1 3
#define QTAB 4


void swapUshort(ushort *a, ushort *b) {
   ushort tmp = *a;
   *a = *b;
   *b = tmp;
}

void swapUshort16(ushort16 *a, ushort16 *b) {
   ushort16 tmp = *a;
   *a = *b;
   *b = tmp;
}

void swapUchar(uchar *a, uchar *b) {
   uchar tmp = *a;
   *a = *b;
   *b = tmp;
}

unsigned char clip(const int x) {
    return (x < 0) ? 0 : ((x > 0xFF) ? 0xFF : (unsigned char) x);
}

#define W1 2841
#define W2 2676
#define W3 2408
#define W5 1609
#define W6 1108
#define W7 565

void rowIDCT(short* blk, local short *blkout) {
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
    x1 = blk[4] << 11;
    x2 = blk[6];
    x3 = blk[2];
    x4 = blk[1];
    x5 = blk[7];
    x6 = blk[5];
    x7 = blk[3];
    x0 = (blk[0] << 11) + 128;
    x8 = W7 * (x4 + x5);
    x4 = x8 + (W1 - W7) * x4;
    x5 = x8 - (W1 + W7) * x5;
    x8 = W3 * (x6 + x7);
    x6 = x8 - (W3 - W5) * x6;
    x7 = x8 - (W3 + W5) * x7;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2);
    x2 = x1 - (W2 + W6) * x2;
    x3 = x1 + (W2 - W6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    blkout[0] = (x7 + x1) >> 8;
    blkout[1] = (x3 + x2) >> 8;
    blkout[2] = (x0 + x4) >> 8;
    blkout[3] = (x8 + x6) >> 8;
    blkout[4] = (x8 - x6) >> 8;
    blkout[5] = (x0 - x4) >> 8;
    blkout[6] = (x3 - x2) >> 8;
    blkout[7] = (x7 - x1) >> 8;
}

void colIDCT(local short* blk, uchar *out) {
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
    x1 = blk[8*4] << 8;
    x2 = blk[8*6];
    x3 = blk[8*2];
    x4 = blk[8*1];
    x5 = blk[8*7];
    x6 = blk[8*5];
    x7 = blk[8*3];
    x0 = (blk[0] << 8) + 8192;
    x8 = W7 * (x4 + x5) + 4;
    x4 = (x8 + (W1 - W7) * x4) >> 3;
    x5 = (x8 - (W1 + W7) * x5) >> 3;
    x8 = W3 * (x6 + x7) + 4;
    x6 = (x8 - (W3 - W5) * x6) >> 3;
    x7 = (x8 - (W3 + W5) * x7) >> 3;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2) + 4;
    x2 = (x1 - (W2 + W6) * x2) >> 3;
    x3 = (x1 + (W2 - W6) * x3) >> 3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    *out = clip(((x7 + x1) >> 14) + 128);
    *(out + 1) = clip(((x3 + x2) >> 14) + 128); 
    *(out + 2) = clip(((x0 + x4) >> 14) + 128); 
    *(out + 3) = clip(((x8 + x6) >> 14) + 128); 
    *(out + 4) = clip(((x8 - x6) >> 14) + 128); 
    *(out + 5) = clip(((x0 - x4) >> 14) + 128); 
    *(out + 6) = clip(((x3 - x2) >> 14) + 128); 
    *(out + 7) = clip(((x7 - x1) >> 14) + 128); 
}

constant char ZZinv[64] = {0, 1, 5, 6, 14, 15, 27, 28, 
                       2, 4, 7, 13, 16, 26, 29, 42, 
                       3, 8, 12, 17, 25, 30, 41, 43, 
                       9, 11, 18, 24, 31, 40, 44, 53, 
                       10, 19, 23, 32, 39, 45, 52, 54,
                       20, 22, 33, 38, 46, 51, 55, 60,
                       21, 34, 37, 47, 50, 56, 59, 61,
                       35, 36, 48, 49, 57, 58, 62, 63};

kernel void Arbiter() {
   uchar source = 0;
   uchar left = 0;
   bool sendEOBs = false;
   struct huffmanContext hc;
   while (true) {
      bool success = false;
      if (sendEOBs) {
         #pragma unroll
         for (int i = 0; i < 16; i++) hc.data[i] = 0;
      } else {
         #pragma unroll
         for (int i = 0; i < COPIES; i++) {
            if (i == source) {
               hc = read_channel_nb_intel(results[i], &success);
            }
          }
      }
      struct DCTContext dc = {
         .tile = hc.tile,
         .cols = hc.cols,
         .output = hc.output,
         .downsample = hc.downsample
      };
      #pragma unroll
      for (int i = 0; i < 16; i++) dc.data[i] = hc.data[i];
      if (!success && left == 0) source++;
      if (success || sendEOBs) {
         write_channel_intel(toDCT, dc);
         if (left == 0) {
            left = 23;
         } else if (left == 1) {
            left = 0;
            source++;
         } else left--;
         sendEOBs = (hc.eob || sendEOBs) && (left & 0x03);
      }
      
      if (source == COPIES) source = 0;
   }
}

// process DCT in 4 cycles per component, multiplex all huffmans
// process tiles of (3 x 2) x (8x8) blocks or (4 + 1 + 1) x (8x8)
// In the first scenario the result is a 16 x 8 pixel block, the second leads to 16 x 16 pixel block
// Commit 16 pixels per cycle to memory (48 bytes) should lead to decent memory efficiency
//    8 / 24 cycles are active - enough margin for unaligned accesses

// Channels should have room for all the other components 11 x 4 x 6 depth, round up to 512

// local work size - 24 threads processing one tile
__attribute__((reqd_work_group_size(24, 1, 1)))
kernel void DCTandRGB(global uchar *output, uchar write) {
   struct DCTContext dc = read_channel_intel(toDCT);
   uint tilesPerRow = (dc.cols + 15) / 16;
   local short inputBlocks[24][16];
   local short blocks[48][8];
   local uchar comp[24][8], comp2[24][8];
   local uchar comp3[24][8], comp4[24][8];
   local uchar final[48 * 16];
   uint inTile = get_local_id(0);
   uint tile = dc.tile / 6;
   uint tileRow = tile / tilesPerRow;
   if (dc.downsample) tileRow *= 2;
   uint tileCol = tile % tilesPerRow;
   #pragma unroll
   for (int i = 0; i < 16; i++) {
      inputBlocks[inTile][i] = dc.data[i];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   short dctIn[16];
   #pragma unroll
   for (int i = 0; i < 16; i++) {
      ushort loc = (inTile / 4) * 64 + ZZinv[(inTile % 4) * 16 + i];
      dctIn[i] = inputBlocks[loc / 16][loc % 16];
   }
   rowIDCT(dctIn, &blocks[inTile * 2][0]);
   rowIDCT(dctIn + 8, &blocks[inTile * 2 + 1][0]);
   barrier(CLK_LOCAL_MEM_FENCE);
   uchar temp[8], temp2[8];
   colIDCT(&blocks[(inTile / 4) * 8][(inTile % 4) * 2], temp);
   colIDCT(&blocks[(inTile / 4) * 8][(inTile % 4) * 2 + 1], temp2);
   #pragma unroll
   for (int i = 0; i < 8; i++) {
      comp[inTile][i] = temp[i];
      comp2[inTile][i] = temp2[i];
      comp3[inTile][i] = temp[i];
      comp4[inTile][i] = temp2[i];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   if (dc.downsample ? inTile < 16 : inTile < 8) {
      uchar outData[48];
      #pragma unroll
      for (int i = 0; i < 16; i++) {
         int y, cb, cr;
         if (!(dc.downsample & 0x02)) {
            y = (int)(i % 2 ? comp2[(i / 8) * 12 + (i % 8) / 2][inTile] : comp[(i / 8) * 12 + (i % 8) / 2][inTile]) << 8;
            cb = (int)(i % 2 ? comp2[(i / 8) * 12 + (i % 8) / 2 + 4][inTile] : comp[(i / 8) * 12 + (i % 8) / 2 + 4][inTile]) - 128;
            cr = (int)(i % 2 ? comp2[(i / 8) * 12 + (i % 8) / 2 + 8][inTile] : comp[(i / 8) * 12 + (i % 8) / 2 + 8][inTile]) - 128;
         }
         if (dc.downsample) {
            y = (int)(i % 2 ? comp4[(inTile / 8) * 8 + (i % 16) / 2][inTile % 8] : comp3[(inTile / 8) * 8 + (i % 16) / 2][inTile % 8]) << 8;
            cb = (int)(((i / 2) % 2) ? comp4[i / 4 + 16][inTile / 2] : comp3[i / 4 + 16][inTile / 2]) - 128;
            cr = (int)(((i / 2) % 2) ? comp4[i / 4 + 20][inTile / 2] : comp3[i / 4 + 20][inTile / 2]) - 128;
         } 
         outData[3 * i] = clip((y +            359 * cr + 128) >> 8);
         outData[3 * i + 1] = clip((y -  88 * cb - 183 * cr + 128) >> 8);
         outData[3 * i + 2] = clip((y + 454 * cb +            128) >> 8);
      }
      int localLoc = inTile * 48;
      #pragma unroll
      for (int i = 0; i < 48; i++) {
         if (write && (tileCol * 16 + i / 3 < dc.cols)) {
            final[localLoc + i] = outData[i];
         }
      }
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   int locBase = (tileRow * tilesPerRow + tileCol * (dc.downsample ? 2 : 1)) * 6 * 64;
   if (dc.downsample ? inTile < 12 : inTile < 6) {
      #pragma unroll
      for (int i = 0; i < 64; i++) {
         output[dc.output + locBase + inTile * 64 + i] = final[inTile * 64 + i];
      }
   }
}

// pointer to entropy data
// pass an approximation of the data size
// this will identify the marker and stop processing accordingly
void read_entropy(global uint * restrict images, int offset, int size, int copy) {
   uchar bytes[8];
   bool altered = false;
   bool done = false;
   int j = 0;
   for (int i = 0; i < size + 1; i++) {
      uint data = images[offset + i];
      uchar outdata[4] = {0, 0, 0, 0};
      bytes[4] = data;
      bytes[5] = data >> 8;
      bytes[6] = data >> 16;
      bytes[7] = data >> 24;
      if (i > 0) {
         if (j < 2048 * 4) {
            write_channel_intel(config[copy], bytes[3] << 24 | bytes[2] << 16 | bytes[1] << 8 | bytes[0]);
            j++;
         } else {
            uchar index = 0;
            uchar bytesToReset = 0xF;
            // Analyze what can be sent from 0 ... 3
            #pragma unroll
            for (int j = 0; j < 4; j++) {
               bool isMarker = (bytes[j] == 0xFF) && ((bytes[j + 1] == 0x00) || (bytes[j + 1] == 0xFF));
               if (!isMarker || done) {
                  uchar towrite = (altered | done) ? 0xFF : bytes[j];
                  #pragma unroll
                  for (int k = 0; k < 4; k++) if (index == k) outdata[k] = towrite;
                  index++;
               }
               if (altered && ((bytes[j] & 0xD0) == 0xD0)) {
                  bytesToReset = index;
               }
               done |= altered && (((bytes[j] & 0xD8) != 0xD0) && (bytes[j] != 0x00) && (bytes[j] != 0xFF));
               altered = bytes[j] == 0xFF;
            }   
            if (index == 0) bytes[0] = 0;
            if (index < 2) bytes[1] = 0;
            if (index < 3) bytes[2] = 0;
            if (index < 4) bytes[3] = 0;
            struct entropyData toSend = {
               .bits = index * 8,
               .data = outdata[0] << 24 | outdata[1] << 16 | outdata[2] << 8 | outdata[3],
               .bitsToReset = bytesToReset * 8,
               .keep_going = !done
            };
            write_channel_intel(entropy_without_markers[copy], toSend);
            if (done) {
               j = 0;
               done = false;
               altered = false;
            }
         }
      } 
      #pragma unroll
      for (int j = 0; j < 4; j++) bytes[j] = bytes[j + 4];
   }
}

void huffmanDecoder(int copy) {
   // The codes are folded (hashed)
   ushort decHufflow[2048 * 4]; // decode memory
   ushort decHuffhigh[512 * 4];
   uchar qTab[4 * 64]; // quantization table

   int outputOffset;
   short cols;
   short options;

   bool compTables[6] = {false, true, true, false, true, true}; // marks the table used by each component
   bool shiftDC[6] = {true, true, true, true, true, true};

   short prevDC[3] = {0, 0, 0}; // the differential value for the DC coefficient
   bool ac = false;
   bool haveReserve = false;
   bool keep_going = true;
   ulong window = 0;

   uchar pos = 0;
   for (ushort i = 0; i < 2048 * 4; i++) {
      uint tmp = read_channel_intel(config[copy]);
      decHufflow[i] = tmp >> 16;
      if (i < 2048) {
         decHuffhigh[i] = tmp;
      } 
      if (i >= 2048 && i < 2048 + 256) {
         qTab[i - 2048] = tmp;
      } 
      if (i == 2048 * 2) {
         cols = tmp;
      } 
      if (i == 2048 * 2 + 1) {
         options = tmp;
      } 
      if (i == 2048 * 2 + 2) {
         outputOffset = tmp & 0xFFFF;
      } 
      if (i == 2048 * 2 + 3) {
         outputOffset = (tmp << 16) | outputOffset;
      } 
   }

   if (options & 0x01) {
      compTables[1] = compTables[2] = false;
      shiftDC[0] = shiftDC[1] = shiftDC[2] = false;
   }

   uchar decodedLength = 0;
   uchar bits = 0;
   uchar bitsToReset;


   struct reserveData {
      uchar bits;
      uchar bitsToReset;
      uint data[3];
      bool keep_going;
   } reserve;

   int frame = 0;
   short block[16];
   #pragma unroll
   for (int i = 0; i < 16; i++) block[i] = 0;
   uint tile = 0;
   ushort fetch1 = 0, fetch2 = 0;
   bool temp_keep_going = true;
   while(keep_going) {
      bits &= 0x3F;
      decodedLength &= 0x1F;
      window <<= decodedLength;
      ulong temp = ((ulong)reserve.data[1] << 32) | reserve.data[0];
      ulong temp2 = ((ulong)reserve.data[2] << 32) | reserve.data[1];
      temp <<= decodedLength;
      temp2 <<= decodedLength;
      reserve.data[2] = (uint)(temp2 >> 32);
      reserve.data[1] = (uint)(temp >> 32);
      reserve.data[0] = temp;
      bits -= decodedLength;
      bits &= 0x3F;
      bitsToReset -=decodedLength;
      bitsToReset &= 0x3F;
      if (bitsToReset == 0) keep_going = temp_keep_going;
      if (!(bits & 0x20) && haveReserve) {
         window |= ((ulong)reserve.data[2] << 32) | (ulong)reserve.data[1];
         if (!((reserve.bitsToReset >> 6) & 0x01)) bitsToReset = bits + reserve.bitsToReset;
         bits += reserve.bits;
         bits &= 0x3F;
         temp_keep_going = reserve.keep_going;
         haveReserve = false;
      }
      if (!haveReserve) {
         // When a reset or done follows, the sender transmists 0xFFFF which is an
         // illegal huffman symbol, together with the actual number of bits until reset
         struct entropyData entropy = read_channel_nb_intel(entropy_without_markers[copy], &haveReserve);
         reserve.data[2] = (ulong)entropy.data >> bits;
         reserve.data[1] = ((ulong)entropy.data << 32) >> bits;
         reserve.data[0] = ((ulong)entropy.data << (63 - bits)) << 1;
         reserve.bitsToReset = entropy.bitsToReset;
         reserve.bits = entropy.bits;
         reserve.keep_going = entropy.keep_going;
      }
 
      bool validIteration = bits & 0x20 || !temp_keep_going;

      ushort header = window >> 48;
      ushort fetch1 = decHufflow[(header >> 5) | (ac << 12) | (compTables[0] << 11)];
      ushort fetch2 = decHuffhigh[(header & 0x1FF) | (ac << 10) | (compTables[0] << 9)];

      ushort decoded = ((fetch1 >> 11) & 0x01) ? fetch2 : fetch1;
      uchar totalBits = decoded & 0x1F;
      uchar huffmanBits = (decoded >> 6) & 0x1F;
      uchar runLength = decoded >> 12;
      uchar symbolBits = totalBits - huffmanBits;
      bool reset = false;

      decodedLength = 0;

      if (validIteration) {
         if ((header & 0xFFFF) == 0xFFFF) {
            decodedLength = bitsToReset;
            ac = false;
            validIteration = false;
            prevDC[0] = prevDC[1] = prevDC[2] = 0;
         } else {
            decodedLength = totalBits;
         }
         if (ac) {
            pos += 1 + runLength;
         } else {
            pos = 0;
         } 
      } 
     
      bool eob = ac && ((symbolBits | runLength) == 0);
      bool newBlock = validIteration && (eob || pos == 63);

      uint headerSymbol = (window << huffmanBits) >> 48;

      ushort result = (headerSymbol << symbolBits) >> 16;
      // Must make this a signed integer
      // If negative, must place leading ones, add one more 1, and set the sign
      // bit
      if ((headerSymbol & 0x8000) && (symbolBits > 0)) {
      } else {
         result ^= 0xFFFF << symbolBits;
         result++;
      }

      uchar writeWhere = pos;
      short writeWhat = ((short)result) * qTab[pos | (compTables[0] << 6)];
      if (validIteration) {
         if (!ac) {
            writeWhat += prevDC[0];
            prevDC[0] = writeWhat;
         }
         if ((writeWhere & 0x0F) == 0x0F) block[15] = writeWhat;
         if (newBlock || ((writeWhere >> 4) != frame) || ((writeWhere & 0x0F) == 0x0F)) {
            struct huffmanContext hc = {
               .cols = cols,
               .output = outputOffset,
               .tile = tile,
               .downsample = options,
               .eob = newBlock
            };
            #pragma unroll
            for (int i = 0; i < 16; i++) {
               hc.data[i] = block[i];
            }
            write_channel_intel(results[copy], hc);
            #pragma unroll
            for (int i = 0; i < 16; i++) {
               block[i] = 0;
            }
         }
         if (!eob) {
            #pragma unroll
            for (uchar i = 0; i < 15; i++) {
               if ((writeWhere & 0x0F) == i) {
                  block[i] = writeWhat;
               }
         }
            frame = (writeWhere + 1) >> 4;
         }

         if (!ac) ac = true;
      }

      if (newBlock) {
         frame = 0;
         ac = false;
         if (shiftDC[0]) {
            int prevTemp = prevDC[0]; prevDC[0] = prevDC[1]; prevDC[1] = prevDC[2]; prevDC[2] = prevTemp;
         }
         // Rotate the component table
         bool tmpComp = compTables[0];
         bool tmpShift = shiftDC[0];
         #pragma unroll
         for (int i = 0; i < 5; i++) {
            compTables[i] = compTables[i + 1];
            shiftDC[i] = shiftDC[i + 1];
         }
         compTables[5] = tmpComp;
         shiftDC[5] = tmpShift;
      }
      tile = newBlock ? tile + 1 : tile;
   }
}

#define DECODER_PAIR(copy) \
\
kernel void huffmanDecoder ## copy () { \
   huffmanDecoder(copy); \
} \
\
kernel void read_entropy ## copy (global uint * restrict entropy, int offset, int size) { \
   read_entropy(entropy, offset, size, copy); \
}

   DECODER_PAIR(0)
#if COPIES > 1
   DECODER_PAIR(1)
#endif
#if COPIES > 2
   DECODER_PAIR(2)
#endif
#if COPIES > 3
   DECODER_PAIR(3)
#endif
#if COPIES > 4
   DECODER_PAIR(4)
#endif
#if COPIES > 5
   DECODER_PAIR(5)
#endif
#if COPIES > 6
   DECODER_PAIR(6)
#endif
#if COPIES > 7
   DECODER_PAIR(7)
#endif
#if COPIES > 8
   DECODER_PAIR(8)
#endif
#if COPIES > 9
   DECODER_PAIR(9)
#endif
#if COPIES > 10
   DECODER_PAIR(10)
#endif
#if COPIES > 11
   DECODER_PAIR(11)
#endif

