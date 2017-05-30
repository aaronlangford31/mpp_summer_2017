#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
  #include "defs.h"
  #include "ppm.h"
}

/* There needs to be some way to map the block idx, thd indx to a chunk of the picture
 * 
 * Under what circumstances would the kernel need to cover more than one pixel?
 *    num_pixels / (blocks * thds) > 1
 * we know num_pixels = dim * dim;
 * we know thds = blockDim.x
 * we know blocks = gridDim.x
 * 
 * So what happens with the following scenario?
 *  dim = 18
 *  thds = 32
 *  blocks = 8
 * 
 *  num_pixels = 18*18 = 324
 *  324 / (32 * 8) = ~1.26
 *  so... each thread covers 1.26 pixels? Options...
 *      Launch another kernel for the remainder
 *      Kernel takes care of ceil(1.26) pixels per thread
 *        But then I've launched 256 thds, but will only be using 162 of them.
 *        Maybe that's just a side effect of the images I choose
 *      Only some threads are taking 2 pixels per thread while others are taking 1
 *        This gets around dead threads, but still will have some kind of if block
 *        that is going to leave some parts of the grid idle for some instructions.
 *        Is this an improvement over the ceil(pixel/thread) strategy? ... yes.
 *          You are using all threads to execute the kernel and the only time a part of the
 *          grid is idle is within an if block statement. Contrast with the ceil strategy
 *          and get that same amount of grid entirely dead.
 * 
 * How then to decide which threads are taking the lower count and which threads are taking
 * the higher count?
 *  Loose algorithm for this:
 *    Choose n threads to execute ceil(pixel/thread) such that: 
 *      (num_pixels - n) / floor(pixel/thread) = 1 or very near 1
 *  Need to optimize this system of equations:
 *    2x + y >= num_pixels
 *    2x + y <= threads
 *
 *  ^^^ Seems a bit too complex ^^^
 *  why not just use mod?
 *    n = num_pixels % threads
 *    m = num_pixels - n
 *    n will be the threads that execute the ceil(pixel/thread) version
 *    m will be the threads that execute the floor(pixel/thread) version
 *    
 *  So now how to map a thread to its proper index, considering it now needs to know
 *  how many pixels the threads prior to it have taken?
 * 
 *  ^^^ Still too complex... ^^^
 *  Doesn't the number of threads per block you pick force the number of blocks you pick?
 *  And vise versa... pick 
 *  
 * .---------------.
 * |_|_|_|_|_|_|_|_|
 * |_|_|_|_|_|_|_|_|
 * |_|_|_|_|_|_|_|_|
 * |_|_|_|_|_|_|_|_|
 * |_|_|_|_|_|_|_|_|
 * |_|_|_|_|_|_|_|_|
 * |_|_|_|_|_|_|_|_|
 * | | | | | | | | |
 * .---------------.
 */  
__global__
void complex_kernel(int dim, rgb_pixel* src, rgb_pixel* dest) {
  // I just am going to assume that the tiles will always be squares
  int thread_dim = gridDim.x * blockDim.x;
  // Need to figure out how many pixels per row/col a thread needs to cover
  int num_pixs = ceil((float)dim / (float)thread_dim);
  
  int row = blockIdx.x * blockDim.x * num_pixs + threadIdx.x * num_pixs;
  int col = blockIdx.y * blockDim.y * num_pixs + threadIdx.x * num_pixs;

  int r, c;
  for(r = 0; r < num_pixs; r++) {
    int roff = r + row;
    if(roff < dim) {
      for(c = 0; c < num_pixs; c++) {
        if(coff < dim) {
          int coff = c + col;
          rgb_pixel p = src[roff * dim + coff];
          p.r = p.g = p.b = (char)(((unsigned)p.r + (unsigned)p.g + (unsigned)p.b) / 3);
          dest[(dim - coff - 1) * dim + (dim - roff - 1)] = p;
        }
      }
    }
  }
}
