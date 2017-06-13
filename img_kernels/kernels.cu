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

 * 30 May 2017
 * 
 * So the question that still is floating around after sleeping on this is whether a kernel
 * is better off doing a 1 pixel operation for every thread while leaving big fractions of
 * the kernel completely idle, or if a kernel is better off mapping work to every thread
 * available. That second option forces the kernel to launch some threads that
 * don't execute every instruction...
 */  
__global__
void complex_kernel(int dim, rgb_pixel* src, rgb_pixel* dest) {
  int c_stride = blockDim.x;
  int r_stride = gridDim.x;

  int i, j;
  for(i=blockIdx.x; i < dim; i+=r_stride) {
    for(j = threadIdx.x; j < dim; j+=c_stride) {
      rgb_pixel px = src[(i*dim) + j];
      px.r = px.g = px.b = ((int)px.r + (int)px.g + (int)px.b) / 3;

      int dest_r, dest_c;
      dest_r = (dim - j - 1);
      dest_c = (dim - i - 1);
      dest[(dest_r * dim) + dest_c] = px;
    }
  }
}

__host__
void launch_complex_kernel(int grid, int block, int dim, rgb_pixel* d_src, rgb_pixel** h_dest) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("Launching complex kernel...\n");
  printf("Blocks: %d \t Threads: %d\n", grid, block);
  rgb_pixel* d_dest;
  cudaMalloc((void**) &d_dest, sizeof(rgb_pixel) * dim * dim); 
 
  dim3 grd(grid);
  dim3 blk(block);

  cudaEventRecord(start);
  complex_kernel<<<grd, blk>>>(dim, d_src, d_dest);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Kernel execution time: %f\n", ms);

  *h_dest = (rgb_pixel*)malloc(sizeof(rgb_pixel) * dim * dim);
  cudaMemcpy(*h_dest, d_dest, sizeof(rgb_pixel) * dim * dim, cudaMemcpyDeviceToHost);

  cudaFree(d_src); cudaFree(d_dest);

}
