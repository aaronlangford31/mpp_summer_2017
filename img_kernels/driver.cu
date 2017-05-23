#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
  #include "defs.h"
  #include "ppm.h"
}
__global__
void generate_simple_image_kernel(int height, int width, rgb_pixel* out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if(row >= height || col >= width) {
    return;
  }

  rgb_pixel px = { 255, 0, 0 };
  out[row * width + col] = px; 
}

__device__ 
int scale(int i, int from, int to) {
  return (int)(((double)i / from) * to);
}

__global__
void generate_gradient_kernel(int height, int width, rgb_pixel* out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if(row >= height || col >= width) {
    return;
  }
  
  rgb_pixel px = {
    scale(row, width, 255),\
    scale(abs(col - row), width, 255),\
    scale(row+col, 2*width, 255)\
  };
  out[row * width + col] = px;
}

__host__
void generate_simple_image(int height, int width, rgb_pixel** out) {
  rgb_pixel* d_image;
  int num_pxls = height * width;
  int size = num_pxls * sizeof(rgb_pixel);
  cudaMalloc((void**) &d_image, size);
  *out = (rgb_pixel*) malloc(size);

  dim3 blk(ceil(height/32.0), ceil(width/32.0));
  dim3 thd(32, 32);
  generate_simple_image_kernel<<<blk, thd>>>(height, width, d_image);
  
  cudaMemcpy(*out, d_image, size, cudaMemcpyDeviceToHost);
  cudaFree(d_image);
}

__host__
void generate_grad_image(int height, int width, rgb_pixel** out) {
  rgb_pixel* d_image;
  int num_pxls = height * width;
  int size = num_pxls * sizeof(rgb_pixel);
  cudaMalloc((void**) &d_image, size);
  *out = (rgb_pixel*) malloc(size);
  
  dim3 blk(ceil(height/32.0), ceil(width/32.0));
  dim3 thd(32,32);
  generate_gradient_kernel<<<blk, thd>>>(height, width, d_image);

  cudaMemcpy(*out, d_image, size, cudaMemcpyDeviceToHost);
  cudaFree(d_image);
}

int main(int argc, char** argv) {
  if(argc < 4) {
    printf("Expected args for height, width, and file name.\n");
    exit(1);
  }
  int h = atoi(argv[1]);
  int w = atoi(argv[2]);
  char* fname = argv[3];
  rgb_pixel* img;

#if MODE == 1
  generate_simple_image(h, w, &img);
#elif MODE == 2
  generate_grad_image(h, w, &img);
#else
  printf("This program was compiled incorrectly. Make sure to specify the MODE flag.\n");
  exit(1);
#endif
  write_ppm(fname, img, h, w);
  return 0;
}
