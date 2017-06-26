#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
  #include "defs.h"
  #include "ppm.h"
  #include "defs.h" 
}

void launch_complex_kernel(int, int, int, int, rgb_pixel*, rgb_pixel**);
void launch_motion_kernel(int, int, int, int, rgb_pixel*, rgb_pixel**);

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
void generate_grad_image(int height, int width, int blocks, int threads, rgb_pixel** d_img_ref) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  printf("Launching image generation kernel...\n");
  rgb_pixel* d_image;
  int num_pxls = height * width;
  int size = num_pxls * sizeof(rgb_pixel);
  cudaMalloc((void**) &d_image, size);
  
  dim3 blk(ceil(height/32.0), ceil(width/32.0));
  dim3 thd(32, 32);

  cudaEventRecord(start);
  generate_gradient_kernel<<<blk, thd>>>(height, width, d_image);
  cudaEventRecord(stop);
  
  cudaEventSynchronize(stop);
  float ms = 0.0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Kernel execution time: %f\n", ms);
  
  *d_img_ref = d_image;
}

int main(int argc, char** argv) {
  if(argc < 6) {
    printf("Expected args for image dim, grid x dim, grid y dim, block dim, and mode.\n");
    exit(1);
  }
  
  int dim, blocksX, blocksY, threads;
  dim = atoi(argv[1]);
  blocksX = atoi(argv[2]);
  blocksY = atoi(argv[3]);
  threads = atoi(argv[4]);
  char mode = argv[5][0];
  
  rgb_pixel* h_img, *d_img;
  generate_grad_image(dim, dim, blocksX, threads, &d_img);

  //h_img = (rgb_pixel*)malloc(dim*dim*sizeof(rgb_pixel));
  //cudaMemcpy(h_img, d_img, sizeof(rgb_pixel) * dim*dim, cudaMemcpyDeviceToHost);
  //write_ppm("actual.ppm", h_img, dim, dim);
  

  if(mode == 'c') {
    launch_complex_kernel(blocksX, blocksY, threads, dim, d_img, &h_img);
  } else if (mode == 'm') {
    launch_motion_kernel(blocksX, blocksY, threads, dim, d_img, &h_img);
  } else {
    exit(1);
  }

  //char img_file[255];
  //sprintf(img_file, "%d.%d.%d.%d.%c.ppm", dim, blocksX, blocksY, threads, mode);
  //write_ppm(img_file, h_img, dim, dim);

  //free(h_img);
  return 0;
}
