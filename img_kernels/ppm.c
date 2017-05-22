#include <stdio.h>
#include <stdlib.h>
#include "defs.h"

void write_ppm(char* fileName, rgb_pixel* pixels, int height, int width) {
  FILE* f = fopen(fileName, "w");
  if(f == NULL) {
    printf("Error opening file '%s'. Aborting process.", fileName);
    exit(1);
  }

  fprintf(f, "P3\r\n");
  fprintf(f, "%d\r\n", width);
  fprintf(f, "%d\r\n", height);
  fprintf(f, "255\r\n");
  
  int i, j;
  for(i = 0; i < height; i++) {
    for(j = 0; j < width; j++) {
      rgb_pixel p = pixels[i*width + j];
      fprintf(f, "%u %u %u\t", p.r, p.g, p.b);
    }
    fprintf(f, "\n");
  }
  fclose(f);
}
