CC=gcc
NVCC=nvcc

default: driver.cu kernels.cu ppm.o
	$(NVCC) driver.cu kernels.cu ppm.o --compiler-options='-Wall' -O2 -o driver.out

ppm.o: ppm.c ppm.h
	$(CC) ppm.c -c -O2 -Wall

clean: 
	-rm -f $(OBJS) driver core *~ *.o
