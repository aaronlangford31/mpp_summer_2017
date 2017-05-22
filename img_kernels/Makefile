CC=gcc
NVCC=nvcc
CFLAGS= -02

v1: driver.cu ppm.o
	$(NVCC) driver.cu ppm.o -o v1-driver.out

ppm.o: ppm.c ppm.h
	$(CC) ppm.c -c -O2 -Wall

clean: 
	-rm -f $(OBJS) driver core *~ *.o
