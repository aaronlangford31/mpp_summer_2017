CC=gcc
NVCC=nvcc

simple: driver.cu ppm.o
	$(NVCC) -D MODE=1 driver.cu ppm.o --compiler-options='-Wall' -O2 -o smpl-driver.out

gradient: driver.cu ppm.o
	$(NVCC) -D MODE=2 driver.cu ppm.o --compiler-options='-Wall' -O2 -o grd-driver.out

ppm.o: ppm.c ppm.h
	$(CC) ppm.c -c -O2 -Wall

clean: 
	-rm -f $(OBJS) driver core *~ *.o
