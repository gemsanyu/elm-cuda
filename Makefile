CPP= g++
CPPFLAGS= -g -O3 -march=native --std=c++11
NVCC= nvcc
NVCCFLAGS= -g -G -std=c++11
LCUDAFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64  -lcudart -lcuda -lcublas -lcusolver
MPICC= mpic++


all: main

main: main.o local_lib.o
	$(CPP) $(CPPFLAGS) $? -o $@ $(LCUDAFLAGS)

main.o: main.cu
	$(NVCC) $(NVCCFLAGS) -c $? -o $@

local_lib.o: local_lib.cu
	$(NVCC) $(NVCCFLAGS) -c $? -o $@

clean:
	rm -rf *.o main
