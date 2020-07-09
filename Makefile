CPP= g++
CPPFLAGS= -g -O3 --std=c++11
NVCC= nvcc
NVCCFLAGS= -g -G -std=c++11
LCUDAFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64  -lcudart -lcuda -lcublas -lcusolver
MPICC= mpic++
MPIINCLUDE = -I/usr/local/openmpi/include -L/usr/local/openmpi/lib -lmpi


all: main

main: main.o local_lib.o helper_lib.o
	$(MPICC) $(CPPFLAGS) $? -o $@ $(LCUDAFLAGS)

main.o: main.cu
	$(NVCC) $(MPIINCLUDE) $(NVCCFLAGS) -c $? -o $@

local_lib.o: local_lib.cu
	$(NVCC) $(NVCCFLAGS) -c $? -o $@

helper_lib.o: helper_lib.cpp
	$(CPP) $(CPPFLAGS) -c $? -o $@

clean:
	rm -rf *.o main
