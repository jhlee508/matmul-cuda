TARGET=main

KERNEL_SRC := $(wildcard kernels/*.cu)
KERNEL_OBJS := $(patsubst kernels/%.cu, kernels/%.o, $(KERNEL_SRC))

OBJECTS=src/util.o src/matmul.o src/main.o $(KERNEL_OBJS)
INCLUDES=-I/usr/local/cuda/include/ -I./kernels -I./include

CXX=g++
CPPFLAGS=-std=c++11 -O3 -Wall -march=native -mavx2 -mno-avx512f -mfma -fopenmp $(INCLUDES)

NVCC=/usr/local/cuda/bin/nvcc
LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lm -lcudart -lcublas -lnvToolsExt
CUDA_ARCH=70 # for Volta 
NVCCFLAGS:=$(foreach opt, $(CPPFLAGS), -Xcompiler=$(opt)) -arch=sm_$(CUDA_ARCH)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

src/matmul.o: src/matmul.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

kernels/%.o: kernels/%.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

src/%.o: src/%.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -rf $(TARGET) $(OBJECTS)
