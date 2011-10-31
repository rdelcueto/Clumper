CC := g++
NVCC := nvcc

LIB_INCLUDES += -I/usr/include -I/usr/local/cuda/include
LD_FLAGS += -lrt -lgomp -lcudart -L/usr/local/cuda/lib64/

ifeq ($(DEBUG), 1)
    CPP_FLAGS += -O0 -g3 -pg -Wall -Wextra
    NVCC_FLAGS += -D_DEBUG
else
    CPP_FLAGS += -O3
endif

ifeq ($(GPU_ARCH), 20)
    NVCC_FLAGS += --gpu-architecture=compute_20 --gpu-code=sm_20
endif

ifeq ($(GPU_ARCH), 11)
      NVCC_FLAGS += --gpu-architecture=compute_11 --gpu-code=sm_11
endif

EXECUTABLE = ../bin/clumper
OBJS := ../bin/clumper.o
SRCS := ./clumper.cpp

CU_OBJS := ../bin/clumper_kernels.o
CU_SRCS := ./clumper_kernels.cu
