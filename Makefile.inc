CC := g++
NVCC := nvcc

LIB_INCLUDES += -I/usr/include -I/usr/local/cuda/include
LD_FLAGS += -lrt -lgomp -lsprng -lcudart -L/usr/local/cuda/lib64/
CPP_FLAGS += -O3
#CPP_FLAGS += -O0 -g3 -pg -Wall -Wextra
#NVCC_FLAGS += -D_DEBUG

EXECUTABLE = ../bin/clumper
OBJS := ../bin/clumper.o
SRCS := ./clumper.cpp

CU_OBJS := ../bin/clumper_kernels.o
CU_SRCS := ./clumper_kernels.cu
