CC := g++
NVCC := nvcc

CUDA_TOOLKIT_PATH ?= /usr/local/cuda
CUDA_SDK_PATH ?= /opt/CUDA_SDK

LIB_INCLUDES += -I/usr/include -I$(CUDA_TOOLKIT_PATH)/include
LD_FLAGS += -lrt -lgomp -largtable2 -lcudart -L$(CUDA_TOOLKIT_PATH)/lib64

ifdef DEBUG
  ifneq ($(DEBUG), 0)
    CPP_FLAGS += -D_DEBUG -O0 -g3 -pg -Wall -Wextra
    NVCC_FLAGS += -D_DEBUG -O0 -G -g -pg
  else
    CPP_FLAGS += -O3 -Wall -Wextra
    NVCC_FLAGS += -O3
  endif
else
    CPP_FLAGS += -O3 -Wall -Wextra
    NVCC_FLAGS += -O3
endif

ifdef GPU_ARCH
    NVCC_FLAGS += --gpu-architecture=compute_$(GPU_ARCH) --gpu-code=sm_$(GPU_ARCH)
endif

EXECUTABLE = ../bin/clumper

OBJS := \
../bin/clumper_kernels.o

SRCS := \
./clumper_kernels.cu

