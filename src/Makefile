include ../Makefile.inc

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(CU_OBJS)
	@echo "Building Clumper Executable ...\n"
	-mkdir -p ../bin
	$(CC) $(CPP_FLAGS) -o $@ $^ $(LD_FLAGS) 

$(OBJS): $(SRCS)
	@echo "Compiling Clumper CPP sources ...\n"
	-mkdir -p ../bin
	$(CC) $(CPP_FLAGS) -o $@ -c $^ $(LIB_INCLUDES) 

$(CU_OBJS): $(CU_SRCS)
	@echo "Compiling Clumper CUDA Kernels ...\n"
	-mkdir -p ../bin
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $^ $(LIB_INCLUDES)

clean:
	rm -fr ../bin

.PHONY: all clean
