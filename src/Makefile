include ../Makefile.inc

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(CU_OBJS)
	@echo "Building Clumper Executable ...\n"
	-mkdir -p ../bin
	$(CC) $(CPP_FLAGS) $(LD_FLAGS) -o $@ $^

$(OBJS): $(SRCS)
	@echo "Compiling Clumper CPP sources ...\n"
	-mkdir -p ../bin
	$(CC) $(CPP_FLAGS) $(LIB_INCLUDES) -o $@ -c $^

$(CU_OBJS): $(CU_SRCS)
	@echo "Compiling Clumper CUDA Kernels ...\n"
	-mkdir -p ../bin
	$(NVCC) $(NVCC_FLAGS) $(LIB_INCLUDES) -o $@ -c $^

clean:
	rm -fr ../bin

.PHONY: all clean
