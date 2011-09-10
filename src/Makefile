include ../Makefile.inc

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) $(CU_OBJS)
	@echo "Building Clumper Executable ...\n"
	$(CC) $(CPP_FLAGS) $(LD_FLAGS) -o $@ $^

$(OBJS): $(SRCS)
	@echo "Compiling Clumper CPP sources ...\n"
	$(CC) $(CPP_FLAGS) $(LIB_INCLUDES) -o $@ -c $^

$(CU_OBJS): $(CU_SRCS)
	@echo "Compiling Clumper CUDA Kernels ...\n"
	$(NVCC) $(NVCC_FLAGS) $(LIB_INCLUDES) -o $@ -c $^

clean:
	rm -f ../bin/*

.PHONY: all clean
