include ../Makefile.inc

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS) clumper.cpp
	@echo "Building Clumper Executable ...\n"
	-mkdir -p ../bin
	$(CC) $(CPP_FLAGS) -o $@ $^ $(LIB_INCLUDES) $(LD_FLAGS) 

$(OBJS): $(SRCS)
	@echo "Compiling Clumper sources ...\n"
	-mkdir -p ../bin
	$(NVCC) $(NVCC_FLAGS) -o $@ -c $^ $(LIB_INCLUDES) $(LD_FLAGS)

clean:
	rm -fr ../bin

.PHONY: all clean
