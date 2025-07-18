CXX ?= g++
NVCC ?= nvcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp -march=native -mtune=native -msse2 -mavx2
#CFLAGS = -Wall -Wconversion -g -fPIC
SHVER = 2

# CUDA configuration
CUDA_ENABLED ?= 1
CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH ?= sm_60

# Check if CUDA is available
ifeq ($(CUDA_ENABLED), 1)
    ifneq ($(wildcard $(CUDA_PATH)),)
        CUDA_AVAILABLE := 1
        CFLAGS += -DUSE_CUDA
        CUDA_CFLAGS = -O3 -arch=$(CUDA_ARCH) --ptxas-options=-v -Xcompiler -fPIC
        CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcurand
        CUDA_INCLUDES = -I$(CUDA_PATH)/include
        CUDA_OBJECTS = rbf_cuda.o
        $(info Building with CUDA support)
    else
        CUDA_AVAILABLE := 0
        CUDA_OBJECTS =
        $(info CUDA not found at $(CUDA_PATH), building CPU-only version)
    endif
else
    CUDA_AVAILABLE := 0
    CUDA_OBJECTS =
    $(info CUDA disabled, building CPU-only version)
endif

# Targets
all: gkmtrain gkmpredict gkmexplain
#all: gkmtrain gkmpredict gkmmatrix 

gkmexplain: gkmexplain.c libsvm.o libsvm_gkm.o $(CUDA_OBJECTS)
	$(CXX) $(CFLAGS) gkmexplain.c libsvm.o libsvm_gkm.o $(CUDA_OBJECTS) -o gkmexplain -lm -lpthread $(CUDA_LIBS)
gkmpredict: gkmpredict.c libsvm.o libsvm_gkm.o $(CUDA_OBJECTS)
	$(CXX) $(CFLAGS) gkmpredict.c libsvm.o libsvm_gkm.o $(CUDA_OBJECTS) -o gkmpredict -lm -lpthread $(CUDA_LIBS)
gkmtrain: gkmtrain.c libsvm.o libsvm_gkm.o $(CUDA_OBJECTS)
	$(CXX) $(CFLAGS) gkmtrain.c libsvm.o libsvm_gkm.o $(CUDA_OBJECTS) -o gkmtrain -lm -lpthread $(CUDA_LIBS)

#for debug
#gkmmatrix: gkmmatrix.c libsvm.o libsvm_gkm.o $(CUDA_OBJECTS)
#	$(CXX) $(CFLAGS) gkmmatrix.c libsvm.o libsvm_gkm.o $(CUDA_OBJECTS) -o gkmmatrix -lm -lpthread $(CUDA_LIBS)

libsvm.o: libsvm.cpp libsvm.h
	$(CXX) $(CFLAGS) -c libsvm.cpp
libsvm_gkm.o: libsvm_gkm.c libsvm_gkm.h rbf_cuda.h
	$(CXX) $(CFLAGS) $(CUDA_INCLUDES) -c libsvm_gkm.c

# CUDA compilation
ifeq ($(CUDA_AVAILABLE), 1)
rbf_cuda.o: rbf_cuda.cu rbf_cuda.h libsvm_gkm.h
	$(NVCC) $(CUDA_CFLAGS) $(CUDA_INCLUDES) -c rbf_cuda.cu
endif

clean:
	rm -f libsvm.o libsvm_gkm.o rbf_cuda.o gkmtrain gkmpredict gkmexplain gkmmatrix

install:
	cp gkmtrain gkmpredict gkmexplain ../bin

# Help target
help:
	@echo "Build options:"
	@echo "  make                   - Build all targets with default settings"
	@echo "  make CUDA_ENABLED=0    - Build without CUDA support"
	@echo "  make CUDA_PATH=/path   - Specify CUDA installation path"
	@echo "  make CUDA_ARCH=sm_XX   - Specify CUDA architecture (e.g., sm_60, sm_70, sm_80)"
	@echo "  make DEBUG=1           - Build with debug symbols and verbose output"
	@echo "  make clean             - Clean build artifacts"
	@echo "  make install           - Install binaries"
	@echo ""
	@echo "GPU Memory Management Features:"
	@echo "  The enhanced version includes adaptive batch sizing and memory monitoring"
	@echo "  Use --gpu-diag, --gpu-tune, and --gpu-reserve options for optimization"

# Debug build option
ifeq ($(DEBUG), 1)
    CFLAGS += -g -DDEBUG
    CUDA_CFLAGS += -g -G -DDEBUG
    $(info Building debug version with verbose GPU diagnostics)
endif

# Test target for GPU features
test-gpu: gkmtrain
	@echo "Testing GPU memory management features..."
	@if [ -f gkmtrain ]; then \
		echo "Running GPU diagnostics test..."; \
		./gkmtrain --gpu-diag --help > /dev/null 2>&1 && echo "✓ GPU diagnostics available" || echo "✗ GPU diagnostics not available"; \
	else \
		echo "✗ gkmtrain not built"; \
	fi

.PHONY: all clean install help test-gpu
