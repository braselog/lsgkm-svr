# GPU Acceleration for lsgkm-svr

This document describes the GPU acceleration features added to the lsgkm-svr package for enhanced performance in RBF kernel computations.

## Overview

The GPU acceleration is implemented using CUDA and provides significant speedup for RBF (Radial Basis Function) kernel computations, particularly beneficial for:

- Multi-Kernel Learning (MKL) with combined GKM and RBF kernels
- Large batch processing
- Training and prediction with RBF-only kernels
- Kernel matrix optimization

## Features

### 1. Automatic GPU Detection and Fallback
- Automatically detects CUDA-capable GPUs at runtime
- Falls back gracefully to CPU computation if GPU is unavailable
- No changes required to existing command-line interfaces

### 2. Optimized RBF Kernel Computation
- **SIMD Acceleration**: Uses AVX2/SSE2 instructions for CPU computation
- **GPU Batch Processing**: Efficient CUDA kernels for large batches
- **Adaptive Processing**: Automatically chooses GPU or CPU based on batch size
- **Memory Management**: Intelligent GPU memory allocation and caching

### 3. Multi-threaded CPU Fallback
- Parallel CPU computation when GPU is not available or not efficient
- Thread pool management for optimal CPU utilization
- SIMD-optimized distance calculations

### 4. Hybrid GPU/CPU Processing
- Large batches are split into GPU-optimal chunks
- Small batches use CPU to avoid GPU overhead
- Automatic load balancing based on available resources

## Supported Kernel Types

The GPU acceleration is active for the following kernel types:
- `MKL_GKM_RBF`: Multi-kernel learning with GKM and RBF kernels
- `MKL_RBF_ONLY`: RBF-only kernels
- `EST_TRUNC_RBF`: Truncated estimation with RBF
- `EST_TRUNC_PW_RBF`: Positional weights with RBF
- `GKM_RBF`: GKM with RBF combination

## Building with GPU Support

### Prerequisites
- NVIDIA GPU with compute capability 6.0 or higher
- CUDA Toolkit 10.0 or later
- GCC/G++ compiler with C++11 support

### Build Instructions

#### Default build (with CUDA if available):
```bash
cd src/
make
```

#### Force CPU-only build:
```bash
make CUDA_ENABLED=0
```

#### Custom CUDA path:
```bash
make CUDA_PATH=/usr/local/cuda-11.8
```

#### Specific GPU architecture:
```bash
make CUDA_ARCH=sm_70  # For Tesla V100
make CUDA_ARCH=sm_80  # For A100
```

### Build Options

- `CUDA_ENABLED=1/0`: Enable/disable CUDA support (default: 1)
- `CUDA_PATH`: Path to CUDA installation (default: /usr/local/cuda)
- `CUDA_ARCH`: Target GPU architecture (default: sm_60)

## Performance Configuration

### Automatic Optimization
- **Batch Size Detection**: Automatically determines optimal batch sizes based on GPU memory
- **Memory Management**: Pre-allocates GPU memory to minimize transfer overhead
- **Kernel Launch Parameters**: Optimized CUDA grid and block sizes

### Manual Tuning
The system uses conservative defaults that work well for most hardware. Advanced users can modify these constants in `rbf_cuda.h`:

```c
#define CUDA_BLOCK_SIZE 256        // CUDA threads per block
#define CUDA_MIN_BATCH_SIZE 1000   // Minimum batch for GPU efficiency
```

## Runtime Behavior

### GPU Detection
```
CUDA Device Information:
========================
Number of devices: 1
Device 0: NVIDIA Tesla V100-SXM2-32GB
  Compute capability: 7.0
  Total memory: 31.75 GB

CUDA context initialized:
  Max batch size: 10000
  Max covariates: 1000
  GPU memory allocated: 156.25 MB
```

### Adaptive Processing
```
RBF batch computation: 5000 samples with 4 threads using AVX2 SIMD mode
Large batch (15000), processing in GPU chunks of 10000
Using GPU for RBF batch computation: n=10000, normalize=1
Using GPU for RBF batch computation: n=5000, normalize=1
```

### Fallback Scenarios
```
GPU computation failed, falling back to CPU
RBF batch computation: 1000 samples single-threaded using SSE2 SIMD mode
```

## Performance Expectations

### Typical Speedups
- **Small batches** (< 1000): CPU preferred (low GPU overhead)
- **Medium batches** (1000-10000): 2-5x speedup on modern GPUs
- **Large batches** (> 10000): 5-20x speedup depending on GPU and data characteristics

### Factors Affecting Performance
- **Batch Size**: Larger batches benefit more from GPU acceleration
- **Number of Covariates**: More features per sample improve GPU utilization
- **GPU Memory**: Larger GPU memory allows bigger batches
- **CPU vs GPU Balance**: Modern CPUs with AVX2 can be competitive for smaller workloads

## Troubleshooting

### Common Issues

#### CUDA Not Found During Build
```
CUDA not found at /usr/local/cuda, building CPU-only version
```
**Solution**: Install CUDA toolkit or specify correct path with `CUDA_PATH`

#### Runtime GPU Initialization Failure
```
Failed to initialize CUDA context, falling back to CPU
```
**Solution**: Check GPU memory availability, reduce batch size, or check CUDA driver

#### Out of Memory Errors
```
CUDA error: out of memory
```
**Solution**: The system will automatically fall back to CPU. Consider reducing dataset size or using a GPU with more memory.

### Debug Information
Enable debug logging to see detailed GPU/CPU decision making:
```bash
export CLOG_LEVEL=DEBUG
./gkmtrain [options]
```

## Architecture Details

### Code Organization
- `rbf_cuda.h`: CUDA interface declarations
- `rbf_cuda.cu`: CUDA kernel implementations
- `libsvm_gkm.c`: Integration and adaptive scheduling
- `Makefile`: Build system with CUDA support

### Memory Management
- **Host Memory**: Double-precision input/output arrays
- **Device Memory**: Single-precision computation (automatic conversion)
- **Caching**: Self-kernel values cached for normalized RBF computation
- **Streaming**: Large datasets processed in chunks to fit GPU memory

### Kernel Design
- **RBF Computation**: Vectorized squared distance calculation + exponential
- **Normalization**: Separate kernel for normalized RBF with cached self-kernels
- **Memory Coalescing**: Optimized memory access patterns for GPU efficiency

## Future Improvements

Potential enhancements for future versions:
- Support for multiple GPUs
- Custom precision options (FP16 for newer GPUs)
- Integration with other kernel types beyond RBF
- Automatic benchmark-based optimization
- Support for AMD GPUs via ROCm/HIP

## License

The GPU acceleration code maintains the same GPL v3 license as the original lsgkm-svr package.
