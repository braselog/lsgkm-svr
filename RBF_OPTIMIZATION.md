# RBF Kernel Optimization

This document describes the optimizations made to the RBF kernel computation in lsgkm-svr to improve performance.

## Optimizations Implemented

### 1. SIMD Vectorization
- Added SIMD-optimized squared distance calculation using AVX2 and SSE2 instructions
- Processes 4 double values at once with AVX2, 2 with SSE2
- Falls back to scalar implementation when SIMD is not available
- Enabled with `-march=native -mtune=native -msse2 -mavx2` compiler flags

### 2. Parallel RBF Computation
- Implemented multi-threaded RBF kernel batch computation using pthreads
- Uses existing `g_param_nthreads` threading infrastructure
- Parallelizes the computation across multiple CPU cores
- Thread pool automatically divides work among available threads

### 3. Batch Processing Optimization
- Replaced loop-based individual RBF kernel calls with batch processing
- Optimized all MKL (Multiple Kernel Learning) functions:
  - `gkmkernel_mkl_kernelfunc_batch()`
  - `gkmkernel_mkl_kernelfunc_batch_all()`
  - `gkmkernel_mkl_kernelfunc_batch_sv()`
- Pre-computes self-kernel values when normalization is needed

### 4. Single Kernel Call Optimization
- Replaced `rbf_kernel()` calls with `rbf_kernel_optimized()` in single computations
- Uses SIMD-optimized distance calculation even for single kernel evaluations

## Performance Improvements Expected

- **SIMD**: 2-4x speedup for squared distance calculations
- **Parallelization**: Near-linear speedup with number of CPU cores for batch operations
- **Batch Processing**: Reduced overhead from individual function calls
- **Combined**: Total speedup of 8-16x for RBF-heavy workloads on modern multi-core CPUs

## Usage

The optimizations are automatically enabled when using MKL kernel types:
- `MKL_GKM_RBF` (kernel type 7)
- `MKL_GKM_ONLY` (kernel type 8) 
- `MKL_RBF_ONLY` (kernel type 9)

To take advantage of threading, set the number of threads:
```bash
# Use 4 threads for computation
gkmtrain -T 4 -t 7 ...
```

## Compiler Requirements

- GCC 4.9+ or Clang 3.5+ for AVX2 support
- SSE2 support is available on most x86-64 processors
- OpenMP support for threading (`-fopenmp`)

## Backward Compatibility

All optimizations are backward compatible:
- Falls back to scalar computation if SIMD is not available
- Single-threaded execution if threading is disabled
- Original kernel functions remain unchanged for non-MKL kernels

## Technical Details

### SIMD Implementation
```c
static inline double simd_squared_distance(const double *a, const double *b, int n)
```
- Uses `_mm256_*` AVX2 intrinsics for 256-bit operations
- Uses `_mm_*` SSE2 intrinsics for 128-bit operations
- Handles non-aligned memory and edge cases

### Threading Implementation
```c
static void rbf_kernel_batch_parallel(const gkm_data *da, const gkm_data **db_array, 
                                     int n, double gamma, double *results, int normalize)
```
- Creates worker threads with `pthread_create()`
- Divides work into chunks based on `g_param_nthreads`
- Handles normalization with cached self-kernel values

### Memory Management
- Careful memory allocation and deallocation to prevent leaks
- Stack-allocated thread data structures where possible
- Efficient memory access patterns for cache performance
