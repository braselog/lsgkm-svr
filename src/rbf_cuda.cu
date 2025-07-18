/* rbf_cuda.cu
 *
 * CUDA implementation for RBF kernel computation acceleration
 * 
 * This file provides GPU acceleration for RBF (Radial Basis Function) kernel
 * computations used in the gkm-SVM implementation.
 */

#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Include the header and gkm_data structure
extern "C" {
#include "rbf_cuda.h"
#include "libsvm_gkm.h"
}

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// CUDA kernel for computing squared distances (non-normalized RBF)
__global__ void rbf_kernel_batch_kernel(const float *da_covariates, 
                                       const float *db_covariates,
                                       float *results, 
                                       int num_covariates,
                                       int batch_size,
                                       float gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    // Compute squared distance between da and db[idx]
    float sum = 0.0f;
    const float *db_ptr = db_covariates + idx * num_covariates;
    
    for (int i = 0; i < num_covariates; i++) {
        float diff = da_covariates[i] - db_ptr[i];
        sum += diff * diff;
    }
    
    // Compute RBF kernel: exp(-gamma * squared_distance)
    results[idx] = expf(-gamma * sum);
}

// CUDA kernel for normalized RBF computation
__global__ void rbf_kernel_normalized_kernel(const float *da_covariates,
                                           const float *db_covariates,
                                           float *results,
                                           const float *self_cache,
                                           int num_covariates,
                                           int batch_size,
                                           float gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    // Compute squared distance between da and db[idx]
    float sum = 0.0f;
    const float *db_ptr = db_covariates + idx * num_covariates;
    
    for (int i = 0; i < num_covariates; i++) {
        float diff = da_covariates[i] - db_ptr[i];
        sum += diff * diff;
    }
    
    // Compute cross kernel
    float cross = expf(-gamma * sum);
    
    // Normalize using cached self-kernels
    float self_a = self_cache[0];         // da self-kernel
    float self_b = self_cache[idx + 1];   // db[idx] self-kernel
    
    if (self_a * self_b > 0.0f) {
        results[idx] = cross / sqrtf(self_a * self_b);
    } else {
        results[idx] = 0.0f;
    }
}

// CUDA kernel for computing self-kernels (for normalization)
__global__ void rbf_self_kernel_batch_kernel(const float *db_covariates,
                                            float *self_cache,
                                            int num_covariates,
                                            int batch_size,
                                            float gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    // Self-kernel is always 1.0 for RBF (exp(-gamma * 0))
    // But we compute it explicitly for consistency
    self_cache[idx + 1] = 1.0f;  // Store at idx+1 (idx 0 is for da)
}

// Check if CUDA is available
extern "C" int cuda_is_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

// Get number of CUDA devices
extern "C" int cuda_get_device_count(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess) ? device_count : 0;
}

// Get device properties
extern "C" int cuda_get_device_properties(int device_id, char *name, size_t name_len,
                                        size_t *total_mem, int *major, int *minor) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    if (name && name_len > 0) {
        strncpy(name, prop.name, name_len - 1);
        name[name_len - 1] = '\0';
    }
    
    if (total_mem) *total_mem = prop.totalGlobalMem;
    if (major) *major = prop.major;
    if (minor) *minor = prop.minor;
    
    return 0;
}

// Print device information
extern "C" void cuda_print_device_info(void) {
    int device_count = cuda_get_device_count();
    
    printf("CUDA Device Information:\n");
    printf("========================\n");
    printf("Number of devices: %d\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        char name[256];
        size_t total_mem;
        int major, minor;
        
        if (cuda_get_device_properties(i, name, sizeof(name), &total_mem, &major, &minor) == 0) {
            printf("Device %d: %s\n", i, name);
            printf("  Compute capability: %d.%d\n", major, minor);
            printf("  Total memory: %.2f GB\n", (double)total_mem / (1024.0 * 1024.0 * 1024.0));
        }
    }
    printf("\n");
}

// Get available GPU memory
extern "C" size_t cuda_get_available_memory(void) {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    return (err == cudaSuccess) ? free_mem : 0;
}

// Estimate maximum batch size based on available memory
extern "C" int cuda_estimate_max_batch_size(int num_covariates) {
    size_t available_mem = cuda_get_available_memory();
    if (available_mem == 0) return 0;
    
    // Conservative estimate: 80% of available memory
    size_t usable_mem = (size_t)(available_mem * 0.8);
    
    // Memory required per sample:
    // - db_covariates: num_covariates * sizeof(float)
    // - results: sizeof(float)
    // - self_cache: sizeof(float)
    size_t mem_per_sample = num_covariates * sizeof(float) + 2 * sizeof(float);
    
    // Add memory for da_covariates (shared across batch)
    size_t fixed_mem = num_covariates * sizeof(float);
    
    int max_batch = (int)((usable_mem - fixed_mem) / mem_per_sample);
    
    // Ensure it's within reasonable bounds
    if (max_batch > 100000) max_batch = 100000;  // Upper limit
    if (max_batch < CUDA_MIN_BATCH_SIZE) return 0;  // Too small to be efficient
    
    return max_batch;
}

// Initialize CUDA context
extern "C" int cuda_init_context(cuda_context_t *ctx, int max_batch_size, int max_covariates) {
    if (!ctx) return -1;
    
    // Initialize context
    memset(ctx, 0, sizeof(cuda_context_t));
    
    // Check if CUDA is available
    if (!cuda_is_available()) {
        fprintf(stderr, "CUDA is not available\n");
        return -1;
    }
    
    // Set device
    CUDA_CHECK(cudaSetDevice(0));
    ctx->device_id = 0;
    
    // Estimate batch size if not provided
    if (max_batch_size <= 0) {
        max_batch_size = cuda_estimate_max_batch_size(max_covariates);
        if (max_batch_size <= 0) {
            fprintf(stderr, "Cannot determine suitable batch size for GPU\n");
            return -1;
        }
    }
    
    ctx->max_batch_size = max_batch_size;
    ctx->max_covariates = max_covariates;
    
    // Calculate memory requirements
    size_t da_size = max_covariates * sizeof(float);
    size_t db_size = max_batch_size * max_covariates * sizeof(float);
    size_t results_size = max_batch_size * sizeof(float);
    size_t cache_size = (max_batch_size + 1) * sizeof(float);  // +1 for da self-kernel
    
    ctx->allocated_size = da_size + db_size + results_size + cache_size;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&ctx->d_da_covariates, da_size));
    CUDA_CHECK(cudaMalloc(&ctx->d_db_covariates, db_size));
    CUDA_CHECK(cudaMalloc(&ctx->d_results, results_size));
    CUDA_CHECK(cudaMalloc(&ctx->d_self_cache, cache_size));
    
    ctx->is_initialized = 1;
    
    printf("CUDA context initialized:\n");
    printf("  Max batch size: %d\n", max_batch_size);
    printf("  Max covariates: %d\n", max_covariates);
    printf("  GPU memory allocated: %.2f MB\n", (double)ctx->allocated_size / (1024.0 * 1024.0));
    
    return 0;
}

// Cleanup CUDA context
extern "C" void cuda_cleanup_context(cuda_context_t *ctx) {
    if (!ctx || !ctx->is_initialized) return;
    
    if (ctx->d_da_covariates) {
        cudaFree(ctx->d_da_covariates);
        ctx->d_da_covariates = NULL;
    }
    
    if (ctx->d_db_covariates) {
        cudaFree(ctx->d_db_covariates);
        ctx->d_db_covariates = NULL;
    }
    
    if (ctx->d_results) {
        cudaFree(ctx->d_results);
        ctx->d_results = NULL;
    }
    
    if (ctx->d_self_cache) {
        cudaFree(ctx->d_self_cache);
        ctx->d_self_cache = NULL;
    }
    
    ctx->is_initialized = 0;
    ctx->allocated_size = 0;
}

// Convert double array to float array
static void double_to_float_array(const double *src, float *dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (float)src[i];
    }
}

// Convert float array to double array
static void float_to_double_array(const float *src, double *dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = (double)src[i];
    }
}

// Main CUDA RBF kernel batch computation
extern "C" int cuda_rbf_kernel_batch(cuda_context_t *ctx,
                                    const struct gkm_data *da,
                                    const struct gkm_data **db_array,
                                    int n, double gamma, double *results, int normalize) {
    if (!ctx || !ctx->is_initialized || !da || !db_array || !results || n <= 0) {
        return -1;
    }
    
    if (n > ctx->max_batch_size) {
        fprintf(stderr, "Batch size %d exceeds maximum %d\n", n, ctx->max_batch_size);
        return -1;
    }
    
    if (da->num_covariates > ctx->max_covariates) {
        fprintf(stderr, "Number of covariates %d exceeds maximum %d\n", 
                da->num_covariates, ctx->max_covariates);
        return -1;
    }
    
    // Validate that all samples have the same number of covariates
    for (int i = 0; i < n; i++) {
        if (db_array[i]->num_covariates != da->num_covariates) {
            fprintf(stderr, "Mismatched number of covariates at sample %d\n", i);
            return -1;
        }
    }
    
    int num_covariates = da->num_covariates;
    float gamma_f = (float)gamma;
    
    // Allocate host memory for data conversion
    float *h_da_covariates = (float*)malloc(num_covariates * sizeof(float));
    float *h_db_covariates = (float*)malloc(n * num_covariates * sizeof(float));
    float *h_results = (float*)malloc(n * sizeof(float));
    float *h_self_cache = NULL;
    
    if (!h_da_covariates || !h_db_covariates || !h_results) {
        fprintf(stderr, "Failed to allocate host memory\n");
        goto cleanup;
    }
    
    // Convert da covariates to float
    double_to_float_array(da->covariates, h_da_covariates, num_covariates);
    
    // Convert db_array covariates to float (interleaved format)
    for (int i = 0; i < n; i++) {
        double_to_float_array(db_array[i]->covariates, 
                             h_db_covariates + i * num_covariates, 
                             num_covariates);
    }
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(ctx->d_da_covariates, h_da_covariates, 
                         num_covariates * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_db_covariates, h_db_covariates, 
                         n * num_covariates * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    int block_size = CUDA_BLOCK_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    if (grid_size > CUDA_MAX_BLOCKS) grid_size = CUDA_MAX_BLOCKS;
    
    if (normalize) {
        // Allocate self-cache memory
        h_self_cache = (float*)malloc((n + 1) * sizeof(float));
        if (!h_self_cache) {
            fprintf(stderr, "Failed to allocate self-cache memory\n");
            goto cleanup;
        }
        
        // Compute da self-kernel (always 1.0 for RBF)
        h_self_cache[0] = 1.0f;
        
        // Launch kernel to compute db self-kernels
        rbf_self_kernel_batch_kernel<<<grid_size, block_size>>>(
            ctx->d_db_covariates, ctx->d_self_cache, num_covariates, n, gamma_f);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy self-cache to device
        CUDA_CHECK(cudaMemcpy(ctx->d_self_cache, h_self_cache, 
                             (n + 1) * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch normalized RBF kernel
        rbf_kernel_normalized_kernel<<<grid_size, block_size>>>(
            ctx->d_da_covariates, ctx->d_db_covariates, ctx->d_results, 
            ctx->d_self_cache, num_covariates, n, gamma_f);
    } else {
        // Launch standard RBF kernel
        rbf_kernel_batch_kernel<<<grid_size, block_size>>>(
            ctx->d_da_covariates, ctx->d_db_covariates, ctx->d_results, 
            num_covariates, n, gamma_f);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, ctx->d_results, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Convert results back to double
    float_to_double_array(h_results, results, n);
    
    // Cleanup host memory
    free(h_da_covariates);
    free(h_db_covariates);
    free(h_results);
    if (h_self_cache) free(h_self_cache);
    
    return 0;
    
cleanup:
    if (h_da_covariates) free(h_da_covariates);
    if (h_db_covariates) free(h_db_covariates);
    if (h_results) free(h_results);
    if (h_self_cache) free(h_self_cache);
    return -1;
}
