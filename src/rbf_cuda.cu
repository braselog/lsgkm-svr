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
#include <time.h>

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

// Get available GPU memory with detailed information
extern "C" size_t cuda_get_detailed_memory_info(size_t *free_mem, size_t *total_mem) {
    cudaError_t err = cudaMemGetInfo(free_mem, total_mem);
    if (err != cudaSuccess) {
        if (free_mem) *free_mem = 0;
        if (total_mem) *total_mem = 0;
        return 0;
    }
    return *free_mem;
}

// Get available GPU memory
extern "C" size_t cuda_get_available_memory(void) {
    size_t free_mem, total_mem;
    return cuda_get_detailed_memory_info(&free_mem, &total_mem);
}

// Update real-time memory information in context
extern "C" int cuda_update_memory_info(cuda_context_t *ctx) {
    if (!ctx) return -1;
    
    size_t free_mem, total_mem;
    size_t result = cuda_get_detailed_memory_info(&free_mem, &total_mem);
    if (result == 0) {
        return -1;
    }
    
    ctx->memory_info.total_memory = total_mem;
    ctx->memory_info.free_memory = free_mem;
    ctx->memory_info.used_memory = total_mem - free_mem;
    
    // Calculate memory pressure (0.0 = no pressure, 1.0 = completely full)
    ctx->memory_info.memory_pressure = (double)(total_mem - free_mem) / (double)total_mem;
    
    // Update timestamp (simplified - could use actual time)
    ctx->memory_info.last_query_time_ms++;
    
    return 0;
}

// Check memory pressure and return severity level
extern "C" int cuda_memory_pressure_check(cuda_context_t *ctx) {
    if (!ctx || !ctx->memory_monitoring_enabled) return 0;
    
    if (cuda_update_memory_info(ctx) != 0) return -1;
    
    double pressure = ctx->memory_info.memory_pressure;
    
    if (pressure < 0.7) return 0;      // Low pressure
    else if (pressure < 0.85) return 1; // Medium pressure
    else if (pressure < 0.95) return 2; // High pressure
    else return 3;                      // Critical pressure
}

// Record batch performance for optimization
extern "C" void cuda_record_batch_performance(cuda_context_t *ctx, int batch_size, 
                                            double execution_time, int success) {
    if (!ctx || !ctx->adaptive_enabled) return;
    
    batch_optimization_t *opt = &ctx->batch_opt;
    int idx = opt->history_index;
    
    // Record data
    opt->batch_sizes[idx] = batch_size;
    opt->execution_times[idx] = execution_time;
    opt->success_flags[idx] = success;
    
    // Update memory pressure at time of execution
    if (ctx->memory_monitoring_enabled) {
        opt->memory_pressures[idx] = (int)(ctx->memory_info.memory_pressure * 100);
    } else {
        opt->memory_pressures[idx] = 0;
    }
    
    // Calculate throughput (elements per second)
    double throughput = success ? (batch_size / execution_time) : 0.0;
    
    // Update best throughput and optimal batch size
    if (success && throughput > opt->best_throughput) {
        opt->best_throughput = throughput;
        opt->optimal_batch_size = batch_size;
    }
    
    // Advance circular buffer
    opt->history_index = (opt->history_index + 1) % CUDA_BATCH_SIZE_HISTORY_LENGTH;
    if (opt->history_count < CUDA_BATCH_SIZE_HISTORY_LENGTH) {
        opt->history_count++;
    }
}

// Calculate optimal batch size based on memory and performance history
extern "C" int cuda_calculate_optimal_batch_size(cuda_context_t *ctx, int num_covariates, 
                                                int target_batch_size) {
    if (!ctx || !ctx->is_initialized) return target_batch_size;
    
    // Update memory information
    if (ctx->memory_monitoring_enabled) {
        cuda_update_memory_info(ctx);
    }
    
    // Calculate memory-based maximum
    size_t available_mem = ctx->memory_info.free_memory;
    if (ctx->reserved_memory > 0) {
        available_mem = (available_mem > ctx->reserved_memory) ? 
                       (available_mem - ctx->reserved_memory) : 0;
    }
    
    // Apply safety margin
    size_t usable_mem = (size_t)(available_mem * (1.0 - CUDA_MEMORY_SAFETY_MARGIN));
    
    // Memory required per sample
    size_t mem_per_sample = num_covariates * sizeof(float) + 2 * sizeof(float);
    size_t fixed_mem = num_covariates * sizeof(float); // da_covariates
    
    int memory_limited_batch = (usable_mem > fixed_mem) ? 
                              (int)((usable_mem - fixed_mem) / mem_per_sample) : 0;
    
    // Consider hardware limits
    int hardware_limited_batch = ctx->max_batch_size;
    
    // Consider memory pressure
    int pressure_level = cuda_memory_pressure_check(ctx);
    double pressure_factor = 1.0;
    switch (pressure_level) {
        case 1: pressure_factor = 0.8; break;  // Medium pressure
        case 2: pressure_factor = 0.6; break;  // High pressure
        case 3: pressure_factor = 0.4; break;  // Critical pressure
        default: pressure_factor = 1.0; break; // Low pressure
    }
    
    int pressure_adjusted_batch = (int)(memory_limited_batch * pressure_factor);
    
    // Use performance history if available
    int optimal_batch = target_batch_size;
    if (ctx->adaptive_enabled && ctx->batch_opt.optimal_batch_size > 0) {
        optimal_batch = ctx->batch_opt.optimal_batch_size;
    }
    
    // Take the minimum of all constraints
    int final_batch = target_batch_size;
    if (memory_limited_batch > 0) final_batch = memory_limited_batch;
    if (pressure_adjusted_batch < final_batch) final_batch = pressure_adjusted_batch;
    if (hardware_limited_batch < final_batch) final_batch = hardware_limited_batch;
    
    // Performance-based adjustment
    if (ctx->adaptive_enabled && optimal_batch < final_batch && optimal_batch >= CUDA_MIN_BATCH_SIZE) {
        final_batch = optimal_batch;
    }
    
    // Ensure minimum batch size
    if (final_batch < CUDA_MIN_BATCH_SIZE) {
        final_batch = (memory_limited_batch >= CUDA_MIN_BATCH_SIZE) ? CUDA_MIN_BATCH_SIZE : 0;
    }
    
    return final_batch;
}

// Adaptive batch size with real-time adjustment
extern "C" int cuda_adaptive_batch_size(cuda_context_t *ctx, int num_covariates, int requested_size) {
    if (!ctx || !ctx->is_initialized) return requested_size;
    
    int optimal_size = cuda_calculate_optimal_batch_size(ctx, num_covariates, requested_size);
    
    // If optimal size is significantly smaller than requested, try to understand why
    if (optimal_size < requested_size * 0.5) {
        int pressure = cuda_memory_pressure_check(ctx);
        if (pressure >= 2) {
            // High memory pressure - consider cleanup or waiting
            printf("Warning: High GPU memory pressure (level %d), reducing batch size from %d to %d\n",
                   pressure, requested_size, optimal_size);
        }
    }
    
    return optimal_size;
}

// Enable/disable adaptive sizing
extern "C" void cuda_enable_adaptive_sizing(cuda_context_t *ctx, int enable) {
    if (!ctx) return;
    ctx->adaptive_enabled = enable ? 1 : 0;
    
    if (enable) {
        // Initialize optimization data
        memset(&ctx->batch_opt, 0, sizeof(batch_optimization_t));
        ctx->batch_opt.optimal_batch_size = ctx->max_batch_size / 2; // Start with conservative estimate
    }
}

// Enable/disable memory monitoring
extern "C" void cuda_enable_memory_monitoring(cuda_context_t *ctx, int enable) {
    if (!ctx) return;
    ctx->memory_monitoring_enabled = enable ? 1 : 0;
    
    if (enable) {
        cuda_update_memory_info(ctx);
    }
}

// Reserve memory for other operations
extern "C" int cuda_reserve_memory(cuda_context_t *ctx, size_t reserve_bytes) {
    if (!ctx) return -1;
    ctx->reserved_memory = reserve_bytes;
    return 0;
}

// Print optimization statistics
extern "C" void cuda_print_optimization_stats(const cuda_context_t *ctx) {
    if (!ctx || !ctx->adaptive_enabled) {
        printf("Adaptive sizing is disabled\n");
        return;
    }
    
    const batch_optimization_t *opt = &ctx->batch_opt;
    
    printf("GPU Batch Optimization Statistics:\n");
    printf("==================================\n");
    printf("History entries: %d/%d\n", opt->history_count, CUDA_BATCH_SIZE_HISTORY_LENGTH);
    printf("Optimal batch size: %d\n", opt->optimal_batch_size);
    printf("Best throughput: %.2f samples/sec\n", opt->best_throughput);
    
    if (ctx->memory_monitoring_enabled) {
        printf("Current memory pressure: %.1f%%\n", ctx->memory_info.memory_pressure * 100);
        printf("Free GPU memory: %.2f MB\n", (double)ctx->memory_info.free_memory / (1024.0 * 1024.0));
    }
    
    if (opt->history_count > 0) {
        printf("\nRecent batch performance:\n");
        printf("Size\tTime(s)\tThroughput\tSuccess\tMemPress\n");
        for (int i = 0; i < opt->history_count; i++) {
            int idx = (opt->history_index - opt->history_count + i + CUDA_BATCH_SIZE_HISTORY_LENGTH) 
                     % CUDA_BATCH_SIZE_HISTORY_LENGTH;
            double throughput = opt->success_flags[idx] ? 
                               (opt->batch_sizes[idx] / opt->execution_times[idx]) : 0.0;
            printf("%d\t%.3f\t%.1f\t\t%s\t%d%%\n",
                   opt->batch_sizes[idx],
                   opt->execution_times[idx],
                   throughput,
                   opt->success_flags[idx] ? "Yes" : "No",
                   opt->memory_pressures[idx]);
        }
    }
    printf("\n");
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
    
    // Mark as initialized early so memory functions work
    ctx->is_initialized = 1;
    
    // Initialize memory monitoring
    ctx->memory_monitoring_enabled = 1;
    ctx->adaptive_enabled = 1;
    cuda_update_memory_info(ctx);
    
    // Estimate batch size if not provided
    if (max_batch_size <= 0) {
        max_batch_size = cuda_estimate_max_batch_size(max_covariates);
        if (max_batch_size <= 0) {
            fprintf(stderr, "Cannot determine suitable batch size for GPU\n");
            return -1;
        }
    }
    
    // Adjust batch size based on current memory conditions
    int optimal_batch = cuda_calculate_optimal_batch_size(ctx, max_covariates, max_batch_size);
    if (optimal_batch > 0 && optimal_batch < max_batch_size) {
        printf("Adjusting batch size from %d to %d based on memory conditions\n", 
               max_batch_size, optimal_batch);
        max_batch_size = optimal_batch;
    }
    
    ctx->max_batch_size = max_batch_size;
    ctx->max_covariates = max_covariates;
    
    // Calculate memory requirements
    size_t da_size = max_covariates * sizeof(float);
    size_t db_size = max_batch_size * max_covariates * sizeof(float);
    size_t results_size = max_batch_size * sizeof(float);
    size_t cache_size = (max_batch_size + 1) * sizeof(float);  // +1 for da self-kernel
    
    ctx->allocated_size = da_size + db_size + results_size + cache_size;
    
    // Check if we have enough memory for allocation
    if (ctx->memory_info.free_memory < ctx->allocated_size * 1.2) // 20% safety margin
    {
        fprintf(stderr, "Insufficient GPU memory: need %.2f MB, have %.2f MB\n",
                (double)ctx->allocated_size / (1024.0 * 1024.0),
                (double)ctx->memory_info.free_memory / (1024.0 * 1024.0));
        return -1;
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&ctx->d_da_covariates, da_size));
    CUDA_CHECK(cudaMalloc(&ctx->d_db_covariates, db_size));
    CUDA_CHECK(cudaMalloc(&ctx->d_results, results_size));
    CUDA_CHECK(cudaMalloc(&ctx->d_self_cache, cache_size));
    
    // Update memory info after allocation
    cuda_update_memory_info(ctx);
    ctx->memory_info.allocated_by_context = ctx->allocated_size;
    
    // Initialize batch optimization
    cuda_enable_adaptive_sizing(ctx, 1);
    
    printf("CUDA context initialized:\n");
    printf("  Max batch size: %d\n", max_batch_size);
    printf("  Max covariates: %d\n", max_covariates);
    printf("  GPU memory allocated: %.2f MB\n", (double)ctx->allocated_size / (1024.0 * 1024.0));
    printf("  GPU memory remaining: %.2f MB\n", (double)ctx->memory_info.free_memory / (1024.0 * 1024.0));
    printf("  Memory pressure: %.1f%%\n", ctx->memory_info.memory_pressure * 100);
    printf("  Adaptive sizing: %s\n", ctx->adaptive_enabled ? "enabled" : "disabled");
    printf("  Memory monitoring: %s\n", ctx->memory_monitoring_enabled ? "enabled" : "disabled");
    
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
                                    const gkm_data *da,
                                    const gkm_data **db_array,
                                    int n, double gamma, double *results, int normalize) {
    if (!ctx || !ctx->is_initialized || !da || !db_array || !results || n <= 0) {
        return -1;
    }
    
    // Declare variables at the beginning to avoid goto initialization issues
    int block_size = CUDA_BLOCK_SIZE;
    int grid_size;
    double execution_time = 0.0;
    
    // Record start time for performance monitoring
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Check memory pressure before proceeding
    int pressure_level = cuda_memory_pressure_check(ctx);
    if (pressure_level >= 3) {
        fprintf(stderr, "Critical GPU memory pressure detected, aborting batch\n");
        cuda_record_batch_performance(ctx, n, 0.0, 0);
        return -1;
    }
    
    // Adaptive batch size adjustment
    int original_n = n;
    if (ctx->adaptive_enabled) {
        int optimal_n = cuda_adaptive_batch_size(ctx, da->num_covariates, n);
        if (optimal_n != n) {
            if (optimal_n <= 0) {
                fprintf(stderr, "Cannot process batch: insufficient GPU memory\n");
                cuda_record_batch_performance(ctx, n, 0.0, 0);
                return -1;
            }
            if (optimal_n < n) {
                printf("Adaptive sizing: reducing batch from %d to %d samples\n", n, optimal_n);
                n = optimal_n;
                // Caller should handle the remaining samples
            }
        }
    }
    
    if (n > ctx->max_batch_size) {
        fprintf(stderr, "Batch size %d exceeds maximum %d\n", n, ctx->max_batch_size);
        cuda_record_batch_performance(ctx, n, 0.0, 0);
        return -1;
    }
    
    if (da->num_covariates > ctx->max_covariates) {
        fprintf(stderr, "Number of covariates %d exceeds maximum %d\n", 
                da->num_covariates, ctx->max_covariates);
        cuda_record_batch_performance(ctx, n, 0.0, 0);
        return -1;
    }
    
    // Validate that all samples have the same number of covariates
    for (int i = 0; i < n; i++) {
        if (db_array[i]->num_covariates != da->num_covariates) {
            fprintf(stderr, "Mismatched number of covariates at sample %d\n", i);
            cuda_record_batch_performance(ctx, n, 0.0, 0);
            return -1;
        }
    }
    
    int num_covariates = da->num_covariates;
    float gamma_f = (float)gamma;
    
    // Monitor memory before allocation
    if (ctx->memory_monitoring_enabled) {
        cuda_update_memory_info(ctx);
    }
    
    // Allocate host memory for data conversion
    float *h_da_covariates = (float*)malloc(num_covariates * sizeof(float));
    float *h_db_covariates = (float*)malloc(n * num_covariates * sizeof(float));
    float *h_results = (float*)malloc(n * sizeof(float));
    float *h_self_cache = NULL;
    
    if (!h_da_covariates || !h_db_covariates || !h_results) {
        fprintf(stderr, "Failed to allocate host memory\n");
        goto cleanup_with_error;
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
    grid_size = (n + block_size - 1) / block_size;
    if (grid_size > CUDA_MAX_BLOCKS) grid_size = CUDA_MAX_BLOCKS;
    
    if (normalize) {
        // Allocate self-cache memory
        h_self_cache = (float*)malloc((n + 1) * sizeof(float));
        if (!h_self_cache) {
            fprintf(stderr, "Failed to allocate self-cache memory\n");
            goto cleanup_with_error;
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
    
    // Calculate execution time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    execution_time = (end_time.tv_sec - start_time.tv_sec) + 
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    
    // Record performance for optimization
    cuda_record_batch_performance(ctx, n, execution_time, 1);
    
    // Cleanup host memory
    free(h_da_covariates);
    free(h_db_covariates);
    free(h_results);
    if (h_self_cache) free(h_self_cache);
    
    // Return the number of samples actually processed (may be less than requested)
    return n;
    
cleanup_with_error:
    if (h_da_covariates) free(h_da_covariates);
    if (h_db_covariates) free(h_db_covariates);
    if (h_results) free(h_results);
    if (h_self_cache) free(h_self_cache);
    
    // Record failed performance
    struct timespec error_time;
    clock_gettime(CLOCK_MONOTONIC, &error_time);
    double error_execution_time = (error_time.tv_sec - start_time.tv_sec) + 
                                 (error_time.tv_nsec - start_time.tv_nsec) / 1e9;
    cuda_record_batch_performance(ctx, original_n, error_execution_time, 0);
    
    return -1;
}

// Advanced GPU memory diagnostics and auto-tuning
extern "C" void cuda_memory_diagnostics(cuda_context_t *ctx) {
    if (!ctx) return;
    
    printf("\n=== GPU Memory Diagnostics ===\n");
    
    // Update memory info
    cuda_update_memory_info(ctx);
    
    // Basic memory information
    printf("Total GPU Memory: %.2f GB\n", 
           (double)ctx->memory_info.total_memory / (1024.0 * 1024.0 * 1024.0));
    printf("Free GPU Memory: %.2f MB\n", 
           (double)ctx->memory_info.free_memory / (1024.0 * 1024.0));
    printf("Used GPU Memory: %.2f MB\n", 
           (double)ctx->memory_info.used_memory / (1024.0 * 1024.0));
    printf("Context Allocation: %.2f MB\n", 
           (double)ctx->memory_info.allocated_by_context / (1024.0 * 1024.0));
    printf("Memory Pressure: %.1f%%\n", ctx->memory_info.memory_pressure * 100);
    
    // Pressure level assessment
    int pressure = cuda_memory_pressure_check(ctx);
    const char* pressure_str[] = {"Low", "Medium", "High", "Critical"};
    printf("Pressure Level: %s (%d)\n", pressure_str[pressure], pressure);
    
    // Memory efficiency recommendations
    printf("\n--- Recommendations ---\n");
    if (pressure >= 2) {
        printf("• High memory pressure detected - consider reducing batch sizes\n");
        printf("• Reserve memory: %.2f MB could be released\n", 
               (double)ctx->reserved_memory / (1024.0 * 1024.0));
    }
    
    if (ctx->adaptive_enabled && ctx->batch_opt.history_count > 5) {
        double avg_throughput = 0.0;
        int successful_runs = 0;
        for (int i = 0; i < ctx->batch_opt.history_count; i++) {
            if (ctx->batch_opt.success_flags[i]) {
                avg_throughput += ctx->batch_opt.batch_sizes[i] / ctx->batch_opt.execution_times[i];
                successful_runs++;
            }
        }
        if (successful_runs > 0) {
            avg_throughput /= successful_runs;
            printf("• Average throughput: %.1f samples/sec\n", avg_throughput);
            printf("• Optimal batch size: %d\n", ctx->batch_opt.optimal_batch_size);
            
            if (ctx->batch_opt.best_throughput > avg_throughput * 1.2) {
                printf("• Performance could be improved by using optimal batch size more consistently\n");
            }
        }
    }
    
    printf("==============================\n\n");
}

// Auto-tune batch size based on system state
extern "C" int cuda_auto_tune_batch_size(cuda_context_t *ctx, int num_covariates, 
                                        int min_samples, int max_samples) {
    if (!ctx || !ctx->is_initialized || !ctx->adaptive_enabled) return -1;
    
    printf("Auto-tuning batch size for %d covariates...\n", num_covariates);
    
    // Test different batch sizes
    int test_sizes[] = {1000, 2000, 5000, 10000, 20000, 50000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    int best_size = 0;
    double best_throughput = 0.0;
    
    for (int i = 0; i < num_tests; i++) {
        int test_size = test_sizes[i];
        if (test_size < min_samples || test_size > max_samples) continue;
        
        // Check if this size is feasible with current memory
        int feasible_size = cuda_calculate_optimal_batch_size(ctx, num_covariates, test_size);
        if (feasible_size < test_size * 0.8) {
            printf("Batch size %d not feasible (would be reduced to %d)\n", test_size, feasible_size);
            continue;
        }
        
        // Create dummy data for testing
        float *test_data = (float*)malloc(test_size * num_covariates * sizeof(float));
        if (!test_data) continue;
        
        // Fill with random data
        for (int j = 0; j < test_size * num_covariates; j++) {
            test_data[j] = (float)rand() / RAND_MAX;
        }
        
        // Time the operation
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Copy to GPU and measure transfer + computation time
        float *d_test_data;
        if (cudaMalloc(&d_test_data, test_size * num_covariates * sizeof(float)) == cudaSuccess) {
            if (cudaMemcpy(d_test_data, test_data, test_size * num_covariates * sizeof(float), 
                          cudaMemcpyHostToDevice) == cudaSuccess) {
                
                // Synchronize to ensure timing accuracy
                cudaDeviceSynchronize();
                
                clock_gettime(CLOCK_MONOTONIC, &end);
                double time_taken = (end.tv_sec - start.tv_sec) + 
                                   (end.tv_nsec - start.tv_nsec) / 1e9;
                
                double throughput = test_size / time_taken;
                printf("Batch size %d: %.2f sec, %.1f samples/sec\n", 
                       test_size, time_taken, throughput);
                
                if (throughput > best_throughput) {
                    best_throughput = throughput;
                    best_size = test_size;
                }
                
                // Record in optimization history
                cuda_record_batch_performance(ctx, test_size, time_taken, 1);
            }
            cudaFree(d_test_data);
        }
        
        free(test_data);
    }
    
    if (best_size > 0) {
        ctx->batch_opt.optimal_batch_size = best_size;
        printf("Auto-tuning complete. Optimal batch size: %d (%.1f samples/sec)\n", 
               best_size, best_throughput);
    } else {
        printf("Auto-tuning failed to find optimal batch size\n");
    }
    
    return best_size;
}

// Memory-aware processing scheduler
extern "C" int cuda_schedule_batch_processing(cuda_context_t *ctx, int total_samples, 
                                            int num_covariates, int **batch_schedule) {
    if (!ctx || !batch_schedule) return -1;
    
    // Update memory state
    cuda_update_memory_info(ctx);
    
    // Calculate base batch size
    int base_batch = cuda_calculate_optimal_batch_size(ctx, num_covariates, 
                                                      ctx->batch_opt.optimal_batch_size > 0 ? 
                                                      ctx->batch_opt.optimal_batch_size : 10000);
    
    if (base_batch <= 0) return -1;
    
    // Calculate number of batches needed
    int num_batches = (total_samples + base_batch - 1) / base_batch;
    *batch_schedule = (int*)malloc(num_batches * sizeof(int));
    if (!*batch_schedule) return -1;
    
    // Plan batches with adaptive sizing
    int remaining = total_samples;
    for (int i = 0; i < num_batches; i++) {
        int batch_size = (remaining > base_batch) ? base_batch : remaining;
        
        // Adjust for memory pressure
        int pressure = cuda_memory_pressure_check(ctx);
        if (pressure >= 2) {
            batch_size = (int)(batch_size * (1.0 - pressure * 0.2)); // Reduce by up to 60%
        }
        
        // Ensure minimum efficiency
        if (batch_size < CUDA_MIN_BATCH_SIZE) {
            batch_size = (remaining >= CUDA_MIN_BATCH_SIZE) ? CUDA_MIN_BATCH_SIZE : remaining;
        }
        
        (*batch_schedule)[i] = batch_size;
        remaining -= batch_size;
        
        if (remaining <= 0) {
            num_batches = i + 1;
            break;
        }
    }
    
    printf("Scheduled %d batches for %d samples:\n", num_batches, total_samples);
    for (int i = 0; i < num_batches; i++) {
        printf("  Batch %d: %d samples\n", i, (*batch_schedule)[i]);
    }
    
    return num_batches;
}
