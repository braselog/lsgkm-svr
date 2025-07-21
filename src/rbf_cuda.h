#ifndef RBF_CUDA_H
#define RBF_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of gkm_data typedef
typedef struct _gkm_data gkm_data;

// CUDA configuration
#define CUDA_BLOCK_SIZE 256
#define CUDA_MAX_BLOCKS 65535
#define CUDA_MIN_BATCH_SIZE 1000  // Minimum batch size for GPU efficiency
#define CUDA_MEMORY_SAFETY_MARGIN 0.1  // 10% safety margin for memory allocation
#define CUDA_MAX_MEMORY_PRESSURE_RETRIES 3
#define CUDA_BATCH_SIZE_HISTORY_LENGTH 10

// Memory monitoring structure
typedef struct {
    size_t total_memory;         // Total GPU memory
    size_t free_memory;          // Current free memory
    size_t used_memory;          // Currently used memory
    size_t allocated_by_context; // Memory allocated by this context
    double memory_pressure;      // Memory pressure ratio (0.0-1.0)
    int last_query_time_ms;      // Last memory query timestamp
} gpu_memory_info_t;

// Batch size optimization history
typedef struct {
    int batch_sizes[CUDA_BATCH_SIZE_HISTORY_LENGTH];
    double execution_times[CUDA_BATCH_SIZE_HISTORY_LENGTH];
    int memory_pressures[CUDA_BATCH_SIZE_HISTORY_LENGTH]; // Pressure level * 100
    int success_flags[CUDA_BATCH_SIZE_HISTORY_LENGTH];    // 1 if successful, 0 if failed
    int history_index;           // Current position in circular buffer
    int history_count;           // Number of entries in history
    int optimal_batch_size;      // Current estimated optimal batch size
    double best_throughput;      // Best throughput seen so far
} batch_optimization_t;

// GPU memory management structure
typedef struct {
    float *d_da_covariates;      // Device memory for query covariates
    float *d_db_covariates;      // Device memory for database covariates (batch)
    float *d_results;            // Device memory for results
    float *d_self_cache;         // Device memory for self-kernel cache
    size_t allocated_size;       // Currently allocated size
    int max_batch_size;          // Maximum batch size supported
    int max_covariates;          // Maximum number of covariates supported
    int device_id;               // CUDA device ID
    int is_initialized;          // Initialization flag
    
    // Enhanced memory management
    gpu_memory_info_t memory_info;     // Real-time memory information
    batch_optimization_t batch_opt;    // Batch size optimization data
    int adaptive_enabled;              // Enable adaptive batch sizing
    int memory_monitoring_enabled;     // Enable real-time memory monitoring
    size_t reserved_memory;            // Memory reserved for other operations
} cuda_context_t;

// Function declarations
int cuda_is_available(void);
int cuda_init_context(cuda_context_t *ctx, int max_batch_size, int max_covariates);
void cuda_cleanup_context(cuda_context_t *ctx);

// RBF kernel computation functions
int cuda_rbf_kernel_batch(cuda_context_t *ctx, 
                         const gkm_data *da, 
                         const gkm_data **db_array,
                         int n, double gamma, double *results, int normalize);

// Enhanced memory management and optimization
int cuda_update_memory_info(cuda_context_t *ctx);
int cuda_calculate_optimal_batch_size(cuda_context_t *ctx, int num_covariates, int target_batch_size);
int cuda_adaptive_batch_size(cuda_context_t *ctx, int num_covariates, int requested_size);
void cuda_record_batch_performance(cuda_context_t *ctx, int batch_size, double execution_time, int success);
int cuda_memory_pressure_check(cuda_context_t *ctx);
void cuda_print_optimization_stats(const cuda_context_t *ctx);

// Advanced memory diagnostics and auto-tuning
void cuda_memory_diagnostics(cuda_context_t *ctx);
int cuda_auto_tune_batch_size(cuda_context_t *ctx, int num_covariates, int min_samples, int max_samples);
int cuda_schedule_batch_processing(cuda_context_t *ctx, int total_samples, int num_covariates, int **batch_schedule);

// Utility functions
int cuda_get_device_count(void);
int cuda_get_device_properties(int device_id, char *name, size_t name_len, 
                              size_t *total_mem, int *major, int *minor);
void cuda_print_device_info(void);

// Memory management
size_t cuda_get_available_memory(void);
int cuda_estimate_max_batch_size(int num_covariates);
size_t cuda_get_detailed_memory_info(size_t *free_mem, size_t *total_mem);
int cuda_reserve_memory(cuda_context_t *ctx, size_t reserve_bytes);
void cuda_enable_adaptive_sizing(cuda_context_t *ctx, int enable);
void cuda_enable_memory_monitoring(cuda_context_t *ctx, int enable);

#ifdef __cplusplus
}
#endif

#endif // RBF_CUDA_H