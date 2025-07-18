#ifndef RBF_CUDA_H
#define RBF_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of gkm_data structure
struct gkm_data;

// CUDA configuration
#define CUDA_BLOCK_SIZE 256
#define CUDA_MAX_BLOCKS 65535
#define CUDA_MIN_BATCH_SIZE 1000  // Minimum batch size for GPU efficiency

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
} cuda_context_t;

// Function declarations
int cuda_is_available(void);
int cuda_init_context(cuda_context_t *ctx, int max_batch_size, int max_covariates);
void cuda_cleanup_context(cuda_context_t *ctx);

// RBF kernel computation functions
int cuda_rbf_kernel_batch(cuda_context_t *ctx, 
                         const struct gkm_data *da, 
                         const struct gkm_data **db_array,
                         int n, double gamma, double *results, int normalize);

// Utility functions
int cuda_get_device_count(void);
int cuda_get_device_properties(int device_id, char *name, size_t name_len, 
                              size_t *total_mem, int *major, int *minor);
void cuda_print_device_info(void);

// Memory management
size_t cuda_get_available_memory(void);
int cuda_estimate_max_batch_size(int num_covariates);

#ifdef __cplusplus
}
#endif

#endif // RBF_CUDA_H