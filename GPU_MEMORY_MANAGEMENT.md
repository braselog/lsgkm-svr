# Enhanced GPU Memory Management for lsgkm-svr

This enhanced version of lsgkm-svr includes advanced GPU memory management with adaptive batch sizing capabilities to optimize performance on large datasets while preventing GPU memory exhaustion.

## New Features

### 1. Real-time GPU Memory Monitoring
- Continuous monitoring of GPU memory usage and pressure levels
- Automatic detection of memory pressure conditions (Low, Medium, High, Critical)
- Dynamic memory allocation tracking

### 2. Adaptive Batch Sizing
- Automatic adjustment of batch sizes based on available GPU memory
- Performance history tracking to optimize batch sizes over time
- Memory pressure-aware batch size reduction
- Fallback to CPU processing when GPU memory is insufficient

### 3. Auto-tuning System
- Automatic discovery of optimal batch sizes for your specific hardware and data
- Performance benchmarking with different batch sizes
- Throughput optimization based on empirical testing

### 4. Memory Reservation
- Ability to reserve GPU memory for other concurrent applications
- Prevents memory conflicts in multi-process environments
- Configurable memory safety margins

## Command Line Options

### GPU Memory Management Options

```bash
--gpu-diag              Print detailed GPU memory diagnostics and optimization stats
--gpu-tune              Perform auto-tuning of batch sizes before training  
--gpu-reserve <MB>      Reserve GPU memory (in MB) for other operations
```

### Usage Examples

#### Basic GPU diagnostics
```bash
./gkmtrain --gpu-diag -t 4 -l 11 -k 7 positive.fa negative.fa model
```

#### Auto-tune batch sizes for optimal performance
```bash
./gkmtrain --gpu-tune -t 4 -l 11 -k 7 positive.fa negative.fa model
```

#### Reserve 1GB of GPU memory for other applications
```bash
./gkmtrain --gpu-reserve 1024 -t 4 -l 11 -k 7 positive.fa negative.fa model
```

#### Combined usage with full optimization
```bash
./gkmtrain --gpu-diag --gpu-tune --gpu-reserve 512 -t 4 -l 11 -k 7 -c 1.0 -v 3 positive.fa negative.fa model
```

## Technical Implementation

### Memory Monitoring Structure
```c
typedef struct {
    size_t total_memory;         // Total GPU memory
    size_t free_memory;          // Current free memory
    size_t used_memory;          // Currently used memory
    size_t allocated_by_context; // Memory allocated by this context
    double memory_pressure;      // Memory pressure ratio (0.0-1.0)
    int last_query_time_ms;      // Last memory query timestamp
} gpu_memory_info_t;
```

### Batch Optimization History
```c
typedef struct {
    int batch_sizes[CUDA_BATCH_SIZE_HISTORY_LENGTH];
    double execution_times[CUDA_BATCH_SIZE_HISTORY_LENGTH];
    int memory_pressures[CUDA_BATCH_SIZE_HISTORY_LENGTH];
    int success_flags[CUDA_BATCH_SIZE_HISTORY_LENGTH];
    int optimal_batch_size;      // Current estimated optimal batch size
    double best_throughput;      // Best throughput seen so far
} batch_optimization_t;
```

### Key Functions

#### Memory Management
- `cuda_update_memory_info()` - Real-time memory information update
- `cuda_memory_pressure_check()` - Check memory pressure level
- `cuda_calculate_optimal_batch_size()` - Calculate optimal batch size
- `cuda_adaptive_batch_size()` - Get adaptive batch size for current conditions

#### Diagnostics and Auto-tuning
- `cuda_memory_diagnostics()` - Comprehensive memory diagnostics
- `cuda_auto_tune_batch_size()` - Auto-tune batch sizes
- `cuda_print_optimization_stats()` - Print performance statistics

#### Batch Processing
- `cuda_schedule_batch_processing()` - Smart batch scheduling
- `cuda_record_batch_performance()` - Record performance metrics

## Memory Pressure Levels

| Level | Pressure | Description | Action |
|-------|----------|-------------|---------|
| 0 | < 70% | Low | Normal processing |
| 1 | 70-85% | Medium | Reduce batch size by 20% |
| 2 | 85-95% | High | Reduce batch size by 40% |
| 3 | > 95% | Critical | Switch to CPU or abort |

## Performance Optimization Tips

### 1. Initial Setup
- Run `--gpu-tune` once to establish optimal batch sizes for your data
- Use the discovered optimal batch size for subsequent runs
- Monitor GPU usage with `--gpu-diag` if experiencing issues

### 2. Memory Management
- Use `--gpu-reserve` to leave memory for other GPU applications
- Consider the total size of your dataset when planning batch sizes
- Monitor memory pressure levels during long-running jobs

### 3. Large Dataset Processing
- The system automatically splits large datasets into manageable chunks
- Adaptive sizing ensures maximum GPU utilization without memory overflow
- Failed GPU batches automatically fall back to CPU processing

### 4. Multi-GPU Systems
- The current implementation uses GPU 0 by default
- Memory monitoring is per-GPU and considers only the active device
- Future versions may include multi-GPU support

## Error Handling and Fallbacks

### Automatic Fallbacks
1. **Memory exhaustion**: Automatically reduces batch size and retries
2. **Critical pressure**: Falls back to CPU processing
3. **GPU failure**: Seamlessly continues with CPU-only processing
4. **Insufficient memory**: Calculates maximum possible batch size

### Error Recovery
- Up to 3 retries with progressively smaller batch sizes
- Intelligent batch size reduction based on memory pressure
- Graceful degradation to CPU processing when needed

## Compilation Notes

### Requirements
- CUDA-capable GPU with compute capability 3.0 or higher
- NVIDIA CUDA Toolkit 8.0 or later
- Sufficient GPU memory (recommended: 4GB or more)

### Compilation Flags
```bash
# Enable CUDA support
make CUDA=1

# Debug build with verbose GPU diagnostics
make CUDA=1 DEBUG=1
```

## Troubleshooting

### Common Issues

#### "CUDA is not available"
- Verify CUDA installation: `nvidia-smi`
- Check CUDA runtime library: `ldconfig -p | grep cuda`
- Ensure GPU compute capability is supported

#### "Insufficient GPU memory"
- Use `--gpu-diag` to see available memory
- Reduce dataset size or use `--gpu-reserve` with smaller value
- Consider upgrading GPU memory

#### "GPU computation failed, falling back to CPU"
- Check GPU memory pressure with `--gpu-diag`
- Verify CUDA driver compatibility
- Monitor GPU temperature and utilization

#### Poor Performance
- Run `--gpu-tune` to optimize batch sizes
- Increase GPU memory allocation if possible
- Check for memory fragmentation with `--gpu-diag`

### Debug Information
Use verbosity level 3 (`-v 3`) to see detailed GPU memory and batch processing information:
```bash
./gkmtrain --gpu-diag -v 3 -t 4 -l 11 -k 7 positive.fa negative.fa model
```

## Demo Script

Run the included demo script to see all features in action:
```bash
./gpu_memory_demo.sh
```

This script demonstrates:
- GPU memory diagnostics
- Batch size auto-tuning
- Memory reservation
- Adaptive batch processing
- Performance monitoring

## Future Enhancements

### Planned Features
- Multi-GPU support for large-scale processing
- CUDA memory pool management
- Advanced performance profiling
- Integration with CUDA Memory Pool API
- Support for CUDA Unified Memory

### Performance Improvements
- Asynchronous memory transfers
- Kernel fusion for better GPU utilization
- Dynamic load balancing between GPU and CPU
- Predictive batch sizing based on data characteristics

## License

This enhanced GPU memory management system maintains the same license as the original lsgkm-svr project. All new features are released under the GNU General Public License v3.0.
