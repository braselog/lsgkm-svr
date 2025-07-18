# MKL Extension for lsgkm-svr

This document describes the Multiple Kernel Learning (MKL) extension for the lsgkm-svr tool, which allows combining gapped k-mer (GKM) kernels with RBF kernels on covariates.

## Overview

The MKL extension adds support for:
1. **Covariate integration**: Use continuous covariates alongside sequence data
2. **RBF kernel**: Gaussian RBF kernel for covariate data
3. **Multiple kernel learning**: Automatically optimize weights for combining kernels
4. **Flexible kernel modes**: GKM+RBF, GKM only, or RBF only

## New Kernel Types

The extension adds three new kernel types:

- **Type 7 (MKL_GKM_RBF)**: Combines GKM kernel on sequences with RBF kernel on covariates
- **Type 8 (MKL_GKM_ONLY)**: Uses only GKM kernel (for comparison with standard mode)
- **Type 9 (MKL_RBF_ONLY)**: Uses only RBF kernel on covariates

## Usage

### Basic Syntax

```bash
gkmtrain [options] <sequences1> <sequences2> <output_prefix> [covariates]
```

### New Command-Line Options

- `-t 7|8|9`: Set kernel type for MKL
- `-G <float>`: Set gamma parameter for RBF kernel on covariates (default: 1.0)
- `-W <float>`: Set initial weight for GKM kernel in MKL (default: 0.5)
- `-R <float>`: Set initial weight for RBF kernel in MKL (default: 0.5)
- `-I <int>`: Set maximum iterations for MKL optimization (default: 100)
- `-E <float>`: Set convergence tolerance for MKL optimization (default: 1e-6)
- `-N`: Enable kernel normalization before combining (recommended)

### Covariate File Format

The covariate file should be tab-separated with one row per sequence:

```
0.5	1.2	-0.3
0.8	-0.5	0.9
-0.2	0.7	1.1
```

- Each row corresponds to a sequence (same order as in FASTA files)
- Columns represent different covariates
- Missing values are not supported

## Examples

### Example 1: Standard MKL with GKM + RBF

```bash
gkmtrain -t 7 -l 10 -k 6 -d 3 -c 1 -G 0.5 -W 0.6 -R 0.4 -I 100 -N \
         positive.fa negative.fa mkl_model covariates.txt
```

### Example 2: RBF-only model

```bash
gkmtrain -t 9 -c 1 -G 0.5 -N \
         positive.fa negative.fa rbf_model covariates.txt
```

### Example 3: Regression with MKL

```bash
gkmtrain -y 3 -t 7 -l 8 -k 5 -d 2 -c 1 -G 0.1 -N \
         sequences.fa labels.txt mkl_regression_model covariates.txt
```

## Algorithm Details

### MKL Optimization

The MKL weight optimization uses a simple gradient-based approach:

1. **Kernel Computation**: Compute individual kernel matrices for GKM and RBF
2. **Alignment Optimization**: Optimize kernel weights to maximize alignment with target labels
3. **Constraint Enforcement**: Ensure non-negative weights that sum to 1
4. **Convergence**: Stop when weight changes fall below tolerance

### Kernel Combination

The final kernel is computed as:
```
K_combined = w_gkm * K_gkm + w_rbf * K_rbf
```

Where:
- `K_gkm`: Normalized GKM kernel matrix
- `K_rbf`: RBF kernel matrix on covariates
- `w_gkm`, `w_rbf`: Learned weights (sum to 1)

### RBF Kernel

The RBF kernel is computed as:
```
K_rbf(x_i, x_j) = exp(-gamma * ||x_i - x_j||²)
```

Where `x_i` and `x_j` are covariate vectors.

## Implementation Details

### Data Structures

- Extended `gkm_data` struct to include covariates
- Added MKL parameters to `svm_parameter` struct
- New kernel types in enumeration

### Key Functions

- `gkmkernel_mkl_kernelfunc()`: Compute combined kernel values
- `gkmkernel_mkl_optimize_weights()`: Optimize kernel weights
- `rbf_kernel()`: Compute RBF kernel for covariates
- `rbf_kernel_normalize()`: Compute normalized RBF kernel

### File Changes

- `libsvm.h`: Extended data structures and parameters
- `libsvm_gkm.h`: Added MKL function declarations
- `libsvm_gkm.c`: Implemented MKL functionality
- `libsvm.cpp`: Updated kernel computation and parameter validation
- `gkmtrain.c`: Added covariate reading and MKL options
- `gkmpredict.c`: Updated for covariate support

## Performance Considerations

- **Memory**: MKL requires storing additional kernel matrices
- **Time**: Kernel optimization adds computational overhead
- **Scaling**: RBF kernel computation is O(n²) in number of samples

## Validation

The implementation includes:
- Parameter validation for MKL-specific options
- Covariate presence checking for MKL modes
- Kernel normalization options
- Convergence monitoring for weight optimization

## Limitations

- Currently supports only dense covariate matrices
- No support for missing covariate values
- Limited to simple gradient-based MKL optimization
- No automatic hyperparameter tuning for RBF gamma

## Future Enhancements

Potential improvements include:
- Advanced MKL optimization algorithms
- Sparse covariate support
- Cross-validation for hyperparameter selection
- Support for different kernel types on covariates
- Kernel-specific regularization

## Testing

Run the test script to verify MKL functionality:

```bash
./test_mkl.sh
```

This will test all MKL modes with synthetic data.
