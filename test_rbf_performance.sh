#!/bin/bash

# Performance test script for RBF kernel optimizations
# This script tests the performance improvements for RBF kernel computation

echo "Testing RBF Kernel Optimizations"
echo "================================"

cd /mnt/mr01-home01/m65338lb/projects/mRNA_LLM-worktrees/translationEfficiency/lsgkm-svr

# Check if test data exists
if [[ ! -f "test_data/positive.fa" || ! -f "test_data/negative.fa" ]]; then
    echo "Error: Test data not found"
    exit 1
fi

echo "Building test models with different kernel types..."

# Test with traditional GKM kernel (baseline)
echo "1. Training GKM model (baseline)..."
time src/gkmtrain -t 2 -T 1 -v 0 test_data/positive.fa test_data/negative.fa baseline_model.txt

# Test with RBF-only kernel using 1 thread
echo "2. Training RBF-only model (1 thread)..."
if [[ -f "test_data/covariates.txt" ]]; then
    time src/gkmtrain -t 9 -T 1 -v 0 -F test_data/covariates.txt test_data/positive.fa test_data/negative.fa rbf_1thread_model.txt
else
    echo "   Skipped: No covariates file found"
fi

# Test with RBF-only kernel using 4 threads
echo "3. Training RBF-only model (4 threads)..."
if [[ -f "test_data/covariates.txt" ]]; then
    time src/gkmtrain -t 9 -T 4 -v 0 -F test_data/covariates.txt test_data/positive.fa test_data/negative.fa rbf_4thread_model.txt
else
    echo "   Skipped: No covariates file found"
fi

# Test with MKL (GKM + RBF) using 1 thread
echo "4. Training MKL model (1 thread)..."
if [[ -f "test_data/covariates.txt" ]]; then
    time src/gkmtrain -t 7 -T 1 -v 0 -F test_data/covariates.txt test_data/positive.fa test_data/negative.fa mkl_1thread_model.txt
else
    echo "   Skipped: No covariates file found"
fi

# Test with MKL (GKM + RBF) using 4 threads
echo "5. Training MKL model (4 threads)..."
if [[ -f "test_data/covariates.txt" ]]; then
    time src/gkmtrain -t 7 -T 4 -v 0 -F test_data/covariates.txt test_data/positive.fa test_data/negative.fa mkl_4thread_model.txt
else
    echo "   Skipped: No covariates file found"
fi

echo ""
echo "Performance test completed!"
echo ""
echo "Expected improvements:"
echo "- RBF 4-thread should be ~4x faster than RBF 1-thread"
echo "- MKL 4-thread should be faster than MKL 1-thread for RBF components"
echo "- Overall speedup depends on the proportion of RBF computation"

# Clean up test models
echo "Cleaning up test files..."
rm -f *thread_model.txt baseline_model.txt

echo "Done!"
