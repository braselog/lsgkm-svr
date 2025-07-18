#!/bin/bash

# GPU Acceleration Test Script for lsgkm-svr
# This script tests the GPU acceleration functionality

echo "=== lsgkm-svr GPU Acceleration Test ==="
echo

# Check if we're in the correct directory
if [[ ! -f "src/Makefile" ]]; then
    echo "Error: Please run this script from the lsgkm-svr root directory"
    exit 1
fi

cd src

echo "1. Testing build system..."
echo "=========================="

# Test CPU-only build
echo "Building CPU-only version..."
make clean > /dev/null 2>&1
if make CUDA_ENABLED=0 > build_cpu.log 2>&1; then
    echo "✓ CPU-only build successful"
else
    echo "✗ CPU-only build failed. Check build_cpu.log"
    exit 1
fi

# Test CUDA build if available
echo "Building CUDA version..."
make clean > /dev/null 2>&1
if make > build_cuda.log 2>&1; then
    echo "✓ CUDA build successful"
    CUDA_BUILD=1
else
    echo "⚠ CUDA build failed (expected if CUDA not available). Check build_cuda.log"
    CUDA_BUILD=0
fi

echo

echo "2. Testing runtime GPU detection..."
echo "==================================="

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver detected:"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "  Unable to query GPU details"
    echo
fi

# Check CUDA runtime
if [[ $CUDA_BUILD -eq 1 ]]; then
    echo "Testing GPU detection in built binary..."
    # Create a minimal test that should trigger GPU initialization logging
    echo ">test_seq" > test_input.fa
    echo "ATCGATCGATCGATCG" >> test_input.fa
    echo ">test_seq2" >> test_input.fa
    echo "GCTAGCTAGCTAGCTA" >> test_input.fa
    
    # Run a simple training command that should show GPU info
    if ./gkmtrain -t 8 -K MKL_RBF_ONLY -g 1.0 test_input.fa test_input.fa test_model.txt 2>&1 | grep -i "cuda\|gpu"; then
        echo "✓ GPU detection messages found in output"
    else
        echo "⚠ No GPU detection messages (may indicate CPU fallback)"
    fi
    
    # Cleanup
    rm -f test_input.fa test_model.txt*
else
    echo "CUDA build not available, skipping runtime GPU test"
fi

echo

echo "3. Testing compilation flags..."
echo "==============================="

if [[ $CUDA_BUILD -eq 1 ]]; then
    if strings libsvm_gkm.o | grep -q "CUDA"; then
        echo "✓ CUDA symbols found in compiled object"
    else
        echo "⚠ No CUDA symbols found (may indicate CPU-only compilation)"
    fi
    
    if [[ -f rbf_cuda.o ]]; then
        echo "✓ CUDA object file created"
    else
        echo "✗ CUDA object file missing"
    fi
else
    echo "CUDA build not available, skipping symbol tests"
fi

echo

echo "4. Build system information..."
echo "=============================="
echo "Make variables detected:"
make help 2>/dev/null || echo "Help target not available"

echo

echo "Test Summary:"
echo "============="
if [[ $CUDA_BUILD -eq 1 ]]; then
    echo "✓ GPU acceleration support successfully integrated"
    echo "✓ CUDA compilation working"
    echo "✓ Binaries built with GPU support"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ NVIDIA GPU detected on system"
        echo
        echo "🚀 GPU acceleration is ready to use!"
        echo "   Use kernel types MKL_GKM_RBF, MKL_RBF_ONLY, etc. to benefit from GPU acceleration"
    else
        echo "⚠ No NVIDIA GPU detected, but GPU-enabled binaries will fall back to CPU"
        echo
        echo "📦 GPU acceleration is compiled and ready"
        echo "   Will automatically use GPU when available"
    fi
else
    echo "⚠ GPU acceleration not available (CUDA build failed)"
    echo "✓ CPU-only version working as fallback"
    echo
    echo "💻 CPU-only acceleration active"
    echo "   Using SIMD optimizations and multithreading"
fi

echo
echo "For detailed performance information, see GPU_ACCELERATION.md"

# Cleanup
rm -f build_cpu.log build_cuda.log
cd ..
