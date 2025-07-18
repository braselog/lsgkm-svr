#!/bin/bash

# Enhanced GPU Memory Management Demo for lsgkm-svr
# This script demonstrates the new adaptive batch sizing and memory monitoring features

set -e

echo "=== Enhanced GPU Memory Management Demo ==="
echo

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU features may not work."
    echo
fi

# Example data paths (adjust these to your actual data)
POSITIVE_FILE="data/positive_sequences.fa"
NEGATIVE_FILE="data/negative_sequences.fa"
OUTPUT_PREFIX="gpu_optimized_model"

# Create example data if it doesn't exist
if [ ! -f "$POSITIVE_FILE" ]; then
    echo "Creating example positive sequences..."
    mkdir -p data
    cat > "$POSITIVE_FILE" << 'EOF'
>seq1
ATCGATCGATCGATCG
>seq2
GCTAGCTAGCTAGCTA
>seq3
TTAATTAATTAATTAA
>seq4
CGGCCGGCCGGCCGGC
>seq5
AAAATTTTCCCCGGGG
EOF
fi

if [ ! -f "$NEGATIVE_FILE" ]; then
    echo "Creating example negative sequences..."
    cat > "$NEGATIVE_FILE" << 'EOF'
>seq1
TGCATGCATGCATGCA
>seq2
AGCTCGATCGATCGAT
>seq3
GGCCAATTGGCCAATT
>seq4
TTTTAAAACCCCGGGG
>seq5
CATGCATGCATGCATG
EOF
fi

echo "Data files prepared."
echo

# Step 1: Show GPU diagnostics before training
echo "Step 1: GPU Memory Diagnostics"
echo "=============================="
./gkmtrain --gpu-diag -t 4 -l 11 -k 7 -v 3 "$POSITIVE_FILE" "$NEGATIVE_FILE" "$OUTPUT_PREFIX" || {
    echo "GPU diagnostics completed with warnings (this is normal if no GPU is available)"
}
echo

# Step 2: Perform auto-tuning
echo "Step 2: GPU Batch Size Auto-Tuning"
echo "=================================="
./gkmtrain --gpu-tune -t 4 -l 11 -k 7 -v 3 "$POSITIVE_FILE" "$NEGATIVE_FILE" "${OUTPUT_PREFIX}_tuned" || {
    echo "Auto-tuning completed with warnings"
}
echo

# Step 3: Train with reserved GPU memory
echo "Step 3: Training with Reserved GPU Memory"
echo "========================================"
./gkmtrain --gpu-reserve 512 -t 4 -l 11 -k 7 -c 1.0 -v 3 "$POSITIVE_FILE" "$NEGATIVE_FILE" "${OUTPUT_PREFIX}_reserved" || {
    echo "Training completed with warnings"
}
echo

# Step 4: Demonstrate adaptive batch processing with different sizes
echo "Step 4: Adaptive Batch Processing Demo"
echo "====================================="

# Create larger dataset for better demonstration
LARGE_POS="data/large_positive.fa"
LARGE_NEG="data/large_negative.fa"

if [ ! -f "$LARGE_POS" ]; then
    echo "Creating larger dataset for batch processing demo..."
    
    # Generate more sequences
    cat > "$LARGE_POS" << 'EOF'
>seq1
ATCGATCGATCGATCGATCGATCGATCG
>seq2
GCTAGCTAGCTAGCTAGCTAGCTAGCTA
>seq3
TTAATTAATTAATTAATTAATTAATTAA
>seq4
CGGCCGGCCGGCCGGCCGGCCGGCCGGC
>seq5
AAAATTTTCCCCGGGGAAAATTTTCCCC
>seq6
ATCGATCGATCGATCGATCGATCGATCG
>seq7
GCTAGCTAGCTAGCTAGCTAGCTAGCTA
>seq8
TTAATTAATTAATTAATTAATTAATTAA
>seq9
CGGCCGGCCGGCCGGCCGGCCGGCCGGC
>seq10
AAAATTTTCCCCGGGGAAAATTTTCCCC
>seq11
ATCGATCGATCGATCGATCGATCGATCG
>seq12
GCTAGCTAGCTAGCTAGCTAGCTAGCTA
>seq13
TTAATTAATTAATTAATTAATTAATTAA
>seq14
CGGCCGGCCGGCCGGCCGGCCGGCCGGC
>seq15
AAAATTTTCCCCGGGGAAAATTTTCCCC
>seq16
ATCGATCGATCGATCGATCGATCGATCG
>seq17
GCTAGCTAGCTAGCTAGCTAGCTAGCTA
>seq18
TTAATTAATTAATTAATTAATTAATTAA
>seq19
CGGCCGGCCGGCCGGCCGGCCGGCCGGC
>seq20
AAAATTTTCCCCGGGGAAAATTTTCCCC
EOF

    cat > "$LARGE_NEG" << 'EOF'
>seq1
TGCATGCATGCATGCATGCATGCATGCA
>seq2
AGCTCGATCGATCGATCGATCGATCGAT
>seq3
GGCCAATTGGCCAATTGGCCAATTGGCC
>seq4
TTTTAAAACCCCGGGGTTTTAAAACCCC
>seq5
CATGCATGCATGCATGCATGCATGCATG
>seq6
TGCATGCATGCATGCATGCATGCATGCA
>seq7
AGCTCGATCGATCGATCGATCGATCGAT
>seq8
GGCCAATTGGCCAATTGGCCAATTGGCC
>seq9
TTTTAAAACCCCGGGGTTTTAAAACCCC
>seq10
CATGCATGCATGCATGCATGCATGCATG
>seq11
TGCATGCATGCATGCATGCATGCATGCA
>seq12
AGCTCGATCGATCGATCGATCGATCGAT
>seq13
GGCCAATTGGCCAATTGGCCAATTGGCC
>seq14
TTTTAAAACCCCGGGGTTTTAAAACCCC
>seq15
CATGCATGCATGCATGCATGCATGCATG
>seq16
TGCATGCATGCATGCATGCATGCATGCA
>seq17
AGCTCGATCGATCGATCGATCGATCGAT
>seq18
GGCCAATTGGCCAATTGGCCAATTGGCC
>seq19
TTTTAAAACCCCGGGGTTTTAAAACCCC
>seq20
CATGCATGCATGCATGCATGCATGCATG
EOF
fi

# Train with adaptive batch processing
./gkmtrain --gpu-diag --gpu-tune -t 4 -l 10 -k 6 -c 1.0 -v 3 "$LARGE_POS" "$LARGE_NEG" "${OUTPUT_PREFIX}_adaptive" || {
    echo "Adaptive training completed with warnings"
}

echo
echo "Step 5: Final GPU Statistics"
echo "==========================="
./gkmtrain --gpu-diag -t 4 -l 8 -k 5 -v 2 "$POSITIVE_FILE" "$NEGATIVE_FILE" "${OUTPUT_PREFIX}_final" || {
    echo "Final diagnostics completed with warnings"
}

echo
echo "=== Demo Complete ==="
echo
echo "Summary of enhanced features demonstrated:"
echo "1. Real-time GPU memory monitoring and pressure detection"
echo "2. Adaptive batch size optimization based on available memory"
echo "3. Auto-tuning of batch sizes for optimal performance"
echo "4. Memory reservation for concurrent GPU operations"
echo "5. Comprehensive diagnostics and performance statistics"
echo
echo "Generated model files:"
ls -la ${OUTPUT_PREFIX}*.model.txt 2>/dev/null || echo "No model files generated (expected if no valid training data)"

echo
echo "For production use:"
echo "- Use --gpu-tune once to find optimal batch sizes for your data"
echo "- Use --gpu-reserve to leave memory for other GPU applications"
echo "- Use --gpu-diag to troubleshoot memory issues"
echo "- The system automatically adapts batch sizes based on memory pressure"
