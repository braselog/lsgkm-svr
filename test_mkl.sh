#!/bin/bash

# Test script for MKL functionality in lsgkm-svr

# Create test data directory
mkdir -p test_data

# Create simple test sequences
cat > test_data/positive.fa << 'EOF'
>seq1
ATCGATCGATCGATCG
>seq2
GCTAGCTAGCTAGCTA
>seq3
TGCATGCATGCATGCA
EOF

cat > test_data/negative.fa << 'EOF'
>seq4
CGTACGTACGTACGTA
>seq5
AGTCAGTCAGTCAGTC
>seq6
TCGATCGATCGATCGA
EOF

# Create test covariates (3 covariates per sequence)
cat > test_data/covariates.txt << 'EOF'
0.5	1.2	-0.3
0.8	-0.5	0.9
-0.2	0.7	1.1
1.3	-0.8	0.4
0.1	0.6	-0.7
-0.9	1.5	0.2
EOF

# Test 1: Standard GKM training (no MKL)
echo "Testing standard GKM training..."
./src/gkmtrain -t 4 -l 6 -k 3 -d 2 -c 1 -v 3 test_data/positive.fa test_data/negative.fa test_data/standard_model

# Test 2: MKL with GKM + RBF on covariates
echo "Testing MKL with GKM + RBF..."
./src/gkmtrain -t 7 -l 6 -k 3 -d 2 -c 1 -G 0.5 -W 0.6 -R 0.4 -I 50 -E 1e-4 -v 3 test_data/positive.fa test_data/negative.fa test_data/mkl_model test_data/covariates.txt

# Test 3: MKL with RBF only
echo "Testing MKL with RBF only..."
./src/gkmtrain -t 9 -c 1 -G 0.5 -v 3 test_data/positive.fa test_data/negative.fa test_data/rbf_model test_data/covariates.txt

# Test 4: MKL with GKM only (should behave like standard GKM)
echo "Testing MKL with GKM only..."
./src/gkmtrain -t 8 -l 6 -k 3 -d 2 -c 1 -v 3 test_data/positive.fa test_data/negative.fa test_data/gkm_only_model test_data/covariates.txt

echo "All tests completed!"
