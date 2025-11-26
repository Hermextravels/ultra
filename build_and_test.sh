#!/bin/bash

echo "=========================================="
echo "Ultimate Bitcoin Puzzle Solver - Build & Test"
echo "=========================================="
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

# CUDA
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA found: $(nvcc --version | grep release | awk '{print $5}')"
else
    echo "❌ CUDA not found. Please install CUDA 12.x"
    exit 1
fi

# GPU check
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU found:"
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -2
else
    echo "⚠️  nvidia-smi not found. Cannot verify GPU"
fi

# OpenSSL
if [ -d "/usr/local/opt/openssl" ] || [ -d "/opt/homebrew/opt/openssl" ]; then
    echo "✅ OpenSSL found"
else
    echo "⚠️  OpenSSL not in standard location. Build may fail."
fi

# GMP
if [ -d "/usr/local/opt/gmp" ] || [ -d "/opt/homebrew/opt/gmp" ]; then
    echo "✅ GMP found"
else
    echo "⚠️  GMP not in standard location. Build may need adjustments."
fi

echo ""
echo "=========================================="
echo "📦 Building Ultimate Solver..."
echo "=========================================="

cd "$(dirname "$0")"

# Clean first
make clean 2>/dev/null

# Build
if make; then
    echo ""
    echo "=========================================="
    echo "✅ BUILD SUCCESSFUL!"
    echo "=========================================="
    echo ""
    
    # Show binary info
    if [ -f "./ultimate_puzzle_solver" ]; then
        echo "Binary size: $(ls -lh ultimate_puzzle_solver | awk '{print $5}')"
        echo "Binary path: $(pwd)/ultimate_puzzle_solver"
        echo ""
        
        # Test help
        echo "=========================================="
        echo "📖 Testing help output..."
        echo "=========================================="
        ./ultimate_puzzle_solver --help 2>&1 | head -20
        
        echo ""
        echo "=========================================="
        echo "✅ READY TO USE!"
        echo "=========================================="
        echo ""
        echo "Quick start:"
        echo ""
        echo "1. Test with known solution (puzzle 66):"
        echo "   ./ultimate_puzzle_solver \\"
        echo "     --gpu=0 \\"
        echo "     --start=20000000000000000 \\"
        echo "     --end=3FFFFFFFFFFFFFFFF \\"
        echo "     --address=13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"
        echo ""
        echo "2. Run on puzzle 71 (single GPU):"
        echo "   ./ultimate_puzzle_solver \\"
        echo "     --gpu=0 \\"
        echo "     --start=20000000000000000 \\"
        echo "     --end=3FFFFFFFFFFFFFFFF \\"
        echo "     --address=1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9 \\"
        echo "     --checkpoint=checkpoint_gpu0_puzzle71.json"
        echo ""
        echo "3. Multi-GPU orchestration:"
        echo "   python3 multi_gpu_orchestrator.py \\"
        echo "     --puzzles=unsolved_71_99.txt \\"
        echo "     --gpus=0,1 \\"
        echo "     --puzzle=71"
        echo ""
        echo "⚠️  Remember: Puzzle 71 will take ~37 years @ 1 GKey/s"
        echo ""
    else
        echo "❌ Binary not found after build!"
        exit 1
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ BUILD FAILED!"
    echo "=========================================="
    echo ""
    echo "Common issues:"
    echo ""
    echo "1. CUDA architecture mismatch:"
    echo "   Edit Makefile and adjust GPU_ARCH for your GPU"
    echo "   Find your GPU's compute capability:"
    echo "   nvidia-smi --query-gpu=compute_cap --format=csv,noheader"
    echo ""
    echo "2. OpenSSL/GMP not found:"
    echo "   macOS: brew install openssl gmp"
    echo "   Linux: sudo apt install libssl-dev libgmp-dev"
    echo ""
    echo "3. CUDA version mismatch:"
    echo "   This requires CUDA 12.x"
    echo "   Check: nvcc --version"
    echo ""
    exit 1
fi
