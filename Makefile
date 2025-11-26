#!/usr/bin/make -f
# Simple Makefile for building GPU solvers
# Usage:
#   make filtered_solver CUDA_ARCH=sm_86
#   make ultra_solver CUDA_ARCH=sm_86 ULTRA_SRC=ultra_optimized_kernel.cu
#   make filtered_solver_fatbin
#   make clean

.PHONY: all clean

CUDA_ARCH ?= sm_86
NVCC ?= nvcc
NVCC_FLAGS := -O3 --use_fast_math -Xptxas -O3,-v -arch=$(CUDA_ARCH)

# Ultra kernel source can be overridden: make ultra_solver ULTRA_SRC=path/to/file.cu
ULTRA_SRC ?= ultra_optimized_kernel.cu

all: filtered_solver

filtered_solver: filtered_search_kernel.cu smart_range_filter.cuh
	$(NVCC) $(NVCC_FLAGS) filtered_search_kernel.cu -o $@

# Fatbin build with multiple architectures
filtered_solver_fatbin: filtered_search_kernel.cu smart_range_filter.cuh
	$(NVCC) -O3 --use_fast_math -Xptxas -O3,-v \
	  -gencode arch=compute_75,code=sm_75 \
	  -gencode arch=compute_86,code=sm_86 \
	  -gencode arch=compute_89,code=sm_89 \
	  filtered_search_kernel.cu -o filtered_solver

ultra_solver: $(ULTRA_SRC)
	$(NVCC) $(NVCC_FLAGS) $(ULTRA_SRC) -o $@

ultra_solver_fatbin: $(ULTRA_SRC)
	$(NVCC) -O3 --use_fast_math -Xptxas -O3,-v \
	  -gencode arch=compute_75,code=sm_75 \
	  -gencode arch=compute_86,code=sm_86 \
	  -gencode arch=compute_89,code=sm_89 \
	  $(ULTRA_SRC) -o ultra_solver

clean:
	rm -f filtered_solver ultra_solver
# Ultimate Bitcoin Puzzle Solver - Makefile

# Compiler and flags
NVCC = nvcc
CXX = g++

# CUDA architecture (adjust for your GPUs)
# T4 = sm_75, A10 = sm_86
GPU_ARCH = -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86

# Optimization flags
NVCC_FLAGS = -O3 -std=c++14 --use_fast_math -Xptxas -O3 $(GPU_ARCH)
CXX_FLAGS = -O3 -std=c++14 -march=native

# Libraries
LIBS = -lcrypto -lssl -lgmp

# Source files
CUDA_SRC = ultimate_puzzle_solver.cu
CPP_SRC = host_utils.cpp
TARGET = ultimate_puzzle_solver

# Build
all: $(TARGET)

$(TARGET): $(CUDA_SRC) $(CPP_SRC)
	@echo "Building Ultimate Bitcoin Puzzle Solver..."
	@echo "GPU Architectures: T4 (sm_75) + A10 (sm_86)"
	$(NVCC) $(NVCC_FLAGS) $(CUDA_SRC) $(CPP_SRC) -o $(TARGET) $(LIBS)
	@echo "✅ Build complete!"
	@echo ""
	@echo "Usage:"
	@echo "  ./$(TARGET) --gpu=0 --start=20000000000000000 --end=3FFFFFFFFFFFFFFFF --address=1xxxxxxxxx"
	@echo ""

clean:
	rm -f $(TARGET) *.o

test: $(TARGET)
	@echo "Testing with known puzzle 66..."
	./$(TARGET) --gpu=0 --start=20000000000000000 --end=3FFFFFFFFFFFFFFFF \
		--address=13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so --test

benchmark: $(TARGET)
	@echo "Running benchmark..."
	./$(TARGET) --gpu=0 --start=1000000000 --end=2000000000 \
		--address=1xxxxxxxxx --benchmark

help:
	@echo "Ultimate Bitcoin Puzzle Solver - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build the solver"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make test     - Test with known solution (puzzle 66)"
	@echo "  make benchmark - Run performance benchmark"
	@echo "  make help     - Show this help"
	@echo ""
	@echo "GPU Support:"
	@echo "  Tesla T4  (sm_75)"
	@echo "  A10       (sm_86)"
	@echo ""

.PHONY: all clean test benchmark help
