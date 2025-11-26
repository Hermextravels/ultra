# Ultimate Solver Optimizations Applied

## ✅ Completed Optimizations (Nov 26, 2025)

### 1. Removed Unused Shared Memory
- **Before**: Allocated `sizeof(uint256_t) * threads * 3` for old xdiff/invd/ydiff buffers
- **After**: Zero shared memory allocation (removed smem parameter entirely)
- **Benefit**: Improved occupancy by reducing shared memory pressure

### 2. Added `--no-psi` Flag for Profiling
- **Default**: ψ(P) dual-check **enabled** (searches both P and ψ(P) per key)
- **Flag**: `--no-psi` to disable endomorphism check
- **Usage**: Profiling to measure EC vs hash bottlenecks
- **Status Display**: Shows `psi: on` or `psi: off` in launch message

### 3. GLV Scalar Multiplication (Framework)
- **Added**: `glv_decompose()` and `ec_mult_glv()` device functions
- **Current**: Falls back to `ec_mult_simple()` (placeholder for full GLV)
- **Integration Point**: Initial k·G computation now calls `ec_mult_glv()`
- **Next Step**: Implement full GLV split with mod-n arithmetic and dual-table precompute

## Current Feature Set

### Kernel Architecture
- **Persistent chunking**: Global atomic work allocation across blocks
- **Jacobian batching**: N-step batching (1-16) with single inversion per batch
- **ψ dual-check**: Optional endomorphism check (x ← β·x mod p)
- **Smart filtering**: Optional key filtering (disabled by default for correctness)
- **GLV-ready**: Framework in place for 30-40% k·G speedup

### CLI Options
```bash
./filtered_solver \
  --blocks 256 \
  --threads 256 \
  --total-keys 1G \
  --chunk-size 4M \
  --batch-steps 12 \
  --mode nofilter \
  --no-psi \
  --start-key <HEX64> \
  --target-hash160 <HEX40>
```

### Performance Tuning Guide
- **batch-steps**: 8-16 (balance register pressure vs inversion reduction)
- **chunk-size**: 1M-16M (balance fetch overhead vs cooperative exit latency)
- **mode**: `nofilter` (correct, all keys) | `filter` (faster, heuristic skip)
- **psi**: On by default (2x hash, catches endomorphism hits) | `--no-psi` (profiling)

## Build & Run
```bash
cd /Users/mac/Desktop/puzzle71/ultimate_solver
make clean && make
./filtered_solver --target-hash160 <puzzle_hash160> --start-key <range_start>
```

## Next Steps (Optional)
1. **Full GLV Implementation**:
   - Add mod-n arithmetic helpers
   - Implement lattice-based scalar decomposition: k = k1 + k2·λ mod n
   - Precompute dual tables: T_G[0..15] and T_ψG[0..15]
   - Replace `ec_mult_simple` fallback with dual-multiply

2. **Further Profiling**:
   - Use `--no-psi` to measure pure EC throughput
   - Nsight Compute analysis for warp occupancy and memory throughput
   - Tune batch-steps per architecture (sm_75 vs sm_86 vs sm_89)

3. **Multi-GPU**:
   - Distribute start_key ranges across GPUs
   - Unified result detection via host polling or CUDA IPC
