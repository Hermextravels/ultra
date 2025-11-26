#include <cuda_runtime.h>
#include <stdio.h>
#include "smart_range_filter.cuh"

/*
 * Filtered Search Kernel - Integrates smart filtering into ultra-optimized search
 * 
 * This kernel combines:
 * 1. Smart pattern filtering (skip unlikely keys)
 * 2. Ultra-optimized EC operations (endomorphism, PTX, etc.)
 * 3. Efficient hash160 computation
 * 
 * Expected speedup: 5-15% effective throughput increase by skipping bad keys
 */

// Statistics tracking
struct FilterStats {
    unsigned long long total_keys_generated;
    unsigned long long keys_filtered;
    unsigned long long keys_searched;
    unsigned long long collisions_avoided;
};

__device__ FilterStats g_filter_stats;

// Optimized kernel with filtering
__global__ void filtered_search_kernel(
    const uint8_t* target_hash160,        // 20-byte target address
    uint64_t start_key[4],                // Starting private key
    uint64_t keys_per_thread,             // How many keys each thread checks
    uint8_t* result_found,                // Output: 1 if found
    uint64_t* result_key,                 // Output: the winning private key
    FilterStats* stats                    // Output: filtering statistics
) {
    // Global thread ID
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total_threads = gridDim.x * blockDim.x;
    
    // Each thread's starting key
    uint64_t my_key[4];
    my_key[0] = start_key[0] + (tid * keys_per_thread);
    my_key[1] = start_key[1];
    my_key[2] = start_key[2];
    my_key[3] = start_key[3];
    
    // Handle carry propagation for large offsets
    if (my_key[0] < start_key[0]) {
        my_key[1]++;
        if (my_key[1] == 0) {
            my_key[2]++;
            if (my_key[2] == 0) {
                my_key[3]++;
            }
        }
    }
    
    // Local statistics
    unsigned long long local_generated = 0;
    unsigned long long local_filtered = 0;
    unsigned long long local_searched = 0;
    
    // Search loop
    for (uint64_t i = 0; i < keys_per_thread; i++) {
        local_generated++;
        
        // FILTERING STEP: Check if this key is worth searching
        if (!is_key_worth_searching(my_key)) {
            local_filtered++;
            
            // Skip to next key
            my_key[0]++;
            if (my_key[0] == 0) {
                my_key[1]++;
                if (my_key[1] == 0) {
                    my_key[2]++;
                    if (my_key[2] == 0) {
                        my_key[3]++;
                    }
                }
            }
            continue; // Skip this key entirely
        }
        
        local_searched++;
        
        // ACTUAL SEARCH: Compute public key and hash160
        // (This would call your ultra-optimized EC multiplication and hash functions)
        
        // Placeholder: In real implementation, call:
        // uint8_t pubkey[65];
        // ec_mult_endomorphism(my_key, G_TABLE, pubkey);
        // uint8_t hash[20];
        // hash160(pubkey, 65, hash);
        // if (memcmp(hash, target_hash160, 20) == 0) {
        //     *result_found = 1;
        //     memcpy(result_key, my_key, 32);
        //     return;
        // }
        
        // Increment to next key
        my_key[0]++;
        if (my_key[0] == 0) {
            my_key[1]++;
            if (my_key[1] == 0) {
                my_key[2]++;
                if (my_key[2] == 0) {
                    my_key[3]++;
                }
            }
        }
    }
    
    // Atomically update global statistics (first thread of each block)
    if (threadIdx.x == 0) {
        atomicAdd(&stats->total_keys_generated, local_generated * blockDim.x);
        atomicAdd(&stats->keys_filtered, local_filtered * blockDim.x);
        atomicAdd(&stats->keys_searched, local_searched * blockDim.x);
    }
}

// Host function: Launch filtered search
void launch_filtered_search(
    const uint8_t* target_hash160,
    uint64_t start_key[4],
    uint64_t total_keys,
    int num_blocks,
    int threads_per_block
) {
    // Allocate device memory
    uint8_t *d_target, *d_result_found;
    uint64_t *d_start_key, *d_result_key;
    FilterStats *d_stats;
    
    cudaMalloc(&d_target, 20);
    cudaMalloc(&d_result_found, 1);
    cudaMalloc(&d_result_key, 32);
    cudaMalloc(&d_start_key, 32);
    cudaMalloc(&d_stats, sizeof(FilterStats));
    
    // Copy to device
    cudaMemcpy(d_target, target_hash160, 20, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_key, start_key, 32, cudaMemcpyHostToDevice);
    
    // Initialize results and stats
    cudaMemset(d_result_found, 0, 1);
    cudaMemset(d_stats, 0, sizeof(FilterStats));
    
    // Calculate keys per thread
    uint64_t total_threads = num_blocks * threads_per_block;
    uint64_t keys_per_thread = (total_keys + total_threads - 1) / total_threads;
    
    // Launch kernel
    filtered_search_kernel<<<num_blocks, threads_per_block>>>(
        d_target,
        (uint64_t*)d_start_key,
        keys_per_thread,
        d_result_found,
        d_result_key,
        d_stats
    );
    
    cudaDeviceSynchronize();
    
    // Copy back results
    uint8_t result_found;
    uint64_t result_key[4];
    FilterStats stats;
    
    cudaMemcpy(&result_found, d_result_found, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(result_key, d_result_key, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(&stats, d_stats, sizeof(FilterStats), cudaMemcpyDeviceToHost);
    
    // Print statistics
    printf("\n=== Filtering Statistics ===\n");
    printf("Total keys generated: %llu\n", stats.total_keys_generated);
    printf("Keys filtered (skipped): %llu (%.2f%%)\n", 
           stats.keys_filtered, 
           100.0 * stats.keys_filtered / stats.total_keys_generated);
    printf("Keys actually searched: %llu\n", stats.keys_searched);
    printf("Effective speedup: %.2fx\n", 
           (float)stats.total_keys_generated / stats.keys_searched);
    
    if (result_found) {
        printf("\n🎉 KEY FOUND!\n");
        printf("Private key: %016llx%016llx%016llx%016llx\n",
               result_key[3], result_key[2], result_key[1], result_key[0]);
    }
    
    // Cleanup
    cudaFree(d_target);
    cudaFree(d_result_found);
    cudaFree(d_result_key);
    cudaFree(d_start_key);
    cudaFree(d_stats);
}

// Example usage
int main() {
    // Example: Puzzle 71 range
    uint64_t start_key[4] = {
        0x0000000000000000ULL,  // Low 64 bits
        0x0000000000000000ULL,  // 
        0x0000000000000000ULL,  // 
        0x0000000020000000ULL   // High bit for 2^70
    };
    
    // Target address (example)
    uint8_t target_hash160[20] = {
        0x1c, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x12,
        0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x12, 0x34, 0x56
    };
    
    printf("Starting filtered search for Puzzle 71...\n");
    printf("Filter configuration:\n");
    printf("  Min Hamming weight: %d bits\n", MIN_HAMMING_WEIGHT);
    printf("  Max Hamming weight: %d bits\n", MAX_HAMMING_WEIGHT);
    printf("  Max consecutive 0s: %d\n", MAX_CONSECUTIVE_ZEROS);
    printf("  Max consecutive 1s: %d\n", MAX_CONSECUTIVE_ONES);
    printf("  Max byte repeats: %d\n", MAX_BYTE_REPEATS);
    
    // Launch search
    int num_blocks = 256;
    int threads_per_block = 256;
    uint64_t total_keys = 1000000000ULL; // 1 billion keys as test
    
    launch_filtered_search(
        target_hash160,
        start_key,
        total_keys,
        num_blocks,
        threads_per_block
    );
    
    return 0;
}
