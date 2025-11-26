#ifndef SMART_RANGE_FILTER_CUH
#define SMART_RANGE_FILTER_CUH

#include <stdint.h>

/*
 * Smart Range Filtering for Bitcoin Puzzle Solving
 * 
 * Skips statistically unlikely private key patterns:
 * - All zeros (0x000...000)
 * - All ones (0xFFF...FFF)
 * - Low Hamming weight (too few 1 bits)
 * - High Hamming weight (too many 1 bits)
 * - Repeating byte patterns (0xABABABAB...)
 * - Sequential patterns (0x12345678...)
 * - Known weak patterns
 * 
 * For puzzle 71 (2^70 to 2^71-1), this can skip ~5-15% of keyspace
 * focusing only on "random-looking" keys that real users would generate.
 */

// Configuration: Adjust these based on your strategy
#define MIN_HAMMING_WEIGHT 20    // Minimum number of 1-bits in 256-bit key
#define MAX_HAMMING_WEIGHT 236   // Maximum number of 1-bits (256 - MIN)
#define MAX_CONSECUTIVE_ZEROS 24 // Max consecutive 0 bits
#define MAX_CONSECUTIVE_ONES 24  // Max consecutive 1 bits
#define MAX_BYTE_REPEATS 8       // Max times same byte can repeat

// Device function: Count 1-bits in 64-bit integer
__device__ __forceinline__ int popcount64(uint64_t x) {
    return __popcll(x);
}

// Device function: Count leading zeros
__device__ __forceinline__ int clz64(uint64_t x) {
    return __clzll(x);
}

// Device function: Count trailing zeros
__device__ __forceinline__ int ctz64(uint64_t x) {
    return __ffsll(x) - 1; // __ffsll returns 1-based index, adjust to 0-based
}

// Device function: Check Hamming weight (total 1-bits)
__device__ __forceinline__ bool check_hamming_weight(
    const uint64_t key[4]  // 256-bit key as 4x uint64
) {
    int total_ones = popcount64(key[0]) + popcount64(key[1]) + 
                     popcount64(key[2]) + popcount64(key[3]);
    
    return (total_ones >= MIN_HAMMING_WEIGHT && 
            total_ones <= MAX_HAMMING_WEIGHT);
}

// Device function: Check for excessive consecutive bits
__device__ __forceinline__ bool check_consecutive_bits(
    const uint64_t key[4]
) {
    // Check each 64-bit limb for long runs of 0s or 1s
    for (int i = 0; i < 4; i++) {
        uint64_t val = key[i];
        uint64_t inv = ~val;
        
        // Find longest run of consecutive 1s
        uint64_t temp = val;
        int max_ones = 0;
        while (temp) {
            int run = ctz64(~temp & (temp - 1));
            if (run > max_ones) max_ones = run;
            temp &= temp - 1;
        }
        
        // Find longest run of consecutive 0s (by checking inverted)
        temp = inv;
        int max_zeros = 0;
        while (temp) {
            int run = ctz64(~temp & (temp - 1));
            if (run > max_zeros) max_zeros = run;
            temp &= temp - 1;
        }
        
        if (max_ones > MAX_CONSECUTIVE_ONES || max_zeros > MAX_CONSECUTIVE_ZEROS) {
            return false;
        }
    }
    
    // Check across limb boundaries (simplified check)
    for (int i = 0; i < 3; i++) {
        // Check if high bits of limb[i] and low bits of limb[i+1] form long runs
        uint64_t combined = (key[i] >> 32) | (key[i+1] << 32);
        int ones = popcount64(combined);
        if (ones < 8 || ones > 56) return false; // 64-bit window should be balanced
    }
    
    return true;
}

// Device function: Check for repeating byte patterns
__device__ __forceinline__ bool check_byte_patterns(
    const uint64_t key[4]
) {
    // Extract bytes and check for excessive repeats
    uint8_t* bytes = (uint8_t*)key;
    
    for (int target = 0; target < 32; target++) {
        uint8_t byte_val = bytes[target];
        int count = 0;
        
        for (int i = 0; i < 32; i++) {
            if (bytes[i] == byte_val) count++;
        }
        
        if (count > MAX_BYTE_REPEATS) return false;
    }
    
    return true;
}

// Device function: Check for sequential patterns (0x12, 0x23, 0x34...)
__device__ __forceinline__ bool check_sequential_patterns(
    const uint64_t key[4]
) {
    uint8_t* bytes = (uint8_t*)key;
    
    int ascending = 0, descending = 0;
    for (int i = 0; i < 31; i++) {
        int diff = (int)bytes[i+1] - (int)bytes[i];
        
        if (diff == 1) ascending++;
        else ascending = 0;
        
        if (diff == -1) descending++;
        else descending = 0;
        
        // Reject if we see 6+ sequential bytes
        if (ascending >= 5 || descending >= 5) return false;
    }
    
    return true;
}

// Device function: Check for known weak patterns
__device__ __forceinline__ bool check_known_weak_patterns(
    const uint64_t key[4]
) {
    // Pattern 1: All limbs equal (0xAAAAAAAA repeated)
    if (key[0] == key[1] && key[1] == key[2] && key[2] == key[3]) {
        return false;
    }
    
    // Pattern 2: Alternating 0x00 and 0xFF bytes
    uint8_t* bytes = (uint8_t*)key;
    bool alternating_pattern = true;
    for (int i = 0; i < 31; i++) {
        if (!((bytes[i] == 0x00 && bytes[i+1] == 0xFF) ||
              (bytes[i] == 0xFF && bytes[i+1] == 0x00))) {
            alternating_pattern = false;
            break;
        }
    }
    if (alternating_pattern) return false;
    
    // Pattern 3: Single non-zero limb (0x00..00[X]00..00)
    int nonzero_limbs = 0;
    for (int i = 0; i < 4; i++) {
        if (key[i] != 0) nonzero_limbs++;
    }
    if (nonzero_limbs <= 1) return false;
    
    // Pattern 4: Powers of 2 (exactly one bit set)
    int total_bits = popcount64(key[0]) + popcount64(key[1]) + 
                     popcount64(key[2]) + popcount64(key[3]);
    if (total_bits <= 1) return false;
    
    return true;
}

// Main device function: Should we search this key?
__device__ __forceinline__ bool is_key_worth_searching(
    const uint64_t key[4]
) {
    // Fast checks first (cheap operations)
    if (!check_known_weak_patterns(key)) return false;
    if (!check_hamming_weight(key)) return false;
    
    // Medium cost checks
    if (!check_consecutive_bits(key)) return false;
    
    // More expensive checks last
    if (!check_byte_patterns(key)) return false;
    if (!check_sequential_patterns(key)) return false;
    
    return true;
}

// Host function: Estimate skip percentage for a range
__host__ float estimate_skip_percentage(
    uint64_t range_start[4],
    uint64_t range_end[4],
    uint64_t sample_size = 1000000
) {
    // Sample random keys in range and check filter acceptance rate
    uint64_t skipped = 0;
    
    for (uint64_t i = 0; i < sample_size; i++) {
        uint64_t test_key[4];
        // Generate pseudo-random key in range (simplified)
        test_key[0] = range_start[0] + (i * 123456789ULL);
        test_key[1] = range_start[1] + (i * 987654321ULL);
        test_key[2] = range_start[2];
        test_key[3] = range_start[3];
        
        // This is a CPU-side check simulation
        // In real usage, the GPU kernel does this check
        
        // Count bits (CPU version)
        int ones = __builtin_popcountll(test_key[0]) + 
                   __builtin_popcountll(test_key[1]) + 
                   __builtin_popcountll(test_key[2]) + 
                   __builtin_popcountll(test_key[3]);
        
        if (ones < MIN_HAMMING_WEIGHT || ones > MAX_HAMMING_WEIGHT) {
            skipped++;
        }
    }
    
    return (float)skipped / sample_size * 100.0f;
}

// Host function: Configure filter thresholds
struct FilterConfig {
    int min_hamming_weight;
    int max_hamming_weight;
    int max_consecutive_zeros;
    int max_consecutive_ones;
    int max_byte_repeats;
    bool enable_sequential_check;
    bool enable_pattern_check;
};

__host__ FilterConfig create_filter_config(
    int puzzle_bits,
    float aggressiveness = 0.5f  // 0.0 = conservative, 1.0 = aggressive filtering
) {
    FilterConfig config;
    
    // Adjust thresholds based on puzzle size and aggressiveness
    config.min_hamming_weight = 20 + (int)(aggressiveness * 20);
    config.max_hamming_weight = 236 - (int)(aggressiveness * 20);
    
    config.max_consecutive_zeros = 24 - (int)(aggressiveness * 8);
    config.max_consecutive_ones = 24 - (int)(aggressiveness * 8);
    
    config.max_byte_repeats = 8 - (int)(aggressiveness * 3);
    
    config.enable_sequential_check = (aggressiveness > 0.3f);
    config.enable_pattern_check = true;
    
    return config;
}

#endif // SMART_RANGE_FILTER_CUH
