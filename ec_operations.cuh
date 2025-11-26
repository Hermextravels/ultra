/*
 * Complete modular arithmetic and EC operations
 * To be #included in ultimate_puzzle_solver.cu
 */

// ================ SECP256K1 CONSTANTS ================

__constant__ uint256_t SECP256K1_P = {
    {0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
     0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}
};

__constant__ uint256_t SECP256K1_N = {
    {0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
     0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}
};

__constant__ uint256_t SECP256K1_Gx = {
    {0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
     0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E}
};

__constant__ uint256_t SECP256K1_Gy = {
    {0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
     0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77}
};

// Endomorphism constants
// lambda used for scalar decomposition (not yet used here, added for future GLV split)
__constant__ uint256_t SECP256K1_LAMBDA = {
    {0xC05C30E0, 0x5363AD4C, 0x8812645A, 0xA5261C02,
     0x20816678, 0x122E22EA, 0x1B23BD72, 0xDF02967C}
};

// beta used for the endomorphism psi: psi(x, y) = (beta * x mod p, y)
__constant__ uint256_t SECP256K1_BETA = {
    {0x657C0710, 0x7AE96A2B, 0xAC3434E9, 0x6E64479E,
     0x12F58995, 0x9CF04975, 0x719501EE, 0xC1396C28}
};

// ================ MODULAR ARITHMETIC COMPLETE ================

__device__ __forceinline__ int cmp256(const uint256_t &a, const uint256_t &b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return 1;
        if (a.v[i] < b.v[i]) return -1;
    }
    return 0;
}

__device__ __forceinline__ void mod_add_complete(uint256_t &result, const uint256_t &a, const uint256_t &b, const uint256_t &mod) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a.v[i] + b.v[i] + carry;
        result.v[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    
    // Reduce if >= mod
    if (cmp256(result, mod) >= 0) {
        uint64_t borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t diff = (uint64_t)result.v[i] - mod.v[i] - borrow;
            result.v[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }
    }
}

__device__ __forceinline__ void mod_sub_complete(uint256_t &result, const uint256_t &a, const uint256_t &b, const uint256_t &mod) {
    uint64_t borrow = 0;
    int need_add = 0;
    
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a.v[i] - b.v[i] - borrow;
        result.v[i] = (uint32_t)diff;
        borrow = (diff >> 32) & 1;
    }
    
    // If result is negative, add mod
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t sum = (uint64_t)result.v[i] + mod.v[i] + carry;
            result.v[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
}

__device__ __forceinline__ void mod_mult_complete(uint256_t &result, const uint256_t &a, const uint256_t &b, const uint256_t &mod) {
    // Long multiplication with reduction
    uint256_t temp;
    for (int i = 0; i < 8; i++) temp.v[i] = 0;
    
    for (int i = 0; i < 8; i++) {
        if (a.v[i] == 0) continue;
        
        uint64_t carry = 0;
        for (int j = 0; j < 8 && i + j < 8; j++) {
            uint64_t prod = (uint64_t)a.v[i] * b.v[j] + temp.v[i+j] + carry;
            temp.v[i+j] = (uint32_t)prod;
            carry = prod >> 32;
        }
    }
    
    // Reduce modulo P using Barrett reduction approximation
    // For secp256k1, P = 2^256 - 2^32 - 977
    // Simplified reduction for this special form
    while (cmp256(temp, mod) >= 0) {
        uint64_t borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t diff = (uint64_t)temp.v[i] - mod.v[i] - borrow;
            temp.v[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }
    }
    
    result = temp;
}

// Modular inverse using Fermat's little theorem: a^(p-2) mod p
__device__ void mod_inv_complete(uint256_t &result, const uint256_t &a, const uint256_t &mod) {
    // Calculate exponent = mod - 2
    uint256_t exp;
    uint64_t borrow = 2;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)mod.v[i] - borrow;
        exp.v[i] = (uint32_t)diff;
        borrow = (diff >> 32) & 1;
    }
    
    // Binary exponentiation
    uint256_t base = a;
    for (int i = 0; i < 8; i++) result.v[i] = 0;
    result.v[0] = 1;  // result = 1
    
    for (int i = 0; i < 256; i++) {
        int word = i / 32;
        int bit = i % 32;
        
        if (exp.v[word] & (1u << bit)) {
            mod_mult_complete(result, result, base, mod);
        }
        
        if (i < 255) {
            mod_mult_complete(base, base, base, mod);
        }
    }
}

// ================ ELLIPTIC CURVE OPERATIONS ================

__device__ void ec_double_complete(ECPoint &result, const ECPoint &p) {
    if (p.infinity) {
        result.infinity = true;
        return;
    }
    
    // lambda = (3*x^2) / (2*y)
    uint256_t x2, three_x2, two_y, lambda;
    
    // x^2
    mod_mult_complete(x2, p.x, p.x, SECP256K1_P);
    
    // 3*x^2
    uint256_t three = {{3, 0, 0, 0, 0, 0, 0, 0}};
    mod_mult_complete(three_x2, x2, three, SECP256K1_P);
    
    // 2*y
    mod_add_complete(two_y, p.y, p.y, SECP256K1_P);
    
    // (2*y)^-1
    uint256_t two_y_inv;
    mod_inv_complete(two_y_inv, two_y, SECP256K1_P);
    
    // lambda = 3*x^2 * (2*y)^-1
    mod_mult_complete(lambda, three_x2, two_y_inv, SECP256K1_P);
    
    // x' = lambda^2 - 2*x
    uint256_t lambda2, two_x;
    mod_mult_complete(lambda2, lambda, lambda, SECP256K1_P);
    mod_add_complete(two_x, p.x, p.x, SECP256K1_P);
    mod_sub_complete(result.x, lambda2, two_x, SECP256K1_P);
    
    // y' = lambda*(x - x') - y
    uint256_t x_diff, temp;
    mod_sub_complete(x_diff, p.x, result.x, SECP256K1_P);
    mod_mult_complete(temp, lambda, x_diff, SECP256K1_P);
    mod_sub_complete(result.y, temp, p.y, SECP256K1_P);
    
    result.infinity = false;
}

__device__ void ec_add_complete(ECPoint &result, const ECPoint &p, const ECPoint &q) {
    if (p.infinity) {
        result = q;
        return;
    }
    if (q.infinity) {
        result = p;
        return;
    }
    
    // Check if same x coordinate
    if (cmp256(p.x, q.x) == 0) {
        if (cmp256(p.y, q.y) == 0) {
            // Same point, do doubling
            ec_double_complete(result, p);
        } else {
            // Inverse points, result is infinity
            result.infinity = true;
        }
        return;
    }
    
    // lambda = (y2 - y1) / (x2 - x1)
    uint256_t y_diff, x_diff, x_diff_inv, lambda;
    mod_sub_complete(y_diff, q.y, p.y, SECP256K1_P);
    mod_sub_complete(x_diff, q.x, p.x, SECP256K1_P);
    mod_inv_complete(x_diff_inv, x_diff, SECP256K1_P);
    mod_mult_complete(lambda, y_diff, x_diff_inv, SECP256K1_P);
    
    // x' = lambda^2 - x1 - x2
    uint256_t lambda2, temp;
    mod_mult_complete(lambda2, lambda, lambda, SECP256K1_P);
    mod_sub_complete(temp, lambda2, p.x, SECP256K1_P);
    mod_sub_complete(result.x, temp, q.x, SECP256K1_P);
    
    // y' = lambda*(x1 - x') - y1
    uint256_t x_diff2, temp2;
    mod_sub_complete(x_diff2, p.x, result.x, SECP256K1_P);
    mod_mult_complete(temp2, lambda, x_diff2, SECP256K1_P);
    mod_sub_complete(result.y, temp2, p.y, SECP256K1_P);
    
    result.infinity = false;
}

// Scalar multiplication using double-and-add
__device__ void ec_mult(ECPoint &result, const uint256_t &scalar, const ECPoint &base) {
    result.infinity = true;
    ECPoint temp = base;
    
    for (int i = 0; i < 256; i++) {
        int word = i / 32;
        int bit = i % 32;
        
        if (scalar.v[word] & (1u << bit)) {
            if (result.infinity) {
                result = temp;
            } else {
                ec_add_complete(result, result, temp);
            }
        }
        
        if (i < 255) {
            ec_double_complete(temp, temp);
        }
    }
}

// ================ BATCH EC MULTIPLICATION (BitCrack style) ================

__device__ void batch_ec_add_optimized(ECPoint *points, int count, uint256_t *scalars) {
    // Montgomery's trick: compute all modular inverses at once
    // For n inversions, needs only 1 expensive inversion + 3n multiplications
    
    if (count <= 0) return;
    
    // Compute products: prod[i] = (x2-x1)_0 * (x2-x1)_1 * ... * (x2-x1)_i
    uint256_t *products = (uint256_t *)malloc(count * sizeof(uint256_t));
    uint256_t *x_diffs = (uint256_t *)malloc(count * sizeof(uint256_t));
    
    ECPoint G;
    G.x = SECP256K1_Gx;
    G.y = SECP256K1_Gy;
    G.infinity = false;
    
    // Precompute current points
    for (int i = 0; i < count; i++) {
        ec_mult(points[i], scalars[i], G);
    }
    
    // Compute differences for lambda calculations
    for (int i = 0; i < count; i++) {
        // For adding G to points[i], x_diff = Gx - points[i].x
        mod_sub_complete(x_diffs[i], SECP256K1_Gx, points[i].x, SECP256K1_P);
        
        if (i == 0) {
            products[0] = x_diffs[0];
        } else {
            mod_mult_complete(products[i], products[i-1], x_diffs[i], SECP256K1_P);
        }
    }
    
    // Single expensive inversion
    uint256_t inv_product;
    mod_inv_complete(inv_product, products[count-1], SECP256K1_P);
    
    // Compute individual inverses in reverse
    uint256_t *x_diff_invs = (uint256_t *)malloc(count * sizeof(uint256_t));
    for (int i = count - 1; i >= 0; i--) {
        if (i > 0) {
            mod_mult_complete(x_diff_invs[i], inv_product, products[i-1], SECP256K1_P);
            mod_mult_complete(inv_product, inv_product, x_diffs[i], SECP256K1_P);
        } else {
            x_diff_invs[0] = inv_product;
        }
    }
    
    // Now compute all point additions with precomputed inverses
    for (int i = 0; i < count; i++) {
        uint256_t lambda, y_diff;
        mod_sub_complete(y_diff, SECP256K1_Gy, points[i].y, SECP256K1_P);
        mod_mult_complete(lambda, y_diff, x_diff_invs[i], SECP256K1_P);
        
        // x' = lambda^2 - x1 - Gx
        uint256_t lambda2, temp;
        mod_mult_complete(lambda2, lambda, lambda, SECP256K1_P);
        mod_sub_complete(temp, lambda2, points[i].x, SECP256K1_P);
        mod_sub_complete(points[i].x, temp, SECP256K1_Gx, SECP256K1_P);
        
        // y' = lambda*(old_x - x') - old_y
        uint256_t old_x = points[i].x;  // Save before overwrite
        uint256_t x_diff_final, temp2;
        mod_sub_complete(x_diff_final, old_x, points[i].x, SECP256K1_P);
        mod_mult_complete(temp2, lambda, x_diff_final, SECP256K1_P);
        mod_sub_complete(points[i].y, temp2, points[i].y, SECP256K1_P);
    }
    
    free(products);
    free(x_diffs);
    free(x_diff_invs);
}
