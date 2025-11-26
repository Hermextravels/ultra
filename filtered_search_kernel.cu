#include <cuda_runtime.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "smart_range_filter.cuh"

// Minimal EC types for secp256k1 operations
struct uint256_t { uint32_t v[8]; };
struct ECPoint { uint256_t x; uint256_t y; bool infinity; };
struct ECPointJ { uint256_t X; uint256_t Y; uint256_t Z; bool infinity; };

// Bring in (simple) EC ops
#include "ec_operations.cuh"

// Inline GPU hash functions
__device__ __constant__ uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define EP1(x) (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define SIG0(x) (ROTR32(x, 7) ^ ROTR32(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR32(x, 17) ^ ROTR32(x, 19) ^ ((x) >> 10))

__device__ void sha256(uint8_t* hash, const uint8_t* data, uint32_t len) {
    uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint32_t w[64]; uint8_t block[64]; uint32_t bitlen = len * 8; uint32_t i, j; uint32_t blocks = (len + 9 + 63) / 64;
    for (uint32_t blk = 0; blk < blocks; blk++) {
        uint32_t block_start = blk * 64;
        for (i = 0; i < 64; i++) {
            if (block_start + i < len) block[i] = data[block_start + i];
            else if (block_start + i == len) block[i] = 0x80;
            else block[i] = 0x00;
        }
        if (blk == blocks - 1) {
            block[63] = (uint8_t)(bitlen); block[62] = (uint8_t)(bitlen >> 8);
            block[61] = (uint8_t)(bitlen >> 16); block[60] = (uint8_t)(bitlen >> 24);
        }
        for (i = 0; i < 16; i++) w[i] = ((uint32_t)block[i*4]<<24)|((uint32_t)block[i*4+1]<<16)|((uint32_t)block[i*4+2]<<8)|((uint32_t)block[i*4+3]);
        for (i = 16; i < 64; i++) w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];
        uint32_t a=state[0], b=state[1], c=state[2], d=state[3], e=state[4], f=state[5], g=state[6], h=state[7];
        for (i = 0; i < 64; i++) {
            uint32_t t1 = h + EP1(e) + CH(e,f,g) + K256[i] + w[i]; uint32_t t2 = EP0(a) + MAJ(a,b,c);
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d; state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }
    for (i=0; i<8; i++) { hash[i*4]=(state[i]>>24)&0xff; hash[i*4+1]=(state[i]>>16)&0xff; hash[i*4+2]=(state[i]>>8)&0xff; hash[i*4+3]=state[i]&0xff; }
}

#define ROTL32_RMD(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
__device__ __constant__ uint32_t RMD_K_LEFT[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
__device__ __constant__ uint32_t RMD_K_RIGHT[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};
__device__ __forceinline__ uint32_t rmd_f(uint32_t x, uint32_t y, uint32_t z, int round) {
    if (round<16) return x^y^z; if (round<32) return (x&y)|(~x&z);
    if (round<48) return (x|~y)^z; if (round<64) return (x&z)|(y&~z); return x^(y|~z);
}
__device__ void ripemd160(uint8_t* hash, const uint8_t* data, uint32_t len) {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    uint8_t block[64]; uint32_t w[16]; uint64_t bitlen = (uint64_t)len * 8;
    const int r_left[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    const int r_right[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    const int s_left[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    const int s_right[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    for (uint32_t blk=0; blk<(len+8+63)/64; blk++) {
        uint32_t block_start = blk * 64;
        for (int i=0; i<64; i++) {
            if (block_start+i<len) block[i]=data[block_start+i];
            else if (block_start+i==len) block[i]=0x80; else block[i]=0x00;
        }
        if (blk==((len+8+63)/64)-1) {
            block[56]=(uint8_t)(bitlen); block[57]=(uint8_t)(bitlen>>8); block[58]=(uint8_t)(bitlen>>16); block[59]=(uint8_t)(bitlen>>24);
            block[60]=(uint8_t)(bitlen>>32); block[61]=(uint8_t)(bitlen>>40); block[62]=(uint8_t)(bitlen>>48); block[63]=(uint8_t)(bitlen>>56);
        }
        for (int i=0; i<16; i++) w[i]=((uint32_t)block[i*4])|((uint32_t)block[i*4+1]<<8)|((uint32_t)block[i*4+2]<<16)|((uint32_t)block[i*4+3]<<24);
        uint32_t al=h[0], bl=h[1], cl=h[2], dl=h[3], el=h[4]; uint32_t ar=h[0], br=h[1], cr=h[2], dr=h[3], er=h[4];
        for (int j=0; j<80; j++) {
            uint32_t tl=ROTL32_RMD(al+rmd_f(bl,cl,dl,j)+w[r_left[j]]+RMD_K_LEFT[j/16], s_left[j])+el;
            al=el; el=dl; dl=ROTL32_RMD(cl,10); cl=bl; bl=tl;
            uint32_t tr=ROTL32_RMD(ar+rmd_f(br,cr,dr,79-j)+w[r_right[j]]+RMD_K_RIGHT[j/16], s_right[j])+er;
            ar=er; er=dr; dr=ROTL32_RMD(cr,10); cr=br; br=tr;
        }
        uint32_t tmp=h[1]+cl+dr; h[1]=h[2]+dl+er; h[2]=h[3]+el+ar; h[3]=h[4]+al+br; h[4]=h[0]+bl+cr; h[0]=tmp;
    }
    for (int i=0; i<5; i++) { hash[i*4]=(h[i]>>0)&0xff; hash[i*4+1]=(h[i]>>8)&0xff; hash[i*4+2]=(h[i]>>16)&0xff; hash[i*4+3]=(h[i]>>24)&0xff; }
}

// ----------------- Helpers -----------------
__device__ __forceinline__ bool is_zero256_dev(const uint256_t &a){ for(int i=0;i<8;i++) if(a.v[i]!=0) return false; return true; }

__device__ __forceinline__ void jacobian_from_affine(ECPointJ &R, const ECPoint &A) {
    R.X = A.x; R.Y = A.y; for (int i=0;i<8;i++) R.Z.v[i] = 0; R.Z.v[0] = 1u; R.infinity = A.infinity;
}

__device__ void jacobian_mixed_add(ECPointJ &R, const ECPoint &Q) {
    if (R.infinity) { jacobian_from_affine(R, Q); return; }
    if (Q.infinity) { return; }
    uint256_t Z1Z1, U2, S2, H, HH, HHH, r, V;
    mod_mult_complete(Z1Z1, R.Z, R.Z, SECP256K1_P);
    mod_mult_complete(U2, Q.x, Z1Z1, SECP256K1_P);
    uint256_t Z1_cu; mod_mult_complete(Z1_cu, Z1Z1, R.Z, SECP256K1_P);
    mod_mult_complete(S2, Q.y, Z1_cu, SECP256K1_P);
    mod_sub_complete(H, U2, R.X, SECP256K1_P);
    if (is_zero256_dev(H)) {
        mod_sub_complete(r, S2, R.Y, SECP256K1_P);
        if (is_zero256_dev(r)) {
            // Doubling via affine fallback
            uint256_t Zin, Zin2, Zin3;
            mod_inv_complete(Zin, R.Z, SECP256K1_P);
            mod_mult_complete(Zin2, Zin, Zin, SECP256K1_P);
            mod_mult_complete(Zin3, Zin2, Zin, SECP256K1_P);
            ECPoint A;
            mod_mult_complete(A.x, R.X, Zin2, SECP256K1_P);
            mod_mult_complete(A.y, R.Y, Zin3, SECP256K1_P);
            A.infinity = false; ECPoint D; ec_double_complete(D, A);
            jacobian_from_affine(R, D); return;
        } else { R.infinity = true; return; }
    }
    mod_mult_complete(HH, H, H, SECP256K1_P);
    mod_mult_complete(HHH, H, HH, SECP256K1_P);
    mod_sub_complete(r, S2, R.Y, SECP256K1_P);
    uint256_t two; for (int i=0;i<8;i++) two.v[i]=0; two.v[0]=2;
    mod_mult_complete(r, r, two, SECP256K1_P);
    mod_mult_complete(V, R.X, HH, SECP256K1_P);
    uint256_t r2, t, twoV;
    mod_mult_complete(r2, r, r, SECP256K1_P);
    mod_add_complete(twoV, V, V, SECP256K1_P);
    mod_sub_complete(t, r2, HHH, SECP256K1_P);
    mod_sub_complete(R.X, t, twoV, SECP256K1_P);
    uint256_t VminusX3, rVx, twoY1, twoY1HHH;
    mod_sub_complete(VminusX3, V, R.X, SECP256K1_P);
    mod_mult_complete(rVx, r, VminusX3, SECP256K1_P);
    mod_add_complete(twoY1, R.Y, R.Y, SECP256K1_P);
    mod_mult_complete(twoY1HHH, twoY1, HHH, SECP256K1_P);
    mod_sub_complete(R.Y, rVx, twoY1HHH, SECP256K1_P);
    uint256_t Z1plusH, Z1plusH2;
    mod_add_complete(Z1plusH, R.Z, H, SECP256K1_P);
    mod_mult_complete(Z1plusH2, Z1plusH, Z1plusH, SECP256K1_P);
    uint256_t Z3tmp;
    mod_sub_complete(Z3tmp, Z1plusH2, Z1Z1, SECP256K1_P);
    mod_sub_complete(R.Z, Z3tmp, HH, SECP256K1_P);
}

__device__ __forceinline__ void add_small_to_key(uint64_t out[4], const uint64_t base[4], unsigned long long add){
    out[0] = base[0] + add; out[1] = base[1]; out[2] = base[2]; out[3] = base[3];
    if (out[0] < base[0]) { out[1]++; if (!out[1]) { out[2]++; if (!out[2]) out[3]++; } }
}

__device__ void batch_invert_thread_fixed(uint256_t* invZ, const uint256_t* Zs, int B){
    uint256_t acc; for(int i=0;i<8;i++) acc.v[i]=(i==0)?1u:0u;
    uint256_t P[16];
    for(int i=0;i<B;i++){ P[i]=acc; mod_mult_complete(acc, acc, Zs[i], SECP256K1_P);} 
    uint256_t inv_acc; mod_inv_complete(inv_acc, acc, SECP256K1_P);
    for(int i=B-1;i>=0;i--){ uint256_t t; mod_mult_complete(t, P[i], inv_acc, SECP256K1_P); invZ[i]=t; mod_mult_complete(inv_acc, inv_acc, Zs[i], SECP256K1_P);} 
}

__device__ __forceinline__ int get_bit_le(const uint256_t& k, int i){ int w=i>>5; int b=i&31; return (k.v[w]>>b)&1; }

__device__ void ec_mult_simple(ECPoint &R, const uint256_t &k, const ECPoint &G){
    R.infinity = true; ECPoint Q = G;
    for(int i=0;i<256;i++){
        if (get_bit_le(k, i)) { if (R.infinity) { R = Q; R.infinity=false; } else { ECPoint T; ec_add_complete(T, R, Q); R = T; } }
        ECPoint D; ec_double_complete(D, Q); Q = D;
    }
}

// GLV scalar decomposition: k = k1 + k2*lambda mod n
__device__ void glv_decompose(uint256_t &k1, uint256_t &k2, bool &neg1, bool &neg2, const uint256_t &k) {
    // Approximation: split k using precomputed constants
    // For production: k2 ≈ k*b2/n, k1 = k - k2*lambda mod n
    // Simplified: k1=k[0..127], k2=k[128..255] with adjustments
    // This is a placeholder; full GLV needs mod-n arithmetic
    k1 = k; k2.v[0]=0; for(int i=1;i<8;i++) k2.v[i]=0;
    neg1=false; neg2=false;
    // TODO: Implement proper GLV split with lattice basis reduction
}

__device__ void ec_mult_glv(ECPoint &R, const uint256_t &k, const ECPoint &G){
    // For now, fall back to simple multiply
    // Full GLV needs: decompose k, compute Q=psi(G), then k1*G + k2*Q
    ec_mult_simple(R, k, G);
}

// ----------------- Kernel -----------------
extern "C" __global__ void filtered_search_kernel(
    const uint8_t* __restrict__ target_hash160,
    const uint64_t* __restrict__ start_key,
    unsigned long long total_keys,
    unsigned long long chunk_size,
    int batch_steps,
    int use_filter,
    int use_psi,
    volatile uint8_t* __restrict__ result_found,
    uint64_t* __restrict__ result_key,
    FilterStats* __restrict__ stats,
    unsigned long long* __restrict__ next_chunk_ptr
){

    // Predefine G
    ECPoint G; G.x = SECP256K1_Gx; G.y = SECP256K1_Gy; G.infinity=false;

    __shared__ unsigned long long chunk_start, chunk_count;
    unsigned long long local_generated=0ULL, local_filtered=0ULL, local_searched=0ULL;

    for(;;){
        if (threadIdx.x==0){
            unsigned long long start = atomicAdd(next_chunk_ptr, chunk_size);
            chunk_start = start;
            unsigned long long remaining = (start < total_keys) ? (total_keys - start) : 0ULL;
            chunk_count = remaining > chunk_size ? chunk_size : remaining;
        }
        __syncthreads();
        if (chunk_count==0ULL) break;

        unsigned int t = threadIdx.x, T = blockDim.x;
        unsigned long long base = chunk_start;
        unsigned long long q = chunk_count / T; unsigned int r = (unsigned int)(chunk_count % T);
        unsigned long long my_count = q + (t < r ? 1ULL : 0ULL);
        unsigned long long my_off = q * t + (t < r ? t : r);
        if (my_count==0ULL){ __syncthreads(); continue; }

        uint64_t my_key[4] = { start_key[0] + my_off + base, start_key[1], start_key[2], start_key[3] };
        if (my_key[0] < start_key[0]) { my_key[1]++; if (!my_key[1]) { my_key[2]++; if (!my_key[2]) my_key[3]++; } }

        uint256_t k;
        k.v[0]=(uint32_t)(my_key[0]&0xffffffffULL); k.v[1]=(uint32_t)(my_key[0]>>32);
        k.v[2]=(uint32_t)(my_key[1]&0xffffffffULL); k.v[3]=(uint32_t)(my_key[1]>>32);
        k.v[4]=(uint32_t)(my_key[2]&0xffffffffULL); k.v[5]=(uint32_t)(my_key[2]>>32);
        k.v[6]=(uint32_t)(my_key[3]&0xffffffffULL); k.v[7]=(uint32_t)(my_key[3]>>32);

        ECPoint P0; ec_mult_glv(P0, k, G);
        ECPointJ PJ; jacobian_from_affine(PJ, P0);
        const int MAX_BATCH=16; int BATCH=batch_steps; if(BATCH<1)BATCH=8; if(BATCH>MAX_BATCH)BATCH=MAX_BATCH;
        unsigned long long processed=0ULL; uint64_t base_key[4]={my_key[0],my_key[1],my_key[2],my_key[3]};

        while (processed < my_count && !*result_found){
            int B = (int)((my_count - processed) < (unsigned long long)BATCH ? (my_count - processed) : (unsigned long long)BATCH);
            uint256_t Xs[MAX_BATCH], Ys[MAX_BATCH], Zs[MAX_BATCH], invZ[MAX_BATCH];
            for(int i=0;i<B;i++){ Xs[i]=PJ.X; Ys[i]=PJ.Y; Zs[i]=PJ.Z; local_generated++; jacobian_mixed_add(PJ, G); }
            batch_invert_thread_fixed(invZ, Zs, B);
            for(int i=0;i<B && !*result_found;i++){
                bool do_hash=true;
                if (use_filter){ uint64_t cand[4]; add_small_to_key(cand, base_key, (unsigned long long)(processed+i)); do_hash=is_key_worth_searching(cand); if(!do_hash){ local_filtered++; continue; } else local_searched++; }
                else local_searched++;
                uint256_t invZ2, invZ3, x_aff, y_aff;
                mod_mult_complete(invZ2, invZ[i], invZ[i], SECP256K1_P);
                mod_mult_complete(invZ3, invZ2, invZ[i], SECP256K1_P);
                mod_mult_complete(x_aff, Xs[i], invZ2, SECP256K1_P);
                mod_mult_complete(y_aff, Ys[i], invZ3, SECP256K1_P);
                uint8_t pk[33]; pk[0]=(y_aff.v[0]&1)?0x03:0x02;
                for(int w=0;w<8;w++){ uint32_t word=x_aff.v[7-w]; pk[1+w*4+0]=(word>>24)&0xFF; pk[1+w*4+1]=(word>>16)&0xFF; pk[1+w*4+2]=(word>>8)&0xFF; pk[1+w*4+3]=(word>>0)&0xFF; }
                uint8_t sha[32], h160[20]; sha256(sha, pk, 33); ripemd160(h160, sha, 32);
                bool matched=true; for(int b=0;b<20;b++){ if(h160[b]!=target_hash160[b]){ matched=false; break; } }
                if(matched){ *result_found=1; uint64_t rk[4]; add_small_to_key(rk, base_key, (unsigned long long)(processed+i)); result_key[0]=rk[0]; result_key[1]=rk[1]; result_key[2]=rk[2]; result_key[3]=rk[3]; break; }
                if (use_psi) {
                    uint256_t psi_x; mod_mult_complete(psi_x, x_aff, SECP256K1_BETA, SECP256K1_P);
                    uint8_t pk2[33]; pk2[0]=pk[0]; for(int w=0;w<8;w++){ uint32_t word=psi_x.v[7-w]; pk2[1+w*4+0]=(word>>24)&0xFF; pk2[1+w*4+1]=(word>>16)&0xFF; pk2[1+w*4+2]=(word>>8)&0xFF; pk2[1+w*4+3]=(word>>0)&0xFF; }
                    sha256(sha, pk2, 33); ripemd160(h160, sha, 32); matched=true; for(int b=0;b<20;b++){ if(h160[b]!=target_hash160[b]){ matched=false; break; } }
                    if(matched){ *result_found=1; uint64_t rk[4]; add_small_to_key(rk, base_key, (unsigned long long)(processed+i)); result_key[0]=rk[0]; result_key[1]=rk[1]; result_key[2]=rk[2]; result_key[3]=rk[3]; break; }
                }
            }
            processed += B;
        }
        __syncthreads(); if (*result_found) break;
    }

    atomicAdd(&stats->total_keys_generated, (unsigned long long)local_generated);
    atomicAdd(&stats->keys_filtered, (unsigned long long)local_filtered);
    atomicAdd(&stats->keys_searched, (unsigned long long)local_searched);
}

// ----------------- Host Launcher -----------------
void launch_filtered_search(
    const uint8_t* target_hash160,
    uint64_t start_key[4],
    uint64_t total_keys,
    int num_blocks,
    int threads_per_block,
    int use_filter,
    int use_psi,
    uint64_t chunk_size_cli,
    int batch_steps
) {
    // Allocate device memory
    uint8_t *d_target, *d_result_found;
    uint64_t *d_start_key, *d_result_key;
    FilterStats *d_stats;
    unsigned long long *d_next_chunk;

    cudaMalloc(&d_target, 20);
    cudaMalloc(&d_result_found, 1);
    cudaMalloc(&d_result_key, 32);
    cudaMalloc(&d_start_key, 32);
    cudaMalloc(&d_stats, sizeof(FilterStats));
    cudaMalloc(&d_next_chunk, sizeof(unsigned long long));

    cudaMemcpy(d_target, target_hash160, 20, cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_key, start_key, 32, cudaMemcpyHostToDevice);

    cudaMemset(d_result_found, 0, 1);
    cudaMemset(d_stats, 0, sizeof(FilterStats));
    cudaMemset(d_next_chunk, 0, sizeof(unsigned long long));

    uint64_t chunk_size = (chunk_size_cli > 0) ? chunk_size_cli : (1024ULL * 1024ULL);

    filtered_search_kernel<<<num_blocks, threads_per_block>>>(
        d_target,
        (uint64_t*)d_start_key,
        total_keys,
        chunk_size,
        batch_steps,
        use_filter,
        use_psi,
        d_result_found,
        d_result_key,
        d_stats,
        d_next_chunk
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(err));
    }

    uint8_t result_found;
    uint64_t result_key[4];
    FilterStats stats;

    cudaMemcpy(&result_found, d_result_found, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(result_key, d_result_key, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(&stats, d_stats, sizeof(FilterStats), cudaMemcpyDeviceToHost);

    printf("\n=== Filtering Statistics ===\n");
    printf("Total keys generated: %" PRIu64 "\n", (uint64_t)stats.total_keys_generated);
    if (stats.total_keys_generated > 0) {
        double skip_pct = 100.0 * (double)stats.keys_filtered / (double)stats.total_keys_generated;
        printf("Keys filtered (skipped): %" PRIu64 " (%.2f%%)\n", (uint64_t)stats.keys_filtered, skip_pct);
    } else {
        printf("Keys filtered (skipped): %" PRIu64 " (0.00%%)\n", (uint64_t)stats.keys_filtered);
    }
    printf("Keys actually searched: %" PRIu64 "\n", (uint64_t)stats.keys_searched);
    if (stats.keys_searched > 0) {
        double eff = (double)stats.total_keys_generated / (double)stats.keys_searched;
        printf("Effective speedup: %.2fx\n", eff);
    } else {
        printf("Effective speedup: 1.00x\n");
    }

    if (result_found) {
         printf("\n🎉 KEY FOUND!\n");
         printf("Private key: %016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "\n",
             (uint64_t)result_key[3], (uint64_t)result_key[2], (uint64_t)result_key[1], (uint64_t)result_key[0]);
    }

    cudaFree(d_target);
    cudaFree(d_result_found);
    cudaFree(d_result_key);
    cudaFree(d_start_key);
    cudaFree(d_stats);
    cudaFree(d_next_chunk);
}

// ----------------- CLI Main -----------------
int main(int argc, char **argv) {
    auto is_hex = [](char c)->int { return isxdigit((unsigned char)c); };
    auto hex_to_byte = [](char h, char l)->uint8_t {
        auto n = [](char c)->uint8_t {
            if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
            if (c >= 'a' && c <= 'f') return (uint8_t)(10 + c - 'a');
            if (c >= 'A' && c <= 'F') return (uint8_t)(10 + c - 'A');
            return (uint8_t)0;
        };
        return (uint8_t)((n(h) << 4) | n(l));
    };

    auto parse_u64_with_suffix = [](const char* s)->uint64_t {
        char *end = nullptr; long double v = strtold(s, &end);
        if (end && *end) { switch (*end) { case 'k': case 'K': v *= 1e3L; break; case 'm': case 'M': v *= 1e6L; break; case 'g': case 'G': v *= 1e9L; break; case 't': case 'T': v *= 1e12L; break; default: break; } }
        if (v < 0) v = 0; if (v > (long double)UINT64_MAX) v = (long double)UINT64_MAX; return (uint64_t)(v + 0.5L);
    };

    int num_blocks = 256;
    int threads_per_block = 256;
    uint64_t total_keys = 1000000000ULL;
    uint64_t chunk_size = 1024ULL * 1024ULL;
    int use_filter = 0;
    int use_psi = 1;
    int batch_steps = 8;
    uint64_t start_key[4] = { 0ULL, 0ULL, 0ULL, 0x0000000020000000ULL };
    uint8_t target_hash160[20] = { 0 };

    auto show_help = [](){
        printf("Usage: filtered_solver [--blocks N] [--threads N] [--total-keys N|N[KMG]]\n");
        printf("                      [--target-hash160 HEX40] [--start-key HEX64] [--chunk-size N|N[KMG]]\n");
        printf("                      [--mode auto|filter|nofilter] [--batch-steps N<=16] [--no-psi]\n");
    };

    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) { show_help(); return 0; }
        else if (!strcmp(a, "--blocks") && i+1 < argc) { num_blocks = (int)strtoul(argv[++i], NULL, 10); }
        else if (!strcmp(a, "--threads") && i+1 < argc) { threads_per_block = (int)strtoul(argv[++i], NULL, 10); }
        else if (!strcmp(a, "--total-keys") && i+1 < argc) { total_keys = parse_u64_with_suffix(argv[++i]); }
        else if (!strcmp(a, "--chunk-size") && i+1 < argc) { chunk_size = parse_u64_with_suffix(argv[++i]); }
        else if (!strcmp(a, "--mode") && i+1 < argc) { const char* m = argv[++i]; if (!strcmp(m, "filter")) use_filter=1; else if(!strcmp(m, "nofilter")) use_filter=0; else use_filter=0; }
        else if (!strcmp(a, "--batch-steps") && i+1 < argc) { batch_steps = (int)strtoul(argv[++i], NULL, 10); if (batch_steps < 1) batch_steps = 1; if (batch_steps > 16) batch_steps = 16; }
        else if (!strcmp(a, "--no-psi")) { use_psi = 0; }
        else if (!strcmp(a, "--target-hash160") && i+1 < argc) {
            const char* hex = argv[++i]; size_t len = strlen(hex);
            if (len == 40) { int ok=1; for (size_t j=0;j<40;j++) if(!is_hex(hex[j])){ ok=0; break; } if (ok) { for(int j=0;j<20;j++) target_hash160[j] = hex_to_byte(hex[2*j], hex[2*j+1]); } }
        } else if (!strcmp(a, "--start-key") && i+1 < argc) {
            const char* hex = argv[++i]; size_t len = strlen(hex);
            if (len == 64) {
                int ok=1; for(size_t j=0;j<64;j++) if(!is_hex(hex[j])){ ok=0; break; }
                if (ok) {
                    uint8_t bytes[32]; for (int j = 0; j < 32; j++) bytes[j] = hex_to_byte(hex[2*j], hex[2*j+1]);
                    auto load_be64 = [](const uint8_t* p)->uint64_t { return ((uint64_t)p[0]<<56)|((uint64_t)p[1]<<48)|((uint64_t)p[2]<<40)|((uint64_t)p[3]<<32)|((uint64_t)p[4]<<24)|((uint64_t)p[5]<<16)|((uint64_t)p[6]<<8)|((uint64_t)p[7]); };
                    start_key[3] = load_be64(&bytes[0]);
                    start_key[2] = load_be64(&bytes[8]);
                    start_key[1] = load_be64(&bytes[16]);
                    start_key[0] = load_be64(&bytes[24]);
                } else {
                    fprintf(stderr, "Invalid hex in --start-key, using default.\n");
                }
            } else {
                fprintf(stderr, "--start-key requires 64 hex chars, got %zu. Using default.\n", len);
            }
        }
    }

    printf("Starting filtered search...\n");
    printf("Filter configuration:\n");
    printf("  Min Hamming weight: %d bits\n", MIN_HAMMING_WEIGHT);
    printf("  Max Hamming weight: %d bits\n", MAX_HAMMING_WEIGHT);
    printf("  Max consecutive 0s: %d\n", MAX_CONSECUTIVE_ZEROS);
    printf("  Max consecutive 1s: %d\n", MAX_CONSECUTIVE_ONES);
    printf("  Max byte repeats: %d\n", MAX_BYTE_REPEATS);

    // Launch search
    uint64_t total_threads = (uint64_t)num_blocks * (uint64_t)threads_per_block;
    printf("Grid: %d blocks x %d threads (total %" PRIu64 ")\n", num_blocks, threads_per_block, total_threads);
        printf("Total keys: %" PRIu64 ", chunk-size: %" PRIu64 ", batch-steps: %d, mode: %s, psi: %s\n",
            total_keys, chunk_size, batch_steps, use_filter ? "filter" : "nofilter", use_psi ? "on" : "off");

    launch_filtered_search(
        target_hash160,
        start_key,
        total_keys,
        num_blocks,
        threads_per_block,
        use_filter,
        use_psi,
        chunk_size,
        batch_steps
    );

    return 0;
}
