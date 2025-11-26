#!/usr/bin/env python3
"""
CPU fallback solver for small ranges using Python (no CUDA).
- Supports compressed/uncompressed pubkeys
- Accepts start key (hex) and total keys (int/hex with K/M/G/T suffix)
- Chunked processing with progress output
- Smart filtering to avoid keys with many zeros/repeats (mirrors GPU filter intent)
NOTE: Intended for correctness checks and tiny windows (<= 2^24).
"""
import argparse, sys, time
from hashlib import sha256, new as hashlib_new
from ecdsa import SECP256k1, SigningKey

def parse_u64_with_suffix(s: str) -> int:
    s = s.strip().lower()
    mult = 1
    if s.endswith('k'): mult = 10**3; s = s[:-1]
    elif s.endswith('m'): mult = 10**6; s = s[:-1]
    elif s.endswith('g'): mult = 10**9; s = s[:-1]
    elif s.endswith('t'): mult = 10**12; s = s[:-1]
    return int(s, 0) * mult

def hamming_weight(x: int) -> int:
    return x.bit_count()

def _max_consecutive_bits(x: int, width: int = 256) -> tuple[int,int]:
    # returns (max_zeros, max_ones)
    max0 = max1 = c0 = c1 = 0
    for i in range(width):
        bit = (x >> i) & 1
        if bit:
            c1 += 1; max1 = max(max1, c1); c0 = 0
        else:
            c0 += 1; max0 = max(max0, c0); c1 = 0
    return max0, max1

def _max_byte_repeats(x: int) -> int:
    # count max repeats of same byte value across the 32 bytes
    b = x.to_bytes(32, 'big')
    maxrep = 1; cur = 1
    for i in range(1, 32):
        if b[i] == b[i-1]:
            cur += 1; maxrep = max(maxrep, cur)
        else:
            cur = 1
    return maxrep

def is_key_worth_searching(k: int, min_hw: int, max_hw: int,
                           max_consecutive_zeros: int = 24,
                           max_consecutive_ones: int = 24,
                           max_byte_repeats: int = 8) -> bool:
    hw = hamming_weight(k)
    if hw < min_hw or hw > max_hw:
        return False
    mz, mo = _max_consecutive_bits(k)
    if mz > max_consecutive_zeros or mo > max_consecutive_ones:
        return False
    if _max_byte_repeats(k) > max_byte_repeats:
        return False
    return True

def hash160_pubkey(priv_int: int, compressed: bool) -> bytes:
    priv_bytes = priv_int.to_bytes(32, 'big')
    sk = SigningKey.from_string(priv_bytes, curve=SECP256k1)
    vk = sk.verifying_key
    if compressed:
        x = vk.to_string()[:32]
        y = vk.to_string()[32:]
        prefix = b"\x02" if (y[-1] & 1) == 0 else b"\x03"
        pub = prefix + x
    else:
        pub = b"\x04" + vk.to_string()
    h1 = sha256(pub).digest()
    return hashlib_new('ripemd160', h1).digest()

def main():
    ap = argparse.ArgumentParser(description='CPU fallback filtered solver')
    ap.add_argument('--target-hash160', required=True, help='HEX40 target hash160')
    ap.add_argument('--start-key', required=True, help='HEX start key (supports 0x)')
    ap.add_argument('--total-keys', required=True, help='Count of keys to search (int/hex with K/M/G/T)')
    ap.add_argument('--pubkey-format', choices=['compressed','uncompressed'], default='compressed')
    ap.add_argument('--min-hw', type=int, default=20)
    ap.add_argument('--max-hw', type=int, default=236)
    ap.add_argument('--chunk-size', default='65536', help='Chunk size (int/hex with K/M/G/T). Default 65536')
    ap.add_argument('--progress-secs', type=float, default=2.0, help='Print progress every N seconds')
    args = ap.parse_args()

    target = bytes.fromhex(args.target_hash160)
    start = int(args.start_key, 0)
    total = parse_u64_with_suffix(args.total_keys)
    chunk = parse_u64_with_suffix(args.chunk_size)
    compressed = (args.pubkey_format == 'compressed')

    print(f"CPU solver: start=0x{start:x}, total={total}, chunk={chunk}, format={'compressed' if compressed else 'uncompressed'}")
    print(f"Target h160: {target.hex()}")

    filtered = 0
    searched = 0
    generated = 0
    t0 = time.time(); last = t0
    end = start + total
    i = start
    first_hash_printed = False

    while i < end:
        cend = min(i + chunk, end)
        while i < cend:
            k = i
            generated += 1
            if not is_key_worth_searching(k, args.min_hw, args.max_hw):
                filtered += 1; i += 1; continue
            searched += 1
            h = hash160_pubkey(k, compressed)
            if not first_hash_printed:
                print(f"First hash160: {h.hex()}")
                first_hash_printed = True
            if h == target:
                print("FOUND")
                print(f"Private key: 0x{k:064x}")
                print(f"Stats: generated={generated}, filtered={filtered}, searched={searched}")
                return 0
            i += 1
        now = time.time()
        if now - last >= args.progress_secs:
            elapsed = now - t0
            rate = (generated / 1e6) / elapsed if elapsed > 0 else 0.0
            pct = (i - start) * 100.0 / total
            print(f"Progress: {pct:.2f}% | Elapsed: {elapsed:.1f}s | Rate: {rate:.2f} MKeys/s | gen={generated} filt={filtered} search={searched}")
            last = now

    print("Not found in range.")
    return 1

if __name__ == '__main__':
    sys.exit(main())
