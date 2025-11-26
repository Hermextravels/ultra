#!/usr/bin/env python3
import sys
import hashlib

ALPHABET = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
ALPHA_MAP = {ALPHABET[i]: i for i in range(len(ALPHABET))}

def b58decode(s: str) -> bytes:
    x = 0
    for c in s.encode():
        if c not in ALPHA_MAP:
            raise ValueError(f"invalid base58 char: {chr(c)}")
        x = x * 58 + ALPHA_MAP[c]
    # convert to bytes (big-endian)
    # determine leading zeros
    n_zeros = len(s) - len(s.lstrip('1'))
    full = x.to_bytes((x.bit_length() + 7) // 8, 'big')
    return b'\x00' * n_zeros + full

def b58check_decode(addr: str) -> bytes:
    data = b58decode(addr)
    if len(data) < 4:
        raise ValueError("too short")
    payload, checksum = data[:-4], data[-4:]
    h = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    if h != checksum:
        raise ValueError("checksum mismatch")
    return payload

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: addr_to_hash160.py <base58_address>")
        sys.exit(1)
    addr = sys.argv[1].strip()
    payload = b58check_decode(addr)
    if len(payload) < 21:
        print("Unexpected payload length")
        sys.exit(2)
    version = payload[0]
    h160 = payload[1:21]
    print(h160.hex())
