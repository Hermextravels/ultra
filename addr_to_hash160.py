import sys, base58, hashlib
addr = sys.argv[1]
payload = base58.b58decode(addr)
h160 = payload[1:-4]
print(h160.hex())
