/*
 * Host utilities for Ultimate Bitcoin Puzzle Solver
 * Handles CPU-side operations, file I/O, address parsing
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <openssl/sha.h>
#include <openssl/ripemd.h>

// Base58 decoding table
static const int8_t base58_decode_table[128] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,-1,-1,-1,-1,-1,-1,
    -1, 9,10,11,12,13,14,15,16,-1,17,18,19,20,21,-1,
    22,23,24,25,26,27,28,29,30,31,32,-1,-1,-1,-1,-1,
    -1,33,34,35,36,37,38,39,40,41,42,43,-1,44,45,46,
    47,48,49,50,51,52,53,54,55,56,57,-1,-1,-1,-1,-1
};

// Decode Bitcoin address to hash160
bool decode_address(const char* address, uint8_t hash160[20]) {
    // Base58 decode
    uint8_t decoded[25];
    memset(decoded, 0, 25);
    
    int len = strlen(address);
    for (int i = 0; i < len; i++) {
        int8_t val = base58_decode_table[(uint8_t)address[i]];
        if (val < 0) return false;
        
        int carry = val;
        for (int j = 24; j >= 0; j--) {
            carry += 58 * decoded[j];
            decoded[j] = carry % 256;
            carry /= 256;
        }
    }
    
    // Verify checksum
    uint8_t hash[32];
    SHA256_CTX sha;
    SHA256_Init(&sha);
    SHA256_Update(&sha, decoded, 21);
    SHA256_Final(hash, &sha);
    SHA256_Init(&sha);
    SHA256_Update(&sha, hash, 32);
    SHA256_Final(hash, &sha);
    
    if (memcmp(hash, decoded + 21, 4) != 0) {
        return false;
    }
    
    // Extract hash160 (skip version byte)
    memcpy(hash160, decoded + 1, 20);
    return true;
}

// Create bloom filter from address hash160
bool create_bloom_filter(const char* address, uint8_t* bloom, size_t bloom_size) {
    uint8_t hash160[20];
    if (!decode_address(address, hash160)) {
        std::cerr << "Failed to decode address: " << address << std::endl;
        return false;
    }
    
    // Initialize bloom filter to all zeros
    memset(bloom, 0, bloom_size);
    
    // Add hash160 to bloom filter with 7 hash functions
    for (int i = 0; i < 7; i++) {
        // Simple hash: rotate hash160 and XOR with index
        uint64_t hash_val = 0;
        for (int j = 0; j < 8; j++) {
            hash_val = (hash_val << 8) | hash160[(j + i * 2) % 20];
        }
        
        uint64_t bit_pos = hash_val % (bloom_size * 8);
        bloom[bit_pos / 8] |= (1 << (bit_pos % 8));
    }
    
    std::cout << "✅ Bloom filter created for address: " << address << std::endl;
    std::cout << "   Hash160: ";
    for (int i = 0; i < 20; i++) {
        printf("%02x", hash160[i]);
    }
    std::cout << std::endl;
    
    return true;
}

// Load multiple addresses into bloom filter
bool load_addresses(const char* filename, uint8_t* bloom, size_t bloom_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    memset(bloom, 0, bloom_size);
    
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        uint8_t hash160[20];
        if (decode_address(line.c_str(), hash160)) {
            // Add to bloom filter
            for (int i = 0; i < 7; i++) {
                uint64_t hash_val = 0;
                for (int j = 0; j < 8; j++) {
                    hash_val = (hash_val << 8) | hash160[(j + i * 2) % 20];
                }
                uint64_t bit_pos = hash_val % (bloom_size * 8);
                bloom[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
            count++;
        }
    }
    
    file.close();
    std::cout << "✅ Loaded " << count << " addresses into bloom filter" << std::endl;
    return count > 0;
}

// Convert private key to WIF format
std::string private_key_to_wif(const uint8_t priv[32], bool compressed = true) {
    uint8_t data[37];
    data[0] = 0x80;  // Mainnet prefix
    memcpy(data + 1, priv, 32);
    
    int len = 33;
    if (compressed) {
        data[33] = 0x01;
        len = 34;
    }
    
    // Add checksum
    uint8_t hash[32];
    SHA256_CTX sha;
    SHA256_Init(&sha);
    SHA256_Update(&sha, data, len);
    SHA256_Final(hash, &sha);
    SHA256_Init(&sha);
    SHA256_Update(&sha, hash, 32);
    SHA256_Final(hash, &sha);
    
    memcpy(data + len, hash, 4);
    len += 4;
    
    // Base58 encode
    static const char base58_chars[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::string result;
    
    // Count leading zeros
    int zeros = 0;
    while (zeros < len && data[zeros] == 0) zeros++;
    
    // Encode
    std::vector<uint8_t> temp(len * 138 / 100 + 1);
    int length = 0;
    for (int i = zeros; i < len; i++) {
        int carry = data[i];
        for (int j = 0; j < length; j++) {
            carry += 256 * temp[j];
            temp[j] = carry % 58;
            carry /= 58;
        }
        while (carry) {
            temp[length++] = carry % 58;
            carry /= 58;
        }
    }
    
    // Add leading 1s
    for (int i = 0; i < zeros; i++) result += '1';
    
    // Reverse and convert to base58
    for (int i = length - 1; i >= 0; i--) {
        result += base58_chars[temp[i]];
    }
    
    return result;
}

// Save found key to file
bool save_found_key(const uint8_t priv[32], const char* address, const char* output_file = "FOUND_KEYS.txt") {
    std::ofstream file(output_file, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file" << std::endl;
        return false;
    }
    
    // Convert to hex
    char hex[65];
    for (int i = 0; i < 32; i++) {
        sprintf(hex + i * 2, "%02x", priv[i]);
    }
    hex[64] = 0;
    
    std::string wif_compressed = private_key_to_wif(priv, true);
    std::string wif_uncompressed = private_key_to_wif(priv, false);
    
    file << "========================================" << std::endl;
    file << "FOUND KEY!" << std::endl;
    file << "Address: " << address << std::endl;
    file << "Private Key (hex): " << hex << std::endl;
    file << "Private Key (WIF compressed): " << wif_compressed << std::endl;
    file << "Private Key (WIF uncompressed): " << wif_uncompressed << std::endl;
    file << "========================================" << std::endl;
    file.close();
    
    return true;
}
