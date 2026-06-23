#define _GNU_SOURCE
#include "captured_file.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#define SHA256_ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_transform(uint32_t state[8], const unsigned char block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = SHA256_ROTR(w[i - 15], 7) ^ SHA256_ROTR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = SHA256_ROTR(w[i - 2], 17) ^ SHA256_ROTR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3],
             e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = SHA256_ROTR(e, 6) ^ SHA256_ROTR(e, 11) ^ SHA256_ROTR(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + sha256_k[i] + w[i];
        uint32_t S0 = SHA256_ROTR(a, 2) ^ SHA256_ROTR(a, 13) ^ SHA256_ROTR(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

int hash_captured(const unsigned char *bytes, size_t size, char out_digest[CAPTURED_SHA256_LEN + 1]) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    uint64_t bit_len = (uint64_t)size * 8;
    size_t offset = 0;
    while (offset + 64 <= size) {
        sha256_transform(state, bytes + offset);
        offset += 64;
    }
    unsigned char last[64];
    size_t remainder = size - offset;
    memcpy(last, bytes + offset, remainder);
    last[remainder] = 0x80;
    memset(last + remainder + 1, 0, 64 - remainder - 1);
    if (remainder >= 56) {
        sha256_transform(state, last);
        memset(last, 0, 56);
    }
    for (int i = 0; i < 8; i++) {
        last[56 + i] = (unsigned char)(bit_len >> (56 - 8 * i));
    }
    sha256_transform(state, last);
    for (int i = 0; i < 8; i++) {
        snprintf(out_digest + i * 8, 9, "%08x", state[i]);
    }
    out_digest[CAPTURED_SHA256_LEN] = 0;
    return 0;
}

int capture_file(const char *path, CapturedFile *out, size_t max_size) {
    memset(out, 0, sizeof(*out));
    if (!path || !path[0]) return -1;
    int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
#ifdef O_BINARY
    flags |= O_BINARY;
#endif
    int fd = open(path, flags);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st)) { close(fd); return -1; }
    if (!S_ISREG(st.st_mode)) { close(fd); return -1; }
    if (st.st_size < 0 || (size_t)st.st_size > max_size) { close(fd); return -1; }
    size_t size = (size_t)st.st_size;
    unsigned char *bytes = malloc(size + 1);
    if (!bytes) { close(fd); return -1; }
    size_t total = 0;
    while (total < size) {
        ssize_t n = read(fd, bytes + total, size - total);
        if (n <= 0) { free(bytes); close(fd); return -1; }
        total += (size_t)n;
    }
    close(fd);
    bytes[size] = 0;
    if (hash_captured(bytes, size, out->sha256)) {
        free(bytes);
        return -1;
    }
    out->bytes = bytes;
    out->size = size;
    return 0;
}

void free_captured(CapturedFile *cf) {
    if (!cf) return;
    free(cf->bytes);
    memset(cf, 0, sizeof(*cf));
}
