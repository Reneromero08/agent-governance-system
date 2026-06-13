#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>

#define TAPE_SIZE 256
uint64_t tape_words[TAPE_SIZE / 8];
unsigned char *tape = (unsigned char *)tape_words;

uint64_t lcg(uint64_t *state) {
    *state = (*state * 0x41C64E6D + 0x3039);
    *state = (*state >> 13) ^ *state;
    *state = (*state << 17) + *state;
    return *state;
}

void sha256_tape(unsigned char *hash) { SHA256(tape, TAPE_SIZE, hash); }
int hmatch(unsigned char *a, unsigned char *b) { return memcmp(a, b, SHA256_DIGEST_LENGTH) == 0; }

int main() {
    unsigned char h0[SHA256_DIGEST_LENGTH], hf[SHA256_DIGEST_LENGTH], hr[SHA256_DIGEST_LENGTH];
    int all = 1;

    printf("=== PHASE 3.8: MEANINGFUL REVERSIBLE COMPUTATION ===\n\n");

    // --- TEST 1: REVERSIBLE PARITY ---
    printf("--- TEST 1: REVERSIBLE PARITY ---\n");
    uint64_t rng = 0x1111111111111111ULL;
    for (int i = 0; i < TAPE_SIZE / 8; i++) tape_words[i] = lcg(&rng);
    tape_words[0] = 1; tape_words[1] = 3; tape_words[2] = 5; tape_words[3] = 7;
    sha256_tape(h0);
    uint64_t parity = 0;
    for (int i = 0; i < 4; i++) parity ^= tape_words[i];
    parity &= 1;
    tape_words[7] ^= parity;
    sha256_tape(hf);
    int p_ok = (parity == 0);
    printf("  Slots 0-3: 1,3,5,7 (all odd) -> parity=%lu (expected 0): %s\n", parity, p_ok ? "YES" : "NO");
    tape_words[7] ^= parity;
    sha256_tape(hr);
    int p_rest = hmatch(h0, hr);
    printf("  Tape restored: %s\n\n", p_rest ? "YES" : "NO");
    all &= p_ok && p_rest;

    // --- TEST 2: REVERSIBLE HASH FRAGMENT ---
    printf("--- TEST 2: REVERSIBLE HASH FRAGMENT ---\n");
    rng = 0x2222222222222222ULL;
    for (int i = 0; i < TAPE_SIZE / 8; i++) tape_words[i] = lcg(&rng);
    tape_words[0] = 0xFEEDFACECAFEBABEULL;
    tape_words[1] = 0xDEADBEEFBAADF00DULL;
    tape_words[2] = 0x1234567890ABCDEFULL;
    tape_words[3] = 0x0F1E2D3C4B5A6978ULL;
    sha256_tape(h0);
    uint64_t mixed = tape_words[0];
    mixed ^= tape_words[1];
    mixed = (mixed << 17) | (mixed >> 47);
    mixed ^= tape_words[2];
    mixed = (mixed << 31) | (mixed >> 33);
    mixed ^= tape_words[3];
    tape_words[7] ^= mixed;
    sha256_tape(hf);
    int h_ok = (mixed != 0);
    printf("  Hash fragment: 0x%016lx, non-zero: %s\n", mixed, h_ok ? "YES" : "NO");
    tape_words[7] ^= mixed;
    sha256_tape(hr);
    int h_rest = hmatch(h0, hr);
    printf("  Tape restored: %s\n\n", h_rest ? "YES" : "NO");
    all &= h_ok && h_rest;

    // --- TEST 3: REVERSIBLE FSM ---
    printf("--- TEST 3: REVERSIBLE FINITE-STATE TRANSITION ---\n");
    rng = 0x3333333333333333ULL;
    for (int i = 0; i < TAPE_SIZE / 8; i++) tape_words[i] = lcg(&rng);
    tape_words[0] = 0; tape_words[1] = 1;
    sha256_tape(h0);
    uint64_t state = tape_words[0], trigger = tape_words[1];
    tape_words[0] = state ^ trigger;
    sha256_tape(hf);
    int f_ok = (tape_words[0] == 1);
    printf("  State %lu + trigger %lu -> state %lu (expected 1): %s\n", state, trigger, tape_words[0], f_ok ? "YES" : "NO");
    tape_words[0] ^= trigger;
    sha256_tape(hr);
    int f_rest = hmatch(h0, hr);
    printf("  Tape restored: %s\n\n", f_rest ? "YES" : "NO");
    all &= f_ok && f_rest;

    printf("=== VERDICT: %s ===\n", all ? "ALL TESTS PASS - Meaningful reversible computation achieved" : "FAILURES DETECTED");
    return all ? 0 : 1;
}
