#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>
#define TAPE_SIZE 256
#define NUM_SEEDS 4

uint64_t lcg(uint64_t *state) {
    *state = (*state * 0x41C64E6D + 0x3039);
    *state = (*state >> 13) ^ *state;
    *state = (*state << 17) + *state;
    return *state;
}

void sha256_tape(unsigned char *t, unsigned char *h) { SHA256(t, TAPE_SIZE, h); }
int hmatch(unsigned char *a, unsigned char *b) { return memcmp(a, b, SHA256_DIGEST_LENGTH) == 0; }

void op_xor(unsigned char *t, int s, uint64_t v) { ((uint64_t *)(t + s * 8))[0] ^= v; }
void op_rotl(unsigned char *t, int s, int n) { uint64_t *p = (uint64_t *)(t + s * 8); n %= 64; *p = (*p << n) | (*p >> (64 - n)); }
void op_rotr(unsigned char *t, int s, int n) { uint64_t *p = (uint64_t *)(t + s * 8); n %= 64; *p = (*p >> n) | (*p << (64 - n)); }
void op_swap(unsigned char *t, int a, int b) { uint64_t *p = (uint64_t *)(t + a * 8), *q = (uint64_t *)(t + b * 8), x = *p; *p = *q; *q = x; }

void op_csum(unsigned char *t, int cs, int *ds, int n) {
    uint64_t c = 0;
    for (int i = 0; i < n; i++) c ^= ((uint64_t *)(t + ds[i] * 8))[0];
    ((uint64_t *)(t + cs * 8))[0] ^= c;
}

int main() {
    printf("=== PHASE 3.7: CATALYTIC OPERATOR LIBRARY ===\n\n");
    int all = 1, pass;
    unsigned char tape[TAPE_SIZE], ha[SHA256_DIGEST_LENGTH], hb[SHA256_DIGEST_LENGTH], hc[SHA256_DIGEST_LENGTH];

    printf("Operator 1: XOR_BIND\n"); pass = 0;
    for (int s = 0; s < NUM_SEEDS; s++) { uint64_t r = 0xDEAD + s * 0x100;
        for (int i = 0; i < TAPE_SIZE / 8; i++) ((uint64_t *)tape)[i] = lcg(&r);
        sha256_tape(tape, ha); op_xor(tape, 0, 0xCAFE); sha256_tape(tape, hb);
        op_xor(tape, 0, 0xCAFE); sha256_tape(tape, hc);
        if (!hmatch(ha, hb) && hmatch(ha, hc)) pass++; }
    printf("  XOR_BIND: %d/%d\n", pass, NUM_SEEDS); all &= (pass == NUM_SEEDS);

    printf("Operator 2: ROTATE_LEFT\n"); pass = 0;
    for (int s = 0; s < NUM_SEEDS; s++) { uint64_t r = 0xDEAD + s * 0x100;
        for (int i = 0; i < TAPE_SIZE / 8; i++) ((uint64_t *)tape)[i] = lcg(&r);
        sha256_tape(tape, ha); op_rotl(tape, 1, 13); sha256_tape(tape, hb);
        op_rotr(tape, 1, 13); sha256_tape(tape, hc);
        if (!hmatch(ha, hb) && hmatch(ha, hc)) pass++; }
    printf("  ROTATE_L: %d/%d\n", pass, NUM_SEEDS); all &= (pass == NUM_SEEDS);

    printf("Operator 3: ROTATE_RIGHT\n"); pass = 0;
    for (int s = 0; s < NUM_SEEDS; s++) { uint64_t r = 0xDEAD + s * 0x100;
        for (int i = 0; i < TAPE_SIZE / 8; i++) ((uint64_t *)tape)[i] = lcg(&r);
        sha256_tape(tape, ha); op_rotr(tape, 2, 7); sha256_tape(tape, hb);
        op_rotl(tape, 2, 7); sha256_tape(tape, hc);
        if (!hmatch(ha, hb) && hmatch(ha, hc)) pass++; }
    printf("  ROTATE_R: %d/%d\n", pass, NUM_SEEDS); all &= (pass == NUM_SEEDS);

    printf("Operator 4: PHASE_TAG\n"); pass = 0;
    for (int s = 0; s < NUM_SEEDS; s++) { uint64_t r = 0xDEAD + s * 0x100;
        for (int i = 0; i < TAPE_SIZE / 8; i++) ((uint64_t *)tape)[i] = lcg(&r);
        sha256_tape(tape, ha); op_xor(tape, 3, 0x1234567890ABCDEFULL); sha256_tape(tape, hb);
        op_xor(tape, 3, 0x1234567890ABCDEFULL); sha256_tape(tape, hc);
        if (!hmatch(ha, hb) && hmatch(ha, hc)) pass++; }
    printf("  PHASE_TAG: %d/%d\n", pass, NUM_SEEDS); all &= (pass == NUM_SEEDS);

    printf("Operator 5: SIGN_BIND\n"); pass = 0;
    for (int s = 0; s < NUM_SEEDS; s++) { uint64_t r = 0xDEAD + s * 0x100;
        for (int i = 0; i < TAPE_SIZE / 8; i++) ((uint64_t *)tape)[i] = lcg(&r);
        sha256_tape(tape, ha);
        uint64_t sig = 0xFEEDFACEC0FFEE00ULL ^ 0x00C0FFEE00000001ULL;
        op_xor(tape, 4, sig); sha256_tape(tape, hb);
        op_xor(tape, 4, sig); sha256_tape(tape, hc);
        if (!hmatch(ha, hb) && hmatch(ha, hc)) pass++; }
    printf("  SIGN_BIND: %d/%d\n", pass, NUM_SEEDS); all &= (pass == NUM_SEEDS);

    printf("Operator 6: PERMUTE_SLOTS\n"); pass = 0;
    for (int s = 0; s < NUM_SEEDS; s++) { uint64_t r = 0xDEAD + s * 0x100;
        for (int i = 0; i < TAPE_SIZE / 8; i++) ((uint64_t *)tape)[i] = lcg(&r);
        sha256_tape(tape, ha); op_swap(tape, 5, 6); sha256_tape(tape, hb);
        op_swap(tape, 5, 6); sha256_tape(tape, hc);
        if (!hmatch(ha, hb) && hmatch(ha, hc)) pass++; }
    printf("  PERMUTE: %d/%d\n", pass, NUM_SEEDS); all &= (pass == NUM_SEEDS);

    printf("Operator 7: CHECKSUM_BIND\n"); pass = 0;
    int ds[] = {0, 1, 2, 3};
    for (int s = 0; s < NUM_SEEDS; s++) { uint64_t r = 0xDEAD + s * 0x100;
        for (int i = 0; i < TAPE_SIZE / 8; i++) ((uint64_t *)tape)[i] = lcg(&r);
        sha256_tape(tape, ha); op_csum(tape, 7, ds, 4); sha256_tape(tape, hb);
        op_csum(tape, 7, ds, 4); sha256_tape(tape, hc);
        if (!hmatch(ha, hb) && hmatch(ha, hc)) pass++; }
    printf("  CHECKSUM: %d/%d\n", pass, NUM_SEEDS); all &= (pass == NUM_SEEDS);

    printf("\n=== VERDICT: %s ===\n", all ? "ALL OPERATORS PASS" : "FAILURES DETECTED");
    return all ? 0 : 1;
}
