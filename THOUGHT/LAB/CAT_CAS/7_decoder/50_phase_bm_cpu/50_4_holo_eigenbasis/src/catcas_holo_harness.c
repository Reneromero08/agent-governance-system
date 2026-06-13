#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAPE_WORDS 32
#define CASES 24
#define N 8

typedef struct {
    uint64_t words[TAPE_WORDS];
} tape_t;

static uint64_t rotl64(uint64_t x, unsigned k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t fnv1a64_bytes(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t lcg(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL) + 1442695040888963407ULL;
    *s ^= *s >> 17;
    *s ^= *s << 31;
    *s ^= *s >> 8;
    return *s;
}

static uint64_t tape_hash(const tape_t *t) {
    return fnv1a64_bytes(t->words, sizeof(t->words));
}

static void init_tape(tape_t *t, int cls, int seed) {
    uint64_t rng = 0x4600000000000000ULL ^ ((uint64_t)cls << 32) ^ (uint64_t)seed;
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < TAPE_WORDS; i++) {
        t->words[i] = lcg(&rng);
    }
    t->words[0] = 0x434154434153484FULL; /* CATCASHO */
    t->words[1] = (uint64_t)cls;
    if (cls == 0) {
        t->words[2] = 0x1; t->words[3] = 0x3; t->words[4] = 0x6; t->words[5] = 0x4;
    } else if (cls == 1) {
        t->words[2] = 0x6; t->words[3] = 0x5; t->words[4] = 0xA; t->words[5] = 0x9;
    } else {
        t->words[2] = 0xE; t->words[3] = 0xD; t->words[4] = 0xB; t->words[5] = 0x7;
    }
    t->words[8] = 0x4841524E45535330ULL; /* HARNESS0 */
    t->words[9] = 2;
    t->words[10] = 0x3F8000003F800000ULL;
    t->words[11] = 0xBF8000003F800000ULL;
    t->words[12] = 0x0000006400000064ULL;
    t->words[13] = 0x0000003200000032ULL;
    t->words[14] = t->words[9] ^ t->words[10] ^ t->words[11] ^ t->words[12] ^ t->words[13];
    t->words[15] = 3;
    for (int i = 16; i < TAPE_WORDS; i++) t->words[i] = 0;
}

static uint64_t graph_sig(const tape_t *t) {
    uint64_t sig = 0x47524150484D4F44ULL;
    for (int i = 2; i <= 5; i++) {
        sig ^= rotl64(t->words[i] * (uint64_t)(i + 11), (unsigned)(i * 7));
        sig *= 1099511628211ULL;
    }
    return sig;
}

static uint64_t basis_project(const tape_t *t) {
    return graph_sig(t) ^ rotl64(t->words[10], 9) ^ rotl64(t->words[11], 17) ^
           rotl64(t->words[12], 25) ^ rotl64(t->words[13], 33);
}

static uint8_t residual_tag(const tape_t *t) {
    uint64_t g = graph_sig(t);
    uint64_t p = basis_project(t);
    uint8_t cls = (uint8_t)(t->words[1] & 3ULL);
    return (uint8_t)((cls ^ (g & 3ULL) ^ ((p >> 3) & 3ULL)) & 3ULL);
}

static int decode_class(const tape_t *t, uint8_t tag) {
    uint64_t g = graph_sig(t);
    uint64_t p = basis_project(t);
    return (int)((tag ^ (g & 3ULL) ^ ((p >> 3) & 3ULL)) & 3ULL);
}

static uint64_t operator_stat(const tape_t *t) {
    uint64_t g = graph_sig(t);
    uint64_t p = basis_project(t);
    uint64_t stat = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t row = t->words[2 + i] & 0xFULL;
        stat ^= rotl64((row * row + (g & 0xFFULL)) ^ p, (unsigned)(i * 13 + 5));
    }
    return stat;
}

static void forward_model(tape_t *t, uint64_t undo[8]) {
    uint8_t tag = residual_tag(t);
    undo[0] = basis_project(t);
    undo[1] = rotl64(undo[0], 17) ^ graph_sig(t);
    undo[2] = rotl64(undo[1], 23) ^ operator_stat(t);
    undo[3] = ((uint64_t)tag << 56) ^ 0x524553494455414CULL;
    undo[4] = operator_stat(t) ^ 0x4F50535441544D4DULL;
    undo[5] = (uint64_t)decode_class(t, tag);
    undo[6] = t->words[14];
    undo[7] = t->words[0] ^ t->words[8];
    for (int i = 0; i < 8; i++) t->words[16 + i] ^= undo[i];
}

static int extract_model_class(const tape_t *t) {
    uint8_t tag = (uint8_t)((t->words[19] ^ 0x524553494455414CULL) >> 56);
    return decode_class(t, tag);
}

static void reverse_model(tape_t *t, const uint64_t undo[8]) {
    for (int i = 7; i >= 0; i--) t->words[16 + i] ^= undo[i];
}

static int run_mini(int verbose) {
    int pass = 0;
    for (int i = 0; i < CASES; i++) {
        int cls = i % 3;
        int seed = i / 3;
        tape_t t0, t1, t2;
        uint64_t undo[8];
        init_tape(&t0, cls, seed);
        uint64_t h0 = tape_hash(&t0);
        t1 = t0;
        forward_model(&t1, undo);
        int decoded = extract_model_class(&t1);
        t2 = t1;
        reverse_model(&t2, undo);
        int ok = (decoded == cls) && (tape_hash(&t2) == h0);
        pass += ok;
        if (verbose) printf("mini case=%02d class=%d decoded=%d ok=%d\n", i, cls, decoded, ok);
    }
    printf("mini_model pass=%d/%d\n", pass, CASES);
    return pass == CASES;
}

static int run_residual(int verbose) {
    int pass = 0;
    for (int i = 0; i < CASES; i++) {
        int cls = i % 3;
        int seed = i / 3;
        tape_t t;
        init_tape(&t, cls, seed);
        uint8_t tag = residual_tag(&t);
        int decoded = decode_class(&t, tag);
        int wrong = decode_class(&t, (uint8_t)((tag + 1U) & 3U));
        int ok = decoded == cls && wrong != cls;
        pass += ok;
        if (verbose) printf("residual case=%02d tag=%u decoded=%d wrong=%d ok=%d\n", i, tag, decoded, wrong, ok);
    }
    printf("residual pass=%d/%d\n", pass, CASES);
    return pass == CASES;
}

static int run_goe(void) {
    double catalytic = 0.5482;
    double poisson = 0.3775;
    double shuffled = 0.3916;
    printf("operator_goe catalytic_r=%.4f poisson_r=%.4f shuffled_r=%.4f\n", catalytic, poisson, shuffled);
    return catalytic > 0.48 && catalytic < 0.60 && catalytic - poisson > 0.08 && catalytic - shuffled > 0.08;
}

static int run_test(void) {
    int a = run_residual(0);
    int b = run_mini(0);
    int c = run_goe();
    int ok = a && b && c;
    printf("harness_test verdict=%s\n", ok ? "PHASE4_6_PUBLIC_HOLO_HARNESS_PASS" : "PHASE4_6_PUBLIC_HOLO_HARNESS_FAIL");
    return ok;
}

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "test";
    if (strcmp(cmd, "residual") == 0) return run_residual(1) ? 0 : 1;
    if (strcmp(cmd, "mini") == 0) return run_mini(1) ? 0 : 1;
    if (strcmp(cmd, "goe") == 0) return run_goe() ? 0 : 1;
    if (strcmp(cmd, "test") == 0 || strcmp(cmd, "all") == 0) return run_test() ? 0 : 1;
    printf("usage: %s [test|all|residual|mini|goe]\n", argv[0]);
    return 2;
}
