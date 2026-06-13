#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAPE_WORDS 32
#define CASES 24

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

static void init_model(tape_t *t, int graph_class, int seed) {
    uint64_t rng = 0x4500000000000000ULL ^ ((uint64_t)graph_class << 32) ^ (uint64_t)seed;
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < TAPE_WORDS; i++) {
        t->words[i] = lcg(&rng);
    }

    t->words[0] = 0x484F4C4F4D494E49ULL; /* HOLOMINI */
    t->words[1] = (uint64_t)graph_class;

    /* Slots 2-5 encode a tiny 4-node graph adjacency row mask. */
    if (graph_class == 0) {
        t->words[2] = 0x1; t->words[3] = 0x3; t->words[4] = 0x6; t->words[5] = 0x4; /* path */
    } else if (graph_class == 1) {
        t->words[2] = 0x6; t->words[3] = 0x5; t->words[4] = 0xA; t->words[5] = 0x9; /* cycle */
    } else {
        t->words[2] = 0xE; t->words[3] = 0xD; t->words[4] = 0xB; t->words[5] = 0x7; /* clique */
    }

    t->words[6] = lcg(&rng);
    t->words[7] = lcg(&rng);
    t->words[8] = 0x4D494E494D4F444CULL; /* MINIMODL */
    t->words[9] = 2;
    t->words[10] = 0x3F8000003F800000ULL;
    t->words[11] = 0xBF8000003F800000ULL;
    t->words[12] = 0x0000006400000064ULL;
    t->words[13] = 0x0000003200000032ULL;
    t->words[14] = t->words[9] ^ t->words[10] ^ t->words[11] ^ t->words[12] ^ t->words[13];
    t->words[15] = 3; /* layer count */
    for (int i = 16; i < TAPE_WORDS; i++) {
        t->words[i] = 0;
    }
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
    return (graph_sig(t) ^ rotl64(t->words[10], 9) ^ rotl64(t->words[11], 17) ^
            rotl64(t->words[12], 25) ^ rotl64(t->words[13], 33));
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

    for (int i = 0; i < 8; i++) {
        t->words[16 + i] ^= undo[i];
    }
}

static int extract_model_class(const tape_t *t) {
    uint8_t tag = (uint8_t)((t->words[19] ^ 0x524553494455414CULL) >> 56);
    return decode_class(t, tag);
}

static void reverse_model(tape_t *t, const uint64_t undo[8]) {
    for (int i = 7; i >= 0; i--) {
        t->words[16 + i] ^= undo[i];
    }
}

int main(void) {
    int pass = 0;
    int wrong_reject = 0;
    int random_reject = 0;
    int restored_count = 0;

    printf("=== PHASE 4.5: .HOLO MINI MODEL ===\n\n");
    for (int i = 0; i < CASES; i++) {
        int cls = i % 3;
        int seed = i / 3;
        tape_t t0, t1, t2;
        uint64_t undo[8];
        init_model(&t0, cls, seed);
        uint64_t h0 = tape_hash(&t0);

        t1 = t0;
        forward_model(&t1, undo);
        int decoded = extract_model_class(&t1);

        uint8_t correct_tag = residual_tag(&t0);
        uint8_t wrong_tag = (uint8_t)((correct_tag + 1U) & 3U);
        int wrong_decoded = decode_class(&t0, wrong_tag);
        int wrong_ok = wrong_decoded != cls;

        uint8_t random_tag = (uint8_t)((seed * 3 + cls + 2) & 3);
        int random_decoded = decode_class(&t0, random_tag);
        int random_ok = random_decoded != cls || random_tag == correct_tag;

        t2 = t1;
        reverse_model(&t2, undo);
        int restored = tape_hash(&t2) == h0;
        int ok = restored && decoded == cls && wrong_ok && random_ok;

        pass += ok;
        restored_count += restored;
        wrong_reject += wrong_ok;
        random_reject += random_ok;

        printf("case %02d class=%d decoded=%d restored=%d wrong_reject=%d random_reject=%d ok=%d\n",
               i, cls, decoded, restored, wrong_ok, random_ok, ok);
    }

    printf("\nSummary:\n");
    printf("  pass: %d/%d\n", pass, CASES);
    printf("  restored: %d/%d\n", restored_count, CASES);
    printf("  wrong residual rejected: %d/%d\n", wrong_reject, CASES);
    printf("  random residual rejected: %d/%d\n", random_reject, CASES);

    if (pass == CASES) {
        printf("=== VERDICT: PHASE4_5_HOLO_MINI_MODEL_PASS ===\n");
        return 0;
    }
    printf("=== VERDICT: PHASE4_5_HOLO_MINI_MODEL_FAIL ===\n");
    return 1;
}
