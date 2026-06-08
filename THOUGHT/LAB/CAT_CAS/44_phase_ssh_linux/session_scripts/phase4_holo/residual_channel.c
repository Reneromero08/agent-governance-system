#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAPE_WORDS 32
#define PROBLEM_WORDS 8
#define FAMILY_COUNT 3
#define SEEDS_PER_FAMILY 8

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

static int pop64(uint64_t x) {
    int n = 0;
    while (x) {
        x &= x - 1;
        n++;
    }
    return n;
}

static uint64_t tape_hash(const tape_t *t) {
    return fnv1a64_bytes(t->words, sizeof(t->words));
}

static void init_tape(tape_t *t, int family, int seed) {
    uint64_t rng = 0x4300000000000000ULL ^ ((uint64_t)family << 40) ^ (uint64_t)seed;
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < TAPE_WORDS; i++) {
        t->words[i] = lcg(&rng);
    }
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        uint64_t x = lcg(&rng);
        if (family == 0) {
            t->words[i] = x ^ (0x1111111111111111ULL * (uint64_t)(i + 1));
        } else if (family == 1) {
            t->words[i] = rotl64(x, (unsigned)(i * 5 + seed)) ^ 0xAAAAAAAAAAAAAAAAULL;
        } else {
            t->words[i] = (x ^ rotl64(x, 17)) + (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 2));
        }
    }

    t->words[8] = 0x484F4C4F52455349ULL;  /* HOLORESI */
    t->words[9] = 2;
    t->words[10] = 0x3F8000003F800000ULL;
    t->words[11] = 0xBF8000003F800000ULL;
    t->words[12] = 0x0000006400000064ULL;
    t->words[13] = 0x0000003200000032ULL;
    t->words[14] = t->words[9] ^ t->words[10] ^ t->words[11] ^ t->words[12] ^ t->words[13];
    t->words[15] = (uint64_t)family;
    for (int i = 16; i < TAPE_WORDS; i++) {
        t->words[i] = 0;
    }
}

static uint64_t relation_signature(const tape_t *t) {
    uint64_t sig = 0xA5A55A5AF00DFACEULL;
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        int j = (i + 1) & 7;
        int k = (i + 3) & 7;
        uint64_t edge = (t->words[i] ^ rotl64(t->words[j], (unsigned)(i + 5))) +
                        (t->words[k] ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1)));
        sig ^= rotl64(edge, (unsigned)(i * 9 + 3));
        sig *= 1099511628211ULL;
    }
    return sig;
}

static uint64_t parity_signature(const tape_t *t) {
    uint64_t sig = 0;
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        sig ^= (uint64_t)(pop64(t->words[i]) & 1) << i;
        sig ^= rotl64(t->words[i], (unsigned)(i + 1));
    }
    return sig;
}

static uint64_t walsh_signature(const tape_t *t) {
    int64_t v[8];
    for (int i = 0; i < 8; i++) {
        v[i] = (int64_t)(int16_t)(t->words[i] & 0xFFFF);
    }
    for (int step = 1; step < 8; step <<= 1) {
        for (int i = 0; i < 8; i += step << 1) {
            for (int j = 0; j < step; j++) {
                int64_t a = v[i + j];
                int64_t b = v[i + j + step];
                v[i + j] = a + b;
                v[i + j + step] = a - b;
            }
        }
    }
    uint64_t sig = 0;
    for (int i = 0; i < 8; i++) {
        sig ^= rotl64((uint64_t)(v[i] < 0 ? -v[i] : v[i]), (unsigned)(i * 8 + 1));
    }
    return sig;
}

static uint64_t graph_signature(const tape_t *t) {
    uint64_t sig = 0x123456789ABCDEF0ULL;
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        for (int j = i + 1; j < PROBLEM_WORDS; j++) {
            uint64_t e = (t->words[i] ^ t->words[j]) & 0xFFFFULL;
            sig ^= rotl64(e * (uint64_t)(i + 1) * (uint64_t)(j + 3), (unsigned)((i + j) & 31));
        }
    }
    return sig;
}

static int expected_answer(const tape_t *t) {
    return (int)((relation_signature(t) ^ rotl64(walsh_signature(t), 11) ^
                  rotl64(graph_signature(t), 23)) &
                 1ULL);
}

static void residual_tags(const tape_t *t, uint8_t tags[4]) {
    uint64_t rel = relation_signature(t);
    uint64_t par = parity_signature(t);
    uint64_t wal = walsh_signature(t);
    uint64_t gra = graph_signature(t);
    tags[0] = (uint8_t)(rel & 3ULL);
    tags[1] = (uint8_t)((par ^ rotl64(wal, 7)) & 3ULL);
    tags[2] = (uint8_t)((wal ^ rotl64(gra, 11)) & 3ULL);
    tags[3] = (uint8_t)((expected_answer(t) ^ tags[0] ^ tags[1] ^ tags[2]) & 1U);
}

static uint64_t pack_tags(const uint8_t tags[4]) {
    return (uint64_t)(tags[0] & 3U) | ((uint64_t)(tags[1] & 3U) << 2) |
           ((uint64_t)(tags[2] & 3U) << 4) | ((uint64_t)(tags[3] & 3U) << 6);
}

static int decode_from_tags(const uint8_t tags[4]) {
    return (int)((tags[0] ^ tags[1] ^ tags[2] ^ tags[3]) & 1U);
}

static void write_residual_channel(tape_t *t, const uint8_t tags[4], uint64_t undo[4]) {
    uint64_t packed = pack_tags(tags);
    undo[0] = packed ^ 0x5253494400000001ULL;
    undo[1] = ((uint64_t)tags[0] << 32) | ((uint64_t)tags[1] << 16) | tags[2];
    undo[2] = ((uint64_t)tags[3] << 48) | (relation_signature(t) & 0xFFFFFFFFULL);
    undo[3] = walsh_signature(t) ^ graph_signature(t);
    t->words[24] ^= undo[0];
    t->words[25] ^= undo[1];
    t->words[26] ^= undo[2];
    t->words[27] ^= undo[3];
}

static void clear_residual_channel(tape_t *t, const uint64_t undo[4]) {
    t->words[27] ^= undo[3];
    t->words[26] ^= undo[2];
    t->words[25] ^= undo[1];
    t->words[24] ^= undo[0];
}

static double residual_strength(const tape_t *base, const tape_t *candidate) {
    uint8_t tags[4];
    uint64_t undo[4];
    residual_tags(base, tags);
    tape_t tmp = *base;
    write_residual_channel(&tmp, tags, undo);
    int matches = 0;
    for (int i = 24; i <= 27; i++) {
        matches += candidate->words[i] == tmp.words[i];
    }
    return (double)matches / 4.0;
}

int main(void) {
    int pass = 0;
    int total = 0;
    int wrong_residual_rejected = 0;
    int random_residual_rejected = 0;
    int destructive_rejected = 0;

    printf("=== PHASE 4.3: RESIDUAL CHANNEL ===\n");
    printf("families=%d seeds_per_family=%d\n\n", FAMILY_COUNT, SEEDS_PER_FAMILY);

    for (int family = 0; family < FAMILY_COUNT; family++) {
        for (int seed = 0; seed < SEEDS_PER_FAMILY; seed++) {
            tape_t t0, t1, t2;
            uint8_t tags[4];
            uint64_t undo[4];
            init_tape(&t0, family, seed);
            uint64_t h0 = tape_hash(&t0);
            residual_tags(&t0, tags);
            int expected = expected_answer(&t0);
            int decoded = decode_from_tags(tags);

            t1 = t0;
            write_residual_channel(&t1, tags, undo);
            double strength = residual_strength(&t0, &t1);

            t2 = t1;
            clear_residual_channel(&t2, undo);
            int restored = tape_hash(&t2) == h0;

            uint8_t wrong[4] = {tags[0], tags[1], tags[2], (uint8_t)(tags[3] ^ 1U)};
            tape_t wrong_t = t0;
            uint64_t wrong_undo[4];
            write_residual_channel(&wrong_t, wrong, wrong_undo);
            int wrong_decoded = decode_from_tags(wrong);
            int wrong_reject = wrong_decoded != expected;

            uint8_t random_tags[4] = {(uint8_t)(seed & 3), (uint8_t)((seed + 1) & 3),
                                      (uint8_t)((seed + 2) & 3), (uint8_t)((seed + 3) & 3)};
            tape_t random_t = t0;
            uint64_t random_undo[4];
            write_residual_channel(&random_t, random_tags, random_undo);
            int random_reject = residual_strength(&t0, &random_t) < 1.0;

            tape_t destructive = t0;
            for (int i = 24; i <= 27; i++) {
                destructive.words[i] = 0xDEADBEEFCAFEBABEULL;
            }
            int destructive_reject = residual_strength(&t0, &destructive) < 1.0;

            int ok = restored && decoded == expected && strength == 1.0 && wrong_reject &&
                     random_reject && destructive_reject;
            pass += ok;
            wrong_residual_rejected += wrong_reject;
            random_residual_rejected += random_reject;
            destructive_rejected += destructive_reject;
            total++;

            printf("case F%d_S%02d restored=%d decoded=%d expected=%d strength=%.3f wrong_reject=%d random_reject=%d destructive_reject=%d ok=%d\n",
                   family, seed, restored, decoded, expected, strength, wrong_reject,
                   random_reject, destructive_reject, ok);
        }
    }

    printf("\nSummary:\n");
    printf("  pass: %d/%d\n", pass, total);
    printf("  wrong residual rejected: %d/%d\n", wrong_residual_rejected, total);
    printf("  random residual rejected: %d/%d\n", random_residual_rejected, total);
    printf("  destructive residual rejected: %d/%d\n", destructive_rejected, total);

    if (pass == total) {
        printf("=== VERDICT: PHASE4_3_RESIDUAL_CHANNEL_PASS ===\n");
        return 0;
    }
    printf("=== VERDICT: PHASE4_3_RESIDUAL_CHANNEL_FAIL ===\n");
    return 1;
}
