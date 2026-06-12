#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#define LINES 64
#define LINE_BYTES 64
#define LINE_STRIDE 4096
#define TRIALS 512
#define REPS 64

typedef struct {
    uint8_t *bytes;
} tape_t;

static volatile uint64_t sink = 0;

static uint64_t fnv1a64(const void *data, size_t len) {
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

static uint64_t rdtscp_now(void) {
    unsigned aux;
    return __rdtscp(&aux);
}

static void init_tape(tape_t *t, uint64_t seed) {
    uint64_t rng = seed;
    for (int i = 0; i < LINES; i++) {
        uint8_t *line = t->bytes + ((size_t)i * LINE_STRIDE);
        for (int j = 0; j < LINE_BYTES; j += 8) {
            uint64_t v = lcg(&rng);
            memcpy(line + j, &v, sizeof(v));
        }
    }
}

static uint64_t tape_hash(tape_t *t) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < LINES; i++) {
        h ^= fnv1a64(t->bytes + ((size_t)i * LINE_STRIDE), LINE_BYTES);
        h *= 1099511628211ULL;
    }
    return h;
}

static void flush_tape(tape_t *t) {
    for (int i = 0; i < LINES; i++) {
        _mm_clflush(t->bytes + ((size_t)i * LINE_STRIDE));
    }
    _mm_mfence();
}

static void touch_line(tape_t *t, int line, int reps) {
    volatile uint64_t *p = (volatile uint64_t *)(t->bytes + ((size_t)(line & (LINES - 1)) * LINE_STRIDE));
    for (int r = 0; r < reps; r++) {
        sink ^= p[0] + (uint64_t)line + (uint64_t)r;
    }
}

static void reversible_xor_line(tape_t *t, int line, uint64_t mask) {
    uint64_t *p = (uint64_t *)(t->bytes + ((size_t)(line & (LINES - 1)) * LINE_STRIDE));
    p[0] ^= mask;
    p[1] ^= (mask << 7) | (mask >> 57);
    p[1] ^= (mask << 7) | (mask >> 57);
    p[0] ^= mask;
}

static int line_in_group(int mode, int line) {
    if (mode == 0) return line >= 9 && line <= 14;   /* shared basis */
    if (mode == 1) return line >= 16 && line <= 23;  /* rotation chain */
    if (mode == 2) return line >= 24 && line <= 27;  /* residual tags */
    if (mode == 3) return (line >= 9 && line <= 27); /* mini-model path */
    return ((line * 17 + 5) & 63) < 12;              /* matched random reversible */
}

static const char *mode_name(int mode) {
    switch (mode) {
        case 0: return "basis";
        case 1: return "rotation";
        case 2: return "residual";
        case 3: return "mini";
        default: return "random_reversible";
    }
}

static void apply_holo_schedule(tape_t *t, int mode, int trial) {
    uint64_t mask = 0x484F4C4F00000000ULL ^ (uint64_t)mode ^ ((uint64_t)trial << 8);
    for (int pass = 0; pass < 4; pass++) {
        for (int line = 0; line < LINES; line++) {
            if (!line_in_group(mode, line)) continue;
            touch_line(t, line, REPS);
            reversible_xor_line(t, line, mask ^ (uint64_t)(line * 0x9E3779B1U));
        }
    }
}

static uint64_t measure_line(tape_t *t, int line) {
    volatile uint64_t *p = (volatile uint64_t *)(t->bytes + ((size_t)(line & (LINES - 1)) * LINE_STRIDE));
    uint64_t a = rdtscp_now();
    sink ^= p[0];
    uint64_t b = rdtscp_now();
    return b - a;
}

int main(void) {
    tape_t t;
    if (posix_memalign((void **)&t.bytes, 4096, (size_t)LINES * LINE_STRIDE) != 0) {
        return 2;
    }

    printf("mode,trial,hash_restored,group_cycles,other_cycles,contrast_cycles\n");
    int total_restored = 0;
    int total = 0;

    for (int trial = 0; trial < TRIALS; trial++) {
        for (int mode = 0; mode < 5; mode++) {
            init_tape(&t, 0x4404000000000000ULL ^ (uint64_t)trial);
            uint64_t h0 = tape_hash(&t);
            flush_tape(&t);

            apply_holo_schedule(&t, mode, trial);

            uint64_t group_sum = 0, other_sum = 0;
            int group_n = 0, other_n = 0;
            for (int line = 0; line < LINES; line++) {
                int probe_line = (line * 37 + trial * 11) & (LINES - 1);
                uint64_t c = measure_line(&t, probe_line);
                if (line_in_group(mode, probe_line)) {
                    group_sum += c;
                    group_n++;
                } else {
                    other_sum += c;
                    other_n++;
                }
            }
            double group = group_n ? (double)group_sum / (double)group_n : 0.0;
            double other = other_n ? (double)other_sum / (double)other_n : 0.0;

            uint64_t h1 = tape_hash(&t);
            int restored = (h0 == h1);
            total_restored += restored;
            total++;

            printf("%s,%d,%d,%.3f,%.3f,%.3f\n",
                   mode_name(mode), trial, restored, group, other, other - group);
        }
    }

    fprintf(stderr, "PHASE4B_CACHE_HOLOGRAM_AFTERIMAGE restored=%d/%d sink=%llu\n",
            total_restored, total, (unsigned long long)sink);
    free(t.bytes);
    return total_restored == total ? 0 : 1;
}
