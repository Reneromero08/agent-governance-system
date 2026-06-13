#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#define LINES 64
#define LINE_BYTES 64
#define LINE_STRIDE 4096
#define MODES 5
#define TOUCHES 12
#define TRIALS 768
#define REPS 72

typedef struct {
    uint8_t *bytes;
} tape_t;

static volatile uint64_t sink = 0;

static const char *mode_name(int mode) {
    switch (mode) {
        case 0: return "basis";
        case 1: return "rotation";
        case 2: return "residual";
        case 3: return "mini";
        default: return "random_reversible";
    }
}

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

static uint8_t *line_ptr(tape_t *t, int line) {
    return t->bytes + ((size_t)(line & (LINES - 1)) * LINE_STRIDE);
}

static void init_tape(tape_t *t, uint64_t seed) {
    uint64_t rng = seed;
    for (int i = 0; i < LINES; i++) {
        uint8_t *line = line_ptr(t, i);
        for (int j = 0; j < LINE_BYTES; j += 8) {
            uint64_t v = lcg(&rng);
            memcpy(line + j, &v, sizeof(v));
        }
    }
}

static uint64_t tape_hash(tape_t *t) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < LINES; i++) {
        h ^= fnv1a64(line_ptr(t, i), LINE_BYTES);
        h *= 1099511628211ULL;
    }
    return h;
}

static void flush_tape(tape_t *t) {
    for (int i = 0; i < LINES; i++) {
        _mm_clflush(line_ptr(t, i));
    }
    _mm_mfence();
}

static void touch_line(tape_t *t, int line, int reps) {
    volatile uint64_t *p = (volatile uint64_t *)line_ptr(t, line);
    for (int r = 0; r < reps; r++) {
        sink ^= p[0] + (uint64_t)line + (uint64_t)r;
    }
}

static void reversible_xor_line(tape_t *t, int line, uint64_t mask) {
    uint64_t *p = (uint64_t *)line_ptr(t, line);
    p[0] ^= mask;
    p[1] ^= (mask << 9) | (mask >> 55);
    p[1] ^= (mask << 9) | (mask >> 55);
    p[0] ^= mask;
}

static void mode_lines(int mode, int trial, int out[TOUCHES]) {
    static const int basis[TOUCHES] = {9, 10, 11, 12, 13, 14, 9, 10, 11, 12, 13, 14};
    static const int rotation[TOUCHES] = {16, 17, 18, 19, 20, 21, 22, 23, 16, 18, 20, 22};
    static const int residual[TOUCHES] = {24, 25, 26, 27, 24, 25, 26, 27, 24, 25, 26, 27};
    static const int mini[TOUCHES] = {9, 16, 24, 10, 17, 25, 11, 18, 26, 12, 19, 27};
    const int *src = basis;
    if (mode == 1) src = rotation;
    else if (mode == 2) src = residual;
    else if (mode == 3) src = mini;

    if (mode < 4) {
        for (int i = 0; i < TOUCHES; i++) out[i] = src[i];
        return;
    }

    uint64_t rng = 0x52414E44484F4C4FULL ^ (uint64_t)trial;
    for (int i = 0; i < TOUCHES; i++) {
        out[i] = (int)(lcg(&rng) & (LINES - 1));
    }
}

static void apply_schedule(tape_t *t, int mode, int trial, const int lines[TOUCHES]) {
    uint64_t mask = 0x484F4C4F4D4F4445ULL ^ (uint64_t)mode ^ ((uint64_t)trial << 11);
    for (int pass = 0; pass < 6; pass++) {
        for (int i = 0; i < TOUCHES; i++) {
            int idx = (i * 5 + pass * 7) % TOUCHES;
            int line = lines[idx];
            touch_line(t, line, REPS);
            reversible_xor_line(t, line, mask ^ (uint64_t)(i * 0x9E3779B1U) ^ (uint64_t)pass);
        }
    }
}

static uint64_t measure_line(tape_t *t, int line) {
    volatile uint64_t *p = (volatile uint64_t *)line_ptr(t, line);
    uint64_t a = rdtscp_now();
    sink ^= p[0];
    uint64_t b = rdtscp_now();
    return b - a;
}

static int contains_line(const int lines[TOUCHES], int line) {
    for (int i = 0; i < TOUCHES; i++) {
        if (lines[i] == line) return 1;
    }
    return 0;
}

int main(void) {
    tape_t t;
    if (posix_memalign((void **)&t.bytes, 4096, (size_t)LINES * LINE_STRIDE) != 0) return 2;

    printf("mode,trial,hash_restored");
    for (int line = 0; line < LINES; line++) printf(",l%02d", line);
    printf(",target_mean,other_mean,contrast\n");

    int restored_total = 0;
    int total = 0;

    for (int trial = 0; trial < TRIALS; trial++) {
        for (int mode = 0; mode < MODES; mode++) {
            int lines[TOUCHES];
            mode_lines(mode, trial, lines);
            init_tape(&t, 0x4404B00000000000ULL ^ (uint64_t)trial);
            uint64_t h0 = tape_hash(&t);
            flush_tape(&t);

            apply_schedule(&t, mode, trial, lines);

            double samples[LINES];
            for (int probe = 0; probe < LINES; probe++) {
                int line = (probe * 29 + trial * 13) & (LINES - 1);
                samples[line] = (double)measure_line(&t, line);
            }

            double target_sum = 0.0, other_sum = 0.0;
            int target_n = 0, other_n = 0;
            for (int line = 0; line < LINES; line++) {
                if (contains_line(lines, line)) {
                    target_sum += samples[line];
                    target_n++;
                } else {
                    other_sum += samples[line];
                    other_n++;
                }
            }
            double target_mean = target_n ? target_sum / target_n : 0.0;
            double other_mean = other_n ? other_sum / other_n : 0.0;

            uint64_t h1 = tape_hash(&t);
            int restored = (h0 == h1);
            restored_total += restored;
            total++;

            printf("%s,%d,%d", mode_name(mode), trial, restored);
            for (int line = 0; line < LINES; line++) printf(",%.3f", samples[line]);
            printf(",%.3f,%.3f,%.3f\n", target_mean, other_mean, other_mean - target_mean);
        }
    }

    fprintf(stderr, "PHASE4B_CACHE_HOLOGRAM_MODE_CLASSIFIER restored=%d/%d sink=%llu\n",
            restored_total, total, (unsigned long long)sink);
    free(t.bytes);
    return restored_total == total ? 0 : 1;
}
