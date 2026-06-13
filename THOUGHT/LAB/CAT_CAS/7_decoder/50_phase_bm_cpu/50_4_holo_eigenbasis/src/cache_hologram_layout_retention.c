#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#define LINES 64
#define LINE_BYTES 64
#define LINE_STRIDE 4096
#define MODES 4
#define FAMILIES 3
#define LAYOUTS 2
#define DELAYS 4
#define TOUCHES 12
#define TRIALS 256
#define REPS 72

typedef struct {
    uint8_t *bytes;
} tape_t;

static volatile uint64_t sink = 0;
static const int delay_pauses[DELAYS] = {0, 512, 4096, 32768};

static const char *mode_name(int mode) {
    switch (mode & 3) {
        case 0: return "basis";
        case 1: return "rotation";
        case 2: return "residual";
        default: return "mini";
    }
}

static const char *family_name(int family) {
    switch (family) {
        case 0: return "real";
        case 1: return "pseudo";
        default: return "wrong";
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

static int map_line(int layout, int canonical) {
    if (layout == 0) return canonical & (LINES - 1);
    return ((canonical * 13) + 7) & (LINES - 1);
}

static uint8_t *line_ptr(tape_t *t, int physical_line) {
    return t->bytes + ((size_t)(physical_line & (LINES - 1)) * LINE_STRIDE);
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

static void touch_line(tape_t *t, int physical_line, int reps) {
    volatile uint64_t *p = (volatile uint64_t *)line_ptr(t, physical_line);
    for (int r = 0; r < reps; r++) {
        sink ^= p[0] + (uint64_t)physical_line + (uint64_t)r;
    }
}

static void reversible_xor_line(tape_t *t, int physical_line, uint64_t mask) {
    uint64_t *p = (uint64_t *)line_ptr(t, physical_line);
    p[0] ^= mask;
    p[1] ^= (mask << 9) | (mask >> 55);
    p[1] ^= (mask << 9) | (mask >> 55);
    p[0] ^= mask;
}

static void passive_delay(int pauses) {
    for (int i = 0; i < pauses; i++) {
        _mm_pause();
    }
}

static void real_canonical_lines(int mode, int out[TOUCHES]) {
    static const int basis[TOUCHES] = {9, 10, 11, 12, 13, 14, 9, 10, 11, 12, 13, 14};
    static const int rotation[TOUCHES] = {16, 17, 18, 19, 20, 21, 22, 23, 16, 18, 20, 22};
    static const int residual[TOUCHES] = {24, 25, 26, 27, 24, 25, 26, 27, 24, 25, 26, 27};
    static const int mini[TOUCHES] = {9, 16, 24, 10, 17, 25, 11, 18, 26, 12, 19, 27};
    const int *src = basis;
    if (mode == 1) src = rotation;
    else if (mode == 2) src = residual;
    else if (mode == 3) src = mini;
    for (int i = 0; i < TOUCHES; i++) out[i] = src[i];
}

static void pseudo_canonical_lines(int mode, int out[TOUCHES]) {
    static const int basis[TOUCHES] = {33, 34, 35, 36, 37, 38, 33, 34, 35, 36, 37, 38};
    static const int rotation[TOUCHES] = {40, 41, 42, 43, 44, 45, 46, 47, 40, 42, 44, 46};
    static const int residual[TOUCHES] = {52, 53, 54, 55, 52, 53, 54, 55, 52, 53, 54, 55};
    static const int mini[TOUCHES] = {33, 40, 52, 34, 41, 53, 35, 42, 54, 36, 43, 55};
    const int *src = basis;
    if (mode == 1) src = rotation;
    else if (mode == 2) src = residual;
    else if (mode == 3) src = mini;
    for (int i = 0; i < TOUCHES; i++) out[i] = src[i];
}

static void schedule_lines(int family, int declared_mode, int *actual_mode, int out[TOUCHES]) {
    if (family == 0) {
        *actual_mode = declared_mode;
        real_canonical_lines(declared_mode, out);
    } else if (family == 1) {
        *actual_mode = -1;
        pseudo_canonical_lines(declared_mode, out);
    } else {
        *actual_mode = (declared_mode + 1) & 3;
        real_canonical_lines(*actual_mode, out);
    }
}

static void apply_schedule(tape_t *t, int layout, int family, int declared_mode, int actual_mode,
                           int trial, int delay_class, const int canonical_lines[TOUCHES]) {
    uint64_t mask = 0x4C4159524554484FULL ^ (uint64_t)layout ^ ((uint64_t)family << 5) ^
                    ((uint64_t)declared_mode << 11) ^ ((uint64_t)(actual_mode + 3) << 19) ^
                    ((uint64_t)trial << 27) ^ ((uint64_t)delay_class << 43);
    for (int pass = 0; pass < 6; pass++) {
        for (int i = 0; i < TOUCHES; i++) {
            int idx = (i * 5 + pass * 7) % TOUCHES;
            int physical = map_line(layout, canonical_lines[idx]);
            touch_line(t, physical, REPS);
            reversible_xor_line(t, physical, mask ^ (uint64_t)(i * 0x9E3779B1U) ^ (uint64_t)pass);
        }
    }
}

static uint64_t measure_line(tape_t *t, int physical_line) {
    volatile uint64_t *p = (volatile uint64_t *)line_ptr(t, physical_line);
    uint64_t a = rdtscp_now();
    sink ^= p[0];
    uint64_t b = rdtscp_now();
    return b - a;
}

int main(void) {
    tape_t t;
    if (posix_memalign((void **)&t.bytes, 4096, (size_t)LINES * LINE_STRIDE) != 0) return 2;

    printf("delay_class,delay_pauses,layout,family,declared_mode,actual_mode,trial,hash_restored");
    for (int line = 0; line < LINES; line++) printf(",l%02d", line);
    printf("\n");

    int restored_total = 0;
    int total = 0;

    for (int delay_class = 0; delay_class < DELAYS; delay_class++) {
        for (int trial = 0; trial < TRIALS; trial++) {
            for (int layout = 0; layout < LAYOUTS; layout++) {
                for (int family = 0; family < FAMILIES; family++) {
                    for (int declared = 0; declared < MODES; declared++) {
                        int actual = declared;
                        int canonical_lines[TOUCHES];
                        schedule_lines(family, declared, &actual, canonical_lines);

                        init_tape(&t, 0x4404B000F0000000ULL ^ (uint64_t)trial ^
                                      ((uint64_t)layout << 52) ^ ((uint64_t)delay_class << 48));
                        uint64_t h0 = tape_hash(&t);
                        flush_tape(&t);

                        apply_schedule(&t, layout, family, declared, actual, trial, delay_class, canonical_lines);
                        passive_delay(delay_pauses[delay_class]);

                        double samples[LINES];
                        for (int probe = 0; probe < LINES; probe++) {
                            int physical = (probe * 29 + trial * 13 + layout * 11 + family * 7 +
                                           declared * 3 + delay_class * 5) & (LINES - 1);
                            samples[physical] = (double)measure_line(&t, physical);
                        }

                        uint64_t h1 = tape_hash(&t);
                        int restored = (h0 == h1);
                        restored_total += restored;
                        total++;

                        printf("%d,%d,%d,%s,%s,%s,%d,%d", delay_class, delay_pauses[delay_class],
                               layout, family_name(family), mode_name(declared),
                               actual >= 0 ? mode_name(actual) : "pseudo", trial, restored);
                        for (int line = 0; line < LINES; line++) printf(",%.3f", samples[line]);
                        printf("\n");
                    }
                }
            }
        }
    }

    fprintf(stderr, "PHASE4B_CACHE_HOLOGRAM_LAYOUT_RETENTION restored=%d/%d sink=%llu\n",
            restored_total, total, (unsigned long long)sink);
    free(t.bytes);
    return restored_total == total ? 0 : 1;
}
