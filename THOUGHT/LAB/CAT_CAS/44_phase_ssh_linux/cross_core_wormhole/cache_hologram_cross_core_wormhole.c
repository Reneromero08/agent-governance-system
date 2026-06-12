/*
 * cache_hologram_cross_core_wormhole.c
 * ====================================
 * CROSS-CORE WORMHOLE physical harness (Stage 2). Drop-in extension of
 *   session_scripts/phase4_holo/cache_hologram_cross_core.c
 * that adds the two pieces Codex's failing protocol is missing, as diagnosed
 * by the traversable-wormhole framework:
 *
 *   (1) OPENING COUPLING  - a coordinated cross-core access window. The writer
 *       holds the .holo line family resident in the shared LLC (a sustained
 *       refresh = the GJW negative-energy coupling that holds the throat open)
 *       WHILE the observer takes its first-access timings. This replaces the
 *       old "writer finishes, then observer probes cold/evicted lines" with a
 *       synchronized co-access (the eviction-set handshake).
 *
 *   (2) OBSERVER UNSCRAMBLE - the writer lays the message into the family
 *       through a reversible key-permutation (the SYK scramble of residency)
 *       plus a relational phase ramp (graded residency across the family). The
 *       observer probes in the INVERSE-permuted (coordinated) order and writes
 *       a DE-PERMUTED per-line vector, so the canonical-line analyzer sees the
 *       concentrated .holo signature instead of scrambled radiation.
 *
 * Everything else matches Codex's harness: writer core 0, observer core 1,
 * reversible XOR writes, byte-hash restore, real/pseudo/wrong matched-null
 * families, and the EXACT CSV schema
 *   family,declared_mode,actual_mode,trial,hash_restored,l00..l63
 * so analyze_cache_hologram_matched_nulls.py runs unchanged.
 *
 * Build (on the live Phenom, Linux):
 *   gcc -O2 cache_hologram_cross_core_wormhole.c -lpthread -lm -o cc_wormhole
 *   ./cc_wormhole > phase4b_cache_hologram_cross_core_wormhole.csv
 *   python3 analyze_cache_hologram_matched_nulls.py \
 *       phase4b_cache_hologram_cross_core_wormhole.csv summary.json
 *
 * Built-in A/B control: compile with -DWORMHOLE=0 to disable the opening
 * coupling + unscramble and reproduce Codex's failing raw cross-core read.
 *
 * NOTE: Sim-verified in cross_core_wormhole_sim.py (Stage 1). This is the
 * physical harness; the live run on the Phenom is Codex's job.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <x86intrin.h>

#ifndef WORMHOLE
#define WORMHOLE 1            /* 1 = opening coupling + unscramble; 0 = Codex naive read */
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LINES 64
#define LINE_BYTES 64
#define LINE_STRIDE 4096
#define MODES 4
#define FAMILIES 3
#define TOUCHES 12
#define TRIALS 320
#define REPS 96
#ifndef OPEN_REFRESH
#define OPEN_REFRESH 24       /* NEW: writer refresh budget per coupling poll (override -DOPEN_REFRESH=N) */
#endif
#ifndef RAMP_REPS
#define RAMP_REPS 40          /* NEW: relational phase-ramp amplitude (override -DRAMP_REPS=N) */
#endif
#ifndef PROBE_AVG
#define PROBE_AVG 1           /* NEW: observer averages PROBE_AVG first-access timings/line */
#endif
#define PHASE_LEVELS 8        /* NEW: number of distinguishable relational phase tags */
#ifndef WRITER_CORE
#define WRITER_CORE 0         /* override at compile time: -DWRITER_CORE=N */
#endif
#ifndef OBSERVER_CORE
#define OBSERVER_CORE 1       /* override at compile time: -DOBSERVER_CORE=N */
#endif
#ifndef SEED_WINDOW
#define SEED_WINDOW 0         /* independent realization index: -DSEED_WINDOW=K */
#endif

typedef struct { uint8_t *bytes; } tape_t;

typedef struct {
    tape_t tape;
    atomic_int state;         /* 0 idle, 1 coupling-window open, 2 observer done, 3 shutdown */
    int trial;
    int family;
    int declared;
    int actual;
    int perm[LINES];          /* NEW: shared reversible schedule permutation (canonical slot -> physical line) */
    int inv[LINES];           /* NEW: inverse permutation (physical line -> canonical slot) */
    int sched_phys[TOUCHES];  /* NEW: physical lines the writer refreshes during the coupling window */
    double samples[LINES];    /* de-permuted (canonical) timing vector emitted to the analyzer */
} shared_t;

static volatile uint64_t sink = 0;
static shared_t shared;

static const char *mode_name(int mode) {
    switch (mode & 3) { case 0: return "basis"; case 1: return "rotation";
                        case 2: return "residual"; default: return "mini"; }
}
static const char *family_name(int family) {
    switch (family) { case 0: return "real"; case 1: return "pseudo"; default: return "wrong"; }
}
static void pin_core(int core) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    sched_setaffinity(0, sizeof(set), &set);
}
static uint64_t fnv1a64(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
static uint64_t lcg(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL) + 1442695040888963407ULL;
    *s ^= *s >> 17; *s ^= *s << 31; *s ^= *s >> 8; return *s;
}
static uint64_t rdtscp_now(void) { unsigned aux; return __rdtscp(&aux); }
static uint8_t *line_ptr(tape_t *t, int line) {
    return t->bytes + ((size_t)(line & (LINES - 1)) * LINE_STRIDE);
}
static void init_tape(tape_t *t, uint64_t seed) {
    uint64_t rng = seed;
    for (int i = 0; i < LINES; i++) {
        uint8_t *line = line_ptr(t, i);
        for (int j = 0; j < LINE_BYTES; j += 8) { uint64_t v = lcg(&rng); memcpy(line + j, &v, sizeof(v)); }
    }
}
static uint64_t tape_hash(tape_t *t) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < LINES; i++) { h ^= fnv1a64(line_ptr(t, i), LINE_BYTES); h *= 1099511628211ULL; }
    return h;
}
static void flush_tape(tape_t *t) {
    for (int i = 0; i < LINES; i++) _mm_clflush(line_ptr(t, i));
    _mm_mfence();
}
static void touch_line(tape_t *t, int line, int reps) {
    volatile uint64_t *p = (volatile uint64_t *)line_ptr(t, line);
    for (int r = 0; r < reps; r++) sink ^= p[0] + (uint64_t)line + (uint64_t)r;
}
static void reversible_xor_line(tape_t *t, int line, uint64_t mask) {
    uint64_t *p = (uint64_t *)line_ptr(t, line);
    p[0] ^= mask; p[1] ^= (mask << 9) | (mask >> 55);
    p[1] ^= (mask << 9) | (mask >> 55); p[0] ^= mask;   /* self-inverse: bytes restore */
}

/* ---- canonical .holo line families (identical to Codex's real/pseudo maps) ---- */
static void real_mode_lines(int mode, int out[TOUCHES]) {
    static const int basis[TOUCHES]    = {9,10,11,12,13,14,9,10,11,12,13,14};
    static const int rotation[TOUCHES] = {16,17,18,19,20,21,22,23,16,18,20,22};
    static const int residual[TOUCHES] = {24,25,26,27,24,25,26,27,24,25,26,27};
    static const int mini[TOUCHES]     = {9,16,24,10,17,25,11,18,26,12,19,27};
    const int *src = basis;
    if (mode == 1) src = rotation; else if (mode == 2) src = residual; else if (mode == 3) src = mini;
    for (int i = 0; i < TOUCHES; i++) out[i] = src[i];
}
static void pseudo_mode_lines(int mode, int out[TOUCHES]) {
    static const int basis[TOUCHES]    = {33,34,35,36,37,38,33,34,35,36,37,38};
    static const int rotation[TOUCHES] = {40,41,42,43,44,45,46,47,40,42,44,46};
    static const int residual[TOUCHES] = {52,53,54,55,52,53,54,55,52,53,54,55};
    static const int mini[TOUCHES]     = {33,40,52,34,41,53,35,42,54,36,43,55};
    const int *src = basis;
    if (mode == 1) src = rotation; else if (mode == 2) src = residual; else if (mode == 3) src = mini;
    for (int i = 0; i < TOUCHES; i++) out[i] = src[i];
}
static void schedule_lines(int family, int declared_mode, int *actual_mode, int out[TOUCHES]) {
    if (family == 0) { *actual_mode = declared_mode; real_mode_lines(declared_mode, out); }
    else if (family == 1) { *actual_mode = -1; pseudo_mode_lines(declared_mode, out); }
    else { *actual_mode = (declared_mode + 1) & 3; real_mode_lines(*actual_mode, out); }
}

/* ---- NEW: reversible schedule permutation (the SYK scramble of residency) ---- */
static void build_perm(uint64_t key, int perm[LINES], int inv[LINES]) {
    for (int i = 0; i < LINES; i++) perm[i] = i;
    uint64_t s = key ? key : 0x9E3779B97F4A7C15ULL;
    for (int i = LINES - 1; i > 0; i--) {            /* Fisher-Yates */
        int j = (int)(lcg(&s) % (uint64_t)(i + 1));
        int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    for (int i = 0; i < LINES; i++) inv[perm[i]] = i;
}

/* relational phase tag: deterministic per (trial,declared), recoverable by ramp fit */
static int phase_tag(int trial, int declared) { return (trial * 3 + declared * 5) % PHASE_LEVELS; }

/* writer: lay the schedule into PHYSICAL lines via perm, with a relational phase ramp */
static void apply_schedule_wormhole(tape_t *t, int family, int declared_mode, int actual_mode,
                                    int trial, const int lines[TOUCHES], const int perm[LINES],
                                    int sched_phys_out[TOUCHES]) {
    uint64_t mask = 0x43524F5353434F52ULL ^ (uint64_t)family ^
                    ((uint64_t)declared_mode << 8) ^ ((uint64_t)(actual_mode + 3) << 17) ^
                    ((uint64_t)trial << 25);
    double theta = 2.0 * M_PI * (double)phase_tag(trial, declared_mode) / (double)PHASE_LEVELS;
    for (int i = 0; i < TOUCHES; i++) sched_phys_out[i] = perm[lines[i] & (LINES - 1)];
    for (int pass = 0; pass < 6; pass++) {
        for (int i = 0; i < TOUCHES; i++) {
            int idx = (i * 5 + pass * 7) % TOUCHES;          /* same scramble order as Codex */
            int phys = perm[lines[idx] & (LINES - 1)];        /* NEW: through the permutation */
            int ramp = (int)lround((double)RAMP_REPS * cos(theta + 2.0 * M_PI * (double)idx / (double)TOUCHES));
            int reps = REPS + ramp; if (reps < 1) reps = 1;   /* NEW: relational phase ramp */
            touch_line(t, phys, reps);
            reversible_xor_line(t, phys, mask ^ (uint64_t)(i * 0x9E3779B1U) ^ (uint64_t)pass);
        }
    }
}

/* writer: hold the throat open (refresh the schedule physical lines in the LLC) */
static void coupling_refresh(tape_t *t, const int sched_phys[TOUCHES]) {
    for (int i = 0; i < TOUCHES; i++) touch_line(t, sched_phys[i], OPEN_REFRESH);
}

static uint64_t measure_line(tape_t *t, int line) {
    volatile uint64_t *p = (volatile uint64_t *)line_ptr(t, line);
    uint64_t a = rdtscp_now(); sink ^= p[0]; uint64_t b = rdtscp_now();
    return b - a;
}

static void *observer_thread(void *arg) {
    (void)arg;
    pin_core(OBSERVER_CORE);
    for (;;) {
        int st = atomic_load_explicit(&shared.state, memory_order_acquire);
        if (st == 3) break;
        if (st != 1) { _mm_pause(); continue; }
#if WORMHOLE
        /* OPENING COUPLING: first-access timing inside the writer's sustained
         * refresh window, probed in the INVERSE-permuted (coordinated) order,
         * then written DE-PERMUTED so canonical-line centroids see the .holo. */
        for (int s = 0; s < LINES; s++) {
            int phys = shared.perm[s];                 /* canonical slot s lives on physical line phys */
            shared.samples[s] = (double)measure_line(&shared.tape, phys);
        }
#else
        /* NAIVE (Codex): raw per-line timing, fixed scan, no de-permutation. */
        int trial = shared.trial, family = shared.family, declared = shared.declared;
        for (int probe = 0; probe < LINES; probe++) {
            int line = (probe * 29 + trial * 13 + family * 7 + declared * 3) & (LINES - 1);
            shared.samples[line] = (double)measure_line(&shared.tape, line);
        }
#endif
        atomic_store_explicit(&shared.state, 2, memory_order_release);
    }
    return NULL;
}

int main(void) {
    if (posix_memalign((void **)&shared.tape.bytes, 4096, (size_t)LINES * LINE_STRIDE) != 0) return 2;
    atomic_init(&shared.state, 0);

    pthread_t observer;
    if (pthread_create(&observer, NULL, observer_thread, NULL) != 0) return 3;
    pin_core(WRITER_CORE);

    printf("family,declared_mode,actual_mode,trial,hash_restored");
    for (int line = 0; line < LINES; line++) printf(",l%02d", line);
    printf("\n");

    int restored_total = 0, total = 0;

    for (int trial = 0; trial < TRIALS; trial++) {
        for (int family = 0; family < FAMILIES; family++) {
            for (int declared = 0; declared < MODES; declared++) {
                int actual = declared, lines[TOUCHES];
                schedule_lines(family, declared, &actual, lines);

                /* shared schedule key -> reversible permutation known to both cores.
                 * SEED_WINDOW mixes in an independent realization offset so each
                 * window is a fresh permutation/tape draw (reproducibility test). */
                uint64_t sw = (uint64_t)(SEED_WINDOW)*0x100000001B3ULL;
                uint64_t key = 0xA5A5C0DEULL ^ ((uint64_t)trial << 20) ^
                               ((uint64_t)family << 12) ^ ((uint64_t)declared << 4) ^ sw;
                build_perm(key, shared.perm, shared.inv);

                init_tape(&shared.tape, 0x4404B000D0000000ULL ^ (uint64_t)trial ^ (sw << 8));
                uint64_t h0 = tape_hash(&shared.tape);
                flush_tape(&shared.tape);

                apply_schedule_wormhole(&shared.tape, family, declared, actual, trial,
                                        lines, shared.perm, shared.sched_phys);

                shared.trial = trial; shared.family = family;
                shared.declared = declared; shared.actual = actual;
                atomic_store_explicit(&shared.state, 1, memory_order_release);
#if WORMHOLE
                /* hold the throat open until the observer finishes its co-access read */
                while (atomic_load_explicit(&shared.state, memory_order_acquire) != 2) {
                    coupling_refresh(&shared.tape, shared.sched_phys);
                }
#else
                while (atomic_load_explicit(&shared.state, memory_order_acquire) != 2) _mm_pause();
#endif
                uint64_t h1 = tape_hash(&shared.tape);
                int restored = (h0 == h1);
                restored_total += restored; total++;

                printf("%s,%s,%s,%d,%d", family_name(family), mode_name(declared),
                       actual >= 0 ? mode_name(actual) : "pseudo", trial, restored);
                for (int line = 0; line < LINES; line++) printf(",%.3f", shared.samples[line]);
                printf("\n");

                atomic_store_explicit(&shared.state, 0, memory_order_release);
            }
        }
    }

    atomic_store_explicit(&shared.state, 3, memory_order_release);
    pthread_join(observer, NULL);

    fprintf(stderr,
            "PHASE4B_CACHE_HOLOGRAM_CROSS_CORE_WORMHOLE wormhole=%d seed_window=%d restored=%d/%d writer_core=%d observer_core=%d sink=%llu\n",
            WORMHOLE, (int)(SEED_WINDOW), restored_total, total, WRITER_CORE, OBSERVER_CORE, (unsigned long long)sink);
    free(shared.tape.bytes);
    return restored_total == total ? 0 : 1;
}
