/*
 * cache_hologram_cross_core_conflict.c
 * ====================================
 * EXP44 SLOT 1 - CONFLICT-DISPLACEMENT SHARED-CACHE READOUT.
 *
 * Own-hardware EE/measurement research into inter-core physical coupling as a
 * computing medium, on the lab owner's AMD Phenom II X6 1090T (Debian 13).
 * Userspace only: no firmware, no voltage, no board modification.
 *
 * This is a drop-in sibling of
 *   session_scripts/phase4_holo/cache_hologram_cross_core.c
 * that REPLACES the occupancy/warmth carrier (which homogenized away under
 * co-access and failed the 0.60 mode gate) with a CONFLICT-DISPLACEMENT carrier
 * grounded in the MEASURED 5.10D effect on THIS box:
 *   same-address ~319 cyc/touch vs different-address ~293 vs none ~270,
 *   p=0.0005, family sign-agreement 1.0, swap-robust.
 *
 * INVERTED HOMOGENIZATION (the whole point of SLOT 1): co-access here does NOT
 * average a per-line warmth pattern away -- it CAUSES eviction of the RECEIVER's
 * OWN cache sets. The receiver pre-fills (primes) 64 eviction sets with its own
 * data, then times its own ring re-traversal per set (probe). The sender, pinned
 * to the other core, continuously hammers the CURRENT MODE's aggressor eviction
 * sets during the probe window. A targeted set => the receiver's own lines were
 * evicted => slower re-access (the +A displacement). Homogenization works in our
 * FAVOR: the eviction the sender induces IS the signal.
 *
 * CARRIER SLOTS:
 *   warmth (base)  : the 64 CSV slots are single cache LINES.
 *   conflict (this): the 64 CSV slots are 64 EVICTION SETS, each W>=48 lines in
 *                    one L3 congruence class (4096-byte stride within a set maps
 *                    the W lines to the same L3 set). The emitted per-slot value
 *                    is the receiver's median re-access latency for that set's
 *                    ring traversal.
 *
 * The reversible XOR-borrow + byte-hash restore on the MESSAGE TAPE is preserved
 * bit-exact (hash_restored must stay 1). The CSV schema is identical
 *   family,declared_mode,actual_mode,trial,hash_restored,l00..l63
 * so analyze_cache_hologram_matched_nulls.py runs UNCHANGED.
 *
 * A/B control:
 *   -DCONFLICT=1  (default) : conflict-displacement prime+probe read (the new angle).
 *   -DCONFLICT=0            : occupancy/warmth fallback (receiver times the SENDER's
 *                             warmed lines, no prime). MUST reproduce ~0.25-0.275.
 *
 * Build (on the live Phenom, Linux):
 *   gcc -O2 -pthread -DCONFLICT=1 -o conf cache_hologram_cross_core_conflict.c   (add -lm last if needed)
 *   ./conf > conf.csv
 *   python3 analyze_cache_hologram_matched_nulls.py conf.csv summary.json
 *
 * Compile-time knobs (sweep harness sets these):
 *   -DN_AVG=K        receiver averages K independent probe passes per drive point
 *   -DEVSET_W=W      eviction-set width (lines per L3 congruence class): 48/64/96
 *   -DWRITER_CORE=c  sender (aggressor) core
 *   -DOBSERVER_CORE=c receiver (prime+probe) core
 *   -DSEED_WINDOW=k  independent realization index (fresh tape/perm/RNG draw)
 *
 * CLAIM CAP: MODE traversal only. NOT phase/relational recovery (Slot 2), NOT a
 * lattice/crypto/quadrature claim.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#ifndef CONFLICT
#define CONFLICT 1            /* 1 = conflict-displacement prime+probe; 0 = warmth fallback */
#endif

#define LINES 64
#define LINE_BYTES 64
#define LINE_STRIDE 4096
#define MODES 4
#define FAMILIES 3
#define TOUCHES 12
#ifndef TRIALS
#define TRIALS 320            /* per-(family,mode) trial count; even/odd = train/test split */
#endif
#define REPS 96
#define ECHO_REPS 192

#ifndef EVSET_W
#define EVSET_W 48            /* lines per eviction set (L3 congruence class). Sweep: 48/64/96 */
#endif
#ifndef N_AVG
#define N_AVG 1               /* receiver averages N_AVG probe passes per drive point. Sweep: 1/4/16/64 */
#endif
#ifndef SET_STRIDE
#define SET_STRIDE 4096       /* within-set stride: maps the W lines to the SAME L3 set */
#endif
#ifndef HAMMER_REPS
#define HAMMER_REPS 64        /* sender hammer reps per aggressor line per coupling poll */
#endif
#ifndef WRITER_CORE
#define WRITER_CORE 0         /* sender / aggressor core (-DWRITER_CORE=N) */
#endif
#ifndef OBSERVER_CORE
#define OBSERVER_CORE 1       /* receiver / prime+probe core (-DOBSERVER_CORE=N) */
#endif
#ifndef SEED_WINDOW
#define SEED_WINDOW 0         /* independent realization index (-DSEED_WINDOW=K) */
#endif

/* The 64 SETS the sender targets per MODE (the .holo MODE_SETS, identical
 * geometry to the analyzer's MODE_SETS + the base real/pseudo/wrong line maps).
 * In conflict mode these index EVICTION SETS instead of single lines. */

typedef struct { uint8_t *bytes; } tape_t;          /* the reversible MESSAGE tape (bit-exact) */

typedef struct {
    tape_t  tape;                                   /* message tape: 64 lines, reversible XOR + hash */
    uint8_t *prime;                                 /* receiver-owned: 64 sets x EVSET_W lines */
    uint8_t *aggr;                                  /* sender-owned mirror: 64 sets x EVSET_W lines */
    atomic_int state;                               /* 0 idle, 1 probe-window open, 2 done, 3 shutdown */
    int trial, family, declared, actual;
    int agg_sets[TOUCHES];                          /* the eviction sets the sender hammers this drive point */
    int n_agg;
    double samples[LINES];                          /* per-set re-access latency vector -> analyzer */
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

/* ---- MESSAGE TAPE (reversible, bit-exact; unchanged from base) ---- */
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
    p[1] ^= (mask << 9) | (mask >> 55); p[0] ^= mask;   /* self-inverse: bytes restore bit-exact */
}

/* ---- canonical .holo families (identical to base real/pseudo/wrong maps) ---- */
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

/* keep the base scramble/echo so warmth fallback reproduces the base read */
static void apply_schedule(tape_t *t, int family, int declared_mode, int actual_mode, int trial,
                           const int lines[TOUCHES]) {
    uint64_t mask = 0x43524F5353434F52ULL ^ (uint64_t)family ^
                    ((uint64_t)declared_mode << 8) ^ ((uint64_t)(actual_mode + 3) << 17) ^
                    ((uint64_t)trial << 25);
    for (int pass = 0; pass < 6; pass++) {
        for (int i = 0; i < TOUCHES; i++) {
            int idx = (i * 5 + pass * 7) % TOUCHES;
            int line = lines[idx];
            touch_line(t, line, REPS);
            reversible_xor_line(t, line, mask ^ (uint64_t)(i * 0x9E3779B1U) ^ (uint64_t)pass);
        }
    }
}
static void echo_schedule(tape_t *t, const int lines[TOUCHES]) {
    for (int pass = 0; pass < 4; pass++)
        for (int i = 0; i < TOUCHES; i++) touch_line(t, lines[(i * 5 + pass * 7) % TOUCHES], ECHO_REPS);
}

/* ============================ CONFLICT CARRIER ============================ */
/* Eviction-set geometry: 64 sets, each EVSET_W lines. A set's W lines are
 * SET_STRIDE apart (same L3 congruence class). Sets are spaced one cache line
 * (64B) apart at the base so they occupy distinct congruence classes.
 *   addr(set, k) = base + set*64 + k*SET_STRIDE ,  k in [0, EVSET_W)
 */
static volatile uint64_t *evset_line(uint8_t *buf, int set, int k) {
    return (volatile uint64_t *)(buf + (size_t)set * 64 + (size_t)k * SET_STRIDE);
}

/* receiver PRIME: fill its own eviction set (bring all W lines resident) */
static void prime_set(uint8_t *buf, int set) {
    for (int k = 0; k < EVSET_W; k++) { volatile uint64_t *p = evset_line(buf, set, k); sink ^= *p; }
}

/* receiver PROBE: time its own ring re-traversal of the set; return total cycles.
 * If the sender evicted these lines, re-access pays L3/mem misses => slower. */
static uint64_t probe_set(uint8_t *buf, int set) {
    uint64_t a = rdtscp_now();
    for (int k = 0; k < EVSET_W; k++) { volatile uint64_t *p = evset_line(buf, set, k); sink ^= *p; }
    uint64_t b = rdtscp_now();
    return b - a;
}

/* sender HAMMER: continuously access aggressor lines in the targeted sets,
 * evicting the receiver's resident lines from the shared L3 congruence class. */
static void hammer_sets(uint8_t *buf, const int sets[TOUCHES], int n) {
    for (int rep = 0; rep < HAMMER_REPS; rep++)
        for (int s = 0; s < n; s++) {
            int set = sets[s];
            for (int k = 0; k < EVSET_W; k++) { volatile uint64_t *p = evset_line(buf, set, k); sink ^= *p; }
        }
}

static void init_buf(uint8_t *buf, size_t bytes, uint64_t seed) {
    uint64_t rng = seed;
    for (size_t j = 0; j + 8 <= bytes; j += 64) { uint64_t v = lcg(&rng); memcpy(buf + j, &v, sizeof(v)); }
}

/* warmth fallback measure (base behavior): single-line first-access time */
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
#if CONFLICT
        /* PRIME+PROBE: prime all 64 sets, then probe each per-set ring traversal
         * while the sender hammers this MODE's aggressor sets. Average N_AVG
         * passes per set for noise reduction (the on-box analog of sim's N). */
        for (int s = 0; s < LINES; s++) prime_set(shared.prime, s);
        for (int s = 0; s < LINES; s++) {
            double acc = 0.0;
            for (int a = 0; a < N_AVG; a++) {
                prime_set(shared.prime, s);          /* re-prime this set */
                acc += (double)probe_set(shared.prime, s) / (double)EVSET_W;  /* cycles per touch */
            }
            shared.samples[s] = acc / (double)N_AVG;
        }
#else
        /* WARMTH FALLBACK (base): raw per-line first-access timing on the message
         * tape, fixed scan order, no prime, no eviction sets. Must reproduce ~0.275. */
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
    size_t buf_bytes = (size_t)LINES * 64 + (size_t)EVSET_W * SET_STRIDE; /* covers all 64 sets x W lines */
    if (posix_memalign((void **)&shared.tape.bytes, 4096, (size_t)LINES * LINE_STRIDE) != 0) return 2;
#if CONFLICT
    if (posix_memalign((void **)&shared.prime, 4096, buf_bytes) != 0) return 2;
    if (posix_memalign((void **)&shared.aggr,  4096, buf_bytes) != 0) return 2;
#endif
    atomic_init(&shared.state, 0);

    pthread_t observer;
    if (pthread_create(&observer, NULL, observer_thread, NULL) != 0) return 3;
    pin_core(WRITER_CORE);

    printf("family,declared_mode,actual_mode,trial,hash_restored");
    for (int line = 0; line < LINES; line++) printf(",l%02d", line);
    printf("\n");

    int restored_total = 0, total = 0;
    uint64_t sw = (uint64_t)(SEED_WINDOW) * 0x100000001B3ULL;

    for (int trial = 0; trial < TRIALS; trial++) {
        for (int family = 0; family < FAMILIES; family++) {
            for (int declared = 0; declared < MODES; declared++) {
                int actual = declared, lines[TOUCHES];
                schedule_lines(family, declared, &actual, lines);

                /* message tape: reversible, bit-exact (per-window fresh draw) */
                init_tape(&shared.tape, 0x4404B000D0000000ULL ^ (uint64_t)trial ^ (sw << 8));
                uint64_t h0 = tape_hash(&shared.tape);
                flush_tape(&shared.tape);
                apply_schedule(&shared.tape, family, declared, actual, trial, lines);
                echo_schedule(&shared.tape, lines);

#if CONFLICT
                /* conflict buffers: fresh draw per window; the sender targets the
                 * SAME eviction sets the .holo MODE selects (lines[] reused as set ids). */
                init_buf(shared.prime, buf_bytes, 0x5111A6E70000ULL ^ (uint64_t)trial ^ (sw << 11));
                init_buf(shared.aggr,  buf_bytes, 0xA66E55070000ULL ^ (uint64_t)trial ^ (sw << 13));
                for (int i = 0; i < TOUCHES; i++) shared.agg_sets[i] = lines[i] & (LINES - 1);
                shared.n_agg = TOUCHES;
#endif
                shared.trial = trial; shared.family = family;
                shared.declared = declared; shared.actual = actual;
                atomic_store_explicit(&shared.state, 1, memory_order_release);
#if CONFLICT
                /* keep hammering the targeted sets until the receiver finishes probing */
                while (atomic_load_explicit(&shared.state, memory_order_acquire) != 2)
                    hammer_sets(shared.aggr, shared.agg_sets, shared.n_agg);
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
            "PHASE4B_CACHE_HOLOGRAM_CROSS_CORE_CONFLICT conflict=%d evset_w=%d n_avg=%d seed_window=%d "
            "restored=%d/%d writer_core=%d observer_core=%d sink=%llu\n",
            CONFLICT, EVSET_W, N_AVG, (int)(SEED_WINDOW), restored_total, total,
            WRITER_CORE, OBSERVER_CORE, (unsigned long long)sink);

    free(shared.tape.bytes);
#if CONFLICT
    free(shared.prime); free(shared.aggr);
#endif
    return restored_total == total ? 0 : 1;
}
