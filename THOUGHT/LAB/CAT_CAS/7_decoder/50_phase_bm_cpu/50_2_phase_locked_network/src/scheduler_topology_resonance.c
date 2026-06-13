#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <time.h>

typedef struct {
    int core;
    int mode;
    int offset_iters;
    int rounds;
    uint64_t seed;
    atomic_int *go;
    atomic_uint_fast64_t *shared;
    uint64_t elapsed_ns;
    uint64_t carrier;
    uint64_t hash_before;
    uint64_t hash_after;
} worker_t;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static uint64_t hash_tape(uint64_t *tape, int n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < n; i++) {
        h ^= tape[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

static int answer_for_seed(uint64_t seed) {
    uint64_t x = mix64(seed ^ 0xc0010068ULL);
    return (int)((__builtin_popcountll(x) ^ (x >> 7) ^ (x >> 19)) & 1ULL);
}

static void pin_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "sched_setaffinity core %d failed: %s\n", core, strerror(errno));
        exit(2);
    }
}

static void *worker_main(void *arg) {
    worker_t *w = (worker_t *)arg;
    pin_core(w->core);

    enum { N = 128 };
    uint64_t tape[N];
    for (int i = 0; i < N; i++) {
        tape[i] = mix64(w->seed + (uint64_t)i * 0x9e3779b97f4a7c15ULL);
    }
    w->hash_before = hash_tape(tape, N);

    while (atomic_load_explicit(w->go, memory_order_acquire) == 0) {
    }
    for (int i = 0; i < w->offset_iters; i++) {
        __asm__ __volatile__("" ::: "memory");
    }

    uint64_t carrier = mix64(w->seed ^ 0xa5a5a5a5ULL);
    uint64_t start = now_ns();
    for (int r = 0; r < w->rounds; r++) {
        int idx = (r * 13 + 7) & (N - 1);
        int jdx = (r * 29 + 3) & (N - 1);
        uint64_t shared_value = 0;
        if (w->mode == 1) {
            shared_value = atomic_load_explicit(w->shared, memory_order_relaxed);
        } else if (w->mode == 2) {
            shared_value = atomic_fetch_add_explicit(w->shared, 0x9e3779b97f4a7c15ULL, memory_order_relaxed);
        }
        uint64_t mask = mix64(tape[jdx] ^ carrier ^ shared_value ^ (uint64_t)r);
        tape[idx] ^= mask;
        carrier = mix64(carrier ^ tape[idx] ^ shared_value ^ (uint64_t)idx);
        tape[idx] ^= mask;
    }
    w->elapsed_ns = now_ns() - start;
    w->carrier = carrier;
    w->hash_after = hash_tape(tape, N);
    return NULL;
}

static const char *mode_name(int mode) {
    if (mode == 0) return "independent";
    if (mode == 1) return "shared_read";
    return "shared_atomic";
}

int main(int argc, char **argv) {
    int core_a = 2, core_b = 3, trials = 16, rounds = 8192, seed_start = 20000;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--core-a") && i + 1 < argc) core_a = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--core-b") && i + 1 < argc) core_b = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trials") && i + 1 < argc) trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed-start") && i + 1 < argc) seed_start = atoi(argv[++i]);
    }

    int offsets[] = {0, 64, 256, 1024};
    printf("seed,mode,offset_iters,core_a,core_b,answer,restore_ok,elapsed_a_ns,elapsed_b_ns,delta_abs_ns,sum_ns,carrier_low\n");
    for (int mode = 0; mode < 3; mode++) {
        for (int oi = 0; oi < 4; oi++) {
            for (int t = 0; t < trials; t++) {
                uint64_t seed = (uint64_t)seed_start + (uint64_t)(mode * 100000 + oi * 10000 + t);
                atomic_int go;
                atomic_uint_fast64_t shared;
                atomic_init(&go, 0);
                atomic_init(&shared, mix64(seed));
                worker_t a = {core_a, mode, 0, rounds, seed ^ 0x1111ULL, &go, &shared, 0, 0, 0, 0};
                worker_t b = {core_b, mode, offsets[oi], rounds, seed ^ 0x2222ULL, &go, &shared, 0, 0, 0, 0};
                pthread_t ta, tb;
                pthread_create(&ta, NULL, worker_main, &a);
                pthread_create(&tb, NULL, worker_main, &b);
                atomic_store_explicit(&go, 1, memory_order_release);
                pthread_join(ta, NULL);
                pthread_join(tb, NULL);
                uint64_t delta = a.elapsed_ns > b.elapsed_ns ? a.elapsed_ns - b.elapsed_ns : b.elapsed_ns - a.elapsed_ns;
                int restore_ok = (a.hash_before == a.hash_after) && (b.hash_before == b.hash_after);
                int answer = answer_for_seed(seed);
                printf("%" PRIu64 ",%s,%d,%d,%d,%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d\n",
                       seed, mode_name(mode), offsets[oi], core_a, core_b, answer, restore_ok,
                       a.elapsed_ns, b.elapsed_ns, delta, a.elapsed_ns + b.elapsed_ns,
                       (int)((a.carrier ^ b.carrier) & 1ULL));
            }
        }
    }
    return 0;
}
