#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#define SEGMENTS 64
#define SEGMENT_US 50000

typedef struct {
    volatile uint64_t state;
    volatile uint64_t edge;
    volatile uint64_t stop;
    volatile uint64_t counters[8];
} shared_t;

static shared_t *g;

static void pin_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

static uint64_t mix_step(uint64_t x) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    x ^= x >> 17;
    x *= 0x9e3779b97f4a7c15ULL;
    return x ^ (x >> 31);
}

static void worker(int core, int role) {
    pin_core(core);
    uint64_t x = 0x123456789abcdef0ULL ^ (uint64_t)(core * 0x10001 + role);
    while (!__atomic_load_n(&g->stop, __ATOMIC_RELAXED)) {
        uint64_t state = __atomic_load_n(&g->state, __ATOMIC_RELAXED);
        uint64_t local_mode = (state >> (role * 4)) & 0xf;
        if (local_mode == 0) {
            _mm_pause();
        } else if (local_mode == 1) {
            x = mix_step(x);
        } else if (local_mode == 2) {
            for (int i = 0; i < 64; i++) x = mix_step(x + (uint64_t)i);
        } else if (local_mode == 3) {
            __atomic_fetch_xor(&g->edge, mix_step(x), __ATOMIC_SEQ_CST);
            x += g->edge | 1ULL;
        } else {
            x += __builtin_popcountll(x) + local_mode;
            if (x & 1) x = mix_step(x);
        }
        g->counters[core] = x;
    }
    _exit(0);
}

static void set_state(uint64_t segment) {
    static const uint64_t pattern[] = {
        0x0111, 0x0121, 0x0131, 0x0011,
        0x0101, 0x0110, 0x0331, 0x0441,
    };
    uint64_t s = pattern[segment % (sizeof(pattern) / sizeof(pattern[0]))];
    __atomic_store_n(&g->state, s, __ATOMIC_RELAXED);
    __atomic_fetch_add(&g->edge, 1, __ATOMIC_SEQ_CST);
}

int main(int argc, char **argv) {
    int segments = (argc > 1) ? atoi(argv[1]) : SEGMENTS;
    int segment_us = (argc > 2) ? atoi(argv[2]) : SEGMENT_US;
    if (segments < 1) segments = SEGMENTS;
    if (segment_us < 1000) segment_us = SEGMENT_US;

    g = mmap(NULL, sizeof(shared_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (g == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    memset(g, 0, sizeof(*g));

    pid_t p3 = fork();
    if (p3 == 0) worker(3, 0);
    pid_t p4 = fork();
    if (p4 == 0) worker(4, 1);
    pid_t p5 = fork();
    if (p5 == 0) worker(5, 2);

    pin_core(2);
    printf("segment,tsc,state,edge,c3,c4,c5\n");
    for (int i = 0; i < segments; i++) {
        set_state((uint64_t)i);
        uint64_t t = __rdtsc();
        uint64_t state = __atomic_load_n(&g->state, __ATOMIC_RELAXED);
        uint64_t edge = __atomic_load_n(&g->edge, __ATOMIC_RELAXED);
        printf("%d,%lu,0x%lx,%lu,%lu,%lu,%lu\n", i, t, state, edge,
               g->counters[3], g->counters[4], g->counters[5]);
        fflush(stdout);
        usleep((useconds_t)segment_us);
    }

    __atomic_store_n(&g->stop, 1, __ATOMIC_RELAXED);
    waitpid(p3, NULL, 0);
    waitpid(p4, NULL, 0);
    waitpid(p5, NULL, 0);
    return 0;
}
