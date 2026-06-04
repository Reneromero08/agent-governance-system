#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <x86intrin.h>

#define SAMPLES 32768
#define SAMPLE_DELAY 256

typedef struct {
    volatile uint64_t counter[8] __attribute__((aligned(64)));
    volatile uint64_t shared_line __attribute__((aligned(64)));
    volatile uint64_t stop __attribute__((aligned(64)));
    volatile uint64_t pad[64];
} shared_t;

static shared_t *g;

static void pin_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

static uint64_t step_lcg(uint64_t x) {
    x = x * 0x41C64E6DULL + 0x3039ULL;
    x ^= x >> 13;
    x += x << 17;
    return x;
}

static void worker(int core, int mode, int shared) {
    pin_core(core);
    uint64_t x = 0x9e3779b97f4a7c15ULL ^ (uint64_t)core;
    uint64_t local[64];
    for (int i = 0; i < 64; i++) local[i] = x + (uint64_t)i;

    while (!__atomic_load_n(&g->stop, __ATOMIC_RELAXED)) {
        if (mode == 0) {
            _mm_pause();
        } else if (mode == 1) {
            x = step_lcg(x);
            g->counter[core] = x;
        } else if (mode == 2) {
            x = x * 6364136223846793005ULL + 1442695040888963407ULL;
            x ^= x * 0xd6e8feb86659fd93ULL;
            g->counter[core] = x;
        } else if (mode == 3) {
            for (int i = 0; i < 64; i++) {
                local[i] = step_lcg(local[i] + x);
                x ^= local[i];
            }
            g->counter[core] = x;
        } else if (mode == 4) {
            x = step_lcg(x);
            __atomic_fetch_xor(&g->shared_line, x, __ATOMIC_SEQ_CST);
            g->counter[core] = x;
        } else if (mode == 5) {
            x += (uint64_t)(__builtin_popcountll(x) + 1);
            if (x & 1) x = step_lcg(x);
            g->counter[core] = x;
        }
        if (shared && mode != 4) {
            __atomic_fetch_add(&g->shared_line, x | 1ULL, __ATOMIC_SEQ_CST);
        }
    }
    _exit(0);
}

static int parse_mode(const char *s) {
    if (!strcmp(s, "off")) return 0;
    if (!strcmp(s, "lcg")) return 1;
    if (!strcmp(s, "mul")) return 2;
    if (!strcmp(s, "mem")) return 3;
    if (!strcmp(s, "atomic")) return 4;
    if (!strcmp(s, "branch")) return 5;
    return atoi(s);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s label mode3 mode4 mode5 shared sampler_core\n", argv[0]);
        return 2;
    }
    const char *label = argv[1];
    int mode3 = parse_mode(argv[2]);
    int mode4 = parse_mode(argv[3]);
    int mode5 = parse_mode(argv[4]);
    int shared = atoi(argv[5]);
    int sampler_core = atoi(argv[6]);

    g = mmap(NULL, sizeof(shared_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (g == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    memset(g, 0, sizeof(shared_t));

    pid_t p3 = fork();
    if (p3 == 0) worker(3, mode3, shared);
    pid_t p4 = fork();
    if (p4 == 0) worker(4, mode4, shared);
    pid_t p5 = fork();
    if (p5 == 0) worker(5, mode5, 0);

    pin_core(sampler_core);
    usleep(100000);
    printf("# label=%s mode3=%s mode4=%s mode5=%s shared=%d samples=%d\n",
           label, argv[2], argv[3], argv[4], shared, SAMPLES);
    printf("i,tsc,c3,c4,c5,shared\n");
    for (int i = 0; i < SAMPLES; i++) {
        uint64_t t = __rdtsc();
        uint64_t c3 = __atomic_load_n(&g->counter[3], __ATOMIC_RELAXED);
        uint64_t c4 = __atomic_load_n(&g->counter[4], __ATOMIC_RELAXED);
        uint64_t c5 = __atomic_load_n(&g->counter[5], __ATOMIC_RELAXED);
        uint64_t sl = __atomic_load_n(&g->shared_line, __ATOMIC_RELAXED);
        printf("%d,%lu,%lu,%lu,%lu,%lu\n", i, t, c3, c4, c5, sl);
        for (volatile int d = 0; d < SAMPLE_DELAY; d++);
    }
    __atomic_store_n(&g->stop, 1, __ATOMIC_RELAXED);
    waitpid(p3, NULL, 0);
    waitpid(p4, NULL, 0);
    waitpid(p5, NULL, 0);
    return 0;
}
