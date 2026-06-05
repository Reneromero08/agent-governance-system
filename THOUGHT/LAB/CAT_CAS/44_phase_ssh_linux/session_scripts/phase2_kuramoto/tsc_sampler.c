#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <x86intrin.h>
#define NSAMPLES 2000000

int main(int argc, char **argv) {
    int core = atoi(argv[1]);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    uint64_t *buffer = (uint64_t *)aligned_alloc(64, NSAMPLES * sizeof(uint64_t));
    if (!buffer) { perror("alloc"); return 1; }
    for (int i = 0; i < 1000; i++) asm volatile("" ::: "memory");
    uint64_t start = __rdtsc();
    for (int i = 0; i < NSAMPLES; i++) buffer[i] = __rdtsc();
    uint64_t end = __rdtsc();
    fwrite(buffer, sizeof(uint64_t), NSAMPLES, stdout);
    fflush(stdout);
    fprintf(stderr, "Sampled %d in %lu cycles (%.2f cycles/sample)\n", NSAMPLES, end-start, (double)(end-start)/NSAMPLES);
    free(buffer);
    return 0;
}
