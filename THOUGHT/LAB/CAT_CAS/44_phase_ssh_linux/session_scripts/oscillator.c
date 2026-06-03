#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <unistd.h>

static inline uint64_t osc_step(uint64_t x) {
    asm volatile("" : "+r"(x));
    x = (x * 0x41C64E6D + 0x3039);
    x = (x >> 13) ^ x;
    x = (x << 17) + x;
    return x;
}

int main(int argc, char **argv) {
    int core = atoi(argv[1]);
    unsigned long iterations = strtoul(argv[2], NULL, 10);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    uint64_t x = 0xDEADBEEFCAFEBABE;
    for (unsigned long i = 0; i < iterations; i++) x = osc_step(x);
    printf("%lu\n", x);
    return 0;
}
