#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>

volatile uint64_t *shared_line __attribute__((aligned(64)));

void lock_oscillator(int core, uint64_t iterations) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    uint64_t x = 0xDEADBEEF;

    for (uint64_t i = 0; i < iterations; i++) {
        x = (x * 0x41C64E6D + 0x3039);
        x = (x >> 13) ^ x;
        x = (x << 17) + x;

        if (i % 100 == 0) {
            uint64_t old, newval;
            do {
                old = __atomic_load_n(shared_line, __ATOMIC_RELAXED);
                newval = (old & 0xFFFFFFFF00000000) | (x & 0xFFFFFFFF);
            } while (!__atomic_compare_exchange_n(shared_line, &old, newval, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
        }
    }
}

int main() {
    shared_line = (volatile uint64_t *)mmap(NULL, 64, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
    *((uint64_t *)shared_line) = 0;

    pid_t p3, p4;
    uint64_t samples[2000];

    if ((p3 = fork()) == 0) {
        lock_oscillator(3, 50000000);
        _exit(0);
    }
    if ((p4 = fork()) == 0) {
        lock_oscillator(4, 50000000);
        _exit(0);
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    for (int i = 0; i < 2000; i++) {
        samples[i] = __atomic_load_n(shared_line, __ATOMIC_SEQ_CST);
        for (volatile int d = 0; d < 500; d++);
    }

    fwrite(samples, sizeof(uint64_t), 2000, stdout);

    waitpid(p3, NULL, 0);
    waitpid(p4, NULL, 0);
    return 0;
}
