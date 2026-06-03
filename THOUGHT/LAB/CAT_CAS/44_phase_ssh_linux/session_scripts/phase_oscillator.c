#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

static volatile uint64_t *phase_buffer;

void oscillator(int core, uint64_t iterations) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    uint64_t x = 0xDEADBEEF;
    volatile uint64_t *my_phase = phase_buffer + core;

    for (uint64_t i = 0; i < iterations; i++) {
        x = (x * 0x41C64E6D + 0x3039);
        x = (x >> 13) ^ x;
        x = (x << 17) + x;

        if (i % 1000 == 0) {
            __atomic_store_n(my_phase, x & 0xFFFFFFFF, __ATOMIC_RELAXED);
        }
    }
}

int main() {
    phase_buffer = (volatile uint64_t *)mmap(NULL, 4096,
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    pid_t pid3, pid4;

    if ((pid3 = fork()) == 0) {
        oscillator(3, 100000000);
        _exit(0);
    }

    if ((pid4 = fork()) == 0) {
        oscillator(4, 100000000);
        _exit(0);
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(2, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    uint64_t phase_samples[2000];
    for (int i = 0; i < 2000; i++) {
        uint64_t p3 = __atomic_load_n(&phase_buffer[3], __ATOMIC_RELAXED);
        uint64_t p4 = __atomic_load_n(&phase_buffer[4], __ATOMIC_RELAXED);
        phase_samples[i] = (p3 << 32) | (p4 & 0xFFFFFFFF);
        for (volatile int d = 0; d < 200; d++);
    }

    fwrite(phase_samples, sizeof(uint64_t), 2000, stdout);

    waitpid(pid3, NULL, 0);
    waitpid(pid4, NULL, 0);
    munmap((void *)phase_buffer, 4096);
    return 0;
}
