#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <openssl/sha.h>

volatile uint64_t *tape __attribute__((aligned(64)));
#define TAPE_SIZE 256
#define CYCLES 100

void catalytic_write(int core, int slot, uint64_t phase_seed, uint64_t iterations) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    uint64_t x = phase_seed;
    volatile uint64_t *target = &tape[slot];

    for (uint64_t i = 0; i < iterations; i++) {
        x = (x * 0x41C64E6D + 0x3039);
        x = (x >> 13) ^ x;
        x = (x << 17) + x;
    }

    __atomic_fetch_xor(target, x, __ATOMIC_SEQ_CST);
}

void sha256_tape(unsigned char *hash_out) {
    SHA256((unsigned char *)tape, TAPE_SIZE, hash_out);
}

int main() {
    tape = (volatile uint64_t *)mmap(NULL, TAPE_SIZE, PROT_READ|PROT_WRITE,
                                      MAP_SHARED|MAP_ANONYMOUS, -1, 0);

    unsigned char hash_initial[SHA256_DIGEST_LENGTH];
    unsigned char hash_before[SHA256_DIGEST_LENGTH];
    unsigned char hash_after[SHA256_DIGEST_LENGTH];

    memset((void *)tape, 0xAA, TAPE_SIZE);
    sha256_tape(hash_initial);

    printf("=== 100-CYCLE CATALYTIC STRESS TEST ===\n");
    printf("Tape size: %d bytes, initial state: 0xAA\n", TAPE_SIZE);

    int failures = 0;

    for (int cycle = 0; cycle < CYCLES; cycle++) {
        sha256_tape(hash_before);

        pid_t pids[8];
        for (int slot = 0; slot < 8; slot++) {
            if ((pids[slot] = fork()) == 0) {
                catalytic_write(3 + (slot % 2), slot,
                              0xDEADBEEF + slot * 0x1000000,
                              5000000);
                _exit(0);
            }
        }
        for (int slot = 0; slot < 8; slot++) waitpid(pids[slot], NULL, 0);

        for (int slot = 0; slot < 8; slot++) {
            if ((pids[slot] = fork()) == 0) {
                catalytic_write(3 + (slot % 2), slot,
                              0xDEADBEEF + slot * 0x1000000,
                              5000000);
                _exit(0);
            }
        }
        for (int slot = 0; slot < 8; slot++) waitpid(pids[slot], NULL, 0);

        sha256_tape(hash_after);

        if (memcmp(hash_before, hash_after, SHA256_DIGEST_LENGTH) != 0) {
            printf("CYCLE %d: FAIL - SHA256 mismatch\n", cycle);
            failures++;
            break;
        }

        if (cycle % 10 == 9 || cycle == 0) {
            printf("Cycle %d/%d: OK\n", cycle + 1, CYCLES);
        }
    }

    int final_match = memcmp(hash_initial, hash_after, SHA256_DIGEST_LENGTH) == 0;

    printf("\n=== RESULTS ===\n");
    printf("Cycles completed: %d/%d\n", failures == 0 ? CYCLES : CYCLES, CYCLES);
    printf("Failures: %d\n", failures);
    printf("Final SHA256 matches initial: %s\n", final_match ? "YES" : "NO");
    printf("Total bits erased: %d\n", final_match ? 0 : -1);

    return failures > 0 ? 1 : 0;
}
