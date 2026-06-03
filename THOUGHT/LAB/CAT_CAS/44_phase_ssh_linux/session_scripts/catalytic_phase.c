#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#define OPENSSL_API_COMPAT 0x10100000L
#include <openssl/sha.h>

volatile uint64_t *tape __attribute__((aligned(64)));
#define TAPE_SIZE 256
#define SLOTS (TAPE_SIZE / 8)

void catalytic_write(int core, int slot, uint64_t phase_val, uint64_t iterations) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    uint64_t x = phase_val;
    for (uint64_t i = 0; i < iterations; i++) {
        x = (x * 0x41C64E6D + 0x3039);
        x = (x >> 13) ^ x;
        x = (x << 17) + x;
    }
    __atomic_fetch_xor(&tape[slot], x, __ATOMIC_SEQ_CST);
}

void sha256_tape(unsigned char *hash_out) {
    SHA256((unsigned char *)tape, TAPE_SIZE, hash_out);
}

void print_hash(const char *label, unsigned char *h) {
    printf("%s: ", label);
    for (int i = 0; i < 8; i++) printf("%02x", h[i]);
    printf("...\n");
}

int main() {
    tape = (volatile uint64_t *)mmap(NULL, TAPE_SIZE, PROT_READ|PROT_WRITE,
                                      MAP_SHARED|MAP_ANONYMOUS, -1, 0);

    unsigned char h0[SHA256_DIGEST_LENGTH], h1[SHA256_DIGEST_LENGTH];
    unsigned char hf[SHA256_DIGEST_LENGTH], hr[SHA256_DIGEST_LENGTH];
    int all_pass = 1;

    // === HARDENING GATE 1: RANDOM INITIAL TAPE ===
    memset((void *)tape, 0, TAPE_SIZE);
    srand(time(NULL));
    for (int i = 0; i < SLOTS; i++) {
        tape[i] = ((uint64_t)rand() << 32) | rand();
    }
    sha256_tape(h0);
    print_hash("Initial (random)", h0);

    // Verify tape has non-trivial content
    int is_zero = 1;
    for (int i = 0; i < SLOTS; i++) if (tape[i] != 0) is_zero = 0;
    printf("Gate 1 (non-zero tape): %s\n", !is_zero ? "PASS" : "FAIL");
    if (is_zero) all_pass = 0;

    // === HARDENING GATE 2: FORWARD MODIFIES TAPE ===
    pid_t p3 = fork();
    if (p3 == 0) { catalytic_write(3, 0, 0xCAFEBABE, 10000000); _exit(0); }
    pid_t p4 = fork();
    if (p4 == 0) { catalytic_write(4, 1, 0xDEADBEEF, 10000000); _exit(0); }
    waitpid(p3, NULL, 0);
    waitpid(p4, NULL, 0);

    sha256_tape(hf);
    print_hash("After forward  ", hf);
    int modified = memcmp(h0, hf, SHA256_DIGEST_LENGTH) != 0;
    printf("Gate 2 (tape modified): %s\n", modified ? "PASS" : "FAIL");
    if (!modified) all_pass = 0;

    // === HARDENING GATE 3: SLOTS 0-1 NON-TRIVIAL ===
    int slots_modified = (tape[0] != 0 || tape[1] != 0);
    printf("Gate 3 (XOR values present): ");
    for (int i = 0; i < 4; i++) printf("0x%016lx ", tape[i]);
    printf("\n");
    printf("Gate 3 result: %s\n", slots_modified ? "PASS" : "FAIL");
    if (!slots_modified) all_pass = 0;

    // === HARDENING GATE 4: REVERSE RESTORES ===
    p3 = fork();
    if (p3 == 0) { catalytic_write(3, 0, 0xCAFEBABE, 10000000); _exit(0); }
    p4 = fork();
    if (p4 == 0) { catalytic_write(4, 1, 0xDEADBEEF, 10000000); _exit(0); }
    waitpid(p3, NULL, 0);
    waitpid(p4, NULL, 0);

    sha256_tape(hr);
    print_hash("After reverse  ", hr);
    int restored = memcmp(h0, hr, SHA256_DIGEST_LENGTH) == 0;
    printf("Gate 4 (tape restored): %s\n", restored ? "PASS" : "FAIL");
    if (!restored) all_pass = 0;

    // === HARDENING GATE 5: MULTIPLE CYCLES ===
    int cycles_ok = 1;
    for (int cycle = 0; cycle < 5; cycle++) {
        sha256_tape(h1);
        p3 = fork(); if (p3 == 0) { catalytic_write(3, cycle % SLOTS, 0xBEEF0000 + cycle, 5000000); _exit(0); }
        waitpid(p3, NULL, 0);
        p3 = fork(); if (p3 == 0) { catalytic_write(3, cycle % SLOTS, 0xBEEF0000 + cycle, 5000000); _exit(0); }
        waitpid(p3, NULL, 0);
        sha256_tape(h1); // re-hash after reverse
        if (memcmp(h0, h1, SHA256_DIGEST_LENGTH) != 0) {
            cycles_ok = 0;
            printf("Cycle %d: CORRUPTION\n", cycle);
        }
    }
    printf("Gate 5 (%d cycles): %s\n", 5, cycles_ok ? "PASS" : "FAIL");
    if (!cycles_ok) all_pass = 0;

    // === HARDENING GATE 6: DIFFERENT SEEDS, SAME SLOT ===
    sha256_tape(h1);
    int seeds_ok = 1;
    for (int s = 0; s < 4; s++) {
        p3 = fork(); if (p3 == 0) { catalytic_write(3, 0, 0x11111111 * (s+1), 5000000); _exit(0); }
        waitpid(p3, NULL, 0);
        sha256_tape(h1);
        if (memcmp(h0, h1, SHA256_DIGEST_LENGTH) == 0) seeds_ok = 0;
        p3 = fork(); if (p3 == 0) { catalytic_write(3, 0, 0x11111111 * (s+1), 5000000); _exit(0); }
        waitpid(p3, NULL, 0);
        sha256_tape(h1);
        if (memcmp(h0, h1, SHA256_DIGEST_LENGTH) != 0) seeds_ok = 0;
    }
    printf("Gate 6 (multi-seed XOR): %s\n", seeds_ok ? "PASS" : "FAIL");
    if (!seeds_ok) all_pass = 0;

    printf("\n=== VERDICT: %s ===\n", all_pass ? "ALL GATES PASS" : "FAILURES DETECTED");
    return all_pass ? 0 : 1;
}
