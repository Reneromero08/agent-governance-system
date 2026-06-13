#define _GNU_SOURCE
#include <openssl/sha.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define TAPE_SIZE 4096
#define LANES 8
#define CYCLES 100
#define ITERATIONS 3000000ULL

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t bytes;
    uint64_t invariant_id;
    double eigenbasis[8][8];
    uint64_t rotation_chain[16];
    uint8_t payload[1024];
} holo_payload_t;

static volatile uint64_t *tape;

static uint64_t fnv1a64(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void sha256_hex(const void *data, size_t len, char out[65]) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256((const unsigned char *)data, len, digest);
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        sprintf(out + i * 2, "%02x", digest[i]);
    }
    out[64] = '\0';
}

static void fill_holo_payload(holo_payload_t *h) {
    memset(h, 0, sizeof(*h));
    memcpy(h->magic, ".holo\0\0", 8);
    h->version = 1;
    h->bytes = sizeof(*h);
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            double base = (r == c) ? 1.0 : 0.0;
            h->eigenbasis[r][c] = base + ((double)((r + 1) * (c + 3) % 17) / 1024.0);
        }
    }
    for (int i = 0; i < 16; i++) {
        h->rotation_chain[i] = 0x9e3779b97f4a7c15ULL ^ (0x100000001b3ULL * (uint64_t)(i + 1));
    }
    for (int i = 0; i < (int)sizeof(h->payload); i++) {
        h->payload[i] = (uint8_t)((i * 73 + (i >> 2) * 19 + 0x5a) & 0xff);
    }
    h->invariant_id = fnv1a64(&h->eigenbasis[0][0], sizeof(h->eigenbasis)) ^
                      fnv1a64(&h->rotation_chain[0], sizeof(h->rotation_chain)) ^
                      fnv1a64(&h->payload[0], sizeof(h->payload));
}

static uint64_t lane_mask(int lane, uint64_t invariant) {
    uint64_t x = invariant ^ (0xd6e8feb86659fd93ULL * (uint64_t)(lane + 1));
    for (uint64_t i = 0; i < ITERATIONS; i++) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x *= 2685821657736338717ULL;
    }
    return x;
}

static void catalytic_lane(int core, int lane, uint64_t invariant) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    sched_setaffinity(0, sizeof(set), &set);

    uint64_t mask = lane_mask(lane, invariant);
    int stride = LANES;
    for (int slot = lane; slot < TAPE_SIZE / 8; slot += stride) {
        __atomic_fetch_xor(&tape[slot], mask ^ (0x9e3779b97f4a7c15ULL * (uint64_t)slot),
                           __ATOMIC_SEQ_CST);
    }
}

static int run_pass(uint64_t invariant) {
    pid_t pids[LANES];
    for (int lane = 0; lane < LANES; lane++) {
        pids[lane] = fork();
        if (pids[lane] == 0) {
            catalytic_lane(3 + (lane & 1), lane, invariant);
            _exit(0);
        }
        if (pids[lane] < 0) return 0;
    }
    for (int lane = 0; lane < LANES; lane++) {
        int status = 0;
        waitpid(pids[lane], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return 0;
    }
    return 1;
}

int main(void) {
    tape = mmap(NULL, TAPE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (tape == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    holo_payload_t h;
    fill_holo_payload(&h);
    memset((void *)tape, 0xa5, TAPE_SIZE);
    memcpy((void *)tape, &h, sizeof(h));

    char initial_sha[65], before_sha[65], forward_sha[65], after_sha[65];
    sha256_hex((const void *)tape, TAPE_SIZE, initial_sha);
    uint64_t expected_invariant = h.invariant_id;
    uint64_t restored_invariant = 0;
    int failures = 0;
    int modified_count = 0;

    printf("=== HOLO CATALYTIC TAPE GOAL ===\n");
    printf("Tape bytes: %d\n", TAPE_SIZE);
    printf("Payload bytes: %zu\n", sizeof(h));
    printf("Invariant: 0x%016lx\n", expected_invariant);
    printf("Initial SHA256: %s\n", initial_sha);

    for (int cycle = 0; cycle < CYCLES; cycle++) {
        sha256_hex((const void *)tape, TAPE_SIZE, before_sha);
        if (!run_pass(expected_invariant)) {
            printf("Cycle %d forward pass failed\n", cycle);
            failures++;
            break;
        }
        sha256_hex((const void *)tape, TAPE_SIZE, forward_sha);
        if (strcmp(before_sha, forward_sha) != 0) modified_count++;
        if (!run_pass(expected_invariant)) {
            printf("Cycle %d reverse pass failed\n", cycle);
            failures++;
            break;
        }
        sha256_hex((const void *)tape, TAPE_SIZE, after_sha);
        memcpy(&h, (const void *)tape, sizeof(h));
        restored_invariant = h.invariant_id;
        if (strcmp(before_sha, after_sha) != 0 || restored_invariant != expected_invariant ||
            memcmp(h.magic, ".holo\0\0", 8) != 0) {
            printf("Cycle %d restore mismatch\n", cycle);
            failures++;
            break;
        }
        if (cycle == 0 || cycle == 9 || cycle == 49 || cycle == 99) {
            printf("Cycle %03d forward_changed=%s restored_sha=%s\n",
                   cycle + 1, strcmp(before_sha, forward_sha) != 0 ? "YES" : "NO", after_sha);
        }
    }

    sha256_hex((const void *)tape, TAPE_SIZE, after_sha);
    printf("Final SHA256: %s\n", after_sha);
    printf("Cycles: %d/%d\n", failures ? modified_count : CYCLES, CYCLES);
    printf("Forward modifications observed: %d/%d\n", modified_count, CYCLES);
    printf("Invariant restored: %s\n", restored_invariant == expected_invariant ? "YES" : "NO");
    printf("SHA restored: %s\n", strcmp(initial_sha, after_sha) == 0 ? "YES" : "NO");
    printf("=== VERDICT: %s ===\n", failures == 0 && modified_count == CYCLES &&
                                      restored_invariant == expected_invariant &&
                                      strcmp(initial_sha, after_sha) == 0
                                  ? "HOLO_TAPE_RESTORED"
                                  : "HOLO_TAPE_FAILED");
    return failures == 0 && modified_count == CYCLES && restored_invariant == expected_invariant &&
                   strcmp(initial_sha, after_sha) == 0
               ? 0
               : 1;
}
