#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <openssl/sha.h>

#define TAPE_KB 64
#define MAX_ITERS 512
#define PREDECLARED_FP 42

static uint64_t rng_s;
static uint64_t rng64(void) {
    uint64_t x = rng_s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_s = x;
    return x * UINT64_C(0x2545F4914F6CDD1D);
}

static void fill_tape(uint8_t *t, size_t n, uint64_t s) {
    size_t i;
    rng_s = s | UINT64_C(1);
    for (i = 0; i < n; ++i) t[i] = (uint8_t)(rng64() & 0xffU);
}

static void tape_sha256(const uint8_t *t, size_t n, uint8_t h[32]) {
    SHA256(t, n, h);
}

static void xor_mask(uint8_t *t, size_t n, uint64_t s) {
    size_t i;
    rng_s = s | UINT64_C(1);
    for (i = 0; i < n; ++i) t[i] ^= (uint8_t)(rng64() & 0xffU);
}

static int hash_eq(const uint8_t *a, const uint8_t *b) {
    return memcmp(a, b, 32) == 0;
}

/*
 * Unique integer contraction onto PREDECLARED_FP.
 * C integer division truncates toward zero, so every nonzero signed distance
 * is strictly reduced and both adjacent states map to the fixed point:
 *   f(41)=42, f(42)=42, f(43)=42.
 */
static int f_contract(int x) {
    return PREDECLARED_FP + (x - PREDECLARED_FP) / 2;
}

static int f_nofp(int x) {
    return (x + 1) % 256;
}

static int run_contract(uint8_t *tape, size_t tape_size, uint64_t tid,
                        int start_x, int *final_x, int *iterations,
                        int *tape_ok) {
    int x = start_x;
    int it;
    *tape_ok = 1;
    for (it = 0; it < MAX_ITERS; ++it) {
        uint8_t before[32], after[32];
        uint64_t mask_seed = tid ^ (uint64_t)it ^ UINT64_C(0xA5A5);
        int next;
        fill_tape(tape, tape_size, tid + (uint64_t)it);
        tape_sha256(tape, tape_size, before);
        xor_mask(tape, tape_size, mask_seed);
        next = f_contract(x);
        xor_mask(tape, tape_size, mask_seed);
        tape_sha256(tape, tape_size, after);
        if (!hash_eq(before, after)) {
            *tape_ok = 0;
            break;
        }
        x = next;
        if (x == PREDECLARED_FP && f_contract(x) == x) {
            ++it;
            *final_x = x;
            *iterations = it;
            return 1;
        }
    }
    *final_x = x;
    *iterations = it;
    return 0;
}

int main(int argc, char **argv) {
    int seeds = 10;
    uint64_t base_seed = 42;
    const char *csv_path = "fp_results.csv";
    size_t tape_size = (size_t)TAPE_KB * 1024U;
    uint8_t *tape;
    FILE *csv;
    int catalytic_trials = 0;
    int catalytic_pass = 0;
    int baseline_pass = 0;
    int identity_pass = 0;
    int wrong_restore_pass = 0;
    int nofp_pass = 0;
    int replay_pass = 0;
    int s;

    for (s = 1; s < argc; ++s) {
        if (!strcmp(argv[s], "--seeds") && s + 1 < argc) seeds = atoi(argv[++s]);
        else if (!strcmp(argv[s], "--seed") && s + 1 < argc) base_seed = (uint64_t)strtoull(argv[++s], NULL, 10);
        else if (!strcmp(argv[s], "--csv") && s + 1 < argc) csv_path = argv[++s];
    }
    if (seeds <= 0) {
        fprintf(stderr, "seeds must be positive\n");
        return 2;
    }

    tape = (uint8_t *)malloc(tape_size);
    if (!tape) return 1;
    csv = fopen(csv_path, "w");
    if (!csv) {
        free(tape);
        return 1;
    }

    fprintf(csv, "seed,mode,start_x,final_x,iterations,converged,tape_ok,baseline_iters\n");
    printf("EXP50 L3 MECHANICAL WARMUP seeds=%d fp=%d map=signed_distance_halving_v2\n",
           seeds, PREDECLARED_FP);

    for (s = 0; s < seeds; ++s) {
        uint64_t seed = base_seed + (uint64_t)s * 1000U;
        int start;
        for (start = 10; start <= 250; start += 30) {
            uint64_t tid = seed + (uint64_t)start;
            int final_x, iterations, tape_ok;
            int converged = run_contract(tape, tape_size, tid, start,
                                         &final_x, &iterations, &tape_ok);
            ++catalytic_trials;
            if (converged && tape_ok && final_x == PREDECLARED_FP) ++catalytic_pass;
            fprintf(csv, "%llu,catalytic_loop,%d,%d,%d,%s,%s,NA\n",
                    (unsigned long long)tid, start, final_x, iterations,
                    converged ? "YES" : "NO", tape_ok ? "PASS" : "FAIL");
        }

        {
            uint64_t tid = seed + 777U;
            int x;
            int found = -1;
            int iterations = 0;
            for (x = 0; x < 256; ++x) {
                ++iterations;
                if (f_contract(x) == x) {
                    found = x;
                    break;
                }
            }
            if (found == PREDECLARED_FP) ++baseline_pass;
            fprintf(csv, "%llu,forward_scan,0,%d,%d,%s,NA,%d\n",
                    (unsigned long long)tid, found, iterations,
                    found == PREDECLARED_FP ? "YES" : "NO", iterations);
        }

        {
            uint64_t tid = seed + 888U;
            uint8_t before[32], after[32];
            uint64_t mask_seed = tid ^ UINT64_C(0xA5A5);
            fill_tape(tape, tape_size, tid);
            tape_sha256(tape, tape_size, before);
            xor_mask(tape, tape_size, mask_seed);
            xor_mask(tape, tape_size, mask_seed);
            tape_sha256(tape, tape_size, after);
            if (hash_eq(before, after)) ++identity_pass;
            fprintf(csv, "%llu,identity_loop,10,10,1,YES,%s,NA\n",
                    (unsigned long long)tid,
                    hash_eq(before, after) ? "PASS" : "FAIL");
        }

        {
            uint64_t tid = seed + 1111U;
            uint8_t before[32], after[32];
            uint64_t mask_seed = tid ^ UINT64_C(0xA5A5);
            fill_tape(tape, tape_size, tid);
            tape_sha256(tape, tape_size, before);
            xor_mask(tape, tape_size, mask_seed);
            (void)f_contract(10);
            xor_mask(tape, tape_size, mask_seed ^ UINT64_C(0xDEAD));
            tape_sha256(tape, tape_size, after);
            if (!hash_eq(before, after)) ++wrong_restore_pass;
            fprintf(csv, "%llu,wrong_restore,10,NA,1,%s,NA,NA\n",
                    (unsigned long long)tid,
                    !hash_eq(before, after) ? "FAIL_OK" : "UNEXPECTED_PASS");
        }

        {
            uint64_t tid = seed + 2222U;
            int x = 10;
            int it;
            int converged = 0;
            int tape_ok = 1;
            for (it = 0; it < MAX_ITERS; ++it) {
                uint8_t before[32], after[32];
                uint64_t mask_seed = tid ^ (uint64_t)it ^ UINT64_C(0xA5A5);
                int next;
                fill_tape(tape, tape_size, tid + (uint64_t)it);
                tape_sha256(tape, tape_size, before);
                xor_mask(tape, tape_size, mask_seed);
                next = f_nofp(x);
                xor_mask(tape, tape_size, mask_seed);
                tape_sha256(tape, tape_size, after);
                if (!hash_eq(before, after)) {
                    tape_ok = 0;
                    break;
                }
                if (next == x) {
                    converged = 1;
                    break;
                }
                x = next;
            }
            if (!converged && tape_ok) ++nofp_pass;
            fprintf(csv, "%llu,no_fp_negative,10,%d,%d,%s,%s,NA\n",
                    (unsigned long long)tid, x, it,
                    !converged ? "NO_FP_OK" : "UNEXPECTED",
                    tape_ok ? "PASS" : "FAIL");
        }

        {
            uint64_t tid = seed + 999U;
            int x1, x2, it1, it2, tape1, tape2;
            int c1 = run_contract(tape, tape_size, tid, 10, &x1, &it1, &tape1);
            int c2 = run_contract(tape, tape_size, tid, 10, &x2, &it2, &tape2);
            int replay = c1 && c2 && tape1 && tape2 && x1 == x2 && it1 == it2;
            if (replay) ++replay_pass;
            fprintf(csv, "%llu,replay,10,%d,%d,%s,%s,NA\n",
                    (unsigned long long)tid, x1, it1,
                    replay ? "YES" : "NO", tape1 && tape2 ? "PASS" : "FAIL");
        }
    }

    fclose(csv);
    printf("\n=== RESULTS ===\n");
    printf("catalytic_loop: %d/%d\n", catalytic_pass, catalytic_trials);
    printf("forward_scan: %d/%d\n", baseline_pass, seeds);
    printf("identity: %d/%d\n", identity_pass, seeds);
    printf("wrong_restore: %d/%d\n", wrong_restore_pass, seeds);
    printf("no_fp_negative: %d/%d\n", nofp_pass, seeds);
    printf("replay: %d/%d\n", replay_pass, seeds);

    {
        int all = catalytic_pass == catalytic_trials &&
                  baseline_pass == seeds &&
                  identity_pass == seeds &&
                  wrong_restore_pass == seeds &&
                  nofp_pass == seeds &&
                  replay_pass == seeds;
        printf("VERDICT: %s\n", all ? "L3_MECHANICAL_WARMUP_PASS" : "L3_MECHANICAL_WARMUP_FAIL");
        printf("CLAIM: software/tape hygiene only; not frontier evidence\n");
        free(tape);
        return all ? 0 : 1;
    }
}
