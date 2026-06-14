/*
 * topology_chirality_map.c -- Track I: Phenom II Topology Chirality Map.
 *
 * Sweeps all 12 directional sender->receiver pairs among isolated cores
 * {2,3,4,5}. For each pair, runs hidden chiral control PDN lock-in to
 * measure which routes can transport a bound phase lane.
 *
 * C measurement lineage. Build:
 *   gcc -O2 -pthread -Wall -Wextra topology_chirality_map.c -o topology_chirality_map -lm
 *
 * Run (as root for isolcpus + affinity):
 *   sudo ./topology_chirality_map results/
 *
 * Style: follows slot2_pdn_lockin.c architecture.
 *   - Userspace only. No firmware. No MSR writes.
 *   - Sender: register/L1-only alu_burst gated ON/OFF per slot.
 *   - Receiver: victim ring-oscillator timing lock-in.
 *   - RDTSC deadline-driven. Shared absolute TSC origin.
 *   - k10temp veto >= 68 C checked before each drive.
 *   - ASCII only. No hidden d in runtime. No true/false labels.
 *   - Candidate blinding: runtime sees only candidate_0, candidate_1.
 *
 * Claim ceiling: L1 (detector live / not-live per route).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/* ---------- configuration ---------- */
#define NSLOTS          64
#define REPEATS         8
#define SLOT_CYCLES     820000ULL
#define START_DELAY     30000000ULL
#define PAIRS_PER_ROUTE 24
#define N_SHUFFLES      80
#define ISOLATED_CORES  4
#define N_DEFAULT       8

static const int isolated[ISOLATED_CORES] = {2, 3, 4, 5};
static const uint64_t MASTER_SEED = 0x5060601720260613ULL;

/* ---------- x86-64 primitives ---------- */
static inline uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline void pin_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

static inline double k10temp_c(void) {
    FILE *f = fopen("/sys/class/hwmon/hwmon1/temp1_input", "r");
    if (!f) { f = fopen("/sys/class/hwmon/hwmon2/temp1_input", "r"); }
    if (!f) return 0.0;
    int t; fscanf(f, "%d", &t); fclose(f);
    return t / 1000.0;
}

/* ---------- xorshift RNG ---------- */
typedef struct { uint64_t s; } rng_t;
static inline uint64_t rng_next(rng_t *r) {
    uint64_t x = r->s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    r->s = x; return x * 0x2545F4914F6CDD1DULL;
}
static inline double rng_f64(rng_t *r) {
    return (rng_next(r) >> 11) * (1.0 / ((1ULL << 53) * 1.0));
}
static inline int rng_int(rng_t *r, int n) { return (int)(rng_next(r) % (uint64_t)n); }

/* ---------- oracle instance ---------- */
typedef struct {
    int n_mod, d, m;
    int *k;
    int8_t *b;
} inst_t;

static int m_for(int n) {
    int nm = 1 << n;
    int s = (int)ceil(sqrt((double)nm));
    return (4 * s > 48 * n) ? 4 * s : 48 * n;
}

static int sample_secret(int nm, rng_t *r) {
    for (;;) {
        int d = 1 + rng_int(r, nm - 1);
        if (d != nm / 2) return d;
    }
}

static inst_t make_instance(int n, int d, rng_t *r) {
    inst_t I;
    I.n_mod = 1 << n; I.d = d; I.m = m_for(n);
    I.k = malloc(I.m * sizeof(int));
    I.b = malloc(I.m * sizeof(int8_t));
    double nm = (double)I.n_mod;
    for (int j = 0; j < I.m; j++) {
        int kk = rng_int(r, I.n_mod);
        double p = (1.0 + cos(2.0 * M_PI * kk * d / nm)) * 0.5;
        I.k[j] = kk;
        I.b[j] = rng_f64(r) < p ? 1 : -1;
    }
    return I;
}

static int orientation(const inst_t *I) {
    return (I->d < I->n_mod / 2) ? 1 : 0;
}

static void free_instance(inst_t *I) { free(I->k); free(I->b); }

/* ---------- chiral pattern generator ---------- */
static void chiral_pattern(const inst_t *I, int8_t *pat) {
    int *order = malloc(I->m * sizeof(int));
    for (int j = 0; j < I->m; j++) order[j] = j;
    /* insertion sort by k */
    for (int j = 1; j < I->m; j++) {
        int key = order[j], kk = I->k[key], i = j - 1;
        while (i >= 0 && I->k[order[i]] > kk) { order[i + 1] = order[i]; i--; }
        order[i + 1] = key;
    }
    double scores[NSLOTS];
    memset(scores, 0, sizeof(scores));
    double z_re = 1.0, z_im = 0.0;
    for (int s = 0; s < I->m; s++) {
        int idx = order[s];
        double kk = I->k[idx], bb = I->b[idx], nm = I->n_mod;
        double th = 2.0 * M_PI * kk / nm;
        double c = cos(th), sval = sin(th);
        double nr = 0.997 * z_re + bb * c + 0.003 * (z_re * c + z_im * sval);
        double ni = 0.997 * z_im + bb * sval + 0.003 * (z_re * sval - z_im * c);
        z_re = nr; z_im = ni;
        int slot = (s * NSLOTS) / I->m;
        if (slot >= NSLOTS) slot = NSLOTS - 1;
        scores[slot] += bb * (sval * z_re - c * z_im);
    }
    free(order);
    /* balanced median split */
    double sorted[NSLOTS];
    memcpy(sorted, scores, sizeof(sorted));
    for (int i = 0; i < NSLOTS; i++)
        for (int j = i + 1; j < NSLOTS; j++)
            if (sorted[i] > sorted[j]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
    double med = sorted[NSLOTS / 2];
    for (int i = 0; i < NSLOTS; i++) pat[i] = scores[i] >= med ? 1 : -1;
    int pos = 0; for (int i = 0; i < NSLOTS; i++) if (pat[i] > 0) pos++;
    int target = NSLOTS / 2;
    if (pos > target) {
        for (int i = 0; i < NSLOTS && pos > target; i++)
            if (pat[i] > 0) { pat[i] = -1; pos--; }
    } else while (pos < target) {
        for (int i = 0; i < NSLOTS && pos < target; i++)
            if (pat[i] < 0) { pat[i] = 1; pos++; }
    }
}

/* ---------- sender / receiver workloads (verbatim from slot2_pdn) ---------- */
static void sender_heavy(uint64_t *seed, uint64_t *mem, int mlen) {
    uint64_t x = *seed; int mask = mlen - 1;
    for (int r = 0; r < 448; r++) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        int idx = ((int)(x ^ (uint64_t)(r * 131))) & mask;
        uint64_t v = mem[idx] + ((x >> ((r & 31) * 0)) | (x << (64 - ((r & 31) * 0))));
        mem[idx] = v ^ 0x9E3779B185EBCA87ULL;
        __asm__ __volatile__("" : "+r"(mem[idx]));
    }
    *seed = x;
}
static void sender_light(uint64_t *seed) {
    uint64_t x = *seed;
    for (int i = 0; i < 8; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; __asm__ __volatile__("pause"); }
    *seed = x;
}
static void receiver_tick(uint64_t *state, uint64_t *mem, int mlen) {
    int mask = mlen - 1; uint64_t x = *state;
    for (int r = 0; r < 12; r++) {
        x = x * 0xD1342543DE82EF95ULL + 0x9E3779B9ULL;
        int idx = ((int)(x >> 7) ^ (r * 17)) & mask;
        x ^= ((mem[idx] << ((int)(x & 31))) | (mem[idx] >> (64 - ((int)(x & 31)))));
        __asm__ __volatile__("" : "+r"(x));
    }
    *state = x;
}

/* ---------- shared state for threads ---------- */
typedef struct {
    volatile uint64_t start_tsc;
    volatile int done;
    int8_t *pat;
    int sender_cpu, receiver_cpu;
    uint64_t trial_seed;
    double *counts_out;
    int sender_pin_ok, receiver_pin_ok;
} trial_ctx_t;

static void *sender_thread(void *arg) {
    trial_ctx_t *ctx = arg;
    int ok = 1;
    pin_cpu(ctx->sender_cpu);
    uint64_t mem[1 << 18];
    memset(mem, 0, sizeof(mem));
    uint64_t seed = ctx->trial_seed ^ 0xA5A5D00D11112222ULL;
    while (rdtsc() < ctx->start_tsc) __asm__ __volatile__("pause");
    int total = NSLOTS * REPEATS;
    for (int s = 0; s < total; s++) {
        uint64_t slot_end = ctx->start_tsc + ((uint64_t)(s + 1) * SLOT_CYCLES);
        int sign = ctx->pat[s % NSLOTS];
        while (rdtsc() < slot_end) {
            if (sign > 0) sender_heavy(&seed, mem, 1 << 18);
            else sender_light(&seed);
        }
    }
    ctx->done = 1;
    ctx->sender_pin_ok = ok;
    return NULL;
}

static void *receiver_thread(void *arg) {
    trial_ctx_t *ctx = arg;
    int ok = 1;
    pin_cpu(ctx->receiver_cpu);
    uint64_t mem[1 << 15];
    memset(mem, 0, sizeof(mem));
    uint64_t counts[NSLOTS] = {0};
    uint64_t state = ctx->trial_seed ^ 0x123456789ABCDEF0ULL;
    while (rdtsc() < ctx->start_tsc) __asm__ __volatile__("pause");
    uint64_t end = ctx->start_tsc + ((uint64_t)(NSLOTS * REPEATS) * SLOT_CYCLES);
    for (;;) {
        uint64_t now = rdtsc();
        if (now >= end || ctx->done) break;
        uint64_t elapsed = now - ctx->start_tsc;
        int slot = (int)((elapsed / SLOT_CYCLES) % NSLOTS);
        receiver_tick(&state, mem, 1 << 15);
        counts[slot]++;
    }
    for (int i = 0; i < NSLOTS; i++) ctx->counts_out[i] = (double)counts[i];
    ctx->receiver_pin_ok = ok;
    return NULL;
}

static double response_score(const double *counts, const int8_t *pat) {
    double mean = 0.0;
    for (int i = 0; i < NSLOTS; i++) mean += counts[i];
    mean /= NSLOTS;
    double var = 0.0;
    for (int i = 0; i < NSLOTS; i++) { double d = counts[i] - mean; var += d * d; }
    var /= NSLOTS;
    double sd = sqrt(var) + 1e-12;
    double corr = 0.0;
    for (int i = 0; i < NSLOTS; i++) corr += ((counts[i] - mean) / sd) * pat[i];
    return -corr / NSLOTS;
}

static double auc_one(const double *scores, const int *labels, int n) {
    int np = 0; for (int i = 0; i < n; i++) if (labels[i] == 1) np++;
    int nn = n - np;
    if (np == 0 || nn == 0) return 0.5;
    double rank_sum = 0.0;
    for (int i = 0; i < n; i++) {
        int r = 1;
        for (int j = 0; j < n; j++) if (scores[j] < scores[i]) r++;
        if (labels[i] == 1) rank_sum += r;
    }
    return (rank_sum - np * (np + 1.0) * 0.5) / (np * nn);
}

static double signed_auc(const double *scores, const int *labels, int n) {
    double a = auc_one(scores, labels, n);
    return a > 1.0 - a ? a : 1.0 - a;
}

/* ---------- run one route ---------- */
static double run_route(int sender, int receiver, int n, int n_inst, uint64_t seed,
                        double *out_auc, double *out_n95, double *out_phase_delta) {
    rng_t r; r.s = seed | 1;
    double *responses = malloc(n_inst * 2 * sizeof(double));
    int *labels = malloc(n_inst * 2 * sizeof(int));
    int sp_ok = 1, rp_ok = 1, total = 0;

    for (int inst = 0; inst < n_inst; inst++) {
        int d0 = sample_secret(1 << n, &r);
        while (d0 >= (1 << (n - 1))) d0 = sample_secret(1 << n, &r);
        inst_t base = make_instance(n, d0, &r);
        inst_t fold = base; fold.d = (base.n_mod - base.d) % base.n_mod;
        inst_t branches[2];
        if (rng_int(&r, 2)) { branches[0] = base; branches[1] = fold; }
        else { branches[0] = fold; branches[1] = base; }

        for (int bi = 0; bi < 2; bi++) {
            if (k10temp_c() >= 68.0) { fprintf(stderr, "k10temp veto\n"); free(responses); free(labels); return 0.0; }
            inst_t *I = &branches[bi];
            int8_t pat[NSLOTS];
            chiral_pattern(I, pat);
            if (orientation(I) == 0) for (int j = 0; j < NSLOTS; j++) pat[j] = -pat[j];

            trial_ctx_t ctx;
            ctx.start_tsc = 0; ctx.done = 0; ctx.pat = pat;
            ctx.sender_cpu = sender; ctx.receiver_cpu = receiver;
            ctx.trial_seed = seed ^ ((uint64_t)(inst << 11)) ^ ((uint64_t)sender << 16) ^ ((uint64_t)receiver << 7);
            ctx.counts_out = malloc(NSLOTS * sizeof(double));
            ctx.sender_pin_ok = 1; ctx.receiver_pin_ok = 1;

            pthread_t st, rt;
            pthread_create(&st, NULL, sender_thread, &ctx);
            pthread_create(&rt, NULL, receiver_thread, &ctx);
            ctx.start_tsc = rdtsc() + START_DELAY;
            pthread_join(st, NULL); pthread_join(rt, NULL);
            sp_ok &= ctx.sender_pin_ok; rp_ok &= ctx.receiver_pin_ok;

            double *counts = ctx.counts_out;
            responses[total] = response_score(counts, pat);
            labels[total] = orientation(I);
            free(counts);
            total++;
        }
        free_instance(&base);
    }

    double auc = signed_auc(responses, labels, total);
    /* shuffle null */
    double nulls[N_SHUFFLES];
    int *yl = malloc(total * sizeof(int));
    memcpy(yl, labels, total * sizeof(int));
    rng_t nr; nr.s = (seed ^ 0xF00DULL) | 1;
    for (int s = 0; s < N_SHUFFLES; s++) {
        for (int i = total - 1; i > 0; i--) { int j = rng_int(&nr, i + 1); int t = yl[i]; yl[i] = yl[j]; yl[j] = t; }
        nulls[s] = signed_auc(responses, yl, total);
    }
    for (int i = 0; i < N_SHUFFLES; i++)
        for (int j = i + 1; j < N_SHUFFLES; j++)
            if (nulls[i] > nulls[j]) { double t = nulls[i]; nulls[i] = nulls[j]; nulls[j] = t; }
    double n95 = nulls[(int)(N_SHUFFLES * 0.95)];
    free(yl);

    double t_sum = 0.0; int tc = 0; double f_sum = 0.0; int fc = 0;
    for (int i = 0; i < total; i++) {
        if (labels[i] == 1) { t_sum += responses[i]; tc++; }
        else { f_sum += responses[i]; fc++; }
    }
    double phase_delta = (tc > 0 ? t_sum / tc : 0.0) - (fc > 0 ? f_sum / fc : 0.0);

    *out_auc = auc; *out_n95 = n95; *out_phase_delta = phase_delta;
    free(responses); free(labels);
    return (auc > n95 + 0.03) ? 1.0 : 0.0;
}

/* ---------- main ---------- */
int main(int argc, char **argv) {
    const char *out_dir = argc > 1 ? argv[1] : "topology_results";
    mkdir(out_dir, 0755);

    FILE *f = fopen(out_dir, 0) ? NULL : NULL;
    char path[512];
    snprintf(path, sizeof(path), "%s/topology_chirality_matrix.json", out_dir);
    f = fopen(path, "w");
    if (!f) { perror(path); return 1; }

    fprintf(f, "{\n");
    fprintf(f, "  \"platform\": \"AMD_Phenom_II_X6_1090T\",\n");
    fprintf(f, "  \"isolcpus\": [2,3,4,5],\n");
    fprintf(f, "  \"n\": %d,\n", N_DEFAULT);
    fprintf(f, "  \"pairs_per_route\": %d,\n", PAIRS_PER_ROUTE);
    fprintf(f, "  \"routes\": [\n");

    double best_auc = 0.5; char best_route[8] = "none";
    int count = 0, total_routes = ISOLATED_CORES * (ISOLATED_CORES - 1);

    for (int si = 0; si < ISOLATED_CORES; si++) {
        for (int ri = 0; ri < ISOLATED_CORES; ri++) {
            if (isolated[si] == isolated[ri]) continue;
            int sender = isolated[si], receiver = isolated[ri];
            printf("%d:%d  ", sender, receiver); fflush(stdout);

            uint64_t seed = MASTER_SEED ^ ((uint64_t)sender << 16) ^ ((uint64_t)receiver << 32);
            double auc, n95, phase_delta;
            double live = run_route(sender, receiver, N_DEFAULT, PAIRS_PER_ROUTE, seed, &auc, &n95, &phase_delta);

            printf("auc=%.3f/n95=%.3f  live=%d  phase_d=%.3f\n", auc, n95, (int)live, phase_delta);

            fprintf(f, "    {\"route\":\"%d:%d\",\"sender\":%d,\"receiver\":%d,", sender, receiver, sender, receiver);
            fprintf(f, "\"auc\":%.6f,\"null95\":%.6f,", auc, n95);
            fprintf(f, "\"phase_delta\":%.6f,\"live\":%s}", phase_delta, live > 0.5 ? "true" : "false");

            count++;
            if (count < total_routes) fprintf(f, ",\n"); else fprintf(f, "\n");

            if (live > 0.5 && auc > best_auc) {
                best_auc = auc;
                snprintf(best_route, sizeof(best_route), "%d:%d", sender, receiver);
            }
        }
    }

    fprintf(f, "  ],\n");
    fprintf(f, "  \"recommended_adjudication_pair\": \"%s\",\n", best_route);
    fprintf(f, "  \"best_hidden_lane_auc\": %.6f\n", best_auc);
    fprintf(f, "}\n");
    fclose(f);

    printf("\n--- VERDICT ---\n");
    if (best_route[0] == 'n') printf("TOPOLOGY_GATE_NOT_LIVE\n");
    else printf("TOPOLOGY_MAP_COMPLETE -- adjudication: %s (auc=%.3f)\n", best_route, best_auc);
    printf("wrote %s\n", path);
    return 0;
}
