#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_N 16
#define MAX_EDGES 160
#define PATHS 24
#define STEPS 48

typedef struct {
    int n;
    int ne;
    int edge[MAX_EDGES][3];
    int j[MAX_N][MAX_N];
    const char *name;
} problem_t;

typedef struct {
    double best;
    double mean;
    int ground_hits;
} stats_t;

static uint64_t lcg(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL) + 1442695040888963407ULL;
    return *s;
}

static double urand(uint64_t *s) {
    return (lcg(s) >> 11) * (1.0 / 9007199254740992.0);
}

static int rnd_int(uint64_t *s, int n) {
    return (int)((lcg(s) >> 32) % (uint64_t)n);
}

static void clear_problem(problem_t *p, const char *name, int n) {
    memset(p, 0, sizeof(*p));
    p->name = name;
    p->n = n;
}

static void add_edge(problem_t *p, int a, int b, int sign) {
    if (a == b || p->j[a][b] || p->ne >= MAX_EDGES) {
        return;
    }
    p->j[a][b] = sign;
    p->j[b][a] = sign;
    p->edge[p->ne][0] = a;
    p->edge[p->ne][1] = b;
    p->edge[p->ne][2] = sign;
    p->ne++;
}

static int energy_bits(const int *bits, const int j[MAX_N][MAX_N], int n) {
    int e = 0;
    for (int a = 0; a < n; a++) {
        int sa = bits[a] ? 1 : -1;
        for (int b = a + 1; b < n; b++) {
            if (!j[a][b]) {
                continue;
            }
            int sb = bits[b] ? 1 : -1;
            e -= j[a][b] * sa * sb;
        }
    }
    return e;
}

static int brute_ground(const problem_t *p) {
    int best = 1000000;
    int bits[MAX_N];
    uint64_t total = 1ULL << p->n;
    for (uint64_t mask = 0; mask < total; mask++) {
        for (int i = 0; i < p->n; i++) {
            bits[i] = (mask >> i) & 1;
        }
        int e = energy_bits(bits, p->j, p->n);
        if (e < best) {
            best = e;
        }
    }
    return best;
}

static void decode_phase(const double *theta, int *bits, int n) {
    for (int i = 0; i < n; i++) {
        bits[i] = cos(theta[i]) >= 0.0;
    }
}

static void init_phase(double *theta, int n, uint64_t *rng) {
    for (int i = 0; i < n; i++) {
        theta[i] = 2.0 * M_PI * urand(rng);
    }
}

static void bloch_update(double *theta, const int j[MAX_N][MAX_N], int n, uint64_t *rng) {
    double next[MAX_N];
    for (int step = 0; step < STEPS; step++) {
        for (int i = 0; i < n; i++) {
            double field = 0.0;
            double align = 0.0;
            for (int k = 0; k < n; k++) {
                if (!j[i][k]) {
                    continue;
                }
                field += j[i][k] * sin(theta[k] - theta[i]);
                align += j[i][k] * cos(theta[k]);
            }
            double complex_bias = 0.10 * sin(atan2(field, align + 1e-9) - theta[i]);
            double anneal = 0.018 * (1.0 - (double)step / (double)STEPS);
            next[i] = theta[i] + 0.17 * field + complex_bias + anneal * (urand(rng) - 0.5);
        }
        memcpy(theta, next, sizeof(double) * n);
    }
}

static int run_bloch(const problem_t *p, uint64_t *rng, const int alt_j[MAX_N][MAX_N]) {
    double theta[MAX_N];
    int bits[MAX_N];
    init_phase(theta, p->n, rng);
    bloch_update(theta, alt_j ? alt_j : p->j, p->n, rng);
    decode_phase(theta, bits, p->n);
    return energy_bits(bits, p->j, p->n);
}

static int run_random_phase(const problem_t *p, uint64_t *rng) {
    double theta[MAX_N];
    int bits[MAX_N];
    init_phase(theta, p->n, rng);
    decode_phase(theta, bits, p->n);
    return energy_bits(bits, p->j, p->n);
}

static int run_random_spin(const problem_t *p, uint64_t *rng) {
    int bits[MAX_N];
    for (int i = 0; i < p->n; i++) {
        bits[i] = rnd_int(rng, 2);
    }
    return energy_bits(bits, p->j, p->n);
}

static int run_edge_solver(const problem_t *p, uint64_t *rng) {
    int bits[MAX_N];
    for (int i = 0; i < p->n; i++) {
        bits[i] = rnd_int(rng, 2);
    }
    for (int step = 0; step < STEPS; step++) {
        int changed = 0;
        for (int e = 0; e < p->ne; e++) {
            int a = p->edge[e][0];
            int b = p->edge[e][1];
            int sign = p->edge[e][2];
            int same = bits[a] == bits[b];
            if ((sign > 0 && !same) || (sign < 0 && same)) {
                bits[a] ^= 1;
                changed = 1;
            }
        }
        if (!changed) {
            break;
        }
    }
    return energy_bits(bits, p->j, p->n);
}

static void copy_j(int dst[MAX_N][MAX_N], const int src[MAX_N][MAX_N]) {
    memcpy(dst, src, sizeof(int) * MAX_N * MAX_N);
}

static void sign_shuffle_j(int dst[MAX_N][MAX_N], const problem_t *p, uint64_t *rng) {
    memset(dst, 0, sizeof(int) * MAX_N * MAX_N);
    for (int e = 0; e < p->ne; e++) {
        int a = p->edge[e][0];
        int b = p->edge[e][1];
        int sign = rnd_int(rng, 2) ? 1 : -1;
        dst[a][b] = sign;
        dst[b][a] = sign;
    }
}

static void edge_rewire_j(int dst[MAX_N][MAX_N], const problem_t *p, uint64_t *rng) {
    memset(dst, 0, sizeof(int) * MAX_N * MAX_N);
    for (int e = 0; e < p->ne; e++) {
        int sign = p->edge[e][2];
        for (int tries = 0; tries < 64; tries++) {
            int a = rnd_int(rng, p->n);
            int b = rnd_int(rng, p->n);
            if (a != b && !dst[a][b]) {
                dst[a][b] = sign;
                dst[b][a] = sign;
                break;
            }
        }
    }
}

static stats_t summarize(const int *values, int count, int ground) {
    stats_t st = {1000000.0, 0.0, 0};
    for (int i = 0; i < count; i++) {
        if (values[i] < st.best) {
            st.best = values[i];
        }
        if (values[i] == ground) {
            st.ground_hits++;
        }
        st.mean += values[i];
    }
    st.mean /= (double)count;
    return st;
}

static void make_suite(problem_t *p, int *count) {
    uint64_t rng = 0xB10C000000000001ULL;
    *count = 0;

    clear_problem(&p[(*count)++], "ferro_chain_n12", 12);
    for (int i = 0; i < 11; i++) {
        add_edge(&p[*count - 1], i, i + 1, 1);
    }

    clear_problem(&p[(*count)++], "anti_chain_n12", 12);
    for (int i = 0; i < 11; i++) {
        add_edge(&p[*count - 1], i, i + 1, -1);
    }

    clear_problem(&p[(*count)++], "frustrated_ring_n12", 12);
    for (int i = 0; i < 12; i++) {
        add_edge(&p[*count - 1], i, (i + 1) % 12, 1);
    }
    add_edge(&p[*count - 1], 0, 6, -1);
    add_edge(&p[*count - 1], 3, 9, -1);

    clear_problem(&p[(*count)++], "random_sparse_n16", 16);
    while (p[*count - 1].ne < 28) {
        int a = rnd_int(&rng, 16);
        int b = rnd_int(&rng, 16);
        int sign = rnd_int(&rng, 2) ? 1 : -1;
        add_edge(&p[*count - 1], a, b, sign);
    }

    clear_problem(&p[(*count)++], "planted_bipartite_n16", 16);
    for (int a = 0; a < 16; a++) {
        for (int b = a + 1; b < 16; b++) {
            int same_partition = (a < 8) == (b < 8);
            add_edge(&p[*count - 1], a, b, same_partition ? 1 : -1);
        }
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    problem_t suite[8];
    int suite_count = 0;
    make_suite(suite, &suite_count);
    printf("=== PHASE2B.5C BLOCH / COMPLEX-PLANE ISING PORT ===\n");
    printf("Paths=%d steps=%d. Active phase-coupled software oracle with nulls.\n\n", PATHS, STEPS);

    int pass_all_nulls = 0;
    for (int pi = 0; pi < suite_count; pi++) {
        const problem_t *p = &suite[pi];
        int ground = brute_ground(p);
        int e_bloch[PATHS], e_rand_phase[PATHS], e_rand_spin[PATHS];
        int e_sign_shuffle[PATHS], e_edge_rewire[PATHS], e_edge_solver[PATHS];
        uint64_t rng = 0xC0B10C0000000001ULL + (uint64_t)pi * 0x10001ULL;
        for (int path = 0; path < PATHS; path++) {
            int j_alt[MAX_N][MAX_N];
            e_bloch[path] = run_bloch(p, &rng, NULL);
            e_rand_phase[path] = run_random_phase(p, &rng);
            e_rand_spin[path] = run_random_spin(p, &rng);
            sign_shuffle_j(j_alt, p, &rng);
            e_sign_shuffle[path] = run_bloch(p, &rng, j_alt);
            edge_rewire_j(j_alt, p, &rng);
            e_edge_rewire[path] = run_bloch(p, &rng, j_alt);
            e_edge_solver[path] = run_edge_solver(p, &rng);
        }
        stats_t bloch = summarize(e_bloch, PATHS, ground);
        stats_t rand_phase = summarize(e_rand_phase, PATHS, ground);
        stats_t rand_spin = summarize(e_rand_spin, PATHS, ground);
        stats_t sign_shuffle = summarize(e_sign_shuffle, PATHS, ground);
        stats_t edge_rewire = summarize(e_edge_rewire, PATHS, ground);
        stats_t edge_solver = summarize(e_edge_solver, PATHS, ground);
        int beats_nulls = bloch.mean < rand_phase.mean && bloch.mean < rand_spin.mean &&
            bloch.mean < sign_shuffle.mean && bloch.mean < edge_rewire.mean;
        pass_all_nulls += beats_nulls;
        printf("=== %s N=%d edges=%d ground=%d ===\n", p->name, p->n, p->ne, ground);
        printf("  BlochComplex: best=%6.0f mean=%+9.3f hits=%d/%d\n", bloch.best, bloch.mean, bloch.ground_hits, PATHS);
        printf("  RandPhase:    best=%6.0f mean=%+9.3f hits=%d/%d\n", rand_phase.best, rand_phase.mean, rand_phase.ground_hits, PATHS);
        printf("  RandSpin:     best=%6.0f mean=%+9.3f hits=%d/%d\n", rand_spin.best, rand_spin.mean, rand_spin.ground_hits, PATHS);
        printf("  SignShuffle:  best=%6.0f mean=%+9.3f hits=%d/%d\n", sign_shuffle.best, sign_shuffle.mean, sign_shuffle.ground_hits, PATHS);
        printf("  EdgeRewire:   best=%6.0f mean=%+9.3f hits=%d/%d\n", edge_rewire.best, edge_rewire.mean, edge_rewire.ground_hits, PATHS);
        printf("  ActiveEdge:   best=%6.0f mean=%+9.3f hits=%d/%d\n", edge_solver.best, edge_solver.mean, edge_solver.ground_hits, PATHS);
        printf("  Null gate:    %s\n\n", beats_nulls ? "PASS" : "FAIL");
    }

    printf("=== VERDICT ===\n");
    printf("Problems passing all four null means: %d/%d\n", pass_all_nulls, suite_count);
    if (pass_all_nulls == suite_count) {
        printf("PHASE2B_5C_BLOCH_COMPLEX_ISING_ACTIVE_ORACLE_PASS\n");
    } else if (pass_all_nulls > 0) {
        printf("PHASE2B_5C_BLOCH_COMPLEX_ISING_PARTIAL\n");
    } else {
        printf("PHASE2B_5C_BLOCH_COMPLEX_ISING_NEGATIVE\n");
    }
    printf("Classification: active software phase oracle, not passive Kuramoto evidence.\n");
    return 0;
}
