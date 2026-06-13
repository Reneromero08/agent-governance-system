#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_VARS 10
#define MAX_CLAUSES 36
#define PROBLEMS 5
#define NULL_RUNS 32

typedef struct {
    int var[3];
    int sign[3];
} clause_t;

typedef struct {
    const char *name;
    int vars;
    int clauses;
    clause_t c[MAX_CLAUSES];
} sat_t;

typedef struct {
    int sat;
    double score;
    uint32_t assignment;
} result_t;

static uint64_t lcg(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL) + 1442695040888963407ULL;
    return *s;
}

static int rnd_int(uint64_t *s, int n) {
    return (int)((lcg(s) >> 32) % (uint64_t)n);
}

static double urand(uint64_t *s) {
    return (lcg(s) >> 11) * (1.0 / 9007199254740992.0);
}

static void add_clause(sat_t *p, int a, int as, int b, int bs, int c, int cs) {
    int idx = p->clauses++;
    p->c[idx].var[0] = a; p->c[idx].sign[0] = as;
    p->c[idx].var[1] = b; p->c[idx].sign[1] = bs;
    p->c[idx].var[2] = c; p->c[idx].sign[2] = cs;
}

static int lit_true(uint32_t assignment, int var, int sign) {
    int bit = (assignment >> var) & 1U;
    return sign > 0 ? bit : !bit;
}

static int clause_sat(uint32_t assignment, const clause_t *c) {
    return lit_true(assignment, c->var[0], c->sign[0]) ||
        lit_true(assignment, c->var[1], c->sign[1]) ||
        lit_true(assignment, c->var[2], c->sign[2]);
}

static int sat_count(const sat_t *p, uint32_t assignment) {
    int count = 0;
    for (int i = 0; i < p->clauses; i++) count += clause_sat(assignment, &p->c[i]);
    return count;
}

static double optical_score(const sat_t *p, uint32_t assignment, int mode, uint64_t *rng) {
    double re = 0.0, im = 0.0;
    for (int i = 0; i < p->clauses; i++) {
        int ok = clause_sat(assignment, &p->c[i]);
        double theta;
        if (mode == 0) {
            theta = ok ? 0.0 : M_PI;
        } else if (mode == 1) {
            theta = 2.0 * M_PI * urand(rng);
        } else {
            theta = 0.0;
        }
        double weight = mode == 2 ? 1.0 : (ok ? 1.0 : 0.55);
        re += weight * cos(theta + 0.013 * (double)(i + 1));
        im += weight * sin(theta + 0.013 * (double)(i + 1));
    }
    return sqrt(re * re + im * im);
}

static result_t solve_phase(const sat_t *p, int mode, uint64_t seed) {
    result_t best = {-1, -1.0, 0};
    uint64_t rng = seed;
    uint32_t total = 1U << p->vars;
    for (uint32_t a = 0; a < total; a++) {
        double score = optical_score(p, a, mode, &rng);
        int sat = sat_count(p, a);
        if (score > best.score) {
            best.score = score;
            best.sat = sat;
            best.assignment = a;
        }
    }
    return best;
}

static int brute_best_sat(const sat_t *p) {
    int best = -1;
    uint32_t total = 1U << p->vars;
    for (uint32_t a = 0; a < total; a++) {
        int sat = sat_count(p, a);
        if (sat > best) best = sat;
    }
    return best;
}

static void make_random_problem(sat_t *p, const char *name, int vars, int clauses, uint64_t *rng) {
    memset(p, 0, sizeof(*p));
    p->name = name;
    p->vars = vars;
    while (p->clauses < clauses) {
        int a = rnd_int(rng, vars), b = rnd_int(rng, vars), c = rnd_int(rng, vars);
        if (a == b || a == c || b == c) continue;
        add_clause(p, a, rnd_int(rng, 2) ? 1 : -1,
            b, rnd_int(rng, 2) ? 1 : -1,
            c, rnd_int(rng, 2) ? 1 : -1);
    }
}

static void make_suite(sat_t *p) {
    uint64_t rng = 0x326F707469635341ULL;
    memset(p, 0, sizeof(sat_t) * PROBLEMS);
    p[0].name = "sat_chain_n6"; p[0].vars = 6;
    add_clause(&p[0], 0, 1, 1, 1, 2, -1);
    add_clause(&p[0], 1, -1, 2, 1, 3, 1);
    add_clause(&p[0], 2, 1, 3, -1, 4, 1);
    add_clause(&p[0], 3, 1, 4, -1, 5, 1);
    add_clause(&p[0], 0, -1, 4, 1, 5, -1);

    p[1].name = "xorish_n7"; p[1].vars = 7;
    for (int i = 0; i < 7; i++) {
        add_clause(&p[1], i, 1, (i + 1) % 7, -1, (i + 3) % 7, 1);
    }

    make_random_problem(&p[2], "random_n8_c18", 8, 18, &rng);
    make_random_problem(&p[3], "random_n9_c24", 9, 24, &rng);
    make_random_problem(&p[4], "random_n10_c30", 10, 30, &rng);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    sat_t suite[PROBLEMS];
    make_suite(suite);
    int pass = 0;
    printf("=== PHASE2B.5B OPTICAL 3-SAT PHASE PORT ===\n");
    printf("Problems=%d null_runs=%d\n\n", PROBLEMS, NULL_RUNS);
    for (int i = 0; i < PROBLEMS; i++) {
        const sat_t *p = &suite[i];
        int ground = brute_best_sat(p);
        result_t optical = solve_phase(p, 0, 0x5100000000000001ULL + (uint64_t)i);
        result_t ablated = solve_phase(p, 2, 0x5200000000000001ULL + (uint64_t)i);
        double random_mean = 0.0;
        int random_best = -1;
        for (int r = 0; r < NULL_RUNS; r++) {
            result_t rr = solve_phase(p, 1, 0x5300000000000001ULL + (uint64_t)i * 101ULL + (uint64_t)r);
            random_mean += rr.sat;
            if (rr.sat > random_best) random_best = rr.sat;
        }
        random_mean /= NULL_RUNS;
        int gate = optical.sat == ground && optical.sat >= ablated.sat && optical.sat >= random_best;
        if (gate) pass++;
        printf("=== %s vars=%d clauses=%d best_sat=%d ===\n", p->name, p->vars, p->clauses, ground);
        printf("  OpticalPhase: sat=%d score=%.3f assignment=0x%x\n", optical.sat, optical.score, optical.assignment);
        printf("  AblatedPhase: sat=%d score=%.3f assignment=0x%x\n", ablated.sat, ablated.score, ablated.assignment);
        printf("  RandomPhase:  best_sat=%d mean_sat=%.3f\n", random_best, random_mean);
        printf("  Null gate:    %s\n\n", gate ? "PASS" : "FAIL");
    }
    printf("=== VERDICT ===\n");
    printf("Optical phase null gates: %d/%d\n", pass, PROBLEMS);
    if (pass == PROBLEMS) printf("PHASE2B_5B_OPTICAL_3SAT_PHASE_PORT_PASS\n");
    else if (pass > 0) printf("PHASE2B_5B_OPTICAL_3SAT_PHASE_PORT_PARTIAL\n");
    else printf("PHASE2B_5B_OPTICAL_3SAT_PHASE_PORT_NEGATIVE\n");
    printf("Classification: active phase mapping, not passive Kuramoto evidence.\n");
    return 0;
}
