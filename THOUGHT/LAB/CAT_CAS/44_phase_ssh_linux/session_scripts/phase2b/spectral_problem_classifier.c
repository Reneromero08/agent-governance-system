#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_N 16
#define MAX_EDGES 160
#define PATHS 16
#define STEPS 48

typedef enum {
    SOLVER_EDGE = 0,
    SOLVER_VERTEX = 1,
    SOLVER_BLOCH = 2
} solver_t;

typedef struct {
    int n;
    int ne;
    int j[MAX_N][MAX_N];
    int edge[MAX_EDGES][3];
    const char *name;
} problem_t;

typedef struct {
    double density;
    double frustration;
    double degree_cv;
    double signed_radius;
    double abs_radius;
    double signed_abs_ratio;
} features_t;

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

static const char *solver_name(solver_t s) {
    if (s == SOLVER_EDGE) return "active_edge";
    if (s == SOLVER_VERTEX) return "vertex_phase";
    return "bloch_complex";
}

static void clear_problem(problem_t *p, const char *name, int n) {
    memset(p, 0, sizeof(*p));
    p->name = name;
    p->n = n;
}

static void add_edge(problem_t *p, int a, int b, int sign) {
    if (a == b || p->j[a][b] || p->ne >= MAX_EDGES) return;
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
            if (!j[a][b]) continue;
            int sb = bits[b] ? 1 : -1;
            e -= j[a][b] * sa * sb;
        }
    }
    return e;
}

static void decode_phase(const double *theta, int *bits, int n) {
    for (int i = 0; i < n; i++) bits[i] = cos(theta[i]) >= 0.0;
}

static void init_phase(double *theta, int n, uint64_t *rng) {
    for (int i = 0; i < n; i++) theta[i] = 2.0 * M_PI * urand(rng);
}

static int run_edge(const problem_t *p, uint64_t *rng) {
    int bits[MAX_N];
    for (int i = 0; i < p->n; i++) bits[i] = rnd_int(rng, 2);
    for (int step = 0; step < STEPS; step++) {
        int changed = 0;
        for (int e = 0; e < p->ne; e++) {
            int a = p->edge[e][0], b = p->edge[e][1], sign = p->edge[e][2];
            int same = bits[a] == bits[b];
            if ((sign > 0 && !same) || (sign < 0 && same)) {
                bits[a] ^= 1;
                changed = 1;
            }
        }
        if (!changed) break;
    }
    return energy_bits(bits, p->j, p->n);
}

static int run_vertex(const problem_t *p, uint64_t *rng) {
    double theta[MAX_N], next[MAX_N];
    int bits[MAX_N];
    init_phase(theta, p->n, rng);
    for (int step = 0; step < STEPS; step++) {
        for (int i = 0; i < p->n; i++) {
            double g = 0.0;
            for (int k = 0; k < p->n; k++) {
                if (p->j[i][k]) g += p->j[i][k] * sin(theta[i] - theta[k]);
            }
            next[i] = theta[i] - 0.12 * g;
        }
        memcpy(theta, next, sizeof(double) * p->n);
    }
    decode_phase(theta, bits, p->n);
    return energy_bits(bits, p->j, p->n);
}

static int run_bloch(const problem_t *p, uint64_t *rng) {
    double theta[MAX_N], next[MAX_N];
    int bits[MAX_N];
    init_phase(theta, p->n, rng);
    for (int step = 0; step < STEPS; step++) {
        for (int i = 0; i < p->n; i++) {
            double field = 0.0, align = 0.0;
            for (int k = 0; k < p->n; k++) {
                if (!p->j[i][k]) continue;
                field += p->j[i][k] * sin(theta[k] - theta[i]);
                align += p->j[i][k] * cos(theta[k]);
            }
            next[i] = theta[i] + 0.17 * field + 0.10 * sin(atan2(field, align + 1e-9) - theta[i]);
        }
        memcpy(theta, next, sizeof(double) * p->n);
    }
    decode_phase(theta, bits, p->n);
    return energy_bits(bits, p->j, p->n);
}

static double mean_solver(const problem_t *p, solver_t solver, uint64_t seed) {
    double sum = 0.0;
    uint64_t rng = seed;
    for (int path = 0; path < PATHS; path++) {
        if (solver == SOLVER_EDGE) sum += run_edge(p, &rng);
        else if (solver == SOLVER_VERTEX) sum += run_vertex(p, &rng);
        else sum += run_bloch(p, &rng);
    }
    return sum / (double)PATHS;
}

static double spectral_radius(const problem_t *p, int absolute_matrix) {
    double v[MAX_N], w[MAX_N];
    for (int i = 0; i < p->n; i++) v[i] = 1.0 / sqrt((double)p->n);
    double lambda = 0.0;
    for (int iter = 0; iter < 32; iter++) {
        double norm = 0.0;
        for (int i = 0; i < p->n; i++) {
            w[i] = 0.0;
            for (int k = 0; k < p->n; k++) {
                int val = p->j[i][k];
                if (absolute_matrix && val < 0) val = -val;
                w[i] += val * v[k];
            }
            norm += w[i] * w[i];
        }
        norm = sqrt(norm) + 1e-12;
        lambda = norm;
        for (int i = 0; i < p->n; i++) v[i] = w[i] / norm;
    }
    return lambda;
}

static features_t features(const problem_t *p) {
    features_t f;
    memset(&f, 0, sizeof(f));
    f.density = (2.0 * p->ne) / (double)(p->n * (p->n - 1));

    int degree[MAX_N] = {0};
    for (int e = 0; e < p->ne; e++) {
        degree[p->edge[e][0]]++;
        degree[p->edge[e][1]]++;
    }
    double mean = 0.0;
    for (int i = 0; i < p->n; i++) mean += degree[i];
    mean /= (double)p->n;
    double var = 0.0;
    for (int i = 0; i < p->n; i++) var += (degree[i] - mean) * (degree[i] - mean);
    var /= (double)p->n;
    f.degree_cv = sqrt(var) / (mean + 1e-9);

    int tri = 0, frustrated = 0;
    for (int a = 0; a < p->n; a++) {
        for (int b = a + 1; b < p->n; b++) {
            for (int c = b + 1; c < p->n; c++) {
                if (p->j[a][b] && p->j[b][c] && p->j[a][c]) {
                    tri++;
                    if (p->j[a][b] * p->j[b][c] * p->j[a][c] < 0) frustrated++;
                }
            }
        }
    }
    f.frustration = tri ? (double)frustrated / (double)tri : 0.0;
    f.signed_radius = spectral_radius(p, 0);
    f.abs_radius = spectral_radius(p, 1);
    f.signed_abs_ratio = f.signed_radius / (f.abs_radius + 1e-9);
    return f;
}

static solver_t predict(features_t f) {
    if (f.density > 0.45 && f.frustration < 0.20) return SOLVER_EDGE;
    if (f.density < 0.20 && f.degree_cv < 0.10) return SOLVER_BLOCH;
    if (f.frustration > 0.12) return SOLVER_BLOCH;
    if (f.degree_cv > 0.35) return SOLVER_BLOCH;
    if (f.signed_abs_ratio < 0.75) return SOLVER_BLOCH;
    return SOLVER_EDGE;
}

static void make_suite(problem_t *p, int *count) {
    uint64_t rng = 0x5D00000000000001ULL;
    *count = 0;

    clear_problem(&p[(*count)++], "heldout_ferro_ring_n14", 14);
    for (int i = 0; i < 14; i++) add_edge(&p[*count - 1], i, (i + 1) % 14, 1);

    clear_problem(&p[(*count)++], "heldout_odd_anti_ring_n13", 13);
    for (int i = 0; i < 13; i++) add_edge(&p[*count - 1], i, (i + 1) % 13, -1);

    clear_problem(&p[(*count)++], "heldout_sparse_mixed_n16", 16);
    while (p[*count - 1].ne < 30) {
        add_edge(&p[*count - 1], rnd_int(&rng, 16), rnd_int(&rng, 16), rnd_int(&rng, 2) ? 1 : -1);
    }

    clear_problem(&p[(*count)++], "heldout_noisy_planted_n16", 16);
    for (int a = 0; a < 16; a++) {
        for (int b = a + 1; b < 16; b++) {
            int same = (a < 8) == (b < 8);
            int sign = same ? 1 : -1;
            if (rnd_int(&rng, 17) == 0) sign = -sign;
            add_edge(&p[*count - 1], a, b, sign);
        }
    }

    clear_problem(&p[(*count)++], "heldout_chord_frustrated_n14", 14);
    for (int i = 0; i < 14; i++) add_edge(&p[*count - 1], i, (i + 1) % 14, 1);
    for (int i = 0; i < 7; i++) add_edge(&p[*count - 1], i, i + 7, -1);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    problem_t suite[8];
    int count = 0;
    make_suite(suite, &count);
    int correct = 0;
    printf("=== PHASE2B.5D SPECTRAL PROBLEM CLASSIFIER ===\n");
    printf("Held-out problems=%d paths=%d steps=%d\n\n", count, PATHS, STEPS);
    for (int i = 0; i < count; i++) {
        features_t f = features(&suite[i]);
        double edge = mean_solver(&suite[i], SOLVER_EDGE, 0xE000000000000001ULL + i);
        double vertex = mean_solver(&suite[i], SOLVER_VERTEX, 0xE100000000000001ULL + i);
        double bloch = mean_solver(&suite[i], SOLVER_BLOCH, 0xE200000000000001ULL + i);
        solver_t actual = SOLVER_EDGE;
        double best = edge;
        if (vertex < best) { best = vertex; actual = SOLVER_VERTEX; }
        if (bloch < best) { best = bloch; actual = SOLVER_BLOCH; }
        solver_t pred = predict(f);
        double pred_mean = pred == SOLVER_EDGE ? edge : (pred == SOLVER_VERTEX ? vertex : bloch);
        int accepted = pred_mean <= best + 0.001;
        if (accepted) correct++;
        printf("=== %s ===\n", suite[i].name);
        printf("features density=%.3f frustration=%.3f degree_cv=%.3f signed_abs_ratio=%.3f\n",
            f.density, f.frustration, f.degree_cv, f.signed_abs_ratio);
        printf("means edge=%+.3f vertex=%+.3f bloch=%+.3f\n", edge, vertex, bloch);
        printf("predict=%s actual_best=%s pred_mean=%+.3f result=%s\n\n",
            solver_name(pred), solver_name(actual), pred_mean, accepted ? "PASS" : "FAIL");
    }
    printf("=== VERDICT ===\n");
    printf("Classifier accuracy: %d/%d\n", correct, count);
    if (correct == count) printf("PHASE2B_5D_SPECTRAL_CLASSIFIER_PASS\n");
    else if (correct >= 3) printf("PHASE2B_5D_SPECTRAL_CLASSIFIER_PARTIAL\n");
    else printf("PHASE2B_5D_SPECTRAL_CLASSIFIER_NEGATIVE\n");
    printf("Classification: software routing aid only, not passive Kuramoto evidence.\n");
    return 0;
}
