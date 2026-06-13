#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 8
#define TAPE_WORDS 32
#define PROBLEM_WORDS 8
#define MATRICES 96
#define PI 3.14159265358979323846

typedef struct {
    uint64_t words[TAPE_WORDS];
} tape_t;

typedef struct {
    const char *name;
    double mean_r;
    double std_r;
    double target_delta;
    int count;
} result_t;

static uint64_t rotl64(uint64_t x, unsigned k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t lcg(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL) + 1442695040888963407ULL;
    *s ^= *s >> 17;
    *s ^= *s << 31;
    *s ^= *s >> 8;
    return *s;
}

static int pop64(uint64_t x) {
    int n = 0;
    while (x) {
        x &= x - 1;
        n++;
    }
    return n;
}

static double unit_from_u64(uint64_t x) {
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

static double centered_sum(uint64_t *rng) {
    double s = 0.0;
    for (int i = 0; i < 6; i++) {
        s += unit_from_u64(lcg(rng));
    }
    return s - 3.0;
}

static void init_tape(tape_t *t, int idx) {
    int family = idx % 3;
    int seed = idx / 3;
    uint64_t rng = 0x44A0000000000000ULL ^ ((uint64_t)family << 40) ^ (uint64_t)seed;
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < TAPE_WORDS; i++) {
        t->words[i] = lcg(&rng);
    }
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        uint64_t x = lcg(&rng);
        if (family == 0) {
            t->words[i] = x ^ (0x1111111111111111ULL * (uint64_t)(i + 1));
        } else if (family == 1) {
            t->words[i] = rotl64(x, (unsigned)(i * 5 + seed)) ^ 0xAAAAAAAAAAAAAAAAULL;
        } else {
            t->words[i] = (x ^ rotl64(x, 17)) + (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 2));
        }
    }
    t->words[8] = 0x4F504552474F4530ULL; /* OPERGOE0 */
    t->words[9] = 2;
    t->words[10] = 0x3F8000003F800000ULL;
    t->words[11] = 0xBF8000003F800000ULL;
    t->words[12] = 0x0000006400000064ULL;
    t->words[13] = 0x0000003200000032ULL;
    t->words[14] = t->words[9] ^ t->words[10] ^ t->words[11] ^ t->words[12] ^ t->words[13];
}

static uint64_t relation_signature(const tape_t *t) {
    uint64_t sig = 0xA5A55A5AF00DFACEULL;
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        int j = (i + 1) & 7;
        int k = (i + 3) & 7;
        uint64_t edge = (t->words[i] ^ rotl64(t->words[j], (unsigned)(i + 5))) +
                        (t->words[k] ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1)));
        sig ^= rotl64(edge, (unsigned)(i * 9 + 3));
        sig *= 1099511628211ULL;
    }
    return sig;
}

static uint64_t parity_signature(const tape_t *t) {
    uint64_t sig = 0;
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        sig ^= (uint64_t)(pop64(t->words[i]) & 1) << i;
        sig ^= rotl64(t->words[i], (unsigned)(i + 1));
    }
    return sig;
}

static uint64_t walsh_signature(const tape_t *t) {
    int64_t v[8];
    for (int i = 0; i < 8; i++) {
        v[i] = (int64_t)(int16_t)(t->words[i] & 0xFFFF);
    }
    for (int step = 1; step < 8; step <<= 1) {
        for (int i = 0; i < 8; i += step << 1) {
            for (int j = 0; j < step; j++) {
                int64_t a = v[i + j];
                int64_t b = v[i + j + step];
                v[i + j] = a + b;
                v[i + j + step] = a - b;
            }
        }
    }
    uint64_t sig = 0;
    for (int i = 0; i < 8; i++) {
        sig ^= rotl64((uint64_t)(v[i] < 0 ? -v[i] : v[i]), (unsigned)(i * 8 + 1));
    }
    return sig;
}

static uint64_t graph_signature(const tape_t *t) {
    uint64_t sig = 0x123456789ABCDEF0ULL;
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        for (int j = i + 1; j < PROBLEM_WORDS; j++) {
            uint64_t e = (t->words[i] ^ t->words[j]) & 0xFFFFULL;
            sig ^= rotl64(e * (uint64_t)(i + 1) * (uint64_t)(j + 3), (unsigned)((i + j) & 31));
        }
    }
    return sig;
}

static void residual_tags(const tape_t *t, uint8_t tags[4]) {
    uint64_t rel = relation_signature(t);
    uint64_t par = parity_signature(t);
    uint64_t wal = walsh_signature(t);
    uint64_t gra = graph_signature(t);
    int answer = (int)((rel ^ rotl64(wal, 11) ^ rotl64(gra, 23)) & 1ULL);
    tags[0] = (uint8_t)(rel & 3ULL);
    tags[1] = (uint8_t)((par ^ rotl64(wal, 7)) & 3ULL);
    tags[2] = (uint8_t)((wal ^ rotl64(gra, 11)) & 3ULL);
    tags[3] = (uint8_t)((answer ^ tags[0] ^ tags[1] ^ tags[2]) & 1U);
}

static void build_catalytic_matrix(int idx, double a[N][N]) {
    tape_t t;
    uint8_t tags[4];
    init_tape(&t, idx);
    residual_tags(&t, tags);
    uint64_t seed = relation_signature(&t) ^ rotl64(walsh_signature(&t), 19) ^
                    rotl64(graph_signature(&t), 37) ^ ((uint64_t)tags[0] << 8) ^
                    ((uint64_t)tags[1] << 16) ^ ((uint64_t)tags[2] << 24) ^
                    ((uint64_t)tags[3] << 32);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 0.0;
        }
    }
    for (int i = 0; i < N; i++) {
        a[i][i] = centered_sum(&seed) * 0.35;
        for (int j = i + 1; j < N; j++) {
            double v = centered_sum(&seed) / sqrt((double)N);
            a[i][j] = v;
            a[j][i] = v;
        }
    }
}

static void build_poisson_matrix(int idx, double a[N][N]) {
    uint64_t seed = 0x505049534F4E0000ULL ^ (uint64_t)idx;
    double x = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 0.0;
        }
    }
    for (int i = 0; i < N; i++) {
        double u = unit_from_u64(lcg(&seed));
        if (u < 1e-9) u = 1e-9;
        x += -log(u);
        a[i][i] = x;
    }
}

static void build_shuffled_matrix(int idx, double a[N][N]) {
    tape_t t;
    uint8_t tags[4];
    init_tape(&t, idx);
    residual_tags(&t, tags);
    uint64_t seed = graph_signature(&t) ^ 0x53485546464C4544ULL ^ (uint64_t)idx;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 0.0;
        }
    }
    for (int i = 0; i < N; i++) {
        a[i][i] = centered_sum(&seed);
    }
    for (int i = 0; i < N; i++) {
        int j = (i * 5 + 3) & 7;
        if (i != j) {
            double v = 0.04 * centered_sum(&seed);
            a[i][j] = v;
            a[j][i] = v;
        }
    }
}

static void jacobi_eigen(double a[N][N], double eig[N]) {
    for (int iter = 0; iter < 80; iter++) {
        int p = 0, q = 1;
        double max = fabs(a[p][q]);
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                double v = fabs(a[i][j]);
                if (v > max) {
                    max = v;
                    p = i;
                    q = j;
                }
            }
        }
        if (max < 1e-10) break;
        double app = a[p][p];
        double aqq = a[q][q];
        double apq = a[p][q];
        double phi = 0.5 * atan2(2.0 * apq, aqq - app);
        double c = cos(phi);
        double s = sin(phi);
        for (int k = 0; k < N; k++) {
            if (k != p && k != q) {
                double aik = a[k][p];
                double akq = a[k][q];
                a[k][p] = a[p][k] = c * aik - s * akq;
                a[k][q] = a[q][k] = s * aik + c * akq;
            }
        }
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = a[q][p] = 0.0;
    }
    for (int i = 0; i < N; i++) eig[i] = a[i][i];
}

static int cmp_double(const void *pa, const void *pb) {
    double a = *(const double *)pa;
    double b = *(const double *)pb;
    return (a > b) - (a < b);
}

static double spacing_ratio(double eig[N]) {
    qsort(eig, N, sizeof(double), cmp_double);
    double sum = 0.0;
    int count = 0;
    for (int i = 1; i < N - 1; i++) {
        double s0 = eig[i] - eig[i - 1];
        double s1 = eig[i + 1] - eig[i];
        if (s0 > 1e-9 && s1 > 1e-9) {
            sum += s0 < s1 ? s0 / s1 : s1 / s0;
            count++;
        }
    }
    return count ? sum / (double)count : 0.0;
}

static result_t run_family(const char *name, void (*builder)(int, double[N][N])) {
    double vals[MATRICES];
    double sum = 0.0;
    for (int m = 0; m < MATRICES; m++) {
        double a[N][N];
        double eig[N];
        builder(m, a);
        jacobi_eigen(a, eig);
        vals[m] = spacing_ratio(eig);
        sum += vals[m];
    }
    double mean = sum / (double)MATRICES;
    double var = 0.0;
    for (int m = 0; m < MATRICES; m++) {
        double d = vals[m] - mean;
        var += d * d;
    }
    result_t r;
    r.name = name;
    r.mean_r = mean;
    r.std_r = sqrt(var / (double)MATRICES);
    r.target_delta = fabs(mean - 0.5359);
    r.count = MATRICES;
    return r;
}

int main(void) {
    result_t cat = run_family("catalytic_operator", build_catalytic_matrix);
    result_t poi = run_family("poisson_diagonal_null", build_poisson_matrix);
    result_t shu = run_family("shuffled_operator_null", build_shuffled_matrix);

    printf("=== PHASE 4.4A: OPERATOR GOE VALIDATION ===\n\n");
    printf("%-24s count mean_r std_r delta_to_goe\n", "family");
    printf("%-24s %5d %.4f %.4f %.4f\n", cat.name, cat.count, cat.mean_r, cat.std_r, cat.target_delta);
    printf("%-24s %5d %.4f %.4f %.4f\n", poi.name, poi.count, poi.mean_r, poi.std_r, poi.target_delta);
    printf("%-24s %5d %.4f %.4f %.4f\n", shu.name, shu.count, shu.mean_r, shu.std_r, shu.target_delta);

    int catalytic_gate = cat.mean_r >= 0.48 && cat.mean_r <= 0.60;
    int null_separation = (cat.mean_r - poi.mean_r) > 0.08 && (cat.mean_r - shu.mean_r) > 0.08;
    int closer_to_goe = cat.target_delta < poi.target_delta && cat.target_delta < shu.target_delta;

    printf("\nGates:\n");
    printf("  catalytic in GOE-like spacing window: %s\n", catalytic_gate ? "YES" : "NO");
    printf("  separated from Poisson/shuffled nulls: %s\n", null_separation ? "YES" : "NO");
    printf("  closer to GOE target than nulls: %s\n", closer_to_goe ? "YES" : "NO");

    if (catalytic_gate && null_separation && closer_to_goe) {
        printf("=== VERDICT: PHASE4_4A_OPERATOR_GOE_PASS ===\n");
        return 0;
    }
    printf("=== VERDICT: PHASE4_4A_OPERATOR_GOE_FAIL ===\n");
    return 1;
}
