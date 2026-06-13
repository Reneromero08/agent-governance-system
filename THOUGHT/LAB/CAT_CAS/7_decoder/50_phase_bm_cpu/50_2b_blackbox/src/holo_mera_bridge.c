#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N 16
#define TAPE_WORDS 64
#define PATHS 24
#define STEPS 48

#define SLOT_MAGIC 0
#define SLOT_META 1
#define SLOT_ENERGY 2
#define SLOT_ORACLE_BITS 3
#define SLOT_NULL_BITS 4
#define SLOT_MERA_ROOT 5
#define SLOT_ACCUM 6
#define SLOT_BASIS0 8
#define SLOT_STATE0 16
#define SLOT_RESID0 40

typedef struct {
    int ne;
    int j[N][N];
    int edge[160][3];
} problem_t;

static uint64_t tape[TAPE_WORDS];

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

static uint64_t fnv1a64(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void add_edge(problem_t *p, int a, int b, int sign) {
    if (a == b || p->j[a][b]) return;
    p->j[a][b] = sign;
    p->j[b][a] = sign;
    p->edge[p->ne][0] = a;
    p->edge[p->ne][1] = b;
    p->edge[p->ne][2] = sign;
    p->ne++;
}

static void make_problem(problem_t *p) {
    memset(p, 0, sizeof(*p));
    uint64_t rng = 0x5EED500000000001ULL;
    while (p->ne < 30) {
        add_edge(p, rnd_int(&rng, N), rnd_int(&rng, N), rnd_int(&rng, 2) ? 1 : -1);
    }
}

static int energy_bits(const int *bits, const int j[N][N]) {
    int e = 0;
    for (int a = 0; a < N; a++) {
        int sa = bits[a] ? 1 : -1;
        for (int b = a + 1; b < N; b++) {
            if (!j[a][b]) continue;
            int sb = bits[b] ? 1 : -1;
            e -= j[a][b] * sa * sb;
        }
    }
    return e;
}

static void decode_phase(const double *theta, int *bits) {
    for (int i = 0; i < N; i++) bits[i] = cos(theta[i]) >= 0.0;
}

static uint64_t pack_bits(const int *bits) {
    uint64_t x = 0;
    for (int i = 0; i < N; i++) x |= ((uint64_t)(bits[i] & 1)) << i;
    return x;
}

static void init_phase(double *theta, uint64_t *rng) {
    for (int i = 0; i < N; i++) theta[i] = 2.0 * M_PI * urand(rng);
}

static int run_bloch(const problem_t *p, uint64_t *rng, int *bits) {
    double theta[N], next[N];
    init_phase(theta, rng);
    for (int step = 0; step < STEPS; step++) {
        for (int i = 0; i < N; i++) {
            double field = 0.0, align = 0.0;
            for (int k = 0; k < N; k++) {
                if (!p->j[i][k]) continue;
                field += p->j[i][k] * sin(theta[k] - theta[i]);
                align += p->j[i][k] * cos(theta[k]);
            }
            next[i] = theta[i] + 0.17 * field + 0.10 * sin(atan2(field, align + 1e-9) - theta[i]);
        }
        memcpy(theta, next, sizeof(theta));
    }
    decode_phase(theta, bits);
    return energy_bits(bits, p->j);
}

static int run_random_bits(const problem_t *p, uint64_t *rng, int *bits) {
    for (int i = 0; i < N; i++) bits[i] = rnd_int(rng, 2);
    return energy_bits(bits, p->j);
}

static uint64_t rotl64(uint64_t x, int r) {
    r &= 63;
    return r ? ((x << r) | (x >> (64 - r))) : x;
}

static void tape_init(const problem_t *p, uint64_t seed) {
    uint64_t rng = seed;
    for (int i = 0; i < TAPE_WORDS; i++) tape[i] = lcg(&rng);
    tape[SLOT_MAGIC] = 0x484F4C4F4D455241ULL;
    tape[SLOT_META] = ((uint64_t)N << 32) | (uint64_t)p->ne;
    for (int i = 0; i < 8; i++) {
        tape[SLOT_BASIS0 + i] = 0x9E3779B97F4A7C15ULL ^ (uint64_t)(i * 0x1010101U);
    }
}

static uint64_t mera_root_from_bits(uint64_t bits, uint64_t basis_seed) {
    uint64_t layer = bits ^ basis_seed;
    layer ^= rotl64(layer & 0x5555ULL, 7);
    layer ^= rotl64(layer & 0x3333ULL, 13);
    layer ^= rotl64(layer & 0x0F0FULL, 23);
    return layer ^ rotl64(layer, 31);
}

static void bridge_forward(uint64_t bits, int energy, uint64_t tag) {
    uint64_t root = mera_root_from_bits(bits, tape[SLOT_BASIS0]);
    uint64_t residual = (bits ^ root ^ tag) & 0xFFFFULL;
    tape[SLOT_ORACLE_BITS] ^= bits;
    tape[SLOT_ENERGY] ^= (uint64_t)(uint32_t)energy;
    tape[SLOT_MERA_ROOT] ^= root;
    tape[SLOT_RESID0] ^= residual;
    tape[SLOT_ACCUM] ^= (root ^ residual ^ tag);
    for (int i = 0; i < N; i++) {
        uint64_t bit = (bits >> i) & 1ULL;
        tape[SLOT_STATE0 + i] ^= rotl64(tape[SLOT_BASIS0 + (i & 7)] ^ tag, (int)(bit + i + 1));
    }
}

static void bridge_reverse(uint64_t bits, int energy, uint64_t tag) {
    for (int i = N - 1; i >= 0; i--) {
        uint64_t bit = (bits >> i) & 1ULL;
        tape[SLOT_STATE0 + i] ^= rotl64(tape[SLOT_BASIS0 + (i & 7)] ^ tag, (int)(bit + i + 1));
    }
    uint64_t root = mera_root_from_bits(bits, tape[SLOT_BASIS0]);
    uint64_t residual = (bits ^ root ^ tag) & 0xFFFFULL;
    tape[SLOT_ACCUM] ^= (root ^ residual ^ tag);
    tape[SLOT_RESID0] ^= residual;
    tape[SLOT_MERA_ROOT] ^= root;
    tape[SLOT_ENERGY] ^= (uint64_t)(uint32_t)energy;
    tape[SLOT_ORACLE_BITS] ^= bits;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    problem_t p;
    make_problem(&p);
    int restore_pass = 0, forward_changed = 0;
    int oracle_better = 0;
    int oracle_e[PATHS], null_e[PATHS];
    uint64_t rng = 0x484D425249444745ULL;
    printf("=== PHASE2B.5E .HOLO / MERA BRIDGE ===\n");
    printf("Problem N=%d edges=%d paths=%d steps=%d\n\n", N, p.ne, PATHS, STEPS);
    for (int path = 0; path < PATHS; path++) {
        int obits[N], nbits[N];
        tape_init(&p, 0x7000000000000001ULL + (uint64_t)path);
        uint64_t h0 = fnv1a64(tape, sizeof(tape));
        int oe = run_bloch(&p, &rng, obits);
        int ne = run_random_bits(&p, &rng, nbits);
        uint64_t opack = pack_bits(obits);
        uint64_t npack = pack_bits(nbits);
        uint64_t tag = 0xB10C000000000000ULL ^ (uint64_t)path;
        bridge_forward(opack, oe, tag);
        uint64_t hf = fnv1a64(tape, sizeof(tape));
        bridge_reverse(opack, oe, tag);
        uint64_t hr = fnv1a64(tape, sizeof(tape));
        oracle_e[path] = oe;
        null_e[path] = ne;
        if (hf != h0) forward_changed++;
        if (hr == h0) restore_pass++;
        if (oe < ne) oracle_better++;
        tape[SLOT_NULL_BITS] ^= npack;
        (void)fnv1a64(tape, sizeof(tape));
    }
    double osum = 0.0, nsum = 0.0;
    int obest = 1000000, nbest = 1000000;
    for (int i = 0; i < PATHS; i++) {
        osum += oracle_e[i];
        nsum += null_e[i];
        if (oracle_e[i] < obest) obest = oracle_e[i];
        if (null_e[i] < nbest) nbest = null_e[i];
    }
    printf("Oracle best=%d mean=%+.3f\n", obest, osum / PATHS);
    printf("Null   best=%d mean=%+.3f\n", nbest, nsum / PATHS);
    printf("Oracle beats paired null: %d/%d\n", oracle_better, PATHS);
    printf("Forward changed tape: %d/%d\n", forward_changed, PATHS);
    printf("Reverse restored tape: %d/%d\n", restore_pass, PATHS);
    printf("\n=== VERDICT ===\n");
    if (restore_pass == PATHS && forward_changed == PATHS && osum < nsum && oracle_better >= 18) {
        printf("PHASE2B_5E_HOLO_MERA_BRIDGE_PASS\n");
    } else if (restore_pass == PATHS && forward_changed == PATHS && osum < nsum) {
        printf("PHASE2B_5E_HOLO_MERA_BRIDGE_PARTIAL\n");
    } else {
        printf("PHASE2B_5E_HOLO_MERA_BRIDGE_NEGATIVE\n");
    }
    printf("Classification: active phase-oracle to .holo tape bridge, not passive Kuramoto evidence.\n");
    return 0;
}
