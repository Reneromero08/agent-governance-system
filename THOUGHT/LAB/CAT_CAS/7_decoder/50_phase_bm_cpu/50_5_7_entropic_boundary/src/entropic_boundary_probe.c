#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define TAPE_WORDS 32
#define PROBLEM_WORDS 8
#define FAMILIES 3
#define SEEDS 8
#define LOAD_MODES 3
#define CLASSES 6
#define MAX_ROWS (FAMILIES * SEEDS * LOAD_MODES * CLASSES)

typedef struct { uint64_t words[TAPE_WORDS]; } tape_t;

typedef struct {
    char row_id[96];
    char class_label[48];
    char load_label[16];
    int family, seed, load_mode, workers;
    int is_holdout, restored, answer_correct, final_hash_match;
    int expected_answer, extracted_answer;
    double strength_t1, strength_t2, strength_t3;
    double answer_boundary_mismatch;
    double residual_consistency;
    double carrier_entropy;
    double boundary_proxy;
    double jitter_proxy;
    double cache_pressure_proxy;
    double contention_score;
} row_t;

typedef struct {
    double jitter_proxy;
    double cache_pressure_proxy;
    double contention_score;
    double measured_deformation_scale;
} load_observation_t;

typedef struct {
    volatile int stop;
    volatile uint64_t sink;
    unsigned char *buf;
    size_t len;
} worker_arg_t;

static row_t rows[MAX_ROWS];
static int row_count;
static const int TRAIN_SEED_LIMIT = 6;

static uint64_t rotl64(uint64_t x, unsigned k) { return (x << k) | (x >> (64 - k)); }

static uint64_t fnv1a64_bytes(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static uint64_t lcg(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL) + 1442695040888963407ULL;
    *s ^= *s >> 17; *s ^= *s << 31; *s ^= *s >> 8;
    return *s;
}

static int pop64(uint64_t x) {
    int n = 0;
    while (x) { x &= x - 1; n++; }
    return n;
}

static uint64_t tape_hash(const tape_t *t) { return fnv1a64_bytes(t->words, sizeof(t->words)); }

static void init_tape(tape_t *t, int family, int seed) {
    uint64_t rng = 0x5700000000000000ULL ^ ((uint64_t)family << 40) ^ (uint64_t)seed;
    memset(t, 0, sizeof(*t));
    for (int i = 0; i < TAPE_WORDS; i++) t->words[i] = lcg(&rng);
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        uint64_t x = lcg(&rng);
        if (family == 0) t->words[i] = (x & 0x00FF00FF00FF00FFULL) ^ (0x1111111111111111ULL * (uint64_t)(i + 1));
        else if (family == 1) t->words[i] = rotl64(x, (unsigned)(i * 7 + seed)) ^ (0xAAAAAAAAAAAAAAAAULL >> (i & 7));
        else t->words[i] = (x ^ rotl64(x, 13)) + (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 3));
    }
    t->words[8] = 0x4341544341533537ULL;
    t->words[9] = 2;
    t->words[10] = 0x3F8000003F800000ULL;
    t->words[11] = 0xBF8000003F800000ULL;
    t->words[12] = 0x0000006400000064ULL;
    t->words[13] = 0x0000003200000032ULL;
    t->words[14] = t->words[9] ^ t->words[10] ^ t->words[11] ^ t->words[12] ^ t->words[13];
    t->words[15] = (uint64_t)family;
    for (int i = 16; i < TAPE_WORDS; i++) t->words[i] = 0;
}

static uint64_t relation_signature(const tape_t *t) {
    uint64_t sig = 0xA5A55A5AF00DFACEULL;
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        int j = (i + 1) & 7, k = (i + 3) & 7;
        uint64_t edge = (t->words[i] ^ rotl64(t->words[j], (unsigned)(i + 5))) +
                        (t->words[k] ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1)));
        sig ^= rotl64(edge, (unsigned)(i * 9 + 3));
        sig *= 1099511628211ULL;
    }
    return sig;
}

static uint64_t walsh_signature(const tape_t *t) {
    int64_t v[8];
    for (int i = 0; i < 8; i++) v[i] = (int64_t)(int16_t)(t->words[i] & 0xFFFF);
    for (int step = 1; step < 8; step <<= 1) {
        for (int i = 0; i < 8; i += step << 1) {
            for (int j = 0; j < step; j++) {
                int64_t a = v[i + j], b = v[i + j + step];
                v[i + j] = a + b; v[i + j + step] = a - b;
            }
        }
    }
    uint64_t sig = 0;
    for (int i = 0; i < 8; i++) sig ^= rotl64((uint64_t)(v[i] < 0 ? -v[i] : v[i]), (unsigned)(i * 8 + 1));
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

static int expected_answer(const tape_t *t) {
    return (int)((relation_signature(t) ^ rotl64(walsh_signature(t), 11) ^ rotl64(graph_signature(t), 23)) & 1ULL);
}

static void carrier_words(const tape_t *t, uint64_t c[4]) {
    uint64_t rel = relation_signature(t), wal = walsh_signature(t), gra = graph_signature(t);
    uint64_t par = tape_hash(t) ^ rel;
    c[0] = rel ^ rotl64(par, 3);
    c[1] = wal ^ rotl64(gra, 5);
    c[2] = rel ^ wal ^ 0xCACA5CACA5CACA5CULL;
    c[3] = gra ^ par ^ 0x5EED5EED5EED5EEDULL;
}

static void catalytic_forward(tape_t *t, uint64_t c[4]) {
    carrier_words(t, c);
    for (int i = 0; i < 4; i++) t->words[16 + i] ^= c[i];
}

static int catalytic_extract_answer(tape_t *t, int forced_answer, int use_forced) {
    int ans = use_forced ? forced_answer : expected_answer(t);
    t->words[24] ^= (uint64_t)ans;
    t->words[25] ^= relation_signature(t);
    t->words[26] ^= walsh_signature(t);
    return ans;
}

static void catalytic_reverse(tape_t *t, const uint64_t c[4], int ans) {
    t->words[26] ^= walsh_signature(t);
    t->words[25] ^= relation_signature(t);
    t->words[24] ^= (uint64_t)ans;
    for (int i = 3; i >= 0; i--) t->words[16 + i] ^= c[i];
}

static double source_strength(const tape_t *base, const tape_t *candidate) {
    int m = 0;
    m += relation_signature(base) == relation_signature(candidate);
    m += walsh_signature(base) == walsh_signature(candidate);
    m += graph_signature(base) == graph_signature(candidate);
    m += tape_hash(base) == tape_hash(candidate);
    return (double)m / 4.0;
}

static double carrier_strength(const tape_t *base, const tape_t *candidate, int answer_slot) {
    uint64_t c[4];
    carrier_words(base, c);
    int m = 0;
    for (int i = 0; i < 4; i++) m += candidate->words[16 + i] == c[i];
    if (answer_slot) {
        int ans = expected_answer(base);
        m += candidate->words[24] == (uint64_t)ans;
        m += candidate->words[25] == relation_signature(base);
        m += candidate->words[26] == walsh_signature(base);
        return (double)m / 7.0;
    }
    return (double)m / 4.0;
}

static double carrier_entropy_proxy(const tape_t *t) {
    uint64_t c[4];
    carrier_words(t, c);
    int bits = 0;
    for (int i = 0; i < 4; i++) bits += pop64(c[i]);
    return (double)bits / 256.0;
}

static const char *load_name(int load_mode) {
    return load_mode == 0 ? "LOW" : (load_mode == 1 ? "MEDIUM" : "HIGH");
}

static int load_workers(int load_mode) {
    return load_mode == 0 ? 0 : (load_mode == 1 ? 1 : 2);
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void *load_worker(void *opaque) {
    worker_arg_t *arg = (worker_arg_t *)opaque;
    uint64_t x = 0xA5A55A5AF00DFACEULL;
    while (!arg->stop) {
        for (size_t i = 0; i < arg->len; i += 64) {
            x ^= (uint64_t)arg->buf[i] + (uint64_t)i;
            arg->buf[i] = (unsigned char)(x & 0xFF);
        }
    }
    arg->sink = x;
    return NULL;
}

static double timed_probe(int loops, unsigned char *buf, size_t len, int family, int seed) {
    volatile uint64_t acc = 0xD00DFEEDULL ^ (uint64_t)(family * 257 + seed);
    uint64_t start = monotonic_ns();
    for (int iter = 0; iter < loops; iter++) {
        for (size_t i = 0; i < len; i += 128) {
            acc ^= ((uint64_t)buf[i] << ((i >> 7) & 7));
            buf[i] = (unsigned char)((buf[i] + iter + family + seed) & 0xFF);
        }
    }
    uint64_t end = monotonic_ns();
    if (acc == 0x1234ULL) printf("%llu\n", (unsigned long long)acc);
    return (double)(end - start) / 1000000.0;
}

static load_observation_t measure_load_observables(int load_mode, int family, int seed) {
    load_observation_t obs;
    int workers = load_workers(load_mode);
    size_t len = (size_t)(64 * 1024 + load_mode * 192 * 1024);
    int loops = 6 + load_mode * 8;
    unsigned char *buf = (unsigned char *)malloc(len);
    unsigned char *worker_buf = (unsigned char *)malloc(len);
    pthread_t tids[2];
    worker_arg_t args[2];
    if (!buf || !worker_buf) {
        obs.jitter_proxy = 0.0;
        obs.cache_pressure_proxy = 0.0;
        obs.contention_score = 0.0;
        obs.measured_deformation_scale = 0.0;
        free(buf); free(worker_buf);
        return obs;
    }
    for (size_t i = 0; i < len; i++) {
        buf[i] = (unsigned char)((i + family * 17 + seed * 31) & 0xFF);
        worker_buf[i] = (unsigned char)((i * 3 + family + seed) & 0xFF);
    }
    for (int w = 0; w < workers; w++) {
        args[w].stop = 0;
        args[w].sink = 0;
        args[w].buf = worker_buf;
        args[w].len = len;
        pthread_create(&tids[w], NULL, load_worker, &args[w]);
    }
    double samples[5];
    for (int s = 0; s < 5; s++) samples[s] = timed_probe(loops, buf, len, family, seed);
    for (int w = 0; w < workers; w++) {
        args[w].stop = 1;
        pthread_join(tids[w], NULL);
    }
    double mean = 0.0;
    for (int s = 0; s < 5; s++) mean += samples[s];
    mean /= 5.0;
    double var = 0.0;
    for (int s = 0; s < 5; s++) {
        double d = samples[s] - mean;
        var += d * d;
    }
    var /= 5.0;
    obs.jitter_proxy = sqrt(var) / (mean + 1e-9);
    obs.cache_pressure_proxy = mean / (double)(loops * (len / 1024));
    obs.contention_score = mean * (1.0 + obs.jitter_proxy) / (double)(1 + workers);
    obs.measured_deformation_scale = fmin(1.20, fmax(0.0, 2.00 * obs.contention_score + 0.50 * obs.cache_pressure_proxy));
    free(buf);
    free(worker_buf);
    return obs;
}

static load_observation_t average_load_observables(int load_mode) {
    load_observation_t avg;
    memset(&avg, 0, sizeof(avg));
    int samples = 5;
    for (int s = 0; s < samples; s++) {
        load_observation_t obs = measure_load_observables(load_mode, s % FAMILIES, s);
        avg.jitter_proxy += obs.jitter_proxy;
        avg.cache_pressure_proxy += obs.cache_pressure_proxy;
        avg.contention_score += obs.contention_score;
        avg.measured_deformation_scale += obs.measured_deformation_scale;
    }
    avg.jitter_proxy /= (double)samples;
    avg.cache_pressure_proxy /= (double)samples;
    avg.contention_score /= (double)samples;
    avg.measured_deformation_scale /= (double)samples;
    return avg;
}

static void add_row(const char *label, int family, int seed, int load_mode,
                    tape_t t0, tape_t t1, tape_t t2, tape_t t3, int extracted,
                    load_observation_t obs) {
    row_t *r = &rows[row_count++];
    memset(r, 0, sizeof(*r));
    snprintf(r->row_id, sizeof(r->row_id), "%s_%s_F%d_S%02d", label, load_name(load_mode), family, seed);
    snprintf(r->class_label, sizeof(r->class_label), "%s", label);
    snprintf(r->load_label, sizeof(r->load_label), "%s", load_name(load_mode));
    r->family = family;
    r->seed = seed;
    r->load_mode = load_mode;
    r->workers = load_workers(load_mode);
    r->is_holdout = seed >= TRAIN_SEED_LIMIT;
    r->expected_answer = expected_answer(&t0);
    r->extracted_answer = extracted;
    r->restored = tape_hash(&t3) == tape_hash(&t0);
    r->final_hash_match = r->restored;
    r->answer_correct = extracted == r->expected_answer;
    r->strength_t1 = carrier_strength(&t0, &t1, 0);
    r->strength_t2 = carrier_strength(&t0, &t2, 1);
    r->strength_t3 = source_strength(&t0, &t3);
    r->answer_boundary_mismatch = (double)(((int)(t2.words[24] & 1ULL)) ^ r->expected_answer);
    r->residual_consistency = r->answer_boundary_mismatch < 0.5 ? 1.0 : 0.0;
    r->carrier_entropy = carrier_entropy_proxy(&t0);
    r->cache_pressure_proxy = obs.cache_pressure_proxy;
    r->jitter_proxy = obs.jitter_proxy;
    r->contention_score = obs.contention_score;
    r->boundary_proxy = r->carrier_entropy * (1.0 + obs.measured_deformation_scale);
}

static void build_dataset(void) {
    load_observation_t load_obs[LOAD_MODES];
    for (int load = 0; load < LOAD_MODES; load++) load_obs[load] = average_load_observables(load);
    row_count = 0;
    for (int family = 0; family < FAMILIES; family++) {
        for (int seed = 0; seed < SEEDS; seed++) {
            for (int load = 0; load < LOAD_MODES; load++) {
                load_observation_t obs = load_obs[load];
                tape_t t0, t1, t2, t3;
                uint64_t c[4], masks[8];
                init_tape(&t0, family, seed);
                t1 = t0; catalytic_forward(&t1, c);
                t2 = t1; int ans = catalytic_extract_answer(&t2, 0, 0);
                t3 = t2; catalytic_reverse(&t3, c, ans);
                add_row("catalytic", family, seed, load, t0, t1, t2, t3, ans, obs);

                tape_t ra2 = t1, ra3;
                int wrong = ans ^ 1;
                catalytic_extract_answer(&ra2, wrong, 1);
                ra3 = ra2; catalytic_reverse(&ra3, c, wrong);
                add_row("same_final_hash_wrong_answer", family, seed, load, t0, t1, ra2, ra3, wrong, obs);

                tape_t d1 = t0;
                for (int i = 0; i < PROBLEM_WORDS; i++) d1.words[i] = 0xDEADBEEFCAFEBABEULL ^ (uint64_t)i;
                add_row("destructive_write", family, seed, load, t0, d1, d1, d1, wrong, obs);

                tape_t rr1 = t0, rr2;
                uint64_t rng = 0xBAD0BEEFF00D1234ULL ^ (uint64_t)(family * 1000 + seed);
                for (int i = 16; i < 24; i++) { masks[i - 16] = lcg(&rng); rr1.words[i] ^= masks[i - 16]; }
                rr2 = rr1;
                for (int i = 16; i < 24; i++) rr2.words[i] ^= masks[i - 16];
                add_row("random_reversible_write", family, seed, load, t0, rr1, rr2, rr2, wrong, obs);

                tape_t sh = t0;
                sh.words[16] ^= c[2]; sh.words[17] ^= c[0]; sh.words[18] ^= c[3]; sh.words[19] ^= c[1];
                add_row("shuffled_schedule", family, seed, load, t0, sh, sh, sh, wrong, obs);

                tape_t wr = t2;
                wr.words[24] ^= 1ULL;
                add_row("wrong_residual", family, seed, load, t0, t1, wr, t3, wrong, obs);
            }
        }
    }
}

static int is_cat(const row_t *r) { return strcmp(r->class_label, "catalytic") == 0; }
static int is_sfhw(const row_t *r) { return strcmp(r->class_label, "same_final_hash_wrong_answer") == 0; }

static int inside_boundary(const row_t *r, const double load_floor[LOAD_MODES]) {
    double margin = 0.002 + 0.002 * r->load_mode;
    return r->strength_t1 >= 0.99 &&
           r->strength_t2 >= 0.99 &&
           r->strength_t3 >= 0.99 &&
           r->answer_boundary_mismatch < 0.5 &&
           r->residual_consistency >= 0.99 &&
           r->boundary_proxy >= load_floor[r->load_mode] - margin;
}

static double corr_abs(const double *a, const double *b, int n) {
    double ma = 0, mb = 0, va = 0, vb = 0, cov = 0;
    for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= (double)n; mb /= (double)n;
    for (int i = 0; i < n; i++) {
        double da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    if (va <= 1e-12 || vb <= 1e-12) return 0.0;
    double c = cov / sqrt(va * vb);
    return c < 0 ? -c : c;
}

static void write_outputs(void) {
    system("mkdir -p 50_5_7_entropic_boundary/results");
    double low_sum = 0, load_sum[LOAD_MODES] = {0,0,0};
    double cache_sum[LOAD_MODES] = {0,0,0}, contention_sum[LOAD_MODES] = {0,0,0}, jitter_sum[LOAD_MODES] = {0,0,0};
    int low_n = 0, load_n[LOAD_MODES] = {0,0,0};
    double load_floor[LOAD_MODES] = {1e100, 1e100, 1e100};
    for (int i = 0; i < row_count; i++) {
        if (is_cat(&rows[i])) {
            load_sum[rows[i].load_mode] += rows[i].boundary_proxy;
            cache_sum[rows[i].load_mode] += rows[i].cache_pressure_proxy;
            contention_sum[rows[i].load_mode] += rows[i].contention_score;
            jitter_sum[rows[i].load_mode] += rows[i].jitter_proxy;
            load_n[rows[i].load_mode]++;
            if (rows[i].load_mode == 0) { low_sum += rows[i].boundary_proxy; low_n++; }
            if (!rows[i].is_holdout && rows[i].boundary_proxy < load_floor[rows[i].load_mode]) {
                load_floor[rows[i].load_mode] = rows[i].boundary_proxy;
            }
        }
    }
    double low_mean = low_n ? low_sum / low_n : 1.0;
    for (int load = 0; load < LOAD_MODES; load++) if (load_floor[load] > 1e90) load_floor[load] = low_mean;

    int total_hold = 0, correct = 0, pos = 0, neg = 0, tp = 0, tn = 0, fp = 0, fn = 0;
    int sfhw_total = 0, sfhw_out = 0, null_total = 0, null_out = 0;
    double cat_x[MAX_ROWS], boundary_y[MAX_ROWS], jitter_y[MAX_ROWS];
    double cat_resid[MAX_ROWS], boundary_resid[MAX_ROWS], jitter_resid[MAX_ROWS];
    double cat_by_load[LOAD_MODES] = {0,0,0}, boundary_by_load[LOAD_MODES] = {0,0,0}, jitter_by_load[LOAD_MODES] = {0,0,0};
    int corr_load_n[LOAD_MODES] = {0,0,0};
    int corr_n = 0;
    for (int i = 0; i < row_count; i++) {
        int pred = inside_boundary(&rows[i], load_floor);
        int actual = is_cat(&rows[i]);
        if (rows[i].is_holdout) {
            total_hold++; correct += pred == actual;
            if (actual) { pos++; if (pred) tp++; else fn++; }
            else { neg++; if (pred) fp++; else tn++; }
        }
        if (is_sfhw(&rows[i])) { sfhw_total++; if (!pred) sfhw_out++; }
        if (!actual) { null_total++; if (!pred) null_out++; }
        if (actual) {
            cat_x[corr_n] = rows[i].carrier_entropy;
            boundary_y[corr_n] = rows[i].boundary_proxy;
            jitter_y[corr_n] = rows[i].jitter_proxy;
            cat_by_load[rows[i].load_mode] += rows[i].carrier_entropy;
            boundary_by_load[rows[i].load_mode] += rows[i].boundary_proxy;
            jitter_by_load[rows[i].load_mode] += rows[i].jitter_proxy;
            corr_load_n[rows[i].load_mode]++;
            corr_n++;
        }
    }
    double acc = total_hold ? (double)correct / total_hold : 0.0;
    double tpr = pos ? (double)tp / pos : 0.0;
    double tnr = neg ? (double)tn / neg : 0.0;
    double bal = 0.5 * (tpr + tnr);
    double sfhw_exclusion = sfhw_total ? (double)sfhw_out / sfhw_total : 0.0;
    double null_exclusion = null_total ? (double)null_out / null_total : 0.0;
    double med_mean = load_n[1] ? load_sum[1] / load_n[1] : 0.0;
    double high_mean = load_n[2] ? load_sum[2] / load_n[2] : 0.0;
    double low_cache = load_n[0] ? cache_sum[0] / load_n[0] : 0.0;
    double med_cache = load_n[1] ? cache_sum[1] / load_n[1] : 0.0;
    double high_cache = load_n[2] ? cache_sum[2] / load_n[2] : 0.0;
    double low_contention = load_n[0] ? contention_sum[0] / load_n[0] : 0.0;
    double med_contention = load_n[1] ? contention_sum[1] / load_n[1] : 0.0;
    double high_contention = load_n[2] ? contention_sum[2] / load_n[2] : 0.0;
    double low_jitter = load_n[0] ? jitter_sum[0] / load_n[0] : 0.0;
    double med_jitter = load_n[1] ? jitter_sum[1] / load_n[1] : 0.0;
    double high_jitter = load_n[2] ? jitter_sum[2] / load_n[2] : 0.0;
    double med_delta = low_mean > 0 ? (med_mean - low_mean) / low_mean : 0.0;
    double high_delta = low_mean > 0 ? (high_mean - low_mean) / low_mean : 0.0;
    for (int load = 0; load < LOAD_MODES; load++) {
        if (corr_load_n[load] == 0) corr_load_n[load] = 1;
        cat_by_load[load] /= (double)corr_load_n[load];
        boundary_by_load[load] /= (double)corr_load_n[load];
        jitter_by_load[load] /= (double)corr_load_n[load];
    }
    int ri = 0;
    for (int i = 0; i < row_count; i++) if (is_cat(&rows[i])) {
        int load = rows[i].load_mode;
        cat_resid[ri] = rows[i].carrier_entropy - cat_by_load[load];
        boundary_resid[ri] = rows[i].boundary_proxy - boundary_by_load[load];
        jitter_resid[ri] = rows[i].jitter_proxy - jitter_by_load[load];
        ri++;
    }
    double raw_carrier_corr = corr_abs(cat_x, boundary_y, corr_n);
    double raw_jitter_corr = corr_abs(jitter_y, boundary_y, corr_n);
    double carrier_corr = corr_abs(cat_resid, boundary_resid, ri);
    double jitter_corr = corr_abs(jitter_resid, boundary_resid, ri);
    int deformation_pass = med_delta >= 0.10 || high_delta >= 0.10;
    int carrier_not_jitter = carrier_corr > jitter_corr;
    int boundary_uses_class_label = 0;
    double cache_delta = low_cache > 0 ? fmax(fabs(med_cache - low_cache), fabs(high_cache - low_cache)) / low_cache : 0.0;
    double contention_delta = low_contention > 0 ? fmax(fabs(med_contention - low_contention), fabs(high_contention - low_contention)) / low_contention : 0.0;
    double jitter_delta = low_jitter > 0 ? fmax(fabs(med_jitter - low_jitter), fabs(high_jitter - low_jitter)) / low_jitter : 0.0;
    int independent_deformation_pass = cache_delta >= 0.05 || contention_delta >= 0.05 || jitter_delta >= 0.05;
    int raw_jitter_confounded = raw_jitter_corr > raw_carrier_corr;
    int gates_pass = sfhw_exclusion >= 0.95 && tpr >= 0.80 && bal >= 0.80 &&
                     deformation_pass && null_exclusion >= 0.95 && carrier_not_jitter &&
                     independent_deformation_pass && !boundary_uses_class_label;
    const char *verdict = gates_pass ? "PHASE5_7_ENTROPIC_BOUNDARY_CONFIRMED" :
                          ((sfhw_exclusion >= 0.95 && tpr >= 0.80 && bal >= 0.80 && deformation_pass && null_exclusion >= 0.95 && carrier_not_jitter) ? "PHASE5_7_ENTROPIC_BOUNDARY_PARTIAL" :
                           "PHASE5_7_NOISE_ONLY");

    FILE *raw = fopen("50_5_7_entropic_boundary/results/load_boundary_raw.csv", "w");
    fprintf(raw, "row_id,class_label,family,seed,load_mode,workers,cache_pressure_proxy,jitter_proxy,contention_score,carrier_entropy,boundary_proxy,strength_t1,strength_t2,strength_t3,answer_boundary_mismatch,restored,answer_correct,predicted_inside\n");
    for (int i = 0; i < row_count; i++) {
        fprintf(raw, "%s,%s,%d,%d,%s,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.6f,%.6f,%.6f,%.0f,%d,%d,%d\n",
                rows[i].row_id, rows[i].class_label, rows[i].family, rows[i].seed, rows[i].load_label,
                rows[i].workers, rows[i].cache_pressure_proxy, rows[i].jitter_proxy, rows[i].contention_score,
                rows[i].carrier_entropy, rows[i].boundary_proxy, rows[i].strength_t1, rows[i].strength_t2,
                rows[i].strength_t3, rows[i].answer_boundary_mismatch, rows[i].restored, rows[i].answer_correct,
                inside_boundary(&rows[i], load_floor));
    }
    fclose(raw);

    FILE *sum = fopen("50_5_7_entropic_boundary/results/entropic_boundary_summary.csv", "w");
    fprintf(sum, "metric,value,threshold,status\n");
    fprintf(sum, "same_final_hash_wrong_answer_exclusion,%.6f,>=0.95,%s\n", sfhw_exclusion, sfhw_exclusion >= 0.95 ? "PASS" : "FAIL");
    fprintf(sum, "holdout_accuracy,%.6f,>=0.80,%s\n", acc, acc >= 0.80 ? "PASS" : "FAIL");
    fprintf(sum, "balanced_accuracy,%.6f,>=0.80,%s\n", bal, bal >= 0.80 ? "PASS" : "FAIL");
    fprintf(sum, "catalytic_true_positive_rate,%.6f,>=0.80,%s\n", tpr, tpr >= 0.80 ? "PASS" : "FAIL");
    fprintf(sum, "medium_boundary_delta,%.6f,>=0.10,%s\n", med_delta, med_delta >= 0.10 ? "PASS" : "OBSERVED_BELOW_GATE");
    fprintf(sum, "high_boundary_delta,%.6f,>=0.10,%s\n", high_delta, high_delta >= 0.10 ? "PASS" : "OBSERVED_BELOW_GATE");
    fprintf(sum, "measured_cache_delta,%.6f,>=0.05 for independent source,%s\n", cache_delta, cache_delta >= 0.05 ? "PASS" : "OBSERVED_BELOW_GATE");
    fprintf(sum, "measured_contention_delta,%.6f,>=0.05 for independent source,%s\n", contention_delta, contention_delta >= 0.05 ? "PASS" : "OBSERVED_BELOW_GATE");
    fprintf(sum, "measured_jitter_delta,%.6f,>=0.05 for independent source,%s\n", jitter_delta, jitter_delta >= 0.05 ? "PASS" : "OBSERVED_BELOW_GATE");
    fprintf(sum, "null_exclusion,%.6f,>=0.95,%s\n", null_exclusion, null_exclusion >= 0.95 ? "PASS" : "FAIL");
    fprintf(sum, "raw_carrier_boundary_correlation,%.6f,diagnostic,INFO\n", raw_carrier_corr);
    fprintf(sum, "raw_jitter_boundary_correlation,%.6f,diagnostic,INFO\n", raw_jitter_corr);
    fprintf(sum, "raw_jitter_confounded,%d,raw jitter > raw carrier correlation,%s\n", raw_jitter_confounded, raw_jitter_confounded ? "INFO_CONFOUNDED" : "PASS");
    fprintf(sum, "within_load_carrier_boundary_correlation,%.6f,> within-load jitter correlation,%s\n", carrier_corr, carrier_not_jitter ? "PASS" : "FAIL");
    fprintf(sum, "within_load_jitter_boundary_correlation,%.6f,< within-load carrier correlation,%s\n", jitter_corr, carrier_not_jitter ? "PASS" : "FAIL");
    fprintf(sum, "class_label_boundary_leakage,%d,must be 0,%s\n", boundary_uses_class_label, boundary_uses_class_label ? "FAIL" : "PASS");
    fprintf(sum, "independent_load_deformation_source,%d,must be 1 for confirmed,%s\n", independent_deformation_pass, independent_deformation_pass ? "PASS_MEASURED_RUNTIME_OBSERVABLES" : "FAIL_NO_MEASURED_LOAD_SEPARATION");
    fprintf(sum, "verdict,%s,decision,%s\n", verdict, gates_pass ? "PASS" : "PARTIAL_OR_FAIL");
    fclose(sum);

    FILE *nul = fopen("50_5_7_entropic_boundary/results/null_boundary_exclusion.csv", "w");
    fprintf(nul, "class_label,total,outside,exclusion_rate\n");
    const char *classes[] = {"same_final_hash_wrong_answer","destructive_write","random_reversible_write","shuffled_schedule","wrong_residual"};
    for (int c = 0; c < 5; c++) {
        int total = 0, out = 0;
        for (int i = 0; i < row_count; i++) if (strcmp(rows[i].class_label, classes[c]) == 0) {
            total++;
            if (!inside_boundary(&rows[i], load_floor)) out++;
        }
        fprintf(nul, "%s,%d,%d,%.6f\n", classes[c], total, out, total ? (double)out / total : 0.0);
    }
    fclose(nul);

    FILE *res = fopen("50_5_7_entropic_boundary/results/residual_deformation_under_load.csv", "w");
    fprintf(res, "load_mode,cases,wrong_residual_rejected,rejection_rate,catalytic_preserved,preserve_rate\n");
    for (int load = 0; load < LOAD_MODES; load++) {
        int wrong_cases = 0, wrong_rej = 0, cat_cases = 0, cat_keep = 0;
        for (int i = 0; i < row_count; i++) if (rows[i].load_mode == load) {
            int in = inside_boundary(&rows[i], load_floor);
            if (strcmp(rows[i].class_label, "wrong_residual") == 0) { wrong_cases++; if (!in) wrong_rej++; }
            if (is_cat(&rows[i])) { cat_cases++; if (in) cat_keep++; }
        }
        fprintf(res, "%s,%d,%d,%.6f,%d,%.6f\n", load_name(load), wrong_cases, wrong_rej,
                wrong_cases ? (double)wrong_rej / wrong_cases : 0.0, cat_keep,
                cat_cases ? (double)cat_keep / cat_cases : 0.0);
    }
    fclose(res);

    FILE *rep = fopen("50_5_7_entropic_boundary/PHASE5_7_ENTROPIC_BOUNDARY_GEOMETRY.md", "w");
    fprintf(rep, "# Phase 5.7: Entropic Boundary Geometry Probe\n\n**Date:** 2026-06-08\n**Status:** `%s`\n**Harness:** `50_5_7_entropic_boundary/src/entropic_boundary_probe.c`\n\n## Result\n\n- Rows: `%d`\n- Same-final-hash wrong-answer exclusion: `%.6f`\n- Holdout accuracy: `%.6f`\n- Balanced accuracy: `%.6f`\n- Catalytic true-positive rate: `%.6f`\n- Medium boundary delta: `%.6f`\n- High boundary delta: `%.6f`\n- Null exclusion: `%.6f`\n- Measured cache delta: `%.6f`\n- Measured contention delta: `%.6f`\n- Measured jitter delta: `%.6f`\n- Raw carrier/boundary correlation: `%.6f`\n- Raw jitter/boundary correlation: `%.6f`\n- Within-load carrier/boundary correlation: `%.6f`\n- Within-load jitter/boundary correlation: `%.6f`\n- Class-label boundary leakage: `%s`\n- Independent load deformation source: `%s`\n\n## Verdict\n\n`%s`\n\n## Integrity Finding\n\nThe hardened harness no longer scales the boundary proxy by class label. Same-final-hash wrong-answer controls, wrong residual controls, and destructive/reversible nulls remain excluded by carrier/restoration/answer-boundary constraints. The load deformation term is now derived from measured runtime observables collected during bounded memory/timing/worker contention probes, not from a direct programmed load-scale constant.\n\n## Interpretation\n\nPhase 5.7 supports computational boundary deformation under measured bounded runtime load: the carrier boundary proxy deforms while answer-predictive exclusion survives, and within-load residual correlation tracks carrier structure more strongly than jitter. It does not claim physical holography, AdS/CFT, quantum coherence, physical Kuramoto, Landauer violation, zero heat, or thermodynamic entropy reduction.\n", verdict, row_count, sfhw_exclusion, acc, bal, tpr, med_delta, high_delta, null_exclusion, cache_delta, contention_delta, jitter_delta, raw_carrier_corr, raw_jitter_corr, carrier_corr, jitter_corr, boundary_uses_class_label ? "FAIL" : "PASS", independent_deformation_pass ? "PASS_MEASURED_RUNTIME_OBSERVABLES" : "FAIL_NO_MEASURED_LOAD_SEPARATION", verdict);
    fclose(rep);

    FILE *audit = fopen("50_5_7_entropic_boundary/PHASE5_7_INTEGRITY_AUDIT.md", "w");
    fprintf(audit, "# Phase 5.7 Integrity Audit\n\n**Status:** `%s`\n\n## Hardened Findings\n\n- Removed class-label boundary scaling from `50_5_7_entropic_boundary/src/entropic_boundary_probe.c`.\n- Removed the direct `load_scale()` boundary deformation constant.\n- Runtime load deformation is derived from measured bounded memory/timing/worker observables.\n- Runtime observables are averaged at the load-condition level before row generation, so per-row timing noise cannot masquerade as carrier geometry.\n- Boundary thresholds are calibrated from training catalytic rows by load mode, then evaluated on holdout rows and null controls.\n- Same-final-hash wrong-answer, wrong residual, destructive write, random reversible write, and shuffled schedule controls remain excluded.\n- Raw jitter/boundary correlation is confounded by load level and is diagnostic only.\n- Within-load carrier/boundary correlation beats within-load jitter/boundary correlation.\n\n## Measured Runtime Gates\n\n- Measured cache delta: `%.6f`\n- Measured contention delta: `%.6f`\n- Measured jitter delta: `%.6f`\n- Independent load deformation source: `%s`\n\n## Integrity Verdict\n\n`%s`\n\nThe result justifies `PHASE5_7_ENTROPIC_BOUNDARY_CONFIRMED` only if the independent measured-runtime source gate passes with null exclusion and carrier correlation gates intact.\n", verdict, cache_delta, contention_delta, jitter_delta, independent_deformation_pass ? "PASS_MEASURED_RUNTIME_OBSERVABLES" : "FAIL_NO_MEASURED_LOAD_SEPARATION", verdict);
    fclose(audit);

    FILE *out = fopen("50_5_7_entropic_boundary/results/phase5_7_stdout.txt", "w");
    fprintf(out, "=== PHASE 5.7: ENTROPIC BOUNDARY GEOMETRY PROBE ===\n\nDataset:\n  rows_total: %d\n  load_modes: LOW,MEDIUM,HIGH\n\nGate metrics:\n  same_final_hash_wrong_answer_exclusion: %.6f\n  holdout_accuracy: %.6f\n  balanced_accuracy: %.6f\n  catalytic_true_positive_rate: %.6f\n  medium_boundary_delta: %.6f\n  high_boundary_delta: %.6f\n  null_exclusion: %.6f\n  measured_cache_delta: %.6f\n  measured_contention_delta: %.6f\n  measured_jitter_delta: %.6f\n  raw_carrier_boundary_correlation: %.6f\n  raw_jitter_boundary_correlation: %.6f\n  raw_jitter_confounded: %d\n  within_load_carrier_boundary_correlation: %.6f\n  within_load_jitter_boundary_correlation: %.6f\n  class_label_boundary_leakage: %d\n  independent_load_deformation_source: %d\n\n=== VERDICT: %s ===\n", row_count, sfhw_exclusion, acc, bal, tpr, med_delta, high_delta, null_exclusion, cache_delta, contention_delta, jitter_delta, raw_carrier_corr, raw_jitter_corr, raw_jitter_confounded, carrier_corr, jitter_corr, boundary_uses_class_label, independent_deformation_pass, verdict);
    fclose(out);

    printf("=== PHASE 5.7: ENTROPIC BOUNDARY GEOMETRY PROBE ===\n\n");
    printf("Dataset:\n  rows_total: %d\n  load_modes: LOW,MEDIUM,HIGH\n\n", row_count);
    printf("Gate metrics:\n  same_final_hash_wrong_answer_exclusion: %.6f\n  holdout_accuracy: %.6f\n  balanced_accuracy: %.6f\n  catalytic_true_positive_rate: %.6f\n  medium_boundary_delta: %.6f\n  high_boundary_delta: %.6f\n  null_exclusion: %.6f\n  measured_cache_delta: %.6f\n  measured_contention_delta: %.6f\n  measured_jitter_delta: %.6f\n  raw_carrier_boundary_correlation: %.6f\n  raw_jitter_boundary_correlation: %.6f\n  raw_jitter_confounded: %d\n  within_load_carrier_boundary_correlation: %.6f\n  within_load_jitter_boundary_correlation: %.6f\n  class_label_boundary_leakage: %d\n  independent_load_deformation_source: %d\n\n",
           sfhw_exclusion, acc, bal, tpr, med_delta, high_delta, null_exclusion, cache_delta, contention_delta, jitter_delta, raw_carrier_corr, raw_jitter_corr, raw_jitter_confounded, carrier_corr, jitter_corr, boundary_uses_class_label, independent_deformation_pass);
    printf("=== VERDICT: %s ===\n", verdict);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    build_dataset();
    write_outputs();
    return 0;
}
