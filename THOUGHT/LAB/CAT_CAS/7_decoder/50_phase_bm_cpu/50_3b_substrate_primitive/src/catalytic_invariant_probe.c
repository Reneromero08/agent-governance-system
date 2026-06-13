#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAPE_WORDS 32
#define PROBLEM_WORDS 8
#define FAMILY_COUNT 3
#define SEEDS_PER_FAMILY 8
#define PI 3.14159265358979323846

typedef struct {
    uint64_t words[TAPE_WORDS];
} tape_t;

typedef struct {
    uint64_t relation_sig;
    uint64_t parity_sig;
    uint64_t walsh_sig;
    uint64_t graph_sig;
    uint64_t corr_sig;
    uint64_t mi_sig;
    uint64_t holo_sig;
    uint64_t checksum_sig;
    double phase_mag;
} features_t;

typedef struct {
    const char *name;
    double avg_restored;
    double avg_answer_correct;
    double avg_strength_t0;
    double avg_strength_t1;
    double avg_strength_t2;
    double avg_strength_t3;
    double avg_corr;
    double avg_null_effect;
    int rows;
} summary_t;

static uint64_t rotl64(uint64_t x, unsigned k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t fnv1a64_bytes(const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
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

static uint64_t tape_hash(const tape_t *t) {
    return fnv1a64_bytes(t->words, sizeof(t->words));
}

static void init_tape(tape_t *t, int family, int seed) {
    uint64_t rng = 0x3B00000000000000ULL ^ ((uint64_t)family << 40) ^ (uint64_t)seed;
    memset(t, 0, sizeof(*t));

    for (int i = 0; i < TAPE_WORDS; i++) {
        t->words[i] = lcg(&rng);
    }

    for (int i = 0; i < PROBLEM_WORDS; i++) {
        uint64_t x = lcg(&rng);
        if (family == 0) {
            t->words[i] = (x & 0x00FF00FF00FF00FFULL) ^ (0x1111111111111111ULL * (uint64_t)(i + 1));
        } else if (family == 1) {
            t->words[i] = rotl64(x, (unsigned)(i * 7 + seed)) ^ (0xAAAAAAAAAAAAAAAAULL >> (i & 7));
        } else {
            t->words[i] = (x ^ rotl64(x, 13)) + (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 3));
        }
    }

    t->words[8] = 0x4341544341533350ULL;  /* CATCAS3P */
    t->words[9] = 2;
    t->words[10] = 0x3F8000003F800000ULL;
    t->words[11] = 0xBF8000003F800000ULL;
    t->words[12] = 0x0000006400000064ULL;
    t->words[13] = 0x0000003200000032ULL;
    t->words[14] = t->words[9] ^ t->words[10] ^ t->words[11] ^ t->words[12] ^ t->words[13];
    t->words[15] = (uint64_t)family;
    for (int i = 16; i < TAPE_WORDS; i++) {
        t->words[i] = 0;
    }
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

static uint64_t correlation_signature(const tape_t *t) {
    uint64_t sig = 0;
    for (int bit = 0; bit < 16; bit++) {
        int ones = 0;
        for (int i = 0; i < PROBLEM_WORDS; i++) {
            ones += (int)((t->words[i] >> bit) & 1ULL);
        }
        sig |= (uint64_t)(ones >= 4) << bit;
    }
    return sig;
}

static uint64_t mutual_info_signature(const tape_t *t) {
    uint64_t sig = 0;
    for (int bit = 0; bit < 8; bit++) {
        int same = 0;
        for (int i = 0; i < PROBLEM_WORDS; i++) {
            int j = (i + 1) & 7;
            same += (((t->words[i] >> bit) ^ (t->words[j] >> bit)) & 1ULL) ? 0 : 1;
        }
        sig |= (uint64_t)same << (bit * 4);
    }
    return sig;
}

static uint64_t holo_signature(const tape_t *t) {
    return t->words[9] ^ rotl64(t->words[10], 7) ^ rotl64(t->words[11], 13) ^
           rotl64(t->words[12], 19) ^ rotl64(t->words[13], 29) ^ t->words[14];
}

static double phase_magnitude(const tape_t *t) {
    double re = 0.0;
    double im = 0.0;
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        double x = (double)(int16_t)((t->words[i] >> 16) & 0xFFFF);
        double ang = 2.0 * PI * (double)i / (double)PROBLEM_WORDS;
        re += x * cos(ang);
        im += x * sin(ang);
    }
    return sqrt(re * re + im * im);
}

static features_t features(const tape_t *t) {
    features_t f;
    f.relation_sig = relation_signature(t);
    f.parity_sig = parity_signature(t);
    f.walsh_sig = walsh_signature(t);
    f.graph_sig = graph_signature(t);
    f.corr_sig = correlation_signature(t);
    f.mi_sig = mutual_info_signature(t);
    f.holo_sig = holo_signature(t);
    f.checksum_sig = tape_hash(t) ^ t->words[8] ^ t->words[14];
    f.phase_mag = phase_magnitude(t);
    return f;
}

static int expected_answer(const tape_t *t) {
    uint64_t r = relation_signature(t);
    uint64_t w = walsh_signature(t);
    uint64_t g = graph_signature(t);
    return (int)((r ^ rotl64(w, 11) ^ rotl64(g, 23)) & 1ULL);
}

static double match_strength(const features_t *a, const features_t *b) {
    int matches = 0;
    matches += a->relation_sig == b->relation_sig;
    matches += a->parity_sig == b->parity_sig;
    matches += a->walsh_sig == b->walsh_sig;
    matches += a->graph_sig == b->graph_sig;
    matches += a->corr_sig == b->corr_sig;
    matches += a->mi_sig == b->mi_sig;
    matches += a->holo_sig == b->holo_sig;
    matches += fabs(a->phase_mag - b->phase_mag) < 0.000001;
    return (double)matches / 8.0;
}

static double answer_correlation(const features_t *f, int answer) {
    int predicted = (int)((f->relation_sig ^ rotl64(f->walsh_sig, 11) ^
                           rotl64(f->graph_sig, 23)) &
                          1ULL);
    return predicted == answer ? 1.0 : 0.0;
}

static void carrier_words(const tape_t *t, uint64_t *undo) {
    uint64_t rel = relation_signature(t);
    uint64_t par = parity_signature(t);
    uint64_t wal = walsh_signature(t);
    uint64_t gra = graph_signature(t);
    undo[0] = rel ^ rotl64(par, 3);
    undo[1] = wal ^ rotl64(gra, 5);
    undo[2] = rel ^ wal ^ 0xCACA5CACA5CACA5CULL;
    undo[3] = gra ^ par ^ 0x5EED5EED5EED5EEDULL;
}

static void catalytic_forward(tape_t *t, uint64_t *undo) {
    carrier_words(t, undo);

    t->words[16] ^= undo[0];
    t->words[17] ^= undo[1];
    t->words[18] ^= undo[2];
    t->words[19] ^= undo[3];
}

static double source_strength(const tape_t *base, const tape_t *candidate) {
    features_t fb = features(base);
    features_t fc = features(candidate);
    return match_strength(&fb, &fc);
}

static double carrier_strength(const tape_t *base, const tape_t *candidate, int require_answer_slot) {
    uint64_t carrier[4];
    carrier_words(base, carrier);
    int matches = 0;
    matches += candidate->words[16] == carrier[0];
    matches += candidate->words[17] == carrier[1];
    matches += candidate->words[18] == carrier[2];
    matches += candidate->words[19] == carrier[3];
    if (require_answer_slot) {
        int ans = expected_answer(base);
        matches += candidate->words[24] == (uint64_t)ans;
        matches += candidate->words[25] == relation_signature(base);
        matches += candidate->words[26] == walsh_signature(base);
        return (double)matches / 7.0;
    }
    return (double)matches / 4.0;
}

static int catalytic_extract_answer(tape_t *t) {
    int ans = expected_answer(t);
    t->words[24] ^= (uint64_t)ans;
    t->words[25] ^= relation_signature(t);
    t->words[26] ^= walsh_signature(t);
    return ans;
}

static void catalytic_reverse(tape_t *t, const uint64_t *undo, int ans) {
    t->words[26] ^= walsh_signature(t);
    t->words[25] ^= relation_signature(t);
    t->words[24] ^= (uint64_t)ans;
    t->words[19] ^= undo[3];
    t->words[18] ^= undo[2];
    t->words[17] ^= undo[1];
    t->words[16] ^= undo[0];
}

static void destructive_write(tape_t *t) {
    for (int i = 0; i < PROBLEM_WORDS; i++) {
        t->words[i] = 0xDEADBEEFCAFEBABEULL ^ (uint64_t)i;
    }
}

static void random_reversible_write(tape_t *t, uint64_t *mask_out, uint64_t seed) {
    uint64_t rng = seed ^ 0xBAD0BEEFF00D1234ULL;
    for (int i = 16; i < 24; i++) {
        mask_out[i - 16] = lcg(&rng);
        t->words[i] ^= mask_out[i - 16];
    }
}

static void reverse_random_reversible_write(tape_t *t, const uint64_t *mask_in) {
    for (int i = 16; i < 24; i++) {
        t->words[i] ^= mask_in[i - 16];
    }
}

static void update_summary(summary_t *s, double restored, double answer_correct,
                           double st0, double st1, double st2, double st3,
                           double corr, double effect) {
    s->avg_restored += restored;
    s->avg_answer_correct += answer_correct;
    s->avg_strength_t0 += st0;
    s->avg_strength_t1 += st1;
    s->avg_strength_t2 += st2;
    s->avg_strength_t3 += st3;
    s->avg_corr += corr;
    s->avg_null_effect += effect;
    s->rows++;
}

static void emit_summary(FILE *csv, const summary_t *s) {
    double n = s->rows ? (double)s->rows : 1.0;
    fprintf(csv, "SUMMARY,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            s->name, s->rows, s->avg_restored / n, s->avg_answer_correct / n,
            s->avg_strength_t0 / n, s->avg_strength_t1 / n, s->avg_strength_t2 / n,
            s->avg_strength_t3 / n, s->avg_corr / n, s->avg_null_effect / n);
}

int main(void) {
    system("mkdir -p 50_3b_substrate_primitive/results");
    FILE *csv = fopen("50_3b_substrate_primitive/results/invariant_probe_summary.csv", "w");
    if (!csv) {
        perror("50_3b_substrate_primitive/results/invariant_probe_summary.csv");
        return 2;
    }

    fprintf(csv, "kind,case,rows,restored,answer_correct,strength_t0,strength_t1,strength_t2,strength_t3,answer_corr,null_effect\n");

    summary_t cat = {0}; cat.name = "catalytic";
    summary_t destructive = {0}; destructive.name = "destructive_write";
    summary_t random_rev = {0}; random_rev.name = "random_reversible_write";
    summary_t random_ans = {0}; random_ans.name = "random_answer";
    summary_t shuffled = {0}; shuffled.name = "shuffled_schedule";
    summary_t same_hash_wrong = {0}; same_hash_wrong.name = "same_final_hash_wrong_answer";

    int confirmed_rows = 0;
    int total_rows = 0;

    printf("=== PHASE 3B: CATALYTIC INVARIANT PROBE ===\n");
    printf("families=%d seeds_per_family=%d\n\n", FAMILY_COUNT, SEEDS_PER_FAMILY);

    for (int family = 0; family < FAMILY_COUNT; family++) {
        for (int seed = 0; seed < SEEDS_PER_FAMILY; seed++) {
            tape_t t0, t1, t2, t3;
            uint64_t undo[4];
            init_tape(&t0, family, seed);
            uint64_t h0 = tape_hash(&t0);
            int expected = expected_answer(&t0);

            t1 = t0;
            catalytic_forward(&t1, undo);

            t2 = t1;
            int answer = catalytic_extract_answer(&t2);
            features_t f2 = features(&t2);

            t3 = t2;
            catalytic_reverse(&t3, undo, answer);
            uint64_t h3 = tape_hash(&t3);

            double st0 = source_strength(&t0, &t0);
            double st1 = carrier_strength(&t0, &t1, 0);
            double st2 = carrier_strength(&t0, &t2, 1);
            double st3 = source_strength(&t0, &t3);
            double corr = answer_correlation(&f2, answer);
            double restored = h0 == h3 ? 1.0 : 0.0;
            double correct = answer == expected ? 1.0 : 0.0;

            tape_t nd = t0;
            destructive_write(&nd);
            features_t fd = features(&nd);
            double destructive_strength = source_strength(&t0, &nd);
            update_summary(&destructive, 0.0, 0.0, st0, destructive_strength,
                           destructive_strength, destructive_strength,
                           answer_correlation(&fd, answer), st2 - destructive_strength);

            tape_t nr = t0;
            uint64_t masks[8];
            random_reversible_write(&nr, masks, (uint64_t)family * 1000ULL + (uint64_t)seed);
            features_t fr1 = features(&nr);
            reverse_random_reversible_write(&nr, masks);
            double rr_st1 = carrier_strength(&t0, &nr, 0);
            double rr_st3 = source_strength(&t0, &nr);
            double rr_restored = tape_hash(&nr) == h0 ? 1.0 : 0.0;
            update_summary(&random_rev, rr_restored, 0.0, st0, rr_st1, rr_st1,
                           rr_st3, answer_correlation(&fr1, answer), st2 - rr_st1);

            int wrong_answer = answer ^ 1;
            update_summary(&random_ans, restored, wrong_answer == expected ? 1.0 : 0.0,
                           st0, st1, st2, st3, answer_correlation(&f2, wrong_answer),
                           corr - answer_correlation(&f2, wrong_answer));

            tape_t sh = t0;
            uint64_t sh_undo[4];
            carrier_words(&sh, sh_undo);
            sh.words[16] ^= sh_undo[2];
            sh.words[17] ^= sh_undo[0];
            sh.words[18] ^= sh_undo[3];
            sh.words[19] ^= sh_undo[1];
            features_t fsh = features(&sh);
            double shuffled_strength = carrier_strength(&t0, &sh, 0);
            update_summary(&shuffled, tape_hash(&sh) == h0 ? 1.0 : 0.0, 0.0,
                           st0, shuffled_strength, shuffled_strength,
                           shuffled_strength, answer_correlation(&fsh, answer),
                           st2 - shuffled_strength);

            update_summary(&same_hash_wrong, restored, wrong_answer == expected ? 1.0 : 0.0,
                           st0, st1, st2, st3, answer_correlation(&f2, wrong_answer),
                           corr - answer_correlation(&f2, wrong_answer));

            double null_max = destructive_strength;
            if (rr_st1 > null_max) null_max = rr_st1;
            double effect = st2 - null_max;
            update_summary(&cat, restored, correct, st0, st1, st2, st3, corr, effect);

            int accepted = restored && correct && st1 >= 0.75 && st2 >= 0.75 && st3 >= 0.99 &&
                           corr >= 1.0 && effect > 0.05;
            confirmed_rows += accepted ? 1 : 0;
            total_rows++;

            fprintf(csv, "CASE,F%d_S%02d,1,%.0f,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    family, seed, restored, correct, st0, st1, st2, st3, corr, effect);
        }
    }

    emit_summary(csv, &cat);
    emit_summary(csv, &destructive);
    emit_summary(csv, &random_rev);
    emit_summary(csv, &random_ans);
    emit_summary(csv, &shuffled);
    emit_summary(csv, &same_hash_wrong);
    fclose(csv);

    double rate = total_rows ? (double)confirmed_rows / (double)total_rows : 0.0;
    const char *verdict = "RESIDUAL_ARTIFACT_ONLY";
    if (rate == 1.0 && cat.avg_answer_correct / cat.rows == 1.0 && cat.avg_restored / cat.rows == 1.0 &&
        same_hash_wrong.avg_corr / same_hash_wrong.rows < 1.0) {
        verdict = "RELATIONAL_INVARIANT_CONFIRMED";
    } else if (rate >= 0.75) {
        verdict = "RELATIONAL_INVARIANT_CANDIDATE";
    } else if (cat.avg_restored / cat.rows == 1.0) {
        verdict = "RESTORE_WITHOUT_INVARIANT";
    }

    printf("Rows accepted: %d/%d\n", confirmed_rows, total_rows);
    printf("CSV: 50_3b_substrate_primitive/results/invariant_probe_summary.csv\n");
    printf("Same-final-hash wrong-answer control answer-corr: %.3f\n",
           same_hash_wrong.avg_corr / (double)same_hash_wrong.rows);
    printf("=== VERDICT: %s ===\n", verdict);

    return (strcmp(verdict, "RELATIONAL_INVARIANT_CONFIRMED") == 0 ||
            strcmp(verdict, "RELATIONAL_INVARIANT_CANDIDATE") == 0)
               ? 0
               : 1;
}
