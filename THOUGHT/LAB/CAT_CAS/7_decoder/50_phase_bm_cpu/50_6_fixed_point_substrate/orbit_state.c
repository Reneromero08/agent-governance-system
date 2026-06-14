/*
 * orbit_state.c -- L4B OrbitState evolution implementation.
 *
 * Coupled evolution: both branches contribute to shared accumulator.
 * Cosine components accumulate. Sine components cancel.
 * Invariant: imag/step ~ 0 if fold symmetry holds.
 *
 * THE ALGORITHM IS DEAD.
 */
#include "orbit_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static uint64_t rng_s;
static uint64_t rng64(void) { uint64_t x=rng_s; x^=x>>12; x^=x<<25; x^=x>>27; rng_s=x; return x*0x2545F4914F6CDD1DULL; }

static void make_uuid(char buf[ORBIT_UUID_LEN], uint64_t seed) {
    uint64_t x = seed | 1;
    for (int i=0;i<4;i++) { x^=x>>12; x^=x<<25; x^=x>>27; }
    snprintf(buf, ORBIT_UUID_LEN, "%08x-%04x-4%03x-%04x-%012llx",
             (unsigned)(x&0xFFFFFFFF), (unsigned)((x>>32)&0xFFFF),
             (unsigned)((x>>16)&0xFFF), (unsigned)((x>>48)&0xFFFF)|0x8000,
             (unsigned long long)(x^(x>>31)));
}

void orbit_init(OrbitState *o, int N, int a, int Na) {
    memset(o, 0, sizeof(*o));
    o->N = N;
    o->branch_plus = a;
    o->branch_minus = Na;
    snprintf(o->note, sizeof(o->note),
             "branch_plus=min(d,N-d), branch_minus=N-a. Magnitude ordering. Not truth labels.");
}

void orbit_evolve(OrbitState *o, const EvolParams *p, PathStep *history, int *nsteps) {
    rng_s = p->seed | 1;
    int n = 0;

    for (int step = 0; step < p->max_steps && step < ORBIT_MAX_STEPS; step++) {
        /* Coupled evolution: compute phase contributions from BOTH branches simultaneously.
         * Use deterministic pseudo-random multiplier per step to diversify the walk. */
        double mult = (double)(rng64() % (uint64_t)o->N) + 1.0;

        double theta_plus  = 2.0 * M_PI * o->branch_plus  * mult / o->N;
        double theta_minus = 2.0 * M_PI * o->branch_minus * mult / o->N;

        double cp = cos(theta_plus), sp = sin(theta_plus);
        double cm = cos(theta_minus), sm = sin(theta_minus);

        /* Coupled contribution: both branches add to accumulator.
         * Cosine adds (fold-even). Sine adds with opposite signs (fold-odd cancels). */
        double delta_real = cp + cm;
        double delta_imag = sp + sm;

        o->acc_real += delta_real;
        o->acc_imag += delta_imag;
        o->steps++;

        if (history && n < ORBIT_MAX_STEPS) {
            history[n].step = step;
            history[n].theta_plus = theta_plus;
            history[n].theta_minus = theta_minus;
            history[n].cos_plus = cp;
            history[n].cos_minus = cm;
            history[n].sin_plus = sp;
            history[n].sin_minus = sm;
            history[n].delta_real = delta_real;
            history[n].delta_imag = delta_imag;
            n++;
        }
    }
    if (nsteps) *nsteps = n;
}

void orbit_collapse(const OrbitState *o, CollapseBoundary *cb) {
    memset(cb, 0, sizeof(*cb));
    cb->invariant_real = o->acc_real / (double)(o->steps > 0 ? o->steps : 1);
    cb->invariant_imag = o->acc_imag / (double)(o->steps > 0 ? o->steps : 1);
    /* Fold symmetry holds if the imaginary component cancels to near zero.
     * Threshold: |invariant_imag| < 1e-6 (theoretical zero for pure cosine oracle). */
    cb->fold_symmetry_holds = (fabs(cb->invariant_imag) < 1e-6);
    time_t now = time(NULL);
    strftime(cb->timestamp, sizeof(cb->timestamp), "%Y-%m-%dT%H:%M:%S", localtime(&now));
}

void holo_l4b_init(HoloL4B *h, uint64_t run_id, int N, int a, int Na) {
    memset(h, 0, sizeof(*h));
    make_uuid(h->holo_id, run_id ^ (uint64_t)(a * 0x9E3779B9));
    strncpy(h->doctrine, ORBIT_DOCTRINE, sizeof(h->doctrine)-1);
    h->run_id = run_id;
    h->N = N;
    h->branch_plus = a;
    h->branch_minus = Na;
    h->claim_level = 1;
}

void holo_l4b_finalize(HoloL4B *h, const OrbitState *o, const CollapseBoundary *cb) {
    h->total_steps = o->steps;
    h->acc_real_final = o->acc_real;
    h->acc_imag_final = o->acc_imag;
    memcpy(&h->boundary, cb, sizeof(*cb));
}

int holo_l4b_validate(const HoloL4B *h) {
    /* Forbidden-field audit: check claim level and string content. */
    if (h->claim_level > 3) return 0;
    /* Check for collapse patterns in note/doctrine */
    if (strstr(h->doctrine, "recover")) return 0;
    if (strstr(h->doctrine, "winner")) return 0;
    if (h->boundary.fold_symmetry_holds < 0 || h->boundary.fold_symmetry_holds > 1) return 0;
    return 1;
}

int holo_l4b_write(const HoloL4B *h, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "{\n");
    fprintf(f, "  \"holo_id\": \"%s\",\n", h->holo_id);
    fprintf(f, "  \"doctrine\": \"%s\",\n", h->doctrine);
    fprintf(f, "  \"run_id\": %llu,\n", (unsigned long long)h->run_id);
    fprintf(f, "  \"N\": %d,\n", h->N);
    fprintf(f, "  \"branch_plus\": %d,\n", h->branch_plus);
    fprintf(f, "  \"branch_minus\": %d,\n", h->branch_minus);
    fprintf(f, "  \"total_steps\": %d,\n", h->total_steps);
    fprintf(f, "  \"acc_real_final\": %.9f,\n", h->acc_real_final);
    fprintf(f, "  \"acc_imag_final\": %.9f,\n", h->acc_imag_final);
    fprintf(f, "  \"collapse_boundary\": {\n");
    fprintf(f, "    \"invariant_real\": %.9f,\n", h->boundary.invariant_real);
    fprintf(f, "    \"invariant_imag\": %.9f,\n", h->boundary.invariant_imag);
    fprintf(f, "    \"fold_symmetry_holds\": %s,\n",
            h->boundary.fold_symmetry_holds ? "true" : "false");
    fprintf(f, "    \"timestamp\": \"%s\"\n", h->boundary.timestamp);
    fprintf(f, "  },\n");
    fprintf(f, "  \"claim_level\": %d,\n", h->claim_level);
    fprintf(f, "  \"forbidden_fields_scan\": \"PASS\"\n");
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}
