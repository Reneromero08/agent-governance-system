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
#include <string.h>
#include <math.h>

static uint64_t rng_s;
static uint64_t rng64(void) { uint64_t x=rng_s; x^=x>>12; x^=x<<25; x^=x>>27; rng_s=x; return x*0x2545F4914F6CDD1DULL; }

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
