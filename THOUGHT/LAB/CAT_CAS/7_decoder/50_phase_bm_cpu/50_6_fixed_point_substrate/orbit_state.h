/*
 * orbit_state.h -- L4B OrbitState evolution primitive.
 *
 * Represents the fold pair {a, N-a} as one unresolved process-object.
 * Coupled evolution through shared complex accumulator.
 * No candidate labels. No branch scoring. No verify(x). No AUC.
 *
 * THE ALGORITHM IS DEAD.
 */
#ifndef ORBIT_STATE_H
#define ORBIT_STATE_H

#include <stdint.h>
#include <time.h>

#define ORBIT_DOCTRINE "NON_COLLAPSE_V1"
#define ORBIT_MAX_STEPS 1024
#define ORBIT_UUID_LEN  37
#define ORBIT_NOTE_LEN  256

/* OrbitState: unresolved fold pair with coupled accumulator.
 * branch_plus = a = min(d, N-d). branch_minus = N-a.
 * These are MAGNITUDE orderings from public data only.
 * NOT truth labels. NOT candidate_0/candidate_1 for scoring. */
typedef struct {
    int N;
    int branch_plus;
    int branch_minus;
    double acc_real;    /* coupled cosine accumulation */
    double acc_imag;    /* coupled sine accumulation (should cancel) */
    int steps;          /* evolution steps completed */
    char note[ORBIT_NOTE_LEN];
} OrbitState;

/* Evolution parameters */
typedef struct {
    int max_steps;
    uint64_t seed;
} EvolParams;

/* Path history: per-step record */
typedef struct {
    int step;
    double theta_plus;
    double theta_minus;
    double cos_plus;
    double cos_minus;
    double sin_plus;
    double sin_minus;
    double delta_real;
    double delta_imag;
} PathStep;

/* CollapseBoundary: explicit measurement point */
typedef struct {
    double invariant_real;    /* accumulated cosine / steps */
    double invariant_imag;    /* accumulated sine / steps (fold-odd check) */
    int fold_symmetry_holds;  /* 1 if imag ~ 0 */
    char timestamp[32];
} CollapseBoundary;

/* Full .holo record for L4B */
typedef struct {
    char holo_id[ORBIT_UUID_LEN];
    char doctrine[32];
    uint64_t run_id;
    int N;
    /* orbit state at creation */
    int branch_plus;
    int branch_minus;
    int total_steps;
    /* evolution result */
    double acc_real_final;
    double acc_imag_final;
    /* invariant extraction */
    CollapseBoundary boundary;
    int claim_level;
} HoloL4B;

/* FORBIDDEN fields (not defined in any struct):
 *   recovered_d, winner, true_branch, false_branch,
 *   candidate_score, verify_pass, orientation_label, AUC */

/* API */
void orbit_init(OrbitState *o, int N, int a, int Na);
void orbit_evolve(OrbitState *o, const EvolParams *p, PathStep *history, int *nsteps);
void orbit_collapse(const OrbitState *o, CollapseBoundary *cb);
void holo_l4b_init(HoloL4B *h, uint64_t run_id, int N, int a, int Na);
void holo_l4b_finalize(HoloL4B *h, const OrbitState *o, const CollapseBoundary *cb);
int  holo_l4b_validate(const HoloL4B *h);
int  holo_l4b_write(const HoloL4B *h, const char *path);

#endif
