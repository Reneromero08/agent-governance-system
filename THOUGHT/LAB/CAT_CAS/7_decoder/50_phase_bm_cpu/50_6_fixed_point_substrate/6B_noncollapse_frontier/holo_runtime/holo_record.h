/*
 * holo_record.h -- L4A .holo schema for non-collapse substrate records.
 *
 * Defines the data structures for storing OrbitState and process-object
 * records before any measurement or collapse boundary is crossed.
 *
 * FORBIDDEN FIELDS (must not exist in any .holo):
 *   hidden_d, winner, candidate_0_truth, candidate_1_truth,
 *   recovered_d, orientation_label, posthoc_selected_result,
 *   true_branch, false_branch
 *
 * These fields ARE NOT DEFINED in this header. Any attempt to use them
 * is a compile-time or architecture-level collapse violation.
 *
 * Doctrine: THE_ALGORITHM_IS_DEAD.
 */
#ifndef HOLO_RECORD_H
#define HOLO_RECORD_H

#include <stdint.h>
#include <time.h>

#define HOLO_DOCTRINE_VERSION "NON_COLLAPSE_V1"
#define HOLO_CARRIER_CLASS_B   "B_PDN_COMMON_MODE"
#define HOLO_MAX_NOTE          256
#define HOLO_MAX_HYPOTHESIS    256
#define HOLO_UUID_LEN          37

/* ======= OrbitState: structural branch coordinates only ======= */
/* branch_plus and branch_minus are structural conjugate coordinates.
 * They are NOT truth labels. They are NOT orientation labels.
 * They are NOT candidate winners.
 * branch_plus = min(a, N-a) always. branch_minus = max(a, N-a) always.
 * This is a MAGNITUDE ordering from public data, not a truth assignment. */
typedef struct {
    int branch_plus_value;   /* a = min(d, N-d) -- public magnitude */
    int branch_minus_value;  /* N-a -- public mirror */
    int N;                   /* modulus */
    char relation[16];       /* "conjugate" */
    char assignment_note[HOLO_MAX_NOTE];
} FoldPair;

/* ======= PhaseRelation: lock-in I/Q per branch window ======= */
/* Records the raw phase relation between branch walk windows.
 * No scoring. No classification. No orientation extraction. */
typedef struct {
    double q_plus;    /* Lock-in Q, branch_plus window */
    double q_minus;   /* Lock-in Q, branch_minus window */
    double i_plus;    /* Lock-in I, branch_plus window */
    double i_minus;   /* Lock-in I, branch_minus window */
} PhaseRelation;

/* ======= PathHistory: append-only walk trajectory ======= */
typedef struct {
    int steps;        /* number of walk steps */
    double tape_delta_diagnostic; /* post-restore ring-osc period (diagnostic only) */
} PathHistory;

/* ======= TapeResidue: post-restore physical state ======= */
typedef struct {
    double ring_osc_period; /* diagnostic measurement only */
    int sha_restored;       /* 1 if SHA matches */
} TapeResidue;

/* ======= SubstrateMemory: physical carrier record ======= */
typedef struct {
    char note[HOLO_MAX_NOTE]; /* "PDN/common-mode carrier. Lock-in I/Q recorded." */
} SubstrateMemory;

/* ======= Cancellation transcript ======= */
typedef struct {
    double q_common; /* (Q_plus + Q_minus)/2 -- fold-even component */
    double q_diff;   /* Q_plus - Q_minus -- predeclared residue coordinate */
    char method[32]; /* "Q_diff = Q_plus - Q_minus" */
} CancellationTranscript;

/* ======= Residue hypothesis (predeclared before measurement) ======= */
typedef struct {
    char hypothesis[HOLO_MAX_HYPOTHESIS];
    int predeclared; /* must be 1 */
} ResidueHypothesis;

/* ======= CollapseBoundary ======= */
typedef struct {
    char timestamp[32]; /* ISO timestamp */
} CollapseBoundary;

/* ======= Measurement record (only after boundary) ======= */
typedef struct {
    double q_diff_magnitude;
    int q_diff_sign;       /* -1, 0, or +1 */
    double same_orbit_q_diff;
    double dummy_orbit_q_diff;
    int label_swap_pass;   /* 1 if Q_diff sign flipped under swap */
} MeasurementRecord;

/* ======= Controls bitmask ======= */
#define CTRL_SAME_ORBIT         (1U << 0)
#define CTRL_DUMMY_ORBIT        (1U << 1)
#define CTRL_LABEL_SWAP         (1U << 2)
#define CTRL_PHASE_RANDOMIZED   (1U << 3)
#define CTRL_PATH_SHUFFLED      (1U << 4)
#define CTRL_CARRIER_OFF        (1U << 5)
#define CTRL_MEASUREMENT_ORDER  (1U << 6)
#define CTRL_WRONG_RESTORE      (1U << 7)
#define CTRL_REPLAY             (1U << 8)
#define CTRL_SESSION_REPEAT     (1U << 9)
#define CTRL_LEAKAGE_AUDIT      (1U << 10)
#define CTRL_POSTHOC_AUDIT      (1U << 11)

/* ======= Verdict labels ======= */
#define VERDICT_PROTOCOL_READY     "L4A_CLASS_B_PROTOCOL_READY_NOT_RUN"
#define VERDICT_NO_RESIDUE         "L4A_CLASS_B_RUN_CLEAN_NO_RESIDUE"
#define VERDICT_RESIDUE_CANDIDATE  "L4A_CLASS_B_RESIDUE_CANDIDATE_FOUND"
#define VERDICT_COLLAPSE           "L4A_CLASS_B_COLLAPSE_CONTAMINATION_FOUND"
#define VERDICT_NULL_FAILED        "L4A_CLASS_B_NULL_CONTROL_FAILED"
#define VERDICT_MEASUREMENT_INVALID "L4A_CLASS_B_MEASUREMENT_INVALID"
#define VERDICT_NEEDS_REDESIGN     "L4A_CLASS_B_NEEDS_REDESIGN"

/* ======= Full .holo record ======= */
/* FORBIDDEN fields (not present in this struct):
 *   hidden_d, winner, candidate_0_truth, candidate_1_truth,
 *   recovered_d, orientation_label, posthoc_selected_result.
 * If you need these, you are collapsing. Stop. */
typedef struct {
    char holo_id[HOLO_UUID_LEN];
    char doctrine_version[32];
    uint64_t run_id;
    uint64_t seed;
    int N;
    FoldPair orbit_state;
    PhaseRelation phase_relation;
    PathHistory path_history;
    TapeResidue tape_residue;
    SubstrateMemory substrate_memory;
    char carrier_class[32];
    char workload_signature[64];
    int sender_core_plus;
    int sender_core_minus;
    int receiver_core;
    CancellationTranscript cancellation_transcript;
    ResidueHypothesis residue_hypothesis;
    CollapseBoundary collapse_boundary;
    MeasurementRecord measurement_record;
    unsigned int controls;
    char verdict[64];
    int claim_level;
} HoloRecord;

/* ======= Public API ======= */

/* Initialize a blank .holo record with doctrine version and defaults. */
void holo_init(HoloRecord *h, uint64_t run_id, uint64_t seed, int N);

/* Set the OrbitState from public fold magnitudes. */
void holo_set_orbit(HoloRecord *h, int a, int Na);

/* Validate the record contains no collapse contamination.
 * Returns 1 if clean, 0 if collapse detected. */
int holo_validate_no_collapse(const HoloRecord *h);

/* Print the doctrine guard banner. */
void holo_doctrine_guard(void);

/* Write .holo record to file as JSON. Returns 0 on success. */
int holo_write_json(const HoloRecord *h, const char *path);

#endif /* HOLO_RECORD_H */
