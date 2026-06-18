#ifndef HOLO_INVARIANT_FAMILY_H
#define HOLO_INVARIANT_FAMILY_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define HOLO_INVARIANT_COUNT 8U
#define HOLO_INVARIANT_TEXT_LEN 80

typedef enum {
    HOLO_INV_ORBIT_CONSERVATION = 0,
    HOLO_INV_RELATION_BASIS = 1,
    HOLO_INV_PATH_COMPOSITION = 2,
    HOLO_INV_RESTORATION_CLOSURE = 3,
    HOLO_INV_EXCHANGE_COVARIANCE = 4,
    HOLO_INV_SERIALIZATION = 5,
    HOLO_INV_PATH_ORDER = 6,
    HOLO_INV_SOFTWARE_HOLONOMY = 7
} HoloInvariantKind;

typedef enum {
    HOLO_INV_RESULT_PENDING = 0,
    HOLO_INV_RESULT_PASS = 1,
    HOLO_INV_RESULT_FAIL = 2,
    HOLO_INV_RESULT_DEFERRED_NOT_WELL_DEFINED = 3
} HoloInvariantResult;

typedef struct {
    HoloInvariantKind kind;
    char invariant_id[HOLO_INVARIANT_TEXT_LEN];
    char operator_id[HOLO_INVARIANT_TEXT_LEN];
    char declaration_phase[HOLO_INVARIANT_TEXT_LEN];
    char equality_rule[HOLO_INVARIANT_TEXT_LEN];
    char evidence_level[HOLO_INVARIANT_TEXT_LEN];
    int predeclared;
    int evaluated;
    int passed;
    int claim_level;
    double tolerance;
    HoloInvariantResult result;
    uint64_t digest_a;
    uint64_t digest_b;
    uint64_t digest_c;
    double scalar_a;
    double scalar_b;
    double scalar_c;
    int flag_a;
    int flag_b;
    int flag_c;
    int flag_d;
} HoloInvariantRecord;

typedef struct {
    char family_id[HOLO_INVARIANT_TEXT_LEN];
    HoloInvariantRecord records[HOLO_INVARIANT_COUNT];
    size_t count;
    int predeclared;
    int extracted;
    int sealed;
    int claim_level;
    uint64_t path_history_digest;
    uint64_t family_digest;
} HoloInvariantFamily;

struct HoloObject;

const char *holo_invariant_kind_name(HoloInvariantKind kind);
const char *holo_invariant_result_name(HoloInvariantResult result);
int holo_invariant_family_init(HoloInvariantFamily *family);
int holo_invariant_family_register(HoloInvariantFamily *family, HoloInvariantKind kind);
int holo_invariant_family_set_tolerance(HoloInvariantFamily *family,
                                        HoloInvariantKind kind, double tolerance);
int holo_invariant_family_set_operator(HoloInvariantFamily *family,
                                       HoloInvariantKind kind, const char *operator_id);
int holo_invariant_family_set_result(HoloInvariantFamily *family,
                                     HoloInvariantKind kind, HoloInvariantResult result);
int holo_invariant_family_evaluate(struct HoloObject *object);
int holo_invariant_family_mark_serialization(HoloInvariantFamily *family,
                                             uint64_t reloaded_digest,
                                             uint64_t recomputed_digest);
int holo_invariant_family_validate(const HoloInvariantFamily *family);
int holo_invariant_family_equal(const HoloInvariantFamily *left,
                                const HoloInvariantFamily *right,
                                int ignore_serialization_state);
int holo_invariant_family_write_json(FILE *file, const HoloInvariantFamily *family);
int holo_invariant_family_read_json(const char *json, HoloInvariantFamily *family);

#endif
