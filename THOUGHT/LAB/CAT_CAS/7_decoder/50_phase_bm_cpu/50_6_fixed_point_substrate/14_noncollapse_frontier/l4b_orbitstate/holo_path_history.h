#ifndef HOLO_PATH_HISTORY_H
#define HOLO_PATH_HISTORY_H

#include "orbit_state.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define HOLO_PATH_OPERATOR "orbit_coupled_phase_walk_v1"
#define HOLO_PATH_OPERATOR_LEN 40
#define HOLO_PATH_MAX_SERIALIZED_STEPS 1048576U

typedef enum {
    HOLO_PATH_OK = 0,
    HOLO_PATH_ERR_NULL = -1,
    HOLO_PATH_ERR_CAPACITY = -2,
    HOLO_PATH_ERR_ORDER = -3,
    HOLO_PATH_ERR_OPERATOR = -4,
    HOLO_PATH_ERR_NUMERIC = -5,
    HOLO_PATH_ERR_CONTINUITY = -6,
    HOLO_PATH_ERR_SEALED = -7,
    HOLO_PATH_ERR_CORRUPT = -8,
    HOLO_PATH_ERR_IO = -9,
    HOLO_PATH_ERR_PARSE = -10,
    HOLO_PATH_ERR_RESTORATION = -11
} HoloPathResult;

typedef struct {
    uint32_t step_index;
    char operator_id[HOLO_PATH_OPERATOR_LEN];
    uint64_t operator_parameter;
    uint64_t pre_acc_real_bits;
    uint64_t pre_acc_imag_bits;
    uint64_t post_acc_real_bits;
    uint64_t post_acc_imag_bits;
    uint64_t pre_state_digest;
    uint64_t post_state_digest;
    uint64_t step_digest;
} HoloPathStep;

typedef struct HoloPathHistory {
    HoloPathStep *steps;
    size_t count;
    size_t capacity;
    int appendable;
    int reversible;
    int sealed;
    int restoration_verified;
    int serialized_roundtrip;
    uint64_t initial_state_digest;
    uint64_t terminal_state_digest;
    uint64_t restored_state_digest;
} HoloPathHistory;

uint64_t holo_orbit_state_digest(const OrbitState *state);
int holo_orbit_state_equal_bitwise(const OrbitState *left, const OrbitState *right);
int holo_path_history_init(HoloPathHistory *history, size_t initial_capacity,
                           const OrbitState *initial_state);
void holo_path_history_reset(HoloPathHistory *history, const OrbitState *initial_state);
void holo_path_history_destroy(HoloPathHistory *history);
int holo_path_history_append(HoloPathHistory *history, const HoloPathStep *step);
int holo_path_history_validate(const HoloPathHistory *history);
int holo_path_history_seal(HoloPathHistory *history);
int holo_path_apply_step(HoloPathHistory *history, OrbitState *state,
                         uint64_t operator_parameter);
int holo_path_evolve(HoloPathHistory *history, OrbitState *state,
                     const EvolParams *params);
int holo_path_reverse(HoloPathHistory *history, const OrbitState *terminal_state,
                      OrbitState *restored_state);
int holo_path_history_write_json(FILE *file, const HoloPathHistory *history);
int holo_path_history_read_json(const char *json, HoloPathHistory **history_out);
int holo_path_history_read_file(const char *path, HoloPathHistory **history_out);

#endif
