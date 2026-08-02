#ifndef FAMILY10H_POST_SOURCE_OPERATOR_RUNTIME_H
#define FAMILY10H_POST_SOURCE_OPERATOR_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define F10HI_MAX_WORD_OPERATIONS 64u

typedef enum f10hi_phase {
    F10HI_PHASE_ALLOCATED = 0,
    F10HI_PHASE_SOURCE_PREPARED = 1,
    F10HI_PHASE_SOURCE_DEAD_SEALED = 2,
    F10HI_PHASE_RECEIVER_WORD_OPEN = 3,
    F10HI_PHASE_RECEIVER_WORD_CLOSED = 4,
    F10HI_PHASE_RESTORATION_RECORDED = 5,
    F10HI_PHASE_DESTROYED = 6
} f10hi_phase;

typedef enum f10hi_operator_kind {
    F10HI_OP_REMOTE_STORE_SAME_VALUE_EXTRACTION_TARGET = 1,
    F10HI_OP_QUERY_PROBE = 2,
    F10HI_OP_DESTRUCTIVE_RESET = 3,
    F10HI_OP_UNRESOLVED_SECOND_GENERATOR = 4,
    F10HI_OP_UNQUALIFIED_INVERSE = 5
} f10hi_operator_kind;

typedef struct f10hi_operator_spec {
    f10hi_operator_kind operator_kind;
    uint64_t operator_instance_id;
    uint32_t line_set_id;
    uint32_t route_id;
    int32_t executor_core;
    uint32_t amplitude;
    uint64_t operation_budget;
    bool byte_preserving_required;
    uint64_t inverse_of_instance_id;
} f10hi_operator_spec;

typedef struct f10hi_source_death_seal {
    int64_t source_pid;
    int32_t wait_status;
    bool waitpid_exited_zero;
    bool source_alive;
    uint32_t open_source_ipc;
    uint32_t source_helper_count;
    uint64_t preparation_nonce;
    uint64_t seal_nonce;
    bool challenge_selected_before_source_death;
} f10hi_source_death_seal;

typedef struct f10hi_backend_receipt {
    uint64_t state_token_before;
    uint64_t state_token_after;
    uint64_t backend_receipt_id;
    bool byte_digest_preserved;
    bool synthetic_backend;
} f10hi_backend_receipt;

typedef struct f10hi_restoration_observation {
    uint64_t logical_byte_digest_before;
    uint64_t logical_byte_digest_after;
    uint64_t state_tomography_receipt_id;
    bool state_equivalence_passed;
    bool independent_output_retained;
} f10hi_restoration_observation;

typedef int (*f10hi_apply_backend_fn)(
    void *backend_context,
    const f10hi_operator_spec *operator_spec,
    f10hi_backend_receipt *receipt_out
);

typedef struct f10hi_runtime {
    f10hi_phase phase;
    uint64_t carrier_id;
    uint64_t preparation_nonce;
    f10hi_source_death_seal source_death;
    uint64_t challenge_nonce;
    uint64_t word_hash;
    size_t operation_count;
    bool backend_is_synthetic;
    f10hi_apply_backend_fn apply_backend;
    void *backend_context;
    f10hi_restoration_observation restoration;
} f10hi_runtime;

int f10hi_runtime_init(
    f10hi_runtime *runtime,
    uint64_t carrier_id,
    f10hi_apply_backend_fn apply_backend,
    void *backend_context,
    bool backend_is_synthetic
);
int f10hi_mark_source_prepared(f10hi_runtime *runtime, uint64_t preparation_nonce);
int f10hi_seal_source_death(
    f10hi_runtime *runtime,
    const f10hi_source_death_seal *source_death
);
int f10hi_open_receiver_word(f10hi_runtime *runtime, uint64_t challenge_nonce);
int f10hi_apply_operator(
    f10hi_runtime *runtime,
    const f10hi_operator_spec *operator_spec,
    f10hi_backend_receipt *receipt_out
);
int f10hi_close_receiver_word(f10hi_runtime *runtime);
int f10hi_record_restoration_observation(
    f10hi_runtime *runtime,
    const f10hi_restoration_observation *observation
);
int f10hi_destroy_runtime(f10hi_runtime *runtime);
int f10hi_runtime_self_test(void);

#endif
