#include "post_source_operator_runtime.h"

#include <string.h>

static uint64_t f10hi_mix(uint64_t state, uint64_t value) {
    state ^= value + UINT64_C(0x9e3779b97f4a7c15) + (state << 6) + (state >> 2);
    return state;
}

static int f10hi_operator_is_admissible_extraction_target(
    const f10hi_operator_spec *operator_spec
) {
    if (operator_spec == NULL) {
        return 0;
    }
    if (
        operator_spec->operator_kind
        != F10HI_OP_REMOTE_STORE_SAME_VALUE_EXTRACTION_TARGET
    ) {
        return 0;
    }
    if (
        operator_spec->operator_instance_id == 0u
        || operator_spec->amplitude == 0u
        || operator_spec->operation_budget == 0u
        || !operator_spec->byte_preserving_required
        || operator_spec->inverse_of_instance_id != 0u
    ) {
        return 0;
    }
    return 1;
}

int f10hi_runtime_init(
    f10hi_runtime *runtime,
    uint64_t carrier_id,
    f10hi_apply_backend_fn apply_backend,
    void *backend_context,
    bool backend_is_synthetic
) {
    if (runtime == NULL || carrier_id == 0u || apply_backend == NULL) {
        return 0;
    }
    memset(runtime, 0, sizeof(*runtime));
    runtime->phase = F10HI_PHASE_ALLOCATED;
    runtime->carrier_id = carrier_id;
    runtime->apply_backend = apply_backend;
    runtime->backend_context = backend_context;
    runtime->backend_is_synthetic = backend_is_synthetic;
    return 1;
}

int f10hi_mark_source_prepared(f10hi_runtime *runtime, uint64_t preparation_nonce) {
    if (
        runtime == NULL
        || runtime->phase != F10HI_PHASE_ALLOCATED
        || preparation_nonce == 0u
    ) {
        return 0;
    }
    runtime->preparation_nonce = preparation_nonce;
    runtime->phase = F10HI_PHASE_SOURCE_PREPARED;
    return 1;
}

int f10hi_seal_source_death(
    f10hi_runtime *runtime,
    const f10hi_source_death_seal *source_death
) {
    if (
        runtime == NULL
        || source_death == NULL
        || runtime->phase != F10HI_PHASE_SOURCE_PREPARED
    ) {
        return 0;
    }
    if (
        source_death->source_pid <= 0
        || !source_death->waitpid_exited_zero
        || source_death->source_alive
        || source_death->open_source_ipc != 0u
        || source_death->source_helper_count != 0u
        || source_death->preparation_nonce != runtime->preparation_nonce
        || source_death->seal_nonce == 0u
        || source_death->seal_nonce == runtime->preparation_nonce
        || source_death->challenge_selected_before_source_death
    ) {
        return 0;
    }
    runtime->source_death = *source_death;
    runtime->phase = F10HI_PHASE_SOURCE_DEAD_SEALED;
    return 1;
}

int f10hi_open_receiver_word(f10hi_runtime *runtime, uint64_t challenge_nonce) {
    if (
        runtime == NULL
        || runtime->phase != F10HI_PHASE_SOURCE_DEAD_SEALED
        || challenge_nonce == 0u
        || challenge_nonce == runtime->preparation_nonce
        || challenge_nonce == runtime->source_death.seal_nonce
    ) {
        return 0;
    }
    runtime->challenge_nonce = challenge_nonce;
    runtime->word_hash = f10hi_mix(runtime->carrier_id, challenge_nonce);
    runtime->operation_count = 0u;
    runtime->phase = F10HI_PHASE_RECEIVER_WORD_OPEN;
    return 1;
}

int f10hi_apply_operator(
    f10hi_runtime *runtime,
    const f10hi_operator_spec *operator_spec,
    f10hi_backend_receipt *receipt_out
) {
    f10hi_backend_receipt receipt;
    if (
        runtime == NULL
        || receipt_out == NULL
        || runtime->phase != F10HI_PHASE_RECEIVER_WORD_OPEN
        || runtime->operation_count >= F10HI_MAX_WORD_OPERATIONS
        || !f10hi_operator_is_admissible_extraction_target(operator_spec)
    ) {
        return 0;
    }
    memset(&receipt, 0, sizeof(receipt));
    if (!runtime->apply_backend(runtime->backend_context, operator_spec, &receipt)) {
        return 0;
    }
    if (
        receipt.backend_receipt_id == 0u
        || receipt.synthetic_backend != runtime->backend_is_synthetic
        || (operator_spec->byte_preserving_required && !receipt.byte_digest_preserved)
    ) {
        return 0;
    }
    runtime->word_hash = f10hi_mix(
        runtime->word_hash,
        (uint64_t)operator_spec->operator_kind
    );
    runtime->word_hash = f10hi_mix(
        runtime->word_hash,
        operator_spec->operator_instance_id
    );
    runtime->word_hash = f10hi_mix(runtime->word_hash, operator_spec->line_set_id);
    runtime->word_hash = f10hi_mix(runtime->word_hash, operator_spec->route_id);
    runtime->word_hash = f10hi_mix(runtime->word_hash, operator_spec->amplitude);
    runtime->word_hash = f10hi_mix(
        runtime->word_hash,
        operator_spec->operation_budget
    );
    runtime->word_hash = f10hi_mix(
        runtime->word_hash,
        receipt.backend_receipt_id
    );
    ++runtime->operation_count;
    *receipt_out = receipt;
    return 1;
}

int f10hi_close_receiver_word(f10hi_runtime *runtime) {
    if (
        runtime == NULL
        || runtime->phase != F10HI_PHASE_RECEIVER_WORD_OPEN
        || runtime->operation_count == 0u
    ) {
        return 0;
    }
    runtime->phase = F10HI_PHASE_RECEIVER_WORD_CLOSED;
    return 1;
}

int f10hi_record_restoration_observation(
    f10hi_runtime *runtime,
    const f10hi_restoration_observation *observation
) {
    if (
        runtime == NULL
        || observation == NULL
        || runtime->phase != F10HI_PHASE_RECEIVER_WORD_CLOSED
        || observation->state_tomography_receipt_id == 0u
    ) {
        return 0;
    }
    runtime->restoration = *observation;
    runtime->phase = F10HI_PHASE_RESTORATION_RECORDED;
    return 1;
}

int f10hi_destroy_runtime(f10hi_runtime *runtime) {
    if (
        runtime == NULL
        || runtime->phase != F10HI_PHASE_RESTORATION_RECORDED
    ) {
        return 0;
    }
    memset(runtime, 0, sizeof(*runtime));
    runtime->phase = F10HI_PHASE_DESTROYED;
    return 1;
}

struct f10hi_synthetic_backend {
    uint64_t state_token;
    uint64_t receipt_counter;
};

static int f10hi_synthetic_apply(
    void *backend_context,
    const f10hi_operator_spec *operator_spec,
    f10hi_backend_receipt *receipt_out
) {
    struct f10hi_synthetic_backend *backend =
        (struct f10hi_synthetic_backend *)backend_context;
    uint64_t before = 0u;
    uint64_t after = 0u;
    if (
        backend == NULL
        || operator_spec == NULL
        || receipt_out == NULL
        || operator_spec->operator_kind
            != F10HI_OP_REMOTE_STORE_SAME_VALUE_EXTRACTION_TARGET
    ) {
        return 0;
    }
    before = backend->state_token;
    after = f10hi_mix(before, operator_spec->operator_instance_id);
    after = f10hi_mix(after, operator_spec->line_set_id);
    after = f10hi_mix(after, operator_spec->route_id);
    after = f10hi_mix(after, operator_spec->amplitude);
    after = f10hi_mix(after, operator_spec->operation_budget);
    backend->state_token = after;
    ++backend->receipt_counter;
    receipt_out->state_token_before = before;
    receipt_out->state_token_after = after;
    receipt_out->backend_receipt_id = backend->receipt_counter;
    receipt_out->byte_digest_preserved = true;
    receipt_out->synthetic_backend = true;
    return 1;
}

int f10hi_runtime_self_test(void) {
    f10hi_runtime runtime;
    struct f10hi_synthetic_backend backend;
    f10hi_source_death_seal bad_death;
    f10hi_source_death_seal good_death;
    f10hi_operator_spec valid_operator;
    f10hi_operator_spec invalid_operator;
    f10hi_backend_receipt receipt;
    f10hi_restoration_observation restoration;

    memset(&runtime, 0, sizeof(runtime));
    memset(&backend, 0, sizeof(backend));
    backend.state_token = UINT64_C(0x1122334455667788);

    if (!f10hi_runtime_init(
            &runtime,
            UINT64_C(0x1001),
            f10hi_synthetic_apply,
            &backend,
            true
        )) {
        return 0;
    }
    if (f10hi_open_receiver_word(&runtime, UINT64_C(0x3001))) {
        return 0;
    }
    if (!f10hi_mark_source_prepared(&runtime, UINT64_C(0x2001))) {
        return 0;
    }

    memset(&bad_death, 0, sizeof(bad_death));
    bad_death.source_pid = 41;
    bad_death.waitpid_exited_zero = true;
    bad_death.source_alive = true;
    bad_death.preparation_nonce = UINT64_C(0x2001);
    bad_death.seal_nonce = UINT64_C(0x2801);
    if (f10hi_seal_source_death(&runtime, &bad_death)) {
        return 0;
    }

    memset(&good_death, 0, sizeof(good_death));
    good_death.source_pid = 41;
    good_death.wait_status = 0;
    good_death.waitpid_exited_zero = true;
    good_death.source_alive = false;
    good_death.open_source_ipc = 0u;
    good_death.source_helper_count = 0u;
    good_death.preparation_nonce = UINT64_C(0x2001);
    good_death.seal_nonce = UINT64_C(0x2801);
    good_death.challenge_selected_before_source_death = false;
    if (!f10hi_seal_source_death(&runtime, &good_death)) {
        return 0;
    }
    if (f10hi_open_receiver_word(&runtime, UINT64_C(0x2001))) {
        return 0;
    }
    if (!f10hi_open_receiver_word(&runtime, UINT64_C(0x3001))) {
        return 0;
    }

    memset(&invalid_operator, 0, sizeof(invalid_operator));
    invalid_operator.operator_kind = F10HI_OP_QUERY_PROBE;
    invalid_operator.operator_instance_id = 1u;
    invalid_operator.amplitude = 1u;
    invalid_operator.operation_budget = 1u;
    invalid_operator.byte_preserving_required = true;
    if (f10hi_apply_operator(&runtime, &invalid_operator, &receipt)) {
        return 0;
    }
    invalid_operator.operator_kind = F10HI_OP_DESTRUCTIVE_RESET;
    if (f10hi_apply_operator(&runtime, &invalid_operator, &receipt)) {
        return 0;
    }
    invalid_operator.operator_kind = F10HI_OP_UNRESOLVED_SECOND_GENERATOR;
    if (f10hi_apply_operator(&runtime, &invalid_operator, &receipt)) {
        return 0;
    }

    memset(&valid_operator, 0, sizeof(valid_operator));
    valid_operator.operator_kind =
        F10HI_OP_REMOTE_STORE_SAME_VALUE_EXTRACTION_TARGET;
    valid_operator.operator_instance_id = UINT64_C(0xA001);
    valid_operator.line_set_id = 7u;
    valid_operator.route_id = 2u;
    valid_operator.executor_core = 5;
    valid_operator.amplitude = 96u;
    valid_operator.operation_budget = 4096u;
    valid_operator.byte_preserving_required = true;
    valid_operator.inverse_of_instance_id = UINT64_C(0xA000);
    if (f10hi_apply_operator(&runtime, &valid_operator, &receipt)) {
        return 0;
    }
    valid_operator.inverse_of_instance_id = 0u;
    if (!f10hi_apply_operator(&runtime, &valid_operator, &receipt)) {
        return 0;
    }
    if (
        !receipt.synthetic_backend
        || !receipt.byte_digest_preserved
        || receipt.backend_receipt_id == 0u
        || runtime.operation_count != 1u
        || runtime.word_hash == 0u
    ) {
        return 0;
    }
    if (!f10hi_close_receiver_word(&runtime)) {
        return 0;
    }
    if (f10hi_apply_operator(&runtime, &valid_operator, &receipt)) {
        return 0;
    }

    memset(&restoration, 0, sizeof(restoration));
    restoration.logical_byte_digest_before = UINT64_C(0x55AA);
    restoration.logical_byte_digest_after = UINT64_C(0x55AA);
    restoration.state_tomography_receipt_id = UINT64_C(0x7001);
    restoration.state_equivalence_passed = false;
    restoration.independent_output_retained = false;
    if (!f10hi_record_restoration_observation(&runtime, &restoration)) {
        return 0;
    }
    if (!f10hi_destroy_runtime(&runtime)) {
        return 0;
    }
    if (runtime.phase != F10HI_PHASE_DESTROYED) {
        return 0;
    }
    return 1;
}

#ifdef F10HI_BUILD_SELF_TEST
#include <stdio.h>

int main(void) {
    int passed = f10hi_runtime_self_test();
    printf(
        "{\"schema\":\"FAMILY10H_POST_SOURCE_OPERATOR_RUNTIME_I2A_SELF_TEST_V1\","
        "\"passed\":%s,"
        "\"backend\":\"synthetic_lifecycle_only\","
        "\"physical_backend_implemented\":false,"
        "\"h1_generator_established\":false,"
        "\"h1_generator_pair_established\":false,"
        "\"live_execution_authorized\":false}\n",
        passed ? "true" : "false"
    );
    return passed ? 0 : 1;
}
#endif
