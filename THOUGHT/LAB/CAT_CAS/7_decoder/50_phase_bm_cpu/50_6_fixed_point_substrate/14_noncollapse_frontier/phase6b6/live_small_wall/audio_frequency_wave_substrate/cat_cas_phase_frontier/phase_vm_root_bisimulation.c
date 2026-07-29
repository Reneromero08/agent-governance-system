#define main streaming_phase_vm_embedded_main
#include "streaming_phase_vm.c"
#undef main

#include <inttypes.h>

/*
 * Diagnostic for the root-locked subset of the streaming phase VM.
 *
 * The production source above supplies the actual complex twin-rail carrier
 * and native phase operations.  A separate uint8_t transition system below
 * represents only the semantic Q3 relation at each register.  The diagnostic
 * compares both systems after every forward and inverse operation, then
 * restores and reuses the actual complex carrier.
 *
 * This deliberately inspects intermediate state and is not a CATVM custody
 * experiment.  Its purpose is to test whether the declared root-locked
 * software phase operations possess a resource absent from the strongest
 * directly matched compact symbolic recurrence.
 */

#define BISIM_EXHAUSTIVE_REGISTERS 5U
#define BISIM_CHAIN_REGISTERS 8U
#define BISIM_VARIANTS 9U
#define BISIM_EXHAUSTIVE_STATES 243U
#define BISIM_TOLERANCE 2.0e-11

struct symbolic_state {
    size_t registers;
    uint8_t symbol[BISIM_CHAIN_REGISTERS];
};

struct bisim_metrics {
    uint64_t trace_hash;
    uint64_t operation_cases;
    uint64_t checkpoints;
    uint64_t compared_relation_cells;
    uint64_t cswap_active_cases;
    uint64_t pcswap_active_cases;
    double maximum_root_distance;
    double maximum_restoration_error;
};

static void bisim_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static uint8_t bisim_mod3(int value) {
    int reduced = value % Q;
    if (reduced < 0) {
        reduced += Q;
    }
    return (uint8_t)reduced;
}

static void bisim_symbolic_apply(
    struct symbolic_state *state,
    const struct instruction *instruction,
    int inverse
) {
    const int direction = inverse ? -1 : 1;
    if (instruction->op == OP_ROT) {
        state->symbol[instruction->target] = bisim_mod3(
            (int)state->symbol[instruction->target]
                + direction * instruction->amount
        );
        return;
    }
    if (instruction->op == OP_ADD) {
        state->symbol[instruction->target] = bisim_mod3(
            (int)state->symbol[instruction->target]
                + direction * (int)state->symbol[instruction->a]
        );
        return;
    }
    if (instruction->op == OP_MULADD) {
        state->symbol[instruction->target] = bisim_mod3(
            (int)state->symbol[instruction->target]
                + direction
                    * (int)state->symbol[instruction->a]
                    * (int)state->symbol[instruction->b]
        );
        return;
    }
    if (instruction->op == OP_SWAP) {
        const uint8_t temporary = state->symbol[instruction->a];
        state->symbol[instruction->a] =
            state->symbol[instruction->b];
        state->symbol[instruction->b] = temporary;
        return;
    }
    if (instruction->op == OP_CSWAP) {
        if (state->symbol[instruction->target] == 1U) {
            const uint8_t temporary = state->symbol[instruction->a];
            state->symbol[instruction->a] =
                state->symbol[instruction->b];
            state->symbol[instruction->b] = temporary;
        }
        return;
    }
    if (instruction->op == OP_PCSWAP) {
        const uint8_t routed = bisim_mod3(
            (int)state->symbol[instruction->target]
                * (int)state->symbol[instruction->a]
        );
        if (routed == 1U) {
            const uint8_t temporary = state->symbol[instruction->b];
            state->symbol[instruction->b] =
                state->symbol[instruction->c];
            state->symbol[instruction->c] = temporary;
        }
        return;
    }
    bisim_fail("symbolic evaluator received unknown opcode");
}

static void bisim_set_symbols(
    struct carrier *carrier,
    const struct symbolic_state *state
) {
    if (carrier->registers != state->registers) {
        bisim_fail("carrier/symbolic register mismatch");
    }
    for (size_t index = 0U; index < state->registers; ++index) {
        carrier->working[index] =
            carrier->baseline[index]
            * root_of_unity((int)state->symbol[index]);
    }
}

static void bisim_compare(
    const struct carrier *carrier,
    const struct symbolic_state *state,
    struct bisim_metrics *metrics
) {
    if (carrier->registers != state->registers) {
        bisim_fail("comparison register mismatch");
    }
    for (size_t index = 0U; index < state->registers; ++index) {
        double distance = 0.0;
        const int symbol = decode_relation(
            relation(carrier, index),
            &distance
        );
        if (distance > metrics->maximum_root_distance) {
            metrics->maximum_root_distance = distance;
        }
        if (
            distance > BISIM_TOLERANCE
            || symbol != (int)state->symbol[index]
        ) {
            bisim_fail("native/symbolic relation mismatch");
        }
        metrics->compared_relation_cells += 1U;
    }
}

static void bisim_hash_byte(struct bisim_metrics *metrics, uint8_t value) {
    metrics->trace_hash ^= (uint64_t)value;
    metrics->trace_hash *= UINT64_C(1099511628211);
}

static void bisim_hash_u64(
    struct bisim_metrics *metrics,
    uint64_t value
) {
    for (unsigned int shift = 0U; shift < 64U; shift += 8U) {
        bisim_hash_byte(
            metrics,
            (uint8_t)((value >> shift) & UINT64_C(0xff))
        );
    }
}

static void bisim_hash_state(
    struct bisim_metrics *metrics,
    uint8_t tag,
    uint8_t variant,
    uint64_t case_index,
    uint8_t direction,
    uint64_t step,
    const struct symbolic_state *state
) {
    bisim_hash_byte(metrics, tag);
    bisim_hash_byte(metrics, variant);
    bisim_hash_u64(metrics, case_index);
    bisim_hash_byte(metrics, direction);
    bisim_hash_u64(metrics, step);
    bisim_hash_byte(metrics, (uint8_t)state->registers);
    for (size_t index = 0U; index < state->registers; ++index) {
        bisim_hash_byte(metrics, state->symbol[index]);
    }
}

static struct instruction bisim_variant(size_t variant) {
    const struct instruction variants[BISIM_VARIANTS] = {
        {
            .op = OP_ROT,
            .a = 0U,
            .b = 0U,
            .c = 0U,
            .target = 0U,
            .amount = 1
        },
        {
            .op = OP_ROT,
            .a = 0U,
            .b = 0U,
            .c = 0U,
            .target = 0U,
            .amount = 2
        },
        {
            .op = OP_ROT,
            .a = 0U,
            .b = 0U,
            .c = 0U,
            .target = 0U,
            .amount = 0
        },
        {
            .op = OP_ADD,
            .a = 1U,
            .b = 0U,
            .c = 0U,
            .target = 0U,
            .amount = 0
        },
        {
            .op = OP_MULADD,
            .a = 1U,
            .b = 2U,
            .c = 0U,
            .target = 0U,
            .amount = 0
        },
        {
            .op = OP_MULADD,
            .a = 1U,
            .b = 1U,
            .c = 0U,
            .target = 0U,
            .amount = 0
        },
        {
            .op = OP_SWAP,
            .a = 1U,
            .b = 2U,
            .c = 0U,
            .target = 0U,
            .amount = 0
        },
        {
            .op = OP_CSWAP,
            .a = 1U,
            .b = 2U,
            .c = 0U,
            .target = 0U,
            .amount = 0
        },
        {
            .op = OP_PCSWAP,
            .a = 1U,
            .b = 2U,
            .c = 3U,
            .target = 0U,
            .amount = 0
        }
    };
    if (variant >= BISIM_VARIANTS) {
        bisim_fail("variant index out of range");
    }
    return variants[variant];
}

static struct symbolic_state bisim_state_from_index(uint64_t value) {
    struct symbolic_state state = {
        .registers = BISIM_EXHAUSTIVE_REGISTERS,
        .symbol = {0U}
    };
    uint64_t remaining = value;
    for (size_t index = 0U; index < state.registers; ++index) {
        state.symbol[index] = (uint8_t)(remaining % Q);
        remaining /= Q;
    }
    return state;
}

static void bisim_run_exhaustive(struct bisim_metrics *metrics) {
    for (size_t variant = 0U; variant < BISIM_VARIANTS; ++variant) {
        const struct instruction instruction = bisim_variant(variant);
        for (
            uint64_t case_index = 0U;
            case_index < BISIM_EXHAUSTIVE_STATES;
            ++case_index
        ) {
            const struct symbolic_state initial =
                bisim_state_from_index(case_index);
            struct symbolic_state symbolic = initial;
            struct carrier native = make_carrier(
                BISIM_EXHAUSTIVE_REGISTERS,
                (int)(101U + variant)
            );
            bisim_set_symbols(&native, &symbolic);
            struct carrier before = clone_carrier(&native);

            if (
                instruction.op == OP_CSWAP
                && symbolic.symbol[instruction.target] == 1U
            ) {
                metrics->cswap_active_cases += 1U;
            }
            if (
                instruction.op == OP_PCSWAP
                && bisim_mod3(
                    (int)symbolic.symbol[instruction.target]
                        * (int)symbolic.symbol[instruction.a]
                ) == 1U
            ) {
                metrics->pcswap_active_cases += 1U;
            }

            apply_forward(&native, &instruction);
            bisim_symbolic_apply(&symbolic, &instruction, 0);
            bisim_compare(&native, &symbolic, metrics);
            bisim_hash_state(
                metrics,
                UINT8_C(0xe0),
                (uint8_t)(variant + 1U),
                case_index,
                0U,
                0U,
                &symbolic
            );
            metrics->checkpoints += 1U;

            apply_inverse(&native, &instruction, 0);
            bisim_symbolic_apply(&symbolic, &instruction, 1);
            bisim_compare(&native, &symbolic, metrics);
            bisim_hash_state(
                metrics,
                UINT8_C(0xe0),
                (uint8_t)(variant + 1U),
                case_index,
                1U,
                0U,
                &symbolic
            );
            metrics->checkpoints += 1U;

            if (
                memcmp(
                    symbolic.symbol,
                    initial.symbol,
                    initial.registers * sizeof(initial.symbol[0])
                ) != 0
            ) {
                bisim_fail("symbolic exhaustive inverse did not restore");
            }
            const double restoration = carrier_error(&native, &before);
            if (restoration > metrics->maximum_restoration_error) {
                metrics->maximum_restoration_error = restoration;
            }
            if (restoration > BISIM_TOLERANCE) {
                bisim_fail("native exhaustive inverse did not restore");
            }
            metrics->operation_cases += 1U;
            free_carrier(&before);
            free_carrier(&native);
        }
    }
}

static const struct instruction BISIM_PROGRAM_ONE[] = {
    {.op = OP_ROT, .target = 0U, .amount = 1},
    {.op = OP_ADD, .a = 0U, .target = 1U},
    {.op = OP_MULADD, .a = 1U, .b = 2U, .target = 3U},
    {.op = OP_SWAP, .a = 3U, .b = 4U},
    {.op = OP_CSWAP, .a = 4U, .b = 5U, .target = 0U},
    {.op = OP_PCSWAP, .a = 2U, .b = 5U, .c = 6U, .target = 0U},
    {.op = OP_ROT, .target = 0U, .amount = 1},
    {.op = OP_CSWAP, .a = 6U, .b = 7U, .target = 0U},
    {.op = OP_PCSWAP, .a = 2U, .b = 1U, .c = 7U, .target = 0U},
    {.op = OP_ADD, .a = 4U, .target = 7U},
    {.op = OP_MULADD, .a = 6U, .b = 1U, .target = 2U},
    {.op = OP_SWAP, .a = 2U, .b = 3U}
};

static const struct instruction BISIM_PROGRAM_TWO[] = {
    {.op = OP_ROT, .target = 7U, .amount = 2},
    {.op = OP_MULADD, .a = 0U, .b = 1U, .target = 2U},
    {.op = OP_ADD, .a = 2U, .target = 4U},
    {.op = OP_SWAP, .a = 0U, .b = 6U},
    {.op = OP_CSWAP, .a = 3U, .b = 5U, .target = 7U},
    {.op = OP_PCSWAP, .a = 4U, .b = 1U, .c = 6U, .target = 7U},
    {.op = OP_ADD, .a = 5U, .target = 3U},
    {.op = OP_ROT, .target = 1U, .amount = 1}
};

static struct symbolic_state bisim_chain_initial(void) {
    const struct symbolic_state state = {
        .registers = BISIM_CHAIN_REGISTERS,
        .symbol = {0U, 2U, 1U, 0U, 1U, 2U, 0U, 2U}
    };
    return state;
}

static void bisim_run_program_forward(
    struct carrier *native,
    struct symbolic_state *symbolic,
    const struct instruction *program,
    size_t program_length,
    uint8_t program_id,
    struct bisim_metrics *metrics
) {
    for (size_t index = 0U; index < program_length; ++index) {
        apply_forward(native, &program[index]);
        bisim_symbolic_apply(symbolic, &program[index], 0);
        bisim_compare(native, symbolic, metrics);
        bisim_hash_state(
            metrics,
            UINT8_C(0xc0),
            program_id,
            0U,
            0U,
            index,
            symbolic
        );
        metrics->checkpoints += 1U;
    }
}

static void bisim_run_program_inverse(
    struct carrier *native,
    struct symbolic_state *symbolic,
    const struct instruction *program,
    size_t program_length,
    uint8_t program_id,
    struct bisim_metrics *metrics
) {
    for (size_t reverse = program_length; reverse-- > 0U;) {
        apply_inverse(native, &program[reverse], 0);
        bisim_symbolic_apply(symbolic, &program[reverse], 1);
        bisim_compare(native, symbolic, metrics);
        bisim_hash_state(
            metrics,
            UINT8_C(0xc0),
            program_id,
            0U,
            1U,
            program_length - 1U - reverse,
            symbolic
        );
        metrics->checkpoints += 1U;
    }
}

static void bisim_copy_boundary(
    uint8_t destination[BISIM_CHAIN_REGISTERS],
    const struct symbolic_state *state
) {
    memcpy(
        destination,
        state->symbol,
        BISIM_CHAIN_REGISTERS * sizeof(destination[0])
    );
}

static int bisim_boundary_equal(
    const uint8_t left[BISIM_CHAIN_REGISTERS],
    const uint8_t right[BISIM_CHAIN_REGISTERS]
) {
    return memcmp(
        left,
        right,
        BISIM_CHAIN_REGISTERS * sizeof(left[0])
    ) == 0;
}

static double bisim_native_relation_error(
    const struct carrier *left,
    const struct carrier *right
) {
    if (left->registers != right->registers) {
        bisim_fail("native relation comparison register mismatch");
    }
    double maximum = 0.0;
    for (size_t index = 0U; index < left->registers; ++index) {
        maximum = fmax(
            maximum,
            cabs(relation(left, index) - relation(right, index))
        );
    }
    return maximum;
}

static double bisim_missing_inverse_error(void) {
    struct symbolic_state symbolic = bisim_chain_initial();
    struct carrier native = make_carrier(BISIM_CHAIN_REGISTERS, 211);
    bisim_set_symbols(&native, &symbolic);
    struct carrier before = clone_carrier(&native);
    for (
        size_t index = 0U;
        index < sizeof(BISIM_PROGRAM_ONE) / sizeof(BISIM_PROGRAM_ONE[0]);
        ++index
    ) {
        apply_forward(&native, &BISIM_PROGRAM_ONE[index]);
    }
    const double error = carrier_error(&native, &before);
    free_carrier(&before);
    free_carrier(&native);
    return error;
}

static double bisim_wrong_inverse_error(void) {
    struct symbolic_state symbolic = bisim_chain_initial();
    struct carrier native = make_carrier(BISIM_CHAIN_REGISTERS, 223);
    bisim_set_symbols(&native, &symbolic);
    struct carrier before = clone_carrier(&native);
    const size_t program_length =
        sizeof(BISIM_PROGRAM_ONE) / sizeof(BISIM_PROGRAM_ONE[0]);
    for (size_t index = 0U; index < program_length; ++index) {
        apply_forward(&native, &BISIM_PROGRAM_ONE[index]);
    }
    for (size_t reverse = program_length; reverse-- > 0U;) {
        apply_inverse(
            &native,
            &BISIM_PROGRAM_ONE[reverse],
            reverse == program_length - 1U
        );
    }
    const double error = carrier_error(&native, &before);
    free_carrier(&before);
    free_carrier(&native);
    return error;
}

static double bisim_reordered_inverse_error(void) {
    struct symbolic_state symbolic = bisim_chain_initial();
    struct carrier native = make_carrier(BISIM_CHAIN_REGISTERS, 227);
    bisim_set_symbols(&native, &symbolic);
    struct carrier before = clone_carrier(&native);
    const size_t program_length =
        sizeof(BISIM_PROGRAM_ONE) / sizeof(BISIM_PROGRAM_ONE[0]);
    for (size_t index = 0U; index < program_length; ++index) {
        apply_forward(&native, &BISIM_PROGRAM_ONE[index]);
    }
    for (size_t index = 0U; index < program_length; ++index) {
        apply_inverse(&native, &BISIM_PROGRAM_ONE[index], 0);
    }
    const double error = carrier_error(&native, &before);
    free_carrier(&before);
    free_carrier(&native);
    return error;
}

static void bisim_print_symbols(
    const uint8_t value[BISIM_CHAIN_REGISTERS]
) {
    printf("[");
    for (size_t index = 0U; index < BISIM_CHAIN_REGISTERS; ++index) {
        printf("%s%u", index == 0U ? "" : ",", (unsigned int)value[index]);
    }
    printf("]");
}

int main(void) {
    struct bisim_metrics metrics = {
        .trace_hash = UINT64_C(14695981039346656037),
        .operation_cases = 0U,
        .checkpoints = 0U,
        .compared_relation_cells = 0U,
        .cswap_active_cases = 0U,
        .pcswap_active_cases = 0U,
        .maximum_root_distance = 0.0,
        .maximum_restoration_error = 0.0
    };
    bisim_run_exhaustive(&metrics);

    struct symbolic_state initial = bisim_chain_initial();
    struct symbolic_state symbolic = initial;
    struct carrier native = make_carrier(BISIM_CHAIN_REGISTERS, 197);
    bisim_set_symbols(&native, &symbolic);
    struct carrier original = clone_carrier(&native);
    double complex *const baseline_backing = native.baseline;
    double complex *const working_backing = native.working;

    const size_t program_one_length =
        sizeof(BISIM_PROGRAM_ONE) / sizeof(BISIM_PROGRAM_ONE[0]);
    bisim_run_program_forward(
        &native,
        &symbolic,
        BISIM_PROGRAM_ONE,
        program_one_length,
        1U,
        &metrics
    );
    uint8_t primary_boundary[BISIM_CHAIN_REGISTERS];
    bisim_copy_boundary(primary_boundary, &symbolic);
    bisim_run_program_inverse(
        &native,
        &symbolic,
        BISIM_PROGRAM_ONE,
        program_one_length,
        1U,
        &metrics
    );
    const double primary_restoration = carrier_error(&native, &original);
    if (
        primary_restoration > BISIM_TOLERANCE
        || memcmp(
            symbolic.symbol,
            initial.symbol,
            BISIM_CHAIN_REGISTERS * sizeof(initial.symbol[0])
        ) != 0
    ) {
        bisim_fail("primary chained transaction did not restore");
    }

    struct carrier fresh = make_carrier(BISIM_CHAIN_REGISTERS, 197);
    struct symbolic_state fresh_symbolic = initial;
    bisim_set_symbols(&fresh, &fresh_symbolic);
    const size_t program_two_length =
        sizeof(BISIM_PROGRAM_TWO) / sizeof(BISIM_PROGRAM_TWO[0]);
    bisim_run_program_forward(
        &native,
        &symbolic,
        BISIM_PROGRAM_TWO,
        program_two_length,
        2U,
        &metrics
    );
    uint8_t reuse_boundary[BISIM_CHAIN_REGISTERS];
    bisim_copy_boundary(reuse_boundary, &symbolic);
    for (size_t index = 0U; index < program_two_length; ++index) {
        apply_forward(&fresh, &BISIM_PROGRAM_TWO[index]);
        bisim_symbolic_apply(
            &fresh_symbolic,
            &BISIM_PROGRAM_TWO[index],
            0
        );
    }
    uint8_t fresh_boundary[BISIM_CHAIN_REGISTERS];
    bisim_copy_boundary(fresh_boundary, &fresh_symbolic);
    if (!bisim_boundary_equal(reuse_boundary, fresh_boundary)) {
        bisim_fail("fresh/restored reuse boundary mismatch");
    }
    const double fresh_restored_native_boundary_error =
        bisim_native_relation_error(&native, &fresh);
    if (fresh_restored_native_boundary_error > BISIM_TOLERANCE) {
        bisim_fail("fresh/restored native relation boundary mismatch");
    }
    bisim_run_program_inverse(
        &native,
        &symbolic,
        BISIM_PROGRAM_TWO,
        program_two_length,
        2U,
        &metrics
    );
    const double reuse_restoration = carrier_error(&native, &original);
    if (
        reuse_restoration > BISIM_TOLERANCE
        || memcmp(
            symbolic.symbol,
            initial.symbol,
            BISIM_CHAIN_REGISTERS * sizeof(initial.symbol[0])
        ) != 0
    ) {
        bisim_fail("reuse chained transaction did not restore");
    }
    if (
        native.baseline != baseline_backing
        || native.working != working_backing
    ) {
        bisim_fail("actual carrier backing changed during reuse");
    }
    metrics.maximum_restoration_error = fmax(
        metrics.maximum_restoration_error,
        fmax(primary_restoration, reuse_restoration)
    );

    const double missing_inverse_error = bisim_missing_inverse_error();
    const double wrong_inverse_error = bisim_wrong_inverse_error();
    const double reordered_inverse_error = bisim_reordered_inverse_error();
    if (
        missing_inverse_error <= BISIM_TOLERANCE
        || wrong_inverse_error <= BISIM_TOLERANCE
        || reordered_inverse_error <= BISIM_TOLERANCE
    ) {
        bisim_fail("inverse control unexpectedly restored");
    }

    printf("{\n");
    printf("  \"result\": \"PASS\",\n");
    printf(
        "  \"claim_candidate\": "
        "\"BOUNDED_ROOT_LOCKED_PHASE_VM_OPERATION_TRACE_CLASSICAL_"
        "BISIMULATION_WITH_NUMERICAL_RESTORATION_AND_REUSE\",\n"
    );
    printf("  \"alphabet\": \"Q3_ROOTS_OF_UNITY\",\n");
    printf("  \"native_backend\": \"streaming_phase_vm.c\",\n");
    printf(
        "  \"scope\": \"ROOT_LOCKED_PUBLIC_HOLO_OPERATION_DOMAIN\",\n"
    );
    printf("  \"exhaustive\": {\n");
    printf("    \"registers\": %u,\n", BISIM_EXHAUSTIVE_REGISTERS);
    printf("    \"operation_variants\": %u,\n", BISIM_VARIANTS);
    printf(
        "    \"input_states_per_variant\": %u,\n",
        BISIM_EXHAUSTIVE_STATES
    );
    printf(
        "    \"operation_cases\": %" PRIu64 ",\n",
        metrics.operation_cases
    );
    printf(
        "    \"forward_inverse_checkpoints\": %" PRIu64 ",\n",
        UINT64_C(2) * metrics.operation_cases
    );
    printf(
        "    \"cswap_active_cases\": %" PRIu64 ",\n",
        metrics.cswap_active_cases
    );
    printf(
        "    \"pcswap_active_cases\": %" PRIu64 "\n",
        metrics.pcswap_active_cases
    );
    printf("  },\n");
    printf("  \"chained_transactions\": {\n");
    printf("    \"registers\": %u,\n", BISIM_CHAIN_REGISTERS);
    printf("    \"primary_forward_steps\": %zu,\n", program_one_length);
    printf("    \"reuse_forward_steps\": %zu,\n", program_two_length);
    printf("    \"primary_boundary\": ");
    bisim_print_symbols(primary_boundary);
    printf(",\n");
    printf("    \"reuse_boundary\": ");
    bisim_print_symbols(reuse_boundary);
    printf(",\n");
    printf("    \"fresh_boundary\": ");
    bisim_print_symbols(fresh_boundary);
    printf(",\n");
    printf(
        "    \"fresh_restored_boundary_equal\": %s,\n",
        bisim_boundary_equal(reuse_boundary, fresh_boundary)
            ? "true"
            : "false"
    );
    printf(
        "    \"fresh_restored_native_boundary_max_abs\": %.17g,\n"
        "    \"fresh_restored_native_boundary_equal_within_tolerance\": "
        "true,\n",
        fresh_restored_native_boundary_error
    );
    printf(
        "    \"same_carrier_backing_reused\": true,\n"
        "    \"restoration_generation_sequence\": [1,2],\n"
        "    \"baseline_reload_bytes\": 0\n"
    );
    printf("  },\n");
    printf("  \"trace\": {\n");
    printf(
        "    \"semantic_trace_fnv1a64\": \"%016" PRIx64 "\",\n",
        metrics.trace_hash
    );
    printf(
        "    \"trace_hash_role\": "
        "\"DETERMINISTIC_REPLAY_COMMITMENT_NOT_COLLISION_RESISTANT_"
        "PROOF\",\n"
    );
    printf(
        "    \"checkpoints\": %" PRIu64 ",\n",
        metrics.checkpoints
    );
    printf(
        "    \"compared_relation_cells\": %" PRIu64 ",\n",
        metrics.compared_relation_cells
    );
    printf(
        "    \"intermediate_state_inspected_by_diagnostic\": true,\n"
        "    \"intermediate_state_emitted\": false\n"
    );
    printf("  },\n");
    printf("  \"numerics\": {\n");
    printf(
        "    \"predeclared_tolerance\": %.12g,\n",
        BISIM_TOLERANCE
    );
    printf(
        "    \"maximum_root_distance\": %.17g,\n",
        metrics.maximum_root_distance
    );
    printf(
        "    \"maximum_restoration_error\": %.17g,\n",
        metrics.maximum_restoration_error
    );
    printf(
        "    \"primary_restoration_error\": %.17g,\n",
        primary_restoration
    );
    printf(
        "    \"reuse_restoration_error\": %.17g\n",
        reuse_restoration
    );
    printf("  },\n");
    printf("  \"controls\": {\n");
    printf(
        "    \"missing_inverse_restored\": false,\n"
        "    \"missing_inverse_error\": %.17g,\n",
        missing_inverse_error
    );
    printf(
        "    \"wrong_inverse_restored\": false,\n"
        "    \"wrong_inverse_error\": %.17g,\n",
        wrong_inverse_error
    );
    printf(
        "    \"reordered_inverse_applicable\": true,\n"
        "    \"reordered_inverse_restored\": false,\n"
        "    \"reordered_inverse_error\": %.17g\n",
        reordered_inverse_error
    );
    printf("  },\n");
    printf("  \"matched_compact_classical\": {\n");
    printf(
        "    \"representation\": \"UINT8_Q3_SYMBOL_PER_REGISTER\",\n"
        "    \"transition_parity_after_every_operation\": true,\n"
        "    \"inverse_parity_after_every_operation\": true,\n"
        "    \"implementation_payload_bytes_per_register\": 1,\n"
        "    \"information_lower_bound_bits_per_register\": "
        "1.584962500721156,\n"
        "    \"two_bit_packing_available\": true,\n"
        "    \"program_specific_optimization_may_be_smaller\": true\n"
    );
    printf("  },\n");
    printf("  \"resource_law\": {\n");
    printf(
        "    \"native_complex128_rails_per_register\": 2,\n"
        "    \"native_heap_payload_bytes_per_register\": %zu,\n",
        UINT64_C(2) * sizeof(double complex)
    );
    printf(
        "    \"classical_uint8_payload_bytes_per_register\": "
        "%zu,\n",
        sizeof(uint8_t)
    );
    printf(
        "    \"exhaustive_one_native_carrier_heap_payload_bytes\": "
        "%zu,\n",
        BISIM_EXHAUSTIVE_REGISTERS
            * UINT64_C(2) * sizeof(double complex)
    );
    printf(
        "    \"exhaustive_simultaneous_native_and_baseline_clone_heap_"
        "payload_bytes\": %zu,\n",
        BISIM_EXHAUSTIVE_REGISTERS
            * UINT64_C(4) * sizeof(double complex)
    );
    printf(
        "    \"exhaustive_one_classical_state_payload_bytes\": %zu,\n",
        BISIM_EXHAUSTIVE_REGISTERS * sizeof(uint8_t)
    );
    printf(
        "    \"chain_one_native_carrier_heap_payload_bytes\": %zu,\n",
        BISIM_CHAIN_REGISTERS
            * UINT64_C(2) * sizeof(double complex)
    );
    printf(
        "    \"chain_simultaneous_three_native_carrier_heap_payload_"
        "bytes\": %zu,\n",
        BISIM_CHAIN_REGISTERS
            * UINT64_C(6) * sizeof(double complex)
    );
    printf(
        "    \"chain_one_classical_state_payload_bytes\": %zu,\n",
        BISIM_CHAIN_REGISTERS * sizeof(uint8_t)
    );
    printf(
        "    \"public_program_descriptor_shared_between_paths\": true,\n"
        "    \"trace_symbolic_and_stack_buffers_are_verification_only\": "
        "true,\n"
        "    \"accepted_proof_total_memory_claimed\": false,\n"
        "    \"runtime_advantage_claimed\": false,\n"
        "    \"whole_process_peak_claimed\": false\n"
    );
    printf("  },\n");
    printf(
        "  \"restoration_classification\": "
        "\"NUMERICAL_PHYSICAL_STATE_RESTORATION\",\n"
    );
    printf(
        "  \"finite_deterministic_identity_simulation_lemma\": "
        "\"A_CLASSICAL_SIMULATOR_MAY_STORE_THE_SAME_FINITE_MACHINE_STATE_"
        "AND_APPLY_THE_SAME_DETERMINISTIC_TRANSITION;_THE_ROOT_LOCKED_"
        "SUBDOMAIN_HERE_ADMITS_THE_STRICTLY_SMALLER_Q3_SYMBOLIC_STATE\",\n"
    );
    printf(
        "  \"lemma_scope\": "
        "\"FINITE_DETERMINISTIC_SOFTWARE_TRANSITION_SYSTEMS_ONLY\",\n"
    );
    printf(
        "  \"operand_position_coverage\": "
        "\"NINE_CANONICAL_LEGAL_WIRING_VARIANTS_ALL_243_Q3_STATES\",\n"
    );
    printf(
        "  \"all_register_placements_executed\": false,\n"
        "  \"register_permutation_equivariance_claimed_from_source\": "
        "false,\n"
    );
    printf(
        "  \"exceptions_not_adjudicated\": "
        "[\"PHYSICAL_ANALOG_RESOURCES\",\"EXTERNAL_ORACLES\","
        "\"NONDETERMINISTIC_RESOURCES\",\"UNBOUNDED_PRECISION_MODELS\"],\n"
    );
    printf(
        "  \"catvm_custody_established\": false,\n"
        "  \"distinct_phase_resource_established\": false,\n"
        "  \"computational_advantage\": false,\n"
        "  \"small_wall_crossed\": false,\n"
        "  \"physical_waveform_execution\": false,\n"
        "  \"physical_bit_replacement\": false,\n"
        "  \"unbounded_computation_established\": false,\n"
        "  \"claim_ceiling\": "
        "\"LINUX_X86_64_DIRECT_PROCESS_COMPLEX128_Q3_ROOT_LOCKED_"
        "STREAMING_PHASE_VM_SIX_OPCODES_NINE_CANONICAL_LEGAL_WIRING_"
        "VARIANTS_ALL_243_Q3_STATES_AND_TWO_EIGHT_REGISTER_CHAINED_"
        "PROGRAMS_SOFTWARE_ONLY\",\n"
        "  \"terminal\": false\n"
    );
    printf("}\n");

    free_carrier(&fresh);
    free_carrier(&original);
    free_carrier(&native);
    return 0;
}
