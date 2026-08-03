#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: native contraction across a two-Boolean-port
 * unresolved interface.
 *
 * A four-port Boolean/F3 relation has 16 multiaffine coefficients, each held
 * as one relative cube-root phase.  CONTRACT2 closes two shared Boolean ports
 * by fixed coefficient-ring phase algebra.  It never decodes a coefficient,
 * visits a shared-port assignment, or materializes a truth table.
 */

#define main algebraic_series_parallel_embedded_main
#include "algebraic_series_parallel_phase.c"
#undef main

#define WIDE_CCOUNT 16U
#define UNION_CCOUNT 64U
#define WIDE_RELATIONS 3U
#define WIDE_TOKEN_CAPACITY 20U
#define WIDE_CARRIER_CELLS 96U

#define F_START 0U
#define G_START 16U
#define K_START 32U
#define H_START 48U
#define Z_START 64U
#define WIDE_BOUNDARY_START 80U

enum wide_mode {
    WIDE_CORRECT = 0,
    WIDE_WRONG_BOUNDARY_INVERSE = 1,
    WIDE_OMIT_PARENT_INVERSE = 2,
    WIDE_REORDER_INVERSES = 3,
    WIDE_BYPASS_FIRST_NORM = 4,
    WIDE_ORDINARY_SUM_FIRST_NORM = 5,
    WIDE_SWAP_SHARED_PORTS = 6
};

struct wide_program {
    int coefficient[WIDE_RELATIONS][WIDE_CCOUNT];
    uint64_t source_hash;
};

struct contract2_descriptor {
    size_t left_start;
    size_t right_start;
    size_t output_start;
};

struct wide_boundary {
    int coefficient[WIDE_CCOUNT];
    double maximum_root_error;
};

struct wide_execution {
    struct wide_boundary boundary;
    double displacement_l2;
    double restoration_max_abs;
    double integrity_max_abs;
};

static size_t wide_tokenize(
    char *line,
    char *token[WIDE_TOKEN_CAPACITY]
) {
    size_t count = 0U;
    char *save = NULL;
    for (
        char *part = strtok_r(line, " ", &save);
        part != NULL;
        part = strtok_r(NULL, " ", &save)
    ) {
        if (count == WIDE_TOKEN_CAPACITY) {
            fail("too many wide-interface fields");
        }
        token[count++] = part;
    }
    return count;
}

static struct wide_program read_wide_program(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct wide_program program = {
        .source_hash = UINT64_C(14695981039346656037)
    };
    static const char *const expected_name[WIDE_RELATIONS] = {
        "F", "G", "K"
    };
    char line[LINE_CAPACITY];
    size_t line_number = 0U;
    size_t relation = 0U;
    int header = 0;
    int end = 0;
    while (fgets(line, sizeof(line), stream) != NULL) {
        ++line_number;
        const size_t length = strlen(line);
        program.source_hash = hash_bytes(
            program.source_hash,
            (const unsigned char *)line,
            length
        );
        if (length == 0U || line[length - 1U] != '\n') {
            fail_line("every wide-interface record must end with LF", line_number);
        }
        if (memchr(line, '\r', length) != NULL) {
            fail_line("CR bytes are forbidden", line_number);
        }
        line[length - 1U] = '\0';
        if (end) {
            fail_line("record after END", line_number);
        }
        char *token[WIDE_TOKEN_CAPACITY] = {0};
        const size_t count = wide_tokenize(line, token);
        if (!header) {
            if (
                count != 2U
                || strcmp(token[0], "CATCAS_WIDE2_RELATION_CHAIN") != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid wide-interface header", line_number);
            }
            header = 1;
            continue;
        }
        if (strcmp(token[0], "RELATION") == 0) {
            if (
                relation == WIDE_RELATIONS
                || count != WIDE_CCOUNT + 2U
                || strcmp(token[1], expected_name[relation]) != 0
            ) {
                fail_line("invalid ordered wide relation", line_number);
            }
            for (size_t index = 0U; index < WIDE_CCOUNT; ++index) {
                char *tail = NULL;
                const long value = strtol(token[index + 2U], &tail, 10);
                if (
                    tail == token[index + 2U]
                    || *tail != '\0'
                    || value < 0
                    || value > 2
                ) {
                    fail_line("invalid BOOLEAN_F3 coefficient", line_number);
                }
                program.coefficient[relation][index] = (int)value;
            }
            ++relation;
        } else if (strcmp(token[0], "END") == 0) {
            if (count != 1U || relation != WIDE_RELATIONS) {
                fail_line("invalid wide-interface END", line_number);
            }
            end = 1;
        } else {
            fail_line("unknown wide-interface record", line_number);
        }
    }
    if (ferror(stream) || fclose(stream) != 0) {
        fail("failed to read wide-interface source");
    }
    if (!header || !end || relation != WIDE_RELATIONS) {
        fail("wide-interface source is incomplete");
    }
    return program;
}

static void read_wide_poly(
    const struct carrier *carrier,
    size_t start,
    double complex output[WIDE_CCOUNT]
) {
    for (size_t index = 0U; index < WIDE_CCOUNT; ++index) {
        output[index] = relative(carrier, start + index);
    }
}

static void wide_poly_multiply(
    const double complex left[WIDE_CCOUNT],
    const double complex right[WIDE_CCOUNT],
    double complex output[WIDE_CCOUNT]
) {
    for (size_t out = 0U; out < WIDE_CCOUNT; ++out) {
        output[out] = root3(0);
        for (size_t l = 0U; l < WIDE_CCOUNT; ++l) {
            for (size_t r = 0U; r < WIDE_CCOUNT; ++r) {
                if ((l | r) == out) {
                    output[out] = lock_f3_phase(
                        output[out] * symbol_product(left[l], right[r])
                    );
                }
            }
        }
    }
}

static size_t insert_zero_bit(size_t mask, size_t bit) {
    const size_t lower_mask = ((size_t)1U << bit) - 1U;
    return (
        (mask & lower_mask)
        | ((mask & ~lower_mask) << 1U)
    );
}

static double complex restricted_coefficient(
    const double complex *input,
    size_t reduced_mask,
    size_t eliminated_bit,
    int at_one
) {
    const size_t without = insert_zero_bit(
        reduced_mask,
        eliminated_bit
    );
    if (!at_one) {
        return input[without];
    }
    return lock_f3_phase(
        input[without]
        * input[without | ((size_t)1U << eliminated_bit)]
    );
}

static void boolean_norm(
    const double complex *input,
    size_t input_variables,
    size_t eliminated_bit,
    double complex *output,
    enum wide_mode mode
) {
    if (
        input_variables < 1U
        || input_variables > 6U
        || eliminated_bit >= input_variables
    ) {
        fail("invalid Boolean norm geometry");
    }
    const size_t output_count =
        (size_t)1U << (input_variables - 1U);
    double complex restrict_zero[UNION_CCOUNT / 2U];
    double complex restrict_one[UNION_CCOUNT / 2U];
    for (size_t index = 0U; index < output_count; ++index) {
        restrict_zero[index] = restricted_coefficient(
            input,
            index,
            eliminated_bit,
            0
        );
        restrict_one[index] = restricted_coefficient(
            input,
            index,
            eliminated_bit,
            1
        );
    }
    if (mode == WIDE_BYPASS_FIRST_NORM) {
        for (size_t out = 0U; out < output_count; ++out) {
            output[out] = restrict_zero[out];
        }
        return;
    }
    if (mode == WIDE_ORDINARY_SUM_FIRST_NORM) {
        for (size_t out = 0U; out < output_count; ++out) {
            output[out] = lock_f3_phase(
                restrict_zero[out] * restrict_one[out]
            );
        }
        return;
    }
    for (size_t out = 0U; out < output_count; ++out) {
        output[out] = root3(0);
        for (size_t l = 0U; l < output_count; ++l) {
            for (size_t r = 0U; r < output_count; ++r) {
                if ((l | r) == out) {
                    output[out] = lock_f3_phase(
                        output[out]
                        * symbol_product(
                            restrict_zero[l],
                            restrict_one[r]
                        )
                    );
                }
            }
        }
    }
}

static size_t swap_low_two_bits(size_t mask) {
    return (
        (mask & ~(size_t)3U)
        | ((mask & (size_t)1U) << 1U)
        | ((mask & (size_t)2U) >> 1U)
    );
}

static void contract2_factors(
    const struct carrier *carrier,
    const struct contract2_descriptor *descriptor,
    enum wide_mode mode,
    size_t operation_index,
    double complex output[WIDE_CCOUNT]
) {
    if (carrier == NULL || descriptor == NULL || output == NULL) {
        fail("null carrier or descriptor for CONTRACT2");
    }
    double complex left[WIDE_CCOUNT];
    double complex right[WIDE_CCOUNT];
    double complex left_squared[WIDE_CCOUNT];
    double complex right_squared[WIDE_CCOUNT];
    double complex intersection[UNION_CCOUNT];
    double complex first_norm[UNION_CCOUNT / 2U];

    read_wide_poly(carrier, descriptor->left_start, left);
    read_wide_poly(carrier, descriptor->right_start, right);
    if (mode == WIDE_SWAP_SHARED_PORTS && operation_index == 0U) {
        double complex swapped[WIDE_CCOUNT];
        for (size_t mask = 0U; mask < WIDE_CCOUNT; ++mask) {
            swapped[mask] = right[swap_low_two_bits(mask)];
        }
        memcpy(right, swapped, sizeof(right));
    }
    wide_poly_multiply(left, left, left_squared);
    wide_poly_multiply(right, right, right_squared);

    for (size_t mask = 0U; mask < UNION_CCOUNT; ++mask) {
        double complex f = root3(0);
        double complex g = root3(0);
        if ((mask & (size_t)0x30U) == 0U) {
            f = left_squared[mask & (WIDE_CCOUNT - 1U)];
        }
        if ((mask & (size_t)0x03U) == 0U) {
            const size_t local_g = (
                ((mask >> 2U) & (size_t)0x03U)
                | (((mask >> 4U) & (size_t)0x03U) << 2U)
            );
            g = right_squared[local_g];
        }
        intersection[mask] = lock_f3_phase(f * g);
    }

    const enum wide_mode first_norm_mode =
        operation_index == 0U ? mode : WIDE_CORRECT;
    boolean_norm(
        intersection,
        6U,
        2U,
        first_norm,
        first_norm_mode
    );
    boolean_norm(
        first_norm,
        5U,
        2U,
        output,
        WIDE_CORRECT
    );
}

static void apply_wide_factor(
    struct carrier *carrier,
    size_t start,
    const double complex factor[WIDE_CCOUNT],
    int inverse
) {
    for (size_t index = 0U; index < WIDE_CCOUNT; ++index) {
        multiply_cell(
            carrier,
            start + index,
            inverse ? conj(factor[index]) : factor[index]
        );
    }
}

static void apply_contract2(
    struct carrier *carrier,
    const struct contract2_descriptor *descriptor,
    enum wide_mode mode,
    size_t operation_index,
    int inverse
) {
    double complex factor[WIDE_CCOUNT];
    contract2_factors(
        carrier,
        descriptor,
        mode,
        operation_index,
        factor
    );
    apply_wide_factor(
        carrier,
        descriptor->output_start,
        factor,
        inverse
    );
}

static void encode_wide_program(
    struct carrier *carrier,
    const struct wide_program *program,
    int inverse
) {
    static const size_t start[WIDE_RELATIONS] = {
        F_START, G_START, K_START
    };
    if (!inverse) {
        for (size_t relation = 0U; relation < WIDE_RELATIONS; ++relation) {
            for (size_t index = 0U; index < WIDE_CCOUNT; ++index) {
                const double complex factor =
                    root3(program->coefficient[relation][index]);
                multiply_cell(
                    carrier,
                    start[relation] + index,
                    factor
                );
            }
        }
        return;
    }
    for (size_t relation = WIDE_RELATIONS; relation > 0U; --relation) {
        for (size_t index = WIDE_CCOUNT; index > 0U; --index) {
            const double complex factor = root3(
                program->coefficient[relation - 1U][index - 1U]
            );
            multiply_cell(
                carrier,
                start[relation - 1U] + index - 1U,
                conj(factor)
            );
        }
    }
}

static struct wide_boundary latch_wide_boundary(
    const struct carrier *carrier
) {
    struct wide_boundary boundary = {{0}, 0.0};
    for (size_t index = 0U; index < WIDE_CCOUNT; ++index) {
        double distance = 0.0;
        boundary.coefficient[index] = decode_root(
            relative(carrier, WIDE_BOUNDARY_START + index),
            &distance
        );
        if (distance > boundary.maximum_root_error) {
            boundary.maximum_root_error = distance;
        }
    }
    return boundary;
}

static struct wide_execution execute_wide(
    struct carrier *carrier,
    const struct wide_program *program,
    enum wide_mode mode
) {
    static const struct contract2_descriptor descriptor[2] = {
        {F_START, G_START, H_START},
        {H_START, K_START, Z_START}
    };
    struct carrier borrowed = snapshot_carrier(carrier);
    encode_wide_program(carrier, program, 0);
    apply_contract2(carrier, &descriptor[0], mode, 0U, 0);
    apply_contract2(carrier, &descriptor[1], mode, 1U, 0);

    double complex boundary_factor[WIDE_CCOUNT];
    read_wide_poly(carrier, Z_START, boundary_factor);
    apply_wide_factor(
        carrier,
        WIDE_BOUNDARY_START,
        boundary_factor,
        0
    );
    struct wide_execution execution = {
        .boundary = latch_wide_boundary(carrier),
        .displacement_l2 = displacement(carrier, &borrowed)
    };
    double complex rotated[WIDE_CCOUNT];
    for (size_t index = 0U; index < WIDE_CCOUNT; ++index) {
        rotated[index] = boundary_factor[
            (index + 1U) % WIDE_CCOUNT
        ];
    }
    apply_wide_factor(
        carrier,
        WIDE_BOUNDARY_START,
        mode == WIDE_WRONG_BOUNDARY_INVERSE
            ? rotated
            : boundary_factor,
        1
    );

    if (mode == WIDE_REORDER_INVERSES) {
        apply_contract2(
            carrier,
            &descriptor[0],
            mode,
            0U,
            1
        );
        apply_contract2(
            carrier,
            &descriptor[1],
            mode,
            1U,
            1
        );
    } else {
        if (mode != WIDE_OMIT_PARENT_INVERSE) {
            apply_contract2(
                carrier,
                &descriptor[1],
                mode,
                1U,
                1
            );
        }
        apply_contract2(
            carrier,
            &descriptor[0],
            mode,
            0U,
            1
        );
    }
    encode_wide_program(carrier, program, 1);
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
    free_carrier(&borrowed);
    return execution;
}

static int wide_boundaries_differ(
    const struct wide_boundary *left,
    const struct wide_boundary *right
) {
    for (size_t index = 0U; index < WIDE_CCOUNT; ++index) {
        if (left->coefficient[index] != right->coefficient[index]) {
            return 1;
        }
    }
    return 0;
}

static void print_wide_execution(
    const char *mode,
    const struct wide_execution *execution
) {
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"NATIVE_WIDTH2_TYPED_RELATIONAL_PHASE_CONTRACTION\","
        "\"shared_interface_boolean_ports\":2,"
        "\"relation_ports\":4,"
        "\"relation_phase_cells\":16,"
        "\"contract2_body_definitions\":1,"
        "\"contract2_instance_descriptors\":2,"
        "\"contract2_descriptor_bytes_each_current_abi\":24,"
        "\"contract2_descriptor_storage_bytes_current_abi\":48,"
        "\"forward_contract2_calls_per_transaction\":2,"
        "\"inverse_contract2_calls_per_transaction\":2,"
        "\"accepted_carrier_creation_count\":1,"
        "\"accepted_transactions_on_same_carrier\":2,"
        "\"input_relation_encoding_cells\":48,"
        "\"resident_intermediate_cells\":16,"
        "\"public_projection_cells\":16,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"serialized_intermediate_coefficients\":0,"
        "\"shared_assignment_loops\":0,"
        "\"tuple_witness_truth_table_slots\":0,"
        "\"carrier_cells\":96,"
        "\"live_carrier_complex_values\":192,"
        "\"live_carrier_bytes\":3072,"
        "\"comparison_snapshot_complex_values\":192,"
        "\"carrier_and_snapshot_complex_values\":384,"
        "\"maximum_contract2_workspace_complex_values\":240,"
        "\"symbol_product_calls_per_contract2\":1792,"
        "\"coefficient_accumulation_additions_per_contract2\":1792,"
        "\"restriction_and_intersection_additions_per_contract2\":112,"
        "\"retained_boundary_inverse_factor_complex_values\":16,"
        "\"control_rotation_complex_values\":16,"
        "\"accounted_phase_execution_upper_bound_complex_values\":656,"
        "\"accounted_phase_execution_upper_bound_bytes\":10496,"
        "\"coexisting_program_storage_bytes\":%zu,"
        "\"boundary_coefficients\":[",
        mode,
        2U * sizeof(struct wide_program)
    );
    for (size_t index = 0U; index < WIDE_CCOUNT; ++index) {
        printf(
            "%s%d",
            index == 0U ? "" : ",",
            execution->boundary.coefficient[index]
        );
    }
    printf(
        "],\"maximum_root_error\":%.12g,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_max_abs\":%.12g}\n",
        execution->boundary.maximum_root_error,
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->integrity_max_abs
    );
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--project-intermediate") == 0) {
        fail("only the final width-two boundary may be projected");
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        const struct contract2_descriptor descriptor = {
            F_START, G_START, H_START
        };
        double complex output[WIDE_CCOUNT];
        contract2_factors(
            NULL,
            &descriptor,
            WIDE_CORRECT,
            0U,
            output
        );
    }
    if (argc != 2 && argc != 3) {
        fprintf(
            stderr,
            "usage: %s PRIMARY.wr2 [REUSE.wr2]\n",
            argv[0]
        );
        return 2;
    }
    const struct wide_program primary = read_wide_program(argv[1]);
    const struct wide_program reuse =
        argc == 3 ? read_wide_program(argv[2]) : primary;
    struct process carrier_shape = {
        .carrier_cells = WIDE_CARRIER_CELLS
    };
    struct carrier carrier = make_carrier(&carrier_shape, 6101);
    const struct wide_execution nominal = execute_wide(
        &carrier,
        &primary,
        WIDE_CORRECT
    );
    const struct wide_execution reused = execute_wide(
        &carrier,
        &reuse,
        WIDE_CORRECT
    );
    free_carrier(&carrier);

    struct wide_execution control[5];
    static const enum wide_mode control_mode[5] = {
        WIDE_WRONG_BOUNDARY_INVERSE,
        WIDE_OMIT_PARENT_INVERSE,
        WIDE_REORDER_INVERSES,
        WIDE_BYPASS_FIRST_NORM,
        WIDE_ORDINARY_SUM_FIRST_NORM
    };
    for (size_t index = 0U; index < 5U; ++index) {
        carrier = make_carrier(&carrier_shape, 6101);
        control[index] = execute_wide(
            &carrier,
            &primary,
            control_mode[index]
        );
        free_carrier(&carrier);
    }
    carrier = make_carrier(&carrier_shape, 6101);
    const struct wide_execution swapped = execute_wide(
        &carrier,
        &primary,
        WIDE_SWAP_SHARED_PORTS
    );
    free_carrier(&carrier);

    print_wide_execution("width2-contract-chain", &nominal);
    print_wide_execution(
        "actual-restored-cross-program-width2-reuse",
        &reused
    );
    print_wide_execution("wrong-boundary-inverse", &control[0]);
    print_wide_execution("omitted-parent-inverse", &control[1]);
    print_wide_execution("reordered-noncommuting-inverses", &control[2]);
    print_wide_execution("bypassed-first-boolean-norm", &control[3]);
    print_wide_execution("ordinary-sum-first-boolean-norm", &control[4]);
    print_wide_execution("swapped-shared-port-order", &swapped);
    const int bypass_differs =
        wide_boundaries_differ(&nominal.boundary, &control[3].boundary);
    const int ordinary_differs =
        wide_boundaries_differ(&nominal.boundary, &control[4].boundary);
    const int swap_differs =
        wide_boundaries_differ(&nominal.boundary, &swapped.boundary);
    printf(
        "{\"mode\":\"width2-control-applicability\","
        "\"wrong_boundary\":%s,"
        "\"omitted_parent\":%s,"
        "\"reordered_noncommuting\":%s,"
        "\"bypassed_norm\":%s,"
        "\"ordinary_sum_norm\":%s,"
        "\"swapped_shared_ports\":%s}\n",
        control[0].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        control[1].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        control[2].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        bypass_differs ? "true" : "false",
        ordinary_differs ? "true" : "false",
        swap_differs ? "true" : "false"
    );

    const int valid = (
        nominal.boundary.maximum_root_error <= ROOT_TOLERANCE
        && reused.boundary.maximum_root_error <= ROOT_TOLERANCE
        && nominal.restoration_max_abs <= RESTORATION_TOLERANCE
        && reused.restoration_max_abs <= RESTORATION_TOLERANCE
        && control[0].restoration_max_abs >= CONTROL_MINIMUM
        && control[1].restoration_max_abs >= CONTROL_MINIMUM
        && control[2].restoration_max_abs >= CONTROL_MINIMUM
        && control[3].restoration_max_abs <= RESTORATION_TOLERANCE
        && control[4].restoration_max_abs <= RESTORATION_TOLERANCE
        && swapped.restoration_max_abs <= RESTORATION_TOLERANCE
        && bypass_differs
        && ordinary_differs
        && swap_differs
    );
    return valid ? 0 : 1;
}
