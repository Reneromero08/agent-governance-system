#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: compact full-boundary closure for a typed family
 * of projected affine Boolean relations.
 *
 * A port value has Q quotient bits and F unconstrained fibre bits.  A relation
 * R(A,a) is the complete many-to-many relation
 *
 *     pi(y) = A pi(x) XOR a.
 *
 * The fibre coordinates never need enumeration.  Native phase composition is
 *
 *     R(B,b) o R(A,a) = R(B A, B a XOR b),
 *
 * so the complete surviving relation remains Q*(Q+1) phase coefficients.
 * No tuple, assignment, witness, or dense relation table is materialized.
 */

#define main algebraic_series_parallel_embedded_main
#include "algebraic_series_parallel_phase.c"
#undef main

#ifndef QA_Q
#define QA_Q 8U
#endif
#ifndef QA_FIBRE
#define QA_FIBRE 4U
#endif
#define QA_RELATIONS 3U
#define QA_BLOCK_CELLS (QA_Q * QA_Q + QA_Q)
#define QA_CARRIER_CELLS (6U * QA_BLOCK_CELLS)
#define QA_F_START 0U
#define QA_G_START (1U * QA_BLOCK_CELLS)
#define QA_K_START (2U * QA_BLOCK_CELLS)
#define QA_H_START (3U * QA_BLOCK_CELLS)
#define QA_Z_START (4U * QA_BLOCK_CELLS)
#define QA_BOUNDARY_START (5U * QA_BLOCK_CELLS)
#ifndef QA_REPEAT_CYCLES
#define QA_REPEAT_CYCLES 128U
#endif

#if QA_Q < 1U || QA_Q > 64U
#error "QA_Q must be in 1..64"
#endif

#if QA_FIBRE < 1U || QA_FIBRE > 20U
#error "QA_FIBRE must be in 1..20"
#endif

enum qa_algebra_mode {
    QA_ALGEBRA_CORRECT = 0,
    QA_ALGEBRA_F3_SUM = 1,
    QA_ALGEBRA_REVERSE_ORDER = 2
};

enum qa_restore_mode {
    QA_RESTORE_CORRECT = 0,
    QA_RESTORE_WRONG_PARENT = 1,
    QA_RESTORE_MISSING_PARENT = 2,
    QA_RESTORE_REORDERED = 3,
    QA_RESTORE_SNAPSHOT = 4
};

struct qa_program {
    unsigned char matrix[QA_RELATIONS][QA_Q][QA_Q];
    unsigned char bias[QA_RELATIONS][QA_Q];
    uint64_t source_hash;
};

struct qa_descriptor {
    size_t left_start;
    size_t right_start;
    size_t output_start;
};

struct qa_stats {
    uint64_t native_compose_calls;
    uint64_t native_and_products;
    uint64_t native_xor_accumulations;
    uint64_t symbol_product_calls;
    uint64_t carrier_reads;
    uint64_t phase_cell_updates;
    uint64_t inverse_factor_recomputations;
    uint64_t boundary_decodes;
    uint64_t restoration_cell_checks;
};

struct qa_boundary {
    int coefficient[QA_BLOCK_CELLS];
    uint64_t hash;
    double maximum_root_error;
};

struct qa_execution {
    struct qa_boundary boundary;
    struct qa_stats stats;
    double displacement_l2;
    double restoration_max_abs;
    double integrity_max_abs;
    int snapshot_loaded;
};

static size_t qa_matrix_cell(size_t start, size_t row, size_t column) {
    return start + row * QA_Q + column;
}

static size_t qa_bias_cell(size_t start, size_t row) {
    return start + QA_Q * QA_Q + row;
}

static unsigned char qa_public_bit(
    size_t variant,
    size_t relation,
    size_t row,
    size_t column
) {
    if (row == column) {
        return 1U;
    }
    const size_t selector =
        5U * row + 3U * column + 2U * relation + variant;
    if ((relation + variant) % 2U == 0U) {
        return column < row && selector % 5U == 0U;
    }
    return column > row && selector % 5U == 0U;
}

static struct qa_program qa_make_program(size_t variant) {
    struct qa_program program = {
        .source_hash = UINT64_C(14695981039346656037)
    };
    for (size_t relation = 0U; relation < QA_RELATIONS; ++relation) {
        for (size_t row = 0U; row < QA_Q; ++row) {
            for (size_t column = 0U; column < QA_Q; ++column) {
                const unsigned char bit = qa_public_bit(
                    variant, relation, row, column
                );
                program.matrix[relation][row][column] = bit;
                program.source_hash = hash_bytes(
                    program.source_hash, &bit, 1U
                );
            }
            const unsigned char bias =
                (unsigned char)((3U * row + relation + variant) % 2U);
            program.bias[relation][row] = bias;
            program.source_hash = hash_bytes(
                program.source_hash, &bias, 1U
            );
        }
    }
    return program;
}

static void qa_update_cell(
    struct carrier *carrier,
    size_t cell,
    double complex factor,
    struct qa_stats *stats
) {
    multiply_cell(carrier, cell, factor);
    ++stats->phase_cell_updates;
}

static double complex qa_read(
    const struct carrier *carrier,
    size_t cell,
    struct qa_stats *stats
) {
    ++stats->carrier_reads;
    return relative(carrier, cell);
}

static double complex qa_and(
    double complex left,
    double complex right,
    struct qa_stats *stats
) {
    ++stats->native_and_products;
    ++stats->symbol_product_calls;
    return symbol_product(left, right);
}

static double complex qa_xor(
    double complex left,
    double complex right,
    enum qa_algebra_mode mode,
    struct qa_stats *stats
) {
    ++stats->native_xor_accumulations;
    if (mode == QA_ALGEBRA_F3_SUM) {
        return lock_f3_phase(left * right);
    }
    ++stats->symbol_product_calls;
    return lock_f3_phase(
        left * right * symbol_product(left, right)
    );
}

static double complex qa_matrix_factor(
    const struct carrier *carrier,
    const struct qa_descriptor *descriptor,
    size_t row,
    size_t column,
    enum qa_algebra_mode mode,
    struct qa_stats *stats
) {
    double complex factor = root3(0);
    for (size_t shared = 0U; shared < QA_Q; ++shared) {
        size_t right_row = row;
        size_t right_column = shared;
        size_t left_row = shared;
        size_t left_column = column;
        if (mode == QA_ALGEBRA_REVERSE_ORDER) {
            right_row = shared;
            right_column = column;
            left_row = row;
            left_column = shared;
        }
        const double complex right = qa_read(
            carrier,
            qa_matrix_cell(
                descriptor->right_start, right_row, right_column
            ),
            stats
        );
        const double complex left = qa_read(
            carrier,
            qa_matrix_cell(
                descriptor->left_start, left_row, left_column
            ),
            stats
        );
        factor = qa_xor(
            factor, qa_and(right, left, stats), mode, stats
        );
    }
    return factor;
}

static double complex qa_bias_factor(
    const struct carrier *carrier,
    const struct qa_descriptor *descriptor,
    size_t row,
    enum qa_algebra_mode mode,
    struct qa_stats *stats
) {
    double complex factor = qa_read(
        carrier,
        qa_bias_cell(
            mode == QA_ALGEBRA_REVERSE_ORDER
                ? descriptor->left_start
                : descriptor->right_start,
            row
        ),
        stats
    );
    for (size_t shared = 0U; shared < QA_Q; ++shared) {
        double complex matrix;
        double complex bias;
        if (mode == QA_ALGEBRA_REVERSE_ORDER) {
            matrix = qa_read(
                carrier,
                qa_matrix_cell(
                    descriptor->left_start, row, shared
                ),
                stats
            );
            bias = qa_read(
                carrier,
                qa_bias_cell(descriptor->right_start, shared),
                stats
            );
        } else {
            matrix = qa_read(
                carrier,
                qa_matrix_cell(
                    descriptor->right_start, row, shared
                ),
                stats
            );
            bias = qa_read(
                carrier,
                qa_bias_cell(descriptor->left_start, shared),
                stats
            );
        }
        factor = qa_xor(
            factor, qa_and(matrix, bias, stats), mode, stats
        );
    }
    return factor;
}

static void qa_apply_compose(
    struct carrier *carrier,
    const struct qa_descriptor *descriptor,
    enum qa_algebra_mode mode,
    int inverse,
    int wrong_first_factor,
    struct qa_stats *stats
) {
    ++stats->native_compose_calls;
    for (size_t row = 0U; row < QA_Q; ++row) {
        for (size_t column = 0U; column < QA_Q; ++column) {
            double complex factor = qa_matrix_factor(
                carrier, descriptor, row, column, mode, stats
            );
            if (
                inverse
                && wrong_first_factor
                && row == 0U
                && column == 0U
            ) {
                factor = lock_f3_phase(factor * root3(1));
            }
            qa_update_cell(
                carrier,
                qa_matrix_cell(descriptor->output_start, row, column),
                inverse ? conj(factor) : factor,
                stats
            );
        }
        const double complex factor = qa_bias_factor(
            carrier, descriptor, row, mode, stats
        );
        qa_update_cell(
            carrier,
            qa_bias_cell(descriptor->output_start, row),
            inverse ? conj(factor) : factor,
            stats
        );
    }
    if (inverse) {
        ++stats->inverse_factor_recomputations;
    }
}

static void qa_encode_relation(
    struct carrier *carrier,
    const struct qa_program *program,
    size_t relation,
    size_t start,
    int inverse,
    struct qa_stats *stats
) {
    for (size_t row = 0U; row < QA_Q; ++row) {
        for (size_t column = 0U; column < QA_Q; ++column) {
            const double complex factor = root3(
                program->matrix[relation][row][column]
            );
            qa_update_cell(
                carrier,
                qa_matrix_cell(start, row, column),
                inverse ? conj(factor) : factor,
                stats
            );
        }
        const double complex factor = root3(program->bias[relation][row]);
        qa_update_cell(
            carrier,
            qa_bias_cell(start, row),
            inverse ? conj(factor) : factor,
            stats
        );
    }
}

static void qa_copy_block(
    struct carrier *carrier,
    size_t source,
    size_t target,
    int inverse,
    struct qa_stats *stats
) {
    for (size_t index = 0U; index < QA_BLOCK_CELLS; ++index) {
        const double complex factor = qa_read(
            carrier, source + index, stats
        );
        qa_update_cell(
            carrier,
            target + index,
            inverse ? conj(factor) : factor,
            stats
        );
    }
}

static struct qa_boundary qa_latch_boundary(
    const struct carrier *carrier,
    struct qa_stats *stats
) {
    struct qa_boundary boundary = {
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t index = 0U; index < QA_BLOCK_CELLS; ++index) {
        double error = 0.0;
        boundary.coefficient[index] = decode_root(
            relative(carrier, QA_BOUNDARY_START + index), &error
        );
        if (error > boundary.maximum_root_error) {
            boundary.maximum_root_error = error;
        }
        const unsigned char byte =
            (unsigned char)boundary.coefficient[index];
        boundary.hash = hash_bytes(boundary.hash, &byte, 1U);
        ++stats->boundary_decodes;
    }
    return boundary;
}

static struct qa_execution qa_execute_ordered(
    struct carrier *carrier,
    const struct qa_program *program,
    enum qa_algebra_mode algebra_mode,
    enum qa_restore_mode restore_mode,
    int right_grouped
) {
    if (carrier == NULL || program == NULL) {
        fail("null carrier or quotient-affine program");
    }
    struct qa_execution execution = {0};
    struct carrier borrowed = snapshot_carrier(carrier);
    const struct qa_descriptor child = {
        .left_start = right_grouped ? QA_G_START : QA_F_START,
        .right_start = right_grouped ? QA_K_START : QA_G_START,
        .output_start = QA_H_START
    };
    const struct qa_descriptor parent = {
        .left_start = right_grouped ? QA_F_START : QA_H_START,
        .right_start = right_grouped ? QA_H_START : QA_K_START,
        .output_start = QA_Z_START
    };

    qa_encode_relation(
        carrier, program, 0U, QA_F_START, 0, &execution.stats
    );
    qa_encode_relation(
        carrier, program, 1U, QA_G_START, 0, &execution.stats
    );
    qa_encode_relation(
        carrier, program, 2U, QA_K_START, 0, &execution.stats
    );
    qa_apply_compose(
        carrier, &child, algebra_mode, 0, 0, &execution.stats
    );
    qa_apply_compose(
        carrier, &parent, algebra_mode, 0, 0, &execution.stats
    );
    qa_copy_block(
        carrier, QA_Z_START, QA_BOUNDARY_START, 0, &execution.stats
    );
    execution.boundary = qa_latch_boundary(carrier, &execution.stats);
    execution.displacement_l2 = displacement(carrier, &borrowed);

    if (restore_mode == QA_RESTORE_SNAPSHOT) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        execution.snapshot_loaded = 1;
    } else {
        qa_copy_block(
            carrier, QA_Z_START, QA_BOUNDARY_START, 1, &execution.stats
        );
        if (restore_mode == QA_RESTORE_REORDERED) {
            qa_apply_compose(
                carrier, &child, algebra_mode, 1, 0, &execution.stats
            );
            qa_apply_compose(
                carrier, &parent, algebra_mode, 1, 0, &execution.stats
            );
        } else {
            if (restore_mode != QA_RESTORE_MISSING_PARENT) {
                qa_apply_compose(
                    carrier,
                    &parent,
                    algebra_mode,
                    1,
                    restore_mode == QA_RESTORE_WRONG_PARENT,
                    &execution.stats
                );
            }
            qa_apply_compose(
                carrier, &child, algebra_mode, 1, 0, &execution.stats
            );
        }
        qa_encode_relation(
            carrier, program, 2U, QA_K_START, 1, &execution.stats
        );
        qa_encode_relation(
            carrier, program, 1U, QA_G_START, 1, &execution.stats
        );
        qa_encode_relation(
            carrier, program, 0U, QA_F_START, 1, &execution.stats
        );
    }
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
    execution.stats.restoration_cell_checks += carrier->cells;
    free_carrier(&borrowed);
    return execution;
}

static struct qa_execution qa_execute(
    struct carrier *carrier,
    const struct qa_program *program,
    enum qa_algebra_mode algebra_mode,
    enum qa_restore_mode restore_mode
) {
    return qa_execute_ordered(
        carrier, program, algebra_mode, restore_mode, 0
    );
}

static void qa_print_coefficients(const struct qa_boundary *boundary) {
    putchar('[');
    for (size_t index = 0U; index < QA_BLOCK_CELLS; ++index) {
        if (index != 0U) {
            putchar(',');
        }
        printf("%d", boundary->coefficient[index]);
    }
    putchar(']');
}

static void qa_print_execution(
    const char *mode,
    const struct qa_program *program,
    const struct qa_execution *execution,
    const char *restoration_path
) {
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"BOUNDED_PROJECTED_AFFINE_PHASE_RELATION_"
        "COMPOSITION_WITH_COMPACT_FULL_BOUNDARY\","
        "\"relation_family\":\"PROJECTED_AFFINE_F2\","
        "\"quotient_bits\":%u,"
        "\"fibre_bits\":%u,"
        "\"port_bits\":%u,"
        "\"genuinely_many_to_many\":true,"
        "\"targets_per_source\":%llu,"
        "\"sources_per_target\":%llu,"
        "\"factor_coefficients\":%u,"
        "\"dense_pair_table_exponent\":%u,"
        "\"dense_multiaffine_coefficient_exponent\":%u,"
        "\"full_boundary_materialized\":true,"
        "\"sparse_boundary_query\":false,"
        "\"tuple_slots\":0,"
        "\"assignment_slots\":0,"
        "\"witness_slots\":0,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"serialized_intermediate_coefficients\":0,"
        "\"source_fnv1a64\":\"%016llx\","
        "\"public_program_coefficients\":%u,"
        "\"program_storage_bytes_current_abi\":%u,"
        "\"coexisting_program_storage_bytes_current_abi\":%u,"
        "\"compiled_descriptor_bytes_current_abi\":%u,"
        "\"execution_result_bytes_current_abi\":%u,"
        "\"boundary_coefficients\":",
        mode,
        (unsigned)QA_Q,
        (unsigned)QA_FIBRE,
        (unsigned)(QA_Q + QA_FIBRE),
        (unsigned long long)(UINT64_C(1) << QA_FIBRE),
        (unsigned long long)(UINT64_C(1) << QA_FIBRE),
        (unsigned)QA_BLOCK_CELLS,
        (unsigned)(2U * (QA_Q + QA_FIBRE)),
        (unsigned)(2U * (QA_Q + QA_FIBRE)),
        (unsigned long long)program->source_hash,
        (unsigned)(QA_RELATIONS * QA_BLOCK_CELLS),
        (unsigned)sizeof(struct qa_program),
        (unsigned)(2U * sizeof(struct qa_program)),
        (unsigned)(2U * sizeof(struct qa_descriptor)),
        (unsigned)sizeof(struct qa_execution)
    );
    qa_print_coefficients(&execution->boundary);
    printf(
        ",\"boundary_fnv1a64\":\"%016llx\","
        "\"maximum_root_error\":%.12g,"
        "\"restoration_path\":\"%s\","
        "\"snapshot_loaded\":%s,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_max_abs\":%.12g,"
        "\"carrier_cells\":%u,"
        "\"live_carrier_complex_values\":%u,"
        "\"live_carrier_bytes\":%u,"
        "\"comparison_snapshot_complex_values\":%u,"
        "\"carrier_and_snapshot_complex_values\":%u,"
        "\"carrier_and_snapshot_heap_bytes\":%u,"
        "\"native_compose_calls\":%llu,"
        "\"native_and_products\":%llu,"
        "\"native_xor_accumulations\":%llu,"
        "\"symbol_product_calls\":%llu,"
        "\"carrier_reads\":%llu,"
        "\"phase_cell_updates\":%llu,"
        "\"inverse_factor_recomputations\":%llu,"
        "\"boundary_decodes\":%llu,"
        "\"restoration_cell_checks\":%llu}\n",
        (unsigned long long)execution->boundary.hash,
        execution->boundary.maximum_root_error,
        restoration_path,
        execution->snapshot_loaded ? "true" : "false",
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->integrity_max_abs,
        (unsigned)QA_CARRIER_CELLS,
        (unsigned)(2U * QA_CARRIER_CELLS),
        (unsigned)(2U * QA_CARRIER_CELLS * sizeof(double complex)),
        (unsigned)(2U * QA_CARRIER_CELLS),
        (unsigned)(4U * QA_CARRIER_CELLS),
        (unsigned)(
            4U * QA_CARRIER_CELLS * sizeof(double complex)
        ),
        (unsigned long long)execution->stats.native_compose_calls,
        (unsigned long long)execution->stats.native_and_products,
        (unsigned long long)execution->stats.native_xor_accumulations,
        (unsigned long long)execution->stats.symbol_product_calls,
        (unsigned long long)execution->stats.carrier_reads,
        (unsigned long long)execution->stats.phase_cell_updates,
        (unsigned long long)execution->stats.inverse_factor_recomputations,
        (unsigned long long)execution->stats.boundary_decodes,
        (unsigned long long)execution->stats.restoration_cell_checks
    );
}

static int qa_boundary_equal(
    const struct qa_boundary *left,
    const struct qa_boundary *right
) {
    return memcmp(
        left->coefficient,
        right->coefficient,
        sizeof(left->coefficient)
    ) == 0;
}

static void qa_set_group_relation(
    struct qa_program *program,
    size_t relation,
    size_t index
) {
    static const unsigned char matrix[6][2][2] = {
        {{1U, 0U}, {0U, 1U}},
        {{0U, 1U}, {1U, 0U}},
        {{1U, 1U}, {0U, 1U}},
        {{1U, 0U}, {1U, 1U}},
        {{0U, 1U}, {1U, 1U}},
        {{1U, 1U}, {1U, 0U}}
    };
    if (QA_Q != 2U || relation >= QA_RELATIONS || index >= 24U) {
        fail("invalid quotient-affine group calibration");
    }
    const size_t matrix_index = index / 4U;
    const size_t bias = index % 4U;
    for (size_t row = 0U; row < 2U; ++row) {
        for (size_t column = 0U; column < 2U; ++column) {
            program->matrix[relation][row][column] =
                matrix[matrix_index][row][column];
        }
        program->bias[relation][row] =
            (unsigned char)((bias >> row) & 1U);
    }
}

static int qa_group_boundary_valid(const struct qa_boundary *boundary) {
    if (QA_Q != 2U) {
        return 0;
    }
    for (size_t index = 0U; index < QA_BLOCK_CELLS; ++index) {
        if (
            boundary->coefficient[index] != 0
            && boundary->coefficient[index] != 1
        ) {
            return 0;
        }
    }
    const int determinant =
        (
            boundary->coefficient[0] * boundary->coefficient[3]
        ) ^ (
            boundary->coefficient[1] * boundary->coefficient[2]
        );
    return determinant == 1;
}

static int qa_run_group_calibration(void) {
    if (QA_Q != 2U || QA_FIBRE != 1U) {
        fail("group calibration requires quotient 2 and fibre 1");
    }
    const struct process shape = {.carrier_cells = QA_CARRIER_CELLS};
    size_t pair_closures = 0U;
    size_t associativity_triples = 0U;
    double maximum_restoration_error = 0.0;

    for (size_t left = 0U; left < 24U; ++left) {
        for (size_t right = 0U; right < 24U; ++right) {
            struct qa_program program = {0};
            qa_set_group_relation(&program, 0U, left);
            qa_set_group_relation(&program, 1U, right);
            qa_set_group_relation(&program, 2U, 0U);
            struct carrier carrier = make_carrier(&shape, 6311);
            const struct qa_execution execution = qa_execute(
                &carrier,
                &program,
                QA_ALGEBRA_CORRECT,
                QA_RESTORE_CORRECT
            );
            free_carrier(&carrier);
            if (
                !qa_group_boundary_valid(&execution.boundary)
                || execution.restoration_max_abs > RESTORATION_TOLERANCE
            ) {
                return 1;
            }
            if (
                execution.restoration_max_abs
                > maximum_restoration_error
            ) {
                maximum_restoration_error =
                    execution.restoration_max_abs;
            }
            ++pair_closures;
        }
    }

    for (size_t first = 0U; first < 24U; ++first) {
        for (size_t second = 0U; second < 24U; ++second) {
            for (size_t third = 0U; third < 24U; ++third) {
                struct qa_program program = {0};
                qa_set_group_relation(&program, 0U, first);
                qa_set_group_relation(&program, 1U, second);
                qa_set_group_relation(&program, 2U, third);
                struct carrier left_carrier = make_carrier(&shape, 6313);
                struct carrier right_carrier = make_carrier(&shape, 6313);
                const struct qa_execution left_grouped = qa_execute_ordered(
                    &left_carrier,
                    &program,
                    QA_ALGEBRA_CORRECT,
                    QA_RESTORE_CORRECT,
                    0
                );
                const struct qa_execution right_grouped =
                    qa_execute_ordered(
                        &right_carrier,
                        &program,
                        QA_ALGEBRA_CORRECT,
                        QA_RESTORE_CORRECT,
                        1
                    );
                free_carrier(&left_carrier);
                free_carrier(&right_carrier);
                if (
                    !qa_boundary_equal(
                        &left_grouped.boundary,
                        &right_grouped.boundary
                    )
                    || left_grouped.restoration_max_abs
                        > RESTORATION_TOLERANCE
                    || right_grouped.restoration_max_abs
                        > RESTORATION_TOLERANCE
                ) {
                    return 1;
                }
                if (
                    left_grouped.restoration_max_abs
                    > maximum_restoration_error
                ) {
                    maximum_restoration_error =
                        left_grouped.restoration_max_abs;
                }
                if (
                    right_grouped.restoration_max_abs
                    > maximum_restoration_error
                ) {
                    maximum_restoration_error =
                        right_grouped.restoration_max_abs;
                }
                ++associativity_triples;
            }
        }
    }
    printf(
        "{\"mode\":\"q2-affine-group-calibration\","
        "\"relation_count\":24,"
        "\"ordered_pair_closures\":%zu,"
        "\"associativity_triples\":%zu,"
        "\"retained_relation_lookup_entries\":0,"
        "\"tuple_slots\":0,"
        "\"assignment_slots\":0,"
        "\"witness_slots\":0,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"maximum_restoration_error\":%.12g}\n",
        pair_closures,
        associativity_triples,
        maximum_restoration_error
    );
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--project-intermediate") == 0) {
        fail("only the compact full final boundary may be projected");
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        fail("null carrier or quotient-affine program");
    }
    if (argc == 2 && strcmp(argv[1], "--exhaustive-group") == 0) {
        return qa_run_group_calibration();
    }
    if (argc != 1) {
        fail("usage: algebraic_quotient_affine_phase");
    }

    const struct qa_program primary = qa_make_program(0U);
    const struct qa_program reuse = qa_make_program(1U);
    const struct process shape = {.carrier_cells = QA_CARRIER_CELLS};
    struct carrier carrier = make_carrier(&shape, 6301);
    const struct qa_execution nominal = qa_execute(
        &carrier, &primary, QA_ALGEBRA_CORRECT, QA_RESTORE_CORRECT
    );
    const struct qa_execution reused = qa_execute(
        &carrier, &reuse, QA_ALGEBRA_CORRECT, QA_RESTORE_CORRECT
    );
    double repeated_max_abs = nominal.restoration_max_abs;
    if (reused.restoration_max_abs > repeated_max_abs) {
        repeated_max_abs = reused.restoration_max_abs;
    }
    for (size_t cycle = 0U; cycle < QA_REPEAT_CYCLES; ++cycle) {
        const struct qa_execution repeated = qa_execute(
            &carrier,
            cycle % 2U == 0U ? &primary : &reuse,
            QA_ALGEBRA_CORRECT,
            QA_RESTORE_CORRECT
        );
        if (repeated.restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = repeated.restoration_max_abs;
        }
    }
    free_carrier(&carrier);

    struct qa_execution control[5];
    const enum qa_algebra_mode control_algebra[5] = {
        QA_ALGEBRA_CORRECT,
        QA_ALGEBRA_CORRECT,
        QA_ALGEBRA_CORRECT,
        QA_ALGEBRA_F3_SUM,
        QA_ALGEBRA_REVERSE_ORDER
    };
    const enum qa_restore_mode control_restore[5] = {
        QA_RESTORE_WRONG_PARENT,
        QA_RESTORE_MISSING_PARENT,
        QA_RESTORE_REORDERED,
        QA_RESTORE_CORRECT,
        QA_RESTORE_CORRECT
    };
    for (size_t index = 0U; index < 5U; ++index) {
        carrier = make_carrier(&shape, 6301);
        control[index] = qa_execute(
            &carrier,
            &primary,
            control_algebra[index],
            control_restore[index]
        );
        free_carrier(&carrier);
    }
    carrier = make_carrier(&shape, 6301);
    const struct qa_execution snapshot = qa_execute(
        &carrier, &primary, QA_ALGEBRA_CORRECT, QA_RESTORE_SNAPSHOT
    );
    free_carrier(&carrier);

    qa_print_execution(
        "projected-affine-primary",
        &primary,
        &nominal,
        "actual_inverse"
    );
    qa_print_execution(
        "projected-affine-reuse",
        &reuse,
        &reused,
        "actual_inverse"
    );
    qa_print_execution(
        "projected-affine-snapshot",
        &primary,
        &snapshot,
        "snapshot_reload"
    );
    printf(
        "{\"mode\":\"controls\","
        "\"wrong_parent_inverse\":%s,"
        "\"missing_parent_inverse\":%s,"
        "\"reordered_noncommuting_inverse\":%s,"
        "\"f3_sum_changes_full_boundary\":%s,"
        "\"reverse_order_changes_full_boundary\":%s,"
        "\"altered_algebra_histories_restore\":%s,"
        "\"snapshot_restores\":%s,"
        "\"actual_restored_cross_program_reuse\":%s,"
        "\"same_carrier_transactions\":%u,"
        "\"repeated_reuse_max_restoration_abs\":%.12g}\n",
        control[0].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        control[1].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        control[2].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        !qa_boundary_equal(&nominal.boundary, &control[3].boundary)
            ? "true" : "false",
        !qa_boundary_equal(&nominal.boundary, &control[4].boundary)
            ? "true" : "false",
        control[3].restoration_max_abs <= RESTORATION_TOLERANCE
            && control[4].restoration_max_abs <= RESTORATION_TOLERANCE
            ? "true" : "false",
        snapshot.restoration_max_abs <= RESTORATION_TOLERANCE
            && snapshot.snapshot_loaded
            ? "true" : "false",
        reused.restoration_max_abs <= RESTORATION_TOLERANCE
            && !qa_boundary_equal(&nominal.boundary, &reused.boundary)
            ? "true" : "false",
        (unsigned)(QA_REPEAT_CYCLES + 2U),
        repeated_max_abs
    );

    const int pass =
        nominal.boundary.maximum_root_error <= ROOT_TOLERANCE
        && reused.boundary.maximum_root_error <= ROOT_TOLERANCE
        && nominal.restoration_max_abs <= RESTORATION_TOLERANCE
        && reused.restoration_max_abs <= RESTORATION_TOLERANCE
        && control[0].restoration_max_abs >= CONTROL_MINIMUM
        && control[1].restoration_max_abs >= CONTROL_MINIMUM
        && control[2].restoration_max_abs >= CONTROL_MINIMUM
        && !qa_boundary_equal(&nominal.boundary, &control[3].boundary)
        && !qa_boundary_equal(&nominal.boundary, &control[4].boundary)
        && control[3].restoration_max_abs <= RESTORATION_TOLERANCE
        && control[4].restoration_max_abs <= RESTORATION_TOLERANCE
        && snapshot.restoration_max_abs <= RESTORATION_TOLERANCE
        && snapshot.snapshot_loaded
        && repeated_max_abs <= RESTORATION_TOLERANCE;
    return pass ? 0 : 1;
}
