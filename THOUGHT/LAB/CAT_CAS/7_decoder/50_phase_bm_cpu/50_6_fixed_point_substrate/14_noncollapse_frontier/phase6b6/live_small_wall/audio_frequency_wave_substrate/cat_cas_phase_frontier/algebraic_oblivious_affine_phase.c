#define _POSIX_C_SOURCE 200809L

/*
 * Mutable prototype: coefficient-oblivious projection of general affine
 * GF(2) relation equations using only resident F3 phase symbols.
 *
 * Width is deliberately fixed at two while the reversible row-reduction law
 * is calibrated.  Loop bounds and memory addresses depend only on fixed
 * public capacity.  Pivot, rank, active-row, and elimination controls remain
 * phase symbols and never select a host pivot, loop bound, or address.
 */

#define main algebraic_series_parallel_embedded_main
#include "algebraic_series_parallel_phase.c"
#undef main

#define GA_WIDTH 2U
#define GA_ROWS 4U
#define GA_ROW_CELLS 6U
#define GA_BLOCK_CELLS (GA_ROWS * GA_ROW_CELLS + 1U)
#define GA_RELATION_BLOCKS 6U
#define GA_JOINT_ROWS (2U * GA_ROWS)
#define GA_JOINT_COLUMNS (3U * GA_WIDTH + 1U)
#define GA_REDUCTION_COLUMNS (3U * GA_WIDTH)

#define GA_F_START 0U
#define GA_G_START (1U * GA_BLOCK_CELLS)
#define GA_K_START (2U * GA_BLOCK_CELLS)
#define GA_H_START (3U * GA_BLOCK_CELLS)
#define GA_Z_START (4U * GA_BLOCK_CELLS)
#define GA_BOUNDARY_START (5U * GA_BLOCK_CELLS)
#define GA_WORK_START (GA_RELATION_BLOCKS * GA_BLOCK_CELLS)
#define GA_JOINT_START GA_WORK_START
#define GA_ACTIVE_START (GA_JOINT_START + GA_JOINT_ROWS * GA_JOINT_COLUMNS)
#define GA_Y_PIVOT_START (GA_ACTIVE_START + GA_JOINT_ROWS)
#define GA_EXT_PIVOT_START (GA_Y_PIVOT_START + GA_WIDTH)
#define GA_SELECT_START (GA_EXT_PIVOT_START + 2U * GA_WIDTH)
#define GA_ELIM_START \
    (GA_SELECT_START + GA_REDUCTION_COLUMNS * GA_JOINT_ROWS)
#define GA_INPUT_EMPTY_CELL \
    (GA_ELIM_START + GA_REDUCTION_COLUMNS * GA_JOINT_ROWS)
#define GA_INCONSISTENCY_CELL (GA_INPUT_EMPTY_CELL + 1U)
#define GA_CARRIER_CELLS (GA_INCONSISTENCY_CELL + 1U)
#define GA_REPEAT_CYCLES 64U
#define GA_SEMANTIC_VARIANTS 9U
#define GA_WORKSPACE_TOLERANCE 2.0e-12

enum ga_restore_mode {
    GA_RESTORE_CORRECT = 0,
    GA_RESTORE_WRONG_PARENT = 1,
    GA_RESTORE_MISSING_PARENT = 2,
    GA_RESTORE_REORDERED = 3,
    GA_RESTORE_SNAPSHOT = 4
};

struct ga_program {
    unsigned char relation[3][GA_BLOCK_CELLS];
};

struct ga_descriptor {
    size_t left_start;
    size_t right_start;
    size_t output_start;
};

struct ga_stats {
    uint64_t compose_calls;
    uint64_t phase_ands;
    uint64_t phase_xors;
    uint64_t phase_ors;
    uint64_t phase_nots;
    uint64_t conditional_swaps;
    uint64_t conditional_row_adds;
    uint64_t phase_cell_updates;
    uint64_t carrier_reads;
    uint64_t boundary_decodes;
    uint64_t workspace_tolerance_checks;
};

struct ga_boundary {
    int coefficient[GA_BLOCK_CELLS];
    uint64_t hash;
    double maximum_root_error;
};

struct ga_execution {
    struct ga_boundary boundary;
    struct ga_stats stats;
    double displacement_l2;
    double restoration_max_abs;
    double integrity_max_abs;
    int workspace_cleared;
    int snapshot_loaded;
};

static size_t ga_relation_cell(
    size_t start,
    size_t row,
    size_t field
) {
    return start + row * GA_ROW_CELLS + field;
}

static size_t ga_empty_cell(size_t start) {
    return start + GA_ROWS * GA_ROW_CELLS;
}

static size_t ga_joint_cell(size_t row, size_t column) {
    return GA_JOINT_START + row * GA_JOINT_COLUMNS + column;
}

static size_t ga_active_cell(size_t row) {
    return GA_ACTIVE_START + row;
}

static size_t ga_pivot_cell(size_t stage) {
    return stage < GA_WIDTH
        ? GA_Y_PIVOT_START + stage
        : GA_EXT_PIVOT_START + stage - GA_WIDTH;
}

static size_t ga_select_cell(size_t stage, size_t row) {
    return GA_SELECT_START + stage * GA_JOINT_ROWS + row;
}

static size_t ga_elim_cell(size_t stage, size_t row) {
    return GA_ELIM_START + stage * GA_JOINT_ROWS + row;
}

static double complex ga_read(
    const struct carrier *carrier,
    size_t cell,
    struct ga_stats *stats
) {
    ++stats->carrier_reads;
    return relative(carrier, cell);
}

static void ga_update(
    struct carrier *carrier,
    size_t cell,
    double complex factor,
    struct ga_stats *stats
) {
    multiply_cell(carrier, cell, factor);
    ++stats->phase_cell_updates;
}

static double complex ga_and(
    double complex left,
    double complex right,
    struct ga_stats *stats
) {
    ++stats->phase_ands;
    return symbol_product(left, right);
}

static double complex ga_xor(
    double complex left,
    double complex right,
    struct ga_stats *stats
) {
    ++stats->phase_xors;
    return lock_f3_phase(
        left * right * symbol_product(left, right)
    );
}

static double complex ga_or(
    double complex left,
    double complex right,
    struct ga_stats *stats
) {
    ++stats->phase_ors;
    const double complex product = symbol_product(left, right);
    return lock_f3_phase(left * right * product * product);
}

static double complex ga_not(
    double complex value,
    struct ga_stats *stats
) {
    ++stats->phase_nots;
    return lock_f3_phase(root3(1) * conj(value));
}

static double complex ga_gate(
    double complex active,
    double complex value,
    struct ga_stats *stats
) {
    return ga_and(active, value, stats);
}

static void ga_set_equation(
    unsigned char relation[GA_BLOCK_CELLS],
    size_t row,
    const unsigned char coefficient[4],
    unsigned char constant
) {
    relation[row * GA_ROW_CELLS] = 1U;
    memcpy(
        &relation[row * GA_ROW_CELLS + 1U],
        coefficient,
        4U
    );
    relation[row * GA_ROW_CELLS + 5U] = constant;
}

static struct ga_program ga_make_program(size_t variant) {
    struct ga_program program = {0};
    if (variant == 0U) {
        static const unsigned char f[4] = {1U, 0U, 1U, 1U};
        static const unsigned char g[4] = {1U, 1U, 1U, 0U};
        static const unsigned char k0[4] = {1U, 0U, 1U, 1U};
        static const unsigned char k1[4] = {0U, 1U, 0U, 1U};
        ga_set_equation(program.relation[0], 3U, f, 0U);
        ga_set_equation(program.relation[1], 2U, g, 1U);
        ga_set_equation(program.relation[2], 1U, k0, 0U);
        ga_set_equation(program.relation[2], 3U, k1, 0U);
    } else if (variant == 1U) {
        static const unsigned char f[4] = {0U, 1U, 1U, 0U};
        static const unsigned char g[4] = {1U, 0U, 0U, 1U};
        static const unsigned char k[4] = {0U, 1U, 1U, 0U};
        ga_set_equation(program.relation[0], 2U, f, 1U);
        ga_set_equation(program.relation[1], 3U, g, 0U);
        ga_set_equation(program.relation[2], 1U, k, 1U);
    } else if (variant == 2U || variant == 3U) {
        static const unsigned char f[4] = {1U, 0U, 1U, 1U};
        static const unsigned char g[4] = {1U, 1U, 1U, 0U};
        static const unsigned char k0[4] = {1U, 0U, 1U, 1U};
        static const unsigned char k1[4] = {0U, 1U, 0U, 1U};
        ga_set_equation(program.relation[0], 0U, f, 0U);
        ga_set_equation(program.relation[1], 0U, g, 1U);
        ga_set_equation(program.relation[2], 0U, k1, 0U);
        ga_set_equation(program.relation[2], 2U, k0, 0U);
        if (variant == 3U) {
            ga_set_equation(program.relation[0], 3U, f, 0U);
            ga_set_equation(program.relation[1], 2U, g, 1U);
        }
    } else if (variant == 5U) {
        static const unsigned char left_y0[4] = {0U, 0U, 1U, 0U};
        static const unsigned char right_y0[4] = {1U, 0U, 0U, 0U};
        ga_set_equation(program.relation[0], 3U, left_y0, 0U);
        ga_set_equation(program.relation[1], 0U, right_y0, 0U);
        ga_set_equation(program.relation[1], 2U, right_y0, 0U);
    } else if (variant == 6U) {
        program.relation[0][GA_ROWS * GA_ROW_CELLS] = 1U;
    } else if (variant == 7U) {
        static const unsigned char left_y0[4] = {0U, 0U, 1U, 0U};
        static const unsigned char right_y0[4] = {1U, 0U, 0U, 0U};
        ga_set_equation(program.relation[0], 3U, left_y0, 0U);
        ga_set_equation(program.relation[1], 0U, right_y0, 1U);
        ga_set_equation(program.relation[1], 1U, right_y0, 1U);
    } else if (variant == 8U) {
        static const unsigned char x0[4] = {1U, 0U, 0U, 0U};
        static const unsigned char x1[4] = {0U, 1U, 0U, 0U};
        static const unsigned char y0[4] = {0U, 0U, 1U, 0U};
        static const unsigned char y1[4] = {0U, 0U, 0U, 1U};
        ga_set_equation(program.relation[0], 0U, x0, 0U);
        ga_set_equation(program.relation[0], 1U, x1, 0U);
        ga_set_equation(program.relation[0], 2U, y0, 0U);
        ga_set_equation(program.relation[0], 3U, y1, 0U);
        ga_set_equation(program.relation[2], 0U, y0, 0U);
        ga_set_equation(program.relation[2], 3U, y1, 0U);
    } else if (variant != 4U) {
        fail("unknown oblivious affine program variant");
    }
    return program;
}

static int ga_workspace_is_zero(const struct carrier *carrier) {
    for (size_t cell = GA_WORK_START; cell < GA_CARRIER_CELLS; ++cell) {
        if (
            cabs(relative(carrier, cell) - root3(0))
            > GA_WORKSPACE_TOLERANCE
        ) {
            return 0;
        }
    }
    return 1;
}

static void ga_clear_actual(
    struct carrier *carrier,
    size_t cell,
    struct ga_stats *stats
) {
    const double complex value = ga_read(carrier, cell, stats);
    ga_update(carrier, cell, conj(value), stats);
}

static double complex ga_row_eligible(
    const struct carrier *carrier,
    size_t stage,
    size_t row,
    int for_selection,
    struct ga_stats *stats
) {
    double complex eligible = ga_read(
        carrier, ga_active_cell(row), stats
    );
    if (stage >= GA_WIDTH) {
        for (size_t prior = 0U; prior < GA_WIDTH; ++prior) {
            if (row == prior) {
                eligible = ga_and(
                    eligible,
                    ga_not(
                        ga_read(carrier, ga_pivot_cell(prior), stats),
                        stats
                    ),
                    stats
                );
            }
        }
    }
    if (for_selection) {
        const size_t target = stage;
        for (size_t prior = 0U; prior < target; ++prior) {
            if (row == prior) {
                eligible = ga_and(
                    eligible,
                    ga_not(
                        ga_read(carrier, ga_pivot_cell(prior), stats),
                        stats
                    ),
                    stats
                );
            }
        }
    }
    return eligible;
}

static void ga_conditional_swap_rows(
    struct carrier *carrier,
    size_t first,
    size_t second,
    double complex control,
    struct ga_stats *stats
) {
    ++stats->conditional_swaps;
    if (first == second) {
        return;
    }
    double complex first_value[GA_JOINT_COLUMNS + 1U];
    double complex second_value[GA_JOINT_COLUMNS + 1U];
    double complex first_target[GA_JOINT_COLUMNS + 1U];
    double complex second_target[GA_JOINT_COLUMNS + 1U];
    for (size_t field = 0U; field <= GA_JOINT_COLUMNS; ++field) {
        const size_t first_cell = field == GA_JOINT_COLUMNS
            ? ga_active_cell(first)
            : ga_joint_cell(first, field);
        const size_t second_cell = field == GA_JOINT_COLUMNS
            ? ga_active_cell(second)
            : ga_joint_cell(second, field);
        first_value[field] = ga_read(carrier, first_cell, stats);
        second_value[field] = ga_read(carrier, second_cell, stats);
        const double complex difference = ga_xor(
            first_value[field], second_value[field], stats
        );
        const double complex toggle = ga_and(
            control, difference, stats
        );
        first_target[field] = ga_xor(
            first_value[field], toggle, stats
        );
        second_target[field] = ga_xor(
            second_value[field], toggle, stats
        );
    }
    for (size_t field = 0U; field <= GA_JOINT_COLUMNS; ++field) {
        const size_t first_cell = field == GA_JOINT_COLUMNS
            ? ga_active_cell(first)
            : ga_joint_cell(first, field);
        const size_t second_cell = field == GA_JOINT_COLUMNS
            ? ga_active_cell(second)
            : ga_joint_cell(second, field);
        ga_update(
            carrier,
            first_cell,
            first_target[field] * conj(first_value[field]),
            stats
        );
        ga_update(
            carrier,
            second_cell,
            second_target[field] * conj(second_value[field]),
            stats
        );
    }
}

static void ga_conditional_row_add(
    struct carrier *carrier,
    size_t source,
    size_t target,
    double complex control,
    struct ga_stats *stats
) {
    ++stats->conditional_row_adds;
    for (size_t column = 0U; column < GA_JOINT_COLUMNS; ++column) {
        const size_t source_cell = ga_joint_cell(source, column);
        const size_t target_cell = ga_joint_cell(target, column);
        const double complex source_value = ga_read(
            carrier, source_cell, stats
        );
        const double complex target_value = ga_read(
            carrier, target_cell, stats
        );
        const double complex addend = ga_and(
            control, source_value, stats
        );
        const double complex desired = ga_xor(
            target_value, addend, stats
        );
        ga_update(
            carrier,
            target_cell,
            desired * conj(target_value),
            stats
        );
    }
}

static void ga_encode_joint(
    struct carrier *carrier,
    const struct ga_descriptor *descriptor,
    struct ga_stats *stats
) {
    const double complex left_empty = ga_read(
        carrier, ga_empty_cell(descriptor->left_start), stats
    );
    const double complex right_empty = ga_read(
        carrier, ga_empty_cell(descriptor->right_start), stats
    );
    const double complex input_empty = ga_or(
        left_empty, right_empty, stats
    );
    ga_update(
        carrier, GA_INPUT_EMPTY_CELL, input_empty, stats
    );
    const double complex enabled = ga_not(input_empty, stats);
    for (size_t row = 0U; row < GA_JOINT_ROWS; ++row) {
        const int from_right = row >= GA_ROWS;
        const size_t local = row % GA_ROWS;
        const size_t source = from_right
            ? descriptor->right_start
            : descriptor->left_start;
        const double complex active = ga_and(
            enabled,
            ga_read(
                carrier, ga_relation_cell(source, local, 0U), stats
            ),
            stats
        );
        ga_update(carrier, ga_active_cell(row), active, stats);
        for (size_t column = 0U; column < GA_JOINT_COLUMNS; ++column) {
            double complex value = root3(0);
            if (!from_right) {
                if (column < GA_WIDTH) {
                    value = ga_read(
                        carrier,
                        ga_relation_cell(
                            source, local, 1U + GA_WIDTH + column
                        ),
                        stats
                    );
                } else if (column < 2U * GA_WIDTH) {
                    value = ga_read(
                        carrier,
                        ga_relation_cell(
                            source, local, 1U + column - GA_WIDTH
                        ),
                        stats
                    );
                } else if (column == 3U * GA_WIDTH) {
                    value = ga_read(
                        carrier,
                        ga_relation_cell(source, local, 5U),
                        stats
                    );
                }
            } else {
                if (column < GA_WIDTH) {
                    value = ga_read(
                        carrier,
                        ga_relation_cell(source, local, 1U + column),
                        stats
                    );
                } else if (
                    column >= 2U * GA_WIDTH
                    && column < 3U * GA_WIDTH
                ) {
                    value = ga_read(
                        carrier,
                        ga_relation_cell(
                            source,
                            local,
                            1U + GA_WIDTH + column - 2U * GA_WIDTH
                        ),
                        stats
                    );
                } else if (column == 3U * GA_WIDTH) {
                    value = ga_read(
                        carrier,
                        ga_relation_cell(source, local, 5U),
                        stats
                    );
                }
            }
            ga_update(
                carrier,
                ga_joint_cell(row, column),
                ga_gate(active, value, stats),
                stats
            );
        }
    }
}

static void ga_reduce_stage(
    struct carrier *carrier,
    size_t stage,
    struct ga_stats *stats
) {
    const size_t target = stage;
    const size_t column = stage < GA_WIDTH
        ? stage
        : stage;
    double complex found = root3(0);
    for (size_t row = 0U; row < GA_JOINT_ROWS; ++row) {
        const double complex eligible = ga_row_eligible(
            carrier, stage, row, 1, stats
        );
        const double complex bit = ga_read(
            carrier, ga_joint_cell(row, column), stats
        );
        const double complex select = ga_and(
            ga_and(eligible, bit, stats),
            ga_not(found, stats),
            stats
        );
        ga_update(
            carrier, ga_select_cell(stage, row), select, stats
        );
        ga_conditional_swap_rows(
            carrier, target, row, select, stats
        );
        found = ga_or(found, select, stats);
    }
    ga_update(carrier, ga_pivot_cell(stage), found, stats);
    for (size_t row = 0U; row < GA_JOINT_ROWS; ++row) {
        if (row == target) {
            continue;
        }
        const double complex eligible = ga_row_eligible(
            carrier, stage, row, 0, stats
        );
        const double complex bit = ga_read(
            carrier, ga_joint_cell(row, column), stats
        );
        const double complex control = ga_and(
            ga_and(found, eligible, stats), bit, stats
        );
        ga_update(
            carrier, ga_elim_cell(stage, row), control, stats
        );
        ga_conditional_row_add(
            carrier, target, row, control, stats
        );
    }
}

static void ga_reverse_stage(
    struct carrier *carrier,
    size_t stage,
    struct ga_stats *stats
) {
    const size_t target = stage;
    for (size_t row = GA_JOINT_ROWS; row-- > 0U;) {
        if (row == target) {
            continue;
        }
        const double complex control = ga_read(
            carrier, ga_elim_cell(stage, row), stats
        );
        ga_conditional_row_add(
            carrier, target, row, control, stats
        );
    }
    for (size_t row = GA_JOINT_ROWS; row-- > 0U;) {
        const double complex select = ga_read(
            carrier, ga_select_cell(stage, row), stats
        );
        ga_conditional_swap_rows(
            carrier, target, row, select, stats
        );
    }
    for (size_t row = 0U; row < GA_JOINT_ROWS; ++row) {
        ga_clear_actual(carrier, ga_select_cell(stage, row), stats);
        ga_clear_actual(carrier, ga_elim_cell(stage, row), stats);
    }
    ga_clear_actual(carrier, ga_pivot_cell(stage), stats);
}

static double complex ga_nonpivot_active(
    const struct carrier *carrier,
    size_t row,
    struct ga_stats *stats
) {
    double complex active = ga_read(
        carrier, ga_active_cell(row), stats
    );
    for (size_t stage = 0U; stage < GA_REDUCTION_COLUMNS; ++stage) {
        if (row == stage) {
            active = ga_and(
                active,
                ga_not(
                    ga_read(carrier, ga_pivot_cell(stage), stats),
                    stats
                ),
                stats
            );
        }
    }
    return active;
}

static double complex ga_output_empty(
    struct carrier *carrier,
    struct ga_stats *stats
) {
    double complex inconsistent = root3(0);
    for (size_t row = 0U; row < GA_JOINT_ROWS; ++row) {
        const double complex active = ga_nonpivot_active(
            carrier, row, stats
        );
        const double complex constant = ga_read(
            carrier,
            ga_joint_cell(row, GA_JOINT_COLUMNS - 1U),
            stats
        );
        inconsistent = ga_or(
            inconsistent,
            ga_and(active, constant, stats),
            stats
        );
    }
    ga_update(
        carrier, GA_INCONSISTENCY_CELL, inconsistent, stats
    );
    return ga_or(
        ga_read(carrier, GA_INPUT_EMPTY_CELL, stats),
        inconsistent,
        stats
    );
}

static double complex ga_output_factor(
    const struct carrier *carrier,
    size_t field,
    double complex output_empty,
    struct ga_stats *stats
) {
    if (field == GA_ROWS * GA_ROW_CELLS) {
        return output_empty;
    }
    const size_t row = field / GA_ROW_CELLS;
    const size_t part = field % GA_ROW_CELLS;
    const size_t stage = GA_WIDTH + row;
    const double complex active = ga_and(
        ga_read(carrier, ga_pivot_cell(stage), stats),
        ga_not(output_empty, stats),
        stats
    );
    if (part == 0U) {
        return active;
    }
    const size_t joint_column = part == 5U
        ? GA_JOINT_COLUMNS - 1U
        : GA_WIDTH + part - 1U;
    return ga_gate(
        active,
        ga_read(
            carrier, ga_joint_cell(stage, joint_column), stats
        ),
        stats
    );
}

static void ga_apply_compose(
    struct carrier *carrier,
    const struct ga_descriptor *descriptor,
    int inverse,
    int wrong_first_factor,
    struct ga_stats *stats
) {
    ++stats->compose_calls;
    ++stats->workspace_tolerance_checks;
    if (!ga_workspace_is_zero(carrier)) {
        fail("oblivious affine workspace not zero on entry");
    }
    ga_encode_joint(carrier, descriptor, stats);
    for (size_t stage = 0U; stage < GA_REDUCTION_COLUMNS; ++stage) {
        ga_reduce_stage(carrier, stage, stats);
    }
    const double complex output_empty = ga_output_empty(carrier, stats);
    for (size_t field = 0U; field < GA_BLOCK_CELLS; ++field) {
        double complex factor = ga_output_factor(
            carrier, field, output_empty, stats
        );
        if (inverse && wrong_first_factor && field == 0U) {
            factor = ga_xor(factor, root3(1), stats);
        }
        ga_update(
            carrier,
            descriptor->output_start + field,
            inverse ? conj(factor) : factor,
            stats
        );
    }
    ga_clear_actual(carrier, GA_INCONSISTENCY_CELL, stats);
    for (size_t stage = GA_REDUCTION_COLUMNS; stage-- > 0U;) {
        ga_reverse_stage(carrier, stage, stats);
    }
    for (size_t row = 0U; row < GA_JOINT_ROWS; ++row) {
        for (size_t column = 0U; column < GA_JOINT_COLUMNS; ++column) {
            ga_clear_actual(
                carrier, ga_joint_cell(row, column), stats
            );
        }
        ga_clear_actual(carrier, ga_active_cell(row), stats);
    }
    ga_clear_actual(carrier, GA_INPUT_EMPTY_CELL, stats);
    ++stats->workspace_tolerance_checks;
    if (!ga_workspace_is_zero(carrier)) {
        fail("oblivious affine workspace not cleared");
    }
}

static void ga_encode_relation(
    struct carrier *carrier,
    const unsigned char relation[GA_BLOCK_CELLS],
    size_t start,
    int inverse,
    struct ga_stats *stats
) {
    for (size_t field = 0U; field < GA_BLOCK_CELLS; ++field) {
        const double complex factor = root3(relation[field]);
        ga_update(
            carrier,
            start + field,
            inverse ? conj(factor) : factor,
            stats
        );
    }
}

static void ga_copy_block(
    struct carrier *carrier,
    size_t source,
    size_t target,
    int inverse,
    struct ga_stats *stats
) {
    for (size_t field = 0U; field < GA_BLOCK_CELLS; ++field) {
        const double complex factor = ga_read(
            carrier, source + field, stats
        );
        ga_update(
            carrier,
            target + field,
            inverse ? conj(factor) : factor,
            stats
        );
    }
}

static struct ga_boundary ga_latch_boundary(
    const struct carrier *carrier,
    struct ga_stats *stats
) {
    struct ga_boundary boundary = {
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t field = 0U; field < GA_BLOCK_CELLS; ++field) {
        double error = 0.0;
        boundary.coefficient[field] = decode_root(
            relative(carrier, GA_BOUNDARY_START + field), &error
        );
        if (error > boundary.maximum_root_error) {
            boundary.maximum_root_error = error;
        }
        const unsigned char byte =
            (unsigned char)boundary.coefficient[field];
        boundary.hash = hash_bytes(boundary.hash, &byte, 1U);
        ++stats->boundary_decodes;
    }
    return boundary;
}

static struct ga_execution ga_execute(
    struct carrier *carrier,
    const struct ga_program *program,
    enum ga_restore_mode restore_mode
) {
    struct ga_execution execution = {0};
    struct carrier borrowed = snapshot_carrier(carrier);
    const struct ga_descriptor child = {
        .left_start = GA_F_START,
        .right_start = GA_G_START,
        .output_start = GA_H_START
    };
    const struct ga_descriptor parent = {
        .left_start = GA_H_START,
        .right_start = GA_K_START,
        .output_start = GA_Z_START
    };
    ga_encode_relation(
        carrier, program->relation[0], GA_F_START, 0, &execution.stats
    );
    ga_encode_relation(
        carrier, program->relation[1], GA_G_START, 0, &execution.stats
    );
    ga_encode_relation(
        carrier, program->relation[2], GA_K_START, 0, &execution.stats
    );
    ga_apply_compose(carrier, &child, 0, 0, &execution.stats);
    ga_apply_compose(carrier, &parent, 0, 0, &execution.stats);
    ga_copy_block(
        carrier, GA_Z_START, GA_BOUNDARY_START, 0, &execution.stats
    );
    execution.boundary = ga_latch_boundary(carrier, &execution.stats);
    execution.displacement_l2 = displacement(carrier, &borrowed);
    if (restore_mode == GA_RESTORE_SNAPSHOT) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        execution.snapshot_loaded = 1;
    } else {
        ga_copy_block(
            carrier, GA_Z_START, GA_BOUNDARY_START, 1, &execution.stats
        );
        if (restore_mode == GA_RESTORE_REORDERED) {
            ga_apply_compose(carrier, &child, 1, 0, &execution.stats);
            ga_apply_compose(carrier, &parent, 1, 0, &execution.stats);
        } else {
            if (restore_mode != GA_RESTORE_MISSING_PARENT) {
                ga_apply_compose(
                    carrier,
                    &parent,
                    1,
                    restore_mode == GA_RESTORE_WRONG_PARENT,
                    &execution.stats
                );
            }
            ga_apply_compose(carrier, &child, 1, 0, &execution.stats);
        }
        ga_encode_relation(
            carrier, program->relation[2], GA_K_START, 1, &execution.stats
        );
        ga_encode_relation(
            carrier, program->relation[1], GA_G_START, 1, &execution.stats
        );
        ga_encode_relation(
            carrier, program->relation[0], GA_F_START, 1, &execution.stats
        );
    }
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
    execution.workspace_cleared = ga_workspace_is_zero(carrier);
    free_carrier(&borrowed);
    return execution;
}

static int ga_boundary_equal(
    const struct ga_boundary *left,
    const struct ga_boundary *right
) {
    return memcmp(
        left->coefficient,
        right->coefficient,
        sizeof(left->coefficient)
    ) == 0;
}

static int ga_boundary_is_universal(const struct ga_boundary *boundary) {
    for (size_t field = 0U; field < GA_BLOCK_CELLS; ++field) {
        if (boundary->coefficient[field] != 0) {
            return 0;
        }
    }
    return 1;
}

static int ga_boundary_is_empty(const struct ga_boundary *boundary) {
    for (size_t field = 0U; field + 1U < GA_BLOCK_CELLS; ++field) {
        if (boundary->coefficient[field] != 0) {
            return 0;
        }
    }
    return boundary->coefficient[GA_BLOCK_CELLS - 1U] == 1;
}

static int ga_stats_same_schedule(
    const struct ga_stats *left,
    const struct ga_stats *right
) {
    return left->compose_calls == right->compose_calls
        && left->phase_ands == right->phase_ands
        && left->phase_xors == right->phase_xors
        && left->phase_ors == right->phase_ors
        && left->phase_nots == right->phase_nots
        && left->conditional_swaps == right->conditional_swaps
        && left->conditional_row_adds == right->conditional_row_adds
        && left->phase_cell_updates == right->phase_cell_updates
        && left->carrier_reads == right->carrier_reads
        && left->workspace_tolerance_checks
            == right->workspace_tolerance_checks;
}

static void ga_print_boundary(const struct ga_boundary *boundary) {
    putchar('[');
    for (size_t field = 0U; field < GA_BLOCK_CELLS; ++field) {
        if (field != 0U) {
            putchar(',');
        }
        printf("%d", boundary->coefficient[field]);
    }
    putchar(']');
}

static void ga_print_execution(
    const char *mode,
    const struct ga_execution *execution
) {
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"BOUNDED_WIDTH2_COEFFICIENT_OBLIVIOUS_GENERAL_"
        "AFFINE_PHASE_RELATION_COMPOSITION\","
        "\"relation_family\":\"GENERAL_AFFINE_F2_EQUATION_SYSTEMS_"
        "FIXED_WIDTH2\","
        "\"width\":2,"
        "\"relation_rows\":4,"
        "\"relation_cells\":25,"
        "\"complete_boundary_materialized\":true,"
        "\"canonical_boundary\":true,"
        "\"dense_relation_entries\":16,"
        "\"polynomial_descriptor_cells\":25,"
        "\"joint_workspace_cells\":56,"
        "\"phase_control_cells\":112,"
        "\"total_workspace_cells\":168,"
        "\"carrier_cells\":%u,"
        "\"live_carrier_complex_values\":%zu,"
        "\"live_carrier_bytes\":%zu,"
        "\"comparison_snapshot_complex_values\":%zu,"
        "\"comparison_snapshot_bytes\":%zu,"
        "\"carrier_and_snapshot_heap_bytes\":%zu,"
        "\"public_program_coefficients\":%zu,"
        "\"program_storage_bytes_current_abi\":%zu,"
        "\"coexisting_program_storage_bytes_current_abi\":%zu,"
        "\"compiled_descriptor_bytes_current_abi\":%zu,"
        "\"execution_result_bytes_current_abi\":%zu,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"serialized_intermediate_coefficients\":0,"
        "\"host_selected_pivots\":0,"
        "\"fixed_host_loop_address_schedule\":true,"
        "\"tuple_slots\":0,"
        "\"assignment_slots\":0,"
        "\"witness_slots\":0,"
        "\"inverse_factor_recomputations\":%u,"
        "\"restoration_cell_checks\":%u,"
        "\"restoration_path\":\"%s\","
        "\"boundary_coefficients\":",
        mode,
        (unsigned)GA_CARRIER_CELLS,
        2U * (size_t)GA_CARRIER_CELLS,
        2U * (size_t)GA_CARRIER_CELLS * sizeof(double complex),
        2U * (size_t)GA_CARRIER_CELLS,
        2U * (size_t)GA_CARRIER_CELLS * sizeof(double complex),
        4U * (size_t)GA_CARRIER_CELLS * sizeof(double complex),
        3U * (size_t)GA_BLOCK_CELLS,
        sizeof(struct ga_program),
        2U * sizeof(struct ga_program),
        2U * sizeof(struct ga_descriptor),
        sizeof(struct ga_execution),
        execution->snapshot_loaded ? 0U : 2U,
        (unsigned)GA_CARRIER_CELLS,
        execution->snapshot_loaded ? "snapshot_reload" : "actual_inverse"
    );
    ga_print_boundary(&execution->boundary);
    printf(
        ",\"boundary_fnv1a64\":\"%016llx\","
        "\"maximum_root_error\":%.12g,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_max_abs\":%.12g,"
        "\"workspace_cleared\":%s,"
        "\"workspace_clear_tolerance\":%.12g,"
        "\"snapshot_loaded\":%s,"
        "\"compose_calls\":%llu,"
        "\"phase_ands\":%llu,"
        "\"phase_xors\":%llu,"
        "\"phase_ors\":%llu,"
        "\"phase_nots\":%llu,"
        "\"conditional_swaps\":%llu,"
        "\"conditional_row_adds\":%llu,"
        "\"phase_cell_updates\":%llu,"
        "\"carrier_reads\":%llu,"
        "\"boundary_decodes\":%llu,"
        "\"workspace_tolerance_checks\":%llu}\n",
        (unsigned long long)execution->boundary.hash,
        execution->boundary.maximum_root_error,
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->integrity_max_abs,
        execution->workspace_cleared ? "true" : "false",
        GA_WORKSPACE_TOLERANCE,
        execution->snapshot_loaded ? "true" : "false",
        (unsigned long long)execution->stats.compose_calls,
        (unsigned long long)execution->stats.phase_ands,
        (unsigned long long)execution->stats.phase_xors,
        (unsigned long long)execution->stats.phase_ors,
        (unsigned long long)execution->stats.phase_nots,
        (unsigned long long)execution->stats.conditional_swaps,
        (unsigned long long)execution->stats.conditional_row_adds,
        (unsigned long long)execution->stats.phase_cell_updates,
        (unsigned long long)execution->stats.carrier_reads,
        (unsigned long long)execution->stats.boundary_decodes,
        (unsigned long long)execution->stats.workspace_tolerance_checks
    );
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--project-intermediate") == 0) {
        fail("only final affine equation boundary may be projected");
    }
    if (argc == 2 && strcmp(argv[1], "--project-controls") == 0) {
        fail("phase pivot and elimination controls are not projectable");
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        fail("null oblivious affine carrier");
    }
    if (argc != 1) {
        fail("usage: algebraic_oblivious_affine_phase");
    }
    struct ga_program program[GA_SEMANTIC_VARIANTS];
    for (size_t variant = 0U; variant < GA_SEMANTIC_VARIANTS; ++variant) {
        program[variant] = ga_make_program(variant);
    }
    const struct process shape = {.carrier_cells = GA_CARRIER_CELLS};
    struct carrier carrier = make_carrier(&shape, 6401);
    struct ga_execution semantic[GA_SEMANTIC_VARIANTS];
    double repeated_max_abs = 0.0;
    for (size_t variant = 0U; variant < GA_SEMANTIC_VARIANTS; ++variant) {
        semantic[variant] = ga_execute(
            &carrier, &program[variant], GA_RESTORE_CORRECT
        );
        if (semantic[variant].restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = semantic[variant].restoration_max_abs;
        }
    }
    for (size_t cycle = 0U; cycle < GA_REPEAT_CYCLES; ++cycle) {
        const struct ga_execution repeated = ga_execute(
            &carrier,
            &program[cycle % GA_SEMANTIC_VARIANTS],
            GA_RESTORE_CORRECT
        );
        if (repeated.restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = repeated.restoration_max_abs;
        }
    }
    free_carrier(&carrier);

    struct ga_execution control[3];
    const enum ga_restore_mode mode[3] = {
        GA_RESTORE_WRONG_PARENT,
        GA_RESTORE_MISSING_PARENT,
        GA_RESTORE_REORDERED
    };
    for (size_t index = 0U; index < 3U; ++index) {
        carrier = make_carrier(&shape, 6401);
        control[index] = ga_execute(&carrier, &program[0], mode[index]);
        free_carrier(&carrier);
    }
    carrier = make_carrier(&shape, 6401);
    const struct ga_execution snapshot = ga_execute(
        &carrier, &program[0], GA_RESTORE_SNAPSHOT
    );
    free_carrier(&carrier);

    static const char *const semantic_mode[GA_SEMANTIC_VARIANTS] = {
        "oblivious-affine-primary",
        "oblivious-affine-reuse",
        "oblivious-affine-permuted",
        "oblivious-affine-duplicated",
        "oblivious-affine-universal-rank0",
        "oblivious-affine-universal-rank1",
        "oblivious-affine-input-empty",
        "oblivious-affine-composed-empty",
        "oblivious-affine-full-rank"
    };
    for (size_t variant = 0U; variant < GA_SEMANTIC_VARIANTS; ++variant) {
        ga_print_execution(semantic_mode[variant], &semantic[variant]);
    }
    ga_print_execution("oblivious-affine-snapshot", &snapshot);
    int fixed_operation_schedule = 1;
    for (size_t variant = 1U; variant < GA_SEMANTIC_VARIANTS; ++variant) {
        fixed_operation_schedule = fixed_operation_schedule
            && ga_stats_same_schedule(
                &semantic[0].stats, &semantic[variant].stats
            );
    }
    printf(
        "{\"mode\":\"controls\","
        "\"wrong_parent_inverse\":%s,"
        "\"missing_parent_inverse\":%s,"
        "\"reordered_inverse\":%s,"
        "\"wrong_parent_restoration_abs\":%.12g,"
        "\"missing_parent_restoration_abs\":%.12g,"
        "\"reordered_restoration_abs\":%.12g,"
        "\"different_reuse_boundary\":%s,"
        "\"row_permutation_invariant\":%s,"
        "\"duplicate_equation_invariant\":%s,"
        "\"rank0_universal\":%s,"
        "\"rank1_universal\":%s,"
        "\"input_empty_propagated\":%s,"
        "\"multi_contradiction_empty\":%s,"
        "\"canonical_full_rank_boundary\":%s,"
        "\"fixture_hashes_match\":%s,"
        "\"fixed_operation_schedule\":%s,"
        "\"same_carrier_transactions\":%u,"
        "\"repeated_reuse_max_restoration_abs\":%.12g}\n",
        control[0].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        control[1].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        control[2].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        control[0].restoration_max_abs,
        control[1].restoration_max_abs,
        control[2].restoration_max_abs,
        !ga_boundary_equal(
            &semantic[0].boundary, &semantic[1].boundary
        )
            ? "true" : "false",
        ga_boundary_equal(
            &semantic[0].boundary, &semantic[2].boundary
        ) ? "true" : "false",
        ga_boundary_equal(
            &semantic[0].boundary, &semantic[3].boundary
        ) ? "true" : "false",
        ga_boundary_is_universal(&semantic[4].boundary)
            ? "true" : "false",
        ga_boundary_is_universal(&semantic[5].boundary)
            ? "true" : "false",
        ga_boundary_is_empty(&semantic[6].boundary)
            ? "true" : "false",
        ga_boundary_is_empty(&semantic[7].boundary)
            ? "true" : "false",
        semantic[8].boundary.hash == UINT64_C(0x2a656a080ba3a76b)
            ? "true" : "false",
        semantic[0].boundary.hash == UINT64_C(0xa96f5625e4054ee6)
            && semantic[1].boundary.hash == UINT64_C(0x0183d6882c23ee46)
            && semantic[4].boundary.hash == UINT64_C(0xd4657f55662f817f)
            && semantic[6].boundary.hash == UINT64_C(0xd4657e55662f7fcc)
            ? "true" : "false",
        fixed_operation_schedule ? "true" : "false",
        (unsigned)(GA_REPEAT_CYCLES + GA_SEMANTIC_VARIANTS),
        repeated_max_abs
    );

    int semantic_integrity = 1;
    for (size_t variant = 0U; variant < GA_SEMANTIC_VARIANTS; ++variant) {
        semantic_integrity = semantic_integrity
            && semantic[variant].boundary.maximum_root_error
                <= ROOT_TOLERANCE
            && semantic[variant].restoration_max_abs
                <= RESTORATION_TOLERANCE
            && semantic[variant].workspace_cleared;
    }
    const int pass =
        semantic_integrity
        && control[0].restoration_max_abs >= CONTROL_MINIMUM
        && control[1].restoration_max_abs >= CONTROL_MINIMUM
        && control[2].restoration_max_abs >= CONTROL_MINIMUM
        && snapshot.snapshot_loaded
        && snapshot.restoration_max_abs <= RESTORATION_TOLERANCE
        && repeated_max_abs <= RESTORATION_TOLERANCE
        && !ga_boundary_equal(
            &semantic[0].boundary, &semantic[1].boundary
        )
        && ga_boundary_equal(
            &semantic[0].boundary, &semantic[2].boundary
        )
        && ga_boundary_equal(
            &semantic[0].boundary, &semantic[3].boundary
        )
        && ga_boundary_is_universal(&semantic[4].boundary)
        && ga_boundary_is_universal(&semantic[5].boundary)
        && ga_boundary_is_empty(&semantic[6].boundary)
        && ga_boundary_is_empty(&semantic[7].boundary)
        && semantic[8].boundary.hash == UINT64_C(0x2a656a080ba3a76b)
        && semantic[0].boundary.hash == UINT64_C(0xa96f5625e4054ee6)
        && semantic[1].boundary.hash == UINT64_C(0x0183d6882c23ee46)
        && semantic[4].boundary.hash == UINT64_C(0xd4657f55662f817f)
        && semantic[6].boundary.hash == UINT64_C(0xd4657e55662f7fcc)
        && fixed_operation_schedule;
    return pass ? 0 : 1;
}
