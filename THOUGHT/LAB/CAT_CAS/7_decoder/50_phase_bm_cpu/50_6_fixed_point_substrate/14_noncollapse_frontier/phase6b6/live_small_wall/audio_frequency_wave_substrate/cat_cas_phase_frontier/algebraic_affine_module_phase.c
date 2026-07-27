#define _POSIX_C_SOURCE 200809L

/*
 * Mutable successor: a public series/parallel module tree over complete
 * width-parametric affine GF(2) relation messages.
 *
 * A child COMPOSE writes a canonical phase-resident relation, INTERSECT
 * consumes that actual child directly, and a root COMPOSE consumes the
 * actual intersection output.  Shared witnesses and all intermediate
 * relation coefficients remain unresolved; only the final canonical root
 * relation is decoded.
 */

#ifndef AM_WIDTH
#define AM_WIDTH 3U
#endif
#define GA_WIDTH AM_WIDTH
#define GA_REPEAT_CYCLES 1U
#define GA_PUBLIC_MAIN algebraic_oblivious_affine_embedded_main
#include "algebraic_oblivious_affine_phase.c"
#undef GA_PUBLIC_MAIN

#define AM_VARIANTS 6U
#define AM_LEAF_RELATIONS 4U
#define AM_RELATION_BLOCKS 8U
#define AM_F_START (0U * GA_BLOCK_CELLS)
#define AM_G_START (1U * GA_BLOCK_CELLS)
#define AM_P_START (2U * GA_BLOCK_CELLS)
#define AM_K_START (3U * GA_BLOCK_CELLS)
#define AM_R_START (4U * GA_BLOCK_CELLS)
#define AM_BOUNDARY_START GA_BOUNDARY_START
#define AM_I_START GA_CARRIER_CELLS
#define AM_ROOT_START (GA_CARRIER_CELLS + GA_BLOCK_CELLS)
#define AM_CARRIER_CELLS (GA_CARRIER_CELLS + 2U * GA_BLOCK_CELLS)
#ifndef AM_REUSE_CYCLES
#define AM_REUSE_CYCLES 48U
#endif

_Static_assert(
    AM_BOUNDARY_START == 5U * GA_BLOCK_CELLS,
    "module boundary must occupy the sixth pre-work relation block"
);
_Static_assert(
    AM_CARRIER_CELLS
        == AM_RELATION_BLOCKS * GA_BLOCK_CELLS
            + (GA_CARRIER_CELLS - 6U * GA_BLOCK_CELLS),
    "module carrier law mismatch"
);
_Static_assert(AM_WIDTH >= 3U, "mixed affine fixtures require width >= 3");

enum am_operation_kind {
    AM_COMPOSE = 0,
    AM_INTERSECT = 1
};

enum am_port_type {
    AM_PORT_X = 1,
    AM_PORT_U = 2,
    AM_PORT_B = 3,
    AM_PORT_C = 4
};

enum am_restore_mode {
    AM_RESTORE_CORRECT = 0,
    AM_RESTORE_WRONG_ROOT = 1,
    AM_RESTORE_MISSING_ROOT = 2,
    AM_RESTORE_REORDERED = 3,
    AM_RESTORE_SNAPSHOT = 4
};

enum am_algebra_mode {
    AM_ALGEBRA_NATIVE = 0,
    AM_ALGEBRA_BYPASS_LEFT = 1,
    AM_ALGEBRA_BYPASS_RIGHT = 2,
    AM_ALGEBRA_COMPOSE_INSTEAD = 3
};

struct am_program {
    unsigned char relation[AM_LEAF_RELATIONS][GA_BLOCK_CELLS];
};

struct am_operation {
    enum am_operation_kind kind;
    struct ga_descriptor descriptor;
    enum am_port_type left_domain;
    enum am_port_type left_codomain;
    enum am_port_type right_domain;
    enum am_port_type right_codomain;
    enum am_port_type output_domain;
    enum am_port_type output_codomain;
};

struct am_block_signature {
    size_t start;
    enum am_port_type domain;
    enum am_port_type codomain;
};

struct am_execution {
    struct ga_boundary boundary;
    struct ga_stats stats;
    double restoration_max_abs;
    double integrity_max_abs;
    int workspace_cleared;
    int snapshot_loaded;
};

static const struct am_operation am_topology[3] = {
    {
        .kind = AM_COMPOSE,
        .descriptor = {
            .left_start = AM_F_START,
            .right_start = AM_G_START,
            .output_start = AM_R_START
        },
        .left_domain = AM_PORT_X,
        .left_codomain = AM_PORT_U,
        .right_domain = AM_PORT_U,
        .right_codomain = AM_PORT_B,
        .output_domain = AM_PORT_X,
        .output_codomain = AM_PORT_B
    },
    {
        .kind = AM_INTERSECT,
        .descriptor = {
            .left_start = AM_R_START,
            .right_start = AM_P_START,
            .output_start = AM_I_START
        },
        .left_domain = AM_PORT_X,
        .left_codomain = AM_PORT_B,
        .right_domain = AM_PORT_X,
        .right_codomain = AM_PORT_B,
        .output_domain = AM_PORT_X,
        .output_codomain = AM_PORT_B
    },
    {
        .kind = AM_COMPOSE,
        .descriptor = {
            .left_start = AM_I_START,
            .right_start = AM_K_START,
            .output_start = AM_ROOT_START
        },
        .left_domain = AM_PORT_X,
        .left_codomain = AM_PORT_B,
        .right_domain = AM_PORT_B,
        .right_codomain = AM_PORT_C,
        .output_domain = AM_PORT_X,
        .output_codomain = AM_PORT_C
    }
};

static const struct am_block_signature am_block_signatures[7] = {
    {AM_F_START, AM_PORT_X, AM_PORT_U},
    {AM_G_START, AM_PORT_U, AM_PORT_B},
    {AM_P_START, AM_PORT_X, AM_PORT_B},
    {AM_K_START, AM_PORT_B, AM_PORT_C},
    {AM_R_START, AM_PORT_X, AM_PORT_B},
    {AM_I_START, AM_PORT_X, AM_PORT_B},
    {AM_ROOT_START, AM_PORT_X, AM_PORT_C}
};

static int am_block_signature_matches(
    size_t start,
    enum am_port_type domain,
    enum am_port_type codomain
) {
    for (size_t block = 0U; block < 7U; ++block) {
        if (am_block_signatures[block].start == start) {
            return am_block_signatures[block].domain == domain
                && am_block_signatures[block].codomain == codomain;
        }
    }
    return 0;
}

static int am_operation_type_valid(const struct am_operation *operation) {
    if (
        !am_block_signature_matches(
            operation->descriptor.left_start,
            operation->left_domain,
            operation->left_codomain
        )
        || !am_block_signature_matches(
            operation->descriptor.right_start,
            operation->right_domain,
            operation->right_codomain
        )
        || !am_block_signature_matches(
            operation->descriptor.output_start,
            operation->output_domain,
            operation->output_codomain
        )
    ) {
        return 0;
    }
    if (operation->kind == AM_COMPOSE) {
        return operation->left_codomain == operation->right_domain
            && operation->output_domain == operation->left_domain
            && operation->output_codomain == operation->right_codomain;
    }
    return operation->left_domain == operation->right_domain
        && operation->left_codomain == operation->right_codomain
        && operation->output_domain == operation->left_domain
        && operation->output_codomain == operation->left_codomain;
}

static int am_topology_type_valid(void) {
    for (size_t operation = 0U; operation < 3U; ++operation) {
        if (!am_operation_type_valid(&am_topology[operation])) {
            return 0;
        }
    }
    return 1;
}

static void am_equation(
    unsigned char relation[GA_BLOCK_CELLS],
    size_t row,
    unsigned char constant,
    const size_t *variable,
    size_t variables
) {
    ga_begin_equation(relation, row, constant);
    for (size_t index = 0U; index < variables; ++index) {
        ga_set_term(relation, row, variable[index]);
    }
}

static struct am_program am_make_program(size_t variant) {
    struct am_program program = {0};
    if (variant == 0U) {
        const size_t f0[] = {0U, GA_WIDTH + 2U};
        const size_t f1[] = {1U, GA_WIDTH};
        const size_t f2[] = {2U, GA_WIDTH + 1U};
        const size_t g0[] = {0U, GA_WIDTH + 1U};
        const size_t g1[] = {1U, GA_WIDTH + 1U, GA_WIDTH + 2U};
        const size_t g2[] = {2U, GA_WIDTH, GA_WIDTH + 1U};
        const size_t p0[] = {0U, GA_WIDTH, GA_WIDTH + 1U};
        const size_t p1[] = {1U, GA_WIDTH + 1U};
        const size_t p2[] = {2U, GA_WIDTH, GA_WIDTH + 2U};
        const size_t k0[] = {
            0U, GA_WIDTH, GA_WIDTH + 1U, GA_WIDTH + 2U
        };
        const size_t k1[] = {1U, GA_WIDTH + 1U, GA_WIDTH + 2U};
        const size_t k2[] = {2U, GA_WIDTH + 2U};
        am_equation(program.relation[0], 0U, 1U, f0, 2U);
        am_equation(program.relation[0], 1U, 0U, f1, 2U);
        am_equation(program.relation[0], 2U, 0U, f2, 2U);
        am_equation(program.relation[1], 0U, 0U, g0, 2U);
        am_equation(program.relation[1], 1U, 0U, g1, 3U);
        am_equation(program.relation[1], 2U, 1U, g2, 3U);
        am_equation(program.relation[2], 0U, 0U, p0, 3U);
        am_equation(program.relation[2], 1U, 0U, p1, 2U);
        am_equation(program.relation[2], 2U, 1U, p2, 3U);
        am_equation(program.relation[3], 0U, 1U, k0, 4U);
        am_equation(program.relation[3], 1U, 1U, k1, 3U);
        am_equation(program.relation[3], 2U, 1U, k2, 2U);
    } else if (variant == 1U) {
        for (size_t bit = 0U; bit < GA_WIDTH; ++bit) {
            const size_t identity[] = {bit, GA_WIDTH + bit};
            am_equation(
                program.relation[0], bit, 0U, identity, 2U
            );
        }
        const size_t g0[] = {2U, GA_WIDTH};
        const size_t g1[] = {0U, GA_WIDTH + 1U};
        const size_t g2[] = {1U, GA_WIDTH + 2U};
        const size_t p0[] = {GA_WIDTH};
        const size_t p1[] = {0U, GA_WIDTH + 1U};
        const size_t p2[] = {1U, GA_WIDTH + 2U};
        const size_t k0[] = {0U, GA_WIDTH};
        const size_t k1[] = {1U, GA_WIDTH + 1U};
        const size_t k2[] = {2U, GA_WIDTH + 2U};
        am_equation(program.relation[1], 0U, 0U, g0, 2U);
        am_equation(program.relation[1], 1U, 0U, g1, 2U);
        am_equation(program.relation[1], 2U, 0U, g2, 2U);
        am_equation(program.relation[2], 0U, 0U, p0, 1U);
        am_equation(program.relation[2], 1U, 0U, p1, 2U);
        am_equation(program.relation[2], 2U, 0U, p2, 2U);
        am_equation(program.relation[3], 0U, 1U, k0, 2U);
        am_equation(program.relation[3], 1U, 0U, k1, 2U);
        am_equation(program.relation[3], 2U, 1U, k2, 2U);
    } else if (variant == 2U) {
        program.relation[0][GA_BLOCK_CELLS - 1U] = 1U;
    } else if (variant == 3U) {
        for (size_t bit = 0U; bit < GA_WIDTH; ++bit) {
            const size_t identity[] = {bit, GA_WIDTH + bit};
            am_equation(
                program.relation[0], bit, 0U, identity, 2U
            );
            am_equation(
                program.relation[1], bit, 0U, identity, 2U
            );
            am_equation(
                program.relation[3], bit, 0U, identity, 2U
            );
        }
    } else if (variant == 4U) {
        const size_t f[] = {0U};
        const size_t p[] = {0U};
        am_equation(program.relation[0], 0U, 0U, f, 1U);
        am_equation(program.relation[2], 1U, 1U, p, 1U);
    } else if (variant == 5U) {
        const size_t f[] = {0U, GA_WIDTH};
        const size_t g[] = {0U, GA_WIDTH};
        const size_t p[] = {0U, GA_WIDTH};
        const size_t f_high[] = {
            GA_WIDTH - 1U, 2U * GA_WIDTH - 1U
        };
        const size_t g_high[] = {
            GA_WIDTH - 1U, 2U * GA_WIDTH - 1U
        };
        const size_t p_high[] = {
            GA_WIDTH - 1U, 2U * GA_WIDTH - 1U
        };
        am_equation(program.relation[0], 0U, 0U, f, 2U);
        am_equation(
            program.relation[0],
            GA_ROWS - 1U,
            0U,
            f_high,
            2U
        );
        am_equation(program.relation[1], 1U, 0U, g, 2U);
        am_equation(
            program.relation[1],
            GA_ROWS - 1U,
            1U,
            g_high,
            2U
        );
        am_equation(program.relation[2], 2U, 0U, p, 2U);
        am_equation(
            program.relation[2],
            GA_ROWS - 1U,
            1U,
            p_high,
            2U
        );
        for (size_t bit = 0U; bit < GA_WIDTH; ++bit) {
            const size_t identity[] = {bit, GA_WIDTH + bit};
            am_equation(
                program.relation[3], bit, 0U, identity, 2U
            );
        }
    } else {
        fail("unknown affine module program variant");
    }
    return program;
}

static void am_encode_intersection(
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
    ga_update(carrier, GA_INPUT_EMPTY_CELL, input_empty, stats);
    const double complex enabled = ga_not(input_empty, stats);
    for (size_t row = 0U; row < GA_JOINT_ROWS; ++row) {
        const size_t local = row % GA_ROWS;
        const size_t source = row >= GA_ROWS
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
            if (
                column >= GA_WIDTH
                && column < 3U * GA_WIDTH
            ) {
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
                    ga_relation_cell(
                        source, local, GA_ROW_CELLS - 1U
                    ),
                    stats
                );
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

static void am_apply_intersect(
    struct carrier *carrier,
    const struct ga_descriptor *descriptor,
    int inverse,
    int wrong_first_factor,
    struct ga_stats *stats
) {
    ++stats->compose_calls;
    if (!ga_workspace_is_zero(carrier, stats)) {
        fail("affine module workspace not zero on intersection entry");
    }
    am_encode_intersection(carrier, descriptor, stats);
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
    if (!ga_workspace_is_zero(carrier, stats)) {
        fail("affine module workspace not cleared after intersection");
    }
}

static void am_apply_operation(
    struct carrier *carrier,
    const struct am_operation *operation,
    int inverse,
    int wrong_root,
    enum am_algebra_mode algebra_mode,
    struct ga_stats *stats
) {
    if (operation->kind == AM_COMPOSE) {
        ga_apply_compose(
            carrier,
            &operation->descriptor,
            inverse,
            wrong_root,
            stats
        );
    } else if (algebra_mode == AM_ALGEBRA_BYPASS_LEFT) {
        ga_copy_block(
            carrier,
            operation->descriptor.left_start,
            operation->descriptor.output_start,
            inverse,
            stats
        );
    } else if (algebra_mode == AM_ALGEBRA_BYPASS_RIGHT) {
        ga_copy_block(
            carrier,
            operation->descriptor.right_start,
            operation->descriptor.output_start,
            inverse,
            stats
        );
    } else if (algebra_mode == AM_ALGEBRA_COMPOSE_INSTEAD) {
        ga_apply_compose(
            carrier, &operation->descriptor, inverse, 0, stats
        );
    } else {
        am_apply_intersect(
            carrier,
            &operation->descriptor,
            inverse,
            wrong_root,
            stats
        );
    }
}

static struct am_execution am_execute(
    struct carrier *carrier,
    const struct am_program *program,
    enum am_restore_mode restore_mode,
    enum am_algebra_mode algebra_mode
) {
    struct am_execution execution = {0};
    struct carrier borrowed = snapshot_carrier(carrier);
    const size_t leaf_start[AM_LEAF_RELATIONS] = {
        AM_F_START, AM_G_START, AM_P_START, AM_K_START
    };
    for (size_t leaf = 0U; leaf < AM_LEAF_RELATIONS; ++leaf) {
        ga_encode_relation(
            carrier,
            program->relation[leaf],
            leaf_start[leaf],
            0,
            &execution.stats
        );
    }
    for (size_t operation = 0U; operation < 3U; ++operation) {
        am_apply_operation(
            carrier,
            &am_topology[operation],
            0,
            0,
            algebra_mode,
            &execution.stats
        );
    }
    ga_copy_block(
        carrier,
        AM_ROOT_START,
        AM_BOUNDARY_START,
        0,
        &execution.stats
    );
    execution.boundary = ga_latch_boundary(carrier, &execution.stats);
    if (restore_mode == AM_RESTORE_SNAPSHOT) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        execution.snapshot_loaded = 1;
    } else {
        ga_copy_block(
            carrier,
            AM_ROOT_START,
            AM_BOUNDARY_START,
            1,
            &execution.stats
        );
        if (restore_mode == AM_RESTORE_REORDERED) {
            am_apply_operation(
                carrier,
                &am_topology[1],
                1,
                0,
                algebra_mode,
                &execution.stats
            );
            am_apply_operation(
                carrier,
                &am_topology[2],
                1,
                0,
                algebra_mode,
                &execution.stats
            );
            am_apply_operation(
                carrier,
                &am_topology[0],
                1,
                0,
                algebra_mode,
                &execution.stats
            );
        } else {
            if (restore_mode != AM_RESTORE_MISSING_ROOT) {
                am_apply_operation(
                    carrier,
                    &am_topology[2],
                    1,
                    restore_mode == AM_RESTORE_WRONG_ROOT,
                    algebra_mode,
                    &execution.stats
                );
            }
            for (size_t operation = 2U; operation-- > 0U;) {
                am_apply_operation(
                    carrier,
                    &am_topology[operation],
                    1,
                    0,
                    algebra_mode,
                    &execution.stats
                );
            }
        }
        for (size_t leaf = AM_LEAF_RELATIONS; leaf-- > 0U;) {
            ga_encode_relation(
                carrier,
                program->relation[leaf],
                leaf_start[leaf],
                1,
                &execution.stats
            );
        }
    }
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.stats.restoration_cell_checks += carrier->cells;
    execution.integrity_max_abs = integrity(carrier);
    execution.stats.integrity_cell_checks += carrier->cells;
    execution.workspace_cleared = ga_workspace_is_zero(
        carrier, &execution.stats
    );
    free_carrier(&borrowed);
    return execution;
}

static int am_stats_same_schedule(
    const struct ga_stats *left,
    const struct ga_stats *right
) {
    return ga_stats_same_schedule(left, right);
}

static void am_print_execution(
    const char *mode,
    const struct am_execution *execution
) {
    const size_t workspace_cells =
        GA_CARRIER_CELLS - 6U * GA_BLOCK_CELLS;
    const unsigned long long logical_inspections =
        execution->stats.carrier_reads
        + execution->stats.boundary_decodes
        + execution->stats.workspace_cell_checks
        + execution->stats.restoration_cell_checks
        + execution->stats.integrity_cell_checks;
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"BOUNDED_WIDTH_PARAMETRIC_AFFINE_COMPOSITION_"
        "AND_INTERSECTION_MODULE_CLOSURE\","
        "\"width\":%u,"
        "\"public_topology\":\"COMPOSE(INTERSECT(COMPOSE(F,G),P),K)\","
        "\"nominal_type_validation\":true,"
        "\"relation_block_type_bindings\":7,"
        "\"typed_leaf_relations\":4,"
        "\"compose_nodes\":2,"
        "\"intersect_nodes\":1,"
        "\"resident_child_messages\":2,"
        "\"relation_cells\":%u,"
        "\"complete_boundary_materialized\":true,"
        "\"relation_blocks\":%u,"
        "\"workspace_cells\":%zu,"
        "\"carrier_cells\":%u,"
        "\"live_carrier_bytes\":%zu,"
        "\"carrier_and_snapshot_heap_bytes\":%zu,"
        "\"coexisting_program_storage_bytes_current_abi\":%zu,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"serialized_intermediate_coefficients\":0,"
        "\"host_selected_pivots\":0,"
        "\"tuple_slots\":0,"
        "\"assignment_slots\":0,"
        "\"witness_slots\":0,"
        "\"inverse_factor_recomputations\":%u,"
        "\"snapshot_loaded\":%s,"
        "\"snapshot_creation_complex_values_copied\":%zu,"
        "\"snapshot_reload_complex_values_copied\":%zu,"
        "\"boundary_coefficients\":",
        mode,
        (unsigned)GA_WIDTH,
        (unsigned)GA_BLOCK_CELLS,
        (unsigned)AM_RELATION_BLOCKS,
        workspace_cells,
        (unsigned)AM_CARRIER_CELLS,
        2U * (size_t)AM_CARRIER_CELLS * sizeof(double complex),
        4U * (size_t)AM_CARRIER_CELLS * sizeof(double complex),
        AM_VARIANTS * sizeof(struct am_program),
        execution->snapshot_loaded ? 0U : 3U,
        execution->snapshot_loaded ? "true" : "false",
        2U * (size_t)AM_CARRIER_CELLS,
        execution->snapshot_loaded
            ? (size_t)AM_CARRIER_CELLS
            : 0U
    );
    ga_print_boundary(&execution->boundary);
    printf(
        ",\"boundary_fnv1a64\":\"%016llx\","
        "\"maximum_root_error\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_max_abs\":%.12g,"
        "\"workspace_cleared\":%s,"
        "\"operation_calls\":%llu,"
        "\"phase_ands\":%llu,"
        "\"phase_xors\":%llu,"
        "\"phase_ors\":%llu,"
        "\"phase_nots\":%llu,"
        "\"conditional_swaps\":%llu,"
        "\"conditional_row_adds\":%llu,"
        "\"phase_cell_updates\":%llu,"
        "\"native_phase_carrier_reads\":%llu,"
        "\"boundary_decodes\":%llu,"
        "\"workspace_scan_passes\":%llu,"
        "\"workspace_cell_checks\":%llu,"
        "\"restoration_cell_checks\":%llu,"
        "\"integrity_cell_checks\":%llu,"
        "\"logical_carrier_cell_inspections\":%llu}\n",
        (unsigned long long)execution->boundary.hash,
        execution->boundary.maximum_root_error,
        execution->restoration_max_abs,
        execution->integrity_max_abs,
        execution->workspace_cleared ? "true" : "false",
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
        (unsigned long long)execution->stats.workspace_tolerance_checks,
        (unsigned long long)execution->stats.workspace_cell_checks,
        (unsigned long long)execution->stats.restoration_cell_checks,
        (unsigned long long)execution->stats.integrity_cell_checks,
        logical_inspections
    );
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--project-intermediate") == 0) {
        fail("affine module intermediate projection denied");
    }
    if (argc == 2 && strcmp(argv[1], "--project-controls") == 0) {
        fail("affine module phase-control projection denied");
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        fail("null affine module carrier");
    }
    if (argc == 2 && strcmp(argv[1], "--invalid-topology") == 0) {
        struct am_operation invalid = am_topology[0];
        invalid.descriptor.right_start = AM_K_START;
        if (!am_operation_type_valid(&invalid)) {
            fail("affine module nominal type mismatch");
        }
        fail("invalid topology control did not reject");
    }
    if (argc != 1) {
        fail("usage: algebraic_affine_module_phase");
    }
    if (!am_topology_type_valid()) {
        fail("affine module nominal type mismatch");
    }
    struct am_program program[AM_VARIANTS];
    for (size_t variant = 0U; variant < AM_VARIANTS; ++variant) {
        program[variant] = am_make_program(variant);
    }
    const struct process shape = {.carrier_cells = AM_CARRIER_CELLS};
    struct carrier carrier = make_carrier(&shape, 6701);
    struct am_execution semantic[AM_VARIANTS];
    double repeated_max_abs = 0.0;
    for (size_t variant = 0U; variant < AM_VARIANTS; ++variant) {
        semantic[variant] = am_execute(
            &carrier,
            &program[variant],
            AM_RESTORE_CORRECT,
            AM_ALGEBRA_NATIVE
        );
        if (semantic[variant].restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = semantic[variant].restoration_max_abs;
        }
    }
    for (size_t cycle = 0U; cycle < AM_REUSE_CYCLES; ++cycle) {
        const struct am_execution repeated = am_execute(
            &carrier,
            &program[cycle % AM_VARIANTS],
            AM_RESTORE_CORRECT,
            AM_ALGEBRA_NATIVE
        );
        if (repeated.restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = repeated.restoration_max_abs;
        }
    }
    free_carrier(&carrier);
    struct am_execution control[3];
    const enum am_restore_mode control_mode[3] = {
        AM_RESTORE_WRONG_ROOT,
        AM_RESTORE_MISSING_ROOT,
        AM_RESTORE_REORDERED
    };
    for (size_t index = 0U; index < 3U; ++index) {
        carrier = make_carrier(&shape, 6701);
        control[index] = am_execute(
            &carrier,
            &program[0],
            control_mode[index],
            AM_ALGEBRA_NATIVE
        );
        free_carrier(&carrier);
    }
    carrier = make_carrier(&shape, 6701);
    const struct am_execution snapshot = am_execute(
        &carrier,
        &program[0],
        AM_RESTORE_SNAPSHOT,
        AM_ALGEBRA_NATIVE
    );
    free_carrier(&carrier);
    struct am_execution altered[3];
    const enum am_algebra_mode altered_mode[3] = {
        AM_ALGEBRA_BYPASS_LEFT,
        AM_ALGEBRA_BYPASS_RIGHT,
        AM_ALGEBRA_COMPOSE_INSTEAD
    };
    for (size_t index = 0U; index < 3U; ++index) {
        carrier = make_carrier(&shape, 6701);
        altered[index] = am_execute(
            &carrier,
            &program[0],
            AM_RESTORE_CORRECT,
            altered_mode[index]
        );
        free_carrier(&carrier);
    }
    for (size_t variant = 0U; variant < AM_VARIANTS; ++variant) {
        am_print_execution(
            variant == 0U
                ? "affine-module-primary"
                : variant == 1U
                    ? "affine-module-reuse"
                    : "affine-module-semantic",
            &semantic[variant]
        );
    }
    am_print_execution("snapshot-baseline", &snapshot);
    printf(
        "{\"mode\":\"controls\","
        "\"different_reuse_boundary\":%s,"
        "\"input_empty_propagated\":%s,"
        "\"universal_branch_identity\":%s,"
        "\"cross_branch_contradiction_empty\":%s,"
        "\"duplicate_intersection_idempotent\":%s,"
        "\"right_operand_high_index_covered\":true,"
        "\"fixed_operation_schedule\":%s,"
        "\"wrong_root_inverse\":%s,"
        "\"missing_root_inverse\":%s,"
        "\"reordered_root_inverse\":%s,"
        "\"intersection_left_bypass_changes_boundary\":%s,"
        "\"intersection_right_bypass_changes_boundary\":%s,"
        "\"composition_substitution_changes_boundary\":%s,"
        "\"altered_algebra_paths_restored\":%s,"
        "\"nominal_type_validation\":true,"
        "\"wrong_root_restoration_abs\":%.12g,"
        "\"missing_root_restoration_abs\":%.12g,"
        "\"reordered_root_restoration_abs\":%.12g,"
        "\"same_carrier_transactions\":%u,"
        "\"repeated_reuse_max_restoration_abs\":%.12g}\n",
        semantic[0].boundary.hash != semantic[1].boundary.hash
            ? "true" : "false",
        semantic[2].boundary.coefficient[GA_BLOCK_CELLS - 1U] == 1
            ? "true" : "false",
        semantic[3].boundary.coefficient[GA_BLOCK_CELLS - 1U] == 0
            ? "true" : "false",
        semantic[4].boundary.coefficient[GA_BLOCK_CELLS - 1U] == 1
            ? "true" : "false",
        semantic[5].boundary.coefficient[GA_BLOCK_CELLS - 1U] == 0
            ? "true" : "false",
        am_stats_same_schedule(&semantic[0].stats, &semantic[1].stats)
            ? "true" : "false",
        control[0].restoration_max_abs > 1.0e-6 ? "true" : "false",
        control[1].restoration_max_abs > 1.0e-6 ? "true" : "false",
        control[2].restoration_max_abs > 1.0e-6 ? "true" : "false",
        !ga_boundary_equal(
            &semantic[0].boundary, &altered[0].boundary
        ) ? "true" : "false",
        !ga_boundary_equal(
            &semantic[0].boundary, &altered[1].boundary
        ) ? "true" : "false",
        !ga_boundary_equal(
            &semantic[0].boundary, &altered[2].boundary
        ) ? "true" : "false",
        (
            altered[0].restoration_max_abs <= GA_WORKSPACE_TOLERANCE
            && altered[1].restoration_max_abs
                <= GA_WORKSPACE_TOLERANCE
            && altered[2].restoration_max_abs
                <= GA_WORKSPACE_TOLERANCE
        ) ? "true" : "false",
        control[0].restoration_max_abs,
        control[1].restoration_max_abs,
        control[2].restoration_max_abs,
        (unsigned)(AM_VARIANTS + AM_REUSE_CYCLES),
        repeated_max_abs
    );
    return 0;
}
