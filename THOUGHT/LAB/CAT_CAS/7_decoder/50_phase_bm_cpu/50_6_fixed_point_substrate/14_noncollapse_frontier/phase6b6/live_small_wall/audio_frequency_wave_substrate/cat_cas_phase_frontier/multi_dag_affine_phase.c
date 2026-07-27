#define _POSIX_C_SOURCE 200809L

/*
 * Bounded nested two-fanout successor.
 *
 * This translation unit deliberately reuses the branch-native phase carrier,
 * affine relation kernels, public topology compiler, live-set scheduler, and
 * actual inverse path.  The accepted graph has two same-typed native-produced
 * messages.  S is consumed to produce T; S and T then remain simultaneously
 * resident, each with two edge-indexed forward/inverse custody observations.
 * Only the final ROOT relation is projected.
 */

#ifndef MF_WIDTH
#define MF_WIDTH 3U
#endif
#define DC_WIDTH MF_WIDTH
#define DC_LEAF_BODIES 3U
#define DC_REUSE_CYCLES 0U
#define DC_CLAIM \
    "BOUNDED_NESTED_AFFINE_DAG_MULTI_SHARED_RESIDENT_MESSAGE_CUSTODY"
#define DC_PUBLIC_MAIN dag_affine_single_embedded_main
#include "dag_affine_fanout_phase.c"
#undef DC_PUBLIC_MAIN

#define MF_VARIANTS 5U
#ifndef MF_RUN_VARIANTS
#define MF_RUN_VARIANTS MF_VARIANTS
#endif
#ifndef MF_REUSE_CYCLES
#define MF_REUSE_CYCLES 12U
#endif
#ifndef MF_SCALE_ONLY
#define MF_SCALE_ONLY 0
#endif

_Static_assert(MF_RUN_VARIANTS >= 2U, "multi DAG reuse required");
_Static_assert(MF_RUN_VARIANTS <= MF_VARIANTS, "multi DAG variants exceeded");

static size_t mf_find_public(
    const struct dc_compiled *compiled,
    unsigned public_id
) {
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        if (compiled->graph.node[index].id == public_id) {
            return index;
        }
    }
    fail("multi DAG public node unavailable");
    return 0U;
}

static struct rc_program mf_make_program(size_t variant) {
    struct rc_program program = {0};
    for (size_t leaf = 0U; leaf < DC_LEAF_BODIES; ++leaf) {
        for (size_t bit = 0U; bit < GA_WIDTH; ++bit) {
            const size_t identity[] = {bit};
            rc_set_map_row(
                program.leaf[leaf], bit, 0U, identity, 1U
            );
        }
    }
    if (variant == 1U) {
        return program;
    }

    const size_t bit0[] = {0U};
    const size_t bit1[] = {1U};
    const size_t bit2[] = {2U};
    const size_t bit01[] = {0U, 1U};
    const size_t bit02[] = {0U, 2U};

    /* F:A->B swaps bits zero and one. */
    rc_set_map_row(program.leaf[0], 0U, 0U, bit1, 1U);
    rc_set_map_row(program.leaf[0], 1U, 0U, bit0, 1U);
    rc_set_map_row(program.leaf[0], 2U, 0U, bit2, 1U);

    /* G:B->A makes S(x)=(x0+x1,x1,x2). */
    rc_set_map_row(program.leaf[1], 0U, 0U, bit01, 2U);
    rc_set_map_row(program.leaf[1], 1U, 0U, bit0, 1U);
    rc_set_map_row(program.leaf[1], 2U, 0U, bit2, 1U);

    /* H:A->A makes T(x)=(S0+S2,S1,S2). */
    rc_set_map_row(program.leaf[2], 0U, 0U, bit02, 2U);
    rc_set_map_row(program.leaf[2], 1U, 0U, bit1, 1U);
    rc_set_map_row(program.leaf[2], 2U, 0U, bit2, 1U);

    if (variant == 2U) {
        for (size_t bit = 0U; bit < GA_WIDTH; ++bit) {
            const size_t identity[] = {bit};
            rc_set_map_row(
                program.leaf[2], bit, 0U, identity, 1U
            );
        }
    } else if (variant == 3U) {
        rc_set_map_row(program.leaf[1], 0U, 0U, bit0, 1U);
        rc_set_map_row(program.leaf[1], 1U, 0U, bit2, 1U);
        rc_set_map_row(program.leaf[1], 2U, 0U, bit1, 1U);
    } else if (variant == 4U) {
        for (size_t bit = 0U; bit < GA_WIDTH; ++bit) {
            const size_t identity[] = {bit};
            rc_set_map_row(
                program.leaf[2],
                bit,
                bit == 0U ? 1U : 0U,
                identity,
                1U
            );
        }
    } else if (variant != 0U) {
        fail("multi DAG program variant unknown");
    }
    return program;
}

static const struct dc_owner_audit *mf_audit(
    const struct dc_execution *execution,
    unsigned public_id
) {
    for (
        size_t index = 0U;
        index < execution->shared_owner_count;
        ++index
    ) {
        if (
            execution->shared_owner[index].public_node_id
                == public_id
        ) {
            return &execution->shared_owner[index];
        }
    }
    fail("multi DAG shared audit unavailable");
    return NULL;
}

static int mf_nested_order(const struct dc_execution *execution) {
    const struct dc_owner_audit *s = mf_audit(execution, 704U);
    const struct dc_owner_audit *t = mf_audit(execution, 705U);
    return s->birth_event < t->birth_event
        && t->birth_event < execution->projection_event
        && execution->projection_event
            < execution->root_inverse_event
        && execution->root_inverse_event < t->inverse_event
        && t->inverse_event < t->release_event
        && t->release_event < s->inverse_event
        && s->inverse_event < s->release_event;
}

static void mf_test_cross_substitution(
    const struct dc_compiled *compiled,
    const struct rc_program *program,
    size_t source,
    size_t target
) {
    const struct process shape = {
        .carrier_cells = rc_carrier_cells(compiled->graph.count)
    };
    struct carrier carrier = make_carrier(&shape, 9127);
    struct rc_runtime runtime = {
        .carrier = &carrier,
        .compiled = &compiled->graph,
        .program = program,
        .pool = compiled->graph.count
    };
    struct dc_custody custody = {0};
    struct rc_lease node_lease[RC_MAX_NODES];
    (void)dc_forward(
        &runtime,
        compiled,
        &custody,
        node_lease,
        0,
        0,
        0,
        target
    );
    if (
        !rc_signature_equal(
            compiled->graph.node[source].signature,
            compiled->graph.node[target].signature
        )
    ) {
        free_carrier(&carrier);
        fail("multi DAG cross-substitution control is not same typed");
    }
    dc_validate_shared_handle(
        compiled,
        &custody,
        target,
        custody.owner[source].lease
    );
    free_carrier(&carrier);
    fail("multi DAG cross-substitution was not rejected");
}

static void mf_manifest_control(
    struct rc_manifest *manifest,
    const char *control
) {
    if (strcmp(control, "--control-cycle") == 0) {
        for (size_t index = 0U; index < manifest->count; ++index) {
            if (manifest->node[index].id == 704U) {
                manifest->node[index].left_id = manifest->root_id;
                return;
            }
        }
    } else if (strcmp(control, "--control-type") == 0) {
        for (size_t index = 0U; index < manifest->count; ++index) {
            if (manifest->node[index].id == 703U) {
                manifest->node[index].signature.domain = RC_PORT_B;
                return;
            }
        }
    } else if (strcmp(control, "--control-disconnected") == 0) {
        manifest->root_id = 706U;
        return;
    } else {
        fail("multi DAG manifest control unknown");
    }
    fail("multi DAG manifest control target unavailable");
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fail(
            "usage: multi_dag_affine_phase DAG_MANIFEST "
            "DUPLICATE_TREE_MANIFEST [CONTROL]"
        );
    }
    if (
        argc == 4
        && (
            strcmp(argv[3], "--project-s") == 0
            || strcmp(argv[3], "--project-t") == 0
            || strcmp(argv[3], "--project-controls") == 0
        )
    ) {
        fail("multi DAG internal projection denied");
    }
    if (argc == 4 && strcmp(argv[3], "--null-carrier") == 0) {
        fail("null multi DAG carrier");
    }

    struct rc_manifest manifest = {0};
    struct rc_manifest duplicate_manifest = {0};
    rc_load_manifest(argv[1], &manifest);
    rc_load_manifest(argv[2], &duplicate_manifest);
    if (
        argc == 4
        && (
            strcmp(argv[3], "--control-cycle") == 0
            || strcmp(argv[3], "--control-type") == 0
            || strcmp(argv[3], "--control-disconnected") == 0
        )
    ) {
        mf_manifest_control(&manifest, argv[3]);
    }
    const struct dc_compiled compiled = dc_compile(&manifest, 2U);
    const struct dc_compiled duplicate =
        dc_compile(&duplicate_manifest, 0U);
    const size_t s = mf_find_public(&compiled, 704U);
    const size_t t = mf_find_public(&compiled, 705U);
    if (
        !compiled.is_shared[s]
        || !compiled.is_shared[t]
        || compiled.graph.node[t].left != s
        || !rc_signature_equal(
            compiled.graph.node[s].signature,
            compiled.graph.node[t].signature
        )
    ) {
        fail("multi DAG lacks nested same-typed shared owners");
    }

    struct rc_program program[MF_VARIANTS];
    for (size_t variant = 0U; variant < MF_VARIANTS; ++variant) {
        program[variant] = mf_make_program(variant);
    }
    if (argc == 4) {
        if (strcmp(argv[3], "--control-copy-s") == 0) {
            dc_test_copy_substitution(&compiled, &program[0], s);
        } else if (strcmp(argv[3], "--control-copy-t") == 0) {
            dc_test_copy_substitution(&compiled, &program[0], t);
        } else if (strcmp(argv[3], "--control-cross-s-t") == 0) {
            mf_test_cross_substitution(
                &compiled, &program[0], s, t
            );
        } else if (strcmp(argv[3], "--control-cross-t-s") == 0) {
            mf_test_cross_substitution(
                &compiled, &program[0], t, s
            );
        } else if (strcmp(argv[3], "--control-stale-s") == 0) {
            dc_test_forward_control(
                &compiled, &program[0], 1, 0, 0, s
            );
        } else if (strcmp(argv[3], "--control-stale-t") == 0) {
            dc_test_forward_control(
                &compiled, &program[0], 1, 0, 0, t
            );
        } else if (strcmp(argv[3], "--control-skip-s") == 0) {
            dc_test_forward_control(
                &compiled, &program[0], 0, 1, 0, s
            );
        } else if (strcmp(argv[3], "--control-skip-t") == 0) {
            dc_test_forward_control(
                &compiled, &program[0], 0, 1, 0, t
            );
        } else if (strcmp(argv[3], "--control-double-s") == 0) {
            dc_test_forward_control(
                &compiled, &program[0], 0, 0, 1, s
            );
        } else if (strcmp(argv[3], "--control-double-t") == 0) {
            dc_test_forward_control(
                &compiled, &program[0], 0, 0, 1, t
            );
        } else if (strcmp(argv[3], "--reordered-inverse-s") == 0) {
            dc_test_reordered(&compiled, &program[0], s);
        } else if (strcmp(argv[3], "--reordered-inverse-t") == 0) {
            dc_test_reordered(&compiled, &program[0], t);
        } else if (
            strcmp(argv[3], "--control-cycle") != 0
            && strcmp(argv[3], "--control-type") != 0
            && strcmp(argv[3], "--control-disconnected") != 0
        ) {
            fail("multi DAG option unknown");
        }
        fail("multi DAG control was not rejected");
    }

    const struct process shape = {
        .carrier_cells = rc_carrier_cells(compiled.graph.count)
    };
    struct carrier carrier = make_carrier(&shape, 9127);
    struct dc_machine machine = {0};
    struct dc_execution semantic[MF_RUN_VARIANTS];
    double repeated_max_abs = 0.0;
    for (size_t variant = 0U; variant < MF_RUN_VARIANTS; ++variant) {
        semantic[variant] = dc_execute(
            &carrier,
            &compiled,
            &program[variant],
            &machine,
            RC_RESTORE_CORRECT
        );
        if (semantic[variant].restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = semantic[variant].restoration_max_abs;
        }
    }
#if MF_REUSE_CYCLES > 0U
    for (size_t cycle = 0U; cycle < MF_REUSE_CYCLES; ++cycle) {
        const struct dc_execution repeated = dc_execute(
            &carrier,
            &compiled,
            &program[cycle % 2U],
            &machine,
            RC_RESTORE_CORRECT
        );
        if (repeated.restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = repeated.restoration_max_abs;
        }
    }
#endif
    free_carrier(&carrier);

    for (size_t variant = 0U; variant < MF_RUN_VARIANTS; ++variant) {
        dc_print(
            variant == 0U
                ? "multi-dag-primary"
                : variant == 1U
                    ? "multi-dag-reuse"
                    : "multi-dag-semantic",
            &compiled,
            &semantic[variant]
        );
    }
    if (MF_SCALE_ONLY) {
        return 0;
    }

    struct dc_machine control_machine = {0};
    carrier = make_carrier(&shape, 9127);
    const struct dc_execution wrong = dc_execute(
        &carrier,
        &compiled,
        &program[0],
        &control_machine,
        RC_RESTORE_WRONG_ROOT
    );
    free_carrier(&carrier);
    control_machine = (struct dc_machine){0};
    carrier = make_carrier(&shape, 9127);
    const struct dc_execution missing = dc_execute(
        &carrier,
        &compiled,
        &program[0],
        &control_machine,
        RC_RESTORE_MISSING_ROOT
    );
    free_carrier(&carrier);
    control_machine = (struct dc_machine){0};
    carrier = make_carrier(&shape, 9127);
    const struct dc_execution snapshot = dc_execute(
        &carrier,
        &compiled,
        &program[0],
        &control_machine,
        RC_RESTORE_SNAPSHOT
    );
    free_carrier(&carrier);

    const struct process duplicate_shape = {
        .carrier_cells = rc_carrier_cells(duplicate.graph.count)
    };
    struct dc_machine duplicate_machine = {0};
    carrier = make_carrier(&duplicate_shape, 9127);
    const struct dc_execution duplicate_run = dc_execute(
        &carrier,
        &duplicate,
        &program[0],
        &duplicate_machine,
        RC_RESTORE_CORRECT
    );
    free_carrier(&carrier);
    dc_print("snapshot-baseline", &compiled, &snapshot);
    dc_print(
        "duplicate-tree-phase-baseline",
        &duplicate,
        &duplicate_run
    );

    printf(
        "{\"mode\":\"multi-dag-controls\","
        "\"nested_same_typed_shared_dependency\":true,"
        "\"nested_lifetime_order_observed\":%s,"
        "\"all_shared_receipts_exact\":%s,"
        "\"primary_reuse_boundary_different\":%s,"
        "\"h_branch_essential\":%s,"
        "\"g_branch_essential\":%s,"
        "\"empty_intersection_control\":%s,"
        "\"coefficient_oblivious_schedule\":%s,"
        "\"wrong_root_inverse\":%s,"
        "\"missing_root_inverse\":%s,"
        "\"duplicate_tree_boundary_equal\":%s,"
        "\"duplicate_tree_restored\":%s,"
        "\"duplicate_tree_fails_shared_predicate\":%s,"
        "\"snapshot_separate_weaker_path\":true,"
        "\"snapshot_post_projection_inverse_operations\":0,"
        "\"accepted_working_slots\":%zu,"
        "\"duplicate_tree_working_slots\":%zu,"
        "\"relation_slot_reduction\":%zu,"
        "\"accepted_native_operation_calls\":%llu,"
        "\"duplicate_tree_native_operation_calls\":%llu,"
        "\"accepted_leaf_encode_calls\":%llu,"
        "\"duplicate_tree_leaf_encode_calls\":%llu,"
        "\"wrong_root_restoration_abs\":%.12g,"
        "\"missing_root_restoration_abs\":%.12g,"
        "\"same_carrier_transactions\":%u,"
        "\"repeated_reuse_max_restoration_abs\":%.12g}\n",
        mf_nested_order(&semantic[0]) ? "true" : "false",
        (
            mf_audit(&semantic[0], 704U)->receipt_exact
            && mf_audit(&semantic[0], 705U)->receipt_exact
        ) ? "true" : "false",
        semantic[0].boundary.hash != semantic[1].boundary.hash
            ? "true" : "false",
        semantic[0].boundary.hash != semantic[2].boundary.hash
            ? "true" : "false",
        semantic[0].boundary.hash != semantic[3].boundary.hash
            ? "true" : "false",
        semantic[4].boundary.coefficient[GA_BLOCK_CELLS - 1U] == 1
            ? "true" : "false",
        ga_stats_same_schedule(
            &semantic[0].stats, &semantic[1].stats
        ) ? "true" : "false",
        wrong.restoration_max_abs > 1.0e-6 ? "true" : "false",
        missing.restoration_max_abs > 1.0e-6 ? "true" : "false",
        ga_boundary_equal(
            &semantic[0].boundary, &duplicate_run.boundary
        ) ? "true" : "false",
        (
            duplicate_run.restoration_max_abs
                <= GA_WORKSPACE_TOLERANCE
            && duplicate_run.allocator_restored
        ) ? "true" : "false",
        (
            duplicate.fanout_nodes == 0U
            && duplicate_run.shared_forward_materializations == 0U
        ) ? "true" : "false",
        compiled.graph.count,
        duplicate.graph.count,
        duplicate.graph.count - compiled.graph.count,
        (unsigned long long)(
            semantic[0].forward_operations
            + semantic[0].inverse_operations
        ),
        (unsigned long long)(
            duplicate_run.forward_operations
            + duplicate_run.inverse_operations
        ),
        (unsigned long long)(
            semantic[0].forward_leaf_encodes
            + semantic[0].inverse_leaf_encodes
        ),
        (unsigned long long)(
            duplicate_run.forward_leaf_encodes
            + duplicate_run.inverse_leaf_encodes
        ),
        wrong.restoration_max_abs,
        missing.restoration_max_abs,
        (unsigned)(MF_RUN_VARIANTS + MF_REUSE_CYCLES),
        repeated_max_abs
    );
    return 0;
}
