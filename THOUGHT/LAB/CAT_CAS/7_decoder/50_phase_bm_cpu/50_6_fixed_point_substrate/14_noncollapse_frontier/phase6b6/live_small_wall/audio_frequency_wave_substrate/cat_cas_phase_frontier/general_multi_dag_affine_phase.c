#define _POSIX_C_SOURCE 200809L

/*
 * Retain-all compiler-derived custody successor.
 *
 * The public graph is larger than the prior exact-two fixture and contains
 * four native-produced shared owners with compiler-derived fanouts 4,3,3,2.
 * Every unique node is materialized once and retained until dependency-
 * ordered actual inverse. Only ROOT is copied to the public boundary.
 */

#ifndef GM_WIDTH
#define GM_WIDTH 3U
#endif
#ifndef GM_MAX_FANOUT
#define GM_MAX_FANOUT 4U
#endif
#ifndef GM_NODE_CAP
#define GM_NODE_CAP 15U
#endif
#ifndef GM_OCCURRENCE_ONLY
#define GM_OCCURRENCE_ONLY 0
#endif
#define RC_MAX_NODES GM_NODE_CAP
#define DC_WIDTH GM_WIDTH
#define DC_LEAF_BODIES 4U
#define DC_MAX_FANOUT GM_MAX_FANOUT
#if !GM_OCCURRENCE_ONLY
#define DC_ENABLE_EDGE_RECEIPT_CONTROLS 1
#endif
#define DC_REUSE_CYCLES 0U
#define DC_CLAIM \
    "BOUNDED_FOUR_OWNER_HETEROGENEOUS_FANOUT_RETAIN_ALL_AFFINE_DAG_CUSTODY_ESTABLISHED"
#define DC_PUBLIC_MAIN general_multi_dag_embedded_main
#include "dag_affine_fanout_phase.c"
#undef DC_PUBLIC_MAIN

#include <limits.h>

#define GM_VARIANTS 5U
#ifndef GM_RUN_VARIANTS
#define GM_RUN_VARIANTS GM_VARIANTS
#endif
#ifndef GM_REUSE_CYCLES
#define GM_REUSE_CYCLES 12U
#endif
#ifndef GM_SCALE_ONLY
#define GM_SCALE_ONLY 0
#endif

_Static_assert(GM_RUN_VARIANTS >= 2U, "general DAG reuse required");
_Static_assert(GM_RUN_VARIANTS <= GM_VARIANTS, "general DAG variants exceeded");

static struct rc_program gm_make_program(size_t variant) {
    struct rc_program program = dc_make_program(1U);
    static const unsigned char row[GM_VARIANTS][4U][3U] = {
        {{2U,1U,4U},{0U,7U,4U},{4U,2U,1U},{6U,5U,1U}},
        {{2U,1U,4U},{2U,1U,4U},{4U,2U,1U},{4U,2U,1U}},
        {{2U,1U,4U},{0U,5U,0U},{4U,2U,1U},{2U,2U,5U}},
        {{2U,1U,4U},{0U,0U,7U},{4U,2U,1U},{7U,5U,1U}},
        {{2U,1U,4U},{4U,1U,6U},{4U,2U,1U},{3U,4U,7U}}
    };
    static const unsigned char constant[GM_VARIANTS][4U] = {
        {0U,0U,0U,3U},
        {0U,0U,0U,0U},
        {0U,5U,0U,7U},
        {0U,2U,0U,4U},
        {0U,0U,0U,0U}
    };
    if (variant >= GM_VARIANTS) {
        fail("general DAG program variant unknown");
    }
    for (size_t leaf = 0U; leaf < 4U; ++leaf) {
        for (size_t output = 0U; output < 3U; ++output) {
            size_t input[3U];
            size_t inputs = 0U;
            for (size_t bit = 0U; bit < 3U; ++bit) {
                if ((row[variant][leaf][output] >> bit) & 1U) {
                    input[inputs++] = bit;
                }
            }
            rc_set_map_row(
                program.leaf[leaf],
                output,
                (unsigned char)(
                    (constant[variant][leaf] >> output) & 1U
                ),
                input,
                inputs
            );
        }
    }
    return program;
}

#if !GM_OCCURRENCE_ONLY
static const unsigned gm_shared_id[] = {805U, 806U, 807U, 808U};
static const size_t gm_shared_degree[] = {4U, 3U, 3U, 2U};

static size_t gm_find_public(
    const struct dc_compiled *compiled,
    unsigned public_id
) {
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        if (compiled->graph.node[index].id == public_id) {
            return index;
        }
    }
    fail("general DAG public node unavailable");
    return 0U;
}

static const struct dc_owner_audit *gm_audit(
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
    fail("general DAG shared audit unavailable");
    return NULL;
}

static int gm_owner_id_valid(unsigned public_id) {
    for (
        size_t owner = 0U;
        owner < sizeof(gm_shared_id) / sizeof(gm_shared_id[0]);
        ++owner
    ) {
        if (gm_shared_id[owner] == public_id) {
            return 1;
        }
    }
    return 0;
}

static int gm_parse_owner_option(
    const char *option,
    const char *prefix,
    unsigned *public_id
) {
    const size_t length = strlen(prefix);
    if (strncmp(option, prefix, length) != 0) {
        return 0;
    }
    char *end = NULL;
    const unsigned long parsed = strtoul(option + length, &end, 10);
    if (
        option[length] == '\0'
        || end == NULL
        || *end != '\0'
        || parsed > UINT_MAX
        || !gm_owner_id_valid((unsigned)parsed)
    ) {
        fail("general DAG control owner invalid");
    }
    *public_id = (unsigned)parsed;
    return 1;
}

static int gm_all_receipts_exact(
    const struct dc_execution *execution
) {
    if (
        execution->shared_owner_count
        != sizeof(gm_shared_id) / sizeof(gm_shared_id[0])
    ) {
        return 0;
    }
    for (
        size_t owner = 0U;
        owner < sizeof(gm_shared_id) / sizeof(gm_shared_id[0]);
        ++owner
    ) {
        const struct dc_owner_audit *audit =
            gm_audit(execution, gm_shared_id[owner]);
        if (
            audit->declared_consumers != gm_shared_degree[owner]
            || !audit->receipt_exact
        ) {
            return 0;
        }
    }
    return 1;
}

static int gm_all_shared_edges_exact(
    const struct dc_execution *execution
) {
    if (execution->shared_edge_count != 12U) {
        return 0;
    }
    for (
        size_t edge = 0U;
        edge < execution->shared_edge_count;
        ++edge
    ) {
        if (
            !execution->shared_edge[edge].forward_seen
            || !execution->shared_edge[edge].inverse_seen
            || !execution->shared_edge[edge].same_owner_generation
        ) {
            return 0;
        }
    }
    return 1;
}

static int gm_lifecycle_order(
    const struct dc_execution *execution
) {
    const struct dc_owner_audit *s = gm_audit(execution, 805U);
    const struct dc_owner_audit *t = gm_audit(execution, 806U);
    const struct dc_owner_audit *u = gm_audit(execution, 807U);
    const struct dc_owner_audit *v = gm_audit(execution, 808U);
    return s->birth_event < t->birth_event
        && t->birth_event < u->birth_event
        && u->birth_event < v->birth_event
        && v->birth_event < execution->projection_event
        && execution->projection_event
            < execution->root_inverse_event
        && execution->root_inverse_event < v->inverse_event
        && v->inverse_event < v->release_event
        && v->release_event < u->inverse_event
        && u->inverse_event < u->release_event
        && u->release_event < t->inverse_event
        && t->inverse_event < t->release_event
        && t->release_event < s->inverse_event
        && s->inverse_event < s->release_event;
}

static int gm_semantic_hashes_distinct(
    const struct dc_execution execution[GM_RUN_VARIANTS]
) {
    for (size_t left = 0U; left < GM_RUN_VARIANTS; ++left) {
        for (
            size_t right = left + 1U;
            right < GM_RUN_VARIANTS;
            ++right
        ) {
            if (
                execution[left].boundary.hash
                == execution[right].boundary.hash
            ) {
                return 0;
            }
        }
    }
    return 1;
}

static void gm_require_topology(const struct dc_compiled *compiled) {
    if (
        compiled->graph.count != 15U
        || compiled->graph.leaves != 4U
        || compiled->graph.compose_nodes != 8U
        || compiled->graph.intersect_nodes != 3U
        || compiled->edge_count != 22U
        || compiled->graph.depth != 5U
        || compiled->fanout_nodes != 4U
        || compiled->maximum_fanout != 4U
    ) {
        fail("general DAG topology law mismatch");
    }
    for (
        size_t owner = 0U;
        owner < sizeof(gm_shared_id) / sizeof(gm_shared_id[0]);
        ++owner
    ) {
        const size_t index =
            gm_find_public(compiled, gm_shared_id[owner]);
        if (
            !compiled->is_shared[index]
            || compiled->graph.node[index].indegree
                != gm_shared_degree[owner]
        ) {
            fail("general DAG compiler-derived owner law mismatch");
        }
    }
}

static void gm_test_cross_substitution(
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
        fail("general DAG cross-substitution is not same typed");
    }
    dc_validate_shared_handle(
        compiled,
        &custody,
        target,
        custody.owner[source].lease
    );
    free_carrier(&carrier);
    fail("general DAG cross-substitution was not rejected");
}

static void gm_run_control(
    const char *option,
    const struct dc_compiled *compiled,
    const struct rc_program *program
) {
    unsigned public_id = 0U;
    if (gm_parse_owner_option(
        option, "--control-copy-", &public_id
    )) {
        if (GM_NODE_CAP <= compiled->graph.count) {
            fail("general DAG copy control requires bounded spare slot");
        }
        dc_test_copy_substitution(
            compiled, program, gm_find_public(compiled, public_id)
        );
    }
    if (gm_parse_owner_option(
        option, "--control-stale-", &public_id
    )) {
        dc_test_forward_control(
            compiled,
            program,
            1,
            0,
            0,
            gm_find_public(compiled, public_id)
        );
    }
    if (gm_parse_owner_option(
        option, "--control-skip-", &public_id
    )) {
        dc_test_forward_control(
            compiled,
            program,
            0,
            1,
            0,
            gm_find_public(compiled, public_id)
        );
    }
    if (gm_parse_owner_option(
        option, "--control-double-", &public_id
    )) {
        dc_test_forward_control(
            compiled,
            program,
            0,
            0,
            1,
            gm_find_public(compiled, public_id)
        );
    }
    if (gm_parse_owner_option(
        option, "--reordered-inverse-", &public_id
    )) {
        dc_test_reordered(
            compiled, program, gm_find_public(compiled, public_id)
        );
    }
    unsigned source_id = 0U;
    unsigned target_id = 0U;
    char extra = '\0';
    if (
        sscanf(
            option,
            "--control-cross-%u-%u%c",
            &source_id,
            &target_id,
            &extra
        ) == 2
        && gm_owner_id_valid(source_id)
        && gm_owner_id_valid(target_id)
        && source_id != target_id
    ) {
        gm_test_cross_substitution(
            compiled,
            program,
            gm_find_public(compiled, source_id),
            gm_find_public(compiled, target_id)
        );
    }
    unsigned ordinal = 0U;
    if (
        sscanf(
            option,
            "--control-edge-missing-%u-%u%c",
            &public_id,
            &ordinal,
            &extra
        ) == 2
        && gm_owner_id_valid(public_id)
    ) {
        dc_test_edge_receipt_control(
            compiled,
            program,
            gm_find_public(compiled, public_id),
            ordinal,
            DC_EDGE_RECEIPT_MISSING_FORWARD
        );
    }
    if (
        sscanf(
            option,
            "--control-edge-stale-%u-%u%c",
            &public_id,
            &ordinal,
            &extra
        ) == 2
        && gm_owner_id_valid(public_id)
    ) {
        dc_test_edge_receipt_control(
            compiled,
            program,
            gm_find_public(compiled, public_id),
            ordinal,
            DC_EDGE_RECEIPT_STALE_GENERATION
        );
    }
    if (
        sscanf(
            option,
            "--control-edge-swap-%u-%u%c",
            &source_id,
            &target_id,
            &extra
        ) == 2
        && gm_owner_id_valid(source_id)
        && gm_owner_id_valid(target_id)
        && source_id != target_id
    ) {
        dc_test_swapped_edge_receipts(
            compiled,
            program,
            gm_find_public(compiled, source_id),
            gm_find_public(compiled, target_id)
        );
    }
    fail("general DAG option unknown");
}
#endif

int main(int argc, char **argv) {
#if GM_OCCURRENCE_ONLY
    if (argc != 2) {
        fail(
            "usage: general_multi_dag_affine_phase "
            "OCCURRENCE_TREE_MANIFEST"
        );
    }
    struct rc_manifest occurrence_manifest = {0};
    rc_load_manifest(argv[1], &occurrence_manifest);
    const struct dc_compiled occurrence =
        dc_compile(&occurrence_manifest, 0U);
    if (
        occurrence.graph.count != 51U
        || occurrence.graph.leaves != 26U
        || occurrence.graph.compose_nodes != 22U
        || occurrence.graph.intersect_nodes != 3U
        || occurrence.edge_count != 50U
        || occurrence.graph.depth != 5U
    ) {
        fail("general DAG occurrence expansion law mismatch");
    }
    const struct rc_program program = gm_make_program(0U);
    const struct process shape = {
        .carrier_cells = rc_carrier_cells(occurrence.graph.count)
    };
    struct carrier carrier = make_carrier(&shape, 9127);
    struct dc_machine machine = {0};
    const struct dc_execution execution = dc_execute(
        &carrier,
        &occurrence,
        &program,
        &machine,
        RC_RESTORE_CORRECT
    );
    free_carrier(&carrier);
    dc_print(
        "occurrence-tree-phase-baseline",
        &occurrence,
        &execution
    );
    return 0;
#else
    if (argc < 2 || argc > 3) {
        fail(
            "usage: general_multi_dag_affine_phase "
            "DAG_MANIFEST [CONTROL]"
        );
    }
    if (
        argc == 3
        && (
            strcmp(argv[2], "--project-intermediate") == 0
            || strcmp(argv[2], "--project-custody") == 0
        )
    ) {
        fail("general DAG internal projection denied");
    }
    if (argc == 3 && strcmp(argv[2], "--null-carrier") == 0) {
        fail("null general DAG carrier");
    }

    struct rc_manifest manifest = {0};
    rc_load_manifest(argv[1], &manifest);
    const struct dc_compiled compiled = dc_compile(&manifest, 4U);
    gm_require_topology(&compiled);

    struct rc_program program[GM_VARIANTS];
    for (size_t variant = 0U; variant < GM_VARIANTS; ++variant) {
        program[variant] = gm_make_program(variant);
    }
    if (argc == 3) {
        gm_run_control(argv[2], &compiled, &program[0]);
        fail("general DAG control was not rejected");
    }

    const struct process shape = {
        .carrier_cells = rc_carrier_cells(compiled.graph.count)
    };
    struct carrier carrier = make_carrier(&shape, 9127);
    struct dc_machine machine = {0};
    struct dc_execution semantic[GM_RUN_VARIANTS];
    double repeated_max_abs = 0.0;
    for (size_t variant = 0U; variant < GM_RUN_VARIANTS; ++variant) {
        semantic[variant] = dc_execute(
            &carrier,
            &compiled,
            &program[variant],
            &machine,
            RC_RESTORE_CORRECT
        );
        if (
            semantic[variant].restoration_max_abs
            > repeated_max_abs
        ) {
            repeated_max_abs =
                semantic[variant].restoration_max_abs;
        }
    }
#if GM_REUSE_CYCLES > 0U
    for (size_t cycle = 0U; cycle < GM_REUSE_CYCLES; ++cycle) {
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

    for (size_t variant = 0U; variant < GM_RUN_VARIANTS; ++variant) {
        dc_print(
            variant == 0U
                ? "general-dag-primary"
                : variant == 1U
                    ? "general-dag-reuse"
                    : "general-dag-semantic",
            &compiled,
            &semantic[variant]
        );
    }
    if (GM_SCALE_ONLY) {
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

    dc_print("snapshot-baseline", &compiled, &snapshot);

    printf(
        "{\"mode\":\"general-dag-controls\","
        "\"compiler_derived_shared_owner_set\":true,"
        "\"shared_owner_degrees\":[4,3,3,2],"
        "\"all_shared_receipts_exact\":%s,"
        "\"all_twelve_shared_edges_exact\":%s,"
        "\"all_shared_pair_nonalias_at_projection\":%s,"
        "\"all_six_shared_owner_pairs_nonalias\":%s,"
        "\"all_shared_live_after_root_inverse\":%s,"
        "\"nested_lifecycle_order_observed\":%s,"
        "\"all_semantic_boundary_hashes_distinct\":%s,"
        "\"primary_reuse_boundary_different\":%s,"
        "\"coefficient_oblivious_schedule\":%s,"
        "\"wrong_root_inverse\":%s,"
        "\"missing_root_inverse\":%s,"
        "\"snapshot_separate_weaker_path\":true,"
        "\"snapshot_post_projection_inverse_operations\":0,"
        "\"accepted_working_slots\":%zu,"
        "\"accepted_native_operation_calls\":%llu,"
        "\"accepted_leaf_encode_calls\":%llu,"
        "\"wrong_root_restoration_abs\":%.12g,"
        "\"missing_root_restoration_abs\":%.12g,"
        "\"same_carrier_transactions\":%u,"
        "\"repeated_reuse_max_restoration_abs\":%.12g}\n",
        gm_all_receipts_exact(&semantic[0]) ? "true" : "false",
        gm_all_shared_edges_exact(&semantic[0]) ? "true" : "false",
        semantic[0].shared_all_pair_nonalias_at_projection
            ? "true"
            : "false",
        semantic[0].shared_pair_nonalias_pairs_at_projection == 6U
            ? "true"
            : "false",
        semantic[0].shared_live_after_root_inverse
            ? "true"
            : "false",
        gm_lifecycle_order(&semantic[0]) ? "true" : "false",
        gm_semantic_hashes_distinct(semantic) ? "true" : "false",
        semantic[0].boundary.hash != semantic[1].boundary.hash
            ? "true"
            : "false",
        ga_stats_same_schedule(
            &semantic[0].stats, &semantic[1].stats
        ) ? "true" : "false",
        wrong.restoration_max_abs > 1.0e-6 ? "true" : "false",
        missing.restoration_max_abs > 1.0e-6 ? "true" : "false",
        compiled.graph.count,
        (unsigned long long)(
            semantic[0].forward_operations
            + semantic[0].inverse_operations
        ),
        (unsigned long long)(
            semantic[0].forward_leaf_encodes
            + semantic[0].inverse_leaf_encodes
        ),
        wrong.restoration_max_abs,
        missing.restoration_max_abs,
        (unsigned)(GM_RUN_VARIANTS + GM_REUSE_CYCLES),
        repeated_max_abs
    );
    return 0;
#endif
}
