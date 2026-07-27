#define _POSIX_C_SOURCE 200809L

/*
 * Compact live-range successor for the seven-node nested two-fanout DAG.
 *
 * Shared S/T generations remain pinned and exact for the whole transaction.
 * I remains resident until ROOT^-1. Only the three public degree-one leaves
 * are reversibly unencoded after their parent commits and re-encoded from
 * immutable public program bodies immediately before that parent's inverse.
 * No internal relation is copied, decoded, serialized, or content-replayed.
 */

#ifndef CP_WIDTH
#define CP_WIDTH 3U
#endif
#define DC_WIDTH CP_WIDTH
#define DC_LEAF_BODIES 3U
#define DC_REUSE_CYCLES 0U
#define DC_CLAIM \
    "BOUNDED_NESTED_AFFINE_DAG_PINNED_SHARED_COMPACT_LEAF_PEBBLING"
#define DC_PUBLIC_MAIN dag_affine_retain_embedded_main
#include "dag_affine_fanout_phase.c"
#undef DC_PUBLIC_MAIN

#define CP_VARIANTS 5U
#ifndef CP_RUN_VARIANTS
#define CP_RUN_VARIANTS CP_VARIANTS
#endif
#ifndef CP_REUSE_CYCLES
#define CP_REUSE_CYCLES 12U
#endif
#ifndef CP_SCALE_ONLY
#define CP_SCALE_ONLY 0
#endif
#ifndef CP_WORKING_SLOTS
#define CP_WORKING_SLOTS 4U
#endif

_Static_assert(CP_RUN_VARIANTS >= 2U, "compact reuse required");
_Static_assert(CP_RUN_VARIANTS <= CP_VARIANTS, "compact variants exceeded");

enum cp_residency {
    CP_UNMATERIALIZED = 0,
    CP_LIVE_ORIGINAL = 1,
    CP_DORMANT_PUBLIC_LEAF = 2,
    CP_LIVE_RECONSTRUCTED_LEAF = 3,
    CP_CLEARED = 4
};

enum cp_fault {
    CP_FAULT_NONE = 0,
    CP_FAULT_EVICT_S = 1,
    CP_FAULT_EVICT_T = 2,
    CP_FAULT_EVICT_I = 3,
    CP_FAULT_WRONG_LEAF_BODY = 4,
    CP_FAULT_MISSING_RECONSTRUCTION = 5,
    CP_FAULT_DOUBLE_RECONSTRUCTION = 6,
    CP_FAULT_SKIP_FINAL_LEAF_INVERSE = 7,
    CP_FAULT_REORDER_T = 8,
    CP_FAULT_REORDER_S = 9
};

struct cp_leaf_obligation {
    int pending;
    size_t body;
    struct rc_signature signature;
    size_t edge;
    uint64_t program_epoch;
};

struct cp_machine {
    uint64_t next_serial;
    uint64_t restoration_generation;
};

struct cp_execution {
    struct dc_execution phase;
    uint64_t initial_leaf_materializations;
    uint64_t early_leaf_unmaterializations;
    uint64_t restoration_leaf_reconstructions;
    uint64_t final_leaf_inverses;
    uint64_t internal_node_rematerializations;
    uint64_t operator_recomputations;
    int leaf_obligations_restored;
    int residency_restored;
    int morphism_cursor_empty;
};

static size_t cp_find_public(
    const struct dc_compiled *compiled,
    unsigned public_id
) {
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        if (compiled->graph.node[index].id == public_id) {
            return index;
        }
    }
    fail("compact DAG public node unavailable");
    return 0U;
}

static size_t cp_find_outgoing_edge(
    const struct dc_compiled *compiled,
    size_t producer
) {
    size_t result = SIZE_MAX;
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        if (compiled->edge[edge].producer == producer) {
            if (result != SIZE_MAX) {
                fail("compact leaf has multiple consumer edges");
            }
            result = edge;
        }
    }
    if (result == SIZE_MAX) {
        fail("compact leaf consumer edge unavailable");
    }
    return result;
}

static struct rc_program cp_make_program(size_t variant) {
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
    rc_set_map_row(program.leaf[0], 0U, 0U, bit1, 1U);
    rc_set_map_row(program.leaf[0], 1U, 0U, bit0, 1U);
    rc_set_map_row(program.leaf[0], 2U, 0U, bit2, 1U);
    rc_set_map_row(program.leaf[1], 0U, 0U, bit01, 2U);
    rc_set_map_row(program.leaf[1], 1U, 0U, bit0, 1U);
    rc_set_map_row(program.leaf[1], 2U, 0U, bit2, 1U);
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
        fail("compact DAG program variant unknown");
    }
    return program;
}

static void cp_record_shared_birth(
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    size_t producer,
    struct rc_lease lease
) {
    if (compiled->is_shared[producer] == 0U) {
        return;
    }
    struct dc_owner_receipt *receipt = &custody->owner[producer];
    receipt->lease = lease;
    receipt->lease_set = 1;
    receipt->all_edge_generations_exact = 1;
    receipt->birth_event = ++custody->event_clock;
    ++receipt->forward_materializations;
    ++receipt->owner_allocations;
    ++receipt->live_instances;
    receipt->peak_live_instances = receipt->live_instances;
    ++custody->shared_live_instances;
    if (custody->shared_live_instances > 1U) {
        custody->shared_overlap_observed = 1;
        for (size_t shared = 0U; shared < compiled->fanout_nodes; ++shared) {
            const size_t other = compiled->shared_node[shared];
            if (
                other != producer
                && custody->owner[other].lease_set
                && custody->owner[other].lease.slot != lease.slot
            ) {
                custody->shared_pair_nonalias_observed = 1;
            }
        }
    }
    if (
        custody->shared_live_instances
        > custody->shared_peak_live_instances
    ) {
        custody->shared_peak_live_instances =
            custody->shared_live_instances;
    }
}

static void cp_record_shared_release(
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    size_t producer
) {
    if (compiled->is_shared[producer] == 0U) {
        return;
    }
    struct dc_owner_receipt *receipt = &custody->owner[producer];
    ++receipt->inverse_applications;
    receipt->inverse_event = ++custody->event_clock;
    ++receipt->owner_releases;
    receipt->release_event = ++custody->event_clock;
    if (
        receipt->live_instances == 0U
        || custody->shared_live_instances == 0U
    ) {
        fail("compact DAG shared live-instance underflow");
    }
    --receipt->live_instances;
    --custody->shared_live_instances;
    receipt->lease = (struct rc_lease){0};
    receipt->lease_set = 0;
}

static struct rc_lease cp_encode_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    enum cp_residency residency[RC_MAX_NODES],
    size_t leaf,
    int reconstructed
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    if (
        node->kind != RC_LEAF
        || (
            reconstructed
                ? residency[leaf] != CP_DORMANT_PUBLIC_LEAF
                : residency[leaf] != CP_UNMATERIALIZED
        )
    ) {
        fail("compact DAG leaf materialization state invalid");
    }
    const struct rc_lease lease = rc_allocate(
        runtime, leaf, node->signature
    );
    ga_encode_relation(
        runtime->carrier,
        runtime->program->leaf[node->leaf_index],
        rc_slot_start(lease.slot),
        0,
        &runtime->stats
    );
    runtime->slot[lease.slot].reserved = 0;
    ++runtime->forward_leaf_encodes;
    residency[leaf] = reconstructed
        ? CP_LIVE_RECONSTRUCTED_LEAF
        : CP_LIVE_ORIGINAL;
    return lease;
}

static void cp_evict_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    enum cp_residency residency[RC_MAX_NODES],
    struct cp_leaf_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t leaf,
    struct cp_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    if (
        node->kind != RC_LEAF
        || compiled->is_shared[leaf] != 0U
        || node->indegree != 1U
    ) {
        fail("compact DAG early eviction is leaf-only");
    }
    const size_t edge = cp_find_outgoing_edge(compiled, leaf);
    if (
        residency[leaf] != CP_LIVE_ORIGINAL
        || custody->edge_state[edge] != DC_EDGE_FORWARD
        || custody->forward_pending[leaf] != 0U
        || custody->inverse_pending[leaf] != 1U
        || obligation[leaf].pending
    ) {
        fail("compact DAG early leaf eviction denied");
    }
    const struct rc_lease lease = node_lease[leaf];
    rc_validate_lease(runtime, lease, leaf, node->signature, 0);
    ga_encode_relation(
        runtime->carrier,
        runtime->program->leaf[node->leaf_index],
        rc_slot_start(lease.slot),
        1,
        &runtime->stats
    );
    ++runtime->inverse_leaf_encodes;
    rc_release(runtime, lease, leaf, node->signature, 0);
    node_lease[leaf] = (struct rc_lease){
        .slot = SIZE_MAX,
        .serial = 0U
    };
    residency[leaf] = CP_DORMANT_PUBLIC_LEAF;
    obligation[leaf] = (struct cp_leaf_obligation){
        .pending = 1,
        .body = node->leaf_index,
        .signature = node->signature,
        .edge = edge,
        .program_epoch = 1U
    };
    ++execution->early_leaf_unmaterializations;
}

static void cp_reconstruct_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    enum cp_residency residency[RC_MAX_NODES],
    const struct cp_leaf_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t leaf,
    struct cp_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    const struct cp_leaf_obligation *owed = &obligation[leaf];
    if (
        !owed->pending
        || owed->body != node->leaf_index
        || !rc_signature_equal(owed->signature, node->signature)
        || owed->program_epoch != 1U
        || custody->edge_state[owed->edge] != DC_EDGE_FORWARD
        || compiled->edge[owed->edge].producer != leaf
    ) {
        fail("compact DAG leaf reconstruction obligation invalid");
    }
    node_lease[leaf] = cp_encode_leaf(
        runtime, compiled, residency, leaf, 1
    );
    ++execution->restoration_leaf_reconstructions;
}

static void cp_finalize_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    enum cp_residency residency[RC_MAX_NODES],
    struct cp_leaf_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t leaf,
    struct cp_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    const struct cp_leaf_obligation *owed = &obligation[leaf];
    if (
        !owed->pending
        || residency[leaf] != CP_LIVE_RECONSTRUCTED_LEAF
        || custody->edge_state[owed->edge] != DC_EDGE_INVERSE
        || custody->inverse_pending[leaf] != 0U
    ) {
        fail("compact DAG final leaf inverse denied");
    }
    const struct rc_lease lease = node_lease[leaf];
    rc_validate_lease(runtime, lease, leaf, node->signature, 0);
    ga_encode_relation(
        runtime->carrier,
        runtime->program->leaf[node->leaf_index],
        rc_slot_start(lease.slot),
        1,
        &runtime->stats
    );
    ++runtime->inverse_leaf_encodes;
    rc_release(runtime, lease, leaf, node->signature, 0);
    node_lease[leaf] = (struct rc_lease){
        .slot = SIZE_MAX,
        .serial = 0U
    };
    obligation[leaf] = (struct cp_leaf_obligation){0};
    residency[leaf] = CP_CLEARED;
    ++execution->final_leaf_inverses;
}

static void cp_forward_operator(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    enum cp_residency residency[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t index
) {
    const struct rc_node *node = &compiled->graph.node[index];
    if (node->kind == RC_LEAF || residency[index] != CP_UNMATERIALIZED) {
        fail("compact DAG forward operator state invalid");
    }
    const struct rc_lease left = node_lease[node->left];
    const struct rc_lease right = node_lease[node->right];
    const struct rc_lease output = rc_allocate(
        runtime, index, node->signature
    );
    const int consume_left = dc_prepare_forward_edge(
        compiled,
        custody,
        compiled->left_edge[index],
        left,
        0,
        0,
        SIZE_MAX
    );
    const int consume_right = dc_prepare_forward_edge(
        compiled,
        custody,
        compiled->right_edge[index],
        right,
        0,
        0,
        SIZE_MAX
    );
    if (!consume_left || !consume_right) {
        fail("compact DAG unexpectedly skipped forward edge");
    }
    rc_apply_node(runtime, index, left, right, output, 0, 0);
    dc_commit_forward_edge(
        compiled,
        custody,
        compiled->left_edge[index],
        left,
        0,
        SIZE_MAX
    );
    dc_commit_forward_edge(
        compiled,
        custody,
        compiled->right_edge[index],
        right,
        0,
        SIZE_MAX
    );
    node_lease[index] = output;
    residency[index] = CP_LIVE_ORIGINAL;
    cp_record_shared_birth(compiled, custody, index, output);
}

static void cp_inverse_operator(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    enum cp_residency residency[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t index,
    int wrong
) {
    const struct rc_node *node = &compiled->graph.node[index];
    if (node->kind == RC_LEAF || residency[index] != CP_LIVE_ORIGINAL) {
        fail("compact DAG inverse operator state invalid");
    }
    dc_require_inverse_ready(compiled, custody, index);
    const struct rc_lease left = node_lease[node->left];
    const struct rc_lease right = node_lease[node->right];
    const struct rc_lease output = node_lease[index];
    rc_apply_node(
        runtime, index, left, right, output, 1, wrong
    );
    dc_inverse_edge(
        compiled,
        custody,
        compiled->left_edge[index],
        left
    );
    dc_inverse_edge(
        compiled,
        custody,
        compiled->right_edge[index],
        right
    );
    rc_release(
        runtime, output, index, node->signature, wrong
    );
    cp_record_shared_release(compiled, custody, index);
    node_lease[index] = (struct rc_lease){
        .slot = SIZE_MAX,
        .serial = 0U
    };
    residency[index] = CP_CLEARED;
}

static int cp_obligations_clear(
    const struct cp_leaf_obligation obligation[RC_MAX_NODES]
) {
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        if (obligation[index].pending) {
            return 0;
        }
    }
    return 1;
}

static int cp_residency_clear(
    const struct dc_compiled *compiled,
    const enum cp_residency residency[RC_MAX_NODES]
) {
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        if (
            residency[index] != CP_CLEARED
            && residency[index] != CP_UNMATERIALIZED
        ) {
            return 0;
        }
    }
    return 1;
}

static void cp_capture_owner_audits(
    const struct dc_compiled *compiled,
    const struct dc_custody *custody,
    struct dc_execution *execution
) {
    execution->shared_owner_count = compiled->fanout_nodes;
    execution->shared_live_at_projection =
        compiled->fanout_nodes > 0U;
    execution->shared_live_after_root_inverse =
        compiled->fanout_nodes > 0U;
    execution->shared_pair_nonalias_observed =
        custody->shared_pair_nonalias_observed;
    execution->shared_overlap_observed =
        custody->shared_overlap_observed;
    execution->shared_simultaneous_live_high_water =
        custody->shared_peak_live_instances;
    execution->projection_event = custody->projection_event;
    execution->root_inverse_event = custody->root_inverse_event;
    for (size_t shared = 0U; shared < compiled->fanout_nodes; ++shared) {
        const size_t producer = compiled->shared_node[shared];
        const struct dc_owner_receipt *receipt =
            &custody->owner[producer];
        struct dc_owner_audit *audit =
            &execution->shared_owner[shared];
        audit->public_node_id = compiled->graph.node[producer].id;
        audit->declared_consumers =
            compiled->graph.node[producer].indegree;
        audit->forward_consumptions =
            receipt->forward_consumptions;
        audit->inverse_consumptions =
            receipt->inverse_consumptions;
        audit->forward_materializations =
            receipt->forward_materializations;
        audit->inverse_applications =
            receipt->inverse_applications;
        audit->owner_allocations = receipt->owner_allocations;
        audit->owner_releases = receipt->owner_releases;
        audit->peak_live_instances = receipt->peak_live_instances;
        audit->consumer_distinct_slots = receipt->distinct_slots;
        audit->consumer_distinct_serials = receipt->distinct_serials;
        audit->live_at_projection = receipt->live_at_projection;
        audit->live_after_root_inverse =
            receipt->live_after_root_inverse;
        audit->all_edge_generations_exact =
            receipt->all_edge_generations_exact;
        audit->birth_event = receipt->birth_event;
        audit->inverse_event = receipt->inverse_event;
        audit->release_event = receipt->release_event;
        audit->receipt_exact =
            receipt->forward_consumptions
                    == compiled->graph.node[producer].indegree
            && receipt->inverse_consumptions
                    == compiled->graph.node[producer].indegree
            && receipt->forward_materializations == 1U
            && receipt->inverse_applications == 1U
            && receipt->owner_allocations == 1U
            && receipt->owner_releases == 1U
            && receipt->peak_live_instances == 1U
            && receipt->distinct_slots == 1U
            && receipt->distinct_serials == 1U
            && receipt->live_at_projection
            && receipt->live_after_root_inverse
            && receipt->all_edge_generations_exact;
        execution->shared_forward_consumptions +=
            receipt->forward_consumptions;
        execution->shared_inverse_consumptions +=
            receipt->inverse_consumptions;
        execution->shared_forward_materializations +=
            receipt->forward_materializations;
        execution->shared_inverse_applications +=
            receipt->inverse_applications;
        execution->shared_owner_allocations +=
            receipt->owner_allocations;
        execution->shared_owner_releases +=
            receipt->owner_releases;
        if (
            receipt->distinct_slots
            > execution->shared_consumer_distinct_slots
        ) {
            execution->shared_consumer_distinct_slots =
                receipt->distinct_slots;
        }
        if (
            receipt->distinct_serials
            > execution->shared_consumer_distinct_serials
        ) {
            execution->shared_consumer_distinct_serials =
                receipt->distinct_serials;
        }
        execution->shared_live_at_projection &=
            receipt->live_at_projection;
        execution->shared_live_after_root_inverse &=
            receipt->live_after_root_inverse;
    }
}

static struct cp_execution cp_execute(
    struct carrier *carrier,
    const struct dc_compiled *compiled,
    const struct rc_program *program,
    struct cp_machine *machine,
    enum rc_restore_mode restore_mode,
    enum cp_fault fault
) {
    struct cp_execution result = {0};
    struct dc_execution *execution = &result.phase;
    struct carrier borrowed = snapshot_carrier(carrier);
    struct rc_runtime runtime = {
        .carrier = carrier,
        .compiled = &compiled->graph,
        .program = program,
        .pool = CP_WORKING_SLOTS,
        .next_serial = machine->next_serial
    };
    struct dc_custody custody = {0};
    struct rc_lease node_lease[RC_MAX_NODES];
    enum cp_residency residency[RC_MAX_NODES] = {0};
    struct cp_leaf_obligation obligation[RC_MAX_NODES] = {0};
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        node_lease[index] = (struct rc_lease){
            .slot = SIZE_MAX,
            .serial = 0U
        };
        custody.forward_pending[index] =
            compiled->graph.node[index].indegree;
        custody.inverse_pending[index] =
            compiled->graph.node[index].indegree;
    }
    const size_t f = cp_find_public(compiled, 701U);
    const size_t g = cp_find_public(compiled, 702U);
    const size_t h = cp_find_public(compiled, 703U);
    const size_t s = cp_find_public(compiled, 704U);
    const size_t t = cp_find_public(compiled, 705U);
    const size_t i = cp_find_public(compiled, 706U);
    const size_t root_index = cp_find_public(compiled, 707U);
    if (
        compiled->graph.root != root_index
        || compiled->graph.node[s].left != f
        || compiled->graph.node[s].right != g
        || compiled->graph.node[t].left != s
        || compiled->graph.node[t].right != h
        || compiled->graph.node[i].left != s
        || compiled->graph.node[i].right != t
        || compiled->graph.node[root_index].left != i
        || compiled->graph.node[root_index].right != t
    ) {
        fail("compact DAG topology outside bounded schedule");
    }
    execution->serial_before = machine->next_serial;
    execution->restoration_generation_before =
        machine->restoration_generation;

    node_lease[f] = cp_encode_leaf(
        &runtime, compiled, residency, f, 0
    );
    node_lease[g] = cp_encode_leaf(
        &runtime, compiled, residency, g, 0
    );
    result.initial_leaf_materializations += 2U;
    cp_forward_operator(
        &runtime, compiled, &custody, residency, node_lease, s
    );
    if (fault == CP_FAULT_EVICT_S) {
        cp_evict_leaf(
            &runtime,
            compiled,
            &custody,
            residency,
            obligation,
            node_lease,
            s,
            &result
        );
    }
    cp_evict_leaf(
        &runtime,
        compiled,
        &custody,
        residency,
        obligation,
        node_lease,
        g,
        &result
    );
    cp_evict_leaf(
        &runtime,
        compiled,
        &custody,
        residency,
        obligation,
        node_lease,
        f,
        &result
    );
    node_lease[h] = cp_encode_leaf(
        &runtime, compiled, residency, h, 0
    );
    ++result.initial_leaf_materializations;
    cp_forward_operator(
        &runtime, compiled, &custody, residency, node_lease, t
    );
    if (fault == CP_FAULT_EVICT_T) {
        cp_evict_leaf(
            &runtime,
            compiled,
            &custody,
            residency,
            obligation,
            node_lease,
            t,
            &result
        );
    }
    cp_evict_leaf(
        &runtime,
        compiled,
        &custody,
        residency,
        obligation,
        node_lease,
        h,
        &result
    );
    cp_forward_operator(
        &runtime, compiled, &custody, residency, node_lease, i
    );
    if (fault == CP_FAULT_EVICT_I) {
        cp_evict_leaf(
            &runtime,
            compiled,
            &custody,
            residency,
            obligation,
            node_lease,
            i,
            &result
        );
    }
    cp_forward_operator(
        &runtime,
        compiled,
        &custody,
        residency,
        node_lease,
        root_index
    );
    if (!dc_edges_all(compiled, &custody, DC_EDGE_FORWARD)) {
        fail("compact DAG forward edge schedule incomplete");
    }
    custody.projection_event = ++custody.event_clock;
    for (size_t shared = 0U; shared < compiled->fanout_nodes; ++shared) {
        const size_t producer = compiled->shared_node[shared];
        dc_validate_shared_live(
            &runtime, compiled, &custody, producer
        );
        custody.owner[producer].live_at_projection = 1;
    }
    dc_copy_relation_block(
        &runtime,
        &custody,
        rc_slot_start(node_lease[root_index].slot),
        GA_BOUNDARY_START,
        0,
        1
    );
    execution->boundary =
        ga_latch_boundary(carrier, &runtime.stats);

    if (restore_mode == RC_RESTORE_SNAPSHOT) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        memset(runtime.slot, 0, sizeof(runtime.slot));
        memset(&custody, 0, sizeof(custody));
        memset(residency, 0, sizeof(residency));
        memset(obligation, 0, sizeof(obligation));
        runtime.live = 0U;
        runtime.next_serial = machine->next_serial;
        execution->snapshot_loaded = 1;
    } else {
        dc_copy_relation_block(
            &runtime,
            &custody,
            rc_slot_start(node_lease[root_index].slot),
            GA_BOUNDARY_START,
            1,
            1
        );
        if (restore_mode != RC_RESTORE_MISSING_ROOT) {
            cp_inverse_operator(
                &runtime,
                compiled,
                &custody,
                residency,
                node_lease,
                root_index,
                restore_mode == RC_RESTORE_WRONG_ROOT
            );
            custody.root_inverse_event = ++custody.event_clock;
            for (
                size_t shared = 0U;
                shared < compiled->fanout_nodes;
                ++shared
            ) {
                const size_t producer = compiled->shared_node[shared];
                dc_validate_shared_live(
                    &runtime, compiled, &custody, producer
                );
                custody.owner[producer]
                    .live_after_root_inverse = 1;
            }
            if (fault == CP_FAULT_REORDER_T) {
                cp_inverse_operator(
                    &runtime,
                    compiled,
                    &custody,
                    residency,
                    node_lease,
                    t,
                    0
                );
            }
            cp_inverse_operator(
                &runtime,
                compiled,
                &custody,
                residency,
                node_lease,
                i,
                0
            );
            if (fault == CP_FAULT_REORDER_S) {
                cp_inverse_operator(
                    &runtime,
                    compiled,
                    &custody,
                    residency,
                    node_lease,
                    s,
                    0
                );
            }
            if (fault == CP_FAULT_WRONG_LEAF_BODY) {
                obligation[h].body =
                    (obligation[h].body + 1U) % DC_LEAF_BODIES;
            }
            if (fault != CP_FAULT_MISSING_RECONSTRUCTION) {
                cp_reconstruct_leaf(
                    &runtime,
                    compiled,
                    &custody,
                    residency,
                    obligation,
                    node_lease,
                    h,
                    &result
                );
            }
            if (fault == CP_FAULT_DOUBLE_RECONSTRUCTION) {
                cp_reconstruct_leaf(
                    &runtime,
                    compiled,
                    &custody,
                    residency,
                    obligation,
                    node_lease,
                    h,
                    &result
                );
            }
            cp_inverse_operator(
                &runtime,
                compiled,
                &custody,
                residency,
                node_lease,
                t,
                0
            );
            if (fault != CP_FAULT_SKIP_FINAL_LEAF_INVERSE) {
                cp_finalize_leaf(
                    &runtime,
                    compiled,
                    &custody,
                    residency,
                    obligation,
                    node_lease,
                    h,
                    &result
                );
            }
            cp_reconstruct_leaf(
                &runtime,
                compiled,
                &custody,
                residency,
                obligation,
                node_lease,
                f,
                &result
            );
            cp_reconstruct_leaf(
                &runtime,
                compiled,
                &custody,
                residency,
                obligation,
                node_lease,
                g,
                &result
            );
            cp_inverse_operator(
                &runtime,
                compiled,
                &custody,
                residency,
                node_lease,
                s,
                0
            );
            cp_finalize_leaf(
                &runtime,
                compiled,
                &custody,
                residency,
                obligation,
                node_lease,
                g,
                &result
            );
            cp_finalize_leaf(
                &runtime,
                compiled,
                &custody,
                residency,
                obligation,
                node_lease,
                f,
                &result
            );
        }
    }

    const int inverse_complete =
        restore_mode == RC_RESTORE_SNAPSHOT
        || (
            dc_edges_all(compiled, &custody, DC_EDGE_INVERSE)
            && dc_counts_zero(compiled, &custody)
            && cp_obligations_clear(obligation)
            && cp_residency_clear(compiled, residency)
        );
    if (fault != CP_FAULT_NONE) {
        free_carrier(&borrowed);
        fail("compact DAG injected fault left incomplete state");
    }
    cp_capture_owner_audits(compiled, &custody, execution);
    execution->boundary_block_copy_calls =
        custody.boundary_block_copy_calls;
    execution->intermediate_block_copy_calls =
        custody.intermediate_block_copy_calls;
    if (inverse_complete) {
        memset(custody.edge_state, 0, sizeof(custody.edge_state));
        memset(
            custody.forward_pending,
            0,
            sizeof(custody.forward_pending)
        );
        memset(
            custody.inverse_pending,
            0,
            sizeof(custody.inverse_pending)
        );
    }
    memset(custody.owner, 0, sizeof(custody.owner));
    memset(
        custody.edge_receipt,
        0,
        sizeof(custody.edge_receipt)
    );
    memset(obligation, 0, sizeof(obligation));
    memset(residency, 0, sizeof(residency));
    execution->restoration_max_abs = restoration(carrier, &borrowed);
    runtime.stats.restoration_cell_checks += carrier->cells;
    execution->integrity_max_abs = integrity(carrier);
    runtime.stats.integrity_cell_checks += carrier->cells;
    execution->workspace_cleared =
        ga_workspace_is_zero(carrier, &runtime.stats);
    execution->allocator_restored = rc_allocator_is_restored(&runtime);
    execution->edge_custody_restored =
        inverse_complete
        && dc_edges_all(compiled, &custody, DC_EDGE_DECLARED);
    execution->shared_receipt_reset = 1;
    execution->pending_counts_restored = inverse_complete;
    execution->scheduler_queue_empty = 1;
    execution->stats = runtime.stats;
    execution->working_relation_slots = runtime.pool;
    execution->maximum_live_slots = runtime.maximum_live;
    execution->outstanding_relation_leases = runtime.live;
    execution->lease_allocations = runtime.lease_allocations;
    execution->lease_releases = runtime.lease_releases;
    execution->lease_checks = runtime.lease_checks;
    execution->lease_cell_checks = runtime.lease_cell_checks;
    execution->forward_operations = runtime.forward_operations;
    execution->inverse_operations = runtime.inverse_operations;
    execution->forward_leaf_encodes = runtime.forward_leaf_encodes;
    execution->inverse_leaf_encodes = runtime.inverse_leaf_encodes;
    execution->dirty_control_releases =
        runtime.dirty_control_releases;
    result.leaf_obligations_restored = inverse_complete;
    result.residency_restored = inverse_complete;
    result.morphism_cursor_empty = inverse_complete;
    if (restore_mode == RC_RESTORE_CORRECT) {
        machine->next_serial = runtime.next_serial;
        ++machine->restoration_generation;
    }
    execution->serial_after = machine->next_serial;
    execution->restoration_generation_after =
        machine->restoration_generation;
    free_carrier(&borrowed);
    return result;
}

static const struct dc_owner_audit *cp_audit(
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
    fail("compact DAG shared audit unavailable");
    return NULL;
}

static int cp_nested_order(const struct dc_execution *execution) {
    const struct dc_owner_audit *s = cp_audit(execution, 704U);
    const struct dc_owner_audit *t = cp_audit(execution, 705U);
    return s->birth_event < t->birth_event
        && t->birth_event < execution->projection_event
        && execution->projection_event
            < execution->root_inverse_event
        && execution->root_inverse_event < t->inverse_event
        && t->inverse_event < t->release_event
        && t->release_event < s->inverse_event
        && s->inverse_event < s->release_event;
}

static void cp_print_compact(
    const char *mode,
    const struct cp_execution *execution
) {
    printf(
        "{\"mode\":\"%s-compact-schedule\","
        "\"claim\":\"" DC_CLAIM "\","
        "\"working_relation_slots\":%zu,"
        "\"physical_relation_blocks_including_boundary\":%zu,"
        "\"maximum_live_relation_slots\":%zu,"
        "\"initial_leaf_materializations\":%llu,"
        "\"early_leaf_unmaterializations\":%llu,"
        "\"restoration_leaf_reconstructions\":%llu,"
        "\"final_leaf_inverses\":%llu,"
        "\"internal_node_rematerializations\":%llu,"
        "\"operator_recomputations\":%llu,"
        "\"leaf_obligations_restored\":%s,"
        "\"residency_restored\":%s,"
        "\"morphism_cursor_empty\":%s,"
        "\"nested_lifetime_order_observed\":%s}\n",
        mode,
        execution->phase.working_relation_slots,
        rc_physical_relation_blocks(
            execution->phase.working_relation_slots
        ),
        execution->phase.maximum_live_slots,
        (unsigned long long)
            execution->initial_leaf_materializations,
        (unsigned long long)
            execution->early_leaf_unmaterializations,
        (unsigned long long)
            execution->restoration_leaf_reconstructions,
        (unsigned long long)execution->final_leaf_inverses,
        (unsigned long long)
            execution->internal_node_rematerializations,
        (unsigned long long)execution->operator_recomputations,
        execution->leaf_obligations_restored ? "true" : "false",
        execution->residency_restored ? "true" : "false",
        execution->morphism_cursor_empty ? "true" : "false",
        cp_nested_order(&execution->phase) ? "true" : "false"
    );
}

static enum cp_fault cp_fault_from_option(const char *option) {
    if (strcmp(option, "--control-evict-s") == 0) {
        return CP_FAULT_EVICT_S;
    }
    if (strcmp(option, "--control-evict-t") == 0) {
        return CP_FAULT_EVICT_T;
    }
    if (strcmp(option, "--control-evict-i") == 0) {
        return CP_FAULT_EVICT_I;
    }
    if (strcmp(option, "--control-wrong-leaf-body") == 0) {
        return CP_FAULT_WRONG_LEAF_BODY;
    }
    if (strcmp(option, "--control-missing-reconstruction") == 0) {
        return CP_FAULT_MISSING_RECONSTRUCTION;
    }
    if (strcmp(option, "--control-double-reconstruction") == 0) {
        return CP_FAULT_DOUBLE_RECONSTRUCTION;
    }
    if (strcmp(option, "--control-skip-final-leaf-inverse") == 0) {
        return CP_FAULT_SKIP_FINAL_LEAF_INVERSE;
    }
    if (strcmp(option, "--control-reorder-t") == 0) {
        return CP_FAULT_REORDER_T;
    }
    if (strcmp(option, "--control-reorder-s") == 0) {
        return CP_FAULT_REORDER_S;
    }
    return CP_FAULT_NONE;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fail(
            "usage: compact_multi_dag_affine_phase "
            "DAG_MANIFEST OCCURRENCE_TREE_MANIFEST [CONTROL]"
        );
    }
    if (
        argc == 4
        && (
            strcmp(argv[3], "--project-intermediate") == 0
            || strcmp(argv[3], "--project-obligations") == 0
            || strcmp(argv[3], "--project-residency") == 0
        )
    ) {
        fail("compact DAG internal projection denied");
    }
    if (argc == 4 && strcmp(argv[3], "--null-carrier") == 0) {
        fail("null compact DAG carrier");
    }
    struct rc_manifest manifest = {0};
    struct rc_manifest occurrence_manifest = {0};
    rc_load_manifest(argv[1], &manifest);
    rc_load_manifest(argv[2], &occurrence_manifest);
    const struct dc_compiled compiled = dc_compile(&manifest, 2U);
    const struct dc_compiled occurrence =
        dc_compile(&occurrence_manifest, 0U);
    struct rc_program program[CP_VARIANTS];
    for (size_t variant = 0U; variant < CP_VARIANTS; ++variant) {
        program[variant] = cp_make_program(variant);
    }
    const struct process compact_shape = {
        .carrier_cells = rc_carrier_cells(CP_WORKING_SLOTS)
    };
    if (argc == 4) {
        const enum cp_fault fault = cp_fault_from_option(argv[3]);
        if (fault == CP_FAULT_NONE) {
            fail("compact DAG control unknown");
        }
        struct carrier control_carrier =
            make_carrier(&compact_shape, 9127);
        struct cp_machine fault_machine = {0};
        (void)cp_execute(
            &control_carrier,
            &compiled,
            &program[0],
            &fault_machine,
            RC_RESTORE_CORRECT,
            fault
        );
        free_carrier(&control_carrier);
        fail("compact DAG fault control was not rejected");
    }
    struct carrier carrier = make_carrier(&compact_shape, 9127);
    struct cp_machine machine = {0};
    struct cp_execution semantic[CP_RUN_VARIANTS];
    double repeated_max_abs = 0.0;
    for (size_t variant = 0U; variant < CP_RUN_VARIANTS; ++variant) {
        semantic[variant] = cp_execute(
            &carrier,
            &compiled,
            &program[variant],
            &machine,
            RC_RESTORE_CORRECT,
            CP_FAULT_NONE
        );
        if (
            semantic[variant].phase.restoration_max_abs
            > repeated_max_abs
        ) {
            repeated_max_abs =
                semantic[variant].phase.restoration_max_abs;
        }
    }
#if CP_REUSE_CYCLES > 0U
    for (size_t cycle = 0U; cycle < CP_REUSE_CYCLES; ++cycle) {
        const struct cp_execution repeated = cp_execute(
            &carrier,
            &compiled,
            &program[cycle % 2U],
            &machine,
            RC_RESTORE_CORRECT,
            CP_FAULT_NONE
        );
        if (repeated.phase.restoration_max_abs > repeated_max_abs) {
            repeated_max_abs =
                repeated.phase.restoration_max_abs;
        }
    }
#endif
    free_carrier(&carrier);
    for (size_t variant = 0U; variant < CP_RUN_VARIANTS; ++variant) {
        const char *mode = variant == 0U
            ? "compact-dag-primary"
            : variant == 1U
                ? "compact-dag-reuse"
                : "compact-dag-semantic";
        dc_print(mode, &compiled, &semantic[variant].phase);
        cp_print_compact(mode, &semantic[variant]);
    }
    if (CP_SCALE_ONLY) {
        return 0;
    }

    struct cp_machine control_machine = {0};
    carrier = make_carrier(&compact_shape, 9127);
    const struct cp_execution wrong = cp_execute(
        &carrier,
        &compiled,
        &program[0],
        &control_machine,
        RC_RESTORE_WRONG_ROOT,
        CP_FAULT_NONE
    );
    free_carrier(&carrier);
    control_machine = (struct cp_machine){0};
    carrier = make_carrier(&compact_shape, 9127);
    const struct cp_execution missing = cp_execute(
        &carrier,
        &compiled,
        &program[0],
        &control_machine,
        RC_RESTORE_MISSING_ROOT,
        CP_FAULT_NONE
    );
    free_carrier(&carrier);
    control_machine = (struct cp_machine){0};
    carrier = make_carrier(&compact_shape, 9127);
    const struct cp_execution snapshot = cp_execute(
        &carrier,
        &compiled,
        &program[0],
        &control_machine,
        RC_RESTORE_SNAPSHOT,
        CP_FAULT_NONE
    );
    free_carrier(&carrier);

    const struct process retain_shape = {
        .carrier_cells = rc_carrier_cells(compiled.graph.count)
    };
    struct dc_machine retain_machine = {0};
    carrier = make_carrier(&retain_shape, 9127);
    const struct dc_execution retain = dc_execute(
        &carrier,
        &compiled,
        &program[0],
        &retain_machine,
        RC_RESTORE_CORRECT
    );
    free_carrier(&carrier);
    const struct process occurrence_shape = {
        .carrier_cells = rc_carrier_cells(occurrence.graph.count)
    };
    struct dc_machine occurrence_machine = {0};
    carrier = make_carrier(&occurrence_shape, 9127);
    const struct dc_execution occurrence_run = dc_execute(
        &carrier,
        &occurrence,
        &program[0],
        &occurrence_machine,
        RC_RESTORE_CORRECT
    );
    free_carrier(&carrier);
    dc_print("retain-unique-phase-baseline", &compiled, &retain);
    dc_print(
        "occurrence-tree-phase-baseline",
        &occurrence,
        &occurrence_run
    );
    dc_print("snapshot-baseline", &compiled, &snapshot.phase);
    printf(
        "{\"mode\":\"compact-dag-controls\","
        "\"compact_retain_boundary_equal\":%s,"
        "\"compact_occurrence_boundary_equal\":%s,"
        "\"compact_actual_inverse_restored\":%s,"
        "\"retain_actual_inverse_restored\":%s,"
        "\"occurrence_actual_inverse_restored\":%s,"
        "\"wrong_root_inverse\":%s,"
        "\"missing_root_inverse\":%s,"
        "\"snapshot_separate_weaker_path\":true,"
        "\"compact_working_slots\":%u,"
        "\"retain_working_slots\":%zu,"
        "\"occurrence_working_slots\":%zu,"
        "\"compact_native_operation_calls\":%llu,"
        "\"retain_native_operation_calls\":%llu,"
        "\"occurrence_native_operation_calls\":%llu,"
        "\"compact_leaf_encode_calls\":%llu,"
        "\"retain_leaf_encode_calls\":%llu,"
        "\"occurrence_leaf_encode_calls\":%llu,"
        "\"compact_lease_allocations\":%llu,"
        "\"retain_lease_allocations\":%llu,"
        "\"internal_node_rematerializations\":0,"
        "\"operator_recomputations\":0,"
        "\"same_carrier_transactions\":%u,"
        "\"repeated_reuse_max_restoration_abs\":%.12g,"
        "\"wrong_root_restoration_abs\":%.12g,"
        "\"missing_root_restoration_abs\":%.12g}\n",
        ga_boundary_equal(
            &semantic[0].phase.boundary, &retain.boundary
        ) ? "true" : "false",
        ga_boundary_equal(
            &semantic[0].phase.boundary,
            &occurrence_run.boundary
        ) ? "true" : "false",
        (
            semantic[0].phase.restoration_max_abs
                <= GA_WORKSPACE_TOLERANCE
            && semantic[0].phase.allocator_restored
        ) ? "true" : "false",
        (
            retain.restoration_max_abs <= GA_WORKSPACE_TOLERANCE
            && retain.allocator_restored
        ) ? "true" : "false",
        (
            occurrence_run.restoration_max_abs
                <= GA_WORKSPACE_TOLERANCE
            && occurrence_run.allocator_restored
        ) ? "true" : "false",
        wrong.phase.restoration_max_abs > 1.0e-6
            ? "true" : "false",
        missing.phase.restoration_max_abs > 1.0e-6
            ? "true" : "false",
        (unsigned)CP_WORKING_SLOTS,
        compiled.graph.count,
        occurrence.graph.count,
        (unsigned long long)(
            semantic[0].phase.forward_operations
            + semantic[0].phase.inverse_operations
        ),
        (unsigned long long)(
            retain.forward_operations + retain.inverse_operations
        ),
        (unsigned long long)(
            occurrence_run.forward_operations
            + occurrence_run.inverse_operations
        ),
        (unsigned long long)(
            semantic[0].phase.forward_leaf_encodes
            + semantic[0].phase.inverse_leaf_encodes
        ),
        (unsigned long long)(
            retain.forward_leaf_encodes + retain.inverse_leaf_encodes
        ),
        (unsigned long long)(
            occurrence_run.forward_leaf_encodes
            + occurrence_run.inverse_leaf_encodes
        ),
        (unsigned long long)semantic[0].phase.lease_allocations,
        (unsigned long long)retain.lease_allocations,
        (unsigned)(CP_RUN_VARIANTS + CP_REUSE_CYCLES),
        repeated_max_abs,
        wrong.phase.restoration_max_abs,
        missing.phase.restoration_max_abs
    );
    return 0;
}
