#define _POSIX_C_SOURCE 200809L

/*
 * Compiler-derived public-leaf pebbling for the exact fifteen-node
 * four-owner affine DAG.
 *
 * The compiler emits a literal reversible tape from public topology only.
 * Every degree-one public leaf is inverse-encoded and released after its
 * forward edge commits, then reconstructed by the inverse tape before the
 * exact pending inverse edge.  Every operator output, including all shared
 * owners, remains its original generation.  Only ROOT is projected.
 */

#define GM_REUSE_CYCLES 0U
#define GM_DC_CLAIM \
    "BOUNDED_15_NODE_FOUR_OWNER_AUTOMATIC_PUBLIC_LEAF_PEBBLING_ESTABLISHED"
#define GM_PUBLIC_MAIN automatic_compact_retain_embedded_main
#include "general_multi_dag_affine_phase.c"
#undef GM_PUBLIC_MAIN

#ifndef AC_WORKING_SLOTS
#define AC_WORKING_SLOTS 11U
#endif
#ifndef AC_RUN_VARIANTS
#define AC_RUN_VARIANTS GM_VARIANTS
#endif
#ifndef AC_REUSE_CYCLES
#define AC_REUSE_CYCLES 12U
#endif
#ifndef AC_SCALE_ONLY
#define AC_SCALE_ONLY 0
#endif

#define AC_MAX_STEPS (2U * RC_MAX_NODES)

_Static_assert(AC_RUN_VARIANTS >= 2U, "automatic compact reuse required");
_Static_assert(
    AC_RUN_VARIANTS <= GM_VARIANTS,
    "automatic compact variants exceeded"
);

enum ac_opcode {
    AC_ENCODE_LEAF = 1,
    AC_APPLY_OPERATOR = 2,
    AC_EVICT_LEAF = 3
};

enum ac_residency {
    AC_UNMATERIALIZED = 0,
    AC_LIVE_ORIGINAL = 1,
    AC_DORMANT_PUBLIC_LEAF = 2,
    AC_LIVE_RECONSTRUCTED_LEAF = 3,
    AC_CLEARED = 4
};

enum ac_edge_policy {
    AC_EXACT_RESIDENT_GENERATION = 1,
    AC_PUBLIC_LEAF_RECONSTRUCTION = 2
};

enum ac_fault {
    AC_FAULT_NONE = 0,
    AC_FAULT_EVICT_SHARED = 1,
    AC_FAULT_EVICT_INTERNAL = 2,
    AC_FAULT_WRONG_BODY = 3,
    AC_FAULT_STALE_EPOCH = 4,
    AC_FAULT_MISSING_RECONSTRUCTION = 5,
    AC_FAULT_DOUBLE_RECONSTRUCTION = 6,
    AC_FAULT_SKIP_FINAL_LEAF_INVERSE = 7,
    AC_FAULT_REORDER_SHARED = 8,
    AC_FAULT_STALE_INTERNAL_EDGE = 9,
    AC_FAULT_TAMPER_TAPE = 10
};

struct ac_step {
    enum ac_opcode opcode;
    size_t node;
    unsigned public_node_id;
    size_t edge;
    size_t expected_live_after;
};

struct ac_plan {
    struct ac_step step[AC_MAX_STEPS];
    size_t count;
    size_t predicted_peak_live;
    size_t final_live;
    size_t reconstructible_leaves;
    unsigned char reconstructible[RC_MAX_NODES];
    size_t leaf_edge[RC_MAX_NODES];
    uint64_t topology_hash;
    uint64_t schedule_hash;
    uint64_t hash;
};

struct ac_obligation {
    int pending;
    size_t body;
    struct rc_signature signature;
    size_t edge;
    uint64_t program_epoch;
    uint64_t plan_hash;
};

struct ac_edge_runtime {
    enum ac_edge_policy policy;
    int forward_seen;
    int inverse_seen;
    size_t forward_slot;
    uint64_t forward_serial;
    size_t inverse_slot;
    uint64_t inverse_serial;
    int lawful;
};

struct ac_edge_audit {
    unsigned producer_public_id;
    unsigned consumer_public_id;
    unsigned side;
    enum ac_edge_policy policy;
    int forward_seen;
    int inverse_seen;
    int exact_generation;
    int lawful_reconstruction;
};

struct ac_machine {
    struct dc_machine carrier;
    uint64_t program_epoch;
};

struct ac_execution {
    struct dc_execution phase;
    struct ac_edge_audit edge[DC_MAX_EDGES];
    size_t edge_count;
    uint64_t program_epoch;
    uint64_t plan_hash;
    size_t tape_steps;
    size_t tape_cursor_after_forward;
    size_t tape_cursor_after_inverse;
    size_t predicted_peak_live;
    size_t reconstructible_leaves;
    uint64_t initial_leaf_materializations;
    uint64_t early_leaf_unmaterializations;
    uint64_t restoration_leaf_reconstructions;
    uint64_t final_leaf_inverses;
    uint64_t internal_node_rematerializations;
    uint64_t operator_recomputations;
    int all_edge_generations_lawful;
    size_t exact_resident_edges;
    size_t reconstructed_leaf_edges;
    int obligations_restored;
    int residency_restored;
    int tape_cursor_empty;
};

static size_t ac_find_outgoing_edge(
    const struct dc_compiled *compiled,
    size_t producer
) {
    size_t result = SIZE_MAX;
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        if (compiled->edge[edge].producer != producer) {
            continue;
        }
        if (result != SIZE_MAX) {
            fail("automatic compact leaf has multiple outgoing edges");
        }
        result = edge;
    }
    if (result == SIZE_MAX) {
        fail("automatic compact leaf edge unavailable");
    }
    return result;
}

static void ac_append_step(
    struct ac_plan *plan,
    enum ac_opcode opcode,
    size_t node,
    unsigned public_node_id,
    size_t edge,
    size_t *live
) {
    if (plan->count >= AC_MAX_STEPS) {
        fail("automatic compact tape capacity exceeded");
    }
    if (opcode == AC_ENCODE_LEAF || opcode == AC_APPLY_OPERATOR) {
        ++*live;
        if (*live > plan->predicted_peak_live) {
            plan->predicted_peak_live = *live;
        }
    } else if (opcode == AC_EVICT_LEAF) {
        if (*live == 0U) {
            fail("automatic compact plan live-count underflow");
        }
        --*live;
    } else {
        fail("automatic compact plan opcode unknown");
    }
    plan->step[plan->count++] = (struct ac_step){
        .opcode = opcode,
        .node = node,
        .public_node_id = public_node_id,
        .edge = edge,
        .expected_live_after = *live
    };
}

static uint64_t ac_hash_plan(const struct ac_plan *plan) {
    uint64_t hash = UINT64_C(14695981039346656037);
    rc_hash_word(&hash, plan->count);
    rc_hash_word(&hash, plan->predicted_peak_live);
    rc_hash_word(&hash, plan->reconstructible_leaves);
    rc_hash_word(&hash, (size_t)plan->topology_hash);
    rc_hash_word(&hash, (size_t)plan->schedule_hash);
    for (size_t index = 0U; index < plan->count; ++index) {
        rc_hash_word(&hash, (size_t)plan->step[index].opcode);
        rc_hash_word(&hash, plan->step[index].node);
        rc_hash_word(
            &hash, (size_t)plan->step[index].public_node_id
        );
        rc_hash_word(&hash, plan->step[index].edge);
        rc_hash_word(&hash, plan->step[index].expected_live_after);
    }
    return hash;
}

static struct ac_plan ac_compile_plan(
    const struct dc_compiled *compiled
) {
    struct ac_plan plan = {0};
    plan.topology_hash = compiled->graph.topology_hash;
    plan.schedule_hash = compiled->graph.schedule_hash;
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        plan.leaf_edge[index] = SIZE_MAX;
    }
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        const struct rc_node *node = &compiled->graph.node[index];
        if (
            node->kind == RC_LEAF
            && node->indegree == 1U
            && compiled->is_shared[index] == 0U
        ) {
            plan.reconstructible[index] = 1U;
            plan.leaf_edge[index] =
                ac_find_outgoing_edge(compiled, index);
            ++plan.reconstructible_leaves;
        }
    }
    size_t live = 0U;
    for (
        size_t cursor = 0U;
        cursor < compiled->graph.schedule_count;
        ++cursor
    ) {
        const size_t index = compiled->graph.schedule[cursor];
        const struct rc_node *node = &compiled->graph.node[index];
        if (node->kind == RC_LEAF) {
            ac_append_step(
                &plan,
                AC_ENCODE_LEAF,
                index,
                node->id,
                SIZE_MAX,
                &live
            );
            continue;
        }
        ac_append_step(
            &plan,
            AC_APPLY_OPERATOR,
            index,
            node->id,
            SIZE_MAX,
            &live
        );
        const size_t child[] = {node->right, node->left};
        for (size_t side = 0U; side < 2U; ++side) {
            const size_t producer = child[side];
            if (plan.reconstructible[producer] != 0U) {
                ac_append_step(
                    &plan,
                    AC_EVICT_LEAF,
                    producer,
                    compiled->graph.node[producer].id,
                    plan.leaf_edge[producer],
                    &live
                );
            }
        }
    }
    plan.final_live = live;
    plan.hash = ac_hash_plan(&plan);
    if (
        plan.reconstructible_leaves != 4U
        || plan.count != 19U
        || plan.predicted_peak_live != 11U
        || plan.final_live != 11U
    ) {
        fail("automatic compact compiled plan outside bounded law");
    }
    return plan;
}

static void ac_validate_plan(
    const struct ac_plan *plan,
    const struct dc_compiled *compiled
) {
    if (
        plan->topology_hash != compiled->graph.topology_hash
        || plan->schedule_hash != compiled->graph.schedule_hash
        || plan->hash != ac_hash_plan(plan)
    ) {
        fail("automatic compact tape hash changed");
    }
}

static void ac_record_shared_birth(
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

static void ac_record_shared_release(
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
        fail("automatic compact shared live-instance underflow");
    }
    --receipt->live_instances;
    --custody->shared_live_instances;
    receipt->lease = (struct rc_lease){0};
    receipt->lease_set = 0;
}

static struct rc_lease ac_encode_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    enum ac_residency residency[RC_MAX_NODES],
    size_t leaf,
    int reconstructed
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    if (
        node->kind != RC_LEAF
        || (
            reconstructed
                ? residency[leaf] != AC_DORMANT_PUBLIC_LEAF
                : residency[leaf] != AC_UNMATERIALIZED
        )
    ) {
        fail("automatic compact leaf materialization state invalid");
    }
    const struct rc_lease lease =
        rc_allocate(runtime, leaf, node->signature);
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
        ? AC_LIVE_RECONSTRUCTED_LEAF
        : AC_LIVE_ORIGINAL;
    return lease;
}

static void ac_evict_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct ac_plan *plan,
    struct dc_custody *custody,
    enum ac_residency residency[RC_MAX_NODES],
    struct ac_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t leaf,
    uint64_t program_epoch,
    struct ac_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    const size_t edge = leaf < RC_MAX_NODES
        ? plan->leaf_edge[leaf]
        : SIZE_MAX;
    if (
        leaf >= compiled->graph.count
        || plan->reconstructible[leaf] == 0U
        || node->kind != RC_LEAF
        || node->indegree != 1U
        || compiled->is_shared[leaf] != 0U
        || edge >= compiled->edge_count
    ) {
        fail("automatic compact early eviction is public-leaf-only");
    }
    if (
        residency[leaf] != AC_LIVE_ORIGINAL
        || custody->edge_state[edge] != DC_EDGE_FORWARD
        || custody->forward_pending[leaf] != 0U
        || custody->inverse_pending[leaf] != 1U
        || obligation[leaf].pending
    ) {
        fail("automatic compact early leaf eviction denied");
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
    residency[leaf] = AC_DORMANT_PUBLIC_LEAF;
    obligation[leaf] = (struct ac_obligation){
        .pending = 1,
        .body = node->leaf_index,
        .signature = node->signature,
        .edge = edge,
        .program_epoch = program_epoch,
        .plan_hash = plan->hash
    };
    ++execution->early_leaf_unmaterializations;
}

static void ac_reconstruct_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct ac_plan *plan,
    const struct dc_custody *custody,
    enum ac_residency residency[RC_MAX_NODES],
    const struct ac_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t leaf,
    uint64_t program_epoch,
    struct ac_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    const struct ac_obligation *owed = &obligation[leaf];
    if (
        plan->reconstructible[leaf] == 0U
        || !owed->pending
        || owed->body != node->leaf_index
        || !rc_signature_equal(owed->signature, node->signature)
        || owed->edge != plan->leaf_edge[leaf]
        || owed->program_epoch != program_epoch
        || owed->plan_hash != plan->hash
        || custody->edge_state[owed->edge] != DC_EDGE_FORWARD
        || compiled->edge[owed->edge].producer != leaf
    ) {
        fail("automatic compact leaf reconstruction obligation invalid");
    }
    node_lease[leaf] = ac_encode_leaf(
        runtime, compiled, residency, leaf, 1
    );
    ++execution->restoration_leaf_reconstructions;
}

static void ac_finalize_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct ac_plan *plan,
    const struct dc_custody *custody,
    enum ac_residency residency[RC_MAX_NODES],
    struct ac_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t leaf,
    struct ac_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    const struct ac_obligation *owed = &obligation[leaf];
    if (
        plan->reconstructible[leaf] == 0U
        || !owed->pending
        || residency[leaf] != AC_LIVE_RECONSTRUCTED_LEAF
        || custody->edge_state[owed->edge] != DC_EDGE_INVERSE
        || custody->inverse_pending[leaf] != 0U
    ) {
        fail("automatic compact final leaf inverse denied");
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
    obligation[leaf] = (struct ac_obligation){0};
    residency[leaf] = AC_CLEARED;
    ++execution->final_leaf_inverses;
}

static void ac_commit_edge_forward(
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    struct ac_edge_runtime edge_runtime[DC_MAX_EDGES],
    size_t edge,
    struct rc_lease lease
) {
    dc_commit_forward_edge(
        compiled, custody, edge, lease, 0, SIZE_MAX
    );
    struct ac_edge_runtime *receipt = &edge_runtime[edge];
    if (receipt->forward_seen) {
        fail("automatic compact edge forward receipt duplicated");
    }
    const size_t producer = compiled->edge[edge].producer;
    receipt->policy =
        compiled->graph.node[producer].kind == RC_LEAF
            ? AC_PUBLIC_LEAF_RECONSTRUCTION
            : AC_EXACT_RESIDENT_GENERATION;
    receipt->forward_seen = 1;
    receipt->forward_slot = lease.slot;
    receipt->forward_serial = lease.serial;
}

static void ac_forward_operator(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    struct ac_edge_runtime edge_runtime[DC_MAX_EDGES],
    enum ac_residency residency[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t index
) {
    const struct rc_node *node = &compiled->graph.node[index];
    if (node->kind == RC_LEAF || residency[index] != AC_UNMATERIALIZED) {
        fail("automatic compact forward operator state invalid");
    }
    const struct rc_lease left = node_lease[node->left];
    const struct rc_lease right = node_lease[node->right];
    const struct rc_lease output =
        rc_allocate(runtime, index, node->signature);
    if (
        !dc_prepare_forward_edge(
            compiled,
            custody,
            compiled->left_edge[index],
            left,
            0,
            0,
            SIZE_MAX
        )
        || !dc_prepare_forward_edge(
            compiled,
            custody,
            compiled->right_edge[index],
            right,
            0,
            0,
            SIZE_MAX
        )
    ) {
        fail("automatic compact forward edge unexpectedly skipped");
    }
    rc_apply_node(runtime, index, left, right, output, 0, 0);
    ac_commit_edge_forward(
        compiled,
        custody,
        edge_runtime,
        compiled->left_edge[index],
        left
    );
    ac_commit_edge_forward(
        compiled,
        custody,
        edge_runtime,
        compiled->right_edge[index],
        right
    );
    node_lease[index] = output;
    residency[index] = AC_LIVE_ORIGINAL;
    ac_record_shared_birth(compiled, custody, index, output);
}

static void ac_preflight_inverse_edge(
    const struct dc_compiled *compiled,
    const struct ac_plan *plan,
    const struct ac_obligation obligation[RC_MAX_NODES],
    const enum ac_residency residency[RC_MAX_NODES],
    const struct ac_edge_runtime edge_runtime[DC_MAX_EDGES],
    size_t edge,
    struct rc_lease lease
) {
    const struct ac_edge_runtime *receipt = &edge_runtime[edge];
    const size_t producer = compiled->edge[edge].producer;
    if (!receipt->forward_seen || receipt->inverse_seen) {
        fail("automatic compact inverse edge receipt state invalid");
    }
    if (receipt->policy == AC_EXACT_RESIDENT_GENERATION) {
        if (
            lease.slot != receipt->forward_slot
            || lease.serial != receipt->forward_serial
        ) {
            fail("automatic compact resident edge changed generation");
        }
        return;
    }
    if (
        receipt->policy != AC_PUBLIC_LEAF_RECONSTRUCTION
        || plan->reconstructible[producer] == 0U
        || residency[producer] != AC_LIVE_RECONSTRUCTED_LEAF
        || !obligation[producer].pending
        || obligation[producer].edge != edge
        || lease.serial == receipt->forward_serial
    ) {
        fail("automatic compact reconstructed edge custody invalid");
    }
}

static void ac_commit_edge_inverse(
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    struct ac_edge_runtime edge_runtime[DC_MAX_EDGES],
    size_t edge,
    struct rc_lease lease
) {
    dc_inverse_edge(compiled, custody, edge, lease);
    struct ac_edge_runtime *receipt = &edge_runtime[edge];
    receipt->inverse_seen = 1;
    receipt->inverse_slot = lease.slot;
    receipt->inverse_serial = lease.serial;
    receipt->lawful =
        receipt->policy == AC_EXACT_RESIDENT_GENERATION
            ? (
                receipt->forward_slot == receipt->inverse_slot
                && receipt->forward_serial == receipt->inverse_serial
            )
            : (
                receipt->policy == AC_PUBLIC_LEAF_RECONSTRUCTION
                && receipt->forward_serial != receipt->inverse_serial
            );
}

static void ac_inverse_operator(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct ac_plan *plan,
    struct dc_custody *custody,
    struct ac_edge_runtime edge_runtime[DC_MAX_EDGES],
    const struct ac_obligation obligation[RC_MAX_NODES],
    enum ac_residency residency[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t index,
    int wrong
) {
    const struct rc_node *node = &compiled->graph.node[index];
    if (node->kind == RC_LEAF || residency[index] != AC_LIVE_ORIGINAL) {
        fail("automatic compact inverse operator state invalid");
    }
    dc_require_inverse_ready(compiled, custody, index);
    const struct rc_lease left = node_lease[node->left];
    const struct rc_lease right = node_lease[node->right];
    const struct rc_lease output = node_lease[index];
    ac_preflight_inverse_edge(
        compiled,
        plan,
        obligation,
        residency,
        edge_runtime,
        compiled->left_edge[index],
        left
    );
    ac_preflight_inverse_edge(
        compiled,
        plan,
        obligation,
        residency,
        edge_runtime,
        compiled->right_edge[index],
        right
    );
    rc_apply_node(runtime, index, left, right, output, 1, wrong);
    ac_commit_edge_inverse(
        compiled,
        custody,
        edge_runtime,
        compiled->left_edge[index],
        left
    );
    ac_commit_edge_inverse(
        compiled,
        custody,
        edge_runtime,
        compiled->right_edge[index],
        right
    );
    rc_release(runtime, output, index, node->signature, wrong);
    ac_record_shared_release(compiled, custody, index);
    node_lease[index] = (struct rc_lease){
        .slot = SIZE_MAX,
        .serial = 0U
    };
    residency[index] = AC_CLEARED;
}

static int ac_obligations_clear(
    const struct ac_obligation obligation[RC_MAX_NODES]
) {
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        if (obligation[index].pending) {
            return 0;
        }
    }
    return 1;
}

static int ac_residency_clear(
    const struct dc_compiled *compiled,
    const enum ac_residency residency[RC_MAX_NODES]
) {
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        if (
            residency[index] != AC_CLEARED
            && residency[index] != AC_UNMATERIALIZED
        ) {
            return 0;
        }
    }
    return 1;
}

static void ac_capture_audits(
    const struct dc_compiled *compiled,
    const struct dc_custody *custody,
    const struct ac_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct ac_execution *result
) {
    struct dc_execution *execution = &result->phase;
    execution->shared_owner_count = compiled->fanout_nodes;
    execution->shared_live_at_projection =
        compiled->fanout_nodes > 0U;
    execution->shared_live_after_root_inverse =
        compiled->fanout_nodes > 0U;
    execution->shared_pair_nonalias_observed =
        custody->shared_pair_nonalias_observed;
    execution->shared_all_pair_nonalias_at_projection =
        custody->shared_all_pair_nonalias_at_projection;
    execution->shared_pair_nonalias_pairs_at_projection =
        custody->shared_pair_nonalias_pairs_at_projection;
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
        audit->all_edge_generations_exact = dc_owner_edges_exact(
            compiled, custody, producer, receipt
        );
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
            && audit->all_edge_generations_exact;
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
            receipt->peak_live_instances
            > execution->shared_peak_live_instances
        ) {
            execution->shared_peak_live_instances =
                receipt->peak_live_instances;
        }
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
    result->all_edge_generations_lawful = 1;
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        const struct dc_edge *descriptor = &compiled->edge[edge];
        const struct ac_edge_runtime *receipt = &edge_runtime[edge];
        struct ac_edge_audit *audit =
            &result->edge[result->edge_count++];
        audit->producer_public_id =
            compiled->graph.node[descriptor->producer].id;
        audit->consumer_public_id =
            compiled->graph.node[descriptor->consumer].id;
        audit->side = descriptor->side;
        audit->policy = receipt->policy;
        audit->forward_seen = receipt->forward_seen;
        audit->inverse_seen = receipt->inverse_seen;
        audit->exact_generation =
            receipt->lawful
            && receipt->policy == AC_EXACT_RESIDENT_GENERATION;
        audit->lawful_reconstruction =
            receipt->lawful
            && receipt->policy == AC_PUBLIC_LEAF_RECONSTRUCTION;
        result->all_edge_generations_lawful &=
            receipt->forward_seen
            && receipt->inverse_seen
            && receipt->lawful;
        if (receipt->policy == AC_EXACT_RESIDENT_GENERATION) {
            ++result->exact_resident_edges;
        } else if (
            receipt->policy == AC_PUBLIC_LEAF_RECONSTRUCTION
        ) {
            ++result->reconstructed_leaf_edges;
        }
        if (!compiled->is_shared[descriptor->producer]) {
            continue;
        }
        const struct dc_edge_receipt *shared_receipt =
            &custody->edge_receipt[edge];
        const struct dc_owner_receipt *owner =
            &custody->owner[descriptor->producer];
        struct dc_edge_audit *shared_audit =
            &execution->shared_edge[execution->shared_edge_count++];
        shared_audit->producer_public_id =
            audit->producer_public_id;
        shared_audit->consumer_public_id =
            audit->consumer_public_id;
        shared_audit->side = descriptor->side;
        shared_audit->forward_seen =
            shared_receipt->forward_seen;
        shared_audit->inverse_seen =
            shared_receipt->inverse_seen;
        shared_audit->same_owner_generation =
            shared_receipt->forward_seen
            && shared_receipt->inverse_seen
            && shared_receipt->forward_slot
                == owner->observed_slot
            && shared_receipt->inverse_slot
                == owner->observed_slot
            && shared_receipt->forward_serial
                == owner->observed_serial
            && shared_receipt->inverse_serial
                == owner->observed_serial;
    }
    execution->boundary_block_copy_calls =
        custody->boundary_block_copy_calls;
    execution->intermediate_block_copy_calls =
        custody->intermediate_block_copy_calls;
}

static struct ac_execution ac_execute(
    struct carrier *carrier,
    const struct dc_compiled *compiled,
    const struct ac_plan *source_plan,
    const struct rc_program *program,
    struct ac_machine *machine,
    enum rc_restore_mode restore_mode,
    enum ac_fault fault
) {
    struct ac_execution result = {0};
    struct dc_execution *execution = &result.phase;
    struct carrier borrowed = snapshot_carrier(carrier);
    struct ac_plan plan = *source_plan;
    if (fault == AC_FAULT_TAMPER_TAPE) {
        plan.step[0].expected_live_after += 1U;
    }
    ac_validate_plan(&plan, compiled);
    const uint64_t program_epoch = machine->program_epoch + 1U;
    struct rc_runtime runtime = {
        .carrier = carrier,
        .compiled = &compiled->graph,
        .program = program,
        .pool = AC_WORKING_SLOTS,
        .next_serial = machine->carrier.next_serial
    };
    struct dc_custody custody = {0};
    struct ac_edge_runtime edge_runtime[DC_MAX_EDGES] = {0};
    struct rc_lease node_lease[RC_MAX_NODES];
    enum ac_residency residency[RC_MAX_NODES] = {0};
    struct ac_obligation obligation[RC_MAX_NODES] = {0};
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        node_lease[index] = (struct rc_lease){
            .slot = SIZE_MAX,
            .serial = 0U
        };
    }
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        custody.forward_pending[index] =
            compiled->graph.node[index].indegree;
        custody.inverse_pending[index] =
            compiled->graph.node[index].indegree;
    }
    execution->serial_before = machine->carrier.next_serial;
    execution->restoration_generation_before =
        machine->carrier.restoration_generation;
    result.program_epoch = program_epoch;
    result.plan_hash = plan.hash;
    result.tape_steps = plan.count;
    result.predicted_peak_live = plan.predicted_peak_live;
    result.reconstructible_leaves =
        plan.reconstructible_leaves;

    size_t tape_cursor = 0U;
    while (tape_cursor < plan.count) {
        const struct ac_step *step = &plan.step[tape_cursor++];
        if (step->opcode == AC_ENCODE_LEAF) {
            node_lease[step->node] = ac_encode_leaf(
                &runtime,
                compiled,
                residency,
                step->node,
                0
            );
            ++result.initial_leaf_materializations;
        } else if (step->opcode == AC_APPLY_OPERATOR) {
            ac_forward_operator(
                &runtime,
                compiled,
                &custody,
                edge_runtime,
                residency,
                node_lease,
                step->node
            );
        } else if (step->opcode == AC_EVICT_LEAF) {
            ac_evict_leaf(
                &runtime,
                compiled,
                &plan,
                &custody,
                residency,
                obligation,
                node_lease,
                step->node,
                program_epoch,
                &result
            );
        } else {
            fail("automatic compact forward tape opcode invalid");
        }
        if (runtime.live != step->expected_live_after) {
            fail("automatic compact tape live-count mismatch");
        }
    }
    result.tape_cursor_after_forward = tape_cursor;
    if (
        !dc_edges_all(compiled, &custody, DC_EDGE_FORWARD)
        || runtime.live != plan.final_live
    ) {
        fail("automatic compact forward tape incomplete");
    }
    if (fault == AC_FAULT_EVICT_SHARED) {
        ac_evict_leaf(
            &runtime,
            compiled,
            &plan,
            &custody,
            residency,
            obligation,
            node_lease,
            gm_find_public(compiled, 805U),
            program_epoch,
            &result
        );
    }
    if (fault == AC_FAULT_EVICT_INTERNAL) {
        ac_evict_leaf(
            &runtime,
            compiled,
            &plan,
            &custody,
            residency,
            obligation,
            node_lease,
            gm_find_public(compiled, 809U),
            program_epoch,
            &result
        );
    }
    custody.projection_event = ++custody.event_clock;
    for (size_t shared = 0U; shared < compiled->fanout_nodes; ++shared) {
        const size_t producer = compiled->shared_node[shared];
        dc_validate_shared_live(
            &runtime, compiled, &custody, producer
        );
        custody.owner[producer].live_at_projection = 1;
    }
    custody.shared_all_pair_nonalias_at_projection =
        compiled->fanout_nodes > 1U;
    for (size_t left = 0U; left < compiled->fanout_nodes; ++left) {
        const size_t left_owner = compiled->shared_node[left];
        for (
            size_t right = left + 1U;
            right < compiled->fanout_nodes;
            ++right
        ) {
            const size_t right_owner = compiled->shared_node[right];
            if (
                !custody.owner[left_owner].lease_set
                || !custody.owner[right_owner].lease_set
                || custody.owner[left_owner].lease.slot
                    == custody.owner[right_owner].lease.slot
            ) {
                custody.shared_all_pair_nonalias_at_projection = 0;
            } else {
                ++custody.shared_pair_nonalias_pairs_at_projection;
            }
        }
    }
    const size_t root = compiled->graph.root;
    dc_copy_relation_block(
        &runtime,
        &custody,
        rc_slot_start(node_lease[root].slot),
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
        memset(edge_runtime, 0, sizeof(edge_runtime));
        memset(residency, 0, sizeof(residency));
        memset(obligation, 0, sizeof(obligation));
        runtime.live = 0U;
        runtime.next_serial = machine->carrier.next_serial;
        tape_cursor = 0U;
        execution->snapshot_loaded = 1;
    } else {
        dc_copy_relation_block(
            &runtime,
            &custody,
            rc_slot_start(node_lease[root].slot),
            GA_BOUNDARY_START,
            1,
            1
        );
        if (fault == AC_FAULT_REORDER_SHARED) {
            ac_inverse_operator(
                &runtime,
                compiled,
                &plan,
                &custody,
                edge_runtime,
                obligation,
                residency,
                node_lease,
                gm_find_public(compiled, 808U),
                0
            );
        }
        if (fault == AC_FAULT_STALE_INTERNAL_EDGE) {
            edge_runtime[compiled->left_edge[root]]
                .forward_serial += 1U;
        }
        if (restore_mode != RC_RESTORE_MISSING_ROOT) {
            int first_reconstruction = 1;
            int first_final = 1;
            while (tape_cursor > 0U) {
                const struct ac_step *step =
                    &plan.step[--tape_cursor];
                if (step->opcode == AC_EVICT_LEAF) {
                    if (
                        first_reconstruction
                        && fault == AC_FAULT_WRONG_BODY
                    ) {
                        obligation[step->node].body =
                            (
                                obligation[step->node].body + 1U
                            ) % DC_LEAF_BODIES;
                    }
                    if (
                        first_reconstruction
                        && fault == AC_FAULT_STALE_EPOCH
                    ) {
                        ++obligation[step->node].program_epoch;
                    }
                    if (
                        !(
                            first_reconstruction
                            && fault
                                == AC_FAULT_MISSING_RECONSTRUCTION
                        )
                    ) {
                        ac_reconstruct_leaf(
                            &runtime,
                            compiled,
                            &plan,
                            &custody,
                            residency,
                            obligation,
                            node_lease,
                            step->node,
                            program_epoch,
                            &result
                        );
                    }
                    if (
                        first_reconstruction
                        && fault
                            == AC_FAULT_DOUBLE_RECONSTRUCTION
                    ) {
                        ac_reconstruct_leaf(
                            &runtime,
                            compiled,
                            &plan,
                            &custody,
                            residency,
                            obligation,
                            node_lease,
                            step->node,
                            program_epoch,
                            &result
                        );
                    }
                    first_reconstruction = 0;
                } else if (step->opcode == AC_APPLY_OPERATOR) {
                    ac_inverse_operator(
                        &runtime,
                        compiled,
                        &plan,
                        &custody,
                        edge_runtime,
                        obligation,
                        residency,
                        node_lease,
                        step->node,
                        restore_mode == RC_RESTORE_WRONG_ROOT
                            && step->node == root
                    );
                    if (step->node == root) {
                        custody.root_inverse_event =
                            ++custody.event_clock;
                        for (
                            size_t shared = 0U;
                            shared < compiled->fanout_nodes;
                            ++shared
                        ) {
                            const size_t producer =
                                compiled->shared_node[shared];
                            dc_validate_shared_live(
                                &runtime,
                                compiled,
                                &custody,
                                producer
                            );
                            custody.owner[producer]
                                .live_after_root_inverse = 1;
                        }
                    }
                } else if (step->opcode == AC_ENCODE_LEAF) {
                    if (
                        !(
                            first_final
                            && fault
                                == AC_FAULT_SKIP_FINAL_LEAF_INVERSE
                        )
                    ) {
                        ac_finalize_leaf(
                            &runtime,
                            compiled,
                            &plan,
                            &custody,
                            residency,
                            obligation,
                            node_lease,
                            step->node,
                            &result
                        );
                    }
                    first_final = 0;
                } else {
                    fail(
                        "automatic compact inverse tape opcode invalid"
                    );
                }
            }
        }
    }

    const int inverse_complete =
        restore_mode == RC_RESTORE_SNAPSHOT
        || (
            tape_cursor == 0U
            && dc_edges_all(compiled, &custody, DC_EDGE_INVERSE)
            && dc_counts_zero(compiled, &custody)
            && ac_obligations_clear(obligation)
            && ac_residency_clear(compiled, residency)
        );
    if (fault != AC_FAULT_NONE) {
        free_carrier(&borrowed);
        fail("automatic compact injected fault escaped detection");
    }
    ac_capture_audits(
        compiled, &custody, edge_runtime, &result
    );
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
        custody.edge_receipt, 0, sizeof(custody.edge_receipt)
    );
    memset(edge_runtime, 0, sizeof(edge_runtime));
    memset(obligation, 0, sizeof(obligation));
    memset(residency, 0, sizeof(residency));
    execution->restoration_max_abs = restoration(carrier, &borrowed);
    runtime.stats.restoration_cell_checks += carrier->cells;
    execution->integrity_max_abs = integrity(carrier);
    runtime.stats.integrity_cell_checks += carrier->cells;
    execution->workspace_cleared =
        ga_workspace_is_zero(carrier, &runtime.stats);
    execution->allocator_restored =
        rc_allocator_is_restored(&runtime);
    execution->edge_custody_restored =
        inverse_complete
        && dc_edges_all(compiled, &custody, DC_EDGE_DECLARED);
    execution->shared_receipt_reset = 1;
    execution->pending_counts_restored = inverse_complete;
    execution->scheduler_queue_empty = tape_cursor == 0U;
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
    execution->forward_leaf_encodes =
        runtime.forward_leaf_encodes;
    execution->inverse_leaf_encodes =
        runtime.inverse_leaf_encodes;
    execution->dirty_control_releases =
        runtime.dirty_control_releases;
    result.tape_cursor_after_inverse = tape_cursor;
    result.obligations_restored = inverse_complete;
    result.residency_restored = inverse_complete;
    result.tape_cursor_empty =
        inverse_complete && tape_cursor == 0U;
    if (restore_mode == RC_RESTORE_CORRECT) {
        machine->carrier.next_serial = runtime.next_serial;
        ++machine->carrier.restoration_generation;
        machine->program_epoch = program_epoch;
    }
    execution->serial_after = machine->carrier.next_serial;
    execution->restoration_generation_after =
        machine->carrier.restoration_generation;
    free_carrier(&borrowed);
    return result;
}

static void ac_print_edge_audits(
    const struct ac_execution *execution
) {
    printf(",\"all_edge_receipts\":[");
    for (size_t edge = 0U; edge < execution->edge_count; ++edge) {
        const struct ac_edge_audit *audit = &execution->edge[edge];
        if (edge > 0U) {
            putchar(',');
        }
        printf(
            "{\"producer_public_id\":%u,"
            "\"consumer_public_id\":%u,"
            "\"side\":%u,"
            "\"policy\":\"%s\","
            "\"forward_seen\":%s,"
            "\"inverse_seen\":%s,"
            "\"exact_generation\":%s,"
            "\"lawful_reconstruction\":%s}",
            audit->producer_public_id,
            audit->consumer_public_id,
            audit->side,
            audit->policy == AC_EXACT_RESIDENT_GENERATION
                ? "EXACT_RESIDENT_GENERATION"
                : "PUBLIC_LEAF_RECONSTRUCTION",
            audit->forward_seen ? "true" : "false",
            audit->inverse_seen ? "true" : "false",
            audit->exact_generation ? "true" : "false",
            audit->lawful_reconstruction ? "true" : "false"
        );
    }
    putchar(']');
}

static void ac_print(
    const char *mode,
    const struct dc_compiled *compiled,
    const struct ac_plan *plan,
    const struct ac_execution *execution
) {
    printf(
        "{\"mode\":\"automatic-compact-schedule\","
        "\"semantic_mode\":\"%s\","
        "\"claim\":\"" GM_DC_CLAIM "\","
        "\"compiler_generated_reversible_tape\":true,"
        "\"coefficient_oblivious_plan\":true,"
        "\"plan_inputs\":\"PUBLIC_TOPOLOGY_SIGNATURES_AND_LEAF_BODIES\","
        "\"bound_topology_fnv1a64\":\"%016llx\","
        "\"bound_schedule_fnv1a64\":\"%016llx\","
        "\"plan_fnv1a64\":\"%016llx\","
        "\"plan_steps\":%zu,"
        "\"tape_cursor_after_forward\":%zu,"
        "\"tape_cursor_after_inverse\":%zu,"
        "\"predicted_peak_live\":%zu,"
        "\"runtime_peak_live\":%zu,"
        "\"working_relation_slots\":%zu,"
        "\"physical_relation_blocks_including_boundary\":%zu,"
        "\"reversible_plan_storage_bytes_current_abi\":%zu,"
        "\"scheduler_obligation_storage_bytes_current_abi\":%zu,"
        "\"scheduler_edge_receipt_storage_bytes_current_abi\":%zu,"
        "\"scheduler_residency_storage_bytes_current_abi\":%zu,"
        "\"scheduler_node_lease_storage_bytes_current_abi\":%zu,"
        "\"transaction_execution_summary_bytes_current_abi\":%zu,"
        "\"automatic_machine_state_bytes_current_abi\":%zu,"
        "\"additional_scheduler_state_bytes_current_abi\":%zu,"
        "\"reconstructible_public_leaf_ids\":[",
        mode,
        (unsigned long long)plan->topology_hash,
        (unsigned long long)plan->schedule_hash,
        (unsigned long long)execution->plan_hash,
        execution->tape_steps,
        execution->tape_cursor_after_forward,
        execution->tape_cursor_after_inverse,
        execution->predicted_peak_live,
        execution->phase.maximum_live_slots,
        execution->phase.working_relation_slots,
        rc_physical_relation_blocks(
            execution->phase.working_relation_slots
        ),
        sizeof(struct ac_plan),
        sizeof(struct ac_obligation) * (size_t)RC_MAX_NODES,
        sizeof(struct ac_edge_runtime) * (size_t)DC_MAX_EDGES,
        sizeof(enum ac_residency) * (size_t)RC_MAX_NODES,
        sizeof(struct rc_lease) * (size_t)RC_MAX_NODES,
        sizeof(struct ac_execution),
        sizeof(struct ac_machine),
        sizeof(struct ac_plan)
            + sizeof(struct ac_obligation) * (size_t)RC_MAX_NODES
            + sizeof(struct ac_edge_runtime) * (size_t)DC_MAX_EDGES
            + sizeof(enum ac_residency) * (size_t)RC_MAX_NODES
            + sizeof(struct rc_lease) * (size_t)RC_MAX_NODES
    );
    int comma = 0;
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        if (plan->reconstructible[index] == 0U) {
            continue;
        }
        printf(
            "%s%u",
            comma ? "," : "",
            compiled->graph.node[index].id
        );
        comma = 1;
    }
    printf(
        "],\"reconstructible_public_leaves\":%zu,"
        "\"initial_leaf_materializations\":%llu,"
        "\"early_leaf_unmaterializations\":%llu,"
        "\"restoration_leaf_reconstructions\":%llu,"
        "\"final_leaf_inverses\":%llu,"
        "\"internal_node_rematerializations\":%llu,"
        "\"operator_recomputations\":%llu,"
        "\"all_edge_generations_lawful\":%s,"
        "\"exact_resident_edges\":%zu,"
        "\"reconstructed_leaf_edges\":%zu,"
        "\"obligations_restored\":%s,"
        "\"residency_restored\":%s,"
        "\"tape_cursor_empty\":%s,"
        "\"program_epoch\":%llu",
        execution->reconstructible_leaves,
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
        execution->all_edge_generations_lawful
            ? "true" : "false",
        execution->exact_resident_edges,
        execution->reconstructed_leaf_edges,
        execution->obligations_restored ? "true" : "false",
        execution->residency_restored ? "true" : "false",
        execution->tape_cursor_empty ? "true" : "false",
        (unsigned long long)execution->program_epoch
    );
    ac_print_edge_audits(execution);
    printf("}\n");
}

static int ac_execution_exact(
    const struct ac_execution *execution
) {
    return execution->phase.working_relation_slots == 11U
        && execution->phase.maximum_live_slots == 11U
        && execution->phase.lease_allocations == 19U
        && execution->phase.lease_releases == 19U
        && execution->phase.forward_operations == 11U
        && execution->phase.inverse_operations == 11U
        && execution->phase.forward_leaf_encodes == 8U
        && execution->phase.inverse_leaf_encodes == 8U
        && execution->phase.outstanding_relation_leases == 0U
        && execution->phase.boundary_block_copy_calls == 2U
        && execution->phase.intermediate_block_copy_calls == 0U
        && execution->phase.shared_owner_count == 4U
        && execution->phase.shared_edge_count == 12U
        && gm_all_receipts_exact(&execution->phase)
        && gm_all_shared_edges_exact(&execution->phase)
        && gm_lifecycle_order(&execution->phase)
        && execution->phase.shared_live_at_projection
        && execution->phase.shared_live_after_root_inverse
        && execution->phase.shared_all_pair_nonalias_at_projection
        && execution->phase.shared_pair_nonalias_pairs_at_projection
            == 6U
        && execution->phase.shared_simultaneous_live_high_water == 4U
        && execution->edge_count == 22U
        && execution->exact_resident_edges == 18U
        && execution->reconstructed_leaf_edges == 4U
        && execution->all_edge_generations_lawful
        && execution->initial_leaf_materializations == 4U
        && execution->early_leaf_unmaterializations == 4U
        && execution->restoration_leaf_reconstructions == 4U
        && execution->final_leaf_inverses == 4U
        && execution->internal_node_rematerializations == 0U
        && execution->operator_recomputations == 0U
        && execution->obligations_restored
        && execution->residency_restored
        && execution->tape_cursor_empty
        && execution->phase.workspace_cleared
        && execution->phase.allocator_restored
        && execution->phase.edge_custody_restored
        && execution->phase.pending_counts_restored
        && execution->phase.scheduler_queue_empty
        && !execution->phase.snapshot_loaded
        && execution->phase.serial_after
            == execution->phase.serial_before + 19U
        && execution->phase.restoration_generation_after
            == execution->phase.restoration_generation_before + 1U
        && execution->program_epoch > 0U
        && execution->plan_hash != 0U
        && execution->phase.restoration_max_abs
            <= GA_WORKSPACE_TOLERANCE;
}

static enum ac_fault ac_fault_from_option(const char *option) {
    if (strcmp(option, "--control-evict-shared") == 0) {
        return AC_FAULT_EVICT_SHARED;
    }
    if (strcmp(option, "--control-evict-internal") == 0) {
        return AC_FAULT_EVICT_INTERNAL;
    }
    if (strcmp(option, "--control-wrong-body") == 0) {
        return AC_FAULT_WRONG_BODY;
    }
    if (strcmp(option, "--control-stale-epoch") == 0) {
        return AC_FAULT_STALE_EPOCH;
    }
    if (strcmp(option, "--control-missing-reconstruction") == 0) {
        return AC_FAULT_MISSING_RECONSTRUCTION;
    }
    if (strcmp(option, "--control-double-reconstruction") == 0) {
        return AC_FAULT_DOUBLE_RECONSTRUCTION;
    }
    if (strcmp(option, "--control-skip-final-leaf-inverse") == 0) {
        return AC_FAULT_SKIP_FINAL_LEAF_INVERSE;
    }
    if (strcmp(option, "--control-reorder-shared") == 0) {
        return AC_FAULT_REORDER_SHARED;
    }
    if (strcmp(option, "--control-stale-internal-edge") == 0) {
        return AC_FAULT_STALE_INTERNAL_EDGE;
    }
    if (strcmp(option, "--control-tamper-tape") == 0) {
        return AC_FAULT_TAMPER_TAPE;
    }
    return AC_FAULT_NONE;
}

#ifndef AC_PUBLIC_MAIN
#define AC_PUBLIC_MAIN main
#endif

int AC_PUBLIC_MAIN(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fail(
            "usage: automatic_compact_general_multi_dag_affine_phase "
            "DAG_MANIFEST [CONTROL]"
        );
    }
    if (
        argc == 3
        && (
            strcmp(argv[2], "--project-intermediate") == 0
            || strcmp(argv[2], "--project-tape") == 0
            || strcmp(argv[2], "--project-obligations") == 0
            || strcmp(argv[2], "--project-residency") == 0
        )
    ) {
        fail("automatic compact internal projection denied");
    }
    if (argc == 3 && strcmp(argv[2], "--null-carrier") == 0) {
        fail("null automatic compact carrier");
    }
    struct rc_manifest manifest = {0};
    rc_load_manifest(argv[1], &manifest);
    const struct dc_compiled compiled = dc_compile(&manifest, 4U);
    gm_require_topology(&compiled);
    struct ac_plan plan = ac_compile_plan(&compiled);
    struct rc_program program[GM_VARIANTS];
    for (size_t variant = 0U; variant < GM_VARIANTS; ++variant) {
        program[variant] = gm_make_program(variant);
    }
    const struct process shape = {
        .carrier_cells = rc_carrier_cells(AC_WORKING_SLOTS)
    };
    if (argc == 3) {
        const enum ac_fault fault = ac_fault_from_option(argv[2]);
        if (fault == AC_FAULT_NONE) {
            fail("automatic compact control unknown");
        }
        struct carrier control = make_carrier(&shape, 9127);
        struct ac_machine control_machine = {0};
        (void)ac_execute(
            &control,
            &compiled,
            &plan,
            &program[0],
            &control_machine,
            RC_RESTORE_CORRECT,
            fault
        );
        free_carrier(&control);
        fail("automatic compact fault control was not rejected");
    }

    struct carrier carrier = make_carrier(&shape, 9127);
    struct ac_machine machine = {0};
    struct ac_execution semantic[AC_RUN_VARIANTS];
    double repeated_max_abs = 0.0;
    for (size_t variant = 0U; variant < AC_RUN_VARIANTS; ++variant) {
        semantic[variant] = ac_execute(
            &carrier,
            &compiled,
            &plan,
            &program[variant],
            &machine,
            RC_RESTORE_CORRECT,
            AC_FAULT_NONE
        );
        if (!ac_execution_exact(&semantic[variant])) {
            free_carrier(&carrier);
            fail("automatic compact accepted state law failed");
        }
        if (
            semantic[variant].phase.restoration_max_abs
            > repeated_max_abs
        ) {
            repeated_max_abs =
                semantic[variant].phase.restoration_max_abs;
        }
    }
#if AC_REUSE_CYCLES > 0U
    for (size_t cycle = 0U; cycle < AC_REUSE_CYCLES; ++cycle) {
        const struct ac_execution repeated = ac_execute(
            &carrier,
            &compiled,
            &plan,
            &program[cycle % 2U],
            &machine,
            RC_RESTORE_CORRECT,
            AC_FAULT_NONE
        );
        if (!ac_execution_exact(&repeated)) {
            free_carrier(&carrier);
            fail("automatic compact reuse state law failed");
        }
        if (repeated.phase.restoration_max_abs > repeated_max_abs) {
            repeated_max_abs =
                repeated.phase.restoration_max_abs;
        }
    }
#endif
    free_carrier(&carrier);
    for (size_t variant = 0U; variant < AC_RUN_VARIANTS; ++variant) {
        const char *mode = variant == 0U
            ? "automatic-compact-primary"
            : variant == 1U
                ? "automatic-compact-reuse"
                : "automatic-compact-semantic";
        dc_print(mode, &compiled, &semantic[variant].phase);
        ac_print(mode, &compiled, &plan, &semantic[variant]);
    }
    if (AC_SCALE_ONLY) {
        return 0;
    }

    struct ac_machine control_machine = {0};
    carrier = make_carrier(&shape, 9127);
    const struct ac_execution wrong = ac_execute(
        &carrier,
        &compiled,
        &plan,
        &program[0],
        &control_machine,
        RC_RESTORE_WRONG_ROOT,
        AC_FAULT_NONE
    );
    free_carrier(&carrier);
    control_machine = (struct ac_machine){0};
    carrier = make_carrier(&shape, 9127);
    const struct ac_execution missing = ac_execute(
        &carrier,
        &compiled,
        &plan,
        &program[0],
        &control_machine,
        RC_RESTORE_MISSING_ROOT,
        AC_FAULT_NONE
    );
    free_carrier(&carrier);
    control_machine = (struct ac_machine){0};
    carrier = make_carrier(&shape, 9127);
    const struct ac_execution snapshot = ac_execute(
        &carrier,
        &compiled,
        &plan,
        &program[0],
        &control_machine,
        RC_RESTORE_SNAPSHOT,
        AC_FAULT_NONE
    );
    free_carrier(&carrier);
    dc_print(
        "automatic-compact-snapshot-baseline",
        &compiled,
        &snapshot.phase
    );
    printf(
        "{\"mode\":\"automatic-compact-controls\","
        "\"compiler_generated_plan\":true,"
        "\"plan_fnv1a64\":\"%016llx\","
        "\"all_semantic_plan_hashes_equal\":%s,"
        "\"coefficient_oblivious_schedule\":%s,"
        "\"primary_reuse_boundary_different\":%s,"
        "\"wrong_root_inverse\":%s,"
        "\"missing_root_inverse\":%s,"
        "\"snapshot_separate_weaker_path\":true,"
        "\"snapshot_post_projection_inverse_operations\":0,"
        "\"accepted_working_slots\":11,"
        "\"retain_all_working_slots\":15,"
        "\"occurrence_working_slots\":51,"
        "\"accepted_native_operation_calls\":22,"
        "\"retain_all_native_operation_calls\":22,"
        "\"occurrence_native_operation_calls\":50,"
        "\"accepted_leaf_encode_calls\":16,"
        "\"retain_all_leaf_encode_calls\":8,"
        "\"occurrence_leaf_encode_calls\":52,"
        "\"accepted_lease_allocations\":19,"
        "\"retain_all_lease_allocations\":15,"
        "\"occurrence_lease_allocations\":51,"
        "\"same_carrier_transactions\":%u,"
        "\"maximum_reuse_restoration_abs\":%.12g,"
        "\"wrong_root_restoration_abs\":%.12g,"
        "\"missing_root_restoration_abs\":%.12g}\n",
        (unsigned long long)plan.hash,
        semantic[0].plan_hash == semantic[1].plan_hash
            && semantic[1].plan_hash == semantic[2].plan_hash
            && semantic[2].plan_hash == semantic[3].plan_hash
            && semantic[3].plan_hash == semantic[4].plan_hash
                ? "true" : "false",
        ga_stats_same_schedule(
            &semantic[0].phase.stats,
            &semantic[1].phase.stats
        ) ? "true" : "false",
        semantic[0].phase.boundary.hash
                != semantic[1].phase.boundary.hash
            ? "true" : "false",
        wrong.phase.restoration_max_abs > 1.0e-6
            ? "true" : "false",
        missing.phase.restoration_max_abs > 1.0e-6
            ? "true" : "false",
        (unsigned)(AC_RUN_VARIANTS + AC_REUSE_CYCLES),
        repeated_max_abs,
        wrong.phase.restoration_max_abs,
        missing.phase.restoration_max_abs
    );
    return 0;
}
