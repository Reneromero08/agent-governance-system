#define _POSIX_C_SOURCE 200809L

/*
 * Compiler-planned one-layer internal rematerialization for the exact
 * fifteen-node four-owner affine DAG.
 *
 * The established leaf-only implementation is embedded as a backend and
 * regression foundation.  This successor overlays a physical activation
 * ledger on the backend's one-transition-per-public-edge logical custody.
 * Four topology-selected, nonshared degree-one operators whose operands are
 * pinned shared owners are actually inversed before projection and actually
 * recomputed for inverse closure.  No relation contents are saved.
 */

#define AC_REUSE_CYCLES 0U
#define AC_PUBLIC_MAIN ir_leaf_only_embedded_main
#include "automatic_compact_general_multi_dag_affine_phase.c"
#undef AC_PUBLIC_MAIN

#ifndef IR_WORKING_SLOTS
#define IR_WORKING_SLOTS 8U
#endif
#ifndef IR_RUN_VARIANTS
#define IR_RUN_VARIANTS GM_VARIANTS
#endif
#ifndef IR_REUSE_CYCLES
#define IR_REUSE_CYCLES 12U
#endif
#ifndef IR_SCALE_ONLY
#define IR_SCALE_ONLY 0
#endif

#define IR_MAX_STEPS (2U * RC_MAX_NODES)
#define IR_MAX_ACTIVATIONS 2U
#define IR_CLAIM \
    "BOUNDED_15_NODE_FOUR_OWNER_INTERNAL_OPERATOR_REMATERIALIZATION_ESTABLISHED"

_Static_assert(IR_RUN_VARIANTS >= 2U, "internal reuse required");
_Static_assert(
    IR_RUN_VARIANTS <= GM_VARIANTS,
    "internal variants exceeded"
);

enum ir_opcode {
    IR_ENCODE_LEAF = 1,
    IR_APPLY_OPERATOR = 2,
    IR_EVICT_LEAF = 3,
    IR_EVICT_OPERATOR = 4
};

enum ir_residency {
    IR_UNMATERIALIZED = 0,
    IR_LIVE_ORIGINAL = 1,
    IR_DORMANT_LEAF = 2,
    IR_LIVE_RECONSTRUCTED_LEAF = 3,
    IR_DORMANT_OPERATOR = 4,
    IR_LIVE_REMATERIALIZED_OPERATOR = 5,
    IR_CLEARED = 6
};

enum ir_policy {
    IR_EXACT_SINGLE_ACTIVATION = 1,
    IR_EXACT_MULTI_ACTIVATION = 2,
    IR_PUBLIC_LEAF_RECONSTRUCTION = 3,
    IR_INTERNAL_OPERATOR_RECONSTRUCTION = 4
};

enum ir_activation_state {
    IR_ACTIVATION_UNUSED = 0,
    IR_ACTIVATION_FORWARD_OPEN = 1,
    IR_ACTIVATION_INVERSE_CLOSED = 2
};

enum ir_obligation_kind {
    IR_OBLIGATION_NONE = 0,
    IR_OBLIGATION_PUBLIC_LEAF = 1,
    IR_OBLIGATION_INTERNAL_OPERATOR = 2
};

enum ir_obligation_stage {
    IR_STAGE_NONE = 0,
    IR_STAGE_DORMANT = 1,
    IR_STAGE_RECONSTRUCTED = 2,
    IR_STAGE_OUTGOING_INVERSED = 3
};

enum ir_fault {
    IR_FAULT_NONE = 0,
    IR_FAULT_STALE_CONSUMER_ACTIVATION = 1,
    IR_FAULT_MISSING_ACTIVATION_CLOSE = 2,
    IR_FAULT_STALE_PRODUCER_ACTIVATION = 3,
    IR_FAULT_MISSING_OPERATOR_RECONSTRUCTION = 4,
    IR_FAULT_DOUBLE_OPERATOR_RECONSTRUCTION = 5,
    IR_FAULT_STALE_OBLIGATION_EPOCH = 6,
    IR_FAULT_EVICT_NONELIGIBLE = 7,
    IR_FAULT_TAMPER_TAPE = 8
};

struct ir_step {
    enum ir_opcode opcode;
    size_t node;
    unsigned public_node_id;
    size_t edge;
    size_t expected_live_after;
};

struct ir_plan {
    struct ir_step step[IR_MAX_STEPS];
    size_t count;
    size_t predicted_peak_live;
    size_t final_live;
    size_t reconstructible_leaves;
    size_t rematerializable_operators;
    unsigned char reconstructible_leaf[RC_MAX_NODES];
    unsigned char rematerializable_operator[RC_MAX_NODES];
    size_t outgoing_edge[RC_MAX_NODES];
    uint64_t topology_hash;
    uint64_t schedule_hash;
    uint64_t hash;
};

struct ir_activation {
    enum ir_activation_state state;
    uint64_t consumer_activation;
    uint64_t forward_producer_activation;
    uint64_t inverse_producer_activation;
    size_t forward_slot;
    uint64_t forward_serial;
    size_t inverse_slot;
    uint64_t inverse_serial;
    int lawful;
};

struct ir_edge_runtime {
    enum ir_policy policy;
    size_t activation_limit;
    size_t receipt_count;
    size_t closed_count;
    struct ir_activation receipt[IR_MAX_ACTIVATIONS];
};

struct ir_obligation {
    enum ir_obligation_kind kind;
    enum ir_obligation_stage stage;
    size_t node;
    unsigned public_node_id;
    enum rc_kind operator_kind;
    size_t body;
    struct rc_signature signature;
    unsigned left_public_id;
    unsigned right_public_id;
    size_t edge;
    uint64_t program_epoch;
    uint64_t plan_hash;
    uint64_t original_activation;
    uint64_t replacement_activation;
    size_t original_slot;
    uint64_t original_serial;
};

struct ir_owner {
    int live;
    struct rc_lease lease;
    uint64_t forward_activations;
    uint64_t inverse_activations;
    int live_at_projection;
    int live_after_root_inverse;
};

struct ir_edge_audit {
    unsigned producer_public_id;
    unsigned consumer_public_id;
    unsigned side;
    enum ir_policy policy;
    size_t activation_limit;
    size_t receipt_count;
    size_t closed_count;
    int all_activations_lawful;
};

struct ir_machine {
    uint64_t next_serial;
    uint64_t restoration_generation;
    uint64_t program_epoch;
};

struct ir_execution {
    struct ga_boundary boundary;
    struct ga_stats stats;
    struct ir_edge_audit edge[DC_MAX_EDGES];
    size_t edge_count;
    struct ir_owner owner[RC_MAX_NODES];
    uint64_t program_epoch;
    uint64_t plan_hash;
    size_t tape_steps;
    size_t tape_cursor_after_forward;
    size_t tape_cursor_after_inverse;
    size_t predicted_peak_live;
    size_t projection_live;
    size_t maximum_live;
    size_t working_slots;
    uint64_t lease_allocations;
    uint64_t lease_releases;
    uint64_t forward_operations;
    uint64_t inverse_operations;
    uint64_t forward_leaf_encodes;
    uint64_t inverse_leaf_encodes;
    uint64_t early_leaf_unmaterializations;
    uint64_t leaf_reconstructions;
    uint64_t final_leaf_inverses;
    uint64_t early_internal_unmaterializations;
    uint64_t internal_node_rematerializations;
    uint64_t operator_recomputations;
    uint64_t physical_edge_forward_activations;
    uint64_t physical_edge_inverse_activations;
    size_t multi_activation_edges;
    size_t second_activations;
    size_t exact_activation_receipts;
    size_t reconstructed_leaf_edges;
    size_t reconstructed_operator_edges;
    uint64_t shared_forward_activations;
    uint64_t shared_inverse_activations;
    int all_receipts_lawful;
    int obligations_restored;
    int residency_restored;
    int logical_custody_restored;
    int allocator_restored;
    int tape_cursor_empty;
    int workspace_cleared;
    int snapshot_loaded;
    uint64_t boundary_block_copy_calls;
    uint64_t intermediate_block_copy_calls;
    uint64_t serial_before;
    uint64_t serial_after;
    uint64_t restoration_generation_before;
    uint64_t restoration_generation_after;
    double restoration_max_abs;
    double integrity_max_abs;
};

static size_t ir_outgoing_edge(
    const struct dc_compiled *compiled,
    size_t producer
) {
    size_t result = SIZE_MAX;
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        if (compiled->edge[edge].producer != producer) {
            continue;
        }
        if (result != SIZE_MAX) {
            fail("internal rematerialization expected degree one");
        }
        result = edge;
    }
    if (result == SIZE_MAX) {
        fail("internal rematerialization outgoing edge absent");
    }
    return result;
}

static void ir_append_step(
    struct ir_plan *plan,
    enum ir_opcode opcode,
    size_t node,
    unsigned public_node_id,
    size_t edge,
    size_t *live
) {
    if (plan->count >= IR_MAX_STEPS) {
        fail("internal rematerialization tape capacity exceeded");
    }
    if (opcode == IR_ENCODE_LEAF || opcode == IR_APPLY_OPERATOR) {
        ++*live;
        if (*live > plan->predicted_peak_live) {
            plan->predicted_peak_live = *live;
        }
    } else if (
        opcode == IR_EVICT_LEAF || opcode == IR_EVICT_OPERATOR
    ) {
        if (*live == 0U) {
            fail("internal rematerialization live-count underflow");
        }
        --*live;
    } else {
        fail("internal rematerialization tape opcode invalid");
    }
    plan->step[plan->count++] = (struct ir_step){
        .opcode = opcode,
        .node = node,
        .public_node_id = public_node_id,
        .edge = edge,
        .expected_live_after = *live
    };
}

static uint64_t ir_hash_plan(const struct ir_plan *plan) {
    uint64_t hash = UINT64_C(14695981039346656037);
    rc_hash_word(&hash, plan->count);
    rc_hash_word(&hash, plan->predicted_peak_live);
    rc_hash_word(&hash, plan->final_live);
    rc_hash_word(&hash, plan->reconstructible_leaves);
    rc_hash_word(&hash, plan->rematerializable_operators);
    rc_hash_word(&hash, (size_t)plan->topology_hash);
    rc_hash_word(&hash, (size_t)plan->schedule_hash);
    for (size_t index = 0U; index < plan->count; ++index) {
        const struct ir_step *step = &plan->step[index];
        rc_hash_word(&hash, (size_t)step->opcode);
        rc_hash_word(&hash, step->node);
        rc_hash_word(&hash, (size_t)step->public_node_id);
        rc_hash_word(&hash, step->edge);
        rc_hash_word(&hash, step->expected_live_after);
    }
    return hash;
}

static struct ir_plan ir_compile_plan(
    const struct dc_compiled *compiled
) {
    struct ir_plan plan = {0};
    plan.topology_hash = compiled->graph.topology_hash;
    plan.schedule_hash = compiled->graph.schedule_hash;
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        plan.outgoing_edge[index] = SIZE_MAX;
    }
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        const struct rc_node *node = &compiled->graph.node[index];
        if (
            node->kind == RC_LEAF
            && node->indegree == 1U
            && compiled->is_shared[index] == 0U
        ) {
            plan.reconstructible_leaf[index] = 1U;
            plan.outgoing_edge[index] =
                ir_outgoing_edge(compiled, index);
            ++plan.reconstructible_leaves;
            continue;
        }
        if (
            node->kind != RC_LEAF
            && node->indegree == 1U
            && compiled->is_shared[index] == 0U
            && compiled->is_shared[node->left] != 0U
            && compiled->is_shared[node->right] != 0U
        ) {
            const size_t edge = ir_outgoing_edge(compiled, index);
            if (compiled->edge[edge].consumer == compiled->graph.root) {
                continue;
            }
            plan.rematerializable_operator[index] = 1U;
            plan.outgoing_edge[index] = edge;
            ++plan.rematerializable_operators;
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
        ir_append_step(
            &plan,
            node->kind == RC_LEAF
                ? IR_ENCODE_LEAF
                : IR_APPLY_OPERATOR,
            index,
            node->id,
            SIZE_MAX,
            &live
        );
        if (node->kind == RC_LEAF) {
            continue;
        }
        const size_t child[] = {node->right, node->left};
        for (size_t side = 0U; side < 2U; ++side) {
            const size_t producer = child[side];
            if (plan.reconstructible_leaf[producer] != 0U) {
                ir_append_step(
                    &plan,
                    IR_EVICT_LEAF,
                    producer,
                    compiled->graph.node[producer].id,
                    plan.outgoing_edge[producer],
                    &live
                );
            } else if (
                plan.rematerializable_operator[producer] != 0U
            ) {
                ir_append_step(
                    &plan,
                    IR_EVICT_OPERATOR,
                    producer,
                    compiled->graph.node[producer].id,
                    plan.outgoing_edge[producer],
                    &live
                );
            }
        }
    }
    plan.final_live = live;
    plan.hash = ir_hash_plan(&plan);
    if (
        plan.reconstructible_leaves != 4U
        || plan.rematerializable_operators != 4U
        || plan.count != 23U
        || plan.predicted_peak_live != 8U
        || plan.final_live != 7U
    ) {
        fail("internal rematerialization plan outside bounded law");
    }
    return plan;
}

static void ir_validate_plan(
    const struct ir_plan *plan,
    const struct dc_compiled *compiled
) {
    if (
        plan->topology_hash != compiled->graph.topology_hash
        || plan->schedule_hash != compiled->graph.schedule_hash
        || plan->hash != ir_hash_plan(plan)
    ) {
        fail("internal rematerialization tape hash changed");
    }
    for (size_t index = 0U; index < plan->count; ++index) {
        if (
            plan->step[index].node >= compiled->graph.count
            || plan->step[index].public_node_id
                != compiled->graph.node[plan->step[index].node].id
        ) {
            fail("internal rematerialization public identity changed");
        }
    }
}

static enum ir_policy ir_edge_policy(
    const struct dc_compiled *compiled,
    const struct ir_plan *plan,
    size_t edge
) {
    const size_t producer = compiled->edge[edge].producer;
    const size_t consumer = compiled->edge[edge].consumer;
    if (plan->rematerializable_operator[consumer] != 0U) {
        return IR_EXACT_MULTI_ACTIVATION;
    }
    if (plan->reconstructible_leaf[producer] != 0U) {
        return IR_PUBLIC_LEAF_RECONSTRUCTION;
    }
    if (plan->rematerializable_operator[producer] != 0U) {
        return IR_INTERNAL_OPERATOR_RECONSTRUCTION;
    }
    return IR_EXACT_SINGLE_ACTIVATION;
}

static void ir_initialize_edges(
    const struct dc_compiled *compiled,
    const struct ir_plan *plan,
    struct ir_edge_runtime edge_runtime[DC_MAX_EDGES]
) {
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        edge_runtime[edge].policy =
            ir_edge_policy(compiled, plan, edge);
        edge_runtime[edge].activation_limit =
            edge_runtime[edge].policy == IR_EXACT_MULTI_ACTIVATION
                ? 2U
                : 1U;
    }
}

static void ir_validate_shared_owner(
    const struct dc_compiled *compiled,
    const struct ir_owner owner[RC_MAX_NODES],
    size_t producer,
    struct rc_lease lease
) {
    if (compiled->is_shared[producer] == 0U) {
        return;
    }
    if (
        !owner[producer].live
        || owner[producer].lease.slot != lease.slot
        || owner[producer].lease.serial != lease.serial
    ) {
        fail("internal rematerialization substituted shared owner");
    }
}

static void ir_preflight_forward_activation(
    const struct dc_compiled *compiled,
    const struct ir_edge_runtime edge_runtime[DC_MAX_EDGES],
    const struct ir_owner owner[RC_MAX_NODES],
    size_t edge,
    struct rc_lease lease,
    uint64_t consumer_activation,
    uint64_t producer_activation
) {
    const struct ir_edge_runtime *runtime = &edge_runtime[edge];
    if (
        runtime->receipt_count >= runtime->activation_limit
        || runtime->receipt_count >= IR_MAX_ACTIVATIONS
        || runtime->closed_count != runtime->receipt_count
    ) {
        fail("internal rematerialization activation begin state invalid");
    }
    if (
        runtime->receipt_count > 0U
        && runtime->receipt[runtime->receipt_count - 1U].state
            != IR_ACTIVATION_INVERSE_CLOSED
    ) {
        fail("internal rematerialization prior activation still open");
    }
    const size_t producer = compiled->edge[edge].producer;
    ir_validate_shared_owner(
        compiled, owner, producer, lease
    );
    if (
        runtime->policy != IR_EXACT_MULTI_ACTIVATION
        && consumer_activation != 0U
    ) {
        fail("internal rematerialization unexpected consumer activation");
    }
    if (
        compiled->is_shared[producer] != 0U
        && producer_activation != 0U
    ) {
        fail("internal rematerialization changed shared activation");
    }
}

static void ir_commit_forward_activation(
    const struct dc_compiled *compiled,
    struct ir_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct ir_owner owner[RC_MAX_NODES],
    struct ir_execution *execution,
    size_t edge,
    struct rc_lease lease,
    uint64_t consumer_activation,
    uint64_t producer_activation
) {
    struct ir_edge_runtime *runtime = &edge_runtime[edge];
    struct ir_activation *receipt =
        &runtime->receipt[runtime->receipt_count++];
    *receipt = (struct ir_activation){
        .state = IR_ACTIVATION_FORWARD_OPEN,
        .consumer_activation = consumer_activation,
        .forward_producer_activation = producer_activation,
        .forward_slot = lease.slot,
        .forward_serial = lease.serial
    };
    ++execution->physical_edge_forward_activations;
    const size_t producer = compiled->edge[edge].producer;
    if (compiled->is_shared[producer] != 0U) {
        ++owner[producer].forward_activations;
        ++execution->shared_forward_activations;
    }
}

static void ir_preflight_inverse_activation(
    const struct dc_compiled *compiled,
    const struct ir_edge_runtime edge_runtime[DC_MAX_EDGES],
    const struct ir_owner owner[RC_MAX_NODES],
    size_t edge,
    struct rc_lease lease,
    uint64_t consumer_activation,
    uint64_t producer_activation,
    int reconstruction_bound
) {
    const struct ir_edge_runtime *runtime = &edge_runtime[edge];
    if (
        runtime->closed_count >= runtime->receipt_count
        || runtime->closed_count >= IR_MAX_ACTIVATIONS
    ) {
        fail("internal rematerialization activation close unavailable");
    }
    const struct ir_activation *receipt =
        &runtime->receipt[runtime->closed_count];
    if (
        receipt->state != IR_ACTIVATION_FORWARD_OPEN
        || receipt->consumer_activation != consumer_activation
    ) {
        fail("internal rematerialization consumer activation stale");
    }
    const size_t producer = compiled->edge[edge].producer;
    ir_validate_shared_owner(
        compiled, owner, producer, lease
    );
    if (
        runtime->policy == IR_EXACT_SINGLE_ACTIVATION
        || runtime->policy == IR_EXACT_MULTI_ACTIVATION
    ) {
        if (
            receipt->forward_producer_activation
                != producer_activation
            || receipt->forward_slot != lease.slot
            || receipt->forward_serial != lease.serial
        ) {
            fail("internal rematerialization exact activation changed");
        }
        return;
    }
    if (
        !reconstruction_bound
        || (
            runtime->policy != IR_PUBLIC_LEAF_RECONSTRUCTION
            && runtime->policy
                != IR_INTERNAL_OPERATOR_RECONSTRUCTION
        )
        || receipt->forward_producer_activation
            == producer_activation
        || receipt->forward_serial == lease.serial
    ) {
        fail("internal rematerialization reconstruction activation invalid");
    }
}

static void ir_commit_inverse_activation(
    const struct dc_compiled *compiled,
    struct ir_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct ir_owner owner[RC_MAX_NODES],
    struct ir_execution *execution,
    size_t edge,
    struct rc_lease lease,
    uint64_t producer_activation
) {
    struct ir_edge_runtime *runtime = &edge_runtime[edge];
    struct ir_activation *receipt =
        &runtime->receipt[runtime->closed_count++];
    receipt->state = IR_ACTIVATION_INVERSE_CLOSED;
    receipt->inverse_producer_activation = producer_activation;
    receipt->inverse_slot = lease.slot;
    receipt->inverse_serial = lease.serial;
    receipt->lawful =
        (
            runtime->policy == IR_EXACT_SINGLE_ACTIVATION
            || runtime->policy == IR_EXACT_MULTI_ACTIVATION
        )
            ? (
                receipt->forward_producer_activation
                    == receipt->inverse_producer_activation
                && receipt->forward_slot == receipt->inverse_slot
                && receipt->forward_serial == receipt->inverse_serial
            )
            : (
                receipt->forward_producer_activation
                    != receipt->inverse_producer_activation
                && receipt->forward_serial != receipt->inverse_serial
            );
    ++execution->physical_edge_inverse_activations;
    const size_t producer = compiled->edge[edge].producer;
    if (compiled->is_shared[producer] != 0U) {
        ++owner[producer].inverse_activations;
        ++execution->shared_inverse_activations;
    }
}

static uint64_t ir_node_activation(
    const enum ir_residency residency[RC_MAX_NODES],
    size_t node
) {
    return (
        residency[node] == IR_LIVE_RECONSTRUCTED_LEAF
        || residency[node] == IR_LIVE_REMATERIALIZED_OPERATOR
    ) ? 1U : 0U;
}

static int ir_cross_obligation_valid(
    const struct ir_plan *plan,
    const struct ir_obligation obligation[RC_MAX_NODES],
    const enum ir_residency residency[RC_MAX_NODES],
    size_t producer,
    size_t edge,
    uint64_t program_epoch
) {
    const struct ir_obligation *owed = &obligation[producer];
    const int expected_live =
        owed->kind == IR_OBLIGATION_PUBLIC_LEAF
            ? residency[producer] == IR_LIVE_RECONSTRUCTED_LEAF
            : residency[producer]
                == IR_LIVE_REMATERIALIZED_OPERATOR;
    return owed->stage == IR_STAGE_RECONSTRUCTED
        && expected_live
        && owed->node == producer
        && owed->edge == edge
        && owed->program_epoch == program_epoch
        && owed->plan_hash == plan->hash
        && owed->original_activation == 0U
        && owed->replacement_activation == 1U;
}

static void ir_record_owner_birth(
    const struct dc_compiled *compiled,
    struct dc_custody *logical,
    struct ir_owner owner[RC_MAX_NODES],
    size_t node,
    struct rc_lease lease
) {
    if (compiled->is_shared[node] == 0U) {
        return;
    }
    if (owner[node].live) {
        fail("internal rematerialization shared owner duplicated");
    }
    owner[node].live = 1;
    owner[node].lease = lease;
    ac_record_shared_birth(compiled, logical, node, lease);
}

static void ir_record_owner_release(
    const struct dc_compiled *compiled,
    struct dc_custody *logical,
    struct ir_owner owner[RC_MAX_NODES],
    size_t node
) {
    if (compiled->is_shared[node] == 0U) {
        return;
    }
    if (!owner[node].live) {
        fail("internal rematerialization shared owner release invalid");
    }
    ac_record_shared_release(compiled, logical, node);
    owner[node].live = 0;
    owner[node].lease = (struct rc_lease){0};
}

static void ir_forward_operator(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct ir_plan *plan,
    struct dc_custody *logical,
    struct ir_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct ir_owner owner[RC_MAX_NODES],
    enum ir_residency residency[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t index,
    uint64_t consumer_activation,
    int logical_forward,
    struct ir_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[index];
    const enum ir_residency expected =
        consumer_activation == 0U
            ? IR_UNMATERIALIZED
            : IR_DORMANT_OPERATOR;
    if (
        node->kind == RC_LEAF
        || residency[index] != expected
        || (
            consumer_activation == 1U
            && plan->rematerializable_operator[index] == 0U
        )
    ) {
        fail("internal rematerialization forward operator state invalid");
    }
    const size_t left_edge = compiled->left_edge[index];
    const size_t right_edge = compiled->right_edge[index];
    const struct rc_lease left = node_lease[node->left];
    const struct rc_lease right = node_lease[node->right];
    const uint64_t left_activation =
        ir_node_activation(residency, node->left);
    const uint64_t right_activation =
        ir_node_activation(residency, node->right);
    ir_preflight_forward_activation(
        compiled,
        edge_runtime,
        owner,
        left_edge,
        left,
        consumer_activation,
        left_activation
    );
    ir_preflight_forward_activation(
        compiled,
        edge_runtime,
        owner,
        right_edge,
        right,
        consumer_activation,
        right_activation
    );
    if (
        logical_forward
        && (
            !dc_prepare_forward_edge(
                compiled, logical, left_edge, left, 0, 0, SIZE_MAX
            )
            || !dc_prepare_forward_edge(
                compiled, logical, right_edge, right, 0, 0, SIZE_MAX
            )
        )
    ) {
        fail("internal rematerialization logical edge skipped");
    }
    const struct rc_lease output =
        rc_allocate(runtime, index, node->signature);
    rc_apply_node(runtime, index, left, right, output, 0, 0);
    ir_commit_forward_activation(
        compiled,
        edge_runtime,
        owner,
        execution,
        left_edge,
        left,
        consumer_activation,
        left_activation
    );
    ir_commit_forward_activation(
        compiled,
        edge_runtime,
        owner,
        execution,
        right_edge,
        right,
        consumer_activation,
        right_activation
    );
    if (logical_forward) {
        dc_commit_forward_edge(
            compiled, logical, left_edge, left, 0, SIZE_MAX
        );
        dc_commit_forward_edge(
            compiled, logical, right_edge, right, 0, SIZE_MAX
        );
    }
    node_lease[index] = output;
    residency[index] = consumer_activation == 0U
        ? IR_LIVE_ORIGINAL
        : IR_LIVE_REMATERIALIZED_OPERATOR;
    ir_record_owner_birth(
        compiled, logical, owner, index, output
    );
    if (consumer_activation == 1U) {
        ++execution->internal_node_rematerializations;
        ++execution->operator_recomputations;
    }
}

static void ir_inverse_operator(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct ir_plan *plan,
    struct dc_custody *logical,
    struct ir_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct ir_owner owner[RC_MAX_NODES],
    struct ir_obligation obligation[RC_MAX_NODES],
    enum ir_residency residency[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t index,
    int wrong,
    enum ir_fault fault,
    struct ir_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[index];
    if (
        node->kind == RC_LEAF
        || (
            residency[index] != IR_LIVE_ORIGINAL
            && residency[index]
                != IR_LIVE_REMATERIALIZED_OPERATOR
        )
    ) {
        fail("internal rematerialization inverse operator state invalid");
    }
    const uint64_t consumer_activation =
        ir_node_activation(residency, index);
    if (
        consumer_activation == 1U
        && (
            plan->rematerializable_operator[index] == 0U
            || obligation[index].stage
                != IR_STAGE_OUTGOING_INVERSED
        )
    ) {
        fail("internal rematerialization final operator obligation invalid");
    }
    dc_require_inverse_ready(compiled, logical, index);
    const size_t input_edge[] = {
        compiled->left_edge[index],
        compiled->right_edge[index]
    };
    const size_t input_node[] = {node->left, node->right};
    struct rc_lease input_lease[2] = {
        node_lease[node->left],
        node_lease[node->right]
    };
    uint64_t producer_activation[2] = {
        ir_node_activation(residency, node->left),
        ir_node_activation(residency, node->right)
    };
    for (size_t side = 0U; side < 2U; ++side) {
        const size_t producer = input_node[side];
        const int cross_bound = ir_cross_obligation_valid(
            plan,
            obligation,
            residency,
            producer,
            input_edge[side],
            execution->program_epoch
        );
        uint64_t presented_consumer = consumer_activation;
        uint64_t presented_producer = producer_activation[side];
        if (
            fault == IR_FAULT_STALE_CONSUMER_ACTIVATION
            && compiled->graph.node[index].id == 809U
            && consumer_activation == 1U
            && side == 0U
        ) {
            presented_consumer = 0U;
        }
        if (
            fault == IR_FAULT_STALE_PRODUCER_ACTIVATION
            && compiled->graph.node[index].id == 813U
            && compiled->graph.node[producer].id == 809U
        ) {
            presented_producer = 0U;
        }
        ir_preflight_inverse_activation(
            compiled,
            edge_runtime,
            owner,
            input_edge[side],
            input_lease[side],
            presented_consumer,
            presented_producer,
            cross_bound
        );
    }
    const struct rc_lease output = node_lease[index];
    rc_apply_node(
        runtime,
        index,
        input_lease[0],
        input_lease[1],
        output,
        1,
        wrong
    );
    for (size_t side = 0U; side < 2U; ++side) {
        ir_commit_inverse_activation(
            compiled,
            edge_runtime,
            owner,
            execution,
            input_edge[side],
            input_lease[side],
            producer_activation[side]
        );
        dc_inverse_edge(
            compiled, logical, input_edge[side], input_lease[side]
        );
        const size_t producer = input_node[side];
        if (
            edge_runtime[input_edge[side]].policy
                == IR_PUBLIC_LEAF_RECONSTRUCTION
            || edge_runtime[input_edge[side]].policy
                == IR_INTERNAL_OPERATOR_RECONSTRUCTION
        ) {
            obligation[producer].stage =
                IR_STAGE_OUTGOING_INVERSED;
        }
    }
    rc_release(
        runtime, output, index, node->signature, wrong
    );
    ir_record_owner_release(
        compiled, logical, owner, index
    );
    node_lease[index] = (struct rc_lease){
        .slot = SIZE_MAX,
        .serial = 0U
    };
    residency[index] = IR_CLEARED;
    if (consumer_activation == 1U) {
        obligation[index] = (struct ir_obligation){0};
    }
}

static void ir_encode_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    enum ir_residency residency[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t leaf,
    int reconstructed,
    struct ir_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    const enum ir_residency expected =
        reconstructed ? IR_DORMANT_LEAF : IR_UNMATERIALIZED;
    if (node->kind != RC_LEAF || residency[leaf] != expected) {
        fail("internal rematerialization leaf materialization invalid");
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
    node_lease[leaf] = lease;
    residency[leaf] = reconstructed
        ? IR_LIVE_RECONSTRUCTED_LEAF
        : IR_LIVE_ORIGINAL;
    if (reconstructed) {
        ++execution->leaf_reconstructions;
    }
}

static void ir_evict_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct ir_plan *plan,
    const struct dc_custody *logical,
    const struct ir_edge_runtime edge_runtime[DC_MAX_EDGES],
    enum ir_residency residency[RC_MAX_NODES],
    struct ir_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t leaf,
    uint64_t program_epoch,
    struct ir_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    const size_t edge = plan->outgoing_edge[leaf];
    if (
        plan->reconstructible_leaf[leaf] == 0U
        || node->kind != RC_LEAF
        || residency[leaf] != IR_LIVE_ORIGINAL
        || edge >= compiled->edge_count
        || logical->edge_state[edge] != DC_EDGE_FORWARD
        || edge_runtime[edge].receipt_count != 1U
        || edge_runtime[edge].closed_count != 0U
        || edge_runtime[edge].receipt[0].state
            != IR_ACTIVATION_FORWARD_OPEN
        || obligation[leaf].kind != IR_OBLIGATION_NONE
    ) {
        fail("internal rematerialization early leaf eviction denied");
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
    residency[leaf] = IR_DORMANT_LEAF;
    obligation[leaf] = (struct ir_obligation){
        .kind = IR_OBLIGATION_PUBLIC_LEAF,
        .stage = IR_STAGE_DORMANT,
        .node = leaf,
        .public_node_id = node->id,
        .operator_kind = node->kind,
        .body = node->leaf_index,
        .signature = node->signature,
        .edge = edge,
        .program_epoch = program_epoch,
        .plan_hash = plan->hash,
        .original_activation = 0U,
        .replacement_activation = 1U,
        .original_slot = lease.slot,
        .original_serial = lease.serial
    };
    ++execution->early_leaf_unmaterializations;
}

static void ir_reconstruct_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct ir_plan *plan,
    enum ir_residency residency[RC_MAX_NODES],
    struct ir_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t leaf,
    uint64_t program_epoch,
    struct ir_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    struct ir_obligation *owed = &obligation[leaf];
    if (
        owed->kind != IR_OBLIGATION_PUBLIC_LEAF
        || owed->stage != IR_STAGE_DORMANT
        || owed->node != leaf
        || owed->public_node_id != node->id
        || owed->body != node->leaf_index
        || !rc_signature_equal(owed->signature, node->signature)
        || owed->edge != plan->outgoing_edge[leaf]
        || owed->program_epoch != program_epoch
        || owed->plan_hash != plan->hash
    ) {
        fail("internal rematerialization leaf obligation invalid");
    }
    ir_encode_leaf(
        runtime,
        compiled,
        residency,
        node_lease,
        leaf,
        1,
        execution
    );
    if (node_lease[leaf].serial == owed->original_serial) {
        fail("internal rematerialization leaf generation not renewed");
    }
    owed->stage = IR_STAGE_RECONSTRUCTED;
}

static void ir_finalize_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    enum ir_residency residency[RC_MAX_NODES],
    struct ir_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t leaf,
    struct ir_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[leaf];
    if (
        obligation[leaf].kind != IR_OBLIGATION_PUBLIC_LEAF
        || obligation[leaf].stage
            != IR_STAGE_OUTGOING_INVERSED
        || residency[leaf] != IR_LIVE_RECONSTRUCTED_LEAF
    ) {
        fail("internal rematerialization final leaf inverse denied");
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
    residency[leaf] = IR_CLEARED;
    obligation[leaf] = (struct ir_obligation){0};
    ++execution->final_leaf_inverses;
}

static void ir_evict_operator(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct ir_plan *plan,
    const struct dc_custody *logical,
    struct ir_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct ir_owner owner[RC_MAX_NODES],
    enum ir_residency residency[RC_MAX_NODES],
    struct ir_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t index,
    uint64_t program_epoch,
    enum ir_fault fault,
    struct ir_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[index];
    const size_t outgoing = index < RC_MAX_NODES
        ? plan->outgoing_edge[index]
        : SIZE_MAX;
    if (
        index >= compiled->graph.count
        || plan->rematerializable_operator[index] == 0U
        || node->kind == RC_LEAF
        || compiled->is_shared[index] != 0U
        || node->indegree != 1U
        || residency[index] != IR_LIVE_ORIGINAL
        || outgoing >= compiled->edge_count
        || logical->edge_state[outgoing] != DC_EDGE_FORWARD
        || edge_runtime[outgoing].receipt_count != 1U
        || edge_runtime[outgoing].closed_count != 0U
        || edge_runtime[outgoing].receipt[0].state
            != IR_ACTIVATION_FORWARD_OPEN
        || obligation[index].kind != IR_OBLIGATION_NONE
    ) {
        fail("internal rematerialization operator eviction ineligible");
    }
    const size_t input_edge[] = {
        compiled->left_edge[index],
        compiled->right_edge[index]
    };
    const size_t input_node[] = {node->left, node->right};
    const struct rc_lease input_lease[] = {
        node_lease[node->left],
        node_lease[node->right]
    };
    for (size_t side = 0U; side < 2U; ++side) {
        ir_preflight_inverse_activation(
            compiled,
            edge_runtime,
            owner,
            input_edge[side],
            input_lease[side],
            0U,
            ir_node_activation(residency, input_node[side]),
            0
        );
    }
    const struct rc_lease output = node_lease[index];
    rc_apply_node(
        runtime,
        index,
        input_lease[0],
        input_lease[1],
        output,
        1,
        0
    );
    for (size_t side = 0U; side < 2U; ++side) {
        if (
            fault == IR_FAULT_MISSING_ACTIVATION_CLOSE
            && node->id == 809U
            && side == 0U
        ) {
            continue;
        }
        ir_commit_inverse_activation(
            compiled,
            edge_runtime,
            owner,
            execution,
            input_edge[side],
            input_lease[side],
            ir_node_activation(residency, input_node[side])
        );
    }
    rc_release(runtime, output, index, node->signature, 0);
    node_lease[index] = (struct rc_lease){
        .slot = SIZE_MAX,
        .serial = 0U
    };
    residency[index] = IR_DORMANT_OPERATOR;
    obligation[index] = (struct ir_obligation){
        .kind = IR_OBLIGATION_INTERNAL_OPERATOR,
        .stage = IR_STAGE_DORMANT,
        .node = index,
        .public_node_id = node->id,
        .operator_kind = node->kind,
        .signature = node->signature,
        .left_public_id = compiled->graph.node[node->left].id,
        .right_public_id = compiled->graph.node[node->right].id,
        .edge = outgoing,
        .program_epoch = program_epoch,
        .plan_hash = plan->hash,
        .original_activation = 0U,
        .replacement_activation = 1U,
        .original_slot = output.slot,
        .original_serial = output.serial
    };
    ++execution->early_internal_unmaterializations;
}

static void ir_reconstruct_operator(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct ir_plan *plan,
    struct dc_custody *logical,
    struct ir_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct ir_owner owner[RC_MAX_NODES],
    enum ir_residency residency[RC_MAX_NODES],
    struct ir_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t index,
    uint64_t program_epoch,
    struct ir_execution *execution
) {
    const struct rc_node *node = &compiled->graph.node[index];
    struct ir_obligation *owed = &obligation[index];
    if (
        plan->rematerializable_operator[index] == 0U
        || owed->kind != IR_OBLIGATION_INTERNAL_OPERATOR
        || owed->stage != IR_STAGE_DORMANT
        || owed->node != index
        || owed->public_node_id != node->id
        || owed->operator_kind != node->kind
        || !rc_signature_equal(owed->signature, node->signature)
        || owed->left_public_id
            != compiled->graph.node[node->left].id
        || owed->right_public_id
            != compiled->graph.node[node->right].id
        || owed->edge != plan->outgoing_edge[index]
        || owed->program_epoch != program_epoch
        || owed->plan_hash != plan->hash
        || residency[index] != IR_DORMANT_OPERATOR
    ) {
        fail("internal rematerialization operator obligation invalid");
    }
    ir_forward_operator(
        runtime,
        compiled,
        plan,
        logical,
        edge_runtime,
        owner,
        residency,
        node_lease,
        index,
        1U,
        0,
        execution
    );
    if (node_lease[index].serial == owed->original_serial) {
        fail("internal rematerialization operator generation not renewed");
    }
    owed->stage = IR_STAGE_RECONSTRUCTED;
}

static int ir_obligations_clear(
    const struct ir_obligation obligation[RC_MAX_NODES]
) {
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        if (obligation[index].kind != IR_OBLIGATION_NONE) {
            return 0;
        }
    }
    return 1;
}

static int ir_residency_clear(
    const struct dc_compiled *compiled,
    const enum ir_residency residency[RC_MAX_NODES]
) {
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        if (
            residency[index] != IR_UNMATERIALIZED
            && residency[index] != IR_CLEARED
        ) {
            return 0;
        }
    }
    return 1;
}

static int ir_receipts_complete(
    const struct dc_compiled *compiled,
    const struct ir_edge_runtime edge_runtime[DC_MAX_EDGES]
) {
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        const struct ir_edge_runtime *runtime = &edge_runtime[edge];
        if (
            runtime->receipt_count != runtime->activation_limit
            || runtime->closed_count != runtime->activation_limit
        ) {
            return 0;
        }
        for (
            size_t activation = 0U;
            activation < runtime->activation_limit;
            ++activation
        ) {
            if (
                runtime->receipt[activation].state
                    != IR_ACTIVATION_INVERSE_CLOSED
                || !runtime->receipt[activation].lawful
            ) {
                return 0;
            }
        }
    }
    return 1;
}

static void ir_capture_audits(
    const struct dc_compiled *compiled,
    const struct dc_custody *logical,
    const struct ir_edge_runtime edge_runtime[DC_MAX_EDGES],
    const struct ir_owner owner[RC_MAX_NODES],
    struct ir_execution *execution
) {
    execution->all_receipts_lawful = 1;
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        const struct dc_edge *descriptor = &compiled->edge[edge];
        const struct ir_edge_runtime *runtime = &edge_runtime[edge];
        struct ir_edge_audit *audit =
            &execution->edge[execution->edge_count++];
        audit->producer_public_id =
            compiled->graph.node[descriptor->producer].id;
        audit->consumer_public_id =
            compiled->graph.node[descriptor->consumer].id;
        audit->side = descriptor->side;
        audit->policy = runtime->policy;
        audit->activation_limit = runtime->activation_limit;
        audit->receipt_count = runtime->receipt_count;
        audit->closed_count = runtime->closed_count;
        audit->all_activations_lawful = 1;
        for (
            size_t activation = 0U;
            activation < runtime->receipt_count;
            ++activation
        ) {
            audit->all_activations_lawful &=
                runtime->receipt[activation].state
                    == IR_ACTIVATION_INVERSE_CLOSED
                && runtime->receipt[activation].lawful;
            if (
                runtime->policy == IR_EXACT_SINGLE_ACTIVATION
                || runtime->policy == IR_EXACT_MULTI_ACTIVATION
            ) {
                ++execution->exact_activation_receipts;
            }
        }
        audit->all_activations_lawful &=
            runtime->receipt_count == runtime->activation_limit
            && runtime->closed_count == runtime->activation_limit;
        execution->all_receipts_lawful &=
            audit->all_activations_lawful;
        if (runtime->policy == IR_EXACT_MULTI_ACTIVATION) {
            ++execution->multi_activation_edges;
            execution->second_activations +=
                runtime->receipt_count - 1U;
        } else if (
            runtime->policy == IR_PUBLIC_LEAF_RECONSTRUCTION
        ) {
            ++execution->reconstructed_leaf_edges;
        } else if (
            runtime->policy
                == IR_INTERNAL_OPERATOR_RECONSTRUCTION
        ) {
            ++execution->reconstructed_operator_edges;
        }
    }
    for (size_t shared = 0U; shared < compiled->fanout_nodes; ++shared) {
        const size_t node = compiled->shared_node[shared];
        execution->owner[node] = owner[node];
        const struct dc_owner_receipt *logical_owner =
            &logical->owner[node];
        execution->all_receipts_lawful &=
            logical_owner->forward_consumptions
                == compiled->graph.node[node].indegree
            && logical_owner->inverse_consumptions
                == compiled->graph.node[node].indegree
            && dc_owner_edges_exact(
                compiled, logical, node, logical_owner
            );
    }
}

static struct ir_execution ir_execute(
    struct carrier *carrier,
    const struct dc_compiled *compiled,
    const struct ir_plan *source_plan,
    const struct rc_program *program,
    struct ir_machine *machine,
    enum rc_restore_mode restore_mode,
    enum ir_fault fault
) {
    struct ir_execution result = {0};
    struct carrier borrowed = snapshot_carrier(carrier);
    struct ir_plan plan = *source_plan;
    if (fault == IR_FAULT_TAMPER_TAPE) {
        ++plan.step[0].expected_live_after;
    }
    ir_validate_plan(&plan, compiled);
    const uint64_t program_epoch = machine->program_epoch + 1U;
    struct rc_runtime runtime = {
        .carrier = carrier,
        .compiled = &compiled->graph,
        .program = program,
        .pool = IR_WORKING_SLOTS,
        .next_serial = machine->next_serial
    };
    struct dc_custody logical = {0};
    struct ir_edge_runtime edge_runtime[DC_MAX_EDGES] = {0};
    struct ir_owner owner[RC_MAX_NODES] = {0};
    enum ir_residency residency[RC_MAX_NODES] = {0};
    struct ir_obligation obligation[RC_MAX_NODES] = {0};
    struct rc_lease node_lease[RC_MAX_NODES];
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        node_lease[index] = (struct rc_lease){
            .slot = SIZE_MAX,
            .serial = 0U
        };
    }
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        logical.forward_pending[index] =
            compiled->graph.node[index].indegree;
        logical.inverse_pending[index] =
            compiled->graph.node[index].indegree;
    }
    ir_initialize_edges(compiled, &plan, edge_runtime);
    result.program_epoch = program_epoch;
    result.plan_hash = plan.hash;
    result.tape_steps = plan.count;
    result.predicted_peak_live = plan.predicted_peak_live;
    result.working_slots = IR_WORKING_SLOTS;
    result.serial_before = machine->next_serial;
    result.restoration_generation_before =
        machine->restoration_generation;

    size_t tape_cursor = 0U;
    while (tape_cursor < plan.count) {
        const struct ir_step *step = &plan.step[tape_cursor++];
        if (step->opcode == IR_ENCODE_LEAF) {
            ir_encode_leaf(
                &runtime,
                compiled,
                residency,
                node_lease,
                step->node,
                0,
                &result
            );
        } else if (step->opcode == IR_APPLY_OPERATOR) {
            ir_forward_operator(
                &runtime,
                compiled,
                &plan,
                &logical,
                edge_runtime,
                owner,
                residency,
                node_lease,
                step->node,
                0U,
                1,
                &result
            );
        } else if (step->opcode == IR_EVICT_LEAF) {
            ir_evict_leaf(
                &runtime,
                compiled,
                &plan,
                &logical,
                edge_runtime,
                residency,
                obligation,
                node_lease,
                step->node,
                program_epoch,
                &result
            );
        } else if (step->opcode == IR_EVICT_OPERATOR) {
            ir_evict_operator(
                &runtime,
                compiled,
                &plan,
                &logical,
                edge_runtime,
                owner,
                residency,
                obligation,
                node_lease,
                step->node,
                program_epoch,
                fault,
                &result
            );
        } else {
            fail("internal rematerialization forward opcode invalid");
        }
        if (runtime.live != step->expected_live_after) {
            fail("internal rematerialization tape live-count mismatch");
        }
    }
    result.tape_cursor_after_forward = tape_cursor;
    result.projection_live = runtime.live;
    if (
        !dc_edges_all(compiled, &logical, DC_EDGE_FORWARD)
        || runtime.live != plan.final_live
    ) {
        fail("internal rematerialization forward tape incomplete");
    }
    if (fault == IR_FAULT_EVICT_NONELIGIBLE) {
        ir_evict_operator(
            &runtime,
            compiled,
            &plan,
            &logical,
            edge_runtime,
            owner,
            residency,
            obligation,
            node_lease,
            gm_find_public(compiled, 813U),
            program_epoch,
            fault,
            &result
        );
    }

    logical.projection_event = ++logical.event_clock;
    for (size_t shared = 0U; shared < compiled->fanout_nodes; ++shared) {
        const size_t node = compiled->shared_node[shared];
        dc_validate_shared_live(
            &runtime, compiled, &logical, node
        );
        owner[node].live_at_projection = 1;
        logical.owner[node].live_at_projection = 1;
    }
    for (size_t left = 0U; left < compiled->fanout_nodes; ++left) {
        const size_t left_node = compiled->shared_node[left];
        for (
            size_t right = left + 1U;
            right < compiled->fanout_nodes;
            ++right
        ) {
            const size_t right_node = compiled->shared_node[right];
            if (
                owner[left_node].lease.slot
                == owner[right_node].lease.slot
            ) {
                fail("internal rematerialization shared owners alias");
            }
        }
    }
    const size_t root = compiled->graph.root;
    dc_copy_relation_block(
        &runtime,
        &logical,
        rc_slot_start(node_lease[root].slot),
        GA_BOUNDARY_START,
        0,
        1
    );
    result.boundary =
        ga_latch_boundary(carrier, &runtime.stats);

    if (restore_mode == RC_RESTORE_SNAPSHOT) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        memset(runtime.slot, 0, sizeof(runtime.slot));
        runtime.live = 0U;
        runtime.next_serial = machine->next_serial;
        memset(&logical, 0, sizeof(logical));
        memset(edge_runtime, 0, sizeof(edge_runtime));
        memset(owner, 0, sizeof(owner));
        memset(residency, 0, sizeof(residency));
        memset(obligation, 0, sizeof(obligation));
        tape_cursor = 0U;
        result.snapshot_loaded = 1;
    } else if (restore_mode != RC_RESTORE_MISSING_ROOT) {
        dc_copy_relation_block(
            &runtime,
            &logical,
            rc_slot_start(node_lease[root].slot),
            GA_BOUNDARY_START,
            1,
            1
        );
        int first_operator_reconstruction = 1;
        while (tape_cursor > 0U) {
            const struct ir_step *step =
                &plan.step[--tape_cursor];
            if (step->opcode == IR_EVICT_LEAF) {
                ir_reconstruct_leaf(
                    &runtime,
                    compiled,
                    &plan,
                    residency,
                    obligation,
                    node_lease,
                    step->node,
                    program_epoch,
                    &result
                );
            } else if (step->opcode == IR_EVICT_OPERATOR) {
                if (
                    first_operator_reconstruction
                    && fault == IR_FAULT_STALE_OBLIGATION_EPOCH
                ) {
                    ++obligation[step->node].program_epoch;
                }
                if (
                    !(
                        first_operator_reconstruction
                        && fault
                            == IR_FAULT_MISSING_OPERATOR_RECONSTRUCTION
                    )
                ) {
                    ir_reconstruct_operator(
                        &runtime,
                        compiled,
                        &plan,
                        &logical,
                        edge_runtime,
                        owner,
                        residency,
                        obligation,
                        node_lease,
                        step->node,
                        program_epoch,
                        &result
                    );
                }
                if (
                    first_operator_reconstruction
                    && fault
                        == IR_FAULT_DOUBLE_OPERATOR_RECONSTRUCTION
                ) {
                    ir_reconstruct_operator(
                        &runtime,
                        compiled,
                        &plan,
                        &logical,
                        edge_runtime,
                        owner,
                        residency,
                        obligation,
                        node_lease,
                        step->node,
                        program_epoch,
                        &result
                    );
                }
                first_operator_reconstruction = 0;
            } else if (step->opcode == IR_APPLY_OPERATOR) {
                ir_inverse_operator(
                    &runtime,
                    compiled,
                    &plan,
                    &logical,
                    edge_runtime,
                    owner,
                    obligation,
                    residency,
                    node_lease,
                    step->node,
                    restore_mode == RC_RESTORE_WRONG_ROOT
                        && step->node == root,
                    fault,
                    &result
                );
                if (step->node == root) {
                    logical.root_inverse_event =
                        ++logical.event_clock;
                    for (
                        size_t shared = 0U;
                        shared < compiled->fanout_nodes;
                        ++shared
                    ) {
                        const size_t node =
                            compiled->shared_node[shared];
                        dc_validate_shared_live(
                            &runtime, compiled, &logical, node
                        );
                        owner[node].live_after_root_inverse = 1;
                        logical.owner[node]
                            .live_after_root_inverse = 1;
                    }
                    if (restore_mode == RC_RESTORE_WRONG_ROOT) {
                        break;
                    }
                }
            } else if (step->opcode == IR_ENCODE_LEAF) {
                ir_finalize_leaf(
                    &runtime,
                    compiled,
                    residency,
                    obligation,
                    node_lease,
                    step->node,
                    &result
                );
            } else {
                fail("internal rematerialization inverse opcode invalid");
            }
        }
    }

    const int inverse_complete =
        restore_mode == RC_RESTORE_SNAPSHOT
        || (
            tape_cursor == 0U
            && dc_edges_all(compiled, &logical, DC_EDGE_INVERSE)
            && dc_counts_zero(compiled, &logical)
            && ir_receipts_complete(compiled, edge_runtime)
            && ir_obligations_clear(obligation)
            && ir_residency_clear(compiled, residency)
        );
    if (fault != IR_FAULT_NONE) {
        free_carrier(&borrowed);
        fail("internal rematerialization injected fault escaped");
    }
    if (
        restore_mode == RC_RESTORE_CORRECT
        && !inverse_complete
    ) {
        free_carrier(&borrowed);
        fail("internal rematerialization inverse incomplete");
    }
    if (restore_mode != RC_RESTORE_SNAPSHOT) {
        ir_capture_audits(
            compiled, &logical, edge_runtime, owner, &result
        );
    }
    result.logical_custody_restored = inverse_complete;
    result.obligations_restored =
        restore_mode == RC_RESTORE_SNAPSHOT
        || ir_obligations_clear(obligation);
    result.residency_restored =
        restore_mode == RC_RESTORE_SNAPSHOT
        || ir_residency_clear(compiled, residency);
    result.allocator_restored =
        rc_allocator_is_restored(&runtime);
    result.tape_cursor_empty = tape_cursor == 0U;
    result.boundary_block_copy_calls =
        logical.boundary_block_copy_calls;
    result.intermediate_block_copy_calls =
        logical.intermediate_block_copy_calls;
    if (inverse_complete) {
        memset(&logical, 0, sizeof(logical));
        memset(edge_runtime, 0, sizeof(edge_runtime));
        memset(owner, 0, sizeof(owner));
        memset(obligation, 0, sizeof(obligation));
        memset(residency, 0, sizeof(residency));
    }
    result.restoration_max_abs = restoration(carrier, &borrowed);
    runtime.stats.restoration_cell_checks += carrier->cells;
    result.integrity_max_abs = integrity(carrier);
    runtime.stats.integrity_cell_checks += carrier->cells;
    result.workspace_cleared =
        ga_workspace_is_zero(carrier, &runtime.stats);
    result.stats = runtime.stats;
    result.maximum_live = runtime.maximum_live;
    result.lease_allocations = runtime.lease_allocations;
    result.lease_releases = runtime.lease_releases;
    result.forward_operations = runtime.forward_operations;
    result.inverse_operations = runtime.inverse_operations;
    result.forward_leaf_encodes = runtime.forward_leaf_encodes;
    result.inverse_leaf_encodes = runtime.inverse_leaf_encodes;
    result.tape_cursor_after_inverse = tape_cursor;
    if (restore_mode == RC_RESTORE_CORRECT) {
        machine->next_serial = runtime.next_serial;
        ++machine->restoration_generation;
        machine->program_epoch = program_epoch;
    }
    result.serial_after = machine->next_serial;
    result.restoration_generation_after =
        machine->restoration_generation;
    free_carrier(&borrowed);
    return result;
}

static int ir_execution_exact(
    const struct dc_compiled *compiled,
    const struct ir_execution *execution
) {
    const size_t s = gm_find_public(compiled, 805U);
    const size_t t = gm_find_public(compiled, 806U);
    const size_t u = gm_find_public(compiled, 807U);
    const size_t v = gm_find_public(compiled, 808U);
    return execution->working_slots == 8U
        && execution->maximum_live == 8U
        && execution->projection_live == 7U
        && execution->tape_steps == 23U
        && execution->tape_cursor_after_forward == 23U
        && execution->tape_cursor_after_inverse == 0U
        && execution->lease_allocations == 23U
        && execution->lease_releases == 23U
        && execution->forward_operations == 15U
        && execution->inverse_operations == 15U
        && execution->forward_leaf_encodes == 8U
        && execution->inverse_leaf_encodes == 8U
        && execution->early_leaf_unmaterializations == 4U
        && execution->leaf_reconstructions == 4U
        && execution->final_leaf_inverses == 4U
        && execution->early_internal_unmaterializations == 4U
        && execution->internal_node_rematerializations == 4U
        && execution->operator_recomputations == 4U
        && execution->physical_edge_forward_activations == 30U
        && execution->physical_edge_inverse_activations == 30U
        && execution->multi_activation_edges == 8U
        && execution->second_activations == 8U
        && execution->exact_activation_receipts == 22U
        && execution->reconstructed_leaf_edges == 4U
        && execution->reconstructed_operator_edges == 4U
        && execution->shared_forward_activations == 20U
        && execution->shared_inverse_activations == 20U
        && execution->owner[s].forward_activations == 6U
        && execution->owner[s].inverse_activations == 6U
        && execution->owner[t].forward_activations == 4U
        && execution->owner[t].inverse_activations == 4U
        && execution->owner[u].forward_activations == 6U
        && execution->owner[u].inverse_activations == 6U
        && execution->owner[v].forward_activations == 4U
        && execution->owner[v].inverse_activations == 4U
        && execution->owner[s].live_at_projection
        && execution->owner[t].live_at_projection
        && execution->owner[u].live_at_projection
        && execution->owner[v].live_at_projection
        && execution->owner[s].live_after_root_inverse
        && execution->owner[t].live_after_root_inverse
        && execution->owner[u].live_after_root_inverse
        && execution->owner[v].live_after_root_inverse
        && execution->all_receipts_lawful
        && execution->obligations_restored
        && execution->residency_restored
        && execution->logical_custody_restored
        && execution->allocator_restored
        && execution->tape_cursor_empty
        && execution->workspace_cleared
        && !execution->snapshot_loaded
        && execution->boundary_block_copy_calls == 2U
        && execution->intermediate_block_copy_calls == 0U
        && execution->serial_after
            == execution->serial_before + 23U
        && execution->restoration_generation_after
            == execution->restoration_generation_before + 1U
        && execution->restoration_max_abs
            <= GA_WORKSPACE_TOLERANCE;
}

static const char *ir_policy_name(enum ir_policy policy) {
    if (policy == IR_EXACT_SINGLE_ACTIVATION) {
        return "EXACT_SINGLE_ACTIVATION";
    }
    if (policy == IR_EXACT_MULTI_ACTIVATION) {
        return "EXACT_MULTI_ACTIVATION";
    }
    if (policy == IR_PUBLIC_LEAF_RECONSTRUCTION) {
        return "PUBLIC_LEAF_RECONSTRUCTION";
    }
    if (policy == IR_INTERNAL_OPERATOR_RECONSTRUCTION) {
        return "INTERNAL_OPERATOR_RECONSTRUCTION";
    }
    fail("internal rematerialization policy invalid");
    return "INVALID";
}

static void ir_print_edge_audits(
    const struct ir_execution *execution
) {
    printf(",\"activation_edge_receipts\":[");
    for (size_t edge = 0U; edge < execution->edge_count; ++edge) {
        const struct ir_edge_audit *audit = &execution->edge[edge];
        if (edge > 0U) {
            putchar(',');
        }
        printf(
            "{\"producer_public_id\":%u,"
            "\"consumer_public_id\":%u,"
            "\"side\":%u,"
            "\"policy\":\"%s\","
            "\"activation_limit\":%zu,"
            "\"receipt_count\":%zu,"
            "\"closed_count\":%zu,"
            "\"all_activations_lawful\":%s}",
            audit->producer_public_id,
            audit->consumer_public_id,
            audit->side,
            ir_policy_name(audit->policy),
            audit->activation_limit,
            audit->receipt_count,
            audit->closed_count,
            audit->all_activations_lawful ? "true" : "false"
        );
    }
    putchar(']');
}

static void ir_print(
    const char *mode,
    const struct dc_compiled *compiled,
    const struct ir_plan *plan,
    const struct ir_execution *execution
) {
    const size_t carrier_cells =
        rc_carrier_cells(execution->working_slots);
    const size_t workspace_cells =
        GA_CARRIER_CELLS - GA_RELATION_BLOCKS * GA_BLOCK_CELLS;
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"" IR_CLAIM "\","
        "\"claim_ceiling\":\"BOUNDED_WIDTH16_SOFTWARE_GF2_"
        "EXACT_15_NODE_ONE_LAYER_FOUR_INTERNAL_OPERATOR_"
        "REMATERIALIZATION_23_STEP_REVERSIBLE_TAPE_8_WORKING_"
        "SLOTS_9_PHYSICAL_BLOCKS_MULTI_ACTIVATION_EDGE_CUSTODY_"
        "REFERENCE_ONLY\","
        "\"width\":%u,"
        "\"manifest_nodes\":%zu,"
        "\"manifest_leaves\":%zu,"
        "\"compose_nodes\":%zu,"
        "\"intersect_nodes\":%zu,"
        "\"dag_edges\":%zu,"
        "\"dag_depth\":%zu,"
        "\"fanout_nodes\":%zu,"
        "\"maximum_fanout\":%zu,"
        "\"compiler_generated_reversible_tape\":true,"
        "\"coefficient_oblivious_plan\":true,"
        "\"plan_inputs\":\"PUBLIC_TOPOLOGY_SIGNATURES_AND_RECIPES\","
        "\"bound_topology_fnv1a64\":\"%016llx\","
        "\"bound_schedule_fnv1a64\":\"%016llx\","
        "\"plan_fnv1a64\":\"%016llx\","
        "\"plan_steps\":%zu,"
        "\"tape_cursor_after_forward\":%zu,"
        "\"tape_cursor_after_inverse\":%zu,"
        "\"predicted_peak_live\":%zu,"
        "\"runtime_peak_live\":%zu,"
        "\"projection_live\":%zu,"
        "\"working_relation_slots\":%zu,"
        "\"physical_relation_blocks_including_boundary\":%zu,"
        "\"relation_cells\":%u,"
        "\"workspace_cells\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"live_carrier_bytes\":%zu,"
        "\"carrier_and_verification_snapshot_heap_bytes\":%zu,"
        "\"compiled_topology_bytes_current_abi\":%zu,"
        "\"program_storage_bytes_current_abi\":%zu,"
        "\"logical_custody_state_bytes_current_abi\":%zu,"
        "\"activation_receipt_state_bytes_current_abi\":%zu,"
        "\"reversible_plan_storage_bytes_current_abi\":%zu,"
        "\"obligation_storage_bytes_current_abi\":%zu,"
        "\"residency_storage_bytes_current_abi\":%zu,"
        "\"node_lease_storage_bytes_current_abi\":%zu,"
        "\"execution_summary_bytes_current_abi\":%zu,"
        "\"machine_state_bytes_current_abi\":%zu,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"serialized_intermediate_coefficients\":0,"
        "\"intermediate_relation_hashes\":0,"
        "\"pre_inverse_carrier_aggregate_outputs\":0,"
        "\"host_selected_pivots\":0,"
        "\"tuple_slots\":0,"
        "\"assignment_slots\":0,"
        "\"witness_slots\":0,"
        "\"content_dependent_receipts\":0,"
        "\"boundary_coefficients\":",
        mode,
        (unsigned)GA_WIDTH,
        compiled->graph.count,
        compiled->graph.leaves,
        compiled->graph.compose_nodes,
        compiled->graph.intersect_nodes,
        compiled->edge_count,
        compiled->graph.depth,
        compiled->fanout_nodes,
        compiled->maximum_fanout,
        (unsigned long long)plan->topology_hash,
        (unsigned long long)plan->schedule_hash,
        (unsigned long long)execution->plan_hash,
        execution->tape_steps,
        execution->tape_cursor_after_forward,
        execution->tape_cursor_after_inverse,
        execution->predicted_peak_live,
        execution->maximum_live,
        execution->projection_live,
        execution->working_slots,
        rc_physical_relation_blocks(execution->working_slots),
        (unsigned)GA_BLOCK_CELLS,
        workspace_cells,
        carrier_cells,
        2U * carrier_cells * sizeof(double complex),
        4U * carrier_cells * sizeof(double complex),
        sizeof(*compiled),
        (size_t)GM_VARIANTS * sizeof(struct rc_program),
        sizeof(struct dc_custody),
        sizeof(struct ir_edge_runtime) * (size_t)DC_MAX_EDGES,
        sizeof(struct ir_plan),
        sizeof(struct ir_obligation) * (size_t)RC_MAX_NODES,
        sizeof(enum ir_residency) * (size_t)RC_MAX_NODES,
        sizeof(struct rc_lease) * (size_t)RC_MAX_NODES,
        sizeof(struct ir_execution),
        sizeof(struct ir_machine)
    );
    ga_print_boundary(&execution->boundary);
    printf(
        ",\"boundary_fnv1a64\":\"%016llx\","
        "\"maximum_root_error\":%.12g,"
        "\"restoration_tolerance\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_max_abs\":%.12g,"
        "\"workspace_cleared\":%s,"
        "\"allocator_restored\":%s,"
        "\"logical_custody_restored\":%s,"
        "\"obligations_restored\":%s,"
        "\"residency_restored\":%s,"
        "\"tape_cursor_empty\":%s,"
        "\"snapshot_loaded\":%s,"
        "\"lease_allocations\":%llu,"
        "\"lease_releases\":%llu,"
        "\"forward_native_operations\":%llu,"
        "\"inverse_native_operations\":%llu,"
        "\"native_operation_calls\":%llu,"
        "\"forward_leaf_encodes\":%llu,"
        "\"inverse_leaf_encodes\":%llu,"
        "\"leaf_encode_calls\":%llu,"
        "\"early_leaf_unmaterializations\":%llu,"
        "\"leaf_reconstructions\":%llu,"
        "\"final_leaf_inverses\":%llu,"
        "\"early_internal_unmaterializations\":%llu,"
        "\"internal_node_rematerializations\":%llu,"
        "\"operator_recomputations\":%llu,"
        "\"physical_edge_forward_activations\":%llu,"
        "\"physical_edge_inverse_activations\":%llu,"
        "\"multi_activation_edges\":%zu,"
        "\"second_activations\":%zu,"
        "\"exact_activation_receipts\":%zu,"
        "\"reconstructed_leaf_edges\":%zu,"
        "\"reconstructed_operator_edges\":%zu,"
        "\"shared_forward_activations\":%llu,"
        "\"shared_inverse_activations\":%llu,"
        "\"all_receipts_lawful\":%s,"
        "\"boundary_block_copy_calls\":%llu,"
        "\"intermediate_block_copy_calls\":%llu,"
        "\"serial_before\":%llu,"
        "\"serial_after\":%llu,"
        "\"restoration_generation_before\":%llu,"
        "\"restoration_generation_after\":%llu,"
        "\"phase_ands\":%llu,"
        "\"phase_xors\":%llu,"
        "\"phase_cell_updates\":%llu",
        (unsigned long long)execution->boundary.hash,
        execution->boundary.maximum_root_error,
        GA_WORKSPACE_TOLERANCE,
        execution->restoration_max_abs,
        execution->integrity_max_abs,
        execution->workspace_cleared ? "true" : "false",
        execution->allocator_restored ? "true" : "false",
        execution->logical_custody_restored ? "true" : "false",
        execution->obligations_restored ? "true" : "false",
        execution->residency_restored ? "true" : "false",
        execution->tape_cursor_empty ? "true" : "false",
        execution->snapshot_loaded ? "true" : "false",
        (unsigned long long)execution->lease_allocations,
        (unsigned long long)execution->lease_releases,
        (unsigned long long)execution->forward_operations,
        (unsigned long long)execution->inverse_operations,
        (unsigned long long)(
            execution->forward_operations
            + execution->inverse_operations
        ),
        (unsigned long long)execution->forward_leaf_encodes,
        (unsigned long long)execution->inverse_leaf_encodes,
        (unsigned long long)(
            execution->forward_leaf_encodes
            + execution->inverse_leaf_encodes
        ),
        (unsigned long long)
            execution->early_leaf_unmaterializations,
        (unsigned long long)execution->leaf_reconstructions,
        (unsigned long long)execution->final_leaf_inverses,
        (unsigned long long)
            execution->early_internal_unmaterializations,
        (unsigned long long)
            execution->internal_node_rematerializations,
        (unsigned long long)execution->operator_recomputations,
        (unsigned long long)
            execution->physical_edge_forward_activations,
        (unsigned long long)
            execution->physical_edge_inverse_activations,
        execution->multi_activation_edges,
        execution->second_activations,
        execution->exact_activation_receipts,
        execution->reconstructed_leaf_edges,
        execution->reconstructed_operator_edges,
        (unsigned long long)execution->shared_forward_activations,
        (unsigned long long)execution->shared_inverse_activations,
        execution->all_receipts_lawful ? "true" : "false",
        (unsigned long long)execution->boundary_block_copy_calls,
        (unsigned long long)execution->intermediate_block_copy_calls,
        (unsigned long long)execution->serial_before,
        (unsigned long long)execution->serial_after,
        (unsigned long long)
            execution->restoration_generation_before,
        (unsigned long long)
            execution->restoration_generation_after,
        (unsigned long long)execution->stats.phase_ands,
        (unsigned long long)execution->stats.phase_xors,
        (unsigned long long)execution->stats.phase_cell_updates
    );
    printf(
        ",\"shared_owner_activation_receipts\":["
    );
    for (size_t shared = 0U; shared < compiled->fanout_nodes; ++shared) {
        const size_t node = compiled->shared_node[shared];
        if (shared > 0U) {
            putchar(',');
        }
        printf(
            "{\"public_node_id\":%u,"
            "\"declared_consumers\":%zu,"
            "\"forward_activations\":%llu,"
            "\"inverse_activations\":%llu,"
            "\"one_original_generation\":true,"
            "\"live_at_projection\":%s,"
            "\"live_after_root_inverse\":%s}",
            compiled->graph.node[node].id,
            compiled->graph.node[node].indegree,
            (unsigned long long)
                execution->owner[node].forward_activations,
            (unsigned long long)
                execution->owner[node].inverse_activations,
            execution->owner[node].live_at_projection
                ? "true" : "false",
            execution->owner[node].live_after_root_inverse
                ? "true" : "false"
        );
    }
    putchar(']');
    ir_print_edge_audits(execution);
    printf("}\n");
}

static enum ir_fault ir_fault_from_option(const char *option) {
    if (
        strcmp(option, "--control-stale-consumer-activation") == 0
    ) {
        return IR_FAULT_STALE_CONSUMER_ACTIVATION;
    }
    if (
        strcmp(option, "--control-missing-activation-close") == 0
    ) {
        return IR_FAULT_MISSING_ACTIVATION_CLOSE;
    }
    if (
        strcmp(option, "--control-stale-producer-activation") == 0
    ) {
        return IR_FAULT_STALE_PRODUCER_ACTIVATION;
    }
    if (
        strcmp(option, "--control-missing-operator-reconstruction")
            == 0
    ) {
        return IR_FAULT_MISSING_OPERATOR_RECONSTRUCTION;
    }
    if (
        strcmp(option, "--control-double-operator-reconstruction")
            == 0
    ) {
        return IR_FAULT_DOUBLE_OPERATOR_RECONSTRUCTION;
    }
    if (
        strcmp(option, "--control-stale-obligation-epoch") == 0
    ) {
        return IR_FAULT_STALE_OBLIGATION_EPOCH;
    }
    if (strcmp(option, "--control-evict-noneligible") == 0) {
        return IR_FAULT_EVICT_NONELIGIBLE;
    }
    if (strcmp(option, "--control-tamper-tape") == 0) {
        return IR_FAULT_TAMPER_TAPE;
    }
    return IR_FAULT_NONE;
}

#ifndef IR_PUBLIC_MAIN
#define IR_PUBLIC_MAIN main
#endif

int IR_PUBLIC_MAIN(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fail(
            "usage: internal_rematerializing_general_multi_dag_"
            "affine_phase MANIFEST [CONTROL]"
        );
    }
    if (
        argc == 3
        && (
            strcmp(argv[2], "--project-intermediate") == 0
            || strcmp(argv[2], "--project-tape") == 0
            || strcmp(argv[2], "--project-obligations") == 0
            || strcmp(argv[2], "--project-residency") == 0
            || strcmp(argv[2], "--project-activation-receipts")
                == 0
        )
    ) {
        fail("internal rematerialization projection denied");
    }
    if (argc == 3 && strcmp(argv[2], "--null-carrier") == 0) {
        fail("null internal rematerialization carrier");
    }

    struct rc_manifest manifest = {0};
    rc_load_manifest(argv[1], &manifest);
    const struct dc_compiled compiled = dc_compile(&manifest, 4U);
    gm_require_topology(&compiled);
    const struct ir_plan plan = ir_compile_plan(&compiled);
    struct rc_program program[GM_VARIANTS];
    for (size_t variant = 0U; variant < GM_VARIANTS; ++variant) {
        program[variant] = gm_make_program(variant);
    }
    const struct process shape = {
        .carrier_cells = rc_carrier_cells(IR_WORKING_SLOTS)
    };
    if (argc == 3) {
        const enum ir_fault fault = ir_fault_from_option(argv[2]);
        if (fault == IR_FAULT_NONE) {
            fail("internal rematerialization control unknown");
        }
        struct carrier control = make_carrier(&shape, 9127);
        struct ir_machine control_machine = {0};
        (void)ir_execute(
            &control,
            &compiled,
            &plan,
            &program[0],
            &control_machine,
            RC_RESTORE_CORRECT,
            fault
        );
        free_carrier(&control);
        fail("internal rematerialization fault control escaped");
    }

    struct carrier carrier = make_carrier(&shape, 9127);
    struct ir_machine machine = {0};
    struct ir_execution semantic[IR_RUN_VARIANTS];
    double repeated_max_abs = 0.0;
    for (size_t variant = 0U; variant < IR_RUN_VARIANTS; ++variant) {
        semantic[variant] = ir_execute(
            &carrier,
            &compiled,
            &plan,
            &program[variant],
            &machine,
            RC_RESTORE_CORRECT,
            IR_FAULT_NONE
        );
        if (!ir_execution_exact(&compiled, &semantic[variant])) {
            free_carrier(&carrier);
            fail("internal rematerialization accepted law failed");
        }
        if (
            semantic[variant].restoration_max_abs
            > repeated_max_abs
        ) {
            repeated_max_abs =
                semantic[variant].restoration_max_abs;
        }
    }
#if IR_REUSE_CYCLES > 0U
    for (size_t cycle = 0U; cycle < IR_REUSE_CYCLES; ++cycle) {
        const struct ir_execution repeated = ir_execute(
            &carrier,
            &compiled,
            &plan,
            &program[cycle % 2U],
            &machine,
            RC_RESTORE_CORRECT,
            IR_FAULT_NONE
        );
        if (!ir_execution_exact(&compiled, &repeated)) {
            free_carrier(&carrier);
            fail("internal rematerialization reuse law failed");
        }
        if (repeated.restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = repeated.restoration_max_abs;
        }
    }
#endif
    free_carrier(&carrier);
    for (size_t variant = 0U; variant < IR_RUN_VARIANTS; ++variant) {
        const char *mode = variant == 0U
            ? "internal-remat-primary"
            : variant == 1U
                ? "internal-remat-reuse"
                : "internal-remat-semantic";
        ir_print(mode, &compiled, &plan, &semantic[variant]);
    }
    if (IR_SCALE_ONLY) {
        return 0;
    }

    struct ir_machine control_machine = {0};
    carrier = make_carrier(&shape, 9127);
    const struct ir_execution wrong = ir_execute(
        &carrier,
        &compiled,
        &plan,
        &program[0],
        &control_machine,
        RC_RESTORE_WRONG_ROOT,
        IR_FAULT_NONE
    );
    free_carrier(&carrier);
    control_machine = (struct ir_machine){0};
    carrier = make_carrier(&shape, 9127);
    const struct ir_execution missing = ir_execute(
        &carrier,
        &compiled,
        &plan,
        &program[0],
        &control_machine,
        RC_RESTORE_MISSING_ROOT,
        IR_FAULT_NONE
    );
    free_carrier(&carrier);
    control_machine = (struct ir_machine){0};
    carrier = make_carrier(&shape, 9127);
    const struct ir_execution snapshot = ir_execute(
        &carrier,
        &compiled,
        &plan,
        &program[0],
        &control_machine,
        RC_RESTORE_SNAPSHOT,
        IR_FAULT_NONE
    );
    free_carrier(&carrier);
    ir_print(
        "internal-remat-snapshot-baseline",
        &compiled,
        &plan,
        &snapshot
    );
    printf(
        "{\"mode\":\"internal-remat-controls\","
        "\"plan_fnv1a64\":\"%016llx\","
        "\"primary_reuse_boundary_different\":%s,"
        "\"wrong_root_inverse_detected\":%s,"
        "\"missing_root_inverse_detected\":%s,"
        "\"snapshot_separate_weaker_path\":true,"
        "\"snapshot_post_projection_inverse_operations\":0,"
        "\"accepted_working_slots\":8,"
        "\"leaf_only_working_slots\":11,"
        "\"retain_all_working_slots\":15,"
        "\"occurrence_working_slots\":51,"
        "\"accepted_native_operation_calls\":30,"
        "\"accepted_leaf_encode_calls\":16,"
        "\"accepted_lease_allocations\":23,"
        "\"same_carrier_transactions\":%u,"
        "\"maximum_reuse_restoration_abs\":%.12g,"
        "\"wrong_root_restoration_abs\":%.12g,"
        "\"missing_root_restoration_abs\":%.12g}\n",
        (unsigned long long)plan.hash,
        !ga_boundary_equal(
            &semantic[0].boundary, &semantic[1].boundary
        ) ? "true" : "false",
        wrong.restoration_max_abs > 1e-6 ? "true" : "false",
        missing.restoration_max_abs > 1e-6 ? "true" : "false",
        (unsigned)(IR_RUN_VARIANTS + IR_REUSE_CYCLES),
        repeated_max_abs,
        wrong.restoration_max_abs,
        missing.restoration_max_abs
    );
    return 0;
}
