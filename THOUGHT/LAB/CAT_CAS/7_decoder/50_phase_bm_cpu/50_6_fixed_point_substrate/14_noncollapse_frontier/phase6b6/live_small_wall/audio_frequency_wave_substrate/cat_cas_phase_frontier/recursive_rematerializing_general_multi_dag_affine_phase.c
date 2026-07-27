#define _POSIX_C_SOURCE 200809L

/*
 * Compiler-expanded rank-two rematerialization for the exact public
 * fifteen-node affine DAG.
 *
 * This successor embeds the established native phase backend and one-layer
 * executor, but owns a separate action compiler and activation ledger.  The
 * compiler selects public node 813 as the unique lowest-ID maximum-rank
 * borrower.  Its actual inverse requires reconstructed 809 and 810; literal
 * reversal reconstructs those operands again before reconstructing 813.
 * Relation contents are never stored in plans, receipts, or obligations.
 */

#define IR_REUSE_CYCLES 0U
#define IR_PUBLIC_MAIN rr_predecessor_embedded_main
#include "internal_rematerializing_general_multi_dag_affine_phase.c"
#undef IR_PUBLIC_MAIN

#ifndef RR_WORKING_SLOTS
#define RR_WORKING_SLOTS 9U
#endif
#ifndef RR_RUN_VARIANTS
#define RR_RUN_VARIANTS GM_VARIANTS
#endif
#ifndef RR_REUSE_CYCLES
#define RR_REUSE_CYCLES 12U
#endif
#ifndef RR_SCALE_ONLY
#define RR_SCALE_ONLY 0
#endif

#define RR_MAX_ACTIONS 32U
#define RR_MAX_EDGE_ACTIVATIONS 4U
#define RR_MAX_NODE_ACTIVATIONS 4U
#define RR_NO_GENERATION UINT64_MAX
#define RR_CLAIM \
    "BOUNDED_15_NODE_RANK2_RECURSIVE_OPERATOR_REMATERIALIZATION_ESTABLISHED"
#define RR_CEILING \
    "BOUNDED_WIDTH16_SOFTWARE_GF2_EXACT_15_NODE_SINGLE_RANK2_" \
    "BORROWER_28_FORWARD_28_REVERSE_ACTIONS_9_WORKING_SLOTS_" \
    "10_PHYSICAL_BLOCKS_MAX_ACTIVATION_ORDINAL3_REFERENCE_ONLY"

_Static_assert(RR_RUN_VARIANTS >= 2U, "recursive reuse required");
_Static_assert(
    RR_RUN_VARIANTS <= GM_VARIANTS,
    "recursive variants exceeded"
);

enum rr_opcode {
    RR_LEAF_FORWARD = 1,
    RR_LEAF_INVERSE = 2,
    RR_OPERATOR_FORWARD = 3,
    RR_OPERATOR_INVERSE = 4
};

enum rr_purpose {
    RR_ORIGINAL = 1,
    RR_PUBLIC_LEAF_EVICTION = 2,
    RR_OPERATOR_EVICTION = 3,
    RR_OPERATOR_RECONSTRUCTION = 4,
    RR_OPERATOR_SUSPENSION = 5,
    RR_FINAL_RESTORATION = 6
};

enum rr_receipt_policy {
    RR_RECEIPT_EXACT = 1,
    RR_RECEIPT_LEAF_REBIND = 2,
    RR_RECEIPT_OPERATOR_REBIND = 3
};

enum rr_receipt_state {
    RR_RECEIPT_UNUSED = 0,
    RR_RECEIPT_FORWARD_OPEN = 1,
    RR_RECEIPT_INVERSE_CLOSED = 2
};

enum rr_fault {
    RR_FAULT_NONE = 0,
    RR_FAULT_DEEP_STALE_CONSUMER = 1,
    RR_FAULT_DEEP_STALE_PRODUCER = 2,
    RR_FAULT_REBIND_SWAP = 3,
    RR_FAULT_ACTIVATION_REUSE = 4,
    RR_FAULT_MISSING_NESTED_CLOSE = 5,
    RR_FAULT_MISSING_NESTED_CHILD = 6,
    RR_FAULT_TAMPER_TAPE = 7
};

struct rr_action {
    enum rr_opcode opcode;
    enum rr_purpose purpose;
    size_t node;
    unsigned public_node_id;
    uint64_t generation;
    int logical_transition;
    size_t expected_live_after;
};

struct rr_receipt_plan {
    enum rr_receipt_policy policy;
    size_t edge;
    size_t ordinal;
    size_t producer;
    size_t consumer;
    uint64_t consumer_generation;
    uint64_t forward_producer_generation;
    uint64_t inverse_producer_generation;
    size_t forward_action;
    size_t inverse_action;
};

struct rr_edge_plan {
    size_t activation_limit;
    struct rr_receipt_plan receipt[RR_MAX_EDGE_ACTIVATIONS];
};

struct rr_plan {
    struct rr_action forward[RR_MAX_ACTIONS];
    struct rr_action reverse[RR_MAX_ACTIONS];
    size_t forward_count;
    size_t reverse_count;
    size_t predicted_peak_live;
    size_t projection_live;
    size_t maximum_rank;
    size_t selected_target;
    unsigned selected_public_id;
    unsigned char reconstructible_leaf[RC_MAX_NODES];
    unsigned char rank[RC_MAX_NODES];
    unsigned char rematerializable[RC_MAX_NODES];
    size_t outgoing_edge[RC_MAX_NODES];
    struct rr_edge_plan edge[DC_MAX_EDGES];
    size_t receipt_count;
    size_t exact_receipts;
    size_t leaf_rebind_receipts;
    size_t operator_rebind_receipts;
    uint64_t topology_hash;
    uint64_t schedule_hash;
    uint64_t hash;
};

struct rr_symbolic_node {
    int seen;
    int live;
    uint64_t next_generation;
    uint64_t live_generation;
};

struct rr_symbolic_edge {
    size_t opened;
    size_t closed;
};

struct rr_generation {
    int born;
    int live;
    int closed;
    size_t slot;
    uint64_t serial;
};

struct rr_node_runtime {
    int seen_original;
    int obligation;
    int live;
    uint64_t next_generation;
    uint64_t live_generation;
    size_t original_slot;
    uint64_t original_serial;
    uint64_t births;
    uint64_t closes;
    struct rr_generation generation[RR_MAX_NODE_ACTIVATIONS];
};

struct rr_receipt_runtime {
    enum rr_receipt_state state;
    size_t forward_slot;
    uint64_t forward_serial;
    size_t inverse_slot;
    uint64_t inverse_serial;
    int authorization_consumed;
    int lawful;
};

struct rr_edge_runtime {
    size_t opened;
    size_t closed;
    struct rr_receipt_runtime receipt[RR_MAX_EDGE_ACTIVATIONS];
};

struct rr_obligation {
    int active;
    size_t node;
    unsigned public_node_id;
    enum rc_kind kind;
    struct rc_signature signature;
    unsigned left_public_id;
    unsigned right_public_id;
    size_t outgoing_edge;
    size_t rank;
    uint64_t program_epoch;
    uint64_t plan_hash;
    size_t original_slot;
    uint64_t original_serial;
    uint64_t reconstruction_episodes;
};

struct rr_machine {
    uint64_t next_serial;
    uint64_t restoration_generation;
    uint64_t program_epoch;
};

struct rr_execution {
    struct ga_boundary boundary;
    struct ga_stats stats;
    struct ir_owner owner[RC_MAX_NODES];
    uint64_t plan_hash;
    uint64_t program_epoch;
    size_t forward_actions;
    size_t reverse_actions;
    size_t forward_cursor;
    size_t reverse_cursor;
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
    uint64_t physical_forward_receipts;
    uint64_t physical_inverse_receipts;
    size_t multi_activation_edges;
    size_t second_or_later_activations;
    size_t exact_receipts;
    size_t leaf_rebind_receipts;
    size_t operator_rebind_receipts;
    size_t maximum_activation_ordinal;
    size_t nested_frame_depth;
    size_t activation_chain_depth;
    uint64_t shared_forward_activations;
    uint64_t shared_inverse_activations;
    uint64_t operator_reconstructions;
    uint64_t operator_suspensions;
    uint64_t persistent_obligations_created;
    uint64_t persistent_obligations_cleared;
    int receipts_restored;
    int nodes_restored;
    int obligations_restored;
    int logical_custody_restored;
    int allocator_restored;
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

static int rr_is_forward(enum rr_opcode opcode) {
    return opcode == RR_LEAF_FORWARD
        || opcode == RR_OPERATOR_FORWARD;
}

static int rr_is_leaf(enum rr_opcode opcode) {
    return opcode == RR_LEAF_FORWARD
        || opcode == RR_LEAF_INVERSE;
}

static enum rr_opcode rr_inverse_opcode(enum rr_opcode opcode) {
    if (opcode == RR_LEAF_FORWARD) {
        return RR_LEAF_INVERSE;
    }
    if (opcode == RR_LEAF_INVERSE) {
        return RR_LEAF_FORWARD;
    }
    if (opcode == RR_OPERATOR_FORWARD) {
        return RR_OPERATOR_INVERSE;
    }
    if (opcode == RR_OPERATOR_INVERSE) {
        return RR_OPERATOR_FORWARD;
    }
    fail("recursive rematerialization opcode invalid");
    return RR_LEAF_FORWARD;
}

static size_t rr_outgoing_edge(
    const struct dc_compiled *compiled,
    size_t producer
) {
    return ir_outgoing_edge(compiled, producer);
}

static void rr_symbolic_apply(
    const struct dc_compiled *compiled,
    struct rr_symbolic_node node_state[RC_MAX_NODES],
    struct rr_action *action,
    size_t *live,
    size_t *peak
) {
    if (action->node >= compiled->graph.count) {
        fail("recursive rematerialization symbolic node invalid");
    }
    struct rr_symbolic_node *state = &node_state[action->node];
    if (rr_is_forward(action->opcode)) {
        if (
            state->live
            || action->generation != state->next_generation
        ) {
            fail("recursive rematerialization symbolic birth invalid");
        }
        if (!rr_is_leaf(action->opcode)) {
            const struct rc_node *node =
                &compiled->graph.node[action->node];
            if (
                !node_state[node->left].live
                || !node_state[node->right].live
            ) {
                fail("recursive rematerialization symbolic child dormant");
            }
        }
        state->seen = 1;
        state->live = 1;
        state->live_generation = action->generation;
        ++state->next_generation;
        ++*live;
        if (*live > *peak) {
            *peak = *live;
        }
    } else {
        if (
            !state->live
            || action->generation != state->live_generation
            || *live == 0U
        ) {
            fail("recursive rematerialization symbolic close invalid");
        }
        state->live = 0;
        state->live_generation = RR_NO_GENERATION;
        --*live;
    }
    action->expected_live_after = *live;
}

static void rr_append_forward(
    struct rr_plan *plan,
    const struct dc_compiled *compiled,
    struct rr_symbolic_node node_state[RC_MAX_NODES],
    enum rr_opcode opcode,
    enum rr_purpose purpose,
    size_t node,
    int logical_transition,
    size_t *live
) {
    if (plan->forward_count >= RR_MAX_ACTIONS) {
        fail("recursive rematerialization forward tape capacity");
    }
    struct rr_symbolic_node *state = &node_state[node];
    const uint64_t generation = rr_is_forward(opcode)
        ? state->next_generation
        : state->live_generation;
    struct rr_action *action =
        &plan->forward[plan->forward_count++];
    *action = (struct rr_action){
        .opcode = opcode,
        .purpose = purpose,
        .node = node,
        .public_node_id = compiled->graph.node[node].id,
        .generation = generation,
        .logical_transition = logical_transition
    };
    rr_symbolic_apply(
        compiled,
        node_state,
        action,
        live,
        &plan->predicted_peak_live
    );
}

static void rr_build_reverse(
    struct rr_plan *plan,
    const struct dc_compiled *compiled,
    struct rr_symbolic_node node_state[RC_MAX_NODES],
    size_t *live
) {
    size_t peak = plan->predicted_peak_live;
    for (
        size_t cursor = plan->forward_count;
        cursor > 0U;
        --cursor
    ) {
        if (plan->reverse_count >= RR_MAX_ACTIONS) {
            fail("recursive rematerialization reverse tape capacity");
        }
        const struct rr_action *forward =
            &plan->forward[cursor - 1U];
        struct rr_symbolic_node *state =
            &node_state[forward->node];
        const enum rr_opcode opcode =
            rr_inverse_opcode(forward->opcode);
        const uint64_t generation = rr_is_forward(opcode)
            ? state->next_generation
            : state->live_generation;
        struct rr_action *reverse =
            &plan->reverse[plan->reverse_count++];
        *reverse = (struct rr_action){
            .opcode = opcode,
            .purpose = rr_is_forward(opcode)
                ? RR_OPERATOR_RECONSTRUCTION
                : RR_FINAL_RESTORATION,
            .node = forward->node,
            .public_node_id = forward->public_node_id,
            .generation = generation,
            .logical_transition =
                forward->logical_transition
        };
        if (rr_is_leaf(opcode)) {
            reverse->purpose = rr_is_forward(opcode)
                ? RR_PUBLIC_LEAF_EVICTION
                : RR_FINAL_RESTORATION;
        } else if (
            !rr_is_forward(opcode)
            && !forward->logical_transition
        ) {
            reverse->purpose = RR_OPERATOR_SUSPENSION;
        }
        rr_symbolic_apply(
            compiled,
            node_state,
            reverse,
            live,
            &peak
        );
    }
    plan->predicted_peak_live = peak;
    if (*live != 0U) {
        fail("recursive rematerialization symbolic tape not closed");
    }
}

static void rr_receipt_open(
    struct rr_plan *plan,
    struct rr_symbolic_edge edge_state[DC_MAX_EDGES],
    size_t edge,
    uint64_t consumer_generation,
    uint64_t producer_generation,
    size_t action
) {
    struct rr_edge_plan *edge_plan = &plan->edge[edge];
    struct rr_symbolic_edge *state = &edge_state[edge];
    if (
        state->opened >= RR_MAX_EDGE_ACTIVATIONS
        || state->opened != state->closed
    ) {
        fail("recursive rematerialization symbolic receipt open");
    }
    const size_t ordinal = state->opened++;
    struct rr_receipt_plan *receipt =
        &edge_plan->receipt[ordinal];
    *receipt = (struct rr_receipt_plan){
        .edge = edge,
        .ordinal = ordinal,
        .consumer_generation = consumer_generation,
        .forward_producer_generation = producer_generation,
        .inverse_producer_generation = RR_NO_GENERATION,
        .forward_action = action,
        .inverse_action = SIZE_MAX
    };
}

static void rr_receipt_close(
    struct rr_plan *plan,
    const struct dc_compiled *compiled,
    struct rr_symbolic_edge edge_state[DC_MAX_EDGES],
    size_t edge,
    uint64_t consumer_generation,
    uint64_t producer_generation,
    size_t action
) {
    struct rr_symbolic_edge *state = &edge_state[edge];
    if (state->closed >= state->opened) {
        fail("recursive rematerialization symbolic receipt close");
    }
    const size_t ordinal = state->closed++;
    struct rr_receipt_plan *receipt =
        &plan->edge[edge].receipt[ordinal];
    if (
        receipt->edge != edge
        || receipt->ordinal != ordinal
        || receipt->consumer_generation != consumer_generation
        || receipt->inverse_action != SIZE_MAX
    ) {
        fail("recursive rematerialization receipt pairing invalid");
    }
    const size_t producer = compiled->edge[edge].producer;
    const size_t consumer = compiled->edge[edge].consumer;
    receipt->producer = producer;
    receipt->consumer = consumer;
    receipt->inverse_producer_generation = producer_generation;
    receipt->inverse_action = action;
    if (
        receipt->forward_producer_generation
        == receipt->inverse_producer_generation
    ) {
        receipt->policy = RR_RECEIPT_EXACT;
        ++plan->exact_receipts;
    } else if (compiled->graph.node[producer].kind == RC_LEAF) {
        if (plan->reconstructible_leaf[producer] == 0U) {
            fail("recursive rematerialization unauthorized leaf rebind");
        }
        receipt->policy = RR_RECEIPT_LEAF_REBIND;
        ++plan->leaf_rebind_receipts;
    } else {
        if (plan->rematerializable[producer] == 0U) {
            fail("recursive rematerialization unauthorized op rebind");
        }
        receipt->policy = RR_RECEIPT_OPERATOR_REBIND;
        ++plan->operator_rebind_receipts;
    }
    ++plan->receipt_count;
}

static void rr_compile_receipts(
    struct rr_plan *plan,
    const struct dc_compiled *compiled
) {
    struct rr_symbolic_node node_state[RC_MAX_NODES] = {0};
    struct rr_symbolic_edge edge_state[DC_MAX_EDGES] = {0};
    size_t live = 0U;
    size_t peak = 0U;
    const size_t total =
        plan->forward_count + plan->reverse_count;
    for (size_t action_index = 0U; action_index < total; ++action_index) {
        struct rr_action *action = action_index < plan->forward_count
            ? &plan->forward[action_index]
            : &plan->reverse[action_index - plan->forward_count];
        const struct rc_node *node =
            &compiled->graph.node[action->node];
        if (!rr_is_leaf(action->opcode)) {
            const size_t input_edge[] = {
                compiled->left_edge[action->node],
                compiled->right_edge[action->node]
            };
            const size_t input_node[] = {
                node->left,
                node->right
            };
            for (size_t side = 0U; side < 2U; ++side) {
                if (!node_state[input_node[side]].live) {
                    fail(
                        "recursive rematerialization receipt operand dormant"
                    );
                }
                if (rr_is_forward(action->opcode)) {
                    rr_receipt_open(
                        plan,
                        edge_state,
                        input_edge[side],
                        action->generation,
                        node_state[input_node[side]].live_generation,
                        action_index
                    );
                } else {
                    rr_receipt_close(
                        plan,
                        compiled,
                        edge_state,
                        input_edge[side],
                        action->generation,
                        node_state[input_node[side]].live_generation,
                        action_index
                    );
                }
            }
        }
        rr_symbolic_apply(
            compiled,
            node_state,
            action,
            &live,
            &peak
        );
    }
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        if (
            edge_state[edge].opened == 0U
            || edge_state[edge].opened != edge_state[edge].closed
            || edge_state[edge].opened > RR_MAX_EDGE_ACTIVATIONS
        ) {
            fail("recursive rematerialization receipt plan incomplete");
        }
        plan->edge[edge].activation_limit =
            edge_state[edge].opened;
    }
    if (live != 0U || peak != plan->predicted_peak_live) {
        fail("recursive rematerialization receipt simulation changed");
    }
}

static uint64_t rr_hash_plan(const struct rr_plan *plan) {
    uint64_t hash = UINT64_C(14695981039346656037);
    rc_hash_word(&hash, plan->forward_count);
    rc_hash_word(&hash, plan->reverse_count);
    rc_hash_word(&hash, plan->predicted_peak_live);
    rc_hash_word(&hash, plan->projection_live);
    rc_hash_word(&hash, plan->maximum_rank);
    rc_hash_word(&hash, plan->selected_target);
    rc_hash_word(&hash, plan->selected_public_id);
    rc_hash_word(&hash, (size_t)plan->topology_hash);
    rc_hash_word(&hash, (size_t)plan->schedule_hash);
    for (size_t node = 0U; node < RC_MAX_NODES; ++node) {
        rc_hash_word(&hash, plan->reconstructible_leaf[node]);
        rc_hash_word(&hash, plan->rank[node]);
        rc_hash_word(&hash, plan->rematerializable[node]);
    }
    for (size_t direction = 0U; direction < 2U; ++direction) {
        const struct rr_action *action = direction == 0U
            ? plan->forward
            : plan->reverse;
        const size_t count = direction == 0U
            ? plan->forward_count
            : plan->reverse_count;
        for (size_t index = 0U; index < count; ++index) {
            rc_hash_word(&hash, (size_t)action[index].opcode);
            rc_hash_word(&hash, (size_t)action[index].purpose);
            rc_hash_word(&hash, action[index].node);
            rc_hash_word(&hash, action[index].public_node_id);
            rc_hash_word(&hash, (size_t)action[index].generation);
            rc_hash_word(
                &hash,
                (size_t)action[index].logical_transition
            );
            rc_hash_word(&hash, action[index].expected_live_after);
        }
    }
    for (size_t edge = 0U; edge < DC_MAX_EDGES; ++edge) {
        rc_hash_word(&hash, plan->edge[edge].activation_limit);
        for (
            size_t ordinal = 0U;
            ordinal < plan->edge[edge].activation_limit;
            ++ordinal
        ) {
            const struct rr_receipt_plan *receipt =
                &plan->edge[edge].receipt[ordinal];
            rc_hash_word(&hash, (size_t)receipt->policy);
            rc_hash_word(&hash, receipt->edge);
            rc_hash_word(&hash, receipt->ordinal);
            rc_hash_word(&hash, receipt->producer);
            rc_hash_word(&hash, receipt->consumer);
            rc_hash_word(
                &hash,
                (size_t)receipt->consumer_generation
            );
            rc_hash_word(
                &hash,
                (size_t)receipt->forward_producer_generation
            );
            rc_hash_word(
                &hash,
                (size_t)receipt->inverse_producer_generation
            );
            rc_hash_word(&hash, receipt->forward_action);
            rc_hash_word(&hash, receipt->inverse_action);
        }
    }
    return hash;
}

static struct rr_plan rr_compile_plan(
    const struct dc_compiled *compiled
) {
    struct rr_plan plan = {0};
    plan.topology_hash = compiled->graph.topology_hash;
    plan.schedule_hash = compiled->graph.schedule_hash;
    plan.selected_target = SIZE_MAX;
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        plan.outgoing_edge[index] = SIZE_MAX;
    }
    for (
        size_t cursor = 0U;
        cursor < compiled->graph.schedule_count;
        ++cursor
    ) {
        const size_t index = compiled->graph.schedule[cursor];
        const struct rc_node *node = &compiled->graph.node[index];
        if (
            node->kind == RC_LEAF
            && node->indegree == 1U
            && compiled->is_shared[index] == 0U
        ) {
            plan.reconstructible_leaf[index] = 1U;
            plan.outgoing_edge[index] =
                rr_outgoing_edge(compiled, index);
            continue;
        }
        if (
            node->kind == RC_LEAF
            || node->indegree != 1U
            || compiled->is_shared[index] != 0U
        ) {
            continue;
        }
        unsigned char rank = 0U;
        if (
            compiled->is_shared[node->left] != 0U
            && compiled->is_shared[node->right] != 0U
        ) {
            rank = 1U;
        } else if (
            plan.rank[node->left] > 0U
            && plan.rank[node->right] > 0U
        ) {
            const unsigned char child_rank =
                plan.rank[node->left] > plan.rank[node->right]
                ? plan.rank[node->left]
                : plan.rank[node->right];
            if (child_rank == UCHAR_MAX) {
                fail("recursive rematerialization rank overflow");
            }
            rank = (unsigned char)(child_rank + 1U);
        }
        plan.rank[index] = rank;
        if (rank > 0U) {
            plan.outgoing_edge[index] =
                rr_outgoing_edge(compiled, index);
        }
        if (rank > plan.maximum_rank) {
            plan.maximum_rank = rank;
            plan.selected_target = index;
            plan.selected_public_id = node->id;
        } else if (
            rank == plan.maximum_rank
            && rank > 1U
            && (
                plan.selected_target == SIZE_MAX
                || node->id < plan.selected_public_id
            )
        ) {
            plan.selected_target = index;
            plan.selected_public_id = node->id;
        }
    }
    if (
        plan.maximum_rank != 2U
        || plan.selected_target == SIZE_MAX
        || plan.selected_public_id != 813U
    ) {
        fail("recursive rematerialization target selection failed");
    }
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        if (
            plan.rank[index] == 1U
            || index == plan.selected_target
        ) {
            plan.rematerializable[index] = 1U;
        }
    }

    struct rr_symbolic_node node_state[RC_MAX_NODES] = {0};
    size_t live = 0U;
    for (
        size_t cursor = 0U;
        cursor < compiled->graph.schedule_count;
        ++cursor
    ) {
        const size_t index = compiled->graph.schedule[cursor];
        const struct rc_node *node = &compiled->graph.node[index];
        rr_append_forward(
            &plan,
            compiled,
            node_state,
            node->kind == RC_LEAF
                ? RR_LEAF_FORWARD
                : RR_OPERATOR_FORWARD,
            RR_ORIGINAL,
            index,
            node->kind != RC_LEAF,
            &live
        );
        if (node->kind == RC_LEAF) {
            continue;
        }
        const size_t child[] = {node->right, node->left};
        for (size_t side = 0U; side < 2U; ++side) {
            const size_t producer = child[side];
            if (plan.reconstructible_leaf[producer] != 0U) {
                rr_append_forward(
                    &plan,
                    compiled,
                    node_state,
                    RR_LEAF_INVERSE,
                    RR_PUBLIC_LEAF_EVICTION,
                    producer,
                    0,
                    &live
                );
            } else if (plan.rank[producer] == 1U) {
                rr_append_forward(
                    &plan,
                    compiled,
                    node_state,
                    RR_OPERATOR_INVERSE,
                    RR_OPERATOR_EVICTION,
                    producer,
                    0,
                    &live
                );
            }
        }
    }
    const struct rc_node *target =
        &compiled->graph.node[plan.selected_target];
    rr_append_forward(
        &plan,
        compiled,
        node_state,
        RR_OPERATOR_FORWARD,
        RR_OPERATOR_RECONSTRUCTION,
        target->left,
        0,
        &live
    );
    rr_append_forward(
        &plan,
        compiled,
        node_state,
        RR_OPERATOR_FORWARD,
        RR_OPERATOR_RECONSTRUCTION,
        target->right,
        0,
        &live
    );
    rr_append_forward(
        &plan,
        compiled,
        node_state,
        RR_OPERATOR_INVERSE,
        RR_OPERATOR_EVICTION,
        plan.selected_target,
        0,
        &live
    );
    rr_append_forward(
        &plan,
        compiled,
        node_state,
        RR_OPERATOR_INVERSE,
        RR_OPERATOR_SUSPENSION,
        target->right,
        0,
        &live
    );
    rr_append_forward(
        &plan,
        compiled,
        node_state,
        RR_OPERATOR_INVERSE,
        RR_OPERATOR_SUSPENSION,
        target->left,
        0,
        &live
    );
    plan.projection_live = live;
    rr_build_reverse(&plan, compiled, node_state, &live);
    rr_compile_receipts(&plan, compiled);
    plan.hash = rr_hash_plan(&plan);
    if (
        plan.forward_count != 28U
        || plan.reverse_count != 28U
        || plan.predicted_peak_live != 9U
        || plan.projection_live != 6U
        || plan.receipt_count != 40U
        || plan.exact_receipts != 29U
        || plan.leaf_rebind_receipts != 4U
        || plan.operator_rebind_receipts != 7U
    ) {
        fail("recursive rematerialization compiled law changed");
    }
    return plan;
}

static void rr_validate_plan(
    const struct rr_plan *plan,
    const struct dc_compiled *compiled
) {
    if (
        plan->topology_hash != compiled->graph.topology_hash
        || plan->schedule_hash != compiled->graph.schedule_hash
        || plan->hash != rr_hash_plan(plan)
        || plan->selected_target >= compiled->graph.count
        || compiled->graph.node[plan->selected_target].id
            != plan->selected_public_id
    ) {
        fail("recursive rematerialization plan provenance changed");
    }
    for (size_t direction = 0U; direction < 2U; ++direction) {
        const struct rr_action *action = direction == 0U
            ? plan->forward
            : plan->reverse;
        const size_t count = direction == 0U
            ? plan->forward_count
            : plan->reverse_count;
        for (size_t index = 0U; index < count; ++index) {
            if (
                action[index].node >= compiled->graph.count
                || action[index].public_node_id
                    != compiled->graph.node[action[index].node].id
            ) {
                fail(
                    "recursive rematerialization action identity changed"
                );
            }
        }
    }
}

static void rr_validate_obligation(
    const struct rr_plan *plan,
    const struct dc_compiled *compiled,
    const struct rr_obligation *obligation,
    size_t node,
    uint64_t program_epoch
) {
    const struct rc_node *descriptor = &compiled->graph.node[node];
    if (
        !obligation->active
        || obligation->node != node
        || obligation->public_node_id != descriptor->id
        || obligation->kind != descriptor->kind
        || !rc_signature_equal(
            obligation->signature,
            descriptor->signature
        )
        || obligation->left_public_id
            != (
                descriptor->kind == RC_LEAF
                ? 0U
                : compiled->graph.node[descriptor->left].id
            )
        || obligation->right_public_id
            != (
                descriptor->kind == RC_LEAF
                ? 0U
                : compiled->graph.node[descriptor->right].id
            )
        || obligation->outgoing_edge
            != plan->outgoing_edge[node]
        || obligation->rank != plan->rank[node]
        || obligation->program_epoch != program_epoch
        || obligation->plan_hash != plan->hash
    ) {
        fail("recursive rematerialization obligation invalid");
    }
}

static void rr_create_obligation(
    const struct rr_plan *plan,
    const struct dc_compiled *compiled,
    struct rr_obligation obligation[RC_MAX_NODES],
    const struct rr_action *action,
    struct rc_lease lease,
    uint64_t program_epoch,
    struct rr_execution *execution
) {
    const size_t node = action->node;
    const struct rc_node *descriptor = &compiled->graph.node[node];
    if (
        obligation[node].active
        || plan->outgoing_edge[node] >= compiled->edge_count
        || (
            descriptor->kind == RC_LEAF
            && plan->reconstructible_leaf[node] == 0U
        )
        || (
            descriptor->kind != RC_LEAF
            && plan->rematerializable[node] == 0U
        )
    ) {
        fail("recursive rematerialization obligation creation denied");
    }
    obligation[node] = (struct rr_obligation){
        .active = 1,
        .node = node,
        .public_node_id = descriptor->id,
        .kind = descriptor->kind,
        .signature = descriptor->signature,
        .left_public_id = descriptor->kind == RC_LEAF
            ? 0U
            : compiled->graph.node[descriptor->left].id,
        .right_public_id = descriptor->kind == RC_LEAF
            ? 0U
            : compiled->graph.node[descriptor->right].id,
        .outgoing_edge = plan->outgoing_edge[node],
        .rank = plan->rank[node],
        .program_epoch = program_epoch,
        .plan_hash = plan->hash,
        .original_slot = lease.slot,
        .original_serial = lease.serial
    };
    ++execution->persistent_obligations_created;
}

static void rr_birth_node(
    struct rr_node_runtime node_runtime[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t node,
    uint64_t generation,
    struct rc_lease lease
) {
    struct rr_node_runtime *state = &node_runtime[node];
    if (
        state->live
        || generation != state->next_generation
        || generation >= RR_MAX_NODE_ACTIVATIONS
        || state->generation[generation].born
    ) {
        fail("recursive rematerialization activation birth invalid");
    }
    struct rr_generation *record =
        &state->generation[generation];
    *record = (struct rr_generation){
        .born = 1,
        .live = 1,
        .slot = lease.slot,
        .serial = lease.serial
    };
    state->seen_original = 1;
    state->live = 1;
    state->live_generation = generation;
    state->next_generation = generation + 1U;
    ++state->births;
    node_lease[node] = lease;
}

static void rr_close_node(
    struct rr_node_runtime node_runtime[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    size_t node,
    uint64_t generation
) {
    struct rr_node_runtime *state = &node_runtime[node];
    if (
        !state->live
        || state->live_generation != generation
        || generation >= RR_MAX_NODE_ACTIVATIONS
    ) {
        fail("recursive rematerialization activation close invalid");
    }
    struct rr_generation *record =
        &state->generation[generation];
    if (
        !record->born
        || !record->live
        || record->closed
        || record->slot != node_lease[node].slot
        || record->serial != node_lease[node].serial
    ) {
        fail("recursive rematerialization generation record invalid");
    }
    record->live = 0;
    record->closed = 1;
    state->live = 0;
    state->live_generation = RR_NO_GENERATION;
    ++state->closes;
    node_lease[node] = (struct rc_lease){
        .slot = SIZE_MAX,
        .serial = 0U
    };
}

static void rr_validate_shared(
    const struct dc_compiled *compiled,
    const struct ir_owner owner[RC_MAX_NODES],
    size_t producer,
    struct rc_lease lease,
    uint64_t generation
) {
    if (compiled->is_shared[producer] == 0U) {
        return;
    }
    if (
        generation != 0U
        || !owner[producer].live
        || owner[producer].lease.slot != lease.slot
        || owner[producer].lease.serial != lease.serial
    ) {
        fail("recursive rematerialization shared owner changed");
    }
}

static const struct rr_receipt_plan *rr_preflight_open(
    const struct rr_plan *plan,
    const struct dc_compiled *compiled,
    const struct rr_edge_runtime edge_runtime[DC_MAX_EDGES],
    const struct ir_owner owner[RC_MAX_NODES],
    size_t edge,
    size_t action_index,
    uint64_t consumer_generation,
    uint64_t producer_generation,
    struct rc_lease producer_lease
) {
    const struct rr_edge_plan *edge_plan = &plan->edge[edge];
    const struct rr_edge_runtime *runtime = &edge_runtime[edge];
    if (
        runtime->opened >= edge_plan->activation_limit
        || runtime->opened >= RR_MAX_EDGE_ACTIVATIONS
        || runtime->opened != runtime->closed
    ) {
        fail("recursive rematerialization prior receipt still open");
    }
    const struct rr_receipt_plan *receipt =
        &edge_plan->receipt[runtime->opened];
    if (
        receipt->edge != edge
        || receipt->ordinal != runtime->opened
        || receipt->forward_action != action_index
        || receipt->consumer_generation != consumer_generation
        || receipt->forward_producer_generation
            != producer_generation
    ) {
        fail("recursive rematerialization receipt open mismatch");
    }
    rr_validate_shared(
        compiled,
        owner,
        compiled->edge[edge].producer,
        producer_lease,
        producer_generation
    );
    return receipt;
}

static void rr_commit_open(
    struct rr_edge_runtime edge_runtime[DC_MAX_EDGES],
    const struct rr_receipt_plan *receipt,
    struct rc_lease producer_lease,
    struct rr_execution *execution,
    const struct dc_compiled *compiled
) {
    struct rr_edge_runtime *runtime =
        &edge_runtime[receipt->edge];
    struct rr_receipt_runtime *record =
        &runtime->receipt[runtime->opened++];
    *record = (struct rr_receipt_runtime){
        .state = RR_RECEIPT_FORWARD_OPEN,
        .forward_slot = producer_lease.slot,
        .forward_serial = producer_lease.serial
    };
    ++execution->physical_forward_receipts;
    const size_t producer =
        compiled->edge[receipt->edge].producer;
    if (compiled->is_shared[producer] != 0U) {
        ++execution->owner[producer].forward_activations;
        ++execution->shared_forward_activations;
    }
}

static const struct rr_receipt_plan *rr_preflight_close(
    const struct rr_plan *plan,
    const struct dc_compiled *compiled,
    const struct rr_edge_runtime edge_runtime[DC_MAX_EDGES],
    const struct ir_owner owner[RC_MAX_NODES],
    const struct rr_obligation obligation[RC_MAX_NODES],
    size_t edge,
    size_t action_index,
    uint64_t consumer_generation,
    uint64_t producer_generation,
    struct rc_lease producer_lease,
    uint64_t program_epoch,
    enum rr_fault fault,
    size_t side
) {
    const struct rr_edge_plan *edge_plan = &plan->edge[edge];
    const struct rr_edge_runtime *runtime = &edge_runtime[edge];
    if (
        runtime->closed >= runtime->opened
        || runtime->closed >= edge_plan->activation_limit
    ) {
        fail("recursive rematerialization receipt close unavailable");
    }
    const size_t ordinal = runtime->closed;
    const struct rr_receipt_plan *receipt =
        &edge_plan->receipt[ordinal];
    uint64_t presented_consumer = consumer_generation;
    uint64_t presented_producer = producer_generation;
    if (
        fault == RR_FAULT_DEEP_STALE_CONSUMER
        && compiled->graph.node[receipt->consumer].id == 809U
        && consumer_generation == 3U
        && side == 0U
    ) {
        presented_consumer = 1U;
    }
    if (
        fault == RR_FAULT_DEEP_STALE_PRODUCER
        && compiled->graph.node[receipt->consumer].id == 813U
        && consumer_generation == 1U
        && side == 0U
    ) {
        presented_producer = 2U;
    }
    if (
        receipt->edge != edge
        || receipt->ordinal != ordinal
        || receipt->inverse_action != action_index
        || receipt->consumer_generation != presented_consumer
        || receipt->inverse_producer_generation
            != presented_producer
        || runtime->receipt[ordinal].state
            != RR_RECEIPT_FORWARD_OPEN
    ) {
        fail("recursive rematerialization receipt close mismatch");
    }
    if (
        fault == RR_FAULT_REBIND_SWAP
        && compiled->graph.node[receipt->consumer].id == 813U
        && consumer_generation == 1U
        && side == 0U
    ) {
        fail("recursive rematerialization rebind authorization swapped");
    }
    rr_validate_shared(
        compiled,
        owner,
        receipt->producer,
        producer_lease,
        producer_generation
    );
    const struct rr_receipt_runtime *record =
        &runtime->receipt[ordinal];
    if (receipt->policy == RR_RECEIPT_EXACT) {
        if (
            receipt->forward_producer_generation
                != receipt->inverse_producer_generation
            || record->forward_slot != producer_lease.slot
            || record->forward_serial != producer_lease.serial
        ) {
            fail("recursive rematerialization exact receipt changed");
        }
    } else {
        if (
            receipt->forward_producer_generation
                == receipt->inverse_producer_generation
            || record->forward_serial == producer_lease.serial
            || record->authorization_consumed
        ) {
            fail("recursive rematerialization rebind not fresh");
        }
        rr_validate_obligation(
            plan,
            compiled,
            &obligation[receipt->producer],
            receipt->producer,
            program_epoch
        );
    }
    return receipt;
}

static void rr_commit_close(
    struct rr_edge_runtime edge_runtime[DC_MAX_EDGES],
    const struct rr_receipt_plan *receipt,
    struct rc_lease producer_lease,
    struct rr_execution *execution,
    const struct dc_compiled *compiled
) {
    struct rr_edge_runtime *runtime =
        &edge_runtime[receipt->edge];
    struct rr_receipt_runtime *record =
        &runtime->receipt[runtime->closed++];
    record->state = RR_RECEIPT_INVERSE_CLOSED;
    record->inverse_slot = producer_lease.slot;
    record->inverse_serial = producer_lease.serial;
    record->authorization_consumed =
        receipt->policy != RR_RECEIPT_EXACT;
    record->lawful = 1;
    ++execution->physical_inverse_receipts;
    const size_t producer =
        compiled->edge[receipt->edge].producer;
    if (compiled->is_shared[producer] != 0U) {
        ++execution->owner[producer].inverse_activations;
        ++execution->shared_inverse_activations;
    }
}

static void rr_execute_leaf(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct rr_plan *plan,
    struct rr_node_runtime node_runtime[RC_MAX_NODES],
    struct rr_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    const struct rr_action *action,
    uint64_t program_epoch,
    struct rr_execution *execution
) {
    const size_t node = action->node;
    const struct rc_node *descriptor = &compiled->graph.node[node];
    if (descriptor->kind != RC_LEAF) {
        fail("recursive rematerialization leaf action on operator");
    }
    if (action->opcode == RR_LEAF_FORWARD) {
        if (action->generation > 0U) {
            rr_validate_obligation(
                plan,
                compiled,
                &obligation[node],
                node,
                program_epoch
            );
            ++obligation[node].reconstruction_episodes;
        } else if (obligation[node].active) {
            fail("recursive rematerialization original leaf obligated");
        }
        const struct rc_lease lease =
            rc_allocate(runtime, node, descriptor->signature);
        ga_encode_relation(
            runtime->carrier,
            runtime->program->leaf[descriptor->leaf_index],
            rc_slot_start(lease.slot),
            0,
            &runtime->stats
        );
        runtime->slot[lease.slot].reserved = 0;
        ++runtime->forward_leaf_encodes;
        rr_birth_node(
            node_runtime,
            node_lease,
            node,
            action->generation,
            lease
        );
        return;
    }
    if (action->opcode != RR_LEAF_INVERSE) {
        fail("recursive rematerialization leaf opcode invalid");
    }
    const struct rc_lease lease = node_lease[node];
    if (
        !node_runtime[node].live
        || node_runtime[node].live_generation
            != action->generation
    ) {
        fail("recursive rematerialization leaf generation absent");
    }
    rc_validate_lease(
        runtime, lease, node, descriptor->signature, 0
    );
    ga_encode_relation(
        runtime->carrier,
        runtime->program->leaf[descriptor->leaf_index],
        rc_slot_start(lease.slot),
        1,
        &runtime->stats
    );
    ++runtime->inverse_leaf_encodes;
    rc_release(runtime, lease, node, descriptor->signature, 0);
    rr_close_node(
        node_runtime,
        node_lease,
        node,
        action->generation
    );
    if (action->purpose == RR_PUBLIC_LEAF_EVICTION) {
        rr_create_obligation(
            plan,
            compiled,
            obligation,
            action,
            lease,
            program_epoch,
            execution
        );
    } else {
        rr_validate_obligation(
            plan,
            compiled,
            &obligation[node],
            node,
            program_epoch
        );
        obligation[node] = (struct rr_obligation){0};
        ++execution->persistent_obligations_cleared;
    }
}

static void rr_execute_operator_forward(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct rr_plan *plan,
    struct dc_custody *logical,
    struct rr_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct rr_node_runtime node_runtime[RC_MAX_NODES],
    struct rr_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    const struct rr_action *action,
    size_t action_index,
    uint64_t program_epoch,
    enum rr_fault fault,
    struct rr_execution *execution
) {
    const size_t index = action->node;
    const struct rc_node *node = &compiled->graph.node[index];
    if (node->kind == RC_LEAF) {
        fail("recursive rematerialization operator forward leaf");
    }
    uint64_t presented_generation = action->generation;
    if (
        fault == RR_FAULT_ACTIVATION_REUSE
        && node->id == 809U
        && action->generation == 3U
    ) {
        presented_generation = 2U;
    }
    if (
        node_runtime[index].live
        || node_runtime[index].next_generation
            != presented_generation
    ) {
        fail("recursive rematerialization operator generation reused");
    }
    if (action->generation > 0U) {
        rr_validate_obligation(
            plan,
            compiled,
            &obligation[index],
            index,
            program_epoch
        );
        ++obligation[index].reconstruction_episodes;
    } else if (obligation[index].active) {
        fail("recursive rematerialization original operator obligated");
    }
    if (
        fault == RR_FAULT_MISSING_NESTED_CHILD
        && node->id == 813U
        && action->generation == 1U
    ) {
        fail("recursive rematerialization nested child missing");
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
    const struct rr_receipt_plan *receipt[2];
    for (size_t side = 0U; side < 2U; ++side) {
        const size_t producer = input_node[side];
        if (!node_runtime[producer].live) {
            fail("recursive rematerialization forward child dormant");
        }
        receipt[side] = rr_preflight_open(
            plan,
            compiled,
            edge_runtime,
            execution->owner,
            input_edge[side],
            action_index,
            action->generation,
            node_runtime[producer].live_generation,
            input_lease[side]
        );
        if (
            action->logical_transition
            && !dc_prepare_forward_edge(
                compiled,
                logical,
                input_edge[side],
                input_lease[side],
                0,
                0,
                SIZE_MAX
            )
        ) {
            fail("recursive rematerialization logical forward skipped");
        }
    }
    const struct rc_lease output =
        rc_allocate(runtime, index, node->signature);
    rc_apply_node(
        runtime,
        index,
        input_lease[0],
        input_lease[1],
        output,
        0,
        0
    );
    for (size_t side = 0U; side < 2U; ++side) {
        rr_commit_open(
            edge_runtime,
            receipt[side],
            input_lease[side],
            execution,
            compiled
        );
        if (action->logical_transition) {
            dc_commit_forward_edge(
                compiled,
                logical,
                input_edge[side],
                input_lease[side],
                0,
                SIZE_MAX
            );
        }
    }
    rr_birth_node(
        node_runtime,
        node_lease,
        index,
        action->generation,
        output
    );
    if (compiled->is_shared[index] != 0U) {
        ir_record_owner_birth(
            compiled,
            logical,
            execution->owner,
            index,
            output
        );
    }
    if (action->generation > 0U) {
        ++execution->operator_reconstructions;
    }
}

static void rr_execute_operator_inverse(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct rr_plan *plan,
    struct dc_custody *logical,
    struct rr_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct rr_node_runtime node_runtime[RC_MAX_NODES],
    struct rr_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    const struct rr_action *action,
    size_t action_index,
    uint64_t program_epoch,
    enum rr_fault fault,
    int wrong,
    struct rr_execution *execution
) {
    const size_t index = action->node;
    const struct rc_node *node = &compiled->graph.node[index];
    if (
        node->kind == RC_LEAF
        || !node_runtime[index].live
        || node_runtime[index].live_generation
            != action->generation
    ) {
        fail("recursive rematerialization operator inverse absent");
    }
    if (action->logical_transition) {
        dc_require_inverse_ready(compiled, logical, index);
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
    const struct rr_receipt_plan *receipt[2];
    for (size_t side = 0U; side < 2U; ++side) {
        const size_t producer = input_node[side];
        if (!node_runtime[producer].live) {
            fail("recursive rematerialization inverse child dormant");
        }
        receipt[side] = rr_preflight_close(
            plan,
            compiled,
            edge_runtime,
            execution->owner,
            obligation,
            input_edge[side],
            action_index,
            action->generation,
            node_runtime[producer].live_generation,
            input_lease[side],
            program_epoch,
            fault,
            side
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
        if (
            fault == RR_FAULT_MISSING_NESTED_CLOSE
            && node->id == 809U
            && action->generation == 1U
            && side == 0U
        ) {
            continue;
        }
        rr_commit_close(
            edge_runtime,
            receipt[side],
            input_lease[side],
            execution,
            compiled
        );
        if (action->logical_transition) {
            dc_inverse_edge(
                compiled,
                logical,
                input_edge[side],
                input_lease[side]
            );
        }
    }
    rc_release(runtime, output, index, node->signature, wrong);
    if (compiled->is_shared[index] != 0U) {
        ir_record_owner_release(
            compiled, logical, execution->owner, index
        );
    }
    rr_close_node(
        node_runtime,
        node_lease,
        index,
        action->generation
    );
    if (!action->logical_transition) {
        if (!obligation[index].active) {
            rr_create_obligation(
                plan,
                compiled,
                obligation,
                action,
                output,
                program_epoch,
                execution
            );
        } else {
            rr_validate_obligation(
                plan,
                compiled,
                &obligation[index],
                index,
                program_epoch
            );
            ++execution->operator_suspensions;
        }
    } else if (obligation[index].active) {
        rr_validate_obligation(
            plan,
            compiled,
            &obligation[index],
            index,
            program_epoch
        );
        obligation[index] = (struct rr_obligation){0};
        ++execution->persistent_obligations_cleared;
    }
}

static void rr_execute_action(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    const struct rr_plan *plan,
    struct dc_custody *logical,
    struct rr_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct rr_node_runtime node_runtime[RC_MAX_NODES],
    struct rr_obligation obligation[RC_MAX_NODES],
    struct rc_lease node_lease[RC_MAX_NODES],
    const struct rr_action *action,
    size_t action_index,
    uint64_t program_epoch,
    enum rr_fault fault,
    int wrong,
    struct rr_execution *execution
) {
    if (
        action->node >= compiled->graph.count
        || action->public_node_id
            != compiled->graph.node[action->node].id
    ) {
        fail("recursive rematerialization runtime action changed");
    }
    if (rr_is_leaf(action->opcode)) {
        rr_execute_leaf(
            runtime,
            compiled,
            plan,
            node_runtime,
            obligation,
            node_lease,
            action,
            program_epoch,
            execution
        );
    } else if (action->opcode == RR_OPERATOR_FORWARD) {
        rr_execute_operator_forward(
            runtime,
            compiled,
            plan,
            logical,
            edge_runtime,
            node_runtime,
            obligation,
            node_lease,
            action,
            action_index,
            program_epoch,
            fault,
            execution
        );
    } else if (action->opcode == RR_OPERATOR_INVERSE) {
        rr_execute_operator_inverse(
            runtime,
            compiled,
            plan,
            logical,
            edge_runtime,
            node_runtime,
            obligation,
            node_lease,
            action,
            action_index,
            program_epoch,
            fault,
            wrong,
            execution
        );
    } else {
        fail("recursive rematerialization runtime opcode invalid");
    }
    if (runtime->live != action->expected_live_after) {
        fail("recursive rematerialization live annotation changed");
    }
}

static int rr_receipts_complete(
    const struct rr_plan *plan,
    const struct dc_compiled *compiled,
    const struct rr_edge_runtime edge_runtime[DC_MAX_EDGES]
) {
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        if (
            edge_runtime[edge].opened
                != plan->edge[edge].activation_limit
            || edge_runtime[edge].closed
                != plan->edge[edge].activation_limit
        ) {
            return 0;
        }
        for (
            size_t ordinal = 0U;
            ordinal < plan->edge[edge].activation_limit;
            ++ordinal
        ) {
            const struct rr_receipt_runtime *receipt =
                &edge_runtime[edge].receipt[ordinal];
            if (
                receipt->state != RR_RECEIPT_INVERSE_CLOSED
                || !receipt->lawful
                || (
                    plan->edge[edge].receipt[ordinal].policy
                        != RR_RECEIPT_EXACT
                    && !receipt->authorization_consumed
                )
            ) {
                return 0;
            }
        }
    }
    return 1;
}

static int rr_nodes_complete(
    const struct dc_compiled *compiled,
    const struct rr_node_runtime node_runtime[RC_MAX_NODES]
) {
    for (size_t node = 0U; node < compiled->graph.count; ++node) {
        const struct rr_node_runtime *state = &node_runtime[node];
        if (
            !state->seen_original
            || state->live
            || state->births != state->closes
            || state->live_generation != RR_NO_GENERATION
        ) {
            return 0;
        }
        for (
            size_t generation = 0U;
            generation < state->next_generation;
            ++generation
        ) {
            if (
                generation >= RR_MAX_NODE_ACTIVATIONS
                || !state->generation[generation].born
                || state->generation[generation].live
                || !state->generation[generation].closed
            ) {
                return 0;
            }
        }
    }
    return 1;
}

static int rr_obligations_complete(
    const struct rr_obligation obligation[RC_MAX_NODES]
) {
    for (size_t node = 0U; node < RC_MAX_NODES; ++node) {
        if (obligation[node].active) {
            return 0;
        }
    }
    return 1;
}

static void rr_capture_counts(
    const struct rr_plan *plan,
    const struct dc_compiled *compiled,
    const struct rr_edge_runtime edge_runtime[DC_MAX_EDGES],
    struct rr_execution *execution
) {
    execution->exact_receipts = plan->exact_receipts;
    execution->leaf_rebind_receipts =
        plan->leaf_rebind_receipts;
    execution->operator_rebind_receipts =
        plan->operator_rebind_receipts;
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        const size_t limit = plan->edge[edge].activation_limit;
        if (limit > 1U) {
            ++execution->multi_activation_edges;
            execution->second_or_later_activations += limit - 1U;
        }
        if (
            edge_runtime[edge].opened != limit
            || edge_runtime[edge].closed != limit
        ) {
            continue;
        }
        for (size_t ordinal = 0U; ordinal < limit; ++ordinal) {
            const struct rr_receipt_plan *receipt =
                &plan->edge[edge].receipt[ordinal];
            if (
                receipt->forward_producer_generation
                    > execution->maximum_activation_ordinal
            ) {
                execution->maximum_activation_ordinal =
                    (size_t)receipt->forward_producer_generation;
            }
            if (
                receipt->inverse_producer_generation
                    > execution->maximum_activation_ordinal
            ) {
                execution->maximum_activation_ordinal =
                    (size_t)receipt->inverse_producer_generation;
            }
        }
    }
}

static struct rr_execution rr_execute(
    struct carrier *carrier,
    const struct dc_compiled *compiled,
    const struct rr_plan *source_plan,
    const struct rc_program *program,
    struct rr_machine *machine,
    enum rc_restore_mode restore_mode,
    enum rr_fault fault
) {
    struct rr_execution result = {0};
    struct carrier borrowed = snapshot_carrier(carrier);
    struct rr_plan plan = *source_plan;
    if (fault == RR_FAULT_TAMPER_TAPE) {
        ++plan.forward[0].expected_live_after;
    }
    rr_validate_plan(&plan, compiled);
    const uint64_t program_epoch = machine->program_epoch + 1U;
    struct rc_runtime runtime = {
        .carrier = carrier,
        .compiled = &compiled->graph,
        .program = program,
        .pool = RR_WORKING_SLOTS,
        .next_serial = machine->next_serial
    };
    struct dc_custody logical = {0};
    struct rr_edge_runtime edge_runtime[DC_MAX_EDGES] = {0};
    struct rr_node_runtime node_runtime[RC_MAX_NODES] = {0};
    struct rr_obligation obligation[RC_MAX_NODES] = {0};
    struct rc_lease node_lease[RC_MAX_NODES];
    for (size_t node = 0U; node < RC_MAX_NODES; ++node) {
        node_runtime[node].live_generation = RR_NO_GENERATION;
        node_lease[node] = (struct rc_lease){
            .slot = SIZE_MAX,
            .serial = 0U
        };
    }
    for (size_t node = 0U; node < compiled->graph.count; ++node) {
        logical.forward_pending[node] =
            compiled->graph.node[node].indegree;
        logical.inverse_pending[node] =
            compiled->graph.node[node].indegree;
    }
    result.plan_hash = plan.hash;
    result.program_epoch = program_epoch;
    result.forward_actions = plan.forward_count;
    result.reverse_actions = plan.reverse_count;
    result.predicted_peak_live = plan.predicted_peak_live;
    result.working_slots = RR_WORKING_SLOTS;
    result.nested_frame_depth = plan.maximum_rank;
    result.activation_chain_depth = plan.maximum_rank + 1U;
    result.serial_before = machine->next_serial;
    result.restoration_generation_before =
        machine->restoration_generation;

    for (
        size_t action = 0U;
        action < plan.forward_count;
        ++action
    ) {
        rr_execute_action(
            &runtime,
            compiled,
            &plan,
            &logical,
            edge_runtime,
            node_runtime,
            obligation,
            node_lease,
            &plan.forward[action],
            action,
            program_epoch,
            fault,
            0,
            &result
        );
        ++result.forward_cursor;
    }
    result.projection_live = runtime.live;
    if (
        runtime.live != plan.projection_live
        || !dc_edges_all(compiled, &logical, DC_EDGE_FORWARD)
    ) {
        fail("recursive rematerialization forward tape incomplete");
    }
    logical.projection_event = ++logical.event_clock;
    for (size_t shared = 0U; shared < compiled->fanout_nodes; ++shared) {
        const size_t node = compiled->shared_node[shared];
        dc_validate_shared_live(
            &runtime, compiled, &logical, node
        );
        result.owner[node].live_at_projection = 1;
        logical.owner[node].live_at_projection = 1;
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
        memset(node_runtime, 0, sizeof(node_runtime));
        memset(obligation, 0, sizeof(obligation));
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
        for (
            size_t action = 0U;
            action < plan.reverse_count;
            ++action
        ) {
            const struct rr_action *step = &plan.reverse[action];
            const int wrong =
                restore_mode == RC_RESTORE_WRONG_ROOT
                && step->node == root
                && step->opcode == RR_OPERATOR_INVERSE;
            rr_execute_action(
                &runtime,
                compiled,
                &plan,
                &logical,
                edge_runtime,
                node_runtime,
                obligation,
                node_lease,
                step,
                plan.forward_count + action,
                program_epoch,
                fault,
                wrong,
                &result
            );
            ++result.reverse_cursor;
            if (
                step->node == root
                && step->opcode == RR_OPERATOR_INVERSE
            ) {
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
                    result.owner[node]
                        .live_after_root_inverse = 1;
                    logical.owner[node]
                        .live_after_root_inverse = 1;
                }
                if (wrong) {
                    break;
                }
            }
        }
    }

    const int inverse_complete =
        restore_mode == RC_RESTORE_SNAPSHOT
        || (
            result.reverse_cursor == plan.reverse_count
            && dc_edges_all(compiled, &logical, DC_EDGE_INVERSE)
            && dc_counts_zero(compiled, &logical)
            && rr_receipts_complete(
                &plan, compiled, edge_runtime
            )
            && rr_nodes_complete(compiled, node_runtime)
            && rr_obligations_complete(obligation)
        );
    if (fault != RR_FAULT_NONE) {
        free_carrier(&borrowed);
        fail("recursive rematerialization injected fault escaped");
    }
    if (
        restore_mode == RC_RESTORE_CORRECT
        && !inverse_complete
    ) {
        free_carrier(&borrowed);
        fail("recursive rematerialization inverse incomplete");
    }
    if (restore_mode != RC_RESTORE_SNAPSHOT) {
        rr_capture_counts(
            &plan, compiled, edge_runtime, &result
        );
    }
    result.receipts_restored =
        restore_mode == RC_RESTORE_SNAPSHOT
        || rr_receipts_complete(&plan, compiled, edge_runtime);
    result.nodes_restored =
        restore_mode == RC_RESTORE_SNAPSHOT
        || rr_nodes_complete(compiled, node_runtime);
    result.obligations_restored =
        restore_mode == RC_RESTORE_SNAPSHOT
        || rr_obligations_complete(obligation);
    result.logical_custody_restored = inverse_complete;
    result.allocator_restored =
        rc_allocator_is_restored(&runtime);
    result.boundary_block_copy_calls =
        logical.boundary_block_copy_calls;
    result.intermediate_block_copy_calls =
        logical.intermediate_block_copy_calls;
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

static int rr_execution_exact(
    const struct dc_compiled *compiled,
    const struct rr_execution *execution
) {
    const size_t s = gm_find_public(compiled, 805U);
    const size_t t = gm_find_public(compiled, 806U);
    const size_t u = gm_find_public(compiled, 807U);
    const size_t v = gm_find_public(compiled, 808U);
    return execution->working_slots == 9U
        && execution->maximum_live == 9U
        && execution->projection_live == 6U
        && execution->forward_actions == 28U
        && execution->reverse_actions == 28U
        && execution->forward_cursor == 28U
        && execution->reverse_cursor == 28U
        && execution->lease_allocations == 28U
        && execution->lease_releases == 28U
        && execution->forward_operations == 20U
        && execution->inverse_operations == 20U
        && execution->forward_leaf_encodes == 8U
        && execution->inverse_leaf_encodes == 8U
        && execution->physical_forward_receipts == 40U
        && execution->physical_inverse_receipts == 40U
        && execution->multi_activation_edges == 10U
        && execution->second_or_later_activations == 18U
        && execution->exact_receipts == 29U
        && execution->leaf_rebind_receipts == 4U
        && execution->operator_rebind_receipts == 7U
        && execution->maximum_activation_ordinal == 3U
        && execution->nested_frame_depth == 2U
        && execution->activation_chain_depth == 3U
        && execution->shared_forward_activations == 28U
        && execution->shared_inverse_activations == 28U
        && execution->owner[s].forward_activations == 8U
        && execution->owner[s].inverse_activations == 8U
        && execution->owner[t].forward_activations == 6U
        && execution->owner[t].inverse_activations == 6U
        && execution->owner[u].forward_activations == 10U
        && execution->owner[u].inverse_activations == 10U
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
        && execution->operator_reconstructions == 9U
        && execution->operator_suspensions == 4U
        && execution->persistent_obligations_created == 9U
        && execution->persistent_obligations_cleared == 9U
        && execution->receipts_restored
        && execution->nodes_restored
        && execution->obligations_restored
        && execution->logical_custody_restored
        && execution->allocator_restored
        && execution->workspace_cleared
        && !execution->snapshot_loaded
        && execution->boundary_block_copy_calls == 2U
        && execution->intermediate_block_copy_calls == 0U
        && execution->serial_after
            == execution->serial_before + 28U
        && execution->restoration_generation_after
            == execution->restoration_generation_before + 1U
        && execution->restoration_max_abs
            <= GA_WORKSPACE_TOLERANCE;
}

static const char *rr_policy_name(
    enum rr_receipt_policy policy
) {
    if (policy == RR_RECEIPT_EXACT) {
        return "EXACT";
    }
    if (policy == RR_RECEIPT_LEAF_REBIND) {
        return "PUBLIC_LEAF_REBIND";
    }
    if (policy == RR_RECEIPT_OPERATOR_REBIND) {
        return "INTERNAL_OPERATOR_REBIND";
    }
    fail("recursive rematerialization receipt policy invalid");
    return "INVALID";
}

static void rr_print_receipt_plan(
    const struct rr_plan *plan,
    const struct dc_compiled *compiled
) {
    printf(",\"activation_edge_plans\":[");
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        if (edge > 0U) {
            putchar(',');
        }
        printf(
            "{\"producer_public_id\":%u,"
            "\"consumer_public_id\":%u,"
            "\"side\":%u,"
            "\"activation_limit\":%zu,"
            "\"receipts\":[",
            compiled->graph.node[
                compiled->edge[edge].producer
            ].id,
            compiled->graph.node[
                compiled->edge[edge].consumer
            ].id,
            compiled->edge[edge].side,
            plan->edge[edge].activation_limit
        );
        for (
            size_t ordinal = 0U;
            ordinal < plan->edge[edge].activation_limit;
            ++ordinal
        ) {
            if (ordinal > 0U) {
                putchar(',');
            }
            const struct rr_receipt_plan *receipt =
                &plan->edge[edge].receipt[ordinal];
            printf(
                "{\"ordinal\":%zu,"
                "\"policy\":\"%s\","
                "\"consumer_activation\":%llu,"
                "\"forward_producer_activation\":%llu,"
                "\"inverse_producer_activation\":%llu,"
                "\"forward_action\":%zu,"
                "\"inverse_action\":%zu}",
                ordinal,
                rr_policy_name(receipt->policy),
                (unsigned long long)
                    receipt->consumer_generation,
                (unsigned long long)
                    receipt->forward_producer_generation,
                (unsigned long long)
                    receipt->inverse_producer_generation,
                receipt->forward_action,
                receipt->inverse_action
            );
        }
        printf("]}");
    }
    putchar(']');
}

static void rr_print(
    const char *mode,
    const struct dc_compiled *compiled,
    const struct rr_plan *plan,
    const struct rr_execution *execution
) {
    const size_t carrier_cells =
        rc_carrier_cells(execution->working_slots);
    const size_t workspace_cells =
        GA_CARRIER_CELLS - GA_RELATION_BLOCKS * GA_BLOCK_CELLS;
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"" RR_CLAIM "\","
        "\"claim_ceiling\":\"" RR_CEILING "\","
        "\"width\":%u,"
        "\"manifest_nodes\":%zu,"
        "\"manifest_leaves\":%zu,"
        "\"compose_nodes\":%zu,"
        "\"intersect_nodes\":%zu,"
        "\"dag_edges\":%zu,"
        "\"dag_depth\":%zu,"
        "\"fanout_nodes\":%zu,"
        "\"compiler_generated_forward_and_literal_reverse_tapes\":true,"
        "\"coefficient_oblivious_plan\":true,"
        "\"rank_selection\":\"MAX_RANK_THEN_PUBLIC_ID_ASC\","
        "\"selected_rank2_public_id\":%u,"
        "\"nested_rematerialization_frame_depth\":%zu,"
        "\"activation_chain_depth\":%zu,"
        "\"maximum_activation_ordinal\":%zu,"
        "\"bound_topology_fnv1a64\":\"%016llx\","
        "\"bound_schedule_fnv1a64\":\"%016llx\","
        "\"plan_fnv1a64\":\"%016llx\","
        "\"forward_actions\":%zu,"
        "\"reverse_actions\":%zu,"
        "\"forward_cursor\":%zu,"
        "\"reverse_cursor\":%zu,"
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
        "\"activation_plan_bytes_current_abi\":%zu,"
        "\"activation_runtime_bytes_current_abi\":%zu,"
        "\"node_generation_state_bytes_current_abi\":%zu,"
        "\"obligation_storage_bytes_current_abi\":%zu,"
        "\"node_lease_storage_bytes_current_abi\":%zu,"
        "\"execution_summary_bytes_current_abi\":%zu,"
        "\"machine_state_bytes_current_abi\":%zu,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"serialized_intermediate_coefficients\":0,"
        "\"intermediate_relation_hashes\":0,"
        "\"pre_inverse_carrier_aggregate_outputs\":0,"
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
        plan->selected_public_id,
        execution->nested_frame_depth,
        execution->activation_chain_depth,
        execution->maximum_activation_ordinal,
        (unsigned long long)plan->topology_hash,
        (unsigned long long)plan->schedule_hash,
        (unsigned long long)execution->plan_hash,
        execution->forward_actions,
        execution->reverse_actions,
        execution->forward_cursor,
        execution->reverse_cursor,
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
        sizeof(struct rr_plan),
        sizeof(struct rr_edge_runtime) * (size_t)DC_MAX_EDGES,
        sizeof(struct rr_node_runtime) * (size_t)RC_MAX_NODES,
        sizeof(struct rr_obligation) * (size_t)RC_MAX_NODES,
        sizeof(struct rc_lease) * (size_t)RC_MAX_NODES,
        sizeof(struct rr_execution),
        sizeof(struct rr_machine)
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
        "\"receipts_restored\":%s,"
        "\"nodes_restored\":%s,"
        "\"obligations_restored\":%s,"
        "\"snapshot_loaded\":%s,"
        "\"lease_allocations\":%llu,"
        "\"lease_releases\":%llu,"
        "\"forward_native_operations\":%llu,"
        "\"inverse_native_operations\":%llu,"
        "\"native_operation_calls\":%llu,"
        "\"forward_leaf_encodes\":%llu,"
        "\"inverse_leaf_encodes\":%llu,"
        "\"leaf_encode_calls\":%llu,"
        "\"operator_reconstructions\":%llu,"
        "\"operator_suspensions\":%llu,"
        "\"persistent_obligations_created\":%llu,"
        "\"persistent_obligations_cleared\":%llu,"
        "\"physical_edge_forward_activations\":%llu,"
        "\"physical_edge_inverse_activations\":%llu,"
        "\"multi_activation_edges\":%zu,"
        "\"second_or_later_activations\":%zu,"
        "\"exact_activation_receipts\":%zu,"
        "\"leaf_rebind_receipts\":%zu,"
        "\"operator_rebind_receipts\":%zu,"
        "\"shared_forward_activations\":%llu,"
        "\"shared_inverse_activations\":%llu,"
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
        execution->receipts_restored ? "true" : "false",
        execution->nodes_restored ? "true" : "false",
        execution->obligations_restored ? "true" : "false",
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
        (unsigned long long)execution->operator_reconstructions,
        (unsigned long long)execution->operator_suspensions,
        (unsigned long long)
            execution->persistent_obligations_created,
        (unsigned long long)
            execution->persistent_obligations_cleared,
        (unsigned long long)
            execution->physical_forward_receipts,
        (unsigned long long)
            execution->physical_inverse_receipts,
        execution->multi_activation_edges,
        execution->second_or_later_activations,
        execution->exact_receipts,
        execution->leaf_rebind_receipts,
        execution->operator_rebind_receipts,
        (unsigned long long)
            execution->shared_forward_activations,
        (unsigned long long)
            execution->shared_inverse_activations,
        (unsigned long long)
            execution->boundary_block_copy_calls,
        (unsigned long long)
            execution->intermediate_block_copy_calls,
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
    printf(",\"shared_owner_activation_receipts\":[");
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
    rr_print_receipt_plan(plan, compiled);
    printf("}\n");
}

static enum rr_fault rr_fault_from_option(const char *option) {
    if (strcmp(option, "--control-deep-stale-consumer") == 0) {
        return RR_FAULT_DEEP_STALE_CONSUMER;
    }
    if (strcmp(option, "--control-deep-stale-producer") == 0) {
        return RR_FAULT_DEEP_STALE_PRODUCER;
    }
    if (strcmp(option, "--control-rebind-swap") == 0) {
        return RR_FAULT_REBIND_SWAP;
    }
    if (strcmp(option, "--control-activation-reuse") == 0) {
        return RR_FAULT_ACTIVATION_REUSE;
    }
    if (strcmp(option, "--control-missing-nested-close") == 0) {
        return RR_FAULT_MISSING_NESTED_CLOSE;
    }
    if (strcmp(option, "--control-missing-nested-child") == 0) {
        return RR_FAULT_MISSING_NESTED_CHILD;
    }
    if (strcmp(option, "--control-tamper-tape") == 0) {
        return RR_FAULT_TAMPER_TAPE;
    }
    return RR_FAULT_NONE;
}

#ifndef RR_PUBLIC_MAIN
#define RR_PUBLIC_MAIN main
#endif
int RR_PUBLIC_MAIN(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fail(
            "usage: recursive_rematerializing_general_multi_dag_"
            "affine_phase MANIFEST [CONTROL]"
        );
    }
    if (
        argc == 3
        && (
            strcmp(argv[2], "--project-intermediate") == 0
            || strcmp(argv[2], "--project-forward-tape") == 0
            || strcmp(argv[2], "--project-reverse-tape") == 0
            || strcmp(argv[2], "--project-obligations") == 0
            || strcmp(argv[2], "--project-node-generations") == 0
            || strcmp(argv[2], "--project-activation-receipts")
                == 0
            || strcmp(argv[2], "--debug-dump") == 0
        )
    ) {
        fail("recursive rematerialization projection denied");
    }
    if (argc == 3 && strcmp(argv[2], "--null-carrier") == 0) {
        fail("null recursive rematerialization carrier");
    }

    struct rc_manifest manifest = {0};
    rc_load_manifest(argv[1], &manifest);
    const struct dc_compiled compiled = dc_compile(&manifest, 4U);
    gm_require_topology(&compiled);
    const struct rr_plan plan = rr_compile_plan(&compiled);
    struct rc_program program[GM_VARIANTS];
    for (size_t variant = 0U; variant < GM_VARIANTS; ++variant) {
        program[variant] = gm_make_program(variant);
    }
    const struct process shape = {
        .carrier_cells = rc_carrier_cells(RR_WORKING_SLOTS)
    };
    if (argc == 3) {
        const enum rr_fault fault = rr_fault_from_option(argv[2]);
        if (fault == RR_FAULT_NONE) {
            fail("recursive rematerialization control unknown");
        }
        struct carrier control = make_carrier(&shape, 9323);
        struct rr_machine control_machine = {0};
        (void)rr_execute(
            &control,
            &compiled,
            &plan,
            &program[0],
            &control_machine,
            RC_RESTORE_CORRECT,
            fault
        );
        free_carrier(&control);
        fail("recursive rematerialization fault control escaped");
    }

    struct carrier carrier = make_carrier(&shape, 9323);
    struct rr_machine machine = {0};
    struct rr_execution semantic[RR_RUN_VARIANTS];
    double repeated_max_abs = 0.0;
    for (size_t variant = 0U; variant < RR_RUN_VARIANTS; ++variant) {
        semantic[variant] = rr_execute(
            &carrier,
            &compiled,
            &plan,
            &program[variant],
            &machine,
            RC_RESTORE_CORRECT,
            RR_FAULT_NONE
        );
        if (!rr_execution_exact(&compiled, &semantic[variant])) {
            free_carrier(&carrier);
            fail("recursive rematerialization accepted law failed");
        }
        if (
            semantic[variant].restoration_max_abs
            > repeated_max_abs
        ) {
            repeated_max_abs =
                semantic[variant].restoration_max_abs;
        }
    }
#if RR_REUSE_CYCLES > 0U
    for (size_t cycle = 0U; cycle < RR_REUSE_CYCLES; ++cycle) {
        const struct rr_execution repeated = rr_execute(
            &carrier,
            &compiled,
            &plan,
            &program[cycle % 2U],
            &machine,
            RC_RESTORE_CORRECT,
            RR_FAULT_NONE
        );
        if (!rr_execution_exact(&compiled, &repeated)) {
            free_carrier(&carrier);
            fail("recursive rematerialization reuse law failed");
        }
        if (repeated.restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = repeated.restoration_max_abs;
        }
    }
#endif
    free_carrier(&carrier);
    for (size_t variant = 0U; variant < RR_RUN_VARIANTS; ++variant) {
        const char *mode = variant == 0U
            ? "recursive-remat-primary"
            : variant == 1U
                ? "recursive-remat-reuse"
                : "recursive-remat-semantic";
        rr_print(
            mode,
            &compiled,
            &plan,
            &semantic[variant]
        );
    }
    if (RR_SCALE_ONLY) {
        return 0;
    }

    struct rr_machine control_machine = {0};
    carrier = make_carrier(&shape, 9323);
    const struct rr_execution wrong = rr_execute(
        &carrier,
        &compiled,
        &plan,
        &program[0],
        &control_machine,
        RC_RESTORE_WRONG_ROOT,
        RR_FAULT_NONE
    );
    free_carrier(&carrier);
    control_machine = (struct rr_machine){0};
    carrier = make_carrier(&shape, 9323);
    const struct rr_execution missing = rr_execute(
        &carrier,
        &compiled,
        &plan,
        &program[0],
        &control_machine,
        RC_RESTORE_MISSING_ROOT,
        RR_FAULT_NONE
    );
    free_carrier(&carrier);
    control_machine = (struct rr_machine){0};
    carrier = make_carrier(&shape, 9323);
    const struct rr_execution snapshot = rr_execute(
        &carrier,
        &compiled,
        &plan,
        &program[0],
        &control_machine,
        RC_RESTORE_SNAPSHOT,
        RR_FAULT_NONE
    );
    free_carrier(&carrier);
    rr_print(
        "recursive-remat-snapshot-baseline",
        &compiled,
        &plan,
        &snapshot
    );
    printf(
        "{\"mode\":\"recursive-remat-controls\","
        "\"plan_fnv1a64\":\"%016llx\","
        "\"primary_reuse_boundary_different\":%s,"
        "\"wrong_root_inverse_detected\":%s,"
        "\"missing_root_inverse_detected\":%s,"
        "\"snapshot_separate_weaker_path\":true,"
        "\"snapshot_post_projection_inverse_operations\":0,"
        "\"accepted_working_slots\":9,"
        "\"one_layer_working_slots\":8,"
        "\"leaf_only_working_slots\":11,"
        "\"retain_all_working_slots\":15,"
        "\"occurrence_working_slots\":51,"
        "\"accepted_native_operation_calls\":40,"
        "\"accepted_leaf_encode_calls\":16,"
        "\"accepted_lease_allocations\":28,"
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
        (unsigned)(RR_RUN_VARIANTS + RR_REUSE_CYCLES),
        repeated_max_abs,
        wrong.restoration_max_abs,
        missing.restoration_max_abs
    );
    return 0;
}
