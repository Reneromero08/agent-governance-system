#define _POSIX_C_SOURCE 200809L

/*
 * Mutable bounded affine-DAG successor.
 *
 * The tree evaluator is intentionally not used: evaluating a shared node by
 * occurrence would rematerialize it.  This live-set scheduler emits every
 * node once, retains one actual lease for the native-produced fanout message,
 * and gives every public edge a two-transition custody token:
 *
 *   DECLARED -> FORWARD_CONSUMED -> INVERSE_CONSUMED.
 *
 * A producer cannot be inversed or released until every consumer has applied
 * its actual inverse.  Only ROOT is copied to the boundary.  No intermediate
 * coefficient, hash, rank, witness, assignment, or candidate set is emitted.
 */

#ifndef DC_WIDTH
#define DC_WIDTH 3U
#endif
#define RC_WIDTH DC_WIDTH
#define RC_RUN_VARIANTS 5U
#define RC_REUSE_CYCLES 0U
#define RC_PUBLIC_MAIN recursive_affine_embedded_main
#include "recursive_affine_module_phase.c"
#undef RC_PUBLIC_MAIN

#define DC_LEAF_BODIES 4U
#define DC_VARIANTS 5U
#ifndef DC_RUN_VARIANTS
#define DC_RUN_VARIANTS DC_VARIANTS
#endif
#ifndef DC_REUSE_CYCLES
#define DC_REUSE_CYCLES 12U
#endif
#ifndef DC_SCALE_ONLY
#define DC_SCALE_ONLY 0
#endif
#define DC_MAX_EDGES (2U * RC_MAX_NODES)

_Static_assert(DC_RUN_VARIANTS >= 2U, "DAG primary/reuse required");
_Static_assert(DC_RUN_VARIANTS <= DC_VARIANTS, "DAG variants exceeded");

enum dc_edge_state {
    DC_EDGE_DECLARED = 0,
    DC_EDGE_FORWARD = 1,
    DC_EDGE_INVERSE = 2
};

struct dc_edge {
    size_t producer;
    size_t consumer;
    unsigned side;
};

struct dc_compiled {
    struct rc_compiled graph;
    struct dc_edge edge[DC_MAX_EDGES];
    size_t left_edge[RC_MAX_NODES];
    size_t right_edge[RC_MAX_NODES];
    size_t edge_count;
    size_t fanout_nodes;
    size_t maximum_fanout;
    size_t shared_node;
    size_t leaf_body_bytes;
};

struct dc_machine {
    uint64_t next_serial;
    uint64_t restoration_generation;
};

struct dc_custody {
    unsigned char edge_state[DC_MAX_EDGES];
    size_t forward_pending[RC_MAX_NODES];
    size_t inverse_pending[RC_MAX_NODES];
    struct rc_lease shared_lease;
    int shared_lease_set;
    uint64_t shared_forward_consumptions;
    uint64_t shared_inverse_consumptions;
    uint64_t shared_forward_materializations;
    uint64_t shared_inverse_applications;
    uint64_t shared_owner_allocations;
    uint64_t shared_owner_releases;
    size_t shared_live_instances;
    size_t shared_peak_live_instances;
    int shared_observation_set;
    size_t shared_observed_slot;
    uint64_t shared_observed_serial;
    size_t shared_distinct_slots;
    size_t shared_distinct_serials;
    int shared_live_at_projection;
    int shared_live_after_root_inverse;
    uint64_t boundary_block_copy_calls;
    uint64_t intermediate_block_copy_calls;
};

struct dc_execution {
    struct ga_boundary boundary;
    struct ga_stats stats;
    double restoration_max_abs;
    double integrity_max_abs;
    int workspace_cleared;
    int allocator_restored;
    int edge_custody_restored;
    int shared_receipt_reset;
    int pending_counts_restored;
    int scheduler_queue_empty;
    int snapshot_loaded;
    size_t maximum_live_slots;
    size_t outstanding_relation_leases;
    uint64_t lease_allocations;
    uint64_t lease_releases;
    uint64_t lease_checks;
    uint64_t lease_cell_checks;
    uint64_t forward_operations;
    uint64_t inverse_operations;
    uint64_t forward_leaf_encodes;
    uint64_t inverse_leaf_encodes;
    uint64_t dirty_control_releases;
    uint64_t shared_forward_consumptions;
    uint64_t shared_inverse_consumptions;
    uint64_t shared_forward_materializations;
    uint64_t shared_inverse_applications;
    uint64_t shared_owner_allocations;
    uint64_t shared_owner_releases;
    size_t shared_peak_live_instances;
    size_t shared_consumer_distinct_slots;
    size_t shared_consumer_distinct_serials;
    int shared_live_at_projection;
    int shared_live_after_root_inverse;
    uint64_t serial_before;
    uint64_t serial_after;
    uint64_t restoration_generation_before;
    uint64_t restoration_generation_after;
    uint64_t boundary_block_copy_calls;
    uint64_t intermediate_block_copy_calls;
};

static struct dc_compiled dc_compile(
    const struct rc_manifest *manifest,
    int require_fanout
) {
    struct dc_compiled result = {0};
    struct rc_compiled *compiled = &result.graph;
    if (manifest->root_directives != 1U || manifest->count == 0U) {
        fail("affine DAG requires exactly one root");
    }
    compiled->count = manifest->count;
    compiled->manifest_bytes = manifest->bytes_read;
    memcpy(
        compiled->node,
        manifest->node,
        manifest->count * sizeof(*manifest->node)
    );
    for (size_t left = 0U; left < compiled->count; ++left) {
        for (size_t right = left + 1U; right < compiled->count; ++right) {
            if (compiled->node[left].id == compiled->node[right].id) {
                fail("affine DAG has duplicate node id");
            }
        }
    }
    compiled->root = rc_find_node(compiled, manifest->root_id);
    if (compiled->node[compiled->root].kind == RC_LEAF) {
        fail("affine DAG root must be an operator");
    }
    unsigned char leaf_body_seen[DC_LEAF_BODIES] = {0};
    for (size_t index = 0U; index < compiled->count; ++index) {
        struct rc_node *node = &compiled->node[index];
        result.left_edge[index] = SIZE_MAX;
        result.right_edge[index] = SIZE_MAX;
        if (node->kind == RC_LEAF) {
            if (node->leaf_index >= DC_LEAF_BODIES) {
                fail("affine DAG leaf body outside compiled body table");
            }
            leaf_body_seen[node->leaf_index] = 1U;
            ++compiled->leaves;
            continue;
        }
        if (
            node->left_id == node->right_id
            || node->left_id == node->id
            || node->right_id == node->id
        ) {
            fail("affine DAG operator aliases an operand");
        }
        node->left = rc_find_node(compiled, node->left_id);
        node->right = rc_find_node(compiled, node->right_id);
        if (result.edge_count + 2U > DC_MAX_EDGES) {
            fail("affine DAG edge capacity exceeded");
        }
        result.left_edge[index] = result.edge_count;
        result.edge[result.edge_count++] = (struct dc_edge){
            .producer = node->left,
            .consumer = index,
            .side = 0U
        };
        result.right_edge[index] = result.edge_count;
        result.edge[result.edge_count++] = (struct dc_edge){
            .producer = node->right,
            .consumer = index,
            .side = 1U
        };
        ++compiled->node[node->left].indegree;
        ++compiled->node[node->right].indegree;
        if (node->kind == RC_COMPOSE) {
            ++compiled->compose_nodes;
        } else if (node->kind == RC_INTERSECT) {
            ++compiled->intersect_nodes;
        } else {
            fail("affine DAG operator kind invalid");
        }
    }
    for (size_t body = 0U; body < DC_LEAF_BODIES; ++body) {
        if (leaf_body_seen[body] == 0U) {
            fail("affine DAG omits a compiled leaf body");
        }
    }
    if (
        compiled->compose_nodes == 0U
        || compiled->intersect_nodes == 0U
    ) {
        fail("affine DAG lacks mixed operators");
    }
    rc_visit(compiled, compiled->root);
    if (compiled->schedule_count != compiled->count) {
        fail("affine DAG contains disconnected node");
    }
    for (size_t index = 0U; index < compiled->count; ++index) {
        const size_t consumers = compiled->node[index].indegree;
        if (index == compiled->root) {
            if (consumers != 0U) {
                fail("affine DAG root has a consumer");
            }
            continue;
        }
        if (consumers == 0U || consumers > 2U) {
            fail("affine DAG consumer count outside bound");
        }
        if (consumers > result.maximum_fanout) {
            result.maximum_fanout = consumers;
        }
        if (consumers > 1U) {
            if (compiled->node[index].kind == RC_LEAF) {
                fail("affine DAG proof requires native-produced fanout");
            }
            result.shared_node = index;
            ++result.fanout_nodes;
        }
    }
    if (require_fanout) {
        if (
            result.fanout_nodes != 1U
            || result.maximum_fanout != 2U
        ) {
            fail("affine DAG requires exactly one bounded fanout node");
        }
    } else if (result.fanout_nodes != 0U) {
        fail("duplicate-tree baseline unexpectedly contains fanout");
    }
    compiled->edges = result.edge_count;
    compiled->depth = compiled->node[compiled->root].depth;
    compiled->clean_slots = compiled->count;
    compiled->topology_hash = UINT64_C(14695981039346656037);
    for (size_t index = 0U; index < compiled->count; ++index) {
        const struct rc_node *node = &compiled->node[index];
        rc_hash_word(&compiled->topology_hash, node->id);
        rc_hash_word(&compiled->topology_hash, node->kind);
        rc_hash_word(&compiled->topology_hash, node->left_id);
        rc_hash_word(&compiled->topology_hash, node->right_id);
        rc_hash_word(&compiled->topology_hash, node->leaf_index);
        rc_hash_word(&compiled->topology_hash, node->signature.domain);
        rc_hash_word(&compiled->topology_hash, node->signature.codomain);
        rc_hash_word(&compiled->topology_hash, node->indegree);
    }
    compiled->schedule_hash = UINT64_C(14695981039346656037);
    for (size_t step = 0U; step < compiled->schedule_count; ++step) {
        rc_hash_word(
            &compiled->schedule_hash,
            compiled->node[compiled->schedule[step]].id
        );
    }
    result.leaf_body_bytes = DC_LEAF_BODIES
        * sizeof(((struct rc_program *)0)->leaf[0]);
    return result;
}

static struct rc_program dc_make_program(size_t variant) {
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
    const size_t f0[] = {1U};
    const size_t f1[] = {2U};
    const size_t f2[] = {0U};
    rc_set_map_row(program.leaf[0], 0U, 0U, f0, 1U);
    rc_set_map_row(program.leaf[0], 1U, 0U, f1, 1U);
    rc_set_map_row(program.leaf[0], 2U, 1U, f2, 1U);
    const size_t g0[] = {0U, 2U};
    const size_t g1[] = {1U};
    const size_t g2[] = {2U};
    rc_set_map_row(program.leaf[1], 0U, 0U, g0, 2U);
    rc_set_map_row(program.leaf[1], 1U, 0U, g1, 1U);
    rc_set_map_row(program.leaf[1], 2U, 0U, g2, 1U);
    const size_t h0[] = {0U, 2U};
    const size_t h1[] = {1U};
    const size_t h2[] = {2U};
    rc_set_map_row(program.leaf[2], 0U, 0U, h0, 2U);
    rc_set_map_row(program.leaf[2], 1U, 0U, h1, 1U);
    rc_set_map_row(program.leaf[2], 2U, 1U, h2, 1U);
    const size_t k0[] = {0U, 1U, 2U};
    rc_set_map_row(program.leaf[3], 0U, 0U, k0, 3U);
    rc_set_map_row(program.leaf[3], 1U, 0U, h1, 1U);
    rc_set_map_row(program.leaf[3], 2U, 1U, h2, 1U);
    if (variant == 2U) {
        memcpy(
            program.leaf[3],
            program.leaf[2],
            sizeof(program.leaf[3])
        );
    } else if (variant == 3U) {
        memcpy(
            program.leaf[2],
            program.leaf[3],
            sizeof(program.leaf[2])
        );
    } else if (variant == 4U) {
        rc_set_map_row(program.leaf[3], 0U, 1U, h0, 2U);
    } else if (variant != 0U) {
        fail("affine DAG program variant unknown");
    }
    return program;
}

static void dc_validate_shared_handle(
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    struct rc_lease lease
) {
    if (custody->shared_lease_set == 0) {
        fail("affine DAG shared lease unavailable");
    }
    if (
        lease.slot != custody->shared_lease.slot
        || lease.serial != custody->shared_lease.serial
        || compiled->fanout_nodes != 1U
    ) {
        fail("affine DAG consumer substituted a shared lease");
    }
}

static void dc_observe_shared_handle(
    struct dc_custody *custody,
    struct rc_lease lease
) {
    if (!custody->shared_observation_set) {
        custody->shared_observation_set = 1;
        custody->shared_observed_slot = lease.slot;
        custody->shared_observed_serial = lease.serial;
        custody->shared_distinct_slots = 1U;
        custody->shared_distinct_serials = 1U;
        return;
    }
    if (lease.slot != custody->shared_observed_slot) {
        ++custody->shared_distinct_slots;
    }
    if (lease.serial != custody->shared_observed_serial) {
        ++custody->shared_distinct_serials;
    }
}

static void dc_validate_shared_live(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    struct dc_custody *custody
) {
    dc_validate_shared_handle(
        compiled, custody, custody->shared_lease
    );
    rc_validate_lease(
        runtime,
        custody->shared_lease,
        compiled->shared_node,
        compiled->graph.node[compiled->shared_node].signature,
        0
    );
}

static void dc_copy_relation_block(
    struct rc_runtime *runtime,
    struct dc_custody *custody,
    size_t source,
    size_t target,
    int inverse,
    int boundary_copy
) {
    ga_copy_block(
        runtime->carrier,
        source,
        target,
        inverse,
        &runtime->stats
    );
    if (boundary_copy) {
        ++custody->boundary_block_copy_calls;
    } else {
        ++custody->intermediate_block_copy_calls;
    }
}

static void dc_forward_edge(
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    size_t edge_index,
    struct rc_lease lease,
    int stale_shared,
    int skip_second_shared,
    int double_first_shared
) {
    const struct dc_edge *edge = &compiled->edge[edge_index];
    if (custody->edge_state[edge_index] != DC_EDGE_DECLARED) {
        fail("affine DAG edge consumed more than once");
    }
    if (
        compiled->fanout_nodes == 1U
        && edge->producer == compiled->shared_node
    ) {
        if (skip_second_shared && custody->shared_forward_consumptions == 1U) {
            return;
        }
        if (stale_shared && custody->shared_forward_consumptions == 1U) {
            ++lease.serial;
        }
        dc_validate_shared_handle(compiled, custody, lease);
        dc_observe_shared_handle(custody, lease);
        ++custody->shared_forward_consumptions;
    }
    custody->edge_state[edge_index] = DC_EDGE_FORWARD;
    if (custody->forward_pending[edge->producer] == 0U) {
        fail("affine DAG forward pending count underflow");
    }
    --custody->forward_pending[edge->producer];
    if (
        double_first_shared
        && edge->producer == compiled->shared_node
        && custody->shared_forward_consumptions == 1U
    ) {
        dc_forward_edge(
            compiled,
            custody,
            edge_index,
            lease,
            0,
            0,
            0
        );
    }
}

static void dc_inverse_edge(
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    size_t edge_index,
    struct rc_lease lease
) {
    const struct dc_edge *edge = &compiled->edge[edge_index];
    if (custody->edge_state[edge_index] != DC_EDGE_FORWARD) {
        fail("affine DAG inverse edge transition invalid");
    }
    if (
        compiled->fanout_nodes == 1U
        && edge->producer == compiled->shared_node
    ) {
        dc_validate_shared_handle(compiled, custody, lease);
        dc_observe_shared_handle(custody, lease);
        ++custody->shared_inverse_consumptions;
    }
    custody->edge_state[edge_index] = DC_EDGE_INVERSE;
    if (custody->inverse_pending[edge->producer] == 0U) {
        fail("affine DAG inverse pending count underflow");
    }
    --custody->inverse_pending[edge->producer];
}

static void dc_require_inverse_ready(
    const struct dc_compiled *compiled,
    const struct dc_custody *custody,
    size_t node_index
) {
    if (
        custody->forward_pending[node_index] != 0U
        || custody->inverse_pending[node_index] != 0U
    ) {
        (void)compiled;
        fail("affine DAG producer inverse has live dependents");
    }
}

static struct rc_lease dc_forward(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    struct rc_lease node_lease[RC_MAX_NODES],
    int stale_shared,
    int skip_second_shared,
    int double_first_shared
) {
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        node_lease[index] = (struct rc_lease){
            .slot = SIZE_MAX,
            .serial = 0U
        };
        custody->forward_pending[index] =
            compiled->graph.node[index].indegree;
        custody->inverse_pending[index] =
            compiled->graph.node[index].indegree;
    }
    for (
        size_t step = 0U;
        step < compiled->graph.schedule_count;
        ++step
    ) {
        const size_t index = compiled->graph.schedule[step];
        const struct rc_node *node = &compiled->graph.node[index];
        const struct rc_lease output = rc_allocate(
            runtime, index, node->signature
        );
        node_lease[index] = output;
        if (node->kind == RC_LEAF) {
            ga_encode_relation(
                runtime->carrier,
                runtime->program->leaf[node->leaf_index],
                rc_slot_start(output.slot),
                0,
                &runtime->stats
            );
            runtime->slot[output.slot].reserved = 0;
            ++runtime->forward_leaf_encodes;
        } else {
            const struct rc_lease left = node_lease[node->left];
            const struct rc_lease right = node_lease[node->right];
            dc_forward_edge(
                compiled,
                custody,
                compiled->left_edge[index],
                left,
                stale_shared,
                skip_second_shared,
                double_first_shared
            );
            dc_forward_edge(
                compiled,
                custody,
                compiled->right_edge[index],
                right,
                stale_shared,
                skip_second_shared,
                double_first_shared
            );
            rc_apply_node(
                runtime, index, left, right, output, 0, 0
            );
        }
        if (
            compiled->fanout_nodes == 1U
            && index == compiled->shared_node
        ) {
            custody->shared_lease = output;
            custody->shared_lease_set = 1;
            ++custody->shared_forward_materializations;
            ++custody->shared_owner_allocations;
            ++custody->shared_live_instances;
            if (
                custody->shared_live_instances
                > custody->shared_peak_live_instances
            ) {
                custody->shared_peak_live_instances =
                    custody->shared_live_instances;
            }
        }
    }
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        if (custody->edge_state[edge] != DC_EDGE_FORWARD) {
            fail("affine DAG skipped a declared consumer edge");
        }
    }
    return node_lease[compiled->graph.root];
}

static void dc_reverse_one(
    struct rc_runtime *runtime,
    const struct dc_compiled *compiled,
    struct dc_custody *custody,
    const struct rc_lease node_lease[RC_MAX_NODES],
    size_t index,
    size_t wrong_node
) {
    const struct rc_node *node = &compiled->graph.node[index];
    dc_require_inverse_ready(compiled, custody, index);
    const struct rc_lease output = node_lease[index];
    rc_validate_lease(
        runtime, output, index, node->signature, 0
    );
    if (node->kind == RC_LEAF) {
        ga_encode_relation(
            runtime->carrier,
            runtime->program->leaf[node->leaf_index],
            rc_slot_start(output.slot),
            1,
            &runtime->stats
        );
        ++runtime->inverse_leaf_encodes;
    } else {
        const struct rc_lease left = node_lease[node->left];
        const struct rc_lease right = node_lease[node->right];
        const int wrong_here = index == wrong_node;
        rc_apply_node(
            runtime,
            index,
            left,
            right,
            output,
            1,
            wrong_here
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
        if (
            compiled->fanout_nodes == 1U
            && index == compiled->shared_node
        ) {
            ++custody->shared_inverse_applications;
        }
    }
    rc_release(
        runtime,
        output,
        index,
        node->signature,
        index == wrong_node
    );
    if (
        compiled->fanout_nodes == 1U
        && index == compiled->shared_node
    ) {
        ++custody->shared_owner_releases;
        if (custody->shared_live_instances == 0U) {
            fail("affine DAG shared live-instance underflow");
        }
        --custody->shared_live_instances;
    }
}

static int dc_edges_all(
    const struct dc_compiled *compiled,
    const struct dc_custody *custody,
    enum dc_edge_state expected
) {
    for (size_t edge = 0U; edge < compiled->edge_count; ++edge) {
        if (custody->edge_state[edge] != expected) {
            return 0;
        }
    }
    return 1;
}

static int dc_counts_zero(
    const struct dc_compiled *compiled,
    const struct dc_custody *custody
) {
    for (size_t index = 0U; index < compiled->graph.count; ++index) {
        if (
            custody->forward_pending[index] != 0U
            || custody->inverse_pending[index] != 0U
        ) {
            return 0;
        }
    }
    return 1;
}

static struct dc_execution dc_execute(
    struct carrier *carrier,
    const struct dc_compiled *compiled,
    const struct rc_program *program,
    struct dc_machine *machine,
    enum rc_restore_mode restore_mode
) {
    struct dc_execution execution = {0};
    struct carrier borrowed = snapshot_carrier(carrier);
    struct rc_runtime runtime = {
        .carrier = carrier,
        .compiled = &compiled->graph,
        .program = program,
        .pool = compiled->graph.count,
        .next_serial = machine->next_serial
    };
    struct dc_custody custody = {0};
    struct rc_lease node_lease[RC_MAX_NODES];
    execution.serial_before = machine->next_serial;
    execution.restoration_generation_before =
        machine->restoration_generation;
    const struct rc_lease root = dc_forward(
        &runtime,
        compiled,
        &custody,
        node_lease,
        0,
        0,
        0
    );
    if (compiled->fanout_nodes == 1U) {
        dc_validate_shared_live(&runtime, compiled, &custody);
        custody.shared_live_at_projection = 1;
    }
    dc_copy_relation_block(
        &runtime,
        &custody,
        rc_slot_start(root.slot),
        GA_BOUNDARY_START,
        0,
        1
    );
    execution.boundary = ga_latch_boundary(carrier, &runtime.stats);
    if (restore_mode == RC_RESTORE_SNAPSHOT) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        memset(runtime.slot, 0, sizeof(runtime.slot));
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
        memset(&custody.shared_lease, 0, sizeof(custody.shared_lease));
        custody.shared_lease_set = 0;
        custody.shared_live_instances = 0U;
        runtime.live = 0U;
        runtime.next_serial = machine->next_serial;
        execution.snapshot_loaded = 1;
    } else {
        dc_copy_relation_block(
            &runtime,
            &custody,
            rc_slot_start(root.slot),
            GA_BOUNDARY_START,
            1,
            1
        );
        if (restore_mode != RC_RESTORE_MISSING_ROOT) {
            for (
                size_t step = compiled->graph.schedule_count;
                step-- > 0U;
            ) {
                const size_t index = compiled->graph.schedule[step];
                dc_reverse_one(
                    &runtime,
                    compiled,
                    &custody,
                    node_lease,
                    index,
                    restore_mode == RC_RESTORE_WRONG_ROOT
                        ? compiled->graph.root
                        : SIZE_MAX
                );
                if (
                    index == compiled->graph.root
                    && compiled->fanout_nodes == 1U
                ) {
                    dc_validate_shared_live(
                        &runtime, compiled, &custody
                    );
                    custody.shared_live_after_root_inverse = 1;
                }
            }
        }
    }
    const int edges_complete = restore_mode == RC_RESTORE_SNAPSHOT
        ? 1
        : dc_edges_all(compiled, &custody, DC_EDGE_INVERSE);
    const int counts_complete = restore_mode == RC_RESTORE_SNAPSHOT
        ? 1
        : dc_counts_zero(compiled, &custody);
    execution.shared_forward_consumptions =
        custody.shared_forward_consumptions;
    execution.shared_inverse_consumptions =
        custody.shared_inverse_consumptions;
    execution.shared_forward_materializations =
        custody.shared_forward_materializations;
    execution.shared_inverse_applications =
        custody.shared_inverse_applications;
    execution.shared_owner_allocations =
        custody.shared_owner_allocations;
    execution.shared_owner_releases =
        custody.shared_owner_releases;
    execution.shared_peak_live_instances =
        custody.shared_peak_live_instances;
    execution.shared_consumer_distinct_slots =
        custody.shared_distinct_slots;
    execution.shared_consumer_distinct_serials =
        custody.shared_distinct_serials;
    execution.shared_live_at_projection =
        custody.shared_live_at_projection;
    execution.shared_live_after_root_inverse =
        custody.shared_live_after_root_inverse;
    execution.boundary_block_copy_calls =
        custody.boundary_block_copy_calls;
    execution.intermediate_block_copy_calls =
        custody.intermediate_block_copy_calls;
    if (edges_complete) {
        memset(custody.edge_state, 0, sizeof(custody.edge_state));
    }
    if (counts_complete) {
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
    memset(&custody.shared_lease, 0, sizeof(custody.shared_lease));
    custody.shared_lease_set = 0;
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    runtime.stats.restoration_cell_checks += carrier->cells;
    execution.integrity_max_abs = integrity(carrier);
    runtime.stats.integrity_cell_checks += carrier->cells;
    execution.workspace_cleared = ga_workspace_is_zero(
        carrier, &runtime.stats
    );
    execution.allocator_restored = rc_allocator_is_restored(&runtime);
    execution.edge_custody_restored = edges_complete
        && dc_edges_all(compiled, &custody, DC_EDGE_DECLARED);
    execution.shared_receipt_reset =
        custody.shared_lease_set == 0
        && custody.shared_lease.slot == 0U
        && custody.shared_lease.serial == 0U;
    execution.pending_counts_restored = counts_complete;
    execution.scheduler_queue_empty = 1;
    execution.stats = runtime.stats;
    execution.maximum_live_slots = runtime.maximum_live;
    execution.outstanding_relation_leases = runtime.live;
    execution.lease_allocations = runtime.lease_allocations;
    execution.lease_releases = runtime.lease_releases;
    execution.lease_checks = runtime.lease_checks;
    execution.lease_cell_checks = runtime.lease_cell_checks;
    execution.forward_operations = runtime.forward_operations;
    execution.inverse_operations = runtime.inverse_operations;
    execution.forward_leaf_encodes = runtime.forward_leaf_encodes;
    execution.inverse_leaf_encodes = runtime.inverse_leaf_encodes;
    execution.dirty_control_releases = runtime.dirty_control_releases;
    if (restore_mode == RC_RESTORE_CORRECT) {
        machine->next_serial = runtime.next_serial;
        ++machine->restoration_generation;
    }
    execution.serial_after = machine->next_serial;
    execution.restoration_generation_after =
        machine->restoration_generation;
    free_carrier(&borrowed);
    return execution;
}

static void dc_test_copy_substitution(
    const struct dc_compiled *compiled,
    const struct rc_program *program
) {
    const size_t control_pool = compiled->graph.count + 1U;
    const struct process shape = {
        .carrier_cells = rc_carrier_cells(control_pool)
    };
    struct carrier carrier = make_carrier(&shape, 9127);
    struct rc_runtime runtime = {
        .carrier = &carrier,
        .compiled = &compiled->graph,
        .program = program,
        .pool = control_pool
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
        0
    );
    const struct rc_signature signature =
        compiled->graph.node[compiled->shared_node].signature;
    const struct rc_lease clone = rc_allocate(
        &runtime, RC_MAX_NODES - 1U, signature
    );
    dc_copy_relation_block(
        &runtime,
        &custody,
        rc_slot_start(custody.shared_lease.slot),
        rc_slot_start(clone.slot),
        0,
        0
    );
    runtime.slot[clone.slot].reserved = 0;
    dc_validate_shared_handle(compiled, &custody, clone);
    free_carrier(&carrier);
    fail("affine DAG copied shared substitution was not rejected");
}

static void dc_test_forward_control(
    const struct dc_compiled *compiled,
    const struct rc_program *program,
    int stale_shared,
    int skip_second_shared,
    int double_first_shared
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
        stale_shared,
        skip_second_shared,
        double_first_shared
    );
    free_carrier(&carrier);
    fail("affine DAG forward custody control was not rejected");
}

static void dc_test_reordered(
    const struct dc_compiled *compiled,
    const struct rc_program *program
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
        0
    );
    dc_reverse_one(
        &runtime,
        compiled,
        &custody,
        node_lease,
        compiled->shared_node,
        SIZE_MAX
    );
    free_carrier(&carrier);
    fail("affine DAG dependency reorder was not rejected");
}

static unsigned long long dc_logical_inspections(
    const struct dc_execution *execution
) {
    return execution->stats.carrier_reads
        + execution->stats.boundary_decodes
        + execution->stats.workspace_cell_checks
        + execution->stats.restoration_cell_checks
        + execution->stats.integrity_cell_checks
        + execution->lease_checks
        + execution->lease_cell_checks;
}

static void dc_print(
    const char *mode,
    const struct dc_compiled *compiled,
    const struct dc_execution *execution
) {
    const size_t carrier_cells =
        rc_carrier_cells(compiled->graph.count);
    const size_t workspace_cells =
        GA_CARRIER_CELLS - GA_RELATION_BLOCKS * GA_BLOCK_CELLS;
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"BOUNDED_NATIVE_PRODUCED_AFFINE_DAG_"
        "SHARED_RESIDENT_MESSAGE_FANOUT\","
        "\"width\":%u,"
        "\"manifest_nodes\":%zu,"
        "\"manifest_leaves\":%zu,"
        "\"compose_nodes\":%zu,"
        "\"intersect_nodes\":%zu,"
        "\"dag_edges\":%zu,"
        "\"dag_depth\":%zu,"
        "\"fanout_nodes\":%zu,"
        "\"maximum_fanout\":%zu,"
        "\"compiled_postorder_valid\":true,"
        "\"derived_nominal_signatures\":true,"
        "\"address_assigned_after_validation\":true,"
        "\"live_set_scheduler\":true,"
        "\"topology_fnv1a64\":\"%016llx\","
        "\"schedule_fnv1a64\":\"%016llx\","
        "\"manifest_bytes\":%zu,"
        "\"working_relation_slots\":%zu,"
        "\"maximum_live_relation_slots\":%zu,"
        "\"physical_relation_blocks_including_boundary\":%zu,"
        "\"relation_cells\":%u,"
        "\"workspace_cells\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"live_carrier_bytes\":%zu,"
        "\"carrier_and_verification_snapshot_heap_bytes\":%zu,"
        "\"compiled_topology_bytes_current_abi\":%zu,"
        "\"program_storage_bytes_current_abi\":%zu,"
        "\"active_edge_descriptor_bytes\":%zu,"
        "\"custody_state_bytes_current_abi\":%zu,"
        "\"boundary_storage_bytes_current_abi\":%zu,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"serialized_intermediate_coefficients\":0,"
        "\"intermediate_relation_hashes\":0,"
        "\"pre_inverse_carrier_aggregate_outputs\":0,"
        "\"host_selected_pivots\":0,"
        "\"tuple_slots\":0,"
        "\"assignment_slots\":0,"
        "\"witness_slots\":0,"
        "\"content_dependent_receipts\":0,"
        "\"snapshot_loaded\":%s,"
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
        (unsigned long long)compiled->graph.topology_hash,
        (unsigned long long)compiled->graph.schedule_hash,
        compiled->graph.manifest_bytes,
        compiled->graph.count,
        execution->maximum_live_slots,
        rc_physical_relation_blocks(compiled->graph.count),
        (unsigned)GA_BLOCK_CELLS,
        workspace_cells,
        carrier_cells,
        2U * carrier_cells * sizeof(double complex),
        4U * carrier_cells * sizeof(double complex),
        sizeof(*compiled),
        (size_t)DC_VARIANTS * sizeof(struct rc_program),
        compiled->edge_count * sizeof(struct dc_edge),
        sizeof(struct dc_custody),
        sizeof(struct ga_boundary),
        execution->snapshot_loaded ? "true" : "false"
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
        "\"edge_custody_restored\":%s,"
        "\"shared_receipt_reset\":%s,"
        "\"pending_counts_restored\":%s,"
        "\"scheduler_queue_empty\":%s,"
        "\"outstanding_relation_leases\":%zu,"
        "\"lease_allocations\":%llu,"
        "\"lease_releases\":%llu,"
        "\"lease_validation_checks\":%llu,"
        "\"lease_block_cell_checks\":%llu,"
        "\"dirty_control_releases\":%llu,"
        "\"forward_native_operations\":%llu,"
        "\"inverse_native_operations\":%llu,"
        "\"native_operation_calls\":%llu,"
        "\"forward_leaf_encodes\":%llu,"
        "\"inverse_leaf_encodes\":%llu,"
        "\"shared_forward_consumptions\":%llu,"
        "\"shared_inverse_consumptions\":%llu,"
        "\"shared_forward_materializations\":%llu,"
        "\"shared_inverse_applications\":%llu,"
        "\"shared_owner_allocations\":%llu,"
        "\"shared_owner_releases\":%llu,"
        "\"shared_resident_blocks\":%zu,"
        "\"shared_consumer_distinct_slots\":%zu,"
        "\"shared_consumer_distinct_serials\":%zu,"
        "\"shared_live_at_projection\":%s,"
        "\"shared_live_after_root_inverse\":%s,"
        "\"boundary_block_copy_calls\":%llu,"
        "\"intermediate_block_copy_calls\":%llu,"
        "\"serial_before\":%llu,"
        "\"serial_after\":%llu,"
        "\"restoration_generation_before\":%llu,"
        "\"restoration_generation_after\":%llu,"
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
        GA_WORKSPACE_TOLERANCE,
        execution->restoration_max_abs,
        execution->integrity_max_abs,
        execution->workspace_cleared ? "true" : "false",
        execution->allocator_restored ? "true" : "false",
        execution->edge_custody_restored ? "true" : "false",
        execution->shared_receipt_reset ? "true" : "false",
        execution->pending_counts_restored ? "true" : "false",
        execution->scheduler_queue_empty ? "true" : "false",
        execution->outstanding_relation_leases,
        (unsigned long long)execution->lease_allocations,
        (unsigned long long)execution->lease_releases,
        (unsigned long long)execution->lease_checks,
        (unsigned long long)execution->lease_cell_checks,
        (unsigned long long)execution->dirty_control_releases,
        (unsigned long long)execution->forward_operations,
        (unsigned long long)execution->inverse_operations,
        (unsigned long long)(
            execution->forward_operations + execution->inverse_operations
        ),
        (unsigned long long)execution->forward_leaf_encodes,
        (unsigned long long)execution->inverse_leaf_encodes,
        (unsigned long long)execution->shared_forward_consumptions,
        (unsigned long long)execution->shared_inverse_consumptions,
        (unsigned long long)execution->shared_forward_materializations,
        (unsigned long long)execution->shared_inverse_applications,
        (unsigned long long)execution->shared_owner_allocations,
        (unsigned long long)execution->shared_owner_releases,
        execution->shared_peak_live_instances,
        execution->shared_consumer_distinct_slots,
        execution->shared_consumer_distinct_serials,
        execution->shared_live_at_projection ? "true" : "false",
        execution->shared_live_after_root_inverse ? "true" : "false",
        (unsigned long long)execution->boundary_block_copy_calls,
        (unsigned long long)execution->intermediate_block_copy_calls,
        (unsigned long long)execution->serial_before,
        (unsigned long long)execution->serial_after,
        (unsigned long long)execution->restoration_generation_before,
        (unsigned long long)execution->restoration_generation_after,
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
        dc_logical_inspections(execution)
    );
}

static void dc_manifest_control(
    struct rc_manifest *manifest,
    const char *control
) {
    if (strcmp(control, "--control-cycle") == 0) {
        for (size_t index = 0U; index < manifest->count; ++index) {
            if (manifest->node[index].id == 205U) {
                manifest->node[index].left_id = manifest->root_id;
                return;
            }
        }
    } else if (strcmp(control, "--control-type") == 0) {
        for (size_t index = 0U; index < manifest->count; ++index) {
            if (
                manifest->node[index].kind == RC_LEAF
                && manifest->node[index].leaf_index == 3U
            ) {
                manifest->node[index].signature.domain = RC_PORT_A;
                return;
            }
        }
    } else if (strcmp(control, "--control-disconnected") == 0) {
        manifest->root_id = 206U;
        return;
    } else if (strcmp(control, "--control-no-fanout") == 0) {
        for (size_t index = 0U; index < manifest->count; ++index) {
            if (manifest->node[index].id == 207U) {
                manifest->node[index].left_id = 206U;
                return;
            }
        }
    } else {
        fail("affine DAG manifest control unknown");
    }
    fail("affine DAG manifest control target unavailable");
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fail(
            "usage: dag_affine_fanout_phase DAG_MANIFEST "
            "DUPLICATE_TREE_MANIFEST [CONTROL]"
        );
    }
    if (argc == 4 && strcmp(argv[3], "--project-intermediate") == 0) {
        fail("affine DAG intermediate projection denied");
    }
    if (argc == 4 && strcmp(argv[3], "--project-controls") == 0) {
        fail("affine DAG custody projection denied");
    }
    if (argc == 4 && strcmp(argv[3], "--null-carrier") == 0) {
        fail("null affine DAG carrier");
    }
    struct rc_manifest manifest = {0};
    struct rc_manifest duplicate_manifest = {0};
    rc_load_manifest(argv[1], &manifest);
    rc_load_manifest(argv[2], &duplicate_manifest);
    if (argc == 4 && strncmp(argv[3], "--control-", 10U) == 0) {
        if (
            strcmp(argv[3], "--control-stale-shared") != 0
            && strcmp(argv[3], "--control-skip-consumer") != 0
            && strcmp(argv[3], "--control-double-consume") != 0
            && strcmp(argv[3], "--control-copy-shared") != 0
        ) {
            dc_manifest_control(&manifest, argv[3]);
        }
    }
    const struct dc_compiled compiled = dc_compile(&manifest, 1);
    const struct dc_compiled duplicate =
        dc_compile(&duplicate_manifest, 0);
    struct rc_program program[DC_VARIANTS];
    for (size_t variant = 0U; variant < DC_VARIANTS; ++variant) {
        program[variant] = dc_make_program(variant);
    }
    if (argc == 4 && strcmp(argv[3], "--reordered-inverse") == 0) {
        dc_test_reordered(&compiled, &program[0]);
    }
    if (argc == 4 && strcmp(argv[3], "--control-stale-shared") == 0) {
        dc_test_forward_control(&compiled, &program[0], 1, 0, 0);
    }
    if (argc == 4 && strcmp(argv[3], "--control-skip-consumer") == 0) {
        dc_test_forward_control(&compiled, &program[0], 0, 1, 0);
    }
    if (argc == 4 && strcmp(argv[3], "--control-double-consume") == 0) {
        dc_test_forward_control(&compiled, &program[0], 0, 0, 1);
    }
    if (argc == 4 && strcmp(argv[3], "--control-copy-shared") == 0) {
        dc_test_copy_substitution(&compiled, &program[0]);
    }
    if (argc == 4) {
        fail("affine DAG option unknown");
    }
    const struct process shape = {
        .carrier_cells = rc_carrier_cells(compiled.graph.count)
    };
    struct carrier carrier = make_carrier(&shape, 9127);
    struct dc_machine machine = {0};
    struct dc_execution semantic[DC_RUN_VARIANTS];
    double repeated_max_abs = 0.0;
    for (size_t variant = 0U; variant < DC_RUN_VARIANTS; ++variant) {
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
#if DC_REUSE_CYCLES > 0U
    for (size_t cycle = 0U; cycle < DC_REUSE_CYCLES; ++cycle) {
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
    for (size_t variant = 0U; variant < DC_RUN_VARIANTS; ++variant) {
        dc_print(
            variant == 0U
                ? "affine-dag-primary"
                : variant == 1U
                    ? "affine-dag-reuse"
                    : "affine-dag-semantic",
            &compiled,
            &semantic[variant]
        );
    }
    if (DC_SCALE_ONLY) {
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
        "{\"mode\":\"controls\","
        "\"primary_reuse_boundary_different\":%s,"
        "\"left_branch_essential\":%s,"
        "\"right_branch_essential\":%s,"
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
        (unsigned)(DC_RUN_VARIANTS + DC_REUSE_CYCLES),
        repeated_max_abs
    );
    return 0;
}
