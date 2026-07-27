#define _POSIX_C_SOURCE 200809L

/*
 * Mutable successor: a bounded public topology compiler and reversible
 * pebble allocator over the native width-parametric affine phase backend.
 *
 * The manifest contains only public nominal geometry.  It may declare nodes
 * in any order.  The compiler resolves IDs, derives signatures, rejects
 * cycles/fanout/disconnected nodes, emits a deterministic postorder, and
 * computes the exact clean relation-slot requirement.  Execution never
 * projects an internal relation.  A parent consumes its actual resident
 * children, then the children are actually inversed to release their leases.
 * Restoration reconstructs resident children, inverses the actual parent,
 * and recursively clears the reconstruction.  Snapshot reload remains a
 * separately labelled baseline.
 */

#ifndef RC_WIDTH
#define RC_WIDTH 3U
#endif
#define AM_WIDTH RC_WIDTH
#define AM_REUSE_CYCLES 1U
#define AM_PUBLIC_MAIN affine_module_embedded_main
#include "algebraic_affine_module_phase.c"
#undef AM_PUBLIC_MAIN

#define RC_MAX_NODES 15U
#define RC_LEAVES 8U
#define RC_VARIANTS 5U
#ifndef RC_RUN_VARIANTS
#define RC_RUN_VARIANTS RC_VARIANTS
#endif
#ifndef RC_REUSE_CYCLES
#define RC_REUSE_CYCLES 12U
#endif
#ifndef RC_SCALE_ONLY
#define RC_SCALE_ONLY 0
#endif

_Static_assert(RC_RUN_VARIANTS >= 2U, "primary and reuse variants required");
_Static_assert(RC_RUN_VARIANTS <= RC_VARIANTS, "too many variants selected");

enum rc_kind {
    RC_LEAF = 0,
    RC_COMPOSE = 1,
    RC_INTERSECT = 2
};

enum rc_port {
    RC_PORT_A = 1,
    RC_PORT_B = 2,
    RC_PORT_C = 3,
    RC_PORT_D = 4,
    RC_PORT_E = 5
};

enum rc_restore_mode {
    RC_RESTORE_CORRECT = 0,
    RC_RESTORE_WRONG_ROOT = 1,
    RC_RESTORE_MISSING_ROOT = 2,
    RC_RESTORE_SNAPSHOT = 3
};

struct rc_signature {
    enum rc_port domain;
    enum rc_port codomain;
};

struct rc_node {
    unsigned id;
    enum rc_kind kind;
    unsigned left_id;
    unsigned right_id;
    size_t left;
    size_t right;
    size_t leaf_index;
    struct rc_signature signature;
    size_t indegree;
    unsigned char color;
    size_t depth;
    size_t eval_slots;
    size_t uneval_slots;
};

struct rc_manifest {
    struct rc_node node[RC_MAX_NODES];
    size_t count;
    unsigned root_id;
    size_t root_directives;
    size_t bytes_read;
};

struct rc_compiled {
    struct rc_node node[RC_MAX_NODES];
    size_t count;
    size_t root;
    size_t schedule[RC_MAX_NODES];
    size_t schedule_count;
    size_t leaves;
    size_t compose_nodes;
    size_t intersect_nodes;
    size_t edges;
    size_t depth;
    size_t clean_slots;
    uint64_t topology_hash;
    uint64_t schedule_hash;
    size_t manifest_bytes;
};

struct rc_program {
    unsigned char leaf[RC_LEAVES][GA_BLOCK_CELLS];
};

struct rc_lease {
    size_t slot;
    uint64_t serial;
};

struct rc_slot {
    int live;
    int reserved;
    size_t owner;
    uint64_t serial;
    struct rc_signature signature;
};

struct rc_runtime {
    struct carrier *carrier;
    const struct rc_compiled *compiled;
    const struct rc_program *program;
    struct rc_slot slot[RC_MAX_NODES];
    size_t pool;
    size_t live;
    size_t maximum_live;
    uint64_t next_serial;
    uint64_t lease_allocations;
    uint64_t lease_releases;
    uint64_t lease_checks;
    uint64_t lease_cell_checks;
    uint64_t forward_operations;
    uint64_t inverse_operations;
    uint64_t forward_leaf_encodes;
    uint64_t inverse_leaf_encodes;
    uint64_t dirty_control_releases;
    struct ga_stats stats;
};

struct rc_execution {
    struct ga_boundary boundary;
    struct ga_stats stats;
    double restoration_max_abs;
    double integrity_max_abs;
    int workspace_cleared;
    int snapshot_loaded;
    int allocator_restored;
    size_t maximum_live_slots;
    uint64_t lease_allocations;
    uint64_t lease_releases;
    uint64_t lease_checks;
    uint64_t lease_cell_checks;
    uint64_t forward_operations;
    uint64_t inverse_operations;
    uint64_t forward_leaf_encodes;
    uint64_t inverse_leaf_encodes;
    uint64_t dirty_control_releases;
    size_t outstanding_relation_leases;
};

static int rc_signature_equal(
    struct rc_signature left,
    struct rc_signature right
) {
    return left.domain == right.domain
        && left.codomain == right.codomain;
}

static enum rc_port rc_parse_port(char symbol) {
    switch (symbol) {
        case 'A':
            return RC_PORT_A;
        case 'B':
            return RC_PORT_B;
        case 'C':
            return RC_PORT_C;
        case 'D':
            return RC_PORT_D;
        case 'E':
            return RC_PORT_E;
        default:
            fail("recursive affine manifest has unknown nominal port");
    }
    return RC_PORT_A;
}

static void rc_load_manifest(
    const char *path,
    struct rc_manifest *manifest
) {
    FILE *input = fopen(path, "r");
    if (input == NULL) {
        fail("recursive affine manifest unavailable");
    }
    char *line = NULL;
    size_t capacity = 0U;
    while (getline(&line, &capacity, input) >= 0) {
        manifest->bytes_read += strlen(line);
        char keyword[16] = {0};
        if (sscanf(line, " %15s", keyword) != 1) {
            continue;
        }
        if (keyword[0] == '#') {
            continue;
        }
        if (strcmp(keyword, "root") == 0) {
            unsigned id = 0U;
            char extra = '\0';
            if (sscanf(line, " root %u %c", &id, &extra) != 1) {
                free(line);
                fclose(input);
                fail("recursive affine root declaration malformed");
            }
            manifest->root_id = id;
            ++manifest->root_directives;
            continue;
        }
        if (manifest->count >= RC_MAX_NODES) {
            free(line);
            fclose(input);
            fail("recursive affine manifest exceeds node bound");
        }
        struct rc_node *node = &manifest->node[manifest->count];
        node->left = SIZE_MAX;
        node->right = SIZE_MAX;
        node->leaf_index = SIZE_MAX;
        if (strcmp(keyword, "leaf") == 0) {
            char domain = '\0';
            char codomain = '\0';
            unsigned id = 0U;
            unsigned leaf = 0U;
            char extra = '\0';
            if (
                sscanf(
                    line,
                    " leaf %u %c %c %u %c",
                    &id,
                    &domain,
                    &codomain,
                    &leaf,
                    &extra
                ) != 4
            ) {
                free(line);
                fclose(input);
                fail("recursive affine leaf declaration malformed");
            }
            node->id = id;
            node->kind = RC_LEAF;
            node->leaf_index = leaf;
            node->signature.domain = rc_parse_port(domain);
            node->signature.codomain = rc_parse_port(codomain);
        } else if (
            strcmp(keyword, "compose") == 0
            || strcmp(keyword, "intersect") == 0
        ) {
            unsigned id = 0U;
            unsigned left = 0U;
            unsigned right = 0U;
            char extra = '\0';
            if (
                sscanf(
                    line,
                    " %*s %u %u %u %c",
                    &id,
                    &left,
                    &right,
                    &extra
                ) != 3
            ) {
                free(line);
                fclose(input);
                fail("recursive affine operator declaration malformed");
            }
            node->id = id;
            node->kind = strcmp(keyword, "compose") == 0
                ? RC_COMPOSE
                : RC_INTERSECT;
            node->left_id = left;
            node->right_id = right;
        } else {
            free(line);
            fclose(input);
            fail("recursive affine manifest has unknown declaration");
        }
        ++manifest->count;
    }
    const int read_failed = ferror(input);
    free(line);
    if (fclose(input) != 0 || read_failed) {
        fail("recursive affine manifest read failed");
    }
}

static size_t rc_find_node(
    const struct rc_compiled *compiled,
    unsigned id
) {
    for (size_t index = 0U; index < compiled->count; ++index) {
        if (compiled->node[index].id == id) {
            return index;
        }
    }
    fail("recursive affine manifest references unknown node");
    return SIZE_MAX;
}

static size_t rc_max_size(size_t left, size_t right) {
    return left > right ? left : right;
}

static void rc_hash_word(uint64_t *hash, uint64_t value) {
    for (size_t byte = 0U; byte < sizeof(value); ++byte) {
        *hash ^= (unsigned char)(value >> (8U * byte));
        *hash *= UINT64_C(1099511628211);
    }
}

static void rc_visit(struct rc_compiled *compiled, size_t index) {
    struct rc_node *node = &compiled->node[index];
    if (node->color == 1U) {
        fail("recursive affine manifest contains cycle");
    }
    if (node->color == 2U) {
        return;
    }
    node->color = 1U;
    if (node->kind == RC_LEAF) {
        node->depth = 0U;
        node->eval_slots = 1U;
        node->uneval_slots = 1U;
    } else {
        rc_visit(compiled, node->left);
        rc_visit(compiled, node->right);
        const struct rc_node *left = &compiled->node[node->left];
        const struct rc_node *right = &compiled->node[node->right];
        if (node->kind == RC_COMPOSE) {
            if (left->signature.codomain != right->signature.domain) {
                fail("recursive affine composition nominal type mismatch");
            }
            node->signature.domain = left->signature.domain;
            node->signature.codomain = right->signature.codomain;
        } else {
            if (!rc_signature_equal(left->signature, right->signature)) {
                fail("recursive affine intersection nominal type mismatch");
            }
            node->signature = left->signature;
        }
        node->depth = 1U + rc_max_size(left->depth, right->depth);
        size_t need = left->eval_slots;
        need = rc_max_size(need, 1U + right->eval_slots);
        need = rc_max_size(need, 3U);
        need = rc_max_size(need, 2U + right->uneval_slots);
        need = rc_max_size(need, 1U + left->uneval_slots);
        node->eval_slots = need;
        need = 1U + left->eval_slots;
        need = rc_max_size(need, 2U + right->eval_slots);
        need = rc_max_size(need, 3U);
        need = rc_max_size(need, 2U + right->uneval_slots);
        need = rc_max_size(need, 1U + left->uneval_slots);
        node->uneval_slots = need;
    }
    node->color = 2U;
    if (compiled->schedule_count >= RC_MAX_NODES) {
        fail("recursive affine compiled schedule exceeds bound");
    }
    compiled->schedule[compiled->schedule_count++] = index;
}

static struct rc_compiled rc_compile(
    const struct rc_manifest *manifest
) {
    struct rc_compiled compiled = {0};
    if (manifest->root_directives != 1U) {
        fail("recursive affine manifest requires exactly one root");
    }
    if (manifest->count == 0U) {
        fail("recursive affine manifest is empty");
    }
    compiled.count = manifest->count;
    compiled.manifest_bytes = manifest->bytes_read;
    memcpy(
        compiled.node,
        manifest->node,
        manifest->count * sizeof(*manifest->node)
    );
    for (size_t left = 0U; left < compiled.count; ++left) {
        for (size_t right = left + 1U; right < compiled.count; ++right) {
            if (compiled.node[left].id == compiled.node[right].id) {
                fail("recursive affine manifest has duplicate node id");
            }
        }
    }
    compiled.root = rc_find_node(&compiled, manifest->root_id);
    if (compiled.node[compiled.root].kind == RC_LEAF) {
        fail("recursive affine root must be an operator");
    }
    unsigned char leaf_seen[RC_LEAVES] = {0};
    for (size_t index = 0U; index < compiled.count; ++index) {
        struct rc_node *node = &compiled.node[index];
        if (node->kind == RC_LEAF) {
            if (
                node->leaf_index >= RC_LEAVES
                || leaf_seen[node->leaf_index] != 0U
            ) {
                fail("recursive affine leaf body ownership invalid");
            }
            leaf_seen[node->leaf_index] = 1U;
            ++compiled.leaves;
            continue;
        }
        if (node->left_id == node->right_id || node->left_id == node->id) {
            fail("recursive affine operator aliases an operand");
        }
        if (node->right_id == node->id) {
            fail("recursive affine operator contains self reference");
        }
        node->left = rc_find_node(&compiled, node->left_id);
        node->right = rc_find_node(&compiled, node->right_id);
        ++compiled.node[node->left].indegree;
        ++compiled.node[node->right].indegree;
        if (node->kind == RC_COMPOSE) {
            ++compiled.compose_nodes;
        } else if (node->kind == RC_INTERSECT) {
            ++compiled.intersect_nodes;
        } else {
            fail("recursive affine operator kind invalid");
        }
    }
    if (
        compiled.leaves != RC_LEAVES
        || compiled.compose_nodes == 0U
        || compiled.intersect_nodes == 0U
    ) {
        fail("recursive affine manifest lacks required mixed tree");
    }
    rc_visit(&compiled, compiled.root);
    for (size_t index = 0U; index < compiled.count; ++index) {
        const size_t expected = index == compiled.root ? 0U : 1U;
        if (compiled.node[index].indegree != expected) {
            fail("recursive affine manifest is not a single tree");
        }
    }
    if (compiled.schedule_count != compiled.count) {
        fail("recursive affine manifest contains disconnected node");
    }
    compiled.edges = 2U * (
        compiled.compose_nodes + compiled.intersect_nodes
    );
    compiled.depth = compiled.node[compiled.root].depth;
    compiled.clean_slots = rc_max_size(
        compiled.node[compiled.root].eval_slots,
        compiled.node[compiled.root].uneval_slots
    );
    if (compiled.clean_slots > RC_MAX_NODES) {
        fail("recursive affine clean allocation exceeds bound");
    }
    compiled.topology_hash = UINT64_C(14695981039346656037);
    for (size_t index = 0U; index < compiled.count; ++index) {
        const struct rc_node *node = &compiled.node[index];
        rc_hash_word(&compiled.topology_hash, node->id);
        rc_hash_word(&compiled.topology_hash, node->kind);
        rc_hash_word(&compiled.topology_hash, node->left_id);
        rc_hash_word(&compiled.topology_hash, node->right_id);
        rc_hash_word(&compiled.topology_hash, node->leaf_index);
        rc_hash_word(&compiled.topology_hash, node->signature.domain);
        rc_hash_word(&compiled.topology_hash, node->signature.codomain);
    }
    compiled.schedule_hash = UINT64_C(14695981039346656037);
    for (size_t step = 0U; step < compiled.schedule_count; ++step) {
        rc_hash_word(
            &compiled.schedule_hash,
            compiled.node[compiled.schedule[step]].id
        );
    }
    return compiled;
}

static size_t rc_slot_start(size_t slot) {
    return slot < 5U
        ? slot * GA_BLOCK_CELLS
        : GA_CARRIER_CELLS + (slot - 5U) * GA_BLOCK_CELLS;
}

static size_t rc_carrier_cells(size_t working_slots) {
    return GA_CARRIER_CELLS
        + (
            working_slots > 5U
                ? (working_slots - 5U) * GA_BLOCK_CELLS
                : 0U
        );
}

static size_t rc_physical_relation_blocks(size_t working_slots) {
    return working_slots + 1U > GA_RELATION_BLOCKS
        ? working_slots + 1U
        : GA_RELATION_BLOCKS;
}

static void rc_set_map_row(
    unsigned char relation[GA_BLOCK_CELLS],
    size_t output,
    unsigned char constant,
    const size_t *input,
    size_t inputs
) {
    memset(
        relation + output * GA_ROW_CELLS,
        0,
        GA_ROW_CELLS * sizeof(*relation)
    );
    ga_begin_equation(relation, output, constant);
    for (size_t term = 0U; term < inputs; ++term) {
        ga_set_term(relation, output, input[term]);
    }
    ga_set_term(relation, output, GA_WIDTH + output);
}

static struct rc_program rc_make_program(size_t variant) {
    struct rc_program program = {0};
    for (size_t leaf = 0U; leaf < RC_LEAVES; ++leaf) {
        for (size_t bit = 0U; bit < GA_WIDTH; ++bit) {
            const size_t identity[] = {bit};
            rc_set_map_row(
                program.leaf[leaf], bit, 0U, identity, 1U
            );
        }
    }
    if (variant == 0U) {
        const size_t f0[] = {1U};
        const size_t f1[] = {2U};
        const size_t f2[] = {0U};
        rc_set_map_row(program.leaf[0], 0U, 0U, f0, 1U);
        rc_set_map_row(program.leaf[0], 1U, 0U, f1, 1U);
        rc_set_map_row(program.leaf[0], 2U, 1U, f2, 1U);
        const size_t g0[] = {0U, 2U};
        const size_t g1[] = {0U};
        const size_t g2[] = {0U, 1U};
        rc_set_map_row(program.leaf[1], 0U, 1U, g0, 2U);
        rc_set_map_row(program.leaf[1], 1U, 0U, g1, 1U);
        rc_set_map_row(program.leaf[1], 2U, 0U, g2, 2U);
        const size_t p0[] = {0U, 1U};
        const size_t p1[] = {1U};
        const size_t p2[] = {0U, 1U, 2U};
        rc_set_map_row(program.leaf[3], 0U, 0U, p0, 2U);
        rc_set_map_row(program.leaf[3], 1U, 0U, p1, 1U);
        rc_set_map_row(program.leaf[3], 2U, 1U, p2, 3U);
        const size_t q0[] = {0U, 2U};
        rc_set_map_row(program.leaf[7], 0U, 0U, q0, 2U);
    } else if (variant == 1U) {
        /* Eight unrelated identity bodies exercise the same compiled tree. */
    } else if (variant == 2U) {
        const size_t flipped[] = {0U};
        rc_set_map_row(program.leaf[7], 0U, 1U, flipped, 1U);
    } else if (variant == 3U) {
        const size_t high[] = {0U, GA_WIDTH - 1U};
        rc_set_map_row(program.leaf[7], 0U, 0U, high, 2U);
    } else if (variant == 4U) {
        memset(program.leaf[0], 0, sizeof(program.leaf[0]));
        program.leaf[0][GA_BLOCK_CELLS - 1U] = 1U;
    } else {
        fail("recursive affine program variant unknown");
    }
    return program;
}

static int rc_block_is_zero(
    struct rc_runtime *runtime,
    size_t slot
) {
    const size_t start = rc_slot_start(slot);
    for (size_t field = 0U; field < GA_BLOCK_CELLS; ++field) {
        ++runtime->lease_cell_checks;
        if (
            cabs(relative(runtime->carrier, start + field) - root3(0))
                > GA_WORKSPACE_TOLERANCE
        ) {
            return 0;
        }
    }
    return 1;
}

static struct rc_lease rc_allocate(
    struct rc_runtime *runtime,
    size_t owner,
    struct rc_signature signature
) {
    for (size_t slot = 0U; slot < runtime->pool; ++slot) {
        if (
            runtime->slot[slot].live
            && runtime->slot[slot].owner == owner
        ) {
            fail("recursive affine logical node has overlapping lease");
        }
    }
    for (size_t slot = 0U; slot < runtime->pool; ++slot) {
        if (!runtime->slot[slot].live) {
            struct rc_slot *entry = &runtime->slot[slot];
            entry->live = 1;
            entry->reserved = 1;
            entry->owner = owner;
            entry->signature = signature;
            entry->serial = ++runtime->next_serial;
            ++runtime->lease_allocations;
            ++runtime->live;
            if (runtime->live > runtime->maximum_live) {
                runtime->maximum_live = runtime->live;
            }
            return (struct rc_lease){
                .slot = slot,
                .serial = entry->serial
            };
        }
    }
    fail("recursive affine clean relation-slot pool exhausted");
    return (struct rc_lease){0};
}

static void rc_validate_lease(
    struct rc_runtime *runtime,
    struct rc_lease lease,
    size_t owner,
    struct rc_signature signature,
    int require_reserved
) {
    ++runtime->lease_checks;
    if (lease.slot >= runtime->pool) {
        fail("recursive affine lease outside pool");
    }
    const struct rc_slot *entry = &runtime->slot[lease.slot];
    if (
        !entry->live
        || entry->serial != lease.serial
        || entry->owner != owner
        || !rc_signature_equal(entry->signature, signature)
        || entry->reserved != require_reserved
    ) {
        fail("recursive affine stale or mistyped relation lease");
    }
}

static void rc_release(
    struct rc_runtime *runtime,
    struct rc_lease lease,
    size_t owner,
    struct rc_signature signature,
    int allow_dirty_control
) {
    rc_validate_lease(runtime, lease, owner, signature, 0);
    if (!allow_dirty_control && !rc_block_is_zero(runtime, lease.slot)) {
        fail("recursive affine lease released before exact inverse");
    }
    if (allow_dirty_control) {
        ++runtime->dirty_control_releases;
    }
    memset(&runtime->slot[lease.slot], 0, sizeof(runtime->slot[0]));
    --runtime->live;
    ++runtime->lease_releases;
}

static int rc_allocator_is_restored(const struct rc_runtime *runtime) {
    if (runtime->live != 0U) {
        return 0;
    }
    for (size_t slot = 0U; slot < runtime->pool; ++slot) {
        if (runtime->slot[slot].live) {
            return 0;
        }
    }
    return 1;
}

static void rc_apply_node(
    struct rc_runtime *runtime,
    size_t node_index,
    struct rc_lease left_lease,
    struct rc_lease right_lease,
    struct rc_lease output_lease,
    int inverse,
    int wrong_first_factor
) {
    const struct rc_node *node = &runtime->compiled->node[node_index];
    const struct rc_node *left = &runtime->compiled->node[node->left];
    const struct rc_node *right = &runtime->compiled->node[node->right];
    rc_validate_lease(
        runtime, left_lease, node->left, left->signature, 0
    );
    rc_validate_lease(
        runtime, right_lease, node->right, right->signature, 0
    );
    rc_validate_lease(
        runtime,
        output_lease,
        node_index,
        node->signature,
        inverse ? 0 : 1
    );
    if (
        output_lease.slot == left_lease.slot
        || output_lease.slot == right_lease.slot
    ) {
        fail("recursive affine output aliases live operand");
    }
    const struct ga_descriptor descriptor = {
        .left_start = rc_slot_start(left_lease.slot),
        .right_start = rc_slot_start(right_lease.slot),
        .output_start = rc_slot_start(output_lease.slot)
    };
    if (node->kind == RC_COMPOSE) {
        ga_apply_compose(
            runtime->carrier,
            &descriptor,
            inverse,
            wrong_first_factor,
            &runtime->stats
        );
    } else {
        am_apply_intersect(
            runtime->carrier,
            &descriptor,
            inverse,
            wrong_first_factor,
            &runtime->stats
        );
    }
    if (inverse) {
        ++runtime->inverse_operations;
    } else {
        ++runtime->forward_operations;
        runtime->slot[output_lease.slot].reserved = 0;
    }
}

static struct rc_lease rc_eval(
    struct rc_runtime *runtime,
    size_t node_index
);

static void rc_uneval(
    struct rc_runtime *runtime,
    size_t node_index,
    struct rc_lease output,
    size_t wrong_node
);

static struct rc_lease rc_eval(
    struct rc_runtime *runtime,
    size_t node_index
) {
    const struct rc_node *node = &runtime->compiled->node[node_index];
    if (node->kind == RC_LEAF) {
        struct rc_lease output = rc_allocate(
            runtime, node_index, node->signature
        );
        ga_encode_relation(
            runtime->carrier,
            runtime->program->leaf[node->leaf_index],
            rc_slot_start(output.slot),
            0,
            &runtime->stats
        );
        runtime->slot[output.slot].reserved = 0;
        ++runtime->forward_leaf_encodes;
        return output;
    }
    const struct rc_lease left = rc_eval(runtime, node->left);
    const struct rc_lease right = rc_eval(runtime, node->right);
    const struct rc_lease output = rc_allocate(
        runtime, node_index, node->signature
    );
    rc_apply_node(
        runtime, node_index, left, right, output, 0, 0
    );
    rc_uneval(runtime, node->right, right, SIZE_MAX);
    rc_uneval(runtime, node->left, left, SIZE_MAX);
    return output;
}

static void rc_uneval(
    struct rc_runtime *runtime,
    size_t node_index,
    struct rc_lease output,
    size_t wrong_node
) {
    const struct rc_node *node = &runtime->compiled->node[node_index];
    rc_validate_lease(
        runtime, output, node_index, node->signature, 0
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
        rc_release(runtime, output, node_index, node->signature, 0);
        return;
    }
    const struct rc_lease left = rc_eval(runtime, node->left);
    const struct rc_lease right = rc_eval(runtime, node->right);
    const int wrong_here = node_index == wrong_node;
    rc_apply_node(
        runtime,
        node_index,
        left,
        right,
        output,
        1,
        wrong_here
    );
    rc_uneval(runtime, node->right, right, SIZE_MAX);
    rc_uneval(runtime, node->left, left, SIZE_MAX);
    rc_release(
        runtime,
        output,
        node_index,
        node->signature,
        wrong_here
    );
}

static struct rc_lease rc_retain_forward(
    struct rc_runtime *runtime,
    size_t node_slot[RC_MAX_NODES]
) {
    for (size_t index = 0U; index < RC_MAX_NODES; ++index) {
        node_slot[index] = SIZE_MAX;
    }
    for (
        size_t step = 0U;
        step < runtime->compiled->schedule_count;
        ++step
    ) {
        const size_t node_index = runtime->compiled->schedule[step];
        const struct rc_node *node =
            &runtime->compiled->node[node_index];
        const struct rc_lease output = rc_allocate(
            runtime, node_index, node->signature
        );
        node_slot[node_index] = output.slot;
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
            const struct rc_lease left = {
                .slot = node_slot[node->left],
                .serial = runtime->slot[node_slot[node->left]].serial
            };
            const struct rc_lease right = {
                .slot = node_slot[node->right],
                .serial = runtime->slot[node_slot[node->right]].serial
            };
            rc_apply_node(
                runtime, node_index, left, right, output, 0, 0
            );
        }
    }
    const size_t root_slot = node_slot[runtime->compiled->root];
    return (struct rc_lease){
        .slot = root_slot,
        .serial = runtime->slot[root_slot].serial
    };
}

static void rc_retain_reverse(
    struct rc_runtime *runtime,
    const size_t node_slot[RC_MAX_NODES]
) {
    for (
        size_t step = runtime->compiled->schedule_count;
        step-- > 0U;
    ) {
        const size_t node_index = runtime->compiled->schedule[step];
        const struct rc_node *node =
            &runtime->compiled->node[node_index];
        const struct rc_lease output = {
            .slot = node_slot[node_index],
            .serial = runtime->slot[node_slot[node_index]].serial
        };
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
            const struct rc_lease left = {
                .slot = node_slot[node->left],
                .serial = runtime->slot[node_slot[node->left]].serial
            };
            const struct rc_lease right = {
                .slot = node_slot[node->right],
                .serial = runtime->slot[node_slot[node->right]].serial
            };
            rc_apply_node(
                runtime, node_index, left, right, output, 1, 0
            );
        }
        rc_release(runtime, output, node_index, node->signature, 0);
    }
}

static struct rc_execution rc_execute(
    struct carrier *carrier,
    const struct rc_compiled *compiled,
    const struct rc_program *program,
    size_t pool,
    enum rc_restore_mode restore_mode,
    int retain_all
) {
    struct rc_execution execution = {0};
    struct carrier borrowed = snapshot_carrier(carrier);
    struct rc_runtime runtime = {
        .carrier = carrier,
        .compiled = compiled,
        .program = program,
        .pool = pool
    };
    size_t node_slot[RC_MAX_NODES];
    const struct rc_lease root = retain_all
        ? rc_retain_forward(&runtime, node_slot)
        : rc_eval(&runtime, compiled->root);
    rc_validate_lease(
        &runtime,
        root,
        compiled->root,
        compiled->node[compiled->root].signature,
        0
    );
    ga_copy_block(
        carrier,
        rc_slot_start(root.slot),
        GA_BOUNDARY_START,
        0,
        &runtime.stats
    );
    execution.boundary = ga_latch_boundary(carrier, &runtime.stats);
    if (restore_mode == RC_RESTORE_SNAPSHOT) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        memset(runtime.slot, 0, sizeof(runtime.slot));
        runtime.live = 0U;
        execution.snapshot_loaded = 1;
    } else {
        ga_copy_block(
            carrier,
            rc_slot_start(root.slot),
            GA_BOUNDARY_START,
            1,
            &runtime.stats
        );
        if (restore_mode != RC_RESTORE_MISSING_ROOT) {
            if (retain_all) {
                rc_retain_reverse(&runtime, node_slot);
            } else {
                rc_uneval(
                    &runtime,
                    compiled->root,
                    root,
                    restore_mode == RC_RESTORE_WRONG_ROOT
                        ? compiled->root
                        : SIZE_MAX
                );
            }
        }
    }
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    runtime.stats.restoration_cell_checks += carrier->cells;
    execution.integrity_max_abs = integrity(carrier);
    runtime.stats.integrity_cell_checks += carrier->cells;
    execution.workspace_cleared = ga_workspace_is_zero(
        carrier, &runtime.stats
    );
    execution.allocator_restored = rc_allocator_is_restored(&runtime);
    execution.stats = runtime.stats;
    execution.maximum_live_slots = runtime.maximum_live;
    execution.lease_allocations = runtime.lease_allocations;
    execution.lease_releases = runtime.lease_releases;
    execution.lease_checks = runtime.lease_checks;
    execution.lease_cell_checks = runtime.lease_cell_checks;
    execution.forward_operations = runtime.forward_operations;
    execution.inverse_operations = runtime.inverse_operations;
    execution.forward_leaf_encodes = runtime.forward_leaf_encodes;
    execution.inverse_leaf_encodes = runtime.inverse_leaf_encodes;
    execution.dirty_control_releases =
        runtime.dirty_control_releases;
    execution.outstanding_relation_leases = runtime.live;
    free_carrier(&borrowed);
    return execution;
}

static void rc_test_reordered(
    const struct rc_compiled *compiled,
    const struct rc_program *program
) {
    const size_t pool = compiled->clean_slots;
    const struct process shape = {
        .carrier_cells = rc_carrier_cells(pool)
    };
    struct carrier carrier = make_carrier(&shape, 7717);
    struct rc_runtime runtime = {
        .carrier = &carrier,
        .compiled = compiled,
        .program = program,
        .pool = pool
    };
    const struct rc_lease root = rc_eval(&runtime, compiled->root);
    const struct rc_node *root_node = &compiled->node[compiled->root];
    const struct rc_lease left = rc_eval(&runtime, root_node->left);
    const struct rc_lease right = rc_eval(&runtime, root_node->right);
    rc_uneval(&runtime, root_node->left, left, SIZE_MAX);
    rc_apply_node(
        &runtime,
        compiled->root,
        left,
        right,
        root,
        1,
        0
    );
    free_carrier(&carrier);
    fail("recursive affine reordered inverse was not rejected");
}

static unsigned long long rc_logical_inspections(
    const struct rc_execution *execution
) {
    return execution->stats.carrier_reads
        + execution->stats.boundary_decodes
        + execution->stats.workspace_cell_checks
        + execution->stats.restoration_cell_checks
        + execution->stats.integrity_cell_checks
        + execution->lease_checks
        + execution->lease_cell_checks;
}

static void rc_print_execution(
    const char *mode,
    const struct rc_compiled *compiled,
    const struct rc_execution *execution,
    size_t pool,
    int retain_all
) {
    const size_t carrier_cells = rc_carrier_cells(pool);
    const size_t workspace_cells =
        GA_CARRIER_CELLS - GA_RELATION_BLOCKS * GA_BLOCK_CELLS;
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"BOUNDED_PUBLIC_RECURSIVE_AFFINE_"
        "SERIES_INTERSECTION_TREE_COMPILATION\","
        "\"width\":%u,"
        "\"manifest_nodes\":%zu,"
        "\"manifest_leaves\":%zu,"
        "\"compose_nodes\":%zu,"
        "\"intersect_nodes\":%zu,"
        "\"tree_edges\":%zu,"
        "\"tree_depth\":%zu,"
        "\"manifest_dependency_ordered\":false,"
        "\"compiled_postorder_valid\":true,"
        "\"derived_nominal_signatures\":true,"
        "\"address_assigned_after_validation\":true,"
        "\"tree_only_no_dag_fanout\":true,"
        "\"topology_fnv1a64\":\"%016llx\","
        "\"schedule_fnv1a64\":\"%016llx\","
        "\"manifest_bytes\":%zu,"
        "\"reversible_pebbling\":%s,"
        "\"working_relation_slots\":%zu,"
        "\"maximum_live_relation_slots\":%zu,"
        "\"physical_relation_blocks_including_boundary\":%zu,"
        "\"retain_all_relation_blocks_including_boundary\":%zu,"
        "\"relation_cells\":%u,"
        "\"workspace_cells\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"live_carrier_bytes\":%zu,"
        "\"carrier_and_verification_snapshot_heap_bytes\":%zu,"
        "\"compiled_topology_bytes_current_abi\":%zu,"
        "\"program_storage_bytes_current_abi\":%zu,"
        "\"boundary_storage_bytes_current_abi\":%zu,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"serialized_intermediate_coefficients\":0,"
        "\"pre_inverse_carrier_aggregate_outputs\":0,"
        "\"host_selected_pivots\":0,"
        "\"tuple_slots\":0,"
        "\"assignment_slots\":0,"
        "\"witness_slots\":0,"
        "\"snapshot_loaded\":%s,"
        "\"snapshot_creation_complex_values_copied\":%zu,"
        "\"snapshot_reload_complex_values_copied\":%zu,"
        "\"boundary_coefficients\":",
        mode,
        (unsigned)GA_WIDTH,
        compiled->count,
        compiled->leaves,
        compiled->compose_nodes,
        compiled->intersect_nodes,
        compiled->edges,
        compiled->depth,
        (unsigned long long)compiled->topology_hash,
        (unsigned long long)compiled->schedule_hash,
        compiled->manifest_bytes,
        retain_all ? "false" : "true",
        pool,
        execution->maximum_live_slots,
        rc_physical_relation_blocks(pool),
        compiled->count + 1U,
        (unsigned)GA_BLOCK_CELLS,
        workspace_cells,
        carrier_cells,
        2U * carrier_cells * sizeof(double complex),
        4U * carrier_cells * sizeof(double complex),
        sizeof(*compiled),
        RC_VARIANTS * sizeof(struct rc_program),
        sizeof(struct ga_boundary),
        execution->snapshot_loaded ? "true" : "false",
        2U * carrier_cells,
        execution->snapshot_loaded ? carrier_cells : 0U
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
        "\"outstanding_relation_leases\":%zu,"
        "\"lease_allocations\":%llu,"
        "\"lease_releases\":%llu,"
        "\"lease_validation_checks\":%llu,"
        "\"lease_block_cell_checks\":%llu,"
        "\"dirty_control_releases\":%llu,"
        "\"forward_native_operations\":%llu,"
        "\"inverse_native_operations\":%llu,"
        "\"native_operation_calls\":%llu,"
        "\"transient_forward_recomputations\":%llu,"
        "\"inverse_factor_recomputations\":%llu,"
        "\"forward_leaf_encodes\":%llu,"
        "\"inverse_leaf_encodes\":%llu,"
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
        (unsigned long long)(
            retain_all
                ? 0U
                : execution->forward_operations
                    - compiled->compose_nodes
                    - compiled->intersect_nodes
        ),
        (unsigned long long)execution->inverse_operations,
        (unsigned long long)execution->forward_leaf_encodes,
        (unsigned long long)execution->inverse_leaf_encodes,
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
        rc_logical_inspections(execution)
    );
}

static void rc_apply_manifest_control(
    struct rc_manifest *manifest,
    const char *control
) {
    if (strcmp(control, "--control-cycle") == 0) {
        for (size_t index = 0U; index < manifest->count; ++index) {
            if (manifest->node[index].id == 111U) {
                manifest->node[index].left_id = manifest->root_id;
                return;
            }
        }
    } else if (strcmp(control, "--control-unknown-child") == 0) {
        for (size_t index = 0U; index < manifest->count; ++index) {
            if (manifest->node[index].id == manifest->root_id) {
                manifest->node[index].left_id = 999999U;
                return;
            }
        }
    } else if (strcmp(control, "--control-fanout") == 0) {
        for (size_t index = 0U; index < manifest->count; ++index) {
            if (manifest->node[index].id == 110U) {
                manifest->node[index].left_id = 101U;
                return;
            }
        }
    } else if (strcmp(control, "--control-type-mismatch") == 0) {
        for (size_t index = 0U; index < manifest->count; ++index) {
            if (manifest->node[index].id == 108U) {
                manifest->node[index].signature.domain = RC_PORT_A;
                return;
            }
        }
    } else if (strcmp(control, "--control-duplicate-id") == 0) {
        manifest->node[manifest->count - 1U].id =
            manifest->node[0].id;
        return;
    } else if (strcmp(control, "--control-unreachable") == 0) {
        manifest->root_id = 111U;
        return;
    } else if (strcmp(control, "--control-multiple-root") == 0) {
        manifest->root_directives = 2U;
        return;
    } else if (strcmp(control, "--control-leaf-owner") == 0) {
        for (size_t index = 0U; index < manifest->count; ++index) {
            if (
                manifest->node[index].kind == RC_LEAF
                && manifest->node[index].leaf_index == 7U
            ) {
                manifest->node[index].leaf_index = 0U;
                return;
            }
        }
    } else {
        fail("recursive affine control unknown");
    }
    fail("recursive affine control target unavailable");
}

#ifndef RC_PUBLIC_MAIN
#define RC_PUBLIC_MAIN main
#endif

int RC_PUBLIC_MAIN(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fail(
            "usage: recursive_affine_module_phase MANIFEST [CONTROL]"
        );
    }
    if (argc == 3 && strcmp(argv[2], "--project-intermediate") == 0) {
        fail("recursive affine intermediate projection denied");
    }
    if (argc == 3 && strcmp(argv[2], "--project-controls") == 0) {
        fail("recursive affine phase-control projection denied");
    }
    if (argc == 3 && strcmp(argv[2], "--null-carrier") == 0) {
        fail("null recursive affine carrier");
    }
    struct rc_manifest manifest = {0};
    rc_load_manifest(argv[1], &manifest);
    if (argc == 3 && strncmp(argv[2], "--control-", 10U) == 0) {
        rc_apply_manifest_control(&manifest, argv[2]);
    }
    const struct rc_compiled compiled = rc_compile(&manifest);
    struct rc_program program[RC_VARIANTS];
    for (size_t variant = 0U; variant < RC_VARIANTS; ++variant) {
        program[variant] = rc_make_program(variant);
    }
    if (argc == 3 && strcmp(argv[2], "--reordered-inverse") == 0) {
        rc_test_reordered(&compiled, &program[0]);
    }
    if (argc == 3) {
        fail("recursive affine option unknown");
    }
    const size_t clean_pool = compiled.clean_slots;
    const struct process clean_shape = {
        .carrier_cells = rc_carrier_cells(clean_pool)
    };
    struct carrier carrier = make_carrier(&clean_shape, 7717);
    struct rc_execution semantic[RC_RUN_VARIANTS];
    double repeated_max_abs = 0.0;
    for (size_t variant = 0U; variant < RC_RUN_VARIANTS; ++variant) {
        semantic[variant] = rc_execute(
            &carrier,
            &compiled,
            &program[variant],
            clean_pool,
            RC_RESTORE_CORRECT,
            0
        );
        if (semantic[variant].restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = semantic[variant].restoration_max_abs;
        }
    }
#if RC_REUSE_CYCLES > 0U
    for (size_t cycle = 0U; cycle < RC_REUSE_CYCLES; ++cycle) {
        const struct rc_execution repeated = rc_execute(
            &carrier,
            &compiled,
            &program[cycle % 2U],
            clean_pool,
            RC_RESTORE_CORRECT,
            0
        );
        if (repeated.restoration_max_abs > repeated_max_abs) {
            repeated_max_abs = repeated.restoration_max_abs;
        }
    }
#endif
    free_carrier(&carrier);
    for (size_t variant = 0U; variant < RC_RUN_VARIANTS; ++variant) {
        rc_print_execution(
            variant == 0U
                ? "recursive-affine-primary"
                : variant == 1U
                    ? "recursive-affine-reuse"
                    : "recursive-affine-semantic",
            &compiled,
            &semantic[variant],
            clean_pool,
            0
        );
    }
    if (RC_SCALE_ONLY) {
        return 0;
    }
    carrier = make_carrier(&clean_shape, 7717);
    const struct rc_execution wrong = rc_execute(
        &carrier,
        &compiled,
        &program[0],
        clean_pool,
        RC_RESTORE_WRONG_ROOT,
        0
    );
    free_carrier(&carrier);
    carrier = make_carrier(&clean_shape, 7717);
    const struct rc_execution missing = rc_execute(
        &carrier,
        &compiled,
        &program[0],
        clean_pool,
        RC_RESTORE_MISSING_ROOT,
        0
    );
    free_carrier(&carrier);
    carrier = make_carrier(&clean_shape, 7717);
    const struct rc_execution snapshot = rc_execute(
        &carrier,
        &compiled,
        &program[0],
        clean_pool,
        RC_RESTORE_SNAPSHOT,
        0
    );
    free_carrier(&carrier);
    const size_t retain_pool = compiled.count;
    const struct process retain_shape = {
        .carrier_cells = rc_carrier_cells(retain_pool)
    };
    carrier = make_carrier(&retain_shape, 7717);
    const struct rc_execution retained = rc_execute(
        &carrier,
        &compiled,
        &program[0],
        retain_pool,
        RC_RESTORE_CORRECT,
        1
    );
    free_carrier(&carrier);
    rc_print_execution(
        "snapshot-baseline",
        &compiled,
        &snapshot,
        clean_pool,
        0
    );
    rc_print_execution(
        "retain-all-phase-baseline",
        &compiled,
        &retained,
        retain_pool,
        1
    );
    printf(
        "{\"mode\":\"controls\","
        "\"primary_reuse_boundary_different\":%s,"
        "\"contradictory_branch_empty\":%s,"
        "\"highest_input_column_restriction_nonempty\":%s,"
        "\"empty_leaf_absorber\":%s,"
        "\"coefficient_oblivious_schedule\":%s,"
        "\"wrong_root_inverse\":%s,"
        "\"missing_root_inverse\":%s,"
        "\"retain_all_boundary_equal\":%s,"
        "\"retain_all_restored\":%s,"
        "\"snapshot_separate_weaker_path\":true,"
        "\"snapshot_post_projection_inverse_factor_recomputations\":0,"
        "\"clean_slots\":%zu,"
        "\"retain_all_working_slots\":%zu,"
        "\"relation_slot_reduction\":%zu,"
        "\"pebbled_native_operation_calls\":%llu,"
        "\"retain_all_native_operation_calls\":%llu,"
        "\"wrong_root_restoration_abs\":%.12g,"
        "\"missing_root_restoration_abs\":%.12g,"
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
        ga_stats_same_schedule(
            &semantic[0].stats, &semantic[1].stats
        ) ? "true" : "false",
        wrong.restoration_max_abs > 1.0e-6 ? "true" : "false",
        missing.restoration_max_abs > 1.0e-6 ? "true" : "false",
        ga_boundary_equal(
            &semantic[0].boundary, &retained.boundary
        ) ? "true" : "false",
        (
            retained.restoration_max_abs <= GA_WORKSPACE_TOLERANCE
            && retained.allocator_restored
        ) ? "true" : "false",
        clean_pool,
        retain_pool,
        retain_pool - clean_pool,
        (unsigned long long)(
            semantic[0].forward_operations
            + semantic[0].inverse_operations
        ),
        (unsigned long long)(
            retained.forward_operations
            + retained.inverse_operations
        ),
        wrong.restoration_max_abs,
        missing.restoration_max_abs,
        (unsigned)(RC_RUN_VARIANTS + RC_REUSE_CYCLES),
        repeated_max_abs
    );
    return 0;
}
