#define _POSIX_C_SOURCE 200809L

/*
 * Separate compact GF(2) adjudicator for the bounded fanout experiment.
 * It shares no phase implementation and emits only the final root relation.
 */

#ifndef DF_WIDTH
#define DF_WIDTH 3U
#endif
#define RR_WIDTH DF_WIDTH
#define RR_RUN_VARIANTS 5U
#define RR_PUBLIC_MAIN recursive_affine_reference_embedded_main
#include "recursive_affine_module_reference.c"
#undef RR_PUBLIC_MAIN

#ifndef DF_LEAF_BODIES
#define DF_LEAF_BODIES 4U
#endif
#define DF_VARIANTS 5U
#ifndef DF_MAX_FANOUT
#define DF_MAX_FANOUT 2U
#endif
#ifndef DF_RUN_VARIANTS
#define DF_RUN_VARIANTS DF_VARIANTS
#endif

static void df_compile(struct rr_graph *graph) {
    if (graph->root_directives != 1U || graph->count == 0U) {
        fail("affine DAG reference requires one root");
    }
    for (size_t left = 0U; left < graph->count; ++left) {
        for (size_t right = left + 1U; right < graph->count; ++right) {
            if (graph->node[left].id == graph->node[right].id) {
                fail("affine DAG reference duplicate node id");
            }
        }
    }
    graph->root = rr_find(graph, graph->root_id);
    if (graph->node[graph->root].kind == RR_LEAF) {
        fail("affine DAG reference root is leaf");
    }
    unsigned char body_seen[DF_LEAF_BODIES] = {0};
    size_t compose_nodes = 0U;
    size_t intersect_nodes = 0U;
    for (size_t index = 0U; index < graph->count; ++index) {
        struct rr_node *node = &graph->node[index];
        if (node->kind == RR_LEAF) {
            if (node->leaf_index >= DF_LEAF_BODIES) {
                fail("affine DAG reference leaf body invalid");
            }
            body_seen[node->leaf_index] = 1U;
            continue;
        }
        if (
            node->left_id == node->right_id
            || node->left_id == node->id
            || node->right_id == node->id
        ) {
            fail("affine DAG reference operand alias");
        }
        node->left = rr_find(graph, node->left_id);
        node->right = rr_find(graph, node->right_id);
        ++graph->node[node->left].indegree;
        ++graph->node[node->right].indegree;
        if (node->kind == RR_COMPOSE) {
            ++compose_nodes;
        } else {
            ++intersect_nodes;
        }
    }
    for (size_t body = 0U; body < DF_LEAF_BODIES; ++body) {
        if (body_seen[body] == 0U) {
            fail("affine DAG reference omits leaf body");
        }
    }
    if (compose_nodes == 0U || intersect_nodes == 0U) {
        fail("affine DAG reference lacks mixed operators");
    }
    rr_visit(graph, graph->root);
    if (graph->schedule_count != graph->count) {
        fail("affine DAG reference disconnected");
    }
    for (size_t index = 0U; index < graph->count; ++index) {
        const size_t consumers = graph->node[index].indegree;
        if (index == graph->root) {
            if (consumers != 0U) {
                fail("affine DAG reference root has consumer");
            }
        } else if (consumers == 0U || consumers > DF_MAX_FANOUT) {
            fail("affine DAG reference fanout outside bound");
        }
    }
}

static struct rr_program df_make_program(size_t variant) {
    struct rr_program program = {0};
    for (size_t leaf = 0U; leaf < DF_LEAF_BODIES; ++leaf) {
        for (size_t bit = 0U; bit < R_WIDTH; ++bit) {
            const size_t identity[] = {bit};
            rr_set_map_row(
                &program.leaf[leaf], bit, 0U, identity, 1U
            );
        }
    }
    if (variant == 1U) {
        return program;
    }
    const size_t f0[] = {1U};
    const size_t f1[] = {2U};
    const size_t f2[] = {0U};
    rr_set_map_row(&program.leaf[0], 0U, 0U, f0, 1U);
    rr_set_map_row(&program.leaf[0], 1U, 0U, f1, 1U);
    rr_set_map_row(&program.leaf[0], 2U, 1U, f2, 1U);
    const size_t g0[] = {0U, 2U};
    const size_t g1[] = {1U};
    const size_t g2[] = {2U};
    rr_set_map_row(&program.leaf[1], 0U, 0U, g0, 2U);
    rr_set_map_row(&program.leaf[1], 1U, 0U, g1, 1U);
    rr_set_map_row(&program.leaf[1], 2U, 0U, g2, 1U);
    const size_t h0[] = {0U, 2U};
    rr_set_map_row(&program.leaf[2], 0U, 0U, h0, 2U);
    rr_set_map_row(&program.leaf[2], 1U, 0U, g1, 1U);
    rr_set_map_row(&program.leaf[2], 2U, 1U, g2, 1U);
    const size_t k0[] = {0U, 1U, 2U};
    rr_set_map_row(&program.leaf[3], 0U, 0U, k0, 3U);
    rr_set_map_row(&program.leaf[3], 1U, 0U, g1, 1U);
    rr_set_map_row(&program.leaf[3], 2U, 1U, g2, 1U);
    if (variant == 2U) {
        program.leaf[3] = program.leaf[2];
    } else if (variant == 3U) {
        program.leaf[2] = program.leaf[3];
    } else if (variant == 4U) {
        rr_set_map_row(&program.leaf[3], 0U, 1U, h0, 2U);
    } else if (variant != 0U) {
        fail("affine DAG reference variant unknown");
    }
    return program;
}

#ifndef DF_PUBLIC_MAIN
#define DF_PUBLIC_MAIN main
#endif

int DF_PUBLIC_MAIN(int argc, char **argv) {
    if (argc != 2) {
        fail("usage: dag_affine_fanout_reference MANIFEST");
    }
    struct rr_graph graph = {0};
    rr_load(argv[1], &graph);
    df_compile(&graph);
    for (size_t variant = 0U; variant < DF_RUN_VARIANTS; ++variant) {
        const struct rr_program program = df_make_program(variant);
        const struct relation boundary = rr_evaluate(&graph, &program);
        rr_print(variant, &boundary);
    }
    return 0;
}
