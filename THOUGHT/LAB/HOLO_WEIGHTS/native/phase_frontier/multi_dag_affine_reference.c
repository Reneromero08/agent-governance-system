#define _POSIX_C_SOURCE 200809L

/*
 * Independent conventional GF(2) adjudicator for the nested two-fanout DAG.
 * It shares no phase carrier or native phase implementation and emits only
 * the complete final ROOT relation.
 */

#ifndef MF_REF_WIDTH
#define MF_REF_WIDTH 3U
#endif
#define DF_WIDTH MF_REF_WIDTH
#define DF_LEAF_BODIES 3U
#define DF_PUBLIC_MAIN dag_affine_reference_embedded_main
#include "dag_affine_fanout_reference.c"
#undef DF_PUBLIC_MAIN

#define MF_REF_VARIANTS 5U

static void mf_ref_compile(
    struct rr_graph *graph,
    size_t required_fanout_nodes
) {
    if (graph->root_directives != 1U || graph->count == 0U) {
        fail("multi DAG reference requires one root");
    }
    for (size_t left = 0U; left < graph->count; ++left) {
        for (size_t right = left + 1U; right < graph->count; ++right) {
            if (graph->node[left].id == graph->node[right].id) {
                fail("multi DAG reference duplicate node id");
            }
        }
    }
    graph->root = rr_find(graph, graph->root_id);
    if (graph->node[graph->root].kind == RR_LEAF) {
        fail("multi DAG reference root is leaf");
    }
    unsigned char body_seen[DF_LEAF_BODIES] = {0};
    size_t compose_nodes = 0U;
    size_t intersect_nodes = 0U;
    for (size_t index = 0U; index < graph->count; ++index) {
        struct rr_node *node = &graph->node[index];
        if (node->kind == RR_LEAF) {
            if (node->leaf_index >= DF_LEAF_BODIES) {
                fail("multi DAG reference leaf body invalid");
            }
            body_seen[node->leaf_index] = 1U;
            continue;
        }
        if (
            node->left_id == node->right_id
            || node->left_id == node->id
            || node->right_id == node->id
        ) {
            fail("multi DAG reference operand alias");
        }
        node->left = rr_find(graph, node->left_id);
        node->right = rr_find(graph, node->right_id);
        ++graph->node[node->left].indegree;
        ++graph->node[node->right].indegree;
        if (node->kind == RR_COMPOSE) {
            ++compose_nodes;
        } else if (node->kind == RR_INTERSECT) {
            ++intersect_nodes;
        } else {
            fail("multi DAG reference operator invalid");
        }
    }
    for (size_t body = 0U; body < DF_LEAF_BODIES; ++body) {
        if (body_seen[body] == 0U) {
            fail("multi DAG reference omits leaf body");
        }
    }
    if (compose_nodes == 0U || intersect_nodes == 0U) {
        fail("multi DAG reference lacks mixed operators");
    }
    rr_visit(graph, graph->root);
    if (graph->schedule_count != graph->count) {
        fail("multi DAG reference disconnected");
    }
    size_t fanout_nodes = 0U;
    size_t maximum_fanout = 0U;
    for (size_t index = 0U; index < graph->count; ++index) {
        const size_t consumers = graph->node[index].indegree;
        if (index == graph->root) {
            if (consumers != 0U) {
                fail("multi DAG reference root has consumer");
            }
            continue;
        }
        if (consumers == 0U || consumers > 2U) {
            fail("multi DAG reference fanout outside bound");
        }
        if (consumers > maximum_fanout) {
            maximum_fanout = consumers;
        }
        if (consumers > 1U) {
            if (graph->node[index].kind == RR_LEAF) {
                fail("multi DAG reference leaf fanout");
            }
            ++fanout_nodes;
        }
    }
    if (
        fanout_nodes != required_fanout_nodes
        || (
            required_fanout_nodes > 0U
            && maximum_fanout != 2U
        )
    ) {
        fail("multi DAG reference shared-node count mismatch");
    }
}

static struct rr_program mf_ref_program(size_t variant) {
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
    const size_t bit0[] = {0U};
    const size_t bit1[] = {1U};
    const size_t bit2[] = {2U};
    const size_t bit01[] = {0U, 1U};
    const size_t bit02[] = {0U, 2U};
    rr_set_map_row(&program.leaf[0], 0U, 0U, bit1, 1U);
    rr_set_map_row(&program.leaf[0], 1U, 0U, bit0, 1U);
    rr_set_map_row(&program.leaf[0], 2U, 0U, bit2, 1U);
    rr_set_map_row(&program.leaf[1], 0U, 0U, bit01, 2U);
    rr_set_map_row(&program.leaf[1], 1U, 0U, bit0, 1U);
    rr_set_map_row(&program.leaf[1], 2U, 0U, bit2, 1U);
    rr_set_map_row(&program.leaf[2], 0U, 0U, bit02, 2U);
    rr_set_map_row(&program.leaf[2], 1U, 0U, bit1, 1U);
    rr_set_map_row(&program.leaf[2], 2U, 0U, bit2, 1U);
    if (variant == 2U) {
        for (size_t bit = 0U; bit < R_WIDTH; ++bit) {
            const size_t identity[] = {bit};
            rr_set_map_row(
                &program.leaf[2], bit, 0U, identity, 1U
            );
        }
    } else if (variant == 3U) {
        rr_set_map_row(&program.leaf[1], 0U, 0U, bit0, 1U);
        rr_set_map_row(&program.leaf[1], 1U, 0U, bit2, 1U);
        rr_set_map_row(&program.leaf[1], 2U, 0U, bit1, 1U);
    } else if (variant == 4U) {
        for (size_t bit = 0U; bit < R_WIDTH; ++bit) {
            const size_t identity[] = {bit};
            rr_set_map_row(
                &program.leaf[2],
                bit,
                bit == 0U ? 1U : 0U,
                identity,
                1U
            );
        }
    } else if (variant != 0U) {
        fail("multi DAG reference variant unknown");
    }
    return program;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fail(
            "usage: multi_dag_affine_reference "
            "MANIFEST EXPECTED_FANOUT_NODES"
        );
    }
    char *end = NULL;
    const unsigned long expected = strtoul(argv[2], &end, 10);
    if (*argv[2] == '\0' || end == NULL || *end != '\0' || expected > 2U) {
        fail("multi DAG reference expected fanout invalid");
    }
    struct rr_graph graph = {0};
    rr_load(argv[1], &graph);
    mf_ref_compile(&graph, (size_t)expected);
    for (size_t variant = 0U; variant < MF_REF_VARIANTS; ++variant) {
        const struct rr_program program = mf_ref_program(variant);
        const struct relation boundary = rr_evaluate(&graph, &program);
        rr_print(variant, &boundary);
    }
    return 0;
}
