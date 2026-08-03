#define _POSIX_C_SOURCE 200809L

/*
 * Independent conventional GF(2) adjudicator for the bounded four-owner DAG.
 * It shares no phase carrier or native phase implementation and emits only
 * complete final ROOT relations.
 */

#ifndef GM_REF_WIDTH
#define GM_REF_WIDTH 3U
#endif
#define RR_MAX_NODES 15U
#define DF_WIDTH GM_REF_WIDTH
#define DF_LEAF_BODIES 4U
#define DF_MAX_FANOUT 4U
#define DF_PUBLIC_MAIN general_multi_dag_reference_embedded_main
#include "dag_affine_fanout_reference.c"
#undef DF_PUBLIC_MAIN

#define GM_REF_VARIANTS 5U

static struct rr_program gm_ref_program(size_t variant) {
    struct rr_program program = df_make_program(1U);
    static const unsigned char row[GM_REF_VARIANTS][4U][3U] = {
        {{2U,1U,4U},{0U,7U,4U},{4U,2U,1U},{6U,5U,1U}},
        {{2U,1U,4U},{2U,1U,4U},{4U,2U,1U},{4U,2U,1U}},
        {{2U,1U,4U},{0U,5U,0U},{4U,2U,1U},{2U,2U,5U}},
        {{2U,1U,4U},{0U,0U,7U},{4U,2U,1U},{7U,5U,1U}},
        {{2U,1U,4U},{4U,1U,6U},{4U,2U,1U},{3U,4U,7U}}
    };
    static const unsigned char constant[GM_REF_VARIANTS][4U] = {
        {0U,0U,0U,3U},
        {0U,0U,0U,0U},
        {0U,5U,0U,7U},
        {0U,2U,0U,4U},
        {0U,0U,0U,0U}
    };
    if (variant >= GM_REF_VARIANTS) {
        fail("general DAG reference variant unknown");
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
            rr_set_map_row(
                &program.leaf[leaf],
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

static size_t gm_ref_find(
    const struct rr_graph *graph,
    unsigned public_id
) {
    for (size_t index = 0U; index < graph->count; ++index) {
        if (graph->node[index].id == public_id) {
            return index;
        }
    }
    fail("general DAG reference public node unavailable");
    return 0U;
}

static void gm_ref_replace_edge(
    struct rr_graph *graph,
    unsigned producer_id,
    unsigned consumer_id,
    char side,
    unsigned replacement_id
) {
    (void)gm_ref_find(graph, producer_id);
    const size_t consumer = gm_ref_find(graph, consumer_id);
    (void)gm_ref_find(graph, replacement_id);
    unsigned *edge = side == 'L'
        ? &graph->node[consumer].left_id
        : side == 'R'
            ? &graph->node[consumer].right_id
            : NULL;
    if (edge == NULL || *edge != producer_id) {
        fail("general DAG reference edge replacement invalid");
    }
    *edge = replacement_id;
}

int main(int argc, char **argv) {
    if (
        argc != 2
        && !(
            (argc == 7 || argc == 8)
            && strcmp(argv[2], "--replace-edge") == 0
        )
    ) {
        fail(
            "usage: general_multi_dag_affine_reference MANIFEST "
            "[--replace-edge PRODUCER CONSUMER SIDE REPLACEMENT "
            "[VARIANT]]"
        );
    }
    struct rr_graph graph = {0};
    rr_load(argv[1], &graph);
    unsigned control_producer = 0U;
    unsigned control_consumer = 0U;
    unsigned control_replacement = 0U;
    char control_side = '\0';
    if (argc >= 7) {
        char *producer_end = NULL;
        char *consumer_end = NULL;
        char *replacement_end = NULL;
        const unsigned long producer =
            strtoul(argv[3], &producer_end, 10);
        const unsigned long consumer =
            strtoul(argv[4], &consumer_end, 10);
        const unsigned long replacement =
            strtoul(argv[6], &replacement_end, 10);
        if (
            producer_end == NULL
            || consumer_end == NULL
            || replacement_end == NULL
            || *producer_end != '\0'
            || *consumer_end != '\0'
            || *replacement_end != '\0'
            || strlen(argv[5]) != 1U
        ) {
            fail("general DAG reference edge replacement malformed");
        }
        control_producer = (unsigned)producer;
        control_consumer = (unsigned)consumer;
        control_side = argv[5][0];
        control_replacement = (unsigned)replacement;
    }
    if (argc >= 7) {
        gm_ref_replace_edge(
            &graph,
            control_producer,
            control_consumer,
            control_side,
            control_replacement
        );
    }
    /*
     * Compile only after the control mutation.  This makes every semantic
     * control pass the same alias, cycle, connectivity, and fanout laws as a
     * public accepted graph rather than mutating resolved indices behind the
     * compiler.
     */
    df_compile(&graph);
    const unsigned shared_id[] = {805U, 806U, 807U, 808U};
    const size_t shared_degree[] = {4U, 3U, 3U, 2U};
    size_t fanout_nodes = 0U;
    size_t maximum_fanout = 0U;
    for (size_t index = 0U; index < graph.count; ++index) {
        if (graph.node[index].indegree > 1U) {
            ++fanout_nodes;
        }
        if (graph.node[index].indegree > maximum_fanout) {
            maximum_fanout = graph.node[index].indegree;
        }
    }
    if (graph.count != 15U) {
        fail("general DAG reference topology law mismatch");
    }
    if (argc == 2) {
        if (fanout_nodes != 4U || maximum_fanout != 4U) {
            fail("general DAG reference topology law mismatch");
        }
        for (
            size_t owner = 0U;
            owner < sizeof(shared_id) / sizeof(shared_id[0]);
            ++owner
        ) {
            const size_t index = gm_ref_find(&graph, shared_id[owner]);
            if (graph.node[index].indegree != shared_degree[owner]) {
                fail("general DAG reference shared degree mismatch");
            }
        }
    }
    size_t first_variant = 0U;
    size_t variants = argc == 2 ? GM_REF_VARIANTS : 1U;
    if (argc == 8) {
        char *variant_end = NULL;
        const unsigned long parsed =
            strtoul(argv[7], &variant_end, 10);
        if (
            variant_end == NULL
            || *variant_end != '\0'
            || parsed >= GM_REF_VARIANTS
        ) {
            fail("general DAG reference variant control invalid");
        }
        first_variant = (size_t)parsed;
    }
    for (
        size_t offset = 0U;
        offset < variants;
        ++offset
    ) {
        const size_t variant = first_variant + offset;
        const struct rr_program program = gm_ref_program(variant);
        const struct relation boundary = rr_evaluate(&graph, &program);
        rr_print(variant, &boundary);
    }
    return 0;
}
