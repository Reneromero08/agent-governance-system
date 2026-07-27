#define _POSIX_C_SOURCE 200809L

#include <stdlib.h>
#include <string.h>

/*
 * Independent conventional GF(2) adjudicator for the public recursive
 * affine tree.  It parses and validates the same public nominal topology,
 * but evaluates compact coefficient rows conventionally.  It materializes
 * no assignments, tuples, candidates, or shared-interface witnesses and
 * emits only the complete final root boundary.
 */

#ifndef RR_WIDTH
#define RR_WIDTH 3U
#endif
#define AM_REF_WIDTH RR_WIDTH
#define AM_REF_PUBLIC_MAIN affine_module_reference_embedded_main
#include "affine_module_reference.c"
#undef AM_REF_PUBLIC_MAIN

static void rr_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}
#define fail rr_fail

#ifndef RR_MAX_NODES
#define RR_MAX_NODES 15U
#endif
#define RR_LEAVES 8U
#define RR_VARIANTS 5U
#ifndef RR_RUN_VARIANTS
#define RR_RUN_VARIANTS RR_VARIANTS
#endif

_Static_assert(RR_RUN_VARIANTS >= 2U, "reference primary/reuse required");
_Static_assert(RR_RUN_VARIANTS <= RR_VARIANTS, "too many reference variants");

enum rr_kind {
    RR_LEAF = 0,
    RR_COMPOSE = 1,
    RR_INTERSECT = 2
};

enum rr_port {
    RR_PORT_A = 1,
    RR_PORT_B = 2,
    RR_PORT_C = 3,
    RR_PORT_D = 4,
    RR_PORT_E = 5
};

struct rr_signature {
    enum rr_port domain;
    enum rr_port codomain;
};

struct rr_node {
    unsigned id;
    enum rr_kind kind;
    unsigned left_id;
    unsigned right_id;
    size_t left;
    size_t right;
    size_t leaf_index;
    struct rr_signature signature;
    size_t indegree;
    unsigned char color;
};

struct rr_graph {
    struct rr_node node[RR_MAX_NODES];
    size_t count;
    unsigned root_id;
    size_t root_directives;
    size_t root;
    size_t schedule[RR_MAX_NODES];
    size_t schedule_count;
};

struct rr_program {
    struct relation leaf[RR_LEAVES];
};

static enum rr_port rr_port_from_symbol(char symbol) {
    switch (symbol) {
        case 'A':
            return RR_PORT_A;
        case 'B':
            return RR_PORT_B;
        case 'C':
            return RR_PORT_C;
        case 'D':
            return RR_PORT_D;
        case 'E':
            return RR_PORT_E;
        default:
            fail("recursive reference unknown nominal port");
    }
    return RR_PORT_A;
}

static int rr_signature_equal(
    struct rr_signature left,
    struct rr_signature right
) {
    return left.domain == right.domain
        && left.codomain == right.codomain;
}

static void rr_load(const char *path, struct rr_graph *graph) {
    FILE *input = fopen(path, "r");
    if (input == NULL) {
        fail("recursive reference manifest unavailable");
    }
    char *line = NULL;
    size_t capacity = 0U;
    while (getline(&line, &capacity, input) >= 0) {
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
                fail("recursive reference malformed root");
            }
            graph->root_id = id;
            ++graph->root_directives;
            continue;
        }
        if (graph->count >= RR_MAX_NODES) {
            free(line);
            fclose(input);
            fail("recursive reference node capacity exceeded");
        }
        struct rr_node *node = &graph->node[graph->count];
        node->left = SIZE_MAX;
        node->right = SIZE_MAX;
        node->leaf_index = SIZE_MAX;
        if (strcmp(keyword, "leaf") == 0) {
            unsigned id = 0U;
            unsigned leaf = 0U;
            char domain = '\0';
            char codomain = '\0';
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
                fail("recursive reference malformed leaf");
            }
            node->id = id;
            node->kind = RR_LEAF;
            node->leaf_index = leaf;
            node->signature.domain = rr_port_from_symbol(domain);
            node->signature.codomain = rr_port_from_symbol(codomain);
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
                fail("recursive reference malformed operator");
            }
            node->id = id;
            node->kind = strcmp(keyword, "compose") == 0
                ? RR_COMPOSE
                : RR_INTERSECT;
            node->left_id = left;
            node->right_id = right;
        } else {
            free(line);
            fclose(input);
            fail("recursive reference unknown declaration");
        }
        ++graph->count;
    }
    const int read_failed = ferror(input);
    free(line);
    if (fclose(input) != 0 || read_failed) {
        fail("recursive reference manifest read failed");
    }
}

static size_t rr_find(const struct rr_graph *graph, unsigned id) {
    for (size_t index = 0U; index < graph->count; ++index) {
        if (graph->node[index].id == id) {
            return index;
        }
    }
    fail("recursive reference unknown child");
    return SIZE_MAX;
}

static void rr_visit(struct rr_graph *graph, size_t index) {
    struct rr_node *node = &graph->node[index];
    if (node->color == 1U) {
        fail("recursive reference cycle");
    }
    if (node->color == 2U) {
        return;
    }
    node->color = 1U;
    if (node->kind != RR_LEAF) {
        rr_visit(graph, node->left);
        rr_visit(graph, node->right);
        const struct rr_node *left = &graph->node[node->left];
        const struct rr_node *right = &graph->node[node->right];
        if (node->kind == RR_COMPOSE) {
            if (left->signature.codomain != right->signature.domain) {
                fail("recursive reference composition type mismatch");
            }
            node->signature.domain = left->signature.domain;
            node->signature.codomain = right->signature.codomain;
        } else {
            if (!rr_signature_equal(left->signature, right->signature)) {
                fail("recursive reference intersection type mismatch");
            }
            node->signature = left->signature;
        }
    }
    node->color = 2U;
    graph->schedule[graph->schedule_count++] = index;
}

static void rr_compile(struct rr_graph *graph) {
    if (graph->root_directives != 1U || graph->count == 0U) {
        fail("recursive reference requires one root");
    }
    for (size_t left = 0U; left < graph->count; ++left) {
        for (size_t right = left + 1U; right < graph->count; ++right) {
            if (graph->node[left].id == graph->node[right].id) {
                fail("recursive reference duplicate node id");
            }
        }
    }
    graph->root = rr_find(graph, graph->root_id);
    if (graph->node[graph->root].kind == RR_LEAF) {
        fail("recursive reference root is leaf");
    }
    unsigned char leaf_seen[RR_LEAVES] = {0};
    size_t leaves = 0U;
    size_t compose_nodes = 0U;
    size_t intersect_nodes = 0U;
    for (size_t index = 0U; index < graph->count; ++index) {
        struct rr_node *node = &graph->node[index];
        if (node->kind == RR_LEAF) {
            if (
                node->leaf_index >= RR_LEAVES
                || leaf_seen[node->leaf_index] != 0U
            ) {
                fail("recursive reference leaf owner invalid");
            }
            leaf_seen[node->leaf_index] = 1U;
            ++leaves;
            continue;
        }
        if (
            node->left_id == node->right_id
            || node->left_id == node->id
            || node->right_id == node->id
        ) {
            fail("recursive reference operand alias");
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
    if (
        leaves != RR_LEAVES
        || compose_nodes == 0U
        || intersect_nodes == 0U
    ) {
        fail("recursive reference mixed tree incomplete");
    }
    for (size_t index = 0U; index < graph->count; ++index) {
        const size_t expected = index == graph->root ? 0U : 1U;
        if (graph->node[index].indegree != expected) {
            fail("recursive reference graph is not tree");
        }
    }
    rr_visit(graph, graph->root);
    if (graph->schedule_count != graph->count) {
        fail("recursive reference disconnected node");
    }
}

static void rr_set_map_row(
    struct relation *relation,
    size_t output,
    unsigned char constant,
    const size_t *input,
    size_t inputs
) {
    memset(
        relation->cell + output * R_ROW_CELLS,
        0,
        R_ROW_CELLS * sizeof(*relation->cell)
    );
    begin_equation(relation, output, constant);
    for (size_t term = 0U; term < inputs; ++term) {
        set_term(relation, output, input[term]);
    }
    set_term(relation, output, R_WIDTH + output);
}

static struct rr_program rr_make_program(size_t variant) {
    struct rr_program program = {0};
    for (size_t leaf = 0U; leaf < RR_LEAVES; ++leaf) {
        for (size_t bit = 0U; bit < R_WIDTH; ++bit) {
            const size_t identity[] = {bit};
            rr_set_map_row(
                &program.leaf[leaf], bit, 0U, identity, 1U
            );
        }
    }
    if (variant == 0U) {
        const size_t f0[] = {1U};
        const size_t f1[] = {2U};
        const size_t f2[] = {0U};
        rr_set_map_row(&program.leaf[0], 0U, 0U, f0, 1U);
        rr_set_map_row(&program.leaf[0], 1U, 0U, f1, 1U);
        rr_set_map_row(&program.leaf[0], 2U, 1U, f2, 1U);
        const size_t g0[] = {0U, 2U};
        const size_t g1[] = {0U};
        const size_t g2[] = {0U, 1U};
        rr_set_map_row(&program.leaf[1], 0U, 1U, g0, 2U);
        rr_set_map_row(&program.leaf[1], 1U, 0U, g1, 1U);
        rr_set_map_row(&program.leaf[1], 2U, 0U, g2, 2U);
        const size_t p0[] = {0U, 1U};
        const size_t p1[] = {1U};
        const size_t p2[] = {0U, 1U, 2U};
        rr_set_map_row(&program.leaf[3], 0U, 0U, p0, 2U);
        rr_set_map_row(&program.leaf[3], 1U, 0U, p1, 1U);
        rr_set_map_row(&program.leaf[3], 2U, 1U, p2, 3U);
        const size_t q0[] = {0U, 2U};
        rr_set_map_row(&program.leaf[7], 0U, 0U, q0, 2U);
    } else if (variant == 1U) {
        /* Independent all-identity reuse body. */
    } else if (variant == 2U) {
        const size_t flipped[] = {0U};
        rr_set_map_row(&program.leaf[7], 0U, 1U, flipped, 1U);
    } else if (variant == 3U) {
        const size_t high[] = {0U, R_WIDTH - 1U};
        rr_set_map_row(&program.leaf[7], 0U, 0U, high, 2U);
    } else if (variant == 4U) {
        memset(&program.leaf[0], 0, sizeof(program.leaf[0]));
        program.leaf[0].cell[R_BLOCK_CELLS - 1U] = 1U;
    } else {
        fail("recursive reference variant unknown");
    }
    return program;
}

static struct relation rr_evaluate(
    const struct rr_graph *graph,
    const struct rr_program *program
) {
    struct relation value[RR_MAX_NODES] = {{{0}}};
    for (size_t step = 0U; step < graph->schedule_count; ++step) {
        const size_t index = graph->schedule[step];
        const struct rr_node *node = &graph->node[index];
        if (node->kind == RR_LEAF) {
            value[index] = program->leaf[node->leaf_index];
        } else if (node->kind == RR_COMPOSE) {
            value[index] = compose(
                &value[node->left], &value[node->right]
            );
        } else {
            value[index] = intersect(
                &value[node->left], &value[node->right]
            );
        }
    }
    return value[graph->root];
}

static void rr_print(size_t variant, const struct relation *boundary) {
    const uint64_t hash = hash_bytes(
        UINT64_C(14695981039346656037),
        boundary->cell,
        sizeof(boundary->cell)
    );
    printf(
        "{\"variant\":%zu,\"width\":%u,"
        "\"boundary_coefficients\":[",
        variant,
        (unsigned)R_WIDTH
    );
    for (size_t field = 0U; field < R_BLOCK_CELLS; ++field) {
        if (field != 0U) {
            putchar(',');
        }
        printf("%u", (unsigned)boundary->cell[field]);
    }
    printf(
        "],\"boundary_fnv1a64\":\"%016llx\","
        "\"intermediate_boundaries_emitted\":0,"
        "\"tuple_slots\":0,\"assignment_slots\":0,"
        "\"witness_slots\":0}\n",
        (unsigned long long)hash
    );
}

#ifndef RR_PUBLIC_MAIN
#define RR_PUBLIC_MAIN main
#endif

int RR_PUBLIC_MAIN(int argc, char **argv) {
    if (argc != 2) {
        fail("usage: recursive_affine_module_reference MANIFEST");
    }
    struct rr_graph graph = {0};
    rr_load(argv[1], &graph);
    rr_compile(&graph);
    for (size_t variant = 0U; variant < RR_RUN_VARIANTS; ++variant) {
        const struct rr_program program = rr_make_program(variant);
        const struct relation boundary = rr_evaluate(&graph, &program);
        rr_print(variant, &boundary);
    }
    return 0;
}
