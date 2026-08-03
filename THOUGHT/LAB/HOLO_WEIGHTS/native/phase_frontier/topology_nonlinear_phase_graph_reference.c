#define _POSIX_C_SOURCE 200809L

/*
 * Independent matched compact reference for the public nonlinear phase graph.
 * It stores one double angle per public cell and has no catalytic carrier,
 * hidden custody, inverse execution, restoration, or reuse.
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPR_MAX_WIDTH 32U
#define NPR_MAX_EDGES 64U
#define NPR_HASH_SCALE 1000000000.0

enum npr_program {
    NPR_PRIMARY = 0,
    NPR_REUSE = 1
};

struct npr_edge {
    size_t id;
    size_t target;
    size_t source;
    int strength_milli;
};

struct npr_graph {
    size_t width;
    size_t output[2];
    size_t count;
    struct npr_edge edge[NPR_MAX_EDGES];
    unsigned char dependency[NPR_MAX_EDGES][NPR_MAX_EDGES];
};

static void npr_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static size_t npr_find(
    const struct npr_graph *graph,
    size_t id
) {
    for (size_t index = 0U; index < graph->count; ++index) {
        if (graph->edge[index].id == id) {
            return index;
        }
    }
    return SIZE_MAX;
}

static struct npr_graph npr_load(const char *path) {
    FILE *stream = fopen(path, "r");
    if (stream == NULL) {
        npr_fail("reference cannot open topology");
    }
    struct npr_graph graph = {0};
    char line[160];
    int header = 0;
    int width = 0;
    int outputs = 0;
    int dependencies = 0;
    int end = 0;
    while (fgets(line, sizeof(line), stream) != NULL) {
        char extra = '\0';
        if (strcmp(line, "NONLINEAR_PHASE_GRAPH 1\n") == 0) {
            if (header) {
                npr_fail("reference duplicate header");
            }
            header = 1;
            continue;
        }
        size_t first = 0U;
        size_t second = 0U;
        if (sscanf(line, "WIDTH %zu %c", &first, &extra) == 1) {
            if (
                !header || width || first < 3U || first > NPR_MAX_WIDTH
            ) {
                npr_fail("reference width invalid");
            }
            graph.width = first;
            width = 1;
            continue;
        }
        if (
            sscanf(line, "OUTPUTS %zu %zu %c", &first, &second, &extra)
                == 2
        ) {
            if (
                !width
                || outputs
                || first >= graph.width
                || second >= graph.width
                || first == second
            ) {
                npr_fail("reference outputs invalid");
            }
            graph.output[0] = first;
            graph.output[1] = second;
            outputs = 1;
            continue;
        }
        size_t id = 0U;
        int strength = 0;
        if (
            sscanf(
                line,
                "SHEAR %zu %zu %zu %d %c",
                &id,
                &first,
                &second,
                &strength,
                &extra
            ) == 4
        ) {
            if (
                !outputs
                || dependencies
                || graph.count >= NPR_MAX_EDGES
                || id == 0U
                || npr_find(&graph, id) != SIZE_MAX
                || first >= graph.width
                || second >= graph.width
                || first == second
                || strength == 0
            ) {
                npr_fail("reference shear invalid");
            }
            graph.edge[graph.count] = (struct npr_edge){
                .id = id,
                .target = first,
                .source = second,
                .strength_milli = strength
            };
            ++graph.count;
            continue;
        }
        if (
            sscanf(line, "AFTER %zu %zu %c", &first, &second, &extra)
                == 2
        ) {
            dependencies = 1;
            const size_t dependent = npr_find(&graph, first);
            const size_t prerequisite = npr_find(&graph, second);
            if (
                dependent == SIZE_MAX
                || prerequisite == SIZE_MAX
                || dependent == prerequisite
                || graph.dependency[dependent][prerequisite]
            ) {
                npr_fail("reference dependency invalid");
            }
            graph.dependency[dependent][prerequisite] = 1U;
            continue;
        }
        if (strcmp(line, "END\n") == 0 || strcmp(line, "END") == 0) {
            end = 1;
            break;
        }
        npr_fail("reference topology syntax invalid");
    }
    if (
        fclose(stream) != 0
        || !header
        || !width
        || !outputs
        || !end
        || graph.count < 2U
    ) {
        npr_fail("reference topology incomplete");
    }

    struct npr_edge scheduled[NPR_MAX_EDGES] = {{0}};
    unsigned char emitted[NPR_MAX_EDGES] = {0};
    for (size_t cursor = 0U; cursor < graph.count; ++cursor) {
        size_t selected = SIZE_MAX;
        for (size_t candidate = 0U; candidate < graph.count; ++candidate) {
            if (emitted[candidate]) {
                continue;
            }
            int ready = 1;
            for (
                size_t prerequisite = 0U;
                prerequisite < graph.count;
                ++prerequisite
            ) {
                if (
                    graph.dependency[candidate][prerequisite]
                    && !emitted[prerequisite]
                ) {
                    ready = 0;
                    break;
                }
            }
            if (
                ready
                && (
                    selected == SIZE_MAX
                    || graph.edge[candidate].id
                        < graph.edge[selected].id
                )
            ) {
                selected = candidate;
            }
        }
        if (selected == SIZE_MAX) {
            npr_fail("reference topology cyclic");
        }
        emitted[selected] = 1U;
        scheduled[cursor] = graph.edge[selected];
    }
    memcpy(graph.edge, scheduled, sizeof(scheduled));
    return graph;
}

static size_t npr_rounds(const char *text) {
    if (text == NULL || text[0] < '1' || text[0] > '9') {
        npr_fail("reference rounds invalid");
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            npr_fail("reference rounds invalid");
        }
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long parsed = strtoul(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed < 1U
        || parsed > 4096U
    ) {
        npr_fail("reference rounds invalid");
    }
    return (size_t)parsed;
}

static double npr_initial(
    enum npr_program program,
    size_t cell
) {
    const size_t residue = (
        (program == NPR_PRIMARY ? 7U : 11U) * cell
        + (program == NPR_PRIMARY ? 3U : 5U)
    ) % 23U;
    return 0.071 * (double)(residue + 1U)
        - (program == NPR_PRIMARY ? 0.91 : 1.17);
}

static void npr_execute(
    const struct npr_graph *graph,
    size_t rounds,
    enum npr_program program,
    double angle[NPR_MAX_WIDTH]
) {
    for (size_t cell = 0U; cell < graph->width; ++cell) {
        angle[cell] = npr_initial(program, cell);
    }
    for (size_t round = 0U; round < rounds; ++round) {
        const double round_scale =
            1.0 + 0.03125 * (double)(round % 5U);
        for (size_t edge = 0U; edge < graph->count; ++edge) {
            const struct npr_edge *operation = &graph->edge[edge];
            angle[operation->target] +=
                (double)operation->strength_milli
                * 0.001
                * round_scale
                * sin(angle[operation->source]);
        }
    }
}

static uint64_t npr_hash_word(uint64_t hash, uint64_t value) {
    for (size_t byte = 0U; byte < sizeof(value); ++byte) {
        hash ^= (value >> (8U * byte)) & UINT64_C(255);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t npr_hash(
    const struct npr_graph *graph,
    const double angle[NPR_MAX_WIDTH]
) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t output = 0U; output < 2U; ++output) {
        const size_t cell = graph->output[output];
        const int64_t real = (int64_t)llround(
            cos(angle[cell]) * NPR_HASH_SCALE
        );
        const int64_t imaginary = (int64_t)llround(
            sin(angle[cell]) * NPR_HASH_SCALE
        );
        hash = npr_hash_word(hash, (uint64_t)real);
        hash = npr_hash_word(hash, (uint64_t)imaginary);
    }
    return hash;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        npr_fail(
            "usage: topology_nonlinear_phase_graph_reference "
            "TOPOLOGY ROUNDS"
        );
    }
    const struct npr_graph graph = npr_load(argv[1]);
    const size_t rounds = npr_rounds(argv[2]);
    double primary[NPR_MAX_WIDTH] = {0};
    double reuse[NPR_MAX_WIDTH] = {0};
    npr_execute(&graph, rounds, NPR_PRIMARY, primary);
    npr_execute(&graph, rounds, NPR_REUSE, reuse);
    printf(
        "{\"result\":\"PASS\",\"width\":%zu,\"rounds\":%zu,"
        "\"edges\":%zu,\"classical_state_bytes\":%zu,"
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"primary_interference_probability\":%.17g,"
        "\"materialized_phase_paths\":0}\n",
        graph.width,
        rounds,
        graph.count,
        graph.width * sizeof(double),
        (unsigned long long)npr_hash(&graph, primary),
        (unsigned long long)npr_hash(&graph, reuse),
        0.5 * (
            1.0 + cos(
                primary[graph.output[0]]
                - primary[graph.output[1]]
            )
        )
    );
    return 0;
}
