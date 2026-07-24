#define _POSIX_C_SOURCE 200809L

/*
 * Independent scalar adjudicator for the public series/parallel relation
 * language.  This program contains no complex numbers or carrier operations.
 * It reduces the declared graph in F3 and, for bounded graphs, separately
 * enumerates every complete Boolean assignment to check the projected
 * two-terminal relation.
 */

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NODES 48U
#define MAX_RELATIONS 96U
#define MAX_ELIMINATIONS 46U
#define MAX_EDGES 256U
#define MAX_IDENTIFIER 31U
#define LINE_CAPACITY 512U
#define TOKEN_CAPACITY 10U
#define ENUMERATION_INTERNAL_LIMIT 20U

enum vertex_kind { EXTERNAL_VERTEX = 0, INTERNAL_VERTEX = 1 };

struct vertex {
    char name[MAX_IDENTIFIER + 1U];
    enum vertex_kind kind;
};

struct relation {
    char name[MAX_IDENTIFIER + 1U];
    size_t first;
    size_t second;
    int coefficient[4];
};

struct specification {
    struct vertex vertex[MAX_NODES];
    struct relation relation[MAX_RELATIONS];
    size_t elimination[MAX_ELIMINATIONS];
    size_t external[2];
    size_t vertex_count;
    size_t relation_count;
    size_t elimination_count;
    uint64_t source_hash;
};

struct edge {
    size_t first;
    size_t second;
    int coefficient[4];
    int active;
};

struct reduction {
    int coefficient[4];
    size_t compositions;
    size_t intersections;
};

static void die(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void die_line(const char *message, size_t line) {
    fprintf(stderr, "%s at line %zu\n", message, line);
    exit(2);
}

static int mod3(int value) {
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static uint64_t hash_bytes(
    uint64_t hash,
    const unsigned char *bytes,
    size_t count
) {
    for (size_t index = 0U; index < count; ++index) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int valid_identifier(const char *text) {
    const size_t length = strlen(text);
    if (
        length == 0U
        || length > MAX_IDENTIFIER
        || text[0] < 'A'
        || text[0] > 'Z'
    ) {
        return 0;
    }
    for (size_t index = 1U; index < length; ++index) {
        const unsigned char byte = (unsigned char)text[index];
        if (!(isupper(byte) || isdigit(byte) || byte == '_')) {
            return 0;
        }
    }
    return 1;
}

static size_t tokenize(
    char *line,
    char *tokens[TOKEN_CAPACITY]
) {
    size_t count = 0U;
    char *state = NULL;
    for (
        char *token = strtok_r(line, " ", &state);
        token != NULL;
        token = strtok_r(NULL, " ", &state)
    ) {
        if (count == TOKEN_CAPACITY) {
            die("too many tokens");
        }
        tokens[count++] = token;
    }
    return count;
}

static size_t lookup_vertex(
    const struct specification *specification,
    const char *name,
    size_t line
) {
    for (
        size_t index = 0U;
        index < specification->vertex_count;
        ++index
    ) {
        if (strcmp(specification->vertex[index].name, name) == 0) {
            return index;
        }
    }
    die_line("undeclared vertex", line);
    return 0U;
}

static struct specification read_specification(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct specification specification = {
        .source_hash = UINT64_C(14695981039346656037)
    };
    char line[LINE_CAPACITY];
    size_t line_number = 0U;
    int header_seen = 0;
    int type_seen = 0;
    int relation_seen = 0;
    int elimination_seen = 0;
    int end_seen = 0;
    while (fgets(line, sizeof(line), stream) != NULL) {
        ++line_number;
        const size_t length = strlen(line);
        specification.source_hash = hash_bytes(
            specification.source_hash,
            (const unsigned char *)line,
            length
        );
        if (length == 0U || line[length - 1U] != '\n') {
            die_line("every record must end with LF", line_number);
        }
        if (memchr(line, '\r', length) != NULL) {
            die_line("CR bytes are forbidden", line_number);
        }
        line[length - 1U] = '\0';
        if (end_seen) {
            die_line("record after END", line_number);
        }
        char *tokens[TOKEN_CAPACITY] = {0};
        const size_t count = tokenize(line, tokens);
        if (count == 0U) {
            die_line("blank records are forbidden", line_number);
        }
        if (!header_seen) {
            if (
                count != 2U
                || strcmp(
                    tokens[0],
                    "CATCAS_SERIES_PARALLEL_RELATION"
                ) != 0
                || strcmp(tokens[1], "1") != 0
            ) {
                die_line("invalid header", line_number);
            }
            header_seen = 1;
        } else if (strcmp(tokens[0], "TYPE") == 0) {
            if (
                type_seen
                || count != 2U
                || strcmp(tokens[1], "BOOLEAN_F3") != 0
            ) {
                die_line("invalid TYPE", line_number);
            }
            type_seen = 1;
        } else if (strcmp(tokens[0], "NODE") == 0) {
            if (!type_seen || relation_seen || count != 3U) {
                die_line("invalid NODE", line_number);
            }
            if (
                specification.vertex_count == MAX_NODES
                || !valid_identifier(tokens[1])
            ) {
                die_line("invalid vertex identifier", line_number);
            }
            for (
                size_t index = 0U;
                index < specification.vertex_count;
                ++index
            ) {
                if (
                    strcmp(
                        specification.vertex[index].name,
                        tokens[1]
                    ) == 0
                ) {
                    die_line("duplicate vertex identifier", line_number);
                }
            }
            struct vertex *vertex =
                &specification.vertex[specification.vertex_count++];
            memcpy(
                vertex->name,
                tokens[1],
                strlen(tokens[1]) + 1U
            );
            if (strcmp(tokens[2], "EXTERNAL") == 0) {
                vertex->kind = EXTERNAL_VERTEX;
            } else if (strcmp(tokens[2], "INTERNAL") == 0) {
                vertex->kind = INTERNAL_VERTEX;
            } else {
                die_line("unknown vertex kind", line_number);
            }
        } else if (strcmp(tokens[0], "RELATION") == 0) {
            if (
                !type_seen
                || specification.vertex_count == 0U
                || elimination_seen
                || count != 8U
                || specification.relation_count == MAX_RELATIONS
                || !valid_identifier(tokens[1])
            ) {
                die_line("invalid RELATION", line_number);
            }
            for (
                size_t index = 0U;
                index < specification.relation_count;
                ++index
            ) {
                if (
                    strcmp(
                        specification.relation[index].name,
                        tokens[1]
                    ) == 0
                ) {
                    die_line(
                        "duplicate relation identifier",
                        line_number
                    );
                }
            }
            relation_seen = 1;
            struct relation *relation =
                &specification.relation[
                    specification.relation_count++
                ];
            memcpy(
                relation->name,
                tokens[1],
                strlen(tokens[1]) + 1U
            );
            relation->first = lookup_vertex(
                &specification,
                tokens[2],
                line_number
            );
            relation->second = lookup_vertex(
                &specification,
                tokens[3],
                line_number
            );
            if (relation->first == relation->second) {
                die_line("self relation is forbidden", line_number);
            }
            for (size_t index = 0U; index < 4U; ++index) {
                if (
                    strlen(tokens[4U + index]) != 1U
                    || tokens[4U + index][0] < '0'
                    || tokens[4U + index][0] > '2'
                ) {
                    die_line("invalid F3 coefficient", line_number);
                }
                relation->coefficient[index] =
                    tokens[4U + index][0] - '0';
            }
        } else if (strcmp(tokens[0], "ELIMINATE") == 0) {
            if (
                !relation_seen
                || count != 2U
                || specification.elimination_count == MAX_ELIMINATIONS
            ) {
                die_line("invalid ELIMINATE", line_number);
            }
            elimination_seen = 1;
            specification.elimination[
                specification.elimination_count++
            ] = lookup_vertex(
                &specification,
                tokens[1],
                line_number
            );
        } else if (strcmp(tokens[0], "END") == 0) {
            if (count != 1U) {
                die_line("invalid END", line_number);
            }
            end_seen = 1;
        } else {
            die_line("unknown record", line_number);
        }
    }
    if (ferror(stream) || fclose(stream) != 0) {
        die("failed to read specification");
    }
    if (
        !header_seen
        || !type_seen
        || !relation_seen
        || !elimination_seen
        || !end_seen
    ) {
        die("specification is incomplete");
    }
    size_t external_count = 0U;
    for (
        size_t index = 0U;
        index < specification.vertex_count;
        ++index
    ) {
        if (
            specification.vertex[index].kind == EXTERNAL_VERTEX
        ) {
            if (external_count == 2U) {
                die("exactly two external vertices are required");
            }
            specification.external[external_count++] = index;
        }
    }
    if (external_count != 2U) {
        die("exactly two external vertices are required");
    }
    return specification;
}

static int evaluate(
    const int coefficient[4],
    int first,
    int second
) {
    return mod3(
        coefficient[0]
        + coefficient[1] * first
        + coefficient[2] * second
        + coefficient[3] * first * second
    );
}

static void interpolate(
    const int value[4],
    int coefficient[4]
) {
    coefficient[0] = mod3(value[0]);
    coefficient[1] = mod3(value[1] - value[0]);
    coefficient[2] = mod3(value[2] - value[0]);
    coefficient[3] = mod3(
        value[3] - value[1] - value[2] + value[0]
    );
}

static int evaluate_oriented(
    const struct edge *edge,
    size_t first,
    size_t second,
    int first_value,
    int second_value
) {
    if (edge->first == first && edge->second == second) {
        return evaluate(
            edge->coefficient,
            first_value,
            second_value
        );
    }
    if (edge->first == second && edge->second == first) {
        return evaluate(
            edge->coefficient,
            second_value,
            first_value
        );
    }
    die("edge orientation mismatch");
    return 0;
}

static void intersect_edges(
    const struct edge *left,
    const struct edge *right,
    size_t first,
    size_t second,
    int output[4]
) {
    int value[4] = {0};
    for (int second_value = 0; second_value < 2; ++second_value) {
        for (int first_value = 0; first_value < 2; ++first_value) {
            const size_t row =
                (size_t)first_value + 2U * (size_t)second_value;
            const int left_value = evaluate_oriented(
                left,
                first,
                second,
                first_value,
                second_value
            );
            const int right_value = evaluate_oriented(
                right,
                first,
                second,
                first_value,
                second_value
            );
            value[row] = mod3(
                left_value * left_value
                + right_value * right_value
            );
        }
    }
    interpolate(value, output);
}

static void compose_edges(
    const struct edge *left,
    const struct edge *right,
    size_t first,
    size_t middle,
    size_t second,
    int output[4]
) {
    int value[4] = {0};
    for (int second_value = 0; second_value < 2; ++second_value) {
        for (int first_value = 0; first_value < 2; ++first_value) {
            int product = 1;
            for (int middle_value = 0; middle_value < 2; ++middle_value) {
                const int left_value = evaluate_oriented(
                    left,
                    first,
                    middle,
                    first_value,
                    middle_value
                );
                const int right_value = evaluate_oriented(
                    right,
                    middle,
                    second,
                    middle_value,
                    second_value
                );
                product = mod3(
                    product * mod3(
                        left_value * left_value
                        + right_value * right_value
                    )
                );
            }
            value[
                (size_t)first_value + 2U * (size_t)second_value
            ] = product;
        }
    }
    interpolate(value, output);
}

static int edge_matches(
    const struct edge *edge,
    size_t first,
    size_t second
) {
    return edge->active && (
        (edge->first == first && edge->second == second)
        || (edge->first == second && edge->second == first)
    );
}

static size_t append_edge(
    struct edge edges[MAX_EDGES],
    size_t *edge_count,
    size_t first,
    size_t second,
    const int coefficient[4]
) {
    if (*edge_count == MAX_EDGES) {
        die("edge capacity exceeded");
    }
    const size_t index = (*edge_count)++;
    edges[index].first = first;
    edges[index].second = second;
    memcpy(
        edges[index].coefficient,
        coefficient,
        sizeof(edges[index].coefficient)
    );
    edges[index].active = 1;
    return index;
}

static size_t merge_parallel(
    struct edge edges[MAX_EDGES],
    size_t *edge_count,
    size_t current,
    size_t *intersection_count
) {
    for (;;) {
        size_t parallel = SIZE_MAX;
        for (size_t index = 0U; index < *edge_count; ++index) {
            if (
                index != current
                && edge_matches(
                    &edges[index],
                    edges[current].first,
                    edges[current].second
                )
            ) {
                parallel = index;
                break;
            }
        }
        if (parallel == SIZE_MAX) {
            return current;
        }
        int coefficient[4];
        intersect_edges(
            &edges[current],
            &edges[parallel],
            edges[current].first,
            edges[current].second,
            coefficient
        );
        const size_t first = edges[current].first;
        const size_t second = edges[current].second;
        edges[current].active = 0;
        edges[parallel].active = 0;
        current = append_edge(
            edges,
            edge_count,
            first,
            second,
            coefficient
        );
        ++*intersection_count;
    }
}

static size_t other_vertex(
    const struct edge *edge,
    size_t vertex
) {
    if (edge->first == vertex) {
        return edge->second;
    }
    if (edge->second == vertex) {
        return edge->first;
    }
    die("edge is not incident to vertex");
    return 0U;
}

static struct reduction reduce_specification(
    const struct specification *specification
) {
    struct edge edges[MAX_EDGES] = {{0}};
    size_t edge_count = 0U;
    struct reduction reduction = {{0}, 0U, 0U};
    for (
        size_t index = 0U;
        index < specification->relation_count;
        ++index
    ) {
        const struct relation *relation =
            &specification->relation[index];
        size_t current = append_edge(
            edges,
            &edge_count,
            relation->first,
            relation->second,
            relation->coefficient
        );
        current = merge_parallel(
            edges,
            &edge_count,
            current,
            &reduction.intersections
        );
        (void)current;
    }
    int eliminated[MAX_NODES] = {0};
    for (
        size_t order = 0U;
        order < specification->elimination_count;
        ++order
    ) {
        const size_t middle = specification->elimination[order];
        if (
            specification->vertex[middle].kind != INTERNAL_VERTEX
            || eliminated[middle]
        ) {
            die("invalid elimination vertex");
        }
        eliminated[middle] = 1;
        size_t incident[2] = {SIZE_MAX, SIZE_MAX};
        size_t incident_count = 0U;
        for (size_t index = 0U; index < edge_count; ++index) {
            if (
                edges[index].active
                && (
                    edges[index].first == middle
                    || edges[index].second == middle
                )
            ) {
                if (incident_count == 2U) {
                    die("elimination vertex is not degree two");
                }
                incident[incident_count++] = index;
            }
        }
        if (incident_count != 2U) {
            die("elimination vertex is not degree two");
        }
        size_t first = other_vertex(&edges[incident[0]], middle);
        size_t second = other_vertex(&edges[incident[1]], middle);
        if (first == second) {
            die("elimination creates self relation");
        }
        if (second < first) {
            const size_t swap_vertex = first;
            const size_t swap_edge = incident[0];
            first = second;
            second = swap_vertex;
            incident[0] = incident[1];
            incident[1] = swap_edge;
        }
        int coefficient[4];
        compose_edges(
            &edges[incident[0]],
            &edges[incident[1]],
            first,
            middle,
            second,
            coefficient
        );
        edges[incident[0]].active = 0;
        edges[incident[1]].active = 0;
        size_t current = append_edge(
            edges,
            &edge_count,
            first,
            second,
            coefficient
        );
        ++reduction.compositions;
        current = merge_parallel(
            edges,
            &edge_count,
            current,
            &reduction.intersections
        );
        (void)current;
    }
    for (
        size_t index = 0U;
        index < specification->vertex_count;
        ++index
    ) {
        if (
            specification->vertex[index].kind == INTERNAL_VERTEX
            && !eliminated[index]
        ) {
            die("not every internal vertex was eliminated");
        }
    }
    size_t final = SIZE_MAX;
    size_t active_count = 0U;
    for (size_t index = 0U; index < edge_count; ++index) {
        if (edges[index].active) {
            ++active_count;
            final = index;
        }
    }
    if (
        active_count != 1U
        || !edge_matches(
            &edges[final],
            specification->external[0],
            specification->external[1]
        )
        || reduction.intersections == 0U
    ) {
        die("graph did not reduce to one parallel-bearing boundary");
    }
    int final_values[4] = {0};
    for (int second_value = 0; second_value < 2; ++second_value) {
        for (int first_value = 0; first_value < 2; ++first_value) {
            const size_t row =
                (size_t)first_value + 2U * (size_t)second_value;
            final_values[row] = evaluate_oriented(
                &edges[final],
                specification->external[0],
                specification->external[1],
                first_value,
                second_value
            );
        }
    }
    interpolate(final_values, reduction.coefficient);
    return reduction;
}

static unsigned zero_mask(const int coefficient[4]) {
    unsigned mask = 0U;
    for (int second = 0; second < 2; ++second) {
        for (int first = 0; first < 2; ++first) {
            if (evaluate(coefficient, first, second) == 0) {
                mask |= 1U << (
                    (unsigned)first + 2U * (unsigned)second
                );
            }
        }
    }
    return mask;
}

static unsigned enumerate_projection(
    const struct specification *specification,
    uint64_t *assignments_checked
) {
    size_t internal[MAX_ELIMINATIONS] = {0};
    size_t internal_count = 0U;
    for (
        size_t index = 0U;
        index < specification->vertex_count;
        ++index
    ) {
        if (specification->vertex[index].kind == INTERNAL_VERTEX) {
            internal[internal_count++] = index;
        }
    }
    if (internal_count > ENUMERATION_INTERNAL_LIMIT) {
        *assignments_checked = 0U;
        return 0U;
    }
    int value[MAX_NODES] = {0};
    unsigned mask = 0U;
    const uint64_t internal_assignments =
        UINT64_C(1) << internal_count;
    for (int second = 0; second < 2; ++second) {
        for (int first = 0; first < 2; ++first) {
            value[specification->external[0]] = first;
            value[specification->external[1]] = second;
            int exists = 0;
            for (
                uint64_t assignment = 0U;
                assignment < internal_assignments;
                ++assignment
            ) {
                ++*assignments_checked;
                for (size_t index = 0U; index < internal_count; ++index) {
                    value[internal[index]] =
                        (int)((assignment >> index) & UINT64_C(1));
                }
                int satisfies = 1;
                for (
                    size_t index = 0U;
                    index < specification->relation_count;
                    ++index
                ) {
                    const struct relation *relation =
                        &specification->relation[index];
                    if (
                        evaluate(
                            relation->coefficient,
                            value[relation->first],
                            value[relation->second]
                        ) != 0
                    ) {
                        satisfies = 0;
                        break;
                    }
                }
                if (satisfies) {
                    exists = 1;
                }
            }
            if (exists) {
                mask |= 1U << (
                    (unsigned)first + 2U * (unsigned)second
                );
            }
        }
    }
    return mask;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s PROCESS.aspr\n", argv[0]);
        return 2;
    }
    const struct specification specification =
        read_specification(argv[1]);
    const struct reduction reduction =
        reduce_specification(&specification);
    size_t internal_count = 0U;
    for (
        size_t index = 0U;
        index < specification.vertex_count;
        ++index
    ) {
        internal_count +=
            specification.vertex[index].kind == INTERNAL_VERTEX;
    }
    uint64_t assignments_checked = 0U;
    const unsigned reduced_mask = zero_mask(reduction.coefficient);
    const unsigned enumerated_mask = enumerate_projection(
        &specification,
        &assignments_checked
    );
    const int enumeration_performed =
        internal_count <= ENUMERATION_INTERNAL_LIMIT;
    const int match = (
        !enumeration_performed
        || enumerated_mask == reduced_mask
    );
    printf(
        "{\"mode\":\"independent-scalar-adjudication\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"node_count\":%zu,"
        "\"input_relation_count\":%zu,"
        "\"internal_node_count\":%zu,"
        "\"scalar_compositions\":%zu,"
        "\"scalar_intersections\":%zu,"
        "\"projected_coefficients\":[%d,%d,%d,%d],"
        "\"projected_zero_mask\":%u,"
        "\"enumeration_performed\":%s,"
        "\"complete_assignments_checked\":%llu,"
        "\"enumerated_zero_mask\":%u,"
        "\"projection_matches_enumeration\":%s}\n",
        (unsigned long long)specification.source_hash,
        specification.vertex_count,
        specification.relation_count,
        internal_count,
        reduction.compositions,
        reduction.intersections,
        reduction.coefficient[0],
        reduction.coefficient[1],
        reduction.coefficient[2],
        reduction.coefficient[3],
        reduced_mask,
        enumeration_performed ? "true" : "false",
        (unsigned long long)assignments_checked,
        enumerated_mask,
        match ? "true" : "false"
    );
    return match ? 0 : 1;
}
