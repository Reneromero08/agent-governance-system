#define _POSIX_C_SOURCE 200809L

/*
 * Separate bounded scalar adjudicator for algebraic_relation_tree_phase.c.
 * This executable is not linked into the native phase engine.
 */

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CCOUNT 4U
#define MAX_NODES 64U
#define MAX_EDGES 63U
#define MAX_EXTERNALS 32U
#define MAX_PAIRS 496U
#define MAX_PATH 63U
#define MAX_ID 31U
#define LINE_CAP 512U
#define TOKEN_CAP 10U

enum kind { EXTERNAL = 0, INTERNAL = 1 };

struct node {
    char name[MAX_ID + 1U];
    enum kind kind;
    size_t degree;
    size_t value_index;
};

struct edge {
    char name[MAX_ID + 1U];
    size_t first;
    size_t second;
    int c[CCOUNT];
};

struct path {
    size_t first;
    size_t second;
    size_t length;
    size_t edge[MAX_PATH];
    size_t node[MAX_PATH + 1U];
    int c[CCOUNT];
};

struct process {
    struct node node[MAX_NODES];
    struct edge edge[MAX_EDGES];
    struct path path[MAX_PAIRS];
    size_t node_count;
    size_t edge_count;
    size_t external[MAX_EXTERNALS];
    size_t external_count;
    size_t internal_count;
    size_t path_count;
    uint64_t source_hash;
};

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void fail_line(const char *message, size_t line) {
    fprintf(stderr, "%s at line %zu\n", message, line);
    exit(2);
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

static int f3(int value) {
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static int evaluate(const int c[CCOUNT], int x, int y) {
    return f3(c[0] + c[1] * x + c[2] * y + c[3] * x * y);
}

static int bi_total(const int c[CCOUNT]) {
    for (int x = 0; x < 2; ++x) {
        int found = 0;
        for (int y = 0; y < 2; ++y) {
            found |= evaluate(c, x, y) == 0;
        }
        if (!found) {
            return 0;
        }
    }
    for (int y = 0; y < 2; ++y) {
        int found = 0;
        for (int x = 0; x < 2; ++x) {
            found |= evaluate(c, x, y) == 0;
        }
        if (!found) {
            return 0;
        }
    }
    return 1;
}

static int valid_id(const char *text) {
    const size_t length = strlen(text);
    if (
        length == 0U
        || length > MAX_ID
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

static size_t tokenize(char *line, char *token[TOKEN_CAP]) {
    size_t count = 0U;
    char *save = NULL;
    for (
        char *part = strtok_r(line, " ", &save);
        part != NULL;
        part = strtok_r(NULL, " ", &save)
    ) {
        if (count == TOKEN_CAP) {
            fail("too many tokens");
        }
        token[count++] = part;
    }
    return count;
}

static size_t node_index(
    const struct process *process,
    const char *name,
    size_t line
) {
    for (size_t index = 0U; index < process->node_count; ++index) {
        if (strcmp(process->node[index].name, name) == 0) {
            return index;
        }
    }
    fail_line("undeclared endpoint", line);
    return 0U;
}

static int shared_edge(
    const struct process *process,
    size_t first,
    size_t second
) {
    for (size_t index = 0U; index < process->edge_count; ++index) {
        const struct edge *edge = &process->edge[index];
        if (
            (edge->first == first && edge->second == second)
            || (edge->first == second && edge->second == first)
        ) {
            return 1;
        }
    }
    return 0;
}

static void transpose(int output[CCOUNT], const int input[CCOUNT]) {
    output[0] = input[0];
    output[1] = input[2];
    output[2] = input[1];
    output[3] = input[3];
}

static void resultant(
    int output[CCOUNT],
    const int left[CCOUNT],
    const int right[CCOUNT]
) {
    output[0] = f3(left[0] * right[2] - left[2] * right[0]);
    output[1] = f3(left[1] * right[2] - left[3] * right[0]);
    output[2] = f3(left[0] * right[3] - left[2] * right[1]);
    output[3] = f3(left[1] * right[3] - left[3] * right[1]);
}

static void oriented_edge(
    const struct process *process,
    size_t edge_index,
    size_t desired_first,
    int output[CCOUNT]
) {
    const struct edge *edge = &process->edge[edge_index];
    if (edge->first == desired_first) {
        memcpy(output, edge->c, CCOUNT * sizeof(*output));
    } else {
        transpose(output, edge->c);
    }
}

static void build_path(
    struct process *process,
    size_t first,
    size_t second
) {
    if (process->path_count == MAX_PAIRS) {
        fail("pair capacity exceeded");
    }
    size_t queue[MAX_NODES];
    size_t parent[MAX_NODES];
    size_t parent_edge[MAX_NODES];
    int seen[MAX_NODES] = {0};
    for (size_t index = 0U; index < process->node_count; ++index) {
        parent[index] = SIZE_MAX;
        parent_edge[index] = SIZE_MAX;
    }
    size_t head = 0U;
    size_t tail = 0U;
    queue[tail++] = first;
    seen[first] = 1;
    while (head < tail && !seen[second]) {
        const size_t current = queue[head++];
        for (size_t edge_index = 0U; edge_index < process->edge_count; ++edge_index) {
            const struct edge *edge = &process->edge[edge_index];
            size_t next = SIZE_MAX;
            if (edge->first == current) {
                next = edge->second;
            } else if (edge->second == current) {
                next = edge->first;
            }
            if (next != SIZE_MAX && !seen[next]) {
                seen[next] = 1;
                parent[next] = current;
                parent_edge[next] = edge_index;
                queue[tail++] = next;
            }
        }
    }
    if (!seen[second]) {
        fail("tree is disconnected");
    }
    size_t reverse_node[MAX_PATH + 1U];
    size_t reverse_edge[MAX_PATH];
    size_t length = 0U;
    size_t cursor = second;
    reverse_node[0] = cursor;
    while (cursor != first) {
        if (length == MAX_PATH) {
            fail("path capacity exceeded");
        }
        reverse_edge[length] = parent_edge[cursor];
        cursor = parent[cursor];
        reverse_node[++length] = cursor;
    }
    struct path *path = &process->path[process->path_count++];
    path->first = first;
    path->second = second;
    path->length = length;
    for (size_t index = 0U; index <= length; ++index) {
        path->node[index] = reverse_node[length - index];
    }
    for (size_t index = 0U; index < length; ++index) {
        path->edge[index] = reverse_edge[length - 1U - index];
    }
    oriented_edge(process, path->edge[0], path->node[0], path->c);
    for (size_t position = 1U; position < length; ++position) {
        int right[CCOUNT];
        int next[CCOUNT];
        oriented_edge(
            process,
            path->edge[position],
            path->node[position + 1U],
            right
        );
        resultant(next, path->c, right);
        memcpy(path->c, next, sizeof(next));
    }
}

static void validate_and_plan(struct process *process) {
    if (
        process->node_count < 2U
        || process->edge_count + 1U != process->node_count
    ) {
        fail("relations must form a finite tree");
    }
    for (size_t index = 0U; index < process->edge_count; ++index) {
        ++process->node[process->edge[index].first].degree;
        ++process->node[process->edge[index].second].degree;
    }
    size_t queue[MAX_NODES];
    int seen[MAX_NODES] = {0};
    size_t head = 0U;
    size_t tail = 0U;
    queue[tail++] = 0U;
    seen[0] = 1;
    while (head < tail) {
        const size_t current = queue[head++];
        for (size_t edge_index = 0U; edge_index < process->edge_count; ++edge_index) {
            const struct edge *edge = &process->edge[edge_index];
            size_t next = SIZE_MAX;
            if (edge->first == current) {
                next = edge->second;
            } else if (edge->second == current) {
                next = edge->first;
            }
            if (next != SIZE_MAX && !seen[next]) {
                seen[next] = 1;
                queue[tail++] = next;
            }
        }
    }
    if (tail != process->node_count) {
        fail("tree is disconnected");
    }
    for (size_t index = 0U; index < process->node_count; ++index) {
        struct node *node = &process->node[index];
        if (node->kind == EXTERNAL) {
            if (node->degree != 1U) {
                fail("external node must have degree one");
            }
            if (process->external_count == MAX_EXTERNALS) {
                fail("external capacity exceeded");
            }
            node->value_index = process->external_count;
            process->external[process->external_count++] = index;
        } else {
            if (node->degree < 2U) {
                fail("internal node must have degree at least two");
            }
            node->value_index = process->internal_count++;
        }
    }
    if (process->external_count < 2U || process->internal_count > 20U) {
        fail("reference enumeration bound exceeded");
    }
    for (size_t first = 0U; first < process->external_count; ++first) {
        for (size_t second = first + 1U; second < process->external_count; ++second) {
            build_path(
                process,
                process->external[first],
                process->external[second]
            );
        }
    }
}

static struct process read_process(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct process process = {
        .source_hash = UINT64_C(14695981039346656037)
    };
    char line[LINE_CAP];
    size_t line_number = 0U;
    int header = 0;
    int type = 0;
    int relations = 0;
    int end = 0;
    while (fgets(line, sizeof(line), stream) != NULL) {
        ++line_number;
        const size_t length = strlen(line);
        process.source_hash = hash_bytes(
            process.source_hash,
            (const unsigned char *)line,
            length
        );
        if (length == 0U || line[length - 1U] != '\n') {
            fail_line("record lacks LF", line_number);
        }
        if (memchr(line, '\r', length) != NULL) {
            fail_line("CR is forbidden", line_number);
        }
        line[length - 1U] = '\0';
        if (end) {
            fail_line("record after END", line_number);
        }
        char *token[TOKEN_CAP] = {0};
        const size_t count = tokenize(line, token);
        if (count == 0U) {
            fail_line("blank records are forbidden", line_number);
        }
        if (!header) {
            if (
                count != 2U
                || strcmp(token[0], "CATCAS_ALGEBRAIC_RELATION_TREE") != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid header", line_number);
            }
            header = 1;
        } else if (strcmp(token[0], "TYPE") == 0) {
            if (
                type
                || count != 2U
                || strcmp(token[1], "BOOLEAN_F3") != 0
            ) {
                fail_line("invalid TYPE", line_number);
            }
            type = 1;
        } else if (strcmp(token[0], "NODE") == 0) {
            if (!type || relations || count != 3U) {
                fail_line("invalid NODE", line_number);
            }
            if (process.node_count == MAX_NODES || !valid_id(token[1])) {
                fail_line("invalid node identifier", line_number);
            }
            for (size_t index = 0U; index < process.node_count; ++index) {
                if (strcmp(process.node[index].name, token[1]) == 0) {
                    fail_line("duplicate node", line_number);
                }
            }
            struct node *node = &process.node[process.node_count++];
            memcpy(node->name, token[1], strlen(token[1]) + 1U);
            if (strcmp(token[2], "EXTERNAL") == 0) {
                node->kind = EXTERNAL;
            } else if (strcmp(token[2], "INTERNAL") == 0) {
                node->kind = INTERNAL;
            } else {
                fail_line("invalid node kind", line_number);
            }
        } else if (strcmp(token[0], "RELATION") == 0) {
            if (!type || process.node_count == 0U || count != 8U) {
                fail_line("invalid RELATION", line_number);
            }
            relations = 1;
            if (process.edge_count == MAX_EDGES || !valid_id(token[1])) {
                fail_line("invalid relation identifier", line_number);
            }
            for (size_t index = 0U; index < process.edge_count; ++index) {
                if (strcmp(process.edge[index].name, token[1]) == 0) {
                    fail_line("duplicate relation", line_number);
                }
            }
            const size_t first = node_index(&process, token[2], line_number);
            const size_t second = node_index(&process, token[3], line_number);
            if (first == second || shared_edge(&process, first, second)) {
                fail_line("invalid tree edge", line_number);
            }
            struct edge *edge = &process.edge[process.edge_count++];
            memcpy(edge->name, token[1], strlen(token[1]) + 1U);
            edge->first = first;
            edge->second = second;
            for (size_t index = 0U; index < CCOUNT; ++index) {
                if (
                    strlen(token[4U + index]) != 1U
                    || token[4U + index][0] < '0'
                    || token[4U + index][0] > '2'
                ) {
                    fail_line("invalid coefficient", line_number);
                }
                edge->c[index] = token[4U + index][0] - '0';
            }
            if (!bi_total(edge->c)) {
                fail_line("non-bi-total relation", line_number);
            }
        } else if (strcmp(token[0], "END") == 0) {
            if (count != 1U) {
                fail_line("invalid END", line_number);
            }
            end = 1;
        } else {
            fail_line("unknown record", line_number);
        }
    }
    if (ferror(stream) || fclose(stream) != 0) {
        fail("process read failed");
    }
    if (!header || !type || !relations || !end) {
        fail("process incomplete");
    }
    validate_and_plan(&process);
    return process;
}

static uint64_t boundary_hash(const struct process *process) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0U; index < process->path_count; ++index) {
        const struct path *path = &process->path[index];
        const char *first = process->node[path->first].name;
        const char *second = process->node[path->second].name;
        hash = hash_bytes(
            hash,
            (const unsigned char *)first,
            strlen(first)
        );
        const unsigned char separator = ':';
        hash = hash_bytes(hash, &separator, 1U);
        hash = hash_bytes(
            hash,
            (const unsigned char *)second,
            strlen(second)
        );
        for (size_t coefficient = 0U; coefficient < CCOUNT; ++coefficient) {
            const unsigned char byte =
                (unsigned char)path->c[coefficient];
            hash = hash_bytes(hash, &byte, 1U);
        }
    }
    return hash;
}

static int path_projection_accepts(
    const struct process *process,
    uint64_t external_assignment
) {
    for (size_t index = 0U; index < process->path_count; ++index) {
        const struct path *path = &process->path[index];
        const int first = (
            external_assignment
            >> process->node[path->first].value_index
        ) & UINT64_C(1);
        const int second = (
            external_assignment
            >> process->node[path->second].value_index
        ) & UINT64_C(1);
        if (evaluate(path->c, first, second) != 0) {
            return 0;
        }
    }
    return 1;
}

static uint64_t exact_witnesses(
    const struct process *process,
    uint64_t external_assignment
) {
    const uint64_t internal_limit =
        UINT64_C(1) << process->internal_count;
    uint64_t witnesses = 0U;
    for (
        uint64_t internal_assignment = 0U;
        internal_assignment < internal_limit;
        ++internal_assignment
    ) {
        int accepted = 1;
        for (size_t edge_index = 0U; edge_index < process->edge_count; ++edge_index) {
            const struct edge *edge = &process->edge[edge_index];
            const struct node *first_node = &process->node[edge->first];
            const struct node *second_node = &process->node[edge->second];
            const uint64_t first_bits = first_node->kind == EXTERNAL
                ? external_assignment
                : internal_assignment;
            const uint64_t second_bits = second_node->kind == EXTERNAL
                ? external_assignment
                : internal_assignment;
            const int first = (
                first_bits >> first_node->value_index
            ) & UINT64_C(1);
            const int second = (
                second_bits >> second_node->value_index
            ) & UINT64_C(1);
            if (evaluate(edge->c, first, second) != 0) {
                accepted = 0;
                break;
            }
        }
        witnesses += (uint64_t)accepted;
    }
    return witnesses;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s PROCESS.art\n", argv[0]);
        return 2;
    }
    const struct process process = read_process(argv[1]);
    const uint64_t external_limit =
        UINT64_C(1) << process.external_count;
    uint64_t projected = 0U;
    uint64_t exact = 0U;
    uint64_t mismatches = 0U;
    uint64_t multi_witness = 0U;
    for (
        uint64_t assignment = 0U;
        assignment < external_limit;
        ++assignment
    ) {
        const int projection = path_projection_accepts(
            &process,
            assignment
        );
        const uint64_t witnesses = exact_witnesses(
            &process,
            assignment
        );
        projected += (uint64_t)projection;
        exact += (uint64_t)(witnesses != 0U);
        mismatches += (uint64_t)(projection != (witnesses != 0U));
        multi_witness += (uint64_t)(witnesses > 1U);
    }
    printf(
        "{\"mode\":\"scalar-relation-tree-reference\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"boundary_factor_fnv1a64\":\"%016llx\","
        "\"external_assignments\":%llu,"
        "\"internal_assignments_per_boundary\":%llu,"
        "\"projected_assignments\":%llu,"
        "\"exact_assignments\":%llu,"
        "\"multi_witness_assignments\":%llu,"
        "\"mismatches\":%llu}\n",
        (unsigned long long)process.source_hash,
        (unsigned long long)boundary_hash(&process),
        (unsigned long long)external_limit,
        (unsigned long long)(
            UINT64_C(1) << process.internal_count
        ),
        (unsigned long long)projected,
        (unsigned long long)exact,
        (unsigned long long)multi_witness,
        (unsigned long long)mismatches
    );
    return mismatches == 0U ? 0 : 1;
}
