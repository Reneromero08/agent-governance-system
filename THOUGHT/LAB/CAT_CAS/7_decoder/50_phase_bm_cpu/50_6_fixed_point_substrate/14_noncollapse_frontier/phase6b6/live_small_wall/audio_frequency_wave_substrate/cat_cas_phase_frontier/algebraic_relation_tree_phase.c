#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: generic tree-shaped relational phase closure.
 *
 * Public bi-total Boolean_F3 relations occupy the edges of a typed tree.
 * For each pair of external leaves, the unique path is composed by a chain
 * of four-cell complex phase resultants.  Every non-final resultant remains
 * resident in the borrowed carrier and directly feeds the next resultant.
 * No intermediate coefficient, tuple, witness, or truth table is decoded.
 */

#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COEFFICIENT_COUNT 4U
#define MAX_NODES 64U
#define MAX_EDGES 63U
#define MAX_EXTERNALS 32U
#define MAX_PAIRS 496U
#define MAX_PATH_EDGES 63U
#define MAX_IDENTIFIER 31U
#define LINE_CAPACITY 512U
#define TOKEN_CAPACITY 10U

static const double ROOT_TOLERANCE = 3.0e-10;
static const double RESTORATION_TOLERANCE = 2.0e-12;
static const double CONTROL_MINIMUM = 1.0e-3;

enum node_kind { NODE_EXTERNAL = 0, NODE_INTERNAL = 1 };
enum inverse_mode {
    INVERSE_CORRECT = 0,
    INVERSE_WRONG_BOUNDARY = 1,
    FORWARD_SCRAMBLED_TOPOLOGY = 2,
    INVERSE_OMITTED_MESSAGE = 3,
    INVERSE_BYPASSED_PATH = 4
};

struct node {
    char name[MAX_IDENTIFIER + 1U];
    enum node_kind kind;
    size_t degree;
};

struct edge {
    char name[MAX_IDENTIFIER + 1U];
    size_t first;
    size_t second;
    int coefficient[COEFFICIENT_COUNT];
};

struct path {
    size_t external_first;
    size_t external_second;
    size_t length;
    size_t edge[MAX_PATH_EDGES];
    size_t node[MAX_PATH_EDGES + 1U];
    size_t message_offset;
    size_t message_count;
};

struct process {
    struct node node[MAX_NODES];
    struct edge edge[MAX_EDGES];
    struct path path[MAX_PAIRS];
    size_t node_count;
    size_t edge_count;
    size_t external[MAX_EXTERNALS];
    size_t external_count;
    size_t path_count;
    size_t message_count;
    uint64_t source_fnv1a64;
};

struct carrier {
    double complex *baseline;
    double complex *working;
    size_t cells;
};

struct boundary_record {
    uint64_t factor_fnv1a64;
    int (*coefficient)[COEFFICIENT_COUNT];
    size_t count;
    double maximum_root_error;
};

struct execution {
    struct boundary_record boundary;
    double displacement_l2;
    double restoration_max_abs;
    double carrier_integrity_error;
    int wrong_applicable;
    int topology_applicable;
    int omitted_message_applicable;
};

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void fail_line(const char *message, size_t line_number) {
    fprintf(stderr, "%s at line %zu\n", message, line_number);
    exit(2);
}

static void *checked_calloc(size_t count, size_t size) {
    if (count != 0U && size > SIZE_MAX / count) {
        fail("allocation size overflow");
    }
    void *memory = calloc(count, size);
    if (memory == NULL) {
        fail("allocation failed");
    }
    return memory;
}

static uint64_t fnv1a64_update(
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
    int normalized = value % 3;
    return normalized < 0 ? normalized + 3 : normalized;
}

static int evaluate_relation(
    const int coefficient[COEFFICIENT_COUNT],
    int first,
    int second
) {
    return f3(
        coefficient[0]
        + coefficient[1] * first
        + coefficient[2] * second
        + coefficient[3] * first * second
    );
}

static int relation_is_bi_total(
    const int coefficient[COEFFICIENT_COUNT]
) {
    for (int first = 0; first < 2; ++first) {
        int found = 0;
        for (int second = 0; second < 2; ++second) {
            found |= evaluate_relation(coefficient, first, second) == 0;
        }
        if (!found) {
            return 0;
        }
    }
    for (int second = 0; second < 2; ++second) {
        int found = 0;
        for (int first = 0; first < 2; ++first) {
            found |= evaluate_relation(coefficient, first, second) == 0;
        }
        if (!found) {
            return 0;
        }
    }
    return 1;
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

static size_t tokenize(char *line, char *token[TOKEN_CAPACITY]) {
    size_t count = 0U;
    char *save = NULL;
    for (
        char *part = strtok_r(line, " ", &save);
        part != NULL;
        part = strtok_r(NULL, " ", &save)
    ) {
        if (count == TOKEN_CAPACITY) {
            fail("too many tokens");
        }
        token[count++] = part;
    }
    return count;
}

static int parse_coefficient(const char *text, size_t line_number) {
    if (
        strlen(text) != 1U
        || text[0] < '0'
        || text[0] > '2'
    ) {
        fail_line("coefficient must be one canonical F3 digit", line_number);
    }
    return text[0] - '0';
}

static size_t find_node(
    const struct process *process,
    const char *name,
    size_t line_number
) {
    for (size_t index = 0U; index < process->node_count; ++index) {
        if (strcmp(process->node[index].name, name) == 0) {
            return index;
        }
    }
    fail_line("relation endpoint is not a declared node", line_number);
    return 0U;
}

static int nodes_share_edge(
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

static void build_path(
    struct process *process,
    size_t external_first,
    size_t external_second
) {
    if (process->path_count == MAX_PAIRS) {
        fail("external-pair capacity exceeded");
    }
    size_t queue[MAX_NODES];
    size_t predecessor_node[MAX_NODES];
    size_t predecessor_edge[MAX_NODES];
    int seen[MAX_NODES] = {0};
    for (size_t index = 0U; index < process->node_count; ++index) {
        predecessor_node[index] = SIZE_MAX;
        predecessor_edge[index] = SIZE_MAX;
    }
    size_t head = 0U;
    size_t tail = 0U;
    queue[tail++] = external_first;
    seen[external_first] = 1;
    while (head < tail && !seen[external_second]) {
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
                predecessor_node[next] = current;
                predecessor_edge[next] = edge_index;
                queue[tail++] = next;
            }
        }
    }
    if (!seen[external_second]) {
        fail("tree is disconnected");
    }
    size_t reverse_node[MAX_PATH_EDGES + 1U];
    size_t reverse_edge[MAX_PATH_EDGES];
    size_t length = 0U;
    size_t cursor = external_second;
    reverse_node[0] = cursor;
    while (cursor != external_first) {
        if (length == MAX_PATH_EDGES) {
            fail("path edge capacity exceeded");
        }
        reverse_edge[length] = predecessor_edge[cursor];
        cursor = predecessor_node[cursor];
        reverse_node[++length] = cursor;
    }
    struct path *path = &process->path[process->path_count++];
    path->external_first = external_first;
    path->external_second = external_second;
    path->length = length;
    for (size_t index = 0U; index <= length; ++index) {
        path->node[index] = reverse_node[length - index];
    }
    for (size_t index = 0U; index < length; ++index) {
        path->edge[index] = reverse_edge[length - 1U - index];
    }
    path->message_offset = process->message_count;
    path->message_count = length > 1U ? length - 2U : 0U;
    process->message_count += path->message_count;
}

static void validate_and_plan(struct process *process) {
    if (
        process->node_count < 2U
        || process->edge_count + 1U != process->node_count
    ) {
        fail("relations must form a finite tree");
    }
    for (size_t edge_index = 0U; edge_index < process->edge_count; ++edge_index) {
        ++process->node[process->edge[edge_index].first].degree;
        ++process->node[process->edge[edge_index].second].degree;
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
        if (node->kind == NODE_EXTERNAL) {
            if (node->degree != 1U) {
                fail("external node must have degree one");
            }
            if (process->external_count == MAX_EXTERNALS) {
                fail("external node capacity exceeded");
            }
            process->external[process->external_count++] = index;
        } else if (node->degree < 2U) {
            fail("internal node must have degree at least two");
        }
    }
    if (process->external_count < 2U) {
        fail("tree needs at least two external nodes");
    }
    for (size_t first = 0U; first < process->external_count; ++first) {
        for (
            size_t second = first + 1U;
            second < process->external_count;
            ++second
        ) {
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
        .source_fnv1a64 = UINT64_C(14695981039346656037)
    };
    char line[LINE_CAPACITY];
    size_t line_number = 0U;
    int header_seen = 0;
    int type_seen = 0;
    int relation_seen = 0;
    int end_seen = 0;
    while (fgets(line, sizeof(line), stream) != NULL) {
        ++line_number;
        const size_t length = strlen(line);
        process.source_fnv1a64 = fnv1a64_update(
            process.source_fnv1a64,
            (const unsigned char *)line,
            length
        );
        if (length == 0U || line[length - 1U] != '\n') {
            fail_line("every record must end with LF", line_number);
        }
        if (memchr(line, '\r', length) != NULL) {
            fail_line("CR bytes are forbidden", line_number);
        }
        line[length - 1U] = '\0';
        if (end_seen) {
            fail_line("record after END", line_number);
        }
        char *token[TOKEN_CAPACITY] = {0};
        const size_t count = tokenize(line, token);
        if (count == 0U) {
            fail_line("blank records are forbidden", line_number);
        }
        if (!header_seen) {
            if (
                count != 2U
                || strcmp(token[0], "CATCAS_ALGEBRAIC_RELATION_TREE") != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid relation-tree header", line_number);
            }
            header_seen = 1;
            continue;
        }
        if (strcmp(token[0], "TYPE") == 0) {
            if (
                type_seen
                || count != 2U
                || strcmp(token[1], "BOOLEAN_F3") != 0
            ) {
                fail_line("invalid TYPE", line_number);
            }
            type_seen = 1;
            continue;
        }
        if (strcmp(token[0], "NODE") == 0) {
            if (!type_seen || relation_seen || count != 3U) {
                fail_line("invalid NODE", line_number);
            }
            if (process.node_count == MAX_NODES) {
                fail_line("node capacity exceeded", line_number);
            }
            if (!valid_identifier(token[1])) {
                fail_line("invalid node identifier", line_number);
            }
            for (size_t index = 0U; index < process.node_count; ++index) {
                if (strcmp(process.node[index].name, token[1]) == 0) {
                    fail_line("duplicate node identifier", line_number);
                }
            }
            struct node *node = &process.node[process.node_count++];
            memcpy(node->name, token[1], strlen(token[1]) + 1U);
            if (strcmp(token[2], "EXTERNAL") == 0) {
                node->kind = NODE_EXTERNAL;
            } else if (strcmp(token[2], "INTERNAL") == 0) {
                node->kind = NODE_INTERNAL;
            } else {
                fail_line("unknown node kind", line_number);
            }
            continue;
        }
        if (strcmp(token[0], "RELATION") == 0) {
            if (!type_seen || process.node_count == 0U || count != 8U) {
                fail_line("invalid RELATION", line_number);
            }
            relation_seen = 1;
            if (process.edge_count == MAX_EDGES) {
                fail_line("relation capacity exceeded", line_number);
            }
            if (!valid_identifier(token[1])) {
                fail_line("invalid relation identifier", line_number);
            }
            for (size_t index = 0U; index < process.edge_count; ++index) {
                if (strcmp(process.edge[index].name, token[1]) == 0) {
                    fail_line("duplicate relation identifier", line_number);
                }
            }
            const size_t first = find_node(&process, token[2], line_number);
            const size_t second = find_node(&process, token[3], line_number);
            if (first == second) {
                fail_line("self relations are forbidden", line_number);
            }
            if (nodes_share_edge(&process, first, second)) {
                fail_line("parallel tree relation", line_number);
            }
            struct edge *edge = &process.edge[process.edge_count++];
            memcpy(edge->name, token[1], strlen(token[1]) + 1U);
            edge->first = first;
            edge->second = second;
            for (size_t coefficient = 0U; coefficient < COEFFICIENT_COUNT; ++coefficient) {
                edge->coefficient[coefficient] = parse_coefficient(
                    token[4U + coefficient],
                    line_number
                );
            }
            if (!relation_is_bi_total(edge->coefficient)) {
                fail_line(
                    "relation is not bi-total on Boolean_F3",
                    line_number
                );
            }
            continue;
        }
        if (strcmp(token[0], "END") == 0) {
            if (count != 1U) {
                fail_line("invalid END", line_number);
            }
            end_seen = 1;
            continue;
        }
        fail_line("unknown relation-tree record", line_number);
    }
    if (ferror(stream)) {
        fail("failed to read complete relation tree");
    }
    if (fclose(stream) != 0) {
        perror(path);
        exit(2);
    }
    if (
        !header_seen
        || !type_seen
        || !relation_seen
        || !end_seen
    ) {
        fail("relation-tree process is incomplete");
    }
    validate_and_plan(&process);
    return process;
}

static double complex unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        fail("nonfinite phase state");
    }
    return value / magnitude;
}

static double complex root3(int amount) {
    const int normalized = f3(amount);
    if (normalized == 0) {
        return 1.0 + 0.0 * I;
    }
    if (normalized == 1) {
        return -0.5 + 0.86602540378443864676 * I;
    }
    return -0.5 - 0.86602540378443864676 * I;
}

static size_t input_cells(const struct process *process) {
    return process->edge_count * COEFFICIENT_COUNT;
}

static size_t message_cells(const struct process *process) {
    return process->message_count * COEFFICIENT_COUNT;
}

static size_t carrier_cells(const struct process *process) {
    return (
        process->edge_count
        + process->message_count
        + process->path_count
    ) * COEFFICIENT_COUNT;
}

static size_t input_start(size_t edge) {
    return edge * COEFFICIENT_COUNT;
}

static size_t message_start(
    const struct process *process,
    size_t message
) {
    return input_cells(process) + message * COEFFICIENT_COUNT;
}

static size_t boundary_start(
    const struct process *process,
    size_t boundary
) {
    return (
        input_cells(process)
        + message_cells(process)
        + boundary * COEFFICIENT_COUNT
    );
}

static struct carrier make_carrier(
    const struct process *process,
    int identity
) {
    const size_t cells = carrier_cells(process);
    struct carrier carrier = {
        .baseline = checked_calloc(cells, sizeof(*carrier.baseline)),
        .working = checked_calloc(cells, sizeof(*carrier.working)),
        .cells = cells
    };
    for (size_t index = 0U; index < cells; ++index) {
        const double angle =
            0.211
            + 0.057 * (double)index
            + 0.017 * sin(0.23 * (double)index + 0.019 * identity);
        carrier.baseline[index] = cexp(I * angle);
        carrier.working[index] = carrier.baseline[index];
    }
    return carrier;
}

static struct carrier snapshot_carrier(const struct carrier *source) {
    struct carrier snapshot = {
        .baseline = checked_calloc(source->cells, sizeof(*snapshot.baseline)),
        .working = checked_calloc(source->cells, sizeof(*snapshot.working)),
        .cells = source->cells
    };
    memcpy(
        snapshot.baseline,
        source->baseline,
        source->cells * sizeof(*snapshot.baseline)
    );
    memcpy(
        snapshot.working,
        source->working,
        source->cells * sizeof(*snapshot.working)
    );
    return snapshot;
}

static void free_carrier(struct carrier *carrier) {
    free(carrier->baseline);
    free(carrier->working);
    carrier->baseline = NULL;
    carrier->working = NULL;
    carrier->cells = 0U;
}

static double complex relation(
    const struct carrier *carrier,
    size_t cell
) {
    return carrier->working[cell] * conj(carrier->baseline[cell]);
}

static void multiply_relation(
    struct carrier *carrier,
    size_t cell,
    double complex factor
) {
    carrier->working[cell] = unit(
        relation(carrier, cell) * factor
    ) * carrier->baseline[cell];
}

static double complex product_factor(
    double complex left,
    double complex right
) {
    const double complex left_squared = conj(left);
    const double complex right_squared = conj(right);
    const double complex product = left * right;
    return unit(
        (
            1.0
            + left
            + left_squared
            + right
            + right_squared
            + root3(2) * (product + conj(product))
            + root3(1) * (
                left * right_squared
                + left_squared * right
            )
        ) / 3.0
    );
}

static size_t oriented_cell(
    size_t start,
    size_t coefficient,
    int transpose
) {
    static const size_t transposed[COEFFICIENT_COUNT] = {
        0U, 2U, 1U, 3U
    };
    return start + (
        transpose ? transposed[coefficient] : coefficient
    );
}

static double complex difference_product(
    const struct carrier *carrier,
    size_t positive_left,
    size_t positive_right,
    size_t negative_left,
    size_t negative_right
) {
    return unit(
        product_factor(
            relation(carrier, positive_left),
            relation(carrier, positive_right)
        )
        * conj(
            product_factor(
                relation(carrier, negative_left),
                relation(carrier, negative_right)
            )
        )
    );
}

static void resultant_factors(
    const struct carrier *carrier,
    size_t left,
    int left_transpose,
    size_t right,
    int right_transpose,
    double complex factor[COEFFICIENT_COUNT]
) {
    const size_t l0 = oriented_cell(left, 0U, left_transpose);
    const size_t l1 = oriented_cell(left, 1U, left_transpose);
    const size_t l2 = oriented_cell(left, 2U, left_transpose);
    const size_t l3 = oriented_cell(left, 3U, left_transpose);
    const size_t r0 = oriented_cell(right, 0U, right_transpose);
    const size_t r1 = oriented_cell(right, 1U, right_transpose);
    const size_t r2 = oriented_cell(right, 2U, right_transpose);
    const size_t r3 = oriented_cell(right, 3U, right_transpose);
    factor[0] = difference_product(carrier, l0, r2, l2, r0);
    factor[1] = difference_product(carrier, l1, r2, l3, r0);
    factor[2] = difference_product(carrier, l0, r3, l2, r1);
    factor[3] = difference_product(carrier, l1, r3, l3, r1);
}

static void read_oriented_factor(
    const struct carrier *carrier,
    size_t start,
    int transpose,
    double complex factor[COEFFICIENT_COUNT]
) {
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        factor[index] = relation(
            carrier,
            oriented_cell(start, index, transpose)
        );
    }
}

static void apply_factor(
    struct carrier *carrier,
    size_t output,
    const double complex factor[COEFFICIENT_COUNT],
    int inverse
) {
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        multiply_relation(
            carrier,
            output + index,
            inverse ? conj(factor[index]) : factor[index]
        );
    }
}

static void apply_resultant(
    struct carrier *carrier,
    size_t left,
    int left_transpose,
    size_t right,
    int right_transpose,
    size_t output,
    int inverse
) {
    double complex factor[COEFFICIENT_COUNT];
    resultant_factors(
        carrier,
        left,
        left_transpose,
        right,
        right_transpose,
        factor
    );
    apply_factor(carrier, output, factor, inverse);
}

static void apply_encoding(
    struct carrier *carrier,
    size_t start,
    const int coefficient[COEFFICIENT_COUNT],
    int inverse
) {
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        const double complex factor = root3(coefficient[index]);
        multiply_relation(
            carrier,
            start + index,
            inverse ? conj(factor) : factor
        );
    }
}

static int edge_transposed_from(
    const struct process *process,
    size_t edge_index,
    size_t desired_first
) {
    return process->edge[edge_index].first != desired_first;
}

static void path_final_factor(
    const struct process *process,
    const struct carrier *carrier,
    const struct path *path,
    int bypass,
    int scramble,
    double complex factor[COEFFICIENT_COUNT]
) {
    if (path->length == 1U) {
        read_oriented_factor(
            carrier,
            input_start(path->edge[0]),
            edge_transposed_from(
                process,
                path->edge[0],
                path->node[0]
            ) ^ scramble,
            factor
        );
        return;
    }
    size_t left = input_start(path->edge[0]);
    int left_transpose = edge_transposed_from(
        process,
        path->edge[0],
        path->node[0]
    );
    size_t final_edge_position = path->length - 1U;
    if (!bypass && path->length > 2U) {
        left = message_start(
            process,
            path->message_offset + path->message_count - 1U
        );
        left_transpose = 0;
    } else if (bypass && path->length > 2U) {
        final_edge_position = path->length - 1U;
    }
    const size_t right_edge = path->edge[final_edge_position];
    resultant_factors(
        carrier,
        left,
        left_transpose,
        input_start(right_edge),
        edge_transposed_from(
            process,
            right_edge,
            path->node[final_edge_position + 1U]
        ) ^ scramble,
        factor
    );
}

static void forward_path(
    const struct process *process,
    struct carrier *carrier,
    size_t path_index,
    int bypass,
    int scramble
) {
    const struct path *path = &process->path[path_index];
    if (path->length == 1U) {
        double complex factor[COEFFICIENT_COUNT];
        path_final_factor(
            process,
            carrier,
            path,
            bypass,
            scramble,
            factor
        );
        apply_factor(
            carrier,
            boundary_start(process, path_index),
            factor,
            0
        );
        return;
    }
    if (bypass && path->length > 2U) {
        double complex factor[COEFFICIENT_COUNT];
        path_final_factor(
            process,
            carrier,
            path,
            1,
            scramble,
            factor
        );
        apply_factor(
            carrier,
            boundary_start(process, path_index),
            factor,
            0
        );
        return;
    }
    size_t left = input_start(path->edge[0]);
    int left_transpose = edge_transposed_from(
        process,
        path->edge[0],
        path->node[0]
    );
    for (size_t position = 1U; position < path->length; ++position) {
        const int final = position + 1U == path->length;
        const size_t output = final
            ? boundary_start(process, path_index)
            : message_start(
                process,
                path->message_offset + position - 1U
            );
        const size_t right_edge = path->edge[position];
        apply_resultant(
            carrier,
            left,
            left_transpose,
            input_start(right_edge),
            edge_transposed_from(
                process,
                right_edge,
                path->node[position + 1U]
            ) ^ (scramble && final),
            output,
            0
        );
        left = output;
        left_transpose = 0;
    }
}

static void inverse_path_boundary(
    const struct process *process,
    struct carrier *carrier,
    size_t path_index,
    int bypass,
    int scramble
) {
    double complex factor[COEFFICIENT_COUNT];
    path_final_factor(
        process,
        carrier,
        &process->path[path_index],
        bypass,
        scramble,
        factor
    );
    apply_factor(
        carrier,
        boundary_start(process, path_index),
        factor,
        1
    );
}

static void inverse_path_messages(
    const struct process *process,
    struct carrier *carrier,
    const struct path *path,
    size_t omitted_message
) {
    if (path->length <= 2U) {
        return;
    }
    for (size_t position = path->length - 1U; position > 1U; --position) {
        const size_t message_index =
            path->message_offset + position - 2U;
        if (message_index == omitted_message) {
            continue;
        }
        const size_t left = position == 2U
            ? input_start(path->edge[0])
            : message_start(process, message_index - 1U);
        const int left_transpose = position == 2U
            ? edge_transposed_from(
                process,
                path->edge[0],
                path->node[0]
            )
            : 0;
        const size_t right_position = position - 1U;
        const size_t right_edge = path->edge[right_position];
        apply_resultant(
            carrier,
            left,
            left_transpose,
            input_start(right_edge),
            edge_transposed_from(
                process,
                right_edge,
                path->node[right_position + 1U]
            ),
            message_start(process, message_index),
            1
        );
    }
}

static int decode_root3(
    double complex value,
    double *distance_out
) {
    int best = 0;
    double best_distance = INFINITY;
    for (int symbol = 0; symbol < 3; ++symbol) {
        const double distance = cabs(value - root3(symbol));
        if (distance < best_distance) {
            best = symbol;
            best_distance = distance;
        }
    }
    *distance_out = best_distance;
    return best;
}

static struct boundary_record latch_boundary(
    const struct process *process,
    const struct carrier *carrier
) {
    struct boundary_record boundary = {
        .factor_fnv1a64 = UINT64_C(14695981039346656037),
        .coefficient = checked_calloc(
            process->path_count,
            sizeof(*boundary.coefficient)
        ),
        .count = process->path_count,
        .maximum_root_error = 0.0
    };
    for (size_t index = 0U; index < process->path_count; ++index) {
        const struct path *path = &process->path[index];
        const char *first = process->node[path->external_first].name;
        const char *second = process->node[path->external_second].name;
        boundary.factor_fnv1a64 = fnv1a64_update(
            boundary.factor_fnv1a64,
            (const unsigned char *)first,
            strlen(first)
        );
        const unsigned char separator = ':';
        boundary.factor_fnv1a64 = fnv1a64_update(
            boundary.factor_fnv1a64,
            &separator,
            1U
        );
        boundary.factor_fnv1a64 = fnv1a64_update(
            boundary.factor_fnv1a64,
            (const unsigned char *)second,
            strlen(second)
        );
        for (size_t coefficient = 0U; coefficient < COEFFICIENT_COUNT; ++coefficient) {
            double distance = 0.0;
            const int decoded = decode_root3(
                relation(
                    carrier,
                    boundary_start(process, index) + coefficient
                ),
                &distance
            );
            boundary.coefficient[index][coefficient] = decoded;
            const unsigned char byte = (unsigned char)decoded;
            boundary.factor_fnv1a64 = fnv1a64_update(
                boundary.factor_fnv1a64,
                &byte,
                1U
            );
            if (distance > boundary.maximum_root_error) {
                boundary.maximum_root_error = distance;
            }
        }
    }
    return boundary;
}

static void free_boundary(struct boundary_record *boundary) {
    free(boundary->coefficient);
    boundary->coefficient = NULL;
    boundary->count = 0U;
}

static int boundary_differs(
    const struct boundary_record *first,
    const struct boundary_record *second
) {
    if (first->count != second->count) {
        return 1;
    }
    return memcmp(
        first->coefficient,
        second->coefficient,
        first->count * sizeof(*first->coefficient)
    ) != 0;
}

static int factors_differ(
    const double complex first[COEFFICIENT_COUNT],
    const double complex second[COEFFICIENT_COUNT]
) {
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        if (cabs(first[index] - second[index]) > ROOT_TOLERANCE) {
            return 1;
        }
    }
    return 0;
}

static int factor_nontrivial(
    const double complex factor[COEFFICIENT_COUNT]
) {
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        if (cabs(factor[index] - 1.0) > ROOT_TOLERANCE) {
            return 1;
        }
    }
    return 0;
}

static size_t find_omittable_message(
    const struct process *process,
    const struct carrier *carrier,
    int *applicable
) {
    *applicable = 0;
    for (size_t path_index = 0U; path_index < process->path_count; ++path_index) {
        const struct path *path = &process->path[path_index];
        for (size_t local = 0U; local < path->message_count; ++local) {
            double complex factor[COEFFICIENT_COUNT];
            read_oriented_factor(
                carrier,
                message_start(process, path->message_offset + local),
                0,
                factor
            );
            if (factor_nontrivial(factor)) {
                *applicable = 1;
                return path->message_offset + local;
            }
        }
    }
    return SIZE_MAX;
}

static double carrier_displacement(
    const struct carrier *carrier,
    const struct carrier *borrowed
) {
    double squared = 0.0;
    for (size_t index = 0U; index < carrier->cells; ++index) {
        const double difference = cabs(
            carrier->working[index] - borrowed->working[index]
        );
        squared += difference * difference;
    }
    return sqrt(squared);
}

static double restoration_error(
    const struct carrier *carrier,
    const struct carrier *borrowed
) {
    double maximum = 0.0;
    for (size_t index = 0U; index < carrier->cells; ++index) {
        const double difference = cabs(
            carrier->working[index] - borrowed->working[index]
        );
        if (difference > maximum) {
            maximum = difference;
        }
    }
    return maximum;
}

static double integrity_error(const struct carrier *carrier) {
    double maximum = 0.0;
    for (size_t index = 0U; index < carrier->cells; ++index) {
        const double error = fabs(cabs(carrier->working[index]) - 1.0);
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static struct execution execute(
    struct carrier *carrier,
    const struct process *process,
    enum inverse_mode mode
) {
    struct carrier borrowed = snapshot_carrier(carrier);
    for (size_t edge = 0U; edge < process->edge_count; ++edge) {
        apply_encoding(
            carrier,
            input_start(edge),
            process->edge[edge].coefficient,
            0
        );
    }
    const int bypass = mode == INVERSE_BYPASSED_PATH;
    const int scramble = mode == FORWARD_SCRAMBLED_TOPOLOGY;
    for (size_t path = 0U; path < process->path_count; ++path) {
        forward_path(process, carrier, path, bypass, scramble);
    }
    struct execution execution = {
        .boundary = latch_boundary(process, carrier)
    };
    execution.displacement_l2 = carrier_displacement(carrier, &borrowed);

    double complex first_factor[COEFFICIENT_COUNT] = {0};
    double complex previous_factor[COEFFICIENT_COUNT] = {0};
    for (size_t path = 0U; path < process->path_count; ++path) {
        double complex factor[COEFFICIENT_COUNT];
        double complex rotated[COEFFICIENT_COUNT];
        path_final_factor(
            process,
            carrier,
            &process->path[path],
            bypass,
            scramble,
            factor
        );
        for (size_t coefficient = 0U; coefficient < COEFFICIENT_COUNT; ++coefficient) {
            rotated[coefficient] = factor[
                (coefficient + 1U) % COEFFICIENT_COUNT
            ];
        }
        execution.wrong_applicable |= factors_differ(factor, rotated);
        if (path == 0U) {
            memcpy(first_factor, factor, sizeof(first_factor));
        } else {
            execution.topology_applicable |=
                factors_differ(previous_factor, factor);
        }
        memcpy(previous_factor, factor, sizeof(previous_factor));
    }
    execution.topology_applicable |=
        factors_differ(previous_factor, first_factor);
    const size_t omitted_message = find_omittable_message(
        process,
        carrier,
        &execution.omitted_message_applicable
    );

    if (mode == INVERSE_WRONG_BOUNDARY) {
        for (size_t path = 0U; path < process->path_count; ++path) {
            double complex factor[COEFFICIENT_COUNT];
            double complex rotated[COEFFICIENT_COUNT];
            path_final_factor(
                process,
                carrier,
                &process->path[path],
                0,
                0,
                factor
            );
            for (size_t coefficient = 0U; coefficient < COEFFICIENT_COUNT; ++coefficient) {
                rotated[coefficient] = factor[
                    (coefficient + 1U) % COEFFICIENT_COUNT
                ];
            }
            apply_factor(
                carrier,
                boundary_start(process, path),
                rotated,
                1
            );
        }
    } else {
        for (size_t path = process->path_count; path > 0U; --path) {
            inverse_path_boundary(
                process,
                carrier,
                path - 1U,
                bypass,
                scramble
            );
        }
    }

    if (!bypass) {
        for (size_t path = process->path_count; path > 0U; --path) {
            inverse_path_messages(
                process,
                carrier,
                &process->path[path - 1U],
                mode == INVERSE_OMITTED_MESSAGE
                    ? omitted_message
                    : SIZE_MAX
            );
        }
    }
    for (size_t edge = process->edge_count; edge > 0U; --edge) {
        apply_encoding(
            carrier,
            input_start(edge - 1U),
            process->edge[edge - 1U].coefficient,
            1
        );
    }
    execution.restoration_max_abs = restoration_error(carrier, &borrowed);
    execution.carrier_integrity_error = integrity_error(carrier);
    free_carrier(&borrowed);
    return execution;
}

static void print_execution(
    const char *mode,
    const struct process *process,
    const struct execution *execution
) {
    const int *first = execution->boundary.coefficient[0];
    const int *last = execution->boundary.coefficient[
        execution->boundary.count - 1U
    ];
    size_t maximum_path_edges = 0U;
    for (size_t index = 0U; index < process->path_count; ++index) {
        if (process->path[index].length > maximum_path_edges) {
            maximum_path_edges = process->path[index].length;
        }
    }
    printf(
        "{\"mode\":\"%s\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"port_type\":\"BOOLEAN_F3\","
        "\"node_count\":%zu,"
        "\"closed_internal_count\":%zu,"
        "\"external_count\":%zu,"
        "\"input_relation_count\":%zu,"
        "\"unique_external_paths\":%zu,"
        "\"phase_resident_relation_messages\":%zu,"
        "\"maximum_path_edges\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"native_resource_law\":\"O(E+sum_pair_path_length)\","
        "\"tuple_slots\":0,"
        "\"witness_slots\":0,"
        "\"truth_table_slots\":0,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"retained_inverse_factors\":0,"
        "\"boundary_factor_fnv1a64\":\"%016llx\","
        "\"first_factor\":[%d,%d,%d,%d],"
        "\"last_factor\":[%d,%d,%d,%d],"
        "\"maximum_root_error\":%.12g,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_error\":%.12g}\n",
        mode,
        (unsigned long long)process->source_fnv1a64,
        process->node_count,
        process->node_count - process->external_count,
        process->external_count,
        process->edge_count,
        process->path_count,
        process->message_count,
        maximum_path_edges,
        carrier_cells(process),
        (unsigned long long)execution->boundary.factor_fnv1a64,
        first[0], first[1], first[2], first[3],
        last[0], last[1], last[2], last[3],
        execution->boundary.maximum_root_error,
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->carrier_integrity_error
    );
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(
            stderr,
            "usage: %s PROCESS.art [REUSE_PROCESS.art]\n",
            argv[0]
        );
        return 2;
    }
    const struct process process = read_process(argv[1]);
    const struct process reuse_process =
        argc == 3 ? read_process(argv[2]) : read_process(argv[1]);
    if (carrier_cells(&process) != carrier_cells(&reuse_process)) {
        fail("reuse process must have identical carrier geometry");
    }

    struct carrier carrier = make_carrier(&process, 3109);
    struct execution nominal = execute(
        &carrier,
        &process,
        INVERSE_CORRECT
    );
    struct execution reuse = execute(
        &carrier,
        &reuse_process,
        INVERSE_CORRECT
    );
    free_carrier(&carrier);

    carrier = make_carrier(&process, 3109);
    struct execution wrong = execute(
        &carrier,
        &process,
        INVERSE_WRONG_BOUNDARY
    );
    free_carrier(&carrier);
    carrier = make_carrier(&process, 3109);
    struct execution scrambled = execute(
        &carrier,
        &process,
        FORWARD_SCRAMBLED_TOPOLOGY
    );
    free_carrier(&carrier);
    carrier = make_carrier(&process, 3109);
    struct execution omitted = execute(
        &carrier,
        &process,
        INVERSE_OMITTED_MESSAGE
    );
    free_carrier(&carrier);
    carrier = make_carrier(&process, 3109);
    struct execution bypass = execute(
        &carrier,
        &process,
        INVERSE_BYPASSED_PATH
    );
    free_carrier(&carrier);

    print_execution("algebraic-relation-tree", &process, &nominal);
    print_execution(
        argc == 3
            ? "actual-restored-cross-process-reuse"
            : "actual-restored-reuse",
        &reuse_process,
        &reuse
    );
    print_execution("wrong-boundary-inverse", &process, &wrong);
    print_execution(
        "scrambled-path-geometry-forward",
        &process,
        &scrambled
    );
    print_execution(
        "omitted-phase-message-inverse",
        &process,
        &omitted
    );
    print_execution(
        "bypassed-path-memory-forward",
        &process,
        &bypass
    );
    const int bypass_applicable =
        boundary_differs(&nominal.boundary, &bypass.boundary);
    const int topology_applicable =
        boundary_differs(&nominal.boundary, &scrambled.boundary);
    printf(
        "{\"mode\":\"control-applicability\","
        "\"wrong_boundary\":%s,"
        "\"topology_scrambled\":%s,"
        "\"omitted_phase_message\":%s,"
        "\"bypassed_path_memory\":%s}\n",
        wrong.wrong_applicable ? "true" : "false",
        topology_applicable ? "true" : "false",
        omitted.omitted_message_applicable ? "true" : "false",
        bypass_applicable ? "true" : "false"
    );

    const int valid = (
        nominal.boundary.maximum_root_error <= ROOT_TOLERANCE
        && reuse.boundary.maximum_root_error <= ROOT_TOLERANCE
        && bypass.boundary.maximum_root_error <= ROOT_TOLERANCE
        && nominal.restoration_max_abs <= RESTORATION_TOLERANCE
        && reuse.restoration_max_abs <= RESTORATION_TOLERANCE
        && scrambled.restoration_max_abs <= RESTORATION_TOLERANCE
        && bypass.restoration_max_abs <= RESTORATION_TOLERANCE
        && nominal.carrier_integrity_error <= RESTORATION_TOLERANCE
        && reuse.carrier_integrity_error <= RESTORATION_TOLERANCE
        && bypass.carrier_integrity_error <= RESTORATION_TOLERANCE
        && (
            !wrong.wrong_applicable
            || wrong.restoration_max_abs >= CONTROL_MINIMUM
        )
        && (
            !omitted.omitted_message_applicable
            || omitted.restoration_max_abs >= CONTROL_MINIMUM
        )
    );
    free_boundary(&nominal.boundary);
    free_boundary(&reuse.boundary);
    free_boundary(&wrong.boundary);
    free_boundary(&scrambled.boundary);
    free_boundary(&omitted.boundary);
    free_boundary(&bypass.boundary);
    return valid ? 0 : 1;
}
