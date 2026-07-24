#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: public series/parallel relational phase graphs.
 *
 * A public elimination order reduces degree-two internal nodes with exact
 * four-phase Boolean/F3 existential composition.  Parallel relations created
 * by elimination are merged with exact four-phase intersection.  Compilation
 * uses topology only; relation coefficients never feed schedule construction.
 */

#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CCOUNT 4U
#define MAX_NODES 48U
#define MAX_INPUTS 96U
#define MAX_ELIMINATIONS 46U
#define MAX_WORK_EDGES 256U
#define MAX_OPERATIONS 256U
#define MAX_IDENTIFIER 31U
#define LINE_CAPACITY 512U
#define TOKEN_CAPACITY 10U
#define PHASE_LOCK_PASSES 3U

static const double ROOT_TOLERANCE = 4.0e-10;
static const double RESTORATION_TOLERANCE = 2.0e-12;
static const double CONTROL_MINIMUM = 1.0e-3;

enum node_kind { NODE_EXTERNAL = 0, NODE_INTERNAL = 1 };
enum op_kind { OP_COMPOSE = 0, OP_INTERSECT = 1 };
enum execution_mode {
    MODE_CORRECT = 0,
    MODE_WRONG_BOUNDARY_INVERSE = 1,
    MODE_OMITTED_MESSAGE_INVERSE = 2,
    MODE_BYPASS_INTERSECTION = 3,
    MODE_ORDINARY_SUM_INTERSECTION = 4
};

struct node {
    char name[MAX_IDENTIFIER + 1U];
    enum node_kind kind;
};

struct input_relation {
    char name[MAX_IDENTIFIER + 1U];
    size_t first;
    size_t second;
    int coefficient[CCOUNT];
};

struct operation {
    enum op_kind kind;
    size_t left_start;
    int left_transposed;
    size_t right_start;
    int right_transposed;
    size_t output_start;
};

struct process {
    struct node node[MAX_NODES];
    struct input_relation input[MAX_INPUTS];
    size_t elimination[MAX_ELIMINATIONS];
    struct operation operation[MAX_OPERATIONS];
    size_t external[2];
    size_t node_count;
    size_t input_count;
    size_t elimination_count;
    size_t operation_count;
    size_t composition_count;
    size_t intersection_count;
    size_t final_start;
    int final_transposed;
    size_t carrier_cells;
    uint64_t source_hash;
};

struct work_edge {
    size_t first;
    size_t second;
    size_t start;
    int active;
};

struct carrier {
    double complex *baseline;
    double complex *working;
    size_t cells;
};

struct boundary {
    int coefficient[CCOUNT];
    uint64_t hash;
    double maximum_root_error;
};

struct execution {
    struct boundary boundary;
    double displacement_l2;
    double restoration_max_abs;
    double integrity_max_abs;
    int wrong_applicable;
    int omitted_applicable;
};

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void fail_line(const char *message, size_t line) {
    fprintf(stderr, "%s at line %zu\n", message, line);
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
    char *token[TOKEN_CAPACITY]
) {
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

static size_t find_node(
    const struct process *process,
    const char *name,
    size_t line
) {
    for (size_t index = 0U; index < process->node_count; ++index) {
        if (strcmp(process->node[index].name, name) == 0) {
            return index;
        }
    }
    fail_line("undeclared relation or elimination node", line);
    return 0U;
}

static size_t input_start(size_t input) {
    return input * CCOUNT;
}

static size_t operation_start(
    const struct process *process,
    size_t operation
) {
    return process->input_count * CCOUNT + operation * CCOUNT;
}

static size_t boundary_start(const struct process *process) {
    return (
        process->input_count + process->operation_count
    ) * CCOUNT;
}

static int edge_matches(
    const struct work_edge *edge,
    size_t first,
    size_t second
) {
    return edge->active && (
        (edge->first == first && edge->second == second)
        || (edge->first == second && edge->second == first)
    );
}

static size_t other_endpoint(
    const struct work_edge *edge,
    size_t node
) {
    if (edge->first == node) {
        return edge->second;
    }
    if (edge->second == node) {
        return edge->first;
    }
    fail("edge is not incident to requested node");
    return 0U;
}

static size_t add_operation(
    struct process *process,
    enum op_kind kind,
    size_t left_start,
    int left_transposed,
    size_t right_start,
    int right_transposed
) {
    if (process->operation_count == MAX_OPERATIONS) {
        fail("operation capacity exceeded");
    }
    const size_t index = process->operation_count++;
    struct operation *operation = &process->operation[index];
    operation->kind = kind;
    operation->left_start = left_start;
    operation->left_transposed = left_transposed;
    operation->right_start = right_start;
    operation->right_transposed = right_transposed;
    operation->output_start = operation_start(process, index);
    if (kind == OP_COMPOSE) {
        ++process->composition_count;
    } else {
        ++process->intersection_count;
    }
    return operation->output_start;
}

static size_t append_work_edge(
    struct work_edge edge[MAX_WORK_EDGES],
    size_t *edge_count,
    size_t first,
    size_t second,
    size_t start
) {
    if (*edge_count == MAX_WORK_EDGES) {
        fail("work-edge capacity exceeded");
    }
    const size_t index = (*edge_count)++;
    edge[index] = (struct work_edge){
        .first = first,
        .second = second,
        .start = start,
        .active = 1
    };
    return index;
}

static size_t merge_parallel_edges(
    struct process *process,
    struct work_edge edge[MAX_WORK_EDGES],
    size_t *edge_count,
    size_t current
) {
    for (;;) {
        const size_t first = edge[current].first;
        const size_t second = edge[current].second;
        size_t parallel = SIZE_MAX;
        for (size_t index = 0U; index < *edge_count; ++index) {
            if (
                index != current
                && edge_matches(&edge[index], first, second)
            ) {
                parallel = index;
                break;
            }
        }
        if (parallel == SIZE_MAX) {
            return current;
        }
        const int parallel_transposed =
            edge[parallel].first != first;
        const size_t output = add_operation(
            process,
            OP_INTERSECT,
            edge[current].start,
            0,
            edge[parallel].start,
            parallel_transposed
        );
        edge[current].active = 0;
        edge[parallel].active = 0;
        current = append_work_edge(
            edge,
            edge_count,
            first,
            second,
            output
        );
    }
}

static void compile_topology(struct process *process) {
    struct work_edge edge[MAX_WORK_EDGES] = {{0}};
    size_t edge_count = 0U;
    for (size_t index = 0U; index < process->input_count; ++index) {
        const size_t current = append_work_edge(
            edge,
            &edge_count,
            process->input[index].first,
            process->input[index].second,
            input_start(index)
        );
        (void)merge_parallel_edges(
            process,
            edge,
            &edge_count,
            current
        );
    }

    int eliminated[MAX_NODES] = {0};
    for (
        size_t order = 0U;
        order < process->elimination_count;
        ++order
    ) {
        const size_t node = process->elimination[order];
        if (process->node[node].kind != NODE_INTERNAL) {
            fail("external nodes cannot be eliminated");
        }
        if (eliminated[node]) {
            fail("duplicate elimination node");
        }
        eliminated[node] = 1;
        size_t incident[2] = {SIZE_MAX, SIZE_MAX};
        size_t incident_count = 0U;
        for (size_t index = 0U; index < edge_count; ++index) {
            if (
                edge[index].active
                && (
                    edge[index].first == node
                    || edge[index].second == node
                )
            ) {
                if (incident_count == 2U) {
                    fail("elimination node does not have active degree two");
                }
                incident[incident_count++] = index;
            }
        }
        if (incident_count != 2U) {
            fail("elimination node does not have active degree two");
        }
        size_t first_neighbor = other_endpoint(&edge[incident[0]], node);
        size_t second_neighbor = other_endpoint(&edge[incident[1]], node);
        if (first_neighbor == second_neighbor) {
            fail("elimination would create a self relation");
        }
        if (second_neighbor < first_neighbor) {
            const size_t swap_neighbor = first_neighbor;
            const size_t swap_edge = incident[0];
            first_neighbor = second_neighbor;
            second_neighbor = swap_neighbor;
            incident[0] = incident[1];
            incident[1] = swap_edge;
        }
        const int left_transposed = !(
            edge[incident[0]].first == first_neighbor
            && edge[incident[0]].second == node
        );
        const int right_transposed = !(
            edge[incident[1]].first == node
            && edge[incident[1]].second == second_neighbor
        );
        const size_t output = add_operation(
            process,
            OP_COMPOSE,
            edge[incident[0]].start,
            left_transposed,
            edge[incident[1]].start,
            right_transposed
        );
        edge[incident[0]].active = 0;
        edge[incident[1]].active = 0;
        size_t current = append_work_edge(
            edge,
            &edge_count,
            first_neighbor,
            second_neighbor,
            output
        );
        current = merge_parallel_edges(
            process,
            edge,
            &edge_count,
            current
        );
        (void)current;
    }

    size_t internal_count = 0U;
    for (size_t index = 0U; index < process->node_count; ++index) {
        if (process->node[index].kind == NODE_INTERNAL) {
            ++internal_count;
            if (!eliminated[index]) {
                fail("every internal node needs one elimination record");
            }
        }
    }
    if (internal_count != process->elimination_count) {
        fail("elimination count does not match internal nodes");
    }
    size_t final_edge = SIZE_MAX;
    size_t active_count = 0U;
    for (size_t index = 0U; index < edge_count; ++index) {
        if (edge[index].active) {
            ++active_count;
            final_edge = index;
        }
    }
    if (
        active_count != 1U
        || !edge_matches(
            &edge[final_edge],
            process->external[0],
            process->external[1]
        )
    ) {
        fail("elimination did not reduce to one external relation");
    }
    if (process->intersection_count == 0U) {
        fail("series-parallel graph must contain a parallel-path merge");
    }
    process->final_start = edge[final_edge].start;
    process->final_transposed =
        edge[final_edge].first != process->external[0];
    process->carrier_cells = (
        process->input_count
        + process->operation_count
        + 1U
    ) * CCOUNT;
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
    char line[LINE_CAPACITY];
    size_t line_number = 0U;
    int header = 0;
    int type = 0;
    int relation_seen = 0;
    int elimination_seen = 0;
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
            fail_line("every record must end with LF", line_number);
        }
        if (memchr(line, '\r', length) != NULL) {
            fail_line("CR bytes are forbidden", line_number);
        }
        line[length - 1U] = '\0';
        if (end) {
            fail_line("record after END", line_number);
        }
        char *token[TOKEN_CAPACITY] = {0};
        const size_t count = tokenize(line, token);
        if (count == 0U) {
            fail_line("blank records are forbidden", line_number);
        }
        if (!header) {
            if (
                count != 2U
                || strcmp(token[0], "CATCAS_SERIES_PARALLEL_RELATION") != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid series-parallel header", line_number);
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
            if (!type || relation_seen || count != 3U) {
                fail_line("invalid NODE", line_number);
            }
            if (
                process.node_count == MAX_NODES
                || !valid_identifier(token[1])
            ) {
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
        } else if (strcmp(token[0], "RELATION") == 0) {
            if (
                !type
                || process.node_count == 0U
                || elimination_seen
                || count != 8U
            ) {
                fail_line("invalid RELATION", line_number);
            }
            relation_seen = 1;
            if (
                process.input_count == MAX_INPUTS
                || !valid_identifier(token[1])
            ) {
                fail_line("invalid relation identifier", line_number);
            }
            for (size_t index = 0U; index < process.input_count; ++index) {
                if (strcmp(process.input[index].name, token[1]) == 0) {
                    fail_line("duplicate relation identifier", line_number);
                }
            }
            struct input_relation *input =
                &process.input[process.input_count++];
            memcpy(input->name, token[1], strlen(token[1]) + 1U);
            input->first = find_node(&process, token[2], line_number);
            input->second = find_node(&process, token[3], line_number);
            if (input->first == input->second) {
                fail_line("self relations are forbidden", line_number);
            }
            for (size_t index = 0U; index < CCOUNT; ++index) {
                if (
                    strlen(token[4U + index]) != 1U
                    || token[4U + index][0] < '0'
                    || token[4U + index][0] > '2'
                ) {
                    fail_line("invalid F3 coefficient", line_number);
                }
                input->coefficient[index] =
                    token[4U + index][0] - '0';
            }
        } else if (strcmp(token[0], "ELIMINATE") == 0) {
            if (
                !relation_seen
                || count != 2U
                || process.elimination_count == MAX_ELIMINATIONS
            ) {
                fail_line("invalid ELIMINATE", line_number);
            }
            elimination_seen = 1;
            process.elimination[process.elimination_count++] =
                find_node(&process, token[1], line_number);
        } else if (strcmp(token[0], "END") == 0) {
            if (count != 1U) {
                fail_line("invalid END", line_number);
            }
            end = 1;
        } else {
            fail_line("unknown series-parallel record", line_number);
        }
    }
    if (ferror(stream) || fclose(stream) != 0) {
        fail("failed to read series-parallel process");
    }
    if (
        !header
        || !type
        || !relation_seen
        || !elimination_seen
        || !end
    ) {
        fail("series-parallel process is incomplete");
    }
    size_t external_count = 0U;
    for (size_t index = 0U; index < process.node_count; ++index) {
        if (process.node[index].kind == NODE_EXTERNAL) {
            if (external_count == 2U) {
                fail("exactly two external nodes are required");
            }
            process.external[external_count++] = index;
        }
    }
    if (external_count != 2U) {
        fail("exactly two external nodes are required");
    }
    compile_topology(&process);
    return process;
}

static double complex unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        fail("nonfinite phase state");
    }
    return value / magnitude;
}

/*
 * Continuous three-well phase lock.  Every legal F3 phase is a fixed point:
 * for z^3=1, conj(z)^2=z.  The first-order phase error cancels in
 * 2z+conj(z)^2, so repeated applications suppress floating drift without
 * decoding an intermediate coefficient or branching on a phase label.
 */
static double complex lock_f3_phase(double complex value) {
    double complex locked = unit(value);
    for (size_t pass = 0U; pass < PHASE_LOCK_PASSES; ++pass) {
        locked = unit(2.0 * locked + conj(locked) * conj(locked));
    }
    return locked;
}

static double complex root3(int amount) {
    const int value = f3(amount);
    if (value == 0) {
        return 1.0 + 0.0 * I;
    }
    if (value == 1) {
        return -0.5 + 0.86602540378443864676 * I;
    }
    return -0.5 - 0.86602540378443864676 * I;
}

static struct carrier make_carrier(
    const struct process *process,
    int identity
) {
    struct carrier carrier = {
        .baseline = checked_calloc(
            process->carrier_cells,
            sizeof(*carrier.baseline)
        ),
        .working = checked_calloc(
            process->carrier_cells,
            sizeof(*carrier.working)
        ),
        .cells = process->carrier_cells
    };
    for (size_t index = 0U; index < carrier.cells; ++index) {
        const double angle =
            0.229
            + 0.061 * (double)index
            + 0.014 * sin(0.27 * (double)index + 0.023 * identity);
        carrier.baseline[index] = cexp(I * angle);
        carrier.working[index] = carrier.baseline[index];
    }
    return carrier;
}

static struct carrier snapshot_carrier(const struct carrier *source) {
    struct carrier snapshot = {
        .baseline = checked_calloc(
            source->cells,
            sizeof(*snapshot.baseline)
        ),
        .working = checked_calloc(
            source->cells,
            sizeof(*snapshot.working)
        ),
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

static double complex relative(
    const struct carrier *carrier,
    size_t cell
) {
    return carrier->working[cell] * conj(carrier->baseline[cell]);
}

static void multiply_cell(
    struct carrier *carrier,
    size_t cell,
    double complex factor
) {
    carrier->working[cell] = unit(
        relative(carrier, cell) * factor
    ) * carrier->baseline[cell];
}

static size_t oriented_cell(
    size_t start,
    size_t coefficient,
    int transposed
) {
    static const size_t swapped[CCOUNT] = {0U, 2U, 1U, 3U};
    return start + (transposed ? swapped[coefficient] : coefficient);
}

static void read_poly(
    const struct carrier *carrier,
    size_t start,
    int transposed,
    double complex output[CCOUNT]
) {
    for (size_t index = 0U; index < CCOUNT; ++index) {
        output[index] = relative(
            carrier,
            oriented_cell(start, index, transposed)
        );
    }
}

static double complex symbol_product(
    double complex left,
    double complex right
) {
    const double complex left_squared = conj(left);
    const double complex right_squared = conj(right);
    const double complex product = left * right;
    return lock_f3_phase(
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

static void poly_multiply(
    const double complex left[CCOUNT],
    const double complex right[CCOUNT],
    double complex output[CCOUNT]
) {
    for (size_t out = 0U; out < CCOUNT; ++out) {
        output[out] = 1.0 + 0.0 * I;
        for (size_t l = 0U; l < CCOUNT; ++l) {
            for (size_t r = 0U; r < CCOUNT; ++r) {
                if ((l | r) == out) {
                    output[out] = unit(
                        output[out]
                        * symbol_product(left[l], right[r])
                    );
                }
            }
        }
    }
}

static void poly_add(
    const double complex left[CCOUNT],
    const double complex right[CCOUNT],
    double complex output[CCOUNT]
) {
    for (size_t index = 0U; index < CCOUNT; ++index) {
        output[index] = lock_f3_phase(left[index] * right[index]);
    }
}

static void exact_compose_factors(
    const struct carrier *carrier,
    const struct operation *operation,
    double complex output[CCOUNT]
) {
    double complex left[CCOUNT];
    double complex right[CCOUNT];
    read_poly(
        carrier,
        operation->left_start,
        operation->left_transposed,
        left
    );
    read_poly(
        carrier,
        operation->right_start,
        operation->right_transposed,
        right
    );
    const double complex zero = root3(0);
    double complex f0[CCOUNT] = {left[0], left[1], zero, zero};
    double complex f1[CCOUNT] = {
        unit(left[0] * left[2]),
        unit(left[1] * left[3]),
        zero,
        zero
    };
    double complex g0[CCOUNT] = {right[0], zero, right[2], zero};
    double complex g1[CCOUNT] = {
        unit(right[0] * right[1]),
        zero,
        unit(right[2] * right[3]),
        zero
    };
    double complex f0s[CCOUNT];
    double complex f1s[CCOUNT];
    double complex g0s[CCOUNT];
    double complex g1s[CCOUNT];
    double complex k0[CCOUNT];
    double complex k1[CCOUNT];
    poly_multiply(f0, f0, f0s);
    poly_multiply(f1, f1, f1s);
    poly_multiply(g0, g0, g0s);
    poly_multiply(g1, g1, g1s);
    poly_add(f0s, g0s, k0);
    poly_add(f1s, g1s, k1);
    poly_multiply(k0, k1, output);
}

static void intersection_factors(
    const struct carrier *carrier,
    const struct operation *operation,
    int ordinary_sum,
    double complex output[CCOUNT]
) {
    double complex left[CCOUNT];
    double complex right[CCOUNT];
    read_poly(
        carrier,
        operation->left_start,
        operation->left_transposed,
        left
    );
    read_poly(
        carrier,
        operation->right_start,
        operation->right_transposed,
        right
    );
    if (ordinary_sum) {
        poly_add(left, right, output);
        return;
    }
    double complex left_squared[CCOUNT];
    double complex right_squared[CCOUNT];
    poly_multiply(left, left, left_squared);
    poly_multiply(right, right, right_squared);
    poly_add(left_squared, right_squared, output);
}

static void apply_factor(
    struct carrier *carrier,
    size_t output_start,
    const double complex factor[CCOUNT],
    int inverse
) {
    for (size_t index = 0U; index < CCOUNT; ++index) {
        multiply_cell(
            carrier,
            output_start + index,
            inverse ? conj(factor[index]) : factor[index]
        );
    }
}

static void operation_factors(
    const struct carrier *carrier,
    const struct operation *operation,
    enum execution_mode mode,
    size_t controlled_intersection,
    size_t operation_index,
    double complex factor[CCOUNT]
) {
    if (operation->kind == OP_COMPOSE) {
        exact_compose_factors(carrier, operation, factor);
        return;
    }
    if (
        operation_index == controlled_intersection
        && mode == MODE_BYPASS_INTERSECTION
    ) {
        read_poly(
            carrier,
            operation->left_start,
            operation->left_transposed,
            factor
        );
        return;
    }
    intersection_factors(
        carrier,
        operation,
        operation_index == controlled_intersection
            && mode == MODE_ORDINARY_SUM_INTERSECTION,
        factor
    );
}

static void apply_operation(
    struct carrier *carrier,
    const struct operation *operation,
    enum execution_mode mode,
    size_t controlled_intersection,
    size_t operation_index,
    int inverse
) {
    double complex factor[CCOUNT];
    operation_factors(
        carrier,
        operation,
        mode,
        controlled_intersection,
        operation_index,
        factor
    );
    apply_factor(carrier, operation->output_start, factor, inverse);
}

static void apply_encoding(
    struct carrier *carrier,
    size_t start,
    const int coefficient[CCOUNT],
    int inverse
) {
    for (size_t index = 0U; index < CCOUNT; ++index) {
        const double complex factor = root3(coefficient[index]);
        multiply_cell(
            carrier,
            start + index,
            inverse ? conj(factor) : factor
        );
    }
}

static int decode_root(
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

static struct boundary latch_boundary(
    const struct process *process,
    const struct carrier *carrier
) {
    struct boundary boundary = {
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t external = 0U; external < 2U; ++external) {
        const char *name = process->node[process->external[external]].name;
        boundary.hash = hash_bytes(
            boundary.hash,
            (const unsigned char *)name,
            strlen(name)
        );
        if (external == 0U) {
            const unsigned char separator = ':';
            boundary.hash = hash_bytes(
                boundary.hash,
                &separator,
                1U
            );
        }
    }
    for (size_t index = 0U; index < CCOUNT; ++index) {
        double distance = 0.0;
        boundary.coefficient[index] = decode_root(
            relative(carrier, boundary_start(process) + index),
            &distance
        );
        const unsigned char byte =
            (unsigned char)boundary.coefficient[index];
        boundary.hash = hash_bytes(boundary.hash, &byte, 1U);
        if (distance > boundary.maximum_root_error) {
            boundary.maximum_root_error = distance;
        }
    }
    return boundary;
}

static int boundary_differs(
    const struct boundary *first,
    const struct boundary *second
) {
    return memcmp(
        first->coefficient,
        second->coefficient,
        sizeof(first->coefficient)
    ) != 0;
}

static int factor_nontrivial(
    const double complex factor[CCOUNT]
) {
    for (size_t index = 0U; index < CCOUNT; ++index) {
        if (cabs(factor[index] - 1.0) > ROOT_TOLERANCE) {
            return 1;
        }
    }
    return 0;
}

static double displacement(
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

static double restoration(
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

static double integrity(const struct carrier *carrier) {
    double maximum = 0.0;
    for (size_t index = 0U; index < carrier->cells; ++index) {
        const double error = fabs(cabs(carrier->working[index]) - 1.0);
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static size_t first_intersection(const struct process *process) {
    for (size_t index = 0U; index < process->operation_count; ++index) {
        if (process->operation[index].kind == OP_INTERSECT) {
            return index;
        }
    }
    fail("compiled process has no intersection");
    return 0U;
}

static struct execution execute(
    struct carrier *carrier,
    const struct process *process,
    enum execution_mode mode
) {
    struct carrier borrowed = snapshot_carrier(carrier);
    for (size_t input = 0U; input < process->input_count; ++input) {
        apply_encoding(
            carrier,
            input_start(input),
            process->input[input].coefficient,
            0
        );
    }
    const size_t controlled_intersection = first_intersection(process);
    for (
        size_t operation = 0U;
        operation < process->operation_count;
        ++operation
    ) {
        apply_operation(
            carrier,
            &process->operation[operation],
            mode,
            controlled_intersection,
            operation,
            0
        );
    }
    double complex boundary_factor[CCOUNT];
    read_poly(
        carrier,
        process->final_start,
        process->final_transposed,
        boundary_factor
    );
    apply_factor(
        carrier,
        boundary_start(process),
        boundary_factor,
        0
    );
    struct execution execution = {
        .boundary = latch_boundary(process, carrier),
        .displacement_l2 = displacement(carrier, &borrowed)
    };
    double complex rotated[CCOUNT];
    for (size_t index = 0U; index < CCOUNT; ++index) {
        rotated[index] = boundary_factor[(index + 1U) % CCOUNT];
    }
    for (size_t index = 0U; index < CCOUNT; ++index) {
        execution.wrong_applicable |=
            cabs(rotated[index] - boundary_factor[index])
                > ROOT_TOLERANCE;
    }
    size_t omitted_operation = SIZE_MAX;
    for (
        size_t operation = process->operation_count;
        operation > 0U;
        --operation
    ) {
        double complex factor[CCOUNT];
        read_poly(
            carrier,
            process->operation[operation - 1U].output_start,
            0,
            factor
        );
        if (factor_nontrivial(factor)) {
            omitted_operation = operation - 1U;
            execution.omitted_applicable = 1;
            break;
        }
    }
    apply_factor(
        carrier,
        boundary_start(process),
        mode == MODE_WRONG_BOUNDARY_INVERSE
            ? rotated
            : boundary_factor,
        1
    );
    for (
        size_t operation = process->operation_count;
        operation > 0U;
        --operation
    ) {
        const size_t index = operation - 1U;
        if (
            mode == MODE_OMITTED_MESSAGE_INVERSE
            && index == omitted_operation
        ) {
            continue;
        }
        apply_operation(
            carrier,
            &process->operation[index],
            mode,
            controlled_intersection,
            index,
            1
        );
    }
    for (size_t input = process->input_count; input > 0U; --input) {
        apply_encoding(
            carrier,
            input_start(input - 1U),
            process->input[input - 1U].coefficient,
            1
        );
    }
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
    free_carrier(&borrowed);
    return execution;
}

static void print_execution(
    const char *mode,
    const struct process *process,
    const struct execution *execution
) {
    printf(
        "{\"mode\":\"%s\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"topology\":\"PUBLIC_TWO_TERMINAL_SERIES_PARALLEL\","
        "\"node_count\":%zu,"
        "\"input_relation_count\":%zu,"
        "\"eliminated_internal_nodes\":%zu,"
        "\"native_composition_operations\":%zu,"
        "\"native_intersection_operations\":%zu,"
        "\"phase_resident_relation_messages\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"tuple_slots\":0,"
        "\"witness_slots\":0,"
        "\"truth_table_slots\":0,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"retained_inverse_factors\":0,"
        "\"boundary_factor_fnv1a64\":\"%016llx\","
        "\"boundary_coefficients\":[%d,%d,%d,%d],"
        "\"maximum_root_error\":%.12g,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_max_abs\":%.12g}\n",
        mode,
        (unsigned long long)process->source_hash,
        process->node_count,
        process->input_count,
        process->elimination_count,
        process->composition_count,
        process->intersection_count,
        process->operation_count,
        process->carrier_cells,
        (unsigned long long)execution->boundary.hash,
        execution->boundary.coefficient[0],
        execution->boundary.coefficient[1],
        execution->boundary.coefficient[2],
        execution->boundary.coefficient[3],
        execution->boundary.maximum_root_error,
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->integrity_max_abs
    );
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(
            stderr,
            "usage: %s PROCESS.aspr [REUSE_PROCESS.aspr]\n",
            argv[0]
        );
        return 2;
    }
    const struct process process = read_process(argv[1]);
    const struct process reuse_process =
        argc == 3 ? read_process(argv[2]) : read_process(argv[1]);
    if (
        process.carrier_cells != reuse_process.carrier_cells
        || process.input_count != reuse_process.input_count
        || process.operation_count != reuse_process.operation_count
    ) {
        fail("reuse process must have identical carrier geometry");
    }

    struct carrier carrier = make_carrier(&process, 5303);
    const struct execution nominal = execute(
        &carrier,
        &process,
        MODE_CORRECT
    );
    const struct execution reuse = execute(
        &carrier,
        &reuse_process,
        MODE_CORRECT
    );
    free_carrier(&carrier);

    carrier = make_carrier(&process, 5303);
    const struct execution wrong = execute(
        &carrier,
        &process,
        MODE_WRONG_BOUNDARY_INVERSE
    );
    free_carrier(&carrier);
    carrier = make_carrier(&process, 5303);
    const struct execution omitted = execute(
        &carrier,
        &process,
        MODE_OMITTED_MESSAGE_INVERSE
    );
    free_carrier(&carrier);
    carrier = make_carrier(&process, 5303);
    const struct execution bypass = execute(
        &carrier,
        &process,
        MODE_BYPASS_INTERSECTION
    );
    free_carrier(&carrier);
    carrier = make_carrier(&process, 5303);
    const struct execution ordinary = execute(
        &carrier,
        &process,
        MODE_ORDINARY_SUM_INTERSECTION
    );
    free_carrier(&carrier);

    print_execution("series-parallel-relational-phase", &process, &nominal);
    print_execution(
        argc == 3
            ? "actual-restored-cross-process-reuse"
            : "actual-restored-reuse",
        &reuse_process,
        &reuse
    );
    print_execution("wrong-boundary-inverse", &process, &wrong);
    print_execution(
        "omitted-message-inverse",
        &process,
        &omitted
    );
    print_execution(
        "bypassed-first-parallel-intersection",
        &process,
        &bypass
    );
    print_execution(
        "ordinary-sum-first-parallel-intersection",
        &process,
        &ordinary
    );
    const int bypass_applicable =
        boundary_differs(&nominal.boundary, &bypass.boundary);
    const int ordinary_applicable =
        boundary_differs(&nominal.boundary, &ordinary.boundary);
    printf(
        "{\"mode\":\"control-applicability\","
        "\"wrong_boundary\":%s,"
        "\"omitted_message\":%s,"
        "\"bypassed_intersection\":%s,"
        "\"ordinary_sum\":%s}\n",
        wrong.wrong_applicable ? "true" : "false",
        omitted.omitted_applicable ? "true" : "false",
        bypass_applicable ? "true" : "false",
        ordinary_applicable ? "true" : "false"
    );
    const int valid = (
        nominal.boundary.maximum_root_error <= ROOT_TOLERANCE
        && reuse.boundary.maximum_root_error <= ROOT_TOLERANCE
        && bypass.boundary.maximum_root_error <= ROOT_TOLERANCE
        && ordinary.boundary.maximum_root_error <= ROOT_TOLERANCE
        && nominal.restoration_max_abs <= RESTORATION_TOLERANCE
        && reuse.restoration_max_abs <= RESTORATION_TOLERANCE
        && bypass.restoration_max_abs <= RESTORATION_TOLERANCE
        && ordinary.restoration_max_abs <= RESTORATION_TOLERANCE
        && (
            !wrong.wrong_applicable
            || wrong.restoration_max_abs >= CONTROL_MINIMUM
        )
        && (
            !omitted.omitted_applicable
            || omitted.restoration_max_abs >= CONTROL_MINIMUM
        )
    );
    return valid ? 0 : 1;
}
