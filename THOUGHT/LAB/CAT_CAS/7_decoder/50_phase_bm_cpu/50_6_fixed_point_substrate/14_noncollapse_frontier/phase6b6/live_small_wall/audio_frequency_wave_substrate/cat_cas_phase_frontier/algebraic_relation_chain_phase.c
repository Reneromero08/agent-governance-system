/*
 * Mutable CAT_CAS frontier: repeatable algebraic relation chains.
 *
 * Every local relation is a bi-total Boolean_F3 multiaffine zero set.  The
 * bi-total class is closed under relational composition.  Each internal port
 * is closed by the same phase-native resultant used in the reviewed
 * single-port calibration.  The carrier retains one four-phase derived
 * relation per composition layer so the actual forward traversal can be
 * reversed without scalar answers or stored inverse factors.
 */

#define _GNU_SOURCE
#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define COEFFICIENT_COUNT 4U
#define MAX_NAME 31U
#define MAX_TOKENS 12U
#define RESTORATION_TOLERANCE 2e-12
#define CONTROL_MINIMUM 1e-3
#define ROOT_TOLERANCE 2e-10

enum inverse_mode {
    INVERSE_CORRECT = 0,
    INVERSE_WRONG = 1,
    INVERSE_REORDERED = 2,
    INVERSE_OMITTED = 3
};

struct relation_definition {
    char name[MAX_NAME + 1U];
    char first[MAX_NAME + 1U];
    char second[MAX_NAME + 1U];
    int coefficient[COEFFICIENT_COUNT];
};

struct process {
    struct relation_definition *relation;
    size_t relation_count;
    char boundary_first[MAX_NAME + 1U];
    char boundary_second[MAX_NAME + 1U];
    uint64_t source_fnv1a64;
};

struct endpoint_entry {
    char port[MAX_NAME + 1U];
    size_t relation_index;
};

struct relation_name_entry {
    char name[MAX_NAME + 1U];
};

struct carrier {
    double complex *baseline;
    double complex *working;
    size_t cells;
};

struct boundary_record {
    double complex phase[COEFFICIENT_COUNT];
    int coefficient[COEFFICIENT_COUNT];
    double maximum_root_error;
};

struct execution {
    struct boundary_record boundary;
    double displacement_l2;
    double restoration_max_abs;
    double carrier_integrity_error;
    int wrong_applicable;
    int reordered_applicable;
    int omitted_applicable;
};

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void fail_line(const char *message, size_t line_number) {
    fprintf(stderr, "line %zu: %s\n", line_number, message);
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

static void *checked_reallocarray(
    void *memory,
    size_t count,
    size_t size
) {
    if (count != 0U && size > SIZE_MAX / count) {
        fail("allocation size overflow");
    }
    void *result = realloc(memory, count * size);
    if (result == NULL) {
        fail("allocation failed");
    }
    return result;
}

static uint64_t fnv1a64_update(
    uint64_t hash,
    const unsigned char *bytes,
    size_t length
) {
    for (size_t index = 0U; index < length; ++index) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int valid_name(const char *text) {
    const size_t length = strlen(text);
    if (
        length == 0U
        || length > MAX_NAME
        || !(isalpha((unsigned char)text[0]) || text[0] == '_')
    ) {
        return 0;
    }
    for (size_t index = 1U; index < length; ++index) {
        if (
            !isalnum((unsigned char)text[index])
            && text[index] != '_'
        ) {
            return 0;
        }
    }
    return 1;
}

static void copy_name(
    char target[MAX_NAME + 1U],
    const char *source,
    size_t line_number
) {
    if (!valid_name(source)) {
        fail_line("invalid identifier", line_number);
    }
    memcpy(target, source, strlen(source) + 1U);
}

static char *trim(char *text) {
    while (*text == ' ' || *text == '\t') {
        ++text;
    }
    char *end = text + strlen(text);
    while (end > text && (end[-1] == ' ' || end[-1] == '\t')) {
        --end;
    }
    *end = '\0';
    return text;
}

static size_t split_tokens(
    char *line,
    char *tokens[MAX_TOKENS]
) {
    size_t count = 0U;
    char *cursor = line;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        if (count == MAX_TOKENS) {
            fail("too many tokens");
        }
        tokens[count++] = cursor;
        while (
            *cursor != '\0'
            && *cursor != ' '
            && *cursor != '\t'
        ) {
            ++cursor;
        }
        if (*cursor != '\0') {
            *cursor++ = '\0';
        }
    }
    return count;
}

static int parse_coefficient(
    const char *text,
    size_t line_number
) {
    if (
        text[0] < '0'
        || text[0] > '2'
        || text[1] != '\0'
    ) {
        fail_line("coefficient must be canonical 0, 1, or 2", line_number);
    }
    return text[0] - '0';
}

static int mod3(int value) {
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static int affine_has_boolean_zero(int slope, int offset) {
    return offset == 0 || mod3(slope + offset) == 0;
}

static int total_toward_second(
    const struct relation_definition *relation
) {
    return (
        affine_has_boolean_zero(
            relation->coefficient[2],
            relation->coefficient[0]
        )
        && affine_has_boolean_zero(
            mod3(
                relation->coefficient[2]
                + relation->coefficient[3]
            ),
            mod3(
                relation->coefficient[0]
                + relation->coefficient[1]
            )
        )
    );
}

static int total_toward_first(
    const struct relation_definition *relation
) {
    return (
        affine_has_boolean_zero(
            relation->coefficient[1],
            relation->coefficient[0]
        )
        && affine_has_boolean_zero(
            mod3(
                relation->coefficient[1]
                + relation->coefficient[3]
            ),
            mod3(
                relation->coefficient[0]
                + relation->coefficient[2]
            )
        )
    );
}

static struct relation_definition orient(
    const struct relation_definition *source,
    const char *first
) {
    struct relation_definition result = *source;
    if (strcmp(source->first, first) == 0) {
        return result;
    }
    if (strcmp(source->second, first) != 0) {
        fail("relation does not touch expected chain port");
    }
    memcpy(
        result.first,
        source->second,
        strlen(source->second) + 1U
    );
    memcpy(
        result.second,
        source->first,
        strlen(source->first) + 1U
    );
    const int old_10 = result.coefficient[1];
    result.coefficient[1] = result.coefficient[2];
    result.coefficient[2] = old_10;
    return result;
}

static int compare_endpoints(
    const void *left_value,
    const void *right_value
) {
    const struct endpoint_entry *left = left_value;
    const struct endpoint_entry *right = right_value;
    const int by_port = strcmp(left->port, right->port);
    if (by_port != 0) {
        return by_port;
    }
    if (left->relation_index < right->relation_index) {
        return -1;
    }
    return left->relation_index > right->relation_index ? 1 : 0;
}

static int compare_relation_names(
    const void *left_value,
    const void *right_value
) {
    const struct relation_name_entry *left = left_value;
    const struct relation_name_entry *right = right_value;
    return strcmp(left->name, right->name);
}

static size_t endpoint_lower_bound(
    const struct endpoint_entry *endpoint,
    size_t count,
    const char *port
) {
    size_t lower = 0U;
    size_t upper = count;
    while (lower < upper) {
        const size_t middle = lower + (upper - lower) / 2U;
        if (strcmp(endpoint[middle].port, port) < 0) {
            lower = middle + 1U;
        } else {
            upper = middle;
        }
    }
    return lower;
}

static void normalize_chain(struct process *process) {
    if (
        process->relation_count < 2U
        || strcmp(
            process->boundary_first,
            process->boundary_second
        ) == 0
    ) {
        fail("chain requires distinct boundaries and at least two relations");
    }
    if (process->relation_count > SIZE_MAX / 2U) {
        fail("endpoint count overflow");
    }
    struct relation_definition *ordered = checked_calloc(
        process->relation_count,
        sizeof(*ordered)
    );
    unsigned char *used = checked_calloc(
        process->relation_count,
        sizeof(*used)
    );
    struct relation_name_entry *names = checked_calloc(
        process->relation_count,
        sizeof(*names)
    );
    const size_t endpoint_count = 2U * process->relation_count;
    struct endpoint_entry *endpoint = checked_calloc(
        endpoint_count,
        sizeof(*endpoint)
    );
    for (size_t index = 0U; index < process->relation_count; ++index) {
        if (
            strcmp(
                process->relation[index].first,
                process->relation[index].second
            ) == 0
        ) {
            fail("relation has identical endpoints");
        }
        memcpy(
            names[index].name,
            process->relation[index].name,
            strlen(process->relation[index].name) + 1U
        );
        memcpy(
            endpoint[2U * index].port,
            process->relation[index].first,
            strlen(process->relation[index].first) + 1U
        );
        endpoint[2U * index].relation_index = index;
        memcpy(
            endpoint[2U * index + 1U].port,
            process->relation[index].second,
            strlen(process->relation[index].second) + 1U
        );
        endpoint[2U * index + 1U].relation_index = index;
    }
    qsort(
        names,
        process->relation_count,
        sizeof(*names),
        compare_relation_names
    );
    for (size_t index = 1U; index < process->relation_count; ++index) {
        if (strcmp(names[index - 1U].name, names[index].name) == 0) {
            fail("duplicate relation name");
        }
    }
    free(names);
    qsort(
        endpoint,
        endpoint_count,
        sizeof(*endpoint),
        compare_endpoints
    );
    char current[MAX_NAME + 1U];
    memcpy(
        current,
        process->boundary_first,
        strlen(process->boundary_first) + 1U
    );

    for (
        size_t position = 0U;
        position < process->relation_count;
        ++position
    ) {
        size_t match = SIZE_MAX;
        size_t matches = 0U;
        const size_t first_endpoint = endpoint_lower_bound(
            endpoint,
            endpoint_count,
            current
        );
        for (
            size_t index = first_endpoint;
            index < endpoint_count
                && strcmp(endpoint[index].port, current) == 0;
            ++index
        ) {
            const size_t relation_index = endpoint[index].relation_index;
            if (!used[relation_index]) {
                match = relation_index;
                ++matches;
            }
        }
        if (matches != 1U) {
            fail("relation geometry is branching, cyclic, or disconnected");
        }
        ordered[position] = orient(&process->relation[match], current);
        used[match] = 1U;
        const char *next = ordered[position].second;
        const int is_final = position + 1U == process->relation_count;
        if (
            (!is_final && strcmp(next, process->boundary_second) == 0)
            || (
                is_final
                && strcmp(next, process->boundary_second) != 0
            )
        ) {
            fail("relation chain does not terminate at the declared boundary");
        }
        memcpy(current, next, strlen(next) + 1U);
    }

    for (size_t index = 0U; index < process->relation_count; ++index) {
        if (
            !total_toward_second(&ordered[index])
            || !total_toward_first(&ordered[index])
        ) {
            fail("relation is not bi-total on Boolean_F3");
        }
    }
    free(process->relation);
    process->relation = ordered;
    free(used);
    free(endpoint);
}

static struct process read_process(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }

    struct process process = {
        .relation = NULL,
        .relation_count = 0U,
        .source_fnv1a64 = UINT64_C(14695981039346656037)
    };
    size_t capacity = 0U;
    int header_seen = 0;
    int type_seen = 0;
    int boundary_seen = 0;
    int end_seen = 0;
    size_t line_number = 0U;
    char *buffer = NULL;
    size_t buffer_capacity = 0U;
    ssize_t length_read = 0;
    while (
        (length_read = getline(
            &buffer,
            &buffer_capacity,
            stream
        )) >= 0
    ) {
        const size_t raw_length = (size_t)length_read;
        process.source_fnv1a64 = fnv1a64_update(
            process.source_fnv1a64,
            (const unsigned char *)buffer,
            raw_length
        );
        if (memchr(buffer, '\0', raw_length) != NULL) {
            fail("embedded NUL");
        }
        if (memchr(buffer, '\r', raw_length) != NULL) {
            fail("noncanonical carriage return");
        }
        if (raw_length > 0U && buffer[raw_length - 1U] == '\n') {
            buffer[raw_length - 1U] = '\0';
        }
        ++line_number;
        char *line = trim(buffer);
        if (*line == '\0' || *line == '#') {
            continue;
        }
        if (end_seen) {
            fail_line("content follows END", line_number);
        }
        char *token[MAX_TOKENS] = {0};
        const size_t count = split_tokens(line, token);
        if (!header_seen) {
            if (
                count != 2U
                || strcmp(
                    token[0],
                    "CATCAS_ALGEBRAIC_RELATION_CHAIN"
                ) != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid chain header", line_number);
            }
            header_seen = 1;
            continue;
        }
        if (strcmp(token[0], "TYPE") == 0) {
            if (
                count != 2U
                || type_seen
                || strcmp(token[1], "BOOLEAN_F3") != 0
            ) {
                fail_line("invalid TYPE", line_number);
            }
            type_seen = 1;
            continue;
        }
        if (strcmp(token[0], "BOUNDARY") == 0) {
            if (count != 3U || boundary_seen) {
                fail_line("invalid BOUNDARY", line_number);
            }
            copy_name(
                process.boundary_first,
                token[1],
                line_number
            );
            copy_name(
                process.boundary_second,
                token[2],
                line_number
            );
            boundary_seen = 1;
            continue;
        }
        if (strcmp(token[0], "RELATION") == 0) {
            if (
                count != 9U
                || strcmp(token[8], "ZEROSET") != 0
            ) {
                fail_line("invalid RELATION", line_number);
            }
            if (process.relation_count == capacity) {
                const size_t next =
                    capacity == 0U ? 8U : capacity * 2U;
                if (next <= capacity) {
                    fail("relation capacity overflow");
                }
                process.relation = checked_reallocarray(
                    process.relation,
                    next,
                    sizeof(*process.relation)
                );
                capacity = next;
            }
            struct relation_definition *relation =
                &process.relation[process.relation_count];
            copy_name(relation->name, token[1], line_number);
            copy_name(relation->first, token[2], line_number);
            copy_name(relation->second, token[3], line_number);
            for (
                size_t coefficient = 0U;
                coefficient < COEFFICIENT_COUNT;
                ++coefficient
            ) {
                relation->coefficient[coefficient] =
                    parse_coefficient(
                        token[4U + coefficient],
                        line_number
                    );
            }
            ++process.relation_count;
            continue;
        }
        if (strcmp(token[0], "END") == 0) {
            if (count != 1U) {
                fail_line("invalid END", line_number);
            }
            end_seen = 1;
            continue;
        }
        fail_line("unknown chain record", line_number);
    }
    if (ferror(stream)) {
        fail("failed to read complete chain");
    }
    free(buffer);
    if (fclose(stream) != 0) {
        perror(path);
        exit(2);
    }
    if (!header_seen || !type_seen || !boundary_seen || !end_seen) {
        fail("chain is incomplete");
    }
    normalize_chain(&process);
    return process;
}

static size_t input_start(size_t relation_index) {
    return COEFFICIENT_COUNT * relation_index;
}

static size_t derived_base(size_t relation_count) {
    return COEFFICIENT_COUNT * relation_count;
}

static size_t derived_start(
    size_t relation_count,
    size_t step
) {
    return (
        derived_base(relation_count)
        + COEFFICIENT_COUNT * (step - 1U)
    );
}

static size_t carrier_cells(size_t relation_count) {
    if (relation_count > SIZE_MAX / 8U) {
        fail("carrier cell count overflow");
    }
    return 8U * relation_count - 4U;
}

static double complex unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        fail("nonfinite phase state");
    }
    return value / magnitude;
}

static double complex root3(int amount) {
    int normalized = amount % 3;
    if (normalized < 0) {
        normalized += 3;
    }
    if (normalized == 0) {
        return 1.0 + 0.0 * I;
    }
    if (normalized == 1) {
        return -0.5 + 0.86602540378443864676 * I;
    }
    return -0.5 - 0.86602540378443864676 * I;
}

static struct carrier make_carrier(size_t cells, int identity) {
    struct carrier carrier = {
        .baseline = checked_calloc(cells, sizeof(*carrier.baseline)),
        .working = checked_calloc(cells, sizeof(*carrier.working)),
        .cells = cells
    };
    for (size_t index = 0U; index < cells; ++index) {
        const double angle =
            0.101
            + 0.043 * (double)(index % 104729U)
            + 0.013 * sin(
                0.17 * (double)(index % 65521U)
                + 0.029 * (double)identity
            );
        carrier.baseline[index] = cexp(I * angle);
        carrier.working[index] = carrier.baseline[index];
    }
    return carrier;
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
    size_t right,
    double complex factor[COEFFICIENT_COUNT]
) {
    factor[0] = difference_product(
        carrier,
        left,
        right + 1U,
        left + 2U,
        right
    );
    factor[1] = difference_product(
        carrier,
        left + 1U,
        right + 1U,
        left + 3U,
        right
    );
    factor[2] = difference_product(
        carrier,
        left,
        right + 3U,
        left + 2U,
        right + 2U
    );
    factor[3] = difference_product(
        carrier,
        left + 1U,
        right + 3U,
        left + 3U,
        right + 2U
    );
}

static void apply_resultant(
    struct carrier *carrier,
    size_t left,
    size_t right,
    size_t output,
    int inverse
) {
    double complex factor[COEFFICIENT_COUNT];
    resultant_factors(carrier, left, right, factor);
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        multiply_relation(
            carrier,
            output + index,
            inverse ? conj(factor[index]) : factor[index]
        );
    }
}

static void apply_wrong_resultant_inverse(
    struct carrier *carrier,
    size_t left,
    size_t right,
    size_t output
) {
    double complex factor[COEFFICIENT_COUNT];
    resultant_factors(carrier, left, right, factor);
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        multiply_relation(
            carrier,
            output + index,
            conj(factor[(index + 1U) % COEFFICIENT_COUNT])
        );
    }
}

static void apply_encoding(
    struct carrier *carrier,
    size_t start,
    const int coefficient[COEFFICIENT_COUNT],
    int inverse
) {
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        double complex factor = root3(coefficient[index]);
        if (inverse) {
            factor = conj(factor);
        }
        multiply_relation(carrier, start + index, factor);
    }
}

static size_t left_start_for_step(
    size_t relation_count,
    size_t step
) {
    return (
        step == 1U
        ? input_start(0U)
        : derived_start(relation_count, step - 1U)
    );
}

static void apply_step(
    struct carrier *carrier,
    size_t relation_count,
    size_t step,
    int inverse
) {
    apply_resultant(
        carrier,
        left_start_for_step(relation_count, step),
        input_start(step),
        derived_start(relation_count, step),
        inverse
    );
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
    const struct carrier *carrier,
    size_t start
) {
    struct boundary_record boundary = {
        .maximum_root_error = 0.0
    };
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        boundary.phase[index] = relation(carrier, start + index);
        double distance = 0.0;
        boundary.coefficient[index] =
            decode_root3(boundary.phase[index], &distance);
        if (distance > boundary.maximum_root_error) {
            boundary.maximum_root_error = distance;
        }
    }
    return boundary;
}

static double carrier_displacement(
    const struct carrier *carrier,
    const struct carrier *borrowed
) {
    double sum = 0.0;
    for (size_t index = 0U; index < carrier->cells; ++index) {
        const double difference = cabs(
            carrier->working[index] - borrowed->working[index]
        );
        sum += difference * difference;
    }
    return sqrt(sum);
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
        const double error =
            fabs(cabs(carrier->working[index]) - 1.0);
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static struct carrier snapshot_carrier(
    const struct carrier *carrier
) {
    struct carrier snapshot = {
        .baseline = checked_calloc(
            carrier->cells,
            sizeof(*snapshot.baseline)
        ),
        .working = checked_calloc(
            carrier->cells,
            sizeof(*snapshot.working)
        ),
        .cells = carrier->cells
    };
    memcpy(
        snapshot.baseline,
        carrier->baseline,
        carrier->cells * sizeof(*carrier->baseline)
    );
    memcpy(
        snapshot.working,
        carrier->working,
        carrier->cells * sizeof(*carrier->working)
    );
    return snapshot;
}

static struct execution execute(
    struct carrier *carrier,
    const struct process *process,
    enum inverse_mode mode
) {
    struct carrier borrowed = snapshot_carrier(carrier);
    const size_t count = process->relation_count;
    for (size_t index = 0U; index < count; ++index) {
        apply_encoding(
            carrier,
            input_start(index),
            process->relation[index].coefficient,
            0
        );
    }
    for (size_t step = 1U; step < count; ++step) {
        apply_step(carrier, count, step, 0);
    }

    const size_t final_step = count - 1U;
    const size_t final_left =
        left_start_for_step(count, final_step);
    const size_t final_right = input_start(final_step);
    const size_t final_output =
        derived_start(count, final_step);
    struct execution execution = {
        .boundary = latch_boundary(carrier, final_output),
        .displacement_l2 =
            carrier_displacement(carrier, &borrowed),
        .wrong_applicable = 0,
        .reordered_applicable = 0,
        .omitted_applicable = 0
    };

    double complex final_factor[COEFFICIENT_COUNT];
    double complex rotated[COEFFICIENT_COUNT];
    resultant_factors(
        carrier,
        final_left,
        final_right,
        final_factor
    );
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        rotated[index] = final_factor[
            (index + 1U) % COEFFICIENT_COUNT
        ];
        if (cabs(final_factor[index] - 1.0) > ROOT_TOLERANCE) {
            execution.omitted_applicable = 1;
        }
    }
    execution.wrong_applicable =
        factors_differ(final_factor, rotated);

    if (count > 2U) {
        double complex before[COEFFICIENT_COUNT];
        double complex after[COEFFICIENT_COUNT];
        resultant_factors(
            carrier,
            left_start_for_step(count, 2U),
            input_start(2U),
            before
        );
        apply_step(carrier, count, 1U, 1);
        resultant_factors(
            carrier,
            left_start_for_step(count, 2U),
            input_start(2U),
            after
        );
        apply_step(carrier, count, 1U, 0);
        execution.reordered_applicable = factors_differ(before, after);
    } else {
        /*
         * One closure has no distinct inverse ordering.  Forward order and
         * reverse order are the same traversal, so this control is honestly
         * inapplicable for the minimum valid chain.
         */
        execution.reordered_applicable = 0;
    }

    if (mode == INVERSE_CORRECT) {
        for (size_t step = final_step; step > 0U; --step) {
            apply_step(carrier, count, step, 1);
        }
    } else if (mode == INVERSE_WRONG) {
        apply_wrong_resultant_inverse(
            carrier,
            final_left,
            final_right,
            final_output
        );
        for (size_t step = final_step - 1U; step > 0U; --step) {
            apply_step(carrier, count, step, 1);
        }
    } else if (mode == INVERSE_REORDERED) {
        for (size_t step = 1U; step <= final_step; ++step) {
            apply_step(carrier, count, step, 1);
        }
    } else if (mode == INVERSE_OMITTED) {
        for (size_t step = final_step - 1U; step > 0U; --step) {
            apply_step(carrier, count, step, 1);
        }
    } else {
        fail("unknown inverse mode");
    }
    for (size_t index = count; index > 0U; --index) {
        apply_encoding(
            carrier,
            input_start(index - 1U),
            process->relation[index - 1U].coefficient,
            1
        );
    }

    execution.restoration_max_abs =
        restoration_error(carrier, &borrowed);
    execution.carrier_integrity_error = integrity_error(carrier);
    free_carrier(&borrowed);
    return execution;
}

static void print_execution(
    const char *mode,
    const struct process *process,
    const struct execution *execution,
    size_t cells
) {
    printf(
        "{\"mode\":\"%s\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"port_type\":\"BOOLEAN_F3\","
        "\"relation_count\":%zu,"
        "\"internal_port_count\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"tuple_slots\":0,"
        "\"witness_slots\":0,"
        "\"retained_inverse_factors\":0,"
        "\"boundary_coefficients\":[%d,%d,%d,%d],"
        "\"maximum_root_error\":%.12g,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_error\":%.12g}\n",
        mode,
        (unsigned long long)process->source_fnv1a64,
        process->relation_count,
        process->relation_count - 1U,
        cells,
        execution->boundary.coefficient[0],
        execution->boundary.coefficient[1],
        execution->boundary.coefficient[2],
        execution->boundary.coefficient[3],
        execution->boundary.maximum_root_error,
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->carrier_integrity_error
    );
}

static void free_process(struct process *process) {
    free(process->relation);
    process->relation = NULL;
    process->relation_count = 0U;
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(
            stderr,
            "usage: %s PROCESS.arelc [REUSE_PROCESS.arelc]\n",
            argv[0]
        );
        return 2;
    }
    struct process process = read_process(argv[1]);
    struct process reuse_process =
        argc == 3 ? read_process(argv[2]) : read_process(argv[1]);
    if (reuse_process.relation_count != process.relation_count) {
        fail("reuse chain must have the same relation count");
    }
    const size_t cells = carrier_cells(process.relation_count);

    struct carrier carrier = make_carrier(cells, 1543);
    const struct execution nominal = execute(
        &carrier,
        &process,
        INVERSE_CORRECT
    );
    const struct execution reuse = execute(
        &carrier,
        &reuse_process,
        INVERSE_CORRECT
    );
    free_carrier(&carrier);

    carrier = make_carrier(cells, 1543);
    const struct execution wrong = execute(
        &carrier,
        &process,
        INVERSE_WRONG
    );
    free_carrier(&carrier);
    carrier = make_carrier(cells, 1543);
    const struct execution reordered = execute(
        &carrier,
        &process,
        INVERSE_REORDERED
    );
    free_carrier(&carrier);
    carrier = make_carrier(cells, 1543);
    const struct execution omitted = execute(
        &carrier,
        &process,
        INVERSE_OMITTED
    );
    free_carrier(&carrier);

    print_execution(
        "algebraic-relation-chain",
        &process,
        &nominal,
        cells
    );
    print_execution(
        argc == 3
            ? "actual-restored-cross-process-reuse"
            : "actual-restored-reuse",
        &reuse_process,
        &reuse,
        cells
    );
    print_execution("wrong-inverse", &process, &wrong, cells);
    print_execution(
        "reordered-inverse",
        &process,
        &reordered,
        cells
    );
    print_execution("omitted-inverse", &process, &omitted, cells);
    printf(
        "{\"mode\":\"inverse-control-applicability\","
        "\"wrong\":%s,\"reordered\":%s,\"omitted\":%s}\n",
        wrong.wrong_applicable ? "true" : "false",
        reordered.reordered_applicable ? "true" : "false",
        omitted.omitted_applicable ? "true" : "false"
    );

    const int valid = (
        nominal.boundary.maximum_root_error <= ROOT_TOLERANCE
        && reuse.boundary.maximum_root_error <= ROOT_TOLERANCE
        && nominal.restoration_max_abs <= RESTORATION_TOLERANCE
        && reuse.restoration_max_abs <= RESTORATION_TOLERANCE
        && nominal.carrier_integrity_error <= RESTORATION_TOLERANCE
        && reuse.carrier_integrity_error <= RESTORATION_TOLERANCE
        && (
            !wrong.wrong_applicable
            || wrong.restoration_max_abs >= CONTROL_MINIMUM
        )
        && (
            !reordered.reordered_applicable
            || reordered.restoration_max_abs >= CONTROL_MINIMUM
        )
        && (
            !omitted.omitted_applicable
            || omitted.restoration_max_abs >= CONTROL_MINIMUM
        )
    );
    free_process(&process);
    free_process(&reuse_process);
    return valid ? 0 : 1;
}
