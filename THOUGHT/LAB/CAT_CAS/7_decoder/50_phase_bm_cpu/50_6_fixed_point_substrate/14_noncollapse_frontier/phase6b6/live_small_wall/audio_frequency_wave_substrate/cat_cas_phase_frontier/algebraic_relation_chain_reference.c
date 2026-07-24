/*
 * Independent bounded scalar adjudicator for algebraic relation chains.
 *
 * This executable is not linked into the phase engine.  It parses and
 * normalizes the same public chain language, evaluates the Boolean relation
 * masks independently, and checks every coefficient resultant against
 * ordinary existential composition.
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define COEFFICIENT_COUNT 4U
#define MAX_NAME 31U
#define MAX_TOKENS 12U

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

static int evaluate(
    const int coefficient[COEFFICIENT_COUNT],
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

static unsigned int relation_mask(
    const int coefficient[COEFFICIENT_COUNT]
) {
    unsigned int mask = 0U;
    for (int first = 0; first <= 1; ++first) {
        for (int second = 0; second <= 1; ++second) {
            if (evaluate(coefficient, first, second) == 0) {
                mask |= 1U << (unsigned int)(2 * first + second);
            }
        }
    }
    return mask;
}

static unsigned int compose_masks(
    unsigned int left,
    unsigned int right
) {
    unsigned int output = 0U;
    for (int first = 0; first <= 1; ++first) {
        for (int second = 0; second <= 1; ++second) {
            int exists = 0;
            for (int internal = 0; internal <= 1; ++internal) {
                const unsigned int left_bit =
                    1U << (unsigned int)(2 * first + internal);
                const unsigned int right_bit =
                    1U << (unsigned int)(2 * internal + second);
                if ((left & left_bit) != 0U && (right & right_bit) != 0U) {
                    exists = 1;
                }
            }
            if (exists) {
                output |= 1U << (unsigned int)(2 * first + second);
            }
        }
    }
    return output;
}

static void resultant(
    const int left[COEFFICIENT_COUNT],
    const int right[COEFFICIENT_COUNT],
    int output[COEFFICIENT_COUNT]
) {
    output[0] = mod3(left[0] * right[1] - left[2] * right[0]);
    output[1] = mod3(left[1] * right[1] - left[3] * right[0]);
    output[2] = mod3(left[0] * right[3] - left[2] * right[2]);
    output[3] = mod3(left[1] * right[3] - left[3] * right[2]);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s CHAIN.arelc\n", argv[0]);
        return 2;
    }
    struct process process = read_process(argv[1]);
    int accumulator[COEFFICIENT_COUNT];
    memcpy(
        accumulator,
        process.relation[0].coefficient,
        sizeof(accumulator)
    );
    unsigned int extensional_mask = relation_mask(accumulator);
    size_t exact_steps = 0U;
    for (size_t step = 1U; step < process.relation_count; ++step) {
        int next[COEFFICIENT_COUNT];
        resultant(
            accumulator,
            process.relation[step].coefficient,
            next
        );
        extensional_mask = compose_masks(
            extensional_mask,
            relation_mask(process.relation[step].coefficient)
        );
        if (relation_mask(next) != extensional_mask) {
            free(process.relation);
            fail("resultant diverges from existential composition");
        }
        ++exact_steps;
        memcpy(accumulator, next, sizeof(accumulator));
    }
    printf(
        "{\"mode\":\"bounded-bi-total-chain-reference\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"relation_count\":%zu,"
        "\"internal_port_count\":%zu,"
        "\"boundary_coefficients\":[%d,%d,%d,%d],"
        "\"boundary_pair_mask\":%u,"
        "\"exact_composition_steps\":%zu,"
        "\"all_steps_exact\":true}\n",
        (unsigned long long)process.source_fnv1a64,
        process.relation_count,
        process.relation_count - 1U,
        accumulator[0],
        accumulator[1],
        accumulator[2],
        accumulator[3],
        extensional_mask,
        exact_steps
    );
    free(process.relation);
    return 0;
}
