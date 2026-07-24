/*
 * Independent bounded scalar adjudicator for branching relation stars.
 *
 * This executable is not linked into the phase process.  It independently
 * derives every pairwise resultant factor and, for small stars, enumerates
 * boundary assignments to compare the factor graph with exact existential
 * hub projection.
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
#define MAX_EXHAUSTIVE_BRANCHES 20U

struct branch_definition {
    char name[MAX_NAME + 1U];
    char first[MAX_NAME + 1U];
    char second[MAX_NAME + 1U];
    char external[MAX_NAME + 1U];
    int coefficient[COEFFICIENT_COUNT];
};

struct process {
    struct branch_definition *branch;
    size_t branch_count;
    char hub[MAX_NAME + 1U];
    uint64_t source_fnv1a64;
};

struct name_entry {
    char name[MAX_NAME + 1U];
};

struct boundary_record {
    uint64_t factor_fnv1a64;
    size_t factor_count;
    int first_factor[COEFFICIENT_COUNT];
    int last_factor[COEFFICIENT_COUNT];
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

static int total_toward_hub(
    const struct branch_definition *branch
) {
    return (
        affine_has_boolean_zero(
            branch->coefficient[2],
            branch->coefficient[0]
        )
        && affine_has_boolean_zero(
            mod3(
                branch->coefficient[2]
                + branch->coefficient[3]
            ),
            mod3(
                branch->coefficient[0]
                + branch->coefficient[1]
            )
        )
    );
}

static int total_toward_external(
    const struct branch_definition *branch
) {
    return (
        affine_has_boolean_zero(
            branch->coefficient[1],
            branch->coefficient[0]
        )
        && affine_has_boolean_zero(
            mod3(
                branch->coefficient[1]
                + branch->coefficient[3]
            ),
            mod3(
                branch->coefficient[0]
                + branch->coefficient[2]
            )
        )
    );
}

static int compare_names(
    const void *left_value,
    const void *right_value
) {
    const struct name_entry *left = left_value;
    const struct name_entry *right = right_value;
    return strcmp(left->name, right->name);
}

static int compare_branches(
    const void *left_value,
    const void *right_value
) {
    const struct branch_definition *left = left_value;
    const struct branch_definition *right = right_value;
    return strcmp(left->external, right->external);
}

static void orient_branch(
    struct branch_definition *branch,
    const char *hub
) {
    const int first_is_hub = strcmp(branch->first, hub) == 0;
    const int second_is_hub = strcmp(branch->second, hub) == 0;
    if (first_is_hub == second_is_hub) {
        fail("every relation must touch the hub exactly once");
    }
    if (second_is_hub) {
        memcpy(
            branch->external,
            branch->first,
            strlen(branch->first) + 1U
        );
        return;
    }
    memcpy(
        branch->external,
        branch->second,
        strlen(branch->second) + 1U
    );
    const int old_10 = branch->coefficient[1];
    branch->coefficient[1] = branch->coefficient[2];
    branch->coefficient[2] = old_10;
}

static void normalize_star(struct process *process) {
    if (process->branch_count < 3U) {
        fail("branching star requires at least three relations");
    }
    struct name_entry *names = checked_calloc(
        process->branch_count,
        sizeof(*names)
    );
    for (size_t index = 0U; index < process->branch_count; ++index) {
        orient_branch(&process->branch[index], process->hub);
        memcpy(
            names[index].name,
            process->branch[index].name,
            strlen(process->branch[index].name) + 1U
        );
    }
    qsort(
        names,
        process->branch_count,
        sizeof(*names),
        compare_names
    );
    for (size_t index = 1U; index < process->branch_count; ++index) {
        if (strcmp(names[index - 1U].name, names[index].name) == 0) {
            fail("duplicate relation name");
        }
    }
    free(names);
    qsort(
        process->branch,
        process->branch_count,
        sizeof(*process->branch),
        compare_branches
    );
    for (size_t index = 0U; index < process->branch_count; ++index) {
        if (
            index > 0U
            && strcmp(
                process->branch[index - 1U].external,
                process->branch[index].external
            ) == 0
        ) {
            fail("duplicate external port");
        }
        if (
            !total_toward_hub(&process->branch[index])
            || !total_toward_external(&process->branch[index])
        ) {
            fail("relation is not bi-total on Boolean_F3");
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
        .branch = NULL,
        .branch_count = 0U,
        .source_fnv1a64 = UINT64_C(14695981039346656037)
    };
    size_t branch_capacity = 0U;
    int header_seen = 0;
    int type_seen = 0;
    int hub_seen = 0;
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
                    "CATCAS_ALGEBRAIC_RELATION_STAR"
                ) != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid star header", line_number);
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
        if (strcmp(token[0], "HUB") == 0) {
            if (count != 2U || hub_seen) {
                fail_line("invalid HUB", line_number);
            }
            copy_name(process.hub, token[1], line_number);
            hub_seen = 1;
            continue;
        }
        if (strcmp(token[0], "RELATION") == 0) {
            if (
                count != 9U
                || strcmp(token[8], "ZEROSET") != 0
            ) {
                fail_line("invalid RELATION", line_number);
            }
            if (process.branch_count == branch_capacity) {
                const size_t next =
                    branch_capacity == 0U
                    ? 8U
                    : branch_capacity * 2U;
                if (next <= branch_capacity) {
                    fail("relation capacity overflow");
                }
                process.branch = checked_reallocarray(
                    process.branch,
                    next,
                    sizeof(*process.branch)
                );
                branch_capacity = next;
            }
            struct branch_definition *branch =
                &process.branch[process.branch_count];
            copy_name(branch->name, token[1], line_number);
            copy_name(branch->first, token[2], line_number);
            copy_name(branch->second, token[3], line_number);
            for (
                size_t coefficient = 0U;
                coefficient < COEFFICIENT_COUNT;
                ++coefficient
            ) {
                branch->coefficient[coefficient] =
                    parse_coefficient(
                        token[4U + coefficient],
                        line_number
                    );
            }
            ++process.branch_count;
            continue;
        }
        if (strcmp(token[0], "END") == 0) {
            if (count != 1U) {
                fail_line("invalid END", line_number);
            }
            end_seen = 1;
            continue;
        }
        fail_line("unknown star record", line_number);
    }
    if (ferror(stream)) {
        fail("failed to read complete star");
    }
    free(buffer);
    if (fclose(stream) != 0) {
        perror(path);
        exit(2);
    }
    if (!header_seen || !type_seen || !hub_seen || !end_seen) {
        fail("star is incomplete");
    }
    normalize_star(&process);
    return process;
}

static size_t pair_count(size_t branch_count) {
    size_t left = branch_count;
    size_t right = branch_count - 1U;
    if ((left & 1U) == 0U) {
        left /= 2U;
    } else {
        right /= 2U;
    }
    if (left != 0U && right > SIZE_MAX / left) {
        fail("pair count overflow");
    }
    return left * right;
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

static unsigned int fiber_mask(
    const int coefficient[COEFFICIENT_COUNT],
    int external
) {
    unsigned int mask = 0U;
    for (int hub = 0; hub <= 1; ++hub) {
        if (evaluate(coefficient, external, hub) == 0) {
            mask |= 1U << (unsigned int)hub;
        }
    }
    return mask;
}

static void star_pair_resultant(
    const int left[COEFFICIENT_COUNT],
    const int right[COEFFICIENT_COUNT],
    int output[COEFFICIENT_COUNT]
) {
    output[0] = mod3(left[0] * right[2] - left[2] * right[0]);
    output[1] = mod3(left[1] * right[2] - left[3] * right[0]);
    output[2] = mod3(left[0] * right[3] - left[2] * right[1]);
    output[3] = mod3(left[1] * right[3] - left[3] * right[1]);
}

static struct boundary_record derive_boundary(
    const struct process *process
) {
    struct boundary_record boundary = {
        .factor_fnv1a64 = UINT64_C(14695981039346656037),
        .factor_count = pair_count(process->branch_count)
    };
    size_t factor_index = 0U;
    for (
        size_t left = 0U;
        left + 1U < process->branch_count;
        ++left
    ) {
        for (
            size_t right = left + 1U;
            right < process->branch_count;
            ++right
        ) {
            int factor[COEFFICIENT_COUNT];
            star_pair_resultant(
                process->branch[left].coefficient,
                process->branch[right].coefficient,
                factor
            );
            boundary.factor_fnv1a64 = fnv1a64_update(
                boundary.factor_fnv1a64,
                (const unsigned char *)process->branch[left].external,
                strlen(process->branch[left].external) + 1U
            );
            boundary.factor_fnv1a64 = fnv1a64_update(
                boundary.factor_fnv1a64,
                (const unsigned char *)process->branch[right].external,
                strlen(process->branch[right].external) + 1U
            );
            unsigned char coefficient_bytes[COEFFICIENT_COUNT];
            for (
                size_t coefficient = 0U;
                coefficient < COEFFICIENT_COUNT;
                ++coefficient
            ) {
                coefficient_bytes[coefficient] =
                    (unsigned char)factor[coefficient];
                if (factor_index == 0U) {
                    boundary.first_factor[coefficient] =
                        factor[coefficient];
                }
                boundary.last_factor[coefficient] =
                    factor[coefficient];
            }
            boundary.factor_fnv1a64 = fnv1a64_update(
                boundary.factor_fnv1a64,
                coefficient_bytes,
                sizeof(coefficient_bytes)
            );
            ++factor_index;
        }
    }
    if (factor_index != boundary.factor_count) {
        fail("boundary factor count drift");
    }
    return boundary;
}

static int factor_graph_accepts(
    const struct process *process,
    uint64_t assignment
) {
    for (
        size_t left = 0U;
        left + 1U < process->branch_count;
        ++left
    ) {
        for (
            size_t right = left + 1U;
            right < process->branch_count;
            ++right
        ) {
            int factor[COEFFICIENT_COUNT];
            star_pair_resultant(
                process->branch[left].coefficient,
                process->branch[right].coefficient,
                factor
            );
            const int left_value =
                (int)((assignment >> left) & UINT64_C(1));
            const int right_value =
                (int)((assignment >> right) & UINT64_C(1));
            if (evaluate(factor, left_value, right_value) != 0) {
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s PROCESS.arels\n", argv[0]);
        return 2;
    }
    struct process process = read_process(argv[1]);
    const struct boundary_record boundary =
        derive_boundary(&process);
    const int exhaustive =
        process.branch_count <= MAX_EXHAUSTIVE_BRANCHES;
    uint64_t assignment_rows = 0U;
    uint64_t projected_rows = 0U;
    uint64_t two_witness_rows = 0U;
    int exact = 1;
    if (exhaustive) {
        const uint64_t rows =
            UINT64_C(1) << (unsigned int)process.branch_count;
        for (uint64_t assignment = 0U; assignment < rows; ++assignment) {
            unsigned int common = 3U;
            for (
                size_t branch = 0U;
                branch < process.branch_count;
                ++branch
            ) {
                const int external =
                    (int)((assignment >> branch) & UINT64_C(1));
                common &= fiber_mask(
                    process.branch[branch].coefficient,
                    external
                );
            }
            const int projected = common != 0U;
            const int factorized =
                factor_graph_accepts(&process, assignment);
            exact &= projected == factorized;
            projected_rows += (uint64_t)projected;
            two_witness_rows += (uint64_t)(common == 3U);
            ++assignment_rows;
        }
    }
    printf(
        "{\"mode\":\"bounded-star-existential-reference\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"branch_count\":%zu,"
        "\"boundary_factor_count\":%zu,"
        "\"boundary_factor_fnv1a64\":\"%016llx\","
        "\"first_factor\":[%d,%d,%d,%d],"
        "\"last_factor\":[%d,%d,%d,%d],"
        "\"exhaustive_performed\":%s,"
        "\"assignment_rows\":%llu,"
        "\"projected_boundary_rows\":%llu,"
        "\"two_witness_rows\":%llu,"
        "\"factorized_exact\":%s}\n",
        (unsigned long long)process.source_fnv1a64,
        process.branch_count,
        boundary.factor_count,
        (unsigned long long)boundary.factor_fnv1a64,
        boundary.first_factor[0],
        boundary.first_factor[1],
        boundary.first_factor[2],
        boundary.first_factor[3],
        boundary.last_factor[0],
        boundary.last_factor[1],
        boundary.last_factor[2],
        boundary.last_factor[3],
        exhaustive ? "true" : "false",
        (unsigned long long)assignment_rows,
        (unsigned long long)projected_rows,
        (unsigned long long)two_witness_rows,
        exhaustive ? (exact ? "true" : "false") : "null"
    );
    free(process.branch);
    return exhaustive && !exact ? 1 : 0;
}
