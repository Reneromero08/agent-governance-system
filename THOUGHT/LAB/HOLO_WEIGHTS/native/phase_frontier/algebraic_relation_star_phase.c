/*
 * Mutable CAT_CAS frontier: branching algebraic relation-star closure.
 *
 * Every branch is a bi-total Boolean_F3 multiaffine relation between one
 * external port and a shared hub.  Closing the hub produces a factorized
 * boundary relation: one phase-native resultant for every pair of external
 * ports.  This is exact because nonempty subsets of a two-point hub have a
 * common member exactly when every pair intersects.
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
    INVERSE_GEOMETRY_SCRAMBLED = 2,
    INVERSE_OMITTED = 3
};

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

struct carrier {
    double complex *baseline;
    double complex *working;
    size_t cells;
};

struct boundary_record {
    uint64_t factor_fnv1a64;
    size_t factor_count;
    int first_factor[COEFFICIENT_COUNT];
    int last_factor[COEFFICIENT_COUNT];
    double maximum_root_error;
};

struct execution {
    struct boundary_record boundary;
    double displacement_l2;
    double restoration_max_abs;
    double carrier_integrity_error;
    int wrong_applicable;
    int geometry_scramble_applicable;
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

static size_t input_start(size_t branch_index) {
    return COEFFICIENT_COUNT * branch_index;
}

static size_t factor_base(size_t branch_count) {
    return COEFFICIENT_COUNT * branch_count;
}

static size_t factor_start(
    size_t branch_count,
    size_t factor_index
) {
    return factor_base(branch_count) + COEFFICIENT_COUNT * factor_index;
}

static size_t carrier_cells(size_t branch_count) {
    if (branch_count > SIZE_MAX / COEFFICIENT_COUNT) {
        fail("input carrier size overflow");
    }
    const size_t inputs = COEFFICIENT_COUNT * branch_count;
    const size_t factors = pair_count(branch_count);
    if (factors > (SIZE_MAX - inputs) / COEFFICIENT_COUNT) {
        fail("factor carrier size overflow");
    }
    return inputs + COEFFICIENT_COUNT * factors;
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
            0.137
            + 0.037 * (double)(index % 104729U)
            + 0.011 * sin(
                0.19 * (double)(index % 65521U)
                + 0.031 * (double)identity
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

static void star_resultant_factors(
    const struct carrier *carrier,
    size_t left,
    size_t right,
    double complex factor[COEFFICIENT_COUNT]
) {
    factor[0] = difference_product(
        carrier,
        left,
        right + 2U,
        left + 2U,
        right
    );
    factor[1] = difference_product(
        carrier,
        left + 1U,
        right + 2U,
        left + 3U,
        right
    );
    factor[2] = difference_product(
        carrier,
        left,
        right + 3U,
        left + 2U,
        right + 1U
    );
    factor[3] = difference_product(
        carrier,
        left + 1U,
        right + 3U,
        left + 3U,
        right + 1U
    );
}

static void apply_factor_to_output(
    struct carrier *carrier,
    const double complex factor[COEFFICIENT_COUNT],
    size_t output,
    int inverse
) {
    for (size_t coefficient = 0U; coefficient < COEFFICIENT_COUNT; ++coefficient) {
        multiply_relation(
            carrier,
            output + coefficient,
            inverse
                ? conj(factor[coefficient])
                : factor[coefficient]
        );
    }
}

static void apply_pair(
    struct carrier *carrier,
    size_t branch_count,
    size_t left_branch,
    size_t right_branch,
    size_t factor_index,
    int inverse
) {
    double complex factor[COEFFICIENT_COUNT];
    star_resultant_factors(
        carrier,
        input_start(left_branch),
        input_start(right_branch),
        factor
    );
    apply_factor_to_output(
        carrier,
        factor,
        factor_start(branch_count, factor_index),
        inverse
    );
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
    const double complex one = 1.0 + 0.0 * I;
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        if (cabs(factor[index] - one) > ROOT_TOLERANCE) {
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
    const struct process *process
) {
    struct boundary_record boundary = {
        .factor_fnv1a64 = UINT64_C(14695981039346656037),
        .factor_count = pair_count(process->branch_count),
        .maximum_root_error = 0.0
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
                double distance = 0.0;
                const int decoded = decode_root3(
                    relation(
                        carrier,
                        factor_start(
                            process->branch_count,
                            factor_index
                        ) + coefficient
                    ),
                    &distance
                );
                coefficient_bytes[coefficient] =
                    (unsigned char)decoded;
                if (factor_index == 0U) {
                    boundary.first_factor[coefficient] = decoded;
                }
                boundary.last_factor[coefficient] = decoded;
                if (distance > boundary.maximum_root_error) {
                    boundary.maximum_root_error = distance;
                }
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

static void inspect_control_applicability(
    const struct carrier *carrier,
    size_t branch_count,
    struct execution *execution
) {
    double complex first[COEFFICIENT_COUNT] = {0};
    double complex previous[COEFFICIENT_COUNT] = {0};
    size_t factor_index = 0U;
    for (size_t left = 0U; left + 1U < branch_count; ++left) {
        for (size_t right = left + 1U; right < branch_count; ++right) {
            double complex factor[COEFFICIENT_COUNT];
            double complex rotated[COEFFICIENT_COUNT];
            star_resultant_factors(
                carrier,
                input_start(left),
                input_start(right),
                factor
            );
            for (
                size_t coefficient = 0U;
                coefficient < COEFFICIENT_COUNT;
                ++coefficient
            ) {
                rotated[coefficient] = factor[
                    (coefficient + 1U) % COEFFICIENT_COUNT
                ];
            }
            execution->wrong_applicable |=
                factors_differ(factor, rotated);
            if (factor_index == 0U) {
                memcpy(first, factor, sizeof(first));
            } else {
                execution->geometry_scramble_applicable |=
                    factors_differ(previous, factor);
            }
            memcpy(previous, factor, sizeof(previous));
            ++factor_index;
        }
    }
    execution->geometry_scramble_applicable |=
        factors_differ(previous, first);
    execution->omitted_applicable = factor_nontrivial(previous);
}

static void forward_pairs(
    struct carrier *carrier,
    size_t branch_count
) {
    size_t factor_index = 0U;
    for (size_t left = 0U; left + 1U < branch_count; ++left) {
        for (size_t right = left + 1U; right < branch_count; ++right) {
            apply_pair(
                carrier,
                branch_count,
                left,
                right,
                factor_index,
                0
            );
            ++factor_index;
        }
    }
}

static void inverse_pairs_correct(
    struct carrier *carrier,
    size_t branch_count,
    int omit_last
) {
    size_t factor_index = pair_count(branch_count);
    for (size_t left = branch_count - 1U; left-- > 0U;) {
        for (
            size_t right = branch_count;
            right-- > left + 1U;
        ) {
            --factor_index;
            if (omit_last && factor_index + 1U == pair_count(branch_count)) {
                continue;
            }
            apply_pair(
                carrier,
                branch_count,
                left,
                right,
                factor_index,
                1
            );
        }
    }
    if (factor_index != 0U) {
        fail("inverse pair traversal drift");
    }
}

static void inverse_pairs_wrong(
    struct carrier *carrier,
    size_t branch_count
) {
    size_t factor_index = 0U;
    for (size_t left = 0U; left + 1U < branch_count; ++left) {
        for (size_t right = left + 1U; right < branch_count; ++right) {
            double complex factor[COEFFICIENT_COUNT];
            double complex rotated[COEFFICIENT_COUNT];
            star_resultant_factors(
                carrier,
                input_start(left),
                input_start(right),
                factor
            );
            for (
                size_t coefficient = 0U;
                coefficient < COEFFICIENT_COUNT;
                ++coefficient
            ) {
                rotated[coefficient] = factor[
                    (coefficient + 1U) % COEFFICIENT_COUNT
                ];
            }
            apply_factor_to_output(
                carrier,
                rotated,
                factor_start(branch_count, factor_index),
                1
            );
            ++factor_index;
        }
    }
}

static void inverse_pairs_geometry_scrambled(
    struct carrier *carrier,
    size_t branch_count
) {
    const size_t factors = pair_count(branch_count);
    size_t factor_index = 0U;
    for (size_t left = 0U; left + 1U < branch_count; ++left) {
        for (size_t right = left + 1U; right < branch_count; ++right) {
            double complex factor[COEFFICIENT_COUNT];
            star_resultant_factors(
                carrier,
                input_start(left),
                input_start(right),
                factor
            );
            const size_t wrong_output = (factor_index + 1U) % factors;
            apply_factor_to_output(
                carrier,
                factor,
                factor_start(branch_count, wrong_output),
                1
            );
            ++factor_index;
        }
    }
}

static struct execution execute(
    struct carrier *carrier,
    const struct process *process,
    enum inverse_mode mode
) {
    struct carrier borrowed = snapshot_carrier(carrier);
    for (size_t index = 0U; index < process->branch_count; ++index) {
        apply_encoding(
            carrier,
            input_start(index),
            process->branch[index].coefficient,
            0
        );
    }
    forward_pairs(carrier, process->branch_count);
    struct execution execution = {
        .boundary = latch_boundary(carrier, process),
        .displacement_l2 = carrier_displacement(carrier, &borrowed),
        .wrong_applicable = 0,
        .geometry_scramble_applicable = 0,
        .omitted_applicable = 0
    };
    inspect_control_applicability(
        carrier,
        process->branch_count,
        &execution
    );

    if (mode == INVERSE_CORRECT) {
        inverse_pairs_correct(carrier, process->branch_count, 0);
    } else if (mode == INVERSE_WRONG) {
        inverse_pairs_wrong(carrier, process->branch_count);
    } else if (mode == INVERSE_GEOMETRY_SCRAMBLED) {
        inverse_pairs_geometry_scrambled(
            carrier,
            process->branch_count
        );
    } else if (mode == INVERSE_OMITTED) {
        inverse_pairs_correct(carrier, process->branch_count, 1);
    } else {
        fail("unknown inverse mode");
    }
    for (size_t index = process->branch_count; index > 0U; --index) {
        apply_encoding(
            carrier,
            input_start(index - 1U),
            process->branch[index - 1U].coefficient,
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
        "\"branch_count\":%zu,"
        "\"closed_hub_count\":1,"
        "\"boundary_factor_count\":%zu,"
        "\"boundary_coefficient_count\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"tuple_slots\":0,"
        "\"witness_slots\":0,"
        "\"truth_table_slots\":0,"
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
        process->branch_count,
        execution->boundary.factor_count,
        COEFFICIENT_COUNT * execution->boundary.factor_count,
        cells,
        (unsigned long long)execution->boundary.factor_fnv1a64,
        execution->boundary.first_factor[0],
        execution->boundary.first_factor[1],
        execution->boundary.first_factor[2],
        execution->boundary.first_factor[3],
        execution->boundary.last_factor[0],
        execution->boundary.last_factor[1],
        execution->boundary.last_factor[2],
        execution->boundary.last_factor[3],
        execution->boundary.maximum_root_error,
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->carrier_integrity_error
    );
}

static void free_process(struct process *process) {
    free(process->branch);
    process->branch = NULL;
    process->branch_count = 0U;
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(
            stderr,
            "usage: %s PROCESS.arels [REUSE_PROCESS.arels]\n",
            argv[0]
        );
        return 2;
    }
    struct process process = read_process(argv[1]);
    struct process reuse_process =
        argc == 3 ? read_process(argv[2]) : read_process(argv[1]);
    if (reuse_process.branch_count != process.branch_count) {
        fail("reuse star must have the same branch count");
    }
    const size_t cells = carrier_cells(process.branch_count);

    struct carrier carrier = make_carrier(cells, 2027);
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

    carrier = make_carrier(cells, 2027);
    const struct execution wrong = execute(
        &carrier,
        &process,
        INVERSE_WRONG
    );
    free_carrier(&carrier);
    carrier = make_carrier(cells, 2027);
    const struct execution scrambled = execute(
        &carrier,
        &process,
        INVERSE_GEOMETRY_SCRAMBLED
    );
    free_carrier(&carrier);
    carrier = make_carrier(cells, 2027);
    const struct execution omitted = execute(
        &carrier,
        &process,
        INVERSE_OMITTED
    );
    free_carrier(&carrier);

    print_execution(
        "algebraic-relation-star",
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
        "geometry-scrambled-inverse",
        &process,
        &scrambled,
        cells
    );
    print_execution("omitted-inverse", &process, &omitted, cells);
    printf(
        "{\"mode\":\"inverse-control-applicability\","
        "\"wrong\":%s,\"geometry_scrambled\":%s,\"omitted\":%s,"
        "\"pair_order_commutative\":true}\n",
        wrong.wrong_applicable ? "true" : "false",
        scrambled.geometry_scramble_applicable ? "true" : "false",
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
            !scrambled.geometry_scramble_applicable
            || scrambled.restoration_max_abs >= CONTROL_MINIMUM
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
