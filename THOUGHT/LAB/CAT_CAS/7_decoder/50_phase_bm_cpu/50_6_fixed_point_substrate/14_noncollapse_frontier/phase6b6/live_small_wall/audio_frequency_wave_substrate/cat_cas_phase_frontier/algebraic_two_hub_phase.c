#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: two-hub relation-valued phase memory.
 *
 * Five public bi-total Boolean_F3 relations form the tree
 *
 *     A--U--V--C
 *     B--'  '--D
 *
 * The first native resultant closes U between a left relation and the
 * U--V bridge.  Its four complex phases remain resident as a binary
 * relation from the left external port to V.  A second native resultant
 * consumes that phase-resident relation directly with a right relation.
 * No coefficient is decoded between the two closures.
 */

#include <complex.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COEFFICIENT_COUNT 4U
#define RELATION_COUNT 5U
#define MESSAGE_COUNT 2U
#define BOUNDARY_COUNT 6U
#define INPUT_CELLS (RELATION_COUNT * COEFFICIENT_COUNT)
#define MESSAGE_CELLS (MESSAGE_COUNT * COEFFICIENT_COUNT)
#define BOUNDARY_CELLS (BOUNDARY_COUNT * COEFFICIENT_COUNT)
#define CARRIER_CELLS (INPUT_CELLS + MESSAGE_CELLS + BOUNDARY_CELLS)
#define LINE_CAPACITY 512U
#define TOKEN_CAPACITY 12U

static const double ROOT_TOLERANCE = 2.0e-10;
static const double RESTORATION_TOLERANCE = 2.0e-12;
static const double CONTROL_MINIMUM = 1.0e-3;

enum role {
    ROLE_LEFT_A = 0,
    ROLE_LEFT_B = 1,
    ROLE_BRIDGE = 2,
    ROLE_RIGHT_C = 3,
    ROLE_RIGHT_D = 4
};

enum inverse_mode {
    INVERSE_CORRECT = 0,
    INVERSE_WRONG_BOUNDARY = 1,
    INVERSE_SCRAMBLED_BOUNDARY = 2,
    INVERSE_OMITTED_MESSAGE = 3,
    INVERSE_BYPASS_MESSAGE = 4
};

struct relation_spec {
    int coefficient[COEFFICIENT_COUNT];
    int present;
};

struct process {
    struct relation_spec relation[RELATION_COUNT];
    uint64_t source_fnv1a64;
};

struct carrier {
    double complex *baseline;
    double complex *working;
    size_t cells;
};

struct boundary_record {
    uint64_t factor_fnv1a64;
    int coefficient[BOUNDARY_COUNT][COEFFICIENT_COUNT];
    double maximum_root_error;
};

struct execution {
    struct boundary_record boundary;
    double displacement_l2;
    double restoration_max_abs;
    double carrier_integrity_error;
    int wrong_applicable;
    int geometry_applicable;
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
    if (normalized < 0) {
        normalized += 3;
    }
    return normalized;
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
        int has_second = 0;
        for (int second = 0; second < 2; ++second) {
            has_second |= evaluate_relation(
                coefficient,
                first,
                second
            ) == 0;
        }
        if (!has_second) {
            return 0;
        }
    }
    for (int second = 0; second < 2; ++second) {
        int has_first = 0;
        for (int first = 0; first < 2; ++first) {
            has_first |= evaluate_relation(
                coefficient,
                first,
                second
            ) == 0;
        }
        if (!has_first) {
            return 0;
        }
    }
    return 1;
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

static enum role parse_role(const char *text, size_t line_number) {
    if (strcmp(text, "LEFT_A") == 0) {
        return ROLE_LEFT_A;
    }
    if (strcmp(text, "LEFT_B") == 0) {
        return ROLE_LEFT_B;
    }
    if (strcmp(text, "BRIDGE") == 0) {
        return ROLE_BRIDGE;
    }
    if (strcmp(text, "RIGHT_C") == 0) {
        return ROLE_RIGHT_C;
    }
    if (strcmp(text, "RIGHT_D") == 0) {
        return ROLE_RIGHT_D;
    }
    fail_line("unknown relation role", line_number);
    return ROLE_LEFT_A;
}

static const char *role_first(enum role role) {
    static const char *const first[RELATION_COUNT] = {
        "A", "B", "U", "C", "D"
    };
    return first[(size_t)role];
}

static const char *role_second(enum role role) {
    static const char *const second[RELATION_COUNT] = {
        "U", "U", "V", "V", "V"
    };
    return second[(size_t)role];
}

static void transpose_coefficients(
    int output[COEFFICIENT_COUNT],
    const int input[COEFFICIENT_COUNT]
) {
    output[0] = input[0];
    output[1] = input[2];
    output[2] = input[1];
    output[3] = input[3];
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
    int hubs_seen = 0;
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
                || strcmp(token[0], "CATCAS_ALGEBRAIC_TWO_HUB_TREE") != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid two-hub header", line_number);
            }
            header_seen = 1;
            continue;
        }
        if (strcmp(token[0], "TYPE") == 0) {
            if (
                type_seen
                || hubs_seen
                || count != 2U
                || strcmp(token[1], "BOOLEAN_F3") != 0
            ) {
                fail_line("invalid TYPE", line_number);
            }
            type_seen = 1;
            continue;
        }
        if (strcmp(token[0], "HUBS") == 0) {
            if (
                !type_seen
                || hubs_seen
                || count != 3U
                || strcmp(token[1], "U") != 0
                || strcmp(token[2], "V") != 0
            ) {
                fail_line("invalid HUBS", line_number);
            }
            hubs_seen = 1;
            continue;
        }
        if (strcmp(token[0], "RELATION") == 0) {
            if (!type_seen || !hubs_seen || count != 8U) {
                fail_line("invalid RELATION", line_number);
            }
            const enum role role = parse_role(token[1], line_number);
            struct relation_spec *spec = &process.relation[(size_t)role];
            if (spec->present) {
                fail_line("duplicate relation role", line_number);
            }
            int parsed[COEFFICIENT_COUNT];
            for (
                size_t coefficient = 0U;
                coefficient < COEFFICIENT_COUNT;
                ++coefficient
            ) {
                parsed[coefficient] = parse_coefficient(
                    token[4U + coefficient],
                    line_number
                );
            }
            if (
                strcmp(token[2], role_first(role)) == 0
                && strcmp(token[3], role_second(role)) == 0
            ) {
                memcpy(spec->coefficient, parsed, sizeof(parsed));
            } else if (
                strcmp(token[2], role_second(role)) == 0
                && strcmp(token[3], role_first(role)) == 0
            ) {
                transpose_coefficients(spec->coefficient, parsed);
            } else {
                fail_line("relation endpoints do not match role", line_number);
            }
            if (!relation_is_bi_total(spec->coefficient)) {
                fail_line(
                    "relation is not bi-total on Boolean_F3",
                    line_number
                );
            }
            spec->present = 1;
            continue;
        }
        if (strcmp(token[0], "END") == 0) {
            if (count != 1U) {
                fail_line("invalid END", line_number);
            }
            end_seen = 1;
            continue;
        }
        fail_line("unknown two-hub record", line_number);
    }
    if (ferror(stream)) {
        fail("failed to read complete two-hub process");
    }
    if (fclose(stream) != 0) {
        perror(path);
        exit(2);
    }
    if (!header_seen || !type_seen || !hubs_seen || !end_seen) {
        fail("two-hub process is incomplete");
    }
    for (size_t role = 0U; role < RELATION_COUNT; ++role) {
        if (!process.relation[role].present) {
            fail("two-hub process is missing a relation role");
        }
    }
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

static struct carrier make_carrier(int identity) {
    struct carrier carrier = {
        .baseline = checked_calloc(
            CARRIER_CELLS,
            sizeof(*carrier.baseline)
        ),
        .working = checked_calloc(
            CARRIER_CELLS,
            sizeof(*carrier.working)
        ),
        .cells = CARRIER_CELLS
    };
    for (size_t index = 0U; index < carrier.cells; ++index) {
        const double angle =
            0.173
            + 0.071 * (double)index
            + 0.013 * sin(0.29 * (double)index + 0.031 * identity);
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

static size_t input_start(enum role role) {
    return (size_t)role * COEFFICIENT_COUNT;
}

static size_t message_start(size_t message) {
    return INPUT_CELLS + message * COEFFICIENT_COUNT;
}

static size_t boundary_start(size_t boundary) {
    return INPUT_CELLS + MESSAGE_CELLS + boundary * COEFFICIENT_COUNT;
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

static void apply_factor(
    struct carrier *carrier,
    size_t output,
    const double complex factor[COEFFICIENT_COUNT],
    int inverse
) {
    for (
        size_t coefficient = 0U;
        coefficient < COEFFICIENT_COUNT;
        ++coefficient
    ) {
        multiply_relation(
            carrier,
            output + coefficient,
            inverse
                ? conj(factor[coefficient])
                : factor[coefficient]
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
    for (
        size_t index = 0U;
        index < COEFFICIENT_COUNT;
        ++index
    ) {
        double complex factor = root3(coefficient[index]);
        if (inverse) {
            factor = conj(factor);
        }
        multiply_relation(carrier, start + index, factor);
    }
}

static void forward_messages(struct carrier *carrier) {
    apply_resultant(
        carrier,
        input_start(ROLE_LEFT_A),
        0,
        input_start(ROLE_BRIDGE),
        1,
        message_start(0U),
        0
    );
    apply_resultant(
        carrier,
        input_start(ROLE_LEFT_B),
        0,
        input_start(ROLE_BRIDGE),
        1,
        message_start(1U),
        0
    );
}

static void apply_boundary_index(
    struct carrier *carrier,
    size_t index,
    int bypass_messages,
    int inverse
) {
    static const enum role left_role[4] = {
        ROLE_LEFT_A, ROLE_RIGHT_C, ROLE_LEFT_A, ROLE_LEFT_A
    };
    static const enum role right_role[4] = {
        ROLE_LEFT_B, ROLE_RIGHT_D, ROLE_RIGHT_C, ROLE_RIGHT_D
    };
    if (index < 2U) {
        apply_resultant(
            carrier,
            input_start(left_role[index]),
            0,
            input_start(right_role[index]),
            0,
            boundary_start(index),
            inverse
        );
        return;
    }
    const size_t cross = index - 2U;
    const size_t left_message = cross / 2U;
    const enum role right =
        (cross & 1U) == 0U ? ROLE_RIGHT_C : ROLE_RIGHT_D;
    apply_resultant(
        carrier,
        bypass_messages
            ? input_start(
                left_message == 0U ? ROLE_LEFT_A : ROLE_LEFT_B
            )
            : message_start(left_message),
        0,
        input_start(right),
        0,
        boundary_start(index),
        inverse
    );
}

static void forward_boundaries(
    struct carrier *carrier,
    int bypass_messages
) {
    for (size_t index = 0U; index < BOUNDARY_COUNT; ++index) {
        apply_boundary_index(
            carrier,
            index,
            bypass_messages,
            0
        );
    }
}

static void inverse_boundaries_correct(
    struct carrier *carrier,
    int bypass_messages
) {
    for (size_t index = BOUNDARY_COUNT; index > 0U; --index) {
        apply_boundary_index(
            carrier,
            index - 1U,
            bypass_messages,
            1
        );
    }
}

static void inverse_boundaries_wrong(struct carrier *carrier) {
    for (size_t index = 0U; index < BOUNDARY_COUNT; ++index) {
        double complex factor[COEFFICIENT_COUNT];
        double complex rotated[COEFFICIENT_COUNT];
        if (index < 2U) {
            static const enum role left[2] = {
                ROLE_LEFT_A, ROLE_RIGHT_C
            };
            static const enum role right[2] = {
                ROLE_LEFT_B, ROLE_RIGHT_D
            };
            resultant_factors(
                carrier,
                input_start(left[index]),
                0,
                input_start(right[index]),
                0,
                factor
            );
        } else {
            const size_t cross = index - 2U;
            resultant_factors(
                carrier,
                message_start(cross / 2U),
                0,
                input_start(
                    (cross & 1U) == 0U
                        ? ROLE_RIGHT_C
                        : ROLE_RIGHT_D
                ),
                0,
                factor
            );
        }
        for (
            size_t coefficient = 0U;
            coefficient < COEFFICIENT_COUNT;
            ++coefficient
        ) {
            rotated[coefficient] = factor[
                (coefficient + 1U) % COEFFICIENT_COUNT
            ];
        }
        apply_factor(
            carrier,
            boundary_start(index),
            rotated,
            1
        );
    }
}

static void inverse_boundaries_scrambled(struct carrier *carrier) {
    for (size_t index = 0U; index < BOUNDARY_COUNT; ++index) {
        double complex factor[COEFFICIENT_COUNT];
        if (index < 2U) {
            static const enum role left[2] = {
                ROLE_LEFT_A, ROLE_RIGHT_C
            };
            static const enum role right[2] = {
                ROLE_LEFT_B, ROLE_RIGHT_D
            };
            resultant_factors(
                carrier,
                input_start(left[index]),
                0,
                input_start(right[index]),
                0,
                factor
            );
        } else {
            const size_t cross = index - 2U;
            resultant_factors(
                carrier,
                message_start(cross / 2U),
                0,
                input_start(
                    (cross & 1U) == 0U
                        ? ROLE_RIGHT_C
                        : ROLE_RIGHT_D
                ),
                0,
                factor
            );
        }
        apply_factor(
            carrier,
            boundary_start((index + 1U) % BOUNDARY_COUNT),
            factor,
            1
        );
    }
}

static void inverse_messages(
    struct carrier *carrier,
    int omit_second
) {
    for (size_t index = MESSAGE_COUNT; index > 0U; --index) {
        const size_t message = index - 1U;
        if (omit_second && message == 1U) {
            continue;
        }
        apply_resultant(
            carrier,
            input_start(
                message == 0U ? ROLE_LEFT_A : ROLE_LEFT_B
            ),
            0,
            input_start(ROLE_BRIDGE),
            1,
            message_start(message),
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
    const struct carrier *carrier
) {
    static const char *const label[BOUNDARY_COUNT] = {
        "A:B", "C:D", "A:C", "A:D", "B:C", "B:D"
    };
    struct boundary_record boundary = {
        .factor_fnv1a64 = UINT64_C(14695981039346656037),
        .maximum_root_error = 0.0
    };
    for (size_t index = 0U; index < BOUNDARY_COUNT; ++index) {
        boundary.factor_fnv1a64 = fnv1a64_update(
            boundary.factor_fnv1a64,
            (const unsigned char *)label[index],
            strlen(label[index])
        );
        for (
            size_t coefficient = 0U;
            coefficient < COEFFICIENT_COUNT;
            ++coefficient
        ) {
            double distance = 0.0;
            const int decoded = decode_root3(
                relation(
                    carrier,
                    boundary_start(index) + coefficient
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

static int boundary_differs(
    const struct boundary_record *first,
    const struct boundary_record *second
) {
    return memcmp(
        first->coefficient,
        second->coefficient,
        sizeof(first->coefficient)
    ) != 0;
}

static int factors_differ(
    const double complex first[COEFFICIENT_COUNT],
    const double complex second[COEFFICIENT_COUNT]
) {
    for (
        size_t index = 0U;
        index < COEFFICIENT_COUNT;
        ++index
    ) {
        if (cabs(first[index] - second[index]) > ROOT_TOLERANCE) {
            return 1;
        }
    }
    return 0;
}

static int factor_nontrivial(
    const double complex factor[COEFFICIENT_COUNT]
) {
    for (
        size_t index = 0U;
        index < COEFFICIENT_COUNT;
        ++index
    ) {
        if (cabs(factor[index] - 1.0) > ROOT_TOLERANCE) {
            return 1;
        }
    }
    return 0;
}

static void inspect_applicability(
    const struct carrier *carrier,
    struct execution *execution
) {
    double complex first[COEFFICIENT_COUNT] = {0};
    double complex previous[COEFFICIENT_COUNT] = {0};
    for (size_t index = 0U; index < BOUNDARY_COUNT; ++index) {
        double complex factor[COEFFICIENT_COUNT];
        double complex rotated[COEFFICIENT_COUNT];
        if (index < 2U) {
            static const enum role left[2] = {
                ROLE_LEFT_A, ROLE_RIGHT_C
            };
            static const enum role right[2] = {
                ROLE_LEFT_B, ROLE_RIGHT_D
            };
            resultant_factors(
                carrier,
                input_start(left[index]),
                0,
                input_start(right[index]),
                0,
                factor
            );
        } else {
            const size_t cross = index - 2U;
            resultant_factors(
                carrier,
                message_start(cross / 2U),
                0,
                input_start(
                    (cross & 1U) == 0U
                        ? ROLE_RIGHT_C
                        : ROLE_RIGHT_D
                ),
                0,
                factor
            );
        }
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
        if (index == 0U) {
            memcpy(first, factor, sizeof(first));
        } else {
            execution->geometry_applicable |=
                factors_differ(previous, factor);
        }
        memcpy(previous, factor, sizeof(previous));
    }
    execution->geometry_applicable |=
        factors_differ(previous, first);
    double complex final_message[COEFFICIENT_COUNT];
    resultant_factors(
        carrier,
        input_start(ROLE_LEFT_B),
        0,
        input_start(ROLE_BRIDGE),
        1,
        final_message
    );
    execution->omitted_message_applicable =
        factor_nontrivial(final_message);
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
    for (size_t role = 0U; role < RELATION_COUNT; ++role) {
        apply_encoding(
            carrier,
            input_start((enum role)role),
            process->relation[role].coefficient,
            0
        );
    }
    forward_messages(carrier);
    const int bypass = mode == INVERSE_BYPASS_MESSAGE;
    forward_boundaries(carrier, bypass);
    struct execution execution = {
        .boundary = latch_boundary(carrier)
    };
    execution.displacement_l2 =
        carrier_displacement(carrier, &borrowed);
    inspect_applicability(carrier, &execution);

    if (mode == INVERSE_WRONG_BOUNDARY) {
        inverse_boundaries_wrong(carrier);
    } else if (mode == INVERSE_SCRAMBLED_BOUNDARY) {
        inverse_boundaries_scrambled(carrier);
    } else {
        inverse_boundaries_correct(carrier, bypass);
    }
    inverse_messages(
        carrier,
        mode == INVERSE_OMITTED_MESSAGE
    );
    for (size_t role = RELATION_COUNT; role > 0U; --role) {
        apply_encoding(
            carrier,
            input_start((enum role)(role - 1U)),
            process->relation[role - 1U].coefficient,
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
    const struct execution *execution
) {
    printf(
        "{\"mode\":\"%s\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"port_type\":\"BOOLEAN_F3\","
        "\"closed_hub_count\":2,"
        "\"input_relation_count\":5,"
        "\"phase_resident_relation_messages\":2,"
        "\"boundary_factor_count\":6,"
        "\"carrier_cells\":52,"
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
        (unsigned long long)execution->boundary.factor_fnv1a64,
        execution->boundary.coefficient[0][0],
        execution->boundary.coefficient[0][1],
        execution->boundary.coefficient[0][2],
        execution->boundary.coefficient[0][3],
        execution->boundary.coefficient[BOUNDARY_COUNT - 1U][0],
        execution->boundary.coefficient[BOUNDARY_COUNT - 1U][1],
        execution->boundary.coefficient[BOUNDARY_COUNT - 1U][2],
        execution->boundary.coefficient[BOUNDARY_COUNT - 1U][3],
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
            "usage: %s PROCESS.aret [REUSE_PROCESS.aret]\n",
            argv[0]
        );
        return 2;
    }
    const struct process process = read_process(argv[1]);
    const struct process reuse_process =
        argc == 3 ? read_process(argv[2]) : read_process(argv[1]);

    struct carrier carrier = make_carrier(2081);
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

    carrier = make_carrier(2081);
    const struct execution wrong = execute(
        &carrier,
        &process,
        INVERSE_WRONG_BOUNDARY
    );
    free_carrier(&carrier);
    carrier = make_carrier(2081);
    const struct execution scrambled = execute(
        &carrier,
        &process,
        INVERSE_SCRAMBLED_BOUNDARY
    );
    free_carrier(&carrier);
    carrier = make_carrier(2081);
    const struct execution omitted = execute(
        &carrier,
        &process,
        INVERSE_OMITTED_MESSAGE
    );
    free_carrier(&carrier);
    carrier = make_carrier(2081);
    const struct execution bypass = execute(
        &carrier,
        &process,
        INVERSE_BYPASS_MESSAGE
    );
    free_carrier(&carrier);

    print_execution("algebraic-two-hub-tree", &process, &nominal);
    print_execution(
        argc == 3
            ? "actual-restored-cross-process-reuse"
            : "actual-restored-reuse",
        &reuse_process,
        &reuse
    );
    print_execution("wrong-boundary-inverse", &process, &wrong);
    print_execution(
        "geometry-scrambled-boundary-inverse",
        &process,
        &scrambled
    );
    print_execution(
        "omitted-relation-message-inverse",
        &process,
        &omitted
    );
    print_execution(
        "bypassed-relation-message-forward",
        &process,
        &bypass
    );
    const int bypass_applicable =
        boundary_differs(&nominal.boundary, &bypass.boundary);
    printf(
        "{\"mode\":\"control-applicability\","
        "\"wrong_boundary\":%s,"
        "\"geometry_scrambled\":%s,"
        "\"omitted_relation_message\":%s,"
        "\"bypassed_relation_message\":%s}\n",
        wrong.wrong_applicable ? "true" : "false",
        scrambled.geometry_applicable ? "true" : "false",
        omitted.omitted_message_applicable ? "true" : "false",
        bypass_applicable ? "true" : "false"
    );

    const int valid = (
        nominal.boundary.maximum_root_error <= ROOT_TOLERANCE
        && reuse.boundary.maximum_root_error <= ROOT_TOLERANCE
        && bypass.boundary.maximum_root_error <= ROOT_TOLERANCE
        && nominal.restoration_max_abs <= RESTORATION_TOLERANCE
        && reuse.restoration_max_abs <= RESTORATION_TOLERANCE
        && bypass.restoration_max_abs <= RESTORATION_TOLERANCE
        && nominal.carrier_integrity_error <= RESTORATION_TOLERANCE
        && reuse.carrier_integrity_error <= RESTORATION_TOLERANCE
        && bypass.carrier_integrity_error <= RESTORATION_TOLERANCE
        && (
            !wrong.wrong_applicable
            || wrong.restoration_max_abs >= CONTROL_MINIMUM
        )
        && (
            !scrambled.geometry_applicable
            || scrambled.restoration_max_abs >= CONTROL_MINIMUM
        )
        && (
            !omitted.omitted_message_applicable
            || omitted.restoration_max_abs >= CONTROL_MINIMUM
        )
    );
    return valid ? 0 : 1;
}
