/*
 * Mutable CAT_CAS frontier: algebraic open relations over Boolean F3 ports.
 *
 * A binary relation is the Boolean-domain zero set of
 *
 *   c00 + c10*x + c01*y + c11*x*y = 0  (mod 3).
 *
 * The native carrier stores the four polynomial coefficients as relative
 * complex phases.  For relations f(x,y)=A(x)y+B(x) and
 * g(y,z)=C(z)y+D(z), the shared y port is closed by the resultant
 *
 *   R(x,z) = B(x)C(z) - A(x)D(z).
 *
 * Each output coefficient is formed by roots-of-unity product polynomials.
 * The native operator does not enumerate x, y, z, tuples, witnesses, or
 * assignments and never decodes a coefficient during forward evolution.
 */

#define _GNU_SOURCE
#include <complex.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PORT_COUNT 3U
#define RELATION_COUNT 2U
#define COEFFICIENT_COUNT 4U
#define CARRIER_CELLS 12U
#define MAX_NAME 31U
#define MAX_INPUT_BYTES 8192U
#define MAX_TOKENS 12U
#define RESTORATION_TOLERANCE 2e-12
#define CONTROL_MINIMUM 1e-3
#define ROOT_TOLERANCE 2e-10

enum cell_index {
    LEFT_00 = 0,
    LEFT_10 = 1,
    LEFT_01 = 2,
    LEFT_11 = 3,
    RIGHT_00 = 4,
    RIGHT_10 = 5,
    RIGHT_01 = 6,
    RIGHT_11 = 7,
    BOUNDARY_00 = 8,
    BOUNDARY_10 = 9,
    BOUNDARY_01 = 10,
    BOUNDARY_11 = 11
};

enum inverse_mode {
    INVERSE_CORRECT = 0,
    INVERSE_WRONG = 1,
    INVERSE_REORDERED = 2,
    INVERSE_OMITTED = 3
};

struct port_definition {
    char name[MAX_NAME + 1U];
};

struct relation_definition {
    char name[MAX_NAME + 1U];
    char first[MAX_NAME + 1U];
    char second[MAX_NAME + 1U];
    int coefficient[COEFFICIENT_COUNT];
};

struct parsed_process {
    struct port_definition ports[PORT_COUNT];
    struct relation_definition relations[RELATION_COUNT];
    size_t port_count;
    size_t relation_count;
    char closed_port[MAX_NAME + 1U];
    char boundary_first[MAX_NAME + 1U];
    char boundary_second[MAX_NAME + 1U];
    int close_seen;
    int boundary_seen;
    uint64_t source_fnv1a64;
};

struct process {
    struct relation_definition left;
    struct relation_definition right;
    char boundary_first[MAX_NAME + 1U];
    char boundary_second[MAX_NAME + 1U];
    uint64_t source_fnv1a64;
};

struct carrier {
    double complex baseline[CARRIER_CELLS];
    double complex working[CARRIER_CELLS];
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

static uint64_t fnv1a64(const unsigned char *bytes, size_t length) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0; index < length; ++index) {
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
        text[0] == '\0'
        || text[1] != '\0'
        || text[0] < '0'
        || text[0] > '2'
    ) {
        fail_line("coefficient must be canonical 0, 1, or 2", line_number);
    }
    return text[0] - '0';
}

static int find_port(
    const struct parsed_process *parsed,
    const char *name
) {
    for (size_t index = 0U; index < parsed->port_count; ++index) {
        if (strcmp(parsed->ports[index].name, name) == 0) {
            return (int)index;
        }
    }
    return -1;
}

static int relation_connects(
    const struct relation_definition *relation,
    const char *first,
    const char *second
) {
    return (
        (
            strcmp(relation->first, first) == 0
            && strcmp(relation->second, second) == 0
        )
        || (
            strcmp(relation->first, second) == 0
            && strcmp(relation->second, first) == 0
        )
    );
}

static struct relation_definition orient_relation(
    const struct relation_definition *source,
    const char *first,
    const char *second
) {
    struct relation_definition result = *source;
    if (
        strcmp(source->first, first) == 0
        && strcmp(source->second, second) == 0
    ) {
        return result;
    }
    if (
        strcmp(source->first, second) != 0
        || strcmp(source->second, first) != 0
    ) {
        fail("relation orientation contradicts geometry");
    }
    memcpy(result.first, first, strlen(first) + 1U);
    memcpy(result.second, second, strlen(second) + 1U);
    const int old_10 = result.coefficient[1];
    result.coefficient[1] = result.coefficient[2];
    result.coefficient[2] = old_10;
    return result;
}

static int mod3(int value) {
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static int affine_has_boolean_zero(int slope, int offset) {
    return offset == 0 || mod3(slope + offset) == 0;
}

static int left_total_toward_internal(
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

static int right_total_toward_internal(
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

static struct process validate_process(
    const struct parsed_process *parsed
) {
    if (
        parsed->port_count != PORT_COUNT
        || parsed->relation_count != RELATION_COUNT
        || !parsed->close_seen
        || !parsed->boundary_seen
    ) {
        fail("process has incomplete typed geometry");
    }
    const int closed = find_port(parsed, parsed->closed_port);
    const int boundary_first =
        find_port(parsed, parsed->boundary_first);
    const int boundary_second =
        find_port(parsed, parsed->boundary_second);
    if (
        closed < 0
        || boundary_first < 0
        || boundary_second < 0
        || closed == boundary_first
        || closed == boundary_second
        || boundary_first == boundary_second
    ) {
        fail("invalid closed or boundary port geometry");
    }
    int left_index = -1;
    int right_index = -1;
    for (size_t index = 0U; index < RELATION_COUNT; ++index) {
        const struct relation_definition *relation =
            &parsed->relations[index];
        if (
            find_port(parsed, relation->first) < 0
            || find_port(parsed, relation->second) < 0
            || strcmp(relation->first, relation->second) == 0
        ) {
            fail("relation endpoint is not a distinct declared port");
        }
        if (
            relation_connects(
                relation,
                parsed->boundary_first,
                parsed->closed_port
            )
        ) {
            if (left_index >= 0) {
                fail("multiple left relations");
            }
            left_index = (int)index;
        } else if (
            relation_connects(
                relation,
                parsed->closed_port,
                parsed->boundary_second
            )
        ) {
            if (right_index >= 0) {
                fail("multiple right relations");
            }
            right_index = (int)index;
        } else {
            fail("relations do not form the declared open chain");
        }
    }
    if (left_index < 0 || right_index < 0) {
        fail("open chain is not composable");
    }
    struct process process = {
        .left = orient_relation(
            &parsed->relations[left_index],
            parsed->boundary_first,
            parsed->closed_port
        ),
        .right = orient_relation(
            &parsed->relations[right_index],
            parsed->closed_port,
            parsed->boundary_second
        ),
        .source_fnv1a64 = parsed->source_fnv1a64
    };
    memcpy(
        process.boundary_first,
        parsed->boundary_first,
        strlen(parsed->boundary_first) + 1U
    );
    memcpy(
        process.boundary_second,
        parsed->boundary_second,
        strlen(parsed->boundary_second) + 1U
    );
    if (
        !left_total_toward_internal(&process.left)
        || !right_total_toward_internal(&process.right)
    ) {
        fail("relation is not total toward the closed Boolean port");
    }
    return process;
}

static struct process read_process(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    if (fseek(stream, 0L, SEEK_END) != 0) {
        perror(path);
        exit(2);
    }
    const long end = ftell(stream);
    if (end < 0L || (unsigned long)end > MAX_INPUT_BYTES) {
        fail("process byte count is outside the accepted envelope");
    }
    if (fseek(stream, 0L, SEEK_SET) != 0) {
        perror(path);
        exit(2);
    }
    const size_t length = (size_t)end;
    char *bytes = malloc(length + 1U);
    if (bytes == NULL) {
        fail("allocation failed");
    }
    if (fread(bytes, 1U, length, stream) != length) {
        fail("failed to read complete process");
    }
    if (fclose(stream) != 0) {
        perror(path);
        exit(2);
    }
    if (memchr(bytes, '\0', length) != NULL) {
        fail("embedded NUL");
    }
    if (memchr(bytes, '\r', length) != NULL) {
        fail("noncanonical carriage return");
    }
    const uint64_t source_hash =
        fnv1a64((const unsigned char *)bytes, length);
    bytes[length] = '\0';

    struct parsed_process parsed = {
        .port_count = 0U,
        .relation_count = 0U,
        .close_seen = 0,
        .boundary_seen = 0,
        .source_fnv1a64 = source_hash
    };
    int header_seen = 0;
    int end_seen = 0;
    size_t line_number = 0U;
    char *save = NULL;
    for (
        char *raw = strtok_r(bytes, "\n", &save);
        raw != NULL;
        raw = strtok_r(NULL, "\n", &save)
    ) {
        ++line_number;
        char *line = trim(raw);
        if (*line == '\0' || *line == '#') {
            continue;
        }
        if (end_seen) {
            fail_line("content follows END", line_number);
        }
        char *tokens[MAX_TOKENS] = {0};
        const size_t token_count = split_tokens(line, tokens);
        if (!header_seen) {
            if (
                token_count != 2U
                || strcmp(
                    tokens[0],
                    "CATCAS_ALGEBRAIC_RELATION_PROCESS"
                ) != 0
                || strcmp(tokens[1], "1") != 0
            ) {
                fail_line("invalid process header", line_number);
            }
            header_seen = 1;
            continue;
        }
        if (strcmp(tokens[0], "PORT") == 0) {
            if (
                token_count != 3U
                || parsed.port_count == PORT_COUNT
                || strcmp(tokens[2], "BOOLEAN_F3") != 0
            ) {
                fail_line("invalid PORT", line_number);
            }
            struct port_definition *port =
                &parsed.ports[parsed.port_count];
            copy_name(port->name, tokens[1], line_number);
            if (find_port(&parsed, port->name) >= 0) {
                fail_line("duplicate PORT", line_number);
            }
            ++parsed.port_count;
            continue;
        }
        if (strcmp(tokens[0], "RELATION") == 0) {
            if (
                token_count != 9U
                || parsed.relation_count == RELATION_COUNT
            ) {
                fail_line("invalid RELATION", line_number);
            }
            struct relation_definition *relation =
                &parsed.relations[parsed.relation_count];
            copy_name(relation->name, tokens[1], line_number);
            copy_name(relation->first, tokens[2], line_number);
            copy_name(relation->second, tokens[3], line_number);
            for (
                size_t coefficient = 0U;
                coefficient < COEFFICIENT_COUNT;
                ++coefficient
            ) {
                relation->coefficient[coefficient] =
                    parse_coefficient(
                        tokens[4U + coefficient],
                        line_number
                    );
            }
            if (strcmp(tokens[8], "ZEROSET") != 0) {
                fail_line("RELATION must end with ZEROSET", line_number);
            }
            for (
                size_t index = 0U;
                index < parsed.relation_count;
                ++index
            ) {
                if (
                    strcmp(
                        relation->name,
                        parsed.relations[index].name
                    ) == 0
                ) {
                    fail_line("duplicate relation name", line_number);
                }
            }
            ++parsed.relation_count;
            continue;
        }
        if (strcmp(tokens[0], "CLOSE") == 0) {
            if (token_count != 2U || parsed.close_seen) {
                fail_line("invalid CLOSE", line_number);
            }
            copy_name(parsed.closed_port, tokens[1], line_number);
            parsed.close_seen = 1;
            continue;
        }
        if (strcmp(tokens[0], "BOUNDARY") == 0) {
            if (token_count != 3U || parsed.boundary_seen) {
                fail_line("invalid BOUNDARY", line_number);
            }
            copy_name(parsed.boundary_first, tokens[1], line_number);
            copy_name(parsed.boundary_second, tokens[2], line_number);
            parsed.boundary_seen = 1;
            continue;
        }
        if (strcmp(tokens[0], "END") == 0) {
            if (token_count != 1U) {
                fail_line("invalid END", line_number);
            }
            end_seen = 1;
            continue;
        }
        fail_line("unknown process record", line_number);
    }
    free(bytes);
    if (!header_seen || !end_seen) {
        fail("process is incomplete");
    }
    return validate_process(&parsed);
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

static struct carrier make_carrier(int identity) {
    struct carrier carrier;
    for (size_t index = 0U; index < CARRIER_CELLS; ++index) {
        const double angle =
            0.137
            + 0.061 * (double)index
            + 0.019 * sin(
                0.29 * (double)index + 0.037 * (double)identity
            );
        carrier.baseline[index] = cexp(I * angle);
        carrier.working[index] = carrier.baseline[index];
    }
    return carrier;
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

static void elimination_factors(
    const struct carrier *carrier,
    double complex factor[COEFFICIENT_COUNT]
) {
    factor[0] = difference_product(
        carrier,
        LEFT_00,
        RIGHT_10,
        LEFT_01,
        RIGHT_00
    );
    factor[1] = difference_product(
        carrier,
        LEFT_10,
        RIGHT_10,
        LEFT_11,
        RIGHT_00
    );
    factor[2] = difference_product(
        carrier,
        LEFT_00,
        RIGHT_11,
        LEFT_01,
        RIGHT_01
    );
    factor[3] = difference_product(
        carrier,
        LEFT_10,
        RIGHT_11,
        LEFT_11,
        RIGHT_01
    );
}

static void apply_elimination(
    struct carrier *carrier,
    int inverse
) {
    double complex factor[COEFFICIENT_COUNT];
    elimination_factors(carrier, factor);
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        multiply_relation(
            carrier,
            BOUNDARY_00 + index,
            inverse ? conj(factor[index]) : factor[index]
        );
    }
}

static void apply_wrong_elimination_inverse(
    struct carrier *carrier
) {
    double complex factor[COEFFICIENT_COUNT];
    elimination_factors(carrier, factor);
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        multiply_relation(
            carrier,
            BOUNDARY_00 + index,
            conj(factor[(index + 1U) % COEFFICIENT_COUNT])
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
    struct boundary_record boundary = {
        .maximum_root_error = 0.0
    };
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        boundary.phase[index] =
            relation(carrier, BOUNDARY_00 + index);
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
    for (size_t index = 0U; index < CARRIER_CELLS; ++index) {
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
    for (size_t index = 0U; index < CARRIER_CELLS; ++index) {
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
    for (size_t index = 0U; index < CARRIER_CELLS; ++index) {
        const double error =
            fabs(cabs(carrier->working[index]) - 1.0);
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
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

static struct execution execute(
    struct carrier *carrier,
    const struct process *process,
    enum inverse_mode mode
) {
    const struct carrier borrowed = *carrier;
    apply_encoding(
        carrier,
        LEFT_00,
        process->left.coefficient,
        0
    );
    apply_encoding(
        carrier,
        RIGHT_00,
        process->right.coefficient,
        0
    );
    apply_elimination(carrier, 0);

    struct execution execution = {
        .boundary = latch_boundary(carrier),
        .displacement_l2 =
            carrier_displacement(carrier, &borrowed),
        .wrong_applicable = 0,
        .reordered_applicable = 0,
        .omitted_applicable = 0
    };

    double complex factor[COEFFICIENT_COUNT];
    double complex rotated[COEFFICIENT_COUNT];
    elimination_factors(carrier, factor);
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        rotated[index] = factor[
            (index + 1U) % COEFFICIENT_COUNT
        ];
        if (cabs(factor[index] - 1.0) > ROOT_TOLERANCE) {
            execution.omitted_applicable = 1;
        }
    }
    execution.wrong_applicable = factors_differ(factor, rotated);

    struct carrier reordered_probe = *carrier;
    apply_encoding(
        &reordered_probe,
        LEFT_00,
        process->left.coefficient,
        1
    );
    double complex reordered_factor[COEFFICIENT_COUNT];
    elimination_factors(&reordered_probe, reordered_factor);
    execution.reordered_applicable =
        factors_differ(factor, reordered_factor);

    if (mode == INVERSE_CORRECT) {
        apply_elimination(carrier, 1);
        apply_encoding(
            carrier,
            RIGHT_00,
            process->right.coefficient,
            1
        );
        apply_encoding(
            carrier,
            LEFT_00,
            process->left.coefficient,
            1
        );
    } else if (mode == INVERSE_WRONG) {
        apply_wrong_elimination_inverse(carrier);
        apply_encoding(
            carrier,
            RIGHT_00,
            process->right.coefficient,
            1
        );
        apply_encoding(
            carrier,
            LEFT_00,
            process->left.coefficient,
            1
        );
    } else if (mode == INVERSE_REORDERED) {
        apply_encoding(
            carrier,
            LEFT_00,
            process->left.coefficient,
            1
        );
        apply_elimination(carrier, 1);
        apply_encoding(
            carrier,
            RIGHT_00,
            process->right.coefficient,
            1
        );
    } else if (mode == INVERSE_OMITTED) {
        apply_encoding(
            carrier,
            RIGHT_00,
            process->right.coefficient,
            1
        );
        apply_encoding(
            carrier,
            LEFT_00,
            process->left.coefficient,
            1
        );
    } else {
        fail("unknown inverse mode");
    }
    execution.restoration_max_abs =
        restoration_error(carrier, &borrowed);
    execution.carrier_integrity_error = integrity_error(carrier);
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
        "\"carrier_cells\":%u,"
        "\"tuple_slots\":0,"
        "\"witness_slots\":0,"
        "\"retained_inverse_factors\":0,"
        "\"boundary_coefficients\":[%d,%d,%d,%d],"
        "\"boundary_phases\":["
        "[%.12g,%.12g],[%.12g,%.12g],"
        "[%.12g,%.12g],[%.12g,%.12g]],"
        "\"maximum_root_error\":%.12g,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_error\":%.12g}\n",
        mode,
        (unsigned long long)process->source_fnv1a64,
        CARRIER_CELLS,
        execution->boundary.coefficient[0],
        execution->boundary.coefficient[1],
        execution->boundary.coefficient[2],
        execution->boundary.coefficient[3],
        creal(execution->boundary.phase[0]),
        cimag(execution->boundary.phase[0]),
        creal(execution->boundary.phase[1]),
        cimag(execution->boundary.phase[1]),
        creal(execution->boundary.phase[2]),
        cimag(execution->boundary.phase[2]),
        creal(execution->boundary.phase[3]),
        cimag(execution->boundary.phase[3]),
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
            "usage: %s PROCESS.arel [REUSE_PROCESS.arel]\n",
            argv[0]
        );
        return 2;
    }
    const struct process process = read_process(argv[1]);
    const struct process reuse_process =
        argc == 3 ? read_process(argv[2]) : process;

    struct carrier carrier = make_carrier(913);
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

    struct carrier wrong_carrier = make_carrier(913);
    const struct execution wrong = execute(
        &wrong_carrier,
        &process,
        INVERSE_WRONG
    );
    struct carrier reordered_carrier = make_carrier(913);
    const struct execution reordered = execute(
        &reordered_carrier,
        &process,
        INVERSE_REORDERED
    );
    struct carrier omitted_carrier = make_carrier(913);
    const struct execution omitted = execute(
        &omitted_carrier,
        &process,
        INVERSE_OMITTED
    );

    print_execution("algebraic-open-relation", &process, &nominal);
    print_execution(
        argc == 3
            ? "actual-restored-cross-process-reuse"
            : "actual-restored-reuse",
        &reuse_process,
        &reuse
    );
    print_execution("wrong-inverse", &process, &wrong);
    print_execution("reordered-inverse", &process, &reordered);
    print_execution("omitted-inverse", &process, &omitted);
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
    return valid ? 0 : 1;
}
