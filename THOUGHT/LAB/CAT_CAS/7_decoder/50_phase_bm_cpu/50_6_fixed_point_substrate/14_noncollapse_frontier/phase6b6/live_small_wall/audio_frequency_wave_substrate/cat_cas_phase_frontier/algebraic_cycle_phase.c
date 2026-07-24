#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: exact cyclic relational phase closure.
 *
 *     A--U--W--V--B
 *        '--Z--'
 *
 * The two U--V paths are composed independently, intersected in phase, then
 * composed with the two leaf relations.  All intermediate relations remain
 * as four complex phases.  The exact composition operator works for empty
 * and partial Boolean relations; no bi-total assumption or scalar witness
 * feedback is used by recurrence.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CCOUNT 4U
#define INPUT_COUNT 6U
#define MESSAGE_COUNT 4U
#define BOUNDARY_COUNT 1U
#define CELLS ((INPUT_COUNT + MESSAGE_COUNT + BOUNDARY_COUNT) * CCOUNT)
#define LINE_CAP 512U
#define TOKEN_CAP 12U

static const double ROOT_TOLERANCE = 4.0e-10;
static const double RESTORATION_TOLERANCE = 2.0e-12;
static const double CONTROL_MINIMUM = 1.0e-3;

enum role {
    LEAF_A = 0,
    UPPER_LEFT = 1,
    UPPER_RIGHT = 2,
    LOWER_LEFT = 3,
    LOWER_RIGHT = 4,
    LEAF_B = 5
};

enum mode {
    MODE_CORRECT = 0,
    MODE_WRONG_BOUNDARY_INVERSE = 1,
    MODE_OMIT_INTERSECTION_INVERSE = 2,
    MODE_BYPASS_LOWER_PATH = 3,
    MODE_ORDINARY_SUM_INTERSECTION = 4
};

struct relation_spec {
    int c[CCOUNT];
    int present;
};

struct process {
    struct relation_spec relation[INPUT_COUNT];
    uint64_t source_hash;
};

struct carrier {
    double complex baseline[CELLS];
    double complex working[CELLS];
};

struct boundary {
    int c[CCOUNT];
    uint64_t hash;
    double maximum_root_error;
};

struct execution {
    struct boundary boundary;
    double displacement_l2;
    double restoration_max_abs;
    double integrity_max_abs;
    int omitted_applicable;
    int wrong_applicable;
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

static double complex unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        fail("nonfinite phase state");
    }
    return value / magnitude;
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

static enum role parse_role(const char *text, size_t line) {
    static const char *const name[INPUT_COUNT] = {
        "LEAF_A", "UPPER_LEFT", "UPPER_RIGHT",
        "LOWER_LEFT", "LOWER_RIGHT", "LEAF_B"
    };
    for (size_t index = 0U; index < INPUT_COUNT; ++index) {
        if (strcmp(text, name[index]) == 0) {
            return (enum role)index;
        }
    }
    fail_line("unknown cycle relation role", line);
    return LEAF_A;
}

static const char *role_first(enum role role) {
    static const char *const first[INPUT_COUNT] = {
        "A", "U", "W", "U", "Z", "B"
    };
    return first[(size_t)role];
}

static const char *role_second(enum role role) {
    static const char *const second[INPUT_COUNT] = {
        "U", "W", "V", "Z", "V", "V"
    };
    return second[(size_t)role];
}

static void transpose(
    int output[CCOUNT],
    const int input[CCOUNT]
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
        .source_hash = UINT64_C(14695981039346656037)
    };
    char line[LINE_CAP];
    size_t line_number = 0U;
    int header = 0;
    int type = 0;
    int topology = 0;
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
        char *token[TOKEN_CAP] = {0};
        const size_t count = tokenize(line, token);
        if (count == 0U) {
            fail_line("blank records are forbidden", line_number);
        }
        if (!header) {
            if (
                count != 2U
                || strcmp(token[0], "CATCAS_ALGEBRAIC_CYCLE") != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid cycle header", line_number);
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
        } else if (strcmp(token[0], "TOPOLOGY") == 0) {
            if (
                !type
                || topology
                || count != 2U
                || strcmp(token[1], "DIAMOND_U_W_Z_V") != 0
            ) {
                fail_line("invalid TOPOLOGY", line_number);
            }
            topology = 1;
        } else if (strcmp(token[0], "RELATION") == 0) {
            if (!type || !topology || count != 8U) {
                fail_line("invalid RELATION", line_number);
            }
            const enum role role = parse_role(token[1], line_number);
            struct relation_spec *spec = &process.relation[(size_t)role];
            if (spec->present) {
                fail_line("duplicate cycle relation role", line_number);
            }
            int parsed[CCOUNT];
            for (size_t index = 0U; index < CCOUNT; ++index) {
                if (
                    strlen(token[4U + index]) != 1U
                    || token[4U + index][0] < '0'
                    || token[4U + index][0] > '2'
                ) {
                    fail_line("invalid F3 coefficient", line_number);
                }
                parsed[index] = token[4U + index][0] - '0';
            }
            if (
                strcmp(token[2], role_first(role)) == 0
                && strcmp(token[3], role_second(role)) == 0
            ) {
                memcpy(spec->c, parsed, sizeof(parsed));
            } else if (
                strcmp(token[2], role_second(role)) == 0
                && strcmp(token[3], role_first(role)) == 0
            ) {
                transpose(spec->c, parsed);
            } else {
                fail_line("relation endpoints do not match role", line_number);
            }
            spec->present = 1;
        } else if (strcmp(token[0], "END") == 0) {
            if (count != 1U) {
                fail_line("invalid END", line_number);
            }
            end = 1;
        } else {
            fail_line("unknown cycle record", line_number);
        }
    }
    if (ferror(stream) || fclose(stream) != 0) {
        fail("failed to read cycle process");
    }
    if (!header || !type || !topology || !end) {
        fail("cycle process is incomplete");
    }
    for (size_t role = 0U; role < INPUT_COUNT; ++role) {
        if (!process.relation[role].present) {
            fail("cycle process is missing a relation role");
        }
    }
    return process;
}

static size_t input_start(enum role role) {
    return (size_t)role * CCOUNT;
}

static size_t message_start(size_t message) {
    return INPUT_COUNT * CCOUNT + message * CCOUNT;
}

static size_t boundary_start(void) {
    return (INPUT_COUNT + MESSAGE_COUNT) * CCOUNT;
}

static struct carrier make_carrier(int identity) {
    struct carrier carrier;
    for (size_t index = 0U; index < CELLS; ++index) {
        const double angle =
            0.193
            + 0.083 * (double)index
            + 0.011 * sin(0.31 * (double)index + 0.017 * identity);
        carrier.baseline[index] = cexp(I * angle);
        carrier.working[index] = carrier.baseline[index];
    }
    return carrier;
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
                        output[out] * symbol_product(left[l], right[r])
                    );
                }
            }
        }
    }
}

static void poly_square(
    const double complex input[CCOUNT],
    double complex output[CCOUNT]
) {
    poly_multiply(input, input, output);
}

static void poly_add(
    const double complex left[CCOUNT],
    const double complex right[CCOUNT],
    double complex output[CCOUNT]
) {
    for (size_t index = 0U; index < CCOUNT; ++index) {
        output[index] = unit(left[index] * right[index]);
    }
}

static void exact_compose_factors(
    const struct carrier *carrier,
    size_t left_start,
    int left_transposed,
    size_t right_start,
    int right_transposed,
    double complex output[CCOUNT]
) {
    double complex left[CCOUNT];
    double complex right[CCOUNT];
    read_poly(carrier, left_start, left_transposed, left);
    read_poly(carrier, right_start, right_transposed, right);

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
    double complex f0_squared[CCOUNT];
    double complex g0_squared[CCOUNT];
    double complex f1_squared[CCOUNT];
    double complex g1_squared[CCOUNT];
    double complex k0[CCOUNT];
    double complex k1[CCOUNT];
    poly_square(f0, f0_squared);
    poly_square(g0, g0_squared);
    poly_square(f1, f1_squared);
    poly_square(g1, g1_squared);
    poly_add(f0_squared, g0_squared, k0);
    poly_add(f1_squared, g1_squared, k1);
    poly_multiply(k0, k1, output);
}

static void intersection_factors(
    const struct carrier *carrier,
    size_t first_start,
    size_t second_start,
    int ordinary_sum,
    double complex output[CCOUNT]
) {
    double complex first[CCOUNT];
    double complex second[CCOUNT];
    read_poly(carrier, first_start, 0, first);
    read_poly(carrier, second_start, 0, second);
    if (ordinary_sum) {
        poly_add(first, second, output);
        return;
    }
    double complex first_squared[CCOUNT];
    double complex second_squared[CCOUNT];
    poly_square(first, first_squared);
    poly_square(second, second_squared);
    poly_add(first_squared, second_squared, output);
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

static void apply_exact_compose(
    struct carrier *carrier,
    size_t left_start,
    int left_transposed,
    size_t right_start,
    int right_transposed,
    size_t output_start,
    int inverse
) {
    double complex factor[CCOUNT];
    exact_compose_factors(
        carrier,
        left_start,
        left_transposed,
        right_start,
        right_transposed,
        factor
    );
    apply_factor(carrier, output_start, factor, inverse);
}

static void apply_intersection(
    struct carrier *carrier,
    enum mode mode,
    int inverse
) {
    double complex factor[CCOUNT];
    if (mode == MODE_BYPASS_LOWER_PATH) {
        read_poly(carrier, message_start(0U), 0, factor);
    } else {
        intersection_factors(
            carrier,
            message_start(0U),
            message_start(1U),
            mode == MODE_ORDINARY_SUM_INTERSECTION,
            factor
        );
    }
    apply_factor(carrier, message_start(2U), factor, inverse);
}

static void apply_encoding(
    struct carrier *carrier,
    size_t start,
    const int c[CCOUNT],
    int inverse
) {
    for (size_t index = 0U; index < CCOUNT; ++index) {
        const double complex factor = root3(c[index]);
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

static struct boundary latch_boundary(const struct carrier *carrier) {
    struct boundary boundary = {
        .hash = UINT64_C(14695981039346656037)
    };
    static const unsigned char label[] = "A:B";
    boundary.hash = hash_bytes(
        boundary.hash,
        label,
        sizeof(label) - 1U
    );
    for (size_t index = 0U; index < CCOUNT; ++index) {
        double distance = 0.0;
        boundary.c[index] = decode_root(
            relative(carrier, boundary_start() + index),
            &distance
        );
        const unsigned char byte = (unsigned char)boundary.c[index];
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
    return memcmp(first->c, second->c, sizeof(first->c)) != 0;
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
    for (size_t index = 0U; index < CELLS; ++index) {
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
    for (size_t index = 0U; index < CELLS; ++index) {
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
    for (size_t index = 0U; index < CELLS; ++index) {
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
    enum mode mode
) {
    const struct carrier borrowed = *carrier;
    for (size_t role = 0U; role < INPUT_COUNT; ++role) {
        apply_encoding(
            carrier,
            input_start((enum role)role),
            process->relation[role].c,
            0
        );
    }
    apply_exact_compose(
        carrier,
        input_start(UPPER_LEFT), 0,
        input_start(UPPER_RIGHT), 0,
        message_start(0U), 0
    );
    apply_exact_compose(
        carrier,
        input_start(LOWER_LEFT), 0,
        input_start(LOWER_RIGHT), 0,
        message_start(1U), 0
    );
    apply_intersection(carrier, mode, 0);
    apply_exact_compose(
        carrier,
        input_start(LEAF_A), 0,
        message_start(2U), 0,
        message_start(3U), 0
    );
    apply_exact_compose(
        carrier,
        message_start(3U), 0,
        input_start(LEAF_B), 1,
        boundary_start(), 0
    );

    struct execution execution = {
        .boundary = latch_boundary(carrier),
        .displacement_l2 = displacement(carrier, &borrowed)
    };
    double complex core_factor[CCOUNT];
    read_poly(carrier, message_start(2U), 0, core_factor);
    execution.omitted_applicable = factor_nontrivial(core_factor);

    double complex boundary_factor[CCOUNT];
    double complex rotated[CCOUNT];
    exact_compose_factors(
        carrier,
        message_start(3U), 0,
        input_start(LEAF_B), 1,
        boundary_factor
    );
    for (size_t index = 0U; index < CCOUNT; ++index) {
        rotated[index] = boundary_factor[(index + 1U) % CCOUNT];
    }
    execution.wrong_applicable = memcmp(
        boundary_factor,
        rotated,
        sizeof(boundary_factor)
    ) != 0;
    apply_factor(
        carrier,
        boundary_start(),
        mode == MODE_WRONG_BOUNDARY_INVERSE
            ? rotated
            : boundary_factor,
        1
    );
    apply_exact_compose(
        carrier,
        input_start(LEAF_A), 0,
        message_start(2U), 0,
        message_start(3U), 1
    );
    if (mode != MODE_OMIT_INTERSECTION_INVERSE) {
        apply_intersection(carrier, mode, 1);
    }
    apply_exact_compose(
        carrier,
        input_start(LOWER_LEFT), 0,
        input_start(LOWER_RIGHT), 0,
        message_start(1U), 1
    );
    apply_exact_compose(
        carrier,
        input_start(UPPER_LEFT), 0,
        input_start(UPPER_RIGHT), 0,
        message_start(0U), 1
    );
    for (size_t role = INPUT_COUNT; role > 0U; --role) {
        apply_encoding(
            carrier,
            input_start((enum role)(role - 1U)),
            process->relation[role - 1U].c,
            1
        );
    }
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
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
        "\"topology\":\"DIAMOND_U_W_Z_V\","
        "\"input_relations\":6,"
        "\"closed_internal_nodes\":4,"
        "\"phase_resident_relation_messages\":4,"
        "\"carrier_cells\":44,"
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
        (unsigned long long)execution->boundary.hash,
        execution->boundary.c[0],
        execution->boundary.c[1],
        execution->boundary.c[2],
        execution->boundary.c[3],
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
            "usage: %s CYCLE.arc [REUSE_CYCLE.arc]\n",
            argv[0]
        );
        return 2;
    }
    const struct process process = read_process(argv[1]);
    const struct process reuse_process =
        argc == 3 ? read_process(argv[2]) : read_process(argv[1]);
    struct carrier carrier = make_carrier(4211);
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

    carrier = make_carrier(4211);
    const struct execution wrong = execute(
        &carrier,
        &process,
        MODE_WRONG_BOUNDARY_INVERSE
    );
    carrier = make_carrier(4211);
    const struct execution omitted = execute(
        &carrier,
        &process,
        MODE_OMIT_INTERSECTION_INVERSE
    );
    carrier = make_carrier(4211);
    const struct execution bypass = execute(
        &carrier,
        &process,
        MODE_BYPASS_LOWER_PATH
    );
    carrier = make_carrier(4211);
    const struct execution ordinary = execute(
        &carrier,
        &process,
        MODE_ORDINARY_SUM_INTERSECTION
    );

    print_execution("algebraic-cycle-exact", &process, &nominal);
    print_execution(
        argc == 3
            ? "actual-restored-cross-process-reuse"
            : "actual-restored-reuse",
        &reuse_process,
        &reuse
    );
    print_execution("wrong-boundary-inverse", &process, &wrong);
    print_execution(
        "omitted-cycle-intersection-inverse",
        &process,
        &omitted
    );
    print_execution("bypassed-lower-cycle-path", &process, &bypass);
    print_execution(
        "ordinary-sum-instead-of-intersection",
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
        "\"omitted_intersection\":%s,"
        "\"bypassed_cycle_path\":%s,"
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
