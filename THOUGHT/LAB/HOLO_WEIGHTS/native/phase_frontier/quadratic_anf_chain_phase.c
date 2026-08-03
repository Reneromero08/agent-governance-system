#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: fixed-schema Boolean ANF relation composition.
 *
 * Public relations:
 *   F(a,b;u) = u + alpha + beta*a*b
 *   G(u,c;v) = v + gamma + delta*u*c
 *   J(v,e;d) = d + eta + theta*v*e
 *
 * Monic substitution closes u and v without expanding Boolean valuations:
 *   H(a,b,c;v) =
 *       v + gamma + delta*alpha*c + delta*beta*a*b*c
 *   Z(a,b,c,e;d) =
 *       d + eta + theta*gamma*e
 *         + theta*delta*alpha*c*e
 *         + theta*delta*beta*a*b*c*e
 *
 * Every coefficient operation is performed on resident F3 phase symbols.
 * H remains unresolved and resident until the actual Z inverse has consumed
 * it. Only the fixed five-coefficient Z boundary is decoded.
 */

#include <complex.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QB_INPUT_RELATIONS 3U
#define QB_INPUT_COEFFICIENTS 3U
#define QB_H_COEFFICIENTS 4U
#define QB_Z_COEFFICIENTS 5U
#define QB_REPEAT_CYCLES 256U
#define QB_PHASE_LOCK_PASSES 3U
#define QB_RESTORATION_TOLERANCE 2.0e-12
#define QB_ROOT_TOLERANCE 2.0e-10
#define QB_CONTROL_MINIMUM_ERROR 1.0e-3
#define QB_MAX_SOURCE_BYTES 4096U

#define QB_F_START 0U
#define QB_G_START (QB_F_START + QB_INPUT_COEFFICIENTS)
#define QB_J_START (QB_G_START + QB_INPUT_COEFFICIENTS)
#define QB_H_START (QB_J_START + QB_INPUT_COEFFICIENTS)
#define QB_Z_START (QB_H_START + QB_H_COEFFICIENTS)
#define QB_BOUNDARY_START (QB_Z_START + QB_Z_COEFFICIENTS)
#define QB_CARRIER_CELLS (QB_BOUNDARY_START + QB_Z_COEFFICIENTS)

enum qb_input_coefficient {
    QB_MONIC = 0,
    QB_CONSTANT = 1,
    QB_NONLINEAR = 2
};

enum qb_h_coefficient {
    QB_H_V = 0,
    QB_H_ONE = 1,
    QB_H_C = 2,
    QB_H_ABC = 3
};

enum qb_z_coefficient {
    QB_Z_D = 0,
    QB_Z_ONE = 1,
    QB_Z_E = 2,
    QB_Z_CE = 3,
    QB_Z_ABCE = 4
};

enum qb_inverse_mode {
    QB_INVERSE_CORRECT = 0,
    QB_INVERSE_WRONG_Z = 1,
    QB_INVERSE_MISSING_Z = 2,
    QB_INVERSE_REORDERED = 3,
    QB_INVERSE_SNAPSHOT = 4,
    QB_INVERSE_ALTERED_FORWARD = 5
};

struct qb_program {
    unsigned char coefficient[QB_INPUT_RELATIONS][QB_INPUT_COEFFICIENTS];
    uint64_t source_fnv1a64;
};

struct qb_carrier {
    double complex baseline[QB_CARRIER_CELLS];
    double complex working[QB_CARRIER_CELLS];
};

struct qb_stats {
    uint64_t phase_ands;
    uint64_t carrier_reads;
    uint64_t phase_cell_updates;
    uint64_t boundary_decodes;
    uint64_t intermediate_decodes;
    uint64_t intermediate_copies;
    uint64_t boundary_copies;
    uint64_t snapshot_loads;
};

struct qb_boundary {
    int coefficient[QB_Z_COEFFICIENTS];
    double maximum_root_error;
};

struct qb_execution {
    struct qb_boundary boundary;
    struct qb_stats stats;
    double restoration_max_abs;
    double carrier_integrity_max_abs;
    int actual_inverse;
    int snapshot_loaded;
    int altered_forward;
};

static void qb_fail(const char *message) {
    fprintf(stderr, "quadratic ANF phase error: %s\n", message);
    exit(2);
}

static uint64_t qb_hash_bytes(
    uint64_t hash,
    const unsigned char *bytes,
    size_t count
) {
    for (size_t index = 0U; index < count; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t qb_plan_hash(void) {
    static const unsigned char plan[] =
        "QANF1|ENCODE:F,G,J|"
        "H:G.v*F.u,G.1*F.u,G.uc*F.1,G.uc*F.ab|"
        "Z:J.d*H.v,J.1*H.v,J.ve*H.1,J.ve*H.c,J.ve*H.abc|"
        "BOUNDARY:Z|INVERSE:BOUNDARY,Z,H,J,G,F";
    return qb_hash_bytes(
        UINT64_C(14695981039346656037),
        plan,
        sizeof(plan) - 1U
    );
}

static char *qb_trim(char *text) {
    while (*text == ' ' || *text == '\t') {
        ++text;
    }
    size_t length = strlen(text);
    while (
        length > 0U
        && (text[length - 1U] == ' ' || text[length - 1U] == '\t')
    ) {
        text[--length] = '\0';
    }
    return text;
}

static size_t qb_split(
    char *line,
    char *token[],
    size_t capacity
) {
    size_t count = 0U;
    char *save = NULL;
    for (
        char *item = strtok_r(line, " \t", &save);
        item != NULL;
        item = strtok_r(NULL, " \t", &save)
    ) {
        if (count == capacity) {
            qb_fail("too many tokens");
        }
        token[count++] = item;
    }
    return count;
}

static unsigned char qb_parse_bit(const char *text) {
    if (strcmp(text, "0") == 0) {
        return 0U;
    }
    if (strcmp(text, "1") == 0) {
        return 1U;
    }
    qb_fail("coefficient is not a canonical GF2 bit");
    return 0U;
}

static struct qb_program qb_read_program(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        qb_fail("cannot open public program");
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        fclose(file);
        qb_fail("cannot seek public program");
    }
    const long measured = ftell(file);
    if (measured < 0L || (unsigned long)measured > QB_MAX_SOURCE_BYTES) {
        fclose(file);
        qb_fail("public program size is invalid");
    }
    if (fseek(file, 0L, SEEK_SET) != 0) {
        fclose(file);
        qb_fail("cannot rewind public program");
    }
    const size_t length = (size_t)measured;
    char *source = calloc(length + 1U, 1U);
    if (source == NULL) {
        fclose(file);
        qb_fail("public program allocation failed");
    }
    if (fread(source, 1U, length, file) != length || ferror(file)) {
        free(source);
        fclose(file);
        qb_fail("cannot read public program");
    }
    if (fclose(file) != 0) {
        free(source);
        qb_fail("cannot close public program");
    }
    for (size_t index = 0U; index < length; ++index) {
        if (source[index] == '\0' || source[index] == '\r') {
            free(source);
            qb_fail("public program contains a forbidden byte");
        }
    }

    struct qb_program program;
    memset(&program, 0, sizeof(program));
    program.source_fnv1a64 = qb_hash_bytes(
        UINT64_C(14695981039346656037),
        (const unsigned char *)source,
        length
    );

    size_t state = 0U;
    char *line_save = NULL;
    for (
        char *raw = strtok_r(source, "\n", &line_save);
        raw != NULL;
        raw = strtok_r(NULL, "\n", &line_save)
    ) {
        char *line = qb_trim(raw);
        if (*line == '\0' || *line == '#') {
            continue;
        }
        char *token[5];
        const size_t count = qb_split(line, token, 5U);
        if (
            state == 0U
            && count == 2U
            && strcmp(token[0], "CATCAS_QUADRATIC_ANF_CHAIN") == 0
            && strcmp(token[1], "1") == 0
        ) {
            ++state;
            continue;
        }
        if (
            state == 1U
            && count == 2U
            && strcmp(token[0], "TYPE") == 0
            && strcmp(token[1], "BOOLEAN_ANF_GF2") == 0
        ) {
            ++state;
            continue;
        }
        if (state >= 2U && state <= 4U && count == 4U) {
            static const char *const name[QB_INPUT_RELATIONS] = {
                "F", "G", "J"
            };
            const size_t relation = state - 2U;
            if (strcmp(token[0], name[relation]) != 0) {
                free(source);
                qb_fail("public relation order is invalid");
            }
            for (
                size_t coefficient = 0U;
                coefficient < QB_INPUT_COEFFICIENTS;
                ++coefficient
            ) {
                program.coefficient[relation][coefficient] =
                    qb_parse_bit(token[coefficient + 1U]);
            }
            if (program.coefficient[relation][QB_MONIC] != 1U) {
                free(source);
                qb_fail("public relation is not a monic definition");
            }
            ++state;
            continue;
        }
        if (
            state == 5U
            && count == 1U
            && strcmp(token[0], "END") == 0
        ) {
            ++state;
            continue;
        }
        free(source);
        qb_fail("public program record is invalid");
    }
    free(source);
    if (state != 6U) {
        qb_fail("public program is incomplete");
    }
    return program;
}

static double complex qb_unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        qb_fail("nonfinite phase state");
    }
    return value / magnitude;
}

static double complex qb_lock(double complex value) {
    double complex locked = qb_unit(value);
    for (size_t pass = 0U; pass < QB_PHASE_LOCK_PASSES; ++pass) {
        locked = qb_unit(2.0 * locked + conj(locked) * conj(locked));
    }
    return locked;
}

static double complex qb_root3(unsigned int value) {
    if (value == 0U) {
        return 1.0 + 0.0 * I;
    }
    if (value == 1U) {
        return -0.5 + 0.86602540378443864676 * I;
    }
    return -0.5 - 0.86602540378443864676 * I;
}

static double complex qb_symbol_product(
    double complex left,
    double complex right,
    struct qb_stats *stats
) {
    ++stats->phase_ands;
    const double complex left_squared = conj(left);
    const double complex right_squared = conj(right);
    const double complex product = left * right;
    return qb_lock(
        (
            1.0
            + left
            + left_squared
            + right
            + right_squared
            + qb_root3(2U) * (product + conj(product))
            + qb_root3(1U) * (
                left * right_squared
                + left_squared * right
            )
        ) / 3.0
    );
}

static struct qb_carrier qb_make_carrier(int identity) {
    struct qb_carrier carrier;
    for (size_t cell = 0U; cell < QB_CARRIER_CELLS; ++cell) {
        const double angle =
            0.193
            + 0.071 * (double)cell
            + 0.013 * sin(0.31 * (double)cell + 0.017 * (double)identity);
        carrier.baseline[cell] = cexp(I * angle);
        carrier.working[cell] = carrier.baseline[cell];
    }
    return carrier;
}

static double complex qb_relative(
    const struct qb_carrier *carrier,
    size_t cell,
    struct qb_stats *stats
) {
    ++stats->carrier_reads;
    return carrier->working[cell] * conj(carrier->baseline[cell]);
}

static void qb_multiply_cell(
    struct qb_carrier *carrier,
    size_t cell,
    double complex factor,
    struct qb_stats *stats
) {
    const double complex resident = qb_relative(carrier, cell, stats);
    carrier->working[cell] =
        qb_lock(resident * factor) * carrier->baseline[cell];
    ++stats->phase_cell_updates;
}

static void qb_apply_encoding(
    struct qb_carrier *carrier,
    const struct qb_program *program,
    int inverse,
    struct qb_stats *stats
) {
    static const size_t start[QB_INPUT_RELATIONS] = {
        QB_F_START, QB_G_START, QB_J_START
    };
    if (!inverse) {
        for (size_t relation = 0U; relation < QB_INPUT_RELATIONS; ++relation) {
            for (
                size_t coefficient = 0U;
                coefficient < QB_INPUT_COEFFICIENTS;
                ++coefficient
            ) {
                qb_multiply_cell(
                    carrier,
                    start[relation] + coefficient,
                    qb_root3(program->coefficient[relation][coefficient]),
                    stats
                );
            }
        }
    } else {
        for (
            size_t relation = QB_INPUT_RELATIONS;
            relation > 0U;
            --relation
        ) {
            for (
                size_t coefficient = QB_INPUT_COEFFICIENTS;
                coefficient > 0U;
                --coefficient
            ) {
                qb_multiply_cell(
                    carrier,
                    start[relation - 1U] + coefficient - 1U,
                    conj(qb_root3(
                        program->coefficient[relation - 1U][coefficient - 1U]
                    )),
                    stats
                );
            }
        }
    }
}

static void qb_apply_h(
    struct qb_carrier *carrier,
    int inverse,
    struct qb_stats *stats
) {
    static const size_t left[QB_H_COEFFICIENTS] = {
        QB_G_START + QB_MONIC,
        QB_G_START + QB_CONSTANT,
        QB_G_START + QB_NONLINEAR,
        QB_G_START + QB_NONLINEAR
    };
    static const size_t right[QB_H_COEFFICIENTS] = {
        QB_F_START + QB_MONIC,
        QB_F_START + QB_MONIC,
        QB_F_START + QB_CONSTANT,
        QB_F_START + QB_NONLINEAR
    };
    for (size_t coefficient = 0U; coefficient < QB_H_COEFFICIENTS; ++coefficient) {
        const double complex factor = qb_symbol_product(
            qb_relative(carrier, left[coefficient], stats),
            qb_relative(carrier, right[coefficient], stats),
            stats
        );
        qb_multiply_cell(
            carrier,
            QB_H_START + coefficient,
            inverse ? conj(factor) : factor,
            stats
        );
    }
}

static size_t qb_z_right_source(size_t coefficient, int altered) {
    static const size_t normal[QB_Z_COEFFICIENTS] = {
        QB_H_START + QB_H_V,
        QB_H_START + QB_H_V,
        QB_H_START + QB_H_ONE,
        QB_H_START + QB_H_C,
        QB_H_START + QB_H_ABC
    };
    static const size_t changed[QB_Z_COEFFICIENTS] = {
        QB_H_START + QB_H_V,
        QB_H_START + QB_H_V,
        QB_H_START + QB_H_C,
        QB_H_START + QB_H_ABC,
        QB_H_START + QB_H_ONE
    };
    return altered ? changed[coefficient] : normal[coefficient];
}

static void qb_apply_z(
    struct qb_carrier *carrier,
    int inverse,
    int altered,
    int wrong_inverse,
    struct qb_stats *stats
) {
    static const size_t left[QB_Z_COEFFICIENTS] = {
        QB_J_START + QB_MONIC,
        QB_J_START + QB_CONSTANT,
        QB_J_START + QB_NONLINEAR,
        QB_J_START + QB_NONLINEAR,
        QB_J_START + QB_NONLINEAR
    };
    for (size_t coefficient = 0U; coefficient < QB_Z_COEFFICIENTS; ++coefficient) {
        const double complex factor = qb_symbol_product(
            qb_relative(carrier, left[coefficient], stats),
            qb_relative(
                carrier,
                qb_z_right_source(coefficient, altered),
                stats
            ),
            stats
        );
        qb_multiply_cell(
            carrier,
            QB_Z_START + coefficient,
            inverse && !(wrong_inverse && coefficient == QB_Z_D)
                ? conj(factor)
                : factor,
            stats
        );
    }
}

static void qb_copy_boundary(
    struct qb_carrier *carrier,
    int inverse,
    struct qb_stats *stats
) {
    for (size_t coefficient = 0U; coefficient < QB_Z_COEFFICIENTS; ++coefficient) {
        const double complex factor = qb_relative(
            carrier,
            QB_Z_START + coefficient,
            stats
        );
        qb_multiply_cell(
            carrier,
            QB_BOUNDARY_START + coefficient,
            inverse ? conj(factor) : factor,
            stats
        );
    }
    ++stats->boundary_copies;
}

static int qb_decode_root(
    double complex value,
    double *error_out
) {
    int best = 0;
    double best_error = INFINITY;
    for (int symbol = 0; symbol < 3; ++symbol) {
        const double error = cabs(value - qb_root3((unsigned int)symbol));
        if (error < best_error) {
            best = symbol;
            best_error = error;
        }
    }
    *error_out = best_error;
    return best;
}

static struct qb_boundary qb_latch_boundary(
    const struct qb_carrier *carrier,
    struct qb_stats *stats
) {
    struct qb_boundary boundary;
    memset(&boundary, 0, sizeof(boundary));
    for (size_t coefficient = 0U; coefficient < QB_Z_COEFFICIENTS; ++coefficient) {
        double error = 0.0;
        boundary.coefficient[coefficient] = qb_decode_root(
            qb_relative(carrier, QB_BOUNDARY_START + coefficient, stats),
            &error
        );
        if (boundary.coefficient[coefficient] > 1) {
            qb_fail("final boundary coefficient left the Boolean subset");
        }
        if (error > boundary.maximum_root_error) {
            boundary.maximum_root_error = error;
        }
        ++stats->boundary_decodes;
    }
    return boundary;
}

static double qb_restoration_error(
    const struct qb_carrier *carrier,
    const struct qb_carrier *snapshot
) {
    double maximum = 0.0;
    for (size_t cell = 0U; cell < QB_CARRIER_CELLS; ++cell) {
        const double error = cabs(
            carrier->working[cell] - snapshot->working[cell]
        );
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static double qb_integrity_error(const struct qb_carrier *carrier) {
    double maximum = 0.0;
    for (size_t cell = 0U; cell < QB_CARRIER_CELLS; ++cell) {
        const double error = fabs(cabs(carrier->working[cell]) - 1.0);
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static struct qb_execution qb_execute(
    struct qb_carrier *carrier,
    const struct qb_program *program,
    enum qb_inverse_mode mode
) {
    const struct qb_carrier snapshot = *carrier;
    struct qb_execution execution;
    memset(&execution, 0, sizeof(execution));

    qb_apply_encoding(carrier, program, 0, &execution.stats);
    qb_apply_h(carrier, 0, &execution.stats);
    qb_apply_z(
        carrier,
        0,
        mode == QB_INVERSE_ALTERED_FORWARD,
        0,
        &execution.stats
    );
    qb_copy_boundary(carrier, 0, &execution.stats);
    execution.boundary = qb_latch_boundary(carrier, &execution.stats);

    if (mode == QB_INVERSE_SNAPSHOT) {
        memcpy(
            carrier->working,
            snapshot.working,
            sizeof(carrier->working)
        );
        ++execution.stats.snapshot_loads;
        execution.snapshot_loaded = 1;
    } else {
        qb_copy_boundary(carrier, 1, &execution.stats);
        if (mode == QB_INVERSE_REORDERED) {
            qb_apply_h(carrier, 1, &execution.stats);
            qb_apply_z(carrier, 1, 0, 0, &execution.stats);
        } else {
            if (mode != QB_INVERSE_MISSING_Z) {
                qb_apply_z(
                    carrier,
                    1,
                    mode == QB_INVERSE_ALTERED_FORWARD,
                    mode == QB_INVERSE_WRONG_Z,
                    &execution.stats
                );
            }
            qb_apply_h(carrier, 1, &execution.stats);
        }
        qb_apply_encoding(carrier, program, 1, &execution.stats);
        execution.actual_inverse = mode == QB_INVERSE_CORRECT;
        execution.altered_forward = mode == QB_INVERSE_ALTERED_FORWARD;
    }

    execution.restoration_max_abs =
        qb_restoration_error(carrier, &snapshot);
    execution.carrier_integrity_max_abs = qb_integrity_error(carrier);
    return execution;
}

static int qb_boundary_differs(
    const struct qb_boundary *left,
    const struct qb_boundary *right
) {
    for (size_t coefficient = 0U; coefficient < QB_Z_COEFFICIENTS; ++coefficient) {
        if (left->coefficient[coefficient] != right->coefficient[coefficient]) {
            return 1;
        }
    }
    return 0;
}

static int qb_stats_equal(
    const struct qb_stats *left,
    const struct qb_stats *right
) {
    return
        left->phase_ands == right->phase_ands
        && left->carrier_reads == right->carrier_reads
        && left->phase_cell_updates == right->phase_cell_updates
        && left->boundary_decodes == right->boundary_decodes
        && left->intermediate_decodes == right->intermediate_decodes
        && left->intermediate_copies == right->intermediate_copies
        && left->boundary_copies == right->boundary_copies
        && left->snapshot_loads == right->snapshot_loads;
}

static void qb_print_boundary(const struct qb_boundary *boundary) {
    putchar('[');
    for (size_t coefficient = 0U; coefficient < QB_Z_COEFFICIENTS; ++coefficient) {
        if (coefficient != 0U) {
            putchar(',');
        }
        printf("%d", boundary->coefficient[coefficient]);
    }
    putchar(']');
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--project-intermediate") == 0) {
        qb_fail("only the final Z signature may be projected");
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        qb_fail("carrier is required");
    }
    if (argc != 3) {
        qb_fail("usage: phase PRIMARY.qanf REUSE.qanf");
    }

    const struct qb_program primary_program = qb_read_program(argv[1]);
    const struct qb_program reuse_program = qb_read_program(argv[2]);
    struct qb_carrier carrier = qb_make_carrier(41);

    const struct qb_execution primary = qb_execute(
        &carrier,
        &primary_program,
        QB_INVERSE_CORRECT
    );
    const struct qb_execution reuse = qb_execute(
        &carrier,
        &reuse_program,
        QB_INVERSE_CORRECT
    );
    if (
        primary.restoration_max_abs > QB_RESTORATION_TOLERANCE
        || reuse.restoration_max_abs > QB_RESTORATION_TOLERANCE
        || primary.boundary.maximum_root_error > QB_ROOT_TOLERANCE
        || reuse.boundary.maximum_root_error > QB_ROOT_TOLERANCE
        || !qb_stats_equal(&primary.stats, &reuse.stats)
    ) {
        qb_fail("accepted transaction law failed");
    }

    double maximum_reuse_error =
        fmax(primary.restoration_max_abs, reuse.restoration_max_abs);
    for (size_t cycle = 0U; cycle < QB_REPEAT_CYCLES; ++cycle) {
        const struct qb_program *program =
            (cycle & 1U) == 0U ? &primary_program : &reuse_program;
        const struct qb_execution repeated = qb_execute(
            &carrier,
            program,
            QB_INVERSE_CORRECT
        );
        if (
            repeated.restoration_max_abs > QB_RESTORATION_TOLERANCE
            || !qb_stats_equal(&primary.stats, &repeated.stats)
        ) {
            qb_fail("repeated restored-carrier reuse failed");
        }
        maximum_reuse_error = fmax(
            maximum_reuse_error,
            repeated.restoration_max_abs
        );
    }

    struct qb_carrier wrong_carrier = qb_make_carrier(43);
    struct qb_carrier missing_carrier = qb_make_carrier(47);
    struct qb_carrier reordered_carrier = qb_make_carrier(53);
    struct qb_carrier snapshot_carrier = qb_make_carrier(59);
    struct qb_carrier altered_carrier = qb_make_carrier(61);
    const struct qb_execution wrong = qb_execute(
        &wrong_carrier,
        &primary_program,
        QB_INVERSE_WRONG_Z
    );
    const struct qb_execution missing = qb_execute(
        &missing_carrier,
        &primary_program,
        QB_INVERSE_MISSING_Z
    );
    const struct qb_execution reordered = qb_execute(
        &reordered_carrier,
        &primary_program,
        QB_INVERSE_REORDERED
    );
    const struct qb_execution snapshot = qb_execute(
        &snapshot_carrier,
        &primary_program,
        QB_INVERSE_SNAPSHOT
    );
    const struct qb_execution altered = qb_execute(
        &altered_carrier,
        &primary_program,
        QB_INVERSE_ALTERED_FORWARD
    );
    if (
        wrong.restoration_max_abs < QB_CONTROL_MINIMUM_ERROR
        || missing.restoration_max_abs < QB_CONTROL_MINIMUM_ERROR
        || reordered.restoration_max_abs < QB_CONTROL_MINIMUM_ERROR
        || snapshot.restoration_max_abs > QB_RESTORATION_TOLERANCE
        || snapshot.stats.snapshot_loads != 1U
        || !qb_boundary_differs(&primary.boundary, &altered.boundary)
        || altered.restoration_max_abs > QB_RESTORATION_TOLERANCE
    ) {
        qb_fail("mechanism control failed");
    }

    struct qb_program cut_program = primary_program;
    const int cut_applicable =
        cut_program.coefficient[0][QB_NONLINEAR] == 1U;
    struct qb_carrier cut_carrier = qb_make_carrier(67);
    cut_program.coefficient[0][QB_NONLINEAR] = 0U;
    const struct qb_execution cut = qb_execute(
        &cut_carrier,
        &cut_program,
        QB_INVERSE_CORRECT
    );
    if (
        cut.restoration_max_abs > QB_RESTORATION_TOLERANCE
        || (
            cut_applicable
            && !qb_boundary_differs(&primary.boundary, &cut.boundary)
        )
    ) {
        qb_fail("quadratic-term cut control failed");
    }

    printf(
        "{\"result\":\"PASS\","
        "\"schema\":\"CATCAS_QUADRATIC_ANF_CHAIN_1\","
        "\"plan_fnv1a64\":\"%016llx\","
        "\"primary_source_fnv1a64\":\"%016llx\","
        "\"reuse_source_fnv1a64\":\"%016llx\","
        "\"primary_boundary\":",
        (unsigned long long)qb_plan_hash(),
        (unsigned long long)primary_program.source_fnv1a64,
        (unsigned long long)reuse_program.source_fnv1a64
    );
    qb_print_boundary(&primary.boundary);
    printf(",\"reuse_boundary\":");
    qb_print_boundary(&reuse.boundary);
    printf(",\"altered_hidden_source_boundary\":");
    qb_print_boundary(&altered.boundary);
    printf(",\"quadratic_cut_boundary\":");
    qb_print_boundary(&cut.boundary);
    printf(
        ",\"primary_highest_degree_coefficient\":%d,"
        "\"reuse_highest_degree_coefficient\":%d,"
        "\"carrier_cells\":%u,"
        "\"live_carrier_bytes\":%zu,"
        "\"comparison_snapshot_bytes\":%zu,"
        "\"input_signature_cells\":%u,"
        "\"resident_intermediate_cells\":%u,"
        "\"resident_final_cells\":%u,"
        "\"public_boundary_cells\":%u,"
        "\"dense_five_variable_membership_entries\":32,"
        "\"same_carrier_transactions\":%u,"
        "\"carrier_creations_on_accepted_path\":1,"
        "\"restoration_tolerance\":%.12g,"
        "\"root_tolerance\":%.12g,"
        "\"primary_restoration_max_abs\":%.12g,"
        "\"reuse_restoration_max_abs\":%.12g,"
        "\"repeated_reuse_max_abs\":%.12g,"
        "\"primary_root_max_abs\":%.12g,"
        "\"phase_ands_per_transaction\":%llu,"
        "\"carrier_reads_per_transaction\":%llu,"
        "\"phase_cell_updates_per_transaction\":%llu,"
        "\"boundary_decodes_per_transaction\":%llu,"
        "\"intermediate_decodes\":%llu,"
        "\"intermediate_copies\":%llu,"
        "\"boundary_copies_per_transaction\":%llu,"
        "\"wrong_inverse_restoration_max_abs\":%.12g,"
        "\"missing_inverse_restoration_max_abs\":%.12g,"
        "\"reordered_inverse_restoration_max_abs\":%.12g,"
        "\"snapshot_restoration_max_abs\":%.12g,"
        "\"snapshot_inverse_executed\":false,"
        "\"altered_hidden_source_restores\":true,"
        "\"quadratic_cut_applicable\":%s,"
        "\"quadratic_cut_restores\":true,"
        "\"actual_inverse\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"final_boundary_only\":true,"
        "\"full_boundary_anf_signature\":true,"
        "\"terminal\":false}\n",
        primary.boundary.coefficient[QB_Z_ABCE],
        reuse.boundary.coefficient[QB_Z_ABCE],
        QB_CARRIER_CELLS,
        sizeof(carrier),
        sizeof(struct qb_carrier),
        QB_INPUT_RELATIONS * QB_INPUT_COEFFICIENTS,
        QB_H_COEFFICIENTS,
        QB_Z_COEFFICIENTS,
        QB_Z_COEFFICIENTS,
        2U + QB_REPEAT_CYCLES,
        QB_RESTORATION_TOLERANCE,
        QB_ROOT_TOLERANCE,
        primary.restoration_max_abs,
        reuse.restoration_max_abs,
        maximum_reuse_error,
        primary.boundary.maximum_root_error,
        (unsigned long long)primary.stats.phase_ands,
        (unsigned long long)primary.stats.carrier_reads,
        (unsigned long long)primary.stats.phase_cell_updates,
        (unsigned long long)primary.stats.boundary_decodes,
        (unsigned long long)primary.stats.intermediate_decodes,
        (unsigned long long)primary.stats.intermediate_copies,
        (unsigned long long)primary.stats.boundary_copies,
        wrong.restoration_max_abs,
        missing.restoration_max_abs,
        reordered.restoration_max_abs,
        snapshot.restoration_max_abs,
        cut_applicable ? "true" : "false"
    );
    return 0;
}
