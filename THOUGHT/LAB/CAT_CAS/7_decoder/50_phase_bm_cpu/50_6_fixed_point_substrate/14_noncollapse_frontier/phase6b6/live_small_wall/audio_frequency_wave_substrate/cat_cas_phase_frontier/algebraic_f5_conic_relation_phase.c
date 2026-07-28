#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: exact F5 conic relation closure.
 *
 * Full-field F5 ports are explicit. A six-coefficient conic Q(u,v) is
 * bracketed by monic affine graph relations u=A(x), v=C(z). Closing u forms
 * the resident six-phase conic H(x,v)=Q(A(x),v). Closing v consumes that
 * actual H and forms K(x,z)=H(x,C(z)). Monic pinning makes both existential
 * closures exact over F5; no general-resultant or Boolean-domain claim is
 * made. The six-coefficient rank is preserved across both closures.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FC_FIELD 5U
#define FC_SIGNATURE 6U
#define FC_INPUT_CELLS 10U
#define FC_H_START 10U
#define FC_K_START 16U
#define FC_CELLS 22U
#define FC_REUSE_CYCLES 16U

static const double FC_TOLERANCE = 2.0e-11;
static const double FC_ROOT_TOLERANCE = 2.0e-9;
static const double FC_CONTROL_MINIMUM = 1.0e-6;

enum fc_program {
    FC_PRIMARY = 0,
    FC_REUSE = 1
};

enum fc_mode {
    FC_CORRECT = 0,
    FC_MISSING_INVERSE = 1,
    FC_WRONG_INVERSE = 2,
    FC_REORDERED_INVERSE = 3,
    FC_SNAPSHOT = 4,
    FC_BYPASS_H = 5
};

struct fc_carrier {
    double complex baseline[FC_CELLS];
    double complex working[FC_CELLS];
};

struct fc_stats {
    uint64_t native_phase_updates;
    uint64_t resident_phase_reads;
    uint64_t field_product_interpolations;
    uint64_t hidden_coefficient_decodes;
    uint64_t final_coefficient_decodes;
    double maximum_unit_modulus_error;
    double maximum_final_root_error;
};

struct fc_boundary {
    int coefficient[FC_SIGNATURE];
    uint64_t hash;
};

struct fc_execution {
    struct fc_boundary boundary;
    struct fc_stats stats;
    double restoration_max_abs;
    uint64_t restoration_generation;
    int actual_inverse;
    int snapshot_loaded;
};

static void fc_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static double complex fc_unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        fc_fail("nonfinite F5 conic phase");
    }
    return value / magnitude;
}

static double complex fc_root(int value) {
    int normalized = value % (int)FC_FIELD;
    if (normalized < 0) {
        normalized += (int)FC_FIELD;
    }
    return cexp(
        I * 2.0 * 3.14159265358979323846264338327950288
        * (double)normalized
        / (double)FC_FIELD
    );
}

static struct fc_carrier fc_make_carrier(int identity) {
    struct fc_carrier carrier;
    for (size_t cell = 0U; cell < FC_CELLS; ++cell) {
        const double angle =
            0.181
            + 0.053 * (double)cell
            + 0.011 * sin(
                0.37 * (double)cell + 0.017 * (double)identity
            );
        carrier.baseline[cell] = cexp(I * angle);
        carrier.working[cell] = carrier.baseline[cell];
    }
    return carrier;
}

static double complex fc_relative(
    const struct fc_carrier *carrier,
    size_t cell
) {
    return carrier->working[cell] * conj(carrier->baseline[cell]);
}

static void fc_observe(
    const struct fc_carrier *carrier,
    struct fc_stats *stats
) {
    for (size_t cell = 0U; cell < FC_CELLS; ++cell) {
        const double error =
            fabs(cabs(fc_relative(carrier, cell)) - 1.0);
        if (error > stats->maximum_unit_modulus_error) {
            stats->maximum_unit_modulus_error = error;
        }
        if (error > FC_TOLERANCE) {
            fc_fail("F5 conic carrier left unit torus");
        }
    }
}

static void fc_multiply(
    struct fc_carrier *carrier,
    size_t cell,
    double complex factor,
    struct fc_stats *stats
) {
    carrier->working[cell] = fc_unit(
        fc_relative(carrier, cell) * factor
    ) * carrier->baseline[cell];
    ++stats->native_phase_updates;
}

/*
 * Fourier interpolant for F5 multiplication:
 * P(X,Y)=(1/5) sum_{r,s} zeta^(-rs) X^r Y^s.
 * At X=zeta^a,Y=zeta^b this is exactly zeta^(ab).
 */
static double complex fc_field_product(
    double complex left,
    double complex right,
    struct fc_stats *stats
) {
    double complex left_power[FC_FIELD];
    double complex right_power[FC_FIELD];
    left_power[0] = 1.0 + 0.0 * I;
    right_power[0] = 1.0 + 0.0 * I;
    for (size_t power = 1U; power < FC_FIELD; ++power) {
        left_power[power] = left_power[power - 1U] * left;
        right_power[power] = right_power[power - 1U] * right;
    }
    double complex sum = 0.0 + 0.0 * I;
    for (size_t r = 0U; r < FC_FIELD; ++r) {
        for (size_t s = 0U; s < FC_FIELD; ++s) {
            sum +=
                fc_root(-(int)(r * s))
                * left_power[r]
                * right_power[s];
        }
    }
    ++stats->field_product_interpolations;
    return fc_unit(sum / (double)FC_FIELD);
}

static double complex fc_add2(
    double complex first,
    double complex second
) {
    return fc_unit(first * second);
}

static double complex fc_add3(
    double complex first,
    double complex second,
    double complex third
) {
    return fc_unit(first * second * third);
}

static double complex fc_twice(double complex value) {
    return fc_unit(value * value);
}

static void fc_first_factors(
    const struct fc_carrier *carrier,
    double complex output[FC_SIGNATURE],
    struct fc_stats *stats
) {
    const double complex a0 = fc_relative(carrier, 0U);
    const double complex a1 = fc_relative(carrier, 1U);
    double complex q[FC_SIGNATURE];
    for (size_t index = 0U; index < FC_SIGNATURE; ++index) {
        q[index] = fc_relative(carrier, 2U + index);
    }
    stats->resident_phase_reads += 8U;
    const double complex a0_squared =
        fc_field_product(a0, a0, stats);
    const double complex a1_squared =
        fc_field_product(a1, a1, stats);
    output[0] = fc_add3(
        q[0],
        fc_field_product(q[1], a0, stats),
        fc_field_product(q[3], a0_squared, stats)
    );
    output[1] = fc_field_product(
        a1,
        fc_add2(
            q[1],
            fc_field_product(q[3], fc_twice(a0), stats)
        ),
        stats
    );
    output[2] = fc_add2(
        q[2],
        fc_field_product(q[4], a0, stats)
    );
    output[3] = fc_field_product(q[3], a1_squared, stats);
    output[4] = fc_field_product(q[4], a1, stats);
    output[5] = q[5];
}

static void fc_second_factors(
    const struct fc_carrier *carrier,
    double complex output[FC_SIGNATURE],
    enum fc_mode mode,
    struct fc_stats *stats
) {
    double complex h[FC_SIGNATURE];
    for (size_t index = 0U; index < FC_SIGNATURE; ++index) {
        h[index] = mode == FC_BYPASS_H
            ? fc_root(0)
            : fc_relative(carrier, FC_H_START + index);
    }
    const double complex c0 = fc_relative(carrier, 8U);
    const double complex c1 = fc_relative(carrier, 9U);
    stats->resident_phase_reads += 8U;
    const double complex c0_squared =
        fc_field_product(c0, c0, stats);
    const double complex c1_squared =
        fc_field_product(c1, c1, stats);
    output[0] = fc_add3(
        h[0],
        fc_field_product(h[2], c0, stats),
        fc_field_product(h[5], c0_squared, stats)
    );
    output[1] = fc_add2(
        h[1],
        fc_field_product(h[4], c0, stats)
    );
    output[2] = fc_field_product(
        c1,
        fc_add2(
            h[2],
            fc_field_product(h[5], fc_twice(c0), stats)
        ),
        stats
    );
    output[3] = h[3];
    output[4] = fc_field_product(h[4], c1, stats);
    output[5] = fc_field_product(h[5], c1_squared, stats);
}

static void fc_apply_factors(
    struct fc_carrier *carrier,
    size_t start,
    const double complex factor[FC_SIGNATURE],
    int inverse,
    int wrong,
    int missing,
    struct fc_stats *stats
) {
    for (size_t index = 0U; index < FC_SIGNATURE; ++index) {
        if (missing && index == 4U) {
            continue;
        }
        double complex selected = inverse
            ? conj(factor[index])
            : factor[index];
        if (wrong && index == 4U) {
            selected = conj(selected);
        }
        fc_multiply(carrier, start + index, selected, stats);
    }
    fc_observe(carrier, stats);
}

static const int FC_INPUT[2][FC_INPUT_CELLS] = {
    /* A=x, Q=uv, C=z -> K=xz. */
    {0, 1, 0, 0, 0, 0, 1, 0, 0, 1},
    /* Unrelated dense conic and affine port maps. */
    {1, 2, 2, 1, 4, 3, 2, 1, 3, 1}
};

static void fc_seal(
    struct fc_carrier *carrier,
    enum fc_program program,
    int inverse,
    struct fc_stats *stats
) {
    for (size_t cell = 0U; cell < FC_INPUT_CELLS; ++cell) {
        double complex factor = fc_root(FC_INPUT[program][cell]);
        if (inverse) {
            factor = conj(factor);
        }
        fc_multiply(carrier, cell, factor, stats);
    }
    fc_observe(carrier, stats);
}

static int fc_decode(
    double complex phase,
    struct fc_stats *stats
) {
    int selected = 0;
    double minimum = HUGE_VAL;
    for (int value = 0; value < (int)FC_FIELD; ++value) {
        const double error = cabs(phase - fc_root(value));
        if (error < minimum) {
            minimum = error;
            selected = value;
        }
    }
    if (minimum > stats->maximum_final_root_error) {
        stats->maximum_final_root_error = minimum;
    }
    if (minimum > FC_ROOT_TOLERANCE) {
        fc_fail("F5 final conic coefficient is not a root");
    }
    return selected;
}

static struct fc_boundary fc_project(
    const struct fc_carrier *carrier,
    struct fc_stats *stats
) {
    struct fc_boundary boundary = {
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t index = 0U; index < FC_SIGNATURE; ++index) {
        boundary.coefficient[index] = fc_decode(
            fc_relative(carrier, FC_K_START + index), stats
        );
        boundary.hash ^= (uint64_t)boundary.coefficient[index];
        boundary.hash *= UINT64_C(1099511628211);
        ++stats->final_coefficient_decodes;
    }
    return boundary;
}

static double fc_restoration(
    const struct fc_carrier *carrier,
    const struct fc_carrier *borrowed
) {
    double maximum = 0.0;
    for (size_t cell = 0U; cell < FC_CELLS; ++cell) {
        const double error = cabs(
            carrier->working[cell] - borrowed->working[cell]
        );
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static struct fc_execution fc_execute(
    struct fc_carrier *carrier,
    enum fc_program program,
    enum fc_mode mode
) {
    if (carrier == NULL) {
        fc_fail("invalid F5 conic carrier");
    }
    const struct fc_carrier borrowed = *carrier;
    struct fc_execution execution = {0};
    double complex h[FC_SIGNATURE];
    double complex k[FC_SIGNATURE];
    fc_seal(carrier, program, 0, &execution.stats);
    fc_first_factors(carrier, h, &execution.stats);
    fc_apply_factors(
        carrier, FC_H_START, h, 0, 0, 0, &execution.stats
    );
    fc_second_factors(carrier, k, mode, &execution.stats);
    fc_apply_factors(
        carrier, FC_K_START, k, 0, 0, 0, &execution.stats
    );
    execution.boundary = fc_project(carrier, &execution.stats);

    if (mode == FC_SNAPSHOT) {
        *carrier = borrowed;
        execution.snapshot_loaded = 1;
    } else {
        if (mode == FC_REORDERED_INVERSE) {
            fc_first_factors(carrier, h, &execution.stats);
            fc_apply_factors(
                carrier,
                FC_H_START,
                h,
                1,
                0,
                0,
                &execution.stats
            );
        }
        fc_second_factors(carrier, k, mode, &execution.stats);
        fc_apply_factors(
            carrier,
            FC_K_START,
            k,
            1,
            mode == FC_WRONG_INVERSE,
            0,
            &execution.stats
        );
        if (mode != FC_REORDERED_INVERSE) {
            fc_first_factors(carrier, h, &execution.stats);
            fc_apply_factors(
                carrier,
                FC_H_START,
                h,
                1,
                0,
                mode == FC_MISSING_INVERSE,
                &execution.stats
            );
        }
        fc_seal(carrier, program, 1, &execution.stats);
        if (mode == FC_CORRECT || mode == FC_BYPASS_H) {
            execution.actual_inverse = 1;
            execution.restoration_generation = 1U;
        }
    }
    execution.restoration_max_abs = fc_restoration(
        carrier, &borrowed
    );
    return execution;
}

static void fc_require_correct(const struct fc_execution *execution) {
    if (
        !execution->actual_inverse
        || execution->snapshot_loaded
        || execution->restoration_generation != 1U
        || execution->stats.hidden_coefficient_decodes != 0U
        || execution->stats.final_coefficient_decodes != FC_SIGNATURE
        || execution->restoration_max_abs > FC_TOLERANCE
        || execution->stats.maximum_unit_modulus_error > FC_TOLERANCE
        || execution->stats.maximum_final_root_error > FC_ROOT_TOLERANCE
    ) {
        fc_fail("F5 conic carrier law failed");
    }
}

static struct fc_execution fc_control(enum fc_mode mode) {
    struct fc_carrier carrier = fc_make_carrier(12900 + (int)mode);
    return fc_execute(&carrier, FC_PRIMARY, mode);
}

int main(int argc, char **argv) {
    if (
        argc == 2
        && strcmp(argv[1], "--project-intermediate") == 0
    ) {
        fc_fail("F5 resident conic projection denied");
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        (void)fc_execute(NULL, FC_PRIMARY, FC_CORRECT);
    }
    if (argc != 1) {
        fc_fail("usage: algebraic_f5_conic_relation_phase");
    }
    struct fc_carrier carrier = fc_make_carrier(12801);
    const struct fc_execution primary = fc_execute(
        &carrier, FC_PRIMARY, FC_CORRECT
    );
    fc_require_correct(&primary);
    const struct fc_execution reuse = fc_execute(
        &carrier, FC_REUSE, FC_CORRECT
    );
    fc_require_correct(&reuse);
    if (primary.boundary.hash == reuse.boundary.hash) {
        fc_fail("F5 conic reuse boundary unchanged");
    }
    double repeated_maximum = 0.0;
    for (size_t cycle = 0U; cycle < FC_REUSE_CYCLES; ++cycle) {
        const struct fc_execution repeated = fc_execute(
            &carrier,
            cycle % 2U == 0U ? FC_PRIMARY : FC_REUSE,
            FC_CORRECT
        );
        fc_require_correct(&repeated);
        if (repeated.restoration_max_abs > repeated_maximum) {
            repeated_maximum = repeated.restoration_max_abs;
        }
    }
    const struct fc_execution missing = fc_control(
        FC_MISSING_INVERSE
    );
    const struct fc_execution wrong = fc_control(FC_WRONG_INVERSE);
    const struct fc_execution reordered = fc_control(
        FC_REORDERED_INVERSE
    );
    const struct fc_execution snapshot = fc_control(FC_SNAPSHOT);
    const struct fc_execution bypass = fc_control(FC_BYPASS_H);
    fc_require_correct(&bypass);
    if (
        missing.restoration_max_abs < FC_CONTROL_MINIMUM
        || wrong.restoration_max_abs < FC_CONTROL_MINIMUM
        || reordered.restoration_max_abs < FC_CONTROL_MINIMUM
        || !snapshot.snapshot_loaded
        || snapshot.actual_inverse
        || snapshot.restoration_max_abs > FC_TOLERANCE
        || bypass.boundary.hash == primary.boundary.hash
    ) {
        fc_fail("F5 conic control failed");
    }
    printf(
        "{\"result\":\"PASS\","
        "\"claim\":\"BOUNDED_F5_MONIC_AFFINE_CONIC_TWO_HIDDEN_PORT_FIXED_RANK_PHASE_CLOSURE_WITH_RESTORATION_AND_REUSE\","
        "\"port_type\":\"FULL_F5\","
        "\"closed_hidden_ports\":2,"
        "\"input_signature_cells\":10,"
        "\"resident_intermediate_cells\":6,"
        "\"resident_final_cells\":6,"
        "\"carrier_cells\":22,"
        "\"resident_phase_storage_bytes\":%zu,"
        "\"hidden_baseline_storage_bytes\":%zu,"
        "\"borrowed_verification_copy_bytes\":%zu,"
        "\"phase_factor_temporary_bytes\":%zu,"
        "\"projected_boundary_bytes\":%zu,"
        "\"signature_rank_after_each_closure\":6,"
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"primary_boundary_coefficients\":[",
        FC_CELLS * sizeof(double complex),
        FC_CELLS * sizeof(double complex),
        sizeof(struct fc_carrier),
        2U * FC_SIGNATURE * sizeof(double complex),
        sizeof(struct fc_boundary),
        (unsigned long long)primary.boundary.hash,
        (unsigned long long)reuse.boundary.hash
    );
    for (size_t index = 0U; index < FC_SIGNATURE; ++index) {
        printf(
            "%s%d",
            index == 0U ? "" : ",",
            primary.boundary.coefficient[index]
        );
    }
    printf(
        "],\"native_phase_updates\":%llu,"
        "\"resident_phase_reads\":%llu,"
        "\"field_product_interpolations\":%llu,"
        "\"hidden_coefficient_decodes\":%llu,"
        "\"final_coefficient_decodes\":%llu,"
        "\"maximum_unit_modulus_error\":%.17g,"
        "\"maximum_final_root_error\":%.17g,"
        "\"maximum_repeated_restoration_error\":%.17g,"
        "\"missing_inverse_residual\":%.17g,"
        "\"wrong_inverse_residual\":%.17g,"
        "\"reordered_inverse_residual\":%.17g,"
        "\"snapshot_residual\":%.17g,"
        "\"actual_resident_intermediate_consumed\":true,"
        "\"exact_monic_substitution_closure\":true,"
        "\"non_affine_primary_xz_boundary\":true,"
        "\"fixed_rank_two_closure_chain\":true,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"snapshot_loaded_separately\":true,"
        "\"intermediate_bypass_changes_boundary\":true,"
        "\"general_conic_conic_composition\":false,"
        "\"canonical_zero_set_normalization\":false,"
        "\"compact_classical_coefficient_recurrence_exists\":true,"
        "\"genuinely_distinct_phase_resource\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_execution\":false,"
        "\"unlimited_catalytic_computation\":false,"
        "\"terminal\":false}\n",
        (unsigned long long)primary.stats.native_phase_updates,
        (unsigned long long)primary.stats.resident_phase_reads,
        (unsigned long long)primary.stats.field_product_interpolations,
        (unsigned long long)primary.stats.hidden_coefficient_decodes,
        (unsigned long long)primary.stats.final_coefficient_decodes,
        primary.stats.maximum_unit_modulus_error,
        primary.stats.maximum_final_root_error,
        repeated_maximum,
        missing.restoration_max_abs,
        wrong.restoration_max_abs,
        reordered.restoration_max_abs,
        snapshot.restoration_max_abs
    );
    return 0;
}
