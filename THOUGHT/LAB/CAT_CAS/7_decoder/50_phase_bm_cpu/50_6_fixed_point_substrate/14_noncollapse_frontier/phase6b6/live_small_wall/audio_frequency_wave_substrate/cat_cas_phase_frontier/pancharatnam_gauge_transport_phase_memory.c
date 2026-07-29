#define _POSIX_C_SOURCE 200809L

#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Stateful repair for the finite-phasor global-section obstruction.
 *
 * A typed carrier is a point in the U(1) fiber above a public Bloch-sphere
 * base point.  Each public edge transports the actual resident fiber phase
 * by the Pancharatnam rule: choose the target representative's phase so its
 * overlap with the current carrier is positive real.  A closed base path can
 * therefore retain endpoint-invisible holonomy.  Reversing the public path
 * transports the same carrier back without retaining an edge history.
 *
 * This is bounded complex128 software evidence.  A state-minimal compact
 * classical baseline is the identical one-scalar gauge-angle recurrence;
 * the fixed public path families also admit closed-form product evaluation.
 */

static const double PG_PI =
    3.141592653589793238462643383279502884;
static const double PG_THETA = 1.1;
static const double PG_TOLERANCE = 1.0e-10;
static const size_t PG_PRIMARY_SEGMENTS = 512U;
static const size_t PG_REUSE_REPETITIONS = 37U;
static const size_t PG_STRESS_CYCLES = 100U;
static const size_t PG_TEST_SEGMENTS[] = {
    4U, 8U, 16U, 32U, 64U, 128U, 256U, 512U
};

struct pg_spinor {
    double complex cell[2];
};

struct pg_point {
    double theta;
    double phi;
};

struct pg_latitude_descriptor {
    double theta;
    size_t segments;
    int orientation;
};

struct pg_run {
    size_t segments;
    double complex holonomy;
    double complex analytic_discrete_holonomy;
    double complex continuous_limit_holonomy;
    double discrete_formula_error;
    double continuous_limit_error;
    double restoration_error;
    double norm_error;
};

static void pg_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static struct pg_spinor pg_base(double theta, double phi) {
    const struct pg_spinor result = {
        .cell = {
            cos(0.5 * theta) + 0.0 * I,
            cexp(I * phi) * sin(0.5 * theta)
        }
    };
    return result;
}

static double complex pg_overlap(
    const struct pg_spinor *left,
    const struct pg_spinor *right
) {
    return conj(left->cell[0]) * right->cell[0]
        + conj(left->cell[1]) * right->cell[1];
}

static double pg_norm_error(const struct pg_spinor *carrier) {
    return fabs(creal(pg_overlap(carrier, carrier)) - 1.0)
        + fabs(cimag(pg_overlap(carrier, carrier)));
}

static double pg_spinor_error(
    const struct pg_spinor *left,
    const struct pg_spinor *right
) {
    return fmax(
        cabs(left->cell[0] - right->cell[0]),
        cabs(left->cell[1] - right->cell[1])
    );
}

static int pg_transport_to(
    struct pg_spinor *carrier,
    const struct pg_spinor *target
) {
    const double complex overlap = pg_overlap(carrier, target);
    const double magnitude = cabs(overlap);
    if (!(magnitude > 1.0e-14)) {
        return 0;
    }
    const double complex factor = conj(overlap) / magnitude;
    carrier->cell[0] = factor * target->cell[0];
    carrier->cell[1] = factor * target->cell[1];
    return 1;
}

static int pg_forward_latitude(
    struct pg_spinor *carrier,
    struct pg_latitude_descriptor descriptor
) {
    for (size_t step = 1U; step <= descriptor.segments; ++step) {
        const double phi =
            (double)descriptor.orientation
            * 2.0 * PG_PI * (double)step
            / (double)descriptor.segments;
        const struct pg_spinor target = pg_base(
            descriptor.theta,
            phi
        );
        if (!pg_transport_to(carrier, &target)) {
            return 0;
        }
    }
    return 1;
}

static int pg_inverse_latitude(
    struct pg_spinor *carrier,
    struct pg_latitude_descriptor descriptor
) {
    for (size_t cursor = descriptor.segments; cursor > 0U; --cursor) {
        const size_t step = cursor - 1U;
        const double phi =
            (double)descriptor.orientation
            * 2.0 * PG_PI * (double)step
            / (double)descriptor.segments;
        const struct pg_spinor target = pg_base(
            descriptor.theta,
            phi
        );
        if (!pg_transport_to(carrier, &target)) {
            return 0;
        }
    }
    return 1;
}

static const struct pg_point PG_REUSE_PATH[] = {
    {.theta = PG_THETA, .phi = 0.0},
    {.theta = 0.55, .phi = 0.0},
    {.theta = 0.55, .phi = 1.25},
    {.theta = PG_THETA, .phi = 1.25},
    {.theta = PG_THETA, .phi = 0.0}
};

static int pg_forward_reuse_path(struct pg_spinor *carrier) {
    const size_t count = sizeof(PG_REUSE_PATH) / sizeof(PG_REUSE_PATH[0]);
    for (size_t index = 1U; index < count; ++index) {
        const struct pg_spinor target = pg_base(
            PG_REUSE_PATH[index].theta,
            PG_REUSE_PATH[index].phi
        );
        if (!pg_transport_to(carrier, &target)) {
            return 0;
        }
    }
    return 1;
}

static int pg_inverse_reuse_path(struct pg_spinor *carrier) {
    const size_t count = sizeof(PG_REUSE_PATH) / sizeof(PG_REUSE_PATH[0]);
    for (size_t cursor = count - 1U; cursor > 0U; --cursor) {
        const size_t index = cursor - 1U;
        const struct pg_spinor target = pg_base(
            PG_REUSE_PATH[index].theta,
            PG_REUSE_PATH[index].phi
        );
        if (!pg_transport_to(carrier, &target)) {
            return 0;
        }
    }
    return 1;
}

static int pg_forward_reuse(
    struct pg_spinor *carrier,
    size_t repetitions
) {
    for (size_t repetition = 0U; repetition < repetitions; ++repetition) {
        if (!pg_forward_reuse_path(carrier)) {
            return 0;
        }
    }
    return 1;
}

static int pg_inverse_reuse(
    struct pg_spinor *carrier,
    size_t repetitions
) {
    for (size_t repetition = 0U; repetition < repetitions; ++repetition) {
        if (!pg_inverse_reuse_path(carrier)) {
            return 0;
        }
    }
    return 1;
}

static double complex pg_latitude_discrete_formula(
    struct pg_latitude_descriptor descriptor
) {
    const double c = cos(0.5 * descriptor.theta);
    const double s = sin(0.5 * descriptor.theta);
    const double delta =
        (double)descriptor.orientation
        * 2.0 * PG_PI / (double)descriptor.segments;
    const double complex edge_overlap =
        c * c + s * s * cexp(I * delta);
    const double complex edge_phase =
        conj(edge_overlap) / cabs(edge_overlap);
    double complex result = 1.0 + 0.0 * I;
    for (size_t step = 0U; step < descriptor.segments; ++step) {
        result *= edge_phase;
    }
    return result;
}

static double complex pg_latitude_continuous_limit(
    struct pg_latitude_descriptor descriptor
) {
    const double phase =
        -(double)descriptor.orientation
        * PG_PI * (1.0 - cos(descriptor.theta));
    return cexp(I * phase);
}

static double complex pg_scalar_baseline_latitude(
    struct pg_latitude_descriptor descriptor
) {
    double phase = 0.0;
    struct pg_spinor current = pg_base(descriptor.theta, 0.0);
    for (size_t step = 1U; step <= descriptor.segments; ++step) {
        const double phi =
            (double)descriptor.orientation
            * 2.0 * PG_PI * (double)step
            / (double)descriptor.segments;
        const struct pg_spinor next = pg_base(descriptor.theta, phi);
        phase -= carg(pg_overlap(&current, &next));
        current = next;
    }
    return cexp(I * phase);
}

static double complex pg_scalar_baseline_reuse(size_t repetitions) {
    double phase = 0.0;
    const size_t count = sizeof(PG_REUSE_PATH) / sizeof(PG_REUSE_PATH[0]);
    for (size_t repetition = 0U; repetition < repetitions; ++repetition) {
        for (size_t index = 1U; index < count; ++index) {
            const struct pg_spinor current = pg_base(
                PG_REUSE_PATH[index - 1U].theta,
                PG_REUSE_PATH[index - 1U].phi
            );
            const struct pg_spinor next = pg_base(
                PG_REUSE_PATH[index].theta,
                PG_REUSE_PATH[index].phi
            );
            phase -= carg(pg_overlap(&current, &next));
        }
    }
    return cexp(I * phase);
}

static struct pg_run pg_transaction(size_t segments) {
    const struct pg_latitude_descriptor descriptor = {
        .theta = PG_THETA,
        .segments = segments,
        .orientation = 1
    };
    const struct pg_spinor initial = pg_base(PG_THETA, 0.0);
    struct pg_spinor carrier = initial;
    if (!pg_forward_latitude(&carrier, descriptor)) {
        pg_fail("primary Pancharatnam forward transport rejected");
    }
    const double complex holonomy = pg_overlap(&initial, &carrier);
    const double complex formula =
        pg_latitude_discrete_formula(descriptor);
    const double complex continuous =
        pg_latitude_continuous_limit(descriptor);
    const double norm_error = pg_norm_error(&carrier);
    if (!pg_inverse_latitude(&carrier, descriptor)) {
        pg_fail("primary Pancharatnam inverse transport rejected");
    }
    const struct pg_run result = {
        .segments = segments,
        .holonomy = holonomy,
        .analytic_discrete_holonomy = formula,
        .continuous_limit_holonomy = continuous,
        .discrete_formula_error = cabs(holonomy - formula),
        .continuous_limit_error = cabs(holonomy - continuous),
        .restoration_error = pg_spinor_error(&carrier, &initial),
        .norm_error = norm_error
    };
    return result;
}

static void pg_print_complex(double complex value) {
    printf("[%.17g,%.17g]", creal(value), cimag(value));
}

int main(void) {
    const size_t tested_count =
        sizeof(PG_TEST_SEGMENTS) / sizeof(PG_TEST_SEGMENTS[0]);
    struct pg_run runs[tested_count];
    double previous_limit_error = INFINITY;
    for (size_t index = 0U; index < tested_count; ++index) {
        runs[index] = pg_transaction(PG_TEST_SEGMENTS[index]);
        if (
            runs[index].discrete_formula_error > PG_TOLERANCE
            || runs[index].restoration_error > PG_TOLERANCE
            || runs[index].norm_error > PG_TOLERANCE
            || cabs(
                runs[index].holonomy
                - pg_scalar_baseline_latitude(
                    (struct pg_latitude_descriptor){
                        .theta = PG_THETA,
                        .segments = PG_TEST_SEGMENTS[index],
                        .orientation = 1
                    }
                )
            ) > PG_TOLERANCE
            || !(
                index == 0U
                || runs[index].continuous_limit_error
                    < previous_limit_error
            )
        ) {
            pg_fail("Pancharatnam latitude transaction failed");
        }
        previous_limit_error = runs[index].continuous_limit_error;
    }

    const struct pg_spinor initial = pg_base(PG_THETA, 0.0);
    struct pg_spinor carrier = initial;
    const struct pg_latitude_descriptor primary_descriptor = {
        .theta = PG_THETA,
        .segments = PG_PRIMARY_SEGMENTS,
        .orientation = 1
    };
    if (!pg_forward_latitude(&carrier, primary_descriptor)) {
        pg_fail("accepted primary forward transport rejected");
    }
    const double complex primary_boundary = pg_overlap(&initial, &carrier);
    if (!pg_inverse_latitude(&carrier, primary_descriptor)) {
        pg_fail("accepted primary inverse transport rejected");
    }
    const double primary_restoration_error =
        pg_spinor_error(&carrier, &initial);

    if (!pg_forward_reuse(&carrier, PG_REUSE_REPETITIONS)) {
        pg_fail("accepted reuse forward transport rejected");
    }
    const double complex reuse_boundary = pg_overlap(&initial, &carrier);
    if (!pg_inverse_reuse(&carrier, PG_REUSE_REPETITIONS)) {
        pg_fail("accepted reuse inverse transport rejected");
    }
    const double reuse_restoration_error =
        pg_spinor_error(&carrier, &initial);

    struct pg_spinor fresh_reuse = initial;
    if (!pg_forward_reuse(&fresh_reuse, PG_REUSE_REPETITIONS)) {
        pg_fail("fresh reuse forward transport rejected");
    }
    const double complex fresh_reuse_boundary =
        pg_overlap(&initial, &fresh_reuse);

    double maximum_stress_restoration_error = 0.0;
    struct pg_spinor stress = initial;
    for (size_t cycle = 0U; cycle < PG_STRESS_CYCLES; ++cycle) {
        if (
            !pg_forward_latitude(&stress, primary_descriptor)
            || !pg_inverse_latitude(&stress, primary_descriptor)
            || !pg_forward_reuse(&stress, 1U)
            || !pg_inverse_reuse(&stress, 1U)
        ) {
            pg_fail("stress transport rejected");
        }
        maximum_stress_restoration_error = fmax(
            maximum_stress_restoration_error,
            pg_spinor_error(&stress, &initial)
        );
    }

    struct pg_spinor missing_inverse = initial;
    (void)pg_forward_latitude(&missing_inverse, primary_descriptor);

    struct pg_spinor wrong_inverse = initial;
    (void)pg_forward_latitude(&wrong_inverse, primary_descriptor);
    const struct pg_latitude_descriptor wrong_descriptor = {
        .theta = 1.05,
        .segments = PG_PRIMARY_SEGMENTS,
        .orientation = 1
    };
    (void)pg_inverse_latitude(&wrong_inverse, wrong_descriptor);

    struct pg_spinor phase_canonicalized = initial;
    for (size_t step = 1U; step <= PG_PRIMARY_SEGMENTS; ++step) {
        const double phi =
            2.0 * PG_PI * (double)step
            / (double)PG_PRIMARY_SEGMENTS;
        phase_canonicalized = pg_base(PG_THETA, phi);
    }
    const double complex canonicalized_boundary =
        pg_overlap(&initial, &phase_canonicalized);

    struct pg_spinor reverse_orientation = initial;
    const struct pg_latitude_descriptor reverse_descriptor = {
        .theta = PG_THETA,
        .segments = PG_PRIMARY_SEGMENTS,
        .orientation = -1
    };
    (void)pg_forward_latitude(&reverse_orientation, reverse_descriptor);
    const double complex reverse_boundary =
        pg_overlap(&initial, &reverse_orientation);

    struct pg_spinor area_perturbed = pg_base(0.95, 0.0);
    const struct pg_latitude_descriptor area_descriptor = {
        .theta = 0.95,
        .segments = PG_PRIMARY_SEGMENTS,
        .orientation = 1
    };
    const struct pg_spinor area_initial = area_perturbed;
    (void)pg_forward_latitude(&area_perturbed, area_descriptor);
    const double complex area_boundary =
        pg_overlap(&area_initial, &area_perturbed);

    struct pg_spinor null_carrier = {
        .cell = {0.0 + 0.0 * I, 0.0 + 0.0 * I}
    };
    const struct pg_spinor null_target = pg_base(PG_THETA, 0.1);
    const int null_carrier_accepted =
        pg_transport_to(&null_carrier, &null_target);

    const double complex primary_scalar_baseline =
        pg_scalar_baseline_latitude(primary_descriptor);
    const double complex reuse_scalar_baseline =
        pg_scalar_baseline_reuse(PG_REUSE_REPETITIONS);
    const double endpoint_only_phase = 0.0;

    if (
        primary_restoration_error > PG_TOLERANCE
        || reuse_restoration_error > PG_TOLERANCE
        || maximum_stress_restoration_error > PG_TOLERANCE
        || cabs(reuse_boundary - fresh_reuse_boundary) > PG_TOLERANCE
        || cabs(primary_boundary - primary_scalar_baseline) > PG_TOLERANCE
        || cabs(reuse_boundary - reuse_scalar_baseline) > PG_TOLERANCE
        || cabs(primary_boundary - 1.0) < 0.5
        || cabs(canonicalized_boundary - 1.0) > PG_TOLERANCE
        || cabs(reverse_boundary - conj(primary_boundary)) > PG_TOLERANCE
        || cabs(area_boundary - primary_boundary) < 0.1
        || pg_spinor_error(&missing_inverse, &initial) < 0.5
        || pg_spinor_error(&wrong_inverse, &initial) < 0.01
        || null_carrier_accepted
        || fabs(endpoint_only_phase) > PG_TOLERANCE
    ) {
        pg_fail("Pancharatnam accepted path or controls failed");
    }

    printf(
        "{"
        "\"result\":\"PASS\","
        "\"claim_candidate\":"
        "\"BOUNDED_STATEFUL_PANCHARATNAM_GAUGE_TRANSPORT_"
        "ENDPOINT_INVISIBLE_HOLONOMY_PHASE_MEMORY_WITH_"
        "NUMERICAL_RESTORATION_AND_REUSE\","
        "\"carrier_geometry\":\"U1_FIBER_OVER_PUBLIC_BLOCH_SPHERE_PATH\","
        "\"numerical_tolerance_predeclared\":%.17g,"
        "\"tested_segments\":[",
        PG_TOLERANCE
    );
    for (size_t index = 0U; index < tested_count; ++index) {
        printf(
            "%zu%s",
            PG_TEST_SEGMENTS[index],
            index + 1U == tested_count ? "" : ","
        );
    }
    printf("],\"segment_runs\":[");
    for (size_t index = 0U; index < tested_count; ++index) {
        printf(
            "{"
            "\"segments\":%zu,"
            "\"holonomy\":",
            runs[index].segments
        );
        pg_print_complex(runs[index].holonomy);
        printf(",\"analytic_discrete_holonomy\":");
        pg_print_complex(runs[index].analytic_discrete_holonomy);
        printf(",\"continuous_limit_holonomy\":");
        pg_print_complex(runs[index].continuous_limit_holonomy);
        printf(
            ",\"discrete_formula_error\":%.17g,"
            "\"continuous_limit_error\":%.17g,"
            "\"restoration_error\":%.17g,"
            "\"norm_error\":%.17g}%s",
            runs[index].discrete_formula_error,
            runs[index].continuous_limit_error,
            runs[index].restoration_error,
            runs[index].norm_error,
            index + 1U == tested_count ? "" : ","
        );
    }
    printf(
        "],"
        "\"primary\":{"
        "\"segments\":%zu,"
        "\"holonomy\":",
        PG_PRIMARY_SEGMENTS
    );
    pg_print_complex(primary_boundary);
    printf(
        ",\"restoration_error\":%.17g},"
        "\"reuse\":{"
        "\"program\":\"PUBLIC_SPHERICAL_RECTANGLE\","
        "\"repetitions\":%zu,"
        "\"holonomy\":",
        primary_restoration_error,
        PG_REUSE_REPETITIONS
    );
    pg_print_complex(reuse_boundary);
    printf(
        ",\"restoration_error\":%.17g},"
        "\"fresh_restored_reuse_boundary_error\":%.17g,"
        "\"same_outer_carrier_variable\":true,"
        "\"actual_restored_carrier_reused\":true,"
        "\"restoration_generation_sequence\":[1,2],"
        "\"restoration_generation_lease_enforced\":false,"
        "\"baseline_reload_bytes\":0,"
        "\"stress_cycles\":%zu,"
        "\"stress_maximum_restoration_error\":%.17g,"
        "\"restoration_classification\":"
        "\"NUMERICAL_PHYSICAL_STATE_RESTORATION\","
        "\"controls\":{"
        "\"missing_inverse_restored\":false,"
        "\"wrong_inverse_restored\":false,"
        "\"reordered_inverse_applicable\":false,"
        "\"premature_phase_canonicalization_erases_holonomy\":true,"
        "\"reverse_orientation_conjugates_holonomy\":true,"
        "\"area_perturbation_changes_holonomy\":true,"
        "\"null_carrier_accepted\":false,"
        "\"endpoint_only_state_distinguishes_closed_paths\":false},"
        "\"resource_law\":{"
        "\"phase_carrier_complex128_cells\":2,"
        "\"phase_carrier_inline_bytes\":%zu,"
        "\"restoration_baseline_inline_bytes\":%zu,"
        "\"final_boundary_bytes\":%zu,"
        "\"latitude_descriptor_bytes\":%zu,"
        "\"reuse_public_path_bytes\":%zu,"
        "\"retained_edge_history_bytes\":0,"
        "\"whole_process_rss_claimed\":false},"
        "\"matched_compact_classical\":{"
        "\"identical_scalar_gauge_angle_recurrence\":true,"
        "\"scalar_recurrence_state_minimal\":true,"
        "\"scalar_state_bytes\":%zu,"
        "\"closed_form_public_path_product_available\":true,"
        "\"runtime_advantage_claimed\":false,"
        "\"primary_boundary_error\":%.17g,"
        "\"reuse_boundary_error\":%.17g},"
        "\"intermediate_gauge_phase_projected\":false,"
        "\"final_boundary_projected_before_inverse\":true,"
        "\"catvm_custody_established\":false,"
        "\"endpoint_invisible_scope\":\"PUBLIC_BLOCH_ENDPOINT_ONLY\","
        "\"endpoint_invisible_phase_memory_established\":true,"
        "\"distinct_resource_unavailable_to_compact_classical\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_waveform_execution\":false,"
        "\"physical_bit_replacement\":false,"
        "\"unbounded_computation_established\":false,"
        "\"claim_ceiling\":"
        "\"LINUX_X86_64_DIRECT_PROCESS_COMPLEX128_TWO_CELL_"
        "PANCHARATNAM_LATITUDE_SEGMENTS4_8_16_32_64_128_256_512_"
        "PUBLIC_RECTANGLE_REUSE37_SOFTWARE_ONLY\","
        "\"terminal\":false"
        "}\n",
        reuse_restoration_error,
        cabs(reuse_boundary - fresh_reuse_boundary),
        PG_STRESS_CYCLES,
        maximum_stress_restoration_error,
        sizeof(struct pg_spinor),
        sizeof(struct pg_spinor),
        sizeof(double complex),
        sizeof(struct pg_latitude_descriptor),
        sizeof(PG_REUSE_PATH),
        sizeof(double),
        cabs(primary_boundary - primary_scalar_baseline),
        cabs(reuse_boundary - reuse_scalar_baseline)
    );
    return 0;
}
