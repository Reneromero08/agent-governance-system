#define _POSIX_C_SOURCE 200809L

#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Non-Abelian successor to scalar Pancharatnam phase memory.
 *
 * A public point specifies a bright ray in C^3.  The catalytic carrier is an
 * orthonormal two-frame in its dark subspace.  Each public edge transports
 * the actual resident frame with the unitary polar factor that makes the
 * old/new two-frame overlap positive Hermitian.  Closed public loops in two
 * phase coordinates retain noncommuting U(2) holonomies.  The inverse
 * rematerializes public vertices in reverse; no edge history is retained.
 *
 * This is bounded complex128 direct-process software evidence.  The matched
 * compact classical method is the identical 2x2 holonomy recurrence, and
 * these fixed loop modules also have closed-form products.
 */

static const double WZ_PI =
    3.141592653589793238462643383279502884;
static const double WZ_ALPHA = 1.0;
static const double WZ_BETA = 0.7;
static const double WZ_TOLERANCE = 1.0e-9;
static const size_t WZ_SEGMENTS = 512U;
static const size_t WZ_STRESS_CYCLES = 100U;

struct wz_mat2 {
    double complex cell[2][2];
};

struct wz_frame {
    double complex cell[3][2];
};

struct wz_point {
    double alpha;
    double beta;
    double phi1;
    double phi2;
};

struct wz_loop {
    double alpha;
    double beta;
    size_t segments;
    int axis;
    int orientation;
};

static void wz_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static struct wz_mat2 wz_identity(void) {
    const struct wz_mat2 result = {
        .cell = {
            {1.0 + 0.0 * I, 0.0 + 0.0 * I},
            {0.0 + 0.0 * I, 1.0 + 0.0 * I}
        }
    };
    return result;
}

static struct wz_mat2 wz_adjoint(struct wz_mat2 value) {
    struct wz_mat2 result;
    for (size_t row = 0U; row < 2U; ++row) {
        for (size_t column = 0U; column < 2U; ++column) {
            result.cell[row][column] =
                conj(value.cell[column][row]);
        }
    }
    return result;
}

static struct wz_mat2 wz_multiply(
    struct wz_mat2 left,
    struct wz_mat2 right
) {
    struct wz_mat2 result = {
        .cell = {
            {0.0 + 0.0 * I, 0.0 + 0.0 * I},
            {0.0 + 0.0 * I, 0.0 + 0.0 * I}
        }
    };
    for (size_t row = 0U; row < 2U; ++row) {
        for (size_t column = 0U; column < 2U; ++column) {
            for (size_t inner = 0U; inner < 2U; ++inner) {
                result.cell[row][column] +=
                    left.cell[row][inner]
                    * right.cell[inner][column];
            }
        }
    }
    return result;
}

static struct wz_mat2 wz_power(
    struct wz_mat2 base,
    size_t exponent
) {
    struct wz_mat2 result = wz_identity();
    size_t remaining = exponent;
    while (remaining > 0U) {
        if ((remaining & 1U) != 0U) {
            result = wz_multiply(base, result);
        }
        remaining >>= 1U;
        if (remaining > 0U) {
            base = wz_multiply(base, base);
        }
    }
    return result;
}

static double wz_mat_error(
    struct wz_mat2 left,
    struct wz_mat2 right
) {
    double maximum = 0.0;
    for (size_t row = 0U; row < 2U; ++row) {
        for (size_t column = 0U; column < 2U; ++column) {
            maximum = fmax(
                maximum,
                cabs(left.cell[row][column] - right.cell[row][column])
            );
        }
    }
    return maximum;
}

static double wz_mat_frobenius(
    struct wz_mat2 value
) {
    double squared = 0.0;
    for (size_t row = 0U; row < 2U; ++row) {
        for (size_t column = 0U; column < 2U; ++column) {
            const double magnitude = cabs(value.cell[row][column]);
            squared += magnitude * magnitude;
        }
    }
    return sqrt(squared);
}

static struct wz_mat2 wz_subtract(
    struct wz_mat2 left,
    struct wz_mat2 right
) {
    struct wz_mat2 result;
    for (size_t row = 0U; row < 2U; ++row) {
        for (size_t column = 0U; column < 2U; ++column) {
            result.cell[row][column] =
                left.cell[row][column] - right.cell[row][column];
        }
    }
    return result;
}

static double wz_unitarity_error(struct wz_mat2 value) {
    return wz_mat_error(
        wz_multiply(wz_adjoint(value), value),
        wz_identity()
    );
}

static struct wz_frame wz_public_frame(struct wz_point point) {
    const double ca = cos(point.alpha);
    const double sa = sin(point.alpha);
    const double cb = cos(point.beta);
    const double sb = sin(point.beta);
    const double complex e1 = cexp(I * point.phi1);
    const double complex e2 = cexp(I * point.phi2);
    const struct wz_frame result = {
        .cell = {
            {-sa + 0.0 * I, 0.0 + 0.0 * I},
            {ca * cb * e1, -sb * e1},
            {ca * sb * e2, cb * e2}
        }
    };
    return result;
}

static void wz_public_bright(
    struct wz_point point,
    double complex bright[3]
) {
    const double ca = cos(point.alpha);
    const double sa = sin(point.alpha);
    const double cb = cos(point.beta);
    const double sb = sin(point.beta);
    bright[0] = ca + 0.0 * I;
    bright[1] = sa * cb * cexp(I * point.phi1);
    bright[2] = sa * sb * cexp(I * point.phi2);
}

static struct wz_mat2 wz_frame_overlap(
    const struct wz_frame *left,
    const struct wz_frame *right
) {
    struct wz_mat2 result = {
        .cell = {
            {0.0 + 0.0 * I, 0.0 + 0.0 * I},
            {0.0 + 0.0 * I, 0.0 + 0.0 * I}
        }
    };
    for (size_t row = 0U; row < 2U; ++row) {
        for (size_t column = 0U; column < 2U; ++column) {
            for (size_t coordinate = 0U; coordinate < 3U; ++coordinate) {
                result.cell[row][column] +=
                    conj(left->cell[coordinate][row])
                    * right->cell[coordinate][column];
            }
        }
    }
    return result;
}

static struct wz_frame wz_frame_right_multiply(
    const struct wz_frame *frame,
    struct wz_mat2 value
) {
    struct wz_frame result;
    for (size_t coordinate = 0U; coordinate < 3U; ++coordinate) {
        for (size_t column = 0U; column < 2U; ++column) {
            result.cell[coordinate][column] =
                frame->cell[coordinate][0] * value.cell[0][column]
                + frame->cell[coordinate][1] * value.cell[1][column];
        }
    }
    return result;
}

static double wz_frame_error(
    const struct wz_frame *left,
    const struct wz_frame *right
) {
    double maximum = 0.0;
    for (size_t coordinate = 0U; coordinate < 3U; ++coordinate) {
        for (size_t column = 0U; column < 2U; ++column) {
            maximum = fmax(
                maximum,
                cabs(
                    left->cell[coordinate][column]
                    - right->cell[coordinate][column]
                )
            );
        }
    }
    return maximum;
}

static double wz_bright_dark_error(
    const double complex bright[3],
    const struct wz_frame *frame
) {
    double maximum = 0.0;
    for (size_t column = 0U; column < 2U; ++column) {
        double complex overlap = 0.0 + 0.0 * I;
        for (size_t coordinate = 0U; coordinate < 3U; ++coordinate) {
            overlap +=
                conj(bright[coordinate])
                * frame->cell[coordinate][column];
        }
        maximum = fmax(maximum, cabs(overlap));
    }
    return maximum;
}

static double wz_hermitian_error(struct wz_mat2 value) {
    return wz_mat_error(value, wz_adjoint(value));
}

static int wz_positive_hermitian(struct wz_mat2 value) {
    const double determinant =
        creal(value.cell[0][0]) * creal(value.cell[1][1])
        - pow(cabs(value.cell[0][1]), 2.0);
    return wz_hermitian_error(value) <= WZ_TOLERANCE
        && creal(value.cell[0][0]) > 0.0
        && creal(value.cell[1][1]) > 0.0
        && determinant > 0.0;
}

static int wz_inverse_square_root_hermitian(
    struct wz_mat2 value,
    struct wz_mat2 *result
) {
    const double s00 = creal(value.cell[0][0]);
    const double s11 = creal(value.cell[1][1]);
    const double complex s01 = value.cell[0][1];
    const double determinant =
        s00 * s11 - pow(cabs(s01), 2.0);
    if (!(determinant > 1.0e-24)) {
        return 0;
    }
    const double delta = sqrt(determinant);
    const double scale = sqrt(s00 + s11 + 2.0 * delta);
    const double a00 = s00 + delta;
    const double a11 = s11 + delta;
    const double inverse_determinant =
        1.0 / (a00 * a11 - pow(cabs(s01), 2.0));
    result->cell[0][0] =
        scale * a11 * inverse_determinant + 0.0 * I;
    result->cell[0][1] =
        -scale * s01 * inverse_determinant;
    result->cell[1][0] =
        -scale * conj(s01) * inverse_determinant;
    result->cell[1][1] =
        scale * a00 * inverse_determinant + 0.0 * I;
    return 1;
}

static int wz_transport_to(
    struct wz_frame *carrier,
    const struct wz_frame *target
) {
    const struct wz_mat2 overlap =
        wz_frame_overlap(carrier, target);
    const struct wz_mat2 gram =
        wz_multiply(overlap, wz_adjoint(overlap));
    struct wz_mat2 inverse_square_root;
    if (!wz_inverse_square_root_hermitian(
        gram,
        &inverse_square_root
    )) {
        return 0;
    }
    const struct wz_mat2 correction = wz_multiply(
        wz_adjoint(overlap),
        inverse_square_root
    );
    if (wz_unitarity_error(correction) > 1.0e-8) {
        return 0;
    }
    *carrier = wz_frame_right_multiply(target, correction);
    return 1;
}

static int wz_edge_correction(
    const struct wz_frame *left,
    const struct wz_frame *right,
    struct wz_mat2 *correction
) {
    const struct wz_mat2 overlap =
        wz_frame_overlap(left, right);
    const struct wz_mat2 gram =
        wz_multiply(overlap, wz_adjoint(overlap));
    struct wz_mat2 inverse_square_root;
    if (!wz_inverse_square_root_hermitian(
        gram,
        &inverse_square_root
    )) {
        return 0;
    }
    *correction = wz_multiply(
        wz_adjoint(overlap),
        inverse_square_root
    );
    if (wz_unitarity_error(*correction) > 1.0e-8) {
        return 0;
    }
    return 1;
}

static struct wz_point wz_point_for_step(
    struct wz_loop descriptor,
    size_t step
) {
    const double phase =
        (double)descriptor.orientation
        * 2.0 * WZ_PI * (double)step
        / (double)descriptor.segments;
    const struct wz_point result = {
        .alpha = descriptor.alpha,
        .beta = descriptor.beta,
        .phi1 = descriptor.axis == 1 ? phase : 0.0,
        .phi2 = descriptor.axis == 2 ? phase : 0.0
    };
    return result;
}

static int wz_forward_loop(
    struct wz_frame *carrier,
    struct wz_loop descriptor
) {
    for (size_t step = 1U; step <= descriptor.segments; ++step) {
        const struct wz_frame target =
            wz_public_frame(wz_point_for_step(descriptor, step));
        if (!wz_transport_to(carrier, &target)) {
            return 0;
        }
    }
    return 1;
}

static int wz_inverse_loop(
    struct wz_frame *carrier,
    struct wz_loop descriptor
) {
    for (size_t cursor = descriptor.segments; cursor > 0U; --cursor) {
        const struct wz_frame target = wz_public_frame(
            wz_point_for_step(descriptor, cursor - 1U)
        );
        if (!wz_transport_to(carrier, &target)) {
            return 0;
        }
    }
    return 1;
}

static struct wz_loop wz_loop_descriptor(
    int axis,
    int orientation
) {
    const struct wz_loop result = {
        .alpha = WZ_ALPHA,
        .beta = WZ_BETA,
        .segments = WZ_SEGMENTS,
        .axis = axis,
        .orientation = orientation
    };
    return result;
}

static struct wz_mat2 wz_discrete_loop_formula(
    int axis,
    int orientation
) {
    const struct wz_loop descriptor =
        wz_loop_descriptor(axis, orientation);
    const struct wz_frame left = wz_public_frame(
        wz_point_for_step(descriptor, 0U)
    );
    const struct wz_frame right = wz_public_frame(
        wz_point_for_step(descriptor, 1U)
    );
    struct wz_mat2 correction;
    if (!wz_edge_correction(&left, &right, &correction)) {
        wz_fail("discrete loop formula edge rejected");
    }
    return wz_power(correction, descriptor.segments);
}

static struct wz_mat2 wz_continuous_loop(int axis, int orientation) {
    const double ca = cos(WZ_ALPHA);
    const double cb = cos(WZ_BETA);
    const double sb = sin(WZ_BETA);
    double vector[2];
    if (axis == 1) {
        vector[0] = ca * cb;
        vector[1] = -sb;
    } else {
        vector[0] = ca * sb;
        vector[1] = cb;
    }
    const double norm_squared =
        vector[0] * vector[0] + vector[1] * vector[1];
    const double complex coefficient =
        (cexp(
            -I * (double)orientation * 2.0 * WZ_PI * norm_squared
        ) - 1.0) / norm_squared;
    struct wz_mat2 result = wz_identity();
    for (size_t row = 0U; row < 2U; ++row) {
        for (size_t column = 0U; column < 2U; ++column) {
            result.cell[row][column] +=
                coefficient * vector[row] * vector[column];
        }
    }
    return result;
}

static int wz_primary_forward(struct wz_frame *carrier) {
    return wz_forward_loop(carrier, wz_loop_descriptor(1, 1))
        && wz_forward_loop(carrier, wz_loop_descriptor(2, 1));
}

static int wz_primary_inverse(struct wz_frame *carrier) {
    return wz_inverse_loop(carrier, wz_loop_descriptor(2, 1))
        && wz_inverse_loop(carrier, wz_loop_descriptor(1, 1));
}

static int wz_reuse_forward(struct wz_frame *carrier) {
    return wz_forward_loop(carrier, wz_loop_descriptor(1, -1))
        && wz_forward_loop(carrier, wz_loop_descriptor(2, 1))
        && wz_forward_loop(carrier, wz_loop_descriptor(1, 1));
}

static int wz_reuse_inverse(struct wz_frame *carrier) {
    return wz_inverse_loop(carrier, wz_loop_descriptor(1, 1))
        && wz_inverse_loop(carrier, wz_loop_descriptor(2, 1))
        && wz_inverse_loop(carrier, wz_loop_descriptor(1, -1));
}

static void wz_print_complex(double complex value) {
    printf("[%.17g,%.17g]", creal(value), cimag(value));
}

static void wz_print_mat(struct wz_mat2 value) {
    printf("[");
    for (size_t row = 0U; row < 2U; ++row) {
        printf("[");
        for (size_t column = 0U; column < 2U; ++column) {
            wz_print_complex(value.cell[row][column]);
            printf("%s", column == 1U ? "" : ",");
        }
        printf("]%s", row == 1U ? "" : ",");
    }
    printf("]");
}

int main(void) {
    const struct wz_point base_point = {
        .alpha = WZ_ALPHA,
        .beta = WZ_BETA,
        .phi1 = 0.0,
        .phi2 = 0.0
    };
    const struct wz_frame initial = wz_public_frame(base_point);
    double complex bright[3];
    wz_public_bright(base_point, bright);
    const double initial_orthonormality_error = wz_unitarity_error(
        wz_frame_overlap(&initial, &initial)
    );
    const double bright_dark_orthogonality_error =
        wz_bright_dark_error(bright, &initial);
    struct wz_frame edge_transport = initial;
    const struct wz_frame edge_target = wz_public_frame(
        wz_point_for_step(wz_loop_descriptor(1, 1), 1U)
    );
    if (!wz_transport_to(&edge_transport, &edge_target)) {
        wz_fail("geometry control edge transport rejected");
    }
    const struct wz_mat2 transported_edge_overlap =
        wz_frame_overlap(&initial, &edge_transport);
    struct wz_frame carrier = initial;

    if (!wz_primary_forward(&carrier)) {
        wz_fail("primary non-Abelian forward transport rejected");
    }
    const struct wz_mat2 primary_boundary =
        wz_frame_overlap(&initial, &carrier);
    if (!wz_primary_inverse(&carrier)) {
        wz_fail("primary non-Abelian inverse transport rejected");
    }
    const double primary_restoration_error =
        wz_frame_error(&carrier, &initial);

    if (!wz_reuse_forward(&carrier)) {
        wz_fail("reuse non-Abelian forward transport rejected");
    }
    const struct wz_mat2 reuse_boundary =
        wz_frame_overlap(&initial, &carrier);
    if (!wz_reuse_inverse(&carrier)) {
        wz_fail("reuse non-Abelian inverse transport rejected");
    }
    const double reuse_restoration_error =
        wz_frame_error(&carrier, &initial);

    struct wz_frame fresh_reuse = initial;
    if (!wz_reuse_forward(&fresh_reuse)) {
        wz_fail("fresh reuse non-Abelian transport rejected");
    }
    const struct wz_mat2 fresh_reuse_boundary =
        wz_frame_overlap(&initial, &fresh_reuse);

    const struct wz_mat2 h1 = wz_discrete_loop_formula(1, 1);
    const struct wz_mat2 h2 = wz_discrete_loop_formula(2, 1);
    const struct wz_mat2 analytic_primary = wz_multiply(h2, h1);
    const struct wz_mat2 analytic_reordered = wz_multiply(h1, h2);
    const double noncommutator_norm = wz_mat_frobenius(
        wz_subtract(analytic_primary, analytic_reordered)
    );
    const struct wz_mat2 analytic_reuse = wz_multiply(
        h1,
        wz_multiply(h2, wz_adjoint(h1))
    );
    const struct wz_mat2 continuous_h1 = wz_continuous_loop(1, 1);
    const struct wz_mat2 continuous_h2 = wz_continuous_loop(2, 1);
    const struct wz_mat2 continuous_primary =
        wz_multiply(continuous_h2, continuous_h1);

    struct wz_frame reordered_forward = initial;
    (void)wz_forward_loop(
        &reordered_forward,
        wz_loop_descriptor(2, 1)
    );
    (void)wz_forward_loop(
        &reordered_forward,
        wz_loop_descriptor(1, 1)
    );
    const struct wz_mat2 reordered_boundary =
        wz_frame_overlap(&initial, &reordered_forward);

    struct wz_frame missing_inverse = initial;
    (void)wz_primary_forward(&missing_inverse);

    struct wz_frame wrong_inverse = initial;
    (void)wz_primary_forward(&wrong_inverse);
    struct wz_loop wrong_loop = wz_loop_descriptor(2, 1);
    wrong_loop.alpha = 0.92;
    (void)wz_inverse_loop(&wrong_inverse, wrong_loop);
    (void)wz_inverse_loop(
        &wrong_inverse,
        wz_loop_descriptor(1, 1)
    );

    struct wz_frame reordered_inverse = initial;
    (void)wz_primary_forward(&reordered_inverse);
    (void)wz_inverse_loop(
        &reordered_inverse,
        wz_loop_descriptor(1, 1)
    );
    (void)wz_inverse_loop(
        &reordered_inverse,
        wz_loop_descriptor(2, 1)
    );

    struct wz_frame null_carrier = {
        .cell = {
            {0.0 + 0.0 * I, 0.0 + 0.0 * I},
            {0.0 + 0.0 * I, 0.0 + 0.0 * I},
            {0.0 + 0.0 * I, 0.0 + 0.0 * I}
        }
    };
    const struct wz_frame null_target = wz_public_frame(
        wz_point_for_step(wz_loop_descriptor(1, 1), 1U)
    );
    const int null_carrier_accepted =
        wz_transport_to(&null_carrier, &null_target);

    double maximum_stress_restoration_error = 0.0;
    struct wz_frame stress = initial;
    for (size_t cycle = 0U; cycle < WZ_STRESS_CYCLES; ++cycle) {
        if (
            !wz_primary_forward(&stress)
            || !wz_primary_inverse(&stress)
            || !wz_reuse_forward(&stress)
            || !wz_reuse_inverse(&stress)
        ) {
            wz_fail("non-Abelian stress transport rejected");
        }
        maximum_stress_restoration_error = fmax(
            maximum_stress_restoration_error,
            wz_frame_error(&stress, &initial)
        );
    }

    const double primary_formula_error =
        wz_mat_error(primary_boundary, analytic_primary);
    const double primary_continuous_limit_error =
        wz_mat_error(primary_boundary, continuous_primary);
    const double reordered_formula_error =
        wz_mat_error(reordered_boundary, analytic_reordered);
    const double reuse_formula_error =
        wz_mat_error(reuse_boundary, analytic_reuse);
    const double fresh_reuse_boundary_error =
        wz_mat_error(reuse_boundary, fresh_reuse_boundary);
    const double primary_unitarity_error =
        wz_unitarity_error(primary_boundary);
    const double reuse_unitarity_error =
        wz_unitarity_error(reuse_boundary);

    if (
        primary_formula_error > WZ_TOLERANCE
        || reordered_formula_error > WZ_TOLERANCE
        || reuse_formula_error > WZ_TOLERANCE
        || fresh_reuse_boundary_error > WZ_TOLERANCE
        || primary_restoration_error > WZ_TOLERANCE
        || reuse_restoration_error > WZ_TOLERANCE
        || maximum_stress_restoration_error > WZ_TOLERANCE
        || primary_unitarity_error > WZ_TOLERANCE
        || reuse_unitarity_error > WZ_TOLERANCE
        || initial_orthonormality_error > WZ_TOLERANCE
        || bright_dark_orthogonality_error > WZ_TOLERANCE
        || !wz_positive_hermitian(transported_edge_overlap)
        || !(noncommutator_norm > 0.1)
        || wz_frame_error(&missing_inverse, &initial) < 0.1
        || wz_frame_error(&wrong_inverse, &initial) < 0.01
        || wz_frame_error(&reordered_inverse, &initial) < 0.01
        || null_carrier_accepted
    ) {
        wz_fail("non-Abelian accepted path or controls failed");
    }

    printf(
        "{"
        "\"result\":\"PASS\","
        "\"claim_candidate\":"
        "\"BOUNDED_NONABELIAN_WILCZEK_ZEE_SHARED_PHASE_FRAME_"
        "NONCOMMUTING_HOLONOMY_COMPOSITION_WITH_NUMERICAL_"
        "RESTORATION_AND_REUSE\","
        "\"carrier_geometry\":\"U2_DARK_FRAME_OVER_PUBLIC_CP2_BRIGHT_RAY\","
        "\"public_parameters\":{\"alpha\":%.17g,\"beta\":%.17g},"
        "\"loop_segments\":%zu,"
        "\"numerical_tolerance_predeclared\":%.17g,"
        "\"primary\":{"
        "\"public_word\":[\"PHI1_POSITIVE_LOOP\",\"PHI2_POSITIVE_LOOP\"],"
        "\"boundary\":",
        WZ_ALPHA,
        WZ_BETA,
        WZ_SEGMENTS,
        WZ_TOLERANCE
    );
    wz_print_mat(primary_boundary);
    printf(
        ",\"discrete_formula_boundary\":"
    );
    wz_print_mat(analytic_primary);
    printf(
        ",\"formula_error\":%.17g,"
        "\"continuous_limit_error\":%.17g,"
        "\"unitarity_error\":%.17g,"
        "\"restoration_error\":%.17g},"
        "\"reordered_forward\":{"
        "\"public_word\":[\"PHI2_POSITIVE_LOOP\",\"PHI1_POSITIVE_LOOP\"],"
        "\"boundary\":",
        primary_formula_error,
        primary_continuous_limit_error,
        primary_unitarity_error,
        primary_restoration_error
    );
    wz_print_mat(reordered_boundary);
    printf(
        ",\"discrete_formula_boundary\":"
    );
    wz_print_mat(analytic_reordered);
    printf(
        ",\"formula_error\":%.17g,"
        "\"boundary_difference_frobenius\":%.17g},"
        "\"reuse\":{"
        "\"public_word\":[\"PHI1_NEGATIVE_LOOP\","
        "\"PHI2_POSITIVE_LOOP\",\"PHI1_POSITIVE_LOOP\"],"
        "\"boundary\":",
        reordered_formula_error,
        noncommutator_norm
    );
    wz_print_mat(reuse_boundary);
    printf(
        ",\"discrete_formula_boundary\":"
    );
    wz_print_mat(analytic_reuse);
    printf(
        ",\"formula_error\":%.17g,"
        "\"unitarity_error\":%.17g,"
        "\"restoration_error\":%.17g},"
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
        "\"reordered_inverse_applicable\":true,"
        "\"reordered_inverse_restored\":false,"
        "\"noncommuting_loop_order_changes_boundary\":true,"
        "\"public_dark_frame_orthonormal\":true,"
        "\"public_bright_dark_orthogonal\":true,"
        "\"transported_edge_overlap_positive_hermitian\":true,"
        "\"null_carrier_accepted\":false},"
        "\"resource_law\":{"
        "\"resident_phase_frame_complex128_cells\":6,"
        "\"resident_phase_frame_inline_bytes\":%zu,"
        "\"restoration_verification_baseline_inline_bytes\":%zu,"
        "\"final_boundary_inline_bytes\":%zu,"
        "\"public_loop_descriptor_inline_bytes_each\":%zu,"
        "\"declared_named_per_edge_object_subtotal_bytes\":%zu,"
        "\"o2_compiler_reported_transport_function_stack_bytes\":784,"
        "\"complete_per_edge_peak_bounded\":false,"
        "\"retained_edge_history_bytes\":0,"
        "\"compiler_stack_padding_code_libm_allocator_and_whole_process_"
        "excluded\":true,"
        "\"whole_process_rss_claimed\":false},"
        "\"matched_compact_classical\":{"
        "\"identical_2x2_holonomy_recurrence\":true,"
        "\"matrix_state_inline_bytes\":%zu,"
        "\"matrix_state_minimal_claimed\":false,"
        "\"closed_form_fixed_loop_modules_available\":true,"
        "\"primary_boundary_error\":%.17g,"
        "\"reuse_boundary_error\":%.17g,"
        "\"runtime_advantage_claimed\":false},"
        "\"shared_phase_frame_consumed_by_multiple_modules\":true,"
        "\"intermediate_frame_projected\":false,"
        "\"final_boundary_projected_before_inverse\":true,"
        "\"catvm_custody_established\":false,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_waveform_execution\":false,"
        "\"physical_bit_replacement\":false,"
        "\"unbounded_computation_established\":false,"
        "\"claim_ceiling\":"
        "\"LINUX_X86_64_DIRECT_PROCESS_COMPLEX128_C3_DARK_TWO_FRAME_"
        "PUBLIC_CP2_PHI1_PHI2_LOOPS_SEGMENTS512_PRIMARY_TWO_MODULE_"
        "REUSE_THREE_MODULE_SOFTWARE_ONLY\","
        "\"terminal\":false"
        "}\n",
        reuse_formula_error,
        reuse_unitarity_error,
        reuse_restoration_error,
        fresh_reuse_boundary_error,
        WZ_STRESS_CYCLES,
        maximum_stress_restoration_error,
        sizeof(struct wz_frame),
        sizeof(struct wz_frame),
        sizeof(struct wz_mat2),
        sizeof(struct wz_loop),
        2U * sizeof(struct wz_frame) + 4U * sizeof(struct wz_mat2),
        sizeof(struct wz_mat2),
        primary_formula_error,
        reuse_formula_error
    );
    return 0;
}
