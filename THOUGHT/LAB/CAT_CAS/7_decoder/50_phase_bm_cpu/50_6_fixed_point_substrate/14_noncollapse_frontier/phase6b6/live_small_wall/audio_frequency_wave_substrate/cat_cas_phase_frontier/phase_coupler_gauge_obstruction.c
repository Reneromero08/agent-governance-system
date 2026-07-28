#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: finite-phasor amplitude/gauge obstruction.
 *
 * D_n(theta)=(1/n) sum_j exp(i theta_j) maps ordered unit phasors to one
 * complex amplitude. Two phasors are singular at zero. Three equilateral
 * phasors remove that local rank defect. A fixed-third-phase chart has a
 * nonsingular Jacobian at the equilateral zero, so this repair is locally
 * real. It cannot be global: on the disk boundary, triangle equality forces
 * all three phasors to equal the boundary value. Each component therefore
 * has winding one, while a circle-valued map extending over the disk must
 * have boundary winding zero. This diagnostic retires only the finite
 * ordered unit-phasor arithmetic-mean encoding as a memoryless global
 * amplitude section; it does not rule out other native phase couplers.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const double PC_PI =
    3.14159265358979323846264338327950288;
static const double PC_TOLERANCE = 2.0e-13;
static const double PC_GAUGE_SHIFT = 0.37;
static const size_t PC_BOUNDARY_SAMPLES = 64U;

struct pc_jacobian {
    double minimum_eigenvalue;
    double maximum_eigenvalue;
    int rank;
};

static void pc_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static double complex pc_decode(const double *angle, size_t count) {
    double complex sum = 0.0 + 0.0 * I;
    for (size_t index = 0U; index < count; ++index) {
        sum += cexp(I * angle[index]);
    }
    return sum / (double)count;
}

static struct pc_jacobian pc_jacobian(
    const double *angle,
    size_t count
) {
    double xx = 0.0;
    double xy = 0.0;
    double yy = 0.0;
    const double scale = 1.0 / (double)count;
    for (size_t index = 0U; index < count; ++index) {
        const double x = -sin(angle[index]) * scale;
        const double y = cos(angle[index]) * scale;
        xx += x * x;
        xy += x * y;
        yy += y * y;
    }
    const double trace = xx + yy;
    const double discriminant = sqrt(
        fmax(0.0, (xx - yy) * (xx - yy) + 4.0 * xy * xy)
    );
    const double minimum = 0.5 * (trace - discriminant);
    const double maximum = 0.5 * (trace + discriminant);
    const struct pc_jacobian result = {
        .minimum_eigenvalue = minimum,
        .maximum_eigenvalue = maximum,
        .rank = (minimum > PC_TOLERANCE ? 1 : 0)
            + (maximum > PC_TOLERANCE ? 1 : 0)
    };
    return result;
}

static double pc_phase_residual(
    const double *first,
    const double *second,
    size_t count
) {
    double maximum = 0.0;
    for (size_t index = 0U; index < count; ++index) {
        const double error = cabs(
            cexp(I * first[index]) - cexp(I * second[index])
        );
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static int pc_boundary_winding(void) {
    double accumulated = 0.0;
    double complex previous = 1.0 + 0.0 * I;
    for (size_t sample = 1U; sample <= PC_BOUNDARY_SAMPLES; ++sample) {
        const double angle =
            2.0 * PC_PI * (double)sample
            / (double)PC_BOUNDARY_SAMPLES;
        const double complex current = cexp(I * angle);
        accumulated += carg(conj(previous) * current);
        previous = current;
    }
    return (int)llround(accumulated / (2.0 * PC_PI));
}

int main(void) {
    const double two_zero[2] = {0.0, PC_PI};
    const double three_zero[3] = {
        0.0,
        2.0 * PC_PI / 3.0,
        4.0 * PC_PI / 3.0
    };
    const double three_boundary[3] = {0.0, 0.0, 0.0};
    double shifted_zero[3] = {0};
    for (size_t index = 0U; index < 3U; ++index) {
        shifted_zero[index] = three_zero[index] + PC_GAUGE_SHIFT;
    }

    const double complex two_amplitude = pc_decode(two_zero, 2U);
    const double complex three_amplitude = pc_decode(three_zero, 3U);
    const double complex shifted_amplitude = pc_decode(
        shifted_zero, 3U
    );
    const double complex boundary_amplitude = pc_decode(
        three_boundary, 3U
    );
    const struct pc_jacobian two_jacobian = pc_jacobian(
        two_zero, 2U
    );
    const struct pc_jacobian three_jacobian = pc_jacobian(
        three_zero, 3U
    );
    const struct pc_jacobian boundary_jacobian = pc_jacobian(
        three_boundary, 3U
    );
    const double gauge_residual = pc_phase_residual(
        three_zero, shifted_zero, 3U
    );
    const double fixed_third_phase_jacobian_determinant =
        sqrt(3.0) / 18.0;
    const double local_x = 1.0e-5;
    const double local_y = -2.0e-5;
    const double local_delta_one =
        3.0 * local_y - sqrt(3.0) * local_x;
    const double local_delta_two = -2.0 * sqrt(3.0) * local_x;
    const double local_angles[3] = {
        three_zero[0] + local_delta_one,
        three_zero[1] + local_delta_two,
        three_zero[2]
    };
    const double complex local_target = local_x + I * local_y;
    const double local_linearization_residual = cabs(
        pc_decode(local_angles, 3U) - local_target
    );
    const int component_winding = pc_boundary_winding();
    const double complex fixed_r = cexp(I * three_zero[2]);
    const double complex fixed_r_failure_z = -fixed_r;
    const double fixed_r_chart_required_pair_sum_modulus = cabs(
        3.0 * fixed_r_failure_z - fixed_r
    );

    if (
        cabs(two_amplitude) > PC_TOLERANCE
        || cabs(three_amplitude) > PC_TOLERANCE
        || cabs(shifted_amplitude) > PC_TOLERANCE
        || cabs(boundary_amplitude - 1.0) > PC_TOLERANCE
        || two_jacobian.rank != 1
        || three_jacobian.rank != 2
        || boundary_jacobian.rank != 1
        || fabs(two_jacobian.maximum_eigenvalue - 0.5)
            > PC_TOLERANCE
        || fabs(three_jacobian.minimum_eigenvalue - 1.0 / 6.0)
            > PC_TOLERANCE
        || fabs(three_jacobian.maximum_eigenvalue - 1.0 / 6.0)
            > PC_TOLERANCE
        || fabs(boundary_jacobian.maximum_eigenvalue - 1.0 / 3.0)
            > PC_TOLERANCE
        || gauge_residual <= 0.3
        || fabs(
            fixed_third_phase_jacobian_determinant
            - sqrt(3.0) / 18.0
        ) > PC_TOLERANCE
        || local_linearization_residual > 1.0e-9
        || component_winding != 1
        || fabs(fixed_r_chart_required_pair_sum_modulus - 4.0)
            > PC_TOLERANCE
    ) {
        pc_fail("finite-phasor gauge diagnostic failed");
    }

    printf(
        "{\"result\":\"PASS\","
        "\"claim\":\"ORDERED_THREE_UNIT_PHASOR_MEAN_HAS_SMOOTH_ZERO_LOCAL_SECTION_BUT_NO_GLOBAL_CONTINUOUS_GAUGE_FREE_DISK_SECTION\","
        "\"two_phasor_zero_jacobian_rank\":%d,"
        "\"two_phasor_zero_jacobian_eigenvalues\":[%.17g,%.17g],"
        "\"three_phasor_equilateral_zero_jacobian_rank\":%d,"
        "\"three_phasor_equilateral_zero_jacobian_eigenvalues\":[%.17g,%.17g],"
        "\"fixed_third_phase_jacobian_determinant\":%.17g,"
        "\"local_section_linearization_residual\":%.17g,"
        "\"three_phasor_boundary_jacobian_rank\":%d,"
        "\"three_phasor_boundary_jacobian_eigenvalues\":[%.17g,%.17g],"
        "\"boundary_sample_count\":%zu,"
        "\"boundary_component_windings\":[%d,%d,%d],"
        "\"boundary_triangle_equality_unique_preimage\":true,"
        "\"disk_to_circle_extension_boundary_winding_required_zero\":true,"
        "\"global_continuous_gauge_free_section\":false,"
        "\"fixed_r_chart_failure_required_pair_sum_modulus\":%.17g,"
        "\"fixed_r_chart_maximum_pair_sum_modulus\":2,"
        "\"fixed_r_chart_global\":false,"
        "\"canonical_zero_amplitude_residual\":%.17g,"
        "\"shifted_zero_amplitude_residual\":%.17g,"
        "\"gauge_shift_radians\":%.17g,"
        "\"actual_phase_cell_residual_same_amplitude\":%.17g,"
        "\"two_phasor_zero_local_section_singular\":true,"
        "\"three_phasor_zero_local_section_exists\":true,"
        "\"three_phasor_zero_local_fiber_dimension\":1,"
        "\"amplitude_only_state_distinguishes_gauge_orbit\":false,"
        "\"amplitude_only_canonicalization_restores_both_actual_carriers\":false,"
        "\"gauge_retention_or_reversible_gauge_transport_required\":true,"
        "\"global_smooth_section_over_closed_disk_established\":false,"
        "\"all_finite_phase_couplers_impossible\":false,"
        "\"host_amplitude_split_eliminated\":false,"
        "\"genuinely_distinct_phase_resource\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_execution\":false,"
        "\"unlimited_catalytic_computation\":false,"
        "\"terminal\":false}\n",
        two_jacobian.rank,
        two_jacobian.minimum_eigenvalue,
        two_jacobian.maximum_eigenvalue,
        three_jacobian.rank,
        three_jacobian.minimum_eigenvalue,
        three_jacobian.maximum_eigenvalue,
        fixed_third_phase_jacobian_determinant,
        local_linearization_residual,
        boundary_jacobian.rank,
        boundary_jacobian.minimum_eigenvalue,
        boundary_jacobian.maximum_eigenvalue,
        PC_BOUNDARY_SAMPLES,
        component_winding,
        component_winding,
        component_winding,
        fixed_r_chart_required_pair_sum_modulus,
        cabs(three_amplitude),
        cabs(shifted_amplitude),
        PC_GAUGE_SHIFT,
        gauge_residual
    );
    return 0;
}
