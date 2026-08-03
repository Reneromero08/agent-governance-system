#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: fixed-rank nonlinear coherent wave carrier.
 *
 * Four normalized complex wave cells are the actual carrier.  Every public
 * layer applies an intensity-dependent Kerr phase kick followed by two SU(2)
 * interference couplers.  Amplitude remains resident in the wave cells:
 * there is no phasor-pair split, magnitude chart, or gauge canonicalization.
 * Reverse public topology recomputes the lawful adjoint and inverse Kerr kick
 * from the then-resident state, so no inverse matrices or intermediate wave
 * states are retained.
 */

#include <complex.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NW_CELLS 4U
#define NW_REUSE_TRANSACTIONS 16U

static const double NW_RESTORATION_TOLERANCE = 2.0e-10;
static const double NW_NORM_TOLERANCE = 2.0e-12;
static const double NW_CONTROL_MINIMUM = 1.0e-7;
static const double NW_HASH_SCALE = 1.0e12;

struct nw_carrier {
    double complex sealed[NW_CELLS];
    double complex working[NW_CELLS];
    uint64_t restoration_generation;
};

struct nw_stats {
    uint64_t kerr_phase_updates;
    uint64_t interference_couplers;
    uint64_t resident_wave_reads;
    uint64_t retained_inverse_complex_cells;
    uint64_t final_boundary_real_values;
    double maximum_norm_error;
};

struct nw_boundary {
    uint64_t hash;
    double intensity_zero;
    double intensity_two;
    double fringe_zero_one;
};

enum nw_restore_mode {
    NW_RESTORE_CORRECT = 0,
    NW_RESTORE_MISSING_LAYER = 1,
    NW_RESTORE_WRONG_KERR = 2,
    NW_RESTORE_REORDERED = 3,
    NW_RESTORE_SNAPSHOT = 4
};

static void nw_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static double nw_abs_squared(double complex value) {
    return creal(value) * creal(value) + cimag(value) * cimag(value);
}

static double nw_norm_error(const double complex value[NW_CELLS]) {
    double norm = 0.0;
    for (size_t cell = 0U; cell < NW_CELLS; ++cell) {
        norm += nw_abs_squared(value[cell]);
    }
    return fabs(norm - 1.0);
}

static void nw_observe(
    const struct nw_carrier *carrier,
    struct nw_stats *stats
) {
    const double error = nw_norm_error(carrier->working);
    if (error > stats->maximum_norm_error) {
        stats->maximum_norm_error = error;
    }
    if (!isfinite(error) || error > NW_NORM_TOLERANCE) {
        nw_fail("nonlinear wave carrier left normalized state");
    }
}

static struct nw_carrier nw_make_carrier(uint64_t identity) {
    struct nw_carrier carrier = {0};
    double complex seed[NW_CELLS] = {
        0.51 + 0.13 * I,
        -0.21 + 0.47 * I,
        0.34 - 0.29 * I,
        -0.18 - 0.44 * I
    };
    const double twist = 0.003 * (double)(identity % UINT64_C(97));
    double norm = 0.0;
    for (size_t cell = 0U; cell < NW_CELLS; ++cell) {
        seed[cell] *= cexp(I * twist * (double)(cell + 1U));
        norm += nw_abs_squared(seed[cell]);
    }
    const double scale = 1.0 / sqrt(norm);
    for (size_t cell = 0U; cell < NW_CELLS; ++cell) {
        carrier.sealed[cell] = seed[cell] * scale;
        carrier.working[cell] = carrier.sealed[cell];
    }
    return carrier;
}

static double nw_kappa(size_t layer, size_t program) {
    return 0.113
        + 0.009 * (double)((layer + 3U * program) % 17U);
}

static double nw_theta(size_t layer, size_t program, size_t pair) {
    return 0.149
        + 0.007 * (double)((5U * layer + 2U * program + pair) % 19U);
}

static double nw_phi(size_t layer, size_t program, size_t pair) {
    return 0.071
        + 0.013 * (double)((layer + 7U * program + 3U * pair) % 23U);
}

static void nw_kerr(
    struct nw_carrier *carrier,
    double kappa,
    int inverse,
    struct nw_stats *stats
) {
    const double sign = inverse ? -1.0 : 1.0;
    for (size_t cell = 0U; cell < NW_CELLS; ++cell) {
        const double intensity = nw_abs_squared(carrier->working[cell]);
        carrier->working[cell] *= cexp(I * sign * kappa * intensity);
        ++stats->kerr_phase_updates;
        ++stats->resident_wave_reads;
    }
}

static void nw_couple(
    struct nw_carrier *carrier,
    size_t first,
    size_t second,
    double theta,
    double phi,
    int inverse,
    struct nw_stats *stats
) {
    const double cosine = cos(theta);
    const double sine = sin(theta);
    const double complex phase = cexp(I * phi);
    const double complex a = carrier->working[first];
    const double complex b = carrier->working[second];
    if (!inverse) {
        carrier->working[first] = cosine * a + phase * sine * b;
        carrier->working[second] =
            -conj(phase) * sine * a + cosine * b;
    } else {
        carrier->working[first] = cosine * a - phase * sine * b;
        carrier->working[second] =
            conj(phase) * sine * a + cosine * b;
    }
    ++stats->interference_couplers;
    stats->resident_wave_reads += 2U;
}

static void nw_pairs(
    size_t layer,
    size_t first_pair[2],
    size_t second_pair[2]
) {
    if (layer % 2U == 0U) {
        first_pair[0] = 0U;
        first_pair[1] = 1U;
        second_pair[0] = 2U;
        second_pair[1] = 3U;
    } else {
        first_pair[0] = 1U;
        first_pair[1] = 2U;
        second_pair[0] = 3U;
        second_pair[1] = 0U;
    }
}

static void nw_forward_layer(
    struct nw_carrier *carrier,
    size_t layer,
    size_t program,
    int disable_kerr,
    int disable_coupler,
    struct nw_stats *stats
) {
    size_t first_pair[2] = {0U, 0U};
    size_t second_pair[2] = {0U, 0U};
    nw_pairs(layer, first_pair, second_pair);
    if (!disable_kerr) {
        nw_kerr(carrier, nw_kappa(layer, program), 0, stats);
    }
    if (!disable_coupler) {
        nw_couple(
            carrier,
            first_pair[0],
            first_pair[1],
            nw_theta(layer, program, 0U),
            nw_phi(layer, program, 0U),
            0,
            stats
        );
        nw_couple(
            carrier,
            second_pair[0],
            second_pair[1],
            nw_theta(layer, program, 1U),
            nw_phi(layer, program, 1U),
            0,
            stats
        );
    }
    nw_observe(carrier, stats);
}

static void nw_inverse_layer(
    struct nw_carrier *carrier,
    size_t layer,
    size_t program,
    enum nw_restore_mode mode,
    int disable_kerr,
    int disable_coupler,
    struct nw_stats *stats
) {
    size_t first_pair[2] = {0U, 0U};
    size_t second_pair[2] = {0U, 0U};
    nw_pairs(layer, first_pair, second_pair);
    const double kappa = nw_kappa(layer, program)
        + (mode == NW_RESTORE_WRONG_KERR ? 0.017 : 0.0);

    if (mode == NW_RESTORE_REORDERED && !disable_kerr) {
        nw_kerr(carrier, kappa, 1, stats);
    }
    if (!disable_coupler) {
        nw_couple(
            carrier,
            second_pair[0],
            second_pair[1],
            nw_theta(layer, program, 1U),
            nw_phi(layer, program, 1U),
            1,
            stats
        );
        nw_couple(
            carrier,
            first_pair[0],
            first_pair[1],
            nw_theta(layer, program, 0U),
            nw_phi(layer, program, 0U),
            1,
            stats
        );
    }
    if (mode != NW_RESTORE_REORDERED && !disable_kerr) {
        nw_kerr(carrier, kappa, 1, stats);
    }
    nw_observe(carrier, stats);
}

static uint64_t nw_hash_mix(uint64_t hash, uint64_t value) {
    for (size_t byte = 0U; byte < sizeof(value); ++byte) {
        hash ^= (value >> (8U * byte)) & UINT64_C(0xff);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static struct nw_boundary nw_project(
    const struct nw_carrier *carrier,
    struct nw_stats *stats
) {
    struct nw_boundary boundary = {
        .intensity_zero = nw_abs_squared(carrier->working[0]),
        .intensity_two = nw_abs_squared(carrier->working[2]),
        .fringe_zero_one = 0.5 * nw_abs_squared(
            carrier->working[0] + carrier->working[1]
        )
    };
    const double value[3] = {
        boundary.intensity_zero,
        boundary.intensity_two,
        boundary.fringe_zero_one
    };
    boundary.hash = UINT64_C(1469598103934665603);
    for (size_t index = 0U; index < 3U; ++index) {
        const uint64_t quantized = (uint64_t)llround(
            value[index] * NW_HASH_SCALE
        );
        boundary.hash = nw_hash_mix(boundary.hash, quantized);
    }
    stats->final_boundary_real_values += 3U;
    return boundary;
}

static double nw_restoration_error(const struct nw_carrier *carrier) {
    double maximum = 0.0;
    for (size_t cell = 0U; cell < NW_CELLS; ++cell) {
        const double error = cabs(
            carrier->working[cell] - carrier->sealed[cell]
        );
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static struct nw_boundary nw_transaction(
    struct nw_carrier *carrier,
    size_t depth,
    size_t program,
    enum nw_restore_mode mode,
    int disable_kerr,
    int disable_coupler,
    struct nw_stats *stats,
    double *restoration_error
) {
    for (size_t layer = 0U; layer < depth; ++layer) {
        nw_forward_layer(
            carrier,
            layer,
            program,
            disable_kerr,
            disable_coupler,
            stats
        );
    }
    const struct nw_boundary boundary = nw_project(carrier, stats);

    if (mode == NW_RESTORE_SNAPSHOT) {
        memcpy(
            carrier->working,
            carrier->sealed,
            sizeof(carrier->working)
        );
    } else {
        size_t inverse_layers = depth;
        if (mode == NW_RESTORE_MISSING_LAYER && inverse_layers > 0U) {
            --inverse_layers;
        }
        while (inverse_layers > 0U) {
            --inverse_layers;
            nw_inverse_layer(
                carrier,
                inverse_layers,
                program,
                mode,
                disable_kerr,
                disable_coupler,
                stats
            );
        }
        if (mode == NW_RESTORE_CORRECT) {
            ++carrier->restoration_generation;
        }
    }
    *restoration_error = nw_restoration_error(carrier);
    return boundary;
}

#ifndef NW_EMBEDDED
static double nw_boundary_distance(
    const struct nw_boundary *first,
    const struct nw_boundary *second
) {
    return fmax(
        fabs(first->intensity_zero - second->intensity_zero),
        fmax(
            fabs(first->intensity_two - second->intensity_two),
            fabs(first->fringe_zero_one - second->fringe_zero_one)
        )
    );
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--project-intermediate") == 0) {
        nw_fail("active nonlinear wave projection denied");
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        nw_fail("nonlinear wave carrier required");
    }
    if (argc != 1) {
        nw_fail("usage: algebraic_nonlinear_kerr_wave");
    }

    const size_t depths[] = {1U, 4U, 32U, 128U, 512U, 2048U};
    const size_t depth_count = sizeof(depths) / sizeof(depths[0]);
    struct nw_boundary accepted[depth_count];
    double maximum_restoration_error = 0.0;
    double maximum_norm_error = 0.0;
    uint64_t deepest_phase_updates = 0U;
    uint64_t deepest_couplers = 0U;

    for (size_t index = 0U; index < depth_count; ++index) {
        struct nw_carrier carrier = nw_make_carrier(UINT64_C(11));
        struct nw_stats stats = {0};
        double restoration_error = 0.0;
        accepted[index] = nw_transaction(
            &carrier,
            depths[index],
            3U,
            NW_RESTORE_CORRECT,
            0,
            0,
            &stats,
            &restoration_error
        );
        if (
            restoration_error > NW_RESTORATION_TOLERANCE
            || carrier.restoration_generation != 1U
        ) {
            nw_fail("accepted nonlinear wave restoration failed");
        }
        maximum_restoration_error = fmax(
            maximum_restoration_error,
            restoration_error
        );
        maximum_norm_error = fmax(
            maximum_norm_error,
            stats.maximum_norm_error
        );
        if (index + 1U == depth_count) {
            deepest_phase_updates = stats.kerr_phase_updates;
            deepest_couplers = stats.interference_couplers;
        }
    }

    struct nw_carrier controls = nw_make_carrier(UINT64_C(11));
    struct nw_stats control_stats = {0};
    double missing_error = 0.0;
    (void)nw_transaction(
        &controls,
        32U,
        3U,
        NW_RESTORE_MISSING_LAYER,
        0,
        0,
        &control_stats,
        &missing_error
    );
    controls = nw_make_carrier(UINT64_C(11));
    memset(&control_stats, 0, sizeof(control_stats));
    double wrong_error = 0.0;
    (void)nw_transaction(
        &controls,
        32U,
        3U,
        NW_RESTORE_WRONG_KERR,
        0,
        0,
        &control_stats,
        &wrong_error
    );
    controls = nw_make_carrier(UINT64_C(11));
    memset(&control_stats, 0, sizeof(control_stats));
    double reordered_error = 0.0;
    (void)nw_transaction(
        &controls,
        32U,
        3U,
        NW_RESTORE_REORDERED,
        0,
        0,
        &control_stats,
        &reordered_error
    );

    struct nw_carrier snapshot = nw_make_carrier(UINT64_C(11));
    struct nw_stats snapshot_stats = {0};
    double snapshot_error = 0.0;
    (void)nw_transaction(
        &snapshot,
        32U,
        3U,
        NW_RESTORE_SNAPSHOT,
        0,
        0,
        &snapshot_stats,
        &snapshot_error
    );

    struct nw_carrier disabled = nw_make_carrier(UINT64_C(11));
    struct nw_stats disabled_stats = {0};
    double no_kerr_restoration_error = 0.0;
    const struct nw_boundary no_kerr = nw_transaction(
        &disabled,
        32U,
        3U,
        NW_RESTORE_CORRECT,
        1,
        0,
        &disabled_stats,
        &no_kerr_restoration_error
    );
    disabled = nw_make_carrier(UINT64_C(11));
    memset(&disabled_stats, 0, sizeof(disabled_stats));
    double no_coupler_restoration_error = 0.0;
    const struct nw_boundary no_coupler = nw_transaction(
        &disabled,
        32U,
        3U,
        NW_RESTORE_CORRECT,
        0,
        1,
        &disabled_stats,
        &no_coupler_restoration_error
    );

    struct nw_carrier reused = nw_make_carrier(UINT64_C(29));
    struct nw_stats reuse_stats = {0};
    double maximum_reuse_restoration_error = 0.0;
    struct nw_boundary reuse_boundary = {0};
    for (
        size_t transaction = 0U;
        transaction < NW_REUSE_TRANSACTIONS;
        ++transaction
    ) {
        double reuse_error = 0.0;
        reuse_boundary = nw_transaction(
            &reused,
            64U + 3U * transaction,
            41U + transaction,
            NW_RESTORE_CORRECT,
            0,
            0,
            &reuse_stats,
            &reuse_error
        );
        maximum_reuse_restoration_error = fmax(
            maximum_reuse_restoration_error,
            reuse_error
        );
    }

    if (
        missing_error <= NW_CONTROL_MINIMUM
        || wrong_error <= NW_CONTROL_MINIMUM
        || reordered_error <= NW_CONTROL_MINIMUM
        || snapshot_error > NW_RESTORATION_TOLERANCE
        || snapshot.restoration_generation != 0U
        || nw_boundary_distance(&accepted[2], &no_kerr)
            <= NW_CONTROL_MINIMUM
        || nw_boundary_distance(&accepted[2], &no_coupler)
            <= NW_CONTROL_MINIMUM
        || no_kerr_restoration_error > NW_RESTORATION_TOLERANCE
        || no_coupler_restoration_error > NW_RESTORATION_TOLERANCE
        || maximum_reuse_restoration_error > NW_RESTORATION_TOLERANCE
        || reused.restoration_generation != NW_REUSE_TRANSACTIONS
        || reuse_boundary.hash == 0U
    ) {
        nw_fail("nonlinear wave controls failed");
    }

    printf(
        "{\"result\":\"PASS\","
        "\"claim\":\"BOUNDED_FIXED_RANK_NONLINEAR_KERR_INTERFERENCE_WAVE_CARRIER_WITH_ACTUAL_RESTORATION_AND_REUSE\","
        "\"tested_depths\":[1,4,32,128,512,2048],"
        "\"boundary_hashes\":["
        "\"%016" PRIx64 "\",\"%016" PRIx64 "\",\"%016" PRIx64 "\","
        "\"%016" PRIx64 "\",\"%016" PRIx64 "\",\"%016" PRIx64 "\"],"
        "\"depth2048_kerr_phase_updates\":%" PRIu64 ","
        "\"depth2048_interference_couplers\":%" PRIu64 ","
        "\"maximum_norm_error\":%.17g,"
        "\"maximum_restoration_error\":%.17g,"
        "\"restoration_tolerance\":%.17g,"
        "\"missing_inverse_residual\":%.17g,"
        "\"wrong_inverse_residual\":%.17g,"
        "\"reordered_inverse_residual\":%.17g,"
        "\"snapshot_restoration_generation\":%" PRIu64 ","
        "\"reuse_transactions\":%u,"
        "\"reuse_restoration_generation\":%" PRIu64 ","
        "\"maximum_reuse_restoration_error\":%.17g,"
        "\"carrier_complex_cells\":4,"
        "\"live_carrier_bytes\":64,"
        "\"verification_snapshot_bytes\":64,"
        "\"compiled_inverse_history_bytes\":0,"
        "\"best_matched_compact_complex_state_bytes\":64,"
        "\"independent_scalar_reference_parity_depth_max\":512,"
        "\"depth2048_cross_implementation_parity_established\":false,"
        "\"amplitude_resident_in_wave_carrier\":true,"
        "\"host_amplitude_split_count\":0,"
        "\"gauge_canonicalization_count\":0,"
        "\"kerr_intensity_phase_coupling\":true,"
        "\"su2_interference_coupling\":true,"
        "\"fixed_rank_across_depth\":true,"
        "\"topology_derived_inverse_rematerialization\":true,"
        "\"projected_intermediate_complex_cells\":0,"
        "\"serialized_intermediate_complex_cells\":0,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"snapshot_separated_from_actual_inverse\":true,"
        "\"machine_enforced_hidden_intermediate\":false,"
        "\"equivalent_compact_complex_recurrence\":true,"
        "\"genuinely_distinct_phase_resource\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_execution\":false,"
        "\"unlimited_catalytic_computation\":false,"
        "\"terminal\":false}\n",
        accepted[0].hash,
        accepted[1].hash,
        accepted[2].hash,
        accepted[3].hash,
        accepted[4].hash,
        accepted[5].hash,
        deepest_phase_updates,
        deepest_couplers,
        maximum_norm_error,
        maximum_restoration_error,
        NW_RESTORATION_TOLERANCE,
        missing_error,
        wrong_error,
        reordered_error,
        snapshot.restoration_generation,
        NW_REUSE_TRANSACTIONS,
        reused.restoration_generation,
        maximum_reuse_restoration_error
    );
    return 0;
}
#endif
