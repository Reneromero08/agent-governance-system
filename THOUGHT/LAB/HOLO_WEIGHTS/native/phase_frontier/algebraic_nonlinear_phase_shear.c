#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: nonlinear unit-phase torus shears.
 *
 * Two resident unit phasors form the complete hidden state.  A public shear
 * rotates one actual resident phasor by k*sin(theta_source), leaving its
 * source fixed.  The inverse is the conjugate rotation while that same source
 * is resident.  Alternating axes form a nonlinear, noncommuting reversible
 * torus map.  Only a copied final two-phase boundary is decoded as an
 * interference intensity.
 *
 * This is a software phase-machine result.  A matched two-angle classical
 * recurrence exists, so it does not establish a distinct phase resource,
 * computational advantage, a Small Wall crossing, or physical execution.
 */

#define main algebraic_series_parallel_embedded_main
#include "algebraic_series_parallel_phase.c"
#undef main

#include <errno.h>

#define PS_RESIDENT_CELLS 2U
#define PS_BOUNDARY_START PS_RESIDENT_CELLS
#define PS_CARRIER_CELLS 4U
#define PS_MIN_DEPTH 3U
#define PS_MAX_DEPTH 4096U
#define PS_REUSE_CYCLES 16U
#define PS_HASH_SCALE 1000000000.0

static const double PS_PHASE_TOLERANCE = 2.0e-12;
static const double PS_REFERENCE_TOLERANCE = 2.0e-10;

enum ps_program {
    PS_PRIMARY = 0,
    PS_REUSE = 1
};

enum ps_mode {
    PS_CORRECT = 0,
    PS_WRONG_BOUNDARY_INVERSE = 1,
    PS_MISSING_SHEAR_INVERSE = 2,
    PS_WRONG_SHEAR_INVERSE = 3,
    PS_REORDERED_INVERSE = 4,
    PS_SNAPSHOT_RELOAD = 5,
    PS_COUPLING_DISABLED = 6,
    PS_BASELINE_NEUTRALIZED = 7
};

struct ps_stats {
    uint64_t forward_shears;
    uint64_t inverse_shears;
    uint64_t phase_source_reads;
    uint64_t native_phase_updates;
    uint64_t final_phase_decodes;
    double maximum_unit_modulus_error;
};

struct ps_projection {
    double complex phase[PS_RESIDENT_CELLS];
    double interference_probability;
    uint64_t hash;
};

struct ps_execution {
    struct ps_projection projection;
    struct ps_stats stats;
    double restoration_max_abs;
    double integrity_max_abs;
    uint64_t restoration_generation;
    int actual_inverse;
    int snapshot_loaded;
};

static size_t ps_parse_depth(const char *text) {
    if (text == NULL || text[0] < '1' || text[0] > '9') {
        fail("depth must be canonical decimal from 3 through 4096");
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            fail("depth must be canonical decimal from 3 through 4096");
        }
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long parsed = strtoul(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed < PS_MIN_DEPTH
        || parsed > PS_MAX_DEPTH
    ) {
        fail("depth must be canonical decimal from 3 through 4096");
    }
    return (size_t)parsed;
}

static void ps_observe_integrity(
    const struct carrier *carrier,
    struct ps_stats *stats
) {
    for (size_t cell = 0U; cell < carrier->cells; ++cell) {
        const double error = fabs(cabs(carrier->working[cell]) - 1.0);
        if (error > stats->maximum_unit_modulus_error) {
            stats->maximum_unit_modulus_error = error;
        }
        if (error > PS_PHASE_TOLERANCE) {
            fail("native phase update left the unit torus");
        }
    }
}

static double ps_initial_angle(
    enum ps_program program,
    size_t cell
) {
    if (program == PS_PRIMARY) {
        return cell == 0U ? 0.37 : -0.82;
    }
    return cell == 0U ? -1.11 : 0.23;
}

static size_t ps_shear_axis(size_t index) {
    return index % PS_RESIDENT_CELLS;
}

static double ps_shear_strength(
    enum ps_program program,
    size_t index
) {
    const size_t offset = program == PS_PRIMARY ? 1U : 4U;
    const size_t residue = (5U * index + offset) % 9U;
    const double magnitude = 0.015 * (double)(residue + 1U);
    return ((index / 2U) % 2U) == 0U
        ? magnitude
        : -magnitude;
}

static double complex ps_source_phase(
    const struct carrier *carrier,
    size_t cell,
    enum ps_mode mode
) {
    if (mode == PS_BASELINE_NEUTRALIZED) {
        return carrier->working[cell];
    }
    return relative(carrier, cell);
}

static void ps_apply_seal(
    struct carrier *carrier,
    enum ps_program program,
    int inverse,
    struct ps_stats *stats
) {
    for (size_t cell = 0U; cell < PS_RESIDENT_CELLS; ++cell) {
        const double complex phase = cexp(
            I * ps_initial_angle(program, cell)
        );
        multiply_cell(
            carrier, cell, inverse ? conj(phase) : phase
        );
        ++stats->native_phase_updates;
    }
    ps_observe_integrity(carrier, stats);
}

static void ps_apply_shear(
    struct carrier *carrier,
    enum ps_program program,
    size_t index,
    int inverse,
    int wrong,
    enum ps_mode mode,
    struct ps_stats *stats
) {
    const size_t target = ps_shear_axis(index);
    const size_t source = 1U - target;
    const double complex source_phase = ps_source_phase(
        carrier, source, mode
    );
    ++stats->phase_source_reads;
    double strength = ps_shear_strength(program, index);
    if (inverse && !wrong) {
        strength = -strength;
    }
    if (mode == PS_COUPLING_DISABLED) {
        strength = 0.0;
    }
    const double modulation = cimag(source_phase);
    const double complex factor = cexp(I * strength * modulation);
    multiply_cell(carrier, target, factor);
    ++stats->native_phase_updates;
    if (inverse) {
        ++stats->inverse_shears;
    } else {
        ++stats->forward_shears;
    }
    ps_observe_integrity(carrier, stats);
}

static void ps_copy_boundary(
    struct carrier *carrier,
    int inverse,
    int wrong,
    struct ps_stats *stats
) {
    for (size_t cell = 0U; cell < PS_RESIDENT_CELLS; ++cell) {
        const double complex phase = relative(carrier, cell);
        ++stats->phase_source_reads;
        multiply_cell(
            carrier,
            PS_BOUNDARY_START + cell,
            inverse && !wrong ? conj(phase) : phase
        );
        ++stats->native_phase_updates;
    }
    ps_observe_integrity(carrier, stats);
}

static uint64_t ps_hash_bytes(
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

static int64_t ps_quantize(double value) {
    if (!isfinite(value) || fabs(value) > 1.000000001) {
        fail("invalid projected phase component");
    }
    return (int64_t)llround(value * PS_HASH_SCALE);
}

static struct ps_projection ps_latch_boundary(
    const struct carrier *carrier,
    struct ps_stats *stats
) {
    struct ps_projection projection = {
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t cell = 0U; cell < PS_RESIDENT_CELLS; ++cell) {
        const double complex phase = relative(
            carrier, PS_BOUNDARY_START + cell
        );
        projection.phase[cell] = phase;
        const int64_t real = ps_quantize(creal(phase));
        const int64_t imaginary = ps_quantize(cimag(phase));
        projection.hash = ps_hash_bytes(
            projection.hash,
            (const unsigned char *)&real,
            sizeof(real)
        );
        projection.hash = ps_hash_bytes(
            projection.hash,
            (const unsigned char *)&imaginary,
            sizeof(imaginary)
        );
        ++stats->final_phase_decodes;
    }
    projection.interference_probability = 0.5 * (
        1.0 + creal(
            projection.phase[0] * conj(projection.phase[1])
        )
    );
    if (
        projection.interference_probability < -PS_REFERENCE_TOLERANCE
        || projection.interference_probability
            > 1.0 + PS_REFERENCE_TOLERANCE
    ) {
        fail("interference projection outside probability interval");
    }
    return projection;
}

static struct ps_execution ps_execute(
    struct carrier *carrier,
    size_t depth,
    enum ps_program program,
    enum ps_mode mode
) {
    if (
        carrier == NULL
        || carrier->cells != PS_CARRIER_CELLS
        || depth < PS_MIN_DEPTH
        || depth > PS_MAX_DEPTH
    ) {
        fail("invalid nonlinear phase-shear transaction");
    }
    struct carrier borrowed = snapshot_carrier(carrier);
    struct ps_execution execution = {0};
    ps_apply_seal(carrier, program, 0, &execution.stats);
    for (size_t index = 0U; index < depth; ++index) {
        ps_apply_shear(
            carrier,
            program,
            index,
            0,
            0,
            mode,
            &execution.stats
        );
    }
    ps_copy_boundary(carrier, 0, 0, &execution.stats);
    execution.projection = ps_latch_boundary(
        carrier, &execution.stats
    );

    if (mode == PS_SNAPSHOT_RELOAD) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        execution.snapshot_loaded = 1;
    } else {
        ps_copy_boundary(
            carrier,
            1,
            mode == PS_WRONG_BOUNDARY_INVERSE,
            &execution.stats
        );
        for (size_t reverse = 0U; reverse < depth; ++reverse) {
            if (
                mode == PS_MISSING_SHEAR_INVERSE
                && reverse == 0U
            ) {
                continue;
            }
            size_t selected = reverse;
            if (mode == PS_REORDERED_INVERSE) {
                if (reverse == 0U) {
                    selected = 1U;
                } else if (reverse == 1U) {
                    selected = 0U;
                }
            }
            const size_t index = depth - 1U - selected;
            ps_apply_shear(
                carrier,
                program,
                index,
                1,
                mode == PS_WRONG_SHEAR_INVERSE
                    && reverse == 0U,
                mode,
                &execution.stats
            );
        }
        ps_apply_seal(
            carrier, program, 1, &execution.stats
        );
        if (mode == PS_CORRECT) {
            execution.actual_inverse = 1;
            execution.restoration_generation = UINT64_C(1);
        }
    }
    execution.restoration_max_abs = restoration(
        carrier, &borrowed
    );
    execution.integrity_max_abs = integrity(carrier);
    free_carrier(&borrowed);
    return execution;
}

static void ps_require_correct(
    const struct ps_execution *execution,
    size_t depth
) {
    if (
        !execution->actual_inverse
        || execution->snapshot_loaded
        || execution->restoration_generation != UINT64_C(1)
        || execution->stats.forward_shears != depth
        || execution->stats.inverse_shears != depth
        || execution->stats.final_phase_decodes
            != PS_RESIDENT_CELLS
        || execution->stats.maximum_unit_modulus_error
            > PS_PHASE_TOLERANCE
        || execution->restoration_max_abs > PS_PHASE_TOLERANCE
        || execution->integrity_max_abs > PS_PHASE_TOLERANCE
    ) {
        fail("nonlinear phase-shear transaction violated carrier law");
    }
}

static struct ps_execution ps_control(
    size_t depth,
    enum ps_program program,
    enum ps_mode mode
) {
    const struct process process = {
        .carrier_cells = PS_CARRIER_CELLS
    };
    struct carrier carrier = make_carrier(
        &process, 11100 + 10 * (int)program + (int)mode
    );
    struct ps_execution execution = ps_execute(
        &carrier, depth, program, mode
    );
    free_carrier(&carrier);
    return execution;
}

int main(int argc, char **argv) {
    if (
        argc == 2
        && strcmp(argv[1], "--project-intermediate") == 0
    ) {
        fail("only the final phase-interference boundary may be projected");
    }
    if (
        argc == 2
        && strcmp(argv[1], "--null-carrier") == 0
    ) {
        (void)ps_execute(
            NULL, 3U, PS_PRIMARY, PS_CORRECT
        );
        fail("null carrier control failed open");
    }
    if (argc != 2) {
        fail(
            "usage: algebraic_nonlinear_phase_shear "
            "<depth 3..4096>"
        );
    }
    const size_t depth = ps_parse_depth(argv[1]);
    const struct process process = {
        .carrier_cells = PS_CARRIER_CELLS
    };
    struct carrier carrier = make_carrier(&process, 11001);
    struct ps_execution primary = ps_execute(
        &carrier, depth, PS_PRIMARY, PS_CORRECT
    );
    ps_require_correct(&primary, depth);
    struct ps_execution reuse = ps_execute(
        &carrier, depth, PS_REUSE, PS_CORRECT
    );
    ps_require_correct(&reuse, depth);
    if (primary.projection.hash == reuse.projection.hash) {
        fail("unrelated nonlinear phase reuse did not change boundary");
    }
    double repeated_max = 0.0;
    for (size_t cycle = 0U; cycle < PS_REUSE_CYCLES; ++cycle) {
        struct ps_execution repeated = ps_execute(
            &carrier,
            depth,
            cycle % 2U == 0U ? PS_PRIMARY : PS_REUSE,
            PS_CORRECT
        );
        ps_require_correct(&repeated, depth);
        if (repeated.restoration_max_abs > repeated_max) {
            repeated_max = repeated.restoration_max_abs;
        }
    }
    free_carrier(&carrier);

    struct ps_execution wrong_boundary = ps_control(
        32U, PS_PRIMARY, PS_WRONG_BOUNDARY_INVERSE
    );
    struct ps_execution missing = ps_control(
        32U, PS_PRIMARY, PS_MISSING_SHEAR_INVERSE
    );
    struct ps_execution wrong_shear = ps_control(
        32U, PS_PRIMARY, PS_WRONG_SHEAR_INVERSE
    );
    struct ps_execution reordered = ps_control(
        32U, PS_PRIMARY, PS_REORDERED_INVERSE
    );
    struct ps_execution snapshot = ps_control(
        32U, PS_PRIMARY, PS_SNAPSHOT_RELOAD
    );
    struct ps_execution disabled = ps_control(
        depth, PS_PRIMARY, PS_COUPLING_DISABLED
    );
    struct ps_execution neutralized = ps_control(
        depth, PS_PRIMARY, PS_BASELINE_NEUTRALIZED
    );
    if (
        wrong_boundary.restoration_max_abs < CONTROL_MINIMUM
        || missing.restoration_max_abs < CONTROL_MINIMUM
        || wrong_shear.restoration_max_abs < CONTROL_MINIMUM
        || reordered.restoration_max_abs < CONTROL_MINIMUM
        || !snapshot.snapshot_loaded
        || snapshot.actual_inverse
        || snapshot.restoration_max_abs > PS_PHASE_TOLERANCE
        || disabled.projection.hash == primary.projection.hash
        || neutralized.projection.hash == primary.projection.hash
    ) {
        fail("nonlinear phase-shear control did not separate");
    }

    printf(
        "{\"mode\":\"nonlinear-unit-phase-torus-shear\","
        "\"claim\":\"BOUNDED_NONLINEAR_UNIT_PHASE_TORUS_SHEAR_COMPOSITION_WITH_ACTUAL_RESTORATION_AND_INTERFERENCE_PROJECTION\","
        "\"depth\":%zu,"
        "\"phase_rank\":2,"
        "\"resident_phase_cells\":2,"
        "\"boundary_phase_cells\":2,"
        "\"carrier_cells\":4,"
        "\"live_carrier_bytes\":%zu,"
        "\"verification_snapshot_bytes\":%zu,"
        "\"accepted_path_peak_carrier_bytes\":%zu,"
        "\"projection_bytes\":%zu,"
        "\"declared_local_shear_temporary_bytes\":%zu,"
        "\"best_matched_classical_state_bytes\":%zu,"
        "\"compiled_shear_list_bytes\":0,"
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"coupling_disabled_boundary_fnv1a64\":\"%016llx\","
        "\"baseline_neutralized_boundary_fnv1a64\":\"%016llx\","
        "\"primary_interference_probability\":%.17g,"
        "\"coupling_disabled_interference_probability\":%.17g,"
        "\"primary_forward_shears\":%llu,"
        "\"primary_inverse_shears\":%llu,"
        "\"primary_phase_source_reads\":%llu,"
        "\"primary_native_phase_updates\":%llu,"
        "\"maximum_unit_modulus_error\":%.17g,"
        "\"maximum_repeated_restoration_error\":%.17g,"
        "\"wrong_boundary_restoration_residual\":%.17g,"
        "\"missing_inverse_restoration_residual\":%.17g,"
        "\"wrong_shear_restoration_residual\":%.17g,"
        "\"reordered_inverse_restoration_residual\":%.17g,"
        "\"snapshot_reload_restoration_residual\":%.17g,"
        "\"final_phase_decodes\":%llu,"
        "\"projected_intermediate_phases\":0,"
        "\"serialized_intermediate_phases\":0,"
        "\"materialized_phase_paths\":0,"
        "\"unit_modulus_carrier_invariant\":true,"
        "\"relative_phase_primitive_consumed\":true,"
        "\"multiply_cell_only_native_updates\":true,"
        "\"nonlinear_phase_coupling\":true,"
        "\"noncommuting_phase_shears\":true,"
        "\"fixed_carrier_across_depth\":true,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"same_carrier_transactions\":%u,"
        "\"coupling_disabled_control_detected\":true,"
        "\"baseline_neutralization_control_detected\":true,"
        "\"null_carrier_execution_rejected\":true,"
        "\"wrong_boundary_inverse_detected\":true,"
        "\"wrong_shear_inverse_detected\":true,"
        "\"missing_shear_inverse_detected\":true,"
        "\"reordered_noncommuting_inverse_detected\":true,"
        "\"snapshot_loaded_separately\":true,"
        "\"boolean_root_locked\":false,"
        "\"gf2_affine\":false,"
        "\"linear_u2_recurrence\":false,"
        "\"compact_classical_two_angle_recurrence_exists\":true,"
        "\"genuinely_distinct_phase_resource\":false,"
        "\"machine_enforced_hidden_intermediate\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_execution\":false,"
        "\"unlimited_catalytic_computation\":false,"
        "\"terminal\":false}\n",
        depth,
        2U * PS_CARRIER_CELLS * sizeof(double complex),
        2U * PS_CARRIER_CELLS * sizeof(double complex),
        4U * PS_CARRIER_CELLS * sizeof(double complex),
        sizeof(struct ps_projection),
        3U * sizeof(double complex)
            + 2U * sizeof(double)
            + 2U * sizeof(size_t),
        2U * sizeof(double),
        (unsigned long long)primary.projection.hash,
        (unsigned long long)reuse.projection.hash,
        (unsigned long long)disabled.projection.hash,
        (unsigned long long)neutralized.projection.hash,
        primary.projection.interference_probability,
        disabled.projection.interference_probability,
        (unsigned long long)primary.stats.forward_shears,
        (unsigned long long)primary.stats.inverse_shears,
        (unsigned long long)primary.stats.phase_source_reads,
        (unsigned long long)primary.stats.native_phase_updates,
        primary.stats.maximum_unit_modulus_error,
        repeated_max,
        wrong_boundary.restoration_max_abs,
        missing.restoration_max_abs,
        wrong_shear.restoration_max_abs,
        reordered.restoration_max_abs,
        snapshot.restoration_max_abs,
        (unsigned long long)primary.stats.final_phase_decodes,
        (unsigned int)(PS_REUSE_CYCLES + 2U)
    );
    return 0;
}
