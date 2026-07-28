#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: coherent F5 quadratic phase-kernel contraction.
 *
 * One live 5x5 unit-phase kernel contracts with a public nondegenerate
 * quadratic module through an actual coherent shared-port sum. A second
 * 5x5 block receives the result; the result and public adjoint reconstruct
 * and clear the source. Reverse contractions restore the source in exact
 * reverse custody order. No intermediate kernel or compact signature is
 * decoded, retained, or serialized.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GK_FIELD 5U
#define GK_BLOCK_CELLS 25U
#define GK_CELLS 50U
#define GK_MAX_DEPTH 2048U
#define GK_REUSE_CYCLES 8U

static const double GK_PI =
    3.14159265358979323846264338327950288;
static const double GK_UNIT_TOLERANCE = 2.0e-11;
static const double GK_RESTORE_TOLERANCE = 2.0e-10;
static const double GK_CONTROL_MINIMUM = 1.0e-6;
static const double GK_PERTURBATION = 0x1p-20;

struct gk_signature {
    int mu;
    int q[6];
};

struct gk_carrier {
    double complex baseline[GK_CELLS];
    double complex working[GK_CELLS];
};

struct gk_stats {
    uint64_t native_phase_updates;
    uint64_t resident_phase_reads;
    uint64_t coherent_path_terms;
    uint64_t hidden_kernel_decodes;
    uint64_t final_phase_decodes;
    double maximum_raw_modulus_error;
    double maximum_reconstruction_error;
    double maximum_unit_modulus_error;
};

struct gk_boundary {
    uint64_t hash;
    double probability[GK_FIELD];
};

struct gk_run {
    struct gk_boundary boundary;
    struct gk_stats stats;
    double restoration_max_abs;
    uint64_t restoration_generation;
};

enum gk_control {
    GK_CORRECT = 0,
    GK_MISSING_INVERSE = 1,
    GK_WRONG_INVERSE = 2,
    GK_REORDERED_INVERSE = 3,
    GK_SNAPSHOT = 4
};

enum gk_kernel_class {
    GK_EXACT_CHIRP = 0,
    GK_PERTURBED_CHIRP = 1,
    GK_GENERIC_PHASE = 2
};

static void gk_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static int gk_mod(int value) {
    int result = value % (int)GK_FIELD;
    return result < 0 ? result + (int)GK_FIELD : result;
}

static double complex gk_unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        gk_fail("nonfinite Gauss-kernel phase");
    }
    return value / magnitude;
}

static double complex gk_root(int value) {
    return cexp(
        I * 2.0 * GK_PI * (double)gk_mod(value) / (double)GK_FIELD
    );
}

static struct gk_carrier gk_make_carrier(int identity) {
    struct gk_carrier carrier;
    for (size_t cell = 0U; cell < GK_CELLS; ++cell) {
        const double angle =
            0.113
            + 0.037 * (double)cell
            + 0.009 * sin(
                0.31 * (double)cell + 0.019 * (double)identity
            );
        carrier.baseline[cell] = cexp(I * angle);
        carrier.working[cell] = carrier.baseline[cell];
    }
    return carrier;
}

static double complex gk_relative(
    const struct gk_carrier *carrier,
    size_t cell
) {
    return carrier->working[cell] * conj(carrier->baseline[cell]);
}

static void gk_observe(
    const struct gk_carrier *carrier,
    struct gk_stats *stats
) {
    for (size_t cell = 0U; cell < GK_CELLS; ++cell) {
        const double error = fabs(cabs(gk_relative(carrier, cell)) - 1.0);
        if (error > stats->maximum_unit_modulus_error) {
            stats->maximum_unit_modulus_error = error;
        }
        if (error > GK_UNIT_TOLERANCE) {
            gk_fail("Gauss-kernel carrier left unit torus");
        }
    }
}

static void gk_multiply(
    struct gk_carrier *carrier,
    size_t cell,
    double complex factor,
    struct gk_stats *stats
) {
    carrier->working[cell] = gk_unit(
        gk_relative(carrier, cell) * factor
    ) * carrier->baseline[cell];
    ++stats->native_phase_updates;
}

static size_t gk_cell(size_t block, size_t row, size_t column) {
    return block * GK_BLOCK_CELLS + row * GK_FIELD + column;
}

static struct gk_signature gk_initial_signature(int program) {
    const struct gk_signature primary = {
        .mu = 1,
        .q = {1, 1, 2, 1, 0, 3}
    };
    const struct gk_signature reuse = {
        .mu = -1,
        .q = {3, 2, 2, 4, 1, 0}
    };
    return program == 0 ? primary : reuse;
}

static struct gk_signature gk_module_signature(size_t operation) {
    struct gk_signature module = {
        .mu = operation % 3U == 0U ? -1 : 1,
        .q = {
            4,
            1,
            1,
            (int)((operation + 2U) % GK_FIELD),
            (int)((2U * operation + 1U) % GK_FIELD),
            (int)((3U * operation) % GK_FIELD)
        }
    };
    return module;
}

static int gk_polynomial(
    const struct gk_signature *signature,
    int x,
    int y
) {
    return gk_mod(
        signature->q[0] * x * x
        + signature->q[1] * x * y
        + signature->q[2] * y * y
        + signature->q[3] * x
        + signature->q[4] * y
        + signature->q[5]
    );
}

static double complex gk_signature_phase(
    const struct gk_signature *signature,
    int x,
    int y
) {
    return (double)signature->mu
        * gk_root(gk_polynomial(signature, x, y));
}

static double complex gk_module_phase(
    size_t operation,
    size_t row,
    size_t column,
    enum gk_kernel_class kernel_class
) {
    const struct gk_signature module = gk_module_signature(operation);
    double complex phase = gk_signature_phase(
        &module, (int)row, (int)column
    );
    if (
        kernel_class == GK_PERTURBED_CHIRP
        && operation == 0U
        && row == 2U
        && column == 3U
    ) {
        phase *= cexp(I * GK_PERTURBATION);
    } else if (kernel_class == GK_GENERIC_PHASE) {
        const double angle =
            0.173
            + 0.419 * (double)row
            + 0.271 * (double)column
            + 0.067 * (double)(row * column)
            + 0.031 * (double)operation;
        phase = cexp(I * angle);
    }
    return phase;
}

static double gk_contract(
    const struct gk_carrier *carrier,
    size_t source_block,
    size_t operation,
    int adjoint,
    enum gk_kernel_class kernel_class,
    double complex output[GK_BLOCK_CELLS],
    struct gk_stats *stats,
    int require_closure
) {
    double maximum_error = 0.0;
    const double inverse_sqrt_field = 1.0 / sqrt((double)GK_FIELD);
    for (size_t x = 0U; x < GK_FIELD; ++x) {
        for (size_t z = 0U; z < GK_FIELD; ++z) {
            double complex sum = 0.0 + 0.0 * I;
            for (size_t shared = 0U; shared < GK_FIELD; ++shared) {
                const size_t source = adjoint
                    ? gk_cell(source_block, x, shared)
                    : gk_cell(source_block, x, shared);
                const double complex module = adjoint
                    ? conj(gk_module_phase(
                        operation, z, shared, kernel_class
                    ))
                    : gk_module_phase(
                        operation, shared, z, kernel_class
                    );
                sum += gk_relative(carrier, source) * module;
                ++stats->resident_phase_reads;
                ++stats->coherent_path_terms;
            }
            const double complex raw = sum * inverse_sqrt_field;
            const double error = fabs(cabs(raw) - 1.0);
            if (error > maximum_error) {
                maximum_error = error;
            }
            if (error > stats->maximum_raw_modulus_error) {
                stats->maximum_raw_modulus_error = error;
            }
            if (require_closure && error > GK_UNIT_TOLERANCE) {
                gk_fail("Gauss-kernel fixed-rank closure failed");
            }
            output[x * GK_FIELD + z] = gk_unit(raw);
        }
    }
    return maximum_error;
}

static void gk_apply_blank(
    struct gk_carrier *carrier,
    size_t block,
    const double complex phase[GK_BLOCK_CELLS],
    struct gk_stats *stats
) {
    for (size_t index = 0U; index < GK_BLOCK_CELLS; ++index) {
        const size_t cell = block * GK_BLOCK_CELLS + index;
        if (cabs(gk_relative(carrier, cell) - 1.0) > GK_RESTORE_TOLERANCE) {
            gk_fail("Gauss-kernel target block is not blank");
        }
        gk_multiply(carrier, cell, phase[index], stats);
    }
}

static void gk_clear_reconstructed(
    struct gk_carrier *carrier,
    size_t block,
    const double complex reconstructed[GK_BLOCK_CELLS],
    struct gk_stats *stats
) {
    for (size_t index = 0U; index < GK_BLOCK_CELLS; ++index) {
        const size_t cell = block * GK_BLOCK_CELLS + index;
        const double error = cabs(
            gk_relative(carrier, cell) - reconstructed[index]
        );
        if (error > stats->maximum_reconstruction_error) {
            stats->maximum_reconstruction_error = error;
        }
        if (error > GK_RESTORE_TOLERANCE) {
            gk_fail("Gauss-kernel adjoint reconstruction failed");
        }
        gk_multiply(carrier, cell, conj(reconstructed[index]), stats);
    }
}

static void gk_forward_step(
    struct gk_carrier *carrier,
    size_t *live_block,
    size_t operation,
    struct gk_stats *stats
) {
    const size_t target_block = 1U - *live_block;
    double complex output[GK_BLOCK_CELLS] = {0};
    double complex reconstruction[GK_BLOCK_CELLS] = {0};
    (void)gk_contract(
        carrier,
        *live_block,
        operation,
        0,
        GK_EXACT_CHIRP,
        output,
        stats,
        1
    );
    gk_apply_blank(carrier, target_block, output, stats);
    (void)gk_contract(
        carrier,
        target_block,
        operation,
        1,
        GK_EXACT_CHIRP,
        reconstruction,
        stats,
        1
    );
    gk_clear_reconstructed(
        carrier, *live_block, reconstruction, stats
    );
    *live_block = target_block;
    gk_observe(carrier, stats);
}

static void gk_inverse_step(
    struct gk_carrier *carrier,
    size_t *live_block,
    size_t operation,
    struct gk_stats *stats
) {
    const size_t source_block = 1U - *live_block;
    double complex source[GK_BLOCK_CELLS] = {0};
    double complex reconstruction[GK_BLOCK_CELLS] = {0};
    (void)gk_contract(
        carrier,
        *live_block,
        operation,
        1,
        GK_EXACT_CHIRP,
        source,
        stats,
        1
    );
    gk_apply_blank(carrier, source_block, source, stats);
    (void)gk_contract(
        carrier,
        source_block,
        operation,
        0,
        GK_EXACT_CHIRP,
        reconstruction,
        stats,
        1
    );
    gk_clear_reconstructed(
        carrier, *live_block, reconstruction, stats
    );
    *live_block = source_block;
    gk_observe(carrier, stats);
}

static void gk_seal(
    struct gk_carrier *carrier,
    size_t block,
    int program,
    int inverse,
    struct gk_stats *stats
) {
    const struct gk_signature signature = gk_initial_signature(program);
    for (size_t x = 0U; x < GK_FIELD; ++x) {
        for (size_t y = 0U; y < GK_FIELD; ++y) {
            double complex phase = gk_signature_phase(
                &signature, (int)x, (int)y
            );
            if (inverse) {
                phase = conj(phase);
            }
            gk_multiply(
                carrier, gk_cell(block, x, y), phase, stats
            );
        }
    }
}

static int gk_decode_phase(double complex phase, struct gk_stats *stats) {
    int selected = 0;
    double minimum = HUGE_VAL;
    for (int code = 0; code < 10; ++code) {
        const double complex candidate =
            code < 5 ? gk_root(code) : -gk_root(code - 5);
        const double error = cabs(phase - candidate);
        if (error < minimum) {
            minimum = error;
            selected = code;
        }
    }
    if (minimum > 2.0e-9) {
        gk_fail("final Gauss-kernel phase is not on chirp alphabet");
    }
    ++stats->final_phase_decodes;
    return selected;
}

static struct gk_boundary gk_project(
    const struct gk_carrier *carrier,
    size_t live_block,
    struct gk_stats *stats
) {
    struct gk_boundary boundary = {
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t z = 0U; z < GK_FIELD; ++z) {
        double complex amplitude = 0.0 + 0.0 * I;
        for (size_t x = 0U; x < GK_FIELD; ++x) {
            const double complex phase = gk_relative(
                carrier, gk_cell(live_block, x, z)
            );
            const int code = gk_decode_phase(phase, stats);
            boundary.hash ^= (uint64_t)code;
            boundary.hash *= UINT64_C(1099511628211);
            amplitude += phase / (double)GK_FIELD;
        }
        boundary.probability[z] =
            creal(amplitude * conj(amplitude));
    }
    return boundary;
}

static double gk_restoration(
    const struct gk_carrier *carrier,
    const struct gk_carrier *borrowed
) {
    double maximum = 0.0;
    for (size_t cell = 0U; cell < GK_CELLS; ++cell) {
        const double error = cabs(
            carrier->working[cell] - borrowed->working[cell]
        );
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static struct gk_run gk_execute(
    struct gk_carrier *carrier,
    int program,
    size_t depth,
    enum gk_control control
) {
    if (carrier == NULL || depth == 0U || depth > GK_MAX_DEPTH) {
        gk_fail("invalid Gauss-kernel transaction");
    }
    const struct gk_carrier borrowed = *carrier;
    struct gk_run run = {0};
    size_t live_block = 0U;
    gk_seal(carrier, live_block, program, 0, &run.stats);
    for (size_t operation = 0U; operation < depth; ++operation) {
        gk_forward_step(carrier, &live_block, operation, &run.stats);
    }
    run.boundary = gk_project(carrier, live_block, &run.stats);
    if (control == GK_SNAPSHOT) {
        *carrier = borrowed;
    } else {
        for (size_t cursor = 0U; cursor < depth; ++cursor) {
            size_t operation = depth - 1U - cursor;
            if (control == GK_REORDERED_INVERSE) {
                operation = cursor;
            } else if (control == GK_WRONG_INVERSE && cursor == 0U) {
                operation = depth;
            }
            if (
                control == GK_MISSING_INVERSE
                && cursor == depth - 1U
            ) {
                continue;
            }
            gk_inverse_step(
                carrier, &live_block, operation, &run.stats
            );
        }
        if (live_block == 0U) {
            gk_seal(carrier, live_block, program, 1, &run.stats);
        }
        if (control == GK_CORRECT) {
            run.restoration_generation = 1U;
        }
    }
    run.restoration_max_abs = gk_restoration(carrier, &borrowed);
    return run;
}

static double gk_off_manifold(
    enum gk_kernel_class kernel_class
) {
    struct gk_carrier carrier = gk_make_carrier(
        kernel_class == GK_PERTURBED_CHIRP ? 9211 : 9212
    );
    struct gk_stats stats = {0};
    double complex output[GK_BLOCK_CELLS] = {0};
    gk_seal(&carrier, 0U, 0, 0, &stats);
    return gk_contract(
        &carrier,
        0U,
        0U,
        0,
        kernel_class,
        output,
        &stats,
        0
    );
}

static void gk_require_correct(const struct gk_run *run) {
    if (
        run->restoration_generation != 1U
        || run->restoration_max_abs > GK_RESTORE_TOLERANCE
        || run->stats.hidden_kernel_decodes != 0U
        || run->stats.final_phase_decodes != GK_BLOCK_CELLS
        || run->stats.maximum_raw_modulus_error > GK_UNIT_TOLERANCE
        || run->stats.maximum_reconstruction_error
            > GK_RESTORE_TOLERANCE
    ) {
        gk_fail("Gauss-kernel carrier law failed");
    }
}

int main(int argc, char **argv) {
    if (
        argc == 2
        && strcmp(argv[1], "--project-intermediate") == 0
    ) {
        gk_fail("resident Gauss kernel projection denied");
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        (void)gk_execute(NULL, 0, 2U, GK_CORRECT);
    }
    if (argc != 1) {
        gk_fail("usage: algebraic_f5_gauss_kernel_phase");
    }

    static const size_t depths[] = {2U, 4U, 8U, 32U, 128U, 512U, 2048U};
    uint64_t depth_hashes[
        sizeof(depths) / sizeof(depths[0])
    ];
    double maximum_restoration = 0.0;
    struct gk_run deepest = {0};
    for (
        size_t index = 0U;
        index < sizeof(depths) / sizeof(depths[0]);
        ++index
    ) {
        struct gk_carrier carrier = gk_make_carrier(
            9000 + (int)index
        );
        const struct gk_run run = gk_execute(
            &carrier, 0, depths[index], GK_CORRECT
        );
        gk_require_correct(&run);
        depth_hashes[index] = run.boundary.hash;
        if (run.restoration_max_abs > maximum_restoration) {
            maximum_restoration = run.restoration_max_abs;
        }
        if (depths[index] == GK_MAX_DEPTH) {
            deepest = run;
        }
    }

    struct gk_carrier reuse_carrier = gk_make_carrier(9100);
    const struct gk_run primary = gk_execute(
        &reuse_carrier, 0, 128U, GK_CORRECT
    );
    const struct gk_run unrelated = gk_execute(
        &reuse_carrier, 1, 128U, GK_CORRECT
    );
    gk_require_correct(&primary);
    gk_require_correct(&unrelated);
    if (primary.boundary.hash == unrelated.boundary.hash) {
        gk_fail("Gauss-kernel reuse program did not change boundary");
    }
    for (size_t cycle = 0U; cycle < GK_REUSE_CYCLES; ++cycle) {
        const struct gk_run reuse = gk_execute(
            &reuse_carrier,
            cycle % 2U == 0U ? 0 : 1,
            32U,
            GK_CORRECT
        );
        gk_require_correct(&reuse);
        if (reuse.restoration_max_abs > maximum_restoration) {
            maximum_restoration = reuse.restoration_max_abs;
        }
    }

    struct gk_carrier missing_carrier = gk_make_carrier(9301);
    struct gk_carrier wrong_carrier = gk_make_carrier(9302);
    struct gk_carrier reordered_carrier = gk_make_carrier(9303);
    struct gk_carrier snapshot_carrier = gk_make_carrier(9304);
    const struct gk_run missing = gk_execute(
        &missing_carrier, 0, 4U, GK_MISSING_INVERSE
    );
    const struct gk_run wrong = gk_execute(
        &wrong_carrier, 0, 4U, GK_WRONG_INVERSE
    );
    const struct gk_run reordered = gk_execute(
        &reordered_carrier, 0, 4U, GK_REORDERED_INVERSE
    );
    const struct gk_run snapshot = gk_execute(
        &snapshot_carrier, 0, 4U, GK_SNAPSHOT
    );
    const double perturbed_error = gk_off_manifold(
        GK_PERTURBED_CHIRP
    );
    const double generic_error = gk_off_manifold(GK_GENERIC_PHASE);
    if (
        missing.restoration_max_abs < GK_CONTROL_MINIMUM
        || wrong.restoration_max_abs < GK_CONTROL_MINIMUM
        || reordered.restoration_max_abs < GK_CONTROL_MINIMUM
        || snapshot.restoration_generation != 0U
        || snapshot.restoration_max_abs > GK_RESTORE_TOLERANCE
        || perturbed_error <= GK_UNIT_TOLERANCE
        || generic_error <= GK_UNIT_TOLERANCE
    ) {
        gk_fail("Gauss-kernel control failed");
    }

    printf(
        "{\"result\":\"PASS\","
        "\"claim\":\"BOUNDED_F5_QUADRATIC_PHASE_KERNEL_COHERENT_SHARED_PORT_GAUSS_CLOSURE_WITH_FIXED_TWO_BLOCK_CUSTODY_RESTORATION_AND_REUSE\","
        "\"tested_depths\":["
    );
    for (
        size_t index = 0U;
        index < sizeof(depths) / sizeof(depths[0]);
        ++index
    ) {
        printf("%s%zu", index == 0U ? "" : ",", depths[index]);
    }
    printf("],\"final_boundary_hashes\":[");
    for (
        size_t index = 0U;
        index < sizeof(depths) / sizeof(depths[0]);
        ++index
    ) {
        printf(
            "%s\"%016llx\"",
            index == 0U ? "" : ",",
            (unsigned long long)depth_hashes[index]
        );
    }
    printf(
        "],\"deepest_probability\":[%.17g,%.17g,%.17g,%.17g,%.17g],"
        "\"phase_kernel_side\":5,"
        "\"hidden_phase_blocks\":2,"
        "\"hidden_phase_cells\":50,"
        "\"retained_inverse_kernel_cells\":0,"
        "\"resident_phase_storage_bytes\":%zu,"
        "\"hidden_baseline_storage_bytes\":%zu,"
        "\"borrowed_verification_copy_bytes\":%zu,"
        "\"explicit_contraction_temporaries_bytes\":%zu,"
        "\"native_phase_updates_at_depth2048\":%llu,"
        "\"resident_phase_reads_at_depth2048\":%llu,"
        "\"coherent_path_terms_at_depth2048\":%llu,"
        "\"hidden_kernel_decodes\":%llu,"
        "\"final_phase_decodes\":%llu,"
        "\"maximum_raw_modulus_error\":%.17g,"
        "\"maximum_reconstruction_error\":%.17g,"
        "\"maximum_repeated_restoration_error\":%.17g,"
        "\"missing_inverse_residual\":%.17g,"
        "\"wrong_inverse_residual\":%.17g,"
        "\"reordered_inverse_residual\":%.17g,"
        "\"snapshot_residual\":%.17g,"
        "\"perturbed_raw_modulus_error\":%.17g,"
        "\"generic_raw_modulus_error\":%.17g,"
        "\"off_manifold_perturbation_radians\":%.17g,"
        "\"actual_coherent_shared_port_sum\":true,"
        "\"actual_adjoint_source_uncompute\":true,"
        "\"fixed_rank_across_tested_depth\":true,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"off_manifold_fixed_rank_closure\":false,"
        "\"compact_weil_recurrence_exists\":true,"
        "\"genuinely_distinct_phase_resource\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_execution\":false,"
        "\"unlimited_catalytic_computation\":false,"
        "\"terminal\":false}\n",
        deepest.boundary.probability[0],
        deepest.boundary.probability[1],
        deepest.boundary.probability[2],
        deepest.boundary.probability[3],
        deepest.boundary.probability[4],
        GK_CELLS * sizeof(double complex),
        GK_CELLS * sizeof(double complex),
        sizeof(struct gk_carrier),
        2U * GK_BLOCK_CELLS * sizeof(double complex),
        (unsigned long long)deepest.stats.native_phase_updates,
        (unsigned long long)deepest.stats.resident_phase_reads,
        (unsigned long long)deepest.stats.coherent_path_terms,
        (unsigned long long)deepest.stats.hidden_kernel_decodes,
        (unsigned long long)deepest.stats.final_phase_decodes,
        deepest.stats.maximum_raw_modulus_error,
        deepest.stats.maximum_reconstruction_error,
        maximum_restoration,
        missing.restoration_max_abs,
        wrong.restoration_max_abs,
        reordered.restoration_max_abs,
        snapshot.restoration_max_abs,
        perturbed_error,
        generic_error,
        GK_PERTURBATION
    );
    return 0;
}
