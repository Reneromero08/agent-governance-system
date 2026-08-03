#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: paired-unit-phasor generic unitary kernels.
 *
 * Each complex matrix entry is the coherent average of an ordered pair of
 * unit phasors. Matrix multiplication is evaluated as 20 actual phasor path
 * products per output entry. A canonical split maps the resulting complex
 * entry back to two phases. This admits magnitude without a hidden scalar
 * carrier, but the split uses amplitude-sensitive host arithmetic and is
 * only a coordinate embedding of an ordinary complex matrix recurrence.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PP_SIDE 5U
#define PP_ENTRIES 25U
#define PP_PAIR 2U
#define PP_BLOCK_CELLS 50U
#define PP_CELLS 100U
#define PP_MAX_DEPTH 32U
#define PP_REUSE_CYCLES 8U

static const double PP_PI =
    3.14159265358979323846264338327950288;
static const double PP_CELL_TOLERANCE = 1.0e-8;
static const double PP_MATRIX_TOLERANCE = 1.0e-9;
static const double PP_UNITARY_TOLERANCE = 2.0e-12;
static const double PP_ENTRY_FLOOR = 1.0e-6;
static const double PP_CONTROL_MINIMUM = 1.0e-6;
static const double PP_HASH_SCALE = 1.0e8;

struct pp_carrier {
    double complex baseline[PP_CELLS];
    double complex working[PP_CELLS];
};

struct pp_stats {
    uint64_t native_phase_updates;
    uint64_t resident_phase_reads;
    uint64_t coherent_phasor_path_terms;
    uint64_t amplitude_sensitive_splits;
    uint64_t hidden_matrix_decodes;
    uint64_t final_matrix_decodes;
    double maximum_split_reconstruction_error;
    double maximum_matrix_reconstruction_error;
    double maximum_phase_gauge_reconstruction_error;
    double maximum_entry_modulus_overshoot;
    double minimum_nonzero_entry_modulus;
    double maximum_unit_modulus_error;
};

struct pp_boundary {
    uint64_t hash;
    double probability[PP_SIDE];
};

struct pp_run {
    struct pp_boundary boundary;
    struct pp_stats stats;
    double restoration_max_abs;
    uint64_t restoration_generation;
};

enum pp_control {
    PP_CORRECT = 0,
    PP_MISSING_INVERSE = 1,
    PP_WRONG_INVERSE = 2,
    PP_REORDERED_INVERSE = 3,
    PP_SNAPSHOT = 4
};

static void pp_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static double complex pp_unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        pp_fail("nonfinite paired phase");
    }
    return value / magnitude;
}

static struct pp_carrier pp_make_carrier(int identity) {
    struct pp_carrier carrier;
    for (size_t cell = 0U; cell < PP_CELLS; ++cell) {
        const double angle =
            0.127
            + 0.029 * (double)cell
            + 0.007 * sin(
                0.23 * (double)cell + 0.013 * (double)identity
            );
        carrier.baseline[cell] = cexp(I * angle);
        carrier.working[cell] = carrier.baseline[cell];
    }
    return carrier;
}

static double complex pp_relative(
    const struct pp_carrier *carrier,
    size_t cell
) {
    return carrier->working[cell] * conj(carrier->baseline[cell]);
}

static size_t pp_cell(
    size_t block,
    size_t row,
    size_t column,
    size_t pair
) {
    return block * PP_BLOCK_CELLS
        + (row * PP_SIDE + column) * PP_PAIR
        + pair;
}

static void pp_observe(
    const struct pp_carrier *carrier,
    struct pp_stats *stats
) {
    for (size_t cell = 0U; cell < PP_CELLS; ++cell) {
        const double error = fabs(cabs(pp_relative(carrier, cell)) - 1.0);
        if (error > stats->maximum_unit_modulus_error) {
            stats->maximum_unit_modulus_error = error;
        }
        if (error > PP_CELL_TOLERANCE) {
            pp_fail("paired-phase carrier left unit torus");
        }
    }
}

static void pp_multiply(
    struct pp_carrier *carrier,
    size_t cell,
    double complex factor,
    struct pp_stats *stats
) {
    carrier->working[cell] = pp_unit(
        pp_relative(carrier, cell) * factor
    ) * carrier->baseline[cell];
    ++stats->native_phase_updates;
}

static void pp_track_entry(double complex value, struct pp_stats *stats) {
    const double magnitude = cabs(value);
    const double overshoot = magnitude > 1.0 ? magnitude - 1.0 : 0.0;
    if (overshoot > stats->maximum_entry_modulus_overshoot) {
        stats->maximum_entry_modulus_overshoot = overshoot;
    }
    if (
        magnitude > 0.0
        && (
            stats->minimum_nonzero_entry_modulus == 0.0
            || magnitude < stats->minimum_nonzero_entry_modulus
        )
    ) {
        stats->minimum_nonzero_entry_modulus = magnitude;
    }
}

static void pp_encode(
    double complex value,
    double complex pair[PP_PAIR],
    struct pp_stats *stats,
    int require_floor
) {
    double magnitude = cabs(value);
    pp_track_entry(value, stats);
    if (magnitude > 1.0 + PP_MATRIX_TOLERANCE) {
        pp_fail("paired-phase entry exceeds unit disk");
    }
    if (magnitude > 1.0) {
        magnitude = 1.0;
    }
    if (require_floor && magnitude < PP_ENTRY_FLOOR) {
        pp_fail("paired-phase entry crossed declared gauge floor");
    }
    if (magnitude == 0.0) {
        pair[0] = 1.0 + 0.0 * I;
        pair[1] = -1.0 + 0.0 * I;
    } else {
        const double complex direction = value / cabs(value);
        const double transverse = sqrt(fmax(0.0, 1.0 - magnitude * magnitude));
        pair[0] = pp_unit(direction * (magnitude + I * transverse));
        pair[1] = pp_unit(direction * (magnitude - I * transverse));
    }
    const double complex reconstructed = (pair[0] + pair[1]) * 0.5;
    const double error = cabs(reconstructed - value);
    if (error > stats->maximum_split_reconstruction_error) {
        stats->maximum_split_reconstruction_error = error;
    }
    if (error > PP_MATRIX_TOLERANCE) {
        pp_fail("paired-phase split reconstruction failed");
    }
    ++stats->amplitude_sensitive_splits;
}

static double complex pp_root(size_t exponent) {
    return cexp(
        I * 2.0 * PP_PI * (double)(exponent % PP_SIDE)
        / (double)PP_SIDE
    );
}

static double complex pp_fourier(size_t row, size_t column) {
    return pp_root(row * column) / sqrt((double)PP_SIDE);
}

static double complex pp_kernel(
    size_t operation,
    size_t row,
    size_t column
) {
    const size_t first = operation % PP_SIDE;
    const size_t second = (first + 1U + (operation / PP_SIDE) % 4U)
        % PP_SIDE;
    const double theta = 0.19 + 0.031 * (double)(operation % 7U);
    const double phi = 0.11 + 0.047 * (double)(operation % 11U);
    const double complex diagonal_first = cexp(
        I * (
            0.071 * (double)(operation + 1U) * (double)(first + 1U)
            + 0.023 * (double)(first * first)
        )
    );
    const double complex diagonal_second = cexp(
        I * (
            0.071 * (double)(operation + 1U) * (double)(second + 1U)
            + 0.023 * (double)(second * second)
        )
    );
    const double complex first_column =
        pp_fourier(row, first) * diagonal_first;
    const double complex second_column =
        pp_fourier(row, second) * diagonal_second;
    if (column == first) {
        return
            cos(theta) * first_column
            - sin(theta) * cexp(-I * phi) * second_column;
    }
    if (column == second) {
        return
            sin(theta) * cexp(I * phi) * first_column
            + cos(theta) * second_column;
    }
    return pp_fourier(row, column) * cexp(
        I * (
            0.071 * (double)(operation + 1U) * (double)(column + 1U)
            + 0.023 * (double)(column * column)
        )
    );
}

static double pp_kernel_unitary_error(size_t operation) {
    double maximum = 0.0;
    for (size_t left = 0U; left < PP_SIDE; ++left) {
        for (size_t right = 0U; right < PP_SIDE; ++right) {
            double complex inner = 0.0 + 0.0 * I;
            for (size_t row = 0U; row < PP_SIDE; ++row) {
                inner += conj(pp_kernel(operation, row, left))
                    * pp_kernel(operation, row, right);
            }
            const double complex expected =
                left == right ? 1.0 + 0.0 * I : 0.0 + 0.0 * I;
            const double error = cabs(inner - expected);
            if (error > maximum) {
                maximum = error;
            }
        }
    }
    return maximum;
}

static void pp_require_unitary(double error) {
    if (error > PP_UNITARY_TOLERANCE) {
        pp_fail("paired-phase kernel is not unitary");
    }
}

static double complex pp_decode_entry(
    const struct pp_carrier *carrier,
    size_t block,
    size_t row,
    size_t column,
    struct pp_stats *stats
) {
    const double complex value = 0.5 * (
        pp_relative(carrier, pp_cell(block, row, column, 0U))
        + pp_relative(carrier, pp_cell(block, row, column, 1U))
    );
    stats->resident_phase_reads += 2U;
    return value;
}

static void pp_compute_product(
    const struct pp_carrier *carrier,
    size_t source_block,
    size_t operation,
    int adjoint,
    double complex output[PP_ENTRIES],
    struct pp_stats *stats
) {
    for (size_t row = 0U; row < PP_SIDE; ++row) {
        for (size_t column = 0U; column < PP_SIDE; ++column) {
            double complex sum = 0.0 + 0.0 * I;
            for (size_t shared = 0U; shared < PP_SIDE; ++shared) {
                const size_t source_column = shared;
                double complex source_pair[PP_PAIR] = {
                    pp_relative(
                        carrier,
                        pp_cell(
                            source_block, row, source_column, 0U
                        )
                    ),
                    pp_relative(
                        carrier,
                        pp_cell(
                            source_block, row, source_column, 1U
                        )
                    )
                };
                stats->resident_phase_reads += 2U;
                const double complex kernel_value = adjoint
                    ? conj(pp_kernel(operation, column, shared))
                    : pp_kernel(operation, shared, column);
                double complex kernel_pair[PP_PAIR] = {0};
                pp_encode(kernel_value, kernel_pair, stats, 1);
                for (size_t left = 0U; left < PP_PAIR; ++left) {
                    for (size_t right = 0U; right < PP_PAIR; ++right) {
                        sum += source_pair[left] * kernel_pair[right] * 0.25;
                        ++stats->coherent_phasor_path_terms;
                    }
                }
            }
            pp_track_entry(sum, stats);
            if (cabs(sum) > 1.0 + PP_MATRIX_TOLERANCE) {
                pp_fail("paired-phase product escaped unit disk");
            }
            output[row * PP_SIDE + column] = sum;
        }
    }
}

static void pp_encode_matrix(
    const double complex matrix[PP_ENTRIES],
    double complex phase[PP_BLOCK_CELLS],
    struct pp_stats *stats
) {
    for (size_t entry = 0U; entry < PP_ENTRIES; ++entry) {
        pp_encode(
            matrix[entry], &phase[entry * PP_PAIR], stats, 1
        );
    }
}

static void pp_apply_blank(
    struct pp_carrier *carrier,
    size_t block,
    const double complex phase[PP_BLOCK_CELLS],
    struct pp_stats *stats
) {
    for (size_t index = 0U; index < PP_BLOCK_CELLS; ++index) {
        const size_t cell = block * PP_BLOCK_CELLS + index;
        if (cabs(pp_relative(carrier, cell) - 1.0) > PP_CELL_TOLERANCE) {
            pp_fail("paired-phase target block is not blank");
        }
        pp_multiply(carrier, cell, phase[index], stats);
    }
}

static void pp_clear_reconstructed(
    struct pp_carrier *carrier,
    size_t block,
    const double complex matrix[PP_ENTRIES],
    const double complex phase[PP_BLOCK_CELLS],
    struct pp_stats *stats
) {
    for (size_t row = 0U; row < PP_SIDE; ++row) {
        for (size_t column = 0U; column < PP_SIDE; ++column) {
            const size_t entry = row * PP_SIDE + column;
            const double complex current = pp_decode_entry(
                carrier, block, row, column, stats
            );
            const double matrix_error = cabs(current - matrix[entry]);
            if (
                matrix_error > stats->maximum_matrix_reconstruction_error
            ) {
                stats->maximum_matrix_reconstruction_error = matrix_error;
            }
            if (matrix_error > PP_MATRIX_TOLERANCE) {
                pp_fail("paired-phase matrix reconstruction failed");
            }
            for (size_t pair = 0U; pair < PP_PAIR; ++pair) {
                const size_t cell = pp_cell(
                    block, row, column, pair
                );
                const double gauge_error = cabs(
                    pp_relative(carrier, cell)
                    - phase[entry * PP_PAIR + pair]
                );
                if (
                    gauge_error
                    > stats->maximum_phase_gauge_reconstruction_error
                ) {
                    stats->maximum_phase_gauge_reconstruction_error =
                        gauge_error;
                }
                if (gauge_error > PP_CELL_TOLERANCE) {
                    pp_fail("paired-phase gauge reconstruction failed");
                }
                pp_multiply(
                    carrier,
                    cell,
                    conj(phase[entry * PP_PAIR + pair]),
                    stats
                );
            }
        }
    }
}

static void pp_forward_step(
    struct pp_carrier *carrier,
    size_t *live_block,
    size_t operation,
    struct pp_stats *stats
) {
    const size_t target_block = 1U - *live_block;
    double complex matrix[PP_ENTRIES] = {0};
    double complex phase[PP_BLOCK_CELLS] = {0};
    pp_compute_product(
        carrier, *live_block, operation, 0, matrix, stats
    );
    pp_encode_matrix(matrix, phase, stats);
    pp_apply_blank(carrier, target_block, phase, stats);
    pp_compute_product(
        carrier, target_block, operation, 1, matrix, stats
    );
    pp_encode_matrix(matrix, phase, stats);
    pp_clear_reconstructed(
        carrier, *live_block, matrix, phase, stats
    );
    *live_block = target_block;
    pp_observe(carrier, stats);
}

static void pp_inverse_step(
    struct pp_carrier *carrier,
    size_t *live_block,
    size_t operation,
    struct pp_stats *stats
) {
    const size_t source_block = 1U - *live_block;
    double complex matrix[PP_ENTRIES] = {0};
    double complex phase[PP_BLOCK_CELLS] = {0};
    pp_compute_product(
        carrier, *live_block, operation, 1, matrix, stats
    );
    pp_encode_matrix(matrix, phase, stats);
    pp_apply_blank(carrier, source_block, phase, stats);
    pp_compute_product(
        carrier, source_block, operation, 0, matrix, stats
    );
    pp_encode_matrix(matrix, phase, stats);
    pp_clear_reconstructed(
        carrier, *live_block, matrix, phase, stats
    );
    *live_block = source_block;
    pp_observe(carrier, stats);
}

static void pp_initial_matrix(
    int program,
    double complex matrix[PP_ENTRIES]
) {
    const size_t operation = program == 0 ? 701U : 907U;
    for (size_t row = 0U; row < PP_SIDE; ++row) {
        for (size_t column = 0U; column < PP_SIDE; ++column) {
            matrix[row * PP_SIDE + column] = pp_kernel(
                operation, row, column
            );
        }
    }
}

static void pp_seal(
    struct pp_carrier *carrier,
    size_t block,
    int program,
    int inverse,
    struct pp_stats *stats
) {
    double complex matrix[PP_ENTRIES] = {0};
    double complex phase[PP_BLOCK_CELLS] = {0};
    pp_initial_matrix(program, matrix);
    pp_encode_matrix(matrix, phase, stats);
    for (size_t index = 0U; index < PP_BLOCK_CELLS; ++index) {
        pp_multiply(
            carrier,
            block * PP_BLOCK_CELLS + index,
            inverse ? conj(phase[index]) : phase[index],
            stats
        );
    }
}

static uint64_t pp_hash_component(uint64_t hash, double value) {
    const int64_t quantized = (int64_t)llround(value * PP_HASH_SCALE);
    uint64_t bits = (uint64_t)quantized;
    for (size_t byte = 0U; byte < sizeof(bits); ++byte) {
        hash ^= bits & UINT64_C(255);
        hash *= UINT64_C(1099511628211);
        bits >>= 8U;
    }
    return hash;
}

static struct pp_boundary pp_project(
    const struct pp_carrier *carrier,
    size_t live_block,
    struct pp_stats *stats
) {
    struct pp_boundary boundary = {
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t column = 0U; column < PP_SIDE; ++column) {
        double complex amplitude = 0.0 + 0.0 * I;
        for (size_t row = 0U; row < PP_SIDE; ++row) {
            const double complex value = pp_decode_entry(
                carrier, live_block, row, column, stats
            );
            boundary.hash = pp_hash_component(
                boundary.hash, creal(value)
            );
            boundary.hash = pp_hash_component(
                boundary.hash, cimag(value)
            );
            amplitude += value / sqrt((double)PP_SIDE);
            ++stats->final_matrix_decodes;
        }
        boundary.probability[column] =
            creal(amplitude * conj(amplitude));
    }
    return boundary;
}

static double pp_restoration(
    const struct pp_carrier *carrier,
    const struct pp_carrier *borrowed
) {
    double maximum = 0.0;
    for (size_t cell = 0U; cell < PP_CELLS; ++cell) {
        const double error = cabs(
            carrier->working[cell] - borrowed->working[cell]
        );
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static struct pp_run pp_execute(
    struct pp_carrier *carrier,
    int program,
    size_t depth,
    enum pp_control control
) {
    if (carrier == NULL || depth == 0U || depth > PP_MAX_DEPTH) {
        pp_fail("invalid paired-phase transaction");
    }
    const struct pp_carrier borrowed = *carrier;
    struct pp_run run = {0};
    size_t live_block = 0U;
    pp_seal(carrier, live_block, program, 0, &run.stats);
    for (size_t operation = 0U; operation < depth; ++operation) {
        pp_require_unitary(pp_kernel_unitary_error(operation));
        pp_forward_step(carrier, &live_block, operation, &run.stats);
    }
    run.boundary = pp_project(carrier, live_block, &run.stats);
    if (control == PP_SNAPSHOT) {
        *carrier = borrowed;
    } else {
        for (size_t cursor = 0U; cursor < depth; ++cursor) {
            size_t operation = depth - 1U - cursor;
            if (control == PP_REORDERED_INVERSE) {
                operation = cursor;
            } else if (control == PP_WRONG_INVERSE && cursor == 0U) {
                operation = depth;
            }
            if (
                control == PP_MISSING_INVERSE
                && cursor == depth - 1U
            ) {
                continue;
            }
            pp_inverse_step(
                carrier, &live_block, operation, &run.stats
            );
        }
        if (live_block == 0U) {
            pp_seal(carrier, live_block, program, 1, &run.stats);
        }
        if (control == PP_CORRECT) {
            run.restoration_generation = 1U;
        }
    }
    run.restoration_max_abs = pp_restoration(carrier, &borrowed);
    return run;
}

static void pp_require_correct(const struct pp_run *run) {
    if (
        run->restoration_generation != 1U
        || run->restoration_max_abs > PP_CELL_TOLERANCE
        || run->stats.hidden_matrix_decodes != 0U
        || run->stats.final_matrix_decodes != PP_ENTRIES
        || run->stats.maximum_split_reconstruction_error
            > PP_MATRIX_TOLERANCE
        || run->stats.maximum_matrix_reconstruction_error
            > PP_MATRIX_TOLERANCE
        || run->stats.maximum_phase_gauge_reconstruction_error
            > PP_CELL_TOLERANCE
        || run->stats.maximum_entry_modulus_overshoot
            > PP_MATRIX_TOLERANCE
        || run->stats.minimum_nonzero_entry_modulus < PP_ENTRY_FLOOR
    ) {
        pp_fail("paired-phase carrier law failed");
    }
}

static double pp_nonunitary_error(void) {
    double maximum = 0.0;
    for (size_t left = 0U; left < PP_SIDE; ++left) {
        for (size_t right = 0U; right < PP_SIDE; ++right) {
            double complex inner = 0.0 + 0.0 * I;
            for (size_t row = 0U; row < PP_SIDE; ++row) {
                double complex left_value = pp_kernel(0U, row, left);
                double complex right_value = pp_kernel(0U, row, right);
                if (left == 0U) {
                    left_value *= 1.01;
                }
                if (right == 0U) {
                    right_value *= 1.01;
                }
                inner += conj(left_value) * right_value;
            }
            const double complex expected =
                left == right ? 1.0 + 0.0 * I : 0.0 + 0.0 * I;
            const double error = cabs(inner - expected);
            if (error > maximum) {
                maximum = error;
            }
        }
    }
    return maximum;
}

int main(int argc, char **argv) {
    if (
        argc == 2
        && strcmp(argv[1], "--project-intermediate") == 0
    ) {
        pp_fail("active paired-phase matrix projection denied");
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        (void)pp_execute(NULL, 0, 2U, PP_CORRECT);
    }
    if (
        argc == 2
        && strcmp(argv[1], "--reject-nonunitary") == 0
    ) {
        pp_require_unitary(pp_nonunitary_error());
        pp_fail("nonunitary paired-phase kernel was accepted");
    }
    if (
        argc == 2
        && strcmp(argv[1], "--reject-expansive-entry") == 0
    ) {
        struct pp_stats stats = {0};
        double complex pair[PP_PAIR] = {0};
        pp_encode(1.01 + 0.0 * I, pair, &stats, 0);
        pp_fail("expansive paired-phase entry was accepted");
    }
    if (argc != 1) {
        pp_fail("usage: algebraic_paired_phase_unitary");
    }

    static const size_t depths[] = {1U, 2U, 4U, 8U, 16U, 32U};
    uint64_t hashes[sizeof(depths) / sizeof(depths[0])] = {0};
    struct pp_run deepest = {0};
    double maximum_restoration = 0.0;
    for (
        size_t index = 0U;
        index < sizeof(depths) / sizeof(depths[0]);
        ++index
    ) {
        struct pp_carrier carrier = pp_make_carrier(
            11000 + (int)index
        );
        const struct pp_run run = pp_execute(
            &carrier, 0, depths[index], PP_CORRECT
        );
        pp_require_correct(&run);
        hashes[index] = run.boundary.hash;
        if (run.restoration_max_abs > maximum_restoration) {
            maximum_restoration = run.restoration_max_abs;
        }
        if (depths[index] == PP_MAX_DEPTH) {
            deepest = run;
        }
    }

    struct pp_carrier reuse_carrier = pp_make_carrier(11200);
    const struct pp_run primary = pp_execute(
        &reuse_carrier, 0, 8U, PP_CORRECT
    );
    const struct pp_run unrelated = pp_execute(
        &reuse_carrier, 1, 8U, PP_CORRECT
    );
    pp_require_correct(&primary);
    pp_require_correct(&unrelated);
    if (primary.boundary.hash == unrelated.boundary.hash) {
        pp_fail("paired-phase reuse boundary unchanged");
    }
    for (size_t cycle = 0U; cycle < PP_REUSE_CYCLES; ++cycle) {
        const struct pp_run reuse = pp_execute(
            &reuse_carrier,
            cycle % 2U == 0U ? 0 : 1,
            4U,
            PP_CORRECT
        );
        pp_require_correct(&reuse);
        if (reuse.restoration_max_abs > maximum_restoration) {
            maximum_restoration = reuse.restoration_max_abs;
        }
    }

    struct pp_carrier missing_carrier = pp_make_carrier(11301);
    struct pp_carrier wrong_carrier = pp_make_carrier(11302);
    struct pp_carrier reordered_carrier = pp_make_carrier(11303);
    struct pp_carrier snapshot_carrier = pp_make_carrier(11304);
    const struct pp_run missing = pp_execute(
        &missing_carrier, 0, 4U, PP_MISSING_INVERSE
    );
    const struct pp_run wrong = pp_execute(
        &wrong_carrier, 0, 4U, PP_WRONG_INVERSE
    );
    const struct pp_run reordered = pp_execute(
        &reordered_carrier, 0, 4U, PP_REORDERED_INVERSE
    );
    const struct pp_run snapshot = pp_execute(
        &snapshot_carrier, 0, 4U, PP_SNAPSHOT
    );
    const double nonunitary_error = pp_nonunitary_error();
    if (
        missing.restoration_max_abs < PP_CONTROL_MINIMUM
        || wrong.restoration_max_abs < PP_CONTROL_MINIMUM
        || reordered.restoration_max_abs < PP_CONTROL_MINIMUM
        || snapshot.restoration_generation != 0U
        || snapshot.restoration_max_abs > PP_CELL_TOLERANCE
        || nonunitary_error <= PP_UNITARY_TOLERANCE
    ) {
        pp_fail("paired-phase control failed");
    }

    printf(
        "{\"result\":\"PASS\","
        "\"claim\":\"BOUNDED_GENERIC_5X5_UNITARY_PAIRED_UNIT_PHASOR_COHERENT_COMPOSITION_WITH_TWO_BLOCK_CUSTODY_RESTORATION_AND_REUSE\","
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
            (unsigned long long)hashes[index]
        );
    }
    printf(
        "],\"deepest_probability\":[%.17g,%.17g,%.17g,%.17g,%.17g],"
        "\"matrix_side\":5,"
        "\"phase_cells_per_complex_entry\":2,"
        "\"hidden_phase_blocks\":2,"
        "\"hidden_phase_cells\":100,"
        "\"retained_inverse_matrix_cells\":0,"
        "\"resident_phase_storage_bytes\":%zu,"
        "\"hidden_baseline_storage_bytes\":%zu,"
        "\"borrowed_verification_copy_bytes\":%zu,"
        "\"explicit_product_temporaries_bytes\":%zu,"
        "\"dense_complex_recurrence_state_bytes\":400,"
        "\"u5_coordinate_dimension_doubles\":25,"
        "\"u5_coordinate_lower_bound_bytes_excluding_chart_metadata\":200,"
        "\"native_phase_updates_at_depth32\":%llu,"
        "\"resident_phase_reads_at_depth32\":%llu,"
        "\"coherent_phasor_path_terms_at_depth32\":%llu,"
        "\"amplitude_sensitive_splits_at_depth32\":%llu,"
        "\"hidden_matrix_decodes\":%llu,"
        "\"final_matrix_decodes\":%llu,"
        "\"maximum_split_reconstruction_error\":%.17g,"
        "\"maximum_matrix_reconstruction_error\":%.17g,"
        "\"maximum_phase_gauge_reconstruction_error\":%.17g,"
        "\"maximum_entry_modulus_overshoot\":%.17g,"
        "\"minimum_nonzero_entry_modulus\":%.17g,"
        "\"declared_entry_floor\":%.17g,"
        "\"cell_restoration_tolerance\":%.17g,"
        "\"matrix_reconstruction_tolerance\":%.17g,"
        "\"unitary_tolerance\":%.17g,"
        "\"maximum_repeated_restoration_error\":%.17g,"
        "\"missing_inverse_residual\":%.17g,"
        "\"wrong_inverse_residual\":%.17g,"
        "\"reordered_inverse_residual\":%.17g,"
        "\"snapshot_residual\":%.17g,"
        "\"nonunitary_control_error\":%.17g,"
        "\"nonunitary_control_rejected\":true,"
        "\"expansive_entry_control_rejected\":true,"
        "\"unrelated_reuse_depth\":8,"
        "\"following_reuse_cycles\":8,"
        "\"actual_paired_phase_coherent_product\":true,"
        "\"actual_adjoint_source_uncompute\":true,"
        "\"generic_off_chirp_unitary_closure\":true,"
        "\"fixed_rank_across_tested_depth\":true,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"amplitude_sensitive_host_split_required\":true,"
        "\"phase_only_without_amplitude_computation\":false,"
        "\"globally_stable_arbitrary_unitary_closure\":false,"
        "\"compact_complex_matrix_recurrence_exists\":true,"
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
        PP_CELLS * sizeof(double complex),
        PP_CELLS * sizeof(double complex),
        sizeof(struct pp_carrier),
        PP_ENTRIES * sizeof(double complex)
            + PP_BLOCK_CELLS * sizeof(double complex),
        (unsigned long long)deepest.stats.native_phase_updates,
        (unsigned long long)deepest.stats.resident_phase_reads,
        (unsigned long long)deepest.stats.coherent_phasor_path_terms,
        (unsigned long long)deepest.stats.amplitude_sensitive_splits,
        (unsigned long long)deepest.stats.hidden_matrix_decodes,
        (unsigned long long)deepest.stats.final_matrix_decodes,
        deepest.stats.maximum_split_reconstruction_error,
        deepest.stats.maximum_matrix_reconstruction_error,
        deepest.stats.maximum_phase_gauge_reconstruction_error,
        deepest.stats.maximum_entry_modulus_overshoot,
        deepest.stats.minimum_nonzero_entry_modulus,
        PP_ENTRY_FLOOR,
        PP_CELL_TOLERANCE,
        PP_MATRIX_TOLERANCE,
        PP_UNITARY_TOLERANCE,
        maximum_restoration,
        missing.restoration_max_abs,
        wrong.restoration_max_abs,
        reordered.restoration_max_abs,
        snapshot.restoration_max_abs,
        nonunitary_error
    );
    return 0;
}
