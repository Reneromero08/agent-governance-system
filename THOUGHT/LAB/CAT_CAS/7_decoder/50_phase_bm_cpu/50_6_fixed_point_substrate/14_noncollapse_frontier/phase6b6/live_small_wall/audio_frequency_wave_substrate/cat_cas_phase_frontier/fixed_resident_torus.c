/*
 * Mutable CAT_CAS frontier: direct C implementation of the fixed-resident
 * reversible torus. This is ordinary CPU execution used to remove Python
 * interpreter overhead and measure the mechanism honestly on the lab target.
 *
 * Native evolution carries only complex relations. Discrete root decoding
 * occurs after the result latch survives the inverse traversal.
 */

#define _GNU_SOURCE
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define M 31
#define K 5
#define CHANNELS 2
#define CELLS (K * CHANNELS * M)
#define CRT_MODULUS 15015
#define RESTORATION_MAX 2.0e-11
#define PI 3.141592653589793238462643383279502884

static const int phase_moduli[K] = {3, 5, 7, 11, 13};

static inline size_t cell_index(int phase, int channel, int residue) {
    return (size_t)((phase * CHANNELS + channel) * M + residue);
}

static inline double complex unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        fprintf(stderr, "collapsed or nonfinite phase relation\n");
        exit(2);
    }
    return value / magnitude;
}

static inline double complex relation_at(
    const double complex *carrier, int phase, int residue
) {
    const double complex reference =
        carrier[cell_index(phase, 0, residue)];
    const double complex signal =
        carrier[cell_index(phase, 1, residue)];
    return unit(signal * conj(reference));
}

static inline double complex complex_power_int(
    double complex base, int exponent
) {
    double complex result = 1.0 + 0.0 * I;
    double complex factor = base;
    int remaining = exponent;
    while (remaining > 0) {
        if (remaining & 1) {
            result *= factor;
        }
        factor *= factor;
        remaining >>= 1;
    }
    return result;
}

static inline double complex phase_lock(
    double complex value, int modulus
) {
    double complex locked = unit(value);
    const double error_force =
        cimag(complex_power_int(locked, modulus));
    const double correction = error_force / (double)modulus;
    locked *= 1.0 - I * correction;
    return unit(locked);
}

static int mod_inverse(int value, int modulus) {
    for (int candidate = 1; candidate < modulus; ++candidate) {
        if ((value * candidate) % modulus == 1) {
            return candidate;
        }
    }
    fprintf(stderr, "missing modular inverse\n");
    exit(2);
}

static int public_weight(uint64_t index, int family) {
    const uint64_t linear = (uint64_t)(17 + 2 * family);
    const uint64_t quadratic = (uint64_t)(5 + family);
    return 1 + (int)(
        (index * linear + index * index * quadratic + 7U + 11U * family)
        % (M - 1)
    );
}

static void make_borrowed(
    double complex *carrier, int identity
) {
    for (int phase = 0; phase < K; ++phase) {
        for (int residue = 0; residue < M; ++residue) {
            const double angle =
                0.173
                + 0.071 * phase
                + 0.037 * residue
                + 0.023 * sin(0.17 * residue + identity * 0.07);
            const double amplitude =
                0.71
                + 0.08 * cos(0.13 * residue + 0.05 * phase);
            const double complex rail = amplitude * cexp(I * angle);
            for (int channel = 0; channel < CHANNELS; ++channel) {
                carrier[cell_index(phase, channel, residue)] =
                    rail;
            }
        }
    }
}

static void seed_carrier(double complex *carrier) {
    for (int phase = 0; phase < K; ++phase) {
        const int modulus = phase_moduli[phase];
        for (int residue = 0; residue < M; ++residue) {
            const double complex initial =
                residue == 0
                ? cexp(2.0 * PI * I / (double)modulus)
                : 1.0 + 0.0 * I;
            const double complex before =
                relation_at(carrier, phase, residue);
            if (cabs(before - 1.0) > 1.0e-12) {
                fprintf(stderr, "borrowed twin rails lost common mode\n");
                exit(2);
            }
            carrier[cell_index(phase, 1, residue)] *= initial;
        }
    }
}

static void forward_shift(double complex *carrier, int shift) {
    double complex before[K][M];
    double complex after[K][M];
    for (int phase = 0; phase < K; ++phase) {
        for (int residue = 0; residue < M; ++residue) {
            before[phase][residue] =
                relation_at(carrier, phase, residue);
        }
    }
    for (int phase = 0; phase < K; ++phase) {
        const int modulus = phase_moduli[phase];
        for (int residue = 0; residue < M; ++residue) {
            const int shifted = (residue - shift + M) % M;
            after[phase][residue] = phase_lock(
                before[phase][residue] * before[phase][shifted],
                modulus
            );
        }
    }
    for (int phase = 0; phase < K; ++phase) {
        for (int residue = 0; residue < M; ++residue) {
            carrier[cell_index(phase, 1, residue)] =
                after[phase][residue]
                * carrier[cell_index(phase, 0, residue)];
        }
    }
}

static void inverse_shift(double complex *carrier, int shift) {
    double complex before[K][M];
    double complex after[K][M];
    for (int phase = 0; phase < K; ++phase) {
        for (int residue = 0; residue < M; ++residue) {
            before[phase][residue] =
                relation_at(carrier, phase, residue);
        }
    }
    for (int phase = 0; phase < K; ++phase) {
        const int modulus = phase_moduli[phase];
        int positions[M];
        for (int index = 0; index < M; ++index) {
            positions[index] = (index * shift) % M;
        }
        double complex alternating = 1.0 + 0.0 * I;
        for (int index = 0; index < M; ++index) {
            const double complex value = before[phase][positions[index]];
            alternating *= (index == 0 || (index & 1))
                ? value
                : conj(value);
        }
        after[phase][positions[0]] = phase_lock(
            complex_power_int(
                alternating,
                mod_inverse(2, modulus)
            ),
            modulus
        );
        for (int index = 1; index < M; ++index) {
            const int current = positions[index];
            const int previous = positions[index - 1];
            after[phase][current] = phase_lock(
                before[phase][current] * conj(after[phase][previous]),
                modulus
            );
        }
    }
    for (int phase = 0; phase < K; ++phase) {
        for (int residue = 0; residue < M; ++residue) {
            carrier[cell_index(phase, 1, residue)] =
                after[phase][residue]
                * carrier[cell_index(phase, 0, residue)];
        }
    }
}

static int decode_latch(
    const double complex latch[K],
    double *maximum_root_distance
) {
    int residues[K];
    *maximum_root_distance = 0.0;
    for (int phase = 0; phase < K; ++phase) {
        const int modulus = phase_moduli[phase];
        const double complex value = unit(latch[phase]);
        double best = INFINITY;
        int best_index = -1;
        for (int candidate = 0; candidate < modulus; ++candidate) {
            const double complex root =
                cexp(2.0 * PI * I * candidate / (double)modulus);
            const double distance = cabs(root - value);
            if (distance < best) {
                best = distance;
                best_index = candidate;
            }
        }
        residues[phase] = best_index;
        if (best > *maximum_root_distance) {
            *maximum_root_distance = best;
        }
    }
    int64_t total = 0;
    for (int phase = 0; phase < K; ++phase) {
        const int modulus = phase_moduli[phase];
        const int partial = CRT_MODULUS / modulus;
        total += (int64_t)residues[phase]
            * partial
            * mod_inverse(partial % modulus, modulus);
    }
    return (int)(total % CRT_MODULUS);
}

static uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000)
        + (uint64_t)value.tv_nsec;
}

static double maximum_error(
    const double complex *left, const double complex *right
) {
    double maximum = 0.0;
    for (size_t index = 0; index < CELLS; ++index) {
        const double error = cabs(left[index] - right[index]);
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static double displacement_l2(
    const double complex *left, const double complex *right
) {
    double sum = 0.0;
    for (size_t index = 0; index < CELLS; ++index) {
        const double error = cabs(left[index] - right[index]);
        sum += error * error;
    }
    return sqrt(sum);
}

static void run_case(
    double complex *borrowed,
    uint64_t steps,
    int family,
    int target
) {
    double complex working[CELLS];
    double complex latch[K];
    memcpy(working, borrowed, sizeof(working));

    const uint64_t phase_start = monotonic_ns();
    seed_carrier(working);
    for (uint64_t index = 0; index < steps; ++index) {
        forward_shift(working, public_weight(index, family));
    }
    const double displacement = displacement_l2(working, borrowed);
    for (int phase = 0; phase < K; ++phase) {
        latch[phase] = relation_at(working, phase, target);
    }
    for (uint64_t index = steps; index-- > 0;) {
        inverse_shift(working, public_weight(index, family));
    }
    for (int phase = 0; phase < K; ++phase) {
        const int modulus = phase_moduli[phase];
        for (int residue = 0; residue < M; ++residue) {
            const double complex initial =
                residue == 0
                ? cexp(2.0 * PI * I / (double)modulus)
                : 1.0 + 0.0 * I;
            working[cell_index(phase, 1, residue)] *=
                conj(initial);
        }
    }
    const uint64_t phase_ns = monotonic_ns() - phase_start;
    const double restoration = maximum_error(working, borrowed);

    double root_distance = 0.0;
    const int phase_count = decode_latch(latch, &root_distance);

    printf(
        "{\"steps\":%llu,\"family\":%d,\"phase_count\":%d,"
        "\"phase_ns\":%llu,"
        "\"resident_complex_cells\":%d,\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,\"root_distance_max\":%.12g,"
        "\"restoration_pass\":%s}\n",
        (unsigned long long)steps,
        family,
        phase_count,
        (unsigned long long)phase_ns,
        CELLS,
        displacement,
        restoration,
        root_distance,
        restoration <= RESTORATION_MAX ? "true" : "false"
    );
    memcpy(borrowed, working, sizeof(working));
}

int main(int argc, char **argv) {
    const uint64_t sizes[] = {
        16, 64, 256, 1024, 4096, 16384, 65536
    };
    double complex carrier[CELLS];
    make_borrowed(carrier, 77);
    if (argc > 1) {
        char *end = NULL;
        const unsigned long long requested = strtoull(argv[1], &end, 10);
        if (
            end == argv[1]
            || *end != '\0'
            || requested == 0
        ) {
            fprintf(stderr, "step count must be a positive integer\n");
            return 2;
        }
        const int family = argc > 2 ? atoi(argv[2]) : 0;
        if (family < 0 || family > 1) {
            fprintf(stderr, "family must be 0 or 1\n");
            return 2;
        }
        run_case(carrier, (uint64_t)requested, family, 17);
        return 0;
    }
    for (int family = 0; family < 2; ++family) {
        for (size_t index = 0; index < sizeof(sizes) / sizeof(sizes[0]); ++index) {
            run_case(carrier, sizes[index], family, 17);
        }
    }
    return 0;
}
