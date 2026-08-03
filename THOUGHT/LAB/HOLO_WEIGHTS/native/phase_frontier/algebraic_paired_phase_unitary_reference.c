/*
 * Independent dense-complex recurrence for the paired-phase unitary fixture.
 * It has no phase pairs, live/blank custody, inverse, restoration, or reuse.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define PR_SIDE 5U
#define PR_ENTRIES 25U

static const double PR_PI =
    3.14159265358979323846264338327950288;
static const double PR_HASH_SCALE = 1.0e8;

static double complex pr_root(size_t exponent) {
    return cexp(
        I * 2.0 * PR_PI * (double)(exponent % PR_SIDE)
        / (double)PR_SIDE
    );
}

static double complex pr_fourier(size_t row, size_t column) {
    return pr_root(row * column) / sqrt((double)PR_SIDE);
}

static double complex pr_kernel(
    size_t operation,
    size_t row,
    size_t column
) {
    const size_t first = operation % PR_SIDE;
    const size_t second = (first + 1U + (operation / PR_SIDE) % 4U)
        % PR_SIDE;
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
        pr_fourier(row, first) * diagonal_first;
    const double complex second_column =
        pr_fourier(row, second) * diagonal_second;
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
    return pr_fourier(row, column) * cexp(
        I * (
            0.071 * (double)(operation + 1U) * (double)(column + 1U)
            + 0.023 * (double)(column * column)
        )
    );
}

static void pr_initial(double complex matrix[PR_ENTRIES]) {
    for (size_t row = 0U; row < PR_SIDE; ++row) {
        for (size_t column = 0U; column < PR_SIDE; ++column) {
            matrix[row * PR_SIDE + column] = pr_kernel(
                701U, row, column
            );
        }
    }
}

static void pr_step(
    double complex matrix[PR_ENTRIES],
    size_t operation
) {
    double complex output[PR_ENTRIES] = {0};
    for (size_t row = 0U; row < PR_SIDE; ++row) {
        for (size_t column = 0U; column < PR_SIDE; ++column) {
            for (size_t shared = 0U; shared < PR_SIDE; ++shared) {
                output[row * PR_SIDE + column] +=
                    matrix[row * PR_SIDE + shared]
                    * pr_kernel(operation, shared, column);
            }
        }
    }
    for (size_t entry = 0U; entry < PR_ENTRIES; ++entry) {
        matrix[entry] = output[entry];
    }
}

static uint64_t pr_hash_component(uint64_t hash, double value) {
    const int64_t quantized = (int64_t)llround(value * PR_HASH_SCALE);
    uint64_t bits = (uint64_t)quantized;
    for (size_t byte = 0U; byte < sizeof(bits); ++byte) {
        hash ^= bits & UINT64_C(255);
        hash *= UINT64_C(1099511628211);
        bits >>= 8U;
    }
    return hash;
}

static uint64_t pr_boundary(
    const double complex matrix[PR_ENTRIES],
    double probability[PR_SIDE]
) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t column = 0U; column < PR_SIDE; ++column) {
        double complex amplitude = 0.0 + 0.0 * I;
        for (size_t row = 0U; row < PR_SIDE; ++row) {
            const double complex value =
                matrix[row * PR_SIDE + column];
            hash = pr_hash_component(hash, creal(value));
            hash = pr_hash_component(hash, cimag(value));
            amplitude += value / sqrt((double)PR_SIDE);
        }
        probability[column] =
            creal(amplitude * conj(amplitude));
    }
    return hash;
}

int main(void) {
    static const size_t depths[] = {1U, 2U, 4U, 8U, 16U, 32U};
    uint64_t hashes[sizeof(depths) / sizeof(depths[0])] = {0};
    double deepest_probability[PR_SIDE] = {0};
    for (
        size_t depth_index = 0U;
        depth_index < sizeof(depths) / sizeof(depths[0]);
        ++depth_index
    ) {
        double complex matrix[PR_ENTRIES] = {0};
        pr_initial(matrix);
        for (
            size_t operation = 0U;
            operation < depths[depth_index];
            ++operation
        ) {
            pr_step(matrix, operation);
        }
        double probability[PR_SIDE] = {0};
        hashes[depth_index] = pr_boundary(matrix, probability);
        if (depth_index + 1U == sizeof(depths) / sizeof(depths[0])) {
            for (size_t index = 0U; index < PR_SIDE; ++index) {
                deepest_probability[index] = probability[index];
            }
        }
    }
    printf("{\"result\":\"PASS\",\"tested_depths\":[");
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
        "\"dense_complex_live_state_bytes\":%zu,"
        "\"streaming_row_scratch_bytes\":%zu,"
        "\"dense_complex_multiply_adds_at_depth32\":%zu,"
        "\"u5_coordinate_dimension_doubles\":25,"
        "\"u5_coordinate_lower_bound_bytes_excluding_chart_metadata\":200,"
        "\"retained_intermediate_matrices\":0}\n",
        deepest_probability[0],
        deepest_probability[1],
        deepest_probability[2],
        deepest_probability[3],
        deepest_probability[4],
        PR_ENTRIES * sizeof(double complex),
        PR_SIDE * sizeof(double complex),
        (size_t)32U * PR_SIDE * PR_SIDE * PR_SIDE
    );
    return 0;
}
