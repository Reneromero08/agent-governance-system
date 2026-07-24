/*
 * Complete scalar adjudication of the fixed phase-native Boolean/F3
 * polynomial operators used by algebraic_cycle_phase.c.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CCOUNT 4U
#define POLYNOMIAL_COUNT 81U

static int f3(int value) {
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static int evaluate(const int c[CCOUNT], int x, int y) {
    return f3(c[0] + c[1] * x + c[2] * y + c[3] * x * y);
}

static void decode_code(size_t code, int c[CCOUNT]) {
    for (size_t index = 0U; index < CCOUNT; ++index) {
        c[index] = (int)(code % 3U);
        code /= 3U;
    }
}

static void add(
    const int left[CCOUNT],
    const int right[CCOUNT],
    int output[CCOUNT]
) {
    for (size_t index = 0U; index < CCOUNT; ++index) {
        output[index] = f3(left[index] + right[index]);
    }
}

static void multiply(
    const int left[CCOUNT],
    const int right[CCOUNT],
    int output[CCOUNT]
) {
    memset(output, 0, CCOUNT * sizeof(*output));
    for (size_t l = 0U; l < CCOUNT; ++l) {
        for (size_t r = 0U; r < CCOUNT; ++r) {
            output[l | r] = f3(
                output[l | r] + left[l] * right[r]
            );
        }
    }
}

static void exact_compose(
    const int left[CCOUNT],
    const int right[CCOUNT],
    int output[CCOUNT]
) {
    const int f0[CCOUNT] = {left[0], left[1], 0, 0};
    const int f1[CCOUNT] = {
        f3(left[0] + left[2]),
        f3(left[1] + left[3]),
        0,
        0
    };
    const int g0[CCOUNT] = {right[0], 0, right[2], 0};
    const int g1[CCOUNT] = {
        f3(right[0] + right[1]),
        0,
        f3(right[2] + right[3]),
        0
    };
    int f0s[CCOUNT];
    int g0s[CCOUNT];
    int f1s[CCOUNT];
    int g1s[CCOUNT];
    int k0[CCOUNT];
    int k1[CCOUNT];
    multiply(f0, f0, f0s);
    multiply(g0, g0, g0s);
    multiply(f1, f1, f1s);
    multiply(g1, g1, g1s);
    add(f0s, g0s, k0);
    add(f1s, g1s, k1);
    multiply(k0, k1, output);
}

static void intersection(
    const int first[CCOUNT],
    const int second[CCOUNT],
    int output[CCOUNT]
) {
    int first_squared[CCOUNT];
    int second_squared[CCOUNT];
    multiply(first, first, first_squared);
    multiply(second, second, second_squared);
    add(first_squared, second_squared, output);
}

static unsigned zero_mask(const int c[CCOUNT]) {
    unsigned mask = 0U;
    for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
            if (evaluate(c, x, y) == 0) {
                mask |= 1U << (2 * x + y);
            }
        }
    }
    return mask;
}

int main(void) {
    uint64_t composition_rows = 0U;
    uint64_t composition_mismatches = 0U;
    uint64_t intersection_rows = 0U;
    uint64_t intersection_mismatches = 0U;
    int selected = 0;
    int selected_upper[CCOUNT] = {0};
    int selected_lower[CCOUNT] = {0};
    unsigned selected_exact = 0U;
    unsigned selected_bypass = 0U;
    unsigned selected_ordinary = 0U;
    const int equality[CCOUNT] = {0, 1, 2, 0};

    for (size_t left_code = 0U; left_code < POLYNOMIAL_COUNT; ++left_code) {
        int left[CCOUNT];
        decode_code(left_code, left);
        for (size_t right_code = 0U; right_code < POLYNOMIAL_COUNT; ++right_code) {
            int right[CCOUNT];
            int composed[CCOUNT];
            int intersected[CCOUNT];
            decode_code(right_code, right);
            exact_compose(left, right, composed);
            intersection(left, right, intersected);
            for (int x = 0; x < 2; ++x) {
                for (int y = 0; y < 2; ++y) {
                    int exists = 0;
                    for (int shared = 0; shared < 2; ++shared) {
                        exists |= (
                            evaluate(left, x, shared) == 0
                            && evaluate(right, shared, y) == 0
                        );
                    }
                    composition_mismatches += (uint64_t)(
                        (evaluate(composed, x, y) == 0) != exists
                    );
                    intersection_mismatches += (uint64_t)(
                        (evaluate(intersected, x, y) == 0)
                        != (
                            evaluate(left, x, y) == 0
                            && evaluate(right, x, y) == 0
                        )
                    );
                    ++composition_rows;
                    ++intersection_rows;
                }
            }

            if (!selected) {
                int upper_message[CCOUNT];
                int lower_message[CCOUNT];
                int core[CCOUNT];
                int ordinary_core[CCOUNT];
                int upper_leaf[CCOUNT];
                int exact_boundary[CCOUNT];
                int bypass_boundary[CCOUNT];
                int ordinary_boundary[CCOUNT];
                exact_compose(left, equality, upper_message);
                exact_compose(right, equality, lower_message);
                intersection(upper_message, lower_message, core);
                add(upper_message, lower_message, ordinary_core);
                exact_compose(equality, core, upper_leaf);
                exact_compose(upper_leaf, equality, exact_boundary);
                exact_compose(equality, upper_message, upper_leaf);
                exact_compose(upper_leaf, equality, bypass_boundary);
                exact_compose(equality, ordinary_core, upper_leaf);
                exact_compose(upper_leaf, equality, ordinary_boundary);
                const unsigned exact_mask = zero_mask(exact_boundary);
                const unsigned bypass_mask = zero_mask(bypass_boundary);
                const unsigned ordinary_mask = zero_mask(ordinary_boundary);
                if (
                    exact_mask != bypass_mask
                    && exact_mask != ordinary_mask
                ) {
                    selected = 1;
                    memcpy(selected_upper, left, sizeof(selected_upper));
                    memcpy(selected_lower, right, sizeof(selected_lower));
                    selected_exact = exact_mask;
                    selected_bypass = bypass_mask;
                    selected_ordinary = ordinary_mask;
                }
            }
        }
    }
    printf(
        "{\"mode\":\"complete-exact-operator-survey\","
        "\"polynomials\":81,"
        "\"ordered_pairs\":6561,"
        "\"composition_rows\":%llu,"
        "\"composition_mismatches\":%llu,"
        "\"intersection_rows\":%llu,"
        "\"intersection_mismatches\":%llu,"
        "\"control_case_found\":%s,"
        "\"control_upper\":[%d,%d,%d,%d],"
        "\"control_lower\":[%d,%d,%d,%d],"
        "\"exact_zero_mask\":%u,"
        "\"bypass_zero_mask\":%u,"
        "\"ordinary_zero_mask\":%u}\n",
        (unsigned long long)composition_rows,
        (unsigned long long)composition_mismatches,
        (unsigned long long)intersection_rows,
        (unsigned long long)intersection_mismatches,
        selected ? "true" : "false",
        selected_upper[0],
        selected_upper[1],
        selected_upper[2],
        selected_upper[3],
        selected_lower[0],
        selected_lower[1],
        selected_lower[2],
        selected_lower[3],
        selected_exact,
        selected_bypass,
        selected_ordinary
    );
    return (
        composition_mismatches == 0U
        && intersection_mismatches == 0U
        && selected
    ) ? 0 : 1;
}
