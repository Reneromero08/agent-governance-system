/*
 * Exhaustive scalar theorem survey for the fixed two-hub Boolean_F3 tree.
 * This executable is independent from the native phase engine.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define COEFFICIENT_COUNT 4U
#define MAX_SIGNATURES 81U
#define BOUNDARY_COUNT 6U

static int f3(int value) {
    int normalized = value % 3;
    if (normalized < 0) {
        normalized += 3;
    }
    return normalized;
}

static int evaluate(
    const int coefficient[COEFFICIENT_COUNT],
    int first,
    int second
) {
    return f3(
        coefficient[0]
        + coefficient[1] * first
        + coefficient[2] * second
        + coefficient[3] * first * second
    );
}

static int bi_total(const int coefficient[COEFFICIENT_COUNT]) {
    for (int first = 0; first < 2; ++first) {
        int found = 0;
        for (int second = 0; second < 2; ++second) {
            found |= evaluate(coefficient, first, second) == 0;
        }
        if (!found) {
            return 0;
        }
    }
    for (int second = 0; second < 2; ++second) {
        int found = 0;
        for (int first = 0; first < 2; ++first) {
            found |= evaluate(coefficient, first, second) == 0;
        }
        if (!found) {
            return 0;
        }
    }
    return 1;
}

static uint8_t relation_mask(
    const int coefficient[COEFFICIENT_COUNT]
) {
    uint8_t mask = 0U;
    for (int first = 0; first < 2; ++first) {
        for (int second = 0; second < 2; ++second) {
            if (evaluate(coefficient, first, second) == 0) {
                mask |= (uint8_t)(1U << (2 * first + second));
            }
        }
    }
    return mask;
}

static void decode_signature(
    size_t ordinal,
    int coefficient[COEFFICIENT_COUNT]
) {
    for (size_t index = 0U; index < COEFFICIENT_COUNT; ++index) {
        coefficient[index] = (int)(ordinal % 3U);
        ordinal /= 3U;
    }
}

static void transpose(
    int output[COEFFICIENT_COUNT],
    const int input[COEFFICIENT_COUNT]
) {
    output[0] = input[0];
    output[1] = input[2];
    output[2] = input[1];
    output[3] = input[3];
}

static void resultant(
    int output[COEFFICIENT_COUNT],
    const int left[COEFFICIENT_COUNT],
    const int right[COEFFICIENT_COUNT]
) {
    output[0] = f3(left[0] * right[2] - left[2] * right[0]);
    output[1] = f3(left[1] * right[2] - left[3] * right[0]);
    output[2] = f3(left[0] * right[3] - left[2] * right[1]);
    output[3] = f3(left[1] * right[3] - left[3] * right[1]);
}

static void derive_boundary(
    int boundary[BOUNDARY_COUNT][COEFFICIENT_COUNT],
    const int a[COEFFICIENT_COUNT],
    const int b[COEFFICIENT_COUNT],
    const int bridge[COEFFICIENT_COUNT],
    const int c[COEFFICIENT_COUNT],
    const int d[COEFFICIENT_COUNT]
) {
    int bridge_vu[COEFFICIENT_COUNT];
    int message_a[COEFFICIENT_COUNT];
    int message_b[COEFFICIENT_COUNT];
    transpose(bridge_vu, bridge);
    resultant(message_a, a, bridge_vu);
    resultant(message_b, b, bridge_vu);
    resultant(boundary[0], a, b);
    resultant(boundary[1], c, d);
    resultant(boundary[2], message_a, c);
    resultant(boundary[3], message_a, d);
    resultant(boundary[4], message_b, c);
    resultant(boundary[5], message_b, d);
}

static int exact_exists(
    const int a_relation[COEFFICIENT_COUNT],
    const int b_relation[COEFFICIENT_COUNT],
    const int bridge[COEFFICIENT_COUNT],
    const int c_relation[COEFFICIENT_COUNT],
    const int d_relation[COEFFICIENT_COUNT],
    int a,
    int b,
    int c,
    int d,
    int *witnesses
) {
    int count = 0;
    for (int u = 0; u < 2; ++u) {
        for (int v = 0; v < 2; ++v) {
            count += (
                evaluate(a_relation, a, u) == 0
                && evaluate(b_relation, b, u) == 0
                && evaluate(bridge, u, v) == 0
                && evaluate(c_relation, c, v) == 0
                && evaluate(d_relation, d, v) == 0
            );
        }
    }
    *witnesses = count;
    return count > 0;
}

static int factorized_accepts(
    int boundary[BOUNDARY_COUNT][COEFFICIENT_COUNT],
    int a,
    int b,
    int c,
    int d
) {
    return (
        evaluate(boundary[0], a, b) == 0
        && evaluate(boundary[1], c, d) == 0
        && evaluate(boundary[2], a, c) == 0
        && evaluate(boundary[3], a, d) == 0
        && evaluate(boundary[4], b, c) == 0
        && evaluate(boundary[5], b, d) == 0
    );
}

int main(void) {
    int signature[MAX_SIGNATURES][COEFFICIENT_COUNT];
    size_t signature_count = 0U;
    uint8_t seen_mask[16] = {0};
    size_t extensional_count = 0U;
    for (size_t ordinal = 0U; ordinal < MAX_SIGNATURES; ++ordinal) {
        int coefficient[COEFFICIENT_COUNT];
        decode_signature(ordinal, coefficient);
        if (!bi_total(coefficient)) {
            continue;
        }
        memcpy(
            signature[signature_count],
            coefficient,
            sizeof(coefficient)
        );
        ++signature_count;
        const uint8_t mask = relation_mask(coefficient);
        if (!seen_mask[mask]) {
            seen_mask[mask] = 1U;
            ++extensional_count;
        }
    }

    uint64_t message_pairs = 0U;
    uint64_t message_bi_total = 0U;
    for (size_t left = 0U; left < signature_count; ++left) {
        for (size_t bridge = 0U; bridge < signature_count; ++bridge) {
            int bridge_vu[COEFFICIENT_COUNT];
            int message[COEFFICIENT_COUNT];
            transpose(bridge_vu, signature[bridge]);
            resultant(message, signature[left], bridge_vu);
            ++message_pairs;
            message_bi_total += (uint64_t)bi_total(message);
        }
    }

    uint64_t tuples = 0U;
    uint64_t assignment_rows = 0U;
    uint64_t exact_rows = 0U;
    uint64_t multi_witness_rows = 0U;
    for (size_t a_index = 0U; a_index < signature_count; ++a_index) {
        for (size_t b_index = 0U; b_index < signature_count; ++b_index) {
            for (
                size_t bridge_index = 0U;
                bridge_index < signature_count;
                ++bridge_index
            ) {
                for (
                    size_t c_index = 0U;
                    c_index < signature_count;
                    ++c_index
                ) {
                    for (
                        size_t d_index = 0U;
                        d_index < signature_count;
                        ++d_index
                    ) {
                        int boundary[
                            BOUNDARY_COUNT
                        ][COEFFICIENT_COUNT];
                        derive_boundary(
                            boundary,
                            signature[a_index],
                            signature[b_index],
                            signature[bridge_index],
                            signature[c_index],
                            signature[d_index]
                        );
                        ++tuples;
                        for (
                            int assignment = 0;
                            assignment < 16;
                            ++assignment
                        ) {
                            const int a = assignment & 1;
                            const int b = (assignment >> 1) & 1;
                            const int c = (assignment >> 2) & 1;
                            const int d = (assignment >> 3) & 1;
                            int witnesses = 0;
                            const int exact = exact_exists(
                                signature[a_index],
                                signature[b_index],
                                signature[bridge_index],
                                signature[c_index],
                                signature[d_index],
                                a,
                                b,
                                c,
                                d,
                                &witnesses
                            );
                            const int factorized = factorized_accepts(
                                boundary,
                                a,
                                b,
                                c,
                                d
                            );
                            ++assignment_rows;
                            exact_rows +=
                                (uint64_t)(exact == factorized);
                            multi_witness_rows +=
                                (uint64_t)(witnesses > 1);
                        }
                    }
                }
            }
        }
    }

    printf(
        "{\"schema\":\"algebraic_two_hub_closure_v1\","
        "\"bi_total_signatures\":%zu,"
        "\"extensional_relations\":%zu,"
        "\"message_pairs\":%llu,"
        "\"message_outputs_bi_total\":%llu,"
        "\"five_relation_signature_tuples\":%llu,"
        "\"assignment_rows\":%llu,"
        "\"exact_rows\":%llu,"
        "\"multi_witness_rows\":%llu}\n",
        signature_count,
        extensional_count,
        (unsigned long long)message_pairs,
        (unsigned long long)message_bi_total,
        (unsigned long long)tuples,
        (unsigned long long)assignment_rows,
        (unsigned long long)exact_rows,
        (unsigned long long)multi_witness_rows
    );
    return (
        signature_count == 17U
        && extensional_count == 7U
        && message_pairs == message_bi_total
        && assignment_rows == exact_rows
    ) ? 0 : 1;
}
