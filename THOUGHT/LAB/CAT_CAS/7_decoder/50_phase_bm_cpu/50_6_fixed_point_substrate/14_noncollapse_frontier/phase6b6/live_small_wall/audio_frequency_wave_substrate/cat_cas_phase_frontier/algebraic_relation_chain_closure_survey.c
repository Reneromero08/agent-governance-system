/*
 * Exhaustive scalar closure survey for the bi-total Boolean_F3 class.
 *
 * This adjudicator is not linked into the native phase process.  It exhausts
 * every 3^4 coefficient signature, every admitted ordered pair, and every
 * admitted ordered triple.
 */

#include <stdio.h>

#define SIGNATURE_COUNT 81
#define MAX_ADMITTED 81
#define MASK_COUNT 16

static int mod3(int value) {
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static void decode_signature(int encoded, int coefficient[4]) {
    for (int index = 0; index < 4; ++index) {
        coefficient[index] = encoded % 3;
        encoded /= 3;
    }
}

static int evaluate(
    const int coefficient[4],
    int first,
    int second
) {
    return mod3(
        coefficient[0]
        + coefficient[1] * first
        + coefficient[2] * second
        + coefficient[3] * first * second
    );
}

static unsigned int relation_mask(const int coefficient[4]) {
    unsigned int mask = 0U;
    for (int first = 0; first <= 1; ++first) {
        for (int second = 0; second <= 1; ++second) {
            if (evaluate(coefficient, first, second) == 0) {
                mask |= 1U << (unsigned int)(2 * first + second);
            }
        }
    }
    return mask;
}

static int affine_has_boolean_zero(int slope, int offset) {
    return offset == 0 || mod3(slope + offset) == 0;
}

static int bi_total(const int coefficient[4]) {
    const int toward_second = (
        affine_has_boolean_zero(coefficient[2], coefficient[0])
        && affine_has_boolean_zero(
            mod3(coefficient[2] + coefficient[3]),
            mod3(coefficient[0] + coefficient[1])
        )
    );
    const int toward_first = (
        affine_has_boolean_zero(coefficient[1], coefficient[0])
        && affine_has_boolean_zero(
            mod3(coefficient[1] + coefficient[3]),
            mod3(coefficient[0] + coefficient[2])
        )
    );
    return toward_second && toward_first;
}

static void resultant(
    const int left[4],
    const int right[4],
    int output[4]
) {
    output[0] = mod3(left[0] * right[1] - left[2] * right[0]);
    output[1] = mod3(left[1] * right[1] - left[3] * right[0]);
    output[2] = mod3(left[0] * right[3] - left[2] * right[2]);
    output[3] = mod3(left[1] * right[3] - left[3] * right[2]);
}

static unsigned int compose_masks(
    unsigned int left,
    unsigned int right
) {
    unsigned int output = 0U;
    for (int first = 0; first <= 1; ++first) {
        for (int second = 0; second <= 1; ++second) {
            for (int internal = 0; internal <= 1; ++internal) {
                const unsigned int left_bit =
                    1U << (unsigned int)(2 * first + internal);
                const unsigned int right_bit =
                    1U << (unsigned int)(2 * internal + second);
                if ((left & left_bit) != 0U && (right & right_bit) != 0U) {
                    output |= 1U << (unsigned int)(2 * first + second);
                }
            }
        }
    }
    return output;
}

int main(void) {
    int admitted[MAX_ADMITTED][4];
    int admitted_count = 0;
    int extensional[MASK_COUNT] = {0};
    for (int encoded = 0; encoded < SIGNATURE_COUNT; ++encoded) {
        int coefficient[4];
        decode_signature(encoded, coefficient);
        if (bi_total(coefficient)) {
            for (int index = 0; index < 4; ++index) {
                admitted[admitted_count][index] = coefficient[index];
            }
            extensional[relation_mask(coefficient)] = 1;
            ++admitted_count;
        }
    }

    unsigned long pair_count = 0UL;
    unsigned long pair_exact = 0UL;
    unsigned long output_closed = 0UL;
    for (int left = 0; left < admitted_count; ++left) {
        for (int right = 0; right < admitted_count; ++right) {
            int output[4];
            resultant(admitted[left], admitted[right], output);
            ++pair_count;
            if (
                relation_mask(output)
                == compose_masks(
                    relation_mask(admitted[left]),
                    relation_mask(admitted[right])
                )
            ) {
                ++pair_exact;
            }
            if (bi_total(output)) {
                ++output_closed;
            }
        }
    }

    unsigned long triple_count = 0UL;
    unsigned long triple_exact = 0UL;
    unsigned long grouping_extensional = 0UL;
    for (int first = 0; first < admitted_count; ++first) {
        for (int second = 0; second < admitted_count; ++second) {
            for (int third = 0; third < admitted_count; ++third) {
                int left_pair[4];
                int right_pair[4];
                int left_grouped[4];
                int right_grouped[4];
                resultant(
                    admitted[first],
                    admitted[second],
                    left_pair
                );
                resultant(left_pair, admitted[third], left_grouped);
                resultant(
                    admitted[second],
                    admitted[third],
                    right_pair
                );
                resultant(admitted[first], right_pair, right_grouped);
                const unsigned int expected = compose_masks(
                    compose_masks(
                        relation_mask(admitted[first]),
                        relation_mask(admitted[second])
                    ),
                    relation_mask(admitted[third])
                );
                ++triple_count;
                if (relation_mask(left_grouped) == expected) {
                    ++triple_exact;
                }
                if (
                    relation_mask(left_grouped)
                    == relation_mask(right_grouped)
                ) {
                    ++grouping_extensional;
                }
            }
        }
    }

    int extensional_count = 0;
    for (int mask = 0; mask < MASK_COUNT; ++mask) {
        extensional_count += extensional[mask] != 0;
    }
    printf(
        "{\"schema\":\"algebraic_relation_chain_closure_v1\","
        "\"bi_total_signatures\":%d,"
        "\"bi_total_extensional_relations\":%d,"
        "\"ordered_pairs\":%lu,"
        "\"pair_exact\":%lu,"
        "\"pair_output_bi_total\":%lu,"
        "\"ordered_triples\":%lu,"
        "\"left_chain_exact\":%lu,"
        "\"grouping_extensional\":%lu}\n",
        admitted_count,
        extensional_count,
        pair_count,
        pair_exact,
        output_closed,
        triple_count,
        triple_exact,
        grouping_extensional
    );
    return (
        admitted_count == 17
        && extensional_count == 7
        && pair_count == 289UL
        && pair_exact == pair_count
        && output_closed == pair_count
        && triple_count == 4913UL
        && triple_exact == triple_count
        && grouping_extensional == triple_count
    ) ? 0 : 1;
}
