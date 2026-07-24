/*
 * Independent exhaustive survey for the algebraic total-to-internal-port law.
 *
 * This scalar verifier is never linked into algebraic_relation_phase.c.  It
 * enumerates all 3^4 left and 3^4 right coefficient signatures and compares
 * the resultant zero set with exact Boolean existential composition.
 */

#include <stdio.h>

#define SIGNATURE_COUNT 81
#define RELATION_MASK_COUNT 16

static int mod3(int value) {
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static void decode_signature(
    int encoded,
    int coefficient[4]
) {
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

static unsigned int existential_composition(
    const int left[4],
    const int right[4]
) {
    unsigned int mask = 0U;
    for (int first = 0; first <= 1; ++first) {
        for (int second = 0; second <= 1; ++second) {
            int exists = 0;
            for (int internal = 0; internal <= 1; ++internal) {
                exists |= (
                    evaluate(left, first, internal) == 0
                    && evaluate(right, internal, second) == 0
                );
            }
            if (exists) {
                mask |= 1U << (unsigned int)(2 * first + second);
            }
        }
    }
    return mask;
}

static void resultant(
    const int left[4],
    const int right[4],
    int boundary[4]
) {
    boundary[0] =
        mod3(left[0] * right[1] - left[2] * right[0]);
    boundary[1] =
        mod3(left[1] * right[1] - left[3] * right[0]);
    boundary[2] =
        mod3(left[0] * right[3] - left[2] * right[2]);
    boundary[3] =
        mod3(left[1] * right[3] - left[3] * right[2]);
}

static int left_total_toward_internal(const int coefficient[4]) {
    for (int first = 0; first <= 1; ++first) {
        if (
            evaluate(coefficient, first, 0) != 0
            && evaluate(coefficient, first, 1) != 0
        ) {
            return 0;
        }
    }
    return 1;
}

static int right_total_toward_internal(const int coefficient[4]) {
    for (int second = 0; second <= 1; ++second) {
        if (
            evaluate(coefficient, 0, second) != 0
            && evaluate(coefficient, 1, second) != 0
        ) {
            return 0;
        }
    }
    return 1;
}

static int count_nonzero_masks(
    const int seen[RELATION_MASK_COUNT]
) {
    int count = 0;
    for (int mask = 0; mask < RELATION_MASK_COUNT; ++mask) {
        count += seen[mask] != 0;
    }
    return count;
}

int main(void) {
    unsigned long raw_exact = 0UL;
    unsigned long raw_inexact = 0UL;
    unsigned long admitted = 0UL;
    unsigned long admitted_exact = 0UL;
    unsigned long admitted_inexact = 0UL;
    int left_total_signatures = 0;
    int right_total_signatures = 0;
    int left_masks[RELATION_MASK_COUNT] = {0};
    int right_masks[RELATION_MASK_COUNT] = {0};

    for (int encoded = 0; encoded < SIGNATURE_COUNT; ++encoded) {
        int coefficient[4];
        decode_signature(encoded, coefficient);
        if (left_total_toward_internal(coefficient)) {
            ++left_total_signatures;
            left_masks[relation_mask(coefficient)] = 1;
        }
        if (right_total_toward_internal(coefficient)) {
            ++right_total_signatures;
            right_masks[relation_mask(coefficient)] = 1;
        }
    }

    for (int left_encoded = 0; left_encoded < SIGNATURE_COUNT; ++left_encoded) {
        int left[4];
        decode_signature(left_encoded, left);
        for (
            int right_encoded = 0;
            right_encoded < SIGNATURE_COUNT;
            ++right_encoded
        ) {
            int right[4];
            int boundary[4];
            decode_signature(right_encoded, right);
            resultant(left, right, boundary);
            const int exact = (
                relation_mask(boundary)
                == existential_composition(left, right)
            );
            if (exact) {
                ++raw_exact;
            } else {
                ++raw_inexact;
            }
            if (
                left_total_toward_internal(left)
                && right_total_toward_internal(right)
            ) {
                ++admitted;
                if (exact) {
                    ++admitted_exact;
                } else {
                    ++admitted_inexact;
                }
            }
        }
    }

    printf(
        "{\"schema\":\"algebraic_totality_survey_v1\","
        "\"raw_signature_pairs\":6561,"
        "\"raw_exact\":%lu,"
        "\"raw_inexact\":%lu,"
        "\"left_total_signatures\":%d,"
        "\"right_total_signatures\":%d,"
        "\"left_total_extensional_relations\":%d,"
        "\"right_total_extensional_relations\":%d,"
        "\"admitted_signature_pairs\":%lu,"
        "\"admitted_exact\":%lu,"
        "\"admitted_inexact\":%lu}\n",
        raw_exact,
        raw_inexact,
        left_total_signatures,
        right_total_signatures,
        count_nonzero_masks(left_masks),
        count_nonzero_masks(right_masks),
        admitted,
        admitted_exact,
        admitted_inexact
    );
    return admitted_inexact == 0UL ? 0 : 1;
}
