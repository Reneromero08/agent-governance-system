/*
 * Exhaustive scalar survey for one branching Boolean_F3 hub elimination.
 *
 * This adjudicator is not linked into the native phase process.  It checks
 * that a family of nonempty Boolean affine fibers has a common hub value
 * exactly when every pairwise phase-resultant boundary factor vanishes.
 */

#include <stdint.h>
#include <stdio.h>

#define SIGNATURE_COUNT 81
#define ADMITTED_COUNT 17

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
    int external,
    int hub
) {
    return mod3(
        coefficient[0]
        + coefficient[1] * external
        + coefficient[2] * hub
        + coefficient[3] * external * hub
    );
}

static unsigned int fiber_mask(
    const int coefficient[4],
    int external
) {
    unsigned int mask = 0U;
    for (int hub = 0; hub <= 1; ++hub) {
        if (evaluate(coefficient, external, hub) == 0) {
            mask |= 1U << (unsigned int)hub;
        }
    }
    return mask;
}

static int affine_has_boolean_zero(int slope, int offset) {
    return offset == 0 || mod3(slope + offset) == 0;
}

static int bi_total(const int coefficient[4]) {
    return (
        affine_has_boolean_zero(coefficient[2], coefficient[0])
        && affine_has_boolean_zero(
            mod3(coefficient[2] + coefficient[3]),
            mod3(coefficient[0] + coefficient[1])
        )
        && affine_has_boolean_zero(coefficient[1], coefficient[0])
        && affine_has_boolean_zero(
            mod3(coefficient[1] + coefficient[3]),
            mod3(coefficient[0] + coefficient[2])
        )
    );
}

static void star_pair_resultant(
    const int left[4],
    const int right[4],
    int output[4]
) {
    output[0] = mod3(left[0] * right[2] - left[2] * right[0]);
    output[1] = mod3(left[1] * right[2] - left[3] * right[0]);
    output[2] = mod3(left[0] * right[3] - left[2] * right[1]);
    output[3] = mod3(left[1] * right[3] - left[3] * right[1]);
}

static int evaluate_factor(
    const int coefficient[4],
    int left_external,
    int right_external
) {
    return mod3(
        coefficient[0]
        + coefficient[1] * left_external
        + coefficient[2] * right_external
        + coefficient[3] * left_external * right_external
    );
}

static int load_admitted(int admitted[ADMITTED_COUNT][4]) {
    int count = 0;
    for (int encoded = 0; encoded < SIGNATURE_COUNT; ++encoded) {
        int coefficient[4];
        decode_signature(encoded, coefficient);
        if (bi_total(coefficient)) {
            if (count >= ADMITTED_COUNT) {
                return -1;
            }
            for (int index = 0; index < 4; ++index) {
                admitted[count][index] = coefficient[index];
            }
            ++count;
        }
    }
    return count;
}

static int check_assignment(
    int degree,
    int admitted[ADMITTED_COUNT][4],
    const int signature_index[4],
    unsigned int assignment,
    uint64_t *two_witness_rows
) {
    unsigned int common = 3U;
    for (int branch = 0; branch < degree; ++branch) {
        const int external =
            (int)((assignment >> (unsigned int)branch) & 1U);
        common &= fiber_mask(
            admitted[signature_index[branch]],
            external
        );
    }
    if (common == 3U) {
        ++*two_witness_rows;
    }
    const int projected = common != 0U;
    int factorized = 1;
    for (int left = 0; left < degree; ++left) {
        for (int right = left + 1; right < degree; ++right) {
            int factor[4];
            star_pair_resultant(
                admitted[signature_index[left]],
                admitted[signature_index[right]],
                factor
            );
            const int left_external =
                (int)((assignment >> (unsigned int)left) & 1U);
            const int right_external =
                (int)((assignment >> (unsigned int)right) & 1U);
            if (
                evaluate_factor(
                    factor,
                    left_external,
                    right_external
                ) != 0
            ) {
                factorized = 0;
            }
        }
    }
    return projected == factorized;
}

static uint64_t survey_degree(
    int degree,
    int admitted[ADMITTED_COUNT][4],
    uint64_t *rows_out,
    uint64_t *two_witness_rows_out
) {
    uint64_t signature_tuples = 1U;
    for (int index = 0; index < degree; ++index) {
        signature_tuples *= ADMITTED_COUNT;
    }
    const unsigned int assignment_count = 1U << (unsigned int)degree;
    uint64_t exact = 0U;
    uint64_t rows = 0U;
    uint64_t two_witness_rows = 0U;
    for (uint64_t encoded = 0U; encoded < signature_tuples; ++encoded) {
        uint64_t cursor = encoded;
        int signature_index[4] = {0};
        for (int branch = 0; branch < degree; ++branch) {
            signature_index[branch] =
                (int)(cursor % ADMITTED_COUNT);
            cursor /= ADMITTED_COUNT;
        }
        for (
            unsigned int assignment = 0U;
            assignment < assignment_count;
            ++assignment
        ) {
            ++rows;
            exact += (uint64_t)check_assignment(
                degree,
                admitted,
                signature_index,
                assignment,
                &two_witness_rows
            );
        }
    }
    *rows_out = rows;
    *two_witness_rows_out = two_witness_rows;
    return exact;
}

int main(void) {
    int admitted[ADMITTED_COUNT][4];
    const int admitted_count = load_admitted(admitted);
    uint64_t degree3_rows = 0U;
    uint64_t degree3_two_witness = 0U;
    const uint64_t degree3_exact = survey_degree(
        3,
        admitted,
        &degree3_rows,
        &degree3_two_witness
    );
    uint64_t degree4_rows = 0U;
    uint64_t degree4_two_witness = 0U;
    const uint64_t degree4_exact = survey_degree(
        4,
        admitted,
        &degree4_rows,
        &degree4_two_witness
    );
    printf(
        "{\"schema\":\"algebraic_relation_star_closure_v1\","
        "\"bi_total_signatures\":%d,"
        "\"degree3_signature_tuples\":4913,"
        "\"degree3_assignment_rows\":%llu,"
        "\"degree3_exact\":%llu,"
        "\"degree3_two_witness_rows\":%llu,"
        "\"degree4_signature_tuples\":83521,"
        "\"degree4_assignment_rows\":%llu,"
        "\"degree4_exact\":%llu,"
        "\"degree4_two_witness_rows\":%llu}\n",
        admitted_count,
        (unsigned long long)degree3_rows,
        (unsigned long long)degree3_exact,
        (unsigned long long)degree3_two_witness,
        (unsigned long long)degree4_rows,
        (unsigned long long)degree4_exact,
        (unsigned long long)degree4_two_witness
    );
    return (
        admitted_count == ADMITTED_COUNT
        && degree3_rows == UINT64_C(39304)
        && degree3_exact == degree3_rows
        && degree4_rows == UINT64_C(1336336)
        && degree4_exact == degree4_rows
        && degree3_two_witness > 0U
        && degree4_two_witness > 0U
    ) ? 0 : 1;
}
