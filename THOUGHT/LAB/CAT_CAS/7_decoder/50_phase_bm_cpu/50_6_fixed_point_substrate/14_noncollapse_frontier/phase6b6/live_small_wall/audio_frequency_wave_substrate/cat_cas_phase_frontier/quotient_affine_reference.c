/*
 * Independent compact GF(2) reference for the projected-affine phase
 * experiment.  This program contains no phase carrier code and materializes
 * no relation table, tuple set, fibre assignment, or witness.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef REF_Q
#define REF_Q 8U
#endif
#define REF_RELATIONS 3U
#define REF_BLOCK_CELLS (REF_Q * REF_Q + REF_Q)

struct ref_relation {
    unsigned char matrix[REF_Q][REF_Q];
    unsigned char bias[REF_Q];
};

struct ref_program {
    struct ref_relation relation[REF_RELATIONS];
};

static uint64_t ref_hash(
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

static unsigned char ref_public_bit(
    size_t variant,
    size_t relation,
    size_t row,
    size_t column
) {
    if (row == column) {
        return 1U;
    }
    const size_t selector =
        5U * row + 3U * column + 2U * relation + variant;
    if ((relation + variant) % 2U == 0U) {
        return column < row && selector % 5U == 0U;
    }
    return column > row && selector % 5U == 0U;
}

static struct ref_program ref_make_program(size_t variant) {
    struct ref_program program = {0};
    for (size_t relation = 0U; relation < REF_RELATIONS; ++relation) {
        for (size_t row = 0U; row < REF_Q; ++row) {
            for (size_t column = 0U; column < REF_Q; ++column) {
                program.relation[relation].matrix[row][column] =
                    ref_public_bit(variant, relation, row, column);
            }
            program.relation[relation].bias[row] =
                (unsigned char)((3U * row + relation + variant) % 2U);
        }
    }
    return program;
}

static struct ref_relation ref_compose(
    const struct ref_relation *left,
    const struct ref_relation *right
) {
    struct ref_relation output = {0};
    for (size_t row = 0U; row < REF_Q; ++row) {
        for (size_t column = 0U; column < REF_Q; ++column) {
            unsigned char value = 0U;
            for (size_t shared = 0U; shared < REF_Q; ++shared) {
                value ^= (unsigned char)(
                    right->matrix[row][shared]
                    & left->matrix[shared][column]
                );
            }
            output.matrix[row][column] = value;
        }
        unsigned char value = right->bias[row];
        for (size_t shared = 0U; shared < REF_Q; ++shared) {
            value ^= (unsigned char)(
                right->matrix[row][shared] & left->bias[shared]
            );
        }
        output.bias[row] = value;
    }
    return output;
}

static size_t ref_rank(const struct ref_relation *relation) {
    unsigned char matrix[REF_Q][REF_Q];
    memcpy(matrix, relation->matrix, sizeof(matrix));
    size_t rank = 0U;
    for (size_t column = 0U; column < REF_Q; ++column) {
        size_t pivot = rank;
        while (pivot < REF_Q && matrix[pivot][column] == 0U) {
            ++pivot;
        }
        if (pivot == REF_Q) {
            continue;
        }
        if (pivot != rank) {
            for (size_t other = 0U; other < REF_Q; ++other) {
                const unsigned char swap = matrix[rank][other];
                matrix[rank][other] = matrix[pivot][other];
                matrix[pivot][other] = swap;
            }
        }
        for (size_t row = 0U; row < REF_Q; ++row) {
            if (row != rank && matrix[row][column] != 0U) {
                for (size_t other = column; other < REF_Q; ++other) {
                    matrix[row][other] ^= matrix[rank][other];
                }
            }
        }
        ++rank;
    }
    return rank;
}

static void ref_print_relation(
    size_t variant,
    const struct ref_program *program,
    const struct ref_relation *boundary
) {
    unsigned char coefficients[REF_BLOCK_CELLS];
    size_t next = 0U;
    for (size_t row = 0U; row < REF_Q; ++row) {
        for (size_t column = 0U; column < REF_Q; ++column) {
            coefficients[next++] = boundary->matrix[row][column];
        }
    }
    for (size_t row = 0U; row < REF_Q; ++row) {
        coefficients[next++] = boundary->bias[row];
    }
    const uint64_t hash = ref_hash(
        UINT64_C(14695981039346656037),
        coefficients,
        sizeof(coefficients)
    );
    printf(
        "{\"variant\":%zu,\"input_ranks\":[%zu,%zu,%zu],"
        "\"boundary_rank\":%zu,\"boundary_coefficients\":[",
        variant,
        ref_rank(&program->relation[0]),
        ref_rank(&program->relation[1]),
        ref_rank(&program->relation[2]),
        ref_rank(boundary)
    );
    for (size_t index = 0U; index < REF_BLOCK_CELLS; ++index) {
        if (index != 0U) {
            putchar(',');
        }
        printf("%u", (unsigned)coefficients[index]);
    }
    printf(
        "],\"boundary_fnv1a64\":\"%016llx\","
        "\"tuple_slots\":0,\"assignment_slots\":0,"
        "\"witness_slots\":0}\n",
        (unsigned long long)hash
    );
}

int main(void) {
    for (size_t variant = 0U; variant < 2U; ++variant) {
        const struct ref_program program = ref_make_program(variant);
        const struct ref_relation intermediate = ref_compose(
            &program.relation[0], &program.relation[1]
        );
        const struct ref_relation boundary = ref_compose(
            &intermediate, &program.relation[2]
        );
        ref_print_relation(variant, &program, &boundary);
    }
    return 0;
}
