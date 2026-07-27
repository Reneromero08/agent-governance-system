/*
 * Independent compact GF(2) reference for the width-parametric oblivious
 * affine phase prototype.  It uses ordinary coefficient-aware row reduction
 * only as an external adjudicator and never materializes assignments or
 * witnesses.
 */

#include <stdint.h>
#include <stdio.h>
#ifndef REF_WIDTH
#define REF_WIDTH 2U
#endif
#define R_WIDTH REF_WIDTH
#define R_ROWS (2U * R_WIDTH)
#define R_ROW_CELLS (2U * R_WIDTH + 2U)
#define R_BLOCK_CELLS (R_ROWS * R_ROW_CELLS + 1U)
#define R_JOINT_ROWS (2U * R_ROWS)
#define R_JOINT_COLUMNS (3U * R_WIDTH + 1U)
#define R_REDUCTION_COLUMNS (3U * R_WIDTH)
#define R_VARIANTS 9U
#ifndef REF_PUBLIC_MAIN
#define REF_PUBLIC_MAIN main
#endif

_Static_assert(R_WIDTH >= 2U, "affine reference requires width >= 2");
_Static_assert(R_WIDTH <= 20U, "affine reference width exceeds bound");

struct relation {
    unsigned char cell[R_BLOCK_CELLS];
};

struct program {
    struct relation relation[3];
};

struct joint {
    unsigned char bit[R_JOINT_ROWS][R_JOINT_COLUMNS];
    size_t rows;
    size_t rank;
    size_t pivot_row[R_REDUCTION_COLUMNS];
    unsigned char input_empty;
};

static uint64_t hash_bytes(
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

static void begin_equation(
    struct relation *relation,
    size_t row,
    unsigned char constant
) {
    relation->cell[row * R_ROW_CELLS] = 1U;
    relation->cell[
        row * R_ROW_CELLS + R_ROW_CELLS - 1U
    ] = constant;
}

static void set_term(
    struct relation *relation,
    size_t row,
    size_t variable
) {
    relation->cell[row * R_ROW_CELLS + 1U + variable] = 1U;
}

static struct program make_program(size_t variant) {
    struct program program = {0};
    if (variant == 0U) {
        begin_equation(&program.relation[0], R_ROWS - 1U, 0U);
        set_term(&program.relation[0], R_ROWS - 1U, 0U);
        set_term(&program.relation[0], R_ROWS - 1U, R_WIDTH);
        set_term(&program.relation[0], R_ROWS - 1U, R_WIDTH + 1U);
        begin_equation(&program.relation[1], R_ROWS - 2U, 1U);
        set_term(&program.relation[1], R_ROWS - 2U, 0U);
        set_term(&program.relation[1], R_ROWS - 2U, 1U);
        set_term(&program.relation[1], R_ROWS - 2U, R_WIDTH);
        begin_equation(&program.relation[2], 1U, 0U);
        set_term(&program.relation[2], 1U, 0U);
        set_term(&program.relation[2], 1U, R_WIDTH);
        set_term(&program.relation[2], 1U, R_WIDTH + 1U);
        begin_equation(&program.relation[2], R_ROWS - 1U, 0U);
        set_term(&program.relation[2], R_ROWS - 1U, 1U);
        set_term(
            &program.relation[2], R_ROWS - 1U, R_WIDTH + 1U
        );
    } else if (variant == 1U) {
        begin_equation(&program.relation[0], R_ROWS - 2U, 1U);
        set_term(&program.relation[0], R_ROWS - 2U, 1U);
        set_term(&program.relation[0], R_ROWS - 2U, R_WIDTH);
        begin_equation(&program.relation[1], R_ROWS - 1U, 0U);
        set_term(&program.relation[1], R_ROWS - 1U, 0U);
        set_term(
            &program.relation[1], R_ROWS - 1U, R_WIDTH + 1U
        );
        begin_equation(&program.relation[2], 1U, 1U);
        set_term(&program.relation[2], 1U, 1U);
        set_term(&program.relation[2], 1U, R_WIDTH);
    } else if (variant == 2U || variant == 3U) {
        begin_equation(&program.relation[0], 0U, 0U);
        set_term(&program.relation[0], 0U, 0U);
        set_term(&program.relation[0], 0U, R_WIDTH);
        set_term(&program.relation[0], 0U, R_WIDTH + 1U);
        begin_equation(&program.relation[1], 0U, 1U);
        set_term(&program.relation[1], 0U, 0U);
        set_term(&program.relation[1], 0U, 1U);
        set_term(&program.relation[1], 0U, R_WIDTH);
        begin_equation(&program.relation[2], 0U, 0U);
        set_term(&program.relation[2], 0U, 1U);
        set_term(&program.relation[2], 0U, R_WIDTH + 1U);
        begin_equation(&program.relation[2], 2U, 0U);
        set_term(&program.relation[2], 2U, 0U);
        set_term(&program.relation[2], 2U, R_WIDTH);
        set_term(&program.relation[2], 2U, R_WIDTH + 1U);
        if (variant == 3U) {
            begin_equation(
                &program.relation[0], R_ROWS - 1U, 0U
            );
            set_term(&program.relation[0], R_ROWS - 1U, 0U);
            set_term(&program.relation[0], R_ROWS - 1U, R_WIDTH);
            set_term(
                &program.relation[0], R_ROWS - 1U, R_WIDTH + 1U
            );
            begin_equation(
                &program.relation[1], R_ROWS - 2U, 1U
            );
            set_term(&program.relation[1], R_ROWS - 2U, 0U);
            set_term(&program.relation[1], R_ROWS - 2U, 1U);
            set_term(&program.relation[1], R_ROWS - 2U, R_WIDTH);
        }
    } else if (variant == 5U) {
        begin_equation(&program.relation[0], R_ROWS - 1U, 0U);
        set_term(&program.relation[0], R_ROWS - 1U, R_WIDTH);
        begin_equation(&program.relation[1], 0U, 0U);
        set_term(&program.relation[1], 0U, 0U);
        begin_equation(&program.relation[1], 2U, 0U);
        set_term(&program.relation[1], 2U, 0U);
    } else if (variant == 6U) {
        program.relation[0].cell[R_BLOCK_CELLS - 1U] = 1U;
    } else if (variant == 7U) {
        begin_equation(&program.relation[0], R_ROWS - 1U, 0U);
        set_term(&program.relation[0], R_ROWS - 1U, R_WIDTH);
        begin_equation(&program.relation[1], 0U, 1U);
        set_term(&program.relation[1], 0U, 0U);
        begin_equation(&program.relation[1], 1U, 1U);
        set_term(&program.relation[1], 1U, 0U);
    } else if (variant == 8U) {
        for (size_t bit = 0U; bit < R_WIDTH; ++bit) {
            begin_equation(&program.relation[0], bit, 0U);
            set_term(&program.relation[0], bit, bit);
            begin_equation(
                &program.relation[0], R_WIDTH + bit, 0U
            );
            set_term(
                &program.relation[0],
                R_WIDTH + bit,
                R_WIDTH + bit
            );
            begin_equation(&program.relation[2], bit, 0U);
            set_term(&program.relation[2], bit, R_WIDTH + bit);
        }
    } else if (variant != 4U) {
        return program;
    }
    return program;
}

static unsigned char relation_field(
    const struct relation *relation,
    size_t row,
    size_t field
) {
    return relation->cell[row * R_ROW_CELLS + field];
}

static void swap_rows(struct joint *joint, size_t first, size_t second) {
    if (first == second) {
        return;
    }
    for (size_t column = 0U; column < R_JOINT_COLUMNS; ++column) {
        const unsigned char swap = joint->bit[first][column];
        joint->bit[first][column] = joint->bit[second][column];
        joint->bit[second][column] = swap;
    }
}

static void reduce_stage(struct joint *joint, size_t stage) {
    size_t selected = joint->rows;
    for (size_t row = joint->rank; row < joint->rows; ++row) {
        if (joint->bit[row][stage] != 0U) {
            selected = row;
            break;
        }
    }
    if (selected == joint->rows) {
        return;
    }
    swap_rows(joint, joint->rank, selected);
    joint->pivot_row[stage] = joint->rank;
    for (size_t row = 0U; row < joint->rows; ++row) {
        if (
            row != joint->rank
            && joint->bit[row][stage] != 0U
        ) {
            for (
                size_t column = 0U;
                column < R_JOINT_COLUMNS;
                ++column
            ) {
                joint->bit[row][column] ^=
                    joint->bit[joint->rank][column];
            }
        }
    }
    ++joint->rank;
}

static struct relation compose(
    const struct relation *left,
    const struct relation *right
) {
    struct joint joint = {0};
    joint.input_empty = (unsigned char)(
        left->cell[R_BLOCK_CELLS - 1U]
        | right->cell[R_BLOCK_CELLS - 1U]
    );
    for (size_t stage = 0U; stage < R_REDUCTION_COLUMNS; ++stage) {
        joint.pivot_row[stage] = SIZE_MAX;
    }
    if (joint.input_empty == 0U) {
        for (size_t row = 0U; row < R_ROWS; ++row) {
            const unsigned char active = relation_field(left, row, 0U);
            if (active != 0U) {
                const size_t target = joint.rows++;
                for (size_t bit = 0U; bit < R_WIDTH; ++bit) {
                    joint.bit[target][bit] = relation_field(
                        left, row, 1U + R_WIDTH + bit
                    );
                    joint.bit[target][R_WIDTH + bit] =
                        relation_field(left, row, 1U + bit);
                }
                joint.bit[target][R_JOINT_COLUMNS - 1U] =
                    relation_field(
                        left, row, R_ROW_CELLS - 1U
                    );
            }
            const unsigned char right_active =
                relation_field(right, row, 0U);
            if (right_active != 0U) {
                const size_t target = joint.rows++;
                for (size_t bit = 0U; bit < R_WIDTH; ++bit) {
                    joint.bit[target][bit] =
                        relation_field(right, row, 1U + bit);
                    joint.bit[target][2U * R_WIDTH + bit] =
                        relation_field(
                            right, row, 1U + R_WIDTH + bit
                        );
                }
                joint.bit[target][R_JOINT_COLUMNS - 1U] =
                    relation_field(
                        right, row, R_ROW_CELLS - 1U
                    );
            }
        }
    }
    for (size_t stage = 0U; stage < R_REDUCTION_COLUMNS; ++stage) {
        reduce_stage(&joint, stage);
    }
    unsigned char inconsistent = 0U;
    for (size_t row = 0U; row < joint.rows; ++row) {
        unsigned char coefficient = 0U;
        for (
            size_t column = 0U;
            column < R_REDUCTION_COLUMNS;
            ++column
        ) {
            coefficient |= joint.bit[row][column];
        }
        if (coefficient == 0U) {
            inconsistent |= joint.bit[row][R_JOINT_COLUMNS - 1U];
        }
    }
    const unsigned char empty =
        (unsigned char)(joint.input_empty | inconsistent);
    struct relation output = {0};
    output.cell[R_BLOCK_CELLS - 1U] = empty;
    if (empty == 0U) {
        for (size_t row = 0U; row < R_ROWS; ++row) {
            const size_t stage = R_WIDTH + row;
            const unsigned char active = (unsigned char)(
                joint.pivot_row[stage] != SIZE_MAX
            );
            const size_t source = joint.pivot_row[stage];
            output.cell[row * R_ROW_CELLS] = active;
            if (active != 0U) {
                for (
                    size_t column = 0U;
                    column < 2U * R_WIDTH;
                    ++column
                ) {
                    output.cell[row * R_ROW_CELLS + 1U + column] =
                        joint.bit[source][R_WIDTH + column];
                }
                output.cell[
                    row * R_ROW_CELLS + R_ROW_CELLS - 1U
                ] = joint.bit[source][R_JOINT_COLUMNS - 1U];
            }
        }
    }
    return output;
}

static void print_relation(size_t variant, const struct relation *relation) {
    const uint64_t hash = hash_bytes(
        UINT64_C(14695981039346656037),
        relation->cell,
        sizeof(relation->cell)
    );
    printf("{\"variant\":%zu,\"boundary_coefficients\":[", variant);
    for (size_t field = 0U; field < R_BLOCK_CELLS; ++field) {
        if (field != 0U) {
            putchar(',');
        }
        printf("%u", (unsigned)relation->cell[field]);
    }
    printf(
        "],\"boundary_fnv1a64\":\"%016llx\","
        "\"tuple_slots\":0,\"assignment_slots\":0,"
        "\"witness_slots\":0}\n",
        (unsigned long long)hash
    );
}

int REF_PUBLIC_MAIN(void) {
    for (size_t variant = 0U; variant < R_VARIANTS; ++variant) {
        const struct program program = make_program(variant);
        const struct relation intermediate = compose(
            &program.relation[0], &program.relation[1]
        );
        const struct relation boundary = compose(
            &intermediate, &program.relation[2]
        );
        print_relation(variant, &boundary);
    }
    return 0;
}
