/*
 * Independent compact GF(2) reference for the fixed-width oblivious affine
 * phase prototype.  It uses ordinary coefficient-aware row reduction only as
 * an external adjudicator and never materializes assignments or witnesses.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define R_WIDTH 2U
#define R_ROWS 4U
#define R_ROW_CELLS 6U
#define R_BLOCK_CELLS 25U
#define R_JOINT_ROWS 8U
#define R_JOINT_COLUMNS 7U
#define R_VARIANTS 9U

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
    size_t pivot_row[6];
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

static void set_equation(
    struct relation *relation,
    size_t row,
    const unsigned char coefficient[4],
    unsigned char constant
) {
    relation->cell[row * R_ROW_CELLS] = 1U;
    memcpy(
        &relation->cell[row * R_ROW_CELLS + 1U],
        coefficient,
        4U
    );
    relation->cell[row * R_ROW_CELLS + 5U] = constant;
}

static struct program make_program(size_t variant) {
    struct program program = {0};
    if (variant == 0U) {
        static const unsigned char f[4] = {1U, 0U, 1U, 1U};
        static const unsigned char g[4] = {1U, 1U, 1U, 0U};
        static const unsigned char k0[4] = {1U, 0U, 1U, 1U};
        static const unsigned char k1[4] = {0U, 1U, 0U, 1U};
        set_equation(&program.relation[0], 3U, f, 0U);
        set_equation(&program.relation[1], 2U, g, 1U);
        set_equation(&program.relation[2], 1U, k0, 0U);
        set_equation(&program.relation[2], 3U, k1, 0U);
    } else if (variant == 1U) {
        static const unsigned char f[4] = {0U, 1U, 1U, 0U};
        static const unsigned char g[4] = {1U, 0U, 0U, 1U};
        static const unsigned char k[4] = {0U, 1U, 1U, 0U};
        set_equation(&program.relation[0], 2U, f, 1U);
        set_equation(&program.relation[1], 3U, g, 0U);
        set_equation(&program.relation[2], 1U, k, 1U);
    } else if (variant == 2U || variant == 3U) {
        static const unsigned char f[4] = {1U, 0U, 1U, 1U};
        static const unsigned char g[4] = {1U, 1U, 1U, 0U};
        static const unsigned char k0[4] = {1U, 0U, 1U, 1U};
        static const unsigned char k1[4] = {0U, 1U, 0U, 1U};
        set_equation(&program.relation[0], 0U, f, 0U);
        set_equation(&program.relation[1], 0U, g, 1U);
        set_equation(&program.relation[2], 0U, k1, 0U);
        set_equation(&program.relation[2], 2U, k0, 0U);
        if (variant == 3U) {
            set_equation(&program.relation[0], 3U, f, 0U);
            set_equation(&program.relation[1], 2U, g, 1U);
        }
    } else if (variant == 5U) {
        static const unsigned char left_y0[4] = {0U, 0U, 1U, 0U};
        static const unsigned char right_y0[4] = {1U, 0U, 0U, 0U};
        set_equation(&program.relation[0], 3U, left_y0, 0U);
        set_equation(&program.relation[1], 0U, right_y0, 0U);
        set_equation(&program.relation[1], 2U, right_y0, 0U);
    } else if (variant == 6U) {
        program.relation[0].cell[24] = 1U;
    } else if (variant == 7U) {
        static const unsigned char left_y0[4] = {0U, 0U, 1U, 0U};
        static const unsigned char right_y0[4] = {1U, 0U, 0U, 0U};
        set_equation(&program.relation[0], 3U, left_y0, 0U);
        set_equation(&program.relation[1], 0U, right_y0, 1U);
        set_equation(&program.relation[1], 1U, right_y0, 1U);
    } else if (variant == 8U) {
        static const unsigned char x0[4] = {1U, 0U, 0U, 0U};
        static const unsigned char x1[4] = {0U, 1U, 0U, 0U};
        static const unsigned char y0[4] = {0U, 0U, 1U, 0U};
        static const unsigned char y1[4] = {0U, 0U, 0U, 1U};
        set_equation(&program.relation[0], 0U, x0, 0U);
        set_equation(&program.relation[0], 1U, x1, 0U);
        set_equation(&program.relation[0], 2U, y0, 0U);
        set_equation(&program.relation[0], 3U, y1, 0U);
        set_equation(&program.relation[2], 0U, y0, 0U);
        set_equation(&program.relation[2], 3U, y1, 0U);
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
        left->cell[24] | right->cell[24]
    );
    for (size_t stage = 0U; stage < 6U; ++stage) {
        joint.pivot_row[stage] = SIZE_MAX;
    }
    if (joint.input_empty == 0U) {
        for (size_t row = 0U; row < R_ROWS; ++row) {
            const unsigned char active = relation_field(left, row, 0U);
            if (active != 0U) {
                const size_t target = joint.rows++;
                joint.bit[target][0] =
                    relation_field(left, row, 3U);
                joint.bit[target][1] =
                    relation_field(left, row, 4U);
                joint.bit[target][2] =
                    relation_field(left, row, 1U);
                joint.bit[target][3] =
                    relation_field(left, row, 2U);
                joint.bit[target][6] =
                    relation_field(left, row, 5U);
            }
            const unsigned char right_active =
                relation_field(right, row, 0U);
            if (right_active != 0U) {
                const size_t target = joint.rows++;
                joint.bit[target][0] =
                    relation_field(right, row, 1U);
                joint.bit[target][1] =
                    relation_field(right, row, 2U);
                joint.bit[target][4] =
                    relation_field(right, row, 3U);
                joint.bit[target][5] =
                    relation_field(right, row, 4U);
                joint.bit[target][6] =
                    relation_field(right, row, 5U);
            }
        }
    }
    for (size_t stage = 0U; stage < 6U; ++stage) {
        reduce_stage(&joint, stage);
    }
    unsigned char inconsistent = 0U;
    for (size_t row = 0U; row < joint.rows; ++row) {
        unsigned char coefficient = 0U;
        for (size_t column = 0U; column < 6U; ++column) {
            coefficient |= joint.bit[row][column];
        }
        if (coefficient == 0U) {
            inconsistent |= joint.bit[row][6];
        }
    }
    const unsigned char empty =
        (unsigned char)(joint.input_empty | inconsistent);
    struct relation output = {0};
    output.cell[24] = empty;
    if (empty == 0U) {
        for (size_t row = 0U; row < R_ROWS; ++row) {
            const size_t stage = R_WIDTH + row;
            const unsigned char active = (unsigned char)(
                joint.pivot_row[stage] != SIZE_MAX
            );
            const size_t source = joint.pivot_row[stage];
            output.cell[row * R_ROW_CELLS] = active;
            if (active != 0U) {
                for (size_t column = 0U; column < 4U; ++column) {
                    output.cell[row * R_ROW_CELLS + 1U + column] =
                        joint.bit[source][R_WIDTH + column];
                }
                output.cell[row * R_ROW_CELLS + 5U] =
                    joint.bit[source][6];
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

int main(void) {
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
