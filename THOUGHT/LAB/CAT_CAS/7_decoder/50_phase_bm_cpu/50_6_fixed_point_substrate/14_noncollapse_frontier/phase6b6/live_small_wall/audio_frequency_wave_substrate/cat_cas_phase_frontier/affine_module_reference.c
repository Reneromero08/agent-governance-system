/*
 * Independent coefficient-aware GF(2) adjudicator for the mixed affine
 * phase module.  It materializes compact equation rows only, never tuples,
 * assignments, candidate sets, or shared-interface witnesses.
 */

#ifndef AM_REF_WIDTH
#define AM_REF_WIDTH 3U
#endif
#define REF_WIDTH AM_REF_WIDTH
#define REF_PUBLIC_MAIN oblivious_affine_reference_embedded_main
#include "oblivious_affine_reference.c"
#undef REF_PUBLIC_MAIN

#define AM_REF_VARIANTS 6U

struct am_ref_program {
    struct relation relation[4];
};

static void am_ref_equation(
    struct relation *relation,
    size_t row,
    unsigned char constant,
    const size_t *variable,
    size_t variables
) {
    begin_equation(relation, row, constant);
    for (size_t index = 0U; index < variables; ++index) {
        set_term(relation, row, variable[index]);
    }
}

static struct am_ref_program am_ref_make_program(size_t variant) {
    struct am_ref_program program = {0};
    if (variant == 0U) {
        const size_t f0[] = {0U, R_WIDTH + 2U};
        const size_t f1[] = {1U, R_WIDTH};
        const size_t f2[] = {2U, R_WIDTH + 1U};
        const size_t g0[] = {0U, R_WIDTH + 1U};
        const size_t g1[] = {1U, R_WIDTH + 1U, R_WIDTH + 2U};
        const size_t g2[] = {2U, R_WIDTH, R_WIDTH + 1U};
        const size_t p0[] = {0U, R_WIDTH, R_WIDTH + 1U};
        const size_t p1[] = {1U, R_WIDTH + 1U};
        const size_t p2[] = {2U, R_WIDTH, R_WIDTH + 2U};
        const size_t k0[] = {
            0U, R_WIDTH, R_WIDTH + 1U, R_WIDTH + 2U
        };
        const size_t k1[] = {1U, R_WIDTH + 1U, R_WIDTH + 2U};
        const size_t k2[] = {2U, R_WIDTH + 2U};
        am_ref_equation(&program.relation[0], 0U, 1U, f0, 2U);
        am_ref_equation(&program.relation[0], 1U, 0U, f1, 2U);
        am_ref_equation(&program.relation[0], 2U, 0U, f2, 2U);
        am_ref_equation(&program.relation[1], 0U, 0U, g0, 2U);
        am_ref_equation(&program.relation[1], 1U, 0U, g1, 3U);
        am_ref_equation(&program.relation[1], 2U, 1U, g2, 3U);
        am_ref_equation(&program.relation[2], 0U, 0U, p0, 3U);
        am_ref_equation(&program.relation[2], 1U, 0U, p1, 2U);
        am_ref_equation(&program.relation[2], 2U, 1U, p2, 3U);
        am_ref_equation(&program.relation[3], 0U, 1U, k0, 4U);
        am_ref_equation(&program.relation[3], 1U, 1U, k1, 3U);
        am_ref_equation(&program.relation[3], 2U, 1U, k2, 2U);
    } else if (variant == 1U) {
        for (size_t bit = 0U; bit < R_WIDTH; ++bit) {
            const size_t identity[] = {bit, R_WIDTH + bit};
            am_ref_equation(
                &program.relation[0], bit, 0U, identity, 2U
            );
        }
        const size_t g0[] = {2U, R_WIDTH};
        const size_t g1[] = {0U, R_WIDTH + 1U};
        const size_t g2[] = {1U, R_WIDTH + 2U};
        const size_t p0[] = {R_WIDTH};
        const size_t p1[] = {0U, R_WIDTH + 1U};
        const size_t p2[] = {1U, R_WIDTH + 2U};
        const size_t k0[] = {0U, R_WIDTH};
        const size_t k1[] = {1U, R_WIDTH + 1U};
        const size_t k2[] = {2U, R_WIDTH + 2U};
        am_ref_equation(&program.relation[1], 0U, 0U, g0, 2U);
        am_ref_equation(&program.relation[1], 1U, 0U, g1, 2U);
        am_ref_equation(&program.relation[1], 2U, 0U, g2, 2U);
        am_ref_equation(&program.relation[2], 0U, 0U, p0, 1U);
        am_ref_equation(&program.relation[2], 1U, 0U, p1, 2U);
        am_ref_equation(&program.relation[2], 2U, 0U, p2, 2U);
        am_ref_equation(&program.relation[3], 0U, 1U, k0, 2U);
        am_ref_equation(&program.relation[3], 1U, 0U, k1, 2U);
        am_ref_equation(&program.relation[3], 2U, 1U, k2, 2U);
    } else if (variant == 2U) {
        program.relation[0].cell[R_BLOCK_CELLS - 1U] = 1U;
    } else if (variant == 3U) {
        for (size_t bit = 0U; bit < R_WIDTH; ++bit) {
            const size_t identity[] = {bit, R_WIDTH + bit};
            am_ref_equation(
                &program.relation[0], bit, 0U, identity, 2U
            );
            am_ref_equation(
                &program.relation[1], bit, 0U, identity, 2U
            );
            am_ref_equation(
                &program.relation[3], bit, 0U, identity, 2U
            );
        }
    } else if (variant == 4U) {
        const size_t f[] = {0U};
        const size_t p[] = {0U};
        am_ref_equation(&program.relation[0], 0U, 0U, f, 1U);
        am_ref_equation(&program.relation[2], 1U, 1U, p, 1U);
    } else if (variant == 5U) {
        const size_t f[] = {0U, R_WIDTH};
        const size_t g[] = {0U, R_WIDTH};
        const size_t p[] = {0U, R_WIDTH};
        const size_t f_high[] = {
            R_WIDTH - 1U, 2U * R_WIDTH - 1U
        };
        const size_t g_high[] = {
            R_WIDTH - 1U, 2U * R_WIDTH - 1U
        };
        const size_t p_high[] = {
            R_WIDTH - 1U, 2U * R_WIDTH - 1U
        };
        am_ref_equation(&program.relation[0], 0U, 0U, f, 2U);
        am_ref_equation(
            &program.relation[0],
            R_ROWS - 1U,
            0U,
            f_high,
            2U
        );
        am_ref_equation(&program.relation[1], 1U, 0U, g, 2U);
        am_ref_equation(
            &program.relation[1],
            R_ROWS - 1U,
            1U,
            g_high,
            2U
        );
        am_ref_equation(&program.relation[2], 2U, 0U, p, 2U);
        am_ref_equation(
            &program.relation[2],
            R_ROWS - 1U,
            1U,
            p_high,
            2U
        );
        for (size_t bit = 0U; bit < R_WIDTH; ++bit) {
            const size_t identity[] = {bit, R_WIDTH + bit};
            am_ref_equation(
                &program.relation[3], bit, 0U, identity, 2U
            );
        }
    }
    return program;
}

static struct relation intersect(
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
        for (size_t side = 0U; side < 2U; ++side) {
            const struct relation *source = side == 0U ? left : right;
            for (size_t row = 0U; row < R_ROWS; ++row) {
                if (relation_field(source, row, 0U) == 0U) {
                    continue;
                }
                const size_t target = joint.rows++;
                for (size_t bit = 0U; bit < 2U * R_WIDTH; ++bit) {
                    joint.bit[target][R_WIDTH + bit] =
                        relation_field(source, row, 1U + bit);
                }
                joint.bit[target][R_JOINT_COLUMNS - 1U] =
                    relation_field(
                        source, row, R_ROW_CELLS - 1U
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
    const unsigned char empty = (unsigned char)(
        joint.input_empty | inconsistent
    );
    struct relation output = {0};
    output.cell[R_BLOCK_CELLS - 1U] = empty;
    if (empty != 0U) {
        return output;
    }
    for (size_t row = 0U; row < R_ROWS; ++row) {
        const size_t stage = R_WIDTH + row;
        const size_t source = joint.pivot_row[stage];
        const unsigned char active = (unsigned char)(
            source != SIZE_MAX
        );
        output.cell[row * R_ROW_CELLS] = active;
        if (active == 0U) {
            continue;
        }
        for (size_t column = 0U; column < 2U * R_WIDTH; ++column) {
            output.cell[row * R_ROW_CELLS + 1U + column] =
                joint.bit[source][R_WIDTH + column];
        }
        output.cell[row * R_ROW_CELLS + R_ROW_CELLS - 1U] =
            joint.bit[source][R_JOINT_COLUMNS - 1U];
    }
    return output;
}

static void am_ref_print(
    size_t variant,
    const struct relation *boundary
) {
    const uint64_t hash = hash_bytes(
        UINT64_C(14695981039346656037),
        boundary->cell,
        sizeof(boundary->cell)
    );
    printf(
        "{\"variant\":%zu,\"boundary_coefficients\":[",
        variant
    );
    for (size_t field = 0U; field < R_BLOCK_CELLS; ++field) {
        if (field != 0U) {
            putchar(',');
        }
        printf("%u", (unsigned)boundary->cell[field]);
    }
    printf(
        "],\"boundary_fnv1a64\":\"%016llx\","
        "\"tuple_slots\":0,\"assignment_slots\":0,"
        "\"witness_slots\":0}\n",
        (unsigned long long)hash
    );
}

int main(void) {
    for (size_t variant = 0U; variant < AM_REF_VARIANTS; ++variant) {
        const struct am_ref_program program =
            am_ref_make_program(variant);
        const struct relation child = compose(
            &program.relation[0], &program.relation[1]
        );
        const struct relation parallel = intersect(
            &child, &program.relation[2]
        );
        const struct relation boundary = compose(
            &parallel, &program.relation[3]
        );
        am_ref_print(variant, &boundary);
    }
    return 0;
}
