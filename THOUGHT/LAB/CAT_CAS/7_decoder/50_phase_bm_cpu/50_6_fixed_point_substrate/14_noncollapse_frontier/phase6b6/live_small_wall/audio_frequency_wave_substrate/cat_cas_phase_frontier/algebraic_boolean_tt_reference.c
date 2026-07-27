#define _POSIX_C_SOURCE 200809L

/*
 * Independent compact Boolean-semiring TT reference.
 *
 * This program contracts only the two local shared-bit values for each output
 * core cell.  It does not enumerate a width-wide assignment, accepted tuple,
 * witness list, or 4^width relation table.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REF_MIN_WIDTH 4U
#define REF_MAX_WIDTH 16U

enum ref_variant {
    REF_NEIGHBOR_AND = 0,
    REF_NEIGHBOR_NAND = 1
};

struct ref_tt {
    size_t width;
    size_t rank;
    size_t cells;
    unsigned char *bit;
};

static void ref_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void *ref_calloc(size_t count, size_t size) {
    if (count != 0U && size > SIZE_MAX / count) {
        ref_fail("reference allocation overflow");
    }
    void *memory = calloc(count, size);
    if (memory == NULL) {
        ref_fail("reference allocation failed");
    }
    return memory;
}

static size_t ref_width(const char *text) {
    if (text == NULL || text[0] < '1' || text[0] > '9') {
        ref_fail("width must be a canonical decimal from 4 through 16");
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            ref_fail("width must be a canonical decimal from 4 through 16");
        }
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long parsed = strtoul(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed < REF_MIN_WIDTH
        || parsed > REF_MAX_WIDTH
    ) {
        ref_fail("width must be a canonical decimal from 4 through 16");
    }
    return (size_t)parsed;
}

static size_t ref_cells(size_t width, size_t rank) {
    return 8U * rank + 4U * (width - 2U) * rank * rank;
}

static struct ref_tt ref_make(size_t width, size_t rank) {
    struct ref_tt tt = {
        .width = width,
        .rank = rank,
        .cells = ref_cells(width, rank)
    };
    tt.bit = ref_calloc(tt.cells, sizeof(*tt.bit));
    return tt;
}

static void ref_free(struct ref_tt *tt) {
    free(tt->bit);
    tt->bit = NULL;
    tt->cells = 0U;
}

static size_t ref_left_rank(const struct ref_tt *tt, size_t site) {
    return site == 0U ? 1U : tt->rank;
}

static size_t ref_right_rank(const struct ref_tt *tt, size_t site) {
    return site + 1U == tt->width ? 1U : tt->rank;
}

static size_t ref_offset(const struct ref_tt *tt, size_t site) {
    return site == 0U
        ? 0U
        : 4U * tt->rank
            + (site - 1U) * 4U * tt->rank * tt->rank;
}

static size_t ref_index(
    const struct ref_tt *tt,
    size_t site,
    size_t left,
    size_t x,
    size_t y,
    size_t right
) {
    const size_t left_rank = ref_left_rank(tt, site);
    const size_t right_rank = ref_right_rank(tt, site);
    if (
        site >= tt->width
        || left >= left_rank
        || right >= right_rank
        || x > 1U
        || y > 1U
    ) {
        ref_fail("reference core coordinate outside rank");
    }
    const size_t index = ref_offset(tt, site)
        + (((left * 2U + x) * 2U + y) * right_rank)
        + right;
    if (index >= tt->cells) {
        ref_fail("reference core index outside block");
    }
    return index;
}

static unsigned char ref_leaf_bit(
    enum ref_variant variant,
    size_t width,
    size_t site,
    size_t left,
    size_t x,
    size_t y,
    size_t right
) {
    if (site + 1U == width) {
        return (unsigned char)(left == x);
    }
    if (site != 0U && left != x) {
        return 0U;
    }
    const size_t expected = variant == REF_NEIGHBOR_AND
        ? x * right
        : 1U - x * right;
    return (unsigned char)(y == expected);
}

static void ref_fill_leaf(
    struct ref_tt *tt,
    enum ref_variant variant
) {
    for (size_t site = 0U; site < tt->width; ++site) {
        const size_t left_rank = ref_left_rank(tt, site);
        const size_t right_rank = ref_right_rank(tt, site);
        for (size_t left = 0U; left < left_rank; ++left) {
            for (size_t x = 0U; x < 2U; ++x) {
                for (size_t y = 0U; y < 2U; ++y) {
                    for (size_t right = 0U; right < right_rank; ++right) {
                        tt->bit[ref_index(
                            tt, site, left, x, y, right
                        )] = ref_leaf_bit(
                            variant,
                            tt->width,
                            site,
                            left,
                            x,
                            y,
                            right
                        );
                    }
                }
            }
        }
    }
}

static uint64_t ref_compose(
    const struct ref_tt *left,
    const struct ref_tt *right,
    struct ref_tt *output
) {
    if (
        left->width != right->width
        || output->width != left->width
        || output->rank != left->rank * right->rank
    ) {
        ref_fail("reference product-rank composition mismatch");
    }
    uint64_t local_shared_value_reads = 0U;
    for (size_t site = 0U; site < output->width; ++site) {
        const size_t lla = ref_left_rank(left, site);
        const size_t llb = ref_left_rank(right, site);
        const size_t rla = ref_right_rank(left, site);
        const size_t rlb = ref_right_rank(right, site);
        for (size_t a = 0U; a < lla; ++a) {
            for (size_t c = 0U; c < llb; ++c) {
                for (size_t x = 0U; x < 2U; ++x) {
                    for (size_t z = 0U; z < 2U; ++z) {
                        for (size_t b = 0U; b < rla; ++b) {
                            for (size_t d = 0U; d < rlb; ++d) {
                                unsigned char value = 0U;
                                for (size_t y = 0U; y < 2U; ++y) {
                                    value = (unsigned char)(
                                        value
                                        || (
                                            left->bit[ref_index(
                                                left,
                                                site,
                                                a,
                                                x,
                                                y,
                                                b
                                            )]
                                            && right->bit[ref_index(
                                                right,
                                                site,
                                                c,
                                                y,
                                                z,
                                                d
                                            )]
                                        )
                                    );
                                    ++local_shared_value_reads;
                                }
                                output->bit[ref_index(
                                    output,
                                    site,
                                    a * llb + c,
                                    x,
                                    z,
                                    b * rlb + d
                                )] = value;
                            }
                        }
                    }
                }
            }
        }
    }
    return local_shared_value_reads;
}

static uint64_t ref_hash_bytes(
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

static uint64_t ref_program_hash(const struct ref_tt *leaf) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t copy = 0U; copy < 3U; ++copy) {
        hash = ref_hash_bytes(hash, leaf->bit, leaf->cells);
    }
    return hash;
}

static size_t ref_ones(const struct ref_tt *tt) {
    size_t ones = 0U;
    for (size_t index = 0U; index < tt->cells; ++index) {
        ones += (size_t)tt->bit[index];
    }
    return ones;
}

static void ref_run_variant(
    size_t width,
    enum ref_variant variant,
    uint64_t *program_hash,
    uint64_t *boundary_hash,
    size_t *boundary_ones,
    uint64_t *local_reads
) {
    struct ref_tt leaf = ref_make(width, 2U);
    struct ref_tt h = ref_make(width, 4U);
    struct ref_tt z = ref_make(width, 8U);
    ref_fill_leaf(&leaf, variant);
    *program_hash = ref_program_hash(&leaf);
    *local_reads = ref_compose(&leaf, &leaf, &h);
    *local_reads += ref_compose(&h, &leaf, &z);
    *boundary_hash = ref_hash_bytes(
        UINT64_C(14695981039346656037), z.bit, z.cells
    );
    *boundary_ones = ref_ones(&z);
    ref_free(&z);
    ref_free(&h);
    ref_free(&leaf);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        ref_fail("usage: algebraic_boolean_tt_reference <width 4..16>");
    }
    const size_t width = ref_width(argv[1]);
    uint64_t primary_program = 0U;
    uint64_t primary_boundary = 0U;
    size_t primary_ones = 0U;
    uint64_t primary_reads = 0U;
    uint64_t reuse_program = 0U;
    uint64_t reuse_boundary = 0U;
    size_t reuse_ones = 0U;
    uint64_t reuse_reads = 0U;
    ref_run_variant(
        width,
        REF_NEIGHBOR_AND,
        &primary_program,
        &primary_boundary,
        &primary_ones,
        &primary_reads
    );
    ref_run_variant(
        width,
        REF_NEIGHBOR_NAND,
        &reuse_program,
        &reuse_boundary,
        &reuse_ones,
        &reuse_reads
    );
    printf(
        "{\"mode\":\"compact-boolean-tt-reference\","
        "\"width\":%zu,"
        "\"primary_program_fnv1a64\":\"%016llx\","
        "\"reuse_program_fnv1a64\":\"%016llx\","
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"primary_boundary_ones\":%zu,"
        "\"reuse_boundary_ones\":%zu,"
        "\"primary_local_shared_value_reads\":%llu,"
        "\"reuse_local_shared_value_reads\":%llu,"
        "\"width_wide_assignments_enumerated\":0,"
        "\"dense_relation_cells_materialized\":0,"
        "\"truth_table_slots\":0}\n",
        width,
        (unsigned long long)primary_program,
        (unsigned long long)reuse_program,
        (unsigned long long)primary_boundary,
        (unsigned long long)reuse_boundary,
        primary_ones,
        reuse_ones,
        (unsigned long long)primary_reads,
        (unsigned long long)reuse_reads
    );
    return 0;
}
