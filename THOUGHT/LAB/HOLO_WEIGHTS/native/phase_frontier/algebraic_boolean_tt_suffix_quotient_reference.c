#define _POSIX_C_SOURCE 200809L

/*
 * Independent local-certificate reference for the Boolean-TT suffix quotient.
 *
 * It constructs raw product-rank cores and prefix reachability only inside
 * this verifier, checks class-uniform local suffix signatures, and emits only
 * final quotient commitments and aggregate certificate counts.  It never
 * enumerates width-wide assignments or materializes a 4^width relation.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QREF_MIN_WIDTH 4U
#define QREF_MAX_WIDTH 16U
#define QREF_MIN_DEPTH 2U
#define QREF_MAX_DEPTH 8U

enum qref_variant {
    QREF_AND = 0,
    QREF_OR = 1
};

struct qref_tt {
    size_t width;
    size_t depth;
    size_t rank;
    size_t cells;
    unsigned char *bit;
};

struct qref_certificate {
    uint64_t reachable_state_checks;
    uint64_t class_uniformity_checks;
    uint64_t quotient_edge_checks;
    uint64_t raw_local_shared_reads;
};

static void qref_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void *qref_calloc(size_t count, size_t size) {
    if (count != 0U && size > SIZE_MAX / count) {
        qref_fail("reference allocation overflow");
    }
    void *memory = calloc(count, size);
    if (memory == NULL) {
        qref_fail("reference allocation failed");
    }
    return memory;
}

static size_t qref_parse(
    const char *text,
    size_t minimum,
    size_t maximum,
    const char *message
) {
    if (text == NULL || text[0] < '1' || text[0] > '9') {
        qref_fail(message);
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            qref_fail(message);
        }
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long parsed = strtoul(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed < minimum
        || parsed > maximum
    ) {
        qref_fail(message);
    }
    return (size_t)parsed;
}

static size_t qref_raw_cells(size_t width, size_t rank) {
    if (rank > SIZE_MAX / 8U || rank > SIZE_MAX / 4U) {
        qref_fail("reference raw cell overflow");
    }
    if (rank > 0U && rank > SIZE_MAX / (4U * rank)) {
        qref_fail("reference raw cell overflow");
    }
    return 8U * rank + 4U * (width - 2U) * rank * rank;
}

static struct qref_tt qref_make(
    size_t width,
    size_t depth
) {
    const size_t rank = (size_t)1U << depth;
    struct qref_tt tt = {
        .width = width,
        .depth = depth,
        .rank = rank,
        .cells = qref_raw_cells(width, rank)
    };
    tt.bit = qref_calloc(tt.cells, sizeof(*tt.bit));
    return tt;
}

static void qref_free(struct qref_tt *tt) {
    free(tt->bit);
    tt->bit = NULL;
    tt->cells = 0U;
}

static size_t qref_left_rank(
    const struct qref_tt *tt,
    size_t site
) {
    return site == 0U ? 1U : tt->rank;
}

static size_t qref_right_rank(
    const struct qref_tt *tt,
    size_t site
) {
    return site + 1U == tt->width ? 1U : tt->rank;
}

static size_t qref_offset(
    const struct qref_tt *tt,
    size_t site
) {
    return site == 0U
        ? 0U
        : 4U * tt->rank
            + (site - 1U) * 4U * tt->rank * tt->rank;
}

static size_t qref_index(
    const struct qref_tt *tt,
    size_t site,
    size_t left,
    size_t x,
    size_t y,
    size_t right
) {
    const size_t left_rank = qref_left_rank(tt, site);
    const size_t right_rank = qref_right_rank(tt, site);
    if (
        site >= tt->width
        || left >= left_rank
        || right >= right_rank
        || x > 1U
        || y > 1U
    ) {
        qref_fail("reference raw coordinate outside rank");
    }
    const size_t index = qref_offset(tt, site)
        + (((left * 2U + x) * 2U + y) * right_rank)
        + right;
    if (index >= tt->cells) {
        qref_fail("reference raw index outside core");
    }
    return index;
}

static unsigned char qref_leaf_bit(
    enum qref_variant variant,
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
    const size_t expected = variant == QREF_AND
        ? x * right
        : (size_t)(x != 0U || right != 0U);
    return (unsigned char)(y == expected);
}

static void qref_fill_leaf(
    struct qref_tt *leaf,
    enum qref_variant variant
) {
    for (size_t site = 0U; site < leaf->width; ++site) {
        const size_t left_rank = qref_left_rank(leaf, site);
        const size_t right_rank = qref_right_rank(leaf, site);
        for (size_t left = 0U; left < left_rank; ++left) {
            for (size_t x = 0U; x < 2U; ++x) {
                for (size_t y = 0U; y < 2U; ++y) {
                    for (
                        size_t right = 0U;
                        right < right_rank;
                        ++right
                    ) {
                        leaf->bit[qref_index(
                            leaf, site, left, x, y, right
                        )] = qref_leaf_bit(
                            variant,
                            leaf->width,
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

static void qref_compose_raw(
    const struct qref_tt *input,
    const struct qref_tt *leaf,
    struct qref_tt *output,
    struct qref_certificate *certificate
) {
    if (
        input->width != leaf->width
        || output->width != input->width
        || output->rank != input->rank * leaf->rank
    ) {
        qref_fail("reference raw product shape mismatch");
    }
    for (size_t site = 0U; site < output->width; ++site) {
        const size_t input_left = qref_left_rank(input, site);
        const size_t leaf_left = qref_left_rank(leaf, site);
        const size_t input_right = qref_right_rank(input, site);
        const size_t leaf_right = qref_right_rank(leaf, site);
        for (size_t a = 0U; a < input_left; ++a) {
            for (size_t c = 0U; c < leaf_left; ++c) {
                for (size_t x = 0U; x < 2U; ++x) {
                    for (size_t z = 0U; z < 2U; ++z) {
                        for (size_t b = 0U; b < input_right; ++b) {
                            for (size_t d = 0U; d < leaf_right; ++d) {
                                unsigned char value = 0U;
                                for (
                                    size_t shared = 0U;
                                    shared < 2U;
                                    ++shared
                                ) {
                                    value = (unsigned char)(
                                        value
                                        || (
                                            input->bit[qref_index(
                                                input,
                                                site,
                                                a,
                                                x,
                                                shared,
                                                b
                                            )]
                                            && leaf->bit[qref_index(
                                                leaf,
                                                site,
                                                c,
                                                shared,
                                                z,
                                                d
                                            )]
                                        )
                                    );
                                    certificate->raw_local_shared_reads += 2U;
                                }
                                output->bit[qref_index(
                                    output,
                                    site,
                                    a * leaf_left + c,
                                    x,
                                    z,
                                    b * leaf_right + d
                                )] = value;
                            }
                        }
                    }
                }
            }
        }
    }
}

static size_t qref_quotient_rank(
    size_t width,
    size_t depth,
    size_t bond
) {
    if (bond == 0U || bond == width) {
        return 1U;
    }
    const size_t suffix = width - bond;
    if (suffix == 1U) {
        return 2U;
    }
    const size_t depth_rank = depth + 1U;
    const size_t horizon_rank = suffix + 2U;
    return depth_rank < horizon_rank ? depth_rank : horizon_rank;
}

static size_t qref_height_class(
    size_t width,
    size_t depth,
    size_t bond,
    size_t height
) {
    if (bond == 0U || bond == width) {
        return 0U;
    }
    const size_t suffix = width - bond;
    if (suffix == 1U) {
        return height == 0U ? 0U : 1U;
    }
    if (depth <= suffix) {
        return height;
    }
    if (height < suffix) {
        return height;
    }
    if (height < depth) {
        return suffix;
    }
    return suffix + 1U;
}

static int qref_height(
    enum qref_variant variant,
    size_t depth,
    size_t state,
    size_t *height
) {
    size_t count = 0U;
    int changed = 0;
    for (size_t layer = 0U; layer < depth; ++layer) {
        const size_t shift = depth - 1U - layer;
        const size_t bit = (state >> shift) & 1U;
        const size_t leading = variant == QREF_AND ? 1U : 0U;
        if (!changed && bit == leading) {
            ++count;
        } else {
            changed = 1;
            if (bit == leading) {
                return 0;
            }
        }
    }
    *height = count;
    return 1;
}

static size_t qref_state_class(
    enum qref_variant variant,
    size_t width,
    size_t depth,
    size_t bond,
    size_t state
) {
    if (bond == 0U || bond == width) {
        return 0U;
    }
    const size_t suffix = width - bond;
    if (suffix == 1U) {
        const size_t first = (state >> (depth - 1U)) & 1U;
        if (variant == QREF_AND) {
            return first;
        }
        return first == 0U ? 1U : 0U;
    }
    size_t height = 0U;
    if (!qref_height(variant, depth, state, &height)) {
        qref_fail("reachable nonthreshold raw state");
    }
    return qref_height_class(width, depth, bond, height);
}

static unsigned char **qref_reachability(
    const struct qref_tt *tt,
    struct qref_certificate *certificate
) {
    unsigned char **reachable = qref_calloc(
        tt->width + 1U, sizeof(*reachable)
    );
    for (size_t bond = 0U; bond <= tt->width; ++bond) {
        const size_t rank = bond == 0U || bond == tt->width
            ? 1U
            : tt->rank;
        reachable[bond] = qref_calloc(
            rank, sizeof(*reachable[bond])
        );
    }
    reachable[0U][0U] = 1U;
    for (size_t site = 0U; site < tt->width; ++site) {
        const size_t left_rank = qref_left_rank(tt, site);
        const size_t right_rank = qref_right_rank(tt, site);
        for (size_t left = 0U; left < left_rank; ++left) {
            if (!reachable[site][left]) {
                continue;
            }
            for (size_t x = 0U; x < 2U; ++x) {
                for (size_t z = 0U; z < 2U; ++z) {
                    for (
                        size_t right = 0U;
                        right < right_rank;
                        ++right
                    ) {
                        if (tt->bit[qref_index(
                            tt, site, left, x, z, right
                        )]) {
                            reachable[site + 1U][right] = 1U;
                        }
                        ++certificate->reachable_state_checks;
                    }
                }
            }
        }
    }
    if (!reachable[tt->width][0U]) {
        qref_fail("raw product relation has no accepting path");
    }
    unsigned char **coaccessible = qref_calloc(
        tt->width + 1U, sizeof(*coaccessible)
    );
    for (size_t bond = 0U; bond <= tt->width; ++bond) {
        const size_t rank = bond == 0U || bond == tt->width
            ? 1U
            : tt->rank;
        coaccessible[bond] = qref_calloc(
            rank, sizeof(*coaccessible[bond])
        );
    }
    coaccessible[tt->width][0U] = 1U;
    for (size_t step = 0U; step < tt->width; ++step) {
        const size_t site = tt->width - 1U - step;
        const size_t left_rank = qref_left_rank(tt, site);
        const size_t right_rank = qref_right_rank(tt, site);
        for (size_t left = 0U; left < left_rank; ++left) {
            for (size_t x = 0U; x < 2U; ++x) {
                for (size_t z = 0U; z < 2U; ++z) {
                    for (
                        size_t right = 0U;
                        right < right_rank;
                        ++right
                    ) {
                        if (
                            coaccessible[site + 1U][right]
                            && tt->bit[qref_index(
                                tt, site, left, x, z, right
                            )]
                        ) {
                            coaccessible[site][left] = 1U;
                        }
                        ++certificate->reachable_state_checks;
                    }
                }
            }
        }
    }
    for (size_t bond = 0U; bond <= tt->width; ++bond) {
        const size_t rank = bond == 0U || bond == tt->width
            ? 1U
            : tt->rank;
        for (size_t state = 0U; state < rank; ++state) {
            reachable[bond][state] = (unsigned char)(
                reachable[bond][state]
                && coaccessible[bond][state]
            );
        }
        free(coaccessible[bond]);
    }
    free(coaccessible);
    return reachable;
}

static void qref_free_reachability(
    unsigned char **reachable,
    size_t width
) {
    for (size_t bond = 0U; bond <= width; ++bond) {
        free(reachable[bond]);
    }
    free(reachable);
}

static unsigned char qref_signature_cell(
    const struct qref_tt *tt,
    unsigned char **reachable,
    enum qref_variant variant,
    size_t site,
    size_t raw_left,
    size_t x,
    size_t z,
    size_t quotient_right
) {
    const size_t right_rank = qref_right_rank(tt, site);
    unsigned char value = 0U;
    for (size_t raw_right = 0U; raw_right < right_rank; ++raw_right) {
        if (
            reachable[site + 1U][raw_right]
            && qref_state_class(
                variant,
                tt->width,
                tt->depth,
                site + 1U,
                raw_right
            ) == quotient_right
        ) {
            value = (unsigned char)(
                value
                || tt->bit[qref_index(
                    tt, site, raw_left, x, z, raw_right
                )]
            );
        }
    }
    return value;
}

static void qref_certify_and_hash(
    const struct qref_tt *tt,
    enum qref_variant variant,
    struct qref_certificate *certificate,
    uint64_t *hash,
    size_t *ones,
    size_t *cells
) {
    unsigned char **reachable = qref_reachability(tt, certificate);
    *hash = UINT64_C(14695981039346656037);
    *ones = 0U;
    *cells = 0U;
    for (size_t site = 0U; site < tt->width; ++site) {
        const size_t quotient_left_rank = qref_quotient_rank(
            tt->width, tt->depth, site
        );
        const size_t quotient_right_rank = qref_quotient_rank(
            tt->width, tt->depth, site + 1U
        );
        const size_t raw_left_rank = qref_left_rank(tt, site);
        for (
            size_t quotient_left = 0U;
            quotient_left < quotient_left_rank;
            ++quotient_left
        ) {
            size_t representative = SIZE_MAX;
            for (
                size_t raw_left = 0U;
                raw_left < raw_left_rank;
                ++raw_left
            ) {
                if (
                    reachable[site][raw_left]
                    && qref_state_class(
                        variant,
                        tt->width,
                        tt->depth,
                        site,
                        raw_left
                    ) == quotient_left
                ) {
                    representative = raw_left;
                    break;
                }
            }
            if (representative == SIZE_MAX) {
                qref_fail("quotient class has no reachable representative");
            }
            for (size_t x = 0U; x < 2U; ++x) {
                for (size_t z = 0U; z < 2U; ++z) {
                    for (
                        size_t quotient_right = 0U;
                        quotient_right < quotient_right_rank;
                        ++quotient_right
                    ) {
                        const unsigned char expected =
                            qref_signature_cell(
                                tt,
                                reachable,
                                variant,
                                site,
                                representative,
                                x,
                                z,
                                quotient_right
                            );
                        for (
                            size_t raw_left = representative + 1U;
                            raw_left < raw_left_rank;
                            ++raw_left
                        ) {
                            if (
                                reachable[site][raw_left]
                                && qref_state_class(
                                    variant,
                                    tt->width,
                                    tt->depth,
                                    site,
                                    raw_left
                                ) == quotient_left
                            ) {
                                const unsigned char observed =
                                    qref_signature_cell(
                                        tt,
                                        reachable,
                                        variant,
                                        site,
                                        raw_left,
                                        x,
                                        z,
                                        quotient_right
                                    );
                                ++certificate->class_uniformity_checks;
                                if (observed != expected) {
                                    qref_fail(
                                        "quotient class suffix signature differs"
                                    );
                                }
                            }
                        }
                        *hash ^= (uint64_t)expected;
                        *hash *= UINT64_C(1099511628211);
                        *ones += (size_t)expected;
                        ++*cells;
                        ++certificate->quotient_edge_checks;
                    }
                }
            }
        }
    }
    qref_free_reachability(reachable, tt->width);
}

static void qref_run_variant(
    size_t width,
    size_t maximum_depth,
    enum qref_variant variant,
    struct qref_certificate *certificate,
    uint64_t *final_hash,
    size_t *final_ones,
    size_t *final_cells
) {
    struct qref_tt leaf = qref_make(width, 1U);
    qref_fill_leaf(&leaf, variant);
    struct qref_tt current = leaf;
    int current_is_leaf = 1;
    for (size_t depth = 2U; depth <= maximum_depth; ++depth) {
        struct qref_tt output = qref_make(width, depth);
        qref_compose_raw(
            &current, &leaf, &output, certificate
        );
        uint64_t quotient_hash = 0U;
        size_t quotient_ones = 0U;
        size_t quotient_cells = 0U;
        qref_certify_and_hash(
            &output,
            variant,
            certificate,
            &quotient_hash,
            &quotient_ones,
            &quotient_cells
        );
        if (!current_is_leaf) {
            qref_free(&current);
        }
        current = output;
        current_is_leaf = 0;
        if (depth == maximum_depth) {
            *final_hash = quotient_hash;
            *final_ones = quotient_ones;
            *final_cells = quotient_cells;
        }
    }
    if (!current_is_leaf) {
        qref_free(&current);
    }
    qref_free(&leaf);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        qref_fail(
            "usage: algebraic_boolean_tt_suffix_quotient_reference "
            "<width 4..16> <depth 2..8>"
        );
    }
    const size_t width = qref_parse(
        argv[1],
        QREF_MIN_WIDTH,
        QREF_MAX_WIDTH,
        "width must be canonical decimal from 4 through 16"
    );
    const size_t depth = qref_parse(
        argv[2],
        QREF_MIN_DEPTH,
        QREF_MAX_DEPTH,
        "depth must be canonical decimal from 2 through 8"
    );
    if (depth > width) {
        qref_fail("depth may not exceed width in quotient reference");
    }
    struct qref_certificate certificate = {0};
    uint64_t and_hash = 0U;
    uint64_t or_hash = 0U;
    size_t and_ones = 0U;
    size_t or_ones = 0U;
    size_t and_cells = 0U;
    size_t or_cells = 0U;
    qref_run_variant(
        width,
        depth,
        QREF_AND,
        &certificate,
        &and_hash,
        &and_ones,
        &and_cells
    );
    qref_run_variant(
        width,
        depth,
        QREF_OR,
        &certificate,
        &or_hash,
        &or_ones,
        &or_cells
    );
    if (and_cells != or_cells) {
        qref_fail("quotient variants disagree on final shape");
    }
    const size_t raw_final_cells = qref_raw_cells(
        width, (size_t)1U << depth
    );
    printf(
        "{\"mode\":\"independent-local-bisimulation-certificate\","
        "\"width\":%zu,"
        "\"depth\":%zu,"
        "\"final_quotient_cells\":%zu,"
        "\"and_boundary_fnv1a64\":\"%016llx\","
        "\"or_boundary_fnv1a64\":\"%016llx\","
        "\"and_boundary_ones\":%zu,"
        "\"or_boundary_ones\":%zu,"
        "\"reachable_state_checks\":%llu,"
        "\"class_uniformity_checks\":%llu,"
        "\"quotient_edge_checks\":%llu,"
        "\"raw_local_shared_reads\":%llu,"
        "\"width_wide_assignments_enumerated\":0,"
        "\"dense_relation_cells_materialized\":0,"
        "\"raw_product_tt_cores_materialized_for_verification\":true,"
        "\"raw_product_tt_final_stage_cells\":%zu,"
        "\"raw_product_tt_final_stage_bytes\":%zu,"
        "\"raw_intermediate_relations_serialized\":0,"
        "\"result\":\"PASS\","
        "\"terminal\":false}\n",
        width,
        depth,
        and_cells,
        (unsigned long long)and_hash,
        (unsigned long long)or_hash,
        and_ones,
        or_ones,
        (unsigned long long)certificate.reachable_state_checks,
        (unsigned long long)certificate.class_uniformity_checks,
        (unsigned long long)certificate.quotient_edge_checks,
        (unsigned long long)certificate.raw_local_shared_reads,
        raw_final_cells,
        raw_final_cells * sizeof(unsigned char)
    );
    return 0;
}
