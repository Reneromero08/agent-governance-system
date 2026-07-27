#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: exact suffix quotient for repeated nonlinear
 * Boolean relation tensor trains.
 *
 * The public neighbor-AND and neighbor-OR schemas have a topology-derived
 * suffix equivalence.  At composition depth d, the final internal bond has
 * rank two; every earlier bond with L remaining sites has rank
 * min(d+1,L+2), rather than raw product rank 2^d.  Composition reads the
 * actual resident depth-(d-1) tensor and the actual resident leaf.  It
 * contracts only local shared bits and quotient-class members.
 *
 * This is a family-scoped quotient law, not general Boolean-TT minimization.
 */

#define main algebraic_series_parallel_embedded_main
#include "algebraic_series_parallel_phase.c"
#undef main

#include <errno.h>

#define QTT_MIN_WIDTH 4U
#define QTT_MAX_WIDTH 16U
#define QTT_MIN_DEPTH 2U
#define QTT_MAX_DEPTH 8U
#define QTT_REUSE_CYCLES 32U

enum qtt_variant {
    QTT_NEIGHBOR_AND = 0,
    QTT_NEIGHBOR_OR = 1
};

enum qtt_mode {
    QTT_CORRECT = 0,
    QTT_WRONG_BOUNDARY_INVERSE = 1,
    QTT_MISSING_DEEPEST_INVERSE = 2,
    QTT_REORDERED_PENULTIMATE_FIRST = 3,
    QTT_SNAPSHOT_RELOAD = 4
};

struct qtt_stage {
    size_t start;
    size_t depth;
    size_t cells;
};

struct qtt_layout {
    size_t width;
    size_t maximum_depth;
    struct qtt_stage stage[QTT_MAX_DEPTH + 1U];
    struct qtt_stage boundary;
    size_t carrier_cells;
};

struct qtt_stats {
    uint64_t logical_phase_ands;
    uint64_t logical_phase_ors;
    uint64_t carrier_reads;
    uint64_t phase_cell_updates;
    uint64_t final_decodes;
    uint64_t quotient_member_terms;
};

struct qtt_projection {
    unsigned char *bit;
    size_t cells;
    size_t ones;
    uint64_t hash;
    double maximum_root_error;
};

struct qtt_execution {
    struct qtt_projection projection;
    struct qtt_stats stats;
    double restoration_max_abs;
    double integrity_max_abs;
    int snapshot_loaded;
};

struct qtt_pair {
    size_t old_state;
    size_t leaf_state;
};

static size_t qtt_parse_canonical(
    const char *text,
    size_t minimum,
    size_t maximum,
    const char *message
) {
    if (text == NULL || text[0] < '1' || text[0] > '9') {
        fail(message);
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            fail(message);
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
        fail(message);
    }
    return (size_t)parsed;
}

static size_t qtt_bond_rank(
    size_t width,
    size_t depth,
    size_t bond
) {
    if (bond == 0U || bond == width) {
        return 1U;
    }
    if (bond > width || depth == 0U) {
        fail("invalid quotient-TT bond");
    }
    const size_t suffix = width - bond;
    if (suffix == 1U) {
        return 2U;
    }
    const size_t depth_rank = depth + 1U;
#ifdef QTT_BAD_HORIZON_CONTROL
    const size_t horizon_rank = suffix + 1U;
#else
    const size_t horizon_rank = suffix + 2U;
#endif
    return depth_rank < horizon_rank ? depth_rank : horizon_rank;
}

static size_t qtt_stage_cells(size_t width, size_t depth) {
    size_t cells = 0U;
    for (size_t site = 0U; site < width; ++site) {
        const size_t left = qtt_bond_rank(width, depth, site);
        const size_t right = qtt_bond_rank(
            width, depth, site + 1U
        );
        if (left > SIZE_MAX / 4U || right > SIZE_MAX / (4U * left)) {
            fail("quotient-TT cell count overflow");
        }
        const size_t core = 4U * left * right;
        if (cells > SIZE_MAX - core) {
            fail("quotient-TT cell count overflow");
        }
        cells += core;
    }
    return cells;
}

static struct qtt_layout qtt_make_layout(
    size_t width,
    size_t maximum_depth
) {
    struct qtt_layout layout = {
        .width = width,
        .maximum_depth = maximum_depth
    };
    size_t next = 0U;
    for (size_t depth = 1U; depth <= maximum_depth; ++depth) {
        const size_t cells = qtt_stage_cells(width, depth);
        layout.stage[depth] = (struct qtt_stage){
            .start = next,
            .depth = depth,
            .cells = cells
        };
        if (next > SIZE_MAX - cells) {
            fail("quotient-TT carrier layout overflow");
        }
        next += cells;
    }
    layout.boundary = (struct qtt_stage){
        .start = next,
        .depth = maximum_depth,
        .cells = layout.stage[maximum_depth].cells
    };
    if (next > SIZE_MAX - layout.boundary.cells) {
        fail("quotient-TT carrier layout overflow");
    }
    layout.carrier_cells = next + layout.boundary.cells;
    return layout;
}

static size_t qtt_site_offset(
    const struct qtt_layout *layout,
    const struct qtt_stage *stage,
    size_t site
) {
    size_t offset = 0U;
    for (size_t prior = 0U; prior < site; ++prior) {
        offset += 4U
            * qtt_bond_rank(layout->width, stage->depth, prior)
            * qtt_bond_rank(
                layout->width, stage->depth, prior + 1U
            );
    }
    return offset;
}

static size_t qtt_index(
    const struct qtt_layout *layout,
    const struct qtt_stage *stage,
    size_t site,
    size_t left,
    size_t x,
    size_t y,
    size_t right
) {
    if (site >= layout->width || x > 1U || y > 1U) {
        fail("quotient-TT coordinate outside public shape");
    }
    const size_t left_rank = qtt_bond_rank(
        layout->width, stage->depth, site
    );
    const size_t right_rank = qtt_bond_rank(
        layout->width, stage->depth, site + 1U
    );
    if (left >= left_rank || right >= right_rank) {
        fail("quotient-TT state outside public rank");
    }
    const size_t local =
        (((left * 2U + x) * 2U + y) * right_rank) + right;
    const size_t index =
        stage->start + qtt_site_offset(layout, stage, site) + local;
    if (index >= stage->start + stage->cells) {
        fail("quotient-TT index outside stage");
    }
    return index;
}

static unsigned char qtt_leaf_bit(
    enum qtt_variant variant,
    const struct qtt_layout *layout,
    size_t site,
    size_t left,
    size_t x,
    size_t y,
    size_t right
) {
    const size_t left_value =
        variant == QTT_NEIGHBOR_OR && site != 0U
            ? 1U - left
            : left;
    const size_t right_value =
        variant == QTT_NEIGHBOR_OR
            && site + 1U != layout->width
            ? 1U - right
            : right;
    if (site + 1U == layout->width) {
        return (unsigned char)(left_value == x);
    }
    if (site != 0U && left_value != x) {
        return 0U;
    }
    const size_t expected = variant == QTT_NEIGHBOR_AND
        ? x * right_value
        : (size_t)(x != 0U || right_value != 0U);
    return (unsigned char)(y == expected);
}

static void qtt_multiply(
    struct carrier *carrier,
    size_t cell,
    double complex factor,
    struct qtt_stats *stats
) {
    multiply_cell(carrier, cell, factor);
    ++stats->phase_cell_updates;
}

static double complex qtt_read(
    const struct carrier *carrier,
    size_t cell,
    struct qtt_stats *stats
) {
    ++stats->carrier_reads;
    return relative(carrier, cell);
}

static double complex qtt_phase_and(
    double complex left,
    double complex right,
    struct qtt_stats *stats
) {
    ++stats->logical_phase_ands;
    return symbol_product(left, right);
}

static double complex qtt_phase_or(
    double complex left,
    double complex right,
    struct qtt_stats *stats
) {
    ++stats->logical_phase_ors;
#ifdef QTT_BAD_OR_CONTROL
    return lock_f3_phase(left * right);
#else
    const double complex overlap = symbol_product(left, right);
    return lock_f3_phase(left * right * conj(overlap));
#endif
}

static void qtt_encode_leaf(
    struct carrier *carrier,
    const struct qtt_layout *layout,
    enum qtt_variant variant,
    int inverse,
    struct qtt_stats *stats
) {
    const struct qtt_stage *leaf = &layout->stage[1U];
    for (size_t step = 0U; step < layout->width; ++step) {
        const size_t site = inverse
            ? layout->width - 1U - step
            : step;
        const size_t left_rank = qtt_bond_rank(
            layout->width, 1U, site
        );
        const size_t right_rank = qtt_bond_rank(
            layout->width, 1U, site + 1U
        );
        for (size_t left = 0U; left < left_rank; ++left) {
            for (size_t x = 0U; x < 2U; ++x) {
                for (size_t y = 0U; y < 2U; ++y) {
                    for (
                        size_t right = 0U;
                        right < right_rank;
                        ++right
                    ) {
                        const unsigned char bit = qtt_leaf_bit(
                            variant,
                            layout,
                            site,
                            left,
                            x,
                            y,
                            right
                        );
                        const double complex factor = root3((int)bit);
                        qtt_multiply(
                            carrier,
                            qtt_index(
                                layout,
                                leaf,
                                site,
                                left,
                                x,
                                y,
                                right
                            ),
                            inverse ? conj(factor) : factor,
                            stats
                        );
                    }
                }
            }
        }
    }
}

static size_t qtt_class_for_height(
    size_t width,
    size_t depth,
    size_t bond,
    size_t height
) {
    if (height > depth) {
        fail("quotient-TT height outside composition depth");
    }
    if (bond == 0U || bond == width) {
        if (height != 0U) {
            fail("quotient-TT boundary height outside rank");
        }
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

static size_t qtt_class_representative_height(
    size_t width,
    size_t depth,
    size_t bond,
    size_t quotient_state
) {
    for (size_t height = 0U; height <= depth; ++height) {
        if (
            qtt_class_for_height(width, depth, bond, height)
            == quotient_state
        ) {
            return height;
        }
    }
    fail("quotient-TT class has no height representative");
    return 0U;
}

static size_t qtt_class_members(
    enum qtt_variant variant,
    size_t width,
    size_t old_depth,
    size_t bond,
    size_t quotient_state,
    struct qtt_pair member[2]
) {
    if (bond == 0U || bond == width) {
        if (quotient_state != 0U) {
            fail("quotient-TT boundary state outside rank");
        }
        member[0] = (struct qtt_pair){0U, 0U};
        return 1U;
    }
    const size_t new_depth = old_depth + 1U;
    const size_t new_rank = qtt_bond_rank(
        width, new_depth, bond
    );
    if (quotient_state >= new_rank) {
        fail("quotient-TT class outside rank");
    }
    const size_t forced_leaf_value =
        variant == QTT_NEIGHBOR_AND ? 0U : 1U;
    const size_t forced_leaf_state =
        variant == QTT_NEIGHBOR_AND
            ? forced_leaf_value
            : 1U - forced_leaf_value;
    const size_t opposite_leaf_state = 1U - forced_leaf_state;
    const size_t old_rank = qtt_bond_rank(
        width, old_depth, bond
    );
    size_t count = 0U;
    for (
        size_t old_state = 0U;
        old_state < old_rank;
        ++old_state
    ) {
        const size_t representative_height =
            qtt_class_representative_height(
                width, old_depth, bond, old_state
            );
        const size_t forced_class = qtt_class_for_height(
            width,
            new_depth,
            bond,
            representative_height
        );
        if (forced_class == quotient_state) {
            if (count >= 2U) {
                fail("quotient-TT member class exceeded bound");
            }
            member[count++] = (struct qtt_pair){
                old_state, forced_leaf_state
            };
        }
        if (
            qtt_class_for_height(
                width, old_depth, bond, old_depth
            ) == old_state
            && qtt_class_for_height(
                width, new_depth, bond, new_depth
            ) == quotient_state
        ) {
            if (count >= 2U) {
                fail("quotient-TT member class exceeded bound");
            }
            member[count++] = (struct qtt_pair){
                old_state, opposite_leaf_state
            };
        }
    }
    if (count == 0U) {
        fail("quotient-TT member class is empty");
    }
    return count;
}

static void qtt_compose_quotient(
    struct carrier *carrier,
    const struct qtt_layout *layout,
    size_t output_depth,
    enum qtt_variant variant,
    int inverse,
    struct qtt_stats *stats
) {
    if (output_depth < 2U || output_depth > layout->maximum_depth) {
        fail("invalid quotient-TT composition depth");
    }
    const struct qtt_stage *input =
        &layout->stage[output_depth - 1U];
    const struct qtt_stage *leaf = &layout->stage[1U];
    const struct qtt_stage *output = &layout->stage[output_depth];
    for (size_t site = 0U; site < layout->width; ++site) {
        const size_t output_left_rank = qtt_bond_rank(
            layout->width, output_depth, site
        );
        const size_t output_right_rank = qtt_bond_rank(
            layout->width, output_depth, site + 1U
        );
        for (
            size_t quotient_left = 0U;
            quotient_left < output_left_rank;
            ++quotient_left
        ) {
            struct qtt_pair left_member[2];
            const size_t left_members = qtt_class_members(
                variant,
                layout->width,
                output_depth - 1U,
                site,
                quotient_left,
                left_member
            );
            (void)left_members;
            const struct qtt_pair representative = left_member[0];
            for (size_t x = 0U; x < 2U; ++x) {
                for (size_t z = 0U; z < 2U; ++z) {
                    for (
                        size_t quotient_right = 0U;
                        quotient_right < output_right_rank;
                        ++quotient_right
                    ) {
                        struct qtt_pair right_member[2];
                        const size_t right_members = qtt_class_members(
                            variant,
                            layout->width,
                            output_depth - 1U,
                            site + 1U,
                            quotient_right,
                            right_member
                        );
                        double complex factor = 0.0;
                        int have_factor = 0;
                        for (
                            size_t member_index = 0U;
                            member_index < right_members;
                            ++member_index
                        ) {
                            for (size_t shared = 0U; shared < 2U; ++shared) {
                                const double complex left_value = qtt_read(
                                    carrier,
                                    qtt_index(
                                        layout,
                                        input,
                                        site,
                                        representative.old_state,
                                        x,
                                        shared,
                                        right_member[member_index].old_state
                                    ),
                                    stats
                                );
                                const double complex right_value = qtt_read(
                                    carrier,
                                    qtt_index(
                                        layout,
                                        leaf,
                                        site,
                                        representative.leaf_state,
                                        shared,
                                        z,
                                        right_member[member_index].leaf_state
                                    ),
                                    stats
                                );
                                const double complex term = qtt_phase_and(
                                    left_value, right_value, stats
                                );
                                ++stats->quotient_member_terms;
                                if (!have_factor) {
                                    factor = term;
                                    have_factor = 1;
                                } else {
                                    factor = qtt_phase_or(
                                        factor, term, stats
                                    );
                                }
                            }
                        }
                        if (!have_factor) {
                            fail("empty quotient-TT member class");
                        }
                        qtt_multiply(
                            carrier,
                            qtt_index(
                                layout,
                                output,
                                site,
                                quotient_left,
                                x,
                                z,
                                quotient_right
                            ),
                            inverse ? conj(factor) : factor,
                            stats
                        );
                    }
                }
            }
        }
    }
}

static void qtt_copy_final(
    struct carrier *carrier,
    const struct qtt_layout *layout,
    int inverse,
    int wrong,
    struct qtt_stats *stats
) {
    const struct qtt_stage *final =
        &layout->stage[layout->maximum_depth];
    for (size_t cell = 0U; cell < final->cells; ++cell) {
        const double complex factor = qtt_read(
            carrier, final->start + cell, stats
        );
        qtt_multiply(
            carrier,
            layout->boundary.start + cell,
            inverse && !wrong ? conj(factor) : factor,
            stats
        );
    }
}

static struct qtt_projection qtt_latch_boundary(
    const struct carrier *carrier,
    const struct qtt_layout *layout,
    struct qtt_stats *stats
) {
    struct qtt_projection projection = {
        .bit = checked_calloc(
            layout->boundary.cells, sizeof(unsigned char)
        ),
        .cells = layout->boundary.cells,
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t cell = 0U; cell < projection.cells; ++cell) {
        double distance = 0.0;
        const int bit = decode_root(
            qtt_read(
                carrier, layout->boundary.start + cell, stats
            ),
            &distance
        );
        if (bit != 0 && bit != 1) {
            fail("quotient-TT final boundary left Boolean alphabet");
        }
        projection.bit[cell] = (unsigned char)bit;
        projection.ones += (size_t)bit;
        projection.hash = hash_bytes(
            projection.hash, &projection.bit[cell], 1U
        );
        if (distance > projection.maximum_root_error) {
            projection.maximum_root_error = distance;
        }
        ++stats->final_decodes;
    }
    return projection;
}

static void qtt_free_projection(struct qtt_projection *projection) {
    free(projection->bit);
    projection->bit = NULL;
    projection->cells = 0U;
}

static struct qtt_execution qtt_execute(
    struct carrier *carrier,
    const struct qtt_layout *layout,
    enum qtt_variant variant,
    enum qtt_mode mode
) {
    if (carrier == NULL || layout == NULL) {
        fail("null quotient-TT carrier");
    }
    if (carrier->cells != layout->carrier_cells) {
        fail("quotient-TT carrier topology mismatch");
    }
    struct carrier borrowed = snapshot_carrier(carrier);
    struct qtt_execution execution = {0};
    qtt_encode_leaf(
        carrier, layout, variant, 0, &execution.stats
    );
    for (
        size_t depth = 2U;
        depth <= layout->maximum_depth;
        ++depth
    ) {
        qtt_compose_quotient(
            carrier,
            layout,
            depth,
            variant,
            0,
            &execution.stats
        );
    }
    qtt_copy_final(
        carrier, layout, 0, 0, &execution.stats
    );
    execution.projection = qtt_latch_boundary(
        carrier, layout, &execution.stats
    );

    if (mode == QTT_SNAPSHOT_RELOAD) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        execution.snapshot_loaded = 1;
    } else {
        int leaf_inverted_early = 0;
        qtt_copy_final(
            carrier,
            layout,
            1,
            mode == QTT_WRONG_BOUNDARY_INVERSE,
            &execution.stats
        );
        if (mode == QTT_REORDERED_PENULTIMATE_FIRST) {
            if (layout->maximum_depth == 2U) {
                qtt_encode_leaf(
                    carrier, layout, variant, 1, &execution.stats
                );
                leaf_inverted_early = 1;
            } else {
                qtt_compose_quotient(
                    carrier,
                    layout,
                    layout->maximum_depth - 1U,
                    variant,
                    1,
                    &execution.stats
                );
            }
        }
        for (
            size_t step = 0U;
            step < layout->maximum_depth - 1U;
            ++step
        ) {
            const size_t depth = layout->maximum_depth - step;
            if (
                mode == QTT_MISSING_DEEPEST_INVERSE
                && depth == layout->maximum_depth
            ) {
                continue;
            }
            if (
                mode == QTT_REORDERED_PENULTIMATE_FIRST
                && depth == layout->maximum_depth - 1U
            ) {
                continue;
            }
            qtt_compose_quotient(
                carrier,
                layout,
                depth,
                variant,
                1,
                &execution.stats
            );
        }
        if (!leaf_inverted_early) {
            qtt_encode_leaf(
                carrier, layout, variant, 1, &execution.stats
            );
        }
    }
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
    free_carrier(&borrowed);
    return execution;
}

static void qtt_require_correct(
    const struct qtt_execution *execution
) {
    if (
        execution->projection.maximum_root_error > ROOT_TOLERANCE
        || execution->restoration_max_abs > RESTORATION_TOLERANCE
        || execution->integrity_max_abs > RESTORATION_TOLERANCE
        || execution->snapshot_loaded
        || execution->stats.final_decodes
            != execution->projection.cells
    ) {
        fail("quotient-TT execution violated custody law");
    }
}

static struct qtt_execution qtt_control(
    const struct qtt_layout *layout,
    enum qtt_mode mode
) {
    const struct process process = {
        .carrier_cells = layout->carrier_cells
    };
    struct carrier carrier = make_carrier(
        &process, 8100 + (int)mode
    );
    struct qtt_execution execution = qtt_execute(
        &carrier, layout, QTT_NEIGHBOR_AND, mode
    );
    free_carrier(&carrier);
    return execution;
}

static uint64_t qtt_rank_plan_hash(
    const struct qtt_layout *layout
) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (
        size_t depth = 1U;
        depth <= layout->maximum_depth;
        ++depth
    ) {
        for (size_t bond = 0U; bond <= layout->width; ++bond) {
            const uint64_t rank = (uint64_t)qtt_bond_rank(
                layout->width, depth, bond
            );
            hash = hash_bytes(
                hash,
                (const unsigned char *)&rank,
                sizeof(rank)
            );
        }
    }
    return hash;
}

int main(int argc, char **argv) {
    if (argc == 4 && strcmp(argv[1], "--project-intermediate") == 0) {
        (void)qtt_parse_canonical(
            argv[2],
            QTT_MIN_WIDTH,
            QTT_MAX_WIDTH,
            "width must be canonical decimal from 4 through 16"
        );
        (void)qtt_parse_canonical(
            argv[3],
            QTT_MIN_DEPTH,
            QTT_MAX_DEPTH,
            "depth must be canonical decimal from 2 through 8"
        );
        fail("only the final quotient-TT boundary may be projected");
    }
    if (argc == 4 && strcmp(argv[1], "--null") == 0) {
        const size_t width = qtt_parse_canonical(
            argv[2],
            QTT_MIN_WIDTH,
            QTT_MAX_WIDTH,
            "width must be canonical decimal from 4 through 16"
        );
        const size_t depth = qtt_parse_canonical(
            argv[3],
            QTT_MIN_DEPTH,
            QTT_MAX_DEPTH,
            "depth must be canonical decimal from 2 through 8"
        );
        const struct qtt_layout layout = qtt_make_layout(width, depth);
        (void)qtt_execute(
            NULL, &layout, QTT_NEIGHBOR_AND, QTT_CORRECT
        );
    }
    if (argc != 3) {
        fail(
            "usage: algebraic_boolean_tt_suffix_quotient_phase "
            "<width 4..16> <depth 2..8>"
        );
    }
    const size_t width = qtt_parse_canonical(
        argv[1],
        QTT_MIN_WIDTH,
        QTT_MAX_WIDTH,
        "width must be canonical decimal from 4 through 16"
    );
    const size_t depth = qtt_parse_canonical(
        argv[2],
        QTT_MIN_DEPTH,
        QTT_MAX_DEPTH,
        "depth must be canonical decimal from 2 through 8"
    );
    if (depth > width) {
        fail("depth may not exceed width in quotient-TT experiment");
    }
    const struct qtt_layout layout = qtt_make_layout(width, depth);
    const struct process process = {
        .carrier_cells = layout.carrier_cells
    };
    struct carrier carrier = make_carrier(&process, 8001);

    struct qtt_execution primary = qtt_execute(
        &carrier, &layout, QTT_NEIGHBOR_AND, QTT_CORRECT
    );
    qtt_require_correct(&primary);
    struct qtt_execution reuse = qtt_execute(
        &carrier, &layout, QTT_NEIGHBOR_OR, QTT_CORRECT
    );
    qtt_require_correct(&reuse);
    if (primary.projection.hash == reuse.projection.hash) {
        fail("quotient-TT unrelated reuse did not change boundary");
    }

    double repeated_max = 0.0;
    for (size_t cycle = 0U; cycle < QTT_REUSE_CYCLES; ++cycle) {
        struct qtt_execution repeated = qtt_execute(
            &carrier,
            &layout,
            cycle % 2U == 0U
                ? QTT_NEIGHBOR_AND
                : QTT_NEIGHBOR_OR,
            QTT_CORRECT
        );
        qtt_require_correct(&repeated);
        if (repeated.restoration_max_abs > repeated_max) {
            repeated_max = repeated.restoration_max_abs;
        }
        qtt_free_projection(&repeated.projection);
    }
    free_carrier(&carrier);

    struct qtt_execution wrong = qtt_control(
        &layout, QTT_WRONG_BOUNDARY_INVERSE
    );
    struct qtt_execution missing = qtt_control(
        &layout, QTT_MISSING_DEEPEST_INVERSE
    );
    struct qtt_execution reordered = qtt_control(
        &layout, QTT_REORDERED_PENULTIMATE_FIRST
    );
    struct qtt_execution snapshot = qtt_control(
        &layout, QTT_SNAPSHOT_RELOAD
    );
    if (
        wrong.restoration_max_abs < CONTROL_MINIMUM
        || missing.restoration_max_abs < CONTROL_MINIMUM
        || reordered.restoration_max_abs < CONTROL_MINIMUM
        || snapshot.restoration_max_abs > RESTORATION_TOLERANCE
        || !snapshot.snapshot_loaded
    ) {
        fail("quotient-TT control did not separate");
    }

    const size_t final_cells = layout.stage[depth].cells;
    const size_t raw_rank = (size_t)1U << depth;
    const size_t raw_final_cells =
        8U * raw_rank
        + 4U * (width - 2U) * raw_rank * raw_rank;
    printf(
        "{\"mode\":\"boolean-tt-suffix-quotient\","
        "\"claim\":\"BOUNDED_BOOLEAN_TT_SUFFIX_QUOTIENT_PHASE_CLOSURE\","
        "\"width\":%zu,"
        "\"depth\":%zu,"
        "\"raw_product_rank\":%zu,"
        "\"maximum_quotient_rank\":%zu,"
        "\"final_quotient_cells\":%zu,"
        "\"raw_product_final_cells\":%zu,"
        "\"final_cell_reduction_ratio\":%.12g,"
        "\"retained_stage_cells\":%zu,"
        "\"boundary_cells\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"rank_plan_fnv1a64\":\"%016llx\","
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"primary_boundary_ones\":%zu,"
        "\"reuse_boundary_ones\":%zu,"
        "\"logical_phase_ands_per_transaction\":%llu,"
        "\"logical_phase_ors_per_transaction\":%llu,"
        "\"carrier_reads_per_transaction\":%llu,"
        "\"phase_cell_updates_per_transaction\":%llu,"
        "\"quotient_member_terms_per_transaction\":%llu,"
        "\"final_decodes_per_transaction\":%llu,"
        "\"decoded_intermediate_cells\":0,"
        "\"serialized_intermediate_cells\":0,"
        "\"materialized_raw_product_cells\":0,"
        "\"width_wide_assignments_enumerated\":0,"
        "\"truth_table_slots\":0,"
        "\"actual_resident_stage_consumed_by_successor\":true,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"same_carrier_transactions\":%u,"
        "\"maximum_repeated_restoration_error\":%.12g,"
        "\"wrong_inverse_restoration_error\":%.12g,"
        "\"missing_inverse_restoration_error\":%.12g,"
        "\"reordered_inverse_restoration_error\":%.12g,"
        "\"snapshot_loaded\":true,"
        "\"snapshot_generation\":0,"
        "\"family_scoped_analytic_quotient\":true,"
        "\"general_boolean_tt_minimization\":false,"
        "\"fixed_rank_unbounded_depth_closure\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_execution\":false,"
        "\"unlimited_catalytic_computation\":false,"
        "\"terminal\":false}\n",
        width,
        depth,
        raw_rank,
        depth + 1U,
        final_cells,
        raw_final_cells,
        (double)raw_final_cells / (double)final_cells,
        layout.boundary.start,
        layout.boundary.cells,
        layout.carrier_cells,
        (unsigned long long)qtt_rank_plan_hash(&layout),
        (unsigned long long)primary.projection.hash,
        (unsigned long long)reuse.projection.hash,
        primary.projection.ones,
        reuse.projection.ones,
        (unsigned long long)primary.stats.logical_phase_ands,
        (unsigned long long)primary.stats.logical_phase_ors,
        (unsigned long long)primary.stats.carrier_reads,
        (unsigned long long)primary.stats.phase_cell_updates,
        (unsigned long long)primary.stats.quotient_member_terms,
        (unsigned long long)primary.stats.final_decodes,
        (unsigned int)(QTT_REUSE_CYCLES + 2U),
        repeated_max,
        wrong.restoration_max_abs,
        missing.restoration_max_abs,
        reordered.restoration_max_abs
    );

    qtt_free_projection(&primary.projection);
    qtt_free_projection(&reuse.projection);
    qtt_free_projection(&wrong.projection);
    qtt_free_projection(&missing.projection);
    qtt_free_projection(&reordered.projection);
    qtt_free_projection(&snapshot.projection);
    return 0;
}
