#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: width-parametric Boolean-semiring tensor-train
 * relations with phase-resident product-rank composition.
 *
 * Three public rank-two relation tensors are encoded in actual carrier cells.
 * H = F ; G is written as a resident rank-four tensor.  Z = H ; J consumes
 * that actual H and is written as a resident rank-eight tensor.  Existential
 * contraction is local at each site and never enumerates the 2^width shared
 * assignments or materializes a 4^width relation table.
 */

#define main algebraic_series_parallel_embedded_main
#include "algebraic_series_parallel_phase.c"
#undef main

#include <errno.h>

#define BTT_MIN_WIDTH 4U
#define BTT_MAX_WIDTH 16U
#define BTT_LEAF_RANK 2U
#define BTT_H_RANK 4U
#define BTT_Z_RANK 8U
#define BTT_REUSE_CYCLES 32U

enum btt_variant {
    BTT_NEIGHBOR_AND = 0,
    BTT_NEIGHBOR_NAND = 1
};

enum btt_mode {
    BTT_CORRECT = 0,
    BTT_WRONG_BOUNDARY_INVERSE = 1,
    BTT_MISSING_H_INVERSE = 2,
    BTT_REORDERED_H_BEFORE_Z_INVERSE = 3,
    BTT_SNAPSHOT_RELOAD = 4
};

struct btt_block {
    size_t start;
    size_t rank;
    size_t cells;
};

struct btt_layout {
    size_t width;
    size_t n2;
    size_t n4;
    size_t n8;
    struct btt_block f;
    struct btt_block g;
    struct btt_block j;
    struct btt_block h;
    struct btt_block z;
    struct btt_block boundary;
    size_t carrier_cells;
};

struct btt_stats {
    uint64_t logical_phase_ands;
    uint64_t logical_phase_ors;
    uint64_t phase_products_inside_or;
    uint64_t carrier_reads;
    uint64_t phase_cell_updates;
    uint64_t final_decodes;
};

struct btt_projection {
    unsigned char *bit;
    size_t cells;
    size_t ones;
    uint64_t hash;
    double maximum_root_error;
};

struct btt_execution {
    struct btt_projection projection;
    struct btt_stats stats;
    double displacement_l2;
    double restoration_max_abs;
    double integrity_max_abs;
    int snapshot_loaded;
};

static size_t btt_parse_width(const char *text) {
    if (
        text == NULL
        || text[0] < '1'
        || text[0] > '9'
    ) {
        fail("width must be a canonical decimal from 4 through 16");
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            fail("width must be a canonical decimal from 4 through 16");
        }
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long parsed = strtoul(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed < BTT_MIN_WIDTH
        || parsed > BTT_MAX_WIDTH
    ) {
        fail("width must be a canonical decimal from 4 through 16");
    }
    return (size_t)parsed;
}

static size_t btt_cells(size_t width, size_t rank) {
    if (width < 2U || rank == 0U) {
        fail("invalid Boolean-TT width or rank");
    }
    return 8U * rank + 4U * (width - 2U) * rank * rank;
}

static struct btt_layout btt_make_layout(size_t width) {
    struct btt_layout layout = {
        .width = width,
        .n2 = btt_cells(width, BTT_LEAF_RANK),
        .n4 = btt_cells(width, BTT_H_RANK),
        .n8 = btt_cells(width, BTT_Z_RANK)
    };
    layout.f = (struct btt_block){
        .start = 0U, .rank = BTT_LEAF_RANK, .cells = layout.n2
    };
    layout.g = (struct btt_block){
        .start = layout.f.start + layout.f.cells,
        .rank = BTT_LEAF_RANK,
        .cells = layout.n2
    };
    layout.j = (struct btt_block){
        .start = layout.g.start + layout.g.cells,
        .rank = BTT_LEAF_RANK,
        .cells = layout.n2
    };
    layout.h = (struct btt_block){
        .start = layout.j.start + layout.j.cells,
        .rank = BTT_H_RANK,
        .cells = layout.n4
    };
    layout.z = (struct btt_block){
        .start = layout.h.start + layout.h.cells,
        .rank = BTT_Z_RANK,
        .cells = layout.n8
    };
    layout.boundary = (struct btt_block){
        .start = layout.z.start + layout.z.cells,
        .rank = BTT_Z_RANK,
        .cells = layout.n8
    };
    layout.carrier_cells =
        layout.boundary.start + layout.boundary.cells;
    if (
        layout.carrier_cells != 624U * width - 1040U
        || layout.n2 != 16U * width - 16U
        || layout.n4 != 64U * width - 96U
        || layout.n8 != 256U * width - 448U
    ) {
        fail("Boolean-TT resource law mismatch");
    }
    return layout;
}

static size_t btt_left_rank(
    const struct btt_layout *layout,
    const struct btt_block *block,
    size_t site
) {
    if (layout == NULL || block == NULL || site >= layout->width) {
        fail("invalid Boolean-TT site");
    }
    return site == 0U ? 1U : block->rank;
}

static size_t btt_right_rank(
    const struct btt_layout *layout,
    const struct btt_block *block,
    size_t site
) {
    if (layout == NULL || block == NULL || site >= layout->width) {
        fail("invalid Boolean-TT site");
    }
    return site + 1U == layout->width ? 1U : block->rank;
}

static size_t btt_site_offset(
    const struct btt_layout *layout,
    const struct btt_block *block,
    size_t site
) {
    if (site >= layout->width) {
        fail("invalid Boolean-TT site");
    }
    if (site == 0U) {
        return 0U;
    }
    return (
        4U * block->rank
        + (site - 1U) * 4U * block->rank * block->rank
    );
}

static size_t btt_index(
    const struct btt_layout *layout,
    const struct btt_block *block,
    size_t site,
    size_t left,
    size_t x,
    size_t y,
    size_t right
) {
    const size_t left_rank = btt_left_rank(layout, block, site);
    const size_t right_rank = btt_right_rank(layout, block, site);
    if (
        left >= left_rank
        || right >= right_rank
        || x > 1U
        || y > 1U
    ) {
        fail("Boolean-TT core coordinate outside public rank");
    }
    const size_t local =
        (((left * 2U + x) * 2U + y) * right_rank) + right;
    const size_t index =
        block->start + btt_site_offset(layout, block, site) + local;
    if (index >= block->start + block->cells) {
        fail("Boolean-TT core index outside block");
    }
    return index;
}

static unsigned char btt_leaf_bit(
    enum btt_variant variant,
    const struct btt_layout *layout,
    size_t site,
    size_t left,
    size_t x,
    size_t y,
    size_t right
) {
    if (site + 1U == layout->width) {
        return (unsigned char)(left == x);
    }
    if (site != 0U && left != x) {
        return 0U;
    }
    const size_t expected = variant == BTT_NEIGHBOR_AND
        ? x * right
        : 1U - x * right;
    return (unsigned char)(y == expected);
}

static void btt_multiply(
    struct carrier *carrier,
    size_t cell,
    double complex factor,
    struct btt_stats *stats
) {
    multiply_cell(carrier, cell, factor);
    ++stats->phase_cell_updates;
}

static double complex btt_read(
    const struct carrier *carrier,
    size_t cell,
    struct btt_stats *stats
) {
    ++stats->carrier_reads;
    return relative(carrier, cell);
}

static double complex btt_phase_and(
    double complex left,
    double complex right,
    struct btt_stats *stats
) {
    ++stats->logical_phase_ands;
    return symbol_product(left, right);
}

static double complex btt_phase_or(
    double complex left,
    double complex right,
    struct btt_stats *stats
) {
    ++stats->logical_phase_ors;
    ++stats->phase_products_inside_or;
    const double complex overlap = symbol_product(left, right);
    return lock_f3_phase(left * right * conj(overlap));
}

static void btt_encode_leaf(
    struct carrier *carrier,
    const struct btt_layout *layout,
    const struct btt_block *block,
    enum btt_variant variant,
    int inverse,
    struct btt_stats *stats
) {
    for (size_t step = 0U; step < layout->width; ++step) {
        const size_t site = inverse
            ? layout->width - 1U - step
            : step;
        const size_t left_rank = btt_left_rank(layout, block, site);
        const size_t right_rank = btt_right_rank(layout, block, site);
        for (size_t left = 0U; left < left_rank; ++left) {
            for (size_t x = 0U; x < 2U; ++x) {
                for (size_t y = 0U; y < 2U; ++y) {
                    for (size_t right = 0U; right < right_rank; ++right) {
                        const unsigned char bit = btt_leaf_bit(
                            variant,
                            layout,
                            site,
                            left,
                            x,
                            y,
                            right
                        );
                        const double complex factor = root3((int)bit);
                        btt_multiply(
                            carrier,
                            btt_index(
                                layout,
                                block,
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

static void btt_compose(
    struct carrier *carrier,
    const struct btt_layout *layout,
    const struct btt_block *left_block,
    const struct btt_block *right_block,
    const struct btt_block *output_block,
    int inverse,
    struct btt_stats *stats
) {
    if (output_block->rank != left_block->rank * right_block->rank) {
        fail("Boolean-TT product-rank composition mismatch");
    }
    for (size_t site = 0U; site < layout->width; ++site) {
        const size_t left_in_left =
            btt_left_rank(layout, left_block, site);
        const size_t left_in_right =
            btt_left_rank(layout, right_block, site);
        const size_t right_in_left =
            btt_right_rank(layout, left_block, site);
        const size_t right_in_right =
            btt_right_rank(layout, right_block, site);
        for (size_t a = 0U; a < left_in_left; ++a) {
            for (size_t c = 0U; c < left_in_right; ++c) {
                const size_t product_left = a * left_in_right + c;
                for (size_t x = 0U; x < 2U; ++x) {
                    for (size_t z = 0U; z < 2U; ++z) {
                        for (size_t b = 0U; b < right_in_left; ++b) {
                            for (
                                size_t d = 0U;
                                d < right_in_right;
                                ++d
                            ) {
                                const size_t product_right =
                                    b * right_in_right + d;
                                const double complex a0 = btt_read(
                                    carrier,
                                    btt_index(
                                        layout,
                                        left_block,
                                        site,
                                        a,
                                        x,
                                        0U,
                                        b
                                    ),
                                    stats
                                );
                                const double complex b0 = btt_read(
                                    carrier,
                                    btt_index(
                                        layout,
                                        right_block,
                                        site,
                                        c,
                                        0U,
                                        z,
                                        d
                                    ),
                                    stats
                                );
                                const double complex a1 = btt_read(
                                    carrier,
                                    btt_index(
                                        layout,
                                        left_block,
                                        site,
                                        a,
                                        x,
                                        1U,
                                        b
                                    ),
                                    stats
                                );
                                const double complex b1 = btt_read(
                                    carrier,
                                    btt_index(
                                        layout,
                                        right_block,
                                        site,
                                        c,
                                        1U,
                                        z,
                                        d
                                    ),
                                    stats
                                );
                                const double complex term0 =
                                    btt_phase_and(a0, b0, stats);
                                const double complex term1 =
                                    btt_phase_and(a1, b1, stats);
                                const double complex factor =
                                    btt_phase_or(term0, term1, stats);
                                btt_multiply(
                                    carrier,
                                    btt_index(
                                        layout,
                                        output_block,
                                        site,
                                        product_left,
                                        x,
                                        z,
                                        product_right
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
    }
}

static void btt_copy_block(
    struct carrier *carrier,
    const struct btt_block *source,
    const struct btt_block *target,
    int inverse,
    int wrong,
    struct btt_stats *stats
) {
    if (source->cells != target->cells) {
        fail("Boolean-TT boundary copy shape mismatch");
    }
    for (size_t cell = 0U; cell < source->cells; ++cell) {
        const double complex factor = btt_read(
            carrier, source->start + cell, stats
        );
        btt_multiply(
            carrier,
            target->start + cell,
            inverse && !wrong ? conj(factor) : factor,
            stats
        );
    }
}

static struct btt_projection btt_latch_boundary(
    const struct carrier *carrier,
    const struct btt_block *boundary,
    struct btt_stats *stats
) {
    struct btt_projection projection = {
        .bit = checked_calloc(boundary->cells, sizeof(unsigned char)),
        .cells = boundary->cells,
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t cell = 0U; cell < boundary->cells; ++cell) {
        double distance = 0.0;
        const int decoded = decode_root(
            btt_read(carrier, boundary->start + cell, stats),
            &distance
        );
        if (decoded != 0 && decoded != 1) {
            fail("final Boolean-TT boundary left the Boolean phase alphabet");
        }
        projection.bit[cell] = (unsigned char)decoded;
        projection.ones += (size_t)decoded;
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

static void btt_free_projection(struct btt_projection *projection) {
    free(projection->bit);
    projection->bit = NULL;
    projection->cells = 0U;
}

static struct process btt_carrier_process(
    const struct btt_layout *layout
) {
    struct process process = {
        .carrier_cells = layout->carrier_cells
    };
    return process;
}

static struct btt_execution btt_execute(
    struct carrier *carrier,
    const struct btt_layout *layout,
    enum btt_variant variant,
    enum btt_mode mode
) {
    if (carrier == NULL || layout == NULL) {
        fail("null Boolean-TT carrier");
    }
    if (carrier->cells != layout->carrier_cells) {
        fail("Boolean-TT carrier topology mismatch");
    }
    struct carrier borrowed = snapshot_carrier(carrier);
    struct btt_execution execution = {0};

    btt_encode_leaf(
        carrier, layout, &layout->f, variant, 0, &execution.stats
    );
    btt_encode_leaf(
        carrier, layout, &layout->g, variant, 0, &execution.stats
    );
    btt_encode_leaf(
        carrier, layout, &layout->j, variant, 0, &execution.stats
    );
    btt_compose(
        carrier,
        layout,
        &layout->f,
        &layout->g,
        &layout->h,
        0,
        &execution.stats
    );
    btt_compose(
        carrier,
        layout,
        &layout->h,
        &layout->j,
        &layout->z,
        0,
        &execution.stats
    );
    btt_copy_block(
        carrier,
        &layout->z,
        &layout->boundary,
        0,
        0,
        &execution.stats
    );
    execution.projection = btt_latch_boundary(
        carrier, &layout->boundary, &execution.stats
    );
    execution.displacement_l2 = displacement(carrier, &borrowed);

    if (mode == BTT_SNAPSHOT_RELOAD) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        execution.snapshot_loaded = 1;
    } else {
        btt_copy_block(
            carrier,
            &layout->z,
            &layout->boundary,
            1,
            mode == BTT_WRONG_BOUNDARY_INVERSE,
            &execution.stats
        );
        if (mode == BTT_REORDERED_H_BEFORE_Z_INVERSE) {
            btt_compose(
                carrier,
                layout,
                &layout->f,
                &layout->g,
                &layout->h,
                1,
                &execution.stats
            );
        }
        btt_compose(
            carrier,
            layout,
            &layout->h,
            &layout->j,
            &layout->z,
            1,
            &execution.stats
        );
        if (
            mode != BTT_MISSING_H_INVERSE
            && mode != BTT_REORDERED_H_BEFORE_Z_INVERSE
        ) {
            btt_compose(
                carrier,
                layout,
                &layout->f,
                &layout->g,
                &layout->h,
                1,
                &execution.stats
            );
        }
        btt_encode_leaf(
            carrier, layout, &layout->j, variant, 1, &execution.stats
        );
        btt_encode_leaf(
            carrier, layout, &layout->g, variant, 1, &execution.stats
        );
        btt_encode_leaf(
            carrier, layout, &layout->f, variant, 1, &execution.stats
        );
    }

    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
    free_carrier(&borrowed);
    return execution;
}

static uint64_t btt_program_hash(
    const struct btt_layout *layout,
    enum btt_variant variant
) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t leaf = 0U; leaf < 3U; ++leaf) {
        (void)leaf;
        for (size_t site = 0U; site < layout->width; ++site) {
            const size_t left_rank =
                site == 0U ? 1U : BTT_LEAF_RANK;
            const size_t right_rank =
                site + 1U == layout->width ? 1U : BTT_LEAF_RANK;
            for (size_t left = 0U; left < left_rank; ++left) {
                for (size_t x = 0U; x < 2U; ++x) {
                    for (size_t y = 0U; y < 2U; ++y) {
                        for (
                            size_t right = 0U;
                            right < right_rank;
                            ++right
                        ) {
                            const unsigned char bit = btt_leaf_bit(
                                variant,
                                layout,
                                site,
                                left,
                                x,
                                y,
                                right
                            );
                            hash = hash_bytes(hash, &bit, 1U);
                        }
                    }
                }
            }
        }
    }
    return hash;
}

static uint64_t btt_fibonacci(size_t ordinal) {
    uint64_t previous = 0U;
    uint64_t current = 1U;
    for (size_t index = 0U; index < ordinal; ++index) {
        const uint64_t next = previous + current;
        previous = current;
        current = next;
    }
    return previous;
}

static void btt_require_correct(
    const struct btt_execution *execution,
    const struct btt_layout *layout
) {
    const uint64_t expected_ands =
        4U * (uint64_t)(layout->n4 + layout->n8);
    const uint64_t expected_ors =
        2U * (uint64_t)(layout->n4 + layout->n8);
    const uint64_t expected_reads =
        8U * (uint64_t)(layout->n4 + layout->n8)
        + 3U * (uint64_t)layout->n8;
    const uint64_t expected_updates =
        6U * (uint64_t)layout->n2
        + 2U * (uint64_t)layout->n4
        + 4U * (uint64_t)layout->n8;
    if (
        execution->projection.cells != layout->n8
        || execution->stats.logical_phase_ands != expected_ands
        || execution->stats.logical_phase_ors != expected_ors
        || execution->stats.phase_products_inside_or != expected_ors
        || execution->stats.carrier_reads != expected_reads
        || execution->stats.phase_cell_updates != expected_updates
        || execution->stats.final_decodes != layout->n8
        || execution->projection.maximum_root_error > ROOT_TOLERANCE
        || execution->restoration_max_abs > RESTORATION_TOLERANCE
        || execution->integrity_max_abs > RESTORATION_TOLERANCE
        || execution->snapshot_loaded
    ) {
        fail("correct Boolean-TT execution violated exact custody law");
    }
}

static struct btt_execution btt_run_control(
    const struct btt_layout *layout,
    enum btt_mode mode
) {
    const struct process process = btt_carrier_process(layout);
    struct carrier carrier = make_carrier(
        &process, 7301 + (int)mode
    );
    struct btt_execution execution = btt_execute(
        &carrier, layout, BTT_NEIGHBOR_AND, mode
    );
    free_carrier(&carrier);
    return execution;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--project") == 0) {
        (void)btt_parse_width(argv[2]);
        fail("only the final rank-eight Boolean-TT boundary may be projected");
    }
    if (argc == 3 && strcmp(argv[1], "--dense") == 0) {
        (void)btt_parse_width(argv[2]);
        fail("dense 4^width relation materialization is forbidden");
    }
    if (argc == 3 && strcmp(argv[1], "--rank-cap-7") == 0) {
        (void)btt_parse_width(argv[2]);
        fail("public rank cap 7 rejects required rank-eight root");
    }
    if (argc == 3 && strcmp(argv[1], "--null") == 0) {
        const size_t width = btt_parse_width(argv[2]);
        const struct btt_layout layout = btt_make_layout(width);
        (void)btt_execute(
            NULL, &layout, BTT_NEIGHBOR_AND, BTT_CORRECT
        );
    }
    if (argc != 2) {
        fail(
            "usage: algebraic_boolean_tt_phase "
            "<width 4..16>"
        );
    }
    const size_t width = btt_parse_width(argv[1]);
    const struct btt_layout layout = btt_make_layout(width);
    const struct process process = btt_carrier_process(&layout);
    struct carrier carrier = make_carrier(&process, 7201);

    struct btt_execution primary = btt_execute(
        &carrier, &layout, BTT_NEIGHBOR_AND, BTT_CORRECT
    );
    btt_require_correct(&primary, &layout);
    struct btt_execution reuse = btt_execute(
        &carrier, &layout, BTT_NEIGHBOR_NAND, BTT_CORRECT
    );
    btt_require_correct(&reuse, &layout);
    if (primary.projection.hash == reuse.projection.hash) {
        fail("unrelated Boolean-TT reuse did not change final boundary");
    }

    double repeated_max_restoration = 0.0;
    for (size_t cycle = 0U; cycle < BTT_REUSE_CYCLES; ++cycle) {
        struct btt_execution repeated = btt_execute(
            &carrier,
            &layout,
            cycle % 2U == 0U
                ? BTT_NEIGHBOR_AND
                : BTT_NEIGHBOR_NAND,
            BTT_CORRECT
        );
        btt_require_correct(&repeated, &layout);
        if (repeated.restoration_max_abs > repeated_max_restoration) {
            repeated_max_restoration = repeated.restoration_max_abs;
        }
        btt_free_projection(&repeated.projection);
    }
    free_carrier(&carrier);

    struct btt_execution wrong = btt_run_control(
        &layout, BTT_WRONG_BOUNDARY_INVERSE
    );
    struct btt_execution missing = btt_run_control(
        &layout, BTT_MISSING_H_INVERSE
    );
    struct btt_execution reordered = btt_run_control(
        &layout, BTT_REORDERED_H_BEFORE_Z_INVERSE
    );
    struct btt_execution snapshot = btt_run_control(
        &layout, BTT_SNAPSHOT_RELOAD
    );
    if (
        wrong.restoration_max_abs < CONTROL_MINIMUM
        || missing.restoration_max_abs < CONTROL_MINIMUM
        || reordered.restoration_max_abs < CONTROL_MINIMUM
        || snapshot.restoration_max_abs > RESTORATION_TOLERANCE
        || !snapshot.snapshot_loaded
    ) {
        fail("Boolean-TT inverse or snapshot control did not separate");
    }

    printf(
        "{\"mode\":\"width-parametric-boolean-tt\","
        "\"claim\":\"BOUNDED_WIDTH_PARAMETRIC_BOOLEAN_TT_MANY_TO_MANY_"
        "RELATION_COMPOSITION_WITH_PRODUCT_RANK_NATIVE_PHASE_CLOSURE_"
        "AND_RESIDENT_INTERMEDIATE\","
        "\"width\":%zu,"
        "\"leaf_rank\":2,"
        "\"resident_h_rank\":4,"
        "\"final_z_rank\":8,"
        "\"leaf_cells_each\":%zu,"
        "\"resident_h_cells\":%zu,"
        "\"final_z_cells\":%zu,"
        "\"boundary_cells\":%zu,"
        "\"carrier_cells\":%zu,"
        "\"live_carrier_bytes\":%zu,"
        "\"verification_snapshot_bytes\":%zu,"
        "\"projected_boundary_bytes\":%zu,"
        "\"primary_program_fnv1a64\":\"%016llx\","
        "\"reuse_program_fnv1a64\":\"%016llx\","
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"primary_boundary_ones\":%zu,"
        "\"reuse_boundary_ones\":%zu,"
        "\"phase_cell_updates_per_transaction\":%llu,"
        "\"logical_phase_ands_per_transaction\":%llu,"
        "\"logical_phase_ors_per_transaction\":%llu,"
        "\"phase_products_inside_or_per_transaction\":%llu,"
        "\"carrier_reads_per_transaction\":%llu,"
        "\"final_decodes_per_transaction\":%llu,"
        "\"decoded_intermediate_cells\":0,"
        "\"serialized_intermediate_cells\":0,"
        "\"copied_intermediate_cells\":0,"
        "\"shared_assignment_loops\":0,"
        "\"local_shared_values_per_output_core\":2,"
        "\"dense_relation_cells_materialized\":0,"
        "\"truth_table_slots\":0,"
        "\"full_output_tt_materialized\":true,"
        "\"actual_resident_h_consumed_by_z\":true,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"same_carrier_transactions\":%u,"
        "\"restoration_generation\":%u,"
        "\"repeated_reuse_max_restoration_abs\":%.12g,"
        "\"restoration_tolerance\":%.12g,"
        "\"primary_many_to_many\":true,"
        "\"leaf_outputs_per_input\":2,"
        "\"leaf_global_accepted_pair_count\":\"2^(%zu+1)\","
        "\"root_outputs_per_input_lower_bound\":2,"
        "\"root_inputs_for_zero_output_lower_bound\":%llu,"
        "\"dense_relation_counterfactual_cells\":\"4^%zu\","
        "\"rank8_smaller_than_dense\":%s,"
        "\"degree4_window_count\":%zu,"
        "\"analytic_fourth_boolean_derivative\":1,"
        "\"analytic_non_affine_certificate\":true,"
        "\"fixed_rank_unbounded_depth_closure\":false,"
        "\"finite_product_rank_closure\":true,"
        "\"rank_minimization_or_recompression\":false,"
        "\"wrong_inverse_restoration_error\":%.12g,"
        "\"missing_inverse_restoration_error\":%.12g,"
        "\"reordered_inverse_restoration_error\":%.12g,"
        "\"snapshot_restoration_error\":%.12g,"
        "\"snapshot_loaded\":true,"
        "\"snapshot_restoration_generation\":0,"
        "\"terminal\":false}\n",
        width,
        layout.n2,
        layout.n4,
        layout.n8,
        layout.n8,
        layout.carrier_cells,
        2U * layout.carrier_cells * sizeof(double complex),
        2U * layout.carrier_cells * sizeof(double complex),
        layout.n8 * sizeof(unsigned char),
        (unsigned long long)btt_program_hash(
            &layout, BTT_NEIGHBOR_AND
        ),
        (unsigned long long)btt_program_hash(
            &layout, BTT_NEIGHBOR_NAND
        ),
        (unsigned long long)primary.projection.hash,
        (unsigned long long)reuse.projection.hash,
        primary.projection.ones,
        reuse.projection.ones,
        (unsigned long long)primary.stats.phase_cell_updates,
        (unsigned long long)primary.stats.logical_phase_ands,
        (unsigned long long)primary.stats.logical_phase_ors,
        (unsigned long long)primary.stats.phase_products_inside_or,
        (unsigned long long)primary.stats.carrier_reads,
        (unsigned long long)primary.stats.final_decodes,
        (unsigned int)(BTT_REUSE_CYCLES + 2U),
        (unsigned int)(BTT_REUSE_CYCLES + 2U),
        repeated_max_restoration,
        RESTORATION_TOLERANCE,
        width,
        (unsigned long long)btt_fibonacci(width + 2U),
        width,
        width >= 5U ? "true" : "false",
        width - 3U,
        wrong.restoration_max_abs,
        missing.restoration_max_abs,
        reordered.restoration_max_abs,
        snapshot.restoration_max_abs
    );

    btt_free_projection(&primary.projection);
    btt_free_projection(&reuse.projection);
    btt_free_projection(&wrong.projection);
    btt_free_projection(&missing.projection);
    btt_free_projection(&reordered.projection);
    btt_free_projection(&snapshot.projection);
    return 0;
}
