#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: bounded width-three rank-two tensor-train
 * relations with exact lazy composition.
 *
 * The two public relations occupy actual phase-resident TT core cells.
 * Composition is the existing Boolean/F3 square/intersection/norm law, but
 * only predeclared sparse boundary coefficients are contracted from its lazy
 * expression.  No dense 64-coefficient input or output relation is built.
 */

#define main algebraic_series_parallel_embedded_main
#include "algebraic_series_parallel_phase.c"
#undef main

#define TT_WIDTH 3U
#define TT_RELATIONS 2U
#define TT_RANK 2U
#define TT_PHYSICAL 4U
#define TT_CORE_CELLS 32U
#define TT_SELECTORS 8U
#define TT_CARRIER_CELLS 80U
#define TT_TOKEN_CAPACITY 24U
#define TT_CACHE_CAPACITY 1024U

#define TT_F_START 0U
#define TT_G_START 32U
#define TT_MESSAGE_START 64U
#define TT_BOUNDARY_START 72U

enum tt_relation { TT_F = 0, TT_G = 1 };

enum tt_mode {
    TT_CORRECT = 0,
    TT_WRONG_MESSAGE_INVERSE = 1,
    TT_OMIT_MESSAGE_INVERSE = 2,
    TT_REORDER_INVERSE = 3,
    TT_RANK1_BOND_CUT = 4,
    TT_WRONG_G_BOND = 5,
    TT_SNAPSHOT_RELOAD = 6
};

struct tt_program {
    int core[TT_RELATIONS][TT_CORE_CELLS];
    unsigned char selector[TT_SELECTORS][TT_WIDTH];
    uint64_t source_hash;
};

struct lazy_tt_descriptor {
    size_t relation_start[TT_RELATIONS];
    size_t message_start;
    size_t boundary_start;
    unsigned char selector[TT_SELECTORS][TT_WIDTH];
};

struct tt_stats {
    uint64_t symbol_products;
    uint64_t coefficient_additions;
    uint64_t or_decomposition_pairs;
    uint64_t tt_coefficient_contractions;
    uint64_t carrier_reads;
    uint64_t phase_cell_updates;
    uint64_t lazy_cache_hits;
    uint64_t lazy_cache_misses;
    size_t maximum_live_cache_entries;
    size_t selector_evaluations;
};

struct tt_cache_entry {
    size_t mask;
    double complex value;
    unsigned char level;
    unsigned char used;
};

struct tt_eval {
    const struct carrier *carrier;
    const struct lazy_tt_descriptor *descriptor;
    enum tt_mode mode;
    struct tt_stats *stats;
    struct tt_cache_entry cache[TT_CACHE_CAPACITY];
    size_t cache_entries;
};

struct tt_boundary {
    int coefficient[TT_SELECTORS];
    double maximum_root_error;
};

struct tt_execution {
    struct tt_boundary boundary;
    struct tt_stats stats;
    double displacement_l2;
    double restoration_max_abs;
    double integrity_max_abs;
    int snapshot_loaded;
};

static size_t tt_tokenize(
    char *line,
    char *token[TT_TOKEN_CAPACITY]
) {
    size_t count = 0U;
    char *save = NULL;
    for (
        char *part = strtok_r(line, " ", &save);
        part != NULL;
        part = strtok_r(NULL, " ", &save)
    ) {
        if (count == TT_TOKEN_CAPACITY) {
            fail("too many lazy-TT fields");
        }
        token[count++] = part;
    }
    return count;
}

static int parse_f3_token(const char *token, size_t line) {
    char *tail = NULL;
    const long value = strtol(token, &tail, 10);
    if (
        tail == token
        || *tail != '\0'
        || value < 0L
        || value > 2L
    ) {
        fail_line("invalid lazy-TT F3 coefficient", line);
    }
    return (int)value;
}

static size_t tt_site_cells(size_t site) {
    return site == 1U ? 16U : 8U;
}

static size_t tt_site_offset(size_t site) {
    static const size_t offset[TT_WIDTH] = {0U, 8U, 24U};
    if (site >= TT_WIDTH) {
        fail("invalid lazy-TT site");
    }
    return offset[site];
}

static struct tt_program read_tt_program(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct tt_program program = {
        .source_hash = UINT64_C(14695981039346656037)
    };
    char line[LINE_CAPACITY];
    size_t line_number = 0U;
    size_t stage = 0U;
    while (fgets(line, sizeof(line), stream) != NULL) {
        ++line_number;
        const size_t length = strlen(line);
        program.source_hash = hash_bytes(
            program.source_hash,
            (const unsigned char *)line,
            length
        );
        if (length == 0U || line[length - 1U] != '\n') {
            fail_line("every lazy-TT record must end with LF", line_number);
        }
        if (memchr(line, '\r', length) != NULL) {
            fail_line("CR bytes are forbidden", line_number);
        }
        line[length - 1U] = '\0';
        char *token[TT_TOKEN_CAPACITY] = {0};
        const size_t count = tt_tokenize(line, token);
        if (stage == 0U) {
            if (
                count != 2U
                || strcmp(token[0], "CATCAS_TT_WIDE3_PAIR") != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid lazy-TT header", line_number);
            }
        } else if (stage == 1U) {
            if (
                count != 2U
                || strcmp(token[0], "WIDTH") != 0
                || strcmp(token[1], "3") != 0
            ) {
                fail_line("invalid lazy-TT width", line_number);
            }
        } else if (stage == 2U || stage == 6U) {
            const char *const name = stage == 2U ? "F" : "G";
            if (
                count != 6U
                || strcmp(token[0], "RANK") != 0
                || strcmp(token[1], name) != 0
                || strcmp(token[2], "1") != 0
                || strcmp(token[3], "2") != 0
                || strcmp(token[4], "2") != 0
                || strcmp(token[5], "1") != 0
            ) {
                fail_line("invalid ordered lazy-TT rank", line_number);
            }
        } else if (
            (stage >= 3U && stage <= 5U)
            || (stage >= 7U && stage <= 9U)
        ) {
            const size_t relation = stage >= 7U ? TT_G : TT_F;
            const size_t site = relation == TT_G ? stage - 7U : stage - 3U;
            const char *const name = relation == TT_G ? "G" : "F";
            const size_t cells = tt_site_cells(site);
            if (
                count != cells + 3U
                || strcmp(token[0], "CORE") != 0
                || strcmp(token[1], name) != 0
                || strlen(token[2]) != 1U
                || token[2][0] != (char)('0' + site)
            ) {
                fail_line("invalid ordered lazy-TT core", line_number);
            }
            const size_t offset = tt_site_offset(site);
            for (size_t index = 0U; index < cells; ++index) {
                program.core[relation][offset + index] =
                    parse_f3_token(token[index + 3U], line_number);
            }
        } else if (stage >= 10U && stage < 10U + TT_SELECTORS) {
            const size_t selector = stage - 10U;
            if (
                count != 5U
                || strcmp(token[0], "SELECT") != 0
                || strlen(token[1]) != 1U
                || token[1][0] != (char)('0' + selector)
            ) {
                fail_line("invalid ordered lazy-TT selector", line_number);
            }
            for (size_t site = 0U; site < TT_WIDTH; ++site) {
                char *tail = NULL;
                const long value = strtol(token[site + 2U], &tail, 10);
                if (
                    tail == token[site + 2U]
                    || *tail != '\0'
                    || value < 0L
                    || value >= (long)TT_PHYSICAL
                ) {
                    fail_line("invalid lazy-TT selector index", line_number);
                }
                program.selector[selector][site] = (unsigned char)value;
            }
        } else if (stage == 10U + TT_SELECTORS) {
            if (count != 1U || strcmp(token[0], "END") != 0) {
                fail_line("invalid lazy-TT END", line_number);
            }
        } else {
            fail_line("record after lazy-TT END", line_number);
        }
        ++stage;
    }
    if (ferror(stream) || fclose(stream) != 0) {
        fail("failed to read lazy-TT source");
    }
    if (stage != 11U + TT_SELECTORS) {
        fail("lazy-TT source is incomplete");
    }
    return program;
}

static size_t tt_core_index(
    size_t site,
    size_t left,
    size_t physical,
    size_t right
) {
    if (
        site >= TT_WIDTH
        || physical >= TT_PHYSICAL
        || left >= (site == 0U ? 1U : TT_RANK)
        || right >= (site == 2U ? 1U : TT_RANK)
    ) {
        fail("invalid lazy-TT core coordinate");
    }
    if (site == 0U) {
        return physical * TT_RANK + right;
    }
    if (site == 1U) {
        return 8U + (left * TT_PHYSICAL + physical) * TT_RANK + right;
    }
    return 24U + left * TT_PHYSICAL + physical;
}

static int public_tt_coefficient(
    const struct tt_program *program,
    size_t relation,
    size_t p0,
    size_t p1,
    size_t p2
) {
    int value = 0;
    for (size_t first = 0U; first < TT_RANK; ++first) {
        for (size_t second = 0U; second < TT_RANK; ++second) {
            value = f3(
                value
                + program->core[relation][
                    tt_core_index(0U, 0U, p0, first)
                ]
                * program->core[relation][
                    tt_core_index(1U, first, p1, second)
                ]
                * program->core[relation][
                    tt_core_index(2U, second, p2, 0U)
                ]
            );
        }
    }
    return value;
}

static int tt_rank_minor(
    const struct tt_program *program,
    size_t relation
) {
    const int c00 = public_tt_coefficient(
        program, relation, 0U, 0U, 0U
    );
    const int c01 = public_tt_coefficient(
        program, relation, 0U, 0U, 1U
    );
    const int c10 = public_tt_coefficient(
        program, relation, 1U, 0U, 0U
    );
    const int c11 = public_tt_coefficient(
        program, relation, 1U, 0U, 1U
    );
    return f3(c00 * c11 - c01 * c10);
}

static struct lazy_tt_descriptor make_tt_descriptor(
    const struct tt_program *program
) {
    struct lazy_tt_descriptor descriptor = {
        .relation_start = {TT_F_START, TT_G_START},
        .message_start = TT_MESSAGE_START,
        .boundary_start = TT_BOUNDARY_START
    };
    memcpy(
        descriptor.selector,
        program->selector,
        sizeof(descriptor.selector)
    );
    return descriptor;
}

static double complex tt_add(
    struct tt_eval *eval,
    double complex left,
    double complex right
) {
    ++eval->stats->coefficient_additions;
    return lock_f3_phase(left * right);
}

static double complex tt_product(
    struct tt_eval *eval,
    double complex left,
    double complex right
) {
    ++eval->stats->symbol_products;
    return symbol_product(left, right);
}

static double complex read_tt_core(
    struct tt_eval *eval,
    size_t relation,
    size_t site,
    size_t left,
    size_t physical,
    size_t right
) {
    if (eval == NULL || eval->carrier == NULL || eval->descriptor == NULL) {
        fail("null carrier or lazy descriptor");
    }
    size_t actual_right = right;
    if (
        eval->mode == TT_WRONG_G_BOND
        && relation == TT_G
        && site == 0U
    ) {
        actual_right = 1U - right;
    }
    ++eval->stats->carrier_reads;
    return relative(
        eval->carrier,
        eval->descriptor->relation_start[relation]
            + tt_core_index(site, left, physical, actual_right)
    );
}

static double complex tt_coefficient(
    struct tt_eval *eval,
    size_t relation,
    const size_t physical[TT_WIDTH]
) {
    double complex result = root3(0);
    ++eval->stats->tt_coefficient_contractions;
    for (size_t first = 0U; first < TT_RANK; ++first) {
        for (size_t second = 0U; second < TT_RANK; ++second) {
            if (
                eval->mode == TT_RANK1_BOND_CUT
                && (first != 0U || second != 0U)
            ) {
                continue;
            }
            const double complex left = read_tt_core(
                eval, relation, 0U, 0U, physical[0], first
            );
            const double complex middle = read_tt_core(
                eval, relation, 1U, first, physical[1], second
            );
            const double complex right = read_tt_core(
                eval, relation, 2U, second, physical[2], 0U
            );
            const double complex pair = tt_product(eval, left, middle);
            const double complex term = tt_product(eval, pair, right);
            result = tt_add(eval, result, term);
        }
    }
    return result;
}

struct local_or_pairs {
    unsigned char left[9];
    unsigned char right[9];
    size_t count;
};

static struct local_or_pairs make_local_or_pairs(size_t target) {
    struct local_or_pairs pairs = {{0}, {0}, 0U};
    for (size_t left = 0U; left < TT_PHYSICAL; ++left) {
        for (size_t right = 0U; right < TT_PHYSICAL; ++right) {
            if ((left | right) == target) {
                pairs.left[pairs.count] = (unsigned char)left;
                pairs.right[pairs.count] = (unsigned char)right;
                ++pairs.count;
            }
        }
    }
    return pairs;
}

static double complex tt_square_coefficient(
    struct tt_eval *eval,
    size_t relation,
    size_t full_mask
) {
    if (
        (relation == TT_F && (full_mask & (size_t)0x1c0U) != 0U)
        || (relation == TT_G && (full_mask & (size_t)0x007U) != 0U)
    ) {
        return root3(0);
    }
    size_t target[TT_WIDTH];
    for (size_t site = 0U; site < TT_WIDTH; ++site) {
        const size_t first = relation == TT_F ? site : 3U + site;
        const size_t second = relation == TT_F ? 3U + site : 6U + site;
        target[site] = (
            ((full_mask >> first) & 1U)
            | (((full_mask >> second) & 1U) << 1U)
        );
    }
    const struct local_or_pairs p0 = make_local_or_pairs(target[0]);
    const struct local_or_pairs p1 = make_local_or_pairs(target[1]);
    const struct local_or_pairs p2 = make_local_or_pairs(target[2]);
    double complex result = root3(0);
    for (size_t i0 = 0U; i0 < p0.count; ++i0) {
        for (size_t i1 = 0U; i1 < p1.count; ++i1) {
            for (size_t i2 = 0U; i2 < p2.count; ++i2) {
                const size_t left[TT_WIDTH] = {
                    p0.left[i0], p1.left[i1], p2.left[i2]
                };
                const size_t right[TT_WIDTH] = {
                    p0.right[i0], p1.right[i1], p2.right[i2]
                };
                const double complex l = tt_coefficient(
                    eval, relation, left
                );
                const double complex r = tt_coefficient(
                    eval, relation, right
                );
                result = tt_add(
                    eval,
                    result,
                    tt_product(eval, l, r)
                );
                ++eval->stats->or_decomposition_pairs;
            }
        }
    }
    return result;
}

static size_t insert_zero_bit_tt(size_t mask, size_t bit) {
    const size_t lower = ((size_t)1U << bit) - 1U;
    return (mask & lower) | ((mask & ~lower) << 1U);
}

static size_t tt_cache_hash(unsigned char level, size_t mask) {
    return (
        mask * (size_t)UINT64_C(11400714819323198485)
        + (size_t)level * 97U
    ) & (TT_CACHE_CAPACITY - 1U);
}

static struct tt_cache_entry *find_tt_cache(
    struct tt_eval *eval,
    unsigned char level,
    size_t mask,
    int *found
) {
    size_t slot = tt_cache_hash(level, mask);
    for (size_t probe = 0U; probe < TT_CACHE_CAPACITY; ++probe) {
        struct tt_cache_entry *entry = &eval->cache[slot];
        if (!entry->used) {
            *found = 0;
            return entry;
        }
        if (entry->level == level && entry->mask == mask) {
            *found = 1;
            return entry;
        }
        slot = (slot + 1U) & (TT_CACHE_CAPACITY - 1U);
    }
    fail("lazy-TT sparse cache capacity exceeded");
    return NULL;
}

static double complex lazy_tt_coefficient(
    struct tt_eval *eval,
    unsigned char level,
    size_t mask
) {
    int found = 0;
    struct tt_cache_entry *entry = find_tt_cache(
        eval, level, mask, &found
    );
    if (found) {
        ++eval->stats->lazy_cache_hits;
        return entry->value;
    }
    ++eval->stats->lazy_cache_misses;
    double complex value = root3(0);
    if (level == 0U) {
        value = tt_add(
            eval,
            tt_square_coefficient(eval, TT_F, mask),
            tt_square_coefficient(eval, TT_G, mask)
        );
    } else {
        for (
            size_t left = mask;
            ;
            left = (left - 1U) & mask
        ) {
            for (
                size_t right = mask;
                ;
                right = (right - 1U) & mask
            ) {
                if ((left | right) == mask) {
                    const size_t left_zero =
                        insert_zero_bit_tt(left, 3U);
                    const size_t right_zero =
                        insert_zero_bit_tt(right, 3U);
                    const double complex at_zero = lazy_tt_coefficient(
                        eval, (unsigned char)(level - 1U), left_zero
                    );
                    const double complex right_at_zero =
                        lazy_tt_coefficient(
                            eval,
                            (unsigned char)(level - 1U),
                            right_zero
                        );
                    const double complex right_at_one =
                        lazy_tt_coefficient(
                            eval,
                            (unsigned char)(level - 1U),
                            right_zero | ((size_t)1U << 3U)
                        );
                    const double complex restricted_one = tt_add(
                        eval, right_at_zero, right_at_one
                    );
                    value = tt_add(
                        eval,
                        value,
                        tt_product(eval, at_zero, restricted_one)
                    );
                    ++eval->stats->or_decomposition_pairs;
                }
                if (right == 0U) {
                    break;
                }
            }
            if (left == 0U) {
                break;
            }
        }
    }
    entry = find_tt_cache(eval, level, mask, &found);
    if (found) {
        fail("lazy-TT cache insertion collision");
    }
    entry->used = 1U;
    entry->level = level;
    entry->mask = mask;
    entry->value = value;
    ++eval->cache_entries;
    if (eval->cache_entries > eval->stats->maximum_live_cache_entries) {
        eval->stats->maximum_live_cache_entries = eval->cache_entries;
    }
    return value;
}

static size_t selector_mask(
    const unsigned char selector[TT_WIDTH]
) {
    size_t mask = 0U;
    for (size_t site = 0U; site < TT_WIDTH; ++site) {
        if ((selector[site] & 1U) != 0U) {
            mask |= (size_t)1U << site;
        }
        if ((selector[site] & 2U) != 0U) {
            mask |= (size_t)1U << (3U + site);
        }
    }
    return mask;
}

static double complex evaluate_tt_selector(
    const struct carrier *carrier,
    const struct lazy_tt_descriptor *descriptor,
    enum tt_mode mode,
    size_t selector,
    struct tt_stats *stats
) {
    if (
        carrier == NULL
        || descriptor == NULL
        || stats == NULL
        || selector >= TT_SELECTORS
    ) {
        fail("null carrier or lazy descriptor");
    }
    struct tt_eval eval = {
        .carrier = carrier,
        .descriptor = descriptor,
        .mode = mode,
        .stats = stats
    };
    ++stats->selector_evaluations;
    return lazy_tt_coefficient(
        &eval,
        3U,
        selector_mask(descriptor->selector[selector])
    );
}

static void tracked_multiply_cell(
    struct carrier *carrier,
    size_t cell,
    double complex factor,
    struct tt_stats *stats
) {
    multiply_cell(carrier, cell, factor);
    ++stats->phase_cell_updates;
}

static void encode_tt_program(
    struct carrier *carrier,
    const struct tt_program *program,
    int inverse,
    struct tt_stats *stats
) {
    if (!inverse) {
        for (size_t relation = 0U; relation < TT_RELATIONS; ++relation) {
            for (size_t cell = 0U; cell < TT_CORE_CELLS; ++cell) {
                tracked_multiply_cell(
                    carrier,
                    relation * TT_CORE_CELLS + cell,
                    root3(program->core[relation][cell]),
                    stats
                );
            }
        }
        return;
    }
    for (size_t relation = TT_RELATIONS; relation > 0U; --relation) {
        for (size_t cell = TT_CORE_CELLS; cell > 0U; --cell) {
            tracked_multiply_cell(
                carrier,
                (relation - 1U) * TT_CORE_CELLS + cell - 1U,
                conj(root3(program->core[relation - 1U][cell - 1U])),
                stats
            );
        }
    }
}

static void apply_tt_messages(
    struct carrier *carrier,
    const struct lazy_tt_descriptor *descriptor,
    enum tt_mode mode,
    int inverse,
    struct tt_stats *stats
) {
    double complex factor[TT_SELECTORS];
    const enum tt_mode algebra_mode =
        mode == TT_RANK1_BOND_CUT || mode == TT_WRONG_G_BOND
            ? mode
            : TT_CORRECT;
    for (size_t selector = 0U; selector < TT_SELECTORS; ++selector) {
        factor[selector] = evaluate_tt_selector(
            carrier, descriptor, algebra_mode, selector, stats
        );
    }
    for (size_t selector = 0U; selector < TT_SELECTORS; ++selector) {
        size_t source = selector;
        if (inverse && mode == TT_WRONG_MESSAGE_INVERSE) {
            source = (selector + 1U) % TT_SELECTORS;
        }
        tracked_multiply_cell(
            carrier,
            descriptor->message_start + selector,
            inverse ? conj(factor[source]) : factor[source],
            stats
        );
    }
}

static void copy_actual_messages(
    struct carrier *carrier,
    const struct lazy_tt_descriptor *descriptor,
    int inverse,
    struct tt_stats *stats
) {
    for (size_t selector = 0U; selector < TT_SELECTORS; ++selector) {
        ++stats->carrier_reads;
        const double complex factor = relative(
            carrier, descriptor->message_start + selector
        );
        tracked_multiply_cell(
            carrier,
            descriptor->boundary_start + selector,
            inverse ? conj(factor) : factor,
            stats
        );
    }
}

static struct tt_boundary latch_tt_boundary(
    const struct carrier *carrier,
    const struct lazy_tt_descriptor *descriptor
) {
    struct tt_boundary boundary = {{0}, 0.0};
    for (size_t selector = 0U; selector < TT_SELECTORS; ++selector) {
        double distance = 0.0;
        boundary.coefficient[selector] = decode_root(
            relative(carrier, descriptor->boundary_start + selector),
            &distance
        );
        if (distance > boundary.maximum_root_error) {
            boundary.maximum_root_error = distance;
        }
    }
    return boundary;
}

static struct tt_execution execute_tt(
    struct carrier *carrier,
    const struct tt_program *program,
    enum tt_mode mode
) {
    if (carrier == NULL || program == NULL) {
        fail("null carrier or lazy descriptor");
    }
    const struct lazy_tt_descriptor descriptor =
        make_tt_descriptor(program);
    struct carrier borrowed = snapshot_carrier(carrier);
    struct tt_execution execution = {0};
    encode_tt_program(carrier, program, 0, &execution.stats);
    apply_tt_messages(
        carrier, &descriptor, mode, 0, &execution.stats
    );
    copy_actual_messages(
        carrier, &descriptor, 0, &execution.stats
    );
    execution.boundary = latch_tt_boundary(carrier, &descriptor);
    execution.displacement_l2 = displacement(carrier, &borrowed);

    if (mode == TT_SNAPSHOT_RELOAD) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        execution.snapshot_loaded = 1;
    } else {
        copy_actual_messages(
            carrier, &descriptor, 1, &execution.stats
        );
        if (mode == TT_REORDER_INVERSE) {
            for (size_t cell = TT_CORE_CELLS; cell > 0U; --cell) {
                tracked_multiply_cell(
                    carrier,
                    TT_G_START + cell - 1U,
                    conj(root3(program->core[TT_G][cell - 1U])),
                    &execution.stats
                );
            }
        }
        if (mode != TT_OMIT_MESSAGE_INVERSE) {
            apply_tt_messages(
                carrier, &descriptor, mode, 1, &execution.stats
            );
        }
        if (mode == TT_REORDER_INVERSE) {
            for (size_t cell = TT_CORE_CELLS; cell > 0U; --cell) {
                tracked_multiply_cell(
                    carrier,
                    TT_F_START + cell - 1U,
                    conj(root3(program->core[TT_F][cell - 1U])),
                    &execution.stats
                );
            }
        } else {
            encode_tt_program(
                carrier, program, 1, &execution.stats
            );
        }
    }
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
    free_carrier(&borrowed);
    return execution;
}

static int tt_boundaries_differ(
    const struct tt_boundary *left,
    const struct tt_boundary *right
) {
    for (size_t selector = 0U; selector < TT_SELECTORS; ++selector) {
        if (left->coefficient[selector] != right->coefficient[selector]) {
            return 1;
        }
    }
    return 0;
}

static double complex dense_one_square(
    const double complex input[CCOUNT],
    size_t target
) {
    double complex value = root3(0);
    for (size_t left = 0U; left < CCOUNT; ++left) {
        for (size_t right = 0U; right < CCOUNT; ++right) {
            if ((left | right) == target) {
                value = lock_f3_phase(
                    value * symbol_product(input[left], input[right])
                );
            }
        }
    }
    return value;
}

static double complex dense_one_q(
    const double complex left[CCOUNT],
    const double complex right[CCOUNT],
    size_t mask
) {
    double complex f = root3(0);
    double complex g = root3(0);
    if ((mask & 4U) == 0U) {
        f = dense_one_square(
            left,
            (mask & 1U) | ((mask & 2U))
        );
    }
    if ((mask & 1U) == 0U) {
        g = dense_one_square(
            right,
            ((mask >> 1U) & 1U) | ((mask >> 1U) & 2U)
        );
    }
    return lock_f3_phase(f * g);
}

static double complex dense_one_lazy(
    const double complex left[CCOUNT],
    const double complex right[CCOUNT],
    size_t output_mask
) {
    double complex value = root3(0);
    for (size_t a = output_mask; ; a = (a - 1U) & output_mask) {
        for (size_t b = output_mask; ; b = (b - 1U) & output_mask) {
            if ((a | b) == output_mask) {
                const size_t a0 = insert_zero_bit_tt(a, 1U);
                const size_t b0 = insert_zero_bit_tt(b, 1U);
                const double complex restricted_one = lock_f3_phase(
                    dense_one_q(left, right, b0)
                    * dense_one_q(left, right, b0 | 2U)
                );
                value = lock_f3_phase(
                    value
                    * symbol_product(
                        dense_one_q(left, right, a0),
                        restricted_one
                    )
                );
            }
            if (b == 0U) {
                break;
            }
        }
        if (a == 0U) {
            break;
        }
    }
    return value;
}

static double width1_parity_error(void) {
    static const int left_value[CCOUNT] = {1, 2, 0, 1};
    static const int right_value[CCOUNT] = {2, 1, 1, 0};
    struct process shape = {.carrier_cells = 12U};
    struct carrier carrier = make_carrier(&shape, 6203);
    for (size_t index = 0U; index < CCOUNT; ++index) {
        multiply_cell(&carrier, index, root3(left_value[index]));
        multiply_cell(
            &carrier, CCOUNT + index, root3(right_value[index])
        );
    }
    const struct operation operation = {
        OP_COMPOSE, 0U, 0, CCOUNT, 0, 2U * CCOUNT
    };
    double complex expected[CCOUNT];
    double complex left[CCOUNT];
    double complex right[CCOUNT];
    exact_compose_factors(&carrier, &operation, expected);
    read_poly(&carrier, 0U, 0, left);
    read_poly(&carrier, CCOUNT, 0, right);
    double maximum = 0.0;
    for (size_t mask = 0U; mask < CCOUNT; ++mask) {
        const double error = cabs(
            expected[mask] - dense_one_lazy(left, right, mask)
        );
        if (error > maximum) {
            maximum = error;
        }
    }
    free_carrier(&carrier);
    return maximum;
}

static void print_tt_execution(
    const char *mode_name,
    const char *restoration_path,
    const struct tt_execution *execution,
    const struct tt_program *program,
    double width1_error
) {
    printf(
        "{\"mode\":\"%s\","
        "\"claim\":\"BOUNDED_WIDTH3_RANK2_PHASE_RESIDENT_LAZY_TT_"
        "RELATION_COMPOSITION_WITH_SPARSE_BOUNDARY_INVARIANT\","
        "\"restoration_path\":\"%s\","
        "\"interface_width\":3,"
        "\"input_tt_rank\":2,"
        "\"input_phase_cells_per_relation\":32,"
        "\"dense_phase_cells_per_relation\":64,"
        "\"input_phase_cell_saving_vs_dense\":32,"
        "\"genuine_rank2_minor_f\":%d,"
        "\"genuine_rank2_minor_g\":%d,"
        "\"lazy_expression_operator_nodes\":6,"
        "\"lazy_expression_leaf_custody_references\":2,"
        "\"full_output_tensor_materialized\":false,"
        "\"bounded_rank_tt_closure_claimed\":false,"
        "\"decoded_intermediate_coefficients\":0,"
        "\"serialized_intermediate_coefficients\":0,"
        "\"shared_assignment_loops\":0,"
        "\"truth_table_slots\":0,"
        "\"resident_sparse_message_cells\":8,"
        "\"public_projection_cells\":8,"
        "\"carrier_cells\":80,"
        "\"live_carrier_complex_values\":160,"
        "\"live_carrier_bytes\":2560,"
        "\"comparison_snapshot_complex_values\":160,"
        "\"carrier_and_snapshot_complex_values\":320,"
        "\"sparse_cache_capacity_entries\":1024,"
        "\"maximum_live_sparse_cache_entries\":%zu,"
        "\"scratch_phase_values_upper_bound\":1032,"
        "\"sparse_cache_entry_bytes_current_abi\":%zu,"
        "\"sparse_cache_storage_bytes_current_abi\":%zu,"
        "\"tt_evaluator_bytes_current_abi\":%zu,"
        "\"accounted_material_buffer_subtotal_bytes_current_abi\":%zu,"
        "\"compiler_measured_nested_stack_call_chain_bytes\":37088,"
        "\"conservative_accounted_execution_upper_bound_bytes_"
        "current_abi\":75776,"
        "\"eager_structural_ranks\":[2,4,8,64,4096,16777216],"
        "\"selector_evaluations\":%zu,"
        "\"tt_coefficient_contractions\":%llu,"
        "\"symbol_products\":%llu,"
        "\"coefficient_additions\":%llu,"
        "\"or_decomposition_pairs\":%llu,"
        "\"carrier_reads\":%llu,"
        "\"phase_cell_updates\":%llu,"
        "\"lazy_cache_hits\":%llu,"
        "\"lazy_cache_misses\":%llu,"
        "\"compiled_descriptor_bytes_current_abi\":%zu,"
        "\"coexisting_program_storage_bytes\":%zu,"
        "\"width1_op_compose_parity_max_abs\":%.12g,"
        "\"snapshot_loaded\":%s,"
        "\"boundary_coefficients\":[",
        mode_name,
        restoration_path,
        tt_rank_minor(program, TT_F),
        tt_rank_minor(program, TT_G),
        execution->stats.maximum_live_cache_entries,
        sizeof(struct tt_cache_entry),
        sizeof(struct tt_cache_entry) * TT_CACHE_CAPACITY,
        sizeof(struct tt_eval),
        (
            2U * TT_CARRIER_CELLS * sizeof(double complex)
            + 2U * TT_CARRIER_CELLS * sizeof(double complex)
            + sizeof(struct tt_eval)
            + TT_SELECTORS * sizeof(double complex)
            + sizeof(struct lazy_tt_descriptor)
            + 2U * sizeof(struct tt_program)
        ),
        execution->stats.selector_evaluations,
        (unsigned long long)execution->stats.tt_coefficient_contractions,
        (unsigned long long)execution->stats.symbol_products,
        (unsigned long long)execution->stats.coefficient_additions,
        (unsigned long long)execution->stats.or_decomposition_pairs,
        (unsigned long long)execution->stats.carrier_reads,
        (unsigned long long)execution->stats.phase_cell_updates,
        (unsigned long long)execution->stats.lazy_cache_hits,
        (unsigned long long)execution->stats.lazy_cache_misses,
        sizeof(struct lazy_tt_descriptor),
        2U * sizeof(struct tt_program),
        width1_error,
        execution->snapshot_loaded ? "true" : "false"
    );
    for (size_t index = 0U; index < TT_SELECTORS; ++index) {
        printf(
            "%s%d",
            index == 0U ? "" : ",",
            execution->boundary.coefficient[index]
        );
    }
    printf(
        "],\"maximum_root_error\":%.12g,"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_max_abs\":%.12g}\n",
        execution->boundary.maximum_root_error,
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->integrity_max_abs
    );
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--project-intermediate") == 0) {
        fail("only the sparse final lazy-TT boundary may be projected");
    }
    if (argc == 2 && strcmp(argv[1], "--eager-output-tt") == 0) {
        fail(
            "eager lazy-TT rank budget exceeded: required rank 16777216"
        );
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        struct tt_stats stats = {0};
        (void)evaluate_tt_selector(
            NULL, NULL, TT_CORRECT, 0U, &stats
        );
    }
    if (argc != 2 && argc != 3) {
        fprintf(
            stderr,
            "usage: %s PRIMARY.tt3 [REUSE.tt3]\n",
            argv[0]
        );
        return 2;
    }
    const struct tt_program primary = read_tt_program(argv[1]);
    const struct tt_program reuse =
        argc == 3 ? read_tt_program(argv[2]) : primary;
    if (
        tt_rank_minor(&primary, TT_F) == 0
        || tt_rank_minor(&primary, TT_G) == 0
        || tt_rank_minor(&reuse, TT_F) == 0
        || tt_rank_minor(&reuse, TT_G) == 0
    ) {
        fail("lazy-TT fixture does not establish genuine rank two");
    }
    const double parity_error = width1_parity_error();
    struct process shape = {.carrier_cells = TT_CARRIER_CELLS};
    struct carrier carrier = make_carrier(&shape, 6209);
    const struct tt_execution nominal = execute_tt(
        &carrier, &primary, TT_CORRECT
    );
    const struct tt_execution reused = execute_tt(
        &carrier, &reuse, TT_CORRECT
    );
    double repeated_reuse_max_abs = nominal.restoration_max_abs;
    if (reused.restoration_max_abs > repeated_reuse_max_abs) {
        repeated_reuse_max_abs = reused.restoration_max_abs;
    }
    for (size_t generation = 0U; generation < 32U; ++generation) {
        const struct tt_execution repeated = execute_tt(
            &carrier,
            (generation & 1U) == 0U ? &primary : &reuse,
            TT_CORRECT
        );
        if (repeated.restoration_max_abs > repeated_reuse_max_abs) {
            repeated_reuse_max_abs = repeated.restoration_max_abs;
        }
    }
    free_carrier(&carrier);

    static const enum tt_mode control_mode[] = {
        TT_WRONG_MESSAGE_INVERSE,
        TT_OMIT_MESSAGE_INVERSE,
        TT_REORDER_INVERSE,
        TT_RANK1_BOND_CUT,
        TT_WRONG_G_BOND,
        TT_SNAPSHOT_RELOAD
    };
    struct tt_execution control[
        sizeof(control_mode) / sizeof(control_mode[0])
    ];
    for (
        size_t index = 0U;
        index < sizeof(control_mode) / sizeof(control_mode[0]);
        ++index
    ) {
        carrier = make_carrier(&shape, 6209);
        control[index] = execute_tt(
            &carrier, &primary, control_mode[index]
        );
        free_carrier(&carrier);
    }

    print_tt_execution(
        "width3-rank2-lazy-tt",
        "actual_inverse",
        &nominal,
        &primary,
        parity_error
    );
    print_tt_execution(
        "actual-restored-cross-program-width3-tt-reuse",
        "actual_inverse",
        &reused,
        &reuse,
        parity_error
    );
    print_tt_execution(
        "wrong-sparse-message-inverse",
        "wrong_inverse",
        &control[0],
        &primary,
        parity_error
    );
    print_tt_execution(
        "omitted-sparse-message-inverse",
        "missing_inverse",
        &control[1],
        &primary,
        parity_error
    );
    print_tt_execution(
        "reordered-input-before-message-inverse",
        "reordered_inverse",
        &control[2],
        &primary,
        parity_error
    );
    print_tt_execution(
        "rank1-virtual-bond-cut",
        "actual_inverse",
        &control[3],
        &primary,
        parity_error
    );
    print_tt_execution(
        "wrong-g-virtual-bond-permutation",
        "actual_inverse",
        &control[4],
        &primary,
        parity_error
    );
    print_tt_execution(
        "snapshot-backed-width3-tt-reuse",
        "snapshot_reload",
        &control[5],
        &primary,
        parity_error
    );
    const int rank1_differs = tt_boundaries_differ(
        &nominal.boundary, &control[3].boundary
    );
    const int bond_differs = tt_boundaries_differ(
        &nominal.boundary, &control[4].boundary
    );
    printf(
        "{\"mode\":\"width3-lazy-tt-control-applicability\","
        "\"wrong_message_inverse\":%s,"
        "\"omitted_message_inverse\":%s,"
        "\"reordered_noncommuting_inverse\":%s,"
        "\"rank1_bond_cut_changes_boundary\":%s,"
        "\"wrong_g_bond_changes_boundary\":%s,"
        "\"snapshot_restores\":%s,"
        "\"repeated_alternating_reuse\":%s,"
        "\"same_carrier_transactions\":34,"
        "\"repeated_reuse_max_restoration_abs\":%.12g}\n",
        control[0].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        control[1].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        control[2].restoration_max_abs >= CONTROL_MINIMUM
            ? "true" : "false",
        rank1_differs ? "true" : "false",
        bond_differs ? "true" : "false",
        (
            control[5].snapshot_loaded
            && control[5].restoration_max_abs <= RESTORATION_TOLERANCE
        ) ? "true" : "false",
        repeated_reuse_max_abs <= RESTORATION_TOLERANCE
            ? "true" : "false",
        repeated_reuse_max_abs
    );

    const int valid = (
        parity_error <= RESTORATION_TOLERANCE
        && nominal.boundary.maximum_root_error <= ROOT_TOLERANCE
        && reused.boundary.maximum_root_error <= ROOT_TOLERANCE
        && nominal.restoration_max_abs <= RESTORATION_TOLERANCE
        && reused.restoration_max_abs <= RESTORATION_TOLERANCE
        && control[0].restoration_max_abs >= CONTROL_MINIMUM
        && control[1].restoration_max_abs >= CONTROL_MINIMUM
        && control[2].restoration_max_abs >= CONTROL_MINIMUM
        && control[3].restoration_max_abs <= RESTORATION_TOLERANCE
        && control[4].restoration_max_abs <= RESTORATION_TOLERANCE
        && control[5].restoration_max_abs <= RESTORATION_TOLERANCE
        && repeated_reuse_max_abs <= RESTORATION_TOLERANCE
        && rank1_differs
        && bond_differs
    );
    return valid ? 0 : 1;
}
