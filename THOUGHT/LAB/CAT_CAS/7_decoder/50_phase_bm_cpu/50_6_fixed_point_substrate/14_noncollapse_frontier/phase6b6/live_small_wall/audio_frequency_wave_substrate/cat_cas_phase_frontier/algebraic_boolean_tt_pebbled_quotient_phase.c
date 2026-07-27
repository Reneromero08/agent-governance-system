#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: topology-derived reversible pebbling for the
 * nonlinear Boolean-TT suffix quotient chain.
 *
 * The retained-all predecessor stores H1..Hd plus a boundary copy.  This
 * machine compiles a public reversible path-pebble tape for
 *
 *     H1 -> H2 -> ... -> Hd
 *
 * where every toggle of Hk consumes the actual resident H(k-1) and H1.
 * Clean phase regions are rebound to later depths only after the actual
 * conjugate operation restores the complete region.  Only Hd is copied and
 * decoded.  The boundary copy is removed, then the exact slot-tagged tape is
 * reversed before H1 is removed.
 *
 * The repair reduces retained phase carrier history while explicitly paying
 * rematerialization work.  It does not change the public Boolean threshold
 * recurrence and therefore cannot establish computational advantage.
 */

#define QTT_EMBEDDED_MAIN qtp_retain_all_embedded_main
#include "algebraic_boolean_tt_suffix_quotient_phase.c"
#undef QTT_EMBEDDED_MAIN

#include <errno.h>

#define QTP_MIN_WIDTH 4U
#define QTP_MAX_WIDTH 16U
#define QTP_MIN_DEPTH 2U
#define QTP_MAX_DEPTH 8U
#define QTP_MAX_PEBBLES 3U
#define QTP_MAX_RAW_MOVES 32U
#define QTP_REUSE_CYCLES 16U

enum qtp_mode {
    QTP_CORRECT = 0,
    QTP_MISSING_INVERSE_MOVE = 1,
    QTP_REORDERED_NONCOMMUTING_INVERSE = 2,
    QTP_SNAPSHOT_RELOAD = 3,
    QTP_WRONG_BOUNDARY_INVERSE = 4,
    QTP_DIRTY_SLOT_INJECTION = 5,
    QTP_GENERATION_TAG_TAMPER = 6
};

struct qtp_move {
    size_t node;
    size_t slot;
    uint64_t generation;
    uint64_t predecessor_generation;
    int add;
};

struct qtp_slot {
    size_t start;
    size_t capacity;
};

struct qtp_plan {
    size_t width;
    size_t depth;
    size_t pebble_count;
    size_t move_count;
    struct qtp_move move[QTP_MAX_RAW_MOVES];
    struct qtp_slot slot[QTP_MAX_PEBBLES];
    struct qtt_stage leaf;
    struct qtt_stage boundary;
    size_t carrier_cells;
    size_t retain_all_carrier_cells;
    size_t active_projection_mask;
    size_t active_projection_cells;
    uint64_t schedule_hash;
};

struct qtp_runtime_slot {
    size_t node;
    uint64_t generation;
};

struct qtp_stats {
    struct qtt_stats phase;
    uint64_t forward_schedule_moves;
    uint64_t inverse_schedule_moves;
    uint64_t stage_additions;
    uint64_t stage_removals;
    uint64_t reconstruction_additions;
    uint64_t reconstruction_cells;
    uint64_t weighted_move_cells;
    uint64_t slot_clean_scans;
    uint64_t slot_rebinds;
};

struct qtp_execution {
    struct qtt_projection projection;
    struct qtp_stats stats;
    double restoration_max_abs;
    double integrity_max_abs;
    double maximum_slot_clean_error;
    uint64_t restoration_generation;
    int actual_inverse;
    int snapshot_loaded;
    int custody_rejected;
    int reordered_applicable;
};

static size_t qtp_parse_canonical(
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

static void qtp_append_raw(
    size_t raw[QTP_MAX_RAW_MOVES],
    size_t *count,
    size_t node
) {
    if (*count >= QTP_MAX_RAW_MOVES || node == 0U) {
        fail("pebble schedule capacity exceeded");
    }
    raw[*count] = node;
    ++*count;
}

static void qtp_generate_recursive(
    size_t pebbles,
    size_t base,
    size_t raw[QTP_MAX_RAW_MOVES],
    size_t *count
) {
    if (pebbles == 0U || pebbles > QTP_MAX_PEBBLES) {
        fail("invalid recursive pebble count");
    }
    if (pebbles == 1U) {
        qtp_append_raw(raw, count, base + 1U);
        return;
    }
    const size_t begin = *count;
    qtp_generate_recursive(
        pebbles - 1U, base, raw, count
    );
    const size_t half = (size_t)1U << (pebbles - 1U);
    qtp_append_raw(raw, count, base + half);
    const size_t prior_end = *count - 1U;
    for (size_t index = prior_end; index > begin; --index) {
        qtp_append_raw(raw, count, raw[index - 1U]);
    }
    qtp_generate_recursive(
        pebbles - 1U, base + half, raw, count
    );
}

static uint64_t qtp_hash_size(uint64_t hash, size_t value) {
    return hash_bytes(
        hash,
        (const unsigned char *)&value,
        sizeof(value)
    );
}

static uint64_t qtp_hash_u64(uint64_t hash, uint64_t value) {
    return hash_bytes(
        hash,
        (const unsigned char *)&value,
        sizeof(value)
    );
}

static uint64_t qtp_plan_hash(const struct qtp_plan *plan) {
    uint64_t hash = UINT64_C(14695981039346656037);
    hash = qtp_hash_size(hash, plan->width);
    hash = qtp_hash_size(hash, plan->depth);
    hash = qtp_hash_size(hash, plan->pebble_count);
    hash = qtp_hash_size(hash, plan->move_count);
    hash = qtp_hash_size(hash, plan->carrier_cells);
    hash = qtp_hash_size(hash, plan->retain_all_carrier_cells);
    hash = qtp_hash_size(hash, plan->active_projection_mask);
    hash = qtp_hash_size(hash, plan->active_projection_cells);
    for (size_t slot = 0U; slot < plan->pebble_count; ++slot) {
        hash = qtp_hash_size(hash, plan->slot[slot].start);
        hash = qtp_hash_size(hash, plan->slot[slot].capacity);
    }
    for (size_t index = 0U; index < plan->move_count; ++index) {
        const struct qtp_move *move = &plan->move[index];
        hash = qtp_hash_size(hash, move->node);
        hash = qtp_hash_size(hash, move->slot);
        hash = qtp_hash_u64(hash, move->generation);
        hash = qtp_hash_u64(
            hash, move->predecessor_generation
        );
        hash = qtp_hash_size(hash, (size_t)move->add);
    }
    return hash;
}

static struct qtp_plan qtp_compile_plan(
    size_t width,
    size_t depth
) {
    struct qtp_plan plan = {
        .width = width,
        .depth = depth
    };
    while (
        ((size_t)1U << plan.pebble_count) < depth
    ) {
        ++plan.pebble_count;
    }
    if (
        plan.pebble_count == 0U
        || plan.pebble_count > QTP_MAX_PEBBLES
    ) {
        fail("unsupported quotient pebble count");
    }
    size_t raw[QTP_MAX_RAW_MOVES] = {0};
    size_t raw_count = 0U;
    qtp_generate_recursive(
        plan.pebble_count, 0U, raw, &raw_count
    );

    size_t active_slot[QTP_MAX_DEPTH] = {0};
    uint64_t active_generation[QTP_MAX_DEPTH] = {0};
    uint64_t generation[QTP_MAX_DEPTH] = {0};
    size_t slot_node[QTP_MAX_PEBBLES] = {0};
    for (size_t node = 0U; node < QTP_MAX_DEPTH; ++node) {
        active_slot[node] = SIZE_MAX;
    }
    const size_t target = depth - 1U;
    int reached = 0;
    for (size_t index = 0U; index < raw_count; ++index) {
        const size_t node = raw[index];
        if (node == 0U || node > target) {
            fail("pebble schedule node outside target path");
        }
        if (
            node > 1U
            && active_generation[node - 1U] == 0U
        ) {
            fail("pebble schedule omitted live predecessor");
        }
        const int add = active_generation[node] == 0U;
        size_t slot = SIZE_MAX;
        if (add) {
            for (
                size_t candidate = 0U;
                candidate < plan.pebble_count;
                ++candidate
            ) {
                if (slot_node[candidate] == 0U) {
                    slot = candidate;
                    break;
                }
            }
            if (slot == SIZE_MAX) {
                fail("pebble schedule exceeded slot bound");
            }
            ++generation[node];
            active_generation[node] = generation[node];
            active_slot[node] = slot;
            slot_node[slot] = node;
            const size_t cells = qtt_stage_cells(
                width, node + 1U
            );
            if (cells > plan.slot[slot].capacity) {
                plan.slot[slot].capacity = cells;
            }
        } else {
            slot = active_slot[node];
            if (
                slot >= plan.pebble_count
                || slot_node[slot] != node
            ) {
                fail("pebble schedule lost slot custody");
            }
        }
        if (plan.move_count >= QTP_MAX_RAW_MOVES) {
            fail("pebble move tape overflow");
        }
        plan.move[plan.move_count++] = (struct qtp_move){
            .node = node,
            .slot = slot,
            .generation = active_generation[node],
            .predecessor_generation = node == 1U
                ? UINT64_C(1)
                : active_generation[node - 1U],
            .add = add
        };
        if (!add) {
            active_generation[node] = 0U;
            active_slot[node] = SIZE_MAX;
            slot_node[slot] = 0U;
        }
        if (add && node == target) {
            reached = 1;
            break;
        }
    }
    if (!reached) {
        fail("pebble schedule did not reach final stage");
    }

    plan.leaf = (struct qtt_stage){
        .start = 0U,
        .depth = 1U,
        .cells = qtt_stage_cells(width, 1U)
    };
    size_t next = plan.leaf.cells;
    for (size_t slot = 0U; slot < plan.pebble_count; ++slot) {
        if (
            plan.slot[slot].capacity == 0U
            || next > SIZE_MAX - plan.slot[slot].capacity
        ) {
            fail("invalid pebble slot capacity");
        }
        plan.slot[slot].start = next;
        next += plan.slot[slot].capacity;
    }
    plan.boundary = (struct qtt_stage){
        .start = next,
        .depth = depth,
        .cells = qtt_stage_cells(width, depth)
    };
    if (next > SIZE_MAX - plan.boundary.cells) {
        fail("pebbled quotient carrier overflow");
    }
    plan.carrier_cells = next + plan.boundary.cells;
    const struct qtt_layout retain = qtt_make_layout(width, depth);
    plan.retain_all_carrier_cells = retain.carrier_cells;
    for (size_t node = 1U; node <= target; ++node) {
        if (active_generation[node] != 0U) {
            plan.active_projection_mask |=
                (size_t)1U << (node - 1U);
            plan.active_projection_cells +=
                qtt_stage_cells(width, node + 1U);
        }
    }
    if (
        (plan.active_projection_mask
            & ((size_t)1U << (target - 1U))) == 0U
    ) {
        fail("final quotient stage absent at projection");
    }
    plan.schedule_hash = qtp_plan_hash(&plan);
    return plan;
}

static double qtp_slot_error(
    const struct carrier *carrier,
    const struct qtp_slot *slot
) {
    double maximum = 0.0;
    for (size_t cell = 0U; cell < slot->capacity; ++cell) {
        const size_t index = slot->start + cell;
        const double error = cabs(
            carrier->working[index] - carrier->baseline[index]
        );
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static size_t qtp_find_node(
    const struct qtp_plan *plan,
    const struct qtp_runtime_slot runtime[QTP_MAX_PEBBLES],
    size_t node
) {
    for (size_t slot = 0U; slot < plan->pebble_count; ++slot) {
        if (runtime[slot].node == node) {
            return slot;
        }
    }
    return SIZE_MAX;
}

static struct qtt_stage qtp_bound_stage(
    const struct qtp_plan *plan,
    size_t slot,
    size_t node
) {
    if (
        slot >= plan->pebble_count
        || node == 0U
        || node >= plan->depth
    ) {
        fail("invalid quotient stage binding");
    }
    const size_t depth = node + 1U;
    const size_t cells = qtt_stage_cells(plan->width, depth);
    if (cells > plan->slot[slot].capacity) {
        fail("quotient stage exceeds compiled slot capacity");
    }
    return (struct qtt_stage){
        .start = plan->slot[slot].start,
        .depth = depth,
        .cells = cells
    };
}

static int qtp_apply_move(
    struct carrier *carrier,
    const struct qtp_plan *plan,
    const struct qtp_move *move,
    int add,
    int inverse_half,
    enum qtt_variant variant,
    struct qtp_runtime_slot runtime[QTP_MAX_PEBBLES],
    struct qtp_stats *stats,
    double *maximum_slot_clean_error
) {
    if (
        move->slot >= plan->pebble_count
        || move->node == 0U
        || move->node >= plan->depth
    ) {
        return 0;
    }
    size_t predecessor_slot = SIZE_MAX;
    uint64_t predecessor_generation = UINT64_C(1);
    struct qtt_stage predecessor = plan->leaf;
    if (move->node > 1U) {
        predecessor_slot = qtp_find_node(
            plan, runtime, move->node - 1U
        );
        if (predecessor_slot == SIZE_MAX) {
            return 0;
        }
        predecessor_generation =
            runtime[predecessor_slot].generation;
        predecessor = qtp_bound_stage(
            plan, predecessor_slot, move->node - 1U
        );
    }
    if (
        predecessor_generation != move->predecessor_generation
    ) {
        return 0;
    }
    if (add) {
        if (
            runtime[move->slot].node != 0U
            || qtp_find_node(plan, runtime, move->node) != SIZE_MAX
        ) {
            return 0;
        }
        const double clean = qtp_slot_error(
            carrier, &plan->slot[move->slot]
        );
        ++stats->slot_clean_scans;
        if (clean > *maximum_slot_clean_error) {
            *maximum_slot_clean_error = clean;
        }
        if (clean > RESTORATION_TOLERANCE) {
            return 0;
        }
    } else if (
        runtime[move->slot].node != move->node
        || runtime[move->slot].generation != move->generation
    ) {
        return 0;
    }

    const size_t output_depth = move->node + 1U;
    const struct qtt_stage output = qtp_bound_stage(
        plan, move->slot, move->node
    );
    struct qtt_layout view = {
        .width = plan->width,
        .maximum_depth = plan->depth,
        .carrier_cells = plan->carrier_cells
    };
    view.stage[1U] = plan->leaf;
    view.stage[output_depth - 1U] = predecessor;
    view.stage[output_depth] = output;
    qtt_compose_quotient(
        carrier,
        &view,
        output_depth,
        variant,
        add ? 0 : 1,
        &stats->phase
    );
    ++stats->weighted_move_cells;
    stats->weighted_move_cells += (uint64_t)(output.cells - 1U);
    if (add) {
        ++stats->stage_additions;
        if (inverse_half || move->generation > UINT64_C(1)) {
            ++stats->reconstruction_additions;
            stats->reconstruction_cells +=
                (uint64_t)output.cells;
        }
        if (runtime[move->slot].generation != 0U) {
            ++stats->slot_rebinds;
        }
        runtime[move->slot] = (struct qtp_runtime_slot){
            .node = move->node,
            .generation = move->generation
        };
    } else {
        ++stats->stage_removals;
        runtime[move->slot].node = 0U;
        runtime[move->slot].generation = move->generation;
        const double clean = qtp_slot_error(
            carrier, &plan->slot[move->slot]
        );
        ++stats->slot_clean_scans;
        if (clean > *maximum_slot_clean_error) {
            *maximum_slot_clean_error = clean;
        }
        if (clean > RESTORATION_TOLERANCE) {
            return 0;
        }
    }
    return 1;
}

static struct qtt_layout qtp_boundary_view(
    const struct qtp_plan *plan,
    size_t final_slot
) {
    struct qtt_layout view = {
        .width = plan->width,
        .maximum_depth = plan->depth,
        .boundary = plan->boundary,
        .carrier_cells = plan->carrier_cells
    };
    view.stage[plan->depth] = qtp_bound_stage(
        plan, final_slot, plan->depth - 1U
    );
    return view;
}

static size_t qtp_reordered_index(const struct qtp_plan *plan) {
    for (size_t reverse = 0U; reverse + 1U < plan->move_count; ++reverse) {
        const size_t first =
            plan->move[plan->move_count - 1U - reverse].node;
        const size_t second =
            plan->move[plan->move_count - 2U - reverse].node;
        if (first + 1U == second || second + 1U == first) {
            return reverse;
        }
    }
    return SIZE_MAX;
}

static struct qtp_execution qtp_execute(
    struct carrier *carrier,
    const struct qtp_plan *plan,
    enum qtt_variant variant,
    enum qtp_mode mode
) {
    if (carrier == NULL || plan == NULL) {
        fail("null pebbled quotient carrier");
    }
    if (carrier->cells != plan->carrier_cells) {
        fail("pebbled quotient carrier topology mismatch");
    }
    struct carrier borrowed = snapshot_carrier(carrier);
    struct qtp_execution execution = {0};
    if (plan->schedule_hash != qtp_plan_hash(plan)) {
        execution.custody_rejected = 1;
        execution.restoration_max_abs = restoration(
            carrier, &borrowed
        );
        execution.integrity_max_abs = integrity(carrier);
        free_carrier(&borrowed);
        return execution;
    }
    struct qtp_runtime_slot runtime[QTP_MAX_PEBBLES] = {{0}};
    struct qtt_layout leaf_view = {
        .width = plan->width,
        .maximum_depth = plan->depth,
        .carrier_cells = plan->carrier_cells
    };
    leaf_view.stage[1U] = plan->leaf;
    qtt_encode_leaf(
        carrier,
        &leaf_view,
        variant,
        0,
        &execution.stats.phase
    );
    int dirty_slot_injected = 0;
    int generation_tag_tampered = 0;
    for (size_t index = 0U; index < plan->move_count; ++index) {
        const struct qtp_move *move = &plan->move[index];
        if (
            mode == QTP_DIRTY_SLOT_INJECTION
            && !dirty_slot_injected
            && move->add
        ) {
            multiply_cell(
                carrier,
                plan->slot[move->slot].start,
                root3(1)
            );
            dirty_slot_injected = 1;
        }
        if (
            mode == QTP_GENERATION_TAG_TAMPER
            && !generation_tag_tampered
            && move->node > 1U
        ) {
            const size_t predecessor_slot = qtp_find_node(
                plan, runtime, move->node - 1U
            );
            if (predecessor_slot == SIZE_MAX) {
                fail("generation control lacks live predecessor");
            }
            ++runtime[predecessor_slot].generation;
            generation_tag_tampered = 1;
        }
        if (
            !qtp_apply_move(
                carrier,
                plan,
                move,
                move->add,
                0,
                variant,
                runtime,
                &execution.stats,
                &execution.maximum_slot_clean_error
            )
        ) {
            execution.custody_rejected = 1;
            break;
        }
        ++execution.stats.forward_schedule_moves;
    }
    const size_t final_slot = qtp_find_node(
        plan, runtime, plan->depth - 1U
    );
    if (
        execution.custody_rejected
        || final_slot == SIZE_MAX
    ) {
        execution.custody_rejected = 1;
        execution.restoration_max_abs = restoration(
            carrier, &borrowed
        );
        execution.integrity_max_abs = integrity(carrier);
        free_carrier(&borrowed);
        return execution;
    }
    const struct qtt_layout boundary_view = qtp_boundary_view(
        plan, final_slot
    );
    qtt_copy_final(
        carrier,
        &boundary_view,
        0,
        0,
        &execution.stats.phase
    );
    execution.projection = qtt_latch_boundary(
        carrier,
        &boundary_view,
        &execution.stats.phase
    );

    if (mode == QTP_SNAPSHOT_RELOAD) {
        memcpy(
            carrier->working,
            borrowed.working,
            carrier->cells * sizeof(*carrier->working)
        );
        execution.snapshot_loaded = 1;
    } else {
        qtt_copy_final(
            carrier,
            &boundary_view,
            1,
            mode == QTP_WRONG_BOUNDARY_INVERSE,
            &execution.stats.phase
        );
        const size_t reordered = qtp_reordered_index(plan);
        execution.reordered_applicable = reordered != SIZE_MAX;
        for (
            size_t reverse = 0U;
            reverse < plan->move_count;
            ++reverse
        ) {
            if (
                mode == QTP_MISSING_INVERSE_MOVE
                && reverse == 0U
            ) {
                continue;
            }
            size_t selected = reverse;
            if (
                mode == QTP_REORDERED_NONCOMMUTING_INVERSE
                && reordered != SIZE_MAX
            ) {
                if (reverse == reordered) {
                    selected = reverse + 1U;
                } else if (reverse == reordered + 1U) {
                    selected = reverse - 1U;
                }
            }
            const struct qtp_move *move =
                &plan->move[plan->move_count - 1U - selected];
            if (
                !qtp_apply_move(
                    carrier,
                    plan,
                    move,
                    !move->add,
                    1,
                    variant,
                    runtime,
                    &execution.stats,
                    &execution.maximum_slot_clean_error
                )
            ) {
                execution.custody_rejected = 1;
                break;
            }
            ++execution.stats.inverse_schedule_moves;
        }
        if (!execution.custody_rejected) {
            for (
                size_t slot = 0U;
                slot < plan->pebble_count;
                ++slot
            ) {
                if (runtime[slot].node != 0U) {
                    execution.custody_rejected = 1;
                }
            }
        }
        if (!execution.custody_rejected) {
            qtt_encode_leaf(
                carrier,
                &leaf_view,
                variant,
                1,
                &execution.stats.phase
            );
            execution.actual_inverse = 1;
            execution.restoration_generation = UINT64_C(1);
        }
    }
    execution.restoration_max_abs = restoration(carrier, &borrowed);
    execution.integrity_max_abs = integrity(carrier);
    free_carrier(&borrowed);
    return execution;
}

static void qtp_require_correct(
    const struct qtp_execution *execution,
    const struct qtp_plan *plan
) {
    if (
        execution->custody_rejected
        || !execution->actual_inverse
        || execution->snapshot_loaded
        || execution->restoration_generation != UINT64_C(1)
        || execution->projection.cells != plan->boundary.cells
        || execution->projection.maximum_root_error > ROOT_TOLERANCE
        || execution->restoration_max_abs > RESTORATION_TOLERANCE
        || execution->integrity_max_abs > RESTORATION_TOLERANCE
        || execution->maximum_slot_clean_error
            > RESTORATION_TOLERANCE
        || execution->stats.forward_schedule_moves
            != plan->move_count
        || execution->stats.inverse_schedule_moves
            != plan->move_count
        || execution->stats.phase.final_decodes
            != execution->projection.cells
    ) {
        fail("pebbled quotient execution violated custody law");
    }
}

static struct qtp_execution qtp_control(
    const struct qtp_plan *plan,
    enum qtp_mode mode
) {
    const struct process process = {
        .carrier_cells = plan->carrier_cells
    };
    struct carrier carrier = make_carrier(
        &process, 9300 + (int)mode
    );
    struct qtp_execution execution = qtp_execute(
        &carrier, plan, QTT_NEIGHBOR_AND, mode
    );
    free_carrier(&carrier);
    return execution;
}

int main(int argc, char **argv) {
    if (argc == 4 && strcmp(argv[1], "--project-intermediate") == 0) {
        (void)qtp_parse_canonical(
            argv[2],
            QTP_MIN_WIDTH,
            QTP_MAX_WIDTH,
            "width must be canonical decimal from 4 through 16"
        );
        (void)qtp_parse_canonical(
            argv[3],
            QTP_MIN_DEPTH,
            QTP_MAX_DEPTH,
            "depth must be canonical decimal from 2 through 8"
        );
        fail("only the final pebbled quotient boundary may be projected");
    }
    if (argc != 3) {
        fail(
            "usage: algebraic_boolean_tt_pebbled_quotient_phase "
            "<width 4..16> <depth 2..8>"
        );
    }
    const size_t width = qtp_parse_canonical(
        argv[1],
        QTP_MIN_WIDTH,
        QTP_MAX_WIDTH,
        "width must be canonical decimal from 4 through 16"
    );
    const size_t depth = qtp_parse_canonical(
        argv[2],
        QTP_MIN_DEPTH,
        QTP_MAX_DEPTH,
        "depth must be canonical decimal from 2 through 8"
    );
    if (depth > width) {
        fail("depth may not exceed width in pebbled quotient experiment");
    }
    const struct qtp_plan plan = qtp_compile_plan(width, depth);
    const struct process process = {
        .carrier_cells = plan.carrier_cells
    };
    struct carrier carrier = make_carrier(&process, 9201);
    struct qtp_execution primary = qtp_execute(
        &carrier, &plan, QTT_NEIGHBOR_AND, QTP_CORRECT
    );
    qtp_require_correct(&primary, &plan);
    struct qtp_execution reuse = qtp_execute(
        &carrier, &plan, QTT_NEIGHBOR_OR, QTP_CORRECT
    );
    qtp_require_correct(&reuse, &plan);
    if (primary.projection.hash == reuse.projection.hash) {
        fail("pebbled quotient unrelated reuse did not change boundary");
    }
    double repeated_max = 0.0;
    for (size_t cycle = 0U; cycle < QTP_REUSE_CYCLES; ++cycle) {
        struct qtp_execution repeated = qtp_execute(
            &carrier,
            &plan,
            cycle % 2U == 0U
                ? QTT_NEIGHBOR_AND
                : QTT_NEIGHBOR_OR,
            QTP_CORRECT
        );
        qtp_require_correct(&repeated, &plan);
        if (repeated.restoration_max_abs > repeated_max) {
            repeated_max = repeated.restoration_max_abs;
        }
        qtt_free_projection(&repeated.projection);
    }
    free_carrier(&carrier);

    struct qtp_execution missing = qtp_control(
        &plan, QTP_MISSING_INVERSE_MOVE
    );
    struct qtp_execution reordered = qtp_control(
        &plan, QTP_REORDERED_NONCOMMUTING_INVERSE
    );
    struct qtp_execution snapshot = qtp_control(
        &plan, QTP_SNAPSHOT_RELOAD
    );
    struct qtp_execution wrong = qtp_control(
        &plan, QTP_WRONG_BOUNDARY_INVERSE
    );
    struct qtp_execution dirty = qtp_control(
        &plan, QTP_DIRTY_SLOT_INJECTION
    );
    struct qtp_execution generation = qtp_control(
        &plan, QTP_GENERATION_TAG_TAMPER
    );
    struct qtp_plan tampered_plan = plan;
    tampered_plan.schedule_hash ^= UINT64_C(1);
    struct qtp_execution tampered = qtp_control(
        &tampered_plan, QTP_CORRECT
    );
    if (
        !missing.custody_rejected
        || missing.restoration_max_abs < CONTROL_MINIMUM
        || !reordered.reordered_applicable
        || !reordered.custody_rejected
        || reordered.restoration_max_abs < CONTROL_MINIMUM
        || !snapshot.snapshot_loaded
        || snapshot.actual_inverse
        || snapshot.restoration_max_abs > RESTORATION_TOLERANCE
        || wrong.custody_rejected
        || !wrong.actual_inverse
        || wrong.restoration_max_abs < CONTROL_MINIMUM
        || !dirty.custody_rejected
        || dirty.restoration_max_abs < CONTROL_MINIMUM
        || !generation.custody_rejected
        || generation.restoration_max_abs < CONTROL_MINIMUM
        || !tampered.custody_rejected
        || tampered.projection.cells != 0U
        || tampered.restoration_max_abs > RESTORATION_TOLERANCE
    ) {
        fail("pebbled quotient control did not separate");
    }

    printf(
        "{\"mode\":\"boolean-tt-pebbled-suffix-quotient\","
        "\"claim\":\"TOPOLOGY_DERIVED_REVERSIBLE_BOOLEAN_TT_QUOTIENT_STAGE_PEBBLING_REDUCES_RETAINED_PHASE_HISTORY\","
        "\"width\":%zu,"
        "\"depth\":%zu,"
        "\"pebble_slots\":%zu,"
        "\"forward_schedule_moves\":%zu,"
        "\"schedule_fnv1a64\":\"%016llx\","
        "\"active_projection_mask\":%zu,"
        "\"active_projection_cells\":%zu,"
        "\"leaf_cells\":%zu,"
        "\"boundary_cells\":%zu,"
        "\"retain_all_carrier_cells\":%zu,"
        "\"pebbled_carrier_cells\":%zu,"
        "\"retain_all_carrier_bytes\":%zu,"
        "\"pebbled_carrier_bytes\":%zu,"
        "\"comparison_snapshot_bytes\":%zu,"
        "\"compiled_plan_bytes\":%zu,"
        "\"projection_bytes\":%zu,"
        "\"carrier_cell_reduction\":%zu,"
        "\"carrier_reduction_ratio\":%.12g,"
        "\"applicability_gate_reduces_carrier\":%s,"
        "\"slot_capacities\":[",
        width,
        depth,
        plan.pebble_count,
        plan.move_count,
        (unsigned long long)plan.schedule_hash,
        plan.active_projection_mask,
        plan.active_projection_cells,
        plan.leaf.cells,
        plan.boundary.cells,
        plan.retain_all_carrier_cells,
        plan.carrier_cells,
        2U * plan.retain_all_carrier_cells
            * sizeof(double complex),
        2U * plan.carrier_cells * sizeof(double complex),
        2U * plan.carrier_cells * sizeof(double complex),
        sizeof(plan),
        plan.boundary.cells,
        plan.retain_all_carrier_cells - plan.carrier_cells,
        (double)plan.retain_all_carrier_cells
            / (double)plan.carrier_cells,
        plan.carrier_cells < plan.retain_all_carrier_cells
            ? "true"
            : "false"
    );
    for (size_t slot = 0U; slot < plan.pebble_count; ++slot) {
        printf(
            "%s%zu",
            slot == 0U ? "" : ",",
            plan.slot[slot].capacity
        );
    }
    printf(
        "],"
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"primary_boundary_ones\":%zu,"
        "\"reuse_boundary_ones\":%zu,"
        "\"logical_phase_ands_per_transaction\":%llu,"
        "\"logical_phase_ors_per_transaction\":%llu,"
        "\"carrier_reads_per_transaction\":%llu,"
        "\"phase_cell_updates_per_transaction\":%llu,"
        "\"quotient_member_terms_per_transaction\":%llu,"
        "\"weighted_pebble_move_cells_per_transaction\":%llu,"
        "\"stage_additions_per_transaction\":%llu,"
        "\"stage_removals_per_transaction\":%llu,"
        "\"reconstruction_additions_per_transaction\":%llu,"
        "\"reconstruction_cells_per_transaction\":%llu,"
        "\"slot_clean_scans_per_transaction\":%llu,"
        "\"maximum_slot_clean_error\":%.12g,"
        "\"final_decodes_per_transaction\":%llu,"
        "\"decoded_intermediate_cells\":0,"
        "\"serialized_intermediate_cells\":0,"
        "\"materialized_raw_product_cells\":0,"
        "\"schedule_derived_from_public_depth\":true,"
        "\"exact_slot_generation_custody\":true,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"same_carrier_transactions\":%u,"
        "\"maximum_repeated_restoration_error\":%.12g,"
        "\"wrong_inverse_detected\":true,"
        "\"missing_inverse_detected\":true,"
        "\"reordered_noncommuting_inverse_detected\":true,"
        "\"dirty_slot_injection_detected\":true,"
        "\"generation_tag_tamper_detected\":true,"
        "\"schedule_hash_tamper_rejected\":true,"
        "\"snapshot_loaded_separately\":true,"
        "\"compact_public_recurrence_still_exists\":true,"
        "\"fixed_rank_unbounded_depth_closure\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_execution\":false,"
        "\"unlimited_catalytic_computation\":false,"
        "\"terminal\":false}\n",
        (unsigned long long)primary.projection.hash,
        (unsigned long long)reuse.projection.hash,
        primary.projection.ones,
        reuse.projection.ones,
        (unsigned long long)
            primary.stats.phase.logical_phase_ands,
        (unsigned long long)
            primary.stats.phase.logical_phase_ors,
        (unsigned long long)primary.stats.phase.carrier_reads,
        (unsigned long long)
            primary.stats.phase.phase_cell_updates,
        (unsigned long long)
            primary.stats.phase.quotient_member_terms,
        (unsigned long long)primary.stats.weighted_move_cells,
        (unsigned long long)primary.stats.stage_additions,
        (unsigned long long)primary.stats.stage_removals,
        (unsigned long long)
            primary.stats.reconstruction_additions,
        (unsigned long long)
            primary.stats.reconstruction_cells,
        (unsigned long long)primary.stats.slot_clean_scans,
        primary.maximum_slot_clean_error,
        (unsigned long long)primary.stats.phase.final_decodes,
        (unsigned int)(QTP_REUSE_CYCLES + 2U),
        repeated_max
    );

    qtt_free_projection(&primary.projection);
    qtt_free_projection(&reuse.projection);
    qtt_free_projection(&missing.projection);
    qtt_free_projection(&reordered.projection);
    qtt_free_projection(&snapshot.projection);
    qtt_free_projection(&wrong.projection);
    qtt_free_projection(&dirty.projection);
    qtt_free_projection(&generation.projection);
    qtt_free_projection(&tampered.projection);
    return 0;
}
