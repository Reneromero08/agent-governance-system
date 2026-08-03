#define _POSIX_C_SOURCE 200809L

/* Warm direct-process comparison for compact, phase, and snapshot paths. */

#define NPG_EMBEDDED 1
#define NPG_NO_REQUIRE 1
#include "topology_nonlinear_phase_graph.c"

#include <time.h>

static uint64_t npb_nanoseconds(const struct timespec *time) {
    return (uint64_t)time->tv_sec * UINT64_C(1000000000)
        + (uint64_t)time->tv_nsec;
}

static uint64_t npb_compact(
    const struct npg_graph *graph,
    size_t rounds,
    enum npg_program program
) {
    double angle[NPG_MAX_WIDTH] = {0};
    for (size_t cell = 0U; cell < graph->width; ++cell) {
        angle[cell] = npg_initial_angle(program, cell);
    }
    for (size_t round = 0U; round < rounds; ++round) {
        const double scale =
            1.0 + 0.03125 * (double)(round % 5U);
        for (size_t edge = 0U; edge < graph->edge_count; ++edge) {
            const struct npg_edge *operation = &graph->edge[edge];
            angle[operation->target] +=
                (double)operation->strength_milli
                * 0.001
                * scale
                * sin(angle[operation->source]);
        }
    }
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t output = 0U; output < 2U; ++output) {
        const size_t cell = graph->output[output];
        hash = npg_hash_word(
            hash,
            (uint64_t)(int64_t)llround(
                cos(angle[cell]) * NPG_HASH_SCALE
            )
        );
        hash = npg_hash_word(
            hash,
            (uint64_t)(int64_t)llround(
                sin(angle[cell]) * NPG_HASH_SCALE
            )
        );
    }
    return hash;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        return 2;
    }
    const struct npg_graph graph = npg_load_graph(argv[1]);
    const size_t rounds = npg_parse_size(
        argv[2], 1U, 4096U, "benchmark rounds invalid"
    );
    const size_t cycles = npg_parse_size(
        argv[3], 1U, 100000U, "benchmark cycles invalid"
    );
    const int compact = strcmp(argv[4], "compact") == 0;
    const int snapshot = strcmp(argv[4], "snapshot") == 0;
    if (!compact && !snapshot && strcmp(argv[4], "phase") != 0) {
        return 2;
    }
    const struct process process = {
        .carrier_cells = graph.width + 2U
    };
    struct carrier carrier = {0};
    if (!compact) {
        carrier = make_carrier(&process, 12501);
    }
    const size_t warmup = 32U;
    volatile uint64_t checksum = 0U;
    struct timespec begin = {0};
    struct timespec end = {0};
    for (size_t cycle = 0U; cycle < warmup + cycles; ++cycle) {
        if (
            cycle == warmup
            && clock_gettime(CLOCK_MONOTONIC, &begin) != 0
        ) {
            return 2;
        }
        const enum npg_program program =
            cycle % 2U == 0U ? NPG_PRIMARY : NPG_REUSE;
        if (compact) {
            checksum ^= npb_compact(&graph, rounds, program);
        } else {
            const struct npg_execution execution = npg_execute(
                &carrier,
                &graph,
                rounds,
                program,
                snapshot ? NPG_SNAPSHOT : NPG_CORRECT
            );
            if (
                execution.restoration_max_abs > NPG_PHASE_TOLERANCE
                || (
                    snapshot
                    && (
                        !execution.snapshot_loaded
                        || execution.actual_inverse
                    )
                )
                || (
                    !snapshot
                    && (
                        execution.snapshot_loaded
                        || !execution.actual_inverse
                    )
                )
            ) {
                return 2;
            }
            checksum ^= execution.projection.hash;
        }
    }
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        return 2;
    }
    if (!compact) {
        free_carrier(&carrier);
    }
    const uint64_t elapsed =
        npb_nanoseconds(&end) - npb_nanoseconds(&begin);
    printf(
        "{\"result\":\"PASS\",\"arm\":\"%s\","
        "\"warmup_transactions\":%zu,"
        "\"timed_transactions\":%zu,"
        "\"elapsed_ns\":%llu,\"ns_per_transaction\":%.17g,"
        "\"checksum\":\"%016llx\"}\n",
        argv[4],
        warmup,
        cycles,
        (unsigned long long)elapsed,
        (double)elapsed / (double)cycles,
        (unsigned long long)checksum
    );
    return 0;
}
