#define _POSIX_C_SOURCE 200809L

/* Warm direct-process comparison for compact, in-place, and snapshot paths. */

#define NW_EMBEDDED 1
#include "algebraic_nonlinear_kerr_wave.c"
#define KR_EMBEDDED 1
#include "algebraic_nonlinear_kerr_wave_reference.c"

#include <errno.h>
#include <time.h>

static uint64_t kwb_nanoseconds(const struct timespec *time) {
    return (uint64_t)time->tv_sec * UINT64_C(1000000000)
        + (uint64_t)time->tv_nsec;
}

static int kwb_parse_size(
    const char *text,
    size_t minimum,
    size_t maximum,
    size_t *value
) {
    errno = 0;
    char *tail = NULL;
    const unsigned long long parsed = strtoull(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed < (unsigned long long)minimum
        || parsed > (unsigned long long)maximum
    ) {
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        return 2;
    }
    size_t depth = 0U;
    size_t cycles = 0U;
    if (
        !kwb_parse_size(argv[1], 1U, 2048U, &depth)
        || !kwb_parse_size(argv[2], 1U, 100000U, &cycles)
    ) {
        return 2;
    }
    const int compact = strcmp(argv[3], "compact") == 0;
    const int snapshot = strcmp(argv[3], "snapshot") == 0;
    const int compare = strcmp(argv[3], "compare") == 0;
    if (
        !compact
        && !snapshot
        && !compare
        && strcmp(argv[3], "phase") != 0
    ) {
        return 2;
    }
    struct nw_carrier carrier = nw_make_carrier(UINT64_C(53));
    struct nw_stats stats = {0};
    if (compare) {
        double primary_restoration = 0.0;
        double reuse_restoration = 0.0;
        const struct nw_boundary primary = nw_transaction(
            &carrier,
            depth,
            3U,
            NW_RESTORE_CORRECT,
            0,
            0,
            &stats,
            &primary_restoration
        );
        const struct nw_boundary reuse = nw_transaction(
            &carrier,
            depth,
            41U,
            NW_RESTORE_CORRECT,
            0,
            0,
            &stats,
            &reuse_restoration
        );
        const uint64_t compact_primary = kr_run(
            depth, UINT64_C(53), 3U
        );
        const uint64_t compact_reuse = kr_run(
            depth, UINT64_C(53), 41U
        );
        if (
            primary.hash != compact_primary
            || reuse.hash != compact_reuse
            || primary_restoration > NW_RESTORATION_TOLERANCE
            || reuse_restoration > NW_RESTORATION_TOLERANCE
        ) {
            return 2;
        }
        printf(
            "{\"result\":\"PASS\","
            "\"primary_boundary_fnv1a64\":\"%016llx\","
            "\"reuse_boundary_fnv1a64\":\"%016llx\","
            "\"compact_primary_boundary_fnv1a64\":\"%016llx\","
            "\"compact_reuse_boundary_fnv1a64\":\"%016llx\","
            "\"primary_restoration_error\":%.17g,"
            "\"reuse_restoration_error\":%.17g,"
            "\"carrier_complex_cells\":4,"
            "\"compact_state_bytes\":64}\n",
            (unsigned long long)primary.hash,
            (unsigned long long)reuse.hash,
            (unsigned long long)compact_primary,
            (unsigned long long)compact_reuse,
            primary_restoration,
            reuse_restoration
        );
        return 0;
    }
    const size_t warmup = 32U;
    volatile uint64_t checksum = UINT64_C(1469598103934665603);
    double maximum_restoration_error = 0.0;
    struct timespec begin = {0};
    struct timespec end = {0};
    for (size_t cycle = 0U; cycle < warmup + cycles; ++cycle) {
        if (
            cycle == warmup
            && clock_gettime(CLOCK_MONOTONIC, &begin) != 0
        ) {
            return 2;
        }
        const size_t program = cycle % 2U == 0U ? 3U : 41U;
        if (compact) {
            checksum ^= kr_run(depth, UINT64_C(53), program);
            checksum *= UINT64_C(1099511628211);
        } else {
            double restoration_error = 0.0;
            const struct nw_boundary boundary = nw_transaction(
                &carrier,
                depth,
                program,
                snapshot
                    ? NW_RESTORE_SNAPSHOT
                    : NW_RESTORE_CORRECT,
                0,
                0,
                &stats,
                &restoration_error
            );
            if (restoration_error > NW_RESTORATION_TOLERANCE) {
                return 2;
            }
            maximum_restoration_error = fmax(
                maximum_restoration_error,
                restoration_error
            );
            checksum ^= boundary.hash;
            checksum *= UINT64_C(1099511628211);
        }
    }
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        return 2;
    }
    const uint64_t elapsed =
        kwb_nanoseconds(&end) - kwb_nanoseconds(&begin);
    printf(
        "{\"result\":\"PASS\",\"arm\":\"%s\","
        "\"warmup_transactions\":%zu,"
        "\"timed_transactions\":%zu,"
        "\"elapsed_ns\":%llu,\"ns_per_transaction\":%.17g,"
        "\"checksum\":\"%016llx\","
        "\"maximum_restoration_error\":%.17g,"
        "\"semantic_state_bytes\":64,"
        "\"compiled_inverse_history_bytes\":0}\n",
        argv[3],
        warmup,
        cycles,
        (unsigned long long)elapsed,
        (double)elapsed / (double)cycles,
        (unsigned long long)checksum,
        maximum_restoration_error
    );
    return 0;
}
