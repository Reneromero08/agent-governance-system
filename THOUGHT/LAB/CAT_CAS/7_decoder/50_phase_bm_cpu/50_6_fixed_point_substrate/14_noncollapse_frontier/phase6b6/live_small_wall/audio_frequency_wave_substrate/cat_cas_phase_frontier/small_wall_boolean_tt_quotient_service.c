#define _GNU_SOURCE

/*
 * Matched persistent-service arms for the growing Boolean-TT quotient
 * Small-Wall diagnostic.
 *
 * QTW_ARM=1: direct public threshold-quotient generator
 * QTW_ARM=2: reviewed phase forward path plus working-state snapshot reload
 * QTW_ARM=3: reviewed phase forward path plus actual inverse restoration
 *
 * All arms accept the same public width/depth/family requests and return only
 * the same final-boundary receipt.  The direct arm streams every quotient
 * cell on every transaction.  It has no cache, answer table, phase carrier,
 * or raw-product representation.
 */

#define QTW_BASELINE 1
#define QTW_SNAPSHOT 2
#define QTW_IN_PLACE 3

#ifndef QTW_ARM
#error "compile with QTW_ARM=1, 2, or 3"
#endif

#if QTW_ARM < QTW_BASELINE || QTW_ARM > QTW_IN_PLACE
#error "invalid QTW_ARM"
#endif

#if QTW_ARM != QTW_BASELINE
#define QTT_EMBEDDED_MAIN qtw_reviewed_standalone_main
#include "algebraic_boolean_tt_suffix_quotient_phase.c"
#undef QTT_EMBEDDED_MAIN
#endif

#include <errno.h>
#include <fcntl.h>
#include <linux/prctl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <seccomp.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define QTW_PROTOCOL "CATVM_BOOLEAN_TT_QUOTIENT_SMALL_WALL_1"
#define QTW_FAMILIES 2U
#define QTW_REQUEST_CAPACITY 128U
#define QTW_RESPONSE_CAPACITY 1024U
#define QTW_MIN_WIDTH 4U
#define QTW_MAX_WIDTH 16U
#define QTW_MIN_DEPTH 2U
#define QTW_MAX_DEPTH 8U
#define QTW_CARRIER_ID 9101

enum qtw_state {
    QTW_READY = 1,
    QTW_RUNNING = 2,
    QTW_FAILED = 3
};

struct qtw_receipt {
    size_t cells;
    size_t ones;
    uint64_t hash;
};

struct qtw_counters {
    uint64_t transactions;
    uint64_t direct_cells;
    uint64_t direct_predicates;
    uint64_t logical_phase_ands;
    uint64_t logical_phase_ors;
    uint64_t carrier_reads;
    uint64_t phase_cell_updates;
    uint64_t quotient_member_terms;
    uint64_t final_decodes;
    uint64_t projected_bytes;
    uint64_t comparison_snapshot_bytes;
    uint64_t restoration_scan_cells;
    uint64_t snapshot_loads;
    uint64_t snapshot_reload_bytes;
    uint64_t actual_inverse_transactions;
    uint64_t cpu_start_ns;
    int timing_active;
};

struct qtw_context {
    size_t width;
    size_t depth;
    size_t final_cells;
    size_t retained_stage_cells;
    size_t carrier_cells;
#if QTW_ARM != QTW_BASELINE
    struct qtt_layout layout;
    struct carrier carrier;
    struct carrier sealed_carrier;
#endif
    struct qtw_counters counters;
    enum qtw_state state;
    uint64_t rank_plan_hash;
    uint64_t carrier_creation_count;
    uint64_t restoration_generation;
    uint64_t total_transactions;
    uint64_t seal_cpu_ns;
    double maximum_restoration_error;
};

static void qtw_secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

#if QTW_ARM == QTW_BASELINE
struct qtw_interval {
    size_t low;
    size_t high;
};

static uint64_t qtw_hash_byte(uint64_t hash, unsigned char byte) {
    hash ^= (uint64_t)byte;
    hash *= UINT64_C(1099511628211);
    return hash;
}
#endif

static int qtw_cpu_time_ns(uint64_t *time_ns) {
    struct timespec measured;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &measured) != 0) {
        return 0;
    }
    if (measured.tv_sec < 0 || measured.tv_nsec < 0) {
        return 0;
    }
    const uint64_t seconds = (uint64_t)measured.tv_sec;
    if (seconds > UINT64_MAX / UINT64_C(1000000000)) {
        return 0;
    }
    *time_ns = seconds * UINT64_C(1000000000)
        + (uint64_t)measured.tv_nsec;
    return 1;
}

static int qtw_parse_size(
    const char *text,
    size_t minimum,
    size_t maximum,
    size_t *value
) {
    if (text == NULL || text[0] < '1' || text[0] > '9') {
        return 0;
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            return 0;
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
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}

static size_t qtw_bond_rank(
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

static size_t qtw_stage_cells(size_t width, size_t depth) {
    size_t cells = 0U;
    for (size_t site = 0U; site < width; ++site) {
        const size_t left = qtw_bond_rank(width, depth, site);
        const size_t right = qtw_bond_rank(
            width, depth, site + 1U
        );
        if (
            left > SIZE_MAX / 4U
            || right > SIZE_MAX / (4U * left)
        ) {
            return 0U;
        }
        const size_t core = 4U * left * right;
        if (cells > SIZE_MAX - core) {
            return 0U;
        }
        cells += core;
    }
    return cells;
}

static int qtw_shape(
    size_t width,
    size_t depth,
    size_t *final_cells,
    size_t *retained_stage_cells,
    size_t *carrier_cells
) {
    size_t retained = 0U;
    for (size_t stage = 1U; stage <= depth; ++stage) {
        const size_t cells = qtw_stage_cells(width, stage);
        if (cells == 0U || retained > SIZE_MAX - cells) {
            return 0;
        }
        retained += cells;
        if (stage == depth) {
            *final_cells = cells;
        }
    }
    if (retained > SIZE_MAX - *final_cells) {
        return 0;
    }
    *retained_stage_cells = retained;
    *carrier_cells = retained + *final_cells;
    return 1;
}

#if QTW_ARM == QTW_BASELINE
static struct qtw_interval qtw_class_interval(
    size_t width,
    size_t depth,
    size_t bond,
    size_t state
) {
    struct qtw_interval interval = {0U, 0U};
    if (bond == 0U || bond == width) {
        return interval;
    }
    const size_t suffix = width - bond;
    if (suffix == 1U) {
        interval.low = state == 0U ? 0U : 1U;
        interval.high = state == 0U ? 0U : depth;
        return interval;
    }
    if (depth <= suffix || state < suffix) {
        interval.low = state;
        interval.high = state;
        return interval;
    }
    if (state == suffix) {
        interval.low = suffix;
        interval.high = depth - 1U;
        return interval;
    }
    interval.low = depth;
    interval.high = depth;
    return interval;
}

static int qtw_contains(
    const struct qtw_interval *interval,
    size_t value
) {
    return interval->low <= value && value <= interval->high;
}

static int qtw_overlap_shifted_middle(
    const struct qtw_interval *left,
    const struct qtw_interval *right,
    size_t depth
) {
    size_t low = left->low > 1U ? left->low : 1U;
    const size_t shifted_low = right->low + 1U;
    if (low < shifted_low) {
        low = shifted_low;
    }
    size_t high = left->high < depth - 1U
        ? left->high
        : depth - 1U;
    const size_t shifted_high = right->high + 1U;
    if (high > shifted_high) {
        high = shifted_high;
    }
    return low <= high;
}

static unsigned char qtw_direct_cell(
    size_t width,
    size_t depth,
    size_t family,
    size_t site,
    size_t left_state,
    size_t x,
    size_t z,
    size_t right_state
) {
    const struct qtw_interval left = qtw_class_interval(
        width, depth, site, left_state
    );
    const struct qtw_interval right = qtw_class_interval(
        width, depth, site + 1U, right_state
    );
    int accepted = 0;
    if (site == 0U) {
        if (family == 0U) {
            accepted =
                (x == 0U && z == 0U)
                || (
                    x == 1U
                    && (
                        (z == 1U && qtw_contains(&right, depth))
                        || (
                            z == 0U
                            && right.low <= depth - 1U
                        )
                    )
                );
        } else {
            accepted =
                (x == 1U && z == 1U)
                || (
                    x == 0U
                    && (
                        (z == 0U && qtw_contains(&right, depth))
                        || (
                            z == 1U
                            && right.low <= depth - 1U
                        )
                    )
                );
        }
    } else if (site + 1U == width) {
        if (family == 0U) {
            accepted =
                (x == 0U && qtw_contains(&left, 0U))
                || (x == 1U && left.high >= 1U);
        } else {
            accepted =
                (x == 1U && qtw_contains(&left, 0U))
                || (x == 0U && left.high >= 1U);
        }
    } else if (family == 0U) {
        accepted =
            (
                x == 0U
                && z == 0U
                && qtw_contains(&left, 0U)
            )
            || (
                x == 1U
                && z == 0U
                && qtw_overlap_shifted_middle(
                    &left, &right, depth
                )
            )
            || (
                x == 1U
                && qtw_contains(&left, depth)
                && (
                    (z == 0U && qtw_contains(&right, depth - 1U))
                    || (z == 1U && qtw_contains(&right, depth))
                )
            );
    } else {
        accepted =
            (
                x == 1U
                && z == 1U
                && qtw_contains(&left, 0U)
            )
            || (
                x == 0U
                && z == 1U
                && qtw_overlap_shifted_middle(
                    &left, &right, depth
                )
            )
            || (
                x == 0U
                && qtw_contains(&left, depth)
                && (
                    (z == 1U && qtw_contains(&right, depth - 1U))
                    || (z == 0U && qtw_contains(&right, depth))
                )
            );
    }
    return (unsigned char)(accepted != 0);
}

static struct qtw_receipt qtw_direct_receipt(
    const struct qtw_context *context,
    size_t family,
    struct qtw_counters *counters
) {
    struct qtw_receipt receipt = {
        .hash = UINT64_C(14695981039346656037)
    };
    for (size_t site = 0U; site < context->width; ++site) {
        const size_t left_rank = qtw_bond_rank(
            context->width, context->depth, site
        );
        const size_t right_rank = qtw_bond_rank(
            context->width, context->depth, site + 1U
        );
        for (size_t left = 0U; left < left_rank; ++left) {
            for (size_t x = 0U; x < 2U; ++x) {
                for (size_t z = 0U; z < 2U; ++z) {
                    for (
                        size_t right = 0U;
                        right < right_rank;
                        ++right
                    ) {
                        const unsigned char bit = qtw_direct_cell(
                            context->width,
                            context->depth,
                            family,
                            site,
                            left,
                            x,
                            z,
                            right
                        );
                        receipt.hash = qtw_hash_byte(
                            receipt.hash, bit
                        );
                        receipt.ones += (size_t)bit;
                        ++receipt.cells;
                        ++counters->direct_cells;
                        ++counters->direct_predicates;
                    }
                }
            }
        }
    }
    counters->projected_bytes += (uint64_t)receipt.cells;
    return receipt;
}
#endif

static size_t qtw_page_round(size_t bytes, size_t page_size) {
    if (
        bytes == 0U
        || page_size == 0U
        || bytes > SIZE_MAX - (page_size - 1U)
    ) {
        return 0U;
    }
    return ((bytes + page_size - 1U) / page_size) * page_size;
}

static int qtw_process_is_untraced(void) {
#ifdef QTW_TRACE_BUILD
    return 1;
#else
    FILE *status = fopen("/proc/self/status", "rb");
    if (status == NULL) {
        return 0;
    }
    char line[256];
    int untraced = 0;
    while (fgets(line, sizeof(line), status) != NULL) {
        unsigned int tracer = 1U;
        if (
            sscanf(line, "TracerPid:%u", &tracer) == 1
            && tracer == 0U
        ) {
            untraced = 1;
            break;
        }
    }
    return fclose(status) == 0 && untraced;
#endif
}

static int qtw_establish_process_guards(void) {
    return
        prctl(PR_SET_DUMPABLE, 0L, 0L, 0L, 0L) == 0
        && prctl(PR_SET_PTRACER, -1L, 0L, 0L, 0L) == 0;
}

static struct qtw_context *qtw_create(
    size_t width,
    size_t depth,
    size_t *mapped_bytes
) {
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        return NULL;
    }
    *mapped_bytes = qtw_page_round(
        sizeof(struct qtw_context),
        (size_t)page_size_long
    );
    if (*mapped_bytes == 0U) {
        return NULL;
    }
    struct qtw_context *context = mmap(
        NULL,
        *mapped_bytes,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );
    if (context == MAP_FAILED) {
        return NULL;
    }
    if (
        mlock(context, *mapped_bytes) != 0
        || madvise(context, *mapped_bytes, MADV_DONTDUMP) != 0
        || madvise(context, *mapped_bytes, MADV_DONTFORK) != 0
    ) {
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
    uint64_t seal_start_ns = 0U;
    uint64_t seal_end_ns = 0U;
    if (
        !qtw_cpu_time_ns(&seal_start_ns)
        || !qtw_shape(
            width,
            depth,
            &context->final_cells,
            &context->retained_stage_cells,
            &context->carrier_cells
        )
    ) {
        qtw_secure_zero(context, sizeof(*context));
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
    context->width = width;
    context->depth = depth;
#if QTW_ARM != QTW_BASELINE
    context->layout = qtt_make_layout(width, depth);
    if (
        context->layout.carrier_cells != context->carrier_cells
        || context->layout.boundary.start
            != context->retained_stage_cells
        || context->layout.boundary.cells != context->final_cells
    ) {
        qtw_secure_zero(context, sizeof(*context));
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
    const struct process process = {
        .carrier_cells = context->carrier_cells
    };
    context->carrier = make_carrier(&process, QTW_CARRIER_ID);
    context->sealed_carrier = snapshot_carrier(&context->carrier);
    context->rank_plan_hash = qtt_rank_plan_hash(&context->layout);
    context->carrier_creation_count = 1U;
#else
    uint64_t plan_hash = UINT64_C(14695981039346656037);
    for (size_t stage = 1U; stage <= depth; ++stage) {
        for (size_t bond = 0U; bond <= width; ++bond) {
            const uint64_t rank = (uint64_t)qtw_bond_rank(
                width, stage, bond
            );
            const unsigned char *bytes =
                (const unsigned char *)&rank;
            for (size_t byte = 0U; byte < sizeof(rank); ++byte) {
                plan_hash = qtw_hash_byte(plan_hash, bytes[byte]);
            }
        }
    }
    context->rank_plan_hash = plan_hash;
#endif
    if (!qtw_cpu_time_ns(&seal_end_ns) || seal_end_ns < seal_start_ns) {
#if QTW_ARM != QTW_BASELINE
        free_carrier(&context->carrier);
        free_carrier(&context->sealed_carrier);
#endif
        qtw_secure_zero(context, sizeof(*context));
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
    context->seal_cpu_ns = seal_end_ns - seal_start_ns;
    context->state = QTW_READY;
#ifndef QTW_SANITIZER_BUILD
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
#if QTW_ARM != QTW_BASELINE
        free_carrier(&context->carrier);
        free_carrier(&context->sealed_carrier);
#endif
        qtw_secure_zero(context, sizeof(*context));
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
#endif
    return context;
}

static void qtw_destroy(
    struct qtw_context *context,
    size_t mapped_bytes
) {
    if (context == NULL) {
        return;
    }
#if QTW_ARM != QTW_BASELINE
    free_carrier(&context->carrier);
    free_carrier(&context->sealed_carrier);
#endif
    qtw_secure_zero(context, sizeof(*context));
    (void)munlock(context, mapped_bytes);
    (void)munmap(context, mapped_bytes);
}

static int qtw_transact(
    struct qtw_context *context,
    size_t family,
    struct qtw_receipt *receipt
) {
    if (
        context == NULL
        || receipt == NULL
        || family >= QTW_FAMILIES
        || context->state != QTW_READY
#if QTW_ARM != QTW_BASELINE
        || context->carrier_creation_count != 1U
        || restoration(
            &context->carrier,
            &context->sealed_carrier
        ) > RESTORATION_TOLERANCE
#else
        || context->carrier_creation_count != 0U
#endif
    ) {
        return 0;
    }
    context->state = QTW_RUNNING;
#if QTW_ARM == QTW_BASELINE
    *receipt = qtw_direct_receipt(
        context, family, &context->counters
    );
#else
    struct qtt_execution execution = qtt_execute(
        &context->carrier,
        &context->layout,
        family == 0U ? QTT_NEIGHBOR_AND : QTT_NEIGHBOR_OR,
#if QTW_ARM == QTW_SNAPSHOT
        QTT_SNAPSHOT_RELOAD
#else
        QTT_CORRECT
#endif
    );
    const int valid =
        execution.projection.cells == context->final_cells
        && execution.projection.maximum_root_error <= ROOT_TOLERANCE
        && execution.restoration_max_abs <= RESTORATION_TOLERANCE
        && execution.integrity_max_abs <= RESTORATION_TOLERANCE
        && execution.stats.final_decodes == context->final_cells
        && execution.stats.phase_cell_updates > 0U
        && execution.stats.carrier_reads > 0U
        && execution.stats.logical_phase_ands > 0U
        && execution.stats.quotient_member_terms > 0U
#if QTW_ARM == QTW_SNAPSHOT
        && execution.snapshot_loaded
#else
        && !execution.snapshot_loaded
#endif
        && restoration(
            &context->carrier,
            &context->sealed_carrier
        ) <= RESTORATION_TOLERANCE;
    if (!valid) {
        qtt_free_projection(&execution.projection);
        context->state = QTW_FAILED;
        return 0;
    }
    *receipt = (struct qtw_receipt){
        .cells = execution.projection.cells,
        .ones = execution.projection.ones,
        .hash = execution.projection.hash
    };
    context->counters.logical_phase_ands +=
        execution.stats.logical_phase_ands;
    context->counters.logical_phase_ors +=
        execution.stats.logical_phase_ors;
    context->counters.carrier_reads +=
        execution.stats.carrier_reads;
    context->counters.phase_cell_updates +=
        execution.stats.phase_cell_updates;
    context->counters.quotient_member_terms +=
        execution.stats.quotient_member_terms;
    context->counters.final_decodes +=
        execution.stats.final_decodes;
    context->counters.projected_bytes +=
        (uint64_t)execution.projection.cells;
    context->counters.comparison_snapshot_bytes +=
        (uint64_t)(
            2U
            * context->carrier_cells
            * sizeof(double complex)
        );
    context->counters.restoration_scan_cells +=
        (uint64_t)context->carrier_cells;
#if QTW_ARM == QTW_SNAPSHOT
    ++context->counters.snapshot_loads;
    context->counters.snapshot_reload_bytes +=
        (uint64_t)(
            context->carrier_cells * sizeof(double complex)
        );
#else
    ++context->counters.actual_inverse_transactions;
    ++context->restoration_generation;
#endif
    if (
        execution.restoration_max_abs
        > context->maximum_restoration_error
    ) {
        context->maximum_restoration_error =
            execution.restoration_max_abs;
    }
    qtt_free_projection(&execution.projection);
#endif
    if (
        receipt->cells != context->final_cells
        || receipt->hash == 0U
    ) {
        context->state = QTW_FAILED;
        return 0;
    }
    ++context->counters.transactions;
    ++context->total_transactions;
    context->state = QTW_READY;
    return 1;
}

static int qtw_reset_counters(struct qtw_context *context) {
    if (context == NULL || context->state != QTW_READY) {
        return 0;
    }
    memset(&context->counters, 0, sizeof(context->counters));
    if (!qtw_cpu_time_ns(&context->counters.cpu_start_ns)) {
        return 0;
    }
    context->counters.timing_active = 1;
    return 1;
}

static int qtw_make_listener(const char *path) {
    struct stat existing;
    if (lstat(path, &existing) == 0 || errno != ENOENT) {
        return -1;
    }
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        return -1;
    }
    const int listener = socket(
        AF_UNIX,
        SOCK_SEQPACKET | SOCK_CLOEXEC,
        0
    );
    if (listener < 0) {
        return -1;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path) + 1U);
    if (
        bind(
            listener,
            (const struct sockaddr *)&address,
            sizeof(address)
        ) != 0
        || chmod(path, S_IRUSR | S_IWUSR) != 0
        || listen(listener, 1) != 0
    ) {
        (void)close(listener);
        (void)unlink(path);
        return -1;
    }
    return listener;
}

static int qtw_peer_is_same_real_user(int client) {
    struct ucred credential;
    socklen_t size = sizeof(credential);
    return
        getsockopt(
            client,
            SOL_SOCKET,
            SO_PEERCRED,
            &credential,
            &size
        ) == 0
        && size == sizeof(credential)
        && credential.uid == getuid();
}

static int qtw_install_seccomp(int client) {
#ifdef QTW_SANITIZER_BUILD
    (void)client;
    return 1;
#else
    scmp_filter_ctx filter = seccomp_init(SCMP_ACT_KILL_PROCESS);
    if (filter == NULL) {
        return 0;
    }
    int ok = 1;
#define QTW_ALLOW_SYSCALL(name) \
    do { \
        if (seccomp_rule_add( \
            filter, \
            SCMP_ACT_ALLOW, \
            SCMP_SYS(name), \
            0 \
        ) != 0) { \
            ok = 0; \
        } \
    } while (0)
    if (
        seccomp_rule_add(
            filter,
            SCMP_ACT_ALLOW,
            SCMP_SYS(recvfrom),
            1,
            SCMP_A0(SCMP_CMP_EQ, (scmp_datum_t)client)
        ) != 0
        || seccomp_rule_add(
            filter,
            SCMP_ACT_ALLOW,
            SCMP_SYS(sendto),
            1,
            SCMP_A0(SCMP_CMP_EQ, (scmp_datum_t)client)
        ) != 0
    ) {
        ok = 0;
    }
    QTW_ALLOW_SYSCALL(brk);
    QTW_ALLOW_SYSCALL(clock_gettime);
    QTW_ALLOW_SYSCALL(close);
    QTW_ALLOW_SYSCALL(exit);
    QTW_ALLOW_SYSCALL(exit_group);
    QTW_ALLOW_SYSCALL(madvise);
    QTW_ALLOW_SYSCALL(mmap);
    QTW_ALLOW_SYSCALL(mprotect);
    QTW_ALLOW_SYSCALL(mremap);
    QTW_ALLOW_SYSCALL(munlock);
    QTW_ALLOW_SYSCALL(munmap);
    QTW_ALLOW_SYSCALL(rt_sigreturn);
#undef QTW_ALLOW_SYSCALL
    if (!ok || seccomp_load(filter) != 0) {
        seccomp_release(filter);
        return 0;
    }
    seccomp_release(filter);
    return 1;
#endif
}

static int qtw_send(int client, const char *response) {
    const size_t bytes = strlen(response);
    return
        bytes > 0U
        && bytes < QTW_RESPONSE_CAPACITY
        && send(client, response, bytes, MSG_NOSIGNAL)
            == (ssize_t)bytes;
}

static int qtw_receive(
    int client,
    char request[QTW_REQUEST_CAPACITY]
) {
    const ssize_t received = recv(
        client,
        request,
        QTW_REQUEST_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (received <= 0) {
        return 0;
    }
    if (
        (size_t)received >= QTW_REQUEST_CAPACITY
        || memchr(request, '\0', (size_t)received) != NULL
    ) {
        qtw_secure_zero(request, QTW_REQUEST_CAPACITY);
        return -1;
    }
    request[received] = '\0';
    return 1;
}

static int qtw_format_receipt(
    char response[QTW_RESPONSE_CAPACITY],
    size_t family,
    const struct qtw_context *context,
    const struct qtw_receipt *receipt
) {
    const int written = snprintf(
        response,
        QTW_RESPONSE_CAPACITY,
        "OK FINAL %zu %016llx %016llx %zu %zu",
        family,
        (unsigned long long)context->rank_plan_hash,
        (unsigned long long)receipt->hash,
        receipt->ones,
        receipt->cells
    );
    return written > 0 && (size_t)written < QTW_RESPONSE_CAPACITY;
}

static int qtw_format_stats(
    char response[QTW_RESPONSE_CAPACITY],
    const struct qtw_context *context
) {
    if (!context->counters.timing_active) {
        return 0;
    }
    uint64_t cpu_end_ns = 0U;
    if (
        !qtw_cpu_time_ns(&cpu_end_ns)
        || cpu_end_ns < context->counters.cpu_start_ns
    ) {
        return 0;
    }
    const int written = snprintf(
        response,
        QTW_RESPONSE_CAPACITY,
        "OK STATS "
        "%llu %llu %llu %llu %llu %llu %llu %llu %llu "
        "%llu %llu %llu %llu %llu %llu %llu %llu %llu "
        "%llu %.17g",
        (unsigned long long)context->counters.transactions,
        (unsigned long long)context->counters.direct_cells,
        (unsigned long long)context->counters.direct_predicates,
        (unsigned long long)context->counters.logical_phase_ands,
        (unsigned long long)context->counters.logical_phase_ors,
        (unsigned long long)context->counters.carrier_reads,
        (unsigned long long)context->counters.phase_cell_updates,
        (unsigned long long)context->counters.quotient_member_terms,
        (unsigned long long)context->counters.final_decodes,
        (unsigned long long)context->counters.projected_bytes,
        (unsigned long long)
            context->counters.comparison_snapshot_bytes,
        (unsigned long long)context->counters.restoration_scan_cells,
        (unsigned long long)context->counters.snapshot_loads,
        (unsigned long long)context->counters.snapshot_reload_bytes,
        (unsigned long long)
            context->counters.actual_inverse_transactions,
        (unsigned long long)context->restoration_generation,
        (unsigned long long)context->carrier_creation_count,
        (unsigned long long)(
            cpu_end_ns - context->counters.cpu_start_ns
        ),
        (unsigned long long)context->seal_cpu_ns,
        context->maximum_restoration_error
    );
    return written > 0 && (size_t)written < QTW_RESPONSE_CAPACITY;
}

static int qtw_serve(
    int client,
    struct qtw_context *context
) {
    char request[QTW_REQUEST_CAPACITY] = {0};
    char response[QTW_RESPONSE_CAPACITY] = {0};
    struct qtw_receipt receipt = {0};
    int running = 1;
    while (running) {
        const int received = qtw_receive(client, request);
        if (received == 0) {
            break;
        }
        if (received < 0) {
            if (!qtw_send(client, "ERR E_PROTOCOL")) {
                break;
            }
            continue;
        }
        if (strcmp(request, "HELLO") == 0) {
            const int written = snprintf(
                response,
                sizeof(response),
                "OK HELLO %s %zu %zu %u",
                QTW_PROTOCOL,
                context->width,
                context->depth,
                QTW_FAMILIES
            );
            if (
                written <= 0
                || (size_t)written >= sizeof(response)
                || !qtw_send(client, response)
            ) {
                break;
            }
        } else if (strcmp(request, "RESET") == 0) {
            if (
                !qtw_reset_counters(context)
                || !qtw_send(client, "OK RESET")
            ) {
                break;
            }
        } else if (strcmp(request, "STATS") == 0) {
            if (
                !qtw_format_stats(response, context)
                || !qtw_send(client, response)
            ) {
                break;
            }
        } else if (
            strlen(request) == 9U
            && strncmp(request, "EXECUTE ", 8U) == 0
            && request[8] >= '0'
            && request[8] <= '1'
        ) {
            const size_t family = (size_t)(request[8] - '0');
            if (
                !qtw_transact(context, family, &receipt)
                || !qtw_format_receipt(
                    response, family, context, &receipt
                )
                || !qtw_send(client, response)
            ) {
                (void)qtw_send(client, "ERR E_MACHINE_LAW");
                break;
            }
        } else if (strcmp(request, "SHUTDOWN") == 0) {
            if (!qtw_send(client, "OK CLOSED")) {
                break;
            }
            running = 0;
        } else if (
            strncmp(request, "PROJECT ", 8U) == 0
            || strcmp(request, "DUMP") == 0
            || strcmp(request, "DEBUG") == 0
            || strcmp(request, "READ CARRIER") == 0
            || strcmp(request, "STATE DETAIL") == 0
        ) {
            if (!qtw_send(
                client,
                "ERR E_INTERMEDIATE_PROJECTION_DENIED"
            )) {
                break;
            }
        } else if (!qtw_send(client, "ERR E_PROTOCOL")) {
            break;
        }
        qtw_secure_zero(request, sizeof(request));
        qtw_secure_zero(response, sizeof(response));
        qtw_secure_zero(&receipt, sizeof(receipt));
    }
    qtw_secure_zero(request, sizeof(request));
    qtw_secure_zero(response, sizeof(response));
    qtw_secure_zero(&receipt, sizeof(receipt));
    return 1;
}

int main(int argc, char **argv) {
#ifdef QTW_SIZE_PROBE
    if (argc == 4 && strcmp(argv[1], "--size-probe") == 0) {
        size_t width = 0U;
        size_t depth = 0U;
        size_t final_cells = 0U;
        size_t retained_stage_cells = 0U;
        size_t carrier_cells = 0U;
        const long page_size_long = sysconf(_SC_PAGESIZE);
        if (
            page_size_long <= 0
            || !qtw_parse_size(
                argv[2], QTW_MIN_WIDTH, QTW_MAX_WIDTH, &width
            )
            || !qtw_parse_size(
                argv[3], QTW_MIN_DEPTH, QTW_MAX_DEPTH, &depth
            )
            || depth > width
            || !qtw_shape(
                width,
                depth,
                &final_cells,
                &retained_stage_cells,
                &carrier_cells
            )
        ) {
            return 2;
        }
        printf(
            "{\"arm\":%d,"
            "\"width\":%zu,"
            "\"depth\":%zu,"
            "\"context_bytes\":%zu,"
            "\"mapped_context_bytes\":%zu,"
            "\"compiled_plan_bytes\":%zu,"
            "\"final_cells\":%zu,"
            "\"retained_stage_cells\":%zu,"
            "\"predecessor_inverse_history_cells\":%zu,"
            "\"carrier_cells\":%zu,"
            "\"carrier_creation_count\":%u,"
            "\"live_carrier_bytes\":%zu,"
            "\"sealed_verification_state_bytes\":%zu,"
            "\"comparison_snapshot_bytes_per_transaction\":%zu,"
            "\"snapshot_reload_bytes_per_transaction\":%zu,"
            "\"projection_bytes_per_transaction\":%zu,"
            "\"rematerialized_cells_per_transaction\":0,"
            "\"request_buffer_bytes\":%u,"
            "\"response_buffer_bytes\":%u}\n",
            QTW_ARM,
            width,
            depth,
            sizeof(struct qtw_context),
            qtw_page_round(
                sizeof(struct qtw_context),
                (size_t)page_size_long
            ),
#if QTW_ARM == QTW_BASELINE
            3U * sizeof(size_t),
#else
            sizeof(struct qtt_layout),
#endif
            final_cells,
            retained_stage_cells,
            retained_stage_cells - final_cells,
            carrier_cells,
#if QTW_ARM == QTW_BASELINE
            0U,
            (size_t)0U,
            (size_t)0U,
            (size_t)0U,
            (size_t)0U,
#else
            1U,
            2U * carrier_cells * sizeof(double complex),
            2U * carrier_cells * sizeof(double complex),
            2U * carrier_cells * sizeof(double complex),
#if QTW_ARM == QTW_SNAPSHOT
            carrier_cells * sizeof(double complex),
#else
            (size_t)0U,
#endif
#endif
            final_cells,
            QTW_REQUEST_CAPACITY,
            QTW_RESPONSE_CAPACITY
        );
        return 0;
    }
#endif
    if (argc != 4) {
        return 2;
    }
    size_t width = 0U;
    size_t depth = 0U;
    if (
        !qtw_parse_size(
            argv[2], QTW_MIN_WIDTH, QTW_MAX_WIDTH, &width
        )
        || !qtw_parse_size(
            argv[3], QTW_MIN_DEPTH, QTW_MAX_DEPTH, &depth
        )
        || depth > width
        || !qtw_process_is_untraced()
        || !qtw_establish_process_guards()
    ) {
        return 2;
    }
    (void)umask(S_IRWXG | S_IRWXO);
    size_t mapped_bytes = 0U;
    struct qtw_context *context = qtw_create(
        width, depth, &mapped_bytes
    );
    if (context == NULL) {
        return 2;
    }
    const int listener = qtw_make_listener(argv[1]);
    if (listener < 0) {
        qtw_destroy(context, mapped_bytes);
        return 2;
    }
    const int client = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0 || !qtw_peer_is_same_real_user(client)) {
        if (client >= 0) {
            (void)close(client);
        }
        (void)close(listener);
        (void)unlink(argv[1]);
        qtw_destroy(context, mapped_bytes);
        return 2;
    }
    (void)close(listener);
    if (unlink(argv[1]) != 0) {
        (void)close(client);
        qtw_destroy(context, mapped_bytes);
        return 2;
    }
    if (!qtw_install_seccomp(client)) {
        (void)close(client);
        qtw_destroy(context, mapped_bytes);
        return 2;
    }
    (void)qtw_serve(client, context);
    (void)close(client);
    qtw_destroy(context, mapped_bytes);
    return 0;
}
