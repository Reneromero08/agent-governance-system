#define _GNU_SOURCE

/*
 * Matched persistent-service arms for the fixed-schema QANF Small-Wall
 * obstruction experiment.
 *
 * QSW_ARM=1: compact conventional four-AND boundary evaluator
 * QSW_ARM=2: reviewed phase forward path plus snapshot reload
 * QSW_ARM=3: reviewed phase forward plus actual inverse restoration
 *
 * All arms expose the same packet protocol and final-boundary bytes.  The
 * compact arm recomputes the formula on every request and has no result cache
 * or answer table.
 */

#define QSW_BASELINE 1
#define QSW_SNAPSHOT 2
#define QSW_IN_PLACE 3

#ifndef QSW_ARM
#error "compile with QSW_ARM=1, 2, or 3"
#endif

#if QSW_ARM < QSW_BASELINE || QSW_ARM > QSW_IN_PLACE
#error "invalid QSW_ARM"
#endif

#if QSW_ARM != QSW_BASELINE
#define main qsw_reviewed_standalone_main
#include "quadratic_anf_chain_phase.c"
#undef main
#endif

#include <errno.h>
#include <fcntl.h>
#include <linux/prctl.h>
#include <math.h>
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

#define QSW_PROTOCOL "CATVM_QANF_SMALL_WALL_COMPARE_1"
#define QSW_VARIANTS 4U
#define QSW_RELATIONS 3U
#define QSW_COEFFICIENTS 3U
#define QSW_BOUNDARY_CELLS 5U
#define QSW_REQUEST_CAPACITY 128U
#define QSW_RESPONSE_CAPACITY 512U
#define QSW_MAX_SOURCE_BYTES 4096U
#define QSW_CARRIER_ID 71
#define QSW_RELOAD_BYTES 368U

enum qsw_state {
    QSW_READY = 1,
    QSW_RUNNING = 2,
    QSW_FAILED = 3
};

struct qsw_program {
    unsigned char coefficient[QSW_RELATIONS][QSW_COEFFICIENTS];
    uint64_t source_fnv1a64;
};

struct qsw_boundary {
    int coefficient[QSW_BOUNDARY_CELLS];
};

struct qsw_counters {
    uint64_t transactions;
    uint64_t boolean_ands;
    uint64_t phase_products;
    uint64_t carrier_reads;
    uint64_t phase_cell_updates;
    uint64_t final_decodes;
    uint64_t snapshot_loads;
    uint64_t snapshot_reload_bytes;
    uint64_t actual_inverse_transactions;
    uint64_t cpu_start_ns;
    int timing_active;
};

struct qsw_context {
    struct qsw_program program[QSW_VARIANTS];
    struct qsw_program sealed_program[QSW_VARIANTS];
#if QSW_ARM != QSW_BASELINE
    struct qb_carrier carrier;
    struct qb_carrier sealed_carrier;
#endif
    struct qsw_counters counters;
    enum qsw_state state;
    uint64_t carrier_creation_count;
    uint64_t restoration_generation;
    uint64_t total_transactions;
    uint64_t seal_cpu_ns;
};

static void qsw_secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static uint64_t qsw_hash_bytes(
    uint64_t hash,
    const unsigned char *bytes,
    size_t count
) {
    for (size_t index = 0U; index < count; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int qsw_cpu_time_ns(uint64_t *time_ns) {
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

static int qsw_read_program(
    const char *path,
    struct qsw_program *program
) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    if (
        fseek(file, 0L, SEEK_END) != 0
        || ftell(file) < 0L
    ) {
        (void)fclose(file);
        return 0;
    }
    const long measured = ftell(file);
    if (
        measured < 0L
        || (unsigned long)measured > QSW_MAX_SOURCE_BYTES
        || fseek(file, 0L, SEEK_SET) != 0
    ) {
        (void)fclose(file);
        return 0;
    }
    const size_t source_bytes = (size_t)measured;
    unsigned char source[QSW_MAX_SOURCE_BYTES + 1U];
    memset(source, 0, sizeof(source));
    if (
        fread(source, 1U, source_bytes, file) != source_bytes
        || ferror(file)
        || fclose(file) != 0
    ) {
        qsw_secure_zero(source, sizeof(source));
        return 0;
    }
    for (size_t index = 0U; index < source_bytes; ++index) {
        if (source[index] == '\0' || source[index] == '\r') {
            qsw_secure_zero(source, sizeof(source));
            return 0;
        }
    }
    source[source_bytes] = '\0';

    char signature[32];
    char type_label[16];
    char type_value[32];
    char relation[QSW_RELATIONS][4];
    char end[8];
    int version = 0;
    int value[QSW_RELATIONS][QSW_COEFFICIENTS];
    int consumed = 0;
    const int fields = sscanf(
        (const char *)source,
        "%31s %d %15s %31s "
        "%3s %d %d %d "
        "%3s %d %d %d "
        "%3s %d %d %d "
        "%7s %n",
        signature,
        &version,
        type_label,
        type_value,
        relation[0],
        &value[0][0],
        &value[0][1],
        &value[0][2],
        relation[1],
        &value[1][0],
        &value[1][1],
        &value[1][2],
        relation[2],
        &value[2][0],
        &value[2][1],
        &value[2][2],
        end,
        &consumed
    );
    int ok =
        fields == 17
        && strcmp(signature, "CATCAS_QUADRATIC_ANF_CHAIN") == 0
        && version == 1
        && strcmp(type_label, "TYPE") == 0
        && strcmp(type_value, "BOOLEAN_ANF_GF2") == 0
        && strcmp(relation[0], "F") == 0
        && strcmp(relation[1], "G") == 0
        && strcmp(relation[2], "J") == 0
        && strcmp(end, "END") == 0
        && consumed >= 0;
    for (
        size_t index = (size_t)(consumed < 0 ? 0 : consumed);
        ok && index < source_bytes;
        ++index
    ) {
        ok = source[index] == ' '
            || source[index] == '\t'
            || source[index] == '\n';
    }
    memset(program, 0, sizeof(*program));
    for (size_t row = 0U; ok && row < QSW_RELATIONS; ++row) {
        for (
            size_t coefficient = 0U;
            coefficient < QSW_COEFFICIENTS;
            ++coefficient
        ) {
            if (
                value[row][coefficient] < 0
                || value[row][coefficient] > 1
            ) {
                ok = 0;
                break;
            }
            program->coefficient[row][coefficient] =
                (unsigned char)value[row][coefficient];
        }
        if (program->coefficient[row][0] != 1U) {
            ok = 0;
        }
    }
    program->source_fnv1a64 = qsw_hash_bytes(
        UINT64_C(14695981039346656037),
        source,
        source_bytes
    );
    qsw_secure_zero(source, sizeof(source));
    if (!ok) {
        qsw_secure_zero(program, sizeof(*program));
    }
    return ok;
}

static size_t qsw_page_round(size_t bytes, size_t page_size) {
    if (bytes > SIZE_MAX - page_size + 1U) {
        return 0U;
    }
    return ((bytes + page_size - 1U) / page_size) * page_size;
}

static int qsw_process_is_untraced(void) {
#ifdef QSW_TRACE_BUILD
    return 1;
#else
    FILE *status = fopen("/proc/self/status", "r");
    if (status == NULL) {
        return 0;
    }
    char line[128];
    int tracer = -1;
    while (fgets(line, sizeof(line), status) != NULL) {
        if (sscanf(line, "TracerPid:\t%d", &tracer) == 1) {
            break;
        }
    }
    const int close_ok = fclose(status) == 0;
    return close_ok && tracer == 0;
#endif
}

static int qsw_establish_process_guards(void) {
    struct rlimit no_core = {.rlim_cur = 0U, .rlim_max = 0U};
    if (
        setrlimit(RLIMIT_CORE, &no_core) != 0
        || prctl(PR_SET_DUMPABLE, 0L, 0L, 0L, 0L) != 0
    ) {
        return 0;
    }
#ifdef PR_SET_PTRACER
    if (prctl(PR_SET_PTRACER, 0L, 0L, 0L, 0L) != 0) {
        return 0;
    }
#endif
    return
        prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) == 0
        && prctl(PR_GET_DUMPABLE, 0L, 0L, 0L, 0L) == 0;
}

static struct qsw_context *qsw_create(
    const char *const path[QSW_VARIANTS],
    size_t *mapped_bytes
) {
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        return NULL;
    }
    *mapped_bytes = qsw_page_round(
        sizeof(struct qsw_context),
        (size_t)page_size_long
    );
    if (*mapped_bytes == 0U) {
        return NULL;
    }
    struct qsw_context *context = mmap(
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
    if (!qsw_cpu_time_ns(&seal_start_ns)) {
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
    for (size_t variant = 0U; variant < QSW_VARIANTS; ++variant) {
        if (!qsw_read_program(path[variant], &context->program[variant])) {
            qsw_secure_zero(context, sizeof(*context));
            (void)munlock(context, *mapped_bytes);
            (void)munmap(context, *mapped_bytes);
            return NULL;
        }
    }
    memcpy(
        context->sealed_program,
        context->program,
        sizeof(context->program)
    );
#if QSW_ARM != QSW_BASELINE
    context->carrier = qb_make_carrier(QSW_CARRIER_ID);
    context->sealed_carrier = context->carrier;
    context->carrier_creation_count = 1U;
#endif
    if (!qsw_cpu_time_ns(&seal_end_ns) || seal_end_ns < seal_start_ns) {
        qsw_secure_zero(context, sizeof(*context));
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
    context->seal_cpu_ns = seal_end_ns - seal_start_ns;
    context->state = QSW_READY;
#ifndef QSW_SANITIZER_BUILD
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        qsw_secure_zero(context, sizeof(*context));
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
#endif
    return context;
}

static void qsw_destroy(
    struct qsw_context *context,
    size_t mapped_bytes
) {
    if (context == NULL) {
        return;
    }
    qsw_secure_zero(context, sizeof(*context));
    (void)munlock(context, mapped_bytes);
    (void)munmap(context, mapped_bytes);
}

#if QSW_ARM == QSW_BASELINE
static struct qsw_boundary qsw_compact_boundary(
    const struct qsw_program *program,
    struct qsw_counters *counters
) {
    const unsigned char alpha = program->coefficient[0][1];
    const unsigned char beta = program->coefficient[0][2];
    const unsigned char gamma = program->coefficient[1][1];
    const unsigned char delta = program->coefficient[1][2];
    const unsigned char eta = program->coefficient[2][1];
    const unsigned char theta = program->coefficient[2][2];
    const unsigned char theta_delta =
        (unsigned char)(theta & delta);
    const unsigned char theta_gamma =
        (unsigned char)(theta & gamma);
    const unsigned char theta_delta_alpha =
        (unsigned char)(theta_delta & alpha);
    const unsigned char theta_delta_beta =
        (unsigned char)(theta_delta & beta);
    counters->boolean_ands += 4U;
    const struct qsw_boundary boundary = {
        .coefficient = {
            1,
            (int)eta,
            (int)theta_gamma,
            (int)theta_delta_alpha,
            (int)theta_delta_beta
        }
    };
    return boundary;
}
#endif

#if QSW_ARM != QSW_BASELINE
static struct qb_program qsw_phase_program(
    const struct qsw_program *program
) {
    struct qb_program phase_program;
    memset(&phase_program, 0, sizeof(phase_program));
    memcpy(
        phase_program.coefficient,
        program->coefficient,
        sizeof(phase_program.coefficient)
    );
    phase_program.source_fnv1a64 = program->source_fnv1a64;
    return phase_program;
}

static int qsw_phase_execution_valid(
    const struct qb_execution *execution
) {
#if QSW_ARM == QSW_SNAPSHOT
    return
        !execution->actual_inverse
        && execution->snapshot_loaded
        && execution->restoration_max_abs == 0.0
        && execution->boundary.maximum_root_error <= QB_ROOT_TOLERANCE
        && execution->stats.phase_ands == 9U
        && execution->stats.carrier_reads == 51U
        && execution->stats.phase_cell_updates == 23U
        && execution->stats.boundary_decodes == 5U
        && execution->stats.intermediate_decodes == 0U
        && execution->stats.intermediate_copies == 0U
        && execution->stats.boundary_copies == 1U
        && execution->stats.snapshot_loads == 1U;
#else
    return
        execution->actual_inverse
        && !execution->snapshot_loaded
        && execution->restoration_max_abs
            <= QB_RESTORATION_TOLERANCE
        && execution->carrier_integrity_max_abs
            <= QB_RESTORATION_TOLERANCE
        && execution->boundary.maximum_root_error <= QB_ROOT_TOLERANCE
        && execution->stats.phase_ands == 18U
        && execution->stats.carrier_reads == 97U
        && execution->stats.phase_cell_updates == 46U
        && execution->stats.boundary_decodes == 5U
        && execution->stats.intermediate_decodes == 0U
        && execution->stats.intermediate_copies == 0U
        && execution->stats.boundary_copies == 2U
        && execution->stats.snapshot_loads == 0U;
#endif
}
#endif

static int qsw_transact(
    struct qsw_context *context,
    size_t variant,
    struct qsw_boundary *boundary
) {
    if (
        context == NULL
        || boundary == NULL
        || variant >= QSW_VARIANTS
        || context->state != QSW_READY
        || memcmp(
            context->program,
            context->sealed_program,
            sizeof(context->program)
        ) != 0
#if QSW_ARM != QSW_BASELINE
        || context->carrier_creation_count != 1U
        || memcmp(
            context->carrier.baseline,
            context->sealed_carrier.baseline,
            sizeof(context->carrier.baseline)
        ) != 0
        || qb_restoration_error(
            &context->carrier,
            &context->sealed_carrier
        ) > QB_RESTORATION_TOLERANCE
#else
        || context->carrier_creation_count != 0U
#endif
    ) {
        return 0;
    }
    context->state = QSW_RUNNING;
#if QSW_ARM == QSW_BASELINE
    *boundary = qsw_compact_boundary(
        &context->program[variant],
        &context->counters
    );
#else
    struct qb_program phase_program =
        qsw_phase_program(&context->program[variant]);
    const struct qb_execution execution = qb_execute(
        &context->carrier,
        &phase_program,
#if QSW_ARM == QSW_SNAPSHOT
        QB_INVERSE_SNAPSHOT
#else
        QB_INVERSE_CORRECT
#endif
    );
    qsw_secure_zero(&phase_program, sizeof(phase_program));
    if (
        !qsw_phase_execution_valid(&execution)
        || qb_restoration_error(
            &context->carrier,
            &context->sealed_carrier
        ) > QB_RESTORATION_TOLERANCE
    ) {
        context->state = QSW_FAILED;
        return 0;
    }
    for (size_t cell = 0U; cell < QSW_BOUNDARY_CELLS; ++cell) {
        boundary->coefficient[cell] =
            execution.boundary.coefficient[cell];
    }
    context->counters.phase_products +=
        execution.stats.phase_ands;
    context->counters.carrier_reads +=
        execution.stats.carrier_reads;
    context->counters.phase_cell_updates +=
        execution.stats.phase_cell_updates;
    context->counters.final_decodes +=
        execution.stats.boundary_decodes;
#if QSW_ARM == QSW_SNAPSHOT
    ++context->counters.snapshot_loads;
    context->counters.snapshot_reload_bytes += QSW_RELOAD_BYTES;
#else
    ++context->counters.actual_inverse_transactions;
    ++context->restoration_generation;
#endif
#endif
    if (
        memcmp(
            context->program,
            context->sealed_program,
            sizeof(context->program)
        ) != 0
    ) {
        context->state = QSW_FAILED;
        return 0;
    }
    ++context->counters.transactions;
    ++context->total_transactions;
    context->state = QSW_READY;
    return 1;
}

static int qsw_reset_counters(struct qsw_context *context) {
    if (context == NULL || context->state != QSW_READY) {
        return 0;
    }
    memset(&context->counters, 0, sizeof(context->counters));
    if (!qsw_cpu_time_ns(&context->counters.cpu_start_ns)) {
        return 0;
    }
    context->counters.timing_active = 1;
    return 1;
}

static int qsw_make_listener(const char *path) {
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

static int qsw_peer_is_same_real_user(int client) {
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

static int qsw_install_seccomp(int client) {
#ifdef QSW_SANITIZER_BUILD
    (void)client;
    return 1;
#else
    scmp_filter_ctx filter = seccomp_init(SCMP_ACT_KILL_PROCESS);
    if (filter == NULL) {
        return 0;
    }
    int ok = 1;
#define QSW_ALLOW_SYSCALL(name) \
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
    QSW_ALLOW_SYSCALL(brk);
    QSW_ALLOW_SYSCALL(clock_gettime);
    QSW_ALLOW_SYSCALL(close);
    QSW_ALLOW_SYSCALL(exit);
    QSW_ALLOW_SYSCALL(exit_group);
    QSW_ALLOW_SYSCALL(madvise);
    QSW_ALLOW_SYSCALL(mmap);
    QSW_ALLOW_SYSCALL(mprotect);
    QSW_ALLOW_SYSCALL(mremap);
    QSW_ALLOW_SYSCALL(munlock);
    QSW_ALLOW_SYSCALL(munmap);
    QSW_ALLOW_SYSCALL(rt_sigreturn);
#undef QSW_ALLOW_SYSCALL
    if (!ok || seccomp_load(filter) != 0) {
        seccomp_release(filter);
        return 0;
    }
    return 1;
#endif
}

static int qsw_send(int client, const char *response) {
    const size_t bytes = strlen(response);
    return
        bytes > 0U
        && bytes < QSW_RESPONSE_CAPACITY
        && send(client, response, bytes, MSG_NOSIGNAL)
            == (ssize_t)bytes;
}

static int qsw_receive(
    int client,
    char request[QSW_REQUEST_CAPACITY]
) {
    const ssize_t received = recv(
        client,
        request,
        QSW_REQUEST_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (received <= 0) {
        return 0;
    }
    if (
        (size_t)received >= QSW_REQUEST_CAPACITY
        || memchr(request, '\0', (size_t)received) != NULL
    ) {
        qsw_secure_zero(request, QSW_REQUEST_CAPACITY);
        return -1;
    }
    request[received] = '\0';
    return 1;
}

static int qsw_format_boundary(
    char response[QSW_RESPONSE_CAPACITY],
    size_t variant,
    const struct qsw_boundary *boundary
) {
    const int written = snprintf(
        response,
        QSW_RESPONSE_CAPACITY,
        "OK FINAL %zu 5 %d %d %d %d %d",
        variant,
        boundary->coefficient[0],
        boundary->coefficient[1],
        boundary->coefficient[2],
        boundary->coefficient[3],
        boundary->coefficient[4]
    );
    return written > 0 && (size_t)written < QSW_RESPONSE_CAPACITY;
}

static int qsw_format_stats(
    char response[QSW_RESPONSE_CAPACITY],
    const struct qsw_context *context
) {
    if (!context->counters.timing_active) {
        return 0;
    }
    uint64_t cpu_end_ns = 0U;
    if (
        !qsw_cpu_time_ns(&cpu_end_ns)
        || cpu_end_ns < context->counters.cpu_start_ns
    ) {
        return 0;
    }
    const int written = snprintf(
        response,
        QSW_RESPONSE_CAPACITY,
        "OK STATS %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
        (unsigned long long)context->counters.transactions,
        (unsigned long long)context->counters.boolean_ands,
        (unsigned long long)context->counters.phase_products,
        (unsigned long long)context->counters.carrier_reads,
        (unsigned long long)context->counters.phase_cell_updates,
        (unsigned long long)context->counters.final_decodes,
        (unsigned long long)context->counters.snapshot_loads,
        (unsigned long long)context->counters.snapshot_reload_bytes,
        (unsigned long long)
            context->counters.actual_inverse_transactions,
        (unsigned long long)context->restoration_generation,
        (unsigned long long)context->carrier_creation_count,
        (unsigned long long)(cpu_end_ns
            - context->counters.cpu_start_ns),
        (unsigned long long)context->seal_cpu_ns
    );
    return written > 0 && (size_t)written < QSW_RESPONSE_CAPACITY;
}

static int qsw_serve(
    int client,
    struct qsw_context *context
) {
    char request[QSW_REQUEST_CAPACITY] = {0};
    char response[QSW_RESPONSE_CAPACITY] = {0};
    struct qsw_boundary boundary = {0};
    int running = 1;
    while (running) {
        const int received = qsw_receive(client, request);
        if (received == 0) {
            break;
        }
        if (received < 0) {
            if (!qsw_send(client, "ERR E_PROTOCOL")) {
                break;
            }
            continue;
        }
        if (strcmp(request, "HELLO") == 0) {
            if (!qsw_send(
                client,
                "OK HELLO CATVM_QANF_SMALL_WALL_COMPARE_1 4 5"
            )) {
                break;
            }
        } else if (strcmp(request, "RESET") == 0) {
            if (
                !qsw_reset_counters(context)
                || !qsw_send(client, "OK RESET")
            ) {
                break;
            }
        } else if (strcmp(request, "STATS") == 0) {
            if (
                !qsw_format_stats(response, context)
                || !qsw_send(client, response)
            ) {
                break;
            }
        } else if (
            strlen(request) == 9U
            && strncmp(request, "EXECUTE ", 8U) == 0
            && request[8] >= '0'
            && request[8] <= '3'
        ) {
            const size_t variant = (size_t)(request[8] - '0');
            if (
                !qsw_transact(context, variant, &boundary)
                || !qsw_format_boundary(
                    response,
                    variant,
                    &boundary
                )
                || !qsw_send(client, response)
            ) {
                (void)qsw_send(client, "ERR E_MACHINE_LAW");
                break;
            }
        } else if (strcmp(request, "SHUTDOWN") == 0) {
            if (!qsw_send(client, "OK CLOSED")) {
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
            if (!qsw_send(
                client,
                "ERR E_INTERMEDIATE_PROJECTION_DENIED"
            )) {
                break;
            }
        } else if (!qsw_send(client, "ERR E_PROTOCOL")) {
            break;
        }
        qsw_secure_zero(request, sizeof(request));
        qsw_secure_zero(response, sizeof(response));
        qsw_secure_zero(&boundary, sizeof(boundary));
    }
    qsw_secure_zero(request, sizeof(request));
    qsw_secure_zero(response, sizeof(response));
    qsw_secure_zero(&boundary, sizeof(boundary));
    return 1;
}

int main(int argc, char **argv) {
#ifdef QSW_SIZE_PROBE
    if (argc == 2 && strcmp(argv[1], "--size-probe") == 0) {
        const long page_size_long = sysconf(_SC_PAGESIZE);
        if (page_size_long <= 0) {
            return 2;
        }
        printf(
            "{\"arm\":%d,"
            "\"context_bytes\":%zu,"
            "\"mapped_context_bytes\":%zu,"
            "\"program_table_bytes\":%zu,"
            "\"sealed_program_table_bytes\":%zu,"
            "\"carrier_creation_count\":%u,"
            "\"live_carrier_bytes\":%zu,"
            "\"sealed_carrier_bytes\":%zu,"
            "\"execution_snapshot_bytes\":%zu,"
            "\"working_reload_bytes\":%u,"
            "\"request_buffer_bytes\":%u,"
            "\"response_buffer_bytes\":%u}\n",
            QSW_ARM,
            sizeof(struct qsw_context),
            qsw_page_round(
                sizeof(struct qsw_context),
                (size_t)page_size_long
            ),
            sizeof(((struct qsw_context *)0)->program),
            sizeof(((struct qsw_context *)0)->sealed_program),
#if QSW_ARM == QSW_BASELINE
            0U,
            (size_t)0U,
            (size_t)0U,
            (size_t)0U,
            0U,
#else
            1U,
            sizeof(struct qb_carrier),
            sizeof(struct qb_carrier),
            sizeof(struct qb_carrier),
#if QSW_ARM == QSW_SNAPSHOT
            QSW_RELOAD_BYTES,
#else
            0U,
#endif
#endif
            QSW_REQUEST_CAPACITY,
            QSW_RESPONSE_CAPACITY
        );
        return 0;
    }
#endif
    if (argc != 6) {
        return 2;
    }
    if (
        !qsw_process_is_untraced()
        || !qsw_establish_process_guards()
    ) {
        return 2;
    }
    (void)umask(S_IRWXG | S_IRWXO);
    const char *const path[QSW_VARIANTS] = {
        argv[2], argv[3], argv[4], argv[5]
    };
    size_t mapped_bytes = 0U;
    struct qsw_context *context =
        qsw_create(path, &mapped_bytes);
    if (context == NULL) {
        return 2;
    }
    const int listener = qsw_make_listener(argv[1]);
    if (listener < 0) {
        qsw_destroy(context, mapped_bytes);
        return 2;
    }
    const int client = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0 || !qsw_peer_is_same_real_user(client)) {
        if (client >= 0) {
            (void)close(client);
        }
        (void)close(listener);
        (void)unlink(argv[1]);
        qsw_destroy(context, mapped_bytes);
        return 2;
    }
    (void)close(listener);
    if (unlink(argv[1]) != 0) {
        (void)close(client);
        qsw_destroy(context, mapped_bytes);
        return 2;
    }
    if (!qsw_install_seccomp(client)) {
        (void)close(client);
        qsw_destroy(context, mapped_bytes);
        return 2;
    }
    (void)qsw_serve(client, context);
    (void)close(client);
    qsw_destroy(context, mapped_bytes);
    return 0;
}
