#define _GNU_SOURCE

/*
 * Minimal CATVM custody boundary for the reviewed quadratic-ANF phase chain.
 *
 * The service owns the only carrier and all F/H/Z phase-resident state.  A
 * production client can request only an atomic public-program transaction:
 * encode F/G/J, construct unresolved H, construct and project final Z, reverse
 * Z/H/J/G/F on the actual carrier, verify restoration, and then return the
 * already-latched five-coefficient boundary.
 */

#define main qcs_reviewed_standalone_main
#include "quadratic_anf_chain_phase.c"
#undef main

#include <fcntl.h>
#include <linux/prctl.h>
#include <seccomp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#define QCS_PROTOCOL "CATVM_QANF_PHASE_1"
#define QCS_VARIANTS 4U
#define QCS_REQUEST_CAPACITY 128U
#define QCS_RESPONSE_CAPACITY 512U
#define QCS_CARRIER_ID 67
#define QCS_EXPECTED_PHASE_ANDS 18U
#define QCS_EXPECTED_CARRIER_READS 97U
#define QCS_EXPECTED_CELL_UPDATES 46U

enum qcs_state {
    QCS_READY = 1,
    QCS_RUNNING = 2,
    QCS_FAILED = 3
};

struct qcs_context {
    struct qb_program program[QCS_VARIANTS];
    struct qb_program sealed_program[QCS_VARIANTS];
    struct qb_carrier carrier;
    struct qb_carrier sealed_state;
    enum qcs_state state;
    uint64_t program_custody_hash;
    uint64_t restoration_generation;
    uint64_t carrier_creation_count;
    uint64_t completed_transactions;
#ifdef CATVM_QANF_TESTING
    enum qb_inverse_mode testing_mode;
    int testing_inert;
#endif
};

struct qcs_outcome {
    struct qb_boundary boundary;
    uint64_t boundary_hash;
    uint64_t plan_hash;
    uint64_t restoration_generation;
    uint64_t carrier_creation_count;
};

static void qcs_secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static size_t qcs_page_round(size_t bytes, size_t page_size) {
    if (bytes > SIZE_MAX - page_size + 1U) {
        return 0U;
    }
    return ((bytes + page_size - 1U) / page_size) * page_size;
}

static uint64_t qcs_program_hash(
    const struct qb_program program[QCS_VARIANTS]
) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t variant = 0U; variant < QCS_VARIANTS; ++variant) {
        hash = qb_hash_bytes(
            hash,
            &program[variant].coefficient[0][0],
            QB_INPUT_RELATIONS * QB_INPUT_COEFFICIENTS
        );
        const uint64_t source_hash = program[variant].source_fnv1a64;
        hash = qb_hash_bytes(
            hash,
            (const unsigned char *)&source_hash,
            sizeof(source_hash)
        );
    }
    return hash;
}

static uint64_t qcs_boundary_hash(const struct qb_boundary *boundary) {
    unsigned char coefficient[QB_Z_COEFFICIENTS];
    for (size_t cell = 0U; cell < QB_Z_COEFFICIENTS; ++cell) {
        coefficient[cell] =
            (unsigned char)boundary->coefficient[cell];
    }
    return qb_hash_bytes(
        UINT64_C(14695981039346656037),
        coefficient,
        sizeof(coefficient)
    );
}

static int qcs_process_is_untraced(void) {
#ifdef CATVM_QANF_TRACE_BUILD
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

static int qcs_establish_process_guards(void) {
    struct rlimit no_core = {.rlim_cur = 0U, .rlim_max = 0U};
    if (setrlimit(RLIMIT_CORE, &no_core) != 0) {
        return 0;
    }
    if (prctl(PR_SET_DUMPABLE, 0L, 0L, 0L, 0L) != 0) {
        return 0;
    }
#ifdef PR_SET_PTRACER
    if (prctl(PR_SET_PTRACER, 0L, 0L, 0L, 0L) != 0) {
        return 0;
    }
#endif
    if (prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) != 0) {
        return 0;
    }
    return prctl(PR_GET_DUMPABLE, 0L, 0L, 0L, 0L) == 0;
}

static struct qcs_context *qcs_create(
    const char *const program_path[QCS_VARIANTS],
    size_t *mapped_bytes
#ifdef CATVM_QANF_TESTING
    ,
    enum qb_inverse_mode testing_mode,
    int testing_inert
#endif
) {
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        return NULL;
    }
    *mapped_bytes = qcs_page_round(
        sizeof(struct qcs_context),
        (size_t)page_size_long
    );
    if (*mapped_bytes == 0U) {
        return NULL;
    }
    struct qcs_context *context = mmap(
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
    for (size_t variant = 0U; variant < QCS_VARIANTS; ++variant) {
        context->program[variant] =
            qb_read_program(program_path[variant]);
    }
    memcpy(
        context->sealed_program,
        context->program,
        sizeof(context->program)
    );
    context->carrier = qb_make_carrier(QCS_CARRIER_ID);
    context->sealed_state = context->carrier;
    context->program_custody_hash =
        qcs_program_hash(context->program);
    context->state = QCS_READY;
    context->carrier_creation_count = 1U;
#ifdef CATVM_QANF_TESTING
    context->testing_mode = testing_mode;
    context->testing_inert = testing_inert;
#endif
#ifndef CATVM_SANITIZER_BUILD
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        qcs_secure_zero(context, sizeof(*context));
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
#endif
    return context;
}

static void qcs_destroy(
    struct qcs_context *context,
    size_t mapped_bytes
) {
    if (context == NULL) {
        return;
    }
    qcs_secure_zero(context, sizeof(*context));
    (void)munlock(context, mapped_bytes);
    (void)munmap(context, mapped_bytes);
}

static int qcs_stats_are_accepted(const struct qb_execution *execution) {
    return
        execution->stats.phase_ands == QCS_EXPECTED_PHASE_ANDS
        && execution->stats.carrier_reads
            == QCS_EXPECTED_CARRIER_READS
        && execution->stats.phase_cell_updates
            == QCS_EXPECTED_CELL_UPDATES
        && execution->stats.boundary_decodes == QB_Z_COEFFICIENTS
        && execution->stats.intermediate_decodes == 0U
        && execution->stats.intermediate_copies == 0U
        && execution->stats.boundary_copies == 2U
        && execution->stats.snapshot_loads == 0U;
}

static int qcs_transact(
    struct qcs_context *context,
    size_t variant,
    struct qcs_outcome *outcome
) {
    if (
        context == NULL
        || outcome == NULL
        || variant >= QCS_VARIANTS
        || context->state != QCS_READY
        || context->carrier_creation_count != 1U
        || memcmp(
            context->program,
            context->sealed_program,
            sizeof(context->program)
        ) != 0
        || qcs_program_hash(context->program)
            != context->program_custody_hash
        || qb_restoration_error(
            &context->carrier,
            &context->sealed_state
        ) > QB_RESTORATION_TOLERANCE
        || memcmp(
            context->carrier.baseline,
            context->sealed_state.baseline,
            sizeof(context->carrier.baseline)
        ) != 0
    ) {
        return 0;
    }
#ifdef CATVM_QANF_TESTING
    if (context->testing_inert) {
        ++context->completed_transactions;
        return 3;
    }
    const enum qb_inverse_mode mode = context->testing_mode;
#else
    const enum qb_inverse_mode mode = QB_INVERSE_CORRECT;
#endif
    context->state = QCS_RUNNING;
    const struct qb_execution execution = qb_execute(
        &context->carrier,
        &context->program[variant],
        mode
    );
    const double sealed_error = qb_restoration_error(
        &context->carrier,
        &context->sealed_state
    );

#ifdef CATVM_QANF_TESTING
    if (
        mode == QB_INVERSE_WRONG_Z
        || mode == QB_INVERSE_MISSING_Z
        || mode == QB_INVERSE_REORDERED
    ) {
        context->state = QCS_FAILED;
        return (
            execution.restoration_max_abs >= QB_CONTROL_MINIMUM_ERROR
            && sealed_error >= QB_CONTROL_MINIMUM_ERROR
        ) ? -1 : 0;
    }
    if (mode == QB_INVERSE_SNAPSHOT) {
        if (
            !execution.snapshot_loaded
            || execution.actual_inverse
            || execution.stats.snapshot_loads != 1U
            || execution.restoration_max_abs != 0.0
            || sealed_error > QB_RESTORATION_TOLERANCE
        ) {
            context->state = QCS_FAILED;
            return 0;
        }
        outcome->boundary = execution.boundary;
        outcome->boundary_hash =
            qcs_boundary_hash(&execution.boundary);
        outcome->plan_hash = qb_plan_hash();
        outcome->restoration_generation =
            context->restoration_generation;
        outcome->carrier_creation_count =
            context->carrier_creation_count;
        ++context->completed_transactions;
        context->state = QCS_READY;
        return 2;
    }
#endif

    if (
        mode != QB_INVERSE_CORRECT
        || !execution.actual_inverse
        || execution.snapshot_loaded
        || execution.restoration_max_abs
            > QB_RESTORATION_TOLERANCE
        || sealed_error > QB_RESTORATION_TOLERANCE
        || execution.carrier_integrity_max_abs
            > QB_RESTORATION_TOLERANCE
        || execution.boundary.maximum_root_error
            > QB_ROOT_TOLERANCE
        || !qcs_stats_are_accepted(&execution)
        || memcmp(
            context->program,
            context->sealed_program,
            sizeof(context->program)
        ) != 0
        || qcs_program_hash(context->program)
            != context->program_custody_hash
    ) {
        context->state = QCS_FAILED;
        return 0;
    }
    ++context->restoration_generation;
    outcome->boundary = execution.boundary;
    outcome->boundary_hash =
        qcs_boundary_hash(&execution.boundary);
    outcome->plan_hash = qb_plan_hash();
    outcome->restoration_generation =
        context->restoration_generation;
    outcome->carrier_creation_count =
        context->carrier_creation_count;
    ++context->completed_transactions;
    context->state = QCS_READY;
    return 1;
}

static int qcs_make_listener(const char *path) {
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

static int qcs_peer_is_same_real_user(int client) {
    struct ucred credential;
    socklen_t size = sizeof(credential);
    return (
        getsockopt(
            client,
            SOL_SOCKET,
            SO_PEERCRED,
            &credential,
            &size
        ) == 0
        && size == sizeof(credential)
        && credential.uid == getuid()
    );
}

static int qcs_install_seccomp(int client) {
#ifdef CATVM_SANITIZER_BUILD
    (void)client;
    return 1;
#else
    scmp_filter_ctx filter = seccomp_init(SCMP_ACT_KILL_PROCESS);
    if (filter == NULL) {
        return 0;
    }
    int ok = 1;
#define QCS_ALLOW_SYSCALL(name) \
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
    QCS_ALLOW_SYSCALL(brk);
    QCS_ALLOW_SYSCALL(close);
    QCS_ALLOW_SYSCALL(exit);
    QCS_ALLOW_SYSCALL(exit_group);
    QCS_ALLOW_SYSCALL(madvise);
    QCS_ALLOW_SYSCALL(mmap);
    QCS_ALLOW_SYSCALL(mprotect);
    QCS_ALLOW_SYSCALL(mremap);
    QCS_ALLOW_SYSCALL(munlock);
    QCS_ALLOW_SYSCALL(munmap);
    QCS_ALLOW_SYSCALL(rt_sigreturn);
#undef QCS_ALLOW_SYSCALL
    if (!ok || seccomp_load(filter) != 0) {
        seccomp_release(filter);
        return 0;
    }
    return 1;
#endif
}

static int qcs_send_response(int client, const char *response) {
    const size_t bytes = strlen(response);
    if (bytes == 0U || bytes >= QCS_RESPONSE_CAPACITY) {
        return 0;
    }
    return send(client, response, bytes, MSG_NOSIGNAL)
        == (ssize_t)bytes;
}

static int qcs_append(
    char response[QCS_RESPONSE_CAPACITY],
    size_t *used,
    const char *format,
    ...
) {
    if (*used >= QCS_RESPONSE_CAPACITY) {
        return 0;
    }
    va_list arguments;
    va_start(arguments, format);
    const int written = vsnprintf(
        response + *used,
        QCS_RESPONSE_CAPACITY - *used,
        format,
        arguments
    );
    va_end(arguments);
    if (
        written < 0
        || (size_t)written >= QCS_RESPONSE_CAPACITY - *used
    ) {
        return 0;
    }
    *used += (size_t)written;
    return 1;
}

static int qcs_format_outcome(
    char response[QCS_RESPONSE_CAPACITY],
    size_t variant,
    const struct qcs_outcome *outcome,
    int snapshot
) {
    size_t used = 0U;
    if (!qcs_append(
        response,
        &used,
        "OK %s %zu %llu %016llx %016llx %llu %u",
        snapshot ? "SNAPSHOT" : "FINAL",
        variant,
        (unsigned long long)outcome->restoration_generation,
        (unsigned long long)outcome->plan_hash,
        (unsigned long long)outcome->boundary_hash,
        (unsigned long long)outcome->carrier_creation_count,
        QB_Z_COEFFICIENTS
    )) {
        return 0;
    }
    for (size_t cell = 0U; cell < QB_Z_COEFFICIENTS; ++cell) {
        if (!qcs_append(
            response,
            &used,
            " %d",
            outcome->boundary.coefficient[cell]
        )) {
            return 0;
        }
    }
    return 1;
}

static int qcs_receive_request(
    int client,
    char request[QCS_REQUEST_CAPACITY]
) {
    const ssize_t received = recv(
        client,
        request,
        QCS_REQUEST_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (received <= 0) {
        return 0;
    }
    if (
        (size_t)received >= QCS_REQUEST_CAPACITY
        || memchr(request, '\0', (size_t)received) != NULL
    ) {
        qcs_secure_zero(request, QCS_REQUEST_CAPACITY);
        return -1;
    }
    request[received] = '\0';
    return 1;
}

static int qcs_projection_request(const char *request) {
    return
        strncmp(request, "PROJECT ", 8U) == 0
        || strcmp(request, "DEBUG") == 0
        || strcmp(request, "DUMP") == 0
        || strcmp(request, "READ CARRIER") == 0
        || strcmp(request, "STATE DETAIL") == 0;
}

static int qcs_serve(
    int client,
    struct qcs_context *context
) {
    char request[QCS_REQUEST_CAPACITY] = {0};
    char response[QCS_RESPONSE_CAPACITY] = {0};
    struct qcs_outcome outcome = {0};
    int keep_running = 1;
    while (keep_running) {
        const int received = qcs_receive_request(client, request);
        if (received == 0) {
            break;
        }
        if (received < 0) {
            if (!qcs_send_response(client, "ERR E_PROTOCOL")) {
                break;
            }
            continue;
        }
        if (strcmp(request, "HELLO") == 0) {
            const int written = snprintf(
                response,
                sizeof(response),
                "OK HELLO %s %016llx 23 18 97 46 5 2 1",
                QCS_PROTOCOL,
                (unsigned long long)qb_plan_hash()
            );
            if (
                written <= 0
                || (size_t)written >= sizeof(response)
                || !qcs_send_response(client, response)
            ) {
                break;
            }
        } else if (qcs_projection_request(request)) {
            if (!qcs_send_response(
                client,
                "ERR E_INTERMEDIATE_PROJECTION_DENIED"
            )) {
                break;
            }
        } else if (
            strlen(request) == 9U
            && strncmp(request, "EXECUTE ", 8U) == 0
            && request[8] >= '0'
            && request[8] <= '3'
        ) {
            const size_t variant = (size_t)(request[8] - '0');
            qcs_secure_zero(&outcome, sizeof(outcome));
            const int transaction =
                qcs_transact(context, variant, &outcome);
            if (transaction < 0) {
                (void)qcs_send_response(
                    client,
                    "ERR E_RESTORATION_DETECTED"
                );
                break;
            }
            if (transaction == 0) {
                (void)qcs_send_response(client, "ERR E_MACHINE_LAW");
                break;
            }
            if (transaction == 3) {
                const int written = snprintf(
                    response,
                    sizeof(response),
                    "OK INERT %llu",
                    (unsigned long long)
                        context->completed_transactions
                );
                if (
                    written <= 0
                    || (size_t)written >= sizeof(response)
                    || !qcs_send_response(client, response)
                ) {
                    break;
                }
                continue;
            }
            qcs_secure_zero(response, sizeof(response));
            if (
                !qcs_format_outcome(
                    response,
                    variant,
                    &outcome,
                    transaction == 2
                )
                || !qcs_send_response(client, response)
            ) {
                break;
            }
        } else if (strcmp(request, "SHUTDOWN") == 0) {
            if (!qcs_send_response(client, "OK CLOSED")) {
                break;
            }
            keep_running = 0;
        } else if (!qcs_send_response(client, "ERR E_PROTOCOL")) {
            break;
        }
        qcs_secure_zero(request, sizeof(request));
        qcs_secure_zero(response, sizeof(response));
        qcs_secure_zero(&outcome, sizeof(outcome));
    }
    qcs_secure_zero(request, sizeof(request));
    qcs_secure_zero(response, sizeof(response));
    qcs_secure_zero(&outcome, sizeof(outcome));
    return 1;
}

int main(int argc, char **argv) {
#ifdef CATVM_QANF_SIZE_PROBE
    if (argc == 2 && strcmp(argv[1], "--size-probe") == 0) {
        const long page_size_long = sysconf(_SC_PAGESIZE);
        if (page_size_long <= 0) {
            return 2;
        }
        printf(
            "{\"context_bytes\":%zu,"
            "\"mapped_context_bytes\":%zu,"
            "\"program_table_bytes\":%zu,"
            "\"sealed_program_table_bytes\":%zu,"
            "\"carrier_cells\":%u,"
            "\"live_carrier_bytes\":%zu,"
            "\"sealed_verification_state_bytes\":%zu,"
            "\"execution_snapshot_bytes\":%zu,"
            "\"execution_summary_bytes\":%zu,"
            "\"request_buffer_bytes\":%u,"
            "\"response_buffer_bytes\":%u}\n",
            sizeof(struct qcs_context),
            qcs_page_round(
                sizeof(struct qcs_context),
                (size_t)page_size_long
            ),
            sizeof(((struct qcs_context *)0)->program),
            sizeof(((struct qcs_context *)0)->sealed_program),
            QB_CARRIER_CELLS,
            sizeof(struct qb_carrier),
            sizeof(struct qb_carrier),
            sizeof(struct qb_carrier),
            sizeof(struct qb_execution),
            QCS_REQUEST_CAPACITY,
            QCS_RESPONSE_CAPACITY
        );
        return 0;
    }
#endif
#ifdef CATVM_QANF_TESTING
    enum qb_inverse_mode testing_mode = QB_INVERSE_CORRECT;
    int testing_inert = 0;
    if (argc != 7) {
        return 2;
    }
    if (strcmp(argv[6], "correct") == 0) {
        testing_mode = QB_INVERSE_CORRECT;
    } else if (strcmp(argv[6], "wrong-z") == 0) {
        testing_mode = QB_INVERSE_WRONG_Z;
    } else if (strcmp(argv[6], "missing-z") == 0) {
        testing_mode = QB_INVERSE_MISSING_Z;
    } else if (strcmp(argv[6], "reordered") == 0) {
        testing_mode = QB_INVERSE_REORDERED;
    } else if (strcmp(argv[6], "snapshot") == 0) {
        testing_mode = QB_INVERSE_SNAPSHOT;
    } else if (strcmp(argv[6], "inert") == 0) {
        testing_inert = 1;
    } else {
        return 2;
    }
#else
    if (argc != 6) {
        return 2;
    }
#endif
    if (
        !qcs_process_is_untraced()
        || !qcs_establish_process_guards()
    ) {
        return 2;
    }
    (void)umask(S_IRWXG | S_IRWXO);
    const char *const program_path[QCS_VARIANTS] = {
        argv[2], argv[3], argv[4], argv[5]
    };
    size_t mapped_bytes = 0U;
    struct qcs_context *context = qcs_create(
        program_path,
        &mapped_bytes
#ifdef CATVM_QANF_TESTING
        ,
        testing_mode,
        testing_inert
#endif
    );
    if (context == NULL) {
        return 2;
    }
    const int listener = qcs_make_listener(argv[1]);
    if (listener < 0) {
        qcs_destroy(context, mapped_bytes);
        return 2;
    }
    const int client = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0 || !qcs_peer_is_same_real_user(client)) {
        if (client >= 0) {
            (void)close(client);
        }
        (void)close(listener);
        (void)unlink(argv[1]);
        qcs_destroy(context, mapped_bytes);
        return 2;
    }
    (void)close(listener);
    if (unlink(argv[1]) != 0) {
        (void)close(client);
        qcs_destroy(context, mapped_bytes);
        return 2;
    }
    if (!qcs_install_seccomp(client)) {
        (void)close(client);
        qcs_destroy(context, mapped_bytes);
        return 2;
    }
    (void)qcs_serve(client, context);
    (void)close(client);
    qcs_destroy(context, mapped_bytes);
    return 0;
}
