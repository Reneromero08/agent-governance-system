#define _GNU_SOURCE

/*
 * Minimal CATVM boundary for the reviewed rank-two affine-DAG scheduler.
 *
 * The reviewed implementation is embedded privately and its standalone main
 * and report functions are discarded by the production link.  This service
 * exposes only an atomic public-variant transaction.  The actual carrier,
 * compiled plan, unresolved relations, activation receipts, obligations, and
 * inverse execution remain inside this non-dumpable process.
 */

#define RR_REUSE_CYCLES 0U
#define RR_PUBLIC_MAIN catvm_rank2_reviewed_standalone_main
#include "recursive_rematerializing_general_multi_dag_affine_phase.c"
#undef RR_PUBLIC_MAIN

#include <errno.h>
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

#define CDG_PROTOCOL "CATVM_RANK2_PHASE_1"
#define CDG_REQUEST_CAPACITY 128U
#define CDG_RESPONSE_CAPACITY 1024U
#define CDG_CARRIER_ID 9323

enum cdg_state {
    CDG_READY = 1,
    CDG_RUNNING = 2,
    CDG_FAILED = 3
};

struct cdg_context {
    struct dc_compiled compiled;
    struct rr_plan plan;
    struct rc_program program[GM_VARIANTS];
    struct carrier carrier;
    struct rr_machine machine;
    enum cdg_state state;
    uint64_t carrier_creation_count;
    uint64_t completed_transactions;
#ifdef CATVM_RANK2_TESTING
    enum rc_restore_mode testing_restore_mode;
    int testing_reordered;
    int testing_inert;
#endif
};

struct cdg_outcome {
    struct ga_boundary boundary;
    uint64_t plan_hash;
    uint64_t topology_hash;
    uint64_t restoration_generation;
    uint64_t carrier_creation_count;
    size_t boundary_cells;
};

static void cdg_secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static size_t cdg_page_round(size_t bytes, size_t page_size) {
    if (bytes > SIZE_MAX - page_size + 1U) {
        return 0U;
    }
    return ((bytes + page_size - 1U) / page_size) * page_size;
}

static int cdg_protect_range(void *memory, size_t bytes) {
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (
        memory == NULL
        || bytes == 0U
        || page_size_long <= 0
    ) {
        return 0;
    }
    const size_t page_size = (size_t)page_size_long;
    const uintptr_t raw = (uintptr_t)memory;
    const uintptr_t begin = raw - raw % page_size;
    if (bytes > UINTPTR_MAX - raw) {
        return 0;
    }
    const uintptr_t raw_end = raw + bytes;
    if (raw_end > UINTPTR_MAX - page_size + 1U) {
        return 0;
    }
    const uintptr_t end =
        ((raw_end + page_size - 1U) / page_size) * page_size;
    const size_t protected_bytes = (size_t)(end - begin);
    return (
        mlock((void *)begin, protected_bytes) == 0
        && madvise(
            (void *)begin,
            protected_bytes,
            MADV_DONTDUMP
        ) == 0
        && madvise(
            (void *)begin,
            protected_bytes,
            MADV_DONTFORK
        ) == 0
    );
}

static int cdg_process_is_untraced(void) {
#ifdef CATVM_RANK2_TRACE_BUILD
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

static int cdg_establish_process_guards(void) {
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

static struct cdg_context *cdg_create(
    const char *manifest_path,
    size_t *mapped_bytes
#ifdef CATVM_RANK2_TESTING
    ,
    enum rc_restore_mode testing_restore_mode,
    int testing_reordered,
    int testing_inert
#endif
) {
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        return NULL;
    }
    const size_t page_size = (size_t)page_size_long;
    *mapped_bytes = cdg_page_round(
        sizeof(struct cdg_context),
        page_size
    );
    if (*mapped_bytes == 0U) {
        return NULL;
    }
    struct cdg_context *context = mmap(
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

    struct rc_manifest manifest = {0};
    rc_load_manifest(manifest_path, &manifest);
    context->compiled = dc_compile(&manifest, 4U);
    gm_require_topology(&context->compiled);
    context->plan = rr_compile_plan(&context->compiled);
    for (size_t variant = 0U; variant < GM_VARIANTS; ++variant) {
        context->program[variant] = gm_make_program(variant);
    }
    const struct process shape = {
        .carrier_cells = rc_carrier_cells(RR_WORKING_SLOTS)
    };
    context->carrier = make_carrier(&shape, CDG_CARRIER_ID);
    const size_t carrier_bytes =
        context->carrier.cells * sizeof(*context->carrier.working);
    if (
        !cdg_protect_range(
            context->carrier.baseline,
            carrier_bytes
        )
        || !cdg_protect_range(
            context->carrier.working,
            carrier_bytes
        )
#ifndef CATVM_SANITIZER_BUILD
        || mlockall(MCL_CURRENT | MCL_FUTURE) != 0
#endif
    ) {
        free_carrier(&context->carrier);
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
    context->state = CDG_READY;
    context->carrier_creation_count = 1U;
#ifdef CATVM_RANK2_TESTING
    context->testing_restore_mode = testing_restore_mode;
    context->testing_reordered = testing_reordered;
    context->testing_inert = testing_inert;
#endif
    return context;
}

static void cdg_destroy(
    struct cdg_context *context,
    size_t mapped_bytes
) {
    if (context == NULL) {
        return;
    }
    if (context->carrier.baseline != NULL) {
        cdg_secure_zero(
            context->carrier.baseline,
            context->carrier.cells
                * sizeof(*context->carrier.baseline)
        );
    }
    if (context->carrier.working != NULL) {
        cdg_secure_zero(
            context->carrier.working,
            context->carrier.cells
                * sizeof(*context->carrier.working)
        );
    }
    free_carrier(&context->carrier);
    cdg_secure_zero(context, sizeof(*context));
    (void)munlock(context, mapped_bytes);
    (void)munmap(context, mapped_bytes);
}

static int cdg_transact(
    struct cdg_context *context,
    size_t variant,
    struct cdg_outcome *outcome
) {
    if (
        context == NULL
        || outcome == NULL
        || variant >= GM_VARIANTS
        || context->state != CDG_READY
        || context->carrier_creation_count != 1U
    ) {
        return 0;
    }
#ifdef CATVM_RANK2_TESTING
    if (context->testing_inert) {
        ++context->completed_transactions;
        return 3;
    }
#endif
    context->state = CDG_RUNNING;
#ifdef CATVM_RANK2_TESTING
    struct rr_plan reordered_plan;
    const struct rr_plan *selected_plan = &context->plan;
    if (context->testing_reordered) {
        reordered_plan = context->plan;
        const size_t root = context->compiled.graph.root;
        const size_t child =
            context->compiled.graph.node[root].left;
        size_t root_inverse = SIZE_MAX;
        size_t child_inverse = SIZE_MAX;
        for (
            size_t action = 0U;
            action < reordered_plan.reverse_count;
            ++action
        ) {
            if (
                reordered_plan.reverse[action].opcode
                    == RR_OPERATOR_INVERSE
                && reordered_plan.reverse[action].node == root
            ) {
                root_inverse = action;
            }
            if (
                reordered_plan.reverse[action].opcode
                    == RR_OPERATOR_INVERSE
                && reordered_plan.reverse[action].node == child
            ) {
                child_inverse = action;
                break;
            }
        }
        if (
            root_inverse == SIZE_MAX
            || child_inverse == SIZE_MAX
            || root_inverse >= child_inverse
        ) {
            context->state = CDG_FAILED;
            return 0;
        }
        const struct rr_action swap =
            reordered_plan.reverse[root_inverse];
        reordered_plan.reverse[root_inverse] =
            reordered_plan.reverse[child_inverse];
        reordered_plan.reverse[child_inverse] = swap;
        selected_plan = &reordered_plan;
    }
#else
    const struct rr_plan *const selected_plan = &context->plan;
#endif
    const struct rr_execution execution = rr_execute(
        &context->carrier,
        &context->compiled,
        selected_plan,
        &context->program[variant],
        &context->machine,
#ifdef CATVM_RANK2_TESTING
        context->testing_restore_mode,
#else
        RC_RESTORE_CORRECT,
#endif
        RR_FAULT_NONE
    );
#ifdef CATVM_RANK2_TESTING
    if (
        context->testing_restore_mode == RC_RESTORE_WRONG_ROOT
        || context->testing_restore_mode == RC_RESTORE_MISSING_ROOT
    ) {
        context->state = CDG_FAILED;
        return execution.restoration_max_abs > 1e-6 ? -1 : 0;
    }
    if (context->testing_restore_mode == RC_RESTORE_SNAPSHOT) {
        if (
            !execution.snapshot_loaded
            || execution.reverse_cursor != 0U
            || execution.restoration_max_abs != 0.0
            || execution.restoration_generation_after
                != execution.restoration_generation_before
            || !execution.workspace_cleared
        ) {
            context->state = CDG_FAILED;
            return 0;
        }
        outcome->boundary = execution.boundary;
        outcome->plan_hash = context->plan.hash;
        outcome->topology_hash = context->plan.topology_hash;
        outcome->restoration_generation =
            context->machine.restoration_generation;
        outcome->carrier_creation_count =
            context->carrier_creation_count;
        outcome->boundary_cells = GA_BLOCK_CELLS;
        ++context->completed_transactions;
        context->state = CDG_READY;
        return 2;
    }
#endif
    if (
        !rr_execution_exact(&context->compiled, &execution)
        || execution.snapshot_loaded
        || execution.intermediate_block_copy_calls != 0U
        || execution.boundary_block_copy_calls != 2U
        || execution.restoration_generation_after
            != execution.restoration_generation_before + 1U
        || context->machine.restoration_generation
            != execution.restoration_generation_after
    ) {
        context->state = CDG_FAILED;
        return 0;
    }
    outcome->boundary = execution.boundary;
    outcome->plan_hash = context->plan.hash;
    outcome->topology_hash = context->plan.topology_hash;
    outcome->restoration_generation =
        context->machine.restoration_generation;
    outcome->carrier_creation_count =
        context->carrier_creation_count;
    outcome->boundary_cells = GA_BLOCK_CELLS;
    ++context->completed_transactions;
    context->state = CDG_READY;
    return 1;
}

static int cdg_make_listener(const char *path) {
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

static int cdg_peer_is_same_real_user(int client) {
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

static int cdg_install_seccomp(int client) {
#ifdef CATVM_SANITIZER_BUILD
    (void)client;
    return 1;
#else
    scmp_filter_ctx filter = seccomp_init(SCMP_ACT_KILL_PROCESS);
    if (filter == NULL) {
        return 0;
    }
    int ok = 1;
#define CDG_ALLOW_SYSCALL(name) \
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
    CDG_ALLOW_SYSCALL(brk);
    CDG_ALLOW_SYSCALL(close);
    CDG_ALLOW_SYSCALL(exit);
    CDG_ALLOW_SYSCALL(exit_group);
    CDG_ALLOW_SYSCALL(madvise);
    CDG_ALLOW_SYSCALL(mmap);
    CDG_ALLOW_SYSCALL(mprotect);
    CDG_ALLOW_SYSCALL(mremap);
    CDG_ALLOW_SYSCALL(munlock);
    CDG_ALLOW_SYSCALL(munmap);
    CDG_ALLOW_SYSCALL(rt_sigreturn);
#undef CDG_ALLOW_SYSCALL
    if (!ok || seccomp_load(filter) != 0) {
        seccomp_release(filter);
        return 0;
    }
    return 1;
#endif
}

static int cdg_send_response(int client, const char *response) {
    const size_t bytes = strlen(response);
    if (bytes == 0U || bytes >= CDG_RESPONSE_CAPACITY) {
        return 0;
    }
    return send(client, response, bytes, MSG_NOSIGNAL)
        == (ssize_t)bytes;
}

static int cdg_append(
    char response[CDG_RESPONSE_CAPACITY],
    size_t *used,
    const char *format,
    ...
) {
    if (*used >= CDG_RESPONSE_CAPACITY) {
        return 0;
    }
    va_list arguments;
    va_start(arguments, format);
    const int written = vsnprintf(
        response + *used,
        CDG_RESPONSE_CAPACITY - *used,
        format,
        arguments
    );
    va_end(arguments);
    if (
        written < 0
        || (size_t)written >= CDG_RESPONSE_CAPACITY - *used
    ) {
        return 0;
    }
    *used += (size_t)written;
    return 1;
}

static int cdg_format_outcome(
    char response[CDG_RESPONSE_CAPACITY],
    size_t variant,
    const struct cdg_outcome *outcome,
    int snapshot
) {
    size_t used = 0U;
    if (!cdg_append(
        response,
        &used,
        "OK %s %zu %llu %016llx %016llx %llu %zu",
        snapshot ? "SNAPSHOT" : "FINAL",
        variant,
        (unsigned long long)outcome->restoration_generation,
        (unsigned long long)outcome->plan_hash,
        (unsigned long long)outcome->boundary.hash,
        (unsigned long long)outcome->carrier_creation_count,
        outcome->boundary_cells
    )) {
        return 0;
    }
    for (size_t cell = 0U; cell < outcome->boundary_cells; ++cell) {
        if (!cdg_append(
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

static int cdg_receive_request(
    int client,
    char request[CDG_REQUEST_CAPACITY]
) {
    const ssize_t received = recv(
        client,
        request,
        CDG_REQUEST_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (received <= 0) {
        return 0;
    }
    if (
        (size_t)received >= CDG_REQUEST_CAPACITY
        || memchr(request, '\0', (size_t)received) != NULL
    ) {
        cdg_secure_zero(request, CDG_REQUEST_CAPACITY);
        return -1;
    }
    request[received] = '\0';
    return 1;
}

static int cdg_projection_request(const char *request) {
    return (
        strncmp(request, "PROJECT ", 8U) == 0
        || strcmp(request, "DEBUG") == 0
        || strcmp(request, "DUMP") == 0
        || strcmp(request, "READ CARRIER") == 0
        || strcmp(request, "STATE DETAIL") == 0
    );
}

static int cdg_serve(
    int client,
    struct cdg_context *context
) {
    char request[CDG_REQUEST_CAPACITY] = {0};
    char response[CDG_RESPONSE_CAPACITY] = {0};
    struct cdg_outcome outcome = {0};
    int keep_running = 1;
    while (keep_running) {
        const int received = cdg_receive_request(client, request);
        if (received == 0) {
            break;
        }
        if (received < 0) {
            if (!cdg_send_response(client, "ERR E_PROTOCOL")) {
                break;
            }
            continue;
        }
        if (strcmp(request, "HELLO") == 0) {
            const int written = snprintf(
                response,
                sizeof(response),
                "OK HELLO %s %016llx %016llx 15 22 28 28 9 1",
                CDG_PROTOCOL,
                (unsigned long long)context->plan.hash,
                (unsigned long long)context->plan.topology_hash
            );
            if (
                written <= 0
                || (size_t)written >= sizeof(response)
                || !cdg_send_response(client, response)
            ) {
                break;
            }
        } else if (cdg_projection_request(request)) {
            if (!cdg_send_response(
                client,
                "ERR E_INTERMEDIATE_PROJECTION_DENIED"
            )) {
                break;
            }
        } else if (
            strlen(request) == 9U
            && strncmp(request, "EXECUTE ", 8U) == 0
            && request[8] >= '0'
            && request[8] <= '4'
        ) {
            const size_t variant = (size_t)(request[8] - '0');
            cdg_secure_zero(&outcome, sizeof(outcome));
            const int transaction = cdg_transact(
                context,
                variant,
                &outcome
            );
            if (transaction < 0) {
                (void)cdg_send_response(
                    client,
                    "ERR E_RESTORATION_DETECTED"
                );
                break;
            }
            if (transaction == 0) {
                (void)cdg_send_response(
                    client,
                    "ERR E_MACHINE_LAW"
                );
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
                    || !cdg_send_response(client, response)
                ) {
                    break;
                }
                continue;
            }
            cdg_secure_zero(response, sizeof(response));
            if (
                !cdg_format_outcome(
                    response,
                    variant,
                    &outcome,
                    transaction == 2
                )
                || !cdg_send_response(client, response)
            ) {
                break;
            }
        } else if (strcmp(request, "SHUTDOWN") == 0) {
            if (!cdg_send_response(client, "OK CLOSED")) {
                break;
            }
            keep_running = 0;
        } else {
            if (!cdg_send_response(client, "ERR E_PROTOCOL")) {
                break;
            }
        }
        cdg_secure_zero(request, sizeof(request));
        cdg_secure_zero(response, sizeof(response));
        cdg_secure_zero(&outcome, sizeof(outcome));
    }
    cdg_secure_zero(request, sizeof(request));
    cdg_secure_zero(response, sizeof(response));
    cdg_secure_zero(&outcome, sizeof(outcome));
    return 1;
}

int main(int argc, char **argv) {
#ifdef CATVM_RANK2_SIZE_PROBE
    if (argc == 2 && strcmp(argv[1], "--size-probe") == 0) {
        printf(
            "{\"context_bytes\":%zu,"
            "\"compiled_topology_bytes\":%zu,"
            "\"plan_bytes\":%zu,"
            "\"program_table_bytes\":%zu,"
            "\"machine_counter_bytes\":%zu,"
            "\"execution_summary_bytes\":%zu,"
            "\"carrier_cells\":%zu,"
            "\"live_carrier_bytes\":%zu,"
            "\"verification_snapshot_bytes\":%zu,"
            "\"request_buffer_bytes\":%u,"
            "\"response_buffer_bytes\":%u}\n",
            sizeof(struct cdg_context),
            sizeof(struct dc_compiled),
            sizeof(struct rr_plan),
            sizeof(((struct cdg_context *)0)->program),
            sizeof(struct rr_machine),
            sizeof(struct rr_execution),
            rc_carrier_cells(RR_WORKING_SLOTS),
            2U * rc_carrier_cells(RR_WORKING_SLOTS)
                * sizeof(double complex),
            2U * rc_carrier_cells(RR_WORKING_SLOTS)
                * sizeof(double complex),
            CDG_REQUEST_CAPACITY,
            CDG_RESPONSE_CAPACITY
        );
        return 0;
    }
#endif
#ifdef CATVM_RANK2_TESTING
    enum rc_restore_mode testing_restore_mode = RC_RESTORE_CORRECT;
    int testing_reordered = 0;
    int testing_inert = 0;
    if (argc != 4) {
        return 2;
    }
    if (strcmp(argv[3], "correct") == 0) {
        testing_restore_mode = RC_RESTORE_CORRECT;
    } else if (strcmp(argv[3], "wrong-root") == 0) {
        testing_restore_mode = RC_RESTORE_WRONG_ROOT;
    } else if (strcmp(argv[3], "missing-root") == 0) {
        testing_restore_mode = RC_RESTORE_MISSING_ROOT;
    } else if (strcmp(argv[3], "snapshot") == 0) {
        testing_restore_mode = RC_RESTORE_SNAPSHOT;
    } else if (strcmp(argv[3], "reordered") == 0) {
        testing_reordered = 1;
    } else if (strcmp(argv[3], "inert") == 0) {
        testing_inert = 1;
    } else {
        return 2;
    }
#else
    if (
        argc != 3
    ) {
        return 2;
    }
#endif
    if (
        !cdg_process_is_untraced()
        || !cdg_establish_process_guards()
    ) {
        return 2;
    }
    (void)umask(S_IRWXG | S_IRWXO);

    size_t mapped_bytes = 0U;
    struct cdg_context *context = cdg_create(
        argv[2],
        &mapped_bytes
#ifdef CATVM_RANK2_TESTING
        ,
        testing_restore_mode,
        testing_reordered,
        testing_inert
#endif
    );
    if (context == NULL) {
        return 2;
    }
    const int listener = cdg_make_listener(argv[1]);
    if (listener < 0) {
        cdg_destroy(context, mapped_bytes);
        return 2;
    }
    const int client = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0 || !cdg_peer_is_same_real_user(client)) {
        if (client >= 0) {
            (void)close(client);
        }
        (void)close(listener);
        (void)unlink(argv[1]);
        cdg_destroy(context, mapped_bytes);
        return 2;
    }
    (void)close(listener);
    if (unlink(argv[1]) != 0) {
        (void)close(client);
        cdg_destroy(context, mapped_bytes);
        return 2;
    }
    if (!cdg_install_seccomp(client)) {
        (void)close(client);
        cdg_destroy(context, mapped_bytes);
        return 2;
    }

    (void)cdg_serve(client, context);
    (void)close(client);
    cdg_destroy(context, mapped_bytes);
    return 0;
}
