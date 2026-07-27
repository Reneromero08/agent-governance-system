#define _GNU_SOURCE

/*
 * Minimal CATVM custody boundary for width-parametric Boolean relation TTs.
 *
 * The service owns the only phase carrier.  H and Z remain service-local;
 * production returns a final rank-eight boundary receipt only after actual
 * Z^-1, H^-1, leaf reversal, restoration, and generation advancement.
 */

#define ALGEBRAIC_BOOLEAN_TT_NO_MAIN 1
#include "algebraic_boolean_tt_phase.c"

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

#define CBTS_PROTOCOL "CATVM_BOOLEAN_TT_PHASE_1"
#define CBTS_VARIANTS 2U
#define CBTS_REQUEST_CAPACITY 128U
#define CBTS_RESPONSE_CAPACITY 512U
#define CBTS_CARRIER_ID 79

enum cbts_state {
    CBTS_READY = 1,
    CBTS_RUNNING = 2,
    CBTS_FAILED = 3
};

struct cbts_context {
    struct btt_layout layout;
    struct btt_layout sealed_layout;
    struct carrier carrier;
    struct carrier sealed_state;
    uint64_t program_hash[CBTS_VARIANTS];
    uint64_t sealed_program_hash[CBTS_VARIANTS];
    enum cbts_state state;
    uint64_t restoration_generation;
    uint64_t carrier_creation_count;
    uint64_t completed_transactions;
#ifdef CATVM_BOOLEAN_TT_TESTING
    enum btt_mode testing_mode;
    int testing_inert;
#endif
};

struct cbts_outcome {
    uint64_t boundary_hash;
    uint64_t plan_hash;
    uint64_t restoration_generation;
    uint64_t carrier_creation_count;
    size_t boundary_ones;
    size_t boundary_cells;
};

static void cbts_secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static size_t cbts_page_round(size_t bytes, size_t page_size) {
    if (bytes > SIZE_MAX - page_size + 1U) {
        return 0U;
    }
    return ((bytes + page_size - 1U) / page_size) * page_size;
}

static uint64_t cbts_plan_hash(const struct cbts_context *context) {
    uint64_t hash = UINT64_C(14695981039346656037);
    const uint64_t word[] = {
        (uint64_t)context->layout.width,
        (uint64_t)context->layout.n2,
        (uint64_t)context->layout.n4,
        (uint64_t)context->layout.n8,
        (uint64_t)context->layout.carrier_cells,
        context->program_hash[0],
        context->program_hash[1]
    };
    return hash_bytes(
        hash,
        (const unsigned char *)word,
        sizeof(word)
    );
}

static int cbts_process_is_untraced(void) {
#ifdef CATVM_BOOLEAN_TT_TRACE_BUILD
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

static int cbts_establish_process_guards(void) {
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

static struct cbts_context *cbts_create(
    size_t width,
    size_t *mapped_bytes
#ifdef CATVM_BOOLEAN_TT_TESTING
    ,
    enum btt_mode testing_mode,
    int testing_inert
#endif
) {
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        return NULL;
    }
    *mapped_bytes = cbts_page_round(
        sizeof(struct cbts_context), (size_t)page_size_long
    );
    if (*mapped_bytes == 0U) {
        return NULL;
    }
    struct cbts_context *context = mmap(
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
    context->layout = btt_make_layout(width);
    context->sealed_layout = context->layout;
    const struct process process =
        btt_carrier_process(&context->layout);
    context->carrier = make_carrier(&process, CBTS_CARRIER_ID);
    context->sealed_state = snapshot_carrier(&context->carrier);
    context->program_hash[BTT_NEIGHBOR_AND] = btt_program_hash(
        &context->layout, BTT_NEIGHBOR_AND
    );
    context->program_hash[BTT_NEIGHBOR_NAND] = btt_program_hash(
        &context->layout, BTT_NEIGHBOR_NAND
    );
    memcpy(
        context->sealed_program_hash,
        context->program_hash,
        sizeof(context->program_hash)
    );
    context->state = CBTS_READY;
    context->carrier_creation_count = 1U;
#ifdef CATVM_BOOLEAN_TT_TESTING
    context->testing_mode = testing_mode;
    context->testing_inert = testing_inert;
#endif
#ifndef CATVM_SANITIZER_BUILD
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        free_carrier(&context->sealed_state);
        free_carrier(&context->carrier);
        cbts_secure_zero(context, sizeof(*context));
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
#endif
    return context;
}

static void cbts_destroy(
    struct cbts_context *context,
    size_t mapped_bytes
) {
    if (context == NULL) {
        return;
    }
    free_carrier(&context->sealed_state);
    free_carrier(&context->carrier);
    cbts_secure_zero(context, sizeof(*context));
    (void)munlock(context, mapped_bytes);
    (void)munmap(context, mapped_bytes);
}

static int cbts_stats_are_accepted(
    const struct btt_execution *execution,
    const struct btt_layout *layout
) {
    return
        execution->stats.logical_phase_ands
            == 4U * (uint64_t)(layout->n4 + layout->n8)
        && execution->stats.logical_phase_ors
            == 2U * (uint64_t)(layout->n4 + layout->n8)
        && execution->stats.phase_products_inside_or
            == execution->stats.logical_phase_ors
        && execution->stats.carrier_reads
            == 8U * (uint64_t)layout->n4
                + 11U * (uint64_t)layout->n8
        && execution->stats.phase_cell_updates
            == 6U * (uint64_t)layout->n2
                + 2U * (uint64_t)layout->n4
                + 4U * (uint64_t)layout->n8
        && execution->stats.final_decodes == layout->n8
        && execution->projection.cells == layout->n8;
}

static int cbts_context_is_sealed(
    const struct cbts_context *context
) {
    return
        context != NULL
        && context->state == CBTS_READY
        && context->carrier_creation_count == 1U
        && memcmp(
            &context->layout,
            &context->sealed_layout,
            sizeof(context->layout)
        ) == 0
        && memcmp(
            context->program_hash,
            context->sealed_program_hash,
            sizeof(context->program_hash)
        ) == 0
        && context->program_hash[0] == btt_program_hash(
            &context->layout, BTT_NEIGHBOR_AND
        )
        && context->program_hash[1] == btt_program_hash(
            &context->layout, BTT_NEIGHBOR_NAND
        )
        && context->carrier.cells == context->sealed_state.cells
        && memcmp(
            context->carrier.baseline,
            context->sealed_state.baseline,
            context->carrier.cells
                * sizeof(*context->carrier.baseline)
        ) == 0
        && restoration(
            &context->carrier, &context->sealed_state
        ) <= RESTORATION_TOLERANCE;
}

static int cbts_transact(
    struct cbts_context *context,
    size_t variant,
    struct cbts_outcome *outcome
) {
    if (
        outcome == NULL
        || variant >= CBTS_VARIANTS
        || !cbts_context_is_sealed(context)
    ) {
        return 0;
    }
#ifdef CATVM_BOOLEAN_TT_TESTING
    if (context->testing_inert) {
        ++context->completed_transactions;
        return 3;
    }
    const enum btt_mode mode = context->testing_mode;
#else
    const enum btt_mode mode = BTT_CORRECT;
#endif
    context->state = CBTS_RUNNING;
    struct btt_execution execution = btt_execute(
        &context->carrier,
        &context->layout,
        (enum btt_variant)variant,
        mode
    );
    const double sealed_error = restoration(
        &context->carrier, &context->sealed_state
    );

#ifdef CATVM_BOOLEAN_TT_TESTING
    if (
        mode == BTT_WRONG_BOUNDARY_INVERSE
        || mode == BTT_MISSING_H_INVERSE
        || mode == BTT_REORDERED_H_BEFORE_Z_INVERSE
    ) {
        context->state = CBTS_FAILED;
        const int detected =
            execution.restoration_max_abs >= CONTROL_MINIMUM
            && sealed_error >= CONTROL_MINIMUM;
        btt_free_projection(&execution.projection);
        return detected ? -1 : 0;
    }
    if (mode == BTT_SNAPSHOT_RELOAD) {
        const int valid =
            execution.snapshot_loaded
            && execution.restoration_max_abs <= RESTORATION_TOLERANCE
            && sealed_error <= RESTORATION_TOLERANCE;
        if (!valid) {
            context->state = CBTS_FAILED;
            btt_free_projection(&execution.projection);
            return 0;
        }
        outcome->boundary_hash = execution.projection.hash;
        outcome->boundary_ones = execution.projection.ones;
        outcome->boundary_cells = execution.projection.cells;
        outcome->plan_hash = cbts_plan_hash(context);
        outcome->restoration_generation =
            context->restoration_generation;
        outcome->carrier_creation_count =
            context->carrier_creation_count;
        ++context->completed_transactions;
        context->state = CBTS_READY;
        btt_free_projection(&execution.projection);
        return 2;
    }
#endif

    if (
        mode != BTT_CORRECT
        || execution.snapshot_loaded
        || execution.restoration_max_abs > RESTORATION_TOLERANCE
        || sealed_error > RESTORATION_TOLERANCE
        || execution.integrity_max_abs > RESTORATION_TOLERANCE
        || execution.projection.maximum_root_error > ROOT_TOLERANCE
        || !cbts_stats_are_accepted(&execution, &context->layout)
    ) {
        context->state = CBTS_FAILED;
        btt_free_projection(&execution.projection);
        return 0;
    }
    context->state = CBTS_READY;
    if (!cbts_context_is_sealed(context)) {
        context->state = CBTS_FAILED;
        btt_free_projection(&execution.projection);
        return 0;
    }
    ++context->restoration_generation;
    outcome->boundary_hash = execution.projection.hash;
    outcome->boundary_ones = execution.projection.ones;
    outcome->boundary_cells = execution.projection.cells;
    outcome->plan_hash = cbts_plan_hash(context);
    outcome->restoration_generation =
        context->restoration_generation;
    outcome->carrier_creation_count =
        context->carrier_creation_count;
    ++context->completed_transactions;
    btt_free_projection(&execution.projection);
    return 1;
}

static int cbts_make_listener(const char *path) {
    struct stat existing;
    if (lstat(path, &existing) == 0 || errno != ENOENT) {
        return -1;
    }
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        return -1;
    }
    const int listener = socket(
        AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0
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

static int cbts_peer_is_same_real_user(int client) {
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

static int cbts_install_seccomp(int client) {
#ifdef CATVM_SANITIZER_BUILD
    (void)client;
    return 1;
#else
    scmp_filter_ctx filter = seccomp_init(SCMP_ACT_KILL_PROCESS);
    if (filter == NULL) {
        return 0;
    }
    int ok = 1;
#define CBTS_ALLOW_SYSCALL(name) \
    do { \
        if (seccomp_rule_add( \
            filter, SCMP_ACT_ALLOW, SCMP_SYS(name), 0 \
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
    CBTS_ALLOW_SYSCALL(brk);
    CBTS_ALLOW_SYSCALL(close);
    CBTS_ALLOW_SYSCALL(exit);
    CBTS_ALLOW_SYSCALL(exit_group);
    CBTS_ALLOW_SYSCALL(madvise);
    CBTS_ALLOW_SYSCALL(mmap);
    CBTS_ALLOW_SYSCALL(mprotect);
    CBTS_ALLOW_SYSCALL(mremap);
    CBTS_ALLOW_SYSCALL(munlock);
    CBTS_ALLOW_SYSCALL(munmap);
    CBTS_ALLOW_SYSCALL(rt_sigreturn);
#undef CBTS_ALLOW_SYSCALL
    if (!ok || seccomp_load(filter) != 0) {
        seccomp_release(filter);
        return 0;
    }
    return 1;
#endif
}

static int cbts_send_response(int client, const char *response) {
    const size_t bytes = strlen(response);
    return
        bytes > 0U
        && bytes < CBTS_RESPONSE_CAPACITY
        && send(client, response, bytes, MSG_NOSIGNAL)
            == (ssize_t)bytes;
}

static int cbts_receive_request(
    int client,
    char request[CBTS_REQUEST_CAPACITY]
) {
    const ssize_t received = recv(
        client,
        request,
        CBTS_REQUEST_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (received <= 0) {
        return 0;
    }
    if (
        (size_t)received >= CBTS_REQUEST_CAPACITY
        || memchr(request, '\0', (size_t)received) != NULL
    ) {
        cbts_secure_zero(request, CBTS_REQUEST_CAPACITY);
        return -1;
    }
    request[received] = '\0';
    return 1;
}

static int cbts_projection_request(const char *request) {
    return
        strncmp(request, "PROJECT ", 8U) == 0
        || strcmp(request, "DEBUG") == 0
        || strcmp(request, "DUMP") == 0
        || strcmp(request, "READ CARRIER") == 0
        || strcmp(request, "STATE DETAIL") == 0
        || strcmp(request, "BOND STATES") == 0
        || strcmp(request, "WITNESSES") == 0;
}

static int cbts_format_outcome(
    char response[CBTS_RESPONSE_CAPACITY],
    size_t variant,
    const struct cbts_outcome *outcome,
    int snapshot
) {
    const int written = snprintf(
        response,
        CBTS_RESPONSE_CAPACITY,
        "OK %s %zu %llu %016llx %016llx %zu %zu %llu",
        snapshot ? "SNAPSHOT" : "FINAL",
        variant,
        (unsigned long long)outcome->restoration_generation,
        (unsigned long long)outcome->plan_hash,
        (unsigned long long)outcome->boundary_hash,
        outcome->boundary_ones,
        outcome->boundary_cells,
        (unsigned long long)outcome->carrier_creation_count
    );
    return
        written > 0
        && (size_t)written < CBTS_RESPONSE_CAPACITY;
}

static int cbts_serve(
    int client,
    struct cbts_context *context
) {
    char request[CBTS_REQUEST_CAPACITY] = {0};
    char response[CBTS_RESPONSE_CAPACITY] = {0};
    struct cbts_outcome outcome = {0};
    int keep_running = 1;
    while (keep_running) {
        const int received = cbts_receive_request(client, request);
        if (received == 0) {
            break;
        }
        if (received < 0) {
            if (!cbts_send_response(client, "ERR E_PROTOCOL")) {
                break;
            }
            continue;
        }
        if (strcmp(request, "HELLO") == 0) {
            const uint64_t ands = 4U * (uint64_t)(
                context->layout.n4 + context->layout.n8
            );
            const uint64_t ors = 2U * (uint64_t)(
                context->layout.n4 + context->layout.n8
            );
            const uint64_t reads =
                8U * (uint64_t)context->layout.n4
                + 11U * (uint64_t)context->layout.n8;
            const uint64_t updates =
                6U * (uint64_t)context->layout.n2
                + 2U * (uint64_t)context->layout.n4
                + 4U * (uint64_t)context->layout.n8;
            const int written = snprintf(
                response,
                sizeof(response),
                "OK HELLO %s %zu %016llx %zu %zu %zu %zu "
                "%llu %llu %llu %llu %zu %llu",
                CBTS_PROTOCOL,
                context->layout.width,
                (unsigned long long)cbts_plan_hash(context),
                context->layout.carrier_cells,
                context->layout.n2,
                context->layout.n4,
                context->layout.n8,
                (unsigned long long)ands,
                (unsigned long long)ors,
                (unsigned long long)reads,
                (unsigned long long)updates,
                context->layout.n8,
                (unsigned long long)context->carrier_creation_count
            );
            if (
                written <= 0
                || (size_t)written >= sizeof(response)
                || !cbts_send_response(client, response)
            ) {
                break;
            }
        } else if (cbts_projection_request(request)) {
            if (!cbts_send_response(
                client,
                "ERR E_INTERMEDIATE_PROJECTION_DENIED"
            )) {
                break;
            }
        } else if (
            strlen(request) == 9U
            && strncmp(request, "EXECUTE ", 8U) == 0
            && (request[8] == '0' || request[8] == '1')
        ) {
            const size_t variant = (size_t)(request[8] - '0');
            cbts_secure_zero(&outcome, sizeof(outcome));
            const int transaction =
                cbts_transact(context, variant, &outcome);
            if (transaction < 0) {
                (void)cbts_send_response(
                    client, "ERR E_RESTORATION_DETECTED"
                );
                break;
            }
            if (transaction == 0) {
                (void)cbts_send_response(
                    client, "ERR E_MACHINE_LAW"
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
                    || !cbts_send_response(client, response)
                ) {
                    break;
                }
                continue;
            }
            if (
                !cbts_format_outcome(
                    response,
                    variant,
                    &outcome,
                    transaction == 2
                )
                || !cbts_send_response(client, response)
            ) {
                break;
            }
        } else if (strcmp(request, "SHUTDOWN") == 0) {
            if (!cbts_send_response(client, "OK CLOSED")) {
                break;
            }
            keep_running = 0;
        } else if (!cbts_send_response(client, "ERR E_PROTOCOL")) {
            break;
        }
        cbts_secure_zero(request, sizeof(request));
        cbts_secure_zero(response, sizeof(response));
        cbts_secure_zero(&outcome, sizeof(outcome));
    }
    cbts_secure_zero(request, sizeof(request));
    cbts_secure_zero(response, sizeof(response));
    cbts_secure_zero(&outcome, sizeof(outcome));
    return 1;
}

int main(int argc, char **argv) {
#ifdef CATVM_BOOLEAN_TT_SIZE_PROBE
    if (argc == 3 && strcmp(argv[1], "--size-probe") == 0) {
        const size_t width = btt_parse_width(argv[2]);
        const struct btt_layout layout = btt_make_layout(width);
        const long page_size_long = sysconf(_SC_PAGESIZE);
        if (page_size_long <= 0) {
            return 2;
        }
        printf(
            "{\"width\":%zu,"
            "\"context_bytes\":%zu,"
            "\"mapped_context_bytes\":%zu,"
            "\"carrier_cells\":%zu,"
            "\"live_carrier_bytes\":%zu,"
            "\"sealed_verification_state_bytes\":%zu,"
            "\"execution_snapshot_bytes\":%zu,"
            "\"projected_boundary_bytes\":%zu,"
            "\"request_buffer_bytes\":%u,"
            "\"response_buffer_bytes\":%u}\n",
            width,
            sizeof(struct cbts_context),
            cbts_page_round(
                sizeof(struct cbts_context),
                (size_t)page_size_long
            ),
            layout.carrier_cells,
            2U * layout.carrier_cells * sizeof(double complex),
            2U * layout.carrier_cells * sizeof(double complex),
            2U * layout.carrier_cells * sizeof(double complex),
            layout.n8 * sizeof(unsigned char),
            CBTS_REQUEST_CAPACITY,
            CBTS_RESPONSE_CAPACITY
        );
        return 0;
    }
#endif
#ifdef CATVM_BOOLEAN_TT_TESTING
    if (argc != 4) {
        return 2;
    }
    enum btt_mode testing_mode = BTT_CORRECT;
    int testing_inert = 0;
    if (strcmp(argv[3], "correct") == 0) {
        testing_mode = BTT_CORRECT;
    } else if (strcmp(argv[3], "wrong") == 0) {
        testing_mode = BTT_WRONG_BOUNDARY_INVERSE;
    } else if (strcmp(argv[3], "missing") == 0) {
        testing_mode = BTT_MISSING_H_INVERSE;
    } else if (strcmp(argv[3], "reordered") == 0) {
        testing_mode = BTT_REORDERED_H_BEFORE_Z_INVERSE;
    } else if (strcmp(argv[3], "snapshot") == 0) {
        testing_mode = BTT_SNAPSHOT_RELOAD;
    } else if (strcmp(argv[3], "inert") == 0) {
        testing_inert = 1;
    } else {
        return 2;
    }
#else
    if (argc != 3) {
        return 2;
    }
#endif
    const size_t width = btt_parse_width(argv[2]);
    if (
        !cbts_process_is_untraced()
        || !cbts_establish_process_guards()
    ) {
        return 2;
    }
    (void)umask(S_IRWXG | S_IRWXO);
    size_t mapped_bytes = 0U;
    struct cbts_context *context = cbts_create(
        width,
        &mapped_bytes
#ifdef CATVM_BOOLEAN_TT_TESTING
        ,
        testing_mode,
        testing_inert
#endif
    );
    if (context == NULL) {
        return 2;
    }
    const int listener = cbts_make_listener(argv[1]);
    if (listener < 0) {
        cbts_destroy(context, mapped_bytes);
        return 2;
    }
    const int client = accept4(
        listener, NULL, NULL, SOCK_CLOEXEC
    );
    if (client < 0 || !cbts_peer_is_same_real_user(client)) {
        if (client >= 0) {
            (void)close(client);
        }
        (void)close(listener);
        (void)unlink(argv[1]);
        cbts_destroy(context, mapped_bytes);
        return 2;
    }
    (void)close(listener);
    if (unlink(argv[1]) != 0) {
        (void)close(client);
        cbts_destroy(context, mapped_bytes);
        return 2;
    }
    if (!cbts_install_seccomp(client)) {
        (void)close(client);
        cbts_destroy(context, mapped_bytes);
        return 2;
    }
    (void)cbts_serve(client, context);
    (void)close(client);
    cbts_destroy(context, mapped_bytes);
    return 0;
}
