#define _GNU_SOURCE

/*
 * Minimal CATVM custody boundary for the topology-compiled nonlinear phase
 * graph. The public controller can request only atomic whole transactions.
 * Resident phases, compiled inverse custody, snapshots, and intermediate
 * source epochs remain inside this non-dumpable single-peer service.
 */

#define NPG_EMBEDDED 1
#include "topology_nonlinear_phase_graph.c"

#include <fcntl.h>
#include <linux/prctl.h>
#include <seccomp.h>
#include <signal.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#define CNG_PROTOCOL "CATVM_NONLINEAR_PHASE_GRAPH_1"
#define CNG_REQUEST_CAPACITY 128U
#define CNG_RESPONSE_CAPACITY 512U
#define CNG_CARRIER_ID 12401

enum cng_state {
    CNG_READY = 1,
    CNG_RUNNING = 2,
    CNG_FAILED = 3
};

struct cng_context {
    struct npg_graph graph;
    struct carrier carrier;
    size_t rounds;
    enum cng_state state;
    uint64_t restoration_generation;
    uint64_t carrier_creation_count;
    uint64_t completed_transactions;
#ifdef CATVM_NPG_TESTING
    enum npg_mode testing_mode;
    int testing_inert;
#endif
};

struct cng_outcome {
    uint64_t boundary_hash;
    uint64_t topology_hash;
    uint64_t restoration_generation;
    uint64_t carrier_creation_count;
    double interference_probability;
};

static void cng_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static size_t cng_page_round(size_t bytes, size_t page_size) {
    if (bytes > SIZE_MAX - page_size + 1U) {
        return 0U;
    }
    return ((bytes + page_size - 1U) / page_size) * page_size;
}

static int cng_protect(void *memory, size_t bytes) {
    const long page_long = sysconf(_SC_PAGESIZE);
    if (memory == NULL || bytes == 0U || page_long <= 0) {
        return 0;
    }
    const size_t page = (size_t)page_long;
    const uintptr_t raw = (uintptr_t)memory;
    const uintptr_t begin = raw - raw % page;
    if (bytes > UINTPTR_MAX - raw) {
        return 0;
    }
    const uintptr_t raw_end = raw + bytes;
    if (raw_end > UINTPTR_MAX - page + 1U) {
        return 0;
    }
    const uintptr_t end = ((raw_end + page - 1U) / page) * page;
    const size_t count = (size_t)(end - begin);
    return (
        mlock((void *)begin, count) == 0
        && madvise((void *)begin, count, MADV_DONTDUMP) == 0
        && madvise((void *)begin, count, MADV_DONTFORK) == 0
    );
}

static int cng_untraced(void) {
#ifdef CATVM_NPG_TRACE_BUILD
    return 1;
#else
    FILE *stream = fopen("/proc/self/status", "r");
    if (stream == NULL) {
        return 0;
    }
    char line[128];
    int tracer = -1;
    while (fgets(line, sizeof(line), stream) != NULL) {
        if (sscanf(line, "TracerPid:\t%d", &tracer) == 1) {
            break;
        }
    }
    return fclose(stream) == 0 && tracer == 0;
#endif
}

static int cng_guards(void) {
    const struct rlimit no_core = {.rlim_cur = 0U, .rlim_max = 0U};
    return (
        setrlimit(RLIMIT_CORE, &no_core) == 0
        && prctl(PR_SET_DUMPABLE, 0L, 0L, 0L, 0L) == 0
#ifdef PR_SET_PTRACER
        && prctl(PR_SET_PTRACER, 0L, 0L, 0L, 0L) == 0
#endif
        && prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) == 0
        && prctl(PR_GET_DUMPABLE, 0L, 0L, 0L, 0L) == 0
    );
}

static struct cng_context *cng_create(
    const char *topology,
    size_t rounds,
    size_t *mapped_bytes
#ifdef CATVM_NPG_TESTING
    ,
    enum npg_mode testing_mode
    ,
    int testing_inert
#endif
) {
    const long page_long = sysconf(_SC_PAGESIZE);
    if (page_long <= 0) {
        return NULL;
    }
    const size_t page = (size_t)page_long;
    *mapped_bytes = cng_page_round(sizeof(struct cng_context), page);
    struct cng_context *context = mmap(
        NULL,
        *mapped_bytes,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );
    if (
        context == MAP_FAILED
        || mlock(context, *mapped_bytes) != 0
        || madvise(context, *mapped_bytes, MADV_DONTDUMP) != 0
        || madvise(context, *mapped_bytes, MADV_DONTFORK) != 0
    ) {
        if (context != MAP_FAILED) {
            (void)munlock(context, *mapped_bytes);
            (void)munmap(context, *mapped_bytes);
        }
        return NULL;
    }
    context->graph = npg_load_graph(topology);
    context->rounds = rounds;
    const struct process process = {
        .carrier_cells = context->graph.width + 2U
    };
    context->carrier = make_carrier(&process, CNG_CARRIER_ID);
    const size_t carrier_bytes =
        context->carrier.cells * sizeof(*context->carrier.working);
    if (
        !cng_protect(context->carrier.baseline, carrier_bytes)
        || !cng_protect(context->carrier.working, carrier_bytes)
#ifndef CATVM_SANITIZER_BUILD
        || mlockall(MCL_CURRENT | MCL_FUTURE) != 0
#endif
    ) {
        free_carrier(&context->carrier);
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
    context->state = CNG_READY;
    context->carrier_creation_count = 1U;
#ifdef CATVM_NPG_TESTING
    context->testing_mode = testing_mode;
    context->testing_inert = testing_inert;
#endif
    return context;
}

static void cng_destroy(
    struct cng_context *context,
    size_t mapped_bytes
) {
    if (context == NULL) {
        return;
    }
    if (context->carrier.baseline != NULL) {
        cng_zero(
            context->carrier.baseline,
            context->carrier.cells
                * sizeof(*context->carrier.baseline)
        );
    }
    if (context->carrier.working != NULL) {
        cng_zero(
            context->carrier.working,
            context->carrier.cells
                * sizeof(*context->carrier.working)
        );
    }
    free_carrier(&context->carrier);
    cng_zero(context, sizeof(*context));
    (void)munlock(context, mapped_bytes);
    (void)munmap(context, mapped_bytes);
}

static int cng_transact(
    struct cng_context *context,
    enum npg_program program,
    struct cng_outcome *outcome
) {
    if (
        context == NULL
        || outcome == NULL
        || context->state != CNG_READY
        || context->carrier_creation_count != 1U
    ) {
        return 0;
    }
    context->state = CNG_RUNNING;
#ifdef CATVM_NPG_TESTING
    if (context->testing_inert) {
        ++context->completed_transactions;
        context->state = CNG_READY;
        outcome->boundary_hash = program == NPG_PRIMARY
            ? UINT64_C(0x1111111111111111)
            : UINT64_C(0x2222222222222222);
        outcome->interference_probability =
            program == NPG_PRIMARY ? 0.25 : 0.75;
        outcome->restoration_generation = 0U;
        outcome->carrier_creation_count =
            context->carrier_creation_count;
        return 3;
    }
#endif
#ifdef CATVM_NPG_TESTING
    const enum npg_mode mode = context->testing_mode;
#else
    const enum npg_mode mode = NPG_CORRECT;
#endif
    const struct npg_execution execution = npg_execute(
        &context->carrier,
        &context->graph,
        context->rounds,
        program,
        mode
    );
#ifdef CATVM_NPG_TESTING
    if (
        mode == NPG_MISSING_INVERSE
        || mode == NPG_WRONG_INVERSE
        || mode == NPG_REORDERED_INVERSE
    ) {
        context->state = CNG_FAILED;
        return execution.restoration_max_abs
            >= NPG_CONTROL_MINIMUM ? -1 : 0;
    }
    if (mode == NPG_SNAPSHOT) {
        if (
            !execution.snapshot_loaded
            || execution.actual_inverse
            || execution.restoration_max_abs > NPG_PHASE_TOLERANCE
        ) {
            context->state = CNG_FAILED;
            return 0;
        }
        outcome->boundary_hash = execution.projection.hash;
        outcome->topology_hash = context->graph.topology_hash;
        outcome->restoration_generation =
            context->restoration_generation;
        outcome->carrier_creation_count =
            context->carrier_creation_count;
        outcome->interference_probability =
            execution.projection.interference_probability;
        ++context->completed_transactions;
        context->state = CNG_READY;
        return 2;
    }
#endif
    npg_require_correct(
        &context->graph, context->rounds, &execution
    );
    ++context->restoration_generation;
    outcome->boundary_hash = execution.projection.hash;
    outcome->topology_hash = context->graph.topology_hash;
    outcome->restoration_generation =
        context->restoration_generation;
    outcome->carrier_creation_count =
        context->carrier_creation_count;
    outcome->interference_probability =
        execution.projection.interference_probability;
    ++context->completed_transactions;
    context->state = CNG_READY;
    return 1;
}

static int cng_listener(const char *path) {
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        return -1;
    }
    const int listener = socket(
        AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0
    );
    if (listener < 0) {
        return -1;
    }
    (void)unlink(path);
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

static int cng_same_user(int client) {
    struct ucred credential;
    socklen_t bytes = sizeof(credential);
    return (
        getsockopt(
            client,
            SOL_SOCKET,
            SO_PEERCRED,
            &credential,
            &bytes
        ) == 0
        && bytes == sizeof(credential)
        && credential.uid == getuid()
    );
}

static int cng_seccomp(int client) {
#ifdef CATVM_SANITIZER_BUILD
    (void)client;
    return 1;
#else
    scmp_filter_ctx filter = seccomp_init(SCMP_ACT_KILL_PROCESS);
    if (filter == NULL) {
        return 0;
    }
    int ok = 1;
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
#define CNG_ALLOW(name) \
    do { \
        if (seccomp_rule_add( \
            filter, SCMP_ACT_ALLOW, SCMP_SYS(name), 0 \
        ) != 0) { \
            ok = 0; \
        } \
    } while (0)
    CNG_ALLOW(brk);
    CNG_ALLOW(close);
    CNG_ALLOW(exit);
    CNG_ALLOW(exit_group);
    CNG_ALLOW(madvise);
    CNG_ALLOW(mmap);
    CNG_ALLOW(mprotect);
    CNG_ALLOW(mremap);
    CNG_ALLOW(munlock);
    CNG_ALLOW(munmap);
    CNG_ALLOW(rt_sigreturn);
#undef CNG_ALLOW
    if (!ok || seccomp_load(filter) != 0) {
        seccomp_release(filter);
        return 0;
    }
    seccomp_release(filter);
    return 1;
#endif
}

static int cng_send(int client, const char *response) {
    const size_t bytes = strlen(response);
    return (
        bytes > 0U
        && bytes < CNG_RESPONSE_CAPACITY
        && send(client, response, bytes, MSG_NOSIGNAL)
            == (ssize_t)bytes
    );
}

static int cng_serve(
    int client,
    struct cng_context *context
) {
    char request[CNG_REQUEST_CAPACITY] = {0};
    char response[CNG_RESPONSE_CAPACITY] = {0};
    struct cng_outcome outcome = {0};
    int running = 1;
    while (running) {
        const ssize_t received = recv(
            client,
            request,
            sizeof(request) - 1U,
            MSG_TRUNC
        );
        if (received <= 0) {
            break;
        }
        if (
            (size_t)received >= sizeof(request)
            || memchr(request, '\0', (size_t)received) != NULL
        ) {
            if (!cng_send(client, "ERR E_PROTOCOL")) {
                break;
            }
            cng_zero(request, sizeof(request));
            continue;
        }
        request[received] = '\0';
        if (strcmp(request, "HELLO") == 0) {
            const int written = snprintf(
                response,
                sizeof(response),
                "OK HELLO %s %016llx %zu %zu %zu 1",
                CNG_PROTOCOL,
                (unsigned long long)context->graph.topology_hash,
                context->graph.width,
                context->graph.edge_count,
                context->rounds
            );
            if (
                written <= 0
                || (size_t)written >= sizeof(response)
                || !cng_send(client, response)
            ) {
                break;
            }
        } else if (
            strncmp(request, "PROJECT", 7U) == 0
            || strcmp(request, "DUMP") == 0
            || strcmp(request, "DEBUG") == 0
            || strcmp(request, "READ CARRIER") == 0
            || strcmp(request, "STATE DETAIL") == 0
        ) {
            if (!cng_send(
                client, "ERR E_INTERMEDIATE_PROJECTION_DENIED"
            )) {
                break;
            }
        } else if (
            strcmp(request, "EXECUTE 0") == 0
            || strcmp(request, "EXECUTE 1") == 0
        ) {
            const enum npg_program program =
                request[8] == '0' ? NPG_PRIMARY : NPG_REUSE;
            cng_zero(&outcome, sizeof(outcome));
            const int result = cng_transact(
                context, program, &outcome
            );
            if (result < 0) {
                (void)cng_send(client, "ERR E_RESTORATION_DETECTED");
                break;
            }
            if (result == 0) {
                (void)cng_send(client, "ERR E_MACHINE_LAW");
                break;
            }
            const int written = snprintf(
                response,
                sizeof(response),
                "OK %s %d %020llu %016llx %+.17e %020llu",
                result == 2
                    ? "SNAPSHOT"
                    : (result == 3 ? "INERT___" : "FINAL___"),
                (int)program,
                (unsigned long long)outcome.restoration_generation,
                (unsigned long long)outcome.boundary_hash,
                outcome.interference_probability,
                (unsigned long long)outcome.carrier_creation_count
            );
            if (
                written <= 0
                || (size_t)written >= sizeof(response)
                || !cng_send(client, response)
            ) {
                break;
            }
        } else if (strcmp(request, "SHUTDOWN") == 0) {
            if (!cng_send(client, "OK CLOSED")) {
                break;
            }
            running = 0;
        } else if (!cng_send(client, "ERR E_PROTOCOL")) {
            break;
        }
        cng_zero(request, sizeof(request));
        cng_zero(response, sizeof(response));
        cng_zero(&outcome, sizeof(outcome));
    }
    cng_zero(request, sizeof(request));
    cng_zero(response, sizeof(response));
    cng_zero(&outcome, sizeof(outcome));
    return 1;
}

int main(int argc, char **argv) {
#ifdef CATVM_NPG_SIZE_PROBE
    if (argc == 2 && strcmp(argv[1], "--size-probe") == 0) {
        const long page_long = sysconf(_SC_PAGESIZE);
        if (page_long <= 0) {
            return 2;
        }
        printf(
            "{\"context_bytes\":%zu,"
            "\"context_mapped_bytes\":%zu,"
            "\"compiled_topology_bytes\":%zu,"
            "\"carrier_descriptor_bytes\":%zu,"
            "\"execution_summary_bytes\":%zu,"
            "\"projection_bytes\":%zu,"
            "\"request_buffer_bytes\":%u,"
            "\"response_buffer_bytes\":%u}\n",
            sizeof(struct cng_context),
            cng_page_round(
                sizeof(struct cng_context), (size_t)page_long
            ),
            sizeof(struct npg_graph),
            sizeof(struct carrier),
            sizeof(struct npg_execution),
            sizeof(struct npg_projection),
            CNG_REQUEST_CAPACITY,
            CNG_RESPONSE_CAPACITY
        );
        return 0;
    }
#endif
#ifdef CATVM_NPG_TESTING
    if (argc != 5) {
        return 2;
    }
#else
    if (argc != 4) {
        return 2;
    }
#endif
    const size_t rounds = npg_parse_size(
        argv[3],
        NPG_MIN_ROUNDS,
        NPG_MAX_ROUNDS,
        "CATVM nonlinear rounds invalid"
    );
#ifdef CATVM_NPG_TESTING
    enum npg_mode testing_mode = NPG_CORRECT;
    int testing_inert = 0;
    if (strcmp(argv[4], "correct") == 0) {
        testing_mode = NPG_CORRECT;
    } else if (strcmp(argv[4], "missing") == 0) {
        testing_mode = NPG_MISSING_INVERSE;
    } else if (strcmp(argv[4], "wrong") == 0) {
        testing_mode = NPG_WRONG_INVERSE;
    } else if (strcmp(argv[4], "reordered") == 0) {
        testing_mode = NPG_REORDERED_INVERSE;
    } else if (strcmp(argv[4], "snapshot") == 0) {
        testing_mode = NPG_SNAPSHOT;
    } else if (strcmp(argv[4], "inert") == 0) {
        testing_inert = 1;
    } else {
        return 2;
    }
#endif
    if (!cng_untraced() || !cng_guards()) {
        return 2;
    }
    (void)umask(S_IRWXG | S_IRWXO);
    size_t mapped_bytes = 0U;
    struct cng_context *context = cng_create(
        argv[2],
        rounds,
        &mapped_bytes
#ifdef CATVM_NPG_TESTING
        ,
        testing_mode,
        testing_inert
#endif
    );
    if (context == NULL) {
        return 2;
    }
    const int listener = cng_listener(argv[1]);
    if (listener < 0) {
        cng_destroy(context, mapped_bytes);
        return 2;
    }
    const int client = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0 || !cng_same_user(client)) {
        if (client >= 0) {
            (void)close(client);
        }
        (void)close(listener);
        (void)unlink(argv[1]);
        cng_destroy(context, mapped_bytes);
        return 2;
    }
    (void)close(listener);
    if (unlink(argv[1]) != 0 || !cng_seccomp(client)) {
        (void)close(client);
        cng_destroy(context, mapped_bytes);
        return 2;
    }
    (void)cng_serve(client, context);
    (void)close(client);
    cng_destroy(context, mapped_bytes);
    return 0;
}
