#define _GNU_SOURCE

/*
 * Minimal CATVM custody boundary for the nonlinear Kerr/interference wave.
 * The controller can request only complete transactions or final boundary
 * projections.  The four-cell wave, verification seal, and all intermediate
 * amplitudes/phases remain in this non-dumpable locked single-peer process.
 */

#define NW_EMBEDDED 1
#include "algebraic_nonlinear_kerr_wave.c"

#include <errno.h>
#include <linux/prctl.h>
#include <seccomp.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#define CKW_PROTOCOL "CATVM_NONLINEAR_KERR_WAVE_1"
#define CKW_REQUEST_CAPACITY 128U
#define CKW_RESPONSE_CAPACITY 512U

struct ckw_context {
    struct nw_carrier carrier;
    struct nw_stats stats;
    size_t depth;
    uint64_t carrier_creation_count;
    uint64_t completed_transactions;
#ifdef CATVM_KERR_TESTING
    enum nw_restore_mode testing_mode;
    int testing_inert;
#endif
};

static void ckw_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static size_t ckw_page_round(size_t bytes, size_t page) {
    if (page == 0U || bytes > SIZE_MAX - page + 1U) {
        return 0U;
    }
    return ((bytes + page - 1U) / page) * page;
}

static int ckw_untraced(void) {
#ifdef CATVM_KERR_TRACE_BUILD
    return 1;
#else
    FILE *stream = fopen("/proc/self/status", "r");
    if (stream == NULL) {
        return 0;
    }
    char line[128] = {0};
    int tracer = -1;
    while (fgets(line, sizeof(line), stream) != NULL) {
        if (sscanf(line, "TracerPid:\t%d", &tracer) == 1) {
            break;
        }
    }
    return fclose(stream) == 0 && tracer == 0;
#endif
}

static int ckw_guards(void) {
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

static struct ckw_context *ckw_create(
    size_t depth,
    size_t *mapped_bytes
#ifdef CATVM_KERR_TESTING
    ,
    enum nw_restore_mode testing_mode,
    int testing_inert
#endif
) {
    const long page_long = sysconf(_SC_PAGESIZE);
    if (page_long <= 0) {
        return NULL;
    }
    const size_t page = (size_t)page_long;
    *mapped_bytes = ckw_page_round(sizeof(struct ckw_context), page);
    struct ckw_context *context = mmap(
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
    memset(context, 0, sizeof(*context));
    context->carrier = nw_make_carrier(UINT64_C(53));
    context->depth = depth;
    context->carrier_creation_count = 1U;
#ifdef CATVM_KERR_TESTING
    context->testing_mode = testing_mode;
    context->testing_inert = testing_inert;
#endif
#ifndef CATVM_SANITIZER_BUILD
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        ckw_zero(context, sizeof(*context));
        (void)munlock(context, *mapped_bytes);
        (void)munmap(context, *mapped_bytes);
        return NULL;
    }
#endif
    return context;
}

static void ckw_destroy(
    struct ckw_context *context,
    size_t mapped_bytes
) {
    if (context == NULL) {
        return;
    }
    ckw_zero(context, sizeof(*context));
    (void)munlock(context, mapped_bytes);
    (void)munmap(context, mapped_bytes);
}

static int ckw_listener(const char *path) {
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

static int ckw_same_user(int client) {
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

static int ckw_seccomp(int client) {
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
#define CKW_ALLOW(name) \
    do { \
        if (seccomp_rule_add( \
            filter, SCMP_ACT_ALLOW, SCMP_SYS(name), 0 \
        ) != 0) { \
            ok = 0; \
        } \
    } while (0)
    CKW_ALLOW(brk);
    CKW_ALLOW(close);
    CKW_ALLOW(exit);
    CKW_ALLOW(exit_group);
    CKW_ALLOW(madvise);
    CKW_ALLOW(mmap);
    CKW_ALLOW(mprotect);
    CKW_ALLOW(mremap);
    CKW_ALLOW(munlock);
    CKW_ALLOW(munmap);
    CKW_ALLOW(rt_sigreturn);
#undef CKW_ALLOW
    if (!ok || seccomp_load(filter) != 0) {
        seccomp_release(filter);
        return 0;
    }
    seccomp_release(filter);
    return 1;
#endif
}

static int ckw_send(int client, const char *response) {
    const size_t bytes = strlen(response);
    return (
        bytes > 0U
        && bytes < CKW_RESPONSE_CAPACITY
        && send(client, response, bytes, MSG_NOSIGNAL)
            == (ssize_t)bytes
    );
}

static int ckw_execute(
    struct ckw_context *context,
    size_t program,
    struct nw_boundary *boundary,
    double *restoration_error,
    int *snapshot
) {
#ifdef CATVM_KERR_TESTING
    if (context->testing_inert) {
        boundary->hash = program == 3U
            ? UINT64_C(0x1111111111111111)
            : UINT64_C(0x2222222222222222);
        boundary->intensity_zero = program == 3U ? 0.25 : 0.50;
        boundary->intensity_two = program == 3U ? 0.50 : 0.25;
        boundary->fringe_zero_one = 0.375;
        *restoration_error = 0.0;
        *snapshot = 0;
        ++context->completed_transactions;
        return 2;
    }
    const enum nw_restore_mode mode = context->testing_mode;
#else
    const enum nw_restore_mode mode = NW_RESTORE_CORRECT;
#endif
    *boundary = nw_transaction(
        &context->carrier,
        context->depth,
        program,
        mode,
        0,
        0,
        &context->stats,
        restoration_error
    );
    *snapshot = mode == NW_RESTORE_SNAPSHOT;
    if (
        mode == NW_RESTORE_MISSING_LAYER
        || mode == NW_RESTORE_WRONG_KERR
        || mode == NW_RESTORE_REORDERED
    ) {
        return *restoration_error > NW_CONTROL_MINIMUM ? -1 : 0;
    }
    if (
        *restoration_error > NW_RESTORATION_TOLERANCE
        || context->carrier_creation_count != 1U
    ) {
        return 0;
    }
    ++context->completed_transactions;
    return mode == NW_RESTORE_SNAPSHOT ? 3 : 1;
}

static int ckw_serve(int client, struct ckw_context *context) {
    char request[CKW_REQUEST_CAPACITY] = {0};
    char response[CKW_RESPONSE_CAPACITY] = {0};
    struct nw_boundary boundary = {0};
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
            if (!ckw_send(client, "ERR E_PROTOCOL")) {
                break;
            }
            ckw_zero(request, sizeof(request));
            continue;
        }
        request[received] = '\0';
        if (strcmp(request, "HELLO") == 0) {
            const int written = snprintf(
                response,
                sizeof(response),
                "OK HELLO %s 4 %zu 1",
                CKW_PROTOCOL,
                context->depth
            );
            if (
                written <= 0
                || (size_t)written >= sizeof(response)
                || !ckw_send(client, response)
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
            if (!ckw_send(
                client, "ERR E_INTERMEDIATE_PROJECTION_DENIED"
            )) {
                break;
            }
        } else if (
            strcmp(request, "EXECUTE 0") == 0
            || strcmp(request, "EXECUTE 1") == 0
        ) {
            const size_t program = request[8] == '0' ? 3U : 41U;
            double restoration_error = 0.0;
            int snapshot = 0;
            ckw_zero(&boundary, sizeof(boundary));
            const int result = ckw_execute(
                context,
                program,
                &boundary,
                &restoration_error,
                &snapshot
            );
            if (result < 0) {
                (void)ckw_send(client, "ERR E_RESTORATION_DETECTED");
                break;
            }
            if (result == 0) {
                (void)ckw_send(client, "ERR E_MACHINE_LAW");
                break;
            }
            const int written = snprintf(
                response,
                sizeof(response),
                "OK %s %d %020llu %016llx %+.17e %+.17e %+.17e "
                "%+.17e %020llu",
                result == 3
                    ? "SNAPSHOT"
                    : (result == 2 ? "INERT___" : "FINAL___"),
                program == 3U ? 0 : 1,
                (unsigned long long)
                    context->carrier.restoration_generation,
                (unsigned long long)boundary.hash,
                boundary.intensity_zero,
                boundary.intensity_two,
                boundary.fringe_zero_one,
                restoration_error,
                (unsigned long long)context->carrier_creation_count
            );
            if (
                written <= 0
                || (size_t)written >= sizeof(response)
                || !ckw_send(client, response)
            ) {
                break;
            }
            (void)snapshot;
        } else if (strcmp(request, "SHUTDOWN") == 0) {
            if (!ckw_send(client, "OK CLOSED")) {
                break;
            }
            running = 0;
        } else if (!ckw_send(client, "ERR E_PROTOCOL")) {
            break;
        }
        ckw_zero(request, sizeof(request));
        ckw_zero(response, sizeof(response));
        ckw_zero(&boundary, sizeof(boundary));
    }
    ckw_zero(request, sizeof(request));
    ckw_zero(response, sizeof(response));
    ckw_zero(&boundary, sizeof(boundary));
    return 1;
}

static int ckw_parse_depth(const char *text, size_t *depth) {
    char *tail = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed < 1ULL
        || parsed > 2048ULL
    ) {
        return 0;
    }
    *depth = (size_t)parsed;
    return 1;
}

int main(int argc, char **argv) {
#ifdef CATVM_KERR_SIZE_PROBE
    if (argc == 2 && strcmp(argv[1], "--size-probe") == 0) {
        const long page_long = sysconf(_SC_PAGESIZE);
        if (page_long <= 0) {
            return 2;
        }
        printf(
            "{\"context_bytes\":%zu,\"context_mapped_bytes\":%zu,"
            "\"carrier_bytes\":%zu,\"stats_bytes\":%zu,"
            "\"request_buffer_bytes\":%u,\"response_buffer_bytes\":%u}\n",
            sizeof(struct ckw_context),
            ckw_page_round(
                sizeof(struct ckw_context), (size_t)page_long
            ),
            sizeof(struct nw_carrier),
            sizeof(struct nw_stats),
            CKW_REQUEST_CAPACITY,
            CKW_RESPONSE_CAPACITY
        );
        return 0;
    }
#endif
    if (argc != 4) {
        return 2;
    }
    size_t depth = 0U;
    if (!ckw_parse_depth(argv[2], &depth)) {
        return 2;
    }
#ifdef CATVM_KERR_TESTING
    enum nw_restore_mode testing_mode = NW_RESTORE_CORRECT;
    int testing_inert = 0;
    if (strcmp(argv[3], "correct") == 0) {
        testing_mode = NW_RESTORE_CORRECT;
    } else if (strcmp(argv[3], "missing") == 0) {
        testing_mode = NW_RESTORE_MISSING_LAYER;
    } else if (strcmp(argv[3], "wrong") == 0) {
        testing_mode = NW_RESTORE_WRONG_KERR;
    } else if (strcmp(argv[3], "reordered") == 0) {
        testing_mode = NW_RESTORE_REORDERED;
    } else if (strcmp(argv[3], "snapshot") == 0) {
        testing_mode = NW_RESTORE_SNAPSHOT;
    } else if (strcmp(argv[3], "inert") == 0) {
        testing_inert = 1;
    } else {
        return 2;
    }
#else
    if (strcmp(argv[3], "correct") != 0) {
        return 2;
    }
#endif
    if (!ckw_untraced() || !ckw_guards()) {
        return 2;
    }
    (void)umask(S_IRWXG | S_IRWXO);
    size_t mapped_bytes = 0U;
    struct ckw_context *context = ckw_create(
        depth,
        &mapped_bytes
#ifdef CATVM_KERR_TESTING
        ,
        testing_mode,
        testing_inert
#endif
    );
    if (context == NULL) {
        return 2;
    }
    const int listener = ckw_listener(argv[1]);
    if (listener < 0) {
        ckw_destroy(context, mapped_bytes);
        return 2;
    }
    const int client = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0 || !ckw_same_user(client)) {
        if (client >= 0) {
            (void)close(client);
        }
        (void)close(listener);
        (void)unlink(argv[1]);
        ckw_destroy(context, mapped_bytes);
        return 2;
    }
    (void)close(listener);
    if (unlink(argv[1]) != 0 || !ckw_seccomp(client)) {
        (void)close(client);
        ckw_destroy(context, mapped_bytes);
        return 2;
    }
    (void)ckw_serve(client, context);
    (void)close(client);
    ckw_destroy(context, mapped_bytes);
    return 0;
}
