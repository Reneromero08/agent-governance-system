#define _GNU_SOURCE

/*
 * Public CATVM controller for the nonlinear Kerr/interference wave.  This
 * translation unit contains no complex carrier type, wave update, seal,
 * expected boundary, or intermediate-state representation.
 */

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#define KWC_REQUEST_CAPACITY 128U
#define KWC_RESPONSE_CAPACITY 512U
#define KWC_MAX_CYCLES 100000U

struct kwc_transport {
    int socket_fd;
    uint64_t request_packets;
    uint64_t response_packets;
    uint64_t request_bytes;
    uint64_t response_bytes;
};

struct kwc_hello {
    char protocol[48];
    size_t cells;
    size_t depth;
    uint64_t carrier_creation_count;
};

struct kwc_projection {
    char kind[16];
    int program;
    uint64_t generation;
    uint64_t boundary_hash;
    double intensity_zero;
    double intensity_two;
    double fringe_zero_one;
    double restoration_error;
    uint64_t carrier_creation_count;
};

struct kwc_access {
    int proc_mem_denied;
    int proc_maps_denied;
    int proc_fd_denied;
    int process_vm_readv_denied;
    int ptrace_denied;
    int pidfd_getfd_denied;
};

static void kwc_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static int kwc_connect(const char *path) {
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        return -1;
    }
    const int descriptor = socket(
        AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0
    );
    if (descriptor < 0) {
        return -1;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path) + 1U);
    if (
        connect(
            descriptor,
            (const struct sockaddr *)&address,
            sizeof(address)
        ) != 0
    ) {
        (void)close(descriptor);
        return -1;
    }
    return descriptor;
}

static int kwc_exchange(
    struct kwc_transport *transport,
    const char *request,
    char response[KWC_RESPONSE_CAPACITY]
) {
    const size_t request_bytes = strlen(request);
    if (
        request_bytes == 0U
        || request_bytes > KWC_REQUEST_CAPACITY
        || send(
            transport->socket_fd,
            request,
            request_bytes,
            MSG_NOSIGNAL
        ) != (ssize_t)request_bytes
    ) {
        return 0;
    }
    ++transport->request_packets;
    transport->request_bytes += request_bytes;
    const ssize_t received = recv(
        transport->socket_fd,
        response,
        KWC_RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= KWC_RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        kwc_zero(response, KWC_RESPONSE_CAPACITY);
        return 0;
    }
    response[received] = '\0';
    ++transport->response_packets;
    transport->response_bytes += (uint64_t)received;
    return 1;
}

static int kwc_parse_hello(
    const char *response,
    struct kwc_hello *hello
) {
    unsigned long long carrier = 0ULL;
    int consumed = 0;
    const int fields = sscanf(
        response,
        "OK HELLO %47s %zu %zu %llu%n",
        hello->protocol,
        &hello->cells,
        &hello->depth,
        &carrier,
        &consumed
    );
    hello->carrier_creation_count = (uint64_t)carrier;
    return (
        fields == 4
        && consumed > 0
        && response[consumed] == '\0'
        && strcmp(
            hello->protocol, "CATVM_NONLINEAR_KERR_WAVE_1"
        ) == 0
        && hello->cells == 4U
        && hello->depth >= 1U
        && hello->depth <= 2048U
        && hello->carrier_creation_count == 1U
    );
}

static int kwc_parse_projection(
    const char *response,
    struct kwc_projection *projection
) {
    char boundary[17] = {0};
    unsigned long long generation = 0ULL;
    unsigned long long carrier = 0ULL;
    int consumed = 0;
    const int fields = sscanf(
        response,
        "OK %15s %d %llu %16s %lf %lf %lf %lf %llu%n",
        projection->kind,
        &projection->program,
        &generation,
        boundary,
        &projection->intensity_zero,
        &projection->intensity_two,
        &projection->fringe_zero_one,
        &projection->restoration_error,
        &carrier,
        &consumed
    );
    errno = 0;
    char *tail = NULL;
    const unsigned long long parsed = strtoull(boundary, &tail, 16);
    projection->generation = (uint64_t)generation;
    projection->boundary_hash = (uint64_t)parsed;
    projection->carrier_creation_count = (uint64_t)carrier;
    return (
        fields == 9
        && consumed > 0
        && response[consumed] == '\0'
        && (projection->program == 0 || projection->program == 1)
        && errno == 0
        && tail != boundary
        && *tail == '\0'
        && projection->intensity_zero >= 0.0
        && projection->intensity_zero <= 1.0
        && projection->intensity_two >= 0.0
        && projection->intensity_two <= 1.0
        && projection->fringe_zero_one >= 0.0
        && projection->fringe_zero_one <= 1.0
        && projection->restoration_error >= 0.0
        && projection->carrier_creation_count == 1U
    );
}

static int kwc_peer_pid(int socket_fd, pid_t *pid) {
    struct ucred credential;
    socklen_t bytes = sizeof(credential);
    if (
        getsockopt(
            socket_fd,
            SOL_SOCKET,
            SO_PEERCRED,
            &credential,
            &bytes
        ) != 0
        || bytes != sizeof(credential)
        || credential.pid <= 0
    ) {
        return 0;
    }
    *pid = credential.pid;
    return 1;
}

static int kwc_attack_proc(pid_t pid, const char *leaf) {
    char path[96] = {0};
    const int written = snprintf(
        path, sizeof(path), "/proc/%ld/%s", (long)pid, leaf
    );
    if (written <= 0 || (size_t)written >= sizeof(path)) {
        return 0;
    }
    errno = 0;
    const int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor >= 0) {
        (void)close(descriptor);
        return 0;
    }
    return errno == EACCES || errno == EPERM;
}

static int kwc_attack_process_vm(pid_t pid) {
    unsigned char local = 0U;
    struct iovec local_iov = {.iov_base = &local, .iov_len = 1U};
    struct iovec remote_iov = {
        .iov_base = (void *)(uintptr_t)1U,
        .iov_len = 1U
    };
    errno = 0;
    const ssize_t result = process_vm_readv(
        pid, &local_iov, 1U, &remote_iov, 1U, 0U
    );
    kwc_zero(&local, sizeof(local));
    return result < 0 && (errno == EPERM || errno == EACCES);
}

static int kwc_attack_ptrace(pid_t pid) {
    errno = 0;
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        return errno == EPERM || errno == EACCES;
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    (void)ptrace(PTRACE_DETACH, pid, NULL, NULL);
    return 0;
}

static int kwc_attack_pidfd(pid_t pid) {
#if defined(SYS_pidfd_open) && defined(SYS_pidfd_getfd)
    errno = 0;
    const int pidfd = (int)syscall(SYS_pidfd_open, pid, 0U);
    if (pidfd < 0) {
        return errno == EPERM || errno == EACCES || errno == ENOSYS;
    }
    errno = 0;
    const int duplicate = (int)syscall(
        SYS_pidfd_getfd, pidfd, 0, 0U
    );
    const int denied = (
        duplicate < 0
        && (errno == EPERM || errno == EACCES || errno == ENOSYS)
    );
    if (duplicate >= 0) {
        (void)close(duplicate);
    }
    (void)close(pidfd);
    return denied;
#else
    (void)pid;
    return 1;
#endif
}

static int kwc_access(
    int socket_fd,
    struct kwc_access *access
) {
    pid_t pid = 0;
    if (!kwc_peer_pid(socket_fd, &pid)) {
        return 0;
    }
    access->proc_mem_denied = kwc_attack_proc(pid, "mem");
    access->proc_maps_denied = kwc_attack_proc(pid, "maps");
    access->proc_fd_denied = kwc_attack_proc(pid, "fd/0");
    access->process_vm_readv_denied = kwc_attack_process_vm(pid);
    access->ptrace_denied = kwc_attack_ptrace(pid);
    access->pidfd_getfd_denied = kwc_attack_pidfd(pid);
    return (
        access->proc_mem_denied
        && access->proc_maps_denied
        && access->proc_fd_denied
        && access->process_vm_readv_denied
        && access->ptrace_denied
        && access->pidfd_getfd_denied
    );
}

static int kwc_denied(
    struct kwc_transport *transport,
    const char *request,
    const char *expected
) {
    char response[KWC_RESPONSE_CAPACITY] = {0};
    const int accepted = (
        kwc_exchange(transport, request, response)
        && strcmp(response, expected) == 0
    );
    kwc_zero(response, sizeof(response));
    return accepted;
}

static int kwc_parse_cycles(const char *text, unsigned long *cycles) {
    errno = 0;
    char *tail = NULL;
    const unsigned long parsed = strtoul(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed < 1U
        || parsed > KWC_MAX_CYCLES
    ) {
        return 0;
    }
    *cycles = parsed;
    return 1;
}

static void kwc_print_projection(
    const struct kwc_projection *projection
) {
    printf(
        "{\"program\":%d,\"generation\":%llu,"
        "\"boundary_fnv1a64\":\"%016llx\","
        "\"intensity_zero\":%.17g,\"intensity_two\":%.17g,"
        "\"fringe_zero_one\":%.17g,"
        "\"restoration_error\":%.17g}",
        projection->program,
        (unsigned long long)projection->generation,
        (unsigned long long)projection->boundary_hash,
        projection->intensity_zero,
        projection->intensity_two,
        projection->fringe_zero_one,
        projection->restoration_error
    );
}

int main(int argc, char **argv) {
    if (argc != 4) {
        return 2;
    }
    unsigned long cycles = 0U;
    if (
        !kwc_parse_cycles(argv[2], &cycles)
        || (
            strcmp(argv[3], "correct") != 0
            && strcmp(argv[3], "snapshot") != 0
            && strcmp(argv[3], "inert") != 0
            && strcmp(argv[3], "restoration") != 0
        )
    ) {
        return 2;
    }
    struct kwc_transport transport = {
        .socket_fd = kwc_connect(argv[1])
    };
    if (transport.socket_fd < 0) {
        return 2;
    }
    char response[KWC_RESPONSE_CAPACITY] = {0};
    struct kwc_hello hello = {0};
    if (
        !kwc_exchange(&transport, "HELLO", response)
        || !kwc_parse_hello(response, &hello)
    ) {
        (void)close(transport.socket_fd);
        return 2;
    }
    kwc_zero(response, sizeof(response));

    if (strcmp(argv[3], "restoration") == 0) {
        const int detected = (
            kwc_exchange(&transport, "EXECUTE 0", response)
            && strcmp(response, "ERR E_RESTORATION_DETECTED") == 0
        );
        (void)close(transport.socket_fd);
        if (!detected) {
            return 2;
        }
        printf(
            "{\"result\":\"PASS\","
            "\"restoration_failure_detected\":true}\n"
        );
        return 0;
    }

    struct kwc_access access = {0};
    if (
        strcmp(argv[3], "correct") == 0
        && !kwc_access(transport.socket_fd, &access)
    ) {
        (void)close(transport.socket_fd);
        return 2;
    }
    static const char *const projection_requests[] = {
        "PROJECT CELL 0",
        "PROJECT WAVE",
        "PROJECT KERR INPUT",
        "DUMP",
        "STATE DETAIL"
    };
    for (
        size_t index = 0U;
        index
            < sizeof(projection_requests)
                / sizeof(projection_requests[0]);
        ++index
    ) {
        if (!kwc_denied(
            &transport,
            projection_requests[index],
            "ERR E_INTERMEDIATE_PROJECTION_DENIED"
        )) {
            (void)close(transport.socket_fd);
            return 2;
        }
    }
    if (
        !kwc_denied(&transport, "EXECUTE NULL", "ERR E_PROTOCOL")
        || !kwc_denied(&transport, "UNKNOWN", "ERR E_PROTOCOL")
    ) {
        (void)close(transport.socket_fd);
        return 2;
    }

    struct kwc_projection primary = {0};
    struct kwc_projection reuse = {0};
    double maximum_restoration_error = 0.0;
    double maximum_boundary_drift = 0.0;
    for (unsigned long cycle = 0U; cycle < cycles; ++cycle) {
        const int program = (int)(cycle % 2U);
        const char *request = program == 0
            ? "EXECUTE 0"
            : "EXECUTE 1";
        struct kwc_projection current = {0};
        if (
            !kwc_exchange(&transport, request, response)
            || !kwc_parse_projection(response, &current)
            || current.program != program
            || current.restoration_error > 2.0e-10
        ) {
            (void)close(transport.socket_fd);
            return 2;
        }
        maximum_restoration_error = fmax(
            maximum_restoration_error,
            current.restoration_error
        );
        const uint64_t expected_generation =
            strcmp(argv[3], "correct") == 0
                ? (uint64_t)cycle + 1U
                : 0U;
        const char *expected_kind =
            strcmp(argv[3], "snapshot") == 0
                ? "SNAPSHOT"
                : (
                    strcmp(argv[3], "inert") == 0
                        ? "INERT___"
                        : "FINAL___"
                );
        if (
            current.generation != expected_generation
            || strcmp(current.kind, expected_kind) != 0
        ) {
            (void)close(transport.socket_fd);
            return 2;
        }
        if (cycle == 0U) {
            primary = current;
        } else if (cycle == 1U) {
            reuse = current;
            if (reuse.boundary_hash == primary.boundary_hash) {
                (void)close(transport.socket_fd);
                return 2;
            }
        } else {
            const struct kwc_projection *expected =
                program == 0 ? &primary : &reuse;
            const double boundary_drift = fmax(
                fabs(
                    current.intensity_zero
                    - expected->intensity_zero
                ),
                fmax(
                    fabs(
                        current.intensity_two
                        - expected->intensity_two
                    ),
                    fabs(
                        current.fringe_zero_one
                        - expected->fringe_zero_one
                    )
                )
            );
            maximum_boundary_drift = fmax(
                maximum_boundary_drift,
                boundary_drift
            );
            if (boundary_drift > 2.0e-10) {
                (void)close(transport.socket_fd);
                return 2;
            }
        }
        kwc_zero(response, sizeof(response));
        kwc_zero(&current, sizeof(current));
    }
    if (
        !kwc_exchange(&transport, "SHUTDOWN", response)
        || strcmp(response, "OK CLOSED") != 0
    ) {
        (void)close(transport.socket_fd);
        return 2;
    }
    (void)close(transport.socket_fd);

    printf(
        "{\"result\":\"PASS\",\"protocol\":\"%s\","
        "\"cells\":%zu,\"depth\":%zu,\"transactions\":%lu,"
        "\"carrier_creation_count\":1,"
        "\"mode\":\"%s\","
        "\"same_service_process\":true,"
        "\"same_actual_restored_carrier\":%s,"
        "\"actual_inverse\":%s,\"snapshot_reload\":%s,"
        "\"all_intermediate_projection_requests_denied\":true,"
        "\"null_carrier_request_denied\":true,"
        "\"unknown_command_denied\":true,"
        "\"proc_mem_denied\":%s,\"proc_maps_denied\":%s,"
        "\"proc_fd_denied\":%s,"
        "\"process_vm_readv_denied\":%s,"
        "\"ptrace_denied\":%s,\"pidfd_getfd_denied\":%s,"
        "\"request_packets\":%llu,\"response_packets\":%llu,"
        "\"request_bytes\":%llu,\"response_bytes\":%llu,"
        "\"continuous_boundary_tolerance\":2e-10,"
        "\"maximum_repeated_boundary_drift\":%.17g,"
        "\"maximum_restoration_error\":%.17g,"
        "\"repeated_boundary_hash_exactness_required\":false,"
        "\"primary\":",
        hello.protocol,
        hello.cells,
        hello.depth,
        cycles,
        argv[3],
        strcmp(argv[3], "correct") == 0 ? "true" : "false",
        strcmp(argv[3], "correct") == 0 ? "true" : "false",
        strcmp(argv[3], "snapshot") == 0 ? "true" : "false",
        access.proc_mem_denied ? "true" : "false",
        access.proc_maps_denied ? "true" : "false",
        access.proc_fd_denied ? "true" : "false",
        access.process_vm_readv_denied ? "true" : "false",
        access.ptrace_denied ? "true" : "false",
        access.pidfd_getfd_denied ? "true" : "false",
        (unsigned long long)transport.request_packets,
        (unsigned long long)transport.response_packets,
        (unsigned long long)transport.request_bytes,
        (unsigned long long)transport.response_bytes,
        maximum_boundary_drift,
        maximum_restoration_error
    );
    kwc_print_projection(&primary);
    printf(",\"reuse\":");
    kwc_print_projection(&reuse);
    printf("}\n");
    kwc_zero(&primary, sizeof(primary));
    kwc_zero(&reuse, sizeof(reuse));
    kwc_zero(&hello, sizeof(hello));
    kwc_zero(&access, sizeof(access));
    kwc_zero(response, sizeof(response));
    return 0;
}
