#define _GNU_SOURCE

/*
 * Public CATVM controller for the nonlinear phase graph. This translation
 * unit has no phase carrier type, phase update primitive, graph evaluator,
 * seal angles, expected boundary, or answer-bearing lookup table.
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

#define CNC_REQUEST_CAPACITY 256U
#define CNC_RESPONSE_CAPACITY 512U
#define CNC_MAX_CYCLES 100000U

struct cnc_transport {
    int socket_fd;
    uint64_t request_packets;
    uint64_t response_packets;
    uint64_t request_bytes;
    uint64_t response_bytes;
};

struct cnc_hello {
    char protocol[40];
    uint64_t topology_hash;
    size_t width;
    size_t edges;
    size_t rounds;
    uint64_t carrier_creation_count;
};

struct cnc_projection {
    int program;
    uint64_t generation;
    uint64_t boundary_hash;
    double probability;
    uint64_t carrier_creation_count;
};

struct cnc_access {
    int proc_mem_denied;
    int proc_maps_denied;
    int proc_fd_denied;
    int process_vm_readv_denied;
    int ptrace_denied;
    int pidfd_getfd_denied;
};

static void cnc_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static int cnc_connect(const char *path) {
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

static int cnc_exchange_bytes(
    struct cnc_transport *transport,
    const void *request,
    size_t request_bytes,
    char response[CNC_RESPONSE_CAPACITY]
) {
    if (
        request_bytes == 0U
        || request_bytes > CNC_REQUEST_CAPACITY
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
        CNC_RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= CNC_RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        cnc_zero(response, CNC_RESPONSE_CAPACITY);
        return 0;
    }
    response[received] = '\0';
    ++transport->response_packets;
    transport->response_bytes += (uint64_t)received;
    return 1;
}

static int cnc_exchange(
    struct cnc_transport *transport,
    const char *request,
    char response[CNC_RESPONSE_CAPACITY]
) {
    return cnc_exchange_bytes(
        transport, request, strlen(request), response
    );
}

static int cnc_parse_hello(
    const char *response,
    struct cnc_hello *hello
) {
    char topology[17] = {0};
    unsigned long long carrier = 0ULL;
    const int fields = sscanf(
        response,
        "OK HELLO %39s %16s %zu %zu %zu %llu",
        hello->protocol,
        topology,
        &hello->width,
        &hello->edges,
        &hello->rounds,
        &carrier
    );
    errno = 0;
    char *tail = NULL;
    const unsigned long long parsed = strtoull(topology, &tail, 16);
    hello->topology_hash = (uint64_t)parsed;
    hello->carrier_creation_count = (uint64_t)carrier;
    return (
        fields == 6
        && strcmp(
            hello->protocol, "CATVM_NONLINEAR_PHASE_GRAPH_1"
        ) == 0
        && errno == 0
        && tail != topology
        && *tail == '\0'
        && hello->width >= 3U
        && hello->edges >= 2U
        && hello->rounds >= 1U
        && hello->rounds <= 4096U
        && hello->carrier_creation_count == 1U
    );
}

static int cnc_parse_projection(
    const char *response,
    struct cnc_projection *projection
) {
    char boundary[17] = {0};
    unsigned long long generation = 0ULL;
    unsigned long long carrier = 0ULL;
    const int fields = sscanf(
        response,
        "OK FINAL___ %d %llu %16s %lf %llu",
        &projection->program,
        &generation,
        boundary,
        &projection->probability,
        &carrier
    );
    errno = 0;
    char *tail = NULL;
    const unsigned long long parsed = strtoull(boundary, &tail, 16);
    projection->generation = (uint64_t)generation;
    projection->boundary_hash = (uint64_t)parsed;
    projection->carrier_creation_count = (uint64_t)carrier;
    return (
        fields == 5
        && (projection->program == 0 || projection->program == 1)
        && errno == 0
        && tail != boundary
        && *tail == '\0'
        && projection->probability >= 0.0
        && projection->probability <= 1.0
        && projection->carrier_creation_count == 1U
    );
}

static int cnc_peer_pid(int socket_fd, pid_t *pid) {
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

static int cnc_attack_proc(pid_t pid, const char *leaf) {
    char path[96];
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

static int cnc_attack_process_vm(pid_t pid) {
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
    cnc_zero(&local, sizeof(local));
    return result < 0 && (errno == EPERM || errno == EACCES);
}

static int cnc_attack_ptrace(pid_t pid) {
    errno = 0;
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        return errno == EPERM || errno == EACCES;
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    (void)ptrace(PTRACE_DETACH, pid, NULL, NULL);
    return 0;
}

static int cnc_attack_pidfd(pid_t pid) {
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

static int cnc_access(
    int socket_fd,
    struct cnc_access *access
) {
    pid_t pid = 0;
    if (!cnc_peer_pid(socket_fd, &pid)) {
        return 0;
    }
    access->proc_mem_denied = cnc_attack_proc(pid, "mem");
    access->proc_maps_denied = cnc_attack_proc(pid, "maps");
    access->proc_fd_denied = cnc_attack_proc(pid, "fd/0");
    access->process_vm_readv_denied = cnc_attack_process_vm(pid);
    access->ptrace_denied = cnc_attack_ptrace(pid);
    access->pidfd_getfd_denied = cnc_attack_pidfd(pid);
    return (
        access->proc_mem_denied
        && access->proc_maps_denied
        && access->proc_fd_denied
        && access->process_vm_readv_denied
        && access->ptrace_denied
        && access->pidfd_getfd_denied
    );
}

static int cnc_denied(
    struct cnc_transport *transport,
    const char *request,
    const char *expected
) {
    char response[CNC_RESPONSE_CAPACITY] = {0};
    const int accepted = (
        cnc_exchange(transport, request, response)
        && strcmp(response, expected) == 0
    );
    cnc_zero(response, sizeof(response));
    return accepted;
}

static void cnc_print_projection(
    const struct cnc_projection *projection
) {
    printf(
        "{\"program\":%d,\"generation\":%llu,"
        "\"boundary_fnv1a64\":\"%016llx\","
        "\"interference_probability\":%.17g}",
        projection->program,
        (unsigned long long)projection->generation,
        (unsigned long long)projection->boundary_hash,
        projection->probability
    );
}

int main(int argc, char **argv) {
    if (argc != 3) {
        return 2;
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long cycles = strtoul(argv[2], &tail, 10);
    if (
        errno != 0
        || tail == argv[2]
        || *tail != '\0'
        || cycles < 2U
        || cycles > CNC_MAX_CYCLES
    ) {
        return 2;
    }
    struct cnc_transport transport = {
        .socket_fd = cnc_connect(argv[1])
    };
    if (transport.socket_fd < 0) {
        return 2;
    }
    char response[CNC_RESPONSE_CAPACITY] = {0};
    struct cnc_hello hello = {0};
    struct cnc_access access = {0};
    if (
        !cnc_exchange(&transport, "HELLO", response)
        || !cnc_parse_hello(response, &hello)
        || !cnc_access(transport.socket_fd, &access)
    ) {
        (void)close(transport.socket_fd);
        return 2;
    }
    cnc_zero(response, sizeof(response));
    static const char *const projection_requests[] = {
        "PROJECT CELL 0",
        "PROJECT SOURCE EPOCH",
        "PROJECT TAPE",
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
        if (!cnc_denied(
            &transport,
            projection_requests[index],
            "ERR E_INTERMEDIATE_PROJECTION_DENIED"
        )) {
            (void)close(transport.socket_fd);
            return 2;
        }
    }
    if (
        !cnc_denied(
            &transport, "EXECUTE NULL", "ERR E_PROTOCOL"
        )
        || !cnc_denied(
            &transport, "UNKNOWN", "ERR E_PROTOCOL"
        )
    ) {
        (void)close(transport.socket_fd);
        return 2;
    }

    struct cnc_projection primary = {0};
    struct cnc_projection reuse = {0};
    for (unsigned long cycle = 0U; cycle < cycles; ++cycle) {
        const int program = (int)(cycle % 2U);
        const char *request = program == 0
            ? "EXECUTE 0"
            : "EXECUTE 1";
        struct cnc_projection current = {0};
        if (
            !cnc_exchange(&transport, request, response)
            || !cnc_parse_projection(response, &current)
            || current.program != program
            || current.generation != (uint64_t)cycle + 1U
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
            const struct cnc_projection *expected =
                program == 0 ? &primary : &reuse;
            if (
                current.boundary_hash != expected->boundary_hash
                || fabs(
                    current.probability - expected->probability
                ) > 2.0e-11
            ) {
                (void)close(transport.socket_fd);
                return 2;
            }
        }
        cnc_zero(response, sizeof(response));
        cnc_zero(&current, sizeof(current));
    }
    if (
        !cnc_exchange(&transport, "SHUTDOWN", response)
        || strcmp(response, "OK CLOSED") != 0
    ) {
        (void)close(transport.socket_fd);
        return 2;
    }
    (void)close(transport.socket_fd);
    printf(
        "{\"result\":\"PASS\","
        "\"protocol\":\"%s\","
        "\"topology_fnv1a64\":\"%016llx\","
        "\"width\":%zu,\"edges\":%zu,\"rounds\":%zu,"
        "\"transactions\":%lu,\"carrier_creation_count\":1,"
        "\"same_service_process\":true,"
        "\"same_actual_restored_carrier\":true,"
        "\"actual_inverse\":true,\"snapshot_reload\":false,"
        "\"all_intermediate_projection_requests_denied\":true,"
        "\"null_carrier_request_denied\":true,"
        "\"unknown_command_denied\":true,"
        "\"proc_mem_denied\":%s,\"proc_maps_denied\":%s,"
        "\"proc_fd_denied\":%s,"
        "\"process_vm_readv_denied\":%s,"
        "\"ptrace_denied\":%s,\"pidfd_getfd_denied\":%s,"
        "\"request_packets\":%llu,\"response_packets\":%llu,"
        "\"request_bytes\":%llu,\"response_bytes\":%llu,"
        "\"primary\":",
        hello.protocol,
        (unsigned long long)hello.topology_hash,
        hello.width,
        hello.edges,
        hello.rounds,
        cycles,
        access.proc_mem_denied ? "true" : "false",
        access.proc_maps_denied ? "true" : "false",
        access.proc_fd_denied ? "true" : "false",
        access.process_vm_readv_denied ? "true" : "false",
        access.ptrace_denied ? "true" : "false",
        access.pidfd_getfd_denied ? "true" : "false",
        (unsigned long long)transport.request_packets,
        (unsigned long long)transport.response_packets,
        (unsigned long long)transport.request_bytes,
        (unsigned long long)transport.response_bytes
    );
    cnc_print_projection(&primary);
    printf(",\"reuse\":");
    cnc_print_projection(&reuse);
    printf("}\n");
    cnc_zero(&primary, sizeof(primary));
    cnc_zero(&reuse, sizeof(reuse));
    cnc_zero(&hello, sizeof(hello));
    cnc_zero(&access, sizeof(access));
    cnc_zero(response, sizeof(response));
    return 0;
}
