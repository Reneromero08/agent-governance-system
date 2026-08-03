#define _GNU_SOURCE

/*
 * Unprivileged controller for the Boolean-TT CATVM boundary.
 *
 * This translation unit contains no phase implementation, TT core generator,
 * relation evaluator, reference hash, expected boundary, or answer table.
 */

#include <errno.h>
#include <fcntl.h>
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

#define CBTC_PROTOCOL "CATVM_BOOLEAN_TT_PHASE_1"
#define CBTC_REQUEST_CAPACITY 128U
#define CBTC_RESPONSE_CAPACITY 512U
#define CBTC_MAX_CYCLES 100000U

struct cbtc_transport {
    int socket_fd;
    uint64_t request_packets;
    uint64_t response_packets;
    uint64_t request_bytes;
    uint64_t response_bytes;
};

struct cbtc_hello {
    char protocol[40];
    size_t width;
    uint64_t plan_hash;
    size_t carrier_cells;
    size_t n2;
    size_t n4;
    size_t n8;
    uint64_t phase_ands;
    uint64_t phase_ors;
    uint64_t carrier_reads;
    uint64_t cell_updates;
    size_t final_decodes;
    uint64_t carrier_creation_count;
};

struct cbtc_projection {
    size_t variant;
    uint64_t generation;
    uint64_t plan_hash;
    uint64_t boundary_hash;
    size_t boundary_ones;
    size_t boundary_cells;
    uint64_t carrier_creation_count;
};

struct cbtc_access {
    int proc_mem_denied;
    int proc_maps_denied;
    int proc_fd_denied;
    int process_vm_readv_denied;
    int ptrace_denied;
    int pidfd_getfd_denied;
};

static void cbtc_secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static int cbtc_parse_u64(
    const char *text,
    int base,
    uint64_t *value
) {
    errno = 0;
    char *tail = NULL;
    const unsigned long long parsed = strtoull(text, &tail, base);
    if (errno != 0 || tail == text || *tail != '\0') {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}

static int cbtc_parse_size(const char *text, size_t *value) {
    uint64_t parsed = 0U;
    if (!cbtc_parse_u64(text, 10, &parsed) || parsed > SIZE_MAX) {
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}

static int cbtc_parse_width(const char *text, size_t *width) {
    return
        cbtc_parse_size(text, width)
        && *width >= 4U
        && *width <= 16U
        && text[0] != '0';
}

static int cbtc_connect(const char *path) {
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        return -1;
    }
    const int socket_fd = socket(
        AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0
    );
    if (socket_fd < 0) {
        return -1;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path) + 1U);
    if (
        connect(
            socket_fd,
            (const struct sockaddr *)&address,
            sizeof(address)
        ) != 0
    ) {
        (void)close(socket_fd);
        return -1;
    }
    return socket_fd;
}

static int cbtc_exchange_bytes(
    struct cbtc_transport *transport,
    const void *request,
    size_t request_bytes,
    char response[CBTC_RESPONSE_CAPACITY]
) {
    if (
        request_bytes == 0U
        || request_bytes > CBTC_REQUEST_CAPACITY
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
        CBTC_RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= CBTC_RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        cbtc_secure_zero(response, CBTC_RESPONSE_CAPACITY);
        return 0;
    }
    response[received] = '\0';
    ++transport->response_packets;
    transport->response_bytes += (uint64_t)received;
    return 1;
}

static int cbtc_exchange(
    struct cbtc_transport *transport,
    const char *request,
    char response[CBTC_RESPONSE_CAPACITY]
) {
    return cbtc_exchange_bytes(
        transport, request, strlen(request), response
    );
}

static int cbtc_parse_hello(
    const char *response,
    size_t expected_width,
    struct cbtc_hello *hello
) {
    char plan[17] = {0};
    unsigned long long ands = 0ULL;
    unsigned long long ors = 0ULL;
    unsigned long long reads = 0ULL;
    unsigned long long updates = 0ULL;
    unsigned long long creations = 0ULL;
    const int fields = sscanf(
        response,
        "OK HELLO %39s %zu %16s %zu %zu %zu %zu "
        "%llu %llu %llu %llu %zu %llu",
        hello->protocol,
        &hello->width,
        plan,
        &hello->carrier_cells,
        &hello->n2,
        &hello->n4,
        &hello->n8,
        &ands,
        &ors,
        &reads,
        &updates,
        &hello->final_decodes,
        &creations
    );
    hello->phase_ands = (uint64_t)ands;
    hello->phase_ors = (uint64_t)ors;
    hello->carrier_reads = (uint64_t)reads;
    hello->cell_updates = (uint64_t)updates;
    hello->carrier_creation_count = (uint64_t)creations;
    return
        fields == 13
        && strcmp(hello->protocol, CBTC_PROTOCOL) == 0
        && cbtc_parse_u64(plan, 16, &hello->plan_hash)
        && hello->plan_hash != 0U
        && hello->width == expected_width
        && hello->n2 == 16U * expected_width - 16U
        && hello->n4 == 64U * expected_width - 96U
        && hello->n8 == 256U * expected_width - 448U
        && hello->carrier_cells == 624U * expected_width - 1040U
        && hello->phase_ands
            == 4U * (uint64_t)(hello->n4 + hello->n8)
        && hello->phase_ors
            == 2U * (uint64_t)(hello->n4 + hello->n8)
        && hello->carrier_reads
            == 8U * (uint64_t)hello->n4
                + 11U * (uint64_t)hello->n8
        && hello->cell_updates
            == 6U * (uint64_t)hello->n2
                + 2U * (uint64_t)hello->n4
                + 4U * (uint64_t)hello->n8
        && hello->final_decodes == hello->n8
        && hello->carrier_creation_count == 1U;
}

static int cbtc_parse_projection(
    const char *response,
    struct cbtc_projection *projection
) {
    char kind[16] = {0};
    char plan[17] = {0};
    char boundary[17] = {0};
    unsigned long long generation = 0ULL;
    unsigned long long creations = 0ULL;
    char extra = '\0';
    const int fields = sscanf(
        response,
        "OK %15s %zu %llu %16s %16s %zu %zu %llu%c",
        kind,
        &projection->variant,
        &generation,
        plan,
        boundary,
        &projection->boundary_ones,
        &projection->boundary_cells,
        &creations,
        &extra
    );
    projection->generation = (uint64_t)generation;
    projection->carrier_creation_count = (uint64_t)creations;
    return
        fields == 8
        && strcmp(kind, "FINAL") == 0
        && cbtc_parse_u64(plan, 16, &projection->plan_hash)
        && cbtc_parse_u64(
            boundary, 16, &projection->boundary_hash
        )
        && projection->boundary_hash != 0U
        && projection->boundary_ones
            <= projection->boundary_cells;
}

static int cbtc_peer_pid(int socket_fd, pid_t *pid) {
    struct ucred credential;
    socklen_t size = sizeof(credential);
    if (
        getsockopt(
            socket_fd,
            SOL_SOCKET,
            SO_PEERCRED,
            &credential,
            &size
        ) != 0
        || size != sizeof(credential)
    ) {
        return 0;
    }
    *pid = credential.pid;
    return *pid > 0;
}

static int cbtc_attack_proc(pid_t pid, const char *leaf) {
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

static int cbtc_attack_process_vm(pid_t pid) {
    unsigned char local = 0U;
    struct iovec local_iov = {.iov_base = &local, .iov_len = 1U};
    struct iovec remote_iov = {
        .iov_base = (void *)(uintptr_t)1U,
        .iov_len = 1U
    };
    errno = 0;
    const ssize_t result = process_vm_readv(
        pid,
        &local_iov,
        1U,
        &remote_iov,
        1U,
        0U
    );
    cbtc_secure_zero(&local, sizeof(local));
    return result < 0 && (errno == EPERM || errno == EACCES);
}

static int cbtc_attack_ptrace(pid_t pid) {
    errno = 0;
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        return errno == EPERM || errno == EACCES;
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    (void)ptrace(PTRACE_DETACH, pid, NULL, NULL);
    return 0;
}

static int cbtc_attack_pidfd_getfd(pid_t pid) {
#if defined(SYS_pidfd_open) && defined(SYS_pidfd_getfd)
    errno = 0;
    const int pidfd = (int)syscall(SYS_pidfd_open, pid, 0U);
    if (pidfd < 0) {
        return
            errno == EPERM
            || errno == EACCES
            || errno == ENOSYS;
    }
    errno = 0;
    const int duplicate = (int)syscall(
        SYS_pidfd_getfd, pidfd, 0, 0U
    );
    const int denied =
        duplicate < 0
        && (
            errno == EPERM
            || errno == EACCES
            || errno == ENOSYS
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

static int cbtc_access_controls(
    int socket_fd,
    struct cbtc_access *controls
) {
    pid_t pid = 0;
    if (!cbtc_peer_pid(socket_fd, &pid)) {
        return 0;
    }
    controls->proc_mem_denied = cbtc_attack_proc(pid, "mem");
    controls->proc_maps_denied = cbtc_attack_proc(pid, "maps");
    controls->proc_fd_denied = cbtc_attack_proc(pid, "fd/0");
    controls->process_vm_readv_denied =
        cbtc_attack_process_vm(pid);
    controls->ptrace_denied = cbtc_attack_ptrace(pid);
    controls->pidfd_getfd_denied =
        cbtc_attack_pidfd_getfd(pid);
    return
        controls->proc_mem_denied
        && controls->proc_maps_denied
        && controls->proc_fd_denied
        && controls->process_vm_readv_denied
        && controls->ptrace_denied
        && controls->pidfd_getfd_denied;
}

static int cbtc_same_projection(
    const struct cbtc_projection *left,
    const struct cbtc_projection *right
) {
    return
        left->boundary_hash == right->boundary_hash
        && left->boundary_ones == right->boundary_ones
        && left->boundary_cells == right->boundary_cells;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        return 2;
    }
    size_t width = 0U;
    size_t cycles = 0U;
    if (
        !cbtc_parse_width(argv[2], &width)
        || !cbtc_parse_size(argv[3], &cycles)
        || cycles > CBTC_MAX_CYCLES
    ) {
        return 2;
    }
    struct cbtc_transport transport = {
        .socket_fd = cbtc_connect(argv[1])
    };
    if (transport.socket_fd < 0) {
        return 2;
    }
    char response[CBTC_RESPONSE_CAPACITY] = {0};
    struct cbtc_hello hello = {0};
    struct cbtc_projection primary = {0};
    struct cbtc_projection reuse = {0};
    struct cbtc_projection current = {0};
    struct cbtc_access access = {0};
    int ok = cbtc_access_controls(
        transport.socket_fd, &access
    );
    ok = ok
        && cbtc_exchange(&transport, "HELLO", response)
        && cbtc_parse_hello(response, width, &hello);

    static const char *const denied[] = {
        "PROJECT F",
        "PROJECT G",
        "PROJECT J",
        "PROJECT H",
        "PROJECT Z",
        "PROJECT BONDS",
        "PROJECT CARRIER",
        "DEBUG",
        "DUMP",
        "READ CARRIER",
        "STATE DETAIL",
        "BOND STATES",
        "WITNESSES"
    };
    for (
        size_t index = 0U;
        ok && index < sizeof(denied) / sizeof(denied[0]);
        ++index
    ) {
        ok = cbtc_exchange(
            &transport, denied[index], response
        ) && strcmp(
            response,
            "ERR E_INTERMEDIATE_PROJECTION_DENIED"
        ) == 0;
    }
    static const unsigned char embedded_nul[] = {
        'H', 'E', 'L', 'L', 'O', '\0', 'X'
    };
    ok = ok
        && cbtc_exchange_bytes(
            &transport,
            embedded_nul,
            sizeof(embedded_nul),
            response
        )
        && strcmp(response, "ERR E_PROTOCOL") == 0;
    unsigned char oversize[CBTC_REQUEST_CAPACITY];
    memset(oversize, 'X', sizeof(oversize));
    ok = ok
        && cbtc_exchange_bytes(
            &transport,
            oversize,
            sizeof(oversize),
            response
        )
        && strcmp(response, "ERR E_PROTOCOL") == 0;
    cbtc_secure_zero(oversize, sizeof(oversize));
    ok = ok
        && cbtc_exchange(&transport, "EXECUTE 2", response)
        && strcmp(response, "ERR E_PROTOCOL") == 0;

    uint64_t expected_generation = 1U;
    ok = ok
        && cbtc_exchange(&transport, "EXECUTE 0", response)
        && cbtc_parse_projection(response, &primary)
        && primary.variant == 0U
        && primary.generation == expected_generation
        && primary.plan_hash == hello.plan_hash
        && primary.boundary_cells == hello.n8
        && primary.carrier_creation_count == 1U;
    ++expected_generation;
    ok = ok
        && cbtc_exchange(&transport, "EXECUTE 1", response)
        && cbtc_parse_projection(response, &reuse)
        && reuse.variant == 1U
        && reuse.generation == expected_generation
        && reuse.plan_hash == hello.plan_hash
        && reuse.boundary_cells == hello.n8
        && reuse.carrier_creation_count == 1U
        && !cbtc_same_projection(&primary, &reuse);

    for (size_t cycle = 0U; ok && cycle < cycles; ++cycle) {
        char command[16];
        const size_t variant = cycle % 2U;
        const int written = snprintf(
            command, sizeof(command), "EXECUTE %zu", variant
        );
        ++expected_generation;
        ok = written > 0
            && (size_t)written < sizeof(command)
            && cbtc_exchange(&transport, command, response)
            && cbtc_parse_projection(response, &current)
            && current.variant == variant
            && current.generation == expected_generation
            && current.plan_hash == hello.plan_hash
            && current.boundary_cells == hello.n8
            && current.carrier_creation_count == 1U
            && cbtc_same_projection(
                variant == 0U ? &primary : &reuse,
                &current
            );
        cbtc_secure_zero(&current, sizeof(current));
    }
    ok = ok
        && cbtc_exchange(&transport, "SHUTDOWN", response)
        && strcmp(response, "OK CLOSED") == 0;
    (void)close(transport.socket_fd);
    if (!ok) {
        return 1;
    }

    printf(
        "{\"result\":\"PASS\","
        "\"protocol\":\"%s\","
        "\"width\":%zu,"
        "\"plan_fnv1a64\":\"%016llx\","
        "\"carrier_cells\":%zu,"
        "\"leaf_cells_each\":%zu,"
        "\"resident_h_cells\":%zu,"
        "\"final_z_cells\":%zu,"
        "\"logical_phase_ands\":%llu,"
        "\"logical_phase_ors\":%llu,"
        "\"carrier_reads\":%llu,"
        "\"phase_cell_updates\":%llu,"
        "\"final_decodes\":%zu,"
        "\"carrier_creation_count\":%llu,"
        "\"transactions\":%zu,"
        "\"request_packets\":%llu,"
        "\"response_packets\":%llu,"
        "\"request_bytes\":%llu,"
        "\"response_bytes\":%llu,"
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"primary_boundary_ones\":%zu,"
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_ones\":%zu,"
        "\"all_projection_requests_denied\":true,"
        "\"embedded_nul_denied\":true,"
        "\"oversize_packet_denied\":true,"
        "\"unknown_command_denied\":true,"
        "\"proc_mem_denied\":%s,"
        "\"proc_maps_denied\":%s,"
        "\"proc_fd_denied\":%s,"
        "\"process_vm_readv_denied\":%s,"
        "\"ptrace_denied\":%s,"
        "\"pidfd_getfd_denied\":%s,"
        "\"actual_inverse\":true,"
        "\"snapshot_reload\":false,"
        "\"same_service_process\":true,"
        "\"same_actual_restored_carrier\":true,"
        "\"controller_phase_core_linked\":false,"
        "\"controller_computes_boundary_independently\":false}\n",
        hello.protocol,
        width,
        (unsigned long long)hello.plan_hash,
        hello.carrier_cells,
        hello.n2,
        hello.n4,
        hello.n8,
        (unsigned long long)hello.phase_ands,
        (unsigned long long)hello.phase_ors,
        (unsigned long long)hello.carrier_reads,
        (unsigned long long)hello.cell_updates,
        hello.final_decodes,
        (unsigned long long)hello.carrier_creation_count,
        cycles + 2U,
        (unsigned long long)transport.request_packets,
        (unsigned long long)transport.response_packets,
        (unsigned long long)transport.request_bytes,
        (unsigned long long)transport.response_bytes,
        (unsigned long long)primary.boundary_hash,
        primary.boundary_ones,
        (unsigned long long)reuse.boundary_hash,
        reuse.boundary_ones,
        access.proc_mem_denied ? "true" : "false",
        access.proc_maps_denied ? "true" : "false",
        access.proc_fd_denied ? "true" : "false",
        access.process_vm_readv_denied ? "true" : "false",
        access.ptrace_denied ? "true" : "false",
        access.pidfd_getfd_denied ? "true" : "false"
    );
    cbtc_secure_zero(response, sizeof(response));
    cbtc_secure_zero(&primary, sizeof(primary));
    cbtc_secure_zero(&reuse, sizeof(reuse));
    cbtc_secure_zero(&current, sizeof(current));
    return 0;
}
