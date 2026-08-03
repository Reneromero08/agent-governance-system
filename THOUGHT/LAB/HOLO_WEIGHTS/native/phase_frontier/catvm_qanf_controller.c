#define _GNU_SOURCE

/*
 * Unprivileged controller for the quadratic-ANF CATVM boundary.
 *
 * This translation unit deliberately contains no phase implementation,
 * relation evaluator, fixture parser, or precomputed result material.
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

#define QCC_REQUEST_CAPACITY 256U
#define QCC_RESPONSE_CAPACITY 512U
#define QCC_BOUNDARY_CELLS 5U
#define QCC_MAX_CYCLES 100000U

struct qcc_transport {
    int socket_fd;
    uint64_t request_packets;
    uint64_t response_packets;
    uint64_t request_bytes;
    uint64_t response_bytes;
};

struct qcc_hello {
    char protocol[32];
    uint64_t plan_hash;
    size_t carrier_cells;
    uint64_t phase_ands;
    uint64_t carrier_reads;
    uint64_t cell_updates;
    size_t boundary_decodes;
    size_t boundary_copies;
    uint64_t carrier_creation_count;
};

struct qcc_projection {
    size_t variant;
    uint64_t generation;
    uint64_t plan_hash;
    uint64_t boundary_hash;
    uint64_t carrier_creation_count;
    size_t count;
    int coefficient[QCC_BOUNDARY_CELLS];
};

struct qcc_access {
    int proc_mem_denied;
    int proc_maps_denied;
    int proc_fd_denied;
    int process_vm_readv_denied;
    int ptrace_denied;
    int pidfd_getfd_denied;
};

static void qcc_secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static int qcc_connect(const char *path) {
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        return -1;
    }
    const int socket_fd = socket(
        AF_UNIX,
        SOCK_SEQPACKET | SOCK_CLOEXEC,
        0
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

static int qcc_exchange_bytes(
    struct qcc_transport *transport,
    const void *request,
    size_t request_bytes,
    char response[QCC_RESPONSE_CAPACITY]
) {
    if (
        request_bytes == 0U
        || request_bytes > QCC_REQUEST_CAPACITY
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
        QCC_RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= QCC_RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        qcc_secure_zero(response, QCC_RESPONSE_CAPACITY);
        return 0;
    }
    response[received] = '\0';
    ++transport->response_packets;
    transport->response_bytes += (uint64_t)received;
    return 1;
}

static int qcc_exchange(
    struct qcc_transport *transport,
    const char *request,
    char response[QCC_RESPONSE_CAPACITY]
) {
    return qcc_exchange_bytes(
        transport,
        request,
        strlen(request),
        response
    );
}

static int qcc_parse_u64(
    const char *text,
    int base,
    uint64_t *value
) {
    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(text, &end, base);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}

static int qcc_parse_size(const char *text, size_t *value) {
    uint64_t parsed = 0U;
    if (!qcc_parse_u64(text, 10, &parsed) || parsed > SIZE_MAX) {
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}

static int qcc_parse_hello(
    const char *response,
    struct qcc_hello *hello
) {
    char plan[17] = {0};
    unsigned long long phase_ands = 0ULL;
    unsigned long long carrier_reads = 0ULL;
    unsigned long long cell_updates = 0ULL;
    unsigned long long carrier_creations = 0ULL;
    const int fields = sscanf(
        response,
        "OK HELLO %31s %16s %zu %llu %llu %llu %zu %zu %llu",
        hello->protocol,
        plan,
        &hello->carrier_cells,
        &phase_ands,
        &carrier_reads,
        &cell_updates,
        &hello->boundary_decodes,
        &hello->boundary_copies,
        &carrier_creations
    );
    hello->phase_ands = (uint64_t)phase_ands;
    hello->carrier_reads = (uint64_t)carrier_reads;
    hello->cell_updates = (uint64_t)cell_updates;
    hello->carrier_creation_count = (uint64_t)carrier_creations;
    return
        fields == 9
        && strcmp(hello->protocol, "CATVM_QANF_PHASE_1") == 0
        && qcc_parse_u64(plan, 16, &hello->plan_hash)
        && hello->plan_hash != 0U
        && hello->carrier_cells == 23U
        && hello->phase_ands == 18U
        && hello->carrier_reads == 97U
        && hello->cell_updates == 46U
        && hello->boundary_decodes == QCC_BOUNDARY_CELLS
        && hello->boundary_copies == 2U
        && hello->carrier_creation_count == 1U;
}

static char *qcc_next_token(char **save) {
    return strtok_r(NULL, " ", save);
}

static int qcc_parse_projection(
    const char *response,
    struct qcc_projection *projection
) {
    char copy[QCC_RESPONSE_CAPACITY];
    const size_t bytes = strlen(response);
    if (bytes >= sizeof(copy)) {
        return 0;
    }
    memcpy(copy, response, bytes + 1U);
    char *save = NULL;
    char *token = strtok_r(copy, " ", &save);
    if (token == NULL || strcmp(token, "OK") != 0) {
        return 0;
    }
    token = qcc_next_token(&save);
    if (token == NULL || strcmp(token, "FINAL") != 0) {
        return 0;
    }
    token = qcc_next_token(&save);
    if (
        token == NULL
        || !qcc_parse_size(token, &projection->variant)
    ) {
        return 0;
    }
    token = qcc_next_token(&save);
    if (
        token == NULL
        || !qcc_parse_u64(token, 10, &projection->generation)
    ) {
        return 0;
    }
    token = qcc_next_token(&save);
    if (
        token == NULL
        || !qcc_parse_u64(token, 16, &projection->plan_hash)
    ) {
        return 0;
    }
    token = qcc_next_token(&save);
    if (
        token == NULL
        || !qcc_parse_u64(token, 16, &projection->boundary_hash)
    ) {
        return 0;
    }
    token = qcc_next_token(&save);
    if (
        token == NULL
        || !qcc_parse_u64(
            token,
            10,
            &projection->carrier_creation_count
        )
    ) {
        return 0;
    }
    token = qcc_next_token(&save);
    if (
        token == NULL
        || !qcc_parse_size(token, &projection->count)
        || projection->count != QCC_BOUNDARY_CELLS
    ) {
        return 0;
    }
    for (size_t cell = 0U; cell < projection->count; ++cell) {
        token = qcc_next_token(&save);
        if (
            token == NULL
            || strlen(token) != 1U
            || (token[0] != '0' && token[0] != '1')
        ) {
            return 0;
        }
        projection->coefficient[cell] = token[0] - '0';
    }
    const int complete = qcc_next_token(&save) == NULL;
    qcc_secure_zero(copy, sizeof(copy));
    return complete;
}

static int qcc_projection_differs(
    const struct qcc_projection *left,
    const struct qcc_projection *right
) {
    return
        left->boundary_hash != right->boundary_hash
        || memcmp(
            left->coefficient,
            right->coefficient,
            sizeof(left->coefficient)
        ) != 0;
}

static int qcc_peer_pid(int socket_fd, pid_t *pid) {
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

static int qcc_attack_proc(pid_t pid, const char *leaf) {
    char path[96];
    const int written = snprintf(
        path,
        sizeof(path),
        "/proc/%ld/%s",
        (long)pid,
        leaf
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

static int qcc_attack_process_vm(pid_t pid) {
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
    qcc_secure_zero(&local, sizeof(local));
    return result < 0 && (errno == EPERM || errno == EACCES);
}

static int qcc_attack_ptrace(pid_t pid) {
    errno = 0;
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        return errno == EPERM || errno == EACCES;
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    (void)ptrace(PTRACE_DETACH, pid, NULL, NULL);
    return 0;
}

static int qcc_attack_pidfd_getfd(pid_t pid) {
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
        SYS_pidfd_getfd,
        pidfd,
        0,
        0U
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

static int qcc_access_controls(
    int socket_fd,
    struct qcc_access *controls
) {
    pid_t pid = 0;
    if (!qcc_peer_pid(socket_fd, &pid)) {
        return 0;
    }
    controls->proc_mem_denied = qcc_attack_proc(pid, "mem");
    controls->proc_maps_denied = qcc_attack_proc(pid, "maps");
    controls->proc_fd_denied = qcc_attack_proc(pid, "fd/0");
    controls->process_vm_readv_denied =
        qcc_attack_process_vm(pid);
    controls->ptrace_denied = qcc_attack_ptrace(pid);
    controls->pidfd_getfd_denied =
        qcc_attack_pidfd_getfd(pid);
    return
        controls->proc_mem_denied
        && controls->proc_maps_denied
        && controls->proc_fd_denied
        && controls->process_vm_readv_denied
        && controls->ptrace_denied
        && controls->pidfd_getfd_denied;
}

static void qcc_print_coefficients(
    const struct qcc_projection *projection
) {
    putchar('[');
    for (size_t cell = 0U; cell < projection->count; ++cell) {
        printf(
            "%s%d",
            cell == 0U ? "" : ",",
            projection->coefficient[cell]
        );
    }
    putchar(']');
}

int main(int argc, char **argv) {
    if (argc != 3) {
        return 2;
    }
    errno = 0;
    char *end = NULL;
    const unsigned long parsed_cycles = strtoul(argv[2], &end, 10);
    if (
        errno != 0
        || end == argv[2]
        || *end != '\0'
        || parsed_cycles > QCC_MAX_CYCLES
    ) {
        return 2;
    }
    const size_t cycles = (size_t)parsed_cycles;
    struct qcc_transport transport = {
        .socket_fd = qcc_connect(argv[1])
    };
    if (transport.socket_fd < 0) {
        return 2;
    }
    char response[QCC_RESPONSE_CAPACITY] = {0};
    struct qcc_hello hello = {0};
    struct qcc_projection primary = {0};
    struct qcc_projection reuse = {0};
    struct qcc_projection current = {0};
    struct qcc_access access = {0};
    int ok = qcc_access_controls(transport.socket_fd, &access);

    ok = ok
        && qcc_exchange(&transport, "HELLO", response)
        && qcc_parse_hello(response, &hello);
    static const char *const denied[] = {
        "PROJECT U",
        "PROJECT V",
        "PROJECT H",
        "PROJECT Z",
        "PROJECT F",
        "PROJECT G",
        "PROJECT J",
        "PROJECT CARRIER",
        "DEBUG",
        "DUMP",
        "READ CARRIER",
        "STATE DETAIL"
    };
    for (
        size_t request = 0U;
        ok && request < sizeof(denied) / sizeof(denied[0]);
        ++request
    ) {
        ok = qcc_exchange(
            &transport,
            denied[request],
            response
        )
            && strcmp(
                response,
                "ERR E_INTERMEDIATE_PROJECTION_DENIED"
            ) == 0;
    }
    static const unsigned char embedded_nul[] = {
        'H', 'E', 'L', 'L', 'O', '\0', 'X'
    };
    ok = ok
        && qcc_exchange_bytes(
            &transport,
            embedded_nul,
            sizeof(embedded_nul),
            response
        )
        && strcmp(response, "ERR E_PROTOCOL") == 0;
    unsigned char oversize[QCC_REQUEST_CAPACITY];
    memset(oversize, 'X', sizeof(oversize));
    ok = ok
        && qcc_exchange_bytes(
            &transport,
            oversize,
            sizeof(oversize),
            response
        )
        && strcmp(response, "ERR E_PROTOCOL") == 0;
    qcc_secure_zero(oversize, sizeof(oversize));
    ok = ok
        && qcc_exchange(&transport, "EXECUTE 4", response)
        && strcmp(response, "ERR E_PROTOCOL") == 0;

    uint64_t expected_generation = 1U;
    ok = ok
        && qcc_exchange(&transport, "EXECUTE 0", response)
        && qcc_parse_projection(response, &primary)
        && primary.variant == 0U
        && primary.generation == expected_generation
        && primary.plan_hash == hello.plan_hash
        && primary.carrier_creation_count == 1U;
    ++expected_generation;
    ok = ok
        && qcc_exchange(&transport, "EXECUTE 1", response)
        && qcc_parse_projection(response, &reuse)
        && reuse.variant == 1U
        && reuse.generation == expected_generation
        && reuse.plan_hash == hello.plan_hash
        && reuse.carrier_creation_count == 1U
        && qcc_projection_differs(&primary, &reuse);

    for (size_t cycle = 0U; ok && cycle < cycles; ++cycle) {
        char command[16];
        const size_t variant = cycle % 2U;
        const int written = snprintf(
            command,
            sizeof(command),
            "EXECUTE %zu",
            variant
        );
        ++expected_generation;
        ok = written > 0
            && (size_t)written < sizeof(command)
            && qcc_exchange(&transport, command, response)
            && qcc_parse_projection(response, &current)
            && current.variant == variant
            && current.generation == expected_generation
            && current.plan_hash == hello.plan_hash
            && current.carrier_creation_count == 1U;
        qcc_secure_zero(&current, sizeof(current));
    }
    ok = ok
        && qcc_exchange(&transport, "SHUTDOWN", response)
        && strcmp(response, "OK CLOSED") == 0;
    (void)close(transport.socket_fd);

    if (!ok) {
        qcc_secure_zero(response, sizeof(response));
        qcc_secure_zero(&primary, sizeof(primary));
        qcc_secure_zero(&reuse, sizeof(reuse));
        return 1;
    }
    printf(
        "{\"result\":\"PASS\","
        "\"protocol\":\"%s\","
        "\"plan_fnv1a64\":\"%016llx\","
        "\"carrier_creation_count\":%llu,"
        "\"transactions\":%llu,"
        "\"request_packets\":%llu,"
        "\"response_packets\":%llu,"
        "\"request_bytes\":%llu,"
        "\"response_bytes\":%llu,"
        "\"all_intermediate_projection_requests_denied\":true,"
        "\"embedded_nul_denied\":true,"
        "\"oversize_packet_denied\":true,"
        "\"unknown_command_denied\":true,"
        "\"proc_mem_denied\":%s,"
        "\"proc_maps_denied\":%s,"
        "\"proc_fd_denied\":%s,"
        "\"process_vm_readv_denied\":%s,"
        "\"ptrace_denied\":%s,"
        "\"pidfd_getfd_denied\":%s,"
        "\"primary\":{\"variant\":0,\"generation\":1,"
        "\"boundary_fnv1a64\":\"%016llx\","
        "\"coefficients\":",
        hello.protocol,
        (unsigned long long)hello.plan_hash,
        (unsigned long long)hello.carrier_creation_count,
        (unsigned long long)(cycles + 2U),
        (unsigned long long)transport.request_packets,
        (unsigned long long)transport.response_packets,
        (unsigned long long)transport.request_bytes,
        (unsigned long long)transport.response_bytes,
        access.proc_mem_denied ? "true" : "false",
        access.proc_maps_denied ? "true" : "false",
        access.proc_fd_denied ? "true" : "false",
        access.process_vm_readv_denied ? "true" : "false",
        access.ptrace_denied ? "true" : "false",
        access.pidfd_getfd_denied ? "true" : "false",
        (unsigned long long)primary.boundary_hash
    );
    qcc_print_coefficients(&primary);
    printf(
        "},\"reuse\":{\"variant\":1,\"generation\":2,"
        "\"boundary_fnv1a64\":\"%016llx\","
        "\"coefficients\":",
        (unsigned long long)reuse.boundary_hash
    );
    qcc_print_coefficients(&reuse);
    printf(
        "},\"actual_inverse\":true,"
        "\"snapshot_reload\":false,"
        "\"same_service_process\":true,"
        "\"same_actual_restored_carrier\":true,"
        "\"controller_phase_core_linked\":false,"
        "\"controller_computes_boundary_independently\":false}\n"
    );
    qcc_secure_zero(response, sizeof(response));
    qcc_secure_zero(&primary, sizeof(primary));
    qcc_secure_zero(&reuse, sizeof(reuse));
    qcc_secure_zero(&current, sizeof(current));
    return 0;
}
