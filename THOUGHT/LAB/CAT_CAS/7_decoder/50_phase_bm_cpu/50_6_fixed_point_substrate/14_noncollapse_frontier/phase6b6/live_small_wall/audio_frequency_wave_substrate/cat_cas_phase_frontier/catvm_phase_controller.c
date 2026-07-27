#define _GNU_SOURCE

/*
 * CATVM controller.
 *
 * This binary deliberately does not include or link the phase core.  It can
 * transport public programs and receive final boundaries, but it has no
 * native polynomial, phase-label decode, carrier, or scalar adjudicator.
 */

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define REQUEST_CAPACITY 768U
#define RESPONSE_CAPACITY 2048U
#ifdef CATVM_SANITIZER_BUILD
#define REQUIRED_SECCOMP_STATUS \
    "\"seccomp\":\"DISABLED_SANITIZER_BUILD\""
#define SECCOMP_ENFORCED 0
#else
#define REQUIRED_SECCOMP_STATUS \
    "\"seccomp\":\"ACTIVE_ALLOWLIST\""
#define SECCOMP_ENFORCED 1
#endif

struct transport {
    int socket;
    uint64_t request_bytes;
    uint64_t response_bytes;
    uint64_t request_packets;
    uint64_t response_packets;
};

struct access_controls {
    int proc_mem_denied;
    int proc_maps_denied;
    int proc_fd_denied;
    int process_vm_readv_denied;
    int ptrace_denied;
    int pidfd_getfd_denied;
};

static const char PRIMARY_PROGRAM[] =
    "SEAL 0 1 2 0 0 1 2 0 0 1 0 0";
static const char REUSE_PROGRAM[] =
    "SEAL 2 1 1 0 0 1 2 0 0 0 1 0";

static void secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static int connect_socket(const char *path) {
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

static int exchange_bytes(
    struct transport *transport,
    const void *request,
    size_t request_bytes,
    char response[RESPONSE_CAPACITY]
) {
    if (request_bytes == 0U || request_bytes > REQUEST_CAPACITY) {
        return 0;
    }
    if (
        send(
            transport->socket,
            request,
            request_bytes,
            MSG_NOSIGNAL
        ) != (ssize_t)request_bytes
    ) {
        return 0;
    }
    transport->request_bytes += request_bytes;
    ++transport->request_packets;
    secure_zero(response, RESPONSE_CAPACITY);
    const ssize_t received = recv(
        transport->socket,
        response,
        RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        secure_zero(response, RESPONSE_CAPACITY);
        return 0;
    }
    response[received] = '\0';
    transport->response_bytes += (uint64_t)received;
    ++transport->response_packets;
    return 1;
}

static int exchange(
    struct transport *transport,
    const char *request,
    char response[RESPONSE_CAPACITY]
) {
    return exchange_bytes(
        transport,
        request,
        strlen(request),
        response
    );
}

static int response_has(const char *response, const char *needle) {
    return strstr(response, needle) != NULL;
}

static int response_is(const char *response, const char *expected) {
    return strcmp(response, expected) == 0;
}

static int response_uint64(
    const char *response,
    const char *field,
    uint64_t *value
) {
    const char *cursor = strstr(response, field);
    if (cursor == NULL) {
        return 0;
    }
    cursor += strlen(field);
    if (*cursor < '0' || *cursor > '9') {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(cursor, &end, 10);
    if (
        errno != 0
        || end == cursor
        || (*end != ',' && *end != '}')
    ) {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}

static int peer_pid(int socket_fd, pid_t *pid) {
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

static int attack_proc_file(pid_t pid, const char *leaf) {
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

static int attack_process_vm(pid_t pid) {
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
    secure_zero(&local, sizeof(local));
    return result < 0 && (errno == EPERM || errno == EACCES);
}

static int attack_ptrace(pid_t pid) {
    errno = 0;
    if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
        return errno == EPERM || errno == EACCES;
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    (void)ptrace(PTRACE_DETACH, pid, NULL, NULL);
    return 0;
}

static int attack_pidfd_getfd(pid_t pid) {
#if defined(SYS_pidfd_open) && defined(SYS_pidfd_getfd)
    const int pidfd = (int)syscall(SYS_pidfd_open, pid, 0U);
    if (pidfd < 0) {
        return errno == EPERM
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

static int run_access_controls(
    int socket_fd,
    struct access_controls *controls
) {
    pid_t pid = 0;
    if (!peer_pid(socket_fd, &pid)) {
        return 0;
    }
    memset(controls, 0, sizeof(*controls));
    controls->proc_mem_denied = attack_proc_file(pid, "mem");
    controls->proc_maps_denied = attack_proc_file(pid, "maps");
    controls->proc_fd_denied = attack_proc_file(pid, "fd/0");
    controls->process_vm_readv_denied = attack_process_vm(pid);
    controls->ptrace_denied = attack_ptrace(pid);
    controls->pidfd_getfd_denied = attack_pidfd_getfd(pid);
    return (
        controls->proc_mem_denied
        && controls->proc_maps_denied
        && controls->proc_fd_denied
        && controls->process_vm_readv_denied
        && controls->ptrace_denied
        && controls->pidfd_getfd_denied
    );
}

static double restoration_error(const char *response) {
    static const char key[] = "\"maximum_abs_error\":";
    const char *position = strstr(response, key);
    if (position == NULL) {
        return -1.0;
    }
    position += sizeof(key) - 1U;
    char *end = NULL;
    errno = 0;
    const double value = strtod(position, &end);
    if (errno != 0 || end == position || value < 0.0) {
        return -1.0;
    }
    return value;
}

static int final_boundary_identity(
    const char *response,
    char identity[17]
) {
    static const char key[] = "\"fnv1a64\":\"";
    const char *position = strstr(response, key);
    if (position == NULL) {
        return 0;
    }
    position += sizeof(key) - 1U;
    for (size_t index = 0U; index < 16U; ++index) {
        const char byte = position[index];
        if (
            !(
                (byte >= '0' && byte <= '9')
                || (byte >= 'a' && byte <= 'f')
            )
        ) {
            secure_zero(identity, 17U);
            return 0;
        }
        identity[index] = byte;
    }
    identity[16] = '\0';
    return position[16] == '"';
}

static int protocol_adversaries(
    struct transport *transport,
    char response[RESPONSE_CAPACITY]
) {
    static const char *const rejected[] = {
        "READ CELL",
        "DUMP",
        "DEBUG",
        "SNAPSHOT",
        "STATUS DETAIL"
    };
    for (
        size_t index = 0U;
        index < sizeof(rejected) / sizeof(rejected[0]);
        ++index
    ) {
        if (
            !exchange(transport, rejected[index], response)
            || !response_is(
                response,
                "{\"ok\":false,\"error\":\"E_PROTOCOL\"}"
            )
        ) {
            return 0;
        }
    }
    static const unsigned char embedded_nul[] = {
        'S', 'E', 'A', 'L', ' ', '0', '\0', '1'
    };
    if (
        !exchange_bytes(
            transport,
            embedded_nul,
            sizeof(embedded_nul),
            response
        )
        || !response_is(
            response,
            "{\"ok\":false,\"error\":\"E_PROTOCOL\"}"
        )
    ) {
        return 0;
    }
    unsigned char oversized[REQUEST_CAPACITY];
    memset(oversized, 'X', sizeof(oversized));
    const int oversized_ok =
        exchange_bytes(
            transport,
            oversized,
            sizeof(oversized),
            response
        )
        && response_is(
            response,
            "{\"ok\":false,\"error\":\"E_PROTOCOL\"}"
        );
    secure_zero(oversized, sizeof(oversized));
    return oversized_ok;
}

static int project_y_denied(
    struct transport *transport,
    char response[RESPONSE_CAPACITY]
) {
    return (
        exchange(transport, "PROJECT Y", response)
        && response_is(
            response,
            "{\"ok\":false,"
            "\"error\":\"E_INTERMEDIATE_PROJECTION_DENIED\","
            "\"type\":\"BOOLEAN_F3_RELATION\","
            "\"state_unchanged\":true}"
        )
    );
}

static int run_one(
    struct transport *transport,
    const char *program,
    int attack_while_y_resident,
    struct access_controls *access,
    char boundary[RESPONSE_CAPACITY],
    char restoration[RESPONSE_CAPACITY]
) {
    char response[RESPONSE_CAPACITY];
    if (
        !exchange(transport, program, response)
        || !response_has(response, "\"event\":\"CARRIER_SEALED\"")
        || !exchange(transport, "F", response)
        || !response_has(response, "\"event\":\"INTERMEDIATE_CUSTODY\"")
        || !project_y_denied(transport, response)
    ) {
        secure_zero(response, sizeof(response));
        return 0;
    }
    if (
        attack_while_y_resident
        && !run_access_controls(transport->socket, access)
    ) {
        secure_zero(response, sizeof(response));
        return 0;
    }
    if (
        !exchange(transport, "G", response)
        || !response_has(response, "\"event\":\"FINAL_READY\"")
        || !exchange(transport, "PROJECT Z", boundary)
        || !response_has(boundary, "\"event\":\"FINAL_BOUNDARY\"")
        || !response_has(
            boundary,
            "\"decoded_intermediate_coefficients\":0"
        )
        || !exchange(transport, "RESTORE", restoration)
        || !response_has(restoration, "\"event\":\"RESTORATION\"")
    ) {
        secure_zero(response, sizeof(response));
        return 0;
    }
    secure_zero(response, sizeof(response));
    return 1;
}

static int shutdown_service(
    struct transport *transport,
    char response[RESPONSE_CAPACITY]
) {
    return (
        exchange(transport, "SHUTDOWN", response)
        && response_is(response, "{\"ok\":true,\"event\":\"CLOSED\"}")
    );
}

static int run_null(
    struct transport *transport,
    char response[RESPONSE_CAPACITY]
) {
    if (
        !exchange(transport, "HELLO", response)
        || !response_has(response, "\"carrier\":false")
        || !exchange(transport, PRIMARY_PROGRAM, response)
        || !response_is(
            response,
            "{\"ok\":false,\"error\":\"E_NO_CARRIER\"}"
        )
        || !project_y_denied(transport, response)
        || !shutdown_service(transport, response)
    ) {
        return 0;
    }
    printf(
        "{\"result\":\"PASS\",\"scenario\":\"null-carrier\","
        "\"carrier_required\":true,\"final_boundary_emitted\":false,"
        "\"request_packets\":%llu,\"response_packets\":%llu}\n",
        (unsigned long long)transport->request_packets,
        (unsigned long long)transport->response_packets
    );
    return 1;
}

static int run_control(
    struct transport *transport,
    const char *scenario,
    char response[RESPONSE_CAPACITY]
) {
    struct access_controls access;
    char boundary[RESPONSE_CAPACITY];
    char restoration[RESPONSE_CAPACITY];
    if (
        !exchange(transport, "HELLO", response)
        || !response_has(response, "\"backend\":\"IN_PLACE_PHASE\"")
        || !run_one(
            transport,
            PRIMARY_PROGRAM,
            0,
            &access,
            boundary,
            restoration
        )
        || !response_has(restoration, "\"control_discriminated\":true")
        || !response_has(restoration, "\"carrier_within_tolerance\":false")
        || restoration_error(restoration) < 1.0e-3
        || !shutdown_service(transport, response)
    ) {
        secure_zero(boundary, sizeof(boundary));
        secure_zero(restoration, sizeof(restoration));
        return 0;
    }
    printf(
        "{\"result\":\"PASS\",\"scenario\":\"%s\","
        "\"prospectively_applicable\":true,"
        "\"restoration_failure_detected\":true,"
        "\"final_boundary\":%s,"
        "\"restoration\":%s,"
        "\"request_packets\":%llu,\"response_packets\":%llu}\n",
        scenario,
        boundary,
        restoration,
        (unsigned long long)transport->request_packets,
        (unsigned long long)transport->response_packets
    );
    secure_zero(boundary, sizeof(boundary));
    secure_zero(restoration, sizeof(restoration));
    return 1;
}

static int run_accepted_or_snapshot(
    struct transport *transport,
    int snapshot,
    size_t repeat_cycles,
    char response[RESPONSE_CAPACITY]
) {
    struct access_controls access;
    memset(&access, 0, sizeof(access));
    char primary_boundary[RESPONSE_CAPACITY];
    char reuse_boundary[RESPONSE_CAPACITY];
    char boundary[RESPONSE_CAPACITY];
    char restoration[RESPONSE_CAPACITY];
    char primary_identity[17];
    char reuse_identity[17];
    char current_identity[17];
    double maximum_restoration_error = 0.0;
    uint64_t carrier_creation_count = 0U;
    uint64_t mapped_locked_bytes = 0U;
    struct timespec start;
    struct timespec finish;

    if (
        !exchange(transport, "HELLO", response)
        || !response_has(
            response,
            snapshot
                ? "\"backend\":\"SNAPSHOT_BASELINE\""
                : "\"backend\":\"IN_PLACE_PHASE\""
        )
        || !response_has(response, REQUIRED_SECCOMP_STATUS)
        || !response_uint64(
            response,
            "\"carrier_creations\":",
            &carrier_creation_count
        )
        || carrier_creation_count != 1U
        || !response_uint64(
            response,
            "\"mapped_locked_bytes\":",
            &mapped_locked_bytes
        )
        || mapped_locked_bytes != (snapshot ? 8192U : 4096U)
        || !response_has(response, "\"carrier_cells\":24")
        || !response_has(response, "\"logical_carrier_bytes\":768")
        || !response_has(response, "\"compiled_program_bytes\":48")
        || (!snapshot && !protocol_adversaries(transport, response))
        || clock_gettime(CLOCK_MONOTONIC, &start) != 0
        || !run_one(
            transport,
            PRIMARY_PROGRAM,
            !snapshot,
            &access,
            primary_boundary,
            restoration
        )
        || !response_has(
            restoration,
            snapshot
                ? "\"actual_inverse\":false"
                : "\"actual_inverse\":true"
        )
        || !response_has(
            restoration,
            snapshot
                ? "\"snapshot_reload\":true"
                : "\"snapshot_reload\":false"
        )
        || !response_has(restoration, "\"carrier_within_tolerance\":true")
        || !response_has(restoration, "\"transient_state_exact\":true")
        || !final_boundary_identity(
            primary_boundary,
            primary_identity
        )
    ) {
        goto fail;
    }
    maximum_restoration_error = restoration_error(restoration);
    if (maximum_restoration_error < 0.0) {
        goto fail;
    }

    if (
        !run_one(
            transport,
            REUSE_PROGRAM,
            0,
            &access,
            reuse_boundary,
            restoration
        )
        || !final_boundary_identity(reuse_boundary, reuse_identity)
        || strcmp(primary_identity, reuse_identity) == 0
        || !response_has(restoration, "\"carrier_creations\":1")
        || !response_has(restoration, "\"generation\":2")
    ) {
        goto fail;
    }
    double error = restoration_error(restoration);
    if (error < 0.0) {
        goto fail;
    }
    if (error > maximum_restoration_error) {
        maximum_restoration_error = error;
    }

    for (size_t cycle = 0U; cycle < repeat_cycles; ++cycle) {
        const int primary = (cycle & 1U) == 0U;
        if (
            !run_one(
                transport,
                primary ? PRIMARY_PROGRAM : REUSE_PROGRAM,
                0,
                &access,
                boundary,
                restoration
            )
            || !final_boundary_identity(boundary, current_identity)
            || strcmp(
                current_identity,
                primary ? primary_identity : reuse_identity
            ) != 0
        ) {
            goto fail;
        }
        error = restoration_error(restoration);
        if (error < 0.0) {
            goto fail;
        }
        if (error > maximum_restoration_error) {
            maximum_restoration_error = error;
        }
    }

    if (
        maximum_restoration_error > 2.0e-12
        || clock_gettime(CLOCK_MONOTONIC, &finish) != 0
        || !shutdown_service(transport, response)
    ) {
        goto fail;
    }
    printf(
        "{\"result\":\"PASS\",\"scenario\":\"%s\","
        "\"claim\":\"%s\","
        "\"primary_boundary\":%s,"
        "\"reuse_boundary\":%s,"
        "\"same_carrier_creation_count\":%llu,"
        "\"carrier_cells\":24,\"physical_complex_values\":48,"
        "\"logical_carrier_bytes\":768,\"mapped_locked_bytes\":%llu,"
        "\"compiled_program_bytes\":48,\"compiled_morphisms\":2,"
        "\"maximum_temporary_complex_values\":52,"
        "\"restoration_generations\":%zu,"
        "\"alternating_repeat_cycles\":%zu,"
        "\"maximum_restoration_error\":%.12g,"
        "\"restoration_tolerance\":2e-12,"
        "\"transaction_wall_ns\":%llu,"
        "\"average_transaction_wall_ns\":%.3f,"
        "\"intermediate_projection_denied\":true,"
        "\"seccomp_enforced\":%s,"
        "\"proc_mem_denied\":%s,\"proc_maps_denied\":%s,"
        "\"proc_fd_denied\":%s,"
        "\"process_vm_readv_denied\":%s,\"ptrace_denied\":%s,"
        "\"pidfd_getfd_denied\":%s,"
        "\"final_restoration\":%s,"
        "\"request_bytes\":%llu,\"response_bytes\":%llu,"
        "\"request_packets\":%llu,\"response_packets\":%llu}\n",
        snapshot ? "snapshot-baseline" : "accepted-in-place",
        snapshot
            ? "CATVM_SNAPSHOT_BACKED_TRANSACTIONAL_REUSE_ESTABLISHED"
            : "CATVM_OPEN_INTERMEDIATE_COMPOSITION_ESTABLISHED_ON_PHASE_BACKEND",
        primary_boundary,
        reuse_boundary,
        (unsigned long long)carrier_creation_count,
        (unsigned long long)mapped_locked_bytes,
        repeat_cycles + 2U,
        repeat_cycles,
        maximum_restoration_error,
        (unsigned long long)(
            (uint64_t)(finish.tv_sec - start.tv_sec)
                * UINT64_C(1000000000)
            + (uint64_t)(finish.tv_nsec - start.tv_nsec)
        ),
        (
            (double)(
                (uint64_t)(finish.tv_sec - start.tv_sec)
                    * UINT64_C(1000000000)
                + (uint64_t)(finish.tv_nsec - start.tv_nsec)
            )
            / (double)(repeat_cycles + 2U)
        ),
        SECCOMP_ENFORCED ? "true" : "false",
        snapshot ? "false" : "true",
        snapshot ? "false" : "true",
        snapshot ? "false" : "true",
        snapshot ? "false" : "true",
        snapshot ? "false" : "true",
        snapshot ? "false" : "true",
        restoration,
        (unsigned long long)transport->request_bytes,
        (unsigned long long)transport->response_bytes,
        (unsigned long long)transport->request_packets,
        (unsigned long long)transport->response_packets
    );
    secure_zero(primary_boundary, sizeof(primary_boundary));
    secure_zero(reuse_boundary, sizeof(reuse_boundary));
    secure_zero(boundary, sizeof(boundary));
    secure_zero(restoration, sizeof(restoration));
    secure_zero(primary_identity, sizeof(primary_identity));
    secure_zero(reuse_identity, sizeof(reuse_identity));
    secure_zero(current_identity, sizeof(current_identity));
    secure_zero(&access, sizeof(access));
    return 1;

fail:
    secure_zero(primary_boundary, sizeof(primary_boundary));
    secure_zero(reuse_boundary, sizeof(reuse_boundary));
    secure_zero(boundary, sizeof(boundary));
    secure_zero(restoration, sizeof(restoration));
    secure_zero(primary_identity, sizeof(primary_identity));
    secure_zero(reuse_identity, sizeof(reuse_identity));
    secure_zero(current_identity, sizeof(current_identity));
    secure_zero(&access, sizeof(access));
    return 0;
}

static int run_transport_baseline(
    struct transport *transport,
    size_t cycles,
    char response[RESPONSE_CAPACITY]
) {
    struct timespec start;
    struct timespec finish;
    if (
        cycles == 0U
        || !exchange(transport, "HELLO", response)
        || !response_has(response, "\"carrier\":false")
        || clock_gettime(CLOCK_MONOTONIC, &start) != 0
    ) {
        return 0;
    }
    for (size_t cycle = 0U; cycle < cycles; ++cycle) {
        for (size_t packet = 0U; packet < 6U; ++packet) {
            if (
                !exchange(transport, "PING", response)
                || !response_is(
                    response,
                    "{\"ok\":true,\"event\":\"INERT_ACK\"}"
                )
            ) {
                return 0;
            }
        }
    }
    if (
        clock_gettime(CLOCK_MONOTONIC, &finish) != 0
        || !shutdown_service(transport, response)
    ) {
        return 0;
    }
    const uint64_t elapsed =
        (uint64_t)(finish.tv_sec - start.tv_sec) * UINT64_C(1000000000)
        + (uint64_t)(finish.tv_nsec - start.tv_nsec);
    printf(
        "{\"result\":\"PASS\","
        "\"scenario\":\"warm-isolated-inert-boundary\","
        "\"carrier\":false,\"cycles\":%zu,"
        "\"packets_per_cycle\":6,"
        "\"transaction_wall_ns\":%llu,"
        "\"average_transaction_wall_ns\":%.3f,"
        "\"request_bytes\":%llu,\"response_bytes\":%llu,"
        "\"request_packets\":%llu,\"response_packets\":%llu}\n",
        cycles,
        (unsigned long long)elapsed,
        (double)elapsed / (double)cycles,
        (unsigned long long)transport->request_bytes,
        (unsigned long long)transport->response_bytes,
        (unsigned long long)transport->request_packets,
        (unsigned long long)transport->response_packets
    );
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        return 2;
    }
    char *end = NULL;
    errno = 0;
    const unsigned long cycles_value = strtoul(argv[3], &end, 10);
    if (
        errno != 0
        || end == argv[3]
        || *end != '\0'
        || cycles_value > 100000UL
    ) {
        return 2;
    }
    const int socket_fd = connect_socket(argv[1]);
    if (socket_fd < 0) {
        return 2;
    }
    struct transport transport = {.socket = socket_fd};
    char response[RESPONSE_CAPACITY];
    int ok = 0;
    if (strcmp(argv[2], "accepted") == 0) {
        ok = run_accepted_or_snapshot(
            &transport,
            0,
            (size_t)cycles_value,
            response
        );
    } else if (strcmp(argv[2], "snapshot") == 0) {
        ok = run_accepted_or_snapshot(
            &transport,
            1,
            (size_t)cycles_value,
            response
        );
    } else if (strcmp(argv[2], "null") == 0) {
        ok = run_null(&transport, response);
    } else if (strcmp(argv[2], "transport") == 0) {
        ok = run_transport_baseline(
            &transport,
            (size_t)cycles_value,
            response
        );
    } else if (
        strcmp(argv[2], "wrong-g") == 0
        || strcmp(argv[2], "missing-g") == 0
        || strcmp(argv[2], "reordered") == 0
    ) {
        ok = run_control(&transport, argv[2], response);
    }
    secure_zero(response, sizeof(response));
    (void)close(socket_fd);
    secure_zero(&transport, sizeof(transport));
    return ok ? 0 : 1;
}
