#define _GNU_SOURCE

/*
 * Minimal CATVM machine boundary.
 *
 * The controller sees only a strict AF_UNIX/SOCK_SEQPACKET command surface.
 * This process alone maps, locks, evolves, restores, and reuses the phase
 * carrier.  It is non-dumpable before carrier allocation and installs a
 * post-accept seccomp allowlist before accepting any transaction command.
 */

#ifdef CATVM_WIDE2_BUILD
#include "catvm_wide2_core.h"
#define CATVM_PROTOCOL_NAME "CATVM_WIDE2_PHASE_1"
#define CATVM_MAXIMUM_TEMPORARY_COMPLEX_VALUES 240U
#else
#include "catvm_phase_core.h"
#define CATVM_PROTOCOL_NAME "CATVM_PHASE_1"
#define CATVM_MAXIMUM_TEMPORARY_COMPLEX_VALUES 52U
#endif

#include <errno.h>
#include <linux/prctl.h>
#include <seccomp.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#define REQUEST_CAPACITY 512U
#define RESPONSE_CAPACITY 2048U
#ifdef CATVM_SANITIZER_BUILD
#define SECCOMP_STATUS "DISABLED_SANITIZER_BUILD"
#else
#define SECCOMP_STATUS "ACTIVE_ALLOWLIST"
#endif

struct service_configuration {
    const char *socket_path;
    enum catvm_backend_kind backend;
    enum catvm_restore_mode restore_mode;
};

static void secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static int memory_is_zero(const void *memory, size_t bytes) {
    const unsigned char *cursor = memory;
    unsigned char combined = 0U;
    while (bytes > 0U) {
        combined |= *cursor;
        ++cursor;
        --bytes;
    }
    return combined == 0U;
}

static int receive_queue_empty(int client) {
    unsigned char byte = 0U;
    errno = 0;
    const ssize_t received = recv(
        client,
        &byte,
        sizeof(byte),
        MSG_PEEK | MSG_DONTWAIT
    );
    return (
        received < 0
        && (errno == EAGAIN || errno == EWOULDBLOCK)
    );
}

static int parse_backend(
    const char *text,
    enum catvm_backend_kind *backend
) {
    if (strcmp(text, "in-place") == 0) {
        *backend = CATVM_BACKEND_IN_PLACE;
        return 1;
    }
    if (strcmp(text, "snapshot") == 0) {
        *backend = CATVM_BACKEND_SNAPSHOT;
        return 1;
    }
    if (strcmp(text, "null") == 0) {
        *backend = CATVM_BACKEND_NULL;
        return 1;
    }
    return 0;
}

static int parse_restore_mode(
    const char *text,
    enum catvm_restore_mode *mode
) {
    if (strcmp(text, "correct") == 0) {
        *mode = CATVM_RESTORE_CORRECT;
        return 1;
    }
    if (strcmp(text, "wrong-g") == 0) {
        *mode = CATVM_RESTORE_WRONG_G;
        return 1;
    }
    if (strcmp(text, "missing-g") == 0) {
        *mode = CATVM_RESTORE_MISSING_G;
        return 1;
    }
    if (strcmp(text, "reordered") == 0) {
        *mode = CATVM_RESTORE_REORDERED;
        return 1;
    }
    if (strcmp(text, "snapshot") == 0) {
        *mode = CATVM_RESTORE_SNAPSHOT;
        return 1;
    }
    return 0;
}

static int configuration_valid(
    const struct service_configuration *configuration
) {
    if (
        configuration->backend == CATVM_BACKEND_IN_PLACE
        && configuration->restore_mode == CATVM_RESTORE_SNAPSHOT
    ) {
        return 0;
    }
    if (
        configuration->backend == CATVM_BACKEND_SNAPSHOT
        && configuration->restore_mode != CATVM_RESTORE_SNAPSHOT
    ) {
        return 0;
    }
    return (
        configuration->backend == CATVM_BACKEND_NULL
        || configuration->backend == CATVM_BACKEND_SNAPSHOT
        || configuration->restore_mode != CATVM_RESTORE_SNAPSHOT
    );
}

static int process_is_untraced(void) {
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
}

static int establish_process_guards(void) {
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

static size_t page_round(size_t bytes, size_t page_size) {
    if (bytes > SIZE_MAX - page_size + 1U) {
        return 0U;
    }
    return ((bytes + page_size - 1U) / page_size) * page_size;
}

static struct catvm_machine *map_machine(
    enum catvm_backend_kind backend,
    size_t *mapped_bytes
) {
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        return NULL;
    }
    const size_t page_size = (size_t)page_size_long;
    *mapped_bytes = page_round(sizeof(struct catvm_machine), page_size);
    if (*mapped_bytes == 0U) {
        return NULL;
    }
    struct catvm_machine *machine = mmap(
        NULL,
        *mapped_bytes,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );
    if (machine == MAP_FAILED) {
        return NULL;
    }
    if (
        mlock(machine, *mapped_bytes) != 0
        || madvise(machine, *mapped_bytes, MADV_DONTDUMP) != 0
        || madvise(machine, *mapped_bytes, MADV_DONTFORK) != 0
        || !catvm_machine_init(machine, backend)
    ) {
        (void)munlock(machine, *mapped_bytes);
        (void)munmap(machine, *mapped_bytes);
        return NULL;
    }
    return machine;
}

static int make_listener(const char *path) {
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
        bind(listener, (const struct sockaddr *)&address, sizeof(address))
            != 0
        || chmod(path, S_IRUSR | S_IWUSR) != 0
        || listen(listener, 1) != 0
    ) {
        (void)close(listener);
        (void)unlink(path);
        return -1;
    }
    return listener;
}

static int peer_is_same_real_user(int client) {
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

static int install_seccomp(int client) {
#ifdef CATVM_SANITIZER_BUILD
    (void)client;
    return 1;
#else
    scmp_filter_ctx filter = seccomp_init(SCMP_ACT_KILL_PROCESS);
    if (filter == NULL) {
        return 0;
    }
    int ok = 1;
#define ALLOW_SYSCALL(name) \
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
    ALLOW_SYSCALL(close);
    ALLOW_SYSCALL(exit);
    ALLOW_SYSCALL(exit_group);
    ALLOW_SYSCALL(rt_sigreturn);
    ALLOW_SYSCALL(munlock);
    ALLOW_SYSCALL(munmap);
#undef ALLOW_SYSCALL
    if (!ok || seccomp_load(filter) != 0) {
        seccomp_release(filter);
        return 0;
    }
    /*
     * Intentionally retain the tiny filter object.  Releasing it could call
     * an allocator syscall after the allowlist is live.
     */
    return 1;
#endif
}

static int send_response(int client, const char *response) {
    const size_t bytes = strlen(response);
    if (bytes == 0U || bytes >= RESPONSE_CAPACITY) {
        return 0;
    }
    return send(client, response, bytes, MSG_NOSIGNAL) == (ssize_t)bytes;
}

static int append_response(
    char response[RESPONSE_CAPACITY],
    size_t *used,
    const char *format,
    ...
) {
    if (*used >= RESPONSE_CAPACITY) {
        return 0;
    }
    va_list arguments;
    va_start(arguments, format);
    const int written = vsnprintf(
        response + *used,
        RESPONSE_CAPACITY - *used,
        format,
        arguments
    );
    va_end(arguments);
    if (
        written < 0
        || (size_t)written >= RESPONSE_CAPACITY - *used
    ) {
        return 0;
    }
    *used += (size_t)written;
    return 1;
}

static int format_projection_response(
    char response[RESPONSE_CAPACITY],
    const struct catvm_projection *projection
) {
    size_t used = 0U;
    if (!append_response(
        response,
        &used,
        "{\"ok\":true,\"event\":\"FINAL_BOUNDARY\","
        "\"port\":\"Z\",\"coefficients\":["
    )) {
        return 0;
    }
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        if (!append_response(
            response,
            &used,
            "%s%d",
            index == 0U ? "" : ",",
            projection->coefficient[index]
        )) {
            return 0;
        }
    }
    return append_response(
        response,
        &used,
        "],\"fnv1a64\":\"%016llx\","
        "\"maximum_root_error\":%.12g,"
        "\"decoded_intermediate_coefficients\":0}",
        (unsigned long long)projection->hash,
        projection->maximum_root_error
    );
}

static int parse_program(
    const char *request,
    struct catvm_program *program
) {
    static const char prefix[] = "SEAL ";
    if (strncmp(request, prefix, sizeof(prefix) - 1U) != 0) {
        return 0;
    }
    const char *cursor = request + sizeof(prefix) - 1U;
    int values[3U * CATVM_RELATION_CELLS];
    for (size_t index = 0U; index < 3U * CATVM_RELATION_CELLS; ++index) {
        if (*cursor < '0' || *cursor > '2') {
            secure_zero(values, sizeof(values));
            return 0;
        }
        values[index] = *cursor - '0';
        ++cursor;
        if (index + 1U < 3U * CATVM_RELATION_CELLS) {
            if (*cursor != ' ') {
                secure_zero(values, sizeof(values));
                return 0;
            }
            ++cursor;
        }
    }
    if (*cursor != '\0') {
        secure_zero(values, sizeof(values));
        return 0;
    }
    for (size_t index = 0U; index < CATVM_RELATION_CELLS; ++index) {
        program->left[index] = values[index];
        program->right[index] =
            values[CATVM_RELATION_CELLS + index];
        program->constraint[index] =
            values[2U * CATVM_RELATION_CELLS + index];
    }
    secure_zero(values, sizeof(values));
    return 1;
}

static const char *bool_json(int value) {
    return value ? "true" : "false";
}

static int serve(
    int client,
    struct catvm_machine *machine,
    size_t mapped_bytes,
    enum catvm_restore_mode configured_restore
) {
    char request[REQUEST_CAPACITY];
    char response[RESPONSE_CAPACITY];
    struct catvm_projection projection;
    int keep_running = 1;
    while (keep_running) {
        secure_zero(request, sizeof(request));
        secure_zero(response, sizeof(response));
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
            if (!send_response(
                client,
                "{\"ok\":false,\"error\":\"E_PROTOCOL\"}"
            )) {
                break;
            }
            continue;
        }
        request[received] = '\0';

        if (strcmp(request, "HELLO") == 0) {
            const size_t mapped_locked_bytes =
                mapped_bytes + machine->snapshot_mapped_bytes;
            const int written = snprintf(
                response,
                sizeof(response),
                "{\"ok\":true,\"protocol\":\"" CATVM_PROTOCOL_NAME "\","
                "\"backend\":\"%s\",\"carrier\":%s,"
                "\"carrier_cells\":%u,\"physical_complex_values\":%u,"
                "\"logical_carrier_bytes\":%zu,"
                "\"mapped_locked_bytes\":%zu,"
                "\"compiled_program_bytes\":%zu,"
                "\"compiled_morphisms\":2,"
                "\"compiled_morphism_descriptor_bytes\":%zu,"
                "\"maximum_temporary_complex_values\":%u,"
                "\"carrier_creations\":%llu,"
                "\"retained_inverse_factors\":0,"
                "\"memory_guard\":\"NON_DUMPABLE_LOCKED_PRIVATE\","
                "\"seccomp\":\"%s\"}",
                catvm_backend_name(machine->backend),
                bool_json(machine->carrier_enabled),
                machine->carrier_enabled ? CATVM_CARRIER_CELLS : 0U,
                machine->carrier_enabled
                    ? 2U * CATVM_CARRIER_CELLS
                    : 0U,
                machine->carrier_enabled
                    ? 2U
                        * CATVM_CARRIER_CELLS
                        * sizeof(double complex)
                    : 0U,
                mapped_locked_bytes,
                sizeof(struct catvm_program),
                2U * sizeof(uint64_t),
                CATVM_MAXIMUM_TEMPORARY_COMPLEX_VALUES,
                (unsigned long long)machine->carrier_creation_count,
                SECCOMP_STATUS
            );
            if (
                written <= 0
                || (size_t)written >= sizeof(response)
                || !send_response(client, response)
            ) {
                break;
            }
            continue;
        }

        if (strcmp(request, "PING") == 0) {
            if (!send_response(
                client,
                "{\"ok\":true,\"event\":\"INERT_ACK\"}"
            )) {
                break;
            }
            continue;
        }

        if (strcmp(request, "PROJECT Y") == 0) {
            if (!send_response(
                client,
                "{\"ok\":false,"
                "\"error\":\"E_INTERMEDIATE_PROJECTION_DENIED\","
                "\"type\":\"BOOLEAN_F3_RELATION\","
                "\"state_unchanged\":true}"
            )) {
                break;
            }
            continue;
        }

        struct catvm_program program;
        memset(&program, 0, sizeof(program));
        if (parse_program(request, &program)) {
            if (!machine->carrier_enabled) {
                secure_zero(&program, sizeof(program));
                if (!send_response(
                    client,
                    "{\"ok\":false,\"error\":\"E_NO_CARRIER\"}"
                )) {
                    break;
                }
                continue;
            }
            const int sealed = catvm_seal(machine, &program);
            secure_zero(&program, sizeof(program));
            if (!send_response(
                client,
                sealed
                    ? "{\"ok\":true,\"event\":\"CARRIER_SEALED\","
                        "\"type\":\"BOOLEAN_F3_RELATION_PROGRAM\"}"
                    : "{\"ok\":false,\"error\":\"E_STATE\"}"
            )) {
                break;
            }
            continue;
        }
        secure_zero(&program, sizeof(program));

        if (strcmp(request, "F") == 0) {
            if (!catvm_apply_f(machine)) {
                if (!send_response(
                    client,
                    "{\"ok\":false,\"error\":\"E_STATE\"}"
                )) {
                    break;
                }
                continue;
            }
            if (!send_response(
                client,
                "{\"ok\":true,\"event\":\"INTERMEDIATE_CUSTODY\","
                "\"type\":\"BOOLEAN_F3_RELATION\","
                "\"projection\":\"DENIED\","
                "\"morphism_depth\":1}"
            )) {
                break;
            }
            continue;
        }

        if (strcmp(request, "G") == 0) {
            if (!catvm_apply_g(machine)) {
                if (!send_response(
                    client,
                    "{\"ok\":false,\"error\":\"E_STATE\"}"
                )) {
                    break;
                }
                continue;
            }
            if (!send_response(
                client,
                "{\"ok\":true,\"event\":\"FINAL_READY\","
                "\"morphism_depth\":2}"
            )) {
                break;
            }
            continue;
        }

        if (strcmp(request, "PROJECT Z") == 0) {
            secure_zero(&projection, sizeof(projection));
            if (!catvm_project_final(machine, &projection)) {
                secure_zero(&projection, sizeof(projection));
                if (!send_response(
                    client,
                    "{\"ok\":false,\"error\":\"E_STATE\"}"
                )) {
                    break;
                }
                continue;
            }
            const int written = format_projection_response(
                response, &projection
            );
            secure_zero(&projection, sizeof(projection));
            if (
                !written || !send_response(client, response)
            ) {
                break;
            }
            continue;
        }

        if (strcmp(request, "RESTORE") == 0) {
            struct catvm_restoration restoration;
            secure_zero(&restoration, sizeof(restoration));
            secure_zero(request, sizeof(request));
            const int protocol_rx_cleared =
                memory_is_zero(request, sizeof(request));
            const int operation_ok = catvm_restore(
                machine,
                configured_restore,
                &restoration
            );
            const int backend_queue_empty =
                receive_queue_empty(client);
            const int accepted_mode =
                configured_restore == CATVM_RESTORE_CORRECT
                || configured_restore == CATVM_RESTORE_SNAPSHOT;
            const int accepted =
                operation_ok
                && accepted_mode
                && restoration.carrier_within_tolerance
                && restoration.invariant_state_exact
                && restoration.generation_transition_exact
                && restoration.transient_state_exact
                && protocol_rx_cleared
                && backend_queue_empty
                && machine->carrier_creation_count == 1U;
            const int control_discriminated =
                operation_ok
                && !accepted_mode
                && restoration.reordered_pair_applicable
                && restoration.maximum_abs_error
                    >= CATVM_CONTROL_MINIMUM;
            const int written = snprintf(
                response,
                sizeof(response),
                "{\"ok\":%s,\"event\":\"RESTORATION\","
                "\"backend\":\"%s\","
                "\"actual_inverse\":%s,"
                "\"snapshot_reload\":%s,"
                "\"control_discriminated\":%s,"
                "\"maximum_abs_error\":%.12g,"
                "\"tolerance\":%.12g,"
                "\"carrier_within_tolerance\":%s,"
                "\"invariant_state_exact\":%s,"
                "\"generation_transition_exact\":%s,"
                "\"transient_state_exact\":%s,"
                "\"contract2_workspace_cleared\":%s,"
                "\"generation\":%llu,"
                "\"carrier_creations\":%llu,"
                "\"native_compose_calls\":%llu,"
                "\"native_intersection_calls\":%llu,"
                "\"native_contract2_calls\":%llu,"
                "\"native_symbol_products\":%llu,"
                "\"coefficient_accumulation_additions\":%llu,"
                "\"restriction_and_intersection_additions\":%llu,"
                "\"phase_cell_updates\":%llu,"
                "\"inverse_factor_recomputations\":%llu,"
                "\"snapshot_bytes_written\":%llu,"
                "\"snapshot_bytes_reloaded\":%llu,"
                "\"boundary_decodes\":%llu,"
                "\"restoration_cell_checks\":%llu,"
                "\"protocol_rx_cleared\":%s,"
                "\"backend_queue_empty\":%s}",
                bool_json(accepted || control_discriminated),
                catvm_backend_name(machine->backend),
                bool_json(restoration.used_actual_inverse),
                bool_json(restoration.used_snapshot_reload),
                bool_json(control_discriminated),
                restoration.maximum_abs_error,
                CATVM_RESTORATION_TOLERANCE,
                bool_json(restoration.carrier_within_tolerance),
                bool_json(restoration.invariant_state_exact),
                bool_json(restoration.generation_transition_exact),
                bool_json(restoration.transient_state_exact),
                bool_json(restoration.workspace_cleared),
                (unsigned long long)restoration.generation_after,
                (unsigned long long)machine->carrier_creation_count,
                (unsigned long long)
                    machine->resources.native_compose_calls,
                (unsigned long long)
                    machine->resources.native_intersection_calls,
                (unsigned long long)
                    machine->resources.native_contract2_calls,
                (unsigned long long)
                    machine->resources.native_symbol_products,
                (unsigned long long)
                    machine->resources.coefficient_accumulation_additions,
                (unsigned long long)
                    machine->resources
                        .restriction_and_intersection_additions,
                (unsigned long long)
                    machine->resources.phase_cell_updates,
                (unsigned long long)
                    machine->resources.inverse_factor_recomputations,
                (unsigned long long)
                    machine->resources.snapshot_bytes_written,
                (unsigned long long)
                    machine->resources.snapshot_bytes_reloaded,
                (unsigned long long)
                    machine->resources.boundary_decodes,
                (unsigned long long)
                    machine->resources.restoration_cell_checks,
                bool_json(protocol_rx_cleared),
                bool_json(backend_queue_empty)
            );
            secure_zero(&restoration, sizeof(restoration));
            if (
                written <= 0
                || (size_t)written >= sizeof(response)
                || !send_response(client, response)
            ) {
                break;
            }
            continue;
        }

        if (strcmp(request, "SHUTDOWN") == 0) {
            if (!send_response(
                client,
                "{\"ok\":true,\"event\":\"CLOSED\"}"
            )) {
                break;
            }
            keep_running = 0;
            continue;
        }

        if (!send_response(
            client,
            "{\"ok\":false,\"error\":\"E_PROTOCOL\"}"
        )) {
            break;
        }
    }
    secure_zero(&projection, sizeof(projection));
    secure_zero(request, sizeof(request));
    secure_zero(response, sizeof(response));
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4 || !process_is_untraced()) {
        return 2;
    }
    struct service_configuration configuration = {
        .socket_path = argv[1]
    };
    if (
        !parse_backend(argv[2], &configuration.backend)
        || !parse_restore_mode(argv[3], &configuration.restore_mode)
        || !configuration_valid(&configuration)
        || !establish_process_guards()
    ) {
        return 2;
    }
    (void)umask(S_IRWXG | S_IRWXO);
    (void)clearenv();

    size_t mapped_bytes = 0U;
    struct catvm_machine *machine = map_machine(
        configuration.backend,
        &mapped_bytes
    );
    if (machine == NULL) {
        return 2;
    }
    const int listener = make_listener(configuration.socket_path);
    if (listener < 0) {
        catvm_machine_destroy(machine);
        (void)munlock(machine, mapped_bytes);
        (void)munmap(machine, mapped_bytes);
        return 2;
    }
    const int client = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0 || !peer_is_same_real_user(client)) {
        if (client >= 0) {
            (void)close(client);
        }
        (void)close(listener);
        (void)unlink(configuration.socket_path);
        catvm_machine_destroy(machine);
        (void)munlock(machine, mapped_bytes);
        (void)munmap(machine, mapped_bytes);
        return 2;
    }
    (void)close(listener);
    if (unlink(configuration.socket_path) != 0) {
        (void)close(client);
        catvm_machine_destroy(machine);
        (void)munlock(machine, mapped_bytes);
        (void)munmap(machine, mapped_bytes);
        return 2;
    }
    if (!install_seccomp(client)) {
        (void)close(client);
        catvm_machine_destroy(machine);
        (void)munlock(machine, mapped_bytes);
        (void)munmap(machine, mapped_bytes);
        return 2;
    }

    (void)serve(
        client,
        machine,
        mapped_bytes,
        configuration.restore_mode
    );
    (void)close(client);
    catvm_machine_destroy(machine);
    (void)munlock(machine, mapped_bytes);
    (void)munmap(machine, mapped_bytes);
    return 0;
}
