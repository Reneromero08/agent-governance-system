#define _POSIX_C_SOURCE 200809L

/*
 * Test-build-only client for CATVM QANF fault and baseline controls.
 * It contains no phase implementation and is not used on the accepted path.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define QCONTROL_RESPONSE_CAPACITY 512U

static int qcontrol_connect(const char *path) {
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        return -1;
    }
    const int descriptor = socket(
        AF_UNIX,
        SOCK_SEQPACKET | SOCK_CLOEXEC,
        0
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

static int qcontrol_exchange(
    int descriptor,
    const char *request,
    char response[QCONTROL_RESPONSE_CAPACITY]
) {
    const size_t bytes = strlen(request);
    if (
        send(descriptor, request, bytes, MSG_NOSIGNAL)
            != (ssize_t)bytes
    ) {
        return 0;
    }
    const ssize_t received = recv(
        descriptor,
        response,
        QCONTROL_RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= QCONTROL_RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        return 0;
    }
    response[received] = '\0';
    return 1;
}

static int qcontrol_snapshot_response(
    const char *response,
    size_t variant
) {
    char expected[32];
    const int written = snprintf(
        expected,
        sizeof(expected),
        "OK SNAPSHOT %zu 0 ",
        variant
    );
    return
        written > 0
        && (size_t)written < sizeof(expected)
        && strncmp(response, expected, strlen(expected)) == 0
        && strstr(response, " 1 5 ") != NULL;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        return 2;
    }
    size_t extra_cycles = 0U;
    if (argc == 4) {
        errno = 0;
        char *end = NULL;
        const unsigned long parsed = strtoul(argv[3], &end, 10);
        if (
            errno != 0
            || end == argv[3]
            || *end != '\0'
            || parsed > 100000UL
        ) {
            return 2;
        }
        extra_cycles = (size_t)parsed;
    }
    const int descriptor = qcontrol_connect(argv[1]);
    if (descriptor < 0) {
        return 2;
    }
    char response[QCONTROL_RESPONSE_CAPACITY] = {0};
    if (
        !qcontrol_exchange(descriptor, "HELLO", response)
        || strncmp(
            response,
            "OK HELLO CATVM_QANF_PHASE_1 ",
            strlen("OK HELLO CATVM_QANF_PHASE_1 ")
        ) != 0
    ) {
        (void)close(descriptor);
        return 1;
    }
    int ok = 0;
    if (
        strcmp(argv[2], "wrong-z") == 0
        || strcmp(argv[2], "missing-z") == 0
        || strcmp(argv[2], "reordered") == 0
    ) {
        ok = qcontrol_exchange(
            descriptor,
            "EXECUTE 0",
            response
        )
            && strcmp(
                response,
                "ERR E_RESTORATION_DETECTED"
            ) == 0;
    } else if (strcmp(argv[2], "snapshot") == 0) {
        ok = qcontrol_exchange(descriptor, "EXECUTE 0", response)
            && qcontrol_snapshot_response(response, 0U)
            && qcontrol_exchange(descriptor, "EXECUTE 1", response)
            && qcontrol_snapshot_response(response, 1U);
        for (
            size_t cycle = 0U;
            ok && cycle < extra_cycles;
            ++cycle
        ) {
            char command[16];
            const size_t variant = cycle % 2U;
            const int written = snprintf(
                command,
                sizeof(command),
                "EXECUTE %zu",
                variant
            );
            ok = written > 0
                && (size_t)written < sizeof(command)
                && qcontrol_exchange(
                    descriptor,
                    command,
                    response
                )
                && qcontrol_snapshot_response(response, variant);
        }
        ok = ok
            && qcontrol_exchange(descriptor, "SHUTDOWN", response)
            && strcmp(response, "OK CLOSED") == 0;
    } else if (strcmp(argv[2], "inert") == 0) {
        ok = 1;
        for (
            size_t cycle = 0U;
            ok && cycle < extra_cycles + 2U;
            ++cycle
        ) {
            char expected[48];
            const int written = snprintf(
                expected,
                sizeof(expected),
                "OK INERT %zu",
                cycle + 1U
            );
            ok = written > 0
                && (size_t)written < sizeof(expected)
                && qcontrol_exchange(
                    descriptor,
                    "EXECUTE 0",
                    response
                )
                && strcmp(response, expected) == 0;
        }
        ok = ok
            && qcontrol_exchange(descriptor, "SHUTDOWN", response)
            && strcmp(response, "OK CLOSED") == 0;
    }
    (void)close(descriptor);
    if (!ok) {
        return 1;
    }
    printf(
        "{\"result\":\"PASS\",\"control\":\"%s\","
        "\"transactions\":%zu,"
        "\"production_protocol_selector\":false}\n",
        argv[2],
        extra_cycles + (
            strcmp(argv[2], "snapshot") == 0
            || strcmp(argv[2], "inert") == 0
                ? 2U
                : 1U
        )
    );
    return 0;
}
