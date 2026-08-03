#define _GNU_SOURCE

/*
 * Test-only client for applicability-gated Boolean-TT CATVM controls.
 * None of these selectors exists in the production protocol.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define CBTC_RESPONSE_CAPACITY 512U

static int control_connect(const char *path) {
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        return -1;
    }
    const int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return -1;
    }
    struct sockaddr_un address;
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, path, strlen(path) + 1U);
    if (
        connect(
            fd,
            (const struct sockaddr *)&address,
            sizeof(address)
        ) != 0
    ) {
        (void)close(fd);
        return -1;
    }
    return fd;
}

static int control_exchange(
    int fd,
    const char *request,
    char response[CBTC_RESPONSE_CAPACITY]
) {
    const size_t bytes = strlen(request);
    if (
        send(fd, request, bytes, MSG_NOSIGNAL) != (ssize_t)bytes
    ) {
        return 0;
    }
    const ssize_t received = recv(
        fd,
        response,
        CBTC_RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= CBTC_RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        return 0;
    }
    response[received] = '\0';
    return 1;
}

static int parse_count(const char *text, size_t *count) {
    errno = 0;
    char *tail = NULL;
    const unsigned long parsed = strtoul(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed > 1000UL
    ) {
        return 0;
    }
    *count = (size_t)parsed;
    return 1;
}

static int snapshot_response(
    const char *response,
    size_t expected_variant,
    uint64_t *hash
) {
    char kind[16] = {0};
    char plan[17] = {0};
    char boundary[17] = {0};
    size_t variant = 0U;
    unsigned long long generation = 1ULL;
    size_t ones = 0U;
    size_t cells = 0U;
    unsigned long long creations = 0ULL;
    char extra = '\0';
    const int fields = sscanf(
        response,
        "OK %15s %zu %llu %16s %16s %zu %zu %llu%c",
        kind,
        &variant,
        &generation,
        plan,
        boundary,
        &ones,
        &cells,
        &creations,
        &extra
    );
    if (
        fields != 8
        || strcmp(kind, "SNAPSHOT") != 0
        || variant != expected_variant
        || generation != 0ULL
        || cells == 0U
        || ones > cells
        || creations != 1ULL
    ) {
        return 0;
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long long parsed = strtoull(
        boundary, &tail, 16
    );
    if (errno != 0 || tail == boundary || *tail != '\0') {
        return 0;
    }
    *hash = (uint64_t)parsed;
    return *hash != 0U;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        return 2;
    }
    size_t cycles = 0U;
    if (!parse_count(argv[3], &cycles)) {
        return 2;
    }
    const int fd = control_connect(argv[1]);
    if (fd < 0) {
        return 2;
    }
    char response[CBTC_RESPONSE_CAPACITY] = {0};
    int ok = control_exchange(fd, "HELLO", response)
        && strncmp(
            response,
            "OK HELLO CATVM_BOOLEAN_TT_PHASE_1 ",
            strlen("OK HELLO CATVM_BOOLEAN_TT_PHASE_1 ")
        ) == 0;

    if (
        strcmp(argv[2], "wrong") == 0
        || strcmp(argv[2], "missing") == 0
        || strcmp(argv[2], "reordered") == 0
    ) {
        ok = ok
            && control_exchange(fd, "EXECUTE 0", response)
            && strcmp(
                response, "ERR E_RESTORATION_DETECTED"
            ) == 0;
    } else if (strcmp(argv[2], "snapshot") == 0) {
        uint64_t primary_hash = 0U;
        uint64_t reuse_hash = 0U;
        for (size_t cycle = 0U; ok && cycle < cycles + 2U; ++cycle) {
            const size_t variant = cycle % 2U;
            char command[16];
            const int written = snprintf(
                command, sizeof(command), "EXECUTE %zu", variant
            );
            uint64_t hash = 0U;
            ok = written > 0
                && (size_t)written < sizeof(command)
                && control_exchange(fd, command, response)
                && snapshot_response(response, variant, &hash);
            if (cycle == 0U) {
                primary_hash = hash;
            } else if (cycle == 1U) {
                reuse_hash = hash;
                ok = ok && reuse_hash != primary_hash;
            } else {
                ok = ok && hash == (
                    variant == 0U ? primary_hash : reuse_hash
                );
            }
        }
        ok = ok
            && control_exchange(fd, "SHUTDOWN", response)
            && strcmp(response, "OK CLOSED") == 0;
    } else if (strcmp(argv[2], "inert") == 0) {
        for (size_t cycle = 0U; ok && cycle < cycles + 1U; ++cycle) {
            ok = control_exchange(fd, "EXECUTE 0", response)
                && strncmp(
                    response,
                    "OK INERT ",
                    strlen("OK INERT ")
                ) == 0;
        }
        ok = ok
            && control_exchange(fd, "SHUTDOWN", response)
            && strcmp(response, "OK CLOSED") == 0;
    } else {
        ok = 0;
    }
    (void)close(fd);
    if (!ok) {
        return 1;
    }
    printf(
        "{\"result\":\"PASS\","
        "\"control\":\"%s\","
        "\"cycles\":%zu,"
        "\"production_protocol_selector\":false}\n",
        argv[2],
        cycles
    );
    return 0;
}
