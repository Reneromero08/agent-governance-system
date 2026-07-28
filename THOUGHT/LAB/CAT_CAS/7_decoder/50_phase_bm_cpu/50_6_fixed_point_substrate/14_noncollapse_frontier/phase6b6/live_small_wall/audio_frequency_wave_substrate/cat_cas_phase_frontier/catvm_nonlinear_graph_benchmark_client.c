#define _POSIX_C_SOURCE 200809L

/*
 * Test-only matched transport timer. It sends the same alternating atomic
 * requests to in-place, snapshot, and inert services and never interprets
 * answer content beyond the public response class.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define CNB_CAPACITY 512U

static int cnb_connect(const char *path) {
    const int descriptor = socket(
        AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0
    );
    if (
        descriptor < 0
        || strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)
    ) {
        if (descriptor >= 0) {
            (void)close(descriptor);
        }
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

static int cnb_exchange(
    int descriptor,
    const char *request,
    char response[CNB_CAPACITY],
    uint64_t *request_bytes,
    uint64_t *response_bytes
) {
    const size_t bytes = strlen(request);
    if (
        send(descriptor, request, bytes, MSG_NOSIGNAL)
            != (ssize_t)bytes
    ) {
        return 0;
    }
    const ssize_t received = recv(
        descriptor, response, CNB_CAPACITY - 1U, MSG_TRUNC
    );
    if (received <= 0 || (size_t)received >= CNB_CAPACITY) {
        return 0;
    }
    response[received] = '\0';
    *request_bytes += bytes;
    *response_bytes += (uint64_t)received;
    return 1;
}

static uint64_t cnb_nanoseconds(const struct timespec *time) {
    return (uint64_t)time->tv_sec * UINT64_C(1000000000)
        + (uint64_t)time->tv_nsec;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        return 2;
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long cycles = strtoul(argv[2], &tail, 10);
    if (
        errno != 0
        || tail == argv[2]
        || *tail != '\0'
        || cycles < 1U
        || cycles > 100000U
        || (
            strcmp(argv[3], "FINAL___") != 0
            && strcmp(argv[3], "SNAPSHOT") != 0
            && strcmp(argv[3], "INERT___") != 0
        )
    ) {
        return 2;
    }
    const int descriptor = cnb_connect(argv[1]);
    if (descriptor < 0) {
        return 2;
    }
    char response[CNB_CAPACITY] = {0};
    uint64_t request_bytes = 0U;
    uint64_t response_bytes = 0U;
    if (
        !cnb_exchange(
            descriptor,
            "HELLO",
            response,
            &request_bytes,
            &response_bytes
        )
        || strncmp(response, "OK HELLO ", 9U) != 0
    ) {
        (void)close(descriptor);
        return 2;
    }
    const unsigned long warmup = 32U;
    struct timespec begin = {0};
    struct timespec end = {0};
    for (
        unsigned long cycle = 0U;
        cycle < warmup + cycles;
        ++cycle
    ) {
        if (cycle == warmup && clock_gettime(CLOCK_MONOTONIC, &begin) != 0) {
            (void)close(descriptor);
            return 2;
        }
        const char *request = cycle % 2U == 0U
            ? "EXECUTE 0"
            : "EXECUTE 1";
        if (
            !cnb_exchange(
                descriptor,
                request,
                response,
                &request_bytes,
                &response_bytes
            )
            || strncmp(response, "OK ", 3U) != 0
            || strstr(response, argv[3]) != response + 3
        ) {
            (void)close(descriptor);
            return 2;
        }
    }
    if (
        clock_gettime(CLOCK_MONOTONIC, &end) != 0
        || !cnb_exchange(
            descriptor,
            "SHUTDOWN",
            response,
            &request_bytes,
            &response_bytes
        )
        || strcmp(response, "OK CLOSED") != 0
    ) {
        (void)close(descriptor);
        return 2;
    }
    (void)close(descriptor);
    const uint64_t elapsed =
        cnb_nanoseconds(&end) - cnb_nanoseconds(&begin);
    printf(
        "{\"result\":\"PASS\",\"arm\":\"%s\","
        "\"warmup_transactions\":%lu,"
        "\"timed_transactions\":%lu,"
        "\"elapsed_ns\":%llu,\"ns_per_transaction\":%.17g,"
        "\"request_bytes_including_warmup\":%llu,"
        "\"response_bytes_including_warmup\":%llu}\n",
        argv[3],
        warmup,
        cycles,
        (unsigned long long)elapsed,
        (double)elapsed / (double)cycles,
        (unsigned long long)request_bytes,
        (unsigned long long)response_bytes
    );
    return 0;
}
