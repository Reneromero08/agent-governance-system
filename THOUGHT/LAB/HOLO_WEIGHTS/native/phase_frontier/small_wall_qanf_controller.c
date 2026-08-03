#define _POSIX_C_SOURCE 200809L

/*
 * Common controller for all three fixed-schema Small-Wall comparison arms.
 * It contains a public variant schedule but no QANF evaluator or expected
 * boundary material.
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

#define QWC_RESPONSE_CAPACITY 512U
#define QWC_VARIANTS 4U
#define QWC_BOUNDARY_CELLS 5U
#define QWC_MAX_TRANSACTIONS 1000000U

struct qwc_transport {
    int socket_fd;
    uint64_t request_packets;
    uint64_t response_packets;
    uint64_t request_bytes;
    uint64_t response_bytes;
};

struct qwc_boundary {
    int coefficient[QWC_BOUNDARY_CELLS];
    int present;
};

struct qwc_stats {
    uint64_t transactions;
    uint64_t boolean_ands;
    uint64_t phase_products;
    uint64_t carrier_reads;
    uint64_t phase_cell_updates;
    uint64_t final_decodes;
    uint64_t snapshot_loads;
    uint64_t snapshot_reload_bytes;
    uint64_t actual_inverse_transactions;
    uint64_t restoration_generation;
    uint64_t carrier_creation_count;
    uint64_t service_cpu_ns;
    uint64_t seal_cpu_ns;
};

static void qwc_secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static int qwc_monotonic_raw_ns(uint64_t *time_ns) {
    struct timespec measured;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &measured) != 0) {
        return 0;
    }
    if (measured.tv_sec < 0 || measured.tv_nsec < 0) {
        return 0;
    }
    const uint64_t seconds = (uint64_t)measured.tv_sec;
    if (seconds > UINT64_MAX / UINT64_C(1000000000)) {
        return 0;
    }
    *time_ns = seconds * UINT64_C(1000000000)
        + (uint64_t)measured.tv_nsec;
    return 1;
}

static int qwc_connect(const char *path) {
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

static int qwc_exchange(
    struct qwc_transport *transport,
    const char *request,
    char response[QWC_RESPONSE_CAPACITY]
) {
    const size_t bytes = strlen(request);
    if (
        bytes == 0U
        || send(
            transport->socket_fd,
            request,
            bytes,
            MSG_NOSIGNAL
        ) != (ssize_t)bytes
    ) {
        return 0;
    }
    ++transport->request_packets;
    transport->request_bytes += bytes;
    const ssize_t received = recv(
        transport->socket_fd,
        response,
        QWC_RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= QWC_RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        qwc_secure_zero(response, QWC_RESPONSE_CAPACITY);
        return 0;
    }
    response[received] = '\0';
    ++transport->response_packets;
    transport->response_bytes += (uint64_t)received;
    return 1;
}

static int qwc_parse_size(const char *text, size_t *value) {
    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (
        errno != 0
        || end == text
        || *end != '\0'
        || parsed > SIZE_MAX
    ) {
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}

static int qwc_parse_boundary(
    const char *response,
    size_t expected_variant,
    struct qwc_boundary *boundary
) {
    size_t variant = 0U;
    size_t cells = 0U;
    int coefficient[QWC_BOUNDARY_CELLS];
    char trailing = '\0';
    const int fields = sscanf(
        response,
        "OK FINAL %zu %zu %d %d %d %d %d %c",
        &variant,
        &cells,
        &coefficient[0],
        &coefficient[1],
        &coefficient[2],
        &coefficient[3],
        &coefficient[4],
        &trailing
    );
    if (
        fields != 7
        || variant != expected_variant
        || cells != QWC_BOUNDARY_CELLS
    ) {
        return 0;
    }
    for (size_t cell = 0U; cell < QWC_BOUNDARY_CELLS; ++cell) {
        if (
            coefficient[cell] != 0
            && coefficient[cell] != 1
        ) {
            return 0;
        }
    }
    if (boundary->present) {
        return memcmp(
            boundary->coefficient,
            coefficient,
            sizeof(coefficient)
        ) == 0;
    }
    memcpy(
        boundary->coefficient,
        coefficient,
        sizeof(coefficient)
    );
    boundary->present = 1;
    return 1;
}

static int qwc_parse_stats(
    const char *response,
    struct qwc_stats *stats
) {
    unsigned long long value[13];
    char trailing = '\0';
    const int fields = sscanf(
        response,
        "OK STATS %llu %llu %llu %llu %llu %llu %llu "
        "%llu %llu %llu %llu %llu %llu %c",
        &value[0],
        &value[1],
        &value[2],
        &value[3],
        &value[4],
        &value[5],
        &value[6],
        &value[7],
        &value[8],
        &value[9],
        &value[10],
        &value[11],
        &value[12],
        &trailing
    );
    if (fields != 13) {
        return 0;
    }
    stats->transactions = (uint64_t)value[0];
    stats->boolean_ands = (uint64_t)value[1];
    stats->phase_products = (uint64_t)value[2];
    stats->carrier_reads = (uint64_t)value[3];
    stats->phase_cell_updates = (uint64_t)value[4];
    stats->final_decodes = (uint64_t)value[5];
    stats->snapshot_loads = (uint64_t)value[6];
    stats->snapshot_reload_bytes = (uint64_t)value[7];
    stats->actual_inverse_transactions = (uint64_t)value[8];
    stats->restoration_generation = (uint64_t)value[9];
    stats->carrier_creation_count = (uint64_t)value[10];
    stats->service_cpu_ns = (uint64_t)value[11];
    stats->seal_cpu_ns = (uint64_t)value[12];
    return 1;
}

static int qwc_execute_pattern(
    struct qwc_transport *transport,
    size_t transactions,
    struct qwc_boundary boundary[QWC_VARIANTS],
    char response[QWC_RESPONSE_CAPACITY]
) {
    static const size_t pattern[8] = {
        0U, 3U, 2U, 1U, 1U, 2U, 3U, 0U
    };
    if ((transactions % 8U) != 0U) {
        return 0;
    }
    for (size_t index = 0U; index < transactions; ++index) {
        const size_t variant = pattern[index % 8U];
        char command[16];
        const int written = snprintf(
            command,
            sizeof(command),
            "EXECUTE %zu",
            variant
        );
        if (
            written <= 0
            || (size_t)written >= sizeof(command)
            || !qwc_exchange(transport, command, response)
            || !qwc_parse_boundary(
                response,
                variant,
                &boundary[variant]
            )
        ) {
            return 0;
        }
    }
    return 1;
}

static void qwc_print_boundary(const struct qwc_boundary *boundary) {
    putchar('[');
    for (size_t cell = 0U; cell < QWC_BOUNDARY_CELLS; ++cell) {
        printf(
            "%s%d",
            cell == 0U ? "" : ",",
            boundary->coefficient[cell]
        );
    }
    putchar(']');
}

int main(int argc, char **argv) {
    if (argc != 4) {
        return 2;
    }
    size_t warm_transactions = 0U;
    size_t timed_transactions = 0U;
    if (
        !qwc_parse_size(argv[2], &warm_transactions)
        || !qwc_parse_size(argv[3], &timed_transactions)
        || warm_transactions > QWC_MAX_TRANSACTIONS
        || timed_transactions > QWC_MAX_TRANSACTIONS
        || (warm_transactions % 8U) != 0U
        || (timed_transactions % 8U) != 0U
    ) {
        return 2;
    }
    struct qwc_transport transport = {
        .socket_fd = qwc_connect(argv[1])
    };
    if (transport.socket_fd < 0) {
        return 2;
    }
    char response[QWC_RESPONSE_CAPACITY] = {0};
    struct qwc_boundary boundary[QWC_VARIANTS];
    memset(boundary, 0, sizeof(boundary));
    struct qwc_stats stats;
    memset(&stats, 0, sizeof(stats));

    int ok = qwc_exchange(&transport, "HELLO", response)
        && strcmp(
            response,
            "OK HELLO CATVM_QANF_SMALL_WALL_COMPARE_1 4 5"
        ) == 0
        && qwc_execute_pattern(
            &transport,
            warm_transactions,
            boundary,
            response
        )
        && qwc_exchange(&transport, "RESET", response)
        && strcmp(response, "OK RESET") == 0;

    const uint64_t timed_request_packets_before =
        transport.request_packets;
    const uint64_t timed_response_packets_before =
        transport.response_packets;
    const uint64_t timed_request_bytes_before =
        transport.request_bytes;
    const uint64_t timed_response_bytes_before =
        transport.response_bytes;
    uint64_t wall_start_ns = 0U;
    uint64_t wall_end_ns = 0U;
    ok = ok
        && qwc_monotonic_raw_ns(&wall_start_ns)
        && qwc_execute_pattern(
            &transport,
            timed_transactions,
            boundary,
            response
        )
        && qwc_monotonic_raw_ns(&wall_end_ns)
        && wall_end_ns >= wall_start_ns
        && qwc_exchange(&transport, "STATS", response)
        && qwc_parse_stats(response, &stats)
        && stats.transactions == timed_transactions;

    const uint64_t timed_request_packets =
        transport.request_packets
        - timed_request_packets_before - 1U;
    const uint64_t timed_response_packets =
        transport.response_packets
        - timed_response_packets_before - 1U;
    const uint64_t stats_request_bytes = strlen("STATS");
    const uint64_t timed_request_bytes =
        transport.request_bytes
        - timed_request_bytes_before - stats_request_bytes;
    const uint64_t timed_response_bytes =
        transport.response_bytes
        - timed_response_bytes_before
        - (uint64_t)strlen(response);
    for (size_t variant = 0U; ok && variant < QWC_VARIANTS; ++variant) {
        ok = boundary[variant].present;
    }
    ok = ok
        && qwc_exchange(&transport, "SHUTDOWN", response)
        && strcmp(response, "OK CLOSED") == 0;
    (void)close(transport.socket_fd);
    if (!ok) {
        qwc_secure_zero(response, sizeof(response));
        qwc_secure_zero(boundary, sizeof(boundary));
        qwc_secure_zero(&stats, sizeof(stats));
        return 1;
    }

    printf(
        "{\"result\":\"PASS\","
        "\"protocol\":\"CATVM_QANF_SMALL_WALL_COMPARE_1\","
        "\"warm_transactions\":%zu,"
        "\"timed_transactions\":%zu,"
        "\"wall_ns\":%llu,"
        "\"service_cpu_ns\":%llu,"
        "\"seal_cpu_ns\":%llu,"
        "\"timed_request_packets\":%llu,"
        "\"timed_response_packets\":%llu,"
        "\"timed_request_bytes\":%llu,"
        "\"timed_response_bytes\":%llu,"
        "\"boolean_ands\":%llu,"
        "\"phase_products\":%llu,"
        "\"carrier_reads\":%llu,"
        "\"phase_cell_updates\":%llu,"
        "\"final_decodes\":%llu,"
        "\"snapshot_loads\":%llu,"
        "\"snapshot_reload_bytes\":%llu,"
        "\"actual_inverse_transactions\":%llu,"
        "\"restoration_generation\":%llu,"
        "\"carrier_creation_count\":%llu,"
        "\"boundaries\":[",
        warm_transactions,
        timed_transactions,
        (unsigned long long)(wall_end_ns - wall_start_ns),
        (unsigned long long)stats.service_cpu_ns,
        (unsigned long long)stats.seal_cpu_ns,
        (unsigned long long)timed_request_packets,
        (unsigned long long)timed_response_packets,
        (unsigned long long)timed_request_bytes,
        (unsigned long long)timed_response_bytes,
        (unsigned long long)stats.boolean_ands,
        (unsigned long long)stats.phase_products,
        (unsigned long long)stats.carrier_reads,
        (unsigned long long)stats.phase_cell_updates,
        (unsigned long long)stats.final_decodes,
        (unsigned long long)stats.snapshot_loads,
        (unsigned long long)stats.snapshot_reload_bytes,
        (unsigned long long)stats.actual_inverse_transactions,
        (unsigned long long)stats.restoration_generation,
        (unsigned long long)stats.carrier_creation_count
    );
    for (size_t variant = 0U; variant < QWC_VARIANTS; ++variant) {
        if (variant != 0U) {
            putchar(',');
        }
        qwc_print_boundary(&boundary[variant]);
    }
    printf(
        "],\"same_public_palindrome_schedule\":true,"
        "\"controller_computes_boundary\":false}\n"
    );
    qwc_secure_zero(response, sizeof(response));
    qwc_secure_zero(boundary, sizeof(boundary));
    qwc_secure_zero(&stats, sizeof(stats));
    return 0;
}
