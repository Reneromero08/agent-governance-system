#define _POSIX_C_SOURCE 200809L

/*
 * Arm-neutral controller for the growing Boolean-TT quotient Small-Wall
 * triad.  It carries only the public family schedule and transport/accounting
 * logic.  It contains no quotient generator, carrier, precomputed result, or
 * extensional result storage.
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

#define QTC_PROTOCOL "CATVM_BOOLEAN_TT_QUOTIENT_SMALL_WALL_1"
#define QTC_RESPONSE_CAPACITY 1024U
#define QTC_FAMILIES 2U
#define QTC_MAX_TRANSACTIONS 1000000U

struct qtc_transport {
    int socket_fd;
    uint64_t request_packets;
    uint64_t response_packets;
    uint64_t request_bytes;
    uint64_t response_bytes;
};

struct qtc_receipt {
    uint64_t plan_hash;
    uint64_t boundary_hash;
    size_t ones;
    size_t cells;
    int present;
};

struct qtc_stats {
    uint64_t transactions;
    uint64_t direct_cells;
    uint64_t direct_predicates;
    uint64_t logical_phase_ands;
    uint64_t logical_phase_ors;
    uint64_t carrier_reads;
    uint64_t phase_cell_updates;
    uint64_t quotient_member_terms;
    uint64_t final_decodes;
    uint64_t projected_bytes;
    uint64_t comparison_snapshot_bytes;
    uint64_t restoration_scan_cells;
    uint64_t snapshot_loads;
    uint64_t snapshot_reload_bytes;
    uint64_t actual_inverse_transactions;
    uint64_t restoration_generation;
    uint64_t carrier_creation_count;
    uint64_t service_cpu_ns;
    uint64_t carrier_seal_cpu_ns;
    double maximum_restoration_error;
};

static void qtc_secure_zero(void *memory, size_t bytes) {
    volatile unsigned char *cursor = memory;
    while (bytes > 0U) {
        *cursor = 0U;
        ++cursor;
        --bytes;
    }
}

static int qtc_monotonic_raw_ns(uint64_t *time_ns) {
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

static int qtc_parse_size(const char *text, size_t *value) {
    if (text == NULL || text[0] < '0' || text[0] > '9') {
        return 0;
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long long parsed = strtoull(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed > SIZE_MAX
    ) {
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}

static int qtc_connect(const char *path) {
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

static int qtc_exchange(
    struct qtc_transport *transport,
    const char *request,
    char response[QTC_RESPONSE_CAPACITY]
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
    transport->request_bytes += (uint64_t)bytes;
    const ssize_t received = recv(
        transport->socket_fd,
        response,
        QTC_RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= QTC_RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        qtc_secure_zero(response, QTC_RESPONSE_CAPACITY);
        return 0;
    }
    response[received] = '\0';
    ++transport->response_packets;
    transport->response_bytes += (uint64_t)received;
    return 1;
}

static int qtc_parse_hex64(const char text[17], uint64_t *value) {
    uint64_t parsed = 0U;
    for (size_t index = 0U; index < 16U; ++index) {
        const unsigned char byte = (unsigned char)text[index];
        unsigned int digit = 0U;
        if (byte >= '0' && byte <= '9') {
            digit = (unsigned int)(byte - '0');
        } else if (byte >= 'a' && byte <= 'f') {
            digit = (unsigned int)(byte - 'a') + 10U;
        } else {
            return 0;
        }
        parsed = (parsed << 4U) | (uint64_t)digit;
    }
    if (text[16] != '\0') {
        return 0;
    }
    *value = parsed;
    return 1;
}

static int qtc_parse_receipt(
    const char *response,
    size_t expected_family,
    struct qtc_receipt *receipt
) {
    size_t family = 0U;
    size_t ones = 0U;
    size_t cells = 0U;
    char plan[17] = {0};
    char boundary[17] = {0};
    char trailing = '\0';
    const int fields = sscanf(
        response,
        "OK FINAL %zu %16[0-9a-f] %16[0-9a-f] %zu %zu %c",
        &family,
        plan,
        boundary,
        &ones,
        &cells,
        &trailing
    );
    struct qtc_receipt parsed = {
        .ones = ones,
        .cells = cells,
        .present = 1
    };
    if (
        fields != 5
        || family != expected_family
        || cells == 0U
        || ones > cells
        || strlen(plan) != 16U
        || strlen(boundary) != 16U
        || !qtc_parse_hex64(plan, &parsed.plan_hash)
        || !qtc_parse_hex64(boundary, &parsed.boundary_hash)
    ) {
        return 0;
    }
    if (receipt->present) {
        return
            receipt->plan_hash == parsed.plan_hash
            && receipt->boundary_hash == parsed.boundary_hash
            && receipt->ones == parsed.ones
            && receipt->cells == parsed.cells;
    }
    *receipt = parsed;
    return 1;
}

static int qtc_parse_stats(
    const char *response,
    struct qtc_stats *stats
) {
    unsigned long long value[19];
    double maximum_restoration_error = 0.0;
    char trailing = '\0';
    const int fields = sscanf(
        response,
        "OK STATS "
        "%llu %llu %llu %llu %llu %llu %llu %llu %llu "
        "%llu %llu %llu %llu %llu %llu %llu %llu %llu "
        "%llu %lf %c",
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
        &value[13],
        &value[14],
        &value[15],
        &value[16],
        &value[17],
        &value[18],
        &maximum_restoration_error,
        &trailing
    );
    if (fields != 20) {
        return 0;
    }
    stats->transactions = (uint64_t)value[0];
    stats->direct_cells = (uint64_t)value[1];
    stats->direct_predicates = (uint64_t)value[2];
    stats->logical_phase_ands = (uint64_t)value[3];
    stats->logical_phase_ors = (uint64_t)value[4];
    stats->carrier_reads = (uint64_t)value[5];
    stats->phase_cell_updates = (uint64_t)value[6];
    stats->quotient_member_terms = (uint64_t)value[7];
    stats->final_decodes = (uint64_t)value[8];
    stats->projected_bytes = (uint64_t)value[9];
    stats->comparison_snapshot_bytes = (uint64_t)value[10];
    stats->restoration_scan_cells = (uint64_t)value[11];
    stats->snapshot_loads = (uint64_t)value[12];
    stats->snapshot_reload_bytes = (uint64_t)value[13];
    stats->actual_inverse_transactions = (uint64_t)value[14];
    stats->restoration_generation = (uint64_t)value[15];
    stats->carrier_creation_count = (uint64_t)value[16];
    stats->service_cpu_ns = (uint64_t)value[17];
    stats->carrier_seal_cpu_ns = (uint64_t)value[18];
    stats->maximum_restoration_error = maximum_restoration_error;
    return 1;
}

static int qtc_execute_pattern(
    struct qtc_transport *transport,
    size_t transactions,
    struct qtc_receipt receipt[QTC_FAMILIES],
    char response[QTC_RESPONSE_CAPACITY]
) {
    static const size_t pattern[4] = {0U, 1U, 1U, 0U};
    if ((transactions % 4U) != 0U) {
        return 0;
    }
    for (size_t index = 0U; index < transactions; ++index) {
        const size_t family = pattern[index % 4U];
        char command[16];
        const int written = snprintf(
            command,
            sizeof(command),
            "EXECUTE %zu",
            family
        );
        if (
            written <= 0
            || (size_t)written >= sizeof(command)
            || !qtc_exchange(transport, command, response)
            || !qtc_parse_receipt(
                response, family, &receipt[family]
            )
        ) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        return 2;
    }
    size_t width = 0U;
    size_t depth = 0U;
    size_t warm_transactions = 0U;
    size_t timed_transactions = 0U;
    if (
        !qtc_parse_size(argv[2], &width)
        || !qtc_parse_size(argv[3], &depth)
        || !qtc_parse_size(argv[4], &warm_transactions)
        || !qtc_parse_size(argv[5], &timed_transactions)
        || width < 4U
        || width > 16U
        || depth < 2U
        || depth > 8U
        || depth > width
        || warm_transactions > QTC_MAX_TRANSACTIONS
        || timed_transactions > QTC_MAX_TRANSACTIONS
        || (warm_transactions % 4U) != 0U
        || (timed_transactions % 4U) != 0U
    ) {
        return 2;
    }
    struct qtc_transport transport = {
        .socket_fd = qtc_connect(argv[1])
    };
    if (transport.socket_fd < 0) {
        return 2;
    }
    char response[QTC_RESPONSE_CAPACITY] = {0};
    char hello[128];
    const int hello_bytes = snprintf(
        hello,
        sizeof(hello),
        "OK HELLO %s %zu %zu %u",
        QTC_PROTOCOL,
        width,
        depth,
        QTC_FAMILIES
    );
    struct qtc_receipt receipt[QTC_FAMILIES];
    memset(receipt, 0, sizeof(receipt));
    struct qtc_stats stats;
    memset(&stats, 0, sizeof(stats));
    int ok =
        hello_bytes > 0
        && (size_t)hello_bytes < sizeof(hello)
        && qtc_exchange(&transport, "HELLO", response)
        && strcmp(response, hello) == 0
        && qtc_execute_pattern(
            &transport,
            warm_transactions,
            receipt,
            response
        )
        && qtc_exchange(&transport, "RESET", response)
        && strcmp(response, "OK RESET") == 0;

    const uint64_t request_packets_before =
        transport.request_packets;
    const uint64_t response_packets_before =
        transport.response_packets;
    const uint64_t request_bytes_before =
        transport.request_bytes;
    const uint64_t response_bytes_before =
        transport.response_bytes;
    uint64_t wall_start_ns = 0U;
    uint64_t wall_end_ns = 0U;
    ok = ok
        && qtc_monotonic_raw_ns(&wall_start_ns)
        && qtc_execute_pattern(
            &transport,
            timed_transactions,
            receipt,
            response
        )
        && qtc_monotonic_raw_ns(&wall_end_ns)
        && wall_end_ns >= wall_start_ns
        && qtc_exchange(&transport, "STATS", response)
        && qtc_parse_stats(response, &stats)
        && stats.transactions == timed_transactions;

    const uint64_t timed_request_packets =
        transport.request_packets - request_packets_before - 1U;
    const uint64_t timed_response_packets =
        transport.response_packets - response_packets_before - 1U;
    const uint64_t timed_request_bytes =
        transport.request_bytes
        - request_bytes_before
        - (uint64_t)strlen("STATS");
    const uint64_t timed_response_bytes =
        transport.response_bytes
        - response_bytes_before
        - (uint64_t)strlen(response);
    ok = ok
        && receipt[0].present
        && receipt[1].present
        && receipt[0].plan_hash == receipt[1].plan_hash
        && receipt[0].cells == receipt[1].cells
        && receipt[0].boundary_hash != receipt[1].boundary_hash
        && qtc_exchange(&transport, "SHUTDOWN", response)
        && strcmp(response, "OK CLOSED") == 0;
    (void)close(transport.socket_fd);
    if (!ok) {
        qtc_secure_zero(response, sizeof(response));
        qtc_secure_zero(receipt, sizeof(receipt));
        qtc_secure_zero(&stats, sizeof(stats));
        return 1;
    }

    printf(
        "{\"result\":\"PASS\","
        "\"protocol\":\"%s\","
        "\"width\":%zu,"
        "\"depth\":%zu,"
        "\"warm_transactions\":%zu,"
        "\"timed_transactions\":%zu,"
        "\"wall_ns\":%llu,"
        "\"service_cpu_ns\":%llu,"
        "\"carrier_seal_cpu_ns\":%llu,"
        "\"timed_request_packets\":%llu,"
        "\"timed_response_packets\":%llu,"
        "\"timed_request_bytes\":%llu,"
        "\"timed_response_bytes\":%llu,"
        "\"direct_cells\":%llu,"
        "\"direct_predicates\":%llu,"
        "\"logical_phase_ands\":%llu,"
        "\"logical_phase_ors\":%llu,"
        "\"carrier_reads\":%llu,"
        "\"phase_cell_updates\":%llu,"
        "\"quotient_member_terms\":%llu,"
        "\"final_decodes\":%llu,"
        "\"projected_bytes\":%llu,"
        "\"comparison_snapshot_bytes\":%llu,"
        "\"restoration_scan_cells\":%llu,"
        "\"snapshot_loads\":%llu,"
        "\"snapshot_reload_bytes\":%llu,"
        "\"actual_inverse_transactions\":%llu,"
        "\"restoration_generation\":%llu,"
        "\"carrier_creation_count\":%llu,"
        "\"maximum_restoration_error\":%.17g,"
        "\"receipts\":["
        "{\"family\":\"AND\",\"plan_fnv1a64\":\"%016llx\","
        "\"boundary_fnv1a64\":\"%016llx\","
        "\"ones\":%zu,\"cells\":%zu},"
        "{\"family\":\"OR\",\"plan_fnv1a64\":\"%016llx\","
        "\"boundary_fnv1a64\":\"%016llx\","
        "\"ones\":%zu,\"cells\":%zu}],"
        "\"same_public_palindrome_schedule\":true,"
        "\"controller_computes_boundary\":false}\n",
        QTC_PROTOCOL,
        width,
        depth,
        warm_transactions,
        timed_transactions,
        (unsigned long long)(wall_end_ns - wall_start_ns),
        (unsigned long long)stats.service_cpu_ns,
        (unsigned long long)stats.carrier_seal_cpu_ns,
        (unsigned long long)timed_request_packets,
        (unsigned long long)timed_response_packets,
        (unsigned long long)timed_request_bytes,
        (unsigned long long)timed_response_bytes,
        (unsigned long long)stats.direct_cells,
        (unsigned long long)stats.direct_predicates,
        (unsigned long long)stats.logical_phase_ands,
        (unsigned long long)stats.logical_phase_ors,
        (unsigned long long)stats.carrier_reads,
        (unsigned long long)stats.phase_cell_updates,
        (unsigned long long)stats.quotient_member_terms,
        (unsigned long long)stats.final_decodes,
        (unsigned long long)stats.projected_bytes,
        (unsigned long long)stats.comparison_snapshot_bytes,
        (unsigned long long)stats.restoration_scan_cells,
        (unsigned long long)stats.snapshot_loads,
        (unsigned long long)stats.snapshot_reload_bytes,
        (unsigned long long)stats.actual_inverse_transactions,
        (unsigned long long)stats.restoration_generation,
        (unsigned long long)stats.carrier_creation_count,
        stats.maximum_restoration_error,
        (unsigned long long)receipt[0].plan_hash,
        (unsigned long long)receipt[0].boundary_hash,
        receipt[0].ones,
        receipt[0].cells,
        (unsigned long long)receipt[1].plan_hash,
        (unsigned long long)receipt[1].boundary_hash,
        receipt[1].ones,
        receipt[1].cells
    );
    qtc_secure_zero(response, sizeof(response));
    qtc_secure_zero(receipt, sizeof(receipt));
    qtc_secure_zero(&stats, sizeof(stats));
    return 0;
}
