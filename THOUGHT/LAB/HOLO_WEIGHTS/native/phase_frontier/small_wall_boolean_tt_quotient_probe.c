#define _POSIX_C_SOURCE 200809L

/* Protocol and no-smuggle probe shared by all quotient comparison arms. */

#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define QTP_RESPONSE_CAPACITY 1024U

static int qtp_connect(const char *path) {
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

static int qtp_exchange(
    int descriptor,
    const void *request,
    size_t request_bytes,
    char response[QTP_RESPONSE_CAPACITY]
) {
    if (
        request_bytes == 0U
        || send(
            descriptor,
            request,
            request_bytes,
            MSG_NOSIGNAL
        ) != (ssize_t)request_bytes
    ) {
        return 0;
    }
    const ssize_t received = recv(
        descriptor,
        response,
        QTP_RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= QTP_RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        return 0;
    }
    response[received] = '\0';
    return 1;
}

static int qtp_text(
    int descriptor,
    const char *request,
    char response[QTP_RESPONSE_CAPACITY]
) {
    return qtp_exchange(
        descriptor,
        request,
        strlen(request),
        response
    );
}

int main(int argc, char **argv) {
    if (argc != 4) {
        return 2;
    }
    const int descriptor = qtp_connect(argv[1]);
    if (descriptor < 0) {
        return 2;
    }
    char response[QTP_RESPONSE_CAPACITY] = {0};
    char hello[128];
    const int hello_bytes = snprintf(
        hello,
        sizeof(hello),
        "OK HELLO CATVM_BOOLEAN_TT_QUOTIENT_SMALL_WALL_1 %s %s 2",
        argv[2],
        argv[3]
    );
    int ok =
        hello_bytes > 0
        && (size_t)hello_bytes < sizeof(hello)
        && qtp_text(descriptor, "HELLO", response)
        && strcmp(response, hello) == 0
        && qtp_text(descriptor, "PROJECT H2", response)
        && strcmp(
            response,
            "ERR E_INTERMEDIATE_PROJECTION_DENIED"
        ) == 0
        && qtp_text(descriptor, "PROJECT H7", response)
        && strcmp(
            response,
            "ERR E_INTERMEDIATE_PROJECTION_DENIED"
        ) == 0
        && qtp_text(descriptor, "READ CARRIER", response)
        && strcmp(
            response,
            "ERR E_INTERMEDIATE_PROJECTION_DENIED"
        ) == 0
        && qtp_text(descriptor, "DUMP", response)
        && strcmp(
            response,
            "ERR E_INTERMEDIATE_PROJECTION_DENIED"
        ) == 0;
    static const unsigned char embedded_nul[] = {
        'H', 'E', 'L', 'L', 'O', '\0', 'X'
    };
    ok = ok
        && qtp_exchange(
            descriptor,
            embedded_nul,
            sizeof(embedded_nul),
            response
        )
        && strcmp(response, "ERR E_PROTOCOL") == 0;
    unsigned char oversize[256];
    memset(oversize, 'X', sizeof(oversize));
    ok = ok
        && qtp_exchange(
            descriptor,
            oversize,
            sizeof(oversize),
            response
        )
        && strcmp(response, "ERR E_PROTOCOL") == 0
        && qtp_text(descriptor, "EXECUTE 2", response)
        && strcmp(response, "ERR E_PROTOCOL") == 0
        && qtp_text(descriptor, "EXECUTE NULL", response)
        && strcmp(response, "ERR E_PROTOCOL") == 0
        && qtp_text(descriptor, "EXECUTE 0", response)
        && strncmp(response, "OK FINAL 0 ", 11U) == 0
        && qtp_text(descriptor, "SHUTDOWN", response)
        && strcmp(response, "OK CLOSED") == 0;
    (void)close(descriptor);
    if (!ok) {
        return 1;
    }
    printf(
        "{\"result\":\"PASS\","
        "\"intermediate_projection_denied\":true,"
        "\"carrier_read_denied\":true,"
        "\"debug_dump_denied\":true,"
        "\"embedded_nul_denied\":true,"
        "\"oversize_packet_denied\":true,"
        "\"unknown_family_denied\":true,"
        "\"null_carrier_request_denied\":true,"
        "\"final_boundary_receipt_only\":true}\n"
    );
    return 0;
}
