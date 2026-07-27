#define _POSIX_C_SOURCE 200809L

/* Protocol/no-smuggle probe shared by all three comparison services. */

#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define QWP_RESPONSE_CAPACITY 512U

static int qwp_connect(const char *path) {
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

static int qwp_exchange(
    int descriptor,
    const void *request,
    size_t request_bytes,
    char response[QWP_RESPONSE_CAPACITY]
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
        QWP_RESPONSE_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= QWP_RESPONSE_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        return 0;
    }
    response[received] = '\0';
    return 1;
}

static int qwp_text(
    int descriptor,
    const char *request,
    char response[QWP_RESPONSE_CAPACITY]
) {
    return qwp_exchange(
        descriptor,
        request,
        strlen(request),
        response
    );
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    const int descriptor = qwp_connect(argv[1]);
    if (descriptor < 0) {
        return 2;
    }
    char response[QWP_RESPONSE_CAPACITY] = {0};
    int ok = qwp_text(descriptor, "HELLO", response)
        && strcmp(
            response,
            "OK HELLO CATVM_QANF_SMALL_WALL_COMPARE_1 4 5"
        ) == 0
        && qwp_text(descriptor, "PROJECT H", response)
        && strcmp(
            response,
            "ERR E_INTERMEDIATE_PROJECTION_DENIED"
        ) == 0
        && qwp_text(descriptor, "PROJECT Z", response)
        && strcmp(
            response,
            "ERR E_INTERMEDIATE_PROJECTION_DENIED"
        ) == 0;
    static const unsigned char embedded_nul[] = {
        'H', 'E', 'L', 'L', 'O', '\0', 'X'
    };
    ok = ok
        && qwp_exchange(
            descriptor,
            embedded_nul,
            sizeof(embedded_nul),
            response
        )
        && strcmp(response, "ERR E_PROTOCOL") == 0;
    unsigned char oversize[256];
    memset(oversize, 'X', sizeof(oversize));
    ok = ok
        && qwp_exchange(
            descriptor,
            oversize,
            sizeof(oversize),
            response
        )
        && strcmp(response, "ERR E_PROTOCOL") == 0
        && qwp_text(descriptor, "EXECUTE 4", response)
        && strcmp(response, "ERR E_PROTOCOL") == 0
        && qwp_text(descriptor, "EXECUTE 0", response)
        && strncmp(response, "OK FINAL 0 5 ", 13U) == 0
        && qwp_text(descriptor, "SHUTDOWN", response)
        && strcmp(response, "OK CLOSED") == 0;
    (void)close(descriptor);
    if (!ok) {
        return 1;
    }
    printf(
        "{\"result\":\"PASS\","
        "\"intermediate_projection_denied\":true,"
        "\"embedded_nul_denied\":true,"
        "\"oversize_packet_denied\":true,"
        "\"unknown_variant_denied\":true,"
        "\"final_boundary_only\":true}\n"
    );
    return 0;
}
