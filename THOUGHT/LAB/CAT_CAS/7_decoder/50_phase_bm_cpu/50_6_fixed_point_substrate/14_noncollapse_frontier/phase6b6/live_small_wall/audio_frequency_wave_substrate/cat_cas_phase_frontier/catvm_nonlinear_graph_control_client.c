#define _GNU_SOURCE

/* Test-only protocol client. It contains no phase or graph implementation. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define CNX_CAPACITY 512U

static int cnx_connect(const char *path) {
    if (strlen(path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
        return -1;
    }
    const int descriptor = socket(
        AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0
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

static int cnx_exchange(
    int descriptor,
    const char *request,
    char response[CNX_CAPACITY]
) {
    const size_t bytes = strlen(request);
    if (
        send(descriptor, request, bytes, MSG_NOSIGNAL)
            != (ssize_t)bytes
    ) {
        return 0;
    }
    const ssize_t received = recv(
        descriptor, response, CNX_CAPACITY - 1U, MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= CNX_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        return 0;
    }
    response[received] = '\0';
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        return 2;
    }
    const int descriptor = cnx_connect(argv[1]);
    if (descriptor < 0) {
        return 2;
    }
    char response[CNX_CAPACITY] = {0};
    if (
        !cnx_exchange(descriptor, "HELLO", response)
        || strncmp(
            response,
            "OK HELLO CATVM_NONLINEAR_PHASE_GRAPH_1 ",
            strlen("OK HELLO CATVM_NONLINEAR_PHASE_GRAPH_1 ")
        ) != 0
    ) {
        (void)close(descriptor);
        return 2;
    }
    int pass = 0;
    if (
        strcmp(argv[2], "missing") == 0
        || strcmp(argv[2], "wrong") == 0
        || strcmp(argv[2], "reordered") == 0
    ) {
        pass = (
            cnx_exchange(descriptor, "EXECUTE 0", response)
            && strcmp(response, "ERR E_RESTORATION_DETECTED") == 0
        );
    } else if (strcmp(argv[2], "snapshot") == 0) {
        unsigned long long generation0 = 1ULL;
        unsigned long long generation1 = 1ULL;
        int program0 = -1;
        int program1 = -1;
        char hash0[17] = {0};
        char hash1[17] = {0};
        double probability0 = 0.0;
        double probability1 = 0.0;
        unsigned long long carrier0 = 0ULL;
        unsigned long long carrier1 = 0ULL;
        pass = (
            cnx_exchange(descriptor, "EXECUTE 0", response)
            && sscanf(
                response,
                "OK SNAPSHOT %d %llu %16s %lf %llu",
                &program0,
                &generation0,
                hash0,
                &probability0,
                &carrier0
            ) == 5
            && cnx_exchange(descriptor, "EXECUTE 1", response)
            && sscanf(
                response,
                "OK SNAPSHOT %d %llu %16s %lf %llu",
                &program1,
                &generation1,
                hash1,
                &probability1,
                &carrier1
            ) == 5
            && program0 == 0
            && program1 == 1
            && generation0 == 0ULL
            && generation1 == 0ULL
            && strcmp(hash0, hash1) != 0
            && probability0 >= 0.0
            && probability0 <= 1.0
            && probability1 >= 0.0
            && probability1 <= 1.0
            && carrier0 == 1ULL
            && carrier1 == 1ULL
            && cnx_exchange(descriptor, "SHUTDOWN", response)
            && strcmp(response, "OK CLOSED") == 0
        );
    }
    (void)close(descriptor);
    if (!pass) {
        return 2;
    }
    printf(
        "{\"result\":\"PASS\",\"control\":\"%s\","
        "\"production_protocol_selector\":false}\n",
        argv[2]
    );
    return 0;
}
