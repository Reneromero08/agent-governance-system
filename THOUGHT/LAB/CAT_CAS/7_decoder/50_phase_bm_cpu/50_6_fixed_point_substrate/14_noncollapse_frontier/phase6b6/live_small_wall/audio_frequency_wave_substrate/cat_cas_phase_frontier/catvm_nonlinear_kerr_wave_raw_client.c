#define _POSIX_C_SOURCE 200809L

/*
 * Test-only raw CATVM response recorder. It has no wave implementation and
 * prints the exact AF_UNIX payload returned for each public command.
 */

#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define KWR_CAPACITY 512U

static int kwr_connect(const char *path) {
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

static int kwr_exchange(
    int descriptor,
    const char *request,
    char response[KWR_CAPACITY]
) {
    const size_t bytes = strlen(request);
    if (
        bytes == 0U
        || send(descriptor, request, bytes, MSG_NOSIGNAL)
            != (ssize_t)bytes
    ) {
        return 0;
    }
    const ssize_t received = recv(
        descriptor,
        response,
        KWR_CAPACITY - 1U,
        MSG_TRUNC
    );
    if (
        received <= 0
        || (size_t)received >= KWR_CAPACITY
        || memchr(response, '\0', (size_t)received) != NULL
    ) {
        return 0;
    }
    response[received] = '\0';
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    const int descriptor = kwr_connect(argv[1]);
    if (descriptor < 0) {
        return 2;
    }
    static const char *const request[] = {
        "HELLO",
        "PROJECT CELL 0",
        "PROJECT WAVE",
        "PROJECT KERR INPUT",
        "DUMP",
        "STATE DETAIL",
        "EXECUTE NULL",
        "UNKNOWN",
        "EXECUTE 0",
        "EXECUTE 1",
        "SHUTDOWN"
    };
    char response[KWR_CAPACITY] = {0};
    for (
        size_t index = 0U;
        index < sizeof(request) / sizeof(request[0]);
        ++index
    ) {
        if (!kwr_exchange(descriptor, request[index], response)) {
            (void)close(descriptor);
            return 2;
        }
        if (printf("%s\n", response) < 0) {
            (void)close(descriptor);
            return 2;
        }
        memset(response, 0, sizeof(response));
    }
    (void)close(descriptor);
    return 0;
}
