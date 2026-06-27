#define _GNU_SOURCE
#include "captured_file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int emit_capture(const CapturedFile *captured) {
    if (printf("%s\n%zu\n", captured->sha256, captured->size) < 0) return -1;
    if (captured->size != 0 &&
        fwrite(captured->bytes, 1, captured->size, stdout) != captured->size) {
        return -1;
    }
    return fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    if (!strcmp(argv[1], "hash")) {
        if (argc != 3) return 2;
        char digest[CAPTURED_SHA256_LEN + 1];
        if (hash_captured((const unsigned char *)argv[2], strlen(argv[2]), digest)) return 3;
        puts(digest);
        return 0;
    }
    if (!strcmp(argv[1], "hash-file")) {
        if (argc != 3) return 2;
        char digest[CAPTURED_SHA256_LEN + 1];
        if (hash_file_streaming(argv[2], digest)) return 3;
        puts(digest);
        return 0;
    }
    if (!strcmp(argv[1], "hash-self")) {
        if (argc != 2) return 2;
        char path[64];
        char digest[CAPTURED_SHA256_LEN + 1];
        int length = snprintf(path, sizeof(path), "/proc/%ld/exe", (long)getpid());
        if (length < 0 || (size_t)length >= sizeof(path) ||
            hash_file_streaming(path, digest)) return 3;
        puts(digest);
        return 0;
    }
    if (!strcmp(argv[1], "capture")) {
        if (argc != 3) return 2;
        CapturedFile captured = {0};
        if (capture_file(argv[2], &captured, 16u << 20)) return 3;
        int result = emit_capture(&captured);
        free_captured(&captured);
        return result ? 4 : 0;
    }
    if (!strcmp(argv[1], "capture-replace")) {
        if (argc != 4) return 2;
        CapturedFile captured = {0};
        if (capture_file(argv[2], &captured, 16u << 20)) return 3;
        if (rename(argv[3], argv[2]) != 0) {
            free_captured(&captured);
            return 4;
        }
        int result = emit_capture(&captured);
        free_captured(&captured);
        return result ? 5 : 0;
    }
    if (!strcmp(argv[1], "write")) {
        if (argc != 4) return 2;
        CapturedFile captured = {0};
        if (capture_file(argv[2], &captured, 16u << 20)) return 3;
        int result = write_captured_exclusive(argv[3], &captured);
        free_captured(&captured);
        return result ? 4 : 0;
    }
    return 2;
}
