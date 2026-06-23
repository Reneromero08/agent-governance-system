#ifndef CATCAS_CAPTURED_FILE_H
#define CATCAS_CAPTURED_FILE_H

#include <stddef.h>
#include <stdint.h>

#define CAPTURED_SHA256_LEN 64
#define CAPTURED_MAX_AUTHORIZATION    (1 << 20)
#define CAPTURED_MAX_SOURCE_BUNDLE    (1 << 20)
#define CAPTURED_MAX_SESSION_MANIFEST (1 << 16)
#define CAPTURED_MAX_SESSION_JSON     (1 << 16)
#define CAPTURED_MAX_WINDOWS_JSONL    (4 << 20)

typedef struct {
    unsigned char *bytes;
    size_t         size;
    char           sha256[CAPTURED_SHA256_LEN + 1];
} CapturedFile;

int  capture_file(const char *path, CapturedFile *out, size_t max_size);
int  hash_captured(const unsigned char *bytes, size_t size, char out_digest[CAPTURED_SHA256_LEN + 1]);
void free_captured(CapturedFile *cf);

#endif
