#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
BASE = ROOT / "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier"
RUNTIME = BASE / "holo_runtime_v2"
RUNNER = RUNTIME / "combined_pdn_runner.c"
HARDWARE = RUNTIME / "combined_pdn_hardware.c"
CAPTURE_H = RUNTIME / "captured_file.h"
CAPTURE_C = RUNTIME / "captured_file.c"
FIXTURE = RUNTIME / "captured_file_fixture.c"
TEST_CAPTURE = RUNTIME / "test_captured_file.py"
TEST_RUNNER = RUNTIME / "test_combined_pdn_runner.py"
MAKEFILE = RUNTIME / "Makefile"
STRICT_WF = ROOT / ".github/workflows/phase6-v2-strict-qualification.yml"
COMBINED_WF = ROOT / ".github/workflows/phase6-combined-campaign-plan.yml"

CAPTURE_H_TEXT = r'''#ifndef CATCAS_CAPTURED_FILE_H
#define CATCAS_CAPTURED_FILE_H

#include <stddef.h>
#include <stdint.h>

#define CAPTURED_SHA256_LEN 64
#define CAPTURED_MAX_AUTHORIZATION    (1u << 20)
#define CAPTURED_MAX_SOURCE_BUNDLE    (1u << 20)
#define CAPTURED_MAX_SESSION_MANIFEST (1u << 16)
#define CAPTURED_MAX_SESSION_JSON     (1u << 16)
#define CAPTURED_MAX_WINDOWS_JSONL    (4u << 20)

typedef struct {
    unsigned char *bytes;
    size_t size;
    char sha256[CAPTURED_SHA256_LEN + 1];
} CapturedFile;

int capture_file(const char *path, CapturedFile *out, size_t max_size);
int hash_captured(const unsigned char *bytes, size_t size,
                  char out_digest[CAPTURED_SHA256_LEN + 1]);
int write_captured_exclusive(const char *path, const CapturedFile *captured);
void free_captured(CapturedFile *captured);

#endif
'''

CAPTURE_C_TEXT = r'''#define _GNU_SOURCE
#include "captured_file.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef O_BINARY
#define O_BINARY 0
#endif

static const uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#define SHA256_ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_transform(uint32_t state[8], const unsigned char block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = SHA256_ROTR(w[i - 15], 7) ^
                      SHA256_ROTR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = SHA256_ROTR(w[i - 2], 17) ^
                      SHA256_ROTR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t s1 = SHA256_ROTR(e, 6) ^ SHA256_ROTR(e, 11) ^
                      SHA256_ROTR(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + ch + sha256_k[i] + w[i];
        uint32_t s0 = SHA256_ROTR(a, 2) ^ SHA256_ROTR(a, 13) ^
                      SHA256_ROTR(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

int hash_captured(const unsigned char *bytes, size_t size,
                  char out_digest[CAPTURED_SHA256_LEN + 1]) {
    if (!out_digest || (!bytes && size != 0) || size > UINT64_MAX / 8u) {
        return -1;
    }
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    size_t offset = 0;
    while (offset + 64 <= size) {
        sha256_transform(state, bytes + offset);
        offset += 64;
    }
    unsigned char tail[128] = {0};
    size_t remainder = size - offset;
    if (remainder != 0) {
        memcpy(tail, bytes + offset, remainder);
    }
    tail[remainder] = 0x80;
    size_t blocks = remainder >= 56 ? 2u : 1u;
    uint64_t bit_length = (uint64_t)size * 8u;
    size_t length_offset = blocks * 64u - 8u;
    for (int i = 0; i < 8; i++) {
        tail[length_offset + (size_t)i] =
            (unsigned char)(bit_length >> (56 - 8 * i));
    }
    for (size_t block = 0; block < blocks; block++) {
        sha256_transform(state, tail + block * 64u);
    }
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 8; i++) {
        for (int byte = 0; byte < 4; byte++) {
            unsigned value = (state[i] >> (24 - 8 * byte)) & 0xffu;
            size_t index = (size_t)i * 8u + (size_t)byte * 2u;
            out_digest[index] = hex[value >> 4];
            out_digest[index + 1] = hex[value & 0xfu];
        }
    }
    out_digest[CAPTURED_SHA256_LEN] = 0;
    return 0;
}

int capture_file(const char *path, CapturedFile *out, size_t max_size) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!path || !path[0] || max_size == 0) return -1;

    int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_BINARY);
    if (fd < 0) return -1;

    struct stat before;
    if (fstat(fd, &before) != 0 || !S_ISREG(before.st_mode) ||
        before.st_size < 0 || (uintmax_t)before.st_size > (uintmax_t)max_size ||
        (uintmax_t)before.st_size >= (uintmax_t)SIZE_MAX) {
        close(fd);
        return -1;
    }

    size_t size = (size_t)before.st_size;
    unsigned char *bytes = malloc(size + 1u);
    if (!bytes) {
        close(fd);
        return -1;
    }

    size_t total = 0;
    while (total < size) {
        ssize_t count = read(fd, bytes + total, size - total);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            free(bytes);
            close(fd);
            return -1;
        }
        total += (size_t)count;
    }

    unsigned char probe;
    ssize_t extra;
    do {
        extra = read(fd, &probe, 1);
    } while (extra < 0 && errno == EINTR);
    if (extra != 0) {
        free(bytes);
        close(fd);
        return -1;
    }

    struct stat after;
    if (fstat(fd, &after) != 0 || !S_ISREG(after.st_mode) ||
        after.st_dev != before.st_dev || after.st_ino != before.st_ino ||
        after.st_size != before.st_size) {
        free(bytes);
        close(fd);
        return -1;
    }

    int close_result = close(fd);
    if (close_result != 0) {
        free(bytes);
        return -1;
    }

    bytes[size] = 0;
    if (hash_captured(bytes, size, out->sha256) != 0) {
        free(bytes);
        return -1;
    }
    out->bytes = bytes;
    out->size = size;
    return 0;
}

int write_captured_exclusive(const char *path, const CapturedFile *captured) {
    if (!path || !path[0] || !captured || (!captured->bytes && captured->size != 0)) {
        return -1;
    }
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_BINARY, 0644);
    if (fd < 0) return -1;

    int failed = 0;
    size_t total = 0;
    while (total < captured->size) {
        ssize_t count = write(fd, captured->bytes + total, captured->size - total);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            failed = 1;
            break;
        }
        total += (size_t)count;
    }
    if (!failed && fsync(fd) != 0) failed = 1;
    if (close(fd) != 0) failed = 1;
    if (failed) {
        unlink(path);
        return -1;
    }
    return 0;
}

void free_captured(CapturedFile *captured) {
    if (!captured) return;
    free(captured->bytes);
    memset(captured, 0, sizeof(*captured));
}
'''

FIXTURE_TEXT = r'''#define _GNU_SOURCE
#include "captured_file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
'''

TEST_CAPTURE_TEXT = r'''from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class CapturedFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temp.name) / "captured_file_fixture"
        result = subprocess.run(
            ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
             str(HERE / "captured_file_fixture.c"), str(HERE / "captured_file.c"),
             "-o", str(cls.binary)],
            text=True, capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def run_fixture(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run([str(self.binary), *args], capture_output=True)

    def test_sha256_known_vectors(self) -> None:
        vectors = (
            "",
            "abc",
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        )
        for value in vectors:
            with self.subTest(value=value):
                result = self.run_fixture("hash", value)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(result.stdout.decode().strip(), hashlib.sha256(value.encode()).hexdigest())

    def test_capture_survives_original_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "input"
            replacement = root / "replacement"
            original.write_bytes(b"before-custody")
            replacement.write_bytes(b"after-replacement")
            result = self.run_fixture("capture-replace", str(original), str(replacement))
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            digest, size, payload = result.stdout.split(b"\n", 2)
            self.assertEqual(int(size), len(b"before-custody"))
            self.assertEqual(payload, b"before-custody")
            self.assertEqual(digest.decode(), hashlib.sha256(payload).hexdigest())
            self.assertEqual(original.read_bytes(), b"after-replacement")

    @unittest.skipIf(os.name == "nt", "O_NOFOLLOW is a Linux custody requirement")
    def test_symlink_capture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "target"
            link = root / "link"
            target.write_bytes(b"bound")
            link.symlink_to(target.name)
            result = self.run_fixture("capture", str(link))
            self.assertNotEqual(result.returncode, 0)

    def test_exclusive_writer_preserves_bytes_and_rejects_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            destination = root / "destination"
            source.write_bytes(b"captured-output")
            first = self.run_fixture("write", str(source), str(destination))
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            second = self.run_fixture("write", str(source), str(destination))
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(destination.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
'''


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_runner() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    helper_anchor = '''static int object_member_count(const char *json, const char *object_name) {
    const char *start, *end;
    if (object_bounds(json, object_name, &start, &end)) return -1;
    int count = 0, in_string = 0, nested = 0;
    for (const char *p = start + 1; p < end; p++) {
        if (*p == '"' && p[-1] != '\\') in_string = !in_string;
        if (in_string) continue;
        if (*p == '{' || *p == '[') nested++;
        else if (*p == '}' || *p == ']') nested--;
        else if (*p == ':' && nested == 0) count++;
    }
    return count;
}
'''
    helper_new = helper_anchor + r'''
static int direct_object_member_count(const char *start, const char *end) {
    int count = 0, in_string = 0, nested = 0;
    for (const char *p = start + 1; p < end; p++) {
        if (*p == '"' && p[-1] != '\\') in_string = !in_string;
        if (in_string) continue;
        if (*p == '{' || *p == '[') nested++;
        else if (*p == '}' || *p == ']') nested--;
        else if (*p == ':' && nested == 0) count++;
    }
    return count;
}

static int bounded_long(const char *start, const char *end,
                        const char *name, long *out) {
    const char *p = object_value(start, end, name);
    if (!p) return -1;
    errno = 0;
    char *tail = NULL;
    long value = strtol(p, &tail, 10);
    if (errno == ERANGE || tail == p || tail > end || !token_end(*tail)) return -1;
    *out = value;
    return 0;
}

static int bounded_sha256(const char *start, const char *end,
                          const char *name, char out[65]) {
    const char *p = object_value(start, end, name);
    if (!p || *p++ != '"') return -1;
    for (size_t i = 0; i < 64; i++) {
        if (p + i >= end || !((p[i] >= '0' && p[i] <= '9') ||
                              (p[i] >= 'a' && p[i] <= 'f'))) return -1;
        out[i] = p[i];
    }
    p += 64;
    if (p >= end || *p++ != '"') return -1;
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p > end || !token_end(*p)) return -1;
    out[64] = 0;
    return 0;
}

static int manifest_file_binding(const char *manifest, const char *name,
                                 long *size, char sha256[65]) {
    const char *files_start, *files_end, *entry_start, *entry_end;
    if (object_member_count(manifest, "files") != 2 ||
        object_bounds(manifest, "files", &files_start, &files_end) ||
        object_bounds(files_start, name, &entry_start, &entry_end) ||
        entry_end > files_end || direct_object_member_count(entry_start, entry_end) != 2 ||
        bounded_long(entry_start, entry_end, "size", size) || *size < 0 ||
        bounded_sha256(entry_start, entry_end, "sha256", sha256)) {
        return -1;
    }
    return 0;
}
'''
    if text.count(helper_anchor) != 1:
        raise RuntimeError("runner parser helper anchor mismatch")
    text = text.replace(helper_anchor, helper_new, 1)

    start = text.index('    {\n        char expected_hash[65];\n        long expected_size;\n', text.index('cannot capture windows.jsonl'))
    end = text.index('\n    free_captured(&captured_manifest);', start)
    manifest_block = r'''    {
        char expected_session_hash[65], expected_windows_hash[65];
        long expected_session_size = -1, expected_windows_size = -1;
        if (manifest_file_binding(manifest, "session.json", &expected_session_size,
                                  expected_session_hash) ||
            (size_t)expected_session_size != schedule.captured_session_json.size) {
            die("manifest session.json size binding mismatch");
        }
        if (strcmp(expected_session_hash, schedule.captured_session_json.sha256)) {
            die("manifest session.json sha256 binding mismatch");
        }
        if (manifest_file_binding(manifest, "windows.jsonl", &expected_windows_size,
                                  expected_windows_hash) ||
            (size_t)expected_windows_size != schedule.captured_windows_jsonl.size) {
            die("manifest windows.jsonl size binding mismatch");
        }
        if (strcmp(expected_windows_hash, schedule.captured_windows_jsonl.sha256)) {
            die("manifest windows.jsonl sha256 binding mismatch");
        }
    }
'''
    text = text[:start] + manifest_block + text[end:]

    old_loop = r'''        FILE *out = fopen(destination, "wbx");
        if (!out || fwrite(inputs[i]->bytes, 1, inputs[i]->size, out) != inputs[i]->size ||
            fclose(out)) {
            die("write captured bytes failed");
        }
'''
    new_loop = r'''        if (write_captured_exclusive(destination, inputs[i])) {
            die("write captured bytes failed");
        }
'''
    if text.count(old_loop) != 1:
        raise RuntimeError("validation captured writer anchor mismatch")
    text = text.replace(old_loop, new_loop, 1)
    RUNNER.write_text(text, encoding="utf-8")


def patch_hardware() -> None:
    text = HARDWARE.read_text(encoding="utf-8")
    start = text.index('static int copy_file(')
    end = text.index('\nstatic const char *inject', start)
    text = text[:start] + text[end + 1:]

    old_session = r'''    {
        FILE *fout = fopen(destination, "wbx");
        if (!fout || fwrite(schedule->captured_session_json.bytes, 1,
                           schedule->captured_session_json.size, fout) !=
                         schedule->captured_session_json.size ||
            fclose(fout)) {
            reason = "INPUT_COPY_FAILURE";
            rc = 5;
            goto done;
        }
    }
'''
    new_session = r'''    if (write_captured_exclusive(destination, &schedule->captured_session_json)) {
        reason = "INPUT_COPY_FAILURE";
        rc = 5;
        goto done;
    }
'''
    if text.count(old_session) != 1:
        raise RuntimeError("hardware session writer anchor mismatch")
    text = text.replace(old_session, new_session, 1)

    old_windows = r'''    {
        FILE *fout = fopen(destination, "wbx");
        if (!fout || fwrite(schedule->captured_windows_jsonl.bytes, 1,
                           schedule->captured_windows_jsonl.size, fout) !=
                         schedule->captured_windows_jsonl.size ||
            fclose(fout)) {
            reason = "INPUT_COPY_FAILURE";
            rc = 5;
            goto done;
        }
    }
'''
    new_windows = r'''    if (write_captured_exclusive(destination, &schedule->captured_windows_jsonl)) {
        reason = "INPUT_COPY_FAILURE";
        rc = 5;
        goto done;
    }
'''
    if text.count(old_windows) != 1:
        raise RuntimeError("hardware windows writer anchor mismatch")
    text = text.replace(old_windows, new_windows, 1)
    HARDWARE.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = TEST_RUNNER.read_text(encoding="utf-8")
    old_manifest = r'''    def test_manifest_size(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            (directory / "windows.jsonl").write_text(
                (directory / "windows.jsonl").read_text() + " ")
            self.assertIn("size mismatch", self.exec_runner(
                directory, Path(temp) / "out", "--validate-only").stderr)

    def test_manifest_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = write_session(Path(temp))
            manifest = json.loads((directory / "session_manifest.json").read_text())
            manifest["files"]["windows.jsonl"]["sha256"] = "0" * 64
            dump(directory / "session_manifest.json", manifest)
            self.assertIn("sha256 mismatch", self.exec_runner(
                directory, Path(temp) / "out", "--validate-only").stderr)
'''
    new_manifest = r'''    def test_manifest_size(self):
        for name in ("session.json", "windows.jsonl"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                directory = write_session(Path(temp))
                path = directory / name
                path.write_bytes(path.read_bytes() + b" ")
                result = self.exec_runner(directory, Path(temp) / "out", "--validate-only")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("size binding mismatch", result.stderr)

    def test_manifest_sha(self):
        for name in ("session.json", "windows.jsonl"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                directory = write_session(Path(temp))
                manifest = json.loads((directory / "session_manifest.json").read_text())
                manifest["files"][name]["sha256"] = "0" * 64
                dump(directory / "session_manifest.json", manifest)
                result = self.exec_runner(directory, Path(temp) / "out", "--validate-only")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("sha256 binding mismatch", result.stderr)
'''
    if text.count(old_manifest) != 1:
        raise RuntimeError("manifest tests anchor mismatch")
    text = text.replace(old_manifest, new_manifest, 1)

    old_hash_assert = '''            self.assertTrue(all(sha(output / name) == binding["sha256"]
                                for name, binding in manifest["files"].items()))
'''
    new_hash_assert = old_hash_assert + '''            self.assertEqual((output / "session.json").read_bytes(),
                             (directory / "session.json").read_bytes())
            self.assertEqual((output / "windows.jsonl").read_bytes(),
                             (directory / "windows.jsonl").read_bytes())
'''
    if text.count(old_hash_assert) != 1:
        raise RuntimeError("validation equality anchor mismatch")
    text = text.replace(old_hash_assert, new_hash_assert, 1)

    insertion = r'''
    def test_blank_authorized_by_is_rejected(self):
        for value in ("", " ", "\t", "\n\t "):
            with self.subTest(value=repr(value)), tempfile.TemporaryDirectory() as temp:
                result = self.authorization_result(
                    Path(temp), auth_mutate=lambda authorization, v=value:
                        authorization.update(authorized_by=v)
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid V2 calibration authorization", result.stderr)

    @unittest.skipIf(os.name == "nt", "symlink custody is a Linux requirement")
    def test_symlinked_immutable_inputs_are_rejected(self):
        for name in ("session_manifest.json", "session.json", "windows.jsonl"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                session = write_session(root)
                path = session / name
                real = session / f"{name}.real"
                path.replace(real)
                path.symlink_to(real.name)
                result = self.exec_runner(session, root / "out", "--validate-only")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("cannot capture", result.stderr)

        for kind in ("authorization", "source_bundle"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                session = write_session(root)
                bundle = write_source_bundle(session)
                authorized_root = root / "authorized"
                authorized_root.mkdir()
                authorization = write_authorization(root, session, bundle, authorized_root)
                path = authorization if kind == "authorization" else bundle
                real = path.with_name(path.name + ".real")
                path.replace(real)
                path.symlink_to(real.name)
                result = self.exec_runner(
                    session, authorized_root / "run", "--hardware",
                    "--authorization-artifact", str(authorization),
                    source_bundle=bundle, fail="thermal",
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("cannot capture", result.stderr)

    def test_mock_output_uses_captured_schedule_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = write_session(root)
            output = root / "out"
            result = self.exec_runner(session, output, "--mock-hardware")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((output / "session.json").read_bytes(),
                             (session / "session.json").read_bytes())
            self.assertEqual((output / "windows.jsonl").read_bytes(),
                             (session / "windows.jsonl").read_bytes())

    def test_no_bound_input_path_reopen_helpers_remain(self):
        runner_source = (HERE / "combined_pdn_runner.c").read_text(encoding="utf-8")
        hardware_source = (HERE / "combined_pdn_hardware.c").read_text(encoding="utf-8")
        self.assertNotIn("hash_file(args->authorization_artifact", hardware_source)
        self.assertNotIn("copy_file(", hardware_source)
        self.assertIn("write_captured_exclusive", runner_source)
        self.assertIn("write_captured_exclusive", hardware_source)

'''
    marker = '\n\nif __name__ == "__main__":\n'
    if text.count(marker) != 1:
        raise RuntimeError("runner test insertion anchor mismatch")
    text = text.replace(marker, '\n' + insertion + marker, 1)
    TEST_RUNNER.write_text(text, encoding="utf-8")


def patch_build_surfaces() -> None:
    replace_once(
        MAKEFILE,
        '\tpython3 test_combined_pdn_runner.py\n',
        '\tpython3 -m unittest -v test_combined_pdn_runner.py test_waveform_equivalence.py test_slot2_primitive_identity.py test_captured_file.py\n',
        "Makefile test target",
    )
    replace_once(
        STRICT_WF,
        'python -m unittest -v test_combined_pdn_runner.py test_waveform_equivalence.py test_slot2_primitive_identity.py',
        'python -m unittest -v test_combined_pdn_runner.py test_waveform_equivalence.py test_slot2_primitive_identity.py test_captured_file.py',
        "strict workflow runtime tests",
    )
    replace_once(
        STRICT_WF,
        'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 python -m unittest -v test_combined_pdn_runner.py\n',
        'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 python -m unittest -v test_combined_pdn_runner.py test_waveform_equivalence.py\n',
        "strict workflow ASan tests",
    )
    replace_once(
        STRICT_WF,
        'UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 python -m unittest -v test_combined_pdn_runner.py\n',
        'UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 python -m unittest -v test_combined_pdn_runner.py test_waveform_equivalence.py\n',
        "strict workflow UBSan tests",
    )
    replace_once(
        COMBINED_WF,
        'python -m unittest -v test_combined_pdn_runner.py > "$RUNNER_TEMP/v2_runner_tests.log" 2>&1',
        'python -m unittest -v test_combined_pdn_runner.py test_captured_file.py > "$RUNNER_TEMP/v2_runner_tests.log" 2>&1',
        "combined workflow runner tests",
    )


def main() -> int:
    CAPTURE_H.write_text(CAPTURE_H_TEXT, encoding="utf-8")
    CAPTURE_C.write_text(CAPTURE_C_TEXT, encoding="utf-8")
    FIXTURE.write_text(FIXTURE_TEXT, encoding="utf-8")
    TEST_CAPTURE.write_text(TEST_CAPTURE_TEXT, encoding="utf-8")
    patch_runner()
    patch_hardware()
    patch_tests()
    patch_build_surfaces()

    allowed = {
        CAPTURE_H.relative_to(ROOT).as_posix(),
        CAPTURE_C.relative_to(ROOT).as_posix(),
        FIXTURE.relative_to(ROOT).as_posix(),
        TEST_CAPTURE.relative_to(ROOT).as_posix(),
        RUNNER.relative_to(ROOT).as_posix(),
        HARDWARE.relative_to(ROOT).as_posix(),
        TEST_RUNNER.relative_to(ROOT).as_posix(),
        MAKEFILE.relative_to(ROOT).as_posix(),
        STRICT_WF.relative_to(ROOT).as_posix(),
        COMBINED_WF.relative_to(ROOT).as_posix(),
    }
    changed = set(subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines())
    untracked = set(subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines())
    actual = changed | untracked
    if actual != allowed:
        raise RuntimeError(f"unexpected changed paths: {sorted(actual ^ allowed)}")
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    print("CUSTODY_HARDENING_RENDERED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
