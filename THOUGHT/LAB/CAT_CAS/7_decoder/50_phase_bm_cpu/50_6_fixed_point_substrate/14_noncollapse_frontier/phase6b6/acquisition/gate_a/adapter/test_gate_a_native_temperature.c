#define _POSIX_C_SOURCE 200809L

#include "captured_file.h"
#include "gate_a_engineering_smoke_runtime.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "test_gate_a_native_temperature: %s\n", message); \
            goto cleanup; \
        } \
    } while (0)

static int path_join(char *output, size_t capacity,
                     const char *left, const char *right) {
    int count = snprintf(output, capacity, "%s/%s", left, right);
    return count < 0 || (size_t)count >= capacity ? -1 : 0;
}

static int write_text(const char *path, const char *text) {
    FILE *file = fopen(path, "wx");
    if (!file) return -1;
    if (fputs(text, file) < 0 || fflush(file) || fsync(fileno(file)) ||
        fclose(file)) {
        return -1;
    }
    return 0;
}

static int add_hwmon(const char *root, int index,
                     const char *driver, const char *temperature) {
    char entry[CP_PATH_MAX], path[CP_PATH_MAX], name[32];
    if (snprintf(name, sizeof(name), "hwmon%d", index) < 0 ||
        path_join(entry, sizeof(entry), root, name) ||
        mkdir(entry, 0700) ||
        path_join(path, sizeof(path), entry, "name") ||
        write_text(path, driver)) {
        return -1;
    }
    if (temperature) {
        if (path_join(path, sizeof(path), entry,
                      GATE_A_TEMPERATURE_INPUT) ||
            write_text(path, temperature)) {
            return -1;
        }
    }
    return 0;
}

static int make_fixture_root(char *output, size_t capacity,
                             const char *base, const char *name) {
    return path_join(output, capacity, base, name) ||
           mkdir(output, 0700) ? -1 : 0;
}

static int file_exists(const char *path) {
    struct stat value;
    return !stat(path, &value) && S_ISREG(value.st_mode);
}

static int file_size_is(const char *path, off_t expected) {
    struct stat value;
    return !stat(path, &value) && S_ISREG(value.st_mode) &&
           value.st_size == expected;
}

static int receipt_contains(const char *path, const char *needle) {
    CapturedFile captured = {0};
    int found = 0;
    if (!capture_file(path, &captured, 1u << 16)) {
        char *text = malloc(captured.size + 1);
        if (text) {
            memcpy(text, captured.bytes, captured.size);
            text[captured.size] = '\0';
            found = strstr(text, needle) != NULL;
            free(text);
        }
    }
    free_captured(&captured);
    return found;
}

static int files_equal(const char *left, const char *right) {
    CapturedFile a = {0}, b = {0};
    int equal = 0;
    if (!capture_file(left, &a, 1u << 16) &&
        !capture_file(right, &b, 1u << 16)) {
        equal = a.size == b.size &&
                !memcmp(a.bytes, b.bytes, a.size);
    }
    free_captured(&a);
    free_captured(&b);
    return equal;
}

static GateASmokeArgs mock_args(const char *output, long pre, long post) {
    GateASmokeArgs args = {
        .output_dir = output,
        .authority_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        .execution_bundle_sha256 =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        .sender_core = 4,
        .receiver_core = 5,
        .read_hz = 8000,
        .slot_s = 0.5,
        .temperature_veto_c = 68.0,
        .required_frequency_khz = 1600000,
        .backend = BACKEND_MOCK,
        .mock_pre_temperature_millidegrees = pre,
        .mock_post_temperature_millidegrees = post,
    };
    return args;
}

static int remove_tree(const char *path) {
    struct stat value;
    if (lstat(path, &value)) return errno == ENOENT ? 0 : -1;
    if (!S_ISDIR(value.st_mode) || S_ISLNK(value.st_mode)) {
        return unlink(path);
    }
    DIR *directory = opendir(path);
    if (!directory) return -1;
    struct dirent *entry;
    int rc = 0;
    while ((entry = readdir(directory)) != NULL) {
        char child[CP_PATH_MAX];
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) {
            continue;
        }
        if (path_join(child, sizeof(child), path, entry->d_name) ||
            remove_tree(child)) {
            rc = -1;
            break;
        }
    }
    if (closedir(directory)) rc = -1;
    if (!rc && rmdir(path)) rc = -1;
    return rc;
}

int main(void) {
    char base[] = "/tmp/gate_a_native_temperature_XXXXXX";
    char root[CP_PATH_MAX], receipt[CP_PATH_MAX];
    char output_pre[CP_PATH_MAX], output_a[CP_PATH_MAX];
    char output_b[CP_PATH_MAX], output_post[CP_PATH_MAX];
    char path_a[CP_PATH_MAX], path_b[CP_PATH_MAX];
    GateASmokeArgs args;
    GateASmokeResult result;
    int rc = 1;

    CHECK(mkdtemp(base) != NULL, "temporary root creation failed");

    CHECK(!make_fixture_root(root, sizeof(root), base, "exact"),
          "exact fixture root failed");
    CHECK(!add_hwmon(root, 0, "nouveau\n", "99000\n") &&
          !add_hwmon(root, 1, "k10temp\n", "67999\n"),
          "exact fixture creation failed");
    CHECK(!path_join(receipt, sizeof(receipt), base, "exact.jsonl"),
          "exact receipt path failed");
    CHECK(gate_a_test_observe_temperature(
              root, "pre_capture", receipt, 100) == 0,
          "exact k10temp selection failed");
    CHECK(receipt_contains(receipt, "\"selected_hwmon_entry\":\"") &&
          receipt_contains(receipt, "/exact/hwmon1\"") &&
          receipt_contains(receipt, "\"raw_millidegrees_c\":67999") &&
          receipt_contains(receipt, "\"k10temp_candidate_count\":1"),
          "exact k10temp receipt mismatch");

    CHECK(!make_fixture_root(root, sizeof(root), base, "gpu_only") &&
          !add_hwmon(root, 0, "nouveau\n", "42000\n") &&
          !path_join(receipt, sizeof(receipt), base, "gpu_only.jsonl"),
          "GPU-only fixture failed");
    CHECK(gate_a_test_observe_temperature(
              root, "pre_capture", receipt, 101) != 0 &&
          receipt_contains(receipt, "\"failure\":\"K10TEMP_CANDIDATE_COUNT\""),
          "GPU-only state did not fail closed");

    CHECK(!make_fixture_root(root, sizeof(root), base, "ambiguous") &&
          !add_hwmon(root, 0, "k10temp\n", "42000\n") &&
          !add_hwmon(root, 1, "k10temp\n", "43000\n") &&
          !path_join(receipt, sizeof(receipt), base, "ambiguous.jsonl"),
          "ambiguous fixture failed");
    CHECK(gate_a_test_observe_temperature(
              root, "pre_capture", receipt, 102) != 0 &&
          receipt_contains(receipt, "\"k10temp_candidate_count\":2"),
          "multiple k10temp devices did not fail closed");

    CHECK(!make_fixture_root(root, sizeof(root), base, "missing") &&
          !add_hwmon(root, 0, "k10temp\n", NULL) &&
          !path_join(receipt, sizeof(receipt), base, "missing.jsonl"),
          "missing-input fixture failed");
    CHECK(gate_a_test_observe_temperature(
              root, "pre_capture", receipt, 103) != 0 &&
          receipt_contains(
              receipt, "\"failure\":\"TEMPERATURE_INPUT_UNOBSERVABLE\""),
          "missing input did not retain a closed failure");

    CHECK(!make_fixture_root(root, sizeof(root), base, "malformed") &&
          !add_hwmon(root, 0, "k10temp\n", "68000junk\n") &&
          !path_join(receipt, sizeof(receipt), base, "malformed.jsonl"),
          "malformed fixture failed");
    CHECK(gate_a_test_observe_temperature(
              root, "pre_capture", receipt, 104) != 0 &&
          receipt_contains(
              receipt, "\"failure\":\"TEMPERATURE_INTEGER_MALFORMED\""),
          "malformed input did not fail closed");

    CHECK(!make_fixture_root(root, sizeof(root), base, "boundary_pass") &&
          !add_hwmon(root, 0, "k10temp\n", "67999\n") &&
          !path_join(receipt, sizeof(receipt), base, "boundary_pass.jsonl"),
          "pass-boundary fixture failed");
    CHECK(gate_a_test_observe_temperature(
              root, "pre_capture", receipt, 105) == 0 &&
          receipt_contains(receipt, "\"veto_passed\":true"),
          "67.999 C did not pass");

    CHECK(!make_fixture_root(root, sizeof(root), base, "boundary_veto") &&
          !add_hwmon(root, 0, "k10temp\n", "68000\n") &&
          !path_join(receipt, sizeof(receipt), base, "boundary_veto.jsonl"),
          "veto-boundary fixture failed");
    CHECK(gate_a_test_observe_temperature(
              root, "pre_capture", receipt, 106) != 0 &&
          receipt_contains(receipt, "\"failure\":\"TEMPERATURE_VETO\""),
          "68.000 C did not veto");

    CHECK(!path_join(output_pre, sizeof(output_pre), base, "runtime_pre"),
          "pre-veto output path failed");
    args = mock_args(output_pre, 68000, 42000);
    memset(&result, 0, sizeof(result));
    CHECK(run_gate_a_engineering_smoke(&args, &result) == 3,
          "pre-capture veto did not fail closed");
    CHECK(result.temperature_receipt_count == 1 &&
          result.sender_start_count == 0 &&
          result.receiver_start_count == 0 &&
          result.hardware_executed == 0,
          "pre-capture veto crossed the physical-start boundary");
    CHECK(!path_join(path_a, sizeof(path_a), output_pre,
                     GATE_A_TEMPERATURE_RECEIPT_FILE) &&
          file_exists(path_a) &&
          receipt_contains(path_a, "\"phase\":\"pre_capture\""),
          "pre-capture failure receipt was not retained");

    CHECK(!path_join(output_a, sizeof(output_a), base, "runtime_success_a") &&
          !path_join(output_b, sizeof(output_b), base, "runtime_success_b"),
          "success output path failed");
    args = mock_args(output_a, 42000, 42000);
    memset(&result, 0, sizeof(result));
    CHECK(run_gate_a_engineering_smoke(&args, &result) == 0 &&
          result.temperature_receipt_count == 2 &&
          result.sender_start_count == 0 &&
          result.receiver_start_count == 0 &&
          result.hardware_executed == 0,
          "mock success temperature custody failed");
    args = mock_args(output_b, 42000, 42000);
    memset(&result, 0, sizeof(result));
    CHECK(run_gate_a_engineering_smoke(&args, &result) == 0,
          "second deterministic mock failed");
    CHECK(!path_join(path_a, sizeof(path_a), output_a,
                     GATE_A_TEMPERATURE_RECEIPT_FILE) &&
          !path_join(path_b, sizeof(path_b), output_b,
                     GATE_A_TEMPERATURE_RECEIPT_FILE) &&
          files_equal(path_a, path_b) &&
          receipt_contains(path_a, "\"phase\":\"pre_capture\"") &&
          receipt_contains(path_a, "\"phase\":\"post_capture\""),
          "mock receipts are incomplete or nondeterministic");

    CHECK(!path_join(output_post, sizeof(output_post), base, "runtime_post"),
          "post-veto output path failed");
    args = mock_args(output_post, 67999, 68000);
    memset(&result, 0, sizeof(result));
    CHECK(run_gate_a_engineering_smoke(&args, &result) == 3 &&
          result.temperature_receipt_count == 2,
          "post-capture veto did not fail closed");
    CHECK(!path_join(path_a, sizeof(path_a), output_post,
                     "raw_samples.bin") &&
          file_size_is(path_a, (off_t)(64000 * 16)),
          "post-capture veto did not preserve complete raw evidence");
    CHECK(!path_join(path_a, sizeof(path_a), output_post,
                     "LOCKIN_IQ.jsonl") && file_exists(path_a) &&
          !path_join(path_a, sizeof(path_a), output_post,
                     "SENDER_LIFECYCLE.jsonl") && file_exists(path_a) &&
          !path_join(path_a, sizeof(path_a), output_post,
                     "slot_trace.jsonl") && file_exists(path_a) &&
          !path_join(path_a, sizeof(path_a), output_post,
                     GATE_A_TEMPERATURE_RECEIPT_FILE) &&
          receipt_contains(path_a, "\"phase\":\"post_capture\"") &&
          receipt_contains(path_a, "\"failure\":\"TEMPERATURE_VETO\""),
          "post-capture failure evidence is incomplete");

    rc = 0;

cleanup:
    if (base[0] && remove_tree(base) && rc == 0) rc = 1;
    if (rc == 0) {
        puts("{\"status\":\"GATE_A_NATIVE_TEMPERATURE_TEST_OK\","
             "\"network_connections\":0,\"hardware_executions\":0,"
             "\"sender_starts\":0,\"receiver_starts\":0,"
             "\"mutation_baseline\":\"native_temperature_closed_receipts\"}");
    }
    return rc;
}
