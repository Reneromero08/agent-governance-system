#define _GNU_SOURCE
#include "combined_pdn_hardware.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define SHA_LEN 64

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("ERROR: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(2);
}

static int path_join(char *out, size_t n, const char *a, const char *b) {
    if (!a || !b || b[0] == '/' || strstr(b, "..")) return -1;
    int k = snprintf(out, n, "%s/%s", a, b);
    return k < 0 || (size_t)k >= n ? -1 : 0;
}

static long file_size(const char *path) {
    struct stat st;
    return stat(path, &st) || !S_ISREG(st.st_mode) ? -1 : (long)st.st_size;
}

static int exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("open %s: %s", path, strerror(errno));
    if (fseek(f, 0, SEEK_END)) die("seek %s", path);
    long size = ftell(f);
    if (size < 0 || fseek(f, 0, SEEK_SET)) die("seek %s", path);
    char *buf = calloc((size_t)size + 1, 1);
    if (!buf) die("oom");
    if (size && fread(buf, 1, (size_t)size, f) != (size_t)size) die("read %s", path);
    fclose(f);
    return buf;
}

static const char *key(const char *json, const char *name) {
    char needle[160];
    snprintf(needle, sizeof(needle), "\"%s\"", name);
    return strstr(json, needle);
}

static const char *value(const char *json, const char *name) {
    const char *p = key(json, name);
    if (!p || !(p = strchr(p, ':'))) return NULL;
    do p++; while (isspace((unsigned char)*p));
    return p;
}

static int jstr(const char *json, const char *name, char *out, size_t n) {
    const char *p = value(json, name);
    if (!p || *p != '"') return -1;
    const char *q = strchr(++p, '"');
    size_t size = q ? (size_t)(q - p) : n;
    if (!q || !size || size >= n) return -1;
    memcpy(out, p, size);
    out[size] = 0;
    return 0;
}

static int jlong(const char *json, const char *name, long *out) {
    const char *p = value(json, name);
    if (!p) return -1;
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p) return -1;
    *out = v;
    return 0;
}

static int jbool(const char *json, const char *name, int *out) {
    const char *p = value(json, name);
    if (!p) return -1;
    if (!strncmp(p, "true", 4)) {
        *out = 1;
        return 0;
    }
    if (!strncmp(p, "false", 5)) {
        *out = 0;
        return 0;
    }
    return -1;
}

static int jnullable_long(const char *json, const char *name, long *out, int *present) {
    const char *p = value(json, name);
    if (!p) return -1;
    if (!strncmp(p, "null", 4)) {
        *present = 0;
        *out = 0;
        return 0;
    }
    *present = 1;
    return jlong(json, name, out);
}

static int shell_safe(const char *path) {
    if (!path || !*path || strstr(path, "..")) return 0;
    for (; *path; path++) {
        if ((unsigned char)*path < 32 || strchr("';&|`$<>", *path)) return 0;
    }
    return 1;
}

static int valid_commit(const char *commit) {
    if (!commit || strlen(commit) != 40) return 0;
    int nonzero = 0;
    for (size_t i = 0; i < 40; i++) {
        char c = commit[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return 0;
        if (c != '0') nonzero = 1;
    }
    return nonzero;
}

static void sha256(const char *path, char out[65]) {
    if (!shell_safe(path)) die("unsafe path: %s", path);
    char command[CP_PATH_MAX + 32];
    if (snprintf(command, sizeof(command), "sha256sum '%s'", path) >= (int)sizeof(command)) {
        die("path too long");
    }
    FILE *f = popen(command, "r");
    char line[128];
    if (!f || !fgets(line, sizeof(line), f)) die("sha256 failed");
    int rc = pclose(f);
    if (rc == -1 || !WIFEXITED(rc) || WEXITSTATUS(rc)) die("sha256 failed");
    for (int i = 0; i < SHA_LEN; i++) {
        if (!isxdigit((unsigned char)line[i])) die("invalid sha256 output");
        out[i] = (char)tolower((unsigned char)line[i]);
    }
    out[64] = 0;
}

static void verify_file(const char *manifest, const char *root, const char *name) {
    const char *entry = key(manifest, name);
    long expected_size;
    char expected_hash[65], path[CP_PATH_MAX], actual_hash[65];
    if (!entry) die("manifest missing %s", name);
    if (jlong(entry, "size", &expected_size) ||
        jstr(entry, "sha256", expected_hash, sizeof(expected_hash)) ||
        strlen(expected_hash) != 64) {
        die("invalid manifest entry %s", name);
    }
    if (path_join(path, sizeof(path), root, name) || file_size(path) != expected_size) {
        die("size mismatch for %s", name);
    }
    sha256(path, actual_hash);
    if (strcmp(expected_hash, actual_hash)) die("sha256 mismatch for %s", name);
}

static RunnerArgs parse_args(int argc, char **argv) {
    RunnerArgs args = {0};
    args.victim = -1;
    args.sender = -1;
    args.pin_khz = 1600000;
    args.slot_s = 0.5;
    args.off_window_s = 0.5;
    args.read_hz = 4000;
    args.temp_veto_c = 68;
    args.backend = BACKEND_REAL;

    for (int i = 1; i < argc; i++) {
        const char *key_name = argv[i];
        const char *arg = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(key_name, "--validate-only")) args.mode = MODE_VALIDATE;
        else if (!strcmp(key_name, "--hardware")) args.mode = MODE_HARDWARE;
        else if (!strcmp(key_name, "--mock-hardware")) {
            args.mode = MODE_HARDWARE;
            args.backend = BACKEND_MOCK;
        } else if (!strcmp(key_name, "--executor-commit") && arg) {
            args.executor_commit = arg;
            i++;
        } else if (!strcmp(key_name, "--session-dir") && arg) {
            args.session_dir = arg;
            i++;
        } else if (!strcmp(key_name, "--output-dir") && arg) {
            args.output_dir = arg;
            i++;
        } else if (!strcmp(key_name, "--victim") && arg) {
            args.victim = atoi(arg);
            i++;
        } else if (!strcmp(key_name, "--sender") && arg) {
            args.sender = atoi(arg);
            i++;
        } else if (!strcmp(key_name, "--pin-khz") && arg) {
            args.pin_khz = atol(arg);
            i++;
        } else if (!strcmp(key_name, "--slot-s") && arg) {
            args.slot_s = atof(arg);
            i++;
        } else if (!strcmp(key_name, "--off-window-s") && arg) {
            args.off_window_s = atof(arg);
            i++;
        } else if (!strcmp(key_name, "--read-hz") && arg) {
            args.read_hz = atol(arg);
            i++;
        } else if (!strcmp(key_name, "--temp-veto-c") && arg) {
            args.temp_veto_c = atof(arg);
            i++;
        } else {
            die("unknown or incomplete option: %s", key_name);
        }
    }

    if (!args.mode || !args.session_dir || !args.output_dir ||
        args.victim < 0 || args.sender < 0 || args.victim == args.sender) {
        die("invalid required arguments");
    }
    if (args.mode == MODE_HARDWARE && !valid_commit(args.executor_commit)) {
        die("hardware mode requires a nonzero lowercase 40-character hex --executor-commit");
    }
    if (!shell_safe(args.session_dir) || !shell_safe(args.output_dir)) die("unsafe path");
    return args;
}

static void copy_exclusive(const char *source, const char *destination) {
    FILE *in = fopen(source, "rb");
    FILE *out = fopen(destination, "wbx");
    if (!in || !out) die("copy failed");
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), in))) {
        if (fwrite(buf, 1, n, out) != n) die("copy failed");
    }
    if (fclose(in) || fclose(out)) die("copy failed");
}

static Schedule load_schedule(const RunnerArgs *args) {
    char path[CP_PATH_MAX], schema[128], manifest_session[128];
    Schedule schedule = {0};

    path_join(path, sizeof(path), args->session_dir, "session_manifest.json");
    char *manifest = slurp(path);
    sha256(path, schedule.session_manifest_sha256);
    if (jstr(manifest, "schema_id", schema, sizeof(schema)) ||
        strcmp(schema, "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1")) {
        die("unexpected session manifest schema");
    }
    if (jstr(manifest, "session_id", manifest_session, sizeof(manifest_session))) {
        die("manifest missing session_id");
    }
    verify_file(manifest, args->session_dir, "session.json");
    verify_file(manifest, args->session_dir, "windows.jsonl");
    free(manifest);

    path_join(path, sizeof(path), args->session_dir, "session.json");
    char *header = slurp(path);
    if (jstr(header, "schema_id", schema, sizeof(schema)) ||
        strcmp(schema, "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V1")) {
        die("unexpected session schema");
    }
    if (jstr(header, "session_id", schedule.session_id, sizeof(schedule.session_id)) ||
        strcmp(schedule.session_id, manifest_session)) {
        die("session ID mismatch");
    }
    if (jstr(header, "route", schedule.route, sizeof(schedule.route)) ||
        jstr(header, "campaign_source_commit", schedule.campaign_source_commit,
             sizeof(schedule.campaign_source_commit)) ||
        jstr(header, "campaign_plan_sha256", schedule.campaign_plan_sha256,
             sizeof(schedule.campaign_plan_sha256))) {
        die("missing session binding");
    }
    long count;
    int restoration = 1;
    if (jlong(header, "window_count", &count) || count <= 0 ||
        jbool(header, "restoration_authorized", &restoration) || restoration) {
        die("invalid session header");
    }
    free(header);

    schedule.count = (size_t)count;
    schedule.windows = calloc(schedule.count, sizeof(*schedule.windows));
    if (!schedule.windows) die("oom");

    path_join(path, sizeof(path), args->session_dir, "windows.jsonl");
    FILE *f = fopen(path, "r");
    if (!f) die("open windows");
    char *line = NULL;
    size_t capacity = 0, n = 0;
    while (getline(&line, &capacity, f) != -1) {
        if (n >= schedule.count) die("extra schedule rows");
        Window *window = &schedule.windows[n];
        long x;
        int present;
        if (jlong(line, "window_index", &x) || x != (long)n) {
            die("window indices not contiguous or duplicate");
        }
        window->window_index = x;
        if (jstr(line, "session_id", window->session_id, sizeof(window->session_id)) ||
            strcmp(window->session_id, schedule.session_id)) {
            die("window session ID mismatch");
        }
        if (jstr(line, "stage", window->stage, sizeof(window->stage)) ||
            jstr(line, "block_id", window->block_id, sizeof(window->block_id)) ||
            jstr(line, "family", window->family, sizeof(window->family)) ||
            jstr(line, "measurement_mode", window->measurement_mode,
                 sizeof(window->measurement_mode)) ||
            jstr(line, "executed_tone_order", window->executed_tone_order,
                 sizeof(window->executed_tone_order)) ||
            jstr(line, "declared_tone_order", window->declared_tone_order,
                 sizeof(window->declared_tone_order))) {
            die("short schedule row");
        }
        if (jstr(line, "actual_mode", window->actual_mode, sizeof(window->actual_mode))) {
            strcpy(window->actual_mode, "null");
        }
        if (jstr(line, "declared_mode", window->declared_mode, sizeof(window->declared_mode))) {
            strcpy(window->declared_mode, "null");
        }
        if (jbool(line, "drive_on", &window->drive_on) ||
            jbool(line, "sender_off_required", &window->sender_off_required)) {
            die("invalid window booleans");
        }
        if (jnullable_long(line, "physical_tone_index", &x, &present)) {
            die("missing physical tone");
        }
        window->physical_tone_index = present ? (int)x : -1;
        if (jnullable_long(line, "codeword_source_index", &x, &present)) {
            die("missing codeword source");
        }
        window->codeword_source_index = present ? (int)x : -1;
        if (jnullable_long(line, "theta_idx", &x, &present)) die("missing theta_idx");
        window->theta_idx = present ? (int)x : -1;
        if (jlong(line, "amplitude_level", &x)) {
            window->amplitude_level = window->drive_on ? 3 : 0;
        } else {
            window->amplitude_level = (int)x;
        }

        if (window->sender_off_required && window->drive_on) {
            die("sender_off_required + drive_on rejection");
        }
        if (!strcmp(window->measurement_mode, "raw_ring_sender_off")) {
            if (!window->sender_off_required || window->drive_on) {
                die("raw_ring_sender_off requires sender off");
            }
        } else if (!strcmp(window->measurement_mode, "lockin_and_raw_ring")) {
            if (window->drive_on &&
                (window->physical_tone_index < 0 || window->codeword_source_index < 0)) {
                die("driven lock-in missing physical tone or codeword source");
            }
        } else {
            die("unsupported measurement mode");
        }
        n++;
    }
    free(line);
    fclose(f);
    if (n != schedule.count) die("short schedule row count");

    if ((!strcmp(schedule.route, "v4s5") && (args->victim != 4 || args->sender != 5)) ||
        (!strcmp(schedule.route, "v2s3") && (args->victim != 2 || args->sender != 3))) {
        die("invalid route/core pair");
    }
    return schedule;
}

static void validation_outputs(const RunnerArgs *args, const Schedule *schedule) {
    if (exists(args->output_dir) || mkdir(args->output_dir, 0755)) {
        die("refusing existing output directory");
    }
    char source[CP_PATH_MAX], destination[CP_PATH_MAX];
    const char *inputs[] = {"session.json", "windows.jsonl"};
    for (int i = 0; i < 2; i++) {
        path_join(source, sizeof(source), args->session_dir, inputs[i]);
        path_join(destination, sizeof(destination), args->output_dir, inputs[i]);
        copy_exclusive(source, destination);
    }
    const char *empty[] = {"raw_samples.bin", "telemetry.csv", "stderr.log"};
    for (int i = 0; i < 3; i++) {
        path_join(destination, sizeof(destination), args->output_dir, empty[i]);
        FILE *f = fopen(destination, "wbx");
        if (!f) die("create output");
        fclose(f);
    }

    path_join(destination, sizeof(destination), args->output_dir, "stdout.log");
    FILE *f = fopen(destination, "wx");
    if (!f) die("create stdout.log");
    fprintf(f, "VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\n");
    fclose(f);

    path_join(destination, sizeof(destination), args->output_dir, "window_results.csv");
    f = fopen(destination, "wx");
    if (!f) die("create window_results.csv");
    fprintf(f, "window_index,session_id,validation_status,hardware_executed\n");
    for (size_t i = 0; i < schedule->count; i++) {
        fprintf(f, "%zu,%s,VALIDATED,0\n", i, schedule->session_id);
    }
    fclose(f);

    path_join(destination, sizeof(destination), args->output_dir, "run.json");
    f = fopen(destination, "wx");
    if (!f) die("create run.json");
    fprintf(f,
            "{\n"
            "  \"schema_id\": \"CAT_CAS_PHASE6_COMBINED_RUN_V1\",\n"
            "  \"session_id\": \"%s\",\n"
            "  \"status\": \"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\",\n"
            "  \"hardware_executed\": false,\n"
            "  \"automatic_retry\": false,\n"
            "  \"restoration_authorized\": false,\n"
            "  \"windows_seen\": %zu\n"
            "}\n",
            schedule->session_id, schedule->count);
    fclose(f);

    if (write_run_manifest(args->output_dir, schedule->session_id,
                           "VALIDATION_ONLY_HARDWARE_NOT_EXECUTED")) {
        die("manifest creation failed");
    }
}

int main(int argc, char **argv) {
    RunnerArgs args = parse_args(argc, argv);
    if (!exists(args.session_dir) || exists(args.output_dir)) {
        die(exists(args.output_dir) ? "refusing existing output directory"
                                    : "session directory does not exist");
    }
    Schedule schedule = load_schedule(&args);
    int rc = 0;
    if (args.mode == MODE_VALIDATE) {
        validation_outputs(&args, &schedule);
        printf("{\"status\":\"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\","
               "\"session_id\":\"%s\",\"windows\":%zu}\n",
               schedule.session_id, schedule.count);
    } else {
        rc = run_hardware(&args, &schedule);
    }
    free(schedule.windows);
    return rc;
}
