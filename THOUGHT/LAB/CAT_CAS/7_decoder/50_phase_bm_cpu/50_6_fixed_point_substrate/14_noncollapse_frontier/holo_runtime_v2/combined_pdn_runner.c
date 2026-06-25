#define _GNU_SOURCE
#include "combined_pdn_hardware.h"
#include "strict_json_validation.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
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

static int exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static const char *key(const char *json, const char *name) {
    return sj_object_value(json, name);
}

static const char *value(const char *json, const char *name) {
    const char *p = key(json, name);
    return p;
}

static int key_count(const char *json, const char *name) {
    return sj_count_key(json, name);
}

static int object_bounds(const char *json, const char *name,
                         const char **start, const char **end) {
    const char *p = value(json, name);
    if (!p || *p != '{') return -1;
    int depth = 0, in_string = 0;
    for (const char *cursor = p; *cursor; cursor++) {
        if (*cursor == '"' && (cursor == p || cursor[-1] != '\\')) in_string = !in_string;
        if (in_string) continue;
        if (*cursor == '{') depth++;
        if (*cursor == '}' && --depth == 0) {
            *start = p;
            *end = cursor;
            return 0;
        }
    }
    return -1;
}

static const char *object_value(const char *start, const char *end, const char *name) {
    return sj_object_value_bounded(start, end, name);
}

static int token_end(char c) {
    return c == 0 || c == ',' || c == '}' || c == ']' || isspace((unsigned char)c);
}

static int object_string(const char *json, const char *object_name,
                         const char *member, char *out, size_t size) {
    const char *start, *end;
    if (object_bounds(json, object_name, &start, &end)) return -1;
    const char *p = object_value(start, end, member);
    if (!p || *p++ != '"') return -1;
    size_t i = 0;
    while (p < end && *p != '"' && i < size - 1) {
        if (*p == '\\') {
            p++;
            if (p >= end || !*p) return -1;
            switch (*p) {
                case '"': out[i++] = '"'; break;
                case '\\': out[i++] = '\\'; break;
                case '/': out[i++] = '/'; break;
                case 'b': out[i++] = '\b'; break;
                case 'f': out[i++] = '\f'; break;
                case 'n': out[i++] = '\n'; break;
                case 'r': out[i++] = '\r'; break;
                case 't': out[i++] = '\t'; break;
                case 'u':
                    for (int j = 0; j < 4; j++) {
                        p++;
                        if (p >= end || !*p || !isxdigit((unsigned char)*p)) return -1;
                    }
                    out[i++] = '?';
                    break;
                default: return -1;
            }
        } else if ((unsigned char)*p < 0x20) {
            return -1;
        } else {
            out[i++] = *p;
        }
        p++;
    }
    if (p >= end || *p != '"') return -1;
    p++;
    if (p >= end || !token_end(*p)) return -1;
    out[i] = 0;
    return 0;
}

static int object_long_pair(const char *json, const char *object_name,
                            const char *member, long *first, long *second) {
    const char *start, *end;
    if (object_bounds(json, object_name, &start, &end)) return -1;
    const char *p = object_value(start, end, member);
    if (!p || *p++ != '[') return -1;
    char *tail = NULL;
    *first = strtol(p, &tail, 10);
    if (tail == p || !token_end(*tail)) return -1;
    p = tail;
    while (isspace((unsigned char)*p)) p++;
    if (*p++ != ',') return -1;
    while (isspace((unsigned char)*p)) p++;
    *second = strtol(p, &tail, 10);
    if (tail == p || !token_end(*tail)) return -1;
    p = tail;
    while (isspace((unsigned char)*p)) p++;
    return *p == ']' ? 0 : -1;
}

static int object_member_count(const char *json, const char *object_name) {
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

static int exact_singleton_string_array(const char *json, const char *name,
                                        const char *target) {
    const char *p = value(json, name);
    if (!p || *p++ != '[') return -1;
    while (isspace((unsigned char)*p)) p++;
    if (*p++ != '"') return -1;
    const char *end = strchr(p, '"');
    if (!end || (size_t)(end - p) != strlen(target) ||
        strncmp(p, target, (size_t)(end - p))) return -1;
    p = end + 1;
    while (isspace((unsigned char)*p)) p++;
    return *p == ']' ? 0 : -1;
}

static int jstr(const char *json, const char *name, char *out, size_t n) {
    const char *p = value(json, name);
    if (!p || *p != '"') return -1;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i < n - 1) {
        if (*p == '\\') {
            p++;
            if (!*p) return -1;
            switch (*p) {
                case '"': out[i++] = '"'; break;
                case '\\': out[i++] = '\\'; break;
                case '/': out[i++] = '/'; break;
                case 'b': out[i++] = '\b'; break;
                case 'f': out[i++] = '\f'; break;
                case 'n': out[i++] = '\n'; break;
                case 'r': out[i++] = '\r'; break;
                case 't': out[i++] = '\t'; break;
                case 'u':
                    for (int j = 0; j < 4; j++) {
                        p++;
                        if (!*p || !isxdigit((unsigned char)*p)) return -1;
                    }
                    out[i++] = '?';
                    break;
                default: return -1;
            }
        } else if ((unsigned char)*p < 0x20) {
            return -1;
        } else {
            out[i++] = *p;
        }
        p++;
    }
    if (*p != '"') return -1;
    p++;
    if (!token_end(*p)) return -1;
    out[i] = 0;
    return 0;
}

static int jlong(const char *json, const char *name, long *out) {
    const char *p = value(json, name);
    if (!p) return -1;
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p || !token_end(*end)) return -1;
    *out = v;
    return 0;
}

static int jdouble(const char *json, const char *name, double *out) {
    const char *p = value(json, name);
    if (!p) return -1;
    if (!strncmp(p, "NaN", 3) || !strncmp(p, "Infinity", 8) ||
        !strncmp(p, "infinity", 8) || !strncmp(p, "-Infinity", 9) ||
        !strncmp(p, "-infinity", 9)) return -1;
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p || !isfinite(v) || !token_end(*end)) return -1;
    *out = v;
    return 0;
}

static int jbool(const char *json, const char *name, int *out) {
    const char *p = value(json, name);
    if (!p) return -1;
    if (!strncmp(p, "true", 4) && token_end(p[4])) {
        *out = 1;
        return 0;
    }
    if (!strncmp(p, "false", 5) && token_end(p[5])) {
        *out = 0;
        return 0;
    }
    return -1;
}

static int jnullable_long(const char *json, const char *name, long *out, int *present) {
    const char *p = value(json, name);
    if (!p) return -1;
    if (!strncmp(p, "null", 4) && token_end(p[4])) {
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

static int zero_commit(const char *commit) {
    return commit && strlen(commit) == 40 && strspn(commit, "0") == 40;
}

static int valid_sha256(const char *digest) {
    if (!digest || strlen(digest) != 64) return 0;
    for (size_t i = 0; i < 64; i++) {
        char c = digest[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return 0;
    }
    return 1;
}

static void sha256(const char *path, char out[65]) {
    if (hash_file_streaming(path, out)) {
        die("sha256 file hash failed: %s", path);
    }
}

static int parse_long_arg(const char *text, long *out) {
    if (!text || !*text || isspace((unsigned char)text[0])) return -1;
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != 0) return -1;
    *out = value;
    return 0;
}

static int parse_int_arg(const char *text, int *out) {
    long value = 0;
    if (parse_long_arg(text, &value) || value < INT_MIN || value > INT_MAX) return -1;
    *out = (int)value;
    return 0;
}

static int parse_double_arg(const char *text, double *out) {
    if (!text || !*text || isspace((unsigned char)text[0])) return -1;
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (errno == ERANGE || end == text || *end != 0 || !isfinite(value)) return -1;
    *out = value;
    return 0;
}

static RunnerArgs parse_args(int argc, char **argv) {
    RunnerArgs args = {0};
    args.victim = -1;
    args.sender = -1;
    args.pin_khz = 1600000;
    args.slot_s = 0.5;
    args.off_window_s = 0.5;
    args.read_hz = 8000;
    args.temp_veto_c = 68;
    args.backend = BACKEND_REAL;

    for (int i = 1; i < argc; i++) {
        const char *key_name = argv[i];
        const char *arg = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(key_name, "--validate-only")) args.mode = MODE_VALIDATE;
        else if (!strcmp(key_name, "--hardware")) args.mode = MODE_HARDWARE;
        else if (!strcmp(key_name, "--engineering-smoke")) {
            args.mode = MODE_HARDWARE;
            args.engineering_smoke = 1;
        } else if (!strcmp(key_name, "--mock-hardware")) {
            args.mode = MODE_HARDWARE;
            args.backend = BACKEND_MOCK;
        } else if (!strcmp(key_name, "--executor-commit") && arg) {
            args.executor_commit = arg;
            i++;
        } else if (!strcmp(key_name, "--source-bundle-manifest") && arg) {
            args.source_bundle_manifest = arg;
            i++;
        } else if (!strcmp(key_name, "--authorization-artifact") && arg) {
            args.authorization_artifact = arg;
            i++;
        } else if (!strcmp(key_name, "--session-dir") && arg) {
            args.session_dir = arg;
            i++;
        } else if (!strcmp(key_name, "--output-dir") && arg) {
            args.output_dir = arg;
            i++;
        } else if (!strcmp(key_name, "--victim") && arg) {
            if (parse_int_arg(arg, &args.victim)) die("invalid numeric arguments");
            i++;
        } else if (!strcmp(key_name, "--sender") && arg) {
            if (parse_int_arg(arg, &args.sender)) die("invalid numeric arguments");
            i++;
        } else if (!strcmp(key_name, "--pin-khz") && arg) {
            if (parse_long_arg(arg, &args.pin_khz)) die("invalid numeric arguments");
            i++;
        } else if (!strcmp(key_name, "--slot-s") && arg) {
            if (parse_double_arg(arg, &args.slot_s)) die("invalid numeric arguments");
            i++;
        } else if (!strcmp(key_name, "--off-window-s") && arg) {
            if (parse_double_arg(arg, &args.off_window_s)) die("invalid numeric arguments");
            i++;
        } else if (!strcmp(key_name, "--read-hz") && arg) {
            if (parse_long_arg(arg, &args.read_hz)) die("invalid numeric arguments");
            i++;
        } else if (!strcmp(key_name, "--temp-veto-c") && arg) {
            if (parse_double_arg(arg, &args.temp_veto_c)) die("invalid numeric arguments");
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
    if (args.mode == MODE_HARDWARE && !args.source_bundle_manifest) {
        die("hardware mode requires --source-bundle-manifest");
    }
    if (args.mode == MODE_HARDWARE && args.backend == BACKEND_REAL &&
        !args.authorization_artifact) {
        die("real V2 calibration requires --authorization-artifact");
    }
    if (args.pin_khz <= 0 || args.read_hz <= 0 ||
        !isfinite(args.slot_s) || args.slot_s <= 0 ||
        !isfinite(args.off_window_s) || args.off_window_s <= 0 ||
        !isfinite(args.temp_veto_c)) {
        die("invalid numeric arguments");
    }
    if (!shell_safe(args.session_dir) || !shell_safe(args.output_dir) ||
        (args.authorization_artifact && !shell_safe(args.authorization_artifact)) ||
        (args.source_bundle_manifest && !shell_safe(args.source_bundle_manifest))) {
        die("unsafe path");
    }
    return args;
}


static void free_schedule_captured(Schedule *schedule) {
    free_captured(&schedule->captured_session_json);
    free_captured(&schedule->captured_windows_jsonl);
}

static Schedule load_schedule(const RunnerArgs *args) {
    char path[CP_PATH_MAX], schema[128], manifest_session[128];
    Schedule schedule = {0};

    if (path_join(path, sizeof(path), args->session_dir, "session_manifest.json")) {
        die("path too long");
    }
    CapturedFile captured_manifest = {0};
    if (capture_file(path, &captured_manifest, CAPTURED_MAX_SESSION_MANIFEST)) {
        die("cannot capture session manifest");
    }
    strcpy(schedule.session_manifest_sha256, captured_manifest.sha256);
    char *manifest = (char *)captured_manifest.bytes;
    manifest[captured_manifest.size] = 0;
    if (jstr(manifest, "schema_id", schema, sizeof(schema)) ||
        strcmp(schema, "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V2")) {
        die("unexpected session manifest schema");
    }
    if (jstr(manifest, "session_id", manifest_session, sizeof(manifest_session))) {
        die("manifest missing session_id");
    }

    if (path_join(path, sizeof(path), args->session_dir, "session.json")) {
        die("path too long");
    }
    if (capture_file(path, &schedule.captured_session_json, CAPTURED_MAX_SESSION_JSON)) {
        die("cannot capture session.json");
    }
    char *header = (char *)schedule.captured_session_json.bytes;
    header[schedule.captured_session_json.size] = 0;
    if (jstr(header, "schema_id", schema, sizeof(schema)) ||
        strcmp(schema, "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V2")) {
        die("unexpected session schema");
    }
    if (jstr(header, "session_id", schedule.session_id, sizeof(schedule.session_id)) ||
        strcmp(schedule.session_id, manifest_session)) {
        die("session ID mismatch");
    }
    if (jstr(header, "route", schedule.route, sizeof(schedule.route)) ||
        jstr(header, "partition", schedule.partition, sizeof(schedule.partition)) ||
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
    if (!valid_commit(schedule.campaign_source_commit) &&
        !(args->engineering_smoke && zero_commit(schedule.campaign_source_commit))) {
        die("real session requires nonzero campaign source commit");
    }

    if (path_join(path, sizeof(path), args->session_dir, "windows.jsonl")) {
        die("path too long");
    }
    if (capture_file(path, &schedule.captured_windows_jsonl, CAPTURED_MAX_WINDOWS_JSONL)) {
        die("cannot capture windows.jsonl");
    }

    {
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

    free_captured(&captured_manifest);

    schedule.count = (size_t)count;
    schedule.windows = calloc(schedule.count, sizeof(*schedule.windows));
    if (!schedule.windows) die("oom");
    FILE *f = fmemopen(schedule.captured_windows_jsonl.bytes,
                       schedule.captured_windows_jsonl.size, "r");
    if (!f) die("fmemopen windows");
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
        if (jnullable_long(line, "receiver_codeword_source_index", &x, &present)) {
            die("missing receiver codeword source");
        }
        window->receiver_codeword_source_index = present ? (int)x : -1;
        if (jnullable_long(line, "sender_codeword_source_index", &x, &present)) {
            die("missing sender codeword source");
        }
        window->sender_codeword_source_index = present ? (int)x : -1;
        if (jnullable_long(line, "receiver_theta_idx", &x, &present)) {
            die("missing receiver theta_idx");
        }
        window->receiver_theta_idx = present ? (int)x : -1;
        if (jnullable_long(line, "sender_theta_idx", &x, &present)) {
            die("missing sender theta_idx");
        }
        window->sender_theta_idx = present ? (int)x : -1;
        if (jbool(line, "shared_schedule", &window->shared_schedule)) {
            die("missing shared_schedule");
        }
        if (jstr(line, "scramble_key_digest", window->scramble_key_digest,
                 sizeof(window->scramble_key_digest)) ||
            !valid_sha256(window->scramble_key_digest)) {
            die("invalid scramble key digest");
        }
        if (jnullable_long(line, "sender_off_control_for_tone_index", &x, &present)) {
            window->sender_off_control_for_tone_index = -1;
        } else {
            window->sender_off_control_for_tone_index = present ? (int)x : -1;
        }
        if (jnullable_long(line, "sender_off_control_theta_idx", &x, &present)) {
            window->sender_off_control_theta_idx = -1;
        } else {
            window->sender_off_control_theta_idx = present ? (int)x : -1;
        }
        if (window->sender_off_required) {
            if (window->sender_off_control_for_tone_index < 0 ||
                window->sender_off_control_for_tone_index > 11 ||
                window->sender_off_control_theta_idx < 0 ||
                window->sender_off_control_theta_idx > 7) {
                die("sender-off control tone invalid");
            }
            if (window->physical_tone_index != -1 ||
                window->drive_on) {
                die("sender-off row must not drive");
            }
        } else {
            if (window->sender_off_control_for_tone_index != -1 ||
                window->sender_off_control_theta_idx != -1) {
                die("sender-on row must not carry control tone");
            }
        }
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
                (window->physical_tone_index < 0 ||
                 window->receiver_codeword_source_index < 0 ||
                 window->sender_codeword_source_index < 0 ||
                 window->receiver_theta_idx < 0 || window->sender_theta_idx < 0)) {
                die("driven lock-in missing physical tone or codeword source");
            }
        } else {
            die("unsupported measurement mode");
        }
        if (window->drive_on) {
            int same = window->receiver_codeword_source_index ==
                           window->sender_codeword_source_index &&
                       window->receiver_theta_idx == window->sender_theta_idx;
            if (window->shared_schedule != same) {
                die("physical scramble schedule does not match declared sharing");
            }
        }
        n++;
    }
    free(line);
    fclose(f);
    if (n != schedule.count) die("short schedule row count");

    if (!strcmp(schedule.route, "v4s5")) {
        if (args->victim != 4 || args->sender != 5) die("invalid route/core pair");
    } else if (!strcmp(schedule.route, "v2s3")) {
        if (args->victim != 2 || args->sender != 3) die("invalid route/core pair");
    } else {
        die("unsupported route");
    }
    return schedule;
}

static void verify_engineering_smoke(const Schedule *schedule) {
    if (strcmp(schedule->session_id, "ENGINEERING_SMOKE_TEST") ||
        strcmp(schedule->partition, "ENGINEERING_SMOKE_TEST_NOT_SCIENTIFIC_ACQUISITION") ||
        strcmp(schedule->route, "v4s5") || schedule->count != 3 ||
        strcmp(schedule->campaign_source_commit, "0000000000000000000000000000000000000000") ||
        strcmp(schedule->campaign_plan_sha256,
               "0000000000000000000000000000000000000000000000000000000000000000")) {
        die("engineering smoke schedule mismatch");
    }
    for (size_t i = 0; i < schedule->count; i++) {
        const Window *window = &schedule->windows[i];
        if (window->window_index != (long)i ||
            strcmp(window->block_id, "ENGINEERING_SMOKE_TEST") ||
            strcmp(window->family, "engineering_smoke") ||
            strcmp(window->executed_tone_order, "ENGINEERING") ||
            strcmp(window->declared_tone_order, "ENGINEERING")) {
            die("engineering smoke window mismatch");
        }
    }
    const Window *first = &schedule->windows[0];
    const Window *second = &schedule->windows[1];
    const Window *off = &schedule->windows[2];
    if (strcmp(first->stage, "ENGINEERING_SMOKE_DRIVEN") ||
        strcmp(first->actual_mode, "basis") || strcmp(first->declared_mode, "basis") ||
        strcmp(first->measurement_mode, "lockin_and_raw_ring") ||
        !first->drive_on || first->sender_off_required || first->amplitude_level != 1 ||
        first->physical_tone_index != 0 ||
        first->receiver_codeword_source_index != 0 ||
        first->sender_codeword_source_index != 0 ||
        first->receiver_theta_idx != 0 || first->sender_theta_idx != 0 ||
        strcmp(second->stage, "ENGINEERING_SMOKE_DRIVEN") ||
        strcmp(second->actual_mode, "rotation") ||
        strcmp(second->declared_mode, "rotation") ||
        strcmp(second->measurement_mode, "lockin_and_raw_ring") ||
        !second->drive_on || second->sender_off_required || second->amplitude_level != 1 ||
        second->physical_tone_index != 1 ||
        second->receiver_codeword_source_index != 1 ||
        second->sender_codeword_source_index != 1 ||
        second->receiver_theta_idx != 1 || second->sender_theta_idx != 1 ||
        strcmp(off->stage, "ENGINEERING_SMOKE_SENDER_OFF") ||
        strcmp(off->actual_mode, "null") || strcmp(off->declared_mode, "null") ||
        strcmp(off->measurement_mode, "raw_ring_sender_off") ||
        off->drive_on || !off->sender_off_required || off->amplitude_level != 0 ||
        off->physical_tone_index != -1 ||
        off->receiver_codeword_source_index != -1 ||
        off->sender_codeword_source_index != -1 ||
        off->receiver_theta_idx != -1 || off->sender_theta_idx != -1) {
        die("engineering smoke window mismatch");
    }
}

static int path_contained_in(const char *root, const char *path) {
    char root_real[CP_PATH_MAX];
    char path_parent_real[CP_PATH_MAX];
    char path_copy[CP_PATH_MAX];
    if (!realpath(root, root_real)) return 0;
    snprintf(path_copy, sizeof(path_copy), "%s", path);
    char *last_slash = strrchr(path_copy, '/');
    if (!last_slash || last_slash == path_copy) return 0;
    *last_slash = 0;
    if (!realpath(path_copy, path_parent_real)) return 0;
    size_t root_len = strlen(root_real);
    if (strncmp(root_real, path_parent_real, root_len)) return 0;
    char sep = path_parent_real[root_len];
    return sep == 0 || sep == '/';
}

static void verify_authorization(RunnerArgs *args, const Schedule *schedule) {
    char schema[96], executor_commit[41], executor_sha[65], actual_executor_sha[65];
    char campaign_source_commit[41];
    char executor_path[64];
    char plan_sha[65], source_bundle_sha[65];
    char bundle_schema[96], bundled_session_manifest_sha[65];
    char output_root[CP_PATH_MAX], authorized_by[256];
    int calibration = 0, acquisition = 1, restoration = 1;
    int target_coupling = 1, small_wall = 1, automatic_retry = 1;
    long pin_khz = 0, read_hz = 0;
    double slot_s = 0, off_window_s = 0, temp_veto_c = 0;
    CapturedFile captured_auth = {0};
    CapturedFile captured_bundle = {0};
    if (capture_file(args->authorization_artifact, &captured_auth,
                     CAPTURED_MAX_AUTHORIZATION)) {
        die("cannot capture authorization artifact");
    }
    if (capture_file(args->source_bundle_manifest, &captured_bundle,
                     CAPTURED_MAX_SOURCE_BUNDLE)) {
        free_captured(&captured_auth);
        die("cannot capture source bundle");
    }
    strcpy(args->authorization_digest, captured_auth.sha256);
    char *authorization = (char *)captured_auth.bytes;
    authorization[captured_auth.size] = 0;
    char *source_bundle = (char *)captured_bundle.bytes;
    source_bundle[captured_bundle.size] = 0;
    const char *authorization_fields[] = {
        "schema_id", "calibration_authorized", "acquisition_authorized",
        "restoration_authorized", "target_coupling_authorized",
        "small_wall_authorized", "automatic_retry", "executor_commit",
        "executor_sha256", "campaign_source_commit", "source_bundle_sha256",
        "campaign_plan_sha256", "session_ids", "route_cores", "pin_khz",
        "slot_s", "off_window_s", "read_hz", "temperature_veto_c",
        "authorized_output_root", "authorized_by"
    };
    const char *source_bundle_fields[] = {"schema_id", "sessions"};
    if (strict_json_document(authorization) ||
        strict_json_exact_top_object(
            authorization, authorization_fields,
            sizeof(authorization_fields) / sizeof(authorization_fields[0])) ||
        strict_json_document(source_bundle) ||
        strict_json_exact_top_object(
            source_bundle, source_bundle_fields,
            sizeof(source_bundle_fields) / sizeof(source_bundle_fields[0]))) {
        free_captured(&captured_auth);
        free_captured(&captured_bundle);
        die("invalid V2 calibration authorization artifact");
    }
    snprintf(executor_path, sizeof(executor_path), "/proc/%ld/exe", (long)getpid());
    sha256(executor_path, actual_executor_sha);
    const char *required_fields[] = {
        "schema_id", "calibration_authorized", "acquisition_authorized",
        "restoration_authorized", "target_coupling_authorized",
        "small_wall_authorized", "automatic_retry", "executor_commit",
        "executor_sha256", "campaign_source_commit", "source_bundle_sha256",
        "campaign_plan_sha256", "session_ids", "route_cores", "pin_khz",
        "slot_s", "off_window_s", "read_hz", "temperature_veto_c",
        "authorized_output_root", "authorized_by"
    };
    int unique = 1;
    for (size_t i = 0; i < sizeof(required_fields) / sizeof(required_fields[0]); i++) {
        if (key_count(authorization, required_fields[i]) != 1) unique = 0;
    }
    long v4_victim = -1, v4_sender = -1, v2_victim = -1, v2_sender = -1;
    int singleton_session = exact_singleton_string_array(
        authorization, "session_ids", schedule->session_id);
    int route_pairs =
        object_member_count(authorization, "route_cores") != 2 ||
        object_long_pair(authorization, "route_cores", "v4s5", &v4_victim, &v4_sender) ||
        object_long_pair(authorization, "route_cores", "v2s3", &v2_victim, &v2_sender) ||
        v4_victim != 4 || v4_sender != 5 || v2_victim != 2 || v2_sender != 3;
    int source_bundle_valid =
        key_count(source_bundle, "schema_id") == 1 &&
        key_count(source_bundle, "sessions") == 1 &&
        !jstr(source_bundle, "schema_id", bundle_schema, sizeof(bundle_schema)) &&
        !strcmp(bundle_schema, "CAT_CAS_PHASE6_V2_SOURCE_BUNDLE_MANIFEST_V1") &&
        object_member_count(source_bundle, "sessions") == 1 &&
        !object_string(source_bundle, "sessions", schedule->session_id,
                       bundled_session_manifest_sha,
                       sizeof(bundled_session_manifest_sha)) &&
        !strcmp(bundled_session_manifest_sha, schedule->session_manifest_sha256);
    int invalid =
        !unique || singleton_session || route_pairs ||
        !source_bundle_valid ||
        jstr(authorization, "schema_id", schema, sizeof(schema)) ||
        strcmp(schema, "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_AUTHORIZATION_V1") ||
        jbool(authorization, "calibration_authorized", &calibration) || !calibration ||
        jbool(authorization, "acquisition_authorized", &acquisition) || acquisition ||
        jbool(authorization, "restoration_authorized", &restoration) || restoration ||
        jbool(authorization, "target_coupling_authorized", &target_coupling) ||
            target_coupling ||
        jbool(authorization, "small_wall_authorized", &small_wall) || small_wall ||
        jbool(authorization, "automatic_retry", &automatic_retry) || automatic_retry ||
        jstr(authorization, "executor_commit", executor_commit, sizeof(executor_commit)) ||
        strcmp(executor_commit, args->executor_commit) ||
        jstr(authorization, "executor_sha256", executor_sha, sizeof(executor_sha)) ||
        strcmp(executor_sha, actual_executor_sha) ||
        jstr(authorization, "campaign_source_commit", campaign_source_commit,
             sizeof(campaign_source_commit)) ||
        !valid_commit(campaign_source_commit) ||
        strcmp(campaign_source_commit, schedule->campaign_source_commit) ||
        jstr(authorization, "campaign_plan_sha256", plan_sha, sizeof(plan_sha)) ||
        strcmp(plan_sha, schedule->campaign_plan_sha256) ||
        jstr(authorization, "source_bundle_sha256", source_bundle_sha,
             sizeof(source_bundle_sha)) || !valid_sha256(source_bundle_sha) ||
        strcmp(source_bundle_sha, captured_bundle.sha256) ||
        jlong(authorization, "pin_khz", &pin_khz) || pin_khz != args->pin_khz ||
        jlong(authorization, "read_hz", &read_hz) || read_hz != args->read_hz ||
        jdouble(authorization, "slot_s", &slot_s) || slot_s != args->slot_s ||
        jdouble(authorization, "off_window_s", &off_window_s) ||
            off_window_s != args->off_window_s ||
        jdouble(authorization, "temperature_veto_c", &temp_veto_c) ||
            temp_veto_c != args->temp_veto_c ||
        jstr(authorization, "authorized_output_root", output_root, sizeof(output_root)) ||
        output_root[0] != '/' || !shell_safe(output_root) ||
        !path_contained_in(output_root, args->output_dir) ||
        jstr(authorization, "authorized_by", authorized_by, sizeof(authorized_by)) ||
        !authorized_by[0] ||
        (strspn(authorized_by, " \t\r\n\v\f") == strlen(authorized_by));
    free_captured(&captured_auth);
    free_captured(&captured_bundle);
    if (invalid) die("invalid V2 calibration authorization artifact");
}

static void verify_execution_gate(RunnerArgs *args, const Schedule *schedule) {
    if (args->mode != MODE_HARDWARE) return;
    if (args->engineering_smoke) verify_engineering_smoke(schedule);
    if (args->backend == BACKEND_REAL) {
        verify_authorization(args, schedule);
    }
}

static void validation_outputs(const RunnerArgs *args, const Schedule *schedule) {
    if (exists(args->output_dir) || mkdir(args->output_dir, 0755)) {
        die("refusing existing output directory");
    }
    char destination[CP_PATH_MAX];
    const CapturedFile *inputs[2] = {&schedule->captured_session_json,
                                     &schedule->captured_windows_jsonl};
    const char *names[2] = {"session.json", "windows.jsonl"};
    for (int i = 0; i < 2; i++) {
        if (path_join(destination, sizeof(destination), args->output_dir, names[i])) {
            die("path too long");
        }
        if (write_captured_exclusive(destination, inputs[i])) {
            die("write captured bytes failed");
        }
    }
    const char *empty[] = {"raw_samples.bin", "telemetry.csv", "stderr.log",
                           "orchestrator_stdout.log", "orchestrator_stderr.log"};
    for (int i = 0; i < 5; i++) {
        if (path_join(destination, sizeof(destination), args->output_dir, empty[i])) {
            die("path too long");
        }
        FILE *f = fopen(destination, "wbx");
        if (!f) die("create output");
        fclose(f);
    }

    if (path_join(destination, sizeof(destination), args->output_dir, "stdout.log")) die("path too long");
    FILE *f = fopen(destination, "wx");
    if (!f) die("create stdout.log");
    fprintf(f, "VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\n");
    fclose(f);

    if (path_join(destination, sizeof(destination), args->output_dir, "window_results.csv")) die("path too long");
    f = fopen(destination, "wx");
    if (!f) die("create window_results.csv");
    fprintf(f, "window_index,session_id,validation_status,hardware_executed\n");
    for (size_t i = 0; i < schedule->count; i++) {
        fprintf(f, "%zu,%s,VALIDATED,0\n", i, schedule->session_id);
    }
    fclose(f);

    if (path_join(destination, sizeof(destination), args->output_dir, "run.json")) die("path too long");
    f = fopen(destination, "wx");
    if (!f) die("create run.json");
    fprintf(f,
            "{\n"
            "  \"schema_id\": \"CAT_CAS_PHASE6_V2_VALIDATION_ONLY_RUN_V1\",\n"
            "  \"session_id\": \"%s\",\n"
            "  \"status\": \"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\",\n"
            "  \"execution_class\": \"VALIDATION_ONLY\",\n"
            "  \"hardware_executed\": false,\n"
            "  \"automatic_retry\": false,\n"
            "  \"restoration_authorized\": false,\n"
            "  \"calibration_authorized\": false,\n"
            "  \"acquisition_authorized\": false,\n"
            "  \"scientific_acquisition_authorized\": false,\n"
            "  \"target_coupling_authorized\": false,\n"
            "  \"small_wall_authorized\": false,\n"
            "  \"physical_carrier_restoration_claimed\": false,\n"
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
    verify_execution_gate(&args, &schedule);
    int rc = 0;
    if (args.mode == MODE_VALIDATE) {
        validation_outputs(&args, &schedule);
        printf("{\"status\":\"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\","
               "\"session_id\":\"%s\",\"windows\":%zu}\n",
               schedule.session_id, schedule.count);
    } else {
        rc = run_hardware(&args, &schedule);
    }
    free_schedule_captured(&schedule);
    free(schedule.windows);
    return rc;
}
