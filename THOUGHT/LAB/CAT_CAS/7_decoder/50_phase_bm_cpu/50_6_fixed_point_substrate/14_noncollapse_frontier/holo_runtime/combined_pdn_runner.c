#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define PATH_MAX_LOCAL 4096
#define SHA_LEN 64

typedef struct {
    const char *session_dir;
    const char *output_dir;
    int victim;
    int sender;
    long pin_khz;
    double slot_s;
    double off_window_s;
    long read_hz;
    double temp_veto_c;
    int validate_only;
} Args;

typedef struct {
    char session_id[128];
    long window_count;
    long driven_windows;
    long sender_off_windows;
} SessionInfo;

static void fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("ERROR: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(2);
}

static int join(char *out, size_t out_size, const char *root, const char *leaf) {
    if (!root || !leaf || leaf[0] == '/' || strstr(leaf, "..") || strchr(leaf, '\'')) return -1;
    int n = snprintf(out, out_size, "%s/%s", root, leaf);
    return n < 0 || (size_t)n >= out_size ? -1 : 0;
}

static int exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static long fsize(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) return -1;
    return (long)st.st_size;
}

static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) fail("open %s: %s", path, strerror(errno));
    if (fseek(f, 0, SEEK_END) != 0) fail("seek %s", path);
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) fail("seek %s", path);
    char *buf = calloc((size_t)n + 1, 1);
    if (!buf) fail("oom");
    if (n && fread(buf, 1, (size_t)n, f) != (size_t)n) fail("read %s", path);
    fclose(f);
    return buf;
}

static const char *key_at(const char *json, const char *key) {
    char needle[160];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    return strstr(json, needle);
}

static int json_string(const char *json, const char *key, char *out, size_t out_size) {
    const char *p = key_at(json, key);
    if (!p || !(p = strchr(p, ':'))) return -1;
    for (++p; *p && isspace((unsigned char)*p); ++p) {}
    if (*p != '"') return -1;
    const char *q = strchr(++p, '"');
    if (!q) return -1;
    size_t n = (size_t)(q - p);
    if (!n || n >= out_size) return -1;
    memcpy(out, p, n);
    out[n] = 0;
    return 0;
}

static int json_long(const char *json, const char *key, long *out) {
    const char *p = key_at(json, key);
    if (!p || !(p = strchr(p, ':'))) return -1;
    for (++p; *p && isspace((unsigned char)*p); ++p) {}
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p) return -1;
    *out = v;
    return 0;
}

static int json_bool(const char *json, const char *key, int *out) {
    const char *p = key_at(json, key);
    if (!p || !(p = strchr(p, ':'))) return -1;
    for (++p; *p && isspace((unsigned char)*p); ++p) {}
    if (strncmp(p, "true", 4) == 0) { *out = 1; return 0; }
    if (strncmp(p, "false", 5) == 0) { *out = 0; return 0; }
    return -1;
}

static int safe_shell_path(const char *path) {
    for (const char *p = path; p && *p; ++p) {
        if ((unsigned char)*p < 32 || strchr("';&|`$<>", *p)) return 0;
    }
    return path && *path;
}

static void sha256_hex(const char *path, char out[SHA_LEN + 1]) {
    if (!safe_shell_path(path)) fail("unsafe path for sha256: %s", path);
    char cmd[PATH_MAX_LOCAL + 64];
    snprintf(cmd, sizeof(cmd), "sha256sum '%s'", path);
    FILE *p = popen(cmd, "r");
    if (!p) fail("sha256sum failed");
    char line[128];
    if (!fgets(line, sizeof(line), p)) fail("sha256sum no output");
    int status = pclose(p);
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) fail("sha256sum failed for %s", path);
    for (int i = 0; i < SHA_LEN; ++i) {
        if (!isxdigit((unsigned char)line[i])) fail("bad sha256 output");
        out[i] = (char)tolower((unsigned char)line[i]);
    }
    out[SHA_LEN] = 0;
}

static void verify_manifest_file(const char *manifest, const char *root, const char *name) {
    const char *entry = key_at(manifest, name);
    if (!entry) fail("session_manifest missing %s", name);
    long want_size = -1;
    char want_sha[SHA_LEN + 1];
    char path[PATH_MAX_LOCAL], got_sha[SHA_LEN + 1];
    if (json_long(entry, "size", &want_size) != 0) fail("manifest missing size for %s", name);
    if (json_string(entry, "sha256", want_sha, sizeof(want_sha)) != 0) fail("manifest missing sha256 for %s", name);
    if (strlen(want_sha) != SHA_LEN) fail("manifest sha256 length invalid for %s", name);
    if (join(path, sizeof(path), root, name) != 0) fail("unsafe manifest path %s", name);
    long got_size = fsize(path);
    if (got_size != want_size) fail("size mismatch for %s", name);
    sha256_hex(path, got_sha);
    if (strcmp(want_sha, got_sha) != 0) fail("sha256 mismatch for %s", name);
}

static void verify_session_manifest(const char *dir) {
    char path[PATH_MAX_LOCAL], schema[128];
    if (join(path, sizeof(path), dir, "session_manifest.json") != 0) fail("bad session path");
    char *manifest = slurp(path);
    if (json_string(manifest, "schema_id", schema, sizeof(schema)) != 0 ||
        strcmp(schema, "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1") != 0) fail("unexpected session manifest schema");
    verify_manifest_file(manifest, dir, "session.json");
    verify_manifest_file(manifest, dir, "windows.jsonl");
    free(manifest);
}

static SessionInfo verify_session(const char *dir) {
    char path[PATH_MAX_LOCAL], schema[128];
    if (join(path, sizeof(path), dir, "session.json") != 0) fail("bad session path");
    char *session = slurp(path);
    SessionInfo info;
    memset(&info, 0, sizeof(info));
    if (json_string(session, "schema_id", schema, sizeof(schema)) != 0 ||
        strcmp(schema, "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V1") != 0) fail("unexpected session schema");
    if (json_string(session, "session_id", info.session_id, sizeof(info.session_id)) != 0) fail("missing session_id");
    if (json_long(session, "window_count", &info.window_count) != 0 || info.window_count <= 0) fail("bad window_count");
    int restoration = 1;
    if (json_bool(session, "restoration_authorized", &restoration) != 0 || restoration) fail("restoration_authorized must be false");
    free(session);

    if (join(path, sizeof(path), dir, "windows.jsonl") != 0) fail("bad windows path");
    FILE *f = fopen(path, "r");
    if (!f) fail("open windows.jsonl: %s", strerror(errno));
    char *line = NULL;
    size_t cap = 0;
    long expected = 0;
    while (getline(&line, &cap, f) != -1) {
        long index = -1;
        char sid[128], mode[64];
        int drive = -1, off = -1;
        if (json_long(line, "window_index", &index) != 0 || index != expected) fail("window_index not contiguous");
        if (json_string(line, "session_id", sid, sizeof(sid)) != 0 || strcmp(sid, info.session_id) != 0) fail("window session_id mismatch");
        if (json_string(line, "measurement_mode", mode, sizeof(mode)) != 0) fail("missing measurement_mode");
        if (json_bool(line, "drive_on", &drive) != 0 || json_bool(line, "sender_off_required", &off) != 0) fail("bad window booleans");
        if (off && drive) fail("sender-off window has drive_on=true");
        if (strcmp(mode, "raw_ring_sender_off") == 0) {
            if (!off || drive) fail("raw_ring_sender_off semantics violated");
            info.sender_off_windows++;
        } else if (strcmp(mode, "lockin_and_raw_ring") == 0) {
            info.driven_windows++;
        } else fail("unsupported measurement_mode %s", mode);
        expected++;
    }
    free(line);
    fclose(f);
    if (expected != info.window_count) fail("window_count mismatch");
    return info;
}

static Args parse_args(int argc, char **argv) {
    Args a;
    memset(&a, 0, sizeof(a));
    a.victim = a.sender = -1;
    a.pin_khz = 1600000;
    a.slot_s = 0.5;
    a.off_window_s = 0.5;
    a.read_hz = 4000;
    a.temp_veto_c = 68.0;
    for (int i = 1; i < argc; ++i) {
        const char *k = argv[i], *v = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (strcmp(k, "--validate-only") == 0) a.validate_only = 1;
        else if (strcmp(k, "--session-dir") == 0 && v) { a.session_dir = v; ++i; }
        else if (strcmp(k, "--output-dir") == 0 && v) { a.output_dir = v; ++i; }
        else if (strcmp(k, "--victim") == 0 && v) { a.victim = atoi(v); ++i; }
        else if (strcmp(k, "--sender") == 0 && v) { a.sender = atoi(v); ++i; }
        else if (strcmp(k, "--pin-khz") == 0 && v) { a.pin_khz = atol(v); ++i; }
        else if (strcmp(k, "--slot-s") == 0 && v) { a.slot_s = atof(v); ++i; }
        else if (strcmp(k, "--off-window-s") == 0 && v) { a.off_window_s = atof(v); ++i; }
        else if (strcmp(k, "--read-hz") == 0 && v) { a.read_hz = atol(v); ++i; }
        else if (strcmp(k, "--temp-veto-c") == 0 && v) { a.temp_veto_c = atof(v); ++i; }
        else fail("unknown or incomplete option: %s", k);
    }
    if (!a.session_dir || !a.output_dir || a.victim < 0 || a.sender < 0) fail("missing required arguments");
    if (a.victim == a.sender) fail("victim and sender cores must differ");
    return a;
}

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    if (!f) fail("write %s: %s", path, strerror(errno));
    fputs(text, f);
    fclose(f);
}

static void copy_file(const char *src, const char *dst) {
    FILE *in = fopen(src, "rb"), *out = fopen(dst, "wbx");
    if (!in || !out) fail("copy failed");
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0) if (fwrite(buf, 1, n, out) != n) fail("copy write failed");
    fclose(in); fclose(out);
}

static void emit_outputs(const Args *a, const SessionInfo *info) {
    if (exists(a->output_dir)) fail("refusing existing output directory");
    if (mkdir(a->output_dir, 0755) != 0) fail("mkdir output: %s", strerror(errno));
    char src[PATH_MAX_LOCAL], dst[PATH_MAX_LOCAL];
    join(src, sizeof(src), a->session_dir, "session.json"); join(dst, sizeof(dst), a->output_dir, "session.json"); copy_file(src, dst);
    join(src, sizeof(src), a->session_dir, "windows.jsonl"); join(dst, sizeof(dst), a->output_dir, "windows.jsonl"); copy_file(src, dst);
    join(dst, sizeof(dst), a->output_dir, "raw_samples.bin"); FILE *raw = fopen(dst, "wbx"); if (!raw) fail("raw create failed"); fclose(raw);
    join(dst, sizeof(dst), a->output_dir, "stdout.log"); write_file(dst, "combined_pdn_runner validation-only scaffold\n");
    join(dst, sizeof(dst), a->output_dir, "stderr.log"); write_file(dst, "");
    join(dst, sizeof(dst), a->output_dir, "window_results.csv");
    FILE *csv = fopen(dst, "wx"); if (!csv) fail("window_results create failed");
    fprintf(csv, "session_id,window_count,driven_windows,sender_off_windows,validate_only,hardware_executed\n");
    fprintf(csv, "%s,%ld,%ld,%ld,1,0\n", info->session_id, info->window_count, info->driven_windows, info->sender_off_windows); fclose(csv);
    join(dst, sizeof(dst), a->output_dir, "telemetry.csv");
    FILE *tel = fopen(dst, "wx"); if (!tel) fail("telemetry create failed");
    fprintf(tel, "victim_core,sender_core,pin_khz,slot_s,off_window_s,read_hz,temp_veto_c\n%d,%d,%ld,%.17g,%.17g,%ld,%.17g\n", a->victim, a->sender, a->pin_khz, a->slot_s, a->off_window_s, a->read_hz, a->temp_veto_c); fclose(tel);
    join(dst, sizeof(dst), a->output_dir, "run.json");
    FILE *run = fopen(dst, "wx"); if (!run) fail("run create failed");
    fprintf(run, "{\n  \"schema_id\": \"CAT_CAS_PHASE6_COMBINED_RUN_V1\",\n  \"session_id\": \"%s\",\n  \"status\": \"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\",\n  \"hardware_executed\": false,\n  \"restoration_authorized\": false,\n  \"automatic_retry\": false,\n  \"victim_core\": %d,\n  \"sender_core\": %d,\n  \"windows_seen\": %ld,\n  \"sender_off_windows\": %ld\n}\n", info->session_id, a->victim, a->sender, info->window_count, info->sender_off_windows); fclose(run);
    const char *names[] = {"run.json", "session.json", "windows.jsonl", "window_results.csv", "raw_samples.bin", "telemetry.csv", "stdout.log", "stderr.log"};
    join(dst, sizeof(dst), a->output_dir, "run_manifest.json");
    FILE *mf = fopen(dst, "wx"); if (!mf) fail("manifest create failed");
    fprintf(mf, "{\n  \"schema_id\": \"CAT_CAS_PHASE6_COMBINED_RUN_MANIFEST_V1\",\n  \"session_id\": \"%s\",\n  \"status\": \"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\",\n  \"files\": {\n", info->session_id);
    for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
        char fp[PATH_MAX_LOCAL], sha[SHA_LEN + 1];
        join(fp, sizeof(fp), a->output_dir, names[i]); sha256_hex(fp, sha);
        fprintf(mf, "    \"%s\": {\"size\": %ld, \"sha256\": \"%s\"}%s\n", names[i], fsize(fp), sha, i + 1 == sizeof(names)/sizeof(names[0]) ? "" : ",");
    }
    fprintf(mf, "  }\n}\n"); fclose(mf);
}

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    if (!exists(args.session_dir)) fail("session directory does not exist");
    verify_session_manifest(args.session_dir);
    SessionInfo info = verify_session(args.session_dir);
    if (!args.validate_only) fail("hardware execution path is not implemented in this scaffold; complete the CAT_CAS local hardware backend");
    emit_outputs(&args, &info);
    printf("{\"status\":\"VALIDATION_ONLY_HARDWARE_NOT_EXECUTED\",\"session_id\":\"%s\",\"windows\":%ld,\"sender_off_windows\":%ld}\n", info.session_id, info.window_count, info.sender_off_windows);
    return 0;
}
