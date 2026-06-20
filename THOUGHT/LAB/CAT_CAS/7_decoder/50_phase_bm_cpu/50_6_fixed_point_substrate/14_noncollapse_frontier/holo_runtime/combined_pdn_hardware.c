#define _GNU_SOURCE
#include "combined_pdn_hardware.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#define NCPU 6
#define MSR_MPERF 0xE7
#define MSR_APERF 0xE8
#define MSR_COFVID 0xC0010071
#define START_GUARD_SECONDS 0.050
#define MAX_EPOCH_SKEW_SECONDS 0.005

static volatile sig_atomic_t interrupted;

static inline uint64_t rdtsc_now(void) {
    unsigned hi, lo;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtscp_now(void) {
    unsigned hi, lo, aux;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    return ((uint64_t)hi << 32) | lo;
}

static void on_signal(int sig) {
    (void)sig;
    interrupted = 1;
}

static int joinp(char *out, size_t n, const char *a, const char *b) {
    size_t x = strlen(a), y = strlen(b);
    if (x + 1 + y + 1 > n) return -1;
    memcpy(out, a, x);
    out[x] = '/';
    memcpy(out + x + 1, b, y + 1);
    return 0;
}

static int read_long(const char *path, long *value) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int rc = fscanf(f, "%ld", value) == 1 ? 0 : -1;
    fclose(f);
    return rc;
}

static int write_long(const char *path, long value) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    int rc = fprintf(f, "%ld\n", value) > 0 ? 0 : -1;
    return fclose(f) || rc ? -1 : 0;
}

static int msr_read(int core, uint32_t reg, uint64_t *value) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/cpu/%d/msr", core);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int rc = pread(fd, value, 8, reg) == 8 ? 0 : -1;
    close(fd);
    return rc;
}

static int pin_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set)) return -1;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) || !CPU_ISSET(core, &set)) return -1;
    return 0;
}

static char temp_path[512];

static int locate_temp(void) {
    DIR *dir = opendir("/sys/class/hwmon");
    if (!dir) return -1;
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (strncmp(entry->d_name, "hwmon", 5)) continue;
        char path[512], name[64];
        snprintf(path, sizeof(path), "/sys/class/hwmon/%s/name", entry->d_name);
        FILE *f = fopen(path, "r");
        if (f && fgets(name, sizeof(name), f)) {
            name[strcspn(name, "\r\n")] = 0;
            if (!strcmp(name, "k10temp")) {
                snprintf(temp_path, sizeof(temp_path),
                         "/sys/class/hwmon/%s/temp1_input", entry->d_name);
                fclose(f);
                closedir(dir);
                return 0;
            }
        }
        if (f) fclose(f);
    }
    closedir(dir);
    return -1;
}

static double temperature(void) {
    long value;
    if (!temp_path[0] || read_long(temp_path, &value)) return NAN;
    return value / 1000.0;
}

static long cur_khz(int core) {
    char path[160];
    long value;
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", core);
    return read_long(path, &value) ? -1 : value;
}

typedef struct {
    long min[NCPU], max[NCPU];
    int have[NCPU], boost_have;
    long boost;
    int changed;
} PinState;

static int snapshot(PinState *state) {
    memset(state, 0, sizeof(*state));
    char path[160];
    for (int core = 0; core < NCPU; core++) {
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_min_freq", core);
        if (read_long(path, &state->min[core])) return -1;
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", core);
        if (read_long(path, &state->max[core])) return -1;
        state->have[core] = 1;
    }
    if (!read_long("/sys/devices/system/cpu/cpufreq/boost", &state->boost)) {
        state->boost_have = 1;
    }
    return 0;
}

static int pin_frequency(PinState *state, long khz) {
    char min_path[160], max_path[160];
    state->changed = 1;
    if (state->boost_have) {
        long readback;
        if (write_long("/sys/devices/system/cpu/cpufreq/boost", 0) ||
            read_long("/sys/devices/system/cpu/cpufreq/boost", &readback) ||
            readback != 0) {
            return -1;
        }
    }
    for (int core = 0; core < NCPU; core++) {
        snprintf(min_path, sizeof(min_path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_min_freq", core);
        snprintf(max_path, sizeof(max_path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", core);
        if (write_long(min_path, khz < state->max[core] ? khz : state->max[core]) ||
            write_long(max_path, khz) || write_long(min_path, khz)) {
            return -1;
        }
        int ok = 0;
        for (int attempt = 0; attempt < 10; attempt++) {
            long min_value, max_value;
            if (!read_long(min_path, &min_value) &&
                !read_long(max_path, &max_value) &&
                min_value == khz && max_value == khz) {
                ok = 1;
                break;
            }
            struct timespec settle = {0, 20000000};
            nanosleep(&settle, NULL);
        }
        if (!ok) return -1;
    }
    return 0;
}

static int restore(PinState *state) {
    char min_path[160], max_path[160];
    int ok = 1;
    for (int core = 0; core < NCPU; core++) {
        if (!state->have[core]) continue;
        snprintf(min_path, sizeof(min_path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_min_freq", core);
        snprintf(max_path, sizeof(max_path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", core);
        if (write_long(min_path, state->min[core]) ||
            write_long(max_path, state->max[core]) ||
            write_long(min_path, state->min[core])) {
            ok = 0;
        }
    }
    if (state->boost_have &&
        write_long("/sys/devices/system/cpu/cpufreq/boost", state->boost)) {
        ok = 0;
    }
    struct timespec settle = {0, 200000000};
    nanosleep(&settle, NULL);
    for (int core = 0; core < NCPU; core++) {
        if (!state->have[core]) continue;
        long min_value, max_value;
        snprintf(min_path, sizeof(min_path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_min_freq", core);
        snprintf(max_path, sizeof(max_path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", core);
        if (read_long(min_path, &min_value) ||
            read_long(max_path, &max_value) ||
            min_value != state->min[core] || max_value != state->max[core]) {
            ok = 0;
        }
    }
    if (state->boost_have) {
        long boost;
        if (read_long("/sys/devices/system/cpu/cpufreq/boost", &boost) ||
            boost != state->boost) {
            ok = 0;
        }
    }
    return ok ? 0 : -1;
}

static double calibrate_tsc(void) {
    struct timespec start, end, sleep_for = {0, 50000000};
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    uint64_t first = rdtsc_now();
    nanosleep(&sleep_for, NULL);
    uint64_t last = rdtsc_now();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    double seconds = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) * 1e-9;
    return seconds > 0 ? (last - first) / seconds : 0;
}

static double alu_burst(uint64_t *state) {
    double acc = 1.000001;
    uint64_t x = *state;
    for (int i = 0; i < 64; i++) {
        acc = acc * 1.00000007 + .99999993;
        x = x * 6364136223846793005ULL + 1;
    }
    *state = x;
    return acc + (x & 255);
}

typedef struct {
    atomic_int stop;
    atomic_int ready;
    atomic_int go;
    atomic_ullong origin;
    atomic_ullong ready_tsc;
    atomic_ullong epoch_tsc;
    atomic_ullong first_drive_tsc;
    pthread_t thread;
    int alive;
    int core;
    int level;
    double half_ticks;
    double phase;
} Sender;

static void *sender_loop(void *opaque) {
    Sender *sender = opaque;
    if (pin_core(sender->core)) return (void *)1;

    atomic_store_explicit(&sender->ready_tsc, rdtsc_now(), memory_order_release);
    atomic_store_explicit(&sender->ready, 1, memory_order_release);
    while (!atomic_load_explicit(&sender->go, memory_order_acquire) &&
           !atomic_load_explicit(&sender->stop, memory_order_acquire)) {
        __asm__ volatile("pause");
    }
    if (atomic_load_explicit(&sender->stop, memory_order_acquire)) return NULL;

    uint64_t origin = atomic_load_explicit(&sender->origin, memory_order_acquire);
    while (rdtsc_now() < origin &&
           !atomic_load_explicit(&sender->stop, memory_order_acquire)) {
        __asm__ volatile("pause");
    }
    atomic_store_explicit(&sender->epoch_tsc, rdtsc_now(), memory_order_release);

    uint64_t seed = 0x9e3779b9u + (uint64_t)sender->core;
    volatile double sink = 0;
    double period = 2 * sender->half_ticks;
    while (!atomic_load_explicit(&sender->stop, memory_order_acquire)) {
        uint64_t now = rdtsc_now();
        double offset = (double)(now - origin) - sender->phase * period;
        long half = (long)floor(offset / sender->half_ticks);
        int quadrant = (int)((half % 8 + 8) % 8);
        if (quadrant < sender->level * 2) {
            unsigned long long expected = 0;
            atomic_compare_exchange_strong_explicit(
                &sender->first_drive_tsc, &expected, now,
                memory_order_release, memory_order_relaxed);
            sink += alu_burst(&seed);
        } else {
            __asm__ volatile("pause");
        }
    }
    return (void *)(uintptr_t)(sink != sink);
}

static int sender_arm(Sender *sender, int core, double tone_hz, double tsc_hz,
                      int phase, int level) {
    memset(sender, 0, sizeof(*sender));
    sender->core = core;
    sender->half_ticks = .5 * tsc_hz / tone_hz;
    sender->phase = phase / 8.0;
    sender->level = level < 1 ? 1 : (level > 3 ? 3 : level);
    atomic_init(&sender->stop, 0);
    atomic_init(&sender->ready, 0);
    atomic_init(&sender->go, 0);
    atomic_init(&sender->origin, 0);
    atomic_init(&sender->ready_tsc, 0);
    atomic_init(&sender->epoch_tsc, 0);
    atomic_init(&sender->first_drive_tsc, 0);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
    int rc = pthread_create(&sender->thread, &attr, sender_loop, sender);
    pthread_attr_destroy(&attr);
    if (rc) return -1;
    sender->alive = 1;

    uint64_t timeout = rdtsc_now() + (uint64_t)(0.5 * tsc_hz);
    while (!atomic_load_explicit(&sender->ready, memory_order_acquire)) {
        if (rdtsc_now() > timeout) return -1;
        __asm__ volatile("pause");
    }
    return 0;
}

static void sender_release(Sender *sender, uint64_t origin) {
    atomic_store_explicit(&sender->origin, origin, memory_order_release);
    atomic_store_explicit(&sender->go, 1, memory_order_release);
}

static int sender_stop(Sender *sender) {
    if (!sender->alive) return 0;
    if (!sender->thread) {
        sender->alive = 0;
        return 0;
    }
    atomic_store_explicit(&sender->stop, 1, memory_order_release);
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += 2;
    void *result = NULL;
    int rc = pthread_timedjoin_np(sender->thread, &result, &timeout);
    sender->alive = 0;
    return rc || result ? -1 : 0;
}

static int capture_at_origin(int core, long hz, double seconds, double tsc_hz,
                             uint64_t origin, uint64_t *receiver_epoch,
                             uint64_t *timestamps, double *observations, int cap) {
    if (pin_core(core)) return -1;
    while (rdtsc_now() < origin && !interrupted) __asm__ volatile("pause");
    *receiver_epoch = rdtsc_now();

    uint64_t end = origin + (uint64_t)(seconds * tsc_hz);
    uint64_t previous = rdtscp_now();
    uint64_t acc = 0x9e3779b9u + (uint64_t)core;
    double span = tsc_hz / hz;
    int count = 0;
    while (count < cap && rdtsc_now() < end && !interrupted) {
        uint64_t now, iterations = 0;
        do {
            acc = acc * 6364136223846793005ULL + 1;
            iterations++;
            now = rdtscp_now();
        } while ((double)(now - previous) < span);
        timestamps[count] = now;
        observations[count] = (double)(now - previous) / iterations;
        previous = now;
        count++;
    }
    return count;
}

static void lockin(const uint64_t *timestamps, const double *samples, int count,
                   double frequency, uint64_t origin, double tsc_hz,
                   double *i_out, double *q_out, double *magnitude, double *floor) {
    double mean = 0;
    for (int i = 0; i < count; i++) mean += samples[i];
    mean /= count;

    double i_acc = 0, q_acc = 0, f_i = 0, f_q = 0, weight = 0;
    for (int i = 0; i < count; i++) {
        double window = .5 * (1 - cos(2 * M_PI * i / (count - 1)));
        double sample = (samples[i] - mean) * window;
        double seconds = (timestamps[i] - origin) / tsc_hz;
        i_acc += sample * cos(2 * M_PI * frequency * seconds);
        q_acc += sample * sin(2 * M_PI * frequency * seconds);
        f_i += sample * cos(2 * M_PI * (frequency * 1.37 + .071) * seconds);
        f_q += sample * sin(2 * M_PI * (frequency * 1.37 + .071) * seconds);
        weight += window;
    }
    *i_out = 2 * i_acc / weight;
    *q_out = 2 * q_acc / weight;
    *magnitude = hypot(*i_out, *q_out);
    *floor = 2 * hypot(f_i, f_q) / weight;
}

static double tone(int index) {
    double low = log(20), high = log(1500), x = index / 11.0;
    return exp(low + (high - low) * x) *
           (1 + .013 * sin(2.399963 * (index + 1)));
}

static int mode_index(const char *mode) {
    const char *modes[] = {"basis", "rotation", "residual", "mini"};
    for (int i = 0; i < 4; i++) {
        if (!strcmp(mode, modes[i])) return i;
    }
    return -1;
}

static int codebook[4][12], codebook_ready;
static uint64_t code_rng;

static uint64_t code_rand(void) {
    uint64_t x = code_rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return code_rng = x;
}

static void make_codebook(void) {
    int weights[4] = {4, 5, 6, 7}, best[4][12], best_distance = -1;
    code_rng = 0x243F6A8885A308D3ULL ^ 7ULL;
    for (int iteration = 0; iteration < 4000; iteration++) {
        int candidate[4][12];
        for (int mode = 0; mode < 4; mode++) {
            int pool[12];
            for (int i = 0; i < 12; i++) {
                candidate[mode][i] = 1;
                pool[i] = i;
            }
            for (int i = 0; i < weights[mode]; i++) {
                int j = i + (int)(code_rand() % (uint64_t)(12 - i));
                int tmp = pool[i];
                pool[i] = pool[j];
                pool[j] = tmp;
                candidate[mode][pool[i]] = -1;
            }
        }
        int distance = 99;
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                int hamming = 0;
                for (int k = 0; k < 12; k++) {
                    hamming += candidate[i][k] != candidate[j][k];
                }
                if (hamming < distance) distance = hamming;
            }
        }
        if (distance > best_distance) {
            best_distance = distance;
            memcpy(best, candidate, sizeof(best));
        }
    }
    memcpy(codebook, best, sizeof(best));
    codebook_ready = 1;
}

static int code_sign(int mode, int source) {
    if (!codebook_ready) make_codebook();
    return mode >= 0 && source >= 0 && source < 12 ? codebook[mode][source] : 0;
}

static int raw_record(FILE *f, uint64_t timestamp, double sample) {
    unsigned char bytes[16];
    for (int i = 0; i < 8; i++) bytes[i] = (unsigned char)(timestamp >> (8 * i));
    uint64_t bits;
    memcpy(&bits, &sample, 8);
    for (int i = 0; i < 8; i++) bytes[8 + i] = (unsigned char)(bits >> (8 * i));
    return fwrite(bytes, 1, 16, f) == 16 ? 0 : -1;
}

static void json_escape(FILE *f, const char *value) {
    for (; *value; value++) {
        if (*value == '"' || *value == '\\') fputc('\\', f);
        fputc(*value, f);
    }
}

static int close_sync(FILE **file) {
    if (!*file) return 0;
    int fd = fileno(*file);
    int rc = fflush(*file);
    if (!rc && fd >= 0) rc = fsync(fd);
    if (fclose(*file)) rc = -1;
    *file = NULL;
    return rc ? -1 : 0;
}

static int hash_file(const char *path, char out[65]) {
    char command[CP_PATH_MAX + 32], line[128];
    if (snprintf(command, sizeof(command), "sha256sum '%s'", path) >=
        (int)sizeof(command)) {
        return -1;
    }
    FILE *f = popen(command, "r");
    if (!f || !fgets(line, sizeof(line), f)) {
        if (f) pclose(f);
        return -1;
    }
    int rc = pclose(f);
    if (rc) return -1;
    memcpy(out, line, 64);
    out[64] = 0;
    return 0;
}

static void cpu_model(char *out, size_t size) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    char line[512];
    snprintf(out, size, "unknown");
    if (!f) return;
    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "model name", 10)) {
            char *value = strchr(line, ':');
            if (value) {
                value++;
                while (*value == ' ' || *value == '\t') value++;
                value[strcspn(value, "\r\n")] = 0;
                snprintf(out, size, "%s", value);
            }
            break;
        }
    }
    fclose(f);
}

int write_run_manifest(const char *dir, const char *session_id, const char *status) {
    const char *names[] = {
        "run.json", "session.json", "windows.jsonl", "window_results.csv",
        "raw_samples.bin", "telemetry.csv", "stdout.log", "stderr.log"
    };
    char path[CP_PATH_MAX];
    if (joinp(path, sizeof(path), dir, "run_manifest.json")) return -1;
    FILE *f = fopen(path, "wx");
    if (!f) return -1;

    fprintf(f,
            "{\n"
            "  \"schema_id\": \"CAT_CAS_PHASE6_COMBINED_RUN_MANIFEST_V1\",\n"
            "  \"session_id\": \"");
    json_escape(f, session_id);
    fprintf(f, "\",\n  \"status\": \"%s\",\n  \"files\": {\n", status);

    for (int i = 0; i < 8; i++) {
        if (joinp(path, sizeof(path), dir, names[i])) {
            fclose(f);
            return -1;
        }
        struct stat st;
        char digest[65];
        if (stat(path, &st) || !S_ISREG(st.st_mode) || hash_file(path, digest)) {
            fclose(f);
            return -1;
        }
        fprintf(f, "    \"%s\": {\"size\": %lld, \"sha256\": \"%s\"}%s\n",
                names[i], (long long)st.st_size, digest, i == 7 ? "" : ",");
    }
    fprintf(f, "  }\n}\n");
    return close_sync(&f);
}

static int copy_file(const char *source, const char *destination) {
    FILE *in = fopen(source, "rb");
    FILE *out = fopen(destination, "wbx");
    if (!in || !out) {
        if (in) fclose(in);
        if (out) fclose(out);
        return -1;
    }
    char buffer[65536];
    size_t n;
    int rc = 0;
    while ((n = fread(buffer, 1, sizeof(buffer), in))) {
        if (fwrite(buffer, 1, n, out) != n) {
            rc = -1;
            break;
        }
    }
    if (fclose(in)) rc = -1;
    if (close_sync(&out)) rc = -1;
    return rc;
}

static const char *inject(void) {
    return getenv("COMBINED_PDN_MOCK_FAIL");
}

static int injected(const char *name) {
    const char *value = inject();
    return value && !strcmp(value, name);
}

static int write_run_json(const RunnerArgs *args, const Schedule *schedule,
                          const PinState *pin, int restored, int hardware,
                          int rc, const char *reason, double tsc_hz,
                          time_t start, time_t end) {
    char path[CP_PATH_MAX], model[256];
    struct utsname system;
    uname(&system);
    cpu_model(model, sizeof(model));
    if (joinp(path, sizeof(path), args->output_dir, "run.json")) return -1;
    FILE *f = fopen(path, "wx");
    if (!f) return -1;

    fprintf(f,
            "{\n"
            "  \"schema_id\": \"CAT_CAS_PHASE6_COMBINED_RUN_V1\",\n"
            "  \"session_id\": \"%s\",\n"
            "  \"campaign_source_commit\": \"%s\",\n"
            "  \"campaign_plan_sha256\": \"%s\",\n"
            "  \"session_manifest_sha256\": \"%s\",\n"
            "  \"executor_git_commit\": \"%s\",\n"
            "  \"host_identity\": \"",
            schedule->session_id, schedule->campaign_source_commit,
            schedule->campaign_plan_sha256, schedule->session_manifest_sha256,
            args->executor_commit);
    json_escape(f, system.nodename);
    fprintf(f, "\",\n  \"kernel_identity\": \"");
    json_escape(f, system.release);
    fprintf(f, "\",\n  \"cpu_model\": \"");
    json_escape(f, model);
    fprintf(f,
            "\",\n"
            "  \"route\": \"%s\",\n"
            "  \"victim_core\": %d,\n"
            "  \"sender_core\": %d,\n"
            "  \"frequency_policy\": %ld,\n"
            "  \"tsc_calibration_hz\": %.17g,\n"
            "  \"read_rate_hz\": %ld,\n"
            "  \"slot_duration_s\": %.17g,\n"
            "  \"sender_off_duration_s\": %.17g,\n"
            "  \"temperature_veto_c\": %.17g,\n"
            "  \"start_timestamp\": %lld,\n"
            "  \"end_timestamp\": %lld,\n"
            "  \"exit_status\": \"%s\",\n"
            "  \"failure_reason\": \"%s\",\n"
            "  \"host_control_state_restored\": %s,\n"
            "  \"physical_carrier_restoration_claimed\": false,\n"
            "  \"automatic_retry\": false,\n"
            "  \"restoration_authorized\": false,\n"
            "  \"hardware_executed\": %s,\n"
            "  \"original_cpufreq_state\": {\"min_khz\": [",
            schedule->route, args->victim, args->sender, args->pin_khz, tsc_hz,
            args->read_hz, args->slot_s, args->off_window_s, args->temp_veto_c,
            (long long)start, (long long)end, rc ? "FAILED" : "COMPLETE",
            reason, restored ? "true" : "false", hardware ? "true" : "false");

    for (int core = 0; core < NCPU; core++) {
        fprintf(f, "%ld%s", pin->min[core], core + 1 < NCPU ? ", " : "");
    }
    fprintf(f, "], \"max_khz\": [");
    for (int core = 0; core < NCPU; core++) {
        fprintf(f, "%ld%s", pin->max[core], core + 1 < NCPU ? ", " : "");
    }
    fprintf(f,
            "], \"boost\": %ld},\n"
            "  \"applied_cpufreq_state\": {\"min_khz\": %ld, "
            "\"max_khz\": %ld, \"boost\": 0},\n"
            "  \"restored_cpufreq_state\": {\"verified\": %s, "
            "\"min_khz\": [",
            pin->boost, args->pin_khz, args->pin_khz,
            restored ? "true" : "false");
    for (int core = 0; core < NCPU; core++) {
        fprintf(f, "%ld%s", pin->min[core], core + 1 < NCPU ? ", " : "");
    }
    fprintf(f, "], \"max_khz\": [");
    for (int core = 0; core < NCPU; core++) {
        fprintf(f, "%ld%s", pin->max[core], core + 1 < NCPU ? ", " : "");
    }
    fprintf(f, "], \"boost\": %ld}\n}\n", pin->boost);
    return close_sync(&f);
}

int run_hardware(const RunnerArgs *args, const Schedule *schedule) {
    int mock = args->backend == BACKEND_MOCK;
    int rc = 0, restored = 0, hardware = 0;
    const char *reason = "";
    PinState pin = {0};
    Sender sender = {0};
    char path[CP_PATH_MAX], destination[CP_PATH_MAX];
    FILE *raw = NULL, *csv = NULL, *telemetry = NULL, *out = NULL, *err = NULL;
    double tsc_hz = mock ? 3214823000.0 : calibrate_tsc();
    time_t start = time(NULL);

    if (mkdir(args->output_dir, 0755)) {
        fprintf(stderr, "ERROR: mkdir output: %s\n", strerror(errno));
        return 2;
    }

    joinp(path, sizeof(path), args->output_dir, "stdout.log");
    out = fopen(path, "wx");
    joinp(path, sizeof(path), args->output_dir, "stderr.log");
    err = fopen(path, "wx");
    if (!out || !err) {
        reason = "LOG_CREATE_FAILURE";
        rc = 5;
        goto done;
    }
    fprintf(out, "ENGINEERING_HARDWARE_EXECUTION backend=%s\n",
            mock ? "mock" : "real");

    joinp(path, sizeof(path), args->session_dir, "session.json");
    joinp(destination, sizeof(destination), args->output_dir, "session.json");
    if (copy_file(path, destination)) {
        reason = "INPUT_COPY_FAILURE";
        rc = 5;
        goto done;
    }
    joinp(path, sizeof(path), args->session_dir, "windows.jsonl");
    joinp(destination, sizeof(destination), args->output_dir, "windows.jsonl");
    if (copy_file(path, destination)) {
        reason = "INPUT_COPY_FAILURE";
        rc = 5;
        goto done;
    }

    joinp(path, sizeof(path), args->output_dir, "raw_samples.bin");
    raw = fopen(path, "wbx");
    joinp(path, sizeof(path), args->output_dir, "window_results.csv");
    csv = fopen(path, "wx");
    joinp(path, sizeof(path), args->output_dir, "telemetry.csv");
    telemetry = fopen(path, "wx");
    if (!raw || !csv || !telemetry) {
        reason = "OUTPUT_CREATE_FAILURE";
        rc = 5;
        goto done;
    }

    fprintf(csv,
            "window_index,session_id,stage,block_id,family,actual_mode,"
            "declared_mode,executed_tone_order,declared_tone_order,"
            "physical_tone_index,codeword_source_index,drive_on,"
            "sender_off_required,measurement_mode,amplitude_level,theta_idx,"
            "slot_start_tsc,capture_deadline_tsc,sender_ready_tsc,"
            "sender_epoch_tsc,first_drive_tsc,receiver_epoch_tsc,"
            "first_sample_tsc,last_sample_tsc,sample_count,temp_before_c,"
            "temp_after_c,frequency_before_khz,frequency_after_khz,"
            "aperf_before,aperf_after,mperf_before,mperf_after,cofvid_before,"
            "cofvid_after,computed_I,computed_Q,magnitude,floor,raw_mean,"
            "raw_min,raw_max,sender_started,sender_stopped,"
            "sender_alive_at_capture,window_status\n");
    fprintf(telemetry,
            "window_index,temp_before_c,temp_after_c,frequency_before_khz,"
            "frequency_after_khz,aperf_before,aperf_after,mperf_before,"
            "mperf_after,cofvid_before,cofvid_after\n");

    if (injected("thermal")) {
        reason = "THERMAL_VETO";
        rc = 3;
        goto cleanup;
    }
    if (injected("cpufreq")) {
        reason = "CPUFREQ_PIN_FAILURE";
        rc = 4;
        goto cleanup;
    }

    if (mock) {
        for (int core = 0; core < NCPU; core++) {
            pin.have[core] = 1;
            pin.min[core] = 800000;
            pin.max[core] = 3200000;
        }
        pin.boost_have = 1;
        pin.boost = 1;
    } else {
        if (geteuid() != 0) {
            reason = "PERMISSION_FAILURE";
            rc = 4;
            goto cleanup;
        }
        if (locate_temp()) {
            reason = "TELEMETRY_FAILURE";
            rc = 4;
            goto cleanup;
        }
        if (snapshot(&pin)) {
            reason = "CPUFREQ_SNAPSHOT_FAILURE";
            rc = 4;
            goto cleanup;
        }
        struct sigaction action = {0};
        action.sa_handler = on_signal;
        sigaction(SIGINT, &action, NULL);
        sigaction(SIGTERM, &action, NULL);
        if (pin_frequency(&pin, args->pin_khz)) {
            reason = "CPUFREQ_PIN_FAILURE";
            rc = 4;
            goto cleanup;
        }
        if (tsc_hz < 100000000) {
            reason = "TSC_CALIBRATION_FAILURE";
            rc = 4;
            goto cleanup;
        }
    }
    hardware = !mock;

    for (size_t index = 0; index < schedule->count; index++) {
        const Window *window = &schedule->windows[index];
        double seconds = window->sender_off_required ?
                         args->off_window_s : args->slot_s;
        int cap = (int)(seconds * args->read_hz) + 32;
        if (mock && cap > 8) cap = 8;
        uint64_t *timestamps = malloc(sizeof(*timestamps) * (size_t)cap);
        double *observations = malloc(sizeof(*observations) * (size_t)cap);
        if (!timestamps || !observations) {
            free(timestamps);
            free(observations);
            reason = "OOM";
            rc = 5;
            goto cleanup;
        }

        double temp_before = mock ? 42 : temperature();
        long freq_before = mock ? args->pin_khz : cur_khz(args->victim);
        uint64_t aperf_before = 0, mperf_before = 0, aperf_after = 0, mperf_after = 0;
        uint64_t cofvid_before = 0, cofvid_after = 0;
        if (!mock &&
            (msr_read(args->victim, MSR_APERF, &aperf_before) ||
             msr_read(args->victim, MSR_MPERF, &mperf_before) ||
             msr_read(args->victim, MSR_COFVID, &cofvid_before))) {
            free(timestamps);
            free(observations);
            reason = "MSR_ACCESS_FAILURE";
            rc = 4;
            goto cleanup;
        }
        if (!isfinite(temp_before) || temp_before >= args->temp_veto_c) {
            free(timestamps);
            free(observations);
            reason = "THERMAL_VETO";
            rc = 3;
            goto cleanup;
        }

        int started = 0, stopped = 1, alive_at_capture = 0;
        uint64_t sender_ready_tsc = 0, sender_epoch_tsc = 0, first_drive_tsc = 0;
        uint64_t receiver_epoch_tsc = 0;
        double frequency = window->physical_tone_index >= 0 ?
                           tone(window->physical_tone_index) : 0;
        int sign = code_sign(mode_index(window->actual_mode),
                             window->codeword_source_index);

        if (!mock && pin_core(args->victim)) {
            free(timestamps);
            free(observations);
            reason = "RECEIVER_AFFINITY_FAILURE";
            rc = 4;
            goto cleanup;
        }

        if (window->drive_on) {
            if (injected("sender_create")) {
                free(timestamps);
                free(observations);
                reason = "SENDER_CREATION_FAILURE";
                rc = 4;
                goto cleanup;
            }
            if (mock) {
                sender.alive = 1;
                sender_ready_tsc = rdtsc_now();
            } else if (sender_arm(&sender, args->sender, frequency, tsc_hz,
                                  (window->theta_idx + (sign < 0 ? 4 : 0)) % 8,
                                  window->amplitude_level)) {
                free(timestamps);
                free(observations);
                reason = "SENDER_CREATION_FAILURE";
                rc = 4;
                goto cleanup;
            } else {
                sender_ready_tsc =
                    atomic_load_explicit(&sender.ready_tsc, memory_order_acquire);
            }
            started = 1;
            stopped = 0;
        }

        if (window->sender_off_required && sender.alive) {
            free(timestamps);
            free(observations);
            reason = "SENDER_OFF_INVARIANT_FAILURE";
            rc = 4;
            goto cleanup;
        }

        uint64_t origin = rdtsc_now() +
                          (uint64_t)((mock ? 0.000001 : START_GUARD_SECONDS) * tsc_hz);
        uint64_t deadline = origin + (uint64_t)(seconds * tsc_hz);

        if (window->drive_on) {
            if (mock) {
                sender_epoch_tsc = injected("late_sender")
                    ? origin + (uint64_t)(0.020 * tsc_hz)
                    : origin + 1;
                first_drive_tsc = sender_epoch_tsc;
            } else {
                sender_release(&sender, origin);
            }
        }

        int count;
        if (mock) {
            receiver_epoch_tsc = origin + 2;
            for (count = 0; count < cap; count++) {
                timestamps[count] = receiver_epoch_tsc + (uint64_t)count + 1;
                observations[count] = 100.0 + .01 * count;
            }
            alive_at_capture = sender.alive;
        } else {
            count = capture_at_origin(args->victim, args->read_hz, seconds, tsc_hz,
                                      origin, &receiver_epoch_tsc,
                                      timestamps, observations, cap);
            alive_at_capture = sender.alive;
            if (window->drive_on) {
                sender_epoch_tsc =
                    atomic_load_explicit(&sender.epoch_tsc, memory_order_acquire);
                first_drive_tsc =
                    atomic_load_explicit(&sender.first_drive_tsc, memory_order_acquire);
            }
        }

        if (sender.alive) {
            if (injected("sender_stop") || (!mock && sender_stop(&sender))) {
                free(timestamps);
                free(observations);
                reason = "SENDER_STOP_JOIN_FAILURE";
                rc = 4;
                goto cleanup;
            }
            sender.alive = 0;
            stopped = 1;
        }

        if (injected("capture") || count < 4) {
            free(timestamps);
            free(observations);
            reason = "SHORT_SAMPLE_ACQUISITION";
            rc = 5;
            goto cleanup;
        }
        uint64_t skew_limit = (uint64_t)(MAX_EPOCH_SKEW_SECONDS * tsc_hz);
        if (receiver_epoch_tsc < origin ||
            receiver_epoch_tsc > origin + skew_limit ||
            timestamps[0] < receiver_epoch_tsc) {
            free(timestamps);
            free(observations);
            reason = "RECEIVER_EPOCH_ALIGNMENT_FAILURE";
            rc = 5;
            goto cleanup;
        }
        if (window->drive_on &&
            (sender_ready_tsc == 0 || sender_ready_tsc >= origin ||
             sender_epoch_tsc < origin || sender_epoch_tsc > origin + skew_limit ||
             first_drive_tsc < origin || first_drive_tsc > deadline)) {
            free(timestamps);
            free(observations);
            reason = "SENDER_EPOCH_ALIGNMENT_FAILURE";
            rc = 5;
            goto cleanup;
        }
        if (window->sender_off_required && first_drive_tsc != 0) {
            free(timestamps);
            free(observations);
            reason = "SENDER_OFF_DRIVE_FAILURE";
            rc = 5;
            goto cleanup;
        }
        if (timestamps[count - 1] > deadline + (uint64_t)(.02 * tsc_hz)) {
            free(timestamps);
            free(observations);
            reason = "CAPTURE_DEADLINE_OVERFLOW";
            rc = 5;
            goto cleanup;
        }

        double i_value = NAN, q_value = NAN, magnitude = NAN, floor = NAN;
        if (!strcmp(window->measurement_mode, "lockin_and_raw_ring") &&
            window->physical_tone_index >= 0) {
            lockin(timestamps, observations, count, frequency, origin, tsc_hz,
                   &i_value, &q_value, &magnitude, &floor);
        }

        double mean = 0, minimum = observations[0], maximum = observations[0];
        for (int i = 0; i < count; i++) {
            if (injected("raw") ||
                raw_record(raw, timestamps[i], observations[i])) {
                free(timestamps);
                free(observations);
                reason = "RAW_WRITER_FAILURE";
                rc = 5;
                goto cleanup;
            }
            mean += observations[i];
            if (observations[i] < minimum) minimum = observations[i];
            if (observations[i] > maximum) maximum = observations[i];
        }
        mean /= count;

        double temp_after = mock ? 42 : temperature();
        long freq_after = mock ? args->pin_khz : cur_khz(args->victim);
        if (!mock &&
            (msr_read(args->victim, MSR_APERF, &aperf_after) ||
             msr_read(args->victim, MSR_MPERF, &mperf_after) ||
             msr_read(args->victim, MSR_COFVID, &cofvid_after))) {
            free(timestamps);
            free(observations);
            reason = "TELEMETRY_FAILURE";
            rc = 4;
            goto cleanup;
        }

        fprintf(csv,
                "%ld,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%s,%d,%d,"
                "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d,%.6f,%.6f,"
                "%ld,%ld,%llu,%llu,%llu,%llu,%d,%d,",
                window->window_index, window->session_id, window->stage,
                window->block_id, window->family, window->actual_mode,
                window->declared_mode, window->executed_tone_order,
                window->declared_tone_order, window->physical_tone_index,
                window->codeword_source_index, window->drive_on,
                window->sender_off_required, window->measurement_mode,
                window->amplitude_level, window->theta_idx,
                (unsigned long long)origin, (unsigned long long)deadline,
                (unsigned long long)sender_ready_tsc,
                (unsigned long long)sender_epoch_tsc,
                (unsigned long long)first_drive_tsc,
                (unsigned long long)receiver_epoch_tsc,
                (unsigned long long)timestamps[0],
                (unsigned long long)timestamps[count - 1], count,
                temp_before, temp_after, freq_before, freq_after,
                (unsigned long long)aperf_before,
                (unsigned long long)aperf_after,
                (unsigned long long)mperf_before,
                (unsigned long long)mperf_after,
                (int)(cofvid_before & 7), (int)(cofvid_after & 7));
        if (isfinite(i_value)) {
            fprintf(csv, "%.17g,%.17g,%.17g,%.17g,",
                    i_value, q_value, magnitude, floor);
        } else {
            fprintf(csv, "null,null,null,null,");
        }
        fprintf(csv, "%.17g,%.17g,%.17g,%d,%d,%d,OK\n",
                mean, minimum, maximum, started, stopped, alive_at_capture);

        fprintf(telemetry,
                "%ld,%.6f,%.6f,%ld,%ld,%llu,%llu,%llu,%llu,%d,%d\n",
                window->window_index, temp_before, temp_after,
                freq_before, freq_after,
                (unsigned long long)aperf_before,
                (unsigned long long)aperf_after,
                (unsigned long long)mperf_before,
                (unsigned long long)mperf_after,
                (int)(cofvid_before & 7), (int)(cofvid_after & 7));

        free(timestamps);
        free(observations);
        if (interrupted) {
            reason = "SIGNAL_ABORT";
            rc = 130;
            goto cleanup;
        }
    }

cleanup:
    if (sender.alive && sender_stop(&sender)) {
        reason = "SENDER_STOP_JOIN_FAILURE";
        rc = 4;
    }
    if (mock) {
        restored = !injected("restore");
    } else {
        restored = !pin.changed || restore(&pin) == 0;
    }
    if (!restored) {
        reason = "RESTORATION_FAILURE";
        rc = 6;
    }

done:
    if (close_sync(&raw) && rc == 0) {
        reason = "RAW_SYNC_FAILURE";
        rc = 5;
    }
    if (close_sync(&csv) && rc == 0) {
        reason = "CSV_SYNC_FAILURE";
        rc = 5;
    }
    if (close_sync(&telemetry) && rc == 0) {
        reason = "TELEMETRY_SYNC_FAILURE";
        rc = 5;
    }

    time_t end = time(NULL);
    if (write_run_json(args, schedule, &pin, restored, hardware, rc, reason,
                       tsc_hz, start, end) && rc == 0) {
        reason = "RUN_JSON_WRITE_FAILURE";
        rc = 5;
    }

    if (out) {
        fprintf(out, "exit_status=%s host_control_state_restored=%d\n",
                rc ? "FAILED" : "COMPLETE", restored);
        if (close_sync(&out) && rc == 0) rc = 5;
    }
    if (err) {
        if (rc) fprintf(err, "failure_reason=%s\n", reason);
        if (close_sync(&err) && rc == 0) rc = 5;
    }

    if (write_run_manifest(args->output_dir, schedule->session_id,
                           rc ? "FAILED" : "COMPLETE") && rc == 0) {
        rc = 5;
    }
    return rc;
}
