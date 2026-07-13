#define _GNU_SOURCE

#include "independent_window_runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define EVENT_COUNT 3
#define PATH_BUFFER 4096

typedef struct {
    const char *name;
    uint64_t config;
} PerfEventSpec;

typedef struct {
    int opened;
    int read_ok;
    int event_order_ok;
    int unmultiplexed;
    int open_errno;
    int read_errno;
    int failed_event_index;
    int cpu_before;
    int cpu_after;
    uint64_t duration_ns;
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t ids[EVENT_COUNT];
    uint64_t cycles;
    uint64_t change_to_dirty;
    uint64_t probe_dirty;
} PerfWindow;

typedef struct {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    struct {
        uint64_t value;
        uint64_t id;
    } values[EVENT_COUNT];
} PerfReadPayload;

typedef struct {
    uint8_t *a;
    uint8_t *b;
    uint64_t initial_a_digest;
    uint64_t initial_b_digest;
    uint64_t baseline_a_digest;
    uint64_t baseline_b_digest;
    uint64_t final_a_digest;
    uint64_t final_b_digest;
} TrialBanks;

typedef struct {
    const char *name;
    const char *stage_sequence[7];
    int sequence_index;
    int baseline_rc;
    int pre_sentinel_rc;
    int rebaseline_rc;
    int source_rc;
    int measure_rc;
    int restore_rc;
    int post_sentinel_rc;
    PerfWindow pre_a;
    PerfWindow pre_b;
    PerfWindow measure;
    PerfWindow post_a;
    PerfWindow post_b;
    char baseline_receipt_id[96];
    char pre_sentinel_receipt_id[96];
    char rebaseline_receipt_id[96];
    char source_receipt_id[96];
    char measure_receipt_id[96];
    char restore_receipt_id[96];
    char post_sentinel_receipt_id[96];
    uint64_t baseline_a_digest;
    uint64_t baseline_b_digest;
    uint64_t rebaseline_a_digest;
    uint64_t rebaseline_b_digest;
    uint64_t restore_a_digest;
    uint64_t restore_b_digest;
    int source_cpu_before;
    int source_cpu_after;
    int source_positive_work;
    int source_negative_work;
    int source_total_work;
    int positive_prefix_unique;
    int negative_prefix_unique;
    const char *measured_logical_role;
    const char *measured_physical_bank;
    int restoration_passed;
} ComponentReceipt;

static const PerfEventSpec PERF_EVENTS[EVENT_COUNT] = {
    {"cpu_cycles_not_halted", INDEPENDENT_WINDOW_EVENT_CYCLES_CONFIG},
    {"cache_block_commands_change_to_dirty", INDEPENDENT_WINDOW_EVENT_C2D_CONFIG},
    {"probe_responses_dirty", INDEPENDENT_WINDOW_EVENT_PROBE_DIRTY_CONFIG},
};

static const int Q_VALUES[INDEPENDENT_WINDOW_Q_COUNT] = {
    -1536, -1024, -512, 0, 512, 1024, 1536
};

static const char *STAGE_SEQUENCE[7] = {
    "receiver_baseline",
    "pre_sentinel",
    "rebaseline",
    "source_encoding",
    "measure_logical_bank",
    "restore_both_banks",
    "post_sentinel",
};

uint32_t independent_window_positive_work(int q) {
    return (uint32_t)(INDEPENDENT_WINDOW_BASE_WORK + q);
}

uint32_t independent_window_negative_work(int q) {
    return (uint32_t)(INDEPENDENT_WINDOW_BASE_WORK - q);
}

uint32_t independent_window_permuted_line(uint32_t index) {
    return (INDEPENDENT_WINDOW_PERM_A * index + INDEPENDENT_WINDOW_PERM_B) &
           (INDEPENDENT_WINDOW_BANK_LINES - 1u);
}

static long perf_event_open_call(
    struct perf_event_attr *attr,
    pid_t pid,
    int cpu,
    int group_fd,
    unsigned long flags
) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

static uint64_t fnv1a64(const uint8_t *data, size_t len) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0; index < len; index++) {
        hash ^= (uint64_t)data[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint8_t pattern_byte(size_t index) {
    return (uint8_t)((index * 131u + 17u) & 0xffu);
}

__attribute__((noinline, noclone))
static void same_value_store_prefix(uint8_t *bank, uint32_t units) {
    for (uint32_t unit = 0u; unit < units; unit++) {
        uint8_t *line = bank + independent_window_permuted_line(unit) * INDEPENDENT_WINDOW_LINE_BYTES;
        volatile uint64_t *slot = (volatile uint64_t *)(void *)line;
        uint64_t value = *slot;
        *slot = value;
    }
    __sync_synchronize();
}

__attribute__((noinline, noclone))
static void same_value_store_sentinel(uint8_t *bank) {
    static const uint32_t starts[INDEPENDENT_WINDOW_SENTINEL_SEGMENTS] = {
        0u, 1024u, 2048u, 3072u
    };
    for (uint32_t segment = 0u; segment < INDEPENDENT_WINDOW_SENTINEL_SEGMENTS; segment++) {
        for (uint32_t offset = 0u; offset < INDEPENDENT_WINDOW_SENTINEL_LINES_PER_SEGMENT; offset++) {
            uint8_t *line = bank + independent_window_permuted_line(starts[segment] + offset) * INDEPENDENT_WINDOW_LINE_BYTES;
            volatile uint64_t *slot = (volatile uint64_t *)(void *)line;
            uint64_t value = *slot;
            *slot = value;
        }
    }
    __sync_synchronize();
}

static int prefix_unique(uint32_t units) {
    unsigned char seen[INDEPENDENT_WINDOW_BANK_LINES];
    memset(seen, 0, sizeof(seen));
    if (units > INDEPENDENT_WINDOW_BANK_LINES) {
        return 0;
    }
    for (uint32_t unit = 0u; unit < units; unit++) {
        uint32_t line = independent_window_permuted_line(unit);
        if (line >= INDEPENDENT_WINDOW_BANK_LINES || seen[line] != 0u) {
            return 0;
        }
        seen[line] = 1u;
    }
    return 1;
}

static int valid_q(int q) {
    for (size_t index = 0u; index < INDEPENDENT_WINDOW_Q_COUNT; index++) {
        if (Q_VALUES[index] == q) {
            return 1;
        }
    }
    return 0;
}

static const char *q0_role_name(int q0_role) {
    if (q0_role == INDEPENDENT_WINDOW_Q0_SIGNAL) {
        return "signal";
    }
    if (q0_role == INDEPENDENT_WINDOW_Q0_NULL_BUILD) {
        return "null_build";
    }
    if (q0_role == INDEPENDENT_WINDOW_Q0_NULL_TEST) {
        return "null_test";
    }
    return "invalid";
}

static const char *source_order_name(int source_positive_first) {
    return source_positive_first ? "positive_first" : "negative_first";
}

static const char *subcapture_order_name(int positive_subcapture_first) {
    return positive_subcapture_first ? "positive_subcapture_first" : "negative_subcapture_first";
}

static int expected_q0_role(int q, int repeat_index) {
    if (q != 0) {
        return INDEPENDENT_WINDOW_Q0_SIGNAL;
    }
    if (repeat_index == 0) {
        return INDEPENDENT_WINDOW_Q0_NULL_BUILD;
    }
    if (repeat_index == 1) {
        return INDEPENDENT_WINDOW_Q0_NULL_TEST;
    }
    return -1;
}

static int parse_trial_line(const char *line, IndependentWindowTrial *trial) {
    int consumed = 0;
    int matched = sscanf(
        line,
        "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d%n",
        &trial->replicate,
        &trial->pair_index,
        &trial->leg_index,
        &trial->trial_index,
        &trial->repeat_index,
        &trial->q,
        &trial->mapping,
        &trial->mapping_order_first,
        &trial->source_positive_first,
        &trial->positive_subcapture_first,
        &trial->q0_role,
        &consumed
    );
    if (matched != 11) {
        return 0;
    }
    while (line[consumed] == ' ' || line[consumed] == '\t' ||
           line[consumed] == '\r' || line[consumed] == '\n') {
        consumed++;
    }
    return line[consumed] == '\0';
}

static int validate_trial_shape(const IndependentWindowTrial *trial) {
    if (trial->replicate < 0 || trial->replicate >= INDEPENDENT_WINDOW_REPLICATES) {
        return 0;
    }
    if (trial->pair_index < 0 ||
        trial->pair_index >= INDEPENDENT_WINDOW_PAIR_COUNT_PER_REPLICATE ||
        trial->leg_index < 0 ||
        trial->leg_index >= INDEPENDENT_WINDOW_MAPPING_COUNT ||
        trial->trial_index < 0 ||
        trial->trial_index >= INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE) {
        return 0;
    }
    if (!valid_q(trial->q) ||
        trial->mapping < 0 ||
        trial->mapping >= INDEPENDENT_WINDOW_MAPPING_COUNT ||
        trial->mapping_order_first < 0 ||
        trial->mapping_order_first >= INDEPENDENT_WINDOW_MAPPING_COUNT) {
        return 0;
    }
    if (trial->source_positive_first < 0 || trial->source_positive_first > 1 ||
        trial->positive_subcapture_first < 0 || trial->positive_subcapture_first > 1) {
        return 0;
    }
    if (trial->q0_role != expected_q0_role(trial->q, trial->repeat_index)) {
        return 0;
    }
    if (independent_window_positive_work(trial->q) +
            independent_window_negative_work(trial->q) !=
        INDEPENDENT_WINDOW_TOTAL_WORK) {
        return 0;
    }
    return 1;
}

int independent_window_validate_schedule_tsv(const char *path) {
    FILE *handle = fopen(path, "r");
    if (handle == NULL) {
        return -errno;
    }

    unsigned char trial_seen[INDEPENDENT_WINDOW_REPLICATES][INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE];
    unsigned char pair_mapping_mask[INDEPENDENT_WINDOW_REPLICATES][INDEPENDENT_WINDOW_PAIR_COUNT_PER_REPLICATE];
    int per_rep_count[INDEPENDENT_WINDOW_REPLICATES] = {0, 0};
    int q0_build_pairs[INDEPENDENT_WINDOW_REPLICATES] = {0, 0};
    int q0_test_pairs[INDEPENDENT_WINDOW_REPLICATES] = {0, 0};
    size_t count = 0u;
    char line[256];

    memset(trial_seen, 0, sizeof(trial_seen));
    memset(pair_mapping_mask, 0, sizeof(pair_mapping_mask));

    while (fgets(line, sizeof(line), handle) != NULL) {
        IndependentWindowTrial trial;
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') {
            continue;
        }
        if (count >= INDEPENDENT_WINDOW_TOTAL_TRIALS) {
            fclose(handle);
            return -E2BIG;
        }
        if (!parse_trial_line(line, &trial) || !validate_trial_shape(&trial)) {
            fclose(handle);
            return -EINVAL;
        }
        if (trial_seen[trial.replicate][trial.trial_index] != 0u) {
            fclose(handle);
            return -EEXIST;
        }
        trial_seen[trial.replicate][trial.trial_index] = 1u;
        pair_mapping_mask[trial.replicate][trial.pair_index] |=
            (unsigned char)(1u << (unsigned int)trial.mapping);
        per_rep_count[trial.replicate]++;
        if (trial.q == 0 && trial.leg_index == 0) {
            if (trial.q0_role == INDEPENDENT_WINDOW_Q0_NULL_BUILD) {
                q0_build_pairs[trial.replicate]++;
            } else if (trial.q0_role == INDEPENDENT_WINDOW_Q0_NULL_TEST) {
                q0_test_pairs[trial.replicate]++;
            }
        }
        count++;
    }
    fclose(handle);

    if (count != INDEPENDENT_WINDOW_TOTAL_TRIALS) {
        return -EMSGSIZE;
    }
    for (int rep = 0; rep < INDEPENDENT_WINDOW_REPLICATES; rep++) {
        if (per_rep_count[rep] != INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE ||
            q0_build_pairs[rep] != 4 ||
            q0_test_pairs[rep] != 4) {
            return -EDOM;
        }
        for (int pair = 0; pair < INDEPENDENT_WINDOW_PAIR_COUNT_PER_REPLICATE; pair++) {
            if (pair_mapping_mask[rep][pair] != 0x03u) {
                return -EINVAL;
            }
        }
    }
    return 0;
}

static int load_schedule(const char *path, int replicate, IndependentWindowTrial *trials, size_t limit) {
    FILE *in = fopen(path, "r");
    if (!in) {
        return -errno;
    }
    size_t count = 0u;
    char line[256];
    while (fgets(line, sizeof(line), in) != NULL) {
        IndependentWindowTrial trial;
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') {
            continue;
        }
        if (!parse_trial_line(line, &trial) || !validate_trial_shape(&trial)) {
            fclose(in);
            return -EINVAL;
        }
        if (trial.replicate == replicate) {
            if (count >= limit) {
                fclose(in);
                return -E2BIG;
            }
            trials[count++] = trial;
        }
    }
    if (ferror(in)) {
        int saved = errno;
        fclose(in);
        return -saved;
    }
    fclose(in);
    return count == INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE ? 0 : -EINVAL;
}

static void fill_perf_attr(struct perf_event_attr *attr, uint64_t config, int disabled) {
    memset(attr, 0, sizeof(*attr));
    attr->type = PERF_TYPE_RAW;
    attr->size = sizeof(*attr);
    attr->config = config;
    attr->disabled = disabled ? 1u : 0u;
    attr->exclude_kernel = 1u;
    attr->exclude_hv = 1u;
    attr->read_format = PERF_FORMAT_GROUP |
        PERF_FORMAT_TOTAL_TIME_ENABLED |
        PERF_FORMAT_TOTAL_TIME_RUNNING |
        PERF_FORMAT_ID;
}

static int open_perf_group_detailed(int fds[EVENT_COUNT], uint64_t ids[EVENT_COUNT], int *failed_event_index) {
    if (failed_event_index) {
        *failed_event_index = -1;
    }
    for (int index = 0; index < EVENT_COUNT; index++) {
        fds[index] = -1;
        ids[index] = 0u;
    }
    for (int index = 0; index < EVENT_COUNT; index++) {
        struct perf_event_attr attr;
        fill_perf_attr(&attr, PERF_EVENTS[index].config, index == 0);
        int group_fd = index == 0 ? -1 : fds[0];
        int fd = (int)perf_event_open_call(&attr, 0, -1, group_fd, PERF_FLAG_FD_CLOEXEC);
        if (fd < 0) {
            int saved = errno;
            if (failed_event_index) {
                *failed_event_index = index;
            }
            for (int close_index = 0; close_index < index; close_index++) {
                close(fds[close_index]);
            }
            return -saved;
        }
        fds[index] = fd;
        if (ioctl(fd, PERF_EVENT_IOC_ID, &ids[index]) != 0) {
            int saved = errno;
            if (failed_event_index) {
                *failed_event_index = index;
            }
            for (int close_index = 0; close_index <= index; close_index++) {
                close(fds[close_index]);
            }
            return -saved;
        }
    }
    return 0;
}

static void close_perf_group(int fds[EVENT_COUNT]) {
    for (int index = 0; index < EVENT_COUNT; index++) {
        if (fds[index] >= 0) {
            close(fds[index]);
            fds[index] = -1;
        }
    }
}

static int decode_perf_payload(const PerfReadPayload *payload, const uint64_t ids[EVENT_COUNT], PerfWindow *out) {
    if (payload->nr != EVENT_COUNT) {
        out->event_order_ok = 0;
        return -EINVAL;
    }
    out->time_enabled = payload->time_enabled;
    out->time_running = payload->time_running;
    out->unmultiplexed = payload->time_enabled > 0u && payload->time_enabled == payload->time_running;
    out->event_order_ok = 1;
    for (uint64_t index = 0u; index < EVENT_COUNT; index++) {
        out->ids[index] = payload->values[index].id;
        if (payload->values[index].id != ids[index]) {
            out->event_order_ok = 0;
        }
    }
    out->cycles = payload->values[0].value;
    out->change_to_dirty = payload->values[1].value;
    out->probe_dirty = payload->values[2].value;
    return out->event_order_ok && out->unmultiplexed ? 0 : -EIO;
}

static int measure_store_window(uint8_t *bank, uint32_t units, int sentinel_mode, PerfWindow *out) {
    int fds[EVENT_COUNT];
    uint64_t ids[EVENT_COUNT];
    int failed_index = -1;
    memset(out, 0, sizeof(*out));
    out->failed_event_index = -1;
    if (pin_to_core(INDEPENDENT_WINDOW_RECEIVER_CORE) != 0) {
        out->open_errno = errno;
        return -errno;
    }
    int open_rc = open_perf_group_detailed(fds, ids, &failed_index);
    if (open_rc != 0) {
        out->open_errno = -open_rc;
        out->failed_event_index = failed_index;
        return open_rc;
    }
    out->opened = 1;
    out->failed_event_index = failed_index;
    for (int index = 0; index < EVENT_COUNT; index++) {
        out->ids[index] = ids[index];
    }
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        close_perf_group(fds);
        out->read_errno = saved;
        return -saved;
    }
    out->cpu_before = sched_getcpu();
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        close_perf_group(fds);
        out->read_errno = saved;
        return -saved;
    }
    if (sentinel_mode) {
        same_value_store_sentinel(bank);
    } else {
        same_value_store_prefix(bank, units);
    }
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        close_perf_group(fds);
        out->read_errno = saved;
        return -saved;
    }
    uint64_t finish = monotonic_ns();
    out->cpu_after = sched_getcpu();
    out->duration_ns = finish - start;
    PerfReadPayload payload;
    memset(&payload, 0, sizeof(payload));
    ssize_t got = read(fds[0], &payload, sizeof(payload));
    close_perf_group(fds);
    size_t expected = sizeof(uint64_t) * 3u + (sizeof(uint64_t) * 2u * EVENT_COUNT);
    if (got != (ssize_t)expected) {
        out->read_errno = got < 0 ? errno : EIO;
        return got < 0 ? -errno : -EIO;
    }
    out->read_ok = 1;
    return decode_perf_payload(&payload, ids, out);
}

static int window_ok(const PerfWindow *window) {
    return window->opened &&
        window->read_ok &&
        window->event_order_ok &&
        window->unmultiplexed &&
        window->time_enabled > 0u &&
        window->time_running > 0u &&
        window->cpu_before == INDEPENDENT_WINDOW_RECEIVER_CORE &&
        window->cpu_after == INDEPENDENT_WINDOW_RECEIVER_CORE;
}

static void print_window(FILE *out, const char *prefix, const PerfWindow *window) {
    fprintf(out,
        "\"%s_opened\":%s,\"%s_read_ok\":%s,"
        "\"%s_event_order_ok\":%s,\"%s_unmultiplexed\":%s,"
        "\"%s_open_errno\":%d,\"%s_read_errno\":%d,"
        "\"%s_failed_event_index\":%d,"
        "\"%s_cpu_before\":%d,\"%s_cpu_after\":%d,"
        "\"%s_cycles\":%" PRIu64 ",\"%s_change_to_dirty\":%" PRIu64 ","
        "\"%s_probe_dirty\":%" PRIu64 ",\"%s_duration_ns\":%" PRIu64 ","
        "\"%s_time_enabled\":%" PRIu64 ",\"%s_time_running\":%" PRIu64 ","
        "\"%s_event_ids\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]",
        prefix, window->opened ? "true" : "false",
        prefix, window->read_ok ? "true" : "false",
        prefix, window->event_order_ok ? "true" : "false",
        prefix, window->unmultiplexed ? "true" : "false",
        prefix, window->open_errno,
        prefix, window->read_errno,
        prefix, window->failed_event_index,
        prefix, window->cpu_before,
        prefix, window->cpu_after,
        prefix, window->cycles,
        prefix, window->change_to_dirty,
        prefix, window->probe_dirty,
        prefix, window->duration_ns,
        prefix, window->time_enabled,
        prefix, window->time_running,
        prefix, window->ids[0],
        window->ids[1],
        window->ids[2]);
}

static int allocate_banks(TrialBanks *banks) {
    memset(banks, 0, sizeof(*banks));
    if (posix_memalign((void **)&banks->a, 4096u, INDEPENDENT_WINDOW_BANK_BYTES) != 0) {
        return -ENOMEM;
    }
    if (posix_memalign((void **)&banks->b, 4096u, INDEPENDENT_WINDOW_BANK_BYTES) != 0) {
        free(banks->a);
        memset(banks, 0, sizeof(*banks));
        return -ENOMEM;
    }
    return 0;
}

static void free_banks(TrialBanks *banks) {
    free(banks->a);
    free(banks->b);
    memset(banks, 0, sizeof(*banks));
}

static void materialize_banks(TrialBanks *banks) {
    for (size_t index = 0u; index < INDEPENDENT_WINDOW_BANK_BYTES; index++) {
        uint8_t value = pattern_byte(index);
        banks->a[index] = value;
        banks->b[index] = value;
    }
    __sync_synchronize();
    banks->initial_a_digest = fnv1a64(banks->a, INDEPENDENT_WINDOW_BANK_BYTES);
    banks->initial_b_digest = fnv1a64(banks->b, INDEPENDENT_WINDOW_BANK_BYTES);
}

static int banks_match_pattern(const TrialBanks *banks) {
    for (size_t index = 0u; index < INDEPENDENT_WINDOW_BANK_BYTES; index++) {
        uint8_t value = pattern_byte(index);
        if (banks->a[index] != value || banks->b[index] != value) {
            return 0;
        }
    }
    return 1;
}

static int receiver_full_baseline(TrialBanks *banks) {
    if (pin_to_core(INDEPENDENT_WINDOW_RECEIVER_CORE) != 0) {
        return -errno;
    }
    same_value_store_prefix(banks->a, INDEPENDENT_WINDOW_BANK_LINES);
    same_value_store_prefix(banks->b, INDEPENDENT_WINDOW_BANK_LINES);
    banks->baseline_a_digest = fnv1a64(banks->a, INDEPENDENT_WINDOW_BANK_BYTES);
    banks->baseline_b_digest = fnv1a64(banks->b, INDEPENDENT_WINDOW_BANK_BYTES);
    if (banks->initial_a_digest != banks->baseline_a_digest ||
        banks->initial_b_digest != banks->baseline_b_digest ||
        banks->baseline_a_digest != banks->baseline_b_digest) {
        return -EIO;
    }
    return 0;
}

static int restore_banks(TrialBanks *banks) {
    if (pin_to_core(INDEPENDENT_WINDOW_RECEIVER_CORE) != 0) {
        return -errno;
    }
    same_value_store_prefix(banks->a, INDEPENDENT_WINDOW_BANK_LINES);
    same_value_store_prefix(banks->b, INDEPENDENT_WINDOW_BANK_LINES);
    banks->final_a_digest = fnv1a64(banks->a, INDEPENDENT_WINDOW_BANK_BYTES);
    banks->final_b_digest = fnv1a64(banks->b, INDEPENDENT_WINDOW_BANK_BYTES);
    if (!banks_match_pattern(banks) ||
        banks->final_a_digest != banks->initial_a_digest ||
        banks->final_b_digest != banks->initial_b_digest ||
        banks->final_a_digest != banks->final_b_digest) {
        return -EIO;
    }
    return 0;
}

static int apply_source_encoding(
    uint8_t *logical_positive,
    uint8_t *logical_negative,
    int q,
    int source_positive_first,
    int *cpu_before,
    int *cpu_after
) {
    if (pin_to_core(INDEPENDENT_WINDOW_SOURCE_CORE) != 0) {
        return -errno;
    }
    *cpu_before = sched_getcpu();
    uint32_t positive_work = independent_window_positive_work(q);
    uint32_t negative_work = independent_window_negative_work(q);
    if (source_positive_first) {
        same_value_store_prefix(logical_positive, positive_work);
        same_value_store_prefix(logical_negative, negative_work);
    } else {
        same_value_store_prefix(logical_negative, negative_work);
        same_value_store_prefix(logical_positive, positive_work);
    }
    *cpu_after = sched_getcpu();
    return 0;
}

static int make_receipt_id(
    char *buffer,
    size_t size,
    const IndependentWindowTrial *trial,
    const char *component,
    const char *stage
) {
    int n = snprintf(
        buffer,
        size,
        "iwv3_r%d_p%d_t%d_l%d_%s_%s",
        trial->replicate,
        trial->pair_index,
        trial->trial_index,
        trial->leg_index,
        component,
        stage
    );
    return n > 0 && (size_t)n < size ? 0 : -ENAMETOOLONG;
}

static int emit_stage_receipt(
    FILE *stages,
    const char *receipt_id,
    const IndependentWindowTrial *trial,
    const char *component,
    int ordinal,
    const char *stage,
    int rc
) {
    fprintf(
        stages,
        "{\"schema_id\":\"%s\",\"stage_receipt_id\":\"%s\","
        "\"replicate\":%d,\"pair\":%d,\"mapping_leg\":%d,"
        "\"trial_index\":%d,\"component\":\"%s\",\"stage_ordinal\":%d,"
        "\"stage_name\":\"%s\",\"return_code\":%d,"
        "\"monotonic_timestamp_ns\":%" PRIu64 "}\n",
        INDEPENDENT_WINDOW_SCHEMA_STAGE,
        receipt_id,
        trial->replicate,
        trial->pair_index,
        trial->leg_index,
        trial->trial_index,
        component,
        ordinal,
        stage,
        rc,
        monotonic_ns()
    );
    return ferror(stages) ? -EIO : 0;
}

static int emit_source_receipt(
    FILE *sources,
    const ComponentReceipt *component,
    const IndependentWindowTrial *trial
) {
    fprintf(
        sources,
        "{\"schema_id\":\"%s\",\"source_receipt_id\":\"%s\","
        "\"replicate\":%d,\"pair\":%d,\"mapping_leg\":%d,\"trial_index\":%d,"
        "\"component\":\"%s\",\"q\":%d,"
        "\"positive_work\":%d,\"negative_work\":%d,\"total_work\":%d,"
        "\"positive_prefix_unique\":%s,\"negative_prefix_unique\":%s,"
        "\"permutation_a\":%u,\"permutation_b\":%u,"
        "\"source_cpu_before\":%d,\"source_cpu_after\":%d,"
        "\"source_order\":\"%s\"}\n",
        INDEPENDENT_WINDOW_SCHEMA_SOURCE,
        component->source_receipt_id,
        trial->replicate,
        trial->pair_index,
        trial->leg_index,
        trial->trial_index,
        component->name,
        trial->q,
        component->source_positive_work,
        component->source_negative_work,
        component->source_total_work,
        component->positive_prefix_unique ? "true" : "false",
        component->negative_prefix_unique ? "true" : "false",
        INDEPENDENT_WINDOW_PERM_A,
        INDEPENDENT_WINDOW_PERM_B,
        component->source_cpu_before,
        component->source_cpu_after,
        source_order_name(trial->source_positive_first)
    );
    return ferror(sources) ? -EIO : 0;
}

static void init_component_receipt(ComponentReceipt *component, const char *name, int sequence_index) {
    memset(component, 0, sizeof(*component));
    component->name = name;
    component->sequence_index = sequence_index;
    for (int index = 0; index < 7; index++) {
        component->stage_sequence[index] = STAGE_SEQUENCE[index];
    }
}

static int run_component(
    FILE *stages,
    FILE *sources,
    TrialBanks *banks,
    const IndependentWindowTrial *trial,
    const char *component_name,
    int sequence_index,
    uint8_t *logical_positive,
    uint8_t *logical_negative,
    uint8_t *measured_bank,
    const char *measured_physical,
    ComponentReceipt *component
) {
    init_component_receipt(component, component_name, sequence_index);
    component->measured_logical_role = component_name;
    component->measured_physical_bank = measured_physical;
    component->source_positive_work = (int)independent_window_positive_work(trial->q);
    component->source_negative_work = (int)independent_window_negative_work(trial->q);
    component->source_total_work = INDEPENDENT_WINDOW_SOURCE_WORK_PER_SUBCAPTURE;
    component->positive_prefix_unique = prefix_unique((uint32_t)component->source_positive_work);
    component->negative_prefix_unique = prefix_unique((uint32_t)component->source_negative_work);

    make_receipt_id(component->baseline_receipt_id, sizeof(component->baseline_receipt_id), trial, component_name, "baseline");
    make_receipt_id(component->pre_sentinel_receipt_id, sizeof(component->pre_sentinel_receipt_id), trial, component_name, "pre_sentinel");
    make_receipt_id(component->rebaseline_receipt_id, sizeof(component->rebaseline_receipt_id), trial, component_name, "rebaseline");
    make_receipt_id(component->source_receipt_id, sizeof(component->source_receipt_id), trial, component_name, "source");
    make_receipt_id(component->measure_receipt_id, sizeof(component->measure_receipt_id), trial, component_name, "measure");
    make_receipt_id(component->restore_receipt_id, sizeof(component->restore_receipt_id), trial, component_name, "restore");
    make_receipt_id(component->post_sentinel_receipt_id, sizeof(component->post_sentinel_receipt_id), trial, component_name, "post_sentinel");

    component->baseline_rc = receiver_full_baseline(banks);
    component->baseline_a_digest = banks->baseline_a_digest;
    component->baseline_b_digest = banks->baseline_b_digest;
    emit_stage_receipt(stages, component->baseline_receipt_id, trial, component_name, 0, STAGE_SEQUENCE[0], component->baseline_rc);

    int pre_a_rc = measure_store_window(banks->a, INDEPENDENT_WINDOW_SENTINEL_LINES, 1, &component->pre_a);
    int pre_b_rc = measure_store_window(banks->b, INDEPENDENT_WINDOW_SENTINEL_LINES, 1, &component->pre_b);
    component->pre_sentinel_rc = pre_a_rc == 0 && pre_b_rc == 0 ? 0 : 1;
    emit_stage_receipt(stages, component->pre_sentinel_receipt_id, trial, component_name, 1, STAGE_SEQUENCE[1], component->pre_sentinel_rc);

    component->rebaseline_rc = receiver_full_baseline(banks);
    component->rebaseline_a_digest = banks->baseline_a_digest;
    component->rebaseline_b_digest = banks->baseline_b_digest;
    emit_stage_receipt(stages, component->rebaseline_receipt_id, trial, component_name, 2, STAGE_SEQUENCE[2], component->rebaseline_rc);

    component->source_rc = apply_source_encoding(
        logical_positive,
        logical_negative,
        trial->q,
        trial->source_positive_first,
        &component->source_cpu_before,
        &component->source_cpu_after
    );
    emit_stage_receipt(stages, component->source_receipt_id, trial, component_name, 3, STAGE_SEQUENCE[3], component->source_rc);
    emit_source_receipt(sources, component, trial);

    component->measure_rc = measure_store_window(measured_bank, INDEPENDENT_WINDOW_BANK_LINES, 0, &component->measure);
    emit_stage_receipt(stages, component->measure_receipt_id, trial, component_name, 4, STAGE_SEQUENCE[4], component->measure_rc);

    component->restore_rc = restore_banks(banks);
    component->restore_a_digest = banks->final_a_digest;
    component->restore_b_digest = banks->final_b_digest;
    emit_stage_receipt(stages, component->restore_receipt_id, trial, component_name, 5, STAGE_SEQUENCE[5], component->restore_rc);

    int post_a_rc = measure_store_window(banks->a, INDEPENDENT_WINDOW_SENTINEL_LINES, 1, &component->post_a);
    int post_b_rc = measure_store_window(banks->b, INDEPENDENT_WINDOW_SENTINEL_LINES, 1, &component->post_b);
    component->post_sentinel_rc = post_a_rc == 0 && post_b_rc == 0 ? 0 : 1;
    emit_stage_receipt(stages, component->post_sentinel_receipt_id, trial, component_name, 6, STAGE_SEQUENCE[6], component->post_sentinel_rc);

    component->restoration_passed =
        component->baseline_rc == 0 &&
        component->pre_sentinel_rc == 0 &&
        component->rebaseline_rc == 0 &&
        component->source_rc == 0 &&
        component->measure_rc == 0 &&
        component->restore_rc == 0 &&
        component->post_sentinel_rc == 0 &&
        window_ok(&component->pre_a) &&
        window_ok(&component->pre_b) &&
        window_ok(&component->measure) &&
        window_ok(&component->post_a) &&
        window_ok(&component->post_b) &&
        component->source_cpu_before == INDEPENDENT_WINDOW_SOURCE_CORE &&
        component->source_cpu_after == INDEPENDENT_WINDOW_SOURCE_CORE &&
        component->positive_prefix_unique &&
        component->negative_prefix_unique &&
        banks_match_pattern(banks);
    return component->restoration_passed ? 0 : 1;
}

static void print_stage_sequence(FILE *out) {
    fprintf(out, "[");
    for (int index = 0; index < 7; index++) {
        fprintf(out, "%s\"%s\"", index == 0 ? "" : ",", STAGE_SEQUENCE[index]);
    }
    fprintf(out, "]");
}

static void print_component_object(FILE *out, const ComponentReceipt *component) {
    fprintf(out,
        "{\"source_receipt_id\":\"%s\",\"measured_logical_role\":\"%s\","
        "\"measured_physical_bank\":\"%s\",\"component_restoration_result\":%s,"
        "\"baseline_receipt_id\":\"%s\",\"rebaseline_receipt_id\":\"%s\","
        "\"measure_receipt_id\":\"%s\",\"restore_receipt_id\":\"%s\","
        "\"baseline_a_digest\":\"%016" PRIx64 "\",\"baseline_b_digest\":\"%016" PRIx64 "\","
        "\"restore_a_digest\":\"%016" PRIx64 "\",\"restore_b_digest\":\"%016" PRIx64 "\",",
        component->source_receipt_id,
        component->measured_logical_role,
        component->measured_physical_bank,
        component->restoration_passed ? "true" : "false",
        component->baseline_receipt_id,
        component->rebaseline_receipt_id,
        component->measure_receipt_id,
        component->restore_receipt_id,
        component->baseline_a_digest,
        component->baseline_b_digest,
        component->restore_a_digest,
        component->restore_b_digest);
    print_window(out, "raw_pmu_window", &component->measure);
    fprintf(out, "}");
}

static int run_leg(
    FILE *raw,
    FILE *sentinels,
    FILE *stages,
    FILE *sources,
    TrialBanks *banks,
    const IndependentWindowTrial *trial
) {
    uint8_t *logical_positive = trial->mapping == 0 ? banks->a : banks->b;
    uint8_t *logical_negative = trial->mapping == 0 ? banks->b : banks->a;
    const char *logical_positive_physical = trial->mapping == 0 ? "A" : "B";
    const char *logical_negative_physical = trial->mapping == 0 ? "B" : "A";

    ComponentReceipt positive;
    ComponentReceipt negative;
    int positive_seq = trial->positive_subcapture_first ? 0 : 1;
    int negative_seq = trial->positive_subcapture_first ? 1 : 0;
    int pos_rc;
    int neg_rc;

    if (trial->positive_subcapture_first) {
        pos_rc = run_component(stages, sources, banks, trial, "positive", positive_seq,
            logical_positive, logical_negative, logical_positive, logical_positive_physical, &positive);
        neg_rc = run_component(stages, sources, banks, trial, "negative", negative_seq,
            logical_positive, logical_negative, logical_negative, logical_negative_physical, &negative);
    } else {
        neg_rc = run_component(stages, sources, banks, trial, "negative", negative_seq,
            logical_positive, logical_negative, logical_negative, logical_negative_physical, &negative);
        pos_rc = run_component(stages, sources, banks, trial, "positive", positive_seq,
            logical_positive, logical_negative, logical_positive, logical_positive_physical, &positive);
    }

    int byte_compare_passed = banks_match_pattern(banks);
    int restoration_passed = positive.restoration_passed && negative.restoration_passed && byte_compare_passed;
    int mapping_leg_source_work = positive.source_total_work + negative.source_total_work;
    int trial_ok = pos_rc == 0 && neg_rc == 0 && restoration_passed &&
        mapping_leg_source_work == INDEPENDENT_WINDOW_SOURCE_WORK_PER_MAPPING_LEG &&
        positive.measure.ids[0] != negative.measure.ids[0];
    const char *first_component = trial->positive_subcapture_first ? "positive" : "negative";
    const char *second_component = trial->positive_subcapture_first ? "negative" : "positive";

    fprintf(raw, "{");
    fprintf(raw,
        "\"schema_id\":\"%s\",\"replicate_index\":%d,\"pair_index\":%d,"
        "\"leg_index\":%d,\"trial_index\":%d,\"repeat_index\":%d,"
        "\"q\":%d,\"q0_role\":\"%s\",\"mapping\":%d,\"mapping_order_first\":%d,"
        "\"bank_allocation_id\":\"rep%d_q%d_src%d_sub%d_r%d\","
        "\"source_order\":\"%s\",\"subcapture_order\":\"%s\","
        "\"subcapture_execution_order\":[\"%s\",\"%s\"],"
        "\"positive_sequence_index\":%d,\"negative_sequence_index\":%d,"
        "\"positive_stage_sequence\":",
        INDEPENDENT_WINDOW_SCHEMA_RAW,
        trial->replicate,
        trial->pair_index,
        trial->leg_index,
        trial->trial_index,
        trial->repeat_index,
        trial->q,
        q0_role_name(trial->q0_role),
        trial->mapping,
        trial->mapping_order_first,
        trial->replicate,
        trial->q,
        trial->source_positive_first ? 0 : 1,
        trial->positive_subcapture_first ? 0 : 1,
        trial->repeat_index,
        source_order_name(trial->source_positive_first),
        subcapture_order_name(trial->positive_subcapture_first),
        first_component,
        second_component,
        positive.sequence_index,
        negative.sequence_index);
    print_stage_sequence(raw);
    fprintf(raw, ",\"negative_stage_sequence\":");
    print_stage_sequence(raw);
    fprintf(raw,
        ",\"logical_positive_physical\":\"%s\",\"logical_negative_physical\":\"%s\","
        "\"line_permutation_a\":%u,\"line_permutation_b\":%u,"
        "\"positive_prefix_unique\":%s,\"negative_prefix_unique\":%s,"
        "\"positive_baseline_rc\":%d,\"positive_rebaseline_rc\":%d,"
        "\"positive_source_rc\":%d,\"positive_restore_rc\":%d,\"positive_measure_rc\":%d,"
        "\"negative_baseline_rc\":%d,\"negative_rebaseline_rc\":%d,"
        "\"negative_source_rc\":%d,\"negative_restore_rc\":%d,\"negative_measure_rc\":%d,"
        "\"positive_source_positive_work\":%d,\"positive_source_negative_work\":%d,"
        "\"positive_source_total_work\":%d,"
        "\"negative_source_positive_work\":%d,\"negative_source_negative_work\":%d,"
        "\"negative_source_total_work\":%d,"
        "\"mapping_leg_source_work\":%d,"
        "\"byte_compare_passed\":%s,\"restoration_passed\":%s,\"trial_ok\":%s,"
        "\"positive_baseline_receipt_id\":\"%s\","
        "\"positive_pre_sentinel_receipt_id\":\"%s\","
        "\"positive_rebaseline_receipt_id\":\"%s\","
        "\"positive_source_receipt_id\":\"%s\","
        "\"positive_measure_receipt_id\":\"%s\","
        "\"positive_restore_receipt_id\":\"%s\","
        "\"positive_post_sentinel_receipt_id\":\"%s\","
        "\"negative_baseline_receipt_id\":\"%s\","
        "\"negative_pre_sentinel_receipt_id\":\"%s\","
        "\"negative_rebaseline_receipt_id\":\"%s\","
        "\"negative_source_receipt_id\":\"%s\","
        "\"negative_measure_receipt_id\":\"%s\","
        "\"negative_restore_receipt_id\":\"%s\","
        "\"negative_post_sentinel_receipt_id\":\"%s\",",
        logical_positive_physical,
        logical_negative_physical,
        INDEPENDENT_WINDOW_PERM_A,
        INDEPENDENT_WINDOW_PERM_B,
        positive.positive_prefix_unique ? "true" : "false",
        positive.negative_prefix_unique ? "true" : "false",
        positive.baseline_rc,
        positive.rebaseline_rc,
        positive.source_rc,
        positive.restore_rc,
        positive.measure_rc,
        negative.baseline_rc,
        negative.rebaseline_rc,
        negative.source_rc,
        negative.restore_rc,
        negative.measure_rc,
        positive.source_positive_work,
        positive.source_negative_work,
        positive.source_total_work,
        negative.source_positive_work,
        negative.source_negative_work,
        negative.source_total_work,
        mapping_leg_source_work,
        byte_compare_passed ? "true" : "false",
        restoration_passed ? "true" : "false",
        trial_ok ? "true" : "false",
        positive.baseline_receipt_id,
        positive.pre_sentinel_receipt_id,
        positive.rebaseline_receipt_id,
        positive.source_receipt_id,
        positive.measure_receipt_id,
        positive.restore_receipt_id,
        positive.post_sentinel_receipt_id,
        negative.baseline_receipt_id,
        negative.pre_sentinel_receipt_id,
        negative.rebaseline_receipt_id,
        negative.source_receipt_id,
        negative.measure_receipt_id,
        negative.restore_receipt_id,
        negative.post_sentinel_receipt_id);
    print_window(raw, "positive_measure", &positive.measure);
    fprintf(raw, ",");
    print_window(raw, "negative_measure", &negative.measure);
    fprintf(raw, ",\"positive_subcapture\":");
    print_component_object(raw, &positive);
    fprintf(raw, ",\"negative_subcapture\":");
    print_component_object(raw, &negative);
    fprintf(raw, "}\n");

    fprintf(sentinels, "{");
    fprintf(sentinels,
        "\"schema_id\":\"%s\",\"replicate_index\":%d,\"pair_index\":%d,"
        "\"leg_index\":%d,\"trial_index\":%d,\"repeat_index\":%d,"
        "\"q\":%d,\"q0_role\":\"%s\",\"mapping\":%d,\"mapping_order_first\":%d,"
        "\"bank_allocation_id\":\"rep%d_q%d_src%d_sub%d_r%d\","
        "\"source_order\":\"%s\",\"subcapture_order\":\"%s\","
        "\"bytes_unchanged\":%s,\"byte_compare_passed\":%s,",
        INDEPENDENT_WINDOW_SCHEMA_SENTINEL,
        trial->replicate,
        trial->pair_index,
        trial->leg_index,
        trial->trial_index,
        trial->repeat_index,
        trial->q,
        q0_role_name(trial->q0_role),
        trial->mapping,
        trial->mapping_order_first,
        trial->replicate,
        trial->q,
        trial->source_positive_first ? 0 : 1,
        trial->positive_subcapture_first ? 0 : 1,
        trial->repeat_index,
        source_order_name(trial->source_positive_first),
        subcapture_order_name(trial->positive_subcapture_first),
        restoration_passed ? "true" : "false",
        byte_compare_passed ? "true" : "false");
    print_window(sentinels, "positive_pre_a", &positive.pre_a);
    fprintf(sentinels, ",");
    print_window(sentinels, "positive_pre_b", &positive.pre_b);
    fprintf(sentinels, ",");
    print_window(sentinels, "positive_post_a", &positive.post_a);
    fprintf(sentinels, ",");
    print_window(sentinels, "positive_post_b", &positive.post_b);
    fprintf(sentinels, ",");
    print_window(sentinels, "negative_pre_a", &negative.pre_a);
    fprintf(sentinels, ",");
    print_window(sentinels, "negative_pre_b", &negative.pre_b);
    fprintf(sentinels, ",");
    print_window(sentinels, "negative_post_a", &negative.post_a);
    fprintf(sentinels, ",");
    print_window(sentinels, "negative_post_b", &negative.post_b);
    fprintf(sentinels, "}\n");
    return trial_ok ? 0 : 1;
}

static int ensure_output_root(const char *output_root) {
    if (mkdir(output_root, 0700) != 0) {
        return -errno;
    }
    return 0;
}

static int join_path(char *buffer, size_t size, const char *root, const char *name) {
    int n = snprintf(buffer, size, "%s/%s", root, name);
    return n > 0 && (size_t)n < size ? 0 : -ENAMETOOLONG;
}

static FILE *open_output(const char *root, const char *name) {
    char path[PATH_BUFFER];
    if (join_path(path, sizeof(path), root, name) != 0) {
        return NULL;
    }
    return fopen(path, "wx");
}

static int run_capture(const char *schedule_path, const char *output_root, int replicate) {
    IndependentWindowTrial trials[INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE];
    int schedule_rc = load_schedule(schedule_path, replicate, trials, INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE);
    if (schedule_rc != 0) {
        fprintf(stderr, "schedule load failed: %d\n", schedule_rc);
        return 2;
    }
    int dir_rc = ensure_output_root(output_root);
    if (dir_rc != 0) {
        fprintf(stderr, "output root create failed: %d\n", dir_rc);
        return 2;
    }
    FILE *raw = open_output(output_root, "RAW_INDEPENDENT_WINDOW_CAPTURE.jsonl");
    FILE *sentinels = open_output(output_root, "INDEPENDENT_WINDOW_SENTINELS.jsonl");
    FILE *stages = open_output(output_root, "INDEPENDENT_WINDOW_STAGE_RECEIPTS.jsonl");
    FILE *sources = open_output(output_root, "INDEPENDENT_WINDOW_SOURCE_RECEIPTS.jsonl");
    if (!raw || !sentinels || !stages || !sources) {
        fprintf(stderr, "cannot create capture output files: %s\n", strerror(errno));
        if (raw) fclose(raw);
        if (sentinels) fclose(sentinels);
        if (stages) fclose(stages);
        if (sources) fclose(sources);
        return 2;
    }
    int failed_trials = 0;
    uint64_t start = monotonic_ns();
    for (int pair = 0; pair < INDEPENDENT_WINDOW_PAIR_COUNT_PER_REPLICATE; pair++) {
        IndependentWindowTrial *first = NULL;
        IndependentWindowTrial *second = NULL;
        for (size_t index = 0u; index < INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE; index++) {
            if (trials[index].pair_index == pair && trials[index].leg_index == 0) {
                first = &trials[index];
            } else if (trials[index].pair_index == pair && trials[index].leg_index == 1) {
                second = &trials[index];
            }
        }
        if (!first || !second ||
            first->q != second->q ||
            first->repeat_index != second->repeat_index ||
            first->mapping == second->mapping ||
            first->source_positive_first != second->source_positive_first ||
            first->positive_subcapture_first != second->positive_subcapture_first ||
            first->mapping_order_first != second->mapping_order_first ||
            first->mapping_order_first != first->mapping) {
            failed_trials += 2;
            continue;
        }
        TrialBanks banks;
        if (allocate_banks(&banks) != 0) {
            failed_trials += 2;
            continue;
        }
        materialize_banks(&banks);
        if (banks.initial_a_digest != banks.initial_b_digest) {
            failed_trials += 2;
            free_banks(&banks);
            continue;
        }
        if (run_leg(raw, sentinels, stages, sources, &banks, first) != 0) {
            failed_trials++;
        }
        if (run_leg(raw, sentinels, stages, sources, &banks, second) != 0) {
            failed_trials++;
        }
        free_banks(&banks);
    }
    uint64_t finish = monotonic_ns();
    fclose(raw);
    fclose(sentinels);
    fclose(stages);
    fclose(sources);

    char summary_path[PATH_BUFFER];
    if (join_path(summary_path, sizeof(summary_path), output_root, "RUNTIME_SUMMARY.json") != 0) {
        fprintf(stderr, "summary path too long\n");
        return 2;
    }
    FILE *summary = fopen(summary_path, "wx");
    if (!summary) {
        fprintf(stderr, "cannot create runtime summary: %s\n", strerror(errno));
        return 2;
    }
    fprintf(summary, "{\n");
    fprintf(summary, "  \"schema_id\": \"%s\",\n", INDEPENDENT_WINDOW_SCHEMA_RUNTIME_SUMMARY);
    fprintf(summary, "  \"run_id\": \"%s\",\n", INDEPENDENT_WINDOW_RUN_ID);
    fprintf(summary, "  \"replicate_index\": %d,\n", replicate);
    fprintf(summary, "  \"status\": \"%s\",\n",
        failed_trials == 0 ? "INDEPENDENT_WINDOW_V3_RUNTIME_COMPLETE" : "INDEPENDENT_WINDOW_V3_RUNTIME_REJECTED_WINDOWS");
    fprintf(summary, "  \"mapping_leg_record_count\": %u,\n", INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE);
    fprintf(summary, "  \"component_window_count\": %u,\n", INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE * 2u);
    fprintf(summary, "  \"failed_trial_count\": %d,\n", failed_trials);
    fprintf(summary, "  \"duration_ns\": %" PRIu64 ",\n", finish - start);
    fprintf(summary, "  \"source_core\": %d,\n", INDEPENDENT_WINDOW_SOURCE_CORE);
    fprintf(summary, "  \"receiver_core\": %d,\n", INDEPENDENT_WINDOW_RECEIVER_CORE);
    fprintf(summary, "  \"frequency_writes\": 0,\n");
    fprintf(summary, "  \"voltage_writes\": 0,\n");
    fprintf(summary, "  \"msr_reads\": 0,\n");
    fprintf(summary, "  \"msr_writes\": 0,\n");
    fprintf(summary, "  \"physical_address_access\": false,\n");
    fprintf(summary, "  \"cache_set_mapping\": false\n");
    fprintf(summary, "}\n");
    fclose(summary);
    printf("{\"status\":\"%s\",\"replicate\":%d,\"failed_trial_count\":%d}\n",
        failed_trials == 0 ? "INDEPENDENT_WINDOW_V3_RUNTIME_COMPLETE" : "INDEPENDENT_WINDOW_V3_RUNTIME_REJECTED_WINDOWS",
        replicate,
        failed_trials);
    return failed_trials == 0 ? 0 : 1;
}

static int pmu_preflight(void) {
    TrialBanks banks;
    PerfWindow window;
    int fds[EVENT_COUNT];
    uint64_t ids[EVENT_COUNT];
    int failed_event_index = -1;
    int rc = 0;
    ssize_t got = -1;
    size_t expected = sizeof(uint64_t) * 3u + (sizeof(uint64_t) * 2u * EVENT_COUNT);
    int bytes_unchanged = 0;
    const char *failure_stage = "";
    memset(&banks, 0, sizeof(banks));
    memset(&window, 0, sizeof(window));
    window.failed_event_index = -1;
    if (allocate_banks(&banks) != 0) {
        failure_stage = "allocate_buffer";
        rc = ENOMEM;
        goto done;
    }
    materialize_banks(&banks);
    if (pin_to_core(INDEPENDENT_WINDOW_RECEIVER_CORE) != 0) {
        failure_stage = "pin_receiver_core";
        rc = errno;
        goto done;
    }
    rc = open_perf_group_detailed(fds, ids, &failed_event_index);
    if (rc != 0) {
        failure_stage = "perf_event_open";
        window.open_errno = -rc;
        window.failed_event_index = failed_event_index;
        rc = -rc;
        goto done;
    }
    window.opened = 1;
    window.failed_event_index = failed_event_index;
    for (int index = 0; index < EVENT_COUNT; index++) {
        window.ids[index] = ids[index];
    }
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        failure_stage = "perf_event_reset";
        rc = errno;
        window.read_errno = rc;
        close_perf_group(fds);
        goto done;
    }
    window.cpu_before = sched_getcpu();
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        failure_stage = "perf_event_enable";
        rc = errno;
        window.read_errno = rc;
        close_perf_group(fds);
        goto done;
    }
    same_value_store_prefix(banks.a, 64u);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        failure_stage = "perf_event_disable";
        rc = errno;
        window.read_errno = rc;
        close_perf_group(fds);
        goto done;
    }
    uint64_t finish = monotonic_ns();
    window.cpu_after = sched_getcpu();
    window.duration_ns = finish - start;
    PerfReadPayload payload;
    memset(&payload, 0, sizeof(payload));
    got = read(fds[0], &payload, sizeof(payload));
    close_perf_group(fds);
    if (got != (ssize_t)expected) {
        failure_stage = "perf_event_read";
        rc = got < 0 ? errno : EIO;
        window.read_errno = rc;
        goto done;
    }
    window.read_ok = 1;
    int decode_rc = decode_perf_payload(&payload, ids, &window);
    if (decode_rc != 0) {
        failure_stage = "decode_group";
        rc = -decode_rc;
    }
done:
    if (banks.a && banks.b) {
        banks.final_a_digest = fnv1a64(banks.a, INDEPENDENT_WINDOW_BANK_BYTES);
        banks.final_b_digest = fnv1a64(banks.b, INDEPENDENT_WINDOW_BANK_BYTES);
        bytes_unchanged = banks_match_pattern(&banks) &&
            banks.initial_a_digest == banks.final_a_digest &&
            banks.initial_b_digest == banks.final_b_digest;
    }
    int passed = rc == 0 &&
        window.opened &&
        window.read_ok &&
        window.event_order_ok &&
        window.unmultiplexed &&
        window.time_enabled > 0u &&
        window.time_enabled == window.time_running &&
        window.cpu_before == INDEPENDENT_WINDOW_RECEIVER_CORE &&
        window.cpu_after == INDEPENDENT_WINDOW_RECEIVER_CORE &&
        window.cycles > 0u &&
        bytes_unchanged;
    printf("{"
        "\"schema_id\":\"%s\","
        "\"status\":\"%s\","
        "\"scientific_classification_emitted\":false,"
        "\"receiver_core\":%d,"
        "\"pid\":0,"
        "\"cpu\":-1,"
        "\"exclude_kernel\":true,"
        "\"exclude_hv\":true,"
        "\"perf_format_group\":true,"
        "\"perf_format_total_time_enabled\":true,"
        "\"perf_format_total_time_running\":true,"
        "\"perf_format_id\":true,"
        "\"event_count\":%d,"
        "\"expected_read_size\":%zu,"
        "\"actual_read_size\":%zd,"
        "\"failed_event_index\":%d,"
        "\"failure_stage\":\"%s\","
        "\"syscall_errno\":%d,"
        "\"bytes_unchanged\":%s,"
        "\"initial_a_digest\":\"%016" PRIx64 "\","
        "\"final_a_digest\":\"%016" PRIx64 "\","
        "\"initial_b_digest\":\"%016" PRIx64 "\","
        "\"final_b_digest\":\"%016" PRIx64 "\",",
        INDEPENDENT_WINDOW_SCHEMA_PMU_PREFLIGHT,
        passed ? "INDEPENDENT_WINDOW_V3_PMU_PREFLIGHT_OK" : "INDEPENDENT_WINDOW_V3_PMU_PREFLIGHT_FAILED",
        INDEPENDENT_WINDOW_RECEIVER_CORE,
        EVENT_COUNT,
        expected,
        got,
        failed_event_index,
        failure_stage,
        rc,
        bytes_unchanged ? "true" : "false",
        banks.initial_a_digest,
        banks.final_a_digest,
        banks.initial_b_digest,
        banks.final_b_digest);
    printf("\"events\":[");
    for (int index = 0; index < EVENT_COUNT; index++) {
        printf("%s{\"index\":%d,\"name\":\"%s\",\"config\":%" PRIu64 ",\"id\":%" PRIu64 "}",
            index == 0 ? "" : ",",
            index,
            PERF_EVENTS[index].name,
            PERF_EVENTS[index].config,
            window.ids[index]);
    }
    printf("],");
    print_window(stdout, "preflight", &window);
    printf("}\n");
    free_banks(&banks);
    return passed ? 0 : 1;
}

static int runtime_self_test(void) {
    if (INDEPENDENT_WINDOW_EVENT_CYCLES_CONFIG != 0x0076ull) return 1;
    if (INDEPENDENT_WINDOW_EVENT_C2D_CONFIG != 0x20eaull) return 1;
    if (INDEPENDENT_WINDOW_EVENT_PROBE_DIRTY_CONFIG != 0x0cecull) return 1;
    if (!prefix_unique(512u)) return 1;
    if (!prefix_unique(2048u)) return 1;
    if (!prefix_unique(3584u)) return 1;
    if (prefix_unique(4097u)) return 1;
    for (size_t index = 0u; index < INDEPENDENT_WINDOW_Q_COUNT; index++) {
        int q = Q_VALUES[index];
        int positive = INDEPENDENT_WINDOW_BASE_WORK + q;
        int negative = INDEPENDENT_WINDOW_BASE_WORK - q;
        if (positive + negative != INDEPENDENT_WINDOW_TOTAL_WORK) return 1;
        if (positive <= 0 || negative <= 0) return 1;
        if (positive >= (int)INDEPENDENT_WINDOW_BANK_LINES) return 1;
        if (negative >= (int)INDEPENDENT_WINDOW_BANK_LINES) return 1;
    }
    if (INDEPENDENT_WINDOW_SOURCE_WORK_PER_MAPPING_LEG !=
        2u * INDEPENDENT_WINDOW_SOURCE_WORK_PER_SUBCAPTURE) return 1;
    if (INDEPENDENT_WINDOW_TOTAL_TRIALS * 2 != INDEPENDENT_WINDOW_TOTAL_COMPONENT_WINDOWS) return 1;
    if (INDEPENDENT_WINDOW_SOURCE_CORE != 4 || INDEPENDENT_WINDOW_RECEIVER_CORE != 5) return 1;
    return 0;
}

static void print_contract_summary(void) {
    printf(
        "{\"schema_id\":\"%s\","
        "\"run_id\":\"%s\","
        "\"bank_lines\":%u,"
        "\"line_bytes\":%u,"
        "\"source_core\":%d,"
        "\"receiver_core\":%d,"
        "\"mapping_leg_source_work\":%u,"
        "\"component_windows\":%d,"
        "\"pmu_preflight\":true,"
        "\"capture_mode\":true}\n",
        INDEPENDENT_WINDOW_SCHEMA_RUNTIME_SUMMARY,
        INDEPENDENT_WINDOW_RUN_ID,
        INDEPENDENT_WINDOW_BANK_LINES,
        INDEPENDENT_WINDOW_LINE_BYTES,
        INDEPENDENT_WINDOW_SOURCE_CORE,
        INDEPENDENT_WINDOW_RECEIVER_CORE,
        INDEPENDENT_WINDOW_SOURCE_WORK_PER_MAPPING_LEG,
        INDEPENDENT_WINDOW_TOTAL_COMPONENT_WINDOWS
    );
}

static void usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s --self-test | --contract-summary | --validate-schedule-tsv <path> | --pmu-preflight | --capture --schedule-tsv <path> --output-root <path> --replicate <0|1>\n",
        program
    );
}

int main(int argc, char **argv) {
    const char *schedule_path = NULL;
    const char *output_root = NULL;
    int replicate = -1;
    int capture = 0;

    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        if (runtime_self_test() != 0) {
            fprintf(stderr, "INDEPENDENT_WINDOW_V3_RUNTIME_SELF_TEST_FAILED\n");
            return 1;
        }
        printf("INDEPENDENT_WINDOW_V3_RUNTIME_SELF_TEST_OK\n");
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--contract-summary") == 0) {
        print_contract_summary();
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--pmu-preflight") == 0) {
        return pmu_preflight();
    }
    if (argc == 3 && strcmp(argv[1], "--validate-schedule-tsv") == 0) {
        int rc = independent_window_validate_schedule_tsv(argv[2]);
        if (rc != 0) {
            fprintf(stderr, "INDEPENDENT_WINDOW_V3_SCHEDULE_TSV_INVALID rc=%d\n", rc);
            return 1;
        }
        printf("INDEPENDENT_WINDOW_V3_SCHEDULE_TSV_OK\n");
        return 0;
    }
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--capture") == 0) {
            capture = 1;
        } else if (strcmp(argv[index], "--schedule-tsv") == 0 && index + 1 < argc) {
            schedule_path = argv[++index];
        } else if (strcmp(argv[index], "--output-root") == 0 && index + 1 < argc) {
            output_root = argv[++index];
        } else if (strcmp(argv[index], "--replicate") == 0 && index + 1 < argc) {
            replicate = atoi(argv[++index]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (capture && schedule_path && output_root && (replicate == 0 || replicate == 1)) {
        return run_capture(schedule_path, output_root, replicate);
    }
    usage(argv[0]);
    return 2;
}
