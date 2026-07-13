#define _GNU_SOURCE

#include "confirmation_v2_runtime.h"

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
    uint64_t config;
} PerfEventSpec;

typedef struct {
    int opened;
    int read_ok;
    int event_order_ok;
    int unmultiplexed;
    int open_errno;
    int read_errno;
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

static const PerfEventSpec PERF_EVENTS[EVENT_COUNT] = {
    {"cpu_cycles_not_halted", BALANCED_TRANSDUCER_EVENT_CYCLES_CONFIG},
    {"cache_block_commands_change_to_dirty", BALANCED_TRANSDUCER_EVENT_C2D_CONFIG},
    {"probe_responses_dirty", BALANCED_TRANSDUCER_EVENT_PROBE_DIRTY_CONFIG},
};

static const int Q_VALUES[BALANCED_TRANSDUCER_Q_COUNT] = {
    -1536, -1024, -512, 0, 512, 1024, 1536
};

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

static size_t permuted_line(uint32_t index) {
    return (size_t)((BALANCED_TRANSDUCER_PERM_A * index + BALANCED_TRANSDUCER_PERM_B) &
                    (BALANCED_TRANSDUCER_BANK_LINES - 1u));
}

__attribute__((noinline, noclone))
static void same_value_store_prefix(uint8_t *bank, uint32_t units) {
    for (uint32_t unit = 0; unit < units; unit++) {
        uint8_t *line = bank + permuted_line(unit) * BALANCED_TRANSDUCER_LINE_BYTES;
        volatile uint64_t *slot = (volatile uint64_t *)(void *)line;
        uint64_t value = *slot;
        *slot = value;
    }
    __sync_synchronize();
}

static int prefix_unique(uint32_t units) {
    unsigned char seen[BALANCED_TRANSDUCER_BANK_LINES];
    memset(seen, 0, sizeof(seen));
    for (uint32_t unit = 0; unit < units; unit++) {
        size_t line = permuted_line(unit);
        if (line >= BALANCED_TRANSDUCER_BANK_LINES || seen[line] != 0u) {
            return 0;
        }
        seen[line] = 1u;
    }
    return 1;
}

static int valid_q(int q) {
    for (size_t index = 0; index < BALANCED_TRANSDUCER_Q_COUNT; index++) {
        if (Q_VALUES[index] == q) {
            return 1;
        }
    }
    return 0;
}

static int allocate_banks(TrialBanks *banks) {
    memset(banks, 0, sizeof(*banks));
    if (posix_memalign((void **)&banks->a, 4096u, BALANCED_TRANSDUCER_BANK_BYTES) != 0) {
        return -1;
    }
    if (posix_memalign((void **)&banks->b, 4096u, BALANCED_TRANSDUCER_BANK_BYTES) != 0) {
        free(banks->a);
        memset(banks, 0, sizeof(*banks));
        return -1;
    }
    return 0;
}

static void free_banks(TrialBanks *banks) {
    free(banks->a);
    free(banks->b);
    memset(banks, 0, sizeof(*banks));
}

static void materialize_banks(TrialBanks *banks) {
    for (size_t index = 0; index < BALANCED_TRANSDUCER_BANK_BYTES; index++) {
        uint8_t value = pattern_byte(index);
        banks->a[index] = value;
        banks->b[index] = value;
    }
    __sync_synchronize();
    banks->initial_a_digest = fnv1a64(banks->a, BALANCED_TRANSDUCER_BANK_BYTES);
    banks->initial_b_digest = fnv1a64(banks->b, BALANCED_TRANSDUCER_BANK_BYTES);
}

static int banks_match_pattern(const TrialBanks *banks) {
    for (size_t index = 0; index < BALANCED_TRANSDUCER_BANK_BYTES; index++) {
        uint8_t value = pattern_byte(index);
        if (banks->a[index] != value || banks->b[index] != value) {
            return 0;
        }
    }
    return 1;
}

static int establish_receiver_baseline(TrialBanks *banks) {
    if (pin_to_core(BALANCED_TRANSDUCER_RECEIVER_CORE) != 0) {
        return -errno;
    }
    same_value_store_prefix(banks->a, BALANCED_TRANSDUCER_BANK_LINES);
    same_value_store_prefix(banks->b, BALANCED_TRANSDUCER_BANK_LINES);
    banks->baseline_a_digest = fnv1a64(banks->a, BALANCED_TRANSDUCER_BANK_BYTES);
    banks->baseline_b_digest = fnv1a64(banks->b, BALANCED_TRANSDUCER_BANK_BYTES);
    if (banks->initial_a_digest != banks->baseline_a_digest ||
        banks->initial_b_digest != banks->baseline_b_digest ||
        banks->baseline_a_digest != banks->baseline_b_digest) {
        return -EIO;
    }
    return 0;
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
        ids[index] = 0;
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

static int open_perf_group(int fds[EVENT_COUNT], uint64_t ids[EVENT_COUNT]) {
    return open_perf_group_detailed(fds, ids, NULL);
}

static void close_perf_group(int fds[EVENT_COUNT]) {
    for (int index = 0; index < EVENT_COUNT; index++) {
        if (fds[index] >= 0) {
            close(fds[index]);
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
    for (uint64_t index = 0; index < EVENT_COUNT; index++) {
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

__attribute__((noinline, noclone))
static void same_value_store_sentinel(uint8_t *bank) {
    static const uint32_t starts[BALANCED_TRANSDUCER_SENTINEL_SEGMENTS] = {
        0u, 1024u, 2048u, 3072u
    };
    for (uint32_t segment = 0; segment < BALANCED_TRANSDUCER_SENTINEL_SEGMENTS; segment++) {
        for (uint32_t offset = 0; offset < BALANCED_TRANSDUCER_SENTINEL_LINES_PER_SEGMENT; offset++) {
            uint8_t *line = bank + permuted_line(starts[segment] + offset) * BALANCED_TRANSDUCER_LINE_BYTES;
            volatile uint64_t *slot = (volatile uint64_t *)(void *)line;
            uint64_t value = *slot;
            *slot = value;
        }
    }
    __sync_synchronize();
}

static int measure_store_window(uint8_t *bank, uint32_t units, int sentinel_mode, PerfWindow *out) {
    int fds[EVENT_COUNT];
    uint64_t ids[EVENT_COUNT];
    memset(out, 0, sizeof(*out));
    if (pin_to_core(BALANCED_TRANSDUCER_RECEIVER_CORE) != 0) {
        out->open_errno = errno;
        return -errno;
    }
    int open_rc = open_perf_group(fds, ids);
    if (open_rc != 0) {
        out->open_errno = -open_rc;
        return open_rc;
    }
    out->opened = 1;
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

static int apply_source_encoding(
    uint8_t *logical_positive,
    uint8_t *logical_negative,
    int q,
    int source_positive_first
) {
    if (pin_to_core(BALANCED_TRANSDUCER_SOURCE_CORE) != 0) {
        return -errno;
    }
    uint32_t positive_work = (uint32_t)(BALANCED_TRANSDUCER_BASE_WORK + q);
    uint32_t negative_work = (uint32_t)(BALANCED_TRANSDUCER_BASE_WORK - q);
    if (source_positive_first) {
        same_value_store_prefix(logical_positive, positive_work);
        same_value_store_prefix(logical_negative, negative_work);
    } else {
        same_value_store_prefix(logical_negative, negative_work);
        same_value_store_prefix(logical_positive, positive_work);
    }
    return 0;
}

static void print_window(FILE *out, const char *prefix, const PerfWindow *window) {
    fprintf(out,
        "\"%s_opened\":%s,\"%s_read_ok\":%s,"
        "\"%s_event_order_ok\":%s,\"%s_unmultiplexed\":%s,"
        "\"%s_open_errno\":%d,\"%s_read_errno\":%d,"
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

static int window_ok(const PerfWindow *window) {
    return window->opened &&
        window->read_ok &&
        window->event_order_ok &&
        window->unmultiplexed &&
        window->time_enabled > 0u &&
        window->time_running > 0u &&
        window->cpu_before == BALANCED_TRANSDUCER_RECEIVER_CORE &&
        window->cpu_after == BALANCED_TRANSDUCER_RECEIVER_CORE;
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
    if (allocate_banks(&banks) != 0) {
        failure_stage = "allocate_buffer";
        rc = ENOMEM;
        goto done;
    }
    materialize_banks(&banks);
    if (pin_to_core(BALANCED_TRANSDUCER_RECEIVER_CORE) != 0) {
        failure_stage = "pin_receiver_core";
        rc = errno;
        goto done;
    }
    rc = open_perf_group_detailed(fds, ids, &failed_event_index);
    if (rc != 0) {
        failure_stage = "perf_event_open";
        window.open_errno = -rc;
        rc = -rc;
        goto done;
    }
    window.opened = 1;
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
        banks.final_a_digest = fnv1a64(banks.a, BALANCED_TRANSDUCER_BANK_BYTES);
        banks.final_b_digest = fnv1a64(banks.b, BALANCED_TRANSDUCER_BANK_BYTES);
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
        window.cpu_before == BALANCED_TRANSDUCER_RECEIVER_CORE &&
        window.cpu_after == BALANCED_TRANSDUCER_RECEIVER_CORE &&
        window.cycles > 0u &&
        bytes_unchanged;
    printf("{"
        "\"schema_id\":\"CAT_CAS_CONFIRMATION_V2_PMU_PREFLIGHT\","
        "\"status\":\"%s\","
        "\"scientific_classification_emitted\":false,"
        "\"receiver_core\":%d,"
        "\"pid\":0,"
        "\"cpu\":-1,"
        "\"exclude_kernel\":true,"
        "\"exclude_hv\":true,"
        "\"event_count\":%d,"
        "\"expected_read_size\":%zu,"
        "\"actual_read_size\":%zd,"
        "\"failed_event_index\":%d,"
        "\"failure_stage\":\"%s\","
        "\"syscall_errno\":%d,"
        "\"bytes_unchanged\":%s,"
        "\"initial_a_digest\":%" PRIu64 ","
        "\"final_a_digest\":%" PRIu64 ","
        "\"initial_b_digest\":%" PRIu64 ","
        "\"final_b_digest\":%" PRIu64 ",",
        passed ? "CONFIRMATION_V2_PMU_PREFLIGHT_OK" : "CONFIRMATION_V2_PMU_PREFLIGHT_FAILED",
        BALANCED_TRANSDUCER_RECEIVER_CORE,
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

static int load_schedule(const char *path, int replicate, BalancedTrial *trials, size_t limit) {
    FILE *in = fopen(path, "r");
    if (!in) {
        return -errno;
    }
    size_t count = 0;
    int row_replicate = 0;
    int pair_index = 0;
    int leg_index = 0;
    int trial_index = 0;
    int repeat_index = 0;
    int q = 0;
    int mapping = 0;
    int mapping_order_first = 0;
    int source_positive_first = 0;
    int receiver_positive_first = 0;
    while (fscanf(
        in,
        "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
        &row_replicate,
        &pair_index,
        &leg_index,
        &trial_index,
        &repeat_index,
        &q,
        &mapping,
        &mapping_order_first,
        &source_positive_first,
        &receiver_positive_first
    ) == 10) {
        if (row_replicate == replicate) {
            if (count >= limit) {
                fclose(in);
                return -E2BIG;
            }
            if (!valid_q(q) ||
                pair_index < 0 ||
                pair_index >= (int)BALANCED_TRANSDUCER_PAIR_COUNT_PER_REPLICATE ||
                (leg_index != 0 && leg_index != 1) ||
                trial_index < 0 ||
                trial_index >= (int)BALANCED_TRANSDUCER_TRIALS_PER_REPLICATE ||
                repeat_index < 0 ||
                repeat_index >= BALANCED_TRANSDUCER_TRIALS_PER_Q_MAPPING ||
                (mapping != 0 && mapping != 1) ||
                (mapping_order_first != 0 && mapping_order_first != 1) ||
                (source_positive_first != 0 && source_positive_first != 1) ||
                (receiver_positive_first != 0 && receiver_positive_first != 1)) {
                fclose(in);
                return -EINVAL;
            }
            trials[count].replicate = row_replicate;
            trials[count].pair_index = pair_index;
            trials[count].leg_index = leg_index;
            trials[count].trial_index = trial_index;
            trials[count].repeat_index = repeat_index;
            trials[count].q = q;
            trials[count].mapping = mapping;
            trials[count].mapping_order_first = mapping_order_first;
            trials[count].source_positive_first = source_positive_first;
            trials[count].receiver_positive_first = receiver_positive_first;
            count++;
        }
    }
    if (ferror(in)) {
        int saved = errno;
        fclose(in);
        return -saved;
    }
    fclose(in);
    if (count != BALANCED_TRANSDUCER_TRIALS_PER_REPLICATE) {
        return -EINVAL;
    }
    return 0;
}

static int ensure_output_root(const char *output_root) {
    if (mkdir(output_root, 0700) != 0) {
        return errno == EEXIST ? 0 : -errno;
    }
    return 0;
}

static int join_path(char *buffer, size_t size, const char *root, const char *name) {
    int n = snprintf(buffer, size, "%s/%s", root, name);
    return n > 0 && (size_t)n < size ? 0 : -ENAMETOOLONG;
}

static int run_leg(FILE *raw, FILE *sentinels, TrialBanks *banks, const BalancedTrial *trial) {
    PerfWindow pre_a;
    PerfWindow pre_b;
    PerfWindow positive;
    PerfWindow negative;
    PerfWindow post_a;
    PerfWindow post_b;
    memset(&pre_a, 0, sizeof(pre_a));
    memset(&pre_b, 0, sizeof(pre_b));
    memset(&positive, 0, sizeof(positive));
    memset(&negative, 0, sizeof(negative));
    memset(&post_a, 0, sizeof(post_a));
    memset(&post_b, 0, sizeof(post_b));

    int baseline_rc = establish_receiver_baseline(banks);
    int pre_a_rc = measure_store_window(banks->a, BALANCED_TRANSDUCER_SENTINEL_LINES, 1, &pre_a);
    int pre_b_rc = measure_store_window(banks->b, BALANCED_TRANSDUCER_SENTINEL_LINES, 1, &pre_b);

    uint8_t *logical_positive = trial->mapping == 0 ? banks->a : banks->b;
    uint8_t *logical_negative = trial->mapping == 0 ? banks->b : banks->a;
    const char *logical_positive_physical = trial->mapping == 0 ? "A" : "B";
    const char *logical_negative_physical = trial->mapping == 0 ? "B" : "A";
    int source_rc = apply_source_encoding(
        logical_positive,
        logical_negative,
        trial->q,
        trial->source_positive_first
    );

    int positive_rc = 0;
    int negative_rc = 0;
    if (trial->receiver_positive_first) {
        positive_rc = measure_store_window(logical_positive, BALANCED_TRANSDUCER_BANK_LINES, 0, &positive);
        negative_rc = measure_store_window(logical_negative, BALANCED_TRANSDUCER_BANK_LINES, 0, &negative);
    } else {
        negative_rc = measure_store_window(logical_negative, BALANCED_TRANSDUCER_BANK_LINES, 0, &negative);
        positive_rc = measure_store_window(logical_positive, BALANCED_TRANSDUCER_BANK_LINES, 0, &positive);
    }

    int post_a_rc = measure_store_window(banks->a, BALANCED_TRANSDUCER_SENTINEL_LINES, 1, &post_a);
    int post_b_rc = measure_store_window(banks->b, BALANCED_TRANSDUCER_SENTINEL_LINES, 1, &post_b);
    banks->final_a_digest = fnv1a64(banks->a, BALANCED_TRANSDUCER_BANK_BYTES);
    banks->final_b_digest = fnv1a64(banks->b, BALANCED_TRANSDUCER_BANK_BYTES);

    uint32_t positive_work = (uint32_t)(BALANCED_TRANSDUCER_BASE_WORK + trial->q);
    uint32_t negative_work = (uint32_t)(BALANCED_TRANSDUCER_BASE_WORK - trial->q);
    int positive_prefix_unique = prefix_unique(positive_work);
    int negative_prefix_unique = prefix_unique(negative_work);
    int source_total_work = (int)(positive_work + negative_work);
    int byte_compare_passed = banks_match_pattern(banks);
    int bytes_unchanged = banks->final_a_digest == banks->baseline_a_digest &&
        banks->final_b_digest == banks->baseline_b_digest &&
        banks->final_a_digest == banks->final_b_digest &&
        byte_compare_passed;
    int pmu_windows_ok = window_ok(&pre_a) && window_ok(&pre_b) &&
        window_ok(&positive) && window_ok(&negative) &&
        window_ok(&post_a) && window_ok(&post_b);
    int source_ok = source_rc == 0 &&
        source_total_work == BALANCED_TRANSDUCER_TOTAL_WORK &&
        positive_prefix_unique && negative_prefix_unique;
    int restoration_passed = baseline_rc == 0 && bytes_unchanged &&
        pre_a_rc == 0 && pre_b_rc == 0 && post_a_rc == 0 && post_b_rc == 0;
    int trial_ok = restoration_passed && pmu_windows_ok && source_ok &&
        positive_rc == 0 && negative_rc == 0;

    const PerfWindow *physical_a = trial->mapping == 0 ? &positive : &negative;
    const PerfWindow *physical_b = trial->mapping == 0 ? &negative : &positive;
    char bank_allocation_id[64];
    int bank_id_rc = snprintf(
        bank_allocation_id,
        sizeof(bank_allocation_id),
        "rep%d_q%d_src%d_recv%d",
        trial->replicate,
        trial->q,
        trial->source_positive_first ? 0 : 1,
        trial->receiver_positive_first ? 0 : 1);
    if (bank_id_rc <= 0 || (size_t)bank_id_rc >= sizeof(bank_allocation_id)) {
        return -ENAMETOOLONG;
    }

    fprintf(raw, "{");
    fprintf(raw,
        "\"schema_id\":\"%s\",\"replicate_index\":%d,\"trial_index\":%d,"
        "\"pair_index\":%d,\"leg_index\":%d,\"repeat_index\":%d,"
        "\"q\":%d,\"mapping\":%d,\"mapping_order_first\":%d,"
        "\"bank_allocation_id\":\"%s\","
        "\"source_order\":\"%s\",\"measurement_order\":\"%s\","
        "\"logical_positive_physical\":\"%s\",\"logical_negative_physical\":\"%s\","
        "\"positive_work\":%u,\"negative_work\":%u,\"source_total_work\":%d,"
        "\"line_permutation_a\":%u,\"line_permutation_b\":%u,"
        "\"positive_prefix_unique\":%s,\"negative_prefix_unique\":%s,"
        "\"baseline_rc\":%d,\"source_rc\":%d,\"positive_rc\":%d,\"negative_rc\":%d,"
        "\"byte_compare_passed\":%s,\"restoration_passed\":%s,\"pmu_windows_ok\":%s,\"trial_ok\":%s,"
        "\"initial_a_digest\":\"%016" PRIx64 "\",\"initial_b_digest\":\"%016" PRIx64 "\","
        "\"baseline_a_digest\":\"%016" PRIx64 "\",\"baseline_b_digest\":\"%016" PRIx64 "\","
        "\"final_a_digest\":\"%016" PRIx64 "\",\"final_b_digest\":\"%016" PRIx64 "\",",
        BALANCED_TRANSDUCER_SCHEMA_RAW,
        trial->replicate,
        trial->trial_index,
        trial->pair_index,
        trial->leg_index,
        trial->repeat_index,
        trial->q,
        trial->mapping,
        trial->mapping_order_first,
        bank_allocation_id,
        trial->source_positive_first ? "positive_first" : "negative_first",
        trial->receiver_positive_first ? "positive_first" : "negative_first",
        logical_positive_physical,
        logical_negative_physical,
        positive_work,
        negative_work,
        source_total_work,
        BALANCED_TRANSDUCER_PERM_A,
        BALANCED_TRANSDUCER_PERM_B,
        positive_prefix_unique ? "true" : "false",
        negative_prefix_unique ? "true" : "false",
        baseline_rc,
        source_rc,
        positive_rc,
        negative_rc,
        byte_compare_passed ? "true" : "false",
        restoration_passed ? "true" : "false",
        pmu_windows_ok ? "true" : "false",
        trial_ok ? "true" : "false",
        banks->initial_a_digest,
        banks->initial_b_digest,
        banks->baseline_a_digest,
        banks->baseline_b_digest,
        banks->final_a_digest,
        banks->final_b_digest);
    print_window(raw, "positive", &positive);
    fprintf(raw, ",");
    print_window(raw, "negative", &negative);
    fprintf(raw,
        ",\"logical_cycles_delta\":%.17g,\"logical_change_to_dirty_delta\":%.17g,"
        "\"logical_probe_dirty_delta\":%.17g,\"logical_duration_ns_delta\":%.17g,"
        "\"physical_a_cycles\":%" PRIu64 ",\"physical_b_cycles\":%" PRIu64 ","
        "\"physical_a_change_to_dirty\":%" PRIu64 ",\"physical_b_change_to_dirty\":%" PRIu64 ","
        "\"physical_a_probe_dirty\":%" PRIu64 ",\"physical_b_probe_dirty\":%" PRIu64 ","
        "\"physical_a_duration_ns\":%" PRIu64 ",\"physical_b_duration_ns\":%" PRIu64 ","
        "\"physical_a_minus_b_cycles_delta\":%.17g,"
        "\"physical_a_minus_b_change_to_dirty_delta\":%.17g,"
        "\"physical_a_minus_b_probe_dirty_delta\":%.17g,"
        "\"physical_a_minus_b_duration_ns_delta\":%.17g}\n",
        (double)positive.cycles - (double)negative.cycles,
        (double)positive.change_to_dirty - (double)negative.change_to_dirty,
        (double)positive.probe_dirty - (double)negative.probe_dirty,
        (double)positive.duration_ns - (double)negative.duration_ns,
        physical_a->cycles,
        physical_b->cycles,
        physical_a->change_to_dirty,
        physical_b->change_to_dirty,
        physical_a->probe_dirty,
        physical_b->probe_dirty,
        physical_a->duration_ns,
        physical_b->duration_ns,
        (double)physical_a->cycles - (double)physical_b->cycles,
        (double)physical_a->change_to_dirty - (double)physical_b->change_to_dirty,
        (double)physical_a->probe_dirty - (double)physical_b->probe_dirty,
        (double)physical_a->duration_ns - (double)physical_b->duration_ns);

    fprintf(sentinels, "{");
    fprintf(sentinels,
        "\"schema_id\":\"%s\",\"replicate_index\":%d,\"trial_index\":%d,"
        "\"pair_index\":%d,\"leg_index\":%d,\"repeat_index\":%d,"
        "\"q\":%d,\"mapping\":%d,\"mapping_order_first\":%d,"
        "\"bank_allocation_id\":\"%s\","
        "\"source_order\":\"%s\",\"measurement_order\":\"%s\","
        "\"pre_a_rc\":%d,\"pre_b_rc\":%d,"
        "\"post_a_rc\":%d,\"post_b_rc\":%d,\"bytes_unchanged\":%s,"
        "\"byte_compare_passed\":%s,",
        BALANCED_TRANSDUCER_SCHEMA_SENTINEL,
        trial->replicate,
        trial->trial_index,
        trial->pair_index,
        trial->leg_index,
        trial->repeat_index,
        trial->q,
        trial->mapping,
        trial->mapping_order_first,
        bank_allocation_id,
        trial->source_positive_first ? "positive_first" : "negative_first",
        trial->receiver_positive_first ? "positive_first" : "negative_first",
        pre_a_rc,
        pre_b_rc,
        post_a_rc,
        post_b_rc,
        bytes_unchanged ? "true" : "false",
        byte_compare_passed ? "true" : "false");
    print_window(sentinels, "pre_a", &pre_a);
    fprintf(sentinels, ",");
    print_window(sentinels, "pre_b", &pre_b);
    fprintf(sentinels, ",");
    print_window(sentinels, "post_a", &post_a);
    fprintf(sentinels, ",");
    print_window(sentinels, "post_b", &post_b);
    fprintf(sentinels,
        ",\"pre_a_minus_b_cycles_delta\":%.17g,"
        "\"post_a_minus_b_cycles_delta\":%.17g,"
        "\"pre_a_minus_b_change_to_dirty_delta\":%.17g,"
        "\"post_a_minus_b_change_to_dirty_delta\":%.17g,"
        "\"pre_a_minus_b_probe_dirty_delta\":%.17g,"
        "\"post_a_minus_b_probe_dirty_delta\":%.17g,"
        "\"pre_a_minus_b_duration_ns_delta\":%.17g,"
        "\"post_a_minus_b_duration_ns_delta\":%.17g}\n",
        (double)pre_a.cycles - (double)pre_b.cycles,
        (double)post_a.cycles - (double)post_b.cycles,
        (double)pre_a.change_to_dirty - (double)pre_b.change_to_dirty,
        (double)post_a.change_to_dirty - (double)post_b.change_to_dirty,
        (double)pre_a.probe_dirty - (double)pre_b.probe_dirty,
        (double)post_a.probe_dirty - (double)post_b.probe_dirty,
        (double)pre_a.duration_ns - (double)pre_b.duration_ns,
        (double)post_a.duration_ns - (double)post_b.duration_ns);
    return trial_ok ? 0 : 1;
}

static int run_capture(const char *schedule_path, const char *output_root, int replicate) {
    BalancedTrial trials[BALANCED_TRANSDUCER_TRIALS_PER_REPLICATE];
    int schedule_rc = load_schedule(
        schedule_path,
        replicate,
        trials,
        BALANCED_TRANSDUCER_TRIALS_PER_REPLICATE
    );
    if (schedule_rc != 0) {
        fprintf(stderr, "schedule load failed: %d\n", schedule_rc);
        return 2;
    }
    int dir_rc = ensure_output_root(output_root);
    if (dir_rc != 0) {
        fprintf(stderr, "output root create failed: %d\n", dir_rc);
        return 2;
    }
    char raw_path[4096];
    char sentinel_path[4096];
    char summary_path[4096];
    if (join_path(raw_path, sizeof(raw_path), output_root, "RAW_TRANSDUCER_CAPTURE.jsonl") != 0 ||
        join_path(sentinel_path, sizeof(sentinel_path), output_root, "RESTORATION_SENTINELS.jsonl") != 0 ||
        join_path(summary_path, sizeof(summary_path), output_root, "RUNTIME_SUMMARY.json") != 0) {
        fprintf(stderr, "output path too long\n");
        return 2;
    }
    FILE *raw = fopen(raw_path, "wx");
    if (!raw) {
        fprintf(stderr, "cannot create raw capture: %s\n", strerror(errno));
        return 2;
    }
    FILE *sentinels = fopen(sentinel_path, "wx");
    if (!sentinels) {
        int saved = errno;
        fclose(raw);
        fprintf(stderr, "cannot create sentinel capture: %s\n", strerror(saved));
        return 2;
    }
    int failed_trials = 0;
    uint64_t start = monotonic_ns();
    for (int pair = 0; pair < (int)BALANCED_TRANSDUCER_PAIR_COUNT_PER_REPLICATE; pair++) {
        BalancedTrial *first = NULL;
        BalancedTrial *second = NULL;
        for (size_t index = 0; index < BALANCED_TRANSDUCER_TRIALS_PER_REPLICATE; index++) {
            if (trials[index].pair_index == pair && trials[index].leg_index == 0) {
                first = &trials[index];
            }
            if (trials[index].pair_index == pair && trials[index].leg_index == 1) {
                second = &trials[index];
            }
        }
        if (!first || !second ||
            first->q != second->q ||
            first->repeat_index != second->repeat_index ||
            first->mapping == second->mapping ||
            first->source_positive_first != second->source_positive_first ||
            first->receiver_positive_first != second->receiver_positive_first ||
            first->mapping_order_first != first->mapping ||
            first->mapping_order_first != second->mapping_order_first) {
            failed_trials += 2;
            continue;
        }
        TrialBanks banks;
        int alloc_rc = allocate_banks(&banks);
        if (alloc_rc != 0) {
            failed_trials += 2;
            continue;
        }
        materialize_banks(&banks);
        if (run_leg(raw, sentinels, &banks, first) != 0) {
            failed_trials++;
        }
        if (run_leg(raw, sentinels, &banks, second) != 0) {
            failed_trials++;
        }
        free_banks(&banks);
    }
    uint64_t finish = monotonic_ns();
    fclose(raw);
    fclose(sentinels);

    FILE *summary = fopen(summary_path, "wx");
    if (!summary) {
        fprintf(stderr, "cannot create runtime summary: %s\n", strerror(errno));
        return 2;
    }
    fprintf(summary, "{\n");
    fprintf(summary, "  \"schema_id\": \"%s\",\n", BALANCED_TRANSDUCER_SCHEMA_SUMMARY);
    fprintf(summary, "  \"replicate_index\": %d,\n", replicate);
    fprintf(summary, "  \"status\": \"%s\",\n",
            failed_trials == 0 ? "CONFIRMATION_V2_RUNTIME_COMPLETE" : "CONFIRMATION_V2_RUNTIME_REJECTED_WINDOWS");
    fprintf(summary, "  \"trial_count\": %u,\n", BALANCED_TRANSDUCER_TRIALS_PER_REPLICATE);
    fprintf(summary, "  \"failed_trial_count\": %d,\n", failed_trials);
    fprintf(summary, "  \"duration_ns\": %" PRIu64 ",\n", finish - start);
    fprintf(summary, "  \"source_core\": %d,\n", BALANCED_TRANSDUCER_SOURCE_CORE);
    fprintf(summary, "  \"receiver_core\": %d,\n", BALANCED_TRANSDUCER_RECEIVER_CORE);
    fprintf(summary, "  \"bank_lines\": %u,\n", BALANCED_TRANSDUCER_BANK_LINES);
    fprintf(summary, "  \"line_bytes\": %u,\n", BALANCED_TRANSDUCER_LINE_BYTES);
    fprintf(summary, "  \"frequency_writes\": 0,\n");
    fprintf(summary, "  \"voltage_writes\": 0,\n");
    fprintf(summary, "  \"msr_reads\": 0,\n");
    fprintf(summary, "  \"msr_writes\": 0,\n");
    fprintf(summary, "  \"physical_address_access\": false,\n");
    fprintf(summary, "  \"cache_set_mapping\": false\n");
    fprintf(summary, "}\n");
    fclose(summary);
    printf("{\"status\":\"%s\",\"replicate\":%d,\"failed_trial_count\":%d}\n",
           failed_trials == 0 ? "CONFIRMATION_V2_RUNTIME_COMPLETE" : "CONFIRMATION_V2_RUNTIME_REJECTED_WINDOWS",
           replicate,
           failed_trials);
    return failed_trials == 0 ? 0 : 1;
}

static int self_test(void) {
    if (BALANCED_TRANSDUCER_EVENT_CYCLES_CONFIG != 0x76ull) return 1;
    if (BALANCED_TRANSDUCER_EVENT_C2D_CONFIG != 0x20eaull) return 1;
    if (BALANCED_TRANSDUCER_EVENT_PROBE_DIRTY_CONFIG != 0x0cecull) return 1;
    if (!prefix_unique(512u)) return 1;
    if (!prefix_unique(2048u)) return 1;
    if (!prefix_unique(3584u)) return 1;
    if (prefix_unique(4097u)) return 1;
    for (size_t index = 0; index < BALANCED_TRANSDUCER_Q_COUNT; index++) {
        int q = Q_VALUES[index];
        int positive = BALANCED_TRANSDUCER_BASE_WORK + q;
        int negative = BALANCED_TRANSDUCER_BASE_WORK - q;
        if (positive + negative != BALANCED_TRANSDUCER_TOTAL_WORK) return 1;
        if (positive <= 0 || negative <= 0) return 1;
        if (positive >= (int)BALANCED_TRANSDUCER_BANK_LINES) return 1;
        if (negative >= (int)BALANCED_TRANSDUCER_BANK_LINES) return 1;
    }
    printf("CONFIRMATION_V2_RUNTIME_SELF_TEST_OK\n");
    return 0;
}

int main(int argc, char **argv) {
    const char *schedule_path = NULL;
    const char *output_root = NULL;
    int replicate = -1;
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    }
    if (argc == 2 && strcmp(argv[1], "--pmu-preflight") == 0) {
        return pmu_preflight();
    }
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--schedule-tsv") == 0 && index + 1 < argc) {
            schedule_path = argv[++index];
        } else if (strcmp(argv[index], "--output-root") == 0 && index + 1 < argc) {
            output_root = argv[++index];
        } else if (strcmp(argv[index], "--replicate") == 0 && index + 1 < argc) {
            replicate = atoi(argv[++index]);
        } else {
            fprintf(stderr, "usage: %s --self-test | --pmu-preflight | --schedule-tsv <path> --output-root <path> --replicate <0|1>\n", argv[0]);
            return 2;
        }
    }
    if (!schedule_path || !output_root || (replicate != 0 && replicate != 1)) {
        fprintf(stderr, "usage: %s --self-test | --pmu-preflight | --schedule-tsv <path> --output-root <path> --replicate <0|1>\n", argv[0]);
        return 2;
    }
    return run_capture(schedule_path, output_root, replicate);
}
