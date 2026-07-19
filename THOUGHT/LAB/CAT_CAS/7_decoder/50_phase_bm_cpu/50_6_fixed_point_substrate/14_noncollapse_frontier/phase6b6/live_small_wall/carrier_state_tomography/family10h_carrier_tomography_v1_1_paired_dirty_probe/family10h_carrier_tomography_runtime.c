#define _GNU_SOURCE

#include "family10h_carrier_tomography_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    uint64_t value;
    uint64_t id;
} PerfValue;

typedef struct {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    PerfValue values[3];
} PerfReadout;

typedef struct {
    int cycles_fd;
    int c2d_fd;
    int probe_fd;
    uint64_t cycles_id;
    uint64_t c2d_id;
    uint64_t probe_id;
} PerfGroup;

typedef struct {
    F10CarrierState state;
    int source_cpu_before;
    int source_cpu_after;
} F10CarrierShared;

#define F10_RUNTIME_AUTH_ENV "FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_AUTHORITY"
#define F10_RUNTIME_AUTH_VALUE "family10h_carrier_tomography_v1_0"

static long perf_event_open_call(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static int open_counter(uint32_t event, uint32_t umask, int group_fd, int *fd_out, uint64_t *id_out) {
    struct perf_event_attr attr;
    long fd = -1;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_RAW;
    attr.size = sizeof(attr);
    attr.config = ((uint64_t)umask << 8) | (uint64_t)event;
    attr.disabled = group_fd < 0 ? 1 : 0;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    fd = perf_event_open_call(&attr, 0, -1, group_fd, 0ul);
    if (fd < 0) {
        return 0;
    }
    if (ioctl((int)fd, PERF_EVENT_IOC_ID, id_out) != 0) {
        close((int)fd);
        return 0;
    }
    *fd_out = (int)fd;
    return 1;
}

static int open_perf_group(PerfGroup *group) {
    memset(group, 0, sizeof(*group));
    group->cycles_fd = -1;
    group->c2d_fd = -1;
    group->probe_fd = -1;
    if (!open_counter(0x076u, 0x00u, -1, &group->cycles_fd, &group->cycles_id)) {
        return 0;
    }
    if (!open_counter(0x0eau, 0x20u, group->cycles_fd, &group->c2d_fd, &group->c2d_id)) {
        return 0;
    }
    if (!open_counter(0x0ecu, 0x0cu, group->cycles_fd, &group->probe_fd, &group->probe_id)) {
        return 0;
    }
    return 1;
}

static void close_perf_group(PerfGroup *group) {
    if (group->probe_fd >= 0) {
        close(group->probe_fd);
    }
    if (group->c2d_fd >= 0) {
        close(group->c2d_fd);
    }
    if (group->cycles_fd >= 0) {
        close(group->cycles_fd);
    }
}

static uint64_t read_value_for_id(const PerfReadout *readout, uint64_t id) {
    uint64_t i = 0;
    for (i = 0; i < readout->nr && i < 3; ++i) {
        if (readout->values[i].id == id) {
            return readout->values[i].value;
        }
    }
    return 0;
}

static int readout_has_id(const PerfReadout *readout, uint64_t id) {
    uint64_t i = 0;
    for (i = 0; i < readout->nr && i < 3; ++i) {
        if (readout->values[i].id == id) {
            return 1;
        }
    }
    return 0;
}

static int current_cpu_checked(void) {
    int cpu = sched_getcpu();
    return cpu;
}

static int pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        return 0;
    }
    return current_cpu_checked() == core;
}

static uint64_t mix_step(uint64_t acc, uint32_t index, uint32_t lane_tag) {
    uint64_t value = (uint64_t)f10_carrier_affine_line(index) + (uint64_t)lane_tag;
    acc ^= value + UINT64_C(0x9e3779b97f4a7c15) + (acc << 6) + (acc >> 2);
    return acc;
}

static volatile uint64_t f10_query_sink = 0;
static uint32_t f10_query_trace_tags[4];
static size_t f10_query_trace_len = 0U;

uint32_t f10_carrier_affine_line(uint32_t line_index) {
    return (uint32_t)(((uint64_t)F10_TOMO_AFFINE_MULTIPLIER * (uint64_t)line_index + F10_TOMO_AFFINE_OFFSET) % F10_TOMO_LINE_COUNT);
}

static uint8_t *mapped_role_lane(F10CarrierState *state, int role_is_b, int map_variant) {
    if ((role_is_b == 0 && map_variant == 0) || (role_is_b != 0 && map_variant != 0)) {
        return state->lane_a;
    }
    return state->lane_b;
}

static const uint8_t *mapped_query_lane(const F10CarrierState *state, int role_is_b, int map_variant) {
    if ((role_is_b == 0 && map_variant == 0) || (role_is_b != 0 && map_variant != 0)) {
        return state->lane_a;
    }
    return state->lane_b;
}

static void write_lane_work(uint8_t *lane, uint32_t work, uint32_t tag) {
    uint32_t i = 0;
    for (i = 0; i < work; ++i) {
        uint32_t offset = f10_carrier_affine_line(i % F10_TOMO_LINE_COUNT) * 64U;
        lane[offset] = (uint8_t)(lane[offset] + (uint8_t)(i + tag));
    }
}

static void write_role_a(F10CarrierPreparation prep, F10CarrierState *state) {
    write_lane_work(mapped_role_lane(state, 0, prep.map_variant), prep.bank_a_work, 0xA0U);
    write_lane_work(state->dummy_a, prep.dummy_a_work, 0xC0U);
}

static void write_role_b(F10CarrierPreparation prep, F10CarrierState *state) {
    write_lane_work(mapped_role_lane(state, 1, prep.map_variant), prep.bank_b_work, 0xB0U);
    write_lane_work(state->dummy_b, prep.dummy_b_work, 0xD0U);
}

int f10_carrier_prepare(F10CarrierPreparation prep, F10CarrierState *state) {
    uint32_t total_work = 0;
    if (state == NULL) {
        return 0;
    }
    total_work = prep.bank_a_work + prep.bank_b_work + prep.dummy_a_work + prep.dummy_b_work;
    if (total_work != F10_TOMO_TOTAL_WORK || prep.dummy_work != prep.dummy_a_work + prep.dummy_b_work) {
        return 0;
    }
    if (prep.map_variant != 0 && prep.map_variant != 1) {
        return 0;
    }
    if (prep.source_order_variant != 0 && prep.source_order_variant != 1) {
        return 0;
    }
    if (prep.source_order_variant == 0) {
        write_role_a(prep, state);
        write_role_b(prep, state);
    } else {
        write_role_b(prep, state);
        write_role_a(prep, state);
    }
    return 1;
}

static uint64_t query_lane(const uint8_t *lane, uint32_t tag) {
    uint64_t acc = UINT64_C(0xCBF29CE484222325) ^ (uint64_t)tag;
    uint32_t i = 0;
    if (f10_query_trace_len < sizeof(f10_query_trace_tags) / sizeof(f10_query_trace_tags[0])) {
        f10_query_trace_tags[f10_query_trace_len] = tag;
    }
    ++f10_query_trace_len;
    for (i = 0; i < F10_TOMO_LINE_COUNT; ++i) {
        uint32_t offset = f10_carrier_affine_line(i) * 64U;
        acc = mix_step(acc ^ lane[offset], i, tag);
    }
    f10_query_sink ^= acc;
    return acc;
}

static uint64_t query_pair_ordered(const uint8_t *first_lane, uint32_t first_tag, const uint8_t *second_lane, uint32_t second_tag) {
    uint64_t first = query_lane(first_lane, first_tag);
    uint64_t second = query_lane(second_lane, second_tag);
    return first ^ (second << 1);
}

static void query_trace_reset(void) {
    memset(f10_query_trace_tags, 0, sizeof(f10_query_trace_tags));
    f10_query_trace_len = 0U;
}

static int query_trace_pair_is(uint32_t first_tag, uint32_t second_tag) {
    return f10_query_trace_len == 2U && f10_query_trace_tags[0] == first_tag && f10_query_trace_tags[1] == second_tag;
}

uint64_t f10_carrier_query(const F10CarrierState *state, const char *query_name) {
    return f10_carrier_query_mapped(state, query_name, 0);
}

uint64_t f10_carrier_query_mapped(const F10CarrierState *state, const char *query_name, int map_variant) {
    if (state == NULL || query_name == NULL) {
        return 0;
    }
    if (strcmp(query_name, "query_A") == 0) {
        return query_lane(mapped_query_lane(state, 0, map_variant), 0xA0U);
    }
    if (strcmp(query_name, "query_B") == 0) {
        return query_lane(mapped_query_lane(state, 1, map_variant), 0xB0U);
    }
    if (strcmp(query_name, "query_A_then_B") == 0) {
        return query_pair_ordered(
            mapped_query_lane(state, 0, map_variant),
            0xA0U,
            mapped_query_lane(state, 1, map_variant),
            0xB0U
        );
    }
    if (strcmp(query_name, "query_B_then_A") == 0) {
        return query_pair_ordered(
            mapped_query_lane(state, 1, map_variant),
            0xB0U,
            mapped_query_lane(state, 0, map_variant),
            0xA0U
        );
    }
    if (strcmp(query_name, "query_sham") == 0) {
        return query_lane(state->sham, 0x50U);
    }
    if (strcmp(query_name, "carrier_off") == 0) {
        return query_pair_ordered(state->dummy_a, 0xC0U, state->dummy_b, 0xD0U);
    }
    return 0;
}

static int check_affine_permutation(void) {
    unsigned char seen[F10_TOMO_LINE_COUNT];
    uint32_t i = 0;
    memset(seen, 0, sizeof(seen));
    for (i = 0; i < F10_TOMO_LINE_COUNT; ++i) {
        uint32_t mapped = f10_carrier_affine_line(i);
        if (mapped >= F10_TOMO_LINE_COUNT || seen[mapped] != 0U) {
            return 0;
        }
        seen[mapped] = 1U;
    }
    return 1;
}

int f10_carrier_runtime_self_test(void) {
    static const int32_t q_values[] = {-1536, -1024, -512, 0, 512, 1024, 1536};
    size_t i = 0;
    if (!check_affine_permutation()) {
        return 0;
    }
    for (i = 0; i < sizeof(q_values) / sizeof(q_values[0]); ++i) {
        F10CarrierPreparation prep;
        F10CarrierState state;
        uint64_t a = 0;
        uint64_t b = 0;
        uint64_t ab = 0;
        uint64_t ba = 0;
        prep.q = q_values[i];
        prep.bank_a_work = (uint32_t)(F10_TOMO_M + prep.q);
        prep.bank_b_work = (uint32_t)(F10_TOMO_M - prep.q);
        prep.dummy_work = 0U;
        prep.dummy_a_work = 0U;
        prep.dummy_b_work = 0U;
        prep.source_off_control = 0;
        prep.map_variant = (int)(i % 2U);
        prep.source_order_variant = (int)(i % 2U);
        memset(&state, 0, sizeof(state));
        if (!f10_carrier_prepare(prep, &state)) {
            return 0;
        }
        a = f10_carrier_query_mapped(&state, "query_A", prep.map_variant);
        b = f10_carrier_query_mapped(&state, "query_B", prep.map_variant);
        if (a == 0U || b == 0U || a == b) {
            return 0;
        }
        query_trace_reset();
        ab = f10_carrier_query_mapped(&state, "query_A_then_B", prep.map_variant);
        if (ab == 0U || ab != (a ^ (b << 1)) || !query_trace_pair_is(0xA0U, 0xB0U)) {
            return 0;
        }
        query_trace_reset();
        ba = f10_carrier_query_mapped(&state, "query_B_then_A", prep.map_variant);
        if (ba == 0U || ba != (b ^ (a << 1)) || !query_trace_pair_is(0xB0U, 0xA0U)) {
            return 0;
        }
    }
    {
        F10CarrierPreparation off_prep;
        F10CarrierState off_state;
        off_prep.q = 0;
        off_prep.bank_a_work = 0U;
        off_prep.bank_b_work = 0U;
        off_prep.dummy_work = F10_TOMO_TOTAL_WORK;
        off_prep.dummy_a_work = F10_TOMO_M;
        off_prep.dummy_b_work = F10_TOMO_M;
        off_prep.source_off_control = 1;
        off_prep.map_variant = 1;
        off_prep.source_order_variant = 1;
        memset(&off_state, 0, sizeof(off_state));
        if (!f10_carrier_prepare(off_prep, &off_state)) {
            return 0;
        }
        if (f10_carrier_query(&off_state, "carrier_off") == 0U) {
            return 0;
        }
    }
    return 1;
}

static int source_death_roundtrip_self_test(void) {
    F10CarrierState *state = (F10CarrierState *)mmap(NULL, sizeof(F10CarrierState), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    F10CarrierPreparation prep;
    pid_t child = 0;
    int status = 0;
    int ok = 0;
    if (state == MAP_FAILED) {
        return 0;
    }
    prep.q = 512;
    prep.bank_a_work = F10_TOMO_M + 512U;
    prep.bank_b_work = F10_TOMO_M - 512U;
    prep.dummy_work = 0U;
    prep.dummy_a_work = 0U;
    prep.dummy_b_work = 0U;
    prep.source_off_control = 0;
    prep.map_variant = 0;
    prep.source_order_variant = 0;
    child = fork();
    if (child == 0) {
        (void)pin_to_core(4);
        _exit(f10_carrier_prepare(prep, state) ? 0 : 3);
    }
    if (child < 0) {
        munmap(state, sizeof(F10CarrierState));
        return 0;
    }
    if (waitpid(child, &status, 0) != child) {
        munmap(state, sizeof(F10CarrierState));
        return 0;
    }
    ok = WIFEXITED(status) && WEXITSTATUS(status) == 0 && f10_carrier_query(state, "query_A") != 0U;
    munmap(state, sizeof(F10CarrierState));
    return ok;
}

static int split_tsv(char *line, char **fields, int max_fields) {
    int count = 0;
    char *cursor = line;
    while (count < max_fields) {
        fields[count++] = cursor;
        cursor = strchr(cursor, '\t');
        if (cursor == NULL) {
            break;
        }
        *cursor = '\0';
        ++cursor;
    }
    if (count > 0) {
        char *end = fields[count - 1] + strlen(fields[count - 1]);
        while (end > fields[count - 1] && (end[-1] == '\n' || end[-1] == '\r')) {
            *--end = '\0';
        }
    }
    return count;
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int sleep_full_ns(uint64_t delay_ns) {
    struct timespec remaining;
    remaining.tv_sec = (time_t)(delay_ns / UINT64_C(1000000000));
    remaining.tv_nsec = (long)(delay_ns % UINT64_C(1000000000));
    while (nanosleep(&remaining, &remaining) != 0) {
        if (errno != EINTR) {
            return 0;
        }
    }
    return 1;
}

static int ensure_dir(const char *path) {
    if (mkdir(path, 0755) == 0) {
        return 1;
    }
    return errno == EEXIST;
}

static int runtime_authority_ready(void) {
    const char *value = getenv(F10_RUNTIME_AUTH_ENV);
    return value != NULL && strcmp(value, F10_RUNTIME_AUTH_VALUE) == 0;
}

static void prefault_state(F10CarrierState *state) {
    volatile uint8_t sink = 0;
    uint32_t i = 0;
    memset(state, 0, sizeof(*state));
    for (i = 0; i < F10_TOMO_LINE_COUNT; ++i) {
        uint32_t offset = i * 64U;
        sink ^= state->lane_a[offset];
        sink ^= state->lane_b[offset];
        sink ^= state->dummy_a[offset];
        sink ^= state->dummy_b[offset];
        sink ^= state->sham[offset];
    }
    f10_query_sink ^= sink;
}

static int execute_schedule(const char *schedule_path, const char *output_root) {
    FILE *schedule = NULL;
    FILE *measurements = NULL;
    FILE *receipts = NULL;
    char measurements_path[4096];
    char receipts_path[4096];
    char line[8192];
    int line_index = -1;
    PerfGroup group;
    int have_pmu = 0;
    if (!runtime_authority_ready()) {
        return 13;
    }
    if (!ensure_dir(output_root)) {
        return 3;
    }
    snprintf(measurements_path, sizeof(measurements_path), "%s/raw_measurements.jsonl", output_root);
    snprintf(receipts_path, sizeof(receipts_path), "%s/source_death_receipts.jsonl", output_root);
    schedule = fopen(schedule_path, "r");
    measurements = fopen(measurements_path, "w");
    receipts = fopen(receipts_path, "w");
    if (schedule == NULL || measurements == NULL || receipts == NULL) {
        return 4;
    }
    have_pmu = open_perf_group(&group);
    if (!have_pmu) {
        fclose(schedule);
        fclose(measurements);
        fclose(receipts);
        return 5;
    }
    while (fgets(line, sizeof(line), schedule) != NULL) {
        char *fields[64];
        int count = 0;
        char *tuple_id = NULL;
        int ordinal = 0;
        int q = 0;
        unsigned int bank_a = 0;
        unsigned int bank_b = 0;
        unsigned int dummy_work = 0;
        unsigned int dummy_a_work = 0;
        unsigned int dummy_b_work = 0;
        const char *query = NULL;
        uint64_t delay_ns = 0;
        uint64_t delay_tolerance_ns = 0;
        int source_cpu = 4;
        int receiver_cpu = 5;
        int map_variant = 0;
        int source_order_variant = 0;
        const char *mapping = NULL;
        const char *source_order = NULL;
        const char *map_lane_a = NULL;
        const char *map_lane_b = NULL;
        F10CarrierShared *shared = NULL;
        F10CarrierState *state = NULL;
        F10CarrierPreparation prep;
        pid_t child = 0;
        int status = 0;
        uint64_t exit_ns = 0;
        uint64_t query_ns = 0;
        PerfReadout readout;
        uint64_t duration_start = 0;
        uint64_t duration_end = 0;
        ssize_t read_size = 0;
        int prefault_cpu = -1;
        int receiver_cpu_before = -1;
        int receiver_cpu_after = -1;
        ++line_index;
        if (line_index == 0) {
            continue;
        }
        count = split_tsv(line, fields, 64);
        if (count < 38) {
            close_perf_group(&group);
            return 6;
        }
        tuple_id = fields[0];
        ordinal = atoi(fields[1]);
        q = atoi(fields[4]);
        bank_a = (unsigned int)strtoul(fields[5], NULL, 10);
        bank_b = (unsigned int)strtoul(fields[6], NULL, 10);
        dummy_work = (unsigned int)strtoul(fields[7], NULL, 10);
        dummy_a_work = (unsigned int)strtoul(fields[8], NULL, 10);
        dummy_b_work = (unsigned int)strtoul(fields[9], NULL, 10);
        mapping = fields[16];
        source_order = fields[20];
        map_lane_a = fields[25];
        map_lane_b = fields[26];
        map_variant = (strcmp(mapping, "map1") == 0) ? 1 : 0;
        source_order_variant = (strcmp(source_order, "B_then_A") == 0) ? 1 : 0;
        delay_ns = (uint64_t)strtoull(fields[15], NULL, 10);
        delay_tolerance_ns = (uint64_t)strtoull(fields[33], NULL, 10);
        source_cpu = atoi(fields[31]);
        receiver_cpu = atoi(fields[32]);
        shared = (F10CarrierShared *)mmap(NULL, sizeof(F10CarrierShared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (shared == MAP_FAILED) {
            close_perf_group(&group);
            return 7;
        }
        state = &shared->state;
        if (!pin_to_core(receiver_cpu)) {
            close_perf_group(&group);
            return 10;
        }
        prefault_cpu = current_cpu_checked();
        prefault_state(state);
        prep.q = q;
        prep.bank_a_work = bank_a;
        prep.bank_b_work = bank_b;
        prep.dummy_work = dummy_work;
        prep.dummy_a_work = dummy_a_work;
        prep.dummy_b_work = dummy_b_work;
        prep.source_off_control = (strcmp(fields[12], "true") == 0);
        prep.map_variant = map_variant;
        prep.source_order_variant = source_order_variant;
        child = fork();
        if (child == 0) {
            if (!pin_to_core(source_cpu)) {
                _exit(10);
            }
            shared->source_cpu_before = current_cpu_checked();
            if (!f10_carrier_prepare(prep, state)) {
                _exit(11);
            }
            shared->source_cpu_after = current_cpu_checked();
            _exit((shared->source_cpu_before == source_cpu && shared->source_cpu_after == source_cpu) ? 0 : 12);
        }
        if (child < 0) {
            close_perf_group(&group);
            return 8;
        }
        if (waitpid(child, &status, 0) != child) {
            close_perf_group(&group);
            return 9;
        }
        exit_ns = monotonic_ns();
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            fprintf(
                receipts,
                "{\"tuple_id\":\"%s\",\"execution_ordinal\":%d,\"source_pid\":%d,\"waitpid_pid\":%d,\"waitpid_status\":\"failed\",\"wait_status_raw\":%d,\"source_exit_monotonic_ns\":%" PRIu64 ",\"query_select_monotonic_ns\":%" PRIu64 ",\"source_alive_during_query\":false,\"source_helper_survives\":false,\"open_source_ipc_after_waitpid\":0,\"query_selected_after_waitpid\":false,\"post_observation_query_or_window_selection\":false,\"source_cpu_before\":%d,\"source_cpu_after\":%d}\n",
                tuple_id,
                ordinal,
                (int)child,
                (int)child,
                status,
                exit_ns,
                exit_ns,
                shared->source_cpu_before,
                shared->source_cpu_after
            );
            munmap(shared, sizeof(F10CarrierShared));
            close_perf_group(&group);
            fclose(schedule);
            fclose(measurements);
            fclose(receipts);
            return 14;
        }
        if (!sleep_full_ns(delay_ns)) {
            close_perf_group(&group);
            return 12;
        }
        query = fields[13];
        if (!pin_to_core(receiver_cpu)) {
            close_perf_group(&group);
            return 10;
        }
        receiver_cpu_before = current_cpu_checked();
        memset(&readout, 0, sizeof(readout));
        if (ioctl(group.cycles_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
            close_perf_group(&group);
            return 15;
        }
        if (ioctl(group.cycles_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
            close_perf_group(&group);
            return 15;
        }
        query_ns = monotonic_ns();
        if (query_ns < exit_ns + delay_ns || query_ns > exit_ns + delay_ns + delay_tolerance_ns) {
            ioctl(group.cycles_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
            close_perf_group(&group);
            return 13;
        }
        duration_start = monotonic_ns();
        (void)f10_carrier_query_mapped(state, query, map_variant);
        duration_end = monotonic_ns();
        receiver_cpu_after = current_cpu_checked();
        if (ioctl(group.cycles_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
            close_perf_group(&group);
            return 15;
        }
        read_size = read(group.cycles_fd, &readout, sizeof(readout));
        if (
            read_size != (ssize_t)sizeof(readout)
            || readout.nr != 3U
            || !readout_has_id(&readout, group.cycles_id)
            || !readout_has_id(&readout, group.c2d_id)
            || !readout_has_id(&readout, group.probe_id)
        ) {
            close_perf_group(&group);
            return 11;
        }
        fprintf(
            measurements,
            "{\"tuple_id\":\"%s\",\"execution_ordinal\":%d,\"change_to_dirty\":%" PRIu64 ",\"dirty_probe_response\":%" PRIu64 ",\"cpu_cycles\":%" PRIu64 ",\"duration_ns\":%" PRIu64 ",\"time_enabled\":%" PRIu64 ",\"time_running\":%" PRIu64 ",\"event_ids\":{\"cpu_cycles_not_halted\":%" PRIu64 ",\"cache_block_commands_change_to_dirty\":%" PRIu64 ",\"probe_responses_dirty\":%" PRIu64 "},\"pmu_read_size\":%zd,\"pmu_value_count\":%" PRIu64 ",\"source_cpu_before\":%d,\"source_cpu_after\":%d,\"prefault_cpu\":%d,\"receiver_cpu_before\":%d,\"receiver_cpu_after\":%d,\"process_custody\":\"source_dead_before_query\",\"mapping_trace\":\"%s\",\"source_order_trace\":\"%s\",\"map_lane_A_trace\":\"%s\",\"map_lane_B_trace\":\"%s\",\"query_start_monotonic_ns\":%" PRIu64 ",\"query_end_monotonic_ns\":%" PRIu64 "}\n",
            tuple_id,
            ordinal,
            read_value_for_id(&readout, group.c2d_id),
            read_value_for_id(&readout, group.probe_id),
            read_value_for_id(&readout, group.cycles_id),
            duration_end - duration_start,
            readout.time_enabled,
            readout.time_running,
            group.cycles_id,
            group.c2d_id,
            group.probe_id,
            read_size,
            readout.nr,
            shared->source_cpu_before,
            shared->source_cpu_after,
            prefault_cpu,
            receiver_cpu_before,
            receiver_cpu_after,
            mapping,
            source_order,
            map_lane_a,
            map_lane_b,
            query_ns,
            duration_end
        );
        fprintf(
            receipts,
            "{\"tuple_id\":\"%s\",\"execution_ordinal\":%d,\"source_pid\":%d,\"waitpid_pid\":%d,\"waitpid_status\":\"%s\",\"wait_status_raw\":%d,\"source_exit_monotonic_ns\":%" PRIu64 ",\"query_select_monotonic_ns\":%" PRIu64 ",\"source_alive_during_query\":false,\"source_helper_survives\":false,\"open_source_ipc_after_waitpid\":0,\"query_selected_after_waitpid\":true,\"post_observation_query_or_window_selection\":false,\"source_cpu_before\":%d,\"source_cpu_after\":%d}\n",
            tuple_id,
            ordinal,
            (int)child,
            (int)child,
            (WIFEXITED(status) && WEXITSTATUS(status) == 0) ? "exited_0" : "failed",
            status,
            exit_ns,
            query_ns,
            shared->source_cpu_before,
            shared->source_cpu_after
        );
        munmap(shared, sizeof(F10CarrierShared));
    }
    close_perf_group(&group);
    fclose(schedule);
    fclose(measurements);
    fclose(receipts);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        int passed = f10_carrier_runtime_self_test() && source_death_roundtrip_self_test();
        printf("{\"schema\":\"FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_SELF_TEST_V1\",\"passed\":%s}\n", passed ? "true" : "false");
        return passed ? 0 : 1;
    }
    if (argc == 4 && strcmp(argv[1], "--execute-schedule") == 0) {
        return execute_schedule(argv[2], argv[3]);
    }
    fprintf(stderr, "usage: %s --self-test | --execute-schedule <schedule.tsv> <output-root>\n", argv[0]);
    return 2;
}
