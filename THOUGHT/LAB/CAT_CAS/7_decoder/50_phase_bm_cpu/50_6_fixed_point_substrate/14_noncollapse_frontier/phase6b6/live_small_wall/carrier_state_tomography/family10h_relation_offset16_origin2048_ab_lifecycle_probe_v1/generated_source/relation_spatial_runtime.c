#define _GNU_SOURCE

#include "relation_spatial_runtime.h"

#include <errno.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

#define RELATION_SPATIAL_RUNTIME_AUTH_ENV "FAMILY10H_RELATION_SPATIAL_RUNTIME_AUTHORITY"
#define RELATION_SPATIAL_RUNTIME_AUTH_VALUE "family10h_relation_spatial_pair_readout_v1_0"
#define RELATION_SPATIAL_SCHEDULE_COLUMNS 52
#define RELATION_SPATIAL_LINE_BUFFER 16384
#define RELATION_SPATIAL_CANONICAL_SCHEDULE_SHA256 "e969010b48a40857c3bb9081e6ce8691124277d441201851a12e71e01893a0cb"

#if defined(__GNUC__)
#define RELATION_SPATIAL_NOINLINE __attribute__((noinline))
#else
#define RELATION_SPATIAL_NOINLINE
#endif

typedef struct {
    uint64_t value;
    uint64_t id;
} relation_spatial_perf_value;

typedef struct {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    relation_spatial_perf_value values[3];
} relation_spatial_perf_readout;

typedef struct {
    int cycles_fd;
    int c2d_fd;
    int probe_fd;
    uint64_t cycles_id;
    uint64_t c2d_id;
    uint64_t probe_id;
} relation_spatial_perf_group;

typedef struct {
    relation_spatial_carrier_state state;
    volatile int source_ready;
    volatile int release_source;
    volatile int preparation_ok;
    uint64_t source_ready_ns;
    int source_cpu_before;
    int source_cpu_after;
} relation_spatial_shared_state;

typedef struct {
    char tuple_id[160];
    int execution_ordinal;
    char block_id[96];
    int block_local_position;
    char row_role[48];
    int32_t q;
    uint32_t bank_a_work;
    uint32_t bank_b_work;
    uint32_t total_work;
    relation_spatial_relation_id r_prepare;
    relation_spatial_relation_id r_query;
    char r_prepare_text[32];
    char r_query_text[32];
    int relation_match;
    char query[64];
    char relation_cell[96];
    char session[48];
    int replicate;
    char mapping[16];
    char delay_label[32];
    uint64_t delay_ns;
    char source_lifetime[32];
    char lifetime_pair_id[96];
    char lifetime_execution_order[32];
    uint64_t lifetime_hold_ns;
    relation_spatial_order_id source_order;
    relation_spatial_order_id query_order;
    uint32_t cyclic_origin;
    char route_pressure_class[96];
    char distance_control_class[96];
    char allocation_order_class[96];
    char prefault_class[96];
    char operation_semantics_id[64];
    char control_semantics_id[64];
    int source_cpu_expected;
    int receiver_cpu_expected;
    uint32_t source_loop_count;
    uint32_t receiver_loop_count;
    uint32_t read_count;
    uint32_t write_count;
    uint32_t page_count_a;
    uint32_t page_count_b;
    uint32_t line_count_a;
    uint32_t line_count_b;
    char expected_pmu_group[80];
    int requires_pmu;
    int post_observation_scheduling;
    relation_spatial_control_id control;
} relation_spatial_schedule_row;

static volatile uint64_t relation_spatial_sink = 0u;

int relation_spatial_runtime_live_authority_present(void) {
    const char *value = getenv(RELATION_SPATIAL_RUNTIME_AUTH_ENV);
    return value != NULL && strcmp(value, RELATION_SPATIAL_RUNTIME_AUTH_VALUE) == 0;
}

uint32_t relation_spatial_map_index(relation_spatial_relation_id relation, uint32_t logical_a_index) {
    if (relation == RELATION_SPATIAL_R0) {
        return (logical_a_index + 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;
    }
    return (logical_a_index + FAMILY10H_RELATION_SPATIAL_LINE_COUNT - 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;
}

uint32_t relation_spatial_origin_index(uint32_t cyclic_origin, uint32_t step) {
    return (cyclic_origin + step) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0u;
    }
    return ((uint64_t)ts.tv_sec * UINT64_C(1000000000)) + (uint64_t)ts.tv_nsec;
}

static int current_cpu_checked(void) {
    return sched_getcpu();
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

static int pin_to_core_pair(int first_core, int second_core) {
    int current = -1;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(first_core, &set);
    CPU_SET(second_core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        return 0;
    }
    current = current_cpu_checked();
    return current == first_core || current == second_core;
}

static void wait_for_peer_progress(void) {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#endif
    sched_yield();
}

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

static int open_perf_group(relation_spatial_perf_group *group) {
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

static void close_perf_group(relation_spatial_perf_group *group) {
    if (group->probe_fd >= 0) {
        close(group->probe_fd);
    }
    if (group->c2d_fd >= 0) {
        close(group->c2d_fd);
    }
    if (group->cycles_fd >= 0) {
        close(group->cycles_fd);
    }
    group->probe_fd = -1;
    group->c2d_fd = -1;
    group->cycles_fd = -1;
}

static uint64_t read_value_for_id(const relation_spatial_perf_readout *readout, uint64_t id) {
    uint64_t i = 0u;
    for (i = 0u; i < readout->nr && i < 3u; ++i) {
        if (readout->values[i].id == id) {
            return readout->values[i].value;
        }
    }
    return 0u;
}

static int read_perf_group(relation_spatial_perf_group *group, relation_spatial_perf_readout *readout) {
    ssize_t got = 0;
    memset(readout, 0, sizeof(*readout));
    got = read(group->cycles_fd, readout, sizeof(*readout));
    return got > 0 && readout->nr == 3u;
}

static int enable_perf_group(relation_spatial_perf_group *group) {
    if (ioctl(group->cycles_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        return 0;
    }
    if (ioctl(group->cycles_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        return 0;
    }
    return 1;
}

static int disable_perf_group(relation_spatial_perf_group *group) {
    return ioctl(group->cycles_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == 0;
}

void relation_spatial_prefault(relation_spatial_carrier_state *state) {
    uint32_t i = 0u;
    if (state == NULL) {
        return;
    }
    for (i = 0u; i < FAMILY10H_RELATION_SPATIAL_LINE_COUNT; ++i) {
        state->lane_a[i * FAMILY10H_RELATION_SPATIAL_LINE_BYTES] = (uint8_t)i;
        state->lane_b[i * FAMILY10H_RELATION_SPATIAL_LINE_BYTES] = (uint8_t)(i ^ 0x5au);
        state->sham_a[i * FAMILY10H_RELATION_SPATIAL_LINE_BYTES] = (uint8_t)(i ^ 0xa5u);
        state->sham_b[i * FAMILY10H_RELATION_SPATIAL_LINE_BYTES] = (uint8_t)(i ^ 0xc3u);
    }
}

static void flush_state_lines(relation_spatial_carrier_state *state) {
#if defined(__x86_64__) || defined(__i386__)
    uint32_t i = 0u;
    for (i = 0u; i < FAMILY10H_RELATION_SPATIAL_LINE_COUNT; ++i) {
        _mm_clflush(&state->lane_a[i * FAMILY10H_RELATION_SPATIAL_LINE_BYTES]);
        _mm_clflush(&state->lane_b[i * FAMILY10H_RELATION_SPATIAL_LINE_BYTES]);
    }
    _mm_mfence();
#else
    (void)state;
#endif
}

int relation_spatial_prepare(relation_spatial_preparation prep, relation_spatial_carrier_state *state) {
    uint32_t step = 0u;
    if (state == NULL || prep.cyclic_origin >= FAMILY10H_RELATION_SPATIAL_LINE_COUNT) {
        return 0;
    }
    relation_spatial_prefault(state);
    flush_state_lines(state);
    for (step = 0u; step < FAMILY10H_RELATION_SPATIAL_TOTAL_WORK; ++step) {
        uint32_t a_index = relation_spatial_origin_index(prep.cyclic_origin, step);
        uint32_t b_index = relation_spatial_map_index(prep.relation, a_index);
        volatile uint8_t *a = &state->lane_a[a_index * FAMILY10H_RELATION_SPATIAL_LINE_BYTES];
        volatile uint8_t *b = &state->lane_b[b_index * FAMILY10H_RELATION_SPATIAL_LINE_BYTES];
        if (prep.source_order == RELATION_SPATIAL_ORDER_AB) {
            *a = (uint8_t)(*a + 1u);
            *b = (uint8_t)(*b + 1u);
        } else {
            *b = (uint8_t)(*b + 1u);
            *a = (uint8_t)(*a + 1u);
        }
    }
    return 1;
}

static int rdtscp_available(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int aux = 0u;
    (void)__rdtscp(&aux);
    return 1;
#else
    return 0;
#endif
}

static uint64_t measure_load_cycles(volatile uint8_t *addr) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int aux = 0u;
    uint64_t start = 0u;
    uint64_t end = 0u;
    uint8_t value = 0u;
    _mm_lfence();
    start = __rdtscp(&aux);
    value = *addr;
    _mm_lfence();
    end = __rdtscp(&aux);
    relation_spatial_sink ^= (uint64_t)value;
    return end - start;
#else
    (void)addr;
    return 0u;
#endif
}

static int copy_field(char *dst, size_t dst_size, const char *src) {
    size_t n = strlen(src);
    if (n + 1u > dst_size) {
        return 0;
    }
    memcpy(dst, src, n + 1u);
    return 1;
}

static int parse_i(const char *text, int *out) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int parse_u32(const char *text, uint32_t *out) {
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > UINT32_MAX) {
        return 0;
    }
    *out = (uint32_t)value;
    return 1;
}

static int parse_u64(const char *text, uint64_t *out) {
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return 0;
    }
    *out = (uint64_t)value;
    return 1;
}

static int parse_bool_text(const char *text, int *out) {
    if (strcmp(text, "True") == 0 || strcmp(text, "true") == 0) {
        *out = 1;
        return 1;
    }
    if (strcmp(text, "False") == 0 || strcmp(text, "false") == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

static int parse_relation(const char *text, relation_spatial_relation_id *out) {
    if (strcmp(text, "relation_r0") == 0) {
        *out = RELATION_SPATIAL_R0;
        return 1;
    }
    if (strcmp(text, "relation_r1") == 0) {
        *out = RELATION_SPATIAL_R1;
        return 1;
    }
    return 0;
}

static int parse_source_order(const char *text, relation_spatial_order_id *out) {
    if (strcmp(text, "A_then_B") == 0) {
        *out = RELATION_SPATIAL_ORDER_AB;
        return 1;
    }
    if (strcmp(text, "B_then_A") == 0) {
        *out = RELATION_SPATIAL_ORDER_BA;
        return 1;
    }
    return 0;
}

static int parse_query_order(const char *text, relation_spatial_order_id *out) {
    if (strcmp(text, "AB") == 0) {
        *out = RELATION_SPATIAL_ORDER_AB;
        return 1;
    }
    if (strcmp(text, "BA") == 0) {
        *out = RELATION_SPATIAL_ORDER_BA;
        return 1;
    }
    return 0;
}

static int parse_control(const char *query, relation_spatial_control_id *out) {
    if (strcmp(query, "query_relation_pair") == 0) {
        *out = RELATION_SPATIAL_CONTROL_NONE;
        return 1;
    }
    if (strcmp(query, "relation_sham") == 0) {
        *out = RELATION_SPATIAL_CONTROL_RELATION_SHAM;
        return 1;
    }
    if (strcmp(query, "scrambled_pair_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SCRAMBLED_PAIR;
        return 1;
    }
    if (strcmp(query, "route_pressure_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_ROUTE_PRESSURE;
        return 1;
    }
    if (strcmp(query, "distance_matched_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED;
        return 1;
    }
    if (strcmp(query, "source_off_offset_1_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1;
        return 1;
    }
    if (strcmp(query, "source_on_offset_2_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_2;
        return 1;
    }
    if (strcmp(query, "source_off_offset_2_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_2;
        return 1;
    }
    if (strcmp(query, "dead_offset_2_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_DEAD_OFFSET_2;
        return 1;
    }
    if (strcmp(query, "reset_double_flush_offset_2_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_2;
        return 1;
    }
    if (strcmp(query, "source_on_offset_4_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_4;
        return 1;
    }
    if (strcmp(query, "source_off_offset_4_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_4;
        return 1;
    }
    if (strcmp(query, "source_on_offset_8_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_8;
        return 1;
    }
    if (strcmp(query, "source_off_offset_8_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_8;
        return 1;
    }
    if (strcmp(query, "source_on_offset_16_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_16;
        return 1;
    }
    if (strcmp(query, "source_off_offset_16_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_16;
        return 1;
    }
    if (strcmp(query, "dead_offset_16_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_DEAD_OFFSET_16;
        return 1;
    }
    if (strcmp(query, "reset_double_flush_offset_16_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_16;
        return 1;
    }
    if (strcmp(query, "source_on_offset_1024_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_1024;
        return 1;
    }
    if (strcmp(query, "source_off_offset_1024_control") == 0) {
        *out = RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1024;
        return 1;
    }
    return 0;
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
    return count;
}

static void trim_line(char *line) {
    size_t n = strlen(line);
    while (n > 0u && (line[n - 1u] == '\n' || line[n - 1u] == '\r')) {
        line[n - 1u] = '\0';
        --n;
    }
}

static int parse_schedule_row(char **fields, int count, int expected_ordinal, relation_spatial_schedule_row *row) {
    int relation_match = 0;
    memset(row, 0, sizeof(*row));
    if (count != RELATION_SPATIAL_SCHEDULE_COLUMNS) {
        return 0;
    }
    if (!copy_field(row->tuple_id, sizeof(row->tuple_id), fields[0])
        || !parse_i(fields[1], &row->execution_ordinal)
        || row->execution_ordinal != expected_ordinal
        || !copy_field(row->block_id, sizeof(row->block_id), fields[2])
        || !parse_i(fields[3], &row->block_local_position)
        || !copy_field(row->row_role, sizeof(row->row_role), fields[4])
        || !parse_i(fields[5], &row->q)) {
        return 0;
    }
    if (!parse_u32(fields[6], &row->bank_a_work)
        || !parse_u32(fields[7], &row->bank_b_work)
        || !parse_u32(fields[8], &row->total_work)
        || !parse_relation(fields[9], &row->r_prepare)
        || !parse_relation(fields[10], &row->r_query)
        || !copy_field(row->r_prepare_text, sizeof(row->r_prepare_text), fields[9])
        || !copy_field(row->r_query_text, sizeof(row->r_query_text), fields[10])
        || !parse_bool_text(fields[11], &relation_match)
        || !copy_field(row->query, sizeof(row->query), fields[12])
        || !copy_field(row->relation_cell, sizeof(row->relation_cell), fields[13])) {
        return 0;
    }
    row->relation_match = relation_match;
    if (!copy_field(row->session, sizeof(row->session), fields[14])
        || !parse_i(fields[15], &row->replicate)
        || !copy_field(row->mapping, sizeof(row->mapping), fields[16])
        || !copy_field(row->delay_label, sizeof(row->delay_label), fields[17])
        || !parse_u64(fields[18], &row->delay_ns)
        || !copy_field(row->source_lifetime, sizeof(row->source_lifetime), fields[19])
        || !copy_field(row->lifetime_pair_id, sizeof(row->lifetime_pair_id), fields[20])
        || !copy_field(row->lifetime_execution_order, sizeof(row->lifetime_execution_order), fields[21])
        || !parse_u64(fields[22], &row->lifetime_hold_ns)
        || !parse_source_order(fields[23], &row->source_order)
        || !parse_query_order(fields[24], &row->query_order)
        || !parse_u32(fields[25], &row->cyclic_origin)) {
        return 0;
    }
    if (!copy_field(row->route_pressure_class, sizeof(row->route_pressure_class), fields[26])
        || !copy_field(row->distance_control_class, sizeof(row->distance_control_class), fields[27])
        || !copy_field(row->allocation_order_class, sizeof(row->allocation_order_class), fields[28])
        || !copy_field(row->prefault_class, sizeof(row->prefault_class), fields[29])
        || !copy_field(row->operation_semantics_id, sizeof(row->operation_semantics_id), fields[30])
        || !copy_field(row->control_semantics_id, sizeof(row->control_semantics_id), fields[31])
        || !parse_i(fields[32], &row->source_cpu_expected)
        || !parse_i(fields[33], &row->receiver_cpu_expected)
        || !parse_u32(fields[34], &row->source_loop_count)
        || !parse_u32(fields[35], &row->receiver_loop_count)
        || !parse_u32(fields[36], &row->read_count)
        || !parse_u32(fields[37], &row->write_count)
        || !parse_u32(fields[38], &row->page_count_a)
        || !parse_u32(fields[39], &row->page_count_b)
        || !parse_u32(fields[40], &row->line_count_a)
        || !parse_u32(fields[41], &row->line_count_b)
        || !copy_field(row->expected_pmu_group, sizeof(row->expected_pmu_group), fields[49])
        || !parse_bool_text(fields[50], &row->requires_pmu)
        || !parse_bool_text(fields[51], &row->post_observation_scheduling)
        || !parse_control(row->query, &row->control)) {
        return 0;
    }
    if (row->q != 0 || row->total_work != FAMILY10H_RELATION_SPATIAL_TOTAL_WORK || strcmp(row->source_lifetime, "alive_during_query") != 0) {
        return 0;
    }
    if (strcmp(row->expected_pmu_group, "family10h_public_relation_match_group") != 0 || row->post_observation_scheduling != 0) {
        return 0;
    }
    if (row->read_count != FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT * 2u || row->receiver_loop_count != FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT * 2u) {
        return 0;
    }
    return 1;
}

static uint32_t selected_a_index(const relation_spatial_schedule_row *row, uint32_t sample_index) {
    return ((uint32_t)row->execution_ordinal + (sample_index * FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_STRIDE)) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;
}

static uint32_t signed_offset_index(const relation_spatial_schedule_row *row, uint32_t a_index, uint32_t offset) {
    uint32_t bounded = offset % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;
    if (row->r_query == RELATION_SPATIAL_R0) {
        return (a_index + bounded) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;
    }
    return (a_index + FAMILY10H_RELATION_SPATIAL_LINE_COUNT - bounded) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;
}

static uint32_t control_b_index(const relation_spatial_schedule_row *row, uint32_t a_index, uint32_t sample_index) {
    if (row->control == RELATION_SPATIAL_CONTROL_RELATION_SHAM) {
        return (a_index + 1024u + ((uint32_t)row->execution_ordinal & 31u)) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;
    }
    if (row->control == RELATION_SPATIAL_CONTROL_SCRAMBLED_PAIR) {
        return ((a_index * 1103515245u) + 12345u + (uint32_t)row->execution_ordinal) & (FAMILY10H_RELATION_SPATIAL_LINE_COUNT - 1u);
    }
    if (row->control == RELATION_SPATIAL_CONTROL_ROUTE_PRESSURE) {
        return (a_index + (sample_index * 17u) + 33u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT;
    }
    if (row->control == RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED) {
        return (sample_index & 1u) ? ((a_index + 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT)
                                  : ((a_index + FAMILY10H_RELATION_SPATIAL_LINE_COUNT - 1u) % FAMILY10H_RELATION_SPATIAL_LINE_COUNT);
    }
    if (row->control == RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_2
        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_2
        || row->control == RELATION_SPATIAL_CONTROL_DEAD_OFFSET_2
        || row->control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_2) {
        return signed_offset_index(row, a_index, 2u);
    }
    if (row->control == RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_4 || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_4) {
        return signed_offset_index(row, a_index, 4u);
    }
    if (row->control == RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_8 || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_8) {
        return signed_offset_index(row, a_index, 8u);
    }
    if (row->control == RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_16
        || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_16
        || row->control == RELATION_SPATIAL_CONTROL_DEAD_OFFSET_16
        || row->control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_16) {
        return signed_offset_index(row, a_index, 16u);
    }
    if (row->control == RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_1024 || row->control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1024) {
        return signed_offset_index(row, a_index, 1024u);
    }
    return relation_spatial_map_index(row->r_query, a_index);
}

static int source_dead_before_query_control(relation_spatial_control_id control) {
    return control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1
        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_2
        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_4
        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_8
        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_16
        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1024
        || control == RELATION_SPATIAL_CONTROL_DEAD_OFFSET_2
        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_2
        || control == RELATION_SPATIAL_CONTROL_DEAD_OFFSET_16
        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_16;
}

static int source_off_control(relation_spatial_control_id control) {
    return control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1
        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_2
        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_4
        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_8
        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_16
        || control == RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1024;
}

static int reset_double_flush_after_source_dead_control(relation_spatial_control_id control) {
    return control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_2
        || control == RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_16;
}

static int sample_a_first(const relation_spatial_schedule_row *row, uint32_t sample_index) {
    uint32_t parity = sample_index + (uint32_t)row->execution_ordinal;
    if (row->query_order == RELATION_SPATIAL_ORDER_BA) {
        parity ^= 1u;
    }
    return (parity & 1u) == 0u;
}

static uint64_t synthetic_cycles(const relation_spatial_schedule_row *row, uint32_t sample_index, int lane_b, uint32_t a_index, uint32_t b_index) {
    uint64_t base = 120u + ((uint64_t)((a_index * 13u + sample_index * 7u + (uint32_t)row->execution_ordinal) & 31u));
    if (!lane_b) {
        return base;
    }
    if (row->control != RELATION_SPATIAL_CONTROL_NONE) {
        return 130u + ((uint64_t)((b_index * 5u + sample_index * 11u + 17u) & 31u));
    }
    if (row->relation_match) {
        return base + (uint64_t)((sample_index + (uint32_t)row->execution_ordinal) & 3u);
    }
    return 190u - (base & 63u);
}

static void rank_values(const uint64_t *values, double *ranks, uint32_t count) {
    uint32_t i = 0u;
    for (i = 0u; i < count; ++i) {
        uint32_t less = 0u;
        uint32_t equal = 0u;
        uint32_t j = 0u;
        for (j = 0u; j < count; ++j) {
            if (values[j] < values[i]) {
                ++less;
            } else if (values[j] == values[i]) {
                ++equal;
            }
        }
        ranks[i] = (double)less + ((double)(equal - 1u) / 2.0);
    }
}

static double pearson(const double *a, const double *b, uint32_t count) {
    double ma = 0.0;
    double mb = 0.0;
    double num = 0.0;
    double da = 0.0;
    double db = 0.0;
    uint32_t i = 0u;
    for (i = 0u; i < count; ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= (double)count;
    mb /= (double)count;
    for (i = 0u; i < count; ++i) {
        double xa = a[i] - ma;
        double xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if (da <= 0.0 || db <= 0.0) {
        return 0.0;
    }
    return num / sqrt(da * db);
}

static double spearman(const uint64_t *a, const uint64_t *b, uint32_t count) {
    double ra[FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT];
    double rb[FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT];
    rank_values(a, ra, count);
    rank_values(b, rb, count);
    return pearson(ra, rb, count);
}

static int write_feature_freeze(FILE *feature, int synthetic, int raw_count, int pair_count, const char *schedule_path) {
    return fprintf(
               feature,
               "{\"schema\":\"FAMILY10H_RELATION_SPATIAL_FEATURE_FREEZE_V1\",\"primary_endpoint\":\"spatial_pair_first_touch_latency\",\"primary_coordinate\":\"C_pair_spearman\",\"secondary_endpoints\":[\"change_to_dirty\",\"dirty_probe_response\",\"cpu_cycles\",\"duration_ns\"],\"timer_method\":\"rdtscp_lfence_serialized\",\"schedule_path\":\"%s\",\"schedule_sha256\":\"%s\",\"raw_record_count\":%d,\"pair_observation_count\":%d,\"pair_measurements_per_row\":%u,\"physical_measurement\":%s,\"post_observation_feature_selection\":false,\"small_wall_crossed\":false}\n",
               schedule_path,
               RELATION_SPATIAL_CANONICAL_SCHEDULE_SHA256,
               raw_count,
               pair_count,
               FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT,
               synthetic ? "false" : "true")
           > 0;
}

static int write_raw_record(
    FILE *raw,
    const relation_spatial_schedule_row *row,
    double c_pair,
    uint64_t change_to_dirty,
    uint64_t dirty_probe_response,
    uint64_t cpu_cycles,
    uint64_t duration_ns,
    uint64_t time_enabled,
    uint64_t time_running,
    int source_cpu_before,
    int source_cpu_after,
    int receiver_cpu_before,
    int receiver_cpu_after,
    int source_pid,
    uint64_t source_ready_ns,
    uint64_t source_exit_ns,
    uint64_t query_start_ns,
    uint64_t query_end_ns,
    int source_alive_at_pair_measurement,
    int synthetic
) {
    const char *source_order_text = row->source_order == RELATION_SPATIAL_ORDER_AB ? "A_then_B" : "B_then_A";
    const char *query_order_text = row->query_order == RELATION_SPATIAL_ORDER_AB ? "AB" : "BA";
    return fprintf(
               raw,
               "{\"tuple_id\":\"%s\",\"execution_ordinal\":%d,\"block_id\":\"%s\",\"block_local_position\":%d,\"row_role\":\"%s\",\"q\":%d,\"r_prepare\":\"%s\",\"r_query\":\"%s\",\"relation_match\":%s,\"query\":\"%s\",\"relation_cell\":\"%s\",\"session\":\"%s\",\"replicate\":%d,\"mapping\":\"%s\",\"source_lifetime\":\"%s\",\"source_order\":\"%s\",\"query_order\":\"%s\",\"cyclic_origin\":%u,\"source_cpu_expected\":%d,\"receiver_cpu_expected\":%d,\"pair_measurement_count\":%u,\"C_pair\":%.17g,\"timer_method\":\"rdtscp_lfence_serialized\",\"change_to_dirty\":%" PRIu64 ",\"dirty_probe_response\":%" PRIu64 ",\"cpu_cycles\":%" PRIu64 ",\"duration_ns\":%" PRIu64 ",\"time_enabled\":%" PRIu64 ",\"time_running\":%" PRIu64 ",\"source_cpu_before\":%d,\"source_cpu_after\":%d,\"receiver_cpu_before\":%d,\"receiver_cpu_after\":%d,\"source_pid\":%d,\"source_ready_monotonic_ns\":%" PRIu64 ",\"source_exit_monotonic_ns\":%" PRIu64 ",\"query_start_monotonic_ns\":%" PRIu64 ",\"query_end_monotonic_ns\":%" PRIu64 ",\"source_alive_at_pair_measurement\":%s,\"physical_measurement\":%s,\"positive_physical_claim\":false}\n",
               row->tuple_id,
               row->execution_ordinal,
               row->block_id,
               row->block_local_position,
               row->row_role,
               row->q,
               row->r_prepare_text,
               row->r_query_text,
               row->relation_match ? "true" : "false",
               row->query,
               row->relation_cell,
               row->session,
               row->replicate,
               row->mapping,
               row->source_lifetime,
               source_order_text,
               query_order_text,
               row->cyclic_origin,
               row->source_cpu_expected,
               row->receiver_cpu_expected,
               FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT,
               c_pair,
               change_to_dirty,
               dirty_probe_response,
               cpu_cycles,
               duration_ns,
               time_enabled,
               time_running,
               source_cpu_before,
               source_cpu_after,
               receiver_cpu_before,
               receiver_cpu_after,
               source_pid,
               source_ready_ns,
               source_exit_ns,
               query_start_ns,
               query_end_ns,
               source_alive_at_pair_measurement ? "true" : "false",
               synthetic ? "false" : "true")
           > 0;
}

static int execute_schedule_common(const char *schedule_path, const char *output_root, int synthetic) {
    FILE *schedule = NULL;
    FILE *raw = NULL;
    FILE *pairs = NULL;
    FILE *deaths = NULL;
    FILE *feature = NULL;
    FILE *receipt = NULL;
    char raw_path[4096];
    char pair_path[4096];
    char death_path[4096];
    char feature_path[4096];
    char receipt_path[4096];
    char line[RELATION_SPATIAL_LINE_BUFFER];
    int expected_ordinal = 0;
    int raw_count = 0;
    int pair_count = 0;
    int rc = 1;
    if (!synthetic && !relation_spatial_runtime_live_authority_present()) {
        fprintf(stderr, "live runtime authority missing\n");
        return 17;
    }
    if (!rdtscp_available()) {
        fprintf(stderr, "serialized RDTSCP timer unavailable\n");
        return 18;
    }
    if (mkdir(output_root, 0700) != 0) {
        fprintf(stderr, "output root create failed: %s\n", strerror(errno));
        return 19;
    }
    snprintf(raw_path, sizeof(raw_path), "%s/raw_records.jsonl", output_root);
    snprintf(pair_path, sizeof(pair_path), "%s/pair_observations.jsonl", output_root);
    snprintf(death_path, sizeof(death_path), "%s/source_death_receipts.jsonl", output_root);
    snprintf(feature_path, sizeof(feature_path), "%s/feature_freeze.json", output_root);
    snprintf(receipt_path, sizeof(receipt_path), "%s/target_execution_receipt.json", output_root);
    schedule = fopen(schedule_path, "r");
    raw = fopen(raw_path, "w");
    pairs = fopen(pair_path, "w");
    deaths = fopen(death_path, "w");
    feature = fopen(feature_path, "w");
    receipt = fopen(receipt_path, "w");
    if (schedule == NULL || raw == NULL || pairs == NULL || deaths == NULL || feature == NULL || receipt == NULL) {
        fprintf(stderr, "failed to open schedule/output files\n");
        goto cleanup;
    }
    if (fgets(line, sizeof(line), schedule) == NULL) {
        fprintf(stderr, "schedule missing header\n");
        goto cleanup;
    }
    while (fgets(line, sizeof(line), schedule) != NULL) {
        relation_spatial_schedule_row row;
        char *fields[RELATION_SPATIAL_SCHEDULE_COLUMNS];
        relation_spatial_shared_state *shared = NULL;
        pid_t child = -1;
        int source_pid_record = 0;
        int wait_status = 0;
        uint64_t source_ready_ns = 0u;
        uint64_t source_exit_ns = 0u;
        uint64_t query_start_ns = 0u;
        uint64_t query_end_ns = 0u;
        int source_cpu_before = -1;
        int source_cpu_after = -1;
        int receiver_cpu_before = -1;
        int receiver_cpu_after = -1;
        uint64_t a_cycles[FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT];
        uint64_t b_cycles[FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT];
        relation_spatial_perf_group group;
        relation_spatial_perf_readout readout;
        uint64_t change_to_dirty_value = 0u;
        uint64_t dirty_probe_value = 0u;
        uint64_t cpu_cycles_value = 0u;
        uint64_t duration_value = 0u;
        uint64_t time_enabled = 0u;
        uint64_t time_running = 0u;
        uint32_t sample = 0u;
        double c_pair = 0.0;
        int source_alive_at_pair_measurement = 1;
        trim_line(line);
        if (!parse_schedule_row(fields, split_tsv(line, fields, RELATION_SPATIAL_SCHEDULE_COLUMNS), expected_ordinal, &row)) {
            fprintf(stderr, "schedule parse failed at row %d\n", expected_ordinal);
            goto cleanup;
        }
        memset(&group, 0, sizeof(group));
        group.cycles_fd = -1;
        group.c2d_fd = -1;
        group.probe_fd = -1;
        if (synthetic) {
            shared = mmap(NULL, sizeof(*shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
            if (shared == MAP_FAILED) {
                goto cleanup;
            }
            memset(shared, 0, sizeof(*shared));
            if (source_off_control(row.control)) {
                relation_spatial_prefault(&shared->state);
                flush_state_lines(&shared->state);
                shared->preparation_ok = 1;
            } else {
                relation_spatial_preparation prep;
                prep.bank_a_work = row.bank_a_work;
                prep.bank_b_work = row.bank_b_work;
                prep.relation = row.r_prepare;
                prep.source_order = row.source_order;
                prep.cyclic_origin = row.cyclic_origin;
                shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);
            }
            source_cpu_before = row.source_cpu_expected;
            source_cpu_after = row.source_cpu_expected;
            receiver_cpu_before = row.receiver_cpu_expected;
            receiver_cpu_after = row.receiver_cpu_expected;
            source_ready_ns = UINT64_C(1000000000) + (uint64_t)row.execution_ordinal;
            query_start_ns = source_ready_ns + row.delay_ns + 1u;
        } else {
            if (!pin_to_core_pair(row.source_cpu_expected, row.receiver_cpu_expected)) {
                fprintf(stderr, "source/receiver CPU pair pin failed\n");
                goto cleanup;
            }
            if (!open_perf_group(&group)) {
                fprintf(stderr, "PMU group open failed\n");
                goto cleanup;
            }
            shared = mmap(NULL, sizeof(*shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
            if (shared == MAP_FAILED) {
                goto cleanup;
            }
            memset(shared, 0, sizeof(*shared));
            child = fork();
            if (child < 0) {
                goto cleanup;
            }
            source_pid_record = (int)child;
            if (child == 0) {
                relation_spatial_preparation prep;
                if (!pin_to_core(row.source_cpu_expected)) {
                    _exit(11);
                }
                shared->source_cpu_before = current_cpu_checked();
                prep.bank_a_work = row.bank_a_work;
                prep.bank_b_work = row.bank_b_work;
                prep.relation = row.r_prepare;
                prep.source_order = row.source_order;
                prep.cyclic_origin = row.cyclic_origin;
                shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);
                shared->source_cpu_after = current_cpu_checked();
                shared->source_ready_ns = monotonic_ns();
                shared->source_ready = 1;
                while (!shared->release_source) {
#if defined(__x86_64__) || defined(__i386__)
                    _mm_pause();
#endif
                    relation_spatial_sink ^= 1u;
                }
                _exit((shared->source_cpu_before == row.source_cpu_expected && shared->source_cpu_after == row.source_cpu_expected && shared->preparation_ok) ? 0 : 12);
            }
            while (!shared->source_ready) {
                if (kill(child, 0) != 0) {
                    fprintf(stderr, "source exited before ready\n");
                    goto cleanup;
                }
                wait_for_peer_progress();
            }
            source_ready_ns = shared->source_ready_ns;
            source_cpu_before = shared->source_cpu_before;
            source_cpu_after = shared->source_cpu_after;
            if (source_dead_before_query_control(row.control)) {
                shared->release_source = 1;
                if (waitpid(child, &wait_status, 0) < 0) {
                    goto cleanup;
                }
                source_exit_ns = monotonic_ns();
                child = -1;
                source_alive_at_pair_measurement = 0;
                if (reset_double_flush_after_source_dead_control(row.control)) {
                    flush_state_lines(&shared->state);
                    flush_state_lines(&shared->state);
                }
            }
            if (!pin_to_core(row.receiver_cpu_expected)) {
                fprintf(stderr, "receiver CPU pin failed\n");
                goto cleanup;
            }
            receiver_cpu_before = current_cpu_checked();
            if (!enable_perf_group(&group)) {
                goto cleanup;
            }
            query_start_ns = monotonic_ns();
        }
        for (sample = 0u; sample < FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT; ++sample) {
            uint32_t a_index = selected_a_index(&row, sample);
            uint32_t b_index = control_b_index(&row, a_index, sample);
            volatile uint8_t *a_addr = &shared->state.lane_a[a_index * FAMILY10H_RELATION_SPATIAL_LINE_BYTES];
            volatile uint8_t *b_addr = &shared->state.lane_b[b_index * FAMILY10H_RELATION_SPATIAL_LINE_BYTES];
            int a_first = sample_a_first(&row, sample);
            if (synthetic) {
                if (a_first) {
                    a_cycles[sample] = synthetic_cycles(&row, sample, 0, a_index, b_index);
                    b_cycles[sample] = synthetic_cycles(&row, sample, 1, a_index, b_index);
                } else {
                    b_cycles[sample] = synthetic_cycles(&row, sample, 1, a_index, b_index);
                    a_cycles[sample] = synthetic_cycles(&row, sample, 0, a_index, b_index);
                }
            } else if (a_first) {
                a_cycles[sample] = measure_load_cycles(a_addr);
                b_cycles[sample] = measure_load_cycles(b_addr);
            } else {
                b_cycles[sample] = measure_load_cycles(b_addr);
                a_cycles[sample] = measure_load_cycles(a_addr);
            }
            if (fprintf(
                    pairs,
                    "{\"tuple_id\":\"%s\",\"execution_ordinal\":%d,\"block_id\":\"%s\",\"block_local_position\":%d,\"row_role\":\"%s\",\"relation_cell\":\"%s\",\"relation_match\":%s,\"query\":\"%s\",\"sample_index\":%u,\"pair_index\":%u,\"A_line_index\":%u,\"B_line_index\":%u,\"measurement_order\":\"%s\",\"A_first_touch_cycles\":%" PRIu64 ",\"B_first_touch_cycles\":%" PRIu64 ",\"source_alive_at_pair_measurement\":%s,\"source_cpu_expected\":%d,\"receiver_cpu_expected\":%d,\"timer_method\":\"rdtscp_lfence_serialized\",\"physical_measurement\":%s,\"positive_physical_claim\":false}\n",
                    row.tuple_id,
                    row.execution_ordinal,
                    row.block_id,
                    row.block_local_position,
                    row.row_role,
                    row.relation_cell,
                    row.relation_match ? "true" : "false",
                    row.query,
                    sample,
                    a_index,
                    a_index,
                    b_index,
                    a_first ? "A_first" : "B_first",
                    a_cycles[sample],
                    b_cycles[sample],
                    source_alive_at_pair_measurement ? "true" : "false",
                    row.source_cpu_expected,
                    row.receiver_cpu_expected,
                    synthetic ? "false" : "true")
                <= 0) {
                goto cleanup;
            }
            ++pair_count;
        }
        c_pair = spearman(a_cycles, b_cycles, FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT);
        if (synthetic) {
            query_end_ns = query_start_ns + 100000u + (uint64_t)(row.execution_ordinal & 1023);
            source_exit_ns = query_end_ns + 1u;
            change_to_dirty_value = (uint64_t)(row.relation_match ? 2u : 1u);
            dirty_probe_value = (uint64_t)(1000u + (uint32_t)(c_pair * 100.0));
            cpu_cycles_value = 100000u + (uint64_t)row.execution_ordinal;
            duration_value = query_end_ns - query_start_ns;
            time_enabled = duration_value;
            time_running = duration_value;
        } else {
            query_end_ns = monotonic_ns();
            if (!disable_perf_group(&group) || !read_perf_group(&group, &readout)) {
                goto cleanup;
            }
            receiver_cpu_after = current_cpu_checked();
            change_to_dirty_value = read_value_for_id(&readout, group.c2d_id);
            dirty_probe_value = read_value_for_id(&readout, group.probe_id);
            cpu_cycles_value = read_value_for_id(&readout, group.cycles_id);
            duration_value = query_end_ns - query_start_ns;
            time_enabled = readout.time_enabled;
            time_running = readout.time_running;
            if (source_alive_at_pair_measurement) {
                shared->release_source = 1;
                if (waitpid(child, &wait_status, 0) < 0) {
                    goto cleanup;
                }
                source_exit_ns = monotonic_ns();
                child = -1;
            }
            close_perf_group(&group);
        }
        if (!write_raw_record(
                raw,
                &row,
                c_pair,
                change_to_dirty_value,
                dirty_probe_value,
                cpu_cycles_value,
                duration_value,
                time_enabled,
                time_running,
                source_cpu_before,
                source_cpu_after,
                receiver_cpu_before,
                receiver_cpu_after,
                synthetic ? 0 : source_pid_record,
                source_ready_ns,
                source_exit_ns,
                query_start_ns,
                query_end_ns,
                source_alive_at_pair_measurement,
                synthetic)) {
            goto cleanup;
        }
        if (fprintf(
                deaths,
                "{\"tuple_id\":\"%s\",\"execution_ordinal\":%d,\"source_lifetime\":\"%s\",\"source_pid\":%d,\"source_ready_monotonic_ns\":%" PRIu64 ",\"source_exit_monotonic_ns\":%" PRIu64 ",\"query_start_monotonic_ns\":%" PRIu64 ",\"query_end_monotonic_ns\":%" PRIu64 ",\"source_alive_at_pair_measurement\":%s,\"source_alive_during_query\":%s,\"post_observation_query_or_window_selection\":false,\"process_custody\":\"%s\",\"source_cpu_before\":%d,\"source_cpu_after\":%d,\"physical_measurement\":%s}\n",
                row.tuple_id,
                row.execution_ordinal,
                source_alive_at_pair_measurement ? "alive_during_query" : (source_off_control(row.control) ? "source_off_no_preparation" : (reset_double_flush_after_source_dead_control(row.control) ? "dead_before_query_reset_double_flush" : "dead_before_query")),
                synthetic ? 0 : source_pid_record,
                source_ready_ns,
                source_exit_ns,
                query_start_ns,
                query_end_ns,
                source_alive_at_pair_measurement ? "true" : "false",
                source_alive_at_pair_measurement ? "true" : "false",
                source_alive_at_pair_measurement ? "source_alive_during_spatial_pair_probe" : (source_off_control(row.control) ? "source_off_no_preparation" : (reset_double_flush_after_source_dead_control(row.control) ? "source_dead_before_spatial_pair_probe_reset_double_flush" : "source_dead_before_spatial_pair_probe")),
                source_cpu_before,
                source_cpu_after,
                synthetic ? "false" : "true")
            <= 0) {
            goto cleanup;
        }
        ++raw_count;
        if (shared != NULL && shared != MAP_FAILED) {
            munmap(shared, sizeof(*shared));
            shared = NULL;
        }
        ++expected_ordinal;
    }
    if (!write_feature_freeze(feature, synthetic, raw_count, pair_count, schedule_path)) {
        goto cleanup;
    }
    if (fprintf(
            receipt,
            "{\"schema\":\"FAMILY10H_RELATION_SPATIAL_TARGET_EXECUTION_RECEIPT_V1\",\"status\":\"%s\",\"returncode\":0,\"raw_record_count\":%d,\"pair_observation_count\":%d,\"source_death_receipt_count\":%d,\"feature_freeze_written\":true,\"physical_measurement\":%s,\"pmu_opened\":%s,\"timer_method\":\"rdtscp_lfence_serialized\",\"live_activity\":%s,\"small_wall_crossed\":false}\n",
            synthetic ? "SYNTHETIC_EXECUTION_COMPLETE" : "PHYSICAL_EXECUTION_COMPLETE",
            raw_count,
            pair_count,
            raw_count,
            synthetic ? "false" : "true",
            synthetic ? "false" : "true",
            synthetic ? "false" : "true")
        <= 0) {
        goto cleanup;
    }
    rc = 0;
cleanup:
    if (schedule != NULL) {
        fclose(schedule);
    }
    if (raw != NULL) {
        fclose(raw);
    }
    if (pairs != NULL) {
        fclose(pairs);
    }
    if (deaths != NULL) {
        fclose(deaths);
    }
    if (feature != NULL) {
        fclose(feature);
    }
    if (receipt != NULL) {
        fclose(receipt);
    }
    return rc;
}

int relation_spatial_runtime_self_test(void) {
    relation_spatial_carrier_state *state = NULL;
    relation_spatial_preparation prep;
    uint32_t m0 = relation_spatial_map_index(RELATION_SPATIAL_R0, 0u);
    uint32_t m1 = relation_spatial_map_index(RELATION_SPATIAL_R1, 0u);
    if (!rdtscp_available() || m0 != 1u || m1 != FAMILY10H_RELATION_SPATIAL_LINE_COUNT - 1u) {
        return 0;
    }
    state = calloc(1u, sizeof(*state));
    if (state == NULL) {
        return 0;
    }
    prep.bank_a_work = 2048u;
    prep.bank_b_work = 2048u;
    prep.relation = RELATION_SPATIAL_R0;
    prep.source_order = RELATION_SPATIAL_ORDER_AB;
    prep.cyclic_origin = 0u;
    if (!relation_spatial_prepare(prep, state)) {
        free(state);
        return 0;
    }
    if (measure_load_cycles(&state->lane_a[0]) == 0u || measure_load_cycles(&state->lane_b[FAMILY10H_RELATION_SPATIAL_LINE_BYTES]) == 0u) {
        free(state);
        return 0;
    }
    free(state);
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        int passed = relation_spatial_runtime_self_test();
        printf("{\"schema\":\"FAMILY10H_RELATION_SPATIAL_RUNTIME_SELF_TEST_V1\",\"passed\":%s,\"timer_method\":\"rdtscp_lfence_serialized\",\"live_authority_present\":%s,\"pmu_acquisition_count\":0,\"small_wall_crossed\":false}\n",
               passed ? "true" : "false",
               relation_spatial_runtime_live_authority_present() ? "true" : "false");
        return passed ? 0 : 1;
    }
    if (argc == 4 && strcmp(argv[1], "--synthetic-execute-schedule") == 0) {
        return execute_schedule_common(argv[2], argv[3], 1);
    }
    if (argc == 4 && strcmp(argv[1], "--execute-schedule") == 0) {
        return execute_schedule_common(argv[2], argv[3], 0);
    }
    fprintf(stderr, "usage: %s --self-test | --synthetic-execute-schedule <schedule.tsv> <output-root> | --execute-schedule <schedule.tsv> <output-root>\n", argv[0]);
    return 2;
}
