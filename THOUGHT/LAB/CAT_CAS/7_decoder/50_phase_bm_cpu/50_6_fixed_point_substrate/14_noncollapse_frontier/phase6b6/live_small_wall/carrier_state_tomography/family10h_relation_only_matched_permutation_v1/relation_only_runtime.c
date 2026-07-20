#define _GNU_SOURCE

#include "relation_only_runtime.h"

#include <errno.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <sched.h>
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

#define RELATION_ONLY_RUNTIME_AUTH_ENV "FAMILY10H_RELATION_ONLY_RUNTIME_AUTHORITY"
#define RELATION_ONLY_RUNTIME_AUTH_VALUE "family10h_relation_only_matched_permutation_v1_0"
#define RELATION_ONLY_SCHEDULE_COLUMNS 48
#define RELATION_ONLY_LINE_BUFFER 16384
#define RELATION_ONLY_CANONICAL_SCHEDULE_SHA256 "a9d22bfa30a5bb10b0b72734b090c87e5b57abd72da8fa45943ea66683c7e471"

typedef struct {
    uint64_t value;
    uint64_t id;
} relation_only_perf_value;

typedef struct {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    relation_only_perf_value values[3];
} relation_only_perf_readout;

typedef struct {
    int cycles_fd;
    int c2d_fd;
    int probe_fd;
    uint64_t cycles_id;
    uint64_t c2d_id;
    uint64_t probe_id;
} relation_only_perf_group;

typedef struct {
    relation_only_carrier_state state;
    int source_cpu_before;
    int source_cpu_after;
} relation_only_shared_state;

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
    relation_only_relation_id r_prepare;
    relation_only_relation_id r_query;
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
    relation_only_order_id source_order;
    relation_only_order_id query_order;
    uint32_t cyclic_origin;
    char operation_semantics_id[64];
    char control_semantics_id[64];
    int source_cpu_expected;
    int receiver_cpu_expected;
    char expected_pmu_group[80];
    int requires_pmu;
    int post_observation_scheduling;
    relation_only_control_id control;
} relation_only_schedule_row;

static volatile uint64_t relation_only_sink = 0u;

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

static int open_perf_group(relation_only_perf_group *group) {
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

static void close_perf_group(relation_only_perf_group *group) {
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

static uint64_t read_value_for_id(const relation_only_perf_readout *readout, uint64_t id) {
    uint64_t i = 0u;
    for (i = 0u; i < readout->nr && i < 3u; ++i) {
        if (readout->values[i].id == id) {
            return readout->values[i].value;
        }
    }
    return 0u;
}

static int readout_has_id(const relation_only_perf_readout *readout, uint64_t id) {
    uint64_t i = 0u;
    for (i = 0u; i < readout->nr && i < 3u; ++i) {
        if (readout->values[i].id == id) {
            return 1;
        }
    }
    return 0;
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

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0u;
    }
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

uint32_t relation_only_map_index(relation_only_relation_id relation, uint32_t logical_a_index) {
    uint32_t index = logical_a_index % FAMILY10H_RELATION_ONLY_LINE_COUNT;
    if (relation == RELATION_ONLY_R0) {
        return (index + 1u) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
    }
    return (index + FAMILY10H_RELATION_ONLY_LINE_COUNT - 1u) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
}

uint32_t relation_only_origin_index(uint32_t cyclic_origin, uint32_t step) {
    return (cyclic_origin + step) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
}

static uint32_t byte_offset(uint32_t line_index) {
    return (line_index % FAMILY10H_RELATION_ONLY_LINE_COUNT) * FAMILY10H_RELATION_ONLY_LINE_BYTES;
}

void relation_only_prefault(relation_only_carrier_state *state) {
    uint32_t i = 0u;
    volatile uint8_t sink = 0u;
    if (state == NULL) {
        return;
    }
    memset(state, 0, sizeof(*state));
    for (i = 0u; i < FAMILY10H_RELATION_ONLY_LINE_COUNT; ++i) {
        uint32_t offset = byte_offset(i);
        sink ^= state->lane_a[offset];
        sink ^= state->lane_b[offset];
        sink ^= state->sham_a[offset];
        sink ^= state->sham_b[offset];
    }
    relation_only_sink ^= (uint64_t)sink;
}

static void write_line(uint8_t *lane, uint32_t line_index, uint32_t tag, uint32_t step) {
    uint32_t offset = byte_offset(line_index);
    lane[offset] = (uint8_t)(lane[offset] + (uint8_t)(tag + step));
}

static void prepare_pair_step(
    relation_only_carrier_state *state,
    relation_only_preparation prep,
    uint32_t step,
    uint32_t a_index,
    uint32_t b_index
) {
    if (prep.source_order == RELATION_ONLY_ORDER_AB) {
        if (step < prep.bank_a_work) {
            write_line(state->lane_a, a_index, 0xA0u, step);
        }
        if (step < prep.bank_b_work) {
            write_line(state->lane_b, b_index, 0xB0u, step);
        }
    } else {
        if (step < prep.bank_b_work) {
            write_line(state->lane_b, b_index, 0xB0u, step);
        }
        if (step < prep.bank_a_work) {
            write_line(state->lane_a, a_index, 0xA0u, step);
        }
    }
}

static uint32_t control_b_index(relation_only_control_id control, uint32_t a_index, uint32_t step) {
    if (control == RELATION_ONLY_CONTROL_RELATION_SHAM) {
        return (a_index + 1024u + ((step * 73u) % FAMILY10H_RELATION_ONLY_LINE_COUNT)) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
    }
    if (control == RELATION_ONLY_CONTROL_ROUTE_PRESSURE_SHAM) {
        return (a_index ^ 2048u) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
    }
    if (control == RELATION_ONLY_CONTROL_DISTANCE) {
        return relation_only_map_index((step & 1u) ? RELATION_ONLY_R0 : RELATION_ONLY_R1, a_index);
    }
    return relation_only_map_index(RELATION_ONLY_R0, a_index);
}

int relation_only_prepare(relation_only_preparation prep, relation_only_carrier_state *state) {
    uint32_t step = 0u;
    if (state == NULL) {
        return 0;
    }
    if (prep.bank_a_work + prep.bank_b_work != FAMILY10H_RELATION_ONLY_TOTAL_WORK) {
        return 0;
    }
    if (prep.cyclic_origin >= FAMILY10H_RELATION_ONLY_LINE_COUNT) {
        return 0;
    }
    for (step = 0u; step < FAMILY10H_RELATION_ONLY_TOTAL_WORK; ++step) {
        uint32_t a_index = relation_only_origin_index(prep.cyclic_origin, step);
        uint32_t b_index = relation_only_map_index(prep.relation, a_index);
        if (prep.control != RELATION_ONLY_CONTROL_NONE) {
            b_index = control_b_index(prep.control, a_index, step);
        }
        if (prep.control == RELATION_ONLY_CONTROL_INDEPENDENT_MARGINAL_REPLAY) {
            uint32_t independent_b = relation_only_origin_index(prep.cyclic_origin, step);
            if (prep.source_order == RELATION_ONLY_ORDER_AB) {
                if (step < prep.bank_a_work) {
                    write_line(state->lane_a, a_index, 0xA0u, step);
                }
                if (step < prep.bank_b_work) {
                    write_line(state->lane_b, independent_b, 0xB0u, step);
                }
            } else {
                if (step < prep.bank_b_work) {
                    write_line(state->lane_b, independent_b, 0xB0u, step);
                }
                if (step < prep.bank_a_work) {
                    write_line(state->lane_a, a_index, 0xA0u, step);
                }
            }
        } else {
            prepare_pair_step(state, prep, step, a_index, b_index);
        }
    }
    return 1;
}

static uint64_t mix(uint64_t acc, uint8_t value, uint32_t line_index, uint32_t tag) {
    acc ^= (uint64_t)value + ((uint64_t)line_index << 8) + (uint64_t)tag;
    acc *= UINT64_C(1099511628211);
    return acc;
}

static uint64_t query_scalar_lane(const uint8_t *lane, uint32_t tag, uint32_t cyclic_origin) {
    uint64_t acc = UINT64_C(1469598103934665603);
    uint32_t step = 0u;
    for (step = 0u; step < FAMILY10H_RELATION_ONLY_TOTAL_WORK; ++step) {
        uint32_t line_index = relation_only_origin_index(cyclic_origin, step);
        uint32_t offset = byte_offset(line_index);
        acc = mix(acc, lane[offset], line_index, tag);
    }
    relation_only_sink ^= acc;
    return acc;
}

static uint64_t query_line_pair(const relation_only_carrier_state *state, uint32_t a_index, uint32_t b_index, relation_only_order_id order) {
    uint64_t acc = UINT64_C(1469598103934665603);
    uint32_t a_offset = byte_offset(a_index);
    uint32_t b_offset = byte_offset(b_index);
    if (order == RELATION_ONLY_ORDER_AB) {
        acc = mix(acc, state->lane_a[a_offset], a_index, 0xA0u);
        acc = mix(acc, state->lane_b[b_offset], b_index, 0xB0u);
    } else {
        acc = mix(acc, state->lane_b[b_offset], b_index, 0xB0u);
        acc = mix(acc, state->lane_a[a_offset], a_index, 0xA0u);
    }
    return acc;
}

uint64_t relation_only_query(relation_only_query_spec query, const relation_only_carrier_state *state) {
    uint64_t acc = UINT64_C(1469598103934665603);
    uint32_t step = 0u;
    if (state == NULL || query.cyclic_origin >= FAMILY10H_RELATION_ONLY_LINE_COUNT) {
        return 0u;
    }
    for (step = 0u; step < FAMILY10H_RELATION_ONLY_TOTAL_WORK; ++step) {
        uint32_t a_index = relation_only_origin_index(query.cyclic_origin, step);
        uint32_t b_index = relation_only_map_index(query.relation, a_index);
        if (query.control != RELATION_ONLY_CONTROL_NONE) {
            b_index = control_b_index(query.control, a_index, step);
        }
        acc ^= query_line_pair(state, a_index, b_index, query.query_order);
        acc *= UINT64_C(1099511628211);
    }
    relation_only_sink ^= acc;
    return acc;
}

int relation_only_runtime_live_authority_present(void) {
    const char *value = getenv(RELATION_ONLY_RUNTIME_AUTH_ENV);
    return value != NULL && strcmp(value, RELATION_ONLY_RUNTIME_AUTH_VALUE) == 0;
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

static int parse_i32(const char *text, int32_t *out) {
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < INT32_MIN || value > INT32_MAX) {
        return 0;
    }
    *out = (int32_t)value;
    return 1;
}

static int parse_i(const char *text, int *out) {
    int32_t value = 0;
    if (!parse_i32(text, &value)) {
        return 0;
    }
    *out = (int)value;
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

static int parse_relation(const char *text, relation_only_relation_id *out) {
    if (strcmp(text, "relation_r0") == 0) {
        *out = RELATION_ONLY_R0;
        return 1;
    }
    if (strcmp(text, "relation_r1") == 0) {
        *out = RELATION_ONLY_R1;
        return 1;
    }
    if (strcmp(text, "control") == 0) {
        *out = RELATION_ONLY_R0;
        return 1;
    }
    return 0;
}

static int parse_source_order(const char *text, relation_only_order_id *out) {
    if (strcmp(text, "A_then_B") == 0) {
        *out = RELATION_ONLY_ORDER_AB;
        return 1;
    }
    if (strcmp(text, "B_then_A") == 0) {
        *out = RELATION_ONLY_ORDER_BA;
        return 1;
    }
    return 0;
}

static int parse_query_order(const char *text, relation_only_order_id *out) {
    if (strcmp(text, "AB") == 0) {
        *out = RELATION_ONLY_ORDER_AB;
        return 1;
    }
    if (strcmp(text, "BA") == 0) {
        *out = RELATION_ONLY_ORDER_BA;
        return 1;
    }
    return 0;
}

static int parse_control(const char *query, relation_only_control_id *out) {
    if (strcmp(query, "relation_sham") == 0) {
        *out = RELATION_ONLY_CONTROL_RELATION_SHAM;
        return 1;
    }
    if (strcmp(query, "route_pressure_sham") == 0) {
        *out = RELATION_ONLY_CONTROL_ROUTE_PRESSURE_SHAM;
        return 1;
    }
    if (strcmp(query, "independent_marginal_replay") == 0) {
        *out = RELATION_ONLY_CONTROL_INDEPENDENT_MARGINAL_REPLAY;
        return 1;
    }
    if (strcmp(query, "distance_control") == 0) {
        *out = RELATION_ONLY_CONTROL_DISTANCE;
        return 1;
    }
    *out = RELATION_ONLY_CONTROL_NONE;
    return 1;
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

static int copy_field(char *dest, size_t dest_size, const char *src) {
    size_t len = strlen(src);
    if (len + 1u > dest_size) {
        return 0;
    }
    memcpy(dest, src, len + 1u);
    return 1;
}

static int parse_schedule_row(char **fields, int count, int expected_ordinal, relation_only_schedule_row *row) {
    if (count != RELATION_ONLY_SCHEDULE_COLUMNS) {
        return 0;
    }
    memset(row, 0, sizeof(*row));
    if (!copy_field(row->tuple_id, sizeof(row->tuple_id), fields[0])) {
        return 0;
    }
    if (!parse_i(fields[1], &row->execution_ordinal) || row->execution_ordinal != expected_ordinal) {
        return 0;
    }
    if (!copy_field(row->block_id, sizeof(row->block_id), fields[2])) {
        return 0;
    }
    if (!parse_i(fields[3], &row->block_local_position)) {
        return 0;
    }
    if (!copy_field(row->row_role, sizeof(row->row_role), fields[4])) {
        return 0;
    }
    if (!parse_i32(fields[5], &row->q) || !parse_u32(fields[6], &row->bank_a_work) || !parse_u32(fields[7], &row->bank_b_work)) {
        return 0;
    }
    if (!parse_u32(fields[8], &row->total_work) || row->total_work != FAMILY10H_RELATION_ONLY_TOTAL_WORK) {
        return 0;
    }
    if (!parse_relation(fields[9], &row->r_prepare) || !parse_relation(fields[10], &row->r_query)) {
        return 0;
    }
    if (!copy_field(row->r_prepare_text, sizeof(row->r_prepare_text), fields[9])
        || !copy_field(row->r_query_text, sizeof(row->r_query_text), fields[10])) {
        return 0;
    }
    if (!parse_bool_text(fields[11], &row->relation_match)) {
        return 0;
    }
    if (!copy_field(row->query, sizeof(row->query), fields[12]) || !copy_field(row->relation_cell, sizeof(row->relation_cell), fields[13])) {
        return 0;
    }
    if (!copy_field(row->session, sizeof(row->session), fields[14]) || !parse_i(fields[15], &row->replicate)) {
        return 0;
    }
    if (!copy_field(row->mapping, sizeof(row->mapping), fields[16]) || !copy_field(row->delay_label, sizeof(row->delay_label), fields[17])) {
        return 0;
    }
    if (!parse_u64(fields[18], &row->delay_ns) || !parse_source_order(fields[19], &row->source_order) || !parse_query_order(fields[20], &row->query_order)) {
        return 0;
    }
    if (!parse_u32(fields[21], &row->cyclic_origin)) {
        return 0;
    }
    if (!(row->cyclic_origin == 0u || row->cyclic_origin == 1024u || row->cyclic_origin == 2048u || row->cyclic_origin == 3072u)) {
        return 0;
    }
    if (!copy_field(row->operation_semantics_id, sizeof(row->operation_semantics_id), fields[26])) {
        return 0;
    }
    if (!copy_field(row->control_semantics_id, sizeof(row->control_semantics_id), fields[27])) {
        return 0;
    }
    if (!parse_i(fields[28], &row->source_cpu_expected) || !parse_i(fields[29], &row->receiver_cpu_expected)) {
        return 0;
    }
    if (!copy_field(row->expected_pmu_group, sizeof(row->expected_pmu_group), fields[45])) {
        return 0;
    }
    if (!parse_bool_text(fields[46], &row->requires_pmu) || !parse_bool_text(fields[47], &row->post_observation_scheduling)) {
        return 0;
    }
    if (row->bank_a_work + row->bank_b_work != FAMILY10H_RELATION_ONLY_TOTAL_WORK) {
        return 0;
    }
    if (strcmp(row->expected_pmu_group, "family10h_public_relation_match_group") != 0 || row->post_observation_scheduling != 0) {
        return 0;
    }
    if (strcmp(row->row_role, "relation_matrix") == 0) {
        row->control = RELATION_ONLY_CONTROL_NONE;
        if (strcmp(row->query, "query_relation_pair") != 0) {
            return 0;
        }
    } else if (strcmp(row->row_role, "scalar_control") == 0) {
        row->control = RELATION_ONLY_CONTROL_NONE;
        if (!(strcmp(row->query, "query_A") == 0 || strcmp(row->query, "query_B") == 0)) {
            return 0;
        }
    } else if (strcmp(row->row_role, "relation_control") == 0) {
        if (!parse_control(row->query, &row->control) || row->control == RELATION_ONLY_CONTROL_NONE) {
            return 0;
        }
    } else {
        return 0;
    }
    return 1;
}

static int ensure_new_dir(const char *path) {
    if (mkdir(path, 0755) == 0) {
        return 1;
    }
    return 0;
}

static int path_join(char *dest, size_t dest_size, const char *root, const char *name) {
    int written = snprintf(dest, dest_size, "%s/%s", root, name);
    return written > 0 && (size_t)written < dest_size;
}

static relation_only_preparation preparation_from_row(const relation_only_schedule_row *row) {
    relation_only_preparation prep;
    prep.q = row->q;
    prep.bank_a_work = row->bank_a_work;
    prep.bank_b_work = row->bank_b_work;
    prep.relation = row->r_prepare;
    prep.source_order = row->source_order;
    prep.control = row->control;
    prep.cyclic_origin = row->cyclic_origin;
    return prep;
}

static uint64_t execute_query_for_row(const relation_only_schedule_row *row, const relation_only_carrier_state *state) {
    if (strcmp(row->query, "query_A") == 0) {
        return query_scalar_lane(state->lane_a, 0xA0u, row->cyclic_origin);
    }
    if (strcmp(row->query, "query_B") == 0) {
        return query_scalar_lane(state->lane_b, 0xB0u, row->cyclic_origin);
    }
    {
        relation_only_query_spec query;
        query.relation = row->r_query;
        query.query_order = row->query_order;
        query.control = row->control;
        query.cyclic_origin = row->cyclic_origin;
        return relation_only_query(query, state);
    }
}

static uint64_t synthetic_metric(const relation_only_schedule_row *row, uint64_t query_hash, const char *metric) {
    uint64_t base = (uint64_t)(1000000 + row->execution_ordinal + (int)row->cyclic_origin);
    if (strcmp(metric, "dirty") == 0) {
        if (strcmp(row->row_role, "relation_matrix") == 0) {
            return (uint64_t)(10000 + row->q + (row->relation_match ? 640 : 32));
        }
        if (strcmp(row->row_role, "relation_control") == 0) {
            return 0u;
        }
        return (uint64_t)(10000 + row->q + (strcmp(row->query, "query_A") == 0 ? 60 : 0));
    }
    if (strcmp(metric, "c2d") == 0) {
        return (query_hash & 1u) + (uint64_t)(row->relation_match != 0);
    }
    if (strcmp(metric, "cycles") == 0) {
        return base + (query_hash & 0xffu);
    }
    return 100000u + (query_hash & 0xffffu);
}

static int write_feature_freeze(FILE *feature, int synthetic, int raw_count, const char *schedule_path) {
    return fprintf(
        feature,
        "{\"schema\":\"FAMILY10H_RELATION_ONLY_FEATURE_FREEZE_V1\",\"primary_endpoint\":\"dirty_probe_response\",\"secondary_endpoints\":[\"change_to_dirty\",\"cpu_cycles\",\"duration_ns\"],\"schedule_path\":\"%s\",\"schedule_sha256\":\"%s\",\"raw_record_count\":%d,\"physical_measurement\":%s,\"post_observation_feature_selection\":false}\n",
        schedule_path,
        RELATION_ONLY_CANONICAL_SCHEDULE_SHA256,
        raw_count,
        synthetic ? "false" : "true"
    ) > 0;
}

static int execute_schedule_common(const char *schedule_path, const char *output_root, int synthetic) {
    FILE *schedule = NULL;
    FILE *raw = NULL;
    FILE *deaths = NULL;
    FILE *feature = NULL;
    FILE *receipt = NULL;
    char raw_path[4096];
    char death_path[4096];
    char feature_path[4096];
    char receipt_path[4096];
    char line[RELATION_ONLY_LINE_BUFFER];
    int expected_ordinal = 0;
    int raw_count = 0;
    int death_count = 0;
    int status_code = 0;
    relation_only_perf_group group;
    int group_opened = 0;
    memset(&group, 0, sizeof(group));
    group.cycles_fd = -1;
    group.c2d_fd = -1;
    group.probe_fd = -1;
    if (!synthetic && !relation_only_runtime_live_authority_present()) {
        return 13;
    }
    if (!ensure_new_dir(output_root)) {
        return 3;
    }
    if (!path_join(raw_path, sizeof(raw_path), output_root, "raw_records.jsonl")
        || !path_join(death_path, sizeof(death_path), output_root, "source_death_receipts.jsonl")
        || !path_join(feature_path, sizeof(feature_path), output_root, "feature_freeze.json")
        || !path_join(receipt_path, sizeof(receipt_path), output_root, "target_execution_receipt.json")) {
        return 4;
    }
    schedule = fopen(schedule_path, "r");
    raw = fopen(raw_path, "w");
    deaths = fopen(death_path, "w");
    feature = fopen(feature_path, "w");
    receipt = fopen(receipt_path, "w");
    if (schedule == NULL || raw == NULL || deaths == NULL || feature == NULL || receipt == NULL) {
        status_code = 4;
        goto finish;
    }
    if (!synthetic) {
        if (!open_perf_group(&group)) {
            status_code = 5;
            goto finish;
        }
        group_opened = 1;
    }
    while (fgets(line, sizeof(line), schedule) != NULL) {
        char *fields[RELATION_ONLY_SCHEDULE_COLUMNS + 2];
        int count = split_tsv(line, fields, RELATION_ONLY_SCHEDULE_COLUMNS + 2);
        relation_only_schedule_row row;
        relation_only_shared_state *shared = NULL;
        relation_only_carrier_state *state = NULL;
        relation_only_preparation prep;
        pid_t child = 0;
        int wait_status = 0;
        uint64_t source_exit_ns = 0u;
        uint64_t query_select_ns = 0u;
        uint64_t query_start_ns = 0u;
        uint64_t query_end_ns = 0u;
        uint64_t query_hash = 0u;
        int source_cpu_before = -1;
        int source_cpu_after = -1;
        int receiver_cpu_before = -1;
        int receiver_cpu_after = -1;
        relation_only_perf_readout readout;
        ssize_t read_size = 0;
        if (expected_ordinal == 0 && count > 0 && strcmp(fields[0], "tuple_id") == 0) {
            continue;
        }
        if (!parse_schedule_row(fields, count, expected_ordinal, &row)) {
            status_code = 6;
            goto finish;
        }
        shared = (relation_only_shared_state *)mmap(NULL, sizeof(*shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (shared == MAP_FAILED) {
            status_code = 7;
            goto finish;
        }
        state = &shared->state;
        relation_only_prefault(state);
        prep = preparation_from_row(&row);
        if (synthetic) {
            source_cpu_before = row.source_cpu_expected;
            source_cpu_after = row.source_cpu_expected;
            if (!relation_only_prepare(prep, state)) {
                munmap(shared, sizeof(*shared));
                status_code = 8;
                goto finish;
            }
            source_exit_ns = UINT64_C(1000000000) + (uint64_t)row.execution_ordinal;
        } else {
            child = fork();
            if (child == 0) {
                if (!pin_to_core(row.source_cpu_expected)) {
                    _exit(10);
                }
                shared->source_cpu_before = current_cpu_checked();
                if (!relation_only_prepare(prep, state)) {
                    _exit(11);
                }
                shared->source_cpu_after = current_cpu_checked();
                _exit((shared->source_cpu_before == row.source_cpu_expected && shared->source_cpu_after == row.source_cpu_expected) ? 0 : 12);
            }
            if (child < 0) {
                munmap(shared, sizeof(*shared));
                status_code = 9;
                goto finish;
            }
            if (waitpid(child, &wait_status, 0) != child) {
                munmap(shared, sizeof(*shared));
                status_code = 10;
                goto finish;
            }
            source_exit_ns = monotonic_ns();
            source_cpu_before = shared->source_cpu_before;
            source_cpu_after = shared->source_cpu_after;
            if (!(WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0)) {
                munmap(shared, sizeof(*shared));
                status_code = 11;
                goto finish;
            }
            if (!sleep_full_ns(row.delay_ns)) {
                munmap(shared, sizeof(*shared));
                status_code = 12;
                goto finish;
            }
            if (!pin_to_core(row.receiver_cpu_expected)) {
                munmap(shared, sizeof(*shared));
                status_code = 13;
                goto finish;
            }
        }
        query_select_ns = synthetic ? source_exit_ns + row.delay_ns : monotonic_ns();
        receiver_cpu_before = synthetic ? row.receiver_cpu_expected : current_cpu_checked();
        memset(&readout, 0, sizeof(readout));
        if (!synthetic) {
            if (ioctl(group.cycles_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0
                || ioctl(group.cycles_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
                munmap(shared, sizeof(*shared));
                status_code = 14;
                goto finish;
            }
        }
        query_start_ns = synthetic ? query_select_ns : monotonic_ns();
        query_hash = execute_query_for_row(&row, state);
        query_end_ns = synthetic ? query_start_ns + synthetic_metric(&row, query_hash, "duration") : monotonic_ns();
        receiver_cpu_after = synthetic ? row.receiver_cpu_expected : current_cpu_checked();
        if (!synthetic) {
            if (ioctl(group.cycles_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
                munmap(shared, sizeof(*shared));
                status_code = 14;
                goto finish;
            }
            read_size = read(group.cycles_fd, &readout, sizeof(readout));
            if (read_size != (ssize_t)sizeof(readout)
                || readout.nr != 3u
                || !readout_has_id(&readout, group.cycles_id)
                || !readout_has_id(&readout, group.c2d_id)
                || !readout_has_id(&readout, group.probe_id)) {
                munmap(shared, sizeof(*shared));
                status_code = 15;
                goto finish;
            }
        }
        fprintf(
            raw,
            "{\"tuple_id\":\"%s\",\"execution_ordinal\":%d,\"block_id\":\"%s\",\"row_role\":\"%s\",\"q\":%" PRId32 ",\"r_prepare\":\"%s\",\"r_query\":\"%s\",\"query\":\"%s\",\"relation_match\":%s,\"session\":\"%s\",\"replicate\":%d,\"mapping\":\"%s\",\"delay_label\":\"%s\",\"delay_ns\":%" PRIu64 ",\"source_order\":\"%s\",\"query_order\":\"%s\",\"cyclic_origin\":%" PRIu32 ",\"expected_pmu_group\":\"family10h_public_relation_match_group\",\"pmu_event_group\":\"family10h_public_relation_match_group\",\"change_to_dirty\":%" PRIu64 ",\"dirty_probe_response\":%" PRIu64 ",\"cpu_cycles\":%" PRIu64 ",\"duration_ns\":%" PRIu64 ",\"time_enabled\":%" PRIu64 ",\"time_running\":%" PRIu64 ",\"source_cpu_before\":%d,\"source_cpu_after\":%d,\"receiver_cpu_before\":%d,\"receiver_cpu_after\":%d,\"process_custody\":\"source_dead_before_query\",\"query_hash\":%" PRIu64 ",\"physical_measurement\":%s,\"positive_physical_claim\":false}\n",
            row.tuple_id,
            row.execution_ordinal,
            row.block_id,
            row.row_role,
            row.q,
            row.r_prepare_text,
            row.r_query_text,
            row.query,
            row.relation_match ? "true" : "false",
            row.session,
            row.replicate,
            row.mapping,
            row.delay_label,
            row.delay_ns,
            row.source_order == RELATION_ONLY_ORDER_AB ? "A_then_B" : "B_then_A",
            row.query_order == RELATION_ONLY_ORDER_AB ? "AB" : "BA",
            row.cyclic_origin,
            synthetic ? synthetic_metric(&row, query_hash, "c2d") : read_value_for_id(&readout, group.c2d_id),
            synthetic ? synthetic_metric(&row, query_hash, "dirty") : read_value_for_id(&readout, group.probe_id),
            synthetic ? synthetic_metric(&row, query_hash, "cycles") : read_value_for_id(&readout, group.cycles_id),
            synthetic ? synthetic_metric(&row, query_hash, "duration") : query_end_ns - query_start_ns,
            synthetic ? 100000u : readout.time_enabled,
            synthetic ? 100000u : readout.time_running,
            source_cpu_before,
            source_cpu_after,
            receiver_cpu_before,
            receiver_cpu_after,
            query_hash,
            synthetic ? "false" : "true"
        );
        fprintf(
            deaths,
            "{\"tuple_id\":\"%s\",\"execution_ordinal\":%d,\"source_pid\":%d,\"waitpid_pid\":%d,\"waitpid_status\":\"exited_0\",\"source_exit_monotonic_ns\":%" PRIu64 ",\"query_select_monotonic_ns\":%" PRIu64 ",\"source_alive_during_query\":false,\"source_helper_survives\":false,\"open_source_ipc_after_waitpid\":0,\"query_selected_after_waitpid\":true,\"post_observation_query_or_window_selection\":false,\"source_cpu_before\":%d,\"source_cpu_after\":%d,\"physical_measurement\":%s}\n",
            row.tuple_id,
            row.execution_ordinal,
            synthetic ? 0 : (int)child,
            synthetic ? 0 : (int)child,
            source_exit_ns,
            query_select_ns,
            source_cpu_before,
            source_cpu_after,
            synthetic ? "false" : "true"
        );
        ++raw_count;
        ++death_count;
        ++expected_ordinal;
        munmap(shared, sizeof(*shared));
    }
    if (!write_feature_freeze(feature, synthetic, raw_count, schedule_path)) {
        status_code = 16;
    }

finish:
    if (group_opened) {
        close_perf_group(&group);
    }
    if (receipt != NULL) {
        fprintf(
            receipt,
            "{\"schema\":\"%s\",\"status\":\"%s\",\"returncode\":%d,\"raw_record_count\":%d,\"source_death_receipt_count\":%d,\"feature_freeze_written\":%s,\"physical_measurement\":%s,\"pmu_opened\":%s,\"live_activity\":%s,\"small_wall_crossed\":false}\n",
            synthetic ? "FAMILY10H_RELATION_ONLY_SYNTHETIC_EXECUTOR_PACKET_V1" : "FAMILY10H_RELATION_ONLY_TARGET_EXECUTION_RECEIPT_V1",
            status_code == 0 ? "complete" : "failed",
            status_code,
            raw_count,
            death_count,
            status_code == 0 ? "true" : "false",
            synthetic ? "false" : "true",
            group_opened ? "true" : "false",
            synthetic ? "false" : "true"
        );
    }
    if (schedule != NULL) {
        fclose(schedule);
    }
    if (raw != NULL) {
        fclose(raw);
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
    return status_code;
}

static int permutation_self_test(void) {
    uint8_t seen_r0[FAMILY10H_RELATION_ONLY_LINE_COUNT];
    uint8_t seen_r1[FAMILY10H_RELATION_ONLY_LINE_COUNT];
    uint32_t i = 0u;
    memset(seen_r0, 0, sizeof(seen_r0));
    memset(seen_r1, 0, sizeof(seen_r1));
    for (i = 0u; i < FAMILY10H_RELATION_ONLY_LINE_COUNT; ++i) {
        uint32_t r0 = relation_only_map_index(RELATION_ONLY_R0, i);
        uint32_t r1 = relation_only_map_index(RELATION_ONLY_R1, i);
        if (r0 >= FAMILY10H_RELATION_ONLY_LINE_COUNT || r1 >= FAMILY10H_RELATION_ONLY_LINE_COUNT) {
            return 0;
        }
        seen_r0[r0] = 1u;
        seen_r1[r1] = 1u;
    }
    for (i = 0u; i < FAMILY10H_RELATION_ONLY_LINE_COUNT; ++i) {
        if (seen_r0[i] != 1u || seen_r1[i] != 1u) {
            return 0;
        }
    }
    return 1;
}

static int control_self_test(void) {
    const relation_only_control_id controls[] = {
        RELATION_ONLY_CONTROL_RELATION_SHAM,
        RELATION_ONLY_CONTROL_ROUTE_PRESSURE_SHAM,
        RELATION_ONLY_CONTROL_INDEPENDENT_MARGINAL_REPLAY,
        RELATION_ONLY_CONTROL_DISTANCE,
    };
    size_t i = 0u;
    for (i = 0u; i < sizeof(controls) / sizeof(controls[0]); ++i) {
        relation_only_carrier_state state;
        relation_only_preparation prep;
        relation_only_query_spec query;
        relation_only_prefault(&state);
        prep.q = 0;
        prep.bank_a_work = FAMILY10H_RELATION_ONLY_M;
        prep.bank_b_work = FAMILY10H_RELATION_ONLY_M;
        prep.relation = RELATION_ONLY_R0;
        prep.source_order = RELATION_ONLY_ORDER_AB;
        prep.control = controls[i];
        prep.cyclic_origin = 1024u;
        query.relation = RELATION_ONLY_R1;
        query.query_order = RELATION_ONLY_ORDER_BA;
        query.control = controls[i];
        query.cyclic_origin = 1024u;
        if (!relation_only_prepare(prep, &state) || relation_only_query(query, &state) == 0u) {
            return 0;
        }
    }
    return 1;
}

int relation_only_runtime_self_test(void) {
    relation_only_carrier_state state;
    const uint32_t origins[] = {0u, 1024u, 2048u, 3072u};
    size_t i = 0u;
    if (!permutation_self_test() || !control_self_test()) {
        return 0;
    }
    for (i = 0u; i < sizeof(origins) / sizeof(origins[0]); ++i) {
        relation_only_preparation prep;
        relation_only_query_spec q0;
        relation_only_query_spec q1;
        uint64_t r0 = 0u;
        uint64_t r1 = 0u;
        uint64_t scalar_a = 0u;
        uint64_t scalar_b = 0u;
        relation_only_prefault(&state);
        prep.q = 512;
        prep.bank_a_work = FAMILY10H_RELATION_ONLY_M + 512u;
        prep.bank_b_work = FAMILY10H_RELATION_ONLY_M - 512u;
        prep.relation = RELATION_ONLY_R0;
        prep.source_order = (i & 1u) ? RELATION_ONLY_ORDER_BA : RELATION_ONLY_ORDER_AB;
        prep.control = RELATION_ONLY_CONTROL_NONE;
        prep.cyclic_origin = origins[i];
        if (!relation_only_prepare(prep, &state)) {
            return 0;
        }
        q0.relation = RELATION_ONLY_R0;
        q0.query_order = RELATION_ONLY_ORDER_AB;
        q0.control = RELATION_ONLY_CONTROL_NONE;
        q0.cyclic_origin = origins[i];
        q1 = q0;
        q1.relation = RELATION_ONLY_R1;
        r0 = relation_only_query(q0, &state);
        r1 = relation_only_query(q1, &state);
        scalar_a = query_scalar_lane(state.lane_a, 0xA0u, origins[i]);
        scalar_b = query_scalar_lane(state.lane_b, 0xB0u, origins[i]);
        if (r0 == 0u || r1 == 0u || r0 == r1 || scalar_a == 0u || scalar_b == 0u || scalar_a == scalar_b) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        int passed = relation_only_runtime_self_test();
        printf("{\"schema\":\"FAMILY10H_RELATION_ONLY_RUNTIME_SELF_TEST_V2\",\"passed\":%s,\"pmu_opened\":false,\"live_activity\":false,\"physical_measurement\":false}\n", passed ? "true" : "false");
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
