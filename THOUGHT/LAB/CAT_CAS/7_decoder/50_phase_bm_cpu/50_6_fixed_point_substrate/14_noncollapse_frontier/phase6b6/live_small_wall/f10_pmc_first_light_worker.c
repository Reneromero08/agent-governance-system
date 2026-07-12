#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef CATCAS_CORE_A
#define CATCAS_CORE_A 4
#endif
#ifndef CATCAS_CORE_B
#define CATCAS_CORE_B 5
#endif
#define CACHE_LINE_BYTES 64
#define CARRIER_LINES 4096
#define READ_ITERATIONS 512
#define WRITE_ITERATIONS 512
#define PINGPONG_ITERATIONS 192
#define OPERATOR_ITERATIONS 96
#define MAX_GROUP_EVENTS 4
#define SUPPORT_EVENTS 8
#define RESULT_SCHEMA "CAT_CAS_F10_PMC_FIRST_LIGHT_RESULT_V1"
#define COHERENCE_RESULT_SCHEMA "CAT_CAS_F10_COHERENCE_OPERATOR_RESULT_V1"
#define PATH_RESULT_SCHEMA "CAT_CAS_F10_PATH_DEPENDENCE_PILOT_RESULT_V1"
#define PATH_DUAL_OBSERVE_RESULT_SCHEMA "CAT_CAS_F10_PATH_DUAL_OBSERVE_RESULT_V1"
#define PATH_RW_OBSERVE_RESULT_SCHEMA "CAT_CAS_F10_PATH_RW_OBSERVE_RESULT_V1"
#define ROUTE_STATE_RESULT_SCHEMA "CAT_CAS_F10_ROUTE_STATE_PILOT_RESULT_V1"
#define PHASE_LOCAL_PMU_RESULT_SCHEMA "CAT_CAS_F10_PHASE_LOCAL_PMU_RESULT_V1"
#define IBS_FIRST_LIGHT_RESULT_SCHEMA "CAT_CAS_F10_IBS_FIRST_LIGHT_RESULT_V1"
#define WC_FLUSH_ORDER_RESULT_SCHEMA "CAT_CAS_F10_WC_FLUSH_ORDER_RESULT_V1"
#define EVICTION_SENTINEL_RESULT_SCHEMA "CAT_CAS_F10_EVICTION_SENTINEL_RESULT_V1"
#define EVICTION_PHASE_LOCAL_RESULT_SCHEMA "CAT_CAS_F10_EVICTION_PHASE_LOCAL_RESULT_V1"
#define EVICTION_PHASE_BRACKETED_RESULT_SCHEMA "CAT_CAS_F10_EVICTION_PHASE_BRACKETED_RESULT_V1"
#define EVICTION_PHASE_BRACKETED_C2D_RESULT_SCHEMA "CAT_CAS_F10_EVICTION_PHASE_BRACKETED_C2D_RESULT_V1"
#define EVICTION_PHASE_BRACKETED_DURATION_RESULT_SCHEMA "CAT_CAS_F10_EVICTION_PHASE_BRACKETED_DURATION_RESULT_V1"
#define HISTORY_SENTINEL_RESULT_SCHEMA "CAT_CAS_F10_HISTORY_SENTINEL_RESULT_V1"
#define LOCKED_HISTORY_RESULT_SCHEMA "CAT_CAS_F10_LOCKED_HISTORY_RESULT_V1"
#define BRANCH_HISTORY_RESULT_SCHEMA "CAT_CAS_F10_BRANCH_HISTORY_RESULT_V1"
#define INDIRECT_TARGET_HISTORY_RESULT_SCHEMA "CAT_CAS_F10_INDIRECT_TARGET_HISTORY_RESULT_V1"
#define TRANSLATION_HISTORY_RESULT_SCHEMA "CAT_CAS_F10_TRANSLATION_HISTORY_RESULT_V1"
#define STORE_LOAD_ALIAS_HISTORY_RESULT_SCHEMA "CAT_CAS_F10_STORE_LOAD_ALIAS_HISTORY_RESULT_V1"
#define PREFETCH_STREAM_RESULT_SCHEMA "CAT_CAS_F10_PREFETCH_STREAM_RESULT_V1"
#define PROCESS_LIFECYCLE_RESULT_SCHEMA "CAT_CAS_F10_PROCESS_LIFECYCLE_RESULT_V1"
#define CODE_FOOTPRINT_HISTORY_RESULT_SCHEMA "CAT_CAS_F10_CODE_FOOTPRINT_HISTORY_RESULT_V1"
#define RETURN_STACK_HISTORY_RESULT_SCHEMA "CAT_CAS_F10_RETURN_STACK_HISTORY_RESULT_V1"
#define FP_PIPELINE_HISTORY_RESULT_SCHEMA "CAT_CAS_F10_FP_PIPELINE_HISTORY_RESULT_V1"
#define PAGE_PERMISSION_HISTORY_RESULT_SCHEMA "CAT_CAS_F10_PAGE_PERMISSION_HISTORY_RESULT_V1"
#define OWNED_RECOVERY_HISTORY_RESULT_SCHEMA "CAT_CAS_F10_OWNED_RECOVERY_HISTORY_RESULT_V1"
#define PHASE_LOCAL_BANK_LINES 512u
#define EVICTION_LINES 262144u
#define EVICTION_ITERATIONS 3
#define BRANCH_PATTERN_BYTES 4096u
#define BRANCH_RESTORE_LOOPS 128u
#define BRANCH_HISTORY_LOOPS 256u
#define BRANCH_SENTINEL_LOOPS 128u
#define LOCKED_HISTORY_LOOPS 4u
#define INDIRECT_TARGET_PATTERN_BYTES 4096u
#define INDIRECT_TARGET_RESTORE_LOOPS 128u
#define INDIRECT_TARGET_HISTORY_LOOPS 256u
#define INDIRECT_TARGET_SENTINEL_LOOPS 128u
#define TRANSLATION_PAGE_COUNT 8192u
#define TRANSLATION_RESTORE_LOOPS 4u
#define TRANSLATION_HISTORY_LOOPS 8u
#define TRANSLATION_SENTINEL_LOOPS 16u
#define STORE_LOAD_ALIAS_PAGE_BYTES 4096u
#define STORE_LOAD_ALIAS_PAGE_COUNT 16u
#define STORE_LOAD_ALIAS_PATTERN_BYTES 4096u
#define STORE_LOAD_ALIAS_RESTORE_LOOPS 64u
#define STORE_LOAD_ALIAS_HISTORY_LOOPS 128u
#define STORE_LOAD_ALIAS_SENTINEL_LOOPS 64u
#define PREFETCH_STREAM_LINES 65536u
#define PREFETCH_HISTORY_LINES 8192u
#define PREFETCH_SENTINEL_LINES 8192u
#define PREFETCH_RESTORE_LOOPS 2u
#define PREFETCH_HISTORY_LOOPS 8u
#define PREFETCH_SENTINEL_LOOPS 8u
#define PROCESS_LIFECYCLE_SOURCE_LOOPS 4u
#define CODE_FOOTPRINT_RESTORE_LOOPS 64u
#define CODE_FOOTPRINT_HISTORY_LOOPS 128u
#define CODE_FOOTPRINT_SENTINEL_LOOPS 64u
#define RETURN_STACK_RESTORE_LOOPS 64u
#define RETURN_STACK_HISTORY_LOOPS 128u
#define RETURN_STACK_SENTINEL_LOOPS 64u
#define FP_PIPELINE_RESTORE_LOOPS 64u
#define FP_PIPELINE_HISTORY_LOOPS 128u
#define FP_PIPELINE_SENTINEL_LOOPS 64u
#define PAGE_PERMISSION_PAGE_COUNT 256u
#define PAGE_PERMISSION_RESTORE_LOOPS 2u
#define PAGE_PERMISSION_HISTORY_LOOPS 2u
#define PAGE_PERMISSION_SENTINEL_LOOPS 64u
#define OWNED_RECOVERY_PAGE_COUNT 256u
#define OWNED_RECOVERY_HISTORY_LOOPS 1u
#define OWNED_RECOVERY_SENTINEL_LOOPS 64u

struct event_def {
    const char *name;
    unsigned int event_select;
    unsigned int unit_mask;
    const char *kind;
};

struct event_result {
    uint64_t value;
    uint64_t id;
};

struct group_result {
    bool opened;
    bool unmultiplexed;
    int open_errno;
    uint64_t time_enabled;
    uint64_t time_running;
    struct event_result events[MAX_GROUP_EVENTS];
};

struct carrier {
    unsigned char *bytes;
    size_t byte_count;
    size_t line_count;
};

struct pingpong_context {
    struct carrier *carrier;
    atomic_int turn;
    atomic_int ready;
    atomic_int done;
    volatile uint64_t sink_a;
    volatile uint64_t sink_b;
};

enum coherence_operator {
    OP_IDENTITY = 0,
    OP_SAME_CORE_PREFETCHW = 1,
    OP_REMOTE_READ_SHARED = 2,
    OP_REMOTE_PREFETCHW = 3,
    OP_SAME_CORE_LOCKED_NOOP = 4,
    OP_REMOTE_LOCKED_NOOP = 5,
    OP_SAME_CORE_STORE_SAME_VALUE = 6,
    OP_REMOTE_STORE_SAME_VALUE = 7
};

struct operator_context {
    struct carrier *carrier;
    enum coherence_operator op;
    atomic_int ready;
    atomic_int done;
    volatile uint64_t sink;
};

enum path_op {
    PATH_REMOTE_STORE = 0,
    PATH_HOME_STORE = 1,
    PATH_REMOTE_READ = 2
};

enum path_mode_kind {
    PATH_MODE_FIXED_CORE = 0,
    PATH_MODE_DUAL_OBSERVE = 1,
    PATH_MODE_RW_OBSERVE = 2
};

struct path_step {
    const char *name;
    enum path_op op;
    int line_set;
};

enum route_op {
    ROUTE_IDENTITY = 0,
    ROUTE_READ = 1,
    ROUTE_STORE = 2
};

enum wc_flush_op {
    WC_OP_IDENTITY = 0,
    WC_OP_FLUSH_ONLY = 1,
    WC_OP_NORMAL_STORE_SAME = 2,
    WC_OP_NT_STORE_SAME = 3,
    WC_OP_FLUSH_THEN_NT_STORE = 4,
    WC_OP_NT_STORE_THEN_FLUSH = 5
};

enum eviction_prep_op {
    EVICT_PREP_NONE = 0,
    EVICT_PREP_HOME_READ = 1,
    EVICT_PREP_REMOTE_READ = 2,
    EVICT_PREP_HOME_WRITE = 3,
    EVICT_PREP_REMOTE_WRITE = 4,
    EVICT_PREP_HOME_THEN_REMOTE_READ = 5,
    EVICT_PREP_REMOTE_THEN_HOME_READ = 6
};

struct route_spec {
    const char *name;
    int home_core;
    int remote_core;
};

struct phase_local_window_spec {
    const char *token;
    const char *role;
    int phase_index;
    size_t line_count;
    size_t start_bank;
};

struct eviction_phase_window_spec {
    const char *token;
    const char *role;
    int phase_index;
    enum eviction_prep_op prep;
};

struct eviction_bracket_token_spec {
    const char *token;
    const char *role;
    int phase_index;
    enum eviction_prep_op center_prep;
};

struct history_sequence_spec {
    const char *name;
    const struct path_step *steps;
    size_t step_count;
};

enum locked_history_kind {
    LOCKED_HISTORY_NEUTRAL = 0,
    LOCKED_HISTORY_FORWARD = 1,
    LOCKED_HISTORY_REVERSE = 2,
    LOCKED_HISTORY_SHUFFLE = 3
};

struct locked_history_sequence_spec {
    const char *name;
    enum locked_history_kind history_pattern;
    unsigned int history_loops;
};

enum branch_pattern_kind {
    BRANCH_PATTERN_NEUTRAL = 0,
    BRANCH_PATTERN_FORWARD = 1,
    BRANCH_PATTERN_REVERSE = 2,
    BRANCH_PATTERN_SHUFFLE = 3,
    BRANCH_PATTERN_SENTINEL = 4
};

struct branch_sequence_spec {
    const char *name;
    enum branch_pattern_kind history_pattern;
    unsigned int history_loops;
};

enum indirect_target_pattern_kind {
    INDIRECT_TARGET_PATTERN_NEUTRAL = 0,
    INDIRECT_TARGET_PATTERN_FORWARD = 1,
    INDIRECT_TARGET_PATTERN_REVERSE = 2,
    INDIRECT_TARGET_PATTERN_SHUFFLE = 3,
    INDIRECT_TARGET_PATTERN_SENTINEL = 4
};

struct indirect_target_sequence_spec {
    const char *name;
    enum indirect_target_pattern_kind history_pattern;
    unsigned int history_loops;
};

enum translation_pattern_kind {
    TRANSLATION_PATTERN_NEUTRAL = 0,
    TRANSLATION_PATTERN_FORWARD = 1,
    TRANSLATION_PATTERN_REVERSE = 2,
    TRANSLATION_PATTERN_SHUFFLE = 3,
    TRANSLATION_PATTERN_SENTINEL = 4
};

struct translation_sequence_spec {
    const char *name;
    enum translation_pattern_kind history_pattern;
    unsigned int history_loops;
};

enum store_load_alias_pattern_kind {
    STORE_LOAD_ALIAS_PATTERN_NEUTRAL = 0,
    STORE_LOAD_ALIAS_PATTERN_FORWARD = 1,
    STORE_LOAD_ALIAS_PATTERN_REVERSE = 2,
    STORE_LOAD_ALIAS_PATTERN_SHUFFLE = 3,
    STORE_LOAD_ALIAS_PATTERN_SENTINEL = 4
};

struct store_load_alias_sequence_spec {
    const char *name;
    enum store_load_alias_pattern_kind history_pattern;
    unsigned int history_loops;
};

enum prefetch_stream_kind {
    PREFETCH_STREAM_NEUTRAL = 0,
    PREFETCH_STREAM_FORWARD = 1,
    PREFETCH_STREAM_REVERSE = 2,
    PREFETCH_STREAM_SHUFFLE = 3,
    PREFETCH_STREAM_SENTINEL = 4
};

struct prefetch_stream_sequence_spec {
    const char *name;
    enum prefetch_stream_kind history_pattern;
    unsigned int history_loops;
};

enum process_lifecycle_kind {
    PROCESS_LIFECYCLE_NEUTRAL = 0,
    PROCESS_LIFECYCLE_FORWARD = 1,
    PROCESS_LIFECYCLE_REVERSE = 2,
    PROCESS_LIFECYCLE_SHUFFLE = 3
};

struct process_lifecycle_sequence_spec {
    const char *name;
    enum process_lifecycle_kind source_pattern;
    unsigned int source_loops;
};

enum code_footprint_kind {
    CODE_FOOTPRINT_NEUTRAL = 0,
    CODE_FOOTPRINT_FORWARD = 1,
    CODE_FOOTPRINT_REVERSE = 2,
    CODE_FOOTPRINT_SHUFFLE = 3,
    CODE_FOOTPRINT_SENTINEL = 4
};

struct code_footprint_sequence_spec {
    const char *name;
    enum code_footprint_kind history_pattern;
    unsigned int history_loops;
};

enum return_stack_kind {
    RETURN_STACK_NEUTRAL = 0,
    RETURN_STACK_FORWARD = 1,
    RETURN_STACK_REVERSE = 2,
    RETURN_STACK_SHUFFLE = 3,
    RETURN_STACK_SENTINEL = 4
};

struct return_stack_sequence_spec {
    const char *name;
    enum return_stack_kind history_pattern;
    unsigned int history_loops;
};

enum fp_pipeline_kind {
    FP_PIPELINE_NEUTRAL = 0,
    FP_PIPELINE_FORWARD = 1,
    FP_PIPELINE_REVERSE = 2,
    FP_PIPELINE_SHUFFLE = 3,
    FP_PIPELINE_SENTINEL = 4
};

struct fp_pipeline_sequence_spec {
    const char *name;
    enum fp_pipeline_kind history_pattern;
    unsigned int history_loops;
};

enum page_permission_kind {
    PAGE_PERMISSION_NEUTRAL = 0,
    PAGE_PERMISSION_FORWARD = 1,
    PAGE_PERMISSION_REVERSE = 2,
    PAGE_PERMISSION_SHUFFLE = 3,
    PAGE_PERMISSION_SENTINEL = 4
};

struct page_permission_sequence_spec {
    const char *name;
    enum page_permission_kind history_pattern;
    unsigned int history_loops;
};

enum owned_recovery_kind {
    OWNED_RECOVERY_NEUTRAL = 0,
    OWNED_RECOVERY_FORWARD = 1,
    OWNED_RECOVERY_REVERSE = 2,
    OWNED_RECOVERY_SHUFFLE = 3,
    OWNED_RECOVERY_SENTINEL = 4
};

struct owned_recovery_sequence_spec {
    const char *name;
    enum owned_recovery_kind history_pattern;
    unsigned int history_loops;
};

static const struct event_def support_events[SUPPORT_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"dc_refills_l2_or_nb_all_states", 0x042, 0x1f, "core"},
    {"dc_refills_from_nb_all_states", 0x043, 0x1f, "core"},
    {"dc_refills_from_nb_owned_modified", 0x043, 0x18, "core"},
    {"dc_lines_evicted_modified", 0x044, 0x10, "core"},
    {"cache_block_commands_change_to_dirty", 0x0ea, 0x20, "northbridge"},
    {"probe_responses_dirty", 0x0ec, 0x0c, "northbridge"},
    {"locked_ops_executed", 0x024, 0x01, "core"}
};

static const struct event_def primary_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"dc_refills_from_nb_all_states", 0x043, 0x1f, "core"},
    {"cache_block_commands_change_to_dirty", 0x0ea, 0x20, "northbridge"},
    {"probe_responses_dirty", 0x0ec, 0x0c, "northbridge"}
};

static const struct event_def fallback_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"dc_refills_l2_or_nb_all_states", 0x042, 0x1f, "core"},
    {"dc_refills_from_nb_all_states", 0x043, 0x1f, "core"},
    {"dc_lines_evicted_modified", 0x044, 0x10, "core"}
};

static const struct event_def branch_history_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"retired_instructions", 0x0c0, 0x00, "core"},
    {"retired_branch_instructions", 0x0c2, 0x00, "core"},
    {"retired_mispredicted_branch_instructions", 0x0c3, 0x00, "core"}
};

static const struct event_def locked_history_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"locked_ops_executed", 0x024, 0x01, "core"},
    {"cache_block_commands_change_to_dirty", 0x0ea, 0x20, "northbridge"},
    {"probe_responses_dirty", 0x0ec, 0x0c, "northbridge"}
};

static const struct event_def translation_history_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"retired_instructions", 0x0c0, 0x00, "core"},
    {"cache_references", 0x07d, 0x07, "core"},
    {"cache_misses", 0x07e, 0x07, "core"}
};

static const struct event_def prefetch_stream_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"retired_instructions", 0x0c0, 0x00, "core"},
    {"cache_references", 0x07d, 0x07, "core"},
    {"cache_misses", 0x07e, 0x07, "core"}
};

static const struct event_def code_footprint_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"retired_instructions", 0x0c0, 0x00, "core"},
    {"cache_references", 0x07d, 0x07, "core"},
    {"cache_misses", 0x07e, 0x07, "core"}
};

static const struct event_def return_stack_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"retired_instructions", 0x0c0, 0x00, "core"},
    {"retired_branch_instructions", 0x0c2, 0x00, "core"},
    {"retired_mispredicted_branch_instructions", 0x0c3, 0x00, "core"}
};

static const struct event_def fp_pipeline_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"retired_instructions", 0x0c0, 0x00, "core"},
    {"cache_references", 0x07d, 0x07, "core"},
    {"cache_misses", 0x07e, 0x07, "core"}
};

static const struct event_def page_permission_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"retired_instructions", 0x0c0, 0x00, "core"},
    {"cache_references", 0x07d, 0x07, "core"},
    {"cache_misses", 0x07e, 0x07, "core"}
};

static const struct event_def owned_recovery_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"retired_instructions", 0x0c0, 0x00, "core"},
    {"cache_references", 0x07d, 0x07, "core"},
    {"cache_misses", 0x07e, 0x07, "core"}
};

struct read_group_payload {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    struct {
        uint64_t value;
        uint64_t id;
    } values[MAX_GROUP_EVENTS];
};

struct read_single_payload {
    uint64_t value;
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t id;
};

struct single_event_result {
    bool opened;
    bool unmultiplexed;
    int open_errno;
    int read_rc;
    uint64_t value;
    uint64_t id;
    uint64_t time_enabled;
    uint64_t time_running;
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

static uint64_t raw_config(unsigned int event_select, unsigned int unit_mask) {
    return (uint64_t)(event_select & 0xffu) |
           ((uint64_t)(unit_mask & 0xffu) << 8) |
           ((uint64_t)(event_select & 0xf00u) << 24);
}

static int self_test(void) {
    if (raw_config(0x076u, 0x00u) != 0x76ull) return 1;
    if (raw_config(0x0eau, 0x20u) != 0x20eaull) return 1;
    if (raw_config(0x0ecu, 0x0cu) != 0x0cecull) return 1;
    if (raw_config(0x1abu, 0xcdu) != 0x10000cdabull) return 1;
    printf("F10_PMC_FIRST_LIGHT_WORKER_SELF_TEST_OK\n");
    return 0;
}

static void fill_attr(struct perf_event_attr *attr, const struct event_def *event, bool disabled) {
    memset(attr, 0, sizeof(*attr));
    attr->type = PERF_TYPE_RAW;
    attr->size = sizeof(*attr);
    attr->config = raw_config(event->event_select, event->unit_mask);
    attr->disabled = disabled ? 1u : 0u;
    attr->exclude_kernel = 1u;
    attr->exclude_hv = 1u;
    attr->read_format = PERF_FORMAT_GROUP |
        PERF_FORMAT_TOTAL_TIME_ENABLED |
        PERF_FORMAT_TOTAL_TIME_RUNNING |
        PERF_FORMAT_ID;
}

static void fill_ibs_attr(
    struct perf_event_attr *attr,
    unsigned int pmu_type,
    uint64_t config,
    unsigned int precise_ip,
    bool disabled
) {
    memset(attr, 0, sizeof(*attr));
    attr->type = pmu_type;
    attr->size = sizeof(*attr);
    attr->config = config;
    attr->sample_period = 4096u;
    attr->sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME;
    attr->disabled = disabled ? 1u : 0u;
    attr->exclude_kernel = 1u;
    attr->exclude_hv = 1u;
    attr->precise_ip = precise_ip;
    attr->read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
        PERF_FORMAT_TOTAL_TIME_RUNNING |
        PERF_FORMAT_ID;
}

static int read_uint_sysfs(const char *path, unsigned int *value_out) {
    FILE *in = fopen(path, "r");
    if (!in) return -errno;
    unsigned int value = 0;
    int matched = fscanf(in, "%u", &value);
    int saved = ferror(in) ? errno : 0;
    fclose(in);
    if (matched != 1) return saved ? -saved : -EINVAL;
    *value_out = value;
    return 0;
}

static int pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t fnv1a64(const unsigned char *data, size_t len) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static unsigned char carrier_pattern_byte(size_t index) {
    return (unsigned char)((index * 131u + 17u) & 0xffu);
}

static uint64_t carrier_pattern_u64(size_t index) {
    uint64_t value = 0;
    for (size_t offset = 0; offset < sizeof(uint64_t); offset++) {
        value |= ((uint64_t)carrier_pattern_byte(index + offset)) << (offset * 8u);
    }
    return value;
}

static void init_carrier(struct carrier *carrier) {
    for (size_t i = 0; i < carrier->byte_count; i++) {
        carrier->bytes[i] = carrier_pattern_byte(i);
    }
}

static void prefault_carrier(struct carrier *carrier) {
    volatile unsigned char sink = 0;
    for (size_t line = 0; line < carrier->line_count; line++) {
        unsigned char *p = carrier->bytes + line * CACHE_LINE_BYTES;
        sink ^= *p;
        *p = (unsigned char)(*p ^ 0u);
    }
    if (sink == 255u) {
        carrier->bytes[0] ^= sink;
    }
}

__attribute__((noinline, noclone))
static void restore_on_core(struct carrier *carrier, int core) {
    if (pin_to_core(core) != 0) return;
    for (size_t line = 0; line < carrier->line_count; line++) {
        size_t base = line * CACHE_LINE_BYTES;
        for (size_t offset = 0; offset < CACHE_LINE_BYTES; offset++) {
            volatile unsigned char *p = (volatile unsigned char *)(void *)(carrier->bytes + base + offset);
            *p = carrier_pattern_byte(base + offset);
        }
    }
}

static void home_core_restore(struct carrier *carrier) {
    restore_on_core(carrier, CATCAS_CORE_A);
}

__attribute__((noinline, noclone))
static void same_core_prefetchw(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            void *p = carrier->bytes + line * CACHE_LINE_BYTES;
            __asm__ __volatile__("prefetchw (%0)" : : "r"(p) : "memory");
        }
    }
}

__attribute__((noinline, noclone))
static void same_core_locked_noop(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            __asm__ __volatile__("lock orq $0, %0" : "+m"(*p) : : "memory", "cc");
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void same_core_store_same_value(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            uint64_t value = *p;
            *p = value;
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void remote_read_shared(struct carrier *carrier) {
    volatile uint64_t sink = 0;
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            sink += *p;
        }
    }
    if (sink == 0x12345678ull) carrier->bytes[0] ^= 1u;
}

__attribute__((noinline, noclone))
static void remote_prefetchw(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            void *p = carrier->bytes + line * CACHE_LINE_BYTES;
            __asm__ __volatile__("prefetchw (%0)" : : "r"(p) : "memory");
        }
    }
}

__attribute__((noinline, noclone))
static void remote_locked_noop(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            __asm__ __volatile__("lock orq $0, %0" : "+m"(*p) : : "memory", "cc");
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void remote_store_same_value(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            uint64_t value = *p;
            *p = value;
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void store_same_value_subset_on_core(struct carrier *carrier, int core, int line_set) {
    if (pin_to_core(core) != 0) return;
    size_t start = line_set == 0 ? 0u : 1u;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = start; line < carrier->line_count; line += 2u) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            uint64_t value = *p;
            *p = value;
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void read_subset_on_core(struct carrier *carrier, int core, int line_set) {
    volatile uint64_t sink = 0;
    if (pin_to_core(core) != 0) return;
    size_t start = line_set == 0 ? 0u : 1u;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = start; line < carrier->line_count; line += 2u) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            sink += *p;
        }
    }
    if (sink == 0x12345678ull) carrier->bytes[0] ^= 1u;
}

__attribute__((noinline, noclone))
static void locked_noop_subset_on_core(struct carrier *carrier, int core, int line_set) {
    if (pin_to_core(core) != 0) return;
    size_t start = line_set == 0 ? 0u : 1u;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = start; line < carrier->line_count; line += 2u) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            __asm__ __volatile__("lock orq $0, %0" : "+m"(*p) : : "memory", "cc");
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void read_all_on_core(struct carrier *carrier, int core) {
    volatile uint64_t sink = 0;
    if (pin_to_core(core) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            sink += *p;
        }
    }
    if (sink == 0xabcdef01ull) carrier->bytes[0] ^= 1u;
}

__attribute__((noinline, noclone))
static void store_same_value_all_on_core(struct carrier *carrier, int core) {
    if (pin_to_core(core) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            uint64_t value = *p;
            *p = value;
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void flush_all_on_core(struct carrier *carrier, int core) {
    if (pin_to_core(core) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            void *p = carrier->bytes + line * CACHE_LINE_BYTES;
            __asm__ __volatile__("clflush (%0)" : : "r"(p) : "memory");
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void nt_store_same_value_all_on_core(struct carrier *carrier, int core) {
    if (pin_to_core(core) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            size_t base = line * CACHE_LINE_BYTES;
            uint64_t value = carrier_pattern_u64(base);
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + base);
            __asm__ __volatile__("movnti %1, %0" : "=m"(*p) : "r"(value) : "memory");
        }
    }
    __asm__ __volatile__("sfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void store_same_value_rotated_span_on_core(struct carrier *carrier, int core, size_t line_count, size_t start_bank) {
    if (pin_to_core(core) != 0) return;
    if (line_count > carrier->line_count) line_count = carrier->line_count;
    size_t bank_lines = PHASE_LOCAL_BANK_LINES;
    if (bank_lines == 0u || bank_lines > carrier->line_count) bank_lines = carrier->line_count;
    size_t start_line = (start_bank * bank_lines) % carrier->line_count;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t idx = 0; idx < line_count; idx++) {
            size_t line = (start_line + idx) % carrier->line_count;
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            uint64_t value = *p;
            *p = value;
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

static int run_path_step_operator(struct carrier *carrier, const struct path_step *step) {
    if (step->line_set != 0 && step->line_set != 1) return -1;
    if (step->op == PATH_REMOTE_READ) {
        read_subset_on_core(carrier, CATCAS_CORE_B, step->line_set);
        return 0;
    }
    if (step->op == PATH_REMOTE_STORE) {
        store_same_value_subset_on_core(carrier, CATCAS_CORE_B, step->line_set);
        return 0;
    }
    if (step->op == PATH_HOME_STORE) {
        store_same_value_subset_on_core(carrier, CATCAS_CORE_A, step->line_set);
        return 0;
    }
    return -1;
}

static int path_step_actor_core(const struct path_step *step) {
    if (step->op == PATH_REMOTE_READ) return CATCAS_CORE_B;
    if (step->op == PATH_REMOTE_STORE) return CATCAS_CORE_B;
    if (step->op == PATH_HOME_STORE) return CATCAS_CORE_A;
    return -1;
}

__attribute__((unused))
static const char *route_op_name(enum route_op op) {
    if (op == ROUTE_IDENTITY) return "identity";
    if (op == ROUTE_READ) return "remote_read";
    if (op == ROUTE_STORE) return "remote_store_same_value";
    return "unknown";
}

__attribute__((unused))
static int run_route_operator(struct carrier *carrier, enum route_op op, int remote_core) {
    if (op == ROUTE_IDENTITY) return 0;
    if (op == ROUTE_READ) {
        read_all_on_core(carrier, remote_core);
        return 0;
    }
    if (op == ROUTE_STORE) {
        store_same_value_all_on_core(carrier, remote_core);
        return 0;
    }
    return -1;
}

static const char *wc_flush_op_name(enum wc_flush_op op) {
    if (op == WC_OP_IDENTITY) return "identity";
    if (op == WC_OP_FLUSH_ONLY) return "flush_only";
    if (op == WC_OP_NORMAL_STORE_SAME) return "normal_store_same_value";
    if (op == WC_OP_NT_STORE_SAME) return "nt_store_same_value";
    if (op == WC_OP_FLUSH_THEN_NT_STORE) return "flush_then_nt_store";
    if (op == WC_OP_NT_STORE_THEN_FLUSH) return "nt_store_then_flush";
    return "unknown";
}

static int run_wc_flush_operator(struct carrier *carrier, enum wc_flush_op op) {
    if (op == WC_OP_IDENTITY) return 0;
    if (op == WC_OP_FLUSH_ONLY) {
        flush_all_on_core(carrier, CATCAS_CORE_B);
        return 0;
    }
    if (op == WC_OP_NORMAL_STORE_SAME) {
        store_same_value_all_on_core(carrier, CATCAS_CORE_B);
        return 0;
    }
    if (op == WC_OP_NT_STORE_SAME) {
        nt_store_same_value_all_on_core(carrier, CATCAS_CORE_B);
        return 0;
    }
    if (op == WC_OP_FLUSH_THEN_NT_STORE) {
        flush_all_on_core(carrier, CATCAS_CORE_B);
        nt_store_same_value_all_on_core(carrier, CATCAS_CORE_B);
        return 0;
    }
    if (op == WC_OP_NT_STORE_THEN_FLUSH) {
        nt_store_same_value_all_on_core(carrier, CATCAS_CORE_B);
        flush_all_on_core(carrier, CATCAS_CORE_B);
        return 0;
    }
    return -1;
}

static const char *eviction_prep_op_name(enum eviction_prep_op op) {
    if (op == EVICT_PREP_NONE) return "none";
    if (op == EVICT_PREP_HOME_READ) return "home_read_eviction";
    if (op == EVICT_PREP_REMOTE_READ) return "remote_read_eviction";
    if (op == EVICT_PREP_HOME_WRITE) return "home_write_eviction";
    if (op == EVICT_PREP_REMOTE_WRITE) return "remote_write_eviction";
    if (op == EVICT_PREP_HOME_THEN_REMOTE_READ) return "home_then_remote_read_eviction";
    if (op == EVICT_PREP_REMOTE_THEN_HOME_READ) return "remote_then_home_read_eviction";
    return "unknown";
}

__attribute__((noinline, noclone))
static void eviction_read_sweep_on_core(struct carrier *eviction, int core) {
    volatile uint64_t sink = 0;
    if (pin_to_core(core) != 0) return;
    for (int iter = 0; iter < EVICTION_ITERATIONS; iter++) {
        for (size_t line = 0; line < eviction->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(eviction->bytes + line * CACHE_LINE_BYTES);
            sink += *p;
        }
    }
    if (sink == 0x456789abull) eviction->bytes[0] ^= 1u;
}

__attribute__((noinline, noclone))
static void eviction_write_sweep_on_core(struct carrier *eviction, int core) {
    if (pin_to_core(core) != 0) return;
    for (int iter = 0; iter < EVICTION_ITERATIONS; iter++) {
        for (size_t line = 0; line < eviction->line_count; line++) {
            size_t base = line * CACHE_LINE_BYTES;
            volatile uint64_t *p = (volatile uint64_t *)(void *)(eviction->bytes + base);
            *p = carrier_pattern_u64(base);
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

static int run_eviction_prep_operator(struct carrier *eviction, enum eviction_prep_op op) {
    if (op == EVICT_PREP_NONE) return 0;
    if (op == EVICT_PREP_HOME_READ) {
        eviction_read_sweep_on_core(eviction, CATCAS_CORE_A);
        return 0;
    }
    if (op == EVICT_PREP_REMOTE_READ) {
        eviction_read_sweep_on_core(eviction, CATCAS_CORE_B);
        return 0;
    }
    if (op == EVICT_PREP_HOME_WRITE) {
        eviction_write_sweep_on_core(eviction, CATCAS_CORE_A);
        return 0;
    }
    if (op == EVICT_PREP_REMOTE_WRITE) {
        eviction_write_sweep_on_core(eviction, CATCAS_CORE_B);
        return 0;
    }
    if (op == EVICT_PREP_HOME_THEN_REMOTE_READ) {
        eviction_read_sweep_on_core(eviction, CATCAS_CORE_A);
        eviction_read_sweep_on_core(eviction, CATCAS_CORE_B);
        return 0;
    }
    if (op == EVICT_PREP_REMOTE_THEN_HOME_READ) {
        eviction_read_sweep_on_core(eviction, CATCAS_CORE_B);
        eviction_read_sweep_on_core(eviction, CATCAS_CORE_A);
        return 0;
    }
    return -1;
}

static void idle_pause(void) {
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 100000000L;
    nanosleep(&ts, NULL);
}

__attribute__((noinline, noclone))
static void core4_read_sweep(struct carrier *carrier) {
    volatile uint64_t sink = 0;
    if (pin_to_core(CATCAS_CORE_A) != 0) return;
    for (int iter = 0; iter < READ_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            sink += *p;
        }
    }
    if (sink == 0xfeedfaceull) carrier->bytes[0] ^= 1u;
}

__attribute__((noinline, noclone))
static void core4_write_sweep(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_A) != 0) return;
    for (int iter = 0; iter < WRITE_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            *p += 1u;
        }
    }
}

static void *pingpong_core_a(void *arg) {
    struct pingpong_context *ctx = (struct pingpong_context *)arg;
    if (pin_to_core(CATCAS_CORE_A) != 0) {
        atomic_store(&ctx->done, 1);
        return NULL;
    }
    atomic_fetch_add(&ctx->ready, 1);
    for (int iter = 0; iter < PINGPONG_ITERATIONS; iter++) {
        while (atomic_load_explicit(&ctx->turn, memory_order_acquire) != 0) {
            if (atomic_load_explicit(&ctx->done, memory_order_acquire) != 0) {
                return NULL;
            }
            sched_yield();
        }
        if (atomic_load_explicit(&ctx->done, memory_order_acquire) != 0) {
            return NULL;
        }
        for (size_t line = 0; line < ctx->carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(ctx->carrier->bytes + line * CACHE_LINE_BYTES);
            *p += 3u;
            ctx->sink_a += *p;
        }
        atomic_store_explicit(&ctx->turn, 1, memory_order_release);
    }
    return NULL;
}

static void *pingpong_core_b(void *arg) {
    struct pingpong_context *ctx = (struct pingpong_context *)arg;
    if (pin_to_core(CATCAS_CORE_B) != 0) {
        atomic_store(&ctx->done, 1);
        return NULL;
    }
    atomic_fetch_add(&ctx->ready, 1);
    for (int iter = 0; iter < PINGPONG_ITERATIONS; iter++) {
        while (atomic_load_explicit(&ctx->turn, memory_order_acquire) != 1) {
            if (atomic_load_explicit(&ctx->done, memory_order_acquire) != 0) {
                return NULL;
            }
            sched_yield();
        }
        if (atomic_load_explicit(&ctx->done, memory_order_acquire) != 0) {
            return NULL;
        }
        for (size_t line = 0; line < ctx->carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(ctx->carrier->bytes + line * CACHE_LINE_BYTES);
            *p += 5u;
            ctx->sink_b += *p;
        }
        atomic_store_explicit(&ctx->turn, 0, memory_order_release);
    }
    return NULL;
}

__attribute__((noinline, noclone))
static int cross_core_pingpong_write(struct carrier *carrier) {
    pthread_t a;
    pthread_t b;
    struct pingpong_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.carrier = carrier;
    atomic_init(&ctx.turn, 0);
    atomic_init(&ctx.ready, 0);
    atomic_init(&ctx.done, 0);
    if (pthread_create(&a, NULL, pingpong_core_a, &ctx) != 0) return -1;
    if (pthread_create(&b, NULL, pingpong_core_b, &ctx) != 0) {
        atomic_store_explicit(&ctx.done, 1, memory_order_release);
        pthread_join(a, NULL);
        return -1;
    }
    while (atomic_load(&ctx.ready) < 2 && atomic_load(&ctx.done) == 0) {
        sched_yield();
    }
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    return atomic_load(&ctx.done) == 0 ? 0 : -1;
}

static const char *coherence_operator_name(enum coherence_operator op) {
    switch (op) {
        case OP_IDENTITY: return "identity_home_prepared";
        case OP_SAME_CORE_PREFETCHW: return "same_core_prefetchw_control";
        case OP_REMOTE_READ_SHARED: return "remote_read_shared";
        case OP_REMOTE_PREFETCHW: return "remote_prefetchw_ownership_request";
        case OP_SAME_CORE_LOCKED_NOOP: return "same_core_locked_logical_noop_control";
        case OP_REMOTE_LOCKED_NOOP: return "remote_locked_logical_noop";
        case OP_SAME_CORE_STORE_SAME_VALUE: return "same_core_store_same_value_control";
        case OP_REMOTE_STORE_SAME_VALUE: return "remote_store_same_value";
        default: return "unknown_operator";
    }
}

static void *remote_operator_thread(void *arg) {
    struct operator_context *ctx = (struct operator_context *)arg;
    atomic_store_explicit(&ctx->ready, 1, memory_order_release);
    if (ctx->op == OP_REMOTE_READ_SHARED) {
        remote_read_shared(ctx->carrier);
    } else if (ctx->op == OP_REMOTE_PREFETCHW) {
        remote_prefetchw(ctx->carrier);
    } else if (ctx->op == OP_REMOTE_LOCKED_NOOP) {
        remote_locked_noop(ctx->carrier);
    } else if (ctx->op == OP_REMOTE_STORE_SAME_VALUE) {
        remote_store_same_value(ctx->carrier);
    } else {
        atomic_store_explicit(&ctx->done, 1, memory_order_release);
        return NULL;
    }
    atomic_store_explicit(&ctx->done, 1, memory_order_release);
    return NULL;
}

static int run_coherence_operator(struct carrier *carrier, enum coherence_operator op) {
    if (op == OP_IDENTITY) return 0;
    if (op == OP_SAME_CORE_PREFETCHW) {
        same_core_prefetchw(carrier);
        return 0;
    }
    if (op == OP_SAME_CORE_LOCKED_NOOP) {
        same_core_locked_noop(carrier);
        return 0;
    }
    if (op == OP_SAME_CORE_STORE_SAME_VALUE) {
        same_core_store_same_value(carrier);
        return 0;
    }
    if (op == OP_REMOTE_READ_SHARED || op == OP_REMOTE_PREFETCHW ||
        op == OP_REMOTE_LOCKED_NOOP || op == OP_REMOTE_STORE_SAME_VALUE) {
        pthread_t thread;
        struct operator_context ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.carrier = carrier;
        ctx.op = op;
        atomic_init(&ctx.ready, 0);
        atomic_init(&ctx.done, 0);
        if (pthread_create(&thread, NULL, remote_operator_thread, &ctx) != 0) return -1;
        while (atomic_load_explicit(&ctx.ready, memory_order_acquire) == 0) {
            sched_yield();
        }
        pthread_join(thread, NULL);
        return atomic_load_explicit(&ctx.done, memory_order_acquire) == 1 ? 0 : -1;
    }
    return -1;
}

static int open_single_event(const struct event_def *event, int core, uint64_t *id_out) {
    struct perf_event_attr attr;
    fill_attr(&attr, event, true);
    int fd = (int)perf_event_open_call(&attr, -1, core, -1, PERF_FLAG_FD_CLOEXEC);
    if (fd < 0) return -errno;
    uint64_t id = 0;
    if (ioctl(fd, PERF_EVENT_IOC_ID, &id) == 0 && id_out) *id_out = id;
    close(fd);
    return 0;
}

static int open_group(const struct event_def group[MAX_GROUP_EVENTS], int core, int fds[MAX_GROUP_EVENTS], uint64_t ids[MAX_GROUP_EVENTS]) {
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        fds[i] = -1;
        ids[i] = 0;
    }
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        struct perf_event_attr attr;
        fill_attr(&attr, &group[i], i == 0);
        int group_fd = i == 0 ? -1 : fds[0];
        fds[i] = (int)perf_event_open_call(&attr, -1, core, group_fd, PERF_FLAG_FD_CLOEXEC);
        if (fds[i] < 0) {
            int saved = errno;
            for (int j = 0; j < i; j++) close(fds[j]);
            return -saved;
        }
        if (ioctl(fds[i], PERF_EVENT_IOC_ID, &ids[i]) != 0) {
            int saved = errno;
            for (int j = 0; j <= i; j++) close(fds[j]);
            return -saved;
        }
    }
    return 0;
}

static int read_group(int leader_fd, struct group_result *result) {
    struct read_group_payload payload;
    memset(&payload, 0, sizeof(payload));
    ssize_t got = read(leader_fd, &payload, sizeof(payload));
    if (got < 0) return -errno;
    if ((size_t)got < sizeof(uint64_t) * 3u) return -EIO;
    if (payload.nr != MAX_GROUP_EVENTS) return -EIO;
    result->time_enabled = payload.time_enabled;
    result->time_running = payload.time_running;
    result->unmultiplexed = payload.time_enabled == payload.time_running;
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        result->events[i].value = payload.values[i].value;
        result->events[i].id = payload.values[i].id;
    }
    return 0;
}

static int read_single_event(int fd, struct single_event_result *result) {
    struct read_single_payload payload;
    memset(&payload, 0, sizeof(payload));
    ssize_t got = read(fd, &payload, sizeof(payload));
    if (got < 0) return -errno;
    if ((size_t)got < sizeof(payload)) return -EIO;
    result->value = payload.value;
    result->time_enabled = payload.time_enabled;
    result->time_running = payload.time_running;
    result->id = payload.id;
    result->unmultiplexed = payload.time_enabled == payload.time_running;
    return 0;
}

static int measure_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const char *window_name,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    init_carrier(carrier);
    prefault_carrier(carrier);
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, CATCAS_CORE_A, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int work_rc = 0;
    if (strcmp(window_name, "idle_pause") == 0) {
        idle_pause();
    } else if (strcmp(window_name, "core4_read_sweep") == 0) {
        core4_read_sweep(carrier);
    } else if (strcmp(window_name, "core4_write_sweep") == 0) {
        core4_write_sweep(carrier);
    } else if (strcmp(window_name, "cross_core_pingpong_write") == 0) {
        work_rc = cross_core_pingpong_write(carrier);
    } else {
        work_rc = -1;
    }
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (work_rc != 0) return -EIO;
    *duration_ns = end >= start ? end - start : 0;
    init_carrier(carrier);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static int measure_coherence_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    enum coherence_operator op,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    int prep_core = (op == OP_SAME_CORE_PREFETCHW ||
        op == OP_SAME_CORE_LOCKED_NOOP ||
        op == OP_SAME_CORE_STORE_SAME_VALUE) ? CATCAS_CORE_B : CATCAS_CORE_A;
    restore_on_core(carrier, prep_core);
    prefault_carrier(carrier);
    restore_on_core(carrier, prep_core);
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int work_rc = run_coherence_operator(carrier, op);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (work_rc != 0) return -EIO;
    *duration_ns = end >= start ? end - start : 0;
    home_core_restore(carrier);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static int measure_path_step(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct path_step *step,
    int observed_core,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, observed_core, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int work_rc = run_path_step_operator(carrier, step);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (work_rc != 0) return -EIO;
    *duration_ns = end >= start ? end - start : 0;
    *digest_after = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static int apply_history_sequence(struct carrier *carrier, const struct history_sequence_spec *sequence) {
    if (!sequence) return 0;
    for (size_t index = 0; index < sequence->step_count; index++) {
        if (run_path_step_operator(carrier, &sequence->steps[index]) != 0) {
            return -1;
        }
    }
    return 0;
}

static int measure_history_sentinel_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct history_sequence_spec *sequence,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_history,
    uint64_t *digest_before_sentinel,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    init_carrier(carrier);
    prefault_carrier(carrier);
    restore_on_core(carrier, CATCAS_CORE_A);
    if (apply_history_sequence(carrier, sequence) != 0) return -EIO;
    *digest_after_history = fnv1a64(carrier->bytes, carrier->byte_count);
    restore_on_core(carrier, CATCAS_CORE_A);
    *digest_before_sentinel = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int work_rc = run_coherence_operator(carrier, OP_REMOTE_STORE_SAME_VALUE);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (work_rc != 0) return -EIO;
    *duration_ns = end >= start ? end - start : 0;
    restore_on_core(carrier, CATCAS_CORE_A);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

__attribute__((unused))
static int measure_route_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct route_spec *route,
    enum route_op op,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    restore_on_core(carrier, route->home_core);
    prefault_carrier(carrier);
    restore_on_core(carrier, route->home_core);
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, route->remote_core, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int work_rc = run_route_operator(carrier, op, route->remote_core);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (work_rc != 0) return -EIO;
    *duration_ns = end >= start ? end - start : 0;
    restore_on_core(carrier, route->home_core);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static uint64_t event_value_by_name(const struct group_result *result, const struct event_def group[MAX_GROUP_EVENTS], const char *name);

static long double normalized_event_value(
    const struct group_result *result,
    const struct event_def group[MAX_GROUP_EVENTS],
    const char *name
) {
    uint64_t cycles = event_value_by_name(result, group, "cpu_cycles_not_halted");
    if (cycles == 0) cycles = 1;
    return (long double)event_value_by_name(result, group, name) / (long double)cycles;
}

static long double path_signed_area(
    const struct group_result *results,
    const struct event_def group[MAX_GROUP_EVENTS],
    int count
) {
    long double area = 0.0L;
    for (int i = 0; i < count; i++) {
        int next = (i + 1) % count;
        long double x0 = normalized_event_value(&results[i], group, "cache_block_commands_change_to_dirty");
        long double y0 = normalized_event_value(&results[i], group, "probe_responses_dirty");
        long double x1 = normalized_event_value(&results[next], group, "cache_block_commands_change_to_dirty");
        long double y1 = normalized_event_value(&results[next], group, "probe_responses_dirty");
        area += x0 * y1 - y0 * x1;
    }
    return area * 0.5L;
}

static long double abs_ld(long double value) {
    return value < 0.0L ? -value : value;
}

__attribute__((unused))
static long double route_vector_distance2(
    const struct group_result *a,
    const struct group_result *b,
    const struct event_def group[MAX_GROUP_EVENTS]
) {
    long double ax = normalized_event_value(a, group, "cache_block_commands_change_to_dirty");
    long double ay = normalized_event_value(a, group, "probe_responses_dirty");
    long double bx = normalized_event_value(b, group, "cache_block_commands_change_to_dirty");
    long double by = normalized_event_value(b, group, "probe_responses_dirty");
    long double dx = ax - bx;
    long double dy = ay - by;
    return dx * dx + dy * dy;
}

static const char *json_bool(bool value) {
    return value ? "true" : "false";
}

static void print_event_def(FILE *out, const struct event_def *event) {
    fprintf(
        out,
        "{\"name\":\"%s\",\"event_select_hex\":\"0x%03x\",\"unit_mask_hex\":\"0x%02x\",\"raw_config_hex\":\"0x%llx\",\"kind\":\"%s\"}",
        event->name,
        event->event_select,
        event->unit_mask,
        (unsigned long long)raw_config(event->event_select, event->unit_mask),
        event->kind
    );
}

static void print_group_defs(FILE *out, const struct event_def group[MAX_GROUP_EVENTS]) {
    fprintf(out, "[");
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        if (i) fprintf(out, ",");
        print_event_def(out, &group[i]);
    }
    fprintf(out, "]");
}

static void print_group_result(FILE *out, const struct event_def group[MAX_GROUP_EVENTS], const struct group_result *result) {
    fprintf(
        out,
        "{\"opened\":%s,\"unmultiplexed\":%s,\"open_errno\":%d,\"time_enabled\":%" PRIu64 ",\"time_running\":%" PRIu64 ",\"counts\":[",
        json_bool(result->opened),
        json_bool(result->unmultiplexed),
        result->open_errno,
        result->time_enabled,
        result->time_running
    );
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        if (i) fprintf(out, ",");
        fprintf(
            out,
            "{\"name\":\"%s\",\"id\":%" PRIu64 ",\"value\":%" PRIu64 "}",
            group[i].name,
            result->events[i].id,
            result->events[i].value
        );
    }
    fprintf(out, "]}");
}

static void print_single_event_result(FILE *out, const struct single_event_result *result) {
    fprintf(
        out,
        "{\"opened\":%s,\"unmultiplexed\":%s,\"open_errno\":%d,\"read_rc\":%d,\"id\":%" PRIu64 ",\"value\":%" PRIu64 ",\"time_enabled\":%" PRIu64 ",\"time_running\":%" PRIu64 "}",
        json_bool(result->opened),
        json_bool(result->unmultiplexed),
        result->open_errno,
        result->read_rc,
        result->id,
        result->value,
        result->time_enabled,
        result->time_running
    );
}

static uint64_t event_value_by_name(const struct group_result *result, const struct event_def group[MAX_GROUP_EVENTS], const char *name) {
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        if (strcmp(group[i].name, name) == 0) return result->events[i].value;
    }
    return 0;
}

static int ensure_dir(const char *path) {
    if (mkdir(path, 0700) == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
}

static int run_coherence_operator_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_B, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }

    const enum coherence_operator ops[] = {
        OP_IDENTITY,
        OP_SAME_CORE_PREFETCHW,
        OP_REMOTE_READ_SHARED,
        OP_REMOTE_PREFETCHW,
        OP_SAME_CORE_LOCKED_NOOP,
        OP_REMOTE_LOCKED_NOOP,
        OP_SAME_CORE_STORE_SAME_VALUE,
        OP_REMOTE_STORE_SAME_VALUE
    };
    enum { OP_WINDOW_COUNT = 8 };
    struct group_result results[OP_WINDOW_COUNT];
    uint64_t durations[OP_WINDOW_COUNT];
    uint64_t digest_before[OP_WINDOW_COUNT];
    uint64_t digest_after[OP_WINDOW_COUNT];
    int window_rc[OP_WINDOW_COUNT];
    for (int i = 0; i < OP_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_before[i] = 0;
        digest_after[i] = 0;
        if (primary_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_coherence_window(
                primary_group,
                ops[i],
                carrier,
                &results[i],
                &durations[i],
                &digest_before[i],
                &digest_after[i]
            );
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool all_restored = true;
    for (int i = 0; i < OP_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        all_restored = all_restored && digest_after[i] == initial_digest;
    }

    uint64_t identity_c2d = event_value_by_name(&results[0], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t same_prefetch_c2d = event_value_by_name(&results[1], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t remote_read_c2d = event_value_by_name(&results[2], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t remote_prefetch_c2d = event_value_by_name(&results[3], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t same_locked_c2d = event_value_by_name(&results[4], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t remote_locked_c2d = event_value_by_name(&results[5], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t same_store_c2d = event_value_by_name(&results[6], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t remote_store_c2d = event_value_by_name(&results[7], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t identity_probe = event_value_by_name(&results[0], primary_group, "probe_responses_dirty");
    uint64_t same_prefetch_probe = event_value_by_name(&results[1], primary_group, "probe_responses_dirty");
    uint64_t remote_read_probe = event_value_by_name(&results[2], primary_group, "probe_responses_dirty");
    uint64_t remote_prefetch_probe = event_value_by_name(&results[3], primary_group, "probe_responses_dirty");
    uint64_t same_locked_probe = event_value_by_name(&results[4], primary_group, "probe_responses_dirty");
    uint64_t remote_locked_probe = event_value_by_name(&results[5], primary_group, "probe_responses_dirty");
    uint64_t same_store_probe = event_value_by_name(&results[6], primary_group, "probe_responses_dirty");
    uint64_t remote_store_probe = event_value_by_name(&results[7], primary_group, "probe_responses_dirty");

    uint64_t prefetch_c2d_control = identity_c2d > same_prefetch_c2d ? identity_c2d : same_prefetch_c2d;
    uint64_t prefetch_probe_control = identity_probe > same_prefetch_probe ? identity_probe : same_prefetch_probe;
    uint64_t locked_c2d_control = same_locked_c2d > remote_read_c2d ? same_locked_c2d : remote_read_c2d;
    if (identity_c2d > locked_c2d_control) locked_c2d_control = identity_c2d;
    uint64_t locked_probe_control = same_locked_probe > remote_read_probe ? same_locked_probe : remote_read_probe;
    if (identity_probe > locked_probe_control) locked_probe_control = identity_probe;
    uint64_t store_c2d_control = same_store_c2d > remote_read_c2d ? same_store_c2d : remote_read_c2d;
    if (identity_c2d > store_c2d_control) store_c2d_control = identity_c2d;
    uint64_t store_probe_control = same_store_probe > remote_read_probe ? same_store_probe : remote_read_probe;
    if (identity_probe > store_probe_control) store_probe_control = identity_probe;

    bool prefetch_c2d_moved = remote_prefetch_c2d > prefetch_c2d_control + 32u &&
        remote_prefetch_c2d > prefetch_c2d_control * 3u;
    bool prefetch_probe_moved = remote_prefetch_probe > prefetch_probe_control + 32u &&
        remote_prefetch_probe > prefetch_probe_control * 3u;
    bool locked_c2d_moved = remote_locked_c2d > locked_c2d_control + 32u &&
        remote_locked_c2d > locked_c2d_control * 3u;
    bool locked_probe_moved = remote_locked_probe > locked_probe_control + 32u &&
        remote_locked_probe > locked_probe_control * 3u;
    bool store_c2d_moved = remote_store_c2d > store_c2d_control + 32u &&
        remote_store_c2d > store_c2d_control * 3u;
    bool store_probe_moved = remote_store_probe > store_probe_control + 32u &&
        remote_store_probe > store_probe_control * 3u;
    bool controlled_state_found = primary_open_rc == 0 && all_windows_ok && all_unmultiplexed &&
        all_restored && (prefetch_c2d_moved || prefetch_probe_moved || locked_c2d_moved ||
        locked_probe_moved || store_c2d_moved || store_probe_moved);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_COHERENCE_OPERATOR_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", COHERENCE_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", controlled_state_found ? "CONTROLLED_COHERENCE_STATE_FOUND" : "CONTROLLED_COHERENCE_STATE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Controlled coherence-operator PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"observed_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"primary\": \"cache_block_commands_change_to_dirty\", \"secondary\": \"probe_responses_dirty\", \"comparison\": \"remote ownership-intent operators greater than identity, read-shared, and same-core controls\"},\n");
    fprintf(out, "  \"operators\": [\n");
    for (int i = 0; i < OP_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"carrier_restored\": %s, \"group\": ",
            coherence_operator_name(ops[i]),
            window_rc[i],
            durations[i],
            digest_before[i],
            digest_after[i],
            json_bool(digest_after[i] == initial_digest));
        print_group_result(out, primary_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == OP_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"change_to_dirty\": {\"identity\": %" PRIu64 ", \"same_core_prefetchw\": %" PRIu64 ", \"remote_read_shared\": %" PRIu64 ", \"remote_prefetchw\": %" PRIu64 ", \"same_core_locked_noop\": %" PRIu64 ", \"remote_locked_noop\": %" PRIu64 ", \"same_core_store_same_value\": %" PRIu64 ", \"remote_store_same_value\": %" PRIu64 "},\n",
        identity_c2d, same_prefetch_c2d, remote_read_c2d, remote_prefetch_c2d, same_locked_c2d, remote_locked_c2d, same_store_c2d, remote_store_c2d);
    fprintf(out, "    \"probe_dirty\": {\"identity\": %" PRIu64 ", \"same_core_prefetchw\": %" PRIu64 ", \"remote_read_shared\": %" PRIu64 ", \"remote_prefetchw\": %" PRIu64 ", \"same_core_locked_noop\": %" PRIu64 ", \"remote_locked_noop\": %" PRIu64 ", \"same_core_store_same_value\": %" PRIu64 ", \"remote_store_same_value\": %" PRIu64 "}\n",
        identity_probe, same_prefetch_probe, remote_read_probe, remote_prefetch_probe, same_locked_probe, remote_locked_probe, same_store_probe, remote_store_probe);
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"carrier_restored\": %s, \"prefetch_change_to_dirty_moved\": %s, \"prefetch_probe_dirty_moved\": %s, \"locked_change_to_dirty_moved\": %s, \"locked_probe_dirty_moved\": %s, \"store_same_value_change_to_dirty_moved\": %s, \"store_same_value_probe_dirty_moved\": %s, \"controlled_state_found\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(all_restored),
        json_bool(prefetch_c2d_moved),
        json_bool(prefetch_probe_moved),
        json_bool(locked_c2d_moved),
        json_bool(locked_probe_moved),
        json_bool(store_c2d_moved),
        json_bool(store_probe_moved),
        json_bool(controlled_state_found));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        controlled_state_found ? "CONTROLLED_COHERENCE_STATE_FOUND" : "CONTROLLED_COHERENCE_STATE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

static uint64_t max_u64(uint64_t a, uint64_t b) {
    return a > b ? a : b;
}

static uint64_t min_u64(uint64_t a, uint64_t b) {
    return a < b ? a : b;
}

static uint64_t abs_diff_u64(uint64_t a, uint64_t b) {
    return a > b ? a - b : b - a;
}

static int run_history_sentinel_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_B, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }

    const struct path_step forward_steps[] = {
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"home_store_set1", PATH_HOME_STORE, 1},
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"home_store_set0", PATH_HOME_STORE, 0},
    };
    const struct path_step reverse_steps[] = {
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"home_store_set0", PATH_HOME_STORE, 0},
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"home_store_set1", PATH_HOME_STORE, 1},
    };
    const struct path_step shuffle_steps[] = {
        {"home_store_set0", PATH_HOME_STORE, 0},
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"home_store_set1", PATH_HOME_STORE, 1},
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
    };
    const struct history_sequence_spec sequences[] = {
        {"identity_restore_only", NULL, 0},
        {"forward_balanced_history", forward_steps, sizeof(forward_steps) / sizeof(forward_steps[0])},
        {"reverse_balanced_history", reverse_steps, sizeof(reverse_steps) / sizeof(reverse_steps[0])},
        {"shuffle_balanced_history", shuffle_steps, sizeof(shuffle_steps) / sizeof(shuffle_steps[0])},
    };
    enum { HISTORY_WINDOW_COUNT = 4 };
    struct group_result results[HISTORY_WINDOW_COUNT];
    uint64_t durations[HISTORY_WINDOW_COUNT];
    uint64_t digest_after_history[HISTORY_WINDOW_COUNT];
    uint64_t digest_before_sentinel[HISTORY_WINDOW_COUNT];
    uint64_t digest_after_restore[HISTORY_WINDOW_COUNT];
    int window_rc[HISTORY_WINDOW_COUNT];
    for (int i = 0; i < HISTORY_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_before_sentinel[i] = 0;
        digest_after_restore[i] = 0;
        if (primary_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_history_sentinel_window(
                primary_group,
                &sequences[i],
                carrier,
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_before_sentinel[i],
                &digest_after_restore[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool bytes_unchanged_after_history = true;
    bool restored_before_sentinel = true;
    bool restored_after_sentinel = true;
    for (int i = 0; i < HISTORY_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        bytes_unchanged_after_history =
            bytes_unchanged_after_history && digest_after_history[i] == initial_digest;
        restored_before_sentinel =
            restored_before_sentinel && digest_before_sentinel[i] == initial_digest;
        restored_after_sentinel =
            restored_after_sentinel && digest_after_restore[i] == initial_digest;
    }

    uint64_t identity_c2d = event_value_by_name(&results[0], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t forward_c2d = event_value_by_name(&results[1], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t reverse_c2d = event_value_by_name(&results[2], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t shuffle_c2d = event_value_by_name(&results[3], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t identity_probe = event_value_by_name(&results[0], primary_group, "probe_responses_dirty");
    uint64_t forward_probe = event_value_by_name(&results[1], primary_group, "probe_responses_dirty");
    uint64_t reverse_probe = event_value_by_name(&results[2], primary_group, "probe_responses_dirty");
    uint64_t shuffle_probe = event_value_by_name(&results[3], primary_group, "probe_responses_dirty");
    uint64_t c2d_delta = abs_diff_u64(forward_c2d, reverse_c2d);
    uint64_t probe_delta = abs_diff_u64(forward_probe, reverse_probe);
    uint64_t c2d_control = abs_diff_u64(identity_c2d, shuffle_c2d);
    uint64_t probe_control = abs_diff_u64(identity_probe, shuffle_probe);
    uint64_t c2d_threshold = max_u64(32u, 3u * c2d_control);
    uint64_t probe_threshold = max_u64(32u, 3u * probe_control);
    bool c2d_history_signal = c2d_delta > c2d_threshold;
    bool probe_history_signal = probe_delta > probe_threshold;
    bool history_sentinel_response = primary_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && bytes_unchanged_after_history &&
        restored_before_sentinel && restored_after_sentinel &&
        (c2d_history_signal || probe_history_signal);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_HISTORY_SENTINEL_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", HISTORY_SENTINEL_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", history_sentinel_response ? "HISTORY_SENTINEL_RESPONSE_FOUND" : "HISTORY_SENTINEL_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Restored history-sentinel PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"observed_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"balanced ownership-transfer sequence over same line sets\", \"restore\": \"common home-core byte restore before sentinel\", \"sentinel\": \"remote_store_same_value measured on observed core\", \"acceptance\": \"forward/reverse sentinel delta exceeds max(32, 3 * neutral/shuffle control spread) for an established coherence counter\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < HISTORY_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_step_count\": %zu, \"digest_after_history_hex\": \"0x%016" PRIx64 "\", \"digest_before_sentinel_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"bytes_unchanged_after_history\": %s, \"restored_before_sentinel\": %s, \"restored_after_sentinel\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].step_count,
            digest_after_history[i],
            digest_before_sentinel[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_before_sentinel[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, primary_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == HISTORY_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"cache_block_commands_change_to_dirty\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_c2d, forward_c2d, reverse_c2d, shuffle_c2d, c2d_delta, c2d_control, c2d_threshold, json_bool(c2d_history_signal));
    fprintf(out, "    \"probe_responses_dirty\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_probe, forward_probe, reverse_probe, shuffle_probe, probe_delta, probe_control, probe_threshold, json_bool(probe_history_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"bytes_unchanged_after_history\": %s, \"restored_before_sentinel\": %s, \"restored_after_sentinel\": %s, \"change_to_dirty_history_signal\": %s, \"probe_dirty_history_signal\": %s, \"history_sentinel_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(bytes_unchanged_after_history),
        json_bool(restored_before_sentinel),
        json_bool(restored_after_sentinel),
        json_bool(c2d_history_signal),
        json_bool(probe_history_signal),
        json_bool(history_sentinel_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        history_sentinel_response ? "HISTORY_SENTINEL_RESPONSE_FOUND" : "HISTORY_SENTINEL_RESPONSE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

static void run_locked_history_pattern(
    struct carrier *carrier,
    int core,
    enum locked_history_kind pattern,
    unsigned int loops
) {
    for (unsigned int loop = 0; loop < loops; loop++) {
        switch (pattern) {
            case LOCKED_HISTORY_NEUTRAL:
                locked_noop_subset_on_core(carrier, core, 0);
                locked_noop_subset_on_core(carrier, core, 1);
                locked_noop_subset_on_core(carrier, core, 1);
                locked_noop_subset_on_core(carrier, core, 0);
                break;
            case LOCKED_HISTORY_FORWARD:
                locked_noop_subset_on_core(carrier, core, 0);
                locked_noop_subset_on_core(carrier, core, 0);
                locked_noop_subset_on_core(carrier, core, 1);
                locked_noop_subset_on_core(carrier, core, 1);
                break;
            case LOCKED_HISTORY_REVERSE:
                locked_noop_subset_on_core(carrier, core, 1);
                locked_noop_subset_on_core(carrier, core, 1);
                locked_noop_subset_on_core(carrier, core, 0);
                locked_noop_subset_on_core(carrier, core, 0);
                break;
            case LOCKED_HISTORY_SHUFFLE:
                locked_noop_subset_on_core(carrier, core, 0);
                locked_noop_subset_on_core(carrier, core, 1);
                locked_noop_subset_on_core(carrier, core, 0);
                locked_noop_subset_on_core(carrier, core, 1);
                break;
        }
    }
}

static int measure_locked_history_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct locked_history_sequence_spec *sequence,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_history,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    init_carrier(carrier);
    prefault_carrier(carrier);
    restore_on_core(carrier, CATCAS_CORE_A);
    run_locked_history_pattern(carrier, CATCAS_CORE_A, sequence->history_pattern,
                               sequence->history_loops);
    *digest_after_history = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    run_locked_history_pattern(carrier, CATCAS_CORE_B, LOCKED_HISTORY_NEUTRAL, 1u);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    restore_on_core(carrier, CATCAS_CORE_A);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_locked_history_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    int locked_fds[MAX_GROUP_EVENTS];
    uint64_t locked_ids[MAX_GROUP_EVENTS];
    int locked_open_rc = open_group(locked_history_group, CATCAS_CORE_B,
                                    locked_fds, locked_ids);
    if (locked_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(locked_fds[i]);
    }

    const struct locked_history_sequence_spec sequences[] = {
        {"neutral_locked_history", LOCKED_HISTORY_NEUTRAL, LOCKED_HISTORY_LOOPS},
        {"forward_locked_history", LOCKED_HISTORY_FORWARD, LOCKED_HISTORY_LOOPS},
        {"reverse_locked_history", LOCKED_HISTORY_REVERSE, LOCKED_HISTORY_LOOPS},
        {"shuffle_locked_history", LOCKED_HISTORY_SHUFFLE, LOCKED_HISTORY_LOOPS},
    };
    enum { LOCKED_HISTORY_WINDOW_COUNT = 4 };
    struct group_result results[LOCKED_HISTORY_WINDOW_COUNT];
    uint64_t durations[LOCKED_HISTORY_WINDOW_COUNT];
    uint64_t digest_after_history[LOCKED_HISTORY_WINDOW_COUNT];
    uint64_t digest_after_restore[LOCKED_HISTORY_WINDOW_COUNT];
    int window_rc[LOCKED_HISTORY_WINDOW_COUNT];
    for (int i = 0; i < LOCKED_HISTORY_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        if (locked_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_locked_history_window(
                locked_history_group,
                &sequences[i],
                carrier,
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool bytes_unchanged_after_history = true;
    bool bytes_unchanged_after_restore = true;
    for (int i = 0; i < LOCKED_HISTORY_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        bytes_unchanged_after_history =
            bytes_unchanged_after_history && digest_after_history[i] == initial_digest;
        bytes_unchanged_after_restore =
            bytes_unchanged_after_restore && digest_after_restore[i] == initial_digest;
    }

    uint64_t identity_c2d = event_value_by_name(&results[0], locked_history_group, "cache_block_commands_change_to_dirty");
    uint64_t forward_c2d = event_value_by_name(&results[1], locked_history_group, "cache_block_commands_change_to_dirty");
    uint64_t reverse_c2d = event_value_by_name(&results[2], locked_history_group, "cache_block_commands_change_to_dirty");
    uint64_t shuffle_c2d = event_value_by_name(&results[3], locked_history_group, "cache_block_commands_change_to_dirty");
    uint64_t c2d_delta = abs_diff_u64(forward_c2d, reverse_c2d);
    uint64_t c2d_control = abs_diff_u64(identity_c2d, shuffle_c2d);
    uint64_t c2d_threshold = max_u64(32u, 3u * c2d_control);
    bool c2d_signal = c2d_delta > c2d_threshold;

    uint64_t identity_probe = event_value_by_name(&results[0], locked_history_group, "probe_responses_dirty");
    uint64_t forward_probe = event_value_by_name(&results[1], locked_history_group, "probe_responses_dirty");
    uint64_t reverse_probe = event_value_by_name(&results[2], locked_history_group, "probe_responses_dirty");
    uint64_t shuffle_probe = event_value_by_name(&results[3], locked_history_group, "probe_responses_dirty");
    uint64_t probe_delta = abs_diff_u64(forward_probe, reverse_probe);
    uint64_t probe_control = abs_diff_u64(identity_probe, shuffle_probe);
    uint64_t probe_threshold = max_u64(32u, 3u * probe_control);
    bool probe_signal = probe_delta > probe_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(1000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;
    bool locked_history_response = locked_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && bytes_unchanged_after_history &&
        bytes_unchanged_after_restore && (c2d_signal || probe_signal || duration_signal);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_LOCKED_HISTORY_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", LOCKED_HISTORY_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", locked_history_response ? "LOCKED_HISTORY_RESPONSE_FOUND" : "LOCKED_HISTORY_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Locked no-op history discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"source_core\": %d, \"sentinel_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false},\n");
    fprintf(out, "  \"locked_history_carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"history_loops\": %u, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, LOCKED_HISTORY_LOOPS, initial_digest);
    fprintf(out, "  \"selected_group\": \"locked_history_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", locked_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, locked_history_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"balanced locked logical no-op histories over CAT_CAS-owned line subsets on source core\", \"sentinel\": \"fixed locked logical no-op sequence on observer core\", \"primary_acceptance\": \"forward/reverse PMU or duration delta exceeds max(floor, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < LOCKED_HISTORY_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"digest_after_history_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, locked_history_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == LOCKED_HISTORY_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"cache_block_commands_change_to_dirty\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_c2d, forward_c2d, reverse_c2d, shuffle_c2d, c2d_delta, c2d_control, c2d_threshold, json_bool(c2d_signal));
    fprintf(out, "    \"probe_responses_dirty\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_probe, forward_probe, reverse_probe, shuffle_probe, probe_delta, probe_control, probe_threshold, json_bool(probe_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"change_to_dirty_signal\": %s, \"probe_dirty_signal\": %s, \"duration_signal\": %s, \"locked_history_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(bytes_unchanged_after_history),
        json_bool(bytes_unchanged_after_restore),
        json_bool(c2d_signal),
        json_bool(probe_signal),
        json_bool(duration_signal),
        json_bool(locked_history_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"locked_history_group\"}\n",
        locked_history_response ? "LOCKED_HISTORY_RESPONSE_FOUND" : "LOCKED_HISTORY_RESPONSE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

static volatile uint64_t branch_history_sink = 0;

static void fill_branch_pattern(unsigned char *pattern, size_t len, enum branch_pattern_kind kind) {
    for (size_t i = 0; i < len; i++) {
        unsigned int value = 0;
        switch (kind) {
            case BRANCH_PATTERN_NEUTRAL:
                value = (unsigned int)(((i * 0x9e3779b1u) ^ (i >> 3)) & 1u);
                break;
            case BRANCH_PATTERN_FORWARD:
                value = (unsigned int)((i & 7u) < 4u);
                break;
            case BRANCH_PATTERN_REVERSE:
                value = (unsigned int)((i & 7u) >= 4u);
                break;
            case BRANCH_PATTERN_SHUFFLE:
                value = (unsigned int)(((i & 3u) == 0u) || ((i & 3u) == 3u));
                break;
            case BRANCH_PATTERN_SENTINEL:
                value = (unsigned int)((((i * 13u) + (i >> 2) + 7u) & 15u) < 8u);
                break;
        }
        pattern[i] = (unsigned char)value;
    }
}

static unsigned char *branch_pattern_by_kind(
    unsigned char *neutral,
    unsigned char *forward,
    unsigned char *reverse,
    unsigned char *shuffle,
    unsigned char *sentinel,
    enum branch_pattern_kind kind
) {
    switch (kind) {
        case BRANCH_PATTERN_NEUTRAL:
            return neutral;
        case BRANCH_PATTERN_FORWARD:
            return forward;
        case BRANCH_PATTERN_REVERSE:
            return reverse;
        case BRANCH_PATTERN_SHUFFLE:
            return shuffle;
        case BRANCH_PATTERN_SENTINEL:
            return sentinel;
    }
    return neutral;
}

static uint64_t branch_pattern_digest(
    const unsigned char *neutral,
    const unsigned char *forward,
    const unsigned char *reverse,
    const unsigned char *shuffle,
    const unsigned char *sentinel
) {
    unsigned char combined[BRANCH_PATTERN_BYTES * 5u];
    memcpy(combined, neutral, BRANCH_PATTERN_BYTES);
    memcpy(combined + BRANCH_PATTERN_BYTES, forward, BRANCH_PATTERN_BYTES);
    memcpy(combined + (BRANCH_PATTERN_BYTES * 2u), reverse, BRANCH_PATTERN_BYTES);
    memcpy(combined + (BRANCH_PATTERN_BYTES * 3u), shuffle, BRANCH_PATTERN_BYTES);
    memcpy(combined + (BRANCH_PATTERN_BYTES * 4u), sentinel, BRANCH_PATTERN_BYTES);
    return fnv1a64(combined, sizeof(combined));
}

__attribute__((noinline))
static void run_branch_pattern(const volatile unsigned char *pattern, size_t len, unsigned int loops) {
    uint64_t acc = branch_history_sink + len + loops;
    for (unsigned int round = 0; round < loops; round++) {
        size_t offset = ((size_t)round * 17u) & (len - 1u);
        for (size_t i = 0; i < len; i++) {
            size_t index = (i + offset) & (len - 1u);
            if (pattern[index] != 0u) {
                acc += ((uint64_t)index ^ (uint64_t)round) + 0x9e3779b97f4a7c15ull;
            } else {
                acc ^= (((uint64_t)index + 1u) * 0xbf58476d1ce4e5b9ull) ^ (uint64_t)round;
            }
            __asm__ __volatile__("" : "+r"(acc) :: "memory");
        }
    }
    branch_history_sink = acc;
}

static int measure_branch_history_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct branch_sequence_spec *sequence,
    unsigned char *neutral,
    unsigned char *forward,
    unsigned char *reverse,
    unsigned char *shuffle,
    unsigned char *sentinel,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *pattern_digest_after_history,
    uint64_t *pattern_digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    if (pin_to_core(CATCAS_CORE_B) != 0) return -EINVAL;
    run_branch_pattern(neutral, BRANCH_PATTERN_BYTES, BRANCH_RESTORE_LOOPS);
    if (sequence->history_loops > 0u) {
        unsigned char *history = branch_pattern_by_kind(neutral, forward, reverse, shuffle, sentinel,
                                                        sequence->history_pattern);
        run_branch_pattern(history, BRANCH_PATTERN_BYTES, sequence->history_loops);
    }
    *pattern_digest_after_history = branch_pattern_digest(neutral, forward, reverse, shuffle, sentinel);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    run_branch_pattern(sentinel, BRANCH_PATTERN_BYTES, BRANCH_SENTINEL_LOOPS);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    run_branch_pattern(neutral, BRANCH_PATTERN_BYTES, BRANCH_RESTORE_LOOPS);
    *pattern_digest_after_restore = branch_pattern_digest(neutral, forward, reverse, shuffle, sentinel);
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_branch_history_mode(const char *output_root) {
    unsigned char *neutral = NULL;
    unsigned char *forward = NULL;
    unsigned char *reverse = NULL;
    unsigned char *shuffle = NULL;
    unsigned char *sentinel = NULL;
    if (posix_memalign((void **)&neutral, CACHE_LINE_BYTES, BRANCH_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&forward, CACHE_LINE_BYTES, BRANCH_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&reverse, CACHE_LINE_BYTES, BRANCH_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&shuffle, CACHE_LINE_BYTES, BRANCH_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&sentinel, CACHE_LINE_BYTES, BRANCH_PATTERN_BYTES) != 0) {
        free(neutral);
        free(forward);
        free(reverse);
        free(shuffle);
        free(sentinel);
        fprintf(stderr, "branch pattern allocation failed\n");
        return 1;
    }
    fill_branch_pattern(neutral, BRANCH_PATTERN_BYTES, BRANCH_PATTERN_NEUTRAL);
    fill_branch_pattern(forward, BRANCH_PATTERN_BYTES, BRANCH_PATTERN_FORWARD);
    fill_branch_pattern(reverse, BRANCH_PATTERN_BYTES, BRANCH_PATTERN_REVERSE);
    fill_branch_pattern(shuffle, BRANCH_PATTERN_BYTES, BRANCH_PATTERN_SHUFFLE);
    fill_branch_pattern(sentinel, BRANCH_PATTERN_BYTES, BRANCH_PATTERN_SENTINEL);
    uint64_t initial_digest = branch_pattern_digest(neutral, forward, reverse, shuffle, sentinel);

    int branch_fds[MAX_GROUP_EVENTS];
    uint64_t branch_ids[MAX_GROUP_EVENTS];
    int branch_open_rc = open_group(branch_history_group, CATCAS_CORE_B, branch_fds, branch_ids);
    if (branch_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(branch_fds[i]);
    }

    const struct branch_sequence_spec sequences[] = {
        {"neutral_restore_only", BRANCH_PATTERN_NEUTRAL, 0u},
        {"forward_branch_history", BRANCH_PATTERN_FORWARD, BRANCH_HISTORY_LOOPS},
        {"reverse_branch_history", BRANCH_PATTERN_REVERSE, BRANCH_HISTORY_LOOPS},
        {"shuffle_branch_history", BRANCH_PATTERN_SHUFFLE, BRANCH_HISTORY_LOOPS},
    };
    enum { BRANCH_WINDOW_COUNT = 4 };
    struct group_result results[BRANCH_WINDOW_COUNT];
    uint64_t durations[BRANCH_WINDOW_COUNT];
    uint64_t digest_after_history[BRANCH_WINDOW_COUNT];
    uint64_t digest_after_restore[BRANCH_WINDOW_COUNT];
    int window_rc[BRANCH_WINDOW_COUNT];
    for (int i = 0; i < BRANCH_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        if (branch_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_branch_history_window(
                branch_history_group,
                &sequences[i],
                neutral,
                forward,
                reverse,
                shuffle,
                sentinel,
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool patterns_unchanged_after_history = true;
    bool patterns_unchanged_after_restore = true;
    for (int i = 0; i < BRANCH_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        patterns_unchanged_after_history =
            patterns_unchanged_after_history && digest_after_history[i] == initial_digest;
        patterns_unchanged_after_restore =
            patterns_unchanged_after_restore && digest_after_restore[i] == initial_digest;
    }

    uint64_t identity_misses = event_value_by_name(&results[0], branch_history_group, "retired_mispredicted_branch_instructions");
    uint64_t forward_misses = event_value_by_name(&results[1], branch_history_group, "retired_mispredicted_branch_instructions");
    uint64_t reverse_misses = event_value_by_name(&results[2], branch_history_group, "retired_mispredicted_branch_instructions");
    uint64_t shuffle_misses = event_value_by_name(&results[3], branch_history_group, "retired_mispredicted_branch_instructions");
    uint64_t miss_delta = abs_diff_u64(forward_misses, reverse_misses);
    uint64_t miss_control = abs_diff_u64(identity_misses, shuffle_misses);
    uint64_t miss_threshold = max_u64(32u, 3u * miss_control);
    bool miss_signal = miss_delta > miss_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(1000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;
    bool branch_history_response = branch_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && patterns_unchanged_after_history &&
        patterns_unchanged_after_restore && miss_signal;

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_BRANCH_HISTORY_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", BRANCH_HISTORY_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", branch_history_response ? "BRANCH_HISTORY_RESPONSE_FOUND" : "BRANCH_HISTORY_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Branch-history PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"branch_core\": %d},\n", CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false},\n");
    fprintf(out, "  \"branch_carrier\": {\"pattern_bytes\": %u, \"restore_loops\": %u, \"history_loops\": %u, \"sentinel_loops\": %u, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        BRANCH_PATTERN_BYTES, BRANCH_RESTORE_LOOPS, BRANCH_HISTORY_LOOPS, BRANCH_SENTINEL_LOOPS, initial_digest);
    fprintf(out, "  \"selected_group\": \"branch_history_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", branch_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, branch_history_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"balanced public branch-outcome training pattern\", \"restore\": \"neutral branch-history wash before every window\", \"sentinel\": \"fixed public branch-outcome sequence measured on the same static branch site\", \"primary_acceptance\": \"forward/reverse retired_mispredicted_branch_instructions delta exceeds max(32, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < BRANCH_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"pattern_digest_after_history_hex\": \"0x%016" PRIx64 "\", \"pattern_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"patterns_unchanged_after_history\": %s, \"patterns_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, branch_history_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == BRANCH_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"retired_mispredicted_branch_instructions\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_misses, forward_misses, reverse_misses, shuffle_misses, miss_delta, miss_control, miss_threshold, json_bool(miss_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"patterns_unchanged_after_history\": %s, \"patterns_unchanged_after_restore\": %s, \"branch_miss_signal\": %s, \"duration_signal\": %s, \"branch_history_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(patterns_unchanged_after_history),
        json_bool(patterns_unchanged_after_restore),
        json_bool(miss_signal),
        json_bool(duration_signal),
        json_bool(branch_history_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"branch_history_group\"}\n",
        branch_history_response ? "BRANCH_HISTORY_RESPONSE_FOUND" : "BRANCH_HISTORY_RESPONSE_NOT_ESTABLISHED",
        result_path);
    free(neutral);
    free(forward);
    free(reverse);
    free(shuffle);
    free(sentinel);
    return 0;
}

static volatile uint64_t indirect_target_history_sink = 0;

typedef uint64_t (*indirect_target_fn)(uint64_t);

__attribute__((noinline))
static uint64_t indirect_target_0(uint64_t x) {
    return (x + 0x9e3779b97f4a7c15ull) ^ (x << 7);
}

__attribute__((noinline))
static uint64_t indirect_target_1(uint64_t x) {
    return (x + 0xbf58476d1ce4e5b9ull) ^ (x >> 5);
}

__attribute__((noinline))
static uint64_t indirect_target_2(uint64_t x) {
    return (x + 0x94d049bb133111ebull) ^ (x << 11);
}

__attribute__((noinline))
static uint64_t indirect_target_3(uint64_t x) {
    return (x + 0x2545f4914f6cdd1dull) ^ (x >> 9);
}

static indirect_target_fn volatile indirect_target_table[4] = {
    indirect_target_0,
    indirect_target_1,
    indirect_target_2,
    indirect_target_3
};

static void fill_indirect_target_pattern(
    unsigned char *pattern,
    size_t len,
    enum indirect_target_pattern_kind kind
) {
    static const unsigned char neutral_seq[8] = {0u, 1u, 2u, 3u, 3u, 2u, 1u, 0u};
    static const unsigned char forward_seq[8] = {0u, 1u, 2u, 3u, 0u, 1u, 2u, 3u};
    static const unsigned char reverse_seq[8] = {3u, 2u, 1u, 0u, 3u, 2u, 1u, 0u};
    static const unsigned char shuffle_seq[8] = {0u, 2u, 3u, 1u, 1u, 3u, 2u, 0u};
    static const unsigned char sentinel_seq[8] = {0u, 3u, 1u, 2u, 2u, 1u, 3u, 0u};
    const unsigned char *seq = neutral_seq;
    switch (kind) {
        case INDIRECT_TARGET_PATTERN_NEUTRAL:
            seq = neutral_seq;
            break;
        case INDIRECT_TARGET_PATTERN_FORWARD:
            seq = forward_seq;
            break;
        case INDIRECT_TARGET_PATTERN_REVERSE:
            seq = reverse_seq;
            break;
        case INDIRECT_TARGET_PATTERN_SHUFFLE:
            seq = shuffle_seq;
            break;
        case INDIRECT_TARGET_PATTERN_SENTINEL:
            seq = sentinel_seq;
            break;
    }
    for (size_t i = 0; i < len; i++) {
        pattern[i] = seq[(i + (i >> 5)) & 7u];
    }
}

static unsigned char *indirect_target_pattern_by_kind(
    unsigned char *neutral,
    unsigned char *forward,
    unsigned char *reverse,
    unsigned char *shuffle,
    unsigned char *sentinel,
    enum indirect_target_pattern_kind kind
) {
    switch (kind) {
        case INDIRECT_TARGET_PATTERN_NEUTRAL:
            return neutral;
        case INDIRECT_TARGET_PATTERN_FORWARD:
            return forward;
        case INDIRECT_TARGET_PATTERN_REVERSE:
            return reverse;
        case INDIRECT_TARGET_PATTERN_SHUFFLE:
            return shuffle;
        case INDIRECT_TARGET_PATTERN_SENTINEL:
            return sentinel;
    }
    return neutral;
}

static uint64_t indirect_target_pattern_digest(
    const unsigned char *neutral,
    const unsigned char *forward,
    const unsigned char *reverse,
    const unsigned char *shuffle,
    const unsigned char *sentinel
) {
    unsigned char combined[INDIRECT_TARGET_PATTERN_BYTES * 5u];
    memcpy(combined, neutral, INDIRECT_TARGET_PATTERN_BYTES);
    memcpy(combined + INDIRECT_TARGET_PATTERN_BYTES, forward, INDIRECT_TARGET_PATTERN_BYTES);
    memcpy(combined + (INDIRECT_TARGET_PATTERN_BYTES * 2u), reverse, INDIRECT_TARGET_PATTERN_BYTES);
    memcpy(combined + (INDIRECT_TARGET_PATTERN_BYTES * 3u), shuffle, INDIRECT_TARGET_PATTERN_BYTES);
    memcpy(combined + (INDIRECT_TARGET_PATTERN_BYTES * 4u), sentinel, INDIRECT_TARGET_PATTERN_BYTES);
    return fnv1a64(combined, sizeof(combined));
}

__attribute__((noinline))
static void run_indirect_target_pattern(
    const volatile unsigned char *pattern,
    size_t len,
    unsigned int loops
) {
    uint64_t acc = indirect_target_history_sink + len + loops + 0x517cc1b727220a95ull;
    for (unsigned int round = 0; round < loops; round++) {
        size_t offset = ((size_t)round * 19u) & (len - 1u);
        for (size_t i = 0; i < len; i++) {
            size_t pattern_index = (i + offset) & (len - 1u);
            unsigned int target_index = (unsigned int)(pattern[pattern_index] & 3u);
            indirect_target_fn fn = indirect_target_table[target_index];
            acc ^= fn(acc + ((uint64_t)round << 32) + (uint64_t)i);
            __asm__ __volatile__("" : "+r"(acc) :: "memory");
        }
    }
    indirect_target_history_sink = acc;
}

static int measure_indirect_target_history_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct indirect_target_sequence_spec *sequence,
    unsigned char *neutral,
    unsigned char *forward,
    unsigned char *reverse,
    unsigned char *shuffle,
    unsigned char *sentinel,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *pattern_digest_after_history,
    uint64_t *pattern_digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    if (pin_to_core(CATCAS_CORE_B) != 0) return -EINVAL;
    run_indirect_target_pattern(neutral, INDIRECT_TARGET_PATTERN_BYTES, INDIRECT_TARGET_RESTORE_LOOPS);
    if (sequence->history_loops > 0u) {
        unsigned char *history = indirect_target_pattern_by_kind(
            neutral, forward, reverse, shuffle, sentinel, sequence->history_pattern);
        run_indirect_target_pattern(history, INDIRECT_TARGET_PATTERN_BYTES, sequence->history_loops);
    }
    *pattern_digest_after_history =
        indirect_target_pattern_digest(neutral, forward, reverse, shuffle, sentinel);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    run_indirect_target_pattern(sentinel, INDIRECT_TARGET_PATTERN_BYTES, INDIRECT_TARGET_SENTINEL_LOOPS);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    run_indirect_target_pattern(neutral, INDIRECT_TARGET_PATTERN_BYTES, INDIRECT_TARGET_RESTORE_LOOPS);
    *pattern_digest_after_restore =
        indirect_target_pattern_digest(neutral, forward, reverse, shuffle, sentinel);
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_indirect_target_history_mode(const char *output_root) {
    unsigned char *neutral = NULL;
    unsigned char *forward = NULL;
    unsigned char *reverse = NULL;
    unsigned char *shuffle = NULL;
    unsigned char *sentinel = NULL;
    if (posix_memalign((void **)&neutral, CACHE_LINE_BYTES, INDIRECT_TARGET_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&forward, CACHE_LINE_BYTES, INDIRECT_TARGET_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&reverse, CACHE_LINE_BYTES, INDIRECT_TARGET_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&shuffle, CACHE_LINE_BYTES, INDIRECT_TARGET_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&sentinel, CACHE_LINE_BYTES, INDIRECT_TARGET_PATTERN_BYTES) != 0) {
        free(neutral);
        free(forward);
        free(reverse);
        free(shuffle);
        free(sentinel);
        fprintf(stderr, "indirect target pattern allocation failed\n");
        return 1;
    }
    fill_indirect_target_pattern(neutral, INDIRECT_TARGET_PATTERN_BYTES, INDIRECT_TARGET_PATTERN_NEUTRAL);
    fill_indirect_target_pattern(forward, INDIRECT_TARGET_PATTERN_BYTES, INDIRECT_TARGET_PATTERN_FORWARD);
    fill_indirect_target_pattern(reverse, INDIRECT_TARGET_PATTERN_BYTES, INDIRECT_TARGET_PATTERN_REVERSE);
    fill_indirect_target_pattern(shuffle, INDIRECT_TARGET_PATTERN_BYTES, INDIRECT_TARGET_PATTERN_SHUFFLE);
    fill_indirect_target_pattern(sentinel, INDIRECT_TARGET_PATTERN_BYTES, INDIRECT_TARGET_PATTERN_SENTINEL);
    uint64_t initial_digest =
        indirect_target_pattern_digest(neutral, forward, reverse, shuffle, sentinel);

    int branch_fds[MAX_GROUP_EVENTS];
    uint64_t branch_ids[MAX_GROUP_EVENTS];
    int branch_open_rc = open_group(branch_history_group, CATCAS_CORE_B, branch_fds, branch_ids);
    if (branch_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(branch_fds[i]);
    }

    const struct indirect_target_sequence_spec sequences[] = {
        {"neutral_indirect_target_history", INDIRECT_TARGET_PATTERN_NEUTRAL, INDIRECT_TARGET_HISTORY_LOOPS},
        {"forward_indirect_target_history", INDIRECT_TARGET_PATTERN_FORWARD, INDIRECT_TARGET_HISTORY_LOOPS},
        {"reverse_indirect_target_history", INDIRECT_TARGET_PATTERN_REVERSE, INDIRECT_TARGET_HISTORY_LOOPS},
        {"shuffle_indirect_target_history", INDIRECT_TARGET_PATTERN_SHUFFLE, INDIRECT_TARGET_HISTORY_LOOPS},
    };
    enum { INDIRECT_TARGET_WINDOW_COUNT = 4 };
    struct group_result results[INDIRECT_TARGET_WINDOW_COUNT];
    uint64_t durations[INDIRECT_TARGET_WINDOW_COUNT];
    uint64_t digest_after_history[INDIRECT_TARGET_WINDOW_COUNT];
    uint64_t digest_after_restore[INDIRECT_TARGET_WINDOW_COUNT];
    int window_rc[INDIRECT_TARGET_WINDOW_COUNT];
    for (int i = 0; i < INDIRECT_TARGET_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        if (branch_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_indirect_target_history_window(
                branch_history_group,
                &sequences[i],
                neutral,
                forward,
                reverse,
                shuffle,
                sentinel,
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool patterns_unchanged_after_history = true;
    bool patterns_unchanged_after_restore = true;
    for (int i = 0; i < INDIRECT_TARGET_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        patterns_unchanged_after_history =
            patterns_unchanged_after_history && digest_after_history[i] == initial_digest;
        patterns_unchanged_after_restore =
            patterns_unchanged_after_restore && digest_after_restore[i] == initial_digest;
    }

    uint64_t identity_misses = event_value_by_name(
        &results[0], branch_history_group, "retired_mispredicted_branch_instructions");
    uint64_t forward_misses = event_value_by_name(
        &results[1], branch_history_group, "retired_mispredicted_branch_instructions");
    uint64_t reverse_misses = event_value_by_name(
        &results[2], branch_history_group, "retired_mispredicted_branch_instructions");
    uint64_t shuffle_misses = event_value_by_name(
        &results[3], branch_history_group, "retired_mispredicted_branch_instructions");
    uint64_t miss_delta = abs_diff_u64(forward_misses, reverse_misses);
    uint64_t miss_control = abs_diff_u64(identity_misses, shuffle_misses);
    uint64_t miss_threshold = max_u64(32u, 3u * miss_control);
    bool miss_signal = miss_delta > miss_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(1000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;
    bool indirect_target_response = branch_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && patterns_unchanged_after_history &&
        patterns_unchanged_after_restore && miss_signal;

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_INDIRECT_TARGET_HISTORY_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", INDIRECT_TARGET_HISTORY_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", indirect_target_response ? "INDIRECT_TARGET_HISTORY_RESPONSE_FOUND" : "INDIRECT_TARGET_HISTORY_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Indirect-target history PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"target_history_core\": %d},\n", CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_code_and_data_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false},\n");
    fprintf(out, "  \"target_history_carrier\": {\"pattern_bytes\": %u, \"restore_loops\": %u, \"history_loops\": %u, \"sentinel_loops\": %u, \"target_count\": 4, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        INDIRECT_TARGET_PATTERN_BYTES,
        INDIRECT_TARGET_RESTORE_LOOPS,
        INDIRECT_TARGET_HISTORY_LOOPS,
        INDIRECT_TARGET_SENTINEL_LOOPS,
        initial_digest);
    fprintf(out, "  \"selected_group\": \"branch_history_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", branch_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, branch_history_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"balanced public indirect-target training pattern at one CAT_CAS-owned call site\", \"restore\": \"neutral target-history wash before every window\", \"sentinel\": \"fixed public indirect-target sequence measured at the same call site\", \"primary_acceptance\": \"forward/reverse retired_mispredicted_branch_instructions delta exceeds max(32, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < INDIRECT_TARGET_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"pattern_digest_after_history_hex\": \"0x%016" PRIx64 "\", \"pattern_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"patterns_unchanged_after_history\": %s, \"patterns_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, branch_history_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == INDIRECT_TARGET_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"retired_mispredicted_branch_instructions\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_misses, forward_misses, reverse_misses, shuffle_misses, miss_delta, miss_control, miss_threshold, json_bool(miss_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"patterns_unchanged_after_history\": %s, \"patterns_unchanged_after_restore\": %s, \"branch_miss_signal\": %s, \"duration_signal\": %s, \"indirect_target_history_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(patterns_unchanged_after_history),
        json_bool(patterns_unchanged_after_restore),
        json_bool(miss_signal),
        json_bool(duration_signal),
        json_bool(indirect_target_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"branch_history_group\"}\n",
        indirect_target_response ? "INDIRECT_TARGET_HISTORY_RESPONSE_FOUND" : "INDIRECT_TARGET_HISTORY_RESPONSE_NOT_ESTABLISHED",
        result_path);
    free(neutral);
    free(forward);
    free(reverse);
    free(shuffle);
    free(sentinel);
    return 0;
}

static volatile uint64_t translation_history_sink = 0;

static size_t translation_index(enum translation_pattern_kind kind, size_t i, size_t page_count) {
    size_t mask = page_count - 1u;
    switch (kind) {
        case TRANSLATION_PATTERN_NEUTRAL:
            return (i * 257u + (i >> 3)) & mask;
        case TRANSLATION_PATTERN_FORWARD:
            return i & mask;
        case TRANSLATION_PATTERN_REVERSE:
            return (page_count - 1u - (i & mask)) & mask;
        case TRANSLATION_PATTERN_SHUFFLE:
            return ((i * 1103515245u) + 12345u) & mask;
        case TRANSLATION_PATTERN_SENTINEL:
            return ((i * 521u) ^ (i >> 1) ^ 0x5a5au) & mask;
    }
    return i & mask;
}

__attribute__((noinline))
static void run_translation_pattern(
    volatile unsigned char *pages,
    size_t page_size,
    size_t page_count,
    enum translation_pattern_kind kind,
    unsigned int loops
) {
    uint64_t acc = translation_history_sink + page_count + loops;
    size_t span = page_count * 2u;
    for (unsigned int round = 0; round < loops; round++) {
        for (size_t i = 0; i < span; i++) {
            size_t index = translation_index(kind, i + ((size_t)round * 131u), page_count);
            acc += (uint64_t)pages[index * page_size];
            acc ^= ((uint64_t)index + 1u) * 0x94d049bb133111ebull;
            __asm__ __volatile__("" : "+r"(acc) :: "memory");
        }
    }
    translation_history_sink = acc;
}

static int measure_translation_history_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct translation_sequence_spec *sequence,
    unsigned char *pages,
    size_t page_size,
    size_t page_count,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_history,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    if (pin_to_core(CATCAS_CORE_B) != 0) return -EINVAL;
    run_translation_pattern(pages, page_size, page_count, TRANSLATION_PATTERN_NEUTRAL,
                            TRANSLATION_RESTORE_LOOPS);
    if (sequence->history_loops > 0u) {
        run_translation_pattern(pages, page_size, page_count, sequence->history_pattern,
                                sequence->history_loops);
    }
    *digest_after_history = fnv1a64(pages, page_size * page_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    run_translation_pattern(pages, page_size, page_count, TRANSLATION_PATTERN_SENTINEL,
                            TRANSLATION_SENTINEL_LOOPS);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    run_translation_pattern(pages, page_size, page_count, TRANSLATION_PATTERN_NEUTRAL,
                            TRANSLATION_RESTORE_LOOPS);
    *digest_after_restore = fnv1a64(pages, page_size * page_count);
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_translation_history_mode(const char *output_root) {
    long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        fprintf(stderr, "page size unavailable\n");
        return 1;
    }
    size_t page_size = (size_t)page_size_long;
    size_t byte_count = page_size * TRANSLATION_PAGE_COUNT;
    unsigned char *pages = NULL;
    if (posix_memalign((void **)&pages, page_size, byte_count) != 0) {
        fprintf(stderr, "translation buffer allocation failed\n");
        return 1;
    }
    for (size_t page = 0; page < TRANSLATION_PAGE_COUNT; page++) {
        pages[page * page_size] = (unsigned char)((page * 131u + 17u) & 0xffu);
    }
    uint64_t initial_digest = fnv1a64(pages, byte_count);

    int translation_fds[MAX_GROUP_EVENTS];
    uint64_t translation_ids[MAX_GROUP_EVENTS];
    int translation_open_rc = open_group(translation_history_group, CATCAS_CORE_B,
                                         translation_fds, translation_ids);
    if (translation_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(translation_fds[i]);
    }

    const struct translation_sequence_spec sequences[] = {
        {"neutral_restore_only", TRANSLATION_PATTERN_NEUTRAL, 0u},
        {"forward_translation_history", TRANSLATION_PATTERN_FORWARD, TRANSLATION_HISTORY_LOOPS},
        {"reverse_translation_history", TRANSLATION_PATTERN_REVERSE, TRANSLATION_HISTORY_LOOPS},
        {"shuffle_translation_history", TRANSLATION_PATTERN_SHUFFLE, TRANSLATION_HISTORY_LOOPS},
    };
    enum { TRANSLATION_WINDOW_COUNT = 4 };
    struct group_result results[TRANSLATION_WINDOW_COUNT];
    uint64_t durations[TRANSLATION_WINDOW_COUNT];
    uint64_t digest_after_history[TRANSLATION_WINDOW_COUNT];
    uint64_t digest_after_restore[TRANSLATION_WINDOW_COUNT];
    int window_rc[TRANSLATION_WINDOW_COUNT];
    for (int i = 0; i < TRANSLATION_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        if (translation_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_translation_history_window(
                translation_history_group,
                &sequences[i],
                pages,
                page_size,
                TRANSLATION_PAGE_COUNT,
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool bytes_unchanged_after_history = true;
    bool bytes_unchanged_after_restore = true;
    for (int i = 0; i < TRANSLATION_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        bytes_unchanged_after_history =
            bytes_unchanged_after_history && digest_after_history[i] == initial_digest;
        bytes_unchanged_after_restore =
            bytes_unchanged_after_restore && digest_after_restore[i] == initial_digest;
    }

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(10000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;

    uint64_t identity_misses = event_value_by_name(&results[0], translation_history_group, "cache_misses");
    uint64_t forward_misses = event_value_by_name(&results[1], translation_history_group, "cache_misses");
    uint64_t reverse_misses = event_value_by_name(&results[2], translation_history_group, "cache_misses");
    uint64_t shuffle_misses = event_value_by_name(&results[3], translation_history_group, "cache_misses");
    uint64_t miss_delta = abs_diff_u64(forward_misses, reverse_misses);
    uint64_t miss_control = abs_diff_u64(identity_misses, shuffle_misses);
    uint64_t miss_threshold = max_u64(32u, 3u * miss_control);
    bool miss_signal = miss_delta > miss_threshold;
    bool translation_history_response = translation_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && bytes_unchanged_after_history &&
        bytes_unchanged_after_restore && duration_signal;

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_TRANSLATION_HISTORY_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) {
        free(pages);
        return 1;
    }
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        free(pages);
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", TRANSLATION_HISTORY_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", translation_history_response ? "TRANSLATION_HISTORY_RESPONSE_FOUND" : "TRANSLATION_HISTORY_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Translation-footprint timing/PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"translation_core\": %d},\n", CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false, \"page_table_mutation\": false},\n");
    fprintf(out, "  \"translation_carrier\": {\"page_count\": %u, \"page_bytes\": %zu, \"byte_count\": %zu, \"restore_loops\": %u, \"history_loops\": %u, \"sentinel_loops\": %u, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        TRANSLATION_PAGE_COUNT, page_size, byte_count, TRANSLATION_RESTORE_LOOPS, TRANSLATION_HISTORY_LOOPS, TRANSLATION_SENTINEL_LOOPS, initial_digest);
    fprintf(out, "  \"selected_group\": \"translation_history_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", translation_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, translation_history_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"balanced public page-footprint training pattern\", \"restore\": \"neutral page-footprint wash before every measured window\", \"sentinel\": \"fixed public page-footprint sequence\", \"primary_acceptance\": \"forward/reverse sentinel duration delta exceeds max(10000 ns, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < TRANSLATION_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"digest_after_history_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, translation_history_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == TRANSLATION_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "    \"cache_misses\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_misses, forward_misses, reverse_misses, shuffle_misses, miss_delta, miss_control, miss_threshold, json_bool(miss_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"duration_signal\": %s, \"cache_miss_signal\": %s, \"translation_history_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(bytes_unchanged_after_history),
        json_bool(bytes_unchanged_after_restore),
        json_bool(duration_signal),
        json_bool(miss_signal),
        json_bool(translation_history_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"translation_history_group\"}\n",
        translation_history_response ? "TRANSLATION_HISTORY_RESPONSE_FOUND" : "TRANSLATION_HISTORY_RESPONSE_NOT_ESTABLISHED",
        result_path);
    free(pages);
    return 0;
}

static volatile uint64_t store_load_alias_sink = 0;

static void fill_store_load_alias_pattern(
    unsigned char *pattern,
    size_t len,
    enum store_load_alias_pattern_kind kind
) {
    static const unsigned char neutral_seq[8] = {0u, 1u, 2u, 3u, 3u, 2u, 1u, 0u};
    static const unsigned char forward_seq[8] = {0u, 1u, 2u, 3u, 0u, 1u, 2u, 3u};
    static const unsigned char reverse_seq[8] = {3u, 2u, 1u, 0u, 3u, 2u, 1u, 0u};
    static const unsigned char shuffle_seq[8] = {0u, 2u, 3u, 1u, 1u, 3u, 2u, 0u};
    static const unsigned char sentinel_seq[8] = {1u, 3u, 0u, 2u, 2u, 0u, 3u, 1u};
    const unsigned char *seq = neutral_seq;
    switch (kind) {
        case STORE_LOAD_ALIAS_PATTERN_NEUTRAL:
            seq = neutral_seq;
            break;
        case STORE_LOAD_ALIAS_PATTERN_FORWARD:
            seq = forward_seq;
            break;
        case STORE_LOAD_ALIAS_PATTERN_REVERSE:
            seq = reverse_seq;
            break;
        case STORE_LOAD_ALIAS_PATTERN_SHUFFLE:
            seq = shuffle_seq;
            break;
        case STORE_LOAD_ALIAS_PATTERN_SENTINEL:
            seq = sentinel_seq;
            break;
    }
    for (size_t i = 0; i < len; i++) {
        pattern[i] = seq[(i + (i >> 6)) & 7u];
    }
}

static unsigned char *store_load_alias_pattern_by_kind(
    unsigned char *neutral,
    unsigned char *forward,
    unsigned char *reverse,
    unsigned char *shuffle,
    unsigned char *sentinel,
    enum store_load_alias_pattern_kind kind
) {
    switch (kind) {
        case STORE_LOAD_ALIAS_PATTERN_NEUTRAL:
            return neutral;
        case STORE_LOAD_ALIAS_PATTERN_FORWARD:
            return forward;
        case STORE_LOAD_ALIAS_PATTERN_REVERSE:
            return reverse;
        case STORE_LOAD_ALIAS_PATTERN_SHUFFLE:
            return shuffle;
        case STORE_LOAD_ALIAS_PATTERN_SENTINEL:
            return sentinel;
    }
    return neutral;
}

static uint64_t store_load_alias_pattern_digest(
    const unsigned char *neutral,
    const unsigned char *forward,
    const unsigned char *reverse,
    const unsigned char *shuffle,
    const unsigned char *sentinel,
    const unsigned char *bytes,
    size_t byte_count
) {
    uint64_t digest = fnv1a64(neutral, STORE_LOAD_ALIAS_PATTERN_BYTES);
    digest ^= fnv1a64(forward, STORE_LOAD_ALIAS_PATTERN_BYTES) + 0x9e3779b97f4a7c15ull;
    digest ^= fnv1a64(reverse, STORE_LOAD_ALIAS_PATTERN_BYTES) + 0xbf58476d1ce4e5b9ull;
    digest ^= fnv1a64(shuffle, STORE_LOAD_ALIAS_PATTERN_BYTES) + 0x94d049bb133111ebull;
    digest ^= fnv1a64(sentinel, STORE_LOAD_ALIAS_PATTERN_BYTES) + 0x2545f4914f6cdd1dull;
    digest ^= fnv1a64(bytes, byte_count);
    return digest;
}

static void init_store_load_alias_bytes(unsigned char *bytes, size_t byte_count) {
    for (size_t i = 0; i < byte_count; i++) {
        bytes[i] = (unsigned char)((i * 131u + (i >> 3) + 17u) & 0xffu);
    }
}

__attribute__((noinline))
static void run_store_load_alias_pattern(
    unsigned char *bytes,
    const volatile unsigned char *pattern,
    size_t len,
    unsigned int loops
) {
    uint64_t acc = store_load_alias_sink + len + loops + 0xd6e8feb86659fd93ull;
    for (unsigned int round = 0; round < loops; round++) {
        size_t offset = ((size_t)round * 23u) & (len - 1u);
        for (size_t i = 0; i < len; i++) {
            size_t pattern_index = (i + offset) & (len - 1u);
            size_t pair = (size_t)(pattern[pattern_index] & 3u);
            size_t line_offset = ((i + ((size_t)round * 5u)) & 7u) * CACHE_LINE_BYTES;
            size_t store_page = pair * 2u;
            size_t load_page = store_page + 1u;
            volatile uint64_t *store_ptr = (volatile uint64_t *)(void *)
                (bytes + (store_page * STORE_LOAD_ALIAS_PAGE_BYTES) + line_offset);
            volatile uint64_t *load_ptr = (volatile uint64_t *)(void *)
                (bytes + (load_page * STORE_LOAD_ALIAS_PAGE_BYTES) + line_offset);
            uint64_t same_value = *store_ptr;
            *store_ptr = same_value;
            acc ^= *load_ptr + same_value + (uint64_t)pattern_index;
            __asm__ __volatile__("" : "+r"(acc) :: "memory");
        }
    }
    store_load_alias_sink = acc;
}

static int measure_store_load_alias_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct store_load_alias_sequence_spec *sequence,
    unsigned char *bytes,
    unsigned char *neutral,
    unsigned char *forward,
    unsigned char *reverse,
    unsigned char *shuffle,
    unsigned char *sentinel,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_history,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    if (pin_to_core(CATCAS_CORE_B) != 0) return -EINVAL;
    size_t byte_count = STORE_LOAD_ALIAS_PAGE_BYTES * STORE_LOAD_ALIAS_PAGE_COUNT;
    run_store_load_alias_pattern(bytes, neutral, STORE_LOAD_ALIAS_PATTERN_BYTES,
                                 STORE_LOAD_ALIAS_RESTORE_LOOPS);
    if (sequence->history_loops > 0u) {
        unsigned char *history = store_load_alias_pattern_by_kind(
            neutral, forward, reverse, shuffle, sentinel, sequence->history_pattern);
        run_store_load_alias_pattern(bytes, history, STORE_LOAD_ALIAS_PATTERN_BYTES,
                                     sequence->history_loops);
    }
    *digest_after_history = store_load_alias_pattern_digest(
        neutral, forward, reverse, shuffle, sentinel, bytes, byte_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    run_store_load_alias_pattern(bytes, sentinel, STORE_LOAD_ALIAS_PATTERN_BYTES,
                                 STORE_LOAD_ALIAS_SENTINEL_LOOPS);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    run_store_load_alias_pattern(bytes, neutral, STORE_LOAD_ALIAS_PATTERN_BYTES,
                                 STORE_LOAD_ALIAS_RESTORE_LOOPS);
    *digest_after_restore = store_load_alias_pattern_digest(
        neutral, forward, reverse, shuffle, sentinel, bytes, byte_count);
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_store_load_alias_history_mode(const char *output_root) {
    unsigned char *bytes = NULL;
    unsigned char *neutral = NULL;
    unsigned char *forward = NULL;
    unsigned char *reverse = NULL;
    unsigned char *shuffle = NULL;
    unsigned char *sentinel = NULL;
    size_t byte_count = STORE_LOAD_ALIAS_PAGE_BYTES * STORE_LOAD_ALIAS_PAGE_COUNT;
    if (posix_memalign((void **)&bytes, STORE_LOAD_ALIAS_PAGE_BYTES, byte_count) != 0 ||
        posix_memalign((void **)&neutral, CACHE_LINE_BYTES, STORE_LOAD_ALIAS_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&forward, CACHE_LINE_BYTES, STORE_LOAD_ALIAS_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&reverse, CACHE_LINE_BYTES, STORE_LOAD_ALIAS_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&shuffle, CACHE_LINE_BYTES, STORE_LOAD_ALIAS_PATTERN_BYTES) != 0 ||
        posix_memalign((void **)&sentinel, CACHE_LINE_BYTES, STORE_LOAD_ALIAS_PATTERN_BYTES) != 0) {
        free(bytes);
        free(neutral);
        free(forward);
        free(reverse);
        free(shuffle);
        free(sentinel);
        fprintf(stderr, "store/load alias allocation failed\n");
        return 1;
    }
    init_store_load_alias_bytes(bytes, byte_count);
    fill_store_load_alias_pattern(neutral, STORE_LOAD_ALIAS_PATTERN_BYTES,
                                  STORE_LOAD_ALIAS_PATTERN_NEUTRAL);
    fill_store_load_alias_pattern(forward, STORE_LOAD_ALIAS_PATTERN_BYTES,
                                  STORE_LOAD_ALIAS_PATTERN_FORWARD);
    fill_store_load_alias_pattern(reverse, STORE_LOAD_ALIAS_PATTERN_BYTES,
                                  STORE_LOAD_ALIAS_PATTERN_REVERSE);
    fill_store_load_alias_pattern(shuffle, STORE_LOAD_ALIAS_PATTERN_BYTES,
                                  STORE_LOAD_ALIAS_PATTERN_SHUFFLE);
    fill_store_load_alias_pattern(sentinel, STORE_LOAD_ALIAS_PATTERN_BYTES,
                                  STORE_LOAD_ALIAS_PATTERN_SENTINEL);
    uint64_t initial_digest = store_load_alias_pattern_digest(
        neutral, forward, reverse, shuffle, sentinel, bytes, byte_count);

    int alias_fds[MAX_GROUP_EVENTS];
    uint64_t alias_ids[MAX_GROUP_EVENTS];
    int alias_open_rc = open_group(translation_history_group, CATCAS_CORE_B,
                                   alias_fds, alias_ids);
    if (alias_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(alias_fds[i]);
    }

    const struct store_load_alias_sequence_spec sequences[] = {
        {"neutral_store_load_alias_history", STORE_LOAD_ALIAS_PATTERN_NEUTRAL,
         STORE_LOAD_ALIAS_HISTORY_LOOPS},
        {"forward_store_load_alias_history", STORE_LOAD_ALIAS_PATTERN_FORWARD,
         STORE_LOAD_ALIAS_HISTORY_LOOPS},
        {"reverse_store_load_alias_history", STORE_LOAD_ALIAS_PATTERN_REVERSE,
         STORE_LOAD_ALIAS_HISTORY_LOOPS},
        {"shuffle_store_load_alias_history", STORE_LOAD_ALIAS_PATTERN_SHUFFLE,
         STORE_LOAD_ALIAS_HISTORY_LOOPS},
    };
    enum { STORE_LOAD_ALIAS_WINDOW_COUNT = 4 };
    struct group_result results[STORE_LOAD_ALIAS_WINDOW_COUNT];
    uint64_t durations[STORE_LOAD_ALIAS_WINDOW_COUNT];
    uint64_t digest_after_history[STORE_LOAD_ALIAS_WINDOW_COUNT];
    uint64_t digest_after_restore[STORE_LOAD_ALIAS_WINDOW_COUNT];
    int window_rc[STORE_LOAD_ALIAS_WINDOW_COUNT];
    for (int i = 0; i < STORE_LOAD_ALIAS_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        if (alias_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_store_load_alias_window(
                translation_history_group,
                &sequences[i],
                bytes,
                neutral,
                forward,
                reverse,
                shuffle,
                sentinel,
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool bytes_unchanged_after_history = true;
    bool bytes_unchanged_after_restore = true;
    for (int i = 0; i < STORE_LOAD_ALIAS_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        bytes_unchanged_after_history =
            bytes_unchanged_after_history && digest_after_history[i] == initial_digest;
        bytes_unchanged_after_restore =
            bytes_unchanged_after_restore && digest_after_restore[i] == initial_digest;
    }

    uint64_t identity_misses = event_value_by_name(&results[0], translation_history_group, "cache_misses");
    uint64_t forward_misses = event_value_by_name(&results[1], translation_history_group, "cache_misses");
    uint64_t reverse_misses = event_value_by_name(&results[2], translation_history_group, "cache_misses");
    uint64_t shuffle_misses = event_value_by_name(&results[3], translation_history_group, "cache_misses");
    uint64_t miss_delta = abs_diff_u64(forward_misses, reverse_misses);
    uint64_t miss_control = abs_diff_u64(identity_misses, shuffle_misses);
    uint64_t miss_threshold = max_u64(32u, 3u * miss_control);
    bool miss_signal = miss_delta > miss_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(1000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;
    bool store_load_alias_response = alias_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && bytes_unchanged_after_history &&
        bytes_unchanged_after_restore && duration_signal;

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path),
                     "%s/F10_STORE_LOAD_ALIAS_HISTORY_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", STORE_LOAD_ALIAS_HISTORY_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", store_load_alias_response ? "STORE_LOAD_ALIAS_HISTORY_RESPONSE_FOUND" : "STORE_LOAD_ALIAS_HISTORY_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Store/load alias-history discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"alias_history_core\": %d},\n", CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false},\n");
    fprintf(out, "  \"alias_history_carrier\": {\"page_bytes\": %u, \"page_count\": %u, \"pattern_bytes\": %u, \"restore_loops\": %u, \"history_loops\": %u, \"sentinel_loops\": %u, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        STORE_LOAD_ALIAS_PAGE_BYTES,
        STORE_LOAD_ALIAS_PAGE_COUNT,
        STORE_LOAD_ALIAS_PATTERN_BYTES,
        STORE_LOAD_ALIAS_RESTORE_LOOPS,
        STORE_LOAD_ALIAS_HISTORY_LOOPS,
        STORE_LOAD_ALIAS_SENTINEL_LOOPS,
        initial_digest);
    fprintf(out, "  \"selected_group\": \"translation_history_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", alias_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, translation_history_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"balanced same-page-offset store/load pairs over CAT_CAS-owned pages\", \"restore\": \"neutral store/load alias wash before every window\", \"sentinel\": \"fixed same-page-offset store/load sequence\", \"primary_acceptance\": \"forward/reverse sentinel duration delta exceeds max(1000 ns, 3 * neutral/shuffle control spread)\", \"secondary_observable\": \"cache_misses\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < STORE_LOAD_ALIAS_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"digest_after_history_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, translation_history_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == STORE_LOAD_ALIAS_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"cache_misses\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_misses, forward_misses, reverse_misses, shuffle_misses, miss_delta, miss_control, miss_threshold, json_bool(miss_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"cache_miss_signal\": %s, \"duration_signal\": %s, \"store_load_alias_history_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(bytes_unchanged_after_history),
        json_bool(bytes_unchanged_after_restore),
        json_bool(miss_signal),
        json_bool(duration_signal),
        json_bool(store_load_alias_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"translation_history_group\"}\n",
        store_load_alias_response ? "STORE_LOAD_ALIAS_HISTORY_RESPONSE_FOUND" : "STORE_LOAD_ALIAS_HISTORY_RESPONSE_NOT_ESTABLISHED",
        result_path);
    free(bytes);
    free(neutral);
    free(forward);
    free(reverse);
    free(shuffle);
    free(sentinel);
    return 0;
}

static volatile uint64_t prefetch_stream_sink = 0;

static size_t prefetch_stream_sentinel_start(void) {
    return PREFETCH_STREAM_LINES / 2u;
}

static size_t prefetch_stream_index(enum prefetch_stream_kind kind, size_t i) {
    size_t sentinel_start = prefetch_stream_sentinel_start();
    size_t history_span = PREFETCH_HISTORY_LINES;
    size_t mask = PREFETCH_STREAM_LINES - 1u;
    switch (kind) {
        case PREFETCH_STREAM_NEUTRAL:
            return ((i * 2053u) + 97u) & mask;
        case PREFETCH_STREAM_FORWARD:
            return (sentinel_start - history_span + (i % history_span)) & mask;
        case PREFETCH_STREAM_REVERSE:
            return (sentinel_start + history_span - 1u - (i % history_span)) & mask;
        case PREFETCH_STREAM_SHUFFLE:
            return (sentinel_start - history_span + (((i * 1103515245u) + 12345u) % (history_span * 2u))) & mask;
        case PREFETCH_STREAM_SENTINEL:
            return (sentinel_start + (i % PREFETCH_SENTINEL_LINES)) & mask;
    }
    return i & mask;
}

__attribute__((noinline))
static void run_prefetch_stream_pattern(
    volatile unsigned char *bytes,
    enum prefetch_stream_kind kind,
    unsigned int loops
) {
    uint64_t acc = prefetch_stream_sink + loops + (uint64_t)kind;
    size_t iterations = kind == PREFETCH_STREAM_SENTINEL ? PREFETCH_SENTINEL_LINES : PREFETCH_HISTORY_LINES;
    for (unsigned int round = 0; round < loops; round++) {
        for (size_t i = 0; i < iterations; i++) {
            size_t index = prefetch_stream_index(kind, i + ((size_t)round * 17u));
            acc += (uint64_t)bytes[index * CACHE_LINE_BYTES];
            acc ^= ((uint64_t)index + 1u) * 0xd6e8feb86659fd93ull;
            __asm__ __volatile__("" : "+r"(acc) :: "memory");
        }
    }
    prefetch_stream_sink = acc;
}

static void flush_prefetch_sentinel_region(unsigned char *bytes) {
    size_t sentinel_start = prefetch_stream_sentinel_start();
    for (size_t i = 0; i < PREFETCH_SENTINEL_LINES; i++) {
        unsigned char *p = bytes + ((sentinel_start + i) * CACHE_LINE_BYTES);
        __asm__ __volatile__("clflush (%0)" : : "r"(p) : "memory");
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

static int measure_prefetch_stream_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct prefetch_stream_sequence_spec *sequence,
    unsigned char *bytes,
    size_t byte_count,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_history,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    if (pin_to_core(CATCAS_CORE_B) != 0) return -EINVAL;
    run_prefetch_stream_pattern(bytes, PREFETCH_STREAM_NEUTRAL, PREFETCH_RESTORE_LOOPS);
    if (sequence->history_loops > 0u) {
        run_prefetch_stream_pattern(bytes, sequence->history_pattern, sequence->history_loops);
    }
    *digest_after_history = fnv1a64(bytes, byte_count);
    flush_prefetch_sentinel_region(bytes);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    run_prefetch_stream_pattern(bytes, PREFETCH_STREAM_SENTINEL, PREFETCH_SENTINEL_LOOPS);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    run_prefetch_stream_pattern(bytes, PREFETCH_STREAM_NEUTRAL, PREFETCH_RESTORE_LOOPS);
    *digest_after_restore = fnv1a64(bytes, byte_count);
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_prefetch_stream_mode(const char *output_root) {
    size_t byte_count = (size_t)PREFETCH_STREAM_LINES * CACHE_LINE_BYTES;
    unsigned char *bytes = NULL;
    if (posix_memalign((void **)&bytes, CACHE_LINE_BYTES, byte_count) != 0) {
        fprintf(stderr, "prefetch stream buffer allocation failed\n");
        return 1;
    }
    for (size_t line = 0; line < PREFETCH_STREAM_LINES; line++) {
        bytes[line * CACHE_LINE_BYTES] = (unsigned char)((line * 73u + 19u) & 0xffu);
    }
    uint64_t initial_digest = fnv1a64(bytes, byte_count);

    int stream_fds[MAX_GROUP_EVENTS];
    uint64_t stream_ids[MAX_GROUP_EVENTS];
    int stream_open_rc = open_group(prefetch_stream_group, CATCAS_CORE_B, stream_fds, stream_ids);
    if (stream_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(stream_fds[i]);
    }

    const struct prefetch_stream_sequence_spec sequences[] = {
        {"neutral_restore_only", PREFETCH_STREAM_NEUTRAL, 0u},
        {"forward_stream_history", PREFETCH_STREAM_FORWARD, PREFETCH_HISTORY_LOOPS},
        {"reverse_stream_history", PREFETCH_STREAM_REVERSE, PREFETCH_HISTORY_LOOPS},
        {"shuffle_stream_history", PREFETCH_STREAM_SHUFFLE, PREFETCH_HISTORY_LOOPS},
    };
    enum { PREFETCH_WINDOW_COUNT = 4 };
    struct group_result results[PREFETCH_WINDOW_COUNT];
    uint64_t durations[PREFETCH_WINDOW_COUNT];
    uint64_t digest_after_history[PREFETCH_WINDOW_COUNT];
    uint64_t digest_after_restore[PREFETCH_WINDOW_COUNT];
    int window_rc[PREFETCH_WINDOW_COUNT];
    for (int i = 0; i < PREFETCH_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        if (stream_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_prefetch_stream_window(
                prefetch_stream_group,
                &sequences[i],
                bytes,
                byte_count,
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool bytes_unchanged_after_history = true;
    bool bytes_unchanged_after_restore = true;
    for (int i = 0; i < PREFETCH_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        bytes_unchanged_after_history =
            bytes_unchanged_after_history && digest_after_history[i] == initial_digest;
        bytes_unchanged_after_restore =
            bytes_unchanged_after_restore && digest_after_restore[i] == initial_digest;
    }

    uint64_t identity_misses = event_value_by_name(&results[0], prefetch_stream_group, "cache_misses");
    uint64_t forward_misses = event_value_by_name(&results[1], prefetch_stream_group, "cache_misses");
    uint64_t reverse_misses = event_value_by_name(&results[2], prefetch_stream_group, "cache_misses");
    uint64_t shuffle_misses = event_value_by_name(&results[3], prefetch_stream_group, "cache_misses");
    uint64_t miss_delta = abs_diff_u64(forward_misses, reverse_misses);
    uint64_t miss_control = abs_diff_u64(identity_misses, shuffle_misses);
    uint64_t miss_threshold = max_u64(32u, 3u * miss_control);
    bool miss_signal = miss_delta > miss_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(10000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;
    bool prefetch_stream_response = stream_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && bytes_unchanged_after_history &&
        bytes_unchanged_after_restore && miss_signal;

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_PREFETCH_STREAM_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) {
        free(bytes);
        return 1;
    }
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        free(bytes);
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", PREFETCH_STREAM_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", prefetch_stream_response ? "PREFETCH_STREAM_RESPONSE_FOUND" : "PREFETCH_STREAM_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Prefetch-stream timing/PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"stream_core\": %d},\n", CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false},\n");
    fprintf(out, "  \"stream_carrier\": {\"line_count\": %u, \"line_bytes\": %d, \"byte_count\": %zu, \"history_lines\": %u, \"sentinel_lines\": %u, \"restore_loops\": %u, \"history_loops\": %u, \"sentinel_loops\": %u, \"sentinel_flushed_before_measurement\": true, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        PREFETCH_STREAM_LINES, CACHE_LINE_BYTES, byte_count, PREFETCH_HISTORY_LINES, PREFETCH_SENTINEL_LINES,
        PREFETCH_RESTORE_LOOPS, PREFETCH_HISTORY_LOOPS, PREFETCH_SENTINEL_LOOPS, initial_digest);
    fprintf(out, "  \"selected_group\": \"prefetch_stream_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", stream_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, prefetch_stream_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"forward or reverse public line-read stream ending adjacent to the sentinel region\", \"restore\": \"neutral stream wash before every measured window\", \"sentinel\": \"fixed forward public line-read stream over flushed sentinel lines\", \"primary_acceptance\": \"forward/reverse cache_misses delta exceeds max(32, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < PREFETCH_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"digest_after_history_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, prefetch_stream_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == PREFETCH_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"cache_misses\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_misses, forward_misses, reverse_misses, shuffle_misses, miss_delta, miss_control, miss_threshold, json_bool(miss_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"cache_miss_signal\": %s, \"duration_signal\": %s, \"prefetch_stream_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(bytes_unchanged_after_history),
        json_bool(bytes_unchanged_after_restore),
        json_bool(miss_signal),
        json_bool(duration_signal),
        json_bool(prefetch_stream_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"prefetch_stream_group\"}\n",
        prefetch_stream_response ? "PREFETCH_STREAM_RESPONSE_FOUND" : "PREFETCH_STREAM_RESPONSE_NOT_ESTABLISHED",
        result_path);
    free(bytes);
    return 0;
}

static volatile uint64_t code_footprint_sink = 0;

__attribute__((noinline, aligned(64)))
static void code_footprint_block0(uint64_t *acc) {
    uint64_t x = *acc + 0x9e3779b97f4a7c15ull;
    for (unsigned int i = 0; i < 8u; i++) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        __asm__ __volatile__(".rept 64\nnop\n.endr" : "+r"(x) :: "memory");
    }
    *acc = x;
}

__attribute__((noinline, aligned(64)))
static void code_footprint_block1(uint64_t *acc) {
    uint64_t x = *acc ^ 0xc2b2ae3d27d4eb4full;
    for (unsigned int i = 0; i < 8u; i++) {
        x = (x << 17) | (x >> 47);
        x += 0x165667b19e3779f9ull;
        __asm__ __volatile__(".rept 64\nnop\n.endr" : "+r"(x) :: "memory");
    }
    *acc = x;
}

__attribute__((noinline, aligned(64)))
static void code_footprint_block2(uint64_t *acc) {
    uint64_t x = *acc + 0xd6e8feb86659fd93ull;
    for (unsigned int i = 0; i < 8u; i++) {
        x ^= x << 29;
        x += (x >> 11) ^ 0x94d049bb133111ebull;
        __asm__ __volatile__(".rept 64\nnop\n.endr" : "+r"(x) :: "memory");
    }
    *acc = x;
}

__attribute__((noinline, aligned(64)))
static void code_footprint_block3(uint64_t *acc) {
    uint64_t x = *acc ^ 0x2545f4914f6cdd1dull;
    for (unsigned int i = 0; i < 8u; i++) {
        x *= 0x369dea0f31a53f85ull;
        x ^= x >> 27;
        __asm__ __volatile__(".rept 64\nnop\n.endr" : "+r"(x) :: "memory");
    }
    *acc = x;
}

__attribute__((noinline))
static void run_code_footprint_block(unsigned int block, uint64_t *acc) {
    switch (block & 3u) {
        case 0u:
            code_footprint_block0(acc);
            break;
        case 1u:
            code_footprint_block1(acc);
            break;
        case 2u:
            code_footprint_block2(acc);
            break;
        default:
            code_footprint_block3(acc);
            break;
    }
}

static unsigned int code_footprint_block_index(enum code_footprint_kind kind, unsigned int i) {
    static const unsigned int neutral_seq[8] = {0u, 2u, 1u, 3u, 3u, 1u, 2u, 0u};
    static const unsigned int forward_seq[8] = {0u, 1u, 2u, 3u, 0u, 1u, 2u, 3u};
    static const unsigned int reverse_seq[8] = {3u, 2u, 1u, 0u, 3u, 2u, 1u, 0u};
    static const unsigned int shuffle_seq[8] = {1u, 3u, 0u, 2u, 2u, 0u, 3u, 1u};
    static const unsigned int sentinel_seq[8] = {0u, 3u, 1u, 2u, 1u, 0u, 2u, 3u};
    const unsigned int *seq = neutral_seq;
    switch (kind) {
        case CODE_FOOTPRINT_NEUTRAL:
            seq = neutral_seq;
            break;
        case CODE_FOOTPRINT_FORWARD:
            seq = forward_seq;
            break;
        case CODE_FOOTPRINT_REVERSE:
            seq = reverse_seq;
            break;
        case CODE_FOOTPRINT_SHUFFLE:
            seq = shuffle_seq;
            break;
        case CODE_FOOTPRINT_SENTINEL:
            seq = sentinel_seq;
            break;
    }
    return seq[i & 7u];
}

__attribute__((noinline))
static void run_code_footprint_pattern(enum code_footprint_kind kind, unsigned int loops) {
    uint64_t acc = code_footprint_sink ^ ((uint64_t)kind << 32) ^ loops;
    for (unsigned int loop = 0; loop < loops; loop++) {
        for (unsigned int i = 0; i < 32u; i++) {
            unsigned int mixed = i + (loop * 3u);
            run_code_footprint_block(code_footprint_block_index(kind, mixed), &acc);
            __asm__ __volatile__("" : "+r"(acc) :: "memory");
        }
    }
    code_footprint_sink = acc;
}

static uint64_t code_footprint_static_digest(void) {
    static const unsigned char patterns[] = {
        0u, 2u, 1u, 3u, 3u, 1u, 2u, 0u,
        0u, 1u, 2u, 3u, 0u, 1u, 2u, 3u,
        3u, 2u, 1u, 0u, 3u, 2u, 1u, 0u,
        1u, 3u, 0u, 2u, 2u, 0u, 3u, 1u,
        0u, 3u, 1u, 2u, 1u, 0u, 2u, 3u,
        'C', 'A', 'T', '_', 'C', 'A', 'S', '_',
        'C', 'O', 'D', 'E', '_', 'F', 'O', 'O',
        'T', 'P', 'R', 'I', 'N', 'T', '_', 'V', '1'
    };
    return fnv1a64(patterns, sizeof(patterns));
}

static int measure_code_footprint_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct code_footprint_sequence_spec *sequence,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_history,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    if (pin_to_core(CATCAS_CORE_B) != 0) return -EINVAL;
    run_code_footprint_pattern(CODE_FOOTPRINT_NEUTRAL, CODE_FOOTPRINT_RESTORE_LOOPS);
    if (sequence->history_loops > 0u) {
        run_code_footprint_pattern(sequence->history_pattern, sequence->history_loops);
    }
    *digest_after_history = code_footprint_static_digest();
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    run_code_footprint_pattern(CODE_FOOTPRINT_SENTINEL, CODE_FOOTPRINT_SENTINEL_LOOPS);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    run_code_footprint_pattern(CODE_FOOTPRINT_NEUTRAL, CODE_FOOTPRINT_RESTORE_LOOPS);
    *digest_after_restore = code_footprint_static_digest();
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_code_footprint_history_mode(const char *output_root) {
    uint64_t initial_digest = code_footprint_static_digest();
    int footprint_fds[MAX_GROUP_EVENTS];
    uint64_t footprint_ids[MAX_GROUP_EVENTS];
    int footprint_open_rc = open_group(code_footprint_group, CATCAS_CORE_B,
                                       footprint_fds, footprint_ids);
    if (footprint_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(footprint_fds[i]);
    }

    const struct code_footprint_sequence_spec sequences[] = {
        {"neutral_restore_only", CODE_FOOTPRINT_NEUTRAL, 0u},
        {"forward_code_history", CODE_FOOTPRINT_FORWARD, CODE_FOOTPRINT_HISTORY_LOOPS},
        {"reverse_code_history", CODE_FOOTPRINT_REVERSE, CODE_FOOTPRINT_HISTORY_LOOPS},
        {"shuffle_code_history", CODE_FOOTPRINT_SHUFFLE, CODE_FOOTPRINT_HISTORY_LOOPS},
    };
    enum { CODE_FOOTPRINT_WINDOW_COUNT = 4 };
    struct group_result results[CODE_FOOTPRINT_WINDOW_COUNT];
    uint64_t durations[CODE_FOOTPRINT_WINDOW_COUNT];
    uint64_t digest_after_history[CODE_FOOTPRINT_WINDOW_COUNT];
    uint64_t digest_after_restore[CODE_FOOTPRINT_WINDOW_COUNT];
    int window_rc[CODE_FOOTPRINT_WINDOW_COUNT];
    for (int i = 0; i < CODE_FOOTPRINT_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        if (footprint_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_code_footprint_window(
                code_footprint_group,
                &sequences[i],
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool patterns_unchanged_after_history = true;
    bool patterns_unchanged_after_restore = true;
    for (int i = 0; i < CODE_FOOTPRINT_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        patterns_unchanged_after_history =
            patterns_unchanged_after_history && digest_after_history[i] == initial_digest;
        patterns_unchanged_after_restore =
            patterns_unchanged_after_restore && digest_after_restore[i] == initial_digest;
    }

    uint64_t identity_misses = event_value_by_name(&results[0], code_footprint_group, "cache_misses");
    uint64_t forward_misses = event_value_by_name(&results[1], code_footprint_group, "cache_misses");
    uint64_t reverse_misses = event_value_by_name(&results[2], code_footprint_group, "cache_misses");
    uint64_t shuffle_misses = event_value_by_name(&results[3], code_footprint_group, "cache_misses");
    uint64_t miss_delta = abs_diff_u64(forward_misses, reverse_misses);
    uint64_t miss_control = abs_diff_u64(identity_misses, shuffle_misses);
    uint64_t miss_threshold = max_u64(32u, 3u * miss_control);
    bool miss_signal = miss_delta > miss_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(10000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;

    bool code_footprint_history_response = footprint_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && patterns_unchanged_after_history &&
        patterns_unchanged_after_restore && (miss_signal || duration_signal);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_CODE_FOOTPRINT_HISTORY_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", CODE_FOOTPRINT_HISTORY_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", code_footprint_history_response ? "CODE_FOOTPRINT_HISTORY_RESPONSE_FOUND" : "CODE_FOOTPRINT_HISTORY_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Compiled code-footprint timing/PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, target coupling, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"code_core\": %d},\n", CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_code_only\": true, \"generated_executable_memory\": false, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false},\n");
    fprintf(out, "  \"code_footprint_carrier\": {\"compiled_blocks\": 4, \"restore_loops\": %u, \"history_loops\": %u, \"sentinel_loops\": %u, \"sentinel_calls_per_loop\": 32, \"digest_kind\": \"fnv1a64_static_pattern\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        CODE_FOOTPRINT_RESTORE_LOOPS, CODE_FOOTPRINT_HISTORY_LOOPS, CODE_FOOTPRINT_SENTINEL_LOOPS, initial_digest);
    fprintf(out, "  \"selected_group\": \"code_footprint_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", footprint_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, code_footprint_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"balanced compiled CAT_CAS-owned code-block call order over fixed noinline blocks\", \"restore\": \"neutral compiled-code footprint wash before and after every measured window\", \"sentinel\": \"fixed compiled-code block-call sequence on the same core\", \"primary_acceptance\": \"forward/reverse sentinel cache_misses delta exceeds max(32, 3 * neutral/shuffle control spread) or duration delta exceeds max(10000 ns, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < CODE_FOOTPRINT_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"digest_after_history_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"patterns_unchanged_after_history\": %s, \"patterns_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, code_footprint_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == CODE_FOOTPRINT_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"cache_misses\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_misses, forward_misses, reverse_misses, shuffle_misses, miss_delta, miss_control, miss_threshold, json_bool(miss_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"patterns_unchanged_after_history\": %s, \"patterns_unchanged_after_restore\": %s, \"cache_miss_signal\": %s, \"duration_signal\": %s, \"code_footprint_history_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(patterns_unchanged_after_history),
        json_bool(patterns_unchanged_after_restore),
        json_bool(miss_signal),
        json_bool(duration_signal),
        json_bool(code_footprint_history_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"code_footprint_group\"}\n",
        code_footprint_history_response ? "CODE_FOOTPRINT_HISTORY_RESPONSE_FOUND" : "CODE_FOOTPRINT_HISTORY_RESPONSE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

static volatile uint64_t return_stack_sink = 0;

__attribute__((noinline))
static uint64_t return_stack_depth_call(unsigned int depth, uint64_t seed) {
    uint64_t acc = seed + 0x9e3779b97f4a7c15ull + (uint64_t)depth;
    if (depth > 0u) {
        acc ^= return_stack_depth_call(depth - 1u, seed + 0xbf58476d1ce4e5b9ull);
        __asm__ __volatile__("" : "+r"(acc) :: "memory");
    }
    acc = (acc << 13) | (acc >> 51);
    return_stack_sink ^= acc;
    return acc;
}

static unsigned int return_stack_depth_for(enum return_stack_kind kind, unsigned int i) {
    static const unsigned int neutral_seq[8] = {4u, 8u, 8u, 4u, 12u, 6u, 6u, 12u};
    static const unsigned int forward_seq[8] = {4u, 8u, 12u, 16u, 20u, 24u, 28u, 32u};
    static const unsigned int reverse_seq[8] = {32u, 28u, 24u, 20u, 16u, 12u, 8u, 4u};
    static const unsigned int shuffle_seq[8] = {8u, 24u, 4u, 28u, 12u, 32u, 16u, 20u};
    static const unsigned int sentinel_seq[8] = {6u, 14u, 22u, 30u, 10u, 18u, 26u, 34u};
    const unsigned int *seq = neutral_seq;
    switch (kind) {
        case RETURN_STACK_NEUTRAL:
            seq = neutral_seq;
            break;
        case RETURN_STACK_FORWARD:
            seq = forward_seq;
            break;
        case RETURN_STACK_REVERSE:
            seq = reverse_seq;
            break;
        case RETURN_STACK_SHUFFLE:
            seq = shuffle_seq;
            break;
        case RETURN_STACK_SENTINEL:
            seq = sentinel_seq;
            break;
    }
    return seq[i & 7u];
}

__attribute__((noinline))
static void run_return_stack_pattern(enum return_stack_kind kind, unsigned int loops) {
    uint64_t acc = return_stack_sink ^ ((uint64_t)kind << 48) ^ loops;
    for (unsigned int loop = 0; loop < loops; loop++) {
        for (unsigned int i = 0; i < 8u; i++) {
            unsigned int mixed = i + (loop * 5u);
            unsigned int depth = return_stack_depth_for(kind, mixed);
            acc ^= return_stack_depth_call(depth, acc + ((uint64_t)loop << 32) + i);
            __asm__ __volatile__("" : "+r"(acc) :: "memory");
        }
    }
    return_stack_sink = acc;
}

static uint64_t return_stack_static_digest(void) {
    static const unsigned char depths[] = {
        4u, 8u, 8u, 4u, 12u, 6u, 6u, 12u,
        4u, 8u, 12u, 16u, 20u, 24u, 28u, 32u,
        32u, 28u, 24u, 20u, 16u, 12u, 8u, 4u,
        8u, 24u, 4u, 28u, 12u, 32u, 16u, 20u,
        6u, 14u, 22u, 30u, 10u, 18u, 26u, 34u,
        'C', 'A', 'T', '_', 'C', 'A', 'S', '_',
        'R', 'E', 'T', 'U', 'R', 'N', '_', 'S',
        'T', 'A', 'C', 'K', '_', 'V', '1'
    };
    return fnv1a64(depths, sizeof(depths));
}

static int measure_return_stack_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct return_stack_sequence_spec *sequence,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_history,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    if (pin_to_core(CATCAS_CORE_B) != 0) return -EINVAL;
    run_return_stack_pattern(RETURN_STACK_NEUTRAL, RETURN_STACK_RESTORE_LOOPS);
    if (sequence->history_loops > 0u) {
        run_return_stack_pattern(sequence->history_pattern, sequence->history_loops);
    }
    *digest_after_history = return_stack_static_digest();
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    run_return_stack_pattern(RETURN_STACK_SENTINEL, RETURN_STACK_SENTINEL_LOOPS);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    run_return_stack_pattern(RETURN_STACK_NEUTRAL, RETURN_STACK_RESTORE_LOOPS);
    *digest_after_restore = return_stack_static_digest();
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_return_stack_history_mode(const char *output_root) {
    uint64_t initial_digest = return_stack_static_digest();
    int return_fds[MAX_GROUP_EVENTS];
    uint64_t return_ids[MAX_GROUP_EVENTS];
    int return_open_rc = open_group(return_stack_group, CATCAS_CORE_B,
                                    return_fds, return_ids);
    if (return_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(return_fds[i]);
    }

    const struct return_stack_sequence_spec sequences[] = {
        {"neutral_restore_only", RETURN_STACK_NEUTRAL, 0u},
        {"forward_return_stack_history", RETURN_STACK_FORWARD, RETURN_STACK_HISTORY_LOOPS},
        {"reverse_return_stack_history", RETURN_STACK_REVERSE, RETURN_STACK_HISTORY_LOOPS},
        {"shuffle_return_stack_history", RETURN_STACK_SHUFFLE, RETURN_STACK_HISTORY_LOOPS},
    };
    enum { RETURN_STACK_WINDOW_COUNT = 4 };
    struct group_result results[RETURN_STACK_WINDOW_COUNT];
    uint64_t durations[RETURN_STACK_WINDOW_COUNT];
    uint64_t digest_after_history[RETURN_STACK_WINDOW_COUNT];
    uint64_t digest_after_restore[RETURN_STACK_WINDOW_COUNT];
    int window_rc[RETURN_STACK_WINDOW_COUNT];
    for (int i = 0; i < RETURN_STACK_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        if (return_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_return_stack_window(
                return_stack_group,
                &sequences[i],
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool patterns_unchanged_after_history = true;
    bool patterns_unchanged_after_restore = true;
    for (int i = 0; i < RETURN_STACK_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        patterns_unchanged_after_history =
            patterns_unchanged_after_history && digest_after_history[i] == initial_digest;
        patterns_unchanged_after_restore =
            patterns_unchanged_after_restore && digest_after_restore[i] == initial_digest;
    }

    uint64_t identity_misses = event_value_by_name(&results[0], return_stack_group, "retired_mispredicted_branch_instructions");
    uint64_t forward_misses = event_value_by_name(&results[1], return_stack_group, "retired_mispredicted_branch_instructions");
    uint64_t reverse_misses = event_value_by_name(&results[2], return_stack_group, "retired_mispredicted_branch_instructions");
    uint64_t shuffle_misses = event_value_by_name(&results[3], return_stack_group, "retired_mispredicted_branch_instructions");
    uint64_t miss_delta = abs_diff_u64(forward_misses, reverse_misses);
    uint64_t miss_control = abs_diff_u64(identity_misses, shuffle_misses);
    uint64_t miss_threshold = max_u64(32u, 3u * miss_control);
    bool miss_signal = miss_delta > miss_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(10000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;

    bool return_stack_history_response = return_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && patterns_unchanged_after_history &&
        patterns_unchanged_after_restore && (miss_signal || duration_signal);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_RETURN_STACK_HISTORY_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", RETURN_STACK_HISTORY_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", return_stack_history_response ? "RETURN_STACK_HISTORY_RESPONSE_FOUND" : "RETURN_STACK_HISTORY_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Return-stack call/return timing/PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, target coupling, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"return_stack_core\": %d},\n", CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_code_only\": true, \"generated_executable_memory\": false, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false},\n");
    fprintf(out, "  \"return_stack_carrier\": {\"compiled_call_site\": true, \"restore_loops\": %u, \"history_loops\": %u, \"sentinel_loops\": %u, \"depths_per_loop\": 8, \"max_depth\": 34, \"digest_kind\": \"fnv1a64_static_depth_pattern\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        RETURN_STACK_RESTORE_LOOPS, RETURN_STACK_HISTORY_LOOPS, RETURN_STACK_SENTINEL_LOOPS, initial_digest);
    fprintf(out, "  \"selected_group\": \"return_stack_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", return_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, return_stack_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"balanced recursive public call-depth sequence over CAT_CAS-owned compiled code\", \"restore\": \"neutral call-depth wash before and after every measured window\", \"sentinel\": \"fixed recursive call-depth sequence on the same core\", \"primary_acceptance\": \"forward/reverse retired_mispredicted_branch_instructions delta exceeds max(32, 3 * neutral/shuffle control spread) or duration delta exceeds max(10000 ns, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < RETURN_STACK_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"digest_after_history_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"patterns_unchanged_after_history\": %s, \"patterns_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, return_stack_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == RETURN_STACK_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"retired_mispredicted_branch_instructions\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_misses, forward_misses, reverse_misses, shuffle_misses, miss_delta, miss_control, miss_threshold, json_bool(miss_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"patterns_unchanged_after_history\": %s, \"patterns_unchanged_after_restore\": %s, \"branch_miss_signal\": %s, \"duration_signal\": %s, \"return_stack_history_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(patterns_unchanged_after_history),
        json_bool(patterns_unchanged_after_restore),
        json_bool(miss_signal),
        json_bool(duration_signal),
        json_bool(return_stack_history_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"return_stack_group\"}\n",
        return_stack_history_response ? "RETURN_STACK_HISTORY_RESPONSE_FOUND" : "RETURN_STACK_HISTORY_RESPONSE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

static volatile double fp_pipeline_sink = 1.125;

__attribute__((noinline))
static double fp_pipeline_block0(double input) {
    volatile double x = input + 0.000000101;
    for (unsigned int i = 0; i < 16u; i++) {
        x = (x * 1.000000119) + 0.000000013;
        __asm__ __volatile__("" ::: "memory");
    }
    return x;
}

__attribute__((noinline))
static double fp_pipeline_block1(double input) {
    volatile double x = input + 0.000000211;
    for (unsigned int i = 0; i < 16u; i++) {
        x = (x + 0.000000017) / 1.000000171;
        __asm__ __volatile__("" ::: "memory");
    }
    return x;
}

__attribute__((noinline))
static double fp_pipeline_block2(double input) {
    volatile double x = input + 0.000000307;
    for (unsigned int i = 0; i < 16u; i++) {
        x = (x * 0.999999881) + 0.000000031;
        __asm__ __volatile__("" ::: "memory");
    }
    return x;
}

__attribute__((noinline))
static double fp_pipeline_block3(double input) {
    volatile double x = input + 0.000000409;
    for (unsigned int i = 0; i < 16u; i++) {
        x = ((x / 1.000000233) * 1.000000017) + 0.000000007;
        __asm__ __volatile__("" ::: "memory");
    }
    return x;
}

__attribute__((noinline))
static double run_fp_pipeline_block(unsigned int block, double input) {
    switch (block & 3u) {
        case 0u:
            return fp_pipeline_block0(input);
        case 1u:
            return fp_pipeline_block1(input);
        case 2u:
            return fp_pipeline_block2(input);
        default:
            return fp_pipeline_block3(input);
    }
}

static unsigned int fp_pipeline_block_index(enum fp_pipeline_kind kind, unsigned int i) {
    static const unsigned int neutral_seq[8] = {0u, 2u, 1u, 3u, 3u, 1u, 2u, 0u};
    static const unsigned int forward_seq[8] = {0u, 1u, 2u, 3u, 0u, 1u, 2u, 3u};
    static const unsigned int reverse_seq[8] = {3u, 2u, 1u, 0u, 3u, 2u, 1u, 0u};
    static const unsigned int shuffle_seq[8] = {1u, 3u, 0u, 2u, 2u, 0u, 3u, 1u};
    static const unsigned int sentinel_seq[8] = {0u, 3u, 1u, 2u, 1u, 0u, 2u, 3u};
    const unsigned int *seq = neutral_seq;
    switch (kind) {
        case FP_PIPELINE_NEUTRAL:
            seq = neutral_seq;
            break;
        case FP_PIPELINE_FORWARD:
            seq = forward_seq;
            break;
        case FP_PIPELINE_REVERSE:
            seq = reverse_seq;
            break;
        case FP_PIPELINE_SHUFFLE:
            seq = shuffle_seq;
            break;
        case FP_PIPELINE_SENTINEL:
            seq = sentinel_seq;
            break;
    }
    return seq[i & 7u];
}

__attribute__((noinline))
static void run_fp_pipeline_pattern(enum fp_pipeline_kind kind, unsigned int loops) {
    double x = fp_pipeline_sink + (double)(kind + 1u) * 0.000001;
    for (unsigned int loop = 0; loop < loops; loop++) {
        for (unsigned int i = 0; i < 32u; i++) {
            unsigned int mixed = i + (loop * 7u);
            x = run_fp_pipeline_block(fp_pipeline_block_index(kind, mixed), x);
            __asm__ __volatile__("" ::: "memory");
        }
    }
    fp_pipeline_sink = x;
}

static uint64_t fp_pipeline_static_digest(void) {
    static const unsigned char patterns[] = {
        0u, 2u, 1u, 3u, 3u, 1u, 2u, 0u,
        0u, 1u, 2u, 3u, 0u, 1u, 2u, 3u,
        3u, 2u, 1u, 0u, 3u, 2u, 1u, 0u,
        1u, 3u, 0u, 2u, 2u, 0u, 3u, 1u,
        0u, 3u, 1u, 2u, 1u, 0u, 2u, 3u,
        'C', 'A', 'T', '_', 'C', 'A', 'S', '_',
        'F', 'P', '_', 'P', 'I', 'P', 'E', '_', 'V', '1'
    };
    return fnv1a64(patterns, sizeof(patterns));
}

static int measure_fp_pipeline_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct fp_pipeline_sequence_spec *sequence,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_history,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    if (pin_to_core(CATCAS_CORE_B) != 0) return -EINVAL;
    run_fp_pipeline_pattern(FP_PIPELINE_NEUTRAL, FP_PIPELINE_RESTORE_LOOPS);
    if (sequence->history_loops > 0u) {
        run_fp_pipeline_pattern(sequence->history_pattern, sequence->history_loops);
    }
    *digest_after_history = fp_pipeline_static_digest();
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    run_fp_pipeline_pattern(FP_PIPELINE_SENTINEL, FP_PIPELINE_SENTINEL_LOOPS);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    run_fp_pipeline_pattern(FP_PIPELINE_NEUTRAL, FP_PIPELINE_RESTORE_LOOPS);
    *digest_after_restore = fp_pipeline_static_digest();
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_fp_pipeline_history_mode(const char *output_root) {
    uint64_t initial_digest = fp_pipeline_static_digest();
    int fp_fds[MAX_GROUP_EVENTS];
    uint64_t fp_ids[MAX_GROUP_EVENTS];
    int fp_open_rc = open_group(fp_pipeline_group, CATCAS_CORE_B, fp_fds, fp_ids);
    if (fp_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fp_fds[i]);
    }

    const struct fp_pipeline_sequence_spec sequences[] = {
        {"neutral_restore_only", FP_PIPELINE_NEUTRAL, 0u},
        {"forward_fp_pipeline_history", FP_PIPELINE_FORWARD, FP_PIPELINE_HISTORY_LOOPS},
        {"reverse_fp_pipeline_history", FP_PIPELINE_REVERSE, FP_PIPELINE_HISTORY_LOOPS},
        {"shuffle_fp_pipeline_history", FP_PIPELINE_SHUFFLE, FP_PIPELINE_HISTORY_LOOPS},
    };
    enum { FP_PIPELINE_WINDOW_COUNT = 4 };
    struct group_result results[FP_PIPELINE_WINDOW_COUNT];
    uint64_t durations[FP_PIPELINE_WINDOW_COUNT];
    uint64_t digest_after_history[FP_PIPELINE_WINDOW_COUNT];
    uint64_t digest_after_restore[FP_PIPELINE_WINDOW_COUNT];
    int window_rc[FP_PIPELINE_WINDOW_COUNT];
    for (int i = 0; i < FP_PIPELINE_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        if (fp_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_fp_pipeline_window(
                fp_pipeline_group,
                &sequences[i],
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool patterns_unchanged_after_history = true;
    bool patterns_unchanged_after_restore = true;
    for (int i = 0; i < FP_PIPELINE_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        patterns_unchanged_after_history =
            patterns_unchanged_after_history && digest_after_history[i] == initial_digest;
        patterns_unchanged_after_restore =
            patterns_unchanged_after_restore && digest_after_restore[i] == initial_digest;
    }

    uint64_t identity_cycles = event_value_by_name(&results[0], fp_pipeline_group, "cpu_cycles_not_halted");
    uint64_t forward_cycles = event_value_by_name(&results[1], fp_pipeline_group, "cpu_cycles_not_halted");
    uint64_t reverse_cycles = event_value_by_name(&results[2], fp_pipeline_group, "cpu_cycles_not_halted");
    uint64_t shuffle_cycles = event_value_by_name(&results[3], fp_pipeline_group, "cpu_cycles_not_halted");
    uint64_t cycles_delta = abs_diff_u64(forward_cycles, reverse_cycles);
    uint64_t cycles_control = abs_diff_u64(identity_cycles, shuffle_cycles);
    uint64_t cycles_threshold = max_u64(10000u, 3u * cycles_control);
    bool cycles_signal = cycles_delta > cycles_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(10000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;

    bool fp_pipeline_history_response = fp_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && patterns_unchanged_after_history &&
        patterns_unchanged_after_restore && (cycles_signal || duration_signal);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_FP_PIPELINE_HISTORY_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", FP_PIPELINE_HISTORY_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", fp_pipeline_history_response ? "FP_PIPELINE_HISTORY_RESPONSE_FOUND" : "FP_PIPELINE_HISTORY_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Floating-point/division pipeline timing/PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, target coupling, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"fp_pipeline_core\": %d},\n", CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_compute_only\": true, \"generated_executable_memory\": false, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false},\n");
    fprintf(out, "  \"fp_pipeline_carrier\": {\"compiled_blocks\": 4, \"restore_loops\": %u, \"history_loops\": %u, \"sentinel_loops\": %u, \"operations_per_block\": 16, \"sentinel_blocks_per_loop\": 32, \"digest_kind\": \"fnv1a64_static_pattern\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        FP_PIPELINE_RESTORE_LOOPS, FP_PIPELINE_HISTORY_LOOPS, FP_PIPELINE_SENTINEL_LOOPS, initial_digest);
    fprintf(out, "  \"selected_group\": \"fp_pipeline_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", fp_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, fp_pipeline_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"balanced public floating-point add/multiply/divide block order over CAT_CAS-owned compiled code\", \"restore\": \"neutral FP pipeline wash before and after every measured window\", \"sentinel\": \"fixed FP arithmetic block sequence on the same core\", \"primary_acceptance\": \"forward/reverse cpu_cycles_not_halted delta exceeds max(10000, 3 * neutral/shuffle control spread) or duration delta exceeds max(10000 ns, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < FP_PIPELINE_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"digest_after_history_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"patterns_unchanged_after_history\": %s, \"patterns_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, fp_pipeline_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == FP_PIPELINE_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"cpu_cycles_not_halted\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_cycles, forward_cycles, reverse_cycles, shuffle_cycles, cycles_delta, cycles_control, cycles_threshold, json_bool(cycles_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"patterns_unchanged_after_history\": %s, \"patterns_unchanged_after_restore\": %s, \"cycle_signal\": %s, \"duration_signal\": %s, \"fp_pipeline_history_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(patterns_unchanged_after_history),
        json_bool(patterns_unchanged_after_restore),
        json_bool(cycles_signal),
        json_bool(duration_signal),
        json_bool(fp_pipeline_history_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"fp_pipeline_group\"}\n",
        fp_pipeline_history_response ? "FP_PIPELINE_HISTORY_RESPONSE_FOUND" : "FP_PIPELINE_HISTORY_RESPONSE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

static volatile uint64_t page_permission_sink = 0;

static size_t page_permission_index(enum page_permission_kind kind, size_t i, size_t page_count) {
    size_t mask = page_count - 1u;
    switch (kind) {
        case PAGE_PERMISSION_NEUTRAL:
            return ((i * 17u) + (i >> 2)) & mask;
        case PAGE_PERMISSION_FORWARD:
            return i & mask;
        case PAGE_PERMISSION_REVERSE:
            return (page_count - 1u - (i & mask)) & mask;
        case PAGE_PERMISSION_SHUFFLE:
            return ((i * 1103515245u) + 12345u) & mask;
        case PAGE_PERMISSION_SENTINEL:
            return ((i * 37u) ^ (i >> 1) ^ 0x5au) & mask;
    }
    return i & mask;
}

static int restore_page_permissions(unsigned char *pages, size_t page_size, size_t page_count) {
    for (size_t page = 0; page < page_count; page++) {
        unsigned char *addr = pages + (page * page_size);
        if (mprotect(addr, page_size, PROT_READ | PROT_WRITE) != 0) return -errno;
        addr[0] = (unsigned char)((addr[0] + 0u) & 0xffu);
    }
    return 0;
}

static int run_page_permission_history_pattern(
    unsigned char *pages,
    size_t page_size,
    size_t page_count,
    enum page_permission_kind kind,
    unsigned int loops
) {
    uint64_t acc = page_permission_sink + page_count + loops + (uint64_t)kind;
    size_t span = page_count;
    for (unsigned int round = 0; round < loops; round++) {
        for (size_t i = 0; i < span; i++) {
            size_t index = page_permission_index(kind, i + ((size_t)round * 29u), page_count);
            unsigned char *addr = pages + (index * page_size);
            if (mprotect(addr, page_size, PROT_NONE) != 0) return -errno;
            if (mprotect(addr, page_size, PROT_READ | PROT_WRITE) != 0) return -errno;
            acc ^= ((uint64_t)index + 1u) * 0x9e3779b97f4a7c15ull;
            addr[0] = (unsigned char)((addr[0] + 0u) & 0xffu);
            __asm__ __volatile__("" : "+r"(acc) :: "memory");
        }
    }
    page_permission_sink = acc;
    return 0;
}

__attribute__((noinline))
static void run_page_permission_sentinel(
    volatile unsigned char *pages,
    size_t page_size,
    size_t page_count,
    unsigned int loops
) {
    uint64_t acc = page_permission_sink + loops + 0x94d049bb133111ebull;
    size_t span = page_count * 2u;
    for (unsigned int round = 0; round < loops; round++) {
        for (size_t i = 0; i < span; i++) {
            size_t index = page_permission_index(
                PAGE_PERMISSION_SENTINEL, i + ((size_t)round * 43u), page_count);
            acc += (uint64_t)pages[index * page_size];
            acc ^= ((uint64_t)index + 1u) * 0xbf58476d1ce4e5b9ull;
            __asm__ __volatile__("" : "+r"(acc) :: "memory");
        }
    }
    page_permission_sink = acc;
}

static int measure_page_permission_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct page_permission_sequence_spec *sequence,
    unsigned char *pages,
    size_t page_size,
    size_t page_count,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_history,
    uint64_t *digest_after_restore,
    int *permission_restore_probe_rc
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    if (pin_to_core(CATCAS_CORE_B) != 0) return -EINVAL;
    int restore_rc = restore_page_permissions(pages, page_size, page_count);
    if (restore_rc != 0) return restore_rc;
    for (unsigned int i = 0; i < PAGE_PERMISSION_RESTORE_LOOPS; i++) {
        int wash_rc = run_page_permission_history_pattern(
            pages, page_size, page_count, PAGE_PERMISSION_NEUTRAL, 1u);
        if (wash_rc != 0) return wash_rc;
    }
    if (sequence->history_loops > 0u) {
        int history_rc = run_page_permission_history_pattern(
            pages, page_size, page_count, sequence->history_pattern,
            sequence->history_loops);
        if (history_rc != 0) return history_rc;
    }
    *digest_after_history = fnv1a64(pages, page_size * page_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    run_page_permission_sentinel(pages, page_size, page_count, PAGE_PERMISSION_SENTINEL_LOOPS);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    int final_restore_rc = restore_page_permissions(pages, page_size, page_count);
    if (permission_restore_probe_rc) *permission_restore_probe_rc = final_restore_rc;
    if (final_restore_rc != 0) return final_restore_rc;
    *digest_after_restore = fnv1a64(pages, page_size * page_count);
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_page_permission_history_mode(const char *output_root) {
    long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        fprintf(stderr, "page size unavailable\n");
        return 1;
    }
    size_t page_size = (size_t)page_size_long;
    size_t byte_count = page_size * PAGE_PERMISSION_PAGE_COUNT;
    unsigned char *pages = NULL;
    if (posix_memalign((void **)&pages, page_size, byte_count) != 0) {
        fprintf(stderr, "page permission buffer allocation failed\n");
        return 1;
    }
    for (size_t page = 0; page < PAGE_PERMISSION_PAGE_COUNT; page++) {
        pages[page * page_size] = (unsigned char)((page * 97u + 23u) & 0xffu);
    }
    uint64_t initial_digest = fnv1a64(pages, byte_count);

    int permission_fds[MAX_GROUP_EVENTS];
    uint64_t permission_ids[MAX_GROUP_EVENTS];
    int permission_open_rc = open_group(page_permission_group, CATCAS_CORE_B,
                                        permission_fds, permission_ids);
    if (permission_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(permission_fds[i]);
    }

    const struct page_permission_sequence_spec sequences[] = {
        {"neutral_restore_only", PAGE_PERMISSION_NEUTRAL, 0u},
        {"forward_page_permission_history", PAGE_PERMISSION_FORWARD, PAGE_PERMISSION_HISTORY_LOOPS},
        {"reverse_page_permission_history", PAGE_PERMISSION_REVERSE, PAGE_PERMISSION_HISTORY_LOOPS},
        {"shuffle_page_permission_history", PAGE_PERMISSION_SHUFFLE, PAGE_PERMISSION_HISTORY_LOOPS},
    };
    enum { PAGE_PERMISSION_WINDOW_COUNT = 4 };
    struct group_result results[PAGE_PERMISSION_WINDOW_COUNT];
    uint64_t durations[PAGE_PERMISSION_WINDOW_COUNT];
    uint64_t digest_after_history[PAGE_PERMISSION_WINDOW_COUNT];
    uint64_t digest_after_restore[PAGE_PERMISSION_WINDOW_COUNT];
    int restore_probe_rc[PAGE_PERMISSION_WINDOW_COUNT];
    int window_rc[PAGE_PERMISSION_WINDOW_COUNT];
    for (int i = 0; i < PAGE_PERMISSION_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        restore_probe_rc[i] = -1;
        if (permission_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_page_permission_window(
                page_permission_group,
                &sequences[i],
                pages,
                page_size,
                PAGE_PERMISSION_PAGE_COUNT,
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i],
                &restore_probe_rc[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool bytes_unchanged_after_history = true;
    bool bytes_unchanged_after_restore = true;
    bool permissions_restored = true;
    for (int i = 0; i < PAGE_PERMISSION_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        bytes_unchanged_after_history =
            bytes_unchanged_after_history && digest_after_history[i] == initial_digest;
        bytes_unchanged_after_restore =
            bytes_unchanged_after_restore && digest_after_restore[i] == initial_digest;
        permissions_restored = permissions_restored && restore_probe_rc[i] == 0;
    }

    uint64_t identity_misses = event_value_by_name(&results[0], page_permission_group, "cache_misses");
    uint64_t forward_misses = event_value_by_name(&results[1], page_permission_group, "cache_misses");
    uint64_t reverse_misses = event_value_by_name(&results[2], page_permission_group, "cache_misses");
    uint64_t shuffle_misses = event_value_by_name(&results[3], page_permission_group, "cache_misses");
    uint64_t miss_delta = abs_diff_u64(forward_misses, reverse_misses);
    uint64_t miss_control = abs_diff_u64(identity_misses, shuffle_misses);
    uint64_t miss_threshold = max_u64(32u, 3u * miss_control);
    bool miss_signal = miss_delta > miss_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(10000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;

    bool page_permission_history_response = permission_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && bytes_unchanged_after_history &&
        bytes_unchanged_after_restore && permissions_restored &&
        (miss_signal || duration_signal);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_PAGE_PERMISSION_HISTORY_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) {
        free(pages);
        return 1;
    }
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        free(pages);
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", PAGE_PERMISSION_HISTORY_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", page_permission_history_response ? "PAGE_PERMISSION_HISTORY_RESPONSE_FOUND" : "PAGE_PERMISSION_HISTORY_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Owned page-permission-state timing/PMU discriminator only; no memory disclosure, isolation bypass, path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, target coupling, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"page_permission_core\": %d},\n", CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_pages_only\": true, \"synthetic_public_contents_only\": true, \"measured_page_faults\": false, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false, \"cross_domain_observation\": false},\n");
    fprintf(out, "  \"page_permission_carrier\": {\"page_count\": %u, \"page_bytes\": %zu, \"byte_count\": %zu, \"restore_loops\": %u, \"history_loops\": %u, \"sentinel_loops\": %u, \"permission_mutation_outside_measured_window\": true, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        PAGE_PERMISSION_PAGE_COUNT, page_size, byte_count,
        PAGE_PERMISSION_RESTORE_LOOPS, PAGE_PERMISSION_HISTORY_LOOPS,
        PAGE_PERMISSION_SENTINEL_LOOPS, initial_digest);
    fprintf(out, "  \"selected_group\": \"page_permission_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", permission_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, page_permission_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"owned-page mprotect PROT_NONE then PROT_READ|PROT_WRITE flips outside the measured window\", \"restore\": \"all owned pages restored to PROT_READ|PROT_WRITE and deterministically probed\", \"sentinel\": \"fixed owned-page read sequence with no intentional page fault\", \"primary_acceptance\": \"forward/reverse cache_misses delta exceeds max(32, 3 * neutral/shuffle control spread) or duration delta exceeds max(10000 ns, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < PAGE_PERMISSION_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"permission_restore_probe_rc\": %d, \"digest_after_history_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            restore_probe_rc[i],
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, page_permission_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == PAGE_PERMISSION_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"cache_misses\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_misses, forward_misses, reverse_misses, shuffle_misses, miss_delta, miss_control, miss_threshold, json_bool(miss_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"permissions_restored\": %s, \"cache_miss_signal\": %s, \"duration_signal\": %s, \"page_permission_history_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(bytes_unchanged_after_history),
        json_bool(bytes_unchanged_after_restore),
        json_bool(permissions_restored),
        json_bool(miss_signal),
        json_bool(duration_signal),
        json_bool(page_permission_history_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"page_permission_group\"}\n",
        page_permission_history_response ? "PAGE_PERMISSION_HISTORY_RESPONSE_FOUND" : "PAGE_PERMISSION_HISTORY_RESPONSE_NOT_ESTABLISHED",
        result_path);
    restore_page_permissions(pages, page_size, PAGE_PERMISSION_PAGE_COUNT);
    free(pages);
    return 0;
}

struct owned_recovery_context {
    unsigned char *pages;
    size_t page_size;
    size_t page_count;
    volatile sig_atomic_t active;
    volatile sig_atomic_t handler_count;
    volatile sig_atomic_t escaped_count;
};

static struct owned_recovery_context owned_recovery_ctx = {NULL, 0u, 0u, 0, 0, 0};
static struct sigaction owned_recovery_old_action;
static bool owned_recovery_handler_installed = false;
static volatile uint64_t owned_recovery_sink = 0;

static void owned_recovery_sigsegv_handler(int sig, siginfo_t *info, void *ucontext) {
    (void)ucontext;
    uintptr_t fault_addr = info ? (uintptr_t)info->si_addr : 0u;
    uintptr_t base = (uintptr_t)owned_recovery_ctx.pages;
    size_t byte_count = owned_recovery_ctx.page_size * owned_recovery_ctx.page_count;
    uintptr_t end = base + byte_count;
    if (owned_recovery_ctx.active != 0 && base != 0u &&
        fault_addr >= base && fault_addr < end && owned_recovery_ctx.page_size > 0u) {
        size_t page_index = (size_t)((fault_addr - base) / owned_recovery_ctx.page_size);
        unsigned char *page = owned_recovery_ctx.pages + (page_index * owned_recovery_ctx.page_size);
        if (mprotect(page, owned_recovery_ctx.page_size, PROT_READ | PROT_WRITE) == 0) {
            owned_recovery_ctx.handler_count++;
            return;
        }
    }
    owned_recovery_ctx.escaped_count++;
    owned_recovery_ctx.active = 0;
    (void)sigaction(SIGSEGV, &owned_recovery_old_action, NULL);
    (void)raise(sig);
}

static int install_owned_recovery_handler(unsigned char *pages, size_t page_size, size_t page_count) {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_sigaction = owned_recovery_sigsegv_handler;
    action.sa_flags = SA_SIGINFO;
    if (sigemptyset(&action.sa_mask) != 0) return -errno;
    owned_recovery_ctx.pages = pages;
    owned_recovery_ctx.page_size = page_size;
    owned_recovery_ctx.page_count = page_count;
    owned_recovery_ctx.active = 0;
    owned_recovery_ctx.handler_count = 0;
    owned_recovery_ctx.escaped_count = 0;
    if (sigaction(SIGSEGV, &action, &owned_recovery_old_action) != 0) return -errno;
    owned_recovery_handler_installed = true;
    return 0;
}

static int restore_owned_recovery_handler(void) {
    owned_recovery_ctx.active = 0;
    if (!owned_recovery_handler_installed) return 0;
    if (sigaction(SIGSEGV, &owned_recovery_old_action, NULL) != 0) return -errno;
    owned_recovery_handler_installed = false;
    owned_recovery_ctx.pages = NULL;
    owned_recovery_ctx.page_size = 0u;
    owned_recovery_ctx.page_count = 0u;
    return 0;
}

static int restore_owned_recovery_permissions(unsigned char *pages, size_t page_size, size_t page_count) {
    for (size_t page = 0; page < page_count; page++) {
        unsigned char *addr = pages + (page * page_size);
        if (mprotect(addr, page_size, PROT_READ | PROT_WRITE) != 0) return -errno;
        addr[0] = (unsigned char)((addr[0] + 0u) & 0xffu);
    }
    return 0;
}

static int check_owned_recovery_resident(unsigned char *pages, size_t page_size, size_t page_count) {
    unsigned char vec[OWNED_RECOVERY_PAGE_COUNT];
    if (page_count > OWNED_RECOVERY_PAGE_COUNT) return -EINVAL;
    memset(vec, 0, sizeof(vec));
    if (mincore(pages, page_size * page_count, vec) != 0) return -errno;
    for (size_t page = 0; page < page_count; page++) {
        if ((vec[page] & 0x1u) == 0u) return -ENOENT;
    }
    return 0;
}

static size_t owned_recovery_group_index(enum owned_recovery_kind kind, size_t position) {
    static const size_t neutral_order[4] = {0u, 2u, 1u, 3u};
    static const size_t forward_order[4] = {0u, 1u, 2u, 3u};
    static const size_t reverse_order[4] = {3u, 2u, 1u, 0u};
    static const size_t shuffle_order[4] = {1u, 3u, 0u, 2u};
    const size_t *order = forward_order;
    switch (kind) {
        case OWNED_RECOVERY_NEUTRAL:
            order = neutral_order;
            break;
        case OWNED_RECOVERY_FORWARD:
            order = forward_order;
            break;
        case OWNED_RECOVERY_REVERSE:
            order = reverse_order;
            break;
        case OWNED_RECOVERY_SHUFFLE:
            order = shuffle_order;
            break;
        case OWNED_RECOVERY_SENTINEL:
            order = forward_order;
            break;
    }
    return order[position & 3u];
}

static int run_owned_recovery_history_pattern(
    unsigned char *pages,
    size_t page_size,
    size_t page_count,
    enum owned_recovery_kind kind,
    unsigned int loops,
    int *handler_delta,
    long *minor_delta
) {
    struct rusage before;
    struct rusage after;
    if (handler_delta) *handler_delta = 0;
    if (minor_delta) *minor_delta = 0;
    if (page_count == 0u || (page_count % 4u) != 0u) return -EINVAL;
    if (getrusage(RUSAGE_SELF, &before) != 0) return -errno;
    sig_atomic_t handler_before = owned_recovery_ctx.handler_count;
    sig_atomic_t escaped_before = owned_recovery_ctx.escaped_count;
    size_t group_pages = page_count / 4u;
    uint64_t acc = owned_recovery_sink + page_count + loops + (uint64_t)kind;
    owned_recovery_ctx.active = 1;
    for (unsigned int loop = 0; loop < loops; loop++) {
        for (size_t group_slot = 0; group_slot < 4u; group_slot++) {
            size_t group = owned_recovery_group_index(kind, group_slot);
            for (size_t offset = 0; offset < group_pages; offset++) {
                size_t page_index = (group * group_pages) + offset;
                unsigned char *addr = pages + (page_index * page_size);
                if (mprotect(addr, page_size, PROT_NONE) != 0) {
                    owned_recovery_ctx.active = 0;
                    return -errno;
                }
                volatile unsigned char *probe = addr;
                acc += (uint64_t)(*probe);
                acc ^= ((uint64_t)page_index + 1u) * 0x9e3779b97f4a7c15ull;
                __asm__ __volatile__("" : "+r"(acc) :: "memory");
            }
        }
    }
    owned_recovery_ctx.active = 0;
    if (getrusage(RUSAGE_SELF, &after) != 0) return -errno;
    if (handler_delta) {
        *handler_delta = (int)(owned_recovery_ctx.handler_count - handler_before);
    }
    if (minor_delta) {
        *minor_delta = after.ru_minflt - before.ru_minflt;
    }
    owned_recovery_sink = acc;
    return owned_recovery_ctx.escaped_count == escaped_before ? 0 : -EFAULT;
}

__attribute__((noinline))
static int run_owned_recovery_sentinel(
    volatile unsigned char *pages,
    size_t page_size,
    size_t page_count,
    unsigned int loops,
    int *handler_delta,
    long *minor_delta
) {
    struct rusage before;
    struct rusage after;
    if (handler_delta) *handler_delta = 0;
    if (minor_delta) *minor_delta = 0;
    if (page_count == 0u || (page_count % 4u) != 0u) return -EINVAL;
    if (getrusage(RUSAGE_SELF, &before) != 0) return -errno;
    sig_atomic_t handler_before = owned_recovery_ctx.handler_count;
    sig_atomic_t escaped_before = owned_recovery_ctx.escaped_count;
    size_t group_pages = page_count / 4u;
    uint64_t acc = owned_recovery_sink + loops + 0xd6e8feb86659fd93ull;
    for (unsigned int loop = 0; loop < loops; loop++) {
        for (size_t group_slot = 0; group_slot < 4u; group_slot++) {
            size_t group = owned_recovery_group_index(OWNED_RECOVERY_SENTINEL, group_slot);
            for (size_t offset = 0; offset < group_pages; offset++) {
                size_t page_index = (group * group_pages) + offset;
                acc += (uint64_t)pages[page_index * page_size];
                acc ^= ((uint64_t)page_index + 1u) * 0xbf58476d1ce4e5b9ull;
                __asm__ __volatile__("" : "+r"(acc) :: "memory");
            }
        }
    }
    if (getrusage(RUSAGE_SELF, &after) != 0) return -errno;
    if (handler_delta) {
        *handler_delta = (int)(owned_recovery_ctx.handler_count - handler_before);
    }
    if (minor_delta) {
        *minor_delta = after.ru_minflt - before.ru_minflt;
    }
    owned_recovery_sink = acc;
    return owned_recovery_ctx.escaped_count == escaped_before ? 0 : -EFAULT;
}

static int measure_owned_recovery_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct owned_recovery_sequence_spec *sequence,
    unsigned char *pages,
    size_t page_size,
    size_t page_count,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_history,
    uint64_t *digest_after_restore,
    int *history_handler_delta,
    long *history_minor_delta,
    int *sentinel_handler_delta,
    long *sentinel_minor_delta,
    int *resident_probe_rc,
    int *permission_restore_probe_rc
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    if (pin_to_core(CATCAS_CORE_B) != 0) return -EINVAL;
    int restore_rc = restore_owned_recovery_permissions(pages, page_size, page_count);
    if (restore_rc != 0) return restore_rc;
    int resident_before_rc = check_owned_recovery_resident(pages, page_size, page_count);
    if (resident_probe_rc) *resident_probe_rc = resident_before_rc;
    if (resident_before_rc != 0) return resident_before_rc;
    int history_rc = run_owned_recovery_history_pattern(
        pages, page_size, page_count, sequence->history_pattern,
        sequence->history_loops, history_handler_delta, history_minor_delta);
    if (history_rc != 0) {
        (void)restore_owned_recovery_permissions(pages, page_size, page_count);
        return history_rc;
    }
    *digest_after_history = fnv1a64(pages, page_size * page_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        (void)restore_owned_recovery_permissions(pages, page_size, page_count);
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        (void)restore_owned_recovery_permissions(pages, page_size, page_count);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        (void)restore_owned_recovery_permissions(pages, page_size, page_count);
        return -saved;
    }
    int sentinel_rc = run_owned_recovery_sentinel(
        pages, page_size, page_count, OWNED_RECOVERY_SENTINEL_LOOPS,
        sentinel_handler_delta, sentinel_minor_delta);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        (void)restore_owned_recovery_permissions(pages, page_size, page_count);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (sentinel_rc != 0) {
        (void)restore_owned_recovery_permissions(pages, page_size, page_count);
        return sentinel_rc;
    }
    if (read_rc != 0) {
        (void)restore_owned_recovery_permissions(pages, page_size, page_count);
        return read_rc;
    }
    int final_restore_rc = restore_owned_recovery_permissions(pages, page_size, page_count);
    if (permission_restore_probe_rc) *permission_restore_probe_rc = final_restore_rc;
    if (final_restore_rc != 0) return final_restore_rc;
    int resident_after_rc = check_owned_recovery_resident(pages, page_size, page_count);
    if (resident_probe_rc) *resident_probe_rc = resident_after_rc;
    if (resident_after_rc != 0) return resident_after_rc;
    *digest_after_restore = fnv1a64(pages, page_size * page_count);
    *duration_ns = end >= start ? end - start : 0;
    return 0;
}

static int run_owned_recovery_history_mode(const char *output_root) {
    long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        fprintf(stderr, "page size unavailable\n");
        return 1;
    }
    size_t page_size = (size_t)page_size_long;
    size_t byte_count = page_size * OWNED_RECOVERY_PAGE_COUNT;
    unsigned char *pages = mmap(NULL, byte_count, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED) {
        fprintf(stderr, "owned recovery mmap failed: %s\n", strerror(errno));
        return 1;
    }
    for (size_t page = 0; page < OWNED_RECOVERY_PAGE_COUNT; page++) {
        pages[page * page_size] = (unsigned char)((page * 131u + 41u) & 0xffu);
    }
    uint64_t initial_digest = fnv1a64(pages, byte_count);
    int initial_resident_rc = check_owned_recovery_resident(pages, page_size, OWNED_RECOVERY_PAGE_COUNT);

    int recovery_fds[MAX_GROUP_EVENTS];
    uint64_t recovery_ids[MAX_GROUP_EVENTS];
    int recovery_open_rc = open_group(owned_recovery_group, CATCAS_CORE_B,
                                      recovery_fds, recovery_ids);
    if (recovery_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(recovery_fds[i]);
    }
    int handler_install_rc = 0;
    if (recovery_open_rc == 0 && initial_resident_rc == 0) {
        handler_install_rc = install_owned_recovery_handler(
            pages, page_size, OWNED_RECOVERY_PAGE_COUNT);
    }

    const struct owned_recovery_sequence_spec sequences[] = {
        {"neutral_owned_recovery_history", OWNED_RECOVERY_NEUTRAL, OWNED_RECOVERY_HISTORY_LOOPS},
        {"forward_owned_recovery_history", OWNED_RECOVERY_FORWARD, OWNED_RECOVERY_HISTORY_LOOPS},
        {"reverse_owned_recovery_history", OWNED_RECOVERY_REVERSE, OWNED_RECOVERY_HISTORY_LOOPS},
        {"shuffle_owned_recovery_history", OWNED_RECOVERY_SHUFFLE, OWNED_RECOVERY_HISTORY_LOOPS},
    };
    enum { OWNED_RECOVERY_WINDOW_COUNT = 4 };
    struct group_result results[OWNED_RECOVERY_WINDOW_COUNT];
    uint64_t durations[OWNED_RECOVERY_WINDOW_COUNT];
    uint64_t digest_after_history[OWNED_RECOVERY_WINDOW_COUNT];
    uint64_t digest_after_restore[OWNED_RECOVERY_WINDOW_COUNT];
    int history_handler_delta[OWNED_RECOVERY_WINDOW_COUNT];
    long history_minor_delta[OWNED_RECOVERY_WINDOW_COUNT];
    int sentinel_handler_delta[OWNED_RECOVERY_WINDOW_COUNT];
    long sentinel_minor_delta[OWNED_RECOVERY_WINDOW_COUNT];
    int resident_probe_rc[OWNED_RECOVERY_WINDOW_COUNT];
    int restore_probe_rc[OWNED_RECOVERY_WINDOW_COUNT];
    int window_rc[OWNED_RECOVERY_WINDOW_COUNT];
    for (int i = 0; i < OWNED_RECOVERY_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_history[i] = 0;
        digest_after_restore[i] = 0;
        history_handler_delta[i] = 0;
        history_minor_delta[i] = 0;
        sentinel_handler_delta[i] = 0;
        sentinel_minor_delta[i] = 0;
        resident_probe_rc[i] = -1;
        restore_probe_rc[i] = -1;
        if (recovery_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else if (initial_resident_rc != 0) {
            window_rc[i] = initial_resident_rc;
        } else if (handler_install_rc != 0) {
            window_rc[i] = handler_install_rc;
        } else {
            window_rc[i] = measure_owned_recovery_window(
                owned_recovery_group,
                &sequences[i],
                pages,
                page_size,
                OWNED_RECOVERY_PAGE_COUNT,
                &results[i],
                &durations[i],
                &digest_after_history[i],
                &digest_after_restore[i],
                &history_handler_delta[i],
                &history_minor_delta[i],
                &sentinel_handler_delta[i],
                &sentinel_minor_delta[i],
                &resident_probe_rc[i],
                &restore_probe_rc[i]);
        }
    }

    int handler_restore_rc = restore_owned_recovery_handler();
    int final_permission_rc = restore_owned_recovery_permissions(pages, page_size, OWNED_RECOVERY_PAGE_COUNT);
    uint64_t final_digest = fnv1a64(pages, byte_count);

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool bytes_unchanged_after_history = true;
    bool bytes_unchanged_after_restore = true;
    bool permissions_restored = final_permission_rc == 0 && handler_restore_rc == 0;
    bool pages_resident = initial_resident_rc == 0;
    bool handler_counts_expected = true;
    bool history_minor_counts_equal = true;
    bool sentinel_handler_zero = true;
    bool sentinel_minor_zero = true;
    int expected_history_handlers = (int)(OWNED_RECOVERY_PAGE_COUNT * OWNED_RECOVERY_HISTORY_LOOPS);
    long reference_history_minor = history_minor_delta[0];
    for (int i = 0; i < OWNED_RECOVERY_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        bytes_unchanged_after_history =
            bytes_unchanged_after_history && digest_after_history[i] == initial_digest;
        bytes_unchanged_after_restore =
            bytes_unchanged_after_restore && digest_after_restore[i] == initial_digest;
        permissions_restored = permissions_restored && restore_probe_rc[i] == 0;
        pages_resident = pages_resident && resident_probe_rc[i] == 0;
        handler_counts_expected =
            handler_counts_expected && history_handler_delta[i] == expected_history_handlers;
        history_minor_counts_equal =
            history_minor_counts_equal && history_minor_delta[i] == reference_history_minor;
        sentinel_handler_zero = sentinel_handler_zero && sentinel_handler_delta[i] == 0;
        sentinel_minor_zero = sentinel_minor_zero && sentinel_minor_delta[i] == 0;
    }
    bool no_escaped_recovery = owned_recovery_ctx.escaped_count == 0;
    bool final_digest_restored = final_digest == initial_digest;

    uint64_t identity_misses = event_value_by_name(&results[0], owned_recovery_group, "cache_misses");
    uint64_t forward_misses = event_value_by_name(&results[1], owned_recovery_group, "cache_misses");
    uint64_t reverse_misses = event_value_by_name(&results[2], owned_recovery_group, "cache_misses");
    uint64_t shuffle_misses = event_value_by_name(&results[3], owned_recovery_group, "cache_misses");
    uint64_t miss_delta = abs_diff_u64(forward_misses, reverse_misses);
    uint64_t miss_control = abs_diff_u64(identity_misses, shuffle_misses);
    uint64_t miss_threshold = max_u64(32u, 3u * miss_control);
    bool miss_signal = miss_delta > miss_threshold;

    uint64_t identity_cycles = event_value_by_name(&results[0], owned_recovery_group, "cpu_cycles_not_halted");
    uint64_t forward_cycles = event_value_by_name(&results[1], owned_recovery_group, "cpu_cycles_not_halted");
    uint64_t reverse_cycles = event_value_by_name(&results[2], owned_recovery_group, "cpu_cycles_not_halted");
    uint64_t shuffle_cycles = event_value_by_name(&results[3], owned_recovery_group, "cpu_cycles_not_halted");
    uint64_t cycles_delta = abs_diff_u64(forward_cycles, reverse_cycles);
    uint64_t cycles_control = abs_diff_u64(identity_cycles, shuffle_cycles);
    uint64_t cycles_threshold = max_u64(10000u, 3u * cycles_control);
    bool cycles_signal = cycles_delta > cycles_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(10000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;

    bool integrity_closed = recovery_open_rc == 0 && handler_install_rc == 0 &&
        all_windows_ok && all_unmultiplexed && bytes_unchanged_after_history &&
        bytes_unchanged_after_restore && final_digest_restored && permissions_restored &&
        pages_resident && handler_counts_expected && history_minor_counts_equal &&
        sentinel_handler_zero && sentinel_minor_zero && no_escaped_recovery;
    bool owned_recovery_history_response = integrity_closed &&
        (miss_signal || cycles_signal || duration_signal);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_OWNED_RECOVERY_HISTORY_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) {
        (void)munmap(pages, byte_count);
        return 1;
    }
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        (void)munmap(pages, byte_count);
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", OWNED_RECOVERY_HISTORY_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", owned_recovery_history_response ? "OWNED_RECOVERY_HISTORY_RESPONSE_FOUND" : "OWNED_RECOVERY_HISTORY_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Owned synchronous page-recovery timing/PMU discriminator only; no memory disclosure, isolation bypass, path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, target coupling, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"owned_recovery_core\": %d},\n", CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_pages_only\": true, \"synthetic_public_contents_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false, \"cross_domain_observation\": false},\n");
    fprintf(out, "  \"owned_recovery_carrier\": {\"page_count\": %u, \"page_bytes\": %zu, \"byte_count\": %zu, \"history_loops\": %u, \"sentinel_loops\": %u, \"history_recovery_count_per_window\": %d, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\", \"final_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        OWNED_RECOVERY_PAGE_COUNT, page_size, byte_count,
        OWNED_RECOVERY_HISTORY_LOOPS, OWNED_RECOVERY_SENTINEL_LOOPS,
        expected_history_handlers, initial_digest, final_digest);
    fprintf(out, "  \"selected_group\": \"owned_recovery_group\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", recovery_open_rc);
    fprintf(out, "  \"handler_install_rc\": %d,\n", handler_install_rc);
    fprintf(out, "  \"handler_restore_rc\": %d,\n", handler_restore_rc);
    fprintf(out, "  \"final_permission_restore_rc\": %d,\n", final_permission_rc);
    fprintf(out, "  \"initial_resident_rc\": %d,\n", initial_resident_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, owned_recovery_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"history\": \"each CAT_CAS-owned page is made temporarily unreadable and synchronously recovered exactly once per history window\", \"balanced_histories\": \"neutral G0-G2-G1-G3, forward G0-G1-G2-G3, reverse G3-G2-G1-G0, shuffle G1-G3-G0-G2\", \"sentinel\": \"fixed no-recovery read sequence G0-G1-G2-G3\", \"restoration\": \"permissions restored to read-write, bytes unchanged, pages resident, handler counts equal, sentinel recovery count zero\", \"primary_acceptance\": \"forward/reverse cache_misses, cycles, or duration delta exceeds max(floor, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < OWNED_RECOVERY_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"history_loops\": %u, \"history_handler_delta\": %d, \"history_minor_delta\": %ld, \"sentinel_handler_delta\": %d, \"sentinel_minor_delta\": %ld, \"resident_probe_rc\": %d, \"permission_restore_probe_rc\": %d, \"digest_after_history_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].history_loops,
            history_handler_delta[i],
            history_minor_delta[i],
            sentinel_handler_delta[i],
            sentinel_minor_delta[i],
            resident_probe_rc[i],
            restore_probe_rc[i],
            digest_after_history[i],
            digest_after_restore[i],
            json_bool(digest_after_history[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, owned_recovery_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == OWNED_RECOVERY_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"cache_misses\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_misses, forward_misses, reverse_misses, shuffle_misses, miss_delta, miss_control, miss_threshold, json_bool(miss_signal));
    fprintf(out, "    \"cpu_cycles_not_halted\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_cycles, forward_cycles, reverse_cycles, shuffle_cycles, cycles_delta, cycles_control, cycles_threshold, json_bool(cycles_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"bytes_unchanged_after_history\": %s, \"bytes_unchanged_after_restore\": %s, \"final_digest_restored\": %s, \"permissions_restored\": %s, \"pages_resident\": %s, \"handler_counts_expected\": %s, \"history_minor_counts_equal\": %s, \"sentinel_handler_zero\": %s, \"sentinel_minor_zero\": %s, \"no_escaped_recovery\": %s, \"cache_miss_signal\": %s, \"cycle_signal\": %s, \"duration_signal\": %s, \"integrity_closed\": %s, \"owned_recovery_history_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(bytes_unchanged_after_history),
        json_bool(bytes_unchanged_after_restore),
        json_bool(final_digest_restored),
        json_bool(permissions_restored),
        json_bool(pages_resident),
        json_bool(handler_counts_expected),
        json_bool(history_minor_counts_equal),
        json_bool(sentinel_handler_zero),
        json_bool(sentinel_minor_zero),
        json_bool(no_escaped_recovery),
        json_bool(miss_signal),
        json_bool(cycles_signal),
        json_bool(duration_signal),
        json_bool(integrity_closed),
        json_bool(owned_recovery_history_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"owned_recovery_group\"}\n",
        owned_recovery_history_response ? "OWNED_RECOVERY_HISTORY_RESPONSE_FOUND" : "OWNED_RECOVERY_HISTORY_RESPONSE_NOT_ESTABLISHED",
        result_path);
    (void)munmap(pages, byte_count);
    return 0;
}

static int wait_child_ok(pid_t pid, int *child_status) {
    int status = 0;
    pid_t waited = 0;
    do {
        waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0) {
        if (child_status) *child_status = -errno;
        return -errno;
    }
    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        if (child_status) *child_status = code;
        return code == 0 ? 0 : -EIO;
    }
    if (WIFSIGNALED(status)) {
        int code = 128 + WTERMSIG(status);
        if (child_status) *child_status = code;
        return -EIO;
    }
    if (child_status) *child_status = -EIO;
    return -EIO;
}

static int run_process_lifecycle_source_pattern(
    struct carrier *carrier,
    enum process_lifecycle_kind pattern,
    unsigned int loops
) {
    if (pin_to_core(CATCAS_CORE_A) != 0) return -errno;
    for (unsigned int loop = 0; loop < loops; loop++) {
        switch (pattern) {
            case PROCESS_LIFECYCLE_NEUTRAL:
                store_same_value_subset_on_core(carrier, CATCAS_CORE_A, 0);
                read_subset_on_core(carrier, CATCAS_CORE_A, 0);
                store_same_value_subset_on_core(carrier, CATCAS_CORE_A, 1);
                read_subset_on_core(carrier, CATCAS_CORE_A, 1);
                break;
            case PROCESS_LIFECYCLE_FORWARD:
                store_same_value_subset_on_core(carrier, CATCAS_CORE_A, 0);
                read_subset_on_core(carrier, CATCAS_CORE_A, 1);
                store_same_value_subset_on_core(carrier, CATCAS_CORE_A, 1);
                read_subset_on_core(carrier, CATCAS_CORE_A, 0);
                break;
            case PROCESS_LIFECYCLE_REVERSE:
                store_same_value_subset_on_core(carrier, CATCAS_CORE_A, 1);
                read_subset_on_core(carrier, CATCAS_CORE_A, 0);
                store_same_value_subset_on_core(carrier, CATCAS_CORE_A, 0);
                read_subset_on_core(carrier, CATCAS_CORE_A, 1);
                break;
            case PROCESS_LIFECYCLE_SHUFFLE:
                read_subset_on_core(carrier, CATCAS_CORE_A, 0);
                store_same_value_subset_on_core(carrier, CATCAS_CORE_A, 1);
                read_subset_on_core(carrier, CATCAS_CORE_A, 1);
                store_same_value_subset_on_core(carrier, CATCAS_CORE_A, 0);
                break;
            default:
                return -EINVAL;
        }
    }
    return 0;
}

static int fork_lifecycle_source_process(
    struct carrier *carrier,
    const struct process_lifecycle_sequence_spec *sequence,
    int *child_status
) {
    pid_t pid = fork();
    if (pid < 0) return -errno;
    if (pid == 0) {
        int rc = run_process_lifecycle_source_pattern(
            carrier, sequence->source_pattern, sequence->source_loops);
        _exit(rc == 0 ? 0 : 101);
    }
    return wait_child_ok(pid, child_status);
}

static int fork_lifecycle_sentinel_process(struct carrier *carrier, int *child_status) {
    pid_t pid = fork();
    if (pid < 0) return -errno;
    if (pid == 0) {
        if (pin_to_core(CATCAS_CORE_B) != 0) _exit(101);
        remote_store_same_value(carrier);
        _exit(0);
    }
    return wait_child_ok(pid, child_status);
}

static int measure_process_lifecycle_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct process_lifecycle_sequence_spec *sequence,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_after_source,
    uint64_t *digest_after_restore,
    int *source_child_status,
    int *sentinel_child_status
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    init_carrier(carrier);
    prefault_carrier(carrier);
    restore_on_core(carrier, CATCAS_CORE_A);
    int source_rc = fork_lifecycle_source_process(carrier, sequence, source_child_status);
    *digest_after_source = fnv1a64(carrier->bytes, carrier->byte_count);
    if (source_rc != 0) return source_rc;
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int sentinel_rc = fork_lifecycle_sentinel_process(carrier, sentinel_child_status);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (sentinel_rc != 0) return sentinel_rc;
    *duration_ns = end >= start ? end - start : 0;
    restore_on_core(carrier, CATCAS_CORE_A);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static int run_process_lifecycle_mode(const char *output_root) {
    size_t byte_count = (size_t)CARRIER_LINES * CACHE_LINE_BYTES;
    unsigned char *bytes = mmap(NULL, byte_count, PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (bytes == MAP_FAILED) {
        fprintf(stderr, "process lifecycle shared carrier allocation failed: %s\n", strerror(errno));
        return 1;
    }
    struct carrier shared_carrier = {
        .bytes = bytes,
        .byte_count = byte_count,
        .line_count = CARRIER_LINES,
    };
    init_carrier(&shared_carrier);
    uint64_t initial_digest = fnv1a64(shared_carrier.bytes, shared_carrier.byte_count);

    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_B, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }

    const struct process_lifecycle_sequence_spec sequences[] = {
        {"neutral_source_process", PROCESS_LIFECYCLE_NEUTRAL, PROCESS_LIFECYCLE_SOURCE_LOOPS},
        {"forward_source_process", PROCESS_LIFECYCLE_FORWARD, PROCESS_LIFECYCLE_SOURCE_LOOPS},
        {"reverse_source_process", PROCESS_LIFECYCLE_REVERSE, PROCESS_LIFECYCLE_SOURCE_LOOPS},
        {"shuffle_source_process", PROCESS_LIFECYCLE_SHUFFLE, PROCESS_LIFECYCLE_SOURCE_LOOPS},
    };
    enum { PROCESS_LIFECYCLE_WINDOW_COUNT = 4 };
    struct group_result results[PROCESS_LIFECYCLE_WINDOW_COUNT];
    uint64_t durations[PROCESS_LIFECYCLE_WINDOW_COUNT];
    uint64_t digest_after_source[PROCESS_LIFECYCLE_WINDOW_COUNT];
    uint64_t digest_after_restore[PROCESS_LIFECYCLE_WINDOW_COUNT];
    int source_child_status[PROCESS_LIFECYCLE_WINDOW_COUNT];
    int sentinel_child_status[PROCESS_LIFECYCLE_WINDOW_COUNT];
    int window_rc[PROCESS_LIFECYCLE_WINDOW_COUNT];
    for (int i = 0; i < PROCESS_LIFECYCLE_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_after_source[i] = 0;
        digest_after_restore[i] = 0;
        source_child_status[i] = -1;
        sentinel_child_status[i] = -1;
        if (primary_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_process_lifecycle_window(
                primary_group,
                &sequences[i],
                &shared_carrier,
                &results[i],
                &durations[i],
                &digest_after_source[i],
                &digest_after_restore[i],
                &source_child_status[i],
                &sentinel_child_status[i]);
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool bytes_unchanged_after_source = true;
    bool bytes_unchanged_after_restore = true;
    bool children_exited_cleanly = true;
    for (int i = 0; i < PROCESS_LIFECYCLE_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        bytes_unchanged_after_source =
            bytes_unchanged_after_source && digest_after_source[i] == initial_digest;
        bytes_unchanged_after_restore =
            bytes_unchanged_after_restore && digest_after_restore[i] == initial_digest;
        children_exited_cleanly = children_exited_cleanly &&
            source_child_status[i] == 0 && sentinel_child_status[i] == 0;
    }

    uint64_t identity_c2d = event_value_by_name(&results[0], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t forward_c2d = event_value_by_name(&results[1], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t reverse_c2d = event_value_by_name(&results[2], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t shuffle_c2d = event_value_by_name(&results[3], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t c2d_delta = abs_diff_u64(forward_c2d, reverse_c2d);
    uint64_t c2d_control = abs_diff_u64(identity_c2d, shuffle_c2d);
    uint64_t c2d_threshold = max_u64(32u, 3u * c2d_control);
    bool c2d_signal = c2d_delta > c2d_threshold;

    uint64_t identity_probe = event_value_by_name(&results[0], primary_group, "probe_responses_dirty");
    uint64_t forward_probe = event_value_by_name(&results[1], primary_group, "probe_responses_dirty");
    uint64_t reverse_probe = event_value_by_name(&results[2], primary_group, "probe_responses_dirty");
    uint64_t shuffle_probe = event_value_by_name(&results[3], primary_group, "probe_responses_dirty");
    uint64_t probe_delta = abs_diff_u64(forward_probe, reverse_probe);
    uint64_t probe_control = abs_diff_u64(identity_probe, shuffle_probe);
    uint64_t probe_threshold = max_u64(32u, 3u * probe_control);
    bool probe_signal = probe_delta > probe_threshold;

    uint64_t identity_duration = durations[0];
    uint64_t forward_duration = durations[1];
    uint64_t reverse_duration = durations[2];
    uint64_t shuffle_duration = durations[3];
    uint64_t duration_delta = abs_diff_u64(forward_duration, reverse_duration);
    uint64_t duration_control = abs_diff_u64(identity_duration, shuffle_duration);
    uint64_t duration_threshold = max_u64(10000u, 3u * duration_control);
    bool duration_signal = duration_delta > duration_threshold;

    bool process_lifecycle_response = primary_open_rc == 0 && all_windows_ok &&
        all_unmultiplexed && children_exited_cleanly &&
        bytes_unchanged_after_source && bytes_unchanged_after_restore &&
        (c2d_signal || probe_signal || duration_signal);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_PROCESS_LIFECYCLE_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) {
        munmap(bytes, byte_count);
        return 1;
    }
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        munmap(bytes, byte_count);
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", PROCESS_LIFECYCLE_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", process_lifecycle_response ? "PROCESS_LIFECYCLE_RESPONSE_FOUND" : "PROCESS_LIFECYCLE_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Process-lifecycle CAT_CAS-owned carrier discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"source_core\": %d, \"measurement_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"shared_anonymous_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false, \"unrelated_process_observation\": false},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"source_loops\": %u, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n",
        shared_carrier.line_count, CACHE_LINE_BYTES, shared_carrier.byte_count,
        PROCESS_LIFECYCLE_SOURCE_LOOPS, initial_digest);
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"source\": \"separate source process applies balanced public read/store history over shared CAT_CAS-owned lines and exits\", \"sentinel\": \"fresh measurement process on the observer core performs byte-preserving remote_store_same_value\", \"primary_acceptance\": \"forward/reverse fresh-process sentinel delta exceeds max(32, 3 * neutral/shuffle control spread) for an established coherence counter\", \"secondary_acceptance\": \"duration delta exceeds max(10000 ns, 3 * neutral/shuffle control spread)\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < PROCESS_LIFECYCLE_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"source_loops\": %u, \"source_child_status\": %d, \"sentinel_child_status\": %d, \"digest_after_source_hex\": \"0x%016" PRIx64 "\", \"digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"bytes_unchanged_after_source\": %s, \"bytes_unchanged_after_restore\": %s, \"group\": ",
            sequences[i].name,
            window_rc[i],
            durations[i],
            sequences[i].source_loops,
            source_child_status[i],
            sentinel_child_status[i],
            digest_after_source[i],
            digest_after_restore[i],
            json_bool(digest_after_source[i] == initial_digest),
            json_bool(digest_after_restore[i] == initial_digest));
        print_group_result(out, primary_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == PROCESS_LIFECYCLE_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"cache_block_commands_change_to_dirty\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_c2d, forward_c2d, reverse_c2d, shuffle_c2d, c2d_delta, c2d_control, c2d_threshold, json_bool(c2d_signal));
    fprintf(out, "    \"probe_responses_dirty\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s},\n",
        identity_probe, forward_probe, reverse_probe, shuffle_probe, probe_delta, probe_control, probe_threshold, json_bool(probe_signal));
    fprintf(out, "    \"duration_ns\": {\"identity\": %" PRIu64 ", \"forward\": %" PRIu64 ", \"reverse\": %" PRIu64 ", \"shuffle\": %" PRIu64 ", \"forward_reverse_delta\": %" PRIu64 ", \"control_floor\": %" PRIu64 ", \"threshold\": %" PRIu64 ", \"signal\": %s}\n",
        identity_duration, forward_duration, reverse_duration, shuffle_duration, duration_delta, duration_control, duration_threshold, json_bool(duration_signal));
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"children_exited_cleanly\": %s, \"bytes_unchanged_after_source\": %s, \"bytes_unchanged_after_restore\": %s, \"change_to_dirty_lifecycle_signal\": %s, \"probe_dirty_lifecycle_signal\": %s, \"duration_lifecycle_signal\": %s, \"process_lifecycle_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(children_exited_cleanly),
        json_bool(bytes_unchanged_after_source),
        json_bool(bytes_unchanged_after_restore),
        json_bool(c2d_signal),
        json_bool(probe_signal),
        json_bool(duration_signal),
        json_bool(process_lifecycle_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        process_lifecycle_response ? "PROCESS_LIFECYCLE_RESPONSE_FOUND" : "PROCESS_LIFECYCLE_RESPONSE_NOT_ESTABLISHED",
        result_path);
    munmap(bytes, byte_count);
    return 0;
}

static int measure_wc_flush_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    enum wc_flush_op op,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    restore_on_core(carrier, CATCAS_CORE_A);
    prefault_carrier(carrier);
    restore_on_core(carrier, CATCAS_CORE_A);
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int work_rc = run_wc_flush_operator(carrier, op);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (work_rc != 0) return -EIO;
    *duration_ns = end >= start ? end - start : 0;
    home_core_restore(carrier);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static int run_wc_flush_order_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_B, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }

    const enum wc_flush_op ops[] = {
        WC_OP_IDENTITY,
        WC_OP_FLUSH_ONLY,
        WC_OP_NORMAL_STORE_SAME,
        WC_OP_NT_STORE_SAME,
        WC_OP_FLUSH_THEN_NT_STORE,
        WC_OP_NT_STORE_THEN_FLUSH
    };
    enum { WC_WINDOW_COUNT = 6 };
    struct group_result results[WC_WINDOW_COUNT];
    uint64_t durations[WC_WINDOW_COUNT];
    uint64_t digest_before[WC_WINDOW_COUNT];
    uint64_t digest_after[WC_WINDOW_COUNT];
    int window_rc[WC_WINDOW_COUNT];
    for (int i = 0; i < WC_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_before[i] = 0;
        digest_after[i] = 0;
        if (primary_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_wc_flush_window(
                primary_group,
                ops[i],
                carrier,
                &results[i],
                &durations[i],
                &digest_before[i],
                &digest_after[i]
            );
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool all_restored = true;
    for (int i = 0; i < WC_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        all_restored = all_restored && digest_after[i] == initial_digest;
    }

    uint64_t c2d[WC_WINDOW_COUNT];
    uint64_t probe[WC_WINDOW_COUNT];
    for (int i = 0; i < WC_WINDOW_COUNT; i++) {
        c2d[i] = event_value_by_name(&results[i], primary_group, "cache_block_commands_change_to_dirty");
        probe[i] = event_value_by_name(&results[i], primary_group, "probe_responses_dirty");
    }
    uint64_t c2d_control_min = min_u64(min_u64(c2d[0], c2d[1]), min_u64(c2d[2], c2d[3]));
    uint64_t c2d_control_max = max_u64(max_u64(c2d[0], c2d[1]), max_u64(c2d[2], c2d[3]));
    uint64_t probe_control_min = min_u64(min_u64(probe[0], probe[1]), min_u64(probe[2], probe[3]));
    uint64_t probe_control_max = max_u64(max_u64(probe[0], probe[1]), max_u64(probe[2], probe[3]));
    uint64_t c2d_order_delta = abs_diff_u64(c2d[4], c2d[5]);
    uint64_t probe_order_delta = abs_diff_u64(probe[4], probe[5]);
    bool c2d_order_response = c2d_order_delta > 32u && c2d_order_delta > (c2d_control_max - c2d_control_min) * 3u;
    bool probe_order_response = probe_order_delta > 32u && probe_order_delta > (probe_control_max - probe_control_min) * 3u;
    bool wc_flush_order_response = primary_open_rc == 0 && all_windows_ok && all_unmultiplexed &&
        all_restored && (c2d_order_response || probe_order_response);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_WC_FLUSH_ORDER_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", WC_FLUSH_ORDER_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", wc_flush_order_response ? "WC_FLUSH_ORDER_RESPONSE_FOUND" : "WC_FLUSH_ORDER_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Write-combining/flush-order PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"observed_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"primary\": \"difference between flush-then-non-temporal-store and non-temporal-store-then-flush\", \"control_floor\": \"identity, flush-only, normal same-value store, and non-temporal same-value store range\", \"acceptance\": \"order delta > 32 and > 3 * control range for either established coherence counter\"},\n");
    fprintf(out, "  \"operators\": [\n");
    for (int i = 0; i < WC_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"carrier_restored\": %s, \"group\": ",
            wc_flush_op_name(ops[i]),
            window_rc[i],
            durations[i],
            digest_before[i],
            digest_after[i],
            json_bool(digest_after[i] == initial_digest));
        print_group_result(out, primary_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == WC_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"change_to_dirty\": {\"identity\": %" PRIu64 ", \"flush_only\": %" PRIu64 ", \"normal_store_same_value\": %" PRIu64 ", \"nt_store_same_value\": %" PRIu64 ", \"flush_then_nt_store\": %" PRIu64 ", \"nt_store_then_flush\": %" PRIu64 ", \"order_delta\": %" PRIu64 ", \"control_range\": %" PRIu64 "},\n",
        c2d[0], c2d[1], c2d[2], c2d[3], c2d[4], c2d[5], c2d_order_delta, c2d_control_max - c2d_control_min);
    fprintf(out, "    \"probe_dirty\": {\"identity\": %" PRIu64 ", \"flush_only\": %" PRIu64 ", \"normal_store_same_value\": %" PRIu64 ", \"nt_store_same_value\": %" PRIu64 ", \"flush_then_nt_store\": %" PRIu64 ", \"nt_store_then_flush\": %" PRIu64 ", \"order_delta\": %" PRIu64 ", \"control_range\": %" PRIu64 "}\n",
        probe[0], probe[1], probe[2], probe[3], probe[4], probe[5], probe_order_delta, probe_control_max - probe_control_min);
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"carrier_restored\": %s, \"change_to_dirty_order_response\": %s, \"probe_dirty_order_response\": %s, \"wc_flush_order_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(all_restored),
        json_bool(c2d_order_response),
        json_bool(probe_order_response),
        json_bool(wc_flush_order_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        wc_flush_order_response ? "WC_FLUSH_ORDER_RESPONSE_FOUND" : "WC_FLUSH_ORDER_RESPONSE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

static int measure_eviction_sentinel_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    enum eviction_prep_op prep,
    struct carrier *carrier,
    struct carrier *eviction,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after_restore,
    uint64_t *eviction_digest_after
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    restore_on_core(eviction, CATCAS_CORE_A);
    int prep_rc = run_eviction_prep_operator(eviction, prep);
    *eviction_digest_after = fnv1a64(eviction->bytes, eviction->byte_count);
    if (prep_rc != 0) return prep_rc;
    restore_on_core(carrier, CATCAS_CORE_A);
    prefault_carrier(carrier);
    restore_on_core(carrier, CATCAS_CORE_A);
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int work_rc = run_coherence_operator(carrier, OP_REMOTE_STORE_SAME_VALUE);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (work_rc != 0) return -EIO;
    *duration_ns = end >= start ? end - start : 0;
    home_core_restore(carrier);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static int run_eviction_sentinel_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    struct carrier eviction;
    eviction.line_count = EVICTION_LINES;
    eviction.byte_count = EVICTION_LINES * CACHE_LINE_BYTES;
    if (posix_memalign((void **)&eviction.bytes, CACHE_LINE_BYTES, eviction.byte_count) != 0) {
        fprintf(stderr, "eviction allocation failed\n");
        return 1;
    }
    init_carrier(&eviction);
    uint64_t eviction_initial_digest = fnv1a64(eviction.bytes, eviction.byte_count);

    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_B, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }

    const enum eviction_prep_op preps[] = {
        EVICT_PREP_NONE,
        EVICT_PREP_HOME_READ,
        EVICT_PREP_REMOTE_READ,
        EVICT_PREP_HOME_WRITE,
        EVICT_PREP_REMOTE_WRITE,
        EVICT_PREP_HOME_THEN_REMOTE_READ,
        EVICT_PREP_REMOTE_THEN_HOME_READ
    };
    enum { EVICTION_WINDOW_COUNT = 7 };
    struct group_result results[EVICTION_WINDOW_COUNT];
    uint64_t durations[EVICTION_WINDOW_COUNT];
    uint64_t digest_before[EVICTION_WINDOW_COUNT];
    uint64_t digest_after[EVICTION_WINDOW_COUNT];
    uint64_t eviction_digest_after[EVICTION_WINDOW_COUNT];
    int window_rc[EVICTION_WINDOW_COUNT];
    for (int i = 0; i < EVICTION_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_before[i] = 0;
        digest_after[i] = 0;
        eviction_digest_after[i] = 0;
        if (primary_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_eviction_sentinel_window(
                primary_group,
                preps[i],
                carrier,
                &eviction,
                &results[i],
                &durations[i],
                &digest_before[i],
                &digest_after[i],
                &eviction_digest_after[i]
            );
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool carrier_restored = true;
    bool eviction_restored = true;
    for (int i = 0; i < EVICTION_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        carrier_restored = carrier_restored && digest_after[i] == initial_digest;
        eviction_restored = eviction_restored && eviction_digest_after[i] == eviction_initial_digest;
    }

    uint64_t c2d[EVICTION_WINDOW_COUNT];
    uint64_t probe[EVICTION_WINDOW_COUNT];
    for (int i = 0; i < EVICTION_WINDOW_COUNT; i++) {
        c2d[i] = event_value_by_name(&results[i], primary_group, "cache_block_commands_change_to_dirty");
        probe[i] = event_value_by_name(&results[i], primary_group, "probe_responses_dirty");
    }
    uint64_t c2d_threshold = max_u64(32u, c2d[0] / 10u);
    uint64_t probe_threshold = max_u64(32u, probe[0] / 10u);
    bool c2d_sentinel_moved = false;
    bool probe_sentinel_moved = false;
    for (int i = 1; i < EVICTION_WINDOW_COUNT; i++) {
        c2d_sentinel_moved = c2d_sentinel_moved || abs_diff_u64(c2d[i], c2d[0]) > c2d_threshold;
        probe_sentinel_moved = probe_sentinel_moved || abs_diff_u64(probe[i], probe[0]) > probe_threshold;
    }
    bool eviction_sentinel_response = primary_open_rc == 0 && all_windows_ok && all_unmultiplexed &&
        carrier_restored && eviction_restored && (c2d_sentinel_moved || probe_sentinel_moved);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_EVICTION_SENTINEL_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) {
        free(eviction.bytes);
        return 1;
    }
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        free(eviction.bytes);
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", EVICTION_SENTINEL_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", eviction_sentinel_response ? "EVICTION_SENTINEL_RESPONSE_FOUND" : "EVICTION_SENTINEL_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Eviction/topology sentinel PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"observed_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"eviction_buffer\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"iterations\": %d, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", eviction.line_count, CACHE_LINE_BYTES, eviction.byte_count, EVICTION_ITERATIONS, eviction_initial_digest);
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"sentinel\": \"remote_store_same_value over carrier after eviction-buffer preconditioning\", \"primary\": \"sentinel response delta versus no-prep baseline\", \"acceptance\": \"any precondition changes Change-to-Dirty or dirty-probe sentinel by more than max(32, baseline/10) while both buffers restore\"},\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < EVICTION_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"prep\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"carrier_restored\": %s, \"eviction_digest_after_hex\": \"0x%016" PRIx64 "\", \"eviction_restored\": %s, \"change_to_dirty_delta_from_baseline\": %" PRIu64 ", \"probe_dirty_delta_from_baseline\": %" PRIu64 ", \"group\": ",
            eviction_prep_op_name(preps[i]),
            window_rc[i],
            durations[i],
            digest_before[i],
            digest_after[i],
            json_bool(digest_after[i] == initial_digest),
            eviction_digest_after[i],
            json_bool(eviction_digest_after[i] == eviction_initial_digest),
            abs_diff_u64(c2d[i], c2d[0]),
            abs_diff_u64(probe[i], probe[0]));
        print_group_result(out, primary_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == EVICTION_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"baseline_counts\": {\"change_to_dirty\": %" PRIu64 ", \"probe_dirty\": %" PRIu64 ", \"change_to_dirty_threshold\": %" PRIu64 ", \"probe_dirty_threshold\": %" PRIu64 "},\n",
        c2d[0], probe[0], c2d_threshold, probe_threshold);
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"carrier_restored\": %s, \"eviction_buffer_restored\": %s, \"change_to_dirty_sentinel_moved\": %s, \"probe_dirty_sentinel_moved\": %s, \"eviction_sentinel_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(carrier_restored),
        json_bool(eviction_restored),
        json_bool(c2d_sentinel_moved),
        json_bool(probe_sentinel_moved),
        json_bool(eviction_sentinel_response));
    fprintf(out, "}\n");
    fclose(out);
    free(eviction.bytes);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        eviction_sentinel_response ? "EVICTION_SENTINEL_RESPONSE_FOUND" : "EVICTION_SENTINEL_RESPONSE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

struct ibs_source_spec {
    const char *name;
    const char *type_path;
    unsigned int fixed_type;
    bool use_fixed_type;
    uint64_t config;
    unsigned int precise_ip;
    const char *config_note;
};

static int run_ibs_workload(const char *workload, struct carrier *carrier) {
    if (strcmp(workload, "idle_pause") == 0) {
        idle_pause();
        return 0;
    }
    if (strcmp(workload, "remote_read_shared") == 0) {
        remote_read_shared(carrier);
        return 0;
    }
    if (strcmp(workload, "remote_store_same_value") == 0) {
        remote_store_same_value(carrier);
        return 0;
    }
    return -EINVAL;
}

static int measure_ibs_window(
    unsigned int pmu_type,
    uint64_t config,
    unsigned int precise_ip,
    const char *workload,
    struct carrier *carrier,
    struct single_event_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after_restore
) {
    memset(result, 0, sizeof(*result));
    result->read_rc = -ENODATA;
    restore_on_core(carrier, CATCAS_CORE_A);
    prefault_carrier(carrier);
    restore_on_core(carrier, CATCAS_CORE_A);
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);

    struct perf_event_attr attr;
    fill_ibs_attr(&attr, pmu_type, config, precise_ip, true);
    int fd = (int)perf_event_open_call(&attr, -1, CATCAS_CORE_B, -1, PERF_FLAG_FD_CLOEXEC);
    if (fd < 0) {
        result->open_errno = errno;
        return -errno;
    }
    result->opened = true;
    if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) != 0) {
        int saved = errno;
        close(fd);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) != 0) {
        int saved = errno;
        close(fd);
        return -saved;
    }
    int work_rc = run_ibs_workload(workload, carrier);
    if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) != 0) {
        int saved = errno;
        close(fd);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    result->read_rc = read_single_event(fd, result);
    close(fd);
    if (result->read_rc != 0) return result->read_rc;
    if (work_rc != 0) return work_rc;
    *duration_ns = end >= start ? end - start : 0;
    home_core_restore(carrier);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static int run_ibs_first_light_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    enum { IBS_SOURCE_COUNT = 6, IBS_WORKLOAD_COUNT = 3 };
    const struct ibs_source_spec sources[IBS_SOURCE_COUNT] = {
        {"ibs_fetch_default", "/sys/bus/event_source/devices/ibs_fetch/type", 0u, false, 0ull, 0u, "direct ibs_fetch config=0"},
        {"ibs_fetch_rand_en", "/sys/bus/event_source/devices/ibs_fetch/type", 0u, false, 1ull << 57, 0u, "direct ibs_fetch rand_en bit 57 set"},
        {"ibs_op_default", "/sys/bus/event_source/devices/ibs_op/type", 0u, false, 0ull, 0u, "direct ibs_op config=0"},
        {"ibs_op_cnt_ctl", "/sys/bus/event_source/devices/ibs_op/type", 0u, false, 1ull << 19, 0u, "direct ibs_op cnt_ctl bit 19 set"},
        {"raw_cycles_precise", "", PERF_TYPE_RAW, true, raw_config(0x076u, 0x00u), 1u, "r076:p precise raw cycles route"},
        {"raw_uops_precise", "", PERF_TYPE_RAW, true, raw_config(0x0c1u, 0x00u), 1u, "r0C1:p precise raw uops route"}
    };
    const char *workloads[IBS_WORKLOAD_COUNT] = {
        "idle_pause",
        "remote_read_shared",
        "remote_store_same_value"
    };
    unsigned int pmu_types[IBS_SOURCE_COUNT] = {0u, 0u, 0u, 0u};
    int type_rc[IBS_SOURCE_COUNT] = {0, 0, 0, 0};
    for (int source = 0; source < IBS_SOURCE_COUNT; source++) {
        if (sources[source].use_fixed_type) {
            pmu_types[source] = sources[source].fixed_type;
            type_rc[source] = 0;
        } else {
            type_rc[source] = read_uint_sysfs(sources[source].type_path, &pmu_types[source]);
        }
    }

    struct single_event_result results[IBS_SOURCE_COUNT][IBS_WORKLOAD_COUNT];
    uint64_t durations[IBS_SOURCE_COUNT][IBS_WORKLOAD_COUNT];
    uint64_t digest_before[IBS_SOURCE_COUNT][IBS_WORKLOAD_COUNT];
    uint64_t digest_after[IBS_SOURCE_COUNT][IBS_WORKLOAD_COUNT];
    int window_rc[IBS_SOURCE_COUNT][IBS_WORKLOAD_COUNT];

    for (int source = 0; source < IBS_SOURCE_COUNT; source++) {
        for (int workload = 0; workload < IBS_WORKLOAD_COUNT; workload++) {
            memset(&results[source][workload], 0, sizeof(results[source][workload]));
            durations[source][workload] = 0;
            digest_before[source][workload] = 0;
            digest_after[source][workload] = 0;
            if (type_rc[source] != 0) {
                window_rc[source][workload] = type_rc[source];
            } else {
                window_rc[source][workload] = measure_ibs_window(
                    pmu_types[source],
                    sources[source].config,
                    sources[source].precise_ip,
                    workloads[workload],
                    carrier,
                    &results[source][workload],
                    &durations[source][workload],
                    &digest_before[source][workload],
                    &digest_after[source][workload]
                );
            }
        }
    }

    bool any_opened = false;
    bool any_read_ok = false;
    bool any_nonzero = false;
    bool all_restored = true;
    bool any_workload_response = false;
    for (int source = 0; source < IBS_SOURCE_COUNT; source++) {
        uint64_t idle_value = results[source][0].value;
        for (int workload = 0; workload < IBS_WORKLOAD_COUNT; workload++) {
            any_opened = any_opened || results[source][workload].opened;
            any_read_ok = any_read_ok || results[source][workload].read_rc == 0;
            any_nonzero = any_nonzero || results[source][workload].value != 0u;
            all_restored = all_restored && (!results[source][workload].opened ||
                digest_after[source][workload] == initial_digest);
            if (workload > 0 && results[source][workload].read_rc == 0 &&
                results[source][workload].value > idle_value) {
                any_workload_response = true;
            }
        }
    }
    bool ibs_available = any_opened && any_read_ok && all_restored;

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_IBS_FIRST_LIGHT_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", IBS_FIRST_LIGHT_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", ibs_available ? "IBS_FIRST_LIGHT_AVAILABLE" : "IBS_FIRST_LIGHT_NOT_AVAILABLE");
    fprintf(out, "  \"claim_ceiling\": \"IBS availability and first-light probe only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"observed_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"predeclared_observable\": {\"primary\": \"ordinary perf_event_open success for ibs_fetch/ibs_op\", \"secondary\": \"readable event value during CAT_CAS-owned idle/read/store windows\", \"purpose\": \"carrier availability before any phase-local claim\"},\n");
    fprintf(out, "  \"sources\": [\n");
    for (int source = 0; source < IBS_SOURCE_COUNT; source++) {
        fprintf(out, "    {\"name\": \"%s\", \"type_path\": \"%s\", \"type_read_rc\": %d, \"pmu_type\": %u, \"config_hex\": \"0x%016" PRIx64 "\", \"precise_ip\": %u, \"config_note\": \"%s\", \"windows\": [\n",
            sources[source].name,
            sources[source].type_path,
            type_rc[source],
            pmu_types[source],
            sources[source].config,
            sources[source].precise_ip,
            sources[source].config_note);
        for (int workload = 0; workload < IBS_WORKLOAD_COUNT; workload++) {
            fprintf(out, "      {\"workload\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"carrier_restored\": %s, \"event\": ",
                workloads[workload],
                window_rc[source][workload],
                durations[source][workload],
                digest_before[source][workload],
                digest_after[source][workload],
                json_bool(window_rc[source][workload] == 0 && digest_after[source][workload] == initial_digest));
            print_single_event_result(out, &results[source][workload]);
            fprintf(out, "}%s\n", workload + 1 == IBS_WORKLOAD_COUNT ? "" : ",");
        }
        fprintf(out, "    ]}%s\n", source + 1 == IBS_SOURCE_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"acceptance\": {\"any_opened\": %s, \"any_read_ok\": %s, \"any_nonzero\": %s, \"all_restored\": %s, \"any_workload_response\": %s, \"ibs_first_light_available\": %s}\n",
        json_bool(any_opened),
        json_bool(any_read_ok),
        json_bool(any_nonzero),
        json_bool(all_restored),
        json_bool(any_workload_response),
        json_bool(ibs_available));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"any_workload_response\":%s}\n",
        ibs_available ? "IBS_FIRST_LIGHT_AVAILABLE" : "IBS_FIRST_LIGHT_NOT_AVAILABLE",
        result_path,
        json_bool(any_workload_response));
    return 0;
}

struct phase_decode {
    long double plus_real;
    long double plus_imag;
    long double minus_real;
    long double minus_imag;
};

static int measure_phase_local_pmu_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct phase_local_window_spec *window,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    restore_on_core(carrier, CATCAS_CORE_A);
    prefault_carrier(carrier);
    restore_on_core(carrier, CATCAS_CORE_A);
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    store_same_value_rotated_span_on_core(carrier, CATCAS_CORE_B, window->line_count, window->start_bank);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    *duration_ns = end >= start ? end - start : 0;
    home_core_restore(carrier);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static void decode_phase_observable(
    const struct phase_local_window_spec windows[12],
    const struct group_result results[12],
    const char *event_name,
    struct phase_decode *decoded
) {
    long double p[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    long double c[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    long double m[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    for (int i = 0; i < 12; i++) {
        int phase = windows[i].phase_index;
        if (phase < 0 || phase >= 4) continue;
        long double value = normalized_event_value(&results[i], primary_group, event_name);
        if (strcmp(windows[i].role, "P") == 0) {
            p[phase] = value;
        } else if (strcmp(windows[i].role, "C") == 0) {
            c[phase] = value;
        } else if (strcmp(windows[i].role, "M") == 0) {
            m[phase] = value;
        }
    }
    long double plus[4];
    long double minus[4];
    for (int phase = 0; phase < 4; phase++) {
        plus[phase] = p[phase] - c[phase];
        minus[phase] = m[phase] - c[phase];
    }
    decoded->plus_real = 0.5L * (plus[0] - plus[2]);
    decoded->plus_imag = 0.5L * (plus[1] - plus[3]);
    decoded->minus_real = 0.5L * (minus[0] - minus[2]);
    decoded->minus_imag = 0.5L * (minus[1] - minus[3]);
}

static void decode_eviction_phase_observable(
    const struct eviction_phase_window_spec windows[12],
    const struct group_result results[12],
    const char *event_name,
    struct phase_decode *decoded
) {
    long double p[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    long double c[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    long double m[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    for (int i = 0; i < 12; i++) {
        int phase = windows[i].phase_index;
        if (phase < 0 || phase >= 4) continue;
        long double value = normalized_event_value(&results[i], primary_group, event_name);
        if (strcmp(windows[i].role, "P") == 0) {
            p[phase] = value;
        } else if (strcmp(windows[i].role, "C") == 0) {
            c[phase] = value;
        } else if (strcmp(windows[i].role, "M") == 0) {
            m[phase] = value;
        }
    }
    long double plus[4];
    long double minus[4];
    for (int phase = 0; phase < 4; phase++) {
        plus[phase] = p[phase] - c[phase];
        minus[phase] = m[phase] - c[phase];
    }
    decoded->plus_real = 0.5L * (plus[0] - plus[2]);
    decoded->plus_imag = 0.5L * (plus[1] - plus[3]);
    decoded->minus_real = 0.5L * (minus[0] - minus[2]);
    decoded->minus_imag = 0.5L * (minus[1] - minus[3]);
}

static void decode_eviction_bracketed_observable(
    const struct eviction_bracket_token_spec tokens[8],
    struct group_result results[8][3],
    const char *event_name,
    struct phase_decode *decoded
) {
    long double p[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    long double m[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    for (int i = 0; i < 8; i++) {
        int phase = tokens[i].phase_index;
        if (phase < 0 || phase >= 4) continue;
        long double before = normalized_event_value(&results[i][0], primary_group, event_name);
        long double center = normalized_event_value(&results[i][1], primary_group, event_name);
        long double after = normalized_event_value(&results[i][2], primary_group, event_name);
        long double bracketed = center - 0.5L * (before + after);
        if (strcmp(tokens[i].role, "P") == 0) {
            p[phase] = bracketed;
        } else if (strcmp(tokens[i].role, "M") == 0) {
            m[phase] = bracketed;
        }
    }
    decoded->plus_real = 0.5L * (p[0] - p[2]);
    decoded->plus_imag = 0.5L * (p[1] - p[3]);
    decoded->minus_real = 0.5L * (m[0] - m[2]);
    decoded->minus_imag = 0.5L * (m[1] - m[3]);
}

static void decode_eviction_bracketed_duration(
    const struct eviction_bracket_token_spec tokens[8],
    uint64_t durations[8][3],
    struct phase_decode *decoded
) {
    long double p[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    long double m[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    for (int i = 0; i < 8; i++) {
        int phase = tokens[i].phase_index;
        if (phase < 0 || phase >= 4) continue;
        long double before = (long double)durations[i][0];
        long double center = (long double)durations[i][1];
        long double after = (long double)durations[i][2];
        long double bracketed = center - 0.5L * (before + after);
        if (strcmp(tokens[i].role, "P") == 0) {
            p[phase] = bracketed;
        } else if (strcmp(tokens[i].role, "M") == 0) {
            m[phase] = bracketed;
        }
    }
    decoded->plus_real = 0.5L * (p[0] - p[2]);
    decoded->plus_imag = 0.5L * (p[1] - p[3]);
    decoded->minus_real = 0.5L * (m[0] - m[2]);
    decoded->minus_imag = 0.5L * (m[1] - m[3]);
}

static bool opposed_sign(long double a, long double b) {
    return (a > 0.0L && b < 0.0L) || (a < 0.0L && b > 0.0L);
}

static void print_phase_decode(
    FILE *out,
    const char *event_name,
    const struct phase_decode *sham,
    const struct phase_decode *candidate
) {
    long double sham_floor = abs_ld(sham->plus_imag);
    if (abs_ld(sham->minus_imag) > sham_floor) sham_floor = abs_ld(sham->minus_imag);
    long double min_candidate = abs_ld(candidate->plus_imag) < abs_ld(candidate->minus_imag) ?
        abs_ld(candidate->plus_imag) : abs_ld(candidate->minus_imag);
    bool opposed = opposed_sign(candidate->plus_imag, candidate->minus_imag);
    bool exceeds_sham = min_candidate > 3.0L * sham_floor;
    fprintf(out, "{\"event\":\"%s\",", event_name);
    fprintf(out, "\"sham\":{\"plus\":{\"real\":%.12Le,\"imag\":%.12Le},\"minus\":{\"real\":%.12Le,\"imag\":%.12Le}},",
        sham->plus_real, sham->plus_imag, sham->minus_real, sham->minus_imag);
    fprintf(out, "\"candidate\":{\"plus\":{\"real\":%.12Le,\"imag\":%.12Le},\"minus\":{\"real\":%.12Le,\"imag\":%.12Le}},",
        candidate->plus_real, candidate->plus_imag, candidate->minus_real, candidate->minus_imag);
    fprintf(out, "\"sham_floor\":%.12Le,\"opposed_sign\":%s,\"exceeds_three_x_sham_floor\":%s,\"phase_local_signal\":%s}",
        sham_floor, json_bool(opposed), json_bool(exceeds_sham), json_bool(opposed && exceeds_sham));
}

static bool phase_decode_signal(const struct phase_decode *sham, const struct phase_decode *candidate) {
    long double sham_floor = abs_ld(sham->plus_imag);
    if (abs_ld(sham->minus_imag) > sham_floor) sham_floor = abs_ld(sham->minus_imag);
    long double min_candidate = abs_ld(candidate->plus_imag) < abs_ld(candidate->minus_imag) ?
        abs_ld(candidate->plus_imag) : abs_ld(candidate->minus_imag);
    return opposed_sign(candidate->plus_imag, candidate->minus_imag) && min_candidate > 3.0L * sham_floor;
}

static int run_eviction_phase_local_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    struct carrier eviction;
    eviction.line_count = EVICTION_LINES;
    eviction.byte_count = EVICTION_LINES * CACHE_LINE_BYTES;
    if (posix_memalign((void **)&eviction.bytes, CACHE_LINE_BYTES, eviction.byte_count) != 0) {
        fprintf(stderr, "eviction allocation failed\n");
        return 1;
    }
    init_carrier(&eviction);
    uint64_t eviction_initial_digest = fnv1a64(eviction.bytes, eviction.byte_count);

    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_B, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }

    enum { EVICTION_PHASE_VARIANT_COUNT = 2, EVICTION_PHASE_WINDOW_COUNT = 12 };
    const enum eviction_prep_op control_prep = EVICT_PREP_REMOTE_READ;
    const enum eviction_prep_op high_prep = EVICT_PREP_HOME_READ;
    const enum eviction_prep_op low_prep = EVICT_PREP_HOME_WRITE;
    const char *variant_names[EVICTION_PHASE_VARIANT_COUNT] = {
        "equal_prep_sham",
        "high_low_candidate"
    };
    const struct eviction_phase_window_spec windows[EVICTION_PHASE_VARIANT_COUNT][EVICTION_PHASE_WINDOW_COUNT] = {
        {
            {"P0", "P", 0, EVICT_PREP_REMOTE_READ},
            {"C0", "C", 0, EVICT_PREP_REMOTE_READ},
            {"M0", "M", 0, EVICT_PREP_REMOTE_READ},
            {"M1", "M", 1, EVICT_PREP_REMOTE_READ},
            {"C1", "C", 1, EVICT_PREP_REMOTE_READ},
            {"P1", "P", 1, EVICT_PREP_REMOTE_READ},
            {"P2", "P", 2, EVICT_PREP_REMOTE_READ},
            {"C2", "C", 2, EVICT_PREP_REMOTE_READ},
            {"M2", "M", 2, EVICT_PREP_REMOTE_READ},
            {"M3", "M", 3, EVICT_PREP_REMOTE_READ},
            {"C3", "C", 3, EVICT_PREP_REMOTE_READ},
            {"P3", "P", 3, EVICT_PREP_REMOTE_READ}
        },
        {
            {"P0", "P", 0, EVICT_PREP_HOME_READ},
            {"C0", "C", 0, EVICT_PREP_REMOTE_READ},
            {"M0", "M", 0, EVICT_PREP_HOME_WRITE},
            {"M1", "M", 1, EVICT_PREP_HOME_WRITE},
            {"C1", "C", 1, EVICT_PREP_REMOTE_READ},
            {"P1", "P", 1, EVICT_PREP_HOME_READ},
            {"P2", "P", 2, EVICT_PREP_HOME_READ},
            {"C2", "C", 2, EVICT_PREP_REMOTE_READ},
            {"M2", "M", 2, EVICT_PREP_HOME_WRITE},
            {"M3", "M", 3, EVICT_PREP_HOME_READ},
            {"C3", "C", 3, EVICT_PREP_REMOTE_READ},
            {"P3", "P", 3, EVICT_PREP_HOME_WRITE}
        }
    };

    struct group_result results[EVICTION_PHASE_VARIANT_COUNT][EVICTION_PHASE_WINDOW_COUNT];
    uint64_t durations[EVICTION_PHASE_VARIANT_COUNT][EVICTION_PHASE_WINDOW_COUNT];
    uint64_t digest_before[EVICTION_PHASE_VARIANT_COUNT][EVICTION_PHASE_WINDOW_COUNT];
    uint64_t digest_after[EVICTION_PHASE_VARIANT_COUNT][EVICTION_PHASE_WINDOW_COUNT];
    uint64_t eviction_digest_after[EVICTION_PHASE_VARIANT_COUNT][EVICTION_PHASE_WINDOW_COUNT];
    int window_rc[EVICTION_PHASE_VARIANT_COUNT][EVICTION_PHASE_WINDOW_COUNT];

    for (int variant = 0; variant < EVICTION_PHASE_VARIANT_COUNT; variant++) {
        for (int i = 0; i < EVICTION_PHASE_WINDOW_COUNT; i++) {
            memset(&results[variant][i], 0, sizeof(results[variant][i]));
            durations[variant][i] = 0;
            digest_before[variant][i] = 0;
            digest_after[variant][i] = 0;
            eviction_digest_after[variant][i] = 0;
            if (primary_open_rc != 0) {
                window_rc[variant][i] = -ENODEV;
            } else {
                window_rc[variant][i] = measure_eviction_sentinel_window(
                    primary_group,
                    windows[variant][i].prep,
                    carrier,
                    &eviction,
                    &results[variant][i],
                    &durations[variant][i],
                    &digest_before[variant][i],
                    &digest_after[variant][i],
                    &eviction_digest_after[variant][i]
                );
            }
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool carrier_restored = true;
    bool eviction_restored = true;
    for (int variant = 0; variant < EVICTION_PHASE_VARIANT_COUNT; variant++) {
        for (int i = 0; i < EVICTION_PHASE_WINDOW_COUNT; i++) {
            all_windows_ok = all_windows_ok && window_rc[variant][i] == 0;
            all_unmultiplexed = all_unmultiplexed && results[variant][i].unmultiplexed;
            carrier_restored = carrier_restored && digest_after[variant][i] == initial_digest;
            eviction_restored = eviction_restored && eviction_digest_after[variant][i] == eviction_initial_digest;
        }
    }

    struct phase_decode sham_c2d;
    struct phase_decode candidate_c2d;
    struct phase_decode sham_probe;
    struct phase_decode candidate_probe;
    decode_eviction_phase_observable(windows[0], results[0], "cache_block_commands_change_to_dirty", &sham_c2d);
    decode_eviction_phase_observable(windows[1], results[1], "cache_block_commands_change_to_dirty", &candidate_c2d);
    decode_eviction_phase_observable(windows[0], results[0], "probe_responses_dirty", &sham_probe);
    decode_eviction_phase_observable(windows[1], results[1], "probe_responses_dirty", &candidate_probe);
    bool c2d_signal = phase_decode_signal(&sham_c2d, &candidate_c2d);
    bool probe_signal = phase_decode_signal(&sham_probe, &candidate_probe);
    bool eviction_phase_local_captured = primary_open_rc == 0 && all_windows_ok && all_unmultiplexed &&
        carrier_restored && eviction_restored && (c2d_signal || probe_signal);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_EVICTION_PHASE_LOCAL_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) {
        free(eviction.bytes);
        return 1;
    }
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        free(eviction.bytes);
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", EVICTION_PHASE_LOCAL_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", eviction_phase_local_captured ? "EVICTION_PHASE_LOCAL_CODED_RESPONSE_FOUND" : "EVICTION_PHASE_LOCAL_CODED_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Phase-local eviction-sentinel PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"observed_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false, \"private_branch_routing\": false},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"eviction_buffer\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"iterations\": %d, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", eviction.line_count, CACHE_LINE_BYTES, eviction.byte_count, EVICTION_ITERATIONS, eviction_initial_digest);
    fprintf(out, "  \"prep_mapping\": {\"control\": \"%s\", \"high\": \"%s\", \"low\": \"%s\"},\n",
        eviction_prep_op_name(control_prep),
        eviction_prep_op_name(high_prep),
        eviction_prep_op_name(low_prep));
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"phase_local_layout\": {\"sequence\": \"P0 C0 M0 M1 C1 P1 P2 C2 M2 M3 C3 P3\", \"phases\": [\"0\", \"pi/2\", \"pi\", \"3pi/2\"], \"plus\": \"P-C\", \"minus\": \"M-C\", \"decode\": \"z=(2/4)*sum(response_k*exp(i*theta_k))\"},\n");
    fprintf(out, "  \"predeclared_observable\": {\"sentinel\": \"remote_store_same_value over carrier after each eviction-buffer precondition\", \"sham\": \"all P/C/M windows use control prep\", \"candidate\": \"P and M swap high/low preps across imaginary quadrature phases\", \"acceptance\": \"opposed candidate Im plus/minus and min(abs(candidate imag)) > 3 * sham_floor for either established coherence counter\"},\n");
    fprintf(out, "  \"variants\": [\n");
    for (int variant = 0; variant < EVICTION_PHASE_VARIANT_COUNT; variant++) {
        fprintf(out, "    {\"name\": \"%s\", \"windows\": [\n", variant_names[variant]);
        for (int i = 0; i < EVICTION_PHASE_WINDOW_COUNT; i++) {
            fprintf(out, "      {\"token\": \"%s\", \"role\": \"%s\", \"phase_index\": %d, \"prep\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"carrier_restored\": %s, \"eviction_digest_after_hex\": \"0x%016" PRIx64 "\", \"eviction_restored\": %s, \"group\": ",
                windows[variant][i].token,
                windows[variant][i].role,
                windows[variant][i].phase_index,
                eviction_prep_op_name(windows[variant][i].prep),
                window_rc[variant][i],
                durations[variant][i],
                digest_before[variant][i],
                digest_after[variant][i],
                json_bool(digest_after[variant][i] == initial_digest),
                eviction_digest_after[variant][i],
                json_bool(eviction_digest_after[variant][i] == eviction_initial_digest));
            print_group_result(out, primary_group, &results[variant][i]);
            fprintf(out, "}%s\n", i + 1 == EVICTION_PHASE_WINDOW_COUNT ? "" : ",");
        }
        fprintf(out, "    ]}%s\n", variant + 1 == EVICTION_PHASE_VARIANT_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"decoded_observables\": [");
    print_phase_decode(out, "cache_block_commands_change_to_dirty", &sham_c2d, &candidate_c2d);
    fprintf(out, ",");
    print_phase_decode(out, "probe_responses_dirty", &sham_probe, &candidate_probe);
    fprintf(out, "],\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"carrier_restored\": %s, \"eviction_buffer_restored\": %s, \"change_to_dirty_phase_local_signal\": %s, \"probe_dirty_phase_local_signal\": %s, \"eviction_phase_local_captured\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(carrier_restored),
        json_bool(eviction_restored),
        json_bool(c2d_signal),
        json_bool(probe_signal),
        json_bool(eviction_phase_local_captured));
    fprintf(out, "}\n");
    fclose(out);
    free(eviction.bytes);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        eviction_phase_local_captured ? "EVICTION_PHASE_LOCAL_CODED_RESPONSE_FOUND" : "EVICTION_PHASE_LOCAL_CODED_RESPONSE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

static int run_eviction_phase_bracketed_mode(
    const char *output_root,
    struct carrier *carrier,
    uint64_t initial_digest,
    bool c2d_strong_mode,
    bool duration_primary_mode
) {
    struct carrier eviction;
    eviction.line_count = EVICTION_LINES;
    eviction.byte_count = EVICTION_LINES * CACHE_LINE_BYTES;
    if (posix_memalign((void **)&eviction.bytes, CACHE_LINE_BYTES, eviction.byte_count) != 0) {
        fprintf(stderr, "eviction allocation failed\n");
        return 1;
    }
    init_carrier(&eviction);
    uint64_t eviction_initial_digest = fnv1a64(eviction.bytes, eviction.byte_count);

    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_B, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }

    enum {
        EVICTION_BRACKET_VARIANT_COUNT = 2,
        EVICTION_BRACKET_TOKEN_COUNT = 8,
        EVICTION_BRACKET_WINDOW_COUNT = 3
    };
    const enum eviction_prep_op control_prep = EVICT_PREP_REMOTE_READ;
    const enum eviction_prep_op high_prep = (c2d_strong_mode || duration_primary_mode) ?
        EVICT_PREP_NONE : EVICT_PREP_HOME_READ;
    const enum eviction_prep_op low_prep = EVICT_PREP_HOME_WRITE;
    const char *variant_names[EVICTION_BRACKET_VARIANT_COUNT] = {
        "equal_bracket_sham",
        (c2d_strong_mode || duration_primary_mode) ?
            "bracketed_none_home_write_candidate" : "bracketed_high_low_candidate"
    };
    const char *bracket_window_names[EVICTION_BRACKET_WINDOW_COUNT] = {
        "control_before",
        "token",
        "control_after"
    };
    const struct eviction_bracket_token_spec tokens[EVICTION_BRACKET_VARIANT_COUNT][EVICTION_BRACKET_TOKEN_COUNT] = {
        {
            {"P0", "P", 0, EVICT_PREP_REMOTE_READ},
            {"M0", "M", 0, EVICT_PREP_REMOTE_READ},
            {"M1", "M", 1, EVICT_PREP_REMOTE_READ},
            {"P1", "P", 1, EVICT_PREP_REMOTE_READ},
            {"P2", "P", 2, EVICT_PREP_REMOTE_READ},
            {"M2", "M", 2, EVICT_PREP_REMOTE_READ},
            {"M3", "M", 3, EVICT_PREP_REMOTE_READ},
            {"P3", "P", 3, EVICT_PREP_REMOTE_READ}
        },
        {
            {"P0", "P", 0, (c2d_strong_mode || duration_primary_mode) ? EVICT_PREP_NONE : EVICT_PREP_HOME_READ},
            {"M0", "M", 0, EVICT_PREP_HOME_WRITE},
            {"M1", "M", 1, EVICT_PREP_HOME_WRITE},
            {"P1", "P", 1, (c2d_strong_mode || duration_primary_mode) ? EVICT_PREP_NONE : EVICT_PREP_HOME_READ},
            {"P2", "P", 2, (c2d_strong_mode || duration_primary_mode) ? EVICT_PREP_NONE : EVICT_PREP_HOME_READ},
            {"M2", "M", 2, EVICT_PREP_HOME_WRITE},
            {"M3", "M", 3, (c2d_strong_mode || duration_primary_mode) ? EVICT_PREP_NONE : EVICT_PREP_HOME_READ},
            {"P3", "P", 3, EVICT_PREP_HOME_WRITE}
        }
    };

    struct group_result results[EVICTION_BRACKET_VARIANT_COUNT][EVICTION_BRACKET_TOKEN_COUNT][EVICTION_BRACKET_WINDOW_COUNT];
    uint64_t durations[EVICTION_BRACKET_VARIANT_COUNT][EVICTION_BRACKET_TOKEN_COUNT][EVICTION_BRACKET_WINDOW_COUNT];
    uint64_t digest_before[EVICTION_BRACKET_VARIANT_COUNT][EVICTION_BRACKET_TOKEN_COUNT][EVICTION_BRACKET_WINDOW_COUNT];
    uint64_t digest_after[EVICTION_BRACKET_VARIANT_COUNT][EVICTION_BRACKET_TOKEN_COUNT][EVICTION_BRACKET_WINDOW_COUNT];
    uint64_t eviction_digest_after[EVICTION_BRACKET_VARIANT_COUNT][EVICTION_BRACKET_TOKEN_COUNT][EVICTION_BRACKET_WINDOW_COUNT];
    int window_rc[EVICTION_BRACKET_VARIANT_COUNT][EVICTION_BRACKET_TOKEN_COUNT][EVICTION_BRACKET_WINDOW_COUNT];

    for (int variant = 0; variant < EVICTION_BRACKET_VARIANT_COUNT; variant++) {
        for (int token = 0; token < EVICTION_BRACKET_TOKEN_COUNT; token++) {
            for (int window = 0; window < EVICTION_BRACKET_WINDOW_COUNT; window++) {
                memset(&results[variant][token][window], 0, sizeof(results[variant][token][window]));
                durations[variant][token][window] = 0;
                digest_before[variant][token][window] = 0;
                digest_after[variant][token][window] = 0;
                eviction_digest_after[variant][token][window] = 0;
                enum eviction_prep_op prep = window == 1 ? tokens[variant][token].center_prep : control_prep;
                if (primary_open_rc != 0) {
                    window_rc[variant][token][window] = -ENODEV;
                } else {
                    window_rc[variant][token][window] = measure_eviction_sentinel_window(
                        primary_group,
                        prep,
                        carrier,
                        &eviction,
                        &results[variant][token][window],
                        &durations[variant][token][window],
                        &digest_before[variant][token][window],
                        &digest_after[variant][token][window],
                        &eviction_digest_after[variant][token][window]
                    );
                }
            }
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool carrier_restored = true;
    bool eviction_restored = true;
    for (int variant = 0; variant < EVICTION_BRACKET_VARIANT_COUNT; variant++) {
        for (int token = 0; token < EVICTION_BRACKET_TOKEN_COUNT; token++) {
            for (int window = 0; window < EVICTION_BRACKET_WINDOW_COUNT; window++) {
                all_windows_ok = all_windows_ok && window_rc[variant][token][window] == 0;
                all_unmultiplexed = all_unmultiplexed && results[variant][token][window].unmultiplexed;
                carrier_restored = carrier_restored && digest_after[variant][token][window] == initial_digest;
                eviction_restored = eviction_restored &&
                    eviction_digest_after[variant][token][window] == eviction_initial_digest;
            }
        }
    }

    struct phase_decode sham_c2d;
    struct phase_decode candidate_c2d;
    struct phase_decode sham_probe;
    struct phase_decode candidate_probe;
    struct phase_decode sham_duration;
    struct phase_decode candidate_duration;
    decode_eviction_bracketed_observable(tokens[0], results[0], "cache_block_commands_change_to_dirty", &sham_c2d);
    decode_eviction_bracketed_observable(tokens[1], results[1], "cache_block_commands_change_to_dirty", &candidate_c2d);
    decode_eviction_bracketed_observable(tokens[0], results[0], "probe_responses_dirty", &sham_probe);
    decode_eviction_bracketed_observable(tokens[1], results[1], "probe_responses_dirty", &candidate_probe);
    decode_eviction_bracketed_duration(tokens[0], durations[0], &sham_duration);
    decode_eviction_bracketed_duration(tokens[1], durations[1], &candidate_duration);
    bool c2d_signal = phase_decode_signal(&sham_c2d, &candidate_c2d);
    bool probe_signal = phase_decode_signal(&sham_probe, &candidate_probe);
    bool duration_signal = phase_decode_signal(&sham_duration, &candidate_duration);
    bool eviction_phase_bracketed_captured = primary_open_rc == 0 && all_windows_ok && all_unmultiplexed &&
        carrier_restored && eviction_restored &&
        (duration_primary_mode ? duration_signal : (c2d_signal || probe_signal));
    const char *result_schema = duration_primary_mode ? EVICTION_PHASE_BRACKETED_DURATION_RESULT_SCHEMA :
        (c2d_strong_mode ? EVICTION_PHASE_BRACKETED_C2D_RESULT_SCHEMA : EVICTION_PHASE_BRACKETED_RESULT_SCHEMA);
    const char *result_file_name = duration_primary_mode ? "F10_EVICTION_PHASE_BRACKETED_DURATION_RESULT.json" :
        (c2d_strong_mode ? "F10_EVICTION_PHASE_BRACKETED_C2D_RESULT.json" : "F10_EVICTION_PHASE_BRACKETED_RESULT.json");
    const char *found_status = duration_primary_mode ? "EVICTION_PHASE_BRACKETED_DURATION_RESPONSE_FOUND" :
        (c2d_strong_mode ? "EVICTION_PHASE_BRACKETED_C2D_RESPONSE_FOUND" : "EVICTION_PHASE_BRACKETED_CODED_RESPONSE_FOUND");
    const char *not_found_status = duration_primary_mode ? "EVICTION_PHASE_BRACKETED_DURATION_RESPONSE_NOT_ESTABLISHED" :
        (c2d_strong_mode ? "EVICTION_PHASE_BRACKETED_C2D_RESPONSE_NOT_ESTABLISHED" : "EVICTION_PHASE_BRACKETED_CODED_RESPONSE_NOT_ESTABLISHED");
    const char *candidate_description = (c2d_strong_mode || duration_primary_mode) ?
        "P uses no-prep centers and M uses home-write centers across imaginary quadrature phases" :
        "P and M centers swap high/low preps across imaginary quadrature phases";
    const char *acceptance_description = duration_primary_mode ?
        "opposed candidate Im plus/minus and min(abs(candidate imag)) > 3 * sham_floor for bracketed duration_ns" :
        "opposed candidate Im plus/minus and min(abs(candidate imag)) > 3 * sham_floor for either established coherence counter";

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/%s", output_root, result_file_name);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) {
        free(eviction.bytes);
        return 1;
    }
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        free(eviction.bytes);
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", result_schema);
    fprintf(out, "  \"status\": \"%s\",\n", eviction_phase_bracketed_captured ? found_status : not_found_status);
    fprintf(out, "  \"claim_ceiling\": \"Bracketed phase-local eviction-sentinel PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"observed_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true, \"physical_address_access\": false, \"cache_set_mapping\": false, \"private_branch_routing\": false},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"eviction_buffer\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"iterations\": %d, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", eviction.line_count, CACHE_LINE_BYTES, eviction.byte_count, EVICTION_ITERATIONS, eviction_initial_digest);
    fprintf(out, "  \"prep_mapping\": {\"control\": \"%s\", \"high\": \"%s\", \"low\": \"%s\"},\n",
        eviction_prep_op_name(control_prep),
        eviction_prep_op_name(high_prep),
        eviction_prep_op_name(low_prep));
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"phase_local_layout\": {\"token_sequence\": \"P0 M0 M1 P1 P2 M2 M3 P3\", \"bracket\": \"control_before token control_after\", \"phases\": [\"0\", \"pi/2\", \"pi\", \"3pi/2\"], \"plus\": \"P - local_control_mean\", \"minus\": \"M - local_control_mean\", \"decode\": \"z=(2/4)*sum(response_k*exp(i*theta_k))\"},\n");
    fprintf(out, "  \"predeclared_observable\": {\"sentinel\": \"remote_store_same_value over carrier after each eviction-buffer precondition\", \"sham\": \"all token centers and bracket controls use control prep\", \"candidate\": \"%s\", \"acceptance\": \"%s\"},\n", candidate_description, acceptance_description);
    fprintf(out, "  \"variants\": [\n");
    for (int variant = 0; variant < EVICTION_BRACKET_VARIANT_COUNT; variant++) {
        fprintf(out, "    {\"name\": \"%s\", \"tokens\": [\n", variant_names[variant]);
        for (int token = 0; token < EVICTION_BRACKET_TOKEN_COUNT; token++) {
            fprintf(out, "      {\"token\": \"%s\", \"role\": \"%s\", \"phase_index\": %d, \"center_prep\": \"%s\", \"bracket_windows\": [\n",
                tokens[variant][token].token,
                tokens[variant][token].role,
                tokens[variant][token].phase_index,
                eviction_prep_op_name(tokens[variant][token].center_prep));
            for (int window = 0; window < EVICTION_BRACKET_WINDOW_COUNT; window++) {
                enum eviction_prep_op prep = window == 1 ? tokens[variant][token].center_prep : control_prep;
                fprintf(out, "        {\"slot\": \"%s\", \"prep\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"carrier_restored\": %s, \"eviction_digest_after_hex\": \"0x%016" PRIx64 "\", \"eviction_restored\": %s, \"group\": ",
                    bracket_window_names[window],
                    eviction_prep_op_name(prep),
                    window_rc[variant][token][window],
                    durations[variant][token][window],
                    digest_before[variant][token][window],
                    digest_after[variant][token][window],
                    json_bool(digest_after[variant][token][window] == initial_digest),
                    eviction_digest_after[variant][token][window],
                    json_bool(eviction_digest_after[variant][token][window] == eviction_initial_digest));
                print_group_result(out, primary_group, &results[variant][token][window]);
                fprintf(out, "}%s\n", window + 1 == EVICTION_BRACKET_WINDOW_COUNT ? "" : ",");
            }
            fprintf(out, "      ]}%s\n", token + 1 == EVICTION_BRACKET_TOKEN_COUNT ? "" : ",");
        }
        fprintf(out, "    ]}%s\n", variant + 1 == EVICTION_BRACKET_VARIANT_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"decoded_observables\": [");
    print_phase_decode(out, "cache_block_commands_change_to_dirty", &sham_c2d, &candidate_c2d);
    fprintf(out, ",");
    print_phase_decode(out, "probe_responses_dirty", &sham_probe, &candidate_probe);
    fprintf(out, ",");
    print_phase_decode(out, "duration_ns", &sham_duration, &candidate_duration);
    fprintf(out, "],\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"carrier_restored\": %s, \"eviction_buffer_restored\": %s, \"change_to_dirty_phase_local_signal\": %s, \"probe_dirty_phase_local_signal\": %s, \"duration_phase_local_signal\": %s, \"eviction_phase_bracketed_captured\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(carrier_restored),
        json_bool(eviction_restored),
        json_bool(c2d_signal),
        json_bool(probe_signal),
        json_bool(duration_signal),
        json_bool(eviction_phase_bracketed_captured));
    fprintf(out, "}\n");
    fclose(out);
    free(eviction.bytes);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        eviction_phase_bracketed_captured ? found_status : not_found_status,
        result_path);
    return 0;
}

static int run_phase_local_pmu_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_B, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }

    enum { PHASE_LOCAL_VARIANT_COUNT = 2, PHASE_LOCAL_WINDOW_COUNT = 12 };
    const size_t small_lines = PHASE_LOCAL_BANK_LINES * 2u;
    const size_t equal_lines = PHASE_LOCAL_BANK_LINES * 4u;
    const size_t large_lines = CARRIER_LINES;
    const char *variant_names[PHASE_LOCAL_VARIANT_COUNT] = {
        "phase_local_sham",
        "phase_local_candidate"
    };
    const struct phase_local_window_spec windows[PHASE_LOCAL_VARIANT_COUNT][PHASE_LOCAL_WINDOW_COUNT] = {
        {
            {"P0", "P", 0, PHASE_LOCAL_BANK_LINES * 4u, 0u},
            {"C0", "C", 0, PHASE_LOCAL_BANK_LINES * 4u, 0u},
            {"M0", "M", 0, PHASE_LOCAL_BANK_LINES * 4u, 0u},
            {"M1", "M", 1, PHASE_LOCAL_BANK_LINES * 4u, 2u},
            {"C1", "C", 1, PHASE_LOCAL_BANK_LINES * 4u, 2u},
            {"P1", "P", 1, PHASE_LOCAL_BANK_LINES * 4u, 2u},
            {"P2", "P", 2, PHASE_LOCAL_BANK_LINES * 4u, 4u},
            {"C2", "C", 2, PHASE_LOCAL_BANK_LINES * 4u, 4u},
            {"M2", "M", 2, PHASE_LOCAL_BANK_LINES * 4u, 4u},
            {"M3", "M", 3, PHASE_LOCAL_BANK_LINES * 4u, 6u},
            {"C3", "C", 3, PHASE_LOCAL_BANK_LINES * 4u, 6u},
            {"P3", "P", 3, PHASE_LOCAL_BANK_LINES * 4u, 6u}
        },
        {
            {"P0", "P", 0, CARRIER_LINES, 0u},
            {"C0", "C", 0, PHASE_LOCAL_BANK_LINES * 4u, 0u},
            {"M0", "M", 0, CARRIER_LINES, 0u},
            {"M1", "M", 1, PHASE_LOCAL_BANK_LINES * 2u, 2u},
            {"C1", "C", 1, PHASE_LOCAL_BANK_LINES * 4u, 2u},
            {"P1", "P", 1, CARRIER_LINES, 0u},
            {"P2", "P", 2, PHASE_LOCAL_BANK_LINES * 2u, 4u},
            {"C2", "C", 2, PHASE_LOCAL_BANK_LINES * 4u, 4u},
            {"M2", "M", 2, PHASE_LOCAL_BANK_LINES * 2u, 4u},
            {"M3", "M", 3, CARRIER_LINES, 0u},
            {"C3", "C", 3, PHASE_LOCAL_BANK_LINES * 4u, 6u},
            {"P3", "P", 3, PHASE_LOCAL_BANK_LINES * 2u, 6u}
        }
    };

    struct group_result results[PHASE_LOCAL_VARIANT_COUNT][PHASE_LOCAL_WINDOW_COUNT];
    uint64_t durations[PHASE_LOCAL_VARIANT_COUNT][PHASE_LOCAL_WINDOW_COUNT];
    uint64_t digest_before[PHASE_LOCAL_VARIANT_COUNT][PHASE_LOCAL_WINDOW_COUNT];
    uint64_t digest_after[PHASE_LOCAL_VARIANT_COUNT][PHASE_LOCAL_WINDOW_COUNT];
    int window_rc[PHASE_LOCAL_VARIANT_COUNT][PHASE_LOCAL_WINDOW_COUNT];

    for (int variant = 0; variant < PHASE_LOCAL_VARIANT_COUNT; variant++) {
        for (int i = 0; i < PHASE_LOCAL_WINDOW_COUNT; i++) {
            memset(&results[variant][i], 0, sizeof(results[variant][i]));
            durations[variant][i] = 0;
            digest_before[variant][i] = 0;
            digest_after[variant][i] = 0;
            if (primary_open_rc != 0) {
                window_rc[variant][i] = -ENODEV;
            } else {
                window_rc[variant][i] = measure_phase_local_pmu_window(
                    primary_group,
                    &windows[variant][i],
                    carrier,
                    &results[variant][i],
                    &durations[variant][i],
                    &digest_before[variant][i],
                    &digest_after[variant][i]
                );
            }
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool all_restored = true;
    for (int variant = 0; variant < PHASE_LOCAL_VARIANT_COUNT; variant++) {
        for (int i = 0; i < PHASE_LOCAL_WINDOW_COUNT; i++) {
            all_windows_ok = all_windows_ok && window_rc[variant][i] == 0;
            all_unmultiplexed = all_unmultiplexed && results[variant][i].unmultiplexed;
            all_restored = all_restored && digest_after[variant][i] == initial_digest;
        }
    }

    struct phase_decode sham_c2d;
    struct phase_decode candidate_c2d;
    struct phase_decode sham_probe;
    struct phase_decode candidate_probe;
    decode_phase_observable(windows[0], results[0], "cache_block_commands_change_to_dirty", &sham_c2d);
    decode_phase_observable(windows[1], results[1], "cache_block_commands_change_to_dirty", &candidate_c2d);
    decode_phase_observable(windows[0], results[0], "probe_responses_dirty", &sham_probe);
    decode_phase_observable(windows[1], results[1], "probe_responses_dirty", &candidate_probe);
    bool c2d_signal = phase_decode_signal(&sham_c2d, &candidate_c2d);
    bool probe_signal = phase_decode_signal(&sham_probe, &candidate_probe);
    bool phase_local_pmu_captured = primary_open_rc == 0 && all_windows_ok && all_unmultiplexed &&
        all_restored && (c2d_signal || probe_signal);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_PHASE_LOCAL_PMU_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", PHASE_LOCAL_PMU_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", phase_local_pmu_captured ? "PHASE_LOCAL_PMU_CODED_RESPONSE_FOUND" : "PHASE_LOCAL_PMU_CODED_RESPONSE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Phase-local ownership-intent PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"observed_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"footprint_lines\": {\"small\": %zu, \"equal\": %zu, \"large\": %zu, \"bank_lines\": %u, \"bank_plan\": \"rotated contiguous bank spans; not fixed-prefix\"},\n",
        small_lines, equal_lines, large_lines, (unsigned int)PHASE_LOCAL_BANK_LINES);
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"phase_local_layout\": {\"sequence\": \"P0 C0 M0 M1 C1 P1 P2 C2 M2 M3 C3 P3\", \"phases\": [\"0\", \"pi/2\", \"pi\", \"3pi/2\"], \"plus\": \"P-C\", \"minus\": \"M-C\", \"decode\": \"z=(2/4)*sum(response_k*exp(i*theta_k))\"},\n");
    fprintf(out, "  \"predeclared_observable\": {\"primary\": \"cache_block_commands_change_to_dirty/cpu_cycles_not_halted\", \"secondary\": \"probe_responses_dirty/cpu_cycles_not_halted\", \"acceptance\": \"opposed candidate Im plus/minus and min(abs(candidate imag)) > 3 * sham_floor for either event\"},\n");
    fprintf(out, "  \"variants\": [\n");
    for (int variant = 0; variant < PHASE_LOCAL_VARIANT_COUNT; variant++) {
        fprintf(out, "    {\"name\": \"%s\", \"windows\": [\n", variant_names[variant]);
        for (int i = 0; i < PHASE_LOCAL_WINDOW_COUNT; i++) {
            fprintf(out, "      {\"token\": \"%s\", \"role\": \"%s\", \"phase_index\": %d, \"line_count\": %zu, \"start_bank\": %zu, \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"carrier_restored\": %s, \"group\": ",
                windows[variant][i].token,
                windows[variant][i].role,
                windows[variant][i].phase_index,
                windows[variant][i].line_count,
                windows[variant][i].start_bank,
                window_rc[variant][i],
                durations[variant][i],
                digest_before[variant][i],
                digest_after[variant][i],
                json_bool(digest_after[variant][i] == initial_digest));
            print_group_result(out, primary_group, &results[variant][i]);
            fprintf(out, "}%s\n", i + 1 == PHASE_LOCAL_WINDOW_COUNT ? "" : ",");
        }
        fprintf(out, "    ]}%s\n", variant + 1 == PHASE_LOCAL_VARIANT_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"decoded_observables\": [");
    print_phase_decode(out, "cache_block_commands_change_to_dirty", &sham_c2d, &candidate_c2d);
    fprintf(out, ",");
    print_phase_decode(out, "probe_responses_dirty", &sham_probe, &candidate_probe);
    fprintf(out, "],\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"carrier_restored\": %s, \"change_to_dirty_phase_local_signal\": %s, \"probe_dirty_phase_local_signal\": %s, \"phase_local_pmu_captured\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(all_restored),
        json_bool(c2d_signal),
        json_bool(probe_signal),
        json_bool(phase_local_pmu_captured));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        phase_local_pmu_captured ? "PHASE_LOCAL_PMU_CODED_RESPONSE_FOUND" : "PHASE_LOCAL_PMU_CODED_RESPONSE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

static int measure_path_sequence(
    const struct path_step steps[4],
    struct carrier *carrier,
    struct group_result results[4],
    uint64_t durations[4],
    uint64_t digest_before[4],
    uint64_t digest_after[4],
    int rc[4],
    uint64_t initial_digest,
    bool observe_actor_core
) {
    home_core_restore(carrier);
    for (int i = 0; i < 4; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_before[i] = 0;
        digest_after[i] = 0;
        int observed_core = observe_actor_core ? path_step_actor_core(&steps[i]) : CATCAS_CORE_B;
        if (observed_core < 0) {
            rc[i] = -EINVAL;
            continue;
        }
        rc[i] = measure_path_step(primary_group, &steps[i], observed_core, carrier, &results[i],
            &durations[i], &digest_before[i], &digest_after[i]);
    }
    home_core_restore(carrier);
    return fnv1a64(carrier->bytes, carrier->byte_count) == initial_digest ? 0 : -1;
}

static void print_path_steps(
    FILE *out,
    const char *name,
    const struct path_step steps[4],
    const struct group_result results[4],
    const uint64_t durations[4],
    const uint64_t digest_before[4],
    const uint64_t digest_after[4],
    const int rc[4],
    long double area,
    uint64_t initial_digest,
    bool observe_actor_core
) {
    fprintf(out, "    {\"name\": \"%s\", \"signed_area_cycles_normalized\": %.12Le, \"steps\": [\n", name, area);
    for (int i = 0; i < 4; i++) {
        int observed_core = observe_actor_core ? path_step_actor_core(&steps[i]) : CATCAS_CORE_B;
        fprintf(out, "      {\"name\": \"%s\", \"line_set\": %d, \"observed_core\": %d, \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_hex\": \"0x%016" PRIx64 "\", \"bytes_unchanged\": %s, \"group\": ",
            steps[i].name,
            steps[i].line_set,
            observed_core,
            rc[i],
            durations[i],
            digest_before[i],
            digest_after[i],
            json_bool(digest_before[i] == initial_digest && digest_after[i] == initial_digest));
        print_group_result(out, primary_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == 4 ? "" : ",");
    }
    fprintf(out, "    ]}");
}

static int probe_primary_group_on_core(int core) {
    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, core, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }
    return primary_open_rc;
}

static int run_path_mode(
    const char *output_root,
    struct carrier *carrier,
    uint64_t initial_digest,
    enum path_mode_kind mode_kind
) {
    bool observe_actor_core = mode_kind == PATH_MODE_DUAL_OBSERVE;
    bool rw_observe = mode_kind == PATH_MODE_RW_OBSERVE;
    const char *schema_id = PATH_RESULT_SCHEMA;
    const char *result_file = "F10_PATH_DEPENDENCE_PILOT_RESULT.json";
    const char *positive_status = "PATH_DEPENDENCE_PILOT_OBSERVED";
    const char *negative_status = "PATH_DEPENDENCE_NOT_ESTABLISHED";
    const char *claim_ceiling = "Path-dependence pilot only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim";
    if (observe_actor_core) {
        schema_id = PATH_DUAL_OBSERVE_RESULT_SCHEMA;
        result_file = "F10_PATH_DUAL_OBSERVE_RESULT.json";
        positive_status = "PATH_DUAL_OBSERVE_CANDIDATE";
        negative_status = "PATH_DUAL_OBSERVE_NOT_ESTABLISHED";
        claim_ceiling = "Dual-observed path pilot only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim";
    } else if (rw_observe) {
        schema_id = PATH_RW_OBSERVE_RESULT_SCHEMA;
        result_file = "F10_PATH_RW_OBSERVE_RESULT.json";
        positive_status = "PATH_RW_OBSERVE_CANDIDATE";
        negative_status = "PATH_RW_OBSERVE_NOT_ESTABLISHED";
        claim_ceiling = "Read/store path pilot only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim";
    }
    int primary_open_rc = probe_primary_group_on_core(CATCAS_CORE_B);
    int home_open_rc = observe_actor_core ? probe_primary_group_on_core(CATCAS_CORE_A) : 0;
    bool groups_available = primary_open_rc == 0 && home_open_rc == 0;

    const struct path_step forward[4] = {
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"home_store_set0", PATH_HOME_STORE, 0},
        {"home_store_set1", PATH_HOME_STORE, 1}
    };
    const struct path_step reverse[4] = {
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"home_store_set1", PATH_HOME_STORE, 1},
        {"home_store_set0", PATH_HOME_STORE, 0}
    };
    const struct path_step shuffle[4] = {
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"home_store_set0", PATH_HOME_STORE, 0},
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"home_store_set1", PATH_HOME_STORE, 1}
    };
    const struct path_step reverse_shuffle[4] = {
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"home_store_set1", PATH_HOME_STORE, 1},
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"home_store_set0", PATH_HOME_STORE, 0}
    };
    const struct path_step identity[4] = {
        {"home_store_set0", PATH_HOME_STORE, 0},
        {"home_store_set1", PATH_HOME_STORE, 1},
        {"home_store_set0", PATH_HOME_STORE, 0},
        {"home_store_set1", PATH_HOME_STORE, 1}
    };
    const struct path_step rw_forward[4] = {
        {"remote_read_set0", PATH_REMOTE_READ, 0},
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"remote_read_set1", PATH_REMOTE_READ, 1},
        {"remote_store_set0", PATH_REMOTE_STORE, 0}
    };
    const struct path_step rw_reverse[4] = {
        {"remote_read_set1", PATH_REMOTE_READ, 1},
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"remote_read_set0", PATH_REMOTE_READ, 0},
        {"remote_store_set1", PATH_REMOTE_STORE, 1}
    };
    const struct path_step rw_shuffle[4] = {
        {"remote_read_set0", PATH_REMOTE_READ, 0},
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"remote_read_set1", PATH_REMOTE_READ, 1},
        {"remote_store_set1", PATH_REMOTE_STORE, 1}
    };
    const struct path_step rw_reverse_shuffle[4] = {
        {"remote_read_set1", PATH_REMOTE_READ, 1},
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"remote_read_set0", PATH_REMOTE_READ, 0},
        {"remote_store_set0", PATH_REMOTE_STORE, 0}
    };
    const struct path_step rw_identity[4] = {
        {"remote_read_set0", PATH_REMOTE_READ, 0},
        {"remote_read_set1", PATH_REMOTE_READ, 1},
        {"remote_read_set0", PATH_REMOTE_READ, 0},
        {"remote_read_set1", PATH_REMOTE_READ, 1}
    };
    const struct path_step *forward_steps = rw_observe ? rw_forward : forward;
    const struct path_step *reverse_steps = rw_observe ? rw_reverse : reverse;
    const struct path_step *shuffle_steps = rw_observe ? rw_shuffle : shuffle;
    const struct path_step *reverse_shuffle_steps = rw_observe ? rw_reverse_shuffle : reverse_shuffle;
    const struct path_step *identity_steps = rw_observe ? rw_identity : identity;

    struct group_result forward_results[4], reverse_results[4], shuffle_results[4], reverse_shuffle_results[4], identity_results[4];
    uint64_t forward_durations[4], reverse_durations[4], shuffle_durations[4], reverse_shuffle_durations[4], identity_durations[4];
    uint64_t forward_before[4], reverse_before[4], shuffle_before[4], reverse_shuffle_before[4], identity_before[4];
    uint64_t forward_after[4], reverse_after[4], shuffle_after[4], reverse_shuffle_after[4], identity_after[4];
    int forward_rc[4], reverse_rc[4], shuffle_rc[4], reverse_shuffle_rc[4], identity_rc[4];
    int forward_restore = groups_available ? measure_path_sequence(forward_steps, carrier, forward_results, forward_durations, forward_before, forward_after, forward_rc, initial_digest, observe_actor_core) : -ENODEV;
    int reverse_restore = groups_available ? measure_path_sequence(reverse_steps, carrier, reverse_results, reverse_durations, reverse_before, reverse_after, reverse_rc, initial_digest, observe_actor_core) : -ENODEV;
    int shuffle_restore = groups_available ? measure_path_sequence(shuffle_steps, carrier, shuffle_results, shuffle_durations, shuffle_before, shuffle_after, shuffle_rc, initial_digest, observe_actor_core) : -ENODEV;
    int reverse_shuffle_restore = groups_available ? measure_path_sequence(reverse_shuffle_steps, carrier, reverse_shuffle_results, reverse_shuffle_durations, reverse_shuffle_before, reverse_shuffle_after, reverse_shuffle_rc, initial_digest, observe_actor_core) : -ENODEV;
    int identity_restore = groups_available ? measure_path_sequence(identity_steps, carrier, identity_results, identity_durations, identity_before, identity_after, identity_rc, initial_digest, observe_actor_core) : -ENODEV;

    bool all_windows_ok = groups_available;
    bool all_unmultiplexed = groups_available;
    bool bytes_unchanged = forward_restore == 0 && reverse_restore == 0 &&
        shuffle_restore == 0 && reverse_shuffle_restore == 0 && identity_restore == 0;
    for (int i = 0; i < 4; i++) {
        all_windows_ok = all_windows_ok && forward_rc[i] == 0 && reverse_rc[i] == 0 &&
            shuffle_rc[i] == 0 && reverse_shuffle_rc[i] == 0 && identity_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && forward_results[i].unmultiplexed &&
            reverse_results[i].unmultiplexed && shuffle_results[i].unmultiplexed &&
            reverse_shuffle_results[i].unmultiplexed && identity_results[i].unmultiplexed;
        bytes_unchanged = bytes_unchanged &&
            forward_before[i] == initial_digest && forward_after[i] == initial_digest &&
            reverse_before[i] == initial_digest && reverse_after[i] == initial_digest &&
            shuffle_before[i] == initial_digest && shuffle_after[i] == initial_digest &&
            reverse_shuffle_before[i] == initial_digest && reverse_shuffle_after[i] == initial_digest &&
            identity_before[i] == initial_digest && identity_after[i] == initial_digest;
    }

    long double forward_area = path_signed_area(forward_results, primary_group, 4);
    long double reverse_area = path_signed_area(reverse_results, primary_group, 4);
    long double shuffle_area = path_signed_area(shuffle_results, primary_group, 4);
    long double reverse_shuffle_area = path_signed_area(reverse_shuffle_results, primary_group, 4);
    long double identity_area = path_signed_area(identity_results, primary_group, 4);
    long double min_oriented = abs_ld(forward_area) < abs_ld(reverse_area) ? abs_ld(forward_area) : abs_ld(reverse_area);
    bool sign_reversal = (forward_area < 0.0L && reverse_area > 0.0L) ||
        (forward_area > 0.0L && reverse_area < 0.0L);
    bool controls_small = min_oriented > 0.0L &&
        abs_ld(shuffle_area) * 4.0L < min_oriented &&
        abs_ld(reverse_shuffle_area) * 4.0L < min_oriented &&
        abs_ld(identity_area) * 4.0L < min_oriented;
    bool path_pilot = all_windows_ok && all_unmultiplexed && bytes_unchanged &&
        sign_reversal && controls_small;

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/%s", output_root, result_file);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", schema_id);
    fprintf(out, "  \"status\": \"%s\",\n", path_pilot ? positive_status : negative_status);
    fprintf(out, "  \"claim_ceiling\": \"%s\",\n", claim_ceiling);
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"remote_core\": %d, \"fixed_observed_core\": %d, \"observe_actor_core\": %s},\n",
        CATCAS_CORE_A,
        CATCAS_CORE_B,
        CATCAS_CORE_B,
        json_bool(observe_actor_core));
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true},\n");
    fprintf(out, "  \"carrier\": {\"line_sets\": 2, \"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"home_group_open_rc\": %d,\n", home_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"coordinate_x\": \"cache_block_commands_change_to_dirty / cpu_cycles_not_halted\", \"coordinate_y\": \"probe_responses_dirty / cpu_cycles_not_halted\", \"signed_area\": \"sum((x_i * y_next) - (y_i * x_next)) / 2 over four path steps\"},\n");
    fprintf(out, "  \"paths\": [\n");
    print_path_steps(out, "forward", forward_steps, forward_results, forward_durations, forward_before, forward_after, forward_rc, forward_area, initial_digest, observe_actor_core);
    fprintf(out, ",\n");
    print_path_steps(out, "reverse", reverse_steps, reverse_results, reverse_durations, reverse_before, reverse_after, reverse_rc, reverse_area, initial_digest, observe_actor_core);
    fprintf(out, ",\n");
    print_path_steps(out, "shuffle", shuffle_steps, shuffle_results, shuffle_durations, shuffle_before, shuffle_after, shuffle_rc, shuffle_area, initial_digest, observe_actor_core);
    fprintf(out, ",\n");
    print_path_steps(out, "reverse_shuffle", reverse_shuffle_steps, reverse_shuffle_results, reverse_shuffle_durations, reverse_shuffle_before, reverse_shuffle_after, reverse_shuffle_rc, reverse_shuffle_area, initial_digest, observe_actor_core);
    fprintf(out, ",\n");
    print_path_steps(out, "identity", identity_steps, identity_results, identity_durations, identity_before, identity_after, identity_rc, identity_area, initial_digest, observe_actor_core);
    fprintf(out, "\n  ],\n");
    fprintf(out, "  \"areas_cycles_normalized\": {\"forward\": %.12Le, \"reverse\": %.12Le, \"shuffle\": %.12Le, \"reverse_shuffle\": %.12Le, \"identity\": %.12Le},\n",
        forward_area, reverse_area, shuffle_area, reverse_shuffle_area, identity_area);
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"bytes_unchanged\": %s, \"sign_reversal\": %s, \"controls_small\": %s, \"path_dependence_pilot\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(bytes_unchanged),
        json_bool(sign_reversal),
        json_bool(controls_small),
        json_bool(path_pilot));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        path_pilot ? positive_status : negative_status,
        result_path);
    return 0;
}

static int run_path_dependence_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    return run_path_mode(output_root, carrier, initial_digest, PATH_MODE_FIXED_CORE);
}

static int run_path_dual_observe_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    return run_path_mode(output_root, carrier, initial_digest, PATH_MODE_DUAL_OBSERVE);
}

static int run_path_rw_observe_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    return run_path_mode(output_root, carrier, initial_digest, PATH_MODE_RW_OBSERVE);
}

static void print_route_windows(
    FILE *out,
    const struct route_spec *route,
    const enum route_op ops[3],
    const struct group_result results[3],
    const uint64_t durations[3],
    const uint64_t digest_before[3],
    const uint64_t digest_after[3],
    const int rc[3],
    uint64_t initial_digest
) {
    fprintf(out, "    {\"name\": \"%s\", \"home_core\": %d, \"remote_core\": %d, \"windows\": [\n",
        route->name,
        route->home_core,
        route->remote_core);
    for (int i = 0; i < 3; i++) {
        fprintf(out, "      {\"op\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"bytes_restored\": %s, \"group\": ",
            route_op_name(ops[i]),
            rc[i],
            durations[i],
            digest_before[i],
            digest_after[i],
            json_bool(digest_before[i] == initial_digest && digest_after[i] == initial_digest));
        print_group_result(out, primary_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == 3 ? "" : ",");
    }
    fprintf(out, "    ]}");
}

static int run_route_state_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    const struct route_spec routes[4] = {
        {"route_4_to_5", 4, 5},
        {"route_5_to_4", 5, 4},
        {"route_2_to_3", 2, 3},
        {"route_3_to_2", 3, 2}
    };
    const enum route_op ops[3] = {
        ROUTE_IDENTITY,
        ROUTE_READ,
        ROUTE_STORE
    };
    int group_open_rc[4];
    bool groups_available = true;
    for (int route = 0; route < 4; route++) {
        group_open_rc[route] = probe_primary_group_on_core(routes[route].remote_core);
        groups_available = groups_available && group_open_rc[route] == 0;
    }

    struct group_result results[4][3];
    uint64_t durations[4][3];
    uint64_t digest_before[4][3];
    uint64_t digest_after[4][3];
    int rc[4][3];
    for (int route = 0; route < 4; route++) {
        for (int op = 0; op < 3; op++) {
            memset(&results[route][op], 0, sizeof(results[route][op]));
            durations[route][op] = 0;
            digest_before[route][op] = 0;
            digest_after[route][op] = 0;
            rc[route][op] = -ENODEV;
        }
    }
    if (groups_available) {
        for (int route = 0; route < 4; route++) {
            for (int op = 0; op < 3; op++) {
                rc[route][op] = measure_route_window(primary_group, &routes[route], ops[op],
                    carrier, &results[route][op], &durations[route][op],
                    &digest_before[route][op], &digest_after[route][op]);
            }
        }
    }
    home_core_restore(carrier);
    bool final_digest_restored = fnv1a64(carrier->bytes, carrier->byte_count) == initial_digest;

    bool all_windows_ok = groups_available;
    bool all_unmultiplexed = groups_available;
    bool bytes_restored = final_digest_restored;
    bool store_visible = groups_available;
    for (int route = 0; route < 4; route++) {
        uint64_t read_c2d = event_value_by_name(&results[route][1], primary_group, "cache_block_commands_change_to_dirty");
        uint64_t read_probe = event_value_by_name(&results[route][1], primary_group, "probe_responses_dirty");
        uint64_t store_c2d = event_value_by_name(&results[route][2], primary_group, "cache_block_commands_change_to_dirty");
        uint64_t store_probe = event_value_by_name(&results[route][2], primary_group, "probe_responses_dirty");
        store_visible = store_visible && (store_c2d + store_probe > read_c2d + read_probe);
        for (int op = 0; op < 3; op++) {
            all_windows_ok = all_windows_ok && rc[route][op] == 0;
            all_unmultiplexed = all_unmultiplexed && results[route][op].unmultiplexed;
            bytes_restored = bytes_restored &&
                digest_before[route][op] == initial_digest &&
                digest_after[route][op] == initial_digest;
        }
    }

    long double direct_identity_distance2 = route_vector_distance2(&results[0][0], &results[2][0], primary_group);
    long double direct_read_distance2 = route_vector_distance2(&results[0][1], &results[2][1], primary_group);
    long double direct_store_distance2 = route_vector_distance2(&results[0][2], &results[2][2], primary_group);
    long double swapped_identity_distance2 = route_vector_distance2(&results[1][0], &results[3][0], primary_group);
    long double swapped_read_distance2 = route_vector_distance2(&results[1][1], &results[3][1], primary_group);
    long double swapped_store_distance2 = route_vector_distance2(&results[1][2], &results[3][2], primary_group);
    bool direct_route_moved = direct_store_distance2 > 0.0L &&
        direct_store_distance2 > direct_identity_distance2 * 9.0L &&
        direct_store_distance2 > direct_read_distance2 * 9.0L;
    bool swapped_route_moved = swapped_store_distance2 > 0.0L &&
        swapped_store_distance2 > swapped_identity_distance2 * 9.0L &&
        swapped_store_distance2 > swapped_read_distance2 * 9.0L;
    bool route_state_response = all_windows_ok && all_unmultiplexed && bytes_restored &&
        store_visible && direct_route_moved && swapped_route_moved;

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_ROUTE_STATE_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", ROUTE_STATE_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", route_state_response ? "ROUTE_STATE_RESPONSE_FOUND" : "ROUTE_STATE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Route-state PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\", \"final_digest_restored\": %s},\n",
        carrier->line_count,
        CACHE_LINE_BYTES,
        carrier->byte_count,
        initial_digest,
        json_bool(final_digest_restored));
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": {\"route_4_to_5\": %d, \"route_5_to_4\": %d, \"route_2_to_3\": %d, \"route_3_to_2\": %d},\n",
        group_open_rc[0],
        group_open_rc[1],
        group_open_rc[2],
        group_open_rc[3]);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"coordinate_x\": \"cache_block_commands_change_to_dirty / cpu_cycles_not_halted\", \"coordinate_y\": \"probe_responses_dirty / cpu_cycles_not_halted\", \"route_distance\": \"squared distance between route vectors\", \"promotion_law\": \"store route distance must exceed matching identity and read route controls by more than 3x magnitude for both 4_to_5_vs_2_to_3 and 5_to_4_vs_3_to_2\"},\n");
    fprintf(out, "  \"routes\": [\n");
    for (int route = 0; route < 4; route++) {
        print_route_windows(out, &routes[route], ops, results[route], durations[route],
            digest_before[route], digest_after[route], rc[route], initial_digest);
        fprintf(out, "%s\n", route + 1 == 4 ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"route_distances_cycles_normalized_squared\": {\"direct_identity\": %.12Le, \"direct_read\": %.12Le, \"direct_store\": %.12Le, \"swapped_identity\": %.12Le, \"swapped_read\": %.12Le, \"swapped_store\": %.12Le},\n",
        direct_identity_distance2,
        direct_read_distance2,
        direct_store_distance2,
        swapped_identity_distance2,
        swapped_read_distance2,
        swapped_store_distance2);
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"bytes_restored\": %s, \"store_visible\": %s, \"direct_route_moved\": %s, \"swapped_route_moved\": %s, \"route_state_response\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(bytes_restored),
        json_bool(store_visible),
        json_bool(direct_route_moved),
        json_bool(swapped_route_moved),
        json_bool(route_state_response));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        route_state_response ? "ROUTE_STATE_RESPONSE_FOUND" : "ROUTE_STATE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

int main(int argc, char **argv) {
    const char *output_root = NULL;
    bool coherence_operator_mode = false;
    bool path_dependence_mode = false;
    bool path_dual_observe_mode = false;
    bool path_rw_observe_mode = false;
    bool route_state_mode = false;
    bool phase_local_pmu_mode = false;
    bool ibs_first_light_mode = false;
    bool wc_flush_order_mode = false;
    bool eviction_sentinel_mode = false;
    bool eviction_phase_local_mode = false;
    bool eviction_phase_bracketed_mode = false;
    bool eviction_phase_bracketed_c2d_mode = false;
    bool eviction_phase_bracketed_duration_mode = false;
    bool history_sentinel_mode = false;
    bool locked_history_mode = false;
    bool branch_history_mode = false;
    bool indirect_target_history_mode = false;
    bool translation_history_mode = false;
    bool store_load_alias_history_mode = false;
    bool prefetch_stream_mode = false;
    bool process_lifecycle_mode = false;
    bool code_footprint_history_mode = false;
    bool return_stack_history_mode = false;
    bool fp_pipeline_history_mode = false;
    bool page_permission_history_mode = false;
    bool owned_recovery_history_mode = false;
    if (argc == 3 && strcmp(argv[1], "--output-root") == 0) {
        output_root = argv[2];
    } else if (argc == 4 && strcmp(argv[1], "--coherence-operators") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        coherence_operator_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--path-dependence") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        path_dependence_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--path-dual-observe") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        path_dual_observe_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--path-rw-observe") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        path_rw_observe_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--route-state") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        route_state_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--phase-local-pmu") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        phase_local_pmu_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--ibs-first-light") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        ibs_first_light_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--wc-flush-order") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        wc_flush_order_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--eviction-sentinel") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        eviction_sentinel_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--eviction-phase-local") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        eviction_phase_local_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--eviction-phase-bracketed") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        eviction_phase_bracketed_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--eviction-phase-bracketed-c2d") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        eviction_phase_bracketed_c2d_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--eviction-phase-bracketed-duration") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        eviction_phase_bracketed_duration_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--history-sentinel") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        history_sentinel_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--locked-history") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        locked_history_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--branch-history") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        branch_history_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--indirect-target-history") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        indirect_target_history_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--translation-history") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        translation_history_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--store-load-alias-history") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        store_load_alias_history_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--prefetch-stream") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        prefetch_stream_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--process-lifecycle") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        process_lifecycle_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--code-footprint-history") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        code_footprint_history_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--return-stack-history") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        return_stack_history_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--fp-pipeline-history") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        fp_pipeline_history_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--page-permission-history") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        page_permission_history_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--owned-recovery-history") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        owned_recovery_history_mode = true;
        output_root = argv[3];
    } else if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    } else {
        fprintf(stderr, "usage: %s --output-root <absolute-output-root> | --coherence-operators --output-root <absolute-output-root> | --path-dependence --output-root <absolute-output-root> | --path-dual-observe --output-root <absolute-output-root> | --path-rw-observe --output-root <absolute-output-root> | --route-state --output-root <absolute-output-root> | --phase-local-pmu --output-root <absolute-output-root> | --ibs-first-light --output-root <absolute-output-root> | --wc-flush-order --output-root <absolute-output-root> | --eviction-sentinel --output-root <absolute-output-root> | --eviction-phase-local --output-root <absolute-output-root> | --eviction-phase-bracketed --output-root <absolute-output-root> | --eviction-phase-bracketed-c2d --output-root <absolute-output-root> | --eviction-phase-bracketed-duration --output-root <absolute-output-root> | --history-sentinel --output-root <absolute-output-root> | --locked-history --output-root <absolute-output-root> | --branch-history --output-root <absolute-output-root> | --indirect-target-history --output-root <absolute-output-root> | --translation-history --output-root <absolute-output-root> | --store-load-alias-history --output-root <absolute-output-root> | --prefetch-stream --output-root <absolute-output-root> | --process-lifecycle --output-root <absolute-output-root> | --code-footprint-history --output-root <absolute-output-root> | --return-stack-history --output-root <absolute-output-root> | --fp-pipeline-history --output-root <absolute-output-root> | --page-permission-history --output-root <absolute-output-root> | --owned-recovery-history --output-root <absolute-output-root>\n", argv[0]);
        return 2;
    }
    if (output_root[0] != '/') {
        fprintf(stderr, "output root must be absolute\n");
        return 2;
    }
    if (ensure_dir(output_root) != 0) {
        fprintf(stderr, "cannot create output root: %s\n", strerror(errno));
        return 1;
    }

    struct carrier carrier;
    carrier.line_count = CARRIER_LINES;
    carrier.byte_count = CARRIER_LINES * CACHE_LINE_BYTES;
    if (posix_memalign((void **)&carrier.bytes, CACHE_LINE_BYTES, carrier.byte_count) != 0) {
        fprintf(stderr, "carrier allocation failed\n");
        return 1;
    }
    init_carrier(&carrier);
    uint64_t initial_digest = fnv1a64(carrier.bytes, carrier.byte_count);

    if (coherence_operator_mode) {
        int rc = run_coherence_operator_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (path_dependence_mode) {
        int rc = run_path_dependence_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (path_dual_observe_mode) {
        int rc = run_path_dual_observe_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (path_rw_observe_mode) {
        int rc = run_path_rw_observe_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (route_state_mode) {
        int rc = run_route_state_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (phase_local_pmu_mode) {
        int rc = run_phase_local_pmu_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (ibs_first_light_mode) {
        int rc = run_ibs_first_light_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (wc_flush_order_mode) {
        int rc = run_wc_flush_order_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (eviction_sentinel_mode) {
        int rc = run_eviction_sentinel_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (eviction_phase_local_mode) {
        int rc = run_eviction_phase_local_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (eviction_phase_bracketed_mode) {
        int rc = run_eviction_phase_bracketed_mode(output_root, &carrier, initial_digest, false, false);
        free(carrier.bytes);
        return rc;
    }
    if (eviction_phase_bracketed_c2d_mode) {
        int rc = run_eviction_phase_bracketed_mode(output_root, &carrier, initial_digest, true, false);
        free(carrier.bytes);
        return rc;
    }
    if (eviction_phase_bracketed_duration_mode) {
        int rc = run_eviction_phase_bracketed_mode(output_root, &carrier, initial_digest, false, true);
        free(carrier.bytes);
        return rc;
    }
    if (history_sentinel_mode) {
        int rc = run_history_sentinel_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (locked_history_mode) {
        int rc = run_locked_history_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (branch_history_mode) {
        int rc = run_branch_history_mode(output_root);
        free(carrier.bytes);
        return rc;
    }
    if (indirect_target_history_mode) {
        int rc = run_indirect_target_history_mode(output_root);
        free(carrier.bytes);
        return rc;
    }
    if (translation_history_mode) {
        int rc = run_translation_history_mode(output_root);
        free(carrier.bytes);
        return rc;
    }
    if (store_load_alias_history_mode) {
        int rc = run_store_load_alias_history_mode(output_root);
        free(carrier.bytes);
        return rc;
    }
    if (prefetch_stream_mode) {
        int rc = run_prefetch_stream_mode(output_root);
        free(carrier.bytes);
        return rc;
    }
    if (process_lifecycle_mode) {
        int rc = run_process_lifecycle_mode(output_root);
        free(carrier.bytes);
        return rc;
    }
    if (code_footprint_history_mode) {
        int rc = run_code_footprint_history_mode(output_root);
        free(carrier.bytes);
        return rc;
    }
    if (return_stack_history_mode) {
        int rc = run_return_stack_history_mode(output_root);
        free(carrier.bytes);
        return rc;
    }
    if (fp_pipeline_history_mode) {
        int rc = run_fp_pipeline_history_mode(output_root);
        free(carrier.bytes);
        return rc;
    }
    if (page_permission_history_mode) {
        int rc = run_page_permission_history_mode(output_root);
        free(carrier.bytes);
        return rc;
    }
    if (owned_recovery_history_mode) {
        int rc = run_owned_recovery_history_mode(output_root);
        free(carrier.bytes);
        return rc;
    }

    uint64_t support_ids[SUPPORT_EVENTS];
    int support_rc[SUPPORT_EVENTS];
    for (int i = 0; i < SUPPORT_EVENTS; i++) {
        support_ids[i] = 0;
        support_rc[i] = open_single_event(&support_events[i], CATCAS_CORE_A, &support_ids[i]);
    }

    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_A, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }
    int fallback_fds[MAX_GROUP_EVENTS];
    uint64_t fallback_ids[MAX_GROUP_EVENTS];
    int fallback_open_rc = open_group(fallback_group, CATCAS_CORE_A, fallback_fds, fallback_ids);
    if (fallback_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fallback_fds[i]);
    }

    const struct event_def *selected_group = NULL;
    const char *selected_group_name = NULL;
    if (primary_open_rc == 0) {
        selected_group = primary_group;
        selected_group_name = "primary_nb_coherence";
    } else if (fallback_open_rc == 0) {
        selected_group = fallback_group;
        selected_group_name = "fallback_core_cache";
    } else {
        selected_group = primary_group;
        selected_group_name = "none_opened";
    }

    const char *windows[] = {
        "idle_pause",
        "core4_read_sweep",
        "core4_write_sweep",
        "cross_core_pingpong_write"
    };
    enum { WINDOW_COUNT = 4 };
    struct group_result results[WINDOW_COUNT];
    uint64_t durations[WINDOW_COUNT];
    uint64_t digest_before[WINDOW_COUNT];
    uint64_t digest_after[WINDOW_COUNT];
    int window_rc[WINDOW_COUNT];
    for (int i = 0; i < WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_before[i] = 0;
        digest_after[i] = 0;
        if (strcmp(selected_group_name, "none_opened") == 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_window(
                selected_group,
                windows[i],
                &carrier,
                &results[i],
                &durations[i],
                &digest_before[i],
                &digest_after[i]
            );
        }
    }

    bool all_unmultiplexed = true;
    bool all_restored = true;
    bool all_windows_ok = true;
    for (int i = 0; i < WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        all_restored = all_restored && digest_after[i] == initial_digest;
    }

    uint64_t idle_c2d = event_value_by_name(&results[0], selected_group, "cache_block_commands_change_to_dirty");
    uint64_t read_c2d = event_value_by_name(&results[1], selected_group, "cache_block_commands_change_to_dirty");
    uint64_t ping_c2d = event_value_by_name(&results[3], selected_group, "cache_block_commands_change_to_dirty");
    uint64_t idle_probe = event_value_by_name(&results[0], selected_group, "probe_responses_dirty");
    uint64_t read_probe = event_value_by_name(&results[1], selected_group, "probe_responses_dirty");
    uint64_t ping_probe = event_value_by_name(&results[3], selected_group, "probe_responses_dirty");
    uint64_t c2d_control = idle_c2d > read_c2d ? idle_c2d : read_c2d;
    uint64_t probe_control = idle_probe > read_probe ? idle_probe : read_probe;
    bool c2d_moved = ping_c2d > c2d_control + 32u && ping_c2d > c2d_control * 3u;
    bool probe_moved = ping_probe > probe_control + 32u && ping_probe > probe_control * 3u;
    bool first_light = strcmp(selected_group_name, "primary_nb_coherence") == 0 &&
        all_windows_ok && all_unmultiplexed && all_restored && (c2d_moved || probe_moved);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_PMC_FIRST_LIGHT_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) {
        free(carrier.bytes);
        return 1;
    }
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        free(carrier.bytes);
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", first_light ? "F10_PMC_FIRST_LIGHT" : "F10_PMC_FIRST_LIGHT_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Family 10h PMU first-light discriminator only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"observed_core\": %d, \"partner_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier.line_count, CACHE_LINE_BYTES, carrier.byte_count, initial_digest);
    fprintf(out, "  \"event_support_matrix\": [\n");
    for (int i = 0; i < SUPPORT_EVENTS; i++) {
        fprintf(out, "    {\"event\": ");
        print_event_def(out, &support_events[i]);
        fprintf(out, ", \"open_rc\": %d, \"supported\": %s, \"id\": %" PRIu64 "}%s\n",
            support_rc[i],
            json_bool(support_rc[i] == 0),
            support_ids[i],
            i + 1 == SUPPORT_EVENTS ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"group_support\": {\n");
    fprintf(out, "    \"primary_nb_coherence\": {\"open_rc\": %d, \"supported\": %s, \"events\": ", primary_open_rc, json_bool(primary_open_rc == 0));
    print_group_defs(out, primary_group);
    fprintf(out, "},\n");
    fprintf(out, "    \"fallback_core_cache\": {\"open_rc\": %d, \"supported\": %s, \"events\": ", fallback_open_rc, json_bool(fallback_open_rc == 0));
    print_group_defs(out, fallback_group);
    fprintf(out, "}\n");
    fprintf(out, "  },\n");
    fprintf(out, "  \"selected_group\": \"%s\",\n", selected_group_name);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, selected_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"carrier_restored\": %s, \"group\": ",
            windows[i],
            window_rc[i],
            durations[i],
            digest_before[i],
            digest_after[i],
            json_bool(digest_after[i] == initial_digest));
        print_group_result(out, selected_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"predeclared_observable\": {\"primary\": \"cache_block_commands_change_to_dirty\", \"secondary\": \"probe_responses_dirty\", \"comparison\": \"cross_core_pingpong_write greater than idle_pause and core4_read_sweep controls\"},\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"carrier_restored\": %s, \"change_to_dirty_moved\": %s, \"probe_dirty_moved\": %s, \"first_light\": %s},\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(all_restored),
        json_bool(c2d_moved),
        json_bool(probe_moved),
        json_bool(first_light));
    fprintf(out, "  \"contrast_counts\": {\"change_to_dirty\": {\"idle\": %" PRIu64 ", \"read_control\": %" PRIu64 ", \"cross_core_transition\": %" PRIu64 "}, \"probe_dirty\": {\"idle\": %" PRIu64 ", \"read_control\": %" PRIu64 ", \"cross_core_transition\": %" PRIu64 "}}\n",
        idle_c2d, read_c2d, ping_c2d, idle_probe, read_probe, ping_probe);
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"%s\"}\n",
        first_light ? "F10_PMC_FIRST_LIGHT" : "F10_PMC_FIRST_LIGHT_NOT_ESTABLISHED",
        result_path,
        selected_group_name);
    free(carrier.bytes);
    return 0;
}
