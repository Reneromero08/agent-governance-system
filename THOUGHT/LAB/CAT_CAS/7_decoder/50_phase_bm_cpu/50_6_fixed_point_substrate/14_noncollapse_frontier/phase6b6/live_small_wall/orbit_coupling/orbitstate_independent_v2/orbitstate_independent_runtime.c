#define _GNU_SOURCE

#include "orbitstate_independent_runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <math.h>
#include <sched.h>
#include <stdarg.h>
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ORBITSTATE_LIVE_AUTHORITY_VALUE "orbitstate_independent_v2_0"

typedef struct {
    uint64_t value;
    uint64_t id;
} PerfGroupValue;

typedef struct {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    PerfGroupValue values[3];
} PerfGroupRead;

typedef struct {
    uint64_t cycles;
    uint64_t change_to_dirty;
    uint64_t probe_dirty;
    uint64_t duration_ns;
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t cycles_id;
    uint64_t change_to_dirty_id;
    uint64_t probe_dirty_id;
    int receiver_cpu_before;
    int receiver_cpu_after;
    size_t read_size;
} PmuMeasurement;

typedef struct {
    int cycles_fd;
    int change_to_dirty_fd;
    int probe_dirty_fd;
    uint64_t cycles_id;
    uint64_t change_to_dirty_id;
    uint64_t probe_dirty_id;
} PmuGroup;

typedef struct {
    char opaque_run_id[96];
    char opaque_group_id[96];
    int replicate;
    int public_phase_index;
    int private_phase_index;
    char physical_mapping[16];
    char condition[96];
    char response_mode[64];
    OrbitState state;
    int polarity_inversion;
    int source_off_dummy_mode;
} PrivateRecord;

typedef struct {
    char opaque_group_id[96];
    char opaque_run_id[96];
    int replicate;
    int public_phase_index;
    double public_phase_radians;
    char physical_mapping[16];
    char mapping_order[32];
    char source_order[32];
    char subcapture_order[32];
    int ordinal;
} PublicRow;

typedef struct {
    uint8_t *bank_a;
    uint8_t *bank_b;
    uint8_t *dummy_bank;
    size_t bytes;
} SharedBanks;

static void failf(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));

static void failf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    exit(2);
}

static int clamp_q(int value) {
    if (value > ORBITSTATE_QUANTIZATION_SCALE) {
        return ORBITSTATE_QUANTIZATION_SCALE;
    }
    if (value < -ORBITSTATE_QUANTIZATION_SCALE) {
        return -ORBITSTATE_QUANTIZATION_SCALE;
    }
    return value;
}

int orbitstate_round_to_i32(double value) {
    if (value >= 0.0) {
        return (int)floor(value + 0.5);
    }
    return (int)ceil(value - 0.5);
}

static double phase_for_index(int index) {
    switch (index) {
    case 0:
        return 0.0;
    case 1:
        return M_PI / 2.0;
    case 2:
        return M_PI;
    case 3:
        return 3.0 * M_PI / 2.0;
    default:
        failf("bad phase index %d", index);
    }
}

int orbitstate_compute_q(
    OrbitState state,
    int public_phase_index,
    int private_phase_index,
    const char *response_mode,
    int polarity_inversion,
    int source_off_dummy_mode
) {
    double phi = 0.0;
    double theta = phase_for_index(public_phase_index);
    double private_theta = phase_for_index(private_phase_index);
    int q_value = 0;
    if (state.modulus != ORBITSTATE_MODULUS) {
        failf("bad OrbitState modulus %" PRIu32, state.modulus);
    }
    if (source_off_dummy_mode != 0 || strcmp(response_mode, "source_off") == 0) {
        return 0;
    }
    phi = (2.0 * M_PI * (double)state.member) / (double)state.modulus;
    if (strcmp(response_mode, "query_off") == 0) {
        q_value = 0;
    } else if (strcmp(response_mode, "pre_projection") == 0) {
        q_value = orbitstate_round_to_i32((double)ORBITSTATE_QUANTIZATION_SCALE * cos(phi - private_theta));
    } else if (strcmp(response_mode, "post_projection") == 0) {
        q_value = orbitstate_round_to_i32((double)ORBITSTATE_QUANTIZATION_SCALE * cos(phi) * cos(theta));
    } else if (strcmp(response_mode, "declaration_sham") == 0) {
        q_value = orbitstate_round_to_i32((double)ORBITSTATE_QUANTIZATION_SCALE * cos(phi));
    } else if (strcmp(response_mode, "equal_orbit_odd_zero") == 0) {
        q_value = orbitstate_round_to_i32((double)ORBITSTATE_QUANTIZATION_SCALE * cos(theta));
    } else {
        failf("unknown response mode %s", response_mode);
    }
    if (polarity_inversion != 0) {
        q_value = -q_value;
    }
    return clamp_q(q_value);
}

static int current_cpu_checked(void) {
    int cpu = sched_getcpu();
    if (cpu < 0) {
        failf("sched_getcpu failed: %s", strerror(errno));
    }
    return cpu;
}

static void pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        failf("sched_setaffinity core %d failed: %s", core, strerror(errno));
    }
    if (current_cpu_checked() != core) {
        failf("core pin verification failed for core %d", core);
    }
}

static char *read_file(const char *path, size_t *out_size) {
    FILE *handle = NULL;
    long size = 0;
    char *buffer = NULL;
    size_t read_size = 0;
    handle = fopen(path, "rb");
    if (handle == NULL) {
        failf("open %s failed: %s", path, strerror(errno));
    }
    if (fseek(handle, 0, SEEK_END) != 0) {
        failf("seek %s failed", path);
    }
    size = ftell(handle);
    if (size < 0) {
        failf("tell %s failed", path);
    }
    rewind(handle);
    buffer = (char *)calloc((size_t)size + 1u, 1u);
    if (buffer == NULL) {
        failf("out of memory reading %s", path);
    }
    read_size = fread(buffer, 1u, (size_t)size, handle);
    if (read_size != (size_t)size) {
        failf("read %s failed", path);
    }
    fclose(handle);
    if (out_size != NULL) {
        *out_size = (size_t)size;
    }
    return buffer;
}

static char *find_record_block(char *json, const char *run_id) {
    char needle[160];
    char *run = NULL;
    char *start = NULL;
    int depth = 0;
    snprintf(needle, sizeof(needle), "\"opaque_run_id\":\"%s\"", run_id);
    run = strstr(json, needle);
    if (run == NULL) {
        snprintf(needle, sizeof(needle), "\"opaque_run_id\": \"%s\"", run_id);
        run = strstr(json, needle);
    }
    if (run == NULL) {
        failf("private source map missing run id %s", run_id);
    }
    start = run;
    while (start > json && *start != '{') {
        --start;
    }
    for (char *cursor = start; *cursor != '\0'; ++cursor) {
        if (*cursor == '{') {
            ++depth;
        } else if (*cursor == '}') {
            --depth;
            if (depth == 0) {
                cursor[1] = '\0';
                return start;
            }
        }
    }
    failf("unterminated private source map record");
}

static void extract_string_field(const char *block, const char *key, char *dest, size_t dest_size) {
    char needle[96];
    const char *cursor = NULL;
    const char *start = NULL;
    const char *end = NULL;
    size_t length = 0;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    cursor = strstr(block, needle);
    if (cursor == NULL) {
        failf("field %s missing", key);
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        failf("field %s missing colon", key);
    }
    start = strchr(cursor, '"');
    if (start == NULL) {
        failf("field %s missing quote", key);
    }
    ++start;
    end = strchr(start, '"');
    if (end == NULL) {
        failf("field %s unterminated", key);
    }
    length = (size_t)(end - start);
    if (length >= dest_size) {
        failf("field %s too long", key);
    }
    memcpy(dest, start, length);
    dest[length] = '\0';
}

static int extract_int_field(const char *block, const char *key) {
    char needle[96];
    const char *cursor = NULL;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    cursor = strstr(block, needle);
    if (cursor == NULL) {
        failf("int field %s missing", key);
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        failf("int field %s missing colon", key);
    }
    return (int)strtol(cursor + 1, NULL, 10);
}

static int extract_bool_field(const char *block, const char *key) {
    char needle[96];
    const char *cursor = NULL;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    cursor = strstr(block, needle);
    if (cursor == NULL) {
        failf("bool field %s missing", key);
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        failf("bool field %s missing colon", key);
    }
    while (*cursor == ':' || *cursor == ' ') {
        ++cursor;
    }
    if (strncmp(cursor, "true", 4u) == 0) {
        return 1;
    }
    if (strncmp(cursor, "false", 5u) == 0) {
        return 0;
    }
    failf("bool field %s malformed", key);
}

static PrivateRecord load_private_record(const char *private_map_path, const char *run_id) {
    size_t size = 0u;
    char *json = read_file(private_map_path, &size);
    char *block = find_record_block(json, run_id);
    PrivateRecord record;
    memset(&record, 0, sizeof(record));
    (void)size;
    extract_string_field(block, "opaque_run_id", record.opaque_run_id, sizeof(record.opaque_run_id));
    extract_string_field(block, "opaque_group_id", record.opaque_group_id, sizeof(record.opaque_group_id));
    extract_string_field(block, "physical_mapping", record.physical_mapping, sizeof(record.physical_mapping));
    extract_string_field(block, "condition", record.condition, sizeof(record.condition));
    extract_string_field(block, "response_mode", record.response_mode, sizeof(record.response_mode));
    record.replicate = extract_int_field(block, "replicate");
    record.public_phase_index = extract_int_field(block, "public_phase_index");
    record.private_phase_index = extract_int_field(block, "private_source_phase_index");
    record.state.modulus = (uint32_t)extract_int_field(block, "modulus");
    record.state.member = (uint32_t)extract_int_field(block, "member");
    record.polarity_inversion = extract_bool_field(block, "polarity_inversion");
    record.source_off_dummy_mode = extract_bool_field(block, "source_off_dummy_mode");
    free(json);
    return record;
}

static size_t affine_line(size_t index) {
    return (size_t)((257u * index + 43u) % ORBITSTATE_BANK_LINES);
}

static uint8_t pattern_byte(size_t index) {
    return (uint8_t)((index * 131u + 17u) & 0xffu);
}

static void same_value_store(volatile uint8_t *bank, int work) {
    int count = work;
    if (count < 0) {
        count = 0;
    }
    for (int i = 0; i < count; ++i) {
        size_t line = affine_line((size_t)i);
        size_t offset = line * ORBITSTATE_LINE_BYTES;
        volatile uint64_t *slot = (volatile uint64_t *)(void *)(bank + offset);
        uint64_t value = *slot;
        *slot = value;
    }
    __sync_synchronize();
}

static void same_value_store_sentinel(volatile uint8_t *bank) {
    static const uint32_t starts[4] = {0u, 1024u, 2048u, 3072u};
    for (uint32_t segment = 0u; segment < 4u; ++segment) {
        for (uint32_t offset = 0u; offset < 16u; ++offset) {
            size_t line = affine_line((size_t)starts[segment] + (size_t)offset);
            volatile uint64_t *slot = (volatile uint64_t *)(void *)(bank + line * ORBITSTATE_LINE_BYTES);
            uint64_t value = *slot;
            *slot = value;
        }
    }
    __sync_synchronize();
}

static void full_bank_touch(volatile uint8_t *bank) {
    same_value_store(bank, ORBITSTATE_TOTAL_WORK);
}

static void source_apply_encoding(
    const PrivateRecord *record,
    SharedBanks *banks,
    FILE *receipt,
    const char *component,
    const char *source_order
) {
    int source_cpu_before = current_cpu_checked();
    int q_value = orbitstate_compute_q(
        record->state,
        record->public_phase_index,
        record->private_phase_index,
        record->response_mode,
        record->polarity_inversion,
        record->source_off_dummy_mode
    );
    int positive_work = ORBITSTATE_BASE_WORK + q_value;
    int negative_work = ORBITSTATE_BASE_WORK - q_value;
    int dummy_work = 0;
    volatile uint8_t *positive_bank = banks->bank_a;
    volatile uint8_t *negative_bank = banks->bank_b;
    if (strcmp(record->physical_mapping, "map1") == 0) {
        positive_bank = banks->bank_b;
        negative_bank = banks->bank_a;
    }
    if (record->source_off_dummy_mode != 0) {
        dummy_work = ORBITSTATE_TOTAL_WORK;
        positive_work = 0;
        negative_work = 0;
        same_value_store(banks->dummy_bank, ORBITSTATE_TOTAL_WORK);
    } else if (strcmp(source_order, "neg_first") == 0) {
        same_value_store(negative_bank, negative_work);
        same_value_store(positive_bank, positive_work);
    } else {
        same_value_store(positive_bank, positive_work);
        same_value_store(negative_bank, negative_work);
    }
    int source_cpu_after = current_cpu_checked();
    if (source_cpu_before != ORBITSTATE_SOURCE_CORE || source_cpu_after != ORBITSTATE_SOURCE_CORE) {
        failf("source CPU migration");
    }
    fprintf(
        receipt,
        "{\"opaque_run_id\":\"%s\",\"component\":\"%s\",\"source_core\":%d,"
        "\"source_cpu_before\":%d,\"source_cpu_after\":%d,"
        "\"q_theta\":%d,\"positive_work\":%d,\"negative_work\":%d,"
        "\"dummy_work\":%d,\"total_work\":%d,\"source_execution_order\":\"%s\","
        "\"receiver_feedback_used_to_select_q\":false}\n",
        record->opaque_run_id,
        component,
        source_cpu_after,
        source_cpu_before,
        source_cpu_after,
        q_value,
        positive_work,
        negative_work,
        dummy_work,
        ORBITSTATE_TOTAL_WORK,
        source_order
    );
    fflush(receipt);
}

static long perf_event_open_checked(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    long fd = syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
    if (fd < 0) {
        failf("perf_event_open failed: %s", strerror(errno));
    }
    return fd;
}

static void open_counter(int *fd_out, uint64_t *id_out, uint64_t event, uint64_t umask, int group_fd, int leader) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_RAW;
    attr.size = sizeof(attr);
    attr.config = event | (umask << 8u);
    attr.disabled = leader != 0 ? 1 : 0;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING | PERF_FORMAT_ID;
    *fd_out = (int)perf_event_open_checked(&attr, 0, -1, group_fd, 0ul);
    if (ioctl(*fd_out, PERF_EVENT_IOC_ID, id_out) != 0) {
        failf("PERF_EVENT_IOC_ID failed: %s", strerror(errno));
    }
}

static PmuGroup open_pmu_group(void) {
    PmuGroup group;
    memset(&group, 0, sizeof(group));
    open_counter(&group.cycles_fd, &group.cycles_id, 0x076u, 0x00u, -1, 1);
    open_counter(&group.change_to_dirty_fd, &group.change_to_dirty_id, 0x0eau, 0x20u, group.cycles_fd, 0);
    open_counter(&group.probe_dirty_fd, &group.probe_dirty_id, 0x0ecu, 0x0cu, group.cycles_fd, 0);
    return group;
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        failf("clock_gettime failed");
    }
    return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static void close_pmu_group(PmuGroup *group) {
    close(group->probe_dirty_fd);
    close(group->change_to_dirty_fd);
    close(group->cycles_fd);
}

static void pmu_reset_enable(PmuGroup *group) {
    if (ioctl(group->cycles_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        failf("PMU reset failed");
    }
    if (ioctl(group->cycles_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        failf("PMU enable failed");
    }
}

static void pmu_disable(PmuGroup *group) {
    if (ioctl(group->cycles_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        failf("PMU disable failed");
    }
}

static uint64_t value_for_id(const PerfGroupRead *readout, uint64_t id) {
    for (uint64_t i = 0u; i < readout->nr; ++i) {
        if (readout->values[i].id == id) {
            return readout->values[i].value;
        }
    }
    failf("event ID drift");
}

static PmuMeasurement measure_store_window(PmuGroup *group, volatile uint8_t *bank, int sentinel_mode) {
    PerfGroupRead readout;
    memset(&readout, 0, sizeof(readout));
    PmuMeasurement measurement;
    memset(&measurement, 0, sizeof(measurement));
    measurement.receiver_cpu_before = current_cpu_checked();
    if (measurement.receiver_cpu_before != ORBITSTATE_RECEIVER_CORE) {
        failf("receiver CPU migration before PMU window");
    }
    uint64_t start_ns = monotonic_ns();
    pmu_reset_enable(group);
    if (sentinel_mode != 0) {
        same_value_store_sentinel(bank);
    } else {
        full_bank_touch(bank);
    }
    pmu_disable(group);
    uint64_t end_ns = monotonic_ns();
    ssize_t got = read(group->cycles_fd, &readout, sizeof(readout));
    measurement.read_size = (size_t)(got > 0 ? got : 0);
    if (got != (ssize_t)sizeof(readout)) {
        failf("perf group read failed or partial PMU read: %s", got < 0 ? strerror(errno) : "short read");
    }
    if (readout.nr != 3u) {
        failf("PMU group count mismatch");
    }
    if (readout.time_enabled == 0u) {
        failf("PMU time_enabled must be positive");
    }
    if (readout.time_enabled != readout.time_running) {
        failf("PMU multiplexing detected");
    }
    measurement.receiver_cpu_after = current_cpu_checked();
    if (measurement.receiver_cpu_after != ORBITSTATE_RECEIVER_CORE) {
        failf("receiver CPU migration after PMU window");
    }
    measurement.duration_ns = end_ns - start_ns;
    measurement.time_enabled = readout.time_enabled;
    measurement.time_running = readout.time_running;
    measurement.cycles_id = group->cycles_id;
    measurement.change_to_dirty_id = group->change_to_dirty_id;
    measurement.probe_dirty_id = group->probe_dirty_id;
    measurement.cycles = value_for_id(&readout, group->cycles_id);
    measurement.change_to_dirty = value_for_id(&readout, group->change_to_dirty_id);
    measurement.probe_dirty = value_for_id(&readout, group->probe_dirty_id);
    if (measurement.cycles == 0u) {
        failf("PMU cycles must be positive");
    }
    return measurement;
}

static PmuMeasurement measure_bank(PmuGroup *group, volatile uint8_t *bank) {
    return measure_store_window(group, bank, 0);
}

static PmuMeasurement measure_sentinel_bank(PmuGroup *group, volatile uint8_t *bank) {
    return measure_store_window(group, bank, 1);
}

static SharedBanks allocate_banks(void) {
    SharedBanks banks;
    banks.bytes = (size_t)ORBITSTATE_BANK_LINES * ORBITSTATE_LINE_BYTES;
    banks.bank_a = mmap(NULL, banks.bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    banks.bank_b = mmap(NULL, banks.bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    banks.dummy_bank = mmap(NULL, banks.bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (banks.bank_a == MAP_FAILED || banks.bank_b == MAP_FAILED || banks.dummy_bank == MAP_FAILED) {
        failf("mmap banks failed: %s", strerror(errno));
    }
    for (size_t index = 0u; index < banks.bytes; ++index) {
        uint8_t value = pattern_byte(index);
        banks.bank_a[index] = value;
        banks.bank_b[index] = value;
        banks.dummy_bank[index] = (uint8_t)(value ^ ORBITSTATE_DUMMY_BANK_INITIAL_VALUE);
    }
    __sync_synchronize();
    return banks;
}

static void free_banks(SharedBanks *banks) {
    munmap(banks->bank_a, banks->bytes);
    munmap(banks->bank_b, banks->bytes);
    munmap(banks->dummy_bank, banks->bytes);
}

static uint64_t bank_digest(volatile uint8_t *bank, size_t bytes) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0u; i < bytes; ++i) {
        hash ^= (uint64_t)bank[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint64_t combined_bank_digest(SharedBanks *banks) {
    uint64_t a = bank_digest(banks->bank_a, banks->bytes);
    uint64_t b = bank_digest(banks->bank_b, banks->bytes);
    return a ^ (b + 0x9e3779b97f4a7c15ull + (a << 6u) + (a >> 2u));
}

static void restore_pattern_bank(uint8_t *bank, size_t bytes, int dummy) {
    for (size_t i = 0u; i < bytes; ++i) {
        uint8_t value = pattern_byte(i);
        bank[i] = dummy != 0 ? (uint8_t)(value ^ ORBITSTATE_DUMMY_BANK_INITIAL_VALUE) : value;
    }
    __sync_synchronize();
}

static void restore_all_banks(SharedBanks *banks) {
    restore_pattern_bank(banks->bank_a, banks->bytes, 0);
    restore_pattern_bank(banks->bank_b, banks->bytes, 0);
    restore_pattern_bank(banks->dummy_bank, banks->bytes, 1);
}

static void source_child_loop(
    int read_fd,
    int write_fd,
    const char *private_map_path,
    SharedBanks *banks,
    const char *source_receipts_path
) {
    FILE *commands = NULL;
    FILE *responses = NULL;
    FILE *receipt = NULL;
    char run_id[96];
    char component[32];
    char source_order[32];
    pin_to_core(ORBITSTATE_SOURCE_CORE);
    commands = fdopen(read_fd, "r");
    responses = fdopen(write_fd, "w");
    receipt = fopen(source_receipts_path, "a");
    if (commands == NULL || responses == NULL || receipt == NULL) {
        failf("source child stream setup failed");
    }
    while (fscanf(commands, "%95s %31s %31s", run_id, component, source_order) == 3) {
        PrivateRecord record = load_private_record(private_map_path, run_id);
        source_apply_encoding(&record, banks, receipt, component, source_order);
        fprintf(responses, "%s %s done\n", run_id, component);
        fflush(responses);
    }
    fclose(receipt);
    fclose(commands);
    fclose(responses);
    _exit(0);
}

static void emit_stage_receipt(
    FILE *stage,
    const PublicRow *row,
    const char *component,
    const char *stage_name,
    int sequence_index,
    SharedBanks *banks,
    int ok
) {
    uint64_t digest_value = combined_bank_digest(banks);
    uint64_t digest_a = bank_digest(banks->bank_a, banks->bytes);
    uint64_t digest_b = bank_digest(banks->bank_b, banks->bytes);
    fprintf(
        stage,
        "{\"opaque_run_id\":\"%s\",\"component\":\"%s\",\"stage\":\"%s\","
        "\"sequence_index\":%d,\"receiver_core\":%d,\"byte_digest\":\"%016" PRIx64 "\","
        "\"physical_a_digest\":\"%016" PRIx64 "\",\"physical_b_digest\":\"%016" PRIx64 "\","
        "\"ok\":%s}\n",
        row->opaque_run_id,
        component,
        stage_name,
        sequence_index,
        ORBITSTATE_RECEIVER_CORE,
        digest_value,
        digest_a,
        digest_b,
        ok != 0 ? "true" : "false"
    );
}

static int parse_public_row(const char *line, PublicRow *row) {
    memset(row, 0, sizeof(*row));
    return sscanf(
        line,
        "%95[^\t]\t%95[^\t]\t%d\t%d\t%lf\t%15[^\t]\t%31[^\t]\t%31[^\t]\t%31[^\t]\t%d",
        row->opaque_group_id,
        row->opaque_run_id,
        &row->replicate,
        &row->public_phase_index,
        &row->public_phase_radians,
        row->physical_mapping,
        row->mapping_order,
        row->source_order,
        row->subcapture_order,
        &row->ordinal
    );
}

static volatile uint8_t *measured_bank_for(const PublicRow *row, const char *component, SharedBanks *banks) {
    if ((strcmp(row->physical_mapping, "map0") == 0 && strcmp(component, "negative") == 0) ||
        (strcmp(row->physical_mapping, "map1") == 0 && strcmp(component, "positive") == 0)) {
        return banks->bank_b;
    }
    return banks->bank_a;
}

static void execute_component_window(
    const PublicRow *row,
    const char *component,
    SharedBanks *banks,
    PmuGroup *pmu,
    FILE *raw,
    FILE *sentinels,
    FILE *stage,
    FILE *child_commands,
    FILE *child_responses
) {
    volatile uint8_t *bank = measured_bank_for(row, component, banks);
    uint64_t baseline_digest = 0u;
    uint64_t pre_digest = 0u;
    uint64_t post_digest = 0u;
    PmuMeasurement pre_a;
    PmuMeasurement pre_b;
    PmuMeasurement post_a;
    PmuMeasurement post_b;
    char done_run[96];
    char done_component[32];
    char done_word[32];

    full_bank_touch(banks->bank_a);
    full_bank_touch(banks->bank_b);
    baseline_digest = combined_bank_digest(banks);
    emit_stage_receipt(stage, row, component, "receiver_full_baseline", 0, banks, 1);

    pre_a = measure_sentinel_bank(pmu, banks->bank_a);
    pre_b = measure_sentinel_bank(pmu, banks->bank_b);
    pre_digest = combined_bank_digest(banks);
    emit_stage_receipt(stage, row, component, "pre_sentinels", 1, banks, pre_digest == baseline_digest);

    full_bank_touch(banks->bank_a);
    full_bank_touch(banks->bank_b);
    emit_stage_receipt(stage, row, component, "receiver_rebaseline", 2, banks, combined_bank_digest(banks) == baseline_digest);

    fprintf(child_commands, "%s %s %s\n", row->opaque_run_id, component, row->source_order);
    fflush(child_commands);
    if (fscanf(child_responses, "%95s %31s %31s", done_run, done_component, done_word) != 3) {
        failf("source child completion missing");
    }
    if (strcmp(done_run, row->opaque_run_id) != 0 || strcmp(done_component, component) != 0 ||
        strcmp(done_word, "done") != 0) {
        failf("source child completion mismatch");
    }
    emit_stage_receipt(stage, row, component, "source_execute", 3, banks, 1);

    PmuMeasurement measurement = measure_bank(pmu, bank);
    emit_stage_receipt(stage, row, component, "receiver_measure", 4, banks, 1);

    restore_all_banks(banks);
    full_bank_touch(banks->bank_a);
    full_bank_touch(banks->bank_b);
    emit_stage_receipt(stage, row, component, "receiver_restoration", 5, banks, combined_bank_digest(banks) == baseline_digest);

    post_a = measure_sentinel_bank(pmu, banks->bank_a);
    post_b = measure_sentinel_bank(pmu, banks->bank_b);
    post_digest = combined_bank_digest(banks);
    emit_stage_receipt(stage, row, component, "post_sentinels", 6, banks, post_digest == baseline_digest);

    fprintf(
        raw,
        "{\"opaque_group_id\":\"%s\",\"opaque_run_id\":\"%s\",\"replicate\":%d,"
        "\"public_decoder_phase_index\":%d,\"physical_mapping\":\"%s\","
        "\"component\":\"%s\",\"measured_bank\":\"%s\",\"receiver_core\":%d,"
        "\"receiver_cpu_before\":%d,\"receiver_cpu_after\":%d,"
        "\"pmu_unmultiplexed\":%s,\"byte_compare_ok\":%s,\"event_ids_valid\":true,"
        "\"counters\":{\"change_to_dirty\":%" PRIu64 ",\"cycles\":%" PRIu64 ","
        "\"probe_dirty\":%" PRIu64 ",\"duration_ns\":%" PRIu64 ","
        "\"time_enabled\":%" PRIu64 ",\"time_running\":%" PRIu64 ","
        "\"cycles_id\":%" PRIu64 ",\"change_to_dirty_id\":%" PRIu64 ","
        "\"probe_dirty_id\":%" PRIu64 ",\"pmu_read_size\":%zu}}\n",
        row->opaque_group_id,
        row->opaque_run_id,
        row->replicate,
        row->public_phase_index,
        row->physical_mapping,
        component,
        bank == banks->bank_a ? "A" : "B",
        ORBITSTATE_RECEIVER_CORE,
        measurement.receiver_cpu_before,
        measurement.receiver_cpu_after,
        measurement.time_enabled == measurement.time_running ? "true" : "false",
        post_digest == baseline_digest ? "true" : "false",
        measurement.change_to_dirty,
        measurement.cycles,
        measurement.probe_dirty,
        measurement.duration_ns,
        measurement.time_enabled,
        measurement.time_running,
        measurement.cycles_id,
        measurement.change_to_dirty_id,
        measurement.probe_dirty_id,
        measurement.read_size
    );
    fprintf(
        sentinels,
        "{\"opaque_run_id\":\"%s\",\"replicate\":%d,\"public_decoder_phase_index\":%d,"
        "\"physical_mapping\":\"%s\",\"component\":\"%s\",\"receiver_core\":%d,"
        "\"pre_digest\":\"%016" PRIx64 "\",\"post_digest\":\"%016" PRIx64 "\","
        "\"pre_a_cycles\":%" PRIu64 ",\"pre_b_cycles\":%" PRIu64 ","
        "\"post_a_cycles\":%" PRIu64 ",\"post_b_cycles\":%" PRIu64 ","
        "\"pre_ok\":%s,\"post_ok\":%s}\n",
        row->opaque_run_id,
        row->replicate,
        row->public_phase_index,
        row->physical_mapping,
        component,
        ORBITSTATE_RECEIVER_CORE,
        pre_digest,
        post_digest,
        pre_a.cycles,
        pre_b.cycles,
        post_a.cycles,
        post_b.cycles,
        pre_digest == baseline_digest ? "true" : "false",
        post_digest == baseline_digest ? "true" : "false"
    );
}

static void execute_mapping_row(
    const PublicRow *row,
    SharedBanks *banks,
    PmuGroup *pmu,
    FILE *raw,
    FILE *sentinels,
    FILE *stage,
    FILE *child_commands,
    FILE *child_responses
) {
    const char *components[2] = {"positive", "negative"};
    if (strcmp(row->subcapture_order, "neg_then_pos") == 0) {
        components[0] = "negative";
        components[1] = "positive";
    }
    for (int component_index = 0; component_index < 2; ++component_index) {
        execute_component_window(row, components[component_index], banks, pmu, raw, sentinels, stage, child_commands, child_responses);
    }
}

static int rows_form_pair(const PublicRow *a, const PublicRow *b) {
    int one_map0 = strcmp(a->physical_mapping, "map0") == 0 || strcmp(b->physical_mapping, "map0") == 0;
    int one_map1 = strcmp(a->physical_mapping, "map1") == 0 || strcmp(b->physical_mapping, "map1") == 0;
    return strcmp(a->opaque_group_id, b->opaque_group_id) == 0 &&
           a->replicate == b->replicate &&
           a->public_phase_index == b->public_phase_index &&
           one_map0 && one_map1 &&
           strcmp(a->physical_mapping, b->physical_mapping) != 0;
}

static void execute_mapping_pair(
    const PublicRow *first,
    const PublicRow *second,
    PmuGroup *pmu,
    const char *private_map_path,
    const char *source_receipts_path,
    FILE *raw,
    FILE *sentinels,
    FILE *stage
) {
    int to_child[2];
    int from_child[2];
    pid_t child = -1;
    FILE *child_commands = NULL;
    FILE *child_responses = NULL;
    SharedBanks banks = allocate_banks();
    if (pipe(to_child) != 0 || pipe(from_child) != 0) {
        failf("pipe failed");
    }
    child = fork();
    if (child < 0) {
        failf("fork failed");
    }
    if (child == 0) {
        close(to_child[1]);
        close(from_child[0]);
        source_child_loop(to_child[0], from_child[1], private_map_path, &banks, source_receipts_path);
    }
    close(to_child[0]);
    close(from_child[1]);
    child_commands = fdopen(to_child[1], "w");
    child_responses = fdopen(from_child[0], "r");
    if (child_commands == NULL || child_responses == NULL) {
        failf("receiver source pipe setup failed");
    }
    execute_mapping_row(first, &banks, pmu, raw, sentinels, stage, child_commands, child_responses);
    execute_mapping_row(second, &banks, pmu, raw, sentinels, stage, child_commands, child_responses);
    fclose(child_commands);
    fclose(child_responses);
    int status = 0;
    if (waitpid(child, &status, 0) < 0) {
        failf("waitpid failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        failf("source child failed");
    }
    free_banks(&banks);
}

static void receiver_execute_rows(
    const char *schedule_tsv,
    const char *private_map_path,
    const char *output_root,
    int replicate
) {
    char raw_path[4096];
    char sentinels_path[4096];
    char stage_path[4096];
    char source_path[4096];
    FILE *schedule = NULL;
    FILE *raw = NULL;
    FILE *sentinels = NULL;
    FILE *stage = NULL;
    PmuGroup pmu;
    char header[1024];
    char line[2048];
    PublicRow rows[72];
    int used[72];
    int row_count = 0;
    int pair_count = 0;
    pin_to_core(ORBITSTATE_RECEIVER_CORE);
    snprintf(raw_path, sizeof(raw_path), "%s/RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl", output_root);
    snprintf(sentinels_path, sizeof(sentinels_path), "%s/ORBITSTATE_RECEIVER_SENTINELS.jsonl", output_root);
    snprintf(stage_path, sizeof(stage_path), "%s/ORBITSTATE_STAGE_RECEIPTS.jsonl", output_root);
    snprintf(source_path, sizeof(source_path), "%s/ORBITSTATE_SOURCE_RECEIPTS.jsonl", output_root);
    schedule = fopen(schedule_tsv, "r");
    raw = fopen(raw_path, "a");
    sentinels = fopen(sentinels_path, "a");
    stage = fopen(stage_path, "a");
    if (schedule == NULL || raw == NULL || sentinels == NULL || stage == NULL) {
        failf("receiver output setup failed");
    }
    if (fgets(header, sizeof(header), schedule) == NULL) {
        failf("empty schedule");
    }
    while (fgets(line, sizeof(line), schedule) != NULL) {
        PublicRow row;
        int matched = parse_public_row(line, &row);
        if (matched != 10) {
            failf("bad schedule row: %s", line);
        }
        if (row.replicate == replicate) {
            if (row_count >= 72) {
                failf("too many replicate rows");
            }
            rows[row_count++] = row;
        }
    }
    fclose(schedule);
    if (row_count != 72) {
        failf("bad replicate row count %d", row_count);
    }
    memset(used, 0, sizeof(used));
    pmu = open_pmu_group();
    for (int index = 0; index < row_count; ++index) {
        int mate = -1;
        if (used[index] != 0) {
            continue;
        }
        for (int candidate = index + 1; candidate < row_count; ++candidate) {
            if (used[candidate] == 0 && rows_form_pair(&rows[index], &rows[candidate])) {
                mate = candidate;
                break;
            }
        }
        if (mate < 0) {
            failf("mapping pair allocation drift for %s", rows[index].opaque_run_id);
        }
        execute_mapping_pair(&rows[index], &rows[mate], &pmu, private_map_path, source_path, raw, sentinels, stage);
        used[index] = 1;
        used[mate] = 1;
        ++pair_count;
    }
    if (pair_count != 36) {
        failf("bad mapping pair count %d", pair_count);
    }
    close_pmu_group(&pmu);
    fclose(raw);
    fclose(sentinels);
    fclose(stage);
}

static int validate_schedule_tsv(const char *path) {
    FILE *handle = fopen(path, "r");
    char header[1024];
    char line[2048];
    int rows = 0;
    int replicate_rows[2] = {0, 0};
    PublicRow all_rows[144];
    int used[144];
    int pair_count = 0;
    if (handle == NULL) {
        failf("open schedule %s failed: %s", path, strerror(errno));
    }
    if (fgets(header, sizeof(header), handle) == NULL) {
        failf("schedule missing header");
    }
    if (strstr(header, "condition") != NULL || strstr(header, "member") != NULL ||
        strstr(header, "q_theta") != NULL || strstr(header, "positive_work") != NULL ||
        strstr(header, "negative_work") != NULL || strstr(header, "private_source_phase") != NULL) {
        failf("public schedule header contains private fields");
    }
    while (fgets(line, sizeof(line), handle) != NULL) {
        PublicRow row;
        int matched = parse_public_row(line, &row);
        if (matched != 10) {
            failf("bad tsv row");
        }
        if (row.replicate < 0 || row.replicate > 1) {
            failf("bad replicate");
        }
        if (row.public_phase_index < 0 || row.public_phase_index > 3) {
            failf("bad phase");
        }
        if (strcmp(row.physical_mapping, "map0") != 0 && strcmp(row.physical_mapping, "map1") != 0) {
            failf("bad mapping");
        }
        if (rows >= 144) {
            failf("too many schedule rows");
        }
        all_rows[rows] = row;
        ++rows;
        ++replicate_rows[row.replicate];
    }
    fclose(handle);
    if (rows != 144 || replicate_rows[0] != 72 || replicate_rows[1] != 72) {
        failf("bad schedule geometry rows=%d rep0=%d rep1=%d", rows, replicate_rows[0], replicate_rows[1]);
    }
    memset(used, 0, sizeof(used));
    for (int index = 0; index < rows; ++index) {
        int mate = -1;
        if (used[index] != 0) {
            continue;
        }
        for (int candidate = index + 1; candidate < rows; ++candidate) {
            if (used[candidate] == 0 && rows_form_pair(&all_rows[index], &all_rows[candidate])) {
                mate = candidate;
                break;
            }
        }
        if (mate < 0) {
            failf("missing mapping pair");
        }
        used[index] = 1;
        used[mate] = 1;
        ++pair_count;
    }
    if (pair_count != 72) {
        failf("bad total mapping pair count %d", pair_count);
    }
    printf("{\"status\":\"ORBITSTATE_PUBLIC_SCHEDULE_TSV_OK\",\"rows\":%d,\"replicate_0\":%d,\"replicate_1\":%d,"
           "\"mapping_pair_allocation\":\"grouped_two_leg_pairs\"}\n",
           rows,
           replicate_rows[0],
           replicate_rows[1]);
    return 0;
}

static int affine_self_test(void) {
    unsigned char seen[ORBITSTATE_BANK_LINES];
    memset(seen, 0, sizeof(seen));
    for (size_t index = 0u; index < ORBITSTATE_BANK_LINES; ++index) {
        size_t line = affine_line(index);
        if (line >= ORBITSTATE_BANK_LINES || seen[line] != 0u) {
            return 0;
        }
        seen[line] = 1u;
    }
    return affine_line(0u) == 43u && affine_line(1u) == 300u;
}

static int self_test(void) {
    OrbitState d = {ORBITSTATE_MODULUS, ORBITSTATE_D_MEMBER};
    OrbitState fold = {ORBITSTATE_MODULUS, ORBITSTATE_FOLD_MEMBER};
    int q_d0 = orbitstate_compute_q(d, 0, 0, "pre_projection", 0, 0);
    int q_fold0 = orbitstate_compute_q(fold, 0, 0, "pre_projection", 0, 0);
    int q_d1 = orbitstate_compute_q(d, 1, 1, "pre_projection", 0, 0);
    int q_inv = orbitstate_compute_q(d, 0, 0, "pre_projection", 1, 0);
    int q_off = orbitstate_compute_q(d, 2, 2, "source_off", 0, 1);
    if (q_d0 != q_fold0 || q_d0 < 1200) {
        failf("d/fold real component self-test failed");
    }
    if (q_d1 < 700 || q_inv != -q_d0 || q_off != 0) {
        failf("OrbitState formula self-test failed");
    }
    if (!affine_self_test()) {
        failf("affine line permutation self-test failed");
    }
    printf(
        "{\"status\":\"ORBITSTATE_RUNTIME_SELF_TEST_OK\",\"d_q0\":%d,\"fold_q0\":%d,"
        "\"d_q1\":%d,\"polarity_q0\":%d,\"source_off_q\":%d,"
        "\"affine_line_permutation\":\"line(i)=(257*i+43) mod 4096\","
        "\"bank_initialization\":\"physical A and B identical\","
        "\"process_boundary\":\"source_child_opens_private_map_after_fork\"}\n",
        q_d0,
        q_fold0,
        q_d1,
        q_inv,
        q_off
    );
    return 0;
}

static int pmu_preflight(void) {
    pin_to_core(ORBITSTATE_RECEIVER_CORE);
    SharedBanks banks = allocate_banks();
    PmuGroup group = open_pmu_group();
    PmuMeasurement measurement = measure_bank(&group, banks.bank_a);
    close_pmu_group(&group);
    free_banks(&banks);
    printf(
        "{\"status\":\"ORBITSTATE_RUNTIME_PMU_PREFLIGHT_OK\","
        "\"scientific_classification_emitted\":false,"
        "\"pid\":0,\"cpu\":-1,\"exclude_kernel\":1,\"exclude_hv\":1,"
        "\"group_count\":3,\"receiver_cpu_before\":%d,\"receiver_cpu_after\":%d,"
        "\"time_enabled\":%" PRIu64 ",\"time_running\":%" PRIu64 ","
        "\"cycles\":%" PRIu64 ",\"change_to_dirty\":%" PRIu64 ",\"probe_dirty\":%" PRIu64 ","
        "\"cycles_id\":%" PRIu64 ",\"change_to_dirty_id\":%" PRIu64 ",\"probe_dirty_id\":%" PRIu64 ","
        "\"duration_ns\":%" PRIu64 ",\"read_size\":%zu}\n",
        measurement.receiver_cpu_before,
        measurement.receiver_cpu_after,
        measurement.time_enabled,
        measurement.time_running,
        measurement.cycles,
        measurement.change_to_dirty,
        measurement.probe_dirty,
        measurement.cycles_id,
        measurement.change_to_dirty_id,
        measurement.probe_dirty_id,
        measurement.duration_ns,
        measurement.read_size
    );
    return 0;
}

static void usage(const char *argv0) {
    fprintf(
        stderr,
        "usage: %s --self-test | --pmu-preflight | --validate-schedule-tsv PATH | "
        "--run-schedule --schedule-tsv PATH --private-map PATH --output-root PATH --replicate N\n",
        argv0
    );
}

static void require_runtime_live_authority(void) {
    const char *commit_binding = getenv("ORBITSTATE_INDEPENDENT_V2_COMMIT_BINDING");
    const char *manifest_binding = getenv("ORBITSTATE_INDEPENDENT_V2_MANIFEST_SHA256");
    const char *live_authority = getenv("ORBITSTATE_INDEPENDENT_V2_LIVE_AUTHORITY");
    if (commit_binding == NULL || strlen(commit_binding) != 40u) {
        failf("runtime missing commit authority");
    }
    if (manifest_binding == NULL || strlen(manifest_binding) != 64u) {
        failf("runtime missing manifest authority");
    }
    if (live_authority == NULL || strcmp(live_authority, ORBITSTATE_LIVE_AUTHORITY_VALUE) != 0) {
        failf("runtime missing live authority");
    }
}

int main(int argc, char **argv) {
    const char *schedule_tsv = NULL;
    const char *private_map = NULL;
    const char *output_root = NULL;
    int replicate = -1;
    int run_schedule = 0;
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--self-test") == 0) {
            return self_test();
        }
        if (strcmp(argv[i], "--pmu-preflight") == 0) {
            require_runtime_live_authority();
            return pmu_preflight();
        }
        if (strcmp(argv[i], "--validate-schedule-tsv") == 0 && i + 1 < argc) {
            return validate_schedule_tsv(argv[++i]);
        }
        if (strcmp(argv[i], "--run-schedule") == 0) {
            run_schedule = 1;
            continue;
        }
        if (strcmp(argv[i], "--schedule-tsv") == 0 && i + 1 < argc) {
            schedule_tsv = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--private-map") == 0 && i + 1 < argc) {
            private_map = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--output-root") == 0 && i + 1 < argc) {
            output_root = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--replicate") == 0 && i + 1 < argc) {
            replicate = (int)strtol(argv[++i], NULL, 10);
            continue;
        }
        usage(argv[0]);
        return 2;
    }
    if (run_schedule != 0) {
        if (schedule_tsv == NULL || private_map == NULL || output_root == NULL || replicate < 0 || replicate > 1) {
            usage(argv[0]);
            return 2;
        }
        require_runtime_live_authority();
        receiver_execute_rows(schedule_tsv, private_map, output_root, replicate);
        printf("{\"status\":\"ORBITSTATE_RUNTIME_REPLICATE_COMPLETE\",\"replicate\":%d,"
               "\"scientific_classification_emitted\":false}\n",
               replicate);
        return 0;
    }
    usage(argv[0]);
    return 2;
}
