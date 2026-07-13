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
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t id;
} PerfRead;

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

static void pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        failf("sched_setaffinity core %d failed: %s", core, strerror(errno));
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

static void same_value_store(volatile uint8_t *bank, int work) {
    int count = work;
    if (count < 0) {
        count = 0;
    }
    for (int i = 0; i < count; ++i) {
        size_t offset = ((size_t)i % ORBITSTATE_BANK_LINES) * ORBITSTATE_LINE_BYTES;
        bank[offset] = bank[offset];
    }
}

static void source_apply_encoding(
    const PrivateRecord *record,
    SharedBanks *banks,
    FILE *receipt,
    const char *component,
    const char *source_order
) {
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
        same_value_store(banks->dummy_bank, dummy_work);
    } else if (strcmp(source_order, "neg_first") == 0) {
        same_value_store(negative_bank, negative_work);
        same_value_store(positive_bank, positive_work);
    } else {
        same_value_store(positive_bank, positive_work);
        same_value_store(negative_bank, negative_work);
    }
    fprintf(
        receipt,
        "{\"opaque_run_id\":\"%s\",\"component\":\"%s\",\"source_core\":%d,"
        "\"q_theta\":%d,\"positive_work\":%d,\"negative_work\":%d,"
        "\"dummy_work\":%d,\"total_work\":%d,\"source_execution_order\":\"%s\","
        "\"receiver_feedback_used_to_select_q\":false}\n",
        record->opaque_run_id,
        component,
        ORBITSTATE_SOURCE_CORE,
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

static void open_counter(int *fd_out, uint64_t *id_out, uint64_t event, uint64_t umask, int group_fd) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_RAW;
    attr.size = sizeof(attr);
    attr.config = event | (umask << 8u);
    attr.disabled = 1;
    attr.exclude_kernel = 0;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING | PERF_FORMAT_ID;
    *fd_out = (int)perf_event_open_checked(&attr, -1, ORBITSTATE_RECEIVER_CORE, group_fd, 0ul);
    if (ioctl(*fd_out, PERF_EVENT_IOC_ID, id_out) != 0) {
        failf("PERF_EVENT_IOC_ID failed: %s", strerror(errno));
    }
}

static PmuGroup open_pmu_group(void) {
    PmuGroup group;
    memset(&group, 0, sizeof(group));
    open_counter(&group.cycles_fd, &group.cycles_id, 0x076u, 0x00u, -1);
    open_counter(&group.change_to_dirty_fd, &group.change_to_dirty_id, 0x0eau, 0x20u, group.cycles_fd);
    open_counter(&group.probe_dirty_fd, &group.probe_dirty_id, 0x0ecu, 0x0cu, group.cycles_fd);
    return group;
}

static PerfRead read_counter(int fd) {
    PerfRead result;
    ssize_t got = read(fd, &result, sizeof(result));
    if (got != (ssize_t)sizeof(result)) {
        failf("perf read failed: %s", strerror(errno));
    }
    return result;
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

static uint64_t measure_bank(PmuGroup *group, volatile uint8_t *bank) {
    PerfRead ctd;
    pmu_reset_enable(group);
    for (size_t i = 0u; i < ORBITSTATE_BANK_LINES; ++i) {
        size_t offset = i * ORBITSTATE_LINE_BYTES;
        bank[offset] = bank[offset];
    }
    pmu_disable(group);
    (void)read_counter(group->cycles_fd);
    ctd = read_counter(group->change_to_dirty_fd);
    (void)read_counter(group->probe_dirty_fd);
    if (ctd.time_enabled != ctd.time_running) {
        failf("PMU multiplexing detected");
    }
    return ctd.value;
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
    memset(banks.bank_a, 0x5au, banks.bytes);
    memset(banks.bank_b, 0xa5u, banks.bytes);
    memset(banks.dummy_bank, 0x3cu, banks.bytes);
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

static void restore_bank(uint8_t *bank, size_t bytes, uint8_t value) {
    for (size_t i = 0u; i < bytes; ++i) {
        bank[i] = value;
    }
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
    const char *run_id,
    const char *component,
    const char *stage_name,
    int sequence_index,
    uint64_t digest_value,
    int ok
) {
    fprintf(
        stage,
        "{\"opaque_run_id\":\"%s\",\"component\":\"%s\",\"stage\":\"%s\","
        "\"sequence_index\":%d,\"receiver_core\":%d,\"byte_digest\":\"%016" PRIx64 "\",\"ok\":%s}\n",
        run_id,
        component,
        stage_name,
        sequence_index,
        ORBITSTATE_RECEIVER_CORE,
        digest_value,
        ok != 0 ? "true" : "false"
    );
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
    FILE *child_responses = NULL;
    int to_child[2];
    int from_child[2];
    pid_t child = -1;
    SharedBanks banks = allocate_banks();
    PmuGroup pmu;
    char header[1024];
    char line[2048];
    pin_to_core(ORBITSTATE_RECEIVER_CORE);
    snprintf(raw_path, sizeof(raw_path), "%s/RAW_ORBITSTATE_RECEIVER_CAPTURE.jsonl", output_root);
    snprintf(sentinels_path, sizeof(sentinels_path), "%s/ORBITSTATE_RECEIVER_SENTINELS.jsonl", output_root);
    snprintf(stage_path, sizeof(stage_path), "%s/ORBITSTATE_STAGE_RECEIPTS.jsonl", output_root);
    snprintf(source_path, sizeof(source_path), "%s/ORBITSTATE_SOURCE_RECEIPTS.jsonl", output_root);
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
        source_child_loop(to_child[0], from_child[1], private_map_path, &banks, source_path);
    }
    close(to_child[0]);
    close(from_child[1]);
    child_responses = fdopen(from_child[0], "r");
    if (child_responses == NULL) {
        failf("receiver completion stream setup failed");
    }
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
    pmu = open_pmu_group();
    while (fgets(line, sizeof(line), schedule) != NULL) {
        char group_id[96];
        char run_id[96];
        char mapping[16];
        char mapping_order[32];
        char source_order[32];
        char subcapture_order[32];
        int row_replicate = -1;
        int phase_index = -1;
        int ordinal = -1;
        double radians = 0.0;
        int matched = sscanf(
            line,
            "%95[^\t]\t%95[^\t]\t%d\t%d\t%lf\t%15[^\t]\t%31[^\t]\t%31[^\t]\t%31[^\t]\t%d",
            group_id,
            run_id,
            &row_replicate,
            &phase_index,
            &radians,
            mapping,
            mapping_order,
            source_order,
            subcapture_order,
            &ordinal
        );
        (void)radians;
        (void)mapping_order;
        (void)source_order;
        (void)subcapture_order;
        (void)ordinal;
        if (matched != 10) {
            failf("bad schedule row: %s", line);
        }
        if (row_replicate != replicate) {
            continue;
        }
        const char *components[2] = {"positive", "negative"};
        if (strcmp(subcapture_order, "neg_then_pos") == 0) {
            components[0] = "negative";
            components[1] = "positive";
        }
        for (int component_index = 0; component_index < 2; ++component_index) {
            const char *component = components[component_index];
            volatile uint8_t *bank = banks.bank_a;
            uint64_t value = 0u;
            uint64_t baseline_digest = 0u;
            uint64_t pre_digest = 0u;
            uint64_t rebaseline_digest = 0u;
            uint64_t source_digest = 0u;
            uint64_t measure_digest = 0u;
            uint64_t restored_digest = 0u;
            uint64_t post_digest = 0u;
            int bank_is_a = 1;
            int restoration_ok = 0;
            char done_run[96];
            char done_component[32];
            char done_word[32];
            if ((strcmp(mapping, "map0") == 0 && strcmp(component, "negative") == 0) ||
                (strcmp(mapping, "map1") == 0 && strcmp(component, "positive") == 0)) {
                bank = banks.bank_b;
            }
            bank_is_a = bank == (volatile uint8_t *)banks.bank_a;
            baseline_digest = bank_digest(bank, banks.bytes);
            emit_stage_receipt(stage, run_id, component, "receiver_full_baseline", 0, baseline_digest, 1);
            same_value_store(bank, 64);
            pre_digest = bank_digest(bank, banks.bytes);
            emit_stage_receipt(stage, run_id, component, "pre_sentinels", 1, pre_digest, pre_digest == baseline_digest);
            same_value_store(bank, 64);
            rebaseline_digest = bank_digest(bank, banks.bytes);
            emit_stage_receipt(stage, run_id, component, "receiver_rebaseline", 2, rebaseline_digest, rebaseline_digest == baseline_digest);
            dprintf(to_child[1], "%s %s %s\n", run_id, component, source_order);
            if (fscanf(child_responses, "%95s %31s %31s", done_run, done_component, done_word) != 3) {
                failf("source child completion missing");
            }
            if (strcmp(done_run, run_id) != 0 || strcmp(done_component, component) != 0 ||
                strcmp(done_word, "done") != 0) {
                failf("source child completion mismatch");
            }
            source_digest = bank_digest(bank, banks.bytes);
            emit_stage_receipt(stage, run_id, component, "source_execute", 3, source_digest, 1);
            value = measure_bank(&pmu, bank);
            measure_digest = bank_digest(bank, banks.bytes);
            emit_stage_receipt(stage, run_id, component, "receiver_measure", 4, measure_digest, 1);
            restore_bank((uint8_t *)bank, banks.bytes, bank_is_a != 0 ? 0x5au : 0xa5u);
            restored_digest = bank_digest(bank, banks.bytes);
            restoration_ok = restored_digest == baseline_digest;
            emit_stage_receipt(stage, run_id, component, "receiver_restoration", 5, restored_digest, restoration_ok);
            post_digest = bank_digest(bank, banks.bytes);
            emit_stage_receipt(stage, run_id, component, "post_sentinels", 6, post_digest, post_digest == baseline_digest);
            fprintf(
                raw,
                "{\"opaque_group_id\":\"%s\",\"opaque_run_id\":\"%s\",\"replicate\":%d,"
                "\"public_decoder_phase_index\":%d,\"physical_mapping\":\"%s\","
                "\"component\":\"%s\",\"measured_bank\":\"%s\",\"receiver_core\":%d,"
                "\"pmu_unmultiplexed\":true,\"byte_compare_ok\":%s,\"event_ids_valid\":true,"
                "\"counters\":{\"change_to_dirty\":%" PRIu64 ",\"cycles\":0}}\n",
                group_id,
                run_id,
                row_replicate,
                phase_index,
                mapping,
                component,
                bank == banks.bank_a ? "A" : "B",
                ORBITSTATE_RECEIVER_CORE,
                restoration_ok != 0 ? "true" : "false",
                value
            );
            fprintf(
                sentinels,
                "{\"opaque_run_id\":\"%s\",\"replicate\":%d,\"public_decoder_phase_index\":%d,"
                "\"physical_mapping\":\"%s\",\"component\":\"%s\",\"receiver_core\":%d,"
                "\"pre_digest\":\"%016" PRIx64 "\",\"post_digest\":\"%016" PRIx64 "\","
                "\"pre_ok\":%s,\"post_ok\":%s}\n",
                run_id,
                row_replicate,
                phase_index,
                mapping,
                component,
                ORBITSTATE_RECEIVER_CORE
                ,
                pre_digest,
                post_digest,
                pre_digest == baseline_digest ? "true" : "false",
                post_digest == baseline_digest ? "true" : "false"
            );
        }
    }
    close_pmu_group(&pmu);
    fclose(schedule);
    fclose(raw);
    fclose(sentinels);
    fclose(stage);
    close(to_child[1]);
    fclose(child_responses);
    if (waitpid(child, NULL, 0) < 0) {
        failf("waitpid failed");
    }
    free_banks(&banks);
}

static int validate_schedule_tsv(const char *path) {
    FILE *handle = fopen(path, "r");
    char header[1024];
    char line[2048];
    int rows = 0;
    int replicate_rows[2] = {0, 0};
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
        char group_id[96];
        char run_id[96];
        char mapping[16];
        char mapping_order[32];
        char source_order[32];
        char subcapture_order[32];
        int replicate = -1;
        int phase_index = -1;
        int ordinal = -1;
        double radians = 0.0;
        int matched = sscanf(
            line,
            "%95[^\t]\t%95[^\t]\t%d\t%d\t%lf\t%15[^\t]\t%31[^\t]\t%31[^\t]\t%31[^\t]\t%d",
            group_id,
            run_id,
            &replicate,
            &phase_index,
            &radians,
            mapping,
            mapping_order,
            source_order,
            subcapture_order,
            &ordinal
        );
        (void)group_id;
        (void)run_id;
        (void)radians;
        (void)mapping_order;
        (void)source_order;
        (void)subcapture_order;
        (void)ordinal;
        if (matched != 10) {
            failf("bad tsv row");
        }
        if (replicate < 0 || replicate > 1) {
            failf("bad replicate");
        }
        if (phase_index < 0 || phase_index > 3) {
            failf("bad phase");
        }
        if (strcmp(mapping, "map0") != 0 && strcmp(mapping, "map1") != 0) {
            failf("bad mapping");
        }
        ++rows;
        ++replicate_rows[replicate];
    }
    fclose(handle);
    if (rows != 144 || replicate_rows[0] != 72 || replicate_rows[1] != 72) {
        failf("bad schedule geometry rows=%d rep0=%d rep1=%d", rows, replicate_rows[0], replicate_rows[1]);
    }
    printf("{\"status\":\"ORBITSTATE_PUBLIC_SCHEDULE_TSV_OK\",\"rows\":%d,\"replicate_0\":%d,\"replicate_1\":%d}\n",
           rows,
           replicate_rows[0],
           replicate_rows[1]);
    return 0;
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
    printf(
        "{\"status\":\"ORBITSTATE_RUNTIME_SELF_TEST_OK\",\"d_q0\":%d,\"fold_q0\":%d,"
        "\"d_q1\":%d,\"polarity_q0\":%d,\"source_off_q\":%d,"
        "\"process_boundary\":\"source_child_opens_private_map_after_fork\"}\n",
        q_d0,
        q_fold0,
        q_d1,
        q_inv,
        q_off
    );
    return 0;
}

static void usage(const char *argv0) {
    fprintf(
        stderr,
        "usage: %s --self-test | --validate-schedule-tsv PATH | "
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
        printf("{\"status\":\"ORBITSTATE_RUNTIME_REPLICATE_COMPLETE\",\"replicate\":%d}\n", replicate);
        return 0;
    }
    usage(argv[0]);
    return 2;
}
