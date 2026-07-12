#define _GNU_SOURCE

#include "orbit_query_runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <math.h>
#include <sched.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define ORBIT_PI 3.14159265358979323846264338327950288

typedef enum {
    MODE_PRE = 0,
    MODE_SOURCE_OFF = 1,
    MODE_QUERY_OFF = 2,
    MODE_POST = 3,
    MODE_DECLARATION_SHAM = 4
} ResponseMode;

typedef struct {
    const char *condition;
    uint32_t member;
    int has_member;
    ResponseMode mode;
    int query_scramble;
    int bank_swap;
    int public_label_swap;
} ConditionSpec;

typedef struct {
    const ConditionSpec *condition;
    OrbitState orbit;
    double phi;
    double post_scalar;
} SourceContext;

typedef struct {
    uint8_t *positive;
    uint8_t *negative;
    uint8_t *dummy;
    size_t bytes_per_bank;
} CarrierBanks;

typedef struct {
    int available;
    uint64_t cycles;
    uint64_t change_to_dirty;
    uint64_t probe_dirty;
    uint64_t duration_ns;
} Measurement;

typedef struct {
    uint32_t positive_work;
    uint32_t negative_work;
    uint32_t dummy_work;
    uint32_t total_work;
    int source_rc;
} WorkReceipt;

typedef struct {
    const char *name;
    uint64_t config;
} PerfEventSpec;

static const ConditionSpec conditions[ORBIT_QUERY_CONDITION_COUNT] = {
    {"pre_projection_d", ORBIT_QUERY_MEMBER_D, 1, MODE_PRE, 0, 0, 0},
    {"pre_projection_fold", ORBIT_QUERY_MEMBER_FOLD, 1, MODE_PRE, 0, 0, 0},
    {"source_off", 0, 0, MODE_SOURCE_OFF, 0, 0, 0},
    {"query_off", ORBIT_QUERY_MEMBER_D, 1, MODE_QUERY_OFF, 0, 0, 0},
    {"post_projection", ORBIT_QUERY_MEMBER_D, 1, MODE_POST, 0, 0, 0},
    {"declaration_sham", ORBIT_QUERY_MEMBER_D, 1, MODE_DECLARATION_SHAM, 0, 0, 0},
    {"query_scramble", ORBIT_QUERY_MEMBER_D, 1, MODE_PRE, 1, 0, 0},
    {"equal_orbit_odd_zero", ORBIT_QUERY_EQUAL_MEMBER, 1, MODE_PRE, 0, 0, 0},
    {"physical_bank_swap", ORBIT_QUERY_MEMBER_D, 1, MODE_PRE, 0, 1, 0},
    {"public_label_swap", ORBIT_QUERY_MEMBER_D, 1, MODE_PRE, 0, 0, 1}
};

static const unsigned condition_order[ORBIT_QUERY_CONDITION_COUNT] = {
    4U, 0U, 6U, 2U, 8U, 1U, 5U, 9U, 3U, 7U
};

static const char *opaque_group_ids[ORBIT_QUERY_CONDITION_COUNT] = {
    "g7c19", "g2f80", "g9a04", "g41de", "gc633",
    "g0b72", "gde58", "g6f31", "g83aa", "g14e5"
};

static const double decoder_phases[ORBIT_QUERY_PHASE_COUNT] = {
    0.0, ORBIT_PI / 2.0, ORBIT_PI, 3.0 * ORBIT_PI / 2.0
};

static long perf_event_open_call(
        struct perf_event_attr *attr,
        pid_t pid,
        int cpu,
        int group_fd,
        unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static int pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static uint64_t fnv1a64(const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < size; index++) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t fnv1a64_zero(size_t size) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < size; index++) {
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int allocate_banks(CarrierBanks *banks) {
    memset(banks, 0, sizeof(*banks));
    banks->bytes_per_bank = (size_t)ORBIT_QUERY_BANK_LINES * ORBIT_QUERY_LINE_BYTES;
    banks->positive = mmap(NULL, banks->bytes_per_bank, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (banks->positive == MAP_FAILED) {
        memset(banks, 0, sizeof(*banks));
        return -1;
    }
    banks->negative = mmap(NULL, banks->bytes_per_bank, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (banks->negative == MAP_FAILED) {
        munmap(banks->positive, banks->bytes_per_bank);
        memset(banks, 0, sizeof(*banks));
        return -1;
    }
    banks->dummy = mmap(NULL, banks->bytes_per_bank, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (banks->dummy == MAP_FAILED) {
        munmap(banks->positive, banks->bytes_per_bank);
        munmap(banks->negative, banks->bytes_per_bank);
        memset(banks, 0, sizeof(*banks));
        return -1;
    }
    return 0;
}

static void free_banks(CarrierBanks *banks) {
    if (banks->positive) munmap(banks->positive, banks->bytes_per_bank);
    if (banks->negative) munmap(banks->negative, banks->bytes_per_bank);
    if (banks->dummy) munmap(banks->dummy, banks->bytes_per_bank);
    memset(banks, 0, sizeof(*banks));
}

static double source_phase(const ConditionSpec *condition, unsigned phase_index) {
    if (condition->query_scramble) {
        return phase_index == 0U || phase_index == 2U ? 0.0 : ORBIT_PI;
    }
    return decoder_phases[phase_index % ORBIT_QUERY_PHASE_COUNT];
}

static unsigned public_phase_ordinal(
        unsigned public_index,
        int replicate,
        unsigned phase_index,
        int label_swap) {
    unsigned offset = (public_index * 3U + (unsigned)replicate + (label_swap ? 1U : 0U)) %
                      ORBIT_QUERY_PHASE_COUNT;
    return (phase_index + offset) % ORBIT_QUERY_PHASE_COUNT;
}

static double orbit_angle(const OrbitState *orbit) {
    return 2.0 * ORBIT_PI * (double)orbit->member / (double)orbit->modulus;
}

static double post_projection_response_from_scalar(double x, unsigned phase_index) {
    return x * cos(decoder_phases[phase_index % ORBIT_QUERY_PHASE_COUNT]);
}

static SourceContext make_source_context(const ConditionSpec *condition) {
    SourceContext context;
    memset(&context, 0, sizeof(context));
    context.condition = condition;
    context.orbit.modulus = ORBIT_QUERY_MODULUS;
    context.orbit.member = condition->member;
    if (condition->has_member) {
        context.phi = orbit_angle(&context.orbit);
        context.post_scalar = cos(context.phi);
    }
    return context;
}

static double source_response(const SourceContext *context, unsigned phase_index) {
    if (!context) return 0.0;
    const ConditionSpec *condition = context->condition;
    if (condition->mode == MODE_SOURCE_OFF || condition->mode == MODE_QUERY_OFF ||
        !condition->has_member) {
        return 0.0;
    }
    if (condition->mode == MODE_POST) {
        return post_projection_response_from_scalar(context->post_scalar, phase_index);
    }
    double theta = source_phase(condition, phase_index);
    if (condition->mode == MODE_DECLARATION_SHAM) {
        return cos(context->phi);
    }
    return cos(context->phi - theta);
}

static int quantize_response(double value) {
    long q = lround(value * ORBIT_QUERY_QUANT_SCALE);
    if (q > (long)ORBIT_QUERY_BASE_WORK) q = (long)ORBIT_QUERY_BASE_WORK;
    if (q < -(long)ORBIT_QUERY_BASE_WORK) q = -(long)ORBIT_QUERY_BASE_WORK;
    return (int)q;
}

static void same_value_store_units(uint8_t *bank, size_t bytes, uint32_t units, uint32_t salt) {
    size_t line_count = bytes / ORBIT_QUERY_LINE_BYTES;
    for (uint32_t unit = 0; unit < units; unit++) {
        size_t line = ((size_t)unit * 131U + (size_t)salt * 17U) & (line_count - 1U);
        volatile uint64_t *p = (volatile uint64_t *)(void *)(bank + line * ORBIT_QUERY_LINE_BYTES);
        uint64_t value = *p;
        *p = value;
    }
}

static int run_source_encoding(
        CarrierBanks *banks,
        const SourceContext *context,
        unsigned phase_index,
        uint32_t *positive_work,
        uint32_t *negative_work,
        uint32_t *dummy_work) {
    const ConditionSpec *condition = context->condition;
    if (condition->mode == MODE_SOURCE_OFF) {
        *positive_work = 0U;
        *negative_work = 0U;
        *dummy_work = ORBIT_QUERY_TOTAL_WORK;
        if (pin_to_core(ORBIT_QUERY_HOME_CORE)) return -1;
        same_value_store_units(banks->dummy, banks->bytes_per_bank,
                               *dummy_work, (uint32_t)(phase_index + 41U));
        __asm__ __volatile__("mfence" : : : "memory");
        return 0;
    }
    int q = quantize_response(source_response(context, phase_index));
    int effective_q = condition->bank_swap ? -q : q;
    *positive_work = (uint32_t)((int)ORBIT_QUERY_BASE_WORK + effective_q);
    *negative_work = (uint32_t)((int)ORBIT_QUERY_BASE_WORK - effective_q);
    *dummy_work = 0U;
    if (pin_to_core(ORBIT_QUERY_HOME_CORE)) return -1;
    for (uint32_t iter = 0; iter < ORBIT_QUERY_OPERATOR_ITERATIONS; iter++) {
        same_value_store_units(banks->positive, banks->bytes_per_bank,
                               *positive_work, (uint32_t)(phase_index + iter));
        same_value_store_units(banks->negative, banks->bytes_per_bank,
                               *negative_work, (uint32_t)(phase_index + iter + 11U));
    }
    __asm__ __volatile__("mfence" : : : "memory");
    return 0;
}

static int open_one_event(uint64_t config, int cpu, int group_fd) {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_RAW;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = 1;
    attr.exclude_kernel = 0;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_GROUP;
    return (int)perf_event_open_call(&attr, -1, cpu, group_fd, PERF_FLAG_FD_CLOEXEC);
}

static int measure_bank(uint8_t *bank, size_t bytes, Measurement *out) {
    static const PerfEventSpec specs[3] = {
        {"cpu_cycles_not_halted", UINT64_C(0x076)},
        {"cache_block_commands_change_to_dirty", UINT64_C(0x20ea)},
        {"probe_responses_dirty", UINT64_C(0x0cec)}
    };
    int fds[3] = {-1, -1, -1};
    memset(out, 0, sizeof(*out));
    if (pin_to_core(ORBIT_QUERY_RECEIVER_CORE)) return -1;
    fds[0] = open_one_event(specs[0].config, ORBIT_QUERY_RECEIVER_CORE, -1);
    if (fds[0] < 0) goto fail;
    for (int index = 1; index < 3; index++) {
        fds[index] = open_one_event(specs[index].config, ORBIT_QUERY_RECEIVER_CORE, fds[0]);
        if (fds[index] < 0) goto fail;
    }
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) ||
        ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP)) {
        goto fail;
    }
    uint64_t start = monotonic_ns();
    same_value_store_units(bank, bytes, ORBIT_QUERY_BANK_LINES, 29U);
    uint64_t finish = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP)) goto fail;
    struct {
        uint64_t nr;
        uint64_t values[3];
    } counts;
    memset(&counts, 0, sizeof(counts));
    ssize_t got = read(fds[0], &counts, sizeof(counts));
    if (got < (ssize_t)(sizeof(uint64_t) * 4U) || counts.nr < 3U) goto fail;
    out->available = 1;
    out->cycles = counts.values[0];
    out->change_to_dirty = counts.values[1];
    out->probe_dirty = counts.values[2];
    out->duration_ns = finish >= start ? finish - start : 0;
    for (int index = 0; index < 3; index++) close(fds[index]);
    return 0;
fail:
    for (int index = 0; index < 3; index++) {
        if (fds[index] >= 0) close(fds[index]);
    }
    return -1;
}

static int ensure_dir(const char *path) {
    if (mkdir(path, 0700) == 0) return 0;
    return errno == EEXIST ? 0 : -1;
}

static int join_path(char *out, size_t out_size, const char *root, const char *name) {
    int wrote = snprintf(out, out_size, "%s/%s", root, name);
    return wrote > 0 && (size_t)wrote < out_size ? 0 : -1;
}

static FILE *open_output(const char *root, const char *name) {
    char path[4096];
    if (join_path(path, sizeof(path), root, name)) return NULL;
    return fopen(path, "wx");
}

static const char *json_bool(int value) {
    return value ? "true" : "false";
}

static int write_manifest(const char *root, int replicate) {
    FILE *f = open_output(root, "CAPTURE_MANIFEST.json");
    if (!f) return -1;
    fprintf(f,
            "{\n"
            "  \"schema_id\": \"%s\",\n"
            "  \"modulus\": %u,\n"
            "  \"replicate_index\": %d,\n"
            "  \"decoder\": \"Z=(2/K)*sum(response_k*exp(i*theta_k))\",\n"
            "  \"operator\": \"byte_preserving_same_value_store\",\n"
            "  \"base_work_units\": %u,\n"
            "  \"total_source_work_units\": %u,\n"
            "  \"groups\": [\n",
            ORBIT_QUERY_SCHEMA_MANIFEST,
            ORBIT_QUERY_MODULUS,
            replicate,
            ORBIT_QUERY_BASE_WORK,
            ORBIT_QUERY_TOTAL_WORK);
    for (unsigned public_index = 0; public_index < ORBIT_QUERY_CONDITION_COUNT; public_index++) {
        fprintf(f,
                "    {\"group_id\":\"%sr%d\",\"replicate_index\":%d,"
                "\"records\":[",
                opaque_group_ids[public_index], replicate, replicate);
        for (unsigned phase = 0; phase < ORBIT_QUERY_PHASE_COUNT; phase++) {
            const ConditionSpec *spec = &conditions[condition_order[public_index]];
            fprintf(f,
                    "{\"opaque_run_id\":\"%sp%ur%d\","
                    "\"phase_ordinal\":%u,"
                    "\"decoder_phase_index\":%u,"
                    "\"decoder_phase_radians\":%.17g}%s",
                    opaque_group_ids[public_index], phase, replicate,
                    public_phase_ordinal(public_index, replicate, phase, spec->public_label_swap),
                    phase,
                    decoder_phases[phase],
                    phase + 1U == ORBIT_QUERY_PHASE_COUNT ? "" : ",");
        }
        fprintf(f, "]}%s\n", public_index + 1U == ORBIT_QUERY_CONDITION_COUNT ? "" : ",");
    }
    fprintf(f, "  ]\n}\n");
    return fclose(f);
}

static int write_unblind(
        const char *root,
        int replicate,
        WorkReceipt receipts[ORBIT_QUERY_CONDITION_COUNT][ORBIT_QUERY_PHASE_COUNT]) {
    FILE *f = open_output(root, "UNBLINDING_MAP.json");
    if (!f) return -1;
    fprintf(f,
            "{\n"
            "  \"schema_id\": \"%s\",\n"
            "  \"replicate_index\": %d,\n"
            "  \"groups\": [\n",
            ORBIT_QUERY_SCHEMA_UNBLIND, replicate);
    for (unsigned public_index = 0; public_index < ORBIT_QUERY_CONDITION_COUNT; public_index++) {
        unsigned condition = condition_order[public_index];
        const ConditionSpec *spec = &conditions[condition];
        SourceContext context = make_source_context(spec);
        fprintf(f,
                "    {\"group_id\":\"%sr%d\",\"replicate_index\":%d,"
                "\"condition\":\"%s\",\"member\":",
                opaque_group_ids[public_index], replicate, replicate, spec->condition);
        if (spec->has_member) {
            fprintf(f, "%u", spec->member);
        } else {
            fputs("null", f);
        }
        fprintf(f,
                ",\"response_mode\":%d,\"bank_swap\":%s,"
                "\"public_label_swap\":%s,\"quantized_work\":[",
                (int)spec->mode, json_bool(spec->bank_swap),
                json_bool(spec->public_label_swap));
        for (unsigned phase = 0; phase < ORBIT_QUERY_PHASE_COUNT; phase++) {
            int q = spec->mode == MODE_SOURCE_OFF
                ? 0 : quantize_response(source_response(&context, phase));
            int effective_q = spec->bank_swap ? -q : q;
            int positive = spec->mode == MODE_SOURCE_OFF
                ? 0 : (int)ORBIT_QUERY_BASE_WORK + effective_q;
            int negative = spec->mode == MODE_SOURCE_OFF
                ? 0 : (int)ORBIT_QUERY_BASE_WORK - effective_q;
            int dummy = spec->mode == MODE_SOURCE_OFF ? (int)ORBIT_QUERY_TOTAL_WORK : 0;
            fprintf(f,
                    "{\"phase_ordinal\":%u,\"q_theta\":%d,"
                    "\"positive_work\":%d,\"negative_work\":%d,"
                    "\"dummy_work\":%d,"
                    "\"total_work\":%u}%s",
                    phase, effective_q, positive, negative, dummy,
                    ORBIT_QUERY_TOTAL_WORK,
                    phase + 1U == ORBIT_QUERY_PHASE_COUNT ? "" : ",");
        }
        fprintf(f, "],\"source_work_receipt\":[");
        for (unsigned phase = 0; phase < ORBIT_QUERY_PHASE_COUNT; phase++) {
            const WorkReceipt *receipt = &receipts[public_index][phase];
            fprintf(f,
                    "{\"phase_ordinal\":%u,"
                    "\"positive_work\":%u,\"negative_work\":%u,"
                    "\"dummy_work\":%u,\"total_work\":%u,"
                    "\"source_rc\":%d}%s",
                    phase,
                    receipt->positive_work,
                    receipt->negative_work,
                    receipt->dummy_work,
                    receipt->total_work,
                    receipt->source_rc,
                    phase + 1U == ORBIT_QUERY_PHASE_COUNT ? "" : ",");
        }
        fprintf(f, "]}%s\n", public_index + 1U == ORBIT_QUERY_CONDITION_COUNT ? "" : ",");
    }
    fprintf(f, "  ]\n}\n");
    return fclose(f);
}

static int write_raw_records(
        const char *root,
        int replicate,
        WorkReceipt receipts[ORBIT_QUERY_CONDITION_COUNT][ORBIT_QUERY_PHASE_COUNT]) {
    FILE *raw = open_output(root, "RAW_CAPTURE.jsonl");
    if (!raw) return -1;
    int failures = 0;
    for (unsigned public_index = 0; public_index < ORBIT_QUERY_CONDITION_COUNT; public_index++) {
        unsigned condition = condition_order[public_index];
        const ConditionSpec *spec = &conditions[condition];
        SourceContext context = make_source_context(spec);
        for (unsigned phase = 0; phase < ORBIT_QUERY_PHASE_COUNT; phase++) {
            CarrierBanks banks;
            if (allocate_banks(&banks)) {
                fclose(raw);
                return -1;
            }
            uint64_t initial_positive = fnv1a64_zero(banks.bytes_per_bank);
            uint64_t initial_negative = fnv1a64_zero(banks.bytes_per_bank);
            uint32_t positive_work = 0;
            uint32_t negative_work = 0;
            uint32_t dummy_work = 0;
            int source_rc = run_source_encoding(&banks, &context, phase,
                                                &positive_work, &negative_work,
                                                &dummy_work);
            receipts[public_index][phase].positive_work = positive_work;
            receipts[public_index][phase].negative_work = negative_work;
            receipts[public_index][phase].dummy_work = dummy_work;
            receipts[public_index][phase].total_work = positive_work + negative_work + dummy_work;
            receipts[public_index][phase].source_rc = source_rc;
            Measurement pos;
            Measurement neg;
            int reverse_order = ((int)phase + replicate) & 1;
            int measure_rc = 0;
            if (reverse_order) {
                measure_rc |= measure_bank(banks.negative, banks.bytes_per_bank, &neg);
                measure_rc |= measure_bank(banks.positive, banks.bytes_per_bank, &pos);
            } else {
                measure_rc |= measure_bank(banks.positive, banks.bytes_per_bank, &pos);
                measure_rc |= measure_bank(banks.negative, banks.bytes_per_bank, &neg);
            }
            uint64_t final_positive = fnv1a64(banks.positive, banks.bytes_per_bank);
            uint64_t final_negative = fnv1a64(banks.negative, banks.bytes_per_bank);
            int restored = initial_positive == final_positive &&
                           initial_negative == final_negative;
            int total_work = positive_work + negative_work + dummy_work ==
                             ORBIT_QUERY_TOTAL_WORK;
            int ok = source_rc == 0 && measure_rc == 0 &&
                     pos.available && neg.available &&
                     restored && total_work;
            if (!ok) failures++;
            fprintf(raw,
                    "{\"schema_id\":\"%s\",\"opaque_run_id\":\"%sp%ur%d\","
                    "\"group_id\":\"%sr%d\",\"replicate_index\":%d,"
                    "\"phase_ordinal\":%u,\"decoder_phase_index\":%u,"
                    "\"measurement_order\":\"%s\","
                    "\"positive_cycles\":%" PRIu64 ","
                    "\"negative_cycles\":%" PRIu64 ","
                    "\"positive_change_to_dirty\":%" PRIu64 ","
                    "\"negative_change_to_dirty\":%" PRIu64 ","
                    "\"positive_probe_dirty\":%" PRIu64 ","
                    "\"negative_probe_dirty\":%" PRIu64 ","
                    "\"positive_duration_ns\":%" PRIu64 ","
                    "\"negative_duration_ns\":%" PRIu64 ","
                    "\"change_to_dirty_delta\":%.17g,"
                    "\"probe_dirty_delta\":%.17g,"
                    "\"duration_delta_ns\":%.17g,"
                    "\"restoration_passed\":%s,"
                    "\"perf_available\":%s,"
                    "\"initial_positive_digest\":\"%016" PRIx64 "\","
                    "\"initial_negative_digest\":\"%016" PRIx64 "\","
                    "\"final_positive_digest\":\"%016" PRIx64 "\","
                    "\"final_negative_digest\":\"%016" PRIx64 "\"}\n",
                    ORBIT_QUERY_SCHEMA_RAW,
                    opaque_group_ids[public_index], phase, replicate,
                    opaque_group_ids[public_index], replicate, replicate,
                    public_phase_ordinal(public_index, replicate, phase, spec->public_label_swap), phase,
                    reverse_order ? "negative_first" : "positive_first",
                    pos.cycles, neg.cycles,
                    pos.change_to_dirty, neg.change_to_dirty,
                    pos.probe_dirty, neg.probe_dirty,
                    pos.duration_ns, neg.duration_ns,
                    (double)pos.change_to_dirty - (double)neg.change_to_dirty,
                    (double)pos.probe_dirty - (double)neg.probe_dirty,
                    (double)pos.duration_ns - (double)neg.duration_ns,
                    json_bool(restored),
                    json_bool(pos.available && neg.available),
                    initial_positive, initial_negative,
                    final_positive, final_negative);
            free_banks(&banks);
        }
    }
    if (fclose(raw)) return -1;
    return failures ? -1 : 0;
}

static int write_summary(const char *root, int replicate, int raw_rc) {
    FILE *f = open_output(root, "RUNTIME_SUMMARY.json");
    if (!f) return -1;
    fprintf(f,
            "{\n"
            "  \"schema_id\": \"%s\",\n"
            "  \"status\": \"%s\",\n"
            "  \"replicate_index\": %d,\n"
            "  \"modulus\": %u,\n"
            "  \"record_count\": %u,\n"
            "  \"home_core\": %d,\n"
            "  \"receiver_core\": %d,\n"
            "  \"bank_lines\": %u,\n"
            "  \"line_bytes\": %u,\n"
            "  \"base_work_units\": %u,\n"
            "  \"total_source_work_units\": %u,\n"
            "  \"frequency_writes\": 0,\n"
            "  \"voltage_writes\": 0,\n"
            "  \"msr_reads\": 0,\n"
            "  \"msr_writes\": 0,\n"
            "  \"physical_address_access\": false,\n"
            "  \"cache_set_mapping\": false\n"
            "}\n",
            ORBIT_QUERY_SCHEMA_SUMMARY,
            raw_rc == 0 ? "ORBIT_QUERY_RUNTIME_COMPLETE" : "ORBIT_QUERY_RUNTIME_FAILED",
            replicate,
            ORBIT_QUERY_MODULUS,
            ORBIT_QUERY_CONDITION_COUNT * ORBIT_QUERY_PHASE_COUNT,
            ORBIT_QUERY_HOME_CORE,
            ORBIT_QUERY_RECEIVER_CORE,
            ORBIT_QUERY_BANK_LINES,
            ORBIT_QUERY_LINE_BYTES,
            ORBIT_QUERY_BASE_WORK,
            ORBIT_QUERY_TOTAL_WORK);
    return fclose(f);
}

static int run_capture(const char *output_root, int replicate) {
    if (!output_root || replicate < 0 || replicate > 99) return -1;
    if (ensure_dir(output_root)) return -1;
    if (write_manifest(output_root, replicate)) return -1;
    WorkReceipt receipts[ORBIT_QUERY_CONDITION_COUNT][ORBIT_QUERY_PHASE_COUNT];
    memset(receipts, 0, sizeof(receipts));
    int raw_rc = write_raw_records(output_root, replicate, receipts);
    if (write_unblind(output_root, replicate, receipts)) return -1;
    if (write_summary(output_root, replicate, raw_rc)) return -1;
    return raw_rc;
}

static int self_test(void) {
    int equal_total = 1;
    int bounded = 1;
    for (unsigned condition = 0; condition < ORBIT_QUERY_CONDITION_COUNT; condition++) {
        const ConditionSpec *spec = &conditions[condition];
        SourceContext context = make_source_context(spec);
        for (unsigned phase = 0; phase < ORBIT_QUERY_PHASE_COUNT; phase++) {
            int q = spec->mode == MODE_SOURCE_OFF
                ? 0 : quantize_response(source_response(&context, phase));
            int effective_q = spec->bank_swap ? -q : q;
            int positive = spec->mode == MODE_SOURCE_OFF
                ? 0 : (int)ORBIT_QUERY_BASE_WORK + effective_q;
            int negative = spec->mode == MODE_SOURCE_OFF
                ? 0 : (int)ORBIT_QUERY_BASE_WORK - effective_q;
            int dummy = spec->mode == MODE_SOURCE_OFF ? (int)ORBIT_QUERY_TOTAL_WORK : 0;
            if (positive + negative + dummy != (int)ORBIT_QUERY_TOTAL_WORK) equal_total = 0;
            if (positive < 0 || negative < 0) bounded = 0;
            if (positive > (int)ORBIT_QUERY_BANK_LINES ||
                negative > (int)ORBIT_QUERY_BANK_LINES ||
                dummy > (int)ORBIT_QUERY_BANK_LINES) bounded = 0;
        }
    }
    printf("{\"schema_id\":\"CAT_CAS_ORBIT_QUERY_RUNTIME_SELF_TEST_V1\","
           "\"status\":\"%s\",\"hardware_executions\":0,"
           "\"lab_device_contact\":false,\"equal_total_work\":%s,"
           "\"bounded_work\":%s,\"condition_count\":%u,\"phase_count\":%u}\n",
           equal_total && bounded ? "ORBIT_QUERY_RUNTIME_SELF_TEST_OK" :
               "ORBIT_QUERY_RUNTIME_SELF_TEST_FAILED",
           json_bool(equal_total),
           json_bool(bounded),
           ORBIT_QUERY_CONDITION_COUNT,
           ORBIT_QUERY_PHASE_COUNT);
    return equal_total && bounded ? 0 : 1;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --output-root <path> --replicate <0-99> | --self-test\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *output_root = NULL;
    int replicate = -1;
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    }
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--output-root") == 0 && index + 1 < argc) {
            output_root = argv[++index];
        } else if (strcmp(argv[index], "--replicate") == 0 && index + 1 < argc) {
            char *end = NULL;
            long parsed = strtol(argv[++index], &end, 10);
            if (!end || *end || parsed < 0 || parsed > 99) {
                usage(argv[0]);
                return 2;
            }
            replicate = (int)parsed;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!output_root || replicate < 0) {
        usage(argv[0]);
        return 2;
    }
    int rc = run_capture(output_root, replicate);
    printf("{\"schema_id\":\"CAT_CAS_ORBIT_QUERY_RUNTIME_EXECUTION_V1\","
           "\"status\":\"%s\",\"replicate_index\":%d,\"output_root\":\"%s\"}\n",
           rc == 0 ? "ORBIT_QUERY_RUNTIME_COMPLETE" : "ORBIT_QUERY_RUNTIME_FAILED",
           replicate, output_root);
    return rc == 0 ? 0 : 1;
}
