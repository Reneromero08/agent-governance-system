#define _POSIX_C_SOURCE 200809L

#include "gate_a_engineering_smoke_runtime.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *expected_schedule_sha256 =
    "418ff6e9801ba5def3f17fb25c7d56f044599e6e5bc8cc3260e0368d4877d116";
#ifdef GATE_A_COMPILED_AUTHORITY_SHA256
static const char *compiled_authority_sha256 = GATE_A_COMPILED_AUTHORITY_SHA256;
#else
static const char *compiled_authority_sha256 = NULL;
#endif
#ifdef GATE_A_COMPILED_OUTPUT_ROOT
static const char *compiled_output_root = GATE_A_COMPILED_OUTPUT_ROOT;
#else
static const char *compiled_output_root = NULL;
#endif
static const char *expected_sequence[16] = {
    "I", "I", "I", "I", "C0", "D0", "S0E", "S0E",
    "S0E", "S0E", "O0", "O0", "A0P", "A0N", "T", "T"
};

static bool string_equal(const char *left, const char *right) {
    return left && right && strcmp(left, right) == 0;
}

static int hex_digest(const char *value, size_t length) {
    if (!value || strlen(value) != length) return 0;
    for (size_t i = 0; i < length; i++) {
        if (!((value[i] >= '0' && value[i] <= '9') ||
              (value[i] >= 'a' && value[i] <= 'f'))) return 0;
    }
    return 1;
}

static const char *option_value(int argc, char **argv, const char *name) {
    const char *value = NULL;
    for (int i = 2; i < argc; i++) {
        if (!string_equal(argv[i], name)) continue;
        if (value || i + 1 >= argc || !strncmp(argv[i + 1], "--", 2)) return NULL;
        value = argv[i + 1];
    }
    return value;
}

static int parse_long_exact(const char *text, long *value) {
    if (!text || !*text) return -1;
    errno = 0;
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (errno || !end || *end) return -1;
    *value = parsed;
    return 0;
}

static int parse_double_exact(const char *text, double *value) {
    if (!text || !*text) return -1;
    errno = 0;
    char *end = NULL;
    double parsed = strtod(text, &end);
    if (errno || !end || *end || !isfinite(parsed)) return -1;
    *value = parsed;
    return 0;
}

static int validate_schedule_semantics(void) {
    unsigned int step_epoch_slots = 0U;
    for (size_t i = 0; i < 16U; ++i) {
        const char *slot = expected_sequence[i];
        if (string_equal(slot, "I") || string_equal(slot, "C0") ||
            string_equal(slot, "D0") || string_equal(slot, "O0") ||
            string_equal(slot, "T")) continue;
        if (string_equal(slot, "S0E")) {
            ++step_epoch_slots;
            continue;
        }
        if (string_equal(slot, "A0P") || string_equal(slot, "A0N")) continue;
        return 1;
    }
    return step_epoch_slots == 4U ? 0 : 1;
}

static int validate_only(void) {
    if (validate_schedule_semantics() != 0) {
        fputs("{\"status\":\"GATE_A_WORKER_VALIDATE_ONLY_FAILED\"}\n", stderr);
        return 1;
    }
    printf("{"
         "\"automatic_retry\":false,"
         "\"control_writes\":0,"
         "\"engineering_smoke_executor_implemented\":true,"
         "\"hardware_executions\":0,"
         "\"msr_reads\":0,"
         "\"msr_writes\":0,"
         "\"receiver_captures\":0,"
         "\"sender_starts\":0,"
         "\"slot_count\":16,"
         "\"status\":\"GATE_A_WORKER_VALIDATE_ONLY_OK\","
         "\"validate_only\":true,"
         "\"live_execution_bound\":%s"
         "}\n", (compiled_authority_sha256 && compiled_output_root) ? "true" : "false");
    return 0;
}

static int probe_only(void) {
    fputs("probe-only is read-only source support and is not run in this phase\n", stderr);
    return 2;
}

static int pilot_variant_value(const char *text) {
    if (string_equal(text, "pn")) return GATE_A_PILOT_PN;
    if (string_equal(text, "np")) return GATE_A_PILOT_NP;
    if (string_equal(text, "anchor-sham")) return GATE_A_PILOT_ANCHOR_SHAM;
    if (string_equal(text, "impulse")) return GATE_A_PILOT_IMPULSE;
    if (string_equal(text, "step-sham")) return GATE_A_PILOT_STEP_SHAM;
    if (string_equal(text, "phase-forward")) return GATE_A_PILOT_PHASE_FORWARD;
    if (string_equal(text, "phase-reverse")) return GATE_A_PILOT_PHASE_REVERSE;
    if (string_equal(text, "value-forward")) return GATE_A_PILOT_VALUE_FORWARD;
    if (string_equal(text, "value-reverse")) return GATE_A_PILOT_VALUE_REVERSE;
    if (string_equal(text, "value-equal")) return GATE_A_PILOT_VALUE_EQUAL;
    if (string_equal(text, "occupancy-forward")) return GATE_A_PILOT_OCCUPANCY_FORWARD;
    if (string_equal(text, "occupancy-reverse")) return GATE_A_PILOT_OCCUPANCY_REVERSE;
    if (string_equal(text, "occupancy-equal")) return GATE_A_PILOT_OCCUPANCY_EQUAL;
    return -1;
}

static size_t pilot_working_set_bytes(int slot, int pilot) {
    if (slot < 6 || slot > 9) return 0;
    if (pilot == GATE_A_PILOT_OCCUPANCY_EQUAL) return GATE_A_OCCUPANCY_EQUAL_BYTES;
    if (pilot == GATE_A_PILOT_OCCUPANCY_FORWARD) {
        return slot == 6 || slot == 9
            ? GATE_A_OCCUPANCY_SMALL_BYTES : GATE_A_OCCUPANCY_LARGE_BYTES;
    }
    if (pilot == GATE_A_PILOT_OCCUPANCY_REVERSE) {
        return slot == 6 || slot == 9
            ? GATE_A_OCCUPANCY_LARGE_BYTES : GATE_A_OCCUPANCY_SMALL_BYTES;
    }
    return 0;
}

static int pilot_value(int slot, int pilot) {
    if (slot < 6 || slot > 9) return -1;
    if (pilot == GATE_A_PILOT_VALUE_EQUAL) return 42;
    if (pilot == GATE_A_PILOT_VALUE_FORWARD) {
        return slot == 6 || slot == 9 ? 125 : 131;
    }
    if (pilot == GATE_A_PILOT_VALUE_REVERSE) {
        return slot == 6 || slot == 9 ? 131 : 125;
    }
    return -1;
}

static int pilot_driven(int slot, int pilot) {
    if (slot >= 6 && slot <= 9) {
        if (pilot == GATE_A_PILOT_STEP_SHAM) return 0;
        if (pilot == GATE_A_PILOT_IMPULSE) return slot == 6;
        return 1;
    }
    return (slot == 12 || slot == 13) && pilot != GATE_A_PILOT_ANCHOR_SHAM;
}

static int pilot_phase(int slot, int pilot) {
    if (slot >= 6 && slot <= 9 && pilot == GATE_A_PILOT_PHASE_FORWARD) {
        return slot < 8 ? 0 : 2;
    }
    if (slot >= 6 && slot <= 9 && pilot == GATE_A_PILOT_PHASE_REVERSE) {
        return slot < 8 ? 2 : 0;
    }
    if (pilot == GATE_A_PILOT_NP && slot == 12) return 4;
    if (pilot == GATE_A_PILOT_NP && slot == 13) return 0;
    return slot == 13 ? 4 : 0;
}

static void emit_slot_records(int pilot) {
    putchar('[');
    for (int slot = 0; slot < 16; slot++) {
        int driven = pilot_driven(slot, pilot);
        if (slot) putchar(',');
        printf("{\"index\":%d,\"token\":\"%s\",\"requested_start_s\":%.1f,"
               "\"requested_end_s\":%.1f,\"drive_on\":%s,"
               "\"amplitude_level\":",
               slot, expected_sequence[slot], slot * 0.5, (slot + 1) * 0.5,
               driven ? "true" : "false");
        if (!driven) {
            fputs("null,\"phase_index\":null,\"sign\":null,"
                  "\"orbit_value\":null,\"working_set_bytes\":0,"
                  "\"sender_epoch_id\":null}", stdout);
        } else {
            int phase = pilot_phase(slot, pilot);
            int sign = phase == 4 ? -1 : 1;
            int orbit_value = pilot_value(slot, pilot);
            const char *epoch;
            if (slot >= 6 && slot <= 9) {
                if (pilot == GATE_A_PILOT_PHASE_FORWARD ||
                    pilot == GATE_A_PILOT_PHASE_REVERSE) {
                    epoch = slot < 8 ? "gate-a:phase:first"
                                     : "gate-a:phase:second";
                } else if (pilot == GATE_A_PILOT_VALUE_FORWARD ||
                           pilot == GATE_A_PILOT_VALUE_REVERSE ||
                           pilot == GATE_A_PILOT_VALUE_EQUAL) {
                    epoch = "gate-a:value:epoch0";
                } else if (pilot == GATE_A_PILOT_OCCUPANCY_FORWARD ||
                           pilot == GATE_A_PILOT_OCCUPANCY_REVERSE ||
                           pilot == GATE_A_PILOT_OCCUPANCY_EQUAL) {
                    epoch = "gate-a:occupancy:epoch0";
                } else {
                    epoch = "gate-a:step:epoch0";
                }
            } else {
                epoch = phase == 4 ? "gate-a:anchor:negative"
                                   : "gate-a:anchor:positive";
            }
            printf("2,\"phase_index\":%d,\"sign\":%d,\"orbit_value\":",
                   phase, sign);
            if (orbit_value >= 0) printf("%d", orbit_value);
            else fputs("null", stdout);
            printf(",\"working_set_bytes\":%zu,\"sender_epoch_id\":\"%s\"}",
                   pilot_working_set_bytes(slot, pilot), epoch);
        }
    }
    putchar(']');
}

static void emit_execution_result(const GateASmokeResult *result,
                                  const char *pilot_text, int pilot) {
    fputs("{\"status\":\"GATE_A_ENGINEERING_SMOKE_COMPLETE\","
          "\"automatic_retry\":false,\"runtime_execution_count\":1,"
          "\"pilot_variant\":\"", stdout);
    fputs(pilot_text, stdout);
    fputs("\","
          "\"slot_records\":", stdout);
    emit_slot_records(pilot);
    printf(",\"capture\":{\"continuous\":true,"
           "\"covers_complete_sequence\":true,\"sample_count\":%d,"
           "\"slot_sample_counts\":[", result->sample_count);
    for (int i = 0; i < 16; i++) {
        if (i) putchar(',');
        printf("%d", result->slot_sample_counts[i]);
    }
    printf("],\"origin_tsc\":%llu,\"deadline_tsc\":%llu,"
           "\"first_sample_tsc\":%llu,\"last_sample_tsc\":%llu,"
           "\"tsc_hz\":%.17g},\"frequency_writes\":%d,\"voltage_writes\":%d,"
           "\"msr_reads\":%d,\"msr_writes\":%d,"
           "\"step_sender_epoch_count\":%d,\"hardware_executed\":%s}\n",
           (unsigned long long)result->capture_origin_tsc,
           (unsigned long long)result->capture_deadline_tsc,
           (unsigned long long)result->capture_first_sample_tsc,
           (unsigned long long)result->capture_last_sample_tsc,
           result->capture_tsc_hz,
           result->frequency_writes, result->voltage_writes,
           result->msr_reads, result->msr_writes,
           result->step_sender_epoch_count,
           result->hardware_executed ? "true" : "false");
}

static int execute_authorized(int argc, char **argv) {
    const char *authority = option_value(argc, argv, "--authority-sha256");
    const char *schedule = option_value(argc, argv, "--schedule-sha256");
    const char *bundle = option_value(argc, argv, "--execution-bundle-sha256");
    const char *output = option_value(argc, argv, "--output-root");
    const char *sender_text = option_value(argc, argv, "--sender-core");
    const char *receiver_text = option_value(argc, argv, "--receiver-core");
    const char *read_text = option_value(argc, argv, "--read-hz");
    const char *slot_text = option_value(argc, argv, "--slot-s");
    const char *temp_text = option_value(argc, argv, "--temperature-veto-c");
    const char *frequency_text = option_value(argc, argv, "--required-frequency-khz");
    const char *pilot_text = option_value(argc, argv, "--pilot-variant");
    long sender = 0, receiver = 0, read_hz = 0, frequency = 0;
    double slot_s = 0, temp = 0;
    if (!pilot_text) pilot_text = "pn";
    int pilot = pilot_variant_value(pilot_text);
    if (!compiled_authority_sha256 || !compiled_output_root ||
        !string_equal(authority, compiled_authority_sha256) ||
        !string_equal(output, compiled_output_root) ||
        !hex_digest(authority, 64) || !hex_digest(bundle, 64) ||
        !string_equal(schedule, expected_schedule_sha256) || !output || output[0] != '/' ||
        parse_long_exact(sender_text, &sender) || sender != 4 ||
        parse_long_exact(receiver_text, &receiver) || receiver != 5 ||
        parse_long_exact(read_text, &read_hz) || read_hz != 8000 ||
        parse_double_exact(slot_text, &slot_s) || slot_s != 0.5 ||
        parse_double_exact(temp_text, &temp) || temp != 68.0 ||
        parse_long_exact(frequency_text, &frequency) || frequency != 1600000 ||
        pilot < GATE_A_PILOT_PN || pilot > GATE_A_PILOT_OCCUPANCY_EQUAL ||
        validate_schedule_semantics()) {
        fputs("execute-authorized requires a worker compiled for the exact validated authority and frozen geometry\n", stderr);
        return 2;
    }
    GateASmokeArgs args = {
        .output_dir = output,
        .authority_sha256 = authority,
        .execution_bundle_sha256 = bundle,
        .sender_core = (int)sender,
        .receiver_core = (int)receiver,
        .read_hz = read_hz,
        .slot_s = slot_s,
        .temperature_veto_c = temp,
        .required_frequency_khz = frequency,
        .pilot_variant = pilot,
        .backend = BACKEND_REAL,
    };
    GateASmokeResult result = {0};
    int rc = run_gate_a_engineering_smoke(&args, &result);
    if (rc) {
        fprintf(stderr, "Gate A runtime failed closed (rc=%d)\n", rc);
        return rc;
    }
    emit_execution_result(&result, pilot_text, pilot);
    return 0;
}

static int self_test(const char *retained_output, int pilot) {
    char base[] = "/tmp/gate_a_worker_selftest_XXXXXX";
    char output[CP_PATH_MAX];
    if (retained_output) {
        if (compiled_authority_sha256 || compiled_output_root ||
            retained_output[0] != '/' || strlen(retained_output) >= sizeof(output)) return 2;
        memcpy(output, retained_output, strlen(retained_output) + 1);
    } else {
        if (!mkdtemp(base)) return 1;
        if (snprintf(output, sizeof(output), "%s/runtime", base) >= (int)sizeof(output)) {
            rmdir(base);
            return 1;
        }
    }
    GateASmokeArgs args = {
        .output_dir = output,
        .authority_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        .execution_bundle_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        .sender_core = 4,
        .receiver_core = 5,
        .read_hz = 8000,
        .slot_s = 0.5,
        .temperature_veto_c = 68.0,
        .required_frequency_khz = 1600000,
        .pilot_variant = pilot,
        .backend = BACKEND_MOCK,
    };
    GateASmokeResult result = {0};
    int rc = run_gate_a_engineering_smoke(&args, &result);
    if (!rc && (result.slot_count != 16 || result.sample_count != 64000 ||
                result.step_sender_epoch_count != 1 || result.hardware_executed ||
                result.sender_start_count != 0 ||
                result.receiver_start_count != 0 ||
                result.temperature_receipt_count != 2 ||
                result.capture_origin_tsc != 1000000000ULL ||
                result.capture_first_sample_tsc != result.capture_origin_tsc ||
                result.capture_last_sample_tsc >= result.capture_deadline_tsc ||
                result.capture_deadline_tsc - result.capture_last_sample_tsc > 400000ULL ||
                result.frequency_writes || result.voltage_writes ||
                result.msr_reads || result.msr_writes)) rc = 1;
    const char *names[] = {
        "raw_samples.bin", "slot_trace.jsonl", "LOCKIN_IQ.jsonl",
        "SENDER_LIFECYCLE.jsonl", GATE_A_TEMPERATURE_RECEIPT_FILE,
        "runtime_result.json"
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        char path[CP_PATH_MAX];
        if (snprintf(path, sizeof(path), "%s/%s", output, names[i]) < (int)sizeof(path)) {
            if (access(path, F_OK)) rc = 1;
            if (!retained_output) unlink(path);
        }
    }
    if (!retained_output) {
        rmdir(output);
        rmdir(base);
    }
    if (rc) return rc;
    puts("{\"status\":\"GATE_A_WORKER_MOCK_SELF_TEST_OK\","
         "\"network_connections\":0,\"hardware_executions\":0,"
         "\"frequency_writes\":0,\"voltage_writes\":0,"
         "\"msr_reads\":0,\"msr_writes\":0,\"slot_count\":16,"
         "\"sample_count\":64000,\"step_sender_epoch_count\":1,"
         "\"temperature_receipt_count\":2}");
    return 0;
}

static int cache_response_self_test(void) {
    const size_t forward[4] = {
        pilot_working_set_bytes(6, GATE_A_PILOT_OCCUPANCY_FORWARD),
        pilot_working_set_bytes(7, GATE_A_PILOT_OCCUPANCY_FORWARD),
        pilot_working_set_bytes(8, GATE_A_PILOT_OCCUPANCY_FORWARD),
        pilot_working_set_bytes(9, GATE_A_PILOT_OCCUPANCY_FORWARD),
    };
    const size_t reverse[4] = {
        pilot_working_set_bytes(6, GATE_A_PILOT_OCCUPANCY_REVERSE),
        pilot_working_set_bytes(7, GATE_A_PILOT_OCCUPANCY_REVERSE),
        pilot_working_set_bytes(8, GATE_A_PILOT_OCCUPANCY_REVERSE),
        pilot_working_set_bytes(9, GATE_A_PILOT_OCCUPANCY_REVERSE),
    };
    for (int index = 0; index < 4; index++) {
        size_t expected_forward = index == 0 || index == 3
            ? GATE_A_OCCUPANCY_SMALL_BYTES : GATE_A_OCCUPANCY_LARGE_BYTES;
        size_t expected_reverse = index == 0 || index == 3
            ? GATE_A_OCCUPANCY_LARGE_BYTES : GATE_A_OCCUPANCY_SMALL_BYTES;
        if (forward[index] != expected_forward ||
            reverse[index] != expected_reverse ||
            pilot_working_set_bytes(
                index + 6, GATE_A_PILOT_OCCUPANCY_EQUAL) !=
                GATE_A_OCCUPANCY_EQUAL_BYTES) return 1;
    }
    return self_test(NULL, GATE_A_PILOT_OCCUPANCY_EQUAL);
}

int main(int argc, char **argv) {
    if (argc == 2 && string_equal(argv[1], "--validate-only")) return validate_only();
    if (argc == 2 && string_equal(argv[1], "--probe-only")) return probe_only();
    if (argc == 2 && string_equal(argv[1], "--self-test")) {
        return self_test(NULL, GATE_A_PILOT_PN);
    }
    if (argc == 2 && string_equal(argv[1], "--self-test-cache-response")) {
        return cache_response_self_test();
    }
    if (argc == 3 && string_equal(argv[1], "--self-test-retain")) {
        return self_test(argv[2], GATE_A_PILOT_PN);
    }
    if (argc == 3 && string_equal(argv[1], "--self-test-cache-response-retain")) {
        return self_test(argv[2], GATE_A_PILOT_OCCUPANCY_EQUAL);
    }
    if (argc >= 2 && string_equal(argv[1], "--execute-authorized")) {
        return execute_authorized(argc, argv);
    }
    fputs("usage: gate_a_worker --validate-only | --self-test | --self-test-cache-response | --self-test-retain ABSOLUTE_OUTPUT | --self-test-cache-response-retain ABSOLUTE_OUTPUT | --execute-authorized ...\n", stderr);
    return 2;
}
