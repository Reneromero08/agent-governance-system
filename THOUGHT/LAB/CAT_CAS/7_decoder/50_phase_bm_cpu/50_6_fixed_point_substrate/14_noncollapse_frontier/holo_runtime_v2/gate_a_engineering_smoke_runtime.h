#ifndef CATCAS_GATE_A_ENGINEERING_SMOKE_RUNTIME_H
#define CATCAS_GATE_A_ENGINEERING_SMOKE_RUNTIME_H

#include "combined_pdn_hardware.h"

#define GATE_A_TEMPERATURE_SCHEMA_ID \
    "CAT_CAS_PHASE6B6_GATE_A_NATIVE_TEMPERATURE_RECEIPT_V1"
#define GATE_A_TEMPERATURE_HWMON_ROOT "/sys/class/hwmon"
#define GATE_A_TEMPERATURE_DRIVER_NAME "k10temp"
#define GATE_A_TEMPERATURE_INPUT "temp1_input"
#define GATE_A_TEMPERATURE_RECEIPT_FILE "TEMPERATURE_RECEIPTS.jsonl"
#define GATE_A_TEMPERATURE_MILLIDEGREES_PER_C 1000L
#define GATE_A_TEMPERATURE_VETO_MILLIDEGREES 68000L
#define GATE_A_TEMPERATURE_MIN_MILLIDEGREES (-40000L)
#define GATE_A_TEMPERATURE_MAX_MILLIDEGREES 125000L

typedef struct {
    const char *output_dir;
    const char *authority_sha256;
    const char *execution_bundle_sha256;
    int sender_core;
    int receiver_core;
    long read_hz;
    double slot_s;
    double temperature_veto_c;
    long required_frequency_khz;
    int backend;
    long mock_pre_temperature_millidegrees;
    long mock_post_temperature_millidegrees;
} GateASmokeArgs;

typedef struct {
    int slot_count;
    int sample_count;
    int slot_sample_counts[16];
    uint64_t capture_origin_tsc;
    uint64_t capture_deadline_tsc;
    uint64_t capture_first_sample_tsc;
    uint64_t capture_last_sample_tsc;
    double capture_tsc_hz;
    int step_sender_epoch_count;
    int hardware_executed;
    int sender_start_count;
    int receiver_start_count;
    int temperature_receipt_count;
    int frequency_writes;
    int voltage_writes;
    int msr_reads;
    int msr_writes;
} GateASmokeResult;

int run_gate_a_engineering_smoke(const GateASmokeArgs *, GateASmokeResult *);

#ifdef GATE_A_NATIVE_TEMPERATURE_TESTING
int gate_a_test_observe_temperature(const char *hwmon_root,
                                    const char *phase,
                                    const char *receipt_path,
                                    uint64_t observation_tsc);
#endif

#endif
