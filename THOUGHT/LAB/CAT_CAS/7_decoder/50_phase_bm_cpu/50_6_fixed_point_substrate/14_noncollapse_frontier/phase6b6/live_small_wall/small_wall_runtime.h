#ifndef CATCAS_SMALL_WALL_RUNTIME_H
#define CATCAS_SMALL_WALL_RUNTIME_H

#include "combined_pdn_hardware.h"

#define GATE_A_TEMPERATURE_SCHEMA_ID \
    "CAT_CAS_PHASE6B6_GATE_A_NATIVE_TEMPERATURE_RECEIPT_V1"
#define GATE_A_TEMPERATURE_HWMON_ROOT "/sys/class/hwmon"
#define GATE_A_TEMPERATURE_DRIVER_NAME "k10temp"
#define GATE_A_TEMPERATURE_INPUT "temp1_input"
#define GATE_A_TEMPERATURE_RECEIPT_FILE "TEMPERATURE_RECEIPTS.jsonl"
#define GATE_A_SAMPLE_TIMING_FILE "sample_timing.bin"
#define GATE_A_TIMING_DIAGNOSTIC_FILE "TIMING_DIAGNOSTIC_SUMMARY.json"
#define GATE_A_SAMPLE_TIMING_SCHEMA_ID \
    "CAT_CAS_READONLY_OCCUPANCY_SAMPLE_TIMING_V2"
#define GATE_A_SAMPLE_TIMING_RECORD_BYTES 80U
#define GATE_A_REALTIME_PRIORITY 10
#define GATE_A_LEGACY_SLOT_COUNT 16
#define GATE_A_READONLY_MICRO_SLOT_COUNT 8
#define GATE_A_CODED_PREPROJECTION_SLOT_COUNT 16
#define GATE_A_READONLY_MICRO_READ_HZ 2000L
#define GATE_A_CODED_PREPROJECTION_READ_HZ 2000L
#define GATE_A_TEMPERATURE_MILLIDEGREES_PER_C 1000L
#define GATE_A_TEMPERATURE_VETO_MILLIDEGREES 68000L
#define GATE_A_TEMPERATURE_MIN_MILLIDEGREES (-40000L)
#define GATE_A_TEMPERATURE_MAX_MILLIDEGREES 125000L

/* Fixed CAT_CAS-owned cache-response geometry shared by runtime and worker. */
#define GATE_A_CACHE_LINE_BYTES 64U
#define GATE_A_OCCUPANCY_SMALL_BYTES (256U * 1024U)
#define GATE_A_OCCUPANCY_EQUAL_BYTES (4U * 1024U * 1024U)
#define GATE_A_OCCUPANCY_LARGE_BYTES (32U * 1024U * 1024U)
#define GATE_A_RESPONSE_BUFFER_BYTES (2U * 1024U * 1024U)
#define GATE_A_RESPONSE_TOUCHES 64U
#define GATE_A_OCCUPANCY_BURST_TOUCHES 256U
#define GATE_A_READONLY_SLOT_TOUCHES 1000003U
#define GATE_A_OCCUPANCY_LINE_STRIDE 8191U
#define GATE_A_RESPONSE_LINE_STRIDE 4093U

enum {
    GATE_A_PILOT_PN = 0,
    GATE_A_PILOT_NP = 1,
    GATE_A_PILOT_ANCHOR_SHAM = 2,
    GATE_A_PILOT_IMPULSE = 3,
    GATE_A_PILOT_STEP_SHAM = 4,
    GATE_A_PILOT_PHASE_FORWARD = 5,
    GATE_A_PILOT_PHASE_REVERSE = 6,
    GATE_A_PILOT_VALUE_FORWARD = 7,
    GATE_A_PILOT_VALUE_REVERSE = 8,
    GATE_A_PILOT_VALUE_EQUAL = 9,
    GATE_A_PILOT_OCCUPANCY_FORWARD = 10,
    GATE_A_PILOT_OCCUPANCY_REVERSE = 11,
    GATE_A_PILOT_OCCUPANCY_EQUAL = 12,
    GATE_A_PILOT_READONLY_OCCUPANCY_FORWARD = 13,
    GATE_A_PILOT_READONLY_OCCUPANCY_REVERSE = 14,
    GATE_A_PILOT_READONLY_OCCUPANCY_EQUAL = 15,
    GATE_A_PILOT_CODED_PREPROJECTION_LOOP = 16,
    GATE_A_PILOT_CODED_PREPROJECTION_RESTORED_LOOP = 17,
    GATE_A_PILOT_CODED_PREPROJECTION_WARM_RESTORED_LOOP = 18,
    GATE_A_PILOT_CODED_PREPROJECTION_WARM_QUERY_SCRAMBLE_LOOP = 19,
    GATE_A_PILOT_CODED_PREPROJECTION_WARM_QUERY_OFF_LOOP = 20,
    GATE_A_PILOT_CODED_PREPROJECTION_WARM_DECLARATION_SHAM_LOOP = 21,
    GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_SHAM_LOOP = 22,
    GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_LOOP = 23,
    GATE_A_PILOT_CODED_PREPROJECTION_ACTIVE_QUERY_LOOP = 24,
    GATE_A_PILOT_CODED_PREPROJECTION_SOURCE_PHASE_CHOP_LOOP = 25
};

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
    int pilot_variant;
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
    uint64_t capture_max_sample_delay_tsc;
    double capture_max_response_cycles_per_access;
    int step_sender_epoch_count;
    int hardware_executed;
    int sender_start_count;
    int receiver_start_count;
    int temperature_receipt_count;
    int frequency_writes;
    int voltage_writes;
    int msr_reads;
    int msr_writes;
    int scheduler_priority_attempt_count;
    int scheduler_priority_success_count;
    uint64_t max_scheduler_lateness_ticks;
    uint64_t scheduler_lateness_reject_threshold_ticks;
    uint64_t missed_deadline_count;
    uint64_t skipped_deadline_count;
    uint64_t max_finish_gap_ticks;
    char capture_quality_classification[64];
    char occupancy_digest_before[65];
    char occupancy_digest_after[65];
    int occupancy_digest_unchanged;
    int occupancy_prefaulted;
    uint64_t occupancy_requested_slot_start_tsc[16];
    uint64_t occupancy_requested_slot_end_tsc[16];
    uint64_t occupancy_burst_start_tsc[16];
    uint64_t occupancy_burst_finish_tsc[16];
    uint64_t occupancy_burst_duration_ticks[16];
    uint64_t occupancy_touch_count[16];
    uint64_t occupancy_footprint_bytes[16];
    uint64_t occupancy_initial_cursor[16];
    uint64_t occupancy_final_cursor[16];
    int occupancy_completed_before_slot_end[16];
} GateASmokeResult;

int run_gate_a_engineering_smoke(const GateASmokeArgs *, GateASmokeResult *);
int gate_a_test_readonly_timing_diagnostics(void);

#ifdef GATE_A_NATIVE_TEMPERATURE_TESTING
int gate_a_test_observe_temperature(const char *hwmon_root,
                                    const char *phase,
                                    const char *receipt_path,
                                    uint64_t observation_tsc);
#endif

#endif
