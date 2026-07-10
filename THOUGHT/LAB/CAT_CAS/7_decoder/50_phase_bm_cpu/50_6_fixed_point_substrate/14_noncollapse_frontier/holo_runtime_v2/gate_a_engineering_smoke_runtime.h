#ifndef CATCAS_GATE_A_ENGINEERING_SMOKE_RUNTIME_H
#define CATCAS_GATE_A_ENGINEERING_SMOKE_RUNTIME_H

#include "combined_pdn_hardware.h"

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
    int frequency_writes;
    int voltage_writes;
    int msr_reads;
    int msr_writes;
} GateASmokeResult;

int run_gate_a_engineering_smoke(const GateASmokeArgs *, GateASmokeResult *);

#endif
