#ifndef CAT_CAS_CARRIER_WITNESS_RAW_H
#define CAT_CAS_CARRIER_WITNESS_RAW_H

#include <stdint.h>
#include <stdio.h>

#define CW_TEXT_LEN 32

typedef struct {
    int symbol_index;
    int bin_index;
    char family[CW_TEXT_LEN];
    char declared_mode[CW_TEXT_LEN];
    char actual_mode[CW_TEXT_LEN];
    int trial;
    int hash_restored;
    int theta_idx;
    double tone_hz;
    int drive_sign;
    double phase_fraction;
    char control[CW_TEXT_LEN];
    uint64_t slot_start_tsc;
    uint64_t capture_deadline_tsc;
    double temp_before_c;
    double temp_after_c;
    long cur_khz_before;
    long cur_khz_after;
    int cofvid_pstate_before;
    int cofvid_pstate_after;
    double computed_i;
    double computed_q;
    double computed_magnitude;
    double computed_floor;
} CarrierWitnessWindow;

typedef struct {
    FILE *raw_file;
    FILE *windows_file;
    uint64_t next_sample_offset;
    uint64_t next_window_index;
    int failed;
} CarrierWitnessRawWriter;

/*
 * Open `raw_samples.bin` and `windows.csv` inside an existing run directory.
 * Existing files are not overwritten.
 */
int carrier_witness_raw_open(CarrierWitnessRawWriter *writer,
                             const char *run_directory);

/*
 * Append one immutable window. Samples are written as little-endian `<Qd>`
 * records: absolute TSC followed by ring-period ticks per inner iteration.
 */
int carrier_witness_raw_append(CarrierWitnessRawWriter *writer,
                               const CarrierWitnessWindow *window,
                               const uint64_t *timestamps,
                               const double *ring_periods,
                               int sample_count);

/* Flush and close both files. Returns nonzero if any prior write failed. */
int carrier_witness_raw_close(CarrierWitnessRawWriter *writer);

#endif
