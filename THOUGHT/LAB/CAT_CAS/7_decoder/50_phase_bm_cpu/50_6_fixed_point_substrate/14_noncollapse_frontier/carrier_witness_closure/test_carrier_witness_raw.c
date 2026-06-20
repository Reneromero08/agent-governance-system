#define _GNU_SOURCE
#include "carrier_witness_raw.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static long file_size(const char *path) {
    struct stat st;
    assert(stat(path, &st) == 0);
    return (long)st.st_size;
}

int main(void) {
    char template[] = "/tmp/carrier-witness-raw-XXXXXX";
    char raw_path[512];
    char csv_path[512];
    char line[2048];
    CarrierWitnessRawWriter writer;
    CarrierWitnessWindow window;
    uint64_t timestamps[4] = {1000, 1100, 1200, 1300};
    double periods[4] = {1.0, 1.1, 0.9, 1.05};
    FILE *csv;

    char *directory = mkdtemp(template);
    assert(directory != NULL);
    memset(&window, 0, sizeof(window));
    window.symbol_index = 0;
    window.bin_index = 0;
    snprintf(window.family, sizeof(window.family), "%s", "preamble");
    snprintf(window.declared_mode, sizeof(window.declared_mode), "%s", "basis");
    snprintf(window.actual_mode, sizeof(window.actual_mode), "%s", "basis");
    window.trial = -1;
    window.hash_restored = 1;
    window.theta_idx = 0;
    window.tone_hz = 125.0;
    window.drive_sign = 1;
    window.phase_fraction = 0.0;
    snprintf(window.control, sizeof(window.control), "%s", "matrix");
    window.slot_start_tsc = 900;
    window.capture_deadline_tsc = 1500;
    window.temp_before_c = 40.0;
    window.temp_after_c = 40.5;
    window.cur_khz_before = 1600000;
    window.cur_khz_after = 1600000;
    window.cofvid_pstate_before = 2;
    window.cofvid_pstate_after = 2;
    window.computed_i = 0.1;
    window.computed_q = -0.2;
    window.computed_magnitude = 0.22360679774997896;
    window.computed_floor = 0.01;

    assert(carrier_witness_raw_open(&writer, directory) == 0);
    assert(carrier_witness_raw_append(&writer, &window, timestamps, periods, 4) == 0);
    assert(carrier_witness_raw_close(&writer) == 0);

    snprintf(raw_path, sizeof(raw_path), "%s/raw_samples.bin", directory);
    snprintf(csv_path, sizeof(csv_path), "%s/windows.csv", directory);
    assert(file_size(raw_path) == 4L * 16L);
    csv = fopen(csv_path, "r");
    assert(csv != NULL);
    assert(fgets(line, sizeof(line), csv) != NULL);
    assert(strstr(line, "sample_offset_records") != NULL);
    assert(fgets(line, sizeof(line), csv) != NULL);
    assert(strstr(line, "preamble,basis,basis") != NULL);
    assert(fclose(csv) == 0);

    unlink(raw_path);
    unlink(csv_path);
    rmdir(directory);
    puts("CARRIER_WITNESS_RAW_WRITER_TEST_PASS");
    return 0;
}
