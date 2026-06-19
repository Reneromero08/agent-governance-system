#include "carrier_witness_raw.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CW_PATH_LEN 1024

static int open_exclusive_file(const char *path, const char *mode, FILE **out) {
    int flags = O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC;
    int fd = open(path, flags, 0644);
    if (fd < 0) return -1;
    FILE *file = fdopen(fd, mode);
    if (!file) {
        int saved = errno;
        close(fd);
        unlink(path);
        errno = saved;
        return -1;
    }
    *out = file;
    return 0;
}

static int write_u64_le(FILE *file, uint64_t value) {
    unsigned char bytes[8];
    for (unsigned int index = 0; index < 8; ++index) {
        bytes[index] = (unsigned char)(value >> (index * 8U));
    }
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes) ? 0 : -1;
}

static int write_double_le(FILE *file, double value) {
    uint64_t bits;
    if (!isfinite(value)) return -1;
    memcpy(&bits, &value, sizeof(bits));
    return write_u64_le(file, bits);
}

static int valid_text(const char *text, size_t limit) {
    if (!text || text[0] == '\0') return 0;
    for (size_t index = 0; index < limit && text[index] != '\0'; ++index) {
        if (text[index] == ',' || text[index] == '\n' || text[index] == '\r') return 0;
    }
    return memchr(text, '\0', limit) != NULL;
}

static int valid_window(const CarrierWitnessWindow *window,
                        const uint64_t *timestamps,
                        const double *periods,
                        int sample_count) {
    if (!window || !timestamps || !periods || sample_count < 4) return 0;
    if (!valid_text(window->family, sizeof(window->family)) ||
        !valid_text(window->declared_mode, sizeof(window->declared_mode)) ||
        !valid_text(window->actual_mode, sizeof(window->actual_mode)) ||
        !valid_text(window->control, sizeof(window->control))) return 0;
    if (window->bin_index < 0 || window->symbol_index < 0 ||
        (window->hash_restored != 0 && window->hash_restored != 1) ||
        (window->drive_sign != -1 && window->drive_sign != 0 && window->drive_sign != 1) ||
        window->slot_start_tsc >= window->capture_deadline_tsc ||
        !isfinite(window->tone_hz) || window->tone_hz <= 0.0 ||
        !isfinite(window->phase_fraction) || window->phase_fraction < 0.0 ||
        window->phase_fraction >= 1.0 ||
        !isfinite(window->temp_before_c) || !isfinite(window->temp_after_c) ||
        window->temp_before_c < -100.0 || window->temp_after_c < -100.0 ||
        !isfinite(window->computed_i) || !isfinite(window->computed_q) ||
        !isfinite(window->computed_magnitude) || !isfinite(window->computed_floor)) {
        return 0;
    }
    for (int index = 0; index < sample_count; ++index) {
        if (!isfinite(periods[index])) return 0;
        if (index > 0 && timestamps[index] <= timestamps[index - 1]) return 0;
    }
    return timestamps[0] >= window->slot_start_tsc;
}

int carrier_witness_raw_open(CarrierWitnessRawWriter *writer,
                             const char *run_directory) {
    char raw_path[CW_PATH_LEN];
    char windows_path[CW_PATH_LEN];
    if (!writer || !run_directory || run_directory[0] == '\0') return -1;
    memset(writer, 0, sizeof(*writer));
    if (snprintf(raw_path, sizeof(raw_path), "%s/raw_samples.bin", run_directory) >=
            (int)sizeof(raw_path) ||
        snprintf(windows_path, sizeof(windows_path), "%s/windows.csv", run_directory) >=
            (int)sizeof(windows_path)) {
        return -1;
    }
    if (open_exclusive_file(raw_path, "wb", &writer->raw_file) != 0) return -1;
    if (open_exclusive_file(windows_path, "w", &writer->windows_file) != 0) {
        int saved = errno;
        fclose(writer->raw_file);
        writer->raw_file = NULL;
        unlink(raw_path);
        errno = saved;
        return -1;
    }
    if (fprintf(writer->windows_file,
                "window_index,sample_offset_records,sample_count,symbol_index,bin_index,"
                "family,declared_mode,actual_mode,trial,hash_restored,theta_idx,tone_hz,"
                "drive_sign,phase_fraction,control,slot_start_tsc,capture_deadline_tsc,"
                "first_sample_tsc,last_sample_tsc,temp_before_c,temp_after_c,"
                "cur_khz_before,cur_khz_after,cofvid_pstate_before,cofvid_pstate_after,"
                "computed_I,computed_Q,computed_magnitude,computed_floor\n") < 0 ||
        fflush(writer->windows_file) != 0) {
        int saved = errno;
        fclose(writer->raw_file);
        fclose(writer->windows_file);
        writer->raw_file = NULL;
        writer->windows_file = NULL;
        unlink(raw_path);
        unlink(windows_path);
        writer->failed = 1;
        errno = saved;
        return -1;
    }
    return 0;
}

int carrier_witness_raw_append(CarrierWitnessRawWriter *writer,
                               const CarrierWitnessWindow *window,
                               const uint64_t *timestamps,
                               const double *ring_periods,
                               int sample_count) {
    if (!writer || !writer->raw_file || !writer->windows_file || writer->failed ||
        !valid_window(window, timestamps, ring_periods, sample_count)) {
        if (writer) writer->failed = 1;
        return -1;
    }

    for (int index = 0; index < sample_count; ++index) {
        if (write_u64_le(writer->raw_file, timestamps[index]) != 0 ||
            write_double_le(writer->raw_file, ring_periods[index]) != 0) {
            writer->failed = 1;
            return -1;
        }
    }

    int rc = fprintf(
        writer->windows_file,
        "%" PRIu64 ",%" PRIu64 ",%d,%d,%d,%s,%s,%s,%d,%d,%d,%.17g,%d,%.17g,%s,"
        "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.17g,%.17g,%ld,%ld,"
        "%d,%d,%.17g,%.17g,%.17g,%.17g\n",
        writer->next_window_index,
        writer->next_sample_offset,
        sample_count,
        window->symbol_index,
        window->bin_index,
        window->family,
        window->declared_mode,
        window->actual_mode,
        window->trial,
        window->hash_restored,
        window->theta_idx,
        window->tone_hz,
        window->drive_sign,
        window->phase_fraction,
        window->control,
        window->slot_start_tsc,
        window->capture_deadline_tsc,
        timestamps[0],
        timestamps[sample_count - 1],
        window->temp_before_c,
        window->temp_after_c,
        window->cur_khz_before,
        window->cur_khz_after,
        window->cofvid_pstate_before,
        window->cofvid_pstate_after,
        window->computed_i,
        window->computed_q,
        window->computed_magnitude,
        window->computed_floor);
    if (rc < 0 || fflush(writer->raw_file) != 0 ||
        fflush(writer->windows_file) != 0) {
        writer->failed = 1;
        return -1;
    }

    writer->next_sample_offset += (uint64_t)sample_count;
    writer->next_window_index++;
    return 0;
}

int carrier_witness_raw_close(CarrierWitnessRawWriter *writer) {
    int failed;
    if (!writer) return -1;
    failed = writer->failed;
    if (writer->raw_file && fclose(writer->raw_file) != 0) failed = 1;
    if (writer->windows_file && fclose(writer->windows_file) != 0) failed = 1;
    writer->raw_file = NULL;
    writer->windows_file = NULL;
    writer->failed = failed;
    return failed ? -1 : 0;
}
