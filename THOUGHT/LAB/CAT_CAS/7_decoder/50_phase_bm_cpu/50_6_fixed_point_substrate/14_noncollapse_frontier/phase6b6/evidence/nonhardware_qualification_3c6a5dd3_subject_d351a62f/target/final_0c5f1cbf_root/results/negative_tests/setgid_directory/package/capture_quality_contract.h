#ifndef CAT_CAS_CAPTURE_QUALITY_CONTRACT_H
#define CAT_CAS_CAPTURE_QUALITY_CONTRACT_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define CAPTURE_COVERAGE_FRACTION_MIN 0.90
#define EMPIRICAL_SAMPLE_RATE_FRACTION_MIN 0.90
#define EMPIRICAL_SAMPLE_RATE_FRACTION_MAX 1.05
#define EMPIRICAL_NYQUIST_MARGIN_MIN 1.50
#define SAMPLE_GAP_MULTIPLE_MAX 4.0

static inline const char *catcas_capture_quality_failure(
        uint64_t first_sample_tsc,
        uint64_t last_sample_tsc,
        size_t sample_count,
        uint64_t capture_origin_tsc,
        uint64_t capture_deadline_tsc,
        double tsc_hz,
        long read_hz,
        double maximum_analysis_frequency_hz,
        double maximum_gap_ticks) {
    if (sample_count < 2 || last_sample_tsc <= first_sample_tsc ||
        capture_deadline_tsc <= capture_origin_tsc ||
        !isfinite(tsc_hz) || tsc_hz <= 0 || read_hz <= 0 ||
        !isfinite(maximum_analysis_frequency_hz) ||
        maximum_analysis_frequency_hz <= 0 ||
        !isfinite(maximum_gap_ticks) || maximum_gap_ticks < 0) {
        return "CAPTURE_QUALITY_INPUT_INVALID";
    }
    double capture_span = (double)(last_sample_tsc - first_sample_tsc);
    double capture_window = (double)(capture_deadline_tsc - capture_origin_tsc);
    double coverage = capture_span / capture_window;
    if (!isfinite(coverage) || coverage < CAPTURE_COVERAGE_FRACTION_MIN) {
        return "CAPTURE_COVERAGE_INSUFFICIENT";
    }
    double empirical_rate = (double)(sample_count - 1) * tsc_hz / capture_span;
    double rate_fraction = empirical_rate / (double)read_hz;
    if (!isfinite(rate_fraction) ||
        rate_fraction < EMPIRICAL_SAMPLE_RATE_FRACTION_MIN ||
        rate_fraction > EMPIRICAL_SAMPLE_RATE_FRACTION_MAX) {
        return "EMPIRICAL_SAMPLE_RATE_OUT_OF_BOUNDS";
    }
    double nyquist_margin = empirical_rate /
        (2.0 * maximum_analysis_frequency_hz);
    if (!isfinite(nyquist_margin) ||
        nyquist_margin < EMPIRICAL_NYQUIST_MARGIN_MIN) {
        return "EMPIRICAL_NYQUIST_MARGIN_INSUFFICIENT";
    }
    double nominal_spacing = tsc_hz / (double)read_hz;
    double gap_multiple = maximum_gap_ticks / nominal_spacing;
    if (!isfinite(gap_multiple) || gap_multiple > SAMPLE_GAP_MULTIPLE_MAX) {
        return "PATHOLOGICAL_TIMESTAMP_GAP";
    }
    return NULL;
}

#endif
