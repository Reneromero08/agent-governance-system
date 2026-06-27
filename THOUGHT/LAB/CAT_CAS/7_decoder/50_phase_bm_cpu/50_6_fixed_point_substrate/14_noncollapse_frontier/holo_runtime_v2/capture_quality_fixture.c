#include "capture_quality_contract.h"
#include <stdio.h>
#include <string.h>

static int expect(const char *name, const char *actual, const char *expected) {
    int ok = (!actual && !expected) || (actual && expected && !strcmp(actual, expected));
    if (!ok) fprintf(stderr, "%s: expected %s, got %s\n", name,
                     expected ? expected : "PASS", actual ? actual : "PASS");
    return ok ? 0 : 1;
}

int main(void) {
    int failures = 0;
    failures += expect("valid", catcas_capture_quality_failure(
        50, 950, 10, 0, 1000, 1000.0, 10, 2.0, 100.0), NULL);
    failures += expect("coverage", catcas_capture_quality_failure(
        50, 850, 9, 0, 1000, 1000.0, 10, 2.0, 100.0),
        "CAPTURE_COVERAGE_INSUFFICIENT");
    failures += expect("rate-low", catcas_capture_quality_failure(
        50, 950, 5, 0, 1000, 1000.0, 10, 1.0, 100.0),
        "EMPIRICAL_SAMPLE_RATE_OUT_OF_BOUNDS");
    failures += expect("rate-high", catcas_capture_quality_failure(
        50, 950, 20, 0, 1000, 1000.0, 10, 1.0, 100.0),
        "EMPIRICAL_SAMPLE_RATE_OUT_OF_BOUNDS");
    failures += expect("nyquist", catcas_capture_quality_failure(
        50, 950, 10, 0, 1000, 1000.0, 10, 4.0, 100.0),
        "EMPIRICAL_NYQUIST_MARGIN_INSUFFICIENT");
    failures += expect("gap", catcas_capture_quality_failure(
        50, 950, 10, 0, 1000, 1000.0, 10, 2.0, 500.0),
        "PATHOLOGICAL_TIMESTAMP_GAP");
    return failures ? 1 : 0;
}
