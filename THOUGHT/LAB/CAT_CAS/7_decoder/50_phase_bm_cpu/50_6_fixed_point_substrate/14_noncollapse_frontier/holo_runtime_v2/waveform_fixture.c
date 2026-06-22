#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static double tone_hz(int index) {
    double low = log(20.0), high = log(1500.0), x = index / 11.0;
    return exp(low + (high - low) * x) *
           (1.0 + .013 * sin(2.399963 * (index + 1)));
}

static int gate(uint64_t timestamp, uint64_t origin, double tsc_hz,
                int tone_index, int phase_index, int amplitude_level) {
    double step_ticks = tsc_hz / (8.0 * tone_hz(tone_index));
    double offset = (double)(timestamp - origin) - phase_index * step_ticks;
    long state = (long)floor(offset / step_ticks);
    int cycle_state = (int)((state % 8 + 8) % 8);
    return cycle_state < amplitude_level * 2;
}

int main(int argc, char **argv) {
    if (argc != 7) return 2;
    uint64_t origin = strtoull(argv[1], NULL, 10);
    double tsc_hz = strtod(argv[2], NULL);
    int tone = atoi(argv[3]), phase = atoi(argv[4]), level = atoi(argv[5]);
    uint64_t count = strtoull(argv[6], NULL, 10);
    for (uint64_t timestamp = origin; timestamp < origin + count; timestamp++) {
        printf("%d\n", gate(timestamp, origin, tsc_hz, tone, phase, level));
    }
    return 0;
}
