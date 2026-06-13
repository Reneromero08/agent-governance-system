#!/bin/bash
cat > /tmp/oscillator.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <unistd.h>

static inline uint64_t osc_step(uint64_t x) {
    asm volatile("" : "+r"(x));
    x = (x * 0x41C64E6D + 0x3039);
    x = (x >> 13) ^ x;
    x = (x << 17) + x;
    return x;
}

int main(int argc, char **argv) {
    int core = atoi(argv[1]);
    unsigned long iterations = strtoul(argv[2], NULL, 10);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    uint64_t x = 0xDEADBEEFCAFEBABE;
    for (unsigned long i = 0; i < iterations; i++) {
        x = osc_step(x);
    }

    printf("%lu\n", x);
    return 0;
}
CEOF

cat > /tmp/tsc_sampler.c << 'CEOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sched.h>
#include <x86intrin.h>

#define NSAMPLES 2000000

int main(int argc, char **argv) {
    int core = atoi(argv[1]);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    uint64_t *buffer = (uint64_t *)aligned_alloc(64, NSAMPLES * sizeof(uint64_t));
    if (!buffer) { perror("alloc"); return 1; }

    for (int i = 0; i < 1000; i++) asm volatile("" ::: "memory");

    uint64_t start = __rdtsc();
    for (int i = 0; i < NSAMPLES; i++) {
        buffer[i] = __rdtsc();
    }
    uint64_t end = __rdtsc();

    fwrite(buffer, sizeof(uint64_t), NSAMPLES, stdout);
    fflush(stdout);

    fprintf(stderr, "Sampled %d timestamps in %lu cycles (%.2f cycles/sample)\n",
            NSAMPLES, end - start, (double)(end - start) / NSAMPLES);
    free(buffer);
    return 0;
}
CEOF

echo "Compiling..."
gcc -O2 -o /tmp/oscillator /tmp/oscillator.c
gcc -O2 -o /tmp/tsc_sampler /tmp/tsc_sampler.c
echo "Compilation complete."

echo ""
echo "=== LAUNCHING TWO-OSCILLATOR NETWORK ==="
/tmp/oscillator 4 2000000000 &
PID4=$!
echo "Core 4 oscillator PID: $PID4"

/tmp/oscillator 5 2000000000 &
PID5=$!
echo "Core 5 oscillator PID: $PID5"

sleep 1

echo ""
echo "=== SAMPLING TSC ON CORE 1 ==="
/tmp/tsc_sampler 1 > /tmp/tsc_data.bin 2>&1
echo "TSC sampling complete."

wait $PID4 2>/dev/null
wait $PID5 2>/dev/null
echo "Oscillators finished."

echo ""
echo "=== ANALYZING BEAT NOTE ==="
python3 << 'PYEOF'
import struct
import numpy as np

with open('/tmp/tsc_data.bin', 'rb') as f:
    data = f.read()
nsamples = len(data) // 8
tsc = np.frombuffer(data, dtype=np.uint64, count=nsamples)

deltas = np.diff(tsc).astype(np.float64)
deltas = deltas - np.mean(deltas)

fft = np.fft.rfft(deltas)
mag = np.abs(fft)
mean_delta = np.mean(np.diff(tsc))
sample_rate_hz = 3.2e9 / mean_delta if mean_delta > 0 else 3.2e9 / 30
freqs = np.fft.rfftfreq(len(deltas), d=1.0/sample_rate_hz)

peaks = []
threshold = np.mean(mag) * 5
for i in range(2, len(mag)-1):
    if mag[i] > mag[i-1] and mag[i] > mag[i+1] and mag[i] > threshold:
        peaks.append((freqs[i]/1e6, mag[i]))
peaks.sort(key=lambda x: x[1], reverse=True)

print(f"Samples: {nsamples}, mean TSC delta: {np.mean(np.diff(tsc)):.1f} cycles")
print(f"Estimated sample rate: {sample_rate_hz/1e6:.1f} MHz")
print(f"Delta std: {np.std(deltas):.1f} cycles")
print()
print("Top beat frequencies (MHz):")
for f_mhz, amp in peaks[:8]:
    print(f"  {f_mhz:.3f} MHz (amplitude {amp:.0f})")

import subprocess
result = subprocess.run(['sensors'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'k10temp' in line or 'temp1' in line:
        print(line.strip())
PYEOF

echo ""
echo "=== EXPERIMENT COMPLETE ==="
