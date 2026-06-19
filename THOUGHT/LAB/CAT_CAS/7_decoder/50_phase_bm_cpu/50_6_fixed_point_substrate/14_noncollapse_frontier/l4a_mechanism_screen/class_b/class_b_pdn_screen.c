/*
 * class_b_pdn_screen.c -- L4A Class B crossed-assignment PDN calibration.
 *
 * This program measures a complex value-dependent PDN coordinate while
 * separating fixed core/route bias. It is a calibration capture, not a
 * fold-orientation, restoration, or wall-crossing experiment.
 *
 * Build:
 *   gcc -O2 -std=gnu11 -pthread -march=amdfam10 -Wall -Wextra -Werror \
 *     class_b_pdn_screen.c -o class_b_pdn_screen -lm
 *
 * THE ALGORITHM IS DEAD: the unresolved orbit is the object; no candidate wins.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_SENDER_A 4
#define DEFAULT_SENDER_B 5
#define DEFAULT_RECEIVER 2
#define DEFAULT_DRIVE_HZ 200.0
#define DEFAULT_SLOT_S 0.4
#define DEFAULT_READ_HZ 4000
#define MAX_SAMPLES 4096
#define TEMP_VETO_C 68.0
#define WORKLOAD_ID "class_b_value_switching_v2"
#define SCHEMA_ID "CAT_CAS_CLASS_B_CROSSOVER_CAPTURE_V1"

typedef struct {
    double i;
    double q;
    double magnitude;
    double temp_before_c;
    double temp_after_c;
    int sender_core;
    int orbit_value;
    int carrier_enabled;
    int sample_count;
} Capture;

typedef struct {
    double re;
    double im;
} ComplexValue;

typedef struct {
    atomic_int stop;
    atomic_int started;
    pthread_t thread;
    int core;
    int orbit_value;
    int modulus;
    uint64_t t0;
    double half_ticks;
} Sender;

typedef struct {
    int core;
    int read_hz;
    int sample_count;
    double tsc_hz;
    uint64_t deadline;
    uint64_t *timestamps;
    double *periods;
    atomic_int *go;
    volatile int ready;
    int affinity_ok;
} Receiver;

static char g_k10_path[512];

static inline uint64_t rdtsc_now(void) {
    unsigned int hi, lo;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtscp_now(void) {
    unsigned int hi, lo, aux;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    return ((uint64_t)hi << 32) | lo;
}

static int pin_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

static int find_k10temp(void) {
    DIR *dir = opendir("/sys/class/hwmon");
    struct dirent *entry;
    if (!dir) return -1;
    while ((entry = readdir(dir)) != NULL) {
        char name_path[512];
        char name[64];
        FILE *file;
        size_t length;
        if (strncmp(entry->d_name, "hwmon", 5) != 0) continue;
        snprintf(name_path, sizeof(name_path), "/sys/class/hwmon/%s/name", entry->d_name);
        file = fopen(name_path, "r");
        if (!file) continue;
        if (!fgets(name, sizeof(name), file)) {
            fclose(file);
            continue;
        }
        fclose(file);
        length = strlen(name);
        while (length > 0 && name[length - 1] == '\n') name[--length] = '\0';
        if (strcmp(name, "k10temp") == 0) {
            snprintf(g_k10_path, sizeof(g_k10_path),
                     "/sys/class/hwmon/%s/temp1_input", entry->d_name);
            closedir(dir);
            return 0;
        }
    }
    closedir(dir);
    return -1;
}

static double k10temp_c(void) {
    FILE *file;
    long millidegrees = 0;
    if (g_k10_path[0] == '\0') return -999.0;
    file = fopen(g_k10_path, "r");
    if (!file) return -999.0;
    if (fscanf(file, "%ld", &millidegrees) != 1) millidegrees = -999000;
    fclose(file);
    return (double)millidegrees / 1000.0;
}

/* Equal-count workload whose integer switching state depends on orbit_value. */
static double value_burst(uint64_t *seed, int orbit_value, int modulus) {
    double a0 = 1.0000001, a1 = 1.0000002, a2 = 1.0000003, a3 = 1.0000004;
    double a4 = 1.0000005, a5 = 1.0000006, a6 = 1.0000007, a7 = 1.0000008;
    uint64_t i0 = *seed ^ UINT64_C(0x9E3779B97F4A7C15);
    uint64_t i1 = i0 * UINT64_C(2654435761) + 1U;
    uint64_t i2 = i1 ^ UINT64_C(0xD1B54A32D192ED03);
    uint64_t i3 = i2 * UINT64_C(1099511628211) + 1U;
    uint64_t operand = (uint64_t)(unsigned int)orbit_value;
    uint64_t mask = (uint64_t)(unsigned int)(modulus - 1);
    int k;

    for (k = 0; k < 64; ++k) {
        operand = (operand * UINT64_C(1103515245) + UINT64_C(12345)) & mask;
        i0 = i0 * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        i1 = i1 * UINT64_C(3935559000370003845) + UINT64_C(2691343689449507681);
        i2 = i2 * UINT64_C(0x2545F4914F6CDD1D) + UINT64_C(0x14057B7EF767814F);
        i3 = i3 * UINT64_C(0x9E3779B97F4A7C15) + UINT64_C(0xBF58476D1CE4E5B9);
        i0 ^= operand * UINT64_C(0xD6E8FEB86659FD93);
        i2 ^= (operand + (uint64_t)k) * UINT64_C(0xA0761D6478BD642F);

        a0 = a0 * 1.0000000007 + 0.9999999993;
        a1 = a1 * 0.9999999993 + 1.0000000007;
        a2 = a2 * 1.0000000011 + 0.9999999989;
        a3 = a3 * 0.9999999989 + 1.0000000011;
        a4 = a4 * 1.0000000013 + 0.9999999987;
        a5 = a5 * 0.9999999987 + 1.0000000013;
        a6 = a6 * 1.0000000003 + 0.9999999997;
        a7 = a7 * 0.9999999997 + 1.0000000003;
        a0 += (double)((i0 >> 40) & 3U);
        a4 += (double)((i2 >> 40) & 3U);
    }

    *seed = i0 ^ i1 ^ i2 ^ i3 ^ operand;
    return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
           (double)((i0 ^ i1 ^ i2 ^ i3 ^ operand) & 0xffU);
}

static void lockin(const uint64_t *timestamps, const double *samples, int count,
                   double frequency, uint64_t t0, double tsc_hz,
                   double *out_i, double *out_q, double *out_magnitude) {
    double mean = 0.0;
    double sum_i = 0.0;
    double sum_q = 0.0;
    double weight_sum = 0.0;
    int index;
    if (count < 4) {
        *out_i = *out_q = *out_magnitude = 0.0;
        return;
    }
    for (index = 0; index < count; ++index) mean += samples[index];
    mean /= (double)count;
    for (index = 0; index < count; ++index) {
        double weight = 0.5 * (1.0 - cos(2.0 * M_PI * index / (double)(count - 1)));
        double dt = (double)(timestamps[index] - t0) / tsc_hz;
        double phase = 2.0 * M_PI * frequency * dt;
        double value = (samples[index] - mean) * weight;
        sum_i += value * cos(phase);
        sum_q += value * sin(phase);
        weight_sum += weight;
    }
    if (weight_sum <= 0.0) weight_sum = 1.0;
    *out_i = 2.0 * sum_i / weight_sum;
    *out_q = 2.0 * sum_q / weight_sum;
    *out_magnitude = hypot(*out_i, *out_q);
}

static void *sender_loop(void *argument) {
    Sender *sender = (Sender *)argument;
    uint64_t seed;
    volatile double sink = 0.0;
    if (pin_core(sender->core) != 0) return NULL;
    seed = ((uint64_t)(unsigned int)sender->core << 32) ^
           (uint64_t)(unsigned int)sender->orbit_value ^ UINT64_C(0xA0761D6478BD642F);
    atomic_store(&sender->started, 1);
    while (rdtsc_now() < sender->t0) __asm__ volatile("pause");
    while (!atomic_load(&sender->stop)) {
        uint64_t now = rdtsc_now();
        double elapsed = (double)(now - sender->t0);
        long half_period = (long)floor(elapsed / sender->half_ticks);
        if ((half_period & 1L) == 0) {
            sink += value_burst(&seed, sender->orbit_value, sender->modulus);
        } else {
            __asm__ volatile("pause");
        }
    }
    __asm__ volatile("" : : "r"(sink));
    return NULL;
}

static int sender_start(Sender *sender, int core, int orbit_value, int modulus,
                        uint64_t t0, double half_ticks) {
    pthread_attr_t attributes;
    cpu_set_t set;
    int rc;
    memset(sender, 0, sizeof(*sender));
    sender->core = core;
    sender->orbit_value = orbit_value;
    sender->modulus = modulus;
    sender->t0 = t0;
    sender->half_ticks = half_ticks;
    atomic_init(&sender->stop, 0);
    atomic_init(&sender->started, 0);
    pthread_attr_init(&attributes);
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    rc = pthread_attr_setaffinity_np(&attributes, sizeof(set), &set);
    if (rc == 0) rc = pthread_create(&sender->thread, &attributes, sender_loop, sender);
    pthread_attr_destroy(&attributes);
    if (rc != 0) return -1;
    while (!atomic_load(&sender->started)) __asm__ volatile("pause");
    return 0;
}

static void sender_stop(Sender *sender) {
    atomic_store(&sender->stop, 1);
    (void)pthread_join(sender->thread, NULL);
}

static void *receiver_loop(void *argument) {
    Receiver *receiver = (Receiver *)argument;
    volatile uint64_t accumulator = UINT64_C(0x9E3779B9);
    double target_ticks = receiver->tsc_hz / (double)receiver->read_hz;
    uint64_t previous;
    int index;
    receiver->affinity_ok = pin_core(receiver->core) == 0;
    for (index = 0; index < 8192; ++index) {
        accumulator = accumulator * UINT64_C(6364136223846793005) + 1U;
    }
    receiver->ready = 1;
    while (atomic_load(receiver->go) == 0) __asm__ volatile("pause");
    previous = rdtscp_now();
    for (index = 0; index < MAX_SAMPLES; ++index) {
        uint64_t iterations = 0;
        uint64_t now;
        do {
            accumulator = accumulator * UINT64_C(6364136223846793005) +
                          UINT64_C(1442695040888963407);
            ++iterations;
            now = rdtscp_now();
        } while ((double)(now - previous) < target_ticks);
        receiver->timestamps[index] = now;
        receiver->periods[index] = (double)(now - previous) / (double)iterations;
        previous = now;
        if (now >= receiver->deadline) {
            ++index;
            break;
        }
    }
    receiver->sample_count = index;
    __asm__ volatile("" : : "r"(accumulator));
    return NULL;
}

static int measure_window(int sender_core, int receiver_core, int orbit_value,
                          int modulus, int carrier_enabled, double drive_hz,
                          double slot_s, int read_hz, double tsc_hz,
                          Capture *capture) {
    uint64_t timestamps[MAX_SAMPLES];
    double periods[MAX_SAMPLES];
    uint64_t launch = rdtsc_now() + (uint64_t)(0.02 * tsc_hz);
    double half_ticks = (tsc_hz / drive_hz) * 0.5;
    atomic_int go;
    Receiver receiver;
    Sender sender;
    pthread_t receiver_thread;
    pthread_attr_t attributes;
    cpu_set_t set;
    int sender_started = 0;
    int rc;

    memset(capture, 0, sizeof(*capture));
    capture->sender_core = sender_core;
    capture->orbit_value = orbit_value;
    capture->carrier_enabled = carrier_enabled;
    capture->temp_before_c = k10temp_c();
    if (capture->temp_before_c >= TEMP_VETO_C) return -2;

    if (carrier_enabled) {
        if (sender_start(&sender, sender_core, orbit_value, modulus, launch, half_ticks) != 0) return -3;
        sender_started = 1;
    }

    atomic_init(&go, 0);
    memset(&receiver, 0, sizeof(receiver));
    receiver.core = receiver_core;
    receiver.read_hz = read_hz;
    receiver.tsc_hz = tsc_hz;
    receiver.deadline = launch + (uint64_t)(slot_s * tsc_hz);
    receiver.timestamps = timestamps;
    receiver.periods = periods;
    receiver.go = &go;

    pthread_attr_init(&attributes);
    CPU_ZERO(&set);
    CPU_SET(receiver_core, &set);
    rc = pthread_attr_setaffinity_np(&attributes, sizeof(set), &set);
    if (rc == 0) rc = pthread_create(&receiver_thread, &attributes, receiver_loop, &receiver);
    pthread_attr_destroy(&attributes);
    if (rc != 0) {
        if (sender_started) sender_stop(&sender);
        return -4;
    }
    while (!receiver.ready) __asm__ volatile("pause");
    while (rdtsc_now() < launch) __asm__ volatile("pause");
    atomic_store(&go, 1);
    (void)pthread_join(receiver_thread, NULL);
    if (sender_started) sender_stop(&sender);
    if (!receiver.affinity_ok || receiver.sample_count < 4) return -5;

    lockin(timestamps, periods, receiver.sample_count, drive_hz, launch, tsc_hz,
           &capture->i, &capture->q, &capture->magnitude);
    capture->sample_count = receiver.sample_count;
    capture->temp_after_c = k10temp_c();
    if (capture->temp_after_c >= TEMP_VETO_C) return -6;
    return 0;
}

static ComplexValue as_complex(const Capture *capture) {
    ComplexValue value = { capture->i, capture->q };
    return value;
}

static ComplexValue complex_sub(ComplexValue left, ComplexValue right) {
    ComplexValue value = { left.re - right.re, left.im - right.im };
    return value;
}

static ComplexValue complex_scale(ComplexValue value, double scale) {
    value.re *= scale;
    value.im *= scale;
    return value;
}

static void write_capture(FILE *file, const char *name, const Capture *capture, int comma) {
    fprintf(file,
            "    \"%s\":{\"core\":%d,\"value\":%d,\"carrier_enabled\":%s,"
            "\"I\":%.17g,\"Q\":%.17g,\"magnitude\":%.17g,"
            "\"samples\":%d,\"temp_before_c\":%.6f,\"temp_after_c\":%.6f}%s\n",
            name, capture->sender_core, capture->orbit_value,
            capture->carrier_enabled ? "true" : "false",
            capture->i, capture->q, capture->magnitude, capture->sample_count,
            capture->temp_before_c, capture->temp_after_c, comma ? "," : "");
}

int main(int argc, char **argv) {
    int modulus = 256;
    int lower = 125;
    int core_a = DEFAULT_SENDER_A;
    int core_b = DEFAULT_SENDER_B;
    int receiver = DEFAULT_RECEIVER;
    int read_hz = DEFAULT_READ_HZ;
    double tsc_hz = 3214823000.0;
    double drive_hz = DEFAULT_DRIVE_HZ;
    double slot_s = DEFAULT_SLOT_S;
    const char *output_path = "results/class_b_crossover_measurement.json";
    Capture c4_lower, c5_mirror, c4_mirror, c5_lower;
    Capture c4_idle, c5_idle, c4_dummy, c5_dummy;
    ComplexValue d_normal, d_swap, r_value, r_core;
    ComplexValue same_orbit, dummy_bias, carrier_off;
    FILE *output;
    int mirror;
    int index;

    for (index = 1; index < argc; ++index) {
        if (!strcmp(argv[index], "--N") && index + 1 < argc) modulus = atoi(argv[++index]);
        else if (!strcmp(argv[index], "--a") && index + 1 < argc) lower = atoi(argv[++index]);
        else if (!strcmp(argv[index], "--tsc-hz") && index + 1 < argc) tsc_hz = strtod(argv[++index], NULL);
        else if (!strcmp(argv[index], "--out") && index + 1 < argc) output_path = argv[++index];
        else if (!strcmp(argv[index], "--core-a") && index + 1 < argc) core_a = atoi(argv[++index]);
        else if (!strcmp(argv[index], "--core-b") && index + 1 < argc) core_b = atoi(argv[++index]);
        else if (!strcmp(argv[index], "--receiver") && index + 1 < argc) receiver = atoi(argv[++index]);
        else {
            fprintf(stderr, "unknown/incomplete argument: %s\n", argv[index]);
            return 2;
        }
    }

    if (modulus <= 2 || (modulus & (modulus - 1)) != 0 || lower <= 0 || lower >= modulus / 2 || tsc_hz <= 0.0) {
        fprintf(stderr, "invalid N/a/tsc configuration\n");
        return 2;
    }
    mirror = modulus - lower;
    if (find_k10temp() != 0) {
        fprintf(stderr, "k10temp not found\n");
        return 2;
    }

#define CAPTURE(dst, core, value, enabled) \
    do { \
        int capture_rc = measure_window((core), receiver, (value), modulus, (enabled), \
                                        drive_hz, slot_s, read_hz, tsc_hz, &(dst)); \
        if (capture_rc != 0) { \
            fprintf(stderr, "capture failed rc=%d core=%d value=%d enabled=%d\n", \
                    capture_rc, (core), (value), (enabled)); \
            return 3; \
        } \
    } while (0)

    CAPTURE(c4_lower, core_a, lower, 1);
    CAPTURE(c5_mirror, core_b, mirror, 1);
    CAPTURE(c4_mirror, core_a, mirror, 1);
    CAPTURE(c5_lower, core_b, lower, 1);
    CAPTURE(c4_idle, core_a, 0, 0);
    CAPTURE(c5_idle, core_b, 0, 0);
    CAPTURE(c4_dummy, core_a, 42, 1);
    CAPTURE(c5_dummy, core_b, 42, 1);
#undef CAPTURE

    d_normal = complex_sub(as_complex(&c4_lower), as_complex(&c5_mirror));
    d_swap = complex_sub(as_complex(&c4_mirror), as_complex(&c5_lower));
    r_value = complex_scale(complex_sub(d_normal, d_swap), 0.5);
    r_core = complex_scale((ComplexValue){d_normal.re + d_swap.re, d_normal.im + d_swap.im}, 0.5);
    same_orbit = complex_sub(as_complex(&c4_lower), as_complex(&c5_lower));
    dummy_bias = complex_sub(as_complex(&c4_dummy), as_complex(&c5_dummy));
    carrier_off = complex_sub(as_complex(&c4_idle), as_complex(&c5_idle));

    output = fopen(output_path, "w");
    if (!output) {
        fprintf(stderr, "cannot open %s: %s\n", output_path, strerror(errno));
        return 4;
    }
    fprintf(output, "{\n");
    fprintf(output, "  \"schema_id\":\"%s\",\n", SCHEMA_ID);
    fprintf(output, "  \"status\":\"CAPTURED_NOT_ADJUDICATED\",\n");
    fprintf(output, "  \"workload_id\":\"%s\",\n", WORKLOAD_ID);
    fprintf(output, "  \"N\":%d,\"a\":%d,\"mirror\":%d,\n", modulus, lower, mirror);
    fprintf(output, "  \"receiver_core\":%d,\"drive_hz\":%.17g,\"slot_s\":%.17g,\"read_hz\":%d,\"tsc_hz\":%.17g,\n",
            receiver, drive_hz, slot_s, read_hz, tsc_hz);
    fprintf(output, "  \"capture_order\":[\"core4/a\",\"core5/mirror\",\"core4/mirror\",\"core5/a\",\"core4/idle\",\"core5/idle\",\"core4/dummy42\",\"core5/dummy42\"],\n");
    fprintf(output, "  \"captures\":{\n");
    write_capture(output, "core_a_lower", &c4_lower, 1);
    write_capture(output, "core_b_mirror", &c5_mirror, 1);
    write_capture(output, "core_a_mirror", &c4_mirror, 1);
    write_capture(output, "core_b_lower", &c5_lower, 1);
    write_capture(output, "core_a_idle", &c4_idle, 1);
    write_capture(output, "core_b_idle", &c5_idle, 1);
    write_capture(output, "core_a_dummy42", &c4_dummy, 1);
    write_capture(output, "core_b_dummy42", &c5_dummy, 0);
    fprintf(output, "  },\n");
    fprintf(output, "  \"coordinates\":{\n");
    fprintf(output, "    \"D_normal\":{\"I\":%.17g,\"Q\":%.17g},\n", d_normal.re, d_normal.im);
    fprintf(output, "    \"D_swap\":{\"I\":%.17g,\"Q\":%.17g},\n", d_swap.re, d_swap.im);
    fprintf(output, "    \"R_value_complex\":{\"I\":%.17g,\"Q\":%.17g,\"magnitude\":%.17g},\n",
            r_value.re, r_value.im, hypot(r_value.re, r_value.im));
    fprintf(output, "    \"R_core_complex\":{\"I\":%.17g,\"Q\":%.17g,\"magnitude\":%.17g},\n",
            r_core.re, r_core.im, hypot(r_core.re, r_core.im));
    fprintf(output, "    \"same_orbit_core_bias\":{\"I\":%.17g,\"Q\":%.17g},\n", same_orbit.re, same_orbit.im);
    fprintf(output, "    \"dummy_core_bias\":{\"I\":%.17g,\"Q\":%.17g},\n", dummy_bias.re, dummy_bias.im);
    fprintf(output, "    \"carrier_off_bias\":{\"I\":%.17g,\"Q\":%.17g}\n", carrier_off.re, carrier_off.im);
    fprintf(output, "  },\n");
    fprintf(output, "  \"claim_ceiling\":\"CHANNEL_CALIBRATION_ONLY\",\n");
    fprintf(output, "  \"forbidden_claims\":[\"orientation\",\"recovered_d\",\"physical_geometric_memory\",\"physical_restoration\"]\n");
    fprintf(output, "}\n");
    if (fclose(output) != 0) return 4;

    printf("L4A_CLASS_B_CROSSOVER_MEASUREMENT_CAPTURED\n");
    printf("artifact=%s\n", output_path);
    printf("R_value=(%.9g, %.9g) magnitude=%.9g\n", r_value.re, r_value.im, hypot(r_value.re, r_value.im));
    printf("R_core=(%.9g, %.9g) magnitude=%.9g\n", r_core.re, r_core.im, hypot(r_core.re, r_core.im));
    printf("No adjudication. No orientation. No restoration claim.\n");
    return 0;
}
