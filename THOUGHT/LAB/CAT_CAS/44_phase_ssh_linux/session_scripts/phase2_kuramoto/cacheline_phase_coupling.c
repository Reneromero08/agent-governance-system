#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_SAMPLES 20000
#define PERIOD 4096u

struct shared_state {
    volatile uint64_t counter_a;
    volatile uint64_t counter_b;
    char pad_a[64];
    volatile uint64_t isolated_a;
    char pad_b[64];
    volatile uint64_t isolated_b;
    volatile int stop;
};

struct worker_args {
    struct shared_state *state;
    int core;
    int worker_id;
    int false_share;
    int atomic_pressure;
};

struct sample {
    uint64_t a;
    uint64_t b;
};

static void pin_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "sched_setaffinity core %d failed: %s\n", core, strerror(errno));
        exit(2);
    }
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void *worker_main(void *opaque) {
    struct worker_args *args = (struct worker_args *)opaque;
    uint64_t local = 0x9e3779b97f4a7c15ull ^ (uint64_t)(args->core * 0x10001u);
    pin_core(args->core);

    while (!__atomic_load_n(&args->state->stop, __ATOMIC_RELAXED)) {
        local += 1u;
        local ^= local << 7;
        local ^= local >> 9;
        local += 0xD1B54A32D192ED03ull;

        if ((local & 0x3Fu) == 0) {
            uint64_t phase = local;
            if (args->false_share) {
                if (args->atomic_pressure) {
                    if (args->worker_id == 0) {
                        __atomic_fetch_add(&args->state->counter_a, phase & 0xFFu, __ATOMIC_SEQ_CST);
                    } else {
                        __atomic_fetch_add(&args->state->counter_b, phase & 0xFFu, __ATOMIC_SEQ_CST);
                    }
                } else if (args->worker_id == 0) {
                    __atomic_store_n(&args->state->counter_a, phase, __ATOMIC_RELEASE);
                } else {
                    __atomic_store_n(&args->state->counter_b, phase, __ATOMIC_RELEASE);
                }
            } else if (args->worker_id == 0) {
                __atomic_store_n(&args->state->isolated_a, phase, __ATOMIC_RELEASE);
            } else {
                __atomic_store_n(&args->state->isolated_b, phase, __ATOMIC_RELEASE);
            }
        }
    }
    return NULL;
}

static double phase_r(const struct sample *samples, int count, int shift_b) {
    double cos_sum = 0.0;
    double sin_sum = 0.0;
    int used = 0;
    for (int i = 0; i < count; i++) {
        int j = (i + shift_b) % count;
        uint32_t delta = (uint32_t)((samples[i].a - samples[j].b) & (PERIOD - 1u));
        double theta = (2.0 * M_PI * (double)delta) / (double)PERIOD;
        cos_sum += cos(theta);
        sin_sum += sin(theta);
        used++;
    }
    if (!used) {
        return 0.0;
    }
    cos_sum /= (double)used;
    sin_sum /= (double)used;
    return sqrt(cos_sum * cos_sum + sin_sum * sin_sum);
}

static double mean_abs_delta_step(const struct sample *samples, int count) {
    if (count < 2) {
        return 0.0;
    }
    double total = 0.0;
    for (int i = 1; i < count; i++) {
        int64_t da = (int64_t)(samples[i].a - samples[i - 1].a);
        int64_t db = (int64_t)(samples[i].b - samples[i - 1].b);
        int64_t diff = da - db;
        total += fabs((double)diff);
    }
    return total / (double)(count - 1);
}

static void run_case(
    const char *mode,
    int core_a,
    int core_b,
    int sample_core,
    int repeat,
    int false_share,
    int atomic_pressure,
    int duration_ms,
    int sample_delay_us
) {
    struct shared_state *state = NULL;
    if (posix_memalign((void **)&state, 64, sizeof(*state)) != 0 || !state) {
        fprintf(stderr, "posix_memalign failed\n");
        exit(2);
    }
    memset(state, 0, sizeof(*state));

    struct worker_args a = {state, core_a, 0, false_share, atomic_pressure};
    struct worker_args b = {state, core_b, 1, false_share, atomic_pressure};
    pthread_t ta;
    pthread_t tb;
    if (pthread_create(&ta, NULL, worker_main, &a) != 0 ||
        pthread_create(&tb, NULL, worker_main, &b) != 0) {
        fprintf(stderr, "pthread_create failed\n");
        exit(2);
    }

    pin_core(sample_core);
    struct sample samples[MAX_SAMPLES];
    int sample_count = 0;
    uint64_t end_ns = monotonic_ns() + (uint64_t)duration_ms * 1000000ull;
    while (monotonic_ns() < end_ns && sample_count < MAX_SAMPLES) {
        if (false_share) {
            samples[sample_count].a = __atomic_load_n(&state->counter_a, __ATOMIC_ACQUIRE);
            samples[sample_count].b = __atomic_load_n(&state->counter_b, __ATOMIC_ACQUIRE);
        } else {
            samples[sample_count].a = __atomic_load_n(&state->isolated_a, __ATOMIC_ACQUIRE);
            samples[sample_count].b = __atomic_load_n(&state->isolated_b, __ATOMIC_ACQUIRE);
        }
        sample_count++;
        if (sample_delay_us > 0) {
            usleep((useconds_t)sample_delay_us);
        }
    }

    __atomic_store_n(&state->stop, 1, __ATOMIC_RELAXED);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);

    double real_r = phase_r(samples, sample_count, 0);
    double null_1 = phase_r(samples, sample_count, 17 % sample_count);
    double null_2 = phase_r(samples, sample_count, 257 % sample_count);
    double null_3 = phase_r(samples, sample_count, sample_count / 2);
    double max_null = fmax(null_1, fmax(null_2, null_3));
    double step_diff = mean_abs_delta_step(samples, sample_count);
    uint64_t final_a = sample_count ? samples[sample_count - 1].a : 0;
    uint64_t final_b = sample_count ? samples[sample_count - 1].b : 0;

    printf("%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%llu,%llu\n",
           mode,
           repeat,
           core_a,
           core_b,
           sample_core,
           sample_count,
           real_r,
           null_1,
           null_2,
           null_3,
           max_null,
           step_diff,
           (unsigned long long)final_a,
           (unsigned long long)final_b);

    free(state);
}

int main(int argc, char **argv) {
    int core_a = 3;
    int core_b = 4;
    int sample_core = 2;
    int repeats = 8;
    int duration_ms = 220;
    int sample_delay_us = 25;

    if (argc > 1) {
        repeats = atoi(argv[1]);
    }
    if (argc > 2) {
        duration_ms = atoi(argv[2]);
    }

    printf("mode,repeat,core_a,core_b,sample_core,samples,real_r,null_shift17_r,null_shift257_r,null_half_r,max_null_r,mean_abs_delta_step,final_a,final_b\n");
    for (int repeat = 0; repeat < repeats; repeat++) {
        run_case("isolated_lines", core_a, core_b, sample_core, repeat, 0, 0, duration_ms, sample_delay_us);
        run_case("false_shared_line", core_a, core_b, sample_core, repeat, 1, 0, duration_ms, sample_delay_us);
        run_case("atomic_same_line", core_a, core_b, sample_core, repeat, 1, 1, duration_ms, sample_delay_us);
    }
    return 0;
}
