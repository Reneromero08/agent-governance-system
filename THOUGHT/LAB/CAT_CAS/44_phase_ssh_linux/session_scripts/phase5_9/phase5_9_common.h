#ifndef PHASE5_9_COMMON_H
#define PHASE5_9_COMMON_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <errno.h>
#include <stdatomic.h>

/* --- Timing primitives --- */
static inline uint64_t rdtsc_start(void) {
    unsigned hi, lo;
    __asm__ volatile(
        "cpuid\n\t"
        "rdtsc\n\t"
        : "=a"(lo), "=d"(hi)
        :
        : "%rbx", "%rcx"
    );
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtsc_end(void) {
    unsigned hi, lo;
    __asm__ volatile(
        "rdtscp\n\t"
        "mov %%eax, %0\n\t"
        "mov %%edx, %1\n\t"
        "cpuid\n\t"
        : "=r"(lo), "=r"(hi)
        :
        : "%rax", "%rbx", "%rcx", "%rdx"
    );
    return ((uint64_t)hi << 32) | lo;
}

static inline void clflush(volatile void *p) {
    __asm__ volatile("clflush (%0)" :: "r"(p));
}

/* --- FNV-1a 64-bit checksum --- */
#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL

static inline uint64_t fnv1a_64(const uint8_t *data, size_t len) {
    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)data[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

/* --- Data structures --- */
#define MAX_TAPE_SIZE 32768
#define MAX_WORKERS 16
#define CACHE_LINE 64
#define CATALYTIC_ALIGN 64

typedef enum {
    WORKER_MODE_NONE = 0,
    WORKER_MODE_CACHE_HAMMER,
    WORKER_MODE_INTEGER_CHURN,
    WORKER_MODE_MIXED,
    WORKER_MODE_THERMAL
} worker_mode_t;

typedef enum {
    PLACEMENT_OFFCORE = 0,
    PLACEMENT_SAME_L3,
    PLACEMENT_SAME_CORE,
    PLACEMENT_NO_WORKERS
} worker_placement_t;

typedef enum {
    CONTROL_EMPTY_TIMING = 0,
    CONTROL_NOP_LOOP,
    CONTROL_IRREVERSIBLE,
    CONTROL_READ_ONLY,
    CONTROL_SHUFFLED,
    CONTROL_SYNTHETIC_CLOUD,
    CONTROL_TRIAL_ORDER,
    CONTROL_MIGRATION_AUDIT
} control_type_t;

typedef struct {
    char run_id[128];
    int measurement_core;
    int tape_size;
    int window_size;
    int iterations;
    worker_mode_t worker_mode;
    worker_placement_t worker_placement;
    int worker_count;
    int worker_cores[MAX_WORKERS];
    char freq_label[64];
    char vid_label[64];
    double measured_vcore;
    int timing_mode;  /* 1=RDTSCP, 2=CLOCK_MONOTONIC_RAW */
} run_config_t;

typedef struct {
    int trial_id;
    int trial_order;
    int iteration_index;
    uint64_t rdtsc_cycles_raw;
    uint64_t rdtsc_cycles_corrected;
    uint64_t clock_ns;
    int restore_ok;
    int timing_mode;
    uint64_t initial_checksum;
    uint64_t final_checksum;
    int hash_match;
} trial_row_t;

typedef struct {
    int trial_count;
    int restore_pass_count;
    uint64_t rdtsc_overhead_cycles;
    uint64_t nop_loop_cycles;
    uint64_t catalytic_loop_cycles;
    uint64_t total_raw_cycles_min;
    uint64_t total_raw_cycles_max;
    double total_raw_cycles_mean;
    double total_raw_cycles_std;
} telemetry_t;

typedef struct {
    atomic_int stop;
    pthread_t thread;
    int core_id;
    int worker_id;
    int start_ok;
    int join_ok;
    worker_mode_t mode;
    size_t buffer_size;
    size_t buffer_mb_actual;  /* actual MB allocated after fallback */
    int stride;
    uint64_t *buffer;  /* aligned, no mlock */
    uint64_t seed;
    int mlock_used;     /* 1 if buffer was mlock'd */
} worker_state_t;

/* --- CPU affinity --- */
static inline int pin_to_core(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("sched_setaffinity");
        return -1;
    }
    return 0;
}

/* --- Memory utilities --- */
/* Allocate page-touched aligned memory WITHOUT mlock.
   Use for worker hammer buffers to avoid mlock limit exhaustion. */
static inline void *aligned_alloc_touched(size_t size, size_t align) {
    void *buf = NULL;
    if (posix_memalign(&buf, align, size) != 0) {
        return NULL;
    }
    memset(buf, 0, size);
    volatile char *vp = (volatile char *)buf;
    for (size_t i = 0; i < size; i += 4096) {
        vp[i] = (char)i;
    }
    return buf;
}

/* Allocate page-touched aligned memory WITH mlock attempt.
   mlock failure is non-fatal. Use for tape/key/backup. */
static inline void *aligned_alloc_locked(size_t size, size_t align) {
    void *buf = aligned_alloc_touched(size, align);
    if (buf) {
        /* Try to lock; non-fatal if fails */
        mlock(buf, size);
    }
    return buf;
}

/* --- File write helper --- */
static inline FILE *open_csv(const char *dir, const char *name, const char *headers) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        return NULL;
    }
    fprintf(f, "%s\n", headers);
    return f;
}

#endif /* PHASE5_9_COMMON_H */
