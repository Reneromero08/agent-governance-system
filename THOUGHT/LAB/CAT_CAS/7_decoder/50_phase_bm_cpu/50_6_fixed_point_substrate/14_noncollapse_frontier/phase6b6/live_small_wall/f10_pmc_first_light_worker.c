#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define CATCAS_CORE_A 4
#define CATCAS_CORE_B 5
#define CACHE_LINE_BYTES 64
#define CARRIER_LINES 4096
#define READ_ITERATIONS 512
#define WRITE_ITERATIONS 512
#define PINGPONG_ITERATIONS 192
#define OPERATOR_ITERATIONS 96
#define MAX_GROUP_EVENTS 4
#define SUPPORT_EVENTS 8
#define RESULT_SCHEMA "CAT_CAS_F10_PMC_FIRST_LIGHT_RESULT_V1"
#define COHERENCE_RESULT_SCHEMA "CAT_CAS_F10_COHERENCE_OPERATOR_RESULT_V1"
#define PATH_RESULT_SCHEMA "CAT_CAS_F10_PATH_DEPENDENCE_PILOT_RESULT_V1"
#define PATH_DUAL_OBSERVE_RESULT_SCHEMA "CAT_CAS_F10_PATH_DUAL_OBSERVE_RESULT_V1"

struct event_def {
    const char *name;
    unsigned int event_select;
    unsigned int unit_mask;
    const char *kind;
};

struct event_result {
    uint64_t value;
    uint64_t id;
};

struct group_result {
    bool opened;
    bool unmultiplexed;
    int open_errno;
    uint64_t time_enabled;
    uint64_t time_running;
    struct event_result events[MAX_GROUP_EVENTS];
};

struct carrier {
    unsigned char *bytes;
    size_t byte_count;
    size_t line_count;
};

struct pingpong_context {
    struct carrier *carrier;
    atomic_int turn;
    atomic_int ready;
    atomic_int done;
    volatile uint64_t sink_a;
    volatile uint64_t sink_b;
};

enum coherence_operator {
    OP_IDENTITY = 0,
    OP_SAME_CORE_PREFETCHW = 1,
    OP_REMOTE_READ_SHARED = 2,
    OP_REMOTE_PREFETCHW = 3,
    OP_SAME_CORE_LOCKED_NOOP = 4,
    OP_REMOTE_LOCKED_NOOP = 5,
    OP_SAME_CORE_STORE_SAME_VALUE = 6,
    OP_REMOTE_STORE_SAME_VALUE = 7
};

struct operator_context {
    struct carrier *carrier;
    enum coherence_operator op;
    atomic_int ready;
    atomic_int done;
    volatile uint64_t sink;
};

enum path_op {
    PATH_REMOTE_STORE = 0,
    PATH_HOME_STORE = 1
};

struct path_step {
    const char *name;
    enum path_op op;
    int line_set;
};

static const struct event_def support_events[SUPPORT_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"dc_refills_l2_or_nb_all_states", 0x042, 0x1f, "core"},
    {"dc_refills_from_nb_all_states", 0x043, 0x1f, "core"},
    {"dc_refills_from_nb_owned_modified", 0x043, 0x18, "core"},
    {"dc_lines_evicted_modified", 0x044, 0x10, "core"},
    {"cache_block_commands_change_to_dirty", 0x0ea, 0x20, "northbridge"},
    {"probe_responses_dirty", 0x0ec, 0x0c, "northbridge"},
    {"locked_ops_executed", 0x024, 0x01, "core"}
};

static const struct event_def primary_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"dc_refills_from_nb_all_states", 0x043, 0x1f, "core"},
    {"cache_block_commands_change_to_dirty", 0x0ea, 0x20, "northbridge"},
    {"probe_responses_dirty", 0x0ec, 0x0c, "northbridge"}
};

static const struct event_def fallback_group[MAX_GROUP_EVENTS] = {
    {"cpu_cycles_not_halted", 0x076, 0x00, "core"},
    {"dc_refills_l2_or_nb_all_states", 0x042, 0x1f, "core"},
    {"dc_refills_from_nb_all_states", 0x043, 0x1f, "core"},
    {"dc_lines_evicted_modified", 0x044, 0x10, "core"}
};

struct read_group_payload {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    struct {
        uint64_t value;
        uint64_t id;
    } values[MAX_GROUP_EVENTS];
};

static long perf_event_open_call(
    struct perf_event_attr *attr,
    pid_t pid,
    int cpu,
    int group_fd,
    unsigned long flags
) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static uint64_t raw_config(unsigned int event_select, unsigned int unit_mask) {
    return (uint64_t)(event_select & 0xffu) |
           ((uint64_t)(unit_mask & 0xffu) << 8) |
           ((uint64_t)(event_select & 0xf00u) << 24);
}

static int self_test(void) {
    if (raw_config(0x076u, 0x00u) != 0x76ull) return 1;
    if (raw_config(0x0eau, 0x20u) != 0x20eaull) return 1;
    if (raw_config(0x0ecu, 0x0cu) != 0x0cecull) return 1;
    if (raw_config(0x1abu, 0xcdu) != 0x10000cdabull) return 1;
    printf("F10_PMC_FIRST_LIGHT_WORKER_SELF_TEST_OK\n");
    return 0;
}

static void fill_attr(struct perf_event_attr *attr, const struct event_def *event, bool disabled) {
    memset(attr, 0, sizeof(*attr));
    attr->type = PERF_TYPE_RAW;
    attr->size = sizeof(*attr);
    attr->config = raw_config(event->event_select, event->unit_mask);
    attr->disabled = disabled ? 1u : 0u;
    attr->exclude_kernel = 1u;
    attr->exclude_hv = 1u;
    attr->read_format = PERF_FORMAT_GROUP |
        PERF_FORMAT_TOTAL_TIME_ENABLED |
        PERF_FORMAT_TOTAL_TIME_RUNNING |
        PERF_FORMAT_ID;
}

static int pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t fnv1a64(const unsigned char *data, size_t len) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static unsigned char carrier_pattern_byte(size_t index) {
    return (unsigned char)((index * 131u + 17u) & 0xffu);
}

static void init_carrier(struct carrier *carrier) {
    for (size_t i = 0; i < carrier->byte_count; i++) {
        carrier->bytes[i] = carrier_pattern_byte(i);
    }
}

static void prefault_carrier(struct carrier *carrier) {
    volatile unsigned char sink = 0;
    for (size_t line = 0; line < carrier->line_count; line++) {
        unsigned char *p = carrier->bytes + line * CACHE_LINE_BYTES;
        sink ^= *p;
        *p = (unsigned char)(*p ^ 0u);
    }
    if (sink == 255u) {
        carrier->bytes[0] ^= sink;
    }
}

__attribute__((noinline, noclone))
static void restore_on_core(struct carrier *carrier, int core) {
    if (pin_to_core(core) != 0) return;
    for (size_t line = 0; line < carrier->line_count; line++) {
        size_t base = line * CACHE_LINE_BYTES;
        for (size_t offset = 0; offset < CACHE_LINE_BYTES; offset++) {
            volatile unsigned char *p = (volatile unsigned char *)(void *)(carrier->bytes + base + offset);
            *p = carrier_pattern_byte(base + offset);
        }
    }
}

static void home_core_restore(struct carrier *carrier) {
    restore_on_core(carrier, CATCAS_CORE_A);
}

__attribute__((noinline, noclone))
static void same_core_prefetchw(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            void *p = carrier->bytes + line * CACHE_LINE_BYTES;
            __asm__ __volatile__("prefetchw (%0)" : : "r"(p) : "memory");
        }
    }
}

__attribute__((noinline, noclone))
static void same_core_locked_noop(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            __asm__ __volatile__("lock orq $0, %0" : "+m"(*p) : : "memory", "cc");
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void same_core_store_same_value(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            uint64_t value = *p;
            *p = value;
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void remote_read_shared(struct carrier *carrier) {
    volatile uint64_t sink = 0;
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            sink += *p;
        }
    }
    if (sink == 0x12345678ull) carrier->bytes[0] ^= 1u;
}

__attribute__((noinline, noclone))
static void remote_prefetchw(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            void *p = carrier->bytes + line * CACHE_LINE_BYTES;
            __asm__ __volatile__("prefetchw (%0)" : : "r"(p) : "memory");
        }
    }
}

__attribute__((noinline, noclone))
static void remote_locked_noop(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            __asm__ __volatile__("lock orq $0, %0" : "+m"(*p) : : "memory", "cc");
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void remote_store_same_value(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_B) != 0) return;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            uint64_t value = *p;
            *p = value;
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

__attribute__((noinline, noclone))
static void store_same_value_subset_on_core(struct carrier *carrier, int core, int line_set) {
    if (pin_to_core(core) != 0) return;
    size_t start = line_set == 0 ? 0u : 1u;
    for (int iter = 0; iter < OPERATOR_ITERATIONS; iter++) {
        for (size_t line = start; line < carrier->line_count; line += 2u) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            uint64_t value = *p;
            *p = value;
        }
    }
    __asm__ __volatile__("mfence" : : : "memory");
}

static int run_path_step_operator(struct carrier *carrier, const struct path_step *step) {
    if (step->line_set != 0 && step->line_set != 1) return -1;
    if (step->op == PATH_REMOTE_STORE) {
        store_same_value_subset_on_core(carrier, CATCAS_CORE_B, step->line_set);
        return 0;
    }
    if (step->op == PATH_HOME_STORE) {
        store_same_value_subset_on_core(carrier, CATCAS_CORE_A, step->line_set);
        return 0;
    }
    return -1;
}

static int path_step_actor_core(const struct path_step *step) {
    if (step->op == PATH_REMOTE_STORE) return CATCAS_CORE_B;
    if (step->op == PATH_HOME_STORE) return CATCAS_CORE_A;
    return -1;
}

static void idle_pause(void) {
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 100000000L;
    nanosleep(&ts, NULL);
}

__attribute__((noinline, noclone))
static void core4_read_sweep(struct carrier *carrier) {
    volatile uint64_t sink = 0;
    if (pin_to_core(CATCAS_CORE_A) != 0) return;
    for (int iter = 0; iter < READ_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            sink += *p;
        }
    }
    if (sink == 0xfeedfaceull) carrier->bytes[0] ^= 1u;
}

__attribute__((noinline, noclone))
static void core4_write_sweep(struct carrier *carrier) {
    if (pin_to_core(CATCAS_CORE_A) != 0) return;
    for (int iter = 0; iter < WRITE_ITERATIONS; iter++) {
        for (size_t line = 0; line < carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(carrier->bytes + line * CACHE_LINE_BYTES);
            *p += 1u;
        }
    }
}

static void *pingpong_core_a(void *arg) {
    struct pingpong_context *ctx = (struct pingpong_context *)arg;
    if (pin_to_core(CATCAS_CORE_A) != 0) {
        atomic_store(&ctx->done, 1);
        return NULL;
    }
    atomic_fetch_add(&ctx->ready, 1);
    for (int iter = 0; iter < PINGPONG_ITERATIONS; iter++) {
        while (atomic_load_explicit(&ctx->turn, memory_order_acquire) != 0) {
            if (atomic_load_explicit(&ctx->done, memory_order_acquire) != 0) {
                return NULL;
            }
            sched_yield();
        }
        if (atomic_load_explicit(&ctx->done, memory_order_acquire) != 0) {
            return NULL;
        }
        for (size_t line = 0; line < ctx->carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(ctx->carrier->bytes + line * CACHE_LINE_BYTES);
            *p += 3u;
            ctx->sink_a += *p;
        }
        atomic_store_explicit(&ctx->turn, 1, memory_order_release);
    }
    return NULL;
}

static void *pingpong_core_b(void *arg) {
    struct pingpong_context *ctx = (struct pingpong_context *)arg;
    if (pin_to_core(CATCAS_CORE_B) != 0) {
        atomic_store(&ctx->done, 1);
        return NULL;
    }
    atomic_fetch_add(&ctx->ready, 1);
    for (int iter = 0; iter < PINGPONG_ITERATIONS; iter++) {
        while (atomic_load_explicit(&ctx->turn, memory_order_acquire) != 1) {
            if (atomic_load_explicit(&ctx->done, memory_order_acquire) != 0) {
                return NULL;
            }
            sched_yield();
        }
        if (atomic_load_explicit(&ctx->done, memory_order_acquire) != 0) {
            return NULL;
        }
        for (size_t line = 0; line < ctx->carrier->line_count; line++) {
            volatile uint64_t *p = (volatile uint64_t *)(void *)(ctx->carrier->bytes + line * CACHE_LINE_BYTES);
            *p += 5u;
            ctx->sink_b += *p;
        }
        atomic_store_explicit(&ctx->turn, 0, memory_order_release);
    }
    return NULL;
}

__attribute__((noinline, noclone))
static int cross_core_pingpong_write(struct carrier *carrier) {
    pthread_t a;
    pthread_t b;
    struct pingpong_context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.carrier = carrier;
    atomic_init(&ctx.turn, 0);
    atomic_init(&ctx.ready, 0);
    atomic_init(&ctx.done, 0);
    if (pthread_create(&a, NULL, pingpong_core_a, &ctx) != 0) return -1;
    if (pthread_create(&b, NULL, pingpong_core_b, &ctx) != 0) {
        atomic_store_explicit(&ctx.done, 1, memory_order_release);
        pthread_join(a, NULL);
        return -1;
    }
    while (atomic_load(&ctx.ready) < 2 && atomic_load(&ctx.done) == 0) {
        sched_yield();
    }
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    return atomic_load(&ctx.done) == 0 ? 0 : -1;
}

static const char *coherence_operator_name(enum coherence_operator op) {
    switch (op) {
        case OP_IDENTITY: return "identity_home_prepared";
        case OP_SAME_CORE_PREFETCHW: return "same_core_prefetchw_control";
        case OP_REMOTE_READ_SHARED: return "remote_read_shared";
        case OP_REMOTE_PREFETCHW: return "remote_prefetchw_ownership_request";
        case OP_SAME_CORE_LOCKED_NOOP: return "same_core_locked_logical_noop_control";
        case OP_REMOTE_LOCKED_NOOP: return "remote_locked_logical_noop";
        case OP_SAME_CORE_STORE_SAME_VALUE: return "same_core_store_same_value_control";
        case OP_REMOTE_STORE_SAME_VALUE: return "remote_store_same_value";
        default: return "unknown_operator";
    }
}

static void *remote_operator_thread(void *arg) {
    struct operator_context *ctx = (struct operator_context *)arg;
    atomic_store_explicit(&ctx->ready, 1, memory_order_release);
    if (ctx->op == OP_REMOTE_READ_SHARED) {
        remote_read_shared(ctx->carrier);
    } else if (ctx->op == OP_REMOTE_PREFETCHW) {
        remote_prefetchw(ctx->carrier);
    } else if (ctx->op == OP_REMOTE_LOCKED_NOOP) {
        remote_locked_noop(ctx->carrier);
    } else if (ctx->op == OP_REMOTE_STORE_SAME_VALUE) {
        remote_store_same_value(ctx->carrier);
    } else {
        atomic_store_explicit(&ctx->done, 1, memory_order_release);
        return NULL;
    }
    atomic_store_explicit(&ctx->done, 1, memory_order_release);
    return NULL;
}

static int run_coherence_operator(struct carrier *carrier, enum coherence_operator op) {
    if (op == OP_IDENTITY) return 0;
    if (op == OP_SAME_CORE_PREFETCHW) {
        same_core_prefetchw(carrier);
        return 0;
    }
    if (op == OP_SAME_CORE_LOCKED_NOOP) {
        same_core_locked_noop(carrier);
        return 0;
    }
    if (op == OP_SAME_CORE_STORE_SAME_VALUE) {
        same_core_store_same_value(carrier);
        return 0;
    }
    if (op == OP_REMOTE_READ_SHARED || op == OP_REMOTE_PREFETCHW ||
        op == OP_REMOTE_LOCKED_NOOP || op == OP_REMOTE_STORE_SAME_VALUE) {
        pthread_t thread;
        struct operator_context ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.carrier = carrier;
        ctx.op = op;
        atomic_init(&ctx.ready, 0);
        atomic_init(&ctx.done, 0);
        if (pthread_create(&thread, NULL, remote_operator_thread, &ctx) != 0) return -1;
        while (atomic_load_explicit(&ctx.ready, memory_order_acquire) == 0) {
            sched_yield();
        }
        pthread_join(thread, NULL);
        return atomic_load_explicit(&ctx.done, memory_order_acquire) == 1 ? 0 : -1;
    }
    return -1;
}

static int open_single_event(const struct event_def *event, int core, uint64_t *id_out) {
    struct perf_event_attr attr;
    fill_attr(&attr, event, true);
    int fd = (int)perf_event_open_call(&attr, -1, core, -1, PERF_FLAG_FD_CLOEXEC);
    if (fd < 0) return -errno;
    uint64_t id = 0;
    if (ioctl(fd, PERF_EVENT_IOC_ID, &id) == 0 && id_out) *id_out = id;
    close(fd);
    return 0;
}

static int open_group(const struct event_def group[MAX_GROUP_EVENTS], int core, int fds[MAX_GROUP_EVENTS], uint64_t ids[MAX_GROUP_EVENTS]) {
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        fds[i] = -1;
        ids[i] = 0;
    }
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        struct perf_event_attr attr;
        fill_attr(&attr, &group[i], i == 0);
        int group_fd = i == 0 ? -1 : fds[0];
        fds[i] = (int)perf_event_open_call(&attr, -1, core, group_fd, PERF_FLAG_FD_CLOEXEC);
        if (fds[i] < 0) {
            int saved = errno;
            for (int j = 0; j < i; j++) close(fds[j]);
            return -saved;
        }
        if (ioctl(fds[i], PERF_EVENT_IOC_ID, &ids[i]) != 0) {
            int saved = errno;
            for (int j = 0; j <= i; j++) close(fds[j]);
            return -saved;
        }
    }
    return 0;
}

static int read_group(int leader_fd, struct group_result *result) {
    struct read_group_payload payload;
    memset(&payload, 0, sizeof(payload));
    ssize_t got = read(leader_fd, &payload, sizeof(payload));
    if (got < 0) return -errno;
    if ((size_t)got < sizeof(uint64_t) * 3u) return -EIO;
    if (payload.nr != MAX_GROUP_EVENTS) return -EIO;
    result->time_enabled = payload.time_enabled;
    result->time_running = payload.time_running;
    result->unmultiplexed = payload.time_enabled == payload.time_running;
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        result->events[i].value = payload.values[i].value;
        result->events[i].id = payload.values[i].id;
    }
    return 0;
}

static int measure_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    const char *window_name,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    init_carrier(carrier);
    prefault_carrier(carrier);
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, CATCAS_CORE_A, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int work_rc = 0;
    if (strcmp(window_name, "idle_pause") == 0) {
        idle_pause();
    } else if (strcmp(window_name, "core4_read_sweep") == 0) {
        core4_read_sweep(carrier);
    } else if (strcmp(window_name, "core4_write_sweep") == 0) {
        core4_write_sweep(carrier);
    } else if (strcmp(window_name, "cross_core_pingpong_write") == 0) {
        work_rc = cross_core_pingpong_write(carrier);
    } else {
        work_rc = -1;
    }
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (work_rc != 0) return -EIO;
    *duration_ns = end >= start ? end - start : 0;
    init_carrier(carrier);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static int measure_coherence_window(
    const struct event_def group[MAX_GROUP_EVENTS],
    enum coherence_operator op,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after_restore
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    int prep_core = (op == OP_SAME_CORE_PREFETCHW ||
        op == OP_SAME_CORE_LOCKED_NOOP ||
        op == OP_SAME_CORE_STORE_SAME_VALUE) ? CATCAS_CORE_B : CATCAS_CORE_A;
    restore_on_core(carrier, prep_core);
    prefault_carrier(carrier);
    restore_on_core(carrier, prep_core);
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, CATCAS_CORE_B, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int work_rc = run_coherence_operator(carrier, op);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (work_rc != 0) return -EIO;
    *duration_ns = end >= start ? end - start : 0;
    home_core_restore(carrier);
    *digest_after_restore = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static int measure_path_step(
    const struct event_def group[MAX_GROUP_EVENTS],
    const struct path_step *step,
    int observed_core,
    struct carrier *carrier,
    struct group_result *result,
    uint64_t *duration_ns,
    uint64_t *digest_before,
    uint64_t *digest_after
) {
    int fds[MAX_GROUP_EVENTS];
    uint64_t ids[MAX_GROUP_EVENTS];
    memset(result, 0, sizeof(*result));
    *digest_before = fnv1a64(carrier->bytes, carrier->byte_count);
    int opened = open_group(group, observed_core, fds, ids);
    if (opened != 0) {
        result->open_errno = -opened;
        return opened;
    }
    result->opened = true;
    if (ioctl(fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t start = monotonic_ns();
    if (ioctl(fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    int work_rc = run_path_step_operator(carrier, step);
    if (ioctl(fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
        int saved = errno;
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
        return -saved;
    }
    uint64_t end = monotonic_ns();
    int read_rc = read_group(fds[0], result);
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fds[i]);
    if (read_rc != 0) return read_rc;
    if (work_rc != 0) return -EIO;
    *duration_ns = end >= start ? end - start : 0;
    *digest_after = fnv1a64(carrier->bytes, carrier->byte_count);
    return 0;
}

static uint64_t event_value_by_name(const struct group_result *result, const struct event_def group[MAX_GROUP_EVENTS], const char *name);

static long double normalized_event_value(
    const struct group_result *result,
    const struct event_def group[MAX_GROUP_EVENTS],
    const char *name
) {
    uint64_t cycles = event_value_by_name(result, group, "cpu_cycles_not_halted");
    if (cycles == 0) cycles = 1;
    return (long double)event_value_by_name(result, group, name) / (long double)cycles;
}

static long double path_signed_area(
    const struct group_result *results,
    const struct event_def group[MAX_GROUP_EVENTS],
    int count
) {
    long double area = 0.0L;
    for (int i = 0; i < count; i++) {
        int next = (i + 1) % count;
        long double x0 = normalized_event_value(&results[i], group, "cache_block_commands_change_to_dirty");
        long double y0 = normalized_event_value(&results[i], group, "probe_responses_dirty");
        long double x1 = normalized_event_value(&results[next], group, "cache_block_commands_change_to_dirty");
        long double y1 = normalized_event_value(&results[next], group, "probe_responses_dirty");
        area += x0 * y1 - y0 * x1;
    }
    return area * 0.5L;
}

static long double abs_ld(long double value) {
    return value < 0.0L ? -value : value;
}

static const char *json_bool(bool value) {
    return value ? "true" : "false";
}

static void print_event_def(FILE *out, const struct event_def *event) {
    fprintf(
        out,
        "{\"name\":\"%s\",\"event_select_hex\":\"0x%03x\",\"unit_mask_hex\":\"0x%02x\",\"raw_config_hex\":\"0x%llx\",\"kind\":\"%s\"}",
        event->name,
        event->event_select,
        event->unit_mask,
        (unsigned long long)raw_config(event->event_select, event->unit_mask),
        event->kind
    );
}

static void print_group_defs(FILE *out, const struct event_def group[MAX_GROUP_EVENTS]) {
    fprintf(out, "[");
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        if (i) fprintf(out, ",");
        print_event_def(out, &group[i]);
    }
    fprintf(out, "]");
}

static void print_group_result(FILE *out, const struct event_def group[MAX_GROUP_EVENTS], const struct group_result *result) {
    fprintf(
        out,
        "{\"opened\":%s,\"unmultiplexed\":%s,\"open_errno\":%d,\"time_enabled\":%" PRIu64 ",\"time_running\":%" PRIu64 ",\"counts\":[",
        json_bool(result->opened),
        json_bool(result->unmultiplexed),
        result->open_errno,
        result->time_enabled,
        result->time_running
    );
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        if (i) fprintf(out, ",");
        fprintf(
            out,
            "{\"name\":\"%s\",\"id\":%" PRIu64 ",\"value\":%" PRIu64 "}",
            group[i].name,
            result->events[i].id,
            result->events[i].value
        );
    }
    fprintf(out, "]}");
}

static uint64_t event_value_by_name(const struct group_result *result, const struct event_def group[MAX_GROUP_EVENTS], const char *name) {
    for (int i = 0; i < MAX_GROUP_EVENTS; i++) {
        if (strcmp(group[i].name, name) == 0) return result->events[i].value;
    }
    return 0;
}

static int ensure_dir(const char *path) {
    if (mkdir(path, 0700) == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
}

static int run_coherence_operator_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_B, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }

    const enum coherence_operator ops[] = {
        OP_IDENTITY,
        OP_SAME_CORE_PREFETCHW,
        OP_REMOTE_READ_SHARED,
        OP_REMOTE_PREFETCHW,
        OP_SAME_CORE_LOCKED_NOOP,
        OP_REMOTE_LOCKED_NOOP,
        OP_SAME_CORE_STORE_SAME_VALUE,
        OP_REMOTE_STORE_SAME_VALUE
    };
    enum { OP_WINDOW_COUNT = 8 };
    struct group_result results[OP_WINDOW_COUNT];
    uint64_t durations[OP_WINDOW_COUNT];
    uint64_t digest_before[OP_WINDOW_COUNT];
    uint64_t digest_after[OP_WINDOW_COUNT];
    int window_rc[OP_WINDOW_COUNT];
    for (int i = 0; i < OP_WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_before[i] = 0;
        digest_after[i] = 0;
        if (primary_open_rc != 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_coherence_window(
                primary_group,
                ops[i],
                carrier,
                &results[i],
                &durations[i],
                &digest_before[i],
                &digest_after[i]
            );
        }
    }

    bool all_windows_ok = true;
    bool all_unmultiplexed = true;
    bool all_restored = true;
    for (int i = 0; i < OP_WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        all_restored = all_restored && digest_after[i] == initial_digest;
    }

    uint64_t identity_c2d = event_value_by_name(&results[0], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t same_prefetch_c2d = event_value_by_name(&results[1], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t remote_read_c2d = event_value_by_name(&results[2], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t remote_prefetch_c2d = event_value_by_name(&results[3], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t same_locked_c2d = event_value_by_name(&results[4], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t remote_locked_c2d = event_value_by_name(&results[5], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t same_store_c2d = event_value_by_name(&results[6], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t remote_store_c2d = event_value_by_name(&results[7], primary_group, "cache_block_commands_change_to_dirty");
    uint64_t identity_probe = event_value_by_name(&results[0], primary_group, "probe_responses_dirty");
    uint64_t same_prefetch_probe = event_value_by_name(&results[1], primary_group, "probe_responses_dirty");
    uint64_t remote_read_probe = event_value_by_name(&results[2], primary_group, "probe_responses_dirty");
    uint64_t remote_prefetch_probe = event_value_by_name(&results[3], primary_group, "probe_responses_dirty");
    uint64_t same_locked_probe = event_value_by_name(&results[4], primary_group, "probe_responses_dirty");
    uint64_t remote_locked_probe = event_value_by_name(&results[5], primary_group, "probe_responses_dirty");
    uint64_t same_store_probe = event_value_by_name(&results[6], primary_group, "probe_responses_dirty");
    uint64_t remote_store_probe = event_value_by_name(&results[7], primary_group, "probe_responses_dirty");

    uint64_t prefetch_c2d_control = identity_c2d > same_prefetch_c2d ? identity_c2d : same_prefetch_c2d;
    uint64_t prefetch_probe_control = identity_probe > same_prefetch_probe ? identity_probe : same_prefetch_probe;
    uint64_t locked_c2d_control = same_locked_c2d > remote_read_c2d ? same_locked_c2d : remote_read_c2d;
    if (identity_c2d > locked_c2d_control) locked_c2d_control = identity_c2d;
    uint64_t locked_probe_control = same_locked_probe > remote_read_probe ? same_locked_probe : remote_read_probe;
    if (identity_probe > locked_probe_control) locked_probe_control = identity_probe;
    uint64_t store_c2d_control = same_store_c2d > remote_read_c2d ? same_store_c2d : remote_read_c2d;
    if (identity_c2d > store_c2d_control) store_c2d_control = identity_c2d;
    uint64_t store_probe_control = same_store_probe > remote_read_probe ? same_store_probe : remote_read_probe;
    if (identity_probe > store_probe_control) store_probe_control = identity_probe;

    bool prefetch_c2d_moved = remote_prefetch_c2d > prefetch_c2d_control + 32u &&
        remote_prefetch_c2d > prefetch_c2d_control * 3u;
    bool prefetch_probe_moved = remote_prefetch_probe > prefetch_probe_control + 32u &&
        remote_prefetch_probe > prefetch_probe_control * 3u;
    bool locked_c2d_moved = remote_locked_c2d > locked_c2d_control + 32u &&
        remote_locked_c2d > locked_c2d_control * 3u;
    bool locked_probe_moved = remote_locked_probe > locked_probe_control + 32u &&
        remote_locked_probe > locked_probe_control * 3u;
    bool store_c2d_moved = remote_store_c2d > store_c2d_control + 32u &&
        remote_store_c2d > store_c2d_control * 3u;
    bool store_probe_moved = remote_store_probe > store_probe_control + 32u &&
        remote_store_probe > store_probe_control * 3u;
    bool controlled_state_found = primary_open_rc == 0 && all_windows_ok && all_unmultiplexed &&
        all_restored && (prefetch_c2d_moved || prefetch_probe_moved || locked_c2d_moved ||
        locked_probe_moved || store_c2d_moved || store_probe_moved);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_COHERENCE_OPERATOR_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", COHERENCE_RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", controlled_state_found ? "CONTROLLED_COHERENCE_STATE_FOUND" : "CONTROLLED_COHERENCE_STATE_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Controlled coherence-operator PMU discriminator only; no path memory, coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"observed_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"primary\": \"cache_block_commands_change_to_dirty\", \"secondary\": \"probe_responses_dirty\", \"comparison\": \"remote ownership-intent operators greater than identity, read-shared, and same-core controls\"},\n");
    fprintf(out, "  \"operators\": [\n");
    for (int i = 0; i < OP_WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"carrier_restored\": %s, \"group\": ",
            coherence_operator_name(ops[i]),
            window_rc[i],
            durations[i],
            digest_before[i],
            digest_after[i],
            json_bool(digest_after[i] == initial_digest));
        print_group_result(out, primary_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == OP_WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"contrast_counts\": {\n");
    fprintf(out, "    \"change_to_dirty\": {\"identity\": %" PRIu64 ", \"same_core_prefetchw\": %" PRIu64 ", \"remote_read_shared\": %" PRIu64 ", \"remote_prefetchw\": %" PRIu64 ", \"same_core_locked_noop\": %" PRIu64 ", \"remote_locked_noop\": %" PRIu64 ", \"same_core_store_same_value\": %" PRIu64 ", \"remote_store_same_value\": %" PRIu64 "},\n",
        identity_c2d, same_prefetch_c2d, remote_read_c2d, remote_prefetch_c2d, same_locked_c2d, remote_locked_c2d, same_store_c2d, remote_store_c2d);
    fprintf(out, "    \"probe_dirty\": {\"identity\": %" PRIu64 ", \"same_core_prefetchw\": %" PRIu64 ", \"remote_read_shared\": %" PRIu64 ", \"remote_prefetchw\": %" PRIu64 ", \"same_core_locked_noop\": %" PRIu64 ", \"remote_locked_noop\": %" PRIu64 ", \"same_core_store_same_value\": %" PRIu64 ", \"remote_store_same_value\": %" PRIu64 "}\n",
        identity_probe, same_prefetch_probe, remote_read_probe, remote_prefetch_probe, same_locked_probe, remote_locked_probe, same_store_probe, remote_store_probe);
    fprintf(out, "  },\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"carrier_restored\": %s, \"prefetch_change_to_dirty_moved\": %s, \"prefetch_probe_dirty_moved\": %s, \"locked_change_to_dirty_moved\": %s, \"locked_probe_dirty_moved\": %s, \"store_same_value_change_to_dirty_moved\": %s, \"store_same_value_probe_dirty_moved\": %s, \"controlled_state_found\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(all_restored),
        json_bool(prefetch_c2d_moved),
        json_bool(prefetch_probe_moved),
        json_bool(locked_c2d_moved),
        json_bool(locked_probe_moved),
        json_bool(store_c2d_moved),
        json_bool(store_probe_moved),
        json_bool(controlled_state_found));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        controlled_state_found ? "CONTROLLED_COHERENCE_STATE_FOUND" : "CONTROLLED_COHERENCE_STATE_NOT_ESTABLISHED",
        result_path);
    return 0;
}

static int measure_path_sequence(
    const struct path_step steps[4],
    struct carrier *carrier,
    struct group_result results[4],
    uint64_t durations[4],
    uint64_t digest_before[4],
    uint64_t digest_after[4],
    int rc[4],
    uint64_t initial_digest,
    bool observe_actor_core
) {
    home_core_restore(carrier);
    for (int i = 0; i < 4; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_before[i] = 0;
        digest_after[i] = 0;
        int observed_core = observe_actor_core ? path_step_actor_core(&steps[i]) : CATCAS_CORE_B;
        if (observed_core < 0) {
            rc[i] = -EINVAL;
            continue;
        }
        rc[i] = measure_path_step(primary_group, &steps[i], observed_core, carrier, &results[i],
            &durations[i], &digest_before[i], &digest_after[i]);
    }
    home_core_restore(carrier);
    return fnv1a64(carrier->bytes, carrier->byte_count) == initial_digest ? 0 : -1;
}

static void print_path_steps(
    FILE *out,
    const char *name,
    const struct path_step steps[4],
    const struct group_result results[4],
    const uint64_t durations[4],
    const uint64_t digest_before[4],
    const uint64_t digest_after[4],
    const int rc[4],
    long double area,
    uint64_t initial_digest,
    bool observe_actor_core
) {
    fprintf(out, "    {\"name\": \"%s\", \"signed_area_cycles_normalized\": %.12Le, \"steps\": [\n", name, area);
    for (int i = 0; i < 4; i++) {
        int observed_core = observe_actor_core ? path_step_actor_core(&steps[i]) : CATCAS_CORE_B;
        fprintf(out, "      {\"name\": \"%s\", \"line_set\": %d, \"observed_core\": %d, \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_hex\": \"0x%016" PRIx64 "\", \"bytes_unchanged\": %s, \"group\": ",
            steps[i].name,
            steps[i].line_set,
            observed_core,
            rc[i],
            durations[i],
            digest_before[i],
            digest_after[i],
            json_bool(digest_before[i] == initial_digest && digest_after[i] == initial_digest));
        print_group_result(out, primary_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == 4 ? "" : ",");
    }
    fprintf(out, "    ]}");
}

static int probe_primary_group_on_core(int core) {
    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, core, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }
    return primary_open_rc;
}

static int run_path_mode(
    const char *output_root,
    struct carrier *carrier,
    uint64_t initial_digest,
    bool observe_actor_core
) {
    const char *schema_id = observe_actor_core ? PATH_DUAL_OBSERVE_RESULT_SCHEMA : PATH_RESULT_SCHEMA;
    const char *result_file = observe_actor_core ? "F10_PATH_DUAL_OBSERVE_RESULT.json" : "F10_PATH_DEPENDENCE_PILOT_RESULT.json";
    const char *positive_status = observe_actor_core ? "PATH_DUAL_OBSERVE_CANDIDATE" : "PATH_DEPENDENCE_PILOT_OBSERVED";
    const char *negative_status = observe_actor_core ? "PATH_DUAL_OBSERVE_NOT_ESTABLISHED" : "PATH_DEPENDENCE_NOT_ESTABLISHED";
    const char *claim_ceiling = observe_actor_core ?
        "Dual-observed path pilot only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim" :
        "Path-dependence pilot only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim";
    int primary_open_rc = probe_primary_group_on_core(CATCAS_CORE_B);
    int home_open_rc = observe_actor_core ? probe_primary_group_on_core(CATCAS_CORE_A) : 0;
    bool groups_available = primary_open_rc == 0 && home_open_rc == 0;

    const struct path_step forward[4] = {
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"home_store_set0", PATH_HOME_STORE, 0},
        {"home_store_set1", PATH_HOME_STORE, 1}
    };
    const struct path_step reverse[4] = {
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"home_store_set1", PATH_HOME_STORE, 1},
        {"home_store_set0", PATH_HOME_STORE, 0}
    };
    const struct path_step shuffle[4] = {
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"home_store_set0", PATH_HOME_STORE, 0},
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"home_store_set1", PATH_HOME_STORE, 1}
    };
    const struct path_step reverse_shuffle[4] = {
        {"remote_store_set1", PATH_REMOTE_STORE, 1},
        {"home_store_set1", PATH_HOME_STORE, 1},
        {"remote_store_set0", PATH_REMOTE_STORE, 0},
        {"home_store_set0", PATH_HOME_STORE, 0}
    };
    const struct path_step identity[4] = {
        {"home_store_set0", PATH_HOME_STORE, 0},
        {"home_store_set1", PATH_HOME_STORE, 1},
        {"home_store_set0", PATH_HOME_STORE, 0},
        {"home_store_set1", PATH_HOME_STORE, 1}
    };

    struct group_result forward_results[4], reverse_results[4], shuffle_results[4], reverse_shuffle_results[4], identity_results[4];
    uint64_t forward_durations[4], reverse_durations[4], shuffle_durations[4], reverse_shuffle_durations[4], identity_durations[4];
    uint64_t forward_before[4], reverse_before[4], shuffle_before[4], reverse_shuffle_before[4], identity_before[4];
    uint64_t forward_after[4], reverse_after[4], shuffle_after[4], reverse_shuffle_after[4], identity_after[4];
    int forward_rc[4], reverse_rc[4], shuffle_rc[4], reverse_shuffle_rc[4], identity_rc[4];
    int forward_restore = groups_available ? measure_path_sequence(forward, carrier, forward_results, forward_durations, forward_before, forward_after, forward_rc, initial_digest, observe_actor_core) : -ENODEV;
    int reverse_restore = groups_available ? measure_path_sequence(reverse, carrier, reverse_results, reverse_durations, reverse_before, reverse_after, reverse_rc, initial_digest, observe_actor_core) : -ENODEV;
    int shuffle_restore = groups_available ? measure_path_sequence(shuffle, carrier, shuffle_results, shuffle_durations, shuffle_before, shuffle_after, shuffle_rc, initial_digest, observe_actor_core) : -ENODEV;
    int reverse_shuffle_restore = groups_available ? measure_path_sequence(reverse_shuffle, carrier, reverse_shuffle_results, reverse_shuffle_durations, reverse_shuffle_before, reverse_shuffle_after, reverse_shuffle_rc, initial_digest, observe_actor_core) : -ENODEV;
    int identity_restore = groups_available ? measure_path_sequence(identity, carrier, identity_results, identity_durations, identity_before, identity_after, identity_rc, initial_digest, observe_actor_core) : -ENODEV;

    bool all_windows_ok = groups_available;
    bool all_unmultiplexed = groups_available;
    bool bytes_unchanged = forward_restore == 0 && reverse_restore == 0 &&
        shuffle_restore == 0 && reverse_shuffle_restore == 0 && identity_restore == 0;
    for (int i = 0; i < 4; i++) {
        all_windows_ok = all_windows_ok && forward_rc[i] == 0 && reverse_rc[i] == 0 &&
            shuffle_rc[i] == 0 && reverse_shuffle_rc[i] == 0 && identity_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && forward_results[i].unmultiplexed &&
            reverse_results[i].unmultiplexed && shuffle_results[i].unmultiplexed &&
            reverse_shuffle_results[i].unmultiplexed && identity_results[i].unmultiplexed;
        bytes_unchanged = bytes_unchanged &&
            forward_before[i] == initial_digest && forward_after[i] == initial_digest &&
            reverse_before[i] == initial_digest && reverse_after[i] == initial_digest &&
            shuffle_before[i] == initial_digest && shuffle_after[i] == initial_digest &&
            reverse_shuffle_before[i] == initial_digest && reverse_shuffle_after[i] == initial_digest &&
            identity_before[i] == initial_digest && identity_after[i] == initial_digest;
    }

    long double forward_area = path_signed_area(forward_results, primary_group, 4);
    long double reverse_area = path_signed_area(reverse_results, primary_group, 4);
    long double shuffle_area = path_signed_area(shuffle_results, primary_group, 4);
    long double reverse_shuffle_area = path_signed_area(reverse_shuffle_results, primary_group, 4);
    long double identity_area = path_signed_area(identity_results, primary_group, 4);
    long double min_oriented = abs_ld(forward_area) < abs_ld(reverse_area) ? abs_ld(forward_area) : abs_ld(reverse_area);
    bool sign_reversal = (forward_area < 0.0L && reverse_area > 0.0L) ||
        (forward_area > 0.0L && reverse_area < 0.0L);
    bool controls_small = min_oriented > 0.0L &&
        abs_ld(shuffle_area) * 4.0L < min_oriented &&
        abs_ld(reverse_shuffle_area) * 4.0L < min_oriented &&
        abs_ld(identity_area) * 4.0L < min_oriented;
    bool path_pilot = all_windows_ok && all_unmultiplexed && bytes_unchanged &&
        sign_reversal && controls_small;

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/%s", output_root, result_file);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) return 1;
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        return 1;
    }
    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", schema_id);
    fprintf(out, "  \"status\": \"%s\",\n", path_pilot ? positive_status : negative_status);
    fprintf(out, "  \"claim_ceiling\": \"%s\",\n", claim_ceiling);
    fprintf(out, "  \"cores\": {\"home_core\": %d, \"remote_core\": %d, \"fixed_observed_core\": %d, \"observe_actor_core\": %s},\n",
        CATCAS_CORE_A,
        CATCAS_CORE_B,
        CATCAS_CORE_B,
        json_bool(observe_actor_core));
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true},\n");
    fprintf(out, "  \"carrier\": {\"line_sets\": 2, \"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier->line_count, CACHE_LINE_BYTES, carrier->byte_count, initial_digest);
    fprintf(out, "  \"selected_group\": \"primary_nb_coherence\",\n");
    fprintf(out, "  \"group_open_rc\": %d,\n", primary_open_rc);
    fprintf(out, "  \"home_group_open_rc\": %d,\n", home_open_rc);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, primary_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"predeclared_observable\": {\"coordinate_x\": \"cache_block_commands_change_to_dirty / cpu_cycles_not_halted\", \"coordinate_y\": \"probe_responses_dirty / cpu_cycles_not_halted\", \"signed_area\": \"sum((x_i * y_next) - (y_i * x_next)) / 2 over four path steps\"},\n");
    fprintf(out, "  \"paths\": [\n");
    print_path_steps(out, "forward", forward, forward_results, forward_durations, forward_before, forward_after, forward_rc, forward_area, initial_digest, observe_actor_core);
    fprintf(out, ",\n");
    print_path_steps(out, "reverse", reverse, reverse_results, reverse_durations, reverse_before, reverse_after, reverse_rc, reverse_area, initial_digest, observe_actor_core);
    fprintf(out, ",\n");
    print_path_steps(out, "shuffle", shuffle, shuffle_results, shuffle_durations, shuffle_before, shuffle_after, shuffle_rc, shuffle_area, initial_digest, observe_actor_core);
    fprintf(out, ",\n");
    print_path_steps(out, "reverse_shuffle", reverse_shuffle, reverse_shuffle_results, reverse_shuffle_durations, reverse_shuffle_before, reverse_shuffle_after, reverse_shuffle_rc, reverse_shuffle_area, initial_digest, observe_actor_core);
    fprintf(out, ",\n");
    print_path_steps(out, "identity", identity, identity_results, identity_durations, identity_before, identity_after, identity_rc, identity_area, initial_digest, observe_actor_core);
    fprintf(out, "\n  ],\n");
    fprintf(out, "  \"areas_cycles_normalized\": {\"forward\": %.12Le, \"reverse\": %.12Le, \"shuffle\": %.12Le, \"reverse_shuffle\": %.12Le, \"identity\": %.12Le},\n",
        forward_area, reverse_area, shuffle_area, reverse_shuffle_area, identity_area);
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"bytes_unchanged\": %s, \"sign_reversal\": %s, \"controls_small\": %s, \"path_dependence_pilot\": %s}\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(bytes_unchanged),
        json_bool(sign_reversal),
        json_bool(controls_small),
        json_bool(path_pilot));
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"primary_nb_coherence\"}\n",
        path_pilot ? positive_status : negative_status,
        result_path);
    return 0;
}

static int run_path_dependence_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    return run_path_mode(output_root, carrier, initial_digest, false);
}

static int run_path_dual_observe_mode(const char *output_root, struct carrier *carrier, uint64_t initial_digest) {
    return run_path_mode(output_root, carrier, initial_digest, true);
}

int main(int argc, char **argv) {
    const char *output_root = NULL;
    bool coherence_operator_mode = false;
    bool path_dependence_mode = false;
    bool path_dual_observe_mode = false;
    if (argc == 3 && strcmp(argv[1], "--output-root") == 0) {
        output_root = argv[2];
    } else if (argc == 4 && strcmp(argv[1], "--coherence-operators") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        coherence_operator_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--path-dependence") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        path_dependence_mode = true;
        output_root = argv[3];
    } else if (argc == 4 && strcmp(argv[1], "--path-dual-observe") == 0 &&
               strcmp(argv[2], "--output-root") == 0) {
        path_dual_observe_mode = true;
        output_root = argv[3];
    } else if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    } else {
        fprintf(stderr, "usage: %s --output-root <absolute-output-root> | --coherence-operators --output-root <absolute-output-root> | --path-dependence --output-root <absolute-output-root> | --path-dual-observe --output-root <absolute-output-root>\n", argv[0]);
        return 2;
    }
    if (output_root[0] != '/') {
        fprintf(stderr, "output root must be absolute\n");
        return 2;
    }
    if (ensure_dir(output_root) != 0) {
        fprintf(stderr, "cannot create output root: %s\n", strerror(errno));
        return 1;
    }

    struct carrier carrier;
    carrier.line_count = CARRIER_LINES;
    carrier.byte_count = CARRIER_LINES * CACHE_LINE_BYTES;
    if (posix_memalign((void **)&carrier.bytes, CACHE_LINE_BYTES, carrier.byte_count) != 0) {
        fprintf(stderr, "carrier allocation failed\n");
        return 1;
    }
    init_carrier(&carrier);
    uint64_t initial_digest = fnv1a64(carrier.bytes, carrier.byte_count);

    if (coherence_operator_mode) {
        int rc = run_coherence_operator_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (path_dependence_mode) {
        int rc = run_path_dependence_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }
    if (path_dual_observe_mode) {
        int rc = run_path_dual_observe_mode(output_root, &carrier, initial_digest);
        free(carrier.bytes);
        return rc;
    }

    uint64_t support_ids[SUPPORT_EVENTS];
    int support_rc[SUPPORT_EVENTS];
    for (int i = 0; i < SUPPORT_EVENTS; i++) {
        support_ids[i] = 0;
        support_rc[i] = open_single_event(&support_events[i], CATCAS_CORE_A, &support_ids[i]);
    }

    int primary_fds[MAX_GROUP_EVENTS];
    uint64_t primary_ids[MAX_GROUP_EVENTS];
    int primary_open_rc = open_group(primary_group, CATCAS_CORE_A, primary_fds, primary_ids);
    if (primary_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(primary_fds[i]);
    }
    int fallback_fds[MAX_GROUP_EVENTS];
    uint64_t fallback_ids[MAX_GROUP_EVENTS];
    int fallback_open_rc = open_group(fallback_group, CATCAS_CORE_A, fallback_fds, fallback_ids);
    if (fallback_open_rc == 0) {
        for (int i = 0; i < MAX_GROUP_EVENTS; i++) close(fallback_fds[i]);
    }

    const struct event_def *selected_group = NULL;
    const char *selected_group_name = NULL;
    if (primary_open_rc == 0) {
        selected_group = primary_group;
        selected_group_name = "primary_nb_coherence";
    } else if (fallback_open_rc == 0) {
        selected_group = fallback_group;
        selected_group_name = "fallback_core_cache";
    } else {
        selected_group = primary_group;
        selected_group_name = "none_opened";
    }

    const char *windows[] = {
        "idle_pause",
        "core4_read_sweep",
        "core4_write_sweep",
        "cross_core_pingpong_write"
    };
    enum { WINDOW_COUNT = 4 };
    struct group_result results[WINDOW_COUNT];
    uint64_t durations[WINDOW_COUNT];
    uint64_t digest_before[WINDOW_COUNT];
    uint64_t digest_after[WINDOW_COUNT];
    int window_rc[WINDOW_COUNT];
    for (int i = 0; i < WINDOW_COUNT; i++) {
        memset(&results[i], 0, sizeof(results[i]));
        durations[i] = 0;
        digest_before[i] = 0;
        digest_after[i] = 0;
        if (strcmp(selected_group_name, "none_opened") == 0) {
            window_rc[i] = -ENODEV;
        } else {
            window_rc[i] = measure_window(
                selected_group,
                windows[i],
                &carrier,
                &results[i],
                &durations[i],
                &digest_before[i],
                &digest_after[i]
            );
        }
    }

    bool all_unmultiplexed = true;
    bool all_restored = true;
    bool all_windows_ok = true;
    for (int i = 0; i < WINDOW_COUNT; i++) {
        all_windows_ok = all_windows_ok && window_rc[i] == 0;
        all_unmultiplexed = all_unmultiplexed && results[i].unmultiplexed;
        all_restored = all_restored && digest_after[i] == initial_digest;
    }

    uint64_t idle_c2d = event_value_by_name(&results[0], selected_group, "cache_block_commands_change_to_dirty");
    uint64_t read_c2d = event_value_by_name(&results[1], selected_group, "cache_block_commands_change_to_dirty");
    uint64_t ping_c2d = event_value_by_name(&results[3], selected_group, "cache_block_commands_change_to_dirty");
    uint64_t idle_probe = event_value_by_name(&results[0], selected_group, "probe_responses_dirty");
    uint64_t read_probe = event_value_by_name(&results[1], selected_group, "probe_responses_dirty");
    uint64_t ping_probe = event_value_by_name(&results[3], selected_group, "probe_responses_dirty");
    uint64_t c2d_control = idle_c2d > read_c2d ? idle_c2d : read_c2d;
    uint64_t probe_control = idle_probe > read_probe ? idle_probe : read_probe;
    bool c2d_moved = ping_c2d > c2d_control + 32u && ping_c2d > c2d_control * 3u;
    bool probe_moved = ping_probe > probe_control + 32u && ping_probe > probe_control * 3u;
    bool first_light = strcmp(selected_group_name, "primary_nb_coherence") == 0 &&
        all_windows_ok && all_unmultiplexed && all_restored && (c2d_moved || probe_moved);

    char result_path[4096];
    int n = snprintf(result_path, sizeof(result_path), "%s/F10_PMC_FIRST_LIGHT_RESULT.json", output_root);
    if (n <= 0 || (size_t)n >= sizeof(result_path)) {
        free(carrier.bytes);
        return 1;
    }
    FILE *out = fopen(result_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open result: %s\n", strerror(errno));
        free(carrier.bytes);
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"schema_id\": \"%s\",\n", RESULT_SCHEMA);
    fprintf(out, "  \"status\": \"%s\",\n", first_light ? "F10_PMC_FIRST_LIGHT" : "F10_PMC_FIRST_LIGHT_NOT_ESTABLISHED");
    fprintf(out, "  \"claim_ceiling\": \"Family 10h PMU first-light discriminator only; no coherence holonomy, OrbitState coupling, fold-odd recovery, or Small Wall crossing claim\",\n");
    fprintf(out, "  \"cores\": {\"observed_core\": %d, \"partner_core\": %d},\n", CATCAS_CORE_A, CATCAS_CORE_B);
    fprintf(out, "  \"perf_interface\": {\"type\": \"PERF_TYPE_RAW\", \"event_format\": \"config:0-7,32-35\", \"umask_format\": \"config:8-15\", \"exclude_kernel\": true, \"exclude_hv\": true},\n");
    fprintf(out, "  \"source_constraints\": {\"direct_msr_access\": false, \"voltage_access\": false, \"frequency_writes\": 0, \"experiment_owned_memory_only\": true},\n");
    fprintf(out, "  \"carrier\": {\"line_count\": %zu, \"line_bytes\": %d, \"byte_count\": %zu, \"digest_kind\": \"fnv1a64\", \"initial_digest_hex\": \"0x%016" PRIx64 "\"},\n", carrier.line_count, CACHE_LINE_BYTES, carrier.byte_count, initial_digest);
    fprintf(out, "  \"event_support_matrix\": [\n");
    for (int i = 0; i < SUPPORT_EVENTS; i++) {
        fprintf(out, "    {\"event\": ");
        print_event_def(out, &support_events[i]);
        fprintf(out, ", \"open_rc\": %d, \"supported\": %s, \"id\": %" PRIu64 "}%s\n",
            support_rc[i],
            json_bool(support_rc[i] == 0),
            support_ids[i],
            i + 1 == SUPPORT_EVENTS ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"group_support\": {\n");
    fprintf(out, "    \"primary_nb_coherence\": {\"open_rc\": %d, \"supported\": %s, \"events\": ", primary_open_rc, json_bool(primary_open_rc == 0));
    print_group_defs(out, primary_group);
    fprintf(out, "},\n");
    fprintf(out, "    \"fallback_core_cache\": {\"open_rc\": %d, \"supported\": %s, \"events\": ", fallback_open_rc, json_bool(fallback_open_rc == 0));
    print_group_defs(out, fallback_group);
    fprintf(out, "}\n");
    fprintf(out, "  },\n");
    fprintf(out, "  \"selected_group\": \"%s\",\n", selected_group_name);
    fprintf(out, "  \"selected_events\": ");
    print_group_defs(out, selected_group);
    fprintf(out, ",\n");
    fprintf(out, "  \"windows\": [\n");
    for (int i = 0; i < WINDOW_COUNT; i++) {
        fprintf(out, "    {\"name\": \"%s\", \"rc\": %d, \"duration_ns\": %" PRIu64 ", \"carrier_digest_before_hex\": \"0x%016" PRIx64 "\", \"carrier_digest_after_restore_hex\": \"0x%016" PRIx64 "\", \"carrier_restored\": %s, \"group\": ",
            windows[i],
            window_rc[i],
            durations[i],
            digest_before[i],
            digest_after[i],
            json_bool(digest_after[i] == initial_digest));
        print_group_result(out, selected_group, &results[i]);
        fprintf(out, "}%s\n", i + 1 == WINDOW_COUNT ? "" : ",");
    }
    fprintf(out, "  ],\n");
    fprintf(out, "  \"predeclared_observable\": {\"primary\": \"cache_block_commands_change_to_dirty\", \"secondary\": \"probe_responses_dirty\", \"comparison\": \"cross_core_pingpong_write greater than idle_pause and core4_read_sweep controls\"},\n");
    fprintf(out, "  \"acceptance\": {\"all_windows_ok\": %s, \"all_unmultiplexed\": %s, \"carrier_restored\": %s, \"change_to_dirty_moved\": %s, \"probe_dirty_moved\": %s, \"first_light\": %s},\n",
        json_bool(all_windows_ok),
        json_bool(all_unmultiplexed),
        json_bool(all_restored),
        json_bool(c2d_moved),
        json_bool(probe_moved),
        json_bool(first_light));
    fprintf(out, "  \"contrast_counts\": {\"change_to_dirty\": {\"idle\": %" PRIu64 ", \"read_control\": %" PRIu64 ", \"cross_core_transition\": %" PRIu64 "}, \"probe_dirty\": {\"idle\": %" PRIu64 ", \"read_control\": %" PRIu64 ", \"cross_core_transition\": %" PRIu64 "}}\n",
        idle_c2d, read_c2d, ping_c2d, idle_probe, read_probe, ping_probe);
    fprintf(out, "}\n");
    fclose(out);
    printf("{\"status\":\"%s\",\"result_path\":\"%s\",\"selected_group\":\"%s\"}\n",
        first_light ? "F10_PMC_FIRST_LIGHT" : "F10_PMC_FIRST_LIGHT_NOT_ESTABLISHED",
        result_path,
        selected_group_name);
    free(carrier.bytes);
    return 0;
}
