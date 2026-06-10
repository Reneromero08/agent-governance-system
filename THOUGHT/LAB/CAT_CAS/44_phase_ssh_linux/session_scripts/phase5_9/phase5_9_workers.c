#include "phase5_9_workers.h"
#include <pthread.h>

/* --- Deterministic LCG --- */
static inline uint64_t lcg_next(uint64_t *seed) {
    *seed = (*seed * 6364136223846793005ULL + 1442695040888963407ULL);
    return *seed;
}

/* --- Cache hammer worker --- */
static void *cache_hammer_worker(void *arg) {
    worker_state_t *ws = (worker_state_t *)arg;
    volatile uint64_t *buf = (volatile uint64_t *)ws->buffer;
    size_t num_words = ws->buffer_size / sizeof(uint64_t);
    size_t stride_words = ws->stride / sizeof(uint64_t);
    if (stride_words < 1) stride_words = 1;

    uint64_t seed = ws->seed;
    size_t pos = 0;

    while (!atomic_load(&ws->stop)) {
        for (size_t i = 0; i < num_words; i += stride_words) {
            pos = (pos + stride_words) % num_words;
            uint64_t val = buf[pos];
            uint64_t next = lcg_next(&seed);
            val ^= next;
            val = (val << 13) | (val >> 51);
            buf[pos] = val;
        }
    }
    return NULL;
}

/* --- Integer churn worker --- */
static void *integer_churn_worker(void *arg) {
    worker_state_t *ws = (worker_state_t *)arg;
    uint64_t seed = ws->seed;
    uint64_t a = lcg_next(&seed);
    uint64_t b = lcg_next(&seed);
    uint64_t c = lcg_next(&seed);
    uint64_t d = lcg_next(&seed);

    while (!atomic_load(&ws->stop)) {
        a ^= lcg_next(&seed);
        b = (b << 17) | (b >> 47);
        c *= 0x9E3779B97F4A7C15ULL;
        d ^= (a + b + c);
        a = (a << 7) | (a >> 57);
        b ^= d;
        c += a;
        d = (d << 23) | (d >> 41);
    }
    return NULL;
}

/* --- Thermal saturation worker --- */
static void *thermal_worker(void *arg) {
    return integer_churn_worker(arg);
}

int worker_start(worker_state_t *ws, int id, int core, worker_mode_t mode,
                 size_t buffer_size, int stride) {
    memset(ws, 0, sizeof(*ws));
    ws->worker_id = id;
    ws->core_id = core;
    ws->mode = mode;
    ws->buffer_size = buffer_size;
    ws->stride = stride;
    ws->start_ok = 0;
    ws->join_ok = 0;
    ws->mlock_used = 0;
    ws->seed = (uint64_t)(id * 131071 + core * 65521 + mode * 65537);
    atomic_init(&ws->stop, 0);

    /* Allocate worker buffer with fallback for cache/mixed modes.
       Use aligned_alloc_touched (NO mlock) to avoid mlock limit. */
    if (mode == WORKER_MODE_CACHE_HAMMER || mode == WORKER_MODE_MIXED) {
        size_t try_sizes[] = {
            WORKER_BUFFER_DEFAULT,
            WORKER_BUFFER_FALLBACK_8,
            WORKER_BUFFER_FALLBACK_4
        };
        int n_tries = sizeof(try_sizes) / sizeof(try_sizes[0]);
        for (int t = 0; t < n_tries; t++) {
            ws->buffer = (uint64_t *)aligned_alloc_touched(try_sizes[t], CACHE_LINE);
            if (ws->buffer) {
                ws->buffer_size = try_sizes[t];
                ws->buffer_mb_actual = try_sizes[t] / (1024 * 1024);
                fprintf(stderr, "Worker %d: buffer allocated at %zu MB\n",
                        id, ws->buffer_mb_actual);
                break;
            }
            fprintf(stderr, "Worker %d: buffer alloc failed at %zu MB, retrying...\n",
                    id, try_sizes[t] / (1024 * 1024));
        }
        if (!ws->buffer) {
            fprintf(stderr, "Worker %d: all buffer alloc attempts failed\n", id);
            return -1;
        }
    }

    /* Validate core ID */
    if (core < 0 || core >= CPU_SETSIZE || core >= 64) {
        fprintf(stderr, "Worker %d: invalid core %d (max %d)\n", id, core, CPU_SETSIZE - 1);
        if (ws->buffer) { free(ws->buffer); ws->buffer = NULL; }
        return -1;
    }

    /* Select routine */
    void *(*routine)(void *) = NULL;
    switch (mode) {
    case WORKER_MODE_CACHE_HAMMER:  routine = cache_hammer_worker; break;
    case WORKER_MODE_INTEGER_CHURN: routine = integer_churn_worker; break;
    case WORKER_MODE_THERMAL:       routine = thermal_worker; break;
    case WORKER_MODE_MIXED:
        routine = (id % 2 == 0) ? cache_hammer_worker : integer_churn_worker;
        break;
    default:
        if (ws->buffer) { free(ws->buffer); ws->buffer = NULL; }
        return -1;
    }

    /* Create JOINABLE thread (not detached — joined in worker_join_all) */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);

    if (pthread_create(&ws->thread, &attr, routine, ws) != 0) {
        perror("pthread_create worker");
        pthread_attr_destroy(&attr);
        if (ws->buffer) { free(ws->buffer); ws->buffer = NULL; }
        return -1;
    }

    pthread_attr_destroy(&attr);
    ws->start_ok = 1;
    return 0;
}

void worker_stop_all(worker_state_t *workers, int count) {
    for (int i = 0; i < count; i++) {
        if (workers[i].start_ok) {
            atomic_store(&workers[i].stop, 1);
        }
    }
}

int worker_join_all(worker_state_t *workers, int count) {
    int failed_joins = 0;
    /* Join each started worker with timeout.
       Buffer is freed ONLY after successful join.
       Failed/timeout joins: buffer retained for safety (no use-after-free). */
    for (int i = 0; i < count; i++) {
        if (workers[i].start_ok) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += 2;

            void *retval = NULL;
            int rc = pthread_timedjoin_np(workers[i].thread, &retval, &ts);
            if (rc == 0) {
                workers[i].join_ok = 1;
                /* Only free buffer after confirmed join */
                if (workers[i].buffer) {
                    free(workers[i].buffer);
                    workers[i].buffer = NULL;
                }
            } else {
                workers[i].join_ok = 0;
                failed_joins++;
                fprintf(stderr, "Worker %d: join failed (rc=%d, %s); buffer retained for safety\n",
                        i, rc, rc == ETIMEDOUT ? "timeout" : "error");
                /* Do NOT free buffer — thread may still be using it */
            }
        }
    }
    return failed_joins;
}
