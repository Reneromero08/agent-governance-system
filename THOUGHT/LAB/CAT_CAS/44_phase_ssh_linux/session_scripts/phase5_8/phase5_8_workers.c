#include "phase5_8_workers.h"
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
        /* XOR / rotate stride through memory */
        for (size_t i = 0; i < num_words; i += stride_words) {
            pos = (pos + stride_words) % num_words;
            uint64_t val = buf[pos];
            uint64_t next = lcg_next(&seed);
            val ^= next;
            val = (val << 13) | (val >> 51);  /* rotate left 13 */
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
        /* Deterministic LCG / xor / rotate / multiply loop */
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

/* --- Thermal saturation worker (long-running integer churn) --- */
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
    ws->seed = (uint64_t)(id * 131071 + core * 65521 + mode * 65537);
    atomic_init(&ws->stop, 0);

    if (mode == WORKER_MODE_CACHE_HAMMER || mode == WORKER_MODE_MIXED) {
        ws->buffer = (uint64_t *)aligned_alloc_locked(buffer_size, CACHE_LINE);
        if (!ws->buffer) {
            fprintf(stderr, "Worker %d: failed to allocate buffer\n", id);
            return -1;
        }
    }

    pthread_t thread;
    void *(*routine)(void *) = NULL;

    switch (mode) {
    case WORKER_MODE_CACHE_HAMMER:
        routine = cache_hammer_worker;
        break;
    case WORKER_MODE_INTEGER_CHURN:
        routine = integer_churn_worker;
        break;
    case WORKER_MODE_THERMAL:
        routine = thermal_worker;
        break;
    case WORKER_MODE_MIXED:
        /* Mixed: half cache hammer, half integer churn.
           For even id -> cache hammer, odd id -> integer churn. */
        routine = (id % 2 == 0) ? cache_hammer_worker : integer_churn_worker;
        break;
    default:
        return -1;
    }

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    /* Validate core ID: must be < CPU_SETSIZE and reasonable for the machine.
       Invalid cores (e.g., 8,10,12 on 6-core CPU) cause affinity failures. */
    if (core < 0 || core >= CPU_SETSIZE || core >= 64) {
        fprintf(stderr, "Worker %d: invalid core %d (max %d)\n", id, core, CPU_SETSIZE - 1);
        pthread_attr_destroy(&attr);
        if (ws->buffer) { munlock(ws->buffer, ws->buffer_size); free(ws->buffer); ws->buffer = NULL; }
        return -1;
    }
    CPU_SET(core, &cpuset);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);

    if (pthread_create(&thread, &attr, routine, ws) != 0) {
        perror("pthread_create worker");
        pthread_attr_destroy(&attr);
        if (ws->buffer) { munlock(ws->buffer, ws->buffer_size); free(ws->buffer); ws->buffer = NULL; }
        return -1;
    }

    pthread_attr_destroy(&attr);
    pthread_detach(thread);
    return 0;
}

void worker_stop_all(worker_state_t *workers, int count) {
    for (int i = 0; i < count; i++) {
        atomic_store(&workers[i].stop, 1);
    }
}

void worker_join_all(worker_state_t *workers, int count) {
    /* Workers are detached; poll stop flag acknowledgment via barrier */
    int all_stopped = 0;
    for (int retry = 0; retry < 1000 && !all_stopped; retry++) {
        usleep(1000);  /* 1ms */
        all_stopped = 1;
        for (int i = 0; i < count; i++) {
            if (atomic_load(&workers[i].stop) == 0) {
                all_stopped = 0;
                break;
            }
        }
    }

    /* Free worker buffers */
    for (int i = 0; i < count; i++) {
        if (workers[i].buffer) {
            munlock(workers[i].buffer, workers[i].buffer_size);
            free(workers[i].buffer);
            workers[i].buffer = NULL;
        }
    }
}
