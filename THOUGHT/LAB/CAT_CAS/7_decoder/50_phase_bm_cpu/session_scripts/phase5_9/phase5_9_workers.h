#ifndef PHASE5_9_WORKERS_H
#define PHASE5_9_WORKERS_H

#include "phase5_9_common.h"
#include <pthread.h>

/* Start a worker thread pinned to a given core.
   Workers are created JOINABLE (not detached).
   Returns 0 on success, -1 on error. */
int worker_start(worker_state_t *ws, int id, int core, worker_mode_t mode,
                 size_t buffer_size, int stride);

/* Signal all workers to stop via atomic stop flag */
void worker_stop_all(worker_state_t *workers, int count);

/* Join all workers with timeout. Only frees buffers after successful join.
   Returns number of failed joins. Never frees buffer of a live thread. */
int worker_join_all(worker_state_t *workers, int count);

/* Cache-line-aligned buffer size for cache hammering.
   Phenom II L3 = 6MB shared. 20MB reliably spills L3.
   Fallback sizes if mlock limit is hit: 8MB, 4MB. */
#define WORKER_BUFFER_DEFAULT (20 * 1024 * 1024)
#define WORKER_BUFFER_FALLBACK_8 (8 * 1024 * 1024)
#define WORKER_BUFFER_FALLBACK_4 (4 * 1024 * 1024)

/* Default stride: 64 bytes (cache line) */
#define WORKER_STRIDE_DEFAULT 64

#endif /* PHASE5_9_WORKERS_H */
