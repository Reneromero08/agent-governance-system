#ifndef PHASE5_8_WORKERS_H
#define PHASE5_8_WORKERS_H

#include "phase5_8_common.h"

/* Start a worker thread pinned to a given core.
   Returns 0 on success, -1 on error. */
int worker_start(worker_state_t *ws, int id, int core, worker_mode_t mode,
                 size_t buffer_size, int stride);

/* Signal all workers to stop */
void worker_stop_all(worker_state_t *workers, int count);

/* Wait for all workers to finish */
void worker_join_all(worker_state_t *workers, int count);

/* Get cache-line-aligned buffer size suitable for cache hammering.
   For Phenom II L3 = 6MB shared, use at least 20MB to reliably spill L3. */
#define WORKER_BUFFER_DEFAULT (20 * 1024 * 1024)

/* Default stride: 64 bytes (cache line) */
#define WORKER_STRIDE_DEFAULT 64

#endif /* PHASE5_8_WORKERS_H */
