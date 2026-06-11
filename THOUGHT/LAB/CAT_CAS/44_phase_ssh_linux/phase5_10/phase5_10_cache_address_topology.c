/*
 * EXP44 Phase 5.10D cache/address topology prep probe.
 *
 * User-space only. This probes whether fixed address-set families produce a
 * reproducible timing carrier beyond compute-only and randomized-layout nulls.
 *
 * Build:
 *   gcc -O2 -pthread -Wall -Wextra phase5_10_cache_address_topology.c -o cache_addr_topology
 *
 * Example:
 *   ./cache_addr_topology --reps 12 --iters 2000 --buf-mb 64 --output phase5_10d_cache_address_topology.csv
 */

#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef enum {
    AGGR_NONE = 0,
    AGGR_COMPUTE = 1,
    AGGR_SAME = 2,
    AGGR_DIFF = 3,
    AGGR_RANDOM = 4
} aggr_mode_t;

typedef struct {
    uint8_t *buf;
    size_t buf_size;
    size_t *idx;
    size_t idx_count;
    volatile int stop;
    aggr_mode_t mode;
    int family;
    int victim_family;
    int aggr_core;
    int pin;
    volatile uint64_t sink;
} aggr_ctx_t;

typedef struct {
    int reps;
    int iters;
    int buf_mb;
    int victim_core;
    int aggr_core;
    int pin;
    const char *output;
} opts_t;

static inline uint64_t rdtscp64(void) {
    uint32_t lo, hi, aux;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) ::);
    return ((uint64_t)hi << 32) | lo;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s [--reps N] [--iters N] [--buf-mb N] [--victim-core N]\n"
            "          [--aggr-core N] [--no-affinity] [--output FILE]\n",
            argv0);
}

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || *s == '\0' || !end || *end != '\0') return -1;
    if (v < 0 || v > 1000000000L) return -1;
    *out = (int)v;
    return 0;
}

static int parse_opts(int argc, char **argv, opts_t *o) {
    o->reps = 12;
    o->iters = 2000;
    o->buf_mb = 64;
    o->victim_core = 2;
    o->aggr_core = 3;
    o->pin = 1;
    o->output = "phase5_10d_cache_address_topology.csv";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &o->reps) != 0) return -1;
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &o->iters) != 0) return -1;
        } else if (strcmp(argv[i], "--buf-mb") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &o->buf_mb) != 0) return -1;
        } else if (strcmp(argv[i], "--victim-core") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &o->victim_core) != 0) return -1;
        } else if (strcmp(argv[i], "--aggr-core") == 0 && i + 1 < argc) {
            if (parse_int(argv[++i], &o->aggr_core) != 0) return -1;
        } else if (strcmp(argv[i], "--no-affinity") == 0) {
            o->pin = 0;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            o->output = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            return -1;
        }
    }
    if (o->reps < 2 || o->iters < 10 || o->buf_mb < 4) return -1;
    return 0;
}

static int pin_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return sched_setaffinity(0, sizeof(set), &set);
}

static uint64_t lcg_next(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL) + 1442695040888963407ULL;
    return *s;
}

static const char *mode_name(aggr_mode_t m) {
    switch (m) {
        case AGGR_NONE: return "none";
        case AGGR_COMPUTE: return "compute";
        case AGGR_SAME: return "same_address";
        case AGGR_DIFF: return "different_address";
        case AGGR_RANDOM: return "random_address";
        default: return "unknown";
    }
}

static size_t family_offset(int family) {
    static const size_t offsets[] = {0, 64, 128, 256, 512, 1024};
    return offsets[family % 6];
}

static size_t make_indices(size_t *idx, size_t n, size_t buf_size, int family, int randomize) {
    const size_t stride = 4096;
    const size_t base = family_offset(family);
    size_t count = 0;
    uint64_t rng = 0x5eedeadULL + (uint64_t)family * 0x10001ULL;
    for (size_t off = base; off + 64 < buf_size && count < n; off += stride) {
        idx[count++] = off;
    }
    if (randomize) {
        for (size_t i = count; i > 1; i--) {
            size_t j = (size_t)(lcg_next(&rng) % i);
            size_t tmp = idx[i - 1];
            idx[i - 1] = idx[j];
            idx[j] = tmp;
        }
    }
    return count;
}

static uint64_t measure_family(uint8_t *buf, size_t *idx, size_t idx_count, int iters, volatile uint64_t *sink) {
    uint64_t sum = 0;
    uint64_t t0 = rdtscp64();
    for (int r = 0; r < iters; r++) {
        for (size_t i = 0; i < idx_count; i += 8) {
            sum += buf[idx[i]];
            buf[idx[i]] = (uint8_t)(buf[idx[i]] + 1u);
        }
    }
    uint64_t t1 = rdtscp64();
    *sink += sum;
    return t1 - t0;
}

static void *aggressor_main(void *arg) {
    aggr_ctx_t *c = (aggr_ctx_t *)arg;
    if (c->pin) (void)pin_core(c->aggr_core);
    size_t local_idx_cap = c->idx_count;
    size_t *local_idx = malloc(local_idx_cap * sizeof(size_t));
    if (!local_idx) return NULL;
    int fam = c->family;
    if (c->mode == AGGR_SAME) fam = c->victim_family;
    if (c->mode == AGGR_DIFF) fam = (c->victim_family + 3) % 6;
    size_t count = make_indices(local_idx, local_idx_cap, c->buf_size, fam, c->mode == AGGR_RANDOM);
    uint64_t x = 0x123456789abcdefULL;
    while (!c->stop) {
        if (c->mode == AGGR_COMPUTE) {
            for (int i = 0; i < 200000; i++) {
                x ^= x << 7;
                x ^= x >> 9;
                x *= 0x9e3779b97f4a7c15ULL;
            }
        } else if (c->mode != AGGR_NONE) {
            for (size_t i = 0; i < count; i += 4) {
                c->buf[local_idx[i]] = (uint8_t)(c->buf[local_idx[i]] + 3u);
                x += c->buf[local_idx[i]];
            }
        }
    }
    c->sink += x;
    free(local_idx);
    return NULL;
}

static void sleep_us(int usec) {
    struct timespec ts;
    ts.tv_sec = usec / 1000000;
    ts.tv_nsec = (long)(usec % 1000000) * 1000L;
    nanosleep(&ts, NULL);
}

int main(int argc, char **argv) {
    opts_t o;
    if (parse_opts(argc, argv, &o) != 0) {
        usage(argv[0]);
        return 2;
    }

    const size_t buf_size = (size_t)o.buf_mb * 1024u * 1024u;
    uint8_t *buf = NULL;
    if (posix_memalign((void **)&buf, 4096, buf_size) != 0 || !buf) {
        fprintf(stderr, "allocation failed: %s\n", strerror(errno));
        return 3;
    }
    for (size_t i = 0; i < buf_size; i++) buf[i] = (uint8_t)(i * 131u + 17u);

    size_t idx_cap = buf_size / 4096 + 8;
    size_t *idx = calloc(idx_cap, sizeof(size_t));
    if (!idx) {
        free(buf);
        return 3;
    }

    FILE *out = fopen(o.output, "w");
    if (!out) {
        fprintf(stderr, "cannot open output %s: %s\n", o.output, strerror(errno));
        free(idx);
        free(buf);
        return 4;
    }
    fprintf(out, "run_id,family,aggressor,rep,iters,idx_count,total_cycles,cycles_per_touch,victim_core,aggr_core,pin_affinity,buf_mb\n");
    fflush(out);

    if (o.pin) (void)pin_core(o.victim_core);

    aggr_mode_t modes[] = {AGGR_NONE, AGGR_COMPUTE, AGGR_SAME, AGGR_DIFF, AGGR_RANDOM};
    const int nmodes = (int)(sizeof(modes) / sizeof(modes[0]));
    volatile uint64_t sink = 0;
    int run_id = 0;

    for (int rep = 0; rep < o.reps; rep++) {
        for (int fam = 0; fam < 6; fam++) {
            size_t idx_count = make_indices(idx, idx_cap, buf_size, fam, 0);
            for (int mi = 0; mi < nmodes; mi++) {
                aggr_mode_t mode = modes[(mi + rep + fam) % nmodes];
                pthread_t th;
                aggr_ctx_t ctx;
                memset(&ctx, 0, sizeof(ctx));
                ctx.buf = buf;
                ctx.buf_size = buf_size;
                ctx.idx = idx;
                ctx.idx_count = idx_cap;
                ctx.mode = mode;
                ctx.family = fam;
                ctx.victim_family = fam;
                ctx.aggr_core = o.aggr_core;
                ctx.pin = o.pin;

                if (mode != AGGR_NONE) {
                    if (pthread_create(&th, NULL, aggressor_main, &ctx) != 0) {
                        fprintf(stderr, "pthread_create failed\n");
                        fclose(out);
                        free(idx);
                        free(buf);
                        return 5;
                    }
                    sleep_us(20000);
                }

                uint64_t cycles = measure_family(buf, idx, idx_count, o.iters, &sink);

                if (mode != AGGR_NONE) {
                    ctx.stop = 1;
                    pthread_join(th, NULL);
                    sink += ctx.sink;
                }

                double touches = (double)o.iters * ((double)(idx_count + 7) / 8.0);
                double cpt = touches > 0.0 ? (double)cycles / touches : 0.0;
                fprintf(out, "%d,%d,%s,%d,%d,%zu,%" PRIu64 ",%.6f,%d,%d,%d,%d\n",
                        run_id++, fam, mode_name(mode), rep, o.iters, idx_count,
                        cycles, cpt, o.victim_core, o.aggr_core, o.pin, o.buf_mb);
                fflush(out);
            }
        }
    }

    fprintf(out, "# SINK,%" PRIu64 "\n", (uint64_t)sink);
    fclose(out);
    free(idx);
    free(buf);
    return 0;
}
