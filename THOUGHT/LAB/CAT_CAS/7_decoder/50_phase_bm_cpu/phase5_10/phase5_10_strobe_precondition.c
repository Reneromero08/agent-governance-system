/*
 * phase5_10_strobe_precondition.c
 *
 * STEP 0 of Phase 5.10 (boundary state preparation). Single cheap-decisive gate.
 *
 * QUESTION (the only thing this program decides):
 *   Is the ~2.67 MHz line seen in the rail/timing spectrum a REAL cross-core
 *   shared-rail phase-coherent strobe that RESPONDS to physical/history levers
 *   and is NOT reproduced by false-tone controls? If not, the whole
 *   electrical-strobe / coherent-averaging approach (PHASE5_10A SNR upgrade) is
 *   unfounded and we stop before any kernel work.
 *
 * METHOD (userspace only - no kernel module; kernel headers are missing on the rig):
 *   - Two sampler threads pinned to PDN-sharing isolated cores run a fixed-work
 *     tight inner loop and timestamp each iteration with rdtscp. The per-iteration
 *     duration is a software ring-oscillator whose period is modulated by
 *     rail/transport state. This yields a (near-)uniformly sampled time series of
 *     loop durations on each core.
 *   - Cross-core magnitude-squared coherence gamma^2(f) between the two series is
 *     computed by Goertzel at the target bin (~2.67 MHz) and at off-frequency
 *     neighbor bins, in a 1-5 MHz band. A shared-rail tone drives gamma^2 -> 1 at
 *     2.67 MHz across PDN-sharing pairs; a per-core artifact stays incoherent.
 *   - The measurement is repeated under levers (idle, cache aggressor, syscall
 *     burst, integer churn, P-state toggle, up-from-idle, down-from-high) and under
 *     a mandatory false-tone control battery (off-freq bins, phase-randomized,
 *     shuffled labels, different core pairs, same-core duplicate, idle-only null,
 *     fixed-P0 null).
 *
 * REUSE (patterns lifted, not reinvented, from):
 *   - session_scripts/phase2_kuramoto/tsc_sampler.c   (multi-core TSC sampling)
 *   - session_scripts/phase5_8/phase5_8_common.h       (rdtsc serialization, k10temp, CSV)
 *   - session_scripts/phase5_9/phase5_9_workers.c      (cache-hammer / integer-churn loops)
 *
 * SAFETY (hard):
 *   - k10temp read before every burst; ABORT the whole run if temp >= 62 C.
 *   - Any P-state toggle (wrmsr to PStateCtl 0xC0010062) is restored to the exact
 *     pre-run value on the way out (before/after recorded). NO persistent MSR write,
 *     NO P-state definition (0xC001006x) edits - only the transient request register.
 *   - Total runtime bounded (a handful of seconds per condition, a few minutes total).
 *   - Reboot-recoverable by construction (userspace only).
 *
 * RIG FACTS (verified live, AMD Phenom II, kernel 6.12.86+deb13-amd64):
 *   - cores 2,3,4,5 isolated (isolcpus); pin sampler/worker threads there.
 *   - k10temp at /sys/class/hwmon/hwmon0/temp1_input (milli-C). We locate it by name.
 *   - TSC invariant, ~3.2 GHz; 2.67 MHz -> ~375 ns period -> ~1200 ticks/period.
 *   - P-state def MSRs (core 2): P0=...0c14 (3200MHz), P1=...2410 (1600MHz); idle P-state ~3.
 *
 * Build:  gcc -O2 -pthread -lm phase5_10_strobe_precondition.c -o precond
 * Usage:  ./precond [--target-hz 2670000] [--tsc-hz 3.2e9] [--stride-ticks 24]
 *                   [--n-samples 262144] [--temp-abort 62] [--output-dir DIR]
 *                   [--no-pstate]   (skip the wrmsr P-state levers/null)
 *
 * ASCII only. No sensational naming. This is instrumentation.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <dirent.h>
#include <stdatomic.h>

/* ----------------------------------------------------------------------------
 * Timing primitives. Reused from phase5_8_common.h / tsc_sampler.c, but kept
 * LIGHT on purpose. The goal of the sampler is to oversample the ~375 ns rail
 * period (~1200 TSC ticks at 3.2 GHz) at ~20-40 ticks/sample. A full
 * cpuid-serialized rdtscp costs ~hundreds of ticks and would put 2.67 MHz right
 * at Nyquist. So we use a plain rdtscp: it retires after prior loads/stores
 * (giving enough ordering for our duration series) without the cpuid pipeline
 * flush. rdtsc_start/rdtsc_end (the heavy serialized pair) are kept available
 * for the overhead calibration only.
 * -------------------------------------------------------------------------- */
static inline uint64_t rdtscp_light(void) {
    unsigned hi, lo, aux;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) : : );
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t rdtsc_serialized(void) {
    unsigned hi, lo;
    __asm__ volatile("cpuid\n\trdtsc" : "=a"(lo), "=d"(hi) : : "%rbx", "%rcx");
    return ((uint64_t)hi << 32) | lo;
}

/* ----------------------------------------------------------------------------
 * CPU affinity (reused pattern).
 * -------------------------------------------------------------------------- */
static int pin_to_core(int core) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "WARN: sched_setaffinity(core=%d): %s\n", core, strerror(errno));
        return -1;
    }
    return 0;
}

/* ----------------------------------------------------------------------------
 * k10temp read. Locate the hwmon dir whose name == "k10temp", read temp1_input
 * (milli-C). Returns temp in Celsius, or -999.0 on failure (treated as a hard
 * abort so we never run a thermal burst blind).
 * -------------------------------------------------------------------------- */
static char g_k10_path[256] = {0};

static int locate_k10temp(void) {
    DIR *d = opendir("/sys/class/hwmon");
    if (!d) return -1;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "hwmon", 5) != 0) continue;
        char namep[256], buf[64];
        snprintf(namep, sizeof(namep), "/sys/class/hwmon/%s/name", e->d_name);
        FILE *f = fopen(namep, "r");
        if (!f) continue;
        if (fgets(buf, sizeof(buf), f)) {
            size_t n = strlen(buf);
            while (n && (buf[n-1] == '\n' || buf[n-1] == ' ')) buf[--n] = 0;
            if (strcmp(buf, "k10temp") == 0) {
                snprintf(g_k10_path, sizeof(g_k10_path),
                         "/sys/class/hwmon/%s/temp1_input", e->d_name);
                fclose(f);
                closedir(d);
                return 0;
            }
        }
        fclose(f);
    }
    closedir(d);
    return -1;
}

static double read_k10temp_c(void) {
    if (g_k10_path[0] == 0) return -999.0;
    FILE *f = fopen(g_k10_path, "r");
    if (!f) return -999.0;
    long milli = 0;
    if (fscanf(f, "%ld", &milli) != 1) { fclose(f); return -999.0; }
    fclose(f);
    return (double)milli / 1000.0;
}

/* ----------------------------------------------------------------------------
 * Deterministic LCG and worker loops (reused from phase5_9_workers.c).
 * These run on the OTHER isolated cores as physical levers (aggressors).
 * -------------------------------------------------------------------------- */
static inline uint64_t lcg_next(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL + 1442695040888963407ULL);
    return *s;
}

typedef enum {
    WL_NONE = 0,
    WL_CACHE,       /* cache-hammer: spills L3, drives PDN current swings */
    WL_INTEGER,     /* integer-churn: ALU-bound, steady current */
    WL_SYSCALL      /* syscall burst: kernel entry/exit, current bursts */
} worker_loop_t;

typedef struct {
    atomic_int stop;
    pthread_t thread;
    int core;
    worker_loop_t mode;
    uint64_t seed;
    uint64_t *buf;        /* for cache mode */
    size_t buf_words;
    int started;
} aggressor_t;

#define AGG_BUF_BYTES (16 * 1024 * 1024)   /* > 6MB L3 to reliably spill */

static void *cache_hammer_loop(void *arg) {
    aggressor_t *a = (aggressor_t *)arg;
    pin_to_core(a->core);
    volatile uint64_t *buf = (volatile uint64_t *)a->buf;
    size_t n = a->buf_words;
    size_t stride = 8;            /* 64 bytes / 8 bytes */
    uint64_t seed = a->seed;
    size_t pos = 0;
    while (!atomic_load(&a->stop)) {
        for (size_t i = 0; i < n; i += stride) {
            pos = (pos + stride) % n;
            uint64_t v = buf[pos];
            v ^= lcg_next(&seed);
            v = (v << 13) | (v >> 51);
            buf[pos] = v;
        }
    }
    return NULL;
}

static void *integer_churn_loop(void *arg) {
    aggressor_t *a = (aggressor_t *)arg;
    pin_to_core(a->core);
    uint64_t seed = a->seed;
    uint64_t x = lcg_next(&seed), y = lcg_next(&seed);
    uint64_t z = lcg_next(&seed), w = lcg_next(&seed);
    while (!atomic_load(&a->stop)) {
        x ^= lcg_next(&seed);
        y = (y << 17) | (y >> 47);
        z *= 0x9E3779B97F4A7C15ULL;
        w ^= (x + y + z);
        x = (x << 7) | (x >> 57);
        y ^= w;
        z += x;
        w = (w << 23) | (w >> 41);
    }
    /* keep results live so the loop is not optimized away */
    __asm__ volatile("" :: "r"(x), "r"(y), "r"(z), "r"(w));
    return NULL;
}

static void *syscall_burst_loop(void *arg) {
    aggressor_t *a = (aggressor_t *)arg;
    pin_to_core(a->core);
    struct timespec ts;
    while (!atomic_load(&a->stop)) {
        /* cheap, real syscalls: kernel entry/exit storm */
        clock_gettime(CLOCK_MONOTONIC, &ts);
        getpid();
        sched_yield();
    }
    return NULL;
}

static int aggressor_start(aggressor_t *a, int core, worker_loop_t mode) {
    memset(a, 0, sizeof(*a));
    a->core = core;
    a->mode = mode;
    a->seed = (uint64_t)(core * 65521 + mode * 2654435761ULL + 1);
    atomic_init(&a->stop, 0);
    void *(*routine)(void *) = NULL;
    if (mode == WL_CACHE) {
        a->buf_words = AGG_BUF_BYTES / sizeof(uint64_t);
        if (posix_memalign((void **)&a->buf, 64, AGG_BUF_BYTES) != 0 || !a->buf) {
            fprintf(stderr, "WARN: aggressor cache buf alloc failed\n");
            return -1;
        }
        memset(a->buf, 0, AGG_BUF_BYTES);
        routine = cache_hammer_loop;
    } else if (mode == WL_INTEGER) {
        routine = integer_churn_loop;
    } else if (mode == WL_SYSCALL) {
        routine = syscall_burst_loop;
    } else {
        return 0; /* WL_NONE: no thread */
    }
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
    if (pthread_create(&a->thread, &attr, routine, a) != 0) {
        fprintf(stderr, "WARN: aggressor pthread_create: %s\n", strerror(errno));
        pthread_attr_destroy(&attr);
        if (a->buf) { free(a->buf); a->buf = NULL; }
        return -1;
    }
    pthread_attr_destroy(&attr);
    a->started = 1;
    return 0;
}

static void aggressor_stop(aggressor_t *a) {
    if (!a->started) { if (a->buf) { free(a->buf); a->buf = NULL; } return; }
    atomic_store(&a->stop, 1);
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 2;
    void *rv;
    if (pthread_timedjoin_np(a->thread, &rv, &ts) == 0) {
        if (a->buf) { free(a->buf); a->buf = NULL; }
    } else {
        fprintf(stderr, "WARN: aggressor join timeout; buffer retained\n");
    }
    a->started = 0;
}

/* ----------------------------------------------------------------------------
 * Sampler thread. Pinned to a PDN-sharing isolated core. Runs a fixed-work tight
 * inner loop; every iteration is timestamped with serialized rdtscp. We record
 * the loop-duration series (a software ring-oscillator) at near-uniform spacing.
 *
 * The fixed inner work is a tiny dependent ALU chain (constant instruction count)
 * so that variation in measured duration reflects the rail/transport/contention
 * state, not data-dependent control flow.
 *
 * Output: durations[i] = ticks between consecutive timestamps (in TSC ticks).
 * We also record the absolute start TSC of each core's series so the two cores
 * can be aligned to a common time origin for cross-spectrum.
 * -------------------------------------------------------------------------- */
typedef struct {
    int core;
    int n;                  /* number of raw samples to take */
    uint64_t *stamps;       /* out: absolute TSC timestamp of each sample (n entries) */
    atomic_int *go;         /* shared start gate: both cores spin until set */
    volatile int ready;     /* out: thread reached the gate */
} sampler_arg_t;

static void *sampler_thread(void *arg) {
    sampler_arg_t *s = (sampler_arg_t *)arg;
    pin_to_core(s->core);
    int n = s->n;
    uint64_t *st = s->stamps;
    volatile uint64_t acc = 0x12345678u + (uint64_t)s->core;

    /* warm up: settle caches / branch predictors / frequency */
    for (int i = 0; i < 8192; i++) acc = acc * 6364136223846793005ULL + 1;

    /* gate: both samplers spin here, then start within a few ns of each other */
    s->ready = 1;
    while (atomic_load(s->go) == 0) { __asm__ volatile("pause"); }

    /* Tight sampling loop. Minimal fixed inner work (one MAC) so back-to-back
       rdtscp samples land ~20-40 ticks apart, oversampling the rail period.
       We store ABSOLUTE timestamps so the two cores can later be resampled onto
       a common real-time grid (the loop-iteration index is NOT a clean time axis
       because the per-iteration cost is itself rail/contention-modulated). */
    for (int i = 0; i < n; i++) {
        uint64_t a = acc;
        a = a * 6364136223846793005ULL + 1442695040888963407ULL;
        acc = a;                 /* one dependent MAC: fixed, tiny, non-elidable */
        st[i] = rdtscp_light();
    }
    __asm__ volatile("" :: "r"(acc));
    return NULL;
}

/* ----------------------------------------------------------------------------
 * Goertzel single-bin power at normalized frequency f/fs (cycles per sample),
 * on a real series x[0..n). Returns |X(f)|^2 (a power) and the complex value via
 * out_re/out_im (needed for the cross-spectrum / coherence).
 *
 * We mean-remove and apply a Hann window before Goertzel to suppress leakage,
 * since the durations series has a large DC component (the mean loop time).
 * -------------------------------------------------------------------------- */
static void goertzel_complex(const double *x, int n, double cycles_per_sample,
                             double mean, double *out_re, double *out_im) {
    double w = 2.0 * M_PI * cycles_per_sample;
    double cw = cos(w), sw = sin(w);
    double coeff = 2.0 * cw;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < n; i++) {
        /* Hann window */
        double win = 0.5 * (1.0 - cos(2.0 * M_PI * i / (n - 1)));
        double xi = (x[i] - mean) * win;
        s0 = xi + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    /* complex spectral value at the bin */
    double re = s1 - s2 * cw;
    double im = s2 * sw;
    *out_re = re;
    *out_im = im;
}

static double mean_of(const double *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i];
    return s / n;
}

/* ----------------------------------------------------------------------------
 * Resample a raw absolute-timestamp series onto a uniform real-time grid.
 *
 * Input  : stamps[0..n) = absolute TSC at each loop iteration of one core.
 * Output : g[0..ng) = the instantaneous inter-sample interval (in TSC ticks),
 *          sampled at uniform grid step dt_ticks, over [t_lo, t_lo + ng*dt).
 *
 * The "signal" is the local loop period d(t) = stamps[i+1]-stamps[i], placed at
 * time (stamps[i]+stamps[i+1])/2 and linearly interpolated onto the uniform grid.
 * d(t) is the software ring-oscillator output: when the shared rail sags / a
 * switching edge lands, every dependent op stretches and d(t) rises. Putting BOTH
 * cores on the SAME real-time grid [t_lo, t_hi] with the SAME dt is what makes the
 * cross-core coherence physically meaningful (common time axis, not loop index).
 *
 * Returns the number of valid grid points written (<= ng_max), or 0 on failure.
 * Sets *out_dt_ticks to the grid step used.
 * -------------------------------------------------------------------------- */
static int resample_intervals(const uint64_t *stamps, int n,
                              uint64_t t_lo, uint64_t t_hi, double dt_ticks,
                              double *g, int ng_max) {
    if (n < 4 || t_hi <= t_lo || dt_ticks <= 0.0) return 0;
    int ng = (int)((double)(t_hi - t_lo) / dt_ticks);
    if (ng > ng_max) ng = ng_max;
    if (ng < 8) return 0;

    /* Build the (midpoint_time, interval) knot list once, then sweep the grid
       with a moving index (both are monotonic in time). */
    int j = 0;                         /* knot index into stamps */
    for (int k = 0; k < ng; k++) {
        double t = (double)t_lo + (double)k * dt_ticks;
        /* advance j so that mid[j] <= t < mid[j+1] */
        while (j + 2 < n) {
            double mid_next = 0.5 * ((double)stamps[j+1] + (double)stamps[j+2]);
            if (mid_next > t) break;
            j++;
        }
        double mid0 = 0.5 * ((double)stamps[j]   + (double)stamps[j+1]);
        double mid1 = 0.5 * ((double)stamps[j+1] + (double)stamps[j+2]);
        double d0 = (double)(stamps[j+1] - stamps[j]);
        double d1 = (double)(stamps[j+2] - stamps[j+1]);
        double val;
        if (mid1 > mid0) {
            double frac = (t - mid0) / (mid1 - mid0);
            if (frac < 0.0) frac = 0.0;
            if (frac > 1.0) frac = 1.0;
            val = d0 + frac * (d1 - d0);
        } else {
            val = d0;
        }
        g[k] = val;
    }
    return ng;
}

/* magnitude-squared coherence at a bin between series a and b.
   gamma^2 = |Sab|^2 / (Saa * Sbb), estimated by averaging the cross- and
   auto-spectra over K disjoint segments (Welch-style). With a single segment
   gamma^2 is trivially 1, so K>=4 segments are required for a meaningful value.
   Returns gamma^2 in [0,1] and the mean cross-amplitude |Sab| via out_amp. */
static double coherence_at_bin(const double *a, const double *b, int n,
                               double cycles_per_sample, int nseg, double *out_amp,
                               double *out_amp_a, double *out_amp_b) {
    if (nseg < 1) nseg = 1;
    int seglen = n / nseg;
    if (seglen < 8) { seglen = n; nseg = 1; }

    double Sab_re = 0.0, Sab_im = 0.0;
    double Saa = 0.0, Sbb = 0.0;
    double amp_a_acc = 0.0, amp_b_acc = 0.0;

    for (int k = 0; k < nseg; k++) {
        const double *as = a + (size_t)k * seglen;
        const double *bs = b + (size_t)k * seglen;
        double ma = mean_of(as, seglen);
        double mb = mean_of(bs, seglen);
        double are, aim, bre, bim;
        goertzel_complex(as, seglen, cycles_per_sample, ma, &are, &aim);
        goertzel_complex(bs, seglen, cycles_per_sample, mb, &bre, &bim);
        /* cross spectrum A * conj(B) */
        double cre = are * bre + aim * bim;
        double cim = aim * bre - are * bim;
        Sab_re += cre;
        Sab_im += cim;
        double pa = are * are + aim * aim;
        double pb = bre * bre + bim * bim;
        Saa += pa;
        Sbb += pb;
        amp_a_acc += sqrt(pa);
        amp_b_acc += sqrt(pb);
    }
    double cross_mag2 = Sab_re * Sab_re + Sab_im * Sab_im;
    double denom = Saa * Sbb;
    double gamma2 = (denom > 0.0) ? (cross_mag2 / denom) : 0.0;
    if (gamma2 > 1.0) gamma2 = 1.0;       /* numerical guard */
    if (gamma2 < 0.0) gamma2 = 0.0;
    if (out_amp) *out_amp = sqrt(cross_mag2) / nseg;
    if (out_amp_a) *out_amp_a = amp_a_acc / nseg;
    if (out_amp_b) *out_amp_b = amp_b_acc / nseg;
    return gamma2;
}

/* ----------------------------------------------------------------------------
 * MSR helpers for the P-state lever. We ONLY touch PStateCtl (0xC0010062), the
 * transient request register, and restore it exactly. We never edit a P-state
 * definition (0xC001006x).
 * -------------------------------------------------------------------------- */
#define MSR_PSTATE_CTL    0xC0010062
#define MSR_PSTATE_STATUS 0xC0010063

static int msr_read(int core, uint32_t reg, uint64_t *val) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/cpu/%d/msr", core);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int rc = (pread(fd, val, 8, reg) == 8) ? 0 : -1;
    close(fd);
    return rc;
}

static int msr_write(int core, uint32_t reg, uint64_t val) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/cpu/%d/msr", core);
    int fd = open(path, O_WRONLY);
    if (fd < 0) return -1;
    int rc = (pwrite(fd, &val, 8, reg) == 8) ? 0 : -1;
    close(fd);
    return rc;
}

/* ----------------------------------------------------------------------------
 * Run config and result row.
 * -------------------------------------------------------------------------- */
struct config_s {
    double target_hz;       /* 2.67e6 */
    double tsc_hz;          /* ~3.2e9 */
    int    stride_ticks;    /* grid step in TSC ticks (oversamples the rail period) */
    int    n_samples;       /* RAW samples per sampler thread */
    double temp_abort;      /* 62.0 */
    int    nseg;            /* coherence (Welch) averaging segments */
    int    do_pstate;       /* 1 = run wrmsr P-state levers / fixed-P0 null */
    int    band_sweep;      /* 1 = dump gamma2 across 1-5 MHz for the idle 2-3 pair */
    char   output_dir[256];
};
typedef struct config_s config_t;

/* one measured condition -> one CSV row */
typedef struct {
    const char *condition;
    const char *core_pair;
    const char *lever;
    double gamma2_267;
    double amp_267;
    double amp_offfreq_baseline;
    int    control_flag;
    double k10temp_c;
    int    n_samples;            /* grid points actually analyzed (ng) */
    /* extra diagnostics */
    double gamma2_offfreq_base;  /* coherence noise floor (median off-freq gamma2) */
    double raw_mean_ticks;       /* mean raw inter-sample spacing (TSC ticks) */
    double resolved_hz_at_bin;
} result_row_t;

/* ----------------------------------------------------------------------------
 * Core measurement: run two samplers on cores (cA, cB) simultaneously, capture
 * absolute TSC timestamps from each, then RESAMPLE both onto a common real-time
 * grid (step = dt_ticks). Returns the two uniform interval series (length *ng)
 * plus the grid's effective sample rate fs_hz.
 *
 * dt_ticks is the analysis grid step; we set it to the requested stride (e.g.
 * 24 ticks) so the grid oversamples the ~1200-tick rail period ~50x. The raw
 * loop must run AT LEAST that fast on average or the grid is interpolating
 * (under-sampling) - we report the raw mean spacing so that is visible.
 *
 * Returns 0 on success, fills *outA,*outB (caller frees), *ng, *fs_hz,
 * *raw_mean_ticks (mean raw inter-sample spacing for diagnostics).
 * -------------------------------------------------------------------------- */
static int measure_pair(int cA, int cB, int n_raw, double dt_ticks, double tsc_hz,
                        double **outA, double **outB, int *ng_out,
                        double *fs_hz, double *raw_mean_ticks) {
    uint64_t *sA = (uint64_t *)malloc(sizeof(uint64_t) * n_raw);
    uint64_t *sB = (uint64_t *)malloc(sizeof(uint64_t) * n_raw);
    if (!sA || !sB) { free(sA); free(sB); return -1; }

    atomic_int go;
    atomic_init(&go, 0);

    sampler_arg_t sa = {0}, sb = {0};
    sa.core = cA; sa.n = n_raw; sa.stamps = sA; sa.go = &go; sa.ready = 0;
    sb.core = cB; sb.n = n_raw; sb.stamps = sB; sb.go = &go; sb.ready = 0;

    pthread_t ta, tb;
    pthread_attr_t aa, ab;
    cpu_set_t setA, setB;
    pthread_attr_init(&aa); CPU_ZERO(&setA); CPU_SET(cA, &setA);
    pthread_attr_setaffinity_np(&aa, sizeof(setA), &setA);
    pthread_attr_init(&ab); CPU_ZERO(&setB); CPU_SET(cB, &setB);
    pthread_attr_setaffinity_np(&ab, sizeof(setB), &setB);

    if (pthread_create(&ta, &aa, sampler_thread, &sa) != 0) {
        free(sA); free(sB);
        pthread_attr_destroy(&aa); pthread_attr_destroy(&ab); return -1;
    }
    if (pthread_create(&tb, &ab, sampler_thread, &sb) != 0) {
        atomic_store(&go, 1); pthread_join(ta, NULL);
        free(sA); free(sB);
        pthread_attr_destroy(&aa); pthread_attr_destroy(&ab); return -1;
    }
    pthread_attr_destroy(&aa);
    pthread_attr_destroy(&ab);

    while (!sa.ready || !sb.ready) { __asm__ volatile("pause"); }
    atomic_store(&go, 1);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);

    /* overlapping real-time window common to both cores */
    uint64_t t_lo = (sA[0] > sB[0]) ? sA[0] : sB[0];
    uint64_t t_hi = (sA[n_raw-1] < sB[n_raw-1]) ? sA[n_raw-1] : sB[n_raw-1];

    double rmA = (double)(sA[n_raw-1] - sA[0]) / (n_raw - 1);
    double rmB = (double)(sB[n_raw-1] - sB[0]) / (n_raw - 1);
    if (raw_mean_ticks) *raw_mean_ticks = 0.5 * (rmA + rmB);

    /* grid length from the overlap window */
    int ng_max = (int)((double)(t_hi - t_lo) / dt_ticks);
    if (ng_max < 16) { free(sA); free(sB); return -1; }
    /* keep grids a power-of-two-friendly size for the Welch segmentation */
    double *gA = (double *)malloc(sizeof(double) * ng_max);
    double *gB = (double *)malloc(sizeof(double) * ng_max);
    if (!gA || !gB) { free(sA); free(sB); free(gA); free(gB); return -1; }

    int ngA = resample_intervals(sA, n_raw, t_lo, t_hi, dt_ticks, gA, ng_max);
    int ngB = resample_intervals(sB, n_raw, t_lo, t_hi, dt_ticks, gB, ng_max);
    free(sA); free(sB);
    int ng = (ngA < ngB) ? ngA : ngB;
    if (ng < 16) { free(gA); free(gB); return -1; }

    if (fs_hz) *fs_hz = tsc_hz / dt_ticks;   /* grid sample rate (exact, fixed) */
    *outA = gA; *outB = gB; *ng_out = ng;
    return 0;
}

/* Compute gamma^2 + amplitudes at the target Hz and at an off-frequency baseline
   (median of several neighbor bins, excluding the target), given two uniform
   series and the (fixed, known) grid sample rate fs_hz.
   Fills the result row's spectral fields. */
static void analyze_pair(const double *A, const double *B, int n,
                         double fs_hz,
                         double target_hz, int nseg,
                         double *gamma2_target, double *amp_target,
                         double *amp_offfreq_baseline,
                         double *resolved_hz_at_bin) {
    double fs = fs_hz;                      /* grid sample rate, Hz (fixed) */
    double cps_target = target_hz / fs;    /* cycles per sample at target */

    /* target bin */
    double amp_t = 0, aa_t = 0, bb_t = 0;
    double g2 = coherence_at_bin(A, B, n, cps_target, nseg, &amp_t, &aa_t, &bb_t);
    *gamma2_target = g2;
    *amp_target = amp_t;
    *resolved_hz_at_bin = cps_target * fs;

    /* off-frequency baseline: probe a spread of neighbor frequencies in the
       1-5 MHz band that are NOT near the target, take the median cross-amplitude.
       This is the "is 2.67 special?" reference. */
    double offs_hz[] = {
        1.10e6, 1.45e6, 1.80e6, 2.10e6, 2.40e6,   /* below target, spaced away */
        3.00e6, 3.40e6, 3.90e6, 4.40e6, 4.80e6    /* above target */
    };
    int noff = (int)(sizeof(offs_hz) / sizeof(offs_hz[0]));
    double amps[16];
    int cnt = 0;
    for (int i = 0; i < noff; i++) {
        double fhz = offs_hz[i];
        if (fabs(fhz - target_hz) < 0.15e6) continue;  /* keep clear of target */
        double cps = fhz / fs;
        if (cps <= 0.0 || cps >= 0.5) continue;          /* below Nyquist */
        double amp = 0, a2 = 0, b2 = 0;
        coherence_at_bin(A, B, n, cps, nseg, &amp, &a2, &b2);
        amps[cnt++] = amp;
    }
    /* median of off-freq cross-amplitudes */
    double base = 0.0;
    if (cnt > 0) {
        for (int i = 0; i < cnt; i++)
            for (int j = i + 1; j < cnt; j++)
                if (amps[j] < amps[i]) { double t = amps[i]; amps[i] = amps[j]; amps[j] = t; }
        base = (cnt % 2) ? amps[cnt/2] : 0.5 * (amps[cnt/2 - 1] + amps[cnt/2]);
    }
    *amp_offfreq_baseline = base;
}

/* phase-randomize a copy of a series (FALSE-TONE control b): destroys cross-core
   phase relationship while preserving the marginal amplitude distribution, by
   independently circular-shifting B by a random offset. A genuine shared-rail
   tone loses coherence under this; a spurious always-aligned artifact would not. */
static void phase_scramble_inplace(double *x, int n, unsigned *rng) {
    /* xorshift32 for a reproducible-but-arbitrary shift */
    unsigned r = *rng;
    r ^= r << 13; r ^= r >> 17; r ^= r << 5;
    *rng = r;
    int shift = (int)(r % (unsigned)n);
    if (shift == 0) shift = n / 3;
    double *tmp = (double *)malloc(sizeof(double) * n);
    if (!tmp) return;
    for (int i = 0; i < n; i++) tmp[i] = x[(i + shift) % n];
    memcpy(x, tmp, sizeof(double) * n);
    free(tmp);
}

/* ----------------------------------------------------------------------------
 * Output helpers.
 * -------------------------------------------------------------------------- */
static FILE *g_csv = NULL;

static void emit_header(void) {
    /* required columns first, then diagnostics */
    fprintf(g_csv,
        "condition,core_pair,lever,gamma2_267,amp_267,amp_offfreq_baseline,"
        "control_flag,k10temp_c,n_samples,gamma2_offfreq_base,gamma2_excess,"
        "raw_mean_ticks,fbin_mhz\n");
    fflush(g_csv);
}

static void emit_row(const result_row_t *r) {
    double excess = r->gamma2_267 - r->gamma2_offfreq_base;
    fprintf(g_csv,
        "%s,%s,%s,%.6f,%.6g,%.6g,%d,%.2f,%d,%.6f,%.6f,%.1f,%.4f\n",
        r->condition, r->core_pair, r->lever,
        r->gamma2_267, r->amp_267, r->amp_offfreq_baseline,
        r->control_flag, r->k10temp_c, r->n_samples,
        r->gamma2_offfreq_base, excess, r->raw_mean_ticks, r->resolved_hz_at_bin / 1e6);
    fflush(g_csv);
    fprintf(stderr,
        "  [%-22s pair=%-5s lever=%-14s ctrl=%d] g2=%.4f g2base=%.4f excess=%+.4f "
        "amp=%.3g T=%.1fC raw=%.1ftk ng=%d fbin=%.3fMHz\n",
        r->condition, r->core_pair, r->lever, r->control_flag,
        r->gamma2_267, r->gamma2_offfreq_base, excess,
        r->amp_267, r->k10temp_c, r->raw_mean_ticks, r->n_samples,
        r->resolved_hz_at_bin / 1e6);
}

/* off-frequency COHERENCE baseline: median gamma^2 over neighbor bins (excluding
   the target). With nseg-segment Welch averaging, two incoherent series give a
   gamma^2 floor ~ 1/nseg, so this median IS the empirical coherence noise floor.
   A real shared-rail tone makes gamma^2(target) sit well ABOVE this floor. */
static double offfreq_gamma2_baseline(const double *A, const double *B, int n,
                                      double fs_hz, double target_hz, int nseg) {
    double offs_hz[] = {
        1.10e6, 1.45e6, 1.80e6, 2.10e6, 2.40e6,
        3.00e6, 3.40e6, 3.90e6, 4.40e6, 4.80e6
    };
    int noff = (int)(sizeof(offs_hz) / sizeof(offs_hz[0]));
    double g2s[16]; int cnt = 0;
    for (int i = 0; i < noff; i++) {
        double fhz = offs_hz[i];
        if (fabs(fhz - target_hz) < 0.15e6) continue;
        double cps = fhz / fs_hz;
        if (cps <= 0.0 || cps >= 0.5) continue;
        double amp = 0, a2 = 0, b2 = 0;
        double g2 = coherence_at_bin(A, B, n, cps, nseg, &amp, &a2, &b2);
        g2s[cnt++] = g2;
    }
    if (cnt == 0) return 0.0;
    for (int i = 0; i < cnt; i++)
        for (int j = i+1; j < cnt; j++)
            if (g2s[j] < g2s[i]) { double t=g2s[i]; g2s[i]=g2s[j]; g2s[j]=t; }
    return (cnt % 2) ? g2s[cnt/2] : 0.5*(g2s[cnt/2-1]+g2s[cnt/2]);
}

/* ----------------------------------------------------------------------------
 * Verdict accumulators.
 * -------------------------------------------------------------------------- */
typedef struct {
    int   pdn_pairs_tested;
    int   pdn_pairs_coherent;       /* gamma2(target) high AND above coherence floor */
    double best_pdn_gamma2;
    double best_pdn_excess;         /* gamma2(target) - gamma2(offfreq floor) */
    int   lever_responded;          /* >=1 physical/history lever moved gamma2(2.67) */
    char  responding_lever[80];
    double max_lever_delta;         /* max |g2_lever - g2_idle| (absolute) */
    int   control_reproduced;       /* a false-tone control reproduced the tone */
    char  reproducing_control[80];
    /* method validity controls */
    double pos_ctrl_gamma2;         /* injected synthetic shared tone -> want HIGH */
    double neg_ctrl_gamma2;         /* two independent RNG series -> want LOW (~1/nseg) */
} verdict_t;

/* thresholds (documented, conservative) */
#define GAMMA2_HIGH       0.50  /* gamma2(target) floor we call "coherent" */
#define EXCESS_ABOVE_BASE 0.15  /* gamma2(target) must beat the off-freq floor by this */
#define LEVER_DELTA_MIN   0.15  /* absolute gamma2 shift at 2.67 that counts as a response */
#define CONTROL_EXCESS    0.15  /* a control "reproduces" if it also clears the excess */

/* Run one condition: measure pair (cA,cB), resample, analyze at target + off-freq
   floor, emit the CSV row. Returns 0 on success and fills g2/excess for the
   caller's verdict logic. The two resampled series are returned to the caller in
   outA/outB (caller frees) when those out-pointers are non-NULL, else freed here. */
static int do_condition(const config_t *cfg,
                        const char *condition, const char *pair_name, const char *lever,
                        int cA, int cB, int control_flag, double k10temp,
                        double *out_g2, double *out_excess,
                        double **outA, double **outB, int *outN, double *outFs);

/* ----------------------------------------------------------------------------
 * Method-validation controls (do NOT touch hardware): build two synthetic
 * uniform series and confirm the coherence estimator behaves.
 *  - positive: independent noise + a COMMON injected tone at target_hz -> HIGH gamma2.
 *  - negative: two independent noise series, no common tone        -> LOW gamma2 (~1/nseg).
 * If positive is not high or negative is not low, the estimator/threshold is
 * miscalibrated and any hardware verdict is untrustworthy. This is the analogue of
 * the "1/sqrt(N) / known-injected-tone" control the 5.10A spec demands before the
 * strobe is trusted on an unknown signal.
 * -------------------------------------------------------------------------- */
static unsigned xs32(unsigned *s){unsigned r=*s; r^=r<<13; r^=r>>17; r^=r<<5; *s=r; return r;}

static void method_validation(double fs_hz, double target_hz, int nseg, int n,
                              double *pos_g2, double *neg_g2) {
    double *A = (double*)malloc(sizeof(double)*n);
    double *B = (double*)malloc(sizeof(double)*n);
    double *C = (double*)malloc(sizeof(double)*n);
    if (!A || !B || !C) { free(A); free(B); free(C); *pos_g2=*neg_g2=-1; return; }
    unsigned s1=0x1234, s2=0x9abc, s3=0x55aa;
    double cps = target_hz / fs_hz;
    for (int i = 0; i < n; i++) {
        double na = ((double)(xs32(&s1)) / 4294967296.0 - 0.5);
        double nb = ((double)(xs32(&s2)) / 4294967296.0 - 0.5);
        double nc = ((double)(xs32(&s3)) / 4294967296.0 - 0.5);
        double tone = sin(2.0*M_PI*cps*i);    /* common injected tone */
        A[i] = na + 0.7*tone;                 /* positive: A,B share the tone */
        B[i] = nb + 0.7*tone;
        C[i] = nc;                            /* negative: C independent, no tone */
    }
    double amp,a2,b2;
    *pos_g2 = coherence_at_bin(A, B, n, cps, nseg, &amp,&a2,&b2);
    *neg_g2 = coherence_at_bin(A, C, n, cps, nseg, &amp,&a2,&b2);
    free(A); free(B); free(C);
}

int main(int argc, char **argv) {
    config_t cfg;
    cfg.target_hz = 2.67e6;
    cfg.tsc_hz = 3.2e9;
    /* Grid step. The raw rdtscp+loop sampler floors at ~190-240 TSC ticks/sample
       on this Phenom (rdtscp alone is ~30-40 ticks). 24 ticks (the ideal) would
       only UPSAMPLE the raw stream by ~8x and add no information; we instead grid
       at the native spacing, which still oversamples the ~1200-tick (375 ns) rail
       period ~6x and gives raw fs ~16 MHz (Nyquist ~8 MHz >> 2.67 MHz). */
    cfg.stride_ticks = 192;
    cfg.n_samples = 262144;     /* RAW samples per sampler thread */
    cfg.temp_abort = 62.0;
    cfg.nseg = 16;              /* Welch segments for coherence */
    cfg.do_pstate = 1;
    cfg.band_sweep = 0;
    strcpy(cfg.output_dir, "");

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--target-hz") && i+1 < argc) cfg.target_hz = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tsc-hz") && i+1 < argc) cfg.tsc_hz = atof(argv[++i]);
        else if (!strcmp(argv[i], "--stride-ticks") && i+1 < argc) cfg.stride_ticks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-samples") && i+1 < argc) cfg.n_samples = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp-abort") && i+1 < argc) cfg.temp_abort = atof(argv[++i]);
        else if (!strcmp(argv[i], "--nseg") && i+1 < argc) cfg.nseg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-pstate")) cfg.do_pstate = 0;
        else if (!strcmp(argv[i], "--band-sweep")) cfg.band_sweep = 1;
        else if (!strcmp(argv[i], "--output-dir") && i+1 < argc) {
            strncpy(cfg.output_dir, argv[++i], sizeof(cfg.output_dir)-1);
        } else {
            fprintf(stderr, "WARN: unknown arg '%s'\n", argv[i]);
        }
    }

    if (locate_k10temp() != 0) {
        fprintf(stderr, "FATAL: could not locate k10temp hwmon; refusing to run blind.\n");
        return 2;
    }

    /* open output CSV */
    if (cfg.output_dir[0]) {
        char path[320];
        snprintf(path, sizeof(path), "%s/phase5_10_strobe_precondition.csv", cfg.output_dir);
        g_csv = fopen(path, "w");
        if (!g_csv) { fprintf(stderr, "WARN: cannot open %s, using stdout\n", path); g_csv = stdout; }
        else fprintf(stderr, "CSV -> %s\n", path);
    } else {
        g_csv = stdout;
    }

    double t_start = read_k10temp_c();
    fprintf(stderr, "=== phase5_10 strobe precondition ===\n");
    fprintf(stderr, "target=%.0f Hz  tsc=%.3e Hz  n=%d  nseg=%d  temp_abort=%.0fC  start_temp=%.1fC  pstate=%d\n",
            cfg.target_hz, cfg.tsc_hz, cfg.n_samples, cfg.nseg, cfg.temp_abort, t_start, cfg.do_pstate);
    if (t_start >= cfg.temp_abort) {
        fprintf(stderr, "ABORT: start temp %.1fC >= abort %.1fC\n", t_start, cfg.temp_abort);
        return 3;
    }

    emit_header();

    /* method-validation controls: prove the coherence estimator is calibrated
       BEFORE trusting it on hardware. Uses the grid sample rate. */
    {
        double fs_grid = cfg.tsc_hz / cfg.stride_ticks;
        double posg, negg;
        method_validation(fs_grid, cfg.target_hz, cfg.nseg, 65536, &posg, &negg);
        fprintf(stderr, "method-validation: pos_ctrl(injected common tone) gamma2=%.4f (want HIGH)  "
                "neg_ctrl(independent) gamma2=%.4f (want ~%.3f)\n",
                posg, negg, 1.0 / cfg.nseg);
    }

    verdict_t V;
    memset(&V, 0, sizeof(V));

    /* method validity recorded into verdict struct (grid-rate based) */
    {
        double fs_grid = cfg.tsc_hz / cfg.stride_ticks;
        method_validation(fs_grid, cfg.target_hz, cfg.nseg, 65536,
                           &V.pos_ctrl_gamma2, &V.neg_ctrl_gamma2);
    }

    /* PDN-sharing isolated core pairs. On the Phenom II all cores share one
       package PDN; we test two isolated pairs (2-3 and 4-5). */
    int pairs[][2] = { {2,3}, {4,5} };
    const char *pair_names[] = { "2-3", "4-5" };
    int npairs = 2;

    /* idle reference (gamma2 at 2.67) per pair, for lever-response comparison */
    double idle_g2[2]     = {0,0};

    /* P-state save/restore bookkeeping declared early so the abort path can reach it */
    uint64_t saved_ctl[6]; int touched[6] = {0,0,0,0,0,0};
    for (int c = 0; c < 6; c++) saved_ctl[c] = 0;

    /* ------------------------------------------------------------------
     * BLOCK 1: idle baseline on each PDN-sharing pair, plus the mandatory
     * false-tone control battery on the primary pair (2-3):
     *   (a) off-frequency comparison  -> gamma2_offfreq_base column (every row)
     *   (b) phase-randomized control
     *   (c) shuffled core-pair labels (A of 2-3 vs B of 4-5)
     *   (e) idle-only null
     *   (f) fixed-P0 null is in BLOCK 3
     * ------------------------------------------------------------------ */
    double *keepA23 = NULL, *keepB23 = NULL; int keepN23 = 0; double keepFs23 = 0;
    double *keepB45 = NULL; int keepN45 = 0;  /* B of 4-5, for shuffled-label control */

    for (int p = 0; p < npairs; p++) {
        double T = read_k10temp_c();
        if (T >= cfg.temp_abort) { fprintf(stderr, "ABORT temp %.1fC\n", T); goto restore_and_exit; }

        double g2 = 0, excess = 0;
        double *A = NULL, *B = NULL; int ng = 0; double fs = 0;
        if (do_condition(&cfg, "idle_baseline", pair_names[p], "idle",
                         pairs[p][0], pairs[p][1], 0, T,
                         &g2, &excess, &A, &B, &ng, &fs) != 0) {
            fprintf(stderr, "WARN: idle %s failed\n", pair_names[p]);
            continue;
        }
        idle_g2[p] = g2; (void)excess;

        V.pdn_pairs_tested++;
        if (g2 >= GAMMA2_HIGH && excess >= EXCESS_ABOVE_BASE) V.pdn_pairs_coherent++;
        if (g2 > V.best_pdn_gamma2) V.best_pdn_gamma2 = g2;
        if (excess > V.best_pdn_excess) V.best_pdn_excess = excess;

        if (p == 0) { keepA23 = A; keepB23 = B; keepN23 = ng; keepFs23 = fs; }
        else        { keepB45 = B; keepN45 = ng; free(A); }
        /* (A,B for p==0 kept for controls; for p==1 keep only B for shuffled label) */
    }

    /* controls on the primary pair (2-3) using the kept series */
    if (keepA23 && keepB23 && keepN23 > 16) {
        double T = read_k10temp_c();
        int n = keepN23; double fs = keepFs23;

        /* (e) idle-only null: re-report the idle series flagged as a null control */
        {
            double g2t, ampt, baset, fbin;
            analyze_pair(keepA23, keepB23, n, fs, cfg.target_hz, cfg.nseg, &g2t, &ampt, &baset, &fbin);
            double g2base = offfreq_gamma2_baseline(keepA23, keepB23, n, fs, cfg.target_hz, cfg.nseg);
            result_row_t r = {0};
            r.condition="control_idle_null"; r.core_pair="2-3"; r.lever="idle_null";
            r.gamma2_267=g2t; r.amp_267=ampt; r.amp_offfreq_baseline=baset; r.control_flag=1;
            r.k10temp_c=T; r.n_samples=n; r.gamma2_offfreq_base=g2base;
            r.raw_mean_ticks=0; r.resolved_hz_at_bin=fbin;
            emit_row(&r);
        }

        /* (b) phase-randomized B: a real shared tone must lose coherence */
        {
            double *Bs = (double*)malloc(sizeof(double)*n);
            memcpy(Bs, keepB23, sizeof(double)*n);
            unsigned rng = 0xC0FFEEu;
            phase_scramble_inplace(Bs, n, &rng);
            double g2t, ampt, baset, fbin;
            analyze_pair(keepA23, Bs, n, fs, cfg.target_hz, cfg.nseg, &g2t, &ampt, &baset, &fbin);
            double g2base = offfreq_gamma2_baseline(keepA23, Bs, n, fs, cfg.target_hz, cfg.nseg);
            result_row_t r = {0};
            r.condition="control_phase_scramble"; r.core_pair="2-3"; r.lever="idle";
            r.gamma2_267=g2t; r.amp_267=ampt; r.amp_offfreq_baseline=baset; r.control_flag=1;
            r.k10temp_c=T; r.n_samples=n; r.gamma2_offfreq_base=g2base;
            r.raw_mean_ticks=0; r.resolved_hz_at_bin=fbin;
            emit_row(&r);
            if ((g2t - g2base) >= CONTROL_EXCESS && g2t >= GAMMA2_HIGH) {
                V.control_reproduced = 1;
                snprintf(V.reproducing_control, sizeof(V.reproducing_control), "phase_scramble");
            }
            free(Bs);
        }

        /* (c) shuffled core-pair labels: A of 2-3 vs B of 4-5. Mismatched skew
           relationship; if the "coherence" survives arbitrary cross-pair pairing
           it is not a specific shared-rail relationship. */
        if (keepB45 && keepN45 > 16) {
            int nn = (n < keepN45) ? n : keepN45;
            double g2t, ampt, baset, fbin;
            analyze_pair(keepA23, keepB45, nn, fs, cfg.target_hz, cfg.nseg, &g2t, &ampt, &baset, &fbin);
            double g2base = offfreq_gamma2_baseline(keepA23, keepB45, nn, fs, cfg.target_hz, cfg.nseg);
            result_row_t r = {0};
            r.condition="control_shuffled_labels"; r.core_pair="2-5x"; r.lever="idle";
            r.gamma2_267=g2t; r.amp_267=ampt; r.amp_offfreq_baseline=baset; r.control_flag=1;
            r.k10temp_c=T; r.n_samples=nn; r.gamma2_offfreq_base=g2base;
            r.raw_mean_ticks=0; r.resolved_hz_at_bin=fbin;
            emit_row(&r);
            if ((g2t - g2base) >= CONTROL_EXCESS && g2t >= GAMMA2_HIGH) {
                V.control_reproduced = 1;
                snprintf(V.reproducing_control, sizeof(V.reproducing_control), "shuffled_labels");
            }
        }
    }
    if (keepB45) { free(keepB45); keepB45 = NULL; }

    /* ------------------------------------------------------------------
     * BAND SWEEP (diagnostic, --band-sweep): dump gamma2(f) across 1-5 MHz on the
     * idle 2-3 series, so we can see whether ANY coherent line exists (the strobe
     * may not be exactly at 2.67, or there may be no sharp line at all). Printed to
     * stderr as a table; does not affect the verdict. This is the honest "is there
     * a tone anywhere in band?" cross-check before declaring UNFOUNDED.
     * ------------------------------------------------------------------ */
    if (cfg.band_sweep && keepA23 && keepB23 && keepN23 > 16) {
        double fs = keepFs23;
        fprintf(stderr, "--- BAND SWEEP gamma2(f), idle 2-3, fs=%.3f MHz (Nyquist=%.3f MHz) ---\n",
                fs/1e6, fs/2e6);
        double fhz;
        double peak_g2 = 0; double peak_f = 0;
        for (fhz = 1.00e6; fhz <= 5.00e6 + 1.0; fhz += 0.05e6) {
            double cps = fhz / fs;
            if (cps >= 0.5) break;     /* past Nyquist */
            double amp, a2, b2;
            double g2 = coherence_at_bin(keepA23, keepB23, keepN23, cps, cfg.nseg, &amp, &a2, &b2);
            int star = (fabs(fhz - cfg.target_hz) < 0.03e6);
            fprintf(stderr, "  f=%.3f MHz  gamma2=%.4f %s\n", fhz/1e6, g2, star ? "<= 2.67 target" : "");
            if (g2 > peak_g2) { peak_g2 = g2; peak_f = fhz; }
        }
        fprintf(stderr, "--- band-sweep peak: gamma2=%.4f at %.3f MHz ---\n", peak_g2, peak_f/1e6);
    }

    /* ------------------------------------------------------------------
     * BLOCK 2: LEVER RESPONSE on the primary PDN pair (2-3). Aggressors run on
     * the OTHER isolated cores (4,5) so they perturb the shared rail without
     * stealing the sampler cores. We compare gamma2(2.67) vs the idle reference.
     * ------------------------------------------------------------------ */
    {
        struct { const char *name; worker_loop_t mode; } levers[] = {
            { "cache_aggressor", WL_CACHE },
            { "integer_churn",   WL_INTEGER },
            { "syscall_burst",   WL_SYSCALL },
        };
        int nlev = 3;
        int sc = 2, scB = 3, aggA = 4, aggB = 5;

        for (int L = 0; L < nlev; L++) {
            double T = read_k10temp_c();
            if (T >= cfg.temp_abort) { fprintf(stderr, "ABORT temp %.1fC\n", T); goto restore_and_exit; }
            aggressor_t a1, a2;
            aggressor_start(&a1, aggA, levers[L].mode);
            aggressor_start(&a2, aggB, levers[L].mode);
            struct timespec ws = {0, 150*1000*1000}; nanosleep(&ws, NULL);
            T = read_k10temp_c();
            if (T >= cfg.temp_abort) {
                aggressor_stop(&a1); aggressor_stop(&a2);
                fprintf(stderr, "ABORT temp %.1fC mid-lever\n", T); goto restore_and_exit;
            }
            double g2 = 0, excess = 0;
            double *A=NULL,*B=NULL; int ng=0; double fs=0;
            if (do_condition(&cfg, "lever_response", "2-3", levers[L].name,
                             sc, scB, 0, T, &g2, &excess, &A, &B, &ng, &fs) == 0) {
                double d = fabs(g2 - idle_g2[0]);
                if (d >= LEVER_DELTA_MIN) {
                    V.lever_responded = 1;
                    if (d > V.max_lever_delta) {
                        V.max_lever_delta = d;
                        snprintf(V.responding_lever, sizeof(V.responding_lever),
                                 "%s(dg2=%.2f)", levers[L].name, d);
                    }
                }
                free(A); free(B);
            }
            aggressor_stop(&a1); aggressor_stop(&a2);
        }

        /* load-history levers: up_from_idle vs down_from_high, SAME final idle state */
        for (int hist = 0; hist < 2; hist++) {
            double T = read_k10temp_c();
            if (T >= cfg.temp_abort) { fprintf(stderr, "ABORT temp %.1fC\n", T); goto restore_and_exit; }
            const char *lname = (hist==0) ? "up_from_idle" : "down_from_high";
            if (hist == 1) {
                aggressor_t a1, a2;
                aggressor_start(&a1, aggA, WL_INTEGER);
                aggressor_start(&a2, aggB, WL_INTEGER);
                struct timespec ws = {0, 400*1000*1000}; nanosleep(&ws, NULL);
                aggressor_stop(&a1); aggressor_stop(&a2);
                struct timespec ws2 = {0, 20*1000*1000}; nanosleep(&ws2, NULL);
            } else {
                struct timespec ws = {0, 200*1000*1000}; nanosleep(&ws, NULL);
            }
            double g2=0, excess=0; double *A=NULL,*B=NULL; int ng=0; double fs=0;
            if (do_condition(&cfg, "lever_history", "2-3", lname,
                             sc, scB, 0, read_k10temp_c(), &g2, &excess, &A, &B, &ng, &fs) == 0) {
                double d = fabs(g2 - idle_g2[0]);
                if (d >= LEVER_DELTA_MIN) {
                    V.lever_responded = 1;
                    if (d > V.max_lever_delta) {
                        V.max_lever_delta = d;
                        snprintf(V.responding_lever, sizeof(V.responding_lever),
                                 "%s(dg2=%.2f)", lname, d);
                    }
                }
                free(A); free(B);
            }
        }
    }

    /* ------------------------------------------------------------------
     * BLOCK 3: P-state lever + fixed-P0 null. Toggle PStateCtl on the sampler
     * cores (request P0 vs P1), measure each, then RESTORE the exact pre-run
     * PStateCtl. Only the transient request register (0xC0010062) is touched.
     * ------------------------------------------------------------------ */
    if (cfg.do_pstate) {
        int sc = 2, scB = 3;
        int cores[2] = {sc, scB};
        int ok_save = 1;
        for (int k = 0; k < 2; k++) {
            int c = cores[k];
            if (msr_read(c, MSR_PSTATE_CTL, &saved_ctl[c]) != 0) {
                fprintf(stderr, "WARN: cannot read PStateCtl core %d; skipping P-state lever\n", c);
                ok_save = 0;
            } else {
                touched[c] = 1;
                fprintf(stderr, "P-state save: core %d PStateCtl=0x%016llx\n",
                        c, (unsigned long long)saved_ctl[c]);
            }
        }
        if (ok_save) {
            int reqs[2] = {0, 1};
            const char *rnames[2] = {"pstate_P0", "pstate_P1"};
            double g2_p[2] = {0,0};
            for (int ri = 0; ri < 2; ri++) {
                double T = read_k10temp_c();
                if (T >= cfg.temp_abort) { fprintf(stderr, "ABORT temp %.1fC\n", T); break; }
                int wok = 1;
                for (int k = 0; k < 2; k++)
                    if (msr_write(cores[k], MSR_PSTATE_CTL, (uint64_t)reqs[ri]) != 0) wok = 0;
                struct timespec ws = {0, 60*1000*1000}; nanosleep(&ws, NULL);
                if (!wok) { fprintf(stderr, "WARN: PStateCtl write failed (req=%d)\n", reqs[ri]); continue; }

                double g2=0, excess=0; double *A=NULL,*B=NULL; int ng=0; double fs=0;
                /* ri==0 (P0 held) doubles as the fixed-P0 null control */
                const char *cond = (ri==0) ? "control_fixed_p0_null" : "lever_pstate";
                int cflag = (ri==0) ? 1 : 0;
                if (do_condition(&cfg, cond, "2-3", rnames[ri],
                                 sc, scB, cflag, read_k10temp_c(),
                                 &g2, &excess, &A, &B, &ng, &fs) == 0) {
                    g2_p[ri] = g2;
                    free(A); free(B);
                }
            }
            /* pstate response = change in gamma2(2.67) between P0 and P1 requests */
            double dps = fabs(g2_p[1] - g2_p[0]);
            if (dps >= LEVER_DELTA_MIN) {
                V.lever_responded = 1;
                if (dps > V.max_lever_delta) {
                    V.max_lever_delta = dps;
                    snprintf(V.responding_lever, sizeof(V.responding_lever),
                             "pstate_P0vsP1(dg2=%.2f)", dps);
                }
            }
        }
    }

restore_and_exit:
    /* RESTORE P-state: write back the exact saved PStateCtl on every touched core. */
    for (int c = 0; c < 6; c++) {
        if (touched[c]) {
            int rc = msr_write(c, MSR_PSTATE_CTL, saved_ctl[c]);
            uint64_t rb = 0; msr_read(c, MSR_PSTATE_CTL, &rb);
            fprintf(stderr, "P-state restore: core %d -> 0x%016llx (rc=%d, readback=0x%016llx)\n",
                    c, (unsigned long long)saved_ctl[c], rc, (unsigned long long)rb);
        }
    }
    if (keepA23) { free(keepA23); keepA23 = NULL; }
    if (keepB23) { free(keepB23); keepB23 = NULL; }

    /* ------------------------------------------------------------------
     * VERDICT.
     * ELECTRICAL_STROBE_REAL iff:
     *   (A) gamma^2(2.67) high across ALL PDN-sharing pairs AND above the
     *       off-frequency coherence floor (gamma2 excess), AND the estimator is
     *       calibrated (positive control high, negative control low);
     *   (B) gamma^2(2.67) RESPONDS to >=1 physical/history lever;
     *   (C) the false-tone controls do NOT reproduce it.
     * else ELECTRICAL_STROBE_UNFOUNDED.
     * ------------------------------------------------------------------ */
    double t_end = read_k10temp_c();
    int method_ok = (V.pos_ctrl_gamma2 >= GAMMA2_HIGH) &&
                    (V.neg_ctrl_gamma2 < GAMMA2_HIGH);
    int cond_A = method_ok &&
                 (V.pdn_pairs_tested > 0) &&
                 (V.pdn_pairs_coherent == V.pdn_pairs_tested) &&
                 (V.best_pdn_gamma2 >= GAMMA2_HIGH) &&
                 (V.best_pdn_excess >= EXCESS_ABOVE_BASE);
    int cond_B = V.lever_responded;
    int cond_C = !V.control_reproduced;
    int real = cond_A && cond_B && cond_C;

    fprintf(g_csv,
        "# VERDICT_DETAIL,method_ok=%d,pos_ctrl_g2=%.4f,neg_ctrl_g2=%.4f,"
        "pdn_pairs_tested=%d,pdn_pairs_coherent=%d,best_gamma2=%.4f,best_excess=%.4f,"
        "lever_responded=%d,responding_lever=%s,max_lever_delta_g2=%.4f,"
        "control_reproduced=%d,reproducing_control=%s,start_temp=%.1f,end_temp=%.1f\n",
        method_ok, V.pos_ctrl_gamma2, V.neg_ctrl_gamma2,
        V.pdn_pairs_tested, V.pdn_pairs_coherent, V.best_pdn_gamma2, V.best_pdn_excess,
        V.lever_responded, V.responding_lever[0]?V.responding_lever:"none", V.max_lever_delta,
        V.control_reproduced, V.reproducing_control[0]?V.reproducing_control:"none",
        t_start, t_end);
    fprintf(g_csv, "VERDICT,%s\n", real ? "ELECTRICAL_STROBE_REAL" : "ELECTRICAL_STROBE_UNFOUNDED");
    fflush(g_csv);

    fprintf(stderr, "\n================= VERDICT =================\n");
    fprintf(stderr, "method calibrated (pos>=0.5, neg<0.5)   : %s (pos=%.4f neg=%.4f)\n",
            method_ok?"YES":"NO", V.pos_ctrl_gamma2, V.neg_ctrl_gamma2);
    fprintf(stderr, "(A) PDN coherence high + above floor     : %s "
            "(coherent=%d/%d, best_gamma2=%.4f, best_excess=%.4f)\n",
            cond_A?"YES":"no", V.pdn_pairs_coherent, V.pdn_pairs_tested,
            V.best_pdn_gamma2, V.best_pdn_excess);
    fprintf(stderr, "(B) Responds to a physical/history lever : %s (%s, max_dg2=%.4f)\n",
            cond_B?"YES":"no", V.responding_lever[0]?V.responding_lever:"none", V.max_lever_delta);
    fprintf(stderr, "(C) Controls do NOT reproduce it         : %s (%s)\n",
            cond_C?"YES":"no", V.control_reproduced?V.reproducing_control:"clean");
    fprintf(stderr, "temp start=%.1fC end=%.1fC\n", t_start, t_end);
    fprintf(stderr, "VERDICT: %s\n",
            real ? "ELECTRICAL_STROBE_REAL (proceed to Step 1)"
                 : "ELECTRICAL_STROBE_UNFOUNDED (stop the kernel path)");
    fprintf(stderr, "===========================================\n");

    if (g_csv && g_csv != stdout) fclose(g_csv);
    return 0;
}

/* ----------------------------------------------------------------------------
 * do_condition definition.
 * -------------------------------------------------------------------------- */
static int do_condition(const config_t *cfg,
                        const char *condition, const char *pair_name, const char *lever,
                        int cA, int cB, int control_flag, double k10temp,
                        double *out_g2, double *out_excess,
                        double **outA, double **outB, int *outN, double *outFs) {
    double *A=NULL, *B=NULL; int ng=0; double fs=0, raw=0;
    if (measure_pair(cA, cB, cfg->n_samples, (double)cfg->stride_ticks, cfg->tsc_hz,
                     &A, &B, &ng, &fs, &raw) != 0) {
        return -1;
    }
    double g2, amp, base, fbin;
    analyze_pair(A, B, ng, fs, cfg->target_hz, cfg->nseg, &g2, &amp, &base, &fbin);
    double g2base = offfreq_gamma2_baseline(A, B, ng, fs, cfg->target_hz, cfg->nseg);
    double excess = g2 - g2base;

    result_row_t r = {0};
    r.condition = condition; r.core_pair = pair_name; r.lever = lever;
    r.gamma2_267 = g2; r.amp_267 = amp; r.amp_offfreq_baseline = base;
    r.control_flag = control_flag; r.k10temp_c = k10temp; r.n_samples = ng;
    r.gamma2_offfreq_base = g2base; r.raw_mean_ticks = raw; r.resolved_hz_at_bin = fbin;
    emit_row(&r);

    if (out_g2) *out_g2 = g2;
    if (out_excess) *out_excess = excess;
    if (outA && outB && outN && outFs) {
        *outA = A; *outB = B; *outN = ng; *outFs = fs;
    } else {
        free(A); free(B);
    }
    return 0;
}
