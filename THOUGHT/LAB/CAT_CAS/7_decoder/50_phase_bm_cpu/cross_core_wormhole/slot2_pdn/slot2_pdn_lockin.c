/*
 * slot2_pdn_lockin.c
 *
 * Cross-core .holo traversal SLOT 2 -- DRIVEN POWER-RAIL LOCK-IN.
 *
 * Carries a .holo footprint (MODE in {0..3} + relational phase tag theta) as a
 * DRIVEN power-draw pattern on a SENDER core, recovered on a RECEIVER core via the
 * PROVEN Exp 5.10 cross-core power-rail lock-in: a register/L1-only alu_burst power
 * virus on one core measurably shifts a victim core's software ring-oscillator
 * timing through the shared package power-delivery network (IR drop). The rail reads
 * INSTANTANEOUS current (no retained occupancy), so the Slot-1 cache failure modes
 * (private-L2 prime+probe, shared-L3-only contention) do not apply here.
 *
 * ARCHITECTURE: two userspace PROCESSES, cross-process, NOT two threads:
 *   - sender   (--role sender):   pinned to a sender core (default 3). Per bin, gates
 *                                 the alu_burst (register/L1-only) ON/OFF as a 50%-duty
 *                                 square wave at the bin tone f_b, generated from rdtsc
 *                                 deadlines so the phase is exactly known. The per-bin
 *                                 +/-1 sign is carried as the drive square-wave PHASE
 *                                 (0 vs pi); a global theta is a shared drive-phase
 *                                 offset. Tones are time-multiplexed, ONE bin per slot
 *                                 (one proven single-tone 5.10 drive at a time -- avoids
 *                                 multi-tone intermod).
 *   - receiver (--role receiver): pinned to the victim core (default 2). Reads the
 *                                 victim ring-osc timing at ~read_hz and, per bin,
 *                                 locks in at f_b (drive bin) and at an off-bin floor.
 *                                 Emits the per-symbol 12-bin complex lock-in vector.
 *
 * SHARED ABSOLUTE-TSC ORIGIN: the TSC is constant_tsc + nonstop_tsc, coherent
 * package-wide. The two processes agree a single absolute t0 = rdtsc_now()+~100ms via
 * a tiny handshake file (sender writes t0 + the symbol schedule; receiver reads it,
 * spins to t0, and both run the same time-multiplexed bin schedule off the SAME clock).
 * Every bin's drive phase is reconstructable on the receiver from t0 + the per-bin tone.
 *
 * MODES (two run modes):
 *   --mode preflight : the GATE. Sender drives ONE tone f_b with the register/L1-only
 *                      alu_burst; receiver locks in at f_b plus an off-bin floor and
 *                      reports SNR_eff against the live OS/thermal background. If the
 *                      single-bin SNR_eff < ~3 the faint compute-only signal is buried
 *                      -> STOP (cheap kill before the full matrix).
 *   --mode matrix    : the full run. NBIN=12 log-spaced non-harmonic tones in [20,1500]
 *                      Hz, time-multiplexed one bin per slot. A train of symbols, each
 *                      a (MODE,theta) drawn from the schedule, is sent: 4-symbol preamble
 *                      then real/pseudo/wrong matched-null families. Per symbol the
 *                      receiver emits the 12-bin complex lock-in vector to a CSV the
 *                      python matched-null analyzer scores.
 *
 * CODEBOOK: 4 codewords over 12 bins, distinct flip-weights {4,5,6,7}, min pairwise
 *   Hamming 7 (the verified permutation-INEQUIVALENT fix, identical to the python
 *   pdn_catalytic_tape_fix_probe.py). The +/-1 per bin is carried as the bin drive sign.
 *
 * SAFETY (identical to the already-run 5.10 harness):
 *   - userspace only; reads APERF/MPERF/COFVID/ring-osc; the ONLY write is the cpufreq
 *     P-state pin (scaling_min==scaling_max + boost=0), restored exactly on exit.
 *   - k10temp veto: abort if >= veto (default 68 C), checked before EVERY drive slot.
 *   - ASCII only. -lm LAST at link (the 5.10 link-order requirement).
 *
 * The alu_burst / ring-osc reader / k10temp / pin primitives are lifted VERBATIM from
 * phase5_10_driven_lockin.c (READ-ONLY source; not modified).
 *
 * Build:  gcc -O2 -pthread -Wall -Wextra slot2_pdn_lockin.c -o slot2 -lm
 * Usage:
 *   one-box orchestration (default): a single invocation forks sender+receiver:
 *     ./slot2 --mode preflight --victim 2 --sender 3 [--bin-hz 200] [--slot-s 0.8] ...
 *     ./slot2 --mode matrix    --victim 2 --sender 3 --out-csv OUT.csv --seed S ...
 *   explicit two-process (debug): --role sender / --role receiver with a shared
 *     --handshake FILE (the orchestrator uses /tmp/slot2_hs_<pid>).
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
#include <sys/wait.h>
#include <sys/stat.h>

/* ======================= timing primitives (VERBATIM 5.10) ================= */
static inline uint64_t rdtscp_now(void) {
    unsigned hi, lo, aux;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) : : );
    return ((uint64_t)hi << 32) | lo;
}
static inline uint64_t rdtsc_now(void) {
    unsigned hi, lo;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi) : : );
    return ((uint64_t)hi << 32) | lo;
}

/* ======================= affinity (VERBATIM 5.10) ========================= */
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

/* ======================= k10temp (VERBATIM 5.10) ========================== */
static char g_k10_path[512] = {0};
static int locate_k10temp(void) {
    DIR *d = opendir("/sys/class/hwmon");
    if (!d) return -1;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "hwmon", 5) != 0) continue;
        char namep[512], buf[64];
        snprintf(namep, sizeof(namep), "/sys/class/hwmon/%s/name", e->d_name);
        FILE *f = fopen(namep, "r");
        if (!f) continue;
        if (fgets(buf, sizeof(buf), f)) {
            size_t n = strlen(buf);
            while (n && (buf[n-1] == '\n' || buf[n-1] == ' ')) buf[--n] = 0;
            if (strcmp(buf, "k10temp") == 0) {
                snprintf(g_k10_path, sizeof(g_k10_path),
                         "/sys/class/hwmon/%s/temp1_input", e->d_name);
                fclose(f); closedir(d); return 0;
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

/* ======================= MSR reads (VERBATIM 5.10) ======================== */
#define MSR_MPERF        0xE7
#define MSR_APERF        0xE8
#define MSR_COFVID_STATUS 0xC0010071
static int msr_read(int core, uint32_t reg, uint64_t *val) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/cpu/%d/msr", core);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int rc = (pread(fd, val, 8, reg) == 8) ? 0 : -1;
    close(fd);
    return rc;
}
static int msr_open(int core) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/cpu/%d/msr", core);
    return open(path, O_RDONLY);
}
static inline int msr_pread(int fd, uint32_t reg, uint64_t *val) {
    return (pread(fd, val, 8, reg) == 8) ? 0 : -1;
}
static int cofvid_pstate(int core) {
    uint64_t v = 0;
    if (msr_read(core, MSR_COFVID_STATUS, &v) != 0) return -1;
    return (int)(v & 0x7);
}

/* ======================= cpufreq P-state pin (VERBATIM 5.10) =============== */
#define NCPU_MAX 6
typedef struct {
    int   pinned_ok, boost_present, boost_orig;
    long  min_orig[NCPU_MAX], max_orig[NCPU_MAX];
    int   have_orig[NCPU_MAX];
    long  pin_khz;
} pinstate_t;
static int read_long_file(const char *path, long *out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long v; int rc = (fscanf(f, "%ld", &v) == 1) ? 0 : -1;
    fclose(f);
    if (rc == 0) *out = v;
    return rc;
}
static int write_long_file(const char *path, long v) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    int rc = (fprintf(f, "%ld\n", v) > 0) ? 0 : -1;
    fclose(f);
    return rc;
}
static int pin_pstate(pinstate_t *ps, long pin_khz) {
    memset(ps, 0, sizeof(*ps));
    ps->pin_khz = pin_khz;
    long b;
    if (read_long_file("/sys/devices/system/cpu/cpufreq/boost", &b) == 0) {
        ps->boost_present = 1; ps->boost_orig = (int)b;
        if (write_long_file("/sys/devices/system/cpu/cpufreq/boost", 0) != 0)
            fprintf(stderr, "WARN: could not disable cpufreq boost\n");
    } else { ps->boost_present = 0; fprintf(stderr, "NOTE: no cpufreq/boost knob\n"); }
    int any = 0;
    for (int c = 0; c < NCPU_MAX; c++) {
        char pmin[128], pmax[128];
        snprintf(pmin, sizeof(pmin), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_min_freq", c);
        snprintf(pmax, sizeof(pmax), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", c);
        long omn, omx;
        if (read_long_file(pmin, &omn) == 0 && read_long_file(pmax, &omx) == 0) {
            ps->min_orig[c] = omn; ps->max_orig[c] = omx; ps->have_orig[c] = 1;
            write_long_file(pmin, pin_khz < omx ? pin_khz : omx);
            write_long_file(pmax, pin_khz);
            write_long_file(pmin, pin_khz);
            any = 1;
        } else ps->have_orig[c] = 0;
    }
    ps->pinned_ok = any;
    return any ? 0 : -1;
}
static void restore_pstate(pinstate_t *ps) {
    for (int c = 0; c < NCPU_MAX; c++) {
        if (!ps->have_orig[c]) continue;
        char pmin[128], pmax[128];
        snprintf(pmin, sizeof(pmin), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_min_freq", c);
        snprintf(pmax, sizeof(pmax), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", c);
        write_long_file(pmin, ps->min_orig[c]);
        write_long_file(pmax, ps->max_orig[c]);
        write_long_file(pmin, ps->min_orig[c]);
        long rmn = -1, rmx = -1;
        read_long_file(pmin, &rmn); read_long_file(pmax, &rmx);
        fprintf(stderr, "P-state restore: cpu%d min->%ld(rb=%ld) max->%ld(rb=%ld)\n",
                c, ps->min_orig[c], rmn, ps->max_orig[c], rmx);
    }
    if (ps->boost_present) {
        write_long_file("/sys/devices/system/cpu/cpufreq/boost", ps->boost_orig);
        long rb = -1; read_long_file("/sys/devices/system/cpu/cpufreq/boost", &rb);
        fprintf(stderr, "P-state restore: boost->%d (rb=%ld)\n", ps->boost_orig, rb);
    }
}
static long read_cur_khz(int core) {
    char p[128];
    snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", core);
    long v; return (read_long_file(p, &v) == 0) ? v : -1;
}

/* ======================= alu_burst power virus (VERBATIM 5.10) =============
 * REGISTER/L1-ONLY power virus: long dependent SSE-double + 64-bit integer chains
 * that saturate the execution units (high dI/dt) but touch NO shared resource beyond
 * registers/L1. Reaches a remote core ONLY through the shared power rail. This is the
 * DRIVE. Lifted verbatim from phase5_10_driven_lockin.c. */
static double alu_burst(uint64_t *iseed) {
    double a0=1.0000001,a1=1.0000002,a2=1.0000003,a3=1.0000004;
    double a4=1.0000005,a5=1.0000006,a6=1.0000007,a7=1.0000008;
    uint64_t i0=*iseed^0x9E3779B97F4A7C15ULL, i1=i0*2654435761ULL+1;
    uint64_t i2=i1^0xD1B54A32D192ED03ULL,     i3=i2*1099511628211ULL+1;
    for (int k = 0; k < 64; k++) {
        a0 = a0*1.0000000007 + 0.9999999993;
        a1 = a1*0.9999999993 + 1.0000000007;
        a2 = a2*1.0000000011 + 0.9999999989;
        a3 = a3*0.9999999989 + 1.0000000011;
        a4 = a4*1.0000000013 + 0.9999999987;
        a5 = a5*0.9999999987 + 1.0000000013;
        a6 = a6*1.0000000003 + 0.9999999997;
        a7 = a7*0.9999999997 + 1.0000000003;
        i0 = i0*6364136223846793005ULL + 1442695040888963407ULL;
        i1 = i1*3935559000370003845ULL + 2691343689449507681ULL;
        i2 = i2*0x2545F4914F6CDD1DULL   + 0x14057B7EF767814FULL;
        i3 = i3*0x9E3779B97F4A7C15ULL   + 0xBF58476D1CE4E5B9ULL;
        a0 += (double)((i0 >> 40) & 0x3);
        a4 += (double)((i2 >> 40) & 0x3);
    }
    *iseed = i0 ^ i1 ^ i2 ^ i3;
    return a0+a1+a2+a3+a4+a5+a6+a7
         + (double)((i0 ^ i1 ^ i2 ^ i3) & 0xff);
}

/* ======================= lock-in (VERBATIM 5.10 math) =====================
 * Demodulate sampled x[i] at times t_tsc[i] against a reference at f_ref_hz with a
 * known phase origin t0_tsc on the SAME tsc clock. Mean-removed, Hann-windowed.
 * Returns I, Q, mag. For the off-bin floor pass a non-bin f_ref_hz. */
static void lockin(const uint64_t *t_tsc, const double *x, int n,
                   double f_ref_hz, uint64_t t0_tsc, double tsc_hz,
                   double phase_offset,
                   double *out_I, double *out_Q, double *out_mag) {
    if (n < 4) { *out_I = *out_Q = *out_mag = 0.0; return; }
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    double I = 0.0, Q = 0.0, wsum = 0.0;
    for (int i = 0; i < n; i++) {
        double win = 0.5 * (1.0 - cos(2.0 * M_PI * i / (n - 1)));
        double dt = (double)(t_tsc[i] - t0_tsc) / tsc_hz;
        double ph = 2.0 * M_PI * f_ref_hz * dt + phase_offset;
        double v = (x[i] - mean) * win;
        I += v * cos(ph);
        Q += v * sin(ph);
        wsum += win;
    }
    if (wsum <= 0.0) wsum = 1.0;
    I = 2.0 * I / wsum;
    Q = 2.0 * Q / wsum;
    *out_I = I; *out_Q = Q;
    *out_mag = sqrt(I * I + Q * Q);
}

/* ===========================================================================
 * .holo codebook -- 4 codewords over NBIN bins, distinct flip-weights {4,5,6,7},
 * min pairwise Hamming 7 (permutation-INEQUIVALENT). Generated deterministically by
 * the SAME random search as pdn_catalytic_tape_fix_probe.py (seed 7), emitted to the
 * CSV header as a comment so the analyzer/sim use the identical book. Computed once
 * at startup and shared (read-only) by sender + receiver via the handshake file.
 * =========================================================================== */
#define NBIN_MAX 16
#define MODES 4

/* deterministic codebook search matching the python make_codebook(seed=7):
 * 4 codewords with weights {4,5,6,7}, maximize min pairwise Hamming. We replicate
 * numpy default_rng(7) choice() with our own RNG ONLY for determinism on THIS box;
 * the receiver and sender both run this identical routine so they agree. The exact
 * book is also written to the handshake so both processes are byte-identical. */
typedef struct { double cw[MODES][NBIN_MAX]; int nbin; int minham; } codebook_t;

/* simple xorshift for the (deterministic) flip-position choice */
static uint64_t cb_rng_state;
static uint64_t cb_rng(void){ uint64_t x=cb_rng_state; x^=x<<13; x^=x>>7; x^=x<<17; cb_rng_state=x; return x; }
static void cb_choice(int nbin, int w, int *idx) {
    /* sample w distinct positions in [0,nbin) (partial Fisher-Yates) */
    int pool[NBIN_MAX];
    for (int i=0;i<nbin;i++) pool[i]=i;
    for (int i=0;i<w;i++) {
        int j = i + (int)(cb_rng() % (uint64_t)(nbin - i));
        int t=pool[i]; pool[i]=pool[j]; pool[j]=t;
        idx[i]=pool[i];
    }
}
static int hamming(const double *a, const double *b, int n){ int d=0; for(int i=0;i<n;i++) if(a[i]!=b[i]) d++; return d; }

static void make_codebook(codebook_t *cb, int nbin, int seed) {
    int weights[MODES] = {4,5,6,7};
    cb->nbin = nbin;
    double best[MODES][NBIN_MAX]; int best_d = -1;
    cb_rng_state = 0x243F6A8885A308D3ULL ^ (uint64_t)seed;  /* deterministic */
    for (int it=0; it<4000; it++) {
        double C[MODES][NBIN_MAX];
        for (int m=0;m<MODES;m++) {
            for (int i=0;i<nbin;i++) C[m][i]=1.0;
            int idx[NBIN_MAX];
            cb_choice(nbin, weights[m], idx);
            for (int j=0;j<weights[m];j++) C[m][idx[j]] = -1.0;
        }
        int d = 1<<30;
        for (int i=0;i<MODES;i++) for (int j=i+1;j<MODES;j++) {
            int h = hamming(C[i],C[j],nbin); if (h<d) d=h;
        }
        if (d > best_d) { best_d=d; memcpy(best,C,sizeof(best)); }
    }
    memcpy(cb->cw, best, sizeof(best));
    cb->minham = best_d;
}

/* ===========================================================================
 * Sender drive: per-bin gated square wave of the alu_burst at tone f_b, with the
 * +/-1 bin sign carried as a pi phase flip of the drive square wave, and a global
 * theta added as a shared drive-phase offset. The drive ON-half is when
 *   floor( (elapsed - phase_ticks) / half_ticks ) is even
 * where phase_ticks encodes (sign?pi:0)+theta as a fraction of the drive period.
 * Drive is pinned to the sender core; reaches the victim ONLY via the rail.
 * =========================================================================== */
typedef struct {
    atomic_int stop;
    pthread_t  thread;
    int        core;
    uint64_t   t0_tsc;       /* absolute shared origin */
    double     half_ticks;   /* half drive period (ticks) for THIS bin */
    double     phase_frac;   /* drive-phase offset as fraction of period [0,1) */
    int        started;
} drive_t;

static void *drive_loop(void *arg) {
    drive_t *d = (drive_t *)arg;
    pin_to_core(d->core);
    uint64_t iseed = (uint64_t)(d->core * 2246822519u + 3266489917u);
    volatile double sink = 0.0;
    double period = 2.0 * d->half_ticks;
    double phase_off = d->phase_frac * period;   /* ticks */
    while (!atomic_load(&d->stop)) {
        uint64_t now = rdtsc_now();
        double elapsed = (double)(now - d->t0_tsc) - phase_off;
        long halfidx = (long)floor(elapsed / d->half_ticks);
        int on = ((halfidx & 1L) == 0);   /* ON when even half */
        if (on) sink += alu_burst(&iseed);
        else    __asm__ volatile("pause");
    }
    (void)sink;
    return NULL;
}
static int drive_start(drive_t *d, int core, uint64_t t0, double half_ticks, double phase_frac) {
    memset(d, 0, sizeof(*d));
    d->core = core; d->t0_tsc = t0; d->half_ticks = half_ticks; d->phase_frac = phase_frac;
    atomic_init(&d->stop, 0);
    pthread_attr_t attr; pthread_attr_init(&attr);
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
    if (pthread_create(&d->thread, &attr, drive_loop, d) != 0) {
        fprintf(stderr, "WARN: drive pthread_create core %d: %s\n", core, strerror(errno));
        pthread_attr_destroy(&attr); return -1;
    }
    pthread_attr_destroy(&attr);
    d->started = 1;
    return 0;
}
static void drive_stop(drive_t *d) {
    if (!d->started) return;
    atomic_store(&d->stop, 1);
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_sec += 2;
    void *rv; pthread_timedjoin_np(d->thread, &rv, &ts);
    d->started = 0;
}

/* ===========================================================================
 * Receiver ring-osc reader (VERBATIM 5.10 reader_thread logic, single channel).
 * Pinned to the victim core. Spins a fixed tiny dependent MAC ring-osc, retaining
 * ~read_hz samples of (t_tsc, ro_period). ro_period reflects rail/thermal state.
 * =========================================================================== */
typedef struct {
    int       core;
    int       n;             /* max samples (buffer capacity) */
    int       read_hz;
    double    tsc_hz;
    uint64_t  t_deadline;    /* ABSOLUTE tsc deadline: stop capturing at/after this */
    uint64_t *t_tsc;
    double   *ro_period;
    atomic_int *go;
    volatile int ready;
    volatile int n_captured;  /* out: samples actually captured before the deadline */
} rxarg_t;

/* The reader captures UNTIL an absolute wall-clock deadline (t_deadline), NOT a fixed
 * sample count. This anchors each slot's capture window to the absolute drive schedule
 * regardless of how the ring-osc period drifts with die temperature -- otherwise a
 * fixed-count reader takes longer than the slot as the part warms, overruns t_end, and
 * the per-slot lateness COMPOUNDS until the capture window slides off the drive entirely
 * (the symbol-index magnitude collapse seen in the first matrix runs). Deadline-bounded
 * capture keeps every slot phase-aligned to the sender. */
static void *rx_thread(void *arg) {
    rxarg_t *r = (rxarg_t *)arg;
    pin_to_core(r->core);
    int n = r->n;
    volatile uint64_t acc = 0x9E3779B9u + (uint64_t)r->core;
    double target_span = r->tsc_hz / (double)r->read_hz;
    for (int i = 0; i < 8192; i++) acc = acc * 6364136223846793005ULL + 1;
    r->ready = 1;
    while (atomic_load(r->go) == 0) { __asm__ volatile("pause"); }
    uint64_t t_prev = rdtscp_now();
    int i = 0;
    for (; i < n; i++) {
        uint64_t t_now, iters = 0;
        do {
            uint64_t a = acc;
            a = a * 6364136223846793005ULL + 1442695040888963407ULL;
            acc = a;
            iters++;
            t_now = rdtscp_now();
        } while ((double)(t_now - t_prev) < target_span);
        double span = (double)(t_now - t_prev);
        r->t_tsc[i] = t_now;
        r->ro_period[i] = span / (double)iters;
        t_prev = t_now;
        if (t_now >= r->t_deadline) { i++; break; }   /* absolute deadline reached */
    }
    r->n_captured = i;
    __asm__ volatile("" :: "r"(acc));
    return NULL;
}

/* ===========================================================================
 * Bin tones: NBIN log-spaced NON-harmonic tones in [f_lo, f_hi]. We nudge each off a
 * pure log grid by an irrational-ish factor so no tone is a small-integer multiple of
 * another (avoids harmonic leakage between bins). Off-bin floor freq per bin = the
 * 5.10 offbin_freq (lands between drive bins).
 * =========================================================================== */
static void make_tones(double *f, int nbin, double f_lo, double f_hi) {
    double llo = log(f_lo), lhi = log(f_hi);
    for (int i = 0; i < nbin; i++) {
        double t = (nbin > 1) ? (double)i / (double)(nbin - 1) : 0.0;
        double base = exp(llo + (lhi - llo) * t);
        /* non-harmonic nudge: multiply by an irrational-ish per-bin factor near 1 */
        double nf = 1.0 + 0.013 * sin(2.399963 * (i + 1));  /* +/-1.3%, aperiodic */
        f[i] = base * nf;
    }
}
static double offbin_freq(double f_drive) { return f_drive * 1.37 + 0.071; }

/* ===========================================================================
 * Handshake file: sender writes the absolute t0 (and the run params it must agree on
 * with the receiver); receiver polls for it, reads t0, and both spin to t0. We use a
 * simple text line. The ORCHESTRATOR (one-box default) forks both processes so they
 * trivially share params via argv; the handshake carries t0 (the one runtime-decided
 * value) reliably across the process boundary.
 * =========================================================================== */
static int write_t0(const char *path, uint64_t t0) {
    char tmp[600]; snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "w"); if (!f) return -1;
    fprintf(f, "%llu\n", (unsigned long long)t0); fclose(f);
    return rename(tmp, path);
}
static int read_t0(const char *path, uint64_t *t0) {
    FILE *f = fopen(path, "r"); if (!f) return -1;
    unsigned long long v=0; int rc = (fscanf(f, "%llu", &v)==1)?0:-1; fclose(f);
    if (rc==0) *t0=(uint64_t)v;
    return rc;
}

/* ===========================================================================
 * Config + symbol schedule.
 * =========================================================================== */
typedef struct {
    int    role;         /* 0 orchestrate(fork both), 1 sender, 2 receiver */
    int    mode_matrix;  /* 0 preflight, 1 matrix */
    int    victim;       /* receiver core */
    int    sender;       /* sender core */
    int    nbin;
    double f_lo, f_hi;
    double slot_s;       /* seconds per bin slot */
    int    read_hz;
    double tsc_hz;
    long   pin_khz;
    int    do_pin;
    double temp_veto;
    int    seed;
    int    trials;       /* matrix: trials per family */
    int    namp_sender;  /* how many sender cores to drive with (default 1) */
    int    sender_cores[NCPU_MAX]; int n_sender_cores;
    char   out_csv[400];
    char   handshake[400];
    double bin_hz;       /* preflight single tone */
    double gap_s;        /* setup gap at the end of each slot: receiver captures only
                            cap_s = slot_s - gap_s, leaving headroom for the next slot's
                            thread spawn so captures stay absolutely aligned. */
    int    ctrl_silent;  /* NEGATIVE CONTROL: sender drives NOTHING (no alu_burst).
                            Receiver still captures + decodes; must sit at chance. */
    int    ctrl_scramble;/* NEGATIVE CONTROL: sender re-permutes each symbol's drive by
                            a per-symbol key the receiver does NOT share (decoy schedule).
                            Receiver decodes against the canonical book; must collapse to
                            chance. Proves the win is the agreed drive, not an artifact. */
} config_t;

/* deterministic per-run RNG for schedule + theta (seeded; recorded) */
static uint64_t g_rng;
static uint64_t xs(void){ uint64_t x=g_rng; x^=x<<13; x^=x>>7; x^=x<<17; g_rng=x; return x; }
static double urand(void){ return (double)(xs() >> 11) / (double)(1ULL<<53); }
static int irand(int n){ return (int)(xs() % (uint64_t)n); }

#define PHASE_LEVELS 8

/* ===========================================================================
 * SEND one symbol: drive each of the NBIN bins, one at a time (time-multiplexed),
 * with the bin's drive sign carried as a pi phase flip and the symbol theta added as
 * a global drive-phase offset. Each bin is driven for slot_s seconds starting at a
 * deterministic absolute TSC time derived from (t0, symbol_index, bin_index, slot).
 * The bins of all symbols form one contiguous schedule on the absolute clock so the
 * receiver knows exactly when each (symbol,bin) drive runs.
 *
 * codeword[bin] in {+1,-1}; sign -> phase_frac 0.0 (for +1) or 0.5 (for -1, = pi).
 * theta -> + theta/(2pi) added to phase_frac (shared global offset).
 * =========================================================================== */
static double bin_slot_start_s(int sym, int bin, int nbin, double slot_s) {
    return (double)(sym * nbin + bin) * slot_s;
}

/* ===========================================================================
 * RECEIVE one symbol: for each bin, capture the ring-osc over that bin's slot window
 * and lock in at f_b (drive bin) and the off-bin floor. Returns the 12 complex
 * lock-in components (I,Q) at the drive bin and the per-bin floor magnitudes.
 * =========================================================================== */

/* ====== the run is orchestrated below in run_* functions ====== */

/* lock in one captured slot at the drive bin + off-bin; outputs I,Q,mag,floor */
static void score_slot(const uint64_t *t, const double *ro, int n,
                       double f_b, uint64_t t0, double tsc_hz,
                       double *I, double *Q, double *mag, double *floor) {
    double i1,q1,m1,i2,q2,m2;
    lockin(t, ro, n, f_b, t0, tsc_hz, 0.0, &i1,&q1,&m1);
    lockin(t, ro, n, offbin_freq(f_b), t0, tsc_hz, 0.0, &i2,&q2,&m2);
    *I=i1; *Q=q1; *mag=m1; *floor=m2;
}

/* Capture the victim ring-osc over the ABSOLUTE window [t_start, t_start + cap_s].
 * The reader thread is created + warmed BEFORE t_start (so go fires exactly at t_start)
 * and stops at the absolute deadline t_start + cap_s. cap_s is set < slot_s by the
 * caller to leave a setup gap for the next slot's thread spawn, so captures never
 * overlap or slide. Returns the number of samples actually captured. */
static int capture_slot(const config_t *cfg, int victim, uint64_t t_start, double cap_s,
                        uint64_t *t, double *ro, int nsamp_cap) {
    atomic_int go; atomic_init(&go, 0);
    rxarg_t ra; memset(&ra, 0, sizeof(ra));
    ra.core=victim; ra.n=nsamp_cap; ra.read_hz=cfg->read_hz; ra.tsc_hz=cfg->tsc_hz;
    ra.t_deadline = t_start + (uint64_t)(cap_s * cfg->tsc_hz);
    ra.t_tsc=t; ra.ro_period=ro; ra.go=&go; ra.ready=0; ra.n_captured=0;
    pthread_t rt; pthread_attr_t attr; pthread_attr_init(&attr);
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(victim, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
    if (pthread_create(&rt, &attr, rx_thread, &ra) != 0) { pthread_attr_destroy(&attr); return -1; }
    pthread_attr_destroy(&attr);
    while (!ra.ready) { __asm__ volatile("pause"); }
    while (rdtsc_now() < t_start) { __asm__ volatile("pause"); }
    atomic_store(&go, 1);
    pthread_join(rt, NULL);
    return ra.n_captured;
}

/* ======================================================================== */
int main(int argc, char **argv);

#include "slot2_pdn_run.h"

int main(int argc, char **argv) {
    config_t cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.role = 0;          /* orchestrate (fork sender+receiver) */
    cfg.mode_matrix = 0;   /* preflight by default */
    cfg.victim = 2;
    cfg.sender = 3;
    cfg.nbin = 12;
    cfg.f_lo = 20.0; cfg.f_hi = 1500.0;
    cfg.slot_s = 0.8;
    cfg.read_hz = 4000;
    cfg.tsc_hz = 3214823000.0;  /* measured true TSC on this box (kernel-refined
                                   3214.823 MHz); used for the intra-slot bin tone.
                                   Per-slot phase reference makes the decode robust to
                                   any residual error, but the accurate rate keeps the
                                   intra-slot carrier tight. Override with --tsc-hz. */
    cfg.pin_khz = 1600000;
    cfg.do_pin = 1;
    cfg.temp_veto = 68.0;
    cfg.seed = 44;
    cfg.gap_s = 0.12;   /* >= a couple thread-spawn + warmup times of headroom */
    cfg.trials = 64;
    cfg.namp_sender = 1;
    cfg.n_sender_cores = 0;
    cfg.bin_hz = 200.0;
    strcpy(cfg.out_csv, "");
    strcpy(cfg.handshake, "");

    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"--role") && i+1<argc) {
            const char *r=argv[++i];
            if (!strcmp(r,"sender")) cfg.role=1;
            else if (!strcmp(r,"receiver")) cfg.role=2;
            else cfg.role=0;
        }
        else if (!strcmp(argv[i],"--mode") && i+1<argc) {
            const char *m=argv[++i];
            cfg.mode_matrix = (!strcmp(m,"matrix")) ? 1 : 0;
        }
        else if (!strcmp(argv[i],"--victim") && i+1<argc) cfg.victim=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--sender") && i+1<argc) cfg.sender=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--nbin") && i+1<argc) cfg.nbin=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--f-lo") && i+1<argc) cfg.f_lo=atof(argv[++i]);
        else if (!strcmp(argv[i],"--f-hi") && i+1<argc) cfg.f_hi=atof(argv[++i]);
        else if (!strcmp(argv[i],"--slot-s") && i+1<argc) cfg.slot_s=atof(argv[++i]);
        else if (!strcmp(argv[i],"--gap-s") && i+1<argc) cfg.gap_s=atof(argv[++i]);
        else if (!strcmp(argv[i],"--read-hz") && i+1<argc) cfg.read_hz=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--tsc-hz") && i+1<argc) cfg.tsc_hz=atof(argv[++i]);
        else if (!strcmp(argv[i],"--pin-khz") && i+1<argc) cfg.pin_khz=atol(argv[++i]);
        else if (!strcmp(argv[i],"--no-pin")) cfg.do_pin=0;
        else if (!strcmp(argv[i],"--temp-veto") && i+1<argc) cfg.temp_veto=atof(argv[++i]);
        else if (!strcmp(argv[i],"--seed") && i+1<argc) cfg.seed=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--trials") && i+1<argc) cfg.trials=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--bin-hz") && i+1<argc) cfg.bin_hz=atof(argv[++i]);
        else if (!strcmp(argv[i],"--namp") && i+1<argc) cfg.namp_sender=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--out-csv") && i+1<argc) strncpy(cfg.out_csv, argv[++i], sizeof(cfg.out_csv)-1);
        else if (!strcmp(argv[i],"--handshake") && i+1<argc) strncpy(cfg.handshake, argv[++i], sizeof(cfg.handshake)-1);
        else if (!strcmp(argv[i],"--silent")) cfg.ctrl_silent=1;          /* neg control */
        else if (!strcmp(argv[i],"--scramble-drive")) cfg.ctrl_scramble=1; /* neg control */
        else fprintf(stderr, "WARN: unknown arg '%s'\n", argv[i]);
    }

    if (locate_k10temp() != 0) {
        fprintf(stderr, "FATAL: cannot locate k10temp; refusing to run blind.\n");
        return 2;
    }

    /* default sender cores = {sender} unless explicitly more requested via namp.
       For namp>1 we drive sender, sender+1, ... within isolated range. */
    cfg.n_sender_cores = 0;
    for (int k=0;k<cfg.namp_sender && k<NCPU_MAX;k++) {
        int c = cfg.sender + k;
        if (c==cfg.victim) c = cfg.sender - 1; /* avoid clashing victim */
        cfg.sender_cores[cfg.n_sender_cores++] = c;
    }

    return run_orchestrate(&cfg);
}
