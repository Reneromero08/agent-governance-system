/*
 * phase5_10_driven_lockin.c
 *
 * Phase 5.10 STEP 1 (driven two-channel lock-in). Step 0 returned a clean NULL
 * reading a free ~2.67 MHz rail tone via cross-core coherence of a software ring
 * oscillator. Diagnosis: that timing loop is a ~1.5-bit THERMOMETER (sub-Hz
 * bandwidth, seconds settling); a 2.67 MHz tone is ~7 decades above its passband
 * and physically cannot arrive. This program replaces the passive free-tone search
 * with a DRIVEN, owned-phase, matched-filter measurement INSIDE the passband.
 *
 * WHAT THIS DECIDES
 *   We DRIVE the shared package rail with a known swept-frequency dI/dt stimulus
 *   (aggressor cores gating heavy load ON/OFF as a 50%-duty square wave at f_drive,
 *   generated from rdtsc deadlines so the phase is exactly known), and LOCK IN to
 *   our own drive phase on a victim core across TWO channels:
 *     (A) ring-oscillator period   -- thermal proxy, slow (sub-Hz pole)
 *     (B) APERF/MPERF effective-frequency ratio -- electrical proxy, microsecond
 *         response (actual delivered clock / reference clock)
 *   The DECISIVE discriminator is a lock-in component that PERSISTS ABOVE THE
 *   THERMAL POLE (>~10 Hz, where heating has rolled off), scales monotonically with
 *   dI/dt amplitude, collapses under a scrambled/off-bin reference, and shows core-
 *   assignment dependence. That = real electrical droop coupling. A response that is
 *   below-pole only and identical across core assignments = just heating; it carries
 *   no rail-state information and the rail is invisible to software (the conclusive
 *   obstruction result -> external rail probe physically required).
 *
 * RIG RECON (verified live on this exact box before writing):
 *   - AMD Phenom II X6 1090T (K10, 45nm), 6 cores, kernel 6.12.86, isolcpus=2-5.
 *   - MPERF=0xE7, APERF=0xE8 ARCHITECTURAL addresses (the 0xC00000E7/E8 variants
 *     return EIO on this part; the 0xE7/0xE8 reads succeed and return monotonic
 *     counters). aperfmperf + constant_tsc + nonstop_tsc + cpb all present.
 *   - PStateCtl(0xC0010062)/PStateStatus(0xC0010063) read as 0 here; the live
 *     P-state is pinned through cpufreq sysfs (scaling_min==scaling_max) plus
 *     boost=0 (the 'userspace' governor is absent; only performance/schedutil).
 *     COFVID_STATUS(0xC0010071) is recorded as a witness of the actual P-state.
 *   - PRECHECK result (measured): with P-state CLAMPED to P1 (1600 MHz) and boost
 *     OFF, APERF/MPERF ratio sits at 0.500000 (== 1600/3200) to ~5 decimals and
 *     does NOT move between idle-executing and a heavy busy loop. MPERF counts at
 *     the P0 reference rate; APERF counts delivered cycles; ratio = f_eff/f_ref.
 *     So the channel EXISTS and is stable to ~1e-5; the experiment is whether it
 *     WOBBLES at f_drive under dI/dt (adaptive clock-stretch / APERF deficit).
 *
 * METHOD (userspace only; no kernel module; reboot-recoverable):
 *   1. PIN P-state on victim + aggressors via cpufreq (min==max) and disable boost.
 *      Record before/after; RESTORE exactly on exit. This is also a CONTROL: with
 *      the divider fixed and boost off, any f_eff deviation is droop, not a governor
 *      hop. If pinning does not hold (governor escapes), we say so and that response
 *      does not count.
 *   2. PRECHECK: APERF/MPERF ratio idle vs heavy load at the pinned P-state. Report
 *      the number to the CSV (precheck rows) and stderr.
 *   3. DRIVE: aggressor cores run a 50%-duty rdtsc-deadline square wave gating the
 *      heaviest available SSE/integer/cache load at f_drive. Sweep f_drive log-spaced
 *      {0.1,0.3,1,3,10,30,100,300,1000} Hz (configurable). Amplitude ladder 1->3->5
 *      aggressor cores.
 *   4. READ both channels on the victim at >= a few kHz: (A) ring-osc period, (B)
 *      APERF/MPERF effective frequency, time-stamped with rdtscp.
 *   5. LOCK-IN: multiply each channel by sin/cos at the exact f_drive (reconstructed
 *      from the same rdtsc clock that generated the drive), integrate over the
 *      capture; output inphase, quad, magnitude per channel per point.
 *   6. CONTROLS: (a) scrambled/off-bin reference (demod at f' != f_drive); true
 *      coupling collapses to the floor, artifact survives. (b) topology: swap victim/
 *      aggressor assignments (victim=2 aggr=3,4,5 vs victim=5 aggr=2,3,4). (c) scaling:
 *      amplitude ladder x two temp bands (idle ~42C, warm ~58-62C via background load).
 *   7. OUTPUT CSV columns: f_drive,channel,inphase,quad,magnitude,amplitude_level,
 *      temp_band,k10temp_c,control_flag,victim_core,aggr_cores (+ diagnostics) and a
 *      VERDICT line.
 *   8. VERDICT: RAIL_COUPLING_OBSERVED iff an above-thermal-pole (>~10 Hz) lock-in
 *      magnitude (i) scales monotonically with dI/dt amplitude, (ii) collapses under
 *      scrambled/off-bin, (iii) shows core-assignment dependence, (iv) has the correct
 *      droop sign (f_eff dips during the load-ON half). RAIL_INVISIBLE_SOFTWARE iff
 *      only below-pole heating appears (identical across assignments) or nothing above
 *      floor at max dI/dt -> external rail probe physically required.
 *
 * SAFETY (hard):
 *   - k10temp read before EVERY drive point; HARD VETO at 68 C (respects the 70 C
 *     high; warm band targets ~60 C max). Refuse to start if k10temp cannot be found.
 *   - cpufreq P-state pin + boost are restored to the exact stock values on exit
 *     (readbacks recorded). NO kernel module, NO persistent MSR writes (we only READ
 *     MSRs: APERF/MPERF/COFVID). Bounded runtime.
 *
 * Build:  gcc -O2 -pthread -Wall -Wextra phase5_10_driven_lockin.c -o lockin -lm
 *         (-lm MUST be last so the math symbols from the source resolve against it)
 * Usage:  ./lockin [--victim N] [--aggr a,b,c] [--freqs "0.1,0.3,1,3,10,30,100,300,1000"]
 *                  [--amp-ladder "1,3,5"] [--pin-khz 1600000] [--read-hz 4000]
 *                  [--cycles 200] [--min-cap-s 0.5] [--max-cap-s 8]
 *                  [--temp-veto 68] [--warm-target 60] [--warm] [--swap-topology]
 *                  [--aggr-mode compute|memory|both] [--output-dir DIR] [--no-pin]
 *                  [--mode sweep|basin-scan]
 *                  [--basin-freq 30] [--basin-amp 3] [--settle-s "0.05,0.2,1.0,4.0"]
 *                  [--repeats 24] [--basin-seed 44] [--sham-history]
 *
 *   --mode sweep (default): the driven two-channel lock-in sweep described above.
 *   --mode basin-scan: load-history / hysteresis characterization. After settling
 *     to a BYTE-IDENTICAL final compute-bound config, does a different LOAD HISTORY
 *     (up_from_idle vs down_from_high) leave a reproducible, non-thermal,
 *     settle-persistent difference in the rail-droop witness (= a retained basin),
 *     or only an instantaneous IR drop that is thermal / decays with settle time?
 *     Basin-scan flags:
 *       [--basin-freq 30] fixed witness drive freq (Hz, above the thermal pole)
 *       [--basin-amp 3]   fixed witness amplitude (aggressor cores; clamped to naggr)
 *       [--settle-s "0.05,0.2,1.0,4.0"]  settle-time sweep (the decisive curve)
 *       [--repeats 24]    repeats per (history, settle_s); must be >= 20
 *       [--basin-seed 44] seed for BOTH the full measurement-order shuffle AND the
 *                         label-scramble control (deterministic, reproducible)
 *       [--sham-history]  SHAM-HISTORY PLACEBO: run BOTH arms with the IDENTICAL
 *                         up_from_idle prelude but keep them LABELED up vs down. A
 *                         clean apparatus must then show NO plateau + a clean
 *                         scramble; a surviving plateau is the decisive proof the
 *                         effect is drift/measurement, not load history.
 *     CONTROL 1 (always on): FULL MEASUREMENT-ORDER RANDOMIZATION. The complete
 *       list of (history, settle, rep) tasks is shuffled into one fully-randomized
 *       order (deterministic from --basin-seed) before executing, so slow drift
 *       (temperature creep) is decorrelated from the up/down labels. If the prior
 *       up-vs-down witness difference was drift, randomization makes the scramble
 *       control clean and shrinks/zeros the history plateau.
 *     Emits phase5_10_basin_scan.csv (analyze_phase5_10.py schema) + a settle-time
 *     curve summary + the within-run thermal drift (start vs end k10temp) +
 *     BASIN_VERDICT. Per-run verdicts: REAL run -> {BASIN_REAL_PLATEAU |
 *     NO_RETAINED_BASIN | ARTIFACT_CONFIRMED | AMBIGUOUS}; SHAM run -> {SHAM_CLEAN |
 *     ARTIFACT_CONFIRMED}. Cross-run rule (orchestrator): BASIN_HISTORY_DEPENDENT
 *     iff REAL=BASIN_REAL_PLATEAU AND SHAM=SHAM_CLEAN; ARTIFACT_CONFIRMED iff either
 *     run=ARTIFACT_CONFIRMED; else AMBIGUOUS. Shares ALL the sweep-mode safety
 *     (P-state pin + boost-off readback + restore-on-exit, k10temp veto before
 *     every measurement, MSR reads only).
 *
 *   --aggr-mode (default both): which aggressor TYPE(s) to run, head-to-head.
 *     compute = register/L1-only power virus (alias: alu): dependent SSE double
 *               mul/add + 64-bit integer mul/add chains, no shared L3/mem/NB traffic.
 *               A normal CPU-bound load that draws power but shares NO resource with
 *               the victim, so any above-pole coupling is via the shared power rail.
 *     memory  = shared-cache hammer (alias: cache): strided L2/L3-spilling RMW
 *               stream -> shared-resource contention reference.
 *     both    = run compute and memory across the same amplitude ladder + drive
 *               sweep so the comparison verdict can separate POWER_COUPLED vs
 *               CONTENTION_COUPLED at matched amplitude.
 *
 * ASCII only. Calm instrumentation, not drama.
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

/* ===========================================================================
 * Timing primitives.
 * rdtscp_light: ordered-enough timestamp without the cpuid pipeline flush, for
 * the fast read loop. rdtsc_now: the same, used to generate drive deadlines so
 * the drive phase is known on the SAME clock the reader stamps with.
 * =========================================================================== */
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

/* ===========================================================================
 * CPU affinity (reused pattern).
 * =========================================================================== */
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

/* ===========================================================================
 * k10temp. Locate the hwmon dir whose name == "k10temp"; read temp1_input
 * (milli-C). Returns Celsius, or -999.0 on failure (treated as a hard abort -
 * we never run a thermal burst blind).
 * =========================================================================== */
/* Large enough for "/sys/class/hwmon/" + d_name (up to NAME_MAX=255) + suffix,
   so gcc -Wextra/-Wformat-truncation does not warn on the snprintf below. */
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

/* ===========================================================================
 * MSR reads. APERF=0xE8, MPERF=0xE7 on this part (architectural addresses).
 * COFVID_STATUS=0xC0010071 is a witness of the live P-state. We ONLY read MSRs.
 * =========================================================================== */
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

/* Open a persistent MSR fd for the fast read loop (avoid open/close per sample). */
static int msr_open(int core) {
    char path[64];
    snprintf(path, sizeof(path), "/dev/cpu/%d/msr", core);
    return open(path, O_RDONLY);
}
static inline int msr_pread(int fd, uint32_t reg, uint64_t *val) {
    return (pread(fd, val, 8, reg) == 8) ? 0 : -1;
}

/* ===========================================================================
 * cpufreq P-state pin. The userspace governor is absent on this kernel, so we
 * pin the live frequency by clamping scaling_min_freq == scaling_max_freq and
 * setting boost=0. Save/restore the stock values exactly.
 *
 * We touch ALL six cores (0..5): the rail is shared package-wide, so every core
 * must hold a fixed divider for the "any deviation is droop" control to be clean,
 * and cores 0/1 host the OS so we want them quiet at a fixed divider too.
 * =========================================================================== */
#define NCPU_MAX 6

typedef struct {
    int   pinned_ok;                 /* 1 if we successfully clamped */
    int   boost_present;
    int   boost_orig;
    long  min_orig[NCPU_MAX];
    long  max_orig[NCPU_MAX];
    int   have_orig[NCPU_MAX];
    long  pin_khz;                   /* the clamped frequency, kHz */
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

    /* boost */
    long b;
    if (read_long_file("/sys/devices/system/cpu/cpufreq/boost", &b) == 0) {
        ps->boost_present = 1;
        ps->boost_orig = (int)b;
        if (write_long_file("/sys/devices/system/cpu/cpufreq/boost", 0) != 0)
            fprintf(stderr, "WARN: could not disable cpufreq boost\n");
    } else {
        ps->boost_present = 0;
        fprintf(stderr, "NOTE: no cpufreq/boost knob (CPB may be off already)\n");
    }

    /* per-core min/max clamp */
    int any = 0;
    for (int c = 0; c < NCPU_MAX; c++) {
        char pmin[128], pmax[128];
        snprintf(pmin, sizeof(pmin), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_min_freq", c);
        snprintf(pmax, sizeof(pmax), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", c);
        long omn, omx;
        if (read_long_file(pmin, &omn) == 0 && read_long_file(pmax, &omx) == 0) {
            ps->min_orig[c] = omn; ps->max_orig[c] = omx; ps->have_orig[c] = 1;
            /* set max first up then min, or min/max in an order that always keeps min<=max */
            write_long_file(pmin, pin_khz < omx ? pin_khz : omx); /* lower min toward target safely */
            write_long_file(pmax, pin_khz);
            write_long_file(pmin, pin_khz);
            any = 1;
        } else {
            ps->have_orig[c] = 0;
        }
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
        /* restore min low first, then max high, then min, to avoid min>max transient */
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

/* read the live frequency the governor settled on for a core (kHz), -1 on fail */
static long read_cur_khz(int core) {
    char p[128];
    snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", core);
    long v; return (read_long_file(p, &v) == 0) ? v : -1;
}

/* COFVID CurPstate index witness (bits 0..2 of status on K10) */
static int cofvid_pstate(int core) {
    uint64_t v = 0;
    if (msr_read(core, MSR_COFVID_STATUS, &v) != 0) return -1;
    return (int)(v & 0x7);
}

/* ===========================================================================
 * Aggressor: 50%-duty square-wave dI/dt drive. The worker spins in the heaviest
 * available SSE/integer/cache load during the ON half of each drive period and
 * idles (pause/short sleep, minimal current) during the OFF half. The phase comes
 * from rdtsc deadlines: ON when ((rdtsc - t0) / half_ticks) is even. K10 is SSE-
 * era (NO AVX); the heavy path mixes a wide SSE2 FP/MAC chain with an L2/L3-
 * spilling cache stream to maximize dI/dt per ON edge.
 * =========================================================================== */
typedef enum { DRV_OFF = 0, DRV_ON = 1 } drv_phase_t;

/* Aggressor TYPE -- the contention control. CACHE hammers shared L3 / memory
   controller / Northbridge (so its above-pole coupling to the victim can be EITHER
   rail droop OR microarchitectural contention -- confounded). ALU is a register/
   L1-only power virus: a tight dependent SSE-double + 64-bit integer arithmetic
   chain that saturates the execution units (high dI/dt) but touches NO shared
   resource beyond its own registers/L1. An ALU aggressor on remote cores can reach
   the victim ONLY through the shared power rail (and thermal, below the pole), so
   any above-pole ALU coupling to the victim timing is rail droop BY CONSTRUCTION. */
typedef enum { AGG_ALU = 0, AGG_CACHE = 1 } agg_mode_t;
/* Public-facing name in the task's vocabulary: AGG_ALU is the COMPUTE-bound
   (register/L1-only) aggressor, AGG_CACHE is the MEMORY-bound (shared-cache)
   aggressor. These strings are what land in the CSV aggr_mode column. */
static const char *agg_mode_name(agg_mode_t m) { return m == AGG_ALU ? "compute" : "memory"; }

typedef struct {
    atomic_int  stop;
    pthread_t   thread;
    int         core;
    int         started;
    uint64_t    t0_tsc;          /* shared drive origin (all aggressors agree) */
    double      half_ticks;      /* half drive period in TSC ticks */
    uint64_t   *buf;             /* cache-thrash buffer (CACHE mode only) */
    size_t      buf_words;
    int         background;      /* 1 = pure sustained heat (warm band), no gating */
    agg_mode_t  mode;            /* AGG_ALU (register/L1-only) or AGG_CACHE (shared) */
} aggressor_t;

#define AGG_BUF_BYTES (16 * 1024 * 1024)   /* > 6MB L3 -> reliable spill */

static inline uint64_t lcg_next(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL + 1442695040888963407ULL);
    return *s;
}

/* One heavy burst of load: ~fixed instruction count, maximal current. Mixes a
   dependent SSE2 double FMA-style chain (8-wide via 4 xmm pairs) with a strided
   cache write stream. Returns a live accumulator so nothing is elided. */
static double heavy_burst(volatile uint64_t *buf, size_t n, uint64_t *seed, size_t *pos) {
    /* SSE2 FP chain: compiler will vectorize these independent accumulators. */
    double a0=1.0001,a1=1.0002,a2=1.0003,a3=1.0004;
    double b0=0.9999,b1=0.9998,b2=0.9997,b3=0.9996;
    for (int k = 0; k < 64; k++) {
        a0 = a0*1.0000001 + b0; b0 = b0*0.9999999 - a0*1e-9;
        a1 = a1*1.0000001 + b1; b1 = b1*0.9999999 - a1*1e-9;
        a2 = a2*1.0000001 + b2; b2 = b2*0.9999999 - a2*1e-9;
        a3 = a3*1.0000001 + b3; b3 = b3*0.9999999 - a3*1e-9;
    }
    /* cache thrash: 512 strided RMW touches that spill L2/L3 */
    size_t p = *pos; uint64_t s = *seed;
    for (int i = 0; i < 512; i++) {
        p = (p + 8) % n;
        uint64_t v = buf[p];
        v ^= lcg_next(&s);
        v = (v << 13) | (v >> 51);
        buf[p] = v;
    }
    *pos = p; *seed = s;
    /* p is already in [0,n); fold a touched word into the sink so the cache
       stream cannot be elided. */
    return a0+a1+a2+a3+b0+b1+b2+b3 + (double)(buf[p] & 0xff);
}

/* REGISTER-ONLY POWER VIRUS (contention control). Saturates the execution units
   with LONG DEPENDENCY CHAINS so the back end stays busy every cycle (high dI/dt),
   but touches NO shared resource: everything lives in registers (8 SSE double lanes
   + 4 64-bit integer lanes), no loads/stores beyond L1, no L3 / memory controller /
   Northbridge traffic. The chains are dependent (each op feeds the next within its
   lane) so the compiler cannot collapse them; the multiple independent lanes keep
   all FP/INT issue ports filled. Matched to heavy_burst's outer count (64) so the
   ON-half work granularity is comparable. Returns a live accumulator (no elision).
   K10 is SSE-era (NO AVX): mulsd/addsd + imul/add are the heavy-current path. */
static double alu_burst(uint64_t *iseed) {
    /* SSE double dependency chains (8 lanes). FMA-style mul-then-add per lane. */
    double a0=1.0000001,a1=1.0000002,a2=1.0000003,a3=1.0000004;
    double a4=1.0000005,a5=1.0000006,a6=1.0000007,a7=1.0000008;
    /* 64-bit integer dependency chains (4 lanes): imul + add, register-only. */
    uint64_t i0=*iseed^0x9E3779B97F4A7C15ULL, i1=i0*2654435761ULL+1;
    uint64_t i2=i1^0xD1B54A32D192ED03ULL,     i3=i2*1099511628211ULL+1;
    for (int k = 0; k < 64; k++) {
        /* each lane: dependent mulsd then addsd (stays in xmm, no memory) */
        a0 = a0*1.0000000007 + 0.9999999993;
        a1 = a1*0.9999999993 + 1.0000000007;
        a2 = a2*1.0000000011 + 0.9999999989;
        a3 = a3*0.9999999989 + 1.0000000011;
        a4 = a4*1.0000000013 + 0.9999999987;
        a5 = a5*0.9999999987 + 1.0000000013;
        a6 = a6*1.0000000003 + 0.9999999997;
        a7 = a7*0.9999999997 + 1.0000000003;
        /* dependent 64-bit imul + add per lane (register-only) */
        i0 = i0*6364136223846793005ULL + 1442695040888963407ULL;
        i1 = i1*3935559000370003845ULL + 2691343689449507681ULL;
        i2 = i2*0x2545F4914F6CDD1DULL   + 0x14057B7EF767814FULL;
        i3 = i3*0x9E3779B97F4A7C15ULL   + 0xBF58476D1CE4E5B9ULL;
        /* light cross-coupling so no lane gets constant-folded out */
        a0 += (double)((i0 >> 40) & 0x3);
        a4 += (double)((i2 >> 40) & 0x3);
    }
    *iseed = i0 ^ i1 ^ i2 ^ i3;
    /* fold integer lanes into the FP sink so nothing is elided */
    return a0+a1+a2+a3+a4+a5+a6+a7
         + (double)((i0 ^ i1 ^ i2 ^ i3) & 0xff);
}

static void *aggressor_loop(void *arg) {
    aggressor_t *a = (aggressor_t *)arg;
    pin_to_core(a->core);
    volatile uint64_t *buf = (volatile uint64_t *)a->buf;
    size_t n = a->buf_words;
    uint64_t seed = (uint64_t)(a->core * 65521u + 12345u);
    uint64_t iseed = (uint64_t)(a->core * 2246822519u + 3266489917u);
    size_t pos = 0;
    volatile double sink = 0.0;
    int is_alu = (a->mode == AGG_ALU);

    if (a->background) {
        /* warm band: sustained heat, no gating, modest dI/dt (we just want temp).
           Warmers always use the cache path -- they only supply DC heat, not the
           f_drive tone, so their resource footprint does not affect the control. */
        while (!atomic_load(&a->stop)) {
            sink += heavy_burst(buf, n, &seed, &pos);
        }
        (void)sink; return NULL;
    }

    /* gated 50%-duty square wave from rdtsc deadlines. The ON-half load is the
       aggressor TYPE: ALU = register/L1-only power virus (rail-only reach to the
       victim), CACHE = shared-resource hammer (rail OR contention). */
    while (!atomic_load(&a->stop)) {
        uint64_t now = rdtsc_now();
        uint64_t elapsed = now - a->t0_tsc;
        /* phase index: ON when floor(elapsed/half) is even */
        uint64_t halfidx = (uint64_t)((double)elapsed / a->half_ticks);
        drv_phase_t ph = (halfidx & 1ULL) ? DRV_OFF : DRV_ON;
        if (ph == DRV_ON) {
            if (is_alu) sink += alu_burst(&iseed);
            else        sink += heavy_burst(buf, n, &seed, &pos);
        } else {
            /* OFF: minimal current. Spin on pause until the half elapses. */
            __asm__ volatile("pause");
        }
    }
    (void)sink;
    return NULL;
}

static int aggressor_start(aggressor_t *a, int core, uint64_t t0, double half_ticks,
                           int background, agg_mode_t mode) {
    memset(a, 0, sizeof(*a));
    a->core = core; a->t0_tsc = t0; a->half_ticks = half_ticks; a->background = background;
    a->mode = mode;
    atomic_init(&a->stop, 0);
    a->buf_words = AGG_BUF_BYTES / sizeof(uint64_t);
    /* CACHE mode (and warmers) need the L3-spilling buffer; the ALU power virus is
       register/L1-only by construction, so it allocates nothing shared. */
    if (mode == AGG_CACHE || background) {
        if (posix_memalign((void **)&a->buf, 64, AGG_BUF_BYTES) != 0 || !a->buf) {
            fprintf(stderr, "WARN: aggressor buf alloc failed core %d\n", core);
            return -1;
        }
        memset(a->buf, 0, AGG_BUF_BYTES);
    } else {
        a->buf = NULL; a->buf_words = 0;
    }
    pthread_attr_t attr; pthread_attr_init(&attr);
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
    if (pthread_create(&a->thread, &attr, aggressor_loop, a) != 0) {
        fprintf(stderr, "WARN: aggressor pthread_create core %d: %s\n", core, strerror(errno));
        pthread_attr_destroy(&attr); free(a->buf); a->buf = NULL; return -1;
    }
    pthread_attr_destroy(&attr);
    a->started = 1;
    return 0;
}

static void aggressor_stop(aggressor_t *a) {
    if (!a->started) { if (a->buf) { free(a->buf); a->buf = NULL; } return; }
    atomic_store(&a->stop, 1);
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_sec += 2;
    void *rv;
    if (pthread_timedjoin_np(a->thread, &rv, &ts) == 0) {
        if (a->buf) { free(a->buf); a->buf = NULL; }
    } else {
        fprintf(stderr, "WARN: aggressor join timeout core %d; buffer retained\n", a->core);
    }
    a->started = 0;
}

/* ===========================================================================
 * Victim reader. Pinned to the victim core. Runs a tight fixed-work ring-osc
 * inner loop and, every msr_decim iterations, reads APERF/MPERF. Each retained
 * sample stores:
 *   - t_tsc       : rdtscp timestamp (same clock as drive phase)
 *   - ro_period   : ring-osc period = ticks since last sample / iters (ticks/iter)
 *   - feff_ratio  : (dAPERF/dMPERF) since the previous MSR read = f_eff/f_ref
 * We oversample f_drive: read cadence ~ read_hz (a few kHz). The ring-osc inner
 * work is a tiny dependent MAC chain (fixed instr count) so its duration reflects
 * rail/thermal state, not data-dependent control flow.
 * =========================================================================== */
typedef struct {
    int      core;
    int      n;                  /* retained samples */
    int      read_hz;            /* target sample cadence */
    double   tsc_hz;             /* for cadence pacing */
    int      msr_fd;             /* persistent victim MSR fd */
    uint64_t *t_tsc;             /* out */
    double   *ro_period;         /* out: ring-osc ticks/iter */
    double   *feff_ratio;        /* out: dAPERF/dMPERF */
    atomic_int *go;
    volatile int ready;
    int      msr_ok;             /* out: 1 if APERF/MPERF reads worked */
} reader_arg_t;

static void *reader_thread(void *arg) {
    reader_arg_t *r = (reader_arg_t *)arg;
    pin_to_core(r->core);
    int n = r->n;
    volatile uint64_t acc = 0x9E3779B9u + (uint64_t)r->core;

    /* pacing: target read_hz samples; choose an inner iteration budget so each
       retained sample spans ~ tsc_hz/read_hz ticks. We adapt: measure one block. */
    double target_span = r->tsc_hz / (double)r->read_hz;   /* ticks per sample */

    /* warm up caches / predictors */
    for (int i = 0; i < 8192; i++) acc = acc * 6364136223846793005ULL + 1;

    r->ready = 1;
    while (atomic_load(r->go) == 0) { __asm__ volatile("pause"); }

    uint64_t mperf0 = 0, aperf0 = 0;
    int have_msr0 = 0;
    r->msr_ok = 1;
    if (r->msr_fd >= 0 &&
        msr_pread(r->msr_fd, MSR_MPERF, &mperf0) == 0 &&
        msr_pread(r->msr_fd, MSR_APERF, &aperf0) == 0) {
        have_msr0 = 1;
    } else {
        r->msr_ok = 0;
    }

    uint64_t t_prev = rdtscp_now();
    for (int i = 0; i < n; i++) {
        /* spin the ring oscillator until ~target_span ticks have elapsed */
        uint64_t t_now;
        uint64_t iters = 0;
        do {
            uint64_t a = acc;
            a = a * 6364136223846793005ULL + 1442695040888963407ULL;
            acc = a;            /* fixed tiny dependent MAC */
            iters++;
            t_now = rdtscp_now();
        } while ((double)(t_now - t_prev) < target_span);

        double span = (double)(t_now - t_prev);
        r->t_tsc[i] = t_now;
        r->ro_period[i] = span / (double)iters;   /* ring-osc period, ticks/iter */

        /* effective-frequency ratio over this sample interval */
        double feff = 0.0;
        if (have_msr0) {
            uint64_t m1 = 0, a1 = 0;
            if (msr_pread(r->msr_fd, MSR_MPERF, &m1) == 0 &&
                msr_pread(r->msr_fd, MSR_APERF, &a1) == 0) {
                uint64_t dm = m1 - mperf0, da = a1 - aperf0;
                feff = (dm > 0) ? ((double)da / (double)dm) : 0.0;
                mperf0 = m1; aperf0 = a1;
            } else {
                r->msr_ok = 0;
            }
        }
        r->feff_ratio[i] = feff;
        t_prev = t_now;
    }
    __asm__ volatile("" :: "r"(acc));
    return NULL;
}

/* ===========================================================================
 * Lock-in. Demodulate a sampled channel x[i] (sampled at times t_tsc[i]) against
 * a reference at frequency f_ref_hz with a known phase origin t0_tsc on the SAME
 * tsc clock. Returns in-phase (I), quadrature (Q), magnitude. We mean-remove and
 * Hann-window over the capture to suppress leakage / drift.
 *
 *   phase(i) = 2*pi*f_ref_hz * (t_tsc[i]-t0_tsc)/tsc_hz
 *   I = sum w_i (x_i-mean) cos(phase_i) ;  Q = sum w_i (x_i-mean) sin(phase_i)
 * normalized by sum of window so I,Q are in x-units of amplitude.
 *
 * For the scrambled-reference control we pass a phase_offset and/or an off-bin
 * f_ref_hz; true coupling at f_drive collapses, an artifact survives.
 * =========================================================================== */
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
        double win = 0.5 * (1.0 - cos(2.0 * M_PI * i / (n - 1)));   /* Hann */
        double dt = (double)(t_tsc[i] - t0_tsc) / tsc_hz;            /* seconds */
        double ph = 2.0 * M_PI * f_ref_hz * dt + phase_offset;
        double v = (x[i] - mean) * win;
        I += v * cos(ph);
        Q += v * sin(ph);
        wsum += win;
    }
    if (wsum <= 0.0) wsum = 1.0;
    /* factor 2/wsum recovers single-sided amplitude of a tone */
    I = 2.0 * I / wsum;
    Q = 2.0 * Q / wsum;
    *out_I = I; *out_Q = Q;
    *out_mag = sqrt(I * I + Q * Q);
}

/* sign of the in-phase droop: with our convention (drive ON = first half of each
   period, phase 0 at t0) a real droop makes f_eff DIP during ON. We define the
   droop sign as the sign of -I_at_drive for the feff channel (positive => correct
   droop sign: f_eff lower while load is ON). Reported, not thresholded hard. */

/* ===========================================================================
 * CSV output.
 * =========================================================================== */
static FILE *g_csv = NULL;

static void emit_header(void) {
    fprintf(g_csv,
        "f_drive,channel,aggr_mode,inphase,quad,magnitude,amplitude_level,temp_band,"
        "k10temp_c,control_flag,victim_core,aggr_cores,"
        "n_samples,read_hz_eff,floor_mag,snr,droop_sign,cofvid_pstate,cur_khz\n");
    fflush(g_csv);
}

typedef struct {
    double f_drive;
    const char *channel;       /* "ring_osc" | "feff" */
    const char *aggr_mode;     /* "compute" | "memory" | "self"(precheck) */
    double I, Q, mag;
    int    amplitude_level;    /* n aggressor cores */
    const char *temp_band;     /* "idle" | "warm" */
    double k10temp_c;
    int    control_flag;       /* 0 real, 1 scrambled/off-bin/null */
    int    victim_core;
    const char *aggr_cores;    /* e.g. "3-4-5" */
    int    n_samples;
    double read_hz_eff;
    double floor_mag;          /* off-bin/scramble floor for this point */
    double snr;                /* mag / floor_mag */
    double droop_sign;
    int    cofvid_pstate;
    long   cur_khz;
} row_t;

static void emit_row(const row_t *r) {
    fprintf(g_csv,
        "%.4f,%s,%s,%.6g,%.6g,%.6g,%d,%s,%.2f,%d,%d,%s,%d,%.1f,%.6g,%.4f,%+.3f,%d,%ld\n",
        r->f_drive, r->channel, r->aggr_mode ? r->aggr_mode : "",
        r->I, r->Q, r->mag, r->amplitude_level, r->temp_band,
        r->k10temp_c, r->control_flag, r->victim_core, r->aggr_cores,
        r->n_samples, r->read_hz_eff, r->floor_mag, r->snr, r->droop_sign,
        r->cofvid_pstate, r->cur_khz);
    fflush(g_csv);
}

/* ===========================================================================
 * Config.
 * =========================================================================== */
/* aggressor-mode selector for the contention control. AGGSEL_BOTH (default) runs
   the full alu-vs-cache head-to-head at matched amplitude ladder + drive sweep. */
typedef enum { AGGSEL_ALU = 0, AGGSEL_CACHE = 1, AGGSEL_BOTH = 2 } aggsel_t;

typedef struct {
    int    victim;
    int    aggr[NCPU_MAX];
    int    naggr;
    double freqs[32];
    int    nfreq;
    int    amp_ladder[NCPU_MAX];
    int    namp;
    long   pin_khz;
    int    read_hz;
    int    cycles;             /* target drive cycles integrated per point */
    double min_cap_s, max_cap_s;
    double temp_veto;
    double warm_target;
    int    do_warm;
    int    swap_topology;
    int    do_pin;
    aggsel_t aggr_sel;         /* alu | cache | both (default both) */
    double tsc_hz;
    char   output_dir[256];
    /* ---- mode + basin-scan parameters ---- */
    int    mode_basin;         /* 0 = sweep (default), 1 = basin-scan */
    double basin_freq;         /* fixed witness drive freq (Hz), default 30 */
    int    basin_namp;         /* fixed witness amplitude (aggressor cores), default = naggr */
    double basin_settle[16];   /* settle-time sweep (s) */
    int    basin_nsettle;      /* count */
    int    basin_repeats;      /* repeats per (history, settle_s); >= 20 */
    int    basin_seed;         /* scramble-permutation + measurement-order seed */
    int    basin_sham;         /* SHAM-HISTORY PLACEBO: both arms run the IDENTICAL
                                  up_from_idle prelude but stay LABELED up/down for
                                  analysis. A clean apparatus must then yield NO
                                  plateau + clean scramble; a surviving plateau is
                                  the decisive drift/measurement-artifact result. */
} config_t;

/* off-bin reference frequencies for the floor: midway between drive bins, where
   no real drive energy lives. The floor is the lock-in magnitude there. */
static double offbin_freq(double f_drive) {
    return f_drive * 1.37 + 0.071;   /* irrational-ish multiplier: lands between bins */
}

/* ===========================================================================
 * Run ONE drive point: set up the aggressor square wave at f_drive with `namp`
 * cores, capture both victim channels over ~cycles drive periods, lock in at the
 * drive bin and at the off-bin floor for BOTH channels, emit rows.
 *
 * Returns 0 on success, <0 on setup failure, +1 on temp veto (caller stops).
 * Fills out_feff_mag / out_feff_floor / out_droop_sign for the verdict logic.
 * =========================================================================== */
static int run_drive_point(const config_t *cfg, double f_drive, int namp,
                           int victim, const int *aggr_list, const char *aggr_label,
                           const char *temp_band, agg_mode_t mode,
                           double *out_feff_mag, double *out_feff_floor,
                           double *out_feff_droop,
                           double *out_ro_mag, double *out_ro_floor) {
    double T = read_k10temp_c();
    if (T >= cfg->temp_veto) {
        fprintf(stderr, "TEMP VETO %.1fC >= %.1fC at f=%.3f amp=%d\n",
                T, cfg->temp_veto, f_drive, namp);
        return 1;
    }

    /* capture length: enough drive cycles, bounded by min/max seconds */
    double cap_s = (double)cfg->cycles / f_drive;
    if (cap_s < cfg->min_cap_s) cap_s = cfg->min_cap_s;
    if (cap_s > cfg->max_cap_s) cap_s = cfg->max_cap_s;
    int nsamp = (int)(cap_s * cfg->read_hz);
    if (nsamp < 64) nsamp = 64;
    if (nsamp > 2000000) nsamp = 2000000;

    /* drive geometry on the TSC clock */
    double drive_period_ticks = cfg->tsc_hz / f_drive;
    double half_ticks = 0.5 * drive_period_ticks;

    /* allocate reader buffers */
    uint64_t *t_tsc = (uint64_t *)malloc(sizeof(uint64_t) * nsamp);
    double   *ro    = (double *)malloc(sizeof(double) * nsamp);
    double   *feff  = (double *)malloc(sizeof(double) * nsamp);
    if (!t_tsc || !ro || !feff) { free(t_tsc); free(ro); free(feff); return -1; }

    int msr_fd = msr_open(victim);
    if (msr_fd < 0) fprintf(stderr, "WARN: victim core %d MSR open failed; feff channel disabled\n", victim);

    /* shared drive origin: a near-future TSC so all aggressors + reader agree */
    uint64_t t0 = rdtsc_now() + (uint64_t)(0.02 * cfg->tsc_hz);  /* +20 ms lead-in */

    /* start aggressors (gated square wave) of the requested TYPE (alu/cache) */
    aggressor_t agg[NCPU_MAX];
    int started = 0;
    for (int k = 0; k < namp; k++) {
        if (aggressor_start(&agg[k], aggr_list[k], t0, half_ticks, 0, mode) == 0) started++;
    }
    if (started < namp)
        fprintf(stderr, "WARN: only %d/%d aggressors started at f=%.3f\n", started, namp, f_drive);

    /* reader */
    atomic_int go; atomic_init(&go, 0);
    reader_arg_t ra; memset(&ra, 0, sizeof(ra));
    ra.core = victim; ra.n = nsamp; ra.read_hz = cfg->read_hz; ra.tsc_hz = cfg->tsc_hz;
    ra.msr_fd = msr_fd; ra.t_tsc = t_tsc; ra.ro_period = ro; ra.feff_ratio = feff;
    ra.go = &go; ra.ready = 0; ra.msr_ok = 0;

    pthread_t rt; pthread_attr_t rattr; pthread_attr_init(&rattr);
    cpu_set_t rset; CPU_ZERO(&rset); CPU_SET(victim, &rset);
    pthread_attr_setaffinity_np(&rattr, sizeof(rset), &rset);
    if (pthread_create(&rt, &rattr, reader_thread, &ra) != 0) {
        fprintf(stderr, "ERR: reader pthread_create core %d\n", victim);
        pthread_attr_destroy(&rattr);
        for (int k = 0; k < namp; k++) aggressor_stop(&agg[k]);
        if (msr_fd >= 0) close(msr_fd);
        free(t_tsc); free(ro); free(feff);
        return -1;
    }
    pthread_attr_destroy(&rattr);

    /* wait for reader at gate, then release exactly at (or just before) t0 */
    while (!ra.ready) { __asm__ volatile("pause"); }
    while (rdtsc_now() < t0) { __asm__ volatile("pause"); }
    atomic_store(&go, 1);
    pthread_join(rt, NULL);

    /* stop aggressors */
    for (int k = 0; k < namp; k++) aggressor_stop(&agg[k]);

    /* live witnesses */
    int ps_wit = cofvid_pstate(victim);
    long khz_wit = read_cur_khz(victim);
    double T_after = read_k10temp_c();

    /* effective read cadence actually achieved */
    double span_s = (double)(t_tsc[nsamp-1] - t_tsc[0]) / cfg->tsc_hz;
    double read_hz_eff = (span_s > 0) ? ((nsamp - 1) / span_s) : 0.0;

    double f_off = offbin_freq(f_drive);
    const char *mode_str = agg_mode_name(mode);

    /* ---- ring-osc channel ---- */
    double I, Q, mag, Ifl, Qfl, magfl;
    lockin(t_tsc, ro, nsamp, f_drive, t0, cfg->tsc_hz, 0.0, &I, &Q, &mag);
    lockin(t_tsc, ro, nsamp, f_off,   t0, cfg->tsc_hz, 0.0, &Ifl, &Qfl, &magfl);
    {
        row_t r = {0};
        r.f_drive=f_drive; r.channel="ring_osc"; r.aggr_mode=mode_str; r.I=I; r.Q=Q; r.mag=mag;
        r.amplitude_level=namp; r.temp_band=temp_band; r.k10temp_c=T_after;
        r.control_flag=0; r.victim_core=victim; r.aggr_cores=aggr_label;
        r.n_samples=nsamp; r.read_hz_eff=read_hz_eff; r.floor_mag=magfl;
        r.snr=(magfl>0)?(mag/magfl):0.0; r.droop_sign=0.0;
        r.cofvid_pstate=ps_wit; r.cur_khz=khz_wit;
        emit_row(&r);
        /* off-bin control row */
        row_t rc = r; rc.channel="ring_osc"; rc.I=Ifl; rc.Q=Qfl; rc.mag=magfl;
        rc.control_flag=1; rc.snr=1.0;
        emit_row(&rc);
        if (out_ro_mag) *out_ro_mag = mag;
        if (out_ro_floor) *out_ro_floor = magfl;
    }

    /* ---- effective-frequency channel ---- */
    double fI=0, fQ=0, fmag=0, fIfl=0, fQfl=0, fmagfl=0;
    if (ra.msr_ok) {
        lockin(t_tsc, feff, nsamp, f_drive, t0, cfg->tsc_hz, 0.0, &fI, &fQ, &fmag);
        lockin(t_tsc, feff, nsamp, f_off,   t0, cfg->tsc_hz, 0.0, &fIfl, &fQfl, &fmagfl);
        /* droop sign: f_eff dips during ON half => in-phase component negative =>
           -fI > 0 is the correct droop sign. */
        double droop = -fI;
        row_t r = {0};
        r.f_drive=f_drive; r.channel="feff"; r.aggr_mode=mode_str; r.I=fI; r.Q=fQ; r.mag=fmag;
        r.amplitude_level=namp; r.temp_band=temp_band; r.k10temp_c=T_after;
        r.control_flag=0; r.victim_core=victim; r.aggr_cores=aggr_label;
        r.n_samples=nsamp; r.read_hz_eff=read_hz_eff; r.floor_mag=fmagfl;
        r.snr=(fmagfl>0)?(fmag/fmagfl):0.0; r.droop_sign=droop;
        r.cofvid_pstate=ps_wit; r.cur_khz=khz_wit;
        emit_row(&r);
        row_t rc = r; rc.channel="feff"; rc.I=fIfl; rc.Q=fQfl; rc.mag=fmagfl;
        rc.control_flag=1; rc.snr=1.0; rc.droop_sign=0.0;
        emit_row(&rc);
        if (out_feff_mag) *out_feff_mag = fmag;
        if (out_feff_floor) *out_feff_floor = fmagfl;
        if (out_feff_droop) *out_feff_droop = droop;
    } else {
        fprintf(stderr, "NOTE: feff channel unavailable at f=%.3f (MSR read failed)\n", f_drive);
        if (out_feff_mag) *out_feff_mag = 0.0;
        if (out_feff_floor) *out_feff_floor = 0.0;
        if (out_feff_droop) *out_feff_droop = 0.0;
    }

    fprintf(stderr,
        "  [f=%7.3f amp=%d vic=%d aggr=%-7s mode=%-5s %s] ro:mag=%.4g/fl=%.4g(snr=%.2f) "
        "feff:mag=%.4g/fl=%.4g(snr=%.2f droop=%+.3g) T=%.1fC ps=%d khz=%ld rd=%.0fHz n=%d\n",
        f_drive, namp, victim, aggr_label, mode_str, temp_band,
        mag, magfl, (magfl>0?mag/magfl:0),
        fmag, fmagfl, (fmagfl>0?fmag/fmagfl:0), -fI,
        T_after, ps_wit, khz_wit, read_hz_eff, nsamp);

    if (msr_fd >= 0) close(msr_fd);
    free(t_tsc); free(ro); free(feff);
    return 0;
}

/* ===========================================================================
 * PRECHECK: APERF/MPERF ratio idle vs heavy load at the pinned P-state, on the
 * victim core. Emits precheck rows (channel="precheck_idle"/"precheck_load") and
 * returns the two ratios + their delta.
 * =========================================================================== */
static void precheck(const config_t *cfg, int victim,
                     double *r_idle, double *r_load) {
    int fd = msr_open(victim);
    *r_idle = *r_load = 0.0;
    if (fd < 0) { fprintf(stderr, "PRECHECK: cannot open victim MSR\n"); return; }
    pin_to_core(victim);

    uint64_t m0, a0, m1, a1;
    struct timespec dur = {0, 200*1000*1000};

    /* idle: just sleep (core executes the reader thread minimally) */
    if (msr_pread(fd, MSR_MPERF, &m0)==0 && msr_pread(fd, MSR_APERF, &a0)==0) {
        nanosleep(&dur, NULL);
        msr_pread(fd, MSR_MPERF, &m1); msr_pread(fd, MSR_APERF, &a1);
        uint64_t dm=m1-m0, da=a1-a0;
        *r_idle = (dm>0)?((double)da/(double)dm):0.0;
    }
    /* load: heavy spin on this core for 200 ms */
    if (msr_pread(fd, MSR_MPERF, &m0)==0 && msr_pread(fd, MSR_APERF, &a0)==0) {
        uint64_t t = rdtsc_now();
        uint64_t budget = (uint64_t)(0.2 * cfg->tsc_hz);
        volatile double s = 1.0; double x0=1.0001,x1=0.9999;
        while (rdtsc_now() - t < budget) {
            for (int k=0;k<256;k++){ x0=x0*1.0000001+x1; x1=x1*0.9999999-x0*1e-9; }
            s += x0+x1;
        }
        (void)s;
        msr_pread(fd, MSR_MPERF, &m1); msr_pread(fd, MSR_APERF, &a1);
        uint64_t dm=m1-m0, da=a1-a0;
        *r_load = (dm>0)?((double)da/(double)dm):0.0;
    }
    close(fd);

    int ps = cofvid_pstate(victim);
    long khz = read_cur_khz(victim);
    double T = read_k10temp_c();
    fprintf(stderr,
        "PRECHECK victim=%d pinned_khz_target=%ld cur_khz=%ld cofvid_ps=%d: "
        "feff_ratio idle=%.6f load=%.6f delta=%.6f  T=%.1fC\n",
        victim, cfg->pin_khz, khz, ps, *r_idle, *r_load, *r_load - *r_idle, T);

    /* emit precheck rows to CSV (control_flag=2 marks precheck) */
    row_t ri = {0};
    ri.f_drive=0; ri.channel="precheck_idle"; ri.aggr_mode="self"; ri.I=*r_idle; ri.Q=0; ri.mag=*r_idle;
    ri.amplitude_level=1; ri.temp_band="idle"; ri.k10temp_c=T; ri.control_flag=2;
    ri.victim_core=victim; ri.aggr_cores="self"; ri.n_samples=0; ri.read_hz_eff=0;
    ri.floor_mag=0; ri.snr=0; ri.droop_sign=0; ri.cofvid_pstate=ps; ri.cur_khz=khz;
    emit_row(&ri);
    row_t rl = ri; rl.channel="precheck_load"; rl.I=*r_load; rl.mag=*r_load;
    emit_row(&rl);
}

/* ===========================================================================
 * BASIN-SCAN MODE (load-history / hysteresis characterization).
 *
 * QUESTION: after settling to a BYTE-IDENTICAL final aggressor config, does a
 * different LOAD HISTORY leave a distinguishable, reproducible difference in the
 * rail-droop witness that is NOT explained by temperature and does NOT decay with
 * settle time? A reproducible, non-thermal, settle-persistent up-vs-down
 * difference = a retained state (a candidate basin). A difference that is purely
 * thermal or decays with settle_s = instantaneous IR drop, no retained state.
 *
 * The witness is the EXACT same driven compute-only ring-osc lock-in validated by
 * the sweep mode (AGG_ALU, register/L1-only -> rail-only reach), at a single fixed
 * drive freq + fixed amplitude. The ONLY thing that differs between the two
 * conditions is the PATH taken to the final config; the final config and the
 * witness drive are identical (built-in matched-final-config control).
 *
 * HISTORY PRELUDES (each ends at the identical final compute-bound config):
 *   up_from_idle   : aggressor cores idle (no threads spawned) for the settle
 *                    window, then start the final gated compute-bound config.
 *   down_from_high : aggressor cores at SUSTAINED high compute-bound load (no
 *                    gating, full-duty alu_burst) for the settle window, then
 *                    come DOWN to the same final gated compute-bound config.
 * The settle window is consumed DURING the prelude state; after the prelude ends
 * at the final config we additionally wait settle_s before firing the witness, so
 * a thermal/decap transient decays with settle_s while a retained state persists.
 *
 * CONTROLS (all mandatory, all built in):
 *   - matched final config: both histories converge to the identical gated alu
 *     drive (same cores, same f_drive, same amplitude, same t0 geometry).
 *   - k10temp recorded per measurement; the history-difference is reported against
 *     the measured thermal slope so a thermal explanation is excluded or flagged.
 *   - scrambled history labels: up/down labels are permuted (deterministic, seeded)
 *     and re-analyzed; a genuine effect must vanish under the scramble.
 *   - >= 20 repeats per (history, settle_s) (REPEATS_MIN, default 24).
 *
 * VERDICT:
 *   BASIN_HISTORY_DEPENDENT iff the up-vs-down witness difference is reproducible,
 *     survives label-scramble, exceeds what the measured k10temp difference
 *     explains, AND persists (nonzero plateau) as settle_s -> large.
 *   NO_RETAINED_BASIN iff the difference is thermal-only or decays to zero with
 *     settle_s (pure instantaneous IR drop).
 *   AMBIGUOUS otherwise.
 *
 * OUTPUT: one CSV row per repeat with columns matching analyze_phase5_10.py
 * (selector = history label; boundary_thickness = basin metric = witness magnitude;
 * basin_class; restoration_failures = 0 (no tape here); k10temp = thermal covariate;
 * physical_state = the pinned P-state / settle bucket), PLUS a settle-time curve
 * summary block and the VERDICT line.
 * =========================================================================== */
typedef enum { HIST_UP = 0, HIST_DOWN = 1 } hist_t;
static const char *hist_name(hist_t h) { return h == HIST_UP ? "up_from_idle" : "down_from_high"; }

/* Sustained full-duty compute-bound load (NO gating) for the down_from_high
   prelude. Reuses alu_burst so the down-history high state is the SAME register/
   L1-only power virus the witness drives at 50% duty -- only the duty differs
   (100% during prelude, gated final config during the witness). */
typedef struct {
    atomic_int  stop;
    pthread_t   thread;
    int         core;
    int         started;
} sustained_t;

static void *sustained_loop(void *arg) {
    sustained_t *s = (sustained_t *)arg;
    pin_to_core(s->core);
    uint64_t iseed = (uint64_t)(s->core * 2246822519u + 3266489917u);
    volatile double sink = 0.0;
    while (!atomic_load(&s->stop)) {
        sink += alu_burst(&iseed);
    }
    (void)sink;
    return NULL;
}

static int sustained_start(sustained_t *s, int core) {
    memset(s, 0, sizeof(*s));
    s->core = core;
    atomic_init(&s->stop, 0);
    pthread_attr_t attr; pthread_attr_init(&attr);
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
    if (pthread_create(&s->thread, &attr, sustained_loop, s) != 0) {
        fprintf(stderr, "WARN: sustained pthread_create core %d: %s\n", core, strerror(errno));
        pthread_attr_destroy(&attr);
        return -1;
    }
    pthread_attr_destroy(&attr);
    s->started = 1;
    return 0;
}

static void sustained_stop(sustained_t *s) {
    if (!s->started) return;
    atomic_store(&s->stop, 1);
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_sec += 2;
    void *rv;
    if (pthread_timedjoin_np(s->thread, &rv, &ts) != 0)
        fprintf(stderr, "WARN: sustained join timeout core %d\n", s->core);
    s->started = 0;
}

/* sleep helper: settle window in seconds (split into 100 ms slices so a temp
   veto mid-window can break out cleanly). Returns 0 normal, 1 if veto tripped. */
static int settle_sleep(double secs, double veto_c) {
    long ns_total = (long)(secs * 1e9);
    long slice = 100*1000*1000;
    while (ns_total > 0) {
        long this_ns = ns_total < slice ? ns_total : slice;
        struct timespec ss = {0, this_ns}; nanosleep(&ss, NULL);
        ns_total -= this_ns;
        if (read_k10temp_c() >= veto_c) return 1;
    }
    return 0;
}

/* ONE basin witness measurement at the identical final config. Runs the prelude
   (history), holds the additional settle_s at the final config, then fires the
   driven compute-only ring-osc lock-in at (f_basin, namp) and returns the
   above-thermal-pole lock-in magnitude + off-bin floor + k10temp.

   SHAM-HISTORY PLACEBO: when cfg->basin_sham is set, the PRELUDE is forced to the
   up_from_idle path for BOTH labels (the `hist` arg still travels through to the
   analysis label). Everything downstream -- the post-prelude settle, the final
   gated compute-bound drive, the witness lock-in -- is byte-for-byte identical to
   the real run, so a sham run is directly comparable: a clean apparatus must then
   show NO plateau and a clean scramble (the labels are meaningless); a surviving
   plateau is a measurement/drift artifact, not a load-history effect.

   Returns 0 on success, 1 on temp veto, <0 on setup failure. */
static int run_basin_witness(const config_t *cfg, hist_t hist, double settle_s,
                             int victim, const int *aggr_list, int naggr, int namp,
                             const char *aggr_label,
                             double *out_mag, double *out_floor,
                             double *out_T, long *out_khz, int *out_ps) {
    double f_basin = cfg->basin_freq;
    double veto = cfg->temp_veto;

    if (read_k10temp_c() >= veto) return 1;

    /* SHAM placebo: both arms run the up_from_idle prelude (no sustained load),
       even though `hist` still carries the up/down LABEL for the analysis. */
    hist_t eff_hist = cfg->basin_sham ? HIST_UP : hist;

    /* ---- PRELUDE: establish the load history during the settle window ---- */
    sustained_t high[NCPU_MAX]; int nhigh = 0;
    if (eff_hist == HIST_DOWN) {
        /* down_from_high: sustained full-duty compute load on the aggressor cores
           for the settle window, then bring DOWN to the gated final config. */
        for (int k = 0; k < namp; k++)
            if (sustained_start(&high[nhigh], aggr_list[k]) == 0) nhigh++;
    }
    /* up_from_idle: aggressor cores idle (no threads) during the settle window. */
    int veto_hit = settle_sleep(settle_s, veto);
    for (int k = 0; k < nhigh; k++) sustained_stop(&high[k]);
    if (veto_hit) return 1;

    /* ---- additional post-prelude settle at the final (idle-ish) config ----
       a thermal/decap transient decays across this window; a retained substrate
       state does not. We wait the SAME settle_s again here so the decisive curve
       is "witness difference vs settle_s". */
    if (settle_sleep(settle_s, veto)) return 1;

    /* ---- WITNESS: identical driven compute-only ring-osc lock-in ---- */
    if (read_k10temp_c() >= veto) return 1;

    double cap_s = (double)cfg->cycles / f_basin;
    if (cap_s < cfg->min_cap_s) cap_s = cfg->min_cap_s;
    if (cap_s > cfg->max_cap_s) cap_s = cfg->max_cap_s;
    int nsamp = (int)(cap_s * cfg->read_hz);
    if (nsamp < 64) nsamp = 64;
    if (nsamp > 2000000) nsamp = 2000000;

    double drive_period_ticks = cfg->tsc_hz / f_basin;
    double half_ticks = 0.5 * drive_period_ticks;

    uint64_t *t_tsc = (uint64_t *)malloc(sizeof(uint64_t) * nsamp);
    double   *ro    = (double *)malloc(sizeof(double) * nsamp);
    double   *feff  = (double *)malloc(sizeof(double) * nsamp);
    if (!t_tsc || !ro || !feff) { free(t_tsc); free(ro); free(feff); return -1; }

    int msr_fd = msr_open(victim);
    uint64_t t0 = rdtsc_now() + (uint64_t)(0.02 * cfg->tsc_hz);

    /* final config: gated 50%-duty compute-bound (AGG_ALU) drive -- byte-identical
       between the two histories. */
    aggressor_t agg[NCPU_MAX]; int started = 0;
    for (int k = 0; k < namp; k++)
        if (aggressor_start(&agg[k], aggr_list[k], t0, half_ticks, 0, AGG_ALU) == 0) started++;
    (void)started;

    atomic_int go; atomic_init(&go, 0);
    reader_arg_t ra; memset(&ra, 0, sizeof(ra));
    ra.core = victim; ra.n = nsamp; ra.read_hz = cfg->read_hz; ra.tsc_hz = cfg->tsc_hz;
    ra.msr_fd = msr_fd; ra.t_tsc = t_tsc; ra.ro_period = ro; ra.feff_ratio = feff;
    ra.go = &go; ra.ready = 0; ra.msr_ok = 0;

    pthread_t rt; pthread_attr_t rattr; pthread_attr_init(&rattr);
    cpu_set_t rset; CPU_ZERO(&rset); CPU_SET(victim, &rset);
    pthread_attr_setaffinity_np(&rattr, sizeof(rset), &rset);
    if (pthread_create(&rt, &rattr, reader_thread, &ra) != 0) {
        pthread_attr_destroy(&rattr);
        for (int k = 0; k < namp; k++) aggressor_stop(&agg[k]);
        if (msr_fd >= 0) close(msr_fd);
        free(t_tsc); free(ro); free(feff);
        return -1;
    }
    pthread_attr_destroy(&rattr);

    while (!ra.ready) { __asm__ volatile("pause"); }
    while (rdtsc_now() < t0) { __asm__ volatile("pause"); }
    atomic_store(&go, 1);
    pthread_join(rt, NULL);

    for (int k = 0; k < namp; k++) aggressor_stop(&agg[k]);

    int ps_wit = cofvid_pstate(victim);
    long khz_wit = read_cur_khz(victim);
    double T_after = read_k10temp_c();

    double f_off = offbin_freq(f_basin);
    double I, Q, mag, Ifl, Qfl, magfl;
    lockin(t_tsc, ro, nsamp, f_basin, t0, cfg->tsc_hz, 0.0, &I, &Q, &mag);
    lockin(t_tsc, ro, nsamp, f_off,   t0, cfg->tsc_hz, 0.0, &Ifl, &Qfl, &magfl);

    if (out_mag)   *out_mag = mag;
    if (out_floor) *out_floor = magfl;
    if (out_T)     *out_T = T_after;
    if (out_khz)   *out_khz = khz_wit;
    if (out_ps)    *out_ps = ps_wit;

    if (msr_fd >= 0) close(msr_fd);
    free(t_tsc); free(ro); free(feff);
    (void)aggr_label; (void)naggr;
    return 0;
}

/* basin-scan CSV: schema consumed by analyze_phase5_10.py.
   selector = history label; boundary_thickness = basin metric (witness magnitude);
   basin_class from frozen-style cuts (computed live for the audit column, the
   analyzer re-derives from frozen thresholds); restoration_failures = 0 (no tape);
   k10temp = thermal covariate; physical_state = pinned-pstate/settle bucket. */
static void basin_emit_header(FILE *f) {
    fprintf(f,
        "run_id,selector,settle_s,scrambled,repeat,"
        "boundary_thickness,witness_floor,snr,basin_class,"
        "restoration_failures,k10temp,physical_state,"
        "victim_core,aggr_cores,f_basin,amplitude_level,cofvid_pstate,cur_khz\n");
    fflush(f);
}

/* live basin class from two cut points on the witness magnitude. This is an
   AUDIT column only: analyze_phase5_10.py re-derives basin_class from frozen
   thresholds (collapsed/mid/high). Cuts are passed in (terciles of this run's
   own metric distribution, computed after all measurements). */
static const char *basin_class_of(double v, double lo, double hi) {
    if (v <= lo) return "collapsed";
    if (v >= hi) return "high";
    return "mid";
}

typedef struct {
    hist_t  hist;
    double  settle_s;
    int     repeat;
    double  mag;
    double  floor;
    double  T;
    long    khz;
    int     ps;
    int     scrambled_label;   /* the permuted label for the scramble control */
} basin_obs_t;

/* mean/stddev helper over a selected subset of observations */
static void mean_sd(const double *x, int n, double *mean, double *sd) {
    if (n <= 0) { *mean = 0; *sd = 0; return; }
    double m = 0; for (int i = 0; i < n; i++) m += x[i]; m /= n;
    double v = 0; for (int i = 0; i < n; i++) { double d = x[i]-m; v += d*d; }
    *sd = (n > 1) ? sqrt(v/(n-1)) : 0.0;
    *mean = m;
}

static int run_basin_scan(config_t *cfg) {
    int victim = cfg->victim;
    int naggr  = cfg->naggr;
    int namp   = cfg->basin_namp;
    if (namp > naggr) namp = naggr;
    if (namp < 1) namp = 1;

    char aggr_label[24]; aggr_label[0]=0;
    { int al = 0; for (int k=0;k<naggr;k++)
        al += snprintf(aggr_label+al, sizeof(aggr_label)-al, "%s%d", k?"-":"", cfg->aggr[k]); }

    /* settle-time sweep (default {0.05,0.2,1.0,4.0}) and repeats (default 24) */
    int nsettle = cfg->basin_nsettle;
    int reps    = cfg->basin_repeats;

    /* open the basin CSV (separate file from the sweep CSV) */
    FILE *bf = g_csv;
    if (cfg->output_dir[0]) {
        char path[320];
        snprintf(path, sizeof(path), "%s/phase5_10_basin_scan.csv", cfg->output_dir);
        bf = fopen(path, "w");
        if (!bf) { fprintf(stderr, "WARN: cannot open %s; using stdout\n", path); bf = stdout; }
        else fprintf(stderr, "BASIN CSV -> %s\n", path);
    } else bf = stdout;
    basin_emit_header(bf);

    /* storage for all observations: 2 histories x nsettle x reps */
    int max_obs = 2 * nsettle * reps;
    basin_obs_t *obs = (basin_obs_t *)calloc(max_obs, sizeof(basin_obs_t));
    if (!obs) { if (bf!=stdout) fclose(bf); return -1; }
    int nobs = 0;

    /* deterministic scramble permutation of the (history) labels: we permute the
       up/down label assigned to each measurement with a fixed seed, so the
       "scrambled" analysis pairs a random half as up and half as down. A genuine
       history effect must VANISH under this permutation. */
    uint64_t scr = 0x9E3779B97F4A7C15ULL ^ (uint64_t)cfg->basin_seed;

    /* ---- CONTROL 1: FULL MEASUREMENT-ORDER RANDOMIZATION ----
       Build the COMPLETE list of (history, settle, rep) measurement tasks and
       shuffle it into a single fully-randomized order (deterministic from
       --basin-seed) before executing ANY of them. The old code ran the two
       histories back-to-back inside each (settle, rep) cell, so any slow drift
       (temperature creep over the run) stayed correlated with the up/down labels;
       the previous run-order "balancing" (alternate-first) did not break that
       coupling. Shuffling the whole task list DECORRELATES drift from history:
       if the up-vs-down effect was drift, full randomization makes the
       label-scramble control clean (scrambled diff -> ~0) and shrinks/zeros the
       history plateau. */
    typedef struct { int si; hist_t hist; int rep; } basin_task_t;
    int ntask = 2 * nsettle * reps;
    basin_task_t *tasks = (basin_task_t *)calloc(ntask>0?ntask:1, sizeof(basin_task_t));
    if (!tasks) { free(obs); if (bf!=stdout && bf!=g_csv) fclose(bf); return -1; }
    {
        int t = 0;
        for (int si = 0; si < nsettle; si++)
            for (int rep = 0; rep < reps; rep++) {
                tasks[t].si = si; tasks[t].hist = HIST_UP;   tasks[t].rep = rep; t++;
                tasks[t].si = si; tasks[t].hist = HIST_DOWN; tasks[t].rep = rep; t++;
            }
    }
    /* Fisher-Yates with an INDEPENDENT seeded PRNG (so the scramble-label stream
       `scr` stays untouched / order-independent). Deterministic from --basin-seed. */
    {
        uint64_t ord = 0xD1B54A32D192ED03ULL ^ ((uint64_t)cfg->basin_seed << 1);
        for (int i = ntask - 1; i > 0; i--) {
            ord = ord * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = (int)((ord >> 33) % (uint64_t)(i + 1));
            basin_task_t tmp = tasks[i]; tasks[i] = tasks[j]; tasks[j] = tmp;
        }
    }

    fprintf(stderr, "=== BASIN SCAN%s: f_basin=%.1fHz amp=%d victim=%d aggr=%s "
                    "settle={", cfg->basin_sham ? " [SHAM-HISTORY PLACEBO]" : "",
                    cfg->basin_freq, namp, victim, aggr_label);
    for (int s=0;s<nsettle;s++) fprintf(stderr, "%.3g%s", cfg->basin_settle[s], s+1<nsettle?",":"");
    fprintf(stderr, "} reps=%d order=fully-randomized(seed=%d)%s ===\n",
            reps, cfg->basin_seed,
            cfg->basin_sham ? " sham=ON(both arms run up_from_idle prelude)" : "");

    /* within-run thermal drift witness: k10temp at the first and last executed
       measurement, so the drift magnitude that the controls must defeat is visible. */
    double T_scan_start = -999.0, T_scan_end = -999.0;

    for (int ti = 0; ti < ntask; ti++) {
        int si = tasks[ti].si;
        hist_t hist = tasks[ti].hist;
        int rep = tasks[ti].rep;
        double settle_s = cfg->basin_settle[si];

        double mag=0, floor=0, T=0; long khz=0; int ps=0;
        int rc = run_basin_witness(cfg, hist, settle_s, victim,
                                   cfg->aggr, naggr, namp, aggr_label,
                                   &mag, &floor, &T, &khz, &ps);
        if (rc == 1) {
            fprintf(stderr, "TEMP VETO during basin scan (settle=%.3g rep=%d hist=%s); "
                            "stopping scan gracefully\n", settle_s, rep, hist_name(hist));
            goto basin_done;
        }
        if (rc != 0) { fprintf(stderr, "WARN: basin witness setup fail; skip\n"); continue; }

        if (T_scan_start <= -900.0) T_scan_start = T;
        T_scan_end = T;

        /* scramble label for this observation (deterministic; order-independent of
           the shuffled execution because `scr` advances once per accepted obs). */
        scr = scr * 6364136223846793005ULL + 1442695040888963407ULL;
        int scrambled_label = (int)((scr >> 33) & 1);

        basin_obs_t *o = &obs[nobs++];
        o->hist = hist; o->settle_s = settle_s; o->repeat = rep;
        o->mag = mag; o->floor = floor; o->T = T; o->khz = khz; o->ps = ps;
        o->scrambled_label = scrambled_label;

        fprintf(stderr, "  [%4d/%4d settle=%.3g rep=%2d hist=%-14s] mag=%.5g fl=%.5g "
                        "snr=%.2f T=%.1fC ps=%d khz=%ld\n",
                ti+1, ntask, settle_s, rep, hist_name(hist), mag, floor,
                (floor>0?mag/floor:0), T, ps, khz);
    }

basin_done:
    free(tasks);
    /* ---- emit per-observation rows (real labels) ----
       basin_class cuts: terciles of the whole-scan witness magnitude (audit only;
       analyzer re-derives from frozen thresholds). */
    {
        double *allmag = (double *)malloc(sizeof(double) * (nobs>0?nobs:1));
        for (int i=0;i<nobs;i++) allmag[i] = obs[i].mag;
        /* simple insertion sort for terciles (nobs is small) */
        for (int i=1;i<nobs;i++){ double k=allmag[i]; int j=i-1;
            while (j>=0 && allmag[j]>k){ allmag[j+1]=allmag[j]; j--; } allmag[j+1]=k; }
        double cut_lo = nobs>0 ? allmag[nobs/3] : 0.0;
        double cut_hi = nobs>0 ? allmag[(2*nobs)/3] : 0.0;
        free(allmag);

        for (int i=0;i<nobs;i++) {
            basin_obs_t *o = &obs[i];
            char run_id[48];
            snprintf(run_id, sizeof(run_id), "%s_s%.3g_r%d", hist_name(o->hist), o->settle_s, o->repeat);
            char phys[32];
            snprintf(phys, sizeof(phys), "ps%d_settle%.3g", o->ps, o->settle_s);
            /* real-label row */
            fprintf(bf,
                "%s,%s,%.4g,%d,%d,%.6g,%.6g,%.4f,%s,%d,%.2f,%s,%d,%s,%.4f,%d,%d,%ld\n",
                run_id, hist_name(o->hist), o->settle_s, 0, o->repeat,
                o->mag, o->floor, (o->floor>0?o->mag/o->floor:0.0),
                basin_class_of(o->mag, cut_lo, cut_hi), 0, o->T, phys,
                victim, aggr_label, cfg->basin_freq, namp, o->ps, o->khz);
            /* scrambled-label row (control_flag via scrambled=1; selector permuted) */
            const char *scr_sel = o->scrambled_label ? "down_from_high" : "up_from_idle";
            char run_id_s[64];
            snprintf(run_id_s, sizeof(run_id_s), "scr_%s_s%.3g_r%d", scr_sel, o->settle_s, o->repeat);
            fprintf(bf,
                "%s,%s,%.4g,%d,%d,%.6g,%.6g,%.4f,%s,%d,%.2f,%s,%d,%s,%.4f,%d,%d,%ld\n",
                run_id_s, scr_sel, o->settle_s, 1, o->repeat,
                o->mag, o->floor, (o->floor>0?o->mag/o->floor:0.0),
                basin_class_of(o->mag, cut_lo, cut_hi), 0, o->T, phys,
                victim, aggr_label, cfg->basin_freq, namp, o->ps, o->khz);
        }
        fflush(bf);
    }

    /* ---- settle-time curve + verdict analysis ----
       For each settle_s: mean up vs mean down witness; the history difference
       diff(settle) = mean_down - mean_up; the matched k10temp difference
       dT(settle) = mean_T_down - mean_T_up; and the scrambled difference
       diff_scr(settle) = mean(scr_down) - mean(scr_up) (must be ~0). */
    fprintf(bf, "# SETTLE_CURVE,settle_s,n_up,n_down,mean_up,sd_up,mean_down,sd_down,"
                "diff,diff_scr,dT,mean_T,thermal_explained_diff,residual_diff\n");

    /* accumulate a thermal slope estimate: regress witness mag on k10temp across
       ALL observations, so we can subtract the thermally-explained part of the
       up/down difference. */
    double sumT=0,sumM=0,sumTT=0,sumTM=0; int ncnt=0;
    for (int i=0;i<nobs;i++){ sumT+=obs[i].T; sumM+=obs[i].mag; sumTT+=obs[i].T*obs[i].T;
        sumTM+=obs[i].T*obs[i].mag; ncnt++; }
    double thermal_slope = 0.0;
    if (ncnt>=3){ double den = ncnt*sumTT - sumT*sumT;
        if (fabs(den) > 1e-12) thermal_slope = (ncnt*sumTM - sumT*sumM)/den; }

    /* per-settle aggregation */
    double diff_by_settle[16]; double resid_by_settle[16];
    double settle_vals[16]; int ns_used = 0;
    double max_abs_scr = 0.0;
    for (int si = 0; si < nsettle && si < 16; si++) {
        double settle_s = cfg->basin_settle[si];
        double up[512], down[512]; int nu=0, nd=0;
        double upT[512], downT[512];
        double scrup[1024], scrdown[1024]; int nsu=0, nsd=0;
        for (int i=0;i<nobs;i++) {
            if (fabs(obs[i].settle_s - settle_s) > 1e-9) continue;
            if (obs[i].hist == HIST_UP) { if(nu<512){up[nu]=obs[i].mag; upT[nu]=obs[i].T; nu++;} }
            else                        { if(nd<512){down[nd]=obs[i].mag; downT[nd]=obs[i].T; nd++;} }
            /* scrambled assignment */
            if (obs[i].scrambled_label==0){ if(nsu<1024) scrup[nsu++]=obs[i].mag; }
            else                          { if(nsd<1024) scrdown[nsd++]=obs[i].mag; }
        }
        double mu,su,md,sd,mtu,mtd,mscru,mscrd,dummy;
        mean_sd(up,nu,&mu,&su); mean_sd(down,nd,&md,&sd);
        mean_sd(upT,nu,&mtu,&dummy); mean_sd(downT,nd,&mtd,&dummy);
        mean_sd(scrup,nsu,&mscru,&dummy); mean_sd(scrdown,nsd,&mscrd,&dummy);
        double diff = md - mu;
        double diff_scr = mscrd - mscru;
        double dT = mtd - mtu;
        double mean_T = 0.5*(mtu+mtd);
        double thermal_explained = thermal_slope * dT;
        double residual = diff - thermal_explained;
        fprintf(bf, "# SETTLE_CURVE,%.4g,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.4f,%.2f,%.6g,%.6g\n",
                settle_s, nu, nd, mu, su, md, sd, diff, diff_scr, dT, mean_T,
                thermal_explained, residual);
        if (nu>0 && nd>0) {
            diff_by_settle[ns_used] = diff;
            resid_by_settle[ns_used] = residual;
            settle_vals[ns_used] = settle_s;
            if (fabs(diff_scr) > max_abs_scr) max_abs_scr = fabs(diff_scr);
            ns_used++;
        }
    }

    /* plateau test: fit residual_diff(settle) = plateau + C/settle (decay model)
       via OLS of residual on (1/settle). The intercept = the settle->inf plateau.
       A retained state -> plateau != 0; pure transient -> residual decays to ~0
       (plateau ~ 0, large positive C at small settle). */
    double plateau = 0.0, slopeC = 0.0, plat_se = INFINITY;
    if (ns_used >= 2) {
        double xb=0,yb=0; for(int i=0;i<ns_used;i++){ xb+=1.0/settle_vals[i]; yb+=resid_by_settle[i]; }
        xb/=ns_used; yb/=ns_used;
        double sxx=0,sxy=0; for(int i=0;i<ns_used;i++){ double dx=1.0/settle_vals[i]-xb;
            sxx+=dx*dx; sxy+=dx*(resid_by_settle[i]-yb); }
        if (fabs(sxx)>1e-18){ slopeC=sxy/sxx; plateau=yb - slopeC*xb;
            double ssr=0; for(int i=0;i<ns_used;i++){ double r=resid_by_settle[i]-(plateau+slopeC*(1.0/settle_vals[i])); ssr+=r*r; }
            if (ns_used>2){ double s2=ssr/(ns_used-2); plat_se=sqrt(s2*(1.0/ns_used + xb*xb/sxx)); }
        } else { plateau = yb; }
    }

    /* reproducibility: largest-settle residual difference vs its scale.
       scramble: max |diff_scr| must be small vs the real residual.
       thermal: residual (thermal-subtracted) must remain the carrier of the diff. */
    double largest_settle_resid = ns_used>0 ? resid_by_settle[ns_used-1] : 0.0;
    double largest_settle_diff  = ns_used>0 ? diff_by_settle[ns_used-1] : 0.0;

    /* within-run thermal drift (start vs end k10temp of the executed scan). This is
       the drift the controls must defeat: the AMBIGUOUS prior had a ~2.4 C run-long
       creep that, at thermal_slope mag/C, can fake a history difference. Reported
       explicitly so the drift magnitude is visible alongside the verdict. */
    double thermal_drift_c   = (T_scan_start > -900.0 && T_scan_end > -900.0)
                                 ? (T_scan_end - T_scan_start) : 0.0;
    double drift_explained   = thermal_slope * thermal_drift_c;  /* mag the drift could fake */

    /* decision thresholds (documented, conservative) */
    int   have_curve      = (ns_used >= 2);
    int   scramble_clean  = have_curve && (max_abs_scr < 0.5 * fabs(largest_settle_diff) + 1e-30);
    int   plateau_nonzero = have_curve && plat_se != INFINITY && fabs(plateau) > 2.0*plat_se;
    int   nonthermal      = have_curve && (fabs(largest_settle_resid) > 0.5*fabs(largest_settle_diff));
    int   decays_to_zero  = have_curve && (plat_se != INFINITY) && (fabs(plateau) <= 2.0*plat_se);
    int   is_sham         = cfg->basin_sham;

    /* ---- VERDICT (sham-aware + full-randomization-aware) ----
       The basin scan now runs in a SINGLE fully-randomized measurement order, so
       drift is decorrelated from the up/down labels: a scramble that stays DIRTY
       under full randomization can no longer be drift-vs-label, it is a real
       label-correlated artifact in the apparatus. Two run roles:

       REAL run (sham OFF): emits BASIN_REAL_*.
         - scramble stays dirty under full randomization  -> ARTIFACT_CONFIRMED.
         - nonzero, non-thermal plateau AND clean scramble -> BASIN_REAL_PLATEAU
           (a genuine retained basin IFF the companion SHAM run shows no plateau).
         - plateau decays to zero / thermal-only           -> NO_RETAINED_BASIN.
         - otherwise                                       -> AMBIGUOUS.

       SHAM run (sham ON, both arms = up_from_idle, labels meaningless): emits
       BASIN_SHAM_*. A clean apparatus MUST show no plateau and a clean scramble.
         - nonzero plateau (or dirty scramble) -> ARTIFACT_CONFIRMED (decisive
           falsification: the ~0.027 effect is measurement/drift, not history).
         - no plateau AND clean scramble       -> SHAM_CLEAN (apparatus passes the
           placebo; the REAL plateau, if any, is then admissible as history).

       Cross-run combination (done by the orchestrator from the two CSVs):
         BASIN_HISTORY_DEPENDENT  iff REAL=BASIN_REAL_PLATEAU AND SHAM=SHAM_CLEAN.
         ARTIFACT_CONFIRMED       iff EITHER run = ARTIFACT_CONFIRMED.
         AMBIGUOUS                otherwise.
       The combined line BASIN_COMBINED_VERDICT below states this rule for the log. */
    const char *verdict;
    if (!have_curve) {
        verdict = "AMBIGUOUS_INSUFFICIENT_SETTLE_COVERAGE";
    } else if (is_sham) {
        if (plateau_nonzero || !scramble_clean)
            verdict = "ARTIFACT_CONFIRMED";          /* sham plateau / dirty scramble */
        else
            verdict = "SHAM_CLEAN";                  /* placebo passes: no plateau, clean scramble */
    } else {
        if (!scramble_clean)
            verdict = "ARTIFACT_CONFIRMED";          /* scramble dirty under full randomization */
        else if (plateau_nonzero && nonthermal)
            verdict = "BASIN_REAL_PLATEAU";          /* candidate basin; needs SHAM_CLEAN to confirm */
        else if (decays_to_zero || !nonthermal)
            verdict = "NO_RETAINED_BASIN";
        else
            verdict = "AMBIGUOUS";
    }

    fprintf(bf,
        "# BASIN_VERDICT_DETAIL,sham=%d,order=fully_randomized,seed=%d,"
        "n_obs=%d,settle_points_used=%d,thermal_slope=%.6g,"
        "plateau=%.6g,plateau_se=%.6g,slope_C=%.6g,largest_settle_diff=%.6g,"
        "largest_settle_residual=%.6g,max_abs_scramble=%.6g,"
        "scramble_clean=%d,plateau_nonzero=%d,nonthermal=%d,decays_to_zero=%d,"
        "T_scan_start=%.2f,T_scan_end=%.2f,thermal_drift_c=%.3f,drift_explained_mag=%.6g\n",
        is_sham, cfg->basin_seed, nobs, ns_used, thermal_slope, plateau, plat_se, slopeC,
        largest_settle_diff, largest_settle_resid, max_abs_scr,
        scramble_clean, plateau_nonzero, nonthermal, decays_to_zero,
        T_scan_start, T_scan_end, thermal_drift_c, drift_explained);
    fprintf(bf, "BASIN_VERDICT,%s\n", verdict);
    /* cross-run combination rule, restated for the log so either CSV is self-describing */
    fprintf(bf, "# BASIN_COMBINED_VERDICT,rule="
                "HISTORY_DEPENDENT iff REAL=BASIN_REAL_PLATEAU and SHAM=SHAM_CLEAN; "
                "ARTIFACT_CONFIRMED iff either run=ARTIFACT_CONFIRMED; else AMBIGUOUS,"
                "this_run=%s,this_verdict=%s\n",
                is_sham ? "SHAM" : "REAL", verdict);
    fflush(bf);

    fprintf(stderr, "\n=============== BASIN VERDICT ===============\n");
    fprintf(stderr, "run role                   : %s%s\n",
            is_sham ? "SHAM-HISTORY PLACEBO" : "REAL",
            is_sham ? " (both arms = up_from_idle; labels meaningless)" : "");
    fprintf(stderr, "measurement order          : FULLY RANDOMIZED (seed=%d)\n", cfg->basin_seed);
    fprintf(stderr, "observations               : %d (2 histories x %d settle x %d reps)\n",
            nobs, nsettle, reps);
    fprintf(stderr, "within-run thermal drift   : start=%.2fC end=%.2fC drift=%+.2fC\n",
            T_scan_start, T_scan_end, thermal_drift_c);
    fprintf(stderr, "thermal slope (mag/C)      : %.6g  (drift could fake mag=%.6g)\n",
            thermal_slope, drift_explained);
    fprintf(stderr, "settle-curve points used   : %d\n", ns_used);
    fprintf(stderr, "plateau (settle->inf resid): %.6g +/- %.6g\n", plateau, plat_se);
    fprintf(stderr, "largest-settle diff/resid  : %.6g / %.6g\n",
            largest_settle_diff, largest_settle_resid);
    fprintf(stderr, "max |scrambled diff|       : %.6g (must be small; clean=%d)\n",
            max_abs_scr, scramble_clean);
    fprintf(stderr, "plateau_nonzero=%d nonthermal=%d decays_to_zero=%d\n",
            plateau_nonzero, nonthermal, decays_to_zero);
    fprintf(stderr, "BASIN_VERDICT: %s\n",
        !strcmp(verdict,"BASIN_REAL_PLATEAU")
            ? "BASIN_REAL_PLATEAU (candidate retained basin; confirm with a SHAM_CLEAN companion run)"
        : !strcmp(verdict,"SHAM_CLEAN")
            ? "SHAM_CLEAN (placebo passed: no plateau, clean scramble -> apparatus admits a real REAL plateau)"
        : !strcmp(verdict,"ARTIFACT_CONFIRMED")
            ? (is_sham ? "ARTIFACT_CONFIRMED (sham produced a plateau/dirty scramble -> the effect is drift/measurement, NOT history)"
                       : "ARTIFACT_CONFIRMED (scramble stayed dirty under full randomization -> label-correlated artifact)")
        : !strcmp(verdict,"NO_RETAINED_BASIN")
            ? "NO_RETAINED_BASIN (thermal-only or decays with settle -> instantaneous IR drop)"
            : verdict);
    fprintf(stderr, "COMBINED RULE: HISTORY_DEPENDENT iff REAL=BASIN_REAL_PLATEAU AND SHAM=SHAM_CLEAN; "
                    "ARTIFACT_CONFIRMED iff either run=ARTIFACT_CONFIRMED; else AMBIGUOUS\n");
    fprintf(stderr, "=============================================\n");

    free(obs);
    if (bf && bf != stdout && bf != g_csv) fclose(bf);
    return 0;
}

/* ===========================================================================
 * main
 * =========================================================================== */
static void parse_int_list(const char *s, int *out, int *n, int maxn) {
    *n = 0;
    char buf[256]; strncpy(buf, s, sizeof(buf)-1); buf[sizeof(buf)-1]=0;
    char *tok = strtok(buf, ",");
    while (tok && *n < maxn) { out[(*n)++] = atoi(tok); tok = strtok(NULL, ","); }
}
static void parse_dbl_list(const char *s, double *out, int *n, int maxn) {
    *n = 0;
    char buf[256]; strncpy(buf, s, sizeof(buf)-1); buf[sizeof(buf)-1]=0;
    char *tok = strtok(buf, ",");
    while (tok && *n < maxn) { out[(*n)++] = atof(tok); tok = strtok(NULL, ","); }
}

int main(int argc, char **argv) {
    config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.victim = 2;
    cfg.aggr[0]=3; cfg.aggr[1]=4; cfg.aggr[2]=5; cfg.naggr=3;
    parse_dbl_list("0.1,0.3,1,3,10,30,100,300,1000", cfg.freqs, &cfg.nfreq, 32);
    cfg.amp_ladder[0]=1; cfg.amp_ladder[1]=3; cfg.amp_ladder[2]=5; cfg.namp=3;
    cfg.pin_khz = 1600000;     /* P1 = 1600 MHz: clean 0.5 reference, headroom for droop */
    cfg.read_hz = 4000;        /* a few kHz: Nyquist ~2 kHz, covers the sweep to 1 kHz */
    cfg.cycles = 200;
    cfg.min_cap_s = 0.5;
    cfg.max_cap_s = 8.0;
    cfg.temp_veto = 68.0;
    cfg.warm_target = 60.0;
    cfg.do_warm = 0;
    cfg.swap_topology = 0;
    cfg.do_pin = 1;
    cfg.aggr_sel = AGGSEL_BOTH; /* head-to-head alu vs cache contention control */
    cfg.tsc_hz = 3.2e9;        /* P0 reference; MPERF counts at this rate */
    strcpy(cfg.output_dir, "");
    /* ---- basin-scan defaults (used only with --mode basin-scan) ---- */
    cfg.mode_basin = 0;        /* default: existing sweep mode */
    cfg.basin_freq = 30.0;     /* fixed witness drive freq: above the thermal pole */
    cfg.basin_namp = 3;        /* fixed witness amplitude (clamped to naggr at run) */
    cfg.basin_settle[0]=0.05; cfg.basin_settle[1]=0.2;
    cfg.basin_settle[2]=1.0;  cfg.basin_settle[3]=4.0; cfg.basin_nsettle=4;
    cfg.basin_repeats = 24;    /* >= 20 repeats per (history, settle_s) */
    cfg.basin_seed = 44;       /* drives BOTH the measurement-order shuffle and the scramble */
    cfg.basin_sham = 0;        /* SHAM-HISTORY PLACEBO off by default (--sham-history) */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--victim") && i+1<argc) cfg.victim=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--aggr") && i+1<argc) parse_int_list(argv[++i], cfg.aggr, &cfg.naggr, NCPU_MAX);
        else if (!strcmp(argv[i],"--freqs") && i+1<argc) parse_dbl_list(argv[++i], cfg.freqs, &cfg.nfreq, 32);
        else if (!strcmp(argv[i],"--amp-ladder") && i+1<argc) parse_int_list(argv[++i], cfg.amp_ladder, &cfg.namp, NCPU_MAX);
        else if (!strcmp(argv[i],"--pin-khz") && i+1<argc) cfg.pin_khz=atol(argv[++i]);
        else if (!strcmp(argv[i],"--read-hz") && i+1<argc) cfg.read_hz=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--cycles") && i+1<argc) cfg.cycles=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--min-cap-s") && i+1<argc) cfg.min_cap_s=atof(argv[++i]);
        else if (!strcmp(argv[i],"--max-cap-s") && i+1<argc) cfg.max_cap_s=atof(argv[++i]);
        else if (!strcmp(argv[i],"--temp-veto") && i+1<argc) cfg.temp_veto=atof(argv[++i]);
        else if (!strcmp(argv[i],"--warm-target") && i+1<argc) cfg.warm_target=atof(argv[++i]);
        else if (!strcmp(argv[i],"--warm")) cfg.do_warm=1;
        else if (!strcmp(argv[i],"--swap-topology")) cfg.swap_topology=1;
        else if (!strcmp(argv[i],"--no-pin")) cfg.do_pin=0;
        else if (!strcmp(argv[i],"--aggr-mode") && i+1<argc) {
            const char *m = argv[++i];
            /* primary names compute|memory (task spec); alu|cache kept as aliases */
            if      (!strcmp(m,"compute") || !strcmp(m,"alu"))   cfg.aggr_sel = AGGSEL_ALU;
            else if (!strcmp(m,"memory")  || !strcmp(m,"cache")) cfg.aggr_sel = AGGSEL_CACHE;
            else if (!strcmp(m,"both"))                          cfg.aggr_sel = AGGSEL_BOTH;
            else fprintf(stderr, "WARN: --aggr-mode '%s' unknown (use compute|memory|both); keeping both\n", m);
        }
        else if (!strcmp(argv[i],"--tsc-hz") && i+1<argc) cfg.tsc_hz=atof(argv[++i]);
        else if (!strcmp(argv[i],"--output-dir") && i+1<argc) strncpy(cfg.output_dir, argv[++i], sizeof(cfg.output_dir)-1);
        else if (!strcmp(argv[i],"--mode") && i+1<argc) {
            const char *m = argv[++i];
            if      (!strcmp(m,"basin-scan") || !strcmp(m,"basin")) cfg.mode_basin = 1;
            else if (!strcmp(m,"sweep"))                            cfg.mode_basin = 0;
            else fprintf(stderr, "WARN: --mode '%s' unknown (use sweep|basin-scan); keeping sweep\n", m);
        }
        else if (!strcmp(argv[i],"--basin-freq") && i+1<argc) cfg.basin_freq=atof(argv[++i]);
        else if (!strcmp(argv[i],"--basin-amp") && i+1<argc) cfg.basin_namp=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--settle-s") && i+1<argc) parse_dbl_list(argv[++i], cfg.basin_settle, &cfg.basin_nsettle, 16);
        else if (!strcmp(argv[i],"--repeats") && i+1<argc) cfg.basin_repeats=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--basin-seed") && i+1<argc) cfg.basin_seed=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--sham-history")) cfg.basin_sham=1;
        else fprintf(stderr, "WARN: unknown arg '%s'\n", argv[i]);
    }

    if (locate_k10temp() != 0) {
        fprintf(stderr, "FATAL: cannot locate k10temp; refusing to run blind.\n");
        return 2;
    }

    if (cfg.output_dir[0]) {
        char path[320];
        snprintf(path, sizeof(path), "%s/phase5_10_driven_lockin.csv", cfg.output_dir);
        g_csv = fopen(path, "w");
        if (!g_csv) { fprintf(stderr, "WARN: cannot open %s; using stdout\n", path); g_csv = stdout; }
        else fprintf(stderr, "CSV -> %s\n", path);
    } else g_csv = stdout;

    double T0 = read_k10temp_c();
    fprintf(stderr, "=== phase5_10 driven two-channel lock-in ===\n");
    fprintf(stderr, "victim=%d aggr=[", cfg.victim);
    for (int k=0;k<cfg.naggr;k++) fprintf(stderr, "%d%s", cfg.aggr[k], k+1<cfg.naggr?",":"");
    fprintf(stderr, "] pin_khz=%ld read_hz=%d cycles=%d veto=%.0fC start_T=%.1fC warm=%d swap=%d\n",
            cfg.pin_khz, cfg.read_hz, cfg.cycles, cfg.temp_veto, T0, cfg.do_warm, cfg.swap_topology);

    if (T0 >= cfg.temp_veto) {
        fprintf(stderr, "ABORT: start temp %.1fC >= veto %.1fC\n", T0, cfg.temp_veto);
        return 3;
    }

    /* sweep-schema header only for sweep mode; basin-scan writes its own header
       to its own CSV. Precheck rows (below) still emit either way as a diagnostic. */
    if (!cfg.mode_basin) emit_header();

    /* ---- STEP 1: PIN P-STATE ---- */
    pinstate_t ps; memset(&ps, 0, sizeof(ps));
    if (cfg.do_pin) {
        if (pin_pstate(&ps, cfg.pin_khz) != 0)
            fprintf(stderr, "WARN: P-state pin had no effect (no cpufreq?); proceeding unpinned\n");
        struct timespec s = {0, 300*1000*1000}; nanosleep(&s, NULL);
        long khz = read_cur_khz(cfg.victim);
        fprintf(stderr, "P-state pinned: target=%ld kHz, victim cur=%ld kHz, cofvid_ps=%d, boost=%d\n",
                cfg.pin_khz, khz, cofvid_pstate(cfg.victim), ps.boost_present?0:-1);
        /* pin sanity: did the governor actually hold? */
        if (khz > 0 && labs(khz - cfg.pin_khz) > 50000) {
            fprintf(stderr, "WARN: pin not held (cur=%ld != target=%ld); governor may hop. "
                            "Any feff movement below is then NOT clean droop.\n", khz, cfg.pin_khz);
        }
    }

    /* ---- STEP 2: PRECHECK ---- */
    double pc_idle=0, pc_load=0;
    precheck(&cfg, cfg.victim, &pc_idle, &pc_load);

    /* ---- MODE: BASIN-SCAN (load-history / hysteresis) ----
       Shares ALL the safety machinery above (P-state pin + boost-off readback,
       k10temp veto, MSR reads only). Runs the up-from-idle vs down-from-high
       settle-time sweep witness, emits the analyze_phase5_10.py-schema rows +
       settle-curve + BASIN_VERDICT, then restores the P-state and exits. */
    if (cfg.mode_basin) {
        run_basin_scan(&cfg);
        if (cfg.do_pin) restore_pstate(&ps);
        double Tend_b = read_k10temp_c();
        fprintf(stderr, "basin-scan done; start_T=%.1fC end_T=%.1fC\n", T0, Tend_b);
        if (g_csv && g_csv != stdout) fclose(g_csv);
        return 0;
    }

    /* topology assignments: primary + optional swap. The swap moves the victim to
       the far end of the package and re-rolls aggressors, testing core-distance
       dependence (a heating artifact is assignment-independent; real PDN coupling
       is not). */
    typedef struct { int victim; int aggr[NCPU_MAX]; int naggr; char label[24]; } topo_t;
    topo_t topos[2];
    int ntopo = 1;
    topos[0].victim = cfg.victim; topos[0].naggr = cfg.naggr;
    for (int k=0;k<cfg.naggr;k++) topos[0].aggr[k]=cfg.aggr[k];
    if (cfg.swap_topology) {
        /* swap: victim <- last aggressor, aggressors <- {old victim + remaining} */
        int newv = cfg.aggr[cfg.naggr-1];
        topos[1].victim = newv; topos[1].naggr = cfg.naggr;
        topos[1].aggr[0] = cfg.victim;
        for (int k=1;k<cfg.naggr;k++) topos[1].aggr[k] = cfg.aggr[k-1];
        ntopo = 2;
    }

    /* verdict accumulators: track above-pole feff lock-in vs amplitude, droop sign,
       scramble/off-bin collapse, and topology dependence. */
    #define POLE_HZ  10.0   /* thermal pole: heating has rolled off above this */
    #define SNR_REAL  3.0   /* above-pole drive bin must beat off-bin floor by 3x */
    double abovepole_feff_by_amp[NCPU_MAX]; int abovepole_seen[NCPU_MAX];
    for (int i=0;i<NCPU_MAX;i++){ abovepole_feff_by_amp[i]=0; abovepole_seen[i]=0; }
    double best_abovepole_snr = 0.0;
    double best_droop_sign = 0.0;
    int    feff_channel_live = 0;
    double topo_feff_primary = 0.0, topo_feff_swapped = -1.0;
    int    any_collapse_ok = 1;     /* off-bin floor stays below the drive bin */
    double belowpole_max_snr = 0.0;

    /* ---- CONTENTION-CONTROL accumulators (alu vs cache, ring-osc timing channel) ----
       The ring-osc timing loop is the ONLY rail witness on this part (APERF/MPERF is
       physically blind to droop here: PLL holds the clock under a pinned P-state, no
       adaptive clock-stretch). So the disambiguation compares the VICTIM RING-OSC
       above-pole lock-in magnitude under the ALU (register/L1-only -> rail-only reach)
       aggressor vs the CACHE (shared-resource -> rail OR contention) aggressor, at
       matched amplitude and drive freq >= POLE_HZ, primary topology / idle band. */
    double ctrl_alu_ro_by_amp[NCPU_MAX];   int ctrl_alu_seen[NCPU_MAX];
    double ctrl_cache_ro_by_amp[NCPU_MAX]; int ctrl_cache_seen[NCPU_MAX];
    for (int i=0;i<NCPU_MAX;i++){
        ctrl_alu_ro_by_amp[i]=0;   ctrl_alu_seen[i]=0;
        ctrl_cache_ro_by_amp[i]=0; ctrl_cache_seen[i]=0;
    }
    double ctrl_alu_best_snr = 0.0, ctrl_cache_best_snr = 0.0;       /* best above-pole ring-osc SNR */
    double ctrl_alu_best_mag = 0.0, ctrl_cache_best_mag = 0.0;       /* mag at that best point */
    int    ctrl_alu_survives_collapse = 0, ctrl_cache_survives_collapse = 0; /* any above-pole bin beats its off-bin floor by SNR_REAL */
    int    alu_ran = 0, cache_ran = 0;

    /* which aggressor TYPEs to run (the contention control) */
    agg_mode_t modes[2]; int nmode = 0;
    if (cfg.aggr_sel == AGGSEL_ALU)       { modes[nmode++] = AGG_ALU; }
    else if (cfg.aggr_sel == AGGSEL_CACHE){ modes[nmode++] = AGG_CACHE; }
    else                                  { modes[nmode++] = AGG_ALU; modes[nmode++] = AGG_CACHE; }

    /* temp bands */
    const char *bands[2] = { "idle", "warm" };
    int nband = cfg.do_warm ? 2 : 1;

    aggressor_t warmers[NCPU_MAX]; int nwarm = 0;  /* background heaters for warm band */

    for (int b = 0; b < nband; b++) {
        const char *band = bands[b];

        if (b == 1) {
            /* warm band: spin sustained background heat on the cores NOT used as the
               primary victim until we reach warm_target, capped by veto. */
            uint64_t t0 = rdtsc_now();
            for (int c = 0; c < NCPU_MAX; c++) {
                if (c == topos[0].victim) continue;
                if (nwarm < NCPU_MAX && aggressor_start(&warmers[nwarm], c, t0, 1.0, 1, AGG_CACHE) == 0) nwarm++;
            }
            fprintf(stderr, "warm band: heating with %d background cores toward %.0fC...\n",
                    nwarm, cfg.warm_target);
            double T;
            for (int it = 0; it < 600; it++) {       /* up to ~60 s */
                struct timespec s = {0, 100*1000*1000}; nanosleep(&s, NULL);
                T = read_k10temp_c();
                if (T >= cfg.warm_target || T >= cfg.temp_veto - 1.0) break;
            }
            fprintf(stderr, "warm band reached T=%.1fC\n", read_k10temp_c());
        }

      for (int mi = 0; mi < nmode; mi++) {
        agg_mode_t mode = modes[mi];
        const char *mname = agg_mode_name(mode);
        if (mode == AGG_ALU)   alu_ran = 1;   else cache_ran = 1;
        fprintf(stderr, "--- aggressor mode = %s (%s) ---\n", mname,
                mode == AGG_ALU ? "compute-bound register/L1-only: shares no resource, rail-only reach"
                                : "memory-bound shared L3/mem/NB hammer: rail OR contention");

        for (int tp = 0; tp < ntopo; tp++) {
            int victim = topos[tp].victim;
            /* build aggressor list (exclude warmers overlap is fine; warmers idle-gate
               at f=1 essentially DC, the gated aggressors dominate dI/dt at f_drive) */
            char aggr_label[24]; aggr_label[0]=0;
            int al = 0;
            for (int k=0;k<topos[tp].naggr;k++) {
                al += snprintf(aggr_label+al, sizeof(aggr_label)-al, "%s%d",
                               k?"-":"", topos[tp].aggr[k]);
            }

            for (int ai = 0; ai < cfg.namp; ai++) {
                int namp = cfg.amp_ladder[ai];
                if (namp > topos[tp].naggr) namp = topos[tp].naggr;

                for (int fi = 0; fi < cfg.nfreq; fi++) {
                    double f = cfg.freqs[fi];
                    if (f <= 0) continue;
                    /* Nyquist guard for the read cadence */
                    if (f > cfg.read_hz / 2.5) {
                        fprintf(stderr, "  skip f=%.1f (> read_hz/2.5=%.1f)\n", f, cfg.read_hz/2.5);
                        continue;
                    }
                    double feff_mag=0, feff_floor=0, feff_droop=0, ro_mag=0, ro_floor=0;
                    int rc = run_drive_point(&cfg, f, namp, victim,
                                             topos[tp].aggr, aggr_label, band, mode,
                                             &feff_mag, &feff_floor, &feff_droop,
                                             &ro_mag, &ro_floor);
                    if (rc == 1) {     /* temp veto: stop the whole sweep gracefully */
                        goto teardown;
                    }
                    if (rc != 0) continue;

                    double feff_snr = (feff_floor>0)?(feff_mag/feff_floor):0.0;
                    if (feff_mag > 0) feff_channel_live = 1;

                    /* verdict bookkeeping (use primary topology, idle band as ref) */
                    if (f >= POLE_HZ) {
                        if (feff_snr > best_abovepole_snr) {
                            best_abovepole_snr = feff_snr;
                            best_droop_sign = feff_droop;
                        }
                        if (tp == 0 && b == 0) {
                            abovepole_feff_by_amp[ai] = feff_mag;
                            abovepole_seen[ai] = 1;
                        }
                    } else {
                        double ro_snr = (ro_floor>0)?(ro_mag/ro_floor):0.0;
                        double s = feff_snr > ro_snr ? feff_snr : ro_snr;
                        if (s > belowpole_max_snr) belowpole_max_snr = s;
                    }
                    /* topology dependence at a fixed high-freq, max amp */
                    if (f >= POLE_HZ && namp == topos[tp].naggr) {
                        if (tp == 0) topo_feff_primary = feff_mag;
                        else         topo_feff_swapped = feff_mag;
                    }
                    /* collapse check: off-bin floor must stay below the drive bin for
                       a genuine tone; if floor >= mag the "signal" is not bin-specific */
                    if (feff_mag > 0 && feff_floor >= feff_mag) any_collapse_ok = 0;

                    /* ---- CONTENTION-CONTROL bookkeeping (compute vs memory) ----
                       The ring-osc timing loop is the only rail witness here, so the
                       compute-vs-memory comparison uses the VICTIM RING-OSC above-pole
                       lock-in (ro_mag) at matched amplitude, primary topology, idle band.
                       Per-amplitude magnitude feeds the monotonic-scaling test; the best
                       above-pole SNR and its survival of the off-bin floor feed the
                       comparison verdict. */
                    if (f >= POLE_HZ && tp == 0 && b == 0) {
                        double ro_snr = (ro_floor>0)?(ro_mag/ro_floor):0.0;
                        if (mode == AGG_ALU) {
                            ctrl_alu_ro_by_amp[ai] = ro_mag; ctrl_alu_seen[ai] = 1;
                            if (ro_snr > ctrl_alu_best_snr) {
                                ctrl_alu_best_snr = ro_snr; ctrl_alu_best_mag = ro_mag;
                            }
                            if (ro_snr >= SNR_REAL) ctrl_alu_survives_collapse = 1;
                        } else { /* AGG_CACHE = memory-bound */
                            ctrl_cache_ro_by_amp[ai] = ro_mag; ctrl_cache_seen[ai] = 1;
                            if (ro_snr > ctrl_cache_best_snr) {
                                ctrl_cache_best_snr = ro_snr; ctrl_cache_best_mag = ro_mag;
                            }
                            if (ro_snr >= SNR_REAL) ctrl_cache_survives_collapse = 1;
                        }
                    }
                }   /* for fi (drive sweep) */
            }       /* for ai (amplitude ladder) */
        }           /* for tp (topology) */
      }             /* for mi (aggressor mode: compute / memory) */
    }               /* for b  (temp band) */

teardown:
    for (int k = 0; k < nwarm; k++) aggressor_stop(&warmers[k]);

    /* ---- restore P-state ---- */
    if (cfg.do_pin) restore_pstate(&ps);

    /* ---- VERDICT ---- */
    /* monotonic amplitude scaling of the above-pole feff lock-in (primary/idle) */
    int amp_pts = 0; double amp_vals[NCPU_MAX];
    for (int i=0;i<cfg.namp;i++) if (abovepole_seen[i]) amp_vals[amp_pts++] = abovepole_feff_by_amp[i];
    int monotonic = (amp_pts >= 2);
    for (int i=1;i<amp_pts;i++) if (amp_vals[i] < amp_vals[i-1]) monotonic = 0;

    int topo_dependent = 0;
    if (topo_feff_swapped >= 0.0) {
        double denom = (topo_feff_primary > topo_feff_swapped ? topo_feff_primary : topo_feff_swapped);
        if (denom > 0 && fabs(topo_feff_primary - topo_feff_swapped)/denom > 0.30)
            topo_dependent = 1;
    }

    /* thresholds (documented, conservative). SNR_REAL/POLE_HZ defined above. */
    int above_pole_present = (best_abovepole_snr >= SNR_REAL);
    int droop_sign_ok      = (best_droop_sign > 0.0);   /* f_eff dips during load ON */
    int collapse_ok        = any_collapse_ok;           /* off-bin floor below drive bin */

    int rail_observed = feff_channel_live && above_pole_present && droop_sign_ok &&
                        collapse_ok && (monotonic || amp_pts < 2 ? above_pole_present : monotonic);
    /* require monotonic scaling explicitly when we have >=2 amplitude points */
    if (amp_pts >= 2 && !monotonic) rail_observed = 0;
    /* require topology dependence only if we actually ran the swap */
    if (cfg.swap_topology && !topo_dependent) rail_observed = 0;

    double T_end = read_k10temp_c();

    fprintf(g_csv,
        "# VERDICT_DETAIL,feff_channel_live=%d,precheck_idle=%.6f,precheck_load=%.6f,"
        "precheck_delta=%.6f,best_abovepole_snr=%.3f,droop_sign=%+.4f,"
        "monotonic_amp=%d,amp_pts=%d,collapse_ok=%d,topo_dependent=%d,"
        "belowpole_max_snr=%.3f,pole_hz=%.1f,start_T=%.1f,end_T=%.1f\n",
        feff_channel_live, pc_idle, pc_load, pc_load - pc_idle,
        best_abovepole_snr, best_droop_sign, monotonic, amp_pts, collapse_ok,
        topo_dependent, belowpole_max_snr, POLE_HZ, T0, T_end);

    const char *verdict =
        rail_observed ? "RAIL_COUPLING_OBSERVED" : "RAIL_INVISIBLE_SOFTWARE";
    fprintf(g_csv, "VERDICT,%s\n", verdict);
    fflush(g_csv);

    fprintf(stderr, "\n================= VERDICT =================\n");
    fprintf(stderr, "feff channel live (APERF/MPERF read OK)  : %s\n", feff_channel_live?"YES":"NO");
    fprintf(stderr, "PRECHECK feff ratio idle=%.6f load=%.6f delta=%.6f\n",
            pc_idle, pc_load, pc_load - pc_idle);
    fprintf(stderr, "above-pole (>=%.0fHz) drive-bin SNR (max) : %.3f (need >=%.1f) -> %s\n",
            POLE_HZ, best_abovepole_snr, (double)SNR_REAL, above_pole_present?"PRESENT":"floor");
    fprintf(stderr, "droop sign (f_eff dips under load ON)     : %+.4f -> %s\n",
            best_droop_sign, droop_sign_ok?"correct":"wrong/none");
    fprintf(stderr, "amplitude monotonic (above pole)         : %s (%d pts)\n",
            (amp_pts<2)?"n/a":(monotonic?"YES":"no"), amp_pts);
    fprintf(stderr, "off-bin collapse (floor below drive bin) : %s\n", collapse_ok?"YES":"no");
    fprintf(stderr, "topology dependence                      : %s\n",
            cfg.swap_topology?(topo_dependent?"YES":"no"):"not tested (--swap-topology)");
    fprintf(stderr, "below-pole max SNR (heating band)        : %.3f\n", belowpole_max_snr);
    fprintf(stderr, "temp start=%.1fC end=%.1fC\n", T0, T_end);
    fprintf(stderr, "VERDICT: %s\n",
        rail_observed ? "RAIL_COUPLING_OBSERVED (proceed; droop is software-visible)"
                      : "RAIL_INVISIBLE_SOFTWARE (obstruction; external rail probe required)");
    fprintf(stderr, "===========================================\n");

    /* =======================================================================
     * COMPARISON VERDICT: POWER-COUPLED vs CONTENTION-COUPLED.
     *
     * Above the thermal pole (>= POLE_HZ) we compare the VICTIM RING-OSC lock-in
     * magnitude under the COMPUTE-bound (no-shared-resource) aggressor against the
     * MEMORY-bound (shared-cache) aggressor, at matched amplitude.
     *
     *   POWER_COUPLED      : the compute-bound load (which shares NO cache/memory/NB
     *                        with the victim) produces a comparable above-pole lock-in
     *                        that (i) scales monotonically with intensity (amplitude
     *                        ladder) and (ii) survives the off-bin/scramble control.
     *                        Then the coupling must be via the shared power delivery
     *                        network -- there is no shared resource for it to contend on.
     *   CONTENTION_COUPLED : memory >> compute and the compute-bound signal sits at the
     *                        floor (does not survive off-bin). The coupling is shared-
     *                        resource contention, not the rail.
     *   MIXED              : neither extreme -- report the compute/memory magnitude ratio.
     *
     * "Comparable" uses RATIO_COMP (compute/memory >= this fraction). "memory >> compute"
     * uses RATIO_DOM (compute/memory <= this fraction). Magnitudes are the best above-pole
     * ring-osc lock-in for each mode at matched amplitude, primary topology, idle band.
     * ======================================================================= */
    #define RATIO_COMP 0.50   /* compute >= 50% of memory => comparable => power-coupled */
    #define RATIO_DOM  0.20   /* compute <= 20% of memory => memory dominates => contention */

    /* compute-bound monotonic-vs-amplitude scaling (primary/idle, above pole) */
    int   alu_amp_pts = 0;   double alu_amp_vals[NCPU_MAX];
    for (int i=0;i<cfg.namp;i++) if (ctrl_alu_seen[i]) alu_amp_vals[alu_amp_pts++] = ctrl_alu_ro_by_amp[i];
    int   alu_monotonic = (alu_amp_pts >= 2);
    for (int i=1;i<alu_amp_pts;i++) if (alu_amp_vals[i] < alu_amp_vals[i-1]) alu_monotonic = 0;

    int   cache_amp_pts = 0; double cache_amp_vals[NCPU_MAX];
    for (int i=0;i<cfg.namp;i++) if (ctrl_cache_seen[i]) cache_amp_vals[cache_amp_pts++] = ctrl_cache_ro_by_amp[i];
    int   cache_monotonic = (cache_amp_pts >= 2);
    for (int i=1;i<cache_amp_pts;i++) if (cache_amp_vals[i] < cache_amp_vals[i-1]) cache_monotonic = 0;

    /* compute/memory above-pole magnitude ratio at matched amplitude */
    double cmp_ratio = (ctrl_cache_best_mag > 0.0)
                         ? (ctrl_alu_best_mag / ctrl_cache_best_mag) : -1.0;

    /* both modes must have actually run for a head-to-head to be meaningful */
    int   have_both = (alu_ran && cache_ran && ctrl_alu_best_mag > 0.0 && ctrl_cache_best_mag > 0.0);

    /* compute-bound coupling qualifies as REAL (rail) if it survives the off-bin
       control and scales monotonically with intensity (or has too few points to test). */
    int   compute_real = ctrl_alu_survives_collapse &&
                         (alu_amp_pts < 2 ? 1 : alu_monotonic);

    const char *cmp_verdict;
    if (!have_both) {
        cmp_verdict = "COMPARISON_NA";   /* ran only one mode, or no above-pole signal */
    } else if (compute_real && cmp_ratio >= RATIO_COMP) {
        cmp_verdict = "POWER_COUPLED";
    } else if (cmp_ratio >= 0.0 && cmp_ratio <= RATIO_DOM && !ctrl_alu_survives_collapse) {
        cmp_verdict = "CONTENTION_COUPLED";
    } else {
        cmp_verdict = "MIXED";
    }

    fprintf(g_csv,
        "# COMPARISON_DETAIL,aggr_mode=%s,"
        "compute_ro_mag=%.6g,compute_best_snr=%.3f,compute_survives=%d,"
        "compute_monotonic=%d,compute_amp_pts=%d,"
        "memory_ro_mag=%.6g,memory_best_snr=%.3f,memory_survives=%d,"
        "memory_monotonic=%d,memory_amp_pts=%d,"
        "compute_over_memory_ratio=%.4f,ratio_comp=%.2f,ratio_dom=%.2f,pole_hz=%.1f\n",
        (cfg.aggr_sel==AGGSEL_ALU?"compute":cfg.aggr_sel==AGGSEL_CACHE?"memory":"both"),
        ctrl_alu_best_mag, ctrl_alu_best_snr, ctrl_alu_survives_collapse,
        alu_monotonic, alu_amp_pts,
        ctrl_cache_best_mag, ctrl_cache_best_snr, ctrl_cache_survives_collapse,
        cache_monotonic, cache_amp_pts,
        cmp_ratio, (double)RATIO_COMP, (double)RATIO_DOM, POLE_HZ);
    fprintf(g_csv, "COMPARISON_VERDICT,%s,compute_over_memory_ratio=%.4f\n",
            cmp_verdict, cmp_ratio);
    fflush(g_csv);

    fprintf(stderr, "\n=========== COMPARISON VERDICT ============\n");
    fprintf(stderr, "(above pole >=%.0fHz, victim ring-osc, matched amplitude)\n", POLE_HZ);
    fprintf(stderr, "compute-bound (no shared resource): mag=%.4g snr=%.3f survives_offbin=%s monotonic=%s(%d pts)\n",
            ctrl_alu_best_mag, ctrl_alu_best_snr, ctrl_alu_survives_collapse?"YES":"no",
            (alu_amp_pts<2)?"n/a":(alu_monotonic?"YES":"no"), alu_amp_pts);
    fprintf(stderr, "memory-bound  (shared L3/mem/NB)  : mag=%.4g snr=%.3f survives_offbin=%s monotonic=%s(%d pts)\n",
            ctrl_cache_best_mag, ctrl_cache_best_snr, ctrl_cache_survives_collapse?"YES":"no",
            (cache_amp_pts<2)?"n/a":(cache_monotonic?"YES":"no"), cache_amp_pts);
    if (have_both)
        fprintf(stderr, "compute/memory magnitude ratio    : %.4f (comparable>=%.2f, mem-dominates<=%.2f)\n",
                cmp_ratio, (double)RATIO_COMP, (double)RATIO_DOM);
    else
        fprintf(stderr, "compute/memory magnitude ratio    : n/a (need both modes with above-pole signal; use --aggr-mode both)\n");
    fprintf(stderr, "COMPARISON_VERDICT: %s\n",
        !have_both          ? "COMPARISON_NA (ran one mode or no above-pole signal)" :
        !strcmp(cmp_verdict,"POWER_COUPLED")
            ? "POWER_COUPLED (compute-bound couples too -> shared power delivery network)" :
        !strcmp(cmp_verdict,"CONTENTION_COUPLED")
            ? "CONTENTION_COUPLED (memory >> compute, compute at floor -> shared-resource contention)"
            : "MIXED (see compute/memory ratio above)");
    fprintf(stderr, "===========================================\n");

    if (g_csv && g_csv != stdout) fclose(g_csv);
    return 0;
}
