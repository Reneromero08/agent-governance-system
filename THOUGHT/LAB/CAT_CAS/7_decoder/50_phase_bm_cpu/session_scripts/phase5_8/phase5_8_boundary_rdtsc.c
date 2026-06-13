#define _GNU_SOURCE
#include "phase5_8_common.h"
#include "phase5_8_workers.h"
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>

/* ── Global defaults ─────────────────────────────────────────── */
static run_config_t g_cfg = {
    .run_id = "default",
    .measurement_core = 3,
    .tape_size = 256,
    .window_size = 256,
    .iterations = 100000,
    .worker_mode = WORKER_MODE_NONE,
    .worker_placement = PLACEMENT_NO_WORKERS,
    .worker_count = 0,
    .freq_label = "nominal",
    .vid_label = "unknown",
    .measured_vcore = 0.0,
    .timing_mode = 1
};

static char g_output_dir[512] = ".";
static int g_control_mode = -1;
static int g_randomize_order = 0;

/* ── CLI parsing ─────────────────────────────────────────────── */
static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "  --core N               Measurement core (default: 3)\n"
        "  --workers C0,C1,...    Comma-separated worker cores\n"
        "  --worker-mode MODE     none|cache|mixed|thermal\n"
        "  --placement MODE       offcore|same_l3|no_workers\n"
        "  --iterations N         Trials (default: 100000)\n"
        "  --window-size N        Window size for analyzer (default: 256)\n"
        "  --tape-size N          Catalytic tape bytes (default: 256)\n"
        "  --freq-label LABEL     Frequency label (default: nominal)\n"
        "  --vid-label LABEL      VID label (default: unknown)\n"
        "  --run-id ID            Run identifier\n"
        "  --output-dir DIR       Output directory (default: .)\n"
        "  --control MODE         empty|nop|irreversible|readonly\n"
        "  --randomize-order      Randomize trial order\n"
        "  --help                 This message\n",
        prog);
}

static int parse_workers(const char *arg) {
    char *s = strdup(arg);
    char *tok = strtok(s, ",");
    int n = 0;
    while (tok && n < MAX_WORKERS) {
        g_cfg.worker_cores[n++] = atoi(tok);
        tok = strtok(NULL, ",");
    }
    g_cfg.worker_count = n;
    free(s);
    return n;
}

static int parse_mode(const char *arg) {
    if (!strcmp(arg, "none") || !strcmp(arg, "no_workers"))
        return WORKER_MODE_NONE;
    if (!strcmp(arg, "cache")) return WORKER_MODE_CACHE_HAMMER;
    if (!strcmp(arg, "mixed")) return WORKER_MODE_MIXED;
    if (!strcmp(arg, "thermal")) return WORKER_MODE_THERMAL;
    if (!strcmp(arg, "integer")) return WORKER_MODE_INTEGER_CHURN;
    fprintf(stderr, "Unknown worker mode: %s\n", arg);
    exit(1);
}

static int parse_placement(const char *arg) {
    if (!strcmp(arg, "offcore")) return PLACEMENT_OFFCORE;
    if (!strcmp(arg, "same_l3")) return PLACEMENT_SAME_L3;
    if (!strcmp(arg, "no_workers")) return PLACEMENT_NO_WORKERS;
    fprintf(stderr, "Unknown placement: %s\n", arg);
    exit(1);
}

static int parse_control(const char *arg) {
    if (!strcmp(arg, "empty")) return CONTROL_EMPTY_TIMING;
    if (!strcmp(arg, "nop")) return CONTROL_NOP_LOOP;
    if (!strcmp(arg, "irreversible")) return CONTROL_IRREVERSIBLE;
    if (!strcmp(arg, "readonly")) return CONTROL_READ_ONLY;
    fprintf(stderr, "Unknown control: %s\n", arg);
    exit(1);
}

static void parse_args(int argc, char **argv) {
    static struct option long_opts[] = {
        {"core", required_argument, 0, 'c'},
        {"workers", required_argument, 0, 'w'},
        {"worker-mode", required_argument, 0, 'm'},
        {"placement", required_argument, 0, 'p'},
        {"iterations", required_argument, 0, 'i'},
        {"window-size", required_argument, 0, 'W'},
        {"tape-size", required_argument, 0, 't'},
        {"freq-label", required_argument, 0, 'f'},
        {"vid-label", required_argument, 0, 'v'},
        {"run-id", required_argument, 0, 'r'},
        {"output-dir", required_argument, 0, 'o'},
        {"control", required_argument, 0, 'C'},
        {"randomize-order", no_argument, &g_randomize_order, 1},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:w:m:p:i:W:t:f:v:r:o:C:h", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'c': g_cfg.measurement_core = atoi(optarg); break;
        case 'w': parse_workers(optarg); break;
        case 'm': g_cfg.worker_mode = parse_mode(optarg); break;
        case 'p': g_cfg.worker_placement = parse_placement(optarg); break;
        case 'i': g_cfg.iterations = atoi(optarg); break;
        case 'W': g_cfg.window_size = atoi(optarg); break;
        case 't': g_cfg.tape_size = atoi(optarg); break;
        case 'f': snprintf(g_cfg.freq_label, sizeof(g_cfg.freq_label), "%s", optarg); break;
        case 'v': snprintf(g_cfg.vid_label, sizeof(g_cfg.vid_label), "%s", optarg); break;
        case 'r': snprintf(g_cfg.run_id, sizeof(g_cfg.run_id), "%s", optarg); break;
        case 'o': snprintf(g_output_dir, sizeof(g_output_dir), "%s", optarg); break;
        case 'C': g_control_mode = parse_control(optarg); break;
        case 'h': print_usage(argv[0]); exit(0);
        case 0: break;
        default: print_usage(argv[0]); exit(1);
        }
    }
}

/* ── Measurement helpers ──────────────────────────────────────── */

/* Measure RDTSC overhead by timing an empty section */
static uint64_t measure_rdtsc_overhead(int samples) {
    uint64_t sum = 0;
    for (int i = 0; i < samples; i++) {
        uint64_t t0 = rdtsc_start();
        uint64_t t1 = rdtsc_end();
        sum += (t1 - t0);
    }
    return sum / samples;
}

/* Generate deterministic tape and key */
static void generate_tape_key(uint8_t *tape, uint8_t *key, size_t size, uint64_t seed) {
    for (size_t i = 0; i < size; i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        tape[i] = (uint8_t)(seed >> 32);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        key[i] = (uint8_t)(seed >> 32);
    }
}

/* Catalytic forward + reverse XOR with timing */
static void catalytic_xor_measured(
    volatile uint8_t *tape, const uint8_t *key, size_t size,
    uint64_t *rdtsc_raw, uint64_t *rdtsc_corrected,
    uint64_t *clock_ns, uint64_t overhead, int timing_mode)
{
    struct timespec ts0, ts1;

    if (timing_mode == 2) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);
    }

    uint64_t t0 = rdtsc_start();

    /* Forward: T' = T XOR K */
    for (size_t i = 0; i < size; i++) {
        tape[i] ^= key[i];
    }

    /* Reverse: T'' = T' XOR K */
    for (size_t i = 0; i < size; i++) {
        tape[i] ^= key[i];
    }

    /* Compiler barrier — do not optimize away */
    __asm__ volatile("" ::: "memory");

    uint64_t t1 = rdtsc_end();

    if (timing_mode == 2) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        *clock_ns = (ts1.tv_sec - ts0.tv_sec) * 1000000000ULL + (ts1.tv_nsec - ts0.tv_nsec);
    } else {
        *clock_ns = 0;
    }

    *rdtsc_raw = t1 - t0;
    *rdtsc_corrected = (*rdtsc_raw > overhead) ? (*rdtsc_raw - overhead) : 0;
}

/* Irreversible destructive write */
static void irreversible_xor_measured(
    volatile uint8_t *tape, const uint8_t *key, size_t size,
    uint64_t *rdtsc_raw, uint64_t *rdtsc_corrected,
    uint64_t *clock_ns, uint64_t overhead, int timing_mode)
{
    struct timespec ts0, ts1;
    if (timing_mode == 2) clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);

    uint64_t t0 = rdtsc_start();
    for (size_t i = 0; i < size; i++) tape[i] ^= key[i];
    __asm__ volatile("" ::: "memory");
    uint64_t t1 = rdtsc_end();

    if (timing_mode == 2) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        *clock_ns = (ts1.tv_sec - ts0.tv_sec) * 1000000000ULL + (ts1.tv_nsec - ts0.tv_nsec);
    } else {
        *clock_ns = 0;
    }
    *rdtsc_raw = t1 - t0;
    *rdtsc_corrected = (*rdtsc_raw > overhead) ? (*rdtsc_raw - overhead) : 0;
}

/* Read-only workload */
static void readonly_measured(
    volatile uint8_t *tape, size_t size,
    uint64_t *rdtsc_raw, uint64_t *rdtsc_corrected,
    uint64_t *clock_ns, uint64_t overhead, int timing_mode)
{
    volatile uint64_t sum = 0;
    struct timespec ts0, ts1;
    if (timing_mode == 2) clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);

    uint64_t t0 = rdtsc_start();
    for (size_t i = 0; i + sizeof(uint64_t) <= size; i += sizeof(uint64_t)) {
        sum += *(volatile uint64_t *)(tape + i);
    }
    __asm__ volatile("" ::: "memory");
    uint64_t t1 = rdtsc_end();

    if (timing_mode == 2) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        *clock_ns = (ts1.tv_sec - ts0.tv_sec) * 1000000000ULL + (ts1.tv_nsec - ts0.tv_nsec);
    } else {
        *clock_ns = 0;
    }
    *rdtsc_raw = t1 - t0;
    *rdtsc_corrected = (*rdtsc_raw > overhead) ? (*rdtsc_raw - overhead) : 0;
}

/* NOP loop measurement */
static void nop_loop_measured(
    uint64_t *rdtsc_raw, uint64_t *rdtsc_corrected,
    uint64_t *clock_ns, uint64_t overhead, int timing_mode, size_t nops)
{
    struct timespec ts0, ts1;
    if (timing_mode == 2) clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);

    uint64_t t0 = rdtsc_start();
    for (size_t i = 0; i < nops; i++) {
        __asm__ volatile("nop");
    }
    __asm__ volatile("" ::: "memory");
    uint64_t t1 = rdtsc_end();

    if (timing_mode == 2) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        *clock_ns = (ts1.tv_sec - ts0.tv_sec) * 1000000000ULL + (ts1.tv_nsec - ts0.tv_nsec);
    } else {
        *clock_ns = 0;
    }
    *rdtsc_raw = t1 - t0;
    *rdtsc_corrected = (*rdtsc_raw > overhead) ? (*rdtsc_raw - overhead) : 0;
}

/* ── Temperature read ────────────────────────────────────────── */
static double read_temperature(void) {
    /* k10temp via /sys/class/hwmon */
    FILE *f = fopen("/sys/class/hwmon/hwmon1/temp1_input", "r");
    if (!f) f = fopen("/sys/class/hwmon/hwmon0/temp1_input", "r");
    if (!f) f = fopen("/sys/class/hwmon/hwmon2/temp1_input", "r");
    if (!f) return -1.0;
    int millideg;
    if (fscanf(f, "%d", &millideg) != 1) { fclose(f); return -1.0; }
    fclose(f);
    return millideg / 1000.0;
}

/* ── Generate trial order ─────────────────────────────────────── */
static void generate_trial_order(int *order, int n, int randomize) {
    for (int i = 0; i < n; i++) order[i] = i;
    if (randomize) {
        uint64_t seed = 42;
        for (int i = n - 1; i > 0; i--) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = (int)(seed % (i + 1));
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
    }
}

/* ── Main ────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    parse_args(argc, argv);

    /* Create output directory */
    mkdir(g_output_dir, 0755);

    /* Pin measurement thread */
    if (pin_to_core(g_cfg.measurement_core) != 0) {
        fprintf(stderr, "FATAL: cannot pin to core %d\n", g_cfg.measurement_core);
        return 1;
    }

    /* Verify affinity */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset);
    int pinned_core = -1;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset)) { pinned_core = i; break; }
    }
    int affinity_ok = (pinned_core == g_cfg.measurement_core);

    /* Read temperature */
    double temp_start = read_temperature();

    /* Measure RDTSC overhead */
    uint64_t rdtsc_overhead = measure_rdtsc_overhead(1000);

    /* Open output files */
    FILE *f_raw = open_csv(g_output_dir, "raw_cycles.csv",
        "run_id,trial_id,trial_order,iteration_index,measurement_core,"
        "worker_count,worker_mode,worker_placement,tape_size,window_size,"
        "freq_label,vid_label,measured_vcore,temperature_start,temperature_end,"
        "rdtsc_cycles_raw,rdtsc_cycles_corrected,clock_ns,restore_ok,timing_mode");

    FILE *f_restore = open_csv(g_output_dir, "restoration_integrity.csv",
        "run_id,trial_id,worker_count,worker_mode,worker_placement,"
        "tape_size,freq_label,vid_label,checksum_type,initial_checksum,"
        "final_checksum,hash_match,restore_failures,logical_bits_erased");

    if (!f_raw || !f_restore) {
        fprintf(stderr, "FATAL: cannot open output files in %s\n", g_output_dir);
        return 1;
    }

    /* Allocate tape and key */
    size_t tape_size = (size_t)g_cfg.tape_size;
    if (tape_size > MAX_TAPE_SIZE) {
        fprintf(stderr, "Tape size %zu exceeds max %d\n", tape_size, MAX_TAPE_SIZE);
        return 1;
    }

    uint8_t *tape = (uint8_t *)aligned_alloc_locked(tape_size, CATALYTIC_ALIGN);
    uint8_t *key = (uint8_t *)aligned_alloc_locked(tape_size, CATALYTIC_ALIGN);
    uint8_t *tape_backup = (uint8_t *)aligned_alloc_locked(tape_size, CATALYTIC_ALIGN);

    if (!tape || !key || !tape_backup) {
        fprintf(stderr, "FATAL: memory allocation failed\n");
        if (tape) { munlock(tape, tape_size); free(tape); }
        if (key) { munlock(key, tape_size); free(key); }
        if (tape_backup) { munlock(tape_backup, tape_size); free(tape_backup); }
        return 1;
    }

    /* Generate tape and key */
    uint64_t tape_seed = (uint64_t)time(NULL) ^ (uint64_t)getpid();
    generate_tape_key(tape, key, tape_size, tape_seed);
    memcpy(tape_backup, tape, tape_size);

    /* Start workers */
    worker_state_t workers[MAX_WORKERS];
    memset(workers, 0, sizeof(workers));
    int worker_start_failures = 0;
    if (g_cfg.worker_count > MAX_WORKERS)
        g_cfg.worker_count = MAX_WORKERS;

    if (g_cfg.worker_mode != WORKER_MODE_NONE && g_cfg.worker_count > 0) {
        for (int w = 0; w < g_cfg.worker_count; w++) {
            worker_mode_t wm = g_cfg.worker_mode;
            if (worker_start(&workers[w], w, g_cfg.worker_cores[w], wm,
                             WORKER_BUFFER_DEFAULT, WORKER_STRIDE_DEFAULT) != 0) {
                fprintf(stderr, "Worker %d failed to start\n", w);
                worker_start_failures++;
            }
        }
        /* Stabilization delay */
        usleep(100000); /* 100ms */
    }

    /* Generate trial order */
    int *trial_order = (int *)malloc((size_t)g_cfg.iterations * sizeof(int));
    if (!trial_order) {
        fprintf(stderr, "FATAL: trial order alloc\n");
        if (g_cfg.worker_count > 0) {
            worker_stop_all(workers, g_cfg.worker_count);
            worker_join_all(workers, g_cfg.worker_count);
        }
        munlock(tape, tape_size); munlock(key, tape_size); munlock(tape_backup, tape_size);
        free(tape); free(key); free(tape_backup);
        return 1;
    }
    generate_trial_order(trial_order, g_cfg.iterations, g_randomize_order);

    /* Pre-compute initial checksum */
    uint64_t checksum_initial = fnv1a_64(tape_backup, tape_size);

    /* ── Measurement loop ──────────────────────────────────── */
    int restore_failures = 0;
    int total_restore_ok = 0;

    /* Set stdout to line-buffered for progress */
    setvbuf(stdout, NULL, _IONBF, 0);

    for (int iter = 0; iter < g_cfg.iterations; iter++) {
        /* Reset tape from backup */
        memcpy((void *)tape, tape_backup, tape_size);

        /* Record which core we're on before measurement */
        int core_before = sched_getcpu();

        uint64_t rdtsc_raw = 0, rdtsc_corrected = 0, clock_ns = 0;
        int restore_ok = 1;
        uint64_t checksum_after = 0;
        int hash_match = 0;

        if (g_control_mode == CONTROL_EMPTY_TIMING) {
            /* Use pre-measured overhead — avoids inline RDTSCP register allocation issues */
            rdtsc_raw = rdtsc_overhead;
            rdtsc_corrected = 0;  /* overhead is already the correction */
            checksum_after = fnv1a_64(tape, tape_size);
        } else if (g_control_mode == CONTROL_NOP_LOOP) {
            nop_loop_measured(&rdtsc_raw, &rdtsc_corrected, &clock_ns,
                              rdtsc_overhead, g_cfg.timing_mode, tape_size);
            checksum_after = fnv1a_64(tape, tape_size);
        } else if (g_control_mode == CONTROL_IRREVERSIBLE) {
            irreversible_xor_measured((volatile uint8_t *)tape, key, tape_size,
                                      &rdtsc_raw, &rdtsc_corrected, &clock_ns,
                                      rdtsc_overhead, g_cfg.timing_mode);
            checksum_after = fnv1a_64(tape, tape_size);
            /* Restore tape from backup after irreversible */
            memcpy((void *)tape, tape_backup, tape_size);
        } else if (g_control_mode == CONTROL_READ_ONLY) {
            readonly_measured((volatile uint8_t *)tape, tape_size,
                              &rdtsc_raw, &rdtsc_corrected, &clock_ns,
                              rdtsc_overhead, g_cfg.timing_mode);
            checksum_after = fnv1a_64(tape, tape_size);
        } else {
            /* Default: catalytic reversible */
            catalytic_xor_measured((volatile uint8_t *)tape, key, tape_size,
                                   &rdtsc_raw, &rdtsc_corrected, &clock_ns,
                                   rdtsc_overhead, g_cfg.timing_mode);
            checksum_after = fnv1a_64(tape, tape_size);
        }

        /* Verify restoration */
        hash_match = (checksum_initial == checksum_after);

        if (g_control_mode != CONTROL_IRREVERSIBLE) {
            restore_ok = (memcmp(tape, tape_backup, tape_size) == 0) && hash_match;
            if (!restore_ok) restore_failures++;
            total_restore_ok += restore_ok ? 1 : 0;
        } else {
            /* Irreversible: restoration doesn't apply, but we backed up */
            restore_ok = 1;
            total_restore_ok++;
        }

        /* Record core after measurement */
        int core_after = sched_getcpu();
        int migration_detected = (core_before != core_after);

        /* Write raw cycle row — outside timing loop */
        fprintf(f_raw,
            "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s,%.4f,%.2f,%.2f,%llu,%llu,%llu,%d,%d\n",
            g_cfg.run_id,
            iter,
            trial_order[iter],
            iter,
            g_cfg.measurement_core,
            g_cfg.worker_count,
            (int)g_cfg.worker_mode,
            (int)g_cfg.worker_placement,
            g_cfg.tape_size,
            g_cfg.window_size,
            g_cfg.freq_label,
            g_cfg.vid_label,
            g_cfg.measured_vcore,
            temp_start,
            read_temperature(),
            (unsigned long long)rdtsc_raw,
            (unsigned long long)rdtsc_corrected,
            (unsigned long long)clock_ns,
            restore_ok,
            g_cfg.timing_mode);

        /* Write restoration row */
        fprintf(f_restore,
            "%s,%d,%d,%d,%d,%d,%s,%s,fnv1a_64,%llu,%llu,%d,%d,%d\n",
            g_cfg.run_id,
            iter,
            g_cfg.worker_count,
            (int)g_cfg.worker_mode,
            (int)g_cfg.worker_placement,
            g_cfg.tape_size,
            g_cfg.freq_label,
            g_cfg.vid_label,
            (unsigned long long)checksum_initial,
            (unsigned long long)checksum_after,
            hash_match,
            restore_failures,
            restore_ok ? 0 : (int)(tape_size * 8));

        /* Migration audit */
        if (migration_detected) {
            fprintf(stderr, "WARNING: migration detected at trial %d: core %d -> %d\n",
                    iter, core_before, core_after);
        }

        /* Progress indicator every 10000 trials */
        if (iter % 10000 == 0 && iter > 0) {
            printf("\r  Trial %d/%d  restore_ok=%d  failures=%d",
                   iter, g_cfg.iterations, total_restore_ok, restore_failures);
        }
    }
    printf("\r  Trial %d/%d  restore_ok=%d  failures=%d  DONE\n",
           g_cfg.iterations, g_cfg.iterations, total_restore_ok, restore_failures);

    /* Stop workers and capture join status */
    int failed_joins = 0;
    int workers_started = 0;
    if (g_cfg.worker_count > 0) {
        worker_stop_all(workers, g_cfg.worker_count);
        failed_joins = worker_join_all(workers, g_cfg.worker_count);
        for (int w = 0; w < g_cfg.worker_count; w++) {
            if (workers[w].start_ok) workers_started++;
        }
    }

    double temp_end = read_temperature();

    /* Close CSV files */
    fclose(f_raw);
    fclose(f_restore);

    /* Write operating points */
    FILE *f_ops = open_csv(g_output_dir, "operating_points.csv",
        "run_id,freq_label,vid_label,measured_vcore,"
        "measurement_core,temperature_start,temperature_end,timing_mode,"
        "tape_size,worker_count,worker_mode,rdtsc_overhead_cycles");
    if (f_ops) {
        fprintf(f_ops, "%s,%s,%s,%.4f,%d,%.2f,%.2f,%d,%d,%d,%d,%llu\n",
                g_cfg.run_id, g_cfg.freq_label, g_cfg.vid_label, g_cfg.measured_vcore,
                g_cfg.measurement_core, temp_start, temp_end, g_cfg.timing_mode,
                g_cfg.tape_size, g_cfg.worker_count, (int)g_cfg.worker_mode,
                (unsigned long long)rdtsc_overhead);
        fclose(f_ops);
    }

    /* Write worker status */
    if (g_cfg.worker_count > 0) {
        FILE *f_ws = open_csv(g_output_dir, "worker_status.csv",
            "run_id,worker_id,core_id,mode,start_ok,join_ok,"
            "buffer_mb_actual,mlock_used,worker_lifetime_ok");
        if (f_ws) {
            for (int w = 0; w < g_cfg.worker_count; w++) {
                fprintf(f_ws, "%s,%d,%d,%d,%d,%d,%zu,%d,%d\n",
                    g_cfg.run_id,
                    workers[w].worker_id,
                    workers[w].core_id,
                    (int)workers[w].mode,
                    workers[w].start_ok,
                    workers[w].join_ok,
                    workers[w].buffer_mb_actual,
                    workers[w].mlock_used,
                    (workers[w].start_ok && workers[w].join_ok) ? 1 : 0);
            }
            fclose(f_ws);
        }
    }

    /* Write telemetry */
    FILE *f_tel = open_csv(g_output_dir, "TELEMETRY_PHASE5_8.txt", "");
    if (f_tel) {
        fprintf(f_tel,
            "============================================================\n"
            "EXP50 PHASE 5.8: BARE-METAL HOLOGRAPHIC BOUNDARY PROBE\n"
            "============================================================\n\n"
            "Date/Time: %s\n"
            "CPU: AMD Phenom II X6 1090T\n"
            "Kernel: Linux\n"
            "Compiler: gcc -O2 -march=native -std=c11\n\n"
            "Measurement core: %d\n"
            "Affinity held: %s\n"
            "Timing mode: %s\n"
            "RDTSC overhead: %llu cycles\n\n"
            "Run ID: %s\n"
            "Tape size: %d\n"
            "Window size: %d\n"
            "Trial count: %d\n"
            "Worker mode: %d\n"
            "Worker count: %d\n"
            "Worker placement: %d\n"
            "Frequency label: %s\n"
            "VID label: %s\n\n"
            "Restoration pass count: %d/%d\n"
            "Restore failures: %d\n\n"
            "Temperature start: %.2f C\n"
            "Temperature end: %.2f C\n"
            "Workers started: %d\n"
            "Worker start failures: %d\n"
            "Failed joins: %d\n"
            "Worker lifetime OK: %s\n",
            __DATE__,
            g_cfg.measurement_core,
            affinity_ok ? "YES" : "NO",
            g_cfg.timing_mode == 1 ? "RDTSCP serialized" : "CLOCK_MONOTONIC_RAW",
            (unsigned long long)rdtsc_overhead,
            g_cfg.run_id,
            g_cfg.tape_size,
            g_cfg.window_size,
            g_cfg.iterations,
            (int)g_cfg.worker_mode,
            g_cfg.worker_count,
            (int)g_cfg.worker_placement,
            g_cfg.freq_label,
            g_cfg.vid_label,
            total_restore_ok,
            g_cfg.iterations,
            restore_failures,
            temp_start,
            temp_end,
            workers_started,
            worker_start_failures,
            failed_joins,
            (worker_start_failures == 0 && failed_joins == 0) ? "YES" : "NO");
        fclose(f_tel);
    }

    /* Cleanup */
    munlock(tape, tape_size);
    munlock(key, tape_size);
    munlock(tape_backup, tape_size);
    free(tape);
    free(key);
    free(tape_backup);
    free(trial_order);

    printf("\nDone. Output in %s/\n", g_output_dir);
    printf("Restoration: %d/%d OK, %d failures\n",
           total_restore_ok, g_cfg.iterations, restore_failures);
    printf("Affinity held: %s\n", affinity_ok ? "YES" : "NO");
    printf("RDTSC overhead: %llu cycles\n", (unsigned long long)rdtsc_overhead);

    return (restore_failures > 0 || worker_start_failures > 0 || failed_joins > 0) ? 1 : 0;
}
