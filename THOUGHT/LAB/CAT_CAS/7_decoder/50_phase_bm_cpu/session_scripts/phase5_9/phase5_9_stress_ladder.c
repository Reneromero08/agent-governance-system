#define _GNU_SOURCE
#include "phase5_9_common.h"
#include "phase5_9_workers.h"
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>

/* ── Global defaults ─────────────────────────────────────────── */
static run_config_t g_cfg = {
    .run_id = "default",
    .measurement_core = 3,
    .tape_size = 256,
    .window_size = 256,
    .iterations = 50000,
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
static int g_randomize_order = 1;
static char g_stress_id[64] = "baseline";
static char g_stress_label[128] = "BASELINE_NO_STRESS";
static int g_stress_level = 0;
static double g_temp_limit = 65.0;

/* ── CLI parsing ─────────────────────────────────────────────── */
static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "  --core N               Measurement core (default: 3)\n"
        "  --workers C0,C1,...    Comma-separated worker cores\n"
        "  --worker-mode MODE     none|cache|mixed|thermal|integer\n"
        "  --iterations N         Trials (default: 50000)\n"
        "  --tape-size N          Catalytic tape bytes (default: 256)\n"
        "  --freq-label LABEL     Frequency label (default: nominal)\n"
        "  --vid-label LABEL      VID label (default: unknown)\n"
        "  --run-id ID            Run identifier\n"
        "  --output-dir DIR       Output directory (default: .)\n"
        "  --stress-id ID         Stress identifier (default: baseline)\n"
        "  --stress-label LABEL   Stress description\n"
        "  --stress-level N       Numeric stress level (0=stable, higher=more stress)\n"
        "  --temp-limit C         Thermal abort threshold C (default: 65)\n"
        "  --control MODE         empty|nop|irreversible|readonly\n"
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
        {"iterations", required_argument, 0, 'i'},
        {"tape-size", required_argument, 0, 't'},
        {"freq-label", required_argument, 0, 'f'},
        {"vid-label", required_argument, 0, 'v'},
        {"run-id", required_argument, 0, 'r'},
        {"output-dir", required_argument, 0, 'o'},
        {"stress-id", required_argument, 0, 'S'},
        {"stress-label", required_argument, 0, 'L'},
        {"stress-level", required_argument, 0, 'l'},
        {"temp-limit", required_argument, 0, 'T'},
        {"control", required_argument, 0, 'C'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:w:m:i:t:f:v:r:o:S:L:l:T:C:h", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'c': g_cfg.measurement_core = atoi(optarg); break;
        case 'w': parse_workers(optarg); break;
        case 'm': g_cfg.worker_mode = parse_mode(optarg); break;
        case 'i': g_cfg.iterations = atoi(optarg); break;
        case 't': g_cfg.tape_size = atoi(optarg); break;
        case 'f': snprintf(g_cfg.freq_label, sizeof(g_cfg.freq_label), "%s", optarg); break;
        case 'v': snprintf(g_cfg.vid_label, sizeof(g_cfg.vid_label), "%s", optarg); break;
        case 'r': snprintf(g_cfg.run_id, sizeof(g_cfg.run_id), "%s", optarg); break;
        case 'o': snprintf(g_output_dir, sizeof(g_output_dir), "%s", optarg); break;
        case 'S': snprintf(g_stress_id, sizeof(g_stress_id), "%s", optarg); break;
        case 'L': snprintf(g_stress_label, sizeof(g_stress_label), "%s", optarg); break;
        case 'l': g_stress_level = atoi(optarg); break;
        case 'T': g_temp_limit = atof(optarg); break;
        case 'C': g_control_mode = parse_control(optarg); break;
        case 'h': print_usage(argv[0]); exit(0);
        default: print_usage(argv[0]); exit(1);
        }
    }
}

/* ── Measurement helpers ──────────────────────────────────────── */

static uint64_t measure_rdtsc_overhead(int samples) {
    uint64_t sum = 0;
    for (int i = 0; i < samples; i++) {
        uint64_t t0 = rdtsc_start();
        uint64_t t1 = rdtsc_end();
        sum += (t1 - t0);
    }
    return sum / samples;
}

static void generate_tape_key(uint8_t *tape, uint8_t *key, size_t size, uint64_t seed) {
    for (size_t i = 0; i < size; i++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        tape[i] = (uint8_t)(seed >> 32);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        key[i] = (uint8_t)(seed >> 32);
    }
}

static void catalytic_xor_measured(
    volatile uint8_t *tape, const uint8_t *key, size_t size,
    uint64_t *rdtsc_raw, uint64_t *rdtsc_corrected,
    uint64_t *clock_ns, uint64_t overhead, int timing_mode)
{
    struct timespec ts0, ts1;
    if (timing_mode == 2) clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);

    uint64_t t0 = rdtsc_start();

    for (size_t i = 0; i < size; i++) tape[i] ^= key[i];
    for (size_t i = 0; i < size; i++) tape[i] ^= key[i];

    __asm__ volatile("" ::: "memory");
    uint64_t t1 = rdtsc_end();

    if (timing_mode == 2) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        *clock_ns = (ts1.tv_sec - ts0.tv_sec) * 1000000000ULL + (ts1.tv_nsec - ts0.tv_nsec);
    } else { *clock_ns = 0; }

    *rdtsc_raw = t1 - t0;
    *rdtsc_corrected = (*rdtsc_raw > overhead) ? (*rdtsc_raw - overhead) : 0;
}

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
    } else { *clock_ns = 0; }
    *rdtsc_raw = t1 - t0;
    *rdtsc_corrected = (*rdtsc_raw > overhead) ? (*rdtsc_raw - overhead) : 0;
}

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
    } else { *clock_ns = 0; }
    *rdtsc_raw = t1 - t0;
    *rdtsc_corrected = (*rdtsc_raw > overhead) ? (*rdtsc_raw - overhead) : 0;
}

/* ── Temperature read ────────────────────────────────────────── */
static double read_temperature(void) {
    FILE *f = fopen("/sys/class/hwmon/hwmon1/temp1_input", "r");
    if (!f) f = fopen("/sys/class/hwmon/hwmon0/temp1_input", "r");
    if (!f) f = fopen("/sys/class/hwmon/hwmon2/temp1_input", "r");
    if (!f) return -1.0;
    int millideg;
    if (fscanf(f, "%d", &millideg) != 1) { fclose(f); return -1.0; }
    fclose(f);
    return millideg / 1000.0;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a, vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

/* ── Main ────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    parse_args(argc, argv);

    mkdir(g_output_dir, 0755);

    if (pin_to_core(g_cfg.measurement_core) != 0) {
        fprintf(stderr, "FATAL: cannot pin to core %d\n", g_cfg.measurement_core);
        return 1;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    sched_getaffinity(0, sizeof(cpu_set_t), &cpuset);
    int pinned_core = -1;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset)) { pinned_core = i; break; }
    }
    int affinity_ok = (pinned_core == g_cfg.measurement_core);

    double temp_start = read_temperature();

    /* Thermal abort check */
    if (temp_start > g_temp_limit && temp_start > 0) {
        fprintf(stderr, "FATAL: temperature %.1fC exceeds limit %.1fC\n", temp_start, g_temp_limit);
        return 2;
    }

    uint64_t rdtsc_overhead = measure_rdtsc_overhead(1000);

    /* Open output files */
    FILE *f_raw = open_csv(g_output_dir, "raw_cycles.csv",
        "run_id,stress_id,stress_label,stress_level,trial_id,trial_order,"
        "iteration_index,measurement_core,worker_count,worker_mode,tape_size,"
        "freq_label,vid_label,temperature_start,temperature_end,"
        "rdtsc_cycles_raw,rdtsc_cycles_corrected,clock_ns,restore_ok,timing_mode");

    FILE *f_restore = open_csv(g_output_dir, "restoration_integrity.csv",
        "run_id,stress_id,trial_id,worker_count,worker_mode,tape_size,"
        "freq_label,vid_label,checksum_type,initial_checksum,final_checksum,"
        "hash_match,restore_failures,logical_bits_erased");

    FILE *f_stress = open_csv(g_output_dir, "stress_ladder.csv",
        "run_id,stress_id,stress_label,stress_level,tape_size,worker_mode,"
        "worker_count,iterations,restore_ok,restore_failures,"
        "rdtsc_mean,rdtsc_std,rdtsc_p50,rdtsc_p99,"
        "temp_start,temp_end,affinity_ok,failed_joins,migrations,"
        "distance_to_failure,restoration_margin,timing_instability_score,"
        "thermal_stress_score,worker_integrity_score");

    if (!f_raw || !f_restore || !f_stress) {
        fprintf(stderr, "FATAL: cannot open output files\n");
        return 1;
    }

    size_t tape_size = (size_t)g_cfg.tape_size;
    if (tape_size > MAX_TAPE_SIZE) {
        fprintf(stderr, "Tape size %zu exceeds max %d\n", tape_size, MAX_TAPE_SIZE);
        return 1;
    }

    uint8_t *tape = (uint8_t *)aligned_alloc_locked(tape_size, CATALYTIC_ALIGN);
    uint8_t *key = (uint8_t *)aligned_alloc_locked(tape_size, CATALYTIC_ALIGN);
    uint8_t *tape_backup = (uint8_t *)aligned_alloc_locked(tape_size, CATALYTIC_ALIGN);
    if (!tape || !key || !tape_backup) {
        fprintf(stderr, "FATAL: mem alloc\n");
        if (tape) { munlock(tape, tape_size); free(tape); }
        if (key) { munlock(key, tape_size); free(key); }
        if (tape_backup) { munlock(tape_backup, tape_size); free(tape_backup); }
        return 1;
    }

    uint64_t tape_seed = (uint64_t)time(NULL) ^ (uint64_t)getpid();
    generate_tape_key(tape, key, tape_size, tape_seed);
    memcpy(tape_backup, tape, tape_size);

    /* Start workers */
    worker_state_t workers[MAX_WORKERS];
    memset(workers, 0, sizeof(workers));
    int worker_start_failures = 0;
    if (g_cfg.worker_count > MAX_WORKERS) g_cfg.worker_count = MAX_WORKERS;

    if (g_cfg.worker_mode != WORKER_MODE_NONE && g_cfg.worker_count > 0) {
        for (int w = 0; w < g_cfg.worker_count; w++) {
            if (worker_start(&workers[w], w, g_cfg.worker_cores[w], g_cfg.worker_mode,
                             WORKER_BUFFER_DEFAULT, WORKER_STRIDE_DEFAULT) != 0) {
                fprintf(stderr, "Worker %d failed to start\n", w);
                worker_start_failures++;
            }
        }
        usleep(100000);
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
    for (int i = 0; i < g_cfg.iterations; i++) trial_order[i] = i;
    if (g_randomize_order) {
        uint64_t seed = 42;
        for (int i = g_cfg.iterations - 1; i > 0; i--) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = (int)(seed % (i + 1));
            int tmp = trial_order[i]; trial_order[i] = trial_order[j]; trial_order[j] = tmp;
        }
    }

    uint64_t checksum_initial = fnv1a_64(tape_backup, tape_size);

    /* ── Measurement loop ──────────────────────────────────── */
    int restore_failures = 0, total_restore_ok = 0, migrations = 0;
    setvbuf(stdout, NULL, _IONBF, 0);

    /* Accumulate cycle stats for p50/p99 */
    uint64_t *cycle_array = (uint64_t *)malloc((size_t)g_cfg.iterations * sizeof(uint64_t));
    if (!cycle_array) {
        fprintf(stderr, "FATAL: cycle array alloc\n");
        if (g_cfg.worker_count > 0) {
            worker_stop_all(workers, g_cfg.worker_count);
            worker_join_all(workers, g_cfg.worker_count);
        }
        munlock(tape, tape_size); munlock(key, tape_size); munlock(tape_backup, tape_size);
        free(tape); free(key); free(tape_backup); free(trial_order);
        return 1;
    }

    /* Read temperature once for the per-trial CSV column (non-critical, after measurement) */
    double temp_during = temp_start;

    for (int iter = 0; iter < g_cfg.iterations; iter++) {
        memcpy((void *)tape, tape_backup, tape_size);
        int core_before = sched_getcpu();

        uint64_t rdtsc_raw = 0, rdtsc_corrected = 0, clock_ns = 0;
        int restore_ok = 1;
        uint64_t checksum_after = 0;
        int hash_match = 0;

        if (g_control_mode == CONTROL_EMPTY_TIMING) {
            rdtsc_raw = rdtsc_overhead;
            rdtsc_corrected = 0;
            checksum_after = fnv1a_64(tape, tape_size);
        } else if (g_control_mode == CONTROL_NOP_LOOP) {
            /* simplified nop */
            uint64_t t0_n = rdtsc_start();
            for (size_t ni = 0; ni < (size_t)tape_size; ni++) __asm__ volatile("nop");
            __asm__ volatile("" ::: "memory");
            uint64_t t1_n = rdtsc_end();
            rdtsc_raw = t1_n - t0_n;
            rdtsc_corrected = (rdtsc_raw > rdtsc_overhead) ? (rdtsc_raw - rdtsc_overhead) : 0;
            checksum_after = fnv1a_64(tape, tape_size);
        } else if (g_control_mode == CONTROL_IRREVERSIBLE) {
            irreversible_xor_measured((volatile uint8_t *)tape, key, tape_size,
                                      &rdtsc_raw, &rdtsc_corrected, &clock_ns,
                                      rdtsc_overhead, g_cfg.timing_mode);
            checksum_after = fnv1a_64(tape, tape_size);
            memcpy((void *)tape, tape_backup, tape_size);
        } else if (g_control_mode == CONTROL_READ_ONLY) {
            readonly_measured((volatile uint8_t *)tape, tape_size,
                              &rdtsc_raw, &rdtsc_corrected, &clock_ns,
                              rdtsc_overhead, g_cfg.timing_mode);
            checksum_after = fnv1a_64(tape, tape_size);
        } else {
            catalytic_xor_measured((volatile uint8_t *)tape, key, tape_size,
                                   &rdtsc_raw, &rdtsc_corrected, &clock_ns,
                                   rdtsc_overhead, g_cfg.timing_mode);
            checksum_after = fnv1a_64(tape, tape_size);
        }

        hash_match = (checksum_initial == checksum_after);
        if (g_control_mode != CONTROL_IRREVERSIBLE) {
            restore_ok = (memcmp(tape, tape_backup, tape_size) == 0) && hash_match;
            if (!restore_ok) restore_failures++;
            total_restore_ok += restore_ok ? 1 : 0;
        } else {
            restore_ok = 1;
            total_restore_ok++;
        }

        int core_after = sched_getcpu();
        int migration_detected = (core_before != core_after);
        if (migration_detected) migrations++;

        cycle_array[iter] = rdtsc_corrected;

        /* Update temperature every 10000 trials (low-cost, non-measurement poll) */
        if (iter % 10000 == 0) {
            temp_during = read_temperature();
        }

        fprintf(f_raw,
            "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s,%.2f,%.2f,%llu,%llu,%llu,%d,%d\n",
            g_cfg.run_id, g_stress_id, g_stress_label, g_stress_level,
            iter, trial_order[iter], iter, g_cfg.measurement_core,
            g_cfg.worker_count, (int)g_cfg.worker_mode, g_cfg.tape_size,
            g_cfg.freq_label, g_cfg.vid_label, temp_start, temp_during,
            (unsigned long long)rdtsc_raw, (unsigned long long)rdtsc_corrected,
            (unsigned long long)clock_ns, restore_ok, g_cfg.timing_mode);

        fprintf(f_restore,
            "%s,%s,%d,%d,%d,%d,%s,%s,fnv1a_64,%llu,%llu,%d,%d,%d\n",
            g_cfg.run_id, g_stress_id, iter, g_cfg.worker_count,
            (int)g_cfg.worker_mode, g_cfg.tape_size,
            g_cfg.freq_label, g_cfg.vid_label,
            (unsigned long long)checksum_initial,
            (unsigned long long)checksum_after,
            hash_match, restore_failures,
            restore_ok ? 0 : (int)(tape_size * 8));

        if (iter % 10000 == 0 && iter > 0) {
            printf("\r  Trial %d/%d  restore_ok=%d  failures=%d",
                   iter, g_cfg.iterations, total_restore_ok, restore_failures);
        }
    }
    printf("\r  Trial %d/%d  restore_ok=%d  failures=%d  DONE\n",
           g_cfg.iterations, g_cfg.iterations, total_restore_ok, restore_failures);

    /* Stop workers */
    int failed_joins = 0, workers_started = 0;
    if (g_cfg.worker_count > 0) {
        worker_stop_all(workers, g_cfg.worker_count);
        failed_joins = worker_join_all(workers, g_cfg.worker_count);
        for (int w = 0; w < g_cfg.worker_count; w++) {
            if (workers[w].start_ok) workers_started++;
        }
    }

    double temp_end = read_temperature();

    /* Compute cycle stats */
    /* Sort cycle_array for p50/p99 */
    qsort(cycle_array, g_cfg.iterations, sizeof(uint64_t), cmp_u64);
    uint64_t p50 = cycle_array[g_cfg.iterations / 2];
    uint64_t p99 = cycle_array[g_cfg.iterations * 99 / 100];

    double sum = 0, sum_sq = 0;
    for (int i = 0; i < g_cfg.iterations; i++) {
        sum += (double)cycle_array[i];
        sum_sq += (double)cycle_array[i] * (double)cycle_array[i];
    }
    double mean = sum / g_cfg.iterations;
    double variance = (sum_sq / g_cfg.iterations) - (mean * mean);
    if (variance < 0) variance = 0;
    double std = sqrt(variance);

    /* ── distance_to_failure ───────────────────────────────── */
    double restoration_margin = (g_cfg.iterations > 0)
        ? (double)total_restore_ok / (double)g_cfg.iterations : 0.0;
    double timing_instability_score = (p99 > 0)
        ? (double)(p99 > p50 ? (double)(p99 - p50) / (double)p99 : 0.0) : 0.0;
    double thermal_stress_score = (g_temp_limit > 0 && temp_end > 0)
        ? (temp_end / g_temp_limit) : 0.0;
    double worker_integrity_score = (g_cfg.worker_count > 0)
        ? (double)(workers_started - failed_joins) / (double)g_cfg.worker_count : 1.0;

    /* distance_to_failure: composite score (higher = closer to failure) */
    double distance_to_failure =
        (1.0 - restoration_margin) * 0.4 +
        timing_instability_score * 0.2 +
        thermal_stress_score * 0.2 +
        (1.0 - worker_integrity_score) * 0.1 +
        ((double)migrations / (double)(g_cfg.iterations > 0 ? g_cfg.iterations : 1)) * 0.1;

    /* Write stress ladder row */
    fprintf(f_stress,
        "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%llu,%llu,%.2f,%.2f,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        g_cfg.run_id, g_stress_id, g_stress_label, g_stress_level,
        g_cfg.tape_size, (int)g_cfg.worker_mode, g_cfg.worker_count,
        g_cfg.iterations, total_restore_ok, restore_failures,
        mean, std, (unsigned long long)p50, (unsigned long long)p99,
        temp_start, temp_end, affinity_ok ? 1 : 0,
        failed_joins, migrations,
        distance_to_failure, restoration_margin, timing_instability_score,
        thermal_stress_score, worker_integrity_score);

    fclose(f_raw);
    fclose(f_restore);
    fclose(f_stress);

    /* Worker status CSV */
    if (g_cfg.worker_count > 0) {
        FILE *f_ws = open_csv(g_output_dir, "worker_status.csv",
            "run_id,worker_id,core_id,mode,start_ok,join_ok,buffer_mb_actual,worker_lifetime_ok");
        if (f_ws) {
            for (int w = 0; w < g_cfg.worker_count; w++) {
                fprintf(f_ws, "%s,%d,%d,%d,%d,%d,%zu,%d\n",
                    g_cfg.run_id, workers[w].worker_id, workers[w].core_id,
                    (int)workers[w].mode, workers[w].start_ok, workers[w].join_ok,
                    workers[w].buffer_mb_actual,
                    (workers[w].start_ok && workers[w].join_ok) ? 1 : 0);
            }
            fclose(f_ws);
        }
    }

    /* Operating points */
    FILE *f_ops = open_csv(g_output_dir, "operating_points.csv",
        "run_id,stress_id,freq_label,vid_label,measurement_core,"
        "temperature_start,temperature_end,tape_size,worker_count,"
        "worker_mode,rdtsc_overhead_cycles,affinity_ok");
    if (f_ops) {
        fprintf(f_ops, "%s,%s,%s,%s,%d,%.2f,%.2f,%d,%d,%d,%llu,%d\n",
            g_cfg.run_id, g_stress_id, g_cfg.freq_label, g_cfg.vid_label,
            g_cfg.measurement_core, temp_start, temp_end, g_cfg.tape_size,
            g_cfg.worker_count, (int)g_cfg.worker_mode,
            (unsigned long long)rdtsc_overhead, affinity_ok ? 1 : 0);
        fclose(f_ops);
    }

    /* TELEMETRY */
    FILE *f_tel = open_csv(g_output_dir, "TELEMETRY_PHASE5_9.txt", "");
    if (f_tel) {
        fprintf(f_tel,
            "============================================================\n"
            "EXP50 PHASE 5.9: BOUNDARY STRESS TEST\n"
            "Instability-Edge Geometry Probe\n"
            "============================================================\n\n"
            "Date: %s\n"
            "CPU: AMD Phenom II X6 1090T\n"
            "Stress ID: %s\n"
            "Stress Label: %s\n"
            "Stress Level: %d\n\n"
            "Measurement core: %d\n"
            "Affinity held: %s\n"
            "RDTSC overhead: %llu cycles\n\n"
            "Run ID: %s\n"
            "Tape size: %d\n"
            "Trial count: %d\n"
            "Worker mode: %d\n"
            "Worker count: %d\n"
            "Frequency label: %s\n"
            "VID label: %s\n\n"
            "Restoration pass: %d/%d\n"
            "Restore failures: %d\n"
            "Migrations: %d\n\n"
            "Temperature start: %.2f C\n"
            "Temperature end: %.2f C\n"
            "Workers started: %d\n"
            "Worker start failures: %d\n"
            "Failed joins: %d\n"
            "Worker lifetime OK: %s\n\n"
            "Raw cycles mean: %.6f\n"
            "Raw cycles std: %.6f\n"
            "Raw cycles p50: %llu\n"
            "Raw cycles p99: %llu\n\n"
            "distance_to_failure: %.6f\n"
            "restoration_margin: %.6f\n"
            "timing_instability_score: %.6f\n"
            "thermal_stress_score: %.6f\n"
            "worker_integrity_score: %.6f\n",
            __DATE__,
            g_stress_id, g_stress_label, g_stress_level,
            g_cfg.measurement_core,
            affinity_ok ? "YES" : "NO",
            (unsigned long long)rdtsc_overhead,
            g_cfg.run_id, g_cfg.tape_size, g_cfg.iterations,
            (int)g_cfg.worker_mode, g_cfg.worker_count,
            g_cfg.freq_label, g_cfg.vid_label,
            total_restore_ok, g_cfg.iterations, restore_failures, migrations,
            temp_start, temp_end,
            workers_started, worker_start_failures, failed_joins,
            (worker_start_failures == 0 && failed_joins == 0) ? "YES" : "NO",
            mean, std,
            (unsigned long long)p50, (unsigned long long)p99,
            distance_to_failure, restoration_margin,
            timing_instability_score, thermal_stress_score,
            worker_integrity_score);
        fclose(f_tel);
    }

    munlock(tape, tape_size); munlock(key, tape_size); munlock(tape_backup, tape_size);
    free(tape); free(key); free(tape_backup);
    free(trial_order); free(cycle_array);

    printf("\nDone. Output in %s/\n", g_output_dir);
    printf("Restoration: %d/%d OK, %d failures\n", total_restore_ok, g_cfg.iterations, restore_failures);
    printf("Affinity held: %s\n", affinity_ok ? "YES" : "NO");
    printf("distance_to_failure: %.6f\n", distance_to_failure);

    return (restore_failures > 0 || worker_start_failures > 0 || failed_joins > 0) ? 1 : 0;
}
