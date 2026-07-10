#include "combined_pdn_hardware.c"
#include "gate_a_engineering_smoke_runtime.h"

/*
 * Gate A is deliberately a separate bounded entry point inside this physical
 * runtime, not a second acquisition engine.  It reuses the runtime's TSC,
 * sender waveform, receiver capture, capture-quality, raw-record, affinity and
 * temperature/frequency observation primitives.  Unlike run_hardware(), it
 * has no path to cpufreq writes, voltage control, restoration, or MSR access.
 */

static const char *gate_a_tokens[16] = {
    "I", "I", "I", "I", "C0", "D0", "S0E", "S0E",
    "S0E", "S0E", "O0", "O0", "A0P", "A0N", "T", "T"
};

#ifdef GATE_A_COMPILED_AUTHORITY_SHA256
static const char *gate_a_runtime_authority_sha256 =
    GATE_A_COMPILED_AUTHORITY_SHA256;
#else
static const char *gate_a_runtime_authority_sha256 = NULL;
#endif
#ifdef GATE_A_COMPILED_OUTPUT_ROOT
static const char *gate_a_runtime_output_root = GATE_A_COMPILED_OUTPUT_ROOT;
#else
static const char *gate_a_runtime_output_root = NULL;
#endif

static int gate_a_driven_slot(int slot) {
    return (slot >= 6 && slot <= 9) || slot == 12 || slot == 13;
}

static int gate_a_phase_index(int slot) {
    return slot == 13 ? 4 : 0;
}

static int gate_a_sign(int slot) {
    return slot == 13 ? -1 : 1;
}

static const char *gate_a_epoch(int slot) {
    if (slot >= 6 && slot <= 9) return "gate-a:step:epoch0";
    if (slot == 12) return "gate-a:anchor:positive";
    if (slot == 13) return "gate-a:anchor:negative";
    return NULL;
}

static uint64_t gate_a_epoch_origin(int slot, uint64_t session_origin,
                                    double slot_s, double tsc_hz) {
    int epoch_slot = slot;
    if (slot >= 6 && slot <= 9) epoch_slot = 6;
    return session_origin + (uint64_t)(epoch_slot * slot_s * tsc_hz);
}

static int gate_a_cycle_state(int slot, uint64_t now, uint64_t session_origin,
                              double slot_s, double tsc_hz) {
    uint64_t epoch_origin = gate_a_epoch_origin(slot, session_origin,
                                                slot_s, tsc_hz);
    double step_ticks = tsc_hz / (8.0 * tone(0));
    int phase = gate_a_phase_index(slot);
    double offset = (double)(now - epoch_origin) - phase * step_ticks;
    long state = (long)floor(offset / step_ticks);
    return (int)((state % 8 + 8) % 8);
}

typedef struct {
    atomic_int stop;
    atomic_int ready;
    atomic_ullong origin;
    atomic_ullong transition_tsc[16];
    pthread_t thread;
    int alive;
    int core;
    double tsc_hz;
    double slot_s;
} GateASender;

static void *gate_a_sender_loop(void *opaque) {
    GateASender *sender = opaque;
    if (pin_core(sender->core)) return (void *)1;
    atomic_store_explicit(&sender->ready, 1, memory_order_release);
    uint64_t origin = 0;
    while (!(origin = atomic_load_explicit(&sender->origin, memory_order_acquire)) &&
           !atomic_load_explicit(&sender->stop, memory_order_acquire)) {
        __asm__ volatile("pause");
    }
    if (atomic_load_explicit(&sender->stop, memory_order_acquire)) return NULL;
    while (rdtsc_now() < origin &&
           !atomic_load_explicit(&sender->stop, memory_order_acquire)) {
        __asm__ volatile("pause");
    }

    uint64_t end = origin + (uint64_t)(16.0 * sender->slot_s * sender->tsc_hz);
    uint64_t seed = 0x9e3779b9u + (uint64_t)sender->core;
    volatile double sink = 0;
    int previous_slot = -1;
    while (!atomic_load_explicit(&sender->stop, memory_order_acquire)) {
        uint64_t now = rdtsc_now();
        if (now >= end) break;
        int slot = (int)((double)(now - origin) /
                         (sender->slot_s * sender->tsc_hz));
        if (slot < 0 || slot >= 16) {
            __asm__ volatile("pause");
            continue;
        }
        if (slot != previous_slot) {
            unsigned long long expected = 0;
            atomic_compare_exchange_strong_explicit(
                &sender->transition_tsc[slot], &expected, now,
                memory_order_release, memory_order_relaxed);
            previous_slot = slot;
        }
        if (!gate_a_driven_slot(slot)) {
            __asm__ volatile("pause");
            continue;
        }
        int cycle_state = gate_a_cycle_state(slot, now, origin,
                                             sender->slot_s, sender->tsc_hz);
        if (cycle_state < 2) {
            sink += alu_burst(&seed);
        } else {
            __asm__ volatile("pause");
        }
    }
    return (void *)(uintptr_t)(sink != sink);
}

static int gate_a_sender_arm(GateASender *sender, int core, double tsc_hz,
                             double slot_s) {
    memset(sender, 0, sizeof(*sender));
    sender->core = core;
    sender->tsc_hz = tsc_hz;
    sender->slot_s = slot_s;
    atomic_init(&sender->stop, 0);
    atomic_init(&sender->ready, 0);
    atomic_init(&sender->origin, 0);
    for (int i = 0; i < 16; i++) atomic_init(&sender->transition_tsc[i], 0);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
    int rc = pthread_create(&sender->thread, &attr, gate_a_sender_loop, sender);
    pthread_attr_destroy(&attr);
    if (rc) return -1;
    sender->alive = 1;
    uint64_t timeout = rdtsc_now() + (uint64_t)(0.5 * tsc_hz);
    while (!atomic_load_explicit(&sender->ready, memory_order_acquire)) {
        if (rdtsc_now() > timeout) return -1;
        __asm__ volatile("pause");
    }
    return 0;
}

static int gate_a_sender_stop(GateASender *sender) {
    if (!sender->alive) return 0;
    atomic_store_explicit(&sender->stop, 1, memory_order_release);
    void *result = NULL;
    int rc = pthread_join(sender->thread, &result);
    sender->alive = 0;
    return rc || result ? -1 : 0;
}

static int gate_a_write_failure(const char *dir, const char *reason) {
    char path[CP_PATH_MAX];
    if (joinp(path, sizeof(path), dir, "runtime_failure.json")) return -1;
    FILE *f = fopen(path, "wx");
    if (!f) return -1;
    if (fputs("{\n  \"schema_id\": \"CAT_CAS_PHASE6B6_GATE_A_RUNTIME_FAILURE_V1\",\n  \"reason\": \"", f) < 0 ||
        json_escape(f, reason) ||
        fputs("\",\n  \"partial_evidence_preserved\": true,\n  \"automatic_retry\": false\n}\n", f) < 0) {
        fclose(f);
        unlink(path);
        return -1;
    }
    return close_sync(&f);
}

static int gate_a_write_result(const GateASmokeArgs *args,
                               const GateASmokeResult *result) {
    char path[CP_PATH_MAX];
    if (joinp(path, sizeof(path), args->output_dir, "runtime_result.json")) return -1;
    FILE *f = fopen(path, "wx");
    if (!f) return -1;
    if (fprintf(f,
            "{\n"
            "  \"schema_id\": \"CAT_CAS_PHASE6B6_GATE_A_RUNTIME_RESULT_V1\",\n"
            "  \"status\": \"GATE_A_ENGINEERING_SMOKE_COMPLETE\",\n"
            "  \"authority_sha256\": \"%s\",\n"
            "  \"execution_bundle_sha256\": \"%s\",\n"
            "  \"slot_count\": %d,\n"
            "  \"sample_count\": %d,\n"
            "  \"capture_origin_tsc\": %llu,\n"
            "  \"capture_deadline_tsc\": %llu,\n"
            "  \"capture_first_sample_tsc\": %llu,\n"
            "  \"capture_last_sample_tsc\": %llu,\n"
            "  \"capture_tsc_hz\": %.17g,\n"
            "  \"step_sender_epoch_count\": %d,\n"
            "  \"hardware_executed\": %s,\n"
            "  \"frequency_writes\": 0,\n"
            "  \"voltage_writes\": 0,\n"
            "  \"msr_reads\": 0,\n"
            "  \"msr_writes\": 0,\n"
            "  \"automatic_retry\": false\n"
            "}\n",
            args->authority_sha256, args->execution_bundle_sha256,
            result->slot_count, result->sample_count,
            (unsigned long long)result->capture_origin_tsc,
            (unsigned long long)result->capture_deadline_tsc,
            (unsigned long long)result->capture_first_sample_tsc,
            (unsigned long long)result->capture_last_sample_tsc,
            result->capture_tsc_hz,
            result->step_sender_epoch_count,
            result->hardware_executed ? "true" : "false") < 0) {
        fclose(f);
        unlink(path);
        return -1;
    }
    return close_sync(&f);
}

int run_gate_a_engineering_smoke(const GateASmokeArgs *args,
                                 GateASmokeResult *result) {
    int mock = args && args->backend == BACKEND_MOCK;
    int rc = 0;
    const char *reason = "";
    FILE *raw = NULL, *trace = NULL;
    uint64_t *timestamps = NULL;
    double *observations = NULL;
    GateASender sender = {0};
    char path[CP_PATH_MAX];
    if (!args || !result || !args->output_dir || !args->authority_sha256 ||
        !args->execution_bundle_sha256 || args->sender_core != 4 ||
        args->receiver_core != 5 || args->read_hz != 8000 ||
        args->slot_s != 0.5 || args->temperature_veto_c != 68.0 ||
        args->required_frequency_khz != 1600000) {
        return 2;
    }
    if (!mock && (!gate_a_runtime_authority_sha256 ||
                  !gate_a_runtime_output_root ||
                  strcmp(args->authority_sha256,
                         gate_a_runtime_authority_sha256) ||
                  strcmp(args->output_dir, gate_a_runtime_output_root))) {
        return 2;
    }
    memset(result, 0, sizeof(*result));
    result->slot_count = 16;
    result->step_sender_epoch_count = 1;
    result->hardware_executed = !mock;

    if (mkdir(args->output_dir, 0700)) return 2;
    if (joinp(path, sizeof(path), args->output_dir, "raw_samples.bin")) return 5;
    raw = fopen(path, "wbx");
    if (joinp(path, sizeof(path), args->output_dir, "slot_trace.jsonl")) {
        reason = "PATH_JOIN_FAILURE";
        rc = 5;
        goto cleanup;
    }
    trace = fopen(path, "wx");
    if (!raw || !trace) {
        reason = "EVIDENCE_OPEN_FAILURE";
        rc = 5;
        goto cleanup;
    }

    double tsc_hz = mock ? 3200000000.0 : calibrate_tsc();
    if (!isfinite(tsc_hz) || tsc_hz < 100000000.0) {
        reason = "TSC_CALIBRATION_FAILURE";
        rc = 4;
        goto cleanup;
    }
    uint64_t mapping_origin = 1000000000ULL;
    uint64_t s0e_origin = gate_a_epoch_origin(6, mapping_origin,
                                              args->slot_s, tsc_hz);
    if (gate_a_epoch_origin(7, mapping_origin, args->slot_s, tsc_hz) != s0e_origin ||
        gate_a_cycle_state(6, s0e_origin, mapping_origin, args->slot_s, tsc_hz) != 0 ||
        gate_a_cycle_state(12, gate_a_epoch_origin(12, mapping_origin, args->slot_s, tsc_hz),
                           mapping_origin, args->slot_s, tsc_hz) != 0 ||
        gate_a_cycle_state(13, gate_a_epoch_origin(13, mapping_origin, args->slot_s, tsc_hz),
                           mapping_origin, args->slot_s, tsc_hz) != 4) {
        reason = "PHYSICAL_MAPPING_FAILURE";
        rc = 4;
        goto cleanup;
    }
    if (!mock) {
        if (locate_temp()) {
            reason = "TEMPERATURE_UNOBSERVABLE";
            rc = 3;
            goto cleanup;
        }
        double observed_temp = temperature();
        if (!isfinite(observed_temp) || observed_temp >= args->temperature_veto_c) {
            reason = "THERMAL_VETO";
            rc = 3;
            goto cleanup;
        }
        long sender_khz = cur_khz(args->sender_core);
        long receiver_khz = cur_khz(args->receiver_core);
        if (sender_khz != args->required_frequency_khz ||
            receiver_khz != args->required_frequency_khz) {
            reason = "FREQUENCY_VETO";
            rc = 3;
            goto cleanup;
        }
    }

    int cap = (int)(args->read_hz * args->slot_s * 16.0) + 32;
    timestamps = calloc((size_t)cap, sizeof(*timestamps));
    observations = calloc((size_t)cap, sizeof(*observations));
    if (!timestamps || !observations) {
        reason = "OOM";
        rc = 5;
        goto cleanup;
    }
    uint64_t origin = mock ? 1000000000ULL :
        rdtsc_now() + (uint64_t)(START_GUARD_SECONDS * tsc_hz);
    int count = 0;
    uint64_t receiver_epoch = 0;
    if (mock) {
        count = args->read_hz * 8;
        double spacing = tsc_hz / args->read_hz;
        receiver_epoch = origin;
        for (int i = 0; i < count; i++) {
            timestamps[i] = origin + (uint64_t)(i * spacing);
            observations[i] = 100.0 + (double)(i % 17) * 0.001;
        }
        for (int i = 0; i < 16; i++) {
            atomic_init(&sender.transition_tsc[i],
                        origin + (uint64_t)(i * args->slot_s * tsc_hz));
        }
    } else {
        if (gate_a_sender_arm(&sender, args->sender_core, tsc_hz, args->slot_s)) {
            reason = "SENDER_ARM_FAILURE";
            rc = 4;
            goto cleanup;
        }
        atomic_store_explicit(&sender.origin, origin, memory_order_release);
        count = capture_at_origin(args->receiver_core, args->read_hz, 8.0,
                                  tsc_hz, origin, &receiver_epoch,
                                  timestamps, observations, cap);
        if (gate_a_sender_stop(&sender)) {
            reason = "SENDER_STOP_FAILURE";
            rc = 4;
            goto cleanup;
        }
    }
    uint64_t deadline = origin + (uint64_t)(8.0 * tsc_hz);
    if (count > 0) {
        result->capture_origin_tsc = origin;
        result->capture_deadline_tsc = deadline;
        result->capture_first_sample_tsc = timestamps[0];
        result->capture_last_sample_tsc = timestamps[count - 1];
        result->capture_tsc_hz = tsc_hz;
        for (int i = 0; i < count; i++) {
            if (raw_record(raw, timestamps[i], observations[i])) {
                reason = "RAW_WRITER_FAILURE";
                rc = 5;
                goto cleanup;
            }
            double offset = (double)(timestamps[i] - origin) / tsc_hz;
            int slot = (int)(offset / args->slot_s);
            if (slot >= 0 && slot < 16) result->slot_sample_counts[slot]++;
        }
    }
    if (close_sync(&raw)) {
        reason = "RAW_SYNC_FAILURE";
        rc = 5;
        goto cleanup;
    }
    if (count < (int)(0.9 * args->read_hz * 8.0)) {
        reason = "SHORT_COMPLETE_CAPTURE";
        rc = 5;
        goto cleanup;
    }
    double max_gap = 0;
    for (int i = 1; i < count; i++) {
        double gap = (double)(timestamps[i] - timestamps[i - 1]);
        if (gap > max_gap) max_gap = gap;
    }
    if (!mock) {
        const char *quality = catcas_capture_quality_failure(
            timestamps[0], timestamps[count - 1], (size_t)count,
            origin, deadline, tsc_hz, args->read_hz, tone(0), max_gap);
        if (quality) {
            reason = quality;
            rc = 5;
            goto cleanup;
        }
    }
    for (int slot = 0; slot < 16; slot++) {
        uint64_t requested = origin +
            (uint64_t)(slot * args->slot_s * tsc_hz);
        uint64_t actual = atomic_load_explicit(&sender.transition_tsc[slot],
                                               memory_order_acquire);
        uint64_t skew = actual > requested ? actual - requested : requested - actual;
        if (!actual || (!mock && skew > (uint64_t)(MAX_EPOCH_SKEW_SECONDS * tsc_hz)) ||
            result->slot_sample_counts[slot] < (int)(0.9 * args->read_hz * args->slot_s)) {
            reason = "SLOT_TIMING_OR_CAPTURE_FAILURE";
            rc = 5;
            goto cleanup;
        }
        int driven = gate_a_driven_slot(slot);
        if (fprintf(trace,
                "{\"index\":%d,\"token\":\"%s\",\"requested_start_s\":%.1f,"
                "\"requested_end_s\":%.1f,\"requested_start_tsc\":%llu,"
                "\"actual_transition_tsc\":%llu,\"sample_count\":%d,"
                "\"drive_on\":%s,\"amplitude_level\":",
                slot, gate_a_tokens[slot], slot * args->slot_s,
                (slot + 1) * args->slot_s,
                (unsigned long long)requested, (unsigned long long)actual,
                result->slot_sample_counts[slot], driven ? "true" : "false") < 0) {
            reason = "TRACE_WRITER_FAILURE";
            rc = 5;
            goto cleanup;
        }
        if (driven) {
            if (fprintf(trace,
                    "2,\"phase_index\":%d,\"sign\":%d,\"sender_epoch_id\":\"%s\"}\n",
                    gate_a_phase_index(slot), gate_a_sign(slot),
                    gate_a_epoch(slot)) < 0) {
                reason = "TRACE_WRITER_FAILURE";
                rc = 5;
                goto cleanup;
            }
        } else if (fputs("null,\"phase_index\":null,\"sign\":null,\"sender_epoch_id\":null}\n", trace) < 0) {
            reason = "TRACE_WRITER_FAILURE";
            rc = 5;
            goto cleanup;
        }
        if (fflush(trace) || fsync(fileno(trace))) {
            reason = "TRACE_SYNC_FAILURE";
            rc = 5;
            goto cleanup;
        }
    }
    result->sample_count = count;
    if (close_sync(&trace)) {
        reason = "TRACE_SYNC_FAILURE";
        rc = 5;
        goto cleanup;
    }
    if (!mock) {
        double observed_temp = temperature();
        if (!isfinite(observed_temp) || observed_temp >= args->temperature_veto_c ||
            cur_khz(args->sender_core) != args->required_frequency_khz ||
            cur_khz(args->receiver_core) != args->required_frequency_khz) {
            reason = "POST_CAPTURE_VETO";
            rc = 3;
            goto cleanup;
        }
    }
    if (gate_a_write_result(args, result)) {
        reason = "RESULT_WRITER_FAILURE";
        rc = 5;
    }

cleanup:
    if (sender.alive && gate_a_sender_stop(&sender) && rc == 0) {
        reason = "SENDER_STOP_FAILURE";
        rc = 4;
    }
    if (raw && close_sync(&raw) && rc == 0) {
        reason = "RAW_SYNC_FAILURE";
        rc = 5;
    }
    if (trace && close_sync(&trace) && rc == 0) {
        reason = "TRACE_SYNC_FAILURE";
        rc = 5;
    }
    free(timestamps);
    free(observations);
    if (rc) gate_a_write_failure(args->output_dir, reason);
    (void)receiver_epoch;
    return rc;
}
