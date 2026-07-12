#include "combined_pdn_hardware.c"
#include "captured_file.h"
#include "gate_a_engineering_smoke_runtime.h"

#include <ctype.h>

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

static int gate_a_policy_limits_exact(int core, long required_khz) {
    char min_path[160], max_path[160];
    long min_khz = 0, max_khz = 0;
    snprintf(min_path, sizeof(min_path),
             "/sys/devices/system/cpu/cpufreq/policy%d/scaling_min_freq", core);
    snprintf(max_path, sizeof(max_path),
             "/sys/devices/system/cpu/cpufreq/policy%d/scaling_max_freq", core);
    return !read_long(min_path, &min_khz) && !read_long(max_path, &max_khz) &&
           min_khz == required_khz && max_khz == required_khz;
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

#define GATE_A_JOIN_GUARD_SECONDS 0.002
#define GATE_A_SENDER_START_SKEW_SECONDS 0.050
#define GATE_A_SENDER_JOIN_SKEW_SECONDS 0.050

typedef struct {
    atomic_int stop;
    atomic_int ready;
    atomic_ullong ready_tsc;
    atomic_ullong epoch_start_tsc;
    atomic_ullong first_drive_tsc;
    atomic_ullong thread_exit_tsc;
    atomic_ullong transition_tsc[16];
    pthread_t thread;
    int alive;
    int joined;
    int core;
    int first_slot;
    int end_slot;
    int phase_index;
    int sign;
    const char *epoch_id;
    double tsc_hz;
    double slot_s;
    uint64_t session_origin;
    uint64_t requested_start_tsc;
    uint64_t requested_end_tsc;
    uint64_t planned_stop_tsc;
    uint64_t thread_create_tsc;
    uint64_t stop_requested_tsc;
    uint64_t thread_join_start_tsc;
    uint64_t thread_join_tsc;
} GateASender;

typedef struct {
    atomic_int ready;
    atomic_int done;
    atomic_ullong ready_tsc;
    pthread_t thread;
    int alive;
    int core;
    long read_hz;
    double tsc_hz;
    uint64_t origin;
    int capacity;
    uint64_t *timestamps;
    double *observations;
    uint64_t receiver_epoch;
    int count;
} GateAReceiver;

static int gate_a_json_u64(FILE *f, uint64_t value) {
    if (!value) return fputs("null", f) < 0 ? -1 : 0;
    return fprintf(f, "%llu", (unsigned long long)value) < 0 ? -1 : 0;
}

static int gate_a_lifecycle_record(FILE *f, const char *record_type,
                                   const char *state, int slot,
                                   uint64_t event_tsc,
                                   uint64_t requested_start,
                                   uint64_t requested_end,
                                   const GateASender *sender) {
    if (!f || !record_type || !state || slot < 0 || slot >= 16 || !event_tsc) return -1;
    if (fprintf(f,
            "{\"schema_id\":\"CAT_CAS_PHASE6B6_GATE_A_SENDER_LIFECYCLE_V1\","
            "\"record_type\":\"%s\",\"event_tsc\":%llu,\"slot_index\":%d,"
            "\"token\":\"%s\",\"sender_state\":\"%s\",\"sender_epoch_id\":",
            record_type, (unsigned long long)event_tsc, slot,
            gate_a_tokens[slot], state) < 0) return -1;
    if (sender) {
        if (fprintf(f, "\"%s\",\"phase_index\":%d,\"sign\":%d,",
                    sender->epoch_id, sender->phase_index, sender->sign) < 0) return -1;
    } else if (fputs("null,\"phase_index\":null,\"sign\":null,", f) < 0) return -1;
    if (fputs("\"requested_start_tsc\":", f) < 0 ||
        gate_a_json_u64(f, requested_start) ||
        fputs(",\"requested_end_tsc\":", f) < 0 ||
        gate_a_json_u64(f, requested_end) ||
        fputs(",\"sender_transition_tsc\":", f) < 0 ||
        gate_a_json_u64(
            f,
            sender && !strcmp(record_type, "slot_transition") && !strcmp(state, "active")
                ? atomic_load_explicit(&sender->transition_tsc[slot], memory_order_acquire)
                : 0) ||
        fputs(",\"thread_create_tsc\":", f) < 0 ||
        gate_a_json_u64(f, sender ? sender->thread_create_tsc : 0) ||
        fputs(",\"thread_ready_tsc\":", f) < 0 ||
        gate_a_json_u64(f, sender ? atomic_load_explicit(&sender->ready_tsc, memory_order_acquire) : 0) ||
        fputs(",\"epoch_start_tsc\":", f) < 0 ||
        gate_a_json_u64(f, sender ? atomic_load_explicit(&sender->epoch_start_tsc, memory_order_acquire) : 0) ||
        fputs(",\"first_drive_tsc\":", f) < 0 ||
        gate_a_json_u64(f, sender ? atomic_load_explicit(&sender->first_drive_tsc, memory_order_acquire) : 0) ||
        fputs(",\"stop_requested_tsc\":", f) < 0 ||
        gate_a_json_u64(f, sender ? sender->stop_requested_tsc : 0) ||
        fputs(",\"thread_exit_tsc\":", f) < 0 ||
        gate_a_json_u64(f, sender ? atomic_load_explicit(&sender->thread_exit_tsc, memory_order_acquire) : 0) ||
        fputs(",\"thread_join_start_tsc\":", f) < 0 ||
        gate_a_json_u64(f, sender ? sender->thread_join_start_tsc : 0) ||
        fputs(",\"thread_join_tsc\":", f) < 0 ||
        gate_a_json_u64(f, sender ? sender->thread_join_tsc : 0) ||
        fputs("}\n", f) < 0 || fflush(f) || fsync(fileno(f))) return -1;
    return 0;
}

static int gate_a_epoch_cycle_state(const GateASender *sender, uint64_t now) {
    double step_ticks = sender->tsc_hz / (8.0 * tone(0));
    double offset = (double)(now - sender->requested_start_tsc) -
                    sender->phase_index * step_ticks;
    long state = (long)floor(offset / step_ticks);
    return (int)((state % 8 + 8) % 8);
}

static void *gate_a_sender_loop(void *opaque) {
    GateASender *sender = opaque;
    if (pin_core(sender->core)) {
        atomic_store_explicit(&sender->thread_exit_tsc, rdtsc_now(), memory_order_release);
        return (void *)1;
    }
    uint64_t ready = rdtsc_now();
    atomic_store_explicit(&sender->ready_tsc, ready, memory_order_release);
    atomic_store_explicit(&sender->epoch_start_tsc, ready, memory_order_release);
    atomic_store_explicit(&sender->ready, 1, memory_order_release);
    uint64_t seed = 0x9e3779b9u + (uint64_t)sender->core +
                    (uint64_t)sender->first_slot;
    volatile double sink = 0;
    int previous_slot = -1;
    while (!atomic_load_explicit(&sender->stop, memory_order_acquire)) {
        uint64_t now = rdtsc_now();
        int slot = sender->first_slot +
            (int)((double)(now - sender->requested_start_tsc) /
                  (sender->slot_s * sender->tsc_hz));
        if (slot < sender->first_slot || slot >= sender->end_slot) {
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
        if (gate_a_epoch_cycle_state(sender, now) < 2) {
            unsigned long long expected = 0;
            atomic_compare_exchange_strong_explicit(
                &sender->first_drive_tsc, &expected, now,
                memory_order_release, memory_order_relaxed);
            sink += alu_burst(&seed);
        } else {
            __asm__ volatile("pause");
        }
    }
    atomic_store_explicit(&sender->thread_exit_tsc, rdtsc_now(), memory_order_release);
    return (void *)(uintptr_t)(sink != sink);
}

static int gate_a_sender_abort(GateASender *sender) {
    if (!sender->alive) return 0;
    if (!sender->stop_requested_tsc) sender->stop_requested_tsc = rdtsc_now();
    atomic_store_explicit(&sender->stop, 1, memory_order_release);
    sender->thread_join_start_tsc = rdtsc_now();
    void *result = NULL;
    int rc = pthread_join(sender->thread, &result);
    sender->thread_join_tsc = rdtsc_now();
    sender->alive = 0;
    sender->joined = rc == 0;
    return rc || result ? -1 : 0;
}

static int gate_a_sender_arm(GateASender *sender, FILE *lifecycle,
                             const GateAReceiver *receiver,
                             GateASmokeResult *result,
                             int core, double tsc_hz, double slot_s,
                             uint64_t session_origin, int first_slot,
                             int end_slot, int phase_index, int sign,
                             const char *epoch_id) {
    memset(sender, 0, sizeof(*sender));
    sender->core = core;
    sender->tsc_hz = tsc_hz;
    sender->slot_s = slot_s;
    sender->session_origin = session_origin;
    sender->first_slot = first_slot;
    sender->end_slot = end_slot;
    sender->phase_index = phase_index;
    sender->sign = sign;
    sender->epoch_id = epoch_id;
    sender->requested_start_tsc = session_origin +
        (uint64_t)(first_slot * slot_s * tsc_hz);
    sender->requested_end_tsc = session_origin +
        (uint64_t)(end_slot * slot_s * tsc_hz);
    sender->planned_stop_tsc = sender->requested_end_tsc -
        (uint64_t)(GATE_A_JOIN_GUARD_SECONDS * tsc_hz);
    atomic_init(&sender->stop, 0);
    atomic_init(&sender->ready, 0);
    atomic_init(&sender->ready_tsc, 0);
    atomic_init(&sender->epoch_start_tsc, 0);
    atomic_init(&sender->first_drive_tsc, 0);
    atomic_init(&sender->thread_exit_tsc, 0);
    for (int i = 0; i < 16; i++) atomic_init(&sender->transition_tsc[i], 0);
    sender->thread_create_tsc = rdtsc_now();
    if (sender->thread_create_tsc < sender->requested_start_tsc ||
        gate_a_lifecycle_record(lifecycle, "sender_state", "starting",
                                first_slot, sender->thread_create_tsc,
                                sender->requested_start_tsc,
                                sender->requested_end_tsc, sender)) return -1;
    /* The SSH parent can arrive with a narrow inherited affinity mask.  Create
       first, then use the loop's verified pin_core(core) transition. */
    int create_rc = pthread_create(
        &sender->thread, NULL, gate_a_sender_loop, sender);
    if (!create_rc) {
        sender->alive = 1;
        result->sender_start_count++;
    }
    if (create_rc) {
        if (sender->alive) (void)gate_a_sender_abort(sender);
        (void)gate_a_lifecycle_record(
            lifecycle, "sender_failure", "not_created", first_slot,
            rdtsc_now(), sender->requested_start_tsc,
            sender->requested_end_tsc, sender);
        return -1;
    }
    uint64_t timeout = sender->requested_start_tsc +
        (uint64_t)(GATE_A_SENDER_START_SKEW_SECONDS * tsc_hz);
    while (!atomic_load_explicit(&sender->ready, memory_order_acquire)) {
        if (atomic_load_explicit(&receiver->done, memory_order_acquire) ||
            rdtsc_now() > timeout) {
            (void)gate_a_sender_abort(sender);
            (void)gate_a_lifecycle_record(
                lifecycle, "sender_failure", "joined", first_slot,
                rdtsc_now(), sender->requested_start_tsc,
                sender->requested_end_tsc, sender);
            return -1;
        }
        __asm__ volatile("pause");
    }
    uint64_t ready = atomic_load_explicit(&sender->ready_tsc, memory_order_acquire);
    if (ready < sender->requested_start_tsc || ready > timeout ||
        gate_a_lifecycle_record(lifecycle, "sender_state", "active",
                                first_slot, ready,
                                sender->requested_start_tsc,
                                sender->requested_end_tsc, sender)) {
        (void)gate_a_sender_abort(sender);
        (void)gate_a_lifecycle_record(
            lifecycle, "sender_failure", "joined", first_slot,
            rdtsc_now(), sender->requested_start_tsc,
            sender->requested_end_tsc, sender);
        return -1;
    }
    return 0;
}

static int gate_a_sender_stop_join(GateASender *sender, FILE *lifecycle,
                                   const GateAReceiver *receiver) {
    if (!sender->alive) return -1;
    int capture_stopped = 0;
    while (rdtsc_now() < sender->planned_stop_tsc) {
        if (atomic_load_explicit(&receiver->done, memory_order_acquire)) {
            capture_stopped = 1;
            break;
        }
        __asm__ volatile("pause");
    }
    sender->stop_requested_tsc = rdtsc_now();
    atomic_store_explicit(&sender->stop, 1, memory_order_release);
    if (gate_a_lifecycle_record(lifecycle, "sender_state", "stopping",
                                sender->end_slot - 1,
                                sender->stop_requested_tsc,
                                sender->requested_start_tsc,
                                sender->requested_end_tsc, sender)) return -1;
    sender->thread_join_start_tsc = rdtsc_now();
    void *result = NULL;
    int rc = pthread_join(sender->thread, &result);
    sender->thread_join_tsc = rdtsc_now();
    sender->alive = 0;
    sender->joined = rc == 0;
    uint64_t exited = atomic_load_explicit(&sender->thread_exit_tsc,
                                           memory_order_acquire);
    if (capture_stopped || rc || result || !exited ||
        sender->thread_join_tsc > sender->requested_end_tsc +
            (uint64_t)(GATE_A_SENDER_JOIN_SKEW_SECONDS * sender->tsc_hz) ||
        gate_a_lifecycle_record(lifecycle, "sender_state", "joined",
                                sender->end_slot - 1,
                                sender->thread_join_tsc,
                                sender->requested_start_tsc,
                                sender->requested_end_tsc, sender)) return -1;
    return 0;
}

static void *gate_a_receiver_loop(void *opaque) {
    GateAReceiver *receiver = opaque;
    if (pin_core(receiver->core)) {
        receiver->count = -1;
        atomic_store_explicit(&receiver->done, 1, memory_order_release);
        return (void *)1;
    }
    atomic_store_explicit(&receiver->ready_tsc, rdtsc_now(), memory_order_release);
    atomic_store_explicit(&receiver->ready, 1, memory_order_release);
    receiver->count = capture_at_origin(
        receiver->core, receiver->read_hz, 8.0, receiver->tsc_hz,
        receiver->origin, &receiver->receiver_epoch,
        receiver->timestamps, receiver->observations, receiver->capacity);
    atomic_store_explicit(&receiver->done, 1, memory_order_release);
    return receiver->count < 0 ? (void *)1 : NULL;
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
            "  \"sender_start_count\": %d,\n"
            "  \"receiver_start_count\": %d,\n"
            "  \"temperature_receipt_count\": %d,\n"
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
            result->hardware_executed ? "true" : "false",
            result->sender_start_count,
            result->receiver_start_count,
            result->temperature_receipt_count) < 0) {
        fclose(f);
        unlink(path);
        return -1;
    }
    return close_sync(&f);
}

static int gate_a_mock_epoch(GateASender *sender, FILE *lifecycle,
                             int core, double tsc_hz, double slot_s,
                             uint64_t session_origin, int first_slot,
                             int end_slot, int phase_index, int sign,
                             const char *epoch_id) {
    memset(sender, 0, sizeof(*sender));
    sender->core = core;
    sender->tsc_hz = tsc_hz;
    sender->slot_s = slot_s;
    sender->session_origin = session_origin;
    sender->first_slot = first_slot;
    sender->end_slot = end_slot;
    sender->phase_index = phase_index;
    sender->sign = sign;
    sender->epoch_id = epoch_id;
    sender->requested_start_tsc = session_origin +
        (uint64_t)(first_slot * slot_s * tsc_hz);
    sender->requested_end_tsc = session_origin +
        (uint64_t)(end_slot * slot_s * tsc_hz);
    sender->planned_stop_tsc = sender->requested_end_tsc -
        (uint64_t)(GATE_A_JOIN_GUARD_SECONDS * tsc_hz);
    sender->thread_create_tsc = sender->requested_start_tsc + 1;
    sender->stop_requested_tsc = sender->planned_stop_tsc;
    sender->thread_join_start_tsc = sender->planned_stop_tsc + 2;
    sender->thread_join_tsc = sender->planned_stop_tsc + 4;
    sender->joined = 1;
    atomic_init(&sender->stop, 0);
    atomic_init(&sender->ready, 1);
    atomic_init(&sender->ready_tsc, sender->requested_start_tsc + 2);
    atomic_init(&sender->epoch_start_tsc, sender->requested_start_tsc + 3);
    uint64_t first_drive = sender->requested_start_tsc + 4;
    if (phase_index == 4) first_drive += (uint64_t)(0.5 / tone(0) * tsc_hz);
    atomic_init(&sender->first_drive_tsc, first_drive);
    atomic_init(&sender->thread_exit_tsc, sender->planned_stop_tsc + 3);
    for (int slot = 0; slot < 16; slot++) {
        uint64_t value = slot >= first_slot && slot < end_slot
            ? session_origin + (uint64_t)(slot * slot_s * tsc_hz) + 3 : 0;
        atomic_init(&sender->transition_tsc[slot], value);
    }
    if (gate_a_lifecycle_record(lifecycle, "sender_state", "starting",
                                first_slot, sender->thread_create_tsc,
                                sender->requested_start_tsc,
                                sender->requested_end_tsc, sender) ||
        gate_a_lifecycle_record(lifecycle, "sender_state", "active",
                                first_slot,
                                atomic_load_explicit(&sender->ready_tsc, memory_order_acquire),
                                sender->requested_start_tsc,
                                sender->requested_end_tsc, sender) ||
        gate_a_lifecycle_record(lifecycle, "sender_state", "stopping",
                                end_slot - 1, sender->stop_requested_tsc,
                                sender->requested_start_tsc,
                                sender->requested_end_tsc, sender) ||
        gate_a_lifecycle_record(lifecycle, "sender_state", "joined",
                                end_slot - 1, sender->thread_join_tsc,
                                sender->requested_start_tsc,
                                sender->requested_end_tsc, sender)) return -1;
    return 0;
}

static int gate_a_validate_epoch(const GateASender *sender) {
    uint64_t ready = atomic_load_explicit(&sender->ready_tsc, memory_order_acquire);
    uint64_t started = atomic_load_explicit(&sender->epoch_start_tsc, memory_order_acquire);
    uint64_t first_drive = atomic_load_explicit(&sender->first_drive_tsc, memory_order_acquire);
    uint64_t exited = atomic_load_explicit(&sender->thread_exit_tsc, memory_order_acquire);
    uint64_t skew = (uint64_t)(GATE_A_SENDER_START_SKEW_SECONDS * sender->tsc_hz);
    uint64_t expected_drive_offset = sender->phase_index == 4
        ? (uint64_t)(0.5 / tone(0) * sender->tsc_hz) : 0;
    uint64_t observed_drive_offset = first_drive >= sender->requested_start_tsc
        ? first_drive - sender->requested_start_tsc : UINT64_MAX;
    if (!sender->joined || sender->alive ||
        sender->thread_create_tsc < sender->requested_start_tsc ||
        ready < sender->thread_create_tsc || ready > sender->requested_start_tsc + skew ||
        started < ready || first_drive < started ||
        sender->stop_requested_tsc <= first_drive || exited < sender->stop_requested_tsc ||
        sender->thread_join_start_tsc < sender->stop_requested_tsc ||
        sender->thread_join_tsc < exited ||
        sender->thread_join_tsc < sender->thread_join_start_tsc ||
        sender->thread_join_tsc > sender->requested_end_tsc +
            (uint64_t)(GATE_A_SENDER_JOIN_SKEW_SECONDS * sender->tsc_hz) ||
        observed_drive_offset < expected_drive_offset ||
        observed_drive_offset > expected_drive_offset + skew) return -1;
    for (int slot = sender->first_slot; slot < sender->end_slot; slot++) {
        uint64_t requested = sender->session_origin +
            (uint64_t)(slot * sender->slot_s * sender->tsc_hz);
        uint64_t actual = atomic_load_explicit(&sender->transition_tsc[slot],
                                               memory_order_acquire);
        if (!actual || actual < requested || actual > requested + skew) return -1;
    }
    return 0;
}

static int gate_a_run_real_capture(const GateASmokeArgs *args,
                                   double tsc_hz, uint64_t origin,
                                   uint64_t *timestamps, double *observations,
                                   int capacity, FILE *lifecycle,
                                   GateASender epochs[3],
                                   uint64_t *receiver_epoch,
                                   GateASmokeResult *result) {
    GateAReceiver receiver;
    memset(&receiver, 0, sizeof(receiver));
    receiver.core = args->receiver_core;
    receiver.read_hz = args->read_hz;
    receiver.tsc_hz = tsc_hz;
    receiver.origin = origin;
    receiver.capacity = capacity;
    receiver.timestamps = timestamps;
    receiver.observations = observations;
    atomic_init(&receiver.ready, 0);
    atomic_init(&receiver.done, 0);
    atomic_init(&receiver.ready_tsc, 0);
    receiver.count = -1;
    if (pthread_create(&receiver.thread, NULL, gate_a_receiver_loop, &receiver)) return -1;
    receiver.alive = 1;
    result->receiver_start_count++;
    result->hardware_executed = 1;
    while (!atomic_load_explicit(&receiver.ready, memory_order_acquire)) {
        if (atomic_load_explicit(&receiver.done, memory_order_acquire) ||
            rdtsc_now() >= origin) goto failure;
        __asm__ volatile("pause");
    }
    if (pin_core(0)) goto failure;
    for (int slot = 0; slot < 16; slot++) {
        uint64_t requested = origin + (uint64_t)(slot * args->slot_s * tsc_hz);
        while (rdtsc_now() < requested) {
            if (atomic_load_explicit(&receiver.done, memory_order_acquire)) goto failure;
            __asm__ volatile("pause");
        }
        if (atomic_load_explicit(&receiver.done, memory_order_acquire)) goto failure;
        if (slot == 6 && gate_a_sender_arm(
                &epochs[0], lifecycle, &receiver, result,
                args->sender_core, tsc_hz, args->slot_s,
                origin, 6, 10, 0, 1, "gate-a:step:epoch0")) goto failure;
        if (slot == 9 && gate_a_sender_stop_join(&epochs[0], lifecycle, &receiver)) goto failure;
        if (slot == 12) {
            if (gate_a_sender_arm(
                    &epochs[1], lifecycle, &receiver, result,
                    args->sender_core, tsc_hz, args->slot_s,
                    origin, 12, 13, 0, 1, "gate-a:anchor:positive") ||
                gate_a_sender_stop_join(&epochs[1], lifecycle, &receiver)) goto failure;
        }
        if (slot == 13) {
            if (gate_a_sender_arm(
                    &epochs[2], lifecycle, &receiver, result,
                    args->sender_core, tsc_hz, args->slot_s,
                    origin, 13, 14, 4, -1, "gate-a:anchor:negative") ||
                gate_a_sender_stop_join(&epochs[2], lifecycle, &receiver)) goto failure;
        }
    }
    uint64_t capture_deadline = origin + (uint64_t)(8.0 * tsc_hz);
    while (rdtsc_now() < capture_deadline) {
        if (atomic_load_explicit(&receiver.done, memory_order_acquire)) {
            if (rdtsc_now() < capture_deadline) goto failure;
            break;
        }
        __asm__ volatile("pause");
    }
    {
        void *receiver_result = NULL;
        int join_rc = pthread_join(receiver.thread, &receiver_result);
        receiver.alive = 0;
        if (join_rc || receiver_result) return -1;
    }
    *receiver_epoch = receiver.receiver_epoch;
    return receiver.count;

failure:
    for (int index = 0; index < 3; index++) {
        if (epochs[index].alive) {
            int aborted = gate_a_sender_abort(&epochs[index]);
            if (!aborted) {
                (void)gate_a_lifecycle_record(
                    lifecycle, "sender_state", "joined",
                    epochs[index].end_slot - 1, epochs[index].thread_join_tsc,
                    epochs[index].requested_start_tsc,
                    epochs[index].requested_end_tsc, &epochs[index]);
            }
        }
    }
    if (receiver.alive) {
        void *receiver_result = NULL;
        (void)pthread_join(receiver.thread, &receiver_result);
        receiver.alive = 0;
    }
    return -1;
}

#define GATE_A_TEMPERATURE_MAX_ENTRIES 64
#define GATE_A_TEMPERATURE_ENTRY_NAME_MAX 64
#define GATE_A_TEMPERATURE_TEXT_MAX 128

typedef struct {
    char phase[32];
    char hwmon_root[CP_PATH_MAX];
    int enumerated_hwmon_count;
    int k10temp_candidate_count;
    char selected_hwmon_entry[CP_PATH_MAX];
    char selected_driver_name[GATE_A_TEMPERATURE_TEXT_MAX];
    char selected_temperature_path[CP_PATH_MAX];
    char raw_temperature_text[GATE_A_TEMPERATURE_TEXT_MAX];
    char raw_temperature_sha256[CAPTURED_SHA256_LEN + 1];
    long raw_millidegrees_c;
    double normalized_temperature_c;
    int selected;
    int raw_available;
    int raw_millidegrees_available;
    int normalized_available;
    int observation_complete;
    int veto_passed;
    const char *failure;
    uint64_t observation_tsc;
} GateATemperatureReceipt;

static int gate_a_copy_text(char *destination, size_t capacity,
                            const char *source) {
    size_t length;
    if (!destination || !capacity || !source) return -1;
    length = strlen(source);
    if (length >= capacity) return -1;
    memcpy(destination, source, length + 1);
    return 0;
}

static int gate_a_hwmon_entry_name(const char *name) {
    size_t index;
    if (!name || strncmp(name, "hwmon", 5) || !name[5]) return 0;
    for (index = 5; name[index]; index++) {
        if (!isdigit((unsigned char)name[index])) return 0;
    }
    return 1;
}

static int gate_a_compare_entry_names(const void *left, const void *right) {
    return strcmp((const char *)left, (const char *)right);
}

static int gate_a_same_identity(const struct stat *left,
                                const struct stat *right) {
    return left->st_dev == right->st_dev &&
           left->st_ino == right->st_ino &&
           left->st_mode == right->st_mode;
}

static int gate_a_read_text_once(const char *path, char *output,
                                 size_t capacity, size_t *size_out) {
    int flags = O_RDONLY;
    int descriptor;
    size_t total = 0;
    unsigned char extra;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    if (!path || !output || capacity < 2 || !size_out) return -1;
    descriptor = open(path, flags);
    if (descriptor < 0) return -1;
    while (total < capacity - 1) {
        ssize_t count = read(descriptor, output + total, capacity - 1 - total);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) {
            close(descriptor);
            return -1;
        }
        if (!count) break;
        total += (size_t)count;
    }
    for (;;) {
        ssize_t count = read(descriptor, &extra, 1);
        if (count < 0 && errno == EINTR) continue;
        if (count != 0) {
            close(descriptor);
            return -1;
        }
        break;
    }
    if (close(descriptor)) return -1;
    if (!total) return -1;
    for (size_t index = 0; index < total; index++) {
        unsigned char value = (unsigned char)output[index];
        if (!value || value > 0x7f) return -1;
    }
    output[total] = '\0';
    *size_out = total;
    return 0;
}

static int gate_a_read_stable_text(const char *path, char *output,
                                   size_t capacity, size_t *size_out) {
    struct stat before, middle, after;
    char repeated[GATE_A_TEMPERATURE_TEXT_MAX];
    size_t first_size = 0, second_size = 0;
    if (capacity > sizeof(repeated) ||
        stat(path, &before) || !S_ISREG(before.st_mode) ||
        gate_a_read_text_once(path, output, capacity, &first_size) ||
        stat(path, &middle) ||
        gate_a_read_text_once(path, repeated, capacity, &second_size) ||
        stat(path, &after) ||
        !gate_a_same_identity(&before, &middle) ||
        !gate_a_same_identity(&middle, &after) ||
        first_size != second_size ||
        memcmp(output, repeated, first_size)) {
        return -1;
    }
    *size_out = first_size;
    return 0;
}

static int gate_a_trimmed_text(const char *raw, size_t raw_size,
                               char *output, size_t capacity) {
    size_t first = 0, last = raw_size, length;
    while (first < last && isspace((unsigned char)raw[first])) first++;
    while (last > first && isspace((unsigned char)raw[last - 1])) last--;
    length = last - first;
    if (!length || length >= capacity) return -1;
    memcpy(output, raw + first, length);
    output[length] = '\0';
    return 0;
}

static int gate_a_parse_millidegrees(const char *raw, size_t raw_size,
                                     long *value) {
    char number[GATE_A_TEMPERATURE_TEXT_MAX];
    char *end = NULL;
    size_t index = 0;
    if (gate_a_trimmed_text(raw, raw_size, number, sizeof(number))) return -1;
    if (number[index] == '+' || number[index] == '-') index++;
    if (!number[index]) return -1;
    for (; number[index]; index++) {
        if (!isdigit((unsigned char)number[index])) return -1;
    }
    errno = 0;
    *value = strtol(number, &end, 10);
    return errno || !end || *end ? -1 : 0;
}

static void gate_a_temperature_init(GateATemperatureReceipt *receipt,
                                    const char *root, const char *phase) {
    memset(receipt, 0, sizeof(*receipt));
    if (gate_a_copy_text(receipt->phase, sizeof(receipt->phase), phase) ||
        gate_a_copy_text(receipt->hwmon_root, sizeof(receipt->hwmon_root), root)) {
        receipt->failure = "TEMPERATURE_CONTRACT_INVALID";
    }
}

static int gate_a_observe_temperature(const char *root, const char *phase,
                                      uint64_t observation_tsc,
                                      GateATemperatureReceipt *receipt) {
    DIR *directory = NULL;
    struct dirent *entry;
    char names[GATE_A_TEMPERATURE_MAX_ENTRIES]
              [GATE_A_TEMPERATURE_ENTRY_NAME_MAX];
    int name_count = 0;
    gate_a_temperature_init(receipt, root, phase);
    if (receipt->failure) goto done;
    directory = opendir(root);
    if (!directory) {
        receipt->failure = "HWMON_ROOT_UNOBSERVABLE";
        goto done;
    }
    errno = 0;
    while ((entry = readdir(directory)) != NULL) {
        if (!gate_a_hwmon_entry_name(entry->d_name)) continue;
        if (name_count >= GATE_A_TEMPERATURE_MAX_ENTRIES ||
            gate_a_copy_text(names[name_count], sizeof(names[name_count]),
                             entry->d_name)) {
            receipt->failure = "HWMON_ENUMERATION_LIMIT";
            goto done;
        }
        name_count++;
    }
    if (errno) {
        receipt->failure = "HWMON_ENUMERATION_UNOBSERVABLE";
        goto done;
    }
    if (!name_count) {
        receipt->failure = "HWMON_ENUMERATION_EMPTY";
        goto done;
    }
    qsort(names, (size_t)name_count, sizeof(names[0]),
          gate_a_compare_entry_names);
    receipt->enumerated_hwmon_count = name_count;
    for (int index = 0; index < name_count; index++) {
        char entry_path[CP_PATH_MAX], name_path[CP_PATH_MAX];
        char driver_raw[GATE_A_TEMPERATURE_TEXT_MAX];
        char driver_name[GATE_A_TEMPERATURE_TEXT_MAX];
        size_t driver_size = 0;
        if (joinp(entry_path, sizeof(entry_path), root, names[index]) ||
            joinp(name_path, sizeof(name_path), entry_path, "name") ||
            gate_a_read_stable_text(name_path, driver_raw,
                                    sizeof(driver_raw), &driver_size)) {
            receipt->failure = "DRIVER_NAME_UNOBSERVABLE";
            goto done;
        }
        if (gate_a_trimmed_text(driver_raw, driver_size, driver_name,
                                sizeof(driver_name))) {
            receipt->failure = "DRIVER_NAME_EMPTY";
            goto done;
        }
        if (strcmp(driver_name, GATE_A_TEMPERATURE_DRIVER_NAME)) continue;
        receipt->k10temp_candidate_count++;
        if (receipt->k10temp_candidate_count == 1) {
            if (gate_a_copy_text(receipt->selected_hwmon_entry,
                                 sizeof(receipt->selected_hwmon_entry),
                                 entry_path) ||
                gate_a_copy_text(receipt->selected_driver_name,
                                 sizeof(receipt->selected_driver_name),
                                 driver_name) ||
                joinp(receipt->selected_temperature_path,
                      sizeof(receipt->selected_temperature_path),
                      entry_path, GATE_A_TEMPERATURE_INPUT)) {
                receipt->failure = "TEMPERATURE_PATH_INVALID";
                goto done;
            }
        }
    }
    if (receipt->k10temp_candidate_count != 1) {
        receipt->failure = "K10TEMP_CANDIDATE_COUNT";
        goto done;
    }
    receipt->selected = 1;
    {
        size_t raw_size = 0;
        if (gate_a_read_stable_text(
                receipt->selected_temperature_path,
                receipt->raw_temperature_text,
                sizeof(receipt->raw_temperature_text), &raw_size)) {
            receipt->failure = "TEMPERATURE_INPUT_UNOBSERVABLE";
            goto done;
        }
        receipt->raw_available = 1;
        if (hash_captured(
                (const unsigned char *)receipt->raw_temperature_text,
                raw_size, receipt->raw_temperature_sha256)) {
            receipt->failure = "TEMPERATURE_RAW_HASH_FAILURE";
            goto done;
        }
        if (gate_a_parse_millidegrees(receipt->raw_temperature_text,
                                      raw_size,
                                      &receipt->raw_millidegrees_c)) {
            receipt->failure = "TEMPERATURE_INTEGER_MALFORMED";
            goto done;
        }
    }
    receipt->raw_millidegrees_available = 1;
    if (receipt->raw_millidegrees_c <
            GATE_A_TEMPERATURE_MIN_MILLIDEGREES ||
        receipt->raw_millidegrees_c >
            GATE_A_TEMPERATURE_MAX_MILLIDEGREES) {
        receipt->failure = "TEMPERATURE_IMPLAUSIBLE";
        goto done;
    }
    receipt->normalized_temperature_c =
        (double)receipt->raw_millidegrees_c /
        (double)GATE_A_TEMPERATURE_MILLIDEGREES_PER_C;
    receipt->normalized_available = 1;
    receipt->observation_complete = 1;
    receipt->veto_passed =
        receipt->raw_millidegrees_c <
        GATE_A_TEMPERATURE_VETO_MILLIDEGREES;
    receipt->failure = receipt->veto_passed ? NULL : "TEMPERATURE_VETO";

done:
    if (directory) {
        if (closedir(directory) && !receipt->failure) {
            receipt->observation_complete = 0;
            receipt->veto_passed = 0;
            receipt->failure = "HWMON_ENUMERATION_UNOBSERVABLE";
        }
    }
    receipt->observation_tsc = observation_tsc
        ? observation_tsc : rdtsc_now();
    return receipt->observation_complete && receipt->veto_passed ? 0 : 1;
}

static int gate_a_mock_temperature(long millidegrees, const char *phase,
                                   uint64_t observation_tsc,
                                   GateATemperatureReceipt *receipt) {
    size_t raw_size;
    if (!millidegrees) millidegrees = 42000;
    gate_a_temperature_init(receipt, GATE_A_TEMPERATURE_HWMON_ROOT, phase);
    receipt->enumerated_hwmon_count = 1;
    receipt->k10temp_candidate_count = 1;
    if (joinp(receipt->selected_hwmon_entry,
              sizeof(receipt->selected_hwmon_entry),
              GATE_A_TEMPERATURE_HWMON_ROOT, "hwmon0") ||
        gate_a_copy_text(receipt->selected_driver_name,
                         sizeof(receipt->selected_driver_name),
                         GATE_A_TEMPERATURE_DRIVER_NAME) ||
        joinp(receipt->selected_temperature_path,
              sizeof(receipt->selected_temperature_path),
              receipt->selected_hwmon_entry,
              GATE_A_TEMPERATURE_INPUT)) {
        receipt->failure = "TEMPERATURE_CONTRACT_INVALID";
        receipt->observation_tsc = observation_tsc;
        return 1;
    }
    receipt->selected = 1;
    if (snprintf(receipt->raw_temperature_text,
                 sizeof(receipt->raw_temperature_text), "%ld\n",
                 millidegrees) < 0) {
        receipt->failure = "TEMPERATURE_RAW_FORMAT_FAILURE";
        receipt->observation_tsc = observation_tsc;
        return 1;
    }
    raw_size = strlen(receipt->raw_temperature_text);
    receipt->raw_available = 1;
    if (hash_captured(
            (const unsigned char *)receipt->raw_temperature_text,
            raw_size, receipt->raw_temperature_sha256)) {
        receipt->failure = "TEMPERATURE_RAW_HASH_FAILURE";
        receipt->observation_tsc = observation_tsc;
        return 1;
    }
    receipt->raw_millidegrees_c = millidegrees;
    receipt->raw_millidegrees_available = 1;
    if (millidegrees < GATE_A_TEMPERATURE_MIN_MILLIDEGREES ||
        millidegrees > GATE_A_TEMPERATURE_MAX_MILLIDEGREES) {
        receipt->failure = "TEMPERATURE_IMPLAUSIBLE";
        receipt->observation_tsc = observation_tsc;
        return 1;
    }
    receipt->normalized_temperature_c =
        (double)millidegrees /
        (double)GATE_A_TEMPERATURE_MILLIDEGREES_PER_C;
    receipt->normalized_available = 1;
    receipt->observation_complete = 1;
    receipt->veto_passed =
        millidegrees < GATE_A_TEMPERATURE_VETO_MILLIDEGREES;
    receipt->failure = receipt->veto_passed ? NULL : "TEMPERATURE_VETO";
    receipt->observation_tsc = observation_tsc;
    return receipt->veto_passed ? 0 : 1;
}

static int gate_a_json_string_or_null(FILE *file, const char *value,
                                      int available) {
    if (!available) return fputs("null", file) < 0 ? -1 : 0;
    if (fputc('"', file) == EOF || json_escape(file, value) ||
        fputc('"', file) == EOF) return -1;
    return 0;
}

static int gate_a_write_temperature_receipt(
        FILE *file, const GateATemperatureReceipt *receipt) {
    if (!file || !receipt || !receipt->observation_tsc ||
        fprintf(file,
            "{\"schema_id\":\"%s\",\"phase\":\"%s\","
            "\"hwmon_root\":\"",
            GATE_A_TEMPERATURE_SCHEMA_ID, receipt->phase) < 0 ||
        json_escape(file, receipt->hwmon_root) ||
        fprintf(file,
            "\",\"required_driver_name\":\"%s\","
            "\"required_temperature_input\":\"%s\","
            "\"millidegrees_per_c\":%ld,"
            "\"enumerated_hwmon_count\":%d,"
            "\"k10temp_candidate_count\":%d,"
            "\"selected_hwmon_entry\":",
            GATE_A_TEMPERATURE_DRIVER_NAME, GATE_A_TEMPERATURE_INPUT,
            GATE_A_TEMPERATURE_MILLIDEGREES_PER_C,
            receipt->enumerated_hwmon_count,
            receipt->k10temp_candidate_count) < 0 ||
        gate_a_json_string_or_null(file, receipt->selected_hwmon_entry,
                                   receipt->selected) ||
        fputs(",\"selected_driver_name\":", file) < 0 ||
        gate_a_json_string_or_null(file, receipt->selected_driver_name,
                                   receipt->selected) ||
        fputs(",\"selected_temperature_path\":", file) < 0 ||
        gate_a_json_string_or_null(file,
                                   receipt->selected_temperature_path,
                                   receipt->selected) ||
        fputs(",\"raw_temperature_text\":", file) < 0 ||
        gate_a_json_string_or_null(file, receipt->raw_temperature_text,
                                   receipt->raw_available) ||
        fputs(",\"raw_temperature_sha256\":", file) < 0 ||
        gate_a_json_string_or_null(file,
                                   receipt->raw_temperature_sha256,
                                   receipt->raw_available) ||
        fputs(",\"raw_millidegrees_c\":", file) < 0) {
        return -1;
    }
    if (receipt->raw_millidegrees_available) {
        if (fprintf(file, "%ld", receipt->raw_millidegrees_c) < 0) return -1;
    } else if (fputs("null", file) < 0) {
        return -1;
    }
    if (fputs(",\"normalized_temperature_c\":", file) < 0) return -1;
    if (receipt->normalized_available) {
        if (fprintf(file, "%.17g", receipt->normalized_temperature_c) < 0) {
            return -1;
        }
    } else if (fputs("null", file) < 0) {
        return -1;
    }
    if (fprintf(file,
            ",\"veto_temperature_c\":%.17g,"
            "\"observation_complete\":%s,\"veto_passed\":%s,"
            "\"failure\":",
            (double)GATE_A_TEMPERATURE_VETO_MILLIDEGREES /
                (double)GATE_A_TEMPERATURE_MILLIDEGREES_PER_C,
            receipt->observation_complete ? "true" : "false",
            receipt->veto_passed ? "true" : "false") < 0 ||
        gate_a_json_string_or_null(file, receipt->failure,
                                   receipt->failure != NULL) ||
        fprintf(file, ",\"observation_tsc\":%llu}\n",
                (unsigned long long)receipt->observation_tsc) < 0 ||
        fflush(file) || fsync(fileno(file))) {
        return -1;
    }
    return 0;
}

static int gate_a_retain_temperature(
        FILE *file, const char *phase, int mock, long mock_millidegrees,
        uint64_t observation_tsc, GateATemperatureReceipt *receipt) {
    int observation = mock
        ? gate_a_mock_temperature(mock_millidegrees, phase,
                                  observation_tsc, receipt)
        : gate_a_observe_temperature(GATE_A_TEMPERATURE_HWMON_ROOT,
                                     phase, observation_tsc, receipt);
    if (gate_a_write_temperature_receipt(file, receipt)) return -1;
    return observation;
}

#ifdef GATE_A_NATIVE_TEMPERATURE_TESTING
int gate_a_test_observe_temperature(const char *hwmon_root,
                                    const char *phase,
                                    const char *receipt_path,
                                    uint64_t observation_tsc) {
    GateATemperatureReceipt receipt;
    FILE *file;
    int observation;
    if (!hwmon_root || !phase || !receipt_path ||
        !observation_tsc ||
        (strcmp(phase, "pre_capture") &&
         strcmp(phase, "post_capture"))) return -1;
    file = fopen(receipt_path, "wx");
    if (!file) return -1;
    observation = gate_a_observe_temperature(
        hwmon_root, phase, observation_tsc, &receipt);
    if (gate_a_write_temperature_receipt(file, &receipt) ||
        close_sync(&file)) return -1;
    return observation;
}
#endif

static int gate_a_write_lockin(const GateASmokeArgs *args,
                               const GateASmokeResult *result,
                               const uint64_t *timestamps,
                               const double *observations,
                               const int starts[16], const int ends[16]) {
    char path[CP_PATH_MAX];
    if (joinp(path, sizeof(path), args->output_dir, "LOCKIN_IQ.jsonl")) return -1;
    FILE *f = fopen(path, "wx");
    if (!f) return -1;
    double frequency = tone(0);
    for (int slot = 0; slot < 16; slot++) {
        int count = ends[slot] - starts[slot];
        if (count < 2) {
            fclose(f);
            return -1;
        }
        uint64_t slot_start = result->capture_origin_tsc +
            (uint64_t)(slot * args->slot_s * result->capture_tsc_hz);
        uint64_t slot_end = result->capture_origin_tsc +
            (uint64_t)((slot + 1) * args->slot_s * result->capture_tsc_hz);
        uint64_t analysis_origin = result->capture_origin_tsc +
            (uint64_t)((slot >= 6 && slot <= 9 ? 6 : slot) *
                       args->slot_s * result->capture_tsc_hz);
        double i_value, q_value, magnitude, floor;
        lockin(timestamps + starts[slot], observations + starts[slot], count,
               frequency, analysis_origin, result->capture_tsc_hz,
               &i_value, &q_value, &magnitude, &floor);
        if (!isfinite(i_value) || !isfinite(q_value) ||
            !isfinite(magnitude) || !isfinite(floor) ||
            fprintf(f,
                "{\"schema_id\":\"CAT_CAS_PHASE6B6_GATE_A_LOCKIN_IQ_V1\","
                "\"slot_index\":%d,\"token\":\"%s\","
                "\"raw_sample_start_index\":%d,\"raw_sample_end_index\":%d,"
                "\"sample_count\":%d,\"tone_frequency_hz\":%.17g,"
                "\"lockin_i\":%.17g,\"lockin_q\":%.17g,"
                "\"magnitude\":%.17g,\"off_frequency_floor\":%.17g,"
                "\"origin_tsc\":%llu,\"slot_start_tsc\":%llu,"
                "\"slot_end_tsc\":%llu}\n",
                slot, gate_a_tokens[slot], starts[slot], ends[slot], count,
                frequency, i_value, q_value, magnitude, floor,
                (unsigned long long)analysis_origin,
                (unsigned long long)slot_start,
                (unsigned long long)slot_end) < 0 ||
            fflush(f) || fsync(fileno(f))) {
            fclose(f);
            return -1;
        }
    }
    return close_sync(&f);
}

int run_gate_a_engineering_smoke(const GateASmokeArgs *args,
                                 GateASmokeResult *result) {
    int mock = args && args->backend == BACKEND_MOCK;
    int rc = 0;
    const char *reason = "";
    FILE *raw = NULL, *trace = NULL, *lifecycle = NULL;
    FILE *temperature_receipts = NULL;
    uint64_t *timestamps = NULL;
    double *observations = NULL;
    GateASender epochs[3];
    int starts[16], ends[16];
    memset(epochs, 0, sizeof(epochs));
    for (int slot = 0; slot < 16; slot++) {
        starts[slot] = -1;
        ends[slot] = -1;
    }
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
    if (mkdir(args->output_dir, 0700)) return 2;
    if (joinp(path, sizeof(path), args->output_dir,
              GATE_A_TEMPERATURE_RECEIPT_FILE)) {
        reason = "PATH_JOIN_FAILURE";
        rc = 5;
        goto cleanup;
    }
    temperature_receipts = fopen(path, "wx");
    if (joinp(path, sizeof(path), args->output_dir, "raw_samples.bin")) {
        reason = "PATH_JOIN_FAILURE";
        rc = 5;
        goto cleanup;
    }
    raw = fopen(path, "wbx");
    if (joinp(path, sizeof(path), args->output_dir, "slot_trace.jsonl")) {
        reason = "PATH_JOIN_FAILURE";
        rc = 5;
        goto cleanup;
    }
    trace = fopen(path, "wx");
    if (joinp(path, sizeof(path), args->output_dir, "SENDER_LIFECYCLE.jsonl")) {
        reason = "PATH_JOIN_FAILURE";
        rc = 5;
        goto cleanup;
    }
    lifecycle = fopen(path, "wx");
    if (!temperature_receipts || !raw || !trace || !lifecycle) {
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
    {
        GateATemperatureReceipt pre_temperature;
        int pre_status = gate_a_retain_temperature(
            temperature_receipts, "pre_capture", mock,
            args->mock_pre_temperature_millidegrees,
            mock ? 900000000ULL : 0, &pre_temperature);
        if (pre_status < 0) {
            reason = "TEMPERATURE_RECEIPT_WRITE_FAILURE";
            rc = 5;
            goto cleanup;
        }
        result->temperature_receipt_count++;
        if (pre_status > 0) {
            reason = pre_temperature.failure
                ? pre_temperature.failure : "TEMPERATURE_UNOBSERVABLE";
            rc = 3;
            goto cleanup;
        }
    }
    if (!mock) {
        if (!gate_a_policy_limits_exact(args->sender_core,
                                        args->required_frequency_khz) ||
            !gate_a_policy_limits_exact(args->receiver_core,
                                        args->required_frequency_khz)) {
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
        if (gate_a_mock_epoch(&epochs[0], lifecycle, args->sender_core,
                              tsc_hz, args->slot_s, origin, 6, 10, 0, 1,
                              "gate-a:step:epoch0") ||
            gate_a_mock_epoch(&epochs[1], lifecycle, args->sender_core,
                              tsc_hz, args->slot_s, origin, 12, 13, 0, 1,
                              "gate-a:anchor:positive") ||
            gate_a_mock_epoch(&epochs[2], lifecycle, args->sender_core,
                              tsc_hz, args->slot_s, origin, 13, 14, 4, -1,
                              "gate-a:anchor:negative")) {
            reason = "MOCK_LIFECYCLE_EVIDENCE_FAILURE";
            rc = 4;
            goto cleanup;
        }
        receiver_epoch = origin;
        for (int i = 0; i < count; i++) {
            timestamps[i] = origin + (uint64_t)(i * spacing);
            observations[i] = 100.0 + (double)(i % 17) * 0.001;
            int slot = (int)((double)(timestamps[i] - origin) /
                             (args->slot_s * tsc_hz));
            if (slot >= 0 && slot < 16 && gate_a_driven_slot(slot)) {
                uint64_t epoch_origin = gate_a_epoch_origin(
                    slot, origin, args->slot_s, tsc_hz);
                double phase = slot == 13 ? M_PI : 0.0;
                double seconds = (double)(timestamps[i] - epoch_origin) / tsc_hz;
                observations[i] += 0.25 * cos(2.0 * M_PI * tone(0) * seconds + phase);
            }
        }
    } else {
        count = gate_a_run_real_capture(
            args, tsc_hz, origin, timestamps, observations, cap,
            lifecycle, epochs, &receiver_epoch, result);
        if (count < 0) {
            reason = "SENDER_OR_RECEIVER_LIFECYCLE_FAILURE";
            rc = 4;
            goto cleanup;
        }
    }
    for (int epoch = 0; epoch < 3; epoch++) {
        if (gate_a_validate_epoch(&epochs[epoch])) {
            reason = "SENDER_EPOCH_CUSTODY_FAILURE";
            rc = 4;
            goto cleanup;
        }
    }
    uint64_t deadline = origin + (uint64_t)(8.0 * tsc_hz);
    while (count > 0 && timestamps[count - 1] >= deadline) count--;
    if (count > 0 && timestamps[0] < origin) {
        reason = "CAPTURE_PRECEDES_ORIGIN";
        rc = 5;
        goto cleanup;
    }
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
            if (slot >= 0 && slot < 16) {
                if (starts[slot] < 0) starts[slot] = i;
                ends[slot] = i + 1;
                result->slot_sample_counts[slot]++;
            }
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
    for (int slot = 0; slot < 16; slot++) {
        if (starts[slot] < 0 || ends[slot] <= starts[slot] ||
            (slot == 0 ? starts[slot] != 0 : starts[slot] != ends[slot - 1])) {
            reason = "RAW_SLOT_RANGE_FAILURE";
            rc = 5;
            goto cleanup;
        }
    }
    if (ends[15] != count || gate_a_write_lockin(
            args, result, timestamps, observations, starts, ends)) {
        reason = "LOCKIN_CUSTODY_FAILURE";
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
        uint64_t actual = timestamps[starts[slot]];
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
        GateASender *slot_sender = NULL;
        const char *state = "not_created";
        if (slot >= 6 && slot <= 9) {
            slot_sender = &epochs[0];
            state = "active";
        } else if (slot == 12) {
            slot_sender = &epochs[1];
            state = "active";
        } else if (slot == 13) {
            slot_sender = &epochs[2];
            state = "active";
        } else if (slot == 10 || slot == 11) {
            slot_sender = &epochs[0];
            state = "joined";
        } else if (slot == 14 || slot == 15) {
            slot_sender = &epochs[2];
            state = "joined";
        }
        if (gate_a_lifecycle_record(
                lifecycle, "slot_transition", state, slot, actual,
                requested, origin +
                    (uint64_t)((slot + 1) * args->slot_s * tsc_hz),
                slot_sender)) {
            reason = "LIFECYCLE_WRITER_FAILURE";
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
    if (close_sync(&lifecycle)) {
        reason = "LIFECYCLE_SYNC_FAILURE";
        rc = 5;
        goto cleanup;
    }
    {
        GateATemperatureReceipt post_temperature;
        int post_status = gate_a_retain_temperature(
            temperature_receipts, "post_capture", mock,
            args->mock_post_temperature_millidegrees,
            mock ? 26600000001ULL : 0, &post_temperature);
        if (post_status < 0) {
            reason = "TEMPERATURE_RECEIPT_WRITE_FAILURE";
            rc = 5;
            goto cleanup;
        }
        result->temperature_receipt_count++;
        if (post_status > 0) {
            reason = post_temperature.failure
                ? post_temperature.failure : "TEMPERATURE_UNOBSERVABLE";
            rc = 3;
            goto cleanup;
        }
    }
    if (!mock &&
        (!gate_a_policy_limits_exact(args->sender_core,
                                     args->required_frequency_khz) ||
         !gate_a_policy_limits_exact(args->receiver_core,
                                     args->required_frequency_khz))) {
        reason = "POST_CAPTURE_FREQUENCY_VETO";
        rc = 3;
        goto cleanup;
    }
    if (close_sync(&temperature_receipts)) {
        reason = "TEMPERATURE_RECEIPT_SYNC_FAILURE";
        rc = 5;
        goto cleanup;
    }
    if (gate_a_write_result(args, result)) {
        reason = "RESULT_WRITER_FAILURE";
        rc = 5;
    }

cleanup:
    for (int epoch = 0; epoch < 3; epoch++) {
        if (epochs[epoch].alive && gate_a_sender_abort(&epochs[epoch]) && rc == 0) {
            reason = "SENDER_ABORT_JOIN_FAILURE";
            rc = 4;
        }
    }
    if (raw && close_sync(&raw) && rc == 0) {
        reason = "RAW_SYNC_FAILURE";
        rc = 5;
    }
    if (trace && close_sync(&trace) && rc == 0) {
        reason = "TRACE_SYNC_FAILURE";
        rc = 5;
    }
    if (lifecycle && close_sync(&lifecycle) && rc == 0) {
        reason = "LIFECYCLE_SYNC_FAILURE";
        rc = 5;
    }
    if (temperature_receipts &&
        close_sync(&temperature_receipts) && rc == 0) {
        reason = "TEMPERATURE_RECEIPT_SYNC_FAILURE";
        rc = 5;
    }
    free(timestamps);
    free(observations);
    if (rc) gate_a_write_failure(args->output_dir, reason);
    (void)receiver_epoch;
    return rc;
}
