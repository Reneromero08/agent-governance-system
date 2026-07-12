#include "combined_pdn_hardware.c"
#include "captured_file.h"
#include "small_wall_runtime.h"

#include <ctype.h>

#if defined(__GNUC__)
#define SMALL_WALL_CRITICAL __attribute__((noinline, noclone))
#else
#define SMALL_WALL_CRITICAL
#endif

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

static const char *readonly_micro_tokens[GATE_A_READONLY_MICRO_SLOT_COUNT] = {
    "I", "I", "F0", "F1", "F1", "F0", "I", "I"
};

static const char *coded_preprojection_tokens[GATE_A_CODED_PREPROJECTION_SLOT_COUNT] = {
    "N0", "SO", "P0", "P1", "P2", "P3", "M0", "M1",
    "M2", "M3", "C0", "C1", "C2", "C3", "N1", "SO"
};

static const char *coded_preprojection_restored_tokens[GATE_A_CODED_PREPROJECTION_SLOT_COUNT] = {
    "WU", "N0", "P0", "P1", "P2", "P3", "M0", "M1",
    "M2", "M3", "C0", "C1", "C2", "C3", "N1", "T"
};

static const char *coded_preprojection_warm_restored_tokens[GATE_A_CODED_PREPROJECTION_SLOT_COUNT] = {
    "WU", "WU", "N0", "P0", "P1", "P2", "P3", "M0",
    "M1", "M2", "M3", "C0", "C1", "C2", "C3", "N1"
};

static const char *coded_preprojection_warm_query_scramble_tokens[GATE_A_CODED_PREPROJECTION_SLOT_COUNT] = {
    "WU", "WU", "N0", "QS0", "QS1", "QS2", "QS3", "QM0",
    "QM1", "QM2", "QM3", "C0", "C1", "C2", "C3", "N1"
};

static const char *coded_preprojection_warm_query_off_tokens[GATE_A_CODED_PREPROJECTION_SLOT_COUNT] = {
    "WU", "WU", "N0", "QO0", "QO1", "QO2", "QO3", "QO4",
    "QO5", "QO6", "QO7", "C0", "C1", "C2", "C3", "N1"
};

static const char *coded_preprojection_warm_phase_local_sham_tokens[GATE_A_CODED_PREPROJECTION_SLOT_COUNT] = {
    "WU", "WU", "N0", "P0", "C0", "M0", "M1", "C1",
    "P1", "P2", "C2", "M2", "M3", "C3", "P3", "N1"
};

static int gate_a_pilot_variant = GATE_A_PILOT_PN;

static int gate_a_coded_preprojection_pilot(void);
static int gate_a_coded_preprojection_query_scramble_pilot(void);
static int gate_a_coded_preprojection_query_off_pilot(void);
static int gate_a_coded_preprojection_declaration_sham_pilot(void);
static int gate_a_coded_preprojection_phase_local_sham_pilot(void);
static int gate_a_coded_preprojection_phase_local_pilot(void);
static int gate_a_coded_preprojection_stimulus_first_slot(void);
static int gate_a_coded_preprojection_stimulus_end_slot(void);
static int gate_a_readonly_occupancy_pilot(void);

/* Closed CAT_CAS-owned shared-cache response geometry.  These buffers contain
   only synthetic experiment data.  The experiment uses no address discovery,
   cache-set mapping, flush instruction, or observation outside this runtime. */
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
    if (gate_a_coded_preprojection_pilot()) {
        return slot >= gate_a_coded_preprojection_stimulus_first_slot() &&
               slot < gate_a_coded_preprojection_stimulus_end_slot();
    }
    if (gate_a_readonly_occupancy_pilot()) {
        return slot >= 2 && slot <= 5;
    }
    if (slot >= 6 && slot <= 9) {
        if (gate_a_pilot_variant == GATE_A_PILOT_STEP_SHAM) return 0;
        if (gate_a_pilot_variant == GATE_A_PILOT_IMPULSE) return slot == 6;
        return 1;
    }
    if (slot == 12 || slot == 13) {
        return gate_a_pilot_variant != GATE_A_PILOT_ANCHOR_SHAM;
    }
    return 0;
}

static int gate_a_step_end_slot(void) {
    if (gate_a_coded_preprojection_pilot()) {
        return gate_a_coded_preprojection_stimulus_end_slot();
    }
    if (gate_a_readonly_occupancy_pilot()) return 6;
    return gate_a_pilot_variant == GATE_A_PILOT_IMPULSE ? 7 : 10;
}

static int gate_a_split_step(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_PHASE_FORWARD ||
           gate_a_pilot_variant == GATE_A_PILOT_PHASE_REVERSE;
}

static int gate_a_value_pilot(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_VALUE_FORWARD ||
           gate_a_pilot_variant == GATE_A_PILOT_VALUE_REVERSE ||
           gate_a_pilot_variant == GATE_A_PILOT_VALUE_EQUAL;
}

static int gate_a_occupancy_pilot(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_OCCUPANCY_FORWARD ||
           gate_a_pilot_variant == GATE_A_PILOT_OCCUPANCY_REVERSE ||
           gate_a_pilot_variant == GATE_A_PILOT_OCCUPANCY_EQUAL ||
           gate_a_pilot_variant == GATE_A_PILOT_READONLY_OCCUPANCY_FORWARD ||
           gate_a_pilot_variant == GATE_A_PILOT_READONLY_OCCUPANCY_REVERSE ||
           gate_a_pilot_variant == GATE_A_PILOT_READONLY_OCCUPANCY_EQUAL ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_RESTORED_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_RESTORED_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_QUERY_SCRAMBLE_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_QUERY_OFF_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_DECLARATION_SHAM_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_SHAM_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_LOOP;
}

static int gate_a_coded_preprojection_pilot(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_RESTORED_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_RESTORED_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_QUERY_SCRAMBLE_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_QUERY_OFF_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_DECLARATION_SHAM_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_SHAM_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_LOOP;
}

static int gate_a_coded_preprojection_query_scramble_pilot(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_QUERY_SCRAMBLE_LOOP;
}

static int gate_a_coded_preprojection_query_off_pilot(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_QUERY_OFF_LOOP;
}

static int gate_a_coded_preprojection_declaration_sham_pilot(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_DECLARATION_SHAM_LOOP;
}

static int gate_a_coded_preprojection_phase_local_sham_pilot(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_SHAM_LOOP;
}

static int gate_a_coded_preprojection_phase_local_pilot(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_SHAM_LOOP ||
           gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_LOOP;
}

static int gate_a_coded_preprojection_warm_restored_pilot(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_RESTORED_LOOP ||
           gate_a_coded_preprojection_query_scramble_pilot() ||
           gate_a_coded_preprojection_query_off_pilot() ||
           gate_a_coded_preprojection_declaration_sham_pilot() ||
           gate_a_coded_preprojection_phase_local_pilot();
}

static int gate_a_coded_preprojection_stimulus_first_slot(void) {
    return gate_a_coded_preprojection_warm_restored_pilot() ? 3 : 2;
}

static int gate_a_coded_preprojection_stimulus_end_slot(void) {
    return gate_a_coded_preprojection_warm_restored_pilot() ? 15 : 14;
}

static int gate_a_readonly_occupancy_pilot(void) {
    return gate_a_pilot_variant == GATE_A_PILOT_READONLY_OCCUPANCY_FORWARD ||
           gate_a_pilot_variant == GATE_A_PILOT_READONLY_OCCUPANCY_REVERSE ||
           gate_a_pilot_variant == GATE_A_PILOT_READONLY_OCCUPANCY_EQUAL ||
           gate_a_coded_preprojection_pilot();
}

static int gate_a_variant_is_readonly_occupancy(int variant) {
    return variant == GATE_A_PILOT_READONLY_OCCUPANCY_FORWARD ||
           variant == GATE_A_PILOT_READONLY_OCCUPANCY_REVERSE ||
           variant == GATE_A_PILOT_READONLY_OCCUPANCY_EQUAL ||
           variant == GATE_A_PILOT_CODED_PREPROJECTION_LOOP ||
           variant == GATE_A_PILOT_CODED_PREPROJECTION_RESTORED_LOOP ||
           variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_RESTORED_LOOP ||
           variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_QUERY_SCRAMBLE_LOOP ||
           variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_QUERY_OFF_LOOP ||
           variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_DECLARATION_SHAM_LOOP ||
           variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_SHAM_LOOP ||
           variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_LOOP;
}

static int gate_a_slot_count(void) {
    if (gate_a_coded_preprojection_pilot()) {
        return GATE_A_CODED_PREPROJECTION_SLOT_COUNT;
    }
    return gate_a_readonly_occupancy_pilot()
        ? GATE_A_READONLY_MICRO_SLOT_COUNT : GATE_A_LEGACY_SLOT_COUNT;
}

static double gate_a_duration_s(const GateASmokeArgs *args) {
    return (double)gate_a_slot_count() * args->slot_s;
}

static const char *gate_a_slot_token(int slot) {
    if (gate_a_coded_preprojection_pilot()) {
        return slot >= 0 && slot < GATE_A_CODED_PREPROJECTION_SLOT_COUNT
            ? (gate_a_coded_preprojection_query_scramble_pilot()
                ? coded_preprojection_warm_query_scramble_tokens[slot]
                : (gate_a_coded_preprojection_query_off_pilot()
                    ? coded_preprojection_warm_query_off_tokens[slot]
                : (gate_a_coded_preprojection_phase_local_pilot()
                    ? coded_preprojection_warm_phase_local_sham_tokens[slot]
                : (gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_WARM_RESTORED_LOOP ||
                   gate_a_coded_preprojection_declaration_sham_pilot()
                    ? coded_preprojection_warm_restored_tokens[slot]
                : (gate_a_pilot_variant == GATE_A_PILOT_CODED_PREPROJECTION_RESTORED_LOOP
                    ? coded_preprojection_restored_tokens[slot]
                    : coded_preprojection_tokens[slot])))))
            : "OUT_OF_RANGE";
    }
    if (gate_a_readonly_occupancy_pilot()) {
        return slot >= 0 && slot < GATE_A_READONLY_MICRO_SLOT_COUNT
            ? readonly_micro_tokens[slot] : "OUT_OF_RANGE";
    }
    return slot >= 0 && slot < GATE_A_LEGACY_SLOT_COUNT
        ? gate_a_tokens[slot] : "OUT_OF_RANGE";
}

static int gate_a_readonly_stimulus_first_slot(void) {
    if (gate_a_coded_preprojection_pilot()) {
        return gate_a_coded_preprojection_stimulus_first_slot();
    }
    return gate_a_readonly_occupancy_pilot() ? 2 : 6;
}

static int gate_a_readonly_stimulus_end_slot(void) {
    if (gate_a_coded_preprojection_pilot()) {
        return gate_a_coded_preprojection_stimulus_end_slot();
    }
    return gate_a_readonly_occupancy_pilot() ? 6 : 10;
}

static size_t gate_a_occupancy_bytes(int slot) {
    if (!gate_a_occupancy_pilot()) return 0;
    if (gate_a_coded_preprojection_pilot()) {
        int first_slot = gate_a_coded_preprojection_stimulus_first_slot();
        int relative = slot - first_slot;
        if (relative < 0 || relative >= 12) return 0;
        if (gate_a_coded_preprojection_phase_local_pilot()) {
            const char *token = gate_a_slot_token(slot);
            if (gate_a_coded_preprojection_phase_local_sham_pilot()) {
                return GATE_A_OCCUPANCY_EQUAL_BYTES;
            }
            if (token[0] == 'C') return GATE_A_OCCUPANCY_EQUAL_BYTES;
            if (token[0] == 'P') {
                return token[1] == '0' || token[1] == '1'
                    ? GATE_A_OCCUPANCY_LARGE_BYTES : GATE_A_OCCUPANCY_SMALL_BYTES;
            }
            if (token[0] == 'M') {
                return token[1] == '0' || token[1] == '3'
                    ? GATE_A_OCCUPANCY_LARGE_BYTES : GATE_A_OCCUPANCY_SMALL_BYTES;
            }
            return GATE_A_OCCUPANCY_EQUAL_BYTES;
        }
        if (relative >= 8) return GATE_A_OCCUPANCY_EQUAL_BYTES;
        if (gate_a_coded_preprojection_query_off_pilot() ||
            gate_a_coded_preprojection_declaration_sham_pilot()) {
            return GATE_A_OCCUPANCY_EQUAL_BYTES;
        }
        if (gate_a_coded_preprojection_query_scramble_pilot()) {
            return relative == 0 || relative == 2 ||
                   relative == 5 || relative == 7
                ? GATE_A_OCCUPANCY_LARGE_BYTES
                : GATE_A_OCCUPANCY_SMALL_BYTES;
        }
        if (relative == 0 || relative == 1 || relative == 4 || relative == 7) {
            return GATE_A_OCCUPANCY_LARGE_BYTES;
        }
        return GATE_A_OCCUPANCY_SMALL_BYTES;
    }
    if (gate_a_readonly_occupancy_pilot()) {
        if (slot < 2 || slot > 5) return 0;
        if (gate_a_pilot_variant == GATE_A_PILOT_READONLY_OCCUPANCY_EQUAL) {
            return GATE_A_OCCUPANCY_EQUAL_BYTES;
        }
        if (gate_a_pilot_variant == GATE_A_PILOT_READONLY_OCCUPANCY_FORWARD) {
            return slot == 2 || slot == 5
                ? GATE_A_OCCUPANCY_SMALL_BYTES : GATE_A_OCCUPANCY_LARGE_BYTES;
        }
        return slot == 2 || slot == 5
            ? GATE_A_OCCUPANCY_LARGE_BYTES : GATE_A_OCCUPANCY_SMALL_BYTES;
    }
    if (slot < 6 || slot > 9) return 0;
    if (gate_a_pilot_variant == GATE_A_PILOT_OCCUPANCY_EQUAL ||
        gate_a_pilot_variant == GATE_A_PILOT_READONLY_OCCUPANCY_EQUAL) {
        return GATE_A_OCCUPANCY_EQUAL_BYTES;
    }
    if (gate_a_pilot_variant == GATE_A_PILOT_OCCUPANCY_FORWARD ||
        gate_a_pilot_variant == GATE_A_PILOT_READONLY_OCCUPANCY_FORWARD) {
        return slot == 6 || slot == 9
            ? GATE_A_OCCUPANCY_SMALL_BYTES : GATE_A_OCCUPANCY_LARGE_BYTES;
    }
    return slot == 6 || slot == 9
        ? GATE_A_OCCUPANCY_LARGE_BYTES : GATE_A_OCCUPANCY_SMALL_BYTES;
}

static const char *gate_a_measurement_mode(void) {
    if (gate_a_coded_preprojection_pilot()) {
        return "catcas_coded_preprojection_response_cycles";
    }
    if (gate_a_readonly_occupancy_pilot()) {
        return "catcas_readonly_occupancy_response_cycles";
    }
    return gate_a_occupancy_pilot()
        ? "catcas_owned_cache_response_cycles"
        : "ring_period_cycles";
}

static const char *gate_a_observation_kind(void) {
    if (gate_a_coded_preprojection_pilot()) {
        return "experiment_owned_coded_preprojection_buffer_cycles_per_access";
    }
    if (gate_a_readonly_occupancy_pilot()) {
        return "experiment_owned_readonly_buffer_cycles_per_access";
    }
    return gate_a_occupancy_pilot()
        ? "experiment_owned_buffer_cycles_per_access"
        : "ring_period_cycles_per_iteration";
}

static const char *gate_a_occupancy_classification(void) {
    if (gate_a_coded_preprojection_pilot()) return "CODED_PREPROJECTION";
    if (gate_a_readonly_occupancy_pilot()) return "READONLY_OCCUPANCY";
    if (gate_a_occupancy_pilot()) return "DIRTY_OCCUPANCY_FABRIC_PRESSURE";
    return "NOT_APPLICABLE";
}

static int gate_a_power_of_two(size_t value) {
    return value && !(value & (value - 1U));
}

static int gate_a_occupancy_geometry_valid(void) {
    const size_t footprints[] = {
        GATE_A_OCCUPANCY_SMALL_BYTES,
        GATE_A_OCCUPANCY_EQUAL_BYTES,
        GATE_A_OCCUPANCY_LARGE_BYTES,
        GATE_A_RESPONSE_BUFFER_BYTES,
    };
    for (size_t index = 0; index < sizeof(footprints) / sizeof(footprints[0]); index++) {
        size_t lines = footprints[index] / GATE_A_CACHE_LINE_BYTES;
        if (footprints[index] % GATE_A_CACHE_LINE_BYTES ||
            !gate_a_power_of_two(lines)) return 0;
    }
    if (!(GATE_A_OCCUPANCY_LINE_STRIDE & 1U) ||
        !(GATE_A_RESPONSE_LINE_STRIDE & 1U) ||
        GATE_A_OCCUPANCY_SMALL_BYTES >= GATE_A_OCCUPANCY_EQUAL_BYTES ||
        GATE_A_OCCUPANCY_EQUAL_BYTES >= GATE_A_OCCUPANCY_LARGE_BYTES) return 0;
    return 1;
}

static int small_wall_request_realtime_thread(void) {
    struct sched_param param;
    memset(&param, 0, sizeof(param));
    param.sched_priority = GATE_A_REALTIME_PRIORITY;
    return sched_setscheduler(0, SCHED_FIFO, &param) == 0;
}

static int gate_a_orbit_value(int slot) {
    if (gate_a_readonly_occupancy_pilot()) return -1;
    if (slot < 6 || slot > 9 || !gate_a_value_pilot()) return -1;
    if (gate_a_pilot_variant == GATE_A_PILOT_VALUE_EQUAL) return 42;
    if (gate_a_pilot_variant == GATE_A_PILOT_VALUE_FORWARD) {
        return slot == 6 || slot == 9 ? 125 : 131;
    }
    return slot == 6 || slot == 9 ? 131 : 125;
}

static int gate_a_coded_phase_index_from_token(const char *token) {
    if (!token || strlen(token) != 2U) return 0;
    if (!(token[0] == 'P' || token[0] == 'M' || token[0] == 'C')) return 0;
    if (token[1] < '0' || token[1] > '3') return 0;
    return (token[1] - '0') * 2;
}

static int gate_a_phase_index(int slot) {
    if (gate_a_coded_preprojection_pilot()) {
        if (gate_a_coded_preprojection_phase_local_pilot()) {
            return gate_a_coded_phase_index_from_token(gate_a_slot_token(slot));
        }
        int relative = slot - gate_a_coded_preprojection_stimulus_first_slot();
        if (relative >= 0 && relative < 12) return (relative % 4) * 2;
        return 0;
    }
    if (gate_a_readonly_occupancy_pilot()) return 0;
    if (slot >= 6 && slot <= 9 && gate_a_pilot_variant == GATE_A_PILOT_PHASE_FORWARD) {
        return slot < 8 ? 0 : 2;
    }
    if (slot >= 6 && slot <= 9 && gate_a_pilot_variant == GATE_A_PILOT_PHASE_REVERSE) {
        return slot < 8 ? 2 : 0;
    }
    if (gate_a_pilot_variant == GATE_A_PILOT_NP && slot == 12) return 4;
    if (gate_a_pilot_variant == GATE_A_PILOT_NP && slot == 13) return 0;
    return slot == 13 ? 4 : 0;
}

static int gate_a_sign(int slot) {
    if (gate_a_coded_preprojection_pilot()) {
        return gate_a_phase_index(slot) >= 4 ? -1 : 1;
    }
    return gate_a_phase_index(slot) == 4 ? -1 : 1;
}

static int gate_a_expected_origin_state(int slot) {
    return (8 - gate_a_phase_index(slot)) % 8;
}

static const char *gate_a_epoch(int slot) {
    if (gate_a_coded_preprojection_pilot()) {
        return slot >= gate_a_coded_preprojection_stimulus_first_slot() &&
               slot < gate_a_coded_preprojection_stimulus_end_slot()
            ? "coded-preprojection:loop:epoch0" : NULL;
    }
    if (gate_a_readonly_occupancy_pilot()) {
        return slot >= 2 && slot <= 5 ? "readonly-occupancy:micro:epoch0" : NULL;
    }
    if (slot >= 6 && slot <= 9) {
        if (gate_a_occupancy_pilot()) return "gate-a:occupancy:epoch0";
        if (gate_a_value_pilot()) return "gate-a:value:epoch0";
        if (gate_a_split_step()) {
            return slot < 8 ? "gate-a:phase:first" : "gate-a:phase:second";
        }
        return "gate-a:step:epoch0";
    }
    if (slot == 12 || slot == 13) {
        return gate_a_phase_index(slot) == 4
            ? "gate-a:anchor:negative" : "gate-a:anchor:positive";
    }
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
    if (gate_a_coded_preprojection_pilot() &&
        slot >= gate_a_coded_preprojection_stimulus_first_slot() &&
        slot < gate_a_coded_preprojection_stimulus_end_slot()) {
        epoch_slot = gate_a_coded_preprojection_stimulus_first_slot();
    } else if (gate_a_readonly_occupancy_pilot() && slot >= 2 && slot <= 5) {
        epoch_slot = 2;
    } else if (slot >= 6 && slot <= 9) {
        epoch_slot = gate_a_split_step() && slot >= 8 ? 8 : 6;
    }
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
#define GATE_A_SCHEDULER_REJECT_MULTIPLE 4.0
#define GATE_A_SCHEDULER_MISSED_MULTIPLE 1.0
#define GATE_A_SCHEDULER_MISSED_FRACTION_MAX 0.01
#define GATE_A_SERVICE_SPIKE_MULTIPLE 4.0

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
    int orbit_value;
    uint64_t switching_bank[256];
    volatile uint64_t *occupancy_bank;
    size_t occupancy_cursor;
    size_t occupancy_initial_cursor[16];
    size_t occupancy_final_cursor[16];
    size_t occupancy_touch_count[16];
    int occupancy_slot_complete[16];
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
    int realtime_attempted;
    int realtime_applied;
    uint64_t burst_start_tsc[16];
    uint64_t burst_finish_tsc[16];
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
    double slot_s;
    double duration_s;
    uint64_t origin;
    int capacity;
    uint64_t *timestamps;
    uint64_t *requested_sample_index;
    uint64_t *requested_tsc;
    double *observations;
    const uint64_t *response_bank;
    size_t response_line_count;
    size_t response_index;
    int cache_response_mode;
    uint64_t receiver_epoch;
    int count;
    int realtime_attempted;
    int realtime_applied;
    uint64_t max_sample_delay_tsc;
    double max_response_cycles_per_access;
    int samples_per_slot;
    int slot_count;
    uint64_t *started_tsc;
    uint64_t *finished_tsc;
    int *requested_slot_index;
    uint64_t *scheduler_lateness_ticks;
    uint64_t *service_ticks;
    uint64_t *finish_gap_ticks;
    uint64_t *missed_deadlines_before_sample;
    int *valid_measurement;
    int *timing_slot_index;
    uint64_t skipped_deadline_count;
} GateAReceiver;

typedef struct {
    uint64_t p50_scheduler_lateness_ticks;
    uint64_t p95_scheduler_lateness_ticks;
    uint64_t p99_scheduler_lateness_ticks;
    uint64_t max_scheduler_lateness_ticks;
    uint64_t max_scheduler_lateness_sample_index;
    uint64_t max_scheduler_lateness_slot_index;
    uint64_t p50_service_ticks;
    uint64_t p95_service_ticks;
    uint64_t p99_service_ticks;
    uint64_t max_service_ticks;
    double p50_service_cycles_per_access;
    double p95_service_cycles_per_access;
    double p99_service_cycles_per_access;
    double max_service_cycles_per_access;
    uint64_t missed_deadline_count;
    uint64_t max_finish_gap_ticks;
    uint64_t scheduler_lateness_reject_threshold_ticks;
    int sender_spill;
    int record_integrity_failure;
    int missing_slot;
    int service_spikes;
    const char *classification;
} GateATimingDiagnostic;

static int gate_a_copy_classification(char destination[64],
                                      const char *classification) {
    size_t length = strlen(classification);
    if (length >= 64U) return -1;
    memcpy(destination, classification, length + 1U);
    return 0;
}

static int gate_a_u64_compare(const void *left, const void *right) {
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static uint64_t gate_a_sorted_percentile(const uint64_t *sorted,
                                         int count,
                                         unsigned percentile) {
    if (!sorted || count <= 0) return 0;
    uint64_t numerator = (uint64_t)(count - 1) * percentile + 99U;
    uint64_t index = numerator / 100U;
    if (index >= (uint64_t)count) index = (uint64_t)count - 1U;
    return sorted[index];
}

static int gate_a_write_u64_le(FILE *file, uint64_t value) {
    unsigned char bytes[8];
    for (int index = 0; index < 8; index++) {
        bytes[index] = (unsigned char)((value >> (unsigned)(index * 8)) & 0xffU);
    }
    return fwrite(bytes, 1U, sizeof(bytes), file) == sizeof(bytes) ? 0 : -1;
}

static int gate_a_write_sample_timing_file(
        const char *output_dir,
        const uint64_t *requested_sample_index,
        const uint64_t *requested_tsc,
        const int *requested_slot_index,
        const uint64_t *started_tsc,
        const uint64_t *finished_tsc,
        const int *actual_slot_index,
        const uint64_t *scheduler_lateness_ticks,
        const uint64_t *service_ticks,
        const uint64_t *missed_deadlines_before_sample,
        const int *valid_measurement,
        int count) {
    char path[CP_PATH_MAX];
    if (joinp(path, sizeof(path), output_dir, GATE_A_SAMPLE_TIMING_FILE)) return -1;
    FILE *file = fopen(path, "wx");
    if (!file) return -1;
    for (int sample = 0; sample < count; sample++) {
        if (gate_a_write_u64_le(file, requested_sample_index[sample]) ||
            gate_a_write_u64_le(file, requested_tsc[sample]) ||
            gate_a_write_u64_le(file, (uint64_t)requested_slot_index[sample]) ||
            gate_a_write_u64_le(file, started_tsc[sample]) ||
            gate_a_write_u64_le(file, finished_tsc[sample]) ||
            gate_a_write_u64_le(file, (uint64_t)actual_slot_index[sample]) ||
            gate_a_write_u64_le(file, scheduler_lateness_ticks[sample]) ||
            gate_a_write_u64_le(file, service_ticks[sample]) ||
            gate_a_write_u64_le(file, missed_deadlines_before_sample[sample]) ||
            gate_a_write_u64_le(file, (uint64_t)valid_measurement[sample])) {
            fclose(file);
            unlink(path);
            return -1;
        }
    }
    return close_sync(&file);
}

static void gate_a_retain_sample_timing_summary(
        const GateASmokeArgs *args,
        GateASmokeResult *result,
        const uint64_t *scheduler_lateness_ticks,
        const uint64_t *finish_gap_ticks,
        const uint64_t *missed_deadlines_before_sample,
        int count) {
    if (!args || !result || !scheduler_lateness_ticks || !finish_gap_ticks ||
        !missed_deadlines_before_sample || count <= 0 ||
        !isfinite(result->capture_tsc_hz) || result->capture_tsc_hz <= 0.0 ||
        args->read_hz <= 0) return;
    double nominal_spacing = result->capture_tsc_hz / (double)args->read_hz;
    if (isfinite(nominal_spacing) && nominal_spacing > 0.0 &&
        nominal_spacing <= (double)UINT64_MAX / GATE_A_SCHEDULER_REJECT_MULTIPLE) {
        result->scheduler_lateness_reject_threshold_ticks =
            (uint64_t)(nominal_spacing * GATE_A_SCHEDULER_REJECT_MULTIPLE);
    }
    for (int sample = 0; sample < count; sample++) {
        if (scheduler_lateness_ticks[sample] >
            result->max_scheduler_lateness_ticks) {
            result->max_scheduler_lateness_ticks =
                scheduler_lateness_ticks[sample];
        }
        result->missed_deadline_count +=
            missed_deadlines_before_sample[sample];
        if (finish_gap_ticks[sample] > result->max_finish_gap_ticks) {
            result->max_finish_gap_ticks = finish_gap_ticks[sample];
        }
    }
    result->skipped_deadline_count = result->missed_deadline_count;
}

static void gate_a_synthesize_legacy_sample_timing(
        const GateASmokeArgs *args,
        const uint64_t *timestamps,
        int count,
        uint64_t origin,
        double tsc_hz,
        uint64_t *requested_sample_index,
        uint64_t *requested_tsc,
        int *requested_slot_index,
        uint64_t *started_tsc,
        uint64_t *finished_tsc,
        int *actual_slot_index,
        uint64_t *scheduler_lateness_ticks,
        uint64_t *service_ticks,
        uint64_t *finish_gap_ticks,
        uint64_t *missed_deadlines_before_sample,
        int *valid_measurement) {
    if (!args || !timestamps || count <= 0 || !isfinite(tsc_hz) ||
        tsc_hz <= 0.0 || args->read_hz <= 0 || args->slot_s <= 0.0 ||
        !requested_sample_index || !requested_tsc || !requested_slot_index ||
        !started_tsc || !finished_tsc || !actual_slot_index ||
        !scheduler_lateness_ticks || !service_ticks || !finish_gap_ticks ||
        !missed_deadlines_before_sample || !valid_measurement) return;
    double spacing = tsc_hz / (double)args->read_hz;
    double slot_ticks = args->slot_s * tsc_hz;
    if (!isfinite(spacing) || spacing <= 0.0 ||
        !isfinite(slot_ticks) || slot_ticks <= 0.0) return;
    uint64_t previous_finished = 0;
    for (int sample = 0; sample < count; sample++) {
        uint64_t requested_index = (uint64_t)sample;
        uint64_t requested = origin + (uint64_t)((double)sample * spacing);
        uint64_t started = timestamps[sample];
        requested_sample_index[sample] = requested_index;
        requested_tsc[sample] = requested;
        requested_slot_index[sample] =
            (int)(((double)(requested - origin)) / slot_ticks);
        started_tsc[sample] = started;
        finished_tsc[sample] = started;
        actual_slot_index[sample] = started >= origin
            ? (int)(((double)(started - origin)) / slot_ticks) : -1;
        scheduler_lateness_ticks[sample] =
            started > requested ? started - requested : 0;
        service_ticks[sample] = 0;
        finish_gap_ticks[sample] =
            sample ? started - previous_finished : 0;
        missed_deadlines_before_sample[sample] = 0;
        valid_measurement[sample] = 1;
        previous_finished = started;
    }
}

static int gate_a_json_u64(FILE *f, uint64_t value) {
    if (!value) return fputs("null", f) < 0 ? -1 : 0;
    return fprintf(f, "%llu", (unsigned long long)value) < 0 ? -1 : 0;
}

static int gate_a_retain_readonly_burst_geometry(
        const GateASmokeArgs *args,
        GateASmokeResult *result,
        const GateASender epochs[4]) {
    if (!args || !result || !epochs) return -1;
    for (int slot = gate_a_readonly_stimulus_first_slot();
         slot < gate_a_readonly_stimulus_end_slot(); slot++) {
        const GateASender *sender = &epochs[0];
        uint64_t requested_start = result->capture_origin_tsc +
            (uint64_t)(slot * args->slot_s * result->capture_tsc_hz);
        uint64_t requested_end = result->capture_origin_tsc +
            (uint64_t)((slot + 1) * args->slot_s * result->capture_tsc_hz);
        uint64_t burst_start = sender->burst_start_tsc[slot];
        uint64_t burst_finish = sender->burst_finish_tsc[slot];
        result->occupancy_requested_slot_start_tsc[slot] = requested_start;
        result->occupancy_requested_slot_end_tsc[slot] = requested_end;
        result->occupancy_burst_start_tsc[slot] = burst_start;
        result->occupancy_burst_finish_tsc[slot] = burst_finish;
        result->occupancy_burst_duration_ticks[slot] =
            burst_finish >= burst_start ? burst_finish - burst_start : 0;
        result->occupancy_footprint_bytes[slot] =
            (uint64_t)gate_a_occupancy_bytes(slot);
        result->occupancy_completed_before_slot_end[slot] =
            burst_start >= requested_start &&
            burst_finish >= burst_start &&
            burst_finish <= requested_end &&
            sender->occupancy_slot_complete[slot];
    }
    return 0;
}

static int gate_a_readonly_timing_diagnostic(
        const GateASmokeArgs *args,
        GateASmokeResult *result,
        const uint64_t *requested_tsc,
        const uint64_t *started_tsc,
        const uint64_t *finished_tsc,
        const uint64_t *scheduler_lateness_ticks,
        const uint64_t *service_ticks,
        const uint64_t *finish_gap_ticks,
        const uint64_t *missed_deadlines_before_sample,
        const int *slot_index,
        int count,
        GateATimingDiagnostic *diagnostic) {
    if (!args || !result || !requested_tsc || !started_tsc || !finished_tsc ||
        !scheduler_lateness_ticks || !service_ticks || !finish_gap_ticks ||
        !missed_deadlines_before_sample ||
        !slot_index || count <= 0 || !diagnostic ||
        !isfinite(result->capture_tsc_hz) || result->capture_tsc_hz <= 0.0 ||
        args->read_hz <= 0 || args->slot_s <= 0.0) return -1;
    memset(diagnostic, 0, sizeof(*diagnostic));
    diagnostic->classification = "CAPTURE_ACCEPTED";
    double nominal_spacing = result->capture_tsc_hz / (double)args->read_hz;
    if (!isfinite(nominal_spacing) || nominal_spacing <= 0.0 ||
        nominal_spacing > (double)UINT64_MAX / GATE_A_SCHEDULER_REJECT_MULTIPLE) {
        diagnostic->record_integrity_failure = 1;
    }
    uint64_t scheduler_reject_threshold = diagnostic->record_integrity_failure
        ? 0 : (uint64_t)(nominal_spacing * GATE_A_SCHEDULER_REJECT_MULTIPLE);
    uint64_t missed_threshold = diagnostic->record_integrity_failure
        ? 0 : (uint64_t)(nominal_spacing * GATE_A_SCHEDULER_MISSED_MULTIPLE);
    diagnostic->scheduler_lateness_reject_threshold_ticks =
        scheduler_reject_threshold;
    result->scheduler_lateness_reject_threshold_ticks =
        scheduler_reject_threshold;
    uint64_t *lateness_sorted = calloc((size_t)count, sizeof(*lateness_sorted));
    uint64_t *service_sorted = calloc((size_t)count, sizeof(*service_sorted));
    if (!lateness_sorted || !service_sorted) {
        free(lateness_sorted);
        free(service_sorted);
        return -1;
    }
    for (int sample = 0; sample < count; sample++) {
        int slot = slot_index[sample];
        if (slot < 0 || slot >= 16) diagnostic->record_integrity_failure = 1;
        if (sample > 0) {
            if (requested_tsc[sample] <= requested_tsc[sample - 1] ||
                finished_tsc[sample] < finished_tsc[sample - 1]) {
                diagnostic->record_integrity_failure = 1;
            }
        }
        if (started_tsc[sample] < requested_tsc[sample] ||
            finished_tsc[sample] < started_tsc[sample]) {
            diagnostic->record_integrity_failure = 1;
        }
        uint64_t expected_lateness = started_tsc[sample] > requested_tsc[sample]
            ? started_tsc[sample] - requested_tsc[sample] : 0;
        uint64_t expected_service = finished_tsc[sample] - started_tsc[sample];
        uint64_t expected_finish_gap = sample
            ? finished_tsc[sample] - finished_tsc[sample - 1] : 0;
        if (scheduler_lateness_ticks[sample] != expected_lateness ||
            service_ticks[sample] != expected_service ||
            finish_gap_ticks[sample] != expected_finish_gap) {
            diagnostic->record_integrity_failure = 1;
        }
        if (scheduler_lateness_ticks[sample] >
            diagnostic->max_scheduler_lateness_ticks) {
            diagnostic->max_scheduler_lateness_ticks =
                scheduler_lateness_ticks[sample];
            diagnostic->max_scheduler_lateness_sample_index = (uint64_t)sample;
            diagnostic->max_scheduler_lateness_slot_index =
                slot >= 0 ? (uint64_t)slot : UINT64_MAX;
        }
        if (scheduler_lateness_ticks[sample] > missed_threshold) {
            diagnostic->missed_deadline_count++;
        }
        diagnostic->missed_deadline_count +=
            missed_deadlines_before_sample[sample];
        if (service_ticks[sample] > diagnostic->max_service_ticks) {
            diagnostic->max_service_ticks = service_ticks[sample];
        }
        if (finish_gap_ticks[sample] > diagnostic->max_finish_gap_ticks) {
            diagnostic->max_finish_gap_ticks = finish_gap_ticks[sample];
        }
        lateness_sorted[sample] = scheduler_lateness_ticks[sample];
        service_sorted[sample] = service_ticks[sample];
    }
    qsort(lateness_sorted, (size_t)count, sizeof(*lateness_sorted),
          gate_a_u64_compare);
    qsort(service_sorted, (size_t)count, sizeof(*service_sorted),
          gate_a_u64_compare);
    diagnostic->p50_scheduler_lateness_ticks =
        gate_a_sorted_percentile(lateness_sorted, count, 50U);
    diagnostic->p95_scheduler_lateness_ticks =
        gate_a_sorted_percentile(lateness_sorted, count, 95U);
    diagnostic->p99_scheduler_lateness_ticks =
        gate_a_sorted_percentile(lateness_sorted, count, 99U);
    diagnostic->p50_service_ticks =
        gate_a_sorted_percentile(service_sorted, count, 50U);
    diagnostic->p95_service_ticks =
        gate_a_sorted_percentile(service_sorted, count, 95U);
    diagnostic->p99_service_ticks =
        gate_a_sorted_percentile(service_sorted, count, 99U);
    diagnostic->p50_service_cycles_per_access =
        (double)diagnostic->p50_service_ticks / GATE_A_RESPONSE_TOUCHES;
    diagnostic->p95_service_cycles_per_access =
        (double)diagnostic->p95_service_ticks / GATE_A_RESPONSE_TOUCHES;
    diagnostic->p99_service_cycles_per_access =
        (double)diagnostic->p99_service_ticks / GATE_A_RESPONSE_TOUCHES;
    diagnostic->max_service_cycles_per_access =
        (double)diagnostic->max_service_ticks / GATE_A_RESPONSE_TOUCHES;
    free(lateness_sorted);
    free(service_sorted);

    int min_slot_samples = (int)(0.9 * args->read_hz * args->slot_s);
    if (min_slot_samples <= 0) diagnostic->record_integrity_failure = 1;
    for (int slot = 0; slot < gate_a_slot_count(); slot++) {
        if (result->slot_sample_counts[slot] < min_slot_samples) {
            diagnostic->missing_slot = 1;
        }
    }
    if (!result->capture_origin_tsc ||
        result->capture_first_sample_tsc < result->capture_origin_tsc ||
        result->capture_last_sample_tsc >= result->capture_deadline_tsc ||
        count != result->sample_count) {
        diagnostic->record_integrity_failure = 1;
    }
    for (int slot = gate_a_readonly_stimulus_first_slot();
         slot < gate_a_readonly_stimulus_end_slot(); slot++) {
        if (result->occupancy_touch_count[slot] !=
                GATE_A_READONLY_SLOT_TOUCHES ||
            !result->occupancy_completed_before_slot_end[slot] ||
            result->occupancy_burst_start_tsc[slot] <
                result->occupancy_requested_slot_start_tsc[slot] ||
            result->occupancy_burst_finish_tsc[slot] >
                result->occupancy_requested_slot_end_tsc[slot] ||
            result->occupancy_burst_finish_tsc[slot] <
                result->occupancy_burst_start_tsc[slot]) {
            diagnostic->sender_spill = 1;
        }
    }
    if (!result->occupancy_digest_unchanged) {
        diagnostic->sender_spill = 1;
    }
    uint64_t missed_limit =
        (uint64_t)(count * GATE_A_SCHEDULER_MISSED_FRACTION_MAX);
    if (missed_limit < 1U) missed_limit = 1U;
    if (diagnostic->record_integrity_failure) {
        diagnostic->classification = "CAPTURE_REJECTED_RECORD_INTEGRITY";
    } else if (diagnostic->missing_slot) {
        diagnostic->classification = "CAPTURE_REJECTED_MISSING_SLOT";
    } else if (diagnostic->sender_spill) {
        diagnostic->classification = "CAPTURE_REJECTED_SENDER_SPILL";
    } else if (diagnostic->max_scheduler_lateness_ticks >
                   scheduler_reject_threshold ||
               diagnostic->missed_deadline_count > missed_limit) {
        diagnostic->classification = "CAPTURE_REJECTED_SCHEDULER_LATENESS";
    } else if (diagnostic->p50_service_ticks > 0 &&
               diagnostic->max_service_ticks >
                   (uint64_t)(diagnostic->p50_service_ticks *
                              GATE_A_SERVICE_SPIKE_MULTIPLE)) {
        diagnostic->service_spikes = 1;
        diagnostic->classification =
            "CAPTURE_ACCEPTED_WITH_SERVICE_SPIKES";
    }
    result->max_scheduler_lateness_ticks =
        diagnostic->max_scheduler_lateness_ticks;
    result->missed_deadline_count = diagnostic->missed_deadline_count;
    result->skipped_deadline_count = diagnostic->missed_deadline_count;
    result->max_finish_gap_ticks = diagnostic->max_finish_gap_ticks;
    if (gate_a_copy_classification(result->capture_quality_classification,
                                   diagnostic->classification)) return -1;
    return 0;
}

static int gate_a_write_timing_diagnostic_summary(
        const GateASmokeArgs *args,
        const GateASmokeResult *result,
        const GateATimingDiagnostic *diagnostic) {
    char path[CP_PATH_MAX];
    if (!args || !result || !diagnostic ||
        joinp(path, sizeof(path), args->output_dir,
              GATE_A_TIMING_DIAGNOSTIC_FILE)) return -1;
    FILE *file = fopen(path, "wx");
    if (!file) return -1;
    if (fprintf(file,
            "{\n"
            "  \"schema_id\": \"CAT_CAS_READONLY_OCCUPANCY_TIMING_DIAGNOSTIC_V1\",\n"
            "  \"sample_timing_schema_id\": \"%s\",\n"
            "  \"sample_timing_file\": \"%s\",\n"
            "  \"sample_timing_record_bytes\": %u,\n"
            "  \"measurement_mode\": \"%s\",\n"
            "  \"quality_thresholds\": {\n"
            "    \"scheduler_reject_multiple_of_nominal_spacing\": %.17g,\n"
            "    \"scheduler_missed_deadline_multiple_of_nominal_spacing\": %.17g,\n"
            "    \"scheduler_missed_deadline_fraction_max\": %.17g,\n"
            "    \"min_slot_coverage_fraction\": %.17g,\n"
            "    \"service_spike_multiple_of_p50\": %.17g\n"
            "  },\n"
            "  \"scheduler_lateness\": {\n"
            "    \"p50_ticks\": %llu,\n"
            "    \"p95_ticks\": %llu,\n"
            "    \"p99_ticks\": %llu,\n"
            "    \"max_ticks\": %llu,\n"
            "    \"max_sample_index\": %llu,\n"
            "    \"max_slot_index\": %llu,\n"
            "    \"reject_threshold_ticks\": %llu\n"
            "  },\n"
            "  \"service_cycles_per_access\": {\n"
            "    \"p50\": %.17g,\n"
            "    \"p95\": %.17g,\n"
            "    \"p99\": %.17g,\n"
            "    \"max\": %.17g\n"
            "  },\n"
            "  \"service_ticks\": {\n"
            "    \"p50\": %llu,\n"
            "    \"p95\": %llu,\n"
            "    \"p99\": %llu,\n"
            "    \"max\": %llu\n"
            "  },\n"
            "  \"missed_deadline_count\": %llu,\n"
            "  \"skipped_deadline_count\": %llu,\n"
            "  \"max_finish_gap_ticks\": %llu,\n"
            "  \"sample_count_per_slot\": [",
            GATE_A_SAMPLE_TIMING_SCHEMA_ID,
            GATE_A_SAMPLE_TIMING_FILE,
            GATE_A_SAMPLE_TIMING_RECORD_BYTES,
            gate_a_measurement_mode(),
            GATE_A_SCHEDULER_REJECT_MULTIPLE,
            GATE_A_SCHEDULER_MISSED_MULTIPLE,
            GATE_A_SCHEDULER_MISSED_FRACTION_MAX,
            0.9,
            GATE_A_SERVICE_SPIKE_MULTIPLE,
            (unsigned long long)diagnostic->p50_scheduler_lateness_ticks,
            (unsigned long long)diagnostic->p95_scheduler_lateness_ticks,
            (unsigned long long)diagnostic->p99_scheduler_lateness_ticks,
            (unsigned long long)diagnostic->max_scheduler_lateness_ticks,
            (unsigned long long)diagnostic->max_scheduler_lateness_sample_index,
            (unsigned long long)diagnostic->max_scheduler_lateness_slot_index,
            (unsigned long long)diagnostic->scheduler_lateness_reject_threshold_ticks,
            diagnostic->p50_service_cycles_per_access,
            diagnostic->p95_service_cycles_per_access,
            diagnostic->p99_service_cycles_per_access,
            diagnostic->max_service_cycles_per_access,
            (unsigned long long)diagnostic->p50_service_ticks,
            (unsigned long long)diagnostic->p95_service_ticks,
            (unsigned long long)diagnostic->p99_service_ticks,
            (unsigned long long)diagnostic->max_service_ticks,
            (unsigned long long)diagnostic->missed_deadline_count,
            (unsigned long long)result->skipped_deadline_count,
            (unsigned long long)diagnostic->max_finish_gap_ticks) < 0) {
        goto write_error;
    }
    for (int slot = 0; slot < gate_a_slot_count(); slot++) {
        if ((slot && fputc(',', file) == EOF) ||
            fprintf(file, "%d", result->slot_sample_counts[slot]) < 0) {
            goto write_error;
        }
    }
    if (fputs("],\n  \"sender_burst_boundaries\": [", file) < 0) goto write_error;
    for (int slot = gate_a_readonly_stimulus_first_slot();
         slot < gate_a_readonly_stimulus_end_slot(); slot++) {
        if ((slot > gate_a_readonly_stimulus_first_slot() &&
             fputc(',', file) == EOF) ||
            fprintf(file,
                "\n"
                "    {\"slot_index\": %d, \"requested_slot_start_tsc\": %llu,"
                " \"requested_slot_end_tsc\": %llu,"
                " \"burst_start_tsc\": %llu,"
                " \"burst_finish_tsc\": %llu,"
                " \"burst_duration_ticks\": %llu,"
                " \"touch_count\": %llu,"
                " \"footprint_bytes\": %llu,"
                " \"initial_cursor\": %llu,"
                " \"final_cursor\": %llu,"
                " \"completed_before_slot_end\": %s}",
                slot,
                (unsigned long long)result->occupancy_requested_slot_start_tsc[slot],
                (unsigned long long)result->occupancy_requested_slot_end_tsc[slot],
                (unsigned long long)result->occupancy_burst_start_tsc[slot],
                (unsigned long long)result->occupancy_burst_finish_tsc[slot],
                (unsigned long long)result->occupancy_burst_duration_ticks[slot],
                (unsigned long long)result->occupancy_touch_count[slot],
                (unsigned long long)result->occupancy_footprint_bytes[slot],
                (unsigned long long)result->occupancy_initial_cursor[slot],
                (unsigned long long)result->occupancy_final_cursor[slot],
                result->occupancy_completed_before_slot_end[slot]
                    ? "true" : "false") < 0) {
            goto write_error;
        }
    }
    if (fprintf(file,
            "\n  ],\n"
            "  \"record_integrity_failure\": %s,\n"
            "  \"missing_slot\": %s,\n"
            "  \"sender_spill\": %s,\n"
            "  \"service_spikes\": %s,\n"
            "  \"capture_quality_classification\": \"%s\"\n"
            "}\n",
            diagnostic->record_integrity_failure ? "true" : "false",
            diagnostic->missing_slot ? "true" : "false",
            diagnostic->sender_spill ? "true" : "false",
            diagnostic->service_spikes ? "true" : "false",
            diagnostic->classification) < 0) goto write_error;
    return close_sync(&file);

write_error:
    fclose(file);
    unlink(path);
    return -1;
}

static void gate_a_test_fill_timing_case(
        GateASmokeResult *result,
        uint64_t requested_tsc[32],
        uint64_t started_tsc[32],
        uint64_t finished_tsc[32],
        uint64_t scheduler_lateness_ticks[32],
        uint64_t service_ticks[32],
        uint64_t finish_gap_ticks[32],
        uint64_t missed_deadlines_before_sample[32],
        int slot_index[32],
        uint64_t service_per_sample,
        int late_last_sample,
        int sender_spill) {
    memset(result, 0, sizeof(*result));
    uint64_t origin = 1000000000ULL;
    uint64_t spacing = 800000ULL;
    result->slot_count = 16;
    result->sample_count = 32;
    result->capture_origin_tsc = origin;
    result->capture_deadline_tsc = origin + 25600000ULL;
    result->capture_first_sample_tsc = origin;
    result->capture_last_sample_tsc = origin + 31ULL * spacing;
    result->capture_tsc_hz = 3200000.0;
    result->occupancy_digest_unchanged = 1;
    uint64_t previous_finished = 0;
    for (int sample = 0; sample < 32; sample++) {
        uint64_t late = (late_last_sample && sample == 31) ? 4000000ULL : 0ULL;
        requested_tsc[sample] = origin + (uint64_t)sample * spacing;
        started_tsc[sample] = requested_tsc[sample] + late;
        finished_tsc[sample] = started_tsc[sample] + service_per_sample;
        scheduler_lateness_ticks[sample] = late;
        service_ticks[sample] = service_per_sample;
        missed_deadlines_before_sample[sample] = 0;
        finish_gap_ticks[sample] =
            sample ? finished_tsc[sample] - previous_finished : 0;
        slot_index[sample] = sample / 2;
        result->slot_sample_counts[slot_index[sample]]++;
        previous_finished = finished_tsc[sample];
    }
    for (int slot = gate_a_readonly_stimulus_first_slot();
         slot < gate_a_readonly_stimulus_end_slot(); slot++) {
        uint64_t start = origin + (uint64_t)slot * 1600000ULL;
        uint64_t end = start + 1600000ULL;
        result->occupancy_requested_slot_start_tsc[slot] = start;
        result->occupancy_requested_slot_end_tsc[slot] = end;
        result->occupancy_burst_start_tsc[slot] = start + 10ULL;
        result->occupancy_burst_finish_tsc[slot] =
            sender_spill && slot == gate_a_readonly_stimulus_first_slot() + 1
                ? end + 1ULL : end - 10ULL;
        result->occupancy_burst_duration_ticks[slot] =
            result->occupancy_burst_finish_tsc[slot] -
            result->occupancy_burst_start_tsc[slot];
        result->occupancy_touch_count[slot] = GATE_A_READONLY_SLOT_TOUCHES;
        result->occupancy_footprint_bytes[slot] =
            (uint64_t)GATE_A_OCCUPANCY_EQUAL_BYTES;
        result->occupancy_completed_before_slot_end[slot] =
            !sender_spill ||
            slot != gate_a_readonly_stimulus_first_slot() + 1;
    }
}

int gate_a_test_readonly_timing_diagnostics(void) {
    GateASmokeArgs args = {
        .read_hz = 4,
        .slot_s = 0.5,
        .pilot_variant = GATE_A_PILOT_READONLY_OCCUPANCY_EQUAL,
    };
    GateASmokeResult result;
    GateATimingDiagnostic diagnostic;
    uint64_t requested[32], started[32], finished[32], lateness[32];
    uint64_t service[32], finish_gap[32], missed[32];
    int slot_index[32];
    gate_a_pilot_variant = GATE_A_PILOT_READONLY_OCCUPANCY_EQUAL;

    gate_a_test_fill_timing_case(
        &result, requested, started, finished, lateness, service,
        finish_gap, missed, slot_index, 6400ULL, 0, 0);
    if (gate_a_readonly_timing_diagnostic(
            &args, &result, requested, started, finished, lateness, service,
            finish_gap, missed, slot_index, 32, &diagnostic) ||
        strcmp(diagnostic.classification, "CAPTURE_ACCEPTED")) return 1;

    gate_a_test_fill_timing_case(
        &result, requested, started, finished, lateness, service,
        finish_gap, missed, slot_index, 6400ULL, 0, 0);
    service[15] = 64000ULL;
    finished[15] = started[15] + service[15];
    finish_gap[15] = finished[15] - finished[14];
    finish_gap[16] = finished[16] - finished[15];
    if (gate_a_readonly_timing_diagnostic(
            &args, &result, requested, started, finished, lateness, service,
            finish_gap, missed, slot_index, 32, &diagnostic) ||
        strcmp(diagnostic.classification,
               "CAPTURE_ACCEPTED_WITH_SERVICE_SPIKES")) return 1;

    gate_a_test_fill_timing_case(
        &result, requested, started, finished, lateness, service,
        finish_gap, missed, slot_index, 6400ULL, 1, 0);
    if (gate_a_readonly_timing_diagnostic(
            &args, &result, requested, started, finished, lateness, service,
            finish_gap, missed, slot_index, 32, &diagnostic) ||
        strcmp(diagnostic.classification,
               "CAPTURE_REJECTED_SCHEDULER_LATENESS")) return 1;

    gate_a_test_fill_timing_case(
        &result, requested, started, finished, lateness, service,
        finish_gap, missed, slot_index, 6400ULL, 0, 1);
    if (gate_a_readonly_timing_diagnostic(
            &args, &result, requested, started, finished, lateness, service,
            finish_gap, missed, slot_index, 32, &diagnostic) ||
        strcmp(diagnostic.classification,
               "CAPTURE_REJECTED_SENDER_SPILL")) return 1;
    return 0;
}

static int gate_a_lifecycle_record(FILE *f, const char *record_type,
                                   const char *state, int slot,
                                   uint64_t event_tsc,
                                   uint64_t requested_start,
                                   uint64_t requested_end,
                                   const GateASender *sender) {
    if (!f || !record_type || !state ||
        slot < 0 || slot >= gate_a_slot_count() || !event_tsc) return -1;
    if (fprintf(f,
            "{\"schema_id\":\"CAT_CAS_PHASE6B6_GATE_A_SENDER_LIFECYCLE_V1\","
            "\"record_type\":\"%s\",\"event_tsc\":%llu,\"slot_index\":%d,"
            "\"token\":\"%s\",\"sender_state\":\"%s\",\"sender_epoch_id\":",
            record_type, (unsigned long long)event_tsc, slot,
            gate_a_slot_token(slot), state) < 0) return -1;
    if (sender) {
        if (fprintf(f, "\"%s\",\"phase_index\":%d,\"sign\":%d,\"orbit_value\":",
                    sender->epoch_id, sender->phase_index, sender->sign) < 0) return -1;
        int orbit_value = gate_a_orbit_value(slot);
        if (orbit_value < 0) orbit_value = sender->orbit_value;
        if (orbit_value >= 0) {
            if (fprintf(f, "%d,", orbit_value) < 0) return -1;
        } else if (fputs("null,", f) < 0) return -1;
    } else if (fputs("null,\"phase_index\":null,\"sign\":null,\"orbit_value\":null,", f) < 0) return -1;
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

/* Equal-count integer path whose switching state consumes the public orbit
   value.  The value changes data, never duration, phase, or control flow. */
static double gate_a_value_burst(uint64_t *seed,
                                 volatile uint64_t switching_bank[256],
                                 int orbit_value) {
    uint64_t byte = (uint64_t)(unsigned int)(orbit_value & 255);
    uint64_t pattern = byte * UINT64_C(0x0101010101010101);
    for (int index = 0; index < 256; index++) {
        switching_bank[index] ^= pattern;
    }
    *seed ^= switching_bank[(unsigned int)orbit_value & 255U] ^ pattern;
    return (double)(*seed & 0xffffU);
}

static double gate_a_occupancy_burst(
        uint64_t *seed, volatile uint64_t *bank,
        size_t footprint_bytes, size_t *cursor) {
    size_t line_count = footprint_bytes / GATE_A_CACHE_LINE_BYTES;
    size_t line_mask = line_count - 1U;
    size_t words_per_line = GATE_A_CACHE_LINE_BYTES / sizeof(uint64_t);
    size_t line = *cursor & line_mask;
    uint64_t value = *seed;
    for (size_t touch = 0; touch < GATE_A_OCCUPANCY_BURST_TOUCHES; touch++) {
        line = (line + GATE_A_OCCUPANCY_LINE_STRIDE) & line_mask;
        size_t word = line * words_per_line;
        value ^= bank[word] + (uint64_t)line;
        bank[word] = value ^ UINT64_C(0x9e3779b97f4a7c15);
    }
    *cursor = line;
    *seed = value;
    return (double)(value & 0xffffU);
}

/* Exact-count read-only stimulus over CAT_CAS-owned synthetic memory.  The
   volatile load is the only occupancy-buffer access in this loop. */
static SMALL_WALL_CRITICAL double small_wall_readonly_occupancy_reads(
        uint64_t *seed, const volatile uint64_t *bank,
        size_t footprint_bytes, size_t *cursor) {
    size_t line_count = footprint_bytes / GATE_A_CACHE_LINE_BYTES;
    size_t line_mask = line_count - 1U;
    size_t words_per_line = GATE_A_CACHE_LINE_BYTES / sizeof(uint64_t);
    size_t line = *cursor & line_mask;
    uint64_t value = *seed;
    for (size_t touch = 0; touch < GATE_A_READONLY_SLOT_TOUCHES; touch++) {
        line = (line + GATE_A_OCCUPANCY_LINE_STRIDE) & line_mask;
        value ^= bank[line * words_per_line] + (uint64_t)line;
    }
    *cursor = line;
    *seed = value;
    return (double)(value & 0xffffU);
}

static SMALL_WALL_CRITICAL uint64_t small_wall_prefault_buffer(
        const volatile uint64_t *bank, size_t bytes) {
    const size_t page_bytes = 4096U;
    size_t words_per_page = page_bytes / sizeof(uint64_t);
    size_t word_count = bytes / sizeof(uint64_t);
    uint64_t sink = 0;
    for (size_t word = 0; word < word_count; word += words_per_page) {
        sink ^= bank[word];
    }
    return sink;
}

static SMALL_WALL_CRITICAL void *gate_a_sender_loop(void *opaque) {
    GateASender *sender = opaque;
    if (pin_core(sender->core)) {
        atomic_store_explicit(&sender->thread_exit_tsc, rdtsc_now(), memory_order_release);
        return (void *)1;
    }
    sender->realtime_attempted = 1;
    sender->realtime_applied = small_wall_request_realtime_thread();
    uint64_t ready = rdtsc_now();
    atomic_store_explicit(&sender->ready_tsc, ready, memory_order_release);
    atomic_store_explicit(&sender->epoch_start_tsc, ready, memory_order_release);
    atomic_store_explicit(&sender->ready, 1, memory_order_release);
    uint64_t seed = 0x9e3779b9u + (uint64_t)sender->core;
    if (sender->orbit_value < 0) seed += (uint64_t)sender->first_slot;
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
            if (gate_a_readonly_occupancy_pilot()) {
                seed = UINT64_C(0x9e3779b9) + (uint64_t)sender->core;
                sender->occupancy_cursor = 0;
                sender->occupancy_initial_cursor[slot] = 0;
            }
        }
        if (gate_a_readonly_occupancy_pilot() &&
            slot >= gate_a_readonly_stimulus_first_slot() &&
            slot < gate_a_readonly_stimulus_end_slot()) {
            if (!sender->occupancy_slot_complete[slot]) {
                unsigned long long expected = 0;
                atomic_compare_exchange_strong_explicit(
                    &sender->first_drive_tsc, &expected, now,
                    memory_order_release, memory_order_relaxed);
                size_t occupancy_bytes = gate_a_occupancy_bytes(slot);
                if (!occupancy_bytes) return (void *)1;
                sender->occupancy_initial_cursor[slot] =
                    sender->occupancy_cursor;
                sender->burst_start_tsc[slot] = rdtscp_now();
                sink += small_wall_readonly_occupancy_reads(
                    &seed, sender->occupancy_bank, occupancy_bytes,
                    &sender->occupancy_cursor);
                sender->burst_finish_tsc[slot] = rdtscp_now();
                sender->occupancy_touch_count[slot] =
                    GATE_A_READONLY_SLOT_TOUCHES;
                sender->occupancy_final_cursor[slot] = sender->occupancy_cursor;
                sender->occupancy_slot_complete[slot] = 1;
            }
            continue;
        }
        if (gate_a_epoch_cycle_state(sender, now) < 2) {
            unsigned long long expected = 0;
            atomic_compare_exchange_strong_explicit(
                &sender->first_drive_tsc, &expected, now,
                memory_order_release, memory_order_relaxed);
            int orbit_value = gate_a_orbit_value(slot);
            if (orbit_value < 0) orbit_value = sender->orbit_value;
            size_t occupancy_bytes = gate_a_occupancy_bytes(slot);
            if (occupancy_bytes) {
                sink += gate_a_occupancy_burst(
                    &seed, sender->occupancy_bank, occupancy_bytes,
                    &sender->occupancy_cursor);
            } else if (orbit_value >= 0) {
                sink += gate_a_value_burst(
                    &seed, sender->switching_bank, orbit_value);
            } else {
                sink += alu_burst(&seed);
            }
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
                             const char *epoch_id,
                             volatile uint64_t *occupancy_bank) {
    memset(sender, 0, sizeof(*sender));
    sender->core = core;
    sender->tsc_hz = tsc_hz;
    sender->slot_s = slot_s;
    sender->session_origin = session_origin;
    sender->first_slot = first_slot;
    sender->end_slot = end_slot;
    sender->phase_index = phase_index;
    sender->sign = sign;
    sender->orbit_value = gate_a_orbit_value(first_slot);
    sender->occupancy_bank = occupancy_bank;
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

static SMALL_WALL_CRITICAL void *gate_a_receiver_loop(void *opaque) {
    GateAReceiver *receiver = opaque;
    if (pin_core(receiver->core)) {
        receiver->count = -1;
        atomic_store_explicit(&receiver->done, 1, memory_order_release);
        return (void *)1;
    }
    receiver->realtime_attempted = 1;
    receiver->realtime_applied = small_wall_request_realtime_thread();
    atomic_store_explicit(&receiver->ready_tsc, rdtsc_now(), memory_order_release);
    atomic_store_explicit(&receiver->ready, 1, memory_order_release);
    if (receiver->cache_response_mode) {
        while (rdtsc_now() < receiver->origin && !interrupted) {
            __asm__ volatile("pause");
        }
        receiver->receiver_epoch = rdtsc_now();
        uint64_t end = receiver->origin +
            (uint64_t)(receiver->duration_s * receiver->tsc_hz);
        double spacing = receiver->tsc_hz / receiver->read_hz;
        double slot_ticks = receiver->slot_s * receiver->tsc_hz;
        size_t words_per_line = GATE_A_CACHE_LINE_BYTES / sizeof(uint64_t);
        size_t index = receiver->response_index;
        int count = 0;
        uint64_t requested_sample_index = 0;
        uint64_t previous_finished = 0;
        uint64_t skipped_before_next_sample = 0;
        while (count < receiver->capacity && !interrupted) {
            uint64_t requested = receiver->origin +
                (uint64_t)((double)requested_sample_index * spacing);
            if (requested >= end) break;
            while (rdtsc_now() < requested && !interrupted) {
                __asm__ volatile("pause");
            }
            uint64_t started = rdtscp_now();
            if (started >= end) break;
            uint64_t scheduler_lateness =
                started > requested ? started - requested : 0;
            if (scheduler_lateness > (uint64_t)spacing) {
                uint64_t missed = (uint64_t)(
                    ((double)scheduler_lateness / spacing)) + 1U;
                if (!missed) missed = 1U;
                requested_sample_index += missed;
                skipped_before_next_sample += missed;
                receiver->skipped_deadline_count += missed;
                continue;
            }
            int valid = 1;
            for (size_t touch = 0; touch < GATE_A_RESPONSE_TOUCHES; touch++) {
                if (index >= receiver->response_line_count) {
                    valid = 0;
                    break;
                }
                index = (size_t)receiver->response_bank[index * words_per_line];
            }
            if (!valid || index >= receiver->response_line_count) {
                count = -1;
                break;
            }
            uint64_t finished = rdtscp_now();
            if (started < requested || finished < started ||
                (count > 0 && finished < previous_finished)) {
                count = -1;
                break;
            }
            uint64_t service = finished - started;
            uint64_t finish_gap = count > 0 ? finished - previous_finished : 0;
            double response_cycles =
                (double)service / GATE_A_RESPONSE_TOUCHES;
            int requested_slot = slot_ticks > 0.0
                ? (int)(((double)(requested - receiver->origin)) / slot_ticks)
                : -1;
            int actual_slot = slot_ticks > 0.0 && started >= receiver->origin
                ? (int)(((double)(started - receiver->origin)) / slot_ticks)
                : -1;
            if (requested_slot < 0 || requested_slot >= receiver->slot_count ||
                actual_slot < 0 || actual_slot >= receiver->slot_count) {
                count = -1;
                break;
            }
            if (scheduler_lateness > receiver->max_sample_delay_tsc) {
                receiver->max_sample_delay_tsc = scheduler_lateness;
            }
            if (response_cycles > receiver->max_response_cycles_per_access) {
                receiver->max_response_cycles_per_access = response_cycles;
            }
            receiver->timestamps[count] = started;
            receiver->requested_sample_index[count] = requested_sample_index;
            receiver->requested_tsc[count] = requested;
            receiver->requested_slot_index[count] = requested_slot;
            receiver->observations[count] = response_cycles;
            receiver->started_tsc[count] = started;
            receiver->finished_tsc[count] = finished;
            receiver->scheduler_lateness_ticks[count] = scheduler_lateness;
            receiver->service_ticks[count] = service;
            receiver->finish_gap_ticks[count] = finish_gap;
            receiver->missed_deadlines_before_sample[count] =
                skipped_before_next_sample;
            receiver->valid_measurement[count] = 1;
            receiver->timing_slot_index[count] = actual_slot;
            previous_finished = finished;
            skipped_before_next_sample = 0;
            count++;
            requested_sample_index++;
        }
        receiver->response_index = index;
        receiver->count = count;
        if (count == receiver->capacity) {
            while (rdtsc_now() < end && !interrupted) {
                __asm__ volatile("pause");
            }
        }
    } else {
        receiver->count = capture_at_origin(
            receiver->core, receiver->read_hz, receiver->duration_s,
            receiver->tsc_hz,
            receiver->origin, &receiver->receiver_epoch,
            receiver->timestamps, receiver->observations, receiver->capacity);
    }
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
            "  \"capture_timestamp_coordinate\": \"%s\",\n"
            "  \"capture_max_sample_delay_tsc\": %llu,\n"
            "  \"capture_max_response_cycles_per_access\": %.17g,\n"
            "  \"sample_timing_schema_id\": \"%s\",\n"
            "  \"sample_timing_file\": \"%s\",\n"
            "  \"timing_diagnostic_file\": \"%s\",\n"
            "  \"capture_quality_classification\": \"%s\",\n"
            "  \"max_scheduler_lateness_ticks\": %llu,\n"
            "  \"scheduler_lateness_reject_threshold_ticks\": %llu,\n"
            "  \"missed_deadline_count\": %llu,\n"
            "  \"skipped_deadline_count\": %llu,\n"
            "  \"max_finish_gap_ticks\": %llu,\n"
            "  \"step_sender_epoch_count\": %d,\n"
            "  \"hardware_executed\": %s,\n"
            "  \"sender_start_count\": %d,\n"
            "  \"receiver_start_count\": %d,\n"
            "  \"temperature_receipt_count\": %d,\n"
            "  \"pilot_variant\": %d,\n"
            "  \"measurement_mode\": \"%s\",\n"
            "  \"observation_kind\": \"%s\",\n"
            "  \"response_buffer_bytes\": %u,\n"
            "  \"response_touches_per_sample\": %u,\n"
            "  \"occupancy_classification\": \"%s\",\n"
            "  \"occupancy_prefaulted\": %s,\n"
            "  \"occupancy_digest_before\": \"%s\",\n"
            "  \"occupancy_digest_after\": \"%s\",\n"
            "  \"occupancy_digest_unchanged\": %s,\n"
            "  \"frequency_writes\": 0,\n"
            "  \"voltage_writes\": 0,\n"
            "  \"msr_reads\": 0,\n"
            "  \"msr_writes\": 0,\n"
            "  \"scheduler_policy_requested\": \"SCHED_FIFO\",\n"
            "  \"scheduler_priority_requested\": %d,\n"
            "  \"scheduler_priority_attempt_count\": %d,\n"
            "  \"scheduler_priority_success_count\": %d,\n"
            "  \"automatic_retry\": false,\n"
            "  \"occupancy_slot_touch_count\": [",
            args->authority_sha256, args->execution_bundle_sha256,
            result->slot_count, result->sample_count,
            (unsigned long long)result->capture_origin_tsc,
            (unsigned long long)result->capture_deadline_tsc,
            (unsigned long long)result->capture_first_sample_tsc,
            (unsigned long long)result->capture_last_sample_tsc,
            result->capture_tsc_hz,
            gate_a_readonly_occupancy_pilot()
                ? "actual_start_tsc" :
                (gate_a_occupancy_pilot()
                    ? "scheduled_request_tsc" : "measurement_finish_tsc"),
            (unsigned long long)result->capture_max_sample_delay_tsc,
            result->capture_max_response_cycles_per_access,
            GATE_A_SAMPLE_TIMING_SCHEMA_ID,
            GATE_A_SAMPLE_TIMING_FILE,
            gate_a_readonly_occupancy_pilot()
                ? GATE_A_TIMING_DIAGNOSTIC_FILE : "not_applicable",
            result->capture_quality_classification[0]
                ? result->capture_quality_classification : "NOT_EVALUATED",
            (unsigned long long)result->max_scheduler_lateness_ticks,
            (unsigned long long)result->scheduler_lateness_reject_threshold_ticks,
            (unsigned long long)result->missed_deadline_count,
            (unsigned long long)result->skipped_deadline_count,
            (unsigned long long)result->max_finish_gap_ticks,
            result->step_sender_epoch_count,
            result->hardware_executed ? "true" : "false",
            result->sender_start_count,
            result->receiver_start_count,
            result->temperature_receipt_count,
            args->pilot_variant,
            gate_a_measurement_mode(),
            gate_a_observation_kind(),
            gate_a_occupancy_pilot() ? GATE_A_RESPONSE_BUFFER_BYTES : 0U,
            gate_a_occupancy_pilot() ? GATE_A_RESPONSE_TOUCHES : 0U,
            gate_a_occupancy_classification(),
            result->occupancy_prefaulted ? "true" : "false",
            result->occupancy_digest_before,
            result->occupancy_digest_after,
            result->occupancy_digest_unchanged ? "true" : "false",
            GATE_A_REALTIME_PRIORITY,
            result->scheduler_priority_attempt_count,
            result->scheduler_priority_success_count) < 0) {
        fclose(f);
        unlink(path);
        return -1;
    }
    for (int slot = 0; slot < gate_a_slot_count(); slot++) {
        if ((slot && fputc(',', f) == EOF) ||
            fprintf(f, "%llu", (unsigned long long)result->occupancy_touch_count[slot]) < 0) {
            fclose(f);
            unlink(path);
            return -1;
        }
    }
    if (fputs("],\n  \"occupancy_initial_cursor\": [", f) < 0) goto write_error;
    for (int slot = 0; slot < gate_a_slot_count(); slot++) {
        if ((slot && fputc(',', f) == EOF) ||
            fprintf(f, "%llu", (unsigned long long)result->occupancy_initial_cursor[slot]) < 0) {
            goto write_error;
        }
    }
    if (fputs("],\n  \"occupancy_final_cursor\": [", f) < 0) goto write_error;
    for (int slot = 0; slot < gate_a_slot_count(); slot++) {
        if ((slot && fputc(',', f) == EOF) ||
            fprintf(f, "%llu", (unsigned long long)result->occupancy_final_cursor[slot]) < 0) {
            goto write_error;
        }
    }
    if (fputs("]\n}\n", f) < 0) goto write_error;
    return close_sync(&f);

write_error:
    fclose(f);
    unlink(path);
    return -1;
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
    sender->orbit_value = gate_a_orbit_value(first_slot);
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
    size_t readonly_cursor = 0;
    for (int slot = 0; slot < gate_a_slot_count(); slot++) {
        uint64_t value = slot >= first_slot && slot < end_slot
            ? session_origin + (uint64_t)(slot * slot_s * tsc_hz) + 3 : 0;
        atomic_init(&sender->transition_tsc[slot], value);
        if (gate_a_readonly_occupancy_pilot() && value &&
            slot >= gate_a_readonly_stimulus_first_slot() &&
            slot < gate_a_readonly_stimulus_end_slot()) {
            size_t line_count = gate_a_occupancy_bytes(slot) /
                GATE_A_CACHE_LINE_BYTES;
            size_t line_mask = line_count - 1U;
            size_t line = readonly_cursor & line_mask;
            sender->occupancy_initial_cursor[slot] = readonly_cursor;
            line = (line + GATE_A_READONLY_SLOT_TOUCHES *
                    GATE_A_OCCUPANCY_LINE_STRIDE) & line_mask;
            sender->occupancy_final_cursor[slot] =
                line;
            readonly_cursor = line;
            sender->occupancy_touch_count[slot] =
                GATE_A_READONLY_SLOT_TOUCHES;
            sender->occupancy_slot_complete[slot] = 1;
            sender->burst_start_tsc[slot] = value;
            sender->burst_finish_tsc[slot] =
                session_origin + (uint64_t)((slot + 1) * slot_s * tsc_hz) -
                3U;
        }
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
    uint64_t expected_drive_offset = (uint64_t)(
        ((double)sender->phase_index / 8.0) / tone(0) * sender->tsc_hz);
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
                                   uint64_t *requested_sample_index,
                                   uint64_t *requested_tsc,
                                   uint64_t *started_tsc,
                                   uint64_t *finished_tsc,
                                   int *requested_slot_index,
                                   uint64_t *scheduler_lateness_ticks,
                                   uint64_t *service_ticks,
                                   uint64_t *finish_gap_ticks,
                                   uint64_t *missed_deadlines_before_sample,
                                   int *valid_measurement,
                                   int *timing_slot_index,
                                   const uint64_t *response_bank,
                                   volatile uint64_t *occupancy_bank,
                                   int capacity, FILE *lifecycle,
                                   GateASender epochs[4],
                                   uint64_t *receiver_epoch,
                                   GateASmokeResult *result) {
    GateAReceiver receiver;
    memset(&receiver, 0, sizeof(receiver));
    receiver.core = args->receiver_core;
    receiver.read_hz = args->read_hz;
    receiver.tsc_hz = tsc_hz;
    receiver.slot_s = args->slot_s;
    receiver.duration_s = gate_a_duration_s(args);
    receiver.origin = origin;
    receiver.capacity = capacity;
    receiver.timestamps = timestamps;
    receiver.requested_sample_index = requested_sample_index;
    receiver.requested_tsc = requested_tsc;
    receiver.observations = observations;
    receiver.started_tsc = started_tsc;
    receiver.finished_tsc = finished_tsc;
    receiver.requested_slot_index = requested_slot_index;
    receiver.scheduler_lateness_ticks = scheduler_lateness_ticks;
    receiver.service_ticks = service_ticks;
    receiver.finish_gap_ticks = finish_gap_ticks;
    receiver.missed_deadlines_before_sample = missed_deadlines_before_sample;
    receiver.valid_measurement = valid_measurement;
    receiver.timing_slot_index = timing_slot_index;
    receiver.samples_per_slot = (int)(args->read_hz * args->slot_s + 0.5);
    receiver.slot_count = gate_a_slot_count();
    receiver.response_bank = response_bank;
    receiver.response_line_count = gate_a_occupancy_pilot()
        ? GATE_A_RESPONSE_BUFFER_BYTES / GATE_A_CACHE_LINE_BYTES : 0;
    receiver.response_index = 0;
    receiver.cache_response_mode = gate_a_occupancy_pilot();
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
    for (int slot = 0; slot < gate_a_slot_count(); slot++) {
        uint64_t requested = origin + (uint64_t)(slot * args->slot_s * tsc_hz);
        while (rdtsc_now() < requested) {
            if (atomic_load_explicit(&receiver.done, memory_order_acquire)) goto failure;
            __asm__ volatile("pause");
        }
        if (atomic_load_explicit(&receiver.done, memory_order_acquire)) goto failure;
        int first_step_slot = gate_a_readonly_occupancy_pilot()
            ? gate_a_readonly_stimulus_first_slot() : 6;
        int first_step_end = gate_a_split_step() ? 8 : gate_a_step_end_slot();
        if (slot == first_step_slot && gate_a_driven_slot(slot) && gate_a_sender_arm(
                &epochs[0], lifecycle, &receiver, result,
                args->sender_core, tsc_hz, args->slot_s,
                origin, first_step_slot, first_step_end,
                gate_a_phase_index(first_step_slot),
                gate_a_sign(first_step_slot),
                gate_a_epoch(first_step_slot), occupancy_bank)) goto failure;
        if (slot == first_step_end - 1 && gate_a_driven_slot(first_step_slot) &&
            gate_a_sender_stop_join(&epochs[0], lifecycle, &receiver)) goto failure;
        if (!gate_a_readonly_occupancy_pilot() &&
            slot == 8 && gate_a_split_step() && gate_a_sender_arm(
                &epochs[1], lifecycle, &receiver, result,
                args->sender_core, tsc_hz, args->slot_s,
                origin, 8, 10, gate_a_phase_index(8),
                gate_a_sign(8), gate_a_epoch(8), occupancy_bank)) goto failure;
        if (!gate_a_readonly_occupancy_pilot() &&
            slot == 9 && gate_a_split_step() &&
            gate_a_sender_stop_join(&epochs[1], lifecycle, &receiver)) goto failure;
        if (!gate_a_readonly_occupancy_pilot() && slot == 12) {
            if (gate_a_driven_slot(slot) && (gate_a_sender_arm(
                    &epochs[2], lifecycle, &receiver, result,
                    args->sender_core, tsc_hz, args->slot_s,
                    origin, 12, 13, gate_a_phase_index(slot),
                    gate_a_sign(slot), gate_a_epoch(slot), NULL) ||
                gate_a_sender_stop_join(&epochs[2], lifecycle, &receiver))) goto failure;
        }
        if (!gate_a_readonly_occupancy_pilot() && slot == 13) {
            if (gate_a_driven_slot(slot) && (gate_a_sender_arm(
                    &epochs[3], lifecycle, &receiver, result,
                    args->sender_core, tsc_hz, args->slot_s,
                    origin, 13, 14, gate_a_phase_index(slot),
                    gate_a_sign(slot), gate_a_epoch(slot), NULL) ||
                gate_a_sender_stop_join(&epochs[3], lifecycle, &receiver))) goto failure;
        }
    }
    uint64_t capture_deadline = origin +
        (uint64_t)(gate_a_duration_s(args) * tsc_hz);
    while (rdtsc_now() < capture_deadline) {
        __asm__ volatile("pause");
    }
    {
        void *receiver_result = NULL;
        int join_rc = pthread_join(receiver.thread, &receiver_result);
        receiver.alive = 0;
        if (join_rc || receiver_result) return -1;
    }
    result->scheduler_priority_attempt_count += receiver.realtime_attempted;
    result->scheduler_priority_success_count += receiver.realtime_applied;
    result->capture_max_sample_delay_tsc = receiver.max_sample_delay_tsc;
    result->capture_max_response_cycles_per_access =
        receiver.max_response_cycles_per_access;
    result->skipped_deadline_count = receiver.skipped_deadline_count;
    for (int index = 0; index < 4; index++) {
        result->scheduler_priority_attempt_count +=
            epochs[index].realtime_attempted;
        result->scheduler_priority_success_count +=
            epochs[index].realtime_applied;
    }
    *receiver_epoch = receiver.receiver_epoch;
    return receiver.count;

failure:
    for (int index = 0; index < 4; index++) {
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
    for (int slot = 0; slot < gate_a_slot_count(); slot++) {
        int count = ends[slot] - starts[slot];
        if (count < 2) {
            fclose(f);
            return -1;
        }
        uint64_t slot_start = result->capture_origin_tsc +
            (uint64_t)(slot * args->slot_s * result->capture_tsc_hz);
        uint64_t slot_end = result->capture_origin_tsc +
            (uint64_t)((slot + 1) * args->slot_s * result->capture_tsc_hz);
        int analysis_origin_slot = slot;
        if (!gate_a_readonly_occupancy_pilot() && slot >= 6 && slot <= 9) {
            analysis_origin_slot = gate_a_split_step() && slot >= 8 ? 8 : 6;
        } else if (gate_a_readonly_occupancy_pilot() &&
                   slot >= gate_a_readonly_stimulus_first_slot() &&
                   slot < gate_a_readonly_stimulus_end_slot()) {
            analysis_origin_slot = gate_a_readonly_stimulus_first_slot();
        }
        uint64_t analysis_origin = result->capture_origin_tsc +
            (uint64_t)(analysis_origin_slot * args->slot_s *
                       result->capture_tsc_hz);
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
                "\"measurement_mode\":\"%s\","
                "\"observation_kind\":\"%s\","
                "\"lockin_i\":%.17g,\"lockin_q\":%.17g,"
                "\"magnitude\":%.17g,\"off_frequency_floor\":%.17g,"
                "\"origin_tsc\":%llu,\"slot_start_tsc\":%llu,"
                "\"slot_end_tsc\":%llu}\n",
                slot, gate_a_slot_token(slot), starts[slot], ends[slot], count,
                frequency, gate_a_measurement_mode(), gate_a_observation_kind(),
                i_value, q_value, magnitude, floor,
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
    uint64_t *requested_sample_index = NULL;
    uint64_t *requested_tsc = NULL;
    uint64_t *started_tsc = NULL;
    uint64_t *finished_tsc = NULL;
    int *requested_slot_index = NULL;
    uint64_t *scheduler_lateness_ticks = NULL;
    uint64_t *service_ticks = NULL;
    uint64_t *finish_gap_ticks = NULL;
    uint64_t *missed_deadlines_before_sample = NULL;
    int *valid_measurement = NULL;
    int *timing_slot_index = NULL;
    double *observations = NULL;
    uint64_t *response_bank = NULL;
    volatile uint64_t *occupancy_bank = NULL;
    GateASender epochs[4];
    int starts[16], ends[16];
    memset(epochs, 0, sizeof(epochs));
    for (int slot = 0; slot < 16; slot++) {
        starts[slot] = -1;
        ends[slot] = -1;
    }
    char path[CP_PATH_MAX];
    if (!args || !result || !args->output_dir || !args->authority_sha256 ||
        !args->execution_bundle_sha256 || args->sender_core != 4 ||
        args->receiver_core != 5 ||
        args->slot_s != 0.5 || args->temperature_veto_c != 68.0 ||
        args->required_frequency_khz != 1600000 ||
        args->pilot_variant < GATE_A_PILOT_PN ||
        args->pilot_variant > GATE_A_PILOT_CODED_PREPROJECTION_WARM_PHASE_LOCAL_LOOP) {
        return 2;
    }
    gate_a_pilot_variant = args->pilot_variant;
    long expected_read_hz = gate_a_variant_is_readonly_occupancy(args->pilot_variant)
        ? GATE_A_READONLY_MICRO_READ_HZ : 8000L;
    if (args->read_hz != expected_read_hz ||
        !gate_a_occupancy_geometry_valid()) {
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
    result->slot_count = gate_a_slot_count();
    result->step_sender_epoch_count = gate_a_readonly_occupancy_pilot()
        ? 1 : (gate_a_pilot_variant == GATE_A_PILOT_STEP_SHAM
            ? 0 : (gate_a_split_step() ? 2 : 1));
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
    if (gate_a_readonly_occupancy_pilot()) {
        int first_slot = gate_a_readonly_stimulus_first_slot();
        int end_slot = gate_a_readonly_stimulus_end_slot();
        uint64_t readonly_origin = gate_a_epoch_origin(first_slot, mapping_origin,
                                                       args->slot_s, tsc_hz);
        int mapping_failed = gate_a_cycle_state(
            first_slot, readonly_origin, mapping_origin, args->slot_s, tsc_hz) !=
                gate_a_expected_origin_state(first_slot);
        for (int slot = first_slot + 1; slot < end_slot; slot++) {
            if (gate_a_epoch_origin(slot, mapping_origin, args->slot_s, tsc_hz) !=
                readonly_origin) {
                mapping_failed = 1;
            }
        }
        if (mapping_failed) {
            reason = "PHYSICAL_MAPPING_FAILURE";
            rc = 4;
            goto cleanup;
        }
    } else {
        uint64_t s0e_origin = gate_a_epoch_origin(6, mapping_origin,
                                                  args->slot_s, tsc_hz);
        if (gate_a_epoch_origin(7, mapping_origin, args->slot_s, tsc_hz) != s0e_origin ||
            gate_a_cycle_state(6, s0e_origin, mapping_origin, args->slot_s, tsc_hz) !=
                gate_a_expected_origin_state(6) ||
            (gate_a_split_step() && gate_a_cycle_state(
                8, gate_a_epoch_origin(8, mapping_origin, args->slot_s, tsc_hz),
                mapping_origin, args->slot_s, tsc_hz) !=
                    gate_a_expected_origin_state(8)) ||
            (gate_a_driven_slot(12) && gate_a_cycle_state(
                12, gate_a_epoch_origin(12, mapping_origin, args->slot_s, tsc_hz),
                mapping_origin, args->slot_s, tsc_hz) !=
                    gate_a_expected_origin_state(12)) ||
            (gate_a_driven_slot(13) && gate_a_cycle_state(
                13, gate_a_epoch_origin(13, mapping_origin, args->slot_s, tsc_hz),
                mapping_origin, args->slot_s, tsc_hz) !=
                    gate_a_expected_origin_state(13))) {
            reason = "PHYSICAL_MAPPING_FAILURE";
            rc = 4;
            goto cleanup;
        }
    }
    if (gate_a_occupancy_pilot()) {
        size_t response_lines =
            GATE_A_RESPONSE_BUFFER_BYTES / GATE_A_CACHE_LINE_BYTES;
        size_t response_words =
            GATE_A_RESPONSE_BUFFER_BYTES / sizeof(uint64_t);
        size_t occupancy_words =
            GATE_A_OCCUPANCY_LARGE_BYTES / sizeof(uint64_t);
        size_t words_per_line =
            GATE_A_CACHE_LINE_BYTES / sizeof(uint64_t);
        response_bank = aligned_alloc(
            GATE_A_CACHE_LINE_BYTES, GATE_A_RESPONSE_BUFFER_BYTES);
        occupancy_bank = aligned_alloc(
            GATE_A_CACHE_LINE_BYTES, GATE_A_OCCUPANCY_LARGE_BYTES);
        if (!response_bank || !occupancy_bank) {
            reason = "CACHE_RESPONSE_BUFFER_ALLOCATION_FAILURE";
            rc = 5;
            goto cleanup;
        }
        memset(response_bank, 0, response_words * sizeof(*response_bank));
        memset((void *)occupancy_bank, 0,
               occupancy_words * sizeof(*occupancy_bank));
        for (size_t line = 0; line < response_lines; line++) {
            response_bank[line * words_per_line] =
                (line + GATE_A_RESPONSE_LINE_STRIDE) & (response_lines - 1U);
        }
        for (size_t word = 0; word < occupancy_words; word++) {
            occupancy_bank[word] =
                UINT64_C(0x9e3779b97f4a7c15) ^ (uint64_t)word;
        }
        {
            volatile uint64_t prefault_sink =
                small_wall_prefault_buffer(
                    response_bank, GATE_A_RESPONSE_BUFFER_BYTES) ^
                small_wall_prefault_buffer(
                    occupancy_bank, GATE_A_OCCUPANCY_LARGE_BYTES);
            (void)prefault_sink;
            result->occupancy_prefaulted = 1;
        }
        if (hash_captured(
                (const unsigned char *)(const void *)occupancy_bank,
                GATE_A_OCCUPANCY_LARGE_BYTES,
                result->occupancy_digest_before)) {
            reason = "OCCUPANCY_PRE_DIGEST_FAILURE";
            rc = 5;
            goto cleanup;
        }
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

    int cap = (int)(args->read_hz * gate_a_duration_s(args)) + 32;
    timestamps = calloc((size_t)cap, sizeof(*timestamps));
    requested_sample_index = calloc((size_t)cap, sizeof(*requested_sample_index));
    requested_tsc = calloc((size_t)cap, sizeof(*requested_tsc));
    observations = calloc((size_t)cap, sizeof(*observations));
    started_tsc = calloc((size_t)cap, sizeof(*started_tsc));
    finished_tsc = calloc((size_t)cap, sizeof(*finished_tsc));
    requested_slot_index = calloc((size_t)cap, sizeof(*requested_slot_index));
    scheduler_lateness_ticks =
        calloc((size_t)cap, sizeof(*scheduler_lateness_ticks));
    service_ticks = calloc((size_t)cap, sizeof(*service_ticks));
    finish_gap_ticks = calloc((size_t)cap, sizeof(*finish_gap_ticks));
    missed_deadlines_before_sample =
        calloc((size_t)cap, sizeof(*missed_deadlines_before_sample));
    valid_measurement = calloc((size_t)cap, sizeof(*valid_measurement));
    timing_slot_index = calloc((size_t)cap, sizeof(*timing_slot_index));
    if (!timestamps || !requested_sample_index || !requested_tsc ||
        !observations || !started_tsc || !finished_tsc ||
        !requested_slot_index || !scheduler_lateness_ticks ||
        !service_ticks || !finish_gap_ticks ||
        !missed_deadlines_before_sample || !valid_measurement ||
        !timing_slot_index) {
        reason = "OOM";
        rc = 5;
        goto cleanup;
    }
    uint64_t origin = mock ? 1000000000ULL :
        rdtsc_now() + (uint64_t)(START_GUARD_SECONDS * tsc_hz);
    int count = 0;
    uint64_t receiver_epoch = 0;
    if (mock) {
        count = (int)(args->read_hz * gate_a_duration_s(args));
        double spacing = tsc_hz / args->read_hz;
        int first_step_slot = gate_a_readonly_occupancy_pilot()
            ? gate_a_readonly_stimulus_first_slot() : 6;
        int first_step_end = gate_a_split_step() ? 8 : gate_a_step_end_slot();
        if (gate_a_mock_epoch(&epochs[0], lifecycle, args->sender_core,
                              tsc_hz, args->slot_s, origin,
                              first_step_slot, first_step_end,
                              gate_a_phase_index(first_step_slot),
                              gate_a_sign(first_step_slot),
                              gate_a_epoch(first_step_slot)) ||
            (!gate_a_readonly_occupancy_pilot() &&
             gate_a_split_step() && gate_a_mock_epoch(
                              &epochs[1], lifecycle, args->sender_core,
                              tsc_hz, args->slot_s, origin, 8, 10,
                              gate_a_phase_index(8), gate_a_sign(8), gate_a_epoch(8))) ||
            (!gate_a_readonly_occupancy_pilot() &&
             gate_a_mock_epoch(&epochs[2], lifecycle, args->sender_core,
                               tsc_hz, args->slot_s, origin, 12, 13,
                               gate_a_phase_index(12), gate_a_sign(12), gate_a_epoch(12))) ||
            (!gate_a_readonly_occupancy_pilot() &&
             gate_a_mock_epoch(&epochs[3], lifecycle, args->sender_core,
                               tsc_hz, args->slot_s, origin, 13, 14,
                               gate_a_phase_index(13), gate_a_sign(13), gate_a_epoch(13)))) {
            reason = "MOCK_LIFECYCLE_EVIDENCE_FAILURE";
            rc = 4;
            goto cleanup;
        }
        receiver_epoch = origin;
        uint64_t previous_finished = 0;
        int samples_per_slot = (int)(args->read_hz * args->slot_s + 0.5);
        for (int i = 0; i < count; i++) {
            uint64_t requested = origin + (uint64_t)(i * spacing);
            timestamps[i] = requested;
            requested_sample_index[i] = (uint64_t)i;
            requested_tsc[i] = requested;
            observations[i] = 100.0 + (double)(i % 17) * 0.001;
            int slot = (int)((double)(timestamps[i] - origin) /
                              (args->slot_s * tsc_hz));
            requested_slot_index[i] = slot;
            if (slot >= 0 && slot < gate_a_slot_count() &&
                gate_a_driven_slot(slot)) {
                uint64_t epoch_origin = gate_a_epoch_origin(
                    slot, origin, args->slot_s, tsc_hz);
                double phase = 2.0 * M_PI * gate_a_phase_index(slot) / 8.0;
                double seconds = (double)(timestamps[i] - epoch_origin) / tsc_hz;
                observations[i] += 0.25 * cos(2.0 * M_PI * tone(0) * seconds + phase);
            }
            uint64_t service = (uint64_t)(observations[i] *
                                          GATE_A_RESPONSE_TOUCHES + 0.5);
            started_tsc[i] = timestamps[i];
            finished_tsc[i] = started_tsc[i] + service;
            scheduler_lateness_ticks[i] = 0;
            service_ticks[i] = service;
            missed_deadlines_before_sample[i] = 0;
            valid_measurement[i] = 1;
            finish_gap_ticks[i] =
                i ? finished_tsc[i] - previous_finished : 0;
            timing_slot_index[i] =
                samples_per_slot > 0 ? i / samples_per_slot : -1;
            previous_finished = finished_tsc[i];
        }
    } else {
        count = gate_a_run_real_capture(
            args, tsc_hz, origin, timestamps, observations,
            requested_sample_index, requested_tsc,
            started_tsc, finished_tsc, requested_slot_index,
            scheduler_lateness_ticks, service_ticks, finish_gap_ticks,
            missed_deadlines_before_sample, valid_measurement,
            timing_slot_index,
            response_bank, occupancy_bank, cap, lifecycle, epochs,
            &receiver_epoch, result);
        if (count < 0) {
            reason = "SENDER_OR_RECEIVER_LIFECYCLE_FAILURE";
            rc = 4;
            goto cleanup;
        }
    }
    int first_stimulus_slot = gate_a_readonly_occupancy_pilot()
        ? gate_a_readonly_stimulus_first_slot() : 6;
    if (gate_a_driven_slot(first_stimulus_slot) &&
        gate_a_validate_epoch(&epochs[0])) {
        reason = "SENDER_EPOCH_CUSTODY_FAILURE";
        rc = 4;
        goto cleanup;
    }
    if (!gate_a_readonly_occupancy_pilot() &&
        gate_a_split_step() && gate_a_validate_epoch(&epochs[1])) {
        reason = "SENDER_EPOCH_CUSTODY_FAILURE";
        rc = 4;
        goto cleanup;
    }
    if (!gate_a_readonly_occupancy_pilot() &&
        gate_a_pilot_variant != GATE_A_PILOT_ANCHOR_SHAM &&
        (gate_a_validate_epoch(&epochs[2]) ||
         gate_a_validate_epoch(&epochs[3]))) {
        reason = "SENDER_EPOCH_CUSTODY_FAILURE";
        rc = 4;
        goto cleanup;
    }
    uint64_t deadline = origin + (uint64_t)(gate_a_duration_s(args) * tsc_hz);
    while (count > 0 && timestamps[count - 1] >= deadline) count--;
    if (count > 0 && timestamps[0] < origin) {
        reason = "CAPTURE_PRECEDES_ORIGIN";
        rc = 5;
        goto cleanup;
    }
    if (count > 0) {
        int samples_per_slot = (int)(args->read_hz * args->slot_s + 0.5);
        if (samples_per_slot <= 0) {
            reason = "SLOT_SAMPLE_GEOMETRY_FAILURE";
            rc = 5;
            goto cleanup;
        }
        result->capture_origin_tsc = origin;
        result->capture_deadline_tsc = deadline;
        result->capture_first_sample_tsc = timestamps[0];
        result->capture_last_sample_tsc = timestamps[count - 1];
        result->capture_tsc_hz = tsc_hz;
        result->sample_count = count;
        for (int i = 0; i < count; i++) {
            if (raw_record(raw, timestamps[i], observations[i])) {
                reason = "RAW_WRITER_FAILURE";
                rc = 5;
                goto cleanup;
            }
            int slot = gate_a_occupancy_pilot()
                ? timing_slot_index[i] : i / samples_per_slot;
            if (slot >= 0 && slot < gate_a_slot_count()) {
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
    if (!gate_a_occupancy_pilot()) {
        gate_a_synthesize_legacy_sample_timing(
            args, timestamps, count, origin, tsc_hz,
            requested_sample_index, requested_tsc, requested_slot_index,
            started_tsc, finished_tsc, timing_slot_index,
            scheduler_lateness_ticks, service_ticks, finish_gap_ticks,
            missed_deadlines_before_sample, valid_measurement);
    }
    if (gate_a_write_sample_timing_file(
            args->output_dir, requested_sample_index, requested_tsc,
            requested_slot_index, started_tsc, finished_tsc,
            timing_slot_index, scheduler_lateness_ticks, service_ticks,
            missed_deadlines_before_sample, valid_measurement, count)) {
        reason = "SAMPLE_TIMING_WRITER_FAILURE";
        rc = 5;
        goto cleanup;
    }
    gate_a_retain_sample_timing_summary(
        args, result, scheduler_lateness_ticks, finish_gap_ticks,
        missed_deadlines_before_sample, count);
    if (count < (int)(0.9 * args->read_hz * gate_a_duration_s(args))) {
        reason = "SHORT_COMPLETE_CAPTURE";
        rc = 5;
        goto cleanup;
    }
    for (int slot = 0; slot < gate_a_slot_count(); slot++) {
        if (starts[slot] < 0 || ends[slot] <= starts[slot] ||
            (slot == 0 ? starts[slot] != 0 : starts[slot] != ends[slot - 1])) {
            reason = "RAW_SLOT_RANGE_FAILURE";
            rc = 5;
            goto cleanup;
        }
    }
    if (ends[gate_a_slot_count() - 1] != count || gate_a_write_lockin(
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
    if (!mock && !gate_a_readonly_occupancy_pilot()) {
        const char *quality = catcas_capture_quality_failure(
            timestamps[0], timestamps[count - 1], (size_t)count,
            origin, deadline, tsc_hz, args->read_hz, tone(0), max_gap);
        if (quality) {
            if (!strcmp(quality, "PATHOLOGICAL_TIMESTAMP_GAP")) {
                if (gate_a_copy_classification(
                        result->capture_quality_classification,
                        "CAPTURE_ACCEPTED_WITH_SERVICE_SPIKES")) {
                    reason = "CAPTURE_CLASSIFICATION_COPY_FAILURE";
                    rc = 5;
                    goto cleanup;
                }
            } else {
                reason = quality;
                rc = 5;
                goto cleanup;
            }
        } else if (gate_a_copy_classification(
                       result->capture_quality_classification,
                       "CAPTURE_ACCEPTED")) {
            reason = "CAPTURE_CLASSIFICATION_COPY_FAILURE";
            rc = 5;
            goto cleanup;
        }
    }
    for (int slot = 0; slot < gate_a_slot_count(); slot++) {
        uint64_t requested = origin +
            (uint64_t)(slot * args->slot_s * tsc_hz);
        uint64_t requested_end = origin +
            (uint64_t)((slot + 1) * args->slot_s * tsc_hz);
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
                "\"measurement_mode\":\"%s\","
                "\"observation_kind\":\"%s\","
                "\"drive_on\":%s,\"amplitude_level\":",
                slot, gate_a_slot_token(slot), slot * args->slot_s,
                (slot + 1) * args->slot_s,
                (unsigned long long)requested, (unsigned long long)actual,
                result->slot_sample_counts[slot], gate_a_measurement_mode(),
                gate_a_observation_kind(),
                driven ? "true" : "false") < 0) {
            reason = "TRACE_WRITER_FAILURE";
            rc = 5;
            goto cleanup;
        }
        GateASender *geometry_sender = NULL;
        if (slot >= gate_a_readonly_stimulus_first_slot() &&
            slot < gate_a_readonly_stimulus_end_slot() &&
            gate_a_driven_slot(slot)) {
            geometry_sender = !gate_a_readonly_occupancy_pilot() &&
                gate_a_split_step() && slot >= 8
                ? &epochs[1] : &epochs[0];
        }
        if (driven) {
            if (fprintf(trace,
                    "2,\"phase_index\":%d,\"sign\":%d,\"orbit_value\":",
                    gate_a_phase_index(slot), gate_a_sign(slot)) < 0) {
                reason = "TRACE_WRITER_FAILURE";
                rc = 5;
                goto cleanup;
            }
            int orbit_value = gate_a_orbit_value(slot);
            if ((orbit_value >= 0 && fprintf(trace, "%d", orbit_value) < 0) ||
                (orbit_value < 0 && fputs("null", trace) < 0) ||
                fprintf(trace, ",\"working_set_bytes\":%zu,"
                        "\"touch_count\":%zu,\"requested_slot_start_tsc\":%llu,"
                        "\"requested_slot_end_tsc\":%llu,"
                        "\"burst_start_tsc\":%llu,"
                        "\"burst_finish_tsc\":%llu,"
                        "\"burst_duration_ticks\":%llu,"
                        "\"initial_cursor\":%zu,"
                        "\"final_cursor\":%zu,"
                        "\"completed_before_slot_end\":%s,"
                        "\"sender_epoch_id\":\"%s\"}\n",
                        gate_a_occupancy_bytes(slot),
                        geometry_sender ? geometry_sender->occupancy_touch_count[slot] : 0U,
                        (unsigned long long)requested,
                        (unsigned long long)requested_end,
                        (unsigned long long)(geometry_sender ? geometry_sender->burst_start_tsc[slot] : 0U),
                        (unsigned long long)(geometry_sender ? geometry_sender->burst_finish_tsc[slot] : 0U),
                        (unsigned long long)(
                            geometry_sender &&
                            geometry_sender->burst_finish_tsc[slot] >=
                                geometry_sender->burst_start_tsc[slot]
                                ? geometry_sender->burst_finish_tsc[slot] -
                                  geometry_sender->burst_start_tsc[slot]
                                : 0U),
                        geometry_sender ? geometry_sender->occupancy_initial_cursor[slot] : 0U,
                        geometry_sender ? geometry_sender->occupancy_final_cursor[slot] : 0U,
                        geometry_sender &&
                            geometry_sender->burst_start_tsc[slot] >= requested &&
                            geometry_sender->burst_finish_tsc[slot] <= requested_end
                                ? "true" : "false",
                        gate_a_epoch(slot)) < 0) {
                reason = "TRACE_WRITER_FAILURE";
                rc = 5;
                goto cleanup;
            }
        } else if (fputs("null,\"phase_index\":null,\"sign\":null,\"orbit_value\":null,\"working_set_bytes\":0,\"touch_count\":0,\"requested_slot_start_tsc\":null,\"requested_slot_end_tsc\":null,\"burst_start_tsc\":null,\"burst_finish_tsc\":null,\"burst_duration_ticks\":null,\"initial_cursor\":0,\"final_cursor\":0,\"completed_before_slot_end\":false,\"sender_epoch_id\":null}\n", trace) < 0) {
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
        if (slot >= gate_a_readonly_stimulus_first_slot() &&
            slot < gate_a_readonly_stimulus_end_slot() &&
            gate_a_driven_slot(slot)) {
            slot_sender = !gate_a_readonly_occupancy_pilot() &&
                gate_a_split_step() && slot >= 8
                ? &epochs[1] : &epochs[0];
            state = "active";
        } else if (!gate_a_readonly_occupancy_pilot() &&
                   slot == 12 && gate_a_driven_slot(slot)) {
            slot_sender = &epochs[2];
            state = "active";
        } else if (!gate_a_readonly_occupancy_pilot() &&
                   slot == 13 && gate_a_driven_slot(slot)) {
            slot_sender = &epochs[3];
            state = "active";
        } else if (!gate_a_readonly_occupancy_pilot() &&
                   (slot >= gate_a_step_end_slot() && slot <= 11) &&
                   gate_a_driven_slot(6)) {
            slot_sender = gate_a_split_step() ? &epochs[1] : &epochs[0];
            state = "joined";
        } else if (!gate_a_readonly_occupancy_pilot() &&
                   (slot == 14 || slot == 15)) {
            slot_sender = &epochs[3];
            state = "joined";
        } else if (gate_a_readonly_occupancy_pilot() &&
                   slot >= gate_a_step_end_slot() &&
                   gate_a_driven_slot(first_stimulus_slot)) {
            slot_sender = &epochs[0];
            state = "joined";
        }
        if (gate_a_lifecycle_record(
                lifecycle, "slot_transition", state, slot, actual,
                requested, requested_end,
                slot_sender)) {
            reason = "LIFECYCLE_WRITER_FAILURE";
            rc = 5;
            goto cleanup;
        }
    }
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
    if (gate_a_occupancy_pilot()) {
        if (hash_captured(
                (const unsigned char *)(const void *)occupancy_bank,
                GATE_A_OCCUPANCY_LARGE_BYTES,
                result->occupancy_digest_after)) {
            reason = "OCCUPANCY_POST_DIGEST_FAILURE";
            rc = 5;
            goto cleanup;
        }
        result->occupancy_digest_unchanged = !strcmp(
            result->occupancy_digest_before,
            result->occupancy_digest_after);
        if (gate_a_readonly_occupancy_pilot()) {
            for (int slot = gate_a_readonly_stimulus_first_slot();
                 slot < gate_a_readonly_stimulus_end_slot(); slot++) {
                result->occupancy_touch_count[slot] =
                    epochs[0].occupancy_touch_count[slot];
                result->occupancy_initial_cursor[slot] =
                    epochs[0].occupancy_initial_cursor[slot];
                result->occupancy_final_cursor[slot] =
                    epochs[0].occupancy_final_cursor[slot];
                if (!epochs[0].occupancy_slot_complete[slot] ||
                    result->occupancy_touch_count[slot] !=
                        GATE_A_READONLY_SLOT_TOUCHES) {
                    reason = "READONLY_OCCUPANCY_EXACT_COUNT_FAILURE";
                    rc = 5;
                    goto cleanup;
                }
            }
            if (gate_a_retain_readonly_burst_geometry(args, result, epochs)) {
                reason = "READONLY_OCCUPANCY_BURST_EVIDENCE_FAILURE";
                rc = 5;
                goto cleanup;
            }
            if (!result->occupancy_digest_unchanged) {
                reason = "READONLY_OCCUPANCY_BUFFER_MUTATED";
                rc = 5;
                goto cleanup;
            }
            {
                GateATimingDiagnostic diagnostic;
                if (gate_a_readonly_timing_diagnostic(
                        args, result, requested_tsc, started_tsc, finished_tsc,
                        scheduler_lateness_ticks, service_ticks,
                        finish_gap_ticks, missed_deadlines_before_sample,
                        timing_slot_index, count,
                        &diagnostic) ||
                    gate_a_write_timing_diagnostic_summary(
                        args, result, &diagnostic)) {
                    reason = "READONLY_OCCUPANCY_DIAGNOSTIC_FAILURE";
                    rc = 5;
                    goto cleanup;
                }
                if (strncmp(diagnostic.classification,
                            "CAPTURE_REJECTED_", 17U) == 0) {
                    reason = diagnostic.classification;
                    rc = 5;
                    goto cleanup;
                }
            }
        }
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
    for (int epoch = 0; epoch < 4; epoch++) {
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
    free(requested_sample_index);
    free(requested_tsc);
    free(started_tsc);
    free(finished_tsc);
    free(requested_slot_index);
    free(scheduler_lateness_ticks);
    free(service_ticks);
    free(finish_gap_ticks);
    free(missed_deadlines_before_sample);
    free(valid_measurement);
    free(timing_slot_index);
    free(observations);
    free(response_bank);
    free((void *)occupancy_bank);
    if (rc) gate_a_write_failure(args->output_dir, reason);
    (void)receiver_epoch;
    return rc;
}
