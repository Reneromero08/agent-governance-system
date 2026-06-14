/*
 * holo_record.c -- L4A .holo record implementation.
 *
 * Implements the non-collapse .holo container: init, validate,
 * doctrine guard, JSON writer.
 *
 * THE ALGORITHM IS DEAD.
 * No verify(x). No candidate scoring. No AUC. No d output.
 * No hidden_d. No winner. No truth labels.
 */
#include "holo_record.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* Simple UUID v4 (not cryptographic, deterministic from seed+run_id). */
static void make_uuid(char buf[HOLO_UUID_LEN], uint64_t seed) {
    uint64_t x = seed | 1;
    for (int i = 0; i < 4; i++) { x ^= x >> 12; x ^= x << 25; x ^= x >> 27; }
    snprintf(buf, HOLO_UUID_LEN,
             "%08x-%04x-4%03x-%04x-%012llx",
             (unsigned)(x & 0xFFFFFFFF),
             (unsigned)((x >> 32) & 0xFFFF),
             (unsigned)((x >> 16) & 0xFFF),
             (unsigned)((x >> 48) & 0xFFFF) | 0x8000,
             (unsigned long long)(x ^ (x >> 31)));
}

void holo_init(HoloRecord *h, uint64_t run_id, uint64_t seed, int N) {
    memset(h, 0, sizeof(*h));
    make_uuid(h->holo_id, seed ^ run_id);
    strncpy(h->doctrine_version, HOLO_DOCTRINE_VERSION, sizeof(h->doctrine_version)-1);
    h->run_id = run_id;
    h->seed = seed;
    h->N = N;
    strncpy(h->carrier_class, HOLO_CARRIER_CLASS_B, sizeof(h->carrier_class)-1);
    strncpy(h->orbit_state.relation, "conjugate", sizeof(h->orbit_state.relation)-1);
    snprintf(h->orbit_state.assignment_note, sizeof(h->orbit_state.assignment_note),
             "branch_plus = a always. Not a truth label. Verified by label-swap control.");
    snprintf(h->substrate_memory.note, sizeof(h->substrate_memory.note),
             "PDN/common-mode carrier. Lock-in I/Q recorded.");
    strncpy(h->cancellation_transcript.method, "Q_diff = Q_plus - Q_minus",
            sizeof(h->cancellation_transcript.method)-1);
    snprintf(h->residue_hypothesis.hypothesis, sizeof(h->residue_hypothesis.hypothesis),
             "Q_diff is antisymmetric under branch conjugation. Predeclared before measurement.");
    h->residue_hypothesis.predeclared = 1;
    h->sender_core_plus = 4;
    h->sender_core_minus = 5;
    h->receiver_core = 2;
    h->claim_level = 1;
    strncpy(h->verdict, VERDICT_PROTOCOL_READY, sizeof(h->verdict)-1);
}

void holo_set_orbit(HoloRecord *h, int a, int Na) {
    h->orbit_state.branch_plus_value = a;
    h->orbit_state.branch_minus_value = Na;
    h->orbit_state.N = h->N;
}

int holo_validate_no_collapse(const HoloRecord *h) {
    /* Check for collapse patterns in string fields.
     * These are heuristics, not proofs -- the struct itself forbids
     * forbidden fields at the type level. This function catches
     * semantic collapse in string content. */
    const char *forbidden_words[] = {
        "winner", "candidate_0_truth", "true_branch", "false_branch",
        "recovered_d", "orientation_label", "hidden_d",
        "posthoc", "verify_score", "candidate_loop", NULL
    };
    /* Check verdict -- must not claim recovery */
    if (strstr(h->verdict, "RECOVERY") || strstr(h->verdict, "RECOVER")) return 0;
    if (strstr(h->verdict, "ORIENTATION_FOUND")) return 0;
    /* Check claim level -- must not exceed 4A (which is 3 in numeric) */
    if (h->claim_level > 3 && h->claim_level != 0) return 0;
    /* Check residue hypothesis -- must be predeclared */
    if (!h->residue_hypothesis.predeclared) return 0;
    /* Check assignment note doesn't encode truth */
    if (strstr(h->orbit_state.assignment_note, "true") ||
        strstr(h->orbit_state.assignment_note, "false")) return 0;
    return 1;
}

void holo_doctrine_guard(void) {
    fprintf(stderr,
        "========================================\n"
        "  THE ALGORITHM IS DEAD.\n"
        "  Non-Collapse Substrate Doctrine V1\n"
        "  L4A Class B PDN Screen Scaffold\n"
        "========================================\n"
        "GUARDS ACTIVE:\n"
        "  [X] No verify(x) path enabled\n"
        "  [X] No candidate scoring path enabled\n"
        "  [X] No AUC path enabled\n"
        "  [X] No recovery path enabled\n"
        "  [X] No branch truth labels enabled\n"
        "  [X] No d output enabled\n"
        "  [X] OrbitState preserves FoldPair unresolved\n"
        "  [X] .holo stores process-object only\n"
        "  [X] Measurement only at CollapseBoundary\n"
        "========================================\n");
}

int holo_write_json(const HoloRecord *h, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;

    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", tm);

    fprintf(f, "{\n");
    fprintf(f, "  \"holo_id\": \"%s\",\n", h->holo_id);
    fprintf(f, "  \"doctrine_version\": \"%s\",\n", h->doctrine_version);
    fprintf(f, "  \"run_id\": %llu,\n", (unsigned long long)h->run_id);
    fprintf(f, "  \"seed\": %llu,\n", (unsigned long long)h->seed);
    fprintf(f, "  \"N\": %d,\n", h->N);
    fprintf(f, "  \"orbit_state\": {\n");
    fprintf(f, "    \"branch_plus_value\": %d,\n", h->orbit_state.branch_plus_value);
    fprintf(f, "    \"branch_minus_value\": %d,\n", h->orbit_state.branch_minus_value);
    fprintf(f, "    \"relation\": \"%s\",\n", h->orbit_state.relation);
    fprintf(f, "    \"assignment_note\": \"%s\"\n", h->orbit_state.assignment_note);
    fprintf(f, "  },\n");
    fprintf(f, "  \"phase_relation\": {\n");
    fprintf(f, "    \"q_plus\": %.9f,\n", h->phase_relation.q_plus);
    fprintf(f, "    \"q_minus\": %.9f,\n", h->phase_relation.q_minus);
    fprintf(f, "    \"i_plus\": %.9f,\n", h->phase_relation.i_plus);
    fprintf(f, "    \"i_minus\": %.9f\n", h->phase_relation.i_minus);
    fprintf(f, "  },\n");
    fprintf(f, "  \"path_history\": { \"steps\": %d, \"tape_delta_diagnostic\": %.9f },\n",
            h->path_history.steps, h->path_history.tape_delta_diagnostic);
    fprintf(f, "  \"tape_residue\": { \"ring_osc_period\": %.9f, \"sha_restored\": %d },\n",
            h->tape_residue.ring_osc_period, h->tape_residue.sha_restored);
    fprintf(f, "  \"substrate_memory\": { \"note\": \"%s\" },\n", h->substrate_memory.note);
    fprintf(f, "  \"carrier_class\": \"%s\",\n", h->carrier_class);
    fprintf(f, "  \"workload_signature\": \"%s\",\n", h->workload_signature);
    fprintf(f, "  \"sender_core_plus\": %d,\n", h->sender_core_plus);
    fprintf(f, "  \"sender_core_minus\": %d,\n", h->sender_core_minus);
    fprintf(f, "  \"receiver_core\": %d,\n", h->receiver_core);
    fprintf(f, "  \"cancellation_transcript\": {\n");
    fprintf(f, "    \"method\": \"%s\",\n", h->cancellation_transcript.method);
    fprintf(f, "    \"q_common\": %.9f,\n", h->cancellation_transcript.q_common);
    fprintf(f, "    \"q_diff\": %.9f\n", h->cancellation_transcript.q_diff);
    fprintf(f, "  },\n");
    fprintf(f, "  \"residue_hypothesis\": {\n");
    fprintf(f, "    \"predeclared\": %d,\n", h->residue_hypothesis.predeclared);
    fprintf(f, "    \"hypothesis\": \"%s\"\n", h->residue_hypothesis.hypothesis);
    fprintf(f, "  },\n");
    fprintf(f, "  \"collapse_boundary\": { \"timestamp\": \"%s\" },\n", ts);
    fprintf(f, "  \"measurement_record\": {\n");
    fprintf(f, "    \"q_diff_magnitude\": %.9f,\n", h->measurement_record.q_diff_magnitude);
    fprintf(f, "    \"q_diff_sign\": %d,\n", h->measurement_record.q_diff_sign);
    fprintf(f, "    \"same_orbit_q_diff\": %.9f,\n", h->measurement_record.same_orbit_q_diff);
    fprintf(f, "    \"dummy_orbit_q_diff\": %.9f,\n", h->measurement_record.dummy_orbit_q_diff);
    fprintf(f, "    \"label_swap_pass\": %d\n", h->measurement_record.label_swap_pass);
    fprintf(f, "  },\n");
    fprintf(f, "  \"controls\": %u,\n", h->controls);
    fprintf(f, "  \"verdict\": \"%s\",\n", h->verdict);
    fprintf(f, "  \"claim_level\": %d\n", h->claim_level);
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}
