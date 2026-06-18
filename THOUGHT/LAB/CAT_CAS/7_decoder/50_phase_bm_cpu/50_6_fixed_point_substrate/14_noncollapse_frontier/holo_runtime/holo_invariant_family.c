#include "holo_invariant_family.h"
#include "holo_geometry.h"
#include "../l4b_orbitstate/holo_path_history.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
static const uint64_t FNV_PRIME = UINT64_C(1099511628211);

static void text_copy(char *dst, size_t size, const char *src) {
    snprintf(dst, size, "%s", src);
}

static uint64_t fnv_u64(uint64_t digest, uint64_t value) {
    unsigned int shift;
    for (shift = 0; shift < 64U; shift += 8U) {
        digest = (digest ^ (unsigned char)(value >> shift)) * FNV_PRIME;
    }
    return digest;
}

static uint64_t fnv_text(uint64_t digest, const char *text) {
    while (*text != '\0') digest = (digest ^ (unsigned char)*text++) * FNV_PRIME;
    return digest;
}

static uint64_t double_bits(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static double bits_double(uint64_t bits) {
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

const char *holo_invariant_kind_name(HoloInvariantKind kind) {
    static const char *names[HOLO_INVARIANT_COUNT] = {
        "orbit_conservation", "relation_basis", "path_composition",
        "restoration_closure", "exchange_covariance", "serialization_invariance",
        "path_order", "software_path_holonomy"
    };
    return (kind >= 0 && (size_t)kind < HOLO_INVARIANT_COUNT) ? names[kind] : "invalid";
}

const char *holo_invariant_result_name(HoloInvariantResult result) {
    switch (result) {
        case HOLO_INV_RESULT_PENDING: return "PENDING_ARTIFACT_RELOAD";
        case HOLO_INV_RESULT_PASS: return "PASS";
        case HOLO_INV_RESULT_FAIL: return "FAIL";
        case HOLO_INV_RESULT_DEFERRED_NOT_WELL_DEFINED:
            return "DEFERRED_NOT_WELL_DEFINED";
        default: return "INVALID";
    }
}

static const char *operator_name(HoloInvariantKind kind) {
    static const char *names[HOLO_INVARIANT_COUNT] = {
        "fold_orbit_conservation_v1", "fold_exchange_involution_v1",
        "forward_reverse_identity_v1", "software_restoration_closure_v1",
        "structural_branch_exchange_v1", "reload_recomputation_v1",
        "ordered_path_continuity_v1", "group_carrier_accumulation_unavailable_v1"
    };
    return names[kind];
}

int holo_invariant_family_register(HoloInvariantFamily *family, HoloInvariantKind kind) {
    HoloInvariantRecord *record;
    if (!family || kind < 0 || (size_t)kind >= HOLO_INVARIANT_COUNT ||
        family->sealed || family->extracted || family->count >= HOLO_INVARIANT_COUNT) return -1;
    if ((size_t)kind != family->count) return -2;
    record = &family->records[family->count];
    memset(record, 0, sizeof(*record));
    record->kind = kind;
    text_copy(record->invariant_id, sizeof(record->invariant_id), holo_invariant_kind_name(kind));
    text_copy(record->operator_id, sizeof(record->operator_id), operator_name(kind));
    text_copy(record->declaration_phase, sizeof(record->declaration_phase), "object_initialization");
    text_copy(record->evidence_level, sizeof(record->evidence_level), "software_architectural");
    record->predeclared = 1;
    record->claim_level = 1;
    record->tolerance = (kind == HOLO_INV_RELATION_BASIS) ? 1e-12 : 0.0;
    family->count++;
    return 0;
}

int holo_invariant_family_init(HoloInvariantFamily *family) {
    size_t i;
    if (!family) return -1;
    memset(family, 0, sizeof(*family));
    text_copy(family->family_id, sizeof(family->family_id), "noncollapse_geometry_v1");
    family->predeclared = 1;
    family->claim_level = 1;
    for (i = 0; i < HOLO_INVARIANT_COUNT; ++i) {
        if (holo_invariant_family_register(family, (HoloInvariantKind)i) != 0) return -2;
    }
    return 0;
}

int holo_invariant_family_set_tolerance(HoloInvariantFamily *family,
                                        HoloInvariantKind kind, double tolerance) {
    HoloInvariantRecord *record;
    if (!family || kind < 0 || (size_t)kind >= family->count || tolerance < 0.0 ||
        family->sealed || family->extracted) return -1;
    record = &family->records[kind];
    if (record->evaluated) return -2;
    record->tolerance = tolerance;
    return 0;
}

int holo_invariant_family_set_operator(HoloInvariantFamily *family,
                                       HoloInvariantKind kind, const char *operator_id) {
    HoloInvariantRecord *record;
    if (!family || !operator_id || kind < 0 || (size_t)kind >= family->count ||
        family->sealed || family->extracted) return -1;
    record = &family->records[kind];
    if (record->evaluated) return -2;
    text_copy(record->operator_id, sizeof(record->operator_id), operator_id);
    return 0;
}

int holo_invariant_family_set_result(HoloInvariantFamily *family,
                                     HoloInvariantKind kind, HoloInvariantResult result) {
    HoloInvariantRecord *record;
    if (!family || kind < 0 || (size_t)kind >= family->count ||
        family->sealed || family->extracted) return -1;
    record = &family->records[kind];
    if (record->evaluated) return -2;
    record->result = result;
    return 0;
}

static uint64_t path_digest(const HoloPathHistory *path) {
    uint64_t digest = FNV_OFFSET;
    size_t i;
    digest = fnv_u64(digest, path->initial_state_digest);
    digest = fnv_u64(digest, path->terminal_state_digest);
    digest = fnv_u64(digest, path->count);
    for (i = 0; i < path->count; ++i) digest = fnv_u64(digest, path->steps[i].step_digest);
    return digest;
}

static uint64_t record_digest(uint64_t digest, const HoloInvariantRecord *record) {
    digest = fnv_u64(digest, (uint64_t)record->kind);
    digest = fnv_text(digest, record->invariant_id);
    digest = fnv_text(digest, record->operator_id);
    digest = fnv_text(digest, record->declaration_phase);
    digest = fnv_text(digest, record->equality_rule);
    digest = fnv_text(digest, record->evidence_level);
    digest = fnv_u64(digest, (uint64_t)record->predeclared);
    digest = fnv_u64(digest, (uint64_t)record->evaluated);
    digest = fnv_u64(digest, (uint64_t)record->passed);
    digest = fnv_u64(digest, (uint64_t)record->result);
    digest = fnv_u64(digest, (uint64_t)record->claim_level);
    digest = fnv_u64(digest, double_bits(record->tolerance));
    digest = fnv_u64(digest, record->digest_a);
    digest = fnv_u64(digest, record->digest_b);
    digest = fnv_u64(digest, record->digest_c);
    digest = fnv_u64(digest, double_bits(record->scalar_a));
    digest = fnv_u64(digest, double_bits(record->scalar_b));
    digest = fnv_u64(digest, double_bits(record->scalar_c));
    digest = fnv_u64(digest, (uint64_t)record->flag_a);
    digest = fnv_u64(digest, (uint64_t)record->flag_b);
    digest = fnv_u64(digest, (uint64_t)record->flag_c);
    digest = fnv_u64(digest, (uint64_t)record->flag_d);
    return digest;
}

static uint64_t family_digest(const HoloInvariantFamily *family) {
    uint64_t digest = fnv_text(FNV_OFFSET, family->family_id);
    size_t i;
    digest = fnv_u64(digest, family->path_history_digest);
    digest = fnv_u64(digest, (uint64_t)family->predeclared);
    digest = fnv_u64(digest, (uint64_t)family->claim_level);
    for (i = 0; i < family->count; ++i) {
        if (i == HOLO_INV_SERIALIZATION) continue;
        digest = record_digest(digest, &family->records[i]);
    }
    return digest;
}

static void finish_record(HoloInvariantRecord *record, int passed) {
    record->evaluated = 1;
    record->passed = passed ? 1 : 0;
    record->result = passed ? HOLO_INV_RESULT_PASS : HOLO_INV_RESULT_FAIL;
}

static int reconstruct_states(HoloObject *object, OrbitState *initial,
                              OrbitState *terminal, OrbitState *restored) {
    HoloPathHistory *path = object->evolution.path_history;
    if (!path || path->count == 0) return -1;
    orbit_init(initial, object->N, (int)object->geometry.coordinates[0],
               (int)object->geometry.coordinates[1]);
    *terminal = *initial;
    terminal->acc_real = bits_double(path->steps[path->count - 1U].post_acc_real_bits);
    terminal->acc_imag = bits_double(path->steps[path->count - 1U].post_acc_imag_bits);
    terminal->steps = (int)path->count;
    if (holo_orbit_state_digest(initial) != path->initial_state_digest ||
        holo_orbit_state_digest(terminal) != path->terminal_state_digest) return -2;
    return holo_path_reverse(path, terminal, restored) == HOLO_PATH_OK ? 0 : -3;
}

static int evaluate_order(const HoloPathHistory *path) {
    HoloPathHistory copy = *path;
    HoloPathStep *steps;
    HoloPathStep swap;
    int rejected;
    if (path->count < 2U) return 0;
    steps = (HoloPathStep *)malloc(path->count * sizeof(*steps));
    if (!steps) return 0;
    memcpy(steps, path->steps, path->count * sizeof(*steps));
    copy.steps = steps;
    copy.capacity = path->count;
    swap = steps[0]; steps[0] = steps[1]; steps[1] = swap;
    rejected = holo_path_history_validate(&copy) != HOLO_PATH_OK;
    free(steps);
    return rejected;
}

int holo_invariant_family_evaluate(HoloObject *object) {
    HoloInvariantFamily *family;
    HoloPathHistory *path;
    HoloInvariantRecord *record;
    OrbitState initial, terminal, restored;
    double rendered[2], rerendered[2];
    double det, trace;
    int exchanged_lower, exchanged_mirror;
    int relation_ok, basis_ok, order_sensitive;
    if (!object || !object->collapse_boundary.crossed) return -1;
    family = &object->invariant_family;
    path = object->evolution.path_history;
    if (!family->predeclared || family->count != HOLO_INVARIANT_COUNT ||
        family->sealed || !path || reconstruct_states(object, &initial, &terminal, &restored) != 0) return -2;

    record = &family->records[HOLO_INV_ORBIT_CONSERVATION];
    record->scalar_a = initial.branch_plus + initial.branch_minus;
    record->scalar_b = terminal.branch_plus + terminal.branch_minus;
    record->scalar_c = restored.branch_plus + restored.branch_minus;
    record->digest_a = (uint64_t)initial.branch_plus * (uint64_t)initial.branch_minus;
    record->digest_b = (uint64_t)terminal.branch_plus * (uint64_t)terminal.branch_minus;
    record->digest_c = (uint64_t)restored.branch_plus * (uint64_t)restored.branch_minus;
    record->flag_a = record->scalar_a == object->N && record->scalar_b == object->N &&
                     record->scalar_c == object->N;
    record->flag_b = record->digest_a == record->digest_b && record->digest_a == record->digest_c;
    finish_record(record, record->flag_a && record->flag_b);

    record = &family->records[HOLO_INV_RELATION_BASIS];
    det = object->geometry.relation_basis[0] * object->geometry.relation_basis[3] -
          object->geometry.relation_basis[1] * object->geometry.relation_basis[2];
    trace = object->geometry.relation_basis[0] + object->geometry.relation_basis[3];
    basis_ok = strcmp(object->geometry.geometry_type, "fold_orbit_relation") == 0 &&
               fabs(object->geometry.relation_basis[0]) <= record->tolerance &&
               fabs(object->geometry.relation_basis[1] - 1.0) <= record->tolerance &&
               fabs(object->geometry.relation_basis[2] - 1.0) <= record->tolerance &&
               fabs(object->geometry.relation_basis[3]) <= record->tolerance &&
               fabs(object->geometry.neutral_reference[0] - (double)object->N / 2.0) <= record->tolerance &&
               fabs(object->geometry.neutral_reference[1] - (double)object->N / 2.0) <= record->tolerance;
    record->flag_a = basis_ok;
    record->flag_b = fabs(det + 1.0) <= record->tolerance;
    record->flag_c = fabs(trace) <= record->tolerance;
    record->flag_d = holo_geometry_render(&object->geometry, rendered) == 0;
    if (record->flag_d) {
        HoloGeometry exchanged = object->geometry;
        exchanged.coordinates[0] = rendered[0]; exchanged.coordinates[1] = rendered[1];
        record->flag_d = holo_geometry_render(&exchanged, rerendered) == 0 &&
                         fabs(rerendered[0] - object->geometry.coordinates[0]) <= record->tolerance &&
                         fabs(rerendered[1] - object->geometry.coordinates[1]) <= record->tolerance;
    }
    record->scalar_a = det; record->scalar_b = trace;
    finish_record(record, record->flag_a && record->flag_b && record->flag_c && record->flag_d);

    record = &family->records[HOLO_INV_PATH_COMPOSITION];
    record->digest_a = path->initial_state_digest;
    record->digest_b = path->terminal_state_digest;
    record->digest_c = path->restored_state_digest;
    record->flag_a = holo_orbit_state_equal_bitwise(&initial, &restored);
    record->flag_b = strcmp(object->evolution.operator_id, HOLO_PATH_OPERATOR) == 0;
    text_copy(record->equality_rule, sizeof(record->equality_rule), "bitwise_numeric_orbit_state");
    text_copy(record->evidence_level, sizeof(record->evidence_level), "executed_reverse_path");
    finish_record(record, record->flag_a && record->flag_b &&
                  record->digest_a == record->digest_c);

    record = &family->records[HOLO_INV_RESTORATION_CLOSURE];
    record->flag_a = 1;
    record->flag_b = path->restoration_verified;
    record->scalar_a = object->restoration.restoration_metric;
    text_copy(record->equality_rule, sizeof(record->equality_rule),
              "inverse_path_reconstructs_initial_orbit_state");
    text_copy(record->evidence_level, sizeof(record->evidence_level),
              "software_path_roundtrip:dedicated_verification_copy");
    finish_record(record, record->flag_b && object->restoration.restored);

    record = &family->records[HOLO_INV_EXCHANGE_COVARIANCE];
    exchanged_lower = initial.branch_minus;
    exchanged_mirror = initial.branch_plus;
    relation_ok = initial.branch_plus + initial.branch_minus ==
                  exchanged_lower + exchanged_mirror &&
                  (uint64_t)initial.branch_plus * (uint64_t)initial.branch_minus ==
                  (uint64_t)exchanged_lower * (uint64_t)exchanged_mirror;
    record->scalar_a = initial.branch_plus + initial.branch_minus;
    record->scalar_b = exchanged_lower + exchanged_mirror;
    record->digest_a = (uint64_t)initial.branch_plus * (uint64_t)initial.branch_minus;
    record->digest_b = (uint64_t)exchanged_lower * (uint64_t)exchanged_mirror;
    record->flag_a = relation_ok;
    record->flag_b = rendered[0] == object->geometry.coordinates[1] &&
                     rendered[1] == object->geometry.coordinates[0];
    record->flag_c = record->flag_a && record->flag_b;
    record->flag_d = 1;
    finish_record(record, record->flag_c);

    record = &family->records[HOLO_INV_SERIALIZATION];
    record->evaluated = 1;
    record->passed = 0;
    record->result = HOLO_INV_RESULT_PENDING;
    text_copy(record->evidence_level, sizeof(record->evidence_level), "artifact_reload_recomputation");

    order_sensitive = evaluate_order(path);
    record = &family->records[HOLO_INV_PATH_ORDER];
    record->flag_a = holo_path_history_validate(path) == HOLO_PATH_OK;
    record->flag_b = 0;
    record->flag_c = order_sensitive;
    finish_record(record, record->flag_a && record->flag_c);

    record = &family->records[HOLO_INV_SOFTWARE_HOLONOMY];
    record->evaluated = 1;
    record->passed = 0;
    record->result = HOLO_INV_RESULT_DEFERRED_NOT_WELL_DEFINED;
    text_copy(record->evidence_level, sizeof(record->evidence_level),
              "missing_group_valued_carrier_transform_per_path_step");

    family->path_history_digest = path_digest(path);
    family->extracted = 1;
    family->sealed = 1;
    family->family_digest = family_digest(family);
    return holo_invariant_family_validate(family) ? 0 : -3;
}

int holo_invariant_family_mark_serialization(HoloInvariantFamily *family,
                                             uint64_t reloaded_digest,
                                             uint64_t recomputed_digest) {
    HoloInvariantRecord *record;
    if (!family || !family->sealed || !family->extracted) return -1;
    record = &family->records[HOLO_INV_SERIALIZATION];
    if (record->result == HOLO_INV_RESULT_PASS) return -2;
    record->digest_a = family->family_digest;
    record->digest_b = reloaded_digest;
    record->digest_c = recomputed_digest;
    record->flag_a = record->digest_a == record->digest_b;
    record->flag_b = record->digest_a == record->digest_c;
    finish_record(record, record->flag_a && record->flag_b);
    return record->passed ? 0 : -3;
}

int holo_invariant_family_validate(const HoloInvariantFamily *family) {
    size_t i;
    if (!family || family->count != HOLO_INVARIANT_COUNT || !family->predeclared ||
        strcmp(family->family_id, "noncollapse_geometry_v1") != 0 ||
        family->claim_level > 2) return 0;
    for (i = 0; i < family->count; ++i) {
        const HoloInvariantRecord *record = &family->records[i];
        if (record->kind != (HoloInvariantKind)i || !record->predeclared ||
            record->claim_level > 2 || strcmp(record->invariant_id, holo_invariant_kind_name(record->kind)) != 0 ||
            strcmp(record->operator_id, operator_name(record->kind)) != 0) return 0;
        if (family->extracted && !record->evaluated) return 0;
        if (family->extracted && i != HOLO_INV_SERIALIZATION && i != HOLO_INV_SOFTWARE_HOLONOMY &&
            (!record->passed || record->result != HOLO_INV_RESULT_PASS)) return 0;
        if (i == HOLO_INV_SOFTWARE_HOLONOMY && family->extracted &&
            record->result != HOLO_INV_RESULT_DEFERRED_NOT_WELL_DEFINED) return 0;
    }
    if (family->extracted && (!family->sealed || family->family_digest != family_digest(family))) return 0;
    return 1;
}

int holo_invariant_family_equal(const HoloInvariantFamily *left,
                                const HoloInvariantFamily *right,
                                int ignore_serialization_state) {
    size_t i;
    if (!left || !right || left->count != right->count ||
        left->path_history_digest != right->path_history_digest ||
        left->family_digest != right->family_digest) return 0;
    for (i = 0; i < left->count; ++i) {
        const HoloInvariantRecord *a = &left->records[i];
        const HoloInvariantRecord *b = &right->records[i];
        if (ignore_serialization_state && i == HOLO_INV_SERIALIZATION) continue;
        if (memcmp(a, b, sizeof(*a)) != 0) return 0;
    }
    return 1;
}

static void write_result(FILE *file, const HoloInvariantRecord *r) {
    switch (r->kind) {
        case HOLO_INV_ORBIT_CONSERVATION:
            fprintf(file, ",\"orbit_sum_initial\":%.0f,\"orbit_sum_terminal\":%.0f,\"orbit_sum_restored\":%.0f,\"orbit_product_initial\":%" PRIu64 ",\"orbit_product_terminal\":%" PRIu64 ",\"orbit_product_restored\":%" PRIu64 ",\"fold_relation_preserved\":%s,\"product_preserved\":%s",
                    r->scalar_a, r->scalar_b, r->scalar_c, r->digest_a, r->digest_b, r->digest_c,
                    r->flag_a ? "true" : "false", r->flag_b ? "true" : "false"); break;
        case HOLO_INV_RELATION_BASIS:
            fprintf(file, ",\"determinant\":%.17g,\"trace\":%.17g,\"basis_matches_exchange\":%s,\"det_minus_one\":%s,\"trace_zero\":%s,\"render_involution\":%s",
                    r->scalar_a, r->scalar_b, r->flag_a ? "true" : "false", r->flag_b ? "true" : "false",
                    r->flag_c ? "true" : "false", r->flag_d ? "true" : "false"); break;
        case HOLO_INV_PATH_COMPOSITION:
            fprintf(file, ",\"initial_state_digest\":\"%016" PRIx64 "\",\"terminal_state_digest\":\"%016" PRIx64 "\",\"restored_state_digest\":\"%016" PRIx64 "\",\"composition_identity\":%s,\"evolution_operator_consistent\":%s",
                    r->digest_a, r->digest_b, r->digest_c, r->flag_a ? "true" : "false",
                    r->flag_b ? "true" : "false"); break;
        case HOLO_INV_RESTORATION_CLOSURE:
            fprintf(file, ",\"restoration_attempted\":%s,\"restoration_verified\":%s,\"restoration_metric\":%.17g",
                    r->flag_a ? "true" : "false", r->flag_b ? "true" : "false", r->scalar_a); break;
        case HOLO_INV_EXCHANGE_COVARIANCE:
            fprintf(file, ",\"original_neutral_sum\":%.0f,\"exchanged_neutral_sum\":%.0f,\"original_neutral_product\":%" PRIu64 ",\"exchanged_neutral_product\":%" PRIu64 ",\"neutral_invariants_equal\":%s,\"indexed_quantities_exchanged\":%s,\"exchange_covariance\":%s,\"branch_neutral\":%s",
                    r->scalar_a, r->scalar_b, r->digest_a, r->digest_b, r->flag_a ? "true" : "false",
                    r->flag_b ? "true" : "false", r->flag_c ? "true" : "false", r->flag_d ? "true" : "false"); break;
        case HOLO_INV_SERIALIZATION:
            fprintf(file, ",\"pre_serialization_digest\":\"%016" PRIx64 "\",\"post_reload_digest\":\"%016" PRIx64 "\",\"recomputed_digest\":\"%016" PRIx64 "\",\"reloaded_equal\":%s,\"recomputed_equal\":%s",
                    r->digest_a, r->digest_b, r->digest_c, r->flag_a ? "true" : "false", r->flag_b ? "true" : "false"); break;
        case HOLO_INV_PATH_ORDER:
            fprintf(file, ",\"ordered_path_valid\":%s,\"permuted_path_valid\":%s,\"order_sensitive\":%s",
                    r->flag_a ? "true" : "false", r->flag_b ? "true" : "false", r->flag_c ? "true" : "false"); break;
        case HOLO_INV_SOFTWARE_HOLONOMY:
            fprintf(file, ",\"status\":\"DEFERRED_NOT_WELL_DEFINED\",\"missing_structure\":\"group_valued_carrier_transform_per_path_step\""); break;
    }
}

int holo_invariant_family_write_json(FILE *file, const HoloInvariantFamily *family) {
    size_t i;
    if (!file || !holo_invariant_family_validate(family)) return -1;
    fprintf(file, "  \"invariant_family\": {\n");
    fprintf(file, "    \"family_id\": \"%s\", \"predeclared\": %s, \"extracted\": %s, \"sealed\": %s, \"claim_level\": %d,\n",
            family->family_id, family->predeclared ? "true" : "false", family->extracted ? "true" : "false",
            family->sealed ? "true" : "false", family->claim_level);
    fprintf(file, "    \"path_history_digest\": \"%016" PRIx64 "\", \"family_digest\": \"%016" PRIx64 "\",\n",
            family->path_history_digest, family->family_digest);
    fprintf(file, "    \"records\": [\n");
    for (i = 0; i < family->count; ++i) {
        const HoloInvariantRecord *r = &family->records[i];
        fprintf(file, "      {\"kind\":\"%s\",\"invariant_id\":\"%s\",\"operator\":\"%s\",\"declaration_phase\":\"%s\",\"predeclared\":%s,\"evaluated\":%s,\"passed\":%s,\"result\":\"%s\",\"tolerance\":%.17g,\"claim_level\":%d,\"equality_rule\":\"%s\",\"evidence_level\":\"%s\"",
                holo_invariant_kind_name(r->kind), r->invariant_id, r->operator_id, r->declaration_phase,
                r->predeclared ? "true" : "false", r->evaluated ? "true" : "false", r->passed ? "true" : "false",
                holo_invariant_result_name(r->result), r->tolerance, r->claim_level, r->equality_rule, r->evidence_level);
        write_result(file, r);
        fprintf(file, "}%s\n", i + 1U == family->count ? "" : ",");
    }
    fprintf(file, "    ]\n  },\n");
    return ferror(file) ? -2 : 0;
}

static const char *find_key_value(const char *start, const char *key) {
    char needle[96];
    const char *p;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = start;
    while ((p = strstr(p, needle)) != NULL) {
        const char *after = p + strlen(needle);
        while (*after == ' ' || *after == '\t') ++after;
        if (*after == ':') return after + 1;
        p = after;
    }
    return NULL;
}

static int parse_bool_after(const char *start, const char *key, int *value) {
    const char *p = find_key_value(start, key);
    if (!p) return 0;
    while (*p == ' ' || *p == '\t') ++p;
    if (strncmp(p, "true", 4) == 0) { *value = 1; return 1; }
    if (strncmp(p, "false", 5) == 0) { *value = 0; return 1; }
    return 0;
}

static int parse_hex_after(const char *start, const char *key, uint64_t *value) {
    const char *p = find_key_value(start, key);
    if (!p) return 0;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p++ != '"') return 0;
    return sscanf(p, "%16" SCNx64, value) == 1;
}

static int parse_double_after(const char *start, const char *key, double *value) {
    const char *p = find_key_value(start, key);
    return p && sscanf(p, "%lf", value) == 1;
}

static int parse_int_after(const char *start, const char *key, int *value) {
    const char *p = find_key_value(start, key);
    return p && sscanf(p, "%d", value) == 1;
}

static int parse_u64_after(const char *start, const char *key, uint64_t *value) {
    const char *p = find_key_value(start, key);
    return p && sscanf(p, "%" SCNu64, value) == 1;
}

static int parse_text_after(const char *start, const char *key, char *out, size_t size) {
    const char *p;
    const char *end;
    p = find_key_value(start, key);
    if (!p) return 0;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p++ != '"') return 0;
    end = strchr(p, '"');
    if (!end || (size_t)(end - p) >= size) return 0;
    memcpy(out, p, (size_t)(end - p));
    out[end - p] = '\0';
    return 1;
}

static int parse_record(const char *start, HoloInvariantRecord *r) {
    int passed, evaluated, predeclared, claim_level;
    char result[48];
    if (!parse_bool_after(start, "predeclared", &predeclared) ||
        !parse_bool_after(start, "evaluated", &evaluated) || !parse_bool_after(start, "passed", &passed) ||
        !parse_int_after(start, "claim_level", &claim_level) ||
        !parse_double_after(start, "tolerance", &r->tolerance) ||
        !parse_text_after(start, "invariant_id", r->invariant_id, sizeof(r->invariant_id)) ||
        !parse_text_after(start, "operator", r->operator_id, sizeof(r->operator_id)) ||
        !parse_text_after(start, "declaration_phase", r->declaration_phase, sizeof(r->declaration_phase)) ||
        !parse_text_after(start, "equality_rule", r->equality_rule, sizeof(r->equality_rule)) ||
        !parse_text_after(start, "evidence_level", r->evidence_level, sizeof(r->evidence_level)) ||
        !parse_text_after(start, "result", result, sizeof(result))) return 0;
    r->predeclared = predeclared; r->evaluated = evaluated; r->passed = passed;
    r->claim_level = claim_level;
    r->result = strcmp(result, "PASS") == 0 ? HOLO_INV_RESULT_PASS :
                strcmp(result, "FAIL") == 0 ? HOLO_INV_RESULT_FAIL :
                strcmp(result, "DEFERRED_NOT_WELL_DEFINED") == 0 ?
                HOLO_INV_RESULT_DEFERRED_NOT_WELL_DEFINED : HOLO_INV_RESULT_PENDING;
    switch (r->kind) {
        case HOLO_INV_ORBIT_CONSERVATION:
            return parse_double_after(start, "orbit_sum_initial", &r->scalar_a) &&
                   parse_double_after(start, "orbit_sum_terminal", &r->scalar_b) &&
                   parse_double_after(start, "orbit_sum_restored", &r->scalar_c) &&
                   parse_u64_after(start, "orbit_product_initial", &r->digest_a) &&
                   parse_u64_after(start, "orbit_product_terminal", &r->digest_b) &&
                   parse_u64_after(start, "orbit_product_restored", &r->digest_c) &&
                   parse_bool_after(start, "fold_relation_preserved", &r->flag_a) &&
                   parse_bool_after(start, "product_preserved", &r->flag_b);
        case HOLO_INV_RELATION_BASIS:
            return parse_double_after(start, "determinant", &r->scalar_a) && parse_double_after(start, "trace", &r->scalar_b) &&
                   parse_bool_after(start, "basis_matches_exchange", &r->flag_a) && parse_bool_after(start, "det_minus_one", &r->flag_b) &&
                   parse_bool_after(start, "trace_zero", &r->flag_c) && parse_bool_after(start, "render_involution", &r->flag_d);
        case HOLO_INV_PATH_COMPOSITION:
            return parse_hex_after(start, "initial_state_digest", &r->digest_a) && parse_hex_after(start, "terminal_state_digest", &r->digest_b) &&
                   parse_hex_after(start, "restored_state_digest", &r->digest_c) && parse_bool_after(start, "composition_identity", &r->flag_a) &&
                   parse_bool_after(start, "evolution_operator_consistent", &r->flag_b);
        case HOLO_INV_RESTORATION_CLOSURE:
            return parse_bool_after(start, "restoration_attempted", &r->flag_a) && parse_bool_after(start, "restoration_verified", &r->flag_b) &&
                   parse_double_after(start, "restoration_metric", &r->scalar_a);
        case HOLO_INV_EXCHANGE_COVARIANCE:
            return parse_double_after(start, "original_neutral_sum", &r->scalar_a) && parse_double_after(start, "exchanged_neutral_sum", &r->scalar_b) &&
                   parse_u64_after(start, "original_neutral_product", &r->digest_a) &&
                   parse_u64_after(start, "exchanged_neutral_product", &r->digest_b) &&
                   parse_bool_after(start, "neutral_invariants_equal", &r->flag_a) && parse_bool_after(start, "indexed_quantities_exchanged", &r->flag_b) &&
                   parse_bool_after(start, "exchange_covariance", &r->flag_c) && parse_bool_after(start, "branch_neutral", &r->flag_d);
        case HOLO_INV_SERIALIZATION:
            return parse_hex_after(start, "pre_serialization_digest", &r->digest_a) && parse_hex_after(start, "post_reload_digest", &r->digest_b) &&
                   parse_hex_after(start, "recomputed_digest", &r->digest_c) && parse_bool_after(start, "reloaded_equal", &r->flag_a) &&
                   parse_bool_after(start, "recomputed_equal", &r->flag_b);
        case HOLO_INV_PATH_ORDER:
            return parse_bool_after(start, "ordered_path_valid", &r->flag_a) && parse_bool_after(start, "permuted_path_valid", &r->flag_b) &&
                   parse_bool_after(start, "order_sensitive", &r->flag_c);
        case HOLO_INV_SOFTWARE_HOLONOMY: return 1;
    }
    return 0;
}

int holo_invariant_family_read_json(const char *json, HoloInvariantFamily *family) {
    const char *section;
    size_t i;
    if (!json || !family || holo_invariant_family_init(family) != 0) return -1;
    section = strstr(json, "\"invariant_family\"");
    if (!section || !parse_text_after(section, "family_id", family->family_id, sizeof(family->family_id)) ||
        !parse_bool_after(section, "predeclared", &family->predeclared) ||
        !parse_int_after(section, "claim_level", &family->claim_level) ||
        !parse_hex_after(section, "path_history_digest", &family->path_history_digest) ||
        !parse_hex_after(section, "family_digest", &family->family_digest)) return -2;
    for (i = 0; i < family->count; ++i) {
        char needle[128];
        const char *record;
        snprintf(needle, sizeof(needle), "\"kind\":\"%s\"", holo_invariant_kind_name((HoloInvariantKind)i));
        record = strstr(section, needle);
        if (!record || !parse_record(record, &family->records[i])) return -30 - (int)i;
    }
    family->extracted = 1;
    family->sealed = 1;
    return holo_invariant_family_validate(family) ? 0 : -4;
}
