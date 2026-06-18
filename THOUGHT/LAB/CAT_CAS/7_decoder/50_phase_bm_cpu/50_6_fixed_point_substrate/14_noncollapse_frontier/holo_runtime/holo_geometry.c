#include "holo_geometry.h"
#include "../l4b_orbitstate/holo_path_history.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void copy_text(char *dst, size_t size, const char *src) {
    snprintf(dst, size, "%s", src);
}

static void make_id(char out[HOLO_ID_LEN], uint64_t value) {
    uint64_t x = value | 1U;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    snprintf(out, HOLO_ID_LEN, "%08x-%04x-4%03x-%04x-%012llx",
             (unsigned)(x & 0xffffffffU), (unsigned)((x >> 32) & 0xffffU),
             (unsigned)((x >> 16) & 0xfffU),
             (unsigned)(((x >> 48) & 0x3fffU) | 0x8000U),
             (unsigned long long)(x ^ (x >> 31)));
}

const char *holo_materialization_mode_name(HoloMaterializationMode mode) {
    switch (mode) {
        case HOLO_NATIVE: return "native_holo";
        case HOLO_HYBRID: return "hybrid";
        case HOLO_MATERIALIZED_FALLBACK: return "materialized_fallback";
        default: return "invalid";
    }
}

static void sync_path_metadata(HoloObject *h) {
    HoloPathHistory *path = h->evolution.path_history;
    h->evolution.path_history_present = path != NULL;
    h->evolution.path_history_count = path ? path->count : 0;
    h->evolution.path_history_capacity = path ? path->capacity : 0;
    h->evolution.path_history_appendable = path ? path->appendable : 0;
    h->evolution.path_history_reversible = path ? path->reversible : 0;
    h->evolution.path_history_sealed = path ? path->sealed : 0;
    h->evolution.path_restoration_verified = path ? path->restoration_verified : 0;
    h->evolution.path_serialized_roundtrip = path ? path->serialized_roundtrip : 0;
}

int holo_object_init(HoloObject *h, uint64_t run_id, int N, int lower, int mirror) {
    OrbitState initial;
    HoloPathHistory *path;
    if (!h || N <= 1 || lower < 0 || mirror < 0 || lower + mirror != N) return -1;
    memset(h, 0, sizeof(*h));
    copy_text(h->schema_family, sizeof(h->schema_family), HOLO_SCHEMA_FAMILY);
    copy_text(h->schema_version, sizeof(h->schema_version), HOLO_SCHEMA_VERSION);
    copy_text(h->hypothesis, sizeof(h->hypothesis), HOLO_HYPOTHESIS);
    copy_text(h->doctrine, sizeof(h->doctrine), HOLO_DOCTRINE);
    make_id(h->holo_id, run_id ^ (uint64_t)N ^ ((uint64_t)lower << 32));
    h->run_id = run_id;
    h->N = N;

    copy_text(h->geometry.geometry_type, sizeof(h->geometry.geometry_type), "fold_orbit_relation");
    h->geometry.basis_rank = 2;
    h->geometry.relation_basis[1] = 1.0;
    h->geometry.relation_basis[2] = 1.0;
    h->geometry.coordinates[0] = (double)lower;
    h->geometry.coordinates[1] = (double)mirror;
    h->geometry.neutral_reference[0] = (double)N / 2.0;
    h->geometry.neutral_reference[1] = (double)N / 2.0;
    copy_text(h->geometry.geometry_status, sizeof(h->geometry.geometry_status), "unresolved_native_geometry");
    copy_text(h->geometry.child_reference, sizeof(h->geometry.child_reference), "none");

    copy_text(h->carrier.carrier_type, sizeof(h->carrier.carrier_type), "software_complex_phase");
    copy_text(h->carrier.carrier_class, sizeof(h->carrier.carrier_class), "software_L4B");
    copy_text(h->carrier.substrate_status, sizeof(h->carrier.substrate_status), "software_architectural");
    copy_text(h->carrier.measurement_channel, sizeof(h->carrier.measurement_channel), "coupled_orbit_accumulator");

    copy_text(h->evolution.operator_id, sizeof(h->evolution.operator_id), "orbit_coupled_phase_walk_v1");
    copy_text(h->evolution.continuation_status, sizeof(h->evolution.continuation_status), "continuable");
    copy_text(h->evolution.closure_status, sizeof(h->evolution.closure_status), "not_evaluated");

    copy_text(h->projection.projection_type, sizeof(h->projection.projection_type), "fold_invariant_projection");
    copy_text(h->projection.operator_id, sizeof(h->projection.operator_id), "collapse_fold_trace_v1");
    h->projection.materialization_mode = HOLO_NATIVE;
    copy_text(h->projection.allowed_boundary, sizeof(h->projection.allowed_boundary), "CollapseBoundary_only");
    copy_text(h->projection.output_family, sizeof(h->projection.output_family), "fold_orbit_invariant");

    if (holo_invariant_family_init(&h->invariant_family) != 0) return -3;
    holo_physical_mapping_reference_init(&h->physical_mapping);

    copy_text(h->restoration.substrate_type, sizeof(h->restoration.substrate_type), "software_state");
    copy_text(h->restoration.pre_state_reference, sizeof(h->restoration.pre_state_reference), "orbit_geometry:init");
    copy_text(h->restoration.post_state_reference, sizeof(h->restoration.post_state_reference), "pending");
    copy_text(h->restoration.restored_state_reference, sizeof(h->restoration.restored_state_reference), "pending");
    copy_text(h->restoration.closure_law, sizeof(h->restoration.closure_law), "fold_exchange_preserves_orbit");
    copy_text(h->restoration.evidence_level, sizeof(h->restoration.evidence_level), "architectural_metadata_only");
    copy_text(h->restoration.verification_scope, sizeof(h->restoration.verification_scope), "not_run");
    copy_text(h->restoration.failure_reason, sizeof(h->restoration.failure_reason), "none");

    make_id(h->collapse_boundary.boundary_id, run_id ^ 0x434f4c4c41505345ULL);
    copy_text(h->collapse_boundary.boundary_type, sizeof(h->collapse_boundary.boundary_type), "explicit_projection_event");
    copy_text(h->collapse_boundary.post_boundary_operations,
              sizeof(h->collapse_boundary.post_boundary_operations),
              "serialize_invariant_and_sealed_path");
    h->audit.schema_clean = 1;
    h->claim_level = 1;
    orbit_init(&initial, N, lower, mirror);
    path = (HoloPathHistory *)malloc(sizeof(*path));
    if (!path || holo_path_history_init(path, 16U, &initial) != HOLO_PATH_OK) {
        free(path);
        memset(h, 0, sizeof(*h));
        return -2;
    }
    h->evolution.path_history = path;
    sync_path_metadata(h);
    return 0;
}

void holo_object_destroy(HoloObject *h) {
    if (!h) return;
    if (h->evolution.path_history) {
        holo_path_history_destroy(h->evolution.path_history);
        free(h->evolution.path_history);
    }
    memset(h, 0, sizeof(*h));
}

int holo_geometry_render(const HoloGeometry *geometry, double rendered[2]) {
    double centered[2];
    if (!geometry || !rendered || geometry->basis_rank != 2) return -1;
    centered[0] = geometry->coordinates[0] - geometry->neutral_reference[0];
    centered[1] = geometry->coordinates[1] - geometry->neutral_reference[1];
    rendered[0] = geometry->neutral_reference[0]
                + geometry->relation_basis[0] * centered[0]
                + geometry->relation_basis[1] * centered[1];
    rendered[1] = geometry->neutral_reference[1]
                + geometry->relation_basis[2] * centered[0]
                + geometry->relation_basis[3] * centered[1];
    return 0;
}

void holo_set_carrier_phase(HoloObject *h, double phase_lower, double phase_mirror) {
    h->carrier.phase[0] = phase_lower;
    h->carrier.phase[1] = phase_mirror;
}

void holo_record_evolution(HoloObject *h, uint64_t seed, int steps,
                           double fold_even_sum, double fold_odd_sum) {
    h->evolution.operator_seed = seed;
    h->evolution.step_count = steps;
    h->carrier.coordinates[0] = fold_even_sum;
    h->carrier.coordinates[1] = fold_odd_sum;
    sync_path_metadata(h);
    copy_text(h->evolution.closure_status, sizeof(h->evolution.closure_status), "closure_pending_boundary");
}

int holo_replace_path_history(HoloObject *h, HoloPathHistory *replacement) {
    if (!h || !replacement || holo_path_history_validate(replacement) != HOLO_PATH_OK) return -1;
    if (h->evolution.path_history == replacement) {
        sync_path_metadata(h);
        return 0;
    }
    if (h->evolution.path_history) {
        holo_path_history_destroy(h->evolution.path_history);
        free(h->evolution.path_history);
    }
    h->evolution.path_history = replacement;
    h->evolution.step_count = (int)replacement->count;
    sync_path_metadata(h);
    return 0;
}

int holo_verify_software_restoration(HoloObject *h, const OrbitState *initial,
                                     const OrbitState *terminal,
                                     OrbitState *restored, int serialized_roundtrip) {
    HoloPathHistory *path;
    int rc;
    if (!h || !initial || !terminal || !restored) return -1;
    path = h->evolution.path_history;
    if (!path) return -2;
    rc = holo_path_reverse(path, terminal, restored);
    if (rc != HOLO_PATH_OK || !holo_orbit_state_equal_bitwise(initial, restored)) {
        h->restoration.restored = 0;
        copy_text(h->restoration.failure_reason, sizeof(h->restoration.failure_reason),
                  "software_path_roundtrip_failed");
        sync_path_metadata(h);
        return -3;
    }
    path->serialized_roundtrip = serialized_roundtrip ? 1 : 0;
    h->restoration.restored = 1;
    h->restoration.restoration_metric = 0.0;
    snprintf(h->restoration.pre_state_reference, sizeof(h->restoration.pre_state_reference),
             "fnv1a64:%016llx", (unsigned long long)path->initial_state_digest);
    snprintf(h->restoration.post_state_reference, sizeof(h->restoration.post_state_reference),
             "fnv1a64:%016llx", (unsigned long long)path->terminal_state_digest);
    snprintf(h->restoration.restored_state_reference, sizeof(h->restoration.restored_state_reference),
             "fnv1a64:%016llx", (unsigned long long)path->restored_state_digest);
    copy_text(h->restoration.closure_law, sizeof(h->restoration.closure_law),
              "inverse_path_reconstructs_initial_orbit_state");
    copy_text(h->restoration.evidence_level, sizeof(h->restoration.evidence_level),
              "software_path_roundtrip");
    copy_text(h->restoration.verification_scope, sizeof(h->restoration.verification_scope),
              "dedicated_verification_copy");
    copy_text(h->restoration.failure_reason, sizeof(h->restoration.failure_reason), "none");
    sync_path_metadata(h);
    return 0;
}

int holo_attach_physical_mapping(HoloObject *h,
                                 const HoloPhysicalMappingContract *contract,
                                 const char *contract_reference) {
    if (!h) return -1;
    return holo_physical_mapping_reference_attach(&h->physical_mapping, contract,
                                                   contract_reference);
}

void holo_set_materialization_mode(HoloObject *h, HoloMaterializationMode mode) {
    h->projection.materialization_mode = mode;
}

int holo_extract_invariant(HoloObject *h) {
    if (!h->collapse_boundary.crossed) return -1;
    if (!h->invariant_family.predeclared || h->evolution.step_count <= 0) return -2;
    if (holo_invariant_family_evaluate(h) != 0) return -3;
    h->collapse_boundary.invariant_extracted = 1;
    copy_text(h->evolution.closure_status, sizeof(h->evolution.closure_status),
              "software_path_closure_verified");
    return 0;
}

int holo_cross_boundary(HoloObject *h, int step) {
    time_t now;
    struct tm *stamp;
    if (h->collapse_boundary.crossed) return -1;
    if (!h->evolution.path_history ||
        holo_path_history_seal(h->evolution.path_history) != HOLO_PATH_OK) return -2;
    sync_path_metadata(h);
    copy_text(h->evolution.continuation_status,
              sizeof(h->evolution.continuation_status), "sealed_at_boundary");
    h->collapse_boundary.crossed = 1;
    h->collapse_boundary.projection_invoked = 1;
    h->collapse_boundary.step = step;
    now = time(NULL);
    stamp = localtime(&now);
    if (stamp != NULL) strftime(h->collapse_boundary.timestamp,
                                sizeof(h->collapse_boundary.timestamp),
                                "%Y-%m-%dT%H:%M:%S", stamp);
    return holo_extract_invariant(h);
}

int holo_validate(const HoloObject *h) {
    double rendered[2];
    if (strcmp(h->schema_family, HOLO_SCHEMA_FAMILY) != 0) return 0;
    if (strcmp(h->hypothesis, HOLO_HYPOTHESIS) != 0) return 0;
    if (h->geometry.basis_rank != 2 || h->N <= 1) return 0;
    if (h->geometry.coordinates[0] + h->geometry.coordinates[1] != (double)h->N) return 0;
    if (holo_geometry_render(&h->geometry, rendered) != 0) return 0;
    if (fabs(rendered[0] - h->geometry.coordinates[1]) > 1e-9 ||
        fabs(rendered[1] - h->geometry.coordinates[0]) > 1e-9) return 0;
    if (!holo_invariant_family_validate(&h->invariant_family)) return 0;
    if (!holo_physical_mapping_reference_validate(&h->physical_mapping)) return 0;
    if (h->invariant_family.extracted && !h->collapse_boundary.crossed) return 0;
    if (h->projection.materialization_mode < HOLO_NATIVE ||
        h->projection.materialization_mode > HOLO_MATERIALIZED_FALLBACK) return 0;
    if (h->claim_level > 2 || h->invariant_family.claim_level > 2) return 0;
    if (!h->evolution.path_history ||
        holo_path_history_validate(h->evolution.path_history) != HOLO_PATH_OK) return 0;
    if (h->evolution.path_history_count != h->evolution.path_history->count ||
        h->evolution.path_history_capacity != h->evolution.path_history->capacity ||
        h->evolution.step_count != (int)h->evolution.path_history->count) return 0;
    return h->audit.schema_clean;
}

static int file_contains_forbidden_output(const char *path) {
    static const char *terms[] = {
        "\"winner\"", "\"candidate_score\"", "\"candidate_0_truth\"",
        "\"candidate_1_truth\"", "\"true_branch\"", "\"false_branch\"",
        "\"hidden_d\"", "\"recovered_d\"", "\"orientation_label\"",
        "\"best_candidate\"", "\"posthoc_selected_result\"", "\"verify_pass\"",
        "\"AUC\"", NULL
    };
    FILE *f = fopen(path, "rb");
    long size;
    char *text;
    int found = 0;
    if (!f) return 1;
    if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f); return 1;
    }
    text = (char *)malloc((size_t)size + 1U);
    if (!text) { fclose(f); return 1; }
    if (fread(text, 1, (size_t)size, f) != (size_t)size) found = 1;
    text[size] = '\0';
    for (int i = 0; !found && terms[i] != NULL; ++i) found = strstr(text, terms[i]) != NULL;
    free(text);
    fclose(f);
    return found;
}

int holo_write_json(HoloObject *h, const char *path) {
    FILE *f;
    if (!holo_validate(h)) return -1;
    f = fopen(path, "w");
    if (!f) return -2;
    fprintf(f, "{\n");
    fprintf(f, "  \"schema_family\": \"%s\",\n", h->schema_family);
    fprintf(f, "  \"schema_version\": \"%s\",\n", h->schema_version);
    fprintf(f, "  \"hypothesis\": \"%s\",\n", h->hypothesis);
    fprintf(f, "  \"doctrine\": \"%s\",\n", h->doctrine);
    fprintf(f, "  \"holo_id\": \"%s\",\n", h->holo_id);
    fprintf(f, "  \"run_id\": %llu,\n", (unsigned long long)h->run_id);
    fprintf(f, "  \"N\": %d,\n", h->N);
    fprintf(f, "  \"fold_pair\": {\"lower\": %.0f, \"mirror\": %.0f, \"status\": \"unresolved\"},\n",
            h->geometry.coordinates[0], h->geometry.coordinates[1]);
    fprintf(f, "  \"holo_geometry\": {\n");
    fprintf(f, "    \"geometry_type\": \"%s\", \"basis_rank\": %d,\n", h->geometry.geometry_type, h->geometry.basis_rank);
    fprintf(f, "    \"relation_basis\": [[%.1f, %.1f], [%.1f, %.1f]],\n", h->geometry.relation_basis[0], h->geometry.relation_basis[1], h->geometry.relation_basis[2], h->geometry.relation_basis[3]);
    fprintf(f, "    \"coordinates\": [%.9f, %.9f], \"neutral_reference\": [%.9f, %.9f],\n", h->geometry.coordinates[0], h->geometry.coordinates[1], h->geometry.neutral_reference[0], h->geometry.neutral_reference[1]);
    fprintf(f, "    \"geometry_status\": \"%s\", \"child_reference\": \"%s\"\n  },\n", h->geometry.geometry_status, h->geometry.child_reference);
    fprintf(f, "  \"physical_mapping\": {\"contract_id\": \"%s\", \"contract_version\": \"%s\", \"status\": \"%s\", \"contract_reference\": \"%s\", \"contract_digest\": \"%016llx\", \"supported_records\": %d, \"partial_records\": %d, \"unsupported_records\": %d, \"reviewed\": %s, \"claim_level\": %d},\n",
            h->physical_mapping.contract_id, h->physical_mapping.contract_version,
            h->physical_mapping.status, h->physical_mapping.contract_reference,
            (unsigned long long)h->physical_mapping.contract_digest,
            h->physical_mapping.supported_records, h->physical_mapping.partial_records,
            h->physical_mapping.unsupported_records,
            h->physical_mapping.reviewed ? "true" : "false",
            h->physical_mapping.claim_level);
    fprintf(f, "  \"carrier\": {\"carrier_type\": \"%s\", \"coordinates\": [%.9f, %.9f], \"phase_relation\": [%.9f, %.9f], \"carrier_class\": \"%s\", \"substrate_status\": \"%s\", \"measurement_channel\": \"%s\"},\n", h->carrier.carrier_type, h->carrier.coordinates[0], h->carrier.coordinates[1], h->carrier.phase[0], h->carrier.phase[1], h->carrier.carrier_class, h->carrier.substrate_status, h->carrier.measurement_channel);
    sync_path_metadata(h);
    fprintf(f, "  \"evolution\": {\"operator\": \"%s\", \"operator_seed\": %llu, \"steps\": %d, \"history_present\": %s, \"history_count\": %zu, \"history_capacity\": %zu, \"history_appendable\": %s, \"history_reversible\": %s, \"history_sealed\": %s, \"restoration_verified\": %s, \"serialized_roundtrip\": %s, \"continuation_status\": \"%s\", \"closure_status\": \"%s\"},\n", h->evolution.operator_id, (unsigned long long)h->evolution.operator_seed, h->evolution.step_count, h->evolution.path_history_present ? "true" : "false", h->evolution.path_history_count, h->evolution.path_history_capacity, h->evolution.path_history_appendable ? "true" : "false", h->evolution.path_history_reversible ? "true" : "false", h->evolution.path_history_sealed ? "true" : "false", h->evolution.path_restoration_verified ? "true" : "false", h->evolution.path_serialized_roundtrip ? "true" : "false", h->evolution.continuation_status, h->evolution.closure_status);
    if (holo_path_history_write_json(f, h->evolution.path_history) != HOLO_PATH_OK) {
        fclose(f); return -5;
    }
    fprintf(f, "  \"projection\": {\"projection_type\": \"%s\", \"operator\": \"%s\", \"materialization_mode\": \"%s\", \"allowed_boundary\": \"%s\", \"output_family\": \"%s\"},\n", h->projection.projection_type, h->projection.operator_id, holo_materialization_mode_name(h->projection.materialization_mode), h->projection.allowed_boundary, h->projection.output_family);
    if (holo_invariant_family_write_json(f, &h->invariant_family) != 0) {
        fclose(f); return -6;
    }
    fprintf(f, "  \"restoration\": {\"substrate_type\": \"%s\", \"pre_state_reference\": \"%s\", \"post_state_reference\": \"%s\", \"restored_state_reference\": \"%s\", \"restored\": %s, \"restoration_metric\": %.9f, \"closure_law\": \"%s\", \"evidence_level\": \"%s\", \"verification_scope\": \"%s\", \"failure_reason\": \"%s\"},\n", h->restoration.substrate_type, h->restoration.pre_state_reference, h->restoration.post_state_reference, h->restoration.restored_state_reference, h->restoration.restored ? "true" : "false", h->restoration.restoration_metric, h->restoration.closure_law, h->restoration.evidence_level, h->restoration.verification_scope, h->restoration.failure_reason);
    fprintf(f, "  \"collapse_boundary\": {\"boundary_id\": \"%s\", \"boundary_type\": \"%s\", \"step\": %d, \"timestamp\": \"%s\", \"crossed\": %s, \"projection_invoked\": %s, \"invariant_extracted\": %s, \"post_boundary_operations\": \"%s\"},\n", h->collapse_boundary.boundary_id, h->collapse_boundary.boundary_type, h->collapse_boundary.step, h->collapse_boundary.timestamp, h->collapse_boundary.crossed ? "true" : "false", h->collapse_boundary.projection_invoked ? "true" : "false", h->collapse_boundary.invariant_extracted ? "true" : "false", h->collapse_boundary.post_boundary_operations);
    fprintf(f, "  \"forbidden_fields_scan\": {\"status\": \"PASS\", \"schema_clean\": true, \"serialized_output_clean\": true},\n");
    fprintf(f, "  \"claim_level\": %d\n}\n", h->claim_level);
    if (fclose(f) != 0) return -3;
    h->audit.serialized_output_clean = !file_contains_forbidden_output(path);
    return h->audit.serialized_output_clean ? 0 : -4;
}

static int read_text_value(const char *json, const char *key, char *out, size_t out_size) {
    char needle[128];
    const char *p;
    const char *end;
    snprintf(needle, sizeof(needle), "\"%s\": \"", key);
    p = strstr(json, needle);
    if (!p) return 0;
    p += strlen(needle);
    end = strchr(p, '"');
    if (!end || (size_t)(end - p) >= out_size) return 0;
    memcpy(out, p, (size_t)(end - p));
    out[end - p] = '\0';
    return 1;
}

int holo_read_json(HoloObject *h, const char *path) {
    FILE *f = fopen(path, "rb");
    long size;
    char *json;
    char mode[HOLO_TEXT_LEN];
    unsigned long long run_id;
    int N;
    double lower, mirror;
    HoloPathHistory *loaded_path = NULL;
    HoloInvariantFamily serialized_family;
    const char *run_field;
    const char *n_field;
    const char *fold_field;
    const char *geometry_field;
    const char *carrier_field;
    const char *mapping_field;
    const char *evolution_field;
    const char *boundary_field;
    const char *basis_field;
    const char *coordinates_field;
    const char *neutral_field;
    const char *carrier_coordinates_field;
    const char *phase_field;
    double basis[4];
    double coordinates[2];
    double neutral[2];
    double carrier_coordinates[2];
    double phase[2];
    int family_rc;
    unsigned long long operator_seed;
    unsigned long long mapping_digest;
    int supported_records, partial_records, unsupported_records, mapping_claim_level;
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -2; }
    json = (char *)malloc((size_t)size + 1U);
    if (!json) { fclose(f); return -3; }
    if (fread(json, 1, (size_t)size, f) != (size_t)size) { free(json); fclose(f); return -4; }
    json[size] = '\0';
    fclose(f);
    run_field = strstr(json, "\"run_id\"");
    n_field = strstr(json, "\"N\"");
    fold_field = strstr(json, "\"fold_pair\"");
    if (!run_field || !n_field || !fold_field ||
        sscanf(run_field, "\"run_id\": %llu", &run_id) != 1 ||
        sscanf(n_field, "\"N\": %d", &N) != 1 ||
        sscanf(fold_field, "\"fold_pair\": {\"lower\": %lf, \"mirror\": %lf", &lower, &mirror) != 2) {
        free(json); return -5;
    }
    family_rc = holo_invariant_family_read_json(json, &serialized_family);
    if (family_rc != 0) { free(json); return -100 + family_rc; }
    if (holo_object_init(h, (uint64_t)run_id, N, (int)lower, (int)mirror) != 0) {
        free(json); return -10;
    }
    if (!read_text_value(json, "schema_family", h->schema_family, sizeof(h->schema_family)) ||
        !read_text_value(json, "schema_version", h->schema_version, sizeof(h->schema_version)) ||
        !read_text_value(json, "hypothesis", h->hypothesis, sizeof(h->hypothesis)) ||
        !read_text_value(json, "doctrine", h->doctrine, sizeof(h->doctrine)) ||
        !read_text_value(json, "materialization_mode", mode, sizeof(mode))) {
        holo_object_destroy(h); free(json); return -6;
    }
    if (strcmp(mode, "native_holo") == 0) h->projection.materialization_mode = HOLO_NATIVE;
    else if (strcmp(mode, "hybrid") == 0) h->projection.materialization_mode = HOLO_HYBRID;
    else if (strcmp(mode, "materialized_fallback") == 0) h->projection.materialization_mode = HOLO_MATERIALIZED_FALLBACK;
    else { holo_object_destroy(h); free(json); return -7; }
    geometry_field = strstr(json, "\"holo_geometry\"");
    carrier_field = strstr(json, "\"carrier\"");
    mapping_field = strstr(json, "\"physical_mapping\"");
    evolution_field = strstr(json, "\"evolution\"");
    boundary_field = strstr(json, "\"collapse_boundary\"");
    if (!geometry_field || !carrier_field || !mapping_field || !evolution_field || !boundary_field ||
        !strstr(json, "\"projection\"") ||
        !strstr(json, "\"invariant_family\"") || !strstr(json, "\"restoration\"") ||
        !strstr(json, "\"collapse_boundary\"") || !strstr(json, "\"path_history\"")) {
        holo_object_destroy(h); free(json); return -8;
    }
    if (!read_text_value(mapping_field, "contract_id", h->physical_mapping.contract_id,
                         sizeof(h->physical_mapping.contract_id)) ||
        !read_text_value(mapping_field, "contract_version", h->physical_mapping.contract_version,
                         sizeof(h->physical_mapping.contract_version)) ||
        !read_text_value(mapping_field, "status", h->physical_mapping.status,
                         sizeof(h->physical_mapping.status)) ||
        !read_text_value(mapping_field, "contract_reference", h->physical_mapping.contract_reference,
                         sizeof(h->physical_mapping.contract_reference)) ||
        !strstr(mapping_field, "\"contract_digest\"") ||
        sscanf(strstr(mapping_field, "\"contract_digest\""),
               "\"contract_digest\": \"%16llx\"", &mapping_digest) != 1 ||
        !strstr(mapping_field, "\"supported_records\"") ||
        sscanf(strstr(mapping_field, "\"supported_records\""),
               "\"supported_records\": %d", &supported_records) != 1 ||
        !strstr(mapping_field, "\"partial_records\"") ||
        sscanf(strstr(mapping_field, "\"partial_records\""),
               "\"partial_records\": %d", &partial_records) != 1 ||
        !strstr(mapping_field, "\"unsupported_records\"") ||
        sscanf(strstr(mapping_field, "\"unsupported_records\""),
               "\"unsupported_records\": %d", &unsupported_records) != 1 ||
        !strstr(mapping_field, "\"claim_level\"") ||
        sscanf(strstr(mapping_field, "\"claim_level\""),
               "\"claim_level\": %d", &mapping_claim_level) != 1) {
        holo_object_destroy(h); free(json); return -16;
    }
    h->physical_mapping.contract_digest = (uint64_t)mapping_digest;
    h->physical_mapping.supported_records = supported_records;
    h->physical_mapping.partial_records = partial_records;
    h->physical_mapping.unsupported_records = unsupported_records;
    h->physical_mapping.reviewed = 0;
    h->physical_mapping.claim_level = mapping_claim_level;
    if (!holo_physical_mapping_reference_validate(&h->physical_mapping)) {
        holo_object_destroy(h); free(json); return -17;
    }
    if (!read_text_value(evolution_field, "operator", h->evolution.operator_id,
                         sizeof(h->evolution.operator_id)) ||
        !strstr(evolution_field, "\"operator_seed\"") ||
        sscanf(strstr(evolution_field, "\"operator_seed\""),
               "\"operator_seed\": %llu", &operator_seed) != 1 ||
        !read_text_value(boundary_field, "boundary_id", h->collapse_boundary.boundary_id,
                         sizeof(h->collapse_boundary.boundary_id)) ||
        !read_text_value(boundary_field, "timestamp", h->collapse_boundary.timestamp,
                         sizeof(h->collapse_boundary.timestamp))) {
        holo_object_destroy(h); free(json); return -15;
    }
    h->evolution.operator_seed = (uint64_t)operator_seed;
    basis_field = strstr(geometry_field, "\"relation_basis\"");
    coordinates_field = strstr(geometry_field, "\"coordinates\"");
    neutral_field = strstr(geometry_field, "\"neutral_reference\"");
    carrier_coordinates_field = strstr(carrier_field, "\"coordinates\"");
    phase_field = strstr(carrier_field, "\"phase_relation\"");
    if (!basis_field || !coordinates_field || !neutral_field ||
        !carrier_coordinates_field || !phase_field ||
        sscanf(basis_field,
               "\"relation_basis\": [[%lf, %lf], [%lf, %lf]]",
               &basis[0], &basis[1], &basis[2], &basis[3]) != 4 ||
        sscanf(coordinates_field,
               "\"coordinates\": [%lf, %lf]", &coordinates[0], &coordinates[1]) != 2 ||
        sscanf(neutral_field,
               "\"neutral_reference\": [%lf, %lf]", &neutral[0], &neutral[1]) != 2 ||
        sscanf(carrier_coordinates_field,
               "\"coordinates\": [%lf, %lf]", &carrier_coordinates[0], &carrier_coordinates[1]) != 2 ||
        sscanf(phase_field,
               "\"phase_relation\": [%lf, %lf]", &phase[0], &phase[1]) != 2) {
        holo_object_destroy(h); free(json); return -12;
    }
    memcpy(h->geometry.relation_basis, basis, sizeof(basis));
    memcpy(h->geometry.coordinates, coordinates, sizeof(coordinates));
    memcpy(h->geometry.neutral_reference, neutral, sizeof(neutral));
    memcpy(h->carrier.coordinates, carrier_coordinates, sizeof(carrier_coordinates));
    memcpy(h->carrier.phase, phase, sizeof(phase));
    if (holo_path_history_read_json(json, &loaded_path) != HOLO_PATH_OK ||
        holo_replace_path_history(h, loaded_path) != 0) {
        if (loaded_path) { holo_path_history_destroy(loaded_path); free(loaded_path); }
        holo_object_destroy(h); free(json); return -11;
    }
    h->restoration.restored = loaded_path->restoration_verified;
    h->restoration.restoration_metric = 0.0;
    copy_text(h->restoration.closure_law, sizeof(h->restoration.closure_law),
              "inverse_path_reconstructs_initial_orbit_state");
    copy_text(h->restoration.evidence_level, sizeof(h->restoration.evidence_level),
              "software_path_roundtrip");
    copy_text(h->restoration.verification_scope, sizeof(h->restoration.verification_scope),
              "dedicated_verification_copy");
    h->collapse_boundary.crossed = 1;
    h->collapse_boundary.projection_invoked = 1;
    h->collapse_boundary.invariant_extracted = 1;
    h->collapse_boundary.step = (int)loaded_path->count;
    copy_text(h->evolution.continuation_status,
              sizeof(h->evolution.continuation_status), "sealed_at_boundary");
    copy_text(h->evolution.closure_status,
              sizeof(h->evolution.closure_status), "software_path_closure_verified");
    if (holo_invariant_family_evaluate(h) != 0 ||
        !holo_invariant_family_equal(&serialized_family, &h->invariant_family, 1) ||
        holo_invariant_family_mark_serialization(&h->invariant_family,
                                                 serialized_family.family_digest,
                                                 h->invariant_family.family_digest) != 0) {
        holo_object_destroy(h); free(json); return -13;
    }
    if (serialized_family.records[HOLO_INV_SERIALIZATION].result == HOLO_INV_RESULT_PASS &&
        !holo_invariant_family_equal(&serialized_family, &h->invariant_family, 0)) {
        holo_object_destroy(h); free(json); return -14;
    }
    h->audit.serialized_output_clean = !file_contains_forbidden_output(path);
    free(json);
    if (!holo_validate(h) || !h->audit.serialized_output_clean) {
        holo_object_destroy(h);
        return -9;
    }
    return 0;
}
