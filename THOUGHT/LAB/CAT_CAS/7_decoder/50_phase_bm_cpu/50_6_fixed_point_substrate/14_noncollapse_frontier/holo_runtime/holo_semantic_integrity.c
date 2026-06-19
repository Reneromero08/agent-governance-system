#include "holo_semantic_integrity.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static int near_expected(double actual, double expected) {
    double scale = 1.0 + fabs(expected);
    return isfinite(actual) && isfinite(expected) &&
           fabs(actual - expected) <= 64.0 * DBL_EPSILON * scale;
}

int holo_path_history_validate_semantic(const HoloPathHistory *history,
                                        const OrbitState *initial_state,
                                        OrbitState *terminal_state_out) {
    OrbitState current;
    size_t index;
    int structural;

    if (!history || !initial_state) return HOLO_PATH_ERR_NULL;
    structural = holo_path_history_validate(history);
    if (structural != HOLO_PATH_OK) return structural;
    if (initial_state->N <= 0 || initial_state->steps != 0 ||
        initial_state->branch_plus < 0 || initial_state->branch_minus < 0 ||
        initial_state->branch_plus + initial_state->branch_minus != initial_state->N ||
        holo_orbit_state_digest(initial_state) != history->initial_state_digest) {
        return HOLO_PATH_ERR_CONTINUITY;
    }

    current = *initial_state;
    for (index = 0; index < history->count; ++index) {
        const HoloPathStep *step = &history->steps[index];
        OrbitState next;
        double theta_plus;
        double theta_minus;
        double expected_real;
        double expected_imag;
        double actual_real;
        double actual_imag;

        if (step->step_index != (uint32_t)current.steps ||
            step->operator_parameter == 0 ||
            step->operator_parameter > (uint64_t)current.N ||
            step->pre_acc_real_bits != double_bits(current.acc_real) ||
            step->pre_acc_imag_bits != double_bits(current.acc_imag) ||
            step->pre_state_digest != holo_orbit_state_digest(&current)) {
            return HOLO_PATH_ERR_OPERATOR;
        }

        theta_plus = 2.0 * M_PI * current.branch_plus *
                     (double)step->operator_parameter / current.N;
        theta_minus = 2.0 * M_PI * current.branch_minus *
                      (double)step->operator_parameter / current.N;
        expected_real = current.acc_real + cos(theta_plus) + cos(theta_minus);
        expected_imag = current.acc_imag + sin(theta_plus) + sin(theta_minus);
        actual_real = bits_double(step->post_acc_real_bits);
        actual_imag = bits_double(step->post_acc_imag_bits);

        if (!near_expected(actual_real, expected_real) ||
            !near_expected(actual_imag, expected_imag)) {
            return HOLO_PATH_ERR_OPERATOR;
        }

        next = current;
        next.acc_real = actual_real;
        next.acc_imag = actual_imag;
        next.steps++;
        if (step->post_state_digest != holo_orbit_state_digest(&next)) {
            return HOLO_PATH_ERR_CONTINUITY;
        }
        current = next;
    }

    if (holo_orbit_state_digest(&current) != history->terminal_state_digest) {
        return HOLO_PATH_ERR_CONTINUITY;
    }
    if (terminal_state_out) *terminal_state_out = current;
    return HOLO_PATH_OK;
}

int holo_object_validate_semantic(const HoloObject *object) {
    OrbitState initial;
    OrbitState terminal;
    double lower;
    double mirror;
    HoloPathHistory *path;

    if (!object || !holo_validate(object)) return 0;
    path = object->evolution.path_history;
    if (!path || !object->evolution.path_history_present) return 0;

    lower = nearbyint(object->geometry.coordinates[0]);
    mirror = nearbyint(object->geometry.coordinates[1]);
    if (fabs(lower - object->geometry.coordinates[0]) > DBL_EPSILON ||
        fabs(mirror - object->geometry.coordinates[1]) > DBL_EPSILON ||
        lower < 0.0 || mirror < 0.0 || lower + mirror != (double)object->N) {
        return 0;
    }

    memset(&initial, 0, sizeof(initial));
    initial.N = object->N;
    initial.branch_plus = (int)lower;
    initial.branch_minus = (int)mirror;
    if (holo_path_history_validate_semantic(path, &initial, &terminal) != HOLO_PATH_OK) {
        return 0;
    }
    if ((size_t)object->evolution.step_count != path->count ||
        terminal.steps != object->evolution.step_count) {
        return 0;
    }

    if (object->collapse_boundary.crossed) {
        if (!path->sealed || path->appendable ||
            !object->collapse_boundary.projection_invoked ||
            !object->collapse_boundary.invariant_extracted ||
            !object->invariant_family.extracted ||
            !object->invariant_family.sealed ||
            strcmp(object->evolution.continuation_status, "sealed_at_boundary") != 0) {
            return 0;
        }
    } else if (object->collapse_boundary.projection_invoked ||
               object->collapse_boundary.invariant_extracted ||
               object->invariant_family.extracted || object->invariant_family.sealed) {
        return 0;
    }

    if (object->restoration.restored) {
        if (!path->restoration_verified ||
            path->restored_state_digest != path->initial_state_digest ||
            strcmp(object->restoration.evidence_level, "software_path_roundtrip") != 0 ||
            strcmp(object->restoration.verification_scope, "dedicated_verification_copy") != 0) {
            return 0;
        }
    }
    return 1;
}

int holo_cross_boundary_atomic(HoloObject *object, int step) {
    HoloObject snapshot;
    HoloPathHistory path_snapshot;
    HoloPathHistory *path;
    int rc;

    if (!object || !object->evolution.path_history) return -1;
    snapshot = *object;
    path = object->evolution.path_history;
    path_snapshot = *path;

    rc = holo_cross_boundary(object, step);
    if (rc != 0 || !holo_object_validate_semantic(object)) {
        *object = snapshot;
        *path = path_snapshot;
        object->evolution.path_history = path;
        return rc != 0 ? rc : -2;
    }
    return 0;
}

static char *read_all(const char *path) {
    FILE *file;
    long size;
    char *text;
    if (!path) return NULL;
    file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    text = (char *)malloc((size_t)size + 1U);
    if (!text) {
        fclose(file);
        return NULL;
    }
    if (fread(text, 1, (size_t)size, file) != (size_t)size) {
        free(text);
        fclose(file);
        return NULL;
    }
    text[size] = '\0';
    fclose(file);
    return text;
}

static const char *section_end(const char *open_brace) {
    int depth = 0;
    int in_string = 0;
    int escaped = 0;
    const char *cursor;
    if (!open_brace || *open_brace != '{') return NULL;
    for (cursor = open_brace; *cursor; ++cursor) {
        char ch = *cursor;
        if (in_string) {
            if (escaped) escaped = 0;
            else if (ch == '\\') escaped = 1;
            else if (ch == '"') in_string = 0;
            continue;
        }
        if (ch == '"') in_string = 1;
        else if (ch == '{') ++depth;
        else if (ch == '}' && --depth == 0) return cursor;
    }
    return NULL;
}

static int find_section(const char *json, const char *name,
                        const char **begin_out, const char **end_out) {
    char needle[128];
    const char *key;
    const char *begin;
    const char *end;
    snprintf(needle, sizeof(needle), "\"%s\"", name);
    key = strstr(json, needle);
    if (!key || !(begin = strchr(key + strlen(needle), '{')) ||
        !(end = section_end(begin))) return 0;
    *begin_out = begin;
    *end_out = end;
    return 1;
}

static const char *find_key_bounded(const char *begin, const char *end,
                                    const char *key) {
    char needle[128];
    const char *found;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    found = strstr(begin, needle);
    return found && found < end ? found : NULL;
}

static int parse_bool_bounded(const char *begin, const char *end,
                              const char *key, int *value) {
    const char *found = find_key_bounded(begin, end, key);
    const char *colon;
    if (!found || !(colon = strchr(found, ':')) || colon >= end) return 0;
    do { ++colon; } while (colon < end && (*colon == ' ' || *colon == '\t'));
    if (end - colon >= 4 && strncmp(colon, "true", 4) == 0) *value = 1;
    else if (end - colon >= 5 && strncmp(colon, "false", 5) == 0) *value = 0;
    else return 0;
    return 1;
}

static int parse_int_bounded(const char *begin, const char *end,
                             const char *key, int *value) {
    const char *found = find_key_bounded(begin, end, key);
    const char *colon;
    if (!found || !(colon = strchr(found, ':')) || colon >= end) return 0;
    return sscanf(colon + 1, "%d", value) == 1;
}

static int parse_text_bounded(const char *begin, const char *end,
                              const char *key, char *output, size_t size) {
    const char *found = find_key_bounded(begin, end, key);
    const char *colon;
    const char *finish;
    size_t length;
    if (!found || !(colon = strchr(found, ':')) || colon >= end) return 0;
    do { ++colon; } while (colon < end && (*colon == ' ' || *colon == '\t'));
    if (colon >= end || *colon != '"') return 0;
    ++colon;
    finish = strchr(colon, '"');
    if (!finish || finish >= end) return 0;
    length = (size_t)(finish - colon);
    if (length >= size) return 0;
    memcpy(output, colon, length);
    output[length] = '\0';
    return 1;
}

int holo_read_json_strict(HoloObject *object, const char *path) {
    char *json;
    const char *evolution_begin;
    const char *evolution_end;
    const char *restoration_begin;
    const char *restoration_end;
    const char *boundary_begin;
    const char *boundary_end;
    char continuation[HOLO_TEXT_LEN];
    char closure[HOLO_TEXT_LEN];
    char restoration_closure[HOLO_TEXT_LEN];
    char evidence_level[HOLO_TEXT_LEN];
    char verification_scope[HOLO_TEXT_LEN];
    int history_present;
    int history_appendable;
    int history_reversible;
    int history_sealed;
    int restoration_verified;
    int serialized_roundtrip;
    int restored;
    int crossed;
    int projection_invoked;
    int invariant_extracted;
    int boundary_step;
    int rc;

    if (!object || !path) return -1;
    json = read_all(path);
    if (!json ||
        !find_section(json, "evolution", &evolution_begin, &evolution_end) ||
        !find_section(json, "restoration", &restoration_begin, &restoration_end) ||
        !find_section(json, "collapse_boundary", &boundary_begin, &boundary_end) ||
        !parse_bool_bounded(evolution_begin, evolution_end, "history_present",
                            &history_present) ||
        !parse_bool_bounded(evolution_begin, evolution_end, "history_appendable",
                            &history_appendable) ||
        !parse_bool_bounded(evolution_begin, evolution_end, "history_reversible",
                            &history_reversible) ||
        !parse_bool_bounded(evolution_begin, evolution_end, "history_sealed",
                            &history_sealed) ||
        !parse_bool_bounded(evolution_begin, evolution_end, "restoration_verified",
                            &restoration_verified) ||
        !parse_bool_bounded(evolution_begin, evolution_end, "serialized_roundtrip",
                            &serialized_roundtrip) ||
        !parse_text_bounded(evolution_begin, evolution_end, "continuation_status",
                            continuation, sizeof(continuation)) ||
        !parse_text_bounded(evolution_begin, evolution_end, "closure_status",
                            closure, sizeof(closure)) ||
        !parse_bool_bounded(restoration_begin, restoration_end, "restored",
                            &restored) ||
        !parse_text_bounded(restoration_begin, restoration_end, "closure_law",
                            restoration_closure, sizeof(restoration_closure)) ||
        !parse_text_bounded(restoration_begin, restoration_end, "evidence_level",
                            evidence_level, sizeof(evidence_level)) ||
        !parse_text_bounded(restoration_begin, restoration_end, "verification_scope",
                            verification_scope, sizeof(verification_scope)) ||
        !parse_int_bounded(boundary_begin, boundary_end, "step", &boundary_step) ||
        !parse_bool_bounded(boundary_begin, boundary_end, "crossed", &crossed) ||
        !parse_bool_bounded(boundary_begin, boundary_end, "projection_invoked",
                            &projection_invoked) ||
        !parse_bool_bounded(boundary_begin, boundary_end, "invariant_extracted",
                            &invariant_extracted)) {
        free(json);
        return -2;
    }

    rc = holo_read_json(object, path);
    if (rc != 0) {
        free(json);
        return rc;
    }

    if (object->evolution.path_history_present != history_present ||
        object->evolution.path_history_appendable != history_appendable ||
        object->evolution.path_history_reversible != history_reversible ||
        object->evolution.path_history_sealed != history_sealed ||
        object->evolution.path_restoration_verified != restoration_verified ||
        object->evolution.path_serialized_roundtrip != serialized_roundtrip ||
        strcmp(object->evolution.continuation_status, continuation) != 0 ||
        strcmp(object->evolution.closure_status, closure) != 0 ||
        object->restoration.restored != restored ||
        strcmp(object->restoration.closure_law, restoration_closure) != 0 ||
        strcmp(object->restoration.evidence_level, evidence_level) != 0 ||
        strcmp(object->restoration.verification_scope, verification_scope) != 0 ||
        object->collapse_boundary.step != boundary_step ||
        object->collapse_boundary.crossed != crossed ||
        object->collapse_boundary.projection_invoked != projection_invoked ||
        object->collapse_boundary.invariant_extracted != invariant_extracted ||
        !holo_object_validate_semantic(object)) {
        holo_object_destroy(object);
        free(json);
        return -3;
    }

    free(json);
    return 0;
}
