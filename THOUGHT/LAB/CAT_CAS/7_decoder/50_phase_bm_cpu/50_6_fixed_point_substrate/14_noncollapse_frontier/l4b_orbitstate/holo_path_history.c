#include "holo_path_history.h"

#include <inttypes.h>
#include <math.h>
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

static uint64_t fnv_byte(uint64_t digest, unsigned char byte) {
    return (digest ^ (uint64_t)byte) * UINT64_C(1099511628211);
}

static uint64_t fnv_u64(uint64_t digest, uint64_t value) {
    for (unsigned int shift = 0; shift < 64U; shift += 8U) {
        digest = fnv_byte(digest, (unsigned char)(value >> shift));
    }
    return digest;
}

static uint64_t fnv_text(uint64_t digest, const char *text, size_t limit) {
    size_t i;
    for (i = 0; i < limit && text[i] != '\0'; ++i) {
        digest = fnv_byte(digest, (unsigned char)text[i]);
    }
    return digest;
}

uint64_t holo_orbit_state_digest(const OrbitState *state) {
    uint64_t digest = UINT64_C(14695981039346656037);
    if (!state) return 0;
    digest = fnv_u64(digest, (uint64_t)(uint32_t)state->N);
    digest = fnv_u64(digest, (uint64_t)(uint32_t)state->branch_plus);
    digest = fnv_u64(digest, (uint64_t)(uint32_t)state->branch_minus);
    digest = fnv_u64(digest, double_bits(state->acc_real));
    digest = fnv_u64(digest, double_bits(state->acc_imag));
    digest = fnv_u64(digest, (uint64_t)(uint32_t)state->steps);
    return digest;
}

int holo_orbit_state_equal_bitwise(const OrbitState *left, const OrbitState *right) {
    if (!left || !right) return 0;
    return left->N == right->N &&
           left->branch_plus == right->branch_plus &&
           left->branch_minus == right->branch_minus &&
           double_bits(left->acc_real) == double_bits(right->acc_real) &&
           double_bits(left->acc_imag) == double_bits(right->acc_imag) &&
           left->steps == right->steps;
}

static uint64_t path_step_digest(const HoloPathStep *step) {
    uint64_t digest = UINT64_C(14695981039346656037);
    digest = fnv_u64(digest, step->step_index);
    digest = fnv_text(digest, step->operator_id, sizeof(step->operator_id));
    digest = fnv_u64(digest, step->operator_parameter);
    digest = fnv_u64(digest, step->pre_acc_real_bits);
    digest = fnv_u64(digest, step->pre_acc_imag_bits);
    digest = fnv_u64(digest, step->post_acc_real_bits);
    digest = fnv_u64(digest, step->post_acc_imag_bits);
    digest = fnv_u64(digest, step->pre_state_digest);
    digest = fnv_u64(digest, step->post_state_digest);
    return digest;
}

int holo_path_history_init(HoloPathHistory *history, size_t initial_capacity,
                           const OrbitState *initial_state) {
    if (!history || !initial_state) return HOLO_PATH_ERR_NULL;
    if (initial_capacity == 0 || initial_capacity > HOLO_PATH_MAX_SERIALIZED_STEPS ||
        initial_capacity > SIZE_MAX / sizeof(HoloPathStep)) return HOLO_PATH_ERR_CAPACITY;
    memset(history, 0, sizeof(*history));
    history->steps = (HoloPathStep *)calloc(initial_capacity, sizeof(HoloPathStep));
    if (!history->steps) return HOLO_PATH_ERR_CAPACITY;
    history->capacity = initial_capacity;
    history->appendable = 1;
    history->reversible = 1;
    history->initial_state_digest = holo_orbit_state_digest(initial_state);
    history->terminal_state_digest = history->initial_state_digest;
    return HOLO_PATH_OK;
}

void holo_path_history_reset(HoloPathHistory *history, const OrbitState *initial_state) {
    if (!history || !initial_state) return;
    if (history->steps && history->capacity > 0) {
        memset(history->steps, 0, history->capacity * sizeof(*history->steps));
    }
    history->count = 0;
    history->appendable = 1;
    history->reversible = 1;
    history->sealed = 0;
    history->restoration_verified = 0;
    history->serialized_roundtrip = 0;
    history->initial_state_digest = holo_orbit_state_digest(initial_state);
    history->terminal_state_digest = history->initial_state_digest;
    history->restored_state_digest = 0;
}

void holo_path_history_destroy(HoloPathHistory *history) {
    if (!history) return;
    free(history->steps);
    memset(history, 0, sizeof(*history));
}

static int grow_history(HoloPathHistory *history) {
    size_t next_capacity;
    HoloPathStep *grown;
    if (history->capacity >= HOLO_PATH_MAX_SERIALIZED_STEPS ||
        history->capacity > SIZE_MAX / 2U) return HOLO_PATH_ERR_CAPACITY;
    next_capacity = history->capacity * 2U;
    if (next_capacity > HOLO_PATH_MAX_SERIALIZED_STEPS) next_capacity = HOLO_PATH_MAX_SERIALIZED_STEPS;
    if (next_capacity <= history->capacity || next_capacity > SIZE_MAX / sizeof(*history->steps)) {
        return HOLO_PATH_ERR_CAPACITY;
    }
    grown = (HoloPathStep *)realloc(history->steps, next_capacity * sizeof(*history->steps));
    if (!grown) return HOLO_PATH_ERR_CAPACITY;
    memset(grown + history->capacity, 0,
           (next_capacity - history->capacity) * sizeof(*history->steps));
    history->steps = grown;
    history->capacity = next_capacity;
    return HOLO_PATH_OK;
}

static int step_numeric_valid(const HoloPathStep *step) {
    return isfinite(bits_double(step->pre_acc_real_bits)) &&
           isfinite(bits_double(step->pre_acc_imag_bits)) &&
           isfinite(bits_double(step->post_acc_real_bits)) &&
           isfinite(bits_double(step->post_acc_imag_bits));
}

int holo_path_history_append(HoloPathHistory *history, const HoloPathStep *step) {
    int rc;
    if (!history || !step || !history->steps) return HOLO_PATH_ERR_NULL;
    if (history->sealed || !history->appendable) return HOLO_PATH_ERR_SEALED;
    if (step->step_index != history->count) return HOLO_PATH_ERR_ORDER;
    if (strcmp(step->operator_id, HOLO_PATH_OPERATOR) != 0) return HOLO_PATH_ERR_OPERATOR;
    if (!step_numeric_valid(step)) return HOLO_PATH_ERR_NUMERIC;
    if (step->step_digest != path_step_digest(step)) return HOLO_PATH_ERR_CORRUPT;
    if (history->count == 0) {
        if (step->pre_state_digest != history->initial_state_digest) return HOLO_PATH_ERR_CONTINUITY;
    } else if (step->pre_state_digest != history->steps[history->count - 1U].post_state_digest) {
        return HOLO_PATH_ERR_CONTINUITY;
    }
    if (history->count == history->capacity) {
        rc = grow_history(history);
        if (rc != HOLO_PATH_OK) return rc;
    }
    history->steps[history->count] = *step;
    history->count++;
    history->terminal_state_digest = step->post_state_digest;
    return HOLO_PATH_OK;
}

int holo_path_history_validate(const HoloPathHistory *history) {
    size_t i;
    if (!history || !history->steps || history->capacity == 0 ||
        history->count > history->capacity || !history->reversible) return HOLO_PATH_ERR_NULL;
    if ((history->sealed && history->appendable) ||
        (history->serialized_roundtrip && !history->restoration_verified) ||
        (history->restoration_verified &&
         history->restored_state_digest != history->initial_state_digest)) {
        return HOLO_PATH_ERR_CORRUPT;
    }
    for (i = 0; i < history->count; ++i) {
        const HoloPathStep *step = &history->steps[i];
        if (step->step_index != i) return HOLO_PATH_ERR_ORDER;
        if (strcmp(step->operator_id, HOLO_PATH_OPERATOR) != 0) return HOLO_PATH_ERR_OPERATOR;
        if (!step_numeric_valid(step) || step->step_digest != path_step_digest(step)) {
            return HOLO_PATH_ERR_CORRUPT;
        }
        if ((i == 0 && step->pre_state_digest != history->initial_state_digest) ||
            (i > 0 && step->pre_state_digest != history->steps[i - 1U].post_state_digest)) {
            return HOLO_PATH_ERR_CONTINUITY;
        }
    }
    if (history->count > 0 &&
        history->terminal_state_digest != history->steps[history->count - 1U].post_state_digest) {
        return HOLO_PATH_ERR_CONTINUITY;
    }
    return HOLO_PATH_OK;
}

int holo_path_history_seal(HoloPathHistory *history) {
    int rc = holo_path_history_validate(history);
    if (rc != HOLO_PATH_OK) return rc;
    history->sealed = 1;
    history->appendable = 0;
    return HOLO_PATH_OK;
}

int holo_path_apply_step(HoloPathHistory *history, OrbitState *state,
                         uint64_t operator_parameter) {
    OrbitState next;
    HoloPathStep step;
    double theta_plus;
    double theta_minus;
    int rc;
    if (!history || !state) return HOLO_PATH_ERR_NULL;
    if (state->N <= 0 || state->steps < 0 || (size_t)state->steps != history->count ||
        operator_parameter == 0 || operator_parameter > (uint64_t)state->N) {
        return HOLO_PATH_ERR_ORDER;
    }
    next = *state;
    theta_plus = 2.0 * M_PI * state->branch_plus * (double)operator_parameter / state->N;
    theta_minus = 2.0 * M_PI * state->branch_minus * (double)operator_parameter / state->N;
    next.acc_real += cos(theta_plus) + cos(theta_minus);
    next.acc_imag += sin(theta_plus) + sin(theta_minus);
    next.steps++;
    if (!isfinite(next.acc_real) || !isfinite(next.acc_imag)) return HOLO_PATH_ERR_NUMERIC;

    memset(&step, 0, sizeof(step));
    step.step_index = (uint32_t)history->count;
    snprintf(step.operator_id, sizeof(step.operator_id), "%s", HOLO_PATH_OPERATOR);
    step.operator_parameter = operator_parameter;
    step.pre_acc_real_bits = double_bits(state->acc_real);
    step.pre_acc_imag_bits = double_bits(state->acc_imag);
    step.post_acc_real_bits = double_bits(next.acc_real);
    step.post_acc_imag_bits = double_bits(next.acc_imag);
    step.pre_state_digest = holo_orbit_state_digest(state);
    step.post_state_digest = holo_orbit_state_digest(&next);
    step.step_digest = path_step_digest(&step);
    rc = holo_path_history_append(history, &step);
    if (rc != HOLO_PATH_OK) return rc;
    *state = next;
    return HOLO_PATH_OK;
}

static uint64_t path_rng(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * UINT64_C(0x2545F4914F6CDD1D);
}

int holo_path_evolve(HoloPathHistory *history, OrbitState *state,
                     const EvolParams *params) {
    uint64_t rng;
    int i;
    int rc;
    if (!history || !state || !params) return HOLO_PATH_ERR_NULL;
    if (params->max_steps < 0 || params->max_steps > ORBIT_MAX_STEPS) return HOLO_PATH_ERR_CAPACITY;
    rng = params->seed | UINT64_C(1);
    for (i = 0; i < params->max_steps; ++i) {
        uint64_t parameter = path_rng(&rng) % (uint64_t)state->N + UINT64_C(1);
        rc = holo_path_apply_step(history, state, parameter);
        if (rc != HOLO_PATH_OK) return rc;
    }
    return HOLO_PATH_OK;
}

int holo_path_reverse(HoloPathHistory *history, const OrbitState *terminal_state,
                      OrbitState *restored_state) {
    OrbitState current;
    size_t i;
    int rc;
    if (!history || !terminal_state || !restored_state) return HOLO_PATH_ERR_NULL;
    rc = holo_path_history_validate(history);
    if (rc != HOLO_PATH_OK) return rc;
    if (holo_orbit_state_digest(terminal_state) != history->terminal_state_digest) {
        return HOLO_PATH_ERR_CONTINUITY;
    }
    current = *terminal_state;
    for (i = history->count; i > 0; --i) {
        const HoloPathStep *step = &history->steps[i - 1U];
        if (holo_orbit_state_digest(&current) != step->post_state_digest) {
            return HOLO_PATH_ERR_CONTINUITY;
        }
        current.acc_real = bits_double(step->pre_acc_real_bits);
        current.acc_imag = bits_double(step->pre_acc_imag_bits);
        current.steps = (int)step->step_index;
        if (holo_orbit_state_digest(&current) != step->pre_state_digest) {
            return HOLO_PATH_ERR_RESTORATION;
        }
    }
    history->restored_state_digest = holo_orbit_state_digest(&current);
    history->restoration_verified =
        history->restored_state_digest == history->initial_state_digest;
    if (!history->restoration_verified) return HOLO_PATH_ERR_RESTORATION;
    *restored_state = current;
    return HOLO_PATH_OK;
}

int holo_path_history_write_json(FILE *file, const HoloPathHistory *history) {
    size_t i;
    if (!file || holo_path_history_validate(history) != HOLO_PATH_OK) return HOLO_PATH_ERR_CORRUPT;
    fprintf(file, "  \"path_history\": {\n");
    fprintf(file, "    \"representation\": \"exact_accumulator_state_bits_v1\",\n");
    fprintf(file, "    \"count\": %zu, \"capacity\": %zu, \"appendable\": %s, \"reversible\": %s, \"sealed\": %s,\n",
            history->count, history->capacity, history->appendable ? "true" : "false",
            history->reversible ? "true" : "false", history->sealed ? "true" : "false");
    fprintf(file, "    \"initial_state_digest\": \"%016" PRIx64 "\", \"terminal_state_digest\": \"%016" PRIx64 "\", \"restored_state_digest\": \"%016" PRIx64 "\",\n",
            history->initial_state_digest, history->terminal_state_digest,
            history->restored_state_digest);
    fprintf(file, "    \"restoration_verified\": %s, \"serialized_roundtrip\": %s,\n",
            history->restoration_verified ? "true" : "false",
            history->serialized_roundtrip ? "true" : "false");
    fprintf(file, "    \"steps\": [\n");
    for (i = 0; i < history->count; ++i) {
        const HoloPathStep *step = &history->steps[i];
        fprintf(file, "      {\"step_index\":%u,\"operator_id\":\"%s\",\"operator_parameter\":%" PRIu64 ",\"pre_acc_real_bits\":\"%016" PRIx64 "\",\"pre_acc_imag_bits\":\"%016" PRIx64 "\",\"post_acc_real_bits\":\"%016" PRIx64 "\",\"post_acc_imag_bits\":\"%016" PRIx64 "\",\"pre_state_digest\":\"%016" PRIx64 "\",\"post_state_digest\":\"%016" PRIx64 "\",\"step_digest\":\"%016" PRIx64 "\"}%s\n",
                step->step_index, step->operator_id, step->operator_parameter,
                step->pre_acc_real_bits, step->pre_acc_imag_bits,
                step->post_acc_real_bits, step->post_acc_imag_bits,
                step->pre_state_digest, step->post_state_digest, step->step_digest,
                i + 1U == history->count ? "" : ",");
    }
    fprintf(file, "    ]\n  },\n");
    return ferror(file) ? HOLO_PATH_ERR_IO : HOLO_PATH_OK;
}

static const char *find_value(const char *json, const char *key) {
    char needle[96];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    return strstr(json, needle);
}

static int parse_size(const char *json, const char *key, size_t *value) {
    const char *field = find_value(json, key);
    unsigned long long parsed;
    if (!field || sscanf(strchr(field, ':') + 1, "%llu", &parsed) != 1 || parsed > SIZE_MAX) return 0;
    *value = (size_t)parsed;
    return 1;
}

static int parse_bool(const char *json, const char *key, int *value) {
    const char *field = find_value(json, key);
    const char *colon;
    if (!field || !(colon = strchr(field, ':'))) return 0;
    while (*++colon == ' ') { }
    if (strncmp(colon, "true", 4) == 0) *value = 1;
    else if (strncmp(colon, "false", 5) == 0) *value = 0;
    else return 0;
    return 1;
}

static int parse_hex(const char *json, const char *key, uint64_t *value) {
    const char *field = find_value(json, key);
    if (!field || sscanf(strchr(field, ':') + 1, " \"%16" SCNx64 "\"", value) != 1) return 0;
    return 1;
}

int holo_path_history_read_json(const char *json, HoloPathHistory **history_out) {
    const char *section;
    const char *cursor;
    HoloPathHistory *history;
    OrbitState placeholder;
    size_t count;
    size_t capacity;
    size_t i;
    int appendable;
    int reversible;
    int sealed;
    int restoration_verified;
    int serialized_roundtrip;
    if (!json || !history_out) return HOLO_PATH_ERR_NULL;
    *history_out = NULL;
    section = strstr(json, "\"path_history\"");
    if (!section || !parse_size(section, "count", &count) ||
        !parse_size(section, "capacity", &capacity) || count > capacity || capacity == 0 ||
        capacity > HOLO_PATH_MAX_SERIALIZED_STEPS ||
        !parse_bool(section, "appendable", &appendable) ||
        !parse_bool(section, "reversible", &reversible) ||
        !parse_bool(section, "sealed", &sealed) ||
        !parse_bool(section, "restoration_verified", &restoration_verified) ||
        !parse_bool(section, "serialized_roundtrip", &serialized_roundtrip)) return HOLO_PATH_ERR_PARSE;
    memset(&placeholder, 0, sizeof(placeholder));
    history = (HoloPathHistory *)malloc(sizeof(*history));
    if (!history) return HOLO_PATH_ERR_CAPACITY;
    if (holo_path_history_init(history, capacity, &placeholder) != HOLO_PATH_OK) {
        free(history); return HOLO_PATH_ERR_CAPACITY;
    }
    if (!parse_hex(section, "initial_state_digest", &history->initial_state_digest) ||
        !parse_hex(section, "terminal_state_digest", &history->terminal_state_digest) ||
        !parse_hex(section, "restored_state_digest", &history->restored_state_digest)) {
        holo_path_history_destroy(history); free(history); return HOLO_PATH_ERR_PARSE;
    }
    cursor = strstr(section, "\"steps\"");
    if (!cursor) { holo_path_history_destroy(history); free(history); return HOLO_PATH_ERR_PARSE; }
    for (i = 0; i < count; ++i) {
        HoloPathStep *step = &history->steps[i];
        unsigned int step_index;
        cursor = strstr(cursor, "{\"step_index\":");
        if (!cursor || sscanf(cursor,
            "{\"step_index\":%u,\"operator_id\":\"%39[^\"]\",\"operator_parameter\":%" SCNu64 ",\"pre_acc_real_bits\":\"%16" SCNx64 "\",\"pre_acc_imag_bits\":\"%16" SCNx64 "\",\"post_acc_real_bits\":\"%16" SCNx64 "\",\"post_acc_imag_bits\":\"%16" SCNx64 "\",\"pre_state_digest\":\"%16" SCNx64 "\",\"post_state_digest\":\"%16" SCNx64 "\",\"step_digest\":\"%16" SCNx64 "\"}",
            &step_index, step->operator_id, &step->operator_parameter,
            &step->pre_acc_real_bits, &step->pre_acc_imag_bits,
            &step->post_acc_real_bits, &step->post_acc_imag_bits,
            &step->pre_state_digest, &step->post_state_digest,
            &step->step_digest) != 10) {
            holo_path_history_destroy(history); free(history); return HOLO_PATH_ERR_PARSE;
        }
        step->step_index = step_index;
        cursor++;
    }
    history->count = count;
    history->appendable = appendable;
    history->reversible = reversible;
    history->sealed = sealed;
    history->restoration_verified = restoration_verified;
    history->serialized_roundtrip = serialized_roundtrip;
    if (holo_path_history_validate(history) != HOLO_PATH_OK) {
        holo_path_history_destroy(history); free(history); return HOLO_PATH_ERR_CORRUPT;
    }
    *history_out = history;
    return HOLO_PATH_OK;
}

int holo_path_history_read_file(const char *path, HoloPathHistory **history_out) {
    FILE *file;
    long size;
    char *json;
    int rc;
    if (!path || !history_out) return HOLO_PATH_ERR_NULL;
    file = fopen(path, "rb");
    if (!file) return HOLO_PATH_ERR_IO;
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) { fclose(file); return HOLO_PATH_ERR_IO; }
    json = (char *)malloc((size_t)size + 1U);
    if (!json) { fclose(file); return HOLO_PATH_ERR_CAPACITY; }
    if (fread(json, 1, (size_t)size, file) != (size_t)size) {
        free(json); fclose(file); return HOLO_PATH_ERR_IO;
    }
    json[size] = '\0';
    fclose(file);
    rc = holo_path_history_read_json(json, history_out);
    free(json);
    return rc;
}
