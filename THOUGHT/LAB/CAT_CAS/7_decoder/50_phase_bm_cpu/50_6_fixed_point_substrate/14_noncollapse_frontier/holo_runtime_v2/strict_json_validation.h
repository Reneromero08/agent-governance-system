#ifndef CAT_CAS_STRICT_JSON_VALIDATION_H
#define CAT_CAS_STRICT_JSON_VALIDATION_H

#include <ctype.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    const char *cursor;
    unsigned depth;
} StrictJsonCursor;

static void sj_skip_ws(StrictJsonCursor *state) {
    while (isspace((unsigned char)*state->cursor)) state->cursor++;
}

static int sj_hex(char c) {
    return (c >= '0' && c <= '9') ||
           (c >= 'a' && c <= 'f') ||
           (c >= 'A' && c <= 'F');
}

static int sj_string(StrictJsonCursor *state, char *out, size_t out_size) {
    sj_skip_ws(state);
    if (*state->cursor++ != '"') return -1;
    size_t used = 0;
    while (*state->cursor && *state->cursor != '"') {
        unsigned char c = (unsigned char)*state->cursor++;
        if (c < 0x20) return -1;
        if (c == '\\') {
            char escape = *state->cursor++;
            if (!escape) return -1;
            if (escape == 'u') {
                for (int i = 0; i < 4; i++) {
                    if (!sj_hex(*state->cursor++)) return -1;
                }
                c = '?';
            } else if (strchr("\"\\/bfnrt", escape)) {
                c = (unsigned char)escape;
            } else {
                return -1;
            }
        }
        if (out) {
            if (used + 1 >= out_size) return -1;
            out[used++] = (char)c;
        }
    }
    if (*state->cursor++ != '"') return -1;
    if (out) out[used] = 0;
    return 0;
}

static int sj_value(StrictJsonCursor *state);

static int sj_number(StrictJsonCursor *state) {
    const char *p = state->cursor;
    if (*p == '-') p++;
    if (*p == '0') {
        p++;
        if (isdigit((unsigned char)*p)) return -1;
    } else {
        if (!isdigit((unsigned char)*p)) return -1;
        while (isdigit((unsigned char)*p)) p++;
    }
    if (*p == '.') {
        p++;
        if (!isdigit((unsigned char)*p)) return -1;
        while (isdigit((unsigned char)*p)) p++;
    }
    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '+' || *p == '-') p++;
        if (!isdigit((unsigned char)*p)) return -1;
        while (isdigit((unsigned char)*p)) p++;
    }
    state->cursor = p;
    return 0;
}

static int sj_array(StrictJsonCursor *state) {
    if (state->depth++ >= 64) return -1;
    state->cursor++;
    sj_skip_ws(state);
    if (*state->cursor == ']') {
        state->cursor++;
        state->depth--;
        return 0;
    }
    for (;;) {
        if (sj_value(state)) return -1;
        sj_skip_ws(state);
        if (*state->cursor == ']') {
            state->cursor++;
            state->depth--;
            return 0;
        }
        if (*state->cursor++ != ',') return -1;
    }
}

static int sj_object(StrictJsonCursor *state) {
    if (state->depth++ >= 64) return -1;
    state->cursor++;
    sj_skip_ws(state);
    if (*state->cursor == '}') {
        state->cursor++;
        state->depth--;
        return 0;
    }
    for (;;) {
        if (sj_string(state, NULL, 0)) return -1;
        sj_skip_ws(state);
        if (*state->cursor++ != ':') return -1;
        if (sj_value(state)) return -1;
        sj_skip_ws(state);
        if (*state->cursor == '}') {
            state->cursor++;
            state->depth--;
            return 0;
        }
        if (*state->cursor++ != ',') return -1;
    }
}

static int sj_value(StrictJsonCursor *state) {
    sj_skip_ws(state);
    switch (*state->cursor) {
        case '"': return sj_string(state, NULL, 0);
        case '{': return sj_object(state);
        case '[': return sj_array(state);
        case 't':
            if (strncmp(state->cursor, "true", 4)) return -1;
            state->cursor += 4;
            return 0;
        case 'f':
            if (strncmp(state->cursor, "false", 5)) return -1;
            state->cursor += 5;
            return 0;
        case 'n':
            if (strncmp(state->cursor, "null", 4)) return -1;
            state->cursor += 4;
            return 0;
        default: return sj_number(state);
    }
}

static int strict_json_document(const char *json) {
    StrictJsonCursor state = {json, 0};
    if (!json || sj_value(&state)) return -1;
    sj_skip_ws(&state);
    return *state.cursor == 0 ? 0 : -1;
}

static int strict_json_exact_top_object(const char *json,
                                        const char *const *allowed,
                                        size_t allowed_count) {
    if (!json || !allowed || allowed_count == 0 || allowed_count > 64) return -1;
    StrictJsonCursor state = {json, 0};
    unsigned char seen[64] = {0};
    size_t members = 0;
    sj_skip_ws(&state);
    if (*state.cursor++ != '{') return -1;
    sj_skip_ws(&state);
    if (*state.cursor == '}') return -1;
    for (;;) {
        char key[160];
        if (sj_string(&state, key, sizeof(key))) return -1;
        size_t match = allowed_count;
        for (size_t i = 0; i < allowed_count; i++) {
            if (!strcmp(key, allowed[i])) {
                match = i;
                break;
            }
        }
        if (match == allowed_count || seen[match]) return -1;
        seen[match] = 1;
        members++;
        sj_skip_ws(&state);
        if (*state.cursor++ != ':') return -1;
        if (sj_value(&state)) return -1;
        sj_skip_ws(&state);
        if (*state.cursor == '}') {
            state.cursor++;
            break;
        }
        if (*state.cursor++ != ',') return -1;
    }
    sj_skip_ws(&state);
    if (*state.cursor != 0 || members != allowed_count) return -1;
    for (size_t i = 0; i < allowed_count; i++) {
        if (!seen[i]) return -1;
    }
    return 0;
}



static int sj_count_key(const char *json, const char *name) {
    if (!json || !name) return 0;
    StrictJsonCursor state = {json, 0};
    sj_skip_ws(&state);
    if (*state.cursor++ != '{') return 0;
    sj_skip_ws(&state);
    int count = 0;
    while (*state.cursor && *state.cursor != '}') {
        char key[160];
        if (sj_string(&state, key, sizeof(key))) return 0;
        sj_skip_ws(&state);
        if (*state.cursor++ != ':') return 0;
        if (sj_value(&state)) return 0;
        if (!strcmp(key, name)) count++;
        sj_skip_ws(&state);
        if (*state.cursor == '}') { state.cursor++; break; }
        if (*state.cursor++ != ',') return 0;
    }
    return count;
}

static const char *sj_object_value(const char *json, const char *name) {
    if (!json || !name) return NULL;
    StrictJsonCursor state = {json, 0};
    sj_skip_ws(&state);
    if (*state.cursor++ != '{') return NULL;
    sj_skip_ws(&state);
    int found = 0;
    const char *result = NULL;
    while (*state.cursor && *state.cursor != '}') {
        char key[160];
        if (sj_string(&state, key, sizeof(key))) return NULL;
        sj_skip_ws(&state);
        if (*state.cursor++ != ':') return NULL;
        const char *val_start = state.cursor;
        if (sj_value(&state)) return NULL;
        if (!strcmp(key, name)) {
            if (found) return NULL;
            found = 1;
            result = val_start;
        }
        sj_skip_ws(&state);
        if (*state.cursor == '}') { state.cursor++; break; }
        if (*state.cursor++ != ',') return 0;
    }
    return result;
}

#endif
