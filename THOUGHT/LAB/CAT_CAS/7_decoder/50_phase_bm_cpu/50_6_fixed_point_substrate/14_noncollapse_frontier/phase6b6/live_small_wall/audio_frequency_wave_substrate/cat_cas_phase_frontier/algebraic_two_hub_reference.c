#define _POSIX_C_SOURCE 200809L

/*
 * Independent bounded scalar reference for algebraic_two_hub_phase.c.
 * This executable is not linked into the native phase engine.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COEFFICIENT_COUNT 4U
#define RELATION_COUNT 5U
#define BOUNDARY_COUNT 6U
#define LINE_CAPACITY 512U
#define TOKEN_CAPACITY 12U

enum role {
    ROLE_LEFT_A = 0,
    ROLE_LEFT_B = 1,
    ROLE_BRIDGE = 2,
    ROLE_RIGHT_C = 3,
    ROLE_RIGHT_D = 4
};

struct relation_spec {
    int coefficient[COEFFICIENT_COUNT];
    int present;
};

struct process {
    struct relation_spec relation[RELATION_COUNT];
    uint64_t source_fnv1a64;
};

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void fail_line(const char *message, size_t line_number) {
    fprintf(stderr, "%s at line %zu\n", message, line_number);
    exit(2);
}

static uint64_t fnv1a64_update(
    uint64_t hash,
    const unsigned char *bytes,
    size_t count
) {
    for (size_t index = 0U; index < count; ++index) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int f3(int value) {
    int normalized = value % 3;
    if (normalized < 0) {
        normalized += 3;
    }
    return normalized;
}

static int evaluate_relation(
    const int coefficient[COEFFICIENT_COUNT],
    int first,
    int second
) {
    return f3(
        coefficient[0]
        + coefficient[1] * first
        + coefficient[2] * second
        + coefficient[3] * first * second
    );
}

static int relation_is_bi_total(
    const int coefficient[COEFFICIENT_COUNT]
) {
    for (int first = 0; first < 2; ++first) {
        int has_second = 0;
        for (int second = 0; second < 2; ++second) {
            has_second |= evaluate_relation(
                coefficient,
                first,
                second
            ) == 0;
        }
        if (!has_second) {
            return 0;
        }
    }
    for (int second = 0; second < 2; ++second) {
        int has_first = 0;
        for (int first = 0; first < 2; ++first) {
            has_first |= evaluate_relation(
                coefficient,
                first,
                second
            ) == 0;
        }
        if (!has_first) {
            return 0;
        }
    }
    return 1;
}

static int parse_coefficient(const char *text, size_t line_number) {
    if (
        strlen(text) != 1U
        || text[0] < '0'
        || text[0] > '2'
    ) {
        fail_line("coefficient must be one canonical F3 digit", line_number);
    }
    return text[0] - '0';
}

static size_t tokenize(
    char *line,
    char *token[TOKEN_CAPACITY]
) {
    size_t count = 0U;
    char *save = NULL;
    for (
        char *part = strtok_r(line, " ", &save);
        part != NULL;
        part = strtok_r(NULL, " ", &save)
    ) {
        if (count == TOKEN_CAPACITY) {
            fail("too many tokens");
        }
        token[count++] = part;
    }
    return count;
}

static enum role parse_role(const char *text, size_t line_number) {
    if (strcmp(text, "LEFT_A") == 0) {
        return ROLE_LEFT_A;
    }
    if (strcmp(text, "LEFT_B") == 0) {
        return ROLE_LEFT_B;
    }
    if (strcmp(text, "BRIDGE") == 0) {
        return ROLE_BRIDGE;
    }
    if (strcmp(text, "RIGHT_C") == 0) {
        return ROLE_RIGHT_C;
    }
    if (strcmp(text, "RIGHT_D") == 0) {
        return ROLE_RIGHT_D;
    }
    fail_line("unknown relation role", line_number);
    return ROLE_LEFT_A;
}

static const char *role_first(enum role role) {
    static const char *const first[RELATION_COUNT] = {
        "A", "B", "U", "C", "D"
    };
    return first[(size_t)role];
}

static const char *role_second(enum role role) {
    static const char *const second[RELATION_COUNT] = {
        "U", "U", "V", "V", "V"
    };
    return second[(size_t)role];
}

static void transpose_coefficients(
    int output[COEFFICIENT_COUNT],
    const int input[COEFFICIENT_COUNT]
) {
    output[0] = input[0];
    output[1] = input[2];
    output[2] = input[1];
    output[3] = input[3];
}

static struct process read_process(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct process process = {
        .source_fnv1a64 = UINT64_C(14695981039346656037)
    };
    char line[LINE_CAPACITY];
    size_t line_number = 0U;
    int header_seen = 0;
    int type_seen = 0;
    int hubs_seen = 0;
    int end_seen = 0;
    while (fgets(line, sizeof(line), stream) != NULL) {
        ++line_number;
        const size_t length = strlen(line);
        process.source_fnv1a64 = fnv1a64_update(
            process.source_fnv1a64,
            (const unsigned char *)line,
            length
        );
        if (length == 0U || line[length - 1U] != '\n') {
            fail_line("every record must end with LF", line_number);
        }
        if (memchr(line, '\r', length) != NULL) {
            fail_line("CR bytes are forbidden", line_number);
        }
        line[length - 1U] = '\0';
        if (end_seen) {
            fail_line("record after END", line_number);
        }
        char *token[TOKEN_CAPACITY] = {0};
        const size_t count = tokenize(line, token);
        if (count == 0U) {
            fail_line("blank records are forbidden", line_number);
        }
        if (!header_seen) {
            if (
                count != 2U
                || strcmp(token[0], "CATCAS_ALGEBRAIC_TWO_HUB_TREE") != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid two-hub header", line_number);
            }
            header_seen = 1;
            continue;
        }
        if (strcmp(token[0], "TYPE") == 0) {
            if (
                type_seen
                || hubs_seen
                || count != 2U
                || strcmp(token[1], "BOOLEAN_F3") != 0
            ) {
                fail_line("invalid TYPE", line_number);
            }
            type_seen = 1;
            continue;
        }
        if (strcmp(token[0], "HUBS") == 0) {
            if (
                !type_seen
                || hubs_seen
                || count != 3U
                || strcmp(token[1], "U") != 0
                || strcmp(token[2], "V") != 0
            ) {
                fail_line("invalid HUBS", line_number);
            }
            hubs_seen = 1;
            continue;
        }
        if (strcmp(token[0], "RELATION") == 0) {
            if (!type_seen || !hubs_seen || count != 8U) {
                fail_line("invalid RELATION", line_number);
            }
            const enum role role = parse_role(token[1], line_number);
            struct relation_spec *spec = &process.relation[(size_t)role];
            if (spec->present) {
                fail_line("duplicate relation role", line_number);
            }
            int parsed[COEFFICIENT_COUNT];
            for (
                size_t coefficient = 0U;
                coefficient < COEFFICIENT_COUNT;
                ++coefficient
            ) {
                parsed[coefficient] = parse_coefficient(
                    token[4U + coefficient],
                    line_number
                );
            }
            if (
                strcmp(token[2], role_first(role)) == 0
                && strcmp(token[3], role_second(role)) == 0
            ) {
                memcpy(spec->coefficient, parsed, sizeof(parsed));
            } else if (
                strcmp(token[2], role_second(role)) == 0
                && strcmp(token[3], role_first(role)) == 0
            ) {
                transpose_coefficients(spec->coefficient, parsed);
            } else {
                fail_line("relation endpoints do not match role", line_number);
            }
            if (!relation_is_bi_total(spec->coefficient)) {
                fail_line(
                    "relation is not bi-total on Boolean_F3",
                    line_number
                );
            }
            spec->present = 1;
            continue;
        }
        if (strcmp(token[0], "END") == 0) {
            if (count != 1U) {
                fail_line("invalid END", line_number);
            }
            end_seen = 1;
            continue;
        }
        fail_line("unknown two-hub record", line_number);
    }
    if (ferror(stream)) {
        fail("failed to read complete two-hub process");
    }
    if (fclose(stream) != 0) {
        perror(path);
        exit(2);
    }
    if (!header_seen || !type_seen || !hubs_seen || !end_seen) {
        fail("two-hub process is incomplete");
    }
    for (size_t role = 0U; role < RELATION_COUNT; ++role) {
        if (!process.relation[role].present) {
            fail("two-hub process is missing a relation role");
        }
    }
    return process;
}

static void resultant(
    int output[COEFFICIENT_COUNT],
    const int left[COEFFICIENT_COUNT],
    const int right[COEFFICIENT_COUNT]
) {
    output[0] = f3(left[0] * right[2] - left[2] * right[0]);
    output[1] = f3(left[1] * right[2] - left[3] * right[0]);
    output[2] = f3(left[0] * right[3] - left[2] * right[1]);
    output[3] = f3(left[1] * right[3] - left[3] * right[1]);
}

static void derive_boundary(
    const struct process *process,
    int boundary[BOUNDARY_COUNT][COEFFICIENT_COUNT]
) {
    int bridge_vu[COEFFICIENT_COUNT];
    int message_a[COEFFICIENT_COUNT];
    int message_b[COEFFICIENT_COUNT];
    transpose_coefficients(
        bridge_vu,
        process->relation[ROLE_BRIDGE].coefficient
    );
    resultant(
        message_a,
        process->relation[ROLE_LEFT_A].coefficient,
        bridge_vu
    );
    resultant(
        message_b,
        process->relation[ROLE_LEFT_B].coefficient,
        bridge_vu
    );
    resultant(
        boundary[0],
        process->relation[ROLE_LEFT_A].coefficient,
        process->relation[ROLE_LEFT_B].coefficient
    );
    resultant(
        boundary[1],
        process->relation[ROLE_RIGHT_C].coefficient,
        process->relation[ROLE_RIGHT_D].coefficient
    );
    resultant(
        boundary[2],
        message_a,
        process->relation[ROLE_RIGHT_C].coefficient
    );
    resultant(
        boundary[3],
        message_a,
        process->relation[ROLE_RIGHT_D].coefficient
    );
    resultant(
        boundary[4],
        message_b,
        process->relation[ROLE_RIGHT_C].coefficient
    );
    resultant(
        boundary[5],
        message_b,
        process->relation[ROLE_RIGHT_D].coefficient
    );
}

static uint64_t boundary_hash(
    int boundary[BOUNDARY_COUNT][COEFFICIENT_COUNT]
) {
    static const char *const label[BOUNDARY_COUNT] = {
        "A:B", "C:D", "A:C", "A:D", "B:C", "B:D"
    };
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0U; index < BOUNDARY_COUNT; ++index) {
        hash = fnv1a64_update(
            hash,
            (const unsigned char *)label[index],
            strlen(label[index])
        );
        for (
            size_t coefficient = 0U;
            coefficient < COEFFICIENT_COUNT;
            ++coefficient
        ) {
            const unsigned char byte =
                (unsigned char)boundary[index][coefficient];
            hash = fnv1a64_update(hash, &byte, 1U);
        }
    }
    return hash;
}

static int exact_exists(
    const struct process *process,
    int a,
    int b,
    int c,
    int d,
    int *witness_count
) {
    int count = 0;
    for (int u = 0; u < 2; ++u) {
        for (int v = 0; v < 2; ++v) {
            count += (
                evaluate_relation(
                    process->relation[ROLE_LEFT_A].coefficient,
                    a,
                    u
                ) == 0
                && evaluate_relation(
                    process->relation[ROLE_LEFT_B].coefficient,
                    b,
                    u
                ) == 0
                && evaluate_relation(
                    process->relation[ROLE_BRIDGE].coefficient,
                    u,
                    v
                ) == 0
                && evaluate_relation(
                    process->relation[ROLE_RIGHT_C].coefficient,
                    c,
                    v
                ) == 0
                && evaluate_relation(
                    process->relation[ROLE_RIGHT_D].coefficient,
                    d,
                    v
                ) == 0
            );
        }
    }
    *witness_count = count;
    return count > 0;
}

static int factorized_accepts(
    int boundary[BOUNDARY_COUNT][COEFFICIENT_COUNT],
    int a,
    int b,
    int c,
    int d
) {
    return (
        evaluate_relation(boundary[0], a, b) == 0
        && evaluate_relation(boundary[1], c, d) == 0
        && evaluate_relation(boundary[2], a, c) == 0
        && evaluate_relation(boundary[3], a, d) == 0
        && evaluate_relation(boundary[4], b, c) == 0
        && evaluate_relation(boundary[5], b, d) == 0
    );
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s PROCESS.aret\n", argv[0]);
        return 2;
    }
    const struct process process = read_process(argv[1]);
    int boundary[BOUNDARY_COUNT][COEFFICIENT_COUNT];
    derive_boundary(&process, boundary);
    int exact_rows = 0;
    int projected_rows = 0;
    int multi_witness_rows = 0;
    for (int assignment = 0; assignment < 16; ++assignment) {
        const int a = assignment & 1;
        const int b = (assignment >> 1) & 1;
        const int c = (assignment >> 2) & 1;
        const int d = (assignment >> 3) & 1;
        int witness_count = 0;
        const int exact = exact_exists(
            &process,
            a,
            b,
            c,
            d,
            &witness_count
        );
        const int factorized = factorized_accepts(
            boundary,
            a,
            b,
            c,
            d
        );
        exact_rows += exact == factorized;
        projected_rows += exact;
        multi_witness_rows += witness_count > 1;
    }
    printf(
        "{\"mode\":\"bounded-two-hub-existential-reference\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"closed_hub_count\":2,"
        "\"phase_resident_relation_messages\":2,"
        "\"boundary_factor_count\":6,"
        "\"boundary_factor_fnv1a64\":\"%016llx\","
        "\"assignment_rows\":16,"
        "\"exact_rows\":%d,"
        "\"projected_boundary_rows\":%d,"
        "\"multi_witness_rows\":%d,"
        "\"factorized_exact\":%s}\n",
        (unsigned long long)process.source_fnv1a64,
        (unsigned long long)boundary_hash(boundary),
        exact_rows,
        projected_rows,
        multi_witness_rows,
        exact_rows == 16 ? "true" : "false"
    );
    return exact_rows == 16 ? 0 : 1;
}
