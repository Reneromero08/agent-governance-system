#define _POSIX_C_SOURCE 200809L

/* Separate scalar and exhaustive adjudicator for algebraic_cycle_phase.c. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CCOUNT 4U
#define RCOUNT 6U
#define LINE_CAP 512U
#define TOKEN_CAP 12U

enum role {
    LEAF_A = 0,
    UPPER_LEFT = 1,
    UPPER_RIGHT = 2,
    LOWER_LEFT = 3,
    LOWER_RIGHT = 4,
    LEAF_B = 5
};

struct relation {
    int c[CCOUNT];
    int present;
};

struct process {
    struct relation relation[RCOUNT];
    uint64_t source_hash;
};

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void fail_line(const char *message, size_t line) {
    fprintf(stderr, "%s at line %zu\n", message, line);
    exit(2);
}

static uint64_t hash_bytes(
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
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static int evaluate(const int c[CCOUNT], int x, int y) {
    return f3(c[0] + c[1] * x + c[2] * y + c[3] * x * y);
}

static size_t tokenize(char *line, char *token[TOKEN_CAP]) {
    size_t count = 0U;
    char *save = NULL;
    for (
        char *part = strtok_r(line, " ", &save);
        part != NULL;
        part = strtok_r(NULL, " ", &save)
    ) {
        if (count == TOKEN_CAP) {
            fail("too many tokens");
        }
        token[count++] = part;
    }
    return count;
}

static enum role parse_role(const char *text, size_t line) {
    static const char *const name[RCOUNT] = {
        "LEAF_A", "UPPER_LEFT", "UPPER_RIGHT",
        "LOWER_LEFT", "LOWER_RIGHT", "LEAF_B"
    };
    for (size_t index = 0U; index < RCOUNT; ++index) {
        if (strcmp(text, name[index]) == 0) {
            return (enum role)index;
        }
    }
    fail_line("unknown role", line);
    return LEAF_A;
}

static const char *role_first(enum role role) {
    static const char *const first[RCOUNT] = {
        "A", "U", "W", "U", "Z", "B"
    };
    return first[(size_t)role];
}

static const char *role_second(enum role role) {
    static const char *const second[RCOUNT] = {
        "U", "W", "V", "Z", "V", "V"
    };
    return second[(size_t)role];
}

static void transpose(int output[CCOUNT], const int input[CCOUNT]) {
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
        .source_hash = UINT64_C(14695981039346656037)
    };
    char line[LINE_CAP];
    size_t line_number = 0U;
    int header = 0;
    int type = 0;
    int topology = 0;
    int end = 0;
    while (fgets(line, sizeof(line), stream) != NULL) {
        ++line_number;
        const size_t length = strlen(line);
        process.source_hash = hash_bytes(
            process.source_hash,
            (const unsigned char *)line,
            length
        );
        if (length == 0U || line[length - 1U] != '\n') {
            fail_line("record lacks LF", line_number);
        }
        if (memchr(line, '\r', length) != NULL) {
            fail_line("CR is forbidden", line_number);
        }
        line[length - 1U] = '\0';
        if (end) {
            fail_line("record after END", line_number);
        }
        char *token[TOKEN_CAP] = {0};
        const size_t count = tokenize(line, token);
        if (count == 0U) {
            fail_line("blank record", line_number);
        }
        if (!header) {
            if (
                count != 2U
                || strcmp(token[0], "CATCAS_ALGEBRAIC_CYCLE") != 0
                || strcmp(token[1], "1") != 0
            ) {
                fail_line("invalid header", line_number);
            }
            header = 1;
        } else if (strcmp(token[0], "TYPE") == 0) {
            if (
                type
                || count != 2U
                || strcmp(token[1], "BOOLEAN_F3") != 0
            ) {
                fail_line("invalid TYPE", line_number);
            }
            type = 1;
        } else if (strcmp(token[0], "TOPOLOGY") == 0) {
            if (
                !type
                || topology
                || count != 2U
                || strcmp(token[1], "DIAMOND_U_W_Z_V") != 0
            ) {
                fail_line("invalid TOPOLOGY", line_number);
            }
            topology = 1;
        } else if (strcmp(token[0], "RELATION") == 0) {
            if (!type || !topology || count != 8U) {
                fail_line("invalid RELATION", line_number);
            }
            const enum role role = parse_role(token[1], line_number);
            struct relation *relation = &process.relation[(size_t)role];
            if (relation->present) {
                fail_line("duplicate relation", line_number);
            }
            int parsed[CCOUNT];
            for (size_t index = 0U; index < CCOUNT; ++index) {
                if (
                    strlen(token[4U + index]) != 1U
                    || token[4U + index][0] < '0'
                    || token[4U + index][0] > '2'
                ) {
                    fail_line("invalid coefficient", line_number);
                }
                parsed[index] = token[4U + index][0] - '0';
            }
            if (
                strcmp(token[2], role_first(role)) == 0
                && strcmp(token[3], role_second(role)) == 0
            ) {
                memcpy(relation->c, parsed, sizeof(parsed));
            } else if (
                strcmp(token[2], role_second(role)) == 0
                && strcmp(token[3], role_first(role)) == 0
            ) {
                transpose(relation->c, parsed);
            } else {
                fail_line("wrong relation endpoints", line_number);
            }
            relation->present = 1;
        } else if (strcmp(token[0], "END") == 0) {
            if (count != 1U) {
                fail_line("invalid END", line_number);
            }
            end = 1;
        } else {
            fail_line("unknown record", line_number);
        }
    }
    if (ferror(stream) || fclose(stream) != 0) {
        fail("read failed");
    }
    if (!header || !type || !topology || !end) {
        fail("process incomplete");
    }
    for (size_t role = 0U; role < RCOUNT; ++role) {
        if (!process.relation[role].present) {
            fail("missing relation");
        }
    }
    return process;
}

static void add(
    const int left[CCOUNT],
    const int right[CCOUNT],
    int output[CCOUNT]
) {
    for (size_t index = 0U; index < CCOUNT; ++index) {
        output[index] = f3(left[index] + right[index]);
    }
}

static void multiply(
    const int left[CCOUNT],
    const int right[CCOUNT],
    int output[CCOUNT]
) {
    memset(output, 0, CCOUNT * sizeof(*output));
    for (size_t l = 0U; l < CCOUNT; ++l) {
        for (size_t r = 0U; r < CCOUNT; ++r) {
            output[l | r] = f3(
                output[l | r] + left[l] * right[r]
            );
        }
    }
}

static void exact_compose(
    const int left[CCOUNT],
    const int right[CCOUNT],
    int output[CCOUNT]
) {
    const int f0[CCOUNT] = {left[0], left[1], 0, 0};
    const int f1[CCOUNT] = {
        f3(left[0] + left[2]),
        f3(left[1] + left[3]),
        0,
        0
    };
    const int g0[CCOUNT] = {right[0], 0, right[2], 0};
    const int g1[CCOUNT] = {
        f3(right[0] + right[1]),
        0,
        f3(right[2] + right[3]),
        0
    };
    int f0s[CCOUNT];
    int f1s[CCOUNT];
    int g0s[CCOUNT];
    int g1s[CCOUNT];
    int k0[CCOUNT];
    int k1[CCOUNT];
    multiply(f0, f0, f0s);
    multiply(f1, f1, f1s);
    multiply(g0, g0, g0s);
    multiply(g1, g1, g1s);
    add(f0s, g0s, k0);
    add(f1s, g1s, k1);
    multiply(k0, k1, output);
}

static void intersection(
    const int first[CCOUNT],
    const int second[CCOUNT],
    int output[CCOUNT]
) {
    int first_squared[CCOUNT];
    int second_squared[CCOUNT];
    multiply(first, first, first_squared);
    multiply(second, second, second_squared);
    add(first_squared, second_squared, output);
}

static unsigned zero_mask(const int c[CCOUNT]) {
    unsigned mask = 0U;
    for (int first = 0; first < 2; ++first) {
        for (int second = 0; second < 2; ++second) {
            if (evaluate(c, first, second) == 0) {
                mask |= 1U << (2 * first + second);
            }
        }
    }
    return mask;
}

static int edge_accepts(
    const struct process *process,
    enum role role,
    int first,
    int second
) {
    return evaluate(process->relation[(size_t)role].c, first, second) == 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s CYCLE.arc\n", argv[0]);
        return 2;
    }
    const struct process process = read_process(argv[1]);
    int upper[CCOUNT];
    int lower[CCOUNT];
    int core[CCOUNT];
    int a_to_v[CCOUNT];
    int leaf_b_transposed[CCOUNT];
    int boundary[CCOUNT];
    exact_compose(
        process.relation[UPPER_LEFT].c,
        process.relation[UPPER_RIGHT].c,
        upper
    );
    exact_compose(
        process.relation[LOWER_LEFT].c,
        process.relation[LOWER_RIGHT].c,
        lower
    );
    intersection(upper, lower, core);
    exact_compose(process.relation[LEAF_A].c, core, a_to_v);
    transpose(leaf_b_transposed, process.relation[LEAF_B].c);
    exact_compose(a_to_v, leaf_b_transposed, boundary);

    unsigned exact_mask = 0U;
    unsigned multi_witness = 0U;
    uint64_t checked_rows = 0U;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            unsigned witnesses = 0U;
            for (int u = 0; u < 2; ++u) {
                for (int v = 0; v < 2; ++v) {
                    for (int w = 0; w < 2; ++w) {
                        for (int z = 0; z < 2; ++z) {
                            ++checked_rows;
                            witnesses += (unsigned)(
                                edge_accepts(&process, LEAF_A, a, u)
                                && edge_accepts(
                                    &process,
                                    UPPER_LEFT,
                                    u,
                                    w
                                )
                                && edge_accepts(
                                    &process,
                                    UPPER_RIGHT,
                                    w,
                                    v
                                )
                                && edge_accepts(
                                    &process,
                                    LOWER_LEFT,
                                    u,
                                    z
                                )
                                && edge_accepts(
                                    &process,
                                    LOWER_RIGHT,
                                    z,
                                    v
                                )
                                && edge_accepts(&process, LEAF_B, b, v)
                            );
                        }
                    }
                }
            }
            if (witnesses != 0U) {
                exact_mask |= 1U << (2 * a + b);
            }
            multi_witness += (unsigned)(witnesses > 1U);
        }
    }
    const unsigned projected_mask = zero_mask(boundary);
    printf(
        "{\"mode\":\"scalar-cycle-reference\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"boundary_coefficients\":[%d,%d,%d,%d],"
        "\"projected_zero_mask\":%u,"
        "\"exact_zero_mask\":%u,"
        "\"complete_assignment_rows\":%llu,"
        "\"multi_witness_boundary_rows\":%u,"
        "\"mismatches\":%u}\n",
        (unsigned long long)process.source_hash,
        boundary[0],
        boundary[1],
        boundary[2],
        boundary[3],
        projected_mask,
        exact_mask,
        (unsigned long long)checked_rows,
        multi_witness,
        projected_mask == exact_mask ? 0U : 1U
    );
    return projected_mask == exact_mask ? 0 : 1;
}
