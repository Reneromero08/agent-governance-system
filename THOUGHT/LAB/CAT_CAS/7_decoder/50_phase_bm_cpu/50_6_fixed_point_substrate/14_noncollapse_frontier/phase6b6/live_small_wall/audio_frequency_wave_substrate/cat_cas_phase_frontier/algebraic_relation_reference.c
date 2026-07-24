/*
 * Independent bounded scalar oracle for algebraic_relation_phase.c.
 *
 * This executable is not linked into the phase process.  It computes the
 * coefficient resultant with integer F3 arithmetic, separately enumerates
 * the eight Boolean (x,y,z) assignments, and checks that the resultant zero
 * set equals the existentially composed relation for the supplied fixture.
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define PORT_COUNT 3U
#define RELATION_COUNT 2U
#define COEFFICIENT_COUNT 4U
#define MAX_NAME 31U
#define MAX_FIELDS 12U

struct port_definition {
    char name[MAX_NAME + 1U];
};

struct relation_definition {
    char name[MAX_NAME + 1U];
    char first[MAX_NAME + 1U];
    char second[MAX_NAME + 1U];
    int coefficient[COEFFICIENT_COUNT];
};

struct raw_process {
    struct port_definition ports[PORT_COUNT];
    struct relation_definition relations[RELATION_COUNT];
    size_t port_count;
    size_t relation_count;
    char closed[MAX_NAME + 1U];
    char boundary_first[MAX_NAME + 1U];
    char boundary_second[MAX_NAME + 1U];
    int close_seen;
    int boundary_seen;
    int header_seen;
    int end_seen;
    uint64_t source_fnv1a64;
};

struct process {
    struct relation_definition left;
    struct relation_definition right;
    uint64_t source_fnv1a64;
};

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static uint64_t fnv1a64(const unsigned char *bytes, size_t length) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0U; index < length; ++index) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static char *trim(char *text) {
    while (isspace((unsigned char)*text)) {
        ++text;
    }
    char *end = text + strlen(text);
    while (end > text && isspace((unsigned char)end[-1])) {
        --end;
    }
    *end = '\0';
    return text;
}

static size_t fields(char *line, char *output[MAX_FIELDS]) {
    size_t count = 0U;
    char *save = NULL;
    for (
        char *field = strtok_r(line, " \t", &save);
        field != NULL;
        field = strtok_r(NULL, " \t", &save)
    ) {
        if (count == MAX_FIELDS) {
            fail("too many fields");
        }
        output[count++] = field;
    }
    return count;
}

static void name_copy(
    char output[MAX_NAME + 1U],
    const char *input
) {
    const size_t length = strlen(input);
    if (
        length == 0U
        || length > MAX_NAME
        || !(isalpha((unsigned char)input[0]) || input[0] == '_')
    ) {
        fail("invalid name");
    }
    for (size_t index = 1U; index < length; ++index) {
        if (
            !isalnum((unsigned char)input[index])
            && input[index] != '_'
        ) {
            fail("invalid name");
        }
    }
    memcpy(output, input, length + 1U);
}

static int coefficient(const char *text) {
    if (
        text[0] < '0'
        || text[0] > '2'
        || text[1] != '\0'
    ) {
        fail("invalid coefficient");
    }
    return text[0] - '0';
}

static int find_port(
    const struct raw_process *process,
    const char *name
) {
    for (size_t index = 0U; index < process->port_count; ++index) {
        if (strcmp(process->ports[index].name, name) == 0) {
            return (int)index;
        }
    }
    return -1;
}

static int connects(
    const struct relation_definition *relation,
    const char *first,
    const char *second
) {
    return (
        (
            strcmp(relation->first, first) == 0
            && strcmp(relation->second, second) == 0
        )
        || (
            strcmp(relation->first, second) == 0
            && strcmp(relation->second, first) == 0
        )
    );
}

static struct relation_definition orient(
    const struct relation_definition *input,
    const char *first,
    const char *second
) {
    struct relation_definition output = *input;
    if (
        strcmp(input->first, first) == 0
        && strcmp(input->second, second) == 0
    ) {
        return output;
    }
    if (
        strcmp(input->first, second) != 0
        || strcmp(input->second, first) != 0
    ) {
        fail("orientation mismatch");
    }
    name_copy(output.first, first);
    name_copy(output.second, second);
    const int swap = output.coefficient[1];
    output.coefficient[1] = output.coefficient[2];
    output.coefficient[2] = swap;
    return output;
}

static int mod3(int value) {
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static int affine_has_boolean_zero(int slope, int offset) {
    return offset == 0 || mod3(slope + offset) == 0;
}

static int left_total_toward_internal(
    const struct relation_definition *relation
) {
    return (
        affine_has_boolean_zero(
            relation->coefficient[2],
            relation->coefficient[0]
        )
        && affine_has_boolean_zero(
            mod3(
                relation->coefficient[2]
                + relation->coefficient[3]
            ),
            mod3(
                relation->coefficient[0]
                + relation->coefficient[1]
            )
        )
    );
}

static int right_total_toward_internal(
    const struct relation_definition *relation
) {
    return (
        affine_has_boolean_zero(
            relation->coefficient[1],
            relation->coefficient[0]
        )
        && affine_has_boolean_zero(
            mod3(
                relation->coefficient[1]
                + relation->coefficient[3]
            ),
            mod3(
                relation->coefficient[0]
                + relation->coefficient[2]
            )
        )
    );
}

static struct process normalize(const struct raw_process *raw) {
    if (
        !raw->header_seen
        || !raw->end_seen
        || raw->port_count != PORT_COUNT
        || raw->relation_count != RELATION_COUNT
        || !raw->close_seen
        || !raw->boundary_seen
        || find_port(raw, raw->closed) < 0
        || find_port(raw, raw->boundary_first) < 0
        || find_port(raw, raw->boundary_second) < 0
        || strcmp(raw->closed, raw->boundary_first) == 0
        || strcmp(raw->closed, raw->boundary_second) == 0
        || strcmp(
            raw->boundary_first,
            raw->boundary_second
        ) == 0
    ) {
        fail("incomplete or invalid process");
    }
    int left = -1;
    int right = -1;
    for (size_t index = 0U; index < RELATION_COUNT; ++index) {
        if (
            connects(
                &raw->relations[index],
                raw->boundary_first,
                raw->closed
            )
        ) {
            if (left >= 0) {
                fail("duplicate left relation");
            }
            left = (int)index;
        } else if (
            connects(
                &raw->relations[index],
                raw->closed,
                raw->boundary_second
            )
        ) {
            if (right >= 0) {
                fail("duplicate right relation");
            }
            right = (int)index;
        } else {
            fail("cut or foreign relation");
        }
    }
    if (left < 0 || right < 0) {
        fail("missing composable relation");
    }
    const struct process process = {
        .left = orient(
            &raw->relations[left],
            raw->boundary_first,
            raw->closed
        ),
        .right = orient(
            &raw->relations[right],
            raw->closed,
            raw->boundary_second
        ),
        .source_fnv1a64 = raw->source_fnv1a64
    };
    if (
        !left_total_toward_internal(&process.left)
        || !right_total_toward_internal(&process.right)
    ) {
        fail("relation is not total toward the closed Boolean port");
    }
    return process;
}

static struct process read_process(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    if (fseek(stream, 0L, SEEK_END) != 0) {
        fail("seek failed");
    }
    const long end = ftell(stream);
    if (end < 0L || end > 8192L) {
        fail("invalid byte count");
    }
    if (fseek(stream, 0L, SEEK_SET) != 0) {
        fail("seek failed");
    }
    const size_t length = (size_t)end;
    unsigned char *source = malloc(length + 1U);
    if (source == NULL) {
        fail("allocation failed");
    }
    if (fread(source, 1U, length, stream) != length) {
        fail("read failed");
    }
    if (memchr(source, '\0', length) != NULL) {
        fail("embedded NUL");
    }
    if (fclose(stream) != 0) {
        fail("close failed");
    }
    const uint64_t source_hash = fnv1a64(source, length);
    free(source);

    stream = fopen(path, "rb");
    if (stream == NULL) {
        fail("reopen failed");
    }
    struct raw_process raw = {
        .source_fnv1a64 = source_hash
    };
    char *buffer = NULL;
    size_t capacity = 0U;
    ssize_t length_read = 0;
    while ((length_read = getline(&buffer, &capacity, stream)) >= 0) {
        if (memchr(buffer, '\0', (size_t)length_read) != NULL) {
            fail("embedded NUL");
        }
        char *line = trim(buffer);
        if (*line == '\0' || *line == '#') {
            continue;
        }
        if (raw.end_seen) {
            fail("content after END");
        }
        char *field[MAX_FIELDS] = {0};
        const size_t count = fields(line, field);
        if (!raw.header_seen) {
            if (
                count != 2U
                || strcmp(
                    field[0],
                    "CATCAS_ALGEBRAIC_RELATION_PROCESS"
                ) != 0
                || strcmp(field[1], "1") != 0
            ) {
                fail("invalid header");
            }
            raw.header_seen = 1;
            continue;
        }
        if (strcmp(field[0], "PORT") == 0) {
            if (
                count != 3U
                || raw.port_count == PORT_COUNT
                || strcmp(field[2], "BOOLEAN_F3") != 0
            ) {
                fail("invalid port");
            }
            struct port_definition *port =
                &raw.ports[raw.port_count];
            name_copy(port->name, field[1]);
            for (size_t index = 0U; index < raw.port_count; ++index) {
                if (
                    strcmp(port->name, raw.ports[index].name) == 0
                ) {
                    fail("duplicate port name");
                }
            }
            ++raw.port_count;
            continue;
        }
        if (strcmp(field[0], "RELATION") == 0) {
            if (
                count != 9U
                || raw.relation_count == RELATION_COUNT
                || strcmp(field[8], "ZEROSET") != 0
            ) {
                fail("invalid relation");
            }
            struct relation_definition *relation =
                &raw.relations[raw.relation_count];
            name_copy(relation->name, field[1]);
            name_copy(relation->first, field[2]);
            name_copy(relation->second, field[3]);
            for (
                size_t index = 0U;
                index < COEFFICIENT_COUNT;
                ++index
            ) {
                relation->coefficient[index] =
                    coefficient(field[4U + index]);
            }
            for (
                size_t index = 0U;
                index < raw.relation_count;
                ++index
            ) {
                if (
                    strcmp(
                        relation->name,
                        raw.relations[index].name
                    ) == 0
                ) {
                    fail("duplicate relation name");
                }
            }
            ++raw.relation_count;
            continue;
        }
        if (strcmp(field[0], "CLOSE") == 0) {
            if (count != 2U || raw.close_seen) {
                fail("invalid close");
            }
            name_copy(raw.closed, field[1]);
            raw.close_seen = 1;
            continue;
        }
        if (strcmp(field[0], "BOUNDARY") == 0) {
            if (count != 3U || raw.boundary_seen) {
                fail("invalid boundary");
            }
            name_copy(raw.boundary_first, field[1]);
            name_copy(raw.boundary_second, field[2]);
            raw.boundary_seen = 1;
            continue;
        }
        if (strcmp(field[0], "END") == 0) {
            if (count != 1U) {
                fail("invalid END");
            }
            raw.end_seen = 1;
            continue;
        }
        fail("unknown record");
    }
    free(buffer);
    if (fclose(stream) != 0) {
        fail("close failed");
    }
    return normalize(&raw);
}

static int evaluate(
    const int coefficient_value[COEFFICIENT_COUNT],
    int first,
    int second
) {
    return mod3(
        coefficient_value[0]
        + coefficient_value[1] * first
        + coefficient_value[2] * second
        + coefficient_value[3] * first * second
    );
}

static void resultant(
    const struct process *process,
    int output[COEFFICIENT_COUNT]
) {
    const int *left = process->left.coefficient;
    const int *right = process->right.coefficient;
    output[0] = mod3(left[0] * right[1] - left[2] * right[0]);
    output[1] = mod3(left[1] * right[1] - left[3] * right[0]);
    output[2] = mod3(left[0] * right[3] - left[2] * right[2]);
    output[3] = mod3(left[1] * right[3] - left[3] * right[2]);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s PROCESS.arel\n", argv[0]);
        return 2;
    }
    const struct process process = read_process(argv[1]);
    int boundary[COEFFICIENT_COUNT];
    resultant(&process, boundary);
    unsigned int pair_mask = 0U;
    unsigned int boundary_pairs = 0U;
    unsigned int derivations = 0U;
    int exact = 1;
    for (int first = 0; first <= 1; ++first) {
        for (int second = 0; second <= 1; ++second) {
            int witnesses = 0;
            for (int internal = 0; internal <= 1; ++internal) {
                if (
                    evaluate(
                        process.left.coefficient,
                        first,
                        internal
                    ) == 0
                    && evaluate(
                        process.right.coefficient,
                        internal,
                        second
                    ) == 0
                ) {
                    ++witnesses;
                }
            }
            const int projected = witnesses > 0;
            const int algebraic =
                evaluate(boundary, first, second) == 0;
            if (projected != algebraic) {
                exact = 0;
            }
            if (projected) {
                pair_mask |= 1U << (unsigned int)(2 * first + second);
                ++boundary_pairs;
                derivations += (unsigned int)witnesses;
            }
        }
    }
    printf(
        "{\"mode\":\"bounded-boolean-extensional-oracle\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"boundary_coefficients\":[%d,%d,%d,%d],"
        "\"boundary_pair_mask\":%u,"
        "\"boundary_pairs\":%u,"
        "\"derivations\":%u,"
        "\"resultant_exact_on_boolean_domain\":%s}\n",
        (unsigned long long)process.source_fnv1a64,
        boundary[0],
        boundary[1],
        boundary[2],
        boundary[3],
        pair_mask,
        boundary_pairs,
        derivations,
        exact ? "true" : "false"
    );
    return exact ? 0 : 1;
}
