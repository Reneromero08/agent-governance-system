/*
 * Independent bounded extensional oracle for open_relation_phase.c.
 *
 * This executable is not linked to the native phase process.  It opens only
 * after the native boundary exists and enumerates the finite Z_N definition
 * to adjudicate the quotient-characteristic result.  It is intentionally
 * bounded.
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define MAX_NAME 31U
#define MAX_PORTS 3U
#define MAX_RELATIONS 2U
#define ORACLE_MODULUS_MAX 4096U

struct port {
    char name[MAX_NAME + 1U];
    uint64_t modulus;
};

struct relation {
    char name[MAX_NAME + 1U];
    char first[MAX_NAME + 1U];
    char second[MAX_NAME + 1U];
    int same;
    int opposite;
};

struct process {
    struct port ports[MAX_PORTS];
    size_t port_count;
    struct relation relations[MAX_RELATIONS];
    size_t relation_count;
    char closed[MAX_NAME + 1U];
    char boundary_first[MAX_NAME + 1U];
    char boundary_second[MAX_NAME + 1U];
    int close_seen;
    int boundary_seen;
};

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static char *trim(char *line) {
    while (isspace((unsigned char)*line)) {
        ++line;
    }
    char *end = line + strlen(line);
    while (end > line && isspace((unsigned char)end[-1])) {
        --end;
    }
    *end = '\0';
    return line;
}

static size_t tokens(
    char *line,
    char **output,
    size_t capacity
) {
    size_t count = 0;
    char *cursor = line;
    while (*cursor != '\0') {
        while (isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        if (count == capacity) {
            fail("too many tokens");
        }
        output[count++] = cursor;
        while (
            *cursor != '\0'
            && !isspace((unsigned char)*cursor)
        ) {
            ++cursor;
        }
        if (*cursor != '\0') {
            *cursor++ = '\0';
        }
    }
    return count;
}

static void name_copy(
    char output[MAX_NAME + 1U],
    const char *input
) {
    const size_t length = strlen(input);
    if (length == 0 || length > MAX_NAME) {
        fail("invalid identifier");
    }
    memcpy(output, input, length + 1U);
}

static uint64_t decimal(const char *text) {
    if (
        text[0] < '0'
        || text[0] > '9'
        || (text[0] == '0' && text[1] != '\0')
    ) {
        fail("invalid decimal");
    }
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (
        end == text
        || *end != '\0'
        || errno == ERANGE
    ) {
        fail("invalid decimal");
    }
    return (uint64_t)value;
}

static int has_port(
    const struct relation *relation,
    const char *name
) {
    return (
        strcmp(relation->first, name) == 0
        || strcmp(relation->second, name) == 0
    );
}

static int connects(
    const struct relation *relation,
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

static struct process read_process(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct process process = {0};
    char *buffer = NULL;
    size_t capacity = 0;
    ssize_t length = 0;
    int header = 0;
    int end = 0;
    while ((length = getline(&buffer, &capacity, stream)) >= 0) {
        if (memchr(buffer, '\0', (size_t)length) != NULL) {
            fail("embedded NUL");
        }
        char *line = trim(buffer);
        if (*line == '\0' || *line == '#') {
            continue;
        }
        char *field[8] = {0};
        const size_t count = tokens(line, field, 8);
        if (!header) {
            if (
                count != 2
                || strcmp(
                    field[0],
                    "CATCAS_OPEN_RELATION_PROCESS"
                ) != 0
                || strcmp(field[1], "1") != 0
            ) {
                fail("invalid header");
            }
            header = 1;
            continue;
        }
        if (strcmp(field[0], "PORT") == 0) {
            if (
                count != 4
                || process.port_count == MAX_PORTS
                || strcmp(field[2], "CYCLIC_PARITY") != 0
            ) {
                fail("invalid port");
            }
            struct port *port = &process.ports[process.port_count++];
            name_copy(port->name, field[1]);
            port->modulus = decimal(field[3]);
            continue;
        }
        if (strcmp(field[0], "RELATION") == 0) {
            if (
                count != 5
                || process.relation_count == MAX_RELATIONS
                || strcmp(field[1], "NONE") == 0
            ) {
                fail("invalid relation");
            }
            struct relation *relation =
                &process.relations[process.relation_count++];
            name_copy(relation->name, field[1]);
            name_copy(relation->first, field[2]);
            name_copy(relation->second, field[3]);
            if (strcmp(field[4], "EMPTY") == 0) {
                relation->same = 0;
                relation->opposite = 0;
            } else if (strcmp(field[4], "SAME") == 0) {
                relation->same = 1;
                relation->opposite = 0;
            } else if (strcmp(field[4], "OPPOSITE") == 0) {
                relation->same = 0;
                relation->opposite = 1;
            } else if (strcmp(field[4], "BOTH") == 0) {
                relation->same = 1;
                relation->opposite = 1;
            } else {
                fail("invalid quotient relation");
            }
            continue;
        }
        if (strcmp(field[0], "CLOSE") == 0) {
            if (count != 2 || process.close_seen) {
                fail("invalid close");
            }
            name_copy(process.closed, field[1]);
            process.close_seen = 1;
            continue;
        }
        if (strcmp(field[0], "BOUNDARY") == 0) {
            if (count != 3 || process.boundary_seen) {
                fail("invalid boundary");
            }
            name_copy(process.boundary_first, field[1]);
            name_copy(process.boundary_second, field[2]);
            process.boundary_seen = 1;
            continue;
        }
        if (strcmp(field[0], "DUPLICATE") == 0) {
            if (count != 2) {
                fail("invalid duplicate");
            }
            continue;
        }
        if (strcmp(field[0], "END") == 0) {
            if (count != 1) {
                fail("invalid end");
            }
            end = 1;
            continue;
        }
        fail("unknown record");
    }
    free(buffer);
    if (fclose(stream) != 0) {
        perror(path);
        exit(2);
    }
    if (
        !header
        || !end
        || process.port_count != MAX_PORTS
        || process.relation_count != MAX_RELATIONS
        || !process.close_seen
        || !process.boundary_seen
    ) {
        fail("incomplete process");
    }
    return process;
}

static int permits(
    const struct relation *relation,
    uint64_t first,
    uint64_t second
) {
    return (
        (
            (((first ^ second) & UINT64_C(1)) == 0)
                && relation->same
        )
        || (
            (((first ^ second) & UINT64_C(1)) == 1)
                && relation->opposite
        )
    );
}

static uint64_t fnv_byte(uint64_t hash, unsigned char value) {
    hash ^= (uint64_t)value;
    hash *= UINT64_C(1099511628211);
    return hash;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s PROCESS.hrel\n", argv[0]);
        return 2;
    }
    const struct process process = read_process(argv[1]);
    const uint64_t modulus = process.ports[0].modulus;
    if (
        modulus < UINT64_C(4)
        || (modulus & UINT64_C(1)) != 0
        || modulus > ORACLE_MODULUS_MAX
        || process.ports[1].modulus != modulus
        || process.ports[2].modulus != modulus
    ) {
        fail("unsupported oracle modulus or type mismatch");
    }
    const struct relation *left = NULL;
    const struct relation *right = NULL;
    for (size_t index = 0; index < MAX_RELATIONS; ++index) {
        if (
            connects(
                &process.relations[index],
                process.boundary_first,
                process.closed
            )
        ) {
            left = &process.relations[index];
        }
        if (
            connects(
                &process.relations[index],
                process.closed,
                process.boundary_second
            )
        ) {
            right = &process.relations[index];
        }
    }
    if (
        left == NULL
        || right == NULL
        || !has_port(left, process.closed)
        || !has_port(right, process.closed)
    ) {
        fail("process is not a two-link open chain");
    }

    const int expected_same = (
        (left->same && right->same)
        || (left->opposite && right->opposite)
    );
    const int expected_opposite = (
        (left->same && right->opposite)
        || (left->opposite && right->same)
    );
    uint64_t boundary_pairs = 0;
    uint64_t derivations = 0;
    uint64_t minimum_witnesses = UINT64_MAX;
    uint64_t maximum_witnesses = 0;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint64_t first = 0; first < modulus; ++first) {
        for (uint64_t second = 0; second < modulus; ++second) {
            uint64_t witnesses = 0;
            for (uint64_t internal = 0; internal < modulus; ++internal) {
                if (
                    permits(left, first, internal)
                    && permits(right, internal, second)
                ) {
                    ++witnesses;
                }
            }
            const int exists = witnesses > 0;
            const int expected = (
                (
                    (((first ^ second) & UINT64_C(1)) == 0)
                    && expected_same
                )
                || (
                    (((first ^ second) & UINT64_C(1)) == 1)
                    && expected_opposite
                )
            );
            if (exists != expected) {
                fail("extensional composition contradicts quotient result");
            }
            hash = fnv_byte(hash, (unsigned char)exists);
            if (exists) {
                ++boundary_pairs;
                derivations += witnesses;
                if (witnesses < minimum_witnesses) {
                    minimum_witnesses = witnesses;
                }
                if (witnesses > maximum_witnesses) {
                    maximum_witnesses = witnesses;
                }
            }
        }
    }
    if (!expected_same && !expected_opposite) {
        minimum_witnesses = 0;
    }
    printf(
        "{\"mode\":\"bounded-extensional-oracle\","
        "\"modulus\":%llu,"
        "\"boundary_same_symbol\":%d,"
        "\"boundary_opposite_symbol\":%d,"
        "\"boundary_pairs\":%llu,"
        "\"derivations\":%llu,"
        "\"minimum_witnesses_per_valid_pair\":%llu,"
        "\"maximum_witnesses_per_valid_pair\":%llu,"
        "\"ordered_boundary_fnv1a64\":\"%016llx\"}\n",
        (unsigned long long)modulus,
        expected_same,
        expected_opposite,
        (unsigned long long)boundary_pairs,
        (unsigned long long)derivations,
        (unsigned long long)minimum_witnesses,
        (unsigned long long)maximum_witnesses,
        (unsigned long long)hash
    );
    return 0;
}
