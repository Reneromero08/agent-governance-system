/*
 * Mutable CAT_CAS frontier: typed open parity-coset relations.
 *
 * This carrier is non-enumerative over Z_N but quotient-extensional.  A
 * CYCLIC_PARITY(N) link denotes all (a,b) in Z_N^2 whose parity difference
 * belongs to a Boolean-lattice subset of the fixed Z_2 quotient: EMPTY,
 * SAME, OPPOSITE, or BOTH.  The local relation is the complete two-slot
 * characteristic vector of that quotient.  Every nonempty link is
 * many-to-many for even N > 2.  Two links compose through one fixed
 * idempotent Boolean-convolution circuit over Z_2 while closing their shared
 * typed port.
 *
 * Both laws execute on relative complex phases.  The carrier never contains
 * individual port values, tuples, witnesses, or paths.  A duplicate-link
 * presentation uses an idempotent phase OR before composition.  Forward
 * factors are recomputed during inverse traversal; no history is retained.
 */

#define _GNU_SOURCE
#include <complex.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define MAX_NAME 31U
#define MAX_LINE 4095U
#define PORT_COUNT 3U
#define RELATION_COUNT 2U
#define CARRIER_CELLS 8U
#define RESTORATION_MAX 2.0e-12
#define WRONG_RESTORATION_MIN 1.0e-3
#define INTEGRITY_MAX 2.0e-12

enum cell_index {
    LEFT_SAME = 0,
    LEFT_OPPOSITE = 1,
    RIGHT_SAME = 2,
    RIGHT_OPPOSITE = 3,
    UNION_SAME = 4,
    UNION_OPPOSITE = 5,
    BOUNDARY_SAME = 6,
    BOUNDARY_OPPOSITE = 7
};

struct port {
    char name[MAX_NAME + 1U];
    uint64_t modulus;
};

struct relation_definition {
    char name[MAX_NAME + 1U];
    char first[MAX_NAME + 1U];
    char second[MAX_NAME + 1U];
    int same;
    int opposite;
};

struct parsed_process {
    struct port ports[PORT_COUNT];
    size_t port_count;
    struct relation_definition relations[RELATION_COUNT];
    size_t relation_count;
    char closed_port[MAX_NAME + 1U];
    char boundary_first[MAX_NAME + 1U];
    char boundary_second[MAX_NAME + 1U];
    char duplicate_relation[MAX_NAME + 1U];
    int close_seen;
    int boundary_seen;
    int duplicate_seen;
    uint64_t source_fnv1a64;
};

struct process {
    uint64_t modulus;
    struct relation_definition left;
    struct relation_definition right;
    int duplicate_left;
    uint64_t source_fnv1a64;
};

struct carrier {
    double complex baseline[CARRIER_CELLS];
    double complex working[CARRIER_CELLS];
};

struct boundary_record {
    double complex same_phase;
    double complex opposite_phase;
    int same_symbol;
    int opposite_symbol;
};

struct execution {
    struct boundary_record boundary;
    double displacement_l2;
    double restoration_max_abs;
    double integrity_error;
};

static void die_line(
    const char *message,
    size_t line_number
) {
    fprintf(stderr, "%s on line %zu\n", message, line_number);
    exit(2);
}

static uint64_t fnv1a_update(
    uint64_t hash,
    const unsigned char *bytes,
    size_t length
) {
    for (size_t index = 0; index < length; ++index) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static char *trim_line(char *line) {
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

static size_t split_tokens(
    char *line,
    char **tokens,
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
            return capacity + 1U;
        }
        tokens[count++] = cursor;
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

static int valid_name(const char *name) {
    const size_t length = strlen(name);
    if (
        length == 0
        || length > MAX_NAME
        || !(
            isalpha((unsigned char)name[0])
            || name[0] == '_'
        )
    ) {
        return 0;
    }
    for (size_t index = 1; index < length; ++index) {
        if (
            !isalnum((unsigned char)name[index])
            && name[index] != '_'
        ) {
            return 0;
        }
    }
    return 1;
}

static void copy_name(
    char target[MAX_NAME + 1U],
    const char *source,
    size_t line_number
) {
    if (!valid_name(source)) {
        die_line("invalid identifier", line_number);
    }
    const size_t length = strlen(source);
    memcpy(target, source, length + 1U);
}

static uint64_t parse_modulus(
    const char *text,
    size_t line_number
) {
    if (
        text[0] < '0'
        || text[0] > '9'
        || (text[0] == '0' && text[1] != '\0')
    ) {
        die_line("invalid modulus", line_number);
    }
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (
        end == text
        || *end != '\0'
        || errno == ERANGE
        || value < 4U
        || (value & 1U) != 0U
    ) {
        die_line("modulus must be an even integer of at least four", line_number);
    }
    return (uint64_t)value;
}

static int find_port(
    const struct parsed_process *parsed,
    const char *name
) {
    for (size_t index = 0; index < parsed->port_count; ++index) {
        if (strcmp(parsed->ports[index].name, name) == 0) {
            return (int)index;
        }
    }
    return -1;
}

static int relation_has_port(
    const struct relation_definition *relation,
    const char *port
) {
    return (
        strcmp(relation->first, port) == 0
        || strcmp(relation->second, port) == 0
    );
}

static int relation_connects(
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

static struct process validate_process(
    const struct parsed_process *parsed
) {
    if (
        parsed->port_count != PORT_COUNT
        || parsed->relation_count != RELATION_COUNT
        || !parsed->close_seen
        || !parsed->boundary_seen
        || !parsed->duplicate_seen
    ) {
        fprintf(stderr, "open process is incomplete\n");
        exit(2);
    }
    const int closed = find_port(parsed, parsed->closed_port);
    const int boundary_first =
        find_port(parsed, parsed->boundary_first);
    const int boundary_second =
        find_port(parsed, parsed->boundary_second);
    if (
        closed < 0
        || boundary_first < 0
        || boundary_second < 0
        || closed == boundary_first
        || closed == boundary_second
        || boundary_first == boundary_second
    ) {
        fprintf(stderr, "invalid closed or boundary port geometry\n");
        exit(2);
    }
    const uint64_t modulus = parsed->ports[0].modulus;
    for (size_t index = 1; index < PORT_COUNT; ++index) {
        if (parsed->ports[index].modulus != modulus) {
            fprintf(stderr, "connected port types do not match\n");
            exit(2);
        }
    }
    int left_index = -1;
    int right_index = -1;
    for (size_t index = 0; index < RELATION_COUNT; ++index) {
        const struct relation_definition *relation =
            &parsed->relations[index];
        if (
            find_port(parsed, relation->first) < 0
            || find_port(parsed, relation->second) < 0
            || strcmp(relation->first, relation->second) == 0
            || !relation_has_port(relation, parsed->closed_port)
        ) {
            fprintf(stderr, "relation does not form the declared open chain\n");
            exit(2);
        }
        if (
            relation_connects(
                relation,
                parsed->boundary_first,
                parsed->closed_port
            )
        ) {
            if (left_index >= 0) {
                fprintf(stderr, "multiple left relations\n");
                exit(2);
            }
            left_index = (int)index;
        } else if (
            relation_connects(
                relation,
                parsed->closed_port,
                parsed->boundary_second
            )
        ) {
            if (right_index >= 0) {
                fprintf(stderr, "multiple right relations\n");
                exit(2);
            }
            right_index = (int)index;
        } else {
            fprintf(stderr, "relation endpoint is outside the declared boundary\n");
            exit(2);
        }
    }
    if (left_index < 0 || right_index < 0) {
        fprintf(stderr, "open chain is not composable\n");
        exit(2);
    }
    int duplicate_left = 0;
    if (strcmp(parsed->duplicate_relation, "NONE") != 0) {
        if (
            strcmp(
                parsed->duplicate_relation,
                parsed->relations[left_index].name
            ) == 0
        ) {
            duplicate_left = 1;
        } else {
            fprintf(stderr, "only the left link may be duplicated in v1\n");
            exit(2);
        }
    }
    return (struct process){
        .modulus = modulus,
        .left = parsed->relations[left_index],
        .right = parsed->relations[right_index],
        .duplicate_left = duplicate_left,
        .source_fnv1a64 = parsed->source_fnv1a64
    };
}

static struct process read_process(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct parsed_process parsed = {
        .port_count = 0,
        .relation_count = 0,
        .close_seen = 0,
        .boundary_seen = 0,
        .duplicate_seen = 0,
        .source_fnv1a64 = UINT64_C(14695981039346656037)
    };
    char *buffer = NULL;
    size_t capacity = 0;
    ssize_t read_length = 0;
    size_t line_number = 0;
    int header_seen = 0;
    int end_seen = 0;
    while ((read_length = getline(&buffer, &capacity, stream)) >= 0) {
        ++line_number;
        const size_t raw_length = (size_t)read_length;
        if (memchr(buffer, '\0', raw_length) != NULL) {
            die_line("embedded NUL", line_number);
        }
        if (raw_length > MAX_LINE) {
            die_line("line exceeds 4095 bytes", line_number);
        }
        parsed.source_fnv1a64 = fnv1a_update(
            parsed.source_fnv1a64,
            (const unsigned char *)buffer,
            raw_length
        );
        char *line = trim_line(buffer);
        if (*line == '\0' || *line == '#') {
            continue;
        }
        if (end_seen) {
            die_line("content follows END", line_number);
        }
        char *tokens[8] = {0};
        const size_t token_count = split_tokens(line, tokens, 8);
        if (token_count > 8) {
            die_line("too many tokens", line_number);
        }
        if (!header_seen) {
            if (
                token_count != 2
                || strcmp(
                    tokens[0],
                    "CATCAS_OPEN_RELATION_PROCESS"
                ) != 0
                || strcmp(tokens[1], "1") != 0
            ) {
                die_line("invalid open-process header", line_number);
            }
            header_seen = 1;
            continue;
        }
        if (strcmp(tokens[0], "PORT") == 0) {
            if (
                token_count != 4
                || parsed.port_count == PORT_COUNT
                || strcmp(tokens[2], "CYCLIC_PARITY") != 0
            ) {
                die_line("invalid PORT", line_number);
            }
            struct port *port = &parsed.ports[parsed.port_count];
            copy_name(port->name, tokens[1], line_number);
            if (find_port(&parsed, port->name) >= 0) {
                die_line("duplicate PORT", line_number);
            }
            port->modulus = parse_modulus(tokens[3], line_number);
            ++parsed.port_count;
            continue;
        }
        if (strcmp(tokens[0], "RELATION") == 0) {
            if (
                token_count != 5
                || parsed.relation_count == RELATION_COUNT
                || strcmp(tokens[1], "NONE") == 0
            ) {
                die_line("invalid RELATION", line_number);
            }
            struct relation_definition *relation =
                &parsed.relations[parsed.relation_count];
            copy_name(relation->name, tokens[1], line_number);
            copy_name(relation->first, tokens[2], line_number);
            copy_name(relation->second, tokens[3], line_number);
            if (strcmp(tokens[4], "EMPTY") == 0) {
                relation->same = 0;
                relation->opposite = 0;
            } else if (strcmp(tokens[4], "SAME") == 0) {
                relation->same = 1;
                relation->opposite = 0;
            } else if (strcmp(tokens[4], "OPPOSITE") == 0) {
                relation->same = 0;
                relation->opposite = 1;
            } else if (strcmp(tokens[4], "BOTH") == 0) {
                relation->same = 1;
                relation->opposite = 1;
            } else {
                die_line("invalid quotient relation", line_number);
            }
            for (size_t index = 0; index < parsed.relation_count; ++index) {
                if (
                    strcmp(
                        relation->name,
                        parsed.relations[index].name
                    ) == 0
                ) {
                    die_line("duplicate relation name", line_number);
                }
            }
            ++parsed.relation_count;
            continue;
        }
        if (strcmp(tokens[0], "CLOSE") == 0) {
            if (token_count != 2 || parsed.close_seen) {
                die_line("invalid CLOSE", line_number);
            }
            copy_name(parsed.closed_port, tokens[1], line_number);
            parsed.close_seen = 1;
            continue;
        }
        if (strcmp(tokens[0], "BOUNDARY") == 0) {
            if (token_count != 3 || parsed.boundary_seen) {
                die_line("invalid BOUNDARY", line_number);
            }
            copy_name(parsed.boundary_first, tokens[1], line_number);
            copy_name(parsed.boundary_second, tokens[2], line_number);
            parsed.boundary_seen = 1;
            continue;
        }
        if (strcmp(tokens[0], "DUPLICATE") == 0) {
            if (token_count != 2 || parsed.duplicate_seen) {
                die_line("invalid DUPLICATE", line_number);
            }
            if (strcmp(tokens[1], "NONE") == 0) {
                memcpy(parsed.duplicate_relation, "NONE", 5U);
            } else {
                copy_name(
                    parsed.duplicate_relation,
                    tokens[1],
                    line_number
                );
            }
            parsed.duplicate_seen = 1;
            continue;
        }
        if (strcmp(tokens[0], "END") == 0) {
            if (token_count != 1) {
                die_line("invalid END", line_number);
            }
            end_seen = 1;
            continue;
        }
        die_line("unknown open-process record", line_number);
    }
    if (ferror(stream)) {
        perror(path);
        exit(2);
    }
    free(buffer);
    if (fclose(stream) != 0) {
        perror(path);
        exit(2);
    }
    if (!header_seen || !end_seen) {
        fprintf(stderr, "open process is incomplete\n");
        exit(2);
    }
    return validate_process(&parsed);
}

static double complex unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        fprintf(stderr, "nonfinite phase state\n");
        exit(2);
    }
    return value / magnitude;
}

static double complex root3(int amount) {
    int normalized = amount % 3;
    if (normalized < 0) {
        normalized += 3;
    }
    if (normalized == 0) {
        return 1.0 + 0.0 * I;
    }
    if (normalized == 1) {
        return -0.5 + 0.86602540378443864676 * I;
    }
    return -0.5 - 0.86602540378443864676 * I;
}

static double complex relation(
    const struct carrier *carrier,
    size_t cell
) {
    return carrier->working[cell] * conj(carrier->baseline[cell]);
}

static void multiply_relation(
    struct carrier *carrier,
    size_t cell,
    double complex factor
) {
    carrier->working[cell] = unit(
        relation(carrier, cell) * factor
    ) * carrier->baseline[cell];
}

static double complex product_factor(
    double complex left,
    double complex right
) {
    const double complex left_squared = conj(left);
    const double complex right_squared = conj(right);
    const double complex product = left * right;
    return unit(
        (
            1.0
            + left
            + left_squared
            + right
            + right_squared
            + root3(2) * (product + conj(product))
            + root3(1) * (
                left * right_squared
                + left_squared * right
            )
        ) / 3.0
    );
}

static double complex boolean_or_factor(
    double complex left,
    double complex right
) {
    return unit(
        left
        * right
        * conj(product_factor(left, right))
    );
}

static struct carrier make_carrier(int identity) {
    struct carrier carrier;
    for (size_t index = 0; index < CARRIER_CELLS; ++index) {
        const double angle =
            0.119
            + 0.071 * (double)index
            + 0.017 * sin(
                0.23 * (double)index + 0.031 * (double)identity
            );
        carrier.baseline[index] = cexp(I * angle);
        carrier.working[index] = carrier.baseline[index];
    }
    return carrier;
}

static void apply_encoding(
    struct carrier *carrier,
    size_t same_cell,
    size_t opposite_cell,
    const struct relation_definition *definition,
    int inverse
) {
    double complex same_factor = root3(definition->same);
    double complex opposite_factor = root3(definition->opposite);
    if (inverse) {
        same_factor = conj(same_factor);
        opposite_factor = conj(opposite_factor);
    }
    multiply_relation(carrier, same_cell, same_factor);
    multiply_relation(carrier, opposite_cell, opposite_factor);
}

static void apply_duplicate_union(
    struct carrier *carrier,
    int inverse
) {
    double complex same_factor = boolean_or_factor(
        relation(carrier, LEFT_SAME),
        relation(carrier, LEFT_SAME)
    );
    double complex opposite_factor = boolean_or_factor(
        relation(carrier, LEFT_OPPOSITE),
        relation(carrier, LEFT_OPPOSITE)
    );
    if (inverse) {
        same_factor = conj(same_factor);
        opposite_factor = conj(opposite_factor);
    }
    multiply_relation(carrier, UNION_SAME, same_factor);
    multiply_relation(carrier, UNION_OPPOSITE, opposite_factor);
}

static void apply_composition(
    struct carrier *carrier,
    int duplicate_left,
    int inverse
) {
    const size_t left_same =
        duplicate_left ? UNION_SAME : LEFT_SAME;
    const size_t left_opposite =
        duplicate_left ? UNION_OPPOSITE : LEFT_OPPOSITE;
    const double complex same_same = product_factor(
        relation(carrier, left_same),
        relation(carrier, RIGHT_SAME)
    );
    const double complex opposite_opposite = product_factor(
        relation(carrier, left_opposite),
        relation(carrier, RIGHT_OPPOSITE)
    );
    const double complex same_opposite = product_factor(
        relation(carrier, left_same),
        relation(carrier, RIGHT_OPPOSITE)
    );
    const double complex opposite_same = product_factor(
        relation(carrier, left_opposite),
        relation(carrier, RIGHT_SAME)
    );
    double complex same_factor = boolean_or_factor(
        same_same,
        opposite_opposite
    );
    double complex opposite_factor = boolean_or_factor(
        same_opposite,
        opposite_same
    );
    if (inverse) {
        same_factor = conj(same_factor);
        opposite_factor = conj(opposite_factor);
    }
    multiply_relation(carrier, BOUNDARY_SAME, same_factor);
    multiply_relation(carrier, BOUNDARY_OPPOSITE, opposite_factor);
}

static int decode_root3(double complex value) {
    int best = 0;
    double best_distance = INFINITY;
    for (int symbol = 0; symbol < 3; ++symbol) {
        const double distance = cabs(value - root3(symbol));
        if (distance < best_distance) {
            best = symbol;
            best_distance = distance;
        }
    }
    return best;
}

static struct boundary_record latch_boundary(
    const struct carrier *carrier
) {
    const double complex same =
        relation(carrier, BOUNDARY_SAME);
    const double complex opposite =
        relation(carrier, BOUNDARY_OPPOSITE);
    return (struct boundary_record){
        .same_phase = same,
        .opposite_phase = opposite,
        .same_symbol = decode_root3(same),
        .opposite_symbol = decode_root3(opposite)
    };
}

static double carrier_displacement(
    const struct carrier *carrier,
    const struct carrier *borrowed
) {
    double sum = 0.0;
    for (size_t index = 0; index < CARRIER_CELLS; ++index) {
        const double difference = cabs(
            carrier->working[index] - borrowed->working[index]
        );
        sum += difference * difference;
    }
    return sqrt(sum);
}

static double restoration_error(
    const struct carrier *carrier,
    const struct carrier *borrowed
) {
    double maximum = 0.0;
    for (size_t index = 0; index < CARRIER_CELLS; ++index) {
        const double difference = cabs(
            carrier->working[index] - borrowed->working[index]
        );
        if (difference > maximum) {
            maximum = difference;
        }
    }
    return maximum;
}

static double integrity_error(const struct carrier *carrier) {
    double maximum = 0.0;
    for (size_t index = 0; index < CARRIER_CELLS; ++index) {
        const double error =
            fabs(cabs(carrier->working[index]) - 1.0);
        if (error > maximum) {
            maximum = error;
        }
    }
    return maximum;
}

static struct execution execute(
    struct carrier *carrier,
    const struct process *process,
    int inverse_mode
) {
    const struct carrier borrowed = *carrier;
    apply_encoding(
        carrier,
        LEFT_SAME,
        LEFT_OPPOSITE,
        &process->left,
        0
    );
    apply_encoding(
        carrier,
        RIGHT_SAME,
        RIGHT_OPPOSITE,
        &process->right,
        0
    );
    if (process->duplicate_left) {
        apply_duplicate_union(carrier, 0);
    }
    apply_composition(carrier, process->duplicate_left, 0);
    const struct boundary_record boundary = latch_boundary(carrier);
    const double displacement =
        carrier_displacement(carrier, &borrowed);
    const double integrity = integrity_error(carrier);

    if (inverse_mode == 0 || inverse_mode == 1) {
        apply_composition(carrier, process->duplicate_left, 1);
        if (process->duplicate_left) {
            apply_duplicate_union(carrier, 1);
        }
        apply_encoding(
            carrier,
            RIGHT_SAME,
            RIGHT_OPPOSITE,
            &process->right,
            1
        );
        apply_encoding(
            carrier,
            LEFT_SAME,
            LEFT_OPPOSITE,
            &process->left,
            1
        );
        if (inverse_mode == 1) {
            multiply_relation(carrier, BOUNDARY_SAME, root3(1));
        }
    } else if (inverse_mode == 2) {
        apply_encoding(
            carrier,
            LEFT_SAME,
            LEFT_OPPOSITE,
            &process->left,
            1
        );
        apply_encoding(
            carrier,
            RIGHT_SAME,
            RIGHT_OPPOSITE,
            &process->right,
            1
        );
        apply_composition(carrier, process->duplicate_left, 1);
        if (process->duplicate_left) {
            apply_duplicate_union(carrier, 1);
        }
    } else if (inverse_mode != 3) {
        fprintf(stderr, "unknown inverse mode\n");
        exit(2);
    }

    return (struct execution){
        .boundary = boundary,
        .displacement_l2 = displacement,
        .restoration_max_abs =
            restoration_error(carrier, &borrowed),
        .integrity_error = integrity
    };
}

static int same_boundary(
    const struct execution *left,
    const struct execution *right
) {
    return (
        left->boundary.same_symbol
            == right->boundary.same_symbol
        && left->boundary.opposite_symbol
            == right->boundary.opposite_symbol
        && cabs(
            left->boundary.same_phase
            - right->boundary.same_phase
        ) <= RESTORATION_MAX
        && cabs(
            left->boundary.opposite_phase
            - right->boundary.opposite_phase
        ) <= RESTORATION_MAX
    );
}

static void print_execution(
    const struct execution *execution,
    const struct process *process,
    const char *mode
) {
    printf(
        "{\"mode\":\"%s\","
        "\"process_fnv1a64\":\"%016llx\","
        "\"port_type\":\"CYCLIC_PARITY\","
        "\"modulus\":%llu,"
        "\"carrier_cells\":%u,"
        "\"witness_slots\":0,"
        "\"retained_inverse_factors\":0,"
        "\"duplicate_left\":%s,"
        "\"boundary_same_symbol\":%d,"
        "\"boundary_opposite_symbol\":%d,"
        "\"boundary_same_phase\":[%.12g,%.12g],"
        "\"boundary_opposite_phase\":[%.12g,%.12g],"
        "\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,"
        "\"carrier_integrity_error\":%.12g}\n",
        mode,
        (unsigned long long)process->source_fnv1a64,
        (unsigned long long)process->modulus,
        CARRIER_CELLS,
        process->duplicate_left ? "true" : "false",
        execution->boundary.same_symbol,
        execution->boundary.opposite_symbol,
        creal(execution->boundary.same_phase),
        cimag(execution->boundary.same_phase),
        creal(execution->boundary.opposite_phase),
        cimag(execution->boundary.opposite_phase),
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->integrity_error
    );
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(
            stderr,
            "usage: %s PROCESS.hrel [REUSE_PROCESS.hrel]\n",
            argv[0]
        );
        return 2;
    }
    const struct process process = read_process(argv[1]);
    struct carrier carrier = make_carrier(211);
    const struct execution first = execute(&carrier, &process, 0);
    print_execution(&first, &process, "open-relation");
    const struct process reuse_process =
        argc == 3 ? read_process(argv[2]) : process;
    if (reuse_process.modulus != process.modulus) {
        fprintf(stderr, "reuse process has a different carrier type\n");
        return 2;
    }
    const struct execution reuse =
        execute(&carrier, &reuse_process, 0);
    print_execution(
        &reuse,
        &reuse_process,
        argc == 3
            ? "actual-restored-cross-process-reuse"
            : "actual-restored-reuse"
    );

    struct carrier wrong_carrier = make_carrier(211);
    const struct execution wrong =
        execute(&wrong_carrier, &process, 1);
    print_execution(&wrong, &process, "wrong-inverse");
    struct carrier reordered_carrier = make_carrier(211);
    const struct execution reordered =
        execute(&reordered_carrier, &process, 2);
    print_execution(&reordered, &process, "reordered-inverse");
    struct carrier omitted_carrier = make_carrier(211);
    const struct execution omitted =
        execute(&omitted_carrier, &process, 3);
    print_execution(&omitted, &process, "omitted-inverse");

    struct carrier corrupted = make_carrier(211);
    corrupted.working[LEFT_SAME] *= 0.5;
    const double corruption_error = integrity_error(&corrupted);
    printf(
        "{\"mode\":\"carrier-failure-control\","
        "\"integrity_error\":%.12g,"
        "\"detected\":%s}\n",
        corruption_error,
        corruption_error > INTEGRITY_MAX ? "true" : "false"
    );

    const int inverse_controls_applicable =
        first.displacement_l2 > WRONG_RESTORATION_MIN;
    const int reordered_control_applicable = (
        first.boundary.same_symbol != 0
        || first.boundary.opposite_symbol != 0
    );
    printf(
        "{\"mode\":\"inverse-control-applicability\","
        "\"wrong\":%s,\"reordered\":%s,\"omitted\":%s}\n",
        inverse_controls_applicable ? "true" : "false",
        reordered_control_applicable ? "true" : "false",
        inverse_controls_applicable ? "true" : "false"
    );
    const int success = (
        first.boundary.same_symbol >= 0
        && first.boundary.same_symbol <= 1
        && first.boundary.opposite_symbol >= 0
        && first.boundary.opposite_symbol <= 1
        && first.restoration_max_abs <= RESTORATION_MAX
        && reuse.restoration_max_abs <= RESTORATION_MAX
        && (
            argc == 3
            || same_boundary(&first, &reuse)
        )
        && first.integrity_error <= INTEGRITY_MAX
        && reuse.integrity_error <= INTEGRITY_MAX
        && (
            !inverse_controls_applicable
            || wrong.restoration_max_abs > WRONG_RESTORATION_MIN
        )
        && (
            !reordered_control_applicable
            || reordered.restoration_max_abs
                > WRONG_RESTORATION_MIN
        )
        && (
            !inverse_controls_applicable
            || omitted.restoration_max_abs
                > WRONG_RESTORATION_MIN
        )
        && corruption_error > INTEGRITY_MAX
    );
    return success ? 0 : 1;
}
