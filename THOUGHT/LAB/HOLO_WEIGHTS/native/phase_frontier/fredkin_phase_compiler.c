/*
 * Mutable CAT_CAS compiler: Fredkin circuit -> public phase program.
 *
 * This compiler emits topology only. It never executes or adjudicates the
 * circuit. Program enables and data become phase-resident registers consumed
 * by streaming_phase_vm.c. The emitted PCSWAP schedule visits each public gate
 * exactly once per cycle; it does not scan a one-hot program counter.
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

struct gate {
    size_t control;
    size_t left;
    size_t right;
    int enabled;
};

struct circuit {
    size_t wires;
    int *input;
    struct gate *gates;
    size_t gate_count;
    size_t gate_capacity;
    uint64_t cycles;
    uint64_t fnv1a64;
};

static uint64_t parse_positive_decimal(
    const char *text, const char *name
) {
    char *end = NULL;
    if (text[0] < '0' || text[0] > '9') {
        fprintf(stderr, "%s must be a positive decimal integer\n", name);
        exit(2);
    }
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (
        end == text
        || *end != '\0'
        || errno == ERANGE
        || value == 0
    ) {
        fprintf(stderr, "%s must be a positive decimal integer\n", name);
        exit(2);
    }
    return (uint64_t)value;
}

static uint64_t parse_positive_record(
    const char *line,
    const char *keyword,
    size_t line_number
) {
    const size_t keyword_length = strlen(keyword);
    if (
        strncmp(line, keyword, keyword_length) != 0
        || !isspace((unsigned char)line[keyword_length])
    ) {
        fprintf(
            stderr,
            "invalid %s on line %zu\n",
            keyword,
            line_number
        );
        exit(2);
    }
    const char *value = line + keyword_length;
    while (isspace((unsigned char)*value)) {
        ++value;
    }
    return parse_positive_decimal(value, keyword);
}

static const char *record_values(
    const char *line,
    const char *keyword,
    size_t line_number
) {
    const size_t keyword_length = strlen(keyword);
    if (
        strncmp(line, keyword, keyword_length) != 0
        || !isspace((unsigned char)line[keyword_length])
    ) {
        fprintf(
            stderr,
            "invalid %s on line %zu\n",
            keyword,
            line_number
        );
        exit(2);
    }
    const char *cursor = line + keyword_length;
    while (isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    return cursor;
}

static uint64_t parse_unsigned_token(
    const char **cursor,
    const char *keyword,
    size_t line_number
) {
    while (isspace((unsigned char)**cursor)) {
        ++*cursor;
    }
    if (**cursor < '0' || **cursor > '9') {
        fprintf(
            stderr,
            "invalid %s on line %zu\n",
            keyword,
            line_number
        );
        exit(2);
    }
    char *end = NULL;
    errno = 0;
    const unsigned long long value = strtoull(*cursor, &end, 10);
    if (
        end == *cursor
        || errno == ERANGE
        || (*end != '\0' && !isspace((unsigned char)*end))
    ) {
        fprintf(
            stderr,
            "invalid %s on line %zu\n",
            keyword,
            line_number
        );
        exit(2);
    }
    *cursor = end;
    return (uint64_t)value;
}

static void require_record_end(
    const char *cursor,
    const char *keyword,
    size_t line_number
) {
    while (isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    if (*cursor != '\0') {
        fprintf(
            stderr,
            "invalid %s on line %zu\n",
            keyword,
            line_number
        );
        exit(2);
    }
}

static void parse_unsigned_record(
    const char *line,
    const char *keyword,
    uint64_t *values,
    size_t value_count,
    size_t line_number
) {
    const char *cursor = record_values(line, keyword, line_number);
    for (size_t index = 0; index < value_count; ++index) {
        values[index] = parse_unsigned_token(
            &cursor,
            keyword,
            line_number
        );
    }
    require_record_end(cursor, keyword, line_number);
}

static inline uint64_t fnv1a_update(
    uint64_t hash, const unsigned char *bytes, size_t length
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

static void require_wire(
    size_t wire, size_t wires, size_t line_number
) {
    if (wire >= wires) {
        fprintf(
            stderr,
            "wire %zu is out of range on line %zu\n",
            wire,
            line_number
        );
        exit(2);
    }
}

static void append_gate(struct circuit *circuit, struct gate gate) {
    if (circuit->gate_count == circuit->gate_capacity) {
        const size_t next = circuit->gate_capacity == 0
            ? 32U
            : circuit->gate_capacity * 2U;
        if (
            next < circuit->gate_capacity
            || next > SIZE_MAX / sizeof(struct gate)
        ) {
            fprintf(stderr, "circuit is too large\n");
            exit(2);
        }
        struct gate *gates = realloc(
            circuit->gates, next * sizeof(struct gate)
        );
        if (gates == NULL) {
            fprintf(stderr, "gate allocation failed\n");
            exit(2);
        }
        circuit->gates = gates;
        circuit->gate_capacity = next;
    }
    circuit->gates[circuit->gate_count++] = gate;
}

static struct circuit read_circuit(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct circuit circuit = {
        .wires = 0,
        .input = NULL,
        .gates = NULL,
        .gate_count = 0,
        .gate_capacity = 0,
        .cycles = 1,
        .fnv1a64 = UINT64_C(14695981039346656037)
    };
    char *buffer = NULL;
    size_t buffer_capacity = 0;
    ssize_t read_length = 0;
    size_t line_number = 0;
    int header_seen = 0;
    int wires_seen = 0;
    int cycles_seen = 0;
    int gate_seen = 0;
    int end_seen = 0;
    while (
        (read_length = getline(&buffer, &buffer_capacity, stream)) >= 0
    ) {
        ++line_number;
        const size_t raw_length = (size_t)read_length;
        if (memchr(buffer, '\0', raw_length) != NULL) {
            fprintf(stderr, "embedded NUL on line %zu\n", line_number);
            exit(2);
        }
        circuit.fnv1a64 = fnv1a_update(
            circuit.fnv1a64,
            (const unsigned char *)buffer,
            raw_length
        );
        if (
            raw_length > 4095U
        ) {
            fprintf(stderr, "line %zu exceeds 4095 bytes\n", line_number);
            exit(2);
        }
        char *line = trim_line(buffer);
        if (*line == '\0' || *line == '#') {
            continue;
        }
        if (end_seen) {
            fprintf(stderr, "content follows END on line %zu\n", line_number);
            exit(2);
        }
        if (!header_seen) {
            uint64_t fields[1] = {0};
            parse_unsigned_record(
                line,
                "CATCAS_FREDKIN_CIRCUIT",
                fields,
                1,
                line_number
            );
            if (fields[0] != UINT64_C(1)) {
                fprintf(stderr, "invalid circuit header\n");
                exit(2);
            }
            header_seen = 1;
            continue;
        }
        if (strcmp(line, "END") == 0) {
            if (!wires_seen || !gate_seen) {
                fprintf(stderr, "END precedes a complete circuit\n");
                exit(2);
            }
            end_seen = 1;
            continue;
        }
        if (strncmp(line, "WIRES", 5U) == 0) {
            uint64_t fields[1] = {0};
            parse_unsigned_record(
                line,
                "WIRES",
                fields,
                1,
                line_number
            );
            if (
                wires_seen
                || gate_seen
                || fields[0] < UINT64_C(3)
                || fields[0] > (uint64_t)(SIZE_MAX / sizeof(int))
            ) {
                fprintf(stderr, "invalid WIRES on line %zu\n", line_number);
                exit(2);
            }
            const size_t wires = (size_t)fields[0];
            circuit.wires = wires;
            circuit.input = calloc(wires, sizeof(int));
            if (circuit.input == NULL) {
                fprintf(stderr, "wire allocation failed\n");
                exit(2);
            }
            wires_seen = 1;
            continue;
        }
        if (strncmp(line, "SET", 3U) == 0) {
            uint64_t fields[2] = {0, 0};
            parse_unsigned_record(
                line,
                "SET",
                fields,
                2,
                line_number
            );
            if (
                !wires_seen
                || gate_seen
                || fields[0] > (uint64_t)SIZE_MAX
                || fields[1] > UINT64_C(2)
            ) {
                fprintf(stderr, "invalid SET on line %zu\n", line_number);
                exit(2);
            }
            const size_t wire = (size_t)fields[0];
            const int symbol = (int)fields[1];
            require_wire(wire, circuit.wires, line_number);
            circuit.input[wire] = symbol;
            continue;
        }
        if (strncmp(line, "CYCLES", 6U) == 0) {
            if (
                !wires_seen
                || cycles_seen
                || gate_seen
            ) {
                fprintf(stderr, "invalid CYCLES on line %zu\n", line_number);
                exit(2);
            }
            circuit.cycles = parse_positive_record(
                line,
                "CYCLES",
                line_number
            );
            cycles_seen = 1;
            continue;
        }
        if (strncmp(line, "FREDKIN", 7U) == 0) {
            uint64_t fields[4] = {0, 0, 0, 0};
            parse_unsigned_record(
                line,
                "FREDKIN",
                fields,
                4,
                line_number
            );
            struct gate gate = {
                .control = (size_t)fields[0],
                .left = (size_t)fields[1],
                .right = (size_t)fields[2],
                .enabled = (int)fields[3]
            };
            if (
                !wires_seen
                || fields[0] > (uint64_t)SIZE_MAX
                || fields[1] > (uint64_t)SIZE_MAX
                || fields[2] > (uint64_t)SIZE_MAX
                || fields[3] > UINT64_C(1)
            ) {
                fprintf(stderr, "invalid FREDKIN on line %zu\n", line_number);
                exit(2);
            }
            require_wire(gate.control, circuit.wires, line_number);
            require_wire(gate.left, circuit.wires, line_number);
            require_wire(gate.right, circuit.wires, line_number);
            if (
                gate.control == gate.left
                || gate.control == gate.right
                || gate.left == gate.right
            ) {
                fprintf(stderr, "FREDKIN wires alias on line %zu\n", line_number);
                exit(2);
            }
            append_gate(&circuit, gate);
            gate_seen = 1;
            continue;
        }
        fprintf(stderr, "unknown circuit record on line %zu\n", line_number);
        exit(2);
    }
    if (
        ferror(stream)
        || !header_seen
        || !wires_seen
        || !gate_seen
        || !end_seen
    ) {
        fprintf(stderr, "circuit is incomplete\n");
        exit(2);
    }
    free(buffer);
    if (fclose(stream) != 0) {
        perror(path);
        exit(2);
    }
    return circuit;
}

static void emit_program(const struct circuit *circuit) {
    if (
        circuit->gate_count > SIZE_MAX - circuit->wires
        || circuit->cycles > UINT64_MAX / (uint64_t)circuit->gate_count
    ) {
        fprintf(stderr, "compiled phase dimensions overflow\n");
        exit(2);
    }
    const size_t program_base = 0;
    const size_t data_base = circuit->gate_count;
    const size_t registers = data_base + circuit->wires;

    printf("CATCAS_PHASE_PROGRAM 1\n");
    printf("REGISTERS %zu\n", registers);
    printf(
        "# compiled_from_fnv1a64 %016llx\n",
        (unsigned long long)circuit->fnv1a64
    );
    printf("# phase-resident program enables and data\n");
    for (size_t index = 0; index < circuit->gate_count; ++index) {
        if (circuit->gates[index].enabled) {
            printf("SET %zu 1\n", program_base + index);
        }
    }
    for (size_t wire = 0; wire < circuit->wires; ++wire) {
        if (circuit->input[wire] != 0) {
            printf(
                "SET %zu %d\n",
                data_base + wire,
                circuit->input[wire]
            );
        }
    }
    printf("PASSES %llu\n", (unsigned long long)circuit->cycles);

    for (size_t index = 0; index < circuit->gate_count; ++index) {
        const struct gate *gate = &circuit->gates[index];
        const size_t program = program_base + index;
        const size_t control = data_base + gate->control;
        const size_t left = data_base + gate->left;
        const size_t right = data_base + gate->right;
        printf("# slot %zu\n", index);
        printf(
            "PCSWAP %zu %zu %zu %zu\n",
            program,
            control,
            left,
            right
        );
    }
    printf("END\n");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s CIRCUIT.phasec\n", argv[0]);
        return 2;
    }
    struct circuit circuit = read_circuit(argv[1]);
    emit_program(&circuit);
    free(circuit.input);
    free(circuit.gates);
    return 0;
}
