/*
 * Mutable CAT_CAS compiler: Fredkin circuit -> public phase program.
 *
 * This compiler emits topology only. It never executes or adjudicates the
 * circuit. Program counter, program enables, workspaces, and data become
 * phase-resident registers consumed by streaming_phase_vm.c.
 */

#define _GNU_SOURCE
#include <ctype.h>
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
        char extra = '\0';
        if (!header_seen) {
            int version = 0;
            if (
                sscanf(
                    line,
                    "CATCAS_FREDKIN_CIRCUIT %d %c",
                    &version,
                    &extra
                ) != 1
                || version != 1
            ) {
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
            size_t wires = 0;
            if (
                wires_seen
                || gate_seen
                || sscanf(line, "WIRES %zu %c", &wires, &extra) != 1
                || wires < 3U
                || wires > SIZE_MAX / sizeof(int)
            ) {
                fprintf(stderr, "invalid WIRES on line %zu\n", line_number);
                exit(2);
            }
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
            size_t wire = 0;
            int symbol = 0;
            if (
                !wires_seen
                || gate_seen
                || sscanf(line, "SET %zu %d %c", &wire, &symbol, &extra) != 2
                || symbol < 0
                || symbol > 2
            ) {
                fprintf(stderr, "invalid SET on line %zu\n", line_number);
                exit(2);
            }
            require_wire(wire, circuit.wires, line_number);
            circuit.input[wire] = symbol;
            continue;
        }
        if (strncmp(line, "CYCLES", 6U) == 0) {
            unsigned long long cycles = 0;
            if (
                !wires_seen
                || cycles_seen
                || gate_seen
                || sscanf(line, "CYCLES %llu %c", &cycles, &extra) != 1
                || cycles == 0
            ) {
                fprintf(stderr, "invalid CYCLES on line %zu\n", line_number);
                exit(2);
            }
            circuit.cycles = (uint64_t)cycles;
            cycles_seen = 1;
            continue;
        }
        if (strncmp(line, "FREDKIN", 7U) == 0) {
            struct gate gate = {
                .control = 0,
                .left = 0,
                .right = 0,
                .enabled = 0
            };
            if (
                !wires_seen
                || sscanf(
                    line,
                    "FREDKIN %zu %zu %zu %d %c",
                    &gate.control,
                    &gate.left,
                    &gate.right,
                    &gate.enabled,
                    &extra
                ) != 4
                || (gate.enabled != 0 && gate.enabled != 1)
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
        circuit->gate_count > (SIZE_MAX - 2U) / 2U
        || circuit->gate_count > UINT64_MAX
        || circuit->cycles > UINT64_MAX / (uint64_t)circuit->gate_count
    ) {
        fprintf(stderr, "compiled phase dimensions overflow\n");
        exit(2);
    }
    const size_t pc_base = 0;
    const size_t program_base = circuit->gate_count;
    const size_t enable = 2U * circuit->gate_count;
    const size_t routed_enable = enable + 1U;
    const size_t data_base = routed_enable + 1U;
    if (circuit->wires > SIZE_MAX - data_base) {
        fprintf(stderr, "compiled register count overflows\n");
        exit(2);
    }
    const size_t registers = data_base + circuit->wires;
    const uint64_t passes =
        (uint64_t)circuit->gate_count * circuit->cycles;

    printf("CATCAS_PHASE_PROGRAM 1\n");
    printf("REGISTERS %zu\n", registers);
    printf(
        "# compiled_from_fnv1a64 %016llx\n",
        (unsigned long long)circuit->fnv1a64
    );
    printf("# phase-resident PC, program bits, workspaces, and data\n");
    printf("SET %zu 1\n", pc_base);
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
    printf("PASSES %llu\n", (unsigned long long)passes);

    for (size_t index = 0; index < circuit->gate_count; ++index) {
        const struct gate *gate = &circuit->gates[index];
        const size_t pc = pc_base + index;
        const size_t program = program_base + index;
        const size_t control = data_base + gate->control;
        const size_t left = data_base + gate->left;
        const size_t right = data_base + gate->right;
        printf("# slot %zu\n", index);
        printf("MULADD %zu %zu %zu\n", pc, program, enable);
        printf(
            "MULADD %zu %zu %zu\n",
            enable,
            control,
            routed_enable
        );
        printf(
            "CSWAP %zu %zu %zu\n",
            routed_enable,
            left,
            right
        );
        printf(
            "MULADD %zu %zu %zu\n",
            enable,
            control,
            routed_enable
        );
        printf(
            "MULADD %zu %zu %zu\n",
            enable,
            control,
            routed_enable
        );
        printf("MULADD %zu %zu %zu\n", pc, program, enable);
        printf("MULADD %zu %zu %zu\n", pc, program, enable);
    }
    for (size_t index = circuit->gate_count; index-- > 1U;) {
        printf("SWAP %zu %zu\n", pc_base + index, pc_base + index - 1U);
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
