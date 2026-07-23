/*
 * Independent compact evaluator for public CAT_CAS phase programs.
 *
 * This executable contains no complex carrier code and is never linked into
 * streaming_phase_vm.c. It adjudicates a sealed native boundary afterward.
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#define Q 3

enum opcode {
    OP_ROT = 1,
    OP_ADD = 2,
    OP_MULADD = 3,
    OP_SWAP = 4,
    OP_CSWAP = 5
};

struct instruction {
    enum opcode op;
    size_t a;
    size_t b;
    size_t target;
    int amount;
};

struct program {
    size_t registers;
    uint8_t *input;
    struct instruction *instructions;
    size_t count;
    size_t capacity;
    uint64_t passes;
    uint64_t fnv1a64;
};

static inline uint64_t monotonic_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000)
        + (uint64_t)value.tv_nsec;
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

static void require_register(
    size_t index, size_t registers, size_t line_number
) {
    if (index >= registers) {
        fprintf(
            stderr,
            "reference register %zu is out of range on line %zu\n",
            index,
            line_number
        );
        exit(2);
    }
}

static void append(
    struct program *program, struct instruction instruction
) {
    if (program->count == program->capacity) {
        const size_t next = program->capacity == 0
            ? 64U
            : program->capacity * 2U;
        if (
            next < program->capacity
            || next > SIZE_MAX / sizeof(struct instruction)
        ) {
            fprintf(stderr, "reference program is too large\n");
            exit(2);
        }
        struct instruction *instructions = realloc(
            program->instructions,
            next * sizeof(struct instruction)
        );
        if (instructions == NULL) {
            fprintf(stderr, "reference allocation failed\n");
            exit(2);
        }
        program->instructions = instructions;
        program->capacity = next;
    }
    program->instructions[program->count++] = instruction;
}

static struct program read_program(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct program program = {
        .registers = 0,
        .input = NULL,
        .instructions = NULL,
        .count = 0,
        .capacity = 0,
        .passes = 1,
        .fnv1a64 = UINT64_C(14695981039346656037)
    };
    char *buffer = NULL;
    size_t buffer_capacity = 0;
    ssize_t read_length = 0;
    size_t line_number = 0;
    int header_seen = 0;
    int registers_seen = 0;
    int passes_seen = 0;
    int instruction_seen = 0;
    int end_seen = 0;
    while (
        (read_length = getline(&buffer, &buffer_capacity, stream)) >= 0
    ) {
        ++line_number;
        const size_t raw_length = (size_t)read_length;
        if (memchr(buffer, '\0', raw_length) != NULL) {
            fprintf(stderr, "reference embedded NUL on line %zu\n", line_number);
            exit(2);
        }
        program.fnv1a64 = fnv1a_update(
            program.fnv1a64,
            (const unsigned char *)buffer,
            raw_length
        );
        if (
            raw_length > 4095U
        ) {
            fprintf(stderr, "reference line %zu is too long\n", line_number);
            exit(2);
        }
        char *line = trim_line(buffer);
        if (*line == '\0' || *line == '#') {
            continue;
        }
        if (end_seen) {
            fprintf(stderr, "reference content follows END\n");
            exit(2);
        }
        char extra = '\0';
        if (!header_seen) {
            int version = 0;
            if (
                sscanf(
                    line,
                    "CATCAS_PHASE_PROGRAM %d %c",
                    &version,
                    &extra
                ) != 1
                || version != 1
            ) {
                fprintf(stderr, "reference header is invalid\n");
                exit(2);
            }
            header_seen = 1;
            continue;
        }
        if (strcmp(line, "END") == 0) {
            if (!registers_seen || !instruction_seen) {
                fprintf(stderr, "reference END precedes program\n");
                exit(2);
            }
            end_seen = 1;
            continue;
        }
        if (strncmp(line, "REGISTERS", 9U) == 0) {
            size_t registers = 0;
            if (
                registers_seen
                || instruction_seen
                || sscanf(line, "REGISTERS %zu %c", &registers, &extra) != 1
                || registers < 3U
                || registers > SIZE_MAX / sizeof(uint8_t)
            ) {
                fprintf(stderr, "reference REGISTERS is invalid\n");
                exit(2);
            }
            program.registers = registers;
            program.input = calloc(registers, sizeof(uint8_t));
            if (program.input == NULL) {
                fprintf(stderr, "reference input allocation failed\n");
                exit(2);
            }
            registers_seen = 1;
            continue;
        }
        if (strncmp(line, "SET", 3U) == 0) {
            size_t target = 0;
            int symbol = 0;
            if (
                !registers_seen
                || instruction_seen
                || sscanf(line, "SET %zu %d %c", &target, &symbol, &extra) != 2
                || symbol < 0
                || symbol >= Q
            ) {
                fprintf(stderr, "reference SET is invalid\n");
                exit(2);
            }
            require_register(target, program.registers, line_number);
            program.input[target] = (uint8_t)symbol;
            continue;
        }
        if (strncmp(line, "PASSES", 6U) == 0) {
            unsigned long long passes = 0;
            if (
                !registers_seen
                || passes_seen
                || instruction_seen
                || sscanf(line, "PASSES %llu %c", &passes, &extra) != 1
                || passes == 0
            ) {
                fprintf(stderr, "reference PASSES is invalid\n");
                exit(2);
            }
            program.passes = (uint64_t)passes;
            passes_seen = 1;
            continue;
        }
        if (!registers_seen) {
            fprintf(stderr, "reference instruction precedes REGISTERS\n");
            exit(2);
        }
        struct instruction instruction = {
            .op = OP_ROT,
            .a = 0,
            .b = 0,
            .target = 0,
            .amount = 0
        };
        if (strncmp(line, "ROT", 3U) == 0) {
            if (
                sscanf(
                    line,
                    "ROT %zu %d %c",
                    &instruction.target,
                    &instruction.amount,
                    &extra
                ) != 2
            ) {
                fprintf(stderr, "reference ROT is invalid\n");
                exit(2);
            }
            instruction.op = OP_ROT;
            require_register(
                instruction.target, program.registers, line_number
            );
        } else if (strncmp(line, "ADD", 3U) == 0) {
            if (
                sscanf(
                    line,
                    "ADD %zu %zu %c",
                    &instruction.a,
                    &instruction.target,
                    &extra
                ) != 2
            ) {
                fprintf(stderr, "reference ADD is invalid\n");
                exit(2);
            }
            instruction.op = OP_ADD;
            require_register(instruction.a, program.registers, line_number);
            require_register(
                instruction.target, program.registers, line_number
            );
            if (instruction.a == instruction.target) {
                fprintf(stderr, "reference ADD aliases target\n");
                exit(2);
            }
        } else if (strncmp(line, "MULADD", 6U) == 0) {
            if (
                sscanf(
                    line,
                    "MULADD %zu %zu %zu %c",
                    &instruction.a,
                    &instruction.b,
                    &instruction.target,
                    &extra
                ) != 3
            ) {
                fprintf(stderr, "reference MULADD is invalid\n");
                exit(2);
            }
            instruction.op = OP_MULADD;
            require_register(instruction.a, program.registers, line_number);
            require_register(instruction.b, program.registers, line_number);
            require_register(
                instruction.target, program.registers, line_number
            );
            if (
                instruction.a == instruction.target
                || instruction.b == instruction.target
            ) {
                fprintf(stderr, "reference MULADD aliases target\n");
                exit(2);
            }
        } else if (strncmp(line, "SWAP", 4U) == 0) {
            if (
                sscanf(
                    line,
                    "SWAP %zu %zu %c",
                    &instruction.a,
                    &instruction.b,
                    &extra
                ) != 2
            ) {
                fprintf(stderr, "reference SWAP is invalid\n");
                exit(2);
            }
            instruction.op = OP_SWAP;
            require_register(instruction.a, program.registers, line_number);
            require_register(instruction.b, program.registers, line_number);
            if (instruction.a == instruction.b) {
                fprintf(stderr, "reference SWAP aliases itself\n");
                exit(2);
            }
        } else if (strncmp(line, "CSWAP", 5U) == 0) {
            if (
                sscanf(
                    line,
                    "CSWAP %zu %zu %zu %c",
                    &instruction.target,
                    &instruction.a,
                    &instruction.b,
                    &extra
                ) != 3
            ) {
                fprintf(stderr, "reference CSWAP is invalid\n");
                exit(2);
            }
            instruction.op = OP_CSWAP;
            require_register(
                instruction.target, program.registers, line_number
            );
            require_register(instruction.a, program.registers, line_number);
            require_register(instruction.b, program.registers, line_number);
            if (
                instruction.a == instruction.b
                || instruction.target == instruction.a
                || instruction.target == instruction.b
            ) {
                fprintf(stderr, "reference CSWAP registers alias\n");
                exit(2);
            }
        } else {
            fprintf(stderr, "reference instruction is unknown\n");
            exit(2);
        }
        append(&program, instruction);
        instruction_seen = 1;
    }
    if (
        ferror(stream)
        || !header_seen
        || !registers_seen
        || !instruction_seen
        || !end_seen
    ) {
        fprintf(stderr, "reference program is incomplete\n");
        exit(2);
    }
    free(buffer);
    if (fclose(stream) != 0) {
        perror(path);
        exit(2);
    }
    return program;
}

static void run_program(const struct program *program) {
    uint8_t *state = malloc(program->registers * sizeof(uint8_t));
    if (state == NULL) {
        fprintf(stderr, "reference state allocation failed\n");
        exit(2);
    }
    memcpy(state, program->input, program->registers * sizeof(uint8_t));
    const uint64_t start = monotonic_ns();
    for (uint64_t pass = 0; pass < program->passes; ++pass) {
        for (size_t index = 0; index < program->count; ++index) {
            const struct instruction *instruction =
                &program->instructions[index];
        if (instruction->op == OP_ROT) {
            int value = (int)state[instruction->target]
                + instruction->amount;
            value %= Q;
            if (value < 0) {
                value += Q;
            }
            state[instruction->target] = (uint8_t)value;
        } else if (instruction->op == OP_ADD) {
            state[instruction->target] = (uint8_t)(
                (
                    state[instruction->target]
                    + state[instruction->a]
                )
                % Q
            );
        } else if (instruction->op == OP_MULADD) {
            state[instruction->target] = (uint8_t)(
                (
                    state[instruction->target]
                    + state[instruction->a] * state[instruction->b]
                )
                % Q
            );
        } else if (instruction->op == OP_SWAP) {
            const uint8_t temporary = state[instruction->a];
            state[instruction->a] = state[instruction->b];
            state[instruction->b] = temporary;
        } else if (instruction->op == OP_CSWAP) {
            if (state[instruction->target] == 1U) {
                const uint8_t temporary = state[instruction->a];
                state[instruction->a] = state[instruction->b];
                state[instruction->b] = temporary;
            }
        } else {
            abort();
        }
        }
    }
    const uint64_t elapsed = monotonic_ns() - start;
    uint64_t boundary_hash = UINT64_C(14695981039346656037);
    size_t boundary_nonzero = 0;
    for (size_t index = 0; index < program->registers; ++index) {
        boundary_hash = fnv1a_update(
            boundary_hash, &state[index], 1U
        );
        if (state[index] != 0U) {
            ++boundary_nonzero;
        }
    }
    printf(
        "{\"public_program_fnv1a64\":\"%016llx\","
        "\"instructions\":%zu,\"passes\":%llu,"
        "\"steps\":%llu,\"registers\":%zu,"
        "\"boundary_fnv1a64\":\"%016llx\","
        "\"boundary_nonzero\":%zu,\"boundary_symbols\":",
        (unsigned long long)program->fnv1a64,
        program->count,
        (unsigned long long)program->passes,
        (unsigned long long)(
            (uint64_t)program->count * program->passes
        ),
        program->registers,
        (unsigned long long)boundary_hash,
        boundary_nonzero
    );
    if (program->registers <= 256U) {
        putchar('[');
        for (size_t index = 0; index < program->registers; ++index) {
            printf("%s%u", index == 0 ? "" : ",", state[index]);
        }
        putchar(']');
    } else {
        fputs("null", stdout);
    }
    printf(
        ",\"compact_forward_ns\":%llu}\n",
        (unsigned long long)elapsed
    );
    free(state);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s PROGRAM.holo\n", argv[0]);
        return 2;
    }
    struct program program = read_program(argv[1]);
    if (
        program.count > UINT64_MAX
        || (
            program.count > 0
            && program.passes > UINT64_MAX / (uint64_t)program.count
        )
    ) {
        fprintf(stderr, "reference step count overflows uint64\n");
        free(program.input);
        free(program.instructions);
        return 2;
    }
    run_program(&program);
    free(program.input);
    free(program.instructions);
    return 0;
}
