/*
 * Independent compact evaluator for public CAT_CAS phase programs.
 *
 * This executable contains no complex carrier code and is never linked into
 * streaming_phase_vm.c. It adjudicates a sealed native boundary afterward.
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <limits.h>
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
    OP_CSWAP = 5,
    OP_PCSWAP = 6
};

struct instruction {
    enum opcode op;
    size_t a;
    size_t b;
    size_t c;
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
    const char *line, const char *keyword
) {
    const size_t keyword_length = strlen(keyword);
    if (
        strncmp(line, keyword, keyword_length) != 0
        || !isspace((unsigned char)line[keyword_length])
    ) {
        fprintf(stderr, "reference %s is invalid\n", keyword);
        exit(2);
    }
    const char *value = line + keyword_length;
    while (isspace((unsigned char)*value)) {
        ++value;
    }
    return parse_positive_decimal(value, keyword);
}

static const char *record_values(
    const char *line, const char *keyword
) {
    const size_t length = strlen(keyword);
    if (
        strncmp(line, keyword, length) != 0
        || !isspace((unsigned char)line[length])
    ) {
        fprintf(stderr, "reference %s is invalid\n", keyword);
        exit(2);
    }
    const char *cursor = line + length;
    while (isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    return cursor;
}

static uint64_t parse_unsigned_token(
    const char **cursor, const char *keyword
) {
    if (**cursor < '0' || **cursor > '9') {
        fprintf(
            stderr,
            "reference unsigned %s field is invalid\n",
            keyword
        );
        exit(2);
    }
    char *end = NULL;
    errno = 0;
    const unsigned long long value =
        strtoull(*cursor, &end, 10);
    if (
        end == *cursor
        || errno == ERANGE
        || (*end != '\0' && !isspace((unsigned char)*end))
    ) {
        fprintf(
            stderr,
            "reference unsigned %s field is invalid\n",
            keyword
        );
        exit(2);
    }
    *cursor = end;
    while (isspace((unsigned char)**cursor)) {
        ++*cursor;
    }
    return (uint64_t)value;
}

static int parse_signed_int_token(
    const char **cursor, const char *keyword
) {
    const char *start = *cursor;
    if (*start == '-') {
        ++start;
    }
    if (*start < '0' || *start > '9') {
        fprintf(
            stderr,
            "reference signed %s field is invalid\n",
            keyword
        );
        exit(2);
    }
    char *end = NULL;
    errno = 0;
    const long value = strtol(*cursor, &end, 10);
    if (
        end == *cursor
        || errno == ERANGE
        || value < INT_MIN
        || value > INT_MAX
        || (*end != '\0' && !isspace((unsigned char)*end))
    ) {
        fprintf(
            stderr,
            "reference signed %s field is invalid\n",
            keyword
        );
        exit(2);
    }
    *cursor = end;
    while (isspace((unsigned char)**cursor)) {
        ++*cursor;
    }
    return (int)value;
}

static void require_record_end(
    const char *cursor, const char *keyword
) {
    if (*cursor != '\0') {
        fprintf(stderr, "reference %s has extra fields\n", keyword);
        exit(2);
    }
}

static void parse_unsigned_record(
    const char *line,
    const char *keyword,
    uint64_t *values,
    size_t count
) {
    const char *cursor = record_values(line, keyword);
    for (size_t index = 0; index < count; ++index) {
        values[index] = parse_unsigned_token(&cursor, keyword);
    }
    require_record_end(cursor, keyword);
}

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
        if (!header_seen) {
            uint64_t fields[1] = {0};
            parse_unsigned_record(
                line, "CATCAS_PHASE_PROGRAM", fields, 1
            );
            if (fields[0] != 1U) {
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
            uint64_t fields[1] = {0};
            parse_unsigned_record(
                line, "REGISTERS", fields, 1
            );
            if (
                registers_seen
                || instruction_seen
                || fields[0] > SIZE_MAX
                || fields[0] < 3U
                || (size_t)fields[0] > SIZE_MAX / sizeof(uint8_t)
            ) {
                fprintf(stderr, "reference REGISTERS is invalid\n");
                exit(2);
            }
            const size_t registers = (size_t)fields[0];
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
            uint64_t fields[2] = {0, 0};
            parse_unsigned_record(line, "SET", fields, 2);
            if (
                !registers_seen
                || instruction_seen
                || fields[0] > SIZE_MAX
                || fields[1] >= Q
            ) {
                fprintf(stderr, "reference SET is invalid\n");
                exit(2);
            }
            const size_t target = (size_t)fields[0];
            const int symbol = (int)fields[1];
            require_register(target, program.registers, line_number);
            program.input[target] = (uint8_t)symbol;
            continue;
        }
        if (strncmp(line, "PASSES", 6U) == 0) {
            if (
                !registers_seen
                || passes_seen
                || instruction_seen
            ) {
                fprintf(stderr, "reference PASSES is invalid\n");
                exit(2);
            }
            program.passes = parse_positive_record(line, "PASSES");
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
            .c = 0,
            .target = 0,
            .amount = 0
        };
        if (strncmp(line, "ROT", 3U) == 0) {
            const char *cursor = record_values(line, "ROT");
            const uint64_t target =
                parse_unsigned_token(&cursor, "ROT");
            instruction.amount =
                parse_signed_int_token(&cursor, "ROT");
            require_record_end(cursor, "ROT");
            if (target > SIZE_MAX) {
                fprintf(stderr, "reference ROT is invalid\n");
                exit(2);
            }
            instruction.target = (size_t)target;
            instruction.op = OP_ROT;
            require_register(
                instruction.target, program.registers, line_number
            );
        } else if (strncmp(line, "ADD", 3U) == 0) {
            uint64_t fields[2] = {0, 0};
            parse_unsigned_record(line, "ADD", fields, 2);
            if (fields[0] > SIZE_MAX || fields[1] > SIZE_MAX) {
                fprintf(stderr, "reference ADD is invalid\n");
                exit(2);
            }
            instruction.a = (size_t)fields[0];
            instruction.target = (size_t)fields[1];
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
            uint64_t fields[3] = {0, 0, 0};
            parse_unsigned_record(line, "MULADD", fields, 3);
            if (
                fields[0] > SIZE_MAX
                || fields[1] > SIZE_MAX
                || fields[2] > SIZE_MAX
            ) {
                fprintf(stderr, "reference MULADD is invalid\n");
                exit(2);
            }
            instruction.a = (size_t)fields[0];
            instruction.b = (size_t)fields[1];
            instruction.target = (size_t)fields[2];
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
            uint64_t fields[2] = {0, 0};
            parse_unsigned_record(line, "SWAP", fields, 2);
            if (fields[0] > SIZE_MAX || fields[1] > SIZE_MAX) {
                fprintf(stderr, "reference SWAP is invalid\n");
                exit(2);
            }
            instruction.a = (size_t)fields[0];
            instruction.b = (size_t)fields[1];
            instruction.op = OP_SWAP;
            require_register(instruction.a, program.registers, line_number);
            require_register(instruction.b, program.registers, line_number);
            if (instruction.a == instruction.b) {
                fprintf(stderr, "reference SWAP aliases itself\n");
                exit(2);
            }
        } else if (strncmp(line, "CSWAP", 5U) == 0) {
            uint64_t fields[3] = {0, 0, 0};
            parse_unsigned_record(line, "CSWAP", fields, 3);
            if (
                fields[0] > SIZE_MAX
                || fields[1] > SIZE_MAX
                || fields[2] > SIZE_MAX
            ) {
                fprintf(stderr, "reference CSWAP is invalid\n");
                exit(2);
            }
            instruction.target = (size_t)fields[0];
            instruction.a = (size_t)fields[1];
            instruction.b = (size_t)fields[2];
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
        } else if (strncmp(line, "PCSWAP", 6U) == 0) {
            uint64_t fields[4] = {0, 0, 0, 0};
            parse_unsigned_record(line, "PCSWAP", fields, 4);
            if (
                fields[0] > SIZE_MAX
                || fields[1] > SIZE_MAX
                || fields[2] > SIZE_MAX
                || fields[3] > SIZE_MAX
            ) {
                fprintf(stderr, "reference PCSWAP is invalid\n");
                exit(2);
            }
            instruction.target = (size_t)fields[0];
            instruction.a = (size_t)fields[1];
            instruction.b = (size_t)fields[2];
            instruction.c = (size_t)fields[3];
            instruction.op = OP_PCSWAP;
            require_register(
                instruction.target, program.registers, line_number
            );
            require_register(instruction.a, program.registers, line_number);
            require_register(instruction.b, program.registers, line_number);
            require_register(instruction.c, program.registers, line_number);
            if (
                instruction.target == instruction.a
                || instruction.target == instruction.b
                || instruction.target == instruction.c
                || instruction.a == instruction.b
                || instruction.a == instruction.c
                || instruction.b == instruction.c
            ) {
                fprintf(stderr, "reference PCSWAP registers alias\n");
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
        } else if (instruction->op == OP_PCSWAP) {
            if (
                (
                    state[instruction->target]
                    * state[instruction->a]
                ) % Q == 1U
            ) {
                const uint8_t temporary = state[instruction->b];
                state[instruction->b] = state[instruction->c];
                state[instruction->c] = temporary;
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
