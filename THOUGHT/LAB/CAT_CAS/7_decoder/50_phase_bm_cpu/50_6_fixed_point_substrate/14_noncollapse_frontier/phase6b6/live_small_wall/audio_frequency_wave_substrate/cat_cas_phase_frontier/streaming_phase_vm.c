/*
 * Mutable CAT_CAS frontier: streaming, program-derived-inverse phase VM.
 *
 * The native state is the relative phase of twin complex rails. The VM uses a
 * ternary root alphabet only at its public load and collapse boundaries.
 * During execution ROT, ADD, MULADD, SWAP, CSWAP, and PCSWAP operate on
 * complex relations. MULADD and PCSWAP are fixed roots-of-unity interpolation
 * networks, not decoded branches.
 *
 * This source contains no conventional evaluator. A separate verifier may
 * adjudicate the emitted boundary after execution.
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <complex.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#define Q 3
#define PI 3.141592653589793238462643383279502884
#define RESTORATION_MAX 2.0e-11
#define LOCK_ERROR_TRIGGER 1.0e-12

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

struct public_program {
    size_t registers;
    int *input_symbols;
    struct instruction *instructions;
    uint64_t instruction_count;
    uint64_t passes;
    size_t instruction_capacity;
    uint64_t fnv1a64;
};

struct instruction_source {
    uint64_t steps;
    int family;
    size_t registers;
    const int *input_symbols;
    const struct instruction *instructions;
    uint64_t instruction_period;
    uint64_t public_program_fnv1a64;
};

struct carrier {
    size_t registers;
    double complex *baseline;
    double complex *working;
};

struct execution {
    uint64_t steps;
    int family;
    size_t registers;
    int *boundary_symbols;
    double maximum_root_distance;
    double displacement_l2;
    double restoration_max_abs;
    uint64_t forward_ns;
    uint64_t inverse_ns;
    uint64_t public_program_fnv1a64;
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

static inline double complex unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        fprintf(stderr, "collapsed or nonfinite phase relation\n");
        exit(2);
    }
    return value / magnitude;
}

static inline double complex root_of_unity(int value) {
    int reduced = value % Q;
    if (reduced < 0) {
        reduced += Q;
    }
    if (reduced == 0) {
        return 1.0 + 0.0 * I;
    }
    if (reduced == 1) {
        return -0.5 + 0.86602540378443864676 * I;
    }
    return -0.5 - 0.86602540378443864676 * I;
}

static inline double complex cube(double complex value) {
    return value * value * value;
}

static inline double complex phase_lock(double complex value) {
    double complex locked = value;
    const double error_force = cimag(cube(locked));
    const double real_part = creal(locked);
    const double imag_part = cimag(locked);
    const double magnitude_error =
        real_part * real_part + imag_part * imag_part - 1.0;
    if (
        fabs(error_force) <= LOCK_ERROR_TRIGGER
        && fabs(magnitude_error) <= LOCK_ERROR_TRIGGER
    ) {
        return locked;
    }
    locked *= 1.0 - I * (error_force / 3.0);
    return unit(locked);
}

static inline double complex relation(
    const struct carrier *carrier, size_t register_index
) {
    return carrier->working[register_index]
        * conj(carrier->baseline[register_index]);
}

static inline void write_relation(
    struct carrier *carrier,
    size_t register_index,
    double complex value
) {
    carrier->working[register_index] =
        phase_lock(value) * carrier->baseline[register_index];
}

static inline void write_relation_unlocked(
    struct carrier *carrier,
    size_t register_index,
    double complex value
) {
    carrier->working[register_index] =
        value * carrier->baseline[register_index];
}

static double complex product_factor(
    double complex left, double complex right
) {
    const double complex left_squared = conj(left);
    const double complex right_squared = conj(right);
    const double complex product = left * right;
    const double complex both_squared = conj(product);
    const double complex left_right_squared = left * right_squared;
    const double complex left_squared_right = left_squared * right;
    const double complex factor = (
        1.0
        + left
        + left_squared
        + right
        + right_squared
        + root_of_unity(2) * (product + both_squared)
        + root_of_unity(1)
            * (left_right_squared + left_squared_right)
    ) / 3.0;
    return factor;
}

static double complex boolean_one_indicator(double complex control) {
    const double complex squared_value = conj(control);
    const double complex squared_symbol =
        product_factor(control, control);
    return phase_lock(
        squared_value * squared_symbol * squared_symbol
    );
}

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
    const size_t length = strlen(keyword);
    if (
        strncmp(line, keyword, length) != 0
        || !isspace((unsigned char)line[length])
    ) {
        fprintf(
            stderr,
            "invalid %s on line %zu\n",
            keyword,
            line_number
        );
        exit(2);
    }
    const char *cursor = line + length;
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
    if (**cursor < '0' || **cursor > '9') {
        fprintf(
            stderr,
            "invalid unsigned %s field on line %zu\n",
            keyword,
            line_number
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
            "invalid unsigned %s field on line %zu\n",
            keyword,
            line_number
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
    const char **cursor,
    const char *keyword,
    size_t line_number
) {
    const char *start = *cursor;
    if (*start == '-') {
        ++start;
    }
    if (*start < '0' || *start > '9') {
        fprintf(
            stderr,
            "invalid signed %s field on line %zu\n",
            keyword,
            line_number
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
            "invalid signed %s field on line %zu\n",
            keyword,
            line_number
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
    const char *cursor,
    const char *keyword,
    size_t line_number
) {
    if (*cursor != '\0') {
        fprintf(
            stderr,
            "extra %s field on line %zu\n",
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
    size_t count,
    size_t line_number
) {
    const char *cursor = record_values(
        line,
        keyword,
        line_number
    );
    for (size_t index = 0; index < count; ++index) {
        values[index] = parse_unsigned_token(
            &cursor,
            keyword,
            line_number
        );
    }
    require_record_end(cursor, keyword, line_number);
}

static struct instruction instruction_at(
    uint64_t index, int family, size_t registers
) {
    const uint64_t block = index / 8U;
    const int slot = (int)(index % 8U);
    const size_t base =
        (size_t)((block * (uint64_t)(5 + 2 * family)) % registers);
    const size_t a = base;
    const size_t b = (base + 1U) % registers;
    const size_t target = (base + 2U) % registers;
    struct instruction instruction = {
        .op = OP_ROT,
        .a = a,
        .b = b,
        .target = target,
        .amount = 1 + family
    };
    switch (slot) {
        case 0:
            instruction.op = OP_ROT;
            instruction.target = target;
            instruction.amount = 1 + family;
            break;
        case 1:
            instruction.op = OP_ADD;
            instruction.a = a;
            instruction.target = target;
            break;
        case 2:
            instruction.op = OP_MULADD;
            instruction.a = a;
            instruction.b = b;
            instruction.target = target;
            break;
        case 3:
            instruction.op = OP_SWAP;
            instruction.a = a;
            instruction.b = b;
            break;
        case 4:
            instruction.op = OP_ROT;
            instruction.target = b;
            instruction.amount = 2 - family;
            break;
        case 5:
            instruction.op = OP_ADD;
            instruction.a = target;
            instruction.target = b;
            break;
        case 6:
            instruction.op = OP_MULADD;
            instruction.a = b;
            instruction.b = target;
            instruction.target = a;
            break;
        case 7:
            instruction.op = OP_SWAP;
            instruction.a = target;
            instruction.b = a;
            break;
        default:
            abort();
    }
    return instruction;
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

static void append_instruction(
    struct public_program *program, struct instruction instruction
) {
    if (program->instruction_count == program->instruction_capacity) {
        size_t next_capacity =
            program->instruction_capacity == 0
                ? 64U
                : program->instruction_capacity * 2U;
        if (
            next_capacity < program->instruction_capacity
            || next_capacity > SIZE_MAX / sizeof(struct instruction)
        ) {
            fprintf(stderr, "public program is too large\n");
            exit(2);
        }
        struct instruction *next = realloc(
            program->instructions,
            next_capacity * sizeof(struct instruction)
        );
        if (next == NULL) {
            fprintf(stderr, "public program allocation failed\n");
            exit(2);
        }
        program->instructions = next;
        program->instruction_capacity = next_capacity;
    }
    program->instructions[program->instruction_count++] = instruction;
}

static void require_register(
    size_t index, size_t registers, size_t line_number
) {
    if (index >= registers) {
        fprintf(
            stderr,
            "register %zu is out of range on line %zu\n",
            index,
            line_number
        );
        exit(2);
    }
}

static struct public_program read_public_program(const char *path) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        perror(path);
        exit(2);
    }
    struct public_program program = {
        .registers = 0,
        .input_symbols = NULL,
        .instructions = NULL,
        .instruction_count = 0,
        .passes = 1,
        .instruction_capacity = 0,
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
            fprintf(stderr, "embedded NUL on line %zu\n", line_number);
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
                "CATCAS_PHASE_PROGRAM",
                fields,
                1,
                line_number
            );
            if (fields[0] != 1U) {
                fprintf(stderr, "invalid program header on line %zu\n", line_number);
                exit(2);
            }
            header_seen = 1;
            continue;
        }
        if (strcmp(line, "END") == 0) {
            if (!registers_seen || !instruction_seen) {
                fprintf(stderr, "END precedes a complete program\n");
                exit(2);
            }
            end_seen = 1;
            continue;
        }
        if (strncmp(line, "REGISTERS", 9U) == 0) {
            uint64_t fields[1] = {0};
            parse_unsigned_record(
                line,
                "REGISTERS",
                fields,
                1,
                line_number
            );
            if (
                registers_seen
                || instruction_seen
                || fields[0] > SIZE_MAX
                || fields[0] < 3U
                || (size_t)fields[0] > SIZE_MAX / sizeof(int)
            ) {
                fprintf(stderr, "invalid REGISTERS on line %zu\n", line_number);
                exit(2);
            }
            const size_t registers = (size_t)fields[0];
            program.registers = registers;
            program.input_symbols = calloc(registers, sizeof(int));
            if (program.input_symbols == NULL) {
                fprintf(stderr, "input allocation failed\n");
                exit(2);
            }
            registers_seen = 1;
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
                !registers_seen
                || instruction_seen
                || fields[0] > SIZE_MAX
                || fields[1] >= Q
            ) {
                fprintf(stderr, "invalid SET on line %zu\n", line_number);
                exit(2);
            }
            const size_t target = (size_t)fields[0];
            const int symbol = (int)fields[1];
            require_register(target, program.registers, line_number);
            program.input_symbols[target] = symbol;
            continue;
        }
        if (strncmp(line, "PASSES", 6U) == 0) {
            if (
                !registers_seen
                || passes_seen
                || instruction_seen
            ) {
                fprintf(stderr, "invalid PASSES on line %zu\n", line_number);
                exit(2);
            }
            program.passes = parse_positive_record(
                line,
                "PASSES",
                line_number
            );
            passes_seen = 1;
            continue;
        }
        if (!registers_seen) {
            fprintf(stderr, "instruction precedes REGISTERS on line %zu\n", line_number);
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
            const char *cursor = record_values(
                line, "ROT", line_number
            );
            const uint64_t target = parse_unsigned_token(
                &cursor, "ROT", line_number
            );
            instruction.amount = parse_signed_int_token(
                &cursor, "ROT", line_number
            );
            require_record_end(cursor, "ROT", line_number);
            if (target > SIZE_MAX) {
                fprintf(stderr, "invalid ROT on line %zu\n", line_number);
                exit(2);
            }
            instruction.target = (size_t)target;
            instruction.op = OP_ROT;
            require_register(
                instruction.target, program.registers, line_number
            );
        } else if (strncmp(line, "ADD", 3U) == 0) {
            uint64_t fields[2] = {0, 0};
            parse_unsigned_record(
                line, "ADD", fields, 2, line_number
            );
            if (fields[0] > SIZE_MAX || fields[1] > SIZE_MAX) {
                fprintf(stderr, "invalid ADD on line %zu\n", line_number);
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
                fprintf(stderr, "ADD source aliases target on line %zu\n", line_number);
                exit(2);
            }
        } else if (strncmp(line, "MULADD", 6U) == 0) {
            uint64_t fields[3] = {0, 0, 0};
            parse_unsigned_record(
                line, "MULADD", fields, 3, line_number
            );
            if (
                fields[0] > SIZE_MAX
                || fields[1] > SIZE_MAX
                || fields[2] > SIZE_MAX
            ) {
                fprintf(stderr, "invalid MULADD on line %zu\n", line_number);
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
                fprintf(
                    stderr,
                    "MULADD input aliases target on line %zu\n",
                    line_number
                );
                exit(2);
            }
        } else if (strncmp(line, "SWAP", 4U) == 0) {
            uint64_t fields[2] = {0, 0};
            parse_unsigned_record(
                line, "SWAP", fields, 2, line_number
            );
            if (fields[0] > SIZE_MAX || fields[1] > SIZE_MAX) {
                fprintf(stderr, "invalid SWAP on line %zu\n", line_number);
                exit(2);
            }
            instruction.a = (size_t)fields[0];
            instruction.b = (size_t)fields[1];
            instruction.op = OP_SWAP;
            require_register(instruction.a, program.registers, line_number);
            require_register(instruction.b, program.registers, line_number);
            if (instruction.a == instruction.b) {
                fprintf(stderr, "SWAP aliases itself on line %zu\n", line_number);
                exit(2);
            }
        } else if (strncmp(line, "CSWAP", 5U) == 0) {
            uint64_t fields[3] = {0, 0, 0};
            parse_unsigned_record(
                line, "CSWAP", fields, 3, line_number
            );
            if (
                fields[0] > SIZE_MAX
                || fields[1] > SIZE_MAX
                || fields[2] > SIZE_MAX
            ) {
                fprintf(stderr, "invalid CSWAP on line %zu\n", line_number);
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
                fprintf(stderr, "CSWAP registers alias on line %zu\n", line_number);
                exit(2);
            }
        } else if (strncmp(line, "PCSWAP", 6U) == 0) {
            uint64_t fields[4] = {0, 0, 0, 0};
            parse_unsigned_record(
                line, "PCSWAP", fields, 4, line_number
            );
            if (
                fields[0] > SIZE_MAX
                || fields[1] > SIZE_MAX
                || fields[2] > SIZE_MAX
                || fields[3] > SIZE_MAX
            ) {
                fprintf(stderr, "invalid PCSWAP on line %zu\n", line_number);
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
                fprintf(
                    stderr,
                    "PCSWAP registers alias on line %zu\n",
                    line_number
                );
                exit(2);
            }
        } else {
            fprintf(stderr, "unknown instruction on line %zu\n", line_number);
            exit(2);
        }
        append_instruction(&program, instruction);
        instruction_seen = 1;
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
    if (!header_seen || !registers_seen || !instruction_seen || !end_seen) {
        fprintf(stderr, "public program is incomplete\n");
        exit(2);
    }
    return program;
}

static void free_public_program(struct public_program *program) {
    free(program->input_symbols);
    free(program->instructions);
    program->input_symbols = NULL;
    program->instructions = NULL;
    program->registers = 0;
    program->instruction_count = 0;
    program->passes = 0;
    program->instruction_capacity = 0;
}

static struct instruction source_instruction_at(
    const struct instruction_source *source, uint64_t index
) {
    if (source->instructions != NULL) {
        return source->instructions[index % source->instruction_period];
    }
    return instruction_at(index, source->family, source->registers);
}

static void apply_forward(
    struct carrier *carrier, const struct instruction *instruction
) {
    if (instruction->op == OP_ROT) {
        write_relation_unlocked(
            carrier,
            instruction->target,
            relation(carrier, instruction->target)
                * root_of_unity(instruction->amount)
        );
        return;
    }
    if (instruction->op == OP_ADD) {
        write_relation(
            carrier,
            instruction->target,
            relation(carrier, instruction->target)
                * relation(carrier, instruction->a)
        );
        return;
    }
    if (instruction->op == OP_MULADD) {
        const double complex factor = product_factor(
            relation(carrier, instruction->a),
            relation(carrier, instruction->b)
        );
        write_relation(
            carrier,
            instruction->target,
            relation(carrier, instruction->target) * factor
        );
        return;
    }
    if (instruction->op == OP_SWAP) {
        const double complex baseline =
            carrier->baseline[instruction->a];
        const double complex working =
            carrier->working[instruction->a];
        carrier->baseline[instruction->a] =
            carrier->baseline[instruction->b];
        carrier->working[instruction->a] =
            carrier->working[instruction->b];
        carrier->baseline[instruction->b] = baseline;
        carrier->working[instruction->b] = working;
        return;
    }
    if (instruction->op == OP_CSWAP) {
        const double complex control = boolean_one_indicator(
            relation(carrier, instruction->target)
        );
        const double complex left = relation(carrier, instruction->a);
        const double complex right = relation(carrier, instruction->b);
        const double complex control_left =
            product_factor(control, left);
        const double complex control_right =
            product_factor(control, right);
        write_relation(
            carrier,
            instruction->a,
            left * control_right * conj(control_left)
        );
        write_relation(
            carrier,
            instruction->b,
            right * control_left * conj(control_right)
        );
        return;
    }
    if (instruction->op == OP_PCSWAP) {
        const double complex routed_control = product_factor(
            relation(carrier, instruction->target),
            relation(carrier, instruction->a)
        );
        const double complex control =
            boolean_one_indicator(routed_control);
        const double complex left = relation(carrier, instruction->b);
        const double complex right = relation(carrier, instruction->c);
        const double complex control_left =
            product_factor(control, left);
        const double complex control_right =
            product_factor(control, right);
        write_relation(
            carrier,
            instruction->b,
            left * control_right * conj(control_left)
        );
        write_relation(
            carrier,
            instruction->c,
            right * control_left * conj(control_right)
        );
        return;
    }
    fprintf(stderr, "unknown phase instruction\n");
    exit(2);
}

static void apply_inverse(
    struct carrier *carrier,
    const struct instruction *instruction,
    int wrong_program
) {
    struct instruction inverse = *instruction;
    if (inverse.op == OP_ROT) {
        write_relation_unlocked(
            carrier,
            inverse.target,
            relation(carrier, inverse.target)
                * conj(root_of_unity(inverse.amount))
        );
    } else if (inverse.op == OP_ADD) {
        write_relation(
            carrier,
            inverse.target,
            relation(carrier, inverse.target)
                * conj(relation(carrier, inverse.a))
        );
    } else if (inverse.op == OP_MULADD) {
        const double complex factor = product_factor(
            relation(carrier, inverse.a),
            relation(carrier, inverse.b)
        );
        write_relation(
            carrier,
            inverse.target,
            relation(carrier, inverse.target) * conj(factor)
        );
    } else if (inverse.op == OP_SWAP) {
        apply_forward(carrier, &inverse);
    } else if (
        inverse.op == OP_CSWAP
        || inverse.op == OP_PCSWAP
    ) {
        apply_forward(carrier, &inverse);
    } else {
        fprintf(stderr, "unknown inverse phase instruction\n");
        exit(2);
    }
    if (wrong_program) {
        const size_t target = (
            inverse.op == OP_SWAP || inverse.op == OP_CSWAP
                || inverse.op == OP_PCSWAP
        )
            ? (
                inverse.op == OP_PCSWAP
                    ? inverse.b
                    : inverse.a
            )
            : inverse.target;
        write_relation_unlocked(
            carrier,
            target,
            relation(carrier, target) * root_of_unity(1)
        );
    }
}

static struct carrier make_carrier(size_t registers, int identity) {
    struct carrier result = {
        .registers = registers,
        .baseline = calloc(registers, sizeof(double complex)),
        .working = calloc(registers, sizeof(double complex))
    };
    if (result.baseline == NULL || result.working == NULL) {
        fprintf(stderr, "carrier allocation failed\n");
        exit(2);
    }
    for (size_t index = 0; index < registers; ++index) {
        const double angle =
            0.173
            + 0.037 * (double)index
            + 0.023 * sin(0.17 * (double)index + identity * 0.07);
        const double complex rail = cexp(I * angle);
        result.baseline[index] = rail;
        result.working[index] = rail;
    }
    return result;
}

static struct carrier clone_carrier(const struct carrier *source) {
    struct carrier result = make_carrier(source->registers, 0);
    memcpy(
        result.baseline,
        source->baseline,
        source->registers * sizeof(double complex)
    );
    memcpy(
        result.working,
        source->working,
        source->registers * sizeof(double complex)
    );
    return result;
}

static void free_carrier(struct carrier *carrier) {
    free(carrier->baseline);
    free(carrier->working);
    carrier->baseline = NULL;
    carrier->working = NULL;
    carrier->registers = 0;
}

static int input_value(size_t register_index, int family) {
    return (int)(
        (
            register_index * register_index
            + (size_t)(2 + family) * register_index
            + 1U
            + (size_t)family
        )
        % Q
    );
}

static int source_input_value(
    const struct instruction_source *source, size_t register_index
) {
    if (source->input_symbols != NULL) {
        return source->input_symbols[register_index];
    }
    return input_value(register_index, source->family);
}

static void load_input(
    struct carrier *carrier, const struct instruction_source *source
) {
    for (size_t index = 0; index < carrier->registers; ++index) {
        carrier->working[index] *=
            root_of_unity(source_input_value(source, index));
    }
}

static void unload_input(
    struct carrier *carrier, const struct instruction_source *source
) {
    for (size_t index = 0; index < carrier->registers; ++index) {
        carrier->working[index] *=
            conj(root_of_unity(source_input_value(source, index)));
    }
}

static double carrier_error(
    const struct carrier *left, const struct carrier *right
) {
    double maximum = 0.0;
    for (size_t index = 0; index < left->registers; ++index) {
        const double baseline_error = cabs(
            left->baseline[index] - right->baseline[index]
        );
        const double working_error = cabs(
            left->working[index] - right->working[index]
        );
        if (baseline_error > maximum) {
            maximum = baseline_error;
        }
        if (working_error > maximum) {
            maximum = working_error;
        }
    }
    return maximum;
}

static double carrier_displacement(
    const struct carrier *left, const struct carrier *right
) {
    double sum = 0.0;
    for (size_t index = 0; index < left->registers; ++index) {
        const double baseline_error = cabs(
            left->baseline[index] - right->baseline[index]
        );
        const double working_error = cabs(
            left->working[index] - right->working[index]
        );
        sum += baseline_error * baseline_error;
        sum += working_error * working_error;
    }
    return sqrt(sum);
}

static int decode_relation(
    double complex value, double *distance
) {
    double best = INFINITY;
    int best_symbol = -1;
    for (int symbol = 0; symbol < Q; ++symbol) {
        const double candidate = cabs(unit(value) - root_of_unity(symbol));
        if (candidate < best) {
            best = candidate;
            best_symbol = symbol;
        }
    }
    *distance = best;
    return best_symbol;
}

static struct execution execute(
    struct carrier *borrowed,
    const struct instruction_source *source,
    int inverse_mode
) {
    struct carrier working = clone_carrier(borrowed);
    double complex *latch = calloc(
        working.registers, sizeof(double complex)
    );
    int *symbols = calloc(working.registers, sizeof(int));
    if (latch == NULL || symbols == NULL) {
        fprintf(stderr, "latch allocation failed\n");
        exit(2);
    }

    load_input(&working, source);
    const uint64_t forward_start = monotonic_ns();
    for (uint64_t index = 0; index < source->steps; ++index) {
        const struct instruction instruction =
            source_instruction_at(source, index);
        apply_forward(&working, &instruction);
    }
    const uint64_t forward_ns = monotonic_ns() - forward_start;
    const double displacement =
        carrier_displacement(&working, borrowed);
    for (size_t index = 0; index < working.registers; ++index) {
        latch[index] = relation(&working, index);
    }

    const uint64_t inverse_start = monotonic_ns();
    if (inverse_mode != 2) {
        for (uint64_t index = source->steps; index-- > 0;) {
            const struct instruction instruction =
                source_instruction_at(source, index);
            apply_inverse(
                &working,
                &instruction,
                inverse_mode == 1 && index == source->steps - 1U
            );
        }
        unload_input(&working, source);
    }
    const uint64_t inverse_ns = monotonic_ns() - inverse_start;

    double maximum_root_distance = 0.0;
    for (size_t index = 0; index < working.registers; ++index) {
        double distance = 0.0;
        symbols[index] = decode_relation(latch[index], &distance);
        if (distance > maximum_root_distance) {
            maximum_root_distance = distance;
        }
    }
    const double restoration = carrier_error(&working, borrowed);
    if (inverse_mode == 0) {
        memcpy(
            borrowed->baseline,
            working.baseline,
            working.registers * sizeof(double complex)
        );
        memcpy(
            borrowed->working,
            working.working,
            working.registers * sizeof(double complex)
        );
    }
    free(latch);
    free_carrier(&working);
    return (struct execution){
        .steps = source->steps,
        .family = source->family,
        .registers = borrowed->registers,
        .boundary_symbols = symbols,
        .maximum_root_distance = maximum_root_distance,
        .displacement_l2 = displacement,
        .restoration_max_abs = restoration,
        .forward_ns = forward_ns,
        .inverse_ns = inverse_ns,
        .public_program_fnv1a64 = source->public_program_fnv1a64
    };
}

static void free_execution(struct execution *execution) {
    free(execution->boundary_symbols);
    execution->boundary_symbols = NULL;
}

static void print_execution(
    const struct execution *execution,
    const char *mode
) {
    uint64_t boundary_hash = UINT64_C(14695981039346656037);
    size_t boundary_nonzero = 0;
    for (size_t index = 0; index < execution->registers; ++index) {
        const unsigned char symbol =
            (unsigned char)execution->boundary_symbols[index];
        boundary_hash = fnv1a_update(boundary_hash, &symbol, 1U);
        if (symbol != 0U) {
            ++boundary_nonzero;
        }
    }
    printf(
        "{\"mode\":\"%s\",\"steps\":%llu,\"family\":%d,"
        "\"registers\":%zu,\"resident_complex_cells\":%zu,"
        "\"history_factor_count\":0,"
        "\"public_program_fnv1a64\":\"%016llx\","
        "\"boundary_fnv1a64\":\"%016llx\","
        "\"boundary_nonzero\":%zu,\"boundary_symbols\":",
        mode,
        (unsigned long long)execution->steps,
        execution->family,
        execution->registers,
        2U * execution->registers,
        (unsigned long long)execution->public_program_fnv1a64,
        (unsigned long long)boundary_hash,
        boundary_nonzero
    );
    if (execution->registers <= 256U) {
        putchar('[');
        for (size_t index = 0; index < execution->registers; ++index) {
            printf(
                "%s%d",
                index == 0 ? "" : ",",
                execution->boundary_symbols[index]
            );
        }
        putchar(']');
    } else {
        fputs("null", stdout);
    }
    printf(
        ",\"root_distance_max\":%.12g,\"displacement_l2\":%.12g,"
        "\"restoration_max_abs\":%.12g,\"restoration_pass\":%s,"
        "\"forward_ns\":%llu,\"inverse_ns\":%llu}\n",
        execution->maximum_root_distance,
        execution->displacement_l2,
        execution->restoration_max_abs,
        execution->restoration_max_abs <= RESTORATION_MAX ? "true" : "false",
        (unsigned long long)execution->forward_ns,
        (unsigned long long)execution->inverse_ns
    );
}

static int run_public_program(const char *path) {
    struct public_program program = read_public_program(path);
    if (
        program.instruction_count > 0
        && program.passes > UINT64_MAX / program.instruction_count
    ) {
        fprintf(stderr, "public program step count overflows uint64\n");
        free_public_program(&program);
        return 2;
    }
    const struct instruction_source source = {
        .steps = program.instruction_count * program.passes,
        .family = -1,
        .registers = program.registers,
        .input_symbols = program.input_symbols,
        .instructions = program.instructions,
        .instruction_period = program.instruction_count,
        .public_program_fnv1a64 = program.fnv1a64
    };
    struct carrier carrier = make_carrier(program.registers, 89);
    struct execution first = execute(&carrier, &source, 0);
    print_execution(&first, "public-program");
    struct execution second = execute(&carrier, &source, 0);
    print_execution(&second, "actual-restored-program-reuse");
    struct execution wrong = execute(&carrier, &source, 1);
    print_execution(&wrong, "wrong-program-inverse");
    struct execution omitted = execute(&carrier, &source, 2);
    print_execution(&omitted, "omitted-inverse");

    const int inverse_controls_applicable =
        first.displacement_l2 > RESTORATION_MAX;
    printf(
        "{\"mode\":\"public-program-inverse-controls\","
        "\"applicable\":%s,"
        "\"forward_displacement_l2\":%.12g,"
        "\"reason\":\"%s\"}\n",
        inverse_controls_applicable ? "true" : "false",
        first.displacement_l2,
        inverse_controls_applicable
            ? "FORWARD_CHANGED_CARRIER"
            : "FORWARD_WAS_IDENTITY"
    );
    const int result = (
        first.restoration_max_abs <= RESTORATION_MAX
        && second.restoration_max_abs <= RESTORATION_MAX
        && (
            !inverse_controls_applicable
            || (
                wrong.restoration_max_abs > RESTORATION_MAX
                && omitted.restoration_max_abs > RESTORATION_MAX
            )
        )
    ) ? 0 : 1;
    free_execution(&first);
    free_execution(&second);
    free_execution(&wrong);
    free_execution(&omitted);
    free_carrier(&carrier);
    free_public_program(&program);
    return result;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--program") == 0) {
        return run_public_program(argv[2]);
    }
    uint64_t steps = UINT64_C(100000);
    size_t registers = 12;
    if (argc > 1) {
        steps = parse_positive_decimal(argv[1], "steps");
        if (steps > UINT64_MAX - UINT64_C(17)) {
            fprintf(stderr, "steps leave no room for reuse control\n");
            return 2;
        }
    }
    if (argc > 2) {
        const uint64_t parsed =
            parse_positive_decimal(argv[2], "register count");
        if (parsed > SIZE_MAX || parsed < 3) {
            fprintf(stderr, "register count must be at least three\n");
            return 2;
        }
        registers = (size_t)parsed;
    }

    struct carrier carrier = make_carrier(registers, 71);
    const struct instruction_source first_source = {
        .steps = steps,
        .family = 0,
        .registers = registers,
        .input_symbols = NULL,
        .instructions = NULL,
        .instruction_period = 0,
        .public_program_fnv1a64 = 0
    };
    const struct instruction_source second_source = {
        .steps = steps + 17U,
        .family = 1,
        .registers = registers,
        .input_symbols = NULL,
        .instructions = NULL,
        .instruction_period = 0,
        .public_program_fnv1a64 = 0
    };
    struct execution first = execute(&carrier, &first_source, 0);
    print_execution(&first, "nominal-family-0");
    struct execution second = execute(&carrier, &second_source, 0);
    print_execution(&second, "actual-restored-reuse-family-1");

    const uint64_t control_steps = steps < 4096U ? steps : 4096U;
    const struct instruction_source control_source = {
        .steps = control_steps,
        .family = 0,
        .registers = registers,
        .input_symbols = NULL,
        .instructions = NULL,
        .instruction_period = 0,
        .public_program_fnv1a64 = 0
    };
    struct execution wrong = execute(&carrier, &control_source, 1);
    print_execution(&wrong, "wrong-program-inverse");
    struct execution omitted = execute(&carrier, &control_source, 2);
    print_execution(&omitted, "omitted-inverse");

    free_execution(&first);
    free_execution(&second);
    free_execution(&wrong);
    free_execution(&omitted);
    free_carrier(&carrier);
    return 0;
}
