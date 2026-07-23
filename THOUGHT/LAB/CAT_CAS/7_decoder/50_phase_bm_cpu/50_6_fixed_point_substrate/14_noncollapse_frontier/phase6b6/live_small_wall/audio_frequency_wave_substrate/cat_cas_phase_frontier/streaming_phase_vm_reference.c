/*
 * Independent conventional evaluator for streaming_phase_vm.c.
 *
 * This file contains no complex carrier code and is never linked into the
 * phase VM. It exists to adjudicate boundary symbols and measure the strongest
 * straightforward compact gate-stream baseline after native execution.
 */

#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define Q 3

enum opcode {
    OP_ROT = 1,
    OP_ADD = 2,
    OP_MULADD = 3,
    OP_SWAP = 4
};

struct instruction {
    enum opcode op;
    size_t a;
    size_t b;
    size_t target;
    int amount;
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

static void execute(
    uint64_t steps, int family, size_t registers
) {
    uint8_t *state = calloc(registers, sizeof(uint8_t));
    if (state == NULL) {
        fprintf(stderr, "state allocation failed\n");
        exit(2);
    }
    for (size_t index = 0; index < registers; ++index) {
        state[index] = (uint8_t)input_value(index, family);
    }
    const uint64_t start = monotonic_ns();
    for (uint64_t index = 0; index < steps; ++index) {
        const struct instruction instruction =
            instruction_at(index, family, registers);
        if (instruction.op == OP_ROT) {
            state[instruction.target] = (uint8_t)(
                (state[instruction.target] + instruction.amount) % Q
            );
        } else if (instruction.op == OP_ADD) {
            state[instruction.target] = (uint8_t)(
                (state[instruction.target] + state[instruction.a]) % Q
            );
        } else if (instruction.op == OP_MULADD) {
            state[instruction.target] = (uint8_t)(
                (
                    state[instruction.target]
                    + state[instruction.a] * state[instruction.b]
                )
                % Q
            );
        } else if (instruction.op == OP_SWAP) {
            const uint8_t temporary = state[instruction.a];
            state[instruction.a] = state[instruction.b];
            state[instruction.b] = temporary;
        } else {
            abort();
        }
    }
    const uint64_t elapsed = monotonic_ns() - start;
    printf(
        "{\"steps\":%llu,\"family\":%d,\"registers\":%zu,"
        "\"boundary_symbols\":[",
        (unsigned long long)steps,
        family,
        registers
    );
    for (size_t index = 0; index < registers; ++index) {
        printf("%s%u", index == 0 ? "" : ",", state[index]);
    }
    printf(
        "],\"compact_forward_ns\":%llu}\n",
        (unsigned long long)elapsed
    );
    free(state);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s STEPS REGISTERS\n", argv[0]);
        return 2;
    }
    char *steps_end = NULL;
    char *registers_end = NULL;
    const uint64_t steps = (uint64_t)strtoull(
        argv[1], &steps_end, 10
    );
    const size_t registers = (size_t)strtoull(
        argv[2], &registers_end, 10
    );
    if (
        steps_end == argv[1]
        || *steps_end != '\0'
        || registers_end == argv[2]
        || *registers_end != '\0'
        || steps == 0
        || registers < 3
    ) {
        fprintf(stderr, "invalid positive steps or register count\n");
        return 2;
    }
    execute(steps, 0, registers);
    execute(steps + 17U, 1, registers);
    return 0;
}
