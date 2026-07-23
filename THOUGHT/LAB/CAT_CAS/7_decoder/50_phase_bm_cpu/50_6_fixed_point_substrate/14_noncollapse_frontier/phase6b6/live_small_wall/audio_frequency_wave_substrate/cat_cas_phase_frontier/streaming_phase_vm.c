/*
 * Mutable CAT_CAS frontier: streaming, program-derived-inverse phase VM.
 *
 * The native state is the relative phase of twin complex rails. The VM uses a
 * ternary root alphabet only at its public load and collapse boundaries.
 * During execution ROT, ADD, MULADD, and SWAP operate on complex relations.
 * MULADD is a fixed roots-of-unity interpolation network, not a decoded branch.
 *
 * This source contains no conventional evaluator. A separate verifier may
 * adjudicate the emitted boundary after execution.
 */

#define _GNU_SOURCE
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Q 3
#define PI 3.141592653589793238462643383279502884
#define RESTORATION_MAX 2.0e-11
#define LOCK_ERROR_TRIGGER 1.0e-12

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
    const double complex left_squared = left * left;
    const double complex right_squared = right * right;
    const double complex product = left * right;
    const double complex both_squared = product * product;
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
    fprintf(stderr, "unknown phase instruction\n");
    exit(2);
}

static void apply_inverse(
    struct carrier *carrier,
    const struct instruction *instruction,
    int wrong_program
) {
    struct instruction inverse = *instruction;
    if (wrong_program && inverse.op == OP_ROT) {
        inverse.amount += 1;
    }
    if (inverse.op == OP_ROT) {
        write_relation_unlocked(
            carrier,
            inverse.target,
            relation(carrier, inverse.target)
                * conj(root_of_unity(inverse.amount))
        );
        return;
    }
    if (inverse.op == OP_ADD) {
        write_relation(
            carrier,
            inverse.target,
            relation(carrier, inverse.target)
                * conj(relation(carrier, inverse.a))
        );
        return;
    }
    if (inverse.op == OP_MULADD) {
        const double complex factor = product_factor(
            relation(carrier, inverse.a),
            relation(carrier, inverse.b)
        );
        write_relation(
            carrier,
            inverse.target,
            relation(carrier, inverse.target) * conj(factor)
        );
        return;
    }
    if (inverse.op == OP_SWAP) {
        apply_forward(carrier, &inverse);
        return;
    }
    fprintf(stderr, "unknown inverse phase instruction\n");
    exit(2);
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

static void load_input(struct carrier *carrier, int family) {
    for (size_t index = 0; index < carrier->registers; ++index) {
        carrier->working[index] *=
            root_of_unity(input_value(index, family));
    }
}

static void unload_input(struct carrier *carrier, int family) {
    for (size_t index = 0; index < carrier->registers; ++index) {
        carrier->working[index] *=
            conj(root_of_unity(input_value(index, family)));
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
    uint64_t steps,
    int family,
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

    load_input(&working, family);
    const uint64_t forward_start = monotonic_ns();
    for (uint64_t index = 0; index < steps; ++index) {
        const struct instruction instruction =
            instruction_at(index, family, working.registers);
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
        for (uint64_t index = steps; index-- > 0;) {
            const struct instruction instruction =
                instruction_at(index, family, working.registers);
            apply_inverse(
                &working,
                &instruction,
                inverse_mode == 1
            );
        }
        unload_input(&working, family);
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
        .steps = steps,
        .family = family,
        .registers = borrowed->registers,
        .boundary_symbols = symbols,
        .maximum_root_distance = maximum_root_distance,
        .displacement_l2 = displacement,
        .restoration_max_abs = restoration,
        .forward_ns = forward_ns,
        .inverse_ns = inverse_ns
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
    printf(
        "{\"mode\":\"%s\",\"steps\":%llu,\"family\":%d,"
        "\"registers\":%zu,\"resident_complex_cells\":%zu,"
        "\"history_factor_count\":0,\"boundary_symbols\":[",
        mode,
        (unsigned long long)execution->steps,
        execution->family,
        execution->registers,
        2U * execution->registers
    );
    for (size_t index = 0; index < execution->registers; ++index) {
        printf(
            "%s%d",
            index == 0 ? "" : ",",
            execution->boundary_symbols[index]
        );
    }
    printf(
        "],\"root_distance_max\":%.12g,\"displacement_l2\":%.12g,"
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

int main(int argc, char **argv) {
    uint64_t steps = UINT64_C(100000);
    size_t registers = 12;
    if (argc > 1) {
        char *end = NULL;
        steps = (uint64_t)strtoull(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || steps == 0) {
            fprintf(stderr, "steps must be a positive integer\n");
            return 2;
        }
    }
    if (argc > 2) {
        char *end = NULL;
        registers = (size_t)strtoull(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || registers < 3) {
            fprintf(stderr, "register count must be at least three\n");
            return 2;
        }
    }

    struct carrier carrier = make_carrier(registers, 71);
    struct execution first = execute(&carrier, steps, 0, 0);
    print_execution(&first, "nominal-family-0");
    struct execution second = execute(&carrier, steps + 17U, 1, 0);
    print_execution(&second, "actual-restored-reuse-family-1");

    const uint64_t control_steps = steps < 4096U ? steps : 4096U;
    struct execution wrong = execute(&carrier, control_steps, 0, 1);
    print_execution(&wrong, "wrong-program-inverse");
    struct execution omitted = execute(&carrier, control_steps, 0, 2);
    print_execution(&omitted, "omitted-inverse");

    free_execution(&first);
    free_execution(&second);
    free_execution(&wrong);
    free_execution(&omitted);
    free_carrier(&carrier);
    return 0;
}
