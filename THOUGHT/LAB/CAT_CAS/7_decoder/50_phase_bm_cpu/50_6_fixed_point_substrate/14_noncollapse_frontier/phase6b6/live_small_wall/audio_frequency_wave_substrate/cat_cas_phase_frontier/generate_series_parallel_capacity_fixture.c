/*
 * Deterministic fixture generator for the bounded public series/parallel
 * relation language.  One through fifteen parallel diamonds are supported.
 * Fifteen diamonds consume 46 nodes, 60 input relations, and 44 eliminations.
 * Pattern zero is the decisive shared-interface discriminator.  Positive
 * pattern numbers deterministically populate every relation with arbitrary
 * F3 coefficients for compiler/reference surveys.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_DIAMONDS 15

static int parse_argument(const char *text, int minimum, int maximum) {
    char *end = NULL;
    const long value = strtol(text, &end, 10);
    if (
        text[0] == '\0'
        || end == NULL
        || *end != '\0'
        || value < minimum
        || value > maximum
    ) {
        fprintf(stderr, "invalid generator argument: %s\n", text);
        exit(2);
    }
    return (int)value;
}

static void anchor_name(int anchor, int diamonds, char output[16]) {
    if (anchor == 0) {
        (void)snprintf(output, 16U, "A");
    } else if (anchor == diamonds) {
        (void)snprintf(output, 16U, "B");
    } else {
        (void)snprintf(output, 16U, "P%d", anchor);
    }
}

static int generated_coefficient(
    int pattern,
    int relation,
    int coefficient
) {
    uint32_t value =
        (uint32_t)pattern * UINT32_C(0x9e3779b9)
        + (uint32_t)(relation + 1) * UINT32_C(0x85ebca6b)
        + (uint32_t)(coefficient + 1) * UINT32_C(0xc2b2ae35);
    value ^= value >> 16U;
    value *= UINT32_C(0x7feb352d);
    value ^= value >> 15U;
    return (int)(value % UINT32_C(3));
}

static void print_relation(
    int relation,
    const char *first,
    const char *second,
    int pattern,
    const int special[4]
) {
    int coefficient[4];
    for (int index = 0; index < 4; ++index) {
        coefficient[index] = pattern == 0
            ? special[index]
            : generated_coefficient(pattern, relation, index);
    }
    printf(
        "RELATION R%d %s %s %d %d %d %d\n",
        relation,
        first,
        second,
        coefficient[0],
        coefficient[1],
        coefficient[2],
        coefficient[3]
    );
}

int main(int argc, char **argv) {
    if (argc > 3) {
        fprintf(stderr, "usage: %s [DIAMONDS [PATTERN]]\n", argv[0]);
        return 2;
    }
    const int diamonds = argc >= 2
        ? parse_argument(argv[1], 1, MAX_DIAMONDS)
        : MAX_DIAMONDS;
    const int pattern = argc == 3
        ? parse_argument(argv[2], 0, 1000000)
        : 0;
    puts("CATCAS_SERIES_PARALLEL_RELATION 1");
    puts("TYPE BOOLEAN_F3");
    puts("NODE A EXTERNAL");
    puts("NODE B EXTERNAL");
    for (int index = 1; index < diamonds; ++index) {
        printf("NODE P%d INTERNAL\n", index);
    }
    for (int index = 0; index < diamonds; ++index) {
        printf("NODE W%d INTERNAL\n", index);
        printf("NODE Z%d INTERNAL\n", index);
    }
    int relation = 0;
    const int equals[4] = {0, 1, 2, 0};
    const int x_zero[4] = {0, 1, 0, 0};
    const int y_zero[4] = {0, 0, 1, 0};
    for (int index = 0; index < diamonds; ++index) {
        char left[16];
        char right[16];
        char w[16];
        char z[16];
        anchor_name(index, diamonds, left);
        anchor_name(index + 1, diamonds, right);
        (void)snprintf(w, sizeof(w), "W%d", index);
        (void)snprintf(z, sizeof(z), "Z%d", index);
        if (index == 0) {
            print_relation(relation++, left, w, pattern, x_zero);
            print_relation(relation++, w, right, pattern, equals);
            print_relation(relation++, left, z, pattern, y_zero);
            print_relation(relation++, z, right, pattern, equals);
        } else {
            print_relation(relation++, left, w, pattern, equals);
            print_relation(relation++, w, right, pattern, equals);
            print_relation(relation++, left, z, pattern, equals);
            print_relation(relation++, z, right, pattern, equals);
        }
    }
    for (int index = 0; index < diamonds; ++index) {
        printf("ELIMINATE W%d\n", index);
        printf("ELIMINATE Z%d\n", index);
    }
    for (int index = 1; index < diamonds; ++index) {
        printf("ELIMINATE P%d\n", index);
    }
    puts("END");
    return 0;
}
