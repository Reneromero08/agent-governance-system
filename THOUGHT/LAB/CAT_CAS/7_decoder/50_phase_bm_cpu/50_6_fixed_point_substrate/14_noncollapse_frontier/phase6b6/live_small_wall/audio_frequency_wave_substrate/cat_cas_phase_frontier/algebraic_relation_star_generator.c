/*
 * Deterministic public generator for scalable branching relation stars.
 *
 * It emits only typed public geometry and branch coefficients.  It performs
 * no phase execution, hub elimination, boundary construction, or oracle work.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct signature {
    int coefficient[4];
};

static const struct signature SIGNATURES[] = {
    {{0, 1, 0, 2}}, /* LEQ */
    {{0, 1, 2, 0}}, /* EQ */
    {{2, 1, 1, 0}}, /* NEQ */
    {{0, 0, 1, 2}}  /* GEQ */
};

static int mod3(int value) {
    int result = value % 3;
    return result < 0 ? result + 3 : result;
}

static int affine_has_boolean_zero(int slope, int offset) {
    return offset == 0 || mod3(slope + offset) == 0;
}

static int bi_total(const int coefficient[4]) {
    return (
        affine_has_boolean_zero(coefficient[2], coefficient[0])
        && affine_has_boolean_zero(
            mod3(coefficient[2] + coefficient[3]),
            mod3(coefficient[0] + coefficient[1])
        )
        && affine_has_boolean_zero(coefficient[1], coefficient[0])
        && affine_has_boolean_zero(
            mod3(coefficient[1] + coefficient[3]),
            mod3(coefficient[0] + coefficient[2])
        )
    );
}

static void decode_signature(int encoded, int coefficient[4]) {
    for (int index = 0; index < 4; ++index) {
        coefficient[index] = encoded % 3;
        encoded /= 3;
    }
}

static size_t parse_count(const char *text) {
    if (text[0] < '1' || text[0] > '9') {
        fprintf(stderr, "count must be a canonical positive decimal\n");
        exit(2);
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            fprintf(stderr, "count must be a canonical positive decimal\n");
            exit(2);
        }
    }
    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (
        errno != 0
        || end == text
        || *end != '\0'
        || parsed > (unsigned long long)SIZE_MAX
        || parsed < 3ULL
    ) {
        fprintf(stderr, "count is outside the accepted range\n");
        exit(2);
    }
    return (size_t)parsed;
}

static unsigned int parse_variant(const char *text) {
    if (
        text[0] < '0'
        || text[0] > '3'
        || text[1] != '\0'
    ) {
        fprintf(stderr, "variant must be canonical 0, 1, 2, or 3\n");
        exit(2);
    }
    return (unsigned int)(text[0] - '0');
}

static unsigned int parse_ordinal(const char *text) {
    if (
        text[0] < '0'
        || text[0] > '9'
        || (text[0] == '0' && text[1] != '\0')
    ) {
        fprintf(stderr, "ordinal must be canonical 0 through 16\n");
        exit(2);
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            fprintf(stderr, "ordinal must be canonical 0 through 16\n");
            exit(2);
        }
    }
    errno = 0;
    char *end = NULL;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (
        errno != 0
        || end == text
        || *end != '\0'
        || parsed > 16UL
    ) {
        fprintf(stderr, "ordinal must be canonical 0 through 16\n");
        exit(2);
    }
    return (unsigned int)parsed;
}

static struct signature signature_by_ordinal(unsigned int ordinal) {
    unsigned int seen = 0U;
    for (int encoded = 0; encoded < 81; ++encoded) {
        struct signature signature;
        decode_signature(encoded, signature.coefficient);
        if (bi_total(signature.coefficient)) {
            if (seen == ordinal) {
                return signature;
            }
            ++seen;
        }
    }
    fprintf(stderr, "internal bi-total ordinal error\n");
    exit(2);
}

static void emit_relation(
    size_t index,
    const int coefficient[4]
) {
    printf(
        "RELATION R%zu X%zu H %d %d %d %d ZEROSET\n",
        index,
        index,
        coefficient[0],
        coefficient[1],
        coefficient[2],
        coefficient[3]
    );
}

int main(int argc, char **argv) {
    if (argc == 5 && strcmp(argv[1], "TRIPLE") == 0) {
        struct signature signature[3];
        for (size_t index = 0U; index < 3U; ++index) {
            signature[index] = signature_by_ordinal(
                parse_ordinal(argv[index + 2U])
            );
        }
        printf(
            "CATCAS_ALGEBRAIC_RELATION_STAR 1\n"
            "TYPE BOOLEAN_F3\n"
            "# exhaustive admitted triple\n"
            "HUB H\n"
        );
        for (size_t index = 0U; index < 3U; ++index) {
            emit_relation(index, signature[index].coefficient);
        }
        printf("END\n");
        return ferror(stdout) ? 2 : 0;
    }
    if (argc != 3) {
        fprintf(
            stderr,
            "usage: %s BRANCH_COUNT VARIANT | TRIPLE A B C\n",
            argv[0]
        );
        return 2;
    }
    const size_t count = parse_count(argv[1]);
    const unsigned int variant = parse_variant(argv[2]);
    printf(
        "CATCAS_ALGEBRAIC_RELATION_STAR 1\n"
        "TYPE BOOLEAN_F3\n"
        "# deterministic public star variant %u\n"
        "HUB H\n",
        variant
    );
    for (size_t index = 0U; index < count; ++index) {
        const size_t signature_index =
            (index + (size_t)variant + index / 11U)
            % (sizeof(SIGNATURES) / sizeof(SIGNATURES[0]));
        const int *coefficient =
            SIGNATURES[signature_index].coefficient;
        emit_relation(index, coefficient);
    }
    printf("END\n");
    if (ferror(stdout)) {
        fprintf(stderr, "failed to write complete star\n");
        return 2;
    }
    return 0;
}
