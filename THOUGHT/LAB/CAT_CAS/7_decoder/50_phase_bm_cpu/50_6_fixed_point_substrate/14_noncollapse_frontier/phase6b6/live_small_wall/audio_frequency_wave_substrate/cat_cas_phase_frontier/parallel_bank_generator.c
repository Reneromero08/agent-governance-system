/*
 * Deterministic public-program generator for the wide parallel PCSWAP bank.
 *
 * This emits input and topology only. It does not execute or inspect a result.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s WIDTH PASSES\n", argv[0]);
        return 2;
    }
    const uint64_t width_u64 =
        parse_positive_decimal(argv[1], "width");
    const uint64_t passes =
        parse_positive_decimal(argv[2], "passes");
    if (width_u64 > SIZE_MAX / 4U) {
        fprintf(stderr, "program dimensions overflow size_t\n");
        return 2;
    }
    const size_t width = (size_t)width_u64;
    const size_t registers = 4U * width;

    printf("CATCAS_PHASE_PROGRAM 1\n");
    printf("REGISTERS %zu\n", registers);
    for (size_t gate = 0; gate < width; ++gate) {
        const size_t data = width + 3U * gate;
        printf("SET %zu 1\n", gate);
        printf("SET %zu 1\n", data);
        printf("SET %zu 2\n", data + 2U);
    }
    printf("PASSES %llu\n", (unsigned long long)passes);
    for (size_t gate = 0; gate < width; ++gate) {
        const size_t data = width + 3U * gate;
        printf(
            "PCSWAP %zu %zu %zu %zu\n",
            gate,
            data,
            data + 1U,
            data + 2U
        );
    }
    printf("END\n");
    if (ferror(stdout)) {
        fprintf(stderr, "program output failed\n");
        return 2;
    }
    return 0;
}
