#define _POSIX_C_SOURCE 200809L

/*
 * Matched compact reference for the public nonlinear two-angle torus map.
 * It has no carrier, borrowed baseline, restoration, or catalytic reuse.
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PR_CELLS 2U
#define PR_MIN_DEPTH 3U
#define PR_MAX_DEPTH 4096U
#define PR_HASH_SCALE 1000000000.0

enum pr_program {
    PR_PRIMARY = 0,
    PR_REUSE = 1
};

static void pr_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static size_t pr_parse_depth(const char *text) {
    if (text == NULL || text[0] < '1' || text[0] > '9') {
        pr_fail("depth must be canonical decimal from 3 through 4096");
    }
    for (size_t index = 1U; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            pr_fail("depth must be canonical decimal from 3 through 4096");
        }
    }
    errno = 0;
    char *tail = NULL;
    const unsigned long parsed = strtoul(text, &tail, 10);
    if (
        errno != 0
        || tail == text
        || *tail != '\0'
        || parsed < PR_MIN_DEPTH
        || parsed > PR_MAX_DEPTH
    ) {
        pr_fail("depth must be canonical decimal from 3 through 4096");
    }
    return (size_t)parsed;
}

static double pr_initial_angle(
    enum pr_program program,
    size_t cell
) {
    if (program == PR_PRIMARY) {
        return cell == 0U ? 0.37 : -0.82;
    }
    return cell == 0U ? -1.11 : 0.23;
}

static double pr_strength(
    enum pr_program program,
    size_t index
) {
    const size_t offset = program == PR_PRIMARY ? 1U : 4U;
    const size_t residue = (5U * index + offset) % 9U;
    const double magnitude = 0.015 * (double)(residue + 1U);
    return ((index / 2U) % 2U) == 0U
        ? magnitude
        : -magnitude;
}

static void pr_execute(
    size_t depth,
    enum pr_program program,
    double angle[PR_CELLS]
) {
    for (size_t cell = 0U; cell < PR_CELLS; ++cell) {
        angle[cell] = pr_initial_angle(program, cell);
    }
    for (size_t index = 0U; index < depth; ++index) {
        const size_t target = index % PR_CELLS;
        const size_t source = 1U - target;
        angle[target] +=
            pr_strength(program, index) * sin(angle[source]);
    }
}

static uint64_t pr_hash_bytes(
    uint64_t hash,
    const unsigned char *bytes,
    size_t count
) {
    for (size_t index = 0U; index < count; ++index) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t pr_hash(
    const double angle[PR_CELLS]
) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t cell = 0U; cell < PR_CELLS; ++cell) {
        const int64_t real = (int64_t)llround(
            cos(angle[cell]) * PR_HASH_SCALE
        );
        const int64_t imaginary = (int64_t)llround(
            sin(angle[cell]) * PR_HASH_SCALE
        );
        hash = pr_hash_bytes(
            hash,
            (const unsigned char *)&real,
            sizeof(real)
        );
        hash = pr_hash_bytes(
            hash,
            (const unsigned char *)&imaginary,
            sizeof(imaginary)
        );
    }
    return hash;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        pr_fail(
            "usage: algebraic_nonlinear_phase_shear_reference "
            "<depth 3..4096>"
        );
    }
    const size_t depth = pr_parse_depth(argv[1]);
    double primary[PR_CELLS];
    double reuse[PR_CELLS];
    pr_execute(depth, PR_PRIMARY, primary);
    pr_execute(depth, PR_REUSE, reuse);
    const double probability = 0.5 * (
        1.0 + cos(primary[0] - primary[1])
    );
    printf(
        "{\"result\":\"PASS\","
        "\"depth\":%zu,"
        "\"classical_state_bytes\":%zu,"
        "\"compiled_shear_list_bytes\":0,"
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"primary_interference_probability\":%.17g,"
        "\"phase_path_cells\":0}\n",
        depth,
        2U * sizeof(double),
        (unsigned long long)pr_hash(primary),
        (unsigned long long)pr_hash(reuse),
        probability
    );
    return 0;
}
