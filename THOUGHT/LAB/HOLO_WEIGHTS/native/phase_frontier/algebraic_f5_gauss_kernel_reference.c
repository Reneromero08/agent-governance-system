/*
 * Independent compact F5 Weil/Gauss recurrence for the accepted chirp class.
 * It has no resident phase kernel, coherent contraction, inverse, or reuse.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GR_FIELD 5U

static const double GR_PI =
    3.14159265358979323846264338327950288;

struct gr_signature {
    int mu;
    int q[6];
};

static int gr_mod(int value) {
    int result = value % (int)GR_FIELD;
    return result < 0 ? result + (int)GR_FIELD : result;
}

static int gr_inverse(int value) {
    const int normalized = gr_mod(value);
    for (int candidate = 1; candidate < (int)GR_FIELD; ++candidate) {
        if (gr_mod(normalized * candidate) == 1) {
            return candidate;
        }
    }
    fprintf(stderr, "noninvertible compact Gauss coefficient\n");
    exit(2);
}

static int gr_legendre(int value) {
    const int normalized = gr_mod(value);
    if (normalized == 1 || normalized == 4) {
        return 1;
    }
    if (normalized == 2 || normalized == 3) {
        return -1;
    }
    fprintf(stderr, "degenerate compact Gauss coefficient\n");
    exit(2);
}

static struct gr_signature gr_initial(void) {
    const struct gr_signature signature = {
        .mu = 1,
        .q = {1, 1, 2, 1, 0, 3}
    };
    return signature;
}

static struct gr_signature gr_module(size_t operation) {
    struct gr_signature module = {
        .mu = operation % 3U == 0U ? -1 : 1,
        .q = {
            4,
            1,
            1,
            (int)((operation + 2U) % GR_FIELD),
            (int)((2U * operation + 1U) % GR_FIELD),
            (int)((3U * operation) % GR_FIELD)
        }
    };
    return module;
}

static struct gr_signature gr_compose(
    struct gr_signature left,
    struct gr_signature right
) {
    const int shared_quadratic = gr_mod(left.q[2] + right.q[0]);
    const int h = gr_inverse(4 * shared_quadratic);
    const int l0 = gr_mod(left.q[4] + right.q[3]);
    struct gr_signature output = {
        .mu = left.mu * right.mu * gr_legendre(shared_quadratic),
        .q = {
            gr_mod(left.q[0] - h * left.q[1] * left.q[1]),
            gr_mod(-2 * h * left.q[1] * right.q[1]),
            gr_mod(right.q[2] - h * right.q[1] * right.q[1]),
            gr_mod(left.q[3] - 2 * h * left.q[1] * l0),
            gr_mod(right.q[4] - 2 * h * right.q[1] * l0),
            gr_mod(left.q[5] + right.q[5] - h * l0 * l0)
        }
    };
    if (output.q[1] == 0) {
        fprintf(stderr, "compact Gauss output lost nondegeneracy\n");
        exit(2);
    }
    return output;
}

static int gr_polynomial(
    const struct gr_signature *signature,
    int x,
    int z
) {
    return gr_mod(
        signature->q[0] * x * x
        + signature->q[1] * x * z
        + signature->q[2] * z * z
        + signature->q[3] * x
        + signature->q[4] * z
        + signature->q[5]
    );
}

static double complex gr_phase(
    const struct gr_signature *signature,
    int x,
    int z
) {
    return (double)signature->mu * cexp(
        I * 2.0 * GR_PI
        * (double)gr_polynomial(signature, x, z)
        / (double)GR_FIELD
    );
}

static uint64_t gr_boundary(
    const struct gr_signature *signature,
    double probability[GR_FIELD]
) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t z = 0U; z < GR_FIELD; ++z) {
        double complex amplitude = 0.0 + 0.0 * I;
        for (size_t x = 0U; x < GR_FIELD; ++x) {
            const int exponent = gr_polynomial(
                signature, (int)x, (int)z
            );
            const int code = signature->mu == 1
                ? exponent
                : 5 + exponent;
            hash ^= (uint64_t)code;
            hash *= UINT64_C(1099511628211);
            amplitude += gr_phase(
                signature, (int)x, (int)z
            ) / (double)GR_FIELD;
        }
        probability[z] = creal(amplitude * conj(amplitude));
    }
    return hash;
}

int main(void) {
    static const size_t depths[] = {
        2U, 4U, 8U, 32U, 128U, 512U, 2048U
    };
    uint64_t hashes[sizeof(depths) / sizeof(depths[0])] = {0};
    double deepest_probability[GR_FIELD] = {0};
    for (
        size_t depth_index = 0U;
        depth_index < sizeof(depths) / sizeof(depths[0]);
        ++depth_index
    ) {
        struct gr_signature current = gr_initial();
        for (
            size_t operation = 0U;
            operation < depths[depth_index];
            ++operation
        ) {
            current = gr_compose(current, gr_module(operation));
        }
        double probability[GR_FIELD] = {0};
        hashes[depth_index] = gr_boundary(&current, probability);
        if (depth_index + 1U == sizeof(depths) / sizeof(depths[0])) {
            for (size_t index = 0U; index < GR_FIELD; ++index) {
                deepest_probability[index] = probability[index];
            }
        }
    }
    printf(
        "{\"result\":\"PASS\",\"tested_depths\":["
    );
    for (
        size_t index = 0U;
        index < sizeof(depths) / sizeof(depths[0]);
        ++index
    ) {
        printf("%s%zu", index == 0U ? "" : ",", depths[index]);
    }
    printf("],\"final_boundary_hashes\":[");
    for (
        size_t index = 0U;
        index < sizeof(depths) / sizeof(depths[0]);
        ++index
    ) {
        printf(
            "%s\"%016llx\"",
            index == 0U ? "" : ",",
            (unsigned long long)hashes[index]
        );
    }
    printf(
        "],\"deepest_probability\":[%.17g,%.17g,%.17g,%.17g,%.17g],"
        "\"compact_signature_state_bytes\":%zu,"
        "\"retained_intermediate_signatures\":0,"
        "\"hidden_kernel_rows_materialized\":0}\n",
        deepest_probability[0],
        deepest_probability[1],
        deepest_probability[2],
        deepest_probability[3],
        deepest_probability[4],
        7U * sizeof(int)
    );
    return 0;
}
