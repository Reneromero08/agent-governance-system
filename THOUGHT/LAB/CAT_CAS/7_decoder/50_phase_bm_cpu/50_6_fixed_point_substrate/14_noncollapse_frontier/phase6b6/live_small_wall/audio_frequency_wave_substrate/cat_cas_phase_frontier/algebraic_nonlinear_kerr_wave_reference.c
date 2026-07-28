#define _POSIX_C_SOURCE 200809L

/*
 * Independent scalar real/imaginary recurrence for the nonlinear Kerr-wave
 * fixture.  This file intentionally does not use C complex arithmetic or the
 * production carrier operations.
 */

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define KR_CELLS 4U
#define KR_HASH_SCALE 1.0e12

struct kr_value {
    double real;
    double imaginary;
};

static double kr_abs_squared(struct kr_value value) {
    return value.real * value.real + value.imaginary * value.imaginary;
}

static struct kr_value kr_multiply(
    struct kr_value first,
    struct kr_value second
) {
    const struct kr_value result = {
        .real = first.real * second.real
            - first.imaginary * second.imaginary,
        .imaginary = first.real * second.imaginary
            + first.imaginary * second.real
    };
    return result;
}

static struct kr_value kr_phase(double angle) {
    const struct kr_value result = {
        .real = cos(angle),
        .imaginary = sin(angle)
    };
    return result;
}

static void kr_initialize(struct kr_value state[KR_CELLS]) {
    const struct kr_value seed[KR_CELLS] = {
        {0.51, 0.13},
        {-0.21, 0.47},
        {0.34, -0.29},
        {-0.18, -0.44}
    };
    const double twist = 0.003 * 11.0;
    double norm = 0.0;
    for (size_t cell = 0U; cell < KR_CELLS; ++cell) {
        state[cell] = kr_multiply(
            seed[cell],
            kr_phase(twist * (double)(cell + 1U))
        );
        norm += kr_abs_squared(state[cell]);
    }
    const double scale = 1.0 / sqrt(norm);
    for (size_t cell = 0U; cell < KR_CELLS; ++cell) {
        state[cell].real *= scale;
        state[cell].imaginary *= scale;
    }
}

static double kr_kappa(size_t layer) {
    return 0.113 + 0.009 * (double)((layer + 9U) % 17U);
}

static double kr_theta(size_t layer, size_t pair) {
    return 0.149
        + 0.007 * (double)((5U * layer + 6U + pair) % 19U);
}

static double kr_phi(size_t layer, size_t pair) {
    return 0.071
        + 0.013 * (double)((layer + 21U + 3U * pair) % 23U);
}

static void kr_kerr(struct kr_value state[KR_CELLS], size_t layer) {
    for (size_t cell = 0U; cell < KR_CELLS; ++cell) {
        state[cell] = kr_multiply(
            state[cell],
            kr_phase(kr_kappa(layer) * kr_abs_squared(state[cell]))
        );
    }
}

static void kr_couple(
    struct kr_value state[KR_CELLS],
    size_t first,
    size_t second,
    size_t layer,
    size_t pair
) {
    const double cosine = cos(kr_theta(layer, pair));
    const double sine = sin(kr_theta(layer, pair));
    const struct kr_value phase = kr_phase(kr_phi(layer, pair));
    const struct kr_value conjugate_phase = {
        phase.real,
        -phase.imaginary
    };
    const struct kr_value a = state[first];
    const struct kr_value b = state[second];
    const struct kr_value phase_b = kr_multiply(phase, b);
    const struct kr_value conjugate_phase_a = kr_multiply(
        conjugate_phase,
        a
    );
    state[first].real = cosine * a.real + sine * phase_b.real;
    state[first].imaginary =
        cosine * a.imaginary + sine * phase_b.imaginary;
    state[second].real =
        -sine * conjugate_phase_a.real + cosine * b.real;
    state[second].imaginary =
        -sine * conjugate_phase_a.imaginary + cosine * b.imaginary;
}

static uint64_t kr_hash_mix(uint64_t hash, uint64_t value) {
    for (size_t byte = 0U; byte < sizeof(value); ++byte) {
        hash ^= (value >> (8U * byte)) & UINT64_C(0xff);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t kr_run(size_t depth) {
    struct kr_value state[KR_CELLS] = {0};
    kr_initialize(state);
    for (size_t layer = 0U; layer < depth; ++layer) {
        kr_kerr(state, layer);
        if (layer % 2U == 0U) {
            kr_couple(state, 0U, 1U, layer, 0U);
            kr_couple(state, 2U, 3U, layer, 1U);
        } else {
            kr_couple(state, 1U, 2U, layer, 0U);
            kr_couple(state, 3U, 0U, layer, 1U);
        }
    }
    const struct kr_value fringe = {
        state[0].real + state[1].real,
        state[0].imaginary + state[1].imaginary
    };
    const double boundary[3] = {
        kr_abs_squared(state[0]),
        kr_abs_squared(state[2]),
        0.5 * kr_abs_squared(fringe)
    };
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0U; index < 3U; ++index) {
        hash = kr_hash_mix(
            hash,
            (uint64_t)llround(boundary[index] * KR_HASH_SCALE)
        );
    }
    return hash;
}

int main(void) {
    const size_t depths[] = {1U, 4U, 32U, 128U, 512U};
    printf(
        "{\"result\":\"PASS\",\"boundary_hashes\":["
        "\"%016" PRIx64 "\",\"%016" PRIx64 "\",\"%016" PRIx64 "\","
        "\"%016" PRIx64 "\",\"%016" PRIx64 "\"],"
        "\"reference_parity_depth_max\":512,"
        "\"state_model\":\"FOUR_COMPLEX_VALUES_AS_EIGHT_SCALAR_DOUBLES\","
        "\"state_bytes\":64,"
        "\"time\":\"O_DEPTH\"}\n",
        kr_run(depths[0]),
        kr_run(depths[1]),
        kr_run(depths[2]),
        kr_run(depths[3]),
        kr_run(depths[4])
    );
    return 0;
}
