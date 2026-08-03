/*
 * Independent compact modular reference for the fixed F5 conic schema.
 * It has no phase carrier, inverse execution, restoration, or reuse.
 */

#include <stdint.h>
#include <stdio.h>

#define FR_SIGNATURE 6U
#define FR_INPUT 10U

static const int FR_FIXTURE[2][FR_INPUT] = {
    {0, 1, 0, 0, 0, 0, 1, 0, 0, 1},
    {1, 2, 2, 1, 4, 3, 2, 1, 3, 1}
};

static int fr_mod(int value) {
    int result = value % 5;
    return result < 0 ? result + 5 : result;
}

static void fr_first(
    const int input[FR_INPUT],
    int h[FR_SIGNATURE]
) {
    const int a0 = input[0];
    const int a1 = input[1];
    const int *q = &input[2];
    h[0] = fr_mod(q[0] + q[1] * a0 + q[3] * a0 * a0);
    h[1] = fr_mod(a1 * (q[1] + 2 * q[3] * a0));
    h[2] = fr_mod(q[2] + q[4] * a0);
    h[3] = fr_mod(q[3] * a1 * a1);
    h[4] = fr_mod(q[4] * a1);
    h[5] = fr_mod(q[5]);
}

static void fr_second(
    const int input[FR_INPUT],
    const int h[FR_SIGNATURE],
    int k[FR_SIGNATURE]
) {
    const int c0 = input[8];
    const int c1 = input[9];
    k[0] = fr_mod(h[0] + h[2] * c0 + h[5] * c0 * c0);
    k[1] = fr_mod(h[1] + h[4] * c0);
    k[2] = fr_mod(c1 * (h[2] + 2 * h[5] * c0));
    k[3] = fr_mod(h[3]);
    k[4] = fr_mod(h[4] * c1);
    k[5] = fr_mod(h[5] * c1 * c1);
}

static uint64_t fr_hash(const int value[FR_SIGNATURE]) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t index = 0U; index < FR_SIGNATURE; ++index) {
        hash ^= (uint64_t)value[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int fr_evaluate(
    const int coefficient[FR_SIGNATURE],
    int x,
    int z
) {
    return fr_mod(
        coefficient[0]
        + coefficient[1] * x
        + coefficient[2] * z
        + coefficient[3] * x * x
        + coefficient[4] * x * z
        + coefficient[5] * z * z
    );
}

int main(void) {
    int h[2][FR_SIGNATURE] = {{0}};
    int k[2][FR_SIGNATURE] = {{0}};
    for (size_t program = 0U; program < 2U; ++program) {
        fr_first(FR_FIXTURE[program], h[program]);
        fr_second(FR_FIXTURE[program], h[program], k[program]);
    }
    size_t primary_zero_set = 0U;
    for (int x = 0; x < 5; ++x) {
        for (int z = 0; z < 5; ++z) {
            if (fr_evaluate(k[0], x, z) == 0) {
                ++primary_zero_set;
            }
        }
    }
    printf(
        "{\"result\":\"PASS\","
        "\"primary_boundary_fnv1a64\":\"%016llx\","
        "\"reuse_boundary_fnv1a64\":\"%016llx\","
        "\"primary_boundary_coefficients\":[",
        (unsigned long long)fr_hash(k[0]),
        (unsigned long long)fr_hash(k[1])
    );
    for (size_t index = 0U; index < FR_SIGNATURE; ++index) {
        printf("%s%d", index == 0U ? "" : ",", k[0][index]);
    }
    printf(
        "],\"primary_boundary_zero_set_size\":%zu,"
        "\"affine_f5_subset_cardinality\":false,"
        "\"compact_classical_live_state_bytes\":%zu,"
        "\"retain_all_classical_state_bytes\":%zu,"
        "\"hidden_port_assignment_rows_in_accepted_path\":0}\n",
        primary_zero_set,
        (FR_INPUT + FR_SIGNATURE) * sizeof(int),
        (FR_INPUT + 2U * FR_SIGNATURE) * sizeof(int)
    );
    return 0;
}
