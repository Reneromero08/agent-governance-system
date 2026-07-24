/*
 * Complete extensional survey for the three-hub discriminator topology:
 *
 *     A--U--V--W--D
 *     B--'  |   '--E
 *           C
 *
 * Every edge ranges over all seven bi-total Boolean binary relations.
 * The survey compares conjunction of all ten leaf-to-leaf path projections
 * with exact existential closure of U,V,W for every external assignment.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define EDGE_COUNT 7U
#define RELATION_COUNT 7U
#define PAIR_COUNT 10U
#define EXTERNAL_COUNT 5U
#define INTERNAL_COUNT 3U
#define COMPLETE_ASSIGNMENTS 256U

struct path {
    size_t length;
    size_t edge[4];
    int reverse[4];
    size_t first_external;
    size_t second_external;
};

static const struct path paths[PAIR_COUNT] = {
    {2U, {0U, 1U}, {0, 1}, 0U, 1U},
    {3U, {0U, 2U, 3U}, {0, 0, 1}, 0U, 2U},
    {4U, {0U, 2U, 4U, 5U}, {0, 0, 0, 1}, 0U, 3U},
    {4U, {0U, 2U, 4U, 6U}, {0, 0, 0, 1}, 0U, 4U},
    {3U, {1U, 2U, 3U}, {0, 0, 1}, 1U, 2U},
    {4U, {1U, 2U, 4U, 5U}, {0, 0, 0, 1}, 1U, 3U},
    {4U, {1U, 2U, 4U, 6U}, {0, 0, 0, 1}, 1U, 4U},
    {3U, {3U, 4U, 5U}, {0, 0, 1}, 2U, 3U},
    {3U, {3U, 4U, 6U}, {0, 0, 1}, 2U, 4U},
    {2U, {5U, 6U}, {0, 1}, 3U, 4U}
};

static int mask_has(uint8_t mask, int first, int second) {
    return (mask & (uint8_t)(1U << (2 * first + second))) != 0U;
}

static uint8_t transpose_mask(uint8_t mask) {
    uint8_t result = 0U;
    for (int first = 0; first < 2; ++first) {
        for (int second = 0; second < 2; ++second) {
            if (mask_has(mask, first, second)) {
                result |= (uint8_t)(1U << (2 * second + first));
            }
        }
    }
    return result;
}

static uint8_t compose(uint8_t left, uint8_t right) {
    uint8_t result = 0U;
    for (int first = 0; first < 2; ++first) {
        for (int last = 0; last < 2; ++last) {
            int found = 0;
            for (int shared = 0; shared < 2; ++shared) {
                found |= (
                    mask_has(left, first, shared)
                    && mask_has(right, shared, last)
                );
            }
            if (found) {
                result |= (uint8_t)(1U << (2 * first + last));
            }
        }
    }
    return result;
}

static int bi_total(uint8_t mask) {
    for (int first = 0; first < 2; ++first) {
        if (
            !mask_has(mask, first, 0)
            && !mask_has(mask, first, 1)
        ) {
            return 0;
        }
    }
    for (int second = 0; second < 2; ++second) {
        if (
            !mask_has(mask, 0, second)
            && !mask_has(mask, 1, second)
        ) {
            return 0;
        }
    }
    return 1;
}

static int node_value(size_t node, size_t assignment) {
    /*
     * Nodes: A=0 B=1 C=2 D=3 E=4 U=5 V=6 W=7.
     * The complete assignment uses the same bit positions.
     */
    return (int)((assignment >> node) & 1U);
}

int main(void) {
    uint8_t relation[RELATION_COUNT] = {0};
    size_t relation_count = 0U;
    for (uint8_t mask = 0U; mask < 16U; ++mask) {
        if (bi_total(mask)) {
            relation[relation_count++] = mask;
        }
    }
    if (relation_count != RELATION_COUNT) {
        fprintf(stderr, "unexpected bi-total relation count\n");
        return 2;
    }

    static const size_t edge_first[EDGE_COUNT] = {
        0U, 1U, 5U, 2U, 6U, 3U, 4U
    };
    static const size_t edge_second[EDGE_COUNT] = {
        5U, 5U, 6U, 6U, 7U, 7U, 7U
    };
    uint64_t edge_accept[EDGE_COUNT][RELATION_COUNT][4] = {{{0}}};
    for (size_t edge = 0U; edge < EDGE_COUNT; ++edge) {
        for (size_t r = 0U; r < RELATION_COUNT; ++r) {
            for (
                size_t assignment = 0U;
                assignment < COMPLETE_ASSIGNMENTS;
                ++assignment
            ) {
                if (
                    mask_has(
                        relation[r],
                        node_value(edge_first[edge], assignment),
                        node_value(edge_second[edge], assignment)
                    )
                ) {
                    edge_accept[edge][r][assignment / 64U] |=
                        UINT64_C(1) << (assignment % 64U);
                }
            }
        }
    }

    uint32_t pair_accept[PAIR_COUNT][16] = {{0}};
    for (size_t pair = 0U; pair < PAIR_COUNT; ++pair) {
        for (uint8_t mask = 0U; mask < 16U; ++mask) {
            uint32_t accepted = 0U;
            for (size_t assignment = 0U; assignment < 32U; ++assignment) {
                const int first = (int)(
                    (assignment >> paths[pair].first_external) & 1U
                );
                const int second = (int)(
                    (assignment >> paths[pair].second_external) & 1U
                );
                if (mask_has(mask, first, second)) {
                    accepted |= (uint32_t)1U << assignment;
                }
            }
            pair_accept[pair][mask] = accepted;
        }
    }

    uint64_t combinations = 1U;
    for (size_t edge = 0U; edge < EDGE_COUNT; ++edge) {
        combinations *= RELATION_COUNT;
    }
    uint64_t mismatch_rows = 0U;
    uint64_t exact_rows = 0U;
    uint64_t multi_witness_rows = 0U;
    for (uint64_t code = 0U; code < combinations; ++code) {
        size_t selection[EDGE_COUNT];
        uint64_t residual = code;
        for (size_t edge = 0U; edge < EDGE_COUNT; ++edge) {
            selection[edge] = (size_t)(residual % RELATION_COUNT);
            residual /= RELATION_COUNT;
        }

        uint64_t full[4] = {
            UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX
        };
        for (size_t edge = 0U; edge < EDGE_COUNT; ++edge) {
            for (size_t word = 0U; word < 4U; ++word) {
                full[word] &= edge_accept[edge][selection[edge]][word];
            }
        }
        uint32_t exact = 0U;
        for (size_t word = 0U; word < 4U; ++word) {
            exact |= (uint32_t)(full[word] & UINT64_C(0xffffffff));
            exact |= (uint32_t)(full[word] >> 32U);
        }

        uint32_t projected = UINT32_MAX;
        for (size_t pair = 0U; pair < PAIR_COUNT; ++pair) {
            uint8_t path_relation = relation[
                selection[paths[pair].edge[0]]
            ];
            if (paths[pair].reverse[0]) {
                path_relation = transpose_mask(path_relation);
            }
            for (size_t step = 1U; step < paths[pair].length; ++step) {
                uint8_t next = relation[
                    selection[paths[pair].edge[step]]
                ];
                if (paths[pair].reverse[step]) {
                    next = transpose_mask(next);
                }
                path_relation = compose(path_relation, next);
            }
            projected &= pair_accept[pair][path_relation];
        }
        mismatch_rows += (uint64_t)__builtin_popcount(projected ^ exact);
        exact_rows += (uint64_t)__builtin_popcount(exact);
        for (size_t external = 0U; external < 32U; ++external) {
            unsigned witnesses = 0U;
            for (size_t internal = 0U; internal < 8U; ++internal) {
                const size_t assignment = external | (internal << 5U);
                witnesses += (unsigned)(
                    (full[assignment / 64U] >> (assignment % 64U)) & 1U
                );
            }
            multi_witness_rows += (uint64_t)(witnesses > 1U);
        }
    }
    printf(
        "{\"mode\":\"complete-three-hub-extensional-survey\","
        "\"bi_total_relations\":%zu,"
        "\"relation_tuples\":%llu,"
        "\"boundary_assignment_rows\":%llu,"
        "\"exact_rows\":%llu,"
        "\"multi_witness_rows\":%llu,"
        "\"mismatch_rows\":%llu}\n",
        relation_count,
        (unsigned long long)combinations,
        (unsigned long long)(combinations * 32U),
        (unsigned long long)exact_rows,
        (unsigned long long)multi_witness_rows,
        (unsigned long long)mismatch_rows
    );
    return mismatch_rows == 0U ? 0 : 1;
}
