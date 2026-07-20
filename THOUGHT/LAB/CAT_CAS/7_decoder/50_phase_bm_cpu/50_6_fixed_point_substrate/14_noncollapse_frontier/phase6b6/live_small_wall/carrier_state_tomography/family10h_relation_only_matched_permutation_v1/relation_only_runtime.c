#include "relation_only_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RELATION_ONLY_RUNTIME_AUTH_ENV "FAMILY10H_RELATION_ONLY_RUNTIME_AUTHORITY"
#define RELATION_ONLY_RUNTIME_AUTH_VALUE "family10h_relation_only_matched_permutation_v1_0"

static volatile uint64_t relation_only_sink = 0u;

uint32_t relation_only_map_index(relation_only_relation_id relation, uint32_t logical_a_index) {
    uint32_t index = logical_a_index % FAMILY10H_RELATION_ONLY_LINE_COUNT;
    if (relation == RELATION_ONLY_R0) {
        return (index + 1u) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
    }
    return (index + FAMILY10H_RELATION_ONLY_LINE_COUNT - 1u) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
}

uint32_t relation_only_origin_index(uint32_t cyclic_origin, uint32_t step) {
    return (cyclic_origin + step) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
}

static uint32_t byte_offset(uint32_t line_index) {
    return (line_index % FAMILY10H_RELATION_ONLY_LINE_COUNT) * FAMILY10H_RELATION_ONLY_LINE_BYTES;
}

void relation_only_prefault(relation_only_carrier_state *state) {
    uint32_t i = 0u;
    volatile uint8_t sink = 0u;
    if (state == NULL) {
        return;
    }
    memset(state, 0, sizeof(*state));
    for (i = 0u; i < FAMILY10H_RELATION_ONLY_LINE_COUNT; ++i) {
        uint32_t offset = byte_offset(i);
        sink ^= state->lane_a[offset];
        sink ^= state->lane_b[offset];
        sink ^= state->sham_a[offset];
        sink ^= state->sham_b[offset];
    }
    relation_only_sink ^= (uint64_t)sink;
}

static void write_line(uint8_t *lane, uint32_t line_index, uint32_t tag, uint32_t step) {
    uint32_t offset = byte_offset(line_index);
    lane[offset] = (uint8_t)(lane[offset] + (uint8_t)(tag + step));
}

static void prepare_pair_step(
    relation_only_carrier_state *state,
    relation_only_preparation prep,
    uint32_t step,
    uint32_t a_index,
    uint32_t b_index
) {
    if (prep.source_order == RELATION_ONLY_ORDER_AB) {
        if (step < prep.bank_a_work) {
            write_line(state->lane_a, a_index, 0xA0u, step);
        }
        if (step < prep.bank_b_work) {
            write_line(state->lane_b, b_index, 0xB0u, step);
        }
    } else {
        if (step < prep.bank_b_work) {
            write_line(state->lane_b, b_index, 0xB0u, step);
        }
        if (step < prep.bank_a_work) {
            write_line(state->lane_a, a_index, 0xA0u, step);
        }
    }
}

static uint32_t control_b_index(relation_only_control_id control, uint32_t a_index, uint32_t step) {
    if (control == RELATION_ONLY_CONTROL_RELATION_SHAM) {
        return (a_index + 1024u + ((step * 73u) % FAMILY10H_RELATION_ONLY_LINE_COUNT)) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
    }
    if (control == RELATION_ONLY_CONTROL_ROUTE_PRESSURE_SHAM) {
        return (a_index ^ 2048u) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
    }
    if (control == RELATION_ONLY_CONTROL_DISTANCE) {
        return relation_only_map_index((step & 1u) ? RELATION_ONLY_R0 : RELATION_ONLY_R1, a_index);
    }
    return relation_only_map_index(RELATION_ONLY_R0, a_index);
}

int relation_only_prepare(relation_only_preparation prep, relation_only_carrier_state *state) {
    uint32_t step = 0u;
    if (state == NULL) {
        return 0;
    }
    if (prep.bank_a_work + prep.bank_b_work != FAMILY10H_RELATION_ONLY_TOTAL_WORK) {
        return 0;
    }
    if (prep.cyclic_origin >= FAMILY10H_RELATION_ONLY_LINE_COUNT) {
        return 0;
    }
    for (step = 0u; step < FAMILY10H_RELATION_ONLY_TOTAL_WORK; ++step) {
        uint32_t a_index = relation_only_origin_index(prep.cyclic_origin, step);
        uint32_t b_index = relation_only_map_index(prep.relation, a_index);
        if (prep.control != RELATION_ONLY_CONTROL_NONE) {
            b_index = control_b_index(prep.control, a_index, step);
        }
        if (prep.control == RELATION_ONLY_CONTROL_INDEPENDENT_MARGINAL_REPLAY) {
            if (prep.source_order == RELATION_ONLY_ORDER_AB) {
                if (step < prep.bank_a_work) {
                    write_line(state->lane_a, a_index, 0xA0u, step);
                }
                if (step < prep.bank_b_work) {
                    write_line(state->lane_b, relation_only_origin_index(prep.cyclic_origin, step), 0xB0u, step);
                }
            } else {
                if (step < prep.bank_b_work) {
                    write_line(state->lane_b, relation_only_origin_index(prep.cyclic_origin, step), 0xB0u, step);
                }
                if (step < prep.bank_a_work) {
                    write_line(state->lane_a, a_index, 0xA0u, step);
                }
            }
        } else {
            prepare_pair_step(state, prep, step, a_index, b_index);
        }
    }
    return 1;
}

static uint64_t mix(uint64_t acc, uint8_t value, uint32_t line_index, uint32_t tag) {
    acc ^= (uint64_t)value + ((uint64_t)line_index << 8) + (uint64_t)tag;
    acc *= UINT64_C(1099511628211);
    return acc;
}

static uint64_t query_line_pair(const relation_only_carrier_state *state, uint32_t a_index, uint32_t b_index, relation_only_order_id order) {
    uint64_t acc = UINT64_C(1469598103934665603);
    uint32_t a_offset = byte_offset(a_index);
    uint32_t b_offset = byte_offset(b_index);
    if (order == RELATION_ONLY_ORDER_AB) {
        acc = mix(acc, state->lane_a[a_offset], a_index, 0xA0u);
        acc = mix(acc, state->lane_b[b_offset], b_index, 0xB0u);
    } else {
        acc = mix(acc, state->lane_b[b_offset], b_index, 0xB0u);
        acc = mix(acc, state->lane_a[a_offset], a_index, 0xA0u);
    }
    return acc;
}

uint64_t relation_only_query(relation_only_query query, const relation_only_carrier_state *state) {
    uint64_t acc = UINT64_C(1469598103934665603);
    uint32_t step = 0u;
    if (state == NULL || query.cyclic_origin >= FAMILY10H_RELATION_ONLY_LINE_COUNT) {
        return 0u;
    }
    for (step = 0u; step < FAMILY10H_RELATION_ONLY_TOTAL_WORK; ++step) {
        uint32_t a_index = relation_only_origin_index(query.cyclic_origin, step);
        uint32_t b_index = relation_only_map_index(query.relation, a_index);
        if (query.control != RELATION_ONLY_CONTROL_NONE) {
            b_index = control_b_index(query.control, a_index, step);
        }
        acc ^= query_line_pair(state, a_index, b_index, query.query_order);
        acc *= UINT64_C(1099511628211);
    }
    relation_only_sink ^= acc;
    return acc;
}

int relation_only_runtime_live_authority_present(void) {
    const char *value = getenv(RELATION_ONLY_RUNTIME_AUTH_ENV);
    return value != NULL && strcmp(value, RELATION_ONLY_RUNTIME_AUTH_VALUE) == 0;
}

static int permutation_self_test(void) {
    uint8_t seen_r0[FAMILY10H_RELATION_ONLY_LINE_COUNT];
    uint8_t seen_r1[FAMILY10H_RELATION_ONLY_LINE_COUNT];
    uint32_t i = 0u;
    memset(seen_r0, 0, sizeof(seen_r0));
    memset(seen_r1, 0, sizeof(seen_r1));
    for (i = 0u; i < FAMILY10H_RELATION_ONLY_LINE_COUNT; ++i) {
        uint32_t r0 = relation_only_map_index(RELATION_ONLY_R0, i);
        uint32_t r1 = relation_only_map_index(RELATION_ONLY_R1, i);
        if (r0 >= FAMILY10H_RELATION_ONLY_LINE_COUNT || r1 >= FAMILY10H_RELATION_ONLY_LINE_COUNT) {
            return 0;
        }
        seen_r0[r0] = 1u;
        seen_r1[r1] = 1u;
    }
    for (i = 0u; i < FAMILY10H_RELATION_ONLY_LINE_COUNT; ++i) {
        if (seen_r0[i] != 1u || seen_r1[i] != 1u) {
            return 0;
        }
    }
    return 1;
}

int relation_only_runtime_self_test(void) {
    relation_only_carrier_state state;
    const uint32_t origins[] = {0u, 1024u, 2048u, 3072u};
    size_t i = 0u;
    if (!permutation_self_test()) {
        return 0;
    }
    for (i = 0u; i < sizeof(origins) / sizeof(origins[0]); ++i) {
        relation_only_preparation prep;
        relation_only_query q0;
        relation_only_query q1;
        uint64_t r0 = 0u;
        uint64_t r1 = 0u;
        relation_only_prefault(&state);
        prep.q = 512;
        prep.bank_a_work = FAMILY10H_RELATION_ONLY_M + 512u;
        prep.bank_b_work = FAMILY10H_RELATION_ONLY_M - 512u;
        prep.relation = RELATION_ONLY_R0;
        prep.source_order = (i & 1u) ? RELATION_ONLY_ORDER_BA : RELATION_ONLY_ORDER_AB;
        prep.control = RELATION_ONLY_CONTROL_NONE;
        prep.cyclic_origin = origins[i];
        if (!relation_only_prepare(prep, &state)) {
            return 0;
        }
        q0.relation = RELATION_ONLY_R0;
        q0.query_order = RELATION_ONLY_ORDER_AB;
        q0.control = RELATION_ONLY_CONTROL_NONE;
        q0.cyclic_origin = origins[i];
        q1 = q0;
        q1.relation = RELATION_ONLY_R1;
        r0 = relation_only_query(q0, &state);
        r1 = relation_only_query(q1, &state);
        if (r0 == 0u || r1 == 0u || r0 == r1) {
            return 0;
        }
    }
    {
        relation_only_preparation prep;
        relation_only_query query;
        relation_only_prefault(&state);
        prep.q = 0;
        prep.bank_a_work = FAMILY10H_RELATION_ONLY_M;
        prep.bank_b_work = FAMILY10H_RELATION_ONLY_M;
        prep.relation = RELATION_ONLY_R0;
        prep.source_order = RELATION_ONLY_ORDER_AB;
        prep.control = RELATION_ONLY_CONTROL_INDEPENDENT_MARGINAL_REPLAY;
        prep.cyclic_origin = 0u;
        query.relation = RELATION_ONLY_R0;
        query.query_order = RELATION_ONLY_ORDER_AB;
        query.control = RELATION_ONLY_CONTROL_INDEPENDENT_MARGINAL_REPLAY;
        query.cyclic_origin = 0u;
        if (!relation_only_prepare(prep, &state) || relation_only_query(query, &state) == 0u) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        int passed = relation_only_runtime_self_test();
        printf("{\"schema\":\"FAMILY10H_RELATION_ONLY_RUNTIME_SELF_TEST_V1\",\"passed\":%s,\"pmu_opened\":false,\"live_activity\":false}\n", passed ? "true" : "false");
        return passed ? 0 : 1;
    }
    if (argc == 4 && strcmp(argv[1], "--execute-schedule") == 0) {
        if (!relation_only_runtime_live_authority_present()) {
            fprintf(stderr, "relation-only runtime authority missing; refusing schedule execution\n");
            return 13;
        }
        fprintf(stderr, "relation-only PMU schedule execution is reserved for a separately authorized target transaction\n");
        return 14;
    }
    fprintf(stderr, "usage: %s --self-test | --execute-schedule <schedule.tsv> <output-root>\n", argv[0]);
    return 2;
}
