#ifndef FAMILY10H_RELATION_ONLY_RUNTIME_H
#define FAMILY10H_RELATION_ONLY_RUNTIME_H

#include <stdint.h>

#define FAMILY10H_RELATION_ONLY_M 2048
#define FAMILY10H_RELATION_ONLY_TOTAL_WORK 4096u
#define FAMILY10H_RELATION_ONLY_LINE_COUNT 4096u
#define FAMILY10H_RELATION_ONLY_LINE_BYTES 64u
#define FAMILY10H_RELATION_ONLY_LANE_BYTES 262144u

typedef enum relation_only_relation_id {
    RELATION_ONLY_R0 = 0,
    RELATION_ONLY_R1 = 1
} relation_only_relation_id;

typedef enum relation_only_order_id {
    RELATION_ONLY_ORDER_AB = 0,
    RELATION_ONLY_ORDER_BA = 1
} relation_only_order_id;

typedef enum relation_only_control_id {
    RELATION_ONLY_CONTROL_NONE = 0,
    RELATION_ONLY_CONTROL_RELATION_SHAM = 1,
    RELATION_ONLY_CONTROL_ROUTE_PRESSURE_SHAM = 2,
    RELATION_ONLY_CONTROL_INDEPENDENT_MARGINAL_REPLAY = 3,
    RELATION_ONLY_CONTROL_DISTANCE = 4
} relation_only_control_id;

typedef struct relation_only_preparation {
    int32_t q;
    uint32_t bank_a_work;
    uint32_t bank_b_work;
    relation_only_relation_id relation;
    relation_only_order_id source_order;
    relation_only_control_id control;
    uint32_t cyclic_origin;
} relation_only_preparation;

typedef struct relation_only_query_spec {
    relation_only_relation_id relation;
    relation_only_order_id query_order;
    relation_only_control_id control;
    uint32_t cyclic_origin;
} relation_only_query_spec;

typedef struct relation_only_carrier_state {
    uint8_t lane_a[FAMILY10H_RELATION_ONLY_LANE_BYTES];
    uint8_t lane_b[FAMILY10H_RELATION_ONLY_LANE_BYTES];
    uint8_t sham_a[FAMILY10H_RELATION_ONLY_LANE_BYTES];
    uint8_t sham_b[FAMILY10H_RELATION_ONLY_LANE_BYTES];
} relation_only_carrier_state;

uint32_t relation_only_map_index(relation_only_relation_id relation, uint32_t logical_a_index);
uint32_t relation_only_origin_index(uint32_t cyclic_origin, uint32_t step);
void relation_only_prefault(relation_only_carrier_state *state);
int relation_only_prepare(relation_only_preparation prep, relation_only_carrier_state *state);
uint64_t relation_only_query(relation_only_query_spec query, const relation_only_carrier_state *state);
int relation_only_runtime_self_test(void);
int relation_only_runtime_live_authority_present(void);

#endif
