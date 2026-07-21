#ifndef FAMILY10H_RELATION_LIFETIME_RUNTIME_H
#define FAMILY10H_RELATION_LIFETIME_RUNTIME_H

#include <stdint.h>

#define FAMILY10H_RELATION_LIFETIME_M 2048
#define FAMILY10H_RELATION_LIFETIME_TOTAL_WORK 4096u
#define FAMILY10H_RELATION_LIFETIME_LINE_COUNT 4096u
#define FAMILY10H_RELATION_LIFETIME_LINE_BYTES 64u
#define FAMILY10H_RELATION_LIFETIME_LANE_BYTES 262144u

typedef enum relation_lifetime_relation_id {
    RELATION_LIFETIME_R0 = 0,
    RELATION_LIFETIME_R1 = 1
} relation_lifetime_relation_id;

typedef enum relation_lifetime_order_id {
    RELATION_LIFETIME_ORDER_AB = 0,
    RELATION_LIFETIME_ORDER_BA = 1
} relation_lifetime_order_id;

typedef enum relation_lifetime_control_id {
    RELATION_LIFETIME_CONTROL_NONE = 0,
    RELATION_LIFETIME_CONTROL_RELATION_SHAM = 1,
    RELATION_LIFETIME_CONTROL_ROUTE_PRESSURE_SHAM = 2,
    RELATION_LIFETIME_CONTROL_INDEPENDENT_MARGINAL_REPLAY = 3,
    RELATION_LIFETIME_CONTROL_DISTANCE = 4
} relation_lifetime_control_id;

typedef struct relation_lifetime_preparation {
    int32_t q;
    uint32_t bank_a_work;
    uint32_t bank_b_work;
    relation_lifetime_relation_id relation;
    relation_lifetime_order_id source_order;
    relation_lifetime_control_id control;
    uint32_t cyclic_origin;
} relation_lifetime_preparation;

typedef struct relation_lifetime_query_spec {
    relation_lifetime_relation_id relation;
    relation_lifetime_order_id query_order;
    relation_lifetime_control_id control;
    uint32_t cyclic_origin;
} relation_lifetime_query_spec;

typedef struct relation_lifetime_carrier_state {
    uint8_t lane_a[FAMILY10H_RELATION_LIFETIME_LANE_BYTES];
    uint8_t lane_b[FAMILY10H_RELATION_LIFETIME_LANE_BYTES];
    uint8_t sham_a[FAMILY10H_RELATION_LIFETIME_LANE_BYTES];
    uint8_t sham_b[FAMILY10H_RELATION_LIFETIME_LANE_BYTES];
} relation_lifetime_carrier_state;

uint32_t relation_lifetime_map_index(relation_lifetime_relation_id relation, uint32_t logical_a_index);
uint32_t relation_lifetime_origin_index(uint32_t cyclic_origin, uint32_t step);
void relation_lifetime_prefault(relation_lifetime_carrier_state *state);
int relation_lifetime_prepare(relation_lifetime_preparation prep, relation_lifetime_carrier_state *state);
uint64_t relation_lifetime_query(relation_lifetime_query_spec query, const relation_lifetime_carrier_state *state);
int relation_lifetime_runtime_self_test(void);
int relation_lifetime_runtime_live_authority_present(void);

#endif
