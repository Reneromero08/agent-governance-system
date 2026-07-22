#ifndef FAMILY10H_RELATION_SPATIAL_RUNTIME_H
#define FAMILY10H_RELATION_SPATIAL_RUNTIME_H

#include <stdint.h>

#define FAMILY10H_RELATION_SPATIAL_TOTAL_WORK 4096u
#define FAMILY10H_RELATION_SPATIAL_LINE_COUNT 4096u
#define FAMILY10H_RELATION_SPATIAL_LINE_BYTES 64u
#define FAMILY10H_RELATION_SPATIAL_LANE_BYTES 262144u
#define FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_COUNT 256u
#define FAMILY10H_RELATION_SPATIAL_PAIR_SAMPLE_STRIDE 16u

typedef enum relation_spatial_relation_id {
    RELATION_SPATIAL_R0 = 0,
    RELATION_SPATIAL_R1 = 1
} relation_spatial_relation_id;

typedef enum relation_spatial_order_id {
    RELATION_SPATIAL_ORDER_AB = 0,
    RELATION_SPATIAL_ORDER_BA = 1
} relation_spatial_order_id;

typedef enum relation_spatial_control_id {
    RELATION_SPATIAL_CONTROL_NONE = 0,
    RELATION_SPATIAL_CONTROL_RELATION_SHAM = 1,
    RELATION_SPATIAL_CONTROL_SCRAMBLED_PAIR = 2,
    RELATION_SPATIAL_CONTROL_ROUTE_PRESSURE = 3,
    RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED = 4
} relation_spatial_control_id;

typedef struct relation_spatial_preparation {
    uint32_t bank_a_work;
    uint32_t bank_b_work;
    relation_spatial_relation_id relation;
    relation_spatial_order_id source_order;
    uint32_t cyclic_origin;
} relation_spatial_preparation;

typedef struct relation_spatial_carrier_state {
    uint8_t lane_a[FAMILY10H_RELATION_SPATIAL_LANE_BYTES];
    uint8_t lane_b[FAMILY10H_RELATION_SPATIAL_LANE_BYTES];
    uint8_t sham_a[FAMILY10H_RELATION_SPATIAL_LANE_BYTES];
    uint8_t sham_b[FAMILY10H_RELATION_SPATIAL_LANE_BYTES];
} relation_spatial_carrier_state;

uint32_t relation_spatial_map_index(relation_spatial_relation_id relation, uint32_t logical_a_index);
uint32_t relation_spatial_origin_index(uint32_t cyclic_origin, uint32_t step);
void relation_spatial_prefault(relation_spatial_carrier_state *state);
int relation_spatial_prepare(relation_spatial_preparation prep, relation_spatial_carrier_state *state);
int relation_spatial_runtime_self_test(void);
int relation_spatial_runtime_live_authority_present(void);

#endif
