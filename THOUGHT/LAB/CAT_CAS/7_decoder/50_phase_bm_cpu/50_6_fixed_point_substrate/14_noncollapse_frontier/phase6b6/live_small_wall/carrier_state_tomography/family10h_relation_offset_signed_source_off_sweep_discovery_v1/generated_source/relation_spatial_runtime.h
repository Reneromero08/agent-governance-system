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
    RELATION_SPATIAL_CONTROL_DISTANCE_MATCHED = 4,
    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1 = 5,
    RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_2 = 6,
    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_2 = 7,
    RELATION_SPATIAL_CONTROL_DEAD_OFFSET_2 = 8,
    RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_OFFSET_2 = 9,
    RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_4 = 10,
    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_4 = 11,
    RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_8 = 12,
    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_8 = 13,
    RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_16 = 14,
    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_16 = 15,
    RELATION_SPATIAL_CONTROL_SOURCE_ON_OFFSET_1024 = 16,
    RELATION_SPATIAL_CONTROL_SOURCE_OFF_OFFSET_1024 = 17
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
