#ifndef FAMILY10H_RELATION_ONLY_RUNTIME_H
#define FAMILY10H_RELATION_ONLY_RUNTIME_H

#include <stdint.h>

#define FAMILY10H_RELATION_ONLY_LINE_COUNT 4096u
#define FAMILY10H_RELATION_ONLY_TOTAL_WORK 4096u

typedef enum relation_only_relation_id {
    RELATION_ONLY_R0 = 0,
    RELATION_ONLY_R1 = 1
} relation_only_relation_id;

typedef struct relation_only_tuple {
    relation_only_relation_id r_prepare;
    relation_only_relation_id r_query;
    uint32_t q;
    uint32_t delay_ns;
    uint32_t source_order;
    uint32_t query_order;
} relation_only_tuple;

uint32_t relation_only_map_index(relation_only_relation_id relation, uint32_t logical_a_index);
int relation_only_runtime_live_disabled(void);

#endif
