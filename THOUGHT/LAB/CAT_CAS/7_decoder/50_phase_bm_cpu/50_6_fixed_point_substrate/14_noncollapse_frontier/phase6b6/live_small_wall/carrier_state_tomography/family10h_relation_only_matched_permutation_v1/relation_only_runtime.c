#include "relation_only_runtime.h"

uint32_t relation_only_map_index(relation_only_relation_id relation, uint32_t logical_a_index) {
    uint32_t index = logical_a_index % FAMILY10H_RELATION_ONLY_LINE_COUNT;
    if (relation == RELATION_ONLY_R0) {
        return (index + 1u) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
    }
    return (index + FAMILY10H_RELATION_ONLY_LINE_COUNT - 1u) % FAMILY10H_RELATION_ONLY_LINE_COUNT;
}

int relation_only_runtime_live_disabled(void) {
    return 1;
}
