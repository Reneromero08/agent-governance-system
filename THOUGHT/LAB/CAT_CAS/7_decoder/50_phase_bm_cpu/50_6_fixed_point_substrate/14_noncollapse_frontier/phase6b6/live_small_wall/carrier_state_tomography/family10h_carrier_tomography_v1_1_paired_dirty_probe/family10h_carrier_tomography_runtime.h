#ifndef FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_H
#define FAMILY10H_CARRIER_TOMOGRAPHY_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#define F10_TOMO_M 2048
#define F10_TOMO_TOTAL_WORK 4096
#define F10_TOMO_LINE_COUNT 4096
#define F10_TOMO_LANE_BYTES 262144
#define F10_TOMO_AFFINE_MULTIPLIER 73
#define F10_TOMO_AFFINE_OFFSET 19

typedef struct {
    int32_t q;
    uint32_t bank_a_work;
    uint32_t bank_b_work;
    uint32_t dummy_work;
    uint32_t dummy_a_work;
    uint32_t dummy_b_work;
    int source_off_control;
    int map_variant;
    int source_order_variant;
} F10CarrierPreparation;

typedef struct {
    uint8_t lane_a[F10_TOMO_LANE_BYTES] __attribute__((aligned(4096)));
    uint8_t lane_b[F10_TOMO_LANE_BYTES] __attribute__((aligned(4096)));
    uint8_t dummy_a[F10_TOMO_LANE_BYTES] __attribute__((aligned(4096)));
    uint8_t dummy_b[F10_TOMO_LANE_BYTES] __attribute__((aligned(4096)));
    uint8_t sham[F10_TOMO_LANE_BYTES] __attribute__((aligned(4096)));
} F10CarrierState;

int f10_carrier_prepare(F10CarrierPreparation prep, F10CarrierState *state);
uint32_t f10_carrier_affine_line(uint32_t line_index);
uint64_t f10_carrier_query(const F10CarrierState *state, const char *query_name);
uint64_t f10_carrier_query_mapped(const F10CarrierState *state, const char *query_name, int map_variant);
int f10_carrier_runtime_self_test(void);

#endif
