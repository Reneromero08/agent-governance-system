#ifndef ORBITSTATE_INDEPENDENT_RUNTIME_H
#define ORBITSTATE_INDEPENDENT_RUNTIME_H

#include <stdint.h>

#define ORBITSTATE_MODULUS 256u
#define ORBITSTATE_D_MEMBER 23u
#define ORBITSTATE_FOLD_MEMBER 233u
#define ORBITSTATE_QUANTIZATION_SCALE 1536
#define ORBITSTATE_BASE_WORK 2048
#define ORBITSTATE_TOTAL_WORK 4096
#define ORBITSTATE_SOURCE_CORE 4
#define ORBITSTATE_RECEIVER_CORE 5
#define ORBITSTATE_LINE_BYTES 64
#define ORBITSTATE_BANK_LINES 4096
#define ORBITSTATE_BANK_INITIAL_VALUE 0x5au
#define ORBITSTATE_DUMMY_BANK_INITIAL_VALUE 0x3cu

typedef struct {
    uint32_t modulus;
    uint32_t member;
} OrbitState;

int orbitstate_round_to_i32(double value);
int orbitstate_compute_q(
    OrbitState state,
    int public_phase_index,
    int private_phase_index,
    const char *response_mode,
    int polarity_inversion,
    int source_off_dummy_mode
);

#endif
