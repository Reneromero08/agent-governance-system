/*
 * Phase-QEMU V1 exact four-mode/two-boson echo device.
 *
 * The hidden carrier is a real homogeneous degree-two polynomial in four
 * creation variables.  Its ten coefficients share a power-of-two
 * denominator.  Two pointer branches model the controlled parity interaction.
 * The guest can upload only a public gate descriptor; it cannot read carrier
 * coefficients, upload an inverse, or release a boundary before the device has
 * applied the public adjoint and verified exact restoration.
 *
 * This remains a deterministic software model.  It establishes neither a
 * physical boson/phonon result nor a resource advantage.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "hw/pci/pci.h"
#include "hw/pci/pci_device.h"
#include "hw/qdev-properties.h"
#include "migration/vmstate.h"
#include "qemu/module.h"
#include "qemu/units.h"
#include "qom/object.h"

#define TYPE_PHASE_QEMU_V1 "phase-qemu-v1"
OBJECT_DECLARE_SIMPLE_TYPE(PhaseQemuV1State, PHASE_QEMU_V1)

#define PHASE_V1_VENDOR_ID PCI_VENDOR_ID_QEMU
#define PHASE_V1_DEVICE_ID 0x11f1
#define PHASE_V1_MAGIC 0x50485631u /* PHV1 */
#define PHASE_V1_ABI 0x00020000u
#define PHASE_V1_BACKEND 1u

#define PHASE_V1_BASIS_CELLS 10u
#define PHASE_V1_POINTER_BRANCHES 2u
#define PHASE_V1_COEFFICIENT_CELLS 20u
#define PHASE_V1_MAX_DESCRIPTOR 16u
#define PHASE_V1_MAX_DENOMINATOR_POWER 30u
#define PHASE_V1_MAX_COEFFICIENT INT64_C(1125899906842624) /* 2^50 */
#define PHASE_V1_LOCKED_BOUNDARY UINT64_MAX

enum PhaseV1Register {
    REG_MAGIC = 0x000,
    REG_ABI = 0x004,
    REG_BACKEND = 0x008,
    REG_CAPABILITIES = 0x00c,
    REG_STATUS = 0x010,
    REG_ERROR = 0x014,
    REG_GENERATION = 0x018,
    REG_RESTORATION_GENERATION = 0x01c,
    REG_ARG0 = 0x020,
    REG_ARG1 = 0x024,
    REG_COMMAND = 0x028,
    REG_LIFECYCLE = 0x02c,
    REG_VIRTUAL_CYCLES = 0x030,
    REG_BOUNDARY_VALUE = 0x040,
    REG_RESOURCE_COEFFICIENT_CELLS = 0x048,
    REG_RESOURCE_SCRATCH_CELLS = 0x04c,
    REG_RESOURCE_FORWARD_GATES = 0x050,
    REG_RESOURCE_INVERSE_GATES = 0x058,
    REG_RESOURCE_POINTER_GATES = 0x060,
    REG_RESOURCE_PEAK_BITS = 0x068,
    REG_REQUEST_OWNER = 0x070,
    REG_REQUEST_PROGRAM = 0x074,
    REG_REQUEST_GENERATION = 0x078,
    REG_DESCRIPTOR_INDEX = 0x07c,
    REG_DESCRIPTOR_WORD = 0x080,
    REG_DESCRIPTOR_LENGTH = 0x084,
    REG_BOUNDARY_MODE = 0x088,
    REG_DESCRIPTOR_FINGERPRINT = 0x090,
    REG_FAULT_MODE = 0x098,
};

enum PhaseV1Command {
    CMD_LEASE = 1,
    CMD_PREPARE = 2,
    CMD_ISOLATE_SOURCE = 3,
    CMD_SEAL_DESCRIPTOR = 4,
    CMD_EXECUTE_ATOMIC = 5,
    CMD_BEGIN_REUSE = 6,
    CMD_SNAPSHOT = 7,
};

enum PhaseV1Gate {
    GATE_A = 1,
    GATE_A_DAG = 2,
    GATE_B = 3,
    GATE_B_DAG = 4,
    GATE_K01 = 5,
    GATE_K03 = 6,
    GATE_KERR_IDENTITY_SHAM = 7,
};

enum PhaseV1FaultMode {
    FAULT_NONE = 0,
    FAULT_MISSING_INVERSE = 1,
    FAULT_WRONG_KERR_INVERSE = 2,
    FAULT_REORDERED_INVERSE_PREFIX = 3,
};

enum PhaseV1Error {
    ERR_NONE = 0,
    ERR_BAD_STATE = 1,
    ERR_BAD_ARGUMENT = 2,
    ERR_TAG_MISMATCH = 3,
    ERR_GENERATION_MISMATCH = 4,
    ERR_DESCRIPTOR_INVALID = 5,
    ERR_SOURCE_NOT_ISOLATED = 6,
    ERR_CARRIER_ABSENT = 7,
    ERR_POINTER_ENTANGLED = 8,
    ERR_RESTORATION_FAILED = 9,
    ERR_OVERFLOW = 10,
    ERR_INVARIANT = 11,
    ERR_SNAPSHOT_REJECTED = 12,
    ERR_SNAPSHOT_LINEAGE = 13,
    ERR_RESPONSE_LOCKED = 14,
};

enum PhaseV1Status {
    ST_CANONICAL = 1u << 0,
    ST_LEASED = 1u << 1,
    ST_PREPARED = 1u << 2,
    ST_SOURCE_ISOLATED = 1u << 3,
    ST_DESCRIPTOR_SEALED = 1u << 4,
    ST_RESPONSE_READY = 1u << 5,
    ST_SPENT = 1u << 6,
    ST_RESTORED = 1u << 7,
    ST_CARRIER_PRESENT = 1u << 8,
    ST_SNAPSHOT_LINEAGE = 1u << 9,
    ST_POINTER_CLEAR = 1u << 10,
};

enum PhaseV1Capability {
    CAP_NOMINAL_COMMAND_TAGS = 1u << 0,
    CAP_EXACT_DYADIC_STATE = 1u << 1,
    CAP_PUBLIC_DESCRIPTOR = 1u << 2,
    CAP_SOURCE_ISOLATION = 1u << 3,
    CAP_ATOMIC_TRANSACTION = 1u << 4,
    CAP_PARITY_POINTER = 1u << 5,
    CAP_PUBLIC_ADJOINT = 1u << 6,
    CAP_RESTORED_REUSE = 1u << 7,
    CAP_CROSS_KERR = 1u << 8,
    CAP_MIGRATION_SHAM_MARKING = 1u << 9,
    CAP_TEST_FAULTS_ACTIVE = 1u << 10,
};

enum PhaseV1Lifecycle {
    LIFE_EMPTY = 0,
    LIFE_LEASED = 1,
    LIFE_PREPARED = 2,
    LIFE_ISOLATED = 3,
    LIFE_SEALED = 4,
    LIFE_EXECUTING = 5,
    LIFE_RESPONSE_READY = 6,
    LIFE_REUSABLE = 7,
    LIFE_SPENT = 8,
    LIFE_SHAM = 9,
};

/* Public occupation-basis order used by the independent oracle. */
static const uint8_t basis_left[PHASE_V1_BASIS_CELLS] = {
    0, 0, 0, 0, 1, 1, 1, 2, 2, 3,
};
static const uint8_t basis_right[PHASE_V1_BASIS_CELLS] = {
    0, 1, 2, 3, 1, 2, 3, 2, 3, 3,
};

/*
 * Raw homogeneous-polynomial substitutions.  Each matrix is the integer
 * numerator of a single-particle orthogonal map with denominator sqrt(2).
 * A = R_01 R_23 and B = R_12 R_03, where
 *   creation_i -> (creation_i - creation_j) / sqrt(2)
 *   creation_j -> (creation_i + creation_j) / sqrt(2).
 * A degree-two carrier therefore acquires one common factor /2 per layer.
 */
static const int8_t transform_a[4][4] = {
    { 1, -1,  0,  0 },
    { 1,  1,  0,  0 },
    { 0,  0,  1, -1 },
    { 0,  0,  1,  1 },
};

static const int8_t transform_b[4][4] = {
    { 1,  0,  0, -1 },
    { 0,  1, -1,  0 },
    { 0,  1,  1,  0 },
    { 1,  0,  0,  1 },
};

struct PhaseQemuV1State {
    PCIDevice parent_obj;
    MemoryRegion mmio;

    bool carrier_present;
    bool configured_carrier_present;
    bool leased;
    bool prepared;
    bool source_isolated;
    bool descriptor_sealed;
    bool response_ready;
    bool spent;
    bool restored;
    bool snapshot_lineage;
    bool pointer_clear;
    bool coupler_active;

    uint32_t error;
    uint32_t lifecycle;
    uint32_t owner_tag;
    uint32_t program_tag;
    uint32_t request_owner_tag;
    uint32_t request_program_tag;
    uint32_t request_generation;
    uint32_t generation;
    uint32_t restoration_generation;
    uint32_t arg0;
    uint32_t arg1;
    uint32_t descriptor_index;
    uint32_t descriptor_length;
    uint32_t boundary_mode;
    uint32_t boundary_value;
    uint32_t denominator_power;
    uint32_t forward_cursor;
    uint32_t inverse_cursor;
    uint32_t committed_prefix;
    uint32_t peak_coefficient_bits;

    uint64_t descriptor_fingerprint;
    uint64_t virtual_cycles;
    uint64_t forward_gates;
    uint64_t inverse_gates;
    uint64_t pointer_gates;

    int64_t coefficient[PHASE_V1_COEFFICIENT_CELLS];
    uint32_t descriptor[PHASE_V1_MAX_DESCRIPTOR];

    /* Test configuration affects execution and is migration-validated. */
    uint32_t fault_mode;
    uint32_t configured_fault_mode;
};

static unsigned coefficient_index(unsigned branch, unsigned basis)
{
    return branch * PHASE_V1_BASIS_CELLS + basis;
}

static unsigned pair_basis_index(unsigned left, unsigned right)
{
    static const uint8_t table[4][4] = {
        { 0, 1, 2, 3 },
        { 1, 4, 5, 6 },
        { 2, 5, 7, 8 },
        { 3, 6, 8, 9 },
    };

    return table[left][right];
}

static unsigned occupation(unsigned basis, unsigned mode)
{
    return (basis_left[basis] == mode) + (basis_right[basis] == mode);
}

static bool coefficient_array_zero(const int64_t *coeff, unsigned branch)
{
    unsigned basis;

    for (basis = 0; basis < PHASE_V1_BASIS_CELLS; basis++) {
        if (coeff[coefficient_index(branch, basis)] != 0) {
            return false;
        }
    }
    return true;
}

static unsigned coefficient_bits(int64_t value)
{
    uint64_t magnitude;
    unsigned bits = 0;

    if (value == 0) {
        return 0;
    }
    magnitude = value < 0 ? (uint64_t)(-(value + 1)) + 1 : value;
    while (magnitude != 0) {
        magnitude >>= 1;
        bits++;
    }
    return bits;
}

static void update_peak_bits(PhaseQemuV1State *s)
{
    unsigned index;

    for (index = 0; index < PHASE_V1_COEFFICIENT_CELLS; index++) {
        s->peak_coefficient_bits = MAX(s->peak_coefficient_bits,
                                       coefficient_bits(s->coefficient[index]));
    }
}

static bool coefficient_bound_valid(const int64_t *coeff)
{
    unsigned index;

    for (index = 0; index < PHASE_V1_COEFFICIENT_CELLS; index++) {
        if (coeff[index] > PHASE_V1_MAX_COEFFICIENT ||
            coeff[index] < -PHASE_V1_MAX_COEFFICIENT) {
            return false;
        }
    }
    return true;
}

static bool norm_is_one(const int64_t *coeff, uint32_t denominator_power)
{
    __int128 numerator = 0;
    __int128 denominator;
    unsigned branch;
    unsigned basis;

    if (denominator_power > PHASE_V1_MAX_DENOMINATOR_POWER) {
        return false;
    }
    denominator = ((__int128)1) << (2 * denominator_power);
    for (branch = 0; branch < PHASE_V1_POINTER_BRANCHES; branch++) {
        for (basis = 0; basis < PHASE_V1_BASIS_CELLS; basis++) {
            int64_t value = coeff[coefficient_index(branch, basis)];
            unsigned weight = basis_left[basis] == basis_right[basis] ? 2 : 1;

            numerator += (__int128)weight * value * value;
        }
    }
    return numerator == denominator;
}

static void canonical_reduce_array(int64_t *coeff, uint32_t *denominator_power)
{
    while (*denominator_power != 0) {
        unsigned index;
        bool all_even = true;

        for (index = 0; index < PHASE_V1_COEFFICIENT_CELLS; index++) {
            if ((coeff[index] & 1) != 0) {
                all_even = false;
                break;
            }
        }
        if (!all_even) {
            break;
        }
        for (index = 0; index < PHASE_V1_COEFFICIENT_CELLS; index++) {
            coeff[index] /= 2;
        }
        (*denominator_power)--;
    }
}

static bool add_checked(int64_t *target, __int128 term)
{
    __int128 result = (__int128)*target + term;

    if (result > PHASE_V1_MAX_COEFFICIENT ||
        result < -PHASE_V1_MAX_COEFFICIENT) {
        return false;
    }
    *target = result;
    return true;
}

static int8_t transform_entry(const int8_t map[4][4], bool transpose,
                              unsigned source, unsigned destination)
{
    return transpose ? map[destination][source] : map[source][destination];
}

static bool apply_exchange(PhaseQemuV1State *s, const int8_t map[4][4],
                           bool transpose)
{
    int64_t next[PHASE_V1_COEFFICIENT_CELLS] = { 0 };
    uint32_t next_denominator = s->denominator_power + 1;
    unsigned branch;
    unsigned basis;

    if (next_denominator > PHASE_V1_MAX_DENOMINATOR_POWER) {
        s->error = ERR_OVERFLOW;
        return false;
    }

    for (branch = 0; branch < PHASE_V1_POINTER_BRANCHES; branch++) {
        for (basis = 0; basis < PHASE_V1_BASIS_CELLS; basis++) {
            int64_t value = s->coefficient[coefficient_index(branch, basis)];
            unsigned source_left = basis_left[basis];
            unsigned source_right = basis_right[basis];
            unsigned destination_left;
            unsigned destination_right;

            if (value == 0) {
                continue;
            }
            for (destination_left = 0; destination_left < 4;
                 destination_left++) {
                int8_t left_factor = transform_entry(map, transpose,
                                                      source_left,
                                                      destination_left);

                if (left_factor == 0) {
                    continue;
                }
                for (destination_right = 0; destination_right < 4;
                     destination_right++) {
                    int8_t right_factor = transform_entry(map, transpose,
                                                           source_right,
                                                           destination_right);
                    unsigned output_basis;
                    __int128 term;

                    if (right_factor == 0) {
                        continue;
                    }
                    output_basis = pair_basis_index(destination_left,
                                                    destination_right);
                    term = (__int128)value * left_factor * right_factor;
                    if (!add_checked(&next[coefficient_index(branch,
                                                             output_basis)],
                                     term)) {
                        s->error = ERR_OVERFLOW;
                        return false;
                    }
                }
            }
        }
    }

    canonical_reduce_array(next, &next_denominator);
    if (!coefficient_bound_valid(next) ||
        !norm_is_one(next, next_denominator)) {
        s->error = ERR_INVARIANT;
        return false;
    }
    memcpy(s->coefficient, next, sizeof(next));
    s->denominator_power = next_denominator;
    update_peak_bits(s);
    return true;
}

static bool apply_kerr(PhaseQemuV1State *s, unsigned first, unsigned second)
{
    unsigned branch;
    unsigned basis;

    for (branch = 0; branch < PHASE_V1_POINTER_BRANCHES; branch++) {
        for (basis = 0; basis < PHASE_V1_BASIS_CELLS; basis++) {
            if ((occupation(basis, first) * occupation(basis, second)) & 1) {
                unsigned index = coefficient_index(branch, basis);

                s->coefficient[index] = -s->coefficient[index];
            }
        }
    }
    return norm_is_one(s->coefficient, s->denominator_power);
}

static bool apply_gate(PhaseQemuV1State *s, uint32_t gate)
{
    bool result;

    s->coupler_active = true;
    switch (gate) {
    case GATE_A:
        result = apply_exchange(s, transform_a, false);
        break;
    case GATE_A_DAG:
        result = apply_exchange(s, transform_a, true);
        break;
    case GATE_B:
        result = apply_exchange(s, transform_b, false);
        break;
    case GATE_B_DAG:
        result = apply_exchange(s, transform_b, true);
        break;
    case GATE_K01:
        result = apply_kerr(s, 0, 1);
        break;
    case GATE_K03:
        result = apply_kerr(s, 0, 3);
        break;
    case GATE_KERR_IDENTITY_SHAM:
        result = norm_is_one(s->coefficient, s->denominator_power);
        break;
    default:
        s->error = ERR_DESCRIPTOR_INVALID;
        result = false;
        break;
    }
    s->coupler_active = false;
    if (!result && s->error == ERR_NONE) {
        s->error = ERR_INVARIANT;
    }
    return result;
}

static uint32_t adjoint_gate(uint32_t gate)
{
    switch (gate) {
    case GATE_A:
        return GATE_A_DAG;
    case GATE_A_DAG:
        return GATE_A;
    case GATE_B:
        return GATE_B_DAG;
    case GATE_B_DAG:
        return GATE_B;
    case GATE_K01:
    case GATE_K03:
    case GATE_KERR_IDENTITY_SHAM:
        return gate;
    default:
        return 0;
    }
}

static bool apply_pointer_gate(PhaseQemuV1State *s)
{
    unsigned basis;

    if (s->boundary_mode >= 4) {
        s->error = ERR_DESCRIPTOR_INVALID;
        return false;
    }
    for (basis = 0; basis < PHASE_V1_BASIS_CELLS; basis++) {
        if (occupation(basis, s->boundary_mode) & 1) {
            unsigned even = coefficient_index(0, basis);
            unsigned odd = coefficient_index(1, basis);
            int64_t temporary = s->coefficient[even];

            s->coefficient[even] = s->coefficient[odd];
            s->coefficient[odd] = temporary;
        }
    }
    s->pointer_gates++;
    s->virtual_cycles++;
    s->pointer_clear = coefficient_array_zero(s->coefficient, 1);
    if (!norm_is_one(s->coefficient, s->denominator_power)) {
        s->error = ERR_INVARIANT;
        return false;
    }
    return true;
}

static bool state_is_initial(const PhaseQemuV1State *s)
{
    unsigned index;

    if (s->denominator_power != 0 ||
        s->coefficient[coefficient_index(0, 2)] != 1) {
        return false;
    }
    for (index = 0; index < PHASE_V1_COEFFICIENT_CELLS; index++) {
        if (index != coefficient_index(0, 2) && s->coefficient[index] != 0) {
            return false;
        }
    }
    return s->pointer_clear && !s->coupler_active;
}

static uint64_t descriptor_fingerprint(const PhaseQemuV1State *s)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    unsigned index;

    hash ^= s->descriptor_length;
    hash *= UINT64_C(1099511628211);
    hash ^= s->boundary_mode;
    hash *= UINT64_C(1099511628211);
    for (index = 0; index < s->descriptor_length; index++) {
        uint32_t word = s->descriptor[index];
        unsigned byte;

        for (byte = 0; byte < 4; byte++) {
            hash ^= (word >> (8 * byte)) & 0xff;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static bool descriptor_valid(const PhaseQemuV1State *s)
{
    unsigned index;

    if (s->descriptor_length == 0 ||
        s->descriptor_length > PHASE_V1_MAX_DESCRIPTOR ||
        s->boundary_mode >= 4) {
        return false;
    }
    for (index = 0; index < s->descriptor_length; index++) {
        if (s->descriptor[index] < GATE_A ||
            s->descriptor[index] > GATE_KERR_IDENTITY_SHAM) {
            return false;
        }
    }
    return true;
}

static uint32_t active_request_error(const PhaseQemuV1State *s)
{
    if (!s->leased || s->request_owner_tag != s->owner_tag ||
        s->request_program_tag != s->program_tag) {
        return ERR_TAG_MISMATCH;
    }
    if (s->request_generation != s->generation) {
        return ERR_GENERATION_MISMATCH;
    }
    return ERR_NONE;
}

static void clear_descriptor(PhaseQemuV1State *s)
{
    memset(s->descriptor, 0, sizeof(s->descriptor));
    s->descriptor_index = 0;
    s->descriptor_length = 0;
    s->boundary_mode = 0;
    s->descriptor_fingerprint = 0;
    s->descriptor_sealed = false;
    s->forward_cursor = 0;
    s->inverse_cursor = 0;
    s->committed_prefix = 0;
}

static void release_tags(PhaseQemuV1State *s)
{
    s->leased = false;
    s->owner_tag = 0;
    s->program_tag = 0;
}

static bool execute_correct_inverse(PhaseQemuV1State *s, unsigned count)
{
    unsigned step;

    for (step = 0; step < count; step++) {
        unsigned descriptor_index = count - 1 - step;
        uint32_t gate = adjoint_gate(s->descriptor[descriptor_index]);

        if (gate == 0 || !apply_gate(s, gate)) {
            return false;
        }
        s->inverse_gates++;
        s->virtual_cycles++;
        s->inverse_cursor = step + 1;
    }
    return true;
}

static bool rollback_forward_prefix(PhaseQemuV1State *s, uint32_t original_error)
{
    bool restored_ok;

    s->error = ERR_NONE;
    restored_ok = execute_correct_inverse(s, s->committed_prefix) &&
                  state_is_initial(s);
    if (!restored_ok) {
        s->error = ERR_RESTORATION_FAILED;
        s->spent = true;
        s->restored = false;
        s->lifecycle = LIFE_SPENT;
        release_tags(s);
        return false;
    }
    s->restoration_generation++;
    s->restored = true;
    s->response_ready = false;
    s->lifecycle = LIFE_REUSABLE;
    release_tags(s);
    s->error = original_error;
    return true;
}

static bool execute_inverse_with_fault(PhaseQemuV1State *s)
{
    unsigned step;
    bool wrong_kerr_used = false;

    if (s->fault_mode == FAULT_MISSING_INVERSE) {
        return true;
    }
    for (step = 0; step < s->descriptor_length; step++) {
        unsigned descriptor_index = s->descriptor_length - 1 - step;
        uint32_t gate;

        if (s->fault_mode == FAULT_REORDERED_INVERSE_PREFIX &&
            s->descriptor_length >= 2) {
            if (step == 0) {
                descriptor_index = s->descriptor_length - 2;
            } else if (step == 1) {
                descriptor_index = s->descriptor_length - 1;
            }
        }
        gate = adjoint_gate(s->descriptor[descriptor_index]);
        if (s->fault_mode == FAULT_WRONG_KERR_INVERSE &&
            !wrong_kerr_used && (gate == GATE_K01 || gate == GATE_K03)) {
            gate = gate == GATE_K01 ? GATE_K03 : GATE_K01;
            wrong_kerr_used = true;
        }
        if (gate == 0 || !apply_gate(s, gate)) {
            return false;
        }
        s->inverse_gates++;
        s->virtual_cycles++;
        s->inverse_cursor = step + 1;
    }
    return true;
}

static void execute_atomic(PhaseQemuV1State *s)
{
    unsigned step;
    bool even_zero;
    bool odd_zero;
    uint32_t saved_boundary = 0;
    uint32_t request_error = active_request_error(s);

    if (request_error != ERR_NONE) {
        s->error = request_error;
        return;
    }
    if (s->snapshot_lineage) {
        s->error = ERR_SNAPSHOT_LINEAGE;
        return;
    }
    if (!s->carrier_present) {
        s->error = ERR_CARRIER_ABSENT;
        return;
    }
    if (!s->source_isolated) {
        s->error = ERR_SOURCE_NOT_ISOLATED;
        return;
    }
    if (!s->descriptor_sealed || s->lifecycle != LIFE_SEALED ||
        !state_is_initial(s)) {
        s->error = ERR_BAD_STATE;
        return;
    }

    s->lifecycle = LIFE_EXECUTING;
    s->restored = false;
    s->response_ready = false;
    s->boundary_value = 0;
    s->forward_cursor = 0;
    s->inverse_cursor = 0;
    s->committed_prefix = 0;

    for (step = 0; step < s->descriptor_length; step++) {
        if (!apply_gate(s, s->descriptor[step])) {
            uint32_t original_error = s->error;

            rollback_forward_prefix(s, original_error);
            return;
        }
        s->forward_gates++;
        s->virtual_cycles++;
        s->forward_cursor = step + 1;
        s->committed_prefix = step + 1;
    }

    if (!apply_pointer_gate(s)) {
        rollback_forward_prefix(s, s->error);
        return;
    }
    even_zero = coefficient_array_zero(s->coefficient, 0);
    odd_zero = coefficient_array_zero(s->coefficient, 1);
    if (even_zero == odd_zero) {
        uint32_t pointer_error = ERR_POINTER_ENTANGLED;

        /* Uncompute the exact pointer interaction before reversing forward. */
        if (!apply_pointer_gate(s)) {
            s->spent = true;
            s->lifecycle = LIFE_SPENT;
            s->error = ERR_INVARIANT;
            release_tags(s);
            return;
        }
        rollback_forward_prefix(s, pointer_error);
        return;
    }
    saved_boundary = even_zero ? 1 : 0;

    /* The boundary bit is retained internally; the quantum pointer is unlatched. */
    if (!apply_pointer_gate(s) || !s->pointer_clear) {
        s->spent = true;
        s->lifecycle = LIFE_SPENT;
        s->error = ERR_INVARIANT;
        release_tags(s);
        return;
    }

    s->error = ERR_NONE;
    if (!execute_inverse_with_fault(s) || !state_is_initial(s)) {
        s->spent = true;
        s->restored = false;
        s->response_ready = false;
        s->lifecycle = LIFE_SPENT;
        s->error = ERR_RESTORATION_FAILED;
        release_tags(s);
        return;
    }

    s->boundary_value = saved_boundary;
    s->restoration_generation++;
    s->restored = true;
    s->response_ready = true;
    s->lifecycle = LIFE_RESPONSE_READY;
    release_tags(s);
    s->error = ERR_NONE;
}

static uint32_t phase_status(const PhaseQemuV1State *s)
{
    uint32_t status = 0;

    if (!s->spent && !s->leased &&
        (!s->prepared || (s->restored && state_is_initial(s)))) {
        status |= ST_CANONICAL;
    }
    if (s->leased) {
        status |= ST_LEASED;
    }
    if (s->prepared) {
        status |= ST_PREPARED;
    }
    if (s->source_isolated) {
        status |= ST_SOURCE_ISOLATED;
    }
    if (s->descriptor_sealed) {
        status |= ST_DESCRIPTOR_SEALED;
    }
    if (s->response_ready && !s->snapshot_lineage) {
        status |= ST_RESPONSE_READY;
    }
    if (s->spent) {
        status |= ST_SPENT;
    }
    if (s->restored) {
        status |= ST_RESTORED;
    }
    if (s->carrier_present) {
        status |= ST_CARRIER_PRESENT;
    }
    if (s->snapshot_lineage) {
        status |= ST_SNAPSHOT_LINEAGE;
    }
    if (s->pointer_clear) {
        status |= ST_POINTER_CLEAR;
    }
    return status;
}

static uint32_t phase_capabilities(const PhaseQemuV1State *s)
{
    uint32_t capabilities = CAP_NOMINAL_COMMAND_TAGS |
                            CAP_EXACT_DYADIC_STATE |
                            CAP_PUBLIC_DESCRIPTOR |
                            CAP_SOURCE_ISOLATION |
                            CAP_ATOMIC_TRANSACTION |
                            CAP_PARITY_POINTER |
                            CAP_PUBLIC_ADJOINT |
                            CAP_RESTORED_REUSE |
                            CAP_CROSS_KERR |
                            CAP_MIGRATION_SHAM_MARKING;

    if (s->fault_mode != FAULT_NONE) {
        capabilities |= CAP_TEST_FAULTS_ACTIVE;
    }
    return capabilities;
}

static void phase_command(PhaseQemuV1State *s, uint32_t command)
{
    uint32_t request_error;

    s->error = ERR_NONE;

    switch (command) {
    case CMD_LEASE:
        if (s->snapshot_lineage) {
            s->error = ERR_SNAPSHOT_LINEAGE;
            return;
        }
        if (!s->carrier_present) {
            s->error = ERR_CARRIER_ABSENT;
            return;
        }
        if (s->leased || s->spent || s->response_ready ||
            s->request_owner_tag == 0 || s->request_program_tag == 0 ||
            (s->lifecycle != LIFE_EMPTY && s->lifecycle != LIFE_REUSABLE)) {
            s->error = ERR_BAD_STATE;
            return;
        }
        if (s->request_generation != s->generation) {
            s->error = ERR_GENERATION_MISMATCH;
            return;
        }
        s->owner_tag = s->request_owner_tag;
        s->program_tag = s->request_program_tag;
        s->leased = true;
        s->lifecycle = s->prepared ? LIFE_ISOLATED : LIFE_LEASED;
        break;
    case CMD_PREPARE:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (s->prepared || s->generation != 1 ||
            s->lifecycle != LIFE_LEASED) {
            s->error = ERR_BAD_STATE;
            return;
        }
        if (!s->carrier_present) {
            s->error = ERR_CARRIER_ABSENT;
            return;
        }
        memset(s->coefficient, 0, sizeof(s->coefficient));
        s->coefficient[coefficient_index(0, 2)] = 1; /* |1,0,1,0> */
        s->denominator_power = 0;
        s->pointer_clear = true;
        s->prepared = true;
        s->source_isolated = false;
        s->restored = false;
        s->lifecycle = LIFE_PREPARED;
        update_peak_bits(s);
        if (!norm_is_one(s->coefficient, s->denominator_power)) {
            s->error = ERR_INVARIANT;
        }
        break;
    case CMD_ISOLATE_SOURCE:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (!s->prepared || s->source_isolated ||
            s->lifecycle != LIFE_PREPARED) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->source_isolated = true;
        s->lifecycle = LIFE_ISOLATED;
        break;
    case CMD_SEAL_DESCRIPTOR:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (!s->source_isolated || s->lifecycle != LIFE_ISOLATED ||
            s->descriptor_sealed ||
            !descriptor_valid(s)) {
            s->error = ERR_DESCRIPTOR_INVALID;
            return;
        }
        s->descriptor_fingerprint = descriptor_fingerprint(s);
        s->descriptor_sealed = true;
        s->lifecycle = LIFE_SEALED;
        break;
    case CMD_EXECUTE_ATOMIC:
        execute_atomic(s);
        break;
    case CMD_BEGIN_REUSE:
        if (s->snapshot_lineage) {
            s->error = ERR_SNAPSHOT_LINEAGE;
            return;
        }
        if (s->spent || !s->restored || s->leased ||
            (s->lifecycle != LIFE_RESPONSE_READY &&
             s->lifecycle != LIFE_REUSABLE)) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->response_ready = false;
        s->boundary_value = 0;
        clear_descriptor(s);
        s->request_owner_tag = 0;
        s->request_program_tag = 0;
        s->request_generation = 0;
        s->generation++;
        s->lifecycle = LIFE_REUSABLE;
        break;
    case CMD_SNAPSHOT:
        s->error = ERR_SNAPSHOT_REJECTED;
        break;
    default:
        s->error = ERR_BAD_ARGUMENT;
        break;
    }
}

static uint64_t phase_mmio_read(void *opaque, hwaddr address, unsigned size)
{
    PhaseQemuV1State *s = opaque;

    (void)size;
    switch (address) {
    case REG_MAGIC:
        return PHASE_V1_MAGIC;
    case REG_ABI:
        return PHASE_V1_ABI;
    case REG_BACKEND:
        return PHASE_V1_BACKEND;
    case REG_CAPABILITIES:
        return phase_capabilities(s);
    case REG_STATUS:
        return phase_status(s);
    case REG_ERROR:
        return s->error;
    case REG_GENERATION:
        return s->generation;
    case REG_RESTORATION_GENERATION:
        return s->restoration_generation;
    case REG_ARG0:
        return s->arg0;
    case REG_ARG1:
        return s->arg1;
    case REG_LIFECYCLE:
        return s->lifecycle;
    case REG_VIRTUAL_CYCLES:
        return s->virtual_cycles;
    case REG_BOUNDARY_VALUE:
        return s->response_ready && !s->snapshot_lineage ?
               s->boundary_value : PHASE_V1_LOCKED_BOUNDARY;
    case REG_RESOURCE_COEFFICIENT_CELLS:
        return PHASE_V1_COEFFICIENT_CELLS;
    case REG_RESOURCE_SCRATCH_CELLS:
        return PHASE_V1_COEFFICIENT_CELLS;
    case REG_RESOURCE_FORWARD_GATES:
        return s->forward_gates;
    case REG_RESOURCE_INVERSE_GATES:
        return s->inverse_gates;
    case REG_RESOURCE_POINTER_GATES:
        return s->pointer_gates;
    case REG_RESOURCE_PEAK_BITS:
        return s->peak_coefficient_bits;
    case REG_REQUEST_OWNER:
        return s->request_owner_tag;
    case REG_REQUEST_PROGRAM:
        return s->request_program_tag;
    case REG_REQUEST_GENERATION:
        return s->request_generation;
    case REG_DESCRIPTOR_INDEX:
        return s->descriptor_index;
    case REG_DESCRIPTOR_LENGTH:
        return s->descriptor_length;
    case REG_BOUNDARY_MODE:
        return s->boundary_mode;
    case REG_DESCRIPTOR_FINGERPRINT:
        return s->descriptor_fingerprint;
    case REG_FAULT_MODE:
        return s->fault_mode;
    default:
        return UINT64_MAX;
    }
}

static void phase_mmio_write(void *opaque, hwaddr address, uint64_t value,
                             unsigned size)
{
    PhaseQemuV1State *s = opaque;
    uint32_t request_error;

    (void)size;
    switch (address) {
    case REG_ARG0:
        s->arg0 = value;
        break;
    case REG_ARG1:
        s->arg1 = value;
        break;
    case REG_COMMAND:
        phase_command(s, value);
        break;
    case REG_REQUEST_OWNER:
        s->request_owner_tag = value;
        break;
    case REG_REQUEST_PROGRAM:
        s->request_program_tag = value;
        break;
    case REG_REQUEST_GENERATION:
        s->request_generation = value;
        break;
    case REG_DESCRIPTOR_INDEX:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
        } else if (s->descriptor_sealed ||
            s->lifecycle != LIFE_ISOLATED || value >= PHASE_V1_MAX_DESCRIPTOR) {
            s->error = ERR_BAD_STATE;
        } else {
            s->descriptor_index = value;
        }
        break;
    case REG_DESCRIPTOR_WORD:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
        } else if (s->descriptor_sealed ||
            s->lifecycle != LIFE_ISOLATED ||
            s->descriptor_index >= PHASE_V1_MAX_DESCRIPTOR) {
            s->error = ERR_BAD_STATE;
        } else if (value < GATE_A || value > GATE_KERR_IDENTITY_SHAM) {
            s->error = ERR_DESCRIPTOR_INVALID;
        } else {
            s->descriptor[s->descriptor_index] = value;
        }
        break;
    case REG_DESCRIPTOR_LENGTH:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
        } else if (s->descriptor_sealed ||
            s->lifecycle != LIFE_ISOLATED || value == 0 ||
            value > PHASE_V1_MAX_DESCRIPTOR) {
            s->error = ERR_DESCRIPTOR_INVALID;
        } else {
            s->descriptor_length = value;
        }
        break;
    case REG_BOUNDARY_MODE:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
        } else if (s->descriptor_sealed ||
            s->lifecycle != LIFE_ISOLATED || value >= 4) {
            s->error = ERR_DESCRIPTOR_INVALID;
        } else {
            s->boundary_mode = value;
        }
        break;
    default:
        break;
    }
}

static const MemoryRegionOps phase_mmio_ops = {
    .read = phase_mmio_read,
    .write = phase_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 8,
        .unaligned = false,
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 8,
    },
};

static void phase_qemu_v1_reset(DeviceState *device)
{
    PhaseQemuV1State *s = PHASE_QEMU_V1(device);

    s->leased = false;
    s->prepared = false;
    s->source_isolated = false;
    s->descriptor_sealed = false;
    s->response_ready = false;
    s->spent = false;
    s->restored = false;
    s->snapshot_lineage = false;
    s->pointer_clear = true;
    s->coupler_active = false;
    s->error = ERR_NONE;
    s->lifecycle = LIFE_EMPTY;
    s->owner_tag = 0;
    s->program_tag = 0;
    s->request_owner_tag = 0;
    s->request_program_tag = 0;
    s->request_generation = 0;
    s->generation = 1;
    s->restoration_generation = 0;
    s->arg0 = 0;
    s->arg1 = 0;
    s->boundary_value = 0;
    s->denominator_power = 0;
    s->descriptor_fingerprint = 0;
    s->virtual_cycles = 0;
    s->forward_gates = 0;
    s->inverse_gates = 0;
    s->pointer_gates = 0;
    s->peak_coefficient_bits = 0;
    memset(s->coefficient, 0, sizeof(s->coefficient));
    clear_descriptor(s);
}

static int phase_qemu_v1_post_load(void *opaque, int version_id)
{
    PhaseQemuV1State *s = opaque;

    (void)version_id;
    if (s->carrier_present != s->configured_carrier_present ||
        s->fault_mode != s->configured_fault_mode ||
        s->fault_mode > FAULT_REORDERED_INVERSE_PREFIX ||
        s->generation == 0 ||
        s->denominator_power > PHASE_V1_MAX_DENOMINATOR_POWER ||
        s->descriptor_length > PHASE_V1_MAX_DESCRIPTOR ||
        s->descriptor_index >= PHASE_V1_MAX_DESCRIPTOR ||
        s->boundary_mode >= 4 ||
        s->lifecycle > LIFE_SHAM ||
        !coefficient_bound_valid(s->coefficient) ||
        (s->prepared && !norm_is_one(s->coefficient,
                                     s->denominator_power)) ||
        (!s->prepared &&
         (!coefficient_array_zero(s->coefficient, 0) ||
          !coefficient_array_zero(s->coefficient, 1))) ||
        (s->descriptor_sealed &&
         (!descriptor_valid(s) ||
          s->descriptor_fingerprint != descriptor_fingerprint(s))) ||
        (s->leased && (s->owner_tag == 0 || s->program_tag == 0)) ||
        (s->response_ready && (!s->restored || s->spent || s->leased)) ||
        (s->restored && !state_is_initial(s)) ||
        s->forward_cursor > s->descriptor_length ||
        s->inverse_cursor > s->descriptor_length ||
        s->committed_prefix > s->descriptor_length ||
        s->coupler_active) {
        return -EINVAL;
    }

    /* Every incoming stream is a reload sham, never accepted in-place reuse. */
    s->snapshot_lineage = true;
    s->response_ready = false;
    s->lifecycle = LIFE_SHAM;
    release_tags(s);
    return 0;
}

static const VMStateDescription vmstate_phase_qemu_v1 = {
    .name = "phase-qemu-v1",
    .version_id = 1,
    .minimum_version_id = 1,
    .post_load = phase_qemu_v1_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, PhaseQemuV1State),
        VMSTATE_BOOL(carrier_present, PhaseQemuV1State),
        VMSTATE_BOOL(leased, PhaseQemuV1State),
        VMSTATE_BOOL(prepared, PhaseQemuV1State),
        VMSTATE_BOOL(source_isolated, PhaseQemuV1State),
        VMSTATE_BOOL(descriptor_sealed, PhaseQemuV1State),
        VMSTATE_BOOL(response_ready, PhaseQemuV1State),
        VMSTATE_BOOL(spent, PhaseQemuV1State),
        VMSTATE_BOOL(restored, PhaseQemuV1State),
        VMSTATE_BOOL(snapshot_lineage, PhaseQemuV1State),
        VMSTATE_BOOL(pointer_clear, PhaseQemuV1State),
        VMSTATE_BOOL(coupler_active, PhaseQemuV1State),
        VMSTATE_UINT32(error, PhaseQemuV1State),
        VMSTATE_UINT32(lifecycle, PhaseQemuV1State),
        VMSTATE_UINT32(owner_tag, PhaseQemuV1State),
        VMSTATE_UINT32(program_tag, PhaseQemuV1State),
        VMSTATE_UINT32(request_owner_tag, PhaseQemuV1State),
        VMSTATE_UINT32(request_program_tag, PhaseQemuV1State),
        VMSTATE_UINT32(request_generation, PhaseQemuV1State),
        VMSTATE_UINT32(generation, PhaseQemuV1State),
        VMSTATE_UINT32(restoration_generation, PhaseQemuV1State),
        VMSTATE_UINT32(arg0, PhaseQemuV1State),
        VMSTATE_UINT32(arg1, PhaseQemuV1State),
        VMSTATE_UINT32(descriptor_index, PhaseQemuV1State),
        VMSTATE_UINT32(descriptor_length, PhaseQemuV1State),
        VMSTATE_UINT32(boundary_mode, PhaseQemuV1State),
        VMSTATE_UINT32(boundary_value, PhaseQemuV1State),
        VMSTATE_UINT32(denominator_power, PhaseQemuV1State),
        VMSTATE_UINT32(forward_cursor, PhaseQemuV1State),
        VMSTATE_UINT32(inverse_cursor, PhaseQemuV1State),
        VMSTATE_UINT32(committed_prefix, PhaseQemuV1State),
        VMSTATE_UINT64(descriptor_fingerprint, PhaseQemuV1State),
        VMSTATE_UINT64(virtual_cycles, PhaseQemuV1State),
        VMSTATE_UINT64(forward_gates, PhaseQemuV1State),
        VMSTATE_UINT64(inverse_gates, PhaseQemuV1State),
        VMSTATE_UINT64(pointer_gates, PhaseQemuV1State),
        VMSTATE_UINT32(peak_coefficient_bits, PhaseQemuV1State),
        VMSTATE_INT64_ARRAY(coefficient, PhaseQemuV1State,
                            PHASE_V1_COEFFICIENT_CELLS),
        VMSTATE_UINT32_ARRAY(descriptor, PhaseQemuV1State,
                             PHASE_V1_MAX_DESCRIPTOR),
        VMSTATE_UINT32(fault_mode, PhaseQemuV1State),
        VMSTATE_END_OF_LIST()
    }
};

static void phase_qemu_v1_realize(PCIDevice *pci_device, Error **errp)
{
    PhaseQemuV1State *s = PHASE_QEMU_V1(pci_device);

    if (s->fault_mode > FAULT_REORDERED_INVERSE_PREFIX) {
        error_setg(errp,
                   "phase-qemu-v1 test-fault-mode must be in [0,3]");
        return;
    }
    s->configured_carrier_present = s->carrier_present;
    s->configured_fault_mode = s->fault_mode;
    memory_region_init_io(&s->mmio, OBJECT(s), &phase_mmio_ops, s,
                          "phase-qemu-v1-mmio", 4 * KiB);
    pci_register_bar(pci_device, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
}

static void phase_qemu_v1_instance_init(Object *object)
{
    PhaseQemuV1State *s = PHASE_QEMU_V1(object);

    s->carrier_present = true;
    s->fault_mode = FAULT_NONE;
}

static const Property phase_qemu_v1_properties[] = {
    DEFINE_PROP_BOOL("carrier-present", PhaseQemuV1State,
                     carrier_present, true),
    DEFINE_PROP_UINT32("test-fault-mode", PhaseQemuV1State,
                       fault_mode, FAULT_NONE),
};

static void phase_qemu_v1_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *device_class = DEVICE_CLASS(klass);
    PCIDeviceClass *pci_class = PCI_DEVICE_CLASS(klass);

    (void)data;
    pci_class->realize = phase_qemu_v1_realize;
    pci_class->vendor_id = PHASE_V1_VENDOR_ID;
    pci_class->device_id = PHASE_V1_DEVICE_ID;
    pci_class->revision = 0x00;
    pci_class->class_id = PCI_CLASS_OTHERS;
    device_class->vmsd = &vmstate_phase_qemu_v1;
    device_class_set_legacy_reset(device_class, phase_qemu_v1_reset);
    device_class_set_props(device_class, phase_qemu_v1_properties);
    set_bit(DEVICE_CATEGORY_MISC, device_class->categories);
}

static const TypeInfo phase_qemu_v1_info = {
    .name = TYPE_PHASE_QEMU_V1,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(PhaseQemuV1State),
    .instance_init = phase_qemu_v1_instance_init,
    .class_init = phase_qemu_v1_class_init,
    .interfaces = (const InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void phase_qemu_v1_register_types(void)
{
    type_register_static(&phase_qemu_v1_info);
}

type_init(phase_qemu_v1_register_types)
