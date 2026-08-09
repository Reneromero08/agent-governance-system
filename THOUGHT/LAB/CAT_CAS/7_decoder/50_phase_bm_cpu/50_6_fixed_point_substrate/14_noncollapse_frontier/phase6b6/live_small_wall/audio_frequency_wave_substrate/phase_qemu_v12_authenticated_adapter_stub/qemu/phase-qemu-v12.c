/*
 * Phase-QEMU V12 authenticated external-adapter protocol device.
 *
 * The guest-visible PCI device is a control and custody plane.  Its public
 * descriptor contains no phase residue, expected answer, inverse, or quantum
 * amplitude.  Test residues enter only through write-only runtime QOM
 * properties after CMD_ARM_PRIVATE.  They are intentionally absent from the
 * BAR, VMState, receipts, and logs.
 *
 * The V12 device preserves the V11 guest register prefix and swappable
 * backend lifecycle while adding a compiled, asynchronous, hardware-
 * disconnected external-adapter test stub.  The test stub authenticates a
 * fixed-size synthetic envelope through a deterministic integrity tag,
 * executes the exact internal algebra only as a model, and returns
 * APPROX_MODEL.  It cannot assert physical return or authorize reuse.
 *
 * The ideal backend materializes the full 96 by 96 density matrix on
 * C_A,R_A,C_B,R_B,K(vac,a,b),R_K over Q(omega), omega^2 = -1-omega.  Queries
 * are applied to that matrix through the controlled-number exponent, never as
 * a direct client-phase shortcut.  The OPEN backend is honest but minimal:
 * zero model strength forwards to the exact engine; nonzero strength is only
 * an analytic APPROX classification stub and can never qualify reuse.  The
 * EXTERNAL is unavailable by default.  It becomes a test-only asynchronous
 * adapter when both test-provider-enabled and test-adapter-enabled are set.
 *
 * This is a deterministic software device model.  It is not evidence of a
 * physical cavity, physical custody, open-system dynamics, or advantage.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "hw/pci/pci.h"
#include "hw/pci/pci_device.h"
#include "hw/qdev-properties.h"
#include "migration/vmstate.h"
#include "qapi/visitor.h"
#include "qemu/module.h"
#include "qemu/units.h"
#include "qom/object.h"

#define TYPE_PHASE_QEMU_V12 "phase-qemu-v12"
OBJECT_DECLARE_SIMPLE_TYPE(PhaseQemuV12State, PHASE_QEMU_V12)

#define PHASE_V12_VENDOR_ID PCI_VENDOR_ID_QEMU
#define PHASE_V12_DEVICE_ID 0x11fc
#define PHASE_V12_REVISION 0x01
#define PHASE_V12_MAGIC 0x50483132u /* PH12 */
#define PHASE_V12_ABI 0x00020000u
#define PHASE_V12_BAR_SIZE (4 * KiB)

#define PHASE_V12_BACKEND_IDEAL 0x0b01u
#define PHASE_V12_BACKEND_OPEN 0x0b02u
#define PHASE_V12_BACKEND_EXTERNAL 0x0b80u

#define PHASE_V12_CLIENT_DIM 2u
#define PHASE_V12_CARRIER_DIM 3u
#define PHASE_V12_SYSTEM_DIM 96u
#define PHASE_V12_STATE_CELLS \
    (PHASE_V12_SYSTEM_DIM * PHASE_V12_SYSTEM_DIM)
#define PHASE_V12_DESCRIPTOR_WORDS 8u
#define PHASE_V12_BOUNDARY_WORDS 16u
#define PHASE_V12_PRIVATE_SLOTS 2u
#define PHASE_V12_PRIVATE_READY_ALL 0x3u
#define PHASE_V12_DENSITY_DENOM_POWER 3u /* eight pure components */
#define PHASE_V12_LOCKED_BOUNDARY UINT64_MAX
#define PHASE_V12_INVALID_OBSERVATION UINT32_MAX
#define PHASE_V12_ACTION_PER_QUERY_Q40 UINT64_C(2302811768238)
#define PHASE_V12_MIGRATION_MARKER 0x5031324du /* P12M */
#define PHASE_V12_BOUNDARY_HEADER UINT64_C(0x5031324200010080)
#define PHASE_V12_RESOURCE_UNKNOWN UINT64_MAX
#define PHASE_V12_RESOURCE_SCHEMA 0x00020001u /* V12.1, UINT64_MAX unknown */
#define PHASE_V12_BACKEND_PENDING UINT32_C(0x100)
#define PHASE_V12_ADAPTER_TAG_BITS 21u
#define PHASE_V12_ADAPTER_TAG_MASK ((UINT64_C(1) << PHASE_V12_ADAPTER_TAG_BITS) - 1)

/*
 * Architecture promotion gate:
 *   qemu_device_implemented = compiled TYPE_PHASE_QEMU_V12 only
 *   common_guest_visible_contract = this single PCI ABI for every backend
 *   reintegration_gate = standalone algebra/twins never qualify by themselves
 */
#define PHASE_V12_STANDALONE_TWIN_QUALIFIES 0

enum PhaseV12Register {
    REG_MAGIC = 0x000,
    REG_ABI = 0x004,
    REG_BACKEND = 0x008,
    REG_CAPABILITIES_LO = 0x00c,
    REG_STATUS_LO = 0x010,
    REG_ERROR = 0x014,
    REG_GENERATION = 0x018,
    REG_EXACT_RETURN_GENERATION = 0x01c,
    REG_ARG0 = 0x020,
    REG_ARG1 = 0x024,
    REG_COMMAND = 0x028,
    REG_LIFECYCLE = 0x02c,
    REG_VIRTUAL_CYCLES = 0x030,
    REG_BOUNDARY_COMMIT_COOKIE = 0x040,
    REG_RESOURCE_STATE_CELLS = 0x048,
    REG_RESOURCE_SCRATCH_CELLS = 0x04c,
    REG_RESOURCE_QUERY_APPLICATIONS = 0x050,
    REG_RESOURCE_RETURN_CHECKS = 0x058,
    REG_RESOURCE_ENVIRONMENT_OPS = 0x060,
    REG_RESOURCE_PEAK_BITS = 0x068,
    REG_REQUEST_OWNER = 0x070,
    REG_REQUEST_PROGRAM = 0x074,
    REG_REQUEST_GENERATION = 0x078,
    REG_DESCRIPTOR_INDEX = 0x07c,
    REG_DESCRIPTOR_WORD = 0x080,
    REG_DESCRIPTOR_LENGTH = 0x084,
    REG_BOUNDARY_SCHEMA = 0x088,
    REG_DESCRIPTOR_FINGERPRINT = 0x090,
    REG_FAULT_MODE = 0x098,
    REG_CAPABILITIES_HI = 0x0a0,
    REG_STATUS_HI = 0x0a4,
    REG_PRIVATE_QUERY_SLOTS = 0x0a8,
    REG_PRIVATE_READY_MASK = 0x0ac,
    REG_RETURN_CLASS = 0x0b0,
    REG_BOUNDARY_LENGTH = 0x0b4,
    REG_ALLOCATION_ID_LO = 0x0b8,
    REG_ALLOCATION_ID_HI = 0x0c0,
    REG_CUSTODY_EPOCH = 0x0c8,
    REG_PREPARATION_RECEIPT_LO = 0x0d0,
    REG_PREPARATION_RECEIPT_HI = 0x0d8,
    REG_RETURN_RECEIPT_LO = 0x0e0,
    REG_RETURN_RECEIPT_HI = 0x0e8,
    REG_RESOURCE_SECRET_STORAGE_BITS = 0x0f0,
    REG_RESOURCE_CONTROL_WORDS = 0x0f8,
    REG_RESOURCE_PREPARATION_OPS = 0x100,
    REG_RESOURCE_CERTIFICATION_OPS = 0x108,
    REG_RESOURCE_LOGICAL_QUERIES = 0x110,
    REG_RESOURCE_DURATION_FS = 0x118,
    REG_RESOURCE_PORT_BANDWIDTH_HZ = 0x120,
    REG_RESOURCE_ACTION_Q40_RAD = 0x128,
    REG_RESOURCE_MEAN_ENERGY_ATTOJ = 0x130,
    REG_RESOURCE_LOSS_Q63 = 0x138,
    REG_RESOURCE_DEPHASING_Q63 = 0x140,
    REG_RESOURCE_ENV_HISTORY_CELLS = 0x148,
    REG_RESOURCE_CUSTODY_TRANSITIONS = 0x150,
    REG_RESOURCE_REUSE_COUNT = 0x158,
    REG_RESOURCE_DISCARDED_TRIALS = 0x160,
    REG_RESOURCE_OUTPUT_HOLD_FS = 0x168,
    REG_RESOURCE_MAINTENANCE_OPS = 0x170,
    REG_RESOURCE_PRECISION_BITS = 0x178,
    REG_RESOURCE_SCHEMA = 0x180,
    REG_RESOURCE_DIGEST_LO = 0x188,
    REG_RESOURCE_DIGEST_HI = 0x190,
    REG_RESOURCE_COMPILER_OPS = 0x198,
    REG_RESOURCE_CONTROLLER_OPS = 0x1a0,
    REG_RESOURCE_CONSTRUCTION_OPS = 0x1a8,
    REG_RESOURCE_SECRET_ENTROPY_BITS = 0x1b0,
    REG_RESOURCE_CARRIER_PHOTON_NUMBER = 0x1b8,
    REG_ADAPTER_STATE = 0x1c0,
    REG_ADAPTER_AUTH_ACCEPTED = 0x1c8,
    REG_ADAPTER_AUTH_REJECTED = 0x1d0,
    REG_ADAPTER_DISPATCHES = 0x1d8,
    REG_ADAPTER_COMPLETIONS = 0x1e0,
    REG_ADAPTER_CANCELS = 0x1e8,
    REG_ADAPTER_VIRTUAL_TICK = 0x1f0,
    REG_ADAPTER_DEADLINE_TICK = 0x1f8,
    REG_BOUNDARY_BASE = 0x200,
    REG_BOUNDARY_LAST = 0x278,
};

enum PhaseV12Command {
    CMD_LEASE = 1,
    CMD_PREPARE = 2,
    CMD_ISOLATE_SOURCE = 3,
    CMD_SEAL_DESCRIPTOR = 4,
    CMD_EXECUTE_ATOMIC = 5,
    CMD_BEGIN_REUSE = 6,
    CMD_SNAPSHOT = 7,
    CMD_ARM_PRIVATE = 8,
    CMD_ACK_RESPONSE = 9,
    CMD_ABORT_PREEXEC = 10,
    CMD_POLL_EXTERNAL = 11,
    CMD_CANCEL_EXTERNAL = 12,
};

enum PhaseV12Lifecycle {
    LIFE_EMPTY = 0,
    LIFE_LEASED = 1,
    LIFE_PREPARED = 2,
    LIFE_ISOLATED = 3,
    LIFE_SEALED = 4,
    LIFE_PRIVATE_ARMED = 5,
    LIFE_PRIVATE_READY = 6,
    LIFE_EXECUTING = 7,
    LIFE_VERIFYING_RETURN = 8,
    LIFE_RESPONSE_READY = 9,
    LIFE_RESPONSE_ACKED = 10,
    LIFE_REUSABLE = 11,
    LIFE_SPENT = 12,
    LIFE_SHAM = 13,
};

enum PhaseV12ReturnClass {
    RETURN_NONE = 0,
    RETURN_EXACT_FORMAL = 1,
    RETURN_APPROX_MODEL = 2,
    RETURN_FAILED = 3,
    RETURN_STATISTICAL_ONLY = 4,
};

enum PhaseV12Error {
    ERR_NONE = 0,
    ERR_BAD_STATE = 1,
    ERR_BAD_ARGUMENT = 2,
    ERR_TAG_MISMATCH = 3,
    ERR_GENERATION_MISMATCH = 4,
    ERR_DESCRIPTOR_INVALID = 5,
    ERR_SOURCE_NOT_ISOLATED = 6,
    ERR_CARRIER_ABSENT = 7,
    ERR_PORT_NOT_CLEAR = 8,
    ERR_RETURN_FAILED = 9,
    ERR_OVERFLOW = 10,
    ERR_INVARIANT = 11,
    ERR_SNAPSHOT_REJECTED = 12,
    ERR_SNAPSHOT_LINEAGE = 13,
    ERR_RESPONSE_LOCKED = 14,
    ERR_BACKEND_UNAVAILABLE = 15,
    ERR_BACKEND_MISMATCH = 16,
    ERR_PRIVATE_NOT_ARMED = 17,
    ERR_PRIVATE_INCOMPLETE = 18,
    ERR_PRIVATE_LINEAGE = 19,
    ERR_PRIVATE_DUPLICATE = 20,
    ERR_SECRET_SMUGGLE = 21,
    ERR_CLIENT_REFERENCE_FAILED = 22,
    ERR_CARRIER_REFERENCE_FAILED = 23,
    ERR_ENVIRONMENT_NOT_FACTORED = 24,
    ERR_TOLERANCE_EXCEEDED = 25,
    ERR_RESPONSE_NOT_ACKED = 26,
    ERR_REUSE_NOT_QUALIFIED = 27,
    ERR_ADAPTER_TIMEOUT = 28,
    ERR_PORT_COMMIT_FAILED = 29,
    ERR_RESOURCE_UNSEALED = 30,
    ERR_CUSTODY_MISMATCH = 31,
    ERR_ADAPTER_AUTH_BINDING = 32,
    ERR_ADAPTER_REPLAY = 33,
    ERR_ADAPTER_ORDER = 34,
    ERR_ADAPTER_EXPIRED = 35,
    ERR_ADAPTER_SLOT = 36,
    ERR_ADAPTER_CANCELED = 37,
};

enum PhaseV12Status {
    ST_CANONICAL = 1u << 0,
    ST_LEASED = 1u << 1,
    ST_PREPARED = 1u << 2,
    ST_SOURCE_ISOLATED = 1u << 3,
    ST_DESCRIPTOR_SEALED = 1u << 4,
    ST_RESPONSE_READY = 1u << 5,
    ST_SPENT = 1u << 6,
    ST_EXACT_RETURN = 1u << 7,
    ST_CARRIER_PRESENT = 1u << 8,
    ST_SNAPSHOT_LINEAGE = 1u << 9,
    ST_PORT_CLEAR = 1u << 10,
    ST_PRIVATE_ARMED = 1u << 11,
    ST_PRIVATE_READY = 1u << 12,
    ST_OUTPUTS_HELD = 1u << 13,
    ST_RETURN_VERIFIED = 1u << 14,
    ST_RESPONSE_ACKED = 1u << 15,
    ST_SAME_ALLOCATION = 1u << 16,
    ST_ENV_FACTORED = 1u << 17,
    ST_NOISY_BACKEND = 1u << 18,
    ST_EXTERNAL_BACKEND = 1u << 19,
    ST_PREP_RECEIPT_READY = 1u << 20,
    ST_RESOURCE_SEALED = 1u << 21,
    ST_SHAM = 1u << 22,
    ST_ADAPTER_AUTHENTICATED = 1u << 23,
    ST_ADAPTER_PENDING = 1u << 24,
    ST_EXTERNAL_REUSE_FORBIDDEN = 1u << 25,
};

enum PhaseV12Capability {
    CAP_NOMINAL_COMMAND_TAGS = 1u << 0,
    CAP_EXACT_CYCLO3_STATE = 1u << 1,
    CAP_PUBLIC_FIXED_DESCRIPTOR = 1u << 2,
    CAP_SOURCE_ISOLATION = 1u << 3,
    CAP_ATOMIC_TRANSACTION = 1u << 4,
    CAP_DUAL_RAIL_N1 = 1u << 5,
    CAP_CARRIER_REFERENCE = 1u << 6,
    CAP_CLIENT_REFERENCES = 1u << 7,
    CAP_TWO_LATE_BOUND_QUERIES = 1u << 8,
    CAP_PRIVATE_QOM_PROVIDER = 1u << 9,
    CAP_SWAPPABLE_BACKEND = 1u << 10,
    CAP_EXACT_RETURN = 1u << 11,
    CAP_APPROX_RETURN = 1u << 12,
    CAP_OPEN_CLASSIFICATION_STUB = 1u << 13,
    CAP_COHERENT_PORT_MODEL = 1u << 14,
    CAP_RESOURCE_VECTOR = 1u << 15,
    CAP_SAME_ALLOCATION_RECEIPT = 1u << 16,
    CAP_MIGRATION_SHAM = 1u << 17,
    CAP_TEST_FAULTS_ACTIVE = 1u << 18,
    CAP_TEST_QOM_OBSERVER = 1u << 19,
    CAP_CONSTANT_PUBLIC_ENVELOPE = 1u << 20,
    CAP_EXTERNAL_ADAPTER_SLOT = 1u << 21,
    CAP_EXTERNAL_AUTH_TEST_STUB = 1u << 22,
    CAP_EXTERNAL_ASYNC_LIFECYCLE = 1u << 23,
    CAP_EXTERNAL_REUSE_FORBIDDEN = 1u << 24,
    CAP_PHYSICAL_RESOURCES_UNKNOWN = 1u << 25,
};

enum PhaseV12AdapterState {
    ADAPTER_DISCONNECTED = 0,
    ADAPTER_READY = 1,
    ADAPTER_WAITING_A = 2,
    ADAPTER_WAITING_B = 3,
    ADAPTER_COMPLETE = 4,
    ADAPTER_CANCELED = 5,
    ADAPTER_TIMED_OUT = 6,
    ADAPTER_SHAM = 7,
};

enum PhaseV12AdapterMode {
    ADAPTER_MODE_COMPLETE = 0,
    ADAPTER_MODE_TIMEOUT = 1,
    ADAPTER_MODE_MAX = ADAPTER_MODE_TIMEOUT,
};

enum PhaseV12FaultMode {
    FAULT_NONE = 0,
    FAULT_VACUUM_CARRIER = 1,
    FAULT_NON_EIGENSTATE_CARRIER = 2,
    FAULT_DIFFERENTIAL_COUPLING = 3,
    FAULT_PORT_STUCK = 4,
    FAULT_ENVIRONMENT_TAG = 5,
    FAULT_STATE_CORRUPT = 6,
    FAULT_RESOURCE_UNSEALED = 7,
    FAULT_MAX = FAULT_RESOURCE_UNSEALED,
};

enum PhaseV12CarrierBasis {
    K_VACUUM = 0,
    K_RAIL_A = 1,
    K_RAIL_B = 2,
};

typedef struct PhaseV12Cyclo3 {
    int64_t one;
    int64_t omega;
} PhaseV12Cyclo3;

typedef struct PhaseV12BackendOps {
    uint32_t id;
    const char *name;
    bool (*lease_preflight)(PhaseQemuV12State *s);
    bool (*prepare)(PhaseQemuV12State *s);
    bool (*supply_clients)(PhaseQemuV12State *s);
    uint32_t (*execute)(PhaseQemuV12State *s);
    uint32_t (*poll)(PhaseQemuV12State *s);
    uint32_t (*verify_return)(PhaseQemuV12State *s);
    void (*cancel)(PhaseQemuV12State *s);
    void (*sanitize)(PhaseQemuV12State *s);
} PhaseV12BackendOps;

struct PhaseQemuV12State {
    PCIDevice parent_obj;
    MemoryRegion mmio;

    const PhaseV12BackendOps *ops;
    bool realized;
    bool carrier_present;
    bool configured_carrier_present;
    bool test_provider_enabled;
    bool configured_test_provider_enabled;
    bool test_adapter_enabled;
    bool configured_test_adapter_enabled;
    uint32_t backend_id;
    uint32_t configured_backend_id;
    uint32_t open_model_q32;
    uint32_t configured_open_model_q32;
    uint32_t fault_mode;
    uint32_t configured_fault_mode;
    uint32_t test_adapter_mode;
    uint32_t configured_test_adapter_mode;

    bool leased;
    bool prepared;
    bool source_isolated;
    bool descriptor_sealed;
    bool private_armed;
    bool response_ready;
    bool response_acked;
    bool spent;
    bool restored;
    bool return_verified;
    bool snapshot_lineage;
    bool port_clear;
    bool outputs_held;
    bool env_factored;
    bool resource_sealed;
    bool reuse_qualified;
    bool same_backing;
    bool migration_sham_latched;
    bool adapter_authenticated_lineage;

    uint32_t error;
    uint32_t lifecycle;
    uint32_t return_class;
    uint32_t owner_tag;
    uint32_t program_tag;
    uint32_t request_owner_tag;
    uint32_t request_program_tag;
    uint32_t request_generation;
    uint32_t generation;
    uint32_t exact_return_generation;
    uint32_t arg0;
    uint32_t arg1;
    uint32_t descriptor_index;
    uint32_t descriptor_length;
    uint32_t boundary_schema;
    uint32_t private_ready_mask;
    uint32_t arm_generation;
    uint32_t density_denom_power;
    uint32_t migration_marker;
    uint32_t adapter_state;

    uint64_t descriptor_fingerprint;
    uint64_t arm_nonce;
    uint64_t virtual_cycles;
    uint64_t allocation_id_lo;
    uint64_t allocation_id_hi;
    uint64_t custody_epoch;
    uint64_t preparation_receipt_lo;
    uint64_t preparation_receipt_hi;
    uint64_t return_receipt_lo;
    uint64_t return_receipt_hi;
    uint64_t output_receipt_lo;
    uint64_t output_receipt_hi;
    uint64_t resource_digest_lo;
    uint64_t resource_digest_hi;
    uint64_t sealed_resource_control_words;
    uint64_t adapter_virtual_tick;
    uint64_t adapter_deadline_tick;
    uint64_t adapter_auth_accepted;
    uint64_t adapter_auth_rejected;
    uint64_t adapter_dispatches;
    uint64_t adapter_completions;
    uint64_t adapter_cancels;
    uint64_t adapter_envelope[PHASE_V12_PRIVATE_SLOTS];

    uint64_t resource_query_applications;
    uint64_t resource_return_checks;
    uint64_t resource_environment_ops;
    uint64_t resource_control_words;
    uint64_t resource_preparation_ops;
    uint64_t resource_certification_ops;
    uint64_t resource_logical_queries;
    uint64_t resource_duration_fs;
    uint64_t resource_port_bandwidth_hz;
    uint64_t resource_action_q40_rad;
    uint64_t resource_mean_energy_attoj;
    uint64_t resource_loss_q63;
    uint64_t resource_dephasing_q63;
    uint64_t resource_env_history_cells;
    uint64_t resource_custody_transitions;
    uint64_t resource_reuse_count;
    uint64_t resource_discarded_trials;
    uint64_t resource_output_hold_fs;
    uint64_t resource_maintenance_ops;

    uint64_t prepare_count;
    uint64_t client_supply_count;
    uint32_t observed_phase[PHASE_V12_PRIVATE_SLOTS];
    uint32_t observed_same_backing;
    uint32_t observed_kr_return;
    uint32_t observed_factorized;
    uint32_t observed_port_clear;
    uint32_t observed_env_factored;

    uint8_t private_residue[PHASE_V12_PRIVATE_SLOTS];
    bool private_bound[PHASE_V12_PRIVATE_SLOTS];
    uintptr_t backing_address;

    uint32_t descriptor[PHASE_V12_DESCRIPTOR_WORDS];
    uint64_t boundary[PHASE_V12_BOUNDARY_WORDS];
    PhaseV12Cyclo3 density[PHASE_V12_STATE_CELLS];
    PhaseV12Cyclo3 scratch[PHASE_V12_STATE_CELLS];
};

static const uint32_t phase_v12_public_descriptor[PHASE_V12_DESCRIPTOR_WORDS] = {
    UINT32_C(0x50313144), UINT32_C(0x00010008),
    UINT32_C(0x00000002), UINT32_C(0x00020102),
    UINT32_C(0x00020011), UINT32_C(0x00030021),
    UINT32_C(0x00000003), UINT32_C(0x00010001),
};

static uint64_t phase_v12_allocation_serial = UINT64_C(1);
static uint64_t phase_v12_transaction_serial = UINT64_C(1);
static uint64_t phase_v12_arm_serial = UINT64_C(1);

static uint64_t fnv_u64(uint64_t hash, uint64_t value)
{
    unsigned byte;

    for (byte = 0; byte < 8; byte++) {
        hash ^= (value >> (8 * byte)) & 0xff;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t receipt_hash(uint64_t domain, uint64_t first,
                             uint64_t second, uint64_t third)
{
    uint64_t hash = UINT64_C(14695981039346656037);

    hash = fnv_u64(hash, domain);
    hash = fnv_u64(hash, first);
    hash = fnv_u64(hash, second);
    return fnv_u64(hash, third);
}

static bool counter_can_add(PhaseQemuV12State *s, uint64_t value,
                            uint64_t addend)
{
    if (value == PHASE_V12_RESOURCE_UNKNOWN ||
        addend >= PHASE_V12_RESOURCE_UNKNOWN - value) {
        s->error = ERR_OVERFLOW;
        return false;
    }
    return true;
}

/*
 * Deterministic test-only integrity tag for the disconnected adapter stub.
 * This is not a cryptographic MAC or AEAD construction.  It exists only to
 * exercise the common device's private binding, replay, order, expiry, and
 * asynchronous lifecycle.  A hardware-connected successor must replace it
 * with an authenticated confidential transport and real key custody.
 */
static uint64_t adapter_test_tag(const PhaseQemuV12State *s,
                                 uint64_t payload)
{
    uint64_t hash = receipt_hash(UINT64_C(0x4d323730),
                                 s->allocation_id_lo,
                                 s->allocation_id_hi,
                                 s->descriptor_fingerprint);

    hash = fnv_u64(hash, s->arm_nonce);
    hash = fnv_u64(hash, s->backend_id);
    return fnv_u64(hash, payload) & PHASE_V12_ADAPTER_TAG_MASK;
}

static unsigned system_index(unsigned ca, unsigned ra, unsigned cb,
                             unsigned rb, unsigned carrier, unsigned rk)
{
    return (((((ca * 2 + ra) * 2 + cb) * 2 + rb) * 3 + carrier) * 2 + rk);
}

static void system_decode(unsigned index, unsigned *ca, unsigned *ra,
                          unsigned *cb, unsigned *rb, unsigned *carrier,
                          unsigned *rk)
{
    *rk = index % 2;
    index /= 2;
    *carrier = index % 3;
    index /= 3;
    *rb = index % 2;
    index /= 2;
    *cb = index % 2;
    index /= 2;
    *ra = index % 2;
    index /= 2;
    *ca = index;
}

static bool cyclo_equal(PhaseV12Cyclo3 left, PhaseV12Cyclo3 right)
{
    return left.one == right.one && left.omega == right.omega;
}

static bool cyclo_zero(PhaseV12Cyclo3 value)
{
    return value.one == 0 && value.omega == 0;
}

static PhaseV12Cyclo3 cyclo_power(int exponent)
{
    int residue = exponent % 3;

    if (residue < 0) {
        residue += 3;
    }
    switch (residue) {
    case 0:
        return (PhaseV12Cyclo3) { .one = 1, .omega = 0 };
    case 1:
        return (PhaseV12Cyclo3) { .one = 0, .omega = 1 };
    default:
        return (PhaseV12Cyclo3) { .one = -1, .omega = -1 };
    }
}

static bool cyclo_multiply_power(PhaseV12Cyclo3 *value, int exponent)
{
    int residue = exponent % 3;
    __int128 one;
    __int128 omega;

    if (residue < 0) {
        residue += 3;
    }
    if (residue == 0 || cyclo_zero(*value)) {
        return true;
    }
    if (residue == 1) {
        one = -(__int128)value->omega;
        omega = (__int128)value->one - value->omega;
    } else {
        one = (__int128)value->omega - value->one;
        omega = -(__int128)value->one;
    }
    if (one < INT64_MIN || one > INT64_MAX ||
        omega < INT64_MIN || omega > INT64_MAX) {
        return false;
    }
    value->one = one;
    value->omega = omega;
    return true;
}

static unsigned carrier_number(unsigned carrier)
{
    return carrier == K_VACUUM ? 0 : 1;
}

static unsigned coupled_number(const PhaseQemuV12State *s, unsigned carrier)
{
    if (s->fault_mode == FAULT_DIFFERENTIAL_COUPLING &&
        carrier == K_RAIL_B) {
        return 2;
    }
    return carrier_number(carrier);
}

static bool normal_component(unsigned index, unsigned *ca_out,
                             unsigned *cb_out)
{
    unsigned ca, ra, cb, rb, carrier, rk;

    system_decode(index, &ca, &ra, &cb, &rb, &carrier, &rk);
    if (ca != ra || cb != rb ||
        carrier != (rk == 0 ? K_RAIL_A : K_RAIL_B)) {
        return false;
    }
    if (ca_out) {
        *ca_out = ca;
    }
    if (cb_out) {
        *cb_out = cb;
    }
    return true;
}

static unsigned prepared_carrier(const PhaseQemuV12State *s, unsigned rk)
{
    switch (s->fault_mode) {
    case FAULT_VACUUM_CARRIER:
        return K_VACUUM;
    case FAULT_NON_EIGENSTATE_CARRIER:
        return rk == 0 ? K_VACUUM : K_RAIL_A;
    default:
        return rk == 0 ? K_RAIL_A : K_RAIL_B;
    }
}

static bool density_trace_one(const PhaseQemuV12State *s)
{
    __int128 one = 0;
    __int128 omega = 0;
    unsigned index;

    if (s->density_denom_power != PHASE_V12_DENSITY_DENOM_POWER) {
        return false;
    }
    for (index = 0; index < PHASE_V12_SYSTEM_DIM; index++) {
        const PhaseV12Cyclo3 *cell =
            &s->density[index * PHASE_V12_SYSTEM_DIM + index];

        one += cell->one;
        omega += cell->omega;
    }
    return one == 8 && omega == 0;
}

static void clear_private(PhaseQemuV12State *s)
{
    memset(s->private_residue, 0, sizeof(s->private_residue));
    memset(s->private_bound, 0, sizeof(s->private_bound));
    memset(s->adapter_envelope, 0, sizeof(s->adapter_envelope));
    s->private_ready_mask = 0;
    s->private_armed = false;
    s->arm_generation = 0;
    s->arm_nonce = 0;
}

static void clear_descriptor(PhaseQemuV12State *s)
{
    memset(s->descriptor, 0, sizeof(s->descriptor));
    s->descriptor_index = 0;
    s->descriptor_length = 0;
    s->boundary_schema = 0;
    s->descriptor_fingerprint = 0;
    s->descriptor_sealed = false;
}

static void clear_boundary(PhaseQemuV12State *s)
{
    memset(s->boundary, 0, sizeof(s->boundary));
    s->response_ready = false;
    s->response_acked = false;
    s->resource_sealed = false;
    s->output_receipt_lo = 0;
    s->output_receipt_hi = 0;
}

static void clear_observers(PhaseQemuV12State *s)
{
    unsigned slot;

    for (slot = 0; slot < PHASE_V12_PRIVATE_SLOTS; slot++) {
        s->observed_phase[slot] = PHASE_V12_INVALID_OBSERVATION;
    }
    s->observed_same_backing = 0;
    s->observed_kr_return = 0;
    s->observed_factorized = 0;
    s->observed_port_clear = 0;
    s->observed_env_factored = 0;
}

static void release_tags(PhaseQemuV12State *s)
{
    s->leased = false;
    s->owner_tag = 0;
    s->program_tag = 0;
}

static uint32_t active_request_error(const PhaseQemuV12State *s)
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

static uint64_t descriptor_fingerprint(const PhaseQemuV12State *s)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    unsigned index;

    hash = fnv_u64(hash, s->descriptor_length);
    hash = fnv_u64(hash, s->boundary_schema);
    for (index = 0; index < s->descriptor_length; index++) {
        hash = fnv_u64(hash, s->descriptor[index]);
    }
    return hash;
}

static bool descriptor_valid(const PhaseQemuV12State *s)
{
    unsigned index;

    if (s->descriptor_length != PHASE_V12_DESCRIPTOR_WORDS ||
        s->boundary_schema != 1) {
        return false;
    }
    for (index = 0; index < PHASE_V12_DESCRIPTOR_WORDS; index++) {
        if (s->descriptor[index] != phase_v12_public_descriptor[index]) {
            return false;
        }
    }
    return true;
}

static bool ideal_prepare(PhaseQemuV12State *s)
{
    unsigned component[8];
    unsigned count = 0;
    unsigned ca, cb, rk, row, column;

    memset(s->density, 0, sizeof(s->density));
    for (ca = 0; ca < 2; ca++) {
        for (cb = 0; cb < 2; cb++) {
            for (rk = 0; rk < 2; rk++) {
                component[count++] = system_index(ca, ca, cb, cb,
                                                   prepared_carrier(s, rk),
                                                   rk);
            }
        }
    }
    for (row = 0; row < count; row++) {
        for (column = 0; column < count; column++) {
            PhaseV12Cyclo3 *cell =
                &s->density[component[row] * PHASE_V12_SYSTEM_DIM +
                            component[column]];

            cell->one = 1;
            cell->omega = 0;
        }
    }
    s->density_denom_power = PHASE_V12_DENSITY_DENOM_POWER;
    return density_trace_one(s);
}

static bool apply_query(PhaseQemuV12State *s, unsigned slot,
                        uint8_t residue)
{
    unsigned ket;

    if (slot >= PHASE_V12_PRIVATE_SLOTS || residue > 2) {
        s->error = ERR_INVARIANT;
        return false;
    }
    if (!counter_can_add(s, s->resource_query_applications, 1) ||
        !counter_can_add(s, s->resource_logical_queries, 1) ||
        !counter_can_add(s, s->resource_action_q40_rad,
                         PHASE_V12_ACTION_PER_QUERY_Q40) ||
        !counter_can_add(s, s->virtual_cycles, PHASE_V12_STATE_CELLS)) {
        return false;
    }
    for (ket = 0; ket < PHASE_V12_SYSTEM_DIM; ket++) {
        unsigned ca_k, ra_k, cb_k, rb_k, carrier_k, rk_k;
        unsigned bra;

        system_decode(ket, &ca_k, &ra_k, &cb_k, &rb_k,
                      &carrier_k, &rk_k);
        for (bra = 0; bra < PHASE_V12_SYSTEM_DIM; bra++) {
            unsigned ca_b, ra_b, cb_b, rb_b, carrier_b, rk_b;
            unsigned client_k;
            unsigned client_b;
            int exponent;
            PhaseV12Cyclo3 *cell =
                &s->density[ket * PHASE_V12_SYSTEM_DIM + bra];

            if (cyclo_zero(*cell)) {
                continue;
            }
            system_decode(bra, &ca_b, &ra_b, &cb_b, &rb_b,
                          &carrier_b, &rk_b);
            client_k = slot == 0 ? ca_k : cb_k;
            client_b = slot == 0 ? ca_b : cb_b;
            exponent = residue *
                ((int)(client_k * coupled_number(s, carrier_k)) -
                 (int)(client_b * coupled_number(s, carrier_b)));
            if (!cyclo_multiply_power(cell, exponent)) {
                s->error = ERR_OVERFLOW;
                return false;
            }
        }
    }
    s->resource_query_applications++;
    s->resource_logical_queries++;
    s->resource_action_q40_rad += PHASE_V12_ACTION_PER_QUERY_Q40;
    s->virtual_cycles += PHASE_V12_STATE_CELLS;
    return true;
}

static uint32_t observe_phase(const PhaseQemuV12State *s, unsigned slot)
{
    unsigned ket;
    unsigned bra;
    PhaseV12Cyclo3 value;
    unsigned residue;

    if (slot == 0) {
        ket = system_index(1, 1, 0, 0, K_RAIL_A, 0);
        bra = system_index(0, 0, 0, 0, K_RAIL_A, 0);
    } else {
        ket = system_index(0, 0, 1, 1, K_RAIL_A, 0);
        bra = system_index(0, 0, 0, 0, K_RAIL_A, 0);
    }
    value = s->density[ket * PHASE_V12_SYSTEM_DIM + bra];
    for (residue = 0; residue < 3; residue++) {
        if (cyclo_equal(value, cyclo_power(residue))) {
            return residue;
        }
    }
    return PHASE_V12_INVALID_OBSERVATION;
}

static bool verify_complete_expected_state(PhaseQemuV12State *s)
{
    unsigned ket;

    if (!counter_can_add(s, s->resource_return_checks,
                         PHASE_V12_STATE_CELLS) ||
        !counter_can_add(s, s->resource_certification_ops,
                         PHASE_V12_STATE_CELLS) ||
        !counter_can_add(s, s->virtual_cycles, PHASE_V12_STATE_CELLS)) {
        return false;
    }

    if (!density_trace_one(s) ||
        s->backing_address != (uintptr_t)&s->density[0]) {
        return false;
    }
    for (ket = 0; ket < PHASE_V12_SYSTEM_DIM; ket++) {
        unsigned ca_k = 0;
        unsigned cb_k = 0;
        bool ket_component = normal_component(ket, &ca_k, &cb_k);
        unsigned bra;

        for (bra = 0; bra < PHASE_V12_SYSTEM_DIM; bra++) {
            unsigned ca_b = 0;
            unsigned cb_b = 0;
            bool bra_component = normal_component(bra, &ca_b, &cb_b);
            PhaseV12Cyclo3 expected = { 0, 0 };
            PhaseV12Cyclo3 actual =
                s->density[ket * PHASE_V12_SYSTEM_DIM + bra];

            if (ket_component && bra_component) {
                int exponent =
                    s->private_residue[0] * ((int)ca_k - (int)ca_b) +
                    s->private_residue[1] * ((int)cb_k - (int)cb_b);

                expected = cyclo_power(exponent);
            }
            if (!cyclo_equal(actual, expected)) {
                return false;
            }
        }
    }
    s->resource_return_checks += PHASE_V12_STATE_CELLS;
    s->resource_certification_ops += PHASE_V12_STATE_CELLS;
    s->virtual_cycles += PHASE_V12_STATE_CELLS;
    s->observed_phase[0] = observe_phase(s, 0);
    s->observed_phase[1] = observe_phase(s, 1);
    return s->observed_phase[0] == s->private_residue[0] &&
           s->observed_phase[1] == s->private_residue[1] &&
           s->port_clear && s->env_factored;
}

static bool supply_fresh_clients(PhaseQemuV12State *s)
{
    PhaseV12Cyclo3 kr[6 * 6] = { 0 };
    PhaseV12Cyclo3 *next = s->scratch;
    unsigned client;
    unsigned kket, rkket, kbra, rkbra;
    unsigned ket;
    unsigned bra;

    if (!counter_can_add(s, s->client_supply_count, 2) ||
        !counter_can_add(s, s->resource_preparation_ops, 4)) {
        return false;
    }
    memset(next, 0, sizeof(s->scratch));

    /* Trace both clients and their references, retaining K,R_K exactly. */
    for (client = 0; client < 16; client++) {
        unsigned ca = (client >> 3) & 1;
        unsigned ra = (client >> 2) & 1;
        unsigned cb = (client >> 1) & 1;
        unsigned rb = client & 1;

        for (kket = 0; kket < 3; kket++) {
            for (rkket = 0; rkket < 2; rkket++) {
                ket = system_index(ca, ra, cb, rb, kket, rkket);
                for (kbra = 0; kbra < 3; kbra++) {
                    for (rkbra = 0; rkbra < 2; rkbra++) {
                        PhaseV12Cyclo3 *target =
                            &kr[(kket * 2 + rkket) * 6 +
                                (kbra * 2 + rkbra)];
                        PhaseV12Cyclo3 value =
                            s->density[ket * PHASE_V12_SYSTEM_DIM +
                                       system_index(ca, ra, cb, rb,
                                                    kbra, rkbra)];
                        __int128 one = (__int128)target->one + value.one;
                        __int128 omega =
                            (__int128)target->omega + value.omega;

                        if (one < INT64_MIN || one > INT64_MAX ||
                            omega < INT64_MIN || omega > INT64_MAX) {
                            s->error = ERR_OVERFLOW;
                            return false;
                        }
                        target->one = one;
                        target->omega = omega;
                    }
                }
            }
        }
    }

    /* Tensor the retained K,R_K marginal with two fresh Bell pairs. */
    for (ket = 0; ket < PHASE_V12_SYSTEM_DIM; ket++) {
        unsigned ca_k, ra_k, cb_k, rb_k;

        system_decode(ket, &ca_k, &ra_k, &cb_k, &rb_k, &kket, &rkket);
        if (ca_k != ra_k || cb_k != rb_k) {
            continue;
        }
        for (bra = 0; bra < PHASE_V12_SYSTEM_DIM; bra++) {
            unsigned ca_b, ra_b, cb_b, rb_b;
            PhaseV12Cyclo3 retained;

            system_decode(bra, &ca_b, &ra_b, &cb_b, &rb_b,
                          &kbra, &rkbra);
            if (ca_b != ra_b || cb_b != rb_b) {
                continue;
            }
            retained = kr[(kket * 2 + rkket) * 6 +
                          (kbra * 2 + rkbra)];
            if ((retained.one & 3) != 0 || (retained.omega & 3) != 0) {
                s->error = ERR_CLIENT_REFERENCE_FAILED;
                return false;
            }
            next[ket * PHASE_V12_SYSTEM_DIM + bra].one = retained.one / 4;
            next[ket * PHASE_V12_SYSTEM_DIM + bra].omega =
                retained.omega / 4;
        }
    }
    memcpy(s->density, next, sizeof(s->density));
    memset(s->scratch, 0, sizeof(s->scratch));
    if (!density_trace_one(s)) {
        s->error = ERR_CLIENT_REFERENCE_FAILED;
        return false;
    }
    s->client_supply_count += 2;
    s->resource_preparation_ops += 4;
    return true;
}

static void ideal_sanitize(PhaseQemuV12State *s)
{
    memset(s->density, 0, sizeof(s->density));
    memset(s->scratch, 0, sizeof(s->scratch));
    s->density_denom_power = 0;
}

static uint32_t ideal_execute(PhaseQemuV12State *s)
{
    s->port_clear = false;
    s->outputs_held = true;
    s->env_factored = true;
    if (!apply_query(s, 0, s->private_residue[0]) ||
        !apply_query(s, 1, s->private_residue[1])) {
        return RETURN_FAILED;
    }
    if (s->fault_mode == FAULT_STATE_CORRUPT) {
        s->density[0].one++;
    }
    if (s->fault_mode == FAULT_ENVIRONMENT_TAG) {
        s->env_factored = false;
    }
    if (s->fault_mode != FAULT_PORT_STUCK) {
        s->port_clear = true;
    }
    return RETURN_NONE;
}

static uint32_t ideal_verify_return(PhaseQemuV12State *s)
{
    bool verified = verify_complete_expected_state(s);

    s->same_backing = s->backing_address == (uintptr_t)&s->density[0];
    s->observed_same_backing = s->same_backing;
    s->observed_port_clear = s->port_clear;
    s->observed_env_factored = s->env_factored;
    s->observed_kr_return = verified;
    s->observed_factorized = verified;
    if (!verified || !s->same_backing) {
        if (!s->port_clear) {
            s->error = ERR_PORT_NOT_CLEAR;
        } else if (!s->env_factored) {
            s->error = ERR_ENVIRONMENT_NOT_FACTORED;
        } else if (!s->same_backing) {
            s->error = ERR_CUSTODY_MISMATCH;
        } else if (s->fault_mode == FAULT_STATE_CORRUPT) {
            s->error = ERR_INVARIANT;
        } else {
            s->error = ERR_CARRIER_REFERENCE_FAILED;
        }
        return RETURN_FAILED;
    }
    return RETURN_EXACT_FORMAL;
}

static uint32_t open_execute(PhaseQemuV12State *s)
{
    return ideal_execute(s);
}

static uint32_t open_verify_return(PhaseQemuV12State *s)
{
    uint32_t result = ideal_verify_return(s);

    if (result == RETURN_FAILED || s->open_model_q32 == 0) {
        return result;
    }

    /*
     * No open dynamics are simulated.  A nonzero model request is classified
     * APPROX and charged analytically, with exact restoration and reuse
     * deliberately withheld.  This is a falsifier stub, not a noise claim.
     */
    if (!counter_can_add(s, s->resource_environment_ops, 1) ||
        !counter_can_add(s, s->resource_env_history_cells, 1)) {
        return RETURN_FAILED;
    }
    s->resource_environment_ops++;
    s->resource_env_history_cells++;
    s->resource_loss_q63 = (uint64_t)s->open_model_q32 << 31;
    s->resource_dephasing_q63 = (uint64_t)s->open_model_q32 << 31;
    s->env_factored = false;
    s->observed_env_factored = 0;
    s->observed_kr_return = 0;
    s->observed_factorized = 0;
    return RETURN_APPROX_MODEL;
}

static bool external_prepare(PhaseQemuV12State *s)
{
    if (!s->test_adapter_enabled || !s->test_provider_enabled) {
        s->error = ERR_BACKEND_UNAVAILABLE;
        return false;
    }
    s->adapter_state = ADAPTER_READY;
    return ideal_prepare(s);
}

static bool available_lease_preflight(PhaseQemuV12State *s)
{
    (void)s;
    return true;
}

static bool external_lease_preflight(PhaseQemuV12State *s)
{
    if (!s->test_adapter_enabled || !s->test_provider_enabled) {
        s->error = ERR_BACKEND_UNAVAILABLE;
        return false;
    }
    return true;
}

static bool external_supply_clients(PhaseQemuV12State *s)
{
    s->error = ERR_REUSE_NOT_QUALIFIED;
    return false;
}

static void external_sanitize(PhaseQemuV12State *s)
{
    ideal_sanitize(s);
    s->adapter_state = ADAPTER_DISCONNECTED;
    s->adapter_deadline_tick = 0;
    s->adapter_authenticated_lineage = false;
    memset(s->adapter_envelope, 0, sizeof(s->adapter_envelope));
}

static uint32_t external_execute(PhaseQemuV12State *s)
{
    if (!s->test_adapter_enabled ||
        s->adapter_state != ADAPTER_READY ||
        s->private_ready_mask != PHASE_V12_PRIVATE_READY_ALL) {
        s->error = ERR_ADAPTER_AUTH_BINDING;
        return RETURN_FAILED;
    }
    if (!counter_can_add(s, s->adapter_virtual_tick, 2) ||
        !counter_can_add(s, s->adapter_dispatches, 1)) {
        return RETURN_FAILED;
    }
    s->outputs_held = true;
    s->port_clear = false;
    s->env_factored = false;
    s->adapter_deadline_tick = s->adapter_virtual_tick + 2;
    s->adapter_dispatches++;
    s->adapter_state = ADAPTER_WAITING_A;
    return PHASE_V12_BACKEND_PENDING;
}

static uint32_t external_poll(PhaseQemuV12State *s)
{
    if (s->adapter_state != ADAPTER_WAITING_A &&
        s->adapter_state != ADAPTER_WAITING_B) {
        s->error = ERR_BAD_STATE;
        return RETURN_FAILED;
    }
    if (!counter_can_add(s, s->adapter_virtual_tick, 1) ||
        !counter_can_add(s, s->adapter_completions, 1)) {
        return RETURN_FAILED;
    }
    s->adapter_virtual_tick++;
    if (s->test_adapter_mode == ADAPTER_MODE_TIMEOUT &&
        s->adapter_virtual_tick >= s->adapter_deadline_tick) {
        if (!counter_can_add(s, s->adapter_cancels, 1)) {
            return RETURN_FAILED;
        }
        s->adapter_cancels++;
        s->adapter_state = ADAPTER_TIMED_OUT;
        s->outputs_held = false;
        s->port_clear = true;
        s->error = ERR_ADAPTER_TIMEOUT;
        return RETURN_FAILED;
    }
    s->adapter_completions++;
    if (s->adapter_state == ADAPTER_WAITING_A) {
        if (!counter_can_add(s, s->adapter_dispatches, 1)) {
            return RETURN_FAILED;
        }
        s->adapter_dispatches++;
        s->adapter_state = ADAPTER_WAITING_B;
        return PHASE_V12_BACKEND_PENDING;
    }
    s->adapter_state = ADAPTER_COMPLETE;
    return ideal_execute(s);
}

static uint32_t external_verify_return(PhaseQemuV12State *s)
{
    uint32_t result = ideal_verify_return(s);

    if (result == RETURN_FAILED) {
        return result;
    }

    /* Exact software algebra is not an exact physical return receipt. */
    s->same_backing = false;
    s->observed_same_backing = 0;
    s->observed_kr_return = 0;
    s->observed_factorized = 0;
    s->observed_env_factored = 0;
    s->env_factored = false;
    s->resource_loss_q63 = PHASE_V12_RESOURCE_UNKNOWN;
    s->resource_dephasing_q63 = PHASE_V12_RESOURCE_UNKNOWN;
    return RETURN_APPROX_MODEL;
}

static void external_cancel(PhaseQemuV12State *s)
{
    if (s->adapter_state == ADAPTER_WAITING_A ||
        s->adapter_state == ADAPTER_WAITING_B) {
        if (!counter_can_add(s, s->adapter_cancels, 1)) {
            s->outputs_held = false;
            s->port_clear = true;
            return;
        }
        s->adapter_cancels++;
        s->adapter_state = ADAPTER_CANCELED;
    }
    s->outputs_held = false;
    s->port_clear = true;
}

static const PhaseV12BackendOps phase_v12_ideal_ops = {
    .id = PHASE_V12_BACKEND_IDEAL,
    .name = "ideal-dual-rail",
    .lease_preflight = available_lease_preflight,
    .prepare = ideal_prepare,
    .supply_clients = supply_fresh_clients,
    .execute = ideal_execute,
    .poll = NULL,
    .verify_return = ideal_verify_return,
    .cancel = NULL,
    .sanitize = ideal_sanitize,
};

static const PhaseV12BackendOps phase_v12_open_ops = {
    .id = PHASE_V12_BACKEND_OPEN,
    .name = "open-classification-stub",
    .lease_preflight = available_lease_preflight,
    .prepare = ideal_prepare,
    .supply_clients = supply_fresh_clients,
    .execute = open_execute,
    .poll = NULL,
    .verify_return = open_verify_return,
    .cancel = NULL,
    .sanitize = ideal_sanitize,
};

static const PhaseV12BackendOps phase_v12_external_ops = {
    .id = PHASE_V12_BACKEND_EXTERNAL,
    .name = "external-unavailable",
    .lease_preflight = external_lease_preflight,
    .prepare = external_prepare,
    .supply_clients = external_supply_clients,
    .execute = external_execute,
    .poll = external_poll,
    .verify_return = external_verify_return,
    .cancel = external_cancel,
    .sanitize = external_sanitize,
};

static const PhaseV12BackendOps *backend_ops_for_id(uint32_t id)
{
    switch (id) {
    case PHASE_V12_BACKEND_IDEAL:
        return &phase_v12_ideal_ops;
    case PHASE_V12_BACKEND_OPEN:
        return &phase_v12_open_ops;
    case PHASE_V12_BACKEND_EXTERNAL:
        return &phase_v12_external_ops;
    default:
        return NULL;
    }
}

static uint64_t resource_peak_bits(const PhaseQemuV12State *s)
{
    /* Explicit device-object plus largest named stack-local allocation floor. */
    return (sizeof(*s) + sizeof(PhaseV12Cyclo3) * 36) * 8;
}

static uint64_t resource_allocated_secret_bits(const PhaseQemuV12State *s)
{
    return (sizeof(s->private_residue) + sizeof(s->private_bound) +
            sizeof(s->private_ready_mask) + sizeof(s->arm_generation) +
            sizeof(s->arm_nonce) + sizeof(s->private_armed) +
            sizeof(s->adapter_envelope)) * 8;
}

static uint64_t resource_carrier_photon_number(const PhaseQemuV12State *s)
{
    /*
     * One photon is an exact property of the internal ideal algebra only.
     * The disconnected external adapter has no physical carrier meter, so a
     * physical photon number is unknown rather than the model value or zero.
     */
    return s->backend_id == PHASE_V12_BACKEND_EXTERNAL ?
           PHASE_V12_RESOURCE_UNKNOWN : 1;
}

static uint64_t reported_resource_control_words(const PhaseQemuV12State *s)
{
    return s->resource_sealed ? s->sealed_resource_control_words :
                                s->resource_control_words;
}

static uint64_t resource_digest(const PhaseQemuV12State *s, uint64_t domain)
{
    uint64_t hash = UINT64_C(14695981039346656037);

    hash = fnv_u64(hash, domain);
    hash = fnv_u64(hash, s->resource_query_applications);
    hash = fnv_u64(hash, s->resource_return_checks);
    hash = fnv_u64(hash, s->resource_environment_ops);
    hash = fnv_u64(hash, s->sealed_resource_control_words);
    hash = fnv_u64(hash, s->resource_preparation_ops);
    hash = fnv_u64(hash, s->resource_certification_ops);
    hash = fnv_u64(hash, s->resource_logical_queries);
    hash = fnv_u64(hash, s->resource_duration_fs);
    hash = fnv_u64(hash, s->resource_port_bandwidth_hz);
    hash = fnv_u64(hash, s->resource_action_q40_rad);
    hash = fnv_u64(hash, s->resource_mean_energy_attoj);
    hash = fnv_u64(hash, s->resource_loss_q63);
    hash = fnv_u64(hash, s->resource_dephasing_q63);
    hash = fnv_u64(hash, s->resource_env_history_cells);
    hash = fnv_u64(hash, s->resource_custody_transitions);
    hash = fnv_u64(hash, s->resource_reuse_count);
    hash = fnv_u64(hash, s->resource_discarded_trials);
    hash = fnv_u64(hash, s->resource_output_hold_fs);
    hash = fnv_u64(hash, s->resource_maintenance_ops);
    hash = fnv_u64(hash, s->adapter_auth_accepted);
    hash = fnv_u64(hash, s->adapter_auth_rejected);
    hash = fnv_u64(hash, s->adapter_dispatches);
    hash = fnv_u64(hash, s->adapter_completions);
    hash = fnv_u64(hash, s->adapter_cancels);
    hash = fnv_u64(hash, s->adapter_virtual_tick);
    hash = fnv_u64(hash, s->adapter_deadline_tick);
    hash = fnv_u64(hash, s->adapter_state);
    hash = fnv_u64(hash, s->adapter_authenticated_lineage);
    hash = fnv_u64(hash, PHASE_V12_RESOURCE_SCHEMA);
    hash = fnv_u64(hash, PHASE_V12_STATE_CELLS);
    hash = fnv_u64(hash, PHASE_V12_STATE_CELLS); /* independent scratch */
    hash = fnv_u64(hash, resource_peak_bits(s));
    hash = fnv_u64(hash, resource_allocated_secret_bits(s));
    hash = fnv_u64(hash, 64); /* exact coefficient precision */
    hash = fnv_u64(hash, 4); /* ceil(log2(3^2)) logical code capacity */
    hash = fnv_u64(hash, resource_carrier_photon_number(s));
    hash = fnv_u64(hash, PHASE_V12_RESOURCE_UNKNOWN); /* compiler */
    hash = fnv_u64(hash, s->sealed_resource_control_words); /* controller */
    return fnv_u64(hash, PHASE_V12_RESOURCE_UNKNOWN); /* construction */
}

static uint32_t phase_status(const PhaseQemuV12State *s)
{
    uint32_t status = 0;

    if (!s->leased && !s->spent && !s->snapshot_lineage &&
        (!s->prepared || (s->restored && s->port_clear))) {
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
    if (s->restored && s->return_class == RETURN_EXACT_FORMAL) {
        status |= ST_EXACT_RETURN;
    }
    if (s->carrier_present) {
        status |= ST_CARRIER_PRESENT;
    }
    if (s->snapshot_lineage) {
        status |= ST_SNAPSHOT_LINEAGE | ST_SHAM;
    }
    if (s->port_clear) {
        status |= ST_PORT_CLEAR;
    }
    if (s->private_armed) {
        status |= ST_PRIVATE_ARMED;
    }
    if (s->private_ready_mask == PHASE_V12_PRIVATE_READY_ALL) {
        status |= ST_PRIVATE_READY;
    }
    if (s->outputs_held) {
        status |= ST_OUTPUTS_HELD;
    }
    if (s->return_verified) {
        status |= ST_RETURN_VERIFIED;
    }
    if (s->response_acked) {
        status |= ST_RESPONSE_ACKED;
    }
    if (s->same_backing) {
        status |= ST_SAME_ALLOCATION;
    }
    if (s->env_factored) {
        status |= ST_ENV_FACTORED;
    }
    if (s->backend_id == PHASE_V12_BACKEND_OPEN) {
        status |= ST_NOISY_BACKEND;
    }
    if (s->backend_id == PHASE_V12_BACKEND_EXTERNAL) {
        status |= ST_EXTERNAL_BACKEND;
    }
    if (s->preparation_receipt_lo != 0 || s->preparation_receipt_hi != 0) {
        status |= ST_PREP_RECEIPT_READY;
    }
    if (s->resource_sealed) {
        status |= ST_RESOURCE_SEALED;
    }
    if (s->backend_id == PHASE_V12_BACKEND_EXTERNAL &&
        s->adapter_authenticated_lineage) {
        status |= ST_ADAPTER_AUTHENTICATED;
    }
    if (s->backend_id == PHASE_V12_BACKEND_EXTERNAL &&
        (s->adapter_state == ADAPTER_WAITING_A ||
         s->adapter_state == ADAPTER_WAITING_B)) {
        status |= ST_ADAPTER_PENDING;
    }
    if (s->backend_id == PHASE_V12_BACKEND_EXTERNAL) {
        status |= ST_EXTERNAL_REUSE_FORBIDDEN;
    }
    return status;
}

static uint32_t phase_capabilities(const PhaseQemuV12State *s)
{
    uint32_t caps = CAP_NOMINAL_COMMAND_TAGS |
                    CAP_EXACT_CYCLO3_STATE |
                    CAP_PUBLIC_FIXED_DESCRIPTOR |
                    CAP_SOURCE_ISOLATION |
                    CAP_ATOMIC_TRANSACTION |
                    CAP_DUAL_RAIL_N1 |
                    CAP_CARRIER_REFERENCE |
                    CAP_CLIENT_REFERENCES |
                    CAP_TWO_LATE_BOUND_QUERIES |
                    CAP_SWAPPABLE_BACKEND |
                    CAP_COHERENT_PORT_MODEL |
                    CAP_RESOURCE_VECTOR |
                    CAP_SAME_ALLOCATION_RECEIPT |
                    CAP_MIGRATION_SHAM |
                    CAP_CONSTANT_PUBLIC_ENVELOPE;

    if (s->test_provider_enabled) {
        caps |= CAP_PRIVATE_QOM_PROVIDER | CAP_TEST_QOM_OBSERVER;
    }
    if (s->backend_id == PHASE_V12_BACKEND_IDEAL ||
        (s->backend_id == PHASE_V12_BACKEND_OPEN &&
         s->open_model_q32 == 0)) {
        caps |= CAP_EXACT_RETURN;
    }
    if (s->backend_id == PHASE_V12_BACKEND_OPEN) {
        caps |= CAP_APPROX_RETURN | CAP_OPEN_CLASSIFICATION_STUB;
    }
    if (s->backend_id == PHASE_V12_BACKEND_EXTERNAL) {
        caps |= CAP_EXTERNAL_ADAPTER_SLOT |
                CAP_EXTERNAL_REUSE_FORBIDDEN |
                CAP_PHYSICAL_RESOURCES_UNKNOWN;
        if (s->test_adapter_enabled) {
            caps |= CAP_EXTERNAL_AUTH_TEST_STUB |
                    CAP_EXTERNAL_ASYNC_LIFECYCLE |
                    CAP_APPROX_RETURN;
        }
    }
    if (s->fault_mode != FAULT_NONE) {
        caps |= CAP_TEST_FAULTS_ACTIVE;
    }
    return caps;
}

static bool seal_response(PhaseQemuV12State *s);

static void spend_generation(PhaseQemuV12State *s, uint32_t error)
{
    s->error = error == ERR_NONE ? ERR_RETURN_FAILED : error;
    s->return_class = RETURN_FAILED;
    s->spent = false;
    s->restored = false;
    s->reuse_qualified = false;
    s->return_verified = false;
    s->response_ready = false;
    s->response_acked = false;
    s->outputs_held = false;
    s->resource_loss_q63 = s->backend_id == PHASE_V12_BACKEND_EXTERNAL ?
                           PHASE_V12_RESOURCE_UNKNOWN : s->resource_loss_q63;
    s->resource_dephasing_q63 = s->backend_id == PHASE_V12_BACKEND_EXTERNAL ?
                                PHASE_V12_RESOURCE_UNKNOWN :
                                s->resource_dephasing_q63;
    if (!counter_can_add(s, s->resource_discarded_trials, 1)) {
        s->spent = true;
        s->lifecycle = LIFE_SPENT;
        clear_private(s);
        release_tags(s);
        return;
    }
    s->resource_discarded_trials++;
    clear_private(s);
    if (!seal_response(s)) {
        s->spent = true;
        s->lifecycle = LIFE_SPENT;
        release_tags(s);
    }
}

static bool seal_response(PhaseQemuV12State *s)
{
    uint64_t transaction;
    uint32_t final_status;

    if (phase_v12_transaction_serial == UINT64_MAX) {
        s->error = ERR_OVERFLOW;
        return false;
    }
    transaction = phase_v12_transaction_serial++;
    s->sealed_resource_control_words = s->resource_control_words;
    s->resource_sealed = true;
    s->lifecycle = LIFE_RESPONSE_READY;
    s->resource_digest_lo = resource_digest(s, UINT64_C(0x52534c4f));
    s->resource_digest_hi = resource_digest(s, UINT64_C(0x52534849));
    s->return_receipt_lo = receipt_hash(UINT64_C(0x5245544c),
        s->allocation_id_lo, s->generation, s->resource_digest_lo);
    s->return_receipt_hi = receipt_hash(UINT64_C(0x52455448),
        s->allocation_id_hi, s->return_class, s->resource_digest_hi);
    s->output_receipt_lo = receipt_hash(UINT64_C(0x4f55544c),
        transaction, s->generation, s->allocation_id_lo);
    s->output_receipt_hi = receipt_hash(UINT64_C(0x4f555448),
        transaction, s->generation, s->allocation_id_hi);

    s->boundary[0] = PHASE_V12_BOUNDARY_HEADER;
    s->boundary[1] = (uint64_t)s->generation |
        ((uint64_t)s->backend_id << 32) |
        ((uint64_t)s->return_class << 48);
    final_status = phase_status(s) | ST_RESPONSE_READY;
    s->boundary[2] = final_status;
    s->boundary[3] = s->output_receipt_lo;
    s->boundary[4] = s->output_receipt_hi;
    s->boundary[5] = s->allocation_id_lo;
    s->boundary[6] = s->allocation_id_hi;
    s->boundary[7] = s->custody_epoch;
    s->boundary[8] = s->preparation_receipt_lo;
    s->boundary[9] = s->preparation_receipt_hi;
    s->boundary[10] = s->return_receipt_lo;
    s->boundary[11] = s->return_receipt_hi;
    s->boundary[12] = s->resource_digest_lo;
    s->boundary[13] = s->resource_digest_hi;
    s->boundary[14] = s->resource_loss_q63;
    s->boundary[15] = s->resource_dephasing_q63;
    /* Commit readiness only after every boundary word is complete. */
    s->response_ready = true;
    return true;
}

static void finish_backend_execution(PhaseQemuV12State *s, uint32_t result)
{
    if (result == RETURN_FAILED) {
        spend_generation(s, s->error);
        return;
    }
    if (result == PHASE_V12_BACKEND_PENDING) {
        return;
    }
    s->lifecycle = LIFE_VERIFYING_RETURN;
    result = s->ops->verify_return(s);
    s->return_class = result;
    if (result == RETURN_FAILED) {
        spend_generation(s, s->error);
        return;
    }
    if (s->fault_mode == FAULT_RESOURCE_UNSEALED) {
        spend_generation(s, ERR_RESOURCE_UNSEALED);
        return;
    }
    if (!counter_can_add(s, s->custody_epoch, 1) ||
        !counter_can_add(s, s->resource_custody_transitions, 1)) {
        spend_generation(s, ERR_OVERFLOW);
        return;
    }
    s->return_verified = result == RETURN_EXACT_FORMAL;
    s->custody_epoch++;
    s->resource_custody_transitions++;
    if (result == RETURN_EXACT_FORMAL) {
        s->restored = true;
        s->reuse_qualified = true;
        s->exact_return_generation = s->generation;
    } else {
        s->restored = false;
        s->reuse_qualified = false;
    }
    clear_private(s);
    if (!seal_response(s)) {
        s->return_class = RETURN_FAILED;
        s->spent = true;
        s->restored = false;
        s->reuse_qualified = false;
        s->return_verified = false;
        s->response_ready = false;
        s->outputs_held = false;
        s->lifecycle = LIFE_SPENT;
        release_tags(s);
    }
}

static void execute_atomic(PhaseQemuV12State *s)
{
    uint32_t request_error = active_request_error(s);
    uint32_t result;

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
    if (!s->descriptor_sealed || s->lifecycle != LIFE_PRIVATE_READY) {
        s->error = ERR_BAD_STATE;
        return;
    }
    if (s->private_ready_mask != PHASE_V12_PRIVATE_READY_ALL ||
        !s->private_bound[0] || !s->private_bound[1]) {
        s->error = ERR_PRIVATE_INCOMPLETE;
        return;
    }
    if (s->arm_generation != s->generation || !s->ops ||
        !s->ops->execute || !s->ops->verify_return) {
        s->error = ERR_PRIVATE_LINEAGE;
        return;
    }
    if (s->resource_action_q40_rad >=
        PHASE_V12_RESOURCE_UNKNOWN -
        2 * PHASE_V12_ACTION_PER_QUERY_Q40) {
        spend_generation(s, ERR_OVERFLOW);
        return;
    }

    s->lifecycle = LIFE_EXECUTING;
    s->error = ERR_NONE;
    s->return_class = RETURN_NONE;
    s->restored = false;
    s->return_verified = false;
    s->reuse_qualified = false;
    clear_boundary(s);
    result = s->ops->execute(s);
    finish_backend_execution(s, result);
}

static void phase_command(PhaseQemuV12State *s, uint32_t command)
{
    uint32_t request_error;

    s->error = ERR_NONE;
    switch (command) {
    case CMD_LEASE:
        if (s->snapshot_lineage) {
            s->error = ERR_SNAPSHOT_LINEAGE;
            return;
        }
        if (!s->ops || !s->ops->lease_preflight ||
            !s->ops->lease_preflight(s)) {
            if (s->error == ERR_NONE) {
                s->error = ERR_BACKEND_UNAVAILABLE;
            }
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
        if (s->prepared &&
            (!s->ops->supply_clients || !s->ops->supply_clients(s))) {
            spend_generation(s, s->error);
            return;
        }
        s->owner_tag = s->request_owner_tag;
        s->program_tag = s->request_program_tag;
        s->leased = true;
        s->response_acked = false;
        s->lifecycle = s->prepared ? LIFE_PREPARED : LIFE_LEASED;
        break;
    case CMD_PREPARE:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (s->prepared || s->generation != 1 ||
            s->lifecycle != LIFE_LEASED || !s->ops) {
            s->error = ERR_BAD_STATE;
            return;
        }
        if (phase_v12_allocation_serial == UINT64_MAX) {
            s->error = ERR_OVERFLOW;
            return;
        }
        if (!counter_can_add(s, s->prepare_count, 1) ||
            !counter_can_add(s, s->client_supply_count, 2) ||
            !counter_can_add(s, s->resource_preparation_ops, 64) ||
            !counter_can_add(s, s->resource_custody_transitions, 1)) {
            return;
        }
        if (!s->ops->prepare(s)) {
            if (s->error == ERR_NONE) {
                s->error = ERR_INVARIANT;
            }
            return;
        }
        s->allocation_id_lo = phase_v12_allocation_serial++;
        s->allocation_id_hi = receipt_hash(UINT64_C(0x414c4c48),
            s->allocation_id_lo, PHASE_V12_DEVICE_ID, s->backend_id);
        s->backing_address = (uintptr_t)&s->density[0];
        s->custody_epoch = 1;
        s->preparation_receipt_lo = receipt_hash(UINT64_C(0x5052454c),
            s->allocation_id_lo, PHASE_V12_SYSTEM_DIM,
            PHASE_V12_DENSITY_DENOM_POWER);
        s->preparation_receipt_hi = receipt_hash(UINT64_C(0x50524548),
            s->allocation_id_hi, s->backend_id, PHASE_V12_DESCRIPTOR_WORDS);
        s->prepared = true;
        s->source_isolated = false;
        s->restored = false;
        s->port_clear = true;
        s->env_factored = true;
        s->same_backing = true;
        s->prepare_count++;
        s->client_supply_count += 2;
        s->resource_preparation_ops += 64;
        s->resource_custody_transitions++;
        s->lifecycle = LIFE_PREPARED;
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
        if (!counter_can_add(s, s->custody_epoch, 1) ||
            !counter_can_add(s, s->resource_custody_transitions, 1)) {
            return;
        }
        s->source_isolated = true;
        s->custody_epoch++;
        s->resource_custody_transitions++;
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
            s->descriptor_length != PHASE_V12_DESCRIPTOR_WORDS ||
            s->boundary_schema != 1) {
            s->error = ERR_DESCRIPTOR_INVALID;
            return;
        }
        if (!descriptor_valid(s)) {
            s->error = ERR_SECRET_SMUGGLE;
            return;
        }
        s->descriptor_fingerprint = descriptor_fingerprint(s);
        s->descriptor_sealed = true;
        s->lifecycle = LIFE_SEALED;
        break;
    case CMD_ARM_PRIVATE:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (!s->test_provider_enabled) {
            s->error = ERR_BACKEND_UNAVAILABLE;
            return;
        }
        if (!s->descriptor_sealed || s->private_armed ||
            s->lifecycle != LIFE_SEALED) {
            s->error = ERR_BAD_STATE;
            return;
        }
        if (phase_v12_arm_serial == UINT64_MAX) {
            s->error = ERR_OVERFLOW;
            return;
        }
        clear_private(s);
        s->adapter_authenticated_lineage = false;
        s->private_armed = true;
        s->arm_generation = s->generation;
        s->arm_nonce = phase_v12_arm_serial++;
        s->lifecycle = LIFE_PRIVATE_ARMED;
        break;
    case CMD_EXECUTE_ATOMIC:
        execute_atomic(s);
        break;
    case CMD_POLL_EXTERNAL:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (s->backend_id != PHASE_V12_BACKEND_EXTERNAL ||
            s->lifecycle != LIFE_EXECUTING || !s->ops || !s->ops->poll) {
            s->error = ERR_BAD_STATE;
            return;
        }
        finish_backend_execution(s, s->ops->poll(s));
        break;
    case CMD_CANCEL_EXTERNAL:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (s->backend_id != PHASE_V12_BACKEND_EXTERNAL ||
            s->lifecycle != LIFE_EXECUTING || !s->ops || !s->ops->cancel) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->ops->cancel(s);
        spend_generation(s, s->error == ERR_OVERFLOW ?
                         ERR_OVERFLOW : ERR_ADAPTER_CANCELED);
        break;
    case CMD_ACK_RESPONSE:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (!s->response_ready || s->lifecycle != LIFE_RESPONSE_READY) {
            s->error = ERR_RESPONSE_LOCKED;
            return;
        }
        s->response_ready = false;
        s->response_acked = true;
        s->outputs_held = false;
        memset(s->boundary, 0, sizeof(s->boundary));
        release_tags(s);
        s->adapter_authenticated_lineage = false;
        if (s->return_class == RETURN_EXACT_FORMAL) {
            s->lifecycle = LIFE_RESPONSE_ACKED;
        } else {
            s->spent = true;
            s->lifecycle = LIFE_SPENT;
        }
        break;
    case CMD_BEGIN_REUSE:
        if (s->snapshot_lineage) {
            s->error = ERR_SNAPSHOT_LINEAGE;
            return;
        }
        if (s->leased || s->spent || !s->response_acked ||
            !s->restored || !s->reuse_qualified || !s->same_backing ||
            s->return_class != RETURN_EXACT_FORMAL ||
            s->lifecycle != LIFE_RESPONSE_ACKED) {
            s->error = ERR_REUSE_NOT_QUALIFIED;
            return;
        }
        if (s->generation == UINT32_MAX) {
            spend_generation(s, ERR_OVERFLOW);
            return;
        }
        if (!counter_can_add(s, s->custody_epoch, 1) ||
            !counter_can_add(s, s->resource_custody_transitions, 1) ||
            !counter_can_add(s, s->resource_reuse_count, 1)) {
            spend_generation(s, ERR_OVERFLOW);
            return;
        }
        clear_descriptor(s);
        clear_private(s);
        s->adapter_authenticated_lineage = false;
        clear_boundary(s);
        clear_observers(s);
        s->request_owner_tag = 0;
        s->request_program_tag = 0;
        s->request_generation = 0;
        s->return_class = RETURN_NONE;
        s->return_verified = false;
        s->reuse_qualified = false;
        s->source_isolated = false;
        s->generation++;
        s->custody_epoch++;
        s->resource_custody_transitions++;
        s->resource_reuse_count++;
        s->resource_sealed = false;
        s->lifecycle = LIFE_REUSABLE;
        break;
    case CMD_SNAPSHOT:
        s->error = ERR_SNAPSHOT_REJECTED;
        break;
    case CMD_ABORT_PREEXEC:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (s->lifecycle < LIFE_SEALED ||
            s->lifecycle > LIFE_PRIVATE_READY) {
            s->error = ERR_BAD_STATE;
            return;
        }
        clear_private(s);
        s->adapter_authenticated_lineage = false;
        clear_descriptor(s);
        s->lifecycle = LIFE_ISOLATED;
        break;
    default:
        s->error = ERR_BAD_ARGUMENT;
        break;
    }
}

static bool register_is_u32(hwaddr address)
{
    switch (address) {
    case REG_MAGIC:
    case REG_ABI:
    case REG_BACKEND:
    case REG_CAPABILITIES_LO:
    case REG_STATUS_LO:
    case REG_ERROR:
    case REG_GENERATION:
    case REG_EXACT_RETURN_GENERATION:
    case REG_ARG0:
    case REG_ARG1:
    case REG_COMMAND:
    case REG_LIFECYCLE:
    case REG_RESOURCE_STATE_CELLS:
    case REG_RESOURCE_SCRATCH_CELLS:
    case REG_REQUEST_OWNER:
    case REG_REQUEST_PROGRAM:
    case REG_REQUEST_GENERATION:
    case REG_DESCRIPTOR_INDEX:
    case REG_DESCRIPTOR_WORD:
    case REG_DESCRIPTOR_LENGTH:
    case REG_BOUNDARY_SCHEMA:
    case REG_FAULT_MODE:
    case REG_CAPABILITIES_HI:
    case REG_STATUS_HI:
    case REG_PRIVATE_QUERY_SLOTS:
    case REG_PRIVATE_READY_MASK:
    case REG_RETURN_CLASS:
    case REG_BOUNDARY_LENGTH:
    case REG_RESOURCE_SCHEMA:
    case REG_ADAPTER_STATE:
        return true;
    default:
        return false;
    }
}

static bool register_is_u64(hwaddr address)
{
    switch (address) {
    case REG_VIRTUAL_CYCLES:
    case REG_BOUNDARY_COMMIT_COOKIE:
    case REG_RESOURCE_QUERY_APPLICATIONS:
    case REG_RESOURCE_RETURN_CHECKS:
    case REG_RESOURCE_ENVIRONMENT_OPS:
    case REG_RESOURCE_PEAK_BITS:
    case REG_DESCRIPTOR_FINGERPRINT:
    case REG_ALLOCATION_ID_LO:
    case REG_ALLOCATION_ID_HI:
    case REG_CUSTODY_EPOCH:
    case REG_PREPARATION_RECEIPT_LO:
    case REG_PREPARATION_RECEIPT_HI:
    case REG_RETURN_RECEIPT_LO:
    case REG_RETURN_RECEIPT_HI:
    case REG_RESOURCE_SECRET_STORAGE_BITS:
    case REG_RESOURCE_CONTROL_WORDS:
    case REG_RESOURCE_PREPARATION_OPS:
    case REG_RESOURCE_CERTIFICATION_OPS:
    case REG_RESOURCE_LOGICAL_QUERIES:
    case REG_RESOURCE_DURATION_FS:
    case REG_RESOURCE_PORT_BANDWIDTH_HZ:
    case REG_RESOURCE_ACTION_Q40_RAD:
    case REG_RESOURCE_MEAN_ENERGY_ATTOJ:
    case REG_RESOURCE_LOSS_Q63:
    case REG_RESOURCE_DEPHASING_Q63:
    case REG_RESOURCE_ENV_HISTORY_CELLS:
    case REG_RESOURCE_CUSTODY_TRANSITIONS:
    case REG_RESOURCE_REUSE_COUNT:
    case REG_RESOURCE_DISCARDED_TRIALS:
    case REG_RESOURCE_OUTPUT_HOLD_FS:
    case REG_RESOURCE_MAINTENANCE_OPS:
    case REG_RESOURCE_PRECISION_BITS:
    case REG_RESOURCE_DIGEST_LO:
    case REG_RESOURCE_DIGEST_HI:
    case REG_RESOURCE_COMPILER_OPS:
    case REG_RESOURCE_CONTROLLER_OPS:
    case REG_RESOURCE_CONSTRUCTION_OPS:
    case REG_RESOURCE_SECRET_ENTROPY_BITS:
    case REG_RESOURCE_CARRIER_PHOTON_NUMBER:
    case REG_ADAPTER_AUTH_ACCEPTED:
    case REG_ADAPTER_AUTH_REJECTED:
    case REG_ADAPTER_DISPATCHES:
    case REG_ADAPTER_COMPLETIONS:
    case REG_ADAPTER_CANCELS:
    case REG_ADAPTER_VIRTUAL_TICK:
    case REG_ADAPTER_DEADLINE_TICK:
        return true;
    default:
        return false;
    }
}

static bool access_width_valid(hwaddr address, unsigned size)
{
    if (address >= REG_BOUNDARY_BASE && address <= REG_BOUNDARY_LAST) {
        return size == 8 && (address & 7) == 0;
    }
    if (register_is_u32(address)) {
        return size == 4;
    }
    if (register_is_u64(address)) {
        return size == 8 && (address & 7) == 0;
    }
    return false;
}

static uint64_t phase_mmio_read(void *opaque, hwaddr address, unsigned size)
{
    PhaseQemuV12State *s = opaque;

    if (!access_width_valid(address, size)) {
        s->error = ERR_BAD_ARGUMENT;
        return size == 4 ? UINT32_MAX : UINT64_MAX;
    }
    if (address >= REG_BOUNDARY_BASE && address <= REG_BOUNDARY_LAST) {
        unsigned index = (address - REG_BOUNDARY_BASE) / 8;

        if (!s->response_ready || s->snapshot_lineage) {
            s->error = ERR_RESPONSE_LOCKED;
            return PHASE_V12_LOCKED_BOUNDARY;
        }
        return s->boundary[index];
    }
    switch (address) {
    case REG_MAGIC:
        return PHASE_V12_MAGIC;
    case REG_ABI:
        return PHASE_V12_ABI;
    case REG_BACKEND:
        return s->backend_id;
    case REG_CAPABILITIES_LO:
        return phase_capabilities(s);
    case REG_STATUS_LO:
        return phase_status(s);
    case REG_ERROR:
        return s->error;
    case REG_GENERATION:
        return s->generation;
    case REG_EXACT_RETURN_GENERATION:
        return s->exact_return_generation;
    case REG_ARG0:
        return s->arg0;
    case REG_ARG1:
        return s->arg1;
    case REG_COMMAND:
        return 0;
    case REG_LIFECYCLE:
        return s->lifecycle;
    case REG_VIRTUAL_CYCLES:
        return s->virtual_cycles;
    case REG_BOUNDARY_COMMIT_COOKIE:
        return s->response_ready && !s->snapshot_lineage ?
               s->boundary[0] : PHASE_V12_LOCKED_BOUNDARY;
    case REG_RESOURCE_STATE_CELLS:
        return PHASE_V12_STATE_CELLS;
    case REG_RESOURCE_SCRATCH_CELLS:
        return PHASE_V12_STATE_CELLS;
    case REG_RESOURCE_QUERY_APPLICATIONS:
        return s->resource_query_applications;
    case REG_RESOURCE_RETURN_CHECKS:
        return s->resource_return_checks;
    case REG_RESOURCE_ENVIRONMENT_OPS:
        return s->resource_environment_ops;
    case REG_RESOURCE_PEAK_BITS:
        return resource_peak_bits(s);
    case REG_REQUEST_OWNER:
        return s->request_owner_tag;
    case REG_REQUEST_PROGRAM:
        return s->request_program_tag;
    case REG_REQUEST_GENERATION:
        return s->request_generation;
    case REG_DESCRIPTOR_INDEX:
        return s->descriptor_index;
    case REG_DESCRIPTOR_WORD:
        return s->descriptor_index < PHASE_V12_DESCRIPTOR_WORDS ?
               s->descriptor[s->descriptor_index] : 0;
    case REG_DESCRIPTOR_LENGTH:
        return s->descriptor_length;
    case REG_BOUNDARY_SCHEMA:
        return s->boundary_schema;
    case REG_DESCRIPTOR_FINGERPRINT:
        return s->descriptor_fingerprint;
    case REG_FAULT_MODE:
        return s->fault_mode;
    case REG_CAPABILITIES_HI:
    case REG_STATUS_HI:
        return 0;
    case REG_PRIVATE_QUERY_SLOTS:
        return PHASE_V12_PRIVATE_SLOTS;
    case REG_PRIVATE_READY_MASK:
        return s->private_ready_mask;
    case REG_RETURN_CLASS:
        return s->return_class;
    case REG_BOUNDARY_LENGTH:
        return PHASE_V12_BOUNDARY_WORDS * 8;
    case REG_ALLOCATION_ID_LO:
        return s->allocation_id_lo;
    case REG_ALLOCATION_ID_HI:
        return s->allocation_id_hi;
    case REG_CUSTODY_EPOCH:
        return s->custody_epoch;
    case REG_PREPARATION_RECEIPT_LO:
        return s->preparation_receipt_lo;
    case REG_PREPARATION_RECEIPT_HI:
        return s->preparation_receipt_hi;
    case REG_RETURN_RECEIPT_LO:
        return s->return_receipt_lo;
    case REG_RETURN_RECEIPT_HI:
        return s->return_receipt_hi;
    case REG_RESOURCE_SECRET_STORAGE_BITS:
        return resource_allocated_secret_bits(s);
    case REG_RESOURCE_CONTROL_WORDS:
        return reported_resource_control_words(s);
    case REG_RESOURCE_PREPARATION_OPS:
        return s->resource_preparation_ops;
    case REG_RESOURCE_CERTIFICATION_OPS:
        return s->resource_certification_ops;
    case REG_RESOURCE_LOGICAL_QUERIES:
        return s->resource_logical_queries;
    case REG_RESOURCE_DURATION_FS:
        return s->resource_duration_fs;
    case REG_RESOURCE_PORT_BANDWIDTH_HZ:
        return s->resource_port_bandwidth_hz;
    case REG_RESOURCE_ACTION_Q40_RAD:
        return s->resource_action_q40_rad;
    case REG_RESOURCE_MEAN_ENERGY_ATTOJ:
        return s->resource_mean_energy_attoj;
    case REG_RESOURCE_LOSS_Q63:
        return s->resource_loss_q63;
    case REG_RESOURCE_DEPHASING_Q63:
        return s->resource_dephasing_q63;
    case REG_RESOURCE_ENV_HISTORY_CELLS:
        return s->resource_env_history_cells;
    case REG_RESOURCE_CUSTODY_TRANSITIONS:
        return s->resource_custody_transitions;
    case REG_RESOURCE_REUSE_COUNT:
        return s->resource_reuse_count;
    case REG_RESOURCE_DISCARDED_TRIALS:
        return s->resource_discarded_trials;
    case REG_RESOURCE_OUTPUT_HOLD_FS:
        return s->resource_output_hold_fs;
    case REG_RESOURCE_MAINTENANCE_OPS:
        return s->resource_maintenance_ops;
    case REG_RESOURCE_PRECISION_BITS:
        return 64;
    case REG_RESOURCE_SCHEMA:
        return PHASE_V12_RESOURCE_SCHEMA;
    case REG_RESOURCE_DIGEST_LO:
        return s->resource_digest_lo;
    case REG_RESOURCE_DIGEST_HI:
        return s->resource_digest_hi;
    case REG_RESOURCE_COMPILER_OPS:
        return PHASE_V12_RESOURCE_UNKNOWN;
    case REG_RESOURCE_CONTROLLER_OPS:
        return reported_resource_control_words(s);
    case REG_RESOURCE_CONSTRUCTION_OPS:
        return PHASE_V12_RESOURCE_UNKNOWN;
    case REG_RESOURCE_SECRET_ENTROPY_BITS:
        return 4;
    case REG_RESOURCE_CARRIER_PHOTON_NUMBER:
        return resource_carrier_photon_number(s);
    case REG_ADAPTER_STATE:
        return s->adapter_state;
    case REG_ADAPTER_AUTH_ACCEPTED:
        return s->adapter_auth_accepted;
    case REG_ADAPTER_AUTH_REJECTED:
        return s->adapter_auth_rejected;
    case REG_ADAPTER_DISPATCHES:
        return s->adapter_dispatches;
    case REG_ADAPTER_COMPLETIONS:
        return s->adapter_completions;
    case REG_ADAPTER_CANCELS:
        return s->adapter_cancels;
    case REG_ADAPTER_VIRTUAL_TICK:
        return s->adapter_virtual_tick;
    case REG_ADAPTER_DEADLINE_TICK:
        return s->adapter_deadline_tick;
    default:
        g_assert_not_reached();
    }
}

static void phase_mmio_write(void *opaque, hwaddr address, uint64_t value,
                             unsigned size)
{
    PhaseQemuV12State *s = opaque;
    uint32_t request_error;

    if (!access_width_valid(address, size)) {
        s->error = ERR_BAD_ARGUMENT;
        return;
    }
    /* Count every width-correct BAR write, including rejected control traffic. */
    if (s->resource_control_words >= PHASE_V12_RESOURCE_UNKNOWN - 1) {
        s->error = ERR_OVERFLOW;
        return;
    }
    s->resource_control_words++;
    if (s->resource_sealed &&
        !(address == REG_COMMAND &&
          (value == CMD_ACK_RESPONSE || value == CMD_BEGIN_REUSE))) {
        s->error = ERR_RESPONSE_LOCKED;
        return;
    }
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
        } else if (s->descriptor_sealed || s->lifecycle != LIFE_ISOLATED ||
                   value >= PHASE_V12_DESCRIPTOR_WORDS) {
            s->error = ERR_BAD_STATE;
        } else {
            s->descriptor_index = value;
        }
        break;
    case REG_DESCRIPTOR_WORD:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
        } else if (s->descriptor_sealed || s->lifecycle != LIFE_ISOLATED ||
                   s->descriptor_index >= PHASE_V12_DESCRIPTOR_WORDS) {
            s->error = ERR_BAD_STATE;
        } else {
            s->descriptor[s->descriptor_index] = value;
        }
        break;
    case REG_DESCRIPTOR_LENGTH:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
        } else if (s->descriptor_sealed || s->lifecycle != LIFE_ISOLATED ||
                   value != PHASE_V12_DESCRIPTOR_WORDS) {
            s->error = ERR_DESCRIPTOR_INVALID;
        } else {
            s->descriptor_length = value;
        }
        break;
    case REG_BOUNDARY_SCHEMA:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
        } else if (s->descriptor_sealed || s->lifecycle != LIFE_ISOLATED ||
                   value != 1) {
            s->error = ERR_DESCRIPTOR_INVALID;
        } else {
            s->boundary_schema = value;
        }
        break;
    default:
        /* Every other defined BAR location is read-only. */
        s->error = ERR_BAD_ARGUMENT;
        break;
    }
}

static const MemoryRegionOps phase_mmio_ops = {
    .read = phase_mmio_read,
    .write = phase_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
        .unaligned = true,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static void private_residue_set(Object *object, Visitor *visitor,
                                const char *name, void *opaque, Error **errp)
{
    PhaseQemuV12State *s = PHASE_QEMU_V12(object);
    unsigned slot = GPOINTER_TO_UINT(opaque);
    uint32_t value;

    if (!visit_type_uint32(visitor, name, &value, errp)) {
        return;
    }
    if (!s->realized || !s->test_provider_enabled) {
        error_setg(errp, "%s is disabled", name);
        return;
    }
    if (s->backend_id == PHASE_V12_BACKEND_EXTERNAL) {
        s->error = ERR_ADAPTER_AUTH_BINDING;
        error_setg(errp, "%s cannot bypass the external adapter envelope", name);
        return;
    }
    if (!s->private_armed || s->lifecycle != LIFE_PRIVATE_ARMED ||
        s->arm_generation != s->generation || s->arm_nonce == 0) {
        error_setg(errp, "%s requires the current ARM_PRIVATE lineage", name);
        return;
    }
    if (slot >= PHASE_V12_PRIVATE_SLOTS || s->private_bound[slot]) {
        error_setg(errp, "%s is already bound", name);
        return;
    }
    if (value > 2) {
        error_setg(errp, "%s residue must be in [0,2]", name);
        return;
    }
    if (s->resource_control_words >= PHASE_V12_RESOURCE_UNKNOWN - 1) {
        s->error = ERR_OVERFLOW;
        error_setg(errp, "%s resource counter is exhausted", name);
        return;
    }

    s->private_residue[slot] = value;
    s->private_bound[slot] = true;
    s->private_ready_mask |= 1u << slot;
    s->resource_control_words++;
    if (s->private_ready_mask == PHASE_V12_PRIVATE_READY_ALL) {
        s->lifecycle = LIFE_PRIVATE_READY;
    }
}

static void adapter_envelope_set(Object *object, Visitor *visitor,
                                 const char *name, void *opaque,
                                 Error **errp)
{
    PhaseQemuV12State *s = PHASE_QEMU_V12(object);
    unsigned property_slot = GPOINTER_TO_UINT(opaque);
    uint64_t value;
    uint64_t payload;
    uint64_t tag;
    unsigned residue;
    unsigned wire_slot;
    unsigned generation;
    unsigned sequence;
    unsigned issued;
    unsigned expires;
    uint32_t reject = ERR_NONE;

    if (!visit_type_uint64(visitor, name, &value, errp)) {
        return;
    }
    if (!s->realized || !s->test_provider_enabled ||
        !s->test_adapter_enabled ||
        s->backend_id != PHASE_V12_BACKEND_EXTERNAL) {
        error_setg(errp, "%s is disabled", name);
        return;
    }
    /*
     * A committed response exposes an immutable response-local resource
     * vector.  Reject late QOM writes without touching the live error or
     * accounting fields; ACK is the only operation that may follow sealing.
     */
    if (s->resource_sealed) {
        error_setg(errp, "%s response resources are sealed", name);
        return;
    }
    if (s->resource_control_words >= PHASE_V12_RESOURCE_UNKNOWN - 1) {
        s->error = ERR_OVERFLOW;
        error_setg(errp, "%s resource counter is exhausted", name);
        return;
    }
    s->resource_control_words++;
    if (!s->private_armed || s->lifecycle != LIFE_PRIVATE_ARMED ||
        s->arm_generation != s->generation || s->arm_nonce == 0) {
        reject = ERR_PRIVATE_LINEAGE;
    }

    payload = value & ((UINT64_C(1) << 43) - 1);
    tag = value >> 43;
    residue = payload & 3u;
    wire_slot = (payload >> 2) & 1u;
    generation = (payload >> 3) & 0xffffu;
    sequence = (payload >> 19) & 0xffu;
    issued = (payload >> 27) & 0xffu;
    expires = (payload >> 35) & 0xffu;

    if (reject == ERR_NONE && tag != adapter_test_tag(s, payload)) {
        reject = ERR_ADAPTER_AUTH_BINDING;
    } else if (reject == ERR_NONE &&
               (property_slot >= PHASE_V12_PRIVATE_SLOTS ||
                wire_slot != property_slot || residue > 2)) {
        reject = ERR_ADAPTER_SLOT;
    } else if (reject == ERR_NONE && s->private_bound[property_slot]) {
        reject = ERR_ADAPTER_REPLAY;
    } else if (reject == ERR_NONE &&
               (sequence != property_slot + 1 ||
                (property_slot == 1 && !s->private_bound[0]))) {
        reject = ERR_ADAPTER_ORDER;
    } else if (reject == ERR_NONE && generation != s->generation) {
        reject = ERR_GENERATION_MISMATCH;
    } else if (reject == ERR_NONE &&
               (s->adapter_virtual_tick > 0xff ||
                issued > s->adapter_virtual_tick ||
                expires <= s->adapter_virtual_tick)) {
        reject = ERR_ADAPTER_EXPIRED;
    }

    if (reject != ERR_NONE) {
        s->error = reject;
        if (!counter_can_add(s, s->adapter_auth_rejected, 1)) {
            error_setg(errp, "%s authentication counter is exhausted", name);
            return;
        }
        s->adapter_auth_rejected++;
        error_setg(errp, "%s rejected external adapter envelope (%u)",
                   name, reject);
        return;
    }
    if (!counter_can_add(s, s->adapter_auth_accepted, 1)) {
        error_setg(errp, "%s authentication counter is exhausted", name);
        return;
    }

    s->adapter_envelope[property_slot] = value;
    s->private_residue[property_slot] = residue;
    s->private_bound[property_slot] = true;
    s->private_ready_mask |= 1u << property_slot;
    s->adapter_auth_accepted++;
    s->error = ERR_NONE;
    if (s->private_ready_mask == PHASE_V12_PRIVATE_READY_ALL) {
        s->adapter_authenticated_lineage = true;
        s->lifecycle = LIFE_PRIVATE_READY;
    }
}

enum PhaseV12ObserverId {
    OBS_PHASE_A,
    OBS_PHASE_B,
    OBS_PREPARE_COUNT,
    OBS_CLIENT_SUPPLY_COUNT,
    OBS_ALLOCATION_LO,
    OBS_ALLOCATION_HI,
    OBS_SAME_BACKING,
    OBS_KR_RETURN,
    OBS_FACTORIZED,
    OBS_PORT_CLEAR,
    OBS_ENV_FACTORED,
    OBS_RETURN_CLASS,
    OBS_ARM_NONCE,
};

static bool test_observer_enabled(PhaseQemuV12State *s, const char *name,
                                  Error **errp)
{
    if (!s->realized || !s->test_provider_enabled) {
        error_setg(errp, "%s is disabled outside test-provider mode", name);
        return false;
    }
    return true;
}

static void test_observer_u32_get(Object *object, Visitor *visitor,
                                  const char *name, void *opaque,
                                  Error **errp)
{
    PhaseQemuV12State *s = PHASE_QEMU_V12(object);
    uint32_t value;

    if (!test_observer_enabled(s, name, errp)) {
        return;
    }
    switch (GPOINTER_TO_UINT(opaque)) {
    case OBS_PHASE_A:
        value = s->observed_phase[0];
        break;
    case OBS_PHASE_B:
        value = s->observed_phase[1];
        break;
    case OBS_SAME_BACKING:
        value = s->observed_same_backing;
        break;
    case OBS_KR_RETURN:
        value = s->observed_kr_return;
        break;
    case OBS_FACTORIZED:
        value = s->observed_factorized;
        break;
    case OBS_PORT_CLEAR:
        value = s->observed_port_clear;
        break;
    case OBS_ENV_FACTORED:
        value = s->observed_env_factored;
        break;
    case OBS_RETURN_CLASS:
        value = s->return_class;
        break;
    default:
        error_setg(errp, "%s has an invalid observer selector", name);
        return;
    }
    visit_type_uint32(visitor, name, &value, errp);
}

static void test_observer_u64_get(Object *object, Visitor *visitor,
                                  const char *name, void *opaque,
                                  Error **errp)
{
    PhaseQemuV12State *s = PHASE_QEMU_V12(object);
    uint64_t value;

    if (!test_observer_enabled(s, name, errp)) {
        return;
    }
    switch (GPOINTER_TO_UINT(opaque)) {
    case OBS_PREPARE_COUNT:
        value = s->prepare_count;
        break;
    case OBS_CLIENT_SUPPLY_COUNT:
        value = s->client_supply_count;
        break;
    case OBS_ALLOCATION_LO:
        value = s->allocation_id_lo;
        break;
    case OBS_ALLOCATION_HI:
        value = s->allocation_id_hi;
        break;
    case OBS_ARM_NONCE:
        value = s->arm_nonce;
        break;
    default:
        error_setg(errp, "%s has an invalid observer selector", name);
        return;
    }
    visit_type_uint64(visitor, name, &value, errp);
}

static void enter_migration_sham(PhaseQemuV12State *s)
{
    if (s->ops && s->ops->sanitize) {
        s->ops->sanitize(s);
    } else {
        memset(s->density, 0, sizeof(s->density));
        memset(s->scratch, 0, sizeof(s->scratch));
        s->density_denom_power = 0;
    }
    clear_descriptor(s);
    clear_private(s);
    clear_boundary(s);
    clear_observers(s);
    release_tags(s);
    s->migration_sham_latched = true;
    s->carrier_present = false;
    s->prepared = false;
    s->source_isolated = false;
    s->spent = true;
    s->restored = false;
    s->return_verified = false;
    s->snapshot_lineage = true;
    s->port_clear = true;
    s->outputs_held = false;
    s->env_factored = false;
    s->same_backing = false;
    s->reuse_qualified = false;
    s->adapter_authenticated_lineage = false;
    s->adapter_state = ADAPTER_SHAM;
    s->adapter_deadline_tick = 0;
    s->return_class = RETURN_FAILED;
    s->error = ERR_SNAPSHOT_LINEAGE;
    s->lifecycle = LIFE_SHAM;
    s->allocation_id_lo = 0;
    s->allocation_id_hi = 0;
    s->custody_epoch = 0;
    s->preparation_receipt_lo = 0;
    s->preparation_receipt_hi = 0;
    s->return_receipt_lo = 0;
    s->return_receipt_hi = 0;
    s->exact_return_generation = 0;
    s->backing_address = 0;
}

static void phase_qemu_v12_reset(DeviceState *device)
{
    PhaseQemuV12State *s = PHASE_QEMU_V12(device);

    if (s->migration_sham_latched || s->snapshot_lineage ||
        s->lifecycle == LIFE_SHAM) {
        enter_migration_sham(s);
        return;
    }

    /*
     * Reset may be canonical only before any external envelope or coherent
     * transaction state exists.  Once private material has been accepted,
     * work is pending, or a response is held, reset cancels once and burns
     * the lineage into the same reset-irreversible SHAM used by migration.
     */
    if (s->backend_id == PHASE_V12_BACKEND_EXTERNAL &&
        (s->private_ready_mask != 0 ||
         s->adapter_state == ADAPTER_WAITING_A ||
         s->adapter_state == ADAPTER_WAITING_B ||
         s->outputs_held || s->response_ready)) {
        if ((s->adapter_state == ADAPTER_WAITING_A ||
             s->adapter_state == ADAPTER_WAITING_B) &&
            s->ops && s->ops->cancel) {
            s->ops->cancel(s);
        }
        enter_migration_sham(s);
        return;
    }

    s->carrier_present = s->configured_carrier_present;
    s->leased = false;
    s->prepared = false;
    s->source_isolated = false;
    s->response_ready = false;
    s->response_acked = false;
    s->spent = false;
    s->restored = false;
    s->return_verified = false;
    s->snapshot_lineage = false;
    s->port_clear = true;
    s->outputs_held = false;
    s->env_factored = true;
    s->resource_sealed = false;
    s->reuse_qualified = false;
    s->same_backing = false;
    s->adapter_authenticated_lineage = false;
    s->error = ERR_NONE;
    s->lifecycle = LIFE_EMPTY;
    s->return_class = RETURN_NONE;
    s->owner_tag = 0;
    s->program_tag = 0;
    s->request_owner_tag = 0;
    s->request_program_tag = 0;
    s->request_generation = 0;
    s->generation = 1;
    s->exact_return_generation = 0;
    s->arg0 = 0;
    s->arg1 = 0;
    s->density_denom_power = 0;
    s->migration_marker = PHASE_V12_MIGRATION_MARKER;
    s->adapter_state = s->test_adapter_enabled ? ADAPTER_READY :
                                                  ADAPTER_DISCONNECTED;
    s->virtual_cycles = 0;
    s->allocation_id_lo = 0;
    s->allocation_id_hi = 0;
    s->custody_epoch = 0;
    s->preparation_receipt_lo = 0;
    s->preparation_receipt_hi = 0;
    s->return_receipt_lo = 0;
    s->return_receipt_hi = 0;
    s->resource_digest_lo = 0;
    s->resource_digest_hi = 0;
    s->sealed_resource_control_words = 0;
    s->adapter_virtual_tick = 0;
    s->adapter_deadline_tick = 0;
    s->adapter_auth_accepted = 0;
    s->adapter_auth_rejected = 0;
    s->adapter_dispatches = 0;
    s->adapter_completions = 0;
    s->adapter_cancels = 0;
    memset(s->adapter_envelope, 0, sizeof(s->adapter_envelope));
    s->resource_query_applications = 0;
    s->resource_return_checks = 0;
    s->resource_environment_ops = 0;
    s->resource_control_words = 0;
    s->resource_preparation_ops = 0;
    s->resource_certification_ops = 0;
    s->resource_logical_queries = 0;
    s->resource_duration_fs = PHASE_V12_RESOURCE_UNKNOWN;
    s->resource_port_bandwidth_hz = PHASE_V12_RESOURCE_UNKNOWN;
    s->resource_action_q40_rad = 0;
    s->resource_mean_energy_attoj = PHASE_V12_RESOURCE_UNKNOWN;
    s->resource_loss_q63 = s->backend_id == PHASE_V12_BACKEND_EXTERNAL ?
                           PHASE_V12_RESOURCE_UNKNOWN : 0;
    s->resource_dephasing_q63 = s->backend_id == PHASE_V12_BACKEND_EXTERNAL ?
                                PHASE_V12_RESOURCE_UNKNOWN : 0;
    s->resource_env_history_cells = 0;
    s->resource_custody_transitions = 0;
    s->resource_reuse_count = 0;
    s->resource_discarded_trials = 0;
    s->resource_output_hold_fs = PHASE_V12_RESOURCE_UNKNOWN;
    s->resource_maintenance_ops = PHASE_V12_RESOURCE_UNKNOWN;
    s->prepare_count = 0;
    s->client_supply_count = 0;
    s->backing_address = 0;
    if (s->ops && s->ops->sanitize) {
        s->ops->sanitize(s);
    } else {
        memset(s->density, 0, sizeof(s->density));
        memset(s->scratch, 0, sizeof(s->scratch));
    }
    clear_descriptor(s);
    clear_private(s);
    clear_boundary(s);
    clear_observers(s);
}

static int phase_qemu_v12_pre_save(void *opaque)
{
    PhaseQemuV12State *s = opaque;

    /* Migration burns the source lineage before any bytes are serialized. */
    if ((s->adapter_state == ADAPTER_WAITING_A ||
         s->adapter_state == ADAPTER_WAITING_B) &&
        s->ops && s->ops->cancel) {
        s->ops->cancel(s);
    }
    enter_migration_sham(s);
    return 0;
}

static int phase_qemu_v12_post_load(void *opaque, int version_id)
{
    PhaseQemuV12State *s = opaque;

    (void)version_id;
    if (s->migration_marker != PHASE_V12_MIGRATION_MARKER ||
        s->backend_id != s->configured_backend_id ||
        s->fault_mode != s->configured_fault_mode ||
        s->test_provider_enabled != s->configured_test_provider_enabled ||
        s->test_adapter_enabled != s->configured_test_adapter_enabled ||
        s->test_adapter_mode != s->configured_test_adapter_mode ||
        s->open_model_q32 != s->configured_open_model_q32 ||
        s->generation == 0) {
        return -EINVAL;
    }

    /* Incoming state is always a sanitized, reset-irreversible lineage sham. */
    enter_migration_sham(s);
    return 0;
}

static const VMStateDescription vmstate_phase_qemu_v12 = {
    .name = "phase-qemu-v12",
    .version_id = 1,
    .minimum_version_id = 1,
    .pre_save = phase_qemu_v12_pre_save,
    .post_load = phase_qemu_v12_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, PhaseQemuV12State),
        VMSTATE_UINT32(migration_marker, PhaseQemuV12State),
        VMSTATE_UINT32(backend_id, PhaseQemuV12State),
        VMSTATE_BOOL(carrier_present, PhaseQemuV12State),
        VMSTATE_BOOL(test_provider_enabled, PhaseQemuV12State),
        VMSTATE_BOOL(test_adapter_enabled, PhaseQemuV12State),
        VMSTATE_UINT32(test_adapter_mode, PhaseQemuV12State),
        VMSTATE_UINT32(fault_mode, PhaseQemuV12State),
        VMSTATE_UINT32(open_model_q32, PhaseQemuV12State),
        VMSTATE_UINT32(generation, PhaseQemuV12State),
        VMSTATE_END_OF_LIST()
    }
};

static void phase_qemu_v12_realize(PCIDevice *pci_device, Error **errp)
{
    PhaseQemuV12State *s = PHASE_QEMU_V12(pci_device);

    s->ops = backend_ops_for_id(s->backend_id);
    if (!s->ops) {
        error_setg(errp,
                   "phase-qemu-v12 backend-id must be 0x0b01, 0x0b02, or 0x0b80");
        return;
    }
    if (s->fault_mode > FAULT_MAX) {
        error_setg(errp, "phase-qemu-v12 test-fault-mode must be in [0,7]");
        return;
    }
    if (s->test_adapter_mode > ADAPTER_MODE_MAX) {
        error_setg(errp, "phase-qemu-v12 test-adapter-mode must be in [0,1]");
        return;
    }
    if (s->test_adapter_enabled &&
        (s->backend_id != PHASE_V12_BACKEND_EXTERNAL ||
         !s->test_provider_enabled)) {
        error_setg(errp,
                   "phase-qemu-v12 test-adapter-enabled requires external backend and test provider");
        return;
    }
    if (s->backend_id != PHASE_V12_BACKEND_OPEN &&
        s->open_model_q32 != 0) {
        error_setg(errp,
                   "phase-qemu-v12 open-model-q32 requires backend-id=0x0b02");
        return;
    }
    s->configured_backend_id = s->backend_id;
    s->configured_carrier_present = s->carrier_present;
    s->configured_test_provider_enabled = s->test_provider_enabled;
    s->configured_test_adapter_enabled = s->test_adapter_enabled;
    s->configured_test_adapter_mode = s->test_adapter_mode;
    s->configured_fault_mode = s->fault_mode;
    s->configured_open_model_q32 = s->open_model_q32;
    memory_region_init_io(&s->mmio, OBJECT(s), &phase_mmio_ops, s,
                          "phase-qemu-v12-mmio", PHASE_V12_BAR_SIZE);
    pci_register_bar(pci_device, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
    s->realized = true;
}

static void phase_qemu_v12_unrealize(DeviceState *device)
{
    PhaseQemuV12State *s = PHASE_QEMU_V12(device);

    if ((s->adapter_state == ADAPTER_WAITING_A ||
         s->adapter_state == ADAPTER_WAITING_B) &&
        s->ops && s->ops->cancel) {
        s->ops->cancel(s);
    }
    if (s->ops && s->ops->sanitize) {
        s->ops->sanitize(s);
    } else {
        memset(s->density, 0, sizeof(s->density));
        memset(s->scratch, 0, sizeof(s->scratch));
    }
    clear_private(s);
    clear_descriptor(s);
    clear_boundary(s);
    clear_observers(s);
    release_tags(s);
    s->adapter_authenticated_lineage = false;
    s->realized = false;
}

static void phase_qemu_v12_instance_init(Object *object)
{
    PhaseQemuV12State *s = PHASE_QEMU_V12(object);

    s->carrier_present = true;
    s->backend_id = PHASE_V12_BACKEND_IDEAL;
    s->fault_mode = FAULT_NONE;
    s->open_model_q32 = 0;
    s->test_provider_enabled = false;
    s->test_adapter_enabled = false;
    s->test_adapter_mode = ADAPTER_MODE_COMPLETE;
    clear_observers(s);

    object_property_add(object, "test-private-a", "uint32", NULL,
                        private_residue_set, NULL, GUINT_TO_POINTER(0));
    object_property_add(object, "test-private-b", "uint32", NULL,
                        private_residue_set, NULL, GUINT_TO_POINTER(1));
    object_property_add(object, "test-adapter-envelope-a", "uint64", NULL,
                        adapter_envelope_set, NULL, GUINT_TO_POINTER(0));
    object_property_add(object, "test-adapter-envelope-b", "uint64", NULL,
                        adapter_envelope_set, NULL, GUINT_TO_POINTER(1));

    object_property_add(object, "test-observe-phase-a", "uint32",
        test_observer_u32_get, NULL, NULL, GUINT_TO_POINTER(OBS_PHASE_A));
    object_property_add(object, "test-observe-phase-b", "uint32",
        test_observer_u32_get, NULL, NULL, GUINT_TO_POINTER(OBS_PHASE_B));
    object_property_add(object, "test-observe-prepare-count", "uint64",
        test_observer_u64_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_PREPARE_COUNT));
    object_property_add(object, "test-observe-client-supply-count", "uint64",
        test_observer_u64_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_CLIENT_SUPPLY_COUNT));
    object_property_add(object, "test-observe-allocation-lo", "uint64",
        test_observer_u64_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_ALLOCATION_LO));
    object_property_add(object, "test-observe-allocation-hi", "uint64",
        test_observer_u64_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_ALLOCATION_HI));
    object_property_add(object, "test-observe-same-backing", "uint32",
        test_observer_u32_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_SAME_BACKING));
    object_property_add(object, "test-observe-kr-return", "uint32",
        test_observer_u32_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_KR_RETURN));
    object_property_add(object, "test-observe-factorized", "uint32",
        test_observer_u32_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_FACTORIZED));
    object_property_add(object, "test-observe-port-clear", "uint32",
        test_observer_u32_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_PORT_CLEAR));
    object_property_add(object, "test-observe-env-factored", "uint32",
        test_observer_u32_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_ENV_FACTORED));
    object_property_add(object, "test-observe-return-class", "uint32",
        test_observer_u32_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_RETURN_CLASS));
    object_property_add(object, "test-observe-arm-nonce", "uint64",
        test_observer_u64_get, NULL, NULL,
        GUINT_TO_POINTER(OBS_ARM_NONCE));
}

static const Property phase_qemu_v12_properties[] = {
    DEFINE_PROP_UINT32("backend-id", PhaseQemuV12State, backend_id,
                       PHASE_V12_BACKEND_IDEAL),
    DEFINE_PROP_BOOL("carrier-present", PhaseQemuV12State,
                     carrier_present, true),
    DEFINE_PROP_BOOL("test-provider-enabled", PhaseQemuV12State,
                     test_provider_enabled, false),
    DEFINE_PROP_BOOL("test-adapter-enabled", PhaseQemuV12State,
                     test_adapter_enabled, false),
    DEFINE_PROP_UINT32("test-adapter-mode", PhaseQemuV12State,
                       test_adapter_mode, ADAPTER_MODE_COMPLETE),
    DEFINE_PROP_UINT32("test-fault-mode", PhaseQemuV12State,
                       fault_mode, FAULT_NONE),
    DEFINE_PROP_UINT32("open-model-q32", PhaseQemuV12State,
                       open_model_q32, 0),
};

static void phase_qemu_v12_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *device_class = DEVICE_CLASS(klass);
    PCIDeviceClass *pci_class = PCI_DEVICE_CLASS(klass);

    (void)data;
    pci_class->realize = phase_qemu_v12_realize;
    pci_class->vendor_id = PHASE_V12_VENDOR_ID;
    pci_class->device_id = PHASE_V12_DEVICE_ID;
    pci_class->revision = PHASE_V12_REVISION;
    pci_class->class_id = PCI_CLASS_OTHERS;
    device_class->vmsd = &vmstate_phase_qemu_v12;
    device_class->unrealize = phase_qemu_v12_unrealize;
    device_class_set_legacy_reset(device_class, phase_qemu_v12_reset);
    device_class_set_props(device_class, phase_qemu_v12_properties);
    set_bit(DEVICE_CATEGORY_MISC, device_class->categories);
}

static const TypeInfo phase_qemu_v12_info = {
    .name = TYPE_PHASE_QEMU_V12,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(PhaseQemuV12State),
    .instance_init = phase_qemu_v12_instance_init,
    .class_init = phase_qemu_v12_class_init,
    .interfaces = (const InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void phase_qemu_v12_register_types(void)
{
    type_register_static(&phase_qemu_v12_info);
}

type_init(phase_qemu_v12_register_types)
