/*
 * Phase-QEMU V13 hardware-adapter absence and attestation gate.
 *
 * V13 is a compiled QEMU PCI sibling behind the common Phase-QEMU register
 * contract.  The complete V12 register map is preserved as a prefix; V13
 * appends typed hardware-gate evidence registers after the locked V12
 * boundary.  There is deliberately no ideal algebra engine in this device.
 * The production hardware backend is unavailable in this milestone and may
 * not fall back to an internal model.
 *
 * An explicitly enabled OFFLINE_FIXTURE backend exercises fixed standard
 * selectors for enrollment, attestation age, replay, downgrade, and accepted
 * channel/appraisal metadata.  The C code selects seven predetermined failure
 * states; it does not parse or appraise real evidence.  Fixture evidence is
 * permanently typed as
 * OFFLINE_STANDARD_VECTOR / TEST_FIXTURE / PROTOCOL_CONFORMANCE_ONLY.  Every
 * nondestructively completed fixture dispatch therefore terminates in a
 * sealed failed-attempt receipt; acknowledgement is the only ordinary
 * transition from that receipt to SPENT.  Reset, migration, and unrealize
 * instead sanitize the lineage into SHAM without acknowledgement.  No path
 * can produce a physical sample or campaign certificate.
 *
 * Migration, reset after activity, and unrealize sanitize ephemeral lineage.
 * VMState contains no digest, nonce, receipt, evidence payload, lease, or live
 * session state.  This device establishes protocol controls only; it is not
 * evidence that hardware exists or that a physical experiment ran.
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

#define TYPE_PHASE_QEMU_V13 "phase-qemu-v13"
OBJECT_DECLARE_SIMPLE_TYPE(PhaseQemuV13State, PHASE_QEMU_V13)

#define PHASE_V13_VENDOR_ID PCI_VENDOR_ID_QEMU
#define PHASE_V13_DEVICE_ID 0x11fd
#define PHASE_V13_REVISION 0x01
#define PHASE_V13_MAGIC 0x50483133u /* PH13 */
#define PHASE_V13_ABI 0x00030000u
#define PHASE_V13_BAR_SIZE (4 * KiB)

#define PHASE_V13_BACKEND_HARDWARE 0x0d80u
#define PHASE_V13_BACKEND_OFFLINE_FIXTURE 0x0df0u
#define PHASE_V13_DESCRIPTOR_WORDS 8u
#define PHASE_V13_BOUNDARY_WORDS 16u
#define PHASE_V13_LOCKED_BOUNDARY UINT64_MAX
#define PHASE_V13_RESOURCE_UNKNOWN UINT64_MAX
#define PHASE_V13_RESOURCE_SCHEMA 0x00030001u
#define PHASE_V13_MIGRATION_MARKER 0x5031334du /* P13M */
#define PHASE_V13_BOUNDARY_HEADER UINT64_C(0x5031334600030080)
#define PHASE_V13_LEASE_TICKS UINT64_C(16)

/* Architecture promotion and anti-substitution gates. */
#define PHASE_V13_STANDALONE_TWIN_QUALIFIES 0
#define PHASE_V13_INTERNAL_IDEAL_SUBSTITUTION_ALLOWED 0
#define PHASE_V13_LIVE_HARDWARE_AVAILABLE 0
#define PHASE_V13_PHYSICAL_OUTPUT_ALLOWED 0
#define PHASE_V13_CAMPAIGN_STATISTICAL_CLASS_ALLOWED 0

/* Exact bounded authority tokens. */
#define PHASE_V13_CLAIM_AUTHORITY "COMPILED_COMMON_PHASE_QEMU_V13_FIXED_OFFLINE_HARDWARE_ADAPTER_SELECTOR_GATE_REJECTS_ABSENT_UNENROLLED_UNATTESTED_STALE_REPLAYED_DOWNGRADED_AND_TEST_FIXTURE_SESSIONS_WITHOUT_PUBLISHING_A_PHYSICAL_OUTPUT_OR_CAMPAIGN_STATISTICAL_CERTIFICATE_AND_WITH_EVERY_NONDESTRUCTIVELY_COMPLETED_DISPATCHED_ATTEMPT_TERMINAL_ACK_THEN_SPENT"
#define PHASE_V13_CLAIM_CEILING "HARDWARE_ABSENT_PROTOCOL_CONFORMANCE_ONLY_NO_AUTHENTICATED_LIVE_DEVICE_SESSION_NO_PHYSICAL_SAMPLE_NO_CAMPAIGN_STATISTICAL_CERTIFICATE_NO_CUSTODY_RETURN_RESTORATION_REUSE_ADVANTAGE_OR_M257_ESCAPE"
#define PHASE_V13_RESTORATION_AUTHORITY "NO_RESTORATION_CLAIM"
#define PHASE_V13_SCOPE_AUTHORITY "COMPILED_QEMU_10_2_4_PHASE_QEMU_V13_COMMON_PCI_BACKEND_HARDWARE_ABSENCE_AND_FIXED_OFFLINE_SYMBOLIC_SELECTOR_STATE_MACHINE_ONLY"
#define PHASE_V13_DISPOSITION_AUTHORITY "V13_ESTABLISHES_COMMON_BACKEND_FAIL_CLOSED_FIXED_SELECTOR_STATE_MACHINE_CONFORMANCE_FOR_DEVICE_ENROLLMENT_ATTESTATION_REPLAY_DOWNGRADE_AND_FIXTURE_DOMAIN_SEPARATION_WITHOUT_A_LIVE_HARDWARE_SESSION_OR_PHYSICAL_OUTPUT_DIRECT_EQUAL_ACCESS_PROTOCOL_COMPARATOR_CONTROLS_AND_M257_REMAINS_INTACT"
#define PHASE_V13_SUCCESSOR_AUTHORITY "USER_AUTHORIZED_PINNED_DEVICE_ENROLLMENT_FOLLOWED_BY_A_PREREGISTERED_BLINDED_DUAL_RAIL_DISPERSIVE_CAPTURE_CAMPAIGN_WITH_DEVICE_SIGNED_RAW_MANIFESTS_INDEPENDENT_MEASUREMENT_AND_FAMILYWISE_STATISTICAL_VALIDATION_BEHIND_THE_COMMON_PHASE_QEMU_BACKEND"

/*
 * The V11/V12 prefix is frozen byte-for-byte by name and offset.  New V13
 * registers begin at 0x280, immediately after REG_BOUNDARY_LAST.
 */
enum PhaseV13Register {
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

    REG_GATE_STATE = 0x280,
    REG_EVIDENCE_ORIGIN = 0x284,
    REG_CHANNEL_SECURITY = 0x288,
    REG_DEVICE_APPRAISAL = 0x28c,
    REG_MEASUREMENT_CLASS = 0x290,
    REG_CUSTODY_PROVENANCE = 0x294,
    REG_RESOURCE_PROVENANCE = 0x298,
    REG_TRUST_DOMAIN = 0x29c,
    REG_REJECTION_REASON = 0x2a0,
    REG_GATE_FLAGS = 0x2a4,
    REG_SECURITY_VERSION = 0x2a8,
    REG_MIN_SECURITY_VERSION = 0x2ac,
    REG_SESSION_EPOCH = 0x2b0,
    REG_ATTESTATION_NONCE = 0x2b8,
    REG_ATTESTATION_AGE_TICKS = 0x2c0,
    REG_MAX_ATTESTATION_AGE_TICKS = 0x2c8,
    REG_DEVICE_ID_DIGEST_LO = 0x2d0,
    REG_DEVICE_ID_DIGEST_HI = 0x2d8,
    REG_MEASUREMENT_DIGEST_LO = 0x2e0,
    REG_MEASUREMENT_DIGEST_HI = 0x2e8,
    REG_CUSTODY_DIGEST_LO = 0x2f0,
    REG_CUSTODY_DIGEST_HI = 0x2f8,
    REG_RESOURCE_PROVENANCE_DIGEST_LO = 0x300,
    REG_RESOURCE_PROVENANCE_DIGEST_HI = 0x308,
    REG_GATE_DISPATCHES = 0x310,
    REG_GATE_TERMINAL_FAILURES = 0x318,
    REG_GATE_ACKS = 0x320,
    REG_GATE_REPLAY_REJECTS = 0x328,
    REG_GATE_DOWNGRADE_REJECTS = 0x330,
    REG_GATE_FIXTURE_REJECTS = 0x338,
    REG_GATE_VIRTUAL_TICK = 0x340,
    REG_LEASE_EXPIRY_TICK = 0x348,
};

enum PhaseV13Command {
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

enum PhaseV13Lifecycle {
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

enum PhaseV13ReturnClass {
    RETURN_NONE = 0,
    RETURN_EXACT_FORMAL = 1,
    RETURN_APPROX_MODEL = 2,
    RETURN_FAILED = 3,
    RETURN_STATISTICAL_ONLY = 4,
};

enum PhaseV13Error {
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
    ERR_HARDWARE_ABSENT = 38,
    ERR_DEVICE_UNENROLLED = 39,
    ERR_DEVICE_UNATTESTED = 40,
    ERR_ATTESTATION_STALE = 41,
    ERR_ATTESTATION_REPLAYED = 42,
    ERR_SECURITY_DOWNGRADE = 43,
    ERR_FIXTURE_TRUST_DOMAIN = 44,
    ERR_EVIDENCE_TYPE_MISMATCH = 45,
    ERR_LEASE_EXPIRED = 46,
    ERR_NO_PHYSICAL_OUTPUT = 47,
};

enum PhaseV13Status {
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
    ST_FIXTURE_DOMAIN = 1u << 26,
    ST_TERMINAL_FAILURE_RECEIPT = 1u << 27,
    ST_HARDWARE_ABSENT = 1u << 28,
};

enum PhaseV13Capability {
    CAP_COMMON_V12_REGISTER_PREFIX = 1u << 0,
    CAP_SWAPPABLE_BACKEND = 1u << 1,
    CAP_HARDWARE_ABSENCE_GATE = 1u << 2,
    CAP_TYPED_EVIDENCE = 1u << 3,
    CAP_ENROLLMENT_GATE = 1u << 4,
    CAP_ATTESTATION_AGE_GATE = 1u << 5,
    CAP_REPLAY_GATE = 1u << 6,
    CAP_DOWNGRADE_GATE = 1u << 7,
    CAP_FIXTURE_DOMAIN_SEPARATION = 1u << 8,
    CAP_FAILURE_RECEIPT_ACK_SPENT = 1u << 9,
    CAP_MIGRATION_SHAM = 1u << 10,
    CAP_NO_PHYSICAL_OUTPUT = 1u << 11,
    CAP_NO_STATISTICAL_CERTIFICATE = 1u << 12,
    CAP_OFFLINE_STANDARD_VECTORS = 1u << 13,
    CAP_PRODUCTION_FAIL_CLOSED = 1u << 14,
};

enum PhaseV13GateState {
    GATE_COLD = 0,
    GATE_PREFLIGHT_REJECTED = 1,
    GATE_LEASED = 2,
    GATE_ATTESTATION_ARMED = 3,
    GATE_DISPATCHING = 4,
    GATE_TERMINAL_FAILED = 5,
    GATE_SPENT = 6,
    GATE_SHAM = 7,
};

enum PhaseV13EvidenceOrigin {
    EVIDENCE_ORIGIN_NONE = 0,
    EVIDENCE_ORIGIN_LIVE_DEVICE = 1,
    EVIDENCE_ORIGIN_OFFLINE_STANDARD_VECTOR = 2,
};

enum PhaseV13ChannelSecurity {
    CHANNEL_SECURITY_NONE = 0,
    CHANNEL_SECURITY_AUTHENTICATED_CONFIDENTIAL = 1,
    CHANNEL_SECURITY_OFFLINE_VECTOR_INTEGRITY = 2,
    CHANNEL_SECURITY_REJECTED = 3,
};

enum PhaseV13DeviceAppraisal {
    DEVICE_APPRAISAL_NONE = 0,
    DEVICE_APPRAISAL_ENROLLED_ATTESTED = 1,
    DEVICE_APPRAISAL_UNENROLLED = 2,
    DEVICE_APPRAISAL_UNATTESTED = 3,
    DEVICE_APPRAISAL_STALE = 4,
    DEVICE_APPRAISAL_REPLAYED = 5,
    DEVICE_APPRAISAL_DOWNGRADED = 6,
    DEVICE_APPRAISAL_FIXTURE_STANDARD_ACCEPTED = 7,
};

enum PhaseV13MeasurementClass {
    MEASUREMENT_CLASS_NONE = 0,
    MEASUREMENT_CLASS_PHYSICAL_SAMPLE = 1,
    MEASUREMENT_CLASS_CAMPAIGN_STATISTICAL_CERTIFICATE = 2,
    MEASUREMENT_CLASS_PROTOCOL_CONFORMANCE_ONLY = 3,
};

enum PhaseV13CustodyProvenance {
    CUSTODY_PROVENANCE_NONE = 0,
    CUSTODY_PROVENANCE_LIVE_DEVICE_SIGNED = 1,
    CUSTODY_PROVENANCE_OFFLINE_VECTOR = 2,
    CUSTODY_PROVENANCE_REJECTED = 3,
};

enum PhaseV13ResourceProvenance {
    RESOURCE_PROVENANCE_NONE = 0,
    RESOURCE_PROVENANCE_LIVE_DEVICE_MANIFEST = 1,
    RESOURCE_PROVENANCE_OFFLINE_VECTOR = 2,
    RESOURCE_PROVENANCE_UNKNOWN = 3,
};

enum PhaseV13TrustDomain {
    TRUST_DOMAIN_NONE = 0,
    TRUST_DOMAIN_PRODUCTION = 1,
    TRUST_DOMAIN_TEST_FIXTURE = 2,
};

enum PhaseV13GateFlag {
    GATE_FLAG_CHANNEL_METADATA_ACCEPTED = 1u << 0,
    GATE_FLAG_APPRAISAL_METADATA_ACCEPTED = 1u << 1,
    GATE_FLAG_ENROLLMENT_PRESENT = 1u << 2,
    GATE_FLAG_ATTESTATION_PRESENT = 1u << 3,
    GATE_FLAG_FRESH = 1u << 4,
    GATE_FLAG_NONREPLAY = 1u << 5,
    GATE_FLAG_SECURITY_VERSION_ACCEPTED = 1u << 6,
    GATE_FLAG_FIXTURE_ONLY = 1u << 7,
    GATE_FLAG_NO_PHYSICAL_OUTPUT = 1u << 8,
    GATE_FLAG_NO_CAMPAIGN_CERTIFICATE = 1u << 9,
};

enum PhaseV13StandardVector {
    STANDARD_VECTOR_ACCEPTED_METADATA = 0,
    STANDARD_VECTOR_UNENROLLED = 1,
    STANDARD_VECTOR_UNATTESTED = 2,
    STANDARD_VECTOR_STALE = 3,
    STANDARD_VECTOR_REPLAYED = 4,
    STANDARD_VECTOR_DOWNGRADED = 5,
    STANDARD_VECTOR_TYPE_MISMATCH = 6,
    STANDARD_VECTOR_MAX = STANDARD_VECTOR_TYPE_MISMATCH,
};

typedef struct PhaseV13BackendOps {
    uint32_t id;
    const char *name;
    bool (*lease_preflight)(PhaseQemuV13State *s);
    uint32_t (*dispatch)(PhaseQemuV13State *s);
    void (*cancel)(PhaseQemuV13State *s);
    void (*sanitize)(PhaseQemuV13State *s);
} PhaseV13BackendOps;

struct PhaseQemuV13State {
    PCIDevice parent_obj;
    MemoryRegion mmio;

    const PhaseV13BackendOps *ops;
    bool realized;
    bool test_fixture_enabled;
    bool configured_test_fixture_enabled;
    bool leased;
    bool prepared;
    bool source_isolated;
    bool descriptor_sealed;
    bool attestation_armed;
    bool response_ready;
    bool response_acked;
    bool receipt_held;
    bool spent;
    bool snapshot_lineage;
    bool terminal_failure_receipt;
    bool dispatched_lineage;
    bool cancel_observed;

    uint32_t backend_id;
    uint32_t configured_backend_id;
    uint32_t standard_vector_id;
    uint32_t configured_standard_vector_id;
    uint32_t error;
    uint32_t lifecycle;
    uint32_t return_class;
    uint32_t generation;
    uint32_t arg0;
    uint32_t arg1;
    uint32_t request_owner;
    uint32_t request_program;
    uint32_t request_generation;
    uint32_t owner_tag;
    uint32_t program_tag;
    uint32_t descriptor_index;
    uint32_t descriptor_length;
    uint32_t boundary_schema;
    uint32_t migration_marker;
    uint32_t gate_state;
    uint32_t evidence_origin;
    uint32_t channel_security;
    uint32_t device_appraisal;
    uint32_t measurement_class;
    uint32_t custody_provenance;
    uint32_t resource_provenance;
    uint32_t trust_domain;
    uint32_t rejection_reason;
    uint32_t gate_flags;
    uint32_t security_version;
    uint32_t min_security_version;

    uint64_t virtual_cycles;
    uint64_t allocation_id_lo;
    uint64_t allocation_id_hi;
    uint64_t custody_epoch;
    uint64_t preparation_receipt_lo;
    uint64_t preparation_receipt_hi;
    uint64_t return_receipt_lo;
    uint64_t return_receipt_hi;
    uint64_t descriptor_fingerprint;
    uint64_t resource_digest_lo;
    uint64_t resource_digest_hi;
    uint64_t resource_control_words;
    uint64_t resource_preparation_ops;
    uint64_t resource_certification_ops;
    uint64_t resource_custody_transitions;
    uint64_t resource_discarded_trials;
    uint64_t resource_controller_ops;
    uint64_t session_epoch;
    uint64_t attestation_nonce;
    uint64_t attestation_age_ticks;
    uint64_t max_attestation_age_ticks;
    uint64_t device_id_digest_lo;
    uint64_t device_id_digest_hi;
    uint64_t measurement_digest_lo;
    uint64_t measurement_digest_hi;
    uint64_t custody_digest_lo;
    uint64_t custody_digest_hi;
    uint64_t resource_provenance_digest_lo;
    uint64_t resource_provenance_digest_hi;
    uint64_t gate_dispatches;
    uint64_t gate_terminal_failures;
    uint64_t gate_acks;
    uint64_t gate_replay_rejects;
    uint64_t gate_downgrade_rejects;
    uint64_t gate_fixture_rejects;
    uint64_t gate_virtual_tick;
    uint64_t lease_expiry_tick;

    uint32_t descriptor[PHASE_V13_DESCRIPTOR_WORDS];
    uint64_t boundary[PHASE_V13_BOUNDARY_WORDS];
};

static const uint32_t phase_v13_public_descriptor[PHASE_V13_DESCRIPTOR_WORDS] = {
    UINT32_C(0x50313347), UINT32_C(0x00030008),
    UINT32_C(0x00000006), UINT32_C(0x00000002),
    UINT32_C(0x00000001), UINT32_C(0x00000003),
    UINT32_C(0x00000000), UINT32_C(0x00000000),
};

static uint64_t phase_v13_allocation_serial = UINT64_C(1);
static uint64_t phase_v13_receipt_serial = UINT64_C(1);

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

static bool counter_room(uint64_t value)
{
    return value < UINT64_MAX;
}

static uint64_t descriptor_fingerprint(const PhaseQemuV13State *s)
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

static bool descriptor_valid(const PhaseQemuV13State *s)
{
    unsigned index;

    if (s->descriptor_length != PHASE_V13_DESCRIPTOR_WORDS ||
        s->boundary_schema != 3) {
        return false;
    }
    for (index = 0; index < PHASE_V13_DESCRIPTOR_WORDS; index++) {
        if (s->descriptor[index] != phase_v13_public_descriptor[index]) {
            return false;
        }
    }
    return true;
}

static void clear_descriptor(PhaseQemuV13State *s)
{
    memset(s->descriptor, 0, sizeof(s->descriptor));
    s->descriptor_index = 0;
    s->descriptor_length = 0;
    s->boundary_schema = 0;
    s->descriptor_fingerprint = 0;
    s->descriptor_sealed = false;
}

static void clear_boundary(PhaseQemuV13State *s)
{
    memset(s->boundary, 0, sizeof(s->boundary));
    s->response_ready = false;
    s->response_acked = false;
    s->receipt_held = false;
    s->terminal_failure_receipt = false;
    s->return_receipt_lo = 0;
    s->return_receipt_hi = 0;
    s->resource_digest_lo = 0;
    s->resource_digest_hi = 0;
}

static void clear_typed_evidence(PhaseQemuV13State *s)
{
    s->evidence_origin = EVIDENCE_ORIGIN_NONE;
    s->channel_security = CHANNEL_SECURITY_NONE;
    s->device_appraisal = DEVICE_APPRAISAL_NONE;
    s->measurement_class = MEASUREMENT_CLASS_NONE;
    s->custody_provenance = CUSTODY_PROVENANCE_NONE;
    s->resource_provenance = RESOURCE_PROVENANCE_NONE;
    s->trust_domain = TRUST_DOMAIN_NONE;
    s->rejection_reason = ERR_NONE;
    s->gate_flags = GATE_FLAG_NO_PHYSICAL_OUTPUT |
                    GATE_FLAG_NO_CAMPAIGN_CERTIFICATE;
    s->security_version = 0;
    s->session_epoch = 0;
    s->attestation_nonce = 0;
    s->attestation_age_ticks = 0;
    s->device_id_digest_lo = 0;
    s->device_id_digest_hi = 0;
    s->measurement_digest_lo = 0;
    s->measurement_digest_hi = 0;
    s->custody_digest_lo = 0;
    s->custody_digest_hi = 0;
    s->resource_provenance_digest_lo = 0;
    s->resource_provenance_digest_hi = 0;
}

static void release_lease(PhaseQemuV13State *s)
{
    s->leased = false;
    s->owner_tag = 0;
    s->program_tag = 0;
    s->lease_expiry_tick = 0;
}

static void sanitize_ephemeral(PhaseQemuV13State *s)
{
    clear_descriptor(s);
    clear_boundary(s);
    clear_typed_evidence(s);
    release_lease(s);
    s->prepared = false;
    s->source_isolated = false;
    s->attestation_armed = false;
    s->dispatched_lineage = false;
    s->allocation_id_lo = 0;
    s->allocation_id_hi = 0;
    s->custody_epoch = 0;
    s->preparation_receipt_lo = 0;
    s->preparation_receipt_hi = 0;
    s->request_owner = 0;
    s->request_program = 0;
    s->request_generation = 0;
}

static uint32_t active_request_error(const PhaseQemuV13State *s)
{
    if (!s->leased || s->request_owner != s->owner_tag ||
        s->request_program != s->program_tag) {
        return ERR_TAG_MISMATCH;
    }
    if (s->request_generation != s->generation) {
        return ERR_GENERATION_MISMATCH;
    }
    if (s->gate_virtual_tick >= s->lease_expiry_tick) {
        return ERR_LEASE_EXPIRED;
    }
    return ERR_NONE;
}

static void fixture_apply_common_metadata(PhaseQemuV13State *s)
{
    uint64_t vector = s->standard_vector_id;

    s->evidence_origin = EVIDENCE_ORIGIN_OFFLINE_STANDARD_VECTOR;
    s->channel_security = CHANNEL_SECURITY_OFFLINE_VECTOR_INTEGRITY;
    s->measurement_class = MEASUREMENT_CLASS_PROTOCOL_CONFORMANCE_ONLY;
    s->custody_provenance = CUSTODY_PROVENANCE_OFFLINE_VECTOR;
    s->resource_provenance = RESOURCE_PROVENANCE_OFFLINE_VECTOR;
    s->trust_domain = TRUST_DOMAIN_TEST_FIXTURE;
    s->security_version = 3;
    s->session_epoch = UINT64_C(0x5354445600000000) | vector;
    s->attestation_nonce = receipt_hash(UINT64_C(0x4d323731), vector,
                                        s->descriptor_fingerprint,
                                        s->generation);
    s->attestation_age_ticks = 1;
    s->device_id_digest_lo = receipt_hash(UINT64_C(0x4445564c), vector, 1, 0);
    s->device_id_digest_hi = receipt_hash(UINT64_C(0x44455648), vector, 1, 0);
    s->measurement_digest_lo = receipt_hash(UINT64_C(0x4d45414c), vector, 2, 0);
    s->measurement_digest_hi = receipt_hash(UINT64_C(0x4d454148), vector, 2, 0);
    s->custody_digest_lo = receipt_hash(UINT64_C(0x4355534c), vector, 3, 0);
    s->custody_digest_hi = receipt_hash(UINT64_C(0x43555348), vector, 3, 0);
    s->resource_provenance_digest_lo =
        receipt_hash(UINT64_C(0x5245534c), vector, 4, 0);
    s->resource_provenance_digest_hi =
        receipt_hash(UINT64_C(0x52455348), vector, 4, 0);
    s->gate_flags = GATE_FLAG_CHANNEL_METADATA_ACCEPTED |
                    GATE_FLAG_FIXTURE_ONLY |
                    GATE_FLAG_NO_PHYSICAL_OUTPUT |
                    GATE_FLAG_NO_CAMPAIGN_CERTIFICATE;
}

static bool hardware_absent_lease_preflight(PhaseQemuV13State *s)
{
    s->gate_state = GATE_PREFLIGHT_REJECTED;
    s->rejection_reason = ERR_HARDWARE_ABSENT;
    s->error = ERR_HARDWARE_ABSENT;
    return false;
}

static uint32_t hardware_absent_dispatch(PhaseQemuV13State *s)
{
    /* Defensive only: lease preflight makes this path unreachable. */
    s->rejection_reason = ERR_HARDWARE_ABSENT;
    return ERR_HARDWARE_ABSENT;
}

static bool offline_fixture_lease_preflight(PhaseQemuV13State *s)
{
    if (!s->test_fixture_enabled ||
        s->standard_vector_id > STANDARD_VECTOR_MAX) {
        s->gate_state = GATE_PREFLIGHT_REJECTED;
        s->rejection_reason = ERR_BACKEND_UNAVAILABLE;
        s->error = ERR_BACKEND_UNAVAILABLE;
        return false;
    }
    return true;
}

static uint32_t fixture_dispatch(PhaseQemuV13State *s)
{
    fixture_apply_common_metadata(s);
    switch (s->standard_vector_id) {
    case STANDARD_VECTOR_ACCEPTED_METADATA:
        s->device_appraisal = DEVICE_APPRAISAL_FIXTURE_STANDARD_ACCEPTED;
        s->gate_flags |= GATE_FLAG_APPRAISAL_METADATA_ACCEPTED |
                         GATE_FLAG_ENROLLMENT_PRESENT |
                         GATE_FLAG_ATTESTATION_PRESENT |
                         GATE_FLAG_FRESH |
                         GATE_FLAG_NONREPLAY |
                         GATE_FLAG_SECURITY_VERSION_ACCEPTED;
        return ERR_FIXTURE_TRUST_DOMAIN;
    case STANDARD_VECTOR_UNENROLLED:
        s->device_appraisal = DEVICE_APPRAISAL_UNENROLLED;
        return ERR_DEVICE_UNENROLLED;
    case STANDARD_VECTOR_UNATTESTED:
        s->device_appraisal = DEVICE_APPRAISAL_UNATTESTED;
        s->gate_flags |= GATE_FLAG_ENROLLMENT_PRESENT;
        return ERR_DEVICE_UNATTESTED;
    case STANDARD_VECTOR_STALE:
        s->device_appraisal = DEVICE_APPRAISAL_STALE;
        s->attestation_age_ticks = s->max_attestation_age_ticks + 1;
        s->gate_flags |= GATE_FLAG_ENROLLMENT_PRESENT |
                         GATE_FLAG_ATTESTATION_PRESENT;
        return ERR_ATTESTATION_STALE;
    case STANDARD_VECTOR_REPLAYED:
        s->device_appraisal = DEVICE_APPRAISAL_REPLAYED;
        s->gate_flags |= GATE_FLAG_ENROLLMENT_PRESENT |
                         GATE_FLAG_ATTESTATION_PRESENT |
                         GATE_FLAG_FRESH;
        return ERR_ATTESTATION_REPLAYED;
    case STANDARD_VECTOR_DOWNGRADED:
        s->device_appraisal = DEVICE_APPRAISAL_DOWNGRADED;
        s->security_version = s->min_security_version - 1;
        s->gate_flags |= GATE_FLAG_ENROLLMENT_PRESENT |
                         GATE_FLAG_ATTESTATION_PRESENT |
                         GATE_FLAG_FRESH |
                         GATE_FLAG_NONREPLAY;
        return ERR_SECURITY_DOWNGRADE;
    case STANDARD_VECTOR_TYPE_MISMATCH:
    default:
        s->device_appraisal = DEVICE_APPRAISAL_NONE;
        s->channel_security = CHANNEL_SECURITY_REJECTED;
        return ERR_EVIDENCE_TYPE_MISMATCH;
    }
}

static void backend_cancel(PhaseQemuV13State *s)
{
    if (s->gate_state == GATE_DISPATCHING) {
        s->cancel_observed = true;
    }
}

static void backend_sanitize(PhaseQemuV13State *s)
{
    clear_typed_evidence(s);
}

static const PhaseV13BackendOps phase_v13_hardware_absent_ops = {
    .id = PHASE_V13_BACKEND_HARDWARE,
    .name = "hardware-absent",
    .lease_preflight = hardware_absent_lease_preflight,
    .dispatch = hardware_absent_dispatch,
    .cancel = backend_cancel,
    .sanitize = backend_sanitize,
};

static const PhaseV13BackendOps phase_v13_offline_fixture_ops = {
    .id = PHASE_V13_BACKEND_OFFLINE_FIXTURE,
    .name = "offline-standard-vector",
    .lease_preflight = offline_fixture_lease_preflight,
    .dispatch = fixture_dispatch,
    .cancel = backend_cancel,
    .sanitize = backend_sanitize,
};

static const PhaseV13BackendOps *backend_ops_for_id(uint32_t id)
{
    switch (id) {
    case PHASE_V13_BACKEND_HARDWARE:
        return &phase_v13_hardware_absent_ops;
    case PHASE_V13_BACKEND_OFFLINE_FIXTURE:
        return &phase_v13_offline_fixture_ops;
    default:
        return NULL;
    }
}

static uint64_t resource_digest(const PhaseQemuV13State *s, uint64_t domain)
{
    uint64_t hash = UINT64_C(14695981039346656037);

    hash = fnv_u64(hash, domain);
    hash = fnv_u64(hash, PHASE_V13_RESOURCE_SCHEMA);
    hash = fnv_u64(hash, s->resource_control_words);
    hash = fnv_u64(hash, s->resource_preparation_ops);
    hash = fnv_u64(hash, s->resource_certification_ops);
    hash = fnv_u64(hash, s->resource_custody_transitions);
    hash = fnv_u64(hash, s->resource_discarded_trials);
    hash = fnv_u64(hash, s->resource_controller_ops);
    hash = fnv_u64(hash, s->resource_provenance);
    hash = fnv_u64(hash, s->resource_provenance_digest_lo);
    return fnv_u64(hash, s->resource_provenance_digest_hi);
}

static uint32_t phase_status(const PhaseQemuV13State *s)
{
    uint32_t status = ST_PORT_CLEAR | ST_EXTERNAL_REUSE_FORBIDDEN |
                      ST_HARDWARE_ABSENT;

    if (!s->snapshot_lineage && s->lifecycle == LIFE_EMPTY) {
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
    if (s->attestation_armed) {
        status |= ST_PRIVATE_ARMED | ST_PRIVATE_READY;
    }
    if (s->response_ready) {
        status |= ST_RESPONSE_READY;
    }
    if (s->response_acked) {
        status |= ST_RESPONSE_ACKED;
    }
    if (s->spent) {
        status |= ST_SPENT;
    }
    if (s->snapshot_lineage) {
        status |= ST_SNAPSHOT_LINEAGE | ST_SHAM;
    }
    if (s->backend_id == PHASE_V13_BACKEND_HARDWARE) {
        status |= ST_EXTERNAL_BACKEND;
    }
    if (s->trust_domain == TRUST_DOMAIN_TEST_FIXTURE) {
        status |= ST_FIXTURE_DOMAIN;
    }
    if (s->terminal_failure_receipt) {
        status |= ST_TERMINAL_FAILURE_RECEIPT;
    }
    return status;
}

static uint32_t phase_capabilities(void)
{
    return CAP_COMMON_V12_REGISTER_PREFIX |
           CAP_SWAPPABLE_BACKEND |
           CAP_HARDWARE_ABSENCE_GATE |
           CAP_TYPED_EVIDENCE |
           CAP_ENROLLMENT_GATE |
           CAP_ATTESTATION_AGE_GATE |
           CAP_REPLAY_GATE |
           CAP_DOWNGRADE_GATE |
           CAP_FIXTURE_DOMAIN_SEPARATION |
           CAP_FAILURE_RECEIPT_ACK_SPENT |
           CAP_MIGRATION_SHAM |
           CAP_NO_PHYSICAL_OUTPUT |
           CAP_NO_STATISTICAL_CERTIFICATE |
           CAP_OFFLINE_STANDARD_VECTORS |
           CAP_PRODUCTION_FAIL_CLOSED;
}

static uint64_t pack_evidence_types(const PhaseQemuV13State *s)
{
    return (uint64_t)s->evidence_origin |
           ((uint64_t)s->channel_security << 8) |
           ((uint64_t)s->device_appraisal << 16) |
           ((uint64_t)s->measurement_class << 24) |
           ((uint64_t)s->custody_provenance << 32) |
           ((uint64_t)s->resource_provenance << 40) |
           ((uint64_t)s->trust_domain << 48);
}

static bool seal_terminal_failure(PhaseQemuV13State *s, uint32_t reason)
{
    uint64_t receipt_id;
    uint64_t types;

    if (reason == ERR_NONE ||
        !counter_room(s->gate_terminal_failures) ||
        !counter_room(s->resource_certification_ops) ||
        !counter_room(s->resource_discarded_trials) ||
        phase_v13_receipt_serial == UINT64_MAX) {
        s->error = ERR_OVERFLOW;
        return false;
    }

    /* Fixture origin can never be promoted into the production trust tuple. */
    if (s->backend_id == PHASE_V13_BACKEND_OFFLINE_FIXTURE &&
        (s->evidence_origin != EVIDENCE_ORIGIN_OFFLINE_STANDARD_VECTOR ||
         s->trust_domain != TRUST_DOMAIN_TEST_FIXTURE ||
         s->measurement_class != MEASUREMENT_CLASS_PROTOCOL_CONFORMANCE_ONLY ||
         s->custody_provenance != CUSTODY_PROVENANCE_OFFLINE_VECTOR ||
         s->resource_provenance != RESOURCE_PROVENANCE_OFFLINE_VECTOR)) {
        reason = ERR_EVIDENCE_TYPE_MISMATCH;
        clear_typed_evidence(s);
        fixture_apply_common_metadata(s);
        s->device_appraisal = DEVICE_APPRAISAL_NONE;
        s->channel_security = CHANNEL_SECURITY_REJECTED;
    }

    receipt_id = phase_v13_receipt_serial++;
    s->rejection_reason = reason;
    s->return_class = RETURN_FAILED;
    s->return_receipt_lo = receipt_hash(UINT64_C(0x4641494c), receipt_id,
                                        s->allocation_id_lo, reason);
    s->return_receipt_hi = receipt_hash(UINT64_C(0x47415445), receipt_id,
                                        s->descriptor_fingerprint,
                                        pack_evidence_types(s));

    /* Account the terminal attempt before sealing its resource digest. */
    s->gate_terminal_failures++;
    s->resource_certification_ops++;
    s->resource_discarded_trials++;
    if (reason == ERR_ATTESTATION_REPLAYED) {
        s->gate_replay_rejects++;
    }
    if (reason == ERR_SECURITY_DOWNGRADE) {
        s->gate_downgrade_rejects++;
    }
    if (s->trust_domain == TRUST_DOMAIN_TEST_FIXTURE) {
        s->gate_fixture_rejects++;
    }
    s->resource_digest_lo = resource_digest(s, UINT64_C(0x52534c4f));
    s->resource_digest_hi = resource_digest(s, UINT64_C(0x52534849));
    types = pack_evidence_types(s);

    s->boundary[0] = PHASE_V13_BOUNDARY_HEADER;
    s->boundary[1] = receipt_id;
    s->boundary[2] = ((uint64_t)s->generation << 32) | reason;
    s->boundary[3] = types;
    s->boundary[4] = s->gate_flags;
    s->boundary[5] = s->device_id_digest_lo;
    s->boundary[6] = s->device_id_digest_hi;
    s->boundary[7] = s->measurement_digest_lo;
    s->boundary[8] = s->measurement_digest_hi;
    s->boundary[9] = s->custody_digest_lo;
    s->boundary[10] = s->custody_digest_hi;
    s->boundary[11] = s->resource_provenance_digest_lo;
    s->boundary[12] = s->resource_provenance_digest_hi;
    s->boundary[13] = s->resource_digest_lo;
    s->boundary[14] = s->resource_digest_hi;
    s->boundary[15] = receipt_hash(UINT64_C(0x434f4d4d),
                                   s->return_receipt_lo,
                                   s->return_receipt_hi,
                                   s->resource_digest_lo ^
                                   s->resource_digest_hi);

    s->terminal_failure_receipt = true;
    s->response_ready = true;
    s->response_acked = false;
    s->receipt_held = true;
    s->gate_state = GATE_TERMINAL_FAILED;
    s->lifecycle = LIFE_RESPONSE_READY;
    s->error = reason;
    return true;
}

static void execute_gate(PhaseQemuV13State *s)
{
    uint32_t request_error = active_request_error(s);
    uint32_t result;

    if (request_error != ERR_NONE) {
        s->error = request_error;
        return;
    }
    if (!s->prepared || !s->source_isolated || !s->descriptor_sealed ||
        !s->attestation_armed || s->lifecycle != LIFE_PRIVATE_READY ||
        !s->ops || !s->ops->dispatch || s->response_ready || s->spent) {
        s->error = ERR_BAD_STATE;
        return;
    }
    if (!counter_room(s->gate_dispatches) ||
        !counter_room(s->resource_control_words) ||
        !counter_room(s->resource_controller_ops) ||
        !counter_room(s->gate_terminal_failures) ||
        !counter_room(s->resource_certification_ops) ||
        !counter_room(s->resource_discarded_trials) ||
        !counter_room(s->gate_fixture_rejects) ||
        !counter_room(s->gate_replay_rejects) ||
        !counter_room(s->gate_downgrade_rejects) ||
        phase_v13_receipt_serial == UINT64_MAX) {
        s->error = ERR_OVERFLOW;
        return;
    }

    s->gate_dispatches++;
    s->resource_control_words++;
    s->resource_controller_ops++;
    s->dispatched_lineage = true;
    s->gate_state = GATE_DISPATCHING;
    s->lifecycle = LIFE_EXECUTING;
    result = s->ops->dispatch(s);
    if (result == ERR_NONE) {
        /* V13 has no success-producing physical backend. */
        result = ERR_NO_PHYSICAL_OUTPUT;
    }
    if (!seal_terminal_failure(s, result)) {
        if (s->ops->cancel) {
            s->ops->cancel(s);
        }
        s->spent = true;
        s->gate_state = GATE_SHAM;
        s->lifecycle = LIFE_SHAM;
        s->snapshot_lineage = true;
    }
}

static void phase_command(PhaseQemuV13State *s, uint32_t command)
{
    uint32_t request_error;

    if (s->snapshot_lineage || s->lifecycle == LIFE_SHAM) {
        s->error = ERR_SNAPSHOT_LINEAGE;
        return;
    }
    s->error = ERR_NONE;
    switch (command) {
    case CMD_LEASE:
        if (s->lifecycle != LIFE_EMPTY || s->leased || s->spent ||
            s->arg0 == 0 || s->arg1 == 0 || !s->ops ||
            !s->ops->lease_preflight) {
            s->error = ERR_BAD_STATE;
            return;
        }
        clear_typed_evidence(s);
        if (!s->ops->lease_preflight(s)) {
            if (s->error == ERR_NONE) {
                s->error = ERR_BACKEND_UNAVAILABLE;
            }
            return;
        }
        if (phase_v13_allocation_serial == UINT64_MAX ||
            s->gate_virtual_tick > UINT64_MAX - PHASE_V13_LEASE_TICKS) {
            s->error = ERR_OVERFLOW;
            return;
        }
        s->allocation_id_lo = phase_v13_allocation_serial++;
        s->allocation_id_hi = receipt_hash(UINT64_C(0x414c4c4f),
                                            s->allocation_id_lo,
                                            s->backend_id,
                                            s->standard_vector_id);
        s->owner_tag = s->arg0;
        s->program_tag = s->arg1;
        s->leased = true;
        s->lease_expiry_tick = s->gate_virtual_tick + PHASE_V13_LEASE_TICKS;
        s->gate_state = GATE_LEASED;
        s->lifecycle = LIFE_LEASED;
        break;
    case CMD_PREPARE:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (s->lifecycle != LIFE_LEASED || s->prepared) {
            s->error = ERR_BAD_STATE;
            return;
        }
        if (!counter_room(s->resource_preparation_ops) ||
            !counter_room(s->resource_custody_transitions)) {
            s->error = ERR_OVERFLOW;
            return;
        }
        s->prepared = true;
        s->resource_preparation_ops++;
        s->resource_custody_transitions++;
        s->preparation_receipt_lo =
            receipt_hash(UINT64_C(0x50524550), s->allocation_id_lo,
                         s->allocation_id_hi, s->generation);
        s->preparation_receipt_hi =
            receipt_hash(UINT64_C(0x4c454153), s->backend_id,
                         s->lease_expiry_tick, s->standard_vector_id);
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
        if (!counter_room(s->custody_epoch) ||
            !counter_room(s->resource_custody_transitions)) {
            s->error = ERR_OVERFLOW;
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
        if (!s->source_isolated || s->descriptor_sealed ||
            s->lifecycle != LIFE_ISOLATED || !descriptor_valid(s)) {
            s->error = ERR_DESCRIPTOR_INVALID;
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
        if (!s->descriptor_sealed || s->attestation_armed ||
            s->lifecycle != LIFE_SEALED) {
            s->error = ERR_BAD_STATE;
            return;
        }
        /* No secret is injected.  This arms only the fixed fixture selector. */
        s->attestation_armed = true;
        s->gate_state = GATE_ATTESTATION_ARMED;
        s->lifecycle = LIFE_PRIVATE_READY;
        break;
    case CMD_EXECUTE_ATOMIC:
        execute_gate(s);
        break;
    case CMD_ACK_RESPONSE:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (!s->response_ready || !s->terminal_failure_receipt ||
            s->return_class != RETURN_FAILED ||
            s->lifecycle != LIFE_RESPONSE_READY) {
            s->error = ERR_RESPONSE_LOCKED;
            return;
        }
        if (!counter_room(s->gate_acks)) {
            s->error = ERR_OVERFLOW;
            return;
        }
        s->gate_acks++;
        memset(s->boundary, 0, sizeof(s->boundary));
        s->response_ready = false;
        s->response_acked = true;
        s->receipt_held = false;
        s->terminal_failure_receipt = false;
        s->spent = true;
        s->gate_state = GATE_SPENT;
        s->lifecycle = LIFE_SPENT;
        release_lease(s);
        break;
    case CMD_ABORT_PREEXEC:
        request_error = active_request_error(s);
        if (request_error != ERR_NONE) {
            s->error = request_error;
            return;
        }
        if (s->lifecycle < LIFE_SEALED ||
            s->lifecycle > LIFE_PRIVATE_READY || s->dispatched_lineage) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->attestation_armed = false;
        clear_descriptor(s);
        clear_typed_evidence(s);
        s->lifecycle = LIFE_ISOLATED;
        break;
    case CMD_BEGIN_REUSE:
        s->error = ERR_REUSE_NOT_QUALIFIED;
        break;
    case CMD_SNAPSHOT:
        s->error = ERR_SNAPSHOT_REJECTED;
        break;
    case CMD_POLL_EXTERNAL:
    case CMD_CANCEL_EXTERNAL:
        s->error = ERR_BAD_STATE;
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
    case REG_GATE_STATE:
    case REG_EVIDENCE_ORIGIN:
    case REG_CHANNEL_SECURITY:
    case REG_DEVICE_APPRAISAL:
    case REG_MEASUREMENT_CLASS:
    case REG_CUSTODY_PROVENANCE:
    case REG_RESOURCE_PROVENANCE:
    case REG_TRUST_DOMAIN:
    case REG_REJECTION_REASON:
    case REG_GATE_FLAGS:
    case REG_SECURITY_VERSION:
    case REG_MIN_SECURITY_VERSION:
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
    case REG_SESSION_EPOCH:
    case REG_ATTESTATION_NONCE:
    case REG_ATTESTATION_AGE_TICKS:
    case REG_MAX_ATTESTATION_AGE_TICKS:
    case REG_DEVICE_ID_DIGEST_LO:
    case REG_DEVICE_ID_DIGEST_HI:
    case REG_MEASUREMENT_DIGEST_LO:
    case REG_MEASUREMENT_DIGEST_HI:
    case REG_CUSTODY_DIGEST_LO:
    case REG_CUSTODY_DIGEST_HI:
    case REG_RESOURCE_PROVENANCE_DIGEST_LO:
    case REG_RESOURCE_PROVENANCE_DIGEST_HI:
    case REG_GATE_DISPATCHES:
    case REG_GATE_TERMINAL_FAILURES:
    case REG_GATE_ACKS:
    case REG_GATE_REPLAY_REJECTS:
    case REG_GATE_DOWNGRADE_REJECTS:
    case REG_GATE_FIXTURE_REJECTS:
    case REG_GATE_VIRTUAL_TICK:
    case REG_LEASE_EXPIRY_TICK:
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
    PhaseQemuV13State *s = opaque;

    if (!access_width_valid(address, size)) {
        s->error = ERR_BAD_ARGUMENT;
        return size == 4 ? UINT32_MAX : UINT64_MAX;
    }
    if (address >= REG_BOUNDARY_BASE && address <= REG_BOUNDARY_LAST) {
        unsigned index = (address - REG_BOUNDARY_BASE) / 8;

        if (!s->response_ready || s->snapshot_lineage) {
            s->error = ERR_RESPONSE_LOCKED;
            return PHASE_V13_LOCKED_BOUNDARY;
        }
        return s->boundary[index];
    }
    switch (address) {
    case REG_MAGIC: return PHASE_V13_MAGIC;
    case REG_ABI: return PHASE_V13_ABI;
    case REG_BACKEND: return s->backend_id;
    case REG_CAPABILITIES_LO: return phase_capabilities();
    case REG_STATUS_LO: return phase_status(s);
    case REG_ERROR: return s->error;
    case REG_GENERATION: return s->generation;
    case REG_EXACT_RETURN_GENERATION: return 0;
    case REG_ARG0: return s->arg0;
    case REG_ARG1: return s->arg1;
    case REG_COMMAND: return 0;
    case REG_LIFECYCLE: return s->lifecycle;
    case REG_VIRTUAL_CYCLES: return s->virtual_cycles;
    case REG_BOUNDARY_COMMIT_COOKIE:
        return s->response_ready ? s->boundary[15] : PHASE_V13_LOCKED_BOUNDARY;
    case REG_RESOURCE_STATE_CELLS: return 0;
    case REG_RESOURCE_SCRATCH_CELLS: return 0;
    case REG_RESOURCE_QUERY_APPLICATIONS: return 0;
    case REG_RESOURCE_RETURN_CHECKS: return 0;
    case REG_RESOURCE_ENVIRONMENT_OPS: return 0;
    case REG_RESOURCE_PEAK_BITS: return sizeof(*s) * 8;
    case REG_REQUEST_OWNER: return s->request_owner;
    case REG_REQUEST_PROGRAM: return s->request_program;
    case REG_REQUEST_GENERATION: return s->request_generation;
    case REG_DESCRIPTOR_INDEX: return s->descriptor_index;
    case REG_DESCRIPTOR_WORD:
        return s->descriptor_index < PHASE_V13_DESCRIPTOR_WORDS ?
               s->descriptor[s->descriptor_index] : 0;
    case REG_DESCRIPTOR_LENGTH: return s->descriptor_length;
    case REG_BOUNDARY_SCHEMA: return s->boundary_schema;
    case REG_DESCRIPTOR_FINGERPRINT: return s->descriptor_fingerprint;
    case REG_FAULT_MODE: return 0;
    case REG_CAPABILITIES_HI:
    case REG_STATUS_HI: return 0;
    case REG_PRIVATE_QUERY_SLOTS: return 0;
    case REG_PRIVATE_READY_MASK: return s->attestation_armed ? 1 : 0;
    case REG_RETURN_CLASS: return s->return_class;
    case REG_BOUNDARY_LENGTH: return PHASE_V13_BOUNDARY_WORDS * 8;
    case REG_ALLOCATION_ID_LO: return s->allocation_id_lo;
    case REG_ALLOCATION_ID_HI: return s->allocation_id_hi;
    case REG_CUSTODY_EPOCH: return s->custody_epoch;
    case REG_PREPARATION_RECEIPT_LO: return s->preparation_receipt_lo;
    case REG_PREPARATION_RECEIPT_HI: return s->preparation_receipt_hi;
    case REG_RETURN_RECEIPT_LO: return s->return_receipt_lo;
    case REG_RETURN_RECEIPT_HI: return s->return_receipt_hi;
    case REG_RESOURCE_SECRET_STORAGE_BITS: return 0;
    case REG_RESOURCE_CONTROL_WORDS: return s->resource_control_words;
    case REG_RESOURCE_PREPARATION_OPS: return s->resource_preparation_ops;
    case REG_RESOURCE_CERTIFICATION_OPS: return s->resource_certification_ops;
    case REG_RESOURCE_LOGICAL_QUERIES: return 0;
    case REG_RESOURCE_DURATION_FS: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_PORT_BANDWIDTH_HZ: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_ACTION_Q40_RAD: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_MEAN_ENERGY_ATTOJ: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_LOSS_Q63: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_DEPHASING_Q63: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_ENV_HISTORY_CELLS: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_CUSTODY_TRANSITIONS:
        return s->resource_custody_transitions;
    case REG_RESOURCE_REUSE_COUNT: return 0;
    case REG_RESOURCE_DISCARDED_TRIALS: return s->resource_discarded_trials;
    case REG_RESOURCE_OUTPUT_HOLD_FS: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_MAINTENANCE_OPS: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_PRECISION_BITS: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_SCHEMA: return PHASE_V13_RESOURCE_SCHEMA;
    case REG_RESOURCE_DIGEST_LO: return s->resource_digest_lo;
    case REG_RESOURCE_DIGEST_HI: return s->resource_digest_hi;
    case REG_RESOURCE_COMPILER_OPS: return 0;
    case REG_RESOURCE_CONTROLLER_OPS: return s->resource_controller_ops;
    case REG_RESOURCE_CONSTRUCTION_OPS: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_RESOURCE_SECRET_ENTROPY_BITS: return 0;
    case REG_RESOURCE_CARRIER_PHOTON_NUMBER: return PHASE_V13_RESOURCE_UNKNOWN;
    case REG_ADAPTER_STATE: return s->gate_state;
    case REG_ADAPTER_AUTH_ACCEPTED: return 0;
    case REG_ADAPTER_AUTH_REJECTED: return s->gate_terminal_failures;
    case REG_ADAPTER_DISPATCHES: return s->gate_dispatches;
    case REG_ADAPTER_COMPLETIONS: return s->gate_terminal_failures;
    case REG_ADAPTER_CANCELS: return s->cancel_observed ? 1 : 0;
    case REG_ADAPTER_VIRTUAL_TICK: return s->gate_virtual_tick;
    case REG_ADAPTER_DEADLINE_TICK: return s->lease_expiry_tick;
    case REG_GATE_STATE: return s->gate_state;
    case REG_EVIDENCE_ORIGIN: return s->evidence_origin;
    case REG_CHANNEL_SECURITY: return s->channel_security;
    case REG_DEVICE_APPRAISAL: return s->device_appraisal;
    case REG_MEASUREMENT_CLASS: return s->measurement_class;
    case REG_CUSTODY_PROVENANCE: return s->custody_provenance;
    case REG_RESOURCE_PROVENANCE: return s->resource_provenance;
    case REG_TRUST_DOMAIN: return s->trust_domain;
    case REG_REJECTION_REASON: return s->rejection_reason;
    case REG_GATE_FLAGS: return s->gate_flags;
    case REG_SECURITY_VERSION: return s->security_version;
    case REG_MIN_SECURITY_VERSION: return s->min_security_version;
    case REG_SESSION_EPOCH: return s->session_epoch;
    case REG_ATTESTATION_NONCE: return s->attestation_nonce;
    case REG_ATTESTATION_AGE_TICKS: return s->attestation_age_ticks;
    case REG_MAX_ATTESTATION_AGE_TICKS: return s->max_attestation_age_ticks;
    case REG_DEVICE_ID_DIGEST_LO: return s->device_id_digest_lo;
    case REG_DEVICE_ID_DIGEST_HI: return s->device_id_digest_hi;
    case REG_MEASUREMENT_DIGEST_LO: return s->measurement_digest_lo;
    case REG_MEASUREMENT_DIGEST_HI: return s->measurement_digest_hi;
    case REG_CUSTODY_DIGEST_LO: return s->custody_digest_lo;
    case REG_CUSTODY_DIGEST_HI: return s->custody_digest_hi;
    case REG_RESOURCE_PROVENANCE_DIGEST_LO:
        return s->resource_provenance_digest_lo;
    case REG_RESOURCE_PROVENANCE_DIGEST_HI:
        return s->resource_provenance_digest_hi;
    case REG_GATE_DISPATCHES: return s->gate_dispatches;
    case REG_GATE_TERMINAL_FAILURES: return s->gate_terminal_failures;
    case REG_GATE_ACKS: return s->gate_acks;
    case REG_GATE_REPLAY_REJECTS: return s->gate_replay_rejects;
    case REG_GATE_DOWNGRADE_REJECTS: return s->gate_downgrade_rejects;
    case REG_GATE_FIXTURE_REJECTS: return s->gate_fixture_rejects;
    case REG_GATE_VIRTUAL_TICK: return s->gate_virtual_tick;
    case REG_LEASE_EXPIRY_TICK: return s->lease_expiry_tick;
    default:
        s->error = ERR_BAD_ARGUMENT;
        return UINT64_MAX;
    }
}

static void phase_mmio_write(void *opaque, hwaddr address, uint64_t value,
                             unsigned size)
{
    PhaseQemuV13State *s = opaque;

    if (!access_width_valid(address, size) ||
        (address >= REG_BOUNDARY_BASE && address <= REG_BOUNDARY_LAST)) {
        s->error = ERR_BAD_ARGUMENT;
        return;
    }
    switch (address) {
    case REG_ARG0:
        s->arg0 = value;
        break;
    case REG_ARG1:
        s->arg1 = value;
        break;
    case REG_REQUEST_OWNER:
        s->request_owner = value;
        break;
    case REG_REQUEST_PROGRAM:
        s->request_program = value;
        break;
    case REG_REQUEST_GENERATION:
        s->request_generation = value;
        break;
    case REG_DESCRIPTOR_INDEX:
        if (s->descriptor_sealed || value >= PHASE_V13_DESCRIPTOR_WORDS) {
            s->error = ERR_BAD_ARGUMENT;
            return;
        }
        s->descriptor_index = value;
        break;
    case REG_DESCRIPTOR_WORD:
        if (s->descriptor_sealed ||
            s->descriptor_index >= PHASE_V13_DESCRIPTOR_WORDS) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->descriptor[s->descriptor_index] = value;
        break;
    case REG_DESCRIPTOR_LENGTH:
        if (s->descriptor_sealed || value > PHASE_V13_DESCRIPTOR_WORDS) {
            s->error = ERR_BAD_ARGUMENT;
            return;
        }
        s->descriptor_length = value;
        break;
    case REG_BOUNDARY_SCHEMA:
        if (s->descriptor_sealed) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->boundary_schema = value;
        break;
    case REG_COMMAND:
        phase_command(s, value);
        break;
    case REG_GATE_VIRTUAL_TICK:
    case REG_ADAPTER_VIRTUAL_TICK:
        if (value < s->gate_virtual_tick || s->response_ready || s->spent) {
            s->error = ERR_BAD_ARGUMENT;
            return;
        }
        s->gate_virtual_tick = value;
        break;
    default:
        s->error = ERR_BAD_ARGUMENT;
        break;
    }
}

static const MemoryRegionOps phase_mmio_ops = {
    .read = phase_mmio_read,
    .write = phase_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        /* Deliver all PCI access shapes to the callback for error latching. */
        .min_access_size = 1,
        .max_access_size = 8,
        .unaligned = true,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
        .unaligned = true,
    },
};

static void enter_migration_sham(PhaseQemuV13State *s)
{
    if (s->gate_state == GATE_DISPATCHING && s->ops && s->ops->cancel) {
        s->ops->cancel(s);
    }
    if (s->ops && s->ops->sanitize) {
        s->ops->sanitize(s);
    }
    sanitize_ephemeral(s);
    s->snapshot_lineage = true;
    s->spent = true;
    s->return_class = RETURN_FAILED;
    s->error = ERR_SNAPSHOT_LINEAGE;
    s->gate_state = GATE_SHAM;
    s->lifecycle = LIFE_SHAM;
}

static void phase_qemu_v13_reset(DeviceState *device)
{
    PhaseQemuV13State *s = PHASE_QEMU_V13(device);

    if (s->snapshot_lineage || s->lifecycle == LIFE_SHAM || s->leased ||
        s->prepared || s->source_isolated || s->descriptor_sealed ||
        s->attestation_armed || s->dispatched_lineage || s->response_ready ||
        s->spent) {
        enter_migration_sham(s);
        return;
    }

    sanitize_ephemeral(s);
    s->snapshot_lineage = false;
    s->spent = false;
    s->return_class = RETURN_NONE;
    s->error = ERR_NONE;
    s->lifecycle = LIFE_EMPTY;
    s->gate_state = GATE_COLD;
    s->generation = 1;
    s->arg0 = 0;
    s->arg1 = 0;
    s->migration_marker = PHASE_V13_MIGRATION_MARKER;
    s->virtual_cycles = 0;
    s->resource_control_words = 0;
    s->resource_preparation_ops = 0;
    s->resource_certification_ops = 0;
    s->resource_custody_transitions = 0;
    s->resource_discarded_trials = 0;
    s->resource_controller_ops = 0;
    s->gate_dispatches = 0;
    s->gate_terminal_failures = 0;
    s->gate_acks = 0;
    s->gate_replay_rejects = 0;
    s->gate_downgrade_rejects = 0;
    s->gate_fixture_rejects = 0;
    s->gate_virtual_tick = 0;
    s->min_security_version = 3;
    s->max_attestation_age_ticks = 8;
    s->cancel_observed = false;
}

static int phase_qemu_v13_pre_save(void *opaque)
{
    PhaseQemuV13State *s = opaque;

    /* Source migration burns and sanitizes before serialization. */
    enter_migration_sham(s);
    return 0;
}

static int phase_qemu_v13_post_load(void *opaque, int version_id)
{
    PhaseQemuV13State *s = opaque;

    (void)version_id;
    if (s->migration_marker != PHASE_V13_MIGRATION_MARKER ||
        s->backend_id != s->configured_backend_id ||
        s->test_fixture_enabled != s->configured_test_fixture_enabled ||
        s->standard_vector_id != s->configured_standard_vector_id ||
        s->generation == 0) {
        return -EINVAL;
    }
    enter_migration_sham(s);
    return 0;
}

static const VMStateDescription vmstate_phase_qemu_v13 = {
    .name = "phase-qemu-v13",
    .version_id = 1,
    .minimum_version_id = 1,
    .pre_save = phase_qemu_v13_pre_save,
    .post_load = phase_qemu_v13_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, PhaseQemuV13State),
        VMSTATE_UINT32(migration_marker, PhaseQemuV13State),
        VMSTATE_UINT32(backend_id, PhaseQemuV13State),
        VMSTATE_BOOL(test_fixture_enabled, PhaseQemuV13State),
        VMSTATE_UINT32(standard_vector_id, PhaseQemuV13State),
        VMSTATE_UINT32(generation, PhaseQemuV13State),
        VMSTATE_END_OF_LIST()
    }
};

static void phase_qemu_v13_realize(PCIDevice *pci_device, Error **errp)
{
    PhaseQemuV13State *s = PHASE_QEMU_V13(pci_device);

    s->ops = backend_ops_for_id(s->backend_id);
    if (!s->ops) {
        error_setg(errp,
                   "phase-qemu-v13 backend-id must be 0x0d80 or 0x0df0");
        return;
    }
    if (s->standard_vector_id > STANDARD_VECTOR_MAX) {
        error_setg(errp,
                   "phase-qemu-v13 test-standard-vector-id must be in [0,6]");
        return;
    }
    if (s->backend_id == PHASE_V13_BACKEND_OFFLINE_FIXTURE &&
        !s->test_fixture_enabled) {
        error_setg(errp,
                   "offline fixture backend requires test-fixture-enabled");
        return;
    }
    if (s->test_fixture_enabled &&
        s->backend_id != PHASE_V13_BACKEND_OFFLINE_FIXTURE) {
        error_setg(errp,
                   "test-fixture-enabled is confined to backend-id=0x0df0");
        return;
    }

    s->configured_backend_id = s->backend_id;
    s->configured_test_fixture_enabled = s->test_fixture_enabled;
    s->configured_standard_vector_id = s->standard_vector_id;
    memory_region_init_io(&s->mmio, OBJECT(s), &phase_mmio_ops, s,
                          "phase-qemu-v13-mmio", PHASE_V13_BAR_SIZE);
    pci_register_bar(pci_device, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
    s->realized = true;
}

static void phase_qemu_v13_unrealize(DeviceState *device)
{
    PhaseQemuV13State *s = PHASE_QEMU_V13(device);

    enter_migration_sham(s);
    s->realized = false;
}

static void phase_qemu_v13_instance_init(Object *object)
{
    PhaseQemuV13State *s = PHASE_QEMU_V13(object);

    s->backend_id = PHASE_V13_BACKEND_HARDWARE;
    s->test_fixture_enabled = false;
    s->standard_vector_id = STANDARD_VECTOR_ACCEPTED_METADATA;
    s->generation = 1;
    s->migration_marker = PHASE_V13_MIGRATION_MARKER;
    s->min_security_version = 3;
    s->max_attestation_age_ticks = 8;
    s->lifecycle = LIFE_EMPTY;
    s->gate_state = GATE_COLD;
    clear_typed_evidence(s);
}

static const Property phase_qemu_v13_properties[] = {
    DEFINE_PROP_UINT32("backend-id", PhaseQemuV13State, backend_id,
                       PHASE_V13_BACKEND_HARDWARE),
    DEFINE_PROP_BOOL("test-fixture-enabled", PhaseQemuV13State,
                     test_fixture_enabled, false),
    DEFINE_PROP_UINT32("test-standard-vector-id", PhaseQemuV13State,
                       standard_vector_id, STANDARD_VECTOR_ACCEPTED_METADATA),
};

static void phase_qemu_v13_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *device_class = DEVICE_CLASS(klass);
    PCIDeviceClass *pci_class = PCI_DEVICE_CLASS(klass);

    (void)data;
    pci_class->realize = phase_qemu_v13_realize;
    pci_class->vendor_id = PHASE_V13_VENDOR_ID;
    pci_class->device_id = PHASE_V13_DEVICE_ID;
    pci_class->revision = PHASE_V13_REVISION;
    pci_class->class_id = PCI_CLASS_OTHERS;
    device_class->vmsd = &vmstate_phase_qemu_v13;
    device_class->unrealize = phase_qemu_v13_unrealize;
    device_class_set_legacy_reset(device_class, phase_qemu_v13_reset);
    device_class_set_props(device_class, phase_qemu_v13_properties);
    set_bit(DEVICE_CATEGORY_MISC, device_class->categories);
}

static const TypeInfo phase_qemu_v13_info = {
    .name = TYPE_PHASE_QEMU_V13,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(PhaseQemuV13State),
    .instance_init = phase_qemu_v13_instance_init,
    .class_init = phase_qemu_v13_class_init,
    .interfaces = (const InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void phase_qemu_v13_register_types(void)
{
    type_register_static(&phase_qemu_v13_info);
}

type_init(phase_qemu_v13_register_types)
