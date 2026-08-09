/*
 * Phase-QEMU V0 PCI device and REFERENCE_HARDWARE_MODEL_0 backend.
 *
 * This is a deterministic virtual-hardware calibration model of the frozen P0
 * source-separated quartz carrier architecture.  It is not a physical result.
 * The MMIO boundary intentionally exposes no carrier, source, detector, bath,
 * or inverse-history coordinates.  Only a released final diagnostic boundary
 * and non-secret custody/resource receipts are readable.
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

#define TYPE_PHASE_QEMU_V0 "phase-qemu-v0"
OBJECT_DECLARE_SIMPLE_TYPE(PhaseQemuV0State, PHASE_QEMU_V0)

#define PHASE_QEMU_VENDOR_ID PCI_VENDOR_ID_QEMU
#define PHASE_QEMU_DEVICE_ID 0x11f0

#define PHASE_QEMU_MAGIC 0x50485630u /* PHV0 */
#define PHASE_QEMU_ABI   0x00010000u
#define PHASE_QEMU_BACKEND_P0 0u

#define Q30_ONE INT64_C(1073741824)
#define P0_DECAY_Q30 INT64_C(1063004406) /* 0.99 per 64-cycle quantum */
#define P0_PREPARE_STEPS 256u
#define P0_CYCLES_PER_STEP 64u
#define P0_MIN_RINGDOWN_STEPS 8u
#define P0_MAX_EVOLVE_STEPS 4096u

enum PhaseQemuRegister {
    REG_MAGIC = 0x000,
    REG_ABI = 0x004,
    REG_BACKEND = 0x008,
    REG_CAPABILITIES = 0x00c,
    REG_STATUS = 0x010,
    REG_ERROR = 0x014,
    REG_GENERATION = 0x018,
    REG_REINITIALIZATIONS = 0x01c,
    REG_ARG0 = 0x020,
    REG_ARG1 = 0x024,
    REG_COMMAND = 0x028,
    REG_BARRIER_RECEIPT = 0x02c,
    REG_VIRTUAL_CYCLES = 0x030,
    REG_BOUNDARY_I = 0x040,
    REG_BOUNDARY_Q = 0x048,
    REG_BOUNDARY_ENERGY = 0x050,
    REG_RESOURCE_CELLS = 0x058,
    REG_RETAINED_HISTORY_CELLS = 0x05c,
    REG_REQUEST_OWNER = 0x060,
    REG_REQUEST_PROGRAM = 0x064,
};

enum PhaseQemuCommand {
    CMD_LEASE = 1,
    CMD_PREPARE = 2,
    CMD_ISOLATE_SOURCE = 3,
    CMD_EVOLVE = 4,
    CMD_PROJECT_BOUNDARY = 5,
    CMD_INVERT = 6,
    CMD_RESTORE = 7,
    CMD_REUSE = 8,
    CMD_RELEASE_DIAGNOSTIC = 9,
    CMD_CANONICAL_MODEL_REINITIALIZE = 10,
};

enum PhaseQemuError {
    ERR_NONE = 0,
    ERR_BAD_STATE = 1,
    ERR_BAD_ARGUMENT = 2,
    ERR_PREMATURE_PROJECTION = 3,
    ERR_P0_INVERSE_UNAVAILABLE = 4,
    ERR_P0_RESTORATION_UNAVAILABLE = 5,
    ERR_RESPONSE_LOCKED = 6,
    ERR_REUSE_REQUIRES_RESTORATION = 7,
    ERR_CUSTODY_MISMATCH = 8,
};

enum PhaseQemuStatus {
    ST_CANONICAL = 1u << 0,
    ST_LEASED = 1u << 1,
    ST_PREPARED = 1u << 2,
    ST_SOURCE_ISOLATED = 1u << 3,
    ST_BOUNDARY_PENDING = 1u << 4,
    ST_RESPONSE_READY = 1u << 5,
    ST_SPENT = 1u << 6,
    ST_REINITIALIZATION_USED = 1u << 7,
    ST_CARRIER_PRESENT = 1u << 8,
};

enum PhaseQemuCapability {
    CAP_OWNER_PROGRAM_CUSTODY_LEASE = 1u << 0,
    CAP_PHASE_PREPARE = 1u << 1,
    CAP_SOURCE_ISOLATION = 1u << 2,
    CAP_VIRTUAL_EVOLUTION = 1u << 3,
    CAP_FINAL_DIAGNOSTIC = 1u << 4,
    CAP_NATIVE_INVERT = 1u << 5,
    CAP_NATIVE_RESTORE = 1u << 6,
    CAP_RESTORED_REUSE = 1u << 7,
};

/*
 * The backend process-object remains partitioned even though V0 ships one
 * concrete model.  Future backends replace these operations without changing
 * the PCI/MMIO custody and response-ordering contract.
 */
typedef struct PhaseBackendOps {
    uint32_t backend_id;
    uint32_t capabilities;
    void (*prepare)(PhaseQemuV0State *s, uint32_t phase_arm);
    void (*isolate)(PhaseQemuV0State *s);
    bool (*evolve)(PhaseQemuV0State *s, uint32_t steps);
    bool (*project)(PhaseQemuV0State *s);
    bool (*invert)(PhaseQemuV0State *s);
    bool (*restore)(PhaseQemuV0State *s);
} PhaseBackendOps;

struct PhaseQemuV0State {
    PCIDevice parent_obj;
    MemoryRegion mmio;

    bool carrier_present;
    bool configured_carrier_present;
    bool leased;
    bool prepared;
    bool source_isolated;
    bool boundary_pending;
    bool response_ready;
    bool spent;

    uint32_t error;
    uint32_t owner_id;
    uint32_t program_id;
    uint32_t request_owner_id;
    uint32_t request_program_id;
    uint32_t generation;
    uint32_t reinitializations;
    uint32_t arg0;
    uint32_t arg1;
    uint32_t phase_arm;
    uint32_t barrier_receipt;
    uint32_t ringdown_steps;

    uint64_t virtual_cycles;
    uint64_t dissipated_energy_q30;

    /* Source domain: stays energized upstream after the isolation event. */
    int64_t source_i;
    int64_t source_q;

    /* Mechanical carrier domain: rotating-frame displacement quadratures. */
    int64_t carrier_i;
    int64_t carrier_q;

    /* High-impedance detector domain with its own finite response state. */
    int64_t detector_i;
    int64_t detector_q;

    /* Boundary domain: never readable until diagnostic release. */
    int64_t boundary_i;
    int64_t boundary_q;
    uint64_t boundary_energy_q30;

    const PhaseBackendOps *ops;
};

static int64_t q30_mul(int64_t a, int64_t b)
{
    return (int64_t)(((__int128)a * (__int128)b) >> 30);
}

static uint64_t q30_energy(int64_t i, int64_t q)
{
    __int128 sum = (__int128)i * i + (__int128)q * q;

    return (uint64_t)(sum >> 30);
}

static bool p0_dynamic_zero(const PhaseQemuV0State *s)
{
    return s->source_i == 0 && s->source_q == 0 &&
           s->carrier_i == 0 && s->carrier_q == 0 &&
           s->detector_i == 0 && s->detector_q == 0 &&
           s->boundary_i == 0 && s->boundary_q == 0 &&
           s->boundary_energy_q30 == 0 && s->virtual_cycles == 0 &&
           s->dissipated_energy_q30 == 0 && s->barrier_receipt == 0 &&
           s->ringdown_steps == 0;
}

static void p0_clear_transaction(PhaseQemuV0State *s,
                                 bool keep_reinitialization_marker)
{
    s->leased = false;
    s->prepared = false;
    s->source_isolated = false;
    s->boundary_pending = false;
    s->response_ready = false;
    s->spent = false;
    s->error = ERR_NONE;
    s->owner_id = 0;
    s->program_id = 0;
    s->request_owner_id = 0;
    s->request_program_id = 0;
    s->arg0 = 0;
    s->arg1 = 0;
    s->phase_arm = 0;
    s->barrier_receipt = 0;
    s->ringdown_steps = 0;
    s->virtual_cycles = 0;
    s->dissipated_energy_q30 = 0;
    s->source_i = 0;
    s->source_q = 0;
    s->carrier_i = 0;
    s->carrier_q = 0;
    s->detector_i = 0;
    s->detector_q = 0;
    s->boundary_i = 0;
    s->boundary_q = 0;
    s->boundary_energy_q30 = 0;
    if (!keep_reinitialization_marker) {
        s->reinitializations = 0;
    }
}

static void p0_prepare(PhaseQemuV0State *s, uint32_t phase_arm)
{
    uint32_t step;
    int64_t sign = phase_arm ? -1 : 1;

    s->phase_arm = phase_arm;
    s->source_i = sign * Q30_ONE;
    s->source_q = 0;

    for (step = 0; step < P0_PREPARE_STEPS; step++) {
        if (s->carrier_present) {
            s->carrier_i += (s->source_i - s->carrier_i) >> 3;
            s->carrier_q += (s->source_q - s->carrier_q) >> 3;
        }
        s->detector_i += (s->carrier_i - s->detector_i) >> 2;
        s->detector_q += (s->carrier_q - s->detector_q) >> 2;
        s->virtual_cycles += P0_CYCLES_PER_STEP;
    }
    s->prepared = true;
}

static void p0_isolate(PhaseQemuV0State *s)
{
    /*
     * Code 8 represents gate OFF, K1/K2 open, K3 guarded after the ordered
     * stability sequence.  The source remains present upstream.  The small
     * signed impulse is a carrier-side switch disturbance, not source replay.
     */
    s->barrier_receipt = 8;
    s->source_isolated = true;
    if (s->carrier_present) {
        s->carrier_q += s->source_i >> 12;
    }
}

static bool p0_evolve(PhaseQemuV0State *s, uint32_t steps)
{
    uint32_t step;

    if (steps == 0 || steps > P0_MAX_EVOLVE_STEPS ||
        s->ringdown_steps > P0_MAX_EVOLVE_STEPS - steps) {
        s->error = ERR_BAD_ARGUMENT;
        return false;
    }

    for (step = 0; step < steps; step++) {
        uint64_t before = q30_energy(s->carrier_i, s->carrier_q);
        int64_t old_i = s->carrier_i;
        int64_t old_q = s->carrier_q;
        int64_t rotated_i = old_i - (old_q >> 12);
        int64_t rotated_q = old_q + (old_i >> 12);
        int64_t target_i;
        int64_t target_q;
        uint64_t after;

        s->carrier_i = q30_mul(rotated_i, P0_DECAY_Q30);
        s->carrier_q = q30_mul(rotated_q, P0_DECAY_Q30);
        after = q30_energy(s->carrier_i, s->carrier_q);
        if (before > after) {
            s->dissipated_energy_q30 += before - after;
        }

        /* Measured source feedthrough is nonzero but strongly attenuated. */
        target_i = s->carrier_i + (s->source_i >> 20);
        target_q = s->carrier_q + (s->source_q >> 20);
        s->detector_i += (target_i - s->detector_i) >> 2;
        s->detector_q += (target_q - s->detector_q) >> 2;
        s->virtual_cycles += P0_CYCLES_PER_STEP;
        s->ringdown_steps++;
    }
    return true;
}

static bool p0_project(PhaseQemuV0State *s)
{
    if (!s->source_isolated || s->barrier_receipt != 8 ||
        s->ringdown_steps < P0_MIN_RINGDOWN_STEPS) {
        s->error = ERR_PREMATURE_PROJECTION;
        return false;
    }

    s->boundary_i = s->detector_i;
    s->boundary_q = s->detector_q;
    s->boundary_energy_q30 = q30_energy(s->detector_i, s->detector_q);
    s->boundary_pending = true;
    return true;
}

static bool p0_invert(PhaseQemuV0State *s)
{
    s->error = ERR_P0_INVERSE_UNAVAILABLE;
    return false;
}

static bool p0_restore(PhaseQemuV0State *s)
{
    s->error = ERR_P0_RESTORATION_UNAVAILABLE;
    return false;
}

static const PhaseBackendOps p0_ops = {
    .backend_id = PHASE_QEMU_BACKEND_P0,
    .capabilities = CAP_OWNER_PROGRAM_CUSTODY_LEASE | CAP_PHASE_PREPARE |
                    CAP_SOURCE_ISOLATION | CAP_VIRTUAL_EVOLUTION |
                    CAP_FINAL_DIAGNOSTIC,
    .prepare = p0_prepare,
    .isolate = p0_isolate,
    .evolve = p0_evolve,
    .project = p0_project,
    .invert = p0_invert,
    .restore = p0_restore,
};

static uint32_t phase_status(const PhaseQemuV0State *s)
{
    uint32_t status = 0;

    if (!s->leased && !s->spent && p0_dynamic_zero(s)) {
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
    if (s->boundary_pending) {
        status |= ST_BOUNDARY_PENDING;
    }
    if (s->response_ready) {
        status |= ST_RESPONSE_READY;
    }
    if (s->spent) {
        status |= ST_SPENT;
    }
    if (s->reinitializations != 0) {
        status |= ST_REINITIALIZATION_USED;
    }
    if (s->carrier_present) {
        status |= ST_CARRIER_PRESENT;
    }
    return status;
}

static void phase_command(PhaseQemuV0State *s, uint32_t command)
{
    s->error = ERR_NONE;

    if (s->leased && command != CMD_LEASE &&
        (s->request_owner_id != s->owner_id ||
         s->request_program_id != s->program_id)) {
        s->error = ERR_CUSTODY_MISMATCH;
        return;
    }

    switch (command) {
    case CMD_LEASE:
        if (s->leased || s->spent || !p0_dynamic_zero(s) ||
            s->request_owner_id == 0 || s->request_program_id == 0) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->owner_id = s->request_owner_id;
        s->program_id = s->request_program_id;
        s->leased = true;
        break;
    case CMD_PREPARE:
        if (!s->leased || s->prepared || s->source_isolated || s->arg0 > 1) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->ops->prepare(s, s->arg0);
        break;
    case CMD_ISOLATE_SOURCE:
        if (!s->leased || !s->prepared || s->source_isolated) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->ops->isolate(s);
        break;
    case CMD_EVOLVE:
        if (!s->leased || !s->prepared || !s->source_isolated ||
            s->boundary_pending) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->ops->evolve(s, s->arg0);
        break;
    case CMD_PROJECT_BOUNDARY:
        if (!s->leased || !s->prepared || s->boundary_pending) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->ops->project(s);
        break;
    case CMD_INVERT:
        if (!s->leased || !s->boundary_pending) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->ops->invert(s);
        break;
    case CMD_RESTORE:
        if (!s->leased || !s->boundary_pending) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->ops->restore(s);
        break;
    case CMD_REUSE:
        s->error = ERR_REUSE_REQUIRES_RESTORATION;
        break;
    case CMD_RELEASE_DIAGNOSTIC:
        if (!s->leased || !s->boundary_pending || s->response_ready) {
            s->error = ERR_BAD_STATE;
            return;
        }
        s->response_ready = true;
        s->boundary_pending = false;
        s->leased = false;
        s->spent = true;
        s->owner_id = 0;
        s->program_id = 0;
        break;
    case CMD_CANONICAL_MODEL_REINITIALIZE:
        if (!s->spent) {
            s->error = ERR_BAD_STATE;
            return;
        }
        /*
         * This is an explicit model-reset sham, not a snapshot, inverse, or
         * physical restoration operation.  Real migration streams are part
         * of the trusted backend and are evaluated by a separate control.
         */
        s->reinitializations++;
        s->generation++;
        p0_clear_transaction(s, true);
        break;
    default:
        s->error = ERR_BAD_ARGUMENT;
        break;
    }
}

static uint64_t phase_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    PhaseQemuV0State *s = opaque;

    switch (addr) {
    case REG_MAGIC:
        return PHASE_QEMU_MAGIC;
    case REG_ABI:
        return PHASE_QEMU_ABI;
    case REG_BACKEND:
        return s->ops->backend_id;
    case REG_CAPABILITIES:
        return s->ops->capabilities;
    case REG_STATUS:
        return phase_status(s);
    case REG_ERROR:
        return s->error;
    case REG_GENERATION:
        return s->generation;
    case REG_REINITIALIZATIONS:
        return s->reinitializations;
    case REG_ARG0:
        return s->arg0;
    case REG_ARG1:
        return s->arg1;
    case REG_BARRIER_RECEIPT:
        return s->barrier_receipt;
    case REG_VIRTUAL_CYCLES:
        return s->virtual_cycles;
    case REG_BOUNDARY_I:
        return s->response_ready ? s->boundary_i : UINT64_MAX;
    case REG_BOUNDARY_Q:
        return s->response_ready ? s->boundary_q : UINT64_MAX;
    case REG_BOUNDARY_ENERGY:
        return s->response_ready ? s->boundary_energy_q30 : UINT64_MAX;
    case REG_RESOURCE_CELLS:
        /* source2 + carrier2 + detector2 + bath1 + boundary3 */
        return 10;
    case REG_RETAINED_HISTORY_CELLS:
        return 0;
    case REG_REQUEST_OWNER:
        return s->request_owner_id;
    case REG_REQUEST_PROGRAM:
        return s->request_program_id;
    default:
        return UINT64_MAX;
    }
}

static void phase_mmio_write(void *opaque, hwaddr addr, uint64_t value,
                             unsigned size)
{
    PhaseQemuV0State *s = opaque;

    switch (addr) {
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
        s->request_owner_id = value;
        break;
    case REG_REQUEST_PROGRAM:
        s->request_program_id = value;
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
    },
    .impl = {
        .min_access_size = 4,
        .max_access_size = 8,
    },
};

static void phase_qemu_v0_reset(DeviceState *dev)
{
    PhaseQemuV0State *s = PHASE_QEMU_V0(dev);

    p0_clear_transaction(s, false);
    s->generation = 0;
}

static int phase_qemu_v0_post_load(void *opaque, int version_id)
{
    PhaseQemuV0State *s = opaque;

    (void)version_id;
    if (s->carrier_present != s->configured_carrier_present ||
        s->phase_arm > 1 || s->ringdown_steps > P0_MAX_EVOLVE_STEPS ||
        (s->barrier_receipt != 0 && s->barrier_receipt != 8) ||
        s->source_isolated != (s->barrier_receipt == 8) ||
        (s->leased && (s->owner_id == 0 || s->program_id == 0)) ||
        (s->spent && s->leased) ||
        (s->boundary_pending &&
         (!s->leased || s->response_ready || s->spent)) ||
        (s->response_ready &&
         (!s->spent || s->leased || s->boundary_pending)) ||
        (!s->prepared && (s->source_isolated || s->boundary_pending))) {
        return -EINVAL;
    }
    return 0;
}

static const VMStateDescription vmstate_phase_qemu_v0 = {
    .name = "phase-qemu-v0",
    .version_id = 1,
    .minimum_version_id = 1,
    .post_load = phase_qemu_v0_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, PhaseQemuV0State),
        VMSTATE_BOOL(carrier_present, PhaseQemuV0State),
        VMSTATE_BOOL(leased, PhaseQemuV0State),
        VMSTATE_BOOL(prepared, PhaseQemuV0State),
        VMSTATE_BOOL(source_isolated, PhaseQemuV0State),
        VMSTATE_BOOL(boundary_pending, PhaseQemuV0State),
        VMSTATE_BOOL(response_ready, PhaseQemuV0State),
        VMSTATE_BOOL(spent, PhaseQemuV0State),
        VMSTATE_UINT32(error, PhaseQemuV0State),
        VMSTATE_UINT32(owner_id, PhaseQemuV0State),
        VMSTATE_UINT32(program_id, PhaseQemuV0State),
        VMSTATE_UINT32(request_owner_id, PhaseQemuV0State),
        VMSTATE_UINT32(request_program_id, PhaseQemuV0State),
        VMSTATE_UINT32(generation, PhaseQemuV0State),
        VMSTATE_UINT32(reinitializations, PhaseQemuV0State),
        VMSTATE_UINT32(arg0, PhaseQemuV0State),
        VMSTATE_UINT32(arg1, PhaseQemuV0State),
        VMSTATE_UINT32(phase_arm, PhaseQemuV0State),
        VMSTATE_UINT32(barrier_receipt, PhaseQemuV0State),
        VMSTATE_UINT32(ringdown_steps, PhaseQemuV0State),
        VMSTATE_UINT64(virtual_cycles, PhaseQemuV0State),
        VMSTATE_UINT64(dissipated_energy_q30, PhaseQemuV0State),
        VMSTATE_INT64(source_i, PhaseQemuV0State),
        VMSTATE_INT64(source_q, PhaseQemuV0State),
        VMSTATE_INT64(carrier_i, PhaseQemuV0State),
        VMSTATE_INT64(carrier_q, PhaseQemuV0State),
        VMSTATE_INT64(detector_i, PhaseQemuV0State),
        VMSTATE_INT64(detector_q, PhaseQemuV0State),
        VMSTATE_INT64(boundary_i, PhaseQemuV0State),
        VMSTATE_INT64(boundary_q, PhaseQemuV0State),
        VMSTATE_UINT64(boundary_energy_q30, PhaseQemuV0State),
        VMSTATE_END_OF_LIST()
    }
};

static void phase_qemu_v0_realize(PCIDevice *pdev, Error **errp)
{
    PhaseQemuV0State *s = PHASE_QEMU_V0(pdev);

    s->ops = &p0_ops;
    s->configured_carrier_present = s->carrier_present;
    memory_region_init_io(&s->mmio, OBJECT(s), &phase_mmio_ops, s,
                          "phase-qemu-v0-mmio", 4 * KiB);
    pci_register_bar(pdev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
}

static void phase_qemu_v0_instance_init(Object *obj)
{
    PhaseQemuV0State *s = PHASE_QEMU_V0(obj);

    s->carrier_present = true;
    s->ops = &p0_ops;
}

static const Property phase_qemu_v0_properties[] = {
    DEFINE_PROP_BOOL("carrier-present", PhaseQemuV0State,
                     carrier_present, true),
};

static void phase_qemu_v0_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->realize = phase_qemu_v0_realize;
    pc->vendor_id = PHASE_QEMU_VENDOR_ID;
    pc->device_id = PHASE_QEMU_DEVICE_ID;
    pc->revision = 0x00;
    pc->class_id = PCI_CLASS_OTHERS;
    dc->vmsd = &vmstate_phase_qemu_v0;
    device_class_set_legacy_reset(dc, phase_qemu_v0_reset);
    device_class_set_props(dc, phase_qemu_v0_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo phase_qemu_v0_info = {
    .name = TYPE_PHASE_QEMU_V0,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(PhaseQemuV0State),
    .instance_init = phase_qemu_v0_instance_init,
    .class_init = phase_qemu_v0_class_init,
    .interfaces = (const InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void phase_qemu_v0_register_types(void)
{
    type_register_static(&phase_qemu_v0_info);
}

type_init(phase_qemu_v0_register_types)
