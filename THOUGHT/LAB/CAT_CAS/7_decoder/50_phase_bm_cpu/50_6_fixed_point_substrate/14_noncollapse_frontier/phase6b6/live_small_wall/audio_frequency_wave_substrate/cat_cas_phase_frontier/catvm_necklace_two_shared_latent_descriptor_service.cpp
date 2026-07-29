#define CATVM_NECKLACE_TWO_SHARED_LATENT_SERVICE_ENTRY \
    fixed_two_port_service_predecessor_main
#include "catvm_necklace_two_shared_latent_service.cpp"
#undef CATVM_NECKLACE_TWO_SHARED_LATENT_SERVICE_ENTRY

#include <array>
#include <limits>

namespace {

/*
 * Bounded public-descriptor compiler over the established two-port carrier.
 *
 * The compiler accepts only public topology fields packed into the existing
 * 32-byte authenticated request.  It has no carrier or boundary input.
 * Owners and the live (id,type,owner,generation,lease) tuples are derived
 * inside the service.  Only forward descriptors are retained; inverse order
 * is rematerialized by reverse traversal after the final boundary is copied
 * internally and before any response is released.
 */

constexpr std::size_t kDescriptorSlots = 3U;
constexpr std::size_t kDescriptorMinimumModules = 4U;
constexpr std::size_t kDescriptorMaximumModules = 8U;
constexpr std::uint32_t kDescriptorProtocolVersion = 1U;
constexpr std::uint64_t kDescriptorChecksumOffset =
    1469598103934665603ULL;
constexpr std::uint64_t kDescriptorChecksumPrime =
    1099511628211ULL;

enum DescriptorCommand : std::uint32_t {
    kDescriptorDeclare = 19,
    kDescriptorAppend = 20,
    kDescriptorSeal = 21,
    kDescriptorStaleEpoch = 22,
    kDescriptorWrongChecksum = 23,
    kDescriptorWrongSlot = 24,
    kDescriptorStaleBoundGeneration = 25,
};

struct DescriptorSlot {
    bool declared = false;
    bool sealed = false;
    std::uint32_t declared_count = 0U;
    std::uint32_t received_count = 0U;
    std::array<std::uint32_t, kDescriptorMaximumModules> wire{};
    std::array<std::uint32_t, kDescriptorMaximumModules> sealed_wire{};
    std::vector<TwoPortModule> program{};
    std::uint32_t stage_cut = 0U;
    std::uint64_t checksum = 0U;
    std::uint64_t epoch = 0U;
};

struct DescriptorServiceState {
    Plan plan = compile_plan();
    std::vector<Complex> baseline = make_two_port_carrier(plan, 0);
    std::vector<Complex> carrier = baseline;
    const Complex *backing = carrier.data();
    PortCustody port_a{
        kPortAId, kPortAType, kPortAOwner, 0U, 0U
    };
    PortCustody port_b{
        kPortBId, kPortBType, kPortBOwner, 0U, 0U
    };
    std::array<DescriptorSlot, kDescriptorSlots> slots{};
    std::vector<BoundTwoPortModule> active_program{};
    std::uint32_t active_slot =
        std::numeric_limits<std::uint32_t>::max();
    std::uint32_t active_stage_cut = 0U;
    std::uint64_t active_checksum = 0U;
    std::uint64_t active_epoch = 0U;
    std::uint32_t active_carrier_generation = 0U;
    std::size_t applied = 0U;
    std::uint32_t generation = 0U;
    std::uint64_t lease = 0U;
    std::uint64_t last_nonce = 0U;
    std::uint64_t next_epoch = 1U;
    bool initialized = false;
    bool staged = false;
    bool poisoned = false;
    bool null_mode = false;
};

static_assert(sizeof(TwoPortModule) == 32U);
static_assert(sizeof(PortCustody) == 24U);
static_assert(sizeof(BoundTwoPortModule) == 88U);
static_assert(sizeof(DescriptorSlot) == 128U);

std::uint32_t descriptor_slot_from_control(std::uint32_t value) {
    return value & 0xffU;
}

std::uint32_t descriptor_count_from_control(std::uint32_t value) {
    return (value >> 8U) & 0xffU;
}

bool descriptor_control_reserved_bits_clear(std::uint32_t value) {
    return (value & 0xffff0000U) == 0U;
}

std::uint32_t descriptor_feature(std::uint32_t value) {
    return value & 0x3U;
}

std::uint32_t descriptor_scope(std::uint32_t value) {
    return (value >> 2U) & 0x3U;
}

std::uint32_t descriptor_axis(std::uint32_t value) {
    return (value >> 4U) & 0x7U;
}

std::uint32_t descriptor_separation(std::uint32_t value) {
    return (value >> 7U) & 0xfU;
}

std::uint32_t descriptor_strength(std::uint32_t value) {
    return (value >> 11U) & 0x1fU;
}

std::uint32_t descriptor_chirp(std::uint32_t value) {
    return (value >> 16U) & 0x1fU;
}

std::uint32_t descriptor_program_slot(std::uint32_t value) {
    return (value >> 21U) & 0x3U;
}

std::uint32_t descriptor_module_index(std::uint32_t value) {
    return (value >> 23U) & 0x7U;
}

bool descriptor_high_bits_clear(std::uint32_t value) {
    return (value >> 26U) == 0U;
}

bool decode_public_descriptor(
    std::uint32_t wire,
    std::uint32_t expected_slot,
    std::uint32_t expected_index,
    TwoPortModule &module
) {
    if (
        !descriptor_high_bits_clear(wire)
        || descriptor_program_slot(wire) != expected_slot
        || descriptor_module_index(wire) != expected_index
    ) {
        return false;
    }
    module.feature = static_cast<TwoPortFeature>(
        descriptor_feature(wire)
    );
    module.scope = static_cast<TwoPortScope>(
        descriptor_scope(wire)
    );
    module.axis = static_cast<TwoPortAxis>(
        descriptor_axis(wire)
    );
    module.separation =
        static_cast<int>(descriptor_separation(wire));
    module.strength =
        static_cast<int>(descriptor_strength(wire));
    module.chirp = static_cast<int>(descriptor_chirp(wire));
    if (module.scope == TwoPortScope::PortA) {
        module.owner_a = kPortAOwner;
        module.owner_b = 0U;
    } else if (module.scope == TwoPortScope::PortB) {
        module.owner_a = 0U;
        module.owner_b = kPortBOwner;
    } else if (module.scope == TwoPortScope::Joint) {
        module.owner_a = kPortAOwner;
        module.owner_b = kPortBOwner;
    }
    return valid_two_port_module(module);
}

std::uint64_t descriptor_checksum_step(
    std::uint64_t checksum,
    std::uint32_t value
) {
    for (std::size_t byte = 0U; byte < 4U; ++byte) {
        checksum ^= static_cast<std::uint8_t>(
            (value >> (8U * byte)) & 0xffU
        );
        checksum *= kDescriptorChecksumPrime;
    }
    return checksum;
}

std::uint32_t descriptor_semantic_word(std::uint32_t wire) {
    return wire & 0x001fffffU;
}

bool compile_public_descriptor_program(
    const std::array<std::uint32_t, kDescriptorMaximumModules> &wire,
    std::uint32_t count,
    std::uint32_t slot,
    std::vector<TwoPortModule> &program,
    std::uint32_t &stage_cut,
    std::uint64_t &checksum
) {
    /*
     * Pure topology compilation: this API intentionally has no carrier,
     * amplitude, boundary, answer, generation, lease, or result argument.
     */
    if (
        slot >= kDescriptorSlots
        || count < kDescriptorMinimumModules
        || count > kDescriptorMaximumModules
    ) {
        return false;
    }
    std::vector<TwoPortModule> candidate;
    candidate.reserve(count);
    std::uint32_t first_joint = count;
    std::uint32_t joint_count = 0U;
    std::uint32_t local_a_count = 0U;
    std::uint32_t local_b_count = 0U;
    bool non_diagonal_local = false;
    for (std::uint32_t index = 0U; index < count; ++index) {
        TwoPortModule module;
        if (!decode_public_descriptor(
                wire[index], slot, index, module
            )) {
            return false;
        }
        for (std::uint32_t prior = 0U; prior < index; ++prior) {
            if (
                descriptor_semantic_word(wire[prior])
                == descriptor_semantic_word(wire[index])
            ) {
                return false;
            }
        }
        if (module.scope == TwoPortScope::PortA) {
            ++local_a_count;
        } else if (module.scope == TwoPortScope::PortB) {
            ++local_b_count;
        } else {
            if (first_joint == count) {
                first_joint = index;
            }
            ++joint_count;
        }
        if (
            module.scope != TwoPortScope::Joint
            && (
                module.axis == TwoPortAxis::X
                || module.axis == TwoPortAxis::Y
            )
        ) {
            non_diagonal_local = true;
        }
        candidate.push_back(module);
    }
    const std::uint32_t candidate_stage_cut = first_joint + 1U;
    bool later_joint = false;
    for (std::uint32_t index = candidate_stage_cut;
         index < count;
         ++index) {
        if (candidate[index].scope == TwoPortScope::Joint) {
            later_joint = true;
        }
    }
    if (
        local_a_count == 0U
        || local_b_count == 0U
        || joint_count < 2U
        || !non_diagonal_local
        || candidate_stage_cut >= count
        || !later_joint
    ) {
        return false;
    }
    std::uint64_t candidate_checksum =
        kDescriptorChecksumOffset;
    candidate_checksum = descriptor_checksum_step(
        candidate_checksum, kDescriptorProtocolVersion
    );
    candidate_checksum = descriptor_checksum_step(
        candidate_checksum, count
    );
    candidate_checksum = descriptor_checksum_step(
        candidate_checksum, candidate_stage_cut
    );
    for (std::uint32_t index = 0U; index < count; ++index) {
        candidate_checksum = descriptor_checksum_step(
            candidate_checksum,
            descriptor_semantic_word(wire[index])
        );
    }
    program = candidate;
    stage_cut = candidate_stage_cut;
    checksum = candidate_checksum;
    return true;
}

bool descriptor_bound_program_valid(
    const std::vector<BoundTwoPortModule> &bound_program,
    const std::vector<TwoPortModule> &program,
    const PortCustody &port_a,
    const PortCustody &port_b
) {
    if (
        bound_program.size() != program.size()
        || !valid_two_port_program(program)
    ) {
        return false;
    }
    for (std::size_t index = 0U;
         index < program.size();
         ++index) {
        if (
            !same_two_port_module(
                bound_program[index].module, program[index]
            )
            || !bound_two_port_module_valid(
                bound_program[index], port_a, port_b
            )
        ) {
            return false;
        }
    }
    return true;
}

bool descriptor_live_custody_valid(
    const DescriptorServiceState &state
) {
    return
        state.initialized
        && state.port_a.id != state.port_b.id
        && state.port_a.type != state.port_b.type
        && state.port_a.owner != state.port_b.owner
        && state.port_a.lease != state.port_b.lease
        && custody_tuple_valid(
            state.port_a,
            kPortAId,
            kPortAType,
            kPortAOwner,
            state.generation
        )
        && custody_tuple_valid(
            state.port_b,
            kPortBId,
            kPortBType,
            kPortBOwner,
            state.generation
        );
}

bool descriptor_slot_immutable(const DescriptorSlot &slot) {
    if (!slot.sealed) {
        return true;
    }
    if (
        slot.declared_count != slot.received_count
        || slot.program.size() != slot.declared_count
    ) {
        return false;
    }
    for (std::uint32_t index = 0U;
         index < slot.declared_count;
         ++index) {
        if (slot.wire[index] != slot.sealed_wire[index]) {
            return false;
        }
    }
    std::vector<TwoPortModule> program;
    std::uint32_t stage_cut = 0U;
    std::uint64_t checksum = 0U;
    return
        compile_public_descriptor_program(
            slot.sealed_wire,
            slot.declared_count,
            descriptor_program_slot(slot.sealed_wire[0]),
            program,
            stage_cut,
            checksum
        )
        && same_two_port_program(program, slot.program)
        && stage_cut == slot.stage_cut
        && checksum == slot.checksum;
}

bool descriptor_active_valid(
    const DescriptorServiceState &state
) {
    if (
        !state.staged
        || state.active_slot >= kDescriptorSlots
    ) {
        return false;
    }
    const DescriptorSlot &slot = state.slots[state.active_slot];
    return
        descriptor_live_custody_valid(state)
        && descriptor_slot_immutable(slot)
        && slot.sealed
        && state.active_checksum == slot.checksum
        && state.active_epoch == slot.epoch
        && state.active_stage_cut == slot.stage_cut
        && state.active_carrier_generation == state.generation
        && descriptor_bound_program_valid(
            state.active_program,
            slot.program,
            state.port_a,
            state.port_b
        );
}

TwoPortResponse descriptor_base(
    const TwoPortRequest &request,
    const DescriptorServiceState &state
) {
    TwoPortResponse response{};
    response.magic = kTwoPortProtocolMagic;
    response.status = kTwoPortOk;
    response.command = request.command;
    response.generation = state.generation;
    response.lease = state.lease;
    return response;
}

TwoPortResponse descriptor_denied(
    const TwoPortRequest &request,
    const DescriptorServiceState &state
) {
    TwoPortResponse response = descriptor_base(request, state);
    response.status = kTwoPortDenied;
    response.receipt = request.nonce ^ 0x4445534344454e59ULL;
    if (state.staged) {
        response.flags |= kTwoPortStageResident;
    }
    return response;
}

bool descriptor_owner_matches(
    const TwoPortRequest &request,
    const DescriptorServiceState &state
) {
    return
        state.initialized
        && request.lease == state.lease
        && request.generation == state.generation
        && request.nonce > state.last_nonce
        && descriptor_live_custody_valid(state);
}

TwoPortResponse descriptor_initialize(
    const TwoPortRequest &request,
    DescriptorServiceState &state
) {
    TwoPortResponse response = descriptor_base(request, state);
    if (
        state.initialized
        || request.generation != 0U
        || request.lease != 0U
        || request.reserved != 0U
    ) {
        response.status = kTwoPortError;
        return response;
    }
    state.initialized = true;
    state.lease = request.nonce ^ kTwoPortLeaseTag;
    state.port_a.lease =
        request.nonce * 0x9e3779b97f4a7c15ULL
        ^ state.lease ^ kPortALeaseTag;
    state.port_b.lease =
        request.nonce * 0xbf58476d1ce4e5b9ULL
        ^ state.lease ^ kPortBLeaseTag;
    if (!descriptor_live_custody_valid(state)) {
        state.poisoned = true;
        response.status = kTwoPortError;
        return response;
    }
    response = descriptor_base(request, state);
    response.receipt = state.lease;
    return response;
}

TwoPortResponse descriptor_declare(
    const TwoPortRequest &request,
    DescriptorServiceState &state
) {
    if (
        state.staged
        || state.poisoned
        || !descriptor_control_reserved_bits_clear(request.reserved)
    ) {
        return descriptor_denied(request, state);
    }
    const std::uint32_t slot_index =
        descriptor_slot_from_control(request.reserved);
    const std::uint32_t count =
        descriptor_count_from_control(request.reserved);
    if (
        slot_index >= kDescriptorSlots
        || count < kDescriptorMinimumModules
        || count > kDescriptorMaximumModules
        || state.slots[slot_index].declared
    ) {
        return descriptor_denied(request, state);
    }
    DescriptorSlot slot;
    slot.declared = true;
    slot.declared_count = count;
    state.slots[slot_index] = slot;
    TwoPortResponse response = descriptor_base(request, state);
    response.native_operations = count;
    return response;
}

TwoPortResponse descriptor_append(
    const TwoPortRequest &request,
    DescriptorServiceState &state
) {
    if (state.staged || state.poisoned) {
        return descriptor_denied(request, state);
    }
    const std::uint32_t slot_index =
        descriptor_program_slot(request.reserved);
    const std::uint32_t index =
        descriptor_module_index(request.reserved);
    if (slot_index >= kDescriptorSlots) {
        return descriptor_denied(request, state);
    }
    DescriptorSlot &slot = state.slots[slot_index];
    TwoPortModule decoded;
    if (
        !slot.declared
        || slot.sealed
        || index != slot.received_count
        || index >= slot.declared_count
        || !decode_public_descriptor(
            request.reserved, slot_index, index, decoded
        )
    ) {
        return descriptor_denied(request, state);
    }
    slot.wire[index] = request.reserved;
    ++slot.received_count;
    TwoPortResponse response = descriptor_base(request, state);
    response.native_operations = slot.received_count;
    return response;
}

TwoPortResponse descriptor_seal(
    const TwoPortRequest &request,
    DescriptorServiceState &state
) {
    if (
        state.staged
        || state.poisoned
        || (request.reserved & ~0xffU) != 0U
    ) {
        return descriptor_denied(request, state);
    }
    const std::uint32_t slot_index = request.reserved;
    if (slot_index >= kDescriptorSlots) {
        return descriptor_denied(request, state);
    }
    DescriptorSlot &slot = state.slots[slot_index];
    if (
        !slot.declared
        || slot.sealed
        || slot.received_count != slot.declared_count
    ) {
        return descriptor_denied(request, state);
    }
    std::vector<TwoPortModule> program;
    std::uint32_t stage_cut = 0U;
    std::uint64_t checksum = 0U;
    if (!compile_public_descriptor_program(
            slot.wire,
            slot.declared_count,
            slot_index,
            program,
            stage_cut,
            checksum
        )) {
        return descriptor_denied(request, state);
    }
    slot.sealed_wire = slot.wire;
    slot.program = program;
    slot.stage_cut = stage_cut;
    slot.checksum = checksum;
    slot.epoch = state.next_epoch++;
    slot.sealed = true;
    if (!descriptor_slot_immutable(slot)) {
        state.poisoned = true;
        TwoPortResponse response = descriptor_base(request, state);
        response.status = kTwoPortError;
        return response;
    }
    TwoPortResponse response = descriptor_base(request, state);
    response.receipt = checksum;
    response.native_operations = slot.declared_count;
    return response;
}

TwoPortResponse descriptor_begin(
    const TwoPortRequest &request,
    DescriptorServiceState &state
) {
    TwoPortResponse response = descriptor_base(request, state);
    const std::uint32_t slot_index = request.reserved;
    if (
        state.null_mode
        || state.staged
        || state.poisoned
        || slot_index >= kDescriptorSlots
        || !state.slots[slot_index].sealed
        || !descriptor_slot_immutable(state.slots[slot_index])
        || state.carrier.data() != state.backing
        || state.applied != 0U
    ) {
        return descriptor_denied(request, state);
    }
    const DescriptorSlot &slot = state.slots[slot_index];
    state.active_program = bind_two_port_program(
        slot.program, state.port_a, state.port_b
    );
    state.active_slot = slot_index;
    state.active_stage_cut = slot.stage_cut;
    state.active_checksum = slot.checksum;
    state.active_epoch = slot.epoch;
    state.active_carrier_generation = state.generation;
    state.staged = true;
    if (!descriptor_active_valid(state)) {
        state.poisoned = true;
        response.status = kTwoPortError;
        return response;
    }
    TwoPortStats stats;
    while (state.applied < state.active_stage_cut) {
        if (!descriptor_active_valid(state)) {
            state.poisoned = true;
            response.status = kTwoPortError;
            return response;
        }
        two_port_forward_module(
            state.carrier,
            state.plan,
            state.active_program[state.applied].module,
            stats
        );
        ++state.applied;
    }
    response = descriptor_base(request, state);
    response.flags |= kTwoPortStageResident;
    response.receipt =
        request.nonce ^ state.lease ^ kTwoPortStageTag
        ^ slot.checksum;
    response.native_operations =
        stats.generator.streamed_generator_terms;
    return response;
}

void descriptor_clear_active(DescriptorServiceState &state) {
    state.active_program.clear();
    state.active_slot =
        std::numeric_limits<std::uint32_t>::max();
    state.active_stage_cut = 0U;
    state.active_checksum = 0U;
    state.active_epoch = 0U;
    state.active_carrier_generation = 0U;
    state.applied = 0U;
    state.staged = false;
}

TwoPortResponse descriptor_continue(
    const TwoPortRequest &request,
    DescriptorServiceState &state
) {
    TwoPortResponse response = descriptor_base(request, state);
    if (
        request.reserved != 0U
        || !descriptor_active_valid(state)
        || state.applied != state.active_stage_cut
    ) {
        return descriptor_denied(request, state);
    }
    TwoPortStats stats;
    while (state.applied < state.active_program.size()) {
        if (!descriptor_active_valid(state)) {
            state.poisoned = true;
            response.status = kTwoPortError;
            return response;
        }
        two_port_forward_module(
            state.carrier,
            state.plan,
            state.active_program[state.applied].module,
            stats
        );
        ++state.applied;
    }
    const Boundary boundary =
        two_port_boundary(state.carrier, state.plan);
    const double norm_error = std::fabs(
        two_port_weighted_norm(state.carrier, state.plan) - 1.0
    );
    while (state.applied > 0U) {
        --state.applied;
        if (!descriptor_active_valid(state)) {
            state.poisoned = true;
            response.status = kTwoPortError;
            return response;
        }
        two_port_inverse_module(
            state.carrier,
            state.plan,
            state.active_program[state.applied].module,
            stats
        );
    }
    const double restoration_error = two_port_l2_distance(
        state.carrier, state.baseline, state.plan
    );
    if (
        restoration_error > kTwoPortTolerance
        || norm_error > kTwoPortTolerance
        || state.carrier.data() != state.backing
    ) {
        state.poisoned = true;
        response.status = kTwoPortError;
        return response;
    }
    ++state.generation;
    ++state.port_a.generation;
    ++state.port_b.generation;
    descriptor_clear_active(state);
    if (!descriptor_live_custody_valid(state)) {
        state.poisoned = true;
        response.status = kTwoPortError;
        return response;
    }
    response = descriptor_base(request, state);
    two_port_copy_boundary(response, boundary);
    response.flags |= kTwoPortRestored;
    if (state.generation > 1U) {
        response.flags |= kTwoPortReuseFlag;
    }
    response.restoration_error = restoration_error;
    response.norm_error = norm_error;
    response.native_operations =
        stats.generator.streamed_generator_terms;
    return response;
}

bool descriptor_rollback_staged(DescriptorServiceState &state) {
    if (!state.staged) {
        return true;
    }
    if (!descriptor_active_valid(state)) {
        return false;
    }
    TwoPortStats stats;
    while (state.applied > 0U) {
        --state.applied;
        if (!descriptor_active_valid(state)) {
            return false;
        }
        two_port_inverse_module(
            state.carrier,
            state.plan,
            state.active_program[state.applied].module,
            stats
        );
    }
    const bool restored =
        two_port_l2_distance(
            state.carrier, state.baseline, state.plan
        ) <= kTwoPortTolerance
        && state.carrier.data() == state.backing;
    descriptor_clear_active(state);
    return restored;
}

TwoPortResponse descriptor_inverse_attack(
    const TwoPortRequest &request,
    DescriptorServiceState &state,
    TwoPortControl control
) {
    TwoPortResponse response = descriptor_base(request, state);
    const std::uint32_t slot_index = request.reserved;
    if (
        state.staged
        || state.poisoned
        || slot_index >= kDescriptorSlots
        || !state.slots[slot_index].sealed
        || !descriptor_slot_immutable(state.slots[slot_index])
    ) {
        return descriptor_denied(request, state);
    }
    const TwoPortRun run = two_port_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        state.slots[slot_index].program,
        control
    );
    response.restoration_error = run.restoration_error;
    response.norm_error = run.norm_error;
    response.native_operations =
        run.stats.generator.streamed_generator_terms;
    state.poisoned = true;
    return response;
}

TwoPortResponse descriptor_metadata_attack(
    const TwoPortRequest &request,
    const DescriptorServiceState &state
) {
    if (request.reserved >= kDescriptorSlots) {
        return descriptor_denied(request, state);
    }
    const DescriptorSlot &slot = state.slots[request.reserved];
    if (!slot.sealed) {
        return descriptor_denied(request, state);
    }
    std::uint64_t epoch = slot.epoch;
    std::uint64_t checksum = slot.checksum;
    std::uint32_t slot_index = request.reserved;
    PortCustody port_a = state.port_a;
    if (request.command == kDescriptorStaleEpoch) {
        --epoch;
    } else if (request.command == kDescriptorWrongChecksum) {
        checksum ^= 1U;
    } else if (request.command == kDescriptorWrongSlot) {
        slot_index = (slot_index + 1U) % kDescriptorSlots;
    } else if (request.command == kDescriptorStaleBoundGeneration) {
        ++port_a.generation;
    }
    const std::vector<BoundTwoPortModule> bound =
        bind_two_port_program(
            slot.program, port_a, state.port_b
        );
    const bool invalid =
        epoch != slot.epoch
        || checksum != slot.checksum
        || slot_index != request.reserved
        || !descriptor_bound_program_valid(
            bound,
            slot.program,
            state.port_a,
            state.port_b
        );
    if (!invalid) {
        TwoPortResponse response = descriptor_base(request, state);
        response.status = kTwoPortError;
        return response;
    }
    return descriptor_denied(request, state);
}

TwoPortResponse descriptor_tuple_attack(
    const TwoPortRequest &request,
    const DescriptorServiceState &state
) {
    if (request.reserved >= kDescriptorSlots) {
        return descriptor_denied(request, state);
    }
    const DescriptorSlot &slot = state.slots[request.reserved];
    if (!slot.sealed) {
        return descriptor_denied(request, state);
    }
    std::vector<BoundTwoPortModule> attacked =
        bind_two_port_program(
            slot.program, state.port_a, state.port_b
        );
    if (attacked.empty()) {
        return descriptor_denied(request, state);
    }
    if (request.command == kTwoPortWrongType) {
        attacked[0].port_a.type ^= 1U;
    } else if (request.command == kTwoPortWrongOwnerA) {
        attacked[0].port_a.owner ^= 1U;
    } else if (request.command == kTwoPortWrongOwnerB) {
        for (BoundTwoPortModule &bound : attacked) {
            if (bound.consumes_b) {
                bound.port_b.owner ^= 1U;
                break;
            }
        }
    } else if (request.command == kTwoPortUndermerge) {
        for (BoundTwoPortModule &bound : attacked) {
            if (bound.consumes_a && bound.consumes_b) {
                bound.consumes_b = false;
                bound.port_b = PortCustody{};
                break;
            }
        }
    } else if (request.command == kTwoPortDuplicatePort) {
        for (BoundTwoPortModule &bound : attacked) {
            if (bound.consumes_a && bound.consumes_b) {
                bound.port_b = bound.port_a;
                break;
            }
        }
    } else if (request.command == kTwoPortStaleInternalGeneration) {
        attacked[0].port_a.generation += 1U;
    } else if (request.command == kTwoPortWrongInternalLease) {
        attacked[0].port_a.lease ^= 1U;
    } else {
        return descriptor_denied(request, state);
    }
    if (descriptor_bound_program_valid(
            attacked,
            slot.program,
            state.port_a,
            state.port_b
        )) {
        TwoPortResponse response = descriptor_base(request, state);
        response.status = kTwoPortError;
        return response;
    }
    return descriptor_denied(request, state);
}

TwoPortResponse descriptor_stop(
    const TwoPortRequest &request,
    DescriptorServiceState &state
) {
    TwoPortResponse response = descriptor_base(request, state);
    if (
        request.reserved != 0U
        || state.null_mode
        || state.poisoned
        || !descriptor_rollback_staged(state)
        || state.carrier.empty()
        || state.carrier.size() != state.baseline.size()
        || state.carrier.data() != state.backing
        || two_port_l2_distance(
            state.carrier, state.baseline, state.plan
        ) > kTwoPortTolerance
    ) {
        state.poisoned = true;
        response.status = kTwoPortError;
        return response;
    }
    response.flags |= kTwoPortRestored;
    return response;
}

TwoPortResponse descriptor_dispatch(
    const TwoPortRequest &request,
    DescriptorServiceState &state
) {
    if (request.magic != kTwoPortProtocolMagic) {
        TwoPortResponse response = descriptor_base(request, state);
        response.status = kTwoPortError;
        return response;
    }
    if (request.command == kTwoPortInitialize) {
        const TwoPortResponse response =
            descriptor_initialize(request, state);
        state.last_nonce = request.nonce;
        return response;
    }
    if (!descriptor_owner_matches(request, state)) {
        return descriptor_denied(request, state);
    }
    state.last_nonce = request.nonce;
    if (state.poisoned && request.command != kTwoPortStop) {
        return descriptor_denied(request, state);
    }
    switch (request.command) {
        case kDescriptorDeclare:
            return descriptor_declare(request, state);
        case kDescriptorAppend:
            return descriptor_append(request, state);
        case kDescriptorSeal:
            return descriptor_seal(request, state);
        case kTwoPortBegin:
            return descriptor_begin(request, state);
        case kTwoPortContinue:
            return descriptor_continue(request, state);
        case kTwoPortProject:
        case kTwoPortSnapshot:
        case kTwoPortNullCarrier:
            return descriptor_denied(request, state);
        case kTwoPortMissingInverse:
            return descriptor_inverse_attack(
                request, state, TwoPortControl::Missing
            );
        case kTwoPortReorderedInverse:
            return descriptor_inverse_attack(
                request, state, TwoPortControl::ReorderedInverse
            );
        case kTwoPortWrongSemantic:
            return descriptor_inverse_attack(
                request, state, TwoPortControl::WrongSemantic
            );
        case kTwoPortWrongType:
        case kTwoPortWrongOwnerA:
        case kTwoPortWrongOwnerB:
        case kTwoPortUndermerge:
        case kTwoPortDuplicatePort:
        case kTwoPortStaleInternalGeneration:
        case kTwoPortWrongInternalLease:
            return descriptor_tuple_attack(request, state);
        case kDescriptorStaleEpoch:
        case kDescriptorWrongChecksum:
        case kDescriptorWrongSlot:
        case kDescriptorStaleBoundGeneration:
            return descriptor_metadata_attack(request, state);
        case kTwoPortStop:
            return descriptor_stop(request, state);
        default:
            return descriptor_denied(request, state);
    }
}

bool descriptor_disconnect_cleanup(
    DescriptorServiceState &state
) {
    if (!state.staged) {
        return true;
    }
    return descriptor_rollback_staged(state);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        return 2;
    }
    const std::string mode = argv[1];
    if (mode != "normal" && mode != "null") {
        return 2;
    }
    if (::prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) {
        return 2;
    }
    const std::string socket_path = argv[2];
    if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
        return 2;
    }
    const int listener = ::socket(
        AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0
    );
    if (listener < 0) {
        return 2;
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(
        address.sun_path,
        socket_path.c_str(),
        socket_path.size() + 1U
    );
    if (
        ::bind(
            listener,
            reinterpret_cast<const sockaddr *>(&address),
            sizeof(address)
        ) != 0
        || ::chmod(socket_path.c_str(), 0600) != 0
        || ::listen(listener, 1) != 0
    ) {
        ::close(listener);
        return 2;
    }
    const int client = ::accept4(
        listener, nullptr, nullptr, SOCK_CLOEXEC
    );
    if (client < 0) {
        ::close(listener);
        return 2;
    }
    ucred credentials{};
    socklen_t credential_size = sizeof(credentials);
    if (
        ::getsockopt(
            client,
            SOL_SOCKET,
            SO_PEERCRED,
            &credentials,
            &credential_size
        ) != 0
        || credentials.uid != ::getuid()
    ) {
        ::close(client);
        ::close(listener);
        return 2;
    }
    DescriptorServiceState state;
    if (mode == "null") {
        state.null_mode = true;
        state.carrier.clear();
        state.carrier.shrink_to_fit();
        state.baseline.clear();
        state.baseline.shrink_to_fit();
        state.backing = nullptr;
    }
    bool stopped = false;
    bool io_failed = false;
    while (!stopped) {
        TwoPortRequest request{};
        if (!two_port_exact_io(
                client, &request, sizeof(request), false
            )) {
            io_failed = true;
            break;
        }
        const TwoPortResponse response =
            descriptor_dispatch(request, state);
        if (!two_port_exact_io(
                client,
                const_cast<TwoPortResponse *>(&response),
                sizeof(response),
                true
            )) {
            io_failed = true;
            break;
        }
        stopped =
            request.command == kTwoPortStop
            && response.status != kTwoPortDenied;
    }
    const bool cleanup_ok = descriptor_disconnect_cleanup(state);
    ::close(client);
    ::close(listener);
    if (!cleanup_ok || (io_failed && state.poisoned)) {
        return 2;
    }
    return stopped || io_failed ? 0 : 2;
}
