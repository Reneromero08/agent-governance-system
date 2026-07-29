#define NECKLACE_TWO_SHARED_LATENT_ENTRY \
    two_shared_latent_direct_predecessor_main
#include "four_rotor_necklace_two_shared_latent_phase.cpp"
#undef NECKLACE_TWO_SHARED_LATENT_ENTRY

#include <cerrno>
#include <cstring>
#include <string>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

constexpr std::uint32_t kTwoPortProtocolMagic = 0x43564c50U;
constexpr std::uint64_t kTwoPortLeaseTag = 0x4c454153454c4154ULL;
constexpr std::uint64_t kTwoPortStageTag = 0x54574f504f525453ULL;
constexpr std::uint64_t kPortALeaseTag = 0x504f5254414c4541ULL;
constexpr std::uint64_t kPortBLeaseTag = 0x504f5254424c4542ULL;
constexpr std::uint32_t kPortAId = 0x4c415431U;
constexpr std::uint32_t kPortBId = 0x4c415432U;
constexpr std::uint32_t kPortAType = 0x50484131U;
constexpr std::uint32_t kPortBType = 0x50484231U;

enum TwoPortCommand : std::uint32_t {
    kTwoPortInitialize = 1,
    kTwoPortBegin = 2,
    kTwoPortProject = 3,
    kTwoPortContinue = 4,
    kTwoPortReuse = 5,
    kTwoPortMissingInverse = 6,
    kTwoPortReorderedInverse = 7,
    kTwoPortWrongSemantic = 8,
    kTwoPortWrongType = 9,
    kTwoPortNullCarrier = 10,
    kTwoPortSnapshot = 11,
    kTwoPortStop = 12,
    kTwoPortWrongOwnerA = 13,
    kTwoPortWrongOwnerB = 14,
    kTwoPortUndermerge = 15,
    kTwoPortDuplicatePort = 16,
    kTwoPortStaleInternalGeneration = 17,
    kTwoPortWrongInternalLease = 18,
};

enum TwoPortStatus : std::uint32_t {
    kTwoPortOk = 0,
    kTwoPortDenied = 1,
    kTwoPortError = 2,
};

enum TwoPortFlags : std::uint32_t {
    kTwoPortBoundaryValid = 1U,
    kTwoPortRestored = 2U,
    kTwoPortStageResident = 4U,
    kTwoPortReuseFlag = 8U,
};

#pragma pack(push, 1)
struct TwoPortRequest {
    std::uint32_t magic;
    std::uint32_t command;
    std::uint32_t generation;
    std::uint32_t reserved;
    std::uint64_t lease;
    std::uint64_t nonce;
};

struct TwoPortResponse {
    std::uint32_t magic;
    std::uint32_t status;
    std::uint32_t command;
    std::uint32_t generation;
    std::uint32_t flags;
    std::uint64_t lease;
    std::uint64_t receipt;
    double boundary[kMaximumCollision + 1];
    double restoration_error;
    double norm_error;
    std::uint64_t native_operations;
};
#pragma pack(pop)

static_assert(sizeof(TwoPortRequest) == 32U);
static_assert(sizeof(TwoPortResponse) == 116U);

struct PortCustody {
    std::uint32_t id = 0;
    std::uint32_t type = 0;
    std::uint32_t owner = 0;
    std::uint32_t generation = 0;
    std::uint64_t lease = 0;
};

struct BoundTwoPortModule {
    TwoPortModule module{};
    bool consumes_a = false;
    bool consumes_b = false;
    PortCustody port_a{};
    PortCustody port_b{};
};

struct TwoPortServiceState {
    Plan plan = compile_plan();
    std::vector<Complex> baseline = make_two_port_carrier(plan, 0);
    std::vector<Complex> carrier = baseline;
    const Complex *backing = carrier.data();
    std::vector<TwoPortModule> program = two_port_primary_program();
    PortCustody port_a{
        kPortAId, kPortAType, kPortAOwner, 0U, 0U
    };
    PortCustody port_b{
        kPortBId, kPortBType, kPortBOwner, 0U, 0U
    };
    std::vector<BoundTwoPortModule> bound_program{};
    bool initialized = false;
    bool staged = false;
    bool poisoned = false;
    bool null_mode = false;
    std::size_t applied = 0;
    std::uint32_t generation = 0;
    std::uint64_t lease = 0;
    std::uint64_t last_nonce = 0;
};

bool same_port_custody(
    const PortCustody &left,
    const PortCustody &right
) {
    return
        left.id == right.id
        && left.type == right.type
        && left.owner == right.owner
        && left.generation == right.generation
        && left.lease == right.lease;
}

bool empty_port_custody(const PortCustody &port) {
    return
        port.id == 0U
        && port.type == 0U
        && port.owner == 0U
        && port.generation == 0U
        && port.lease == 0U;
}

std::vector<BoundTwoPortModule> bind_two_port_program(
    const std::vector<TwoPortModule> &program,
    const PortCustody &port_a,
    const PortCustody &port_b
) {
    std::vector<BoundTwoPortModule> result;
    result.reserve(program.size());
    for (const TwoPortModule &module : program) {
        BoundTwoPortModule bound;
        bound.module = module;
        if (
            module.scope == TwoPortScope::PortA
            || module.scope == TwoPortScope::Joint
        ) {
            bound.consumes_a = true;
            bound.port_a = port_a;
        }
        if (
            module.scope == TwoPortScope::PortB
            || module.scope == TwoPortScope::Joint
        ) {
            bound.consumes_b = true;
            bound.port_b = port_b;
        }
        result.push_back(bound);
    }
    return result;
}

bool same_two_port_module(
    const TwoPortModule &left,
    const TwoPortModule &right
) {
    return
        left.feature == right.feature
        && left.scope == right.scope
        && left.axis == right.axis
        && left.separation == right.separation
        && left.strength == right.strength
        && left.chirp == right.chirp
        && left.owner_a == right.owner_a
        && left.owner_b == right.owner_b;
}

bool same_two_port_program(
    const std::vector<TwoPortModule> &left,
    const std::vector<TwoPortModule> &right
) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (!same_two_port_module(left[index], right[index])) {
            return false;
        }
    }
    return true;
}

bool custody_tuple_valid(
    const PortCustody &port,
    std::uint32_t expected_id,
    std::uint32_t expected_type,
    std::uint32_t expected_owner,
    std::uint32_t generation
) {
    return
        port.id == expected_id
        && port.type == expected_type
        && port.owner == expected_owner
        && port.generation == generation
        && port.lease != 0U;
}

bool bound_two_port_module_valid(
    const BoundTwoPortModule &bound,
    const PortCustody &port_a,
    const PortCustody &port_b
) {
    if (!valid_two_port_module(bound.module)) {
        return false;
    }
    const bool expects_a =
        bound.module.scope == TwoPortScope::PortA
        || bound.module.scope == TwoPortScope::Joint;
    const bool expects_b =
        bound.module.scope == TwoPortScope::PortB
        || bound.module.scope == TwoPortScope::Joint;
    return
        bound.consumes_a == expects_a
        && bound.consumes_b == expects_b
        && (
            expects_a
                ? same_port_custody(bound.port_a, port_a)
                : empty_port_custody(bound.port_a)
        )
        && (
            expects_b
                ? same_port_custody(bound.port_b, port_b)
                : empty_port_custody(bound.port_b)
        );
}

bool bound_two_port_program_valid(
    const std::vector<BoundTwoPortModule> &bound_program,
    const std::vector<TwoPortModule> &public_program,
    const std::vector<TwoPortModule> &expected_program,
    const PortCustody &port_a,
    const PortCustody &port_b
) {
    if (
        bound_program.size() != public_program.size()
        || !same_two_port_program(
            public_program, expected_program
        )
    ) {
        return false;
    }
    for (std::size_t index = 0;
         index < bound_program.size();
         ++index) {
        if (
            !same_two_port_module(
                bound_program[index].module,
                public_program[index]
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

bool two_port_custody_valid(
    const TwoPortServiceState &state
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
        )
        && valid_two_port_program(state.program)
        && bound_two_port_program_valid(
            state.bound_program,
            state.program,
            two_port_primary_program(),
            state.port_a,
            state.port_b
        );
}

TwoPortResponse two_port_base(
    const TwoPortRequest &request,
    const TwoPortServiceState &state
) {
    TwoPortResponse response{};
    response.magic = kTwoPortProtocolMagic;
    response.status = kTwoPortOk;
    response.command = request.command;
    response.generation = state.generation;
    response.lease = state.lease;
    return response;
}

TwoPortResponse two_port_denied(
    const TwoPortRequest &request,
    const TwoPortServiceState &state
) {
    TwoPortResponse response = two_port_base(request, state);
    response.status = kTwoPortDenied;
    response.receipt = request.nonce ^ 0x44454e494544ULL;
    if (state.staged) {
        response.flags |= kTwoPortStageResident;
    }
    return response;
}

void two_port_copy_boundary(
    TwoPortResponse &response,
    const Boundary &boundary
) {
    for (std::size_t index = 0; index < boundary.size(); ++index) {
        response.boundary[index] = boundary[index];
    }
    response.flags |= kTwoPortBoundaryValid;
}

bool two_port_exact_io(
    int fd,
    void *buffer,
    std::size_t size,
    bool writing
) {
    auto *cursor = static_cast<unsigned char *>(buffer);
    std::size_t transferred = 0;
    while (transferred < size) {
        const ssize_t count = writing
            ? ::send(
                fd,
                cursor + transferred,
                size - transferred,
                MSG_NOSIGNAL
            )
            : ::recv(
                fd,
                cursor + transferred,
                size - transferred,
                0
            );
        if (count == 0) {
            return false;
        }
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        transferred += static_cast<std::size_t>(count);
    }
    return true;
}

TwoPortResponse two_port_initialize(
    const TwoPortRequest &request,
    TwoPortServiceState &state
) {
    TwoPortResponse response = two_port_base(request, state);
    if (
        state.initialized
        || request.generation != 0U
        || request.lease != 0U
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
    state.bound_program = bind_two_port_program(
        state.program, state.port_a, state.port_b
    );
    if (!two_port_custody_valid(state)) {
        response.status = kTwoPortError;
        state.poisoned = true;
        return response;
    }
    response = two_port_base(request, state);
    response.receipt = state.lease;
    return response;
}

bool two_port_owner_matches(
    const TwoPortRequest &request,
    const TwoPortServiceState &state
) {
    return
        state.initialized
        && request.lease == state.lease
        && request.generation == state.generation
        && request.nonce > state.last_nonce
        && two_port_custody_valid(state);
}

TwoPortResponse two_port_begin(
    const TwoPortRequest &request,
    TwoPortServiceState &state
) {
    TwoPortResponse response = two_port_base(request, state);
    if (
        state.null_mode
        || state.staged
        || state.poisoned
        || state.generation != 0U
        || !two_port_custody_valid(state)
    ) {
        response.status = kTwoPortError;
        return response;
    }
    TwoPortStats stats;
    while (state.applied < 3U) {
        two_port_forward_module(
            state.carrier,
            state.plan,
            state.bound_program[state.applied].module,
            stats
        );
        ++state.applied;
    }
    state.staged = true;
    response = two_port_base(request, state);
    response.flags |= kTwoPortStageResident;
    response.receipt =
        request.nonce ^ state.lease ^ kTwoPortStageTag;
    response.native_operations =
        stats.generator.streamed_generator_terms;
    return response;
}

TwoPortResponse two_port_continue(
    const TwoPortRequest &request,
    TwoPortServiceState &state
) {
    TwoPortResponse response = two_port_base(request, state);
    if (
        !state.staged
        || state.applied != 3U
        || state.poisoned
        || !two_port_custody_valid(state)
    ) {
        response.status = kTwoPortError;
        return response;
    }
    TwoPortStats stats;
    while (state.applied < state.bound_program.size()) {
        two_port_forward_module(
            state.carrier,
            state.plan,
            state.bound_program[state.applied].module,
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
        two_port_inverse_module(
            state.carrier,
            state.plan,
            state.bound_program[state.applied].module,
            stats
        );
    }
    state.staged = false;
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
    state.bound_program = bind_two_port_program(
        state.program, state.port_a, state.port_b
    );
    if (!two_port_custody_valid(state)) {
        state.poisoned = true;
        response.status = kTwoPortError;
        return response;
    }
    response = two_port_base(request, state);
    two_port_copy_boundary(response, boundary);
    response.flags |= kTwoPortRestored;
    response.restoration_error = restoration_error;
    response.norm_error = norm_error;
    response.native_operations =
        stats.generator.streamed_generator_terms;
    return response;
}

TwoPortRun execute_bound_two_port_transaction(
    std::vector<Complex> &carrier,
    const std::vector<Complex> &baseline,
    const Plan &plan,
    const std::vector<BoundTwoPortModule> &bound_program,
    const std::vector<TwoPortModule> &expected_program,
    const PortCustody &port_a,
    const PortCustody &port_b
) {
    if (
        !bound_two_port_program_valid(
            bound_program,
            expected_program,
            expected_program,
            port_a,
            port_b
        )
    ) {
        fail("invalid bound two-port transaction");
    }
    TwoPortRun run;
    for (const BoundTwoPortModule &bound : bound_program) {
        if (
            !bound_two_port_module_valid(
                bound, port_a, port_b
            )
        ) {
            fail("two-port tuple changed during forward execution");
        }
        two_port_forward_module(
            carrier, plan, bound.module, run.stats
        );
    }
    run.boundary = two_port_boundary(carrier, plan);
    run.norm_error =
        std::fabs(two_port_weighted_norm(carrier, plan) - 1.0);
    for (std::size_t cursor = bound_program.size();
         cursor > 0U;
         --cursor) {
        const BoundTwoPortModule &bound =
            bound_program[cursor - 1U];
        if (
            !bound_two_port_module_valid(
                bound, port_a, port_b
            )
        ) {
            fail("two-port tuple changed during inverse execution");
        }
        two_port_inverse_module(
            carrier, plan, bound.module, run.stats
        );
    }
    run.restoration_error =
        two_port_l2_distance(carrier, baseline, plan);
    return run;
}

TwoPortResponse two_port_reuse(
    const TwoPortRequest &request,
    TwoPortServiceState &state
) {
    TwoPortResponse response = two_port_base(request, state);
    if (
        state.null_mode
        || state.staged
        || state.poisoned
        || state.generation != 1U
        || !two_port_custody_valid(state)
    ) {
        response.status = kTwoPortError;
        return response;
    }
    const std::vector<TwoPortModule> program =
        two_port_reuse_program();
    const std::vector<BoundTwoPortModule> bound_program =
        bind_two_port_program(
            program, state.port_a, state.port_b
        );
    const TwoPortRun actual = execute_bound_two_port_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        bound_program,
        program,
        state.port_a,
        state.port_b
    );
    std::vector<Complex> fresh = state.baseline;
    const TwoPortRun reference = execute_bound_two_port_transaction(
        fresh,
        state.baseline,
        state.plan,
        bound_program,
        program,
        state.port_a,
        state.port_b
    );
    const double boundary_error = boundary_distance(
        actual.boundary, reference.boundary
    );
    if (
        actual.restoration_error > kTwoPortTolerance
        || reference.restoration_error > kTwoPortTolerance
        || boundary_error > kTwoPortTolerance
        || actual.stats.generator.streamed_generator_terms
            != reference.stats.generator.streamed_generator_terms
        || state.carrier.data() != state.backing
    ) {
        state.poisoned = true;
        response.status = kTwoPortError;
        return response;
    }
    ++state.generation;
    ++state.port_a.generation;
    ++state.port_b.generation;
    state.bound_program = bind_two_port_program(
        state.program, state.port_a, state.port_b
    );
    if (!two_port_custody_valid(state)) {
        state.poisoned = true;
        response.status = kTwoPortError;
        return response;
    }
    response = two_port_base(request, state);
    two_port_copy_boundary(response, actual.boundary);
    response.flags |= kTwoPortRestored | kTwoPortReuseFlag;
    response.restoration_error = actual.restoration_error;
    response.norm_error = boundary_error;
    response.native_operations =
        actual.stats.generator.streamed_generator_terms;
    return response;
}

TwoPortResponse two_port_inverse_control(
    const TwoPortRequest &request,
    TwoPortServiceState &state,
    TwoPortControl control
) {
    TwoPortResponse response = two_port_base(request, state);
    if (
        state.null_mode
        || state.staged
        || state.poisoned
        || state.generation != 0U
        || !two_port_custody_valid(state)
    ) {
        response.status = kTwoPortError;
        return response;
    }
    const std::vector<TwoPortModule> controls = {
        state.bound_program[0].module,
        state.bound_program[1].module,
    };
    const TwoPortRun run = two_port_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        controls,
        control
    );
    response.restoration_error = run.restoration_error;
    response.norm_error = run.norm_error;
    response.native_operations =
        run.stats.generator.streamed_generator_terms;
    state.poisoned = true;
    return response;
}

TwoPortResponse two_port_descriptor_attack(
    const TwoPortRequest &request,
    const TwoPortServiceState &state,
    std::uint32_t command
) {
    std::vector<BoundTwoPortModule> attacked =
        state.bound_program;
    if (command == kTwoPortWrongType) {
        attacked[0].port_a.type ^= 1U;
    } else if (command == kTwoPortWrongOwnerA) {
        attacked[0].port_a.owner ^= 1U;
    } else if (command == kTwoPortWrongOwnerB) {
        attacked[1].port_b.owner ^= 1U;
    } else if (command == kTwoPortUndermerge) {
        attacked[1].consumes_b = false;
        attacked[1].port_b = PortCustody{};
    } else if (command == kTwoPortDuplicatePort) {
        attacked[1].port_b = attacked[1].port_a;
    } else if (command == kTwoPortStaleInternalGeneration) {
        attacked[0].port_a.generation += 1U;
    } else if (command == kTwoPortWrongInternalLease) {
        attacked[1].port_b.lease ^= 1U;
    } else {
        TwoPortResponse response = two_port_base(request, state);
        response.status = kTwoPortError;
        return response;
    }
    if (
        bound_two_port_program_valid(
            attacked,
            state.program,
            two_port_primary_program(),
            state.port_a,
            state.port_b
        )
    ) {
        TwoPortResponse response = two_port_base(request, state);
        response.status = kTwoPortError;
        return response;
    }
    return two_port_denied(request, state);
}

bool two_port_rollback_staged(TwoPortServiceState &state) {
    if (!state.staged) {
        return true;
    }
    TwoPortStats cleanup_stats;
    while (state.applied > 0U) {
        --state.applied;
        two_port_inverse_module(
            state.carrier,
            state.plan,
            state.bound_program[state.applied].module,
            cleanup_stats
        );
    }
    state.staged = false;
    return
        two_port_l2_distance(
            state.carrier, state.baseline, state.plan
        ) <= kTwoPortTolerance
        && state.carrier.data() == state.backing;
}

TwoPortResponse two_port_stop(
    const TwoPortRequest &request,
    TwoPortServiceState &state
) {
    TwoPortResponse response = two_port_base(request, state);
    if (
        state.null_mode
        || state.poisoned
        || !two_port_rollback_staged(state)
        || state.carrier.size() != state.baseline.size()
        || state.carrier.empty()
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

TwoPortResponse two_port_dispatch(
    const TwoPortRequest &request,
    TwoPortServiceState &state
) {
    if (request.magic != kTwoPortProtocolMagic) {
        TwoPortResponse response = two_port_base(request, state);
        response.status = kTwoPortError;
        return response;
    }
    if (request.command == kTwoPortInitialize) {
        const TwoPortResponse response =
            two_port_initialize(request, state);
        state.last_nonce = request.nonce;
        return response;
    }
    if (!two_port_owner_matches(request, state)) {
        return two_port_denied(request, state);
    }
    state.last_nonce = request.nonce;
    if (state.poisoned && request.command != kTwoPortStop) {
        return two_port_denied(request, state);
    }
    switch (request.command) {
        case kTwoPortBegin:
            return two_port_begin(request, state);
        case kTwoPortProject:
            return two_port_denied(request, state);
        case kTwoPortContinue:
            return two_port_continue(request, state);
        case kTwoPortReuse:
            return two_port_reuse(request, state);
        case kTwoPortMissingInverse:
            return two_port_inverse_control(
                request, state, TwoPortControl::Missing
            );
        case kTwoPortReorderedInverse:
            return two_port_inverse_control(
                request, state, TwoPortControl::ReorderedInverse
            );
        case kTwoPortWrongSemantic:
            return two_port_inverse_control(
                request, state, TwoPortControl::WrongSemantic
            );
        case kTwoPortWrongType:
        case kTwoPortWrongOwnerA:
        case kTwoPortWrongOwnerB:
        case kTwoPortUndermerge:
        case kTwoPortDuplicatePort:
        case kTwoPortStaleInternalGeneration:
        case kTwoPortWrongInternalLease:
            return two_port_descriptor_attack(
                request, state, request.command
            );
        case kTwoPortNullCarrier:
        case kTwoPortSnapshot:
            return two_port_denied(request, state);
        case kTwoPortStop:
            return two_port_stop(request, state);
        default:
            return two_port_denied(request, state);
    }
}

bool two_port_disconnect_cleanup(
    TwoPortServiceState &state
) {
    if (!state.staged) {
        return true;
    }
    if (!two_port_rollback_staged(state)) {
        return false;
    }
    const std::vector<TwoPortModule> reuse_program =
        two_port_reuse_program();
    const std::vector<BoundTwoPortModule> bound_reuse =
        bind_two_port_program(
            reuse_program, state.port_a, state.port_b
        );
    const TwoPortRun actual = execute_bound_two_port_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        bound_reuse,
        reuse_program,
        state.port_a,
        state.port_b
    );
    std::vector<Complex> fresh = state.baseline;
    const TwoPortRun reference = execute_bound_two_port_transaction(
        fresh,
        state.baseline,
        state.plan,
        bound_reuse,
        reuse_program,
        state.port_a,
        state.port_b
    );
    return
        actual.restoration_error <= kTwoPortTolerance
        && reference.restoration_error <= kTwoPortTolerance
        && boundary_distance(
            actual.boundary, reference.boundary
        ) <= kTwoPortTolerance
        && actual.stats.generator.streamed_generator_terms
            == reference.stats.generator.streamed_generator_terms
        && state.carrier.data() == state.backing;
}

}  // namespace

#ifndef CATVM_NECKLACE_TWO_SHARED_LATENT_SERVICE_ENTRY
#define CATVM_NECKLACE_TWO_SHARED_LATENT_SERVICE_ENTRY main
#endif

int CATVM_NECKLACE_TWO_SHARED_LATENT_SERVICE_ENTRY(
    int argc,
    char **argv
) {
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

    TwoPortServiceState state;
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
            two_port_dispatch(request, state);
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
    const bool cleanup_ok = two_port_disconnect_cleanup(state);
    ::close(client);
    ::close(listener);
    if (!cleanup_ok || (io_failed && state.poisoned)) {
        return 2;
    }
    return stopped || io_failed ? 0 : 2;
}
