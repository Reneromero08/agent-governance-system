#define NECKLACE_SHARED_LATENT_DEPTH_ENTRY \
    shared_latent_depth_direct_predecessor_main
#include "four_rotor_necklace_shared_latent_depth_compiler.cpp"
#undef NECKLACE_SHARED_LATENT_DEPTH_ENTRY

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

constexpr std::uint32_t kDepthProtocolMagic = 0x43564450U;
constexpr std::uint64_t kDepthLeaseTag = 0x4c45415345445054ULL;
constexpr std::uint64_t kDepthStageTag = 0x5354414745445054ULL;
constexpr std::size_t kMaximumDepth = 64U;

enum DepthCommand : std::uint32_t {
    kDepthInitialize = 1,
    kDepthBegin = 2,
    kDepthProject = 3,
    kDepthContinue = 4,
    kDepthReuse = 5,
    kDepthMissingInverse = 6,
    kDepthReorderedInverse = 7,
    kDepthWrongInverseVariant = 8,
    kDepthWrongOwner = 9,
    kDepthNullCarrier = 10,
    kDepthSnapshot = 11,
    kDepthStop = 12,
};

enum DepthStatus : std::uint32_t {
    kDepthOk = 0,
    kDepthDenied = 1,
    kDepthError = 2,
};

enum DepthFlags : std::uint32_t {
    kDepthBoundaryValid = 1U,
    kDepthRestored = 2U,
    kDepthStageResident = 4U,
    kDepthReuseFlag = 8U,
};

#pragma pack(push, 1)
struct DepthRequest {
    std::uint32_t magic;
    std::uint32_t command;
    std::uint32_t generation;
    std::uint32_t parameter;
    std::uint64_t lease;
    std::uint64_t nonce;
};

struct DepthResponse {
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

static_assert(sizeof(DepthRequest) == 32U);
static_assert(sizeof(DepthResponse) == 116U);

struct DepthServiceState {
    Plan plan = compile_plan();
    std::vector<Complex> baseline = make_latent_carrier(plan, 0);
    std::vector<Complex> carrier = baseline;
    const Complex *backing = carrier.data();
    bool initialized = false;
    bool staged = false;
    bool poisoned = false;
    bool null_mode = false;
    std::size_t depth = 0;
    std::size_t applied = 0;
    std::uint32_t variant = 0;
    std::uint32_t generation = 0;
    std::uint64_t lease = 0;
    std::uint64_t last_nonce = 0;
};

std::size_t request_depth(const DepthRequest &request) {
    return static_cast<std::size_t>(request.parameter & 0xffffU);
}

std::uint32_t request_variant(const DepthRequest &request) {
    return request.parameter >> 16U;
}

bool valid_public_topology(
    std::size_t depth,
    std::uint32_t variant
) {
    if (
        depth == 0U
        || depth > kMaximumDepth
        || variant == 0U
        || variant > 0xffffU
    ) {
        return false;
    }
    for (std::size_t ordinal = 0; ordinal < depth; ++ordinal) {
        if (
            !valid_compiled_depth_module(
                compile_depth_module(variant, ordinal)
            )
        ) {
            return false;
        }
    }
    return true;
}

DepthResponse depth_base(
    const DepthRequest &request,
    const DepthServiceState &state
) {
    DepthResponse response{};
    response.magic = kDepthProtocolMagic;
    response.status = kDepthOk;
    response.command = request.command;
    response.generation = state.generation;
    response.lease = state.lease;
    return response;
}

DepthResponse depth_denied(
    const DepthRequest &request,
    const DepthServiceState &state
) {
    DepthResponse response = depth_base(request, state);
    response.status = kDepthDenied;
    response.receipt = request.nonce ^ 0x44454e494544ULL;
    if (state.staged) {
        response.flags |= kDepthStageResident;
    }
    return response;
}

void depth_copy_boundary(
    DepthResponse &response,
    const Boundary &boundary
) {
    for (std::size_t index = 0; index < boundary.size(); ++index) {
        response.boundary[index] = boundary[index];
    }
    response.flags |= kDepthBoundaryValid;
}

bool depth_exact_io(
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

bool depth_owner_matches(
    const DepthRequest &request,
    const DepthServiceState &state
) {
    return
        state.initialized
        && request.lease == state.lease
        && request.generation == state.generation
        && request.nonce > state.last_nonce;
}

DepthResponse depth_initialize(
    const DepthRequest &request,
    DepthServiceState &state
) {
    DepthResponse response = depth_base(request, state);
    if (
        state.initialized
        || request.generation != 0U
        || request.parameter != 0U
        || request.lease != 0U
    ) {
        response.status = kDepthError;
        return response;
    }
    state.initialized = true;
    state.lease = request.nonce ^ kDepthLeaseTag;
    response = depth_base(request, state);
    response.receipt = state.lease;
    return response;
}

DepthResponse depth_begin(
    const DepthRequest &request,
    DepthServiceState &state
) {
    DepthResponse response = depth_base(request, state);
    const std::size_t depth = request_depth(request);
    const std::uint32_t variant = request_variant(request);
    if (
        state.null_mode
        || state.staged
        || state.poisoned
        || state.generation != 0U
        || !valid_public_topology(depth, variant)
    ) {
        response.status = kDepthError;
        return response;
    }
    state.depth = depth;
    state.variant = variant;
    const std::size_t split = (depth + 1U) / 2U;
    DepthRun run;
    while (state.applied < split) {
        const LatentModule module = compile_depth_module(
            state.variant, state.applied
        );
        if (!valid_compiled_depth_module(module)) {
            response.status = kDepthError;
            state.poisoned = true;
            return response;
        }
        latent_forward_module(
            state.carrier, state.plan, module, run.stats
        );
        ++state.applied;
    }
    state.staged = true;
    response = depth_base(request, state);
    response.flags |= kDepthStageResident;
    response.receipt =
        request.nonce ^ state.lease ^ kDepthStageTag;
    response.native_operations =
        run.stats.generator.streamed_generator_terms;
    return response;
}

DepthResponse depth_continue(
    const DepthRequest &request,
    DepthServiceState &state
) {
    DepthResponse response = depth_base(request, state);
    if (
        !state.staged
        || state.poisoned
        || request_depth(request) != state.depth
        || request_variant(request) != state.variant
    ) {
        response.status = kDepthError;
        return response;
    }
    DepthRun run;
    while (state.applied < state.depth) {
        const LatentModule module = compile_depth_module(
            state.variant, state.applied
        );
        if (!valid_compiled_depth_module(module)) {
            response.status = kDepthError;
            state.poisoned = true;
            return response;
        }
        latent_forward_module(
            state.carrier, state.plan, module, run.stats
        );
        ++state.applied;
    }
    const Boundary boundary =
        latent_boundary(state.carrier, state.plan);
    const double norm_error = std::fabs(
        latent_weighted_norm(state.carrier, state.plan) - 1.0
    );
    while (state.applied > 0U) {
        --state.applied;
        const LatentModule module = compile_depth_module(
            state.variant, state.applied
        );
        latent_inverse_module(
            state.carrier, state.plan, module, run.stats
        );
    }
    state.staged = false;
    const double restoration_error = latent_l2_distance(
        state.carrier, state.baseline, state.plan
    );
    if (
        restoration_error > kLatentTolerance
        || norm_error > kLatentTolerance
        || state.carrier.data() != state.backing
    ) {
        state.poisoned = true;
        response.status = kDepthError;
        return response;
    }
    ++state.generation;
    response = depth_base(request, state);
    depth_copy_boundary(response, boundary);
    response.flags |= kDepthRestored;
    response.restoration_error = restoration_error;
    response.norm_error = norm_error;
    response.native_operations =
        run.stats.generator.streamed_generator_terms;
    return response;
}

DepthResponse depth_reuse(
    const DepthRequest &request,
    DepthServiceState &state
) {
    DepthResponse response = depth_base(request, state);
    const std::size_t depth = request_depth(request);
    const std::uint32_t variant = request_variant(request);
    if (
        state.null_mode
        || state.staged
        || state.poisoned
        || state.generation != 1U
        || !valid_public_topology(depth, variant)
    ) {
        response.status = kDepthError;
        return response;
    }
    const DepthRun actual = depth_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        variant,
        depth,
        DepthControl::Correct
    );
    std::vector<Complex> fresh = state.baseline;
    const DepthRun reference = depth_transaction(
        fresh,
        state.baseline,
        state.plan,
        variant,
        depth,
        DepthControl::Correct
    );
    const double boundary_error = boundary_distance(
        actual.boundary, reference.boundary
    );
    if (
        actual.restoration_error > kLatentTolerance
        || reference.restoration_error > kLatentTolerance
        || boundary_error > kLatentTolerance
        || state.carrier.data() != state.backing
    ) {
        state.poisoned = true;
        response.status = kDepthError;
        return response;
    }
    ++state.generation;
    response = depth_base(request, state);
    depth_copy_boundary(response, actual.boundary);
    response.flags |= kDepthRestored | kDepthReuseFlag;
    response.restoration_error = actual.restoration_error;
    response.norm_error = boundary_error;
    response.native_operations =
        actual.stats.generator.streamed_generator_terms;
    return response;
}

DepthResponse depth_inverse_control(
    const DepthRequest &request,
    DepthServiceState &state,
    DepthControl control
) {
    DepthResponse response = depth_base(request, state);
    const std::size_t depth = request_depth(request);
    const std::uint32_t variant = request_variant(request);
    if (
        state.null_mode
        || state.staged
        || state.poisoned
        || state.generation != 0U
        || !valid_public_topology(depth, variant)
    ) {
        response.status = kDepthError;
        return response;
    }
    const DepthRun run = depth_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        variant,
        depth,
        control
    );
    response.restoration_error = run.restoration_error;
    response.norm_error = run.norm_error;
    response.native_operations =
        run.stats.generator.streamed_generator_terms;
    state.poisoned = true;
    return response;
}

DepthResponse depth_dispatch(
    const DepthRequest &request,
    DepthServiceState &state
) {
    if (request.magic != kDepthProtocolMagic) {
        DepthResponse response = depth_base(request, state);
        response.status = kDepthError;
        return response;
    }
    if (request.command == kDepthInitialize) {
        const DepthResponse response =
            depth_initialize(request, state);
        state.last_nonce = request.nonce;
        return response;
    }
    if (!depth_owner_matches(request, state)) {
        return depth_denied(request, state);
    }
    state.last_nonce = request.nonce;
    if (state.poisoned && request.command != kDepthStop) {
        return depth_denied(request, state);
    }
    switch (request.command) {
        case kDepthBegin:
            return depth_begin(request, state);
        case kDepthProject:
            return depth_denied(request, state);
        case kDepthContinue:
            return depth_continue(request, state);
        case kDepthReuse:
            return depth_reuse(request, state);
        case kDepthMissingInverse:
            return depth_inverse_control(
                request, state, DepthControl::MissingInverse
            );
        case kDepthReorderedInverse:
            return depth_inverse_control(
                request, state, DepthControl::ReorderedInverse
            );
        case kDepthWrongInverseVariant:
            return depth_inverse_control(
                request, state, DepthControl::WrongInverseVariant
            );
        case kDepthWrongOwner: {
            LatentModule attacked = compile_depth_module(
                request_variant(request), 0U
            );
            attacked.owner ^= 1U;
            if (valid_compiled_depth_module(attacked)) {
                DepthResponse response = depth_base(request, state);
                response.status = kDepthError;
                return response;
            }
            return depth_denied(request, state);
        }
        case kDepthNullCarrier:
        case kDepthSnapshot:
            return depth_denied(request, state);
        case kDepthStop:
            return depth_base(request, state);
        default:
            return depth_denied(request, state);
    }
}

bool depth_disconnect_cleanup(DepthServiceState &state) {
    if (!state.staged) {
        return true;
    }
    DepthRun cleanup;
    while (state.applied > 0U) {
        --state.applied;
        const LatentModule module = compile_depth_module(
            state.variant, state.applied
        );
        latent_inverse_module(
            state.carrier, state.plan, module, cleanup.stats
        );
    }
    state.staged = false;
    if (
        latent_l2_distance(
            state.carrier, state.baseline, state.plan
        ) > kLatentTolerance
        || state.carrier.data() != state.backing
    ) {
        return false;
    }
    const std::size_t sentinel_depth = 3U;
    const std::uint32_t sentinel_variant = 5U;
    const DepthRun actual = depth_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        sentinel_variant,
        sentinel_depth,
        DepthControl::Correct
    );
    std::vector<Complex> fresh = state.baseline;
    const DepthRun reference = depth_transaction(
        fresh,
        state.baseline,
        state.plan,
        sentinel_variant,
        sentinel_depth,
        DepthControl::Correct
    );
    return
        actual.restoration_error <= kLatentTolerance
        && reference.restoration_error <= kLatentTolerance
        && boundary_distance(
            actual.boundary, reference.boundary
        ) <= kLatentTolerance
        && state.carrier.data() == state.backing;
}

}  // namespace

#ifndef CATVM_NECKLACE_SHARED_LATENT_DEPTH_ENTRY
#define CATVM_NECKLACE_SHARED_LATENT_DEPTH_ENTRY main
#endif

int CATVM_NECKLACE_SHARED_LATENT_DEPTH_ENTRY(
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

    DepthServiceState state;
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
        DepthRequest request{};
        if (!depth_exact_io(
                client, &request, sizeof(request), false
            )) {
            io_failed = true;
            break;
        }
        const DepthResponse response =
            depth_dispatch(request, state);
        if (!depth_exact_io(
                client,
                const_cast<DepthResponse *>(&response),
                sizeof(response),
                true
            )) {
            io_failed = true;
            break;
        }
        stopped = request.command == kDepthStop;
    }
    const bool cleanup_ok = depth_disconnect_cleanup(state);
    ::close(client);
    ::close(listener);
    if (!cleanup_ok || (io_failed && state.poisoned)) {
        return 2;
    }
    return stopped || io_failed ? 0 : 2;
}
