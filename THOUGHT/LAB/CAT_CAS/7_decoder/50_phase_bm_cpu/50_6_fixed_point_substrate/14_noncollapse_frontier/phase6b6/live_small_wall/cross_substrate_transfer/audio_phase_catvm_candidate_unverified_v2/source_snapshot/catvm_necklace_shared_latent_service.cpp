#define NECKLACE_SHARED_LATENT_ENTRY shared_latent_direct_predecessor_main
#include "four_rotor_necklace_shared_latent_phase.cpp"
#undef NECKLACE_SHARED_LATENT_ENTRY

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

constexpr std::uint32_t kLatentProtocolMagic = 0x43564c50U;
constexpr std::uint64_t kLeaseTag = 0x4c454153454c4154ULL;
constexpr std::uint64_t kStageTag = 0x53544147454c4154ULL;

enum LatentCommand : std::uint32_t {
    kLatentInitialize = 1,
    kLatentBegin = 2,
    kLatentProject = 3,
    kLatentContinue = 4,
    kLatentReuse = 5,
    kLatentMissingInverse = 6,
    kLatentReorderedInverse = 7,
    kLatentWrongSemantic = 8,
    kLatentWrongType = 9,
    kLatentNullCarrier = 10,
    kLatentSnapshot = 11,
    kLatentStop = 12,
};

enum LatentStatus : std::uint32_t {
    kLatentOk = 0,
    kLatentDenied = 1,
    kLatentError = 2,
};

enum LatentFlags : std::uint32_t {
    kLatentBoundaryValid = 1U,
    kLatentRestored = 2U,
    kLatentStageResident = 4U,
    kLatentReuseFlag = 8U,
};

#pragma pack(push, 1)
struct LatentRequest {
    std::uint32_t magic;
    std::uint32_t command;
    std::uint32_t generation;
    std::uint32_t reserved;
    std::uint64_t lease;
    std::uint64_t nonce;
};

struct LatentResponse {
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

static_assert(sizeof(LatentRequest) == 32U);
static_assert(sizeof(LatentResponse) == 116U);

struct LatentServiceState {
    Plan plan = compile_plan();
    std::vector<Complex> baseline = make_latent_carrier(plan, 0);
    std::vector<Complex> carrier = baseline;
    const Complex *backing = carrier.data();
    std::vector<LatentModule> program = shared_primary_program();
    bool initialized = false;
    bool staged = false;
    bool poisoned = false;
    bool null_mode = false;
    std::size_t applied = 0;
    std::uint32_t generation = 0;
    std::uint64_t lease = 0;
    std::uint64_t last_nonce = 0;
};

LatentResponse latent_base(
    const LatentRequest &request,
    const LatentServiceState &state
) {
    LatentResponse response{};
    response.magic = kLatentProtocolMagic;
    response.status = kLatentOk;
    response.command = request.command;
    response.generation = state.generation;
    response.lease = state.lease;
    return response;
}

LatentResponse latent_denied(
    const LatentRequest &request,
    const LatentServiceState &state
) {
    LatentResponse response = latent_base(request, state);
    response.status = kLatentDenied;
    response.receipt = request.nonce ^ 0x44454e494544ULL;
    if (state.staged) {
        response.flags |= kLatentStageResident;
    }
    return response;
}

void latent_copy_boundary(
    LatentResponse &response,
    const Boundary &boundary
) {
    for (std::size_t index = 0; index < boundary.size(); ++index) {
        response.boundary[index] = boundary[index];
    }
    response.flags |= kLatentBoundaryValid;
}

bool latent_exact_io(
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

LatentResponse latent_initialize(
    const LatentRequest &request,
    LatentServiceState &state
) {
    LatentResponse response = latent_base(request, state);
    if (
        state.initialized
        || request.generation != 0U
        || request.lease != 0U
    ) {
        response.status = kLatentError;
        return response;
    }
    state.initialized = true;
    state.lease = request.nonce ^ kLeaseTag;
    response = latent_base(request, state);
    response.receipt = state.lease;
    return response;
}

bool latent_owner_matches(
    const LatentRequest &request,
    const LatentServiceState &state
) {
    return
        state.initialized
        && request.lease == state.lease
        && request.generation == state.generation
        && request.nonce > state.last_nonce;
}

LatentResponse latent_begin(
    const LatentRequest &request,
    LatentServiceState &state
) {
    LatentResponse response = latent_base(request, state);
    if (
        state.null_mode
        || state.staged
        || state.poisoned
        || state.generation != 0U
    ) {
        response.status = kLatentError;
        return response;
    }
    LatentStats stats;
    while (state.applied < 2U) {
        latent_forward_module(
            state.carrier,
            state.plan,
            state.program[state.applied],
            stats
        );
        ++state.applied;
    }
    state.staged = true;
    response = latent_base(request, state);
    response.flags |= kLatentStageResident;
    response.receipt =
        request.nonce ^ state.lease ^ kStageTag;
    response.native_operations =
        stats.generator.streamed_generator_terms;
    return response;
}

LatentResponse latent_continue(
    const LatentRequest &request,
    LatentServiceState &state
) {
    LatentResponse response = latent_base(request, state);
    if (
        !state.staged
        || state.applied != 2U
        || state.poisoned
    ) {
        response.status = kLatentError;
        return response;
    }
    LatentStats stats;
    while (state.applied < state.program.size()) {
        latent_forward_module(
            state.carrier,
            state.plan,
            state.program[state.applied],
            stats
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
        latent_inverse_module(
            state.carrier,
            state.plan,
            state.program[state.applied],
            stats
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
        response.status = kLatentError;
        return response;
    }
    ++state.generation;
    response = latent_base(request, state);
    latent_copy_boundary(response, boundary);
    response.flags |= kLatentRestored;
    response.restoration_error = restoration_error;
    response.norm_error = norm_error;
    response.native_operations =
        stats.generator.streamed_generator_terms;
    return response;
}

LatentResponse latent_reuse(
    const LatentRequest &request,
    LatentServiceState &state
) {
    LatentResponse response = latent_base(request, state);
    if (
        state.null_mode
        || state.staged
        || state.poisoned
        || state.generation != 1U
    ) {
        response.status = kLatentError;
        return response;
    }
    const std::vector<LatentModule> program =
        shared_reuse_program();
    const LatentRun actual = latent_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        program,
        LatentControl::Correct
    );
    std::vector<Complex> fresh = state.baseline;
    const LatentRun reference = latent_transaction(
        fresh,
        state.baseline,
        state.plan,
        program,
        LatentControl::Correct
    );
    const double boundary_error = boundary_distance(
        actual.boundary, reference.boundary
    );
    if (
        actual.restoration_error > kLatentTolerance
        || reference.restoration_error > kLatentTolerance
        || boundary_error > kLatentTolerance
        || actual.stats.generator.streamed_generator_terms
            != reference.stats.generator.streamed_generator_terms
        || state.carrier.data() != state.backing
    ) {
        state.poisoned = true;
        response.status = kLatentError;
        return response;
    }
    ++state.generation;
    response = latent_base(request, state);
    latent_copy_boundary(response, actual.boundary);
    response.flags |= kLatentRestored | kLatentReuseFlag;
    response.restoration_error = actual.restoration_error;
    response.norm_error = boundary_error;
    response.native_operations =
        actual.stats.generator.streamed_generator_terms;
    return response;
}

LatentResponse latent_inverse_control(
    const LatentRequest &request,
    LatentServiceState &state,
    LatentControl control
) {
    LatentResponse response = latent_base(request, state);
    if (
        state.null_mode
        || state.staged
        || state.poisoned
        || state.generation != 0U
    ) {
        response.status = kLatentError;
        return response;
    }
    const std::vector<LatentModule> controls = {
        state.program[0], state.program[1]
    };
    const LatentRun run = latent_transaction(
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

LatentResponse latent_dispatch(
    const LatentRequest &request,
    LatentServiceState &state
) {
    if (request.magic != kLatentProtocolMagic) {
        LatentResponse response = latent_base(request, state);
        response.status = kLatentError;
        return response;
    }
    if (request.command == kLatentInitialize) {
        const LatentResponse response =
            latent_initialize(request, state);
        state.last_nonce = request.nonce;
        return response;
    }
    if (!latent_owner_matches(request, state)) {
        return latent_denied(request, state);
    }
    state.last_nonce = request.nonce;
    if (state.poisoned && request.command != kLatentStop) {
        return latent_denied(request, state);
    }
    switch (request.command) {
        case kLatentBegin:
            return latent_begin(request, state);
        case kLatentProject:
            return latent_denied(request, state);
        case kLatentContinue:
            return latent_continue(request, state);
        case kLatentReuse:
            return latent_reuse(request, state);
        case kLatentMissingInverse:
            return latent_inverse_control(
                request, state, LatentControl::Missing
            );
        case kLatentReorderedInverse:
            return latent_inverse_control(
                request, state, LatentControl::ReorderedInverse
            );
        case kLatentWrongSemantic:
            return latent_inverse_control(
                request, state, LatentControl::WrongSemantic
            );
        case kLatentWrongType: {
            LatentModule invalid = state.program[0];
            invalid.axis = static_cast<LatentAxis>(99U);
            if (valid_latent_module(invalid)) {
                LatentResponse response = latent_base(request, state);
                response.status = kLatentError;
                return response;
            }
            return latent_denied(request, state);
        }
        case kLatentNullCarrier:
        case kLatentSnapshot:
            return latent_denied(request, state);
        case kLatentStop:
            return latent_base(request, state);
        default:
            return latent_denied(request, state);
    }
}

bool latent_disconnect_cleanup(LatentServiceState &state) {
    if (!state.staged) {
        return true;
    }
    LatentStats cleanup_stats;
    while (state.applied > 0U) {
        --state.applied;
        latent_inverse_module(
            state.carrier,
            state.plan,
            state.program[state.applied],
            cleanup_stats
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
    const std::vector<LatentModule> reuse_program =
        shared_reuse_program();
    const LatentRun actual = latent_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        reuse_program,
        LatentControl::Correct
    );
    std::vector<Complex> fresh = state.baseline;
    const LatentRun reference = latent_transaction(
        fresh,
        state.baseline,
        state.plan,
        reuse_program,
        LatentControl::Correct
    );
    return
        actual.restoration_error <= kLatentTolerance
        && reference.restoration_error <= kLatentTolerance
        && boundary_distance(
            actual.boundary, reference.boundary
        ) <= kLatentTolerance
        && actual.stats.generator.streamed_generator_terms
            == reference.stats.generator.streamed_generator_terms
        && state.carrier.data() == state.backing;
}

}  // namespace

#ifndef CATVM_NECKLACE_SHARED_LATENT_ENTRY
#define CATVM_NECKLACE_SHARED_LATENT_ENTRY main
#endif

int CATVM_NECKLACE_SHARED_LATENT_ENTRY(int argc, char **argv) {
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

    LatentServiceState state;
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
        LatentRequest request{};
        if (!latent_exact_io(
                client, &request, sizeof(request), false
            )) {
            io_failed = true;
            break;
        }
        const LatentResponse response =
            latent_dispatch(request, state);
        if (!latent_exact_io(
                client,
                const_cast<LatentResponse *>(&response),
                sizeof(response),
                true
            )) {
            io_failed = true;
            break;
        }
        stopped = request.command == kLatentStop;
    }
    const bool cleanup_ok = latent_disconnect_cleanup(state);
    ::close(client);
    ::close(listener);
    if (!cleanup_ok || (io_failed && state.poisoned)) {
        return 2;
    }
    return stopped || io_failed ? 0 : 2;
}
