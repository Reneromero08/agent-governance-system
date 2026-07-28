#include <cerrno>
#include <csignal>
#include <cstring>
#include <string>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

constexpr std::uint32_t kProtocolMagic = 0x43564247U;

enum Command : std::uint32_t {
    kInitialize = 1,
    kDirectBegin = 2,
    kSnapshotBegin = 3,
    kBeginPrimary = 4,
    kProjectIntermediate = 5,
    kContinuePrimary = 6,
    kReuse = 7,
    kMissingInverse = 8,
    kWrongInverse = 9,
    kReorderedInverse = 10,
    kNullCarrier = 11,
    kStop = 12,
    kDirectContinue = 13,
    kSnapshotContinue = 14,
};

enum Status : std::uint32_t {
    kStatusOk = 0,
    kStatusDenied = 1,
    kStatusError = 2,
};

enum Flags : std::uint32_t {
    kBoundaryValid = 1U,
    kRestored = 2U,
    kStageResident = 4U,
    kSnapshotReload = 8U,
    kReuseFlag = 16U,
};

#pragma pack(push, 1)
struct Request {
    std::uint32_t magic;
    std::uint32_t command;
    std::uint64_t nonce;
};

struct Response {
    std::uint32_t magic;
    std::uint32_t status;
    std::uint32_t command;
    std::uint32_t generation;
    std::uint32_t flags;
    std::uint64_t receipt;
    std::uint64_t state_hash;
    double boundary[kMaximumCollision + 1];
    double restoration_error;
    double norm_error;
    std::uint64_t native_operations;
    std::uint64_t snapshot_reload_bytes;
};
#pragma pack(pop)

static_assert(sizeof(Request) == 16U);
static_assert(sizeof(Response) == 124U);

std::uint64_t hash_bytes(
    std::uint64_t value,
    const void *data,
    std::size_t size
) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    for (std::size_t index = 0; index < size; ++index) {
        value ^= bytes[index];
        value *= 1099511628211ULL;
    }
    return value;
}

std::uint64_t carrier_hash(
    const std::vector<Complex> &carrier,
    bool staged
) {
    std::uint64_t result = 1469598103934665603ULL;
    result = hash_bytes(
        result,
        carrier.data(),
        carrier.size() * sizeof(Complex)
    );
    const unsigned char stage = staged ? 1U : 0U;
    return hash_bytes(result, &stage, sizeof(stage));
}

std::uint64_t custody_receipt(
    const std::vector<Complex> &polynomial,
    std::uint64_t nonce
) {
    std::uint64_t result = 1469598103934665603ULL;
    result = hash_bytes(result, &nonce, sizeof(nonce));
    return hash_bytes(
        result,
        polynomial.data(),
        polynomial.size() * sizeof(Complex)
    );
}

void copy_boundary(
    Response &response,
    const Boundary &boundary
) {
    for (std::size_t index = 0; index < boundary.size(); ++index) {
        response.boundary[index] = boundary[index];
    }
    response.flags |= kBoundaryValid;
}

struct ServiceState {
    Plan plan = compile_plan();
    std::vector<Complex> baseline = make_carrier(plan, 0);
    std::vector<Complex> carrier = baseline;
    const Complex *carrier_backing = carrier.data();
    std::vector<Complex> resident_occupation;
    FreePlan resident_free_plan;
    Stats stage_stats;
    FastStats stage_fast_stats;
    bool initialized = false;
    bool staged = false;
    std::uint32_t generation = 0;
};

Response base_response(
    const Request &request,
    const ServiceState &state
) {
    Response response{};
    response.magic = kProtocolMagic;
    response.status = kStatusOk;
    response.command = request.command;
    response.generation = state.generation;
    response.state_hash = carrier_hash(state.carrier, state.staged);
    return response;
}

Response initialize(
    const Request &request,
    ServiceState &state
) {
    if (state.initialized || state.staged) {
        Response response = base_response(request, state);
        response.status = kStatusError;
        return response;
    }
    state.initialized = true;
    return base_response(request, state);
}

Response direct_baseline(
    const Request &request,
    ServiceState &state
) {
    Response response = base_response(request, state);
    if (!state.initialized || state.staged) {
        response.status = kStatusError;
        return response;
    }
    std::vector<Complex> direct = state.baseline;
    Stats stats;
    FastStats fast_stats;
    for (int step = 0; step < kPrimaryDepth; ++step) {
        fast_forward_step(
            direct, state.plan, step, 0, stats, fast_stats
        );
    }
    copy_boundary(response, project_boundary(direct, state.plan));
    response.norm_error =
        std::fabs(weighted_norm(direct, state.plan) - 1.0);
    response.native_operations =
        fast_stats.polynomial_block_terms;
    return response;
}

Response sham_begin(
    const Request &request,
    ServiceState &state
) {
    Response response = base_response(request, state);
    if (!state.initialized || state.staged) {
        response.status = kStatusError;
        return response;
    }
    response.receipt = request.nonce ^ 0x5348414d42454749ULL;
    return response;
}

Response snapshot_sham(
    const Request &request,
    ServiceState &state
) {
    Response response = base_response(request, state);
    if (!state.initialized || state.staged) {
        response.status = kStatusError;
        return response;
    }
    std::vector<Complex> snapshot = state.baseline;
    std::vector<Complex> sham = snapshot;
    Stats stats;
    FastStats fast_stats;
    for (int step = 0; step < kPrimaryDepth; ++step) {
        fast_forward_step(
            sham, state.plan, step, 0, stats, fast_stats
        );
    }
    copy_boundary(response, project_boundary(sham, state.plan));
    sham = snapshot;
    response.restoration_error =
        l2_distance(sham, state.baseline, state.plan);
    response.norm_error =
        std::fabs(weighted_norm(sham, state.plan) - 1.0);
    response.flags |= kRestored | kSnapshotReload;
    response.native_operations =
        fast_stats.polynomial_block_terms;
    response.snapshot_reload_bytes =
        snapshot.size() * sizeof(Complex);
    return response;
}

Response begin_primary(
    const Request &request,
    ServiceState &state
) {
    Response response = base_response(request, state);
    if (!state.initialized || state.staged) {
        response.status = kStatusError;
        return response;
    }
    state.stage_stats = {};
    state.stage_fast_stats = {};
    apply_collision(
        state.carrier,
        state.plan,
        public_kappa(0, 0),
        false,
        state.stage_stats
    );
    state.resident_free_plan = compile_free_plan(
        state.plan, public_chirp(0, 0)
    );
    state.resident_occupation.assign(
        kHistogramDimension, Complex(0.0, 0.0)
    );
    expand_necklaces(
        state.carrier,
        state.plan,
        state.resident_occupation,
        state.stage_fast_stats
    );
    apply_diagonal_polynomial(
        state.resident_occupation,
        state.resident_free_plan.diagonal,
        false
    );
    auto item = state.resident_free_plan.givens.rbegin();
    for (std::size_t index = 0; index < 68U; ++index, ++item) {
        apply_two_mode_polynomial(
            state.resident_occupation,
            *item,
            true,
            state.stage_fast_stats
        );
    }
    state.staged = true;
    response = base_response(request, state);
    response.flags |= kStageResident;
    response.receipt = custody_receipt(
        state.resident_occupation, request.nonce
    );
    response.native_operations =
        state.stage_fast_stats.polynomial_block_terms;
    return response;
}

Response project_intermediate(
    const Request &request,
    ServiceState &state
) {
    Response response = base_response(request, state);
    response.status = kStatusDenied;
    response.flags = state.staged ? kStageResident : 0U;
    response.receipt = request.nonce ^ 0x44454e494544ULL;
    return response;
}

Response continue_primary(
    const Request &request,
    ServiceState &state
) {
    Response response = base_response(request, state);
    if (!state.initialized || !state.staged) {
        response.status = kStatusError;
        return response;
    }
    auto item = state.resident_free_plan.givens.rbegin();
    std::advance(item, 68);
    for (std::size_t index = 68U; index < 136U; ++index, ++item) {
        apply_two_mode_polynomial(
            state.resident_occupation,
            *item,
            true,
            state.stage_fast_stats
        );
    }
    close_necklaces(
        state.resident_occupation,
        state.plan,
        state.carrier,
        state.stage_fast_stats
    );
    state.resident_occupation.clear();
    state.resident_occupation.shrink_to_fit();
    state.resident_free_plan = {};
    state.staged = false;

    for (int step = 1; step < kPrimaryDepth; ++step) {
        fast_forward_step(
            state.carrier,
            state.plan,
            step,
            0,
            state.stage_stats,
            state.stage_fast_stats
        );
    }
    const Boundary boundary =
        project_boundary(state.carrier, state.plan);
    const double norm_error =
        std::fabs(weighted_norm(state.carrier, state.plan) - 1.0);
    for (int step = kPrimaryDepth - 1; step >= 0; --step) {
        fast_inverse_step(
            state.carrier,
            state.plan,
            step,
            0,
            state.stage_stats,
            state.stage_fast_stats
        );
    }
    const double restoration_error =
        l2_distance(state.carrier, state.baseline, state.plan);
    if (
        restoration_error > kRestorationTolerance
        || norm_error > kClosureTolerance
        || state.carrier.data() != state.carrier_backing
    ) {
        response.status = kStatusError;
        return response;
    }
    ++state.generation;
    response = base_response(request, state);
    copy_boundary(response, boundary);
    response.flags |= kRestored;
    response.restoration_error = restoration_error;
    response.norm_error = norm_error;
    response.native_operations =
        state.stage_fast_stats.polynomial_block_terms;
    response.receipt =
        request.nonce ^ 0x524553544f524544ULL;
    return response;
}

Response reuse(
    const Request &request,
    ServiceState &state
) {
    Response response = base_response(request, state);
    if (!state.initialized || state.staged || state.generation != 1U) {
        response.status = kStatusError;
        return response;
    }
    const FastRun run = fast_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        2,
        3,
        Control::Correct
    );
    if (
        run.restoration_error > kRestorationTolerance
        || state.carrier.data() != state.carrier_backing
    ) {
        response.status = kStatusError;
        return response;
    }
    ++state.generation;
    response = base_response(request, state);
    copy_boundary(response, run.boundary);
    response.flags |= kRestored | kReuseFlag;
    response.restoration_error = run.restoration_error;
    response.norm_error = run.norm_error;
    response.native_operations =
        run.fast_stats.polynomial_block_terms;
    return response;
}

Response control(
    const Request &request,
    ServiceState &state,
    Control control_mode
) {
    Response response = base_response(request, state);
    if (!state.initialized || state.staged) {
        response.status = kStatusError;
        return response;
    }
    std::vector<Complex> carrier = state.baseline;
    const FastRun run = fast_transaction(
        carrier, state.baseline, state.plan, 2, 0, control_mode
    );
    response.restoration_error = run.restoration_error;
    response.norm_error = run.norm_error;
    response.native_operations =
        run.fast_stats.polynomial_block_terms;
    return response;
}

Response denied(
    const Request &request,
    ServiceState &state
) {
    Response response = base_response(request, state);
    response.status = kStatusDenied;
    response.receipt = request.nonce ^ 0x44454e494544ULL;
    return response;
}

bool exact_io(
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

Response dispatch(
    const Request &request,
    ServiceState &state
) {
    if (request.magic != kProtocolMagic) {
        Response response = base_response(request, state);
        response.status = kStatusError;
        return response;
    }
    switch (request.command) {
        case kInitialize:
            return initialize(request, state);
        case kDirectBegin:
        case kSnapshotBegin:
            return sham_begin(request, state);
        case kBeginPrimary:
            return begin_primary(request, state);
        case kProjectIntermediate:
            return project_intermediate(request, state);
        case kContinuePrimary:
            return continue_primary(request, state);
        case kReuse:
            return reuse(request, state);
        case kMissingInverse:
            return control(request, state, Control::Missing);
        case kWrongInverse:
            return control(request, state, Control::Wrong);
        case kReorderedInverse:
            return control(request, state, Control::Reordered);
        case kNullCarrier:
            return denied(request, state);
        case kStop:
            return base_response(request, state);
        case kDirectContinue:
            return direct_baseline(request, state);
        case kSnapshotContinue:
            return snapshot_sham(request, state);
        default: {
            Response response = base_response(request, state);
            response.status = kStatusDenied;
            return response;
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    if (::prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) {
        return 2;
    }
    const std::string socket_path = argv[1];
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
    ::unlink(socket_path.c_str());
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
        ::unlink(socket_path.c_str());
        return 2;
    }
    const int client = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
    if (client < 0) {
        ::close(listener);
        ::unlink(socket_path.c_str());
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
        ::unlink(socket_path.c_str());
        return 2;
    }

    ServiceState state;
    bool running = true;
    while (running) {
        Request request{};
        if (!exact_io(client, &request, sizeof(request), false)) {
            break;
        }
        const Response response = dispatch(request, state);
        if (!exact_io(
                client,
                const_cast<Response *>(&response),
                sizeof(response),
                true
            )) {
            break;
        }
        running = request.command != kStop;
    }
    ::close(client);
    ::close(listener);
    ::unlink(socket_path.c_str());
    return running ? 2 : 0;
}
