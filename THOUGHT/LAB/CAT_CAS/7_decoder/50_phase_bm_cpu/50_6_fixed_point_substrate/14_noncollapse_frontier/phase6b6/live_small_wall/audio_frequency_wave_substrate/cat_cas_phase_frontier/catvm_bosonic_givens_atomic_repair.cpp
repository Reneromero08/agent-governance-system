#define BOSONIC_GIVENS_ENTRY bosonic_givens_atomic_repair_predecessor_main
#include "four_rotor_bosonic_givens_phase.cpp"
#undef BOSONIC_GIVENS_ENTRY

#define main catvm_bosonic_givens_rejected_predecessor_service_main
#include "catvm_bosonic_givens_service_tail.inc"
#undef main

#include <cstdlib>

/*
 * Distinct successor repair for the staged bosonic Givens CATVM service.
 *
 * The frozen predecessor remains the authority for its successful-session
 * evidence.  This service adds the missing transaction laws:
 *
 * - launch-time mode separation;
 * - monotone request nonces on one explicit connection lease;
 * - no content-derived carrier or occupation hash on the wire;
 * - inverse cleanup and an internal restored-carrier reuse sentinel when the
 *   client disconnects while the hidden occupation state is resident;
 * - negative inverse controls acting on the actual service carrier; and
 * - poisoning after a deliberately failed inverse control.
 *
 * The numerical restoration class remains
 * NUMERICAL_PHYSICAL_STATE_RESTORATION.
 */

namespace {

enum class RepairMode {
    Direct,
    Snapshot,
    InPlace,
    NullCarrier,
};

struct RepairContext {
    RepairMode mode = RepairMode::InPlace;
    std::uint64_t last_nonce = 0;
    bool poisoned = false;
};

RepairMode parse_mode(const std::string &value) {
    if (value == "direct") {
        return RepairMode::Direct;
    }
    if (value == "snapshot") {
        return RepairMode::Snapshot;
    }
    if (value == "in-place") {
        return RepairMode::InPlace;
    }
    if (value == "null") {
        return RepairMode::NullCarrier;
    }
    std::exit(2);
}

Response repair_response(
    const Request &request,
    const ServiceState &state
) {
    Response response{};
    response.magic = kProtocolMagic;
    response.status = kStatusOk;
    response.command = request.command;
    response.generation = state.generation;
    response.state_hash =
        (static_cast<std::uint64_t>(state.generation) << 32U)
        | (state.staged ? 1ULL : 0ULL);
    return response;
}

Response repair_denied(
    const Request &request,
    const ServiceState &state
) {
    Response response = repair_response(request, state);
    response.status = kStatusDenied;
    response.receipt = request.nonce ^ 0x44454e494544ULL;
    return response;
}

void remove_content_derived_receipts(
    Response &response,
    const Request &request,
    const ServiceState &state
) {
    response.state_hash =
        (static_cast<std::uint64_t>(state.generation) << 32U)
        | (state.staged ? 1ULL : 0ULL);
    if (
        request.command == kBeginPrimary
        && response.status == kStatusOk
    ) {
        response.receipt =
            request.nonce ^ 0x53544147454c4541ULL;
    }
}

bool mode_allows(RepairMode mode, std::uint32_t command) {
    if (command == kInitialize || command == kStop) {
        return true;
    }
    switch (mode) {
        case RepairMode::Direct:
            return command == kDirectBegin
                || command == kDirectContinue;
        case RepairMode::Snapshot:
            return command == kSnapshotBegin
                || command == kSnapshotContinue;
        case RepairMode::InPlace:
            return command == kBeginPrimary
                || command == kProjectIntermediate
                || command == kContinuePrimary
                || command == kReuse
                || command == kMissingInverse
                || command == kWrongInverse
                || command == kReorderedInverse
                || command == kNullCarrier;
        case RepairMode::NullCarrier:
            return command == kNullCarrier;
    }
    return false;
}

Response resident_inverse_control(
    const Request &request,
    ServiceState &state,
    RepairContext &context,
    Control control_mode
) {
    Response response = repair_response(request, state);
    if (
        !state.initialized
        || state.staged
        || state.generation != 0U
        || context.poisoned
    ) {
        response.status = kStatusError;
        return response;
    }
    const FastRun run = fast_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        2,
        0,
        control_mode
    );
    response.restoration_error = run.restoration_error;
    response.norm_error = run.norm_error;
    response.native_operations =
        run.fast_stats.polynomial_block_terms;
    context.poisoned = true;
    return response;
}

Response repaired_reuse(
    const Request &request,
    ServiceState &state
) {
    Response response = repair_response(request, state);
    if (
        !state.initialized
        || state.staged
        || state.generation != 1U
    ) {
        response.status = kStatusError;
        return response;
    }
    const FastRun actual = fast_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        2,
        3,
        Control::Correct
    );
    std::vector<Complex> fresh_carrier = state.baseline;
    const FastRun fresh = fast_transaction(
        fresh_carrier,
        state.baseline,
        state.plan,
        2,
        3,
        Control::Correct
    );
    const double boundary_error =
        boundary_distance(actual.boundary, fresh.boundary);
    if (
        actual.restoration_error > kRestorationTolerance
        || fresh.restoration_error > kRestorationTolerance
        || boundary_error > kClosureTolerance
        || actual.fast_stats.polynomial_block_terms
            != fresh.fast_stats.polynomial_block_terms
        || state.carrier.data() != state.carrier_backing
    ) {
        response.status = kStatusError;
        return response;
    }
    ++state.generation;
    response = repair_response(request, state);
    copy_boundary(response, actual.boundary);
    response.flags |= kRestored | kReuseFlag;
    response.restoration_error = actual.restoration_error;
    response.norm_error = boundary_error;
    response.native_operations =
        actual.fast_stats.polynomial_block_terms;
    return response;
}

Response repaired_dispatch(
    const Request &request,
    ServiceState &state,
    RepairContext &context
) {
    if (request.magic != kProtocolMagic) {
        Response response = repair_response(request, state);
        response.status = kStatusError;
        return response;
    }
    if (
        request.command != kStop
        && request.nonce <= context.last_nonce
    ) {
        return repair_denied(request, state);
    }
    context.last_nonce = request.nonce;
    if (!mode_allows(context.mode, request.command)) {
        return repair_denied(request, state);
    }
    if (context.poisoned && request.command != kStop) {
        return repair_denied(request, state);
    }
    if (
        context.mode == RepairMode::NullCarrier
        && request.command == kNullCarrier
    ) {
        return repair_denied(request, state);
    }
    if (
        context.mode == RepairMode::InPlace
        && request.command == kNullCarrier
    ) {
        return repair_denied(request, state);
    }
    if (
        context.mode == RepairMode::InPlace
        && request.command == kBeginPrimary
        && state.generation != 0U
    ) {
        Response response = repair_response(request, state);
        response.status = kStatusError;
        return response;
    }
    Response response{};
    switch (request.command) {
        case kMissingInverse:
            response = resident_inverse_control(
                request, state, context, Control::Missing
            );
            break;
        case kWrongInverse:
            response = resident_inverse_control(
                request, state, context, Control::Wrong
            );
            break;
        case kReorderedInverse:
            response = resident_inverse_control(
                request, state, context, Control::Reordered
            );
            break;
        case kReuse:
            response = repaired_reuse(request, state);
            break;
        default:
            response = dispatch(request, state);
            break;
    }
    remove_content_derived_receipts(response, request, state);
    return response;
}

bool discard_hidden_stage_and_restore_carrier(ServiceState &state) {
    if (!state.staged) {
        return true;
    }
    state.resident_occupation.clear();
    state.resident_occupation.shrink_to_fit();
    state.resident_free_plan = {};
    state.staged = false;
    Stats cleanup_stats;
    apply_collision(
        state.carrier,
        state.plan,
        public_kappa(0, 0),
        true,
        cleanup_stats
    );
    const double restoration_error =
        l2_distance(state.carrier, state.baseline, state.plan);
    if (
        restoration_error > kRestorationTolerance
        || state.carrier.data() != state.carrier_backing
    ) {
        return false;
    }

    const FastRun actual = fast_transaction(
        state.carrier,
        state.baseline,
        state.plan,
        2,
        3,
        Control::Correct
    );
    std::vector<Complex> fresh_carrier = state.baseline;
    const FastRun fresh = fast_transaction(
        fresh_carrier,
        state.baseline,
        state.plan,
        2,
        3,
        Control::Correct
    );
    return
        actual.restoration_error <= kRestorationTolerance
        && fresh.restoration_error <= kRestorationTolerance
        && boundary_distance(actual.boundary, fresh.boundary)
            <= kClosureTolerance
        && actual.fast_stats.polynomial_block_terms
            == fresh.fast_stats.polynomial_block_terms
        && state.carrier.data() == state.carrier_backing;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        return 2;
    }
    if (::prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) {
        return 2;
    }
    RepairContext context;
    context.mode = parse_mode(argv[1]);
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

    ServiceState state;
    if (context.mode == RepairMode::NullCarrier) {
        state.carrier.clear();
        state.carrier.shrink_to_fit();
        state.baseline.clear();
        state.baseline.shrink_to_fit();
        state.carrier_backing = nullptr;
    }

    bool stopped = false;
    bool io_failed = false;
    while (!stopped) {
        Request request{};
        if (!exact_io(client, &request, sizeof(request), false)) {
            io_failed = true;
            break;
        }
        const Response response =
            repaired_dispatch(request, state, context);
        if (!exact_io(
                client,
                const_cast<Response *>(&response),
                sizeof(response),
                true
            )) {
            io_failed = true;
            break;
        }
        stopped = request.command == kStop;
    }

    const bool cleanup_ok =
        discard_hidden_stage_and_restore_carrier(state);
    ::close(client);
    ::close(listener);
    if (!cleanup_ok) {
        return 2;
    }
    if (io_failed && context.poisoned) {
        return 2;
    }
    return stopped || io_failed ? 0 : 2;
}
