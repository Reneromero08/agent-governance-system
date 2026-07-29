#define CATVM_NECKLACE_SHARED_LATENT_DEPTH_ENTRY \
    catvm_necklace_shared_latent_depth_rejected_source_main
#include "catvm_necklace_shared_latent_depth_service.cpp"
#undef CATVM_NECKLACE_SHARED_LATENT_DEPTH_ENTRY

namespace {

/*
 * Distinct successor to the rejected source above.  The carrier mechanism is
 * unchanged.  This dispatch boundary repairs two independently reproduced
 * custody defects:
 *
 * 1. INITIALIZE may establish the lease exactly once and a rejected repeat
 *    cannot rewind the accepted nonce.
 * 2. Every response emitted while a hidden stage remains resident carries
 *    STAGE_RESIDENT and no boundary.  STOP is denied while resident so its
 *    acknowledgement cannot precede inverse cleanup.
 */
DepthResponse repaired_depth_dispatch(
    const DepthRequest &request,
    DepthServiceState &state
) {
    if (request.magic != kDepthProtocolMagic) {
        DepthResponse response = depth_base(request, state);
        response.status = kDepthError;
        if (state.staged) {
            response.flags |= kDepthStageResident;
        }
        return response;
    }

    if (request.command == kDepthInitialize) {
        if (state.initialized) {
            return depth_denied(request, state);
        }
        const DepthResponse response =
            depth_initialize(request, state);
        if (response.status == kDepthOk) {
            state.last_nonce = request.nonce;
        }
        return response;
    }

    if (state.staged && request.command == kDepthStop) {
        return depth_denied(request, state);
    }

    DepthResponse response = depth_dispatch(request, state);
    if (state.staged) {
        response.flags |= kDepthStageResident;
        response.flags &= ~kDepthBoundaryValid;
        for (double &value : response.boundary) {
            value = 0.0;
        }
    }
    return response;
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
            repaired_depth_dispatch(request, state);
        if (!depth_exact_io(
                client,
                const_cast<DepthResponse *>(&response),
                sizeof(response),
                true
            )) {
            io_failed = true;
            break;
        }
        stopped =
            request.command == kDepthStop
            && response.status == kDepthOk;
    }
    const bool cleanup_ok = depth_disconnect_cleanup(state);
    ::close(client);
    ::close(listener);
    if (!cleanup_ok || (io_failed && state.poisoned)) {
        return 2;
    }
    return stopped || io_failed ? 0 : 2;
}
