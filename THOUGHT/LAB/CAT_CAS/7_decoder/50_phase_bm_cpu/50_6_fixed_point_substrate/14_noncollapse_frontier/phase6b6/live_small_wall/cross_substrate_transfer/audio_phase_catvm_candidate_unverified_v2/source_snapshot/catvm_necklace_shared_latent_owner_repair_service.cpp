#define CATVM_NECKLACE_SHARED_LATENT_ENTRY \
    shared_latent_owner_defect_service_main
#include "catvm_necklace_shared_latent_service.cpp"
#undef CATVM_NECKLACE_SHARED_LATENT_ENTRY

namespace {

constexpr std::uint32_t kWrongModuleOwnerCommand = 13U;
constexpr std::uint32_t kRepairSharedLatentPortOwner = 0x4c415431U;

bool repair_program_owner_matches(
    const std::vector<LatentModule> &program
) {
    if (program.empty()) {
        return false;
    }
    for (const LatentModule &module : program) {
        if (
            !valid_latent_module(module)
            || module.owner != kRepairSharedLatentPortOwner
        ) {
            return false;
        }
    }
    return true;
}

bool repair_service_program_owners_match() {
    return
        repair_program_owner_matches(shared_primary_program())
        && repair_program_owner_matches(shared_reuse_program());
}

LatentResponse repair_owner_dispatch(
    const LatentRequest &request,
    LatentServiceState &state
) {
    if (request.command == kWrongModuleOwnerCommand) {
        if (
            request.magic != kLatentProtocolMagic
            || !latent_owner_matches(request, state)
        ) {
            return latent_denied(request, state);
        }
        state.last_nonce = request.nonce;
        std::vector<LatentModule> attacked = state.program;
        attacked[0].owner = kRepairSharedLatentPortOwner ^ 1U;
        if (repair_program_owner_matches(attacked)) {
            LatentResponse response = latent_base(request, state);
            response.status = kLatentError;
            return response;
        }
        return latent_denied(request, state);
    }
    if (
        request.command != kLatentInitialize
        && !repair_service_program_owners_match()
    ) {
        LatentResponse response = latent_base(request, state);
        response.status = kLatentError;
        return response;
    }
    return latent_dispatch(request, state);
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

    LatentServiceState state;
    if (!repair_service_program_owners_match()) {
        ::close(client);
        ::close(listener);
        return 2;
    }
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
            repair_owner_dispatch(request, state);
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
