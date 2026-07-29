#define NECKLACE_SHARED_LATENT_ENTRY shared_latent_owner_defect_main
#include "four_rotor_necklace_shared_latent_phase.cpp"
#undef NECKLACE_SHARED_LATENT_ENTRY

/*
 * Distinct successor repair for the shared-latent module-owner defect.
 *
 * The predecessor accepted any nonzero LatentModule::owner.  This repair
 * binds every consumer in both public programs to the one declared latent
 * port owner before any carrier operation.  CATVM separately binds each
 * command to the exact transaction lease and restoration generation.
 */

namespace {

constexpr std::uint32_t kSharedLatentPortOwner = 0x4c415431U;

bool latent_program_owner_matches(
    const std::vector<LatentModule> &program,
    std::uint32_t expected_owner
) {
    if (expected_owner == 0U || program.empty()) {
        return false;
    }
    for (const LatentModule &module : program) {
        if (
            !valid_latent_module(module)
            || module.owner != expected_owner
        ) {
            return false;
        }
    }
    return true;
}

bool all_shared_latent_program_owners_match() {
    return
        latent_program_owner_matches(
            shared_primary_program(), kSharedLatentPortOwner
        )
        && latent_program_owner_matches(
            shared_reuse_program(), kSharedLatentPortOwner
        );
}

}  // namespace

#ifndef NECKLACE_SHARED_LATENT_OWNER_REPAIR_ENTRY
#define NECKLACE_SHARED_LATENT_OWNER_REPAIR_ENTRY main
#endif

int NECKLACE_SHARED_LATENT_OWNER_REPAIR_ENTRY() {
    const Plan plan = compile_plan();
    const std::vector<Complex> baseline =
        make_latent_carrier(plan, 0);
    const std::vector<LatentModule> primary_program =
        shared_primary_program();
    const std::vector<LatentModule> reuse_program =
        shared_reuse_program();
    if (!all_shared_latent_program_owners_match()) {
        fail("declared shared latent program owner mismatch");
    }

    std::vector<Complex> carrier = baseline;
    const Complex *backing = carrier.data();
    const LatentRun primary = latent_transaction(
        carrier,
        baseline,
        plan,
        primary_program,
        LatentControl::Correct
    );
    const LatentRun reuse = latent_transaction(
        carrier,
        baseline,
        plan,
        reuse_program,
        LatentControl::Correct
    );
    std::vector<Complex> fresh = baseline;
    const LatentRun fresh_reuse = latent_transaction(
        fresh,
        baseline,
        plan,
        reuse_program,
        LatentControl::Correct
    );
    const double reuse_boundary_error = boundary_distance(
        reuse.boundary, fresh_reuse.boundary
    );
    if (
        primary.restoration_error > kLatentTolerance
        || reuse.restoration_error > kLatentTolerance
        || fresh_reuse.restoration_error > kLatentTolerance
        || reuse_boundary_error > kLatentTolerance
        || carrier.data() != backing
    ) {
        fail("owner-bound shared latent transaction failed");
    }

    std::vector<LatentModule> wrong_owner_program =
        primary_program;
    wrong_owner_program[0].owner = kSharedLatentPortOwner ^ 1U;
    const std::vector<Complex> before_attack = carrier;
    const bool wrong_nonzero_owner_rejected =
        !latent_program_owner_matches(
            wrong_owner_program, kSharedLatentPortOwner
        );
    const double rejected_attack_carrier_error =
        latent_l2_distance(carrier, before_attack, plan);
    if (
        !wrong_nonzero_owner_rejected
        || rejected_attack_carrier_error != 0.0
    ) {
        fail("wrong nonzero module owner was not rejected atomically");
    }

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"OWNER_BOUND_COHERENT_SHARED_LATENT_OBSERVATION_PORT_PHASE_CONTRACTION_ON_NECKLACE_CARRIER\","
        "\"result\":\"PASS\","
        "\"claim_ceiling\":\"DIRECT_PROCESS_GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACES_TWO_CELL_COHERENT_LATENT_FIBER_FOUR_MODULES_EXACT_STATIC_PORT_OWNER_COMPLEX128_SOFTWARE_ONLY\","
        "\"declared_port_owner\":%u,"
        "\"primary_module_owners_checked\":%zu,"
        "\"reuse_module_owners_checked\":%zu,"
        "\"wrong_nonzero_module_owner_rejected\":true,"
        "\"rejected_attack_carrier_error\":%.17g,"
        "\"primary_restoration_error\":%.17g,"
        "\"reuse_restoration_error\":%.17g,"
        "\"fresh_restored_reuse_boundary_error\":%.17g,"
        "\"carrier_backing_preserved\":true,"
        "\"restoration_class\":\"NUMERICAL_PHYSICAL_STATE_RESTORATION\","
        "\"predecessor_source_defect_preserved\":true,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_waveform_execution\":false,"
        "\"terminal\":false",
        kSharedLatentPortOwner,
        primary_program.size(),
        reuse_program.size(),
        rejected_attack_carrier_error,
        primary.restoration_error,
        reuse.restoration_error,
        reuse_boundary_error
    );
    std::printf("}\n");
    return 0;
}
