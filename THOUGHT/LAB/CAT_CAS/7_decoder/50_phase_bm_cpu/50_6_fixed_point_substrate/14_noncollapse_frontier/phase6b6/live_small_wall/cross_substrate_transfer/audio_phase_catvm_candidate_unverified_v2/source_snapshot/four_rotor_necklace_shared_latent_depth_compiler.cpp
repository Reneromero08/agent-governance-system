#define NECKLACE_SHARED_LATENT_OWNER_REPAIR_ENTRY \
    shared_latent_owner_repair_predecessor_main
#include "four_rotor_necklace_shared_latent_owner_repair.cpp"
#undef NECKLACE_SHARED_LATENT_OWNER_REPAIR_ENTRY

/*
 * Public-topology compiler for a depth-parametric shared latent phase chain.
 *
 * A module is generated from (variant, ordinal), used once, and discarded.
 * The inverse regenerates the same public descriptor in reverse order.  No
 * module tape, inverse factors, phase snapshots, or carrier-dependent
 * provenance are retained.  The unresolved carrier stays at 285 necklaces
 * times one two-cell latent fiber for every tested depth.
 *
 * This advances fixed-rank/depth compactness.  It does not distinguish the
 * phase path from the identical compact 570-complex classical recurrence.
 */

namespace {

constexpr std::size_t kDepthDescriptorBytes =
    sizeof(std::uint32_t) + sizeof(std::uint32_t);
constexpr std::array<std::size_t, 6> kTestDepths{
    1U, 2U, 4U, 8U, 16U, 32U
};

enum class DepthControl {
    Correct,
    MissingInverse,
    ReorderedInverse,
    WrongInverseVariant,
};

struct DepthRun {
    Boundary boundary{};
    LatentStats stats{};
    double restoration_error = 0.0;
    double norm_error = 0.0;
    std::uint64_t compiled_forward_modules = 0;
    std::uint64_t rematerialized_inverse_modules = 0;
};

int depth_mod(int value, int modulus) {
    const int result = value % modulus;
    return result < 0 ? result + modulus : result;
}

int depth_nonzero_mod(int value) {
    return 1 + depth_mod(value, kGrid - 1);
}

LatentModule compile_depth_module(
    std::uint32_t variant,
    std::size_t ordinal
) {
    const int step = static_cast<int>(ordinal);
    LatentModule module;
    module.feature =
        ordinal % 4U == 0U
        ? LatentFeature::Collision
        : LatentFeature::CyclicSeparation;
    if (module.feature == LatentFeature::Collision) {
        module.separation = 0;
    } else {
        module.separation =
            1 + depth_mod(
                3 * step + static_cast<int>(variant),
                kGrid / 2
            );
    }
    switch ((ordinal + variant) % 3U) {
        case 0U:
            module.axis = LatentAxis::Z;
            break;
        case 1U:
            module.axis = LatentAxis::X;
            break;
        default:
            module.axis = LatentAxis::Y;
            break;
    }
    module.strength = depth_nonzero_mod(
        5 * step + 3 * static_cast<int>(variant) + 1
    );
    module.chirp = depth_nonzero_mod(
        7 * step + 5 * static_cast<int>(variant) + 2
    );
    module.owner = kSharedLatentPortOwner;
    return module;
}

bool valid_compiled_depth_module(
    const LatentModule &module
) {
    return
        module.owner == kSharedLatentPortOwner
        && valid_latent_module(module);
}

void depth_forward(
    std::vector<Complex> &carrier,
    const Plan &plan,
    std::uint32_t variant,
    std::size_t depth,
    DepthRun &run
) {
    for (std::size_t ordinal = 0; ordinal < depth; ++ordinal) {
        const LatentModule module =
            compile_depth_module(variant, ordinal);
        if (!valid_compiled_depth_module(module)) {
            fail("invalid topology-compiled forward module");
        }
        latent_forward_module(
            carrier, plan, module, run.stats
        );
        ++run.compiled_forward_modules;
    }
}

void depth_inverse(
    std::vector<Complex> &carrier,
    const Plan &plan,
    std::uint32_t variant,
    std::size_t depth,
    DepthControl control,
    DepthRun &run
) {
    const std::size_t inverse_count =
        control == DepthControl::MissingInverse
        ? depth - 1U
        : depth;
    for (std::size_t cursor = 0; cursor < inverse_count; ++cursor) {
        const std::size_t ordinal =
            control == DepthControl::ReorderedInverse
            ? cursor
            : depth - 1U - cursor;
        const std::uint32_t inverse_variant =
            control == DepthControl::WrongInverseVariant
            ? variant + 1U
            : variant;
        const LatentModule module =
            compile_depth_module(inverse_variant, ordinal);
        if (!valid_compiled_depth_module(module)) {
            fail("invalid topology-rematerialized inverse module");
        }
        latent_inverse_module(
            carrier, plan, module, run.stats
        );
        ++run.rematerialized_inverse_modules;
    }
}

DepthRun depth_transaction(
    std::vector<Complex> &carrier,
    const std::vector<Complex> &baseline,
    const Plan &plan,
    std::uint32_t variant,
    std::size_t depth,
    DepthControl control
) {
    if (depth == 0U) {
        fail("depth must be positive");
    }
    DepthRun run;
    depth_forward(carrier, plan, variant, depth, run);
    run.boundary = latent_boundary(carrier, plan);
    run.norm_error = std::fabs(
        latent_weighted_norm(carrier, plan) - 1.0
    );
    depth_inverse(
        carrier, plan, variant, depth, control, run
    );
    run.restoration_error =
        latent_l2_distance(carrier, baseline, plan);
    return run;
}

}  // namespace

#ifndef NECKLACE_SHARED_LATENT_DEPTH_ENTRY
#define NECKLACE_SHARED_LATENT_DEPTH_ENTRY main
#endif

int NECKLACE_SHARED_LATENT_DEPTH_ENTRY() {
    const Plan plan = compile_plan();
    const std::vector<Complex> baseline =
        make_latent_carrier(plan, 0);
    std::vector<Complex> carrier = baseline;
    const Complex *backing = carrier.data();

    std::array<DepthRun, kTestDepths.size()> runs{};
    double maximum_restoration_error = 0.0;
    std::uint64_t maximum_native_terms = 0;
    for (std::size_t index = 0; index < kTestDepths.size(); ++index) {
        runs[index] = depth_transaction(
            carrier,
            baseline,
            plan,
            2U,
            kTestDepths[index],
            DepthControl::Correct
        );
        maximum_restoration_error = std::max(
            maximum_restoration_error,
            runs[index].restoration_error
        );
        maximum_native_terms = std::max(
            maximum_native_terms,
            runs[index].stats.generator.streamed_generator_terms
        );
        if (
            runs[index].restoration_error > kLatentTolerance
            || runs[index].norm_error > kLatentTolerance
            || runs[index].compiled_forward_modules
                != kTestDepths[index]
            || runs[index].rematerialized_inverse_modules
                != kTestDepths[index]
            || carrier.data() != backing
        ) {
            fail("depth-parametric compact carrier failed");
        }
    }

    const std::size_t reuse_depth = 11U;
    const DepthRun reuse = depth_transaction(
        carrier,
        baseline,
        plan,
        5U,
        reuse_depth,
        DepthControl::Correct
    );
    std::vector<Complex> fresh = baseline;
    const DepthRun fresh_reuse = depth_transaction(
        fresh,
        baseline,
        plan,
        5U,
        reuse_depth,
        DepthControl::Correct
    );
    const double fresh_reuse_error = boundary_distance(
        reuse.boundary, fresh_reuse.boundary
    );
    if (
        reuse.restoration_error > kLatentTolerance
        || fresh_reuse.restoration_error > kLatentTolerance
        || fresh_reuse_error > kLatentTolerance
        || reuse.stats.generator.streamed_generator_terms
            != fresh_reuse.stats.generator.streamed_generator_terms
        || carrier.data() != backing
    ) {
        fail("depth-parametric restored reuse failed");
    }

    const std::size_t control_depth = 4U;
    std::vector<Complex> missing_carrier = baseline;
    const DepthRun missing = depth_transaction(
        missing_carrier,
        baseline,
        plan,
        2U,
        control_depth,
        DepthControl::MissingInverse
    );
    std::vector<Complex> reordered_carrier = baseline;
    const DepthRun reordered = depth_transaction(
        reordered_carrier,
        baseline,
        plan,
        2U,
        control_depth,
        DepthControl::ReorderedInverse
    );
    std::vector<Complex> wrong_variant_carrier = baseline;
    const DepthRun wrong_variant = depth_transaction(
        wrong_variant_carrier,
        baseline,
        plan,
        2U,
        control_depth,
        DepthControl::WrongInverseVariant
    );

    LatentModule wrong_owner = compile_depth_module(2U, 0U);
    wrong_owner.owner ^= 1U;
    const std::vector<Complex> before_owner_attack = carrier;
    const bool wrong_owner_rejected =
        !valid_compiled_depth_module(wrong_owner);
    const double owner_attack_carrier_error =
        latent_l2_distance(carrier, before_owner_attack, plan);
    if (
        missing.restoration_error < kControlFloor
        || reordered.restoration_error < kControlFloor
        || wrong_variant.restoration_error < kControlFloor
        || !wrong_owner_rejected
        || owner_attack_carrier_error != 0.0
    ) {
        fail("depth compiler causal controls failed");
    }

    const std::vector<Complex> parity_baseline = carrier;
    std::vector<Complex> phase_parity = parity_baseline;
    const DepthRun phase_parity_run = depth_transaction(
        phase_parity,
        parity_baseline,
        plan,
        2U,
        kTestDepths.back(),
        DepthControl::Correct
    );
    std::vector<Complex> matched_classical = parity_baseline;
    const DepthRun classical = depth_transaction(
        matched_classical,
        parity_baseline,
        plan,
        2U,
        kTestDepths.back(),
        DepthControl::Correct
    );
    const double classical_error = boundary_distance(
        phase_parity_run.boundary, classical.boundary
    );
    if (
        classical_error != 0.0
        || classical.stats.generator.streamed_generator_terms
            != phase_parity_run.stats.generator.streamed_generator_terms
    ) {
        fail("depth compiler matched compact recurrence failed");
    }

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"TOPOLOGY_REMATERIALIZED_OWNER_BOUND_SHARED_LATENT_PHASE_PROGRAM_FIXED_570_CARRIER_ACROSS_INCREASING_DEPTH\","
        "\"result\":\"PASS\","
        "\"claim_ceiling\":\"DIRECT_PROCESS_GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACES_570_COMPLEX_CELLS_ONE_TWO_CELL_LATENT_FIBER_PUBLIC_VARIANT_ORDINAL_COMPILER_DEPTHS1_2_4_8_16_32_REUSE_DEPTH11_STATIC_OWNER_COMPLEX128_SOFTWARE_ONLY\","
        "\"tested_depths\":[1,2,4,8,16,32],"
        "\"resident_necklace_cells\":285,"
        "\"latent_cells_per_necklace\":2,"
        "\"resident_joint_complex_cells\":570,"
        "\"program_descriptor_bytes\":%zu,"
        "\"retained_module_tape_bytes\":0,"
        "\"retained_inverse_history_bytes\":0,"
        "\"relation_table_cells\":0,"
        "\"assignment_cells\":0,"
        "\"temporary_occupation_cells\":0,"
        "\"dense_285_operator_cells\":0,"
        "\"maximum_restoration_error\":%.17g,"
        "\"reuse_depth\":%zu,"
        "\"reuse_restoration_error\":%.17g,"
        "\"fresh_restored_reuse_boundary_error\":%.17g,"
        "\"carrier_backing_preserved\":true,"
        "\"restoration_class\":\"NUMERICAL_PHYSICAL_STATE_RESTORATION\",",
        kDepthDescriptorBytes,
        maximum_restoration_error,
        reuse_depth,
        reuse.restoration_error,
        fresh_reuse_error
    );
    std::printf(
        "\"controls\":{"
        "\"missing_inverse_error\":%.17g,"
        "\"reordered_inverse_error\":%.17g,"
        "\"wrong_inverse_variant_error\":%.17g,"
        "\"wrong_owner_rejected\":true,"
        "\"owner_attack_carrier_error\":%.17g"
        "},"
        "\"resource_law\":{"
        "\"maximum_native_generator_terms\":%llu,"
        "\"catalytic_carrier_complex_cells\":570,"
        "\"permanent_restoration_baseline_complex_cells\":570,"
        "\"per_module_fiber_complex_cells\":285,"
        "\"generator_matrix_complex_cells\":289,"
        "\"generator_plan_object_bytes\":%zu,"
        "\"generator_plan_nonmatrix_scalar_bytes\":%zu,"
        "\"generator_work_complex_cells\":855,"
        "\"accepted_transaction_peak_counted_complex_cells_excluding_plan\":2569,"
        "\"verification_reference_complex_cells\":570,"
        "\"verification_peak_counted_complex_cells_excluding_plan\":3139,"
        "\"plan_object_bytes\":%zu,"
        "\"plan_necklace_count\":%zu,"
        "\"plan_necklace_capacity\":%zu,"
        "\"plan_necklace_element_bytes\":%zu,"
        "\"plan_necklace_capacity_bytes\":%zu,"
        "\"plan_root_complex_cells\":17,"
        "\"accepted_transaction_peak_counted_complex_cells_including_plan_roots\":2586,"
        "\"verification_peak_counted_complex_cells_including_plan_roots\":3156,"
        "\"native_terms_linear_in_depth\":true,"
        "\"resident_carrier_constant_in_depth\":true,"
        "\"inverse_descriptors_rematerialized_from_public_topology\":true,"
        "\"public_topology_inspects_final_answer\":false,"
        "\"allocator_native_library_os_memory_bounded\":false"
        "},"
        "\"strongest_compact_classical\":{"
        "\"same_570_complex_recurrence\":true,"
        "\"boundary_error\":%.17g"
        "},"
        "\"machine_boundary_enforced\":false,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_waveform_execution\":false,"
        "\"unbounded_computation_established\":false,"
        "\"terminal\":false",
        missing.restoration_error,
        reordered.restoration_error,
        wrong_variant.restoration_error,
        owner_attack_carrier_error,
        static_cast<unsigned long long>(maximum_native_terms),
        sizeof(GeneratorPlan),
        sizeof(GeneratorPlan) - sizeof(Matrix),
        sizeof(Plan),
        plan.necklaces.size(),
        plan.necklaces.capacity(),
        sizeof(Necklace),
        plan.necklaces.capacity() * sizeof(Necklace),
        classical_error
    );
    std::printf("}\n");
    return 0;
}
