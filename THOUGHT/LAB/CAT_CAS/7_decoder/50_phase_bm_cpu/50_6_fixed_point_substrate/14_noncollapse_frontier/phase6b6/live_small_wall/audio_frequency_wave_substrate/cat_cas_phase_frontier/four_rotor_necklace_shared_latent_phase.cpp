#define NECKLACE_GENERATOR_ENTRY shared_latent_generator_predecessor_main
#include "four_rotor_necklace_generator_phase.cpp"
#undef NECKLACE_GENERATOR_ENTRY

#ifndef NECKLACE_SHARED_LATENT_ENTRY
#define NECKLACE_SHARED_LATENT_ENTRY main
#endif

/*
 * A genuinely shared unresolved observation port on the compact necklace
 * carrier.
 *
 * Every necklace amplitude carries the same two-cell coherent latent fiber.
 * Typed feature-controlled SU(2) couplings consume that one resident port.
 * Z-, X-, and Y-axis couplings are noncommuting, and Hermitian necklace
 * evolution is interleaved between them.  The latent port is never replaced
 * by public scalar substitution and is never projected.  Only the final
 * seven-bin collision boundary is closed after summing the latent fiber norm.
 *
 * This is a fixed two-cell fiber, not an observation-domain relation table or
 * an assignment list.  The strongest compact classical implementation is the
 * identical 570-complex recurrence, so the result does not establish a
 * distinct phase resource or advantage.
 */

namespace {

constexpr std::size_t kLatentCells = 2U;
constexpr double kLatentTolerance = 6.0e-11;

enum class LatentFeature : std::uint32_t {
    Collision = 1,
    CyclicSeparation = 2,
};

enum class LatentAxis : std::uint32_t {
    X = 1,
    Y = 2,
    Z = 3,
};

struct LatentModule {
    LatentFeature feature = LatentFeature::Collision;
    LatentAxis axis = LatentAxis::Z;
    int separation = 0;
    int strength = 0;
    int chirp = 1;
    std::uint32_t owner = 0;
};

struct LatentStats {
    std::uint64_t coupling_updates = 0;
    std::uint64_t shared_port_consumptions = 0;
    std::uint64_t generator_fiber_copies = 0;
    std::uint64_t relation_table_cells = 0;
    std::uint64_t assignment_cells = 0;
    GeneratorStats generator{};
};

enum class LatentControl {
    Correct,
    Missing,
    WrongSemantic,
    ReorderedInverse,
};

struct LatentRun {
    Boundary boundary{};
    LatentStats stats{};
    double restoration_error = 0.0;
    double norm_error = 0.0;
};

bool valid_latent_module(const LatentModule &module) {
    const bool valid_axis =
        module.axis == LatentAxis::X
        || module.axis == LatentAxis::Y
        || module.axis == LatentAxis::Z;
    if (
        !valid_axis
        || module.strength < 1
        || module.strength >= kGrid
        || module.chirp < 1
        || module.chirp >= kGrid
        || module.owner == 0U
    ) {
        return false;
    }
    if (module.feature == LatentFeature::Collision) {
        return module.separation == 0;
    }
    return
        module.feature == LatentFeature::CyclicSeparation
        && module.separation >= 1
        && module.separation <= kGrid / 2;
}

int latent_feature(
    const Necklace &necklace,
    const LatentModule &module
) {
    if (module.feature == LatentFeature::Collision) {
        return necklace.collisions;
    }
    int result = 0;
    for (int mode = 0; mode < kGrid; ++mode) {
        result += static_cast<int>(necklace.histogram[mode])
            * static_cast<int>(
                necklace.histogram[mod(mode + module.separation)]
            );
    }
    return result;
}

std::vector<Complex> make_latent_carrier(
    const Plan &plan,
    int identity
) {
    const std::vector<Complex> necklace = make_carrier(plan, identity);
    std::vector<Complex> result(
        necklace.size() * kLatentCells,
        Complex(0.0, 0.0)
    );
    const double scale = 1.0 / std::sqrt(2.0);
    const Complex second = std::polar(scale, 2.0 * kPi / 17.0);
    for (std::size_t index = 0; index < necklace.size(); ++index) {
        result[2U * index] = scale * necklace[index];
        result[2U * index + 1U] = second * necklace[index];
    }
    return result;
}

double latent_weighted_norm(
    const std::vector<Complex> &carrier,
    const Plan &plan
) {
    double result = 0.0;
    for (std::size_t index = 0; index < plan.necklaces.size(); ++index) {
        result += static_cast<double>(
            plan.necklaces[index].labelled_weight
        ) * (
            std::norm(carrier[2U * index])
            + std::norm(carrier[2U * index + 1U])
        );
    }
    return result;
}

double latent_l2_distance(
    const std::vector<Complex> &left,
    const std::vector<Complex> &right,
    const Plan &plan
) {
    double result = 0.0;
    for (std::size_t index = 0; index < plan.necklaces.size(); ++index) {
        result += static_cast<double>(
            plan.necklaces[index].labelled_weight
        ) * (
            std::norm(left[2U * index] - right[2U * index])
            + std::norm(
                left[2U * index + 1U] - right[2U * index + 1U]
            )
        );
    }
    return std::sqrt(result);
}

Boundary latent_boundary(
    const std::vector<Complex> &carrier,
    const Plan &plan
) {
    Boundary result{};
    for (std::size_t index = 0; index < plan.necklaces.size(); ++index) {
        result[plan.necklaces[index].collisions] +=
            static_cast<double>(
                plan.necklaces[index].labelled_weight
            ) * (
                std::norm(carrier[2U * index])
                + std::norm(carrier[2U * index + 1U])
            );
    }
    return result;
}

void apply_latent_coupling(
    std::vector<Complex> &carrier,
    const Plan &plan,
    const LatentModule &module,
    bool adjoint,
    LatentStats &stats
) {
    if (!valid_latent_module(module)) {
        fail("invalid typed shared latent module");
    }
    const double sign = adjoint ? -1.0 : 1.0;
    for (std::size_t index = 0; index < plan.necklaces.size(); ++index) {
        const double angle = sign * 2.0 * kPi
            * static_cast<double>(
                mod(
                    module.strength
                    * latent_feature(plan.necklaces[index], module)
                )
            ) / static_cast<double>(kGrid);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const Complex upper = carrier[2U * index];
        const Complex lower = carrier[2U * index + 1U];
        if (module.axis == LatentAxis::Z) {
            carrier[2U * index] =
                std::polar(1.0, angle) * upper;
            carrier[2U * index + 1U] =
                std::polar(1.0, -angle) * lower;
        } else if (module.axis == LatentAxis::X) {
            carrier[2U * index] =
                cosine * upper + Complex(0.0, sine) * lower;
            carrier[2U * index + 1U] =
                Complex(0.0, sine) * upper + cosine * lower;
        } else {
            carrier[2U * index] =
                cosine * upper + sine * lower;
            carrier[2U * index + 1U] =
                -sine * upper + cosine * lower;
        }
        stats.coupling_updates += 2U;
    }
    ++stats.shared_port_consumptions;
}

void apply_latent_generator(
    std::vector<Complex> &carrier,
    const Plan &plan,
    int chirp,
    bool adjoint,
    LatentStats &stats
) {
    for (std::size_t latent = 0; latent < kLatentCells; ++latent) {
        std::vector<Complex> fiber(plan.necklaces.size());
        for (std::size_t index = 0;
             index < plan.necklaces.size();
             ++index) {
            fiber[index] = carrier[2U * index + latent];
            ++stats.generator_fiber_copies;
        }
        generator_free(
            fiber,
            plan,
            chirp,
            adjoint,
            stats.generator
        );
        for (std::size_t index = 0;
             index < plan.necklaces.size();
             ++index) {
            carrier[2U * index + latent] = fiber[index];
            ++stats.generator_fiber_copies;
        }
    }
}

void latent_forward_module(
    std::vector<Complex> &carrier,
    const Plan &plan,
    const LatentModule &module,
    LatentStats &stats
) {
    apply_latent_coupling(
        carrier, plan, module, false, stats
    );
    apply_latent_generator(
        carrier, plan, module.chirp, false, stats
    );
}

void latent_inverse_module(
    std::vector<Complex> &carrier,
    const Plan &plan,
    const LatentModule &module,
    LatentStats &stats
) {
    apply_latent_generator(
        carrier, plan, module.chirp, true, stats
    );
    apply_latent_coupling(
        carrier, plan, module, true, stats
    );
}

std::vector<LatentModule> shared_primary_program() {
    return {
        {
            LatentFeature::Collision,
            LatentAxis::Z,
            0,
            3,
            1,
            0x4c415431U,
        },
        {
            LatentFeature::CyclicSeparation,
            LatentAxis::X,
            1,
            5,
            4,
            0x4c415431U,
        },
        {
            LatentFeature::CyclicSeparation,
            LatentAxis::Y,
            3,
            7,
            6,
            0x4c415431U,
        },
        {
            LatentFeature::Collision,
            LatentAxis::X,
            0,
            11,
            9,
            0x4c415431U,
        },
    };
}

std::vector<LatentModule> shared_reuse_program() {
    return {
        {
            LatentFeature::CyclicSeparation,
            LatentAxis::Y,
            2,
            4,
            3,
            0x4c415431U,
        },
        {
            LatentFeature::Collision,
            LatentAxis::Z,
            0,
            9,
            7,
            0x4c415431U,
        },
        {
            LatentFeature::CyclicSeparation,
            LatentAxis::X,
            7,
            14,
            13,
            0x4c415431U,
        },
    };
}

LatentRun latent_transaction(
    std::vector<Complex> &carrier,
    const std::vector<Complex> &baseline,
    const Plan &plan,
    const std::vector<LatentModule> &program,
    LatentControl control
) {
    LatentRun run;
    for (const LatentModule &module : program) {
        latent_forward_module(carrier, plan, module, run.stats);
    }
    run.boundary = latent_boundary(carrier, plan);
    run.norm_error =
        std::fabs(latent_weighted_norm(carrier, plan) - 1.0);

    const std::size_t minimum =
        control == LatentControl::Missing ? 1U : 0U;
    for (std::size_t cursor = program.size();
         cursor > minimum;
         --cursor) {
        LatentModule module = program[cursor - 1U];
        if (
            control == LatentControl::WrongSemantic
            && cursor == program.size()
        ) {
            module.strength = mod(module.strength + 1);
            if (module.strength == 0) {
                module.strength = 1;
            }
        }
        if (control == LatentControl::ReorderedInverse) {
            apply_latent_coupling(
                carrier, plan, module, true, run.stats
            );
            apply_latent_generator(
                carrier,
                plan,
                module.chirp,
                true,
                run.stats
            );
        } else {
            latent_inverse_module(
                carrier, plan, module, run.stats
            );
        }
    }
    run.restoration_error =
        latent_l2_distance(carrier, baseline, plan);
    return run;
}

Boundary undermerged_boundary(
    const std::vector<Complex> &baseline,
    const Plan &plan,
    const std::vector<LatentModule> &program
) {
    Boundary result{};
    for (const LatentModule &module : program) {
        std::vector<Complex> isolated = baseline;
        const LatentRun run = latent_transaction(
            isolated,
            baseline,
            plan,
            {module},
            LatentControl::Correct
        );
        if (run.restoration_error > kLatentTolerance) {
            fail("undermerge control failed restoration");
        }
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] +=
                run.boundary[index]
                / static_cast<double>(program.size());
        }
    }
    return result;
}

}  // namespace

int NECKLACE_SHARED_LATENT_ENTRY() {
    const Plan plan = compile_plan();
    const std::vector<Complex> baseline =
        make_latent_carrier(plan, 0);
    const std::vector<LatentModule> program =
        shared_primary_program();

    std::vector<Complex> carrier = baseline;
    const Complex *backing = carrier.data();
    const LatentRun primary = latent_transaction(
        carrier,
        baseline,
        plan,
        program,
        LatentControl::Correct
    );
    if (
        primary.restoration_error > kLatentTolerance
        || primary.norm_error > kLatentTolerance
        || carrier.data() != backing
        || primary.stats.relation_table_cells != 0U
        || primary.stats.assignment_cells != 0U
        || primary.stats.shared_port_consumptions
            != 2U * program.size()
    ) {
        fail("shared latent primary failed");
    }

    const std::vector<LatentModule> reuse_program =
        shared_reuse_program();
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
    const double reuse_error = boundary_distance(
        reuse.boundary, fresh_reuse.boundary
    );
    if (
        reuse.restoration_error > kLatentTolerance
        || fresh_reuse.restoration_error > kLatentTolerance
        || reuse_error > kLatentTolerance
        || carrier.data() != backing
        || reuse.stats.generator.streamed_generator_terms
            != fresh_reuse.stats.generator.streamed_generator_terms
    ) {
        fail("shared latent reuse failed");
    }

    std::vector<LatentModule> overmerged = program;
    overmerged[2] = overmerged[1];
    std::vector<Complex> overmerged_carrier = baseline;
    const LatentRun overmerged_run = latent_transaction(
        overmerged_carrier,
        baseline,
        plan,
        overmerged,
        LatentControl::Correct
    );
    const double overmerge_effect = boundary_distance(
        primary.boundary, overmerged_run.boundary
    );
    const Boundary undermerged = undermerged_boundary(
        baseline, plan, program
    );
    const double undermerge_effect = boundary_distance(
        primary.boundary, undermerged
    );

    std::vector<LatentModule> swapped = program;
    std::swap(swapped[0], swapped[1]);
    std::vector<Complex> swapped_carrier = baseline;
    const LatentRun swapped_run = latent_transaction(
        swapped_carrier,
        baseline,
        plan,
        swapped,
        LatentControl::Correct
    );
    const double module_order_effect = boundary_distance(
        primary.boundary, swapped_run.boundary
    );

    std::vector<LatentModule> perturbed = program;
    ++perturbed[3].strength;
    std::vector<Complex> perturbed_carrier = baseline;
    const LatentRun perturbed_run = latent_transaction(
        perturbed_carrier,
        baseline,
        plan,
        perturbed,
        LatentControl::Correct
    );
    const double semantic_effect = boundary_distance(
        primary.boundary, perturbed_run.boundary
    );

    std::vector<Complex> missing_carrier = baseline;
    const LatentRun missing = latent_transaction(
        missing_carrier,
        baseline,
        plan,
        {program[0], program[1]},
        LatentControl::Missing
    );
    std::vector<Complex> reordered_carrier = baseline;
    const LatentRun reordered = latent_transaction(
        reordered_carrier,
        baseline,
        plan,
        {program[0], program[1]},
        LatentControl::ReorderedInverse
    );
    std::vector<Complex> wrong_carrier = baseline;
    const LatentRun wrong = latent_transaction(
        wrong_carrier,
        baseline,
        plan,
        {program[0], program[1]},
        LatentControl::WrongSemantic
    );

    LatentModule invalid = program[0];
    invalid.axis = static_cast<LatentAxis>(99U);
    const bool wrong_type_rejected = !valid_latent_module(invalid);
    if (
        overmerge_effect < 1.0e-5
        || undermerge_effect < 1.0e-5
        || module_order_effect < 1.0e-5
        || semantic_effect < 1.0e-5
        || missing.restoration_error < kControlFloor
        || reordered.restoration_error < kControlFloor
        || wrong.restoration_error < kControlFloor
        || !wrong_type_rejected
    ) {
        fail("shared latent causal controls failed");
    }

    std::vector<Complex> matched_classical = baseline;
    const LatentRun classical = latent_transaction(
        matched_classical,
        baseline,
        plan,
        program,
        LatentControl::Correct
    );
    const double classical_error = boundary_distance(
        primary.boundary, classical.boundary
    );
    if (classical_error != 0.0) {
        fail("shared latent compact classical parity failed");
    }

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"COHERENT_SHARED_LATENT_OBSERVATION_PORT_PHASE_CONTRACTION_ON_NECKLACE_CARRIER\","
        "\"result\":\"PASS\","
        "\"claim_ceiling\":\"DIRECT_PROCESS_GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACES_TWO_CELL_COHERENT_LATENT_FIBER_FOUR_TYPED_MODULES_COMPLEX128_SOFTWARE_ONLY\","
        "\"resident_necklace_cells\":285,"
        "\"latent_cells_per_necklace\":2,"
        "\"resident_joint_complex_cells\":570,"
        "\"shared_latent_port_count\":1,"
        "\"shared_latent_consumers\":4,"
        "\"noncommuting_axes\":[\"Z\",\"X\",\"Y\",\"X\"],"
        "\"public_scalar_observation_substitution\":false,"
        "\"latent_port_projected\":false,"
        "\"relation_table_cells\":0,"
        "\"assignment_cells\":0,"
    );
    std::printf(
        "\"boundary\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"
        "\"primary_restoration_error\":%.17g,"
        "\"reuse_restoration_error\":%.17g,"
        "\"fresh_restored_reuse_boundary_error\":%.17g,"
        "\"carrier_backing_preserved\":true,"
        "\"restoration_generation\":2,"
        "\"restoration_class\":\"NUMERICAL_PHYSICAL_STATE_RESTORATION\",",
        primary.boundary[0],
        primary.boundary[1],
        primary.boundary[2],
        primary.boundary[3],
        primary.boundary[4],
        primary.boundary[5],
        primary.boundary[6],
        primary.restoration_error,
        reuse.restoration_error,
        reuse_error
    );
    std::printf(
        "\"controls\":{"
        "\"overmerge_boundary_effect\":%.17g,"
        "\"undermerge_boundary_effect\":%.17g,"
        "\"module_order_boundary_effect\":%.17g,"
        "\"semantic_perturbation_boundary_effect\":%.17g,"
        "\"missing_inverse_error\":%.17g,"
        "\"reordered_inverse_error\":%.17g,"
        "\"wrong_semantic_inverse_error\":%.17g,"
        "\"wrong_type_rejected\":true},"
        "\"resource_law\":{"
        "\"generator_carrier_sized_work_vectors_per_fiber\":3,"
        "\"temporary_occupation_cells\":0,"
        "\"dense_285_operator_cells\":0,"
        "\"retained_inverse_history_bytes\":0,"
        "\"allocator_native_library_os_memory_bounded\":false},",
        overmerge_effect,
        undermerge_effect,
        module_order_effect,
        semantic_effect,
        missing.restoration_error,
        reordered.restoration_error,
        wrong.restoration_error
    );
    std::printf(
        "\"strongest_compact_classical\":{"
        "\"same_570_complex_recurrence\":true,"
        "\"boundary_error\":%.17g},"
        "\"machine_boundary_enforced\":false,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_waveform_execution\":false,"
        "\"general_catalytic_inference_established\":false,"
        "\"terminal\":false",
        classical_error
    );
    std::printf("}\n");
    return 0;
}
