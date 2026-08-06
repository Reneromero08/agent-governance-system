#define NECKLACE_GENERATOR_ENTRY two_shared_latent_generator_predecessor_main
#include "four_rotor_necklace_generator_phase.cpp"
#undef NECKLACE_GENERATOR_ENTRY

#ifndef NECKLACE_TWO_SHARED_LATENT_ENTRY
#define NECKLACE_TWO_SHARED_LATENT_ENTRY main
#endif

/*
 * Two owner-bound unresolved latent ports on the compact necklace carrier.
 *
 * Each of the 285 public necklaces carries a four-cell tensor fiber indexed
 * by two binary latent coordinates.  Local modules consume exactly one typed
 * port.  A controlled-phase module consumes both ports and cannot be factored
 * into independent one-port updates on the declared product input.  Hermitian
 * necklace evolution is interleaved between typed modules.  Neither latent
 * coordinate is projected; only the final seven-bin collision boundary is
 * closed after summing the joint fiber norm.
 *
 * This remains a finite deterministic complex128 software recurrence.  Its
 * strongest compact classical implementation is the identical 1,140-complex
 * recurrence, so the experiment does not establish a distinct phase resource
 * or a computational advantage.
 */

namespace {

constexpr std::size_t kTwoPortCells = 4U;
constexpr double kTwoPortTolerance = 1.2e-10;
constexpr std::uint32_t kPortAOwner = 0x4c415441U;
constexpr std::uint32_t kPortBOwner = 0x4c415442U;

enum class TwoPortFeature : std::uint32_t {
    Collision = 1,
    CyclicSeparation = 2,
};

enum class TwoPortScope : std::uint32_t {
    PortA = 1,
    PortB = 2,
    Joint = 3,
};

enum class TwoPortAxis : std::uint32_t {
    X = 1,
    Y = 2,
    Z = 3,
    ControlledPhase = 4,
};

struct TwoPortModule {
    TwoPortFeature feature = TwoPortFeature::Collision;
    TwoPortScope scope = TwoPortScope::PortA;
    TwoPortAxis axis = TwoPortAxis::Z;
    int separation = 0;
    int strength = 0;
    int chirp = 1;
    std::uint32_t owner_a = 0;
    std::uint32_t owner_b = 0;
};

struct TwoPortStats {
    std::uint64_t coupling_updates = 0;
    std::uint64_t port_a_consumptions = 0;
    std::uint64_t port_b_consumptions = 0;
    std::uint64_t joint_consumptions = 0;
    std::uint64_t generator_fiber_copies = 0;
    std::uint64_t relation_table_cells = 0;
    std::uint64_t assignment_cells = 0;
    GeneratorStats generator{};
};

enum class TwoPortControl {
    Correct,
    Missing,
    WrongSemantic,
    ReorderedInverse,
};

enum class JointControlMode {
    Native,
    Identity,
    SeparableDiagonal,
};

struct TwoPortRun {
    Boundary boundary{};
    TwoPortStats stats{};
    double restoration_error = 0.0;
    double norm_error = 0.0;
};

bool valid_two_port_module(const TwoPortModule &module) {
    if (
        module.strength < 1
        || module.strength >= kGrid
        || module.chirp < 1
        || module.chirp >= kGrid
    ) {
        return false;
    }
    if (
        module.feature == TwoPortFeature::Collision
        && module.separation != 0
    ) {
        return false;
    }
    if (
        module.feature == TwoPortFeature::CyclicSeparation
        && (
            module.separation < 1
            || module.separation > kGrid / 2
        )
    ) {
        return false;
    }
    if (
        module.feature != TwoPortFeature::Collision
        && module.feature != TwoPortFeature::CyclicSeparation
    ) {
        return false;
    }
    if (module.scope == TwoPortScope::PortA) {
        return
            module.axis != TwoPortAxis::ControlledPhase
            && (
                module.axis == TwoPortAxis::X
                || module.axis == TwoPortAxis::Y
                || module.axis == TwoPortAxis::Z
            )
            && module.owner_a == kPortAOwner
            && module.owner_b == 0U;
    }
    if (module.scope == TwoPortScope::PortB) {
        return
            module.axis != TwoPortAxis::ControlledPhase
            && (
                module.axis == TwoPortAxis::X
                || module.axis == TwoPortAxis::Y
                || module.axis == TwoPortAxis::Z
            )
            && module.owner_a == 0U
            && module.owner_b == kPortBOwner;
    }
    return
        module.scope == TwoPortScope::Joint
        && module.axis == TwoPortAxis::ControlledPhase
        && module.owner_a == kPortAOwner
        && module.owner_b == kPortBOwner;
}

bool valid_two_port_program(
    const std::vector<TwoPortModule> &program
) {
    if (program.empty()) {
        return false;
    }
    for (const TwoPortModule &module : program) {
        if (!valid_two_port_module(module)) {
            return false;
        }
    }
    return true;
}

int two_port_feature(
    const Necklace &necklace,
    const TwoPortModule &module
) {
    if (module.feature == TwoPortFeature::Collision) {
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

std::vector<Complex> make_two_port_carrier(
    const Plan &plan,
    int identity
) {
    const std::vector<Complex> necklace = make_carrier(plan, identity);
    std::vector<Complex> result(
        necklace.size() * kTwoPortCells,
        Complex(0.0, 0.0)
    );
    const double scale = 1.0 / std::sqrt(2.0);
    const std::array<Complex, 2> port_a = {
        Complex(scale, 0.0),
        std::polar(scale, 2.0 * kPi / 17.0),
    };
    const std::array<Complex, 2> port_b = {
        Complex(scale, 0.0),
        std::polar(scale, 4.0 * kPi / 17.0),
    };
    for (std::size_t index = 0; index < necklace.size(); ++index) {
        for (std::size_t a = 0; a < 2U; ++a) {
            for (std::size_t b = 0; b < 2U; ++b) {
                result[4U * index + 2U * a + b] =
                    necklace[index] * port_a[a] * port_b[b];
            }
        }
    }
    return result;
}

double two_port_weighted_norm(
    const std::vector<Complex> &carrier,
    const Plan &plan
) {
    double result = 0.0;
    for (std::size_t index = 0; index < plan.necklaces.size(); ++index) {
        double fiber_norm = 0.0;
        for (std::size_t cell = 0; cell < kTwoPortCells; ++cell) {
            fiber_norm += std::norm(
                carrier[4U * index + cell]
            );
        }
        result += static_cast<double>(
            plan.necklaces[index].labelled_weight
        ) * fiber_norm;
    }
    return result;
}

double two_port_l2_distance(
    const std::vector<Complex> &left,
    const std::vector<Complex> &right,
    const Plan &plan
) {
    double result = 0.0;
    for (std::size_t index = 0; index < plan.necklaces.size(); ++index) {
        double fiber_error = 0.0;
        for (std::size_t cell = 0; cell < kTwoPortCells; ++cell) {
            fiber_error += std::norm(
                left[4U * index + cell]
                - right[4U * index + cell]
            );
        }
        result += static_cast<double>(
            plan.necklaces[index].labelled_weight
        ) * fiber_error;
    }
    return std::sqrt(result);
}

Boundary two_port_boundary(
    const std::vector<Complex> &carrier,
    const Plan &plan
) {
    Boundary result{};
    for (std::size_t index = 0; index < plan.necklaces.size(); ++index) {
        double fiber_norm = 0.0;
        for (std::size_t cell = 0; cell < kTwoPortCells; ++cell) {
            fiber_norm += std::norm(
                carrier[4U * index + cell]
            );
        }
        result[plan.necklaces[index].collisions] +=
            static_cast<double>(
                plan.necklaces[index].labelled_weight
            ) * fiber_norm;
    }
    return result;
}

double maximum_fiber_determinant(
    const std::vector<Complex> &carrier
) {
    double result = 0.0;
    for (std::size_t offset = 0;
         offset < carrier.size();
         offset += kTwoPortCells) {
        const Complex determinant =
            carrier[offset] * carrier[offset + 3U]
            - carrier[offset + 1U] * carrier[offset + 2U];
        result = std::max(result, std::abs(determinant));
    }
    return result;
}

void apply_pair_rotation(
    Complex &upper,
    Complex &lower,
    TwoPortAxis axis,
    double angle
) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const Complex old_upper = upper;
    const Complex old_lower = lower;
    if (axis == TwoPortAxis::Z) {
        upper = std::polar(1.0, angle) * old_upper;
        lower = std::polar(1.0, -angle) * old_lower;
    } else if (axis == TwoPortAxis::X) {
        upper =
            cosine * old_upper + Complex(0.0, sine) * old_lower;
        lower =
            Complex(0.0, sine) * old_upper + cosine * old_lower;
    } else if (axis == TwoPortAxis::Y) {
        upper = cosine * old_upper + sine * old_lower;
        lower = -sine * old_upper + cosine * old_lower;
    } else {
        fail("invalid local two-port axis");
    }
}

void apply_two_port_coupling(
    std::vector<Complex> &carrier,
    const Plan &plan,
    const TwoPortModule &module,
    bool adjoint,
    TwoPortStats &stats
) {
    if (!valid_two_port_module(module)) {
        fail("invalid typed two-port module");
    }
    const double sign = adjoint ? -1.0 : 1.0;
    for (std::size_t index = 0; index < plan.necklaces.size(); ++index) {
        const double angle = sign * 2.0 * kPi
            * static_cast<double>(
                mod(
                    module.strength
                    * two_port_feature(plan.necklaces[index], module)
                )
            ) / static_cast<double>(kGrid);
        const std::size_t offset = 4U * index;
        if (module.scope == TwoPortScope::PortA) {
            apply_pair_rotation(
                carrier[offset],
                carrier[offset + 2U],
                module.axis,
                angle
            );
            apply_pair_rotation(
                carrier[offset + 1U],
                carrier[offset + 3U],
                module.axis,
                angle
            );
            ++stats.port_a_consumptions;
        } else if (module.scope == TwoPortScope::PortB) {
            apply_pair_rotation(
                carrier[offset],
                carrier[offset + 1U],
                module.axis,
                angle
            );
            apply_pair_rotation(
                carrier[offset + 2U],
                carrier[offset + 3U],
                module.axis,
                angle
            );
            ++stats.port_b_consumptions;
        } else {
            carrier[offset + 3U] *= std::polar(1.0, angle);
            ++stats.joint_consumptions;
        }
        stats.coupling_updates += kTwoPortCells;
    }
}

void apply_separable_joint_coupling(
    std::vector<Complex> &carrier,
    const Plan &plan,
    const TwoPortModule &module,
    bool adjoint,
    TwoPortStats &stats
) {
    if (
        !valid_two_port_module(module)
        || module.scope != TwoPortScope::Joint
    ) {
        fail("invalid separable joint surrogate descriptor");
    }
    const double sign = adjoint ? -1.0 : 1.0;
    for (std::size_t index = 0; index < plan.necklaces.size(); ++index) {
        const double angle = sign * 2.0 * kPi
            * static_cast<double>(
                mod(
                    module.strength
                    * two_port_feature(plan.necklaces[index], module)
                )
            ) / static_cast<double>(kGrid);
        const std::size_t offset = 4U * index;
        /*
         * diag(e^ia, 1, 1, e^-ia) is exactly the tensor product
         * diag(e^i(a/2), e^-i(a/2)) on A and the same diagonal on B.
         */
        carrier[offset] *= std::polar(1.0, angle);
        carrier[offset + 3U] *= std::polar(1.0, -angle);
        stats.coupling_updates += kTwoPortCells;
        ++stats.joint_consumptions;
    }
}

void apply_two_port_generator(
    std::vector<Complex> &carrier,
    const Plan &plan,
    int chirp,
    bool adjoint,
    TwoPortStats &stats
) {
    for (std::size_t latent = 0; latent < kTwoPortCells; ++latent) {
        std::vector<Complex> fiber(plan.necklaces.size());
        for (std::size_t index = 0;
             index < plan.necklaces.size();
             ++index) {
            fiber[index] = carrier[4U * index + latent];
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
            carrier[4U * index + latent] = fiber[index];
            ++stats.generator_fiber_copies;
        }
    }
}

void two_port_forward_module(
    std::vector<Complex> &carrier,
    const Plan &plan,
    const TwoPortModule &module,
    TwoPortStats &stats
) {
    apply_two_port_coupling(
        carrier, plan, module, false, stats
    );
    apply_two_port_generator(
        carrier, plan, module.chirp, false, stats
    );
}

void two_port_inverse_module(
    std::vector<Complex> &carrier,
    const Plan &plan,
    const TwoPortModule &module,
    TwoPortStats &stats
) {
    apply_two_port_generator(
        carrier, plan, module.chirp, true, stats
    );
    apply_two_port_coupling(
        carrier, plan, module, true, stats
    );
}

std::vector<TwoPortModule> two_port_primary_program() {
    return {
        {
            TwoPortFeature::Collision,
            TwoPortScope::PortA,
            TwoPortAxis::X,
            0,
            3,
            1,
            kPortAOwner,
            0U,
        },
        {
            TwoPortFeature::CyclicSeparation,
            TwoPortScope::Joint,
            TwoPortAxis::ControlledPhase,
            1,
            5,
            4,
            kPortAOwner,
            kPortBOwner,
        },
        {
            TwoPortFeature::CyclicSeparation,
            TwoPortScope::PortB,
            TwoPortAxis::Y,
            3,
            7,
            6,
            0U,
            kPortBOwner,
        },
        {
            TwoPortFeature::Collision,
            TwoPortScope::PortA,
            TwoPortAxis::Z,
            0,
            11,
            9,
            kPortAOwner,
            0U,
        },
        {
            TwoPortFeature::CyclicSeparation,
            TwoPortScope::Joint,
            TwoPortAxis::ControlledPhase,
            2,
            4,
            8,
            kPortAOwner,
            kPortBOwner,
        },
        {
            TwoPortFeature::CyclicSeparation,
            TwoPortScope::PortB,
            TwoPortAxis::X,
            7,
            14,
            13,
            0U,
            kPortBOwner,
        },
    };
}

std::vector<TwoPortModule> two_port_reuse_program() {
    return {
        {
            TwoPortFeature::CyclicSeparation,
            TwoPortScope::PortB,
            TwoPortAxis::Z,
            2,
            4,
            3,
            0U,
            kPortBOwner,
        },
        {
            TwoPortFeature::Collision,
            TwoPortScope::Joint,
            TwoPortAxis::ControlledPhase,
            0,
            9,
            7,
            kPortAOwner,
            kPortBOwner,
        },
        {
            TwoPortFeature::CyclicSeparation,
            TwoPortScope::PortA,
            TwoPortAxis::Y,
            5,
            12,
            11,
            kPortAOwner,
            0U,
        },
        {
            TwoPortFeature::Collision,
            TwoPortScope::PortB,
            TwoPortAxis::X,
            0,
            6,
            14,
            0U,
            kPortBOwner,
        },
    };
}

TwoPortRun two_port_transaction(
    std::vector<Complex> &carrier,
    const std::vector<Complex> &baseline,
    const Plan &plan,
    const std::vector<TwoPortModule> &program,
    TwoPortControl control
) {
    if (!valid_two_port_program(program)) {
        fail("invalid two-port program");
    }
    TwoPortRun run;
    for (const TwoPortModule &module : program) {
        two_port_forward_module(carrier, plan, module, run.stats);
    }
    run.boundary = two_port_boundary(carrier, plan);
    run.norm_error =
        std::fabs(two_port_weighted_norm(carrier, plan) - 1.0);

    const std::size_t minimum =
        control == TwoPortControl::Missing ? 1U : 0U;
    for (std::size_t cursor = program.size();
         cursor > minimum;
         --cursor) {
        TwoPortModule module = program[cursor - 1U];
        if (
            control == TwoPortControl::WrongSemantic
            && cursor == program.size()
        ) {
            module.strength = mod(module.strength + 1);
            if (module.strength == 0) {
                module.strength = 1;
            }
        }
        if (control == TwoPortControl::ReorderedInverse) {
            apply_two_port_coupling(
                carrier, plan, module, true, run.stats
            );
            apply_two_port_generator(
                carrier,
                plan,
                module.chirp,
                true,
                run.stats
            );
        } else {
            two_port_inverse_module(
                carrier, plan, module, run.stats
            );
        }
    }
    run.restoration_error =
        two_port_l2_distance(carrier, baseline, plan);
    return run;
}

TwoPortRun two_port_joint_control_transaction(
    std::vector<Complex> &carrier,
    const std::vector<Complex> &baseline,
    const Plan &plan,
    const std::vector<TwoPortModule> &program,
    JointControlMode joint_mode
) {
    if (
        !valid_two_port_program(program)
        || joint_mode == JointControlMode::Native
    ) {
        fail("invalid two-port joint-control transaction");
    }
    TwoPortRun run;
    for (const TwoPortModule &module : program) {
        if (module.scope == TwoPortScope::Joint) {
            if (joint_mode == JointControlMode::SeparableDiagonal) {
                apply_separable_joint_coupling(
                    carrier, plan, module, false, run.stats
                );
            } else {
                ++run.stats.joint_consumptions;
            }
            apply_two_port_generator(
                carrier, plan, module.chirp, false, run.stats
            );
        } else {
            two_port_forward_module(
                carrier, plan, module, run.stats
            );
        }
    }
    run.boundary = two_port_boundary(carrier, plan);
    run.norm_error =
        std::fabs(two_port_weighted_norm(carrier, plan) - 1.0);
    for (std::size_t cursor = program.size();
         cursor > 0U;
         --cursor) {
        const TwoPortModule &module = program[cursor - 1U];
        if (module.scope == TwoPortScope::Joint) {
            apply_two_port_generator(
                carrier, plan, module.chirp, true, run.stats
            );
            if (joint_mode == JointControlMode::SeparableDiagonal) {
                apply_separable_joint_coupling(
                    carrier, plan, module, true, run.stats
                );
            } else {
                ++run.stats.joint_consumptions;
            }
        } else {
            two_port_inverse_module(
                carrier, plan, module, run.stats
            );
        }
    }
    run.restoration_error =
        two_port_l2_distance(carrier, baseline, plan);
    return run;
}

}  // namespace

int NECKLACE_TWO_SHARED_LATENT_ENTRY() {
    const Plan plan = compile_plan();
    const std::vector<Complex> baseline =
        make_two_port_carrier(plan, 0);
    const std::vector<TwoPortModule> program =
        two_port_primary_program();
    const std::vector<TwoPortModule> reuse_program =
        two_port_reuse_program();
    if (
        !valid_two_port_program(program)
        || !valid_two_port_program(reuse_program)
    ) {
        fail("declared two-port program invalid");
    }

    const TwoPortModule joint_probe = program[1];
    const double product_determinant =
        maximum_fiber_determinant(baseline);
    std::vector<Complex> entanglement_probe = baseline;
    TwoPortStats entanglement_stats;
    apply_two_port_coupling(
        entanglement_probe,
        plan,
        joint_probe,
        false,
        entanglement_stats
    );
    const double joint_determinant =
        maximum_fiber_determinant(entanglement_probe);
    apply_two_port_coupling(
        entanglement_probe,
        plan,
        joint_probe,
        true,
        entanglement_stats
    );
    const double probe_restoration_error = two_port_l2_distance(
        entanglement_probe, baseline, plan
    );
    if (
        product_determinant > 1.0e-15
        || joint_determinant < 1.0e-8
        || probe_restoration_error > kTwoPortTolerance
    ) {
        fail("joint two-port nonseparability probe failed");
    }

    std::vector<Complex> carrier = baseline;
    const Complex *backing = carrier.data();
    const TwoPortRun primary = two_port_transaction(
        carrier,
        baseline,
        plan,
        program,
        TwoPortControl::Correct
    );
    const TwoPortRun reuse = two_port_transaction(
        carrier,
        baseline,
        plan,
        reuse_program,
        TwoPortControl::Correct
    );
    std::vector<Complex> fresh = baseline;
    const TwoPortRun fresh_reuse = two_port_transaction(
        fresh,
        baseline,
        plan,
        reuse_program,
        TwoPortControl::Correct
    );
    const double reuse_boundary_error = boundary_distance(
        reuse.boundary, fresh_reuse.boundary
    );
    if (
        primary.restoration_error > kTwoPortTolerance
        || primary.norm_error > kTwoPortTolerance
        || reuse.restoration_error > kTwoPortTolerance
        || fresh_reuse.restoration_error > kTwoPortTolerance
        || reuse_boundary_error > kTwoPortTolerance
        || carrier.data() != backing
        || primary.stats.relation_table_cells != 0U
        || primary.stats.assignment_cells != 0U
    ) {
        fail("two-port primary or reuse transaction failed");
    }

    std::vector<TwoPortModule> no_joint;
    for (const TwoPortModule &module : program) {
        if (module.scope != TwoPortScope::Joint) {
            no_joint.push_back(module);
        }
    }
    std::vector<Complex> no_joint_carrier = baseline;
    const TwoPortRun no_joint_run = two_port_transaction(
        no_joint_carrier,
        baseline,
        plan,
        no_joint,
        TwoPortControl::Correct
    );
    const double joint_semantic_effect = boundary_distance(
        primary.boundary, no_joint_run.boundary
    );

    std::vector<Complex> identity_joint_carrier = baseline;
    const TwoPortRun identity_joint_run =
        two_port_joint_control_transaction(
            identity_joint_carrier,
            baseline,
            plan,
            program,
            JointControlMode::Identity
        );
    const double identity_joint_effect = boundary_distance(
        primary.boundary, identity_joint_run.boundary
    );
    std::vector<Complex> separable_joint_carrier = baseline;
    const TwoPortRun separable_joint_run =
        two_port_joint_control_transaction(
            separable_joint_carrier,
            baseline,
            plan,
            program,
            JointControlMode::SeparableDiagonal
        );
    const double separable_joint_effect = boundary_distance(
        primary.boundary, separable_joint_run.boundary
    );

    std::vector<TwoPortModule> swapped = program;
    std::swap(swapped[0], swapped[1]);
    std::vector<Complex> swapped_carrier = baseline;
    const TwoPortRun swapped_run = two_port_transaction(
        swapped_carrier,
        baseline,
        plan,
        swapped,
        TwoPortControl::Correct
    );
    const double module_order_effect = boundary_distance(
        primary.boundary, swapped_run.boundary
    );

    std::vector<TwoPortModule> perturbed = program;
    perturbed[4].strength += 1;
    std::vector<Complex> perturbed_carrier = baseline;
    const TwoPortRun perturbed_run = two_port_transaction(
        perturbed_carrier,
        baseline,
        plan,
        perturbed,
        TwoPortControl::Correct
    );
    const double semantic_effect = boundary_distance(
        primary.boundary, perturbed_run.boundary
    );
    std::array<double, 2> per_joint_consumer_effect{};
    const std::array<std::size_t, 2> joint_indices = {1U, 4U};
    for (std::size_t control_index = 0;
         control_index < joint_indices.size();
         ++control_index) {
        std::vector<TwoPortModule> joint_perturbed = program;
        TwoPortModule &module =
            joint_perturbed[joint_indices[control_index]];
        module.strength = mod(module.strength + 1);
        if (module.strength == 0) {
            module.strength = 1;
        }
        std::vector<Complex> joint_perturbed_carrier = baseline;
        const TwoPortRun joint_perturbed_run = two_port_transaction(
            joint_perturbed_carrier,
            baseline,
            plan,
            joint_perturbed,
            TwoPortControl::Correct
        );
        per_joint_consumer_effect[control_index] = boundary_distance(
            primary.boundary, joint_perturbed_run.boundary
        );
    }

    std::vector<Complex> missing_carrier = baseline;
    const TwoPortRun missing = two_port_transaction(
        missing_carrier,
        baseline,
        plan,
        {program[0], program[1]},
        TwoPortControl::Missing
    );
    std::vector<Complex> reordered_carrier = baseline;
    const TwoPortRun reordered = two_port_transaction(
        reordered_carrier,
        baseline,
        plan,
        {program[0], program[1]},
        TwoPortControl::ReorderedInverse
    );
    std::vector<Complex> wrong_carrier = baseline;
    const TwoPortRun wrong = two_port_transaction(
        wrong_carrier,
        baseline,
        plan,
        {program[0], program[1]},
        TwoPortControl::WrongSemantic
    );

    TwoPortModule wrong_type = program[0];
    wrong_type.scope = TwoPortScope::Joint;
    const bool wrong_type_rejected =
        !valid_two_port_module(wrong_type);
    TwoPortModule wrong_owner = program[1];
    wrong_owner.owner_b ^= 1U;
    const bool wrong_owner_rejected =
        !valid_two_port_module(wrong_owner);
    if (
        joint_semantic_effect < 1.0e-7
        || identity_joint_effect < 1.0e-7
        || separable_joint_effect < 1.0e-7
        || module_order_effect < 1.0e-7
        || semantic_effect < 1.0e-7
        || per_joint_consumer_effect[0] < 1.0e-7
        || per_joint_consumer_effect[1] < 1.0e-7
        || identity_joint_run.restoration_error > kTwoPortTolerance
        || separable_joint_run.restoration_error > kTwoPortTolerance
        || missing.restoration_error < kControlFloor
        || reordered.restoration_error < kControlFloor
        || wrong.restoration_error < kControlFloor
        || !wrong_type_rejected
        || !wrong_owner_rejected
    ) {
        fail("two-port causal controls failed");
    }

    std::vector<Complex> matched_classical = baseline;
    const TwoPortRun classical = two_port_transaction(
        matched_classical,
        baseline,
        plan,
        program,
        TwoPortControl::Correct
    );
    const double classical_error = boundary_distance(
        primary.boundary, classical.boundary
    );
    if (classical_error != 0.0) {
        fail("two-port compact classical parity failed");
    }

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"OWNER_BOUND_TWO_SHARED_LATENT_PORT_JOINT_PHASE_CONTRACTION_ON_NECKLACE_CARRIER\","
        "\"result\":\"PASS\","
        "\"claim_ceiling\":\"DIRECT_PROCESS_GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACES_FOUR_CELL_TWO_BINARY_LATENT_FIBER_SIX_TYPED_MODULES_TWO_JOINT_CONSUMERS_COMPLEX128_SOFTWARE_ONLY\","
        "\"resident_necklace_cells\":285,"
        "\"latent_cells_per_necklace\":4,"
        "\"resident_joint_complex_cells\":1140,"
        "\"shared_latent_port_count\":2,"
        "\"port_a_owner\":%u,"
        "\"port_b_owner\":%u,"
        "\"primary_module_count\":%zu,"
        "\"joint_consumer_count\":2,"
        "\"latent_ports_projected\":false,"
        "\"relation_table_cells\":0,"
        "\"assignment_cells\":0,",
        kPortAOwner,
        kPortBOwner,
        program.size()
    );
    std::printf(
        "\"nonseparability_probe\":{"
        "\"product_input_maximum_fiber_determinant\":%.17g,"
        "\"post_joint_maximum_fiber_determinant\":%.17g,"
        "\"joint_inverse_restoration_error\":%.17g},"
        "\"boundary\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"
        "\"primary_restoration_error\":%.17g,"
        "\"reuse_restoration_error\":%.17g,"
        "\"fresh_restored_reuse_boundary_error\":%.17g,"
        "\"carrier_backing_preserved\":true,"
        "\"restoration_generation\":2,"
        "\"restoration_class\":\"NUMERICAL_PHYSICAL_STATE_RESTORATION\",",
        product_determinant,
        joint_determinant,
        probe_restoration_error,
        primary.boundary[0],
        primary.boundary[1],
        primary.boundary[2],
        primary.boundary[3],
        primary.boundary[4],
        primary.boundary[5],
        primary.boundary[6],
        primary.restoration_error,
        reuse.restoration_error,
        reuse_boundary_error
    );
    std::printf(
        "\"controls\":{"
        "\"joint_module_deletion_boundary_effect\":%.17g,"
        "\"identity_joint_same_generator_boundary_effect\":%.17g,"
        "\"separable_joint_same_generator_boundary_effect\":%.17g,"
        "\"joint_consumer_strength_boundary_effects\":[%.17g,%.17g],"
        "\"module_order_boundary_effect\":%.17g,"
        "\"semantic_perturbation_boundary_effect\":%.17g,"
        "\"missing_inverse_error\":%.17g,"
        "\"reordered_inverse_error\":%.17g,"
        "\"wrong_semantic_inverse_error\":%.17g,"
        "\"wrong_type_rejected\":true,"
        "\"wrong_owner_rejected\":true,"
        "\"undermerge_boundary_effect\":%.17g},"
        "\"resource_law\":{"
        "\"resident_complex_cells\":1140,"
        "\"carrier_payload_bytes\":18240,"
        "\"persistent_baseline_plus_carrier_payload_bytes\":36480,"
        "\"generator_carrier_sized_work_vectors_per_fiber\":3,"
        "\"outer_extraction_fiber_complex_cells\":285,"
        "\"declared_generator_and_extraction_scratch_payload_bytes\":18240,"
        "\"temporary_occupation_cells\":0,"
        "\"dense_285_operator_cells\":0,"
        "\"retained_inverse_history_bytes\":0,"
        "\"reported_scope\":\"DECLARED_STD_VECTOR_COMPLEX_PAYLOADS_ONLY_NOT_TOTAL_PROCESS_PEAK\","
        "\"plan_program_harness_allocator_native_library_os_memory_bounded\":false},",
        joint_semantic_effect,
        identity_joint_effect,
        separable_joint_effect,
        per_joint_consumer_effect[0],
        per_joint_consumer_effect[1],
        module_order_effect,
        semantic_effect,
        missing.restoration_error,
        reordered.restoration_error,
        wrong.restoration_error,
        joint_semantic_effect
    );
    std::printf(
        "\"strongest_compact_classical\":{"
        "\"same_1140_complex_recurrence_exists\":true,"
        "\"identity_reexecution_boundary_error\":%.17g,"
        "\"verification_level\":\"PACKAGE_SELF_REVIEW\","
        "\"separate_reference_parity\":false},"
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
