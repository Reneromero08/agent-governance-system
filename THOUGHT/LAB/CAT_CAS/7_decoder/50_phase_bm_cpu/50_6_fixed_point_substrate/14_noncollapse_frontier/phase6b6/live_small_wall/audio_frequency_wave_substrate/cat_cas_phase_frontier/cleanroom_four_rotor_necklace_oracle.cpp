#define NECKLACE_GENERATOR_ENTRY cleanroom_generator_predecessor_main
#include "four_rotor_necklace_generator_phase.cpp"
#undef NECKLACE_GENERATOR_ENTRY

/*
 * Independent strict-scope oracle for typed public observation substitution.
 *
 * This translation unit deliberately does not include the production
 * relational-inference or coherence packages.  It reimplements descriptor
 * validation, phase substitution, score projection, inverse ordering, and the
 * dephased Markov recurrence from their public equations.  It reuses only the
 * separately audited 285-cell Hermitian necklace generator substrate.
 */

namespace {

enum class OracleKind {
    Collision,
    Separation,
};

struct OracleModule {
    OracleKind kind = OracleKind::Collision;
    int separation = 0;
    int observation_min = 0;
    int observation_max = 0;
    int strength = 0;
    int observed = 0;
    int chirp = 1;
};

struct OracleRun {
    Boundary score{};
    double restoration_error = 0.0;
    double norm_error = 0.0;
    std::uint64_t phase_updates = 0;
    GeneratorStats generator{};
};

struct OracleDephasedStats {
    std::uint64_t transition_coefficients = 0;
    std::uint64_t permanent_terms = 0;
    double probability_sum_error = 0.0;
};

bool valid_module(const OracleModule &module) {
    if (
        module.observed < module.observation_min
        || module.observed > module.observation_max
        || module.chirp < 1
        || module.chirp >= kGrid
    ) {
        return false;
    }
    if (module.kind == OracleKind::Collision) {
        return module.separation == 0
            && module.observation_min == 0
            && module.observation_max == kMaximumCollision;
    }
    return
        module.separation >= 1
        && module.separation <= kGrid / 2
        && module.observation_min == 0
        && module.observation_max == 4;
}

int oracle_feature(
    const Necklace &necklace,
    const OracleModule &module
) {
    if (module.kind == OracleKind::Collision) {
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

bool oracle_diagonal(
    std::vector<Complex> &carrier,
    const Plan &plan,
    const OracleModule &module,
    bool adjoint,
    std::uint64_t &updates
) {
    if (!valid_module(module)) {
        return false;
    }
    const int sign = adjoint ? -1 : 1;
    for (std::size_t index = 0; index < carrier.size(); ++index) {
        const int difference =
            oracle_feature(plan.necklaces[index], module)
            - module.observed;
        carrier[index] *= plan.roots[
            mod(sign * module.strength * difference * difference)
        ];
        ++updates;
    }
    return true;
}

Boundary oracle_score(
    const std::vector<Complex> &carrier,
    const Plan &plan
) {
    Boundary score{};
    for (int hypothesis = 0;
         hypothesis <= kMaximumCollision;
         ++hypothesis) {
        double value = 0.0;
        for (std::size_t index = 0; index < carrier.size(); ++index) {
            if (plan.necklaces[index].collisions == hypothesis) {
                value += static_cast<double>(
                    plan.necklaces[index].labelled_weight
                ) * std::norm(carrier[index]);
            }
        }
        score[hypothesis] = value;
    }
    return score;
}

OracleRun oracle_transaction(
    std::vector<Complex> &carrier,
    const std::vector<Complex> &baseline,
    const Plan &plan,
    const std::vector<OracleModule> &program
) {
    OracleRun run;
    for (const OracleModule &module : program) {
        if (!oracle_diagonal(
                carrier,
                plan,
                module,
                false,
                run.phase_updates
            )) {
            fail("cleanroom oracle rejected a valid module");
        }
        generator_free(
            carrier,
            plan,
            module.chirp,
            false,
            run.generator
        );
    }
    run.score = oracle_score(carrier, plan);
    run.norm_error = std::fabs(weighted_norm(carrier, plan) - 1.0);
    for (auto module = program.rbegin();
         module != program.rend();
         ++module) {
        generator_free(
            carrier,
            plan,
            module->chirp,
            true,
            run.generator
        );
        if (!oracle_diagonal(
                carrier,
                plan,
                *module,
                true,
                run.phase_updates
            )) {
            fail("cleanroom inverse rejected a valid module");
        }
    }
    run.restoration_error = l2_distance(carrier, baseline, plan);
    return run;
}

OracleModule collision(
    int strength,
    int observed,
    int chirp
) {
    return {
        OracleKind::Collision,
        0,
        0,
        kMaximumCollision,
        strength,
        observed,
        chirp,
    };
}

OracleModule separation(
    int distance,
    int strength,
    int observed,
    int chirp
) {
    return {
        OracleKind::Separation,
        distance,
        0,
        4,
        strength,
        observed,
        chirp,
    };
}

std::vector<OracleModule> primary_program() {
    return {
        collision(3, 1, 1),
        separation(1, 5, 2, 4),
        separation(3, 7, 0, 6),
        collision(11, 3, 9),
        separation(5, 13, 1, 12),
        separation(8, 2, 4, 15),
    };
}

std::vector<OracleModule> reuse_program() {
    return {
        separation(2, 4, 3, 3),
        collision(9, 0, 7),
        separation(7, 14, 2, 13),
    };
}

Complex oracle_transition(
    const Plan &plan,
    const Necklace &target,
    const Necklace &source,
    int chirp,
    OracleDephasedStats &stats
) {
    std::array<std::int64_t, kGrid> counts{};
    for (int shift = 0; shift < kGrid; ++shift) {
        const Histogram rotated =
            rotate_histogram(source.histogram, shift);
        const Tuple source_tuple = tuple_from_histogram(rotated);
        std::array<int, kRotors> permutation{};
        std::iota(permutation.begin(), permutation.end(), 0);
        do {
            int exponent = 0;
            for (int rotor = 0; rotor < kRotors; ++rotor) {
                const int difference =
                    static_cast<int>(target.representative[rotor])
                    - static_cast<int>(
                        source_tuple[permutation[rotor]]
                    );
                exponent += chirp * difference * difference;
            }
            ++counts[mod(exponent)];
            ++stats.permanent_terms;
        } while (
            std::next_permutation(
                permutation.begin(), permutation.end()
            )
        );
    }
    Complex result = 0.0;
    for (int exponent = 0; exponent < kGrid; ++exponent) {
        result += static_cast<double>(counts[exponent])
            * plan.roots[exponent];
    }
    result *= std::pow(static_cast<double>(kGrid), -0.5 * kRotors)
        / static_cast<double>(source.permanent_denominator);
    ++stats.transition_coefficients;
    return result;
}

Boundary oracle_dephased_score(
    const std::vector<Complex> &initial,
    const Plan &plan,
    const std::vector<OracleModule> &program,
    OracleDephasedStats &stats
) {
    std::vector<double> probability(initial.size());
    for (std::size_t index = 0; index < initial.size(); ++index) {
        probability[index] = static_cast<double>(
            plan.necklaces[index].labelled_weight
        ) * std::norm(initial[index]);
    }
    for (const OracleModule &module : program) {
        std::vector<double> output(probability.size(), 0.0);
        for (std::size_t target = 0;
             target < plan.necklaces.size();
             ++target) {
            const double target_weight = static_cast<double>(
                plan.necklaces[target].labelled_weight
            );
            for (std::size_t source = 0;
                 source < plan.necklaces.size();
                 ++source) {
                const Complex coefficient = oracle_transition(
                    plan,
                    plan.necklaces[target],
                    plan.necklaces[source],
                    module.chirp,
                    stats
                );
                const double source_weight = static_cast<double>(
                    plan.necklaces[source].labelled_weight
                );
                output[target] += target_weight / source_weight
                    * std::norm(coefficient)
                    * probability[source];
            }
        }
        probability.swap(output);
        const double sum = std::accumulate(
            probability.begin(), probability.end(), 0.0
        );
        stats.probability_sum_error = std::max(
            stats.probability_sum_error,
            std::fabs(sum - 1.0)
        );
    }
    Boundary score{};
    for (std::size_t index = 0; index < probability.size(); ++index) {
        score[plan.necklaces[index].collisions] += probability[index];
    }
    return score;
}

double maximum_case_restoration(
    const std::vector<OracleRun> &runs
) {
    double result = 0.0;
    for (const OracleRun &run : runs) {
        result = std::max(result, run.restoration_error);
    }
    return result;
}

}  // namespace

int main() {
    const Plan plan = compile_plan();
    const std::vector<Complex> baseline = make_carrier(plan, 0);
    const std::vector<OracleModule> primary_ports = primary_program();

    std::vector<std::vector<OracleModule>> programs;
    programs.push_back(primary_ports);

    std::vector<OracleModule> observations = primary_ports;
    const std::array<int, 6> alternate_observations = {4, 0, 3, 2, 4, 1};
    for (std::size_t index = 0; index < observations.size(); ++index) {
        observations[index].observed = alternate_observations[index];
    }
    programs.push_back(observations);

    std::vector<OracleModule> strengths = primary_ports;
    const std::array<int, 6> alternate_strengths = {1, 4, 9, 6, 15, 8};
    for (std::size_t index = 0; index < strengths.size(); ++index) {
        strengths[index].strength = alternate_strengths[index];
    }
    programs.push_back(strengths);

    std::vector<OracleModule> reversed = primary_ports;
    std::reverse(reversed.begin(), reversed.end());
    programs.push_back(reversed);
    programs.push_back({
        separation(2, 3, 0, 2),
        separation(4, 6, 1, 5),
        separation(6, 10, 3, 11),
        separation(7, 15, 4, 14),
    });

    std::vector<OracleRun> runs;
    std::vector<Complex> carrier = baseline;
    const Complex *backing = carrier.data();
    for (const std::vector<OracleModule> &program : programs) {
        const OracleRun run = oracle_transaction(
            carrier, baseline, plan, program
        );
        if (
            run.restoration_error > kRestorationTolerance
            || run.norm_error > kGeneratorTolerance
            || carrier.data() != backing
        ) {
            fail("cleanroom observation transfer case failed");
        }
        runs.push_back(run);
    }

    const std::array<double, 7> expected_primary = {
        0.47745721167016919,
        0.460098045965949,
        0.01934703306632151,
        0.040188169020947427,
        0.0,
        0.0,
        0.0029095402766043921,
    };
    double primary_parity_error = 0.0;
    for (std::size_t index = 0; index < expected_primary.size(); ++index) {
        primary_parity_error = std::max(
            primary_parity_error,
            std::fabs(runs[0].score[index] - expected_primary[index])
        );
    }
    if (primary_parity_error > kGeneratorTolerance) {
        fail("cleanroom final-score parity failed");
    }

    const std::vector<OracleModule> reuse_ports = reuse_program();
    const OracleRun restored_reuse = oracle_transaction(
        carrier, baseline, plan, reuse_ports
    );
    std::vector<Complex> fresh = baseline;
    const OracleRun fresh_reuse = oracle_transaction(
        fresh, baseline, plan, reuse_ports
    );
    const double reuse_error = boundary_distance(
        restored_reuse.score, fresh_reuse.score
    );
    if (
        restored_reuse.restoration_error > kRestorationTolerance
        || fresh_reuse.restoration_error > kRestorationTolerance
        || reuse_error > kGeneratorTolerance
        || carrier.data() != backing
        || restored_reuse.generator.streamed_generator_terms
            != fresh_reuse.generator.streamed_generator_terms
    ) {
        fail("cleanroom fresh/restored reuse failed");
    }

    std::vector<Complex> invalid = baseline;
    std::uint64_t invalid_updates = 0;
    const bool invalid_accepted = oracle_diagonal(
        invalid,
        plan,
        collision(3, 7, 1),
        false,
        invalid_updates
    );
    if (invalid_accepted || invalid_updates != 0U) {
        fail("cleanroom typed-domain rejection failed");
    }

    OracleDephasedStats dephased_stats;
    const Boundary dephased = oracle_dephased_score(
        baseline, plan, primary_ports, dephased_stats
    );
    const double coherence_effect =
        boundary_distance(runs[0].score, dephased);
    std::vector<Complex> matched_classical = baseline;
    const OracleRun matched_classical_run = oracle_transaction(
        matched_classical, baseline, plan, primary_ports
    );
    const double matched_classical_error = boundary_distance(
        runs[0].score, matched_classical_run.score
    );
    if (
        coherence_effect < 1.0e-5
        || dephased_stats.probability_sum_error > kGeneratorTolerance
        || matched_classical_error != 0.0
        || dephased_stats.transition_coefficients != 487350U
        || dephased_stats.permanent_terms != 198838800U
    ) {
        fail("cleanroom coherence diagnostic failed");
    }

    const double observation_effect = boundary_distance(
        runs[0].score, runs[1].score
    );
    const double strength_effect = boundary_distance(
        runs[0].score, runs[2].score
    );
    const double order_effect = boundary_distance(
        runs[0].score, runs[3].score
    );
    const double family_effect = boundary_distance(
        runs[0].score, runs[4].score
    );
    if (
        observation_effect < 1.0e-5
        || strength_effect < 1.0e-5
        || order_effect < 1.0e-5
        || family_effect < 1.0e-5
    ) {
        fail("cleanroom descriptor mutation did not transfer");
    }

    std::printf("{");
    std::printf(
        "\"source_head\":\"65be0046ae02c79ab8c3b3356ef68d891de19e53\","
        "\"result\":\"PASS\","
        "\"oracle_includes_production_observation_package\":false,"
        "\"oracle_calls_production_projection\":false,"
        "\"carrier_cells\":%zu,"
        "\"tested_programs\":5,"
        "\"different_program_family\":true,"
        "\"invalid_typed_observation_rejected\":true,",
        plan.necklaces.size()
    );
    std::printf(
        "\"primary_scores\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"
        "\"primary_production_parity_error\":%.17g,"
        "\"alternate_observation_effect\":%.17g,"
        "\"alternate_strength_effect\":%.17g,"
        "\"reversed_module_order_effect\":%.17g,"
        "\"different_family_effect\":%.17g,"
        "\"maximum_restoration_error\":%.17g,"
        "\"fresh_restored_reuse_error\":%.17g,"
        "\"carrier_backing_preserved\":true,"
        "\"reuse_resource_signature_equal\":true,",
        runs[0].score[0],
        runs[0].score[1],
        runs[0].score[2],
        runs[0].score[3],
        runs[0].score[4],
        runs[0].score[5],
        runs[0].score[6],
        primary_parity_error,
        observation_effect,
        strength_effect,
        order_effect,
        family_effect,
        maximum_case_restoration(runs),
        reuse_error
    );
    std::printf(
        "\"coherence\":{"
        "\"boundary_effect\":%.17g,"
        "\"probability_sum_error\":%.17g,"
        "\"transition_coefficients\":%llu,"
        "\"permanent_terms\":%llu,"
        "\"strongest_compact_coherent_classical_error\":%.17g,"
        "\"distinct_phase_resource_established\":false},"
        "\"restoration_class\":\"NUMERICAL_PHYSICAL_STATE_RESTORATION\","
        "\"dephased_restoration_class\":\"SNAPSHOT_RELOAD\","
        "\"shared_unresolved_observation_port_established\":false,"
        "\"general_inference_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_waveform_execution\":false,"
        "\"terminal\":false",
        coherence_effect,
        dephased_stats.probability_sum_error,
        static_cast<unsigned long long>(
            dephased_stats.transition_coefficients
        ),
        static_cast<unsigned long long>(
            dephased_stats.permanent_terms
        ),
        matched_classical_error
    );
    std::printf("}\n");
    return 0;
}
