#define NECKLACE_COHERENCE_TRIAD_ENTRY necklace_coherence_triad_predecessor_main
#include "four_rotor_necklace_coherence_triad.cpp"
#undef NECKLACE_COHERENCE_TRIAD_ENTRY

/*
 * Typed open observation-relation closure on the compact necklace carrier.
 *
 * Each public signature defines a phase-valued relation R(x, o) between an
 * unresolved necklace x and a typed, open observation port o. Native closure
 * substitutes the public observation without expanding the relation over its
 * observation domain or projecting x. Hermitian generator closure composes
 * the resulting modules. Only the final collision-hypothesis score survives
 * the actual inverse.
 */

namespace {

enum class EvidenceKind : std::uint32_t {
    Collision = 1,
    CyclicSeparation = 2,
};

struct OpenObservationRelation {
    EvidenceKind kind = EvidenceKind::Collision;
    int separation = 0;
    int observation_min = 0;
    int observation_max = 0;
    int strength = 0;
};

struct ObservationClosure {
    OpenObservationRelation relation{};
    int observed_value = 0;
    int chirp = 1;
};

struct InferenceStats {
    std::uint64_t evidence_phase_updates = 0;
    std::uint64_t open_observation_ports_closed = 0;
    std::uint64_t relation_table_cells_materialized = 0;
    GeneratorStats generator{};
};

enum class InferenceControl {
    Correct,
    Missing,
    Wrong,
    Reordered,
};

int cyclic_separation_pairs(
    const Histogram &histogram,
    int separation
) {
    if (separation < 1 || separation > kGrid / 2) {
        fail("invalid cyclic separation evidence");
    }
    int result = 0;
    for (int mode = 0; mode < kGrid; ++mode) {
        result += static_cast<int>(histogram[mode])
            * static_cast<int>(
                histogram[mod(mode + separation)]
            );
    }
    return result;
}

int evidence_value(
    const Necklace &necklace,
    const OpenObservationRelation &relation
) {
    if (relation.kind == EvidenceKind::Collision) {
        return necklace.collisions;
    }
    if (relation.kind == EvidenceKind::CyclicSeparation) {
        return cyclic_separation_pairs(
            necklace.histogram, relation.separation
        );
    }
    fail("unknown evidence port type");
}

void close_open_observation_relation(
    std::vector<Complex> &samples,
    const Plan &plan,
    const ObservationClosure &closure,
    bool adjoint,
    InferenceStats &stats
) {
    if (
        closure.observed_value
            < closure.relation.observation_min
        || closure.observed_value
            > closure.relation.observation_max
    ) {
        fail("observation outside typed relation domain");
    }
    const int sign = adjoint ? -1 : 1;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const int difference =
            evidence_value(
                plan.necklaces[index], closure.relation
            ) - closure.observed_value;
        const int exponent = sign * closure.relation.strength
            * difference * difference;
        samples[index] *= plan.roots[mod(exponent)];
        ++stats.evidence_phase_updates;
    }
    ++stats.open_observation_ports_closed;
}

void forward_evidence_module(
    std::vector<Complex> &samples,
    const Plan &plan,
    const ObservationClosure &closure,
    InferenceStats &stats
) {
    close_open_observation_relation(
        samples, plan, closure, false, stats
    );
    generator_free(
        samples,
        plan,
        closure.chirp,
        false,
        stats.generator
    );
}

void inverse_evidence_module(
    std::vector<Complex> &samples,
    const Plan &plan,
    const ObservationClosure &closure,
    InferenceStats &stats
) {
    generator_free(
        samples,
        plan,
        closure.chirp,
        true,
        stats.generator
    );
    close_open_observation_relation(
        samples, plan, closure, true, stats
    );
}

struct InferenceRun {
    Boundary boundary{};
    Boundary semantic_reference{};
    InferenceStats stats{};
    double restoration_error = 0.0;
    double norm_error = 0.0;
    double elapsed_ms = 0.0;
};

Boundary independently_reference_hypothesis_scores(
    const std::vector<Complex> &samples,
    const Plan &plan
);

InferenceRun inference_transaction(
    std::vector<Complex> &carrier,
    const std::vector<Complex> &baseline,
    const Plan &plan,
    const std::vector<ObservationClosure> &program,
    InferenceControl control
) {
    const auto begin = std::chrono::steady_clock::now();
    InferenceRun result;
    for (const ObservationClosure &closure : program) {
        forward_evidence_module(
            carrier, plan, closure, result.stats
        );
    }
    result.boundary = project_boundary(carrier, plan);
    result.semantic_reference =
        independently_reference_hypothesis_scores(carrier, plan);
    result.norm_error = std::fabs(
        weighted_norm(carrier, plan) - 1.0
    );

    std::size_t minimum_index =
        control == InferenceControl::Missing ? 1U : 0U;
    for (
        std::size_t cursor = program.size();
        cursor > minimum_index;
        --cursor
    ) {
        const std::size_t index = cursor - 1;
        ObservationClosure inverse_closure = program[index];
        if (
            control == InferenceControl::Wrong
            && index == program.size() - 1
        ) {
            ++inverse_closure.relation.strength;
        }
        if (control == InferenceControl::Reordered) {
            close_open_observation_relation(
                carrier,
                plan,
                inverse_closure,
                true,
                result.stats
            );
            generator_free(
                carrier,
                plan,
                inverse_closure.chirp,
                true,
                result.stats.generator
            );
        } else {
            inverse_evidence_module(
                carrier, plan, inverse_closure, result.stats
            );
        }
    }
    result.restoration_error =
        l2_distance(carrier, baseline, plan);
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin
    ).count();
    return result;
}

Boundary dephased_inference_boundary(
    const std::vector<Complex> &initial,
    const Plan &plan,
    const std::vector<ObservationClosure> &program,
    DephasedStats &stats
) {
    std::vector<double> probabilities =
        probabilities_from_carrier(initial, plan);
    for (const ObservationClosure &closure : program) {
        stats.collision_phases_erased += probabilities.size();
        dephased_free(
            probabilities, plan, closure.chirp, stats
        );
    }
    return project_probabilities(probabilities, plan);
}

OpenObservationRelation collision_relation(int strength) {
    return {
        EvidenceKind::Collision,
        0,
        0,
        kMaximumCollision,
        strength,
    };
}

OpenObservationRelation separation_relation(
    int separation,
    int strength
) {
    return {
        EvidenceKind::CyclicSeparation,
        separation,
        0,
        4,
        strength,
    };
}

std::vector<ObservationClosure> primary_program() {
    return {
        {collision_relation(3), 1, 1},
        {separation_relation(1, 5), 2, 4},
        {separation_relation(3, 7), 0, 6},
        {collision_relation(11), 3, 9},
        {separation_relation(5, 13), 1, 12},
        {separation_relation(8, 2), 4, 15},
    };
}

std::vector<ObservationClosure> reuse_program() {
    return {
        {separation_relation(2, 4), 3, 3},
        {collision_relation(9), 0, 7},
        {separation_relation(7, 14), 2, 13},
    };
}

std::vector<ObservationClosure> control_program() {
    return {
        {collision_relation(3), 1, 1},
        {separation_relation(1, 5), 2, 4},
    };
}

Boundary independently_reference_hypothesis_scores(
    const std::vector<Complex> &samples,
    const Plan &plan
) {
    Boundary reference{};
    for (int hypothesis = 0;
         hypothesis <= kMaximumCollision;
         ++hypothesis) {
        double score = 0.0;
        for (std::size_t index = 0;
             index < samples.size();
             ++index) {
            if (
                plan.necklaces[index].collisions
                == hypothesis
            ) {
                score += static_cast<double>(
                    plan.necklaces[index].labelled_weight
                ) * std::norm(samples[index]);
            }
        }
        reference[hypothesis] = score;
    }
    return reference;
}

int decide_hypothesis(const Boundary &scores) {
    return static_cast<int>(
        std::distance(
            scores.begin(),
            std::max_element(scores.begin(), scores.end())
        )
    );
}

}  // namespace

int main() {
    const Plan plan = compile_plan();
    const std::vector<Complex> baseline = make_carrier(plan, 0);
    const std::vector<ObservationClosure> program =
        primary_program();

    std::vector<Complex> carrier = baseline;
    const Complex *carrier_backing = carrier.data();
    const InferenceRun primary = inference_transaction(
        carrier,
        baseline,
        plan,
        program,
        InferenceControl::Correct
    );
    if (
        primary.restoration_error > kRestorationTolerance
        || primary.norm_error > kGeneratorTolerance
        || carrier.data() != carrier_backing
    ) {
        fail("relational inference primary gate failed");
    }

    const std::vector<ObservationClosure> reuse_ports =
        reuse_program();
    const InferenceRun reuse = inference_transaction(
        carrier,
        baseline,
        plan,
        reuse_ports,
        InferenceControl::Correct
    );
    std::vector<Complex> fresh_carrier = baseline;
    const InferenceRun fresh_reuse = inference_transaction(
        fresh_carrier,
        baseline,
        plan,
        reuse_ports,
        InferenceControl::Correct
    );
    const double reuse_boundary_error =
        boundary_distance(reuse.boundary, fresh_reuse.boundary);
    if (
        reuse.restoration_error > kRestorationTolerance
        || reuse_boundary_error > kGeneratorTolerance
        || carrier.data() != carrier_backing
    ) {
        fail("relational inference reuse gate failed");
    }

    std::vector<Complex> classical_carrier = baseline;
    const InferenceRun matched_classical = inference_transaction(
        classical_carrier,
        baseline,
        plan,
        program,
        InferenceControl::Correct
    );
    const double classical_boundary_error = boundary_distance(
        primary.boundary, matched_classical.boundary
    );
    if (classical_boundary_error != 0.0) {
        fail("relational inference classical parity failed");
    }

    std::vector<ObservationClosure> bypass_program = program;
    bypass_program[2].relation.strength = 0;
    std::vector<Complex> bypass_carrier = baseline;
    const InferenceRun bypass = inference_transaction(
        bypass_carrier,
        baseline,
        plan,
        bypass_program,
        InferenceControl::Correct
    );
    const double bypass_boundary_effect =
        boundary_distance(primary.boundary, bypass.boundary);

    std::vector<ObservationClosure> swapped_program = program;
    std::swap(swapped_program[1], swapped_program[2]);
    std::vector<Complex> swapped_carrier = baseline;
    const InferenceRun swapped = inference_transaction(
        swapped_carrier,
        baseline,
        plan,
        swapped_program,
        InferenceControl::Correct
    );
    const double order_boundary_effect =
        boundary_distance(primary.boundary, swapped.boundary);

    DephasedStats dephased_stats;
    const Boundary dephased_boundary = dephased_inference_boundary(
        baseline, plan, program, dephased_stats
    );
    const double coherence_boundary_effect =
        boundary_distance(primary.boundary, dephased_boundary);

    const std::vector<ObservationClosure> controls =
        control_program();
    std::vector<Complex> missing_carrier = baseline;
    const InferenceRun missing = inference_transaction(
        missing_carrier,
        baseline,
        plan,
        controls,
        InferenceControl::Missing
    );
    std::vector<Complex> wrong_carrier = baseline;
    const InferenceRun wrong = inference_transaction(
        wrong_carrier,
        baseline,
        plan,
        controls,
        InferenceControl::Wrong
    );
    std::vector<Complex> reordered_carrier = baseline;
    const InferenceRun reordered = inference_transaction(
        reordered_carrier,
        baseline,
        plan,
        controls,
        InferenceControl::Reordered
    );

    if (
        bypass_boundary_effect < 1.0e-5
        || order_boundary_effect < 1.0e-5
        || coherence_boundary_effect < 1.0e-5
        || missing.restoration_error < kControlFloor
        || wrong.restoration_error < kControlFloor
        || reordered.restoration_error < kControlFloor
    ) {
        fail("relational inference causal controls failed");
    }

    const double semantic_reference_error = boundary_distance(
        primary.boundary, primary.semantic_reference
    );
    const int selected_hypothesis =
        decide_hypothesis(primary.boundary);
    if (
        semantic_reference_error > kGeneratorTolerance
        || selected_hypothesis < 0
        || selected_hypothesis > kMaximumCollision
    ) {
        fail("hypothesis scoring semantic reference failed");
    }

    const std::uint64_t carrier_bytes =
        plan.necklaces.size() * sizeof(Complex);
    const std::uint64_t program_bytes =
        program.capacity() * sizeof(ObservationClosure);
    const std::uint64_t engine_bytes =
        33470U + program_bytes;
    const std::uint64_t lifecycle_bytes =
        engine_bytes + carrier_bytes + 2U * sizeof(Boundary);

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"BOUNDED_COHERENCE_DEPENDENT_TYPED_OPEN_OBSERVATION_RELATION_CLOSURE_AND_CATALYTIC_HYPOTHESIS_SCORING_ON_COMPACT_NECKLACE_PHASE_CARRIER_WITH_ACTUAL_RESTORATION_AND_REUSE\","
    );
    std::printf(
        "\"claim_ceiling\":\"DIRECT_PROCESS_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_GRID17_FOUR_ROTOR_SIX_TYPED_OPEN_PUBLIC_OBSERVATION_PORTS_COMPLEX128_SOFTWARE_ONLY\","
    );
    std::printf("\"result\":\"PASS\",");
    std::printf(
        "\"typed_open_observation_relations\":{"
        "\"ports\":6,"
        "\"kinds\":[\"COLLISION\",\"CYCLIC_SEPARATION\"],"
        "\"cyclic_separations\":[1,3,5,8],"
        "\"relation\":\"R(x,o)=OMEGA^(STRENGTH*(FEATURE(x)-o)^2)\","
        "\"observation_domains\":{"
        "\"COLLISION\":[0,6],"
        "\"CYCLIC_SEPARATION\":[0,4]},"
        "\"public_observations\":[1,2,0,3,1,4],"
        "\"native_substitution_closure\":true,"
        "\"observation_domain_enumerated\":false,"
        "\"relation_table_cells_materialized\":0,"
        "\"intermediate_necklace_cells\":285,"
        "\"accepted_path_intermediate_projected\":false,"
        "\"accepted_path_truth_table_cells\":0,"
        "\"accepted_path_candidate_assignment_cells\":0},"
    );
    std::printf(
        "\"inference_boundary\":{"
        "\"hypothesis_variable\":\"TOTAL_UNORDERED_OCCUPATION_COLLISION_COUNT\","
        "\"hypothesis_domain\":[0,1,2,3,4,5,6],"
        "\"observation_semantics\":\"PUBLIC_EXACT_ROTATION_INVARIANT_FEATURE_VALUES_CLOSED_INTO_TYPED_PHASE_RELATIONS\","
        "\"score_semantics\":\"WEIGHTED_COLLISION_HYPOTHESIS_INTERFERENCE_SCORE_NOT_CALIBRATED_BAYESIAN_POSTERIOR\","
        "\"values\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"
        "\"decision_rule\":\"ARGMAX_SCORE_LOWEST_INDEX_TIEBREAK\","
        "\"selected_hypothesis\":%d,"
        "\"independent_semantic_reference_error\":%.17g,"
        "\"retained_outside_inverse_history\":true},",
        primary.boundary[0],
        primary.boundary[1],
        primary.boundary[2],
        primary.boundary[3],
        primary.boundary[4],
        primary.boundary[5],
        primary.boundary[6],
        selected_hypothesis,
        semantic_reference_error
    );
    std::printf(
        "\"primary\":{"
        "\"restoration_generation\":1,"
        "\"restoration_error\":%.17g,"
        "\"weighted_norm_error\":%.17g,"
        "\"actual_inverse_restoration\":true,"
        "\"carrier_backing_preserved\":true,"
        "\"evidence_phase_updates\":%llu,"
        "\"open_observation_ports_closed\":%llu,"
        "\"relation_table_cells_materialized\":%llu,"
        "\"generator_applications\":%llu,"
        "\"streamed_generator_terms\":%llu,"
        "\"engine_explicit_payload_bytes\":%llu,"
        "\"verification_baseline_bytes\":%llu,"
        "\"projection_boundary_bytes\":%llu,"
        "\"lifecycle_explicit_payload_bytes\":%llu},",
        primary.restoration_error,
        primary.norm_error,
        static_cast<unsigned long long>(
            primary.stats.evidence_phase_updates
        ),
        static_cast<unsigned long long>(
            primary.stats.open_observation_ports_closed
        ),
        static_cast<unsigned long long>(
            primary.stats.relation_table_cells_materialized
        ),
        static_cast<unsigned long long>(
            primary.stats.generator.generator_applications
        ),
        static_cast<unsigned long long>(
            primary.stats.generator.streamed_generator_terms
        ),
        static_cast<unsigned long long>(engine_bytes),
        static_cast<unsigned long long>(carrier_bytes),
        static_cast<unsigned long long>(sizeof(Boundary)),
        static_cast<unsigned long long>(lifecycle_bytes)
    );
    std::printf(
        "\"reuse\":{"
        "\"restoration_generation\":2,"
        "\"restoration_error\":%.17g,"
        "\"fresh_restored_boundary_error\":%.17g,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"carrier_backing_preserved\":true},",
        reuse.restoration_error,
        reuse_boundary_error
    );
    std::printf(
        "\"causal_controls\":{"
        "\"bypassed_evidence_boundary_effect\":%.17g,"
        "\"reordered_module_boundary_effect\":%.17g,"
        "\"initial_and_interstep_dephasing_boundary_effect\":%.17g,"
        "\"missing_inverse_error\":%.17g,"
        "\"wrong_inverse_error\":%.17g,"
        "\"reordered_inverse_error\":%.17g},",
        bypass_boundary_effect,
        order_boundary_effect,
        coherence_boundary_effect,
        missing.restoration_error,
        wrong.restoration_error,
        reordered.restoration_error
    );
    std::printf(
        "\"dephased_comparison\":{"
        "\"initial_and_each_step_necklace_basis_dephasing\":true,"
        "\"evidence_phase_contribution_isolated\":false,"
        "\"outside_accepted_coherent_path\":true,"
        "\"streamed_transition_coefficients\":%llu,"
        "\"permanent_assignment_terms\":%llu},",
        static_cast<unsigned long long>(
            dephased_stats.streamed_transition_coefficients
        ),
        static_cast<unsigned long long>(
            dephased_stats.permanent_assignment_terms
        )
    );
    std::printf(
        "\"matched_compact_classical\":{"
        "\"same_executable_complex_recurrence\":true,"
        "\"boundary_error\":%.17g,"
        "\"restoration_error\":%.17g},",
        classical_boundary_error,
        matched_classical.restoration_error
    );
    std::printf(
        "\"topology_compilation_weak_compositions\":4845,"
        "\"machine_boundary_enforced\":false,"
        "\"no_smuggle_enforced\":false,"
        "\"accepted_path_intermediate_emitted\":false,"
        "\"bounded_catalytic_hypothesis_scoring_contract_established\":true,"
        "\"general_catalytic_inference_established\":false,"
        "\"ground_truth_accuracy_established\":false,"
        "\"calibrated_statistical_inference_established\":false,"
        "\"learning_advantage_established\":false,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"unbounded_computation_established\":false,"
        "\"obstruction\":\"TYPED_OPEN_OBSERVATION_RELATION_PHASE_CLOSURE_AND_BOUNDED_HYPOTHESIS_SCORING_ESTABLISHED_BUT_IDENTICAL_COMPACT_CLASSICAL_COMPLEX_RECURRENCE_INHERITS_IT\","
        "\"terminal\":false"
    );
    std::printf("}\n");
    return 0;
}
