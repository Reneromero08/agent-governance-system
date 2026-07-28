#define NECKLACE_GENERATOR_ENTRY necklace_generator_predecessor_main
#include "four_rotor_necklace_generator_phase.cpp"
#undef NECKLACE_GENERATOR_ENTRY

/*
 * One bounded resource diagnostic:
 *
 *   coherent in-place phase generator
 *   exact initial-and-each-step necklace-basis dephasing sham
 *   best matched compact classical complex recurrence
 *
 * The coherent and classical arms are intentionally the same executable
 * recurrence: this is the strongest matched classical software baseline, not
 * an expanded surrogate.  The dephased arm propagates only necklace
 * probabilities with the exact Markov kernel
 *
 *   P(t|s) = weight(t)/weight(s) * |U(t,s)|^2.
 *
 * The initial carrier is dephased before step zero and the state remains
 * necklace-basis diagonal after every free step. This tests the aggregate
 * effect of input and interstep coherence; it does not isolate the collision
 * phase contribution. Because dephasing is irreversible, that arm restores
 * only by a disclosed snapshot reload and is never accepted as catalytic
 * restoration.
 */

namespace {

struct DephasedStats {
    std::uint64_t collision_phases_erased = 0;
    std::uint64_t streamed_transition_coefficients = 0;
    std::uint64_t permanent_assignment_terms = 0;
    double maximum_probability_sum_error = 0.0;
    double elapsed_ms = 0.0;
};

std::vector<double> probabilities_from_carrier(
    const std::vector<Complex> &samples,
    const Plan &plan
) {
    std::vector<double> result(samples.size());
    for (std::size_t index = 0; index < samples.size(); ++index) {
        result[index] = static_cast<double>(
            plan.necklaces[index].labelled_weight
        ) * std::norm(samples[index]);
    }
    return result;
}

void dephased_free(
    std::vector<double> &probabilities,
    const Plan &plan,
    int chirp,
    DephasedStats &stats
) {
    std::vector<double> output(probabilities.size(), 0.0);
    Stats transition_stats;
    for (
        std::size_t target = 0;
        target < plan.necklaces.size();
        ++target
    ) {
        const double target_weight = static_cast<double>(
            plan.necklaces[target].labelled_weight
        );
        double value = 0.0;
        for (
            std::size_t source = 0;
            source < plan.necklaces.size();
            ++source
        ) {
            const Complex coefficient = transition_coefficient(
                plan,
                plan.necklaces[target],
                plan.necklaces[source],
                chirp,
                false,
                transition_stats
            );
            const double source_weight = static_cast<double>(
                plan.necklaces[source].labelled_weight
            );
            value += target_weight / source_weight
                * std::norm(coefficient)
                * probabilities[source];
        }
        output[target] = value;
    }
    probabilities.swap(output);
    stats.streamed_transition_coefficients +=
        transition_stats.streamed_transition_coefficients;
    stats.permanent_assignment_terms +=
        transition_stats.exact_cyclotomic_permanent_terms;

    const double probability_sum = std::accumulate(
        probabilities.begin(), probabilities.end(), 0.0
    );
    stats.maximum_probability_sum_error = std::max(
        stats.maximum_probability_sum_error,
        std::fabs(probability_sum - 1.0)
    );
}

Boundary project_probabilities(
    const std::vector<double> &probabilities,
    const Plan &plan
) {
    Boundary result{};
    for (
        std::size_t index = 0;
        index < probabilities.size();
        ++index
    ) {
        result[plan.necklaces[index].collisions] +=
            probabilities[index];
    }
    return result;
}

Boundary dephased_forward(
    const std::vector<Complex> &initial,
    const Plan &plan,
    int depth,
    int program_tag,
    DephasedStats &stats
) {
    const auto begin = std::chrono::steady_clock::now();
    std::vector<double> probabilities =
        probabilities_from_carrier(initial, plan);
    for (int step = 0; step < depth; ++step) {
        /*
         * Complete dephasing makes the diagonal collision phase invisible.
         * Count every erased phase application explicitly.
         */
        stats.collision_phases_erased += probabilities.size();
        dephased_free(
            probabilities,
            plan,
            public_chirp(step, program_tag),
            stats
        );
    }
    stats.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin
    ).count();
    return project_probabilities(probabilities, plan);
}

}  // namespace

int main() {
    const Plan plan = compile_plan();
    const std::vector<Complex> initial = make_carrier(plan, 0);

    std::vector<Complex> coherent_carrier = initial;
    const GeneratorRun coherent = generator_transaction(
        coherent_carrier,
        initial,
        plan,
        kPrimaryDepth,
        0,
        Control::Correct
    );
    const GeneratorRun coherent_reuse = generator_transaction(
        coherent_carrier,
        initial,
        plan,
        2,
        3,
        Control::Correct
    );

    /*
     * The best matched classical simulator is the same compact complex
     * recurrence. Execute a second warm arm to establish exact output and
     * resource parity without inventing a weaker classical representation.
     */
    std::vector<Complex> classical_carrier = initial;
    const GeneratorRun classical = generator_transaction(
        classical_carrier,
        initial,
        plan,
        kPrimaryDepth,
        0,
        Control::Correct
    );
    const GeneratorRun classical_reuse = generator_transaction(
        classical_carrier,
        initial,
        plan,
        2,
        3,
        Control::Correct
    );
    const double reuse_boundary_error = boundary_distance(
        coherent_reuse.boundary, classical_reuse.boundary
    );

    const double coherent_classical_boundary_error =
        boundary_distance(coherent.boundary, classical.boundary);
    if (
        coherent_classical_boundary_error != 0.0
        || coherent.restoration_error > kRestorationTolerance
        || classical.restoration_error > kRestorationTolerance
        || coherent_reuse.restoration_error > kRestorationTolerance
        || classical_reuse.restoration_error > kRestorationTolerance
        || reuse_boundary_error > kGeneratorTolerance
    ) {
        fail("coherent/classical matched arm gate failed");
    }

    const std::vector<Complex> snapshot = initial;
    DephasedStats dephased_stats;
    const Boundary dephased = dephased_forward(
        initial,
        plan,
        kPrimaryDepth,
        0,
        dephased_stats
    );
    const double coherence_boundary_effect =
        boundary_distance(coherent.boundary, dephased);
    if (
        coherence_boundary_effect < 1.0e-5
        || dephased_stats.maximum_probability_sum_error > 3.0e-11
    ) {
        fail("coherence ablation did not separate");
    }

    /*
     * Snapshot reload is measured as the sham recovery path. It is deliberately
     * not accepted as inverse restoration.
     */
    std::vector<Complex> sham_carrier = initial;
    sham_carrier = snapshot;
    const GeneratorRun snapshot_reuse = generator_transaction(
        sham_carrier,
        initial,
        plan,
        2,
        3,
        Control::Correct
    );
    if (snapshot_reuse.restoration_error > kRestorationTolerance) {
        fail("snapshot sham reuse gate failed");
    }

    std::vector<Complex> missing_carrier = initial;
    const GeneratorRun missing = generator_transaction(
        missing_carrier,
        initial,
        plan,
        2,
        0,
        Control::Missing
    );
    std::vector<Complex> wrong_carrier = initial;
    const GeneratorRun wrong = generator_transaction(
        wrong_carrier,
        initial,
        plan,
        2,
        0,
        Control::Wrong
    );
    std::vector<Complex> reordered_carrier = initial;
    const GeneratorRun reordered = generator_transaction(
        reordered_carrier,
        initial,
        plan,
        2,
        0,
        Control::Reordered
    );
    if (
        missing.restoration_error < kControlFloor
        || wrong.restoration_error < kControlFloor
        || reordered.restoration_error < kControlFloor
    ) {
        fail("coherence triad inverse controls failed");
    }

    const std::uint64_t carrier_bytes =
        plan.necklaces.size() * sizeof(Complex);
    const std::uint64_t probability_bytes =
        plan.necklaces.size() * sizeof(double);
    const std::uint64_t coherent_engine_bytes = 33470;
    const std::uint64_t coherent_lifecycle_bytes =
        coherent_engine_bytes + carrier_bytes
        + 2U * sizeof(Boundary);
    const std::uint64_t topology_bytes =
        plan.necklaces.capacity() * sizeof(Necklace)
        + plan.roots.size() * sizeof(Complex);
    const std::uint64_t transition_scratch_bytes =
        sizeof(std::array<std::int64_t, kGrid>)
        + 2U * sizeof(Histogram)
        + sizeof(Tuple)
        + sizeof(std::array<int, kRotors>)
        + 4U * sizeof(Complex);
    const std::uint64_t dephased_probability_payload_bytes =
        2U * probability_bytes;
    const std::uint64_t dephased_forward_explicit_payload_bytes =
        topology_bytes + carrier_bytes + carrier_bytes
        + dephased_probability_payload_bytes
        + transition_scratch_bytes + sizeof(Boundary);
    const std::uint64_t dephased_lifecycle_explicit_payload_bytes =
        coherent_engine_bytes + 2U * carrier_bytes
        + 2U * sizeof(Boundary);
    const std::uint64_t comparison_harness_explicit_payload_bytes =
        coherent_engine_bytes + 7U * carrier_bytes
        + 8U * sizeof(GeneratorRun)
        + sizeof(Boundary) + sizeof(DephasedStats);

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"BOUNDED_MATCHED_COHERENT_INITIAL_AND_EACH_STEP_NECKLACE_DEPHASED_CLASSICAL_GENERATOR_SMALL_WALL_RESOURCE_DIAGNOSTIC\","
    );
    std::printf(
        "\"claim_ceiling\":\"DIRECT_PROCESS_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_GRID17_FOUR_ROTOR_DEPTH8_COMPLEX128_SOFTWARE_ONLY\","
    );
    std::printf("\"result\":\"PASS\",");
    std::printf(
        "\"public_instance\":{"
        "\"grid\":17,\"rotors\":4,\"depth\":8,\"program_tag\":0,"
        "\"resident_necklace_cells\":285,"
        "\"boundary_values\":7},"
    );
    std::printf(
        "\"coherent_in_place\":{"
        "\"boundary\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"
        "\"restoration_error\":%.17g,"
        "\"reuse_restoration_error\":%.17g,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"timing_scope\":\"FORWARD_PROJECTION_ACTUAL_INVERSE\","
        "\"forward_projection_inverse_elapsed_ms\":%.17g,"
        "\"engine_explicit_payload_bytes\":%llu,"
        "\"verification_baseline_bytes\":%llu,"
        "\"projection_boundary_bytes\":%llu,"
        "\"lifecycle_explicit_payload_bytes\":%llu},",
        coherent.boundary[0],
        coherent.boundary[1],
        coherent.boundary[2],
        coherent.boundary[3],
        coherent.boundary[4],
        coherent.boundary[5],
        coherent.boundary[6],
        coherent.restoration_error,
        coherent_reuse.restoration_error,
        coherent.elapsed_ms,
        static_cast<unsigned long long>(coherent_engine_bytes),
        static_cast<unsigned long long>(carrier_bytes),
        static_cast<unsigned long long>(sizeof(Boundary)),
        static_cast<unsigned long long>(coherent_lifecycle_bytes)
    );
    std::printf(
        "\"dephased_snapshot_sham\":{"
        "\"basis\":\"GLOBAL_ROTATION_NECKLACE_ORBIT\","
        "\"initial_carrier_dephased\":true,"
        "\"dephased_after_each_free_step\":true,"
        "\"boundary\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],"
        "\"coherence_boundary_effect\":%.17g,"
        "\"maximum_probability_sum_error\":%.17g,"
        "\"collision_phase_updates_erased\":%llu,"
        "\"streamed_transition_coefficients\":%llu,"
        "\"permanent_assignment_terms\":%llu,"
        "\"timing_scope\":\"FORWARD_ONLY_EXCLUDES_SNAPSHOT_AND_REUSE\","
        "\"forward_only_elapsed_ms\":%.17g,"
        "\"lifecycle_timing_measured\":false,"
        "\"probability_payload_bytes\":%llu,"
        "\"public_topology_bytes\":%llu,"
        "\"transition_scratch_bytes\":%llu,"
        "\"forward_explicit_payload_bytes\":%llu,"
        "\"lifecycle_explicit_payload_bytes\":%llu,"
        "\"snapshot_creation_bytes\":%llu,"
        "\"snapshot_reload_bytes\":%llu,"
        "\"actual_inverse_restoration\":false,"
        "\"snapshot_backed_reuse_only\":true,"
        "\"reuse_restoration_error_after_reload\":%.17g},",
        dephased[0],
        dephased[1],
        dephased[2],
        dephased[3],
        dephased[4],
        dephased[5],
        dephased[6],
        coherence_boundary_effect,
        dephased_stats.maximum_probability_sum_error,
        static_cast<unsigned long long>(
            dephased_stats.collision_phases_erased
        ),
        static_cast<unsigned long long>(
            dephased_stats.streamed_transition_coefficients
        ),
        static_cast<unsigned long long>(
            dephased_stats.permanent_assignment_terms
        ),
        dephased_stats.elapsed_ms,
        static_cast<unsigned long long>(
            dephased_probability_payload_bytes
        ),
        static_cast<unsigned long long>(
            topology_bytes
        ),
        static_cast<unsigned long long>(
            transition_scratch_bytes
        ),
        static_cast<unsigned long long>(
            dephased_forward_explicit_payload_bytes
        ),
        static_cast<unsigned long long>(
            dephased_lifecycle_explicit_payload_bytes
        ),
        static_cast<unsigned long long>(carrier_bytes),
        static_cast<unsigned long long>(carrier_bytes),
        snapshot_reuse.restoration_error
    );
    std::printf(
        "\"matched_compact_classical\":{"
        "\"same_executable_recurrence\":true,"
        "\"boundary_error\":%.17g,"
        "\"reuse_boundary_error\":%.17g,"
        "\"restoration_error\":%.17g,"
        "\"reuse_restoration_error\":%.17g,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"timing_scope\":\"FORWARD_PROJECTION_ACTUAL_INVERSE\","
        "\"forward_projection_inverse_elapsed_ms\":%.17g,"
        "\"engine_explicit_payload_bytes\":%llu,"
        "\"verification_baseline_bytes\":%llu,"
        "\"projection_boundary_bytes\":%llu,"
        "\"lifecycle_explicit_payload_bytes\":%llu},",
        coherent_classical_boundary_error,
        reuse_boundary_error,
        classical.restoration_error,
        classical_reuse.restoration_error,
        classical.elapsed_ms,
        static_cast<unsigned long long>(coherent_engine_bytes),
        static_cast<unsigned long long>(carrier_bytes),
        static_cast<unsigned long long>(sizeof(Boundary)),
        static_cast<unsigned long long>(coherent_lifecycle_bytes)
    );
    std::printf(
        "\"traffic\":{"
        "\"protocol_used\":false,"
        "\"logical_protocol_bytes_per_arm\":0},"
        "\"comparison_harness\":{"
        "\"conservative_explicit_payload_bytes\":%llu},"
        "\"controls\":{"
        "\"missing_inverse_error\":%.17g,"
        "\"wrong_inverse_error\":%.17g,"
        "\"reordered_inverse_error\":%.17g},"
        "\"diagnosis\":{"
        "\"initial_and_interstep_necklace_coherence_changes_boundary\":true,"
        "\"collision_phase_contribution_isolated\":false,"
        "\"matched_classical_inherits_coherence_exactly\":true,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"unbounded_computation_established\":false},"
        "\"obstruction\":\"COHERENCE_IS_CAUSAL_BUT_IDENTICAL_COMPACT_CLASSICAL_COMPLEX_RECURRENCE_INHERITS_IT\","
        "\"terminal\":false",
        static_cast<unsigned long long>(
            comparison_harness_explicit_payload_bytes
        ),
        missing.restoration_error,
        wrong.restoration_error,
        reordered.restoration_error
    );
    std::printf("}\n");
    return 0;
}
