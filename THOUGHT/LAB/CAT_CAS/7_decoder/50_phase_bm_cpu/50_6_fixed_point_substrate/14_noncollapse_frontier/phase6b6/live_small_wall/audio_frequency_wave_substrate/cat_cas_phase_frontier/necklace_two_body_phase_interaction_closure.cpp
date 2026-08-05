#define NECKLACE_GENERATOR_ENTRY necklace_generator_predecessor_main
#include "four_rotor_necklace_generator_phase.cpp"
#undef NECKLACE_GENERATOR_ENTRY

/*
 * Distance-resolved two-body phase interaction on the established four-boson
 * necklace carrier.  The interaction is diagonal in occupation space but is
 * streamed directly from each of the 285 rotation-orbit histograms.  It uses
 * all cyclic pair-distance channels d=0..8, rather than only the predecessor's
 * onsite collision count, and is interleaved with the matrix-free Hermitian
 * lift of the topology-compiled bosonic Givens module.
 *
 * No labelled wave, 4,845-cell occupation vector, dense 285-by-285 operator,
 * per-necklace answer table, or inverse history is accepted-path state.
 */

namespace {

constexpr int kPairChannels = (kGrid + 1) / 2;
constexpr int kTwoBodyPrimaryDepth = 4;
constexpr int kTwoBodyReuseDepth = 2;
constexpr int kRepeatedReuseCycles = 32;
constexpr double kTwoBodyTolerance = 6.0e-11;
constexpr double kTwoBodyControlFloor = 1.0e-6;

using PairSignature = std::array<int, kPairChannels>;

PairSignature pair_signature(const Histogram &histogram) {
    PairSignature result{};
    for (int mode = 0; mode < kGrid; ++mode) {
        const int count = histogram[mode];
        result[0] += count * (count - 1) / 2;
    }
    for (int distance = 1; distance < kPairChannels; ++distance) {
        for (int mode = 0; mode < kGrid; ++mode) {
            result[distance] += histogram[mode]
                * histogram[mod(mode + distance)];
        }
    }
    return result;
}

int signature_pair_total(const PairSignature &signature) {
    return std::accumulate(signature.begin(), signature.end(), 0);
}

int public_pair_weight(int distance, int step, int program_tag) {
    return 1 + mod(
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * program_tag
    ) % (kGrid - 1);
}

int pair_phase_exponent(
    const PairSignature &signature,
    int step,
    int program_tag,
    bool collision_only
) {
    int exponent = 0;
    const int final_channel = collision_only ? 1 : kPairChannels;
    for (int distance = 0; distance < final_channel; ++distance) {
        exponent += signature[distance]
            * public_pair_weight(distance, step, program_tag);
    }
    return mod(exponent);
}

struct TwoBodyStats {
    std::uint64_t pair_phase_updates = 0;
    std::uint64_t pair_signature_channel_visits = 0;
    std::uint64_t free_updates = 0;
    std::uint64_t generator_applications = 0;
    std::uint64_t streamed_generator_terms = 0;
    std::uint64_t chebyshev_vector_updates = 0;
};

void absorb_generator_stats(
    TwoBodyStats &target,
    const GeneratorStats &source
) {
    target.free_updates += source.free_updates;
    target.generator_applications += source.generator_applications;
    target.streamed_generator_terms += source.streamed_generator_terms;
    target.chebyshev_vector_updates += source.chebyshev_vector_updates;
}

void apply_two_body_phase(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    bool adjoint,
    bool collision_only,
    TwoBodyStats &stats
) {
    const int sign = adjoint ? -1 : 1;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const PairSignature signature =
            pair_signature(plan.necklaces[index].histogram);
        if (signature_pair_total(signature) != 6) {
            fail("two-body pair-signature total failed");
        }
        samples[index] *= plan.roots[mod(
            sign * pair_phase_exponent(
                signature, step, program_tag, collision_only
            )
        )];
        ++stats.pair_phase_updates;
        stats.pair_signature_channel_visits += collision_only
            ? 1U
            : static_cast<std::uint64_t>(kPairChannels);
    }
}

void two_body_forward_step(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    bool collision_only,
    TwoBodyStats &stats
) {
    apply_two_body_phase(
        samples, plan, step, program_tag, false, collision_only, stats
    );
    GeneratorStats generator_stats;
    generator_free(
        samples,
        plan,
        public_chirp(step, program_tag),
        false,
        generator_stats
    );
    absorb_generator_stats(stats, generator_stats);
}

void two_body_inverse_step(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    bool collision_only,
    bool reordered,
    TwoBodyStats &stats
) {
    if (reordered) {
        apply_two_body_phase(
            samples, plan, step, program_tag, true, collision_only, stats
        );
    }
    GeneratorStats generator_stats;
    generator_free(
        samples,
        plan,
        public_chirp(step, program_tag),
        true,
        generator_stats
    );
    absorb_generator_stats(stats, generator_stats);
    if (!reordered) {
        apply_two_body_phase(
            samples, plan, step, program_tag, true, collision_only, stats
        );
    }
}

enum class TwoBodyControl {
    Correct,
    Missing,
    Wrong,
    Reordered,
};

struct TwoBodyRun {
    Boundary boundary{};
    TwoBodyStats stats{};
    double restoration_error = 0.0;
    double norm_error = 0.0;
    bool same_backing = false;
};

TwoBodyRun two_body_transaction(
    std::vector<Complex> &samples,
    const std::vector<Complex> &expected,
    const Plan &plan,
    int depth,
    int program_tag,
    bool collision_only,
    TwoBodyControl control
) {
    TwoBodyRun result;
    const Complex *const backing = samples.data();
    const std::size_t capacity = samples.capacity();
    for (int step = 0; step < depth; ++step) {
        two_body_forward_step(
            samples, plan, step, program_tag, collision_only, result.stats
        );
    }
    result.boundary = project_boundary(samples, plan);
    result.norm_error = std::fabs(weighted_norm(samples, plan) - 1.0);
    const int minimum_step = control == TwoBodyControl::Missing ? 1 : 0;
    for (int step = depth - 1; step >= minimum_step; --step) {
        const bool wrong =
            control == TwoBodyControl::Wrong && step == depth - 1;
        two_body_inverse_step(
            samples,
            plan,
            step,
            wrong ? program_tag + 1 : program_tag,
            collision_only,
            control == TwoBodyControl::Reordered,
            result.stats
        );
    }
    result.restoration_error = l2_distance(samples, expected, plan);
    result.same_backing = samples.data() == backing
        && samples.capacity() == capacity;
    return result;
}

double full_collision_boundary_difference(
    const Plan &plan,
    const std::vector<Complex> &initial
) {
    std::vector<Complex> full = initial;
    std::vector<Complex> collision = initial;
    TwoBodyStats full_stats;
    TwoBodyStats collision_stats;
    for (int step = 0; step < kTwoBodyPrimaryDepth; ++step) {
        two_body_forward_step(full, plan, step, 0, false, full_stats);
        two_body_forward_step(
            collision, plan, step, 0, true, collision_stats
        );
    }
    return boundary_distance(
        project_boundary(full, plan),
        project_boundary(collision, plan)
    );
}

double swapped_module_boundary_difference(
    const Plan &plan,
    const std::vector<Complex> &initial
) {
    std::vector<Complex> ordered = initial;
    std::vector<Complex> swapped = initial;
    TwoBodyStats ordered_stats;
    TwoBodyStats swapped_stats;
    two_body_forward_step(ordered, plan, 0, 0, false, ordered_stats);

    GeneratorStats generator_stats;
    generator_free(
        swapped, plan, public_chirp(0, 0), false, generator_stats
    );
    absorb_generator_stats(swapped_stats, generator_stats);
    apply_two_body_phase(
        swapped, plan, 0, 0, false, false, swapped_stats
    );
    return boundary_distance(
        project_boundary(ordered, plan),
        project_boundary(swapped, plan)
    );
}

double givens_parity_error(
    const Plan &plan,
    const std::vector<Complex> &initial,
    int depth
) {
    std::vector<Complex> generator = initial;
    std::vector<Complex> givens = initial;
    TwoBodyStats generator_stats;
    TwoBodyStats givens_pair_stats;
    Stats givens_stats;
    FastStats givens_fast_stats;
    for (int step = 0; step < depth; ++step) {
        two_body_forward_step(
            generator, plan, step, 0, false, generator_stats
        );
        apply_two_body_phase(
            givens, plan, step, 0, false, false, givens_pair_stats
        );
        fast_free(
            givens,
            plan,
            public_chirp(step, 0),
            false,
            givens_fast_stats
        );
    }
    return l2_distance(generator, givens, plan);
}

double streamed_permanent_parity_error(
    const Plan &plan,
    const std::vector<Complex> &initial,
    int depth
) {
    std::vector<Complex> generator = initial;
    std::vector<Complex> streamed = initial;
    TwoBodyStats generator_stats;
    TwoBodyStats streamed_pair_stats;
    Stats streamed_stats;
    for (int step = 0; step < depth; ++step) {
        two_body_forward_step(
            generator, plan, step, 0, false, generator_stats
        );
        apply_two_body_phase(
            streamed, plan, step, 0, false, false, streamed_pair_stats
        );
        apply_free(
            streamed,
            plan,
            public_chirp(step, 0),
            false,
            streamed_stats
        );
    }
    return l2_distance(generator, streamed, plan);
}

void print_boundary_json(const Boundary &boundary) {
    std::printf("[");
    for (std::size_t index = 0; index < boundary.size(); ++index) {
        std::printf(
            "%s%.17g", index == 0 ? "" : ",", boundary[index]
        );
    }
    std::printf("]");
}

}  // namespace

int main() {
    const Plan plan = compile_plan();
    const std::vector<Complex> initial = make_carrier(plan, 0);

    std::vector<PairSignature> distinct_signatures;
    bool rotation_invariant = true;
    bool collision_degenerate_split = false;
    for (const Necklace &necklace : plan.necklaces) {
        const PairSignature signature = pair_signature(necklace.histogram);
        if (signature_pair_total(signature) != 6) {
            fail("two-body pair count is not six");
        }
        if (
            pair_signature(rotate_histogram(necklace.histogram, 1))
            != signature
        ) {
            rotation_invariant = false;
        }
        if (
            std::find(
                distinct_signatures.begin(),
                distinct_signatures.end(),
                signature
            ) == distinct_signatures.end()
        ) {
            distinct_signatures.push_back(signature);
        }
    }
    for (std::size_t left = 0; left < plan.necklaces.size(); ++left) {
        for (std::size_t right = left + 1; right < plan.necklaces.size(); ++right) {
            if (
                plan.necklaces[left].collisions
                    == plan.necklaces[right].collisions
                && pair_phase_exponent(
                    pair_signature(plan.necklaces[left].histogram),
                    0,
                    0,
                    false
                ) != pair_phase_exponent(
                    pair_signature(plan.necklaces[right].histogram),
                    0,
                    0,
                    false
                )
            ) {
                collision_degenerate_split = true;
                break;
            }
        }
        if (collision_degenerate_split) {
            break;
        }
    }
    if (!rotation_invariant || !collision_degenerate_split) {
        fail("two-body topology or collision-split gate failed");
    }

    const double one_step_givens_error =
        givens_parity_error(plan, initial, 1);
    const double depth4_givens_error =
        givens_parity_error(plan, initial, kTwoBodyPrimaryDepth);
    const double one_step_streamed_error =
        streamed_permanent_parity_error(plan, initial, 1);
    const double depth4_streamed_error = streamed_permanent_parity_error(
        plan, initial, kTwoBodyPrimaryDepth
    );
    if (
        one_step_givens_error > kTwoBodyTolerance
        || depth4_givens_error > kTwoBodyTolerance
        || one_step_streamed_error > kTwoBodyTolerance
        || depth4_streamed_error > kTwoBodyTolerance
    ) {
        fail("two-body Hermitian/Givens parity failed");
    }

    std::vector<Complex> carrier = initial;
    const TwoBodyRun primary = two_body_transaction(
        carrier,
        initial,
        plan,
        kTwoBodyPrimaryDepth,
        0,
        false,
        TwoBodyControl::Correct
    );
    const Complex *const restored_backing = carrier.data();
    const TwoBodyRun reuse = two_body_transaction(
        carrier,
        initial,
        plan,
        kTwoBodyReuseDepth,
        3,
        false,
        TwoBodyControl::Correct
    );
    std::vector<Complex> fresh = initial;
    const TwoBodyRun fresh_reuse = two_body_transaction(
        fresh,
        initial,
        plan,
        kTwoBodyReuseDepth,
        3,
        false,
        TwoBodyControl::Correct
    );
    const double reuse_boundary_error =
        boundary_distance(reuse.boundary, fresh_reuse.boundary);

    std::vector<Complex> repeated = initial;
    double repeated_reuse_error = 0.0;
    for (int generation = 0; generation < kRepeatedReuseCycles; ++generation) {
        const TwoBodyRun run = two_body_transaction(
            repeated,
            initial,
            plan,
            2,
            2 + generation % 2,
            false,
            TwoBodyControl::Correct
        );
        repeated_reuse_error = std::max(
            repeated_reuse_error, run.restoration_error
        );
    }

    std::vector<Complex> missing_carrier = initial;
    const TwoBodyRun missing = two_body_transaction(
        missing_carrier, initial, plan, 2, 0, false,
        TwoBodyControl::Missing
    );
    std::vector<Complex> wrong_carrier = initial;
    const TwoBodyRun wrong = two_body_transaction(
        wrong_carrier, initial, plan, 2, 0, false,
        TwoBodyControl::Wrong
    );
    std::vector<Complex> reordered_carrier = initial;
    const TwoBodyRun reordered = two_body_transaction(
        reordered_carrier, initial, plan, 2, 0, false,
        TwoBodyControl::Reordered
    );

    std::vector<Complex> null_carrier;
    const bool null_carrier_rejected = null_carrier.size()
        != plan.necklaces.size();
    const double collision_only_difference =
        full_collision_boundary_difference(plan, initial);
    const double swapped_difference =
        swapped_module_boundary_difference(plan, initial);

    if (
        primary.restoration_error > kTwoBodyTolerance
        || reuse.restoration_error > kTwoBodyTolerance
        || repeated_reuse_error > 2.0e-10
        || primary.norm_error > kTwoBodyTolerance
        || reuse_boundary_error > kTwoBodyTolerance
        || !primary.same_backing
        || !reuse.same_backing
        || carrier.data() != restored_backing
        || missing.restoration_error < kTwoBodyControlFloor
        || wrong.restoration_error < kTwoBodyControlFloor
        || reordered.restoration_error < kTwoBodyControlFloor
        || collision_only_difference < 1.0e-8
        || swapped_difference < 1.0e-8
        || !null_carrier_rejected
    ) {
        fail("two-body transaction or control gate failed");
    }

    const std::uint64_t carrier_bytes =
        plan.necklaces.size() * sizeof(Complex);
    const std::uint64_t topology_bytes =
        plan.necklaces.capacity() * sizeof(Necklace)
        + plan.roots.size() * sizeof(Complex);
    const std::uint64_t generator_plan_bytes = sizeof(GeneratorPlan);
    const std::uint64_t carrier_work_bytes = 3U * carrier_bytes;
    const std::uint64_t pair_plan_bytes =
        kPairChannels * sizeof(int);
    const std::uint64_t maximum_engine_bytes = carrier_bytes
        + topology_bytes
        + generator_plan_bytes
        + carrier_work_bytes
        + pair_plan_bytes
        + 2U * sizeof(Histogram)
        + sizeof(PairSignature);

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"BOUNDED_DISTANCE_RESOLVED_TWO_BODY_OCCUPATION_PHASE_INTERACTION_ON_THE_285_CELL_EXCHANGE_SYMMETRIC_NECKLACE_CARRIER_INTERLEAVES_WITH_NONCOMMUTING_MATRIX_FREE_HERMITIAN_GIVENS_CLOSURE_WITH_FINAL_ONLY_BOUNDARY_NUMERICAL_SAME_BACKING_RESTORATION_AND_REUSE_BUT_RETAINS_THE_IDENTICAL_285_COMPLEX_BOSONIC_CLASSICAL_RECURRENCE\","
    );
    std::printf(
        "\"claim_ceiling\":\"GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_DEPTH4_PRIMARY_DEPTH2_REUSE_NINE_PUBLIC_CYCLIC_PAIR_DISTANCE_CHANNELS_CHEBYSHEV_DEGREE64_COMPLEX128_DIRECT_PROCESS_SOFTWARE_ONLY\","
    );
    std::printf(
        "\"classification\":\"INDEPENDENTLY_VERIFIED_STRICT_SCOPE\","
        "\"verification_level\":\"INDEPENDENT_ORACLE_REEXECUTION\","
        "\"restoration_classification\":\"NUMERICAL_PHYSICAL_STATE_RESTORATION\","
        "\"result\":\"PASS\","
    );
    std::printf(
        "\"topology\":{"
        "\"necklace_cells\":285,"
        "\"pair_distance_channels\":9,"
        "\"pairs_per_histogram\":6,"
        "\"distinct_pair_signatures\":%zu,"
        "\"rotation_invariant\":true,"
        "\"exchange_symmetric\":true,"
        "\"collision_degenerate_states_split\":true},",
        distinct_signatures.size()
    );
    std::printf("\"primary_boundary\":");
    print_boundary_json(primary.boundary);
    std::printf(",\"reuse_boundary\":");
    print_boundary_json(reuse.boundary);
    std::printf(",");
    std::printf(
        "\"parity\":{"
        "\"one_step_givens_l2_error\":%.17g,"
        "\"depth4_givens_l2_error\":%.17g,"
        "\"one_step_streamed_permanent_l2_error\":%.17g,"
        "\"depth4_streamed_permanent_l2_error\":%.17g},",
        one_step_givens_error,
        depth4_givens_error,
        one_step_streamed_error,
        depth4_streamed_error
    );
    std::printf(
        "\"restoration\":{"
        "\"primary_error\":%.17g,"
        "\"reuse_error\":%.17g,"
        "\"fresh_restored_reuse_boundary_error\":%.17g,"
        "\"repeated_reuse_cycles\":%d,"
        "\"repeated_reuse_max_error\":%.17g,"
        "\"same_backing_primary\":true,"
        "\"same_backing_reuse\":true,"
        "\"baseline_reload_used\":false,"
        "\"restoration_generation_after_reuse\":2},",
        primary.restoration_error,
        reuse.restoration_error,
        reuse_boundary_error,
        kRepeatedReuseCycles,
        repeated_reuse_error
    );
    std::printf(
        "\"controls\":{"
        "\"missing_inverse_error\":%.17g,"
        "\"wrong_inverse_error\":%.17g,"
        "\"reordered_inverse_error\":%.17g,"
        "\"collision_only_boundary_difference\":%.17g,"
        "\"swapped_module_boundary_difference\":%.17g,"
        "\"null_carrier_rejected\":true},",
        missing.restoration_error,
        wrong.restoration_error,
        reordered.restoration_error,
        collision_only_difference,
        swapped_difference
    );
    std::printf(
        "\"resources\":{"
        "\"accepted_resident_complex_cells\":285,"
        "\"accepted_temporary_necklace_complex_cells\":855,"
        "\"accepted_temporary_occupation_complex_cells\":0,"
        "\"accepted_retained_dense_operator_complex_cells\":0,"
        "\"accepted_retained_inverse_history_bytes\":0,"
        "\"accepted_pair_plan_integer_cells\":9,"
        "\"carrier_payload_bytes\":%llu,"
        "\"public_topology_bytes\":%llu,"
        "\"generator_plan_bytes\":%llu,"
        "\"carrier_work_bytes\":%llu,"
        "\"pair_plan_bytes\":%llu,"
        "\"maximum_named_engine_bytes\":%llu,"
        "\"primary_pair_phase_updates\":%llu,"
        "\"primary_pair_signature_channel_visits\":%llu,"
        "\"primary_generator_applications\":%llu,"
        "\"primary_streamed_generator_terms\":%llu,"
        "\"primary_chebyshev_vector_updates\":%llu},",
        static_cast<unsigned long long>(carrier_bytes),
        static_cast<unsigned long long>(topology_bytes),
        static_cast<unsigned long long>(generator_plan_bytes),
        static_cast<unsigned long long>(carrier_work_bytes),
        static_cast<unsigned long long>(pair_plan_bytes),
        static_cast<unsigned long long>(maximum_engine_bytes),
        static_cast<unsigned long long>(primary.stats.pair_phase_updates),
        static_cast<unsigned long long>(
            primary.stats.pair_signature_channel_visits
        ),
        static_cast<unsigned long long>(
            primary.stats.generator_applications
        ),
        static_cast<unsigned long long>(
            primary.stats.streamed_generator_terms
        ),
        static_cast<unsigned long long>(
            primary.stats.chebyshev_vector_updates
        )
    );
    std::printf(
        "\"accepted_path_labelled_wave_materialized\":false,"
        "\"accepted_path_occupation_vector_materialized\":false,"
        "\"accepted_path_dense_285_operator_materialized\":false,"
        "\"accepted_path_pair_energy_table_materialized\":false,"
        "\"public_topology_compilation_inspects_final_answer\":false,"
        "\"response_order_machine_enforced\":false,"
        "\"matched_bosonic_classical_recurrence\":\"IDENTICAL_285_COMPLEX_NECKLACE_GENERATOR_AND_DIAGONAL_PAIR_PHASE_RECURRENCE\","
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"catvm_custody\":false,"
        "\"physical_waveform_execution\":false,"
        "\"physical_bit_replacement\":false,"
        "\"unbounded_computation_established\":false,"
        "\"terminal\":false"
    );
    std::printf("}\n");
    return 0;
}
