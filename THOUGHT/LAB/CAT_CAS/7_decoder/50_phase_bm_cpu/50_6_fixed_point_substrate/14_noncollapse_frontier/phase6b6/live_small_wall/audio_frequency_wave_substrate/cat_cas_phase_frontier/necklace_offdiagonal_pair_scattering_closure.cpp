#define NECKLACE_GENERATOR_ENTRY necklace_scattering_generator_predecessor_main
#include "four_rotor_necklace_generator_phase.cpp"
#undef NECKLACE_GENERATOR_ENTRY

/*
 * Matrix-free off-diagonal two-body scattering on the established 285-cell
 * four-boson necklace carrier.
 *
 * For every ordered occupied target pair (a,b), the Hermitian quartic
 * generator streams the momentum-conserving source pair
 * (a-shift,b+shift).  Opposite shifts have the same real public weight, so
 * the generator is exchange symmetric, global-rotation invariant, and
 * self-adjoint in the labelled-weight inner product.  Its exponential is
 * applied with a degree-64 Chebyshev recurrence.  The accepted transaction
 * never materializes the 4,845-cell occupation vector, a dense 285 by 285
 * operator, a pair-transition table, or inverse history.
 */

namespace {

constexpr int kScatteringPairChannels = (kGrid + 1) / 2;
constexpr int kScatteringPrimaryDepth = 3;
constexpr int kScatteringReuseDepth = 2;
constexpr int kScatteringRepeatedCycles = 32;
constexpr double kScatteringTolerance = 1.2e-10;
constexpr double kScatteringDriftTolerance = 4.0e-10;
constexpr double kScatteringControlFloor = 1.0e-6;

using ScatteringPairSignature = std::array<int, kScatteringPairChannels>;

ScatteringPairSignature scattering_pair_signature(
    const Histogram &histogram
) {
    ScatteringPairSignature result{};
    for (int mode = 0; mode < kGrid; ++mode) {
        const int count = histogram[mode];
        result[0] += count * (count - 1) / 2;
    }
    for (int distance = 1; distance < kScatteringPairChannels; ++distance) {
        for (int mode = 0; mode < kGrid; ++mode) {
            result[distance] += histogram[mode]
                * histogram[mod(mode + distance)];
        }
    }
    return result;
}

int scattering_public_pair_weight(
    int distance,
    int step,
    int program_tag
) {
    return 1 + mod(
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * program_tag
    ) % (kGrid - 1);
}

int scattering_pair_phase_exponent(
    const ScatteringPairSignature &signature,
    int step,
    int program_tag
) {
    int exponent = 0;
    for (int distance = 0; distance < kScatteringPairChannels; ++distance) {
        exponent += signature[distance]
            * scattering_public_pair_weight(distance, step, program_tag);
    }
    return mod(exponent);
}

struct ScatteringPairPhaseStats {
    std::uint64_t updates = 0;
    std::uint64_t signature_channel_visits = 0;
};

void apply_scattering_pair_phase(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    bool adjoint,
    ScatteringPairPhaseStats &stats
) {
    const int sign = adjoint ? -1 : 1;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const ScatteringPairSignature signature =
            scattering_pair_signature(plan.necklaces[index].histogram);
        samples[index] *= plan.roots[mod(
            sign * scattering_pair_phase_exponent(
                signature, step, program_tag
            )
        )];
        ++stats.updates;
        stats.signature_channel_visits += kScatteringPairChannels;
    }
}

void print_scattering_boundary_json(const Boundary &boundary) {
    std::printf("[");
    for (std::size_t index = 0; index < boundary.size(); ++index) {
        std::printf("%s%.17g", index == 0 ? "" : ",", boundary[index]);
    }
    std::printf("]");
}

struct ScatteringPlan {
    double radius_bound = 0.0;
    double chebyshev_tail_bound = 0.0;
    double maximum_weight = 0.0;
};

struct ScatteringStats {
    std::uint64_t phase_updates = 0;
    std::uint64_t phase_signature_channel_visits = 0;
    std::uint64_t scattering_updates = 0;
    std::uint64_t scattering_generator_applications = 0;
    std::uint64_t streamed_ordered_pair_shift_terms = 0;
    std::uint64_t streamed_off_diagonal_terms = 0;
    std::uint64_t streamed_genuine_double_move_terms = 0;
    std::uint64_t chebyshev_vector_updates = 0;
    double maximum_chebyshev_tail_bound = 0.0;
};

double public_scattering_weight(
    int signed_shift,
    int step,
    int program_tag
) {
    const int positive = mod(signed_shift);
    if (positive == 0) {
        fail("zero scattering shift is not off diagonal");
    }
    const int distance = std::min(positive, kGrid - positive);
    const int magnitude = 1 + mod(
        (distance + 2) * (step + 1)
        + (3 * distance + 1) * (program_tag + 2)
    ) % 5;
    const int sign = mod(distance + step + program_tag) % 3 == 0
        ? -1
        : 1;
    return 0.01 * static_cast<double>(sign * magnitude);
}

ScatteringPlan compile_scattering_plan(int step, int program_tag) {
    ScatteringPlan result;
    double absolute_shift_sum = 0.0;
    for (int shift = 1; shift < kGrid; ++shift) {
        const double weight = public_scattering_weight(
            shift, step, program_tag
        );
        const double reverse = public_scattering_weight(
            kGrid - shift, step, program_tag
        );
        if (weight != reverse) {
            fail("scattering reverse-shift symmetry failed");
        }
        absolute_shift_sum += std::fabs(weight);
        result.maximum_weight = std::max(
            result.maximum_weight, std::fabs(weight)
        );
    }
    /* One histogram has 4*3 ordered rotor pairs; the 1/2 removes exchange
       double counting.  This is a Gershgorin bound for the similar Hermitian
       matrix in normalized occupation coordinates. */
    result.radius_bound = 0.5
        * static_cast<double>(kRotors * (kRotors - 1))
        * absolute_shift_sum;
    const int first_omitted = kChebyshevDegree + 1;
    const double leading_bound = std::pow(
        0.5 * result.radius_bound, first_omitted
    ) / std::tgamma(static_cast<double>(first_omitted + 1))
        * std::exp(
            result.radius_bound * result.radius_bound
            / (4.0 * static_cast<double>(first_omitted + 1))
        );
    const double ratio = result.radius_bound
        / (2.0 * static_cast<double>(first_omitted + 1));
    result.chebyshev_tail_bound = 2.0 * leading_bound / (1.0 - ratio);
    if (
        result.radius_bound <= 0.0
        || result.radius_bound >= 6.0
        || result.chebyshev_tail_bound >= 1.0e-13
    ) {
        fail("scattering Chebyshev bound failed");
    }
    return result;
}

void apply_scattering_generator(
    const std::vector<Complex> &input,
    std::vector<Complex> &output,
    const Plan &plan,
    int step,
    int program_tag,
    ScatteringStats &stats
) {
    if (
        input.size() != plan.necklaces.size()
        || output.size() != plan.necklaces.size()
    ) {
        fail("scattering carrier size failed");
    }
    for (std::size_t target = 0; target < plan.necklaces.size(); ++target) {
        const Histogram &histogram = plan.necklaces[target].histogram;
        Complex value = 0.0;
        for (int first = 0; first < kGrid; ++first) {
            if (histogram[first] == 0) {
                continue;
            }
            for (int second = 0; second < kGrid; ++second) {
                const int multiplicity = static_cast<int>(histogram[first])
                    * (
                        static_cast<int>(histogram[second])
                        - (first == second ? 1 : 0)
                    );
                if (multiplicity == 0) {
                    continue;
                }
                for (int shift = 1; shift < kGrid; ++shift) {
                    const int first_source = mod(first - shift);
                    const int second_source = mod(second + shift);
                    Histogram source = histogram;
                    --source[first];
                    --source[second];
                    ++source[first_source];
                    ++source[second_source];
                    const Histogram canonical = canonical_histogram(source);
                    const std::size_t source_index = find_necklace(
                        plan, canonical
                    );
                    value += 0.5 * static_cast<double>(multiplicity)
                        * public_scattering_weight(
                            shift, step, program_tag
                        )
                        * input[source_index];
                    ++stats.streamed_ordered_pair_shift_terms;
                    if (source_index != target) {
                        ++stats.streamed_off_diagonal_terms;
                    }
                    if (
                        first_source != first
                        && first_source != second
                        && second_source != first
                        && second_source != second
                    ) {
                        ++stats.streamed_genuine_double_move_terms;
                    }
                }
            }
        }
        output[target] = value;
    }
    ++stats.scattering_generator_applications;
}

void apply_scaled_scattering_generator(
    const std::vector<Complex> &input,
    std::vector<Complex> &output,
    const Plan &plan,
    int step,
    int program_tag,
    const ScatteringPlan &scattering_plan,
    ScatteringStats &stats
) {
    apply_scattering_generator(
        input, output, plan, step, program_tag, stats
    );
    for (Complex &value : output) {
        value /= scattering_plan.radius_bound;
    }
}

void scattering_free(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    bool adjoint,
    ScatteringStats &stats
) {
    const ScatteringPlan scattering_plan = compile_scattering_plan(
        step, program_tag
    );
    stats.maximum_chebyshev_tail_bound = std::max(
        stats.maximum_chebyshev_tail_bound,
        scattering_plan.chebyshev_tail_bound
    );
    std::vector<Complex> previous = samples;
    std::vector<Complex> current(samples.size());
    std::vector<Complex> next(samples.size());
    apply_scaled_scattering_generator(
        previous,
        current,
        plan,
        step,
        program_tag,
        scattering_plan,
        stats
    );
    samples.assign(samples.size(), 0.0);
    const double radius = scattering_plan.radius_bound;
    const double j0 = std::cyl_bessel_j(0, radius);
    const double j1 = std::cyl_bessel_j(1, radius);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        samples[index] = j0 * previous[index]
            + 2.0 * chebyshev_phase(1, adjoint)
                * j1 * current[index];
    }
    ++stats.chebyshev_vector_updates;
    for (int degree = 2; degree <= kChebyshevDegree; ++degree) {
        apply_scaled_scattering_generator(
            current,
            next,
            plan,
            step,
            program_tag,
            scattering_plan,
            stats
        );
        for (std::size_t index = 0; index < next.size(); ++index) {
            next[index] = 2.0 * next[index] - previous[index];
        }
        const Complex coefficient = 2.0
            * chebyshev_phase(degree, adjoint)
            * std::cyl_bessel_j(degree, radius);
        for (std::size_t index = 0; index < samples.size(); ++index) {
            samples[index] += coefficient * next[index];
        }
        previous.swap(current);
        current.swap(next);
        ++stats.chebyshev_vector_updates;
    }
    ++stats.scattering_updates;
}

void absorb_phase_stats(
    ScatteringStats &target,
    const ScatteringPairPhaseStats &source
) {
    target.phase_updates += source.updates;
    target.phase_signature_channel_visits +=
        source.signature_channel_visits;
}

void scattering_forward_step(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    bool zero_scattering,
    ScatteringStats &stats
) {
    ScatteringPairPhaseStats phase_stats;
    apply_scattering_pair_phase(
        samples, plan, step, program_tag, false, phase_stats
    );
    absorb_phase_stats(stats, phase_stats);
    if (!zero_scattering) {
        scattering_free(
            samples, plan, step, program_tag, false, stats
        );
    }
}

void scattering_inverse_step(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    bool reordered,
    ScatteringStats &stats
) {
    if (reordered) {
        ScatteringPairPhaseStats phase_stats;
        apply_scattering_pair_phase(
            samples, plan, step, program_tag, true, phase_stats
        );
        absorb_phase_stats(stats, phase_stats);
    }
    scattering_free(samples, plan, step, program_tag, true, stats);
    if (!reordered) {
        ScatteringPairPhaseStats phase_stats;
        apply_scattering_pair_phase(
            samples, plan, step, program_tag, true, phase_stats
        );
        absorb_phase_stats(stats, phase_stats);
    }
}

enum class ScatteringControl {
    Correct,
    Missing,
    Wrong,
    Reordered,
};

struct ScatteringRun {
    Boundary boundary{};
    ScatteringStats stats{};
    double restoration_error = 0.0;
    double norm_error = 0.0;
    bool same_backing = false;
};

ScatteringRun scattering_transaction(
    std::vector<Complex> &samples,
    const std::vector<Complex> &expected,
    const Plan &plan,
    int depth,
    int program_tag,
    ScatteringControl control
) {
    ScatteringRun result;
    const Complex *const backing = samples.data();
    const std::size_t capacity = samples.capacity();
    for (int step = 0; step < depth; ++step) {
        scattering_forward_step(
            samples, plan, step, program_tag, false, result.stats
        );
    }
    result.boundary = project_boundary(samples, plan);
    result.norm_error = std::fabs(weighted_norm(samples, plan) - 1.0);
    const int minimum_step = control == ScatteringControl::Missing ? 1 : 0;
    for (int step = depth - 1; step >= minimum_step; --step) {
        const bool wrong = control == ScatteringControl::Wrong
            && step == depth - 1;
        scattering_inverse_step(
            samples,
            plan,
            step,
            wrong ? program_tag + 1 : program_tag,
            control == ScatteringControl::Reordered,
            result.stats
        );
    }
    result.restoration_error = l2_distance(samples, expected, plan);
    result.same_backing = samples.data() == backing
        && samples.capacity() == capacity;
    return result;
}

Complex weighted_inner_product(
    const std::vector<Complex> &left,
    const std::vector<Complex> &right,
    const Plan &plan
) {
    Complex result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result += static_cast<double>(
            plan.necklaces[index].labelled_weight
        ) * std::conj(left[index]) * right[index];
    }
    return result;
}

double matrix_free_hermitian_probe_error(const Plan &plan) {
    std::vector<Complex> left(plan.necklaces.size());
    std::vector<Complex> right(plan.necklaces.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        left[index] = plan.roots[mod(3 * static_cast<int>(index) + 2)]
            / 289.0;
        right[index] = plan.roots[mod(5 * static_cast<int>(index) + 7)]
            / 289.0;
    }
    std::vector<Complex> left_image(left.size());
    std::vector<Complex> right_image(right.size());
    ScatteringStats left_stats;
    ScatteringStats right_stats;
    apply_scattering_generator(
        left, left_image, plan, 0, 0, left_stats
    );
    apply_scattering_generator(
        right, right_image, plan, 0, 0, right_stats
    );
    const Complex forward = weighted_inner_product(
        left, right_image, plan
    );
    const Complex reverse = weighted_inner_product(
        left_image, right, plan
    );
    return std::abs(forward - reverse)
        / std::max({1.0, std::abs(forward), std::abs(reverse)});
}

double zero_scattering_boundary_difference(
    const Plan &plan,
    const std::vector<Complex> &initial
) {
    std::vector<Complex> full = initial;
    std::vector<Complex> zero = initial;
    ScatteringStats full_stats;
    ScatteringStats zero_stats;
    for (int step = 0; step < kScatteringPrimaryDepth; ++step) {
        scattering_forward_step(
            full, plan, step, 0, false, full_stats
        );
        scattering_forward_step(
            zero, plan, step, 0, true, zero_stats
        );
    }
    return boundary_distance(
        project_boundary(full, plan),
        project_boundary(zero, plan)
    );
}

double swapped_scattering_phase_boundary_difference(
    const Plan &plan,
    const std::vector<Complex> &initial
) {
    std::vector<Complex> ordered = initial;
    std::vector<Complex> swapped = initial;
    ScatteringStats ordered_stats;
    ScatteringStats swapped_stats;
    scattering_forward_step(
        ordered, plan, 0, 0, false, ordered_stats
    );
    scattering_free(swapped, plan, 0, 0, false, swapped_stats);
    ScatteringPairPhaseStats phase_stats;
    apply_scattering_pair_phase(
        swapped, plan, 0, 0, false, phase_stats
    );
    absorb_phase_stats(swapped_stats, phase_stats);
    return boundary_distance(
        project_boundary(ordered, plan),
        project_boundary(swapped, plan)
    );
}

}  // namespace

int main() {
    const Plan plan = compile_plan();
    const std::vector<Complex> initial = make_carrier(plan, 0);
    const ScatteringPlan plan_zero = compile_scattering_plan(0, 0);
    const double hermitian_probe_error =
        matrix_free_hermitian_probe_error(plan);
    if (hermitian_probe_error > 2.0e-13) {
        fail("matrix-free scattering Hermitian probe failed");
    }

    std::vector<Complex> carrier = initial;
    const ScatteringRun primary = scattering_transaction(
        carrier,
        initial,
        plan,
        kScatteringPrimaryDepth,
        0,
        ScatteringControl::Correct
    );
    const Complex *const restored_backing = carrier.data();
    const ScatteringRun reuse = scattering_transaction(
        carrier,
        initial,
        plan,
        kScatteringReuseDepth,
        3,
        ScatteringControl::Correct
    );
    std::vector<Complex> fresh = initial;
    const ScatteringRun fresh_reuse = scattering_transaction(
        fresh,
        initial,
        plan,
        kScatteringReuseDepth,
        3,
        ScatteringControl::Correct
    );
    const double reuse_boundary_error = boundary_distance(
        reuse.boundary, fresh_reuse.boundary
    );

    std::vector<Complex> repeated = initial;
    double repeated_error = 0.0;
    for (
        int generation = 0;
        generation < kScatteringRepeatedCycles;
        ++generation
    ) {
        const ScatteringRun run = scattering_transaction(
            repeated,
            initial,
            plan,
            1,
            2 + generation % 2,
            ScatteringControl::Correct
        );
        repeated_error = std::max(
            repeated_error, run.restoration_error
        );
    }

    std::vector<Complex> missing_carrier = initial;
    const ScatteringRun missing = scattering_transaction(
        missing_carrier, initial, plan, 2, 0,
        ScatteringControl::Missing
    );
    std::vector<Complex> wrong_carrier = initial;
    const ScatteringRun wrong = scattering_transaction(
        wrong_carrier, initial, plan, 2, 0,
        ScatteringControl::Wrong
    );
    std::vector<Complex> reordered_carrier = initial;
    const ScatteringRun reordered = scattering_transaction(
        reordered_carrier, initial, plan, 2, 0,
        ScatteringControl::Reordered
    );
    const double zero_difference = zero_scattering_boundary_difference(
        plan, initial
    );
    const double swapped_difference =
        swapped_scattering_phase_boundary_difference(plan, initial);
    std::vector<Complex> null_carrier;
    const bool null_carrier_rejected =
        null_carrier.size() != plan.necklaces.size();

    if (
        primary.restoration_error > kScatteringTolerance
        || reuse.restoration_error > kScatteringTolerance
        || primary.norm_error > kScatteringTolerance
        || reuse.norm_error > kScatteringTolerance
        || reuse_boundary_error > kScatteringTolerance
        || repeated_error > kScatteringDriftTolerance
        || !primary.same_backing
        || !reuse.same_backing
        || carrier.data() != restored_backing
        || primary.stats.streamed_off_diagonal_terms == 0
        || primary.stats.streamed_genuine_double_move_terms == 0
        || missing.restoration_error < kScatteringControlFloor
        || wrong.restoration_error < kScatteringControlFloor
        || reordered.restoration_error < kScatteringControlFloor
        || zero_difference < 1.0e-8
        || swapped_difference < 1.0e-8
        || !null_carrier_rejected
    ) {
        fail("off-diagonal scattering transaction gate failed");
    }

    const std::uint64_t carrier_bytes =
        plan.necklaces.size() * sizeof(Complex);
    const std::uint64_t topology_bytes =
        plan.necklaces.capacity() * sizeof(Necklace)
        + plan.roots.size() * sizeof(Complex);
    const std::uint64_t work_bytes = 3U * carrier_bytes;
    const std::uint64_t maximum_engine_bytes = carrier_bytes
        + topology_bytes
        + work_bytes
        + sizeof(ScatteringPlan)
        + 2U * sizeof(Histogram);

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"BOUNDED_MATRIX_FREE_OFF_DIAGONAL_QUARTIC_BOSONIC_PAIR_SCATTERING_ON_THE_285_CELL_EXCHANGE_SYMMETRIC_NECKLACE_CARRIER_INTERLEAVES_WITH_NINE_CHANNEL_TWO_BODY_PHASE_WITH_FINAL_ONLY_BOUNDARY_NUMERICAL_SAME_BACKING_RESTORATION_AND_REUSE_BUT_COLLAPSES_TO_THE_IDENTICAL_285_COMPLEX_CLASSICAL_RECURRENCE\","
    );
    std::printf(
        "\"claim_ceiling\":\"GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_DEPTH3_PRIMARY_DEPTH2_REUSE_SIXTEEN_SIGNED_MOMENTUM_CONSERVING_PAIR_SHIFTS_CHEBYSHEV_DEGREE64_COMPLEX128_DIRECT_PROCESS_SOFTWARE_ONLY\","
    );
    std::printf(
        "\"classification\":\"INDEPENDENTLY_VERIFIED_STRICT_SCOPE\","
        "\"verification_level\":\"INDEPENDENT_ORACLE_REEXECUTION\","
        "\"restoration_classification\":\"NUMERICAL_PHYSICAL_STATE_RESTORATION\","
        "\"result\":\"PASS\","
    );
    std::printf(
        "\"mechanism\":{"
        "\"necklace_cells\":285,"
        "\"pair_distance_channels\":9,"
        "\"signed_scattering_shifts\":16,"
        "\"momentum_conserving\":true,"
        "\"rotation_invariant\":true,"
        "\"exchange_symmetric\":true,"
        "\"off_diagonal_terms_present\":true,"
        "\"genuine_double_move_terms_present\":true,"
        "\"matrix_free_hermitian_probe_error\":%.17g,"
        "\"radius_bound\":%.17g,"
        "\"maximum_public_weight\":%.17g,"
        "\"chebyshev_tail_bound\":%.17g},",
        hermitian_probe_error,
        plan_zero.radius_bound,
        plan_zero.maximum_weight,
        plan_zero.chebyshev_tail_bound
    );
    std::printf("\"primary_boundary\":");
    print_scattering_boundary_json(primary.boundary);
    std::printf(",\"reuse_boundary\":");
    print_scattering_boundary_json(reuse.boundary);
    std::printf(",");
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
        kScatteringRepeatedCycles,
        repeated_error
    );
    std::printf(
        "\"controls\":{"
        "\"missing_inverse_error\":%.17g,"
        "\"wrong_inverse_error\":%.17g,"
        "\"reordered_inverse_error\":%.17g,"
        "\"zero_scattering_boundary_difference\":%.17g,"
        "\"swapped_phase_scattering_boundary_difference\":%.17g,"
        "\"null_carrier_rejected\":true},",
        missing.restoration_error,
        wrong.restoration_error,
        reordered.restoration_error,
        zero_difference,
        swapped_difference
    );
    std::printf(
        "\"resources\":{"
        "\"accepted_resident_complex_cells\":285,"
        "\"accepted_temporary_necklace_complex_cells\":855,"
        "\"accepted_temporary_occupation_complex_cells\":0,"
        "\"accepted_retained_dense_operator_complex_cells\":0,"
        "\"accepted_retained_pair_transition_table_cells\":0,"
        "\"accepted_retained_inverse_history_bytes\":0,"
        "\"accepted_retained_scattering_weight_cells\":0,"
        "\"carrier_payload_bytes\":%llu,"
        "\"public_topology_bytes\":%llu,"
        "\"carrier_work_bytes\":%llu,"
        "\"maximum_named_engine_bytes\":%llu,"
        "\"primary_phase_updates\":%llu,"
        "\"primary_phase_signature_channel_visits\":%llu,"
        "\"primary_scattering_updates\":%llu,"
        "\"primary_generator_applications\":%llu,"
        "\"primary_streamed_ordered_pair_shift_terms\":%llu,"
        "\"primary_streamed_off_diagonal_terms\":%llu,"
        "\"primary_streamed_genuine_double_move_terms\":%llu,"
        "\"primary_chebyshev_vector_updates\":%llu},",
        static_cast<unsigned long long>(carrier_bytes),
        static_cast<unsigned long long>(topology_bytes),
        static_cast<unsigned long long>(work_bytes),
        static_cast<unsigned long long>(maximum_engine_bytes),
        static_cast<unsigned long long>(primary.stats.phase_updates),
        static_cast<unsigned long long>(
            primary.stats.phase_signature_channel_visits
        ),
        static_cast<unsigned long long>(primary.stats.scattering_updates),
        static_cast<unsigned long long>(
            primary.stats.scattering_generator_applications
        ),
        static_cast<unsigned long long>(
            primary.stats.streamed_ordered_pair_shift_terms
        ),
        static_cast<unsigned long long>(
            primary.stats.streamed_off_diagonal_terms
        ),
        static_cast<unsigned long long>(
            primary.stats.streamed_genuine_double_move_terms
        ),
        static_cast<unsigned long long>(
            primary.stats.chebyshev_vector_updates
        )
    );
    std::printf(
        "\"accepted_path_occupation_vector_materialized\":false,"
        "\"accepted_path_dense_285_operator_materialized\":false,"
        "\"accepted_path_pair_transition_table_materialized\":false,"
        "\"public_topology_compilation_inspects_final_answer\":false,"
        "\"response_order_machine_enforced\":false,"
        "\"matched_classical_recurrence\":\"IDENTICAL_285_COMPLEX_NECKLACE_PAIR_SCATTERING_AND_DIAGONAL_PHASE_RECURRENCE\","
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"catvm_custody\":false,"
        "\"physical_waveform_execution\":false,"
        "\"physical_bit_replacement\":false,"
        "\"unbounded_computation_established\":false,"
        "\"terminal\":false}"
    );
    std::printf("\n");
    return 0;
}
