#define main necklace_predecessor_main
#include "four_rotor_necklace_orbit_phase.cpp"
#undef main

#ifndef BOSONIC_GIVENS_ENTRY
#define BOSONIC_GIVENS_ENTRY main
#endif

/*
 * Successor to streamed necklace permanents.
 *
 * The resident state remains the 285-amplitude necklace carrier. For a free
 * update it expands only to the 4,845 permutation-symmetric occupation
 * coefficients, applies a topology-compiled 17-mode Givens network through
 * degree-four polynomial blocks, then closes back to necklaces. It does not
 * construct labelled configurations, a transition matrix, or permanent
 * assignment terms.
 */

namespace {

constexpr std::size_t kHistogramDimension = 4845;
constexpr double kDecompositionTolerance = 3.0e-13;
constexpr double kClosureTolerance = 3.0e-11;

struct Givens {
    int upper = 0;
    int lower = 0;
    double cosine = 1.0;
    Complex sine = 0.0;
};

struct FreePlan {
    std::vector<Givens> givens;
    std::array<Complex, kGrid> diagonal{};
    double reconstruction_error = 0.0;
};

using Matrix = std::array<
    std::array<Complex, kGrid>,
    kGrid
>;

Matrix single_particle_matrix(
    const Plan &plan,
    int chirp,
    bool adjoint
) {
    Matrix result{};
    for (int row = 0; row < kGrid; ++row) {
        for (int column = 0; column < kGrid; ++column) {
            result[row][column] = free_entry(
                plan, row, column, chirp, adjoint
            );
        }
    }
    return result;
}

void left_multiply_givens(
    Matrix &matrix,
    const Givens &givens
) {
    for (int column = 0; column < kGrid; ++column) {
        const Complex upper =
            matrix[givens.upper][column];
        const Complex lower =
            matrix[givens.lower][column];
        matrix[givens.upper][column] =
            givens.cosine * upper + givens.sine * lower;
        matrix[givens.lower][column] =
            -std::conj(givens.sine) * upper
            + givens.cosine * lower;
    }
}

FreePlan compile_free_plan(
    const Plan &plan,
    int chirp
) {
    Matrix reduced = single_particle_matrix(
        plan, chirp, false
    );
    FreePlan result;
    result.givens.reserve(
        static_cast<std::size_t>(kGrid * (kGrid - 1) / 2)
    );
    for (int column = 0; column < kGrid - 1; ++column) {
        for (int row = kGrid - 1; row > column; --row) {
            const Complex upper = reduced[row - 1][column];
            const Complex lower = reduced[row][column];
            const double norm = std::hypot(
                std::abs(upper), std::abs(lower)
            );
            Givens givens;
            givens.upper = row - 1;
            givens.lower = row;
            if (norm > 0.0) {
                const Complex alpha =
                    std::abs(upper) > 0.0
                    ? upper / std::abs(upper)
                    : Complex(1.0, 0.0);
                givens.cosine = std::abs(upper) / norm;
                givens.sine =
                    alpha * std::conj(lower) / norm;
            }
            left_multiply_givens(reduced, givens);
            result.givens.push_back(givens);
        }
    }
    for (int row = 0; row < kGrid; ++row) {
        result.diagonal[row] = reduced[row][row];
        for (int column = 0; column < kGrid; ++column) {
            if (row == column) {
                continue;
            }
            result.reconstruction_error = std::max(
                result.reconstruction_error,
                std::abs(reduced[row][column])
            );
        }
        result.reconstruction_error = std::max(
            result.reconstruction_error,
            std::fabs(std::abs(result.diagonal[row]) - 1.0)
        );
    }
    if (
        result.givens.size() != 136U
        || result.reconstruction_error > kDecompositionTolerance
    ) {
        fail("single-particle Givens decomposition failed");
    }
    return result;
}

std::size_t histogram_rank(const Histogram &histogram) {
    std::size_t rank = 0;
    int remaining = kRotors;
    for (int position = 0; position < kGrid - 1; ++position) {
        const int slots_after = kGrid - position - 1;
        for (int value = 0; value < histogram[position]; ++value) {
            rank += choose(
                remaining - value + slots_after - 1,
                slots_after - 1
            );
        }
        remaining -= histogram[position];
    }
    return rank;
}

Histogram histogram_unrank(std::size_t rank) {
    Histogram histogram{};
    int remaining = kRotors;
    for (int position = 0; position < kGrid - 1; ++position) {
        const int slots_after = kGrid - position - 1;
        int value = 0;
        while (value <= remaining) {
            const std::size_t count = choose(
                remaining - value + slots_after - 1,
                slots_after - 1
            );
            if (rank < count) {
                break;
            }
            rank -= count;
            ++value;
        }
        if (value > remaining) {
            fail("histogram unrank failed");
        }
        histogram[position] = static_cast<std::uint8_t>(value);
        remaining -= value;
    }
    histogram[kGrid - 1] =
        static_cast<std::uint8_t>(remaining);
    return histogram;
}

double multinomial(const Histogram &histogram) {
    std::uint64_t denominator = 1;
    for (int count : histogram) {
        denominator *= factorial(count);
    }
    return static_cast<double>(
        factorial(kRotors) / denominator
    );
}

double binomial(int n, int k) {
    return static_cast<double>(choose(n, k));
}

struct FastStats {
    std::uint64_t free_updates = 0;
    std::uint64_t givens_two_mode_updates = 0;
    std::uint64_t polynomial_block_terms = 0;
    std::uint64_t occupation_cells_expanded = 0;
    std::uint64_t occupation_cells_closed = 0;
    double maximum_symmetry_closure_error = 0.0;
    double maximum_decomposition_error = 0.0;
};

void apply_diagonal_polynomial(
    std::vector<Complex> &polynomial,
    const std::array<Complex, kGrid> &diagonal,
    bool adjoint
) {
    for (
        std::size_t index = 0;
        index < polynomial.size();
        ++index
    ) {
        const Histogram histogram = histogram_unrank(index);
        Complex factor = 1.0;
        for (int mode = 0; mode < kGrid; ++mode) {
            const Complex value = adjoint
                ? std::conj(diagonal[mode])
                : diagonal[mode];
            for (int count = 0; count < histogram[mode]; ++count) {
                factor *= value;
            }
        }
        polynomial[index] *= factor;
    }
}

void apply_two_mode_polynomial(
    std::vector<Complex> &polynomial,
    const Givens &givens,
    bool adjoint,
    FastStats &stats
) {
    Complex m00 = givens.cosine;
    Complex m01 = givens.sine;
    Complex m10 = -std::conj(givens.sine);
    Complex m11 = givens.cosine;
    if (adjoint) {
        m01 = -givens.sine;
        m10 = std::conj(givens.sine);
    }

    for (
        std::size_t base_index = 0;
        base_index < polynomial.size();
        ++base_index
    ) {
        Histogram base = histogram_unrank(base_index);
        if (base[givens.lower] != 0) {
            continue;
        }
        const int total = base[givens.upper];
        std::array<Complex, kRotors + 1> old{};
        std::array<Complex, kRotors + 1> updated{};
        for (int upper_count = 0; upper_count <= total; ++upper_count) {
            Histogram member = base;
            member[givens.upper] =
                static_cast<std::uint8_t>(upper_count);
            member[givens.lower] =
                static_cast<std::uint8_t>(total - upper_count);
            old[upper_count] = polynomial[histogram_rank(member)];
        }
        for (int input_upper = 0; input_upper <= total; ++input_upper) {
            const int input_lower = total - input_upper;
            for (
                int choose_upper = 0;
                choose_upper <= input_upper;
                ++choose_upper
            ) {
                for (
                    int choose_lower = 0;
                    choose_lower <= input_lower;
                    ++choose_lower
                ) {
                    const int output_upper =
                        choose_upper + choose_lower;
                    Complex coefficient =
                        binomial(input_upper, choose_upper)
                        * binomial(input_lower, choose_lower);
                    coefficient *= std::pow(m00, choose_upper);
                    coefficient *= std::pow(
                        m10, input_upper - choose_upper
                    );
                    coefficient *= std::pow(m01, choose_lower);
                    coefficient *= std::pow(
                        m11, input_lower - choose_lower
                    );
                    updated[output_upper] +=
                        old[input_upper] * coefficient;
                    ++stats.polynomial_block_terms;
                }
            }
        }
        for (int upper_count = 0; upper_count <= total; ++upper_count) {
            Histogram member = base;
            member[givens.upper] =
                static_cast<std::uint8_t>(upper_count);
            member[givens.lower] =
                static_cast<std::uint8_t>(total - upper_count);
            polynomial[histogram_rank(member)] =
                updated[upper_count];
        }
        ++stats.givens_two_mode_updates;
    }
}

void expand_necklaces(
    const std::vector<Complex> &samples,
    const Plan &plan,
    std::vector<Complex> &polynomial,
    FastStats &stats
) {
    for (
        std::size_t index = 0;
        index < polynomial.size();
        ++index
    ) {
        const Histogram histogram = histogram_unrank(index);
        const Histogram canonical = canonical_histogram(histogram);
        polynomial[index] =
            multinomial(histogram)
            * samples[find_necklace(plan, canonical)];
        ++stats.occupation_cells_expanded;
    }
}

void close_necklaces(
    const std::vector<Complex> &polynomial,
    const Plan &plan,
    std::vector<Complex> &samples,
    FastStats &stats
) {
    for (
        std::size_t necklace = 0;
        necklace < plan.necklaces.size();
        ++necklace
    ) {
        const Histogram &histogram =
            plan.necklaces[necklace].histogram;
        samples[necklace] =
            polynomial[histogram_rank(histogram)]
            / multinomial(histogram);
    }
    for (
        std::size_t index = 0;
        index < polynomial.size();
        ++index
    ) {
        const Histogram histogram = histogram_unrank(index);
        const Histogram canonical = canonical_histogram(histogram);
        const Complex per_configuration =
            polynomial[index] / multinomial(histogram);
        stats.maximum_symmetry_closure_error = std::max(
            stats.maximum_symmetry_closure_error,
            std::abs(
                per_configuration
                - samples[find_necklace(plan, canonical)]
            )
        );
        ++stats.occupation_cells_closed;
    }
    if (stats.maximum_symmetry_closure_error > kClosureTolerance) {
        fail("necklace symmetry closure failed");
    }
}

void fast_free(
    std::vector<Complex> &samples,
    const Plan &plan,
    int chirp,
    bool adjoint,
    FastStats &stats
) {
    const FreePlan free_plan = compile_free_plan(plan, chirp);
    stats.maximum_decomposition_error = std::max(
        stats.maximum_decomposition_error,
        free_plan.reconstruction_error
    );
    std::vector<Complex> polynomial(kHistogramDimension);
    expand_necklaces(samples, plan, polynomial, stats);
    if (!adjoint) {
        apply_diagonal_polynomial(
            polynomial, free_plan.diagonal, false
        );
        for (
            auto item = free_plan.givens.rbegin();
            item != free_plan.givens.rend();
            ++item
        ) {
            apply_two_mode_polynomial(
                polynomial, *item, true, stats
            );
        }
    } else {
        for (const Givens &givens : free_plan.givens) {
            apply_two_mode_polynomial(
                polynomial, givens, false, stats
            );
        }
        apply_diagonal_polynomial(
            polynomial, free_plan.diagonal, true
        );
    }
    close_necklaces(polynomial, plan, samples, stats);
    ++stats.free_updates;
}

void fast_forward_step(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    Stats &stats,
    FastStats &fast_stats
) {
    apply_collision(
        samples, plan, public_kappa(step, program_tag), false, stats
    );
    fast_free(
        samples,
        plan,
        public_chirp(step, program_tag),
        false,
        fast_stats
    );
}

void fast_inverse_step(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    Stats &stats,
    FastStats &fast_stats
) {
    fast_free(
        samples,
        plan,
        public_chirp(step, program_tag),
        true,
        fast_stats
    );
    apply_collision(
        samples, plan, public_kappa(step, program_tag), true, stats
    );
}

struct FastRun {
    Boundary boundary{};
    Stats stats{};
    FastStats fast_stats{};
    double restoration_error = 0.0;
    double norm_error = 0.0;
    double elapsed_ms = 0.0;
};

FastRun fast_transaction(
    std::vector<Complex> &samples,
    const std::vector<Complex> &expected_baseline,
    const Plan &plan,
    int depth,
    int program_tag,
    Control control
) {
    const auto begin = std::chrono::steady_clock::now();
    FastRun result;
    for (int step = 0; step < depth; ++step) {
        fast_forward_step(
            samples,
            plan,
            step,
            program_tag,
            result.stats,
            result.fast_stats
        );
    }
    result.boundary = project_boundary(samples, plan);
    result.norm_error = std::fabs(weighted_norm(samples, plan) - 1.0);
    const int minimum_step =
        control == Control::Missing ? 1 : 0;
    for (int step = depth - 1; step >= minimum_step; --step) {
        if (control == Control::Wrong && step == depth - 1) {
            fast_inverse_step(
                samples,
                plan,
                step,
                program_tag + 1,
                result.stats,
                result.fast_stats
            );
        } else if (control == Control::Reordered) {
            apply_collision(
                samples,
                plan,
                public_kappa(step, program_tag),
                true,
                result.stats
            );
            fast_free(
                samples,
                plan,
                public_chirp(step, program_tag),
                true,
                result.fast_stats
            );
        } else {
            fast_inverse_step(
                samples,
                plan,
                step,
                program_tag,
                result.stats,
                result.fast_stats
            );
        }
    }
    result.restoration_error =
        l2_distance(samples, expected_baseline, plan);
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin
    ).count();
    return result;
}

}  // namespace

int BOSONIC_GIVENS_ENTRY() {
    const Plan plan = compile_plan();
    const std::vector<Complex> initial = make_carrier(plan, 0);

    // Direct streamed-permanent parity for one step and the full primary.
    std::vector<Complex> direct_one = initial;
    Stats direct_one_stats;
    forward_step(direct_one, plan, 0, 0, direct_one_stats);
    std::vector<Complex> fast_one = initial;
    Stats fast_one_stats;
    FastStats fast_one_detail;
    fast_forward_step(
        fast_one, plan, 0, 0, fast_one_stats, fast_one_detail
    );
    const double one_step_error =
        l2_distance(direct_one, fast_one, plan);
    if (one_step_error > kClosureTolerance) {
        fail("Givens one-step predecessor parity failed");
    }
    direct_one.clear();
    direct_one.shrink_to_fit();
    fast_one.clear();
    fast_one.shrink_to_fit();

    std::vector<Complex> direct_primary = initial;
    const Run direct = transaction(
        direct_primary,
        initial,
        plan,
        kPrimaryDepth,
        0,
        Control::Correct
    );
    direct_primary.clear();
    direct_primary.shrink_to_fit();
    std::vector<Complex> carrier = initial;
    const FastRun primary = fast_transaction(
        carrier,
        initial,
        plan,
        kPrimaryDepth,
        0,
        Control::Correct
    );
    const double primary_boundary_error =
        boundary_distance(primary.boundary, direct.boundary);
    if (
        primary_boundary_error > kClosureTolerance
        || primary.restoration_error > kRestorationTolerance
        || primary.norm_error > kClosureTolerance
    ) {
        fail("Givens primary gate failed");
    }

    const FastRun reuse = fast_transaction(
        carrier, initial, plan, 2, 3, Control::Correct
    );
    std::vector<Complex> fresh = initial;
    const FastRun fresh_reuse = fast_transaction(
        fresh, initial, plan, 2, 3, Control::Correct
    );
    const double reuse_boundary_error =
        boundary_distance(reuse.boundary, fresh_reuse.boundary);
    if (
        reuse.restoration_error > kRestorationTolerance
        || reuse_boundary_error > kClosureTolerance
    ) {
        fail("Givens restored-carrier reuse gate failed");
    }

    std::vector<Complex> missing_carrier = initial;
    const FastRun missing = fast_transaction(
        missing_carrier, initial, plan, 2, 0, Control::Missing
    );
    std::vector<Complex> wrong_carrier = initial;
    const FastRun wrong = fast_transaction(
        wrong_carrier, initial, plan, 2, 0, Control::Wrong
    );
    std::vector<Complex> reordered_carrier = initial;
    const FastRun reordered = fast_transaction(
        reordered_carrier, initial, plan, 2, 0, Control::Reordered
    );
    if (
        missing.restoration_error < kControlFloor
        || wrong.restoration_error < kControlFloor
        || reordered.restoration_error < kControlFloor
    ) {
        fail("Givens inverse controls failed");
    }

    const std::uint64_t carrier_bytes =
        plan.necklaces.size() * sizeof(Complex);
    const std::uint64_t topology_bytes =
        plan.necklaces.capacity() * sizeof(Necklace)
        + plan.roots.size() * sizeof(Complex);
    const std::uint64_t occupation_scratch_bytes =
        kHistogramDimension * sizeof(Complex);
    const std::uint64_t givens_plan_bytes =
        136U * sizeof(Givens)
        + kGrid * sizeof(Complex);
    const std::uint64_t matrix_compilation_bytes =
        kGrid * kGrid * sizeof(Complex);
    const std::uint64_t polynomial_block_scratch_bytes =
        2U * (kRotors + 1U) * sizeof(Complex)
        + 3U * sizeof(Histogram);
    const std::uint64_t maximum_engine_bytes =
        carrier_bytes + topology_bytes + occupation_scratch_bytes
        + givens_plan_bytes + polynomial_block_scratch_bytes;
    const std::uint64_t maximum_wrapper_bytes =
        maximum_engine_bytes + carrier_bytes + 2U * sizeof(Boundary);
    const std::uint64_t comparison_harness_peak_bytes =
        maximum_engine_bytes + 2U * carrier_bytes;
    const std::uint64_t compilation_peak_bytes =
        topology_bytes + givens_plan_bytes + matrix_compilation_bytes
        + 3U * sizeof(Histogram) + sizeof(Necklace);

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"BOUNDED_TOPOLOGY_COMPILED_BOSONIC_GIVENS_PHASE_CLOSURE_REPLACES_STREAMED_NECKLACE_TRANSITION_PERMANENTS_WITH_POLYNOMIAL_OCCUPATION_SCRATCH_ACTUAL_RESTORATION_AND_REUSE\","
    );
    std::printf(
        "\"claim_ceiling\":\"EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_GRID17_FOUR_ROTOR_DEPTH8_TESTED_NONZERO_CHIRP_SCHEDULE_COMPLEX128_SOFTWARE_ONLY\","
    );
    std::printf("\"result\":\"PASS\",");
    std::printf(
        "\"resident_necklace_complex_cells\":285,"
        "\"temporary_occupation_complex_cells\":4845,"
        "\"labelled_wave_cells_avoided\":83521,"
        "\"retained_transition_operator_bytes\":0,"
        "\"accepted_path_permanent_assignment_terms_enumerated\":0,"
        "\"comparison_predecessor_permanent_assignment_terms_enumerated\":%llu,",
        static_cast<unsigned long long>(
            direct.stats.exact_cyclotomic_permanent_terms
        )
    );
    std::printf(
        "\"parity\":{"
        "\"one_step_predecessor_l2_error\":%.17g,"
        "\"depth8_predecessor_boundary_error\":%.17g,"
        "\"maximum_single_particle_decomposition_error\":%.17g,"
        "\"maximum_necklace_closure_error\":%.17g},",
        one_step_error,
        primary_boundary_error,
        primary.fast_stats.maximum_decomposition_error,
        primary.fast_stats.maximum_symmetry_closure_error
    );
    std::printf(
        "\"primary\":{"
        "\"depth\":8,"
        "\"weighted_norm_error\":%.17g,"
        "\"restoration_error\":%.17g,"
        "\"actual_inverse_restoration\":true,"
        "\"elapsed_ms\":%.17g,"
        "\"predecessor_elapsed_ms\":%.17g,"
        "\"warm_elapsed_reduction_factor\":%.17g,"
        "\"enumerated_term_count_ratio\":%.17g,"
        "\"predecessor_permanent_terms\":%llu,"
        "\"accepted_polynomial_block_terms\":%llu,"
        "\"resources\":{"
        "\"carrier_payload_bytes\":%llu,"
        "\"public_topology_bytes\":%llu,"
        "\"occupation_scratch_bytes\":%llu,"
        "\"givens_plan_bytes\":%llu,"
        "\"polynomial_block_scratch_bytes\":%llu,"
        "\"compilation_conservative_explicit_payload_bytes\":%llu,"
        "\"maximum_explicit_engine_bytes\":%llu,"
        "\"maximum_explicit_wrapper_bytes\":%llu,"
        "\"comparison_harness_peak_explicit_bytes\":%llu,"
        "\"retained_inverse_history_bytes\":0,"
        "\"occupation_expansion_materialized\":true,"
        "\"labelled_assignment_expansion_materialized\":false,"
        "\"givens_two_mode_updates\":%llu,"
        "\"polynomial_block_terms\":%llu}},",
        primary.norm_error,
        primary.restoration_error,
        primary.elapsed_ms,
        direct.elapsed_ms,
        direct.elapsed_ms / primary.elapsed_ms,
        static_cast<double>(
            direct.stats.exact_cyclotomic_permanent_terms
        ) / static_cast<double>(
            primary.fast_stats.polynomial_block_terms
        ),
        static_cast<unsigned long long>(
            direct.stats.exact_cyclotomic_permanent_terms
        ),
        static_cast<unsigned long long>(
            primary.fast_stats.polynomial_block_terms
        ),
        static_cast<unsigned long long>(carrier_bytes),
        static_cast<unsigned long long>(topology_bytes),
        static_cast<unsigned long long>(occupation_scratch_bytes),
        static_cast<unsigned long long>(givens_plan_bytes),
        static_cast<unsigned long long>(
            polynomial_block_scratch_bytes
        ),
        static_cast<unsigned long long>(compilation_peak_bytes),
        static_cast<unsigned long long>(maximum_engine_bytes),
        static_cast<unsigned long long>(maximum_wrapper_bytes),
        static_cast<unsigned long long>(comparison_harness_peak_bytes),
        static_cast<unsigned long long>(
            primary.fast_stats.givens_two_mode_updates
        ),
        static_cast<unsigned long long>(
            primary.fast_stats.polynomial_block_terms
        )
    );
    std::printf(
        "\"reuse\":{"
        "\"restoration_generation\":2,"
        "\"restoration_error\":%.17g,"
        "\"fresh_restored_boundary_error\":%.17g,"
        "\"actual_restored_carrier_reuse\":true},",
        reuse.restoration_error,
        reuse_boundary_error
    );
    std::printf(
        "\"controls\":{"
        "\"missing_inverse_error\":%.17g,"
        "\"wrong_inverse_error\":%.17g,"
        "\"reordered_inverse_error\":%.17g},",
        missing.restoration_error,
        wrong.restoration_error,
        reordered.restoration_error
    );
    std::printf(
        "\"matched_classical_bosonic_givens_identical\":true,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"unbounded_computation_established\":false,"
        "\"terminal\":false,"
        "\"obstruction\":\"POLYNOMIAL_OCCUPATION_SCRATCH_AND_MATCHED_CLASSICAL_BOSONIC_GIVENS_IDENTITY\""
    );
    std::printf("}\n");
    return 0;
}
