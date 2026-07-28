#define BOSONIC_GIVENS_ENTRY bosonic_givens_predecessor_main
#include "four_rotor_bosonic_givens_phase.cpp"
#undef BOSONIC_GIVENS_ENTRY

/*
 * Direct global-rotation quotient closure.
 *
 * A circulant one-particle free unitary C has a Hermitian circulant logarithm
 * H.  Its four-boson lift is exp(i dGamma(H)).  Because H commutes with global
 * rotation, dGamma(H) acts directly on the 285 necklace amplitudes:
 *
 *   (dGamma(H) s)[m]
 *       = sum_{i:m_i>0} sum_j m_i H_ij
 *           s[canonical(m - e_i + e_j)].
 *
 * The exponential is evaluated with a fixed-degree Chebyshev recurrence.
 * Only carrier-sized work vectors are materialized; no 4,845-cell occupation
 * vector and no 285-by-285 transition operator is retained.
 */

namespace {

constexpr int kChebyshevDegree = 64;
constexpr double kGeneratorTolerance = 3.0e-11;

struct GeneratorPlan {
    Matrix generator{};
    double center = 0.0;
    double radius = 0.0;
    double maximum_eigenvalue_modulus_error = 0.0;
    double maximum_hermitian_error = 0.0;
    double chebyshev_tail_bound = 0.0;
};

struct GeneratorStats {
    std::uint64_t free_updates = 0;
    std::uint64_t generator_applications = 0;
    std::uint64_t streamed_generator_terms = 0;
    std::uint64_t chebyshev_vector_updates = 0;
    double maximum_eigenvalue_modulus_error = 0.0;
    double maximum_hermitian_error = 0.0;
    double maximum_chebyshev_tail_bound = 0.0;
};

GeneratorPlan compile_generator_plan(
    const Plan &plan,
    int chirp
) {
    GeneratorPlan result;
    std::array<double, kGrid> angles{};
    double minimum_angle = std::numeric_limits<double>::infinity();
    double maximum_angle = -std::numeric_limits<double>::infinity();

    for (int momentum = 0; momentum < kGrid; ++momentum) {
        Complex eigenvalue = 0.0;
        for (int difference = 0; difference < kGrid; ++difference) {
            eigenvalue += plan.roots[
                mod(
                    chirp * difference * difference
                    + momentum * difference
                )
            ] / std::sqrt(static_cast<double>(kGrid));
        }
        result.maximum_eigenvalue_modulus_error = std::max(
            result.maximum_eigenvalue_modulus_error,
            std::fabs(std::abs(eigenvalue) - 1.0)
        );
        angles[momentum] = std::arg(eigenvalue);
        minimum_angle = std::min(minimum_angle, angles[momentum]);
        maximum_angle = std::max(maximum_angle, angles[momentum]);
    }

    for (int target = 0; target < kGrid; ++target) {
        for (int source = 0; source < kGrid; ++source) {
            Complex value = 0.0;
            const int difference = target - source;
            for (int momentum = 0; momentum < kGrid; ++momentum) {
                value += angles[momentum]
                    * plan.roots[mod(-momentum * difference)]
                    / static_cast<double>(kGrid);
            }
            result.generator[target][source] = value;
        }
    }
    for (int row = 0; row < kGrid; ++row) {
        for (int column = 0; column < kGrid; ++column) {
            result.maximum_hermitian_error = std::max(
                result.maximum_hermitian_error,
                std::abs(
                    result.generator[row][column]
                    - std::conj(result.generator[column][row])
                )
            );
        }
    }

    const double lower = kRotors * minimum_angle;
    const double upper = kRotors * maximum_angle;
    result.center = 0.5 * (lower + upper);
    result.radius = 0.5 * (upper - lower);
    /*
     * From the Bessel series,
     * |J_n(r)| <= (r/2)^n/n! exp(r^2/(4(n+1))).
     * These bounds decrease with ratio at most r/(2(n+1)), so the
     * geometric envelope below bounds the complete omitted Chebyshev tail.
     */
    const int first_omitted = kChebyshevDegree + 1;
    const double leading_bound =
        std::pow(0.5 * result.radius, first_omitted)
        / std::tgamma(static_cast<double>(first_omitted + 1))
        * std::exp(
            result.radius * result.radius
            / (4.0 * static_cast<double>(first_omitted + 1))
        );
    const double tail_ratio =
        result.radius
        / (2.0 * static_cast<double>(first_omitted + 1));
    result.chebyshev_tail_bound =
        2.0 * leading_bound / (1.0 - tail_ratio);
    if (
        result.maximum_eigenvalue_modulus_error
            > kDecompositionTolerance
        || result.maximum_hermitian_error > kDecompositionTolerance
        || result.chebyshev_tail_bound > 1.0e-13
        || result.radius <= 0.0
    ) {
        fail("necklace generator compilation failed");
    }
    return result;
}

void apply_generator(
    const std::vector<Complex> &input,
    std::vector<Complex> &output,
    const Plan &plan,
    const GeneratorPlan &generator_plan,
    GeneratorStats &stats
) {
    if (
        input.size() != plan.necklaces.size()
        || output.size() != plan.necklaces.size()
    ) {
        fail("necklace generator carrier size failed");
    }
    for (
        std::size_t target = 0;
        target < plan.necklaces.size();
        ++target
    ) {
        const Histogram &histogram =
            plan.necklaces[target].histogram;
        Complex value = 0.0;
        for (int occupied = 0; occupied < kGrid; ++occupied) {
            if (histogram[occupied] == 0) {
                continue;
            }
            for (int source_mode = 0; source_mode < kGrid; ++source_mode) {
                Histogram source = histogram;
                --source[occupied];
                ++source[source_mode];
                value += static_cast<double>(histogram[occupied])
                    * generator_plan.generator[occupied][source_mode]
                    * input[
                        find_necklace(
                            plan,
                            canonical_histogram(source)
                        )
                    ];
                ++stats.streamed_generator_terms;
            }
        }
        output[target] = value;
    }
    ++stats.generator_applications;
}

void apply_scaled_generator(
    const std::vector<Complex> &input,
    std::vector<Complex> &output,
    const Plan &plan,
    const GeneratorPlan &generator_plan,
    GeneratorStats &stats
) {
    apply_generator(
        input, output, plan, generator_plan, stats
    );
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] = (
            output[index] - generator_plan.center * input[index]
        ) / generator_plan.radius;
    }
}

Complex chebyshev_phase(int degree, bool adjoint) {
    const Complex unit = adjoint
        ? Complex(0.0, -1.0)
        : Complex(0.0, 1.0);
    Complex result = 1.0;
    for (int index = 0; index < degree; ++index) {
        result *= unit;
    }
    return result;
}

void generator_free(
    std::vector<Complex> &samples,
    const Plan &plan,
    int chirp,
    bool adjoint,
    GeneratorStats &stats
) {
    const GeneratorPlan generator_plan =
        compile_generator_plan(plan, chirp);
    stats.maximum_eigenvalue_modulus_error = std::max(
        stats.maximum_eigenvalue_modulus_error,
        generator_plan.maximum_eigenvalue_modulus_error
    );
    stats.maximum_hermitian_error = std::max(
        stats.maximum_hermitian_error,
        generator_plan.maximum_hermitian_error
    );
    stats.maximum_chebyshev_tail_bound = std::max(
        stats.maximum_chebyshev_tail_bound,
        generator_plan.chebyshev_tail_bound
    );

    std::vector<Complex> previous = samples;
    std::vector<Complex> current(samples.size());
    std::vector<Complex> next(samples.size());
    apply_scaled_generator(
        previous,
        current,
        plan,
        generator_plan,
        stats
    );

    const double sign = adjoint ? -1.0 : 1.0;
    const Complex global_phase = std::polar(
        1.0,
        sign * generator_plan.center
    );
    const double j0 = std::cyl_bessel_j(
        0, generator_plan.radius
    );
    const double j1 = std::cyl_bessel_j(
        1, generator_plan.radius
    );
    for (std::size_t index = 0; index < samples.size(); ++index) {
        samples[index] = j0 * previous[index]
            + 2.0 * chebyshev_phase(1, adjoint)
                * j1 * current[index];
    }
    ++stats.chebyshev_vector_updates;

    for (int degree = 2; degree <= kChebyshevDegree; ++degree) {
        apply_scaled_generator(
            current,
            next,
            plan,
            generator_plan,
            stats
        );
        for (std::size_t index = 0; index < next.size(); ++index) {
            next[index] =
                2.0 * next[index] - previous[index];
        }
        const Complex coefficient =
            2.0 * chebyshev_phase(degree, adjoint)
            * std::cyl_bessel_j(degree, generator_plan.radius);
        for (std::size_t index = 0; index < samples.size(); ++index) {
            samples[index] += coefficient * next[index];
        }
        previous.swap(current);
        current.swap(next);
        ++stats.chebyshev_vector_updates;
    }
    for (std::size_t index = 0; index < samples.size(); ++index) {
        samples[index] *= global_phase;
    }
    ++stats.free_updates;
}

void generator_forward_step(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    Stats &stats,
    GeneratorStats &generator_stats
) {
    apply_collision(
        samples,
        plan,
        public_kappa(step, program_tag),
        false,
        stats
    );
    generator_free(
        samples,
        plan,
        public_chirp(step, program_tag),
        false,
        generator_stats
    );
}

void generator_inverse_step(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    Stats &stats,
    GeneratorStats &generator_stats
) {
    generator_free(
        samples,
        plan,
        public_chirp(step, program_tag),
        true,
        generator_stats
    );
    apply_collision(
        samples,
        plan,
        public_kappa(step, program_tag),
        true,
        stats
    );
}

struct GeneratorRun {
    Boundary boundary{};
    Stats stats{};
    GeneratorStats generator_stats{};
    double restoration_error = 0.0;
    double norm_error = 0.0;
    double elapsed_ms = 0.0;
};

GeneratorRun generator_transaction(
    std::vector<Complex> &samples,
    const std::vector<Complex> &expected_baseline,
    const Plan &plan,
    int depth,
    int program_tag,
    Control control
) {
    const auto begin = std::chrono::steady_clock::now();
    GeneratorRun result;
    for (int step = 0; step < depth; ++step) {
        generator_forward_step(
            samples,
            plan,
            step,
            program_tag,
            result.stats,
            result.generator_stats
        );
    }
    result.boundary = project_boundary(samples, plan);
    result.norm_error = std::fabs(
        weighted_norm(samples, plan) - 1.0
    );
    const int minimum_step =
        control == Control::Missing ? 1 : 0;
    for (int step = depth - 1; step >= minimum_step; --step) {
        if (control == Control::Wrong && step == depth - 1) {
            generator_inverse_step(
                samples,
                plan,
                step,
                program_tag + 1,
                result.stats,
                result.generator_stats
            );
        } else if (control == Control::Reordered) {
            apply_collision(
                samples,
                plan,
                public_kappa(step, program_tag),
                true,
                result.stats
            );
            generator_free(
                samples,
                plan,
                public_chirp(step, program_tag),
                true,
                result.generator_stats
            );
        } else {
            generator_inverse_step(
                samples,
                plan,
                step,
                program_tag,
                result.stats,
                result.generator_stats
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

int main() {
    const Plan plan = compile_plan();
    const std::vector<Complex> initial = make_carrier(plan, 0);

    std::vector<Complex> predecessor_one = initial;
    Stats predecessor_one_stats;
    FastStats predecessor_one_detail;
    fast_forward_step(
        predecessor_one,
        plan,
        0,
        0,
        predecessor_one_stats,
        predecessor_one_detail
    );
    std::vector<Complex> generator_one = initial;
    Stats generator_one_stats;
    GeneratorStats generator_one_detail;
    generator_forward_step(
        generator_one,
        plan,
        0,
        0,
        generator_one_stats,
        generator_one_detail
    );
    const double one_step_error =
        l2_distance(predecessor_one, generator_one, plan);
    if (one_step_error > kGeneratorTolerance) {
        fail("necklace generator one-step parity failed");
    }

    std::vector<Complex> predecessor_carrier = initial;
    const FastRun predecessor = fast_transaction(
        predecessor_carrier,
        initial,
        plan,
        kPrimaryDepth,
        0,
        Control::Correct
    );
    std::vector<Complex> carrier = initial;
    const GeneratorRun primary = generator_transaction(
        carrier,
        initial,
        plan,
        kPrimaryDepth,
        0,
        Control::Correct
    );
    const double primary_boundary_error =
        boundary_distance(primary.boundary, predecessor.boundary);
    if (
        primary_boundary_error > kGeneratorTolerance
        || primary.restoration_error > kRestorationTolerance
        || primary.norm_error > kGeneratorTolerance
    ) {
        fail("necklace generator primary gate failed");
    }

    const GeneratorRun reuse = generator_transaction(
        carrier, initial, plan, 2, 3, Control::Correct
    );
    std::vector<Complex> fresh = initial;
    const GeneratorRun fresh_reuse = generator_transaction(
        fresh, initial, plan, 2, 3, Control::Correct
    );
    const double reuse_boundary_error =
        boundary_distance(reuse.boundary, fresh_reuse.boundary);
    if (
        reuse.restoration_error > kRestorationTolerance
        || reuse_boundary_error > kGeneratorTolerance
    ) {
        fail("necklace generator reuse gate failed");
    }

    std::vector<Complex> missing_carrier = initial;
    const GeneratorRun missing = generator_transaction(
        missing_carrier, initial, plan, 2, 0, Control::Missing
    );
    std::vector<Complex> wrong_carrier = initial;
    const GeneratorRun wrong = generator_transaction(
        wrong_carrier, initial, plan, 2, 0, Control::Wrong
    );
    std::vector<Complex> reordered_carrier = initial;
    const GeneratorRun reordered = generator_transaction(
        reordered_carrier, initial, plan, 2, 0, Control::Reordered
    );
    if (
        missing.restoration_error < kControlFloor
        || wrong.restoration_error < kControlFloor
        || reordered.restoration_error < kControlFloor
    ) {
        fail("necklace generator inverse controls failed");
    }

    const std::uint64_t carrier_bytes =
        plan.necklaces.size() * sizeof(Complex);
    const std::uint64_t topology_bytes =
        plan.necklaces.capacity() * sizeof(Necklace)
        + plan.roots.size() * sizeof(Complex);
    const std::uint64_t generator_plan_bytes =
        sizeof(GeneratorPlan);
    const std::uint64_t carrier_work_bytes =
        3U * carrier_bytes;
    const std::uint64_t maximum_engine_bytes =
        carrier_bytes + topology_bytes + generator_plan_bytes
        + carrier_work_bytes + 2U * sizeof(Histogram);
    const std::uint64_t maximum_wrapper_bytes =
        maximum_engine_bytes + carrier_bytes + 2U * sizeof(Boundary);
    const std::uint64_t comparison_harness_peak_bytes =
        maximum_engine_bytes + 2U * carrier_bytes
        + kHistogramDimension * sizeof(Complex);

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"BOUNDED_SYMMETRY_PRESERVING_HERMITIAN_NECKLACE_GENERATOR_PHASE_CLOSURE_ELIMINATES_OCCUPATION_EXPANSION_WITH_ACTUAL_RESTORATION_AND_REUSE\","
    );
    std::printf(
        "\"claim_ceiling\":\"EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_GRID17_FOUR_ROTOR_DEPTH8_CHEBYSHEV_DEGREE64_TESTED_NONZERO_CHIRP_COMPLEX128_SOFTWARE_ONLY\","
    );
    std::printf("\"result\":\"PASS\",");
    std::printf(
        "\"resident_necklace_complex_cells\":285,"
        "\"accepted_path_temporary_necklace_work_complex_cells\":855,"
        "\"accepted_path_temporary_occupation_complex_cells\":0,"
        "\"comparison_path_temporary_occupation_complex_cells\":4845,"
        "\"comparison_bosonic_givens_occupation_scratch_bytes\":77520,"
        "\"accepted_path_retained_transition_operator_bytes\":0,"
        "\"labelled_wave_cells_avoided\":83521,"
        "\"accepted_path_permanent_assignment_terms_enumerated\":0,"
        "\"accepted_path_occupation_expansion_materialized\":false,"
        "\"comparison_path_occupation_expansion_materialized\":true,"
    );
    std::printf(
        "\"parity\":{"
        "\"one_step_bosonic_givens_l2_error\":%.17g,"
        "\"depth8_bosonic_givens_boundary_error\":%.17g,"
        "\"maximum_eigenvalue_modulus_error\":%.17g,"
        "\"maximum_generator_hermitian_error\":%.17g,"
        "\"maximum_chebyshev_tail_bound\":%.17g},",
        one_step_error,
        primary_boundary_error,
        primary.generator_stats.maximum_eigenvalue_modulus_error,
        primary.generator_stats.maximum_hermitian_error,
        primary.generator_stats.maximum_chebyshev_tail_bound
    );
    std::printf(
        "\"primary\":{"
        "\"depth\":8,"
        "\"weighted_norm_error\":%.17g,"
        "\"restoration_error\":%.17g,"
        "\"actual_inverse_restoration\":true,"
        "\"elapsed_ms\":%.17g,"
        "\"bosonic_givens_elapsed_ms\":%.17g,"
        "\"streamed_generator_terms\":%llu,"
        "\"generator_applications\":%llu,"
        "\"chebyshev_vector_updates\":%llu,"
        "\"resources\":{"
        "\"carrier_payload_bytes\":%llu,"
        "\"public_topology_bytes\":%llu,"
        "\"generator_plan_bytes\":%llu,"
        "\"carrier_work_vector_bytes\":%llu,"
        "\"occupation_scratch_bytes\":0,"
        "\"maximum_explicit_engine_bytes\":%llu,"
        "\"maximum_explicit_wrapper_bytes\":%llu,"
        "\"comparison_harness_peak_explicit_bytes\":%llu,"
        "\"retained_inverse_history_bytes\":0}},",
        primary.norm_error,
        primary.restoration_error,
        primary.elapsed_ms,
        predecessor.elapsed_ms,
        static_cast<unsigned long long>(
            primary.generator_stats.streamed_generator_terms
        ),
        static_cast<unsigned long long>(
            primary.generator_stats.generator_applications
        ),
        static_cast<unsigned long long>(
            primary.generator_stats.chebyshev_vector_updates
        ),
        static_cast<unsigned long long>(carrier_bytes),
        static_cast<unsigned long long>(topology_bytes),
        static_cast<unsigned long long>(generator_plan_bytes),
        static_cast<unsigned long long>(carrier_work_bytes),
        static_cast<unsigned long long>(maximum_engine_bytes),
        static_cast<unsigned long long>(maximum_wrapper_bytes),
        static_cast<unsigned long long>(comparison_harness_peak_bytes)
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
        "\"matched_classical_generator_identical\":true,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"unbounded_computation_established\":false,"
        "\"obstruction\":\"CHEBYSHEV_GENERATOR_WORK_AND_MATCHED_CLASSICAL_HERMITIAN_QUOTIENT_IDENTITY\","
        "\"terminal\":false"
    );
    std::printf("}\n");
    return 0;
}
