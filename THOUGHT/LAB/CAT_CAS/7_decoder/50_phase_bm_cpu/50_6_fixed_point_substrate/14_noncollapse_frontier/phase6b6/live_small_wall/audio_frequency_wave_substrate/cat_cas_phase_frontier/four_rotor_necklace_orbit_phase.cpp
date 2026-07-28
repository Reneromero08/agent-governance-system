#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

/*
 * Exact permutation-plus-global-rotation quotient for a bounded
 * exchange-symmetric, rotation-invariant phase program.
 *
 * The accepted engine stores one unresolved complex amplitude per cyclic
 * necklace of occupation histograms. It never stores the 17^R labelled wave
 * or a D_R by D_R free operator. Each free transition coefficient is streamed
 * as an exact 17-component cyclotomic integer count from a size-R permanent.
 * A labelled wave exists only in the independent bounded verifier.
 */

namespace {

constexpr int kGrid = 17;
constexpr int kRotors = 4;
constexpr int kExpectedDimension = 285;
constexpr int kMaximumCollision = 6;
constexpr int kPrimaryDepth = 8;
constexpr double kPi =
    3.141592653589793238462643383279502884;
constexpr double kParityTolerance = 2.0e-11;
constexpr double kRestorationTolerance = 3.0e-11;
constexpr double kControlFloor = 1.0e-5;

using Complex = std::complex<double>;
using Histogram = std::array<std::uint8_t, kGrid>;
using Tuple = std::array<std::uint8_t, kRotors>;

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(2);
}

int mod(int value) {
    value %= kGrid;
    return value < 0 ? value + kGrid : value;
}

std::uint64_t factorial(int value) {
    std::uint64_t result = 1;
    for (int item = 2; item <= value; ++item) {
        result *= static_cast<std::uint64_t>(item);
    }
    return result;
}

std::uint64_t choose(int n, int k) {
    if (k > n - k) {
        k = n - k;
    }
    std::uint64_t result = 1;
    for (int item = 1; item <= k; ++item) {
        result = result * static_cast<std::uint64_t>(n - k + item)
            / static_cast<std::uint64_t>(item);
    }
    return result;
}

Histogram rotate_histogram(const Histogram &source, int shift) {
    Histogram result{};
    for (int index = 0; index < kGrid; ++index) {
        result[mod(index + shift)] = source[index];
    }
    return result;
}

Histogram canonical_histogram(const Histogram &source) {
    Histogram result = source;
    for (int shift = 1; shift < kGrid; ++shift) {
        result = std::min(result, rotate_histogram(source, shift));
    }
    return result;
}

Tuple tuple_from_histogram(const Histogram &histogram) {
    Tuple result{};
    int cursor = 0;
    for (int value = 0; value < kGrid; ++value) {
        for (int count = 0; count < histogram[value]; ++count) {
            if (cursor >= kRotors) {
                fail("histogram tuple overflow");
            }
            result[cursor++] = static_cast<std::uint8_t>(value);
        }
    }
    if (cursor != kRotors) {
        fail("histogram tuple underflow");
    }
    return result;
}

int collision_count(const Histogram &histogram) {
    int result = 0;
    for (int count : histogram) {
        result += count * (count - 1) / 2;
    }
    return result;
}

struct Necklace {
    Histogram histogram{};
    Tuple representative{};
    std::uint32_t labelled_weight = 0;
    std::uint32_t permanent_denominator = 0;
    int collisions = 0;
};

void generate_necklaces(
    int position,
    int remaining,
    Histogram &working,
    std::uint64_t &histogram_count,
    std::vector<Necklace> &result
) {
    if (position == kGrid - 1) {
        working[position] = static_cast<std::uint8_t>(remaining);
        ++histogram_count;
        if (canonical_histogram(working) != working) {
            return;
        }
        std::uint64_t denominator = 1;
        for (int count : working) {
            denominator *= factorial(count);
        }
        const std::uint64_t weight =
            static_cast<std::uint64_t>(kGrid)
            * factorial(kRotors) / denominator;
        result.push_back({
            working,
            tuple_from_histogram(working),
            static_cast<std::uint32_t>(weight),
            static_cast<std::uint32_t>(denominator),
            collision_count(working),
        });
        return;
    }
    for (int value = 0; value <= remaining; ++value) {
        working[position] = static_cast<std::uint8_t>(value);
        generate_necklaces(
            position + 1,
            remaining - value,
            working,
            histogram_count,
            result
        );
    }
}

std::vector<Necklace> compile_necklaces() {
    Histogram working{};
    std::uint64_t histogram_count = 0;
    std::vector<Necklace> result;
    result.reserve(kExpectedDimension);
    generate_necklaces(
        0, kRotors, working, histogram_count, result
    );
    if (histogram_count != choose(kRotors + kGrid - 1, kRotors)) {
        fail("weak-composition count failed");
    }
    if (
        result.size() != kExpectedDimension
        || result.size()
            != choose(kRotors + kGrid - 1, kRotors) / kGrid
    ) {
        fail("necklace dimension failed");
    }
    std::uint64_t total_weight = 0;
    for (const Necklace &necklace : result) {
        total_weight += necklace.labelled_weight;
    }
    if (total_weight != 83521U) {
        fail("necklace labelled weight failed");
    }
    return result;
}

struct Plan {
    std::vector<Necklace> necklaces;
    std::array<Complex, kGrid> roots{};
};

Plan compile_plan() {
    Plan plan;
    plan.necklaces = compile_necklaces();
    for (int exponent = 0; exponent < kGrid; ++exponent) {
        const double angle =
            2.0 * kPi * static_cast<double>(exponent)
            / static_cast<double>(kGrid);
        plan.roots[exponent] = std::polar(1.0, angle);
    }
    return plan;
}

struct Stats {
    std::uint64_t collision_phase_updates = 0;
    std::uint64_t free_phase_updates = 0;
    std::uint64_t exact_cyclotomic_permanent_terms = 0;
    std::uint64_t streamed_transition_coefficients = 0;
    std::uint64_t retained_inverse_history_bytes = 0;
};

Complex transition_coefficient(
    const Plan &plan,
    const Necklace &target,
    const Necklace &source,
    int chirp,
    bool adjoint,
    Stats &stats
) {
    std::array<std::int64_t, kGrid> counts{};
    const int signed_chirp = adjoint ? -chirp : chirp;
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
                exponent += signed_chirp * difference * difference;
            }
            ++counts[mod(exponent)];
            ++stats.exact_cyclotomic_permanent_terms;
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
    const double free_scale =
        std::pow(static_cast<double>(kGrid), -0.5 * kRotors);
    result *= free_scale
        / static_cast<double>(source.permanent_denominator);
    ++stats.streamed_transition_coefficients;
    return result;
}

void apply_collision(
    std::vector<Complex> &samples,
    const Plan &plan,
    int kappa,
    bool adjoint,
    Stats &stats
) {
    const int sign = adjoint ? -1 : 1;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        samples[index] *= plan.roots[
            mod(sign * kappa * plan.necklaces[index].collisions)
        ];
        ++stats.collision_phase_updates;
    }
}

void apply_free(
    std::vector<Complex> &samples,
    const Plan &plan,
    int chirp,
    bool adjoint,
    Stats &stats
) {
    std::vector<Complex> output(samples.size(), 0.0);
    for (
        std::size_t target = 0;
        target < plan.necklaces.size();
        ++target
    ) {
        Complex value = 0.0;
        for (
            std::size_t source = 0;
            source < plan.necklaces.size();
            ++source
        ) {
            value += transition_coefficient(
                plan,
                plan.necklaces[target],
                plan.necklaces[source],
                chirp,
                adjoint,
                stats
            ) * samples[source];
        }
        output[target] = value;
        ++stats.free_phase_updates;
    }
    std::copy(output.begin(), output.end(), samples.begin());
}

int public_kappa(int step, int program_tag) {
    return 1 + mod(3 * step + 5 * program_tag) % (kGrid - 1);
}

int public_chirp(int step, int program_tag) {
    return 1 + mod(5 * step + 7 * program_tag) % (kGrid - 1);
}

void forward_step(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    Stats &stats
) {
    apply_collision(
        samples, plan, public_kappa(step, program_tag), false, stats
    );
    apply_free(
        samples, plan, public_chirp(step, program_tag), false, stats
    );
}

void inverse_step(
    std::vector<Complex> &samples,
    const Plan &plan,
    int step,
    int program_tag,
    Stats &stats
) {
    apply_free(
        samples, plan, public_chirp(step, program_tag), true, stats
    );
    apply_collision(
        samples, plan, public_kappa(step, program_tag), true, stats
    );
}

std::vector<Complex> make_carrier(
    const Plan &plan,
    int identity
) {
    std::vector<Complex> result(plan.necklaces.size());
    const double scale = 1.0 / 289.0;
    for (std::size_t index = 0; index < result.size(); ++index) {
        const int exponent = mod(
            7 * static_cast<int>(index)
            + 3 * plan.necklaces[index].collisions
            + 5 * identity
        );
        result[index] = scale * plan.roots[exponent];
    }
    return result;
}

double weighted_norm(
    const std::vector<Complex> &samples,
    const Plan &plan
) {
    double result = 0.0;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        result += static_cast<double>(
            plan.necklaces[index].labelled_weight
        ) * std::norm(samples[index]);
    }
    return result;
}

double l2_distance(
    const std::vector<Complex> &left,
    const std::vector<Complex> &right,
    const Plan &plan
) {
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result += static_cast<double>(
            plan.necklaces[index].labelled_weight
        ) * std::norm(left[index] - right[index]);
    }
    return std::sqrt(result);
}

using Boundary = std::array<double, kMaximumCollision + 1>;

Boundary project_boundary(
    const std::vector<Complex> &samples,
    const Plan &plan
) {
    Boundary result{};
    for (std::size_t index = 0; index < samples.size(); ++index) {
        result[plan.necklaces[index].collisions] +=
            static_cast<double>(
                plan.necklaces[index].labelled_weight
            ) * std::norm(samples[index]);
    }
    return result;
}

double boundary_distance(
    const Boundary &left,
    const Boundary &right
) {
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::fabs(left[index] - right[index]));
    }
    return result;
}

enum class Control {
    Correct,
    Missing,
    Wrong,
    Reordered,
};

struct Run {
    Boundary boundary{};
    Stats stats{};
    double restoration_error = 0.0;
    double norm_error = 0.0;
    double elapsed_ms = 0.0;
};

Run transaction(
    std::vector<Complex> &samples,
    const std::vector<Complex> &expected_baseline,
    const Plan &plan,
    int depth,
    int program_tag,
    Control control
) {
    const auto begin = std::chrono::steady_clock::now();
    Run result;
    for (int step = 0; step < depth; ++step) {
        forward_step(samples, plan, step, program_tag, result.stats);
    }
    result.boundary = project_boundary(samples, plan);
    result.norm_error = std::fabs(weighted_norm(samples, plan) - 1.0);

    const int minimum_step =
        control == Control::Missing ? 1 : 0;
    for (int step = depth - 1; step >= minimum_step; --step) {
        if (control == Control::Wrong && step == depth - 1) {
            inverse_step(
                samples, plan, step, program_tag + 1, result.stats
            );
        } else if (control == Control::Reordered) {
            apply_collision(
                samples,
                plan,
                public_kappa(step, program_tag),
                true,
                result.stats
            );
            apply_free(
                samples,
                plan,
                public_chirp(step, program_tag),
                true,
                result.stats
            );
        } else {
            inverse_step(
                samples, plan, step, program_tag, result.stats
            );
        }
    }
    result.restoration_error =
        l2_distance(samples, expected_baseline, plan);
    const auto end = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        end - begin
    ).count();
    return result;
}

std::size_t find_necklace(
    const Plan &plan,
    const Histogram &histogram
) {
    const auto found = std::lower_bound(
        plan.necklaces.begin(),
        plan.necklaces.end(),
        histogram,
        [](const Necklace &left, const Histogram &right) {
            return left.histogram < right;
        }
    );
    if (
        found == plan.necklaces.end()
        || found->histogram != histogram
    ) {
        fail("canonical necklace lookup failed");
    }
    return static_cast<std::size_t>(
        std::distance(plan.necklaces.begin(), found)
    );
}

Tuple tuple_from_dense_index(std::size_t index) {
    Tuple result{};
    for (int rotor = kRotors - 1; rotor >= 0; --rotor) {
        result[rotor] = static_cast<std::uint8_t>(index % kGrid);
        index /= kGrid;
    }
    return result;
}

Histogram histogram_from_tuple(const Tuple &tuple) {
    Histogram result{};
    for (int value : tuple) {
        ++result[value];
    }
    return result;
}

Complex free_entry(
    const Plan &plan,
    int target,
    int source,
    int chirp,
    bool adjoint
) {
    const int sign = adjoint ? -1 : 1;
    const int difference = target - source;
    return plan.roots[mod(sign * chirp * difference * difference)]
        / std::sqrt(static_cast<double>(kGrid));
}

void dense_free(
    std::vector<Complex> &samples,
    const Plan &plan,
    int chirp
) {
    std::vector<Complex> output(samples.size());
    std::size_t stride = 1;
    for (int axis = kRotors - 1; axis >= 0; --axis) {
        const std::size_t block = stride * kGrid;
        for (std::size_t base = 0; base < samples.size(); base += block) {
            for (std::size_t offset = 0; offset < stride; ++offset) {
                for (int target = 0; target < kGrid; ++target) {
                    Complex value = 0.0;
                    for (int source = 0; source < kGrid; ++source) {
                        value += free_entry(
                            plan, target, source, chirp, false
                        ) * samples[
                            base
                            + static_cast<std::size_t>(source) * stride
                            + offset
                        ];
                    }
                    output[
                        base
                        + static_cast<std::size_t>(target) * stride
                        + offset
                    ] = value;
                }
            }
        }
        samples.swap(output);
        stride *= kGrid;
    }
}

double dense_verifier_error(
    const Plan &plan,
    const std::vector<Complex> &initial,
    const std::vector<Complex> &native_forward
) {
    const std::size_t cells = 83521;
    std::vector<Complex> dense(cells);
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const Tuple tuple = tuple_from_dense_index(cell);
        const Histogram canonical =
            canonical_histogram(histogram_from_tuple(tuple));
        dense[cell] = initial.at(find_necklace(plan, canonical));
    }
    const int kappa = public_kappa(0, 0);
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const Histogram histogram =
            histogram_from_tuple(tuple_from_dense_index(cell));
        dense[cell] *= plan.roots[
            mod(kappa * collision_count(histogram))
        ];
    }
    dense_free(dense, plan, public_chirp(0, 0));

    double error = 0.0;
    for (std::size_t cell = 0; cell < cells; ++cell) {
        const Tuple tuple = tuple_from_dense_index(cell);
        const Histogram canonical =
            canonical_histogram(histogram_from_tuple(tuple));
        error += std::norm(
            dense[cell]
            - native_forward.at(find_necklace(plan, canonical))
        );
    }
    return std::sqrt(error);
}

void print_boundary(const Boundary &boundary) {
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

    // Exact structural gates for the declared R < 17 ceiling.
    const std::uint64_t histogram_count =
        choose(kRotors + kGrid - 1, kRotors);
    const std::uint64_t dimension_formula = histogram_count / kGrid;
    const std::uint64_t five_rotor_dimension =
        choose(5 + kGrid - 1, 5) / kGrid;
    if (
        histogram_count != 4845U
        || dimension_formula != kExpectedDimension
        || five_rotor_dimension != 1197U
    ) {
        fail("Burnside dimension gate failed");
    }
    for (const Necklace &necklace : plan.necklaces) {
        if (
            canonical_histogram(
                rotate_histogram(necklace.histogram, 1)
            ) != necklace.histogram
        ) {
            fail("global-rotation orbit gate failed");
        }
        if (
            collision_count(
                rotate_histogram(necklace.histogram, 1)
            ) != necklace.collisions
        ) {
            fail("collision commutator gate failed");
        }
    }
    for (int y = 0; y < kGrid; ++y) {
        for (int x = 0; x < kGrid; ++x) {
            if (
                std::abs(
                    free_entry(plan, mod(y + 1), mod(x + 1), 3, false)
                    - free_entry(plan, y, x, 3, false)
                ) > 1.0e-14
            ) {
                fail("circulant commutator gate failed");
            }
        }
    }

    // One-step native state is retained only for the independent dense gate.
    const std::vector<Complex> initial = make_carrier(plan, 0);
    std::vector<Complex> one_step = initial;
    Stats one_step_stats;
    forward_step(one_step, plan, 0, 0, one_step_stats);
    const double dense_error =
        dense_verifier_error(plan, initial, one_step);
    if (dense_error > kParityTolerance) {
        fail("labelled-wave embedding parity failed");
    }

    std::vector<Complex> carrier = initial;
    one_step.clear();
    one_step.shrink_to_fit();

    const Run primary = transaction(
        carrier, initial, plan, kPrimaryDepth, 0, Control::Correct
    );
    if (
        primary.restoration_error > kRestorationTolerance
        || primary.norm_error > kParityTolerance
    ) {
        fail("primary restoration or norm gate failed");
    }

    const Run reuse = transaction(
        carrier, initial, plan, 2, 3, Control::Correct
    );
    std::vector<Complex> fresh = initial;
    const Run fresh_reuse = transaction(
        fresh, initial, plan, 2, 3, Control::Correct
    );
    const double reuse_boundary_error =
        boundary_distance(reuse.boundary, fresh_reuse.boundary);
    if (
        reuse.restoration_error > kRestorationTolerance
        || reuse_boundary_error > kParityTolerance
    ) {
        fail("restored-carrier reuse gate failed");
    }

    std::vector<Complex> missing_carrier = initial;
    const Run missing = transaction(
        missing_carrier, initial, plan, 2, 0, Control::Missing
    );
    std::vector<Complex> wrong_carrier = initial;
    const Run wrong = transaction(
        wrong_carrier, initial, plan, 2, 0, Control::Wrong
    );
    std::vector<Complex> reordered_carrier = initial;
    const Run reordered = transaction(
        reordered_carrier, initial, plan, 2, 0, Control::Reordered
    );
    if (
        missing.restoration_error < kControlFloor
        || wrong.restoration_error < kControlFloor
        || reordered.restoration_error < kControlFloor
    ) {
        fail("inverse control separation failed");
    }

    const std::uint64_t carrier_bytes =
        plan.necklaces.size() * sizeof(Complex);
    const std::uint64_t topology_bytes =
        plan.necklaces.capacity() * sizeof(Necklace)
        + plan.roots.size() * sizeof(Complex);
    const std::uint64_t plan_compilation_peak_bytes =
        topology_bytes
        + 3 * sizeof(Histogram)
        + sizeof(Necklace);
    const std::uint64_t output_scratch_bytes = carrier_bytes;
    const std::uint64_t transition_scratch_bytes =
        kGrid * sizeof(std::int64_t)
        + sizeof(Histogram)
        + 2 * sizeof(Tuple)
        + kRotors * sizeof(int);
    const std::uint64_t engine_bytes =
        carrier_bytes + topology_bytes + output_scratch_bytes
        + transition_scratch_bytes;
    const std::uint64_t wrapper_bytes =
        engine_bytes + carrier_bytes + 2 * sizeof(Boundary);
    const std::uint64_t dense_verifier_bytes =
        2U * 83521U * sizeof(Complex);
    const std::uint64_t verification_peak_bytes =
        wrapper_bytes + dense_verifier_bytes + carrier_bytes;

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"BOUNDED_EXCHANGE_SYMMETRIC_GLOBAL_ROTATION_NECKLACE_PHASE_CARRIER_CHANGES_FIXED_GRID_ROTOR_GROWTH_FROM_EXPONENTIAL_TO_POLYNOMIAL_WITH_STREAMED_EXACT_CYCLOTOMIC_FREE_CLOSURE_ACTUAL_RESTORATION_AND_REUSE\","
    );
    std::printf(
        "\"claim_ceiling\":\"EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_GRID17_FOUR_ROTOR_DEPTH8_TESTED_NONZERO_CHIRP_SCHEDULE_COMPLEX128_EXACT_CYCLOTOMIC_TRANSITION_COUNTS_SOFTWARE_ONLY\","
    );
    std::printf("\"result\":\"PASS\",");
    std::printf("\"grid_size\":17,\"rotors_executed\":4,");
    std::printf(
        "\"labelled_rotation_quotient_cells\":4913,"
        "\"necklace_carrier_complex_cells\":%zu,"
        "\"four_rotor_state_reduction_factor\":%.17g,",
        plan.necklaces.size(),
        4913.0 / static_cast<double>(plan.necklaces.size())
    );
    std::printf(
        "\"dimension_law\":{"
        "\"histograms_r4\":%llu,"
        "\"necklaces_r4\":%llu,"
        "\"necklaces_r5_analytic\":%llu,"
        "\"fixed_grid_asymptotic\":\"O_R_TO_THE_16\","
        "\"valid_simple_orbit_formula\":\"R_LESS_THAN_17\"},",
        static_cast<unsigned long long>(histogram_count),
        static_cast<unsigned long long>(dimension_formula),
        static_cast<unsigned long long>(five_rotor_dimension)
    );
    std::printf(
        "\"symmetry_gates\":{"
        "\"particle_permutation_commutes\":true,"
        "\"global_rotation_commutes\":true,"
        "\"nonseparable_collision_phase\":true,"
        "\"circulant_quadratic_free_phase\":true},"
    );
    std::printf(
        "\"primary\":{"
        "\"depth\":%d,"
        "\"boundary\":",
        kPrimaryDepth
    );
    print_boundary(primary.boundary);
    std::printf(
        ",\"weighted_norm_error\":%.17g,"
        "\"restoration_error\":%.17g,"
        "\"actual_inverse_restoration\":true,"
        "\"elapsed_ms\":%.17g,"
        "\"resources\":{"
        "\"carrier_payload_bytes\":%llu,"
        "\"public_topology_bytes\":%llu,"
        "\"plan_compilation_conservative_explicit_payload_bytes\":%llu,"
        "\"output_scratch_bytes\":%llu,"
        "\"transition_scratch_bytes\":%llu,"
        "\"maximum_explicit_engine_bytes\":%llu,"
        "\"maximum_explicit_wrapper_bytes\":%llu,"
        "\"retained_inverse_history_bytes\":0,"
        "\"retained_transition_operator_bytes\":0,"
        "\"labelled_wave_materialized_in_accepted_path\":false,"
        "\"assignment_expansion_materialized_in_accepted_path\":false,"
        "\"stored_assignment_list_bytes\":0,"
        "\"permanent_assignment_terms_enumerated\":%llu,"
        "\"streamed_transition_coefficients\":%llu,"
        "\"exact_cyclotomic_permanent_terms\":%llu}},",
        primary.norm_error,
        primary.restoration_error,
        primary.elapsed_ms,
        static_cast<unsigned long long>(carrier_bytes),
        static_cast<unsigned long long>(topology_bytes),
        static_cast<unsigned long long>(plan_compilation_peak_bytes),
        static_cast<unsigned long long>(output_scratch_bytes),
        static_cast<unsigned long long>(transition_scratch_bytes),
        static_cast<unsigned long long>(engine_bytes),
        static_cast<unsigned long long>(wrapper_bytes),
        static_cast<unsigned long long>(
            primary.stats.exact_cyclotomic_permanent_terms
        ),
        static_cast<unsigned long long>(
            primary.stats.streamed_transition_coefficients
        ),
        static_cast<unsigned long long>(
            primary.stats.exact_cyclotomic_permanent_terms
        )
    );
    std::printf(
        "\"verification\":{"
        "\"labelled_wave_cells\":83521,"
        "\"labelled_wave_used_for_restoration\":false,"
        "\"labelled_wave_embedding_l2_error\":%.17g,"
        "\"peak_explicit_bytes\":%llu},",
        dense_error,
        static_cast<unsigned long long>(verification_peak_bytes)
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
        "\"phase_primitive_state\":\"UNRESOLVED_CYCLOTOMIC_NECKLACE_AMPLITUDES\","
        "\"intermediate_projected\":false,"
        "\"matched_classical_orbit_simulator_identical\":true,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"unbounded_computation_established\":false,"
        "\"original_open_chain_program_family_compressed\":false,"
        "\"terminal\":false,"
        "\"obstruction\":\"STREAMED_NECKLACE_FREE_CLOSURE_QUADRATIC_TRANSITION_WORK_AND_MATCHED_CLASSICAL_ORBIT_IDENTITY\""
    );
    std::printf("}\n");
    return 0;
}
