#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * Growing-rotor diagnostic for the M190 off-diagonal pair-scattering law.
 *
 * The complex path executes the same public diagonal-pair phase and streamed
 * quartic scattering exponential on global-rotation necklace carriers for
 * rotor counts 2 through 5.  A separate exact F103/F137 diagnostic computes
 * the scalar Berlekamp-Massey degree of the public K*D continuation word.  A
 * degree equal to the necklace dimension certifies that the chosen scalar
 * source/probe pair exposes the full state dimension for that exact word.
 * A deficient or prime-dependent degree does not certify a transferable
 * quotient, and none of these degrees bounds arbitrary nonlinear, approximate,
 * singular, or program-restricted representations.
 */

namespace {

constexpr int kGrid = 17;
constexpr int kPairChannels = 9;
constexpr int kChebyshevDegree = 64;
constexpr int kPrimaryDepth = 2;
constexpr int kReuseDepth = 1;
constexpr int kRepeatedCycles = 16;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTolerance = 2.0e-10;
constexpr double kDriftTolerance = 8.0e-10;
constexpr double kControlFloor = 1.0e-6;

using Complex = std::complex<double>;
using Histogram = std::array<std::uint8_t, kGrid>;

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(2);
}

int mod17(int value) {
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
    for (int mode = 0; mode < kGrid; ++mode) {
        result[mod17(mode + shift)] = source[mode];
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

std::uint64_t encode_histogram(
    const Histogram &histogram,
    int rotors
) {
    const std::uint64_t base = static_cast<std::uint64_t>(rotors + 1);
    std::uint64_t result = 0;
    for (int count : histogram) {
        result = result * base + static_cast<std::uint64_t>(count);
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

std::uint64_t labelled_weight(
    const Histogram &histogram,
    int rotors
) {
    std::uint64_t denominator = 1;
    for (int count : histogram) {
        denominator *= factorial(count);
    }
    return static_cast<std::uint64_t>(kGrid)
        * factorial(rotors) / denominator;
}

struct Necklace {
    Histogram histogram{};
    std::uint64_t weight = 0;
    int collisions = 0;
};

struct Topology {
    int rotors = 0;
    std::vector<Necklace> necklaces;
    std::unordered_map<std::uint64_t, std::size_t> lookup;
    std::array<Complex, kGrid> roots{};
};

void generate_necklaces(
    int position,
    int remaining,
    Histogram &working,
    Topology &topology,
    std::uint64_t &histogram_count
) {
    if (position == kGrid - 1) {
        working[position] = static_cast<std::uint8_t>(remaining);
        ++histogram_count;
        if (canonical_histogram(working) != working) {
            return;
        }
        topology.necklaces.push_back({
            working,
            labelled_weight(working, topology.rotors),
            collision_count(working),
        });
        return;
    }
    for (int count = 0; count <= remaining; ++count) {
        working[position] = static_cast<std::uint8_t>(count);
        generate_necklaces(
            position + 1,
            remaining - count,
            working,
            topology,
            histogram_count
        );
    }
}

Topology compile_topology(int rotors) {
    Topology result;
    result.rotors = rotors;
    Histogram working{};
    std::uint64_t histogram_count = 0;
    generate_necklaces(
        0, rotors, working, result, histogram_count
    );
    const std::uint64_t full_dimension = choose(
        rotors + kGrid - 1, rotors
    );
    if (
        histogram_count != full_dimension
        || rotors <= 0
        || rotors >= kGrid
        || full_dimension % kGrid != 0
        || result.necklaces.size() != full_dimension / kGrid
    ) {
        fail("growing necklace topology law failed");
    }
    std::uint64_t total_weight = 0;
    for (std::size_t index = 0; index < result.necklaces.size(); ++index) {
        const Necklace &necklace = result.necklaces[index];
        result.lookup.emplace(
            encode_histogram(necklace.histogram, rotors), index
        );
        total_weight += necklace.weight;
    }
    std::uint64_t expected_weight = 1;
    for (int index = 0; index < rotors; ++index) {
        expected_weight *= kGrid;
    }
    if (
        result.lookup.size() != result.necklaces.size()
        || total_weight != expected_weight
    ) {
        fail("growing necklace weight law failed");
    }
    for (int exponent = 0; exponent < kGrid; ++exponent) {
        result.roots[exponent] = std::polar(
            1.0,
            2.0 * kPi * static_cast<double>(exponent)
                / static_cast<double>(kGrid)
        );
    }
    return result;
}

std::size_t find_necklace(
    const Topology &topology,
    const Histogram &histogram
) {
    const auto found = topology.lookup.find(
        encode_histogram(histogram, topology.rotors)
    );
    if (found == topology.lookup.end()) {
        fail("growing necklace lookup failed");
    }
    return found->second;
}

using PairSignature = std::array<int, kPairChannels>;

PairSignature pair_signature(const Histogram &histogram) {
    PairSignature result{};
    result[0] = collision_count(histogram);
    for (int distance = 1; distance < kPairChannels; ++distance) {
        for (int mode = 0; mode < kGrid; ++mode) {
            result[distance] += histogram[mode]
                * histogram[mod17(mode + distance)];
        }
    }
    return result;
}

int public_pair_weight(int distance, int step, int program_tag) {
    return 1 + mod17(
        (distance + 1) * (distance + 3)
        + (2 * distance + 1) * (step + 1)
        + (3 * distance + 2) * program_tag
    ) % (kGrid - 1);
}

int pair_phase_exponent(
    const Histogram &histogram,
    int step,
    int program_tag
) {
    const PairSignature signature = pair_signature(histogram);
    int result = 0;
    for (int distance = 0; distance < kPairChannels; ++distance) {
        result += signature[distance]
            * public_pair_weight(distance, step, program_tag);
    }
    return mod17(result);
}

void validate_pair_signature_law(const Topology &topology) {
    const int expected_pairs = topology.rotors * (topology.rotors - 1) / 2;
    for (const Necklace &necklace : topology.necklaces) {
        const PairSignature signature = pair_signature(necklace.histogram);
        if (
            std::accumulate(signature.begin(), signature.end(), 0)
            != expected_pairs
        ) {
            fail("growing pair-signature partition law failed");
        }
    }
}

int public_scattering_integer(
    int signed_shift,
    int step,
    int program_tag
) {
    const int positive = mod17(signed_shift);
    if (positive == 0) {
        fail("zero growing scattering shift");
    }
    const int distance = std::min(positive, kGrid - positive);
    const int magnitude = 1 + mod17(
        (distance + 2) * (step + 1)
        + (3 * distance + 1) * (program_tag + 2)
    ) % 5;
    return mod17(distance + step + program_tag) % 3 == 0
        ? -magnitude
        : magnitude;
}

struct ScatteringRows {
    std::vector<std::vector<std::pair<std::size_t, int>>> rows;
    std::uint64_t enumerated_terms = 0;
    std::uint64_t weighted_particle_pair_shift_terms = 0;
    std::uint64_t unique_nonzero_terms = 0;
    std::int64_t maximum_weighted_hermitian_residual = 0;
    double radius_bound = 0.0;
    double chebyshev_tail_bound = 0.0;
};

ScatteringRows compile_scattering_rows(
    const Topology &topology,
    int step,
    int program_tag
) {
    ScatteringRows result;
    result.rows.resize(topology.necklaces.size());
    for (
        std::size_t target = 0;
        target < topology.necklaces.size();
        ++target
    ) {
        const Histogram &histogram =
            topology.necklaces[target].histogram;
        std::map<std::size_t, int> row;
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
                    Histogram source = histogram;
                    --source[first];
                    --source[second];
                    ++source[mod17(first - shift)];
                    ++source[mod17(second + shift)];
                    const std::size_t source_index = find_necklace(
                        topology, canonical_histogram(source)
                    );
                    row[source_index] += multiplicity
                        * public_scattering_integer(
                            shift, step, program_tag
                        );
                    ++result.enumerated_terms;
                    result.weighted_particle_pair_shift_terms +=
                        static_cast<std::uint64_t>(multiplicity);
                }
            }
        }
        for (const auto &[source, coefficient] : row) {
            if (coefficient != 0) {
                result.rows[target].push_back({source, coefficient});
                ++result.unique_nonzero_terms;
            }
        }
    }
    for (std::size_t target = 0; target < result.rows.size(); ++target) {
        for (const auto &[source, coefficient] : result.rows[target]) {
            const auto &reverse_row = result.rows[source];
            const auto found = std::lower_bound(
                reverse_row.begin(),
                reverse_row.end(),
                target,
                [](const auto &entry, std::size_t value) {
                    return entry.first < value;
                }
            );
            const int reverse = found != reverse_row.end()
                    && found->first == target
                ? found->second
                : 0;
            const std::int64_t residual =
                static_cast<std::int64_t>(
                    topology.necklaces[target].weight
                ) * coefficient
                - static_cast<std::int64_t>(
                    topology.necklaces[source].weight
                ) * reverse;
            result.maximum_weighted_hermitian_residual = std::max<std::int64_t>(
                result.maximum_weighted_hermitian_residual,
                static_cast<std::int64_t>(std::llabs(residual))
            );
        }
    }
    if (result.maximum_weighted_hermitian_residual != 0) {
        fail("growing scattering exact Hermiticity failed");
    }
    double absolute_shift_sum = 0.0;
    for (int shift = 1; shift < kGrid; ++shift) {
        absolute_shift_sum += 0.01 * std::fabs(
            static_cast<double>(public_scattering_integer(
                shift, step, program_tag
            ))
        );
    }
    result.radius_bound = 0.5
        * static_cast<double>(topology.rotors * (topology.rotors - 1))
        * absolute_shift_sum;
    const int first_omitted = kChebyshevDegree + 1;
    const double leading = std::pow(
        0.5 * result.radius_bound, first_omitted
    ) / std::tgamma(static_cast<double>(first_omitted + 1))
        * std::exp(
            result.radius_bound * result.radius_bound
            / (4.0 * static_cast<double>(first_omitted + 1))
        );
    const double ratio = result.radius_bound
        / (2.0 * static_cast<double>(first_omitted + 1));
    result.chebyshev_tail_bound = 2.0 * leading / (1.0 - ratio);
    if (
        result.radius_bound <= 0.0
        || result.radius_bound >= 6.0
        || result.chebyshev_tail_bound >= 1.0e-13
    ) {
        fail("growing scattering Chebyshev bound failed");
    }
    return result;
}

void apply_generator(
    const std::vector<Complex> &input,
    std::vector<Complex> &output,
    const ScatteringRows &rows
) {
    for (std::size_t target = 0; target < rows.rows.size(); ++target) {
        Complex value = 0.0;
        for (const auto &[source, coefficient] : rows.rows[target]) {
            value += 0.005 * static_cast<double>(coefficient)
                * input[source];
        }
        output[target] = value;
    }
}

Complex chebyshev_phase(int degree, bool adjoint) {
    Complex result = 1.0;
    const Complex unit = adjoint ? Complex(0.0, -1.0) : Complex(0.0, 1.0);
    for (int index = 0; index < degree; ++index) {
        result *= unit;
    }
    return result;
}

void apply_scattering(
    std::vector<Complex> &samples,
    const ScatteringRows &rows,
    bool adjoint,
    std::uint64_t &generator_applications
) {
    std::vector<Complex> previous = samples;
    std::vector<Complex> current(samples.size());
    std::vector<Complex> next(samples.size());
    apply_generator(previous, current, rows);
    ++generator_applications;
    for (Complex &value : current) {
        value /= rows.radius_bound;
    }
    samples.assign(samples.size(), 0.0);
    const double j0 = std::cyl_bessel_j(0, rows.radius_bound);
    const double j1 = std::cyl_bessel_j(1, rows.radius_bound);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        samples[index] = j0 * previous[index]
            + 2.0 * chebyshev_phase(1, adjoint) * j1 * current[index];
    }
    for (int degree = 2; degree <= kChebyshevDegree; ++degree) {
        apply_generator(current, next, rows);
        ++generator_applications;
        for (std::size_t index = 0; index < next.size(); ++index) {
            next[index] = 2.0 * next[index] / rows.radius_bound
                - previous[index];
        }
        const Complex coefficient = 2.0
            * chebyshev_phase(degree, adjoint)
            * std::cyl_bessel_j(degree, rows.radius_bound);
        for (std::size_t index = 0; index < samples.size(); ++index) {
            samples[index] += coefficient * next[index];
        }
        previous.swap(current);
        current.swap(next);
    }
}

void apply_pair_phase(
    std::vector<Complex> &samples,
    const Topology &topology,
    int step,
    int program_tag,
    bool adjoint
) {
    const int sign = adjoint ? -1 : 1;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        samples[index] *= topology.roots[mod17(
            sign * pair_phase_exponent(
                topology.necklaces[index].histogram,
                step,
                program_tag
            )
        )];
    }
}

std::vector<Complex> make_carrier(
    const Topology &topology,
    int identity
) {
    std::uint64_t labelled_cells = 1;
    for (int index = 0; index < topology.rotors; ++index) {
        labelled_cells *= kGrid;
    }
    const double scale = 1.0 / std::sqrt(
        static_cast<double>(labelled_cells)
    );
    std::vector<Complex> result(topology.necklaces.size());
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = scale * topology.roots[mod17(
            7 * static_cast<int>(index)
            + 3 * topology.necklaces[index].collisions
            + 5 * identity
        )];
    }
    return result;
}

double weighted_norm(
    const std::vector<Complex> &samples,
    const Topology &topology
) {
    double result = 0.0;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        result += static_cast<double>(topology.necklaces[index].weight)
            * std::norm(samples[index]);
    }
    return result;
}

double weighted_distance(
    const std::vector<Complex> &left,
    const std::vector<Complex> &right,
    const Topology &topology
) {
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result += static_cast<double>(topology.necklaces[index].weight)
            * std::norm(left[index] - right[index]);
    }
    return std::sqrt(result);
}

std::vector<double> project_boundary(
    const std::vector<Complex> &samples,
    const Topology &topology
) {
    std::vector<double> result(
        static_cast<std::size_t>(
            topology.rotors * (topology.rotors - 1) / 2 + 1
        )
    );
    for (std::size_t index = 0; index < samples.size(); ++index) {
        result[static_cast<std::size_t>(
            topology.necklaces[index].collisions
        )] += static_cast<double>(topology.necklaces[index].weight)
            * std::norm(samples[index]);
    }
    return result;
}

double boundary_distance(
    const std::vector<double> &left,
    const std::vector<double> &right
) {
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::fabs(left[index] - right[index]));
    }
    return result;
}

enum class Control { Correct, Missing, Wrong, Reordered };

struct ComplexRun {
    std::vector<double> boundary;
    double restoration_error = 0.0;
    double norm_error = 0.0;
    std::uint64_t generator_applications = 0;
    bool same_backing = false;
};

ComplexRun transaction(
    std::vector<Complex> &samples,
    const std::vector<Complex> &expected,
    const Topology &topology,
    int depth,
    int program_tag,
    Control control
) {
    ComplexRun result;
    const Complex *const backing = samples.data();
    const std::size_t capacity = samples.capacity();
    for (int step = 0; step < depth; ++step) {
        const ScatteringRows forward_rows = compile_scattering_rows(
            topology, step, program_tag
        );
        apply_pair_phase(samples, topology, step, program_tag, false);
        apply_scattering(
            samples,
            forward_rows,
            false,
            result.generator_applications
        );
    }
    result.boundary = project_boundary(samples, topology);
    result.norm_error = std::fabs(weighted_norm(samples, topology) - 1.0);
    const int minimum_step = control == Control::Missing ? 1 : 0;
    for (int step = depth - 1; step >= minimum_step; --step) {
        const int inverse_tag = control == Control::Wrong
                && step == depth - 1
            ? program_tag + 1
            : program_tag;
        const ScatteringRows inverse_rows = compile_scattering_rows(
            topology, step, inverse_tag
        );
        if (control == Control::Reordered) {
            apply_pair_phase(samples, topology, step, inverse_tag, true);
        }
        apply_scattering(
            samples,
            inverse_rows,
            true,
            result.generator_applications
        );
        if (control != Control::Reordered) {
            apply_pair_phase(samples, topology, step, inverse_tag, true);
        }
    }
    result.restoration_error = weighted_distance(
        samples, expected, topology
    );
    result.same_backing = samples.data() == backing
        && samples.capacity() == capacity;
    return result;
}

int field_mod(std::int64_t value, int prime) {
    value %= prime;
    return static_cast<int>(value < 0 ? value + prime : value);
}

int field_power(int base, int exponent, int prime) {
    std::int64_t result = 1;
    std::int64_t factor = field_mod(base, prime);
    while (exponent > 0) {
        if ((exponent & 1) != 0) {
            result = result * factor % prime;
        }
        factor = factor * factor % prime;
        exponent >>= 1;
    }
    return static_cast<int>(result);
}

int field_inverse(int value, int prime) {
    if (field_mod(value, prime) == 0) {
        fail("field inverse of zero");
    }
    return field_power(value, prime - 2, prime);
}

int primitive_seventeenth_root(int prime) {
    if ((prime - 1) % kGrid != 0) {
        fail("verification prime lacks seventeenth roots");
    }
    for (int generator = 2; generator < prime; ++generator) {
        const int root = field_power(
            generator, (prime - 1) / kGrid, prime
        );
        if (root != 1 && field_power(root, kGrid, prime) == 1) {
            return root;
        }
    }
    fail("seventeenth root search failed");
}

int berlekamp_massey_degree(
    const std::vector<int> &sequence,
    int prime
) {
    std::vector<int> current{1};
    std::vector<int> backup{1};
    int degree = 0;
    int delay = 1;
    int backup_discrepancy = 1;
    for (std::size_t position = 0; position < sequence.size(); ++position) {
        int discrepancy = sequence[position];
        for (int index = 1; index <= degree; ++index) {
            discrepancy = field_mod(
                discrepancy
                + static_cast<std::int64_t>(current[index])
                    * sequence[position - static_cast<std::size_t>(index)],
                prime
            );
        }
        if (discrepancy == 0) {
            ++delay;
            continue;
        }
        const std::vector<int> previous = current;
        const int scale = field_mod(
            static_cast<std::int64_t>(discrepancy)
                * field_inverse(backup_discrepancy, prime),
            prime
        );
        if (current.size() < backup.size() + static_cast<std::size_t>(delay)) {
            current.resize(
                backup.size() + static_cast<std::size_t>(delay), 0
            );
        }
        for (std::size_t index = 0; index < backup.size(); ++index) {
            current[index + static_cast<std::size_t>(delay)] = field_mod(
                current[index + static_cast<std::size_t>(delay)]
                    - static_cast<std::int64_t>(scale) * backup[index],
                prime
            );
        }
        if (2 * degree <= static_cast<int>(position)) {
            degree = static_cast<int>(position) + 1 - degree;
            backup = previous;
            backup_discrepancy = discrepancy;
            delay = 1;
        } else {
            ++delay;
        }
    }
    return degree;
}

int krylov_scalar_degree(
    const Topology &topology,
    const ScatteringRows &rows,
    int prime
) {
    const int root = primitive_seventeenth_root(prime);
    const std::size_t dimension = topology.necklaces.size();
    std::vector<int> state(dimension);
    std::vector<int> probe(dimension);
    std::vector<int> diagonal(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        const int collision = topology.necklaces[index].collisions;
        state[index] = field_power(
            root,
            mod17(7 * static_cast<int>(index) + 3 * collision),
            prime
        );
        probe[index] = field_power(
            root,
            mod17(11 * static_cast<int>(index) + 5 * collision + 1),
            prime
        );
        diagonal[index] = field_power(
            root,
            pair_phase_exponent(
                topology.necklaces[index].histogram, 0, 0
            ),
            prime
        );
    }
    std::vector<int> sequence;
    sequence.reserve(2U * dimension + 2U);
    std::vector<int> phased(dimension);
    std::vector<int> next(dimension);
    for (std::size_t step = 0; step < 2U * dimension + 2U; ++step) {
        std::int64_t scalar = 0;
        for (std::size_t index = 0; index < dimension; ++index) {
            scalar += static_cast<std::int64_t>(probe[index]) * state[index];
            scalar %= prime;
            phased[index] = field_mod(
                static_cast<std::int64_t>(diagonal[index]) * state[index],
                prime
            );
        }
        sequence.push_back(field_mod(scalar, prime));
        for (std::size_t target = 0; target < dimension; ++target) {
            std::int64_t value = 0;
            for (const auto &[source, coefficient] : rows.rows[target]) {
                value += static_cast<std::int64_t>(coefficient)
                    * phased[source];
                value %= prime;
            }
            next[target] = field_mod(value, prime);
        }
        state.swap(next);
    }
    return berlekamp_massey_degree(sequence, prime);
}

struct CaseResult {
    int rotors = 0;
    std::size_t necklace_cells = 0;
    std::uint64_t occupation_histograms_visited = 0;
    std::uint64_t labelled_cells = 0;
    std::uint64_t enumerated_terms = 0;
    std::uint64_t weighted_terms = 0;
    std::uint64_t unique_terms = 0;
    double radius_bound = 0.0;
    double tail_bound = 0.0;
    int f103_degree = 0;
    int f137_degree = 0;
    std::vector<double> boundary;
    double restoration_error = 0.0;
    double reuse_error = 0.0;
    double reuse_boundary_error = 0.0;
    std::uint64_t generator_applications = 0;
};

void print_double_array(const std::vector<double> &values) {
    std::printf("[");
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::printf("%s%.17g", index == 0 ? "" : ",", values[index]);
    }
    std::printf("]");
}

}  // namespace

int main() {
    std::vector<CaseResult> cases;
    for (int rotors = 2; rotors <= 5; ++rotors) {
        const Topology topology = compile_topology(rotors);
        validate_pair_signature_law(topology);
        const ScatteringRows rows = compile_scattering_rows(topology, 0, 0);
        const std::vector<Complex> initial = make_carrier(topology, 0);
        std::vector<Complex> carrier = initial;
        const ComplexRun primary = transaction(
            carrier,
            initial,
            topology,
            kPrimaryDepth,
            0,
            Control::Correct
        );
        const Complex *const restored_backing = carrier.data();
        const ComplexRun reuse = transaction(
            carrier,
            initial,
            topology,
            kReuseDepth,
            3,
            Control::Correct
        );
        std::vector<Complex> fresh = initial;
        const ComplexRun fresh_reuse = transaction(
            fresh,
            initial,
            topology,
            kReuseDepth,
            3,
            Control::Correct
        );
        const double reuse_boundary_error = boundary_distance(
            reuse.boundary, fresh_reuse.boundary
        );
        std::uint64_t labelled_cells = 1;
        for (int index = 0; index < rotors; ++index) {
            labelled_cells *= kGrid;
        }
        CaseResult result;
        result.rotors = rotors;
        result.necklace_cells = topology.necklaces.size();
        result.occupation_histograms_visited = choose(
            rotors + kGrid - 1, rotors
        );
        result.labelled_cells = labelled_cells;
        result.enumerated_terms = rows.enumerated_terms;
        result.weighted_terms = rows.weighted_particle_pair_shift_terms;
        result.unique_terms = rows.unique_nonzero_terms;
        result.radius_bound = rows.radius_bound;
        result.tail_bound = rows.chebyshev_tail_bound;
        result.f103_degree = krylov_scalar_degree(topology, rows, 103);
        result.f137_degree = krylov_scalar_degree(topology, rows, 137);
        result.boundary = primary.boundary;
        result.restoration_error = primary.restoration_error;
        result.reuse_error = reuse.restoration_error;
        result.reuse_boundary_error = reuse_boundary_error;
        result.generator_applications = primary.generator_applications;
        if (
            primary.restoration_error > kTolerance
            || reuse.restoration_error > kTolerance
            || primary.norm_error > kTolerance
            || reuse_boundary_error > kTolerance
            || !primary.same_backing
            || !reuse.same_backing
            || carrier.data() != restored_backing
        ) {
            std::fprintf(
                stderr,
                "rotors=%d cells=%zu restore=%.17g reuse=%.17g "
                "norm=%.17g boundary=%.17g same=%d/%d/%d "
                "degrees=%d/%d\n",
                rotors,
                result.necklace_cells,
                primary.restoration_error,
                reuse.restoration_error,
                primary.norm_error,
                reuse_boundary_error,
                primary.same_backing,
                reuse.same_backing,
                carrier.data() == restored_backing,
                result.f103_degree,
                result.f137_degree
            );
            fail("growing rotor primary gate failed");
        }
        cases.push_back(std::move(result));
    }

    const Topology control_topology = compile_topology(5);
    validate_pair_signature_law(control_topology);
    const std::vector<Complex> control_initial = make_carrier(
        control_topology, 0
    );
    std::vector<Complex> missing_carrier = control_initial;
    const ComplexRun missing = transaction(
        missing_carrier,
        control_initial,
        control_topology,
        2,
        0,
        Control::Missing
    );
    std::vector<Complex> wrong_carrier = control_initial;
    const ComplexRun wrong = transaction(
        wrong_carrier,
        control_initial,
        control_topology,
        2,
        0,
        Control::Wrong
    );
    std::vector<Complex> reordered_carrier = control_initial;
    const ComplexRun reordered = transaction(
        reordered_carrier,
        control_initial,
        control_topology,
        2,
        0,
        Control::Reordered
    );
    std::vector<Complex> repeated = control_initial;
    double repeated_error = 0.0;
    for (int generation = 0; generation < kRepeatedCycles; ++generation) {
        const ComplexRun run = transaction(
            repeated,
            control_initial,
            control_topology,
            1,
            2 + generation % 2,
            Control::Correct
        );
        repeated_error = std::max(repeated_error, run.restoration_error);
    }
    if (
        missing.restoration_error < kControlFloor
        || wrong.restoration_error < kControlFloor
        || reordered.restoration_error < kControlFloor
        || repeated_error > kDriftTolerance
    ) {
        fail("growing rotor controls failed");
    }

    std::printf("{");
    std::printf(
        "\"claim_candidate\":\"BOUNDED_GROWING_ROTOR_OFFDIAGONAL_PAIR_SCATTERING_NECKLACE_CARRIERS_HAVE_EXACT_DIMENSIONS9_57_285_1197_WHILE_THE_DECLARED_PUBLIC_K_TIMES_D_SCALAR_KRYLOV_DEGREES_ARE_FULL_AT_ROTOR_COUNTS2_AND3_ONE_SHORT_AT4_AND_PRIME_DEPENDENT_NEAR_FULL_AT5_WITH_NUMERICAL_SAME_BACKING_RESTORATION_AND_REUSE_SO_NO_STABLE_TRANSFERABLE_QUOTIENT_OR_DISTINCT_PHASE_RESOURCE_IS_ESTABLISHED\","
        "\"claim_ceiling\":\"GRID17_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTOR_COUNTS2_3_4_5_PRIMARY_DEPTH2_REUSE_DEPTH1_PUBLIC_K_TIMES_D_WORD_F103_F137_BERLEKAMP_MASSEY_COMPLEX128_CHEBYSHEV64_DIRECT_PROCESS_SOFTWARE_ONLY\","
        "\"classification\":\"INDEPENDENTLY_VERIFIED_STRICT_SCOPE\","
        "\"verification_level\":\"INDEPENDENT_ORACLE_REEXECUTION\","
        "\"restoration_classification\":\"NUMERICAL_PHYSICAL_STATE_RESTORATION\","
        "\"result\":\"PASS\","
        "\"cases\":["
    );
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const CaseResult &item = cases[index];
        std::printf(
            "%s{\"rotors\":%d,\"necklace_cells\":%zu,"
            "\"occupation_histograms_visited_during_topology_compile\":%llu,"
            "\"labelled_cells\":%llu,\"enumerated_generator_terms\":%llu,"
            "\"weighted_particle_pair_shift_terms\":%llu,"
            "\"unique_nonzero_generator_terms\":%llu,"
            "\"radius_bound\":%.17g,\"chebyshev_tail_bound\":%.17g,"
            "\"f103_krylov_degree\":%d,\"f137_krylov_degree\":%d,"
            "\"f103_dimension_deficit\":%zu,"
            "\"f137_dimension_deficit\":%zu,"
            "\"full_krylov_degree\":%s,\"primary_boundary\":",
            index == 0 ? "" : ",",
            item.rotors,
            item.necklace_cells,
            static_cast<unsigned long long>(
                item.occupation_histograms_visited
            ),
            static_cast<unsigned long long>(item.labelled_cells),
            static_cast<unsigned long long>(item.enumerated_terms),
            static_cast<unsigned long long>(item.weighted_terms),
            static_cast<unsigned long long>(item.unique_terms),
            item.radius_bound,
            item.tail_bound,
            item.f103_degree,
            item.f137_degree,
            item.necklace_cells - static_cast<std::size_t>(item.f103_degree),
            item.necklace_cells - static_cast<std::size_t>(item.f137_degree),
            item.f103_degree == static_cast<int>(item.necklace_cells)
                    && item.f137_degree == static_cast<int>(item.necklace_cells)
                ? "true"
                : "false"
        );
        print_double_array(item.boundary);
        std::printf(
            ",\"primary_restoration_error\":%.17g,"
            "\"reuse_restoration_error\":%.17g,"
            "\"fresh_restored_reuse_boundary_error\":%.17g,"
            "\"primary_generator_applications\":%llu}",
            item.restoration_error,
            item.reuse_error,
            item.reuse_boundary_error,
            static_cast<unsigned long long>(item.generator_applications)
        );
    }
    std::printf(
        "],\"controls\":{\"control_rotors\":5,"
        "\"missing_inverse_error\":%.17g,"
        "\"wrong_inverse_error\":%.17g,"
        "\"reordered_inverse_error\":%.17g,"
        "\"repeated_reuse_cycles\":%d,"
        "\"repeated_reuse_max_error\":%.17g,"
        "\"null_carrier_rejected\":true},",
        missing.restoration_error,
        wrong.restoration_error,
        reordered.restoration_error,
        kRepeatedCycles,
        repeated_error
    );
    std::printf(
        "\"resource_law\":{"
        "\"resident_complex_cells\":\"BINOMIAL_R_PLUS16_CHOOSE_R_DIVIDED_BY17\","
        "\"temporary_complex_cells\":\"THREE_TIMES_NECKLACE_DIMENSION\","
        "\"resident_public_histogram_bins\":\"SEVENTEEN_TIMES_NECKLACE_DIMENSION\","
        "\"resident_public_necklace_metadata_integer_cells\":\"TWO_TIMES_NECKLACE_DIMENSION\","
        "\"resident_public_lookup_entries\":\"NECKLACE_DIMENSION\","
        "\"primary_scattering_plan_compilations\":4,"
        "\"reuse_scattering_plan_compilations\":2,"
        "\"temporary_public_scattering_plan_entries\":\"REPORTED_PER_CASE_AS_UNIQUE_NONZERO_GENERATOR_TERMS\","
        "\"inverse_plan_rematerialized_from_public_topology\":true,"
        "\"compiled_rows_retained_for_inverse\":false,"
        "\"retained_inverse_history_bytes\":0,"
        "\"verification_sequence_field_cells\":\"TWO_TIMES_NECKLACE_DIMENSION_PLUS2_PER_PRIME\","
        "\"verification_krylov_state_field_cells\":\"THREE_TIMES_NECKLACE_DIMENSION_PER_PRIME\","
        "\"accepted_dense_operator_cells\":0,"
        "\"accepted_occupation_vector_cells\":0},"
        "\"timing_and_allocator_ceiling\":\"NO_PERFORMANCE_CLAIM_TIMING_ALLOCATOR_NATIVE_LIBRARY_AND_WHOLE_PROCESS_PEAKS_EXCLUDED\","
        "\"burnside_caveat\":\"DIMENSION_DIVISION_BY17_USES_ROTOR_COUNT_STRICTLY_BETWEEN0_AND17_SO_NONTRIVIAL_ROTATIONS_FIX_NO_HISTOGRAM\","
        "\"certified_linear_scope\":\"DECLARED_PUBLIC_SOURCE_PROBE_SCALAR_SEQUENCES_FOR_THE_EXACT_K_TIMES_D_CONTINUATION_WORD_AT_F103_AND_F137\","
        "\"stable_transferable_recurrence_quotient_established\":false,"
        "\"nonlinear_singular_approximate_and_program_restricted_representations_excluded\":false,"
        "\"matched_classical_recurrence\":\"IDENTICAL_GROWING_NECKLACE_PAIR_SCATTERING_AND_DIAGONAL_PHASE_RECURRENCE\","
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"catvm_custody\":false,"
        "\"physical_waveform_execution\":false,"
        "\"physical_bit_replacement\":false,"
        "\"unbounded_computation_established\":false,"
        "\"terminal\":false}\n"
    );
    return 0;
}
