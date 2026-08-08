#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * Exact finite-field diagnostic for the public period-10 SU(2)_8 braid word.
 *
 * The public BraidProgram direction has period two and its sign rule has
 * period five, so ten complete sweeps define one repeated monodromy.  This
 * helper reconstructs the A_9 vacuum paths, compiles only block-diagonal local
 * braid actions, streams scalar vacuum-boundary Krylov sequences, and runs
 * Berlekamp--Massey over split primes.  A full modular scalar degree equals
 * the fusion-path dimension and therefore certifies the exact Q(zeta_40)
 * scalar degree: a nonzero reduced Hankel minor is a nonzero exact minor and
 * the carrier dimension is the matching upper bound.
 *
 * This is diagnostic code, not a distinct phase resource.  The same scalar
 * recurrence is available to ordinary classical software, and all topology,
 * compiled local actions, sequences, and recurrence coefficients are counted.
 */

namespace {

constexpr int kLevel = 8;
constexpr int kLabels = 9;
constexpr int kPeriodSweeps = 10;
constexpr int kHoldoutTerms = 64;
constexpr std::array<int, 2> kSplitPrimes{241, 401};
constexpr std::array<int, 7> kStrands{4, 6, 8, 10, 12, 14, 16};
constexpr std::array<int, 2> kFamilies{0, 1};

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(2);
}

int mod(std::int64_t value, int prime) {
    value %= prime;
    return static_cast<int>(value < 0 ? value + prime : value);
}

int power(int base, int exponent, int prime) {
    std::int64_t result = 1;
    std::int64_t factor = mod(base, prime);
    while (exponent > 0) {
        if ((exponent & 1) != 0) {
            result = result * factor % prime;
        }
        factor = factor * factor % prime;
        exponent >>= 1;
    }
    return static_cast<int>(result);
}

int inverse(int value, int prime) {
    if (mod(value, prime) == 0) {
        fail("finite-field inverse of zero");
    }
    return power(value, prime - 2, prime);
}

std::vector<int> prime_factors(int value) {
    std::vector<int> result;
    for (int candidate = 2; candidate * candidate <= value; ++candidate) {
        if (value % candidate != 0) {
            continue;
        }
        result.push_back(candidate);
        while (value % candidate == 0) {
            value /= candidate;
        }
    }
    if (value > 1) {
        result.push_back(value);
    }
    return result;
}

int primitive_root(int prime) {
    const std::vector<int> factors = prime_factors(prime - 1);
    for (int candidate = 2; candidate < prime; ++candidate) {
        bool primitive = true;
        for (int factor : factors) {
            if (power(candidate, (prime - 1) / factor, prime) == 1) {
                primitive = false;
                break;
            }
        }
        if (primitive) {
            return candidate;
        }
    }
    fail("primitive-root search failed");
}

int root40(int prime) {
    if ((prime - 1) % 40 != 0) {
        fail("diagnostic prime does not split zeta40");
    }
    const int result = power(primitive_root(prime), (prime - 1) / 40, prime);
    if (
        power(result, 40, prime) != 1
        || power(result, 20, prime) == 1
        || power(result, 8, prime) == 1
    ) {
        fail("diagnostic root lacks exact order 40");
    }
    return result;
}

using Path = std::vector<std::uint8_t>;

std::uint64_t encode_path(const Path &path) {
    std::uint64_t result = 0;
    for (std::uint8_t label : path) {
        result = (result << 4U) | label;
    }
    return result;
}

struct Topology {
    int strands = 0;
    std::vector<Path> paths;
    std::unordered_map<std::uint64_t, int> index;
    int vacuum_index = -1;
};

void generate_paths(
    int position,
    int strands,
    Path &path,
    std::vector<Path> &output
) {
    if (position == strands) {
        if (path.back() == 0) {
            output.push_back(path);
        }
        return;
    }
    const int label = path.back();
    if (label > 0) {
        path.push_back(static_cast<std::uint8_t>(label - 1));
        generate_paths(position + 1, strands, path, output);
        path.pop_back();
    }
    if (label < kLevel) {
        path.push_back(static_cast<std::uint8_t>(label + 1));
        generate_paths(position + 1, strands, path, output);
        path.pop_back();
    }
}

Topology compile_topology(int strands) {
    if (strands < 4 || strands % 2 != 0 || strands > 16) {
        fail("invalid declared strand count");
    }
    Topology result;
    result.strands = strands;
    Path path{0};
    generate_paths(0, strands, path, result.paths);
    for (std::size_t index = 0; index < result.paths.size(); ++index) {
        result.index.emplace(encode_path(result.paths[index]), static_cast<int>(index));
    }
    Path vacuum(static_cast<std::size_t>(strands + 1));
    for (int position = 0; position <= strands; ++position) {
        vacuum[static_cast<std::size_t>(position)] = static_cast<std::uint8_t>(position % 2);
    }
    const auto found = result.index.find(encode_path(vacuum));
    if (found == result.index.end() || result.index.size() != result.paths.size()) {
        fail("vacuum path topology compilation failed");
    }
    result.vacuum_index = found->second;
    return result;
}

struct PairSpec {
    int low = 0;
    int high = 0;
    int enclosing = 0;
    int low_label = 0;
    int high_label = 0;
};

enum class ScalarKind { UnequalNeighbors, SingleAlternative };

struct ScalarSpec {
    int index = 0;
    ScalarKind kind = ScalarKind::UnequalNeighbors;
    int enclosing = 0;
    int middle = 0;
};

struct GeneratorShape {
    std::vector<ScalarSpec> scalars;
    std::vector<PairSpec> pairs;
};

std::vector<GeneratorShape> compile_shapes(const Topology &topology) {
    std::vector<GeneratorShape> result(static_cast<std::size_t>(topology.strands));
    for (int generator = 1; generator < topology.strands; ++generator) {
        GeneratorShape &shape = result[static_cast<std::size_t>(generator)];
        std::vector<bool> covered(topology.paths.size(), false);
        for (std::size_t index = 0; index < topology.paths.size(); ++index) {
            if (covered[index]) {
                continue;
            }
            const Path &path = topology.paths[index];
            const int left = path[static_cast<std::size_t>(generator - 1)];
            const int middle = path[static_cast<std::size_t>(generator)];
            const int right = path[static_cast<std::size_t>(generator + 1)];
            if (left != right) {
                shape.scalars.push_back({
                    static_cast<int>(index), ScalarKind::UnequalNeighbors, left, middle
                });
                covered[index] = true;
                continue;
            }
            std::array<int, 2> alternatives{-1, -1};
            int count = 0;
            if (left > 0) {
                alternatives[static_cast<std::size_t>(count++)] = left - 1;
            }
            if (left < kLevel) {
                alternatives[static_cast<std::size_t>(count++)] = left + 1;
            }
            if (count == 1) {
                shape.scalars.push_back({
                    static_cast<int>(index), ScalarKind::SingleAlternative, left, middle
                });
                covered[index] = true;
                continue;
            }
            if (middle != alternatives[0]) {
                fail("uncovered paired braid block");
            }
            Path peer = path;
            peer[static_cast<std::size_t>(generator)] =
                static_cast<std::uint8_t>(alternatives[1]);
            const auto found = topology.index.find(encode_path(peer));
            if (found == topology.index.end()) {
                fail("paired braid peer missing from topology");
            }
            const int peer_index = found->second;
            shape.pairs.push_back({
                static_cast<int>(index), peer_index, left,
                alternatives[0], alternatives[1]
            });
            covered[index] = true;
            covered[static_cast<std::size_t>(peer_index)] = true;
        }
        if (!std::all_of(covered.begin(), covered.end(), [](bool value) { return value; })) {
            fail("local braid shape does not partition carrier");
        }
    }
    return result;
}

struct ScalarAction {
    int index = 0;
    int factor = 0;
};

struct PairAction {
    int low = 0;
    int high = 0;
    int a00 = 0;
    int a01 = 0;
    int a10 = 0;
    int a11 = 0;
};

struct GateAction {
    std::vector<ScalarAction> scalars;
    std::vector<PairAction> pairs;
};

struct FieldActions {
    int prime = 0;
    int root = 0;
    std::vector<std::array<GateAction, 2>> gates;
    std::uint64_t retained_action_records = 0;
    std::uint64_t retained_action_integer_cells = 0;
};

FieldActions compile_actions(
    const Topology &topology,
    const std::vector<GeneratorShape> &shapes,
    int prime
) {
    FieldActions result;
    result.prime = prime;
    result.root = root40(prime);
    std::array<int, kLabels> dimensions{};
    dimensions[0] = 1;
    dimensions[1] = mod(
        power(result.root, 2, prime) + power(result.root, 38, prime), prime
    );
    for (int label = 2; label < kLabels; ++label) {
        dimensions[static_cast<std::size_t>(label)] = mod(
            static_cast<std::int64_t>(dimensions[1])
                * dimensions[static_cast<std::size_t>(label - 1)]
                - dimensions[static_cast<std::size_t>(label - 2)],
            prime
        );
    }
    if (mod(
        static_cast<std::int64_t>(dimensions[1]) * dimensions[8] - dimensions[7],
        prime
    ) != 0) {
        fail("modular Jones-Wenzl relation failed");
    }
    result.gates.resize(static_cast<std::size_t>(topology.strands));
    for (int generator = 1; generator < topology.strands; ++generator) {
        for (int exponent_index = 0; exponent_index < 2; ++exponent_index) {
            const bool positive = exponent_index == 1;
            const int alpha = power(result.root, positive ? 11 : 29, prime);
            const int beta = power(result.root, positive ? 29 : 11, prime);
            GateAction &action = result.gates[static_cast<std::size_t>(generator)]
                [static_cast<std::size_t>(exponent_index)];
            for (const ScalarSpec &spec : shapes[static_cast<std::size_t>(generator)].scalars) {
                int factor = alpha;
                if (spec.kind == ScalarKind::SingleAlternative) {
                    factor = mod(
                        alpha + static_cast<std::int64_t>(beta)
                            * dimensions[static_cast<std::size_t>(spec.middle)]
                            * inverse(dimensions[static_cast<std::size_t>(spec.enclosing)], prime),
                        prime
                    );
                }
                action.scalars.push_back({spec.index, factor});
                ++result.retained_action_records;
                result.retained_action_integer_cells += 2;
            }
            for (const PairSpec &spec : shapes[static_cast<std::size_t>(generator)].pairs) {
                const int enclosing_inverse = inverse(
                    dimensions[static_cast<std::size_t>(spec.enclosing)], prime
                );
                const int low_weight = mod(
                    static_cast<std::int64_t>(beta)
                        * enclosing_inverse
                        * dimensions[static_cast<std::size_t>(spec.low_label)],
                    prime
                );
                const int high_weight = mod(
                    static_cast<std::int64_t>(beta)
                        * enclosing_inverse
                        * dimensions[static_cast<std::size_t>(spec.high_label)],
                    prime
                );
                action.pairs.push_back({
                    spec.low,
                    spec.high,
                    mod(alpha + low_weight, prime),
                    high_weight,
                    low_weight,
                    mod(alpha + high_weight, prime),
                });
                ++result.retained_action_records;
                result.retained_action_integer_cells += 6;
            }
        }
    }
    return result;
}

void apply_gate(
    std::vector<int> &state,
    const GateAction &action,
    int prime
) {
    for (const ScalarAction &scalar : action.scalars) {
        state[static_cast<std::size_t>(scalar.index)] = mod(
            static_cast<std::int64_t>(scalar.factor)
                * state[static_cast<std::size_t>(scalar.index)],
            prime
        );
    }
    for (const PairAction &pair : action.pairs) {
        const int low = state[static_cast<std::size_t>(pair.low)];
        const int high = state[static_cast<std::size_t>(pair.high)];
        state[static_cast<std::size_t>(pair.low)] = mod(
            static_cast<std::int64_t>(pair.a00) * low
                + static_cast<std::int64_t>(pair.a01) * high,
            prime
        );
        state[static_cast<std::size_t>(pair.high)] = mod(
            static_cast<std::int64_t>(pair.a10) * low
                + static_cast<std::int64_t>(pair.a11) * high,
            prime
        );
    }
}

struct Operation {
    int generator = 0;
    int exponent = 0;
};

Operation operation(int strands, int family, int round, int offset) {
    const int generator = (round + family) % 2 != 0
        ? strands - 1 - offset
        : offset + 1;
    const int exponent = (3 * round + generator + family) % 5 == 0 ? -1 : 1;
    return {generator, exponent};
}

void apply_period(
    std::vector<int> &state,
    const FieldActions &actions,
    int strands,
    int family,
    bool perturb_last
) {
    for (int round = 0; round < kPeriodSweeps; ++round) {
        for (int offset = 0; offset < strands - 1; ++offset) {
            Operation item = operation(strands, family, round, offset);
            if (perturb_last && round == kPeriodSweeps - 1 && offset == strands - 2) {
                item.exponent = -item.exponent;
            }
            const GateAction &action = actions.gates[static_cast<std::size_t>(item.generator)]
                [static_cast<std::size_t>(item.exponent == 1 ? 1 : 0)];
            apply_gate(state, action, actions.prime);
        }
    }
}

struct BMResult {
    int degree = 0;
    std::vector<int> connection;
    std::uint64_t peak_connection_cells = 0;
};

BMResult berlekamp_massey(
    const std::vector<int> &sequence,
    int training_terms,
    int prime
) {
    std::vector<int> current{1};
    std::vector<int> backup{1};
    int degree = 0;
    int delay = 1;
    int backup_discrepancy = 1;
    std::uint64_t peak = 2;
    for (int position = 0; position < training_terms; ++position) {
        int discrepancy = sequence[static_cast<std::size_t>(position)];
        for (int index = 1; index <= degree; ++index) {
            discrepancy = mod(
                discrepancy + static_cast<std::int64_t>(current[static_cast<std::size_t>(index)])
                    * sequence[static_cast<std::size_t>(position - index)],
                prime
            );
        }
        if (discrepancy == 0) {
            ++delay;
            continue;
        }
        const std::vector<int> previous = current;
        const int scale = mod(
            static_cast<std::int64_t>(discrepancy)
                * inverse(backup_discrepancy, prime),
            prime
        );
        if (current.size() < backup.size() + static_cast<std::size_t>(delay)) {
            current.resize(backup.size() + static_cast<std::size_t>(delay), 0);
        }
        for (std::size_t index = 0; index < backup.size(); ++index) {
            current[index + static_cast<std::size_t>(delay)] = mod(
                current[index + static_cast<std::size_t>(delay)]
                    - static_cast<std::int64_t>(scale) * backup[index],
                prime
            );
        }
        if (2 * degree <= position) {
            degree = position + 1 - degree;
            backup = previous;
            backup_discrepancy = discrepancy;
            delay = 1;
        } else {
            ++delay;
        }
        peak = std::max<std::uint64_t>(peak, current.size() + backup.size());
    }
    current.resize(static_cast<std::size_t>(degree + 1));
    return {degree, current, peak};
}

int recurrence_violations(
    const std::vector<int> &sequence,
    const BMResult &recurrence,
    int begin,
    int prime
) {
    int result = 0;
    for (int position = std::max(begin, recurrence.degree);
         position < static_cast<int>(sequence.size()); ++position) {
        int discrepancy = sequence[static_cast<std::size_t>(position)];
        for (int lag = 1; lag <= recurrence.degree; ++lag) {
            discrepancy = mod(
                discrepancy
                    + static_cast<std::int64_t>(
                        recurrence.connection[static_cast<std::size_t>(lag)]
                    ) * sequence[static_cast<std::size_t>(position - lag)],
                prime
            );
        }
        if (discrepancy != 0) {
            ++result;
        }
    }
    return result;
}

std::uint64_t fnv1a(const std::vector<int> &values) {
    std::uint64_t result = 1469598103934665603ULL;
    for (int value : values) {
        for (int shift = 0; shift < 32; shift += 8) {
            result ^= static_cast<std::uint8_t>(value >> shift);
            result *= 1099511628211ULL;
        }
    }
    return result;
}

struct PrimeResult {
    int prime = 0;
    int root = 0;
    int degree = 0;
    int nonzero_coefficients = 0;
    int training_violations = 0;
    int holdout_violations = 0;
    int undersampled_degree = 0;
    int undersampled_holdout_violations = 0;
    bool perturbation_changes_sequence = false;
    std::uint64_t sequence_digest_fnv1a64 = 0;
    std::uint64_t recurrence_digest_fnv1a64 = 0;
    std::uint64_t retained_action_records = 0;
    std::uint64_t retained_action_integer_cells = 0;
    std::uint64_t peak_bm_connection_cells = 0;
    std::vector<int> first_terms;
};

PrimeResult diagnose_prime(
    const Topology &topology,
    const std::vector<GeneratorShape> &shapes,
    int family,
    int prime
) {
    const FieldActions actions = compile_actions(topology, shapes, prime);
    const int dimension = static_cast<int>(topology.paths.size());
    const int training_terms = 2 * dimension;
    const int total_terms = training_terms + kHoldoutTerms;
    std::vector<int> state(static_cast<std::size_t>(dimension));
    state[static_cast<std::size_t>(topology.vacuum_index)] = 1;
    std::vector<int> perturbed = state;
    std::vector<int> sequence;
    sequence.reserve(static_cast<std::size_t>(total_terms));
    std::vector<int> perturbed_prefix;
    perturbed_prefix.reserve(4);
    for (int term = 0; term < total_terms; ++term) {
        sequence.push_back(state[static_cast<std::size_t>(topology.vacuum_index)]);
        if (term < 4) {
            perturbed_prefix.push_back(
                perturbed[static_cast<std::size_t>(topology.vacuum_index)]
            );
        }
        if (term + 1 < total_terms) {
            apply_period(state, actions, topology.strands, family, false);
            if (term < 3) {
                apply_period(perturbed, actions, topology.strands, family, true);
            }
        }
    }
    const BMResult recurrence = berlekamp_massey(sequence, training_terms, prime);
    const int undersampled_terms = std::max(2, dimension);
    const BMResult undersampled = berlekamp_massey(
        sequence, undersampled_terms, prime
    );
    PrimeResult result;
    result.prime = prime;
    result.root = actions.root;
    result.degree = recurrence.degree;
    result.nonzero_coefficients = static_cast<int>(std::count_if(
        recurrence.connection.begin(), recurrence.connection.end(),
        [](int value) { return value != 0; }
    ));
    result.training_violations = recurrence_violations(
        sequence, recurrence, recurrence.degree, prime
    ) - recurrence_violations(sequence, recurrence, training_terms, prime);
    result.holdout_violations = recurrence_violations(
        sequence, recurrence, training_terms, prime
    );
    result.undersampled_degree = undersampled.degree;
    result.undersampled_holdout_violations = recurrence_violations(
        sequence, undersampled, undersampled_terms, prime
    );
    result.perturbation_changes_sequence = false;
    for (std::size_t index = 0; index < perturbed_prefix.size(); ++index) {
        if (sequence[index] != perturbed_prefix[index]) {
            result.perturbation_changes_sequence = true;
            break;
        }
    }
    result.sequence_digest_fnv1a64 = fnv1a(sequence);
    result.recurrence_digest_fnv1a64 = fnv1a(recurrence.connection);
    result.retained_action_records = actions.retained_action_records;
    result.retained_action_integer_cells = actions.retained_action_integer_cells;
    result.peak_bm_connection_cells = recurrence.peak_connection_cells;
    const int prefix = std::min(12, static_cast<int>(sequence.size()));
    result.first_terms.assign(sequence.begin(), sequence.begin() + prefix);
    if (result.training_violations != 0 || result.holdout_violations != 0) {
        fail("trained scalar recurrence fails sequence verification");
    }
    return result;
}

struct CaseResult {
    int strands = 0;
    int family = 0;
    int dimension = 0;
    std::uint64_t topology_path_label_cells = 0;
    std::uint64_t structural_shape_records = 0;
    std::uint64_t structural_shape_integer_cells = 0;
    std::array<PrimeResult, 2> primes;
};

CaseResult diagnose_case(int strands, int family) {
    const Topology topology = compile_topology(strands);
    const std::vector<GeneratorShape> shapes = compile_shapes(topology);
    CaseResult result;
    result.strands = strands;
    result.family = family;
    result.dimension = static_cast<int>(topology.paths.size());
    result.topology_path_label_cells = static_cast<std::uint64_t>(
        result.dimension * (strands + 1)
    );
    for (int generator = 1; generator < strands; ++generator) {
        const GeneratorShape &shape = shapes[static_cast<std::size_t>(generator)];
        result.structural_shape_records += shape.scalars.size() + shape.pairs.size();
        result.structural_shape_integer_cells += 4 * shape.scalars.size()
            + 5 * shape.pairs.size();
    }
    for (std::size_t index = 0; index < kSplitPrimes.size(); ++index) {
        result.primes[index] = diagnose_prime(
            topology, shapes, family, kSplitPrimes[index]
        );
    }
    return result;
}

void print_int_array(const std::vector<int> &values) {
    std::printf("[");
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::printf("%s%d", index == 0 ? "" : ",", values[index]);
    }
    std::printf("]");
}

bool period_law() {
    for (int strands : kStrands) {
        for (int family : kFamilies) {
            for (int round = 0; round < 20; ++round) {
                for (int offset = 0; offset < strands - 1; ++offset) {
                    const Operation left = operation(strands, family, round, offset);
                    const Operation right = operation(strands, family, round + 10, offset);
                    if (left.generator != right.generator || left.exponent != right.exponent) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    if (!period_law()) {
        fail("public braid word is not period ten");
    }
    std::vector<CaseResult> cases;
    for (int family : kFamilies) {
        for (int strands : kStrands) {
            cases.push_back(diagnose_case(strands, family));
        }
    }
    bool every_full = true;
    bool every_cross_prime = true;
    bool every_perturbation = true;
    for (const CaseResult &item : cases) {
        every_full = every_full
            && item.primes[0].degree == item.dimension
            && item.primes[1].degree == item.dimension;
        every_cross_prime = every_cross_prime
            && item.primes[0].degree == item.primes[1].degree;
        every_perturbation = every_perturbation
            && item.primes[0].perturbation_changes_sequence
            && item.primes[1].perturbation_changes_sequence;
    }
    std::printf("{");
    std::printf(
        "\"schema\":\"cat_cas.su2_level8_period10_monodromy_krylov_core.v1\","
        "\"public_word_period_sweeps\":10,"
        "\"period_law_verified\":true,"
        "\"split_primes\":[241,401],"
        "\"every_case_cross_prime_degree_agreement\":%s,"
        "\"every_case_full_scalar_degree\":%s,"
        "\"every_semantic_perturbation_changes_prefix\":%s,"
        "\"cases\":[",
        every_cross_prime ? "true" : "false",
        every_full ? "true" : "false",
        every_perturbation ? "true" : "false"
    );
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const CaseResult &item = cases[case_index];
        std::printf(
            "%s{\"strands\":%d,\"family\":%d,\"fusion_path_cells\":%d,"
            "\"topology_path_label_cells\":%llu,"
            "\"structural_shape_records\":%llu,"
            "\"structural_shape_integer_cells\":%llu,\"prime_results\":[",
            case_index == 0 ? "" : ",",
            item.strands,
            item.family,
            item.dimension,
            static_cast<unsigned long long>(item.topology_path_label_cells),
            static_cast<unsigned long long>(item.structural_shape_records),
            static_cast<unsigned long long>(item.structural_shape_integer_cells)
        );
        for (std::size_t prime_index = 0; prime_index < item.primes.size(); ++prime_index) {
            const PrimeResult &prime = item.primes[prime_index];
            std::printf(
                "%s{\"prime\":%d,\"root40\":%d,\"scalar_recurrence_degree\":%d,"
                "\"nonzero_recurrence_coefficients\":%d,"
                "\"training_violations\":%d,\"holdout_violations\":%d,"
                "\"undersampled_degree\":%d,\"undersampled_holdout_violations\":%d,"
                "\"semantic_perturbation_changes_prefix\":%s,"
                "\"sequence_digest_fnv1a64\":\"%016llx\","
                "\"recurrence_digest_fnv1a64\":\"%016llx\","
                "\"retained_action_records\":%llu,"
                "\"retained_action_integer_cells\":%llu,"
                "\"peak_bm_connection_cells\":%llu,\"first_terms\":",
                prime_index == 0 ? "" : ",",
                prime.prime,
                prime.root,
                prime.degree,
                prime.nonzero_coefficients,
                prime.training_violations,
                prime.holdout_violations,
                prime.undersampled_degree,
                prime.undersampled_holdout_violations,
                prime.perturbation_changes_sequence ? "true" : "false",
                static_cast<unsigned long long>(prime.sequence_digest_fnv1a64),
                static_cast<unsigned long long>(prime.recurrence_digest_fnv1a64),
                static_cast<unsigned long long>(prime.retained_action_records),
                static_cast<unsigned long long>(prime.retained_action_integer_cells),
                static_cast<unsigned long long>(prime.peak_bm_connection_cells)
            );
            print_int_array(prime.first_terms);
            std::printf("}");
        }
        std::printf("]}");
    }
    std::printf("]}\n");
    return 0;
}
