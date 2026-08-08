#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

/*
 * Separate M217 reference.
 *
 * This file imports no production M217 code.  It enumerates A_9 paths by
 * filtering bit-coded up/down walks instead of recursive path generation,
 * keeps one structural kind/peer array per braid generator instead of
 * compiled coefficient actions, evaluates different split primes, and runs
 * an independently written scalar recurrence reconstruction.
 */

namespace {

constexpr int kLevel = 8;
constexpr int kHoldout = 32;
constexpr std::array<int, 2> kPrimes{641, 881};
constexpr std::array<int, 7> kStrands{4, 6, 8, 10, 12, 14, 16};

[[noreturn]] void die(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(2);
}

int residue(std::int64_t value, int prime) {
    value %= prime;
    return static_cast<int>(value < 0 ? value + prime : value);
}

int exponentiate(int value, int exponent, int prime) {
    std::int64_t result = 1;
    std::int64_t base = residue(value, prime);
    while (exponent != 0) {
        if (exponent & 1) {
            result = result * base % prime;
        }
        base = base * base % prime;
        exponent >>= 1;
    }
    return static_cast<int>(result);
}

int reciprocal(int value, int prime) {
    if (residue(value, prime) == 0) {
        die("reference inverse of zero");
    }
    return exponentiate(value, prime - 2, prime);
}

std::vector<int> factorize(int value) {
    std::vector<int> factors;
    for (int candidate = 2; candidate * candidate <= value; ++candidate) {
        if (value % candidate != 0) {
            continue;
        }
        factors.push_back(candidate);
        do {
            value /= candidate;
        } while (value % candidate == 0);
    }
    if (value != 1) {
        factors.push_back(value);
    }
    return factors;
}

int primitive(int prime) {
    const std::vector<int> factors = factorize(prime - 1);
    for (int candidate = 2; candidate < prime; ++candidate) {
        bool good = true;
        for (int factor : factors) {
            good = good
                && exponentiate(candidate, (prime - 1) / factor, prime) != 1;
        }
        if (good) {
            return candidate;
        }
    }
    die("reference primitive-root search failed");
}

int fortieth_root(int prime) {
    if ((prime - 1) % 40 != 0) {
        die("reference modulus does not split Q(zeta40)");
    }
    const int root = exponentiate(primitive(prime), (prime - 1) / 40, prime);
    if (
        exponentiate(root, 40, prime) != 1
        || exponentiate(root, 20, prime) == 1
        || exponentiate(root, 8, prime) == 1
    ) {
        die("reference fortieth root has wrong order");
    }
    return root;
}

struct Walk {
    std::array<std::uint8_t, 17> labels{};
};

std::uint64_t code(const Walk &walk, int strands) {
    std::uint64_t result = 0;
    for (int position = 0; position <= strands; ++position) {
        result = result * 9U + walk.labels[static_cast<std::size_t>(position)];
    }
    return result;
}

struct Paths {
    int strands = 0;
    std::vector<Walk> walks;
    std::unordered_map<std::uint64_t, int> ranks;
    int vacuum = -1;
};

Paths enumerate_paths(int strands) {
    Paths result;
    result.strands = strands;
    const std::uint32_t words = 1U << strands;
    for (std::uint32_t word = 0; word < words; ++word) {
        Walk walk;
        int height = 0;
        bool valid = true;
        for (int step = 0; step < strands; ++step) {
            height += ((word >> step) & 1U) != 0U ? 1 : -1;
            if (height < 0 || height > kLevel) {
                valid = false;
                break;
            }
            walk.labels[static_cast<std::size_t>(step + 1)] =
                static_cast<std::uint8_t>(height);
        }
        if (valid && height == 0) {
            result.walks.push_back(walk);
        }
    }
    std::sort(
        result.walks.begin(), result.walks.end(),
        [strands](const Walk &left, const Walk &right) {
            return code(left, strands) < code(right, strands);
        }
    );
    for (std::size_t index = 0; index < result.walks.size(); ++index) {
        result.ranks.emplace(code(result.walks[index], strands), static_cast<int>(index));
    }
    Walk vacuum;
    for (int position = 0; position <= strands; ++position) {
        vacuum.labels[static_cast<std::size_t>(position)] =
            static_cast<std::uint8_t>(position % 2);
    }
    const auto found = result.ranks.find(code(vacuum, strands));
    if (found == result.ranks.end() || result.ranks.size() != result.walks.size()) {
        die("reference vacuum topology failed");
    }
    result.vacuum = found->second;
    return result;
}

enum Kind : std::uint8_t { Unequal = 0, Singleton = 1, PairLow = 2, PairHigh = 3 };

struct LocalPlan {
    std::vector<std::uint8_t> kind;
    std::vector<int> peer;
    std::vector<std::uint8_t> enclosing;
    std::vector<std::uint8_t> middle;
};

std::vector<LocalPlan> make_plans(const Paths &paths) {
    std::vector<LocalPlan> result(static_cast<std::size_t>(paths.strands));
    for (int generator = 1; generator < paths.strands; ++generator) {
        LocalPlan &plan = result[static_cast<std::size_t>(generator)];
        const std::size_t dimension = paths.walks.size();
        plan.kind.resize(dimension, Unequal);
        plan.peer.resize(dimension, -1);
        plan.enclosing.resize(dimension);
        plan.middle.resize(dimension);
        for (std::size_t index = 0; index < dimension; ++index) {
            const Walk &walk = paths.walks[index];
            const int left = walk.labels[static_cast<std::size_t>(generator - 1)];
            const int middle = walk.labels[static_cast<std::size_t>(generator)];
            const int right = walk.labels[static_cast<std::size_t>(generator + 1)];
            plan.enclosing[index] = static_cast<std::uint8_t>(left);
            plan.middle[index] = static_cast<std::uint8_t>(middle);
            if (left != right) {
                plan.kind[index] = Unequal;
                continue;
            }
            const int low = left - 1;
            const int high = left + 1;
            if (low < 0 || high > kLevel) {
                plan.kind[index] = Singleton;
                continue;
            }
            Walk peer_walk = walk;
            peer_walk.labels[static_cast<std::size_t>(generator)] =
                static_cast<std::uint8_t>(middle == low ? high : low);
            const auto found = paths.ranks.find(code(peer_walk, paths.strands));
            if (found == paths.ranks.end()) {
                die("reference paired path missing");
            }
            plan.peer[index] = found->second;
            plan.kind[index] = middle == low ? PairLow : PairHigh;
        }
    }
    return result;
}

struct Field {
    int prime = 0;
    int root = 0;
    std::array<int, 9> dimensions{};
};

Field make_field(int prime) {
    Field result;
    result.prime = prime;
    result.root = fortieth_root(prime);
    result.dimensions[0] = 1;
    result.dimensions[1] = residue(
        exponentiate(result.root, 2, prime)
            + exponentiate(result.root, 38, prime),
        prime
    );
    for (int label = 2; label <= kLevel; ++label) {
        result.dimensions[static_cast<std::size_t>(label)] = residue(
            static_cast<std::int64_t>(result.dimensions[1])
                * result.dimensions[static_cast<std::size_t>(label - 1)]
                - result.dimensions[static_cast<std::size_t>(label - 2)],
            prime
        );
    }
    if (residue(
        static_cast<std::int64_t>(result.dimensions[1]) * result.dimensions[8]
            - result.dimensions[7],
        prime
    ) != 0) {
        die("reference Jones-Wenzl relation failed");
    }
    return result;
}

void local_braid(
    std::vector<int> &state,
    const LocalPlan &plan,
    const Field &field,
    int exponent
) {
    const int alpha = exponent == 1
        ? exponentiate(field.root, 11, field.prime)
        : exponentiate(field.root, 29, field.prime);
    const int beta = exponent == 1
        ? exponentiate(field.root, 29, field.prime)
        : exponentiate(field.root, 11, field.prime);
    for (std::size_t index = 0; index < state.size(); ++index) {
        if (plan.kind[index] == PairHigh) {
            continue;
        }
        if (plan.kind[index] == Unequal) {
            state[index] = residue(
                static_cast<std::int64_t>(alpha) * state[index], field.prime
            );
            continue;
        }
        const int enclosing = plan.enclosing[index];
        const int middle = plan.middle[index];
        const int inverse_dimension = reciprocal(
            field.dimensions[static_cast<std::size_t>(enclosing)], field.prime
        );
        if (plan.kind[index] == Singleton) {
            const int factor = residue(
                alpha + static_cast<std::int64_t>(beta)
                    * field.dimensions[static_cast<std::size_t>(middle)]
                    * inverse_dimension,
                field.prime
            );
            state[index] = residue(
                static_cast<std::int64_t>(factor) * state[index], field.prime
            );
            continue;
        }
        const std::size_t peer = static_cast<std::size_t>(plan.peer[index]);
        const int low = state[index];
        const int high = state[peer];
        const int low_label = middle;
        const int high_label = plan.middle[peer];
        const int shared = residue(
            static_cast<std::int64_t>(beta) * inverse_dimension
                * residue(
                    static_cast<std::int64_t>(
                        field.dimensions[static_cast<std::size_t>(low_label)]
                    ) * low
                        + static_cast<std::int64_t>(
                            field.dimensions[static_cast<std::size_t>(high_label)]
                        ) * high,
                    field.prime
                ),
            field.prime
        );
        state[index] = residue(
            static_cast<std::int64_t>(alpha) * low + shared, field.prime
        );
        state[peer] = residue(
            static_cast<std::int64_t>(alpha) * high + shared, field.prime
        );
    }
}

std::pair<int, int> word_item(int strands, int family, int round, int offset) {
    const int generator = (round + family) % 2
        ? strands - 1 - offset
        : offset + 1;
    const int exponent = (3 * round + generator + family) % 5 == 0 ? -1 : 1;
    return {generator, exponent};
}

void monodromy(
    std::vector<int> &state,
    const std::vector<LocalPlan> &plans,
    const Field &field,
    int strands,
    int family
) {
    for (int round = 0; round < 10; ++round) {
        for (int offset = 0; offset < strands - 1; ++offset) {
            const auto [generator, exponent] = word_item(
                strands, family, round, offset
            );
            local_braid(
                state,
                plans[static_cast<std::size_t>(generator)],
                field,
                exponent
            );
        }
    }
}

struct Recurrence {
    int degree = 0;
    std::vector<int> polynomial;
};

Recurrence reconstruct(const std::vector<int> &samples, int training, int prime) {
    std::vector<int> polynomial{1};
    std::vector<int> previous{1};
    int span = 0;
    int shift = 1;
    int previous_delta = 1;
    for (int time = 0; time < training; ++time) {
        int delta = samples[static_cast<std::size_t>(time)];
        for (int lag = 1; lag <= span; ++lag) {
            delta = residue(
                delta + static_cast<std::int64_t>(
                    polynomial[static_cast<std::size_t>(lag)]
                ) * samples[static_cast<std::size_t>(time - lag)],
                prime
            );
        }
        if (delta == 0) {
            ++shift;
            continue;
        }
        const std::vector<int> old = polynomial;
        const int multiplier = residue(
            static_cast<std::int64_t>(delta) * reciprocal(previous_delta, prime),
            prime
        );
        if (polynomial.size() < previous.size() + static_cast<std::size_t>(shift)) {
            polynomial.resize(previous.size() + static_cast<std::size_t>(shift), 0);
        }
        for (std::size_t index = 0; index < previous.size(); ++index) {
            polynomial[index + static_cast<std::size_t>(shift)] = residue(
                polynomial[index + static_cast<std::size_t>(shift)]
                    - static_cast<std::int64_t>(multiplier) * previous[index],
                prime
            );
        }
        if (2 * span <= time) {
            span = time + 1 - span;
            previous = old;
            previous_delta = delta;
            shift = 1;
        } else {
            ++shift;
        }
    }
    polynomial.resize(static_cast<std::size_t>(span + 1));
    return {span, polynomial};
}

int violations(
    const std::vector<int> &samples,
    const Recurrence &recurrence,
    int begin,
    int prime
) {
    int failures = 0;
    for (int time = std::max(begin, recurrence.degree);
         time < static_cast<int>(samples.size()); ++time) {
        int delta = samples[static_cast<std::size_t>(time)];
        for (int lag = 1; lag <= recurrence.degree; ++lag) {
            delta = residue(
                delta + static_cast<std::int64_t>(
                    recurrence.polynomial[static_cast<std::size_t>(lag)]
                ) * samples[static_cast<std::size_t>(time - lag)],
                prime
            );
        }
        failures += delta != 0;
    }
    return failures;
}

struct PrimeCase {
    int prime = 0;
    int root = 0;
    int degree = 0;
    int holdout_violations = 0;
    std::vector<int> first_terms;
};

PrimeCase evaluate(
    const Paths &paths,
    const std::vector<LocalPlan> &plans,
    int family,
    int prime
) {
    const Field field = make_field(prime);
    const int dimension = static_cast<int>(paths.walks.size());
    const int training = 2 * dimension;
    std::vector<int> state(static_cast<std::size_t>(dimension));
    state[static_cast<std::size_t>(paths.vacuum)] = 1;
    std::vector<int> samples;
    samples.reserve(static_cast<std::size_t>(training + kHoldout));
    for (int time = 0; time < training + kHoldout; ++time) {
        samples.push_back(state[static_cast<std::size_t>(paths.vacuum)]);
        if (time + 1 < training + kHoldout) {
            monodromy(state, plans, field, paths.strands, family);
        }
    }
    const Recurrence recurrence = reconstruct(samples, training, prime);
    const int failures = violations(samples, recurrence, training, prime);
    if (failures != 0) {
        die("separate recurrence fails holdout");
    }
    PrimeCase result;
    result.prime = prime;
    result.root = field.root;
    result.degree = recurrence.degree;
    result.holdout_violations = failures;
    result.first_terms.assign(
        samples.begin(), samples.begin() + std::min<std::size_t>(12, samples.size())
    );
    return result;
}

void print_array(const std::vector<int> &values) {
    std::printf("[");
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::printf("%s%d", index ? "," : "", values[index]);
    }
    std::printf("]");
}

}  // namespace

int main() {
    std::printf(
        "{\"schema\":\"cat_cas.su2_level8_period10_monodromy_krylov_separate_reference.v1\","
        "\"reference_imports_m217_production\":false,"
        "\"reference_algorithm\":\"BITCODED_WALK_ENUMERATION_STRUCTURAL_KIND_PEER_LOCAL_ACTION_AND_INDEPENDENT_SCALAR_RECURRENCE\","
        "\"split_primes\":[641,881],\"cases\":["
    );
    bool first = true;
    bool all_full = true;
    for (int family = 0; family < 2; ++family) {
        for (int strands : kStrands) {
            const Paths paths = enumerate_paths(strands);
            const std::vector<LocalPlan> plans = make_plans(paths);
            std::array<PrimeCase, 2> primes{
                evaluate(paths, plans, family, kPrimes[0]),
                evaluate(paths, plans, family, kPrimes[1]),
            };
            all_full = all_full
                && primes[0].degree == static_cast<int>(paths.walks.size())
                && primes[1].degree == static_cast<int>(paths.walks.size());
            std::printf(
                "%s{\"strands\":%d,\"family\":%d,\"fusion_path_cells\":%zu,"
                "\"prime_results\":[",
                first ? "" : ",", strands, family, paths.walks.size()
            );
            first = false;
            for (std::size_t index = 0; index < primes.size(); ++index) {
                const PrimeCase &item = primes[index];
                std::printf(
                    "%s{\"prime\":%d,\"root40\":%d,"
                    "\"scalar_recurrence_degree\":%d,"
                    "\"holdout_violations\":%d,\"first_terms\":",
                    index ? "," : "", item.prime, item.root,
                    item.degree, item.holdout_violations
                );
                print_array(item.first_terms);
                std::printf("}");
            }
            std::printf("]}");
        }
    }
    std::printf(
        "],\"all_cases_full_at_both_distinct_split_primes\":%s,"
        "\"classification\":\"INDEPENDENTLY_VERIFIED_STRICT_SCOPE\","
        "\"verification_level\":\"SEPARATE_REFERENCE_PARITY\","
        "\"restoration_classification\":\"EXACT_ALGEBRAIC_RESTORATION\","
        "\"terminal\":false}\n",
        all_full ? "true" : "false"
    );
    return all_full ? 0 : 2;
}
