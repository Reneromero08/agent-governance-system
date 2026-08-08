#define main m217_separate_reference_unused_main
#include "su2_level8_period10_monodromy_krylov_separate_reference.cpp"
#undef main

/*
 * Separate M218 reference.
 *
 * This translation unit imports no M218 production source.  It starts from
 * the bit-filtered walk enumerator and distinct split-prime arithmetic of the
 * historical M217 separate reference, independently constructs noncrossing
 * pairings in fixed arrays, compiles the deterministic Temperley--Lieb skein
 * reconnection, streams the normalized Markov closure, and reruns the scalar
 * recurrence test at primes 641 and 881.
 */

namespace {

struct RefPairing {
    std::array<std::uint8_t, 16> peer{};
};

std::uint64_t pairing_code(const RefPairing &pairing, int strands) {
    std::uint64_t result = 0;
    for (int index = 0; index < strands; ++index) {
        result = result * 17U + pairing.peer[static_cast<std::size_t>(index)];
    }
    return result;
}

RefPairing pairing_from_walk(const Walk &walk, int strands) {
    RefPairing result;
    std::array<int, 8> stack{};
    int depth = 0;
    for (int position = 0; position < strands; ++position) {
        const int left = walk.labels[static_cast<std::size_t>(position)];
        const int right = walk.labels[static_cast<std::size_t>(position + 1)];
        if (right == left + 1) {
            stack[static_cast<std::size_t>(depth++)] = position;
            continue;
        }
        if (right != left - 1 || depth == 0) {
            die("reference walk does not define a pairing");
        }
        const int peer = stack[static_cast<std::size_t>(--depth)];
        result.peer[static_cast<std::size_t>(position)]
            = static_cast<std::uint8_t>(peer);
        result.peer[static_cast<std::size_t>(peer)]
            = static_cast<std::uint8_t>(position);
    }
    if (depth != 0) {
        die("reference pairing stack did not close");
    }
    return result;
}

struct RefDiagramTopology {
    int strands = 0;
    std::vector<RefPairing> pairings;
    std::unordered_map<std::uint64_t, int> ranks;
    int cups = -1;
    std::vector<std::vector<int>> targets;
    std::vector<std::vector<std::uint8_t>> delta_flags;
};

RefDiagramTopology make_diagrams(const Paths &paths) {
    RefDiagramTopology result;
    result.strands = paths.strands;
    for (const Walk &walk : paths.walks) {
        result.pairings.push_back(pairing_from_walk(walk, paths.strands));
    }
    for (std::size_t index = 0; index < result.pairings.size(); ++index) {
        result.ranks.emplace(
            pairing_code(result.pairings[index], paths.strands),
            static_cast<int>(index)
        );
    }
    RefPairing cups;
    for (int index = 0; index < paths.strands; index += 2) {
        cups.peer[static_cast<std::size_t>(index)]
            = static_cast<std::uint8_t>(index + 1);
        cups.peer[static_cast<std::size_t>(index + 1)]
            = static_cast<std::uint8_t>(index);
    }
    const auto cup = result.ranks.find(pairing_code(cups, paths.strands));
    if (cup == result.ranks.end() || result.ranks.size() != result.pairings.size()) {
        die("reference diagram topology failed");
    }
    result.cups = cup->second;
    result.targets.resize(static_cast<std::size_t>(paths.strands));
    result.delta_flags.resize(static_cast<std::size_t>(paths.strands));
    for (int generator = 1; generator < paths.strands; ++generator) {
        std::vector<int> &targets = result.targets[static_cast<std::size_t>(generator)];
        std::vector<std::uint8_t> &flags
            = result.delta_flags[static_cast<std::size_t>(generator)];
        targets.resize(result.pairings.size());
        flags.resize(result.pairings.size());
        const int left = generator - 1;
        const int right = generator;
        for (std::size_t column = 0; column < result.pairings.size(); ++column) {
            RefPairing transformed = result.pairings[column];
            if (transformed.peer[static_cast<std::size_t>(left)] == right) {
                targets[column] = static_cast<int>(column);
                flags[column] = 1;
                continue;
            }
            const int left_peer = transformed.peer[static_cast<std::size_t>(left)];
            const int right_peer = transformed.peer[static_cast<std::size_t>(right)];
            transformed.peer[static_cast<std::size_t>(left)]
                = static_cast<std::uint8_t>(right);
            transformed.peer[static_cast<std::size_t>(right)]
                = static_cast<std::uint8_t>(left);
            transformed.peer[static_cast<std::size_t>(left_peer)]
                = static_cast<std::uint8_t>(right_peer);
            transformed.peer[static_cast<std::size_t>(right_peer)]
                = static_cast<std::uint8_t>(left_peer);
            const auto found = result.ranks.find(
                pairing_code(transformed, paths.strands)
            );
            if (found == result.ranks.end()) {
                die("reference skein target missing");
            }
            targets[column] = found->second;
        }
    }
    return result;
}

int ref_loops(const RefPairing &left, const RefPairing &right, int strands) {
    std::array<bool, 16> seen{};
    int loops = 0;
    for (int start = 0; start < strands; ++start) {
        if (seen[static_cast<std::size_t>(start)]) {
            continue;
        }
        ++loops;
        std::array<int, 32> pending{};
        int size = 1;
        pending[0] = start;
        while (size != 0) {
            const int item = pending[static_cast<std::size_t>(--size)];
            if (seen[static_cast<std::size_t>(item)]) {
                continue;
            }
            seen[static_cast<std::size_t>(item)] = true;
            pending[static_cast<std::size_t>(size++)]
                = left.peer[static_cast<std::size_t>(item)];
            pending[static_cast<std::size_t>(size++)]
                = right.peer[static_cast<std::size_t>(item)];
        }
    }
    return loops;
}

struct RefDiagramField {
    Field field;
    std::vector<int> closure;
};

RefDiagramField make_diagram_field(const RefDiagramTopology &topology, int prime) {
    RefDiagramField result;
    result.field = make_field(prime);
    const int delta = result.field.dimensions[1];
    const int normalization = reciprocal(
        exponentiate(delta, topology.strands / 2, prime), prime
    );
    const RefPairing &cups = topology.pairings[static_cast<std::size_t>(topology.cups)];
    for (const RefPairing &pairing : topology.pairings) {
        result.closure.push_back(residue(
            static_cast<std::int64_t>(
                exponentiate(delta, ref_loops(cups, pairing, topology.strands), prime)
            ) * normalization,
            prime
        ));
    }
    return result;
}

void ref_skein_gate(
    std::vector<int> &state,
    std::vector<int> &scratch,
    const RefDiagramTopology &topology,
    const RefDiagramField &field,
    int generator,
    int exponent
) {
    const int alpha = exponentiate(
        field.field.root, exponent == 1 ? 11 : 29, field.field.prime
    );
    const int beta = exponentiate(
        field.field.root, exponent == 1 ? 29 : 11, field.field.prime
    );
    const int delta = field.field.dimensions[1];
    std::fill(scratch.begin(), scratch.end(), 0);
    const std::vector<int> &targets
        = topology.targets[static_cast<std::size_t>(generator)];
    const std::vector<std::uint8_t> &flags
        = topology.delta_flags[static_cast<std::size_t>(generator)];
    for (std::size_t column = 0; column < state.size(); ++column) {
        scratch[column] = residue(
            scratch[column] + static_cast<std::int64_t>(alpha) * state[column],
            field.field.prime
        );
        const int row = targets[column];
        scratch[static_cast<std::size_t>(row)] = residue(
            scratch[static_cast<std::size_t>(row)]
                + static_cast<std::int64_t>(beta)
                    * (flags[column] ? delta : 1) * state[column],
            field.field.prime
        );
    }
    state.swap(scratch);
}

void ref_monodromy(
    std::vector<int> &state,
    std::vector<int> &scratch,
    const RefDiagramTopology &topology,
    const RefDiagramField &field,
    int family
) {
    for (int round = 0; round < 10; ++round) {
        for (int offset = 0; offset < topology.strands - 1; ++offset) {
            const auto [generator, exponent] = word_item(
                topology.strands, family, round, offset
            );
            ref_skein_gate(
                state, scratch, topology, field, generator, exponent
            );
        }
    }
}

int ref_boundary(const std::vector<int> &state, const RefDiagramField &field) {
    int result = 0;
    for (std::size_t index = 0; index < state.size(); ++index) {
        result = residue(
            result + static_cast<std::int64_t>(state[index]) * field.closure[index],
            field.field.prime
        );
    }
    return result;
}

PrimeCase evaluate_diagram(
    const RefDiagramTopology &topology,
    int family,
    int prime
) {
    const RefDiagramField field = make_diagram_field(topology, prime);
    const int dimension = static_cast<int>(topology.pairings.size());
    const int training = 2 * dimension;
    std::vector<int> state(static_cast<std::size_t>(dimension));
    std::vector<int> scratch(static_cast<std::size_t>(dimension));
    state[static_cast<std::size_t>(topology.cups)] = 1;
    std::vector<int> samples;
    samples.reserve(static_cast<std::size_t>(training + kHoldout));
    for (int time = 0; time < training + kHoldout; ++time) {
        samples.push_back(ref_boundary(state, field));
        if (time + 1 < training + kHoldout) {
            ref_monodromy(state, scratch, topology, field, family);
        }
    }
    const Recurrence recurrence = reconstruct(samples, training, prime);
    PrimeCase result;
    result.prime = prime;
    result.root = field.field.root;
    result.degree = recurrence.degree;
    result.holdout_violations = violations(samples, recurrence, training, prime);
    result.first_terms.assign(samples.begin(), samples.begin() + 12);
    if (result.holdout_violations != 0) {
        die("reference Markov recurrence fails holdout");
    }
    return result;
}

}  // namespace

int main() {
    std::printf(
        "{\"schema\":\"cat_cas.su2_level8_markov_skein_krylov_separate_reference.v1\","
        "\"reference_imports_m218_production\":false,"
        "\"reference_algorithm\":\"BIT_FILTERED_WALKS_FIXED_ARRAY_PAIRINGS_DISTINCT_SKEIN_TABLE_AND_MARKOV_STREAM\","
        "\"split_primes\":[641,881],\"cases\":["
    );
    bool first = true;
    bool all_full = true;
    for (int family = 0; family < 2; ++family) {
        for (int strands : kStrands) {
            const Paths paths = enumerate_paths(strands);
            const RefDiagramTopology topology = make_diagrams(paths);
            const std::array<PrimeCase, 2> primes{
                evaluate_diagram(topology, family, kPrimes[0]),
                evaluate_diagram(topology, family, kPrimes[1]),
            };
            all_full = all_full
                && primes[0].degree == static_cast<int>(topology.pairings.size())
                && primes[1].degree == static_cast<int>(topology.pairings.size());
            std::printf(
                "%s{\"strands\":%d,\"family\":%d,\"link_pattern_cells\":%zu,"
                "\"prime_results\":[",
                first ? "" : ",", strands, family, topology.pairings.size()
            );
            first = false;
            for (std::size_t index = 0; index < primes.size(); ++index) {
                const PrimeCase &item = primes[index];
                std::printf(
                    "%s{\"prime\":%d,\"root40\":%d,"
                    "\"markov_scalar_recurrence_degree\":%d,"
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
