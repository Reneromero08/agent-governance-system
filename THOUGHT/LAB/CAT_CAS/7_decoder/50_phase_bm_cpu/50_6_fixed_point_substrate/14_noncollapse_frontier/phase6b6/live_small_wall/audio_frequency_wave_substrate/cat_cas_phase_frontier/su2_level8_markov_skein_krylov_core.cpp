#define main m217_period10_core_unused_main
#include "su2_level8_period10_monodromy_krylov_core.cpp"
#undef main

/*
 * M218 exploratory core: execute the same public period-10 braid word in the
 * Kauffman/Temperley--Lieb noncrossing-link-pattern basis and project the
 * normalized plat Markov closure, rather than the M217 vacuum path
 * coordinate.  The imported M217 translation unit supplies only public word,
 * finite-field, path-enumeration and Berlekamp--Massey primitives.  Diagram
 * pairings, skein actions, closure weights and accounting are reconstructed
 * here.
 *
 * At at most sixteen strands every Dyck path has height at most eight, so the
 * level-eight Jones--Wenzl relation removes no link-pattern basis element.
 * This diagnostic therefore tests whether the different final functional and
 * procedural skein action change the recurrence/state law before the first
 * root-of-unity truncation at eighteen strands.  It is not a compactness or
 * phase-resource claim.
 */

namespace {

using Pairing = std::vector<std::uint8_t>;

std::uint64_t encode_pairing(const Pairing &pairing) {
    std::uint64_t result = 0;
    for (std::uint8_t peer : pairing) {
        result = (result << 4U) | peer;
    }
    return result;
}

Pairing path_to_pairing(const Path &path) {
    const int strands = static_cast<int>(path.size()) - 1;
    Pairing pairing(static_cast<std::size_t>(strands));
    std::vector<int> stack;
    stack.reserve(static_cast<std::size_t>(strands / 2));
    for (int position = 0; position < strands; ++position) {
        if (path[static_cast<std::size_t>(position + 1)]
            == path[static_cast<std::size_t>(position)] + 1) {
            stack.push_back(position);
            continue;
        }
        if (stack.empty()) {
            fail("Dyck path cannot be converted to a link pattern");
        }
        const int peer = stack.back();
        stack.pop_back();
        pairing[static_cast<std::size_t>(position)] = static_cast<std::uint8_t>(peer);
        pairing[static_cast<std::size_t>(peer)] = static_cast<std::uint8_t>(position);
    }
    if (!stack.empty()) {
        fail("link-pattern stack did not close");
    }
    return pairing;
}

struct DiagramTopology {
    int strands = 0;
    std::vector<Pairing> pairings;
    std::unordered_map<std::uint64_t, int> index;
    int cup_index = -1;
    std::vector<std::vector<int>> e_targets;
    std::vector<std::vector<int>> e_factors_are_delta;
    std::uint64_t pairing_integer_cells = 0;
    std::uint64_t action_records = 0;
    std::uint64_t action_integer_cells = 0;
};

DiagramTopology compile_diagram_topology(const Topology &paths) {
    DiagramTopology result;
    result.strands = paths.strands;
    result.pairings.reserve(paths.paths.size());
    for (const Path &path : paths.paths) {
        result.pairings.push_back(path_to_pairing(path));
    }
    for (std::size_t index = 0; index < result.pairings.size(); ++index) {
        if (!result.index.emplace(
                encode_pairing(result.pairings[index]), static_cast<int>(index)
            ).second) {
            fail("duplicate link pattern");
        }
    }
    Pairing cups(static_cast<std::size_t>(result.strands));
    for (int index = 0; index < result.strands; index += 2) {
        cups[static_cast<std::size_t>(index)] = static_cast<std::uint8_t>(index + 1);
        cups[static_cast<std::size_t>(index + 1)] = static_cast<std::uint8_t>(index);
    }
    const auto cup = result.index.find(encode_pairing(cups));
    if (cup == result.index.end()) {
        fail("public cup link pattern is absent");
    }
    result.cup_index = cup->second;
    result.e_targets.resize(static_cast<std::size_t>(result.strands));
    result.e_factors_are_delta.resize(static_cast<std::size_t>(result.strands));
    for (int generator = 1; generator < result.strands; ++generator) {
        std::vector<int> &targets = result.e_targets[static_cast<std::size_t>(generator)];
        std::vector<int> &factors = result.e_factors_are_delta[
            static_cast<std::size_t>(generator)
        ];
        targets.resize(result.pairings.size());
        factors.resize(result.pairings.size());
        const int left = generator - 1;
        const int right = generator;
        for (std::size_t column = 0; column < result.pairings.size(); ++column) {
            Pairing transformed = result.pairings[column];
            if (transformed[static_cast<std::size_t>(left)] == right) {
                targets[column] = static_cast<int>(column);
                factors[column] = 1;
            } else {
                const int left_peer = transformed[static_cast<std::size_t>(left)];
                const int right_peer = transformed[static_cast<std::size_t>(right)];
                transformed[static_cast<std::size_t>(left)] = static_cast<std::uint8_t>(right);
                transformed[static_cast<std::size_t>(right)] = static_cast<std::uint8_t>(left);
                transformed[static_cast<std::size_t>(left_peer)]
                    = static_cast<std::uint8_t>(right_peer);
                transformed[static_cast<std::size_t>(right_peer)]
                    = static_cast<std::uint8_t>(left_peer);
                const auto found = result.index.find(encode_pairing(transformed));
                if (found == result.index.end()) {
                    fail("skein reconnection left the noncrossing basis");
                }
                targets[column] = found->second;
                factors[column] = 0;
            }
            ++result.action_records;
            result.action_integer_cells += 2;
        }
    }
    result.pairing_integer_cells = static_cast<std::uint64_t>(
        result.pairings.size() * static_cast<std::size_t>(result.strands)
    );
    return result;
}

int union_loop_count(const Pairing &left, const Pairing &right) {
    std::vector<bool> seen(left.size());
    int loops = 0;
    for (std::size_t start = 0; start < left.size(); ++start) {
        if (seen[start]) {
            continue;
        }
        ++loops;
        std::vector<int> pending{static_cast<int>(start)};
        while (!pending.empty()) {
            const int item = pending.back();
            pending.pop_back();
            if (seen[static_cast<std::size_t>(item)]) {
                continue;
            }
            seen[static_cast<std::size_t>(item)] = true;
            pending.push_back(left[static_cast<std::size_t>(item)]);
            pending.push_back(right[static_cast<std::size_t>(item)]);
        }
    }
    return loops;
}

struct DiagramField {
    int prime = 0;
    int root = 0;
    int delta = 0;
    std::vector<int> closure_weights;
};

DiagramField compile_diagram_field(const DiagramTopology &topology, int prime) {
    DiagramField result;
    result.prime = prime;
    result.root = root40(prime);
    result.delta = mod(
        power(result.root, 2, prime) + power(result.root, 38, prime), prime
    );
    result.closure_weights.reserve(topology.pairings.size());
    const Pairing &cups = topology.pairings[static_cast<std::size_t>(topology.cup_index)];
    const int normalization = power(result.delta, topology.strands / 2, prime);
    const int inverse_normalization = inverse(normalization, prime);
    for (const Pairing &pairing : topology.pairings) {
        result.closure_weights.push_back(mod(
            static_cast<std::int64_t>(
                power(result.delta, union_loop_count(cups, pairing), prime)
            ) * inverse_normalization,
            prime
        ));
    }
    return result;
}

void apply_diagram_gate(
    std::vector<int> &state,
    std::vector<int> &scratch,
    const DiagramTopology &topology,
    const DiagramField &field,
    int generator,
    int exponent
) {
    const int alpha = power(field.root, exponent == 1 ? 11 : 29, field.prime);
    const int beta = power(field.root, exponent == 1 ? 29 : 11, field.prime);
    std::fill(scratch.begin(), scratch.end(), 0);
    const std::vector<int> &targets = topology.e_targets[
        static_cast<std::size_t>(generator)
    ];
    const std::vector<int> &delta_flags = topology.e_factors_are_delta[
        static_cast<std::size_t>(generator)
    ];
    for (std::size_t column = 0; column < state.size(); ++column) {
        const int value = state[column];
        scratch[column] = mod(
            scratch[column] + static_cast<std::int64_t>(alpha) * value,
            field.prime
        );
        const int e_factor = delta_flags[column] ? field.delta : 1;
        const int row = targets[column];
        scratch[static_cast<std::size_t>(row)] = mod(
            scratch[static_cast<std::size_t>(row)]
                + static_cast<std::int64_t>(beta) * e_factor * value,
            field.prime
        );
    }
    state.swap(scratch);
}

void apply_diagram_period(
    std::vector<int> &state,
    std::vector<int> &scratch,
    const DiagramTopology &topology,
    const DiagramField &field,
    int family,
    bool perturb_last
) {
    for (int round = 0; round < kPeriodSweeps; ++round) {
        for (int offset = 0; offset < topology.strands - 1; ++offset) {
            Operation item = operation(topology.strands, family, round, offset);
            if (perturb_last && round == kPeriodSweeps - 1
                && offset == topology.strands - 2) {
                item.exponent = -item.exponent;
            }
            apply_diagram_gate(
                state, scratch, topology, field, item.generator, item.exponent
            );
        }
    }
}

int markov_boundary(
    const std::vector<int> &state,
    const DiagramField &field
) {
    int result = 0;
    for (std::size_t index = 0; index < state.size(); ++index) {
        result = mod(
            result + static_cast<std::int64_t>(state[index])
                * field.closure_weights[index],
            field.prime
        );
    }
    return result;
}

struct DiagramPrimeResult {
    int prime = 0;
    int degree = 0;
    int nonzero_coefficients = 0;
    int holdout_violations = 0;
    bool perturbation_changes_prefix = false;
    std::uint64_t sequence_digest = 0;
    std::uint64_t recurrence_digest = 0;
    std::vector<int> first_terms;
};

DiagramPrimeResult diagnose_diagram_prime(
    const DiagramTopology &topology,
    int family,
    int prime
) {
    const DiagramField field = compile_diagram_field(topology, prime);
    const int dimension = static_cast<int>(topology.pairings.size());
    const int training_terms = 2 * dimension;
    const int total_terms = training_terms + kHoldoutTerms;
    std::vector<int> state(static_cast<std::size_t>(dimension));
    std::vector<int> scratch(static_cast<std::size_t>(dimension));
    std::vector<int> perturbed = state;
    std::vector<int> perturb_scratch = scratch;
    state[static_cast<std::size_t>(topology.cup_index)] = 1;
    perturbed = state;
    std::vector<int> sequence;
    std::vector<int> perturb_prefix;
    sequence.reserve(static_cast<std::size_t>(total_terms));
    for (int term = 0; term < total_terms; ++term) {
        sequence.push_back(markov_boundary(state, field));
        if (term < 4) {
            perturb_prefix.push_back(markov_boundary(perturbed, field));
        }
        if (term + 1 < total_terms) {
            apply_diagram_period(state, scratch, topology, field, family, false);
            if (term < 3) {
                apply_diagram_period(
                    perturbed, perturb_scratch, topology, field, family, true
                );
            }
        }
    }
    const BMResult recurrence = berlekamp_massey(sequence, training_terms, prime);
    DiagramPrimeResult result;
    result.prime = prime;
    result.degree = recurrence.degree;
    result.nonzero_coefficients = static_cast<int>(std::count_if(
        recurrence.connection.begin(), recurrence.connection.end(),
        [](int value) { return value != 0; }
    ));
    result.holdout_violations = recurrence_violations(
        sequence, recurrence, training_terms, prime
    );
    result.perturbation_changes_prefix = false;
    for (std::size_t index = 0; index < perturb_prefix.size(); ++index) {
        result.perturbation_changes_prefix = result.perturbation_changes_prefix
            || perturb_prefix[index] != sequence[index];
    }
    result.sequence_digest = fnv1a(sequence);
    result.recurrence_digest = fnv1a(recurrence.connection);
    result.first_terms.assign(sequence.begin(), sequence.begin() + 12);
    if (result.holdout_violations != 0) {
        fail("diagram Markov recurrence fails holdout");
    }
    return result;
}

void print_u64_hex(std::uint64_t value) {
    std::printf("%016llx", static_cast<unsigned long long>(value));
}

}  // namespace

int main() {
    if (!period_law()) {
        fail("public braid word is not period ten");
    }
    std::printf(
        "{\"schema\":\"cat_cas.su2_level8_markov_skein_krylov_core.v1\","
        "\"public_word_period_sweeps\":10,\"jones_wenzl_truncation_active\":false,"
        "\"first_possible_truncation_strands\":18,\"cases\":["
    );
    bool first_case = true;
    for (int family : kFamilies) {
        for (int strands : kStrands) {
            const Topology paths = compile_topology(strands);
            const DiagramTopology diagrams = compile_diagram_topology(paths);
            const DiagramPrimeResult left = diagnose_diagram_prime(diagrams, family, 241);
            const DiagramPrimeResult right = diagnose_diagram_prime(diagrams, family, 401);
            std::printf(
                "%s{\"strands\":%d,\"family\":%d,\"link_pattern_cells\":%zu,"
                "\"retained_pairing_integer_cells\":%llu,"
                "\"retained_skein_action_records\":%llu,"
                "\"retained_skein_action_integer_cells\":%llu,"
                "\"peak_modular_state_and_scratch_cells\":%zu,\"prime_results\":[",
                first_case ? "" : ",", strands, family, diagrams.pairings.size(),
                static_cast<unsigned long long>(diagrams.pairing_integer_cells),
                static_cast<unsigned long long>(diagrams.action_records),
                static_cast<unsigned long long>(diagrams.action_integer_cells),
                2 * diagrams.pairings.size()
            );
            first_case = false;
            const std::array<DiagramPrimeResult, 2> primes{left, right};
            for (std::size_t index = 0; index < primes.size(); ++index) {
                const DiagramPrimeResult &item = primes[index];
                std::printf(
                    "%s{\"prime\":%d,\"markov_scalar_recurrence_degree\":%d,"
                    "\"nonzero_recurrence_coefficients\":%d,"
                    "\"holdout_violations\":%d,"
                    "\"semantic_perturbation_changes_prefix\":%s,"
                    "\"sequence_digest_fnv1a64\":\"",
                    index == 0 ? "" : ",", item.prime, item.degree,
                    item.nonzero_coefficients, item.holdout_violations,
                    item.perturbation_changes_prefix ? "true" : "false"
                );
                print_u64_hex(item.sequence_digest);
                std::printf("\",\"recurrence_digest_fnv1a64\":\"");
                print_u64_hex(item.recurrence_digest);
                std::printf("\",\"first_terms\":");
                print_int_array(item.first_terms);
                std::printf("}");
            }
            std::printf("]}");
        }
    }
    std::printf("]}\n");
    return 0;
}
