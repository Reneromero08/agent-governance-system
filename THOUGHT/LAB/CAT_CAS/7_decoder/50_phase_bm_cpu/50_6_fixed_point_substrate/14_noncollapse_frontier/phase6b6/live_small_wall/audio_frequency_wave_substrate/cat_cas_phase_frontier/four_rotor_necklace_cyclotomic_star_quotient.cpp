#define main necklace_star_quotient_predecessor_main
#include "four_rotor_necklace_orbit_phase.cpp"
#undef main

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

/*
 * Fixed-width conjugation-preserving quotient of the exact
 * Z[zeta_8,1/2] necklace phase carrier.
 *
 * Four embeddings are retained:
 *
 *   F17: zeta -> 2 and zeta -> 2^-1 = 9
 *   F41: zeta -> 3 and zeta -> 3^-1 = 14
 *
 * Conjugation swaps each pair, so the quotient preserves the cyclotomic
 * involution and its exact star norm.  This removes coefficient-height growth
 * only for the declared residue semantics.  It is not a lossless
 * representation of analytic cyclotomic amplitudes, and its strongest
 * compact classical implementation is the identical four-residue recurrence.
 */

namespace cyclotomic_star_quotient {

constexpr std::size_t kLatentCells = 2;
constexpr std::size_t kCarrierCells =
    static_cast<std::size_t>(kExpectedDimension) * kLatentCells;
constexpr std::uint32_t kPrimaryVariant = 2;
constexpr std::size_t kPrimaryDepth = 4096;
constexpr std::uint32_t kReuseVariant = 5;
constexpr std::size_t kReuseDepth = 1537;
constexpr std::array<std::size_t, 5> kDepths{
    1, 64, 256, 1024, 4096
};
constexpr std::array<std::size_t, 7> kBridgeDepths{
    1, 2, 4, 8, 16, 32, 64
};

struct Embedding {
    std::uint16_t prime;
    std::uint16_t root;
    std::size_t conjugate;
};

constexpr std::array<Embedding, 4> kEmbeddings{{
    {17, 2, 1},
    {17, 9, 0},
    {41, 3, 3},
    {41, 14, 2},
}};

std::uint16_t reduce(std::int64_t value, std::uint16_t prime) {
    value %= static_cast<std::int64_t>(prime);
    if (value < 0) {
        value += prime;
    }
    return static_cast<std::uint16_t>(value);
}

std::uint16_t power_mod(
    std::uint16_t base,
    std::uint64_t exponent,
    std::uint16_t prime
) {
    std::uint64_t result = 1;
    std::uint64_t factor = base;
    while (exponent > 0) {
        if ((exponent & 1U) != 0U) {
            result = result * factor % prime;
        }
        factor = factor * factor % prime;
        exponent >>= 1U;
    }
    return static_cast<std::uint16_t>(result);
}

struct Phase {
    std::array<std::uint16_t, kEmbeddings.size()> residue{};
};

bool operator==(const Phase &left, const Phase &right) {
    return left.residue == right.residue;
}

bool operator!=(const Phase &left, const Phase &right) {
    return !(left == right);
}

Phase zero() {
    return {};
}

Phase one() {
    Phase result;
    result.residue.fill(1);
    return result;
}

Phase add(const Phase &left, const Phase &right) {
    Phase result;
    for (std::size_t index = 0; index < kEmbeddings.size(); ++index) {
        result.residue[index] = reduce(
            static_cast<std::int64_t>(left.residue[index])
                + right.residue[index],
            kEmbeddings[index].prime
        );
    }
    return result;
}

Phase subtract(const Phase &left, const Phase &right) {
    Phase result;
    for (std::size_t index = 0; index < kEmbeddings.size(); ++index) {
        result.residue[index] = reduce(
            static_cast<std::int64_t>(left.residue[index])
                - right.residue[index],
            kEmbeddings[index].prime
        );
    }
    return result;
}

Phase multiply_zeta(const Phase &value, int exponent) {
    Phase result;
    int reduced_exponent = exponent % 8;
    if (reduced_exponent < 0) {
        reduced_exponent += 8;
    }
    for (std::size_t index = 0; index < kEmbeddings.size(); ++index) {
        const Embedding embedding = kEmbeddings[index];
        result.residue[index] = reduce(
            static_cast<std::int64_t>(value.residue[index])
                * power_mod(
                    embedding.root,
                    static_cast<std::uint64_t>(reduced_exponent),
                    embedding.prime
                ),
            embedding.prime
        );
    }
    return result;
}

Phase multiply_inverse_sqrt2(const Phase &value) {
    Phase result;
    for (std::size_t index = 0; index < kEmbeddings.size(); ++index) {
        const Embedding embedding = kEmbeddings[index];
        const std::uint16_t inverse_two = power_mod(
            2,
            static_cast<std::uint64_t>(embedding.prime - 2U),
            embedding.prime
        );
        const std::uint16_t coefficient = reduce(
            (
                static_cast<std::int64_t>(embedding.root)
                - power_mod(embedding.root, 3, embedding.prime)
            ) * inverse_two,
            embedding.prime
        );
        result.residue[index] = reduce(
            static_cast<std::int64_t>(value.residue[index])
                * coefficient,
            embedding.prime
        );
    }
    return result;
}

void hadamard(Phase &upper, Phase &lower) {
    const Phase original_upper = upper;
    const Phase original_lower = lower;
    upper = multiply_inverse_sqrt2(
        add(original_upper, original_lower)
    );
    lower = multiply_inverse_sqrt2(
        subtract(original_upper, original_lower)
    );
}

using Carrier = std::vector<Phase>;
using Boundary = std::array<Phase, kMaximumCollision + 1>;
using StarNorm = std::array<std::uint16_t, kEmbeddings.size()>;

std::size_t cell_index(std::size_t necklace, std::size_t latent) {
    return necklace * kLatentCells + latent;
}

Carrier initial_carrier() {
    Carrier result(kCarrierCells, zero());
    result[cell_index(0, 0)] = one();
    return result;
}

struct PublicModule {
    std::size_t offset = 0;
    std::size_t stride = 1;
};

PublicModule compile_public_module(
    std::uint32_t variant,
    std::size_t ordinal,
    bool topology_perturbation
) {
    constexpr std::array<std::size_t, 4> strides{1, 2, 4, 7};
    PublicModule result;
    result.stride = strides[(variant + ordinal) % strides.size()];
    result.offset = (
        11U * static_cast<std::size_t>(variant)
        + 17U * ordinal
        + (topology_perturbation ? 1U : 0U)
    ) % static_cast<std::size_t>(kExpectedDimension);
    if (
        std::gcd(
            result.stride,
            static_cast<std::size_t>(kExpectedDimension)
        ) != 1U
    ) {
        fail("star quotient public matching is not a permutation");
    }
    return result;
}

int feature_phase(
    const Necklace &necklace,
    std::uint32_t variant,
    std::size_t ordinal,
    std::size_t latent,
    int family
) {
    const std::size_t coordinate =
        (ordinal + latent + static_cast<std::size_t>(family))
        % necklace.representative.size();
    return static_cast<int>(
        (
            static_cast<std::size_t>(necklace.collisions + family)
            + static_cast<std::size_t>(
                necklace.representative[coordinate]
            )
            + (2U * latent + 1U)
                * static_cast<std::size_t>(variant)
            + static_cast<std::size_t>(3 * family + 1) * ordinal
        ) % 8U
    );
}

struct Stats {
    std::uint64_t zeta_multiplications = 0;
    std::uint64_t latent_hadamards = 0;
    std::uint64_t necklace_hadamards = 0;
    std::uint64_t modules = 0;
};

std::uint64_t elementary_operations(const Stats &stats) {
    return stats.zeta_multiplications
        + stats.latent_hadamards
        + stats.necklace_hadamards;
}

void apply_diagonal(
    Carrier &carrier,
    const std::vector<Necklace> &necklaces,
    std::uint32_t variant,
    std::size_t ordinal,
    int family,
    bool inverse,
    bool phase_disabled,
    Stats &stats
) {
    for (
        std::size_t necklace = 0;
        necklace < necklaces.size();
        ++necklace
    ) {
        for (std::size_t latent = 0; latent < kLatentCells; ++latent) {
            if (!phase_disabled) {
                const int exponent = feature_phase(
                    necklaces[necklace],
                    variant,
                    ordinal,
                    latent,
                    family
                );
                carrier[cell_index(necklace, latent)] = multiply_zeta(
                    carrier[cell_index(necklace, latent)],
                    inverse ? -exponent : exponent
                );
            }
            ++stats.zeta_multiplications;
        }
    }
}

void apply_latent_hadamards(
    Carrier &carrier,
    const std::vector<Necklace> &necklaces,
    Stats &stats
) {
    for (
        std::size_t necklace = 0;
        necklace < necklaces.size();
        ++necklace
    ) {
        hadamard(
            carrier[cell_index(necklace, 0)],
            carrier[cell_index(necklace, 1)]
        );
        ++stats.latent_hadamards;
    }
}

void apply_necklace_matching(
    Carrier &carrier,
    const PublicModule &module,
    Stats &stats
) {
    const std::size_t dimension =
        static_cast<std::size_t>(kExpectedDimension);
    for (
        std::size_t cursor = 0;
        cursor + 1 < dimension;
        cursor += 2
    ) {
        const std::size_t upper =
            (module.offset + cursor * module.stride) % dimension;
        const std::size_t lower =
            (module.offset + (cursor + 1U) * module.stride) % dimension;
        for (std::size_t latent = 0; latent < kLatentCells; ++latent) {
            hadamard(
                carrier[cell_index(upper, latent)],
                carrier[cell_index(lower, latent)]
            );
            ++stats.necklace_hadamards;
        }
    }
}

void apply_module(
    Carrier &carrier,
    const std::vector<Necklace> &necklaces,
    std::uint32_t variant,
    std::size_t ordinal,
    bool inverse,
    bool phase_disabled,
    bool topology_perturbation,
    Stats &stats
) {
    const PublicModule module = compile_public_module(
        variant,
        ordinal,
        topology_perturbation
    );
    if (!inverse) {
        apply_diagonal(
            carrier,
            necklaces,
            variant,
            ordinal,
            1,
            false,
            phase_disabled,
            stats
        );
        apply_latent_hadamards(carrier, necklaces, stats);
        apply_necklace_matching(carrier, module, stats);
        apply_diagonal(
            carrier,
            necklaces,
            variant,
            ordinal,
            2,
            false,
            phase_disabled,
            stats
        );
    } else {
        apply_diagonal(
            carrier,
            necklaces,
            variant,
            ordinal,
            2,
            true,
            phase_disabled,
            stats
        );
        apply_necklace_matching(carrier, module, stats);
        apply_latent_hadamards(carrier, necklaces, stats);
        apply_diagonal(
            carrier,
            necklaces,
            variant,
            ordinal,
            1,
            true,
            phase_disabled,
            stats
        );
    }
    ++stats.modules;
}

void forward(
    Carrier &carrier,
    const std::vector<Necklace> &necklaces,
    std::uint32_t variant,
    std::size_t depth,
    bool phase_disabled,
    bool topology_perturbation,
    Stats &stats
) {
    for (std::size_t ordinal = 0; ordinal < depth; ++ordinal) {
        apply_module(
            carrier,
            necklaces,
            variant,
            ordinal,
            false,
            phase_disabled,
            topology_perturbation,
            stats
        );
    }
}

enum class InverseMode {
    CORRECT,
    MISSING,
    WRONG_VARIANT,
    REORDERED,
};

void inverse(
    Carrier &carrier,
    const std::vector<Necklace> &necklaces,
    std::uint32_t variant,
    std::size_t depth,
    InverseMode mode,
    bool phase_disabled,
    bool topology_perturbation,
    Stats &stats
) {
    if (mode == InverseMode::MISSING) {
        return;
    }
    for (std::size_t cursor = 0; cursor < depth; ++cursor) {
        const std::size_t ordinal = mode == InverseMode::REORDERED
            ? cursor
            : depth - cursor - 1U;
        apply_module(
            carrier,
            necklaces,
            mode == InverseMode::WRONG_VARIANT
                ? variant + 1U
                : variant,
            ordinal,
            true,
            phase_disabled,
            topology_perturbation,
            stats
        );
    }
}

Boundary project_boundary(
    const Carrier &carrier,
    const std::vector<Necklace> &necklaces
) {
    Boundary result{};
    for (
        std::size_t necklace = 0;
        necklace < necklaces.size();
        ++necklace
    ) {
        const Necklace &descriptor = necklaces[necklace];
        const std::size_t bin =
            static_cast<std::size_t>(descriptor.collisions);
        result[bin] = add(
            result[bin],
            carrier[cell_index(necklace, 0)]
        );
        result[bin] = add(
            result[bin],
            multiply_zeta(
                carrier[cell_index(necklace, 1)],
                descriptor.representative[0]
            )
        );
    }
    return result;
}

StarNorm star_norm(const Carrier &carrier) {
    StarNorm result{};
    for (std::size_t embedding = 0; embedding < kEmbeddings.size(); ++embedding) {
        const Embedding descriptor = kEmbeddings[embedding];
        std::uint64_t sum = 0;
        for (const Phase &value : carrier) {
            sum += static_cast<std::uint64_t>(
                value.residue[embedding]
            ) * value.residue[descriptor.conjugate];
            sum %= descriptor.prime;
        }
        result[embedding] = static_cast<std::uint16_t>(sum);
    }
    return result;
}

StarNorm incorrect_identity_pair_norm(const Carrier &carrier) {
    StarNorm result{};
    for (std::size_t embedding = 0; embedding < kEmbeddings.size(); ++embedding) {
        const Embedding descriptor = kEmbeddings[embedding];
        std::uint64_t sum = 0;
        for (const Phase &value : carrier) {
            sum += static_cast<std::uint64_t>(
                value.residue[embedding]
            ) * value.residue[embedding];
            sum %= descriptor.prime;
        }
        result[embedding] = static_cast<std::uint16_t>(sum);
    }
    return result;
}

bool norm_is_one(const StarNorm &norm) {
    return std::all_of(
        norm.begin(),
        norm.end(),
        [](std::uint16_t value) {
            return value == 1U;
        }
    );
}

struct Run {
    Boundary boundary{};
    StarNorm forward_star_norm{};
    Stats stats{};
    std::uint64_t forward_elementary_operations = 0;
    bool restored = false;
    bool backing_preserved = false;
};

Run transaction(
    Carrier &carrier,
    const Carrier &baseline,
    const std::vector<Necklace> &necklaces,
    std::uint32_t variant,
    std::size_t depth,
    InverseMode mode = InverseMode::CORRECT,
    bool phase_disabled = false,
    bool topology_perturbation = false
) {
    if (carrier.size() != kCarrierCells) {
        fail("invalid star quotient carrier");
    }
    Run result;
    const Phase *const backing = carrier.data();
    forward(
        carrier,
        necklaces,
        variant,
        depth,
        phase_disabled,
        topology_perturbation,
        result.stats
    );
    result.forward_elementary_operations =
        elementary_operations(result.stats);
    result.forward_star_norm = star_norm(carrier);
    result.boundary = project_boundary(carrier, necklaces);
    inverse(
        carrier,
        necklaces,
        variant,
        depth,
        mode,
        phase_disabled,
        topology_perturbation,
        result.stats
    );
    result.restored = carrier == baseline;
    result.backing_preserved = carrier.data() == backing;
    return result;
}

void print_phase(const Phase &value) {
    std::printf("[");
    for (std::size_t index = 0; index < value.residue.size(); ++index) {
        std::printf(
            "%s%u",
            index == 0 ? "" : ",",
            static_cast<unsigned>(value.residue[index])
        );
    }
    std::printf("]");
}

void print_boundary(const Boundary &boundary) {
    std::printf("[");
    for (std::size_t index = 0; index < boundary.size(); ++index) {
        if (index != 0) {
            std::printf(",");
        }
        print_phase(boundary[index]);
    }
    std::printf("]");
}

void print_norm(const StarNorm &norm) {
    std::printf("[");
    for (std::size_t index = 0; index < norm.size(); ++index) {
        std::printf(
            "%s%u",
            index == 0 ? "" : ",",
            static_cast<unsigned>(norm[index])
        );
    }
    std::printf("]");
}

void print_run(std::size_t depth, const Run &run, bool final) {
    std::printf(
        "{\"depth\":%zu,"
        "\"forward_elementary_operations\":%llu,"
        "\"forward_star_norm\":",
        depth,
        static_cast<unsigned long long>(
            run.forward_elementary_operations
        )
    );
    print_norm(run.forward_star_norm);
    std::printf(",\"boundary\":");
    print_boundary(run.boundary);
    std::printf(
        ",\"exact_algebraic_restoration\":%s,"
        "\"outer_carrier_backing_preserved\":%s}%s",
        run.restored ? "true" : "false",
        run.backing_preserved ? "true" : "false",
        final ? "" : ","
    );
}

}  // namespace cyclotomic_star_quotient

int main() {
    namespace csq = cyclotomic_star_quotient;

    const std::vector<Necklace> necklaces = compile_necklaces();
    if (necklaces.size() != static_cast<std::size_t>(kExpectedDimension)) {
        fail("star quotient necklace geometry failed");
    }
    for (const csq::Embedding embedding : csq::kEmbeddings) {
        if (
            csq::power_mod(embedding.root, 8, embedding.prime) != 1U
            || csq::power_mod(embedding.root, 4, embedding.prime)
                != embedding.prime - 1U
            || csq::power_mod(
                embedding.root,
                static_cast<std::uint64_t>(embedding.prime - 2U),
                embedding.prime
            ) != csq::kEmbeddings[embedding.conjugate].root
        ) {
            fail("star quotient embedding law failed");
        }
    }
    constexpr std::int64_t demonstrated_kernel_element = 17 * 41;
    for (const csq::Embedding embedding : csq::kEmbeddings) {
        if (
            csq::reduce(
                demonstrated_kernel_element,
                embedding.prime
            ) != 0U
        ) {
            fail("star quotient kernel generator failed");
        }
    }

    const csq::Carrier baseline = csq::initial_carrier();
    std::array<csq::Run, csq::kDepths.size()> depth_runs{};
    for (
        std::size_t index = 0;
        index + 1U < csq::kDepths.size();
        ++index
    ) {
        csq::Carrier carrier = baseline;
        depth_runs[index] = csq::transaction(
            carrier,
            baseline,
            necklaces,
            csq::kPrimaryVariant,
            csq::kDepths[index]
        );
        if (
            !depth_runs[index].restored
            || !depth_runs[index].backing_preserved
            || !csq::norm_is_one(depth_runs[index].forward_star_norm)
        ) {
            fail("star quotient depth transaction failed");
        }
    }

    csq::Carrier accepted_carrier = baseline;
    depth_runs.back() = csq::transaction(
        accepted_carrier,
        baseline,
        necklaces,
        csq::kPrimaryVariant,
        csq::kPrimaryDepth
    );
    if (
        !depth_runs.back().restored
        || !depth_runs.back().backing_preserved
        || !csq::norm_is_one(depth_runs.back().forward_star_norm)
    ) {
        fail("star quotient primary transaction failed");
    }
    const csq::Run primary = depth_runs.back();
    const csq::Run reuse = csq::transaction(
        accepted_carrier,
        baseline,
        necklaces,
        csq::kReuseVariant,
        csq::kReuseDepth
    );
    csq::Carrier fresh_reuse_carrier = baseline;
    const csq::Run fresh_reuse = csq::transaction(
        fresh_reuse_carrier,
        baseline,
        necklaces,
        csq::kReuseVariant,
        csq::kReuseDepth
    );
    if (
        !reuse.restored
        || !reuse.backing_preserved
        || !csq::norm_is_one(reuse.forward_star_norm)
        || reuse.boundary != fresh_reuse.boundary
        || reuse.forward_star_norm != fresh_reuse.forward_star_norm
    ) {
        fail("star quotient restored-carrier reuse failed");
    }

    std::array<csq::Run, csq::kBridgeDepths.size()> bridge_runs{};
    for (
        std::size_t index = 0;
        index < csq::kBridgeDepths.size();
        ++index
    ) {
        csq::Carrier carrier = baseline;
        bridge_runs[index] = csq::transaction(
            carrier,
            baseline,
            necklaces,
            csq::kPrimaryVariant,
            csq::kBridgeDepths[index]
        );
        if (
            !bridge_runs[index].restored
            || !csq::norm_is_one(bridge_runs[index].forward_star_norm)
        ) {
            fail("star quotient exact bridge failed");
        }
    }

    csq::Carrier missing_carrier = baseline;
    const csq::Run missing = csq::transaction(
        missing_carrier,
        baseline,
        necklaces,
        csq::kPrimaryVariant,
        8,
        csq::InverseMode::MISSING
    );
    csq::Carrier wrong_carrier = baseline;
    const csq::Run wrong = csq::transaction(
        wrong_carrier,
        baseline,
        necklaces,
        csq::kPrimaryVariant,
        8,
        csq::InverseMode::WRONG_VARIANT
    );
    csq::Carrier reordered_carrier = baseline;
    const csq::Run reordered = csq::transaction(
        reordered_carrier,
        baseline,
        necklaces,
        csq::kPrimaryVariant,
        8,
        csq::InverseMode::REORDERED
    );
    if (missing.restored || wrong.restored || reordered.restored) {
        fail("star quotient inverse controls failed");
    }

    csq::Carrier phase_disabled_carrier = baseline;
    const csq::Run phase_disabled = csq::transaction(
        phase_disabled_carrier,
        baseline,
        necklaces,
        csq::kPrimaryVariant,
        64,
        csq::InverseMode::CORRECT,
        true,
        false
    );
    csq::Carrier perturbed_carrier = baseline;
    const csq::Run perturbed = csq::transaction(
        perturbed_carrier,
        baseline,
        necklaces,
        csq::kPrimaryVariant,
        64,
        csq::InverseMode::CORRECT,
        false,
        true
    );
    csq::Carrier null_carrier(csq::kCarrierCells, csq::zero());
    const csq::Run null_run = csq::transaction(
        null_carrier,
        null_carrier,
        necklaces,
        csq::kPrimaryVariant,
        64
    );
    csq::Carrier norm_control_carrier = baseline;
    csq::Stats norm_control_stats;
    csq::forward(
        norm_control_carrier,
        necklaces,
        csq::kPrimaryVariant,
        64,
        false,
        false,
        norm_control_stats
    );
    const bool wrong_pairing_norm_is_one = csq::norm_is_one(
        csq::incorrect_identity_pair_norm(norm_control_carrier)
    );
    if (
        phase_disabled.boundary == bridge_runs.back().boundary
        || perturbed.boundary == bridge_runs.back().boundary
        || null_run.boundary == bridge_runs.back().boundary
        || wrong_pairing_norm_is_one
    ) {
        fail("star quotient semantic controls failed");
    }

    std::printf(
        "{\"result\":\"PASS\","
        "\"claim_candidate\":"
        "\"BOUNDED_DUAL_PRIME_CONJUGATE_PAIR_CYCLOTOMIC_STAR_QUOTIENT_FIXED_570_CELL_PHASE_CLOSURE_ACROSS_DEPTH4096_WITH_EXACT_RESTORATION_AND_REUSE\","
        "\"quotient\":\"F17_X_F17_X_F41_X_F41_CONJUGATE_PAIR\","
        "\"analytic_cyclotomic_amplitudes_losslessly_preserved\":false,"
        "\"demonstrated_nonzero_kernel_element_integer\":697,"
        "\"demonstrated_nonzero_kernel_element_maps_to_zero\":true,"
        "\"resident_necklace_descriptors\":%zu,"
        "\"latent_cells_per_necklace\":%zu,"
        "\"logical_phase_cells\":%zu,"
        "\"residues_per_phase_cell\":%zu,"
        "\"tested_depths\":[1,64,256,1024,4096],"
        "\"depth_runs\":[",
        necklaces.size(),
        csq::kLatentCells,
        csq::kCarrierCells,
        csq::kEmbeddings.size()
    );
    for (std::size_t index = 0; index < depth_runs.size(); ++index) {
        csq::print_run(
            csq::kDepths[index],
            depth_runs[index],
            index + 1U == depth_runs.size()
        );
    }
    std::printf("],\"exact_bridge_runs\":[");
    for (std::size_t index = 0; index < bridge_runs.size(); ++index) {
        csq::print_run(
            csq::kBridgeDepths[index],
            bridge_runs[index],
            index + 1U == bridge_runs.size()
        );
    }
    std::printf("],\"primary\":");
    csq::print_run(csq::kPrimaryDepth, primary, true);
    std::printf(",\"reuse\":");
    csq::print_run(csq::kReuseDepth, reuse, true);
    std::printf(
        ",\"fresh_reuse_boundary_equal\":true,"
        "\"same_outer_backing_primary_and_reuse\":true,"
        "\"restoration_generation_sequence\":[1,2],"
        "\"baseline_reload_bytes\":0,"
        "\"restoration_class\":\"EXACT_ALGEBRAIC_RESTORATION\","
        "\"controls\":{"
        "\"missing_inverse_restored\":false,"
        "\"wrong_inverse_variant_restored\":false,"
        "\"reordered_inverse_restored\":false,"
        "\"phase_disabled_boundary_differs\":true,"
        "\"topology_perturbation_boundary_differs\":true,"
        "\"null_carrier_boundary_differs\":true,"
        "\"wrong_conjugation_pairing_norm_is_one\":false},"
        "\"resource_law\":{"
        "\"logical_phase_cells_constant\":true,"
        "\"residues_per_phase_cell_constant\":true,"
        "\"phase_cell_bytes\":%zu,"
        "\"carrier_vector_capacity_cells\":%zu,"
        "\"carrier_inline_bytes\":%zu,"
        "\"restoration_baseline_inline_bytes\":%zu,"
        "\"permanent_restoration_baseline_phase_cells\":%zu,"
        "\"final_boundary_phase_cells\":%zu,"
        "\"topology_necklace_count\":%zu,"
        "\"topology_necklace_capacity\":%zu,"
        "\"topology_necklace_element_bytes\":%zu,"
        "\"topology_necklace_capacity_bytes\":%zu,"
        "\"transaction_run_record_bytes\":%zu,"
        "\"accepted_path_declared_payload_subtotal_bytes\":%zu,"
        "\"primary_forward_elementary_operations\":%llu,"
        "\"primary_forward_inverse_elementary_operations\":%llu,"
        "\"reuse_forward_elementary_operations\":%llu,"
        "\"reuse_forward_inverse_elementary_operations\":%llu,"
        "\"retained_module_tape_bytes\":0,"
        "\"retained_inverse_history_bytes\":0,"
        "\"public_topology_inspects_final_answer\":false,"
        "\"dense_570_by_570_operator_cells\":0,"
        "\"relation_table_cells\":0,"
        "\"assignment_cells\":0,"
        "\"allocator_control_block_and_heap_metadata_bounded\":false,"
        "\"verification_harness_and_binary_bytes_included\":false,"
        "\"whole_process_rss_claimed\":false},"
        "\"strongest_compact_classical\":{"
        "\"identical_four_residue_recurrence\":true,"
        "\"boundary_error\":0},"
        "\"intermediate_projected\":false,"
        "\"machine_boundary_enforced\":false,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_waveform_execution\":false,"
        "\"physical_bit_replacement\":false,"
        "\"unbounded_computation_established\":false,"
        "\"terminal\":false}\n",
        sizeof(csq::Phase),
        accepted_carrier.capacity(),
        accepted_carrier.capacity() * sizeof(csq::Phase),
        baseline.capacity() * sizeof(csq::Phase),
        baseline.size(),
        static_cast<std::size_t>(kMaximumCollision + 1),
        necklaces.size(),
        necklaces.capacity(),
        sizeof(Necklace),
        necklaces.capacity() * sizeof(Necklace),
        sizeof(csq::Run),
        accepted_carrier.capacity() * sizeof(csq::Phase)
            + baseline.capacity() * sizeof(csq::Phase)
            + necklaces.capacity() * sizeof(Necklace)
            + sizeof(csq::Run),
        static_cast<unsigned long long>(
            primary.forward_elementary_operations
        ),
        static_cast<unsigned long long>(
            csq::elementary_operations(primary.stats)
        ),
        static_cast<unsigned long long>(
            reuse.forward_elementary_operations
        ),
        static_cast<unsigned long long>(
            csq::elementary_operations(reuse.stats)
        )
    );
    return 0;
}
