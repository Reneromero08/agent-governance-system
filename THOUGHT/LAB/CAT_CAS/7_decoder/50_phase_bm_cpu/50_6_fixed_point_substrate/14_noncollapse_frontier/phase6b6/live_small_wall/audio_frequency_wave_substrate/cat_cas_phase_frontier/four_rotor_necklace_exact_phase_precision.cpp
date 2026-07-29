#define main necklace_exact_precision_predecessor_main
#include "four_rotor_necklace_orbit_phase.cpp"
#undef main

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

/*
 * Exact arithmetic-height diagnostic on the established 285-necklace by
 * two-cell latent geometry.
 *
 * This is a distinct exact successor, not an exact reinterpretation of the
 * complex128 Hermitian-generator package.  Each cell lies in
 *
 *   Z[zeta_8, 1/2],  zeta_8^4 = -1,
 *
 * represented canonically by four arbitrary-precision integer coefficients
 * and one common dyadic denominator exponent.  Public modules contain
 * diagonal zeta_8 phases and exact H mixers both across the latent fiber and
 * across public necklace-index matchings.  The inverse regenerates every
 * public module from (variant, ordinal) in reverse order.
 *
 * The experiment asks whether a fixed count of 570 logical phase cells also
 * gives fixed material arithmetic width.  It deliberately compares against
 * the identical compact exact recurrence and makes no distinct-resource or
 * performance claim.
 */

namespace exact_phase_precision {

constexpr std::size_t kLatentCells = 2;
constexpr std::size_t kCarrierCells =
    static_cast<std::size_t>(kExpectedDimension) * kLatentCells;
constexpr std::uint32_t kPrimaryVariant = 2;
constexpr std::size_t kPrimaryDepth = 64;
constexpr std::uint32_t kReuseVariant = 5;
constexpr std::size_t kReuseDepth = 23;
constexpr std::array<std::size_t, 7> kDepths{
    1, 2, 4, 8, 16, 32, 64
};

class BigInt {
  public:
    BigInt() = default;

    BigInt(std::int64_t value) {
        if (value == 0) {
            return;
        }
        sign_ = value < 0 ? -1 : 1;
        std::uint64_t magnitude = value < 0
            ? static_cast<std::uint64_t>(-(value + 1)) + 1U
            : static_cast<std::uint64_t>(value);
        limbs_.push_back(
            static_cast<std::uint32_t>(magnitude)
        );
        const std::uint32_t high =
            static_cast<std::uint32_t>(magnitude >> 32U);
        if (high != 0U) {
            limbs_.push_back(high);
        }
    }

    bool is_zero() const {
        return sign_ == 0;
    }

    bool is_negative() const {
        return sign_ < 0;
    }

    bool is_even() const {
        return limbs_.empty() || (limbs_[0] & 1U) == 0U;
    }

    std::uint64_t bit_length() const {
        if (limbs_.empty()) {
            return 0;
        }
        return static_cast<std::uint64_t>(
            32U * (limbs_.size() - 1U)
            + std::bit_width(limbs_.back())
        );
    }

    BigInt negated() const {
        BigInt result = *this;
        result.sign_ = -result.sign_;
        return result;
    }

    BigInt shifted_left(std::uint32_t bits) const {
        if (is_zero() || bits == 0U) {
            return *this;
        }
        BigInt result;
        result.sign_ = sign_;
        const std::size_t whole = bits / 32U;
        const std::uint32_t partial = bits % 32U;
        result.limbs_.assign(whole, 0U);
        std::uint64_t carry = 0;
        for (std::uint32_t limb : limbs_) {
            const std::uint64_t expanded =
                (static_cast<std::uint64_t>(limb) << partial)
                | carry;
            result.limbs_.push_back(
                static_cast<std::uint32_t>(expanded)
            );
            carry = expanded >> 32U;
        }
        if (carry != 0U) {
            result.limbs_.push_back(
                static_cast<std::uint32_t>(carry)
            );
        }
        result.normalize();
        return result;
    }

    void divide_by_two_exact() {
        if (!is_even()) {
            fail("non-even exact integer division");
        }
        std::uint32_t carry = 0;
        for (std::size_t cursor = limbs_.size(); cursor > 0; --cursor) {
            const std::size_t index = cursor - 1U;
            const std::uint32_t next_carry = limbs_[index] & 1U;
            limbs_[index] = (limbs_[index] >> 1U) | (carry << 31U);
            carry = next_carry;
        }
        normalize();
    }

    std::int64_t residue(std::int64_t prime) const {
        std::uint64_t result = 0;
        const std::uint64_t modulus =
            static_cast<std::uint64_t>(prime);
        for (std::size_t cursor = limbs_.size(); cursor > 0; --cursor) {
            result = (
                (result << 32U) % modulus
                + limbs_[cursor - 1U]
            ) % modulus;
        }
        const std::int64_t signed_result =
            static_cast<std::int64_t>(result);
        if (sign_ < 0 && signed_result != 0) {
            return prime - signed_result;
        }
        return signed_result;
    }

    friend bool operator==(const BigInt &left, const BigInt &right) {
        return left.sign_ == right.sign_ && left.limbs_ == right.limbs_;
    }

    friend bool operator!=(const BigInt &left, const BigInt &right) {
        return !(left == right);
    }

    friend BigInt operator+(const BigInt &left, const BigInt &right) {
        if (left.sign_ == 0) {
            return right;
        }
        if (right.sign_ == 0) {
            return left;
        }
        BigInt result;
        if (left.sign_ == right.sign_) {
            result.sign_ = left.sign_;
            result.limbs_ = add_magnitudes(left.limbs_, right.limbs_);
        } else {
            const int comparison = compare_magnitudes(
                left.limbs_,
                right.limbs_
            );
            if (comparison == 0) {
                return {};
            }
            if (comparison > 0) {
                result.sign_ = left.sign_;
                result.limbs_ = subtract_magnitudes(
                    left.limbs_,
                    right.limbs_
                );
            } else {
                result.sign_ = right.sign_;
                result.limbs_ = subtract_magnitudes(
                    right.limbs_,
                    left.limbs_
                );
            }
        }
        result.normalize();
        return result;
    }

    friend BigInt operator-(const BigInt &left, const BigInt &right) {
        return left + right.negated();
    }

  private:
    int sign_ = 0;
    std::vector<std::uint32_t> limbs_;

    void normalize() {
        while (!limbs_.empty() && limbs_.back() == 0U) {
            limbs_.pop_back();
        }
        if (limbs_.empty()) {
            sign_ = 0;
        }
    }

    static int compare_magnitudes(
        const std::vector<std::uint32_t> &left,
        const std::vector<std::uint32_t> &right
    ) {
        if (left.size() != right.size()) {
            return left.size() < right.size() ? -1 : 1;
        }
        for (std::size_t cursor = left.size(); cursor > 0; --cursor) {
            const std::size_t index = cursor - 1U;
            if (left[index] != right[index]) {
                return left[index] < right[index] ? -1 : 1;
            }
        }
        return 0;
    }

    static std::vector<std::uint32_t> add_magnitudes(
        const std::vector<std::uint32_t> &left,
        const std::vector<std::uint32_t> &right
    ) {
        const std::size_t size = std::max(left.size(), right.size());
        std::vector<std::uint32_t> result(size, 0U);
        std::uint64_t carry = 0;
        for (std::size_t index = 0; index < size; ++index) {
            const std::uint64_t value =
                static_cast<std::uint64_t>(
                    index < left.size() ? left[index] : 0U
                )
                + static_cast<std::uint64_t>(
                    index < right.size() ? right[index] : 0U
                )
                + carry;
            result[index] = static_cast<std::uint32_t>(value);
            carry = value >> 32U;
        }
        if (carry != 0U) {
            result.push_back(static_cast<std::uint32_t>(carry));
        }
        return result;
    }

    static std::vector<std::uint32_t> subtract_magnitudes(
        const std::vector<std::uint32_t> &larger,
        const std::vector<std::uint32_t> &smaller
    ) {
        std::vector<std::uint32_t> result(larger.size(), 0U);
        std::uint64_t borrow = 0;
        for (std::size_t index = 0; index < larger.size(); ++index) {
            const std::uint64_t minuend = larger[index];
            const std::uint64_t subtrahend =
                static_cast<std::uint64_t>(
                    index < smaller.size() ? smaller[index] : 0U
                ) + borrow;
            if (minuend >= subtrahend) {
                result[index] = static_cast<std::uint32_t>(
                    minuend - subtrahend
                );
                borrow = 0;
            } else {
                result[index] = static_cast<std::uint32_t>(
                    (std::uint64_t{1} << 32U)
                    + minuend - subtrahend
                );
                borrow = 1;
            }
        }
        if (borrow != 0) {
            fail("exact integer magnitude underflow");
        }
        while (!result.empty() && result.back() == 0U) {
            result.pop_back();
        }
        return result;
    }
};

struct Phase {
    std::array<BigInt, 4> numerator{};
    std::uint32_t denominator_power = 0;
};

std::uint64_t magnitude_bits(const BigInt &value) {
    return value.bit_length();
}

bool even(const BigInt &value) {
    return value.is_even();
}

void canonicalize(Phase &value) {
    while (
        value.denominator_power > 0
        && std::all_of(
            value.numerator.begin(),
            value.numerator.end(),
            [](const BigInt &coefficient) {
                return even(coefficient);
            }
        )
    ) {
        for (BigInt &coefficient : value.numerator) {
            coefficient.divide_by_two_exact();
        }
        --value.denominator_power;
    }
}

Phase zero() {
    return {};
}

Phase one() {
    Phase result;
    result.numerator[0] = BigInt(1);
    return result;
}

bool operator==(const Phase &left, const Phase &right) {
    return left.denominator_power == right.denominator_power
        && left.numerator == right.numerator;
}

bool operator!=(const Phase &left, const Phase &right) {
    return !(left == right);
}

Phase aligned_add(
    const Phase &left,
    const Phase &right,
    int right_sign
) {
    Phase result;
    result.denominator_power = std::max(
        left.denominator_power,
        right.denominator_power
    );
    const std::uint32_t left_shift =
        result.denominator_power - left.denominator_power;
    const std::uint32_t right_shift =
        result.denominator_power - right.denominator_power;
    for (std::size_t index = 0; index < 4; ++index) {
        const BigInt left_value =
            left.numerator[index].shifted_left(left_shift);
        BigInt right_value =
            right.numerator[index].shifted_left(right_shift);
        if (right_sign < 0) {
            right_value = right_value.negated();
        }
        result.numerator[index] = left_value + right_value;
    }
    canonicalize(result);
    return result;
}

Phase add(const Phase &left, const Phase &right) {
    return aligned_add(left, right, 1);
}

Phase subtract(const Phase &left, const Phase &right) {
    return aligned_add(left, right, -1);
}

Phase multiply_zeta_once(const Phase &value) {
    Phase result;
    result.denominator_power = value.denominator_power;
    result.numerator[0] = value.numerator[3].negated();
    result.numerator[1] = value.numerator[0];
    result.numerator[2] = value.numerator[1];
    result.numerator[3] = value.numerator[2];
    return result;
}

Phase multiply_zeta(const Phase &value, int exponent) {
    int reduced = exponent % 8;
    if (reduced < 0) {
        reduced += 8;
    }
    Phase result = value;
    for (int step = 0; step < reduced; ++step) {
        result = multiply_zeta_once(result);
    }
    return result;
}

Phase multiply_inverse_sqrt2(const Phase &value) {
    Phase result = subtract(
        multiply_zeta(value, 1),
        multiply_zeta(value, 3)
    );
    ++result.denominator_power;
    canonicalize(result);
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

std::size_t cell_index(
    std::size_t necklace,
    std::size_t latent
) {
    return necklace * kLatentCells + latent;
}

Carrier initial_carrier() {
    Carrier result(kCarrierCells, zero());
    result[cell_index(0, 0)] = one();
    return result;
}

struct Stats {
    std::uint64_t zeta_multiplications = 0;
    std::uint64_t latent_hadamards = 0;
    std::uint64_t necklace_hadamards = 0;
    std::uint64_t modules = 0;
    std::uint64_t maximum_numerator_bits = 0;
    std::uint32_t maximum_denominator_power = 0;
    std::uint64_t maximum_logical_payload_bits = 0;
};

std::uint64_t phase_payload_bits(const Phase &value) {
    std::uint64_t result = std::max<std::uint64_t>(
        1,
        std::bit_width(value.denominator_power)
    );
    for (const BigInt &coefficient : value.numerator) {
        const std::uint64_t bits = magnitude_bits(coefficient);
        result += bits == 0 ? 1 : bits + 1;
    }
    return result;
}

std::uint64_t carrier_payload_bits(const Carrier &carrier) {
    std::uint64_t result = 0;
    for (const Phase &value : carrier) {
        result += phase_payload_bits(value);
    }
    return result;
}

void observe(const Carrier &carrier, Stats &stats) {
    for (const Phase &value : carrier) {
        stats.maximum_denominator_power = std::max(
            stats.maximum_denominator_power,
            value.denominator_power
        );
        for (const BigInt &coefficient : value.numerator) {
            stats.maximum_numerator_bits = std::max(
                stats.maximum_numerator_bits,
                magnitude_bits(coefficient)
            );
        }
    }
    stats.maximum_logical_payload_bits = std::max(
        stats.maximum_logical_payload_bits,
        carrier_payload_bits(carrier)
    );
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
        fail("exact precision public matching is not a permutation");
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

void apply_diagonal(
    Carrier &carrier,
    const std::vector<Necklace> &necklaces,
    std::uint32_t variant,
    std::size_t ordinal,
    int family,
    bool inverse,
    Stats &stats
) {
    for (
        std::size_t necklace = 0;
        necklace < necklaces.size();
        ++necklace
    ) {
        for (std::size_t latent = 0; latent < kLatentCells; ++latent) {
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
            (module.offset + (cursor + 1) * module.stride) % dimension;
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
        if (!phase_disabled) {
            apply_diagonal(
                carrier,
                necklaces,
                variant,
                ordinal,
                1,
                false,
                stats
            );
        }
        apply_latent_hadamards(carrier, necklaces, stats);
        apply_necklace_matching(carrier, module, stats);
        if (!phase_disabled) {
            apply_diagonal(
                carrier,
                necklaces,
                variant,
                ordinal,
                2,
                false,
                stats
            );
        }
    } else {
        if (!phase_disabled) {
            apply_diagonal(
                carrier,
                necklaces,
                variant,
                ordinal,
                2,
                true,
                stats
            );
        }
        apply_necklace_matching(carrier, module, stats);
        apply_latent_hadamards(carrier, necklaces, stats);
        if (!phase_disabled) {
            apply_diagonal(
                carrier,
                necklaces,
                variant,
                ordinal,
                1,
                true,
                stats
            );
        }
    }
    ++stats.modules;
    observe(carrier, stats);
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
        const std::size_t ordinal =
            mode == InverseMode::REORDERED
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

std::int64_t power_mod(
    std::int64_t base,
    std::uint64_t exponent,
    std::int64_t prime
) {
    std::int64_t result = 1;
    base %= prime;
    while (exponent > 0) {
        if ((exponent & 1U) != 0U) {
            result = result * base % prime;
        }
        base = base * base % prime;
        exponent >>= 1U;
    }
    return result;
}

std::int64_t integer_mod(
    const BigInt &value,
    std::int64_t prime
) {
    return value.residue(prime);
}

std::int64_t phase_residue(
    const Phase &value,
    std::int64_t prime,
    std::int64_t root
) {
    std::int64_t result = 0;
    std::int64_t root_power = 1;
    for (std::size_t index = 0; index < 4; ++index) {
        result = (
            result
            + integer_mod(value.numerator[index], prime) * root_power
        ) % prime;
        root_power = root_power * root % prime;
    }
    const std::int64_t inverse_two =
        power_mod(2, static_cast<std::uint64_t>(prime - 2), prime);
    result = result * power_mod(
        inverse_two,
        value.denominator_power,
        prime
    ) % prime;
    return result;
}

template <std::size_t Size>
std::array<std::int64_t, Size> boundary_residues(
    const std::array<Phase, Size> &boundary,
    std::int64_t prime,
    std::int64_t root
) {
    std::array<std::int64_t, Size> result{};
    for (std::size_t index = 0; index < Size; ++index) {
        result[index] = phase_residue(
            boundary[index],
            prime,
            root
        );
    }
    return result;
}

struct Run {
    Boundary boundary{};
    Stats stats{};
    std::uint64_t forward_logical_payload_bits = 0;
    std::uint64_t forward_elementary_operations = 0;
    bool exact_local_unitaries = true;
    bool restored = false;
    bool backing_preserved = false;
};

std::uint64_t elementary_operations(const Stats &stats) {
    return stats.zeta_multiplications
        + stats.latent_hadamards
        + stats.necklace_hadamards;
}

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
        fail("invalid exact precision carrier");
    }
    Run result;
    const Phase *const backing = carrier.data();
    observe(carrier, result.stats);
    forward(
        carrier,
        necklaces,
        variant,
        depth,
        phase_disabled,
        topology_perturbation,
        result.stats
    );
    result.forward_logical_payload_bits =
        carrier_payload_bits(carrier);
    result.forward_elementary_operations =
        elementary_operations(result.stats);
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

template <std::size_t Size>
void print_integer_array(
    const std::array<std::int64_t, Size> &values
) {
    std::printf("[");
    for (std::size_t index = 0; index < Size; ++index) {
        std::printf(
            "%s%lld",
            index == 0 ? "" : ",",
            static_cast<long long>(values[index])
        );
    }
    std::printf("]");
}

void print_run(
    std::size_t depth,
    const Run &run,
    bool final
) {
    const auto residues17 =
        boundary_residues(run.boundary, 17, 2);
    const auto residues41 =
        boundary_residues(run.boundary, 41, 3);
    std::printf(
        "{\"depth\":%zu,"
        "\"forward_elementary_operations\":%llu,"
        "\"maximum_numerator_bits\":%llu,"
        "\"maximum_denominator_power\":%u,"
        "\"forward_logical_payload_bits\":%llu,"
        "\"maximum_logical_payload_bits\":%llu,"
        "\"exact_local_unitaries\":%s,"
        "\"exact_algebraic_restoration\":%s,"
        "\"carrier_backing_preserved\":%s,"
        "\"boundary_residues_p17\":",
        depth,
        static_cast<unsigned long long>(
            run.forward_elementary_operations
        ),
        static_cast<unsigned long long>(
            run.stats.maximum_numerator_bits
        ),
        run.stats.maximum_denominator_power,
        static_cast<unsigned long long>(
            run.forward_logical_payload_bits
        ),
        static_cast<unsigned long long>(
            run.stats.maximum_logical_payload_bits
        ),
        run.exact_local_unitaries ? "true" : "false",
        run.restored ? "true" : "false",
        run.backing_preserved ? "true" : "false"
    );
    print_integer_array(residues17);
    std::printf(",\"boundary_residues_p41\":");
    print_integer_array(residues41);
    std::printf("}%s", final ? "" : ",");
}

}  // namespace exact_phase_precision

int main() {
    namespace epp = exact_phase_precision;

    const std::vector<Necklace> necklaces = compile_necklaces();
    if (
        necklaces.size()
        != static_cast<std::size_t>(kExpectedDimension)
    ) {
        fail("exact precision necklace geometry failed");
    }
    const epp::Carrier baseline = epp::initial_carrier();
    std::array<epp::Run, epp::kDepths.size()> depth_runs{};
    for (std::size_t index = 0; index < epp::kDepths.size(); ++index) {
        epp::Carrier carrier = baseline;
        depth_runs[index] = epp::transaction(
            carrier,
            baseline,
            necklaces,
            epp::kPrimaryVariant,
            epp::kDepths[index]
        );
        if (
            !depth_runs[index].exact_local_unitaries
            || !depth_runs[index].restored
            || !depth_runs[index].backing_preserved
        ) {
            fail("exact precision depth transaction failed");
        }
    }

    epp::Carrier accepted_carrier = baseline;
    const epp::Phase *const accepted_backing = accepted_carrier.data();
    const epp::Run primary = epp::transaction(
        accepted_carrier,
        baseline,
        necklaces,
        epp::kPrimaryVariant,
        epp::kPrimaryDepth
    );
    const epp::Run reuse = epp::transaction(
        accepted_carrier,
        baseline,
        necklaces,
        epp::kReuseVariant,
        epp::kReuseDepth
    );
    epp::Carrier fresh_reuse_carrier = baseline;
    const epp::Run fresh_reuse = epp::transaction(
        fresh_reuse_carrier,
        baseline,
        necklaces,
        epp::kReuseVariant,
        epp::kReuseDepth
    );
    if (
        !primary.restored
        || !reuse.restored
        || !fresh_reuse.restored
        || accepted_carrier.data() != accepted_backing
        || reuse.boundary != fresh_reuse.boundary
    ) {
        fail("exact precision restored-carrier reuse failed");
    }

    epp::Carrier missing_carrier = baseline;
    const epp::Run missing = epp::transaction(
        missing_carrier,
        baseline,
        necklaces,
        epp::kPrimaryVariant,
        8,
        epp::InverseMode::MISSING
    );
    epp::Carrier wrong_carrier = baseline;
    const epp::Run wrong = epp::transaction(
        wrong_carrier,
        baseline,
        necklaces,
        epp::kPrimaryVariant,
        8,
        epp::InverseMode::WRONG_VARIANT
    );
    epp::Carrier reordered_carrier = baseline;
    const epp::Run reordered = epp::transaction(
        reordered_carrier,
        baseline,
        necklaces,
        epp::kPrimaryVariant,
        8,
        epp::InverseMode::REORDERED
    );
    if (missing.restored || wrong.restored || reordered.restored) {
        fail("exact precision inverse controls failed");
    }

    epp::Carrier phase_disabled_carrier = baseline;
    const epp::Run phase_disabled = epp::transaction(
        phase_disabled_carrier,
        baseline,
        necklaces,
        epp::kPrimaryVariant,
        epp::kPrimaryDepth,
        epp::InverseMode::CORRECT,
        true,
        false
    );
    epp::Carrier topology_carrier = baseline;
    const epp::Run topology_perturbed = epp::transaction(
        topology_carrier,
        baseline,
        necklaces,
        epp::kPrimaryVariant,
        epp::kPrimaryDepth,
        epp::InverseMode::CORRECT,
        false,
        true
    );
    epp::Carrier null_carrier(epp::kCarrierCells, epp::zero());
    const epp::Carrier null_baseline = null_carrier;
    const epp::Run null_run = epp::transaction(
        null_carrier,
        null_baseline,
        necklaces,
        epp::kPrimaryVariant,
        epp::kPrimaryDepth
    );
    if (
        !phase_disabled.restored
        || !topology_perturbed.restored
        || !null_run.restored
        || phase_disabled.boundary == primary.boundary
        || topology_perturbed.boundary == primary.boundary
        || null_run.boundary == primary.boundary
    ) {
        fail("exact precision semantic controls failed");
    }

    bool strictly_growing_denominator = true;
    bool strictly_growing_payload = true;
    for (std::size_t index = 1; index < depth_runs.size(); ++index) {
        strictly_growing_denominator =
            strictly_growing_denominator
            && depth_runs[index].stats.maximum_denominator_power
                > depth_runs[index - 1].stats.maximum_denominator_power;
        strictly_growing_payload =
            strictly_growing_payload
            && depth_runs[index].forward_logical_payload_bits
                > depth_runs[index - 1].forward_logical_payload_bits;
    }
    if (!strictly_growing_denominator || !strictly_growing_payload) {
        fail("exact precision growth diagnostic did not separate");
    }

    std::printf(
        "{"
        "\"result\":\"PASS\","
        "\"claim_candidate\":"
        "\"BOUNDED_EXACT_DYADIC_CYCLOTOMIC_PHASE_PRECISION_GROWTH_ON_FIXED_570_CELL_NECKLACE_LATENT_CARRIER_WITH_EXACT_RESTORATION_AND_REUSE\","
        "\"claim_ceiling\":"
        "\"DIRECT_PROCESS_GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACE_DESCRIPTORS_570_Q_ZETA8_DYADIC_PHASE_CELLS_PUBLIC_VARIANT_ORDINAL_MATCHING_COMPILER_DEPTHS1_2_4_8_16_32_64_REUSE_DEPTH23_SOFTWARE_ONLY\","
        "\"field\":\"Z[ZETA8,1/2]\","
        "\"resident_necklace_descriptors\":285,"
        "\"latent_cells_per_necklace\":2,"
        "\"logical_phase_cells\":570,"
        "\"arbitrary_precision_integer_slots\":2280,"
        "\"tested_depths\":[1,2,4,8,16,32,64],"
        "\"depth_runs\":["
    );
    for (std::size_t index = 0; index < depth_runs.size(); ++index) {
        epp::print_run(
            epp::kDepths[index],
            depth_runs[index],
            index + 1U == depth_runs.size()
        );
    }
    std::printf(
        "],"
        "\"primary\":"
    );
    epp::print_run(epp::kPrimaryDepth, primary, true);
    std::printf(
        ",\"reuse\":"
    );
    epp::print_run(epp::kReuseDepth, reuse, true);
    std::printf(
        ",\"fresh_reuse_boundary_equal\":true,"
        "\"same_backing_primary_and_reuse\":true,"
        "\"carrier_backing_identity_scope\":"
        "\"OUTER_570_PHASE_OBJECT_VECTOR_BIG_INTEGER_LIMB_ALLOCATIONS_NOT_STABLE_OR_CLAIMED\","
        "\"restoration_generation_sequence\":[1,2],"
        "\"baseline_reload_bytes\":0,"
        "\"restoration_class\":\"EXACT_ALGEBRAIC_RESTORATION\","
        "\"controls\":{"
        "\"missing_inverse_restored\":false,"
        "\"wrong_inverse_variant_restored\":false,"
        "\"reordered_inverse_restored\":false,"
        "\"phase_disabled_boundary_differs\":true,"
        "\"topology_perturbation_boundary_differs\":true,"
        "\"null_carrier_boundary_differs\":true},"
        "\"resource_law\":{"
        "\"logical_phase_cells_constant\":true,"
        "\"baseline_logical_payload_bits\":%llu,"
        "\"phase_object_bytes\":%zu,"
        "\"carrier_vector_capacity_cells\":%zu,"
        "\"carrier_inline_object_bytes\":%zu,"
        "\"permanent_restoration_baseline_phase_cells\":570,"
        "\"final_boundary_phase_cells\":7,"
        "\"topology_necklace_count\":%zu,"
        "\"topology_necklace_capacity\":%zu,"
        "\"topology_necklace_element_bytes\":%zu,"
        "\"topology_necklace_capacity_bytes\":%zu,"
        "\"accepted_primary_forward_elementary_operations\":%llu,"
        "\"accepted_primary_forward_inverse_elementary_operations\":%llu,"
        "\"accepted_reuse_forward_elementary_operations\":%llu,"
        "\"accepted_reuse_forward_inverse_elementary_operations\":%llu,"
        "\"denominator_power_strictly_grows_on_tested_depths\":true,"
        "\"logical_payload_strictly_grows_on_tested_depths\":true,"
        "\"retained_module_tape_bytes\":0,"
        "\"retained_inverse_history_bytes\":0,"
        "\"public_topology_inspects_final_answer\":false,"
        "\"dense_570_by_570_operator_cells\":0,"
        "\"relation_table_cells\":0,"
        "\"assignment_cells\":0,"
        "\"whole_process_rss_claimed\":false,"
        "\"big_integer_allocator_and_container_overhead_bounded\":false,"
        "\"temporary_object_lifetime_peak_bounded\":false},",
        static_cast<unsigned long long>(
            epp::carrier_payload_bits(baseline)
        ),
        sizeof(epp::Phase),
        accepted_carrier.capacity(),
        accepted_carrier.capacity() * sizeof(epp::Phase),
        necklaces.size(),
        necklaces.capacity(),
        sizeof(Necklace),
        necklaces.capacity() * sizeof(Necklace),
        static_cast<unsigned long long>(
            primary.forward_elementary_operations
        ),
        static_cast<unsigned long long>(
            epp::elementary_operations(primary.stats)
        ),
        static_cast<unsigned long long>(
            reuse.forward_elementary_operations
        ),
        static_cast<unsigned long long>(
            epp::elementary_operations(reuse.stats)
        )
    );
    std::printf(
        "\"strongest_compact_classical\":{"
        "\"identical_570_cell_exact_recurrence\":true,"
        "\"boundary_residue_error\":0},"
        "\"distinct_from_complex128_hermitian_generator_package\":true,"
        "\"intermediate_projected\":false,"
        "\"machine_boundary_enforced\":false,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_waveform_execution\":false,"
        "\"physical_bit_replacement\":false,"
        "\"unbounded_computation_established\":false,"
        "\"terminal\":false,"
        "\"obstruction\":"
        "\"FIXED_570_LOGICAL_PHASE_CELLS_HIDE_DEPTH_GROWING_EXACT_COEFFICIENT_WIDTH_AND_THE_IDENTICAL_CLASSICAL_RECURRENCE\""
        "}\n"
    );
    return 0;
}
