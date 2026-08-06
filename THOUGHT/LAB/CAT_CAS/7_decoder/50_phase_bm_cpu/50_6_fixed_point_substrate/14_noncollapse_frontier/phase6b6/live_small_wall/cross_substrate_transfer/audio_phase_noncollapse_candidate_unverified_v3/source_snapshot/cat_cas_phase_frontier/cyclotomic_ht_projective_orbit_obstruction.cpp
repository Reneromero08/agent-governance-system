#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

/*
 * Exact two-cell diagnostic for the branch-native H and zeta_8 phase gate.
 *
 * Repeated U = H T has infinite analytic projective order.  The certificate
 * uses
 *
 *   q + q^-1 = trace(U)^2 / det(U) - 2 = -1 - sqrt(2)/2,
 *
 * where q is the eigenvalue ratio.  A root of unity would make q + q^-1 an
 * algebraic integer, but -1 - sqrt(2)/2 is not in Z[sqrt(2)], the ring of
 * integers of Q(sqrt(2)).  The initial basis vector is cyclic, so its
 * projective orbit is infinite.
 *
 * This rules out a fixed finite-state quotient as a lossless representation
 * of the complete orbit.  It does not rule out unbounded symbolic state,
 * approximation with an explicit error law, physical continuous carriers,
 * or other phase mechanisms.
 */

namespace ht_orbit {

constexpr std::array<std::size_t, 7> kDepths{
    1, 2, 4, 8, 16, 32, 64
};
constexpr std::size_t kPrimaryDepth = 64;
constexpr std::size_t kReuseDepth = 23;

[[noreturn]] void fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

struct Phase {
    std::array<std::int64_t, 4> numerator{};
    std::uint32_t denominator_power = 0;
};

std::int64_t checked_add(std::int64_t left, std::int64_t right) {
    std::int64_t result = 0;
    if (__builtin_add_overflow(left, right, &result)) {
        fail("bounded exact integer addition overflow");
    }
    return result;
}

std::int64_t checked_multiply(std::int64_t left, std::int64_t right) {
    std::int64_t result = 0;
    if (__builtin_mul_overflow(left, right, &result)) {
        fail("bounded exact integer multiplication overflow");
    }
    return result;
}

std::int64_t power_of_two(std::uint32_t exponent) {
    if (exponent >= 62U) {
        fail("bounded exact dyadic alignment overflow");
    }
    return std::int64_t{1} << exponent;
}

bool is_even(std::int64_t value) {
    return value % 2 == 0;
}

void canonicalize(Phase &value) {
    while (
        value.denominator_power > 0
        && is_even(value.numerator[0])
        && is_even(value.numerator[1])
        && is_even(value.numerator[2])
        && is_even(value.numerator[3])
    ) {
        for (std::int64_t &coefficient : value.numerator) {
            coefficient /= 2;
        }
        --value.denominator_power;
    }
}

Phase zero() {
    return {};
}

Phase integer(std::int64_t value) {
    Phase result;
    result.numerator[0] = value;
    return result;
}

Phase one() {
    return integer(1);
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
    result.denominator_power =
        std::max(left.denominator_power, right.denominator_power);
    const std::uint32_t left_shift =
        result.denominator_power - left.denominator_power;
    const std::uint32_t right_shift =
        result.denominator_power - right.denominator_power;
    for (std::size_t index = 0; index < 4; ++index) {
        const std::int64_t left_value = checked_multiply(
            left.numerator[index],
            power_of_two(left_shift)
        );
        std::int64_t right_value = checked_multiply(
            right.numerator[index],
            power_of_two(right_shift)
        );
        if (right_sign < 0) {
            if (right_value == std::numeric_limits<std::int64_t>::min()) {
                fail("bounded exact integer negation overflow");
            }
            right_value = -right_value;
        }
        result.numerator[index] = checked_add(left_value, right_value);
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

Phase multiply(const Phase &left, const Phase &right) {
    std::array<std::int64_t, 7> convolution{};
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            convolution[i + j] = checked_add(
                convolution[i + j],
                checked_multiply(left.numerator[i], right.numerator[j])
            );
        }
    }
    Phase result;
    result.denominator_power =
        left.denominator_power + right.denominator_power;
    for (std::size_t degree = 0; degree < convolution.size(); ++degree) {
        if (degree < 4) {
            result.numerator[degree] = checked_add(
                result.numerator[degree],
                convolution[degree]
            );
        } else {
            if (
                convolution[degree]
                == std::numeric_limits<std::int64_t>::min()
            ) {
                fail("bounded exact integer negation overflow");
            }
            result.numerator[degree - 4] = checked_add(
                result.numerator[degree - 4],
                -convolution[degree]
            );
        }
    }
    canonicalize(result);
    return result;
}

Phase multiply_zeta_once(const Phase &value) {
    Phase result;
    result.denominator_power = value.denominator_power;
    result.numerator[0] = -value.numerator[3];
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

Phase inverse_sqrt_two() {
    Phase result;
    result.numerator[1] = 1;
    result.numerator[3] = -1;
    result.denominator_power = 1;
    return result;
}

Phase conjugate(const Phase &value) {
    Phase result;
    result.denominator_power = value.denominator_power;
    result.numerator[0] = value.numerator[0];
    result.numerator[1] = -value.numerator[3];
    result.numerator[2] = -value.numerator[2];
    result.numerator[3] = -value.numerator[1];
    return result;
}

void hadamard(Phase &upper, Phase &lower) {
    const Phase original_upper = upper;
    const Phase original_lower = lower;
    const Phase scalar = inverse_sqrt_two();
    upper = multiply(scalar, add(original_upper, original_lower));
    lower = multiply(scalar, subtract(original_upper, original_lower));
}

using Carrier = std::array<Phase, 2>;

Carrier initial_carrier() {
    return {one(), zero()};
}

Phase star_norm(const Carrier &carrier) {
    return add(
        multiply(conjugate(carrier[0]), carrier[0]),
        multiply(conjugate(carrier[1]), carrier[1])
    );
}

void apply_primary_step(Carrier &carrier) {
    carrier[1] = multiply_zeta(carrier[1], 1);
    hadamard(carrier[0], carrier[1]);
}

void apply_primary_inverse(Carrier &carrier) {
    hadamard(carrier[0], carrier[1]);
    carrier[1] = multiply_zeta(carrier[1], -1);
}

void apply_wrong_primary_inverse(Carrier &carrier) {
    hadamard(carrier[0], carrier[1]);
    carrier[1] = multiply_zeta(carrier[1], -3);
}

void apply_reordered_primary_inverse(Carrier &carrier) {
    carrier[1] = multiply_zeta(carrier[1], -1);
    hadamard(carrier[0], carrier[1]);
}

void apply_reuse_step(Carrier &carrier) {
    hadamard(carrier[0], carrier[1]);
    carrier[1] = multiply_zeta(carrier[1], 3);
}

void apply_reuse_inverse(Carrier &carrier) {
    carrier[1] = multiply_zeta(carrier[1], -3);
    hadamard(carrier[0], carrier[1]);
}

void forward_primary(Carrier &carrier, std::size_t depth) {
    for (std::size_t step = 0; step < depth; ++step) {
        apply_primary_step(carrier);
    }
}

void inverse_primary(Carrier &carrier, std::size_t depth) {
    for (std::size_t step = 0; step < depth; ++step) {
        apply_primary_inverse(carrier);
    }
}

void forward_reuse(Carrier &carrier, std::size_t depth) {
    for (std::size_t step = 0; step < depth; ++step) {
        apply_reuse_step(carrier);
    }
}

void inverse_reuse(Carrier &carrier, std::size_t depth) {
    for (std::size_t step = 0; step < depth; ++step) {
        apply_reuse_inverse(carrier);
    }
}

std::uint64_t integer_bits(std::int64_t value) {
    if (value == 0) {
        return 0;
    }
    const std::uint64_t magnitude = value < 0
        ? static_cast<std::uint64_t>(-(value + 1)) + 1U
        : static_cast<std::uint64_t>(value);
    std::uint64_t bits = 0;
    std::uint64_t cursor = magnitude;
    while (cursor != 0U) {
        ++bits;
        cursor >>= 1U;
    }
    return bits;
}

std::uint64_t phase_payload_bits(const Phase &value) {
    std::uint64_t payload = 1;
    std::uint32_t denominator = value.denominator_power;
    while (denominator > 1) {
        ++payload;
        denominator >>= 1U;
    }
    for (const std::int64_t coefficient : value.numerator) {
        const std::uint64_t bits = integer_bits(coefficient);
        payload += bits == 0 ? 1 : bits + 1;
    }
    return payload;
}

std::uint64_t carrier_payload_bits(const Carrier &carrier) {
    return phase_payload_bits(carrier[0]) + phase_payload_bits(carrier[1]);
}

std::uint64_t maximum_numerator_bits(const Carrier &carrier) {
    std::uint64_t result = 0;
    for (const Phase &value : carrier) {
        for (const std::int64_t coefficient : value.numerator) {
            result = std::max(result, integer_bits(coefficient));
        }
    }
    return result;
}

std::uint32_t maximum_denominator_power(const Carrier &carrier) {
    return std::max(
        carrier[0].denominator_power,
        carrier[1].denominator_power
    );
}

bool projectively_equal(const Carrier &left, const Carrier &right) {
    return multiply(left[0], right[1]) == multiply(left[1], right[0]);
}

std::string integer_string(std::int64_t value) {
    return std::to_string(value);
}

void print_phase(const Phase &value) {
    std::printf("{\"numerator\":[");
    for (std::size_t index = 0; index < 4; ++index) {
        std::printf(
            "\"%s\"%s",
            integer_string(value.numerator[index]).c_str(),
            index + 1U == 4U ? "" : ","
        );
    }
    std::printf(
        "],\"denominator_power\":%u}",
        value.denominator_power
    );
}

void print_carrier(const Carrier &carrier) {
    std::printf("[");
    print_phase(carrier[0]);
    std::printf(",");
    print_phase(carrier[1]);
    std::printf("]");
}

struct Run {
    std::size_t depth = 0;
    Carrier boundary{};
    std::uint64_t maximum_numerator_bits = 0;
    std::uint32_t maximum_denominator_power = 0;
    std::uint64_t payload_bits = 0;
    bool restored = false;
    bool backing_preserved = false;
};

Run transaction(std::size_t depth) {
    Carrier carrier = initial_carrier();
    const Phase *const backing = carrier.data();
    forward_primary(carrier, depth);
    Run result;
    result.depth = depth;
    result.boundary = carrier;
    result.maximum_numerator_bits = maximum_numerator_bits(carrier);
    result.maximum_denominator_power =
        maximum_denominator_power(carrier);
    result.payload_bits = carrier_payload_bits(carrier);
    inverse_primary(carrier, depth);
    result.restored = carrier == initial_carrier();
    result.backing_preserved = carrier.data() == backing;
    return result;
}

void verify_theorem_certificate() {
    const Phase c = inverse_sqrt_two();
    const Phase zeta = multiply_zeta(one(), 1);
    const Phase trace = subtract(c, multiply(c, zeta));
    const Phase determinant = multiply_zeta(integer(-1), 1);
    const Phase inverse_determinant = multiply_zeta(one(), 3);
    const Phase invariant = subtract(
        multiply(multiply(trace, trace), inverse_determinant),
        integer(2)
    );
    Phase expected;
    expected.numerator = {
        std::int64_t{-2},
        std::int64_t{-1},
        std::int64_t{0},
        std::int64_t{1}
    };
    expected.denominator_power = 1;
    canonicalize(expected);
    if (
        determinant != multiply_zeta(integer(-1), 1)
        || invariant != expected
        || c == zero()
    ) {
        fail("HT analytic projective-order certificate mismatch");
    }
}

}  // namespace ht_orbit

int main() {
    using namespace ht_orbit;

    verify_theorem_certificate();

    std::array<Run, kDepths.size()> runs{};
    std::vector<Carrier> sampled_states;
    sampled_states.reserve(kDepths.size() + 1U);
    sampled_states.push_back(initial_carrier());
    for (std::size_t index = 0; index < kDepths.size(); ++index) {
        runs[index] = transaction(kDepths[index]);
        if (!runs[index].restored || !runs[index].backing_preserved) {
            fail("HT transaction restoration failure");
        }
        if (star_norm(runs[index].boundary) != one()) {
            fail("HT transaction star norm failure");
        }
        for (const Carrier &earlier : sampled_states) {
            if (projectively_equal(earlier, runs[index].boundary)) {
                fail("sampled HT projective states collided");
            }
        }
        sampled_states.push_back(runs[index].boundary);
    }

    Carrier carrier = initial_carrier();
    const Phase *const backing = carrier.data();
    forward_primary(carrier, kPrimaryDepth);
    const Carrier primary_boundary = carrier;
    inverse_primary(carrier, kPrimaryDepth);
    if (carrier != initial_carrier() || carrier.data() != backing) {
        fail("HT primary restoration failure");
    }
    const std::uint64_t generation_after_primary = 1;

    forward_reuse(carrier, kReuseDepth);
    const Carrier reuse_boundary = carrier;
    inverse_reuse(carrier, kReuseDepth);
    if (carrier != initial_carrier() || carrier.data() != backing) {
        fail("HT reuse restoration failure");
    }
    const std::uint64_t generation_after_reuse = 2;

    Carrier fresh_reuse = initial_carrier();
    forward_reuse(fresh_reuse, kReuseDepth);
    if (reuse_boundary != fresh_reuse) {
        fail("HT fresh/restored reuse mismatch");
    }

    Carrier missing_inverse = initial_carrier();
    forward_primary(missing_inverse, 19);

    Carrier wrong_inverse = initial_carrier();
    forward_primary(wrong_inverse, 19);
    for (std::size_t step = 0; step < 19; ++step) {
        apply_wrong_primary_inverse(wrong_inverse);
    }

    Carrier reordered_inverse = initial_carrier();
    forward_primary(reordered_inverse, 19);
    for (std::size_t step = 0; step < 19; ++step) {
        apply_reordered_primary_inverse(reordered_inverse);
    }

    Carrier phase_disabled = initial_carrier();
    for (std::size_t step = 0; step < 19; ++step) {
        hadamard(phase_disabled[0], phase_disabled[1]);
    }
    Carrier normal_control = initial_carrier();
    forward_primary(normal_control, 19);

    if (
        missing_inverse == initial_carrier()
        || wrong_inverse == initial_carrier()
        || reordered_inverse == initial_carrier()
        || phase_disabled == normal_control
    ) {
        fail("HT control failure");
    }

    const Carrier analytic_one = initial_carrier();
    Carrier analytic_alias = initial_carrier();
    analytic_alias[0] = integer(698);
    const std::int64_t kernel_element = 17 * 41;
    const bool demonstrated_alias_mod_17_and_41 =
        kernel_element % 17 == 0
        && kernel_element % 41 == 0
        && analytic_one != analytic_alias;
    if (!demonstrated_alias_mod_17_and_41) {
        fail("finite quotient alias control failure");
    }

    std::printf(
        "{"
        "\"result\":\"PASS\","
        "\"claim_candidate\":"
        "\"BOUNDED_EXACT_HT_ANALYTIC_INFINITE_PROJECTIVE_ORBIT_"
        "REJECTS_FIXED_FINITE_STATE_LOSSLESS_PHASE_QUOTIENT_WITH_"
        "EXACT_RESTORATION_AND_REUSE\","
        "\"gate_alphabet\":\"Z_ZETA8_DYADIC_H_AND_T\","
        "\"logical_phase_cells\":2,"
        "\"analytic_projective_orbit_infinite\":true,"
        "\"fixed_finite_state_lossless_quotient_possible\":false,"
        "\"theorem_certificate\":{"
        "\"unitary\":\"U_EQUALS_H_T\","
        "\"eigenvalue_ratio_symbol\":\"q\","
        "\"q_plus_inverse_q\":{"
        "\"rational\":-1,"
        "\"sqrt2_numerator\":-1,"
        "\"sqrt2_denominator\":2},"
        "\"quadratic_integer_ring\":\"Z_SQRT2\","
        "\"q_plus_inverse_q_is_algebraic_integer\":false,"
        "\"eigenvalue_ratio_is_root_of_unity\":false,"
        "\"initial_basis_vector_is_cyclic\":true,"
        "\"projective_orbit_is_infinite\":true},"
        "\"tested_depths\":["
    );
    for (std::size_t index = 0; index < kDepths.size(); ++index) {
        std::printf(
            "%zu%s",
            kDepths[index],
            index + 1U == kDepths.size() ? "" : ","
        );
    }
    std::printf("],\"depth_runs\":[");
    for (std::size_t index = 0; index < runs.size(); ++index) {
        const Run &run = runs[index];
        std::printf(
            "{"
            "\"depth\":%zu,"
            "\"maximum_numerator_bits\":%llu,"
            "\"maximum_denominator_power\":%u,"
            "\"logical_payload_bits\":%llu,"
            "\"star_norm_exactly_one\":true,"
            "\"exact_algebraic_restoration\":true,"
            "\"outer_carrier_backing_preserved\":true,"
            "\"boundary\":",
            run.depth,
            static_cast<unsigned long long>(
                run.maximum_numerator_bits
            ),
            run.maximum_denominator_power,
            static_cast<unsigned long long>(run.payload_bits)
        );
        print_carrier(run.boundary);
        std::printf("}%s", index + 1U == runs.size() ? "" : ",");
    }
    std::printf(
        "],"
        "\"sampled_projective_states_distinct\":true,"
        "\"primary\":{"
        "\"depth\":%zu,"
        "\"boundary\":",
        kPrimaryDepth
    );
    print_carrier(primary_boundary);
    std::printf(
        ",\"exact_algebraic_restoration\":true},"
        "\"reuse\":{"
        "\"program\":\"H_THEN_T_CUBED\","
        "\"depth\":%zu,"
        "\"boundary\":",
        kReuseDepth
    );
    print_carrier(reuse_boundary);
    std::printf(
        ",\"exact_algebraic_restoration\":true},"
        "\"fresh_restored_reuse_boundary_equal\":true,"
        "\"same_outer_carrier_backing\":true,"
        "\"restoration_generation_sequence\":[%llu,%llu],"
        "\"baseline_reload_bytes\":0,"
        "\"restoration_classification\":\"EXACT_ALGEBRAIC_RESTORATION\","
        "\"controls\":{"
        "\"missing_inverse_restored\":false,"
        "\"wrong_inverse_restored\":false,"
        "\"reordered_inverse_restored\":false,"
        "\"phase_disabled_boundary_differs\":true,"
        "\"demonstrated_nonzero_kernel_element_integer\":697,"
        "\"analytic_alias_pair_distinct\":true,"
        "\"alias_pair_equal_mod_17_and_41\":true,"
        "\"alias_pair_is_normalized_transaction\":false},"
        "\"resource_law\":{"
        "\"logical_phase_cells\":2,"
        "\"bounded_executor_signed64_integer_slots\":8,"
        "\"phase_object_bytes\":%zu,"
        "\"carrier_inline_bytes\":%zu,"
        "\"restoration_verification_baseline_inline_bytes\":%zu,"
        "\"retained_boundary_inline_bytes\":%zu,"
        "\"transaction_run_record_bytes\":%zu,"
        "\"retained_module_tape_bytes\":0,"
        "\"retained_inverse_history_bytes\":0,"
        "\"public_program_descriptor_bytes\":16,"
        "\"verification_depth_run_array_bytes\":%zu,"
        "\"compiler_binary_allocator_and_whole_process_excluded\":true,"
        "\"fixed_bit_payload_established\":false,"
        "\"whole_process_rss_claimed\":false},"
        "\"matched_compact_classical\":{"
        "\"identical_exact_two_cell_recurrence\":true,"
        "\"boundary_error\":0},"
        "\"catvm_custody_established\":false,"
        "\"distinct_phase_resource_established\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_waveform_execution\":false,"
        "\"physical_bit_replacement\":false,"
        "\"unbounded_computation_established\":false,"
        "\"claim_ceiling\":"
        "\"LINUX_X86_64_DIRECT_PROCESS_TWO_CELL_Z_ZETA8_DYADIC_ANALYTIC_"
        "HT_ORBIT_DEPTHS1_2_4_8_16_32_64_"
        "REUSE_DEPTH23_SOFTWARE_ONLY\","
        "\"terminal\":false"
        "}\n",
        static_cast<unsigned long long>(generation_after_primary),
        static_cast<unsigned long long>(generation_after_reuse),
        sizeof(Phase),
        sizeof(Carrier),
        sizeof(Carrier),
        sizeof(Carrier),
        sizeof(Run),
        sizeof(runs)
    );
    return 0;
}
