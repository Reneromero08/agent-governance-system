#define _POSIX_C_SOURCE 200809L

/*
 * Mutable CAT_CAS frontier: nonlinear symplectic open-relation signatures.
 *
 * A two-mode Kerr Hamiltonian and an exact rational SU(2) mixer define two
 * typed open wave relations.  Native Poisson contraction closes their
 * Hamiltonian signatures through the shared wave port.  Coefficients remain
 * resident as roots of unity in two independent prime fields.  Only the
 * final graded boundary is decoded; inverse contractions then clear the
 * actual resident coefficient blocks and the restored carrier is reused.
 *
 * The accepted result is deliberately bounded.  Successive nonzero Lie
 * grades diagnose growth of the tested polynomial-Hamiltonian signature
 * class.  They do not prove unbounded growth, advantage, physical waveform
 * execution, or a Small Wall crossing.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SR_VARIABLES 4U
#define SR_GRADES 5U
#define SR_INITIAL_DEGREE 4U
#define SR_MAX_DEGREE 12U
#define SR_MAX_COEFFICIENTS 455U
#define SR_PRIMES 2U
#define SR_MAX_PRIME 19U
#define SR_SMALL_TERMS 256U
#define SR_REUSE_CYCLES 8U

static const int SR_FIELD[SR_PRIMES] = {17, 19};
static const double SR_PI =
    3.14159265358979323846264338327950288;
static const double SR_UNIT_TOLERANCE = 5.0e-10;
static const double SR_ROOT_TOLERANCE = 1.0e-3;

struct sr_monomial {
    uint8_t exponent[SR_VARIABLES];
};

struct sr_block {
    size_t degree;
    size_t count;
    struct sr_monomial monomial[SR_MAX_COEFFICIENTS];
    double complex baseline[SR_PRIMES][SR_MAX_COEFFICIENTS];
    double complex working[SR_PRIMES][SR_MAX_COEFFICIENTS];
};

struct sr_carrier {
    struct sr_block block[SR_GRADES];
};

struct sr_small_term {
    struct sr_monomial monomial;
    int coefficient[SR_PRIMES];
};

struct sr_small_poly {
    size_t count;
    struct sr_small_term term[SR_SMALL_TERMS];
};

struct sr_compiled_generator {
    struct sr_small_poly polynomial;
    double complex phase[SR_PRIMES][SR_SMALL_TERMS];
};

struct sr_stats {
    uint64_t native_phase_updates;
    uint64_t resident_phase_reads;
    uint64_t field_product_interpolations;
    uint64_t poisson_monomial_products;
    uint64_t hidden_coefficient_decodes;
    uint64_t final_coefficient_decodes;
    double maximum_unit_modulus_error;
    double maximum_final_root_error;
};

struct sr_grade_boundary {
    size_t degree;
    size_t coefficient_cells;
    size_t nonzero[SR_PRIMES];
    uint64_t hash[SR_PRIMES];
};

struct sr_boundary {
    struct sr_grade_boundary grade[SR_GRADES];
    uint64_t combined_hash;
};

struct sr_execution {
    struct sr_boundary boundary;
    struct sr_stats stats;
    double restoration_max_abs;
    uint64_t restoration_nonidentity_cells;
    uint64_t restoration_generation;
    int actual_inverse;
    int snapshot_loaded;
};

enum sr_program {
    SR_PRIMARY = 0,
    SR_REUSE = 1,
    SR_IDENTITY_MIXER = 2,
    SR_ZERO_KERR = 3,
    SR_SWAPPED = 4
};

enum sr_mode {
    SR_CORRECT = 0,
    SR_MISSING_INVERSE = 1,
    SR_WRONG_INVERSE = 2,
    SR_REORDERED_INVERSE = 3,
    SR_SNAPSHOT = 4
};

static void sr_fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static int sr_mod(int value, int prime) {
    const int residue = value % prime;
    return residue < 0 ? residue + prime : residue;
}

static int sr_inverse(int value, int prime) {
    int t = 0;
    int new_t = 1;
    int r = prime;
    int new_r = sr_mod(value, prime);
    while (new_r != 0) {
        const int quotient = r / new_r;
        const int next_t = t - quotient * new_t;
        const int next_r = r - quotient * new_r;
        t = new_t;
        new_t = next_t;
        r = new_r;
        new_r = next_r;
    }
    if (r != 1) {
        sr_fail("noninvertible public rational coefficient");
    }
    return sr_mod(t, prime);
}

static double complex sr_unit(double complex value) {
    const double magnitude = cabs(value);
    if (!(magnitude > 0.0) || !isfinite(magnitude)) {
        sr_fail("nonfinite symplectic relation phase");
    }
    return value / magnitude;
}

static double complex sr_root(size_t field, int value) {
    const int prime = SR_FIELD[field];
    return cexp(
        I * 2.0 * SR_PI
        * (double)sr_mod(value, prime)
        / (double)prime
    );
}

static int sr_same_monomial(
    const struct sr_monomial *left,
    const struct sr_monomial *right
) {
    return memcmp(left->exponent, right->exponent, SR_VARIABLES) == 0;
}

static size_t sr_fill_monomials(
    size_t degree,
    struct sr_monomial output[SR_MAX_COEFFICIENTS]
) {
    size_t count = 0U;
    for (size_t q1 = 0U; q1 <= degree; ++q1) {
        for (size_t p1 = 0U; p1 <= degree - q1; ++p1) {
            for (
                size_t q2 = 0U;
                q2 <= degree - q1 - p1;
                ++q2
            ) {
                const size_t p2 = degree - q1 - p1 - q2;
                if (count >= SR_MAX_COEFFICIENTS) {
                    sr_fail("homogeneous signature capacity exceeded");
                }
                output[count].exponent[0] = (uint8_t)q1;
                output[count].exponent[1] = (uint8_t)p1;
                output[count].exponent[2] = (uint8_t)q2;
                output[count].exponent[3] = (uint8_t)p2;
                ++count;
            }
        }
    }
    return count;
}

static size_t sr_find_monomial(
    const struct sr_block *block,
    const struct sr_monomial *monomial
) {
    for (size_t index = 0U; index < block->count; ++index) {
        if (sr_same_monomial(&block->monomial[index], monomial)) {
            return index;
        }
    }
    sr_fail("Poisson product left target homogeneous grade");
    return 0U;
}

static struct sr_carrier *sr_make_carrier(uint64_t identity) {
    (void)identity;
    struct sr_carrier *carrier = calloc(1U, sizeof(*carrier));
    if (carrier == NULL) {
        sr_fail("unable to allocate symplectic relation carrier");
    }
    for (size_t grade = 0U; grade < SR_GRADES; ++grade) {
        struct sr_block *block = &carrier->block[grade];
        block->degree = SR_INITIAL_DEGREE + 2U * grade;
        block->count = sr_fill_monomials(
            block->degree, block->monomial
        );
        for (size_t field = 0U; field < SR_PRIMES; ++field) {
            for (size_t cell = 0U; cell < block->count; ++cell) {
                block->baseline[field][cell] = 1.0 + 0.0 * I;
                block->working[field][cell] =
                    block->baseline[field][cell];
            }
        }
    }
    return carrier;
}

static double complex sr_relative(
    const struct sr_block *block,
    size_t field,
    size_t cell
) {
    return block->working[field][cell]
        * conj(block->baseline[field][cell]);
}

static void sr_observe(
    const struct sr_carrier *carrier,
    struct sr_stats *stats
) {
    for (size_t grade = 0U; grade < SR_GRADES; ++grade) {
        const struct sr_block *block = &carrier->block[grade];
        for (size_t field = 0U; field < SR_PRIMES; ++field) {
            for (size_t cell = 0U; cell < block->count; ++cell) {
                const double error =
                    fabs(cabs(sr_relative(block, field, cell)) - 1.0);
                if (error > stats->maximum_unit_modulus_error) {
                    stats->maximum_unit_modulus_error = error;
                }
                if (error > SR_UNIT_TOLERANCE) {
                    sr_fail("symplectic relation carrier left unit torus");
                }
            }
        }
    }
}

static void sr_multiply(
    struct sr_block *block,
    size_t field,
    size_t cell,
    double complex factor,
    struct sr_stats *stats
) {
    block->working[field][cell] = sr_unit(
        sr_relative(block, field, cell) * factor
    ) * block->baseline[field][cell];
    ++stats->native_phase_updates;
}

/*
 * Root-of-unity Fourier interpolant for field multiplication:
 * P(X,Y)=(1/p) sum_{r,s} zeta^(-rs) X^r Y^s.
 * It maps X=zeta^a,Y=zeta^b to zeta^(ab) without decoding a or b.
 */
static double complex sr_field_product(
    double complex left,
    double complex right,
    size_t field,
    struct sr_stats *stats
) {
    const size_t prime = (size_t)SR_FIELD[field];
    double complex left_power[SR_MAX_PRIME];
    double complex right_power[SR_MAX_PRIME];
    left_power[0] = 1.0 + 0.0 * I;
    right_power[0] = 1.0 + 0.0 * I;
    for (size_t power = 1U; power < prime; ++power) {
        left_power[power] = left_power[power - 1U] * left;
        right_power[power] = right_power[power - 1U] * right;
    }
    double complex sum = 0.0 + 0.0 * I;
    for (size_t r = 0U; r < prime; ++r) {
        for (size_t s = 0U; s < prime; ++s) {
            sum +=
                sr_root(field, -(int)(r * s))
                * left_power[r]
                * right_power[s];
        }
    }
    ++stats->field_product_interpolations;
    return sr_unit(sum / (double)prime);
}

static double complex sr_phase_power(
    double complex phase,
    int exponent,
    int prime
) {
    unsigned int power =
        (unsigned int)sr_mod(exponent, prime);
    double complex result = 1.0 + 0.0 * I;
    double complex factor = phase;
    while (power != 0U) {
        if ((power & 1U) != 0U) {
            result *= factor;
        }
        factor *= factor;
        power >>= 1U;
    }
    return sr_unit(result);
}

static void sr_small_add(
    struct sr_small_poly *polynomial,
    const struct sr_monomial *monomial,
    const int coefficient[SR_PRIMES]
) {
    for (size_t index = 0U; index < polynomial->count; ++index) {
        if (sr_same_monomial(
            &polynomial->term[index].monomial, monomial
        )) {
            for (size_t field = 0U; field < SR_PRIMES; ++field) {
                polynomial->term[index].coefficient[field] = sr_mod(
                    polynomial->term[index].coefficient[field]
                        + coefficient[field],
                    SR_FIELD[field]
                );
            }
            return;
        }
    }
    if (polynomial->count >= SR_SMALL_TERMS) {
        sr_fail("public polynomial compiler capacity exceeded");
    }
    struct sr_small_term *term =
        &polynomial->term[polynomial->count++];
    term->monomial = *monomial;
    for (size_t field = 0U; field < SR_PRIMES; ++field) {
        term->coefficient[field] =
            sr_mod(coefficient[field], SR_FIELD[field]);
    }
}

static struct sr_small_poly sr_small_multiply(
    const struct sr_small_poly *left,
    const struct sr_small_poly *right
) {
    struct sr_small_poly result = {0};
    for (size_t a = 0U; a < left->count; ++a) {
        for (size_t b = 0U; b < right->count; ++b) {
            struct sr_monomial monomial = {{0}};
            int coefficient[SR_PRIMES] = {0};
            for (size_t variable = 0U; variable < SR_VARIABLES; ++variable) {
                monomial.exponent[variable] = (uint8_t)(
                    left->term[a].monomial.exponent[variable]
                    + right->term[b].monomial.exponent[variable]
                );
            }
            for (size_t field = 0U; field < SR_PRIMES; ++field) {
                coefficient[field] = sr_mod(
                    left->term[a].coefficient[field]
                    * right->term[b].coefficient[field],
                    SR_FIELD[field]
                );
            }
            sr_small_add(&result, &monomial, coefficient);
        }
    }
    return result;
}

static struct sr_small_poly sr_small_add_poly(
    const struct sr_small_poly *left,
    const struct sr_small_poly *right
) {
    struct sr_small_poly result = *left;
    for (size_t index = 0U; index < right->count; ++index) {
        sr_small_add(
            &result,
            &right->term[index].monomial,
            right->term[index].coefficient
        );
    }
    return result;
}

static struct sr_small_poly sr_linear(
    size_t first_variable,
    int first_numerator,
    size_t second_variable,
    int second_numerator,
    int denominator
) {
    struct sr_small_poly result = {0};
    for (size_t choice = 0U; choice < 2U; ++choice) {
        struct sr_monomial monomial = {{0}};
        monomial.exponent[
            choice == 0U ? first_variable : second_variable
        ] = 1U;
        int coefficient[SR_PRIMES] = {0};
        for (size_t field = 0U; field < SR_PRIMES; ++field) {
            const int numerator = choice == 0U
                ? first_numerator
                : second_numerator;
            coefficient[field] = sr_mod(
                numerator * sr_inverse(denominator, SR_FIELD[field]),
                SR_FIELD[field]
            );
        }
        sr_small_add(&result, &monomial, coefficient);
    }
    return result;
}

static struct sr_small_poly sr_kerr_from_mixer(
    int cosine_numerator,
    int sine_numerator,
    int denominator,
    int zero
) {
    struct sr_small_poly empty = {0};
    if (zero) {
        return empty;
    }
    const struct sr_small_poly q1 = sr_linear(
        0U, cosine_numerator, 2U, -sine_numerator, denominator
    );
    const struct sr_small_poly p1 = sr_linear(
        1U, cosine_numerator, 3U, -sine_numerator, denominator
    );
    const struct sr_small_poly q2 = sr_linear(
        0U, sine_numerator, 2U, cosine_numerator, denominator
    );
    const struct sr_small_poly p2 = sr_linear(
        1U, sine_numerator, 3U, cosine_numerator, denominator
    );
    const struct sr_small_poly q1_squared = sr_small_multiply(&q1, &q1);
    const struct sr_small_poly p1_squared = sr_small_multiply(&p1, &p1);
    const struct sr_small_poly q2_squared = sr_small_multiply(&q2, &q2);
    const struct sr_small_poly p2_squared = sr_small_multiply(&p2, &p2);
    const struct sr_small_poly radius1 =
        sr_small_add_poly(&q1_squared, &p1_squared);
    const struct sr_small_poly radius2 =
        sr_small_add_poly(&q2_squared, &p2_squared);
    const struct sr_small_poly fourth1 =
        sr_small_multiply(&radius1, &radius1);
    const struct sr_small_poly fourth2 =
        sr_small_multiply(&radius2, &radius2);
    return sr_small_add_poly(&fourth1, &fourth2);
}

static struct sr_compiled_generator sr_compile_generator(
    enum sr_program program,
    int seed
) {
    int cosine = 3;
    int sine = 4;
    int denominator = 5;
    int zero = 0;
    if (program == SR_REUSE && !seed) {
        cosine = 4;
        sine = 3;
    } else if (program == SR_IDENTITY_MIXER) {
        cosine = 1;
        sine = 0;
        denominator = 1;
    } else if (program == SR_ZERO_KERR) {
        zero = 1;
    } else if (program == SR_SWAPPED) {
        if (seed) {
            cosine = 3;
            sine = 4;
        } else {
            cosine = 1;
            sine = 0;
            denominator = 1;
        }
    } else if (seed) {
        cosine = 1;
        sine = 0;
        denominator = 1;
    }
    struct sr_compiled_generator compiled = {
        .polynomial = sr_kerr_from_mixer(
            cosine, sine, denominator, zero
        )
    };
    for (size_t field = 0U; field < SR_PRIMES; ++field) {
        for (
            size_t term = 0U;
            term < compiled.polynomial.count;
            ++term
        ) {
            compiled.phase[field][term] = sr_root(
                field,
                compiled.polynomial.term[term].coefficient[field]
            );
        }
    }
    return compiled;
}

static void sr_seal_seed(
    struct sr_carrier *carrier,
    const struct sr_compiled_generator *seed,
    int inverse,
    struct sr_stats *stats
) {
    struct sr_block *block = &carrier->block[0];
    for (size_t term = 0U; term < seed->polynomial.count; ++term) {
        const size_t cell = sr_find_monomial(
            block, &seed->polynomial.term[term].monomial
        );
        for (size_t field = 0U; field < SR_PRIMES; ++field) {
            double complex factor = seed->phase[field][term];
            if (inverse) {
                factor = conj(factor);
            }
            sr_multiply(block, field, cell, factor, stats);
        }
    }
}

static void sr_accumulate_bracket(
    struct sr_carrier *carrier,
    size_t source_grade,
    const struct sr_compiled_generator *generator,
    int inverse,
    int wrong,
    struct sr_stats *stats
) {
    struct sr_block *source = &carrier->block[source_grade];
    struct sr_block *target = &carrier->block[source_grade + 1U];
    for (
        size_t generator_term = 0U;
        generator_term < generator->polynomial.count;
        ++generator_term
    ) {
        const struct sr_monomial *left =
            &generator->polynomial.term[generator_term].monomial;
        for (size_t source_term = 0U; source_term < source->count; ++source_term) {
            const struct sr_monomial *right =
                &source->monomial[source_term];
            for (size_t pair = 0U; pair < 2U; ++pair) {
                const size_t q = 2U * pair;
                const size_t p = q + 1U;
                const int factor =
                    (int)left->exponent[q] * (int)right->exponent[p]
                    - (int)left->exponent[p] * (int)right->exponent[q];
                if (factor == 0) {
                    continue;
                }
                struct sr_monomial product = {{0}};
                for (
                    size_t variable = 0U;
                    variable < SR_VARIABLES;
                    ++variable
                ) {
                    product.exponent[variable] = (uint8_t)(
                        left->exponent[variable]
                        + right->exponent[variable]
                    );
                }
                --product.exponent[q];
                --product.exponent[p];
                const size_t target_term =
                    sr_find_monomial(target, &product);
                for (size_t field = 0U; field < SR_PRIMES; ++field) {
                    const double complex source_phase =
                        sr_relative(source, field, source_term);
                    ++stats->resident_phase_reads;
                    double complex contribution = sr_field_product(
                        generator->phase[field][generator_term],
                        source_phase,
                        field,
                        stats
                    );
                    contribution = sr_phase_power(
                        contribution, factor, SR_FIELD[field]
                    );
                    if (inverse) {
                        contribution = conj(contribution);
                    }
                    if (wrong) {
                        contribution = conj(contribution);
                    }
                    sr_multiply(
                        target,
                        field,
                        target_term,
                        contribution,
                        stats
                    );
                }
                ++stats->poisson_monomial_products;
            }
        }
    }
}

static int sr_decode(
    double complex phase,
    size_t field,
    struct sr_stats *stats
) {
    int selected = 0;
    double minimum = HUGE_VAL;
    for (int value = 0; value < SR_FIELD[field]; ++value) {
        const double error = cabs(phase - sr_root(field, value));
        if (error < minimum) {
            minimum = error;
            selected = value;
        }
    }
    if (minimum > stats->maximum_final_root_error) {
        stats->maximum_final_root_error = minimum;
    }
    if (minimum > SR_ROOT_TOLERANCE) {
        sr_fail("final symplectic signature coefficient is not a root");
    }
    ++stats->final_coefficient_decodes;
    return selected;
}

static uint64_t sr_hash_byte(uint64_t hash, uint8_t value) {
    hash ^= (uint64_t)value;
    hash *= UINT64_C(1099511628211);
    return hash;
}

static struct sr_boundary sr_project(
    const struct sr_carrier *carrier,
    struct sr_stats *stats
) {
    struct sr_boundary boundary = {
        .combined_hash = UINT64_C(14695981039346656037)
    };
    for (size_t grade = 0U; grade < SR_GRADES; ++grade) {
        const struct sr_block *block = &carrier->block[grade];
        struct sr_grade_boundary *projected = &boundary.grade[grade];
        projected->degree = block->degree;
        projected->coefficient_cells = block->count;
        for (size_t field = 0U; field < SR_PRIMES; ++field) {
            uint64_t hash = UINT64_C(14695981039346656037);
            for (size_t cell = 0U; cell < block->count; ++cell) {
                for (
                    size_t variable = 0U;
                    variable < SR_VARIABLES;
                    ++variable
                ) {
                    hash = sr_hash_byte(
                        hash, block->monomial[cell].exponent[variable]
                    );
                }
                const int value = sr_decode(
                    sr_relative(block, field, cell), field, stats
                );
                hash = sr_hash_byte(hash, (uint8_t)value);
                if (value != 0) {
                    ++projected->nonzero[field];
                }
            }
            projected->hash[field] = hash;
            for (size_t byte = 0U; byte < sizeof(hash); ++byte) {
                boundary.combined_hash = sr_hash_byte(
                    boundary.combined_hash,
                    (uint8_t)(hash >> (8U * byte))
                );
            }
        }
    }
    return boundary;
}

static double sr_restoration(const struct sr_carrier *carrier) {
    double maximum = 0.0;
    for (size_t grade = 0U; grade < SR_GRADES; ++grade) {
        const struct sr_block *block = &carrier->block[grade];
        for (size_t field = 0U; field < SR_PRIMES; ++field) {
            for (size_t cell = 0U; cell < block->count; ++cell) {
                const double error = cabs(
                    block->working[field][cell]
                    - block->baseline[field][cell]
                );
                if (error > maximum) {
                    maximum = error;
                }
            }
        }
    }
    return maximum;
}

static uint64_t sr_restoration_nonidentity(
    const struct sr_carrier *carrier
) {
    uint64_t count = 0U;
    for (size_t grade = 0U; grade < SR_GRADES; ++grade) {
        const struct sr_block *block = &carrier->block[grade];
        for (size_t field = 0U; field < SR_PRIMES; ++field) {
            for (size_t cell = 0U; cell < block->count; ++cell) {
                const double complex phase =
                    sr_relative(block, field, cell);
                int selected = 0;
                double minimum = HUGE_VAL;
                for (
                    int value = 0;
                    value < SR_FIELD[field];
                    ++value
                ) {
                    const double error =
                        cabs(phase - sr_root(field, value));
                    if (error < minimum) {
                        minimum = error;
                        selected = value;
                    }
                }
                if (selected != 0) {
                    ++count;
                }
            }
        }
    }
    return count;
}

static void sr_snapshot_reload(struct sr_carrier *carrier) {
    for (size_t grade = 0U; grade < SR_GRADES; ++grade) {
        struct sr_block *block = &carrier->block[grade];
        for (size_t field = 0U; field < SR_PRIMES; ++field) {
            for (size_t cell = 0U; cell < block->count; ++cell) {
                block->working[field][cell] =
                    block->baseline[field][cell];
            }
        }
    }
}

static struct sr_execution sr_execute_on(
    struct sr_carrier *carrier,
    enum sr_program program,
    enum sr_mode mode
) {
    struct sr_execution execution = {0};
    const struct sr_compiled_generator seed =
        sr_compile_generator(program, 1);
    const struct sr_compiled_generator generator =
        sr_compile_generator(program, 0);
    sr_seal_seed(carrier, &seed, 0, &execution.stats);
    for (size_t grade = 0U; grade + 1U < SR_GRADES; ++grade) {
        sr_accumulate_bracket(
            carrier, grade, &generator, 0, 0, &execution.stats
        );
    }
    sr_observe(carrier, &execution.stats);
    execution.boundary = sr_project(carrier, &execution.stats);
    if (mode == SR_SNAPSHOT) {
        sr_snapshot_reload(carrier);
        execution.snapshot_loaded = 1;
    } else if (mode == SR_REORDERED_INVERSE) {
        sr_accumulate_bracket(
            carrier, 0U, &generator, 1, 0, &execution.stats
        );
        for (
            size_t grade = SR_GRADES - 1U;
            grade > 1U;
            --grade
        ) {
            sr_accumulate_bracket(
                carrier,
                grade - 1U,
                &generator,
                1,
                0,
                &execution.stats
            );
        }
        sr_seal_seed(carrier, &seed, 1, &execution.stats);
        execution.actual_inverse = 1;
    } else {
        for (
            size_t grade = SR_GRADES - 1U;
            grade > 0U;
            --grade
        ) {
            if (
                mode == SR_MISSING_INVERSE
                && grade == SR_GRADES - 1U
            ) {
                continue;
            }
            sr_accumulate_bracket(
                carrier,
                grade - 1U,
                &generator,
                1,
                mode == SR_WRONG_INVERSE
                    && grade == SR_GRADES - 1U,
                &execution.stats
            );
        }
        sr_seal_seed(carrier, &seed, 1, &execution.stats);
        execution.actual_inverse = 1;
    }
    execution.restoration_max_abs = sr_restoration(carrier);
    execution.restoration_nonidentity_cells =
        sr_restoration_nonidentity(carrier);
    if (
        !execution.snapshot_loaded
        && execution.actual_inverse
        && execution.restoration_max_abs <= SR_UNIT_TOLERANCE
        && execution.restoration_nonidentity_cells == 0U
    ) {
        execution.restoration_generation = 1U;
    } else {
        execution.actual_inverse = 0;
        execution.restoration_generation = 0U;
    }
    sr_observe(carrier, &execution.stats);
    return execution;
}

static struct sr_execution sr_execute_once(
    uint64_t identity,
    enum sr_program program,
    enum sr_mode mode
) {
    struct sr_carrier *carrier = sr_make_carrier(identity);
    const struct sr_execution execution =
        sr_execute_on(carrier, program, mode);
    free(carrier);
    return execution;
}

static void sr_print_hash(uint64_t hash) {
    printf("\"%016llx\"", (unsigned long long)hash);
}

static void sr_print_boundary(const struct sr_boundary *boundary) {
    printf("[");
    for (size_t grade = 0U; grade < SR_GRADES; ++grade) {
        if (grade != 0U) {
            printf(",");
        }
        const struct sr_grade_boundary *item = &boundary->grade[grade];
        printf(
            "{\"degree\":%zu,\"coefficient_cells\":%zu,"
            "\"nonzero_p17\":%zu,\"nonzero_p19\":%zu,"
            "\"hash_p17\":",
            item->degree,
            item->coefficient_cells,
            item->nonzero[0],
            item->nonzero[1]
        );
        sr_print_hash(item->hash[0]);
        printf(",\"hash_p19\":");
        sr_print_hash(item->hash[1]);
        printf("}");
    }
    printf("]");
}

static size_t sr_total_cells(void) {
    size_t cells = 0U;
    for (size_t grade = 0U; grade < SR_GRADES; ++grade) {
        const size_t degree = SR_INITIAL_DEGREE + 2U * grade;
        struct sr_monomial monomial[SR_MAX_COEFFICIENTS];
        cells += sr_fill_monomials(degree, monomial);
    }
    return cells * SR_PRIMES;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--project-shared-port") == 0) {
        sr_fail("resident shared wave port projection denied");
    }
    if (argc == 2 && strcmp(argv[1], "--null-carrier") == 0) {
        sr_fail("invalid symplectic relation carrier");
    }
    if (argc != 1) {
        sr_fail("unsupported symplectic relation request");
    }

    struct sr_carrier *shared_carrier =
        sr_make_carrier(UINT64_C(811));
    const struct sr_execution primary =
        sr_execute_on(shared_carrier, SR_PRIMARY, SR_CORRECT);
    const struct sr_execution replay =
        sr_execute_once(UINT64_C(811), SR_PRIMARY, SR_CORRECT);
    const struct sr_execution missing =
        sr_execute_once(
            UINT64_C(811), SR_PRIMARY, SR_MISSING_INVERSE
        );
    const struct sr_execution wrong =
        sr_execute_once(UINT64_C(811), SR_PRIMARY, SR_WRONG_INVERSE);
    const struct sr_execution reordered =
        sr_execute_once(
            UINT64_C(811), SR_PRIMARY, SR_REORDERED_INVERSE
        );
    const struct sr_execution snapshot =
        sr_execute_once(UINT64_C(811), SR_PRIMARY, SR_SNAPSHOT);
    const struct sr_execution identity =
        sr_execute_once(
            UINT64_C(913), SR_IDENTITY_MIXER, SR_CORRECT
        );
    const struct sr_execution zero =
        sr_execute_once(UINT64_C(977), SR_ZERO_KERR, SR_CORRECT);
    const struct sr_execution swapped =
        sr_execute_once(UINT64_C(991), SR_SWAPPED, SR_CORRECT);
    const struct sr_execution reuse =
        sr_execute_on(shared_carrier, SR_REUSE, SR_CORRECT);

    double maximum_reuse_restoration = reuse.restoration_max_abs;
    uint64_t reuse_generation = 1U;
    for (size_t cycle = 1U; cycle < SR_REUSE_CYCLES; ++cycle) {
        const enum sr_program program =
            cycle % 2U == 0U ? SR_REUSE : SR_PRIMARY;
        const struct sr_execution repeated =
            sr_execute_on(shared_carrier, program, SR_CORRECT);
        if (repeated.restoration_max_abs > maximum_reuse_restoration) {
            maximum_reuse_restoration =
                repeated.restoration_max_abs;
        }
        ++reuse_generation;
    }
    free(shared_carrier);

    int identity_higher_grades_zero = 1;
    int zero_all_grades_zero = 1;
    for (size_t grade = 0U; grade < SR_GRADES; ++grade) {
        if (
            grade > 0U
            && (
                identity.boundary.grade[grade].nonzero[0] != 0U
                || identity.boundary.grade[grade].nonzero[1] != 0U
            )
        ) {
            identity_higher_grades_zero = 0;
        }
        if (
            zero.boundary.grade[grade].nonzero[0] != 0U
            || zero.boundary.grade[grade].nonzero[1] != 0U
        ) {
            zero_all_grades_zero = 0;
        }
    }
    const int replay_equal =
        primary.boundary.combined_hash
        == replay.boundary.combined_hash;
    const int swap_discriminated =
        primary.boundary.grade[1].hash[0]
            != swapped.boundary.grade[1].hash[0]
        && primary.boundary.grade[1].hash[1]
            != swapped.boundary.grade[1].hash[1];
    const int controls_pass =
        primary.restoration_max_abs <= SR_UNIT_TOLERANCE
        && missing.restoration_max_abs > 1.0e-6
        && wrong.restoration_max_abs > SR_UNIT_TOLERANCE
        && reordered.restoration_max_abs > 1.0e-6
        && primary.restoration_nonidentity_cells == 0U
        && missing.restoration_nonidentity_cells > 0U
        && wrong.restoration_nonidentity_cells > 0U
        && reordered.restoration_nonidentity_cells > 0U
        && snapshot.restoration_max_abs <= SR_UNIT_TOLERANCE
        && identity_higher_grades_zero
        && zero_all_grades_zero
        && swap_discriminated
        && replay_equal
        && primary.stats.hidden_coefficient_decodes == 0U
        && maximum_reuse_restoration <= SR_UNIT_TOLERANCE;
    if (!controls_pass) {
        sr_fail("symplectic relation control failed");
    }

    const size_t coefficient_cells = sr_total_cells();
    printf(
        "{\"claim\":\"BOUNDED_DUAL_PRIME_PHASE_RESIDENT_NONLINEAR_"
        "SYMPLECTIC_LIE_SIGNATURE_GROWTH_WITH_ACTUAL_"
        "RESTORATION_AND_REUSE\","
        "\"result\":\"PASS\","
        "\"port_type\":\"TWO_MODE_CANONICAL_WAVE_PORT\","
        "\"poisson_canonical_index_contraction\":true,"
        "\"shared_wave_port_elimination_established\":false,"
        "\"grades\":%u,"
        "\"maximum_degree\":%u,"
        "\"primary_grades\":",
        SR_GRADES,
        SR_MAX_DEGREE
    );
    sr_print_boundary(&primary.boundary);
    printf(",\"reuse_grades\":");
    sr_print_boundary(&reuse.boundary);
    printf(
        ",\"primary_combined_hash\":"
    );
    sr_print_hash(primary.boundary.combined_hash);
    printf(",\"reuse_combined_hash\":");
    sr_print_hash(reuse.boundary.combined_hash);
    printf(
        ",\"resident_dual_prime_coefficient_cells\":%zu,"
        "\"logical_packed_active_phase_payload_bytes\":%zu,"
        "\"logical_packed_cell_baseline_bytes\":%zu,"
        "\"actual_carrier_allocation_bytes\":%zu,"
        "\"borrowed_verification_copy_bytes\":0,"
        "\"retain_all_grade_blocks\":%u,"
        "\"retained_inverse_kernels\":0,"
        "\"native_phase_updates\":%llu,"
        "\"resident_phase_reads\":%llu,"
        "\"field_product_interpolations\":%llu,"
        "\"poisson_monomial_products\":%llu,"
        "\"hidden_coefficient_decodes\":%llu,"
        "\"final_coefficient_decodes\":%llu,"
        "\"maximum_unit_modulus_error\":%.17g,"
        "\"maximum_final_root_error\":%.17g,"
        "\"restoration_max_abs\":%.17g,"
        "\"maximum_reuse_restoration_error\":%.17g,"
        "\"successful_restoration_receipt\":%llu,"
        "\"same_carrier_reuse_transactions\":%llu,"
        "\"missing_inverse_residual\":%.17g,"
        "\"wrong_inverse_residual\":%.17g,"
        "\"reordered_inverse_residual\":%.17g,"
        "\"missing_inverse_modular_mismatch_cells\":%llu,"
        "\"wrong_inverse_modular_mismatch_cells\":%llu,"
        "\"reordered_inverse_modular_mismatch_cells\":%llu,"
        "\"snapshot_residual\":%.17g,"
        "\"snapshot_restoration_receipt\":%llu,"
        "\"identity_mixer_higher_grades_zero\":%s,"
        "\"zero_kerr_all_grades_zero\":%s,"
        "\"swapped_order_discriminated\":%s,"
        "\"deterministic_replay\":%s,"
        "\"actual_inverse_restoration\":true,"
        "\"actual_restored_carrier_reuse\":true,"
        "\"final_boundary_only_projection\":true,"
        "\"actual_resident_grade_consumed\":true,"
        "\"public_topology_inverse_rematerialization\":true,"
        "\"mathematically_exact_dual_prime_phase_algebra\":true,"
        "\"rational_oracle_required\":true,"
        "\"bounded_polynomial_lie_growth_obstruction\":true,"
        "\"unbounded_growth_proved\":false,"
        "\"fixed_rank_closure_established\":false,"
        "\"genuinely_distinct_phase_resource\":false,"
        "\"computational_advantage\":false,"
        "\"small_wall_crossed\":false,"
        "\"physical_waveform_execution\":false,"
        "\"terminal\":false}\n",
        coefficient_cells,
        coefficient_cells * sizeof(double complex),
        coefficient_cells * sizeof(double complex),
        sizeof(struct sr_carrier),
        SR_GRADES,
        (unsigned long long)primary.stats.native_phase_updates,
        (unsigned long long)primary.stats.resident_phase_reads,
        (unsigned long long)primary.stats.field_product_interpolations,
        (unsigned long long)primary.stats.poisson_monomial_products,
        (unsigned long long)primary.stats.hidden_coefficient_decodes,
        (unsigned long long)primary.stats.final_coefficient_decodes,
        primary.stats.maximum_unit_modulus_error,
        primary.stats.maximum_final_root_error,
        primary.restoration_max_abs,
        maximum_reuse_restoration,
        (unsigned long long)primary.restoration_generation,
        (unsigned long long)reuse_generation,
        missing.restoration_max_abs,
        wrong.restoration_max_abs,
        reordered.restoration_max_abs,
        (unsigned long long)missing.restoration_nonidentity_cells,
        (unsigned long long)wrong.restoration_nonidentity_cells,
        (unsigned long long)reordered.restoration_nonidentity_cells,
        snapshot.restoration_max_abs,
        (unsigned long long)snapshot.restoration_generation,
        identity_higher_grades_zero ? "true" : "false",
        zero_all_grades_zero ? "true" : "false",
        swap_discriminated ? "true" : "false",
        replay_equal ? "true" : "false"
    );
    return 0;
}
