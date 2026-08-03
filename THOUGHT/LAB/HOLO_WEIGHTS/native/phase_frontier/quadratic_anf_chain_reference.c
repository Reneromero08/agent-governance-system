#define _POSIX_C_SOURCE 200809L

/*
 * Independent bounded adjudicator for the fixed quadratic ANF chain.
 *
 * This executable is not linked into the native phase process. It computes
 * the symbolic monic-substitution coefficients and streams the 32 boundary
 * valuations with four hidden-port probes. It stores no extensional relation.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QR_RELATIONS 3U
#define QR_COEFFICIENTS 3U
#define QR_MAX_SOURCE_BYTES 4096U

struct qr_program {
    unsigned char coefficient[QR_RELATIONS][QR_COEFFICIENTS];
    uint64_t source_fnv1a64;
};

static void qr_fail(const char *message) {
    fprintf(stderr, "quadratic ANF reference error: %s\n", message);
    exit(2);
}

static uint64_t qr_hash_bytes(
    uint64_t hash,
    const unsigned char *bytes,
    size_t count
) {
    for (size_t index = 0U; index < count; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static char *qr_trim(char *text) {
    while (*text == ' ' || *text == '\t') {
        ++text;
    }
    size_t length = strlen(text);
    while (
        length > 0U
        && (text[length - 1U] == ' ' || text[length - 1U] == '\t')
    ) {
        text[--length] = '\0';
    }
    return text;
}

static size_t qr_split(
    char *line,
    char *token[],
    size_t capacity
) {
    size_t count = 0U;
    char *save = NULL;
    for (
        char *item = strtok_r(line, " \t", &save);
        item != NULL;
        item = strtok_r(NULL, " \t", &save)
    ) {
        if (count == capacity) {
            qr_fail("too many tokens");
        }
        token[count++] = item;
    }
    return count;
}

static unsigned char qr_bit(const char *text) {
    if (strcmp(text, "0") == 0) {
        return 0U;
    }
    if (strcmp(text, "1") == 0) {
        return 1U;
    }
    qr_fail("coefficient is not a canonical GF2 bit");
    return 0U;
}

static struct qr_program qr_read(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        qr_fail("cannot open public program");
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        fclose(file);
        qr_fail("cannot seek public program");
    }
    const long measured = ftell(file);
    if (measured < 0L || (unsigned long)measured > QR_MAX_SOURCE_BYTES) {
        fclose(file);
        qr_fail("public program size is invalid");
    }
    if (fseek(file, 0L, SEEK_SET) != 0) {
        fclose(file);
        qr_fail("cannot rewind public program");
    }
    const size_t length = (size_t)measured;
    char *source = calloc(length + 1U, 1U);
    if (source == NULL) {
        fclose(file);
        qr_fail("public program allocation failed");
    }
    if (fread(source, 1U, length, file) != length || ferror(file)) {
        free(source);
        fclose(file);
        qr_fail("cannot read public program");
    }
    if (fclose(file) != 0) {
        free(source);
        qr_fail("cannot close public program");
    }
    for (size_t index = 0U; index < length; ++index) {
        if (source[index] == '\0' || source[index] == '\r') {
            free(source);
            qr_fail("public program contains a forbidden byte");
        }
    }

    struct qr_program program;
    memset(&program, 0, sizeof(program));
    program.source_fnv1a64 = qr_hash_bytes(
        UINT64_C(14695981039346656037),
        (const unsigned char *)source,
        length
    );

    size_t state = 0U;
    char *line_save = NULL;
    for (
        char *raw = strtok_r(source, "\n", &line_save);
        raw != NULL;
        raw = strtok_r(NULL, "\n", &line_save)
    ) {
        char *line = qr_trim(raw);
        if (*line == '\0' || *line == '#') {
            continue;
        }
        char *token[5];
        const size_t count = qr_split(line, token, 5U);
        if (
            state == 0U
            && count == 2U
            && strcmp(token[0], "CATCAS_QUADRATIC_ANF_CHAIN") == 0
            && strcmp(token[1], "1") == 0
        ) {
            ++state;
            continue;
        }
        if (
            state == 1U
            && count == 2U
            && strcmp(token[0], "TYPE") == 0
            && strcmp(token[1], "BOOLEAN_ANF_GF2") == 0
        ) {
            ++state;
            continue;
        }
        if (state >= 2U && state <= 4U && count == 4U) {
            static const char *const name[QR_RELATIONS] = {
                "F", "G", "J"
            };
            const size_t relation = state - 2U;
            if (strcmp(token[0], name[relation]) != 0) {
                free(source);
                qr_fail("public relation order is invalid");
            }
            for (
                size_t coefficient = 0U;
                coefficient < QR_COEFFICIENTS;
                ++coefficient
            ) {
                program.coefficient[relation][coefficient] =
                    qr_bit(token[coefficient + 1U]);
            }
            if (program.coefficient[relation][0] != 1U) {
                free(source);
                qr_fail("public relation is not a monic definition");
            }
            ++state;
            continue;
        }
        if (
            state == 5U
            && count == 1U
            && strcmp(token[0], "END") == 0
        ) {
            ++state;
            continue;
        }
        free(source);
        qr_fail("public program record is invalid");
    }
    free(source);
    if (state != 6U) {
        qr_fail("public program is incomplete");
    }
    return program;
}

static unsigned char qr_f(
    const struct qr_program *program,
    unsigned char a,
    unsigned char b,
    unsigned char u
) {
    return (unsigned char)(
        u
        ^ program->coefficient[0][1]
        ^ (program->coefficient[0][2] & a & b)
    );
}

static unsigned char qr_g(
    const struct qr_program *program,
    unsigned char u,
    unsigned char c,
    unsigned char v
) {
    return (unsigned char)(
        v
        ^ program->coefficient[1][1]
        ^ (program->coefficient[1][2] & u & c)
    );
}

static unsigned char qr_j(
    const struct qr_program *program,
    unsigned char v,
    unsigned char e,
    unsigned char d
) {
    return (unsigned char)(
        d
        ^ program->coefficient[2][1]
        ^ (program->coefficient[2][2] & v & e)
    );
}

static void qr_boundary_coefficients(
    const struct qr_program *program,
    unsigned char output[5]
) {
    const unsigned char alpha = program->coefficient[0][1];
    const unsigned char beta = program->coefficient[0][2];
    const unsigned char gamma = program->coefficient[1][1];
    const unsigned char delta = program->coefficient[1][2];
    const unsigned char eta = program->coefficient[2][1];
    const unsigned char theta = program->coefficient[2][2];
    output[0] = 1U;
    output[1] = eta;
    output[2] = (unsigned char)(theta & gamma);
    output[3] = (unsigned char)(theta & delta & alpha);
    output[4] = (unsigned char)(theta & delta & beta);
}

static unsigned char qr_boundary_value(
    const unsigned char coefficient[5],
    unsigned char a,
    unsigned char b,
    unsigned char c,
    unsigned char e,
    unsigned char d
) {
    return (unsigned char)(
        (coefficient[0] & d)
        ^ coefficient[1]
        ^ (coefficient[2] & e)
        ^ (coefficient[3] & c & e)
        ^ (coefficient[4] & a & b & c & e)
    );
}

int main(int argc, char **argv) {
    if (argc != 2) {
        qr_fail("usage: reference PROGRAM.qanf");
    }
    const struct qr_program program = qr_read(argv[1]);
    unsigned char boundary[5];
    qr_boundary_coefficients(&program, boundary);

    size_t external_rows = 0U;
    size_t accepted_rows = 0U;
    size_t hidden_probes = 0U;
    size_t exact_rows = 0U;
    size_t unique_hidden_rows = 0U;
    for (unsigned char a = 0U; a < 2U; ++a) {
        for (unsigned char b = 0U; b < 2U; ++b) {
            for (unsigned char c = 0U; c < 2U; ++c) {
                for (unsigned char e = 0U; e < 2U; ++e) {
                    for (unsigned char d = 0U; d < 2U; ++d) {
                        size_t hidden_solutions = 0U;
                        for (unsigned char u = 0U; u < 2U; ++u) {
                            for (unsigned char v = 0U; v < 2U; ++v) {
                                ++hidden_probes;
                                if (
                                    qr_f(&program, a, b, u) == 0U
                                    && qr_g(&program, u, c, v) == 0U
                                    && qr_j(&program, v, e, d) == 0U
                                ) {
                                    ++hidden_solutions;
                                }
                            }
                        }
                        const int composed = hidden_solutions != 0U;
                        const int symbolic = qr_boundary_value(
                            boundary, a, b, c, e, d
                        ) == 0U;
                        ++external_rows;
                        if (composed) {
                            ++accepted_rows;
                        }
                        if (composed == symbolic) {
                            ++exact_rows;
                        }
                        if (hidden_solutions == 1U) {
                            ++unique_hidden_rows;
                        }
                    }
                }
            }
        }
    }
    if (
        external_rows != 32U
        || accepted_rows != 16U
        || exact_rows != external_rows
        || unique_hidden_rows != accepted_rows
        || hidden_probes != 128U
    ) {
        qr_fail("symbolic substitution does not match bounded semantics");
    }

    printf(
        "{\"result\":\"PASS\","
        "\"source_fnv1a64\":\"%016llx\","
        "\"boundary_coefficients\":[%u,%u,%u,%u,%u],"
        "\"boundary_relation_rows\":%zu,"
        "\"accepted_boundary_rows\":%zu,"
        "\"hidden_port_probes\":%zu,"
        "\"symbolic_rows_exact\":%zu,"
        "\"accepted_rows_with_unique_hidden_state\":%zu,"
        "\"ce_anf_coefficient\":%u,"
        "\"fourth_boolean_derivative\":%u,"
        "\"gf2_affine\":%s,"
        "\"extensional_storage_cells\":0}\n",
        (unsigned long long)program.source_fnv1a64,
        boundary[0],
        boundary[1],
        boundary[2],
        boundary[3],
        boundary[4],
        external_rows,
        accepted_rows,
        hidden_probes,
        exact_rows,
        unique_hidden_rows,
        boundary[3],
        boundary[4],
        boundary[3] == 0U && boundary[4] == 0U ? "true" : "false"
    );
    return 0;
}
