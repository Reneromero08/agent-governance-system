#define _GNU_SOURCE

#ifndef QUALIFIED_V2_SOURCE_PATH
#define QUALIFIED_V2_SOURCE_PATH "../holo_runtime_v2/combined_pdn_hardware.c"
#endif

#define main phase6b6_qualified_v2_main_disabled
#include QUALIFIED_V2_SOURCE_PATH
#undef main

static void emit_int_row(const char *mode, int mode_index_value) {
    printf("{\"mode\":\"%s\",\"row\":[", mode);
    for (int source = 0; source < 12; source++) {
        if (source) {
            putchar(',');
        }
        printf("%d", code_sign(mode_index_value, source));
    }
    printf("]}");
}

static int emit_reference_table(const char *source_sha256) {
    const char *modes[4] = {"basis", "rotation", "residual", "mini"};

    for (int mode = 0; mode < 4; mode++) {
        if (mode_index(modes[mode]) != mode) {
            fprintf(stderr, "qualified V2 mode mapping changed at index %d\n", mode);
            return 2;
        }
    }

    printf("{");
    printf("\"schema_id\":\"CAT_CAS_PHASE6B6_C_REFERENCE_TABLE_V1\",");
    printf("\"format_version\":1,");
    printf("\"qualified_source_sha256\":\"%s\",", source_sha256);
    printf("\"tone_count\":12,");
    printf("\"mode_count\":4,");
    printf("\"mode_names\":[");
    for (int mode = 0; mode < 4; mode++) {
        if (mode) {
            putchar(',');
        }
        printf("\"%s\"", modes[mode]);
    }
    printf("],");
    printf("\"mode_to_codeword_mapping\":{");
    for (int mode = 0; mode < 4; mode++) {
        if (mode) {
            putchar(',');
        }
        printf("\"%s\":%d", modes[mode], mode);
    }
    printf("},");
    printf("\"tones\":[");
    for (int index = 0; index < 12; index++) {
        if (index) {
            putchar(',');
        }
        printf("{\"physical_tone_index\":%d,\"frequency_hz\":%.17g,\"codeword_source_index\":%d}",
               index, tone(index), index);
    }
    printf("],");
    printf("\"codebook\":{");
    for (int mode = 0; mode < 4; mode++) {
        if (mode) {
            putchar(',');
        }
        printf("\"%s\":[", modes[mode]);
        for (int source = 0; source < 12; source++) {
            if (source) {
                putchar(',');
            }
            printf("%d", code_sign(mode, source));
        }
        printf("]");
    }
    printf("},");
    printf("\"codebook_rows\":[");
    for (int mode = 0; mode < 4; mode++) {
        if (mode) {
            putchar(',');
        }
        emit_int_row(modes[mode], mode);
    }
    printf("]}");
    putchar('\n');
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <qualified-source-sha256>\n", argv[0]);
        return 2;
    }
    if (strlen(argv[1]) != 64) {
        fprintf(stderr, "qualified source SHA-256 must be 64 hex characters\n");
        return 2;
    }
    return emit_reference_table(argv[1]);
}
