#include "independent_window_runtime.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int Q_VALUES[INDEPENDENT_WINDOW_Q_COUNT] = {
    -1536, -1024, -512, 0, 512, 1024, 1536
};

uint32_t independent_window_positive_work(int q) {
    return (uint32_t)(INDEPENDENT_WINDOW_BASE_WORK + q);
}

uint32_t independent_window_negative_work(int q) {
    return (uint32_t)(INDEPENDENT_WINDOW_BASE_WORK - q);
}

uint32_t independent_window_permuted_line(uint32_t index) {
    return (INDEPENDENT_WINDOW_PERM_A * index + INDEPENDENT_WINDOW_PERM_B) &
           (INDEPENDENT_WINDOW_BANK_LINES - 1u);
}

static int valid_q(int q) {
    for (size_t index = 0u; index < INDEPENDENT_WINDOW_Q_COUNT; index++) {
        if (Q_VALUES[index] == q) {
            return 1;
        }
    }
    return 0;
}

static int expected_q0_role(int q, int repeat_index) {
    if (q != 0) {
        return INDEPENDENT_WINDOW_Q0_SIGNAL;
    }
    if (repeat_index == 0) {
        return INDEPENDENT_WINDOW_Q0_NULL_BUILD;
    }
    if (repeat_index == 1) {
        return INDEPENDENT_WINDOW_Q0_NULL_TEST;
    }
    return -1;
}

static int prefix_unique(uint32_t units) {
    unsigned char seen[INDEPENDENT_WINDOW_BANK_LINES];
    memset(seen, 0, sizeof(seen));
    for (uint32_t unit = 0u; unit < units; unit++) {
        uint32_t line = independent_window_permuted_line(unit);
        if (line >= INDEPENDENT_WINDOW_BANK_LINES || seen[line] != 0u) {
            return 0;
        }
        seen[line] = 1u;
    }
    return 1;
}

static int parse_trial_line(const char *line, IndependentWindowTrial *trial) {
    int consumed = 0;
    int matched = sscanf(
        line,
        "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d%n",
        &trial->replicate,
        &trial->pair_index,
        &trial->leg_index,
        &trial->trial_index,
        &trial->repeat_index,
        &trial->q,
        &trial->mapping,
        &trial->mapping_order_first,
        &trial->source_positive_first,
        &trial->positive_subcapture_first,
        &trial->q0_role,
        &consumed
    );
    if (matched != 11) {
        return 0;
    }
    while (line[consumed] == ' ' || line[consumed] == '\t' ||
           line[consumed] == '\r' || line[consumed] == '\n') {
        consumed++;
    }
    return line[consumed] == '\0';
}

static int validate_trial_shape(const IndependentWindowTrial *trial) {
    if (trial->replicate < 0 || trial->replicate >= INDEPENDENT_WINDOW_REPLICATES) {
        return 0;
    }
    if (trial->pair_index < 0 ||
        trial->pair_index >= INDEPENDENT_WINDOW_PAIR_COUNT_PER_REPLICATE ||
        trial->leg_index < 0 ||
        trial->leg_index >= INDEPENDENT_WINDOW_MAPPING_COUNT ||
        trial->trial_index < 0 ||
        trial->trial_index >= INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE) {
        return 0;
    }
    if (!valid_q(trial->q) ||
        trial->mapping < 0 ||
        trial->mapping >= INDEPENDENT_WINDOW_MAPPING_COUNT ||
        trial->mapping_order_first < 0 ||
        trial->mapping_order_first >= INDEPENDENT_WINDOW_MAPPING_COUNT) {
        return 0;
    }
    if (trial->source_positive_first < 0 || trial->source_positive_first > 1 ||
        trial->positive_subcapture_first < 0 || trial->positive_subcapture_first > 1) {
        return 0;
    }
    if (trial->q0_role != expected_q0_role(trial->q, trial->repeat_index)) {
        return 0;
    }
    if (independent_window_positive_work(trial->q) +
            independent_window_negative_work(trial->q) !=
        INDEPENDENT_WINDOW_TOTAL_WORK) {
        return 0;
    }
    if (INDEPENDENT_WINDOW_SOURCE_WORK_PER_SUBCAPTURE != INDEPENDENT_WINDOW_TOTAL_WORK) {
        return 0;
    }
    if (INDEPENDENT_WINDOW_SOURCE_WORK_PER_MAPPING_LEG !=
        2u * INDEPENDENT_WINDOW_SOURCE_WORK_PER_SUBCAPTURE) {
        return 0;
    }
    return 1;
}

int independent_window_validate_schedule_tsv(const char *path) {
    FILE *handle = fopen(path, "r");
    if (handle == NULL) {
        return -errno;
    }

    IndependentWindowTrial trials[INDEPENDENT_WINDOW_TOTAL_TRIALS];
    unsigned char trial_seen[INDEPENDENT_WINDOW_REPLICATES][INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE];
    unsigned char pair_mapping_mask[INDEPENDENT_WINDOW_REPLICATES][INDEPENDENT_WINDOW_PAIR_COUNT_PER_REPLICATE];
    int per_rep_count[INDEPENDENT_WINDOW_REPLICATES] = {0, 0};
    int q0_build_pairs[INDEPENDENT_WINDOW_REPLICATES] = {0, 0};
    int q0_test_pairs[INDEPENDENT_WINDOW_REPLICATES] = {0, 0};
    size_t count = 0u;
    char line[256];

    memset(trial_seen, 0, sizeof(trial_seen));
    memset(pair_mapping_mask, 0, sizeof(pair_mapping_mask));

    while (fgets(line, sizeof(line), handle) != NULL) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') {
            continue;
        }
        if (count >= INDEPENDENT_WINDOW_TOTAL_TRIALS) {
            fclose(handle);
            return -E2BIG;
        }
        if (!parse_trial_line(line, &trials[count])) {
            fclose(handle);
            return -EINVAL;
        }
        if (!validate_trial_shape(&trials[count])) {
            fclose(handle);
            return -ERANGE;
        }
        IndependentWindowTrial *trial = &trials[count];
        if (trial_seen[trial->replicate][trial->trial_index] != 0u) {
            fclose(handle);
            return -EEXIST;
        }
        trial_seen[trial->replicate][trial->trial_index] = 1u;
        pair_mapping_mask[trial->replicate][trial->pair_index] |=
            (unsigned char)(1u << (unsigned int)trial->mapping);
        per_rep_count[trial->replicate]++;
        if (trial->q == 0 && trial->leg_index == 0) {
            if (trial->q0_role == INDEPENDENT_WINDOW_Q0_NULL_BUILD) {
                q0_build_pairs[trial->replicate]++;
            } else if (trial->q0_role == INDEPENDENT_WINDOW_Q0_NULL_TEST) {
                q0_test_pairs[trial->replicate]++;
            }
        }
        count++;
    }
    fclose(handle);

    if (count != INDEPENDENT_WINDOW_TOTAL_TRIALS) {
        return -EMSGSIZE;
    }
    for (int rep = 0; rep < INDEPENDENT_WINDOW_REPLICATES; rep++) {
        if (per_rep_count[rep] != INDEPENDENT_WINDOW_TRIALS_PER_REPLICATE ||
            q0_build_pairs[rep] != 4 ||
            q0_test_pairs[rep] != 4) {
            return -EDOM;
        }
        for (int pair = 0; pair < INDEPENDENT_WINDOW_PAIR_COUNT_PER_REPLICATE; pair++) {
            if (pair_mapping_mask[rep][pair] != 0x03u) {
                return -EINVAL;
            }
        }
    }
    return 0;
}

static int runtime_self_test(void) {
    if (!prefix_unique(INDEPENDENT_WINDOW_BANK_LINES)) {
        return 1;
    }
    if (independent_window_positive_work(1536) != 3584u ||
        independent_window_negative_work(1536) != 512u ||
        independent_window_positive_work(-1536) != 512u ||
        independent_window_negative_work(-1536) != 3584u) {
        return 1;
    }
    if (INDEPENDENT_WINDOW_TOTAL_TRIALS * 2 !=
        INDEPENDENT_WINDOW_TOTAL_COMPONENT_WINDOWS) {
        return 1;
    }
    if (INDEPENDENT_WINDOW_SOURCE_CORE != 4 ||
        INDEPENDENT_WINDOW_RECEIVER_CORE != 5 ||
        INDEPENDENT_WINDOW_PERM_A != 257u ||
        INDEPENDENT_WINDOW_PERM_B != 43u) {
        return 1;
    }
    return 0;
}

static void print_contract_summary(void) {
    printf(
        "{\"schema_id\":\"%s\","
        "\"run_id\":\"%s\","
        "\"bank_lines\":%u,"
        "\"line_bytes\":%u,"
        "\"source_core\":%d,"
        "\"receiver_core\":%d,"
        "\"mapping_leg_source_work\":%u,"
        "\"component_windows\":%d}\n",
        INDEPENDENT_WINDOW_SCHEMA_RUNTIME_SUMMARY,
        INDEPENDENT_WINDOW_RUN_ID,
        INDEPENDENT_WINDOW_BANK_LINES,
        INDEPENDENT_WINDOW_LINE_BYTES,
        INDEPENDENT_WINDOW_SOURCE_CORE,
        INDEPENDENT_WINDOW_RECEIVER_CORE,
        INDEPENDENT_WINDOW_SOURCE_WORK_PER_MAPPING_LEG,
        INDEPENDENT_WINDOW_TOTAL_COMPONENT_WINDOWS
    );
}

static void usage(const char *program) {
    fprintf(
        stderr,
        "usage: %s --self-test | --contract-summary | --validate-schedule-tsv <path>\n",
        program
    );
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        if (runtime_self_test() != 0) {
            fprintf(stderr, "INDEPENDENT_WINDOW_V3_RUNTIME_SELF_TEST_FAILED\n");
            return 1;
        }
        printf("INDEPENDENT_WINDOW_V3_RUNTIME_SELF_TEST_OK\n");
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--contract-summary") == 0) {
        print_contract_summary();
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "--validate-schedule-tsv") == 0) {
        int rc = independent_window_validate_schedule_tsv(argv[2]);
        if (rc != 0) {
            fprintf(stderr, "INDEPENDENT_WINDOW_V3_SCHEDULE_TSV_INVALID rc=%d\n", rc);
            return 1;
        }
        printf("INDEPENDENT_WINDOW_V3_SCHEDULE_TSV_OK\n");
        return 0;
    }
    usage(argv[0]);
    return 2;
}
