#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *expected_sequence[16] = {
    "I", "I", "I", "I", "C0", "D0", "S0E", "S0E",
    "S0E", "S0E", "O0", "O0", "A0P", "A0N", "T", "T"
};

static bool string_equal(const char *left, const char *right) {
    return strcmp(left, right) == 0;
}

static int validate_schedule_semantics(void) {
    unsigned int step_epoch_slots = 0U;
    for (size_t i = 0; i < 16U; ++i) {
        const char *slot = expected_sequence[i];
        if (string_equal(slot, "I") || string_equal(slot, "C0") || string_equal(slot, "D0") ||
            string_equal(slot, "O0") || string_equal(slot, "T")) {
            continue;
        }
        if (string_equal(slot, "S0E")) {
            ++step_epoch_slots;
            continue;
        }
        if (string_equal(slot, "A0P") || string_equal(slot, "A0N")) {
            continue;
        }
        return 1;
    }
    return step_epoch_slots == 4U ? 0 : 1;
}

static int validate_only(void) {
    if (validate_schedule_semantics() != 0) {
        fputs("{\"status\":\"GATE_A_WORKER_VALIDATE_ONLY_FAILED\"}\n", stderr);
        return 1;
    }
    puts("{"
         "\"automatic_retry\":false,"
         "\"control_writes\":0,"
         "\"hardware_executions\":0,"
         "\"msr_reads\":0,"
         "\"msr_writes\":0,"
         "\"receiver_captures\":0,"
         "\"sender_starts\":0,"
         "\"slot_count\":16,"
         "\"status\":\"GATE_A_WORKER_VALIDATE_ONLY_OK\","
         "\"validate_only\":true"
         "}");
    return 0;
}

static int probe_only(void) {
    fputs("probe-only is read-only source support and is not run in this phase\n", stderr);
    return 2;
}

static int execute_authorized(int argc, char **argv) {
    bool has_authority = false;
    bool has_schedule = false;
    for (int i = 2; i < argc; ++i) {
        if (string_equal(argv[i], "--authority-sha256")) {
            has_authority = true;
        } else if (string_equal(argv[i], "--schedule-sha256")) {
            has_schedule = true;
        }
    }
    if (!has_authority || !has_schedule) {
        fputs("execute-authorized requires exact authority and schedule identity\n", stderr);
        return 2;
    }
    fputs("live execution is not available in this qualification phase\n", stderr);
    return 3;
}

int main(int argc, char **argv) {
    if (argc == 2 && string_equal(argv[1], "--validate-only")) {
        return validate_only();
    }
    if (argc == 2 && string_equal(argv[1], "--probe-only")) {
        return probe_only();
    }
    if (argc >= 2 && string_equal(argv[1], "--execute-authorized")) {
        return execute_authorized(argc, argv);
    }
    fputs("usage: gate_a_worker --validate-only\n", stderr);
    return 2;
}
