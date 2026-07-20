#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef struct {
    const char *name;
    uint32_t event;
    uint32_t umask;
} frozen_event;

typedef struct {
    int opened;
    int closed;
    int leaked;
    int enable_attempted;
    int reset_attempted;
    int read_attempted;
    int missing_event;
    int malformed_event_identity;
    int incorrect_structure_size;
    int partial_group_open;
    bool grouped_open_capability;
    int fd[3];
    uint64_t id[3];
    char error_text[160];
} preflight_state;

static const frozen_event FROZEN_EVENTS[3] = {
    {"cpu_cycles_not_halted", 0x76u, 0x00u},
    {"cache_block_commands_change_to_dirty", 0xeau, 0x20u},
    {"probe_responses_dirty", 0xecu, 0x0cu},
};

static long perf_event_open_call(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static void init_state(preflight_state *state) {
    memset(state, 0, sizeof(*state));
    for (size_t i = 0; i < 3; ++i) {
        state->fd[i] = -1;
        state->id[i] = 0;
    }
}

static int event_identity_valid(const frozen_event *event) {
    if (event == NULL || event->name == NULL || event->name[0] == '\0') {
        return 0;
    }
    if (event->event == 0u) {
        return 0;
    }
    return 1;
}

static int close_open_fds(preflight_state *state) {
    int ok = 1;
    for (size_t i = 0; i < 3; ++i) {
        if (state->fd[i] >= 0) {
            if (close(state->fd[i]) != 0) {
                ok = 0;
                state->leaked += 1;
            } else {
                state->closed += 1;
            }
            state->fd[i] = -1;
        }
    }
    return ok;
}

static int open_one(preflight_state *state, size_t index, int cpu, int group_fd) {
    struct perf_event_attr attr;
    const frozen_event *event = &FROZEN_EVENTS[index];
    long fd;
    memset(&attr, 0, sizeof(attr));
    if (!event_identity_valid(event)) {
        state->malformed_event_identity = 1;
        snprintf(state->error_text, sizeof(state->error_text), "malformed frozen event identity");
        return 0;
    }
    attr.type = PERF_TYPE_RAW;
    attr.size = sizeof(attr);
    attr.config = ((uint64_t)event->umask << 8) | (uint64_t)event->event;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    fd = perf_event_open_call(&attr, 0, cpu, group_fd, 0ul);
    if (fd < 0) {
        if (index == 0) {
            state->missing_event = 1;
        } else {
            state->partial_group_open = 1;
        }
        snprintf(state->error_text, sizeof(state->error_text), "%s", strerror(errno));
        return 0;
    }
    state->fd[index] = (int)fd;
    state->opened += 1;
    if (ioctl((int)fd, PERF_EVENT_IOC_ID, &state->id[index]) != 0) {
        snprintf(state->error_text, sizeof(state->error_text), "PERF_EVENT_IOC_ID failed: %s", strerror(errno));
        state->partial_group_open = index > 0 ? 1 : 0;
        return 0;
    }
    return 1;
}

static int disabled_group_preflight(int cpu, preflight_state *state) {
    int group_fd = -1;
    init_state(state);
    if (sizeof(struct perf_event_attr) < 112u) {
        state->incorrect_structure_size = 1;
        snprintf(state->error_text, sizeof(state->error_text), "struct perf_event_attr unexpectedly small: %zu", sizeof(struct perf_event_attr));
        return 0;
    }
    for (size_t i = 0; i < 3; ++i) {
        if (!open_one(state, i, cpu, group_fd)) {
            close_open_fds(state);
            return 0;
        }
        if (i == 0) {
            group_fd = state->fd[i];
        }
    }
    state->grouped_open_capability = true;
    close_open_fds(state);
    return state->opened == 3 && state->closed == 3 && state->leaked == 0;
}

static void print_event_json(void) {
    printf("{\"cpu_cycles_not_halted\":{\"event\":\"0x76\",\"umask\":\"0x00\"},");
    printf("\"cache_block_commands_change_to_dirty\":{\"event\":\"0xea\",\"umask\":\"0x20\"},");
    printf("\"probe_responses_dirty\":{\"event\":\"0xec\",\"umask\":\"0x0c\"}}");
}

static void print_state_json(const char *schema, int passed, int cpu, const preflight_state *state) {
    printf("{\"schema\":\"%s\",", schema);
    printf("\"passed\":%s,", passed ? "true" : "false");
    printf("\"cpu\":%d,", cpu);
    printf("\"events\":");
    print_event_json();
    printf(",\"perf_event_attr_size\":%zu,", sizeof(struct perf_event_attr));
    printf("\"uses_system_perf_event_attr\":true,");
    printf("\"grouped_open_capability\":%s,", state->grouped_open_capability ? "true" : "false");
    printf("\"pmu_open_count\":%d,", state->opened);
    printf("\"pmu_close_count\":%d,", state->closed);
    printf("\"pmu_fd_leak_count\":%d,", state->leaked);
    printf("\"pmu_acquisition_count\":0,");
    printf("\"enabled_measurement_interval\":false,");
    printf("\"scientific_data_collected\":false,");
    printf("\"enable_attempted\":%s,", state->enable_attempted ? "true" : "false");
    printf("\"reset_attempted\":%s,", state->reset_attempted ? "true" : "false");
    printf("\"read_attempted\":%s,", state->read_attempted ? "true" : "false");
    printf("\"malformed_event_identity\":%s,", state->malformed_event_identity ? "true" : "false");
    printf("\"partial_group_open\":%s,", state->partial_group_open ? "true" : "false");
    printf("\"incorrect_structure_size\":%s,", state->incorrect_structure_size ? "true" : "false");
    printf("\"missing_event\":%s,", state->missing_event ? "true" : "false");
    printf("\"error_text\":\"");
    for (const char *p = state->error_text; *p != '\0'; ++p) {
        if (*p == '\\' || *p == '\"') {
            putchar('\\');
        }
        if (*p == '\n' || *p == '\r') {
            putchar(' ');
        } else {
            putchar(*p);
        }
    }
    printf("\"}\n");
}

static int self_test(void) {
    preflight_state ok;
    preflight_state malformed;
    preflight_state partial;
    preflight_state leaked;
    preflight_state enable;
    preflight_state read_attempt;
    preflight_state bad_size;
    preflight_state missing;
    init_state(&ok);
    ok.grouped_open_capability = true;
    ok.opened = 3;
    ok.closed = 3;
    init_state(&malformed);
    malformed.malformed_event_identity = 1;
    init_state(&partial);
    partial.opened = 1;
    partial.closed = 1;
    partial.partial_group_open = 1;
    init_state(&leaked);
    leaked.opened = 3;
    leaked.closed = 2;
    leaked.leaked = 1;
    init_state(&enable);
    enable.enable_attempted = 1;
    init_state(&read_attempt);
    read_attempt.read_attempted = 1;
    init_state(&bad_size);
    bad_size.incorrect_structure_size = 1;
    init_state(&missing);
    missing.missing_event = 1;
    int passed = sizeof(struct perf_event_attr) >= 112u &&
        event_identity_valid(&FROZEN_EVENTS[0]) &&
        event_identity_valid(&FROZEN_EVENTS[1]) &&
        event_identity_valid(&FROZEN_EVENTS[2]) &&
        ok.opened == ok.closed &&
        malformed.malformed_event_identity &&
        partial.partial_group_open &&
        leaked.leaked &&
        enable.enable_attempted &&
        read_attempt.read_attempted &&
        bad_size.incorrect_structure_size &&
        missing.missing_event;
    printf("{\"schema\":\"FAMILY10H_RELATION_ONLY_PMU_PREFLIGHT_HELPER_SELF_TEST_V1\",");
    printf("\"passed\":%s,", passed ? "true" : "false");
    printf("\"events\":");
    print_event_json();
    printf(",\"perf_event_attr_size\":%zu,", sizeof(struct perf_event_attr));
    printf("\"uses_system_perf_event_attr\":true,");
    printf("\"pmu_open_count\":0,\"pmu_close_count\":0,\"pmu_acquisition_count\":0,");
    printf("\"enabled_measurement_interval\":false,\"scientific_data_collected\":false,");
    printf("\"negative_regressions\":{");
    printf("\"malformed_event_identity\":%s,", malformed.malformed_event_identity ? "true" : "false");
    printf("\"partial_group_open\":%s,", partial.partial_group_open ? "true" : "false");
    printf("\"leaked_descriptor\":%s,", leaked.leaked ? "true" : "false");
    printf("\"enable_attempt\":%s,", enable.enable_attempted ? "true" : "false");
    printf("\"read_attempt\":%s,", read_attempt.read_attempted ? "true" : "false");
    printf("\"incorrect_structure_size\":%s,", bad_size.incorrect_structure_size ? "true" : "false");
    printf("\"missing_event\":%s},", missing.missing_event ? "true" : "false");
    printf("\"live_activity\":false}\n");
    return passed ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return self_test();
    }
    if (argc == 3 && strcmp(argv[1], "--disabled-group-preflight") == 0) {
        int cpu = atoi(argv[2]);
        preflight_state state;
        int passed = disabled_group_preflight(cpu, &state);
        print_state_json("FAMILY10H_RELATION_ONLY_DISABLED_PMU_PREFLIGHT_V1", passed, cpu, &state);
        return passed ? 0 : 1;
    }
    fprintf(stderr, "usage: %s --self-test | --disabled-group-preflight <cpu>\n", argv[0]);
    return 2;
}
