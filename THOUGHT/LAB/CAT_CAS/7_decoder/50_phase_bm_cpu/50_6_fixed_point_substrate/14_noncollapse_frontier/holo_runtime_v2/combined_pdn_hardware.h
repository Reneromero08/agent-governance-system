#ifndef COMBINED_PDN_HARDWARE_H
#define COMBINED_PDN_HARDWARE_H

#include <stddef.h>

#define CP_PATH_MAX 4096

enum { MODE_VALIDATE = 1, MODE_HARDWARE = 2 };
enum { BACKEND_REAL = 0, BACKEND_MOCK = 1 };

typedef struct {
    const char *session_dir;
    const char *output_dir;
    const char *executor_commit;
    const char *source_bundle_sha256;
    const char *authorization_artifact;
    int engineering_smoke;
    int victim;
    int sender;
    long pin_khz;
    long read_hz;
    double slot_s;
    double off_window_s;
    double temp_veto_c;
    int mode;
    int backend;
} RunnerArgs;

typedef struct {
    long window_index;
    char session_id[128];
    char stage[48];
    char block_id[96];
    char family[48];
    char actual_mode[32];
    char declared_mode[32];
    char executed_tone_order[16];
    char declared_tone_order[16];
    char measurement_mode[32];
    int physical_tone_index;
    int receiver_codeword_source_index;
    int sender_codeword_source_index;
    int drive_on;
    int sender_off_required;
    int amplitude_level;
    int receiver_theta_idx;
    int sender_theta_idx;
    int shared_schedule;
    char scramble_key_digest[65];
} Window;

typedef struct {
    char session_id[128];
    char route[16];
    char partition[96];
    char campaign_source_commit[65];
    char campaign_plan_sha256[65];
    char session_manifest_sha256[65];
    size_t count;
    Window *windows;
} Schedule;

int run_hardware(const RunnerArgs *, const Schedule *);
int write_run_manifest(const char *, const char *, const char *);

#endif
