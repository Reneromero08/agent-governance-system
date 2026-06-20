#ifndef COMBINED_PDN_HARDWARE_H
#define COMBINED_PDN_HARDWARE_H
#include <stddef.h>
#define CP_PATH_MAX 4096
enum {MODE_VALIDATE=1,MODE_HARDWARE=2};enum {BACKEND_REAL=0,BACKEND_MOCK=1};
typedef struct {const char*session_dir,*output_dir,*executor_commit;int victim,sender;long pin_khz,read_hz;double slot_s,off_window_s,temp_veto_c;int mode,backend;} RunnerArgs;
typedef struct {long window_index;char session_id[128],stage[48],block_id[96],family[48],actual_mode[32],declared_mode[32],executed_tone_order[16],declared_tone_order[16],measurement_mode[32];int physical_tone_index,codeword_source_index,drive_on,sender_off_required,amplitude_level,theta_idx;} Window;
typedef struct {char session_id[128],route[16],campaign_source_commit[65],campaign_plan_sha256[65],session_manifest_sha256[65];size_t count;Window*windows;} Schedule;
int run_hardware(const RunnerArgs*,const Schedule*);
int write_run_manifest(const char*,const char*,const char*);
#endif
