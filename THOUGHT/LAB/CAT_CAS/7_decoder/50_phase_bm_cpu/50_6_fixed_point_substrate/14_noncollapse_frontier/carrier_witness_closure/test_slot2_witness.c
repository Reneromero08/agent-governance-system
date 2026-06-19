#define _GNU_SOURCE
#define main slot2_program_main
#include "../../10_cross_core_wormhole/slot2_pdn/slot2_pdn_lockin.c"
#undef main

#include <assert.h>

static void remove_file(const char *directory, const char *name) {
    char path[512];
    snprintf(path,sizeof(path),"%s/%s",directory,name);
    unlink(path);
}

int main(void) {
    config_t cfg;
    char *argv[] = {"slot2","--mode","matrix","--witness-dir","/tmp/new-run",
                    "--run-id","run_1","--campaign-id","campaign_1",
                    "--condition","silent","--trials","4"};
    config_defaults(&cfg);
    assert(parse_args(&cfg,(int)(sizeof(argv)/sizeof(argv[0])),argv)==0);
    assert(cfg.witness_enabled==1 && cfg.ctrl_silent==1 && !strcmp(cfg.condition,"silent"));

    codebook_t cb;
    make_codebook(&cb,cfg.nbin,7);
    symbol_t *symbols=NULL; int npre=0;
    int total=build_schedule(&cfg,&symbols,&npre);
    assert(total==MODES+3*cfg.trials);
    int sign; double phase;
    effective_drive(&cfg,&cb,&symbols[0],0,0,&sign,&phase);
    assert(sign==0);

    cfg.ctrl_silent=0; cfg.ctrl_scramble=1;
    int changed=0;
    for(int b=0;b<cfg.nbin;b++){
        effective_drive(&cfg,&cb,&symbols[0],0,b,&sign,&phase);
        if(sign!=(int)symbol_bin_sign(&cb,&symbols[0],b)) changed=1;
    }
    assert(changed);

    char parent_template[]="/tmp/slot2-witness-test-XXXXXX";
    char *parent=mkdtemp(parent_template);
    assert(parent!=NULL);
    snprintf(cfg.witness_dir,sizeof(cfg.witness_dir),"%s/run_1",parent);
    cfg.ctrl_scramble=0; snprintf(cfg.condition,sizeof(cfg.condition),"%s","matrix");
    assert(prepare_witness_directory(&cfg,0)==0);
    assert(prepare_witness_directory(&cfg,0)!=0);
    double tones[NBIN_MAX]; make_tones(tones,cfg.nbin,cfg.f_lo,cfg.f_hi);
    assert(write_schedule_json(&cfg,&cb,tones,symbols,total,123456789ULL)==0);
    char schedule_path[512]; snprintf(schedule_path,sizeof(schedule_path),"%s/schedule.json",cfg.witness_dir);
    FILE *schedule=fopen(schedule_path,"r"); assert(schedule!=NULL);
    char payload[8192]; size_t count=fread(payload,1,sizeof(payload)-1,schedule); payload[count]=0; fclose(schedule);
    assert(strstr(payload,"\"t0_tsc\": 123456789")!=NULL);
    assert(strstr(payload,"\"drive_signs\"")!=NULL);

    CarrierWitnessRawWriter writer; CarrierWitnessWindow window;
    uint64_t ts[4]={100,200,300,400}; double ro[4]={1,1,1,1};
    memset(&window,0,sizeof(window));
    window.symbol_index=0; window.bin_index=0; window.hash_restored=1; window.tone_hz=20;
    window.drive_sign=1; window.slot_start_tsc=50; window.capture_deadline_tsc=500;
    window.temp_before_c=-999; window.temp_after_c=40;
    snprintf(window.family,sizeof(window.family),"preamble");
    snprintf(window.declared_mode,sizeof(window.declared_mode),"basis");
    snprintf(window.actual_mode,sizeof(window.actual_mode),"basis");
    snprintf(window.control,sizeof(window.control),"matrix");
    assert(carrier_witness_raw_open(&writer,cfg.witness_dir)==0);
    assert(carrier_witness_raw_append(&writer,&window,ts,ro,4)!=0);
    assert(carrier_witness_raw_close(&writer)!=0);

    free(symbols);
    remove_file(cfg.witness_dir,"schedule.json"); remove_file(cfg.witness_dir,"stdout.log");
    remove_file(cfg.witness_dir,"stderr.log"); remove_file(cfg.witness_dir,"raw_samples.bin");
    remove_file(cfg.witness_dir,"windows.csv");
    rmdir(cfg.witness_dir); rmdir(parent);
    puts("SLOT2_WITNESS_INTEGRATION_TEST_PASS");
    return 0;
}
