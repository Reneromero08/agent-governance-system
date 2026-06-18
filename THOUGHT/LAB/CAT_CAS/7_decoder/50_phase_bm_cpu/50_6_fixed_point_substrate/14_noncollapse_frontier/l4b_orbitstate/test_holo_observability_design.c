#include "../holo_runtime/holo_observability_design.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static HoloObservabilityDesign make_design(void) {
    HoloObservabilityDesign d;
    assert(holo_observability_design_init(&d) == 0);
    assert(holo_observability_design_populate_current(&d) == 0);
    assert(holo_observability_design_validate(&d));
    return d;
}

static char *read_file(const char *path, size_t *size_out) {
    FILE *f=fopen(path,"rb"); long n; char *s;
    assert(f); assert(fseek(f,0,SEEK_END)==0); n=ftell(f); assert(n>=0); assert(fseek(f,0,SEEK_SET)==0);
    s=(char*)malloc((size_t)n+1U); assert(s); assert(fread(s,1,(size_t)n,f)==(size_t)n); s[n]='\0'; fclose(f); *size_out=(size_t)n; return s;
}

static void replace_once(const char *path, const char *from, const char *to) {
    size_t n,from_n=strlen(from),to_n=strlen(to),prefix;char*s=read_file(path,&n),*at=strstr(s,from),*out;FILE*f;
    assert(at);prefix=(size_t)(at-s);out=(char*)malloc(n-from_n+to_n+1U);assert(out);
    memcpy(out,s,prefix);memcpy(out+prefix,to,to_n);memcpy(out+prefix+to_n,at+from_n,n-prefix-from_n);
    out[n-from_n+to_n]='\0';f=fopen(path,"wb");assert(f);assert(fwrite(out,1,n-from_n+to_n,f)==n-from_n+to_n);assert(fclose(f)==0);free(out);free(s);
}

static int file_has_key(const char *path,const char *key){size_t n;char*s=read_file(path,&n);int found=strstr(s,key)!=NULL;(void)n;free(s);return found;}

static void test_validation(void) {
    HoloObservabilityDesign d=make_design(),empty;
    HoloOperatorCandidate op=d.operators[0];
    HoloObservableChannel unavailable=d.observables[HOLO_DESIGN_OBSERVABLE_COUNT-1U];
    puts("COMPLETE_OBSERVABILITY_DESIGN_PASS");
    assert(holo_observability_design_init(&empty)==0);
    assert(holo_observability_design_register_operator(&empty,&op)!=0);
    puts("MISSING_STATE_MODEL_REJECTED_PASS");
    unavailable.instrumentation_plan[0]='\0';
    assert(holo_observability_design_register_observable(&empty,&unavailable)!=0);
    snprintf(d.state_models[0].fields,sizeof(d.state_models[0].fields),"internal_pdn_modes");
    d.state_models[0].instrumentation_plan[0]='\0';
    assert(!holo_observability_design_validate(&d));
    puts("UNAVAILABLE_OBSERVABLE_WITHOUT_PLAN_REJECTED_PASS");
    snprintf(d.splits.test_seeds,sizeof(d.splits.test_seeds),"train_1001;test_3002");
    assert(!holo_observability_design_validate(&d));
    puts("DATA_SPLIT_LEAKAGE_REJECTED_PASS");
    d=make_design();snprintf(d.gates[1].gate_id,sizeof(d.gates[1].gate_id),"GX");
    assert(!holo_observability_design_validate(&d));
    puts("MISSING_HELD_OUT_GATE_REJECTED_PASS");
    d=make_design();d.full_physical_observability_claimed=1;assert(!holo_observability_design_validate(&d));
    d.full_physical_observability_claimed=0;d.physical_restoration_claimed=1;assert(!holo_observability_design_validate(&d));
    puts("CLAIM_INFLATION_REJECTED_PASS");
    d=make_design();d.implementation_authorized=1;assert(!holo_observability_design_validate(&d));
    puts("PREMATURE_AUTHORIZATION_REJECTED_PASS");
    d=make_design();d.reviewed_mapping_digest^=1U;assert(!holo_observability_design_validate(&d));
    puts("MAPPING_DIGEST_MISMATCH_REJECTED_PASS");
    d=make_design();assert(holo_observability_design_seal(&d)==0);assert(holo_observability_design_register_input(&d,&d.input_families[0])!=0);
    puts("SEALED_DESIGN_MUTATION_REJECTED_PASS");
    holo_observability_design_destroy(&d);holo_observability_design_destroy(&empty);
}

static void test_review_corrections(void) {
    HoloObservabilityDesign d=make_design();
    const HoloObservabilityTest *repeat=&d.observability_tests[0];
    const HoloObservabilityTest *distinguish=&d.observability_tests[1];
    const HoloObservabilityTest *delay=&d.observability_tests[2];
    const HoloFalsificationCondition *f6=&d.falsifications[5];
    assert(strstr(repeat->metric,"whitened Euclidean") && strstr(repeat->metric,"absolute-TSC") && strstr(repeat->metric,"no dynamic warping"));
    assert(strstr(repeat->threshold,"complete held-out session-route-schedule") && strstr(repeat->threshold,"within=median") && strstr(repeat->threshold,"between=q05"));
    assert(strstr(repeat->decision,"10000-resample session-block-bootstrap") && strstr(repeat->decision,"upper95(within)<lower95(between)"));
    puts("REPEATABILITY_GATE_OPERATIONAL_PASS");
    assert(strstr(distinguish->metric,"classes=idle,low_load,high_load,post_impulse_history") && strstr(distinguish->metric,"balance=1:1:1:1") && strstr(distinguish->metric,"L2 multinomial logistic") && strstr(distinguish->metric,"balanced_accuracy"));
    assert(strstr(distinguish->threshold,"session-level split") && strstr(distinguish->threshold,"chance=0.25") && strstr(distinguish->threshold,"delta_power") && strstr(distinguish->threshold,"alpha=0.05,power=0.80"));
    assert(strstr(distinguish->decision,"10000-resample session-block bootstrap") && !strstr(distinguish->threshold,"0.60"));
    puts("STATE_DISTINGUISHABILITY_GATE_OPERATIONAL_PASS");
    assert(strstr(delay->threshold,"not an authorization gate") && strstr(delay->decision,"if S1 passes sufficiency retain S1"));
    assert(strstr(f6->threshold,"S1 fails") && strstr(f6->threshold,"every S2 L fails predictive sufficiency"));
    assert(strstr(f6->next_action,"if S1 passes select S1 when S2 gain<10 percent") && strstr(f6->next_action,"select smallest sufficient S2 L"));
    puts("F6_CONDITIONAL_SUFFICIENCY_PASS");
    holo_observability_design_destroy(&d);
}

static void expect_tamper_rejected(HoloObservabilityDesign *d,const char *path,const char *from,const char *to,const char *label){HoloObservabilityDesign loaded;assert(holo_observability_design_write_json(d,path)==0);replace_once(path,from,to);assert(holo_observability_design_read_json(&loaded,path)!=0);puts(label);}

static void test_roundtrip_tampering(void) {
    const char *path="holo_observability_design_test.json";HoloObservabilityDesign d=make_design(),loaded;
    assert(holo_observability_design_seal(&d)==0);assert(holo_observability_design_write_json(&d,path)==0);
    assert(holo_observability_design_read_json(&loaded,path)==0);assert(holo_observability_design_equal(&d,&loaded));
    puts("OBSERVABILITY_DESIGN_SERIALIZATION_ROUNDTRIP_PASS");holo_observability_design_destroy(&loaded);
    expect_tamper_rejected(&d,path,"\"reviewed_mapping_digest\":\"0d06f3c8b44f8c55\"","\"reviewed_mapping_digest\":\"0d06f3c8b44f8c54\"","SERIALIZED_MAPPING_DIGEST_TAMPERING_REJECTED_PASS");
    expect_tamper_rejected(&d,path,"\"status\":\"READY_FOR_HUMAN_REVIEW\"","\"status\":\"ACCEPTED\"","SERIALIZED_DESIGN_STATUS_TAMPERING_REJECTED_PASS");
    expect_tamper_rejected(&d,path,"\"implementation_authorized\":false","\"implementation_authorized\":true","SERIALIZED_DESIGN_AUTHORIZATION_TAMPERING_REJECTED_PASS");
    expect_tamper_rejected(&d,path,"\"test_seeds\":\"test_3001;test_3002\"","\"test_seeds\":\"train_1001;test_3002\"","SERIALIZED_DATA_SPLIT_TAMPERING_REJECTED_PASS");
    expect_tamper_rejected(&d,path,"at least 10 percent below min(mean,persistence)","at least 01 percent below min(mean,persistence)","SERIALIZED_ACCEPTANCE_GATE_TAMPERING_REJECTED_PASS");
    expect_tamper_rejected(&d,path,"lockin_I;lockin_Q;ring_osc_period","rail_voltage_current;lockin_Q;ring_osc_period","SERIALIZED_STATE_OBSERVABLE_TAMPERING_REJECTED_PASS");
    assert(holo_observability_design_write_json(&d,path)==0);
    assert(!file_has_key(path,"\"winner\""));assert(!file_has_key(path,"\"candidate_score\""));assert(!file_has_key(path,"\"hidden_d\""));
    assert(!file_has_key(path,"\"recovered_d\""));assert(!file_has_key(path,"\"orientation_label\""));assert(!file_has_key(path,"\"verify_pass\""));assert(!file_has_key(path,"\"AUC\""));
    puts("OBSERVABILITY_DESIGN_FORBIDDEN_FIELDS_PASS");remove(path);holo_observability_design_destroy(&d);
}

static void print_design(void) {
    HoloObservabilityDesign d=make_design();size_t i;assert(holo_observability_design_seal(&d)==0);
    printf("DESIGN id=%s version=%s status=%s mapping=%s:%016llx authorized=%s digest=%016llx\n",d.design_id,d.design_version,holo_experiment_design_status_name(d.status),d.mapping_contract_id,(unsigned long long)d.reviewed_mapping_digest,d.implementation_authorized?"true":"false",(unsigned long long)d.design_digest);
    for(i=0;i<d.state_count;i++)printf("STATE %s | %s | L=%s | %s\n",d.state_models[i].model_id,d.state_models[i].fields,d.state_models[i].history_lengths,d.state_models[i].limitations);
    for(i=0;i<d.input_count;i++)printf("INPUT %s | %s | %s\n",d.input_families[i].input_id,d.input_families[i].duration,d.input_families[i].null_control);
    for(i=0;i<d.observable_count;i++)printf("OBSERVABLE %s | %s | %s\n",d.observables[i].observable_id,holo_observable_availability_name(d.observables[i].availability),d.observables[i].instrumentation_source);
    for(i=0;i<d.operator_count;i++)printf("OPERATOR %s | %s | state=%s | rank=%d\n",d.operators[i].operator_id,holo_operator_family_name(d.operators[i].family),d.operators[i].state_model_id,d.operators[i].complexity_rank);
    for(i=0;i<d.calibration_count;i++)printf("CALIBRATION %s | %s | %s\n",d.calibration[i].stage_id,d.calibration[i].input_id,d.calibration[i].failure_condition);
    for(i=0;i<d.observability_test_count;i++)printf("OBS_GATE %s | %s\n",d.observability_tests[i].test_id,d.observability_tests[i].threshold);
    for(i=0;i<d.gate_count;i++)printf("OP_GATE %s | %s\n",d.gates[i].gate_id,d.gates[i].threshold);
    for(i=0;i<d.falsification_count;i++)printf("FALSIFICATION %s | %s | %s\n",d.falsifications[i].condition_id,d.falsifications[i].threshold,d.falsifications[i].decision);
    puts("L4B5B0_GATE_DECISION=READY_FOR_HUMAN_REVIEW");holo_observability_design_destroy(&d);
}

int main(void){test_validation();test_review_corrections();test_roundtrip_tampering();print_design();puts("HOLO_OBSERVABILITY_DESIGN_TEST_PASS");return 0;}
