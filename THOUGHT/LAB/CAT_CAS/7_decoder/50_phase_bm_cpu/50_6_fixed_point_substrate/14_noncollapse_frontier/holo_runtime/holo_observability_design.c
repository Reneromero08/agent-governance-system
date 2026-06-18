#include "holo_observability_design.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
static const uint64_t FNV_PRIME = UINT64_C(1099511628211);

static void copy_text(char *dst, size_t size, const char *src) {
    size_t n;
    if (!dst || size == 0) return;
    if (!src) src = "";
    n = strlen(src);
    if (n >= size) n = size - 1U;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static int present(const char *text) { return text && text[0] != '\0'; }

static int contains_ci(const char *text, const char *needle) {
    char lower[HOLO_DESIGN_TEXT_LEN];
    size_t i;
    if (!text || !needle) return 0;
    for (i = 0; i + 1U < sizeof(lower) && text[i] != '\0'; ++i)
        lower[i] = (char)tolower((unsigned char)text[i]);
    lower[i] = '\0';
    return strstr(lower, needle) != NULL;
}

const char *holo_observable_availability_name(HoloObservableAvailability value) {
    static const char *names[] = {"AVAILABLE_NOW", "AVAILABLE_WITH_MINOR_INSTRUMENTATION",
        "AVAILABLE_WITH_EXTERNAL_INSTRUMENTATION", "UNAVAILABLE"};
    return value >= HOLO_OBSERVABLE_AVAILABLE_NOW && value <= HOLO_OBSERVABLE_UNAVAILABLE
        ? names[value] : "INVALID";
}

const char *holo_state_model_kind_name(HoloPhysicalStateModelKind value) {
    static const char *names[] = {"MINIMAL_MEASURED", "MEASURED_WITH_CONTEXT", "DELAY_EMBEDDED"};
    return value >= HOLO_STATE_MINIMAL && value <= HOLO_STATE_DELAY_EMBEDDED ? names[value] : "INVALID";
}

const char *holo_operator_family_name(HoloOperatorFamily value) {
    static const char *names[] = {"AFFINE_LINEAR", "ROUTE_CONDITIONED_LINEAR", "BILINEAR", "COMPACT_NONLINEAR"};
    return value >= HOLO_OPERATOR_AFFINE_LINEAR && value <= HOLO_OPERATOR_COMPACT_NONLINEAR
        ? names[value] : "INVALID";
}

const char *holo_experiment_design_status_name(HoloExperimentDesignStatus value) {
    static const char *names[] = {"DRAFT", "READY_FOR_HUMAN_REVIEW", "ACCEPTED", "REJECTED"};
    return value >= HOLO_DESIGN_DRAFT && value <= HOLO_DESIGN_REJECTED ? names[value] : "INVALID";
}

int holo_observability_design_init(HoloObservabilityDesign *d) {
    if (!d) return -1;
    memset(d, 0, sizeof(*d));
    copy_text(d->design_id, sizeof(d->design_id), "l4b5b0_observability_operator_v1");
    copy_text(d->design_version, sizeof(d->design_version), "1.0.0");
    copy_text(d->experiment_family, sizeof(d->experiment_family), "l4b5b0_observability_operator_design");
    copy_text(d->mapping_contract_id, sizeof(d->mapping_contract_id), "l4b5a_pdn_mapping_v1");
    d->reviewed_mapping_digest = HOLO_DESIGN_MAPPING_DIGEST;
    d->status = HOLO_DESIGN_READY_FOR_HUMAN_REVIEW;
    copy_text(d->next_gate, sizeof(d->next_gate), "HUMAN_DESIGN_REVIEW");
    d->review_required = 1;
    d->claim_level = 1;
    return 0;
}

void holo_observability_design_destroy(HoloObservabilityDesign *d) {
    if (d) memset(d, 0, sizeof(*d));
}

int holo_observability_design_register_state(HoloObservabilityDesign *d, const HoloPhysicalStateModel *r) {
    if (!d || !r || d->sealed || d->state_count >= HOLO_DESIGN_STATE_COUNT ||
        !present(r->model_id) || !present(r->fields) || !present(r->available_observables) ||
        !present(r->history_lengths) || r->maximum_history_length < 1 ||
        !present(r->purpose) || !present(r->limitations) ||
        (r->requires_unavailable_observable && !present(r->instrumentation_plan))) return -1;
    d->state_models[d->state_count++] = *r;
    return 0;
}

int holo_observability_design_register_input(HoloObservabilityDesign *d, const HoloInputFamily *r) {
    if (!d || !r || d->sealed || d->input_count >= HOLO_DESIGN_INPUT_COUNT ||
        !present(r->input_id) || !present(r->schedule) || !present(r->seed_policy) ||
        !present(r->duration) || !present(r->purpose) || !present(r->null_control)) return -1;
    d->input_families[d->input_count++] = *r;
    return 0;
}

int holo_observability_design_register_observable(HoloObservabilityDesign *d, const HoloObservableChannel *r) {
    if (!d || !r || d->sealed || d->observable_count >= HOLO_DESIGN_OBSERVABLE_COUNT ||
        !present(r->observable_id) || !present(r->instrumentation_source) ||
        !present(r->sampling_constraints) || !present(r->known_noise) ||
        (r->availability != HOLO_OBSERVABLE_AVAILABLE_NOW && !present(r->instrumentation_plan))) return -1;
    d->observables[d->observable_count++] = *r;
    return 0;
}

static int has_state(const HoloObservabilityDesign *d, const char *id) {
    size_t i;
    for (i = 0; i < d->state_count; ++i) if (strcmp(d->state_models[i].model_id, id) == 0) return 1;
    return 0;
}

int holo_observability_design_register_operator(HoloObservabilityDesign *d, const HoloOperatorCandidate *r) {
    if (!d || !r || d->sealed || d->operator_count >= HOLO_DESIGN_OPERATOR_COUNT ||
        !present(r->operator_id) || !present(r->state_model_id) || !has_state(d, r->state_model_id) ||
        !present(r->model) || !present(r->parameters) || !present(r->required_data) ||
        !present(r->acceptance_conditions) || !present(r->rejection_conditions) ||
        r->complexity_rank < 0 || r->complexity_rank > 3) return -1;
    d->operators[d->operator_count++] = *r;
    return 0;
}

int holo_observability_design_register_calibration(HoloObservabilityDesign *d, const HoloCalibrationStage *r) {
    if (!d || !r || d->sealed || d->calibration_count >= HOLO_DESIGN_CALIBRATION_COUNT ||
        !present(r->stage_id) || !present(r->purpose) || !present(r->input_id) ||
        !present(r->duration) || !present(r->trials) || !present(r->sample_rate) ||
        !present(r->state_preparation) || !present(r->expected_observable) ||
        !present(r->artifact_output) || !present(r->failure_condition)) return -1;
    d->calibration[d->calibration_count++] = *r;
    return 0;
}

int holo_observability_design_register_gate(HoloObservabilityDesign *d, const HoloOperatorAcceptanceGate *r) {
    if (!d || !r || d->sealed || d->gate_count >= HOLO_DESIGN_GATE_COUNT ||
        !present(r->gate_id) || !present(r->metric) || !present(r->threshold) ||
        !present(r->decision)) return -1;
    d->gates[d->gate_count++] = *r;
    return 0;
}

static int tokens_overlap(const char *a, const char *b) {
    char left[HOLO_DESIGN_TEXT_LEN];
    char right[HOLO_DESIGN_TEXT_LEN];
    char *token_a, *token_b, *ctx_a, *ctx_b;
    copy_text(left, sizeof(left), a);
    for (token_a = strtok_r(left, ";", &ctx_a); token_a; token_a = strtok_r(NULL, ";", &ctx_a)) {
        copy_text(right, sizeof(right), b);
        for (token_b = strtok_r(right, ";", &ctx_b); token_b; token_b = strtok_r(NULL, ";", &ctx_b))
            if (strcmp(token_a, token_b) == 0) return 1;
    }
    return 0;
}

static int has_gate(const HoloObservabilityDesign *d, const char *id) {
    size_t i;
    for (i = 0; i < d->gate_count; ++i) if (strcmp(d->gates[i].gate_id, id) == 0) return 1;
    return 0;
}

static uint64_t hash_text(uint64_t h, const char *s) {
    while (*s) h = (h ^ (unsigned char)*s++) * FNV_PRIME;
    return h;
}

static uint64_t hash_u64(uint64_t h, uint64_t v) {
    unsigned int shift;
    for (shift = 0; shift < 64U; shift += 8U) h = (h ^ (unsigned char)(v >> shift)) * FNV_PRIME;
    return h;
}

#define HASH_TEXT_FIELD(h, obj, field) (h) = hash_text((h), (obj)->field)
#define HASH_INT_FIELD(h, obj, field) (h) = hash_u64((h), (uint64_t)(obj)->field)

static uint64_t design_digest(const HoloObservabilityDesign *d) {
    uint64_t h = FNV_OFFSET;
    size_t i;
    HASH_TEXT_FIELD(h,d,design_id); HASH_TEXT_FIELD(h,d,design_version); HASH_TEXT_FIELD(h,d,experiment_family);
    HASH_TEXT_FIELD(h,d,mapping_contract_id); HASH_INT_FIELD(h,d,reviewed_mapping_digest); HASH_INT_FIELD(h,d,status);
    HASH_TEXT_FIELD(h,d,state_space_model); HASH_TEXT_FIELD(h,d,target_claim); HASH_TEXT_FIELD(h,d,forbidden_claim);
    HASH_TEXT_FIELD(h,d,topology_metadata); HASH_TEXT_FIELD(h,d,design_bounds); HASH_TEXT_FIELD(h,d,baselines);
    HASH_TEXT_FIELD(h,d,input_policy); HASH_TEXT_FIELD(h,d,observability_definitions); HASH_TEXT_FIELD(h,d,state_preparation);
    HASH_TEXT_FIELD(h,d,next_gate); HASH_INT_FIELD(h,d,review_required); HASH_INT_FIELD(h,d,human_reviewed);
    HASH_INT_FIELD(h,d,implementation_authorized); HASH_INT_FIELD(h,d,executed);
    HASH_INT_FIELD(h,d,full_physical_observability_claimed); HASH_INT_FIELD(h,d,physical_restoration_claimed);
    HASH_INT_FIELD(h,d,claim_level);
    HASH_TEXT_FIELD(h,&d->splits,training_seeds); HASH_TEXT_FIELD(h,&d->splits,validation_seeds);
    HASH_TEXT_FIELD(h,&d->splits,test_seeds); HASH_TEXT_FIELD(h,&d->splits,training_sessions);
    HASH_TEXT_FIELD(h,&d->splits,validation_sessions); HASH_TEXT_FIELD(h,&d->splits,test_sessions);
    HASH_TEXT_FIELD(h,&d->splits,route_holdout); HASH_TEXT_FIELD(h,&d->splits,normalization_policy);
    HASH_TEXT_FIELD(h,&d->splits,model_selection_policy); HASH_TEXT_FIELD(h,&d->splits,leakage_prevention);
    for (i=0;i<d->state_count;i++){const HoloPhysicalStateModel*r=&d->state_models[i];HASH_TEXT_FIELD(h,r,model_id);HASH_INT_FIELD(h,r,kind);HASH_TEXT_FIELD(h,r,fields);HASH_TEXT_FIELD(h,r,available_observables);HASH_TEXT_FIELD(h,r,history_lengths);HASH_INT_FIELD(h,r,maximum_history_length);HASH_TEXT_FIELD(h,r,purpose);HASH_TEXT_FIELD(h,r,limitations);HASH_INT_FIELD(h,r,requires_unavailable_observable);HASH_TEXT_FIELD(h,r,instrumentation_plan);}
    for (i=0;i<d->input_count;i++){const HoloInputFamily*r=&d->input_families[i];HASH_TEXT_FIELD(h,r,input_id);HASH_TEXT_FIELD(h,r,schedule);HASH_TEXT_FIELD(h,r,seed_policy);HASH_TEXT_FIELD(h,r,duration);HASH_TEXT_FIELD(h,r,purpose);HASH_TEXT_FIELD(h,r,null_control);}
    for (i=0;i<d->observable_count;i++){const HoloObservableChannel*r=&d->observables[i];HASH_TEXT_FIELD(h,r,observable_id);HASH_INT_FIELD(h,r,availability);HASH_TEXT_FIELD(h,r,instrumentation_source);HASH_TEXT_FIELD(h,r,sampling_constraints);HASH_TEXT_FIELD(h,r,known_noise);HASH_TEXT_FIELD(h,r,instrumentation_plan);}
    for (i=0;i<d->operator_count;i++){const HoloOperatorCandidate*r=&d->operators[i];HASH_TEXT_FIELD(h,r,operator_id);HASH_INT_FIELD(h,r,family);HASH_TEXT_FIELD(h,r,state_model_id);HASH_TEXT_FIELD(h,r,model);HASH_TEXT_FIELD(h,r,parameters);HASH_TEXT_FIELD(h,r,required_data);HASH_TEXT_FIELD(h,r,acceptance_conditions);HASH_TEXT_FIELD(h,r,rejection_conditions);HASH_INT_FIELD(h,r,complexity_rank);}
    for (i=0;i<d->calibration_count;i++){const HoloCalibrationStage*r=&d->calibration[i];HASH_TEXT_FIELD(h,r,stage_id);HASH_TEXT_FIELD(h,r,purpose);HASH_TEXT_FIELD(h,r,input_id);HASH_TEXT_FIELD(h,r,duration);HASH_TEXT_FIELD(h,r,trials);HASH_TEXT_FIELD(h,r,sample_rate);HASH_TEXT_FIELD(h,r,state_preparation);HASH_TEXT_FIELD(h,r,expected_observable);HASH_TEXT_FIELD(h,r,artifact_output);HASH_TEXT_FIELD(h,r,failure_condition);}
    for (i=0;i<d->observability_test_count;i++){const HoloObservabilityTest*r=&d->observability_tests[i];HASH_TEXT_FIELD(h,r,test_id);HASH_TEXT_FIELD(h,r,metric);HASH_TEXT_FIELD(h,r,threshold);HASH_TEXT_FIELD(h,r,decision);HASH_TEXT_FIELD(h,r,limitation);}
    for (i=0;i<d->gate_count;i++){const HoloOperatorAcceptanceGate*r=&d->gates[i];HASH_TEXT_FIELD(h,r,gate_id);HASH_TEXT_FIELD(h,r,metric);HASH_TEXT_FIELD(h,r,threshold);HASH_TEXT_FIELD(h,r,decision);HASH_INT_FIELD(h,r,requires_human_approval);}
    for (i=0;i<d->falsification_count;i++){const HoloFalsificationCondition*r=&d->falsifications[i];HASH_TEXT_FIELD(h,r,condition_id);HASH_TEXT_FIELD(h,r,condition);HASH_TEXT_FIELD(h,r,metric);HASH_TEXT_FIELD(h,r,threshold);HASH_TEXT_FIELD(h,r,decision);HASH_TEXT_FIELD(h,r,next_action);}
    for (i=0;i<d->artifact_count;i++){const HoloFutureArtifactContract*r=&d->artifacts[i];HASH_TEXT_FIELD(h,r,artifact_id);HASH_TEXT_FIELD(h,r,schema_id);HASH_TEXT_FIELD(h,r,required_fields);HASH_TEXT_FIELD(h,r,digest_rule);HASH_TEXT_FIELD(h,r,source_run_ids);HASH_INT_FIELD(h,r,claim_level);}
    return h;
}

int holo_observability_design_validate(const HoloObservabilityDesign *d) {
    size_t i;
    if (!d || strcmp(d->design_id,"l4b5b0_observability_operator_v1") || strcmp(d->design_version,"1.0.0") ||
        strcmp(d->experiment_family,"l4b5b0_observability_operator_design") ||
        strcmp(d->mapping_contract_id,"l4b5a_pdn_mapping_v1") || d->reviewed_mapping_digest != HOLO_DESIGN_MAPPING_DIGEST ||
        d->status != HOLO_DESIGN_READY_FOR_HUMAN_REVIEW || !d->review_required || d->human_reviewed ||
        d->implementation_authorized || d->executed || d->full_physical_observability_claimed ||
        d->physical_restoration_claimed || d->claim_level != 1 || strcmp(d->next_gate,"HUMAN_DESIGN_REVIEW") ||
        !present(d->state_space_model) || !present(d->target_claim) || !present(d->forbidden_claim) ||
        contains_ci(d->target_claim,"full physical observability") || contains_ci(d->target_claim,"restoration proven") ||
        !present(d->topology_metadata) || !present(d->design_bounds) || !present(d->baselines) ||
        !present(d->input_policy) || !present(d->observability_definitions) || !present(d->state_preparation) ||
        d->state_count != HOLO_DESIGN_STATE_COUNT || d->input_count != HOLO_DESIGN_INPUT_COUNT ||
        d->observable_count != HOLO_DESIGN_OBSERVABLE_COUNT || d->operator_count != HOLO_DESIGN_OPERATOR_COUNT ||
        d->calibration_count != HOLO_DESIGN_CALIBRATION_COUNT || d->observability_test_count != HOLO_DESIGN_OBSERVABILITY_TEST_COUNT ||
        d->gate_count != HOLO_DESIGN_GATE_COUNT || d->falsification_count != HOLO_DESIGN_FALSIFICATION_COUNT ||
        d->artifact_count != HOLO_DESIGN_ARTIFACT_COUNT || !has_gate(d,"G2")) return 0;
    if (!present(d->splits.training_seeds) || !present(d->splits.validation_seeds) || !present(d->splits.test_seeds) ||
        !present(d->splits.training_sessions) || !present(d->splits.validation_sessions) || !present(d->splits.test_sessions) ||
        tokens_overlap(d->splits.training_seeds,d->splits.validation_seeds) || tokens_overlap(d->splits.training_seeds,d->splits.test_seeds) ||
        tokens_overlap(d->splits.validation_seeds,d->splits.test_seeds) || tokens_overlap(d->splits.training_sessions,d->splits.validation_sessions) ||
        tokens_overlap(d->splits.training_sessions,d->splits.test_sessions) || tokens_overlap(d->splits.validation_sessions,d->splits.test_sessions) ||
        !present(d->splits.route_holdout) || !present(d->splits.normalization_policy) ||
        !present(d->splits.model_selection_policy) || !present(d->splits.leakage_prevention)) return 0;
    for(i=0;i<d->state_count;i++) {
        size_t j;
        if(!present(d->state_models[i].model_id) ||
           (d->state_models[i].requires_unavailable_observable && !present(d->state_models[i].instrumentation_plan))) return 0;
        for(j=0;j<d->observable_count;j++) {
            if(d->observables[j].availability != HOLO_OBSERVABLE_AVAILABLE_NOW &&
               strstr(d->state_models[i].fields,d->observables[j].observable_id) &&
               !present(d->state_models[i].instrumentation_plan)) return 0;
        }
    }
    for(i=0;i<d->observable_count;i++) if(!present(d->observables[i].observable_id) ||
        (d->observables[i].availability != HOLO_OBSERVABLE_AVAILABLE_NOW && !present(d->observables[i].instrumentation_plan))) return 0;
    for(i=0;i<d->operator_count;i++) if(!has_state(d,d->operators[i].state_model_id) || !contains_ci(d->operators[i].required_data,"held-out")) return 0;
    for(i=0;i<d->observability_test_count;i++) if(!present(d->observability_tests[i].metric) || !present(d->observability_tests[i].threshold)) return 0;
    for(i=0;i<d->gate_count;i++) if(!present(d->gates[i].metric) || !present(d->gates[i].threshold)) return 0;
    for(i=0;i<d->falsification_count;i++) if(!present(d->falsifications[i].metric) || !present(d->falsifications[i].threshold) || !present(d->falsifications[i].decision)) return 0;
    for(i=0;i<d->artifact_count;i++) if(!present(d->artifacts[i].schema_id) || !present(d->artifacts[i].required_fields) || !present(d->artifacts[i].digest_rule) || !present(d->artifacts[i].source_run_ids) || d->artifacts[i].claim_level != 1) return 0;
    return !d->sealed || d->design_digest == design_digest(d);
}

int holo_observability_design_seal(HoloObservabilityDesign *d) {
    if (!d || d->sealed || !holo_observability_design_validate(d)) return -1;
    d->sealed = 1;
    d->design_digest = design_digest(d);
    return 0;
}

static int add_state(HoloObservabilityDesign *d, const char *id, HoloPhysicalStateModelKind kind,
                     const char *fields, const char *available, const char *lengths, int maximum,
                     const char *purpose, const char *limitations) {
    HoloPhysicalStateModel r;
    memset(&r,0,sizeof(r)); copy_text(r.model_id,sizeof(r.model_id),id); r.kind=kind;
    copy_text(r.fields,sizeof(r.fields),fields); copy_text(r.available_observables,sizeof(r.available_observables),available);
    copy_text(r.history_lengths,sizeof(r.history_lengths),lengths); r.maximum_history_length=maximum;
    copy_text(r.purpose,sizeof(r.purpose),purpose); copy_text(r.limitations,sizeof(r.limitations),limitations);
    return holo_observability_design_register_state(d,&r);
}

static int add_input(HoloObservabilityDesign *d, const char *id, const char *schedule,
                     const char *seed, const char *duration, const char *purpose, const char *control) {
    HoloInputFamily r;
    memset(&r,0,sizeof(r)); copy_text(r.input_id,sizeof(r.input_id),id); copy_text(r.schedule,sizeof(r.schedule),schedule);
    copy_text(r.seed_policy,sizeof(r.seed_policy),seed); copy_text(r.duration,sizeof(r.duration),duration);
    copy_text(r.purpose,sizeof(r.purpose),purpose); copy_text(r.null_control,sizeof(r.null_control),control);
    return holo_observability_design_register_input(d,&r);
}

static int add_observable(HoloObservabilityDesign *d, const char *id, HoloObservableAvailability availability,
                          const char *source, const char *sampling, const char *noise, const char *plan) {
    HoloObservableChannel r;
    memset(&r,0,sizeof(r)); copy_text(r.observable_id,sizeof(r.observable_id),id); r.availability=availability;
    copy_text(r.instrumentation_source,sizeof(r.instrumentation_source),source);
    copy_text(r.sampling_constraints,sizeof(r.sampling_constraints),sampling);
    copy_text(r.known_noise,sizeof(r.known_noise),noise); copy_text(r.instrumentation_plan,sizeof(r.instrumentation_plan),plan);
    return holo_observability_design_register_observable(d,&r);
}

static int add_operator(HoloObservabilityDesign *d, const char *id, HoloOperatorFamily family,
                        const char *state, const char *model, const char *parameters,
                        const char *data, const char *accept, const char *reject, int rank) {
    HoloOperatorCandidate r;
    memset(&r,0,sizeof(r)); copy_text(r.operator_id,sizeof(r.operator_id),id); r.family=family;
    copy_text(r.state_model_id,sizeof(r.state_model_id),state); copy_text(r.model,sizeof(r.model),model);
    copy_text(r.parameters,sizeof(r.parameters),parameters); copy_text(r.required_data,sizeof(r.required_data),data);
    copy_text(r.acceptance_conditions,sizeof(r.acceptance_conditions),accept);
    copy_text(r.rejection_conditions,sizeof(r.rejection_conditions),reject); r.complexity_rank=rank;
    return holo_observability_design_register_operator(d,&r);
}

static int add_calibration(HoloObservabilityDesign *d, const char *id, const char *purpose,
                           const char *input, const char *duration, const char *trials,
                           const char *observable, const char *artifact, const char *failure) {
    HoloCalibrationStage r;
    memset(&r,0,sizeof(r)); copy_text(r.stage_id,sizeof(r.stage_id),id); copy_text(r.purpose,sizeof(r.purpose),purpose);
    copy_text(r.input_id,sizeof(r.input_id),input); copy_text(r.duration,sizeof(r.duration),duration);
    copy_text(r.trials,sizeof(r.trials),trials); copy_text(r.sample_rate,sizeof(r.sample_rate),"4000 Hz default; fixed per campaign");
    copy_text(r.state_preparation,sizeof(r.state_preparation),"quiesce;fixed warmup;affinity;P-state policy;sync;temperature proxy;baseline");
    copy_text(r.expected_observable,sizeof(r.expected_observable),observable);
    copy_text(r.artifact_output,sizeof(r.artifact_output),artifact); copy_text(r.failure_condition,sizeof(r.failure_condition),failure);
    return holo_observability_design_register_calibration(d,&r);
}

static void add_observability_test(HoloObservabilityDesign *d, const char *id, const char *metric,
                                   const char *threshold, const char *decision, const char *limitation) {
    HoloObservabilityTest *r=&d->observability_tests[d->observability_test_count++];
    memset(r,0,sizeof(*r)); copy_text(r->test_id,sizeof(r->test_id),id); copy_text(r->metric,sizeof(r->metric),metric);
    copy_text(r->threshold,sizeof(r->threshold),threshold); copy_text(r->decision,sizeof(r->decision),decision);
    copy_text(r->limitation,sizeof(r->limitation),limitation);
}

static int add_gate(HoloObservabilityDesign *d, const char *id, const char *metric,
                    const char *threshold, const char *decision) {
    HoloOperatorAcceptanceGate r;
    memset(&r,0,sizeof(r)); copy_text(r.gate_id,sizeof(r.gate_id),id); copy_text(r.metric,sizeof(r.metric),metric);
    copy_text(r.threshold,sizeof(r.threshold),threshold); copy_text(r.decision,sizeof(r.decision),decision);
    r.requires_human_approval=1; return holo_observability_design_register_gate(d,&r);
}

static void add_falsification(HoloObservabilityDesign *d, const char *id, const char *condition,
                              const char *metric, const char *threshold, const char *action) {
    HoloFalsificationCondition *r=&d->falsifications[d->falsification_count++];
    memset(r,0,sizeof(*r)); copy_text(r->condition_id,sizeof(r->condition_id),id); copy_text(r->condition,sizeof(r->condition),condition);
    copy_text(r->metric,sizeof(r->metric),metric); copy_text(r->threshold,sizeof(r->threshold),threshold);
    copy_text(r->decision,sizeof(r->decision),"BLOCK_L4B5B1"); copy_text(r->next_action,sizeof(r->next_action),action);
}

static void add_artifact(HoloObservabilityDesign *d, const char *id, const char *schema, const char *fields) {
    HoloFutureArtifactContract *r=&d->artifacts[d->artifact_count++];
    memset(r,0,sizeof(*r)); copy_text(r->artifact_id,sizeof(r->artifact_id),id); copy_text(r->schema_id,sizeof(r->schema_id),schema);
    copy_text(r->required_fields,sizeof(r->required_fields),fields);
    copy_text(r->digest_rule,sizeof(r->digest_rule),"SHA-256 over canonical artifact bytes");
    copy_text(r->source_run_ids,sizeof(r->source_run_ids),"nonempty immutable run IDs required"); r->claim_level=1;
}

int holo_observability_design_populate_current(HoloObservabilityDesign *d) {
    if (!d || d->state_count || d->input_count || d->observable_count || d->operator_count) return -1;
    copy_text(d->state_space_model,sizeof(d->state_space_model),"x(t+1)=F(x(t),u(t),eta(t));y(t)=H(x(t))+epsilon(t);y is not assumed equal to x");
    copy_text(d->target_claim,sizeof(d->target_claim),"empirical predictive observability of declared measured state under tested input family");
    copy_text(d->forbidden_claim,sizeof(d->forbidden_claim),"no complete substrate observability;no physical geometry;no reversibility;no restoration");
    copy_text(d->topology_metadata,sizeof(d->topology_metadata),"CPU;microcode/BIOS;OS/kernel;governor/P-state;sender cores;receiver core;NUMA/package;TSC;clock;affinity;interrupt policy;ambient/session");
    copy_text(d->design_bounds,sizeof(d->design_bounds),"routes=v2:s3,v4:s5,one optional holdout;sessions=3 train+1 validation+1 test per route;history=1,2,4,8,16,32;trials from pre-run power analysis");
    copy_text(d->baselines,sizeof(d->baselines),"mean;last-value persistence;phase-only;amplitude-only;input-only;route-only;time-index;shuffled-input null;random stable linear operator");
    copy_text(d->input_policy,sizeof(d->input_policy),"predeclared;reproducible;seeded where stochastic;time-bounded;independent of hidden labels;no outcome-conditioned route or seed selection");
    copy_text(d->observability_definitions,sizeof(d->observability_definitions),"instantaneous=y(t);finite-horizon=y/u history;local=perturbation rank surrogate;empirical=measured distinguishability;predictive=held-out sufficiency;full-state=not claimed");
    copy_text(d->state_preparation,sizeof(d->state_preparation),"idle/quiescent interval;fixed warm-up;fixed P-state policy;fixed affinity;receiver sync;temperature proxy;baseline capture;reject failed preparation");

    if(add_state(d,"S0_minimal",HOLO_STATE_MINIMAL,"lockin_I;lockin_Q;ring_osc_period","lockin_I;lockin_Q;ring_osc_period","1",1,"smallest directly measured state candidate","may omit control context and latent memory") ||
       add_state(d,"S1_contextual",HOLO_STATE_CONTEXTUAL,"lockin_I;lockin_Q;ring_osc_period;sender_mode;route;receiver_core;TSC_origin;capture_window","all fields available now","1",1,"test measured state with explicit input and topology context","context can explain response without making hidden substrate observable") ||
       add_state(d,"S2_delay_embedded",HOLO_STATE_DELAY_EMBEDDED,"y(t..t-L+1);u(t..t-L+1);route_context","I/Q;ring period;input history;timing;route","1;2;4;8;16;32",32,"test finite-history reconstruction of predictive state","embedding gain is empirical and does not prove full nonlinear observability")) return -2;

    if(add_input(d,"carrier_off","sender disabled;fixed capture windows","no seed","60 s baseline;3 blocks","measure receiver and environmental floor","constant_idle") ||
       add_input(d,"constant_idle","receiver schedule with idle sender","no seed","60 s;3 blocks","estimate drift and measurement noise","carrier_off") ||
       add_input(d,"constant_load","predeclared load levels 0,25,50,75,100 percent","fixed level order plus counterbalanced order","10 s per level;5 repeats","static response and saturation","constant_idle") ||
       add_input(d,"binary_mode_modulation","two fixed load modes in balanced blocks","partitioned train/validation/test seeds","256 windows per seed default","baseline controlled transition identification","shuffled_input") ||
       add_input(d,"multilevel_amplitude","four predeclared amplitude levels","partitioned seeds","256 windows per seed default","test input-response nonlinearity","constant_load") ||
       add_input(d,"phase_shifted_modulation","periodic load with phases 0,pi/2,pi,3pi/2","partitioned seeds","12 tones;0.5 s slots;repeat count power-derived","identify phase-dependent response","phase_randomized") ||
       add_input(d,"prbs","maximal-length PRBS with recorded polynomial","disjoint sequence seeds","1024 windows per seed default","persistently excite transition dynamics","shuffled_input") ||
       add_input(d,"structured_impulse","one burst followed by quiescent recovery","counterbalanced impulse positions","64 trials default;power review required","estimate impulse response and memory","carrier_off") ||
       add_input(d,"step_input","idle-to-load and load-to-idle steps","counterbalanced direction seeds","30 repeats per level default;power review required","estimate settling and hysteresis","constant_load") ||
       add_input(d,"paired_forward_reverse","predeclared schedule followed by reversed command order","disjoint schedule seeds","30 pairs default;design-only","future asymmetry probe;not restoration","reordered_schedule") ||
       add_input(d,"reordered_wrong_inverse","reordered and independently wrong reverse schedules","disjoint control seeds","matched to paired schedule","negative controls for order and inverse claims","paired_forward_reverse")) return -3;

    if(add_observable(d,"lockin_I",HOLO_OBSERVABLE_AVAILABLE_NOW,"slot2 lock-in demodulator","deadline-bounded fixed window","OS jitter;thermal drift;instrument noise","") ||
       add_observable(d,"lockin_Q",HOLO_OBSERVABLE_AVAILABLE_NOW,"slot2 lock-in demodulator","deadline-bounded fixed window","phase-origin and timing error","") ||
       add_observable(d,"lockin_amplitude",HOLO_OBSERVABLE_AVAILABLE_NOW,"deterministic I/Q recomputation","same I/Q window","Rice bias near noise floor","") ||
       add_observable(d,"lockin_phase",HOLO_OBSERVABLE_AVAILABLE_NOW,"deterministic atan2(Q,I)","undefined below predeclared amplitude floor","phase wrapping;TSC error","") ||
       add_observable(d,"ring_osc_period",HOLO_OBSERVABLE_AVAILABLE_NOW,"victim dependent-MAC ring oscillator","approximately 4000 Hz;absolute deadline","scheduler and microarchitectural jitter","") ||
       add_observable(d,"sample_timing",HOLO_OBSERVABLE_MINOR_INSTRUMENTATION,"RDTSCP timestamps currently held in capture buffers","record every sample","TSC calibration error","persist t_tsc arrays in raw capture artifact") ||
       add_observable(d,"deadline_miss_count",HOLO_OBSERVABLE_MINOR_INSTRUMENTATION,"derive explicit counter from capture deadline and window completion","per window","thread creation and OS latency","add and serialize predeclared lateness/deadline counters") ||
       add_observable(d,"sender_workload_mode",HOLO_OBSERVABLE_AVAILABLE_NOW,"predeclared sender schedule","per window","command is input context not physical state","") ||
       add_observable(d,"sender_core_identity",HOLO_OBSERVABLE_AVAILABLE_NOW,"affinity configuration","per run","migration if affinity fails","") ||
       add_observable(d,"receiver_core_identity",HOLO_OBSERVABLE_AVAILABLE_NOW,"affinity configuration","per run","migration if affinity fails","") ||
       add_observable(d,"TSC_relative_phase",HOLO_OBSERVABLE_AVAILABLE_NOW,"shared absolute TSC origin","per sample/window","clock calibration and phase wrap","") ||
       add_observable(d,"capture_window_index",HOLO_OBSERVABLE_AVAILABLE_NOW,"predeclared schedule","per window","time-index leakage risk","") ||
       add_observable(d,"temperature_proxy",HOLO_OBSERVABLE_MINOR_INSTRUMENTATION,"k10temp readout","before/after trial minimum;higher rate if safe","sensor quantization and lag","record timestamped k10temp without changing veto") ||
       add_observable(d,"frequency_pstate_proxy",HOLO_OBSERVABLE_MINOR_INSTRUMENTATION,"APERF/MPERF/COFVID and cpufreq policy","before/after each trial","counter sampling and policy drift","add read-only timestamped counters") ||
       add_observable(d,"rail_voltage_current",HOLO_OBSERVABLE_EXTERNAL_INSTRUMENTATION,"external rail probe and synchronized ADC","bandwidth must resolve input transitions","probe loading;ADC noise;clock alignment","external instrument with shared trigger and calibration") ||
       add_observable(d,"internal_pdn_modes",HOLO_OBSERVABLE_UNAVAILABLE,"no current direct instrument","not sampled","latent and unidentifiable directly","requires validated indirect model or unavailable on current platform")) return -4;

    if(add_operator(d,"O0_affine",HOLO_OPERATOR_AFFINE_LINEAR,"S0_minimal","x_next=A*x+B*u+c","A;B;c","training sessions and held-out validation/test sessions","passes G1-G10 and beats simpler baselines","held-out failure or unstable rollout",0) ||
       add_operator(d,"O1_route_conditioned",HOLO_OPERATOR_ROUTE_CONDITIONED_LINEAR,"S1_contextual","x_next=A_route*x+B_route*u+c_route","A_route;B_route;c_route","multi-route training and held-out sessions","passes G1-G10;route dependence reported","route parameters unstable or wrong-route control not informative",1) ||
       add_operator(d,"O2_bilinear",HOLO_OPERATOR_BILINEAR,"S2_delay_embedded","x_next=A*x+sum(u_j*N_j*x)+B*u+c","A;N_j;B;c","O0/O1 residual diagnosis plus held-out sessions","only after O0/O1 systematic failure;passes G1-G10","no material held-out gain after complexity penalty",2) ||
       add_operator(d,"O3_compact_nonlinear",HOLO_OPERATOR_COMPACT_NONLINEAR,"S2_delay_embedded","small polynomial,compact reservoir,kernel,or small MLP","bounded predeclared hyperparameters","nested validation and untouched held-out test sessions","simplest compact model passing G1-G10","capacity gain limited to training or opaque leakage",3)) return -5;

    if(add_calibration(d,"C0","instrument idle baseline","constant_idle","60 s blocks","3 blocks per session","noise floor;drift","raw_capture;calibration_metadata","baseline nonstationary beyond approved drift bound") ||
       add_calibration(d,"C1","carrier-off null","carrier_off","60 s blocks","3 blocks per route","null I/Q and ring period","raw_capture","carrier-off produces input-correlated response") ||
       add_calibration(d,"C2","static load response","constant_load","10 s per level","5 counterbalanced repeats","level response and saturation","raw_capture;calibration_metadata","levels indistinguishable or preparation fails") ||
       add_calibration(d,"C3","step response","step_input","fixed 5 s pre,5 s load,10 s recovery","30 repeats default;power-derived final","settling;overshoot;hysteresis","raw_capture","transition unresolved at sample rate") ||
       add_calibration(d,"C4","impulse response","structured_impulse","one window burst plus 32 recovery windows","64 trials default;power-derived final","memory kernel","raw_capture","response not distinguishable from sham") ||
       add_calibration(d,"C5","persistent excitation","prbs","1024 windows per seed","3 train seeds plus validation/test seeds","transition response","raw_capture;identified_operator","operator does not beat persistence") ||
       add_calibration(d,"C6","phase response","phase_shifted_modulation","12 tones;0.5 s fixed slots","power-derived repeats per phase","I/Q phase and amplitude","raw_capture;observability_analysis","phase response not repeatable") ||
       add_calibration(d,"C7","repeatability","binary_mode_modulation","256 windows","30 identical schedules across sessions","trajectory distance distribution","held_out_prediction","within-input spread overlaps between-input spread") ||
       add_calibration(d,"C8","order sensitivity","reordered_wrong_inverse","matched forward,reordered,wrong schedules","30 matched sets default","future-output trajectory","observability_analysis","order effect absent or below noise") ||
       add_calibration(d,"C9","wrong-route replay","binary_mode_modulation","256 windows","matched routes and seeds","route prediction error","held_out_prediction","route classification cannot be determined") ||
       add_calibration(d,"C10","session stability","prbs","1024 windows","3 train,1 validation,1 test session per route","parameter and prediction drift","gate_decision","held-out session performance collapses")) return -6;

    add_observability_test(d,"repeatability","within-schedule trajectory distance normalized by idle noise","95 percent within-input distance below 5th percentile between-input distance","pass repeatability or block","defines measured-trajectory repeatability only");
    add_observability_test(d,"state_distinguishability","common-probe future-output classifier with session holdout","balanced accuracy lower 95 percent CI above 0.60 and effect survives route/time controls","form observable equivalence classes or block","does not identify hidden semiconductor state");
    add_observability_test(d,"delay_embedding_gain","held-out NRMSE for S0,S1,S2 at L=1,2,4,8,16,32","selected L improves at least 10 percent over S1 with nested validation","select smallest qualifying L or retain S1","test set cannot select L");
    add_observability_test(d,"rank_surrogate","singular values of empirical perturbation Jacobian/observability matrix","rank fixed by predeclared noise-floor threshold; condition number reported","report local empirical rank only","not exact nonlinear observability");
    add_observability_test(d,"held_out_prediction","session-held-out one-step and 32-step NRMSE","at least 10 percent below best persistence/mean baseline with bootstrap 95 percent CI","predictively sufficient for tested family or block","adjacent windows are not independent splits");
    add_observability_test(d,"cross_route_generalization","held-out route NRMSE relative to route-specific model","within 15 percent=shared;otherwise route-family or route-specific","classify without forcing invariance","route identity remains explicit");
    add_observability_test(d,"time_stability","same/later session,temperature,P-state,reboot/day prediction degradation","no more than 15 percent degradation for claimed stability scope","restrict scope or block","temperature/day stages may require later campaign scheduling");

    if(add_gate(d,"G1","training NRMSE relative to input-only null","at least 10 percent lower under blocked cross-validation","training fit exceeds null") ||
       add_gate(d,"G2","held-out session one-step NRMSE","at least 10 percent below min(mean,persistence);bootstrap 95 percent CI excludes zero gain","held-out prediction accepted") ||
       add_gate(d,"G3","32-step rollout NRMSE and boundedness","beats persistence and never exceeds 5x observed training range","multi-step rollout stable") ||
       add_gate(d,"G4","residual correlation and leakage ablation","absolute correlation with time/input/timestamp leakage below 0.20;ablation gain persists","leakage not dominant") ||
       add_gate(d,"G5","parameter CV or prediction degradation across repeats","parameter CV <=0.25 or cross-run prediction degradation <=15 percent","repeat stability accepted") ||
       add_gate(d,"G6","route-specific and cross-route held-out errors","route scope classified;shared only within 15 percent of route-specific error","route dependence characterized") ||
       add_gate(d,"G7","block-bootstrap 95 percent intervals","intervals reported;empirical coverage >=0.90 on validation","uncertainty accepted") ||
       add_gate(d,"G8","shuffled-input held-out gain","no positive gain;permutation p>=0.10","shuffled control fails as expected") ||
       add_gate(d,"G9","wrong-route held-out degradation","at least 5 percent degradation or evidence supports shared classification","wrong-route behavior characterized") ||
       add_gate(d,"G10","forbidden serialized fields and source features","zero forbidden fields;zero hidden-label features","no-smuggle gate passes")) return -7;

    add_falsification(d,"F1","identical inputs are not repeatable","within-input versus between-input trajectory distance","separation criterion in repeatability test fails","improve preparation/instrumentation;do not identify operator");
    add_falsification(d,"F2","persistence matches or beats operators","held-out NRMSE","operator improvement below 10 percent or CI includes zero","retain baseline;block operator claim");
    add_falsification(d,"F3","prediction collapses across sessions","test-session NRMSE degradation","greater than 15 percent versus validation","model drift or restrict scope");
    add_falsification(d,"F4","parameters unstable across runs","parameter CV and predictive equivalence","CV above 0.25 with prediction degradation above 15 percent","recalibrate or choose route-conditioned model");
    add_falsification(d,"F5","route identity dominates predictions","route-only baseline versus dynamic operator","route-only within 5 percent of operator","classify route lookup;block transition operator");
    add_falsification(d,"F6","delay embedding gives no state reconstruction gain","nested-validation NRMSE across L","no L improves S1 by 10 percent","retain contextual state;report memory unresolved");
    add_falsification(d,"F7","prepared states indistinguishable","common-probe held-out balanced accuracy","lower 95 percent CI <=0.60","merge observable equivalence classes or block");
    add_falsification(d,"F8","residuals track drift over control","residual correlation with temperature/time/P-state","drift correlation exceeds controlled-input correlation","add instrumentation or block identifiability");
    add_falsification(d,"F9","state preparation not reproducible","baseline Mahalanobis distance","more than 5 percent trials outside predeclared 99 percent baseline envelope","reject trials and redesign preparation");
    add_falsification(d,"F10","instrumentation cannot resolve transitions","sample interval versus estimated settling time and SNR","fewer than 5 samples per settling time or SNR below 3","increase sampling/external instrumentation");

    add_artifact(d,"raw_capture","l4b5b_raw_capture_v1","run_id;route;input;seed;timestamps;I;Q;ring_period;window;preparation;topology");
    add_artifact(d,"calibration_metadata","l4b5b_calibration_metadata_v1","run_ids;instrument calibration;temperature;P-state;sample rate;state preparation;rejections");
    add_artifact(d,"identified_operator","l4b5b_identified_operator_v1","family;state_model;route;sessions;A;B;c;horizon;errors;uncertainty;identified=false until gates");
    add_artifact(d,"held_out_prediction","l4b5b_held_out_prediction_v1","operator_digest;test_run_ids;predictions;observations;baselines;errors;residuals");
    add_artifact(d,"observability_analysis","l4b5b_observability_analysis_v1","state_model;channels;history;rank_surrogate;conditioning;distinguishability;predictive_error;route/session stability;classification");
    add_artifact(d,"gate_decision","l4b5b_gate_decision_v1","design_digest;artifact_digests;G1-G10;F1-F10;decision;human_reviewer;claim_level");

    copy_text(d->splits.training_seeds,sizeof(d->splits.training_seeds),"train_1001;train_1002;train_1003");
    copy_text(d->splits.validation_seeds,sizeof(d->splits.validation_seeds),"validation_2001;validation_2002");
    copy_text(d->splits.test_seeds,sizeof(d->splits.test_seeds),"test_3001;test_3002");
    copy_text(d->splits.training_sessions,sizeof(d->splits.training_sessions),"v2s3_train_01;v2s3_train_02;v2s3_train_03;v4s5_train_01;v4s5_train_02;v4s5_train_03");
    copy_text(d->splits.validation_sessions,sizeof(d->splits.validation_sessions),"v2s3_validation_01;v4s5_validation_01");
    copy_text(d->splits.test_sessions,sizeof(d->splits.test_sessions),"v2s3_test_01;v4s5_test_01");
    copy_text(d->splits.route_holdout,sizeof(d->splits.route_holdout),"one additional route test-only if available;predeclare before outcomes;report unavailable otherwise");
    copy_text(d->splits.normalization_policy,sizeof(d->splits.normalization_policy),"fit normalization on training sessions only;freeze for validation/test");
    copy_text(d->splits.model_selection_policy,sizeof(d->splits.model_selection_policy),"select family and history on validation only;test set opened once after freeze");
    copy_text(d->splits.leakage_prevention,sizeof(d->splits.leakage_prevention),"session-level splits;no adjacent-window split;no future normalization;no success-conditioned route or seed selection");
    return holo_observability_design_validate(d) ? 0 : -8;
}

int holo_observability_design_equal(const HoloObservabilityDesign *a, const HoloObservabilityDesign *b) {
    return a && b && memcmp(a,b,sizeof(*a)) == 0;
}

int holo_observability_design_write_json(const HoloObservabilityDesign *d, const char *path) {
    FILE *f; size_t i;
    if(!d||!path||!d->sealed||!holo_observability_design_validate(d)) return -1;
    f=fopen(path,"w"); if(!f) return -2;
    fprintf(f,"{\n\"experiment_family\":\"%s\",\"design_id\":\"%s\",\"design_version\":\"%s\",\"status\":\"%s\",\"sealed\":true,\"design_digest\":\"%016" PRIx64 "\",\n",
        d->experiment_family,d->design_id,d->design_version,holo_experiment_design_status_name(d->status),d->design_digest);
    fprintf(f,"\"mapping_contract_id\":\"%s\",\"reviewed_mapping_digest\":\"%016" PRIx64 "\",\"claim_level\":%d,\"review_required\":%s,\"human_reviewed\":%s,\"implementation_authorized\":%s,\"executed\":%s,\"full_physical_observability_claimed\":%s,\"physical_restoration_claimed\":%s,\"next_gate\":\"%s\",\n",
        d->mapping_contract_id,d->reviewed_mapping_digest,d->claim_level,d->review_required?"true":"false",
        d->human_reviewed?"true":"false",d->implementation_authorized?"true":"false",d->executed?"true":"false",
        d->full_physical_observability_claimed?"true":"false",d->physical_restoration_claimed?"true":"false",d->next_gate);
    fprintf(f,"\"design_scope\":{\"state_space_model\":\"%s\",\"target_claim\":\"%s\",\"forbidden_claim\":\"%s\",\"topology_metadata\":\"%s\",\"design_bounds\":\"%s\",\"baselines\":\"%s\",\"input_policy\":\"%s\",\"observability_definitions\":\"%s\",\"state_preparation\":\"%s\"},\n",
        d->state_space_model,d->target_claim,d->forbidden_claim,d->topology_metadata,d->design_bounds,d->baselines,d->input_policy,d->observability_definitions,d->state_preparation);
    fprintf(f,"\"data_splits\":{\"training_seeds\":\"%s\",\"validation_seeds\":\"%s\",\"test_seeds\":\"%s\",\"training_sessions\":\"%s\",\"validation_sessions\":\"%s\",\"test_sessions\":\"%s\",\"route_holdout\":\"%s\",\"normalization_policy\":\"%s\",\"model_selection_policy\":\"%s\",\"leakage_prevention\":\"%s\"},\n",
        d->splits.training_seeds,d->splits.validation_seeds,d->splits.test_seeds,d->splits.training_sessions,
        d->splits.validation_sessions,d->splits.test_sessions,d->splits.route_holdout,d->splits.normalization_policy,
        d->splits.model_selection_policy,d->splits.leakage_prevention);
    fprintf(f,"\"state_models\":[\n");
    for(i=0;i<d->state_count;i++){const HoloPhysicalStateModel*r=&d->state_models[i];fprintf(f,"{\"model_id\":\"%s\",\"kind\":\"%s\",\"fields\":\"%s\",\"available_observables\":\"%s\",\"history_lengths\":\"%s\",\"maximum_history_length\":%d,\"purpose\":\"%s\",\"limitations\":\"%s\",\"requires_unavailable_observable\":%s,\"instrumentation_plan\":\"%s\"}%s\n",r->model_id,holo_state_model_kind_name(r->kind),r->fields,r->available_observables,r->history_lengths,r->maximum_history_length,r->purpose,r->limitations,r->requires_unavailable_observable?"true":"false",r->instrumentation_plan,i+1<d->state_count?",":"");}
    fprintf(f,"],\n\"input_families\":[\n");
    for(i=0;i<d->input_count;i++){const HoloInputFamily*r=&d->input_families[i];fprintf(f,"{\"input_id\":\"%s\",\"schedule\":\"%s\",\"seed_policy\":\"%s\",\"duration\":\"%s\",\"purpose\":\"%s\",\"null_control\":\"%s\"}%s\n",r->input_id,r->schedule,r->seed_policy,r->duration,r->purpose,r->null_control,i+1<d->input_count?",":"");}
    fprintf(f,"],\n\"observable_channels\":[\n");
    for(i=0;i<d->observable_count;i++){const HoloObservableChannel*r=&d->observables[i];fprintf(f,"{\"observable_id\":\"%s\",\"availability\":\"%s\",\"instrumentation_source\":\"%s\",\"sampling_constraints\":\"%s\",\"known_noise\":\"%s\",\"instrumentation_plan\":\"%s\"}%s\n",r->observable_id,holo_observable_availability_name(r->availability),r->instrumentation_source,r->sampling_constraints,r->known_noise,r->instrumentation_plan,i+1<d->observable_count?",":"");}
    fprintf(f,"],\n\"operator_candidates\":[\n");
    for(i=0;i<d->operator_count;i++){const HoloOperatorCandidate*r=&d->operators[i];fprintf(f,"{\"operator_id\":\"%s\",\"family\":\"%s\",\"state_model_id\":\"%s\",\"model\":\"%s\",\"parameters\":\"%s\",\"required_data\":\"%s\",\"acceptance_conditions\":\"%s\",\"rejection_conditions\":\"%s\",\"complexity_rank\":%d}%s\n",r->operator_id,holo_operator_family_name(r->family),r->state_model_id,r->model,r->parameters,r->required_data,r->acceptance_conditions,r->rejection_conditions,r->complexity_rank,i+1<d->operator_count?",":"");}
    fprintf(f,"],\n\"calibration_stages\":[\n");
    for(i=0;i<d->calibration_count;i++){const HoloCalibrationStage*r=&d->calibration[i];fprintf(f,"{\"stage_id\":\"%s\",\"purpose\":\"%s\",\"input_id\":\"%s\",\"duration\":\"%s\",\"trials\":\"%s\",\"sample_rate\":\"%s\",\"state_preparation\":\"%s\",\"expected_observable\":\"%s\",\"artifact_output\":\"%s\",\"failure_condition\":\"%s\"}%s\n",r->stage_id,r->purpose,r->input_id,r->duration,r->trials,r->sample_rate,r->state_preparation,r->expected_observable,r->artifact_output,r->failure_condition,i+1<d->calibration_count?",":"");}
    fprintf(f,"],\n\"observability_tests\":[\n");
    for(i=0;i<d->observability_test_count;i++){const HoloObservabilityTest*r=&d->observability_tests[i];fprintf(f,"{\"test_id\":\"%s\",\"metric\":\"%s\",\"threshold\":\"%s\",\"decision\":\"%s\",\"limitation\":\"%s\"}%s\n",r->test_id,r->metric,r->threshold,r->decision,r->limitation,i+1<d->observability_test_count?",":"");}
    fprintf(f,"],\n\"acceptance_gates\":[\n");
    for(i=0;i<d->gate_count;i++){const HoloOperatorAcceptanceGate*r=&d->gates[i];fprintf(f,"{\"gate_id\":\"%s\",\"metric\":\"%s\",\"threshold\":\"%s\",\"decision\":\"%s\",\"requires_human_approval\":%s}%s\n",r->gate_id,r->metric,r->threshold,r->decision,r->requires_human_approval?"true":"false",i+1<d->gate_count?",":"");}
    fprintf(f,"],\n\"falsification_conditions\":[\n");
    for(i=0;i<d->falsification_count;i++){const HoloFalsificationCondition*r=&d->falsifications[i];fprintf(f,"{\"condition_id\":\"%s\",\"condition\":\"%s\",\"metric\":\"%s\",\"threshold\":\"%s\",\"decision\":\"%s\",\"next_action\":\"%s\"}%s\n",r->condition_id,r->condition,r->metric,r->threshold,r->decision,r->next_action,i+1<d->falsification_count?",":"");}
    fprintf(f,"],\n\"artifact_contracts\":[\n");
    for(i=0;i<d->artifact_count;i++){const HoloFutureArtifactContract*r=&d->artifacts[i];fprintf(f,"{\"artifact_id\":\"%s\",\"schema_id\":\"%s\",\"required_fields\":\"%s\",\"digest_rule\":\"%s\",\"source_run_ids\":\"%s\",\"claim_level\":%d}%s\n",r->artifact_id,r->schema_id,r->required_fields,r->digest_rule,r->source_run_ids,r->claim_level,i+1<d->artifact_count?",":"");}
    fprintf(f,"]\n}\n");
    return fclose(f)==0?0:-3;
}

static char *read_all(const char *path) {
    FILE*f=fopen(path,"rb"); long n; char*s;
    if(!f||fseek(f,0,SEEK_END)||(n=ftell(f))<0||fseek(f,0,SEEK_SET)){if(f)fclose(f);return NULL;}
    s=(char*)malloc((size_t)n+1U);if(!s){fclose(f);return NULL;}
    if(fread(s,1,(size_t)n,f)!=(size_t)n){free(s);fclose(f);return NULL;}s[n]='\0';fclose(f);return s;
}

static const char *find_key(const char *start,const char *key){char needle[96];const char*p=start;snprintf(needle,sizeof(needle),"\"%s\"",key);while((p=strstr(p,needle))){const char*a=p+strlen(needle);while(*a==' '||*a=='\t')a++;if(*a==':')return a+1;p=a;}return NULL;}
static int parse_text(const char*start,const char*key,char*out,size_t size){const char*p=find_key(start,key),*e;if(!p)return 0;while(*p==' ')p++;if(*p++!='\"'||!(e=strchr(p,'\"'))||(size_t)(e-p)>=size)return 0;memcpy(out,p,(size_t)(e-p));out[e-p]='\0';return 1;}
static int parse_int(const char*s,const char*k,int*out){const char*p=find_key(s,k);return p&&sscanf(p,"%d",out)==1;}
static int parse_bool(const char*s,const char*k,int*out){const char*p=find_key(s,k);if(!p)return 0;while(*p==' ')p++;if(!strncmp(p,"true",4)){*out=1;return 1;}if(!strncmp(p,"false",5)){*out=0;return 1;}return 0;}
static int parse_hex(const char*s,const char*k,uint64_t*out){char t[24];return parse_text(s,k,t,sizeof(t))&&sscanf(t,"%16" SCNx64,out)==1;}
static HoloObservableAvailability parse_availability(const char*s){int i;for(i=0;i<=HOLO_OBSERVABLE_UNAVAILABLE;i++)if(!strcmp(s,holo_observable_availability_name((HoloObservableAvailability)i)))return(HoloObservableAvailability)i;return(HoloObservableAvailability)-1;}
static HoloPhysicalStateModelKind parse_state_kind(const char*s){int i;for(i=0;i<=HOLO_STATE_DELAY_EMBEDDED;i++)if(!strcmp(s,holo_state_model_kind_name((HoloPhysicalStateModelKind)i)))return(HoloPhysicalStateModelKind)i;return(HoloPhysicalStateModelKind)-1;}
static HoloOperatorFamily parse_operator_family(const char*s){int i;for(i=0;i<=HOLO_OPERATOR_COMPACT_NONLINEAR;i++)if(!strcmp(s,holo_operator_family_name((HoloOperatorFamily)i)))return(HoloOperatorFamily)i;return(HoloOperatorFamily)-1;}
static HoloExperimentDesignStatus parse_design_status(const char*s){int i;for(i=0;i<=HOLO_DESIGN_REJECTED;i++)if(!strcmp(s,holo_experiment_design_status_name((HoloExperimentDesignStatus)i)))return(HoloExperimentDesignStatus)i;return(HoloExperimentDesignStatus)-1;}

int holo_observability_design_read_json(HoloObservabilityDesign *d, const char *path) {
    char*json=read_all(path);char text[HOLO_DESIGN_TEXT_LEN];const char*p;uint64_t expected;int sealed;size_t i;
    if(!json||holo_observability_design_init(d)) {free(json);return -1;}
    if(!parse_text(json,"experiment_family",d->experiment_family,sizeof(d->experiment_family))||
       !parse_text(json,"design_id",d->design_id,sizeof(d->design_id))||!parse_text(json,"design_version",d->design_version,sizeof(d->design_version))||
       !parse_text(json,"status",text,sizeof(text))||!parse_bool(json,"sealed",&sealed)||!sealed||!parse_hex(json,"design_digest",&expected)||
       !parse_text(json,"mapping_contract_id",d->mapping_contract_id,sizeof(d->mapping_contract_id))||!parse_hex(json,"reviewed_mapping_digest",&d->reviewed_mapping_digest)||
       !parse_int(json,"claim_level",&d->claim_level)||!parse_bool(json,"review_required",&d->review_required)||!parse_bool(json,"human_reviewed",&d->human_reviewed)||
       !parse_bool(json,"implementation_authorized",&d->implementation_authorized)||!parse_bool(json,"executed",&d->executed)||
       !parse_bool(json,"full_physical_observability_claimed",&d->full_physical_observability_claimed)||!parse_bool(json,"physical_restoration_claimed",&d->physical_restoration_claimed)||
       !parse_text(json,"next_gate",d->next_gate,sizeof(d->next_gate))||!parse_text(json,"state_space_model",d->state_space_model,sizeof(d->state_space_model))||
       !parse_text(json,"target_claim",d->target_claim,sizeof(d->target_claim))||!parse_text(json,"forbidden_claim",d->forbidden_claim,sizeof(d->forbidden_claim))||
       !parse_text(json,"topology_metadata",d->topology_metadata,sizeof(d->topology_metadata))||!parse_text(json,"design_bounds",d->design_bounds,sizeof(d->design_bounds))||
       !parse_text(json,"baselines",d->baselines,sizeof(d->baselines))||!parse_text(json,"input_policy",d->input_policy,sizeof(d->input_policy))||
       !parse_text(json,"observability_definitions",d->observability_definitions,sizeof(d->observability_definitions))||
       !parse_text(json,"state_preparation",d->state_preparation,sizeof(d->state_preparation))) goto fail;
    d->status=parse_design_status(text);
    if(!parse_text(json,"training_seeds",d->splits.training_seeds,sizeof(d->splits.training_seeds))||
       !parse_text(json,"validation_seeds",d->splits.validation_seeds,sizeof(d->splits.validation_seeds))||
       !parse_text(json,"test_seeds",d->splits.test_seeds,sizeof(d->splits.test_seeds))||
       !parse_text(json,"training_sessions",d->splits.training_sessions,sizeof(d->splits.training_sessions))||
       !parse_text(json,"validation_sessions",d->splits.validation_sessions,sizeof(d->splits.validation_sessions))||
       !parse_text(json,"test_sessions",d->splits.test_sessions,sizeof(d->splits.test_sessions))||
       !parse_text(json,"route_holdout",d->splits.route_holdout,sizeof(d->splits.route_holdout))||
       !parse_text(json,"normalization_policy",d->splits.normalization_policy,sizeof(d->splits.normalization_policy))||
       !parse_text(json,"model_selection_policy",d->splits.model_selection_policy,sizeof(d->splits.model_selection_policy))||
       !parse_text(json,"leakage_prevention",d->splits.leakage_prevention,sizeof(d->splits.leakage_prevention))) goto fail;
    p=strstr(json,"\"state_models\""); if(!p) goto fail;
    for(i=0;i<HOLO_DESIGN_STATE_COUNT;i++) {
        HoloPhysicalStateModel r; memset(&r,0,sizeof(r)); p=strstr(p,"\"model_id\""); if(!p) goto fail;
        if(!parse_text(p,"model_id",r.model_id,sizeof(r.model_id))||!parse_text(p,"kind",text,sizeof(text))||!parse_text(p,"fields",r.fields,sizeof(r.fields))||
           !parse_text(p,"available_observables",r.available_observables,sizeof(r.available_observables))||!parse_text(p,"history_lengths",r.history_lengths,sizeof(r.history_lengths))||
           !parse_int(p,"maximum_history_length",&r.maximum_history_length)||!parse_text(p,"purpose",r.purpose,sizeof(r.purpose))||
           !parse_text(p,"limitations",r.limitations,sizeof(r.limitations))||!parse_bool(p,"requires_unavailable_observable",&r.requires_unavailable_observable)||
           !parse_text(p,"instrumentation_plan",r.instrumentation_plan,sizeof(r.instrumentation_plan))) goto fail;
        r.kind=parse_state_kind(text);
        if(holo_observability_design_register_state(d,&r)) goto fail;
        p+=strlen("\"model_id\"");
    }
    p=strstr(json,"\"input_families\""); if(!p) goto fail;
    for(i=0;i<HOLO_DESIGN_INPUT_COUNT;i++) {
        HoloInputFamily r; memset(&r,0,sizeof(r)); p=strstr(p,"\"input_id\""); if(!p) goto fail;
        if(!parse_text(p,"input_id",r.input_id,sizeof(r.input_id))||!parse_text(p,"schedule",r.schedule,sizeof(r.schedule))||!parse_text(p,"seed_policy",r.seed_policy,sizeof(r.seed_policy))||
           !parse_text(p,"duration",r.duration,sizeof(r.duration))||!parse_text(p,"purpose",r.purpose,sizeof(r.purpose))||!parse_text(p,"null_control",r.null_control,sizeof(r.null_control))||holo_observability_design_register_input(d,&r)) goto fail;
        p+=strlen("\"input_id\"");
    }
    p=strstr(json,"\"observable_channels\""); if(!p) goto fail;
    for(i=0;i<HOLO_DESIGN_OBSERVABLE_COUNT;i++) {
        HoloObservableChannel r; memset(&r,0,sizeof(r)); p=strstr(p,"\"observable_id\""); if(!p) goto fail;
        if(!parse_text(p,"observable_id",r.observable_id,sizeof(r.observable_id))||!parse_text(p,"availability",text,sizeof(text))||
           !parse_text(p,"instrumentation_source",r.instrumentation_source,sizeof(r.instrumentation_source))||!parse_text(p,"sampling_constraints",r.sampling_constraints,sizeof(r.sampling_constraints))||
           !parse_text(p,"known_noise",r.known_noise,sizeof(r.known_noise))||!parse_text(p,"instrumentation_plan",r.instrumentation_plan,sizeof(r.instrumentation_plan))) goto fail;
        r.availability=parse_availability(text);
        if(holo_observability_design_register_observable(d,&r)) goto fail;
        p+=strlen("\"observable_id\"");
    }
    p=strstr(json,"\"operator_candidates\""); if(!p) goto fail;
    for(i=0;i<HOLO_DESIGN_OPERATOR_COUNT;i++) {
        HoloOperatorCandidate r; memset(&r,0,sizeof(r)); p=strstr(p,"\"operator_id\""); if(!p) goto fail;
        if(!parse_text(p,"operator_id",r.operator_id,sizeof(r.operator_id))||!parse_text(p,"family",text,sizeof(text))||!parse_text(p,"state_model_id",r.state_model_id,sizeof(r.state_model_id))||
           !parse_text(p,"model",r.model,sizeof(r.model))||!parse_text(p,"parameters",r.parameters,sizeof(r.parameters))||!parse_text(p,"required_data",r.required_data,sizeof(r.required_data))||
           !parse_text(p,"acceptance_conditions",r.acceptance_conditions,sizeof(r.acceptance_conditions))||!parse_text(p,"rejection_conditions",r.rejection_conditions,sizeof(r.rejection_conditions))||
           !parse_int(p,"complexity_rank",&r.complexity_rank)) goto fail;
        r.family=parse_operator_family(text);
        if(holo_observability_design_register_operator(d,&r)) goto fail;
        p+=strlen("\"operator_id\"");
    }
    p=strstr(json,"\"calibration_stages\""); if(!p) goto fail;
    for(i=0;i<HOLO_DESIGN_CALIBRATION_COUNT;i++) {
        HoloCalibrationStage r; memset(&r,0,sizeof(r)); p=strstr(p,"\"stage_id\""); if(!p) goto fail;
        if(!parse_text(p,"stage_id",r.stage_id,sizeof(r.stage_id))||!parse_text(p,"purpose",r.purpose,sizeof(r.purpose))||!parse_text(p,"input_id",r.input_id,sizeof(r.input_id))||
           !parse_text(p,"duration",r.duration,sizeof(r.duration))||!parse_text(p,"trials",r.trials,sizeof(r.trials))||!parse_text(p,"sample_rate",r.sample_rate,sizeof(r.sample_rate))||
           !parse_text(p,"state_preparation",r.state_preparation,sizeof(r.state_preparation))||!parse_text(p,"expected_observable",r.expected_observable,sizeof(r.expected_observable))||
           !parse_text(p,"artifact_output",r.artifact_output,sizeof(r.artifact_output))||!parse_text(p,"failure_condition",r.failure_condition,sizeof(r.failure_condition))||holo_observability_design_register_calibration(d,&r)) goto fail;
        p+=strlen("\"stage_id\"");
    }
    p=strstr(json,"\"observability_tests\""); if(!p) goto fail;
    for(i=0;i<HOLO_DESIGN_OBSERVABILITY_TEST_COUNT;i++) {
        HoloObservabilityTest*r=&d->observability_tests[d->observability_test_count++]; memset(r,0,sizeof(*r)); p=strstr(p,"\"test_id\"");
        if(!p||!parse_text(p,"test_id",r->test_id,sizeof(r->test_id))||!parse_text(p,"metric",r->metric,sizeof(r->metric))||!parse_text(p,"threshold",r->threshold,sizeof(r->threshold))||!parse_text(p,"decision",r->decision,sizeof(r->decision))||!parse_text(p,"limitation",r->limitation,sizeof(r->limitation))) goto fail;
        p+=strlen("\"test_id\"");
    }
    p=strstr(json,"\"acceptance_gates\""); if(!p) goto fail;
    for(i=0;i<HOLO_DESIGN_GATE_COUNT;i++) {
        HoloOperatorAcceptanceGate r; memset(&r,0,sizeof(r)); p=strstr(p,"\"gate_id\""); if(!p) goto fail;
        if(!parse_text(p,"gate_id",r.gate_id,sizeof(r.gate_id))||!parse_text(p,"metric",r.metric,sizeof(r.metric))||!parse_text(p,"threshold",r.threshold,sizeof(r.threshold))||!parse_text(p,"decision",r.decision,sizeof(r.decision))||!parse_bool(p,"requires_human_approval",&r.requires_human_approval)||holo_observability_design_register_gate(d,&r)) goto fail;
        p+=strlen("\"gate_id\"");
    }
    p=strstr(json,"\"falsification_conditions\"");if(!p)goto fail;
    for(i=0;i<HOLO_DESIGN_FALSIFICATION_COUNT;i++){HoloFalsificationCondition*r=&d->falsifications[d->falsification_count++];memset(r,0,sizeof(*r));p=strstr(p,"\"condition_id\"");if(!p||!parse_text(p,"condition_id",r->condition_id,sizeof(r->condition_id))||!parse_text(p,"condition",r->condition,sizeof(r->condition))||!parse_text(p,"metric",r->metric,sizeof(r->metric))||!parse_text(p,"threshold",r->threshold,sizeof(r->threshold))||!parse_text(p,"decision",r->decision,sizeof(r->decision))||!parse_text(p,"next_action",r->next_action,sizeof(r->next_action)))goto fail;p+=strlen("\"condition_id\"");}
    p=strstr(json,"\"artifact_contracts\"");if(!p)goto fail;
    for(i=0;i<HOLO_DESIGN_ARTIFACT_COUNT;i++){HoloFutureArtifactContract*r=&d->artifacts[d->artifact_count++];memset(r,0,sizeof(*r));p=strstr(p,"\"artifact_id\"");if(!p||!parse_text(p,"artifact_id",r->artifact_id,sizeof(r->artifact_id))||!parse_text(p,"schema_id",r->schema_id,sizeof(r->schema_id))||!parse_text(p,"required_fields",r->required_fields,sizeof(r->required_fields))||!parse_text(p,"digest_rule",r->digest_rule,sizeof(r->digest_rule))||!parse_text(p,"source_run_ids",r->source_run_ids,sizeof(r->source_run_ids))||!parse_int(p,"claim_level",&r->claim_level))goto fail;p+=strlen("\"artifact_id\"");}
    if(holo_observability_design_seal(d)||d->design_digest!=expected)goto fail;
    free(json);return 0;
fail: free(json);holo_observability_design_destroy(d);return -2;
}

void holo_observability_design_reference_init(HoloObservabilityDesignReference *r) {
    if(!r) return;
    memset(r,0,sizeof(*r));copy_text(r->design_id,sizeof(r->design_id),"l4b5b0_observability_operator_v1");
    copy_text(r->design_version,sizeof(r->design_version),"1.0.0");copy_text(r->status,sizeof(r->status),"not_attached");
    copy_text(r->mapping_contract_id,sizeof(r->mapping_contract_id),"l4b5a_pdn_mapping_v1");r->mapping_contract_digest=HOLO_DESIGN_MAPPING_DIGEST;
    copy_text(r->design_reference,sizeof(r->design_reference),"none");r->claim_level=1;
}

int holo_observability_design_reference_attach(HoloObservabilityDesignReference *r,
                                                const HoloObservabilityDesign *d,
                                                const char *reference) {
    if(!r||!d||!reference||!d->sealed||!holo_observability_design_validate(d))return -1;
    holo_observability_design_reference_init(r);copy_text(r->status,sizeof(r->status),holo_experiment_design_status_name(d->status));
    copy_text(r->design_reference,sizeof(r->design_reference),reference);r->design_digest=d->design_digest;
    r->implementation_authorized=d->implementation_authorized;return holo_observability_design_reference_validate(r)?0:-2;
}

int holo_observability_design_reference_validate(const HoloObservabilityDesignReference *r) {
    if(!r||strcmp(r->design_id,"l4b5b0_observability_operator_v1")||strcmp(r->design_version,"1.0.0")||
       strcmp(r->mapping_contract_id,"l4b5a_pdn_mapping_v1")||r->mapping_contract_digest!=HOLO_DESIGN_MAPPING_DIGEST||
       !present(r->design_reference)||r->implementation_authorized||r->claim_level!=1)return 0;
    if(!strcmp(r->status,"not_attached"))return r->design_digest==0&&!strcmp(r->design_reference,"none");
    return !strcmp(r->status,"READY_FOR_HUMAN_REVIEW")&&r->design_digest!=0;
}
