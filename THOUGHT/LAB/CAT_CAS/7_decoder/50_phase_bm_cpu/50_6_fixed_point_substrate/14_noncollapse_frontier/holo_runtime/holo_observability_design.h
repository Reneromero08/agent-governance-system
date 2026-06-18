#ifndef HOLO_OBSERVABILITY_DESIGN_H
#define HOLO_OBSERVABILITY_DESIGN_H

#include <stddef.h>
#include <stdint.h>

#define HOLO_DESIGN_TEXT_LEN 256
#define HOLO_DESIGN_STATE_COUNT 3U
#define HOLO_DESIGN_INPUT_COUNT 11U
#define HOLO_DESIGN_OBSERVABLE_COUNT 16U
#define HOLO_DESIGN_OPERATOR_COUNT 4U
#define HOLO_DESIGN_CALIBRATION_COUNT 11U
#define HOLO_DESIGN_OBSERVABILITY_TEST_COUNT 7U
#define HOLO_DESIGN_GATE_COUNT 10U
#define HOLO_DESIGN_FALSIFICATION_COUNT 10U
#define HOLO_DESIGN_ARTIFACT_COUNT 6U
#define HOLO_DESIGN_MAPPING_DIGEST UINT64_C(0x0d06f3c8b44f8c55)

typedef enum {
    HOLO_OBSERVABLE_AVAILABLE_NOW = 0,
    HOLO_OBSERVABLE_MINOR_INSTRUMENTATION,
    HOLO_OBSERVABLE_EXTERNAL_INSTRUMENTATION,
    HOLO_OBSERVABLE_UNAVAILABLE
} HoloObservableAvailability;

typedef enum {
    HOLO_STATE_MINIMAL = 0,
    HOLO_STATE_CONTEXTUAL,
    HOLO_STATE_DELAY_EMBEDDED
} HoloPhysicalStateModelKind;

typedef enum {
    HOLO_OPERATOR_AFFINE_LINEAR = 0,
    HOLO_OPERATOR_ROUTE_CONDITIONED_LINEAR,
    HOLO_OPERATOR_BILINEAR,
    HOLO_OPERATOR_COMPACT_NONLINEAR
} HoloOperatorFamily;

typedef enum {
    HOLO_DESIGN_DRAFT = 0,
    HOLO_DESIGN_READY_FOR_HUMAN_REVIEW,
    HOLO_DESIGN_ACCEPTED,
    HOLO_DESIGN_REJECTED
} HoloExperimentDesignStatus;

typedef struct {
    char model_id[64];
    HoloPhysicalStateModelKind kind;
    char fields[HOLO_DESIGN_TEXT_LEN];
    char available_observables[HOLO_DESIGN_TEXT_LEN];
    char history_lengths[HOLO_DESIGN_TEXT_LEN];
    int maximum_history_length;
    char purpose[HOLO_DESIGN_TEXT_LEN];
    char limitations[HOLO_DESIGN_TEXT_LEN];
    int requires_unavailable_observable;
    char instrumentation_plan[HOLO_DESIGN_TEXT_LEN];
} HoloPhysicalStateModel;

typedef struct {
    char input_id[64];
    char schedule[HOLO_DESIGN_TEXT_LEN];
    char seed_policy[HOLO_DESIGN_TEXT_LEN];
    char duration[HOLO_DESIGN_TEXT_LEN];
    char purpose[HOLO_DESIGN_TEXT_LEN];
    char null_control[HOLO_DESIGN_TEXT_LEN];
} HoloInputFamily;

typedef struct {
    char observable_id[64];
    HoloObservableAvailability availability;
    char instrumentation_source[HOLO_DESIGN_TEXT_LEN];
    char sampling_constraints[HOLO_DESIGN_TEXT_LEN];
    char known_noise[HOLO_DESIGN_TEXT_LEN];
    char instrumentation_plan[HOLO_DESIGN_TEXT_LEN];
} HoloObservableChannel;

typedef struct {
    char operator_id[64];
    HoloOperatorFamily family;
    char state_model_id[64];
    char model[HOLO_DESIGN_TEXT_LEN];
    char parameters[HOLO_DESIGN_TEXT_LEN];
    char required_data[HOLO_DESIGN_TEXT_LEN];
    char acceptance_conditions[HOLO_DESIGN_TEXT_LEN];
    char rejection_conditions[HOLO_DESIGN_TEXT_LEN];
    int complexity_rank;
} HoloOperatorCandidate;

typedef struct {
    char stage_id[32];
    char purpose[HOLO_DESIGN_TEXT_LEN];
    char input_id[64];
    char duration[HOLO_DESIGN_TEXT_LEN];
    char trials[HOLO_DESIGN_TEXT_LEN];
    char sample_rate[HOLO_DESIGN_TEXT_LEN];
    char state_preparation[HOLO_DESIGN_TEXT_LEN];
    char expected_observable[HOLO_DESIGN_TEXT_LEN];
    char artifact_output[HOLO_DESIGN_TEXT_LEN];
    char failure_condition[HOLO_DESIGN_TEXT_LEN];
} HoloCalibrationStage;

typedef struct {
    char test_id[64];
    char metric[HOLO_DESIGN_TEXT_LEN];
    char threshold[HOLO_DESIGN_TEXT_LEN];
    char decision[HOLO_DESIGN_TEXT_LEN];
    char limitation[HOLO_DESIGN_TEXT_LEN];
} HoloObservabilityTest;

typedef struct {
    char gate_id[16];
    char metric[HOLO_DESIGN_TEXT_LEN];
    char threshold[HOLO_DESIGN_TEXT_LEN];
    char decision[HOLO_DESIGN_TEXT_LEN];
    int requires_human_approval;
} HoloOperatorAcceptanceGate;

typedef struct {
    char condition_id[16];
    char condition[HOLO_DESIGN_TEXT_LEN];
    char metric[HOLO_DESIGN_TEXT_LEN];
    char threshold[HOLO_DESIGN_TEXT_LEN];
    char decision[HOLO_DESIGN_TEXT_LEN];
    char next_action[HOLO_DESIGN_TEXT_LEN];
} HoloFalsificationCondition;

typedef struct {
    char artifact_id[64];
    char schema_id[HOLO_DESIGN_TEXT_LEN];
    char required_fields[HOLO_DESIGN_TEXT_LEN];
    char digest_rule[HOLO_DESIGN_TEXT_LEN];
    char source_run_ids[HOLO_DESIGN_TEXT_LEN];
    int claim_level;
} HoloFutureArtifactContract;

typedef struct {
    char training_seeds[HOLO_DESIGN_TEXT_LEN];
    char validation_seeds[HOLO_DESIGN_TEXT_LEN];
    char test_seeds[HOLO_DESIGN_TEXT_LEN];
    char training_sessions[HOLO_DESIGN_TEXT_LEN];
    char validation_sessions[HOLO_DESIGN_TEXT_LEN];
    char test_sessions[HOLO_DESIGN_TEXT_LEN];
    char route_holdout[HOLO_DESIGN_TEXT_LEN];
    char normalization_policy[HOLO_DESIGN_TEXT_LEN];
    char model_selection_policy[HOLO_DESIGN_TEXT_LEN];
    char leakage_prevention[HOLO_DESIGN_TEXT_LEN];
} HoloDataSplitPlan;

typedef struct {
    HoloPhysicalStateModel state_models[HOLO_DESIGN_STATE_COUNT];
    HoloInputFamily input_families[HOLO_DESIGN_INPUT_COUNT];
    HoloObservableChannel observables[HOLO_DESIGN_OBSERVABLE_COUNT];
    HoloOperatorCandidate operators[HOLO_DESIGN_OPERATOR_COUNT];
    HoloCalibrationStage calibration[HOLO_DESIGN_CALIBRATION_COUNT];
    HoloObservabilityTest observability_tests[HOLO_DESIGN_OBSERVABILITY_TEST_COUNT];
    HoloOperatorAcceptanceGate gates[HOLO_DESIGN_GATE_COUNT];
    HoloFalsificationCondition falsifications[HOLO_DESIGN_FALSIFICATION_COUNT];
    HoloFutureArtifactContract artifacts[HOLO_DESIGN_ARTIFACT_COUNT];
    size_t state_count, input_count, observable_count, operator_count, calibration_count;
    size_t observability_test_count, gate_count, falsification_count, artifact_count;
    HoloDataSplitPlan splits;
    char design_id[64];
    char design_version[32];
    char experiment_family[96];
    char mapping_contract_id[64];
    uint64_t reviewed_mapping_digest;
    HoloExperimentDesignStatus status;
    char state_space_model[HOLO_DESIGN_TEXT_LEN];
    char target_claim[HOLO_DESIGN_TEXT_LEN];
    char forbidden_claim[HOLO_DESIGN_TEXT_LEN];
    char topology_metadata[HOLO_DESIGN_TEXT_LEN];
    char design_bounds[HOLO_DESIGN_TEXT_LEN];
    char baselines[HOLO_DESIGN_TEXT_LEN];
    char input_policy[HOLO_DESIGN_TEXT_LEN];
    char observability_definitions[HOLO_DESIGN_TEXT_LEN];
    char state_preparation[HOLO_DESIGN_TEXT_LEN];
    char next_gate[64];
    int review_required;
    int human_reviewed;
    int implementation_authorized;
    int executed;
    int full_physical_observability_claimed;
    int physical_restoration_claimed;
    int claim_level;
    int sealed;
    uint64_t design_digest;
} HoloObservabilityDesign;

typedef struct {
    char design_id[64];
    char design_version[32];
    char status[64];
    char mapping_contract_id[64];
    uint64_t mapping_contract_digest;
    char design_reference[HOLO_DESIGN_TEXT_LEN];
    uint64_t design_digest;
    int implementation_authorized;
    int claim_level;
} HoloObservabilityDesignReference;

const char *holo_observable_availability_name(HoloObservableAvailability value);
const char *holo_state_model_kind_name(HoloPhysicalStateModelKind value);
const char *holo_operator_family_name(HoloOperatorFamily value);
const char *holo_experiment_design_status_name(HoloExperimentDesignStatus value);
int holo_observability_design_init(HoloObservabilityDesign *design);
void holo_observability_design_destroy(HoloObservabilityDesign *design);
int holo_observability_design_register_state(HoloObservabilityDesign *design, const HoloPhysicalStateModel *record);
int holo_observability_design_register_input(HoloObservabilityDesign *design, const HoloInputFamily *record);
int holo_observability_design_register_observable(HoloObservabilityDesign *design, const HoloObservableChannel *record);
int holo_observability_design_register_operator(HoloObservabilityDesign *design, const HoloOperatorCandidate *record);
int holo_observability_design_register_calibration(HoloObservabilityDesign *design, const HoloCalibrationStage *record);
int holo_observability_design_register_gate(HoloObservabilityDesign *design, const HoloOperatorAcceptanceGate *record);
int holo_observability_design_populate_current(HoloObservabilityDesign *design);
int holo_observability_design_validate(const HoloObservabilityDesign *design);
int holo_observability_design_seal(HoloObservabilityDesign *design);
int holo_observability_design_equal(const HoloObservabilityDesign *left, const HoloObservabilityDesign *right);
int holo_observability_design_write_json(const HoloObservabilityDesign *design, const char *path);
int holo_observability_design_read_json(HoloObservabilityDesign *design, const char *path);
void holo_observability_design_reference_init(HoloObservabilityDesignReference *reference);
int holo_observability_design_reference_attach(HoloObservabilityDesignReference *reference,
                                                const HoloObservabilityDesign *design,
                                                const char *design_reference);
int holo_observability_design_reference_validate(const HoloObservabilityDesignReference *reference);

#endif
