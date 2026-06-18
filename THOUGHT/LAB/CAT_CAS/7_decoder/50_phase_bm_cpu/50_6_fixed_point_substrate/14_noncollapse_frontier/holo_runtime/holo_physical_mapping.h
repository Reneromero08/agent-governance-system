#ifndef HOLO_PHYSICAL_MAPPING_H
#define HOLO_PHYSICAL_MAPPING_H

#include <stddef.h>
#include <stdint.h>

#define HOLO_MAPPING_TEXT_LEN 320
#define HOLO_MAPPING_OBJECT_COUNT 8U
#define HOLO_MAPPING_INVARIANT_COUNT 8U

typedef enum {
    HOLO_EVIDENCE_INVALID = 0,
    HOLO_EVIDENCE_MEASURED,
    HOLO_EVIDENCE_RECOMPUTED_FROM_MEASURED,
    HOLO_EVIDENCE_SIMULATED,
    HOLO_EVIDENCE_SOFTWARE_ONLY,
    HOLO_EVIDENCE_INFERRED,
    HOLO_EVIDENCE_PROPOSED,
    HOLO_EVIDENCE_ABSENT
} HoloEvidenceClass;

typedef enum {
    HOLO_MAP_INVALID = 0,
    HOLO_MAP_SUPPORTED,
    HOLO_MAP_PARTIALLY_SUPPORTED,
    HOLO_MAP_UNSUPPORTED,
    HOLO_MAP_NOT_APPLICABLE
} HoloMappingStatus;

typedef enum {
    HOLO_OBSERVABILITY_INVALID = 0,
    HOLO_OBSERVABLE,
    HOLO_PARTIALLY_OBSERVABLE,
    HOLO_UNOBSERVABLE_WITH_CURRENT_INSTRUMENTS,
    HOLO_OBSERVABILITY_UNDEFINED
} HoloObservability;

typedef enum {
    HOLO_PORTABILITY_SOFTWARE_ONLY = 0,
    HOLO_PORTABILITY_PHYSICALLY_TESTABLE_NOW,
    HOLO_PORTABILITY_PHYSICALLY_TESTABLE_AFTER_CALIBRATION,
    HOLO_PORTABILITY_NO_CURRENT_PHYSICAL_MAPPING
} HoloInvariantPortabilityClass;

typedef enum {
    HOLO_MAPPING_GEOMETRY = 0,
    HOLO_MAPPING_CARRIER,
    HOLO_MAPPING_RELATION_BASIS,
    HOLO_MAPPING_EVOLUTION,
    HOLO_MAPPING_PATH_HISTORY,
    HOLO_MAPPING_RESTORATION,
    HOLO_MAPPING_COLLAPSE_BOUNDARY,
    HOLO_MAPPING_INVARIANT_FAMILY
} HoloMappingKind;

typedef enum {
    HOLO_CLAIM_SOFTWARE_ONLY = 0,
    HOLO_CLAIM_PHYSICAL_CHANNEL,
    HOLO_CLAIM_CANDIDATE_MAPPING,
    HOLO_CLAIM_PHYSICAL_RESTORATION
} HoloClaimScope;

typedef struct {
    HoloMappingKind kind;
    char software_object[HOLO_MAPPING_TEXT_LEN];
    char software_role[HOLO_MAPPING_TEXT_LEN];
    char proposed_physical_correspondent[HOLO_MAPPING_TEXT_LEN];
    char observable[HOLO_MAPPING_TEXT_LEN];
    HoloEvidenceClass evidence_class;
    HoloMappingStatus mapping_status;
    HoloObservability observability;
    HoloClaimScope claim_scope;
    char evidence_reference[HOLO_MAPPING_TEXT_LEN];
    char allowed_claim[HOLO_MAPPING_TEXT_LEN];
    char forbidden_claim[HOLO_MAPPING_TEXT_LEN];
    char missing_measurement[HOLO_MAPPING_TEXT_LEN];
    char falsification_condition[HOLO_MAPPING_TEXT_LEN];
    int claim_level;
    int operator_identification_complete;
    int restoration_observables_complete;
} HoloPhysicalMappingRecord;

typedef struct {
    char invariant_kind[HOLO_MAPPING_TEXT_LEN];
    HoloInvariantPortabilityClass portability;
    char evidence_reference[HOLO_MAPPING_TEXT_LEN];
    char promotion_requirement[HOLO_MAPPING_TEXT_LEN];
} HoloInvariantPortabilityRecord;

typedef struct {
    HoloPhysicalMappingRecord *records;
    size_t count;
    size_t capacity;
    HoloInvariantPortabilityRecord portability[HOLO_MAPPING_INVARIANT_COUNT];
    size_t portability_count;
    char contract_id[HOLO_MAPPING_TEXT_LEN];
    char contract_version[HOLO_MAPPING_TEXT_LEN];
    char status[HOLO_MAPPING_TEXT_LEN];
    char physical_state_vector[HOLO_MAPPING_TEXT_LEN];
    char measured_components[HOLO_MAPPING_TEXT_LEN];
    char unmeasured_components[HOLO_MAPPING_TEXT_LEN];
    char nuisance_variables[HOLO_MAPPING_TEXT_LEN];
    char restoration_required_components[HOLO_MAPPING_TEXT_LEN];
    char restoration_evidence_gate[HOLO_MAPPING_TEXT_LEN];
    char required_controls[HOLO_MAPPING_TEXT_LEN];
    int sealed;
    int reviewed;
    int claim_level;
    uint64_t contract_digest;
} HoloPhysicalMappingContract;

typedef struct {
    char contract_id[64];
    char contract_version[32];
    char status[64];
    char contract_reference[HOLO_MAPPING_TEXT_LEN];
    uint64_t contract_digest;
    int supported_records;
    int partial_records;
    int unsupported_records;
    int reviewed;
    int claim_level;
} HoloPhysicalMappingReference;

const char *holo_evidence_class_name(HoloEvidenceClass value);
const char *holo_mapping_status_name(HoloMappingStatus value);
const char *holo_observability_name(HoloObservability value);
const char *holo_portability_name(HoloInvariantPortabilityClass value);
const char *holo_mapping_kind_name(HoloMappingKind value);
int holo_physical_mapping_init(HoloPhysicalMappingContract *contract, size_t capacity);
void holo_physical_mapping_destroy(HoloPhysicalMappingContract *contract);
int holo_physical_mapping_register(HoloPhysicalMappingContract *contract,
                                   const HoloPhysicalMappingRecord *record);
int holo_physical_mapping_add_portability(HoloPhysicalMappingContract *contract,
                                          const HoloInvariantPortabilityRecord *record);
int holo_physical_mapping_record_validate(const HoloPhysicalMappingRecord *record);
int holo_physical_mapping_validate(const HoloPhysicalMappingContract *contract);
int holo_physical_mapping_seal(HoloPhysicalMappingContract *contract);
int holo_physical_mapping_populate_current(HoloPhysicalMappingContract *contract);
int holo_physical_mapping_equal(const HoloPhysicalMappingContract *left,
                                const HoloPhysicalMappingContract *right);
int holo_physical_mapping_write_json(const HoloPhysicalMappingContract *contract,
                                     const char *path);
int holo_physical_mapping_read_json(HoloPhysicalMappingContract *contract,
                                    const char *path);
void holo_physical_mapping_reference_init(HoloPhysicalMappingReference *reference);
int holo_physical_mapping_reference_attach(HoloPhysicalMappingReference *reference,
                                           const HoloPhysicalMappingContract *contract,
                                           const char *contract_reference);
int holo_physical_mapping_reference_validate(const HoloPhysicalMappingReference *reference);

#endif
