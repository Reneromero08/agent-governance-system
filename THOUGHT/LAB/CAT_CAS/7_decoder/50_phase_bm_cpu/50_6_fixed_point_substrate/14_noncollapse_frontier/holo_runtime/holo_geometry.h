/* Executable geometric-memory schema for the L4B non-collapse runtime. */
#ifndef HOLO_GEOMETRY_H
#define HOLO_GEOMETRY_H

#include <stdint.h>
#include <stddef.h>
#include "holo_invariant_family.h"
#include "holo_physical_mapping.h"
#include "holo_observability_design.h"

#define HOLO_SCHEMA_FAMILY "CAT_CAS_HOLO_GEOMETRY"
#define HOLO_SCHEMA_VERSION "1.4.0"
#define HOLO_HYPOTHESIS "CATALYSIS_IS_THE_HOLOGRAM"
#define HOLO_DOCTRINE "NON_COLLAPSE_V1"
#define HOLO_TEXT_LEN 96
#define HOLO_ID_LEN 37

typedef enum {
    HOLO_NATIVE = 0,
    HOLO_HYBRID = 1,
    HOLO_MATERIALIZED_FALLBACK = 2
} HoloMaterializationMode;

typedef struct HoloPathHistory HoloPathHistory;
typedef struct OrbitState OrbitState;

typedef struct {
    char geometry_type[HOLO_TEXT_LEN];
    int basis_rank;
    double relation_basis[4];
    double coordinates[2];
    double neutral_reference[2];
    char geometry_status[HOLO_TEXT_LEN];
    char child_reference[HOLO_TEXT_LEN];
} HoloGeometry;

typedef struct {
    char carrier_type[HOLO_TEXT_LEN];
    double coordinates[2];
    double phase[2];
    char carrier_class[HOLO_TEXT_LEN];
    char substrate_status[HOLO_TEXT_LEN];
    char measurement_channel[HOLO_TEXT_LEN];
} HoloCarrier;

typedef struct {
    char operator_id[HOLO_TEXT_LEN];
    uint64_t operator_seed;
    int step_count;
    HoloPathHistory *path_history;
    int path_history_present;
    size_t path_history_count;
    size_t path_history_capacity;
    int path_history_appendable;
    int path_history_reversible;
    int path_history_sealed;
    int path_restoration_verified;
    int path_serialized_roundtrip;
    char continuation_status[HOLO_TEXT_LEN];
    char closure_status[HOLO_TEXT_LEN];
} HoloEvolution;

typedef struct {
    char projection_type[HOLO_TEXT_LEN];
    char operator_id[HOLO_TEXT_LEN];
    HoloMaterializationMode materialization_mode;
    char allowed_boundary[HOLO_TEXT_LEN];
    char output_family[HOLO_TEXT_LEN];
} HoloProjection;

typedef struct {
    char substrate_type[HOLO_TEXT_LEN];
    char pre_state_reference[HOLO_TEXT_LEN];
    char post_state_reference[HOLO_TEXT_LEN];
    char restored_state_reference[HOLO_TEXT_LEN];
    int restored;
    double restoration_metric;
    char closure_law[HOLO_TEXT_LEN];
    char evidence_level[HOLO_TEXT_LEN];
    char verification_scope[HOLO_TEXT_LEN];
    char failure_reason[HOLO_TEXT_LEN];
} CatalyticRestoration;

typedef struct {
    char boundary_id[HOLO_ID_LEN];
    char boundary_type[HOLO_TEXT_LEN];
    int step;
    char timestamp[32];
    int crossed;
    int projection_invoked;
    int invariant_extracted;
    char post_boundary_operations[HOLO_TEXT_LEN];
} HoloCollapseBoundary;

typedef struct {
    int schema_clean;
    int serialized_output_clean;
} ForbiddenFieldAudit;

typedef struct HoloObject {
    char schema_family[HOLO_TEXT_LEN];
    char schema_version[HOLO_TEXT_LEN];
    char hypothesis[HOLO_TEXT_LEN];
    char doctrine[HOLO_TEXT_LEN];
    char holo_id[HOLO_ID_LEN];
    uint64_t run_id;
    int N;
    HoloGeometry geometry;
    HoloCarrier carrier;
    HoloEvolution evolution;
    HoloProjection projection;
    HoloInvariantFamily invariant_family;
    HoloPhysicalMappingReference physical_mapping;
    HoloObservabilityDesignReference physical_experiment_design;
    CatalyticRestoration restoration;
    HoloCollapseBoundary collapse_boundary;
    ForbiddenFieldAudit audit;
    int claim_level;
} HoloObject;

const char *holo_materialization_mode_name(HoloMaterializationMode mode);
int holo_object_init(HoloObject *h, uint64_t run_id, int N, int lower, int mirror);
void holo_object_destroy(HoloObject *h);
int holo_geometry_render(const HoloGeometry *geometry, double rendered[2]);
void holo_set_carrier_phase(HoloObject *h, double phase_lower, double phase_mirror);
void holo_record_evolution(HoloObject *h, uint64_t seed, int steps,
                           double fold_even_sum, double fold_odd_sum);
int holo_replace_path_history(HoloObject *h, HoloPathHistory *replacement);
int holo_verify_software_restoration(HoloObject *h, const OrbitState *initial,
                                     const OrbitState *terminal,
                                     OrbitState *restored, int serialized_roundtrip);
int holo_attach_physical_mapping(HoloObject *h,
                                 const HoloPhysicalMappingContract *contract,
                                 const char *contract_reference);
int holo_attach_observability_design(HoloObject *h,
                                     const HoloObservabilityDesign *design,
                                     const char *design_reference);
void holo_set_materialization_mode(HoloObject *h, HoloMaterializationMode mode);
int holo_extract_invariant(HoloObject *h);
int holo_cross_boundary(HoloObject *h, int step);
int holo_validate(const HoloObject *h);
int holo_write_json(HoloObject *h, const char *path);
int holo_read_json(HoloObject *h, const char *path);

#endif
