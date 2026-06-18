#include "holo_physical_mapping.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
static const uint64_t FNV_PRIME = UINT64_C(1099511628211);
static const char *L4B5B_BLOCKED = "NOT_AUTHORIZED_EVIDENCE_MISSING";
static const char *HUMAN_PROJECT_OWNER = "human_project_owner";
static const char *REVIEW_SCOPE = "evidence classifications;support statuses;observability classifications;mapping claim ceilings;L4B.5B gate decision";

static void text_copy(char *dst, size_t size, const char *src) {
    size_t length;
    if (!dst || size == 0) return;
    if (!src) src = "";
    length = strlen(src);
    if (length >= size) length = size - 1U;
    memcpy(dst, src, length);
    dst[length] = '\0';
}

static int text_present(const char *text) { return text && text[0] != '\0'; }

static int contains_lower(const char *text, const char *needle) {
    char lowered[HOLO_MAPPING_TEXT_LEN];
    size_t i;
    if (!text || !needle) return 0;
    for (i = 0; i + 1U < sizeof(lowered) && text[i] != '\0'; ++i) {
        lowered[i] = (char)tolower((unsigned char)text[i]);
    }
    lowered[i] = '\0';
    return strstr(lowered, needle) != NULL;
}

const char *holo_evidence_class_name(HoloEvidenceClass value) {
    static const char *names[] = {"INVALID", "MEASURED", "RECOMPUTED_FROM_MEASURED",
        "SIMULATED", "SOFTWARE_ONLY", "INFERRED", "PROPOSED", "ABSENT"};
    return value >= HOLO_EVIDENCE_INVALID && value <= HOLO_EVIDENCE_ABSENT ? names[value] : "INVALID";
}

const char *holo_mapping_status_name(HoloMappingStatus value) {
    static const char *names[] = {"INVALID", "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE"};
    return value >= HOLO_MAP_INVALID && value <= HOLO_MAP_NOT_APPLICABLE ? names[value] : "INVALID";
}

const char *holo_observability_name(HoloObservability value) {
    static const char *names[] = {"INVALID", "OBSERVABLE", "PARTIALLY_OBSERVABLE",
        "UNOBSERVABLE_WITH_CURRENT_INSTRUMENTS", "UNDEFINED"};
    return value >= HOLO_OBSERVABILITY_INVALID && value <= HOLO_OBSERVABILITY_UNDEFINED ? names[value] : "INVALID";
}

const char *holo_portability_name(HoloInvariantPortabilityClass value) {
    static const char *names[] = {"SOFTWARE_ONLY", "PHYSICALLY_TESTABLE_NOW",
        "PHYSICALLY_TESTABLE_AFTER_CALIBRATION", "NO_CURRENT_PHYSICAL_MAPPING"};
    return value >= HOLO_PORTABILITY_SOFTWARE_ONLY &&
           value <= HOLO_PORTABILITY_NO_CURRENT_PHYSICAL_MAPPING ? names[value] : "INVALID";
}

const char *holo_mapping_kind_name(HoloMappingKind value) {
    static const char *names[] = {"HoloGeometry", "HoloCarrier", "RelationBasis",
        "HoloEvolution", "HoloPathHistory", "CatalyticRestoration",
        "HoloCollapseBoundary", "HoloInvariantFamily"};
    return value >= HOLO_MAPPING_GEOMETRY && value <= HOLO_MAPPING_INVARIANT_FAMILY ? names[value] : "INVALID";
}

const char *holo_review_status_name(HoloReviewStatus value) {
    static const char *names[] = {"UNREVIEWED", "ACCEPTED_AT_STATED_CLAIM_CEILING", "INVALIDATED"};
    return value >= HOLO_REVIEW_UNREVIEWED && value <= HOLO_REVIEW_INVALIDATED ? names[value] : "INVALID";
}

static const char *claim_scope_name(HoloClaimScope value) {
    static const char *names[] = {"SOFTWARE_ONLY", "PHYSICAL_CHANNEL", "CANDIDATE_MAPPING", "PHYSICAL_RESTORATION"};
    return value >= HOLO_CLAIM_SOFTWARE_ONLY && value <= HOLO_CLAIM_PHYSICAL_RESTORATION ? names[value] : "INVALID";
}

int holo_physical_mapping_init(HoloPhysicalMappingContract *contract, size_t capacity) {
    if (!contract || capacity == 0 || capacity > 64U || capacity > SIZE_MAX / sizeof(*contract->records)) return -1;
    memset(contract, 0, sizeof(*contract));
    contract->records = (HoloPhysicalMappingRecord *)calloc(capacity, sizeof(*contract->records));
    if (!contract->records) return -2;
    contract->capacity = capacity;
    text_copy(contract->contract_id, sizeof(contract->contract_id), "l4b5a_pdn_mapping_v1");
    text_copy(contract->contract_version, sizeof(contract->contract_version), "1.0.0");
    text_copy(contract->status, sizeof(contract->status), "evidence_gap_audited");
    text_copy(contract->l4b5b_decision, sizeof(contract->l4b5b_decision), L4B5B_BLOCKED);
    contract->claim_level = 1;
    return 0;
}

void holo_physical_mapping_destroy(HoloPhysicalMappingContract *contract) {
    if (!contract) return;
    free(contract->records);
    memset(contract, 0, sizeof(*contract));
}

int holo_physical_mapping_record_validate(const HoloPhysicalMappingRecord *r) {
    if (!r || r->kind < HOLO_MAPPING_GEOMETRY || r->kind > HOLO_MAPPING_INVARIANT_FAMILY ||
        !text_present(r->software_object) || !text_present(r->software_role) ||
        !text_present(r->proposed_physical_correspondent) || !text_present(r->observable) ||
        !text_present(r->evidence_reference) || !text_present(r->allowed_claim) ||
        !text_present(r->forbidden_claim) || !text_present(r->missing_measurement) ||
        !text_present(r->falsification_condition) || r->evidence_class == HOLO_EVIDENCE_INVALID ||
        r->mapping_status == HOLO_MAP_INVALID || r->observability == HOLO_OBSERVABILITY_INVALID ||
        r->claim_scope < HOLO_CLAIM_SOFTWARE_ONLY || r->claim_scope > HOLO_CLAIM_PHYSICAL_RESTORATION ||
        r->claim_level < 1 || r->claim_level > 2) return 0;
    if (r->mapping_status == HOLO_MAP_SUPPORTED &&
        r->evidence_class != HOLO_EVIDENCE_MEASURED &&
        r->evidence_class != HOLO_EVIDENCE_RECOMPUTED_FROM_MEASURED) return 0;
    if (r->evidence_class == HOLO_EVIDENCE_ABSENT &&
        (r->mapping_status == HOLO_MAP_SUPPORTED || r->mapping_status == HOLO_MAP_PARTIALLY_SUPPORTED)) return 0;
    if (r->evidence_class == HOLO_EVIDENCE_PROPOSED && r->mapping_status == HOLO_MAP_SUPPORTED) return 0;
    if (r->evidence_class == HOLO_EVIDENCE_SOFTWARE_ONLY &&
        (r->claim_scope != HOLO_CLAIM_SOFTWARE_ONLY || contains_lower(r->allowed_claim, "physical"))) return 0;
    if (r->mapping_status == HOLO_MAP_UNSUPPORTED &&
        (contains_lower(r->allowed_claim, "physical proof") || contains_lower(r->allowed_claim, "physical restoration proven"))) return 0;
    if (r->evidence_class == HOLO_EVIDENCE_PROPOSED && contains_lower(r->allowed_claim, "directly measured")) return 0;
    if (r->claim_scope == HOLO_CLAIM_PHYSICAL_CHANNEL &&
        r->evidence_class != HOLO_EVIDENCE_MEASURED &&
        r->evidence_class != HOLO_EVIDENCE_RECOMPUTED_FROM_MEASURED) return 0;
    if (r->kind == HOLO_MAPPING_RELATION_BASIS && r->mapping_status == HOLO_MAP_SUPPORTED &&
        !r->operator_identification_complete) return 0;
    if (r->kind == HOLO_MAPPING_RESTORATION && r->mapping_status == HOLO_MAP_SUPPORTED &&
        (r->observability != HOLO_OBSERVABLE || !r->restoration_observables_complete ||
         (r->evidence_class != HOLO_EVIDENCE_MEASURED &&
          r->evidence_class != HOLO_EVIDENCE_RECOMPUTED_FROM_MEASURED))) return 0;
    if (r->claim_scope == HOLO_CLAIM_PHYSICAL_RESTORATION &&
        (r->mapping_status != HOLO_MAP_SUPPORTED || r->observability != HOLO_OBSERVABLE ||
         !r->restoration_observables_complete)) return 0;
    return 1;
}

int holo_physical_mapping_register(HoloPhysicalMappingContract *contract,
                                   const HoloPhysicalMappingRecord *record) {
    if (!contract || !record || contract->sealed || contract->count >= contract->capacity ||
        !holo_physical_mapping_record_validate(record)) return -1;
    if ((size_t)record->kind != contract->count) return -2;
    contract->records[contract->count++] = *record;
    return 0;
}

int holo_physical_mapping_add_portability(HoloPhysicalMappingContract *contract,
                                          const HoloInvariantPortabilityRecord *record) {
    if (!contract || !record || contract->sealed ||
        contract->portability_count >= HOLO_MAPPING_INVARIANT_COUNT ||
        !text_present(record->invariant_kind) || !text_present(record->evidence_reference) ||
        !text_present(record->promotion_requirement) ||
        record->portability < HOLO_PORTABILITY_SOFTWARE_ONLY ||
        record->portability > HOLO_PORTABILITY_NO_CURRENT_PHYSICAL_MAPPING) return -1;
    contract->portability[contract->portability_count++] = *record;
    return 0;
}

static uint64_t fnv_text(uint64_t digest, const char *text) {
    while (*text != '\0') digest = (digest ^ (unsigned char)*text++) * FNV_PRIME;
    return digest;
}

static uint64_t fnv_u64(uint64_t digest, uint64_t value) {
    unsigned int shift;
    for (shift = 0; shift < 64U; shift += 8U) digest = (digest ^ (unsigned char)(value >> shift)) * FNV_PRIME;
    return digest;
}

static uint64_t mapping_digest(const HoloPhysicalMappingContract *contract) {
    uint64_t d = fnv_text(FNV_OFFSET, contract->contract_id);
    size_t i;
    d = fnv_text(d, contract->contract_version); d = fnv_text(d, contract->status);
    d = fnv_text(d, contract->physical_state_vector); d = fnv_text(d, contract->measured_components);
    d = fnv_text(d, contract->unmeasured_components); d = fnv_text(d, contract->nuisance_variables);
    d = fnv_text(d, contract->restoration_required_components);
    d = fnv_text(d, contract->restoration_evidence_gate); d = fnv_text(d, contract->required_controls);
    /* Preserve the frozen L4B.5A content digest; review metadata is an external envelope. */
    d = fnv_u64(d, (uint64_t)contract->claim_level); d = fnv_u64(d, 0);
    for (i = 0; i < contract->count; ++i) {
        const HoloPhysicalMappingRecord *r = &contract->records[i];
        d = fnv_u64(d, (uint64_t)r->kind); d = fnv_text(d, r->software_object);
        d = fnv_text(d, r->software_role); d = fnv_text(d, r->proposed_physical_correspondent);
        d = fnv_text(d, r->observable); d = fnv_u64(d, (uint64_t)r->evidence_class);
        d = fnv_u64(d, (uint64_t)r->mapping_status); d = fnv_u64(d, (uint64_t)r->observability);
        d = fnv_u64(d, (uint64_t)r->claim_scope); d = fnv_text(d, r->evidence_reference);
        d = fnv_text(d, r->allowed_claim); d = fnv_text(d, r->forbidden_claim);
        d = fnv_text(d, r->missing_measurement); d = fnv_text(d, r->falsification_condition);
        d = fnv_u64(d, (uint64_t)r->claim_level); d = fnv_u64(d, (uint64_t)r->operator_identification_complete);
        d = fnv_u64(d, (uint64_t)r->restoration_observables_complete);
    }
    for (i = 0; i < contract->portability_count; ++i) {
        const HoloInvariantPortabilityRecord *p = &contract->portability[i];
        d = fnv_text(d, p->invariant_kind); d = fnv_u64(d, (uint64_t)p->portability);
        d = fnv_text(d, p->evidence_reference); d = fnv_text(d, p->promotion_requirement);
    }
    return d;
}

uint64_t holo_physical_mapping_recompute_digest(const HoloPhysicalMappingContract *contract) {
    return contract ? mapping_digest(contract) : 0;
}

int holo_physical_mapping_review_valid(const HoloPhysicalMappingContract *contract) {
    const HoloContractReview *review;
    if (!contract) return 0;
    review = &contract->review;
    if (!review->reviewed) {
        return review->status == HOLO_REVIEW_UNREVIEWED && !review->human_review &&
               !text_present(review->reviewer_role) && review->reviewed_contract_digest == 0 &&
               !text_present(review->review_scope) && !contract->implementation_authorized;
    }
    return review->status == HOLO_REVIEW_ACCEPTED_AT_STATED_CLAIM_CEILING &&
           review->human_review && strcmp(review->reviewer_role, HUMAN_PROJECT_OWNER) == 0 &&
           strcmp(review->review_scope, REVIEW_SCOPE) == 0 &&
           review->reviewed_contract_digest == contract->contract_digest &&
           review->reviewed_contract_digest == mapping_digest(contract) &&
           !contract->implementation_authorized &&
           strcmp(contract->l4b5b_decision, L4B5B_BLOCKED) == 0;
}

int holo_physical_mapping_apply_human_review(HoloPhysicalMappingContract *contract,
                                             uint64_t reviewed_contract_digest) {
    HoloContractReview review;
    if (!contract || !contract->sealed || reviewed_contract_digest != HOLO_L4B5A_REVIEWED_DIGEST ||
        contract->contract_digest != reviewed_contract_digest ||
        mapping_digest(contract) != reviewed_contract_digest ||
        strcmp(contract->l4b5b_decision, L4B5B_BLOCKED) != 0 ||
        contract->implementation_authorized) return -1;
    memset(&review, 0, sizeof(review));
    review.status = HOLO_REVIEW_ACCEPTED_AT_STATED_CLAIM_CEILING;
    text_copy(review.reviewer_role, sizeof(review.reviewer_role), HUMAN_PROJECT_OWNER);
    review.reviewed_contract_digest = reviewed_contract_digest;
    text_copy(review.review_scope, sizeof(review.review_scope), REVIEW_SCOPE);
    review.reviewed = 1;
    review.human_review = 1;
    contract->review = review;
    return holo_physical_mapping_review_valid(contract) ? 0 : -2;
}

int holo_physical_mapping_validate(const HoloPhysicalMappingContract *contract) {
    size_t i;
    if (!contract || !contract->records || contract->count != HOLO_MAPPING_OBJECT_COUNT ||
        contract->portability_count != HOLO_MAPPING_INVARIANT_COUNT ||
        !text_present(contract->contract_id) || !text_present(contract->contract_version) ||
        !text_present(contract->status) || !text_present(contract->physical_state_vector) ||
        !text_present(contract->measured_components) || !text_present(contract->unmeasured_components) ||
        !text_present(contract->nuisance_variables) || !text_present(contract->restoration_required_components) ||
        !text_present(contract->restoration_evidence_gate) || !text_present(contract->required_controls) ||
        strcmp(contract->l4b5b_decision, L4B5B_BLOCKED) != 0 || contract->implementation_authorized ||
        contract->claim_level < 1 || contract->claim_level > 2) return 0;
    for (i = 0; i < contract->count; ++i) {
        if ((size_t)contract->records[i].kind != i || !holo_physical_mapping_record_validate(&contract->records[i])) return 0;
    }
    for (i = 0; i < contract->portability_count; ++i) {
        if (!text_present(contract->portability[i].invariant_kind) ||
            !text_present(contract->portability[i].evidence_reference) ||
            !text_present(contract->portability[i].promotion_requirement)) return 0;
    }
    if (contract->sealed && contract->contract_digest != mapping_digest(contract)) return 0;
    if (!holo_physical_mapping_review_valid(contract)) return 0;
    return 1;
}

int holo_physical_mapping_seal(HoloPhysicalMappingContract *contract) {
    if (!contract || contract->sealed || !holo_physical_mapping_validate(contract)) return -1;
    contract->sealed = 1;
    contract->contract_digest = mapping_digest(contract);
    return 0;
}

static void set_record(HoloPhysicalMappingRecord *r, HoloMappingKind kind,
                       const char *role, const char *physical, const char *observable,
                       HoloEvidenceClass evidence, HoloMappingStatus status,
                       HoloObservability observability, HoloClaimScope scope,
                       const char *reference, const char *allowed, const char *forbidden,
                       const char *missing, const char *falsification) {
    memset(r, 0, sizeof(*r)); r->kind = kind;
    text_copy(r->software_object, sizeof(r->software_object), holo_mapping_kind_name(kind));
    text_copy(r->software_role, sizeof(r->software_role), role);
    text_copy(r->proposed_physical_correspondent, sizeof(r->proposed_physical_correspondent), physical);
    text_copy(r->observable, sizeof(r->observable), observable);
    r->evidence_class = evidence; r->mapping_status = status; r->observability = observability; r->claim_scope = scope;
    text_copy(r->evidence_reference, sizeof(r->evidence_reference), reference);
    text_copy(r->allowed_claim, sizeof(r->allowed_claim), allowed);
    text_copy(r->forbidden_claim, sizeof(r->forbidden_claim), forbidden);
    text_copy(r->missing_measurement, sizeof(r->missing_measurement), missing);
    text_copy(r->falsification_condition, sizeof(r->falsification_condition), falsification);
    r->claim_level = 1;
}

static int add_portability(HoloPhysicalMappingContract *c, const char *kind,
                           HoloInvariantPortabilityClass portability, const char *requirement) {
    HoloInvariantPortabilityRecord r;
    memset(&r, 0, sizeof(r)); text_copy(r.invariant_kind, sizeof(r.invariant_kind), kind);
    r.portability = portability;
    text_copy(r.evidence_reference, sizeof(r.evidence_reference),
              "14_noncollapse_frontier/holo_runtime/HOLO_SCHEMA.md#l4b4-non-collapse-invariant-family");
    text_copy(r.promotion_requirement, sizeof(r.promotion_requirement), requirement);
    return holo_physical_mapping_add_portability(c, &r);
}

int holo_physical_mapping_populate_current(HoloPhysicalMappingContract *c) {
    HoloPhysicalMappingRecord r;
    const char *report = "12_chiral_lane_frontier/pdn_slot2_t300/PHASE6_SLOT2_PDN_T300_REPORT.md";
    if (!c || c->count != 0 || c->portability_count != 0) return -1;
    text_copy(c->physical_state_vector, sizeof(c->physical_state_vector),
              "X_phys(t)={lockin_I,lockin_Q,ring_osc_period,sender_schedule,receiver_core,sender_cores,TSC_origin,temperature_proxy,voltage_frequency_state,capture_window}");
    text_copy(c->measured_components, sizeof(c->measured_components),
              "lockin_I;lockin_Q;ring_osc_period samples;sender schedule;core ids;TSC origin;capture window;raw samples not preserved in git");
    text_copy(c->unmeasured_components, sizeof(c->unmeasured_components),
              "rail voltage/current waveform;internal PDN modes;full thermal field;temperature trajectory;voltage/frequency trajectory;complete microarchitectural and allocator state");
    text_copy(c->nuisance_variables, sizeof(c->nuisance_variables),
              "thermal drift;OS jitter;P-state drift;session drift;core-pair route;instrument noise");
    text_copy(c->restoration_required_components, sizeof(c->restoration_required_components),
              "pre terminal restored measurements of the declared observable state;environmental covariates;uncertainty bounds");
    text_copy(c->restoration_evidence_gate, sizeof(c->restoration_evidence_gate),
              "P0 declare state/observable;P1 baseline;P2 disturbance path;P3 terminal;P4 inverse/closure;P5 restored;P6 predeclared comparison;P7 controls;P8 repeat seeds sessions pairs");
    text_copy(c->required_controls, sizeof(c->required_controls),
              "no-disturbance;disturbance-without-restoration;wrong-inverse;reordered-inverse;carrier-off;phase-randomized;session-repeat;core-pair-repeat;thermal-time-matched-sham");

    set_record(&r, HOLO_MAPPING_GEOMETRY, "relational basis and unresolved coordinates",
        "candidate PDN coupling graph or calibrated phase-response manifold", "no measured physical geometry storage observable",
        HOLO_EVIDENCE_PROPOSED, HOLO_MAP_UNSUPPORTED, HOLO_OBSERVABILITY_UNDEFINED, HOLO_CLAIM_CANDIDATE_MAPPING,
        report, "PDN topology is a candidate carrier geometry only", "physical HoloGeometry memory demonstrated",
        "state-bearing geometry measurement and native physical render operation", "candidate topology fails to predict held-out relational responses");
    if (holo_physical_mapping_register(c, &r) != 0) return -2;
    set_record(&r, HOLO_MAPPING_CARRIER, "phase and measurement carrier",
        "shared package PDN driven by sender alu_burst and read by receiver lock-in", "I/Q vectors;ring-osc timing;phase correlation;mode accuracy",
        HOLO_EVIDENCE_MEASURED, HOLO_MAP_SUPPORTED, HOLO_PARTIALLY_OBSERVABLE, HOLO_CLAIM_PHYSICAL_CHANNEL,
        "12_chiral_lane_frontier/pdn_slot2_t300/results/result_slot2_pdn_t300.json", "cross-core PDN channel carries sender-owned mode and relational phase on route 4:5", "carrier stores full .holo geometry or recovers orientation",
        "direct rail waveform and broader route calibration", "route 4:5 fails predeclared multi-seed channel gates against silent and scramble controls");
    if (holo_physical_mapping_register(c, &r) != 0) return -3;
    set_record(&r, HOLO_MAPPING_RELATION_BASIS, "operator acting on unresolved coordinates",
        "identified multi-input multi-output PDN transfer operator", "complex lock-in response under repeated calibrated perturbations",
        HOLO_EVIDENCE_PROPOSED, HOLO_MAP_UNSUPPORTED, HOLO_PARTIALLY_OBSERVABLE, HOLO_CLAIM_CANDIDATE_MAPPING,
        "12_chiral_lane_frontier/pdn_slot2_t300/results/aggregate_t300.json", "core-pair response differences motivate operator identification", "existing pair differences constitute a physical relation basis",
        "calibrated transfer matrix;stability;uncertainty;held-out prediction;inverse or involution test", "identified operator fails held-out response prediction or session stability");
    if (holo_physical_mapping_register(c, &r) != 0) return -4;
    set_record(&r, HOLO_MAPPING_EVOLUTION, "declared ordered operator path",
        "sender workload schedule plus deadline-aligned PDN capture sequence", "ordered command schedule and per-slot I/Q capture windows",
        HOLO_EVIDENCE_MEASURED, HOLO_MAP_PARTIALLY_SUPPORTED, HOLO_PARTIALLY_OBSERVABLE, HOLO_CLAIM_CANDIDATE_MAPPING,
        "10_cross_core_wormhole/slot2_pdn/slot2_pdn_lockin.c", "controlled schedule and ordered readout windows were executed", "control log is a measured reversible physical trajectory",
        "calibrated pre/post state at every step and repeatable physical transition law", "step order does not predict measured state transitions beyond command timing");
    if (holo_physical_mapping_register(c, &r) != 0) return -5;
    set_record(&r, HOLO_MAPPING_PATH_HISTORY, "reversible transition witness",
        "serialized ordered measured physical state transitions", "none sufficient for physical reversal in preserved artifacts",
        HOLO_EVIDENCE_ABSENT, HOLO_MAP_UNSUPPORTED, HOLO_UNOBSERVABLE_WITH_CURRENT_INSTRUMENTS, HOLO_CLAIM_CANDIDATE_MAPPING,
        "10_cross_core_wormhole/slot2_pdn/slot2_pdn_analyze.py", "software path history has no demonstrated physical equivalent", "command schedule or compact score summary is physical path geometry",
        "pre-state;operator;post-state;continuity;environment;inverse;uncertainty per step", "a recorded path cannot reproduce or predict its own measured transitions");
    if (holo_physical_mapping_register(c, &r) != 0) return -6;
    set_record(&r, HOLO_MAPPING_RESTORATION, "inverse path reconstructs declared software state",
        "return of a declared observable physical state after inverse or catalytic closure", "no pre-terminal-restored physical state triple exists",
        HOLO_EVIDENCE_ABSENT, HOLO_MAP_UNSUPPORTED, HOLO_UNOBSERVABLE_WITH_CURRENT_INSTRUMENTS, HOLO_CLAIM_CANDIDATE_MAPPING,
        "12_chiral_lane_frontier/pdn_slot2_t300/results/result_slot2_pdn_t300.json", "hash restoration and P-state cleanup are software/protocol evidence only", "physical catalytic restoration or full-state return",
        "pre/terminal/restored measurements;metric;repeatability;nulls;wrong inverse;observability argument", "restored-state distance is not distinguishable from wrong-inverse and sham controls");
    if (holo_physical_mapping_register(c, &r) != 0) return -7;
    set_record(&r, HOLO_MAPPING_COLLAPSE_BOUNDARY, "explicit final projection event",
        "deadline-bounded ring-osc capture followed by lock-in I/Q projection", "capture window and computed I/Q readout",
        HOLO_EVIDENCE_MEASURED, HOLO_MAP_PARTIALLY_SUPPORTED, HOLO_PARTIALLY_OBSERVABLE, HOLO_CLAIM_PHYSICAL_CHANNEL,
        "10_cross_core_wormhole/slot2_pdn/slot2_pdn_lockin.c", "lock-in capture is an explicit measurement boundary for the channel observable", "ordinary logging is physical collapse or observes full substrate state",
        "predeclared information partition and observability analysis", "boundary output changes under post-window logging alone or leaks future schedule state");
    if (holo_physical_mapping_register(c, &r) != 0) return -8;
    set_record(&r, HOLO_MAPPING_INVARIANT_FAMILY, "predeclared preserved software relations",
        "mixed family of calibrated physical conservation/covariance/closure tests", "measured carrier phase plus recomputed channel statistics only",
        HOLO_EVIDENCE_RECOMPUTED_FROM_MEASURED, HOLO_MAP_PARTIALLY_SUPPORTED, HOLO_PARTIALLY_OBSERVABLE, HOLO_CLAIM_CANDIDATE_MAPPING,
        "12_chiral_lane_frontier/pdn_slot2_t300/results/aggregate_t300.json", "physical portability is classified per invariant and not promoted as a family", "software invariant names establish physical invariants",
        "per-invariant calibration and experiments listed by portability records", "calibrated invariant fails repeatability or controls");
    if (holo_physical_mapping_register(c, &r) != 0) return -9;

    if (add_portability(c, "orbit_conservation", HOLO_PORTABILITY_NO_CURRENT_PHYSICAL_MAPPING, "declare and measure a conserved physical orbit state") != 0 ||
        add_portability(c, "relation_basis", HOLO_PORTABILITY_PHYSICALLY_TESTABLE_AFTER_CALIBRATION, "identify stable transfer operator with held-out prediction") != 0 ||
        add_portability(c, "path_composition", HOLO_PORTABILITY_NO_CURRENT_PHYSICAL_MAPPING, "measure forward and inverse trajectories") != 0 ||
        add_portability(c, "restoration_closure", HOLO_PORTABILITY_NO_CURRENT_PHYSICAL_MAPPING, "pass physical restoration evidence gate") != 0 ||
        add_portability(c, "exchange_covariance", HOLO_PORTABILITY_PHYSICALLY_TESTABLE_AFTER_CALIBRATION, "calibrate structural exchange without branch labels") != 0 ||
        add_portability(c, "serialization_invariance", HOLO_PORTABILITY_SOFTWARE_ONLY, "not applicable to physical substrate state") != 0 ||
        add_portability(c, "path_order", HOLO_PORTABILITY_PHYSICALLY_TESTABLE_AFTER_CALIBRATION, "record ordered physical states and permuted control") != 0 ||
        add_portability(c, "software_path_holonomy", HOLO_PORTABILITY_NO_CURRENT_PHYSICAL_MAPPING, "define group-valued carrier transforms and closed-loop measurement") != 0) return -10;
    return 0;
}

int holo_physical_mapping_equal(const HoloPhysicalMappingContract *a,
                                const HoloPhysicalMappingContract *b) {
    if (!a || !b || a->count != b->count || a->capacity != b->capacity ||
        a->portability_count != b->portability_count || a->sealed != b->sealed ||
        a->claim_level != b->claim_level || a->implementation_authorized != b->implementation_authorized ||
        a->contract_digest != b->contract_digest ||
        memcmp(&a->review, &b->review, sizeof(a->review)) != 0 ||
        memcmp(a->records, b->records, a->count * sizeof(*a->records)) != 0 ||
        memcmp(a->portability, b->portability, sizeof(a->portability)) != 0) return 0;
    return strcmp(a->contract_id, b->contract_id) == 0 && strcmp(a->contract_version, b->contract_version) == 0 &&
        strcmp(a->status, b->status) == 0 && strcmp(a->physical_state_vector, b->physical_state_vector) == 0 &&
        strcmp(a->measured_components, b->measured_components) == 0 && strcmp(a->unmeasured_components, b->unmeasured_components) == 0 &&
        strcmp(a->nuisance_variables, b->nuisance_variables) == 0 &&
        strcmp(a->restoration_required_components, b->restoration_required_components) == 0 &&
        strcmp(a->restoration_evidence_gate, b->restoration_evidence_gate) == 0 &&
        strcmp(a->required_controls, b->required_controls) == 0 &&
        strcmp(a->l4b5b_decision, b->l4b5b_decision) == 0;
}

static void write_record(FILE *f, const HoloPhysicalMappingRecord *r, int comma) {
    fprintf(f, "    {\"mapping_kind\":%d,\"software_object\":\"%s\",\"software_role\":\"%s\",\"physical_correspondent\":\"%s\",\"observable\":\"%s\",\"evidence_class\":\"%s\",\"mapping_status\":\"%s\",\"observability\":\"%s\",\"claim_scope\":\"%s\",\"evidence_reference\":\"%s\",\"allowed_claim\":\"%s\",\"forbidden_claim\":\"%s\",\"missing_measurement\":\"%s\",\"falsification_condition\":\"%s\",\"claim_level\":%d,\"operator_identification_complete\":%s,\"restoration_observables_complete\":%s}%s\n",
        (int)r->kind, r->software_object, r->software_role, r->proposed_physical_correspondent, r->observable,
        holo_evidence_class_name(r->evidence_class), holo_mapping_status_name(r->mapping_status),
        holo_observability_name(r->observability), claim_scope_name(r->claim_scope), r->evidence_reference,
        r->allowed_claim, r->forbidden_claim, r->missing_measurement, r->falsification_condition,
        r->claim_level, r->operator_identification_complete ? "true" : "false",
        r->restoration_observables_complete ? "true" : "false", comma ? "," : "");
}

int holo_physical_mapping_write_json(const HoloPhysicalMappingContract *c, const char *path) {
    FILE *f; size_t i;
    if (!c || !path || !c->sealed || !holo_physical_mapping_validate(c)) return -1;
    f = fopen(path, "w"); if (!f) return -2;
    fprintf(f, "{\n  \"contract_id\": \"%s\",\n  \"contract_version\": \"%s\",\n  \"status\": \"%s\",\n  \"sealed\": true, \"claim_level\": %d,\n  \"contract_digest\": \"%016" PRIx64 "\",\n",
        c->contract_id, c->contract_version, c->status, c->claim_level, c->contract_digest);
    fprintf(f, "  \"review\": {\"reviewed\":%s,\"human_review\":%s,\"reviewer_role\":\"%s\",\"reviewed_contract_digest\":\"%016" PRIx64 "\",\"review_status\":\"%s\",\"review_scope\":\"%s\",\"review_valid\":%s,\"implementation_authorized\":%s},\n",
        c->review.reviewed ? "true" : "false", c->review.human_review ? "true" : "false",
        c->review.reviewer_role, c->review.reviewed_contract_digest,
        holo_review_status_name(c->review.status), c->review.review_scope,
        holo_physical_mapping_review_valid(c) ? "true" : "false",
        c->implementation_authorized ? "true" : "false");
    fprintf(f, "  \"physical_state_definition\": {\"state_vector\":\"%s\",\"measured\":\"%s\",\"unmeasured\":\"%s\",\"nuisance\":\"%s\",\"required_for_restoration\":\"%s\"},\n",
        c->physical_state_vector, c->measured_components, c->unmeasured_components, c->nuisance_variables, c->restoration_required_components);
    fprintf(f, "  \"restoration_evidence_gate\": {\"protocol\":\"%s\",\"controls\":\"%s\"},\n",
        c->restoration_evidence_gate, c->required_controls);
    fprintf(f, "  \"l4b5b_gate\": {\"decision\":\"%s\",\"implementation_authorized\":%s},\n",
        c->l4b5b_decision, c->implementation_authorized ? "true" : "false");
    fprintf(f, "  \"mapping_records\": [\n");
    for (i = 0; i < c->count; ++i) write_record(f, &c->records[i], i + 1U < c->count);
    fprintf(f, "  ],\n  \"invariant_portability\": [\n");
    for (i = 0; i < c->portability_count; ++i) {
        const HoloInvariantPortabilityRecord *p = &c->portability[i];
        fprintf(f, "    {\"invariant_kind\":\"%s\",\"portability\":\"%s\",\"evidence_reference\":\"%s\",\"promotion_requirement\":\"%s\"}%s\n",
            p->invariant_kind, holo_portability_name(p->portability), p->evidence_reference,
            p->promotion_requirement, i + 1U < c->portability_count ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    if (fclose(f) != 0) return -3;
    return 0;
}

static char *read_all(const char *path) {
    FILE *f = fopen(path, "rb"); long size; char *text;
    if (!f || fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) { if (f) fclose(f); return NULL; }
    text = (char *)malloc((size_t)size + 1U); if (!text) { fclose(f); return NULL; }
    if (fread(text, 1, (size_t)size, f) != (size_t)size) { free(text); fclose(f); return NULL; }
    text[size] = '\0'; fclose(f); return text;
}

static const char *find_key(const char *start, const char *key) {
    char needle[96]; const char *p = start; snprintf(needle, sizeof(needle), "\"%s\"", key);
    while ((p = strstr(p, needle)) != NULL) { const char *a = p + strlen(needle); while (*a == ' ' || *a == '\t') ++a; if (*a == ':') return a + 1; p = a; }
    return NULL;
}

static int parse_text(const char *start, const char *key, char *out, size_t size) {
    const char *p = find_key(start, key), *end; if (!p) return 0; while (*p == ' ' || *p == '\t') ++p;
    if (*p++ != '"' || !(end = strchr(p, '"')) || (size_t)(end - p) >= size) return 0;
    memcpy(out, p, (size_t)(end - p)); out[end - p] = '\0'; return 1;
}

static int parse_int(const char *start, const char *key, int *out) { const char *p = find_key(start, key); return p && sscanf(p, "%d", out) == 1; }
static int parse_bool(const char *start, const char *key, int *out) { const char *p = find_key(start, key); if (!p) return 0; while (*p == ' ') ++p; if (!strncmp(p,"true",4)){*out=1;return 1;} if(!strncmp(p,"false",5)){*out=0;return 1;} return 0; }
static int parse_hex(const char *start, const char *key, uint64_t *out) { char text[24]; return parse_text(start,key,text,sizeof(text)) && sscanf(text,"%16" SCNx64,out)==1; }

static HoloEvidenceClass parse_evidence(const char *s) { int i; for(i=1;i<=HOLO_EVIDENCE_ABSENT;i++) if(!strcmp(s,holo_evidence_class_name((HoloEvidenceClass)i))) return (HoloEvidenceClass)i; return HOLO_EVIDENCE_INVALID; }
static HoloMappingStatus parse_status(const char *s) { int i; for(i=1;i<=HOLO_MAP_NOT_APPLICABLE;i++) if(!strcmp(s,holo_mapping_status_name((HoloMappingStatus)i))) return (HoloMappingStatus)i; return HOLO_MAP_INVALID; }
static HoloObservability parse_observability(const char *s) { int i; for(i=1;i<=HOLO_OBSERVABILITY_UNDEFINED;i++) if(!strcmp(s,holo_observability_name((HoloObservability)i))) return (HoloObservability)i; return HOLO_OBSERVABILITY_INVALID; }
static HoloClaimScope parse_scope(const char *s) { int i; for(i=0;i<=HOLO_CLAIM_PHYSICAL_RESTORATION;i++) if(!strcmp(s,claim_scope_name((HoloClaimScope)i))) return (HoloClaimScope)i; return (HoloClaimScope)-1; }
static HoloInvariantPortabilityClass parse_portability(const char *s) { int i; for(i=0;i<=HOLO_PORTABILITY_NO_CURRENT_PHYSICAL_MAPPING;i++) if(!strcmp(s,holo_portability_name((HoloInvariantPortabilityClass)i))) return (HoloInvariantPortabilityClass)i; return (HoloInvariantPortabilityClass)-1; }
static HoloReviewStatus parse_review_status(const char *s) { int i; for(i=0;i<=HOLO_REVIEW_INVALIDATED;i++) if(!strcmp(s,holo_review_status_name((HoloReviewStatus)i))) return (HoloReviewStatus)i; return (HoloReviewStatus)-1; }

int holo_physical_mapping_read_json(HoloPhysicalMappingContract *c, const char *path) {
    char *json = read_all(path); char text[HOLO_MAPPING_TEXT_LEN]; uint64_t expected; size_t i;
    int serialized_sealed, serialized_review_valid, gate_implementation_authorized;
    const char *portability_cursor, *gate_cursor;
    if (!json || holo_physical_mapping_init(c, HOLO_MAPPING_OBJECT_COUNT) != 0) { free(json); return -1; }
    if (!parse_text(json,"contract_id",c->contract_id,sizeof(c->contract_id)) || !parse_text(json,"contract_version",c->contract_version,sizeof(c->contract_version)) ||
        !parse_text(json,"status",c->status,sizeof(c->status)) || !parse_bool(json,"sealed",&serialized_sealed) || !serialized_sealed ||
        !parse_int(json,"claim_level",&c->claim_level) || !parse_hex(json,"contract_digest",&expected) ||
        !parse_bool(json,"reviewed",&c->review.reviewed) || !parse_bool(json,"human_review",&c->review.human_review) ||
        !parse_text(json,"reviewer_role",c->review.reviewer_role,sizeof(c->review.reviewer_role)) ||
        !parse_hex(json,"reviewed_contract_digest",&c->review.reviewed_contract_digest) ||
        !parse_text(json,"review_status",text,sizeof(text)) ||
        !parse_text(json,"review_scope",c->review.review_scope,sizeof(c->review.review_scope)) ||
        !parse_bool(json,"review_valid",&serialized_review_valid) ||
        !parse_bool(json,"implementation_authorized",&c->implementation_authorized) ||
        !parse_text(json,"state_vector",c->physical_state_vector,sizeof(c->physical_state_vector)) || !parse_text(json,"measured",c->measured_components,sizeof(c->measured_components)) ||
        !parse_text(json,"unmeasured",c->unmeasured_components,sizeof(c->unmeasured_components)) || !parse_text(json,"nuisance",c->nuisance_variables,sizeof(c->nuisance_variables)) ||
        !parse_text(json,"required_for_restoration",c->restoration_required_components,sizeof(c->restoration_required_components)) ||
        !parse_text(json,"protocol",c->restoration_evidence_gate,sizeof(c->restoration_evidence_gate)) || !parse_text(json,"controls",c->required_controls,sizeof(c->required_controls))) goto fail;
    c->review.status=parse_review_status(text);
    gate_cursor=strstr(json,"\"l4b5b_gate\"");
    if(!gate_cursor || !parse_text(gate_cursor,"decision",c->l4b5b_decision,sizeof(c->l4b5b_decision)) ||
       !parse_bool(gate_cursor,"implementation_authorized",&gate_implementation_authorized) ||
       gate_implementation_authorized != c->implementation_authorized) goto fail;
    c->sealed = 0;
    for (i=0;i<HOLO_MAPPING_OBJECT_COUNT;i++) {
        char needle[48]; const char *p; HoloPhysicalMappingRecord r; memset(&r,0,sizeof(r));
        snprintf(needle,sizeof(needle),"\"mapping_kind\":%zu",i); p=strstr(json,needle); if(!p) goto fail;
        r.kind=(HoloMappingKind)i;
        if(!parse_text(p,"software_object",r.software_object,sizeof(r.software_object)) || !parse_text(p,"software_role",r.software_role,sizeof(r.software_role)) ||
           !parse_text(p,"physical_correspondent",r.proposed_physical_correspondent,sizeof(r.proposed_physical_correspondent)) || !parse_text(p,"observable",r.observable,sizeof(r.observable)) ||
           !parse_text(p,"evidence_class",text,sizeof(text))) goto fail;
        r.evidence_class=parse_evidence(text);
        if(!parse_text(p,"mapping_status",text,sizeof(text))) goto fail;
        r.mapping_status=parse_status(text);
        if(!parse_text(p,"observability",text,sizeof(text))) goto fail;
        r.observability=parse_observability(text);
        if(!parse_text(p,"claim_scope",text,sizeof(text))) goto fail;
        r.claim_scope=parse_scope(text);
        if(!parse_text(p,"evidence_reference",r.evidence_reference,sizeof(r.evidence_reference)) || !parse_text(p,"allowed_claim",r.allowed_claim,sizeof(r.allowed_claim)) ||
           !parse_text(p,"forbidden_claim",r.forbidden_claim,sizeof(r.forbidden_claim)) || !parse_text(p,"missing_measurement",r.missing_measurement,sizeof(r.missing_measurement)) ||
           !parse_text(p,"falsification_condition",r.falsification_condition,sizeof(r.falsification_condition)) || !parse_int(p,"claim_level",&r.claim_level) ||
           !parse_bool(p,"operator_identification_complete",&r.operator_identification_complete) || !parse_bool(p,"restoration_observables_complete",&r.restoration_observables_complete) ||
           holo_physical_mapping_register(c,&r)!=0) goto fail;
    }
    portability_cursor=strstr(json,"\"invariant_portability\"");
    if(!portability_cursor) goto fail;
    for(i=0;i<HOLO_MAPPING_INVARIANT_COUNT;i++) {
        const char *p=strstr(portability_cursor,"\"invariant_kind\""); HoloInvariantPortabilityRecord pr;
        if(!p) goto fail;
        memset(&pr,0,sizeof(pr));
        if(!parse_text(p,"invariant_kind",pr.invariant_kind,sizeof(pr.invariant_kind)) || !parse_text(p,"portability",text,sizeof(text))) goto fail;
        pr.portability=parse_portability(text);
        if(!parse_text(p,"evidence_reference",pr.evidence_reference,sizeof(pr.evidence_reference)) || !parse_text(p,"promotion_requirement",pr.promotion_requirement,sizeof(pr.promotion_requirement)) ||
           holo_physical_mapping_add_portability(c,&pr)!=0) goto fail;
        portability_cursor=p+strlen("\"invariant_kind\"");
    }
    c->sealed=1; c->contract_digest=expected;
    if(!holo_physical_mapping_validate(c) || c->contract_digest!=mapping_digest(c) ||
       serialized_review_valid != holo_physical_mapping_review_valid(c)) goto fail;
    free(json); return 0;
fail:
    free(json); holo_physical_mapping_destroy(c); return -2;
}

void holo_physical_mapping_reference_init(HoloPhysicalMappingReference *r) {
    if (!r) return;
    memset(r,0,sizeof(*r));
    text_copy(r->contract_id,sizeof(r->contract_id),"l4b5a_pdn_mapping_v1");
    text_copy(r->contract_version,sizeof(r->contract_version),"1.0.0");
    text_copy(r->status,sizeof(r->status),"not_attached");
    text_copy(r->contract_reference,sizeof(r->contract_reference),"none");
    text_copy(r->review_status,sizeof(r->review_status),"UNREVIEWED");
    text_copy(r->l4b5b_decision,sizeof(r->l4b5b_decision),L4B5B_BLOCKED);
    r->claim_level=1;
}

int holo_physical_mapping_reference_attach(HoloPhysicalMappingReference *r, const HoloPhysicalMappingContract *c, const char *reference) {
    size_t i; int supported=0,partial=0,unsupported=0;
    if(!r||!c||!reference||!c->sealed||!holo_physical_mapping_validate(c)) return -1;
    holo_physical_mapping_reference_init(r); text_copy(r->contract_id,sizeof(r->contract_id),c->contract_id); text_copy(r->contract_version,sizeof(r->contract_version),c->contract_version);
    text_copy(r->status,sizeof(r->status),c->status); text_copy(r->contract_reference,sizeof(r->contract_reference),reference); r->contract_digest=c->contract_digest; r->claim_level=c->claim_level;
    text_copy(r->reviewer_role,sizeof(r->reviewer_role),c->review.reviewer_role); r->reviewed_contract_digest=c->review.reviewed_contract_digest;
    text_copy(r->review_status,sizeof(r->review_status),holo_review_status_name(c->review.status)); text_copy(r->l4b5b_decision,sizeof(r->l4b5b_decision),c->l4b5b_decision);
    r->reviewed=c->review.reviewed; r->review_valid=holo_physical_mapping_review_valid(c); r->implementation_authorized=c->implementation_authorized;
    for(i=0;i<c->count;i++){if(c->records[i].mapping_status==HOLO_MAP_SUPPORTED)supported++;else if(c->records[i].mapping_status==HOLO_MAP_PARTIALLY_SUPPORTED)partial++;else if(c->records[i].mapping_status==HOLO_MAP_UNSUPPORTED)unsupported++;}
    r->supported_records=supported;r->partial_records=partial;r->unsupported_records=unsupported; return holo_physical_mapping_reference_validate(r)?0:-2;
}

int holo_physical_mapping_reference_validate(const HoloPhysicalMappingReference *r) {
    if (!r || strcmp(r->contract_id,"l4b5a_pdn_mapping_v1") != 0 ||
        strcmp(r->contract_version,"1.0.0") != 0 || !text_present(r->contract_reference) ||
        strcmp(r->l4b5b_decision,L4B5B_BLOCKED) != 0 || r->implementation_authorized ||
        r->claim_level != 1) return 0;
    if (strcmp(r->status,"not_attached") == 0) {
        return r->contract_digest == 0 && r->supported_records == 0 &&
               r->partial_records == 0 && r->unsupported_records == 0 && !r->reviewed &&
               !r->review_valid && r->reviewed_contract_digest == 0 &&
               strcmp(r->review_status,"UNREVIEWED") == 0 && !text_present(r->reviewer_role);
    }
    return strcmp(r->status,"evidence_gap_audited") == 0 && r->contract_digest != 0 &&
            r->supported_records == 1 && r->partial_records == 3 &&
            r->unsupported_records == 4 && r->reviewed && r->review_valid &&
            strcmp(r->reviewer_role,HUMAN_PROJECT_OWNER) == 0 &&
            r->reviewed_contract_digest == r->contract_digest &&
            strcmp(r->review_status,"ACCEPTED_AT_STATED_CLAIM_CEILING") == 0;
}
