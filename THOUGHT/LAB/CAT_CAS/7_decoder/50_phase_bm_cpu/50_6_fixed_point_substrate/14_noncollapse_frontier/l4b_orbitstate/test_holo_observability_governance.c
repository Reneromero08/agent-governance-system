#include "../holo_runtime/holo_observability_design.h"
#include "../holo_runtime/holo_observability_governance.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static HoloObservabilityDesign make_design(void) {
    HoloObservabilityDesign design;
    assert(holo_observability_design_init(&design) == 0);
    assert(holo_observability_design_populate_current(&design) == 0);
    assert(holo_observability_design_validate(&design));
    return design;
}

static void append_byte(const char *path) {
    FILE *file = fopen(path, "ab");
    assert(file != NULL);
    assert(fputc('\n', file) != EOF);
    assert(fclose(file) == 0);
}

static void test_reference_graph(void) {
    HoloObservabilityDesign design = make_design();
    HoloObservabilityDesign missing_control = design;
    HoloObservabilityDesign missing_artifact = design;
    HoloObservabilityDesign duplicate_gate = design;

    assert(holo_observability_design_validate_references(&design));
    puts("OBSERVABILITY_REFERENCE_GRAPH_CLOSED_PASS");

    snprintf(missing_control.input_families[3].null_control,
             sizeof(missing_control.input_families[3].null_control),
             "%s", "undefined_control");
    assert(holo_observability_design_validate(&missing_control));
    assert(!holo_observability_design_validate_references(&missing_control));
    puts("UNDEFINED_NULL_CONTROL_REJECTED_PASS");

    snprintf(missing_artifact.calibration[0].artifact_output,
             sizeof(missing_artifact.calibration[0].artifact_output),
             "%s", "undefined_artifact");
    assert(holo_observability_design_validate(&missing_artifact));
    assert(!holo_observability_design_validate_references(&missing_artifact));
    puts("UNDEFINED_CALIBRATION_ARTIFACT_REJECTED_PASS");

    snprintf(duplicate_gate.gates[2].gate_id,
             sizeof(duplicate_gate.gates[2].gate_id),
             "%s", duplicate_gate.gates[0].gate_id);
    assert(holo_observability_design_validate(&duplicate_gate));
    assert(!holo_observability_design_validate_references(&duplicate_gate));
    puts("DUPLICATE_GATE_ID_REJECTED_PASS");

    holo_observability_design_destroy(&duplicate_gate);
    holo_observability_design_destroy(&missing_artifact);
    holo_observability_design_destroy(&missing_control);
    holo_observability_design_destroy(&design);
}

static void test_external_review_envelope(void) {
    const char *design_path = "holo_observability_governance_design.json";
    const char *review_path = "holo_observability_governance_review.json";
    HoloObservabilityDesign design = make_design();
    HoloObservabilityReviewEnvelope review;

    assert(holo_observability_design_validate_references(&design));
    assert(holo_observability_design_seal(&design) == 0);
    assert(holo_observability_design_write_json(&design, design_path) == 0);
    assert(holo_observability_review_apply(
               &review, &design, design_path,
               "human_project_owner",
               "design completeness, claim ceiling, reference closure, and non-authorizing review",
               HOLO_DESIGN_REVIEW_ACCEPTED_AT_CLAIM_CEILING) == 0);
    assert(holo_observability_review_validate(&review, &design));
    assert(!design.human_reviewed);
    assert(design.status == HOLO_DESIGN_READY_FOR_HUMAN_REVIEW);
    assert(!design.implementation_authorized);
    puts("EXTERNAL_DIGEST_BOUND_REVIEW_PASS");

    assert(holo_observability_review_write_json(&review, review_path) == 0);
    append_byte(design_path);
    assert(!holo_observability_review_validate(&review, &design));
    puts("DESIGN_ARTIFACT_MUTATION_INVALIDATES_REVIEW_PASS");

    assert(holo_observability_design_write_json(&design, design_path) == 0);
    assert(holo_observability_review_validate(&review, &design));
    puts("REVIEW_DOES_NOT_MUTATE_SCIENTIFIC_DIGEST_PASS");

    remove(review_path);
    remove(design_path);
    holo_observability_design_destroy(&design);
}

int main(void) {
    test_reference_graph();
    test_external_review_envelope();
    puts("HOLO_OBSERVABILITY_GOVERNANCE_TEST_PASS");
    return 0;
}
