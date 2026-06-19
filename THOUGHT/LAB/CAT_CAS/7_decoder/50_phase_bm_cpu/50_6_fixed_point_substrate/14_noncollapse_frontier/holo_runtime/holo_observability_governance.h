#ifndef HOLO_OBSERVABILITY_GOVERNANCE_H
#define HOLO_OBSERVABILITY_GOVERNANCE_H

#include "holo_observability_design.h"

#define HOLO_REVIEW_SHA256_LEN 65
#define HOLO_REVIEW_TEXT_LEN 256

typedef enum {
    HOLO_DESIGN_REVIEW_PENDING = 0,
    HOLO_DESIGN_REVIEW_ACCEPTED_AT_CLAIM_CEILING,
    HOLO_DESIGN_REVIEW_REJECTED
} HoloDesignReviewStatus;

typedef struct {
    char design_id[64];
    char design_version[32];
    uint64_t design_digest;
    char design_artifact[HOLO_REVIEW_TEXT_LEN];
    char design_artifact_sha256[HOLO_REVIEW_SHA256_LEN];
    char reviewer_role[64];
    char review_scope[HOLO_REVIEW_TEXT_LEN];
    HoloDesignReviewStatus status;
    int human_review;
    int implementation_authorized;
    int claim_level;
} HoloObservabilityReviewEnvelope;

/* Close all named references before a design can be reviewed. */
int holo_observability_design_validate_references(
    const HoloObservabilityDesign *design);

/*
 * Bind a human review to the sealed design artifact bytes. Review metadata is
 * deliberately outside HoloObservabilityDesign and therefore cannot change the
 * frozen scientific-content digest it reviews.
 */
int holo_observability_review_apply(
    HoloObservabilityReviewEnvelope *review,
    const HoloObservabilityDesign *design,
    const char *design_artifact,
    const char *reviewer_role,
    const char *review_scope,
    HoloDesignReviewStatus status);

int holo_observability_review_validate(
    const HoloObservabilityReviewEnvelope *review,
    const HoloObservabilityDesign *design);

int holo_observability_review_write_json(
    const HoloObservabilityReviewEnvelope *review,
    const char *path);

const char *holo_design_review_status_name(HoloDesignReviewStatus status);

#endif
