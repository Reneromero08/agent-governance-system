#include "holo_observability_governance.h"

#include <inttypes.h>
#include <openssl/sha.h>
#include <stdio.h>
#include <string.h>

static int present(const char *text) {
    return text && text[0] != '\0';
}

static int state_exists(const HoloObservabilityDesign *design, const char *id) {
    size_t i;
    for (i = 0; i < design->state_count; ++i) {
        if (strcmp(design->state_models[i].model_id, id) == 0) return 1;
    }
    return 0;
}

static int input_exists(const HoloObservabilityDesign *design, const char *id) {
    size_t i;
    for (i = 0; i < design->input_count; ++i) {
        if (strcmp(design->input_families[i].input_id, id) == 0) return 1;
    }
    return 0;
}

static int builtin_control_exists(const char *id) {
    static const char *controls[] = {
        "shuffled_input",
        "phase_randomized",
        "reordered_schedule"
    };
    size_t i;
    for (i = 0; i < sizeof(controls) / sizeof(controls[0]); ++i) {
        if (strcmp(controls[i], id) == 0) return 1;
    }
    return 0;
}

static int unique_text_id(const char *candidate, const char *const *seen,
                          size_t seen_count) {
    size_t i;
    if (!present(candidate)) return 0;
    for (i = 0; i < seen_count; ++i) {
        if (strcmp(candidate, seen[i]) == 0) return 0;
    }
    return 1;
}

int holo_observability_design_validate_references(
    const HoloObservabilityDesign *design) {
    const char *seen[HOLO_DESIGN_ARTIFACT_COUNT];
    int gate_seen[HOLO_DESIGN_GATE_COUNT] = {0};
    int falsification_seen[HOLO_DESIGN_FALSIFICATION_COUNT] = {0};
    size_t i;

    if (!design || !holo_observability_design_validate(design)) return 0;

    for (i = 0; i < design->input_count; ++i) {
        const char *control = design->input_families[i].null_control;
        if (!present(control) ||
            (!input_exists(design, control) && !builtin_control_exists(control))) {
            return 0;
        }
    }

    for (i = 0; i < design->operator_count; ++i) {
        if (!state_exists(design, design->operators[i].state_model_id)) return 0;
    }

    for (i = 0; i < design->calibration_count; ++i) {
        if (!input_exists(design, design->calibration[i].input_id) ||
            !present(design->calibration[i].artifact_output)) return 0;
    }

    for (i = 0; i < design->gate_count; ++i) {
        int index = 0;
        char suffix = '\0';
        if (sscanf(design->gates[i].gate_id, "G%d%c", &index, &suffix) != 1 ||
            index < 1 || index > (int)HOLO_DESIGN_GATE_COUNT ||
            gate_seen[index - 1]) return 0;
        gate_seen[index - 1] = 1;
    }
    for (i = 0; i < HOLO_DESIGN_GATE_COUNT; ++i) {
        if (!gate_seen[i]) return 0;
    }

    for (i = 0; i < design->falsification_count; ++i) {
        int index = 0;
        char suffix = '\0';
        if (sscanf(design->falsifications[i].condition_id, "F%d%c", &index, &suffix) != 1 ||
            index < 1 || index > (int)HOLO_DESIGN_FALSIFICATION_COUNT ||
            falsification_seen[index - 1]) return 0;
        falsification_seen[index - 1] = 1;
    }
    for (i = 0; i < HOLO_DESIGN_FALSIFICATION_COUNT; ++i) {
        if (!falsification_seen[i]) return 0;
    }

    for (i = 0; i < design->artifact_count; ++i) {
        size_t j;
        if (!unique_text_id(design->artifacts[i].artifact_id, seen, i) ||
            !present(design->artifacts[i].schema_id)) return 0;
        seen[i] = design->artifacts[i].artifact_id;
        for (j = 0; j < i; ++j) {
            if (strcmp(design->artifacts[i].schema_id,
                       design->artifacts[j].schema_id) == 0) return 0;
        }
    }

    return 1;
}

static int file_sha256_hex(const char *path, char output[HOLO_REVIEW_SHA256_LEN]) {
    FILE *file;
    SHA256_CTX context;
    unsigned char digest[SHA256_DIGEST_LENGTH];
    unsigned char buffer[8192];
    size_t count;
    size_t i;

    if (!path || !output) return 0;
    file = fopen(path, "rb");
    if (!file) return 0;
    if (SHA256_Init(&context) != 1) {
        fclose(file);
        return 0;
    }
    while ((count = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        if (SHA256_Update(&context, buffer, count) != 1) {
            fclose(file);
            return 0;
        }
    }
    if (ferror(file) || fclose(file) != 0 || SHA256_Final(digest, &context) != 1) {
        return 0;
    }
    for (i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        snprintf(output + i * 2U, 3U, "%02x", digest[i]);
    }
    output[64] = '\0';
    return 1;
}

const char *holo_design_review_status_name(HoloDesignReviewStatus status) {
    switch (status) {
        case HOLO_DESIGN_REVIEW_PENDING: return "PENDING";
        case HOLO_DESIGN_REVIEW_ACCEPTED_AT_CLAIM_CEILING:
            return "ACCEPTED_AT_STATED_CLAIM_CEILING";
        case HOLO_DESIGN_REVIEW_REJECTED: return "REJECTED";
        default: return "INVALID";
    }
}

int holo_observability_review_apply(
    HoloObservabilityReviewEnvelope *review,
    const HoloObservabilityDesign *design,
    const char *design_artifact,
    const char *reviewer_role,
    const char *review_scope,
    HoloDesignReviewStatus status) {
    if (!review || !design || !design_artifact || !present(reviewer_role) ||
        !present(review_scope) || status == HOLO_DESIGN_REVIEW_PENDING ||
        !design->sealed || design->design_digest == 0 ||
        design->status != HOLO_DESIGN_READY_FOR_HUMAN_REVIEW ||
        design->human_reviewed || design->implementation_authorized ||
        design->executed || !holo_observability_design_validate_references(design)) {
        return -1;
    }

    memset(review, 0, sizeof(*review));
    snprintf(review->design_id, sizeof(review->design_id), "%s", design->design_id);
    snprintf(review->design_version, sizeof(review->design_version), "%s",
             design->design_version);
    review->design_digest = design->design_digest;
    snprintf(review->design_artifact, sizeof(review->design_artifact), "%s",
             design_artifact);
    if (!file_sha256_hex(design_artifact, review->design_artifact_sha256)) return -2;
    snprintf(review->reviewer_role, sizeof(review->reviewer_role), "%s", reviewer_role);
    snprintf(review->review_scope, sizeof(review->review_scope), "%s", review_scope);
    review->status = status;
    review->human_review = 1;
    review->implementation_authorized = 0;
    review->claim_level = 1;
    return holo_observability_review_validate(review, design) ? 0 : -3;
}

int holo_observability_review_validate(
    const HoloObservabilityReviewEnvelope *review,
    const HoloObservabilityDesign *design) {
    char digest[HOLO_REVIEW_SHA256_LEN];
    if (!review || !design || !review->human_review ||
        review->implementation_authorized || review->claim_level != 1 ||
        review->status == HOLO_DESIGN_REVIEW_PENDING ||
        !present(review->reviewer_role) || !present(review->review_scope) ||
        strcmp(review->design_id, design->design_id) != 0 ||
        strcmp(review->design_version, design->design_version) != 0 ||
        review->design_digest != design->design_digest ||
        !design->sealed || design->status != HOLO_DESIGN_READY_FOR_HUMAN_REVIEW ||
        design->human_reviewed || design->implementation_authorized ||
        !holo_observability_design_validate_references(design) ||
        !file_sha256_hex(review->design_artifact, digest) ||
        strcmp(digest, review->design_artifact_sha256) != 0) {
        return 0;
    }
    return 1;
}

int holo_observability_review_write_json(
    const HoloObservabilityReviewEnvelope *review,
    const char *path) {
    FILE *file;
    if (!review || !path || !review->human_review ||
        review->status == HOLO_DESIGN_REVIEW_PENDING ||
        review->implementation_authorized || review->claim_level != 1 ||
        !present(review->design_artifact_sha256)) return -1;
    file = fopen(path, "w");
    if (!file) return -2;
    fprintf(file, "{\n");
    fprintf(file, "  \"schema_id\":\"l4b5b0_observability_review_v1\",\n");
    fprintf(file, "  \"design_id\":\"%s\",\n", review->design_id);
    fprintf(file, "  \"design_version\":\"%s\",\n", review->design_version);
    fprintf(file, "  \"design_digest\":\"%016" PRIx64 "\",\n",
            review->design_digest);
    fprintf(file, "  \"design_artifact\":\"%s\",\n", review->design_artifact);
    fprintf(file, "  \"design_artifact_sha256\":\"%s\",\n",
            review->design_artifact_sha256);
    fprintf(file, "  \"human_review\":true,\n");
    fprintf(file, "  \"reviewer_role\":\"%s\",\n", review->reviewer_role);
    fprintf(file, "  \"review_scope\":\"%s\",\n", review->review_scope);
    fprintf(file, "  \"review_status\":\"%s\",\n",
            holo_design_review_status_name(review->status));
    fprintf(file, "  \"implementation_authorized\":false,\n");
    fprintf(file, "  \"claim_level\":1\n");
    fprintf(file, "}\n");
    return fclose(file) == 0 ? 0 : -3;
}
