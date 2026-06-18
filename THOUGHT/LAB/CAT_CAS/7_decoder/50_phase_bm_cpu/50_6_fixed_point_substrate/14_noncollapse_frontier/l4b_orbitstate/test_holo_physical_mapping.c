#include "../holo_runtime/holo_physical_mapping.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static HoloPhysicalMappingContract make_current(void) {
    HoloPhysicalMappingContract contract;
    assert(holo_physical_mapping_init(&contract, HOLO_MAPPING_OBJECT_COUNT) == 0);
    assert(holo_physical_mapping_populate_current(&contract) == 0);
    assert(holo_physical_mapping_validate(&contract));
    return contract;
}

static char *read_file(const char *path, size_t *size_out) {
    FILE *file = fopen(path, "rb");
    long size;
    char *text;
    assert(file != NULL);
    assert(fseek(file, 0, SEEK_END) == 0);
    size = ftell(file);
    assert(size >= 0);
    assert(fseek(file, 0, SEEK_SET) == 0);
    text = (char *)malloc((size_t)size + 1U);
    assert(text != NULL);
    assert(fread(text, 1, (size_t)size, file) == (size_t)size);
    text[size] = '\0';
    fclose(file);
    *size_out = (size_t)size;
    return text;
}

static void replace_once(const char *path, const char *from, const char *to) {
    size_t size;
    size_t from_len = strlen(from);
    size_t to_len = strlen(to);
    char *text = read_file(path, &size);
    char *at = strstr(text, from);
    char *changed;
    FILE *file;
    size_t prefix;
    assert(at != NULL);
    prefix = (size_t)(at - text);
    changed = (char *)malloc(size - from_len + to_len + 1U);
    assert(changed != NULL);
    memcpy(changed, text, prefix);
    memcpy(changed + prefix, to, to_len);
    memcpy(changed + prefix + to_len, at + from_len, size - prefix - from_len);
    changed[size - from_len + to_len] = '\0';
    file = fopen(path, "wb");
    assert(file != NULL);
    assert(fwrite(changed, 1, size - from_len + to_len, file) == size - from_len + to_len);
    assert(fclose(file) == 0);
    free(changed);
    free(text);
}

static int file_has_key(const char *path, const char *key) {
    size_t size;
    char *text = read_file(path, &size);
    int found = strstr(text, key) != NULL;
    (void)size;
    free(text);
    return found;
}

static void test_validation_rules(void) {
    HoloPhysicalMappingContract contract = make_current();
    HoloPhysicalMappingRecord record = contract.records[HOLO_MAPPING_GEOMETRY];
    HoloPhysicalMappingContract incomplete;

    puts("COMPLETE_MAPPING_RECORD_REGISTRATION_PASS");

    record.evidence_class = HOLO_EVIDENCE_INVALID;
    assert(!holo_physical_mapping_record_validate(&record));
    assert(holo_physical_mapping_init(&incomplete, 1U) == 0);
    assert(holo_physical_mapping_register(&incomplete, &record) != 0);
    holo_physical_mapping_destroy(&incomplete);
    puts("MISSING_EVIDENCE_CLASS_REJECTED_PASS");

    record = contract.records[HOLO_MAPPING_GEOMETRY];
    record.evidence_class = HOLO_EVIDENCE_ABSENT;
    record.mapping_status = HOLO_MAP_SUPPORTED;
    assert(!holo_physical_mapping_record_validate(&record));
    puts("ABSENT_SUPPORTED_REJECTED_PASS");

    record = contract.records[HOLO_MAPPING_PATH_HISTORY];
    record.evidence_class = HOLO_EVIDENCE_SOFTWARE_ONLY;
    record.claim_scope = HOLO_CLAIM_PHYSICAL_CHANNEL;
    snprintf(record.allowed_claim, sizeof(record.allowed_claim), "%s", "physical path proven");
    assert(!holo_physical_mapping_record_validate(&record));
    puts("SOFTWARE_ONLY_PHYSICAL_CLAIM_REJECTED_PASS");

    record = contract.records[HOLO_MAPPING_RELATION_BASIS];
    snprintf(record.allowed_claim, sizeof(record.allowed_claim), "%s", "directly measured physical basis");
    assert(!holo_physical_mapping_record_validate(&record));
    puts("PROPOSED_MEASURED_CLAIM_REJECTED_PASS");

    record = contract.records[HOLO_MAPPING_RESTORATION];
    record.evidence_class = HOLO_EVIDENCE_MEASURED;
    record.mapping_status = HOLO_MAP_SUPPORTED;
    record.claim_scope = HOLO_CLAIM_PHYSICAL_RESTORATION;
    record.observability = HOLO_PARTIALLY_OBSERVABLE;
    record.restoration_observables_complete = 0;
    snprintf(record.allowed_claim, sizeof(record.allowed_claim), "%s", "physical restoration proven");
    assert(!holo_physical_mapping_record_validate(&record));
    puts("PARTIAL_OBSERVABILITY_RESTORATION_REJECTED_PASS");

    record = contract.records[HOLO_MAPPING_RELATION_BASIS];
    record.evidence_class = HOLO_EVIDENCE_MEASURED;
    record.mapping_status = HOLO_MAP_SUPPORTED;
    record.operator_identification_complete = 0;
    assert(!holo_physical_mapping_record_validate(&record));
    puts("UNCALIBRATED_RELATION_BASIS_REJECTED_PASS");

    assert(holo_physical_mapping_seal(&contract) == 0);
    assert(holo_physical_mapping_register(&contract, &contract.records[0]) != 0);
    puts("SEALED_CONTRACT_MUTATION_REJECTED_PASS");
    holo_physical_mapping_destroy(&contract);
}

static void test_expected_classification(void) {
    HoloPhysicalMappingContract contract = make_current();
    size_t i;
    assert(contract.records[HOLO_MAPPING_CARRIER].mapping_status == HOLO_MAP_SUPPORTED);
    assert(contract.records[HOLO_MAPPING_CARRIER].evidence_class == HOLO_EVIDENCE_MEASURED);
    assert(contract.records[HOLO_MAPPING_GEOMETRY].mapping_status == HOLO_MAP_UNSUPPORTED);
    assert(contract.records[HOLO_MAPPING_PATH_HISTORY].evidence_class == HOLO_EVIDENCE_ABSENT);
    assert(contract.records[HOLO_MAPPING_RESTORATION].mapping_status == HOLO_MAP_UNSUPPORTED);
    assert(contract.records[HOLO_MAPPING_COLLAPSE_BOUNDARY].mapping_status == HOLO_MAP_PARTIALLY_SUPPORTED);
    assert(contract.portability_count == HOLO_MAPPING_INVARIANT_COUNT);
    for (i = 0; i < contract.portability_count; ++i) {
        assert(contract.portability[i].invariant_kind[0] != '\0');
        assert(contract.portability[i].promotion_requirement[0] != '\0');
    }
    puts("CARRIER_CHANNEL_NOT_GEOMETRIC_MEMORY_PASS");
    puts("SOFTWARE_RESTORATION_NOT_PHYSICAL_RESTORATION_PASS");
    puts("MEASUREMENT_BOUNDARY_NOT_COMPLETE_COLLAPSE_MODEL_PASS");
    puts("INVARIANT_PORTABILITY_COMPLETE_PASS");
    holo_physical_mapping_destroy(&contract);
}

static void test_roundtrip_and_tampering(void) {
    const char *path = "holo_physical_mapping_test.json";
    HoloPhysicalMappingContract contract = make_current();
    HoloPhysicalMappingContract loaded;
    assert(holo_physical_mapping_seal(&contract) == 0);
    assert(holo_physical_mapping_write_json(&contract, path) == 0);
    assert(holo_physical_mapping_read_json(&loaded, path) == 0);
    assert(holo_physical_mapping_equal(&contract, &loaded));
    puts("PHYSICAL_MAPPING_SERIALIZATION_ROUNDTRIP_PASS");
    holo_physical_mapping_destroy(&loaded);

    replace_once(path, "\"mapping_status\":\"UNSUPPORTED\"",
                       "\"mapping_status\":\"SUPPORTED\"");
    assert(holo_physical_mapping_read_json(&loaded, path) != 0);
    puts("SERIALIZED_STATUS_INFLATION_REJECTED_PASS");

    assert(holo_physical_mapping_write_json(&contract, path) == 0);
    replace_once(path, "\"evidence_class\":\"PROPOSED\"",
                       "\"evidence_class\":\"MEASURED\"");
    assert(holo_physical_mapping_read_json(&loaded, path) != 0);
    puts("SERIALIZED_EVIDENCE_MUTATION_REJECTED_PASS");

    assert(holo_physical_mapping_write_json(&contract, path) == 0);
    replace_once(path, "PDN topology is a candidate carrier geometry only",
                       "physical proof of geometric memory");
    assert(holo_physical_mapping_read_json(&loaded, path) != 0);
    puts("SERIALIZED_CLAIM_INFLATION_REJECTED_PASS");

    assert(holo_physical_mapping_write_json(&contract, path) == 0);
    replace_once(path, "\"observability\":\"UNOBSERVABLE_WITH_CURRENT_INSTRUMENTS\"",
                       "\"observability\":\"OBSERVABLE\"");
    assert(holo_physical_mapping_read_json(&loaded, path) != 0);
    puts("SERIALIZED_OBSERVABILITY_TAMPERING_REJECTED_PASS");

    assert(holo_physical_mapping_write_json(&contract, path) == 0);
    replace_once(path, "\"sealed\": true", "\"sealed\": false");
    assert(holo_physical_mapping_read_json(&loaded, path) != 0);
    puts("SERIALIZED_SEAL_TAMPERING_REJECTED_PASS");

    assert(file_has_key(path, "\"physical_state_definition\""));
    assert(file_has_key(path, "\"restoration_evidence_gate\""));
    assert(!file_has_key(path, "\"winner\""));
    assert(!file_has_key(path, "\"candidate_score\""));
    assert(!file_has_key(path, "\"hidden_d\""));
    assert(!file_has_key(path, "\"recovered_d\""));
    assert(!file_has_key(path, "\"orientation_label\""));
    assert(!file_has_key(path, "\"verify_pass\""));
    assert(!file_has_key(path, "\"AUC\""));
    puts("PHYSICAL_MAPPING_FORBIDDEN_FIELDS_PASS");
    holo_physical_mapping_destroy(&contract);
    remove(path);
}

static void print_contract_evidence(void) {
    HoloPhysicalMappingContract contract = make_current();
    size_t i;
    assert(holo_physical_mapping_seal(&contract) == 0);
    for (i = 0; i < contract.count; ++i) {
        HoloPhysicalMappingRecord *r = &contract.records[i];
        printf("MAPPING %s | %s | %s | %s | observable=%s\n",
               r->software_object, holo_evidence_class_name(r->evidence_class),
               holo_mapping_status_name(r->mapping_status),
               holo_observability_name(r->observability), r->observable);
    }
    for (i = 0; i < contract.portability_count; ++i) {
        printf("PORTABILITY %s | %s\n", contract.portability[i].invariant_kind,
               holo_portability_name(contract.portability[i].portability));
    }
    printf("PHYSICAL_STATE %s\n", contract.physical_state_vector);
    printf("MEASURED %s\n", contract.measured_components);
    printf("UNMEASURED %s\n", contract.unmeasured_components);
    printf("NUISANCE %s\n", contract.nuisance_variables);
    printf("RESTORATION_REQUIRED %s\n", contract.restoration_required_components);
    printf("RESTORATION_GATE %s\n", contract.restoration_evidence_gate);
    printf("REQUIRED_CONTROLS %s\n", contract.required_controls);
    printf("contract_digest=%016llx\n", (unsigned long long)contract.contract_digest);
    puts("L4B5B_DECISION=NOT_AUTHORIZED_EVIDENCE_MISSING");
    holo_physical_mapping_destroy(&contract);
}

int main(void) {
    test_validation_rules();
    test_expected_classification();
    test_roundtrip_and_tampering();
    print_contract_evidence();
    puts("HOLO_PHYSICAL_MAPPING_TEST_PASS");
    return 0;
}
