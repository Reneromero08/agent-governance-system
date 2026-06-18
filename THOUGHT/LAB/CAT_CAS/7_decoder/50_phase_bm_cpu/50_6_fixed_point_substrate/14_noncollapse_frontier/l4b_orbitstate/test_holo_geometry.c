#include "orbit_state.h"
#include "holo_path_history.h"
#include "../holo_runtime/holo_geometry.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int output_contains(const char *path, const char *needle) {
    FILE *f = fopen(path, "rb");
    long size;
    char *buffer;
    int found;
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 0; }
    buffer = (char *)malloc((size_t)size + 1U);
    if (!buffer) { fclose(f); return 0; }
    if (fread(buffer, 1, (size_t)size, f) != (size_t)size) {
        free(buffer); fclose(f); return 0;
    }
    fclose(f);
    buffer[size] = '\0';
    found = strstr(buffer, needle) != NULL;
    free(buffer);
    return found;
}

int main(void) {
    const char *path = "holo_geometry_test.holo";
    OrbitState orbit;
    OrbitState initial;
    OrbitState terminal;
    OrbitState restored;
    EvolParams params = { .max_steps = 64, .seed = 42 };
    int steps = 0;
    HoloObject object;
    HoloObject loaded;
    double rendered[2];
    int read_rc;

    orbit_init(&orbit, 256, 23, 233);
    initial = orbit;
    assert(holo_object_init(&object, 42, 256, 23, 233) == 0);
    assert(holo_path_evolve(object.evolution.path_history, &orbit, &params) == HOLO_PATH_OK);
    terminal = orbit;
    steps = orbit.steps;
    holo_record_evolution(&object, params.seed, steps, orbit.acc_real, orbit.acc_imag);
    holo_set_carrier_phase(&object, 1.0, -1.0);

    assert(object.geometry.basis_rank == 2);
    assert(object.geometry.coordinates[0] == 23.0);
    assert(object.geometry.coordinates[1] == 233.0);
    assert(holo_geometry_render(&object.geometry, rendered) == 0);
    assert(rendered[0] == 233.0);
    assert(rendered[1] == 23.0);
    assert(strcmp(object.geometry.geometry_status, "unresolved_native_geometry") == 0);
    assert(object.carrier.phase[0] != 0.0);
    assert(strcmp(object.hypothesis, HOLO_HYPOTHESIS) == 0);
    assert(holo_extract_invariant(&object) == -1);
    assert(object.invariant_family.extracted == 0);

    assert(strcmp(holo_materialization_mode_name(HOLO_NATIVE), "native_holo") == 0);
    holo_set_materialization_mode(&object, HOLO_MATERIALIZED_FALLBACK);
    assert(strcmp(holo_materialization_mode_name(object.projection.materialization_mode),
                  "materialized_fallback") == 0);
    holo_set_materialization_mode(&object, HOLO_NATIVE);

    assert(holo_verify_software_restoration(&object, &initial, &terminal, &restored, 0) == 0);
    assert(holo_cross_boundary(&object, steps) == 0);
    assert(object.collapse_boundary.crossed == 1);
    assert(object.invariant_family.extracted == 1);
    assert(object.invariant_family.records[HOLO_INV_ORBIT_CONSERVATION].passed == 1);
    assert(holo_validate(&object) == 1);

    holo_set_materialization_mode(&object, HOLO_MATERIALIZED_FALLBACK);
    assert(holo_write_json(&object, path) == 0);
    read_rc = holo_read_json(&loaded, path);
    if (read_rc != 0) fprintf(stderr, "holo_read_json rc=%d\n", read_rc);
    assert(read_rc == 0);
    assert(loaded.projection.materialization_mode == HOLO_MATERIALIZED_FALLBACK);
    assert(output_contains(path, "\"materialization_mode\": \"materialized_fallback\""));
    holo_object_destroy(&loaded);

    holo_set_materialization_mode(&object, HOLO_NATIVE);
    assert(holo_write_json(&object, path) == 0);
    read_rc = holo_read_json(&loaded, path);
    if (read_rc != 0) fprintf(stderr, "holo_read_json rc=%d\n", read_rc);
    assert(read_rc == 0);
    assert(loaded.geometry.basis_rank == 2);
    assert(loaded.projection.materialization_mode == HOLO_NATIVE);

    assert(output_contains(path, "\"holo_geometry\""));
    assert(output_contains(path, "\"physical_mapping\""));
    assert(output_contains(path, "\"status\": \"not_attached\""));
    assert(output_contains(path, "\"relation_basis\""));
    assert(output_contains(path, "\"coordinates\""));
    assert(output_contains(path, "\"carrier\""));
    assert(output_contains(path, "\"phase_relation\""));
    assert(output_contains(path, "\"collapse_boundary\""));
    assert(output_contains(path, "\"materialization_mode\": \"native_holo\""));
    assert(!output_contains(path, "\"orientation_label\""));
    assert(!output_contains(path, "orientation"));
    assert(!output_contains(path, "\"winner\""));
    assert(!output_contains(path, "\"verify_pass\""));

    holo_object_destroy(&loaded);
    holo_object_destroy(&object);
    remove(path);
    puts("HOLO_GEOMETRY_TEST_PASS");
    return 0;
}
