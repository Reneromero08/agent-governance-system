#include "orbit_state.h"
#include "../holo_runtime/holo_geometry.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static int output_contains(const char *path, const char *needle) {
    FILE *f = fopen(path, "rb");
    long size;
    char buffer[8192];
    if (!f) return 0;
    size = (long)fread(buffer, 1, sizeof(buffer) - 1U, f);
    fclose(f);
    buffer[size] = '\0';
    return strstr(buffer, needle) != NULL;
}

int main(void) {
    const char *path = "holo_geometry_test.holo";
    OrbitState orbit;
    EvolParams params = { .max_steps = 64, .seed = 42 };
    PathStep history[ORBIT_MAX_STEPS];
    int steps = 0;
    HoloObject object;
    HoloObject loaded;
    double rendered[2];

    orbit_init(&orbit, 256, 23, 233);
    orbit_evolve(&orbit, &params, history, &steps);
    holo_object_init(&object, 42, 256, 23, 233);
    holo_record_evolution(&object, params.seed, steps, orbit.acc_real, orbit.acc_imag);
    holo_set_carrier_phase(&object, history[steps - 1].theta_plus,
                           history[steps - 1].theta_minus);

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
    assert(object.invariant.extracted == 0);

    assert(strcmp(holo_materialization_mode_name(HOLO_NATIVE), "native_holo") == 0);
    holo_set_materialization_mode(&object, HOLO_MATERIALIZED_FALLBACK);
    assert(strcmp(holo_materialization_mode_name(object.projection.materialization_mode),
                  "materialized_fallback") == 0);
    holo_set_materialization_mode(&object, HOLO_NATIVE);

    assert(holo_cross_boundary(&object, steps) == 0);
    assert(object.collapse_boundary.crossed == 1);
    assert(object.invariant.extracted == 1);
    assert(object.invariant.fold_symmetry_holds == 1);
    assert(holo_validate(&object) == 1);

    holo_set_materialization_mode(&object, HOLO_MATERIALIZED_FALLBACK);
    assert(holo_write_json(&object, path) == 0);
    assert(holo_read_json(&loaded, path) == 0);
    assert(loaded.projection.materialization_mode == HOLO_MATERIALIZED_FALLBACK);
    assert(output_contains(path, "\"materialization_mode\": \"materialized_fallback\""));

    holo_set_materialization_mode(&object, HOLO_NATIVE);
    assert(holo_write_json(&object, path) == 0);
    assert(holo_read_json(&loaded, path) == 0);
    assert(loaded.geometry.basis_rank == 2);
    assert(loaded.projection.materialization_mode == HOLO_NATIVE);

    assert(output_contains(path, "\"holo_geometry\""));
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

    remove(path);
    puts("HOLO_GEOMETRY_TEST_PASS");
    return 0;
}
