#ifndef CATVM_PHASE_CORE_H
#define CATVM_PHASE_CORE_H

#include <complex.h>
#include <stddef.h>
#include <stdint.h>

#define CATVM_RELATION_CELLS 4U
#define CATVM_CARRIER_CELLS 24U
#define CATVM_RESTORATION_TOLERANCE 2.0e-12
#define CATVM_ROOT_TOLERANCE 4.0e-10
#define CATVM_CONTROL_MINIMUM 1.0e-3

enum catvm_backend_kind {
    CATVM_BACKEND_IN_PLACE = 0,
    CATVM_BACKEND_SNAPSHOT = 1,
    CATVM_BACKEND_NULL = 2
};

enum catvm_machine_state {
    CATVM_STATE_IDLE = 0,
    CATVM_STATE_SEALED = 1,
    CATVM_STATE_AFTER_F = 2,
    CATVM_STATE_AFTER_G = 3,
    CATVM_STATE_PROJECTED = 4,
    CATVM_STATE_FAILED = 5
};

enum catvm_restore_mode {
    CATVM_RESTORE_CORRECT = 0,
    CATVM_RESTORE_WRONG_G = 1,
    CATVM_RESTORE_MISSING_G = 2,
    CATVM_RESTORE_REORDERED = 3,
    CATVM_RESTORE_SNAPSHOT = 4
};

struct catvm_program {
    int left[CATVM_RELATION_CELLS];
    int right[CATVM_RELATION_CELLS];
    int constraint[CATVM_RELATION_CELLS];
};

struct catvm_projection {
    int coefficient[CATVM_RELATION_CELLS];
    uint64_t hash;
    double maximum_root_error;
};

struct catvm_resource_counters {
    uint64_t transactions_sealed;
    uint64_t native_compose_calls;
    uint64_t native_intersection_calls;
    uint64_t native_symbol_products;
    uint64_t phase_cell_updates;
    uint64_t boundary_decodes;
    uint64_t inverse_factor_recomputations;
    uint64_t snapshot_bytes_written;
    uint64_t snapshot_bytes_reloaded;
    uint64_t restoration_cell_checks;
};

struct catvm_restoration {
    enum catvm_restore_mode mode;
    double maximum_abs_error;
    double carrier_integrity_max_abs;
    double maximum_transient_root_error;
    uint64_t generation_before;
    uint64_t generation_after;
    uint64_t lease_id_before;
    uint64_t lease_id_after;
    size_t morphism_depth_after;
    int open_boundary_after;
    int program_loaded_after;
    int pending_operations_after;
    int invariant_state_exact;
    int generation_transition_exact;
    int transient_state_exact;
    int carrier_within_tolerance;
    int used_actual_inverse;
    int used_snapshot_reload;
    int reordered_pair_applicable;
};

struct catvm_machine {
    double complex baseline[CATVM_CARRIER_CELLS];
    double complex working[CATVM_CARRIER_CELLS];
    double complex *snapshot;
    size_t snapshot_mapped_bytes;
    int snapshot_valid;
    struct catvm_program program;
    struct catvm_resource_counters resources;
    enum catvm_backend_kind backend;
    enum catvm_machine_state state;
    uint64_t lease_id;
    uint64_t topology_digest;
    uint64_t baseline_digest;
    uint64_t carrier_creation_count;
    uint64_t restoration_generation;
    size_t morphism_depth;
    int open_boundary;
    int program_loaded;
    int pending_operations;
    int carrier_enabled;
};

const char *catvm_state_name(enum catvm_machine_state state);
const char *catvm_backend_name(enum catvm_backend_kind backend);

int catvm_machine_init(
    struct catvm_machine *machine,
    enum catvm_backend_kind backend
);
void catvm_machine_destroy(struct catvm_machine *machine);

int catvm_program_valid(const struct catvm_program *program);
int catvm_seal(
    struct catvm_machine *machine,
    const struct catvm_program *program
);
int catvm_apply_f(struct catvm_machine *machine);
int catvm_apply_g(struct catvm_machine *machine);
int catvm_project_final(
    struct catvm_machine *machine,
    struct catvm_projection *projection
);
int catvm_restore(
    struct catvm_machine *machine,
    enum catvm_restore_mode mode,
    struct catvm_restoration *restoration
);

double catvm_carrier_maximum_error(const struct catvm_machine *machine);
double catvm_carrier_integrity(const struct catvm_machine *machine);
double catvm_transient_maximum_root_error(
    const struct catvm_machine *machine
);
int catvm_machine_idle_exact(const struct catvm_machine *machine);

#endif
