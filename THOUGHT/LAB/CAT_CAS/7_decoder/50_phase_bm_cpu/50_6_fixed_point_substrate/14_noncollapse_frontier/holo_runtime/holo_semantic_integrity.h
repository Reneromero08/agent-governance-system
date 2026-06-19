#ifndef HOLO_SEMANTIC_INTEGRITY_H
#define HOLO_SEMANTIC_INTEGRITY_H

#include "holo_geometry.h"
#include "../l4b_orbitstate/holo_path_history.h"

/*
 * Validate that every authenticated path record is also a valid application of
 * the declared orbit_coupled_phase_walk_v1 operator. Structural digests alone
 * are insufficient because a self-consistent forged record can recompute them.
 */
int holo_path_history_validate_semantic(const HoloPathHistory *history,
                                        const OrbitState *initial_state,
                                        OrbitState *terminal_state_out);

/* Validate geometry, path semantics, lifecycle coherence, and restoration scope. */
int holo_object_validate_semantic(const HoloObject *object);

/*
 * Transactional wrapper around holo_cross_boundary(). On failure, path sealing,
 * boundary state, invariant state, and lifecycle strings are restored.
 */
int holo_cross_boundary_atomic(HoloObject *object, int step);

/*
 * Strict artifact reader. It verifies the serialized lifecycle fields before
 * accepting the normalized HoloObject produced by the legacy reader, then runs
 * semantic object validation.
 */
int holo_read_json_strict(HoloObject *object, const char *path);

#endif
