CC ?= gcc
CFLAGS ?= -O2 -std=gnu11 -Wall -Wextra -Werror
LDLIBS ?= -lm -lcrypto

HOLO_RUNTIME = ../holo_runtime/holo_geometry.c \
	../holo_runtime/holo_invariant_family.c \
	../holo_runtime/holo_physical_mapping.c \
	../holo_runtime/holo_observability_design.c \
	../holo_runtime/holo_observability_governance.c \
	../holo_runtime/holo_semantic_integrity.c \
	holo_path_history.c

.PHONY: all test run clean

all: l4b_orbitstate test_holo_geometry test_holo_path_history \
	test_holo_invariant_family test_holo_physical_mapping \
	test_holo_observability_design test_holo_observability_governance \
	test_holo_semantic_integrity

l4b_orbitstate: l4b_orbitstate_evolution.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test_holo_geometry: test_holo_geometry.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test_holo_path_history: test_holo_path_history.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test_holo_invariant_family: test_holo_invariant_family.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test_holo_physical_mapping: test_holo_physical_mapping.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test_holo_observability_design: test_holo_observability_design.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test_holo_observability_governance: test_holo_observability_governance.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test_holo_semantic_integrity: test_holo_semantic_integrity.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: test_holo_geometry test_holo_path_history test_holo_invariant_family \
	test_holo_physical_mapping test_holo_observability_design \
	test_holo_observability_governance test_holo_semantic_integrity
	./test_holo_geometry
	./test_holo_path_history
	./test_holo_invariant_family
	./test_holo_physical_mapping
	./test_holo_observability_design
	./test_holo_observability_governance
	./test_holo_semantic_integrity

run: l4b_orbitstate
	mkdir -p results
	./l4b_orbitstate --N 256 --a 23 --steps 512 --seed 42

clean:
	rm -f l4b_orbitstate test_holo_geometry test_holo_path_history \
		test_holo_invariant_family test_holo_physical_mapping \
		test_holo_observability_design test_holo_observability_governance \
		test_holo_semantic_integrity holo_geometry_test.holo \
		holo_path_history_test.holo holo_invariant_family_test.holo \
		holo_physical_mapping_test.json holo_observability_design_test.json \
		holo_observability_governance_design.json \
		holo_observability_governance_review.json \
		holo_semantic_integrity_test.holo
