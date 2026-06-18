CC ?= gcc
CFLAGS ?= -O2 -std=gnu11 -Wall -Wextra -Werror
LDLIBS ?= -lm

HOLO_RUNTIME = ../holo_runtime/holo_geometry.c ../holo_runtime/holo_invariant_family.c holo_path_history.c

.PHONY: all test run clean

all: l4b_orbitstate test_holo_geometry test_holo_path_history test_holo_invariant_family

l4b_orbitstate: l4b_orbitstate_evolution.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test_holo_geometry: test_holo_geometry.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test_holo_path_history: test_holo_path_history.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test_holo_invariant_family: test_holo_invariant_family.c orbit_state.c $(HOLO_RUNTIME)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: test_holo_geometry test_holo_path_history test_holo_invariant_family
	./test_holo_geometry
	./test_holo_path_history
	./test_holo_invariant_family

run: l4b_orbitstate
	mkdir -p results
	./l4b_orbitstate --N 256 --a 23 --steps 512 --seed 42

clean:
	rm -f l4b_orbitstate test_holo_geometry test_holo_path_history test_holo_invariant_family \
		holo_geometry_test.holo holo_path_history_test.holo holo_invariant_family_test.holo
