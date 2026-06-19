CC ?= gcc
CFLAGS ?= -O2 -std=gnu11 -Wall -Wextra -Werror
PYTHON ?= python3

.PHONY: all test clean slot2

SLOT2_DIR := ../../10_cross_core_wormhole/slot2_pdn

all: test_carrier_witness_raw

test_carrier_witness_raw: test_carrier_witness_raw.c carrier_witness_raw.c carrier_witness_raw.h
	$(CC) $(CFLAGS) test_carrier_witness_raw.c carrier_witness_raw.c -o $@

test_slot2_witness: test_slot2_witness.c carrier_witness_raw.c carrier_witness_raw.h \
	$(SLOT2_DIR)/slot2_pdn_lockin.c $(SLOT2_DIR)/slot2_pdn_run.h
	$(CC) $(CFLAGS) -pthread test_slot2_witness.c carrier_witness_raw.c -o $@ -lm

slot2: carrier_witness_raw.c carrier_witness_raw.h $(SLOT2_DIR)/slot2_pdn_lockin.c $(SLOT2_DIR)/slot2_pdn_run.h
	$(CC) -O2 -pthread -Wall -Wextra -Werror -march=amdfam10 \
		$(SLOT2_DIR)/slot2_pdn_lockin.c carrier_witness_raw.c -o slot2 -lm

test: test_carrier_witness_raw test_slot2_witness
	./test_carrier_witness_raw
	./test_slot2_witness
	$(PYTHON) test_carrier_witness_validate.py
	$(PYTHON) test_carrier_witness_finalize.py
	$(PYTHON) replication_discrepancy/test_replication_discrepancy.py
	$(PYTHON) theory_gate_audit/test_reconcile_gate_layers.py
	$(PYTHON) theory_gate_audit/verify_authority_stack.py
	$(PYTHON) -m py_compile audit_existing_t300.py carrier_witness_validate.py carrier_witness_manifest.py carrier_witness_finalize.py \
		replication_discrepancy/analyze_replication_discrepancy.py replication_discrepancy/test_replication_discrepancy.py \
		theory_gate_audit/reconcile_gate_layers.py theory_gate_audit/test_reconcile_gate_layers.py \
		theory_gate_audit/verify_authority_stack.py

clean:
	rm -f test_carrier_witness_raw test_slot2_witness slot2
