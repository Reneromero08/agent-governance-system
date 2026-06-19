CC ?= gcc
CFLAGS ?= -O2 -std=gnu11 -Wall -Wextra -Werror
PYTHON ?= python3

.PHONY: all test clean

all: test_carrier_witness_raw

test_carrier_witness_raw: test_carrier_witness_raw.c carrier_witness_raw.c carrier_witness_raw.h
	$(CC) $(CFLAGS) test_carrier_witness_raw.c carrier_witness_raw.c -o $@

test: test_carrier_witness_raw
	./test_carrier_witness_raw
	$(PYTHON) test_carrier_witness_validate.py
	$(PYTHON) -m py_compile audit_existing_t300.py carrier_witness_validate.py carrier_witness_manifest.py

clean:
	rm -f test_carrier_witness_raw
