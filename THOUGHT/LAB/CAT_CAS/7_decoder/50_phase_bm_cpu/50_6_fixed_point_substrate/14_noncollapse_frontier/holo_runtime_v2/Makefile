CC ?= gcc
CFLAGS ?= -std=c11 -O2 -pthread -Wall -Wextra -Werror
LDLIBS ?= -lm

RUNNER := combined_pdn_runner

.PHONY: all test

all: $(RUNNER)

$(RUNNER): combined_pdn_runner.c combined_pdn_hardware.c captured_file.c combined_pdn_hardware.h captured_file.h
	$(CC) $(CFLAGS) combined_pdn_runner.c combined_pdn_hardware.c captured_file.c -o $@ $(LDLIBS)

test: $(RUNNER)
	python3 test_combined_pdn_runner.py
