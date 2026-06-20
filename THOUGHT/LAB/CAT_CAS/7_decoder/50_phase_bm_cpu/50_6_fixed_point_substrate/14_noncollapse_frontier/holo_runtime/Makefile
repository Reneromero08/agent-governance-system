CC ?= gcc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror

RUNNER := combined_pdn_runner

.PHONY: all test

all: $(RUNNER)

$(RUNNER): combined_pdn_runner.c
	$(CC) $(CFLAGS) $< -o $@

test: $(RUNNER)
	python3 test_combined_pdn_runner.py
