CC = gcc
CFLAGS = -O2 -march=native -Wall -Wextra -std=c11
CFLAGS_DBG = -O0 -g -Wall -Wextra -std=c11
LDFLAGS = -pthread -lm -lrt

SRCS = phase5_8_boundary_rdtsc.c phase5_8_workers.c
OBJS = $(SRCS:.c=.o)
BIN = phase5_8_boundary_rdtsc

.PHONY: all clean debug run analyze report

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.c phase5_8_common.h phase5_8_workers.h
	$(CC) $(CFLAGS) -c -o $@ $<

debug:
	$(CC) $(CFLAGS_DBG) $(LDFLAGS) -o $(BIN)_dbg $(SRCS)

clean:
	rm -f $(BIN) $(BIN)_dbg $(OBJS)
	rm -f raw_cycles.csv window_features.csv geometry_stats.csv
	rm -f projection_stats.csv restoration_integrity.csv operating_points.csv
	rm -f verdict_gate_audit.csv TELEMETRY_PHASE5_8.txt
	rm -f area_law_scaling_stats.csv silicon_transition_stats.csv

run: $(BIN)
	./run_phase5_8.sh

analyze:
	python3 analyze_phase5_8.py --input-dir .

report: analyze
	@echo "=== EXP50 PHASE 5.8 REPORT ==="
	@cat TELEMETRY_PHASE5_8.txt 2>/dev/null || echo "No telemetry yet"
	@echo ""
	@echo "=== VERDICT GATES ==="
	@cat verdict_gate_audit.csv 2>/dev/null || echo "No verdict yet"
