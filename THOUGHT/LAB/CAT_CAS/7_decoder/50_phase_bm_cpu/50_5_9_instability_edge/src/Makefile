CC = gcc
CFLAGS = -O2 -march=native -Wall -Wextra -std=c11
LDFLAGS = -pthread -lm -lrt

SRCS = phase5_9_stress_ladder.c phase5_9_workers.c
BIN = phase5_9_stress_ladder

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRCS) phase5_9_common.h phase5_9_workers.h
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(BIN) *.o
	rm -f raw_cycles.csv restoration_integrity.csv stress_ladder.csv
	rm -f operating_points.csv worker_status.csv TELEMETRY_PHASE5_9.txt
