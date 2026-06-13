# PHASE2_DEEP_4_MARKERS

## Verdict

PHASE_MARKER_HARNESS_READY

## Source

`session_scripts/phase2_marker_harness.c`

## Function

The marker harness runs on the isolated-core layout:

- Core2 logs marker state and TSC.
- Core3 emits PPU-A workload states.
- Core4 emits PPU-B workload states.
- Core5 emits the reference workload.

It cycles through state words that represent combinations of idle, compute, memory-like, atomic shared-line, and branchy modes. It prints one CSV row per segment:

```text
segment,tsc,state,edge,c3,c4,c5
```

## Sanity Run

Command:

```sh
gcc -O2 -pthread phase2_marker_harness.c -o phase2_marker_harness
./phase2_marker_harness 12 10000 > /tmp/phase2_marker_sample.csv
```

Sample:

```text
segment,tsc,state,edge,c3,c4,c5
0,71291455504002,0x111,1,1311768467463986931,1311768467463528181,1311768467463593719
1,71291487920205,0x121,2,14901733374053020589,3774310008201819642,12100212319698572849
2,71291520270261,0x131,3,13026602269209017763,535129202537593185,1709135074945330213
3,71291552618032,0x11,18201501338794904966,16406172017669111566,1530648047628499769,3305354114958314146
4,71291584964085,0x101,18201501338794904967,11903423603223018753,2621631126089900934,3305354114958314146
5,71291617307326,0x110,18201501338794904968,4218307615831415264,2621631126089900934,10392713223078251883
```

Safety state after sanity run:

```text
P4C3 8000013540003440
P4C4 8000013540003440
k10temp +48.5 C
```

## Use With External Capture

Run a longer capture:

```sh
./phase2_marker_harness 256 50000 > phase2_marker_log.csv
```

Start the scope capture first, then start the marker harness. Align waveform edges to the marker CSV by segment duration and state sequence.

