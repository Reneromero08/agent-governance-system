# Phase-native computer resources

Timings are environment-specific medians. No performance advantage is claimed.

## Program benchmarks

- `prospective_affine_mod7`: phase 650100 ns, conventional comparison 600 ns, ratio 1.08e+03x
- `prospective_binary_add3`: phase 1285300 ns, conventional comparison 2400 ns, ratio 536x
- `prospective_mux_xor_pipeline`: phase 672600 ns, conventional comparison 500 ns, ratio 1.35e+03x
- `prospective_route_compose_mod7`: phase 458300 ns, conventional comparison 600 ns, ratio 764x

## Scaling

- 2 registers: 8192 carrier bytes, 24800 total execution-array bytes, 375400 ns
- 4 registers: 16384 carrier bytes, 49856 total execution-array bytes, 527100 ns
- 8 registers: 32768 carrier bytes, 100736 total execution-array bytes, 814700 ns
- 12 registers: 49152 carrier bytes, 152640 total execution-array bytes, 1093600 ns
- 16 registers: 65536 carrier bytes, 205568 total execution-array bytes, 1439300 ns

The logical phase state is 16 bytes per register, with two active spectral components per register and zero complete configuration modes.
The retained reversible history is O(registers * instructions); it becomes quadratic only in the scaling probe because that probe deliberately uses one instruction per register.
