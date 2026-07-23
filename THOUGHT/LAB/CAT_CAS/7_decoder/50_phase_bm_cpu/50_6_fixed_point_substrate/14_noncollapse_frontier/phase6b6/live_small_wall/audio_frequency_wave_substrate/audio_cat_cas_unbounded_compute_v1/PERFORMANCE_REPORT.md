# Toroidal path-sum performance

Wall timings are local medians and are not evidence identities.

- n=16: phase 4771100 ns; compact DP 125600 ns; Gamma 2.18767
- n=32: phase 10162800 ns; compact DP 247000 ns; Gamma 71913
- n=64: phase 22625900 ns; compact DP 599300 ns; Gamma 1.54676e+14
- n=128: phase 38283800 ns; compact DP 934300 ns; Gamma 1.42776e+33
- n=256: phase 91180100 ns; compact DP 2022200 ns; Gamma 2.43017e+71

Gamma is measured against explicit binary path-work. The compact
integer DP remains faster than this NumPy phase reference, so no
advantage over the best conventional compact algorithm is claimed.
