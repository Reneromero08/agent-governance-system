# F103 C17 Quadratic Mode-Mixing No-Go: Independent Reexecution Review

Date: 2026-08-03

Decision: `INDEPENDENTLY_VERIFIED_STRICT_SCOPE`

Verification level: `INDEPENDENT_ORACLE_REEXECUTION`

Restoration classification: `EXACT_ALGEBRAIC_RESTORATION`

Transient-buffer classification: `NO_RESTORATION_CLAIM`

## Scope

The reviewed package is a Linux direct-process exact finite-field diagnostic.
It uses 51 general `F103[C17]` factors, or 867 resident field coordinates, on
the declared rotating-hub topology. It interleaves cyclic-convolution
triangular shears with one coefficientwise quadratic shared-port shear per
depth step. Three public families were executed at depths 1, 4, 16, 64, 256,
and 1024.

The package is not a CATVM custody result and is not physical waveform
execution.

## Durable evidence

| Artifact | SHA-256 |
|---|---|
| `f103_c17_quadratic_mode_mixing_no_go.py` | `24a1ed77bfc13e7ffe30b1fd1182e23f3907059c459a227a9a4d87a35da2220d` |
| `f103_c17_quadratic_mode_mixing_no_go_oracle.py` | `a96e669ca9b935305db656a35f1bf4d9a7894dd20e8106f8133f80a6ccbc41a5` |
| `f103_c17_superposition_interference_factor_no_go.py` | `a8e4dac14366fb4edbeb7335ba938f8675e352216840d3cb8a3a8c5a8f93a86a` |
| `F103_C17_QUADRATIC_MODE_MIXING_NO_GO_RESULTS.json` | `8281dfd5b01699092f7467766adcf545e93b29eaa3139353e6cddbe1c11ac41b` |
| `F103_C17_QUADRATIC_MODE_MIXING_NO_GO_ORACLE_RESULTS.json` | `0953ac92f1486ddc9c95154be7e3df53030db828ece1d1adad0e19af5f52b838` |
| `qualify_f103_c17_quadratic_mode_mixing_no_go.sh` | `2e23f611c099d9668e789ec5a85115980b59e5abadd1391080b67ca4f0749fdb` |

The strict qualifier reexecuted production and oracle into
`/dev/shm/ags-audio-m151-qualifier-rGdeOp`, byte-compared both outputs to the
sealed repository results, and returned
`QUALIFIED_F103_C17_QUADRATIC_MODE_MIXING_NO_GO_STRICT_SCOPE`.

## Independent reconstruction

The oracle imports neither production nor M150 and does not use NumPy. It
reconstructs the public hub, rotation, linear-offset, quadratic-offset,
observation, and seed formulas. It executes two separate implementations:

1. a pure-Python 867-coordinate coefficient recurrence; and
2. a coupled 17-mode NTT recurrence followed by reconstruction of all 867
   coefficient cells.

Across all 18 cases, 180 independent comparisons passed. They cover the final
state commitment, 17-coordinate boundary, support extrema, coefficient and
spectral inverse restoration, coefficient-versus-coupled-spectral equality,
and both counted nonlinear multiplication laws.

## Findings

The coefficientwise square changes the machine law relative to M150. With the
package's NTT convention,

```text
NTT(a coefficientwise-square a)
    = inverse(17) times circular-self-convolution(NTT(a)).
```

Therefore the 17 character modes no longer evolve independently. The
independent-mode square sham changes the final state, and an explicit
two-mode input generates a third output mode. Each quadratic output mode has
17 input-mode terms in the declared self-convolution.

This does not establish a distinct phase resource. The pure-Python classical
coefficient recurrence uses exactly the same 867 field coordinates and
reconstructs every accepted result. The coupled spectral recurrence also uses
867 coordinates. At depth 1024, the evaluated coefficient path counts
18,957,312 convolution-plus-quadratic core multiplications, while the coupled
spectral path counts 1,410,048. No optimal classical baseline is claimed, but
these executed matched recurrences are sufficient to reject an advantage for
this package.

The accepted carrier retains no inverse-history cells and no restoration
baseline cells. The forward operations are reversed on the same NumPy backing
array, the restoration generation increments exactly, an unrelated depth-613
program agrees with a fresh carrier, and 64 depth-8 reuse cycles restore
exactly. No snapshot reload occurs.

All declared controls pass: missing, wrong, and reordered inverse; premature
and resident-port projection; wrong owner; null carrier and null port; module
reordering; topology mutation; NTT basis round trips; nonlinear and cross-mode
witnesses; and rejection of the independent-mode sham.

## Claim ceiling

The accepted ceiling is the declared `F103[C17]` convolution-plus-quadratic
rotating-hub family across 18 direct-process cases through depth 1024. The
result establishes exact mode coupling, final-only boundary projection,
algebraic restoration, and restored-carrier reuse in this finite software
model.

It does not establish general relational contraction, CATVM custody, a
distinct phase resource, computational advantage, a Small Wall crossing,
physical waveform execution, replacement of physical bits with pi, or
unbounded catalytic computation.

## Next obstruction

The nonlinear primitive defeats simultaneous diagonalization but remains a
fixed 867-coordinate polynomial map with direct compact classical
recurrences. A successor must change that representational law rather than
add depth or fanout to the same finite-coordinate map.
