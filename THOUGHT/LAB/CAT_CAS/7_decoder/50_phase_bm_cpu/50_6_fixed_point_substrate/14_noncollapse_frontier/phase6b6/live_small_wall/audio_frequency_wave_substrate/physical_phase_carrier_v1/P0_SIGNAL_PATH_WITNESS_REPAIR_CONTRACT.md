# P0 actual signal-path witness repair contract

**Status:** `P0_SIGNAL_PATH_WITNESS_REPAIR_CONTRACT_FROZEN`  
**Parent blocked packet:** `d65f9d4f92d708fa54ecc543b776f531d419bb03`  
**Inherited decision:** `P0_BUILD_READINESS_BLOCKED`  
**Inherited blocker:** `P0BR-R3-SIGNAL-POLE`  
**Claim ceiling:** `NON_EXECUTING_P0_SIGNAL_PATH_WITNESS_REPAIR_ONLY`

This contract freezes one non-executing repair candidate. It authorizes no procurement, assembly, wiring, power, instrument command, playback, acquisition, calibration, or physical claim.

## 1. Mechanism defect

The selected Omron `G6K-2F-Y DC5` relays are general signal relays. Their second poles are useful timing witnesses but are not certified force-guided contacts. An auxiliary-pole transition therefore does not prove, under all relevant failure modes, that the signal pole opened during that event.

The existing CH2 sequence remains valuable for command and transition timing. It is not promoted into actual signal-path evidence.

## 2. Selected repair

Freeze an end-to-end electrical transfer witness through the exact K1/K2 signal path:

```text
SDG1032X C2 65.536 kHz reference
        |
        +--> existing passive CH0 reference monitor
        |
        +--> new high-value, current-limited witness injection
              at N_GATE_OUT, downstream of ADG1419 and upstream of K1
                    |
                    K1 signal pole
                    |
                 N_MIDPOINT
                    |
                    K2 signal pole
                    |
             N_ELECTRODE_A / OPA810 / CH1
```

The added branch must be a prospectively frozen, explicitly modeled impedance. Its exact resistor network and C2 amplitude are selected only after the complete circuit model proves adequate pre-open SNR, acceptable carrier disturbance, and bounded post-open capacitive feedthrough.

C2 remains continuously energized and phase-locked to C1 for the complete record. CH0 proves that the reference itself remains present. C2 is not routed through ADG1419, so the ADG1419 DRIVE-to-TERM transition cannot by itself make the downstream C2 witness disappear.

## 3. Required temporal geometry

The witness is adjudicated before K3 can guard the midpoint:

```text
1. C1 and C2 continuously on.
2. C2 transfer through K1/K2 is established on CH1 before t_gate.
3. ADG1419 routes C1 away from DRIVE to its 50.00 ohm termination.
4. After the frozen 250 us delay, K1 and K2 are commanded to release.
5. K3 remains energized and its guard contact remains open.
6. CH2 reaches and holds auxiliary code 0 for the frozen 1000-sample interval.
7. During that same pre-K3 interval, the CH1/CH0 C2 transfer must enter the prospectively frozen isolated-path region.
8. Only after both the timing witness and actual-path transfer witness pass may K3 deenergize to guard and the final code-8/guard interval begin.
```

K3 may not participate in, clamp, or explain the actual-path witness. Any topology or analyzer ordering that evaluates C2 isolation only after K3 guards is invalid.

## 4. Observable

At bound `f2 = f_witness_hz = 2 * f_carrier_hz`, estimate the complex transfer:

```text
H2(window) = Z_CH1,f2(window) / Z_CH0,f2(window)
```

using the same common timebase and phase gauge. Freeze at minimum:

```text
H2_pre      pre-release transfer distribution
H2_open     transfer during the stable code-0, K3-open interval
A2_pre      CH1 pilot amplitude before release
A2_open     CH1 pilot amplitude during the witness interval
R_drop      A2_open / A2_pre
```

The repair must prospectively bind:

```text
minimum pre-open pilot SNR
maximum isolated-path transfer magnitude
maximum isolated-path transfer uncertainty
minimum pre/open separation
maximum permitted pilot-induced change to the bound-frequency carrier preparation
maximum permitted 65.536 kHz nonlinear or resonant residue
window placement and sample count
rank and condition-number gates
differential clipping gate; true input common-mode observability remains a future prerequisite
```

Thresholds must be derived from the exact component/circuit model and committed synthetic calibration packet before any physical data. They may not be fitted to a future DUT record.

## 5. Exact claim meaning

A passing witness may establish only:

```text
ACTUAL_SOURCE_TO_CARRIER_SIGNAL_PATH_ISOLATED_DURING_THE_EVENT
```

It does **not** establish that both K1 and K2 individually opened. With two series barriers, an observed open end-to-end path means at least one actual series signal contact interrupted the path. The packet must stop claiming individual-pole state unless an additional independent mechanism proves it.

The redundant commanded topology may still be documented, but the per-event physical statement is end-to-end isolation.

## 6. Required controls

The repair packet must exercise, synthetically and in the circuit model:

```text
K1 closed, K2 closed, K3 open              witness must remain present
K1 open, K2 closed, K3 open                path-isolation witness must pass
K1 closed, K2 open, K3 open                path-isolation witness must pass
K1 open, K2 open, K3 open                  path-isolation witness must pass
K1/K2 closed, K3 guarded early              must be rejected as guard-masked
C2 absent                                    must be rejected as no witness
C2 present on CH0 but injected at wrong node must be rejected
ADG1419 OFF with K1/K2 closed                witness must remain present
open-contact parasitic maximum               must remain inside frozen isolation law or block
relay bounce/re-entry                         must be rejected
carrier removed                               bounds detector/feedthrough response
exact 1 pF dummy                              bounds capacitive feedthrough response
nonlinear 2f carrier residue                  must not be misclassified as a closed path
channel swap, phase inversion, scale change   must be rejected
```

A force-guided relay substitution remains a fallback architecture only. It is not the selected repair unless the end-to-end pilot cannot close the loading, SNR, feedthrough, or timing laws.

## 7. Required implementation changes

The local qualification agent must update and regenerate only the P0 build-readiness package as necessary:

```text
P0_FINAL_NETLIST.json
P0_NON_PURCHASING_BOM.json
P0_FABRICATION_RELEASE.json
P0_BUILD_READINESS_PACKET.md
P0_FUTURE_PHYSICAL_EXECUTION_CONTRACT.md
p0_scientific_analyzer.py
schemas and strict parsers
synthetic raw fixtures
mutation suite
review and findings artifacts
roadmap P0 receipt/status
```

The analyzer must make the actual-path gate mechanically prior to K3 guard acceptance. A post-guard transfer check is an additional feedthrough control, never a substitute.

## 8. Repair adjudication

Emit exactly one:

```text
P0_SIGNAL_PATH_WITNESS_REPAIR_ESTABLISHED
P0_SIGNAL_PATH_WITNESS_REPAIR_BLOCKED
P0_SIGNAL_PATH_WITNESS_REPAIR_INCONCLUSIVE
```

`ESTABLISHED` requires:

```text
complete circuit-model closure over the new branch
prospectively frozen numeric gates
positive actual-path witness fixtures
all guard-masking and relay-state adversaries rejected
analyzer/data-flow proof of pre-K3 ordering
updated exact netlist, BOM and fabrication coordinates
four independent PASS reviews
zero open material findings
full deterministic and mutation qualification
```

Only after `ESTABLISHED` may the package reconsider:

```text
P0_BUILD_READINESS_PACKET_FROZEN
```

The next authority boundary remains:

```text
USER_AUTHORITY_FOR_P0_PROCUREMENT_OR_UNPOWERED_BUILD
```

Nothing in this contract grants that authority.
