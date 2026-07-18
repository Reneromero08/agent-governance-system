# P0 research synthesis

## Executive result

The P0 software experiment is scientifically legitimate at the **signal-analysis level**: it generates synthetic four-channel records, processes those bytes through the actual analyzer, recovers phase/frequency/decay, and rejects implemented adversaries. The missing bridge is a **device-and-circuit-level model** that predicts those records from the quartz Butterworth–Van Dyke model, source switch, relay network, OPA810 input, digitizer loading, cable/PCB parasitics and noise.

The new research materially supports that bridge:

1. Standard 32.768 kHz quartz tuning forks are routinely represented by a motional series RLC branch in parallel with electrode capacitance. Time-domain transient and ringdown methods are established ways to estimate resonance and Q.
2. Published voltage-mode QTF work shows that input capacitance and load resistance alter peak frequency, response and signal-to-noise ratio. A high-impedance detector is not a neutral observer.
3. Passive electrical damping research likewise shows that the parallel capacitance redirects energy and changes damping. This directly supports the need for the empty and 1 pF dummy controls.
4. Vendor simulation models exist for the ADG1419 and OPA810. A meaningful pre-hardware simulation is therefore achievable without inventing behavioral models for the two most consequential active components.
5. Recent thin-film quartz and silicon phononic work supports the future carrier translation, but none of it constitutes evidence that the P0 apparatus will work.

## Internal consistency of the current synthetic reference

The current synthetic reference uses approximately:

```text
f = 32,768 Hz
Q = 18,530
tau = 0.18 s
```

For an amplitude envelope written as `exp(-pi*f*t/Q)`, the implied amplitude time constant is:

```text
tau = Q / (pi*f) = 0.180001288 s
```

So the synthetic `Q` and `tau` are mutually consistent. That is a real positive result about the simulator's internal physics convention.

## What the research closes

- Exact official landing pages and likely direct-download locations for all 24 core registry records.
- Current revision reconciliation for ADR45xx, SHT4x, ST UM2591 and several other documents.
- A defensible literature basis for QTF ringdown, BVD modeling, voltage-mode readout loading, damping and future phononic translation.
- A concrete simulation-resource path using the vendor ADG1419 and OPA810 models.
- A reproducible source-custody workflow instead of hashes with missing bytes.

## What research alone cannot close

The following are not missing-web-research problems. They are model, calibration or experiment variables:

- Loaded FC-135 motional `R`, `L`, `C`, shunt `C0`, resonance and Q for each mounted unit.
- OPA810 input capacitance/leakage including board, relay, contamination and enclosure parasitics.
- ADG1419 charge injection, feedthrough and isolation in the actual source/load network.
- Relay bounce and release-time distributions in the actual suppression network.
- Spectrum digitizer differential/common-mode input admittance and clock behavior.
- SIGLENT interchannel phase coherence, phase-command latency and output accuracy for the actual firmware.
- Cable, PCB and ground-return parasitics.
- Environmental and mounting sensitivity.

These should become parameter sweeps in simulation, followed later by measured priors or coupon calibration.

## Recommended next scientific simulation

Build a reproducible transient model with these layers:

```text
SDG Thevenin source and continuous 2x phase reference
-> 100 kohm limiter
-> ADG1419 vendor model and 50-ohm OFF termination
-> relay contacts represented first as ideal scheduled switches, then bounce/leakage sweeps
-> FC-135 BVD model plus package shunt capacitance
-> OPA810 vendor model, bias return, output resistor and digitizer load
-> measured/estimated cable and PCB capacitances
-> quantization and calibrated analog noise
-> the existing raw-bundle writer
-> the unchanged P0 scientific analyzer
```

Run parameter sweeps rather than one favorable nominal case. The important output is not merely whether the positive synthetic case passes. It is the region of physical parameter space in which:

- `0` and `pi` remain distinguishable after isolation;
- empty and 1 pF dummy controls fail as intended;
- source-monitor and witness requirements remain satisfied;
- feedthrough and switching transients cannot enter the admitted interval;
- detector loading does not erase or fabricate the relation.

## Claim boundary

A successful circuit/device simulation would establish a stronger software result:

> Under an explicit, source-cited electromechanical and circuit model, the proposed P0 topology is predicted to generate analyzer-admissible phase-retaining ringdown over a declared parameter region and to fail its matched controls outside that mechanism.

It still would not establish a physical observation. It would, however, be substantially more scientific than directly constructing the desired waveform.
