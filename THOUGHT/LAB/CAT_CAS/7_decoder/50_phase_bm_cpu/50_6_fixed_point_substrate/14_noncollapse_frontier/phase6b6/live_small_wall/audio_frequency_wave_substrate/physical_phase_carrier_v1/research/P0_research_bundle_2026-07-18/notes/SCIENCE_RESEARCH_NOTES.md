# Science research notes

## Most relevant references for the non-hardware science

1. **QTF resonance tracking / transient methods**: establishes that a standard 32.768 kHz fork can be characterized through transient, beat-frequency and fitted equivalent-circuit methods.
2. **Voltage-mode QTF readout SNR analysis**: provides a worked BVD plus input-capacitance/load model and SPICE comparison at 32,768 Hz.
3. **Passive electrical damping**: explains why the electrode shunt capacitance and external load affect energy flow and observed Q.
4. **Voltage-induced frequency shift**: demonstrates preparation voltage can perturb fork frequency, requiring a frozen drive envelope.
5. **Thin-film quartz and silicon phononic resonators**: support future translation only.

## Direct implication for the WIP simulator

The present synthetic waveform should become the output of a source-cited generative model. The analyzer itself should remain unchanged so the stronger model cannot tune the adjudicator after seeing outcomes.

## Null structure to preserve

- resonator removed;
- matched 1 pF C0G dummy;
- zero drive;
- source left on;
- source muted rather than isolated;
- wrong termination;
- gate-only and relay-only ablations;
- wrong phase, wrong frequency and timing perturbations.
