# P0 Measurement and Source-Off Plan

**Status:** `P0_BUILD_READINESS_BLOCKED__NOT_EXECUTED`<br>
**Carrier:** hermetic 32.768 kHz quartz tuning-fork mechanical mode<br>
**Source-off topology:** phase gate plus witnessed guarded relay barrier<br>
**Primary observable:** `z(t) = I(t) + iQ(t)`<br>
**Execution authority:** none

## 1. Frozen measurement chain

```text
phase-coherent low-voltage sine source C1 in the source domain
  -> passive 100 kOhm CH0 source-monitor branch at C1_IN
  -> separate 100 kOhm minimum current-limiting resistor
  -> ADG1419BRMZ 8-lead MSOP SPDT at +/-5 V, EN not present,
     switched only through IN
       DRIVE: guarded relay input
       OFF:   50 Ohm source termination
  -> K1 DPDT series relay (signal pole plus independent witness pole)
  -> K3 DPDT midpoint guard (signal to 50 Ohm; spare witness pole)
  -> K2 DPDT final series relay (signal pole plus independent witness pole)
  -> electrode A of the two-terminal quartz tuning fork

electrode B -> single carrier-side analog-reference bond
electrodes A/B -> as-built Rin,U95 >=100 MOhm, Cin,U95 <=4.00 pF voltage input
                  plus fixed >=100 MOhm common-mode bias return
               -> CH1 mechanical-sense voltage

analog-gate logic plus K1/K2/K3 dry-contact witnesses
  -> calibrated 4-bit resistor code on CH2
enclosure accelerometer -> CH3
temperature / humidity -> timestamped environment record
four simultaneous differential raw channels -> immutable native capture
```

`K1` and `K2` are normally open contacts and `K3` is a normally closed
midpoint-to-50-Ohm guard contact. Coil power is required to create the drive
path. Loss of controller or relay power therefore produces the source-off
state: both series contacts open and the midpoint terminates on the source side.
Electrode A remains connected only to the characterized high-impedance sense
load; electrode B remains at the single carrier-side reference bond.

The analog phase gate gives deterministic preparation timing. CH2 gives an
auditable auxiliary-contact timing state, but its spare contacts do not prove
the actual signal poles opened. Neither alone is sufficient for P0. A physical
source-disconnect claim remains blocked until a separately reviewed per-event
actual-signal-path witness or an exact force-guided-contact guarantee is bound
to the selected relay and its failure modes.

CH2 uses four weighted resistors to encode the gate logic state and the three
independent spare relay contacts into 16 prospectively calibrated voltage
levels. Adjacent levels must be separated by at least 10 times the CH2
calibration noise standard deviation. An undecodable or illegal level kills the
arm; no nearest-level guess is allowed.

The prospective CH2 code is exact. Bits are active-high closed/DRIVE states:
`b0=gate DRIVE`, `b1=K1 closed`, `b2=K2 closed`, `b3=K3 closed-to-guard`.
Conductance weights are in the ratio `1:2:4:8` with each fitted value within
0.1% of its sealed calibration value. Code integer is
`c=b0+2*b1+4*b2+8*b3`. For calibrated centroid `mu_c` and worst-case zero-drive
standard deviation `sigma_ch2`, the only accepted band is
`abs(v-mu_c) <= 3*sigma_ch2`; bands must be disjoint and adjacent centroids
must remain at least `10*sigma_ch2` apart. Values outside all bands reject.

Lawful stable states are DRIVE `c=7`, SERIES-OPEN/K3-HELD `c=0`, and guarded
OFF `c=8`. After 250 samples of `c=6`, K1/K2 may transition only through
`{0,2,4,6}` while K3 remains energized. Code 0 must then remain stable for
1,000 consecutive samples before K3 is permitted to release. K3 transition may
use only `{0,8}`, after which code 8 must remain stable for 1,000 consecutive
samples. `t_gate` is the first sample not decoded as DRIVE after the command;
`t_series_open` and `t_contact` are the 1,000th samples of the code-0 and code-8
stability runs. The full ordered transition must finish within 14,500 samples.
The extra 10 ms guard begins only at `t_contact`. Hysteresis and nearest-code
filling are forbidden. Re-entry to any non-8 code after the final stable run
kills the arm.

The preferred relay reference class is Omron G6K-2F-Y: maximum operate time
3 ms, maximum release time 3 ms, maximum bounce time 3 ms, and minimum
insulation resistance 1,000 MOhm at 500 VDC. The frozen analog-gate candidate
is ADG1419BRMZ in the 8-lead MSOP package at +/-5 V. That package has no EN
pin; IN performs the DRIVE-to-OFF route transition. At +/-5 V its specified
full-temperature transition maximum is 560 ns and typical on resistance is
4.5 Ohm. Nonzero charge injection, off capacitance, and the loaded transition
must be measured as disturbances rather than inferred from the unloaded bound.

All source/control grounds, shields, and returns are frozen as follows: C1 is
the only intentional low-impedance source return and one RG-178 shield is the
only intentional low-impedance `AGND_EXPORT` to `AGND_STAR` bond. Relay coils
and gate logic use isolated control power; coil suppression closes only in the
control domain. The digitizer inputs are true differential but are not
galvanically isolated. Their positive-to-ground, negative-to-ground, and
differential admittances are calibrated parasitic return paths included in the
loaded model; they are not claimed to vanish or to provide isolation. A
continuity survey and injection scan must prove this netlist before any later
execution. CH2 records contact witnesses; no unallocated midpoint analog
monitor is claimed.

C2 is not an intended carrier drive, but it is not topologically zero-coupled.
The passive CH0 summing network provides a bounded linear path from C2 through
`R_MON_C2`, the monitor node, `R_MON_C1`, finite C1 source impedance, and
`R_LIMIT` toward the carrier while the source path is in DRIVE. The complete
device/circuit model and the resonator-removed/dummy controls must include and
bound that path. No later result may rely on the false shorthand that C2
"never drives" or is perfectly isolated from the carrier.

Official sources:

- [Analog Devices ADG1419 datasheet](https://www.analog.com/media/en/technical-documentation/data-sheets/adg1419.pdf)
- [Omron G6K relay datasheet](https://components.omron.com/system/files/2026-06/datasheet_pdf/K106-E1.pdf)

No part has been purchased, wired, powered, or physically contacted. No human
vendor outreach, cart, stock check, or procurement contact occurred. Automated
public-source HTTP retrieval attempts are disclosed separately in the research
custody snapshot and private ignored receipt.

## 2. Frozen timing and acquisition envelope

| Parameter | Frozen value |
|---|---:|
| Nominal carrier class | 32.768 kHz |
| Native acquisition rate | exactly 1,000,000 samples/s/channel |
| Simultaneous analog channels | 4 |
| Minimum digitizer resolution | 16 bits |
| Coupling | DC |
| Hardware averaging / DSP filter | disabled |
| Raw record indices | `n = 0..3,100,999` (3,101,000 samples/channel) |
| Nominal source-off command index | `n_cmd = 1,101,000` |
| Pre-command record | exactly 1,101,000 samples = 1.101000 s |
| Post-command record | exactly 2,000,000 samples = 2.000000 s |
| Preparation-onset requirement | observed onset index `n_prep >= 100,000` |
| C1 source command | exactly 32,768 Hz, 0.400 Vpp, 0 V offset, `HIGH_Z` load mode, 50 Ohm physical output |
| C2 gauge command | exactly 65,536 Hz, 0.100 Vpp, 0 V offset, zero phase, `HIGH_Z` load mode, 50 Ohm physical output |
| Qualified preparation interval | exactly the 32,768 continuous C1 cycles immediately preceding `n_gate` |
| Qualified preparation duration | 1.000000 s at nominal frequency |
| Turn-on inside admitted record | none; both outputs are continuous and already stable before recording |
| Constant-amplitude interval | entire record; the final 32,768 pre-gate cycles are the qualified preparation |
| Turn-off ramp | none; source is physically routed away |
| Arm phase offset | exactly 0 or pi |
| Analog gate command-to-logic-edge calibration bound | <= 1.000 us; guard timing only |
| Analog gate IN transition bound at +/-5 V | <= 0.560 us |
| Relay command delay after analog gate | 0.250 ms |
| K1/K2 release plus bounce budget | <= 6.000 ms before the code-0 stability run |
| K3 release plus bounce budget | <= 6.000 ms, commanded only after code 0 is stable for 1.000 ms |
| Complete ordered contact transition | <= 14.500 ms from `t_gate` through the 1,000th code-8 sample |
| Isolation guard | 10.000 ms after the last witnessed contact transition |
| First admissible raw sample | first sample at or after the guard boundary |
| Digitizer channel skew before correction | <= 0.100 us |
| Residual CH0/CH1 phase-time skew after calibration | <= 0.020 us |
| Gate-witness sampling uncertainty | <= 1 sample = 1.000 us; not used as the phase gauge |
| Voltage detector lower bandwidth bound | >= 327.68 kHz |
| Voltage detector small-signal settling budget | <= 10.000 us |
| Controller / driver data buffer in post-source path | none |
| Temperature / RH record cadence | exactly 10 samples/s, mapped to the raw sample counter by the frozen monotonic-clock mapping |
| Primary temperature envelope | 20.00-30.00 C; matched-arm mean delta <= 0.20 C |
| Primary RH envelope | 20.0-60.0 %RH; matched-arm mean delta <= 2.0 percentage points |
| Primary CH3 acceleration | demeaned raw RMS <= 0.050 m/s^2; peak absolute <= 0.500 m/s^2 |
| Matched-arm CH3 RMS delta | <= 0.010 m/s^2 |

The source is already continuously stable when the software-prearmed free-running
record begins. The record contains at least 100,000 samples before the final
qualified 32,768-cycle preparation interval, contains that complete interval,
and continues for 2,000,000 samples after the nominal source-off command. The
digitizer therefore captures continuously across preparation and source-off.
There is no trigger cable. CH2 establishes the source-off event; CH0 establishes
the phase gauge.

CH0 must match the sample-level reconstructed C1+C2 waveform over every sample of
the complete 1,000,000-sample qualified preparation interval with peak residual
no greater than 5 percent of fitted C1 amplitude. This is independent of the
post-gate whole-record reconstruction and contiguous persistence-segment checks.

Define:

```text
t_gate = observed CH2 analog-gate witness transition
t_series_open = 1,000th stable code-0 sample while K3 remains energized
t_contact = 1,000th stable code-8 sample after K3 is permitted to release
t_iso = max(t_gate + 0.560 us, t_contact)
t_admit = t_iso + 10.000 ms
```

`t_gate` and `t_contact` are guard-boundary quantities, not the phase gauge.
They are rounded conservatively to the next raw-sample time before adding the
10.000 ms guard. Phase uses the continuous CH0 dual-tone gauge fit below, so the
1 us witness sampling does not enter the half-turn phase estimate. The
calibrated residual CH0/CH1 skew contributes at most 0.004118 rad at 32.768 kHz
and is included in `u_phi`.

If any contact witness is missing, contradictory, metastable, out of order, or
changes after `t_admit`, the arm is killed. If `t_contact - t_gate > 14.500 ms`
including both 1,000-sample stability runs, the arm is killed. A software
command timestamp never substitutes for `t_iso`.

## 3. Real source-off law

The lawful post-source interval requires all of the following:

1. CH0 proves the source remains present upstream or is terminated exactly as
   declared; it does not disappear into an unobserved software mute.
2. The analog gate is in the OFF route to 50 Ohm.
3. K1 and K2 auxiliary poles are stably open before K3 is allowed to release.
4. K3's auxiliary pole is then stably in its deenergized guard state.
5. No driver, coupling capacitor, active filter, transformer, or output buffer
   exists downstream of K2. The only connected CH1 load is the sealed
   high-impedance differential input, its protection, and its passive bias
   return.
6. CH1 is within its linear range and its analog settling budget expired. Its
   measured input admittance is `Rin,U95 >= 100 MOhm`, `Cin,U95 <= 4.00 pF`; coherent
   detector-only response at `f_ref` is below `T_feed`, and detector-only
   impulse lifetime is <=10 us.
7. The 10 ms guard expired after the final physical witness transition.
8. Dummy-load and resonator-removed controls bound residual feedthrough under
   the frozen feedthrough law.
9. A separately reviewed per-event actual-signal-path witness or exact
   force-guided-contact guarantee proves the signal poles corresponding to
   items 3 and 4. Auxiliary-contact code alone never satisfies this item.

The loaded BVD model is the datasheet motional `R1-L1-C1` series branch in
parallel with `C0`, `Rin`, `Cin`, the passive bias return, carrier-side cable
capacitance, and measured shield/ground admittance. P0 reports only the loaded
`f`, `tau_A`, and `Q`; unloaded datasheet planning values cannot substitute.

The source-off state is therefore electrically auditable and fail-safe. The
analog gate's charge injection, relay contact bounce, relay coil impulse,
capacitive feedthrough, and detector recovery are explicit disturbances.

Kill P0 if any accepted post-source interval is still compatible with source
drive, switch transient, driver storage, detector storage, reference coupling,
digitizer memory, or offline filter memory at or above the frozen control
bounds.

## 4. Matched preparations

| Field | Arm 0 | Arm pi |
|---|---|---|
| Intended phase offset | `delta = 0` | `delta = pi` |
| Calibrated drive frequency | identical `f_ref` | identical `f_ref` |
| C1 source command | 0.400 Vpp, 0 V, `HIGH_Z` | identical |
| C2 gauge command | 0.100 Vpp, 0 V, fixed zero phase | identical |
| Carrier terminal planning bound | <= 0.164658 Vpp; hard cap <= 0.200 Vpp | identical |
| Estimated motional power | planning <= 0.048415 uW; hard cap <= 0.100 uW | identical |
| Estimated motional current | planning <= 0.831646 uA rms; hard cap <= 2.000 uA rms | identical |
| Qualified preparation cycles | final 32,768 cycles of continuous source | identical |
| Turn-on ramp inside record | none | none |
| Constant interval | entire record | same |
| Source-off schedule | frozen timing above | same |
| Termination | analog 50 Ohm plus guarded relay | same |
| Acquisition | same rate, range, channels, duration | same |
| Offline projection | same version and parameters | same |
| Environment | frozen temperature, RH, and CH3 acceleration limits | same |

The only intended difference is the source phase half-turn. Arm labels are
opaque random-free identifiers `A` and `B` in acquisition; the 0/pi mapping is
sealed in a pre-acquisition assignment file and revealed only after hashes and
control validity close.

For each arm, temperature and RH are the arithmetic means of every 10 Hz
environment record whose nearest raw sample lies from `n_admit` through the
last usable I/Q-window center, inclusive. The arm rejects unless every record
is within the primary envelope. CH3 acceleration uses every raw sample over
that same closed interval: subtract its ascending-index arithmetic mean, then
compute `sqrt(sum(a[k]^2)/N)` and the maximum absolute demeaned sample. No
filtering, resampling, or spectral weighting is allowed. Nonfinite, clipped,
missing, or cadence-violating environment data reject. Both arms must satisfy
the individual bounds and the frozen matched-arm deltas above.

Required negative preparations:

```text
amplitude mismatch:  pi arm C1 at 0.320 Vpp, exactly 0.800 times the frozen command
frequency mismatch:  pi arm at f_ref + max(20 Hz, 20 calibrated linewidths)
timing mismatch:     pi arm source-off 0.250 carrier cycle later
wrong phase:         phase offset pi/2
random-phase set:    fixed prospective offsets [pi/7, 3*pi/7, 5*pi/7]
source-left-on:      analog gate and relays remain in DRIVE
```

The fixed random-phase set contains no runtime randomness.

## 5. Reference and I/Q extraction

### 5.1 Reference construction

Calibration before primary acquisition freezes `f_ref`, the continuous C2
phase-gauge method, channel skew, and the software-prearmed record-start law.
There is no hardware trigger. Within each record, sample time is
`t[n]=n/1,000,000` seconds from sample zero. First use exactly the final 100,000
CH0 samples preceding `n_gate` to fit C1 and C2 jointly. Use the five-column
design `X_joint=[cos(phi_index),-sin(phi_index),cos(2*phi_index),
-sin(2*phi_index),1]`, solve one unweighted normal equation by Cholesky, and
require rank 5 and `cond2(X_joint^T X_joint)<=1e8`. Let
`beta=[a_1,b_1,a_2,b_2,c]`, `phi_1=atan2(b_1,a_1)`, and
`phi_2=atan2(b_2,a_2)`. Form the two half-frequency candidates
`wrap(phi_2/2)` and `wrap(phi_2/2+pi)`. Use `phi_1` to select the unique
candidate minimizing `abs(wrap(phi_1-gauge-delta_command))`.
This resolves the unavoidable half-angle branch without filenames or arm
labels. Require the fitted C2/C1 amplitude ratio to lie in `[0.23,0.27]`.
The record-local common gauge is then:

```text
phi_index[n] = 2*pi*f_ref*t[n]
r[n] = exp(i*phi_index[n])
z_gauge[j] = z_index[j] * exp(-i*gauge)
```

CH0 remains a monitor-only copy in the source domain even for zero-drive and
resonator-removed controls. It never enters the carrier path downstream of the
isolation barrier. All joint-fit sums are ascending-index binary64 sums.
Define `phi_drive=phi_1` and
`e_drive=wrap(phi_drive-gauge-delta_command)`; both tone-amplitude squares must
be finite and positive.

Drive-fit covariance is the Section 6 Newey-West sandwich over the single
five-column joint design and its joint residuals, with lag 7 and no
finite-sample multiplier. Define
`g_1=[-b_1/(a_1^2+b_1^2),a_1/(a_1^2+b_1^2),0,0,0]` and
`g_2=[0,0,-b_2/(a_2^2+b_2^2),a_2/(a_2^2+b_2^2),0]`. The complete
gauge-relative fit gradient is `g_e=g_1-0.5*g_2`; therefore
`u_error_fit=sqrt(g_e^T Cov(beta) g_e)`. This explicitly includes the C1/C2
cross-covariance as well as the half-C2 gauge-fit variance. Record
`u_C1`, `u_C2`, `u_gauge=0.5*u_C2`, `Cov(phi_C1,phi_C2)`, and
`u_error_fit` in each arm's result. Set
`U95_drive=1.96*sqrt(u_error_fit^2+u_skew^2+u_drive_cal^2)`, where `u_skew` and
`u_drive_cal` are sealed one-standard-uncertainty terms carried as exact source
metadata fields `phase_skew_standard_uncertainty_rad` and
`phase_drive_cal_standard_uncertainty_rad`. Every arm requires
`abs(e_drive)+U95_drive<=0.010 rad`. The matched pair additionally requires
`abs(wrap(e_drive_pi-e_drive_0))+U95_drive_pi+U95_drive_0<=0.010 rad`.
Equality passes. Failure kills before CH1 metrics. The projection always uses
the index reference and then applies the one record-local gauge rotation to all
CH1 and control projections, so preparation error is measured before the
rotation rather than silently fitted away.

The timing-mismatch control delays source-off by exactly one quarter nominal
carrier cycle. Under an ideal steady locked response its CH0-gauge-relative phase is
expected to remain unchanged; the control measures termination/transient
sensitivity and is categorically barred from matched evidence. No expected
quarter-turn is asserted.

### 5.2 Weighted projection

Use only CH1 raw samples. No acquisition-time demodulator output is
authoritative.

Let `n_admit` be the first raw index at or after the conservatively rounded
`t_admit`. For a single arm, window `j` starts at `s_j=n_admit+256*j`. For the
matched pair, define `g_a=n_admit,a-n_gate,a`, `g=max(g_0,g_pi)`, and use
`s_a,j=n_gate,a+g+256*j`; this gives exactly equal elapsed sample offsets from
the observed gate and discards, rather than interpolates, any extra early
samples. The even-window center is `c_j=s_j+1023.5` samples.

All arithmetic is IEEE-754 binary64, round-to-nearest/ties-to-even, with
nonfinite input forbidden. For every 2,048-sample window:

```text
w[k] = 0.5 - 0.5*cos(2*pi*k/2047), k = 0..2047
W = diag(w[0], ..., w[2047])
X[k] = [cos(phi_index[s_j+k]), -sin(phi_index[s_j+k]), 1]
G = X^T W X; h = X^T W x
beta = CholeskySolve(G, h), with sums accumulated in ascending k order
I = beta[0]
Q = beta[1]
z = I + iQ
A = abs(z)
theta = atan2(Q, I)
```

`G` must have Cholesky rank 3 and two-norm condition number `<=1e8`; equality
passes. The committed analysis implementation and binary64 math-library hash
must pass the sealed conformance vectors before execution. Every sample in a
window must satisfy `t >= t_admit`. No padding, interpolation, extrapolation,
causal state, forward-backward filtering, or pre-source sample may enter an
admissible estimate.

### 5.3 Phase and edge conventions

```text
wrap(x) = ((x + pi) mod 2*pi) - pi, with result in [-pi, pi)
exact +pi maps to -pi
phase is undefined and the window rejects when A < 10*sigma_A
unwrap chooses the unique 2*pi multiple giving the next difference in [-pi, pi)
an exact -pi difference remains -pi
NaN, infinity, failed rank/condition gate, missing samples, clipping, or nonmonotone time reject
```

### 5.4 Noise and uncertainty

The zero-drive, resonator-removed, and dummy-load controls use the identical
projection and its record-local CH0 gauge. Within each control, use only windows
`j=0,8,16,...` so noise windows do not overlap. For I and Q separately compute
the MAD below and take the maximum across the three controls and the ADC
quantization floor `q/sqrt(12)`:

```text
sigma = 1.4826 * median(abs(value - median(value)))
sigma_I = max(sigma_I_zero, sigma_I_removed, sigma_I_dummy, q_I/sqrt(12))
sigma_Q = max(sigma_Q_zero, sigma_Q_removed, sigma_Q_dummy, q_Q/sqrt(12))
sigma_A = sigma_I + sigma_Q
u_A_window = sigma_A
u_phase_window = sigma_A / A
U95_A_window = 1.96*u_A_window
U95_phase_window = 1.96*u_phase_window
```

Zero MAD in all three controls is invalid even when the quantization floor is
nonzero. Sort finite samples increasingly; for an even count the median is the
binary64 arithmetic mean of the two central values. The sum
`sigma_I+sigma_Q` is the conservative covariance-independent bound.
Calibration uncertainty includes corrected residual channel skew,
reference-fit covariance, five single-phase repeat preparations, ADC
quantization, and dummy-load feedthrough. The factor 1.96 is frozen for the
window-level Gaussian-equivalent MAD estimate.

Uncertainty responsibilities do not overlap. Per-arm CH0 drive phase and the
per-arm frequency/decay regressions use only the exact lag-7 Newey-West law.
The six matched-pair relation metrics use only a delete-one-block jackknife:
partition the paired I/Q windows into consecutive blocks of eight windows,
discard a final incomplete block, recompute both arm fits and the relation
metric after deleting each block, and set
`SE_JK=sqrt((B-1)/B*sum((m_-b-mean(m_-b))^2))`. Require `B>=8`. If
`C95_metric` is the prospectively sealed expanded calibration bound, define
`U95_metric=sqrt((1.96*SE_JK)^2+C95_metric^2)`. `C95_metric` is an independent,
prospectively sealed calibration field bound into the threshold hash; it may
not be derived from `T_metric`. No HAC term is added again.
For circular phase, jackknife the Cartesian mean unit vector, transform each
delete-block vector with `atan2`, choose the unique representative within pi of
the full-sample mean, and use the same formula; a zero resultant or an exact-pi
tie rejects. The confidence interval is the closed unwrapped-chart interval
`[mean-U95_phi,mean+U95_phi]` before mapping endpoints through `wrap`.

## 6. Frequency, decay, Q, and usable cycles

On the first consecutive admissible region satisfying `A/sigma_A >= 10`:

```text
theta_unwrapped(t) = unwrap(theta(t))
f_hat = f_ref + slope(theta_unwrapped versus t) / (2*pi)
log(A(t)) = log(A0) - t/tau_A
Q_hat = pi * f_hat * tau_A
```

Both slopes use unweighted ordinary least squares over the same region. Let
`u_j=t_j-mean(t)` and use the exact design row `x_j=[1,u_j]`; beta order is
`[intercept,slope]`. Solve `X^T X beta=X^T y` by binary64 Cholesky with
ascending-index sums. For log amplitude, define
`R^2=1-SSE/SST`, `SST=sum((log(A_j)-mean(log(A)))^2)`; `SST<=0` rejects.
Uncertainty is the exact Newey-West HAC covariance with Bartlett weights and
lag 7, reflecting the eightfold window overlap:

```text
S = Gamma_0 + sum(l=1..7, (1-l/8)*(Gamma_l + transpose(Gamma_l)))
Gamma_l = sum(j=l..M-1, e[j]*e[j-l]*x[j]*transpose(x[j-l]))
Cov(beta) = inverse(X^T X) * S * inverse(X^T X)
```

There is no `1/M`, degrees-of-freedom, HC, or other multiplier. All sums follow
ascending index order. Require finite positive-semidefinite covariance within
`1e-12*max(1,max(abs(Cov)))` eigenvalue tolerance; more-negative covariance
rejects and tolerated negative eigenvalues clamp to zero only for the reported
standard error. Set
`U95(f_hat)=1.96*sqrt(Cov_theta[1,1])/(2*pi)` and, for negative amplitude slope
`s_A`, `U95(tau_A)=1.96*sqrt(Cov_logA[1,1])/(s_A^2)` by the delta method.
Prospectively sealed reference-frequency and transfer-function expanded terms
are then combined once by root-sum-square. No robust outlier removal is allowed.
A clipped, missing, or rejected window terminates the consecutive region.

Usable cycles are:

```text
N_usable = floor(f_hat * (t_last_center - t_first_center))
```

P0 requires `N_usable >= 256`, amplitude SNR at least 10 in every counted
window, and 95% phase uncertainty no greater than 0.050 rad in every counted
window. It additionally requires `f_hat>0`, a negative log-amplitude slope,
`tau_A>0`, `Q_hat>0`, phase advance between hops strictly inside
`(-pi/2,pi/2)`, log-amplitude fit `R^2>=0.95`, fitted amplitude drop at least
0.25 natural-log unit, `U95(f_hat)/f_hat<=2e-6`, and
`U95(tau_A)/tau_A<=0.10`. Equality passes except at the strict unwrap gate.

The 256-cycle floor is fixed prospectively from the reference carrier's
datasheet-derived multi-thousand-cycle planning budget and supplies enough
independent phase evolution for frequency and decay estimation. It is not
chosen from primary data.

## 7. Antipodal and matching metrics

Use the maximal common prefix of the time-aligned matched-pair grid defined in
Section 5.2. It must contain at least 256 cycles. Pairing is by equal `j`; no
interpolation, frequency retiming, phase rotation, or envelope rescaling is
allowed.

Exact helper conventions are:

```text
norm2(v) = sqrt(sum_k abs(v[k])^2), summing in ascending k
rms(v) = norm2(v)/sqrt(len(v))
circular_mean(alpha) = atan2(sum sin(alpha), sum cos(alpha))
```

The circular mean rejects when both sums are zero. All norm and frequency/tau
denominators must be finite and strictly positive; every fitted frequency,
decay constant, and Q must satisfy the positive gates above.

```text
complex negation:
  epsilon_neg =
    norm2(z_pi + z_0) /
    (0.5 * (norm2(z_pi) + norm2(z_0)))

amplitude envelope:
  epsilon_A =
    norm2(abs(z_pi) - abs(z_0)) /
    (0.5 * (norm2(abs(z_pi)) + norm2(abs(z_0))))

frequency:
  epsilon_f = abs(f_pi - f_0) / (0.5 * (f_pi + f_0))

decay:
  epsilon_tau = abs(tau_pi - tau_0) / (0.5 * (tau_pi + tau_0))

half-turn phase:
  epsilon_phi =
    abs(circular_mean(wrap(theta_pi - theta_0 - pi)))

feedthrough:
  epsilon_feed =
    rms(z_dummy_or_removed) /
    min(rms(z_0), rms(z_pi))
```

All three control traces are projected on the same post-event relative-start
grid as the two relation arms. The analyzer intersects the exact retained arm
indices and all control indices; positional pairing without equal indices is
forbidden. `epsilon_feed` is the point-estimate RMS ratio above. For its
expanded uncertainty, each delete-one-block replicate removes the same eight
grid positions from both arms and all controls before recomputing the control
RMS and arm denominator. `U95_feed` is the resulting jackknife term combined
once with independently sealed `C95_feedthrough`. Acceptance uses the same law
as every relation metric: `epsilon_feed + U95_feed <= T_feed`.

If either denominator is zero, nonfinite, or below the admitted SNR floor, the
metric rejects rather than returning zero.

## 8. Prospective threshold-freeze law

Thresholds are calibration-normalized and must be sealed before the primary
0/pi acquisition. Five answer-independent calibration ringdowns all use the
single phase `delta_cal=pi/3`. For their ten unordered trace pairs, evaluate
the same amplitude, frequency, and decay metrics; replace complex sum by
complex difference and replace the half-turn residual by a zero-turn residual.
For each metric let `u_metric` be the maximum of all ten null residuals, all
ten jackknife `U95` values, and the sealed specification-derived calibration
bound. This is a conservative calibration envelope, not a fitted primary-data
quantity:

| Metric | Frozen threshold | Hard pre-execution cap |
|---|---|---:|
| Complex negation | `T_neg = 5*u_neg` | `T_neg <= 0.100` |
| Envelope mismatch | `T_A = 5*u_A` | `T_A <= 0.050` |
| Relative frequency mismatch | `T_f = 5*u_f` | `T_f <= 5e-6` |
| Relative decay mismatch | `T_tau = 5*u_tau` | `T_tau <= 0.050` |
| Half-turn phase error | `T_phi = 5*u_phi` | `T_phi <= 0.050 rad` |
| Source feedthrough | `T_feed = max U95 zero/removed/dummy ratio` | `T_feed <= 0.100` |

The primary passes a metric only when `epsilon <= T`. If calibration produces
any threshold above its cap, physical execution is not authorized under this
packet. Thresholds may never be widened after primary data are observed.

The phase relation additionally requires both equivalent tests under the
frozen common gauge: the 95% confidence set for
`circular_mean(wrap(theta_pi - theta_0))` contains the antipode (`-pi`, with
`+pi` identified by the declared wrap convention) and excludes 0 and
plus/minus pi/2; the 95% confidence interval for the residual
`circular_mean(wrap(theta_pi - theta_0 - pi))` contains 0.

`derived/relation_metrics.json` records all six epsilon values, all six U95
values, and the four confidence-set booleans. Its PASS law uses conservative
upper bounds `epsilon+U95<=T` for all six metrics. The two per-arm
`metrics.json` records must each have `quality_gate_pass=true`. Equality passes.

## 9. Calibration freeze

Before any primary arm, a later authorized executor must seal:

1. final part and instrument datasheet hashes;
2. wiring/topology photographs and continuity records;
3. BVD resonance sweep, linear drive region, and detector transfer function;
4. five zero-drive, five dummy-load, and exactly five `delta_cal=pi/3`
   single-phase repeat ringdowns; no calibration 0/pi pair;
5. `f_ref`, source amplitude, source phase, channel gains, channel skew, and
   switch latency;
6. noise, uncertainty, feedthrough, and the six metric thresholds;
7. exact analysis source hash and environment lock;
8. opaque arm assignment and acquisition order;
9. no-retry and invalidation ledger;
10. the committed analysis conformance vectors and passing implementation hash.

Calibration cannot contain, predict, or optimize the primary 0/pi answer.
There is one calibration campaign. A mechanically detected integrity failure
may invalidate the entire campaign once before threshold sealing, while
preserving all bytes and settings. A second calibration-campaign integrity
failure or any scientific/threshold failure stops; calibration cannot be
repeated to seek narrower noise, a preferred Q, or favorable thresholds.

## 10. Evidence packet and exact formats

Future execution must produce one immutable directory:

```text
p0_evidence_<run_id>/
  packet.json
  hardware_identity.json
  carrier_identity.json
  topology.json
  instruments.json
  calibration.json
  arm_assignment.sealed.json
  arm_assignment.reveal.json
  attempt_ledger.json
  prior_attempts/<attempt_id>/packet.json                    # replacement only
  prior_attempts/<attempt_id>/{hardware,carrier}_identity.json
  prior_attempts/<attempt_id>/{topology,instruments,calibration}.json
  prior_attempts/<attempt_id>/arm_assignment.sealed.json
  prior_attempts/<attempt_id>/contact_counts.json
  prior_attempts/<attempt_id>/raw_manifest.json
  prior_attempts/<attempt_id>/runs/<arm_id>/raw_native.bin
  prior_attempts/<attempt_id>/runs/<arm_id>/raw_descriptor.json
  prior_attempts/<attempt_id>/runs/<arm_id>/environment.csv
  prior_attempts/<attempt_id>/manifest.sha256.json
  runs/<arm_id>/raw_native.bin
  runs/<arm_id>/raw_descriptor.json
  runs/<arm_id>/raw.f64le
  runs/<arm_id>/reference.f64le
  runs/<arm_id>/switch_state.u8
  runs/<arm_id>/environment.csv
  derived/<arm_id>/iq.f64le
  derived/<arm_id>/iq_start_index.u64le
  derived/<arm_id>/metrics.json
  derived/relation_metrics.json
  control_evidence/<control_id>.json
  controls.json
  adjudication.json
  contact_counts.json
  raw_manifest.json
  manifest.sha256.json
```

Every JSON instance validates against `P0_EVIDENCE_SCHEMAS.json`. Canonical JSON
is UTF-8 without BOM, lexicographically sorted-key, two-space-indented,
LF/newline-terminated, and rejects duplicate keys, unknown fields, NaN,
Infinity, negative zero, nonminimal integers, and noncanonical serialization.
Finite noninteger quantities are decimal strings under the schema's grammar.
Timestamps are UTC RFC 3339 with six fractional digits; sample index and the
digitizer timebase are the within-record timing authority.

Manifest keys are normalized ASCII POSIX paths relative to the packet root.
Leading slash, colon/drive roots, backslash, empty segments, `.`/`..` segments,
duplicate normalized identities, symlinks, junctions, and any resolved path
outside the packet root reject. Before any parsing, the raw manifest contains
exactly each arm's `raw_native.bin`, `raw_descriptor.json`, `environment.csv`,
and the sealed assignment file. `switch_state.u8` is necessarily derived from
the already sealed CH2 bytes and therefore appears only in the final manifest.
Final manifest construction adds every packet
file except `manifest.sha256.json` itself. Both maps are sorted by path bytes.

Root construction is acyclic. `adjudication.json` binds the canonical external
raw-root receipt hash, the canonical controls ledger hash, and a canonical
metrics-bundle hash over all arm `metrics.json` records plus
`derived/relation_metrics.json`; it never contains the final manifest hash.
The final manifest then hashes `adjudication.json` and every other packet file
except itself. The external final-root receipt alone binds the final manifest
hash and the preceding raw-root-receipt hash. The controls ledger hash is the
SHA-256 of canonical JSON containing only `ordered_control_ids` and `outcomes`.
The metrics-bundle hash is the SHA-256 of canonical JSON
`{"files":{path:sha256,...}}`, sorted by path bytes.
For each of the exact 30 ordered control IDs, the outcome's `evidence_sha256`
is the hash of `control_evidence/<control_id>.json`. That record binds a unique
acquisition ID and its exact sequence ordinal. Each of the 27 physical controls
must submanifest its own native raw, descriptor, parsed raw, reference, switch,
environment, I/Q, I/Q-index, and metrics files. The three offline controls each
bind one distinct `offline/<control_id>.bin` artifact. Every submanifest path
and hash must resolve in the final packet. The offline records set
`execution_class=OFFLINE` and `physical_authority_consumed=false`; every other
record sets `PHYSICAL` and `true`. Missing, duplicate, reused, misordered,
aliased, or cross-control IDs/acquisitions reject.

`raw_native.bin` is the unmodified instrument export. Its descriptor binds
instrument model/serial/firmware, parser SHA-256, framing, interleave, header,
padding, native signedness, bit width, byte order, sample rate, channel order,
gain, offset, units, sample count, precommand count, and export version.
`raw.f64le` is the parser's canonical frame-major binary64 array of exact shape
`[3101000,4]`, channel order CH0-CH3, no header or padding. Raw native bytes are
write-once and SHA-256 bound before parsing or derivation.

Build-readiness revision B adds a stricter analyzer boundary without claiming a
proprietary parser: the DN2 SDK lossless-export mode must emit a headerless,
sample-major, little-endian signed-int16 `[3101000,4]` payload of exactly
24,808,000 bytes. That SDK export is the analyzer payload and its hashes/counts
must be identical; any additional proprietary container remains immutable and
separately hashed but is never parsed here. The adapter receipt binds its source
hash, SDK/driver identity, export/payload hashes and counts, channel order,
scaling and seven explicit no-transform assertions. `p0_scientific_analyzer.py`
consumes only that canonical signed-int16 payload. The legacy `raw.f64le`
projection remains a derived architecture-conformance artifact and is not
authoritative analyzer input.

`switch_state.u8` is the strict decode of CH2, one byte per raw sample with
bit 0 analog-gate state, bits 1-3 K1/K2/K3 witness states, and bits 4-7 fixed
zero. `reference.f64le` is little-endian binary64, frame-major `[3101000,2]`
in cosine/sine order and no header. Derived `iq.f64le` is little-endian
binary64 `[M,2]` in I,Q order and no header; `iq_start_index.u64le` contains
exactly M little-endian unsigned 64-bit window-start indices, with center
fixed at start plus 1023.5 samples.
Environment CSV columns are fixed:

```text
nearest_raw_sample_index,monotonic_ns,utc_timestamp,sensor_serial_hex,command_hex,temperature_ticks_hex,temperature_crc8_hex,rh_ticks_hex,rh_crc8_hex,temperature_C,rh_percent
```

The CSV is UTF-8 without BOM, LF-terminated, unquoted, comma-delimited, with
the exact eleven-column header above. Integers are minimal unsigned decimal;
timestamp grammar is fixed above; serial and raw-word fields are fixed-width
lowercase hexadecimal; CRC-8 uses polynomial `0x31`, initial `0xff`, and
big-endian word order. Finite decimals use
`-?(0|[1-9][0-9]*)(\.[0-9]+)?` and forbid negative zero. Rows are strictly
increasing by raw index, monotonic nanoseconds, and the exact 10 Hz cadence.

The pre-acquisition calibration record freezes its own creation time, final thresholds, analysis
source SHA-256, dependency lock SHA-256, schema hash, conformance-vector hash,
parser hash, instrument configuration hash, sealed-assignment hash, WORM medium
identity, and Ed25519 public-key bytes plus their SHA-256. It cannot contain
future raw hashes. Before any acquisition, the independent custodian signs and
closes a genesis calibration-root receipt on WORM; that receipt binds the exact
calibration bytes, including the sealed assignment hash. Only then may a later
authorized acquisition begin. The post-acquisition raw manifest binds every
raw byte and its signed raw-root receipt chains from the calibration receipt.
Every rerun creates a new packet and may not overwrite an earlier byte.

The dependency lock binds the full Python version and implementation, Python
executable SHA-256, platform and architecture, byte order, NumPy version,
NumPy distribution `RECORD` SHA-256, NumPy core-binary SHA-256, complete NumPy
build/BLAS/LAPACK/SIMD configuration, and the frozen single-thread environment
for OpenMP, OpenBLAS, MKL, NumExpr and vecLib. Byte-identical verification is
valid only under that exact runtime identity.

Immutability has three independently signed receipt stages. Before acquisition,
the witness stores `<run_id>.calibration_root.json`. Before any parsing, it
stores `<run_id>.raw_root.json`, containing the SHA-256 of a canonical
manifest over native raw, descriptors, environment, and sealed assignment
ciphertext plus the calibration-root-receipt hash. The independent witness uses
its calibrated UTC clock to observe the acquisition-enable edge and final
native-byte closure, then includes `acquisition_started_utc` and
`acquisition_completed_utc` in the signed raw-root receipt. The attempt ledger
must reproduce those exact signed times, and every raw descriptor interval must
fall within its attempt interval. After derivation,
`manifest.sha256.json` hashes every
packet file except itself; the witness stores `<run_id>.final_root.json`
containing the manifest SHA-256 and prior raw-root-receipt hash.

The witness class is an independent custodian using one fresh offline
write-once optical WORM session per run, closed after each receipt. Before
acquisition, calibration seals the medium identity and an Ed25519 public key
plus its SHA-256. The calibration receipt is sequence 1 with 64 zeroes as its
previous hash. The selected raw receipt is sequence 2 when no replacement was
used or sequence 3 after exactly one integrity-invalid predecessor. The final
receipt is respectively sequence 3 or 4 and binds the SHA-256 of the complete
selected raw receipt, including its signature. For every attempt the exact
strict chronology is `calibration bytes < calibration receipt < signed
acquisition start < signed acquisition completion < corresponding raw receipt`;
the packet then requires `selected raw receipt < assignment reveal < final
receipt`.
The signed
message is the exact canonical UTF-8 receipt with only `signature_base64`
omitted. Verification is strict Ed25519 verification with the sealed public
key; the 64-byte signature is canonical base64. The WORM session is outside
the packet and inaccessible to acquisition/analysis roles. Missing, mutable,
unverifiable, reordered, or contradictory receipts kill adjudication.

The committed `p0_packet_validator.py` is a structural preview gate only. It
checks schema dispatch, canonical bytes and safe paths, manifest/receipt/AEAD
chains, exact control/run identities, attempt-ledger structure, metrics-bundle
binding, sign/domain coherence, and structural adjudication consistency. It
always reports `scientific_authority=false` and cannot emit P0A-P0C. It does not
parse the instrument-native export, verify the full binary payload shapes and
CSV cadence, or recompute WLS, environment, regression, uncertainty, control,
and relation metrics from raw bytes. A future deterministic raw-derived
analyzer, malformed/truncated-payload adversaries, and independent review are
hard prerequisites inside any later execution authority. Structural fixture
success is not physical evidence and cannot authorize an acquisition or claim.

## 11. Retry and no-answer law

Primary acquisition is one prospectively ordered matched pair plus the frozen
control matrix. Scientific failure is not retryable within that authority.

One complete replacement sequence is permitted only when an automated integrity-only gate
detects a failure before arm unsealing or any phase-bearing content is exposed:
wrong byte count/hash, range-status clipping flag, failed CH2 witness,
instrument health fault, or environment outside a frozen limit. That gate may
read descriptors, range flags, CH2, environment, and hashes; it may not decode
CH0/CH1 waveforms, phase commands, I/Q, metrics, filenames, or assignment
ciphertext. `attempt_ledger.json` binds the ordered raw-root receipt hashes,
unique attempt IDs, categorical integrity reason, unopened-phase flag, the
single selected attempt, and the SHA-256 of the invalid predecessor's complete
acquisition-packet manifest. That preserved subtree contains the invalid
packet header, frozen identities and calibration, sealed assignment, contact
counts, raw manifest, and all 27 native raw/descriptor/environment triples.
Its inner manifest covers exactly that subtree and the prior signed raw-root
receipt binds its inner raw manifest. The structural preview validator verifies
all of those bytes; a receipt without the preserved acquisition packet is
rejected. The contact ledger must report exactly 27 physical acquisitions per
attempt, including exactly 27 in the preserved predecessor's own contact-count
record. The predecessor and selected bytes for hardware identity, carrier
identity, topology, instrument configuration, calibration, and sealed
assignment must be byte-identical. Any configuration or assignment drift kills
replacement. A second integrity failure stops the run. No threshold, filter,
window, frequency, phase,
amplitude, guard, or arm order may change.

Before acquisition an independent custodian forms canonical plaintext
`{"A":"0","B":"pi","salt_base64":"<32 bytes>"}` or the exact complementary
mapping with A/B reversed, then sets
`commitment_sha256=SHA256(canonical_plaintext_bytes)`. The custodian encrypts
those bytes using RFC 8439 ChaCha20-Poly1305 with a fresh 32-byte key and
12-byte nonce. AAD is canonical UTF-8
`{"record_type":"arm_assignment_aad","run_id":"<run_id>","schema_version":"p0-evidence-v1"}`;
`ciphertext_base64` is ciphertext followed by the 16-byte tag. The sealed file
binds AAD, key, and plaintext hashes plus nonce and cipher profile.

Only opaque arm IDs and precompiled command packets reach the operator or
controller. The custodian keeps the key offline and releases
`arm_assignment.reveal.json` only after raw-root receipt and integrity/replacement
eligibility close. Release exposes the key and salt; strict AEAD verification,
AAD hash, key hash, plaintext commitment, complementary mapping, custodian
identity, and release time must all close. Any mismatch kills the packet.
`mapping_sha256` is the SHA-256 of canonical UTF-8
`{"A":"0_or_pi","B":"pi_or_0"}` with the revealed complementary values;
`commitment_sha256` remains the hash of the full plaintext including salt.

Expected phase relation, expected metric, preferred arm, anticipated Q, and
expected adjudication may not enter acquisition, filenames, metadata,
preprocessing, or control flow.

## 12. Research dependency and unresolved device/circuit simulation

The canonical repository-safe source dependency is
`research/P0_research_bundle_2026-07-18`, imported from commit
`cb53976612cbe83bec82df826a9889418f7e0b89`. Its manifest contains exactly 35
records. Candidate-bound metadata records 11 private locally hash-verified
captures, 6 URL-plus-legacy-hash records without local bytes, and 18 manual
captures. PDFs, HTML, vendor models, receipts, and generated archives remain
ignored and uncommitted. Private refresh uses the repository virtual
environment to run `scripts/download_sources.py --all`,
`scripts/verify_downloads.py`, and `scripts/build_custody_snapshot.py` from the
bundle directory.

The existing synthetic reference remains a valid signal-level analyzer test:
it directly constructs deterministic ringdown waveforms and checks the actual
analyzer. It is not a first-principles prediction of the complete proposed
circuit. The separate unresolved generative layer is:

```text
source
  -> 100 kOhm limiter
  -> ADG1419 vendor model
  -> relay state, bounce, release, leakage, and parasitic model
  -> FC-135 Butterworth-Van Dyke motional R-L-C in parallel with C0
  -> OPA810 vendor model
  -> digitizer differential/common-mode loading
  -> cable, PCB, enclosure, resistor, amplifier, and ADC effects
  -> canonical four-channel raw payload
  -> existing unchanged scientific analyzer
```

The model must sweep motional R/L/C, shunt capacitance, loaded Q/resonance,
OPA810 input capacitance/leakage, switch off-capacitance/charge injection/
leakage, relay timing, cable/board capacitance, source-monitor feedthrough
including the bounded C2 path, digitizer differential/common-mode admittance,
clock/phase-reference error, amplifier/resistor noise, ADC quantization,
environmental perturbation, and matched empty/1 pF dummy controls. It must map
the parameter region that survives the frozen analyzer; inserting one favorable
ringdown cannot establish that the complete circuit generates it.

## 13. Current boundary

This plan is non-executing. It does not authorize powering a source, digitizer,
switch, relay, detector, transducer, or resonator. It generates no physical
data and establishes no physical claim.
