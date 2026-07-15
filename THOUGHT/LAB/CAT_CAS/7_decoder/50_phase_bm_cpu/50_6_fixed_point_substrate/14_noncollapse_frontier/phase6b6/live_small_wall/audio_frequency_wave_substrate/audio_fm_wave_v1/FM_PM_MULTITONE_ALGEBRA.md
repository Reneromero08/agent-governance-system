# FM, PM, And Multitone Algebra

Status: `FROZEN_REFERENCE_ALGEBRA`

## FM

For bounded message `m[n]`, the reference freezes a left-Riemann integral with zero
initial value:

```text
I[0] = 0
I[n] = sum_(r=0)^(n-1) m[r] / 48000
phi[n] = 2*pi*8000*n/48000 + 2*pi*420*I[n]
s[n] = 0.90*cos(phi[n])
```

Recovery constructs the analytic signal, unwraps phase, differentiates with the
centered `numpy.gradient` interior convention, subtracts 8000 Hz, and divides by 420
Hz per message unit. The frozen metric is sample RMSE after the 4096-sample edge crop.

## PM

```text
s[n] = 0.90*cos(2*pi*8000*n/48000 + 1.15*m[n])
```

Recovery unwraps analytic phase, subtracts the carrier, removes the interior mean
phase, and divides by 1.15 rad per message unit. The message fixture is zero-mean and
bandlimited. The same edge crop and a separately frozen phase-domain metric apply.

## Analytic Geometry

Given analytic states `z_s = A_s exp(i phi_s)` and `z_q = A_q exp(i phi_q)`:

```text
z_s * conjugate(z_q) = A_s A_q exp(i(phi_s - phi_q))
z_s * z_q            = A_s A_q exp(i(phi_s + phi_q))
```

The reference checks both identities on the complex unit circle. It never infers a
phase sign from a label string.

## Multitone Geometry

```text
x[n] = Re(sum_k c_k exp(i 2*pi*f_k*n/48000))
```

The filter bank recovers `c_k`, not merely `|c_k|`. A declared null at 7111 Hz tests
off-bin leakage. Because all occupied test frequencies are integer-cycle bins, the
coefficient comparison has no window correction.

## Delay And Rotation

For a circular delay of `d` samples:

```text
c_k' = c_k exp(-i 2*pi*f_k*d/48000)
```

The fixture freezes `d = 37`. Fractional phase rotation is represented directly as
complex multiplication. This is a DSP operator, not a claim that a physical delay
line holds post-source state.

## Correlation And Matched Filtering

Unnormalized correlation is the complex inner product `a^H b`. Normalized correlation
divides by `||a|| ||b||`; it returns zero if either norm is zero. The matched-filter
target is a windowed 1-4 kHz chirp. Its energy-matched, magnitude-spectrum-matched sham
is exact time reversal. The target must exceed the sham normalized score by at least
0.80.

## Convolution

The engine implements full FFT convolution with output length `N+M-1`, zero extension
outside finite support, and no implicit centering. It is compared with direct numerical
convolution at sample RMSE <= `1e-13`.

## Controlled Nonlinear Mixing

The ordinary nonlinear law is:

```text
x = 0.30*cos(2*pi*7000*t) + 0.30*cos(2*pi*8200*t)
y = x + 0.8*x^2
```

It predeclares difference and sum products at 1200 Hz and 15200 Hz, each with complex
coefficient `0.072 + 0i`. These products are conventional polynomial intermodulation.
They are not evidence of catalytic borrowing or relational memory.

## Authoritative Numerical Freeze

All tolerances, comparator directions, edge exclusions, and fixture identities live in
`AUDIO_WAVE_REFERENCE_TESTS.json`; all observations live in
`AUDIO_WAVE_REFERENCE_RESULTS.json`. A prose approximation never overrides those JSON
objects.
