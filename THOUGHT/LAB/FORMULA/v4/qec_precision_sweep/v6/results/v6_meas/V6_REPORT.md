# QEC Precision Sweep v6 -- Fractal Scaling Test

Run id: `v6_meas`
Noise model: `meas`
Source data: `qec_v2_meas`
UTC time: `2026-05-13T11:15:14.372529+00:00`

## Design

For each p, measure per-step sigma at each adjacent distance pair:
```
sigma_{d->d+2} = exp((ln(R_{d+2}) - ln(R_d)) / 2)
```
If the formula describes fractal depth, ln(sigma) should decay
systematically with distance (power law or linear).

## Per-Step Sigma Values

| p | d_jump | delta_logR | sigma_step | ln(sigma) |
|---:|---:|---:|---:|---:|
| 0.0005 | 3->5 | +1.7777 | 2.4323 | +0.8888 |
| 0.0005 | 5->7 | +0.0000 | 1.0000 | +0.0000 |
| 0.0005 | 7->9 | +0.0000 | 1.0000 | +0.0000 |
| 0.0010 | 3->5 | +2.3270 | 3.2011 | +1.1635 |
| 0.0010 | 5->7 | +1.0986 | 1.7321 | +0.5493 |
| 0.0010 | 7->9 | +0.0000 | 1.0000 | +0.0000 |
| 0.0020 | 3->5 | +2.3795 | 3.2862 | +1.1897 |
| 0.0020 | 5->7 | +2.0716 | 2.8173 | +1.0358 |
| 0.0020 | 7->9 | +0.0000 | 1.0000 | +0.0000 |
| 0.0040 | 3->5 | +1.2584 | 1.8761 | +0.6292 |
| 0.0040 | 5->7 | +1.3737 | 1.9874 | +0.6868 |
| 0.0040 | 7->9 | +2.0215 | 2.7477 | +1.0108 |
| 0.0060 | 3->5 | +0.8354 | 1.5185 | +0.4177 |
| 0.0060 | 5->7 | +1.2033 | 1.8251 | +0.6016 |
| 0.0060 | 7->9 | +0.9229 | 1.5864 | +0.4615 |
| 0.0080 | 3->5 | +0.5909 | 1.3437 | +0.2954 |
| 0.0080 | 5->7 | +0.6639 | 1.3937 | +0.3319 |
| 0.0080 | 7->9 | +0.6361 | 1.3744 | +0.3181 |
| 0.0100 | 3->5 | +0.4865 | 1.2754 | +0.2433 |
| 0.0100 | 5->7 | +0.4772 | 1.2695 | +0.2386 |
| 0.0100 | 7->9 | +0.4788 | 1.2705 | +0.2394 |
| 0.0200 | 3->5 | -0.0921 | 0.9550 | -0.0461 |
| 0.0200 | 5->7 | -0.1000 | 0.9513 | -0.0500 |
| 0.0200 | 7->9 | -0.1125 | 0.9453 | -0.0562 |
| 0.0400 | 3->5 | -0.2409 | 0.8865 | -0.1205 |
| 0.0400 | 5->7 | -0.1161 | 0.9436 | -0.0581 |
| 0.0400 | 7->9 | -0.0473 | 0.9766 | -0.0236 |

## Fractal Pattern Analysis

### Power-law decay: ln(sigma) ∝ d^k
- Form: `ln(sigma) = -0.4299 * ln(d) + 1.1219`
- R2: `0.0916`
- Slope (exponent): `-0.4299`

### Linear decay: ln(sigma) ∝ d
- Slope: `-0.0753`
- R2: `0.0927`

### Sigma consistency across p values
- Mean ln(sigma): `0.3684`
- Std ln(sigma): `0.4040`
- Coefficient of variation: `1.0965`

**Weak or no fractal pattern: ln(sigma) vs distance is not systematic.**

## Sigma Sign-Flip (Threshold Crossings)

Sigma > 1 below threshold, sigma < 1 above. Where does it cross?

No threshold crossings detected.

## Predictive Test: sigma_{7->9} from sigma_{3->5}

Using power-law fit across all p values, predict sigma at d=7->9:
- Predicted sigma_79: `1.2559`

| p | actual sigma_79 | predicted | error |
|---:|---:|---:|---:|
| 0.0005 | 1.0000 | 1.2559 | 0.2559 |
| 0.0010 | 1.0000 | 1.2559 | 0.2559 |
| 0.0020 | 1.0000 | 1.2559 | 0.2559 |
| 0.0040 | 2.7477 | 1.2559 | 1.4918 |
| 0.0060 | 1.5864 | 1.2559 | 0.3305 |
| 0.0080 | 1.3744 | 1.2559 | 0.1186 |
| 0.0100 | 1.2705 | 1.2559 | 0.0146 |
| 0.0200 | 0.9453 | 1.2559 | 0.3106 |
| 0.0400 | 0.9766 | 1.2559 | 0.2792 |
