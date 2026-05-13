# QEC Precision Sweep v6 -- Fractal Scaling Test

Run id: `v6_depol`
Noise model: `depol`
Source data: `qec_v2_depol`
UTC time: `2026-05-13T11:15:10.936353+00:00`

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
| 0.0005 | 3->5 | +1.7602 | 2.4112 | +0.8801 |
| 0.0005 | 5->7 | +0.8047 | 1.4953 | +0.4024 |
| 0.0005 | 7->9 | +0.0000 | 1.0000 | +0.0000 |
| 0.0010 | 3->5 | +0.9505 | 1.6084 | +0.4752 |
| 0.0010 | 5->7 | +1.2825 | 1.8988 | +0.6412 |
| 0.0010 | 7->9 | +0.5493 | 1.3161 | +0.2747 |
| 0.0020 | 3->5 | +1.0703 | 1.7077 | +0.5351 |
| 0.0020 | 5->7 | +1.4096 | 2.0234 | +0.7048 |
| 0.0020 | 7->9 | +1.9321 | 2.6276 | +0.9661 |
| 0.0040 | 3->5 | +0.4280 | 1.2386 | +0.2140 |
| 0.0040 | 5->7 | +0.4630 | 1.2605 | +0.2315 |
| 0.0040 | 7->9 | +0.7105 | 1.4265 | +0.3553 |
| 0.0060 | 3->5 | +0.0518 | 1.0263 | +0.0259 |
| 0.0060 | 5->7 | +0.1966 | 1.1033 | +0.0983 |
| 0.0060 | 7->9 | +0.1281 | 1.0662 | +0.0641 |
| 0.0080 | 3->5 | -0.1854 | 0.9115 | -0.0927 |
| 0.0080 | 5->7 | -0.0922 | 0.9549 | -0.0461 |
| 0.0080 | 7->9 | -0.0498 | 0.9754 | -0.0249 |
| 0.0100 | 3->5 | -0.3836 | 0.8255 | -0.1918 |
| 0.0100 | 5->7 | -0.2453 | 0.8846 | -0.1227 |
| 0.0100 | 7->9 | -0.1424 | 0.9313 | -0.0712 |
| 0.0200 | 3->5 | -0.5608 | 0.7555 | -0.2804 |
| 0.0200 | 5->7 | -0.2673 | 0.8749 | -0.1337 |
| 0.0200 | 7->9 | -0.1330 | 0.9357 | -0.0665 |
| 0.0400 | 3->5 | -0.2652 | 0.8758 | -0.1326 |
| 0.0400 | 5->7 | -0.0110 | 0.9945 | -0.0055 |
| 0.0400 | 7->9 | -0.0029 | 0.9985 | -0.0015 |

## Fractal Pattern Analysis

### Power-law decay: ln(sigma) ∝ d^k
- Form: `ln(sigma) = 0.0155 * ln(d) + 0.1469`
- R2: `0.0002`
- Slope (exponent): `0.0155`

### Linear decay: ln(sigma) ∝ d
- Slope: `0.0018`
- R2: `0.0001`

### Sigma consistency across p values
- Mean ln(sigma): `0.1740`
- Std ln(sigma): `0.3319`
- Coefficient of variation: `1.907`

**Weak or no fractal pattern: ln(sigma) vs distance is not systematic.**

## Sigma Sign-Flip (Threshold Crossings)

Sigma > 1 below threshold, sigma < 1 above. Where does it cross?

No threshold crossings detected.

## Predictive Test: sigma_{7->9} from sigma_{3->5}

Using power-law fit across all p values, predict sigma at d=7->9:
- Predicted sigma_79: `1.1962`

| p | actual sigma_79 | predicted | error |
|---:|---:|---:|---:|
| 0.0005 | 1.0000 | 1.1962 | 0.1962 |
| 0.0010 | 1.3161 | 1.1962 | 0.1199 |
| 0.0020 | 2.6276 | 1.1962 | 1.4314 |
| 0.0040 | 1.4265 | 1.1962 | 0.2304 |
| 0.0060 | 1.0662 | 1.1962 | 0.1300 |
| 0.0080 | 0.9754 | 1.1962 | 0.2208 |
| 0.0100 | 0.9313 | 1.1962 | 0.2649 |
| 0.0200 | 0.9357 | 1.1962 | 0.2605 |
| 0.0400 | 0.9985 | 1.1962 | 0.1976 |
