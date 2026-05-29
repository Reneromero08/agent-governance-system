# .holo Formalism: Dimensional Shannon Compression

**Status:** Formal research note v0.1  
**Scope:** Mathematical foundation for `.holo` as a dimensional information
codec, distinct from symbolic recall, content addressing, CAT, CAS, or cassette
lookup.

## Abstract

`.holo` is a lossy dimensional codec for structured observations. Given an
observed signal distribution embedded in a high-dimensional ambient space,
`.holo` estimates the distribution's effective information dimension, stores a
low-dimensional coordinate representation over the dominant information axes,
and renders an approximation by lifting those coordinates back into observable
space.

The codec is "holographic" in the specific mathematical sense that the stored
object is not the full observation. It is a compact generative description:

```text
source observations -> information spectrum -> coordinates + basis -> render
```

The core thesis is:

> Apparent dimensionality is not information dimensionality. Compression is
> possible when the Shannon/eigen spectrum of the observation distribution is
> concentrated on a small number of active dimensions.

## 1. Non-Goals and Boundary

`.holo` is not:

- a hash pointer to known content
- symbolic recall
- a content-addressed storage scheme
- a cassette/CAT/CAS reference protocol
- a promise of exact reconstruction unless operated in a lossless residual mode

Those systems reduce transmission by assuming the receiver already has the
content. `.holo` reduces representation by estimating the active information
dimensions of the data itself.

## 2. Objects and Notation

Let:

| Symbol | Meaning |
|--------|---------|
| `Omega` | Source domain of raw objects |
| `x in Omega` | One raw object: image, audio, activation trace, etc. |
| `A_D` | Domain adapter for domain `D` |
| `X = A_D(x)` | Observation matrix in `R^{m x n}` |
| `m` | Number of samples/patches/windows/tokens |
| `n` | Ambient observed dimension per sample |
| `mu in R^n` | Empirical mean of the observations |
| `X_c = X - 1 mu^T` | Centered observation matrix |
| `C = (1/(m-1)) X_c^T X_c` | Empirical covariance |
| `lambda_1 >= ... >= lambda_r > 0` | Nonzero covariance eigenvalues |
| `r = rank(X_c)` | Empirical rank |
| `U_k` | Top `k` orthonormal eigenvectors, stored row-wise as `basis` |
| `Z_k = X_c U_k` | Low-dimensional coordinates |
| `X_hat_k = Z_k U_k^T + 1 mu^T` | Rendered reconstruction |
| `R_D` | Domain renderer from observations back to raw domain |

The current image adapter uses flattened patches:

```text
A_image(image) = matrix of 8x8x3 patches
R_image(patches) = reassembled pixel image
```

Other domains use different adapters but the same core math.

## 3. Information Spectrum

Define the normalized spectral distribution:

```text
p_i = lambda_i / sum_j lambda_j
```

`p` is a probability distribution over active covariance modes. It measures how
the source variance/information energy is distributed across dimensions.

The cumulative information retained by the top `k` dimensions is:

```text
V(k) = sum_{i=1..k} p_i
```

The discarded tail energy is:

```text
T(k) = 1 - V(k) = sum_{i=k+1..r} p_i
```

`V(k)` is the first operational quality control for `.holo`: if top modes
capture most of the spectral mass, a low-dimensional render can approximate the
source.

## 4. Effective Information Dimensions

`.holo` tracks two effective dimensions because they answer different
questions.

### 4.1 Participation Dimension

```text
D_pr = (sum_i lambda_i)^2 / sum_i lambda_i^2
```

Equivalent form:

```text
D_pr = 1 / sum_i p_i^2
```

`D_pr` is the inverse collision probability of the spectral distribution. It is
the order-2 Renyi effective support size. It strongly rewards dominant modes.

Interpretation:

- `D_pr = 1`: all spectral mass is in one mode.
- `D_pr = r`: spectral mass is uniform across all active modes.
- Small `D_pr` means highly concentrated structure.

### 4.2 Shannon Dimension

```text
H_spectrum = -sum_i p_i log(p_i)
D_shannon = exp(H_spectrum)
```

`D_shannon` is the perplexity of the spectral distribution. It is the order-1
effective support size and is more sensitive to spectral tails than `D_pr`.

Interpretation:

- `D_shannon` estimates how many modes are needed to describe the spectral
  distribution under Shannon entropy.
- `D_shannon >= D_pr` for typical non-uniform spectra.
- Use it when tail detail matters.

### 4.3 Ambient Compression Opportunity

Define the dimensional compression opportunity:

```text
O_pr = n / D_pr
O_shannon = n / D_shannon
```

These are not file compression ratios. They are upper-level indicators of how
overcomplete the ambient representation is relative to its measured information
dimension.

Actual file compression also depends on:

- coordinate precision
- basis storage cost
- codebook size
- entropy coding
- patch/window geometry
- residual layers

## 5. The .holo Codec

A `.holo` codec is a tuple:

```text
H = (A_D, E, Q, S, L, R_D, M)
```

Where:

| Component | Meaning |
|-----------|---------|
| `A_D` | Domain adapter: raw object -> observation matrix |
| `E` | Spectrum estimator: observation matrix -> eigen spectrum |
| `Q` | Dimension selector: spectrum -> retained dimension `k` |
| `S` | Storage transform: observations -> basis + coordinates |
| `L` | Quantizer/entropy coder for coordinates and basis |
| `R_D` | Renderer: reconstructed observations -> raw approximation |
| `M` | Metadata/verifier: shape, dtype, policy, distortion metrics |

Encoding:

```text
X = A_D(x)
mu = mean(X)
X_c = X - mu
C = covariance(X_c)
(lambda_i, u_i) = eig(C)
k = Q(lambda)
Z_k = X_c U_k
payload = L(mu, U_k, Z_k, lambda_1..lambda_k, metadata)
```

Rendering:

```text
(mu, U_k, Z_k, metadata) = decode(payload)
X_hat_k = Z_k U_k^T + mu
x_hat = R_D(X_hat_k)
```

Progressive rendering:

```text
X_hat_j = Z_j U_j^T + mu, for 1 <= j <= k
```

This produces an "essence to detail" render path: lower `j` shows only the
dominant information modes; larger `j` restores finer structure.

## 6. Dimension Selection Policies

The selector `Q(lambda)` can be defined by policy.

### 6.1 Participation Policy

```text
k = ceil(D_pr)
```

Use for aggressive essence renders and hypothesis testing.

### 6.2 Shannon Policy

```text
k = ceil(D_shannon)
```

Use when spectral tails contain perceptually meaningful detail.

### 6.3 Variance Policy

Given target `tau in (0, 1]`:

```text
k_tau = min { k : V(k) >= tau }
```

Use for quality-bounded compression. Common values: `0.90`, `0.95`, `0.99`.

### 6.4 Rate-Distortion Policy

Given a bit budget `B`, choose:

```text
k_B = argmin_k D(X, X_hat_k)
      subject to bits(mu, U_k, Z_k, metadata) <= B
```

Given a distortion budget `epsilon`, choose:

```text
k_epsilon = argmin_k bits(mu, U_k, Z_k, metadata)
            subject to D(X, X_hat_k) <= epsilon
```

This is the paper-grade target for future codec optimization.

### 6.5 Engineering Action Policy

The semiotic-light-cone material makes the key engineering point: the formula is
not decorative. The math must be the control law. For `.holo`, the control law
is an action over retained dimensions.

Let `B(k)` be the payload bits required to store `mu`, `U_k`, `Z_k`, and
metadata. Let `T(k)` be the spectral tail. Define:

```text
A(k) = alpha T(k) + beta B(k)
```

where:

- `alpha` is the cost of lost information,
- `beta` is the cost of storage/transmission,
- `T(k)` is the entropy gradient left uncrossed by the codec,
- `B(k)` is the engineering burden paid to retain dimensions.

Then the operational `.holo` dimension is:

```text
k* = argmin_k A(k)
```

Equivalently, retain dimension `i` while its marginal information per bit
exceeds the rate price:

```text
p_i / DeltaB_i > beta / alpha
```

This is the engineering correction to the naive "use Df as k" rule. `D_pr` and
`D_shannon` diagnose compressibility. They do not by themselves choose the file
format. The actual codec chooses `k` by the action that trades retained
information against payload cost.

## 7. Optimal Linear Projection Theorem

**Theorem 1: PCA optimality for linear `.holo` rendering.**

Let `X_c in R^{m x n}` be centered observations. Among all rank-`k`
orthogonal projections `P` onto a `k`-dimensional subspace, the projection onto
the span of the top `k` covariance eigenvectors minimizes squared
reconstruction error:

```text
U_k = argmin_U ||X_c - X_c U U^T||_F^2
      subject to U^T U = I_k
```

The minimum error is:

```text
||X_c - X_c U_k U_k^T||_F^2 = (m - 1) sum_{i=k+1..r} lambda_i
```

Normalized retained variance is:

```text
V(k) = 1 - ||X_c - X_hat_k||_F^2 / ||X_c||_F^2
     = sum_{i=1..k} lambda_i / sum_{i=1..r} lambda_i
```

**Proof sketch.** This is the Eckart-Young-Mirsky theorem applied to the SVD of
`X_c`. The right singular vectors of `X_c` are covariance eigenvectors. Truncating
to the top `k` singular values gives the unique optimal rank-`k` approximation
when singular values are distinct.

**Implication.** For any fixed `k` and squared-error metric, a linear `.holo`
basis cannot beat PCA on the same observation matrix. Improvements must come
from a better domain adapter, a nonlinear renderer, perceptual loss, better
quantization, or learned priors.

## 8. Quantized .holo Model

The pure projection model stores continuous coordinates. A real file stores
finite precision values.

Let:

```text
Z_q = quantize(Z_k)
U_q = quantize(U_k)
mu_q = quantize(mu)
```

Then:

```text
X_tilde = Z_q U_q^T + mu_q
```

Total distortion decomposes as:

```text
D_total <= D_projection(k) + D_quantization(Z, U, mu) + D_adapter
```

Where:

- `D_projection(k)` is the spectral tail loss.
- `D_quantization` is finite precision/codebook loss.
- `D_adapter` is loss caused by converting raw objects to observations and back
  through a domain adapter.

For vector quantized `.holo`, coordinates are replaced with codebook indices:

```text
c_i = nearest_codebook(Z_i)
Z_hat_i = codebook[c_i]
X_hat_i = Z_hat_i U_k^T + mu
```

Rate becomes:

```text
bits ~= bits(U_k) + bits(mu) + m log2(|C|) + bits(codebook) + metadata
```

This is why VQ can beat raw PCA-coordinate storage: it replaces many coordinate
vectors with compact archetype labels.

## 9. Holographic Principle in This Codec

The word "holographic" is justified by three operational properties:

1. **Stored form is not the observation.** The payload stores a generative
   projection state, not pixels/samples directly.
2. **Global basis, local coordinates.** A shared basis defines the information
   axes; local patches/windows store coordinates on those axes.
3. **Progressive render.** The same payload can render multiple observable
   states by varying retained dimensions `j <= k`.

This is not a metaphysical claim by itself. The testable claim is that many
natural signals have concentrated spectra, allowing compact basis-coordinate
storage to reconstruct useful observations.

## 10. Falsifiable Predictions

`.holo` makes the following empirical predictions.

### P1: Spectrum Predicts Compressibility

If `D_pr << n` and `D_shannon << n`, then low-dimensional projection should
retain high variance at small `k`.

Failure condition: A domain shows low effective dimension but high reconstruction
error at `k ~= D_shannon` under the declared distortion metric.

### P2: Spectral Tail Predicts Detail Loss

Perceptual/detail loss should track the discarded spectral tail `T(k)` within a
domain adapter.

Failure condition: Increasing `V(k)` does not improve distortion or perceptual
metrics monotonically on average.

### P3: Wrong Adapters Inflate Dimension

A poor observation adapter makes structured data look high-dimensional.

Failure condition: Multiple reasonable adapters yield the same high effective
dimension for a domain known to contain strong regularity.

### P4: VQ Helps When Coordinates Cluster

Vector quantization improves rate at similar distortion when coordinate vectors
cluster into repeated archetypes.

Failure condition: The coordinate distribution has no cluster structure and VQ
does not beat scalar/float coordinate storage.

### P5: Noise Raises Effective Dimension

Adding independent noise should flatten the spectrum, increasing `D_pr`,
`D_shannon`, and the required `k_tau`.

Failure condition: Controlled noise injection does not increase effective
dimension or distortion at fixed `k`.

### P6: The Action Chooses the Engineering Point

At fixed source spectrum, increasing the rate price `beta` should monotonically
lower or preserve the chosen `k*`. Decreasing `beta` should monotonically raise
or preserve `k*`.

Failure condition: The implementation's selected `k*` moves opposite the
rate-distortion action under deterministic spectra.

## 11. Success Criteria

For any `.holo` domain experiment, report:

| Metric | Required |
|--------|----------|
| `n` | Ambient observed dimension |
| `m` | Number of observations |
| `D_pr` | Participation dimension |
| `D_shannon` | Shannon dimension |
| `k_tau` | Dimensions for variance targets |
| `V(k)` | Retained variance curve |
| `D_projection` | Projection distortion |
| `D_quantization` | Quantization/VQ distortion if applicable |
| `file_rate` | Bits or bytes per source object |
| `baseline_rate` | JPEG/WebP/AVIF/etc. when relevant |
| `render_metric` | PSNR/SSIM/LPIPS/perplexity/domain metric |

Claims like "30x smaller" must state the baseline, quality metric, source data,
codec settings, and whether basis/codebook overhead is included.

## 12. Minimal Generic Interface

The implementation contract is:

```text
analyze_spectrum(X) -> eigenvalues, V(k), D_pr, D_shannon
choose_k(spectrum, policy) -> k
project(X, k) -> Z_k, U_k, mu
quantize(Z_k, U_k, mu) -> payload
render(payload, j <= k) -> X_hat_j
verify(X, X_hat_j) -> distortion report
```

Domain adapters add:

```text
observe(raw) -> X
assemble(X_hat) -> raw_hat
metric(raw, raw_hat) -> domain scores
```

## 13. Core Research Questions

1. Which domains have stable low information dimension under good adapters?
2. Which effective dimension, `D_pr` or `D_shannon`, best predicts useful `k`?
3. When does VQ/archetype coding beat raw coefficient storage?
4. Can learned bases outperform empirical PCA under perceptual metrics while
   respecting the same spectral constraints?
5. Can `.holo` video exploit shared temporal basis enough to outperform image
   frame compression?

## 14. Short Statement

`.holo` is a dimensional Shannon codec. It compresses observations by estimating
the effective information dimension of their distribution, storing coordinates
on the dominant information axes, and rendering observable data as a projection
from that reduced space.
