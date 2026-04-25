# Report: Next Steps That Are Not Toy Tests

## Purpose

This report replaces the weak direction of "small cluster purity" style probing
with a machine-learning research program that could actually matter.

The question is not:

- can the formula separate obviously pure from obviously mixed text groups?

The real questions are:

- does the formula track representation quality in real models?
- does it predict generalization better than simpler geometry metrics?
- does it help choose layers or checkpoints?
- does it improve training when used as an intervention?
- does it connect to accepted ML quantities rather than only to internal lab language?

If those answers are no, the ML version of the thesis fails.

## Hard Reset

The following directions are explicitly deprioritized:

1. tiny-sample cluster-purity probes as primary evidence;
2. post-hoc narrative rescue when a simple baseline wins;
3. any experiment where labels are implicitly defined by the formula itself;
4. any experiment that cannot beat `E`, margin, alignment/uniformity, effective
   rank, or other accepted baselines.

Small smoke tests can still be used only for runtime validation. They do not
count as evidence for the thesis.

## The Non-Toy Program

The serious program has four tracks.

### Track 1: Layer Selection on Real Benchmarks

#### Question

Can the formula identify the best hidden layer of a pretrained model for a real
downstream task better than simpler baselines?

#### Why this matters

This is a strong unsupervised usefulness test. If the formula captures good
representation geometry, it should tell us which layer is best before we run the
full downstream evaluation.

#### Data

- text: STS-B, SNLI, MNLI, SST-2, AG News
- encoders: BERT-base, RoBERTa-base, MPNet, MiniLM

#### Protocol

1. extract hidden states for every layer;
2. pool them in a fixed way;
3. compute:
   - `E`
   - `grad_S`
   - `R_simple`
   - `R_full`
   - effective rank
   - isotropy
   - alignment/uniformity if defined
4. independently evaluate each layer on the actual downstream task;
5. compare whether the metric ranking predicts the true layer ranking.

#### Success criterion

The formula must predict the best or near-best layer more often than the simpler
baselines across models and tasks.

#### Failure criterion

If `E`, effective rank, isotropy, or alignment/uniformity consistently predict
the better layer and the formula does not, the formula is not useful here.

### Track 2: Checkpoint Selection During Real Training

#### Question

Can the formula select better checkpoints than loss-based or validation-based
heuristics during fine-tuning?

#### Why this matters

This is the first serious causal-adjacent test. A metric that only decorates
finished representations is weak. A metric that helps choose a better checkpoint
has operational value.

#### Data and models

- task: SST-2, AG News, maybe SNLI if runtime allows
- models: DistilBERT or BERT-base for practicality on local hardware

#### Protocol

1. fine-tune for a fixed number of steps;
2. save checkpoints on a fixed schedule;
3. for each checkpoint compute:
   - validation loss
   - validation accuracy
   - formula metrics on hidden representations
   - baseline geometry metrics
4. compare three checkpoint selectors:
   - lowest validation loss
   - highest baseline metric
   - highest formula metric
5. evaluate final chosen checkpoint on held-out test data.

#### Success criterion

Formula-based checkpoint selection must beat or tie the best simpler selector on
multiple runs and seeds.

#### Failure criterion

If formula-based selection loses to loss or simpler geometry metrics, the
formula is not useful as a checkpoint criterion.

### Track 3: Supervised Geometry, Not Raw Homogeneity

#### Question

Does the formula become useful when reformulated around within-class coherence
and between-class separation?

#### Why this matters

Raw mean similarity is too close to the task of "obvious semantic homogeneity."
Real supervised ML depends on class structure, not just local sameness.

#### Required shift

A serious supervised variant should not be:

```text
R = score(one unlabeled cluster)
```

It should be something like:

```text
R_sup = f(within_class_agreement, within_class_dispersion, between_class_overlap)
```

Possible variants to test:

1. `R_within = E_within / grad_S_within`
2. `R_fisher = (E_within / grad_S_within) / max(E_between, eps)`
3. scatter-ratio form using within-class covariance and class-centroid spread

#### Data

- AG News
- SST-2
- CIFAR-10 embeddings if vision is added later

#### Success criterion

The supervised formula variant must beat `E`, margin-like baselines, and
within/between scatter ratios in predicting classification quality.

#### Failure criterion

If the useful version is just a standard Fisher-style scatter ratio with new
notation, then the formula did not add new ML substance.

### Track 4: Intervention, Not Just Correlation

#### Question

Can the formula improve training when used directly?

#### Why this matters

This is the strongest test that can realistically be run locally.

#### Intervention options

1. **regularizer**
   - add a formula-derived term to the loss on hidden representations
2. **batch filter**
   - deprioritize or reject low-quality batches
3. **curriculum**
   - schedule examples according to formula score
4. **checkpoint gate**
   - terminate or retain training based on formula dynamics

#### Success criterion

The intervention must improve held-out task metrics, robustness, or calibration
relative to a no-formula baseline and to simple alternative interventions.

#### Failure criterion

If optimization for the formula hurts task performance or does nothing beyond
simpler metrics, the formula is not causally useful.

## What the Formula Must Beat

Any serious ML claim must compare against:

1. `E`
2. `grad_S`
3. `1 / grad_S`
4. effective rank / participation ratio
5. isotropy / anisotropy scores
6. alignment / uniformity where applicable
7. margin or centroid separation for supervised tasks
8. validation loss for checkpoint selection

If the formula cannot outperform these, the responsible answer is that the ML
mapping is weak or redundant.

## What Is Under Suspicion Right Now

Current evidence makes these the primary suspects:

1. **`sigma`**
   - may be too noisy
   - may be the wrong normalization
   - may be useful only after stronger estimation
2. **`Df`**
   - power-law fitting may be too brittle
   - may be the wrong spectral statistic
3. **the multiplicative term `sigma^Df`**
   - may amplify noise rather than structure

This means the next serious work must include ablations:

- `E`
- `E / grad_S`
- `E / grad_S * sigma`
- `E / grad_S * Df`
- `E / grad_S * sigma^Df`
- alternatives using effective rank or spectral entropy instead of `Df`

These ablations are not toy tests if they are embedded inside Tracks 1 to 4 and
evaluated on real downstream outcomes.

## Theoretical Angles That Actually Matter

The formula should be tested against accepted ML geometry, not against vague
cross-domain rhetoric.

### 1. Alignment / Uniformity

The most plausible bridge is:

- `E` behaves like alignment
- `grad_S` and spectral terms may relate to non-collapse / uniformity

This is more defensible than claiming the formula already captures "all
learning."

### 2. Neural Collapse

Late supervised training induces strong geometric regularities:

- low within-class scatter
- structured class means
- stable simplex-like organization

If the formula maps to something real, it may show up here.

### 3. Fisher-Style Scatter Ratios

A supervised version of the thesis may reduce to a within-class versus
between-class geometry law. If so, that is still valuable, but it must be
demonstrated honestly rather than smuggled in as novelty.

### 4. Spectral Regularization

If the spectral terms are useful, they may connect more naturally to:

- effective rank
- covariance conditioning
- anisotropy control
- representation collapse avoidance

That is a practical ML bridge.

## Recommended Execution Order

### Phase A

Build the serious evaluation pipeline before inventing more theory.

1. layerwise benchmark runner
2. checkpoint-tracking runner
3. supervised geometry runner

### Phase B

Run ablations inside those runners.

1. compare `R_simple`, `R_full`, and simpler baselines
2. isolate whether `sigma` and `Df` help or hurt

### Phase C

Only if the formula still has signal:

1. run an intervention test
2. attempt theoretical reduction to known ML quantities

## Minimal Deliverables

The next implementation pass should produce:

1. `EXPERIMENT_02_LAYER_SELECTION.md`
2. `EXPERIMENT_03_CHECKPOINT_SELECTION.md`
3. `EXPERIMENT_04_SUPERVISED_GEOMETRY.md`
4. runnable code for each
5. explicit pass/fail criteria
6. honest verdict files

## Bottom Line

The correct next step is not "more cluster tests."

The correct next step is:

- real models
- real checkpoints
- real layers
- real downstream metrics
- real baselines
- real interventions

If the formula survives that, it becomes worth taking seriously in ML.
If it does not, the ML version of the thesis should be narrowed or rejected.
