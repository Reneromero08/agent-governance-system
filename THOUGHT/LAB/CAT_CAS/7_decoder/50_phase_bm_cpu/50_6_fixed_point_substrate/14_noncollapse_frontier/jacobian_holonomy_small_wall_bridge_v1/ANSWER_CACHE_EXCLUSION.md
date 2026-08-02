# Bounded Replay and Answer-Cache Exclusion Contract

## Status

```text
FINITE_QUERY_EQUIVALENCE_ACCEPTED
UNRESTRICTED_REPLAY_EXCLUSION_IMPOSSIBLE_FROM_FINITE_DATA
BOUNDED_REPLAY_CLASS_NOT_YET_FROZEN
NO_LIVE_EXECUTION_AUTHORIZED
```

A finite experiment cannot reject an arbitrary table or unrestricted algorithmic
predictor. This contract defines the strongest honest exclusion claim available to a
future Family 10h campaign.

## 1. Threat model

The source knows before closure:

```text
public carrier preparation grammar
public generator definitions
public formula or relation
public challenge generator family
public measurement law
```

The source must not know:

```text
realized loop word
realized word length
realized amplitudes
realized mapping
realized delay
receiver randomness
```

## 2. Frozen bounded replay class

Before acquisition, define `B_replay` by explicit bounds:

```text
maximum source-authored bits
maximum predictor state dimension
maximum automaton states
maximum polynomial degree
maximum lookup entries
maximum arithmetic-circuit nodes
allowed public side information
allowed mapping, route, timing, and address features
allowed training relations and word lengths
```

A claim can reject only this frozen class and strictly weaker subclasses.

## 3. Causal sequence

```text
source prepares carrier
source exits
source death is verified
source descriptors and IPC close
receiver samples hidden challenge
receiver executes challenge generators
receiver measures carrier and accumulator
features freeze
challenge and source labels unblind
```

Any source-visible challenge seed, shared PRNG state, early word generation, or incomplete
capability revocation invalidates the exclusion campaign.

## 4. Challenge grammar

Use a public generator alphabet:

\[
G=\{A,B,A^{-1},B^{-1},\ldots\}.
\]

A realized word is:

\[
\gamma=g_L\cdots g_1.
\]

The held-out campaign must cross:

```text
unseen words
unseen word lengths
unseen compositions
unseen commutators
unseen powers
unseen inverse placements
unseen amplitudes
unseen mappings
unseen delays
unseen sessions
```

The source may know the grammar but not the realized coordinate in the cross-product.

## 5. Required generalization law

The candidate must predict unseen word responses through one frozen transport model:

\[
T_{\gamma_2\gamma_1}=T_{\gamma_2}T_{\gamma_1}.
\]

The same fitted generator maps must explain:

```text
single generators
pairs
contractible words
commutators
inverse commutators
doubled words
cancellation words
```

Refitting a separate scalar response for each word is replay, not composition.

## 6. Capacity separation

Measure a conservative upper bound `C_prep` on source-authored carrier capacity under the
actual preparation and lifetime law.

For a raw answer table over challenge family `Q`, define:

\[
B_{table}=\sum_{q\in Q}bits(response_q).
\]

`B_table > C_prep` rejects only the uncompressed raw-table strategy. It does not reject
a compact algorithm, automaton, or structured predictor. That is why capacity separation
must be combined with held-out composition and causal interventions.

## 7. Predictor suite

Every future analysis must compare against at least:

```text
raw lookup table
bounded finite-state automaton
linear state-space model
polynomial nonlinear state-space model
route and bank model
address and order model
timing-only model
ordinary contention model
word-hash memorizer
public group-law simulator without carrier input
```

All predictors receive exactly the public side information permitted by `B_replay`.
They must use the same training and held-out splits as the candidate carrier model.

## 8. Causal intervention requirement

Generalization alone is not enough. Intervene on:

```text
carrier-off
carrier coupling removed
accumulator coupling removed
generator amplitude zeroed
generator order changed
mapping crossed
reference-only path
word-only replay path
```

A claimed transport response must disappear or transform according to the frozen causal
law. If a word-only predictor reproduces the output without the carrier, the relational
carrier claim fails.

## 9. Allowed result tokens

```text
BOUNDED_REPLAY_CUSTODY_INVALID
BOUNDED_REPLAY_CLASS_NOT_REJECTED
RAW_ANSWER_TABLE_REJECTED_ONLY
BOUNDED_REPLAY_CLASS_REJECTED
```

`BOUNDED_REPLAY_CLASS_REJECTED` requires:

```text
frozen adversary class
valid post-source challenge custody
capacity evidence
held-out group-law generalization
causal carrier intervention
causal accumulator intervention
all fresh replicates passing
```

## 10. Forbidden language

Do not claim:

```text
all answer caches excluded
all classical explanations excluded
information-theoretic relational memory established
```

from a finite campaign.

The strongest accurate statement is:

```text
The frozen bounded replay class failed while one receiver-side transport model
predicted held-out words and passed the declared causal interventions.
```
