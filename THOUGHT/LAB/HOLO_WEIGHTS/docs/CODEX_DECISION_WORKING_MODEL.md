Decision: choose (c), a hybrid exact-shell model with rank-32 projection adapters trained end-to-end by teacher distillation.

Do not call the current k=256 reconstruction “working.” A 0.4–0.55 matrix cosine repeated through 64 layers is structurally beyond zero-shot recovery. Short context does not solve this: a one-token decode still traverses all 64 damaged blocks.

Why this path

(a) Per-layer output adapters alone

Cheap, but unlikely to work. A rank-64/128 adapter after each mixer or FFN can correct errors on the natural activation manifold, but it cannot reliably repair corrupted Q/K attention patterns or GatedDeltaNet recurrent-state updates. Use these only as diagnostics, not the main recovery mechanism.

Estimated training: 12–30 hours for 1M tokens. Probability of genuine coherence: low.

(b) LoRA over the fully compressed model

More capable because it can correct Q/K/V, GDN and MLP projections before nonlinear operations. However, making LoRA also repair damaged embeddings, LM head, norms and small control projections wastes most of its limited rank.

Estimated training: 48–120 GPU-hours. Risk remains high because vocabulary interfaces are compressed.

(c) Hybrid exact shell plus projection adapters — recommended

Keep numerically and semantically fragile tensors exact BF16:

```text
embed_tokens
lm_head
all input/post/final RMSNorm weights
full-attention q_norm and k_norm
GDN norm
GDN A_log and dt_bias
GDN depthwise conv1d
GDN in_proj_a and in_proj_b
full-attention k_proj and v_proj
```

The additional exact storage is worthwhile:

- Embed + untied LM head: about 5.1 GB BF16.
- Full-attention K/V: about 320 MB.
- GDN a/b and all norms/control tensors: well under 100 MB.

Compress the large matrices at k=256:

```text
MLP: gate_proj, up_proj, down_proj
GDN: in_proj_qkv, in_proj_z, out_proj
Full attention: q_proj, o_proj
```

Add trainable LoRA directly to every compressed projection:

```text
rank = 32
alpha = 32
dropout = 0
A initialization = small normal
B initialization = zero
```

For stored `W ≈ U @ SVh`, compute:

```text
linear(x) =
    (x @ SVh.T) @ U.T
    + ((x @ A.T) @ B.T) * alpha/rank
```

Rank 32 gives roughly 200–220M trainable parameters for this selection: approximately 420 MB BF16 parameters, with around 2–3 GB more for gradients and optimizer state. This is feasible with CPU optimizer offload and layer streaming.

Training plan

Use end-to-end text-path distillation, not W-space residual fitting.

```text
Calibration data: 1,000,000 tokens
Epochs:           2
Tokens trained:   2,000,000
Sequence length:  256 initially
Microbatch:       1
Gradient accum:   8
Effective batch:  2048 tokens
Optimizer:        AdamW
Learning rate:    2e-4
Warmup:           3%
Schedule:         cosine decay
Weight decay:     0
Gradient clip:    1.0
```

Use a Q4/Q5 reference Qwen3.6-27B as teacher and precompute top-64 logits:

```text
loss = 0.8 × KL(student, teacher, temperature=2)
     + 0.2 × next-token cross-entropy
```

Data mixture:

```text
40% ordinary high-quality prose
30% dialogue/instruction-response
20% code
10% multilingual/math
```

Use contiguous documents rather than unrelated isolated fragments so GDN state and attention receive realistic calibration.

Hardware estimate on RTX 3060 12 GB:

```text
Teacher target generation: 24–48 CPU-hours
Student training:          48–96 GPU-hours
Expected wall time:        3–6 days
```

This assumes:

- Frozen factors remain in CPU RAM or are streamed layerwise.
- Only the active block and adapters reside on GPU.
- Exact LM-head logits are computed in vocabulary chunks.
- Gradient checkpointing is enabled.
- Vision and MTP are excluded.

Thirty gigabytes of RAM is enough for the exact shell, holo factors, adapter state and moderate buffering, but not the original 52 GB checkpoint resident at once.

(d) Redefining “working”

“Short-context coherence only” is a valid milestone, but not an alternative architecture. Context length reduces temporal-state pressure; it does not reduce the 64-layer accumulation problem.

Use context ≤256 for the first acceptance test. Do not claim a working model merely because one cherry-picked prompt produces recognizable words.

Measurable success criterion

Declare the model working only if all of these pass on held-out data:

```text
1. Held-out NLL gap versus the Q4/Q5 teacher ≤ 0.5 nat/token
   (student perplexity ≤ 1.65 × teacher perplexity).

2. 100 fixed prompts × 64 greedy tokens:
   - no NaN/Inf,
   - no repeated cycle of length ≤16,
   - at least 90 outputs remain grammatical and on-topic.

3. Hidden-state RMS at every layer remains within
   0.25×–4× the teacher layer RMS on calibration sequences.

4. GDN prefill and token-by-token decode agree within
   max absolute logit error 1e-2 for a short deterministic sequence.
```

If rank-32 adapters cannot meet these criteria after 2M tokens, do not continue training indefinitely. The honest conclusion is that k=256 is below the viable base rank. Raise the major MLP/GDN matrices to k=512 before trying rank-64 adapters.

Minimal `.holo`-native forward pass

The engine must implement the real text architecture, not substitute ordinary attention:

```text
exact token embedding
for layers 0..63:
    pre-RMSNorm using (1 + stored_weight)

    if layer % 4 != 3:
        GatedDeltaNet:
            factorized+LoRA qkv/z projections
            exact a/b projections
            exact depthwise causal conv1d width 4 + SiLU
            split Q16/K16/V48, head_dim 128
            repeat Q/K heads 3×
            L2-normalize Q/K
            beta = sigmoid(b)
            g = -exp(A_log) * softplus(a + dt_bias)
            FP32 recurrent delta-rule state [48,128,128]
            exact gated RMSNorm: norm(y) * SiLU(z)
            factorized+LoRA out_proj
    else:
        gated full attention:
            factorized+LoRA interleaved Q/gate projection
            exact K/V projections
            exact per-head Q/K RMSNorm
            partial MRoPE on first 64/256 dimensions
            theta=1e7, GQA 24 Q heads / 4 KV heads
            causal softmax in FP32
            attention output * sigmoid(gate)
            factorized+LoRA o_proj

    residual addition
    post-RMSNorm
    factorized+LoRA SwiGLU MLP
    residual addition

exact final RMSNorm
exact chunked LM head
sampling
```

Cache only full-attention K/V at layers `3,7,…,63`. Each GDN layer instead needs its width-4 convolution history and FP32 recurrent state.

This hybrid is the smallest path with a credible chance of coherent output. It sacrifices an additional ~5.5 GB to preserve the vocabulary and dynamical control shell, while using training capacity where compression actually damaged nonlinear behavior.
hook: Stop
hook: Stop Completed
tokens used
227,726
Decision: choose (c), a hybrid exact-shell model with rank-32 projection adapters trained end-to-end by teacher distillation.

Do not call the current k=256 reconstruction “working.” A 0.4–0.55 matrix cosine repeated through 64 layers is structurally beyond zero-shot recovery. Short context does not solve this: a one-token decode still traverses all 64 damaged blocks.

Why this path

(a) Per-layer output adapters alone

Cheap, but unlikely to work. A rank-64/128 adapter after each mixer or FFN can correct errors on the natural activation manifold, but it cannot reliably repair corrupted Q/K attention patterns or GatedDeltaNet recurrent-state updates. Use these only as diagnostics, not the main recovery mechanism.

Estimated training: 12–30 hours for 1M tokens. Probability of genuine coherence: low.

(b) LoRA over the fully compressed model

More capable because it can correct Q/K/V, GDN and MLP projections before nonlinear operations. However, making LoRA also repair damaged embeddings, LM head, norms and small control projections wastes most of its limited rank.

Estimated training: 48–120 GPU-hours. Risk remains high because vocabulary interfaces are compressed.

(c) Hybrid exact shell plus projection adapters — recommended

Keep numerically and semantically fragile tensors exact BF16:

```text
embed_tokens
lm_head
all input/post/final RMSNorm weights
full-attention q_norm and k_norm
GDN norm
GDN A_log and dt_bias
GDN depthwise conv1d
GDN in_proj_a and in_proj_b
full-attention k_proj and v_proj
```

The additional exact storage is worthwhile:

- Embed + untied LM head: about 5.1 GB BF16.
- Full-attention K/V: about 320 MB.
- GDN a/b and all norms/control tensors: well under 100 MB.

Compress the large matrices at k=256:

```text
MLP: gate_proj, up_proj, down_proj
GDN: in_proj_qkv, in_proj_z, out_proj
Full attention: q_proj, o_proj
```

Add trainable LoRA directly to every compressed projection:

```text
rank = 32
alpha = 32
dropout = 0
A initialization = small normal
B initialization = zero
```

For stored `W ≈ U @ SVh`, compute:

```text
linear(x) =
    (x @ SVh.T) @ U.T
    + ((x @ A.T) @ B.T) * alpha/rank
```

Rank 32 gives roughly 200–220M trainable parameters for this selection: approximately 420 MB BF16 parameters, with around 2–3 GB more for gradients and optimizer state. This is feasible with CPU optimizer offload and layer streaming.

Training plan

Use end-to-end text-path distillation, not W-space residual fitting.

```text
Calibration data: 1,000,000 tokens
Epochs:           2
Tokens trained:   2,000,000
Sequence length:  256 initially
Microbatch:       1
Gradient accum:   8
Effective batch:  2048 tokens
Optimizer:        AdamW
Learning rate:    2e-4
Warmup:           3%
Schedule:         cosine decay
Weight decay:     0
Gradient clip:    1.0
```

Use a Q4/Q5 reference Qwen3.6-27B as teacher and precompute top-64 logits:

```text
loss = 0.8 × KL(student, teacher, temperature=2)
     + 0.2 × next-token cross-entropy
```

Data mixture:

```text
40% ordinary high-quality prose
30% dialogue/instruction-response
20% code
10% multilingual/math
```

Use contiguous documents rather than unrelated isolated fragments so GDN state and attention receive realistic calibration.

Hardware estimate on RTX 3060 12 GB:

```text
Teacher target generation: 24–48 CPU-hours
Student training:          48–96 GPU-hours
Expected wall time:        3–6 days
```

This assumes:

- Frozen factors remain in CPU RAM or are streamed layerwise.
- Only the active block and adapters reside on GPU.
- Exact LM-head logits are computed in vocabulary chunks.
- Gradient checkpointing is enabled.
- Vision and MTP are excluded.

Thirty gigabytes of RAM is enough for the exact shell, holo factors, adapter state and moderate buffering, but not the original 52 GB checkpoint resident at once.

(d) Redefining “working”

“Short-context coherence only” is a valid milestone, but not an alternative architecture. Context length reduces temporal-state pressure; it does not reduce the 64-layer accumulation problem.

Use context ≤256 for the first acceptance test. Do not claim a working model merely because one cherry-picked prompt produces recognizable words.

Measurable success criterion

Declare the model working only if all of these pass on held-out data:

```text
1. Held-out NLL gap versus the Q4/Q5 teacher ≤ 0.5 nat/token
   (student perplexity ≤ 1.65 × teacher perplexity).

2. 100 fixed prompts × 64 greedy tokens:
   - no NaN/Inf,
   - no repeated cycle of length ≤16,
   - at least 90 outputs remain grammatical and on-topic.

3. Hidden-state RMS at every layer remains within
   0.25×–4× the teacher layer RMS on calibration sequences.

4. GDN prefill and token-by-token decode agree within
   max absolute logit error 1e-2 for a short deterministic sequence.
```

If rank-32 adapters cannot meet these criteria after 2M tokens, do not continue training indefinitely. The honest conclusion is that k=256 is below the viable base rank. Raise the major MLP/GDN matrices to k=512 before trying rank-64 adapters.

Minimal `.holo`-native forward pass

The engine must implement the real text architecture, not substitute ordinary attention:

```text
exact token embedding
for layers 0..63:
    pre-RMSNorm using (1 + stored_weight)

    if layer % 4 != 3:
        GatedDeltaNet:
            factorized+LoRA qkv/z projections
            exact a/b projections
            exact depthwise causal conv1d width 4 + SiLU
            split Q16/K16/V48, head_dim 128
            repeat Q/K heads 3×
            L2-normalize Q/K
            beta = sigmoid(b)
            g = -exp(A_log) * softplus(a + dt_bias)
            FP32 recurrent delta-rule state [48,128,128]
            exact gated RMSNorm: norm(y) * SiLU(z)
            factorized+LoRA out_proj
    else:
        gated full attention:
            factorized+LoRA interleaved Q/gate projection
            exact K/V projections
            exact per-head Q/K RMSNorm
            partial MRoPE on first 64/256 dimensions
            theta=1e7, GQA 24 Q heads / 4 KV heads
            causal softmax in FP32
            attention output * sigmoid(gate)
            factorized+LoRA o_proj

    residual addition
    post-RMSNorm
    factorized+LoRA SwiGLU MLP
    residual addition

exact final RMSNorm
exact chunked LM head
sampling
```

Cache only full-attention K/V at layers `3,7,…,63`. Each GDN layer instead needs its width-4 convolution history and FP32 recurrent state.

This hybrid is the smallest path with a credible chance of coherent output. It sacrifices an additional ~5.5 GB to preserve the vocabulary and dynamical control shell, while using training capacity where compression actually damaged nonlinear behavior.
