Qwen3.6-27B uses the `qwen3_5_text` hybrid backbone. Shapes below are checkpoint-exact from the local safetensors headers. Weight matrices use `[out_features, in_features]`; checkpoint tensors are BF16 unless promoted during computation.

1. Tensor inventory

Global:

```text
embed_tokens.weight                         [248320, 5120]
model.norm.weight                           [5120]
lm_head.weight                              [248320, 5120]   # untied
```

Every one of the 64 decoder layers:

```text
input_layernorm.weight                      [5120]
post_attention_layernorm.weight             [5120]

mlp.gate_proj.weight                        [17408, 5120]
mlp.up_proj.weight                          [17408, 5120]
mlp.down_proj.weight                        [5120, 17408]
```

MLP:

```text
MLP(x) = W_down(SiLU(W_gate x) ⊙ (W_up x))
```

Full-attention layers only:

```text
self_attn.q_proj.weight                     [12288, 5120]
self_attn.k_proj.weight                     [1024, 5120]
self_attn.v_proj.weight                     [1024, 5120]
self_attn.o_proj.weight                     [5120, 6144]
self_attn.q_norm.weight                     [256]
self_attn.k_norm.weight                     [256]
```

Dimensions:

```text
Q heads = 24
KV heads = 4
head_dim = 256
Q width = 24 × 256 = 6144
KV width = 4 × 256 = 1024
KV repetition = 24 / 4 = 6
```

`q_proj` is 12288 because it contains both Q and the attention-output gate:

```text
q_proj output logical shape: [B, T, 24, 2, 256]
per-head layout: [q_head_0, gate_head_0, q_head_1, gate_head_1, ...]
```

Do not interpret it as one contiguous 6144-wide Q block followed by one contiguous gate block.

Linear-attention/Gated-DeltaNet layers only:

```text
linear_attn.in_proj_qkv.weight              [10240, 5120]
linear_attn.in_proj_z.weight                [6144, 5120]
linear_attn.in_proj_b.weight                [48, 5120]
linear_attn.in_proj_a.weight                [48, 5120]
linear_attn.conv1d.weight                   [10240, 1, 4]
linear_attn.A_log                           [48]
linear_attn.dt_bias                         [48]
linear_attn.norm.weight                     [128]
linear_attn.out_proj.weight                 [5120, 6144]
```

Projection decomposition:

```text
key_dim   = 16 × 128 = 2048
value_dim = 48 × 128 = 6144
conv_dim  = Q 2048 + K 2048 + V 6144 = 10240

in_proj_qkv row order = [Q, K, V]
```

There are no linear biases other than the learned `dt_bias`.

Norm detail important for `.holo`:

```text
block input/post norms, Q/K norms, final norm:
    RMS(x) = x / sqrt(mean(x²) + 1e-6) × (1 + stored_weight)

linear_attn.norm:
    RMS(y) = y / sqrt(mean(y²) + 1e-6) × stored_weight
```

The GDN norm uses the stored weight directly, not `1 + weight`.

2. `full_attention_interval`

The interval is one-based:

```text
layer_type[i] =
    full_attention   if (i + 1) % full_attention_interval == 0
    linear_attention otherwise
```

With interval 4, the 64-layer pattern is:

```text
LLLF LLLF LLLF LLLF ... repeated 16 times
```

Zero-based full-attention indices:

```text
3, 7, 11, 15, 19, 23, 27, 31,
35, 39, 43, 47, 51, 55, 59, 63
```

Thus there are 48 GDN layers and 16 full-attention layers. The explicit `layer_types` array is authoritative; here it exactly matches the interval rule.

3. Exact linear-attention forward

Let `x_t ∈ R^5120` be the block-pre-normalized token.

Projections:

```text
u_t = W_qkv x_t                    # 10240
z_t = reshape(W_z x_t, 48, 128)
b_t = W_b x_t                      # 48
a_t = W_a x_t                      # 48
```

Apply a bias-free, depthwise, causal width-4 convolution independently to every channel, then SiLU:

```text
c[t,d] = SiLU(
    C[d,0]u[t-3,d] +
    C[d,1]u[t-2,d] +
    C[d,2]u[t-1,d] +
    C[d,3]u[t,d]
)
```

Split and reshape:

```text
q = c[..., 0:2048]                 -> [B,T,16,128]
k = c[..., 2048:4096]              -> [B,T,16,128]
v = c[..., 4096:10240]             -> [B,T,48,128]
```

Repeat each Q/K head three times to align 16 QK heads with 48 V heads:

```text
q, k -> [B,T,48,128]
```

Per token and head:

```text
q̂ = q / sqrt(sum(q²) + 1e-6)
k̂ = k / sqrt(sum(k²) + 1e-6)
q̃ = q̂ / sqrt(128)

β_t = sigmoid(b_t)
g_t = -exp(A_log) × softplus(a_t + dt_bias)
λ_t = exp(g_t)
```

`g ≤ 0`, so `0 < λ ≤ 1`. `A_log`, gate calculations, Q/K/V and recurrent math should be promoted to FP32; this matches `mamba_ssm_dtype = float32`.

For each of 48 heads, keep recurrent state:

```text
S_t ∈ R^(128×128)
```

Exact recurrent delta rule:

```text
S_decay = λ_t S_(t-1)

prediction = k̂_tᵀ S_decay                    # [128]
delta      = β_t (v_t - prediction)          # [128]

S_t = S_decay + k̂_t ⊗ delta                 # [128,128]
y_t = q̃_tᵀ S_t                              # [128]
```

Equivalent single equation:

```text
S_t = λ_t S_(t-1)
    + k̂_t ⊗ {β_t [v_t - k̂_tᵀ(λ_t S_(t-1))]}
```

Then apply the per-head gated RMSNorm:

```text
n_t = RMSNorm_direct_weight(y_t)
r_t = n_t ⊙ SiLU(z_t)
output = W_out flatten(r_t)                   # 6144 -> 5120
```

The chunked prefill kernel is an algebraically parallelized implementation of this recurrence; decode can use the recurrence directly. Minimal caches per GDN layer are the causal-convolution history and 48 recurrent `[128,128]` matrices. This matches the reference Gated DeltaNet implementation. [Hugging Face source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

4. Partial RoPE

RoPE applies only in full-attention layers, and only to Q and K:

```text
head_dim = 256
partial_rotary_factor = 0.25
rotary_dim = 256 × 0.25 = 64
pass-through dimension = 192
```

After per-head Q/K RMSNorm:

```text
q_rot  = q[..., 0:64]
q_pass = q[..., 64:256]
k_rot  = k[..., 0:64]
k_pass = k[..., 64:256]
```

For the rotary section, NeoX-style `rotate_half` splits 64 into two 32-wide halves:

```text
rotate_half([x0, x1]) = [-x1, x0]

q_rot' = q_rot ⊙ cos + rotate_half(q_rot) ⊙ sin
k_rot' = k_rot ⊙ cos + rotate_half(k_rot) ⊙ sin

q' = concat(q_rot', q_pass)
k' = concat(k_rot', k_pass)
```

There are 32 frequency pairs with `rope_theta = 10,000,000`.

For multimodal MRoPE, `mrope_section = [11,11,10]` distributes those 32 frequencies across temporal, height and width positions in an interleaved pattern. For ordinary text, all three position rows are equal, reducing to standard partial RoPE.

5. Gating semantics

Full attention:

```text
(q, gate) = per-head split(q_proj(x))
q = QK_RMSNorm(q)
q, k = partial_RoPE(q, k)

a = causal_softmax(q kᵀ / sqrt(256)) v
a = flatten(a) ⊙ sigmoid(gate)
output = W_o a
```

`attn_output_gate = true` controls this doubled Q projection and sigmoid multiplication before `o_proj`.

`output_gate_type = "swish"` is separate: it controls the GDN `z` output gate:

```text
GDN output gate = SiLU(z) = z × sigmoid(z)
```

It does not change full-attention gating to SiLU; full attention still uses `sigmoid(gate)`. [vLLM’s compatible implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)

6. Layer/block ordering

Each decoder block is pre-norm with two residual branches:

```text
function decoder_layer(x, type):
    residual = x
    h = RMSNorm_offset(x)

    if type == linear_attention:
        h = GatedDeltaNet(h)
    else:
        h = GatedFullAttention(h)

    x = residual + h

    residual = x
    h = RMSNorm_offset(x)
    h = W_down(SiLU(W_gate h) ⊙ W_up h)
    x = residual + h

    return x
```

Whole text model:

```text
token embedding
-> 16 × [GDN block, GDN block, GDN block, full-attention block]
-> final RMSNorm_offset
-> untied LM head
```

The checkpoint also contains a separate one-layer `mtp.*` prediction module; it is not part of the normal 64-layer autoregressive path.
hook: Stop
hook: Stop Completed
tokens used
138,442
Qwen3.6-27B uses the `qwen3_5_text` hybrid backbone. Shapes below are checkpoint-exact from the local safetensors headers. Weight matrices use `[out_features, in_features]`; checkpoint tensors are BF16 unless promoted during computation.

1. Tensor inventory

Global:

```text
embed_tokens.weight                         [248320, 5120]
model.norm.weight                           [5120]
lm_head.weight                              [248320, 5120]   # untied
```

Every one of the 64 decoder layers:

```text
input_layernorm.weight                      [5120]
post_attention_layernorm.weight             [5120]

mlp.gate_proj.weight                        [17408, 5120]
mlp.up_proj.weight                          [17408, 5120]
mlp.down_proj.weight                        [5120, 17408]
```

MLP:

```text
MLP(x) = W_down(SiLU(W_gate x) ⊙ (W_up x))
```

Full-attention layers only:

```text
self_attn.q_proj.weight                     [12288, 5120]
self_attn.k_proj.weight                     [1024, 5120]
self_attn.v_proj.weight                     [1024, 5120]
self_attn.o_proj.weight                     [5120, 6144]
self_attn.q_norm.weight                     [256]
self_attn.k_norm.weight                     [256]
```

Dimensions:

```text
Q heads = 24
KV heads = 4
head_dim = 256
Q width = 24 × 256 = 6144
KV width = 4 × 256 = 1024
KV repetition = 24 / 4 = 6
```

`q_proj` is 12288 because it contains both Q and the attention-output gate:

```text
q_proj output logical shape: [B, T, 24, 2, 256]
per-head layout: [q_head_0, gate_head_0, q_head_1, gate_head_1, ...]
```

Do not interpret it as one contiguous 6144-wide Q block followed by one contiguous gate block.

Linear-attention/Gated-DeltaNet layers only:

```text
linear_attn.in_proj_qkv.weight              [10240, 5120]
linear_attn.in_proj_z.weight                [6144, 5120]
linear_attn.in_proj_b.weight                [48, 5120]
linear_attn.in_proj_a.weight                [48, 5120]
linear_attn.conv1d.weight                   [10240, 1, 4]
linear_attn.A_log                           [48]
linear_attn.dt_bias                         [48]
linear_attn.norm.weight                     [128]
linear_attn.out_proj.weight                 [5120, 6144]
```

Projection decomposition:

```text
key_dim   = 16 × 128 = 2048
value_dim = 48 × 128 = 6144
conv_dim  = Q 2048 + K 2048 + V 6144 = 10240

in_proj_qkv row order = [Q, K, V]
```

There are no linear biases other than the learned `dt_bias`.

Norm detail important for `.holo`:

```text
block input/post norms, Q/K norms, final norm:
    RMS(x) = x / sqrt(mean(x²) + 1e-6) × (1 + stored_weight)

linear_attn.norm:
    RMS(y) = y / sqrt(mean(y²) + 1e-6) × stored_weight
```

The GDN norm uses the stored weight directly, not `1 + weight`.

2. `full_attention_interval`

The interval is one-based:

```text
layer_type[i] =
    full_attention   if (i + 1) % full_attention_interval == 0
    linear_attention otherwise
```

With interval 4, the 64-layer pattern is:

```text
LLLF LLLF LLLF LLLF ... repeated 16 times
```

Zero-based full-attention indices:

```text
3, 7, 11, 15, 19, 23, 27, 31,
35, 39, 43, 47, 51, 55, 59, 63
```

Thus there are 48 GDN layers and 16 full-attention layers. The explicit `layer_types` array is authoritative; here it exactly matches the interval rule.

3. Exact linear-attention forward

Let `x_t ∈ R^5120` be the block-pre-normalized token.

Projections:

```text
u_t = W_qkv x_t                    # 10240
z_t = reshape(W_z x_t, 48, 128)
b_t = W_b x_t                      # 48
a_t = W_a x_t                      # 48
```

Apply a bias-free, depthwise, causal width-4 convolution independently to every channel, then SiLU:

```text
c[t,d] = SiLU(
    C[d,0]u[t-3,d] +
    C[d,1]u[t-2,d] +
    C[d,2]u[t-1,d] +
    C[d,3]u[t,d]
)
```

Split and reshape:

```text
q = c[..., 0:2048]                 -> [B,T,16,128]
k = c[..., 2048:4096]              -> [B,T,16,128]
v = c[..., 4096:10240]             -> [B,T,48,128]
```

Repeat each Q/K head three times to align 16 QK heads with 48 V heads:

```text
q, k -> [B,T,48,128]
```

Per token and head:

```text
q̂ = q / sqrt(sum(q²) + 1e-6)
k̂ = k / sqrt(sum(k²) + 1e-6)
q̃ = q̂ / sqrt(128)

β_t = sigmoid(b_t)
g_t = -exp(A_log) × softplus(a_t + dt_bias)
λ_t = exp(g_t)
```

`g ≤ 0`, so `0 < λ ≤ 1`. `A_log`, gate calculations, Q/K/V and recurrent math should be promoted to FP32; this matches `mamba_ssm_dtype = float32`.

For each of 48 heads, keep recurrent state:

```text
S_t ∈ R^(128×128)
```

Exact recurrent delta rule:

```text
S_decay = λ_t S_(t-1)

prediction = k̂_tᵀ S_decay                    # [128]
delta      = β_t (v_t - prediction)          # [128]

S_t = S_decay + k̂_t ⊗ delta                 # [128,128]
y_t = q̃_tᵀ S_t                              # [128]
```

Equivalent single equation:

```text
S_t = λ_t S_(t-1)
    + k̂_t ⊗ {β_t [v_t - k̂_tᵀ(λ_t S_(t-1))]}
```

Then apply the per-head gated RMSNorm:

```text
n_t = RMSNorm_direct_weight(y_t)
r_t = n_t ⊙ SiLU(z_t)
output = W_out flatten(r_t)                   # 6144 -> 5120
```

The chunked prefill kernel is an algebraically parallelized implementation of this recurrence; decode can use the recurrence directly. Minimal caches per GDN layer are the causal-convolution history and 48 recurrent `[128,128]` matrices. This matches the reference Gated DeltaNet implementation. [Hugging Face source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

4. Partial RoPE

RoPE applies only in full-attention layers, and only to Q and K:

```text
head_dim = 256
partial_rotary_factor = 0.25
rotary_dim = 256 × 0.25 = 64
pass-through dimension = 192
```

After per-head Q/K RMSNorm:

```text
q_rot  = q[..., 0:64]
q_pass = q[..., 64:256]
k_rot  = k[..., 0:64]
k_pass = k[..., 64:256]
```

For the rotary section, NeoX-style `rotate_half` splits 64 into two 32-wide halves:

```text
rotate_half([x0, x1]) = [-x1, x0]

q_rot' = q_rot ⊙ cos + rotate_half(q_rot) ⊙ sin
k_rot' = k_rot ⊙ cos + rotate_half(k_rot) ⊙ sin

q' = concat(q_rot', q_pass)
k' = concat(k_rot', k_pass)
```

There are 32 frequency pairs with `rope_theta = 10,000,000`.

For multimodal MRoPE, `mrope_section = [11,11,10]` distributes those 32 frequencies across temporal, height and width positions in an interleaved pattern. For ordinary text, all three position rows are equal, reducing to standard partial RoPE.

5. Gating semantics

Full attention:

```text
(q, gate) = per-head split(q_proj(x))
q = QK_RMSNorm(q)
q, k = partial_RoPE(q, k)

a = causal_softmax(q kᵀ / sqrt(256)) v
a = flatten(a) ⊙ sigmoid(gate)
output = W_o a
```

`attn_output_gate = true` controls this doubled Q projection and sigmoid multiplication before `o_proj`.

`output_gate_type = "swish"` is separate: it controls the GDN `z` output gate:

```text
GDN output gate = SiLU(z) = z × sigmoid(z)
```

It does not change full-attention gating to SiLU; full attention still uses `sigmoid(gate)`. [vLLM’s compatible implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)

6. Layer/block ordering

Each decoder block is pre-norm with two residual branches:

```text
function decoder_layer(x, type):
    residual = x
    h = RMSNorm_offset(x)

    if type == linear_attention:
        h = GatedDeltaNet(h)
    else:
        h = GatedFullAttention(h)

    x = residual + h

    residual = x
    h = RMSNorm_offset(x)
    h = W_down(SiLU(W_gate h) ⊙ W_up h)
    x = residual + h

    return x
```

Whole text model:

```text
token embedding
-> 16 × [GDN block, GDN block, GDN block, full-attention block]
-> final RMSNorm_offset
-> untied LM head
```

The checkpoint also contains a separate one-layer `mtp.*` prediction module; it is not part of the normal 64-layer autoregressive path.
