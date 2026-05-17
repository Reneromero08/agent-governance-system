"""Block diffusion generation for TraDo/SDAR models.
Adapted from https://github.com/Gen-Verse/dLLM-RL/blob/main/generate.py

Usage:
    output_ids = block_diffusion_generate(
        model, prompt_tokens, mask_id=151669,
        gen_length=200, block_length=4, denoising_steps=4,
        temperature=1.0
    )
"""

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache


def top_k_logits(logits, k):
    if k <= 0: return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask_indices, float('-inf'))


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    if temperature != 1.0: logits = logits / temperature
    if top_k > 0: logits = top_k_logits(logits, top_k)
    if top_p < 1.0: logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    return token.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def block_diffusion_generate(
    model,
    prompt,
    mask_id=151669,
    gen_length=200,
    block_length=4,
    denoising_steps=4,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    remasking_strategy='low_confidence_dynamic',
    confidence_threshold=0.9,
):
    model.eval()
    input_ids = prompt['input_ids']
    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()

    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1).unsqueeze(0)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
        if cur_attn_mask.dim() == 3: cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, :prefill_length]
        model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
              past_key_values=past_key_values, use_cache=True, store_kv=True)

    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    for num_block in range(prefill_blocks, num_blocks):
        cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
        cur_attn_mask = block_diffusion_attention_mask[:, num_block*block_length:(num_block+1)*block_length, :(num_block+1)*block_length]
        if cur_attn_mask.dim() == 3: cur_attn_mask = cur_attn_mask[:, None, :, :]
        cur_position_ids = position_ids[:, num_block*block_length:(num_block+1)*block_length]

        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
                      past_key_values=past_key_values, use_cache=True, store_kv=True)
                break

            logits = model(cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids,
                          past_key_values=past_key_values, use_cache=True, store_kv=False).logits
            x0 = sample_with_temperature_topk_topp(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            confidence = F.softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)
            x0_p = confidence.gather(-1, x0.reshape(-1, 1)).reshape(x0.shape)

            if remasking_strategy == 'low_confidence_dynamic':
                conf = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(conf.shape[0]):
                    high_conf_mask = conf[j] > confidence_threshold
                    if high_conf_mask.sum() >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(conf[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
            else:
                conf = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(conf.shape[0]):
                    _, idx = torch.topk(conf[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            cur_x[transfer_index] = x0[transfer_index]

        x[:, num_block*block_length:(num_block+1)*block_length] = cur_x

    return x
