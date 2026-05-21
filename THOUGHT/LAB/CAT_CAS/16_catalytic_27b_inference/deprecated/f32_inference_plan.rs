// f32 tape compute block — replaced in lib.rs via edit_file below
// This file is a reference, not compilable standalone
// See lib.rs lines ~354-750 for the actual replacement

// ------ OFFSET LAYOUT (all sizes in bytes, f32=4 bytes per value) ------
let input_offset = 0usize;
// input region: HIDDEN_DIM * F32_BYTES * 2 = COMPLEX_DIM bytes (XY channels)
let weight_offset = COMPLEX_DIM;  // store weight bytes for one layer here
let scratch_base = weight_offset + HIDDEN_DIM * F32_BYTES;  // weights: HIDDEN_DIM f32 values
let temp_offset = scratch_base;
let pre_gate_base = temp_offset + COMPLEX_DIM;
let saved_outputs_offset = pre_gate_base + num_layers * COMPLEX_DIM;
let warm_tape_offset = saved_outputs_offset + num_layers * COMPLEX_DIM;
let warm_tape_stride = 4 + COMPLEX_DIM;
let kv_cache_offset = warm_tape_offset + WARM_TAPE_SLOTS * warm_tape_stride;
// ------ end offset layout ------

// COLD MISS: full layer stack
for layer_idx in 0..num_layers {
    let lwo = weight_offset + layer_idx * HIDDEN_DIM * F32_BYTES;
    let layer_save = saved_outputs_offset + layer_idx * COMPLEX_DIM;

    // Dynamic decatalysis: unscramble layer weights into tape
    let original_weight = tape[lwo..lwo + HIDDEN_DIM * F32_BYTES].to_vec();
    let src = &compressed_weights[layer_idx * HIDDEN_DIM..(layer_idx + 1) * HIDDEN_DIM];
    tape[lwo..lwo + HIDDEN_DIM].copy_from_slice(src);
    spn_unscramble(&mut tape, lwo, HIDDEN_DIM, bh_key, &sbox, 12);

    if (layer_idx + 1) % 4 == 0 {
        // GATED ATTENTION — f32 complex
        let attn_idx = layer_idx / 4;
        
        // RMS LayerNorm
        let mut sum_sq = 0.0f32;
        for j in 0..HIDDEN_DIM {
            let rx = tape_f32(&tape, input_offset, j);
            let ry = tape_f32(&tape, input_offset + F32_DIM, j);
            sum_sq += rx * rx + ry * ry;
        }
        let rms = (sum_sq / HIDDEN_DIM as f32 + 1e-5).sqrt();
        let inv_rms = 1.0 / rms;
        let mut nx = vec![0.0f32; HIDDEN_DIM * 2];
        for j in 0..HIDDEN_DIM {
            nx[j] = tape_f32(&tape, input_offset, j) * inv_rms;
            nx[HIDDEN_DIM + j] = tape_f32(&tape, input_offset + F32_DIM, j) * inv_rms;
        }

        // QKV projections
        let slot_base = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + step % MAX_SEQ_LEN) * 4 * F32_DIM;
        for j in 0..HIDDEN_DIM {
            let wq = tape_f32(&tape, lwo, j % 512);
            let wk = tape_f32(&tape, lwo + 512 * F32_BYTES, j % 512);
            let wv = tape_f32(&tape, lwo + 1024 * F32_BYTES, j % 512);
            let qx = wq * nx[j]; let qy = wq * nx[HIDDEN_DIM + j];
            let kx = wk * nx[j]; let ky = wk * nx[HIDDEN_DIM + j];
            let vx = wv * nx[j]; let vy = wv * nx[HIDDEN_DIM + j];
            tape_f32_xor(&mut tape, pre_gate_base + layer_idx * COMPLEX_DIM, j, qx);
            tape_f32_xor(&mut tape, pre_gate_base + layer_idx * COMPLEX_DIM + F32_DIM, j, qy);
            tape_f32_xor(&mut tape, slot_base, j, kx);
            tape_f32_xor(&mut tape, slot_base + F32_DIM, j, ky);
            tape_f32_xor(&mut tape, slot_base + F32_DIM * 2, j, vx);
            tape_f32_xor(&mut tape, slot_base + F32_DIM * 3, j, vy);
            total_entropy += (qx.abs() as u64) + (ky.abs() as u64) + (vx.abs() as u64);
        }

        // Multi-head attention (16 heads, head_dim = HIDDEN_DIM / 16)
        let num_heads = 16;
        let head_dim = HIDDEN_DIM / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_x = vec![0.0f32; HIDDEN_DIM];
        let mut attn_y = vec![0.0f32; HIDDEN_DIM];

        for h in 0..num_heads {
            let h_start = h * head_dim;
            let mut scores = vec![0.0f32; step + 1];
            let mut max_score = f32::NEG_INFINITY;
            for s in 0..=step {
                let so = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + s % MAX_SEQ_LEN) * 4 * F32_DIM;
                let mut dr = 0.0f32; let mut di = 0.0f32;
                for d in 0..head_dim {
                    let qx = tape_f32(&tape, pre_gate_base + layer_idx * COMPLEX_DIM, h_start + d);
                    let qy = tape_f32(&tape, pre_gate_base + layer_idx * COMPLEX_DIM + F32_DIM, h_start + d);
                    let kx = tape_f32(&tape, so, h_start + d);
                    let ky = tape_f32(&tape, so + F32_DIM, h_start + d);
                    dr += qx * kx + qy * ky;
                    di += qy * kx - qx * ky;
                }
                let score = (dr * dr + di * di).sqrt() * scale;
                scores[s] = score;
                if score > max_score { max_score = score; }
            }
            let mut sum_exp = 0.0f32;
            let mut exp_scores = vec![0.0f32; step + 1];
            for s in 0..=step { exp_scores[s] = (scores[s] - max_score).exp(); sum_exp += exp_scores[s]; }
            for d in 0..head_dim {
                let mut vsx = 0.0f32; let mut vsy = 0.0f32;
                for s in 0..=step {
                    let so = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + s % MAX_SEQ_LEN) * 4 * F32_DIM;
                    let vx = tape_f32(&tape, so + F32_DIM * 2, h_start + d);
                    let vy = tape_f32(&tape, so + F32_DIM * 3, h_start + d);
                    let w = exp_scores[s] / sum_exp;
                    vsx += w * vx; vsy += w * vy;
                }
                attn_x[h_start + d] = vsx;
                attn_y[h_start + d] = vsy;
                total_entropy += (vsx.abs() as u64) + (vsy.abs() as u64);
            }
        }

        // Output projection + add to input
        for j in 0..HIDDEN_DIM {
            let wo = tape_f32(&tape, lwo + 1536 * F32_BYTES, j % 512);
            let px = wo * attn_x[j]; let py = wo * attn_y[j];
            tape_f32_xor(&mut tape, layer_save, j, px);
            tape_f32_xor(&mut tape, layer_save + F32_DIM, j, py);
            tape_f32_xor(&mut tape, input_offset, j, px);
            tape_f32_xor(&mut tape, input_offset + F32_DIM, j, py);
            total_entropy += (px.abs() as u64) + (py.abs() as u64);
        }
    } else {
        // DELTANET — f32 complex projection with f32 real weights
        for j in 0..HIDDEN_DIM {
            let w = tape_f32(&tape, lwo, j);
            let x = tape_f32(&tape, input_offset, j);
            let y = tape_f32(&tape, input_offset + F32_DIM, j);
            let vx = w * x; let vy = w * y;
            total_entropy += (vx.abs() as u64) + (vy.abs() as u64);
            // Save original temp
            tape_f32_xor(&mut tape, pre_gate_base + layer_idx * COMPLEX_DIM, j,
                tape_f32(&tape, temp_offset, j));
            tape_f32_xor(&mut tape, pre_gate_base + layer_idx * COMPLEX_DIM + F32_DIM, j,
                tape_f32(&tape, temp_offset + F32_DIM, j));
            tape_f32_xor(&mut tape, temp_offset, j, vx);
            tape_f32_xor(&mut tape, temp_offset + F32_DIM, j, vy);
        }
        for j in 0..HIDDEN_DIM {
            let gx = (0.5 + 0.25 * tape_f32(&tape, temp_offset, j)).clamp(0.0, 1.0);
            let gy = (0.5 + 0.25 * tape_f32(&tape, temp_offset + F32_DIM, j)).clamp(0.0, 1.0);
            total_entropy += (gx.abs() as u64) + (gy.abs() as u64);
            tape_f32_xor(&mut tape, layer_save, j, gx);
            tape_f32_xor(&mut tape, layer_save + F32_DIM, j, gy);
            tape_f32_xor(&mut tape, input_offset, j, gx);
            tape_f32_xor(&mut tape, input_offset + F32_DIM, j, gy);
        }
    }

    // Re-scramble and restore weight substrate
    spn_scramble(&mut tape, lwo, HIDDEN_DIM, bh_key, &sbox, 12);
    tape[lwo..lwo + HIDDEN_DIM].copy_from_slice(&src);
    // restore original substrate (could be dirty from prior layers)
    tape[lwo..lwo + HIDDEN_DIM * F32_BYTES].copy_from_slice(&original_weight);
}

// Output head — max absolute f32 value across first 64 dims
let mut best_score = f32::NEG_INFINITY;
for j in 0..64.min(HIDDEN_DIM) {
    let s = tape_f32(&tape, input_offset, j).abs();
    if s > best_score { best_score = s; best_token = j as u8; }
}

// Uncompute — NO f32 uncompute needed since tape stores values via XOR
// The XOR fabric auto-restores: tape_f32_xor(tape, offset, j, val) followed by
// same call with same val → XOR cancels. But we need the layer-by-layer
// reverse uncompute. For now: skip uncompute (batch the restore in next pass).
// TODO: implement reverse uncompute with f32 tape helpers.
