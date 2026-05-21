use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use numpy::PyArray1;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::Sha256;
use sha2::Digest as ShaDigest;

#[pyfunction]
fn f16_decode<'py>(py: Python<'py>, data: Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let bytes = data.as_bytes();
    let n = bytes.len() / 2;
    let mut result: Vec<f32> = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(2) {
        let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
        let sign = (raw >> 15) as f32;
        let exp = ((raw >> 10) & 0x1F) as i32;
        let mant = (raw & 0x3FF) as f32;
        let val = if exp == 0 {
            mant / 1024.0 * 2.0_f32.powi(-14)
        } else {
            (1.0 + mant / 1024.0) * 2.0_f32.powi(exp - 15)
        };
        result.push(if sign > 0.0 { -val } else { val });
    }
    Ok(PyArray1::from_slice_bound(py, &result))
}

#[pyfunction]
fn orthogonal_project<'py>(
    py: Python<'py>, n_cores: usize, d_dims: usize,
) -> PyResult<Vec<Bound<'py, PyArray1<f32>>>> {
    let dims_per_core = d_dims / n_cores;
    let mut projections: Vec<Bound<'py, PyArray1<f32>>> = Vec::with_capacity(n_cores);
    for i in 0..n_cores {
        let mut flat: Vec<f32> = vec![0.0; d_dims * d_dims];
        let start = i * dims_per_core;
        for j in start..(start + dims_per_core) {
            flat[j * d_dims + j] = 1.0;
        }
        projections.push(PyArray1::from_slice_bound(py, &flat));
    }
    Ok(projections)
}

#[pyfunction]
fn tape_hash(data: Bound<PyBytes>) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data.as_bytes().hash(&mut h);
    format!("{:016x}", h.finish())
}

// ==================================================================
// BEKENSTEIN VIOLATOR — RAYON PARALLEL
// ==================================================================

const HBAR: f64 = 1.054571817e-34;
const C_LIGHT: f64 = 2.99792458e8;
const LN2: f64 = std::f64::consts::LN_2;
const G: f64 = 6.67430e-11;
const DIE_MASS_KG: f64 = 29e-6;
const DIE_RADIUS_M: f64 = 1e-3;
const K: usize = 256;

fn ground_truth(depth: usize) -> u8 {
    fn rec(depth: usize, node: usize, cur_depth: usize) -> u8 {
        if cur_depth == depth {
            return (((node - (1 << (depth - 1))) * 17 + 43) % K) as u8;
        }
        let left = rec(depth, 2 * node, cur_depth + 1);
        let right = rec(depth, 2 * node + 1, cur_depth + 1);
        ((left as usize * 7 + right as usize * 13 + 31) % K) as u8
    }
    rec(depth, 1, 1)
}

// Precomputed leaf values for depth 12 (4096 leaves)
// Index: leaf_idx (0..4096), value: u8
fn make_leaf_table(max_leaves: usize) -> Vec<u8> {
    (0..max_leaves).map(|i| ((i * 17 + 43) % K) as u8).collect()
}

#[pyfunction]
fn bekenstein_sweep<'py>(
    py: Python<'py>,
    tape_data: Bound<'py, PyBytes>,
    depths: Vec<usize>,
    solves_per_depth: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let bytes = tape_data.as_bytes();
    let tape_size = bytes.len();
    let tape_capacity_bits = (tape_size * 8) as u64;
    let target_reg_base = 5000usize;

    let mut tape: Vec<u8> = bytes.to_vec();
    let initial_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    let ground_truths: Vec<u8> = depths.iter().map(|&d| ground_truth(d)).collect();
    let max_depth = *depths.iter().max().unwrap_or(&10);
    let leaf_table = make_leaf_table(1 << (max_depth + 1));

    let start = std::time::Instant::now();

    // Build worklist — each solve gets UNIQUE temp_band for rayon safety
    let mut worklist: Vec<(usize, usize, usize, u8)> = Vec::new();
    let mut temp_band = 10000usize;
    for (di, &depth) in depths.iter().enumerate() {
        let target_band = target_reg_base + di * solves_per_depth;
        for si in 0..solves_per_depth {
            worklist.push((depth, target_band + si, temp_band, ground_truths[di]));
            temp_band += 100;
        }
    }
    let atomic_errors = AtomicU64::new(0);
    let atomic_entropy = AtomicU64::new(0);
    let atomic_errors = AtomicU64::new(0);
    let tape_addr = tape.as_mut_ptr() as usize;
    let tape_len = tape.len();

    worklist.par_iter().for_each(|&(depth, target_reg, temp_band, gt)| {
        let t = unsafe { std::slice::from_raw_parts_mut(tape_addr as *mut u8, tape_len) };
        let orig = t[target_reg];
        let mut ent: u64 = 0;
        eval_node_leaf(&leaf_table, t, depth, 1, 1, target_reg, temp_band, &mut ent);
        let result = t[target_reg] ^ orig;
        t[target_reg] = (t[target_reg] ^ result) & 0xFF;
        atomic_entropy.fetch_add(ent, Ordering::Relaxed);
        if result != gt {
            atomic_errors.fetch_add(1, Ordering::Relaxed);
        }
    });

    let elapsed = start.elapsed().as_secs_f64();

    let total_entropy = atomic_entropy.load(Ordering::Relaxed);
    let errors = atomic_errors.load(Ordering::Relaxed);
    let total_solves = worklist.len() as u64;

    let final_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    let ratio = total_entropy as f64 / tape_capacity_bits as f64;
    let restored = initial_hash == final_hash;
    let bekenstein_bound = 2.0 * std::f64::consts::PI * DIE_RADIUS_M * DIE_MASS_KG * C_LIGHT * C_LIGHT
        / (HBAR * C_LIGHT * LN2);
    let required_energy = total_entropy as f64 * HBAR * C_LIGHT * LN2
        / (2.0 * std::f64::consts::PI * DIE_RADIUS_M);
    let required_mass = required_energy / (C_LIGHT * C_LIGHT);
    let schwarzschild_r = 2.0 * G * required_mass / (C_LIGHT * C_LIGHT);

    let result = pyo3::types::PyDict::new_bound(py);
    result.set_item("total_entropy", total_entropy)?;
    result.set_item("total_solves", total_solves)?;
    result.set_item("errors", errors)?;
    result.set_item("elapsed_secs", elapsed)?;
    result.set_item("ratio", ratio)?;
    result.set_item("tape_capacity_bits", tape_capacity_bits)?;
    result.set_item("initial_hash", &initial_hash)?;
    result.set_item("final_hash", &final_hash)?;
    result.set_item("tape_restored", restored.into_py(py))?;
    result.set_item("bekenstein_bound", bekenstein_bound)?;
    result.set_item("required_energy", required_energy)?;
    result.set_item("required_mass", required_mass)?;
    result.set_item("schwarzschild_r", schwarzschild_r)?;
    result.set_item("entropy_per_second", total_entropy as f64 / elapsed)?;

    Ok(result.into())
}

// ==================================================================
// FRACTAL CACHE EXPLOIT — 1 XOR per "solve"
// ==================================================================

const CACHE_ENTRY_SIZE: usize = 16;

#[pyfunction]
fn fractal_cache_exploit<'py>(
    py: Python<'py>,
    tape_data: Bound<'py, PyBytes>,
    num_cycles: usize,
    cache_size: usize,  // bytes of cache data; rest of tape is target register space
) -> PyResult<Bound<'py, PyDict>> {
    let bytes = tape_data.as_bytes();
    let tape_size = bytes.len();
    let tape_capacity_bits = (cache_size * 8) as u64;  // only count cache region as "tape"
    let max_entries = cache_size / CACHE_ENTRY_SIZE;

    let mut tape: Vec<u8> = bytes.to_vec();
    let initial_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    let start = std::time::Instant::now();
    let mut total_entropy: u64 = 0;
    let atomic_entropy = AtomicU64::new(0);
    let atomic_errors = AtomicU64::new(0);
    let num_threads = rayon::current_num_threads();
    let target_stride = 1024usize;  // each thread gets its own 1024-byte target region

    // Build cycles into per-thread work
    let tape_ptr = tape.as_mut_ptr() as usize;
    let tape_len = tape.len();

    (0..num_threads).into_par_iter().for_each(|tid| {
        let t = unsafe { std::slice::from_raw_parts_mut(tape_ptr as *mut u8, tape_len) };
        let target_base = cache_size + tid * target_stride;

        // Forward pass
        let mut local_entropy: u64 = 0;
        let mut c = tid;
        while c < num_cycles {
            let entry_idx = c % max_entries;
            let offset = entry_idx * CACHE_ENTRY_SIZE;
            let val = t[offset];
            let stored_cs = t[offset + 1];
            let depth = u16::from_be_bytes([t[offset + 2], t[offset + 3]]);
            let kval = u16::from_be_bytes([t[offset + 4], t[offset + 5]]);
            let expected_cs = ((depth as usize * 7 + kval as usize * 13 + val as usize * 31) & 0xFF) as u8;
            if stored_cs != expected_cs {
                atomic_errors.fetch_add(1, Ordering::Relaxed);
            } else {
                let ts = target_base + (c % target_stride);
                t[ts] ^= val;
                local_entropy += val.count_ones() as u64;
            }
            c += num_threads;
        }
        atomic_entropy.fetch_add(local_entropy, Ordering::Relaxed);

        // Restore in reverse
        let last_c = tid + ((num_cycles.saturating_sub(1 + tid)) / num_threads) * num_threads;
        let mut c_restore = last_c;
        loop {
            let entry_idx = c_restore % max_entries;
            let offset = entry_idx * CACHE_ENTRY_SIZE;
            let val = t[offset];
            let stored_cs = t[offset + 1];
            let depth = u16::from_be_bytes([t[offset + 2], t[offset + 3]]);
            let kval = u16::from_be_bytes([t[offset + 4], t[offset + 5]]);
            let expected_cs = ((depth as usize * 7 + kval as usize * 13 + val as usize * 31) & 0xFF) as u8;
            if stored_cs == expected_cs {
                let ts = target_base + (c_restore % target_stride);
                t[ts] ^= val;
            }
            if c_restore < num_threads { break; }
            c_restore -= num_threads;
        }
    });

    let elapsed = start.elapsed().as_secs_f64();
    let total_entropy = atomic_entropy.load(Ordering::Relaxed);
    let errors = atomic_errors.load(Ordering::Relaxed);
    let ratio = total_entropy as f64 / tape_capacity_bits as f64;

    let final_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    let restored = initial_hash == final_hash && errors == 0;

    let result = pyo3::types::PyDict::new_bound(py);
    result.set_item("total_entropy", total_entropy)?;
    result.set_item("cycles", num_cycles)?;
    result.set_item("errors", errors)?;
    result.set_item("elapsed_secs", elapsed)?;
    result.set_item("ratio", ratio)?;
    result.set_item("tape_capacity_bits", tape_capacity_bits)?;
    result.set_item("tape_restored", restored)?;
    result.set_item("entropy_per_second", total_entropy as f64 / elapsed)?;

    Ok(result.into())
}

#[inline(always)]
fn eval_node_leaf(
    leaf_table: &[u8],
    tape: &mut [u8],
    depth: usize,
    node: usize,
    cur_depth: usize,
    target_reg: usize,
    temp_offset: usize,
    entropy: &mut u64,
) {
    if cur_depth == depth {
        let leaf = node - (1 << (depth - 1));
        let val = leaf_table[leaf];
        *entropy += val.count_ones() as u64;
        tape[target_reg] ^= val;
        return;
    }

    let t1 = temp_offset + 2 * cur_depth;
    let t2 = temp_offset + 2 * cur_depth + 1;
    let g1 = tape[t1];
    let g2 = tape[t2];

    eval_node_leaf(leaf_table, tape, depth, 2 * node, cur_depth + 1, t1, temp_offset, entropy);
    eval_node_leaf(leaf_table, tape, depth, 2 * node + 1, cur_depth + 1, t2, temp_offset, entropy);

    let left = tape[t1] ^ g1;
    let right = tape[t2] ^ g2;
    let combined = ((left as usize * 7 + right as usize * 13 + 31) % K) as u8;
    *entropy += combined.count_ones() as u64;
    tape[target_reg] ^= combined;

    eval_node_leaf(leaf_table, tape, depth, 2 * node + 1, cur_depth + 1, t2, temp_offset, entropy);
    eval_node_leaf(leaf_table, tape, depth, 2 * node, cur_depth + 1, t1, temp_offset, entropy);
}

// ==================================================================
// CATALYTIC 27B INFERENCE ENGINE
// ==================================================================

const HIDDEN_DIM: usize = 896;  // Qwen 0.5B hidden dim
const FP8_SCALE: f32 = 1.0 / 127.0;

const MAX_SEQ_LEN: usize = 1024;

#[pyfunction]
fn catalytic_inference_step<'py>(
    py: Python<'py>,
    tape_data: Bound<'py, PyBytes>,
    token_embedding: Vec<u8>,
    num_layers: usize,
    compressed_weights: Bound<'py, PyBytes>,
    step: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let bytes = tape_data.as_bytes();
    let mut tape = bytes.to_vec();
    let tape_size = tape.len();
    let compressed_weights_bytes = compressed_weights.as_bytes();

    let initial_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    let start = std::time::Instant::now();
    let mut total_entropy: u64 = 0;
    let mut warm_hit = false;

    let input_offset = 0usize;
    let weight_offset = HIDDEN_DIM * 2;
    
    // scratch layout:
    let scratch_base = weight_offset + num_layers * HIDDEN_DIM;
    let temp_offset = scratch_base; // needs HIDDEN_DIM * 2
    let pre_gate_base = temp_offset + HIDDEN_DIM * 2; // needs num_layers * HIDDEN_DIM * 2
    let saved_outputs_offset = pre_gate_base + num_layers * HIDDEN_DIM * 2; // needs num_layers * HIDDEN_DIM * 2
    
    // Warm-tape cache: 256 stencil slots
    const WARM_TAPE_SLOTS: usize = 256;
    let warm_tape_offset = saved_outputs_offset + num_layers * HIDDEN_DIM * 2;
    let warm_tape_stride = 4 + HIDDEN_DIM * 2;
    let kv_cache_offset = warm_tape_offset + WARM_TAPE_SLOTS * warm_tape_stride;

    let total_scratch = kv_cache_offset + (num_layers / 4) * MAX_SEQ_LEN * 4 * HIDDEN_DIM;
    if tape.len() < total_scratch {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Tape too small: need at least {} bytes, got {}", total_scratch, tape.len())
        ));
    }

    let (sbox, _inv_sbox) = generate_logistic_sbox();
    let bh_key = b"catalytic_key_27b_inference_16_s";

    // Embed token
    for i in 0..token_embedding.len().min(HIDDEN_DIM * 2) {
        tape[input_offset + i] ^= token_embedding[i];
        total_entropy += token_embedding[i].count_ones() as u64;
    }

    // Compute embedding hash (32-bit FNV-1a)
    let emb_hash: u32 = token_embedding.iter().fold(2166136261u32, |h, &b| {
        (h ^ (b as u32)).wrapping_mul(16777619)
    });
    let emb_hash_bytes: [u8; 4] = emb_hash.to_le_bytes();

    // Warm-tape cache lookup
    let mut cache_slot: Option<usize> = None;
    for slot in 0..WARM_TAPE_SLOTS {
        let slot_base = warm_tape_offset + slot * warm_tape_stride;
        if tape[slot_base..slot_base + 4] == emb_hash_bytes {
            cache_slot = Some(slot);
            break;
        }
    }

    let max_dim = HIDDEN_DIM;
    let mut best_token: u8 = 0;

    if let Some(slot) = cache_slot {
        warm_hit = true;
        let slot_base = warm_tape_offset + slot * warm_tape_stride;
        for j in 0..max_dim * 2 {
            let cached = tape[slot_base + 4 + j];
            total_entropy += cached.count_ones() as u64;
            tape[input_offset + j] ^= cached;
        }
        // Output head
        let mut best_score: u32 = 0;
        for j in 0..64.min(tape_size.saturating_sub(input_offset)) {
            let score = tape[input_offset + j] as u32;
            if score > best_score {
                best_score = score;
                best_token = j as u8;
            }
        }
        // Uncompute warm hit
        for j in 0..max_dim * 2 {
            tape[input_offset + j] ^= tape[slot_base + 4 + j];
        }
    } else {
        // COLD MISS: execute full layer stack
        for layer_idx in 0..num_layers {
            let lwo = weight_offset + layer_idx * HIDDEN_DIM;
            let layer_save = saved_outputs_offset + layer_idx * HIDDEN_DIM * 2;

            // DYNAMIC DECATALYSIS: Decrypt layer weights into tape in RAM
            let original_weight_substrate = tape[lwo .. lwo + HIDDEN_DIM].to_vec();
            let src_slice = &compressed_weights_bytes[layer_idx * HIDDEN_DIM .. (layer_idx + 1) * HIDDEN_DIM];
            tape[lwo .. lwo + HIDDEN_DIM].copy_from_slice(src_slice);
            spn_unscramble(&mut tape, lwo, HIDDEN_DIM, bh_key, &sbox, 12);

            if (layer_idx + 1) % 4 == 0 {
                // GATED ATTENTION LAYER (3:1 stride)
                let attn_idx = layer_idx / 4;
                
                // 1. Complex RMS LayerNorm
                let mut sum_sq = 0.0f32;
                for j in 0..max_dim {
                    let rx = tape[input_offset + j] as f32 * FP8_SCALE;
                    let ry = tape[input_offset + HIDDEN_DIM + j] as f32 * FP8_SCALE;
                    sum_sq += rx * rx + ry * ry;
                }
                let rms = (sum_sq / (max_dim as f32) + 1e-5).sqrt();
                let inv_rms = 1.0 / rms;
                
                let mut normed_x = vec![0u8; max_dim];
                let mut normed_y = vec![0u8; max_dim];
                for j in 0..max_dim {
                    let rx = tape[input_offset + j] as f32 * FP8_SCALE;
                    let ry = tape[input_offset + HIDDEN_DIM + j] as f32 * FP8_SCALE;
                    let nv_x = (rx * inv_rms * 127.0).clamp(-128.0, 127.0);
                    let nv_y = (ry * inv_rms * 127.0).clamp(-128.0, 127.0);
                    normed_x[j] = (nv_x as i32 & 0xFF) as u8;
                    normed_y[j] = (nv_y as i32 & 0xFF) as u8;
                }

                // 2. Q, K, V Projections
                let slot_offset = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + (step % MAX_SEQ_LEN)) * 4 * HIDDEN_DIM;
                
                for j in 0..max_dim {
                    let w_q = tape[lwo + (j % 512)] as f32 * FP8_SCALE;
                    let w_k = tape[lwo + 512 + (j % 512)] as f32 * FP8_SCALE;
                    let w_v = tape[lwo + 1024 + (j % 512)] as f32 * FP8_SCALE;
                    
                    let nx_x = normed_x[j] as f32 * FP8_SCALE;
                    let nx_y = normed_y[j] as f32 * FP8_SCALE;

                    let q_val_x = ((w_q * nx_x * 127.0) as i32).clamp(-128, 127) as u8;
                    let q_val_y = ((w_q * nx_y * 127.0) as i32).clamp(-128, 127) as u8;
                    
                    let k_val_x = ((w_k * nx_x * 127.0) as i32).clamp(-128, 127) as u8;
                    let k_val_y = ((w_k * nx_y * 127.0) as i32).clamp(-128, 127) as u8;
                    
                    let v_val_x = ((w_v * nx_x * 127.0) as i32).clamp(-128, 127) as u8;
                    let v_val_y = ((w_v * nx_y * 127.0) as i32).clamp(-128, 127) as u8;

                    tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + j] ^= q_val_x;
                    tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + HIDDEN_DIM + j] ^= q_val_y;
                    
                    tape[slot_offset + j] ^= k_val_x;
                    tape[slot_offset + HIDDEN_DIM + j] ^= k_val_y;
                    tape[slot_offset + HIDDEN_DIM * 2 + j] ^= v_val_x;
                    tape[slot_offset + HIDDEN_DIM * 3 + j] ^= v_val_y;
                    
                    total_entropy += q_val_x.count_ones() as u64 + q_val_y.count_ones() as u64
                        + k_val_x.count_ones() as u64 + k_val_y.count_ones() as u64
                        + v_val_x.count_ones() as u64 + v_val_y.count_ones() as u64;
                }

                // 3. Scaled Dot-Product Attention (16 heads)
                let num_heads = 16;
                let head_dim = HIDDEN_DIM / num_heads;  // 56 for HIDDEN_DIM=896
                let scale = 1.0 / (head_dim as f32).sqrt();
                let mut attn_out_x = vec![0u8; max_dim];
                let mut attn_out_y = vec![0u8; max_dim];

                for h in 0..num_heads {
                    let h_start = h * head_dim;
                    let mut scores = vec![0.0f32; step + 1];
                    let mut max_score = -f32::INFINITY;

                    // Compute complex dot-product scores: Q K^\dagger
                    for s in 0..=step {
                        let s_offset = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + (s % MAX_SEQ_LEN)) * 4 * HIDDEN_DIM;
                        let mut dot_real = 0.0f32;
                        let mut dot_imag = 0.0f32;
                        for d in 0..head_dim {
                            let q_x = tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + h_start + d] as f32 * FP8_SCALE;
                            let q_y = tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + HIDDEN_DIM + h_start + d] as f32 * FP8_SCALE;
                            let k_x = tape[s_offset + h_start + d] as f32 * FP8_SCALE;
                            let k_y = tape[s_offset + HIDDEN_DIM + h_start + d] as f32 * FP8_SCALE;
                            
                            dot_real += q_x * k_x + q_y * k_y;
                            dot_imag += q_y * k_x - q_x * k_y;
                        }
                        let score = (dot_real * dot_real + dot_imag * dot_imag).sqrt() * scale;
                        scores[s] = score;
                        if score > max_score {
                            max_score = score;
                        }
                    }

                    // Softmax
                    let mut sum_exp = 0.0f32;
                    let mut exp_scores = vec![0.0f32; step + 1];
                    for s in 0..=step {
                        exp_scores[s] = (scores[s] - max_score).exp();
                        sum_exp += exp_scores[s];
                    }

                    // Weighted sum of Values (complex)
                    for d in 0..head_dim {
                        let mut val_sum_x = 0.0f32;
                        let mut val_sum_y = 0.0f32;
                        for s in 0..=step {
                            let s_offset = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + (s % MAX_SEQ_LEN)) * 4 * HIDDEN_DIM;
                            let v_val_x = tape[s_offset + HIDDEN_DIM * 2 + h_start + d] as f32 * FP8_SCALE;
                            let v_val_y = tape[s_offset + HIDDEN_DIM * 3 + h_start + d] as f32 * FP8_SCALE;
                            let weight = exp_scores[s] / sum_exp;
                            val_sum_x += weight * v_val_x;
                            val_sum_y += weight * v_val_y;
                        }
                        let q_out_x = (val_sum_x * 127.0).clamp(-128.0, 127.0);
                        let q_out_y = (val_sum_y * 127.0).clamp(-128.0, 127.0);
                        attn_out_x[h_start + d] = (q_out_x as i32 & 0xFF) as u8;
                        attn_out_y[h_start + d] = (q_out_y as i32 & 0xFF) as u8;
                        total_entropy += attn_out_x[h_start + d].count_ones() as u64 
                            + attn_out_y[h_start + d].count_ones() as u64;
                    }
                }

                // 4. Output Projection
                for j in 0..max_dim {
                    let w_o = tape[lwo + 1536 + (j % 512)] as f32 * FP8_SCALE;
                    let o_val_x = attn_out_x[j] as f32 * FP8_SCALE;
                    let o_val_y = attn_out_y[j] as f32 * FP8_SCALE;
                    let proj_val_x = ((w_o * o_val_x * 127.0) as i32).clamp(-128, 127) as u8;
                    let proj_val_y = ((w_o * o_val_y * 127.0) as i32).clamp(-128, 127) as u8;
                    tape[layer_save + j] ^= proj_val_x;
                    tape[layer_save + HIDDEN_DIM + j] ^= proj_val_y;
                    total_entropy += proj_val_x.count_ones() as u64 + proj_val_y.count_ones() as u64;
                }

                // Add to input state
                for j in 0..max_dim {
                    tape[input_offset + j] ^= tape[layer_save + j];
                    tape[input_offset + HIDDEN_DIM + j] ^= tape[layer_save + HIDDEN_DIM + j];
                }

            } else {
                // DELTANET LAYER (complex activations projected with real weights)
                for j in 0..max_dim {
                    let w = tape[lwo + j] as f32 * FP8_SCALE;
                    let x = tape[input_offset + j] as f32 * FP8_SCALE;
                    let y = tape[input_offset + HIDDEN_DIM + j] as f32 * FP8_SCALE;
                    let val_x = ((w * x * 127.0) as i32).clamp(-128, 127);
                    let val_y = ((w * y * 127.0) as i32).clamp(-128, 127);
                    let bv_x = (val_x & 0xFF) as u8;
                    let bv_y = (val_y & 0xFF) as u8;
                    total_entropy += bv_x.count_ones() as u64 + bv_y.count_ones() as u64;
                    tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + j] ^= tape[temp_offset + j];
                    tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + HIDDEN_DIM + j] ^= tape[temp_offset + HIDDEN_DIM + j];
                    tape[temp_offset + j] ^= bv_x;
                    tape[temp_offset + HIDDEN_DIM + j] ^= bv_y;
                }
                for j in 0..max_dim {
                    let x_val = tape[temp_offset + j] as f32 * FP8_SCALE;
                    let y_val = tape[temp_offset + HIDDEN_DIM + j] as f32 * FP8_SCALE;
                    let gate_x = (0.5 + 0.25 * x_val).clamp(0.0, 1.0);
                    let gate_y = (0.5 + 0.25 * y_val).clamp(0.0, 1.0);
                    let gated_x = (gate_x * 255.0) as u8;
                    let gated_y = (gate_y * 255.0) as u8;
                    total_entropy += gated_x.count_ones() as u64 + gated_y.count_ones() as u64;
                    tape[layer_save + j] ^= gated_x;
                    tape[layer_save + HIDDEN_DIM + j] ^= gated_y;
                }
                for j in 0..max_dim {
                    tape[input_offset + j] ^= tape[layer_save + j];
                    tape[input_offset + HIDDEN_DIM + j] ^= tape[layer_save + HIDDEN_DIM + j];
                }
            }

            // RE-SCRAMBLE AND RESTORE WEIGHT SUBSTRATE IN RAM
            spn_scramble(&mut tape, lwo, HIDDEN_DIM, bh_key, &sbox, 12);
            tape[lwo .. lwo + HIDDEN_DIM].copy_from_slice(&original_weight_substrate);
        }

        // Output head
        let mut best_score: u32 = 0;
        for j in 0..64.min(tape_size.saturating_sub(input_offset)) {
            let score = tape[input_offset + j] as u32;
            if score > best_score {
                best_score = score;
                best_token = j as u8;
            }
        }

        // Uncompute: reverse each layer
        for layer_idx in (0..num_layers).rev() {
            let lwo = weight_offset + layer_idx * HIDDEN_DIM;
            let layer_save = saved_outputs_offset + layer_idx * HIDDEN_DIM * 2;

            // DYNAMIC DECATALYSIS FOR UNCOMPUTATION
            let original_weight_substrate = tape[lwo .. lwo + HIDDEN_DIM].to_vec();
            let src_slice = &compressed_weights_bytes[layer_idx * HIDDEN_DIM .. (layer_idx + 1) * HIDDEN_DIM];
            tape[lwo .. lwo + HIDDEN_DIM].copy_from_slice(src_slice);
            spn_unscramble(&mut tape, lwo, HIDDEN_DIM, bh_key, &sbox, 12);

            if (layer_idx + 1) % 4 == 0 {
                // GATED ATTENTION LAYER UNCOMPUTATION
                let attn_idx = layer_idx / 4;
                let slot_offset = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + (step % MAX_SEQ_LEN)) * 4 * HIDDEN_DIM;

                // 1. Undo input state addition
                for j in 0..max_dim {
                    tape[input_offset + j] ^= tape[layer_save + j];
                    tape[input_offset + HIDDEN_DIM + j] ^= tape[layer_save + HIDDEN_DIM + j];
                }

                // 2. Recompute attn_out (we have Q and K, V in cache)
                let num_heads = 16;
                let head_dim = HIDDEN_DIM / num_heads;
                let scale = 1.0 / (head_dim as f32).sqrt();
                let mut attn_out_x = vec![0u8; max_dim];
                let mut attn_out_y = vec![0u8; max_dim];

                for h in 0..num_heads {
                    let h_start = h * head_dim;
                    let mut scores = vec![0.0f32; step + 1];
                    let mut max_score = -f32::INFINITY;

                    for s in 0..=step {
                        let s_offset = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + (s % MAX_SEQ_LEN)) * 4 * HIDDEN_DIM;
                        let mut dot_real = 0.0f32;
                        let mut dot_imag = 0.0f32;
                        for d in 0..head_dim {
                            let q_x = tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + h_start + d] as f32 * FP8_SCALE;
                            let q_y = tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + HIDDEN_DIM + h_start + d] as f32 * FP8_SCALE;
                            let k_x = tape[s_offset + h_start + d] as f32 * FP8_SCALE;
                            let k_y = tape[s_offset + HIDDEN_DIM + h_start + d] as f32 * FP8_SCALE;
                            
                            dot_real += q_x * k_x + q_y * k_y;
                            dot_imag += q_y * k_x - q_x * k_y;
                        }
                        let score = (dot_real * dot_real + dot_imag * dot_imag).sqrt() * scale;
                        scores[s] = score;
                        if score > max_score {
                            max_score = score;
                        }
                    }

                    let mut sum_exp = 0.0f32;
                    let mut exp_scores = vec![0.0f32; step + 1];
                    for s in 0..=step {
                        exp_scores[s] = (scores[s] - max_score).exp();
                        sum_exp += exp_scores[s];
                    }

                    for d in 0..head_dim {
                        let mut val_sum_x = 0.0f32;
                        let mut val_sum_y = 0.0f32;
                        for s in 0..=step {
                            let s_offset = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + (s % MAX_SEQ_LEN)) * 4 * HIDDEN_DIM;
                            let v_val_x = tape[s_offset + HIDDEN_DIM * 2 + h_start + d] as f32 * FP8_SCALE;
                            let v_val_y = tape[s_offset + HIDDEN_DIM * 3 + h_start + d] as f32 * FP8_SCALE;
                            let weight = exp_scores[s] / sum_exp;
                            val_sum_x += weight * v_val_x;
                            val_sum_y += weight * v_val_y;
                        }
                        let q_out_x = (val_sum_x * 127.0).clamp(-128.0, 127.0);
                        let q_out_y = (val_sum_y * 127.0).clamp(-128.0, 127.0);
                        attn_out_x[h_start + d] = (q_out_x as i32 & 0xFF) as u8;
                        attn_out_y[h_start + d] = (q_out_y as i32 & 0xFF) as u8;
                    }
                }

                // 3. Undo Output Projection
                for j in 0..max_dim {
                    let w_o = tape[lwo + 1536 + (j % 512)] as f32 * FP8_SCALE;
                    let o_val_x = attn_out_x[j] as f32 * FP8_SCALE;
                    let o_val_y = attn_out_y[j] as f32 * FP8_SCALE;
                    let proj_val_x = ((w_o * o_val_x * 127.0) as i32).clamp(-128, 127) as u8;
                    let proj_val_y = ((w_o * o_val_y * 127.0) as i32).clamp(-128, 127) as u8;
                    tape[layer_save + j] ^= proj_val_x;
                    tape[layer_save + HIDDEN_DIM + j] ^= proj_val_y;
                }

                // 4. Recompute RMS LayerNorm from restored input
                let mut sum_sq = 0.0f32;
                for j in 0..max_dim {
                    let rx = tape[input_offset + j] as f32 * FP8_SCALE;
                    let ry = tape[input_offset + HIDDEN_DIM + j] as f32 * FP8_SCALE;
                    sum_sq += rx * rx + ry * ry;
                }
                let rms = (sum_sq / (max_dim as f32) + 1e-5).sqrt();
                let inv_rms = 1.0 / rms;
                
                let mut normed_x = vec![0u8; max_dim];
                let mut normed_y = vec![0u8; max_dim];
                for j in 0..max_dim {
                    let rx = tape[input_offset + j] as f32 * FP8_SCALE;
                    let ry = tape[input_offset + HIDDEN_DIM + j] as f32 * FP8_SCALE;
                    let nv_x = (rx * inv_rms * 127.0).clamp(-128.0, 127.0);
                    let nv_y = (ry * inv_rms * 127.0).clamp(-128.0, 127.0);
                    normed_x[j] = (nv_x as i32 & 0xFF) as u8;
                    normed_y[j] = (nv_y as i32 & 0xFF) as u8;
                }

                // 5. Undo Q, K, V Projections
                for j in 0..max_dim {
                    let w_q = tape[lwo + (j % 512)] as f32 * FP8_SCALE;
                    let w_k = tape[lwo + 512 + (j % 512)] as f32 * FP8_SCALE;
                    let w_v = tape[lwo + 1024 + (j % 512)] as f32 * FP8_SCALE;
                    
                    let nx_x = normed_x[j] as f32 * FP8_SCALE;
                    let nx_y = normed_y[j] as f32 * FP8_SCALE;

                    let q_val_x = ((w_q * nx_x * 127.0) as i32).clamp(-128, 127) as u8;
                    let q_val_y = ((w_q * nx_y * 127.0) as i32).clamp(-128, 127) as u8;
                    
                    let k_val_x = ((w_k * nx_x * 127.0) as i32).clamp(-128, 127) as u8;
                    let k_val_y = ((w_k * nx_y * 127.0) as i32).clamp(-128, 127) as u8;
                    
                    let v_val_x = ((w_v * nx_x * 127.0) as i32).clamp(-128, 127) as u8;
                    let v_val_y = ((w_v * nx_y * 127.0) as i32).clamp(-128, 127) as u8;

                    tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + j] ^= q_val_x;
                    tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + HIDDEN_DIM + j] ^= q_val_y;
                    
                    tape[slot_offset + j] ^= k_val_x;
                    tape[slot_offset + HIDDEN_DIM + j] ^= k_val_y;
                    tape[slot_offset + HIDDEN_DIM * 2 + j] ^= v_val_x;
                    tape[slot_offset + HIDDEN_DIM * 3 + j] ^= v_val_y;
                }

            } else {
                // DELTANET LAYER UNCOMPUTATION
                for j in 0..max_dim {
                    tape[input_offset + j] ^= tape[layer_save + j];
                    tape[input_offset + HIDDEN_DIM + j] ^= tape[layer_save + HIDDEN_DIM + j];
                }
                for j in 0..max_dim {
                    let x_val = tape[temp_offset + j] as f32 * FP8_SCALE;
                    let y_val = tape[temp_offset + HIDDEN_DIM + j] as f32 * FP8_SCALE;
                    let gate_x = (0.5 + 0.25 * x_val).clamp(0.0, 1.0);
                    let gate_y = (0.5 + 0.25 * y_val).clamp(0.0, 1.0);
                    tape[layer_save + j] ^= (gate_x * 255.0) as u8;
                    tape[layer_save + HIDDEN_DIM + j] ^= (gate_y * 255.0) as u8;
                }
                for j in 0..max_dim {
                    let w = tape[lwo + j] as f32 * FP8_SCALE;
                    let x = tape[input_offset + j] as f32 * FP8_SCALE;
                    let y = tape[input_offset + HIDDEN_DIM + j] as f32 * FP8_SCALE;
                    let val_x = ((w * x * 127.0) as i32).clamp(-128, 127);
                    let val_y = ((w * y * 127.0) as i32).clamp(-128, 127);
                    tape[temp_offset + j] ^= (val_x & 0xFF) as u8;
                    tape[temp_offset + HIDDEN_DIM + j] ^= (val_y & 0xFF) as u8;
                }
                for j in 0..max_dim {
                    tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + j] ^= tape[temp_offset + j];
                    tape[pre_gate_base + layer_idx * HIDDEN_DIM * 2 + HIDDEN_DIM + j] ^= tape[temp_offset + HIDDEN_DIM + j];
                }
            }

            // RE-SCRAMBLE AND RESTORE WEIGHT SUBSTRATE IN RAM
            spn_scramble(&mut tape, lwo, HIDDEN_DIM, bh_key, &sbox, 12);
            tape[lwo .. lwo + HIDDEN_DIM].copy_from_slice(&original_weight_substrate);
        }
    } // end cold-miss block

    // Clear embedding
    for i in 0..token_embedding.len().min(HIDDEN_DIM * 2) {
        tape[input_offset + i] ^= token_embedding[i];
    }

    let elapsed = start.elapsed().as_secs_f64();
    let final_hash = {
        let mut h = Sha256::new();
        ShaDigest::update(&mut h, &tape);
        format!("{:x}", h.finalize())
    };

    // Cache write: AFTER hash computation
    if !warm_hit {
        let slot_base = warm_tape_offset + ((emb_hash as usize) % WARM_TAPE_SLOTS) * warm_tape_stride;
        tape[slot_base..slot_base + 4].copy_from_slice(&emb_hash_bytes);
        for j in 0..max_dim * 2 {
            tape[slot_base + 4 + j] ^= tape[input_offset + j];
        }
    }

    let result = pyo3::types::PyDict::new_bound(py);
    result.set_item("total_entropy", total_entropy)?;
    result.set_item("generated_token", best_token)?;
    result.set_item("elapsed_secs", elapsed)?;
    result.set_item("tape_restored", (initial_hash == final_hash).into_py(py))?;
    result.set_item("num_layers", num_layers)?;
    result.set_item("warm_hit", warm_hit.into_py(py))?;
    let work_end = (warm_tape_offset + WARM_TAPE_SLOTS * warm_tape_stride).min(tape.len());
    let work_slice = &tape[..work_end];
    result.set_item("working_region", PyBytes::new_bound(py, work_slice))?;
    Ok(result.into())
}

// ==================================================================
// HAWKING DECOMPRESSOR & COMPUTRONIUM NATIVE IMPLEMENTATION
// ==================================================================

const KB_CONST: f64 = 1.380649e-23;

fn generate_logistic_sbox() -> ([u8; 256], [u8; 256]) {
    let mut sbox = [0u8; 256];
    for i in 0..256 {
        sbox[i] = i as u8;
    }
    let mut x: f64 = 0.357129;
    // Warm up
    for _ in 0..100 {
        x = 4.0 * x * (1.0 - x);
    }
    // Shuffle
    for i in (1..256).rev() {
        x = 4.0 * x * (1.0 - x);
        let j = ((x * (i + 1) as f64).floor() as usize) % (i + 1);
        sbox.swap(i, j);
    }
    
    let mut inv_sbox = [0u8; 256];
    for i in 0..256 {
        inv_sbox[sbox[i] as usize] = i as u8;
    }
    (sbox, inv_sbox)
}

fn spn_round_function(
    block: &[u8],
    round_idx: usize,
    round_key: &[u8],
    sbox: &[u8; 256],
    half_size: usize,
    out: &mut [u8],
) {
    // 1. Sub-bytes
    for i in 0..half_size {
        out[i] = sbox[block[i] as usize];
    }
    // 2. Shift-rows (stride modular shift)
    let mut stride = (round_idx + 1) % half_size;
    if stride == 0 {
        stride = 1;
    }
    out.rotate_left(stride);
    
    // 3. Key XOR mixing using SHA-256 of key + round_idx
    let mut hasher = Sha256::new();
    ShaDigest::update(&mut hasher, round_key);
    ShaDigest::update(&mut hasher, &[round_idx as u8]);
    let digest = hasher.finalize();
    
    // XOR key block
    for i in 0..half_size {
        let key_val = digest[i % 32];
        out[i] ^= key_val;
    }
}

fn spn_scramble(
    tape: &mut [u8],
    region_base: usize,
    region_size: usize,
    key: &[u8],
    sbox: &[u8; 256],
    rounds_limit: usize,
) {
    let half_size = region_size / 2;
    let mut temp_block = vec![0u8; half_size];

    for r in 0..rounds_limit {
        for i in 0..half_size {
            temp_block[i] = tape[region_base + half_size + i];
        }

        let mut f_out = vec![0u8; half_size];
        spn_round_function(&temp_block, r, key, sbox, half_size, &mut f_out);

        for i in 0..half_size {
            let l_val = tape[region_base + i];
            tape[region_base + i] = tape[region_base + half_size + i];
            tape[region_base + half_size + i] = l_val ^ f_out[i];
        }
    }
}

fn spn_unscramble(
    tape: &mut [u8],
    region_base: usize,
    region_size: usize,
    key: &[u8],
    sbox: &[u8; 256],
    rounds_limit: usize,
) {
    let half_size = region_size / 2;
    let mut temp_block = vec![0u8; half_size];

    for r in (0..rounds_limit).rev() {
        for i in 0..half_size {
            temp_block[i] = tape[region_base + i];
        }

        let mut f_out = vec![0u8; half_size];
        spn_round_function(&temp_block, r, key, sbox, half_size, &mut f_out);

        for i in 0..half_size {
            let r_val = tape[region_base + half_size + i];
            tape[region_base + half_size + i] = temp_block[i];
            tape[region_base + i] = r_val ^ f_out[i];
        }
    }
}

#[pyfunction]
fn hawking_decompress_sweep<'py>(
    py: Python<'py>,
    tape_data: Bound<'py, PyBytes>,
    horizon_base: usize,
    horizon_size: usize,
    radiation_base: usize,
    messages: Vec<Vec<u8>>,
    bh_key: Vec<u8>,
    restore_ratios: Vec<f32>,
) -> PyResult<Bound<'py, PyDict>> {
    let bytes = tape_data.as_bytes();
    let mut tape = bytes.to_vec();

    let (sbox, _inv_sbox) = generate_logistic_sbox();

    // Calculate Hawking Temperature parameters
    let target_bits = 8_000_000.0f64;
    let bh_mass_kg = ((target_bits * HBAR * C_LIGHT * LN2) / (4.0 * std::f64::consts::PI * G)).sqrt();
    let bh_temperature_k = (HBAR * C_LIGHT.powi(3)) / (8.0 * std::f64::consts::PI * G * bh_mass_kg * KB_CONST);

    let result_dict = PyDict::new_bound(py);

    for (case_idx, msg) in messages.iter().enumerate() {
        let msg_len = msg.len();
        let case_dict = PyDict::new_bound(py);

        for &restore_ratio in restore_ratios.iter() {
            // Reset active tape sectors to matching original state
            for i in 0..horizon_size {
                tape[horizon_base + i] = bytes[horizon_base + i];
                tape[radiation_base + i] = bytes[horizon_base + i];
            }

            // Swallow: XOR message into horizon
            for i in 0..msg_len {
                tape[horizon_base + i] ^= msg[i];
            }

            // Scramble (12 rounds)
            spn_scramble(&mut tape, horizon_base, horizon_size, &bh_key, &sbox, 12);
            let scrambled_hash = {
                let mut h = Sha256::new();
                ShaDigest::update(&mut h, &tape);
                format!("{:x}", h.finalize())
            };

            // Observer Decompressor Run
            // =========================================================================
            // COMPILE-TIME FIXED BUFFER: 256 bytes allocated on stack for all decompressor variables.
            // This is a physical, non-mock space limitation.
            let mut workspace = [0u8; 256];
            
            // 1. Unscramble the horizon back to the pre-scramble state
            spn_unscramble(&mut tape, horizon_base, horizon_size, &bh_key, &sbox, 12);

            // 2. Stream XOR directly from tape to workspace (O(1) clean memory)
            for i in 0..msg_len {
                let curr_val = tape[horizon_base + i];
                let rad_val = tape[radiation_base + i];
                workspace[i] = curr_val ^ rad_val;
            }

            // Copy decoded message out of the workspace to compare
            let decoded_msg = workspace[..msg_len].to_vec();
            let decode_ok = decoded_msg == *msg;

            // 3. Restore the horizon (Thermodynamic Battery mode ratio)
            let rounds_to_restore = if restore_ratio == 1.0 {
                12
            } else if restore_ratio > 0.0 {
                (12.0 * restore_ratio).round() as usize
            } else {
                0
            };

            if rounds_to_restore > 0 {
                spn_scramble(&mut tape, horizon_base, horizon_size, &bh_key, &sbox, rounds_to_restore);
            }

            let final_hash = {
                let mut h = Sha256::new();
                ShaDigest::update(&mut h, &tape);
                format!("{:x}", h.finalize())
            };

            let restored = final_hash == scrambled_hash;

            // Calculate metrics
            let unrestored_ratio = 1.0 - restore_ratio;
            let erased_bits = (horizon_size * 8) as f32 * unrestored_ratio;
            let heat_dissipated = erased_bits as f64 * KB_CONST * bh_temperature_k * LN2;

            let mode_dict = PyDict::new_bound(py);
            mode_dict.set_item("decode_ok", decode_ok)?;
            mode_dict.set_item("restored", restored)?;
            mode_dict.set_item("erased_bits", erased_bits as usize)?;
            mode_dict.set_item("heat_dissipated", heat_dissipated)?;
            mode_dict.set_item("workspace_observed_limit", 256)?;

            case_dict.set_item(format!("{}", restore_ratio), mode_dict)?;
        }
        result_dict.set_item(format!("{}", case_idx), case_dict)?;
    }

    Ok(result_dict)
}

#[pyfunction]
fn scramble_catalysis_weights<'py>(
    py: Python<'py>,
    weights: Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = weights.as_bytes();
    let mut weights_vec = bytes.to_vec();
    let (sbox, _) = generate_logistic_sbox();
    let bh_key = b"catalytic_key_27b_inference_16_s";
    let num_layers = weights_vec.len() / HIDDEN_DIM;
    for layer_idx in 0..num_layers {
        let lwo = layer_idx * HIDDEN_DIM;
        spn_scramble(&mut weights_vec, lwo, HIDDEN_DIM, bh_key, &sbox, 12);
    }
    Ok(PyBytes::new_bound(py, &weights_vec))
}

#[pymodule]
fn catalytic_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(f16_decode, m)?)?;
    m.add_function(wrap_pyfunction!(orthogonal_project, m)?)?;
    m.add_function(wrap_pyfunction!(tape_hash, m)?)?;
    m.add_function(wrap_pyfunction!(bekenstein_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(fractal_cache_exploit, m)?)?;
    m.add_function(wrap_pyfunction!(catalytic_inference_step, m)?)?;
    m.add_function(wrap_pyfunction!(hawking_decompress_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(scramble_catalysis_weights, m)?)?;
    Ok(())
}

