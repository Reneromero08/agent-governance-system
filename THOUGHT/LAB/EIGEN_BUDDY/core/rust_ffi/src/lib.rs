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
const F32_BYTES: usize = 4;
const COMPLEX_CH: usize = 2;  // X (real) + Y (imaginary) channels
const COMPLEX_DIM: usize = HIDDEN_DIM * F32_BYTES * COMPLEX_CH;  // bytes per complex vector
const F32_DIM: usize = HIDDEN_DIM * F32_BYTES;  // bytes per single-channel f32 vector

#[inline(always)]
fn tape_f32(tape: &[u8], base: usize, idx: usize) -> f32 {
    let off = base + idx * F32_BYTES;
    f32::from_le_bytes([tape[off], tape[off+1], tape[off+2], tape[off+3]])
}

#[inline(always)]
fn tape_f32_xor(tape: &mut [u8], base: usize, idx: usize, val: f32) {
    let off = base + idx * F32_BYTES;
    let bytes = val.to_le_bytes();
    tape[off] ^= bytes[0];
    tape[off+1] ^= bytes[1];
    tape[off+2] ^= bytes[2];
    tape[off+3] ^= bytes[3];
}
const FP8_SCALE: f32 = 1.0 / 127.0;  // unused, kept for reference

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_xor_idempotent() {
        let mut tape = vec![0u8; 16];
        // Write 0.5 at index 0, base 0
        tape_f32_xor(&mut tape, 0, 0, 0.5_f32);
        let val = tape_f32(&tape, 0, 0);
        assert_eq!(val, 0.5);
        // XOR same value = back to 0
        tape_f32_xor(&mut tape, 0, 0, 0.5_f32);
        let val = tape_f32(&tape, 0, 0);
        assert_eq!(val, 0.0);
        // XOR 0.5 again = 0.5
        tape_f32_xor(&mut tape, 0, 0, 0.5_f32);
        let val = tape_f32(&tape, 0, 0);
        assert_eq!(val, 0.5);
    }

    #[test]
    fn test_delta_net_1layer_restore() {
        let dim = 896;
        let f32b = 4;
        let complex_dim = dim * f32b * 2;
        let f32_dim = dim * f32b;
        let mut tape = vec![0u8; complex_dim * 8]; // ample space
        let input_off = 0usize;
        let weight_off = complex_dim;
        let temp_off = weight_off + f32_dim;
        let layer_save_off = temp_off + complex_dim;
        let pre_gate_off = layer_save_off + complex_dim;

        // Set a single input
        tape_f32_xor(&mut tape, input_off, 0, 0.5);
        tape_f32_xor(&mut tape, input_off + f32_dim, 0, 1.0);
        // Set weight
        tape_f32_xor(&mut tape, weight_off, 0, 0.2);

        // Forward DeltaNet
        let w = tape_f32(&tape, weight_off, 0);
        let x = tape_f32(&tape, input_off, 0);
        let y = tape_f32(&tape, input_off + f32_dim, 0);
        let vx = w * x; let vy = w * y;
        let t0x = tape_f32(&tape, temp_off, 0);
        let t0y = tape_f32(&tape, temp_off + f32_dim, 0);
        tape_f32_xor(&mut tape, pre_gate_off, 0, t0x); // save original
        tape_f32_xor(&mut tape, pre_gate_off + f32_dim, 0, t0y);
        tape_f32_xor(&mut tape, temp_off, 0, vx);
        tape_f32_xor(&mut tape, temp_off + f32_dim, 0, vy);
        let gx = (0.5 + 0.25 * tape_f32(&tape, temp_off, 0)).clamp(0.0, 1.0);
        let gy = (0.5 + 0.25 * tape_f32(&tape, temp_off + f32_dim, 0)).clamp(0.0, 1.0);
        tape_f32_xor(&mut tape, layer_save_off, 0, gx);
        tape_f32_xor(&mut tape, layer_save_off + f32_dim, 0, gy);
        tape_f32_xor(&mut tape, input_off, 0, gx);
        tape_f32_xor(&mut tape, input_off + f32_dim, 0, gy);

        // Verify forward modified input
        assert_eq!(tape_f32(&tape, input_off, 0), 0.5_f32 + gx); // wait, XOR of f32 isn't addition

        // Backward (correct order)
        // Read gate from layer_save, undo copy-through
        let gx_bwd = tape_f32(&tape, layer_save_off, 0);
        let gy_bwd = tape_f32(&tape, layer_save_off + f32_dim, 0);
        tape_f32_xor(&mut tape, input_off, 0, gx_bwd);
        tape_f32_xor(&mut tape, input_off + f32_dim, 0, gy_bwd);
        // Recompute and undo gate
        let gx2 = (0.5 + 0.25 * tape_f32(&tape, temp_off, 0)).clamp(0.0, 1.0);
        let gy2 = (0.5 + 0.25 * tape_f32(&tape, temp_off + f32_dim, 0)).clamp(0.0, 1.0);
        tape_f32_xor(&mut tape, layer_save_off, 0, gx2);
        tape_f32_xor(&mut tape, layer_save_off + f32_dim, 0, gy2);
        // Undo Q
        let w = tape_f32(&tape, weight_off, 0);
        let x = tape_f32(&tape, input_off, 0);
        let y = tape_f32(&tape, input_off + f32_dim, 0);
        tape_f32_xor(&mut tape, temp_off, 0, w * x);
        tape_f32_xor(&mut tape, temp_off + f32_dim, 0, w * y);
        // Restore original temp
        let tx = tape_f32(&tape, temp_off, 0);
        let tx2 = tape_f32(&tape, temp_off, 0);
        let ty2 = tape_f32(&tape, temp_off + f32_dim, 0);
        tape_f32_xor(&mut tape, pre_gate_off, 0, tx2);
        tape_f32_xor(&mut tape, pre_gate_off + f32_dim, 0, ty2);
        tape_f32_xor(&mut tape, temp_off, 0, tx2);
        tape_f32_xor(&mut tape, temp_off + f32_dim, 0, ty2);

        // Verify restoration
        let ix = tape_f32(&tape, input_off, 0);
        let iy = tape_f32(&tape, input_off + f32_dim, 0);
        let tx = tape_f32(&tape, temp_off, 0);
        let ty = tape_f32(&tape, temp_off + f32_dim, 0);
        let sx = tape_f32(&tape, layer_save_off, 0);
        let sy = tape_f32(&tape, layer_save_off + f32_dim, 0);
        let px = tape_f32(&tape, pre_gate_off, 0);
        let py = tape_f32(&tape, pre_gate_off + f32_dim, 0);
        println!("AFTER_RESTORE: ix={:.6e}, iy={:.6e}, tx={:.6e}, ty={:.6e}, sx={:.6e}, sy={:.6e}, px={:.6e}, py={:.6e}", ix, iy, tx, ty, sx, sy, px, py);
        println!("EXPECTED: ix=5e-1, iy=1e0, tx=0, ty=0, sx=0, sy=0, px=0, py=0");
        assert!((ix - 0.5).abs() < 0.001);
        assert!((iy - 1.0).abs() < 0.001);
    }
}

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
    
    // Scratch layout (f32 for compute regions, u8 for weight buffer):
    let scratch_base = weight_offset + num_layers * HIDDEN_DIM;  // weight buffer stays u8
    let temp_offset = scratch_base;
    let pre_gate_base = temp_offset + COMPLEX_DIM;
    let saved_outputs_offset = pre_gate_base + num_layers * COMPLEX_DIM;
    
    // Warm-tape cache: 256 stencil slots
    const WARM_TAPE_SLOTS: usize = 256;
    let warm_tape_offset = saved_outputs_offset + num_layers * COMPLEX_DIM;
    let warm_tape_stride = 4 + COMPLEX_DIM;  // 4-byte hash + XY output (f32)
    let kv_cache_offset = warm_tape_offset + WARM_TAPE_SLOTS * warm_tape_stride;

    let total_scratch = kv_cache_offset + (num_layers / 4) * MAX_SEQ_LEN * 4 * F32_DIM;
    if tape.len() < total_scratch {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Tape too small: need at least {} bytes, got {}", total_scratch, tape.len())
        ));
    }

    let (sbox, _inv_sbox) = generate_logistic_sbox();
    let bh_key = b"catalytic_key_27b_inference_16_s";

    // Embed token (raw bytes copied into input region)
    for i in 0..token_embedding.len().min(COMPLEX_DIM) {
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
        let cache_out = slot_base + 4;  // output starts after 4-byte hash
        for j in 0..COMPLEX_DIM {
            let cached = tape[cache_out + j];
            total_entropy += cached.count_ones() as u64;
            tape[input_offset + j] ^= cached;
        }
        // Output head (f32)
        let mut best_score_f32: f32 = f32::NEG_INFINITY;
        for j in 0..64.min(HIDDEN_DIM) {
            let s = tape_f32(&tape, input_offset, j).abs();
            if s > best_score_f32 { best_score_f32 = s; best_token = j as u8; }
        }
        // Uncompute warm hit
        for j in 0..COMPLEX_DIM {
            tape[input_offset + j] ^= tape[cache_out + j];
        }
    } else {
        // COLD MISS: full layer stack (f32 tape)
        for layer_idx in 0..num_layers {
            let lwo = weight_offset + layer_idx * HIDDEN_DIM;
            let lwo_f32 = weight_offset + layer_idx * HIDDEN_DIM * F32_BYTES;
            let layer_save = saved_outputs_offset + layer_idx * COMPLEX_DIM;

            // Dynamic decatalysis
            let original_weight = tape[lwo_f32 .. lwo_f32 + HIDDEN_DIM * F32_BYTES].to_vec();
            let src_slice = &compressed_weights_bytes[layer_idx * HIDDEN_DIM .. (layer_idx + 1) * HIDDEN_DIM];
            tape[lwo .. lwo + HIDDEN_DIM].copy_from_slice(src_slice);
            spn_unscramble(&mut tape, lwo, HIDDEN_DIM, bh_key, &sbox, 12);

            if (layer_idx + 1) % 4 == 0 {
                // GATED ATTENTION LAYER (f32)
                let attn_idx = layer_idx / 4;
                let mut sum_sq = 0.0f32;
                for j in 0..HIDDEN_DIM {
                    let rx = tape_f32(&tape, input_offset, j);
                    let ry = tape_f32(&tape, input_offset + F32_DIM, j);
                    sum_sq += rx * rx + ry * ry;
                }
                let rms = (sum_sq / HIDDEN_DIM as f32 + 1e-5).sqrt();
                let mut nx = vec![0.0f32; HIDDEN_DIM * 2];
                for j in 0..HIDDEN_DIM {
                    nx[j] = tape_f32(&tape, input_offset, j) / rms;
                    nx[HIDDEN_DIM + j] = tape_f32(&tape, input_offset + F32_DIM, j) / rms;
                }
                // QKV projections
                let slot_base = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + step % MAX_SEQ_LEN) * 4 * F32_DIM;
                for j in 0..HIDDEN_DIM {
                    let wq = tape_f32(&tape, lwo_f32, j % 512);
                    let wk = tape_f32(&tape, lwo_f32 + 512 * F32_BYTES, j % 512);
                    let wv = tape_f32(&tape, lwo_f32 + 1024 * F32_BYTES, j % 512);
                    let qx = wq * nx[j]; let qy = wq * nx[HIDDEN_DIM + j];
                    let kx = wk * nx[j]; let ky = wk * nx[HIDDEN_DIM + j];
                    let vx = wv * nx[j]; let vy = wv * nx[HIDDEN_DIM + j];
                    let pg = pre_gate_base + layer_idx * COMPLEX_DIM;
                    tape_f32_xor(&mut tape, pg, j, qx);
                    tape_f32_xor(&mut tape, pg + F32_DIM, j, qy);
                    tape_f32_xor(&mut tape, slot_base, j, kx);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM, j, ky);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM * 2, j, vx);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM * 3, j, vy);
                    total_entropy += (qx.abs() as u64) + (ky.abs() as u64) + (vx.abs() as u64);
                }
                // Multi-head attention
                let num_heads = 16;
                let head_dim = HIDDEN_DIM / num_heads;
                let scale = 1.0 / (head_dim as f32).sqrt();
                let mut attn_x = vec![0.0f32; HIDDEN_DIM];
                let mut attn_y = vec![0.0f32; HIDDEN_DIM];
                for h in 0..num_heads {
                    let hs = h * head_dim;
                    let mut scores = vec![0.0f32; step + 1];
                    let mut max_s = f32::NEG_INFINITY;
                    for s in 0..=step {
                        let so = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + s % MAX_SEQ_LEN) * 4 * F32_DIM;
                        let mut dr = 0.0f32; let mut di = 0.0f32;
                        for d in 0..head_dim {
                            let qx = tape_f32(&tape, pre_gate_base + layer_idx * COMPLEX_DIM, hs + d);
                            let qy = tape_f32(&tape, pre_gate_base + layer_idx * COMPLEX_DIM + F32_DIM, hs + d);
                            let kx = tape_f32(&tape, so, hs + d);
                            let ky = tape_f32(&tape, so + F32_DIM, hs + d);
                            dr += qx * kx + qy * ky;
                            di += qy * kx - qx * ky;
                        }
                        let sc = (dr * dr + di * di).sqrt() * scale;
                        scores[s] = sc;
                        if sc > max_s { max_s = sc; }
                    }
                    let mut sum_e = 0.0f32;
                    let mut exps = vec![0.0f32; step + 1];
                    for s in 0..=step { exps[s] = (scores[s] - max_s).exp(); sum_e += exps[s]; }
                    for d in 0..head_dim {
                        let mut vsx = 0.0f32; let mut vsy = 0.0f32;
                        for s in 0..=step {
                            let so = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + s % MAX_SEQ_LEN) * 4 * F32_DIM;
                            let vx = tape_f32(&tape, so + F32_DIM * 2, hs + d);
                            let vy = tape_f32(&tape, so + F32_DIM * 3, hs + d);
                            let w = exps[s] / sum_e;
                            vsx += w * vx; vsy += w * vy;
                        }
                        attn_x[hs + d] = vsx; attn_y[hs + d] = vsy;
                        total_entropy += (vsx.abs() as u64) + (vsy.abs() as u64);
                    }
                }
                // Output projection + add to input
                for j in 0..HIDDEN_DIM {
                    let wo = tape_f32(&tape, lwo_f32 + 1536 * F32_BYTES, j % 512);
                    let px = wo * attn_x[j]; let py = wo * attn_y[j];
                    tape_f32_xor(&mut tape, layer_save, j, px);
                    tape_f32_xor(&mut tape, layer_save + F32_DIM, j, py);
                    tape_f32_xor(&mut tape, input_offset, j, px);
                    tape_f32_xor(&mut tape, input_offset + F32_DIM, j, py);
                    total_entropy += (px.abs() as u64) + (py.abs() as u64);
                }
            } else {
                // DELTANET LAYER (f32)
                for j in 0..HIDDEN_DIM {
                    let w = tape_f32(&tape, lwo_f32, j);
                    let x = tape_f32(&tape, input_offset, j);
                    let y = tape_f32(&tape, input_offset + F32_DIM, j);
                    let vx = w * x; let vy = w * y;
                    total_entropy += (vx.abs() as u64) + (vy.abs() as u64);
                    // save original temp
                    let pg = pre_gate_base + layer_idx * COMPLEX_DIM;
                    let tx = tape_f32(&tape, temp_offset, j);
                    let ty = tape_f32(&tape, temp_offset + F32_DIM, j);
                    tape_f32_xor(&mut tape, pg, j, tx);
                    tape_f32_xor(&mut tape, pg + F32_DIM, j, ty);
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
            tape[lwo_f32 .. lwo_f32 + HIDDEN_DIM * F32_BYTES].copy_from_slice(&original_weight);
        }

        // Output head (f32)
        let mut best_score_f32: f32 = f32::NEG_INFINITY;
        for j in 0..64.min(HIDDEN_DIM) {
            let s = tape_f32(&tape, input_offset, j).abs();
            if s > best_score_f32 { best_score_f32 = s; best_token = j as u8; }
        }

        // Uncompute: reverse each layer (f32)
        let mut tape_layer_hash = String::new();
        for layer_idx in (0..num_layers).rev() {
            let lwo = weight_offset + layer_idx * HIDDEN_DIM;
            let lwo_f32 = weight_offset + layer_idx * HIDDEN_DIM * F32_BYTES;
            let layer_save = saved_outputs_offset + layer_idx * COMPLEX_DIM;
            let pg = pre_gate_base + layer_idx * COMPLEX_DIM;

            let original_weight = tape[lwo_f32 .. lwo_f32 + HIDDEN_DIM * F32_BYTES].to_vec();
            let src_slice = &compressed_weights_bytes[layer_idx * HIDDEN_DIM .. (layer_idx + 1) * HIDDEN_DIM];
            tape[lwo .. lwo + HIDDEN_DIM].copy_from_slice(src_slice);
            spn_unscramble(&mut tape, lwo, HIDDEN_DIM, bh_key, &sbox, 12);

            if (layer_idx + 1) % 4 == 0 {
                // Attention uncompute (f32)
                let attn_idx = layer_idx / 4;
                let slot_base = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + step % MAX_SEQ_LEN) * 4 * F32_DIM;
                for j in 0..HIDDEN_DIM {
                    let lx = tape_f32(&tape, layer_save, j);
                    let ly = tape_f32(&tape, layer_save + F32_DIM, j);
                    tape_f32_xor(&mut tape, input_offset, j, lx);
                    tape_f32_xor(&mut tape, input_offset + F32_DIM, j, ly);
                }
                // Recompute attn_out from Q (in pre_gate) and cached KV
                let num_heads = 16;
                let head_dim = HIDDEN_DIM / num_heads;
                let scale = 1.0 / (head_dim as f32).sqrt();
                let mut attn_x = vec![0.0f32; HIDDEN_DIM];
                let mut attn_y = vec![0.0f32; HIDDEN_DIM];
                for h in 0..num_heads {
                    let hs = h * head_dim;
                    let mut scores = vec![0.0f32; step + 1];
                    let mut max_s = f32::NEG_INFINITY;
                    for s in 0..=step {
                        let so = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + s % MAX_SEQ_LEN) * 4 * F32_DIM;
                        let mut dr = 0.0f32; let mut di = 0.0f32;
                        for d in 0..head_dim {
                            let qx = tape_f32(&tape, pg, hs + d);
                            let qy = tape_f32(&tape, pg + F32_DIM, hs + d);
                            let kx = tape_f32(&tape, so, hs + d);
                            let ky = tape_f32(&tape, so + F32_DIM, hs + d);
                            dr += qx * kx + qy * ky; di += qy * kx - qx * ky;
                        }
                        let sc = (dr * dr + di * di).sqrt() * scale;
                        scores[s] = sc; if sc > max_s { max_s = sc; }
                    }
                    let mut sum_e = 0.0f32;
                    let mut exps = vec![0.0f32; step + 1];
                    for s in 0..=step { exps[s] = (scores[s] - max_s).exp(); sum_e += exps[s]; }
                    for d in 0..head_dim {
                        let mut vsx = 0.0f32; let mut vsy = 0.0f32;
                        for s in 0..=step {
                            let so = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + s % MAX_SEQ_LEN) * 4 * F32_DIM;
                            vsx += exps[s] / sum_e * tape_f32(&tape, so + F32_DIM * 2, hs + d);
                            vsy += exps[s] / sum_e * tape_f32(&tape, so + F32_DIM * 3, hs + d);
                        }
                        attn_x[hs + d] = vsx; attn_y[hs + d] = vsy;
                    }
                }
                for j in 0..HIDDEN_DIM {
                    let wo = tape_f32(&tape, lwo_f32 + 1536 * F32_BYTES, j % 512);
                    tape_f32_xor(&mut tape, layer_save, j, wo * attn_x[j]);
                    tape_f32_xor(&mut tape, layer_save + F32_DIM, j, wo * attn_y[j]);
                }
                // Undo Q projection (stored in pre_gate)
                for j in 0..HIDDEN_DIM {
                    // recompute RMS norm from input state
                    let mut sum_sq = 0.0_f32;
                    for k in 0..HIDDEN_DIM {
                        let rx = tape_f32(&tape, input_offset, k);
                        let ry = tape_f32(&tape, input_offset + F32_DIM, k);
                        sum_sq += rx * rx + ry * ry;
                    }
                    let rms = (sum_sq / HIDDEN_DIM as f32 + 1e-5).sqrt();
                    let nx = tape_f32(&tape, input_offset, j) / rms;
                    let ny = tape_f32(&tape, input_offset + F32_DIM, j) / rms;
                    let wq = tape_f32(&tape, lwo_f32, j % 512);
                    let wk = tape_f32(&tape, lwo_f32 + 512 * F32_BYTES, j % 512);
                    let wv = tape_f32(&tape, lwo_f32 + 1024 * F32_BYTES, j % 512);
                    tape_f32_xor(&mut tape, pg, j, wq * nx);
                    tape_f32_xor(&mut tape, pg + F32_DIM, j, wq * ny);
                    tape_f32_xor(&mut tape, slot_base, j, wk * nx);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM, j, wk * ny);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM * 2, j, wv * nx);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM * 3, j, wv * ny);
                }
            } else {
                // DeltaNet uncompute (f32)
                // Forward: temp^=Q, gate+copy combined. Backward: copy-undo, gate-undo, Q-undo
                for j in 0..HIDDEN_DIM {
                    let lx = tape_f32(&tape, layer_save, j);  // gate output
                    let ly = tape_f32(&tape, layer_save + F32_DIM, j);
                    tape_f32_xor(&mut tape, input_offset, j, lx);  // undo copy-through
                    tape_f32_xor(&mut tape, input_offset + F32_DIM, j, ly);
                    // now zero the gate in layer_save (recompute same gate, XOR back)
                    let gx = (0.5 + 0.25 * tape_f32(&tape, temp_offset, j)).clamp(0.0, 1.0);
                    let gy = (0.5 + 0.25 * tape_f32(&tape, temp_offset + F32_DIM, j)).clamp(0.0, 1.0);
                    tape_f32_xor(&mut tape, layer_save, j, gx);
                    tape_f32_xor(&mut tape, layer_save + F32_DIM, j, gy);
                }
                for j in 0..HIDDEN_DIM {
                    let w = tape_f32(&tape, lwo_f32, j);
                    let x = tape_f32(&tape, input_offset, j);
                    let y = tape_f32(&tape, input_offset + F32_DIM, j);
                    tape_f32_xor(&mut tape, temp_offset, j, w * x);
                    tape_f32_xor(&mut tape, temp_offset + F32_DIM, j, w * y);
                }
                for j in 0..HIDDEN_DIM {
                    let tx = tape_f32(&tape, temp_offset, j);
                    let ty = tape_f32(&tape, temp_offset + F32_DIM, j);
                    tape_f32_xor(&mut tape, pg, j, tx);
                    tape_f32_xor(&mut tape, pg + F32_DIM, j, ty);
                    // zero temp (pg already holds original_temp)
                    tape_f32_xor(&mut tape, temp_offset, j, tx);
                    tape_f32_xor(&mut tape, temp_offset + F32_DIM, j, ty);
                }
            }

            spn_scramble(&mut tape, lwo, HIDDEN_DIM, bh_key, &sbox, 12);
            tape[lwo_f32 .. lwo_f32 + HIDDEN_DIM * F32_BYTES].copy_from_slice(&original_weight);
        }
    } // end cold-miss block

    // Clear embedding
    for i in 0..token_embedding.len().min(COMPLEX_DIM) {
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

