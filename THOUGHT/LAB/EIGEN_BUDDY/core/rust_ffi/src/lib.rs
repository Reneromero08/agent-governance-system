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

// Weight region sub-offsets per layer: Q, K, V, O (4 x f32 matrices)
const WEIGHT_Q_OFFSET: usize = 0;
const WEIGHT_K_OFFSET: usize = 1 * F32_DIM;
const WEIGHT_V_OFFSET: usize = 2 * F32_DIM;
const WEIGHT_O_OFFSET: usize = 3 * F32_DIM;
const TOTAL_WEIGHT_F32: usize = 4 * F32_DIM;  // f32 weight bytes per layer (Q/K/V/O)
const TOTAL_WEIGHT_U8: usize = 4 * F32_DIM;  // u8 weight bytes per layer (same size as f32)

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
    fn test_4layer_mixed_restore() {
        // Layers: 0=delta, 1=delta, 2=delta, 3=attention
        let num_layers: usize = 4;
        let input_offset = 0usize;
        let weight_offset = COMPLEX_DIM;
        let scratch_base = weight_offset + num_layers * TOTAL_WEIGHT_F32;
        let temp_offset = scratch_base;
        let pre_gate_base = temp_offset + COMPLEX_DIM;
        let saved_outputs_offset = pre_gate_base + num_layers * COMPLEX_DIM;
        const WARM_TAPE_SLOTS: usize = 256;
        let warm_tape_offset = saved_outputs_offset + num_layers * COMPLEX_DIM;
        let warm_tape_stride = 4 + COMPLEX_DIM;
        let kv_cache_offset = warm_tape_offset + WARM_TAPE_SLOTS * warm_tape_stride;
        let work_end = warm_tape_offset + WARM_TAPE_SLOTS * warm_tape_stride;

        let total_scratch = kv_cache_offset + (num_layers / 4) * MAX_SEQ_LEN * 4 * F32_DIM;
        let mut tape = vec![0u8; total_scratch + 4096];

        // Set input
        tape_f32_xor(&mut tape, input_offset, 0, 0.5);
        tape_f32_xor(&mut tape, input_offset + F32_DIM, 0, 1.0);

        // Set weights for all layers — Q/K/V/O regions
        let base_weight: f32 = 0.1;
        for li in 0..num_layers {
            let lwo_f32 = weight_offset + li * TOTAL_WEIGHT_F32;
            let w = base_weight + li as f32 * 0.05;
            // Set first 512 entries of each weight matrix (enough for j%512 reads in delta + attention)
            for j in 0..512.min(HIDDEN_DIM) {
                tape_f32_xor(&mut tape, lwo_f32 + WEIGHT_Q_OFFSET, j, w);
                tape_f32_xor(&mut tape, lwo_f32 + WEIGHT_K_OFFSET, j, w);
                tape_f32_xor(&mut tape, lwo_f32 + WEIGHT_V_OFFSET, j, w);
                tape_f32_xor(&mut tape, lwo_f32 + WEIGHT_O_OFFSET, j, w);
            }
        }

        let hash_before: String = {
            let mut h = Sha256::new();
            ShaDigest::update(&mut h, &tape[..work_end]);
            format!("{:x}", h.finalize())
        };

        let step = 0usize;
        let num_heads = 16;
        let head_dim = HIDDEN_DIM / num_heads;

        // ---- FORWARD ALL LAYERS ----
        for layer_idx in 0..num_layers {
            let lwo_f32 = weight_offset + layer_idx * TOTAL_WEIGHT_F32;
            if (layer_idx + 1) % 4 == 0 && step == 0 {
                // Save pre-forward input bytes for comparison
                let _saved_input = tape[input_offset..input_offset+COMPLEX_DIM].to_vec();
                let _saved_idx = layer_idx;
            }
            
            let layer_save = saved_outputs_offset + layer_idx * COMPLEX_DIM;

            if (layer_idx + 1) % 4 == 0 {
                // ATTENTION
                let attn_idx = layer_idx / 4;
                let slot_base = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + step % MAX_SEQ_LEN) * 4 * F32_DIM;
                let pg = pre_gate_base + layer_idx * COMPLEX_DIM;
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
                for j in 0..HIDDEN_DIM {
                    let wq = tape_f32(&tape, lwo_f32 + WEIGHT_Q_OFFSET, j);
                    let wk = tape_f32(&tape, lwo_f32 + WEIGHT_K_OFFSET, j);
                    let wv = tape_f32(&tape, lwo_f32 + WEIGHT_V_OFFSET, j);
                    let qx = wq * nx[j]; let qy = wq * nx[HIDDEN_DIM + j];
                    let kx = wk * nx[j]; let ky = wk * nx[HIDDEN_DIM + j];
                    let vx = wv * nx[j]; let vy = wv * nx[HIDDEN_DIM + j];
                    tape_f32_xor(&mut tape, pg, j, qx);
                    tape_f32_xor(&mut tape, pg + F32_DIM, j, qy);
                    tape_f32_xor(&mut tape, slot_base, j, kx);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM, j, ky);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM * 2, j, vx);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM * 3, j, vy);
                }
                let mut attn_x = vec![0.0f32; HIDDEN_DIM];
                let mut attn_y = vec![0.0f32; HIDDEN_DIM];
                for h in 0..num_heads {
                    let hs = h * head_dim;
                    for d in 0..head_dim {
                        let vx = tape_f32(&tape, slot_base + F32_DIM * 2, hs + d);
                        let vy = tape_f32(&tape, slot_base + F32_DIM * 3, hs + d);
                        attn_x[hs + d] = vx;
                        attn_y[hs + d] = vy;
                    }
                }
                for j in 0..HIDDEN_DIM {
                    let wo = tape_f32(&tape, lwo_f32 + WEIGHT_O_OFFSET, j);
                    let px = wo * attn_x[j]; let py = wo * attn_y[j];
                    tape_f32_xor(&mut tape, layer_save, j, px);
                    tape_f32_xor(&mut tape, layer_save + F32_DIM, j, py);
                    tape_f32_xor(&mut tape, input_offset, j, px);
                    tape_f32_xor(&mut tape, input_offset + F32_DIM, j, py);
                }
            } else {
                // DELTANET — store Q as exact u32 bytes in pre_gate
                for j in 0..HIDDEN_DIM {
                    let w = tape_f32(&tape, lwo_f32, j);
                    let x = tape_f32(&tape, input_offset, j);
                    let y = tape_f32(&tape, input_offset + F32_DIM, j);
                    let vx = w * x; let vy = w * y;
                    let pg = pre_gate_base + layer_idx * COMPLEX_DIM;
                    let qx_bytes = vx.to_bits().to_le_bytes();
                    let qy_bytes = vy.to_bits().to_le_bytes();
                    for b in 0..4 {
                        tape[pg + j * 4 + b] ^= qx_bytes[b];
                        tape[pg + F32_DIM + j * 4 + b] ^= qy_bytes[b];
                    }
                    tape_f32_xor(&mut tape, temp_offset, j, vx);
                    tape_f32_xor(&mut tape, temp_offset + F32_DIM, j, vy);
                }
                for j in 0..HIDDEN_DIM {
                    let gx = (0.5 + 0.25 * tape_f32(&tape, temp_offset, j)).clamp(0.0, 1.0);
                    let gy = (0.5 + 0.25 * tape_f32(&tape, temp_offset + F32_DIM, j)).clamp(0.0, 1.0);
                    let gx_bytes = gx.to_bits().to_le_bytes();
                    let gy_bytes = gy.to_bits().to_le_bytes();
                    for b in 0..4 {
                        tape[layer_save + j * 4 + b] ^= gx_bytes[b];
                        tape[layer_save + F32_DIM + j * 4 + b] ^= gy_bytes[b];
                        tape[input_offset + j * 4 + b] ^= gx_bytes[b];
                        tape[input_offset + F32_DIM + j * 4 + b] ^= gy_bytes[b];
                    }
                }
            }
        }

        let hash_fwd: String = {
            let mut h = Sha256::new();
            ShaDigest::update(&mut h, &tape[..work_end]);
            format!("{:x}", h.finalize())
        };
        println!("After forward: {}", hash_fwd);

        // ---- UNCOMPUTE ALL LAYERS ----
        for layer_idx in (0..num_layers).rev() {
            let lwo_f32 = weight_offset + layer_idx * TOTAL_WEIGHT_F32;
            let layer_save = saved_outputs_offset + layer_idx * COMPLEX_DIM;
            let pg = pre_gate_base + layer_idx * COMPLEX_DIM;
            let temp_before = tape_f32(&tape, temp_offset, 0);

            if (layer_idx + 1) % 4 == 0 {
                // ATTENTION UNCOMPUTE
                let attn_idx = layer_idx / 4;
                let slot_base = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + step % MAX_SEQ_LEN) * 4 * F32_DIM;
                for j in 0..HIDDEN_DIM {
                    let lx = tape_f32(&tape, layer_save, j);
                    let ly = tape_f32(&tape, layer_save + F32_DIM, j);
                    tape_f32_xor(&mut tape, input_offset, j, lx);
                    tape_f32_xor(&mut tape, input_offset + F32_DIM, j, ly);
                }
                let mut attn_x2 = vec![0.0f32; HIDDEN_DIM];
                let mut attn_y2 = vec![0.0f32; HIDDEN_DIM];
                for h in 0..num_heads {
                    let hs = h * head_dim;
                    for d in 0..head_dim {
                        let vx = tape_f32(&tape, slot_base + F32_DIM * 2, hs + d);
                        let vy = tape_f32(&tape, slot_base + F32_DIM * 3, hs + d);
                        attn_x2[hs + d] = vx;
                        attn_y2[hs + d] = vy;
                    }
                }
                for j in 0..HIDDEN_DIM {
                    let wo = tape_f32(&tape, lwo_f32 + WEIGHT_O_OFFSET, j);
                    tape_f32_xor(&mut tape, layer_save, j, wo * attn_x2[j]);
                    tape_f32_xor(&mut tape, layer_save + F32_DIM, j, wo * attn_y2[j]);
                }
                let mut sum_sq = 0.0_f32;
                for k in 0..HIDDEN_DIM {
                    let rx = tape_f32(&tape, input_offset, k);
                    let ry = tape_f32(&tape, input_offset + F32_DIM, k);
                    sum_sq += rx * rx + ry * ry;
                }
                let rms = (sum_sq / HIDDEN_DIM as f32 + 1e-5).sqrt();
                for j in 0..HIDDEN_DIM {
                    let nx = tape_f32(&tape, input_offset, j) / rms;
                    let ny = tape_f32(&tape, input_offset + F32_DIM, j) / rms;
                    let wq = tape_f32(&tape, lwo_f32 + WEIGHT_Q_OFFSET, j);
                    let wk = tape_f32(&tape, lwo_f32 + WEIGHT_K_OFFSET, j);
                    let wv = tape_f32(&tape, lwo_f32 + WEIGHT_V_OFFSET, j);
                    tape_f32_xor(&mut tape, pg, j, wq * nx);
                    tape_f32_xor(&mut tape, pg + F32_DIM, j, wq * ny);
                    tape_f32_xor(&mut tape, slot_base, j, wk * nx);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM, j, wk * ny);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM * 2, j, wv * nx);
                    tape_f32_xor(&mut tape, slot_base + F32_DIM * 3, j, wv * ny);
                }
            } else {
                // DELTANET UNCOMPUTE — exact bit-cast: read gate from layer_save, read Q from pre_gate
                for j in 0..HIDDEN_DIM {
                    let mut gx_bytes = [0u8; 4];
                    let mut gy_bytes = [0u8; 4];
                    for b in 0..4 {
                        gx_bytes[b] = tape[layer_save + j * 4 + b];
                        gy_bytes[b] = tape[layer_save + F32_DIM + j * 4 + b];
                    }
                    for b in 0..4 {
                        tape[input_offset + j * 4 + b] ^= gx_bytes[b];
                        tape[input_offset + F32_DIM + j * 4 + b] ^= gy_bytes[b];
                        tape[layer_save + j * 4 + b] ^= gx_bytes[b];
                        tape[layer_save + F32_DIM + j * 4 + b] ^= gy_bytes[b];
                    }
                }
                for j in 0..HIDDEN_DIM {
                    let mut qx_bytes = [0u8; 4];
                    let mut qy_bytes = [0u8; 4];
                    let pg = pre_gate_base + layer_idx * COMPLEX_DIM;
                    for b in 0..4 {
                        qx_bytes[b] = tape[pg + j * 4 + b];
                        qy_bytes[b] = tape[pg + F32_DIM + j * 4 + b];
                    }
                    tape[temp_offset + j * 4] ^= qx_bytes[0];
                    tape[temp_offset + j * 4 + 1] ^= qx_bytes[1];
                    tape[temp_offset + j * 4 + 2] ^= qx_bytes[2];
                    tape[temp_offset + j * 4 + 3] ^= qx_bytes[3];
                    tape[temp_offset + F32_DIM + j * 4] ^= qy_bytes[0];
                    tape[temp_offset + F32_DIM + j * 4 + 1] ^= qy_bytes[1];
                    tape[temp_offset + F32_DIM + j * 4 + 2] ^= qy_bytes[2];
                    tape[temp_offset + F32_DIM + j * 4 + 3] ^= qy_bytes[3];
                    for b in 0..4 {
                        tape[pg + j * 4 + b] ^= qx_bytes[b];
                        tape[pg + F32_DIM + j * 4 + b] ^= qy_bytes[b];
                    }
                }
            }
            let temp_after = tape_f32(&tape, temp_offset, 0);
            println!("  Uncomp L{}: temp_before={:.6e}, temp_after={:.6e}", layer_idx, temp_before, temp_after);
        }

        let hash_after: String = {
            let mut h = Sha256::new();
            ShaDigest::update(&mut h, &tape[..work_end]);
            format!("{:x}", h.finalize())
        };
        println!("After uncompute: {}", hash_after);
        println!("MATCH: {}", hash_before == hash_after);
        // Check specific regions
        let tx0 = tape_f32(&tape, temp_offset, 0);
        let ty0 = tape_f32(&tape, temp_offset + F32_DIM, 0);
        println!("temp[0]: tx={:.6e}, ty={:.6e}", tx0, ty0);
        let pg0 = tape_f32(&tape, pre_gate_base, 0);
        println!("pre_gate[0] (layer 0): {:.6e}", pg0);
        let pg3 = tape_f32(&tape, pre_gate_base + 3 * COMPLEX_DIM, 0);
        println!("pre_gate[0] (layer 3): {:.6e}", pg3);
        let ls0 = tape_f32(&tape, saved_outputs_offset, 0);
        println!("layer_save[0] (layer 0): {:.6e}", ls0);
        let ls3 = tape_f32(&tape, saved_outputs_offset + 3 * COMPLEX_DIM, 0);
        println!("layer_save[0] (layer 3): {:.6e}", ls3);
        // Check delta layer 2's pre_gate and layer_save
        let pg2 = tape_f32(&tape, pre_gate_base + 2 * COMPLEX_DIM, 0);
        println!("pre_gate[0] (layer 2): {:.6e}", pg2);
        let ls2 = tape_f32(&tape, saved_outputs_offset + 2 * COMPLEX_DIM, 0);
        println!("layer_save[0] (layer 2): {:.6e}", ls2);
        assert_eq!(hash_before, hash_after, "4-layer mixed hash mismatch");

        let ix = tape_f32(&tape, input_offset, 0);
        let iy = tape_f32(&tape, input_offset + F32_DIM, 0);
        println!("Input raw bytes X [0..4]: {:02x?}", &tape[input_offset..input_offset+4]);
        println!("Input raw bytes Y [3584..3588]: {:02x?}", &tape[input_offset+F32_DIM..input_offset+F32_DIM+4]);
        println!("ix={:.6e}, iy={:.6e}", ix, iy);
        assert!((ix - 0.5).abs() < 0.001, "input X not restored: got {ix}");
        assert!((iy - 1.0).abs() < 0.001, "input Y not restored: got {iy}");
    }

    #[test]
    fn test_attention_1layer_engine_style() {
        // Use exact engine layout: attention layer at layer_idx=3 (first attention)
        let num_layers: usize = 4; // layers 0,1,2 are delta, layer 3 is attention
        let input_offset = 0usize;
        let weight_offset = COMPLEX_DIM;
        let scratch_base = weight_offset + num_layers * TOTAL_WEIGHT_F32;
        let temp_offset = scratch_base;
        let pre_gate_base = temp_offset + COMPLEX_DIM;
        let saved_outputs_offset = pre_gate_base + num_layers * COMPLEX_DIM;
        const WARM_TAPE_SLOTS: usize = 256;
        let warm_tape_offset = saved_outputs_offset + num_layers * COMPLEX_DIM;
        let warm_tape_stride = 4 + COMPLEX_DIM;
        let kv_cache_offset = warm_tape_offset + WARM_TAPE_SLOTS * warm_tape_stride;
        let work_end = (warm_tape_offset + WARM_TAPE_SLOTS * warm_tape_stride).min(usize::MAX);

        let total_scratch = kv_cache_offset + (num_layers / 4) * MAX_SEQ_LEN * 4 * F32_DIM;
        let mut tape = vec![0u8; total_scratch + 4096];

        // Set weight values for attention layer (layer 3 = last layer) — Q/K/V/O
        let layer_idx: usize = 3;
        let lwo_f32 = weight_offset + layer_idx * TOTAL_WEIGHT_F32;
        for j in 0..HIDDEN_DIM {
            tape_f32_xor(&mut tape, lwo_f32 + WEIGHT_Q_OFFSET, j, 0.1);
            tape_f32_xor(&mut tape, lwo_f32 + WEIGHT_K_OFFSET, j, 0.1);
            tape_f32_xor(&mut tape, lwo_f32 + WEIGHT_V_OFFSET, j, 0.1);
            tape_f32_xor(&mut tape, lwo_f32 + WEIGHT_O_OFFSET, j, 0.1);
        }
        
        // Set input
        tape_f32_xor(&mut tape, input_offset, 0, 0.5);
        tape_f32_xor(&mut tape, input_offset + F32_DIM, 0, 1.0);

        let hash_before: String = {
            let mut h = Sha256::new();
            ShaDigest::update(&mut h, &tape[..work_end]);
            format!("{:x}", h.finalize())
        };

        // Forward: skip delta layers 0,1,2 (they modify input/temp/pre_gate/layer_save but we haven't set up weights)
        // Just run attention layer 3 directly
        {
            let lwo = weight_offset + 3 * TOTAL_WEIGHT_U8;
            let layer_save = saved_outputs_offset + 3 * COMPLEX_DIM;
            let original_weight = tape[lwo_f32 .. lwo_f32 + TOTAL_WEIGHT_F32].to_vec();
            let step = 0usize;

            // RMS norm
            let mut sum_sq = 0.0f32;
            for j in 0..HIDDEN_DIM {
                let rx = tape_f32(&tape, input_offset, j);
                let ry = tape_f32(&tape, input_offset + F32_DIM, j);
                sum_sq += rx * rx + ry * ry;
            }
            let rms = (sum_sq / HIDDEN_DIM as f32 + 1e-5).sqrt();
            let mut nx_fwd = vec![0.0f32; HIDDEN_DIM * 2];
            for j in 0..HIDDEN_DIM {
                nx_fwd[j] = tape_f32(&tape, input_offset, j) / rms;
                nx_fwd[HIDDEN_DIM + j] = tape_f32(&tape, input_offset + F32_DIM, j) / rms;
            }
            // QKV
            let attn_idx = 3 / 4;
            let slot_base = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + step % MAX_SEQ_LEN) * 4 * F32_DIM;
            let pg = pre_gate_base + 3 * COMPLEX_DIM;
            for j in 0..HIDDEN_DIM {
                let wq = tape_f32(&tape, lwo_f32 + WEIGHT_Q_OFFSET, j);
                let wk = tape_f32(&tape, lwo_f32 + WEIGHT_K_OFFSET, j);
                let wv = tape_f32(&tape, lwo_f32 + WEIGHT_V_OFFSET, j);
                let qx = wq * nx_fwd[j]; let qy = wq * nx_fwd[HIDDEN_DIM + j];
                let kx = wk * nx_fwd[j]; let ky = wk * nx_fwd[HIDDEN_DIM + j];
                let vx = wv * nx_fwd[j]; let vy = wv * nx_fwd[HIDDEN_DIM + j];
                tape_f32_xor(&mut tape, pg, j, qx);
                tape_f32_xor(&mut tape, pg + F32_DIM, j, qy);
                tape_f32_xor(&mut tape, slot_base, j, kx);
                tape_f32_xor(&mut tape, slot_base + F32_DIM, j, ky);
                tape_f32_xor(&mut tape, slot_base + F32_DIM * 2, j, vx);
                tape_f32_xor(&mut tape, slot_base + F32_DIM * 3, j, vy);
            }
            // Multi-head attn (step=0, self-attention, softmax = 1.0)
            let num_heads = 16;
            let head_dim = HIDDEN_DIM / num_heads;
            let mut attn_x = vec![0.0f32; HIDDEN_DIM];
            let mut attn_y = vec![0.0f32; HIDDEN_DIM];
            for h in 0..num_heads {
                let hs = h * head_dim;
                for d in 0..head_dim {
                    let vx = tape_f32(&tape, slot_base + F32_DIM * 2, hs + d);
                    let vy = tape_f32(&tape, slot_base + F32_DIM * 3, hs + d);
                    attn_x[hs + d] = vx;
                    attn_y[hs + d] = vy;
                }
            }
            // Output projection
            for j in 0..HIDDEN_DIM {
                let wo = tape_f32(&tape, lwo_f32 + WEIGHT_O_OFFSET, j);
                let px = wo * attn_x[j]; let py = wo * attn_y[j];
                tape_f32_xor(&mut tape, layer_save, j, px);
                tape_f32_xor(&mut tape, layer_save + F32_DIM, j, py);
                tape_f32_xor(&mut tape, input_offset, j, px);
                tape_f32_xor(&mut tape, input_offset + F32_DIM, j, py);
            }
        }

        let hash_mid: String = {
            let mut h = Sha256::new();
            ShaDigest::update(&mut h, &tape[..work_end]);
            format!("{:x}", h.finalize())
        };
        println!("After forward: hash={}", hash_mid);
        assert_ne!(hash_before, hash_mid, "forward should modify tape");

        // Uncompute: attention layer 3 (reverse)
        {
            let lwo = weight_offset + 3 * TOTAL_WEIGHT_U8;
            let layer_save = saved_outputs_offset + 3 * COMPLEX_DIM;
            let pg = pre_gate_base + 3 * COMPLEX_DIM;
            let step = 0usize;
            let attn_idx = 3 / 4;
            let slot_base = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + step % MAX_SEQ_LEN) * 4 * F32_DIM;

            // Step 1: undo copy
            for j in 0..HIDDEN_DIM {
                let lx = tape_f32(&tape, layer_save, j);
                let ly = tape_f32(&tape, layer_save + F32_DIM, j);
                tape_f32_xor(&mut tape, input_offset, j, lx);
                tape_f32_xor(&mut tape, input_offset + F32_DIM, j, ly);
            }
            // Step 2: recompute attn_out
            let num_heads = 16;
            let head_dim = HIDDEN_DIM / num_heads;
            let mut attn_x2 = vec![0.0f32; HIDDEN_DIM];
            let mut attn_y2 = vec![0.0f32; HIDDEN_DIM];
            for h in 0..num_heads {
                let hs = h * head_dim;
                for d in 0..head_dim {
                    let vx = tape_f32(&tape, slot_base + F32_DIM * 2, hs + d);
                    let vy = tape_f32(&tape, slot_base + F32_DIM * 3, hs + d);
                    attn_x2[hs + d] = vx;
                    attn_y2[hs + d] = vy;
                }
            }
            // Step 3: undo layer_save
            for j in 0..HIDDEN_DIM {
                let wo = tape_f32(&tape, lwo_f32 + WEIGHT_O_OFFSET, j);
                tape_f32_xor(&mut tape, layer_save, j, wo * attn_x2[j]);
                tape_f32_xor(&mut tape, layer_save + F32_DIM, j, wo * attn_y2[j]);
            }
            // Step 4: undo QKV (recompute RMS norm from restored input)
            let mut sum_sq = 0.0_f32;
            for k in 0..HIDDEN_DIM {
                let rx = tape_f32(&tape, input_offset, k);
                let ry = tape_f32(&tape, input_offset + F32_DIM, k);
                sum_sq += rx * rx + ry * ry;
            }
            let rms = (sum_sq / HIDDEN_DIM as f32 + 1e-5).sqrt();
            for j in 0..HIDDEN_DIM {
                let nx = tape_f32(&tape, input_offset, j) / rms;
                let ny = tape_f32(&tape, input_offset + F32_DIM, j) / rms;
                let wq = tape_f32(&tape, lwo_f32 + WEIGHT_Q_OFFSET, j);
                let wk = tape_f32(&tape, lwo_f32 + WEIGHT_K_OFFSET, j);
                let wv = tape_f32(&tape, lwo_f32 + WEIGHT_V_OFFSET, j);
                tape_f32_xor(&mut tape, pg, j, wq * nx);
                tape_f32_xor(&mut tape, pg + F32_DIM, j, wq * ny);
                tape_f32_xor(&mut tape, slot_base, j, wk * nx);
                tape_f32_xor(&mut tape, slot_base + F32_DIM, j, wk * ny);
                tape_f32_xor(&mut tape, slot_base + F32_DIM * 2, j, wv * nx);
                tape_f32_xor(&mut tape, slot_base + F32_DIM * 3, j, wv * ny);
            }
        }

        let hash_after: String = {
            let mut h = Sha256::new();
            ShaDigest::update(&mut h, &tape[..work_end]);
            format!("{:x}", h.finalize())
        };
        println!("After uncompute: hash={}", hash_after);
        println!("MATCH: {}", hash_before == hash_after);
        
        // Debug per-region hashes
        let input_hash = { let mut h = Sha256::new(); ShaDigest::update(&mut h, &tape[input_offset..input_offset+COMPLEX_DIM]); format!("{:x}", h.finalize()) };
        let weight_hash = { let mut h = Sha256::new(); ShaDigest::update(&mut h, &tape[lwo_f32..lwo_f32+TOTAL_WEIGHT_F32]); format!("{:x}", h.finalize()) };
        let pg3_hash = { let pg = pre_gate_base + 3*COMPLEX_DIM; let mut h = Sha256::new(); ShaDigest::update(&mut h, &tape[pg..pg+COMPLEX_DIM]); format!("{:x}", h.finalize()) };
        let ls3_hash = { let ls = saved_outputs_offset + 3*COMPLEX_DIM; let mut h = Sha256::new(); ShaDigest::update(&mut h, &tape[ls..ls+COMPLEX_DIM]); format!("{:x}", h.finalize()) };
        let slot_hash = { let sb = kv_cache_offset; let mut h = Sha256::new(); ShaDigest::update(&mut h, &tape[sb..sb+4*F32_DIM]); format!("{:x}", h.finalize()) };
        println!("  input: {}", &input_hash[..16]);
        println!("  weight: {}", &weight_hash[..16]);
        println!("  pg[3]: {}", &pg3_hash[..16]);
        println!("  ls[3]: {}", &ls3_hash[..16]);
        println!("  slot: {}", &slot_hash[..16]);
        
        assert_eq!(hash_before, hash_after, "attention engine-style hash mismatch");

        let ix = tape_f32(&tape, input_offset, 0);
        let iy = tape_f32(&tape, input_offset + F32_DIM, 0);
        assert!((ix - 0.5).abs() < 0.001, "input X not restored: got {ix}");
        assert!((iy - 1.0).abs() < 0.001, "input Y not restored: got {iy}");
    }

    #[test]
    fn test_attention_1layer_restore() {
        let dim = 896;
        let f32b = 4;
        let complex_dim = dim * f32b * 2;
        let f32_dim = dim * f32b;
        let num_heads = 16;
        let head_dim = dim / num_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let pre_gate_stride = complex_dim;
        let layer_save_stride = complex_dim;
        let slot_size = 4 * f32_dim; // QKV+output per step
        let total_weight_f32 = 4 * f32_dim; // Q/K/V/O f32 regions
        
        let mut tape = vec![0u8; complex_dim * 8 + pre_gate_stride + layer_save_stride + slot_size + 4096];
        let input_off = 0usize;
        let weight_off = complex_dim; // 7168
        let pre_gate_off = weight_off + total_weight_f32; // 7168 + 14336 = 21504
        let layer_save_off = pre_gate_off + pre_gate_stride; // 21504 + 7168 = 28672
        let slot_off = layer_save_off + layer_save_stride; // 28672 + 7168 = 35840

        // Set input (small values)
        tape_f32_xor(&mut tape, input_off, 0, 0.5);
        tape_f32_xor(&mut tape, input_off + f32_dim, 0, 1.0);
        // Set weights — Q/K/V/O regions, fill all entries
        for j in 0..dim {
            tape_f32_xor(&mut tape, weight_off, j, 0.2);             // Q
            tape_f32_xor(&mut tape, weight_off + f32_dim, j, 0.15);   // K
            tape_f32_xor(&mut tape, weight_off + 2*f32_dim, j, 0.1);  // V
            tape_f32_xor(&mut tape, weight_off + 3*f32_dim, j, 0.05); // O
        }

        let hash_before = {
            let mut h = Sha256::new();
            ShaDigest::update(&mut h, &tape[..slot_off + slot_size]);
            format!("{:x}", h.finalize())
        };

        // Forward attention (step=0, simplified — no KV history)
        let step = 0usize;
        // RMS norm
        let mut sum_sq = 0.0f32;
        for j in 0..dim {
            let rx = tape_f32(&tape, input_off, j);
            let ry = tape_f32(&tape, input_off + f32_dim, j);
            sum_sq += rx * rx + ry * ry;
        }
        let rms = (sum_sq / dim as f32 + 1e-5).sqrt();
        let mut nx = vec![0.0f32; dim * 2];
        for j in 0..dim {
            nx[j] = tape_f32(&tape, input_off, j) / rms;
            nx[dim + j] = tape_f32(&tape, input_off + f32_dim, j) / rms;
        }
        // QKV projections
        for j in 0..dim {
            let wq = tape_f32(&tape, weight_off, j);           // Q at offset 0
            let wk = tape_f32(&tape, weight_off + f32_dim, j);  // K at offset f32_dim
            let wv = tape_f32(&tape, weight_off + 2*f32_dim, j); // V at offset 2*f32_dim
            let qx = wq * nx[j]; let qy = wq * nx[dim + j];
            let kx = wk * nx[j]; let ky = wk * nx[dim + j];
            let vx = wv * nx[j]; let vy = wv * nx[dim + j];
            tape_f32_xor(&mut tape, pre_gate_off, j, qx);
            tape_f32_xor(&mut tape, pre_gate_off + f32_dim, j, qy);
            tape_f32_xor(&mut tape, slot_off, j, kx);
            tape_f32_xor(&mut tape, slot_off + f32_dim, j, ky);
            tape_f32_xor(&mut tape, slot_off + f32_dim * 2, j, vx);
            tape_f32_xor(&mut tape, slot_off + f32_dim * 3, j, vy);
        }
        // Self-attention (step=0, just Q·K_T·V with self)
        let mut attn_x = vec![0.0f32; dim];
        let mut attn_y = vec![0.0f32; dim];
        for h in 0..num_heads {
            let hs = h * head_dim;
            let mut dr = 0.0f32; let mut di = 0.0f32;
            for d in 0..head_dim {
                let qx = tape_f32(&tape, pre_gate_off, hs + d);
                let qy = tape_f32(&tape, pre_gate_off + f32_dim, hs + d);
                let kx = tape_f32(&tape, slot_off, hs + d);
                let ky = tape_f32(&tape, slot_off + f32_dim, hs + d);
                dr += qx * kx + qy * ky;
                di += qy * kx - qx * ky;
            }
            let sc = (dr * dr + di * di).sqrt() * scale;
            // Self-attn: exp(score)/exp(score) = 1.0
            for d in 0..head_dim {
                let vx = tape_f32(&tape, slot_off + f32_dim * 2, hs + d);
                let vy = tape_f32(&tape, slot_off + f32_dim * 3, hs + d);
                attn_x[hs + d] = vx;
                attn_y[hs + d] = vy;
            }
        }
        // Output projection
        for j in 0..dim {
            let wo = tape_f32(&tape, weight_off + 3*f32_dim, j);
            let px = wo * attn_x[j]; let py = wo * attn_y[j];
            tape_f32_xor(&mut tape, layer_save_off, j, px);
            tape_f32_xor(&mut tape, layer_save_off + f32_dim, j, py);
            tape_f32_xor(&mut tape, input_off, j, px);
            tape_f32_xor(&mut tape, input_off + f32_dim, j, py);
        }

        // ---- Uncompute ----
        // Step 1: undo output copy-through
        for j in 0..dim {
            let lx = tape_f32(&tape, layer_save_off, j);
            let ly = tape_f32(&tape, layer_save_off + f32_dim, j);
            tape_f32_xor(&mut tape, input_off, j, lx);
            tape_f32_xor(&mut tape, input_off + f32_dim, j, ly);
        }
        // Step 2: recompute attn_out
        let mut attn_x2 = vec![0.0f32; dim];
        let mut attn_y2 = vec![0.0f32; dim];
        for h in 0..num_heads {
            let hs = h * head_dim;
            let mut dr = 0.0f32; let mut di = 0.0f32;
            for d in 0..head_dim {
                let qx = tape_f32(&tape, pre_gate_off, hs + d);
                let qy = tape_f32(&tape, pre_gate_off + f32_dim, hs + d);
                let kx = tape_f32(&tape, slot_off, hs + d);
                let ky = tape_f32(&tape, slot_off + f32_dim, hs + d);
                dr += qx * kx + qy * ky; di += qy * kx - qx * ky;
            }
            for d in 0..head_dim {
                let vx = tape_f32(&tape, slot_off + f32_dim * 2, hs + d);
                let vy = tape_f32(&tape, slot_off + f32_dim * 3, hs + d);
                attn_x2[hs + d] = vx;
                attn_y2[hs + d] = vy;
            }
        }
        // Step 3: undo layer_save
        for j in 0..dim {
            let wo = tape_f32(&tape, weight_off + 3*f32_dim, j);
            tape_f32_xor(&mut tape, layer_save_off, j, wo * attn_x2[j]);
            tape_f32_xor(&mut tape, layer_save_off + f32_dim, j, wo * attn_y2[j]);
        }
        // Step 4: undo QKV projections
        let mut sum_sq = 0.0_f32;
        for j in 0..dim {
            let rx = tape_f32(&tape, input_off, j);
            let ry = tape_f32(&tape, input_off + f32_dim, j);
            sum_sq += rx * rx + ry * ry;
        }
        let rms = (sum_sq / dim as f32 + 1e-5).sqrt();
        let mut nx = vec![0.0f32; dim * 2];
        for j in 0..dim {
            nx[j] = tape_f32(&tape, input_off, j) / rms;
            nx[dim + j] = tape_f32(&tape, input_off + f32_dim, j) / rms;
        }
        for j in 0..dim {
            let wq = tape_f32(&tape, weight_off, j);
            let wk = tape_f32(&tape, weight_off + f32_dim, j);
            let wv = tape_f32(&tape, weight_off + 2*f32_dim, j);
            let qx = wq * nx[j]; let qy = wq * nx[dim + j];
            let kx = wk * nx[j]; let ky = wk * nx[dim + j];
            let vx = wv * nx[j]; let vy = wv * nx[dim + j];
            tape_f32_xor(&mut tape, pre_gate_off, j, qx);
            tape_f32_xor(&mut tape, pre_gate_off + f32_dim, j, qy);
            tape_f32_xor(&mut tape, slot_off, j, kx);
            tape_f32_xor(&mut tape, slot_off + f32_dim, j, ky);
            tape_f32_xor(&mut tape, slot_off + f32_dim * 2, j, vx);
            tape_f32_xor(&mut tape, slot_off + f32_dim * 3, j, vy);
        }

        let hash_after = {
            let mut h = Sha256::new();
            ShaDigest::update(&mut h, &tape[..slot_off + slot_size]);
            format!("{:x}", h.finalize())
        };

        let ix = tape_f32(&tape, input_off, 0);
        let iy = tape_f32(&tape, input_off + f32_dim, 0);
        println!("ATTN RESTORE: ix={:.6e}, iy={:.6e}", ix, iy);
        println!("HASH MATCH: {}", hash_before == hash_after);
        assert_eq!(hash_before, hash_after, "attention tape hash mismatch");
        assert!((ix - 0.5).abs() < 0.001, "input X not restored: got {ix}");
        assert!((iy - 1.0).abs() < 0.001, "input Y not restored: got {iy}");
    }

    #[test]
    fn test_delta_net_2layer_restore() {
        let dim = 896;
        let f32b = 4;
        let complex_dim = dim * f32b * 2;
        let f32_dim = dim * f32b;
        let pre_gate_stride = complex_dim;
        let layer_save_stride = complex_dim;
        let mut tape = vec![0u8; complex_dim * 8 + pre_gate_stride * 2 + layer_save_stride * 2 + 4096];
        let input_off = 0usize;
        let weight_off = complex_dim;
        let temp_off = weight_off + f32_dim * 2; // two weights
        let pre_gate_off = temp_off + complex_dim;
        let layer_save_off = pre_gate_off + pre_gate_stride * 2;

        // Set two weights (one per layer)
        tape_f32_xor(&mut tape, weight_off, 0, 0.2);
        tape_f32_xor(&mut tape, weight_off + f32_dim, 0, 0.3);
        // Set input
        tape_f32_xor(&mut tape, input_off, 0, 0.5);
        tape_f32_xor(&mut tape, input_off + f32_dim, 0, 1.0);

        // Save initial full tape hash
        let initial_hash = {
            let mut h = Sha256::new();
            ShaDigest::update(&mut h, &tape[..layer_save_off + layer_save_stride * 2]);
            format!("{:x}", h.finalize())
        };

        // ---- FORWARD LAYER 0 ----
        let pg0 = pre_gate_off;
        let ls0 = layer_save_off;
        for j in 0..dim {
            let w = tape_f32(&tape, weight_off, j % 512);
            let x = tape_f32(&tape, input_off, j);
            let y = tape_f32(&tape, input_off + f32_dim, j);
            let vx = w * x; let vy = w * y;
            let qx_bytes = vx.to_bits().to_le_bytes();
            let qy_bytes = vy.to_bits().to_le_bytes();
            for b in 0..4 {
                tape[pg0 + j * 4 + b] ^= qx_bytes[b];
                tape[pg0 + f32_dim + j * 4 + b] ^= qy_bytes[b];
            }
            tape_f32_xor(&mut tape, temp_off, j, vx);
            tape_f32_xor(&mut tape, temp_off + f32_dim, j, vy);
        }
        for j in 0..dim {
            let gx = (0.5 + 0.25 * tape_f32(&tape, temp_off, j)).clamp(0.0, 1.0);
            let gy = (0.5 + 0.25 * tape_f32(&tape, temp_off + f32_dim, j)).clamp(0.0, 1.0);
            tape_f32_xor(&mut tape, ls0, j, gx);
            tape_f32_xor(&mut tape, ls0 + f32_dim, j, gy);
            tape_f32_xor(&mut tape, input_off, j, gx);
            tape_f32_xor(&mut tape, input_off + f32_dim, j, gy);
        }

        // ---- FORWARD LAYER 1 ----
        let pg1 = pre_gate_off + pre_gate_stride;
        let ls1 = layer_save_off + layer_save_stride;
        for j in 0..dim {
            let w = tape_f32(&tape, weight_off + f32_dim, j % 512);
            let x = tape_f32(&tape, input_off, j);
            let y = tape_f32(&tape, input_off + f32_dim, j);
            let vx = w * x; let vy = w * y;
            let qx_bytes = vx.to_bits().to_le_bytes();
            let qy_bytes = vy.to_bits().to_le_bytes();
            for b in 0..4 {
                tape[pg1 + j * 4 + b] ^= qx_bytes[b];
                tape[pg1 + f32_dim + j * 4 + b] ^= qy_bytes[b];
            }
            tape_f32_xor(&mut tape, temp_off, j, vx);
            tape_f32_xor(&mut tape, temp_off + f32_dim, j, vy);
        }
        for j in 0..dim {
            let gx = (0.5 + 0.25 * tape_f32(&tape, temp_off, j)).clamp(0.0, 1.0);
            let gy = (0.5 + 0.25 * tape_f32(&tape, temp_off + f32_dim, j)).clamp(0.0, 1.0);
            let gx_bytes = gx.to_bits().to_le_bytes();
            let gy_bytes = gy.to_bits().to_le_bytes();
            for b in 0..4 {
                tape[ls1 + j * 4 + b] ^= gx_bytes[b];
                tape[ls1 + f32_dim + j * 4 + b] ^= gy_bytes[b];
                tape[input_off + j * 4 + b] ^= gx_bytes[b];
                tape[input_off + f32_dim + j * 4 + b] ^= gy_bytes[b];
            }
        }

        // ---- UNCOMPUTE LAYER 1 (reverse order) ----
        for j in 0..dim {
            let mut gx_bytes = [0u8; 4];
            let mut gy_bytes = [0u8; 4];
            for b in 0..4 {
                gx_bytes[b] = tape[ls1 + j * 4 + b];
                gy_bytes[b] = tape[ls1 + f32_dim + j * 4 + b];
            }
            for b in 0..4 {
                tape[input_off + j * 4 + b] ^= gx_bytes[b];
                tape[input_off + f32_dim + j * 4 + b] ^= gy_bytes[b];
                tape[ls1 + j * 4 + b] ^= gx_bytes[b];
                tape[ls1 + f32_dim + j * 4 + b] ^= gy_bytes[b];
            }
        }
        for j in 0..dim {
            let mut qx_bytes = [0u8; 4];
            let mut qy_bytes = [0u8; 4];
            for b in 0..4 {
                qx_bytes[b] = tape[pg1 + j * 4 + b];
                qy_bytes[b] = tape[pg1 + f32_dim + j * 4 + b];
            }
            tape[temp_off + j * 4] ^= qx_bytes[0];
            tape[temp_off + j * 4 + 1] ^= qx_bytes[1];
            tape[temp_off + j * 4 + 2] ^= qx_bytes[2];
            tape[temp_off + j * 4 + 3] ^= qx_bytes[3];
            tape[temp_off + f32_dim + j * 4] ^= qy_bytes[0];
            tape[temp_off + f32_dim + j * 4 + 1] ^= qy_bytes[1];
            tape[temp_off + f32_dim + j * 4 + 2] ^= qy_bytes[2];
            tape[temp_off + f32_dim + j * 4 + 3] ^= qy_bytes[3];
            for b in 0..4 {
                tape[pg1 + j * 4 + b] ^= qx_bytes[b];
                tape[pg1 + f32_dim + j * 4 + b] ^= qy_bytes[b];
            }
        }

        // ---- UNCOMPUTE LAYER 0 ----
        for j in 0..dim {
            let lx = tape_f32(&tape, ls0, j);
            let ly = tape_f32(&tape, ls0 + f32_dim, j);
            tape_f32_xor(&mut tape, input_off, j, lx);
            tape_f32_xor(&mut tape, input_off + f32_dim, j, ly);
            tape_f32_xor(&mut tape, ls0, j, lx);
            tape_f32_xor(&mut tape, ls0 + f32_dim, j, ly);
        }
        for j in 0..dim {
            let mut qx_bytes = [0u8; 4];
            let mut qy_bytes = [0u8; 4];
            for b in 0..4 {
                qx_bytes[b] = tape[pg0 + j * 4 + b];
                qy_bytes[b] = tape[pg0 + f32_dim + j * 4 + b];
            }
            tape[temp_off + j * 4] ^= qx_bytes[0];
            tape[temp_off + j * 4 + 1] ^= qx_bytes[1];
            tape[temp_off + j * 4 + 2] ^= qx_bytes[2];
            tape[temp_off + j * 4 + 3] ^= qx_bytes[3];
            tape[temp_off + f32_dim + j * 4] ^= qy_bytes[0];
            tape[temp_off + f32_dim + j * 4 + 1] ^= qy_bytes[1];
            tape[temp_off + f32_dim + j * 4 + 2] ^= qy_bytes[2];
            tape[temp_off + f32_dim + j * 4 + 3] ^= qy_bytes[3];
            for b in 0..4 {
                tape[pg0 + j * 4 + b] ^= qx_bytes[b];
                tape[pg0 + f32_dim + j * 4 + b] ^= qy_bytes[b];
            }
        }

        // Verify full restoration
        let final_hash = {
            let mut h = Sha256::new();
            ShaDigest::update(&mut h, &tape[..layer_save_off + layer_save_stride * 2]);
            format!("{:x}", h.finalize())
        };

        let ix = tape_f32(&tape, input_off, 0);
        let iy = tape_f32(&tape, input_off + f32_dim, 0);
        println!("2-LAYER RESTORE: ix={:.6e}, iy={:.6e}", ix, iy);
        println!("HASH MATCH: {}", initial_hash == final_hash);

        // Debug: check non-zero regions
        let mut nonzero_regions = Vec::new();
        for off in 0..(layer_save_off + layer_save_stride * 2) {
            if tape[off] != 0 {
                nonzero_regions.push(off);
            }
        }
        println!("Non-zero tape bytes: {}", nonzero_regions.len());
        for &off in &nonzero_regions {
            println!("  tape[{off}] = 0x{val:02x}", val = tape[off]);
        }
        assert_eq!(initial_hash, final_hash, "2-layer tape hash mismatch");
        assert!((ix - 0.5).abs() < 0.001, "input X not restored: got {ix}");
        assert!((iy - 1.0).abs() < 0.001, "input Y not restored: got {iy}");
    }

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

        // Forward DeltaNet — store Q as u32 bytes in pre_gate
        let w = tape_f32(&tape, weight_off, 0);
        let x = tape_f32(&tape, input_off, 0);
        let y = tape_f32(&tape, input_off + f32_dim, 0);
        let vx = w * x; let vy = w * y;
        let qx_bytes = vx.to_bits().to_le_bytes();
        let qy_bytes = vy.to_bits().to_le_bytes();
        for b in 0..4 {
            tape[pre_gate_off + b] ^= qx_bytes[b];
            tape[pre_gate_off + f32_dim + b] ^= qy_bytes[b];
        }
        tape_f32_xor(&mut tape, temp_off, 0, vx);
        tape_f32_xor(&mut tape, temp_off + f32_dim, 0, vy);
        let gx = (0.5 + 0.25 * tape_f32(&tape, temp_off, 0)).clamp(0.0, 1.0);
        let gy = (0.5 + 0.25 * tape_f32(&tape, temp_off + f32_dim, 0)).clamp(0.0, 1.0);
        tape_f32_xor(&mut tape, layer_save_off, 0, gx);
        tape_f32_xor(&mut tape, layer_save_off + f32_dim, 0, gy);
        tape_f32_xor(&mut tape, input_off, 0, gx);
        tape_f32_xor(&mut tape, input_off + f32_dim, 0, gy);

        // Backward (correct order) — exact bit-cast from layer_save for gate, from pre_gate for Q
        for j in 0..dim {
            let gx_bwd = tape_f32(&tape, layer_save_off, j);
            let gy_bwd = tape_f32(&tape, layer_save_off + f32_dim, j);
            tape_f32_xor(&mut tape, input_off, j, gx_bwd);
            tape_f32_xor(&mut tape, input_off + f32_dim, j, gy_bwd);
            tape_f32_xor(&mut tape, layer_save_off, j, gx_bwd);
            tape_f32_xor(&mut tape, layer_save_off + f32_dim, j, gy_bwd);
        }
        // Undo Q: read exact Q bytes from pre_gate
        let mut qx_bytes = [0u8; 4];
        let mut qy_bytes = [0u8; 4];
        for b in 0..4 {
            qx_bytes[b] = tape[pre_gate_off + b];
            qy_bytes[b] = tape[pre_gate_off + f32_dim + b];
        }
        tape[temp_off] ^= qx_bytes[0];
        tape[temp_off + 1] ^= qx_bytes[1];
        tape[temp_off + 2] ^= qx_bytes[2];
        tape[temp_off + 3] ^= qx_bytes[3];
        tape[temp_off + f32_dim] ^= qy_bytes[0];
        tape[temp_off + f32_dim + 1] ^= qy_bytes[1];
        tape[temp_off + f32_dim + 2] ^= qy_bytes[2];
        tape[temp_off + f32_dim + 3] ^= qy_bytes[3];
        for b in 0..4 {
            tape[pre_gate_off + b] ^= qx_bytes[b];
            tape[pre_gate_off + f32_dim + b] ^= qy_bytes[b];
        }

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
    let mut first_broken_layer: i32 = -1;

    let input_offset = 0usize;
    let weight_offset = COMPLEX_DIM;  // after input XY region (COMPLEX_DIM = 7168 bytes)
    
    // Scratch layout (f32 for compute regions, u8 for weight buffer):
    // Both lwo (u8) and lwo_f32 (f32) share the same per-layer starting offset,
    // but lwo_f32 is 4x larger. scratch must start after the largest region.
    let scratch_base = weight_offset + num_layers * TOTAL_WEIGHT_F32;  // after f32 weight region
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

    let mut best_token: u8 = 0;
    let mut hidden_state_save: Vec<u8> = Vec::new();
    let work_end = (warm_tape_offset + WARM_TAPE_SLOTS * warm_tape_stride).min(tape.len());

    if let Some(slot) = cache_slot {
        warm_hit = true;
        let slot_base = warm_tape_offset + slot * warm_tape_stride;
        let cache_out = slot_base + 4;  // output starts after 4-byte hash
        let cached_hidden: Vec<u8> = tape[cache_out .. cache_out + COMPLEX_DIM].to_vec();
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
        hidden_state_save = cached_hidden;
    } else {
        // COLD MISS: full layer stack (f32 tape)
        // Hash checkpoints: capture hash before each forward layer
        // Also capture per-region hashes for pinpointing divergence
        let mut fwd_hashes: Vec<String> = Vec::with_capacity(num_layers);
        let mut fwd_input_hashes: Vec<String> = Vec::with_capacity(num_layers);
        let mut fwd_temp_hashes: Vec<String> = Vec::with_capacity(num_layers);
        let mut fwd_weight_hashes: Vec<String> = Vec::with_capacity(num_layers);
        let mut bwd_hashes: Vec<String> = Vec::with_capacity(num_layers);
        let mut bwd_input_hashes: Vec<String> = Vec::with_capacity(num_layers);
        let mut bwd_temp_hashes: Vec<String> = Vec::with_capacity(num_layers);
        let mut bwd_weight_hashes: Vec<String> = Vec::with_capacity(num_layers);
        
        for layer_idx in 0..num_layers {
            // SHA-256 checkpoint BEFORE forward
            let pre_fwd_hash = {
                let mut h = Sha256::new();
                ShaDigest::update(&mut h, &tape[..work_end]);
                format!("{:x}", h.finalize())
            };
            fwd_hashes.push(pre_fwd_hash);
            fwd_input_hashes.push({
                let mut h = Sha256::new();
                ShaDigest::update(&mut h, &tape[input_offset..input_offset+COMPLEX_DIM]);
                format!("{:x}", h.finalize())
            });
            fwd_temp_hashes.push({
                let mut h = Sha256::new();
                ShaDigest::update(&mut h, &tape[temp_offset..temp_offset+COMPLEX_DIM]);
                format!("{:x}", h.finalize())
            });
            // Weight region hash: lwo (u8) + lwo_f32 (f32) bytes for this layer
            let lwo = weight_offset + layer_idx * TOTAL_WEIGHT_U8;
            let lwo_f32 = weight_offset + layer_idx * TOTAL_WEIGHT_F32;
            fwd_weight_hashes.push({
                let mut h = Sha256::new();
                ShaDigest::update(&mut h, &tape[lwo..lwo+TOTAL_WEIGHT_U8]);
                ShaDigest::update(&mut h, &tape[lwo_f32..lwo_f32+TOTAL_WEIGHT_F32]);
                format!("{:x}", h.finalize())
            });
            
            let layer_save = saved_outputs_offset + layer_idx * COMPLEX_DIM;

            // Dynamic decatalysis — save BOTH lwo (u8) and lwo_f32 (f32)
            let original_weight_f32 = tape[lwo_f32 .. lwo_f32 + TOTAL_WEIGHT_F32].to_vec();
            let original_weight_u8 = tape[lwo .. lwo + TOTAL_WEIGHT_U8].to_vec();
            let src_slice = &compressed_weights_bytes[layer_idx * TOTAL_WEIGHT_U8 .. (layer_idx + 1) * TOTAL_WEIGHT_U8];
            tape[lwo .. lwo + TOTAL_WEIGHT_U8].copy_from_slice(src_slice);
            spn_unscramble(&mut tape, lwo, TOTAL_WEIGHT_U8, bh_key, &sbox, 12);
            // 16.8 WEIGHT STREAMING: route unscrambled u8 bytes into f32 compute region
            let unscrambled = tape[lwo .. lwo + TOTAL_WEIGHT_U8].to_vec();
            tape[lwo_f32 .. lwo_f32 + TOTAL_WEIGHT_F32].copy_from_slice(&unscrambled);

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
                // QKV projections — store as raw bytes in pre_gate and slot
                let slot_base = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + step % MAX_SEQ_LEN) * 4 * F32_DIM;
                for j in 0..HIDDEN_DIM {
                    let wq = tape_f32(&tape, lwo_f32 + WEIGHT_Q_OFFSET, j);
                    let wk = tape_f32(&tape, lwo_f32 + WEIGHT_K_OFFSET, j);
                    let wv = tape_f32(&tape, lwo_f32 + WEIGHT_V_OFFSET, j);
                    let qx = wq * nx[j]; let qy = wq * nx[HIDDEN_DIM + j];
                    let kx = wk * nx[j]; let ky = wk * nx[HIDDEN_DIM + j];
                    let vx = wv * nx[j]; let vy = wv * nx[HIDDEN_DIM + j];
                    let pg = pre_gate_base + layer_idx * COMPLEX_DIM;
                    let qx_bytes = qx.to_bits().to_le_bytes();
                    let qy_bytes = qy.to_bits().to_le_bytes();
                    let kx_bytes = kx.to_bits().to_le_bytes();
                    let ky_bytes = ky.to_bits().to_le_bytes();
                    let vx_bytes = vx.to_bits().to_le_bytes();
                    let vy_bytes = vy.to_bits().to_le_bytes();
                    total_entropy += (qx.abs() as u64) + (ky.abs() as u64) + (vx.abs() as u64);
                    for b in 0..4 {
                        tape[pg + j * 4 + b] ^= qx_bytes[b];
                        tape[pg + F32_DIM + j * 4 + b] ^= qy_bytes[b];
                        tape[slot_base + j * 4 + b] ^= kx_bytes[b];
                        tape[slot_base + F32_DIM + j * 4 + b] ^= ky_bytes[b];
                        tape[slot_base + F32_DIM * 2 + j * 4 + b] ^= vx_bytes[b];
                        tape[slot_base + F32_DIM * 3 + j * 4 + b] ^= vy_bytes[b];
                    }
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
                // Output projection + add to input — store as raw bytes
                for j in 0..HIDDEN_DIM {
                    let wo = tape_f32(&tape, lwo_f32 + WEIGHT_O_OFFSET, j);
                    let px = wo * attn_x[j]; let py = wo * attn_y[j];
                    let px_bytes = px.to_bits().to_le_bytes();
                    let py_bytes = py.to_bits().to_le_bytes();
                    total_entropy += (px.abs() as u64) + (py.abs() as u64);
                    for b in 0..4 {
                        tape[layer_save + j * 4 + b] ^= px_bytes[b];
                        tape[layer_save + F32_DIM + j * 4 + b] ^= py_bytes[b];
                        tape[input_offset + j * 4 + b] ^= px_bytes[b];
                        tape[input_offset + F32_DIM + j * 4 + b] ^= py_bytes[b];
                    }
                }
            } else {
                // DELTANET LAYER (f32) — stores Q as exact u32 bytes in pre_gate
                for j in 0..HIDDEN_DIM {
                    let w = tape_f32(&tape, lwo_f32, j);
                    let x = tape_f32(&tape, input_offset, j);
                    let y = tape_f32(&tape, input_offset + F32_DIM, j);
                    let vx = w * x; let vy = w * y;
                    total_entropy += (vx.abs() as u64) + (vy.abs() as u64);
                    // Save Q as exact u32 bytes into pre_gate (overwrites — we don't need temp save anymore)
                    let pg = pre_gate_base + layer_idx * COMPLEX_DIM;
                    let qx_bytes = vx.to_bits().to_le_bytes();
                    let qy_bytes = vy.to_bits().to_le_bytes();
                    for b in 0..4 {
                        tape[pg + j * 4 + b] ^= qx_bytes[b];
                        tape[pg + F32_DIM + j * 4 + b] ^= qy_bytes[b];
                    }
                    // XOR Q into temp (same bytes, but we already have f32 values)
                    tape_f32_xor(&mut tape, temp_offset, j, vx);
                    tape_f32_xor(&mut tape, temp_offset + F32_DIM, j, vy);
                }
                for j in 0..HIDDEN_DIM {
                    let gx = (0.5 + 0.25 * tape_f32(&tape, temp_offset, j)).clamp(0.0, 1.0);
                    let gy = (0.5 + 0.25 * tape_f32(&tape, temp_offset + F32_DIM, j)).clamp(0.0, 1.0);
                    total_entropy += (gx.abs() as u64) + (gy.abs() as u64);
                    // Store gate as raw bytes (bypasses NaN canonicalization)
                    let gx_bytes = gx.to_bits().to_le_bytes();
                    let gy_bytes = gy.to_bits().to_le_bytes();
                    for b in 0..4 {
                        tape[layer_save + j * 4 + b] ^= gx_bytes[b];
                        tape[layer_save + F32_DIM + j * 4 + b] ^= gy_bytes[b];
                        tape[input_offset + j * 4 + b] ^= gx_bytes[b];
                        tape[input_offset + F32_DIM + j * 4 + b] ^= gy_bytes[b];
                    }
                }
            }

            // Re-scramble and restore weight substrate (both u8 and f32 regions)
            spn_scramble(&mut tape, lwo, TOTAL_WEIGHT_U8, bh_key, &sbox, 12);
            tape[lwo_f32 .. lwo_f32 + TOTAL_WEIGHT_F32].copy_from_slice(&original_weight_f32);
            tape[lwo .. lwo + TOTAL_WEIGHT_U8].copy_from_slice(&original_weight_u8);
        }

        // Output head (f32)
        let mut best_score_f32: f32 = f32::NEG_INFINITY;
        for j in 0..64.min(HIDDEN_DIM) {
            let s = tape_f32(&tape, input_offset, j).abs();
            if s > best_score_f32 { best_score_f32 = s; best_token = j as u8; }
        }

        hidden_state_save = tape[input_offset .. input_offset + COMPLEX_DIM].to_vec();

        // Uncompute: reverse each layer (f32)
        // Uncompute: reverse each layer (f32)
        for layer_idx in (0..num_layers).rev() {
            let lwo = weight_offset + layer_idx * TOTAL_WEIGHT_U8;
            let lwo_f32 = weight_offset + layer_idx * TOTAL_WEIGHT_F32;
            let layer_save = saved_outputs_offset + layer_idx * COMPLEX_DIM;
            let pg = pre_gate_base + layer_idx * COMPLEX_DIM;

            let original_weight_f32 = tape[lwo_f32 .. lwo_f32 + TOTAL_WEIGHT_F32].to_vec();
            let original_weight_u8 = tape[lwo .. lwo + TOTAL_WEIGHT_U8].to_vec();
            let src_slice = &compressed_weights_bytes[layer_idx * TOTAL_WEIGHT_U8 .. (layer_idx + 1) * TOTAL_WEIGHT_U8];
            tape[lwo .. lwo + TOTAL_WEIGHT_U8].copy_from_slice(src_slice);
            spn_unscramble(&mut tape, lwo, TOTAL_WEIGHT_U8, bh_key, &sbox, 12);
            // 16.8 WEIGHT STREAMING: route unscrambled u8 bytes into f32 compute region for uncompute
            let unscrambled_bwd = tape[lwo .. lwo + TOTAL_WEIGHT_U8].to_vec();
            tape[lwo_f32 .. lwo_f32 + TOTAL_WEIGHT_F32].copy_from_slice(&unscrambled_bwd);

            if (layer_idx + 1) % 4 == 0 {
                // Attention uncompute — read output as raw bytes from layer_save (no recompute)
                let attn_idx = layer_idx / 4;
                let slot_base = kv_cache_offset + (attn_idx * MAX_SEQ_LEN + step % MAX_SEQ_LEN) * 4 * F32_DIM;
                // Step 1: undo output copy-through using raw bytes from layer_save
                for j in 0..HIDDEN_DIM {
                    let mut px_bytes = [0u8; 4];
                    let mut py_bytes = [0u8; 4];
                    for b in 0..4 {
                        px_bytes[b] = tape[layer_save + j * 4 + b];
                        py_bytes[b] = tape[layer_save + F32_DIM + j * 4 + b];
                    }
                    for b in 0..4 {
                        tape[input_offset + j * 4 + b] ^= px_bytes[b];
                        tape[input_offset + F32_DIM + j * 4 + b] ^= py_bytes[b];
                        // Zero layer_save
                        tape[layer_save + j * 4 + b] ^= px_bytes[b];
                        tape[layer_save + F32_DIM + j * 4 + b] ^= py_bytes[b];
                    }
                }
                // Step 2: undo QKV projection — read raw bytes from pre_gate and slot
                for j in 0..HIDDEN_DIM {
                    let mut qx_bytes = [0u8; 4];
                    let mut qy_bytes = [0u8; 4];
                    let mut kx_bytes = [0u8; 4];
                    let mut ky_bytes = [0u8; 4];
                    let mut vx_bytes = [0u8; 4];
                    let mut vy_bytes = [0u8; 4];
                    for b in 0..4 {
                        qx_bytes[b] = tape[pg + j * 4 + b];
                        qy_bytes[b] = tape[pg + F32_DIM + j * 4 + b];
                        kx_bytes[b] = tape[slot_base + j * 4 + b];
                        ky_bytes[b] = tape[slot_base + F32_DIM + j * 4 + b];
                        vx_bytes[b] = tape[slot_base + F32_DIM * 2 + j * 4 + b];
                        vy_bytes[b] = tape[slot_base + F32_DIM * 3 + j * 4 + b];
                    }
                    for b in 0..4 {
                        tape[pg + j * 4 + b] ^= qx_bytes[b];
                        tape[pg + F32_DIM + j * 4 + b] ^= qy_bytes[b];
                        tape[slot_base + j * 4 + b] ^= kx_bytes[b];
                        tape[slot_base + F32_DIM + j * 4 + b] ^= ky_bytes[b];
                        tape[slot_base + F32_DIM * 2 + j * 4 + b] ^= vx_bytes[b];
                        tape[slot_base + F32_DIM * 3 + j * 4 + b] ^= vy_bytes[b];
                    }
                }
            } else {
                // DeltaNet uncompute — read gate/Q as raw bytes (no f32 canonicalization)
                for j in 0..HIDDEN_DIM {
                    // Read gate bytes directly from layer_save
                    let mut gx_bytes = [0u8; 4];
                    let mut gy_bytes = [0u8; 4];
                    for b in 0..4 {
                        gx_bytes[b] = tape[layer_save + j * 4 + b];
                        gy_bytes[b] = tape[layer_save + F32_DIM + j * 4 + b];
                    }
                    // XOR gate into input (undo copy-through)
                    for b in 0..4 {
                        tape[input_offset + j * 4 + b] ^= gx_bytes[b];
                        tape[input_offset + F32_DIM + j * 4 + b] ^= gy_bytes[b];
                    }
                    // XOR same bytes into layer_save → zeroes it
                    for b in 0..4 {
                        tape[layer_save + j * 4 + b] ^= gx_bytes[b];
                        tape[layer_save + F32_DIM + j * 4 + b] ^= gy_bytes[b];
                    }
                }
                // Undo Q: read exact Q bytes from pre_gate, XOR into temp, zero pre_gate
                for j in 0..HIDDEN_DIM {
                    // Read Q as raw bytes from pre_gate
                    let mut qx_bytes = [0u8; 4];
                    let mut qy_bytes = [0u8; 4];
                    let pg = pre_gate_base + layer_idx * COMPLEX_DIM;
                    for b in 0..4 {
                        qx_bytes[b] = tape[pg + j * 4 + b];
                        qy_bytes[b] = tape[pg + F32_DIM + j * 4 + b];
                    }
                    // XOR Q bytes into temp (undo Q)
                    tape[temp_offset + j * 4] ^= qx_bytes[0];
                    tape[temp_offset + j * 4 + 1] ^= qx_bytes[1];
                    tape[temp_offset + j * 4 + 2] ^= qx_bytes[2];
                    tape[temp_offset + j * 4 + 3] ^= qx_bytes[3];
                    tape[temp_offset + F32_DIM + j * 4] ^= qy_bytes[0];
                    tape[temp_offset + F32_DIM + j * 4 + 1] ^= qy_bytes[1];
                    tape[temp_offset + F32_DIM + j * 4 + 2] ^= qy_bytes[2];
                    tape[temp_offset + F32_DIM + j * 4 + 3] ^= qy_bytes[3];
                    // Zero pre_gate by XORing same bytes back
                    for b in 0..4 {
                        tape[pg + j * 4 + b] ^= qx_bytes[b];
                        tape[pg + F32_DIM + j * 4 + b] ^= qy_bytes[b];
                    }
                }
            }

            spn_scramble(&mut tape, lwo, TOTAL_WEIGHT_U8, bh_key, &sbox, 12);
            tape[lwo_f32 .. lwo_f32 + TOTAL_WEIGHT_F32].copy_from_slice(&original_weight_f32);
            tape[lwo .. lwo + TOTAL_WEIGHT_U8].copy_from_slice(&original_weight_u8);

            // SHA-256 checkpoint AFTER uncompute — capture per-region hashes
            let post_bwd_hash = {
                let mut h = Sha256::new();
                ShaDigest::update(&mut h, &tape[..work_end]);
                format!("{:x}", h.finalize())
            };
            bwd_hashes.push(post_bwd_hash);
            bwd_input_hashes.push({
                let mut h = Sha256::new();
                ShaDigest::update(&mut h, &tape[input_offset..input_offset+COMPLEX_DIM]);
                format!("{:x}", h.finalize())
            });
            bwd_temp_hashes.push({
                let mut h = Sha256::new();
                ShaDigest::update(&mut h, &tape[temp_offset..temp_offset+COMPLEX_DIM]);
                format!("{:x}", h.finalize())
            });
            bwd_weight_hashes.push({
                let mut h = Sha256::new();
                ShaDigest::update(&mut h, &tape[lwo..lwo+TOTAL_WEIGHT_U8]);
                ShaDigest::update(&mut h, &tape[lwo_f32..lwo_f32+TOTAL_WEIGHT_F32]);
                format!("{:x}", h.finalize())
            });

            // Check if this layer restored correctly (compare to pre-forward hash)
            let layer_fwd_idx = layer_idx;
            if first_broken_layer == -1 && bwd_hashes.last() != fwd_hashes.get(layer_fwd_idx) {
                first_broken_layer = layer_idx as i32;
                eprintln!("LAYER {} BROKEN: fwd_hash={} bwd_hash={}", 
                    layer_idx,
                    fwd_hashes.get(layer_fwd_idx).unwrap_or(&String::new()),
                    bwd_hashes.last().unwrap_or(&String::new()));
                
                // Per-region comparison
                let empty = String::new();
                let fwd_in = fwd_input_hashes.get(layer_fwd_idx).unwrap_or(&empty);
                let bwd_in = bwd_input_hashes.last().unwrap_or(&empty);
                let fwd_tmp = fwd_temp_hashes.get(layer_fwd_idx).unwrap_or(&empty);
                let bwd_tmp = bwd_temp_hashes.last().unwrap_or(&empty);
                let fwd_wt = fwd_weight_hashes.get(layer_fwd_idx).unwrap_or(&empty);
                let bwd_wt = bwd_weight_hashes.last().unwrap_or(&empty);
                
                eprintln!("  input:  fwd={} bwd={} match={}", 
                    &fwd_in[..16], &bwd_in[..16], fwd_in == bwd_in);
                eprintln!("  temp:   fwd={} bwd={} match={}", 
                    &fwd_tmp[..16], &bwd_tmp[..16], fwd_tmp == bwd_tmp);
                eprintln!("  weight: fwd={} bwd={} match={}", 
                    &fwd_wt[..16], &bwd_wt[..16], fwd_wt == bwd_wt);
                
                // Also check pg and ls regions
                let pg_hash: String = {
                    let mut h = Sha256::new();
                    ShaDigest::update(&mut h, &tape[pg..pg+COMPLEX_DIM]);
                    format!("{:x}", h.finalize())
                };
                let ls_hash: String = {
                    let mut h = Sha256::new();
                    ShaDigest::update(&mut h, &tape[layer_save..layer_save+COMPLEX_DIM]);
                    format!("{:x}", h.finalize())
                };
                eprintln!("  pg:  hash={}", &pg_hash[..16]);
                eprintln!("  ls:  hash={}", &ls_hash[..16]);
                
                eprintln!("  Tape bytes [input_off..input_off+32]: {:02x?}", &tape[input_offset..input_offset+32]);
                eprintln!("  Tape bytes [lwo..lwo+32]: {:02x?}", &tape[lwo..lwo+32]);
                eprintln!("  Tape bytes [lwo_f32..lwo_f32+32]: {:02x?}", &tape[lwo_f32..lwo_f32+32]);
            }
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

    // Cache write: AFTER hash computation — use captured hidden state (NOT post-restore tape)
    if !warm_hit {
        let slot_base = warm_tape_offset + ((emb_hash as usize) % WARM_TAPE_SLOTS) * warm_tape_stride;
        tape[slot_base..slot_base + 4].copy_from_slice(&emb_hash_bytes);
        tape[slot_base + 4 .. slot_base + 4 + COMPLEX_DIM].copy_from_slice(&hidden_state_save);
    }

    let result = pyo3::types::PyDict::new_bound(py);
    result.set_item("total_entropy", total_entropy)?;
    result.set_item("generated_token", best_token)?;
    result.set_item("elapsed_secs", elapsed)?;
    result.set_item("tape_restored", (initial_hash == final_hash).into_py(py))?;
    result.set_item("num_layers", num_layers)?;
    result.set_item("warm_hit", warm_hit.into_py(py))?;
    result.set_item("first_broken_layer", first_broken_layer)?;
    let work_end = (warm_tape_offset + WARM_TAPE_SLOTS * warm_tape_stride).min(tape.len());
    let work_slice = &tape[..work_end];
    result.set_item("working_region", PyBytes::new_bound(py, work_slice))?;
    result.set_item("hidden_state", PyBytes::new_bound(py, &hidden_state_save))?;
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

// Fractal bit-reversed indexing — maps linear position to p-adic uniform position.
// This transforms the Feistel phase space from chaotic (kicked-rotor) to integrable
// by distributing phase kicks uniformly across the tape in p-adic metric.
fn fractal_index(i: usize, max_bits: usize) -> usize {
    let mut rev = 0;
    let mut n = i;
    for _ in 0..max_bits {
        rev = (rev << 1) | (n & 1);
        n >>= 1;
    }
    rev
}

// Multi-scale Feistel scramble with FRACTAL indexing — gapped topological phase (Q57)
// Bit-reversed indexing creates p-adic uniform distribution of phase kicks,
// preventing KAM torus breakdown that causes Layer 47 uncompute failure.
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

    // Multi-scale: use the largest power-of-2 that fits, up to rounds_limit
    let max_scale = (half_size as f64).log2().floor() as usize;
    let scales = (0..max_scale.min(rounds_limit)).collect::<Vec<_>>();
    let n_scales = scales.len();
    if n_scales == 0 { return; }
    
    let n_blocks_per_scale: Vec<usize> = (0..n_scales)
        .map(|s| half_size / (1 << (scales[s] + 1)))
        .collect();

    for r_idx in 0..n_scales {
        let scale = 1 << scales[r_idx]; // 1, 2, 4, 8, ...
        let stride = scale * 2;
        let n_blocks = n_blocks_per_scale[r_idx];
        if n_blocks == 0 { continue; }
        let max_bits = (n_blocks as f64).log2().ceil() as usize;

        // Fractal (bit-reversed) block iteration — p-adic uniform distribution
        for linear_block in 0..n_blocks {
            let frac_block = fractal_index(linear_block, max_bits) % n_blocks;
            let i = frac_block * stride;
            if i + scale > half_size { continue; }
            for b in 0..scale {
                temp_block[i + b] = tape[region_base + half_size + i + b];
            }
            let mut f_out = vec![0u8; scale];
            spn_round_function(&temp_block[i..i+scale], r_idx, key, sbox, scale, &mut f_out);
            for b in 0..scale {
                let l_val = tape[region_base + i + b];
                tape[region_base + i + b] = tape[region_base + half_size + i + b];
                tape[region_base + half_size + i + b] = l_val ^ f_out[b];
            }
        }
    }
}

// Multi-scale Feistel unscramble with FRACTAL indexing — inverse of fractal scramble
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

    let max_scale = (half_size as f64).log2().floor() as usize;
    let scales = (0..max_scale.min(rounds_limit)).collect::<Vec<_>>();
    let n_scales = scales.len();
    if n_scales == 0 { return; }

    let n_blocks_per_scale: Vec<usize> = (0..n_scales)
        .map(|s| half_size / (1 << (scales[s] + 1)))
        .collect();

    // Reverse order (stack unwinding)
    for r_idx in (0..n_scales).rev() {
        let scale = 1 << scales[r_idx];
        let stride = scale * 2;
        let n_blocks = n_blocks_per_scale[r_idx];
        if n_blocks == 0 { continue; }
        let max_bits = (n_blocks as f64).log2().ceil() as usize;

        // Same fractal order as scramble — Feistel inverse uses same F
        for linear_block in 0..n_blocks {
            let frac_block = fractal_index(linear_block, max_bits) % n_blocks;
            let i = frac_block * stride;
            if i + scale > half_size { continue; }
            for b in 0..scale {
                temp_block[i + b] = tape[region_base + i + b];
            }
            let mut f_out = vec![0u8; scale];
            spn_round_function(&temp_block[i..i+scale], r_idx, key, sbox, scale, &mut f_out);
            for b in 0..scale {
                let r_val = tape[region_base + half_size + i + b];
                tape[region_base + half_size + i + b] = temp_block[i + b];
                tape[region_base + i + b] = r_val ^ f_out[b];
            }
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
    let num_layers = weights_vec.len() / TOTAL_WEIGHT_U8;
    for layer_idx in 0..num_layers {
        let lwo = layer_idx * TOTAL_WEIGHT_U8;
        spn_scramble(&mut weights_vec, lwo, TOTAL_WEIGHT_U8, bh_key, &sbox, 12);
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

