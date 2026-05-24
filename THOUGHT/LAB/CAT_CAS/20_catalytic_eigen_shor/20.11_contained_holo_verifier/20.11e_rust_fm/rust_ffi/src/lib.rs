use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1};
use rayon::prelude::*;
use std::f64::consts::PI;

fn mod_exp(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 { return 0; }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    result
}

/// Build catalytic phase grating as complex128 numpy array.
#[pyfunction]
fn build_catalytic_grating<'py>(
    py: Python<'py>,
    a: u64,
    n: u64,
    m: usize,
) -> PyResult<&'py PyArray1<num_complex::Complex64>> {
    let n_f64 = n as f64;
    let two_pi = 2.0 * PI;
    let chunk_size: usize = 1_000_000;
    let num_chunks = (m + chunk_size - 1) / chunk_size;

    let mut data: Vec<num_complex::Complex64> = vec![num_complex::Complex64::new(0.0, 0.0); m];

    let chunks: Vec<(usize, usize)> = (0..num_chunks)
        .map(|i| (i * chunk_size, std::cmp::min((i + 1) * chunk_size, m)))
        .collect();

    data.par_chunks_mut(chunk_size)
        .zip(chunks.par_iter())
        .for_each(|(slice, &(start, end))| {
            let len = end - start;
            if len == 0 { return; }
            let mut val = if start == 0 { 1u64 } else { mod_exp(a, start as u64, n) };
            for i in 0..len {
                let angle = two_pi * (val as f64) / n_f64;
                slice[i] = num_complex::Complex64::new(angle.cos(), angle.sin());
                val = ((val as u128 * a as u128) % n as u128) as u64;
            }
        });

    Ok(data.into_pyarray(py))
}

/// Build modular exponentiation sequence as u64 numpy array.
#[pyfunction]
fn build_mod_exp_sequence<'py>(
    py: Python<'py>,
    a: u64,
    n: u64,
    m: usize,
) -> PyResult<&'py PyArray1<u64>> {
    let chunk_size: usize = 1_000_000;
    let num_chunks = (m + chunk_size - 1) / chunk_size;

    let mut data: Vec<u64> = vec![0u64; m];

    let chunks: Vec<(usize, usize)> = (0..num_chunks)
        .map(|i| (i * chunk_size, std::cmp::min((i + 1) * chunk_size, m)))
        .collect();

    data.par_chunks_mut(chunk_size)
        .zip(chunks.par_iter())
        .for_each(|(slice, &(start, end))| {
            let len = end - start;
            if len == 0 { return; }
            let mut val = if start == 0 { 1u64 } else { mod_exp(a, start as u64, n) };
            for i in 0..len {
                slice[i] = val;
                val = ((val as u128 * a as u128) % n as u128) as u64;
            }
        });

    Ok(data.into_pyarray(py))
}

#[pymodule]
fn catalytic_grating_ffi(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_catalytic_grating, m)?)?;
    m.add_function(wrap_pyfunction!(build_mod_exp_sequence, m)?)?;
    Ok(())
}
